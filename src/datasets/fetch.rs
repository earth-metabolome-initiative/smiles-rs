use alloc::vec::Vec;
use std::{
    env,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use dirs::cache_dir;
use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use tar::Archive;

use super::{
    progress::{ProgressReader, new_byte_progress_bar, progress_label},
    source::{DatasetCollectionSource, DatasetSource},
    types::{
        ArchiveMode, CacheMode, DatasetArtifact, DatasetCollectionArtifact, DatasetCompression,
        DatasetError, DatasetFetchOptions, DatasetFile,
    },
};

const DOWNLOAD_USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

/// Returns the default cache directory used by dataset fetches.
///
/// The directory is `smiles-rs/datasets` under the platform cache directory
/// reported by [`dirs::cache_dir`], which is `$XDG_CACHE_HOME` or
/// `$HOME/.cache` on Linux, `$HOME/Library/Caches` on macOS and
/// `%LOCALAPPDATA%` on Windows. When no cache directory can be determined, the
/// temporary directory is used instead.
///
/// # Examples
///
/// ```
/// use smiles_rs::datasets::default_dataset_cache_dir;
///
/// assert!(default_dataset_cache_dir().ends_with("smiles-rs/datasets"));
/// ```
#[must_use]
pub fn default_dataset_cache_dir() -> PathBuf {
    cache_dir().unwrap_or_else(env::temp_dir).join("smiles-rs").join("datasets")
}

pub(crate) fn fetch_dataset<D: DatasetSource + ?Sized>(
    dataset: &D,
    options: &DatasetFetchOptions,
) -> Result<DatasetArtifact, DatasetError> {
    let cache_root = match &options.cache_dir {
        Some(path) => path.clone(),
        None => default_dataset_cache_dir(),
    };

    let dataset_dir = cache_root.join(dataset.id());
    create_dir_all(&dataset_dir)?;

    let compressed_path = dataset_dir.join(dataset.file_name());
    let decompressed_path = dataset_dir.join(dataset.extracted_file_name());

    let (path, has_decompressed_path, was_downloaded, was_decompressed) =
        match (dataset.compression(), options.archive_mode) {
            (DatasetCompression::None, _)
            | (
                DatasetCompression::Gzip | DatasetCompression::TarGzip | DatasetCompression::Zip,
                ArchiveMode::KeepCompressed,
            ) => {
                let was_downloaded =
                    ensure_downloaded(dataset, &compressed_path, options.cache_mode)?;

                (compressed_path.clone(), false, was_downloaded, false)
            }

            (DatasetCompression::Gzip, ArchiveMode::Decompress | ArchiveMode::KeepBoth) => {
                let (was_downloaded, was_decompressed) = ensure_decompressed(
                    dataset,
                    &compressed_path,
                    &decompressed_path,
                    options.cache_mode,
                )?;

                (decompressed_path.clone(), true, was_downloaded, was_decompressed)
            }

            (DatasetCompression::TarGzip, ArchiveMode::Decompress | ArchiveMode::KeepBoth) => {
                let (was_downloaded, was_extracted) = ensure_extracted_tar_gzip(
                    dataset.url(),
                    &compressed_path,
                    &decompressed_path,
                    options.cache_mode,
                )?;

                (decompressed_path.clone(), true, was_downloaded, was_extracted)
            }

            (DatasetCompression::Zip, ArchiveMode::Decompress | ArchiveMode::KeepBoth) => {
                let (was_downloaded, was_extracted) = ensure_extracted_zip(
                    dataset.url(),
                    &compressed_path,
                    &decompressed_path,
                    options.cache_mode,
                )?;

                (decompressed_path.clone(), true, was_downloaded, was_extracted)
            }
        };

    Ok(DatasetArtifact {
        dataset_id: dataset.id(),
        path,
        compressed_path: compressed_path.is_file().then_some(compressed_path),
        decompressed_path: has_decompressed_path.then_some(decompressed_path),
        was_downloaded,
        was_decompressed,
    })
}

pub(crate) fn fetch_dataset_collection<D: DatasetCollectionSource + ?Sized>(
    dataset: &D,
    options: &DatasetFetchOptions,
) -> Result<DatasetCollectionArtifact, DatasetError> {
    let cache_root = match &options.cache_dir {
        Some(path) => path.clone(),
        None => default_dataset_cache_dir(),
    };
    let dataset_dir = cache_root.join(dataset.id());
    create_dir_all(&dataset_dir)?;

    let mut paths = Vec::new();
    let mut compressed_paths = Vec::new();
    let mut was_downloaded = false;
    let mut was_extracted = false;

    for file in dataset.files() {
        let compressed_path = dataset_dir.join(file.file_name());
        let extracted_path = dataset_dir.join(file.extracted_file_name());
        match file.compression() {
            DatasetCompression::None => {
                was_downloaded |=
                    ensure_downloaded_url(file.url(), &compressed_path, options.cache_mode)?;
                paths.push(compressed_path.clone());
                compressed_paths.push(compressed_path);
            }
            DatasetCompression::Gzip => {
                match options.archive_mode {
                    ArchiveMode::KeepCompressed => {
                        was_downloaded |= ensure_downloaded_url(
                            file.url(),
                            &compressed_path,
                            options.cache_mode,
                        )?;
                        paths.push(compressed_path.clone());
                        compressed_paths.push(compressed_path);
                    }
                    ArchiveMode::Decompress | ArchiveMode::KeepBoth => {
                        let (downloaded, decompressed) = ensure_decompressed_url(
                            file.url(),
                            &compressed_path,
                            &extracted_path,
                            options.cache_mode,
                        )?;
                        was_downloaded |= downloaded;
                        was_extracted |= decompressed;
                        paths.push(extracted_path);
                        if compressed_path.is_file() {
                            compressed_paths.push(compressed_path);
                        }
                    }
                }
            }
            DatasetCompression::TarGzip => {
                match options.archive_mode {
                    ArchiveMode::KeepCompressed => {
                        was_downloaded |= ensure_downloaded_url(
                            file.url(),
                            &compressed_path,
                            options.cache_mode,
                        )?;
                        paths.push(compressed_path.clone());
                        compressed_paths.push(compressed_path);
                    }
                    ArchiveMode::Decompress | ArchiveMode::KeepBoth => {
                        let (downloaded, extracted) = ensure_extracted_tar_gzip(
                            file.url(),
                            &compressed_path,
                            &extracted_path,
                            options.cache_mode,
                        )?;
                        was_downloaded |= downloaded;
                        was_extracted |= extracted;
                        paths.push(extracted_path);
                        if compressed_path.is_file() {
                            compressed_paths.push(compressed_path);
                        }
                    }
                }
            }
            DatasetCompression::Zip => {
                let (downloaded, extracted) = fetch_zip_dataset_collection(
                    options,
                    compressed_path,
                    file,
                    &mut paths,
                    &mut compressed_paths,
                    extracted_path,
                )?;
                was_downloaded |= downloaded;
                was_extracted |= extracted;
            }
        }
    }

    Ok(DatasetCollectionArtifact {
        dataset_id: dataset.id(),
        paths,
        compressed_paths,
        was_downloaded,
        was_extracted,
    })
}

fn ensure_downloaded<D: DatasetSource + ?Sized>(
    dataset: &D,
    target_path: &Path,
    cache_mode: CacheMode,
) -> Result<bool, DatasetError> {
    ensure_downloaded_url(dataset.url(), target_path, cache_mode)
}

fn ensure_downloaded_url(
    url: &'static str,
    target_path: &Path,
    cache_mode: CacheMode,
) -> Result<bool, DatasetError> {
    if cache_mode == CacheMode::UseCache && target_path.is_file() {
        return Ok(false);
    }

    download_to_path(url, target_path)?;
    Ok(true)
}

fn ensure_decompressed<D: DatasetSource + ?Sized>(
    dataset: &D,
    compressed_path: &Path,
    decompressed_path: &Path,
    cache_mode: CacheMode,
) -> Result<(bool, bool), DatasetError> {
    ensure_decompressed_url(dataset.url(), compressed_path, decompressed_path, cache_mode)
}

fn ensure_decompressed_url(
    url: &'static str,
    compressed_path: &Path,
    decompressed_path: &Path,
    cache_mode: CacheMode,
) -> Result<(bool, bool), DatasetError> {
    if cache_mode == CacheMode::UseCache && decompressed_path.is_file() {
        return Ok((false, false));
    }

    let was_downloaded = ensure_downloaded_url(url, compressed_path, cache_mode)?;
    let was_decompressed = gunzip_file(compressed_path, decompressed_path)?;
    Ok((was_downloaded, was_decompressed))
}

fn download_to_path(url: &'static str, target_path: &Path) -> Result<(), DatasetError> {
    let client = Client::builder()
        .user_agent(DOWNLOAD_USER_AGENT)
        .connect_timeout(Duration::from_secs(30))
        .build()
        .map_err(|source| DatasetError::Download { url, source })?;
    let response = client
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|source| DatasetError::Download { url, source })?;
    let file_label = progress_label("downloading", target_path);
    let progress_bar = new_byte_progress_bar(response.content_length(), &file_label);
    let mut response = ProgressReader::new(response, progress_bar.clone());

    write_parent_dir(target_path)?;
    let partial = PartialPath::new(target_path);
    let file = File::create(partial.path())
        .map_err(|source| DatasetError::Io { path: partial.path().to_path_buf(), source })?;
    let mut writer = BufWriter::new(file);
    if let Err(source) = io::copy(&mut response, &mut writer) {
        progress_bar.abandon();
        return Err(DatasetError::Io { path: target_path.to_path_buf(), source });
    }
    writer
        .flush()
        .map_err(|source| DatasetError::Io { path: target_path.to_path_buf(), source })?;
    progress_bar.finish_and_clear();

    fs::rename(partial.path(), target_path)
        .map_err(|source| DatasetError::Io { path: target_path.to_path_buf(), source })?;
    Ok(())
}

pub(crate) fn gunzip_file(
    compressed_path: &Path,
    decompressed_path: &Path,
) -> Result<bool, DatasetError> {
    write_parent_dir(decompressed_path)?;
    let source_file = File::open(compressed_path)
        .map_err(|source| DatasetError::Io { path: compressed_path.to_path_buf(), source })?;
    let progress_bar = new_byte_progress_bar(
        source_file.metadata().ok().map(|metadata| metadata.len()),
        &progress_label("decompressing", decompressed_path),
    );
    let source_file = ProgressReader::new(source_file, progress_bar.clone());
    let mut decoder = GzDecoder::new(source_file);
    let partial = PartialPath::new(decompressed_path);
    let target_file = File::create(partial.path())
        .map_err(|source| DatasetError::Io { path: partial.path().to_path_buf(), source })?;
    let mut writer = BufWriter::new(target_file);
    if let Err(source) = io::copy(&mut decoder, &mut writer) {
        progress_bar.abandon();
        return Err(DatasetError::Io { path: decompressed_path.to_path_buf(), source });
    }
    writer
        .flush()
        .map_err(|source| DatasetError::Io { path: decompressed_path.to_path_buf(), source })?;
    progress_bar.finish_and_clear();

    remove_path_if_exists(decompressed_path)?;
    fs::rename(partial.path(), decompressed_path)
        .map_err(|source| DatasetError::Io { path: decompressed_path.to_path_buf(), source })?;
    Ok(true)
}

fn ensure_extracted_tar_gzip(
    url: &'static str,
    compressed_path: &Path,
    extracted_path: &Path,
    cache_mode: CacheMode,
) -> Result<(bool, bool), DatasetError> {
    if cache_mode == CacheMode::UseCache && extracted_path.is_dir() {
        return Ok((false, false));
    }

    let was_downloaded = ensure_downloaded_url(url, compressed_path, cache_mode)?;
    let was_extracted = untar_gzip_file(compressed_path, extracted_path)?;
    Ok((was_downloaded, was_extracted))
}

fn ensure_extracted_zip(
    url: &'static str,
    compressed_path: &Path,
    extracted_path: &Path,
    cache_mode: CacheMode,
) -> Result<(bool, bool), DatasetError> {
    if cache_mode == CacheMode::UseCache && extracted_path.is_file() {
        return Ok((false, false));
    }
    let was_downloaded = ensure_downloaded_url(url, compressed_path, cache_mode)?;
    let was_extracted = unzip_file(compressed_path, extracted_path)?;
    Ok((was_downloaded, was_extracted))
}

pub(crate) fn unzip_file(
    compressed_path: &Path,
    extracted_path: &Path,
) -> Result<bool, DatasetError> {
    write_parent_dir(extracted_path)?;
    let source_file = File::open(compressed_path)
        .map_err(|source| DatasetError::Io { path: compressed_path.to_path_buf(), source })?;
    let mut archive = zip::ZipArchive::new(source_file).map_err(|source| {
        DatasetError::Io {
            path: compressed_path.to_path_buf(),
            source: io::Error::new(io::ErrorKind::InvalidData, source),
        }
    })?;
    let expected_name = extracted_path
        .file_name()
        .unwrap_or_else(|| unreachable!("extracted path will have a file name"));
    let mut entry_index = None;
    for index in 0..archive.len() {
        let entry = archive.by_index(index).map_err(|source| {
            DatasetError::Io {
                path: compressed_path.to_path_buf(),
                source: io::Error::new(io::ErrorKind::InvalidData, source),
            }
        })?;
        if entry.is_file() && Path::new(entry.name()).file_name() == Some(expected_name) {
            entry_index = Some(index);
            break;
        }
    }
    let index = entry_index.ok_or_else(|| {
        DatasetError::Io {
            path: extracted_path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::NotFound,
                "zip archive did not contain expected file",
            ),
        }
    })?;
    let partial = PartialPath::new(extracted_path);
    {
        let mut entry = archive.by_index(index).map_err(|source| {
            DatasetError::Io {
                path: compressed_path.to_path_buf(),
                source: io::Error::new(io::ErrorKind::InvalidData, source),
            }
        })?;

        let progress_bar = new_byte_progress_bar(
            Some(entry.size()),
            &progress_label("extracting", extracted_path),
        );
        let mut entry = ProgressReader::new(&mut entry, progress_bar.clone());

        let target_file = File::create(partial.path())
            .map_err(|source| DatasetError::Io { path: partial.path().to_path_buf(), source })?;

        let mut writer = BufWriter::new(target_file);

        if let Err(source) = io::copy(&mut entry, &mut writer) {
            progress_bar.abandon();
            return Err(DatasetError::Io { path: extracted_path.to_path_buf(), source });
        }

        writer
            .flush()
            .map_err(|source| DatasetError::Io { path: extracted_path.to_path_buf(), source })?;
        progress_bar.finish_and_clear();
    }

    remove_path_if_exists(extracted_path)?;

    fs::rename(partial.path(), extracted_path)
        .map_err(|source| DatasetError::Io { path: extracted_path.to_path_buf(), source })?;

    Ok(true)
}

pub(crate) fn untar_gzip_file(
    compressed_path: &Path,
    extracted_path: &Path,
) -> Result<bool, DatasetError> {
    write_parent_dir(extracted_path)?;
    let source_file = File::open(compressed_path)
        .map_err(|source| DatasetError::Io { path: compressed_path.to_path_buf(), source })?;
    let progress_bar = new_byte_progress_bar(
        source_file.metadata().ok().map(|metadata| metadata.len()),
        &progress_label("extracting", extracted_path),
    );
    let source_file = ProgressReader::new(source_file, progress_bar.clone());
    let decoder = GzDecoder::new(source_file);
    let mut archive = Archive::new(decoder);

    let partial = PartialPath::new(extracted_path);
    remove_path_if_exists(partial.path())?;
    create_dir_all(partial.path())?;
    if let Err(source) = archive.unpack(partial.path()) {
        progress_bar.abandon();
        return Err(DatasetError::Io { path: extracted_path.to_path_buf(), source });
    }
    progress_bar.finish_and_clear();

    let extracted_name = extracted_path
        .file_name()
        .unwrap_or_else(|| unreachable!("extracted path has a file name"));
    let unpacked_path = partial.path().join(extracted_name);
    if !unpacked_path.exists() {
        return Err(DatasetError::Io {
            path: extracted_path.to_path_buf(),
            source: io::Error::new(
                io::ErrorKind::NotFound,
                "tar archive did not contain the expected top-level directory",
            ),
        });
    }

    remove_path_if_exists(extracted_path)?;
    fs::rename(&unpacked_path, extracted_path)
        .map_err(|source| DatasetError::Io { path: extracted_path.to_path_buf(), source })?;
    Ok(true)
}

fn remove_path_if_exists(path: &Path) -> Result<(), DatasetError> {
    if !path.exists() {
        return Ok(());
    }
    let result = if path.is_dir() { fs::remove_dir_all(path) } else { fs::remove_file(path) };
    result.map_err(|source| DatasetError::Io { path: path.to_path_buf(), source })
}

fn temporary_download_path(target_path: &Path) -> PathBuf {
    let file_name = target_path
        .file_name()
        .map_or_else(|| "download".into(), |name| name.to_string_lossy().into_owned());
    target_path.with_file_name(format!("{file_name}.part"))
}

/// Path a download or extraction writes to before it is renamed into place.
///
/// Dropping the guard deletes whatever is still there, so a failed transfer
/// leaves no partial file or directory behind.
struct PartialPath(PathBuf);

impl PartialPath {
    fn new(target_path: &Path) -> Self {
        Self(temporary_download_path(target_path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for PartialPath {
    fn drop(&mut self) {
        let _ = remove_path_if_exists(&self.0);
    }
}

fn write_parent_dir(path: &Path) -> Result<(), DatasetError> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    Ok(())
}

fn create_dir_all(path: &Path) -> Result<(), DatasetError> {
    fs::create_dir_all(path).map_err(|source| DatasetError::Io { path: path.to_path_buf(), source })
}

fn fetch_zip_dataset_collection(
    options: &DatasetFetchOptions,
    compressed_path: PathBuf,
    file: DatasetFile,
    paths: &mut Vec<PathBuf>,
    compressed_paths: &mut Vec<PathBuf>,
    extracted_path: PathBuf,
) -> Result<(bool, bool), DatasetError> {
    match options.archive_mode {
        ArchiveMode::KeepCompressed => {
            let downloaded =
                ensure_downloaded_url(file.url(), &compressed_path, options.cache_mode)?;
            paths.push(compressed_path.clone());
            compressed_paths.push(compressed_path);

            Ok((downloaded, false))
        }
        ArchiveMode::Decompress | ArchiveMode::KeepBoth => {
            let (downloaded, extracted) = ensure_extracted_zip(
                file.url(),
                &compressed_path,
                &extracted_path,
                options.cache_mode,
            )?;
            paths.push(extracted_path);
            if compressed_path.is_file() {
                compressed_paths.push(compressed_path);
            }
            Ok((downloaded, extracted))
        }
    }
}
