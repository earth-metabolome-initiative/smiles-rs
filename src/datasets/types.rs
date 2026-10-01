use alloc::{string::String, vec::Vec};
use std::{
    io,
    path::{Path, PathBuf},
};

use thiserror::Error;

/// Compression used by the upstream dataset artifact.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DatasetCompression {
    /// The upstream file is not compressed.
    None,
    /// The upstream file is gzip-compressed.
    Gzip,
    /// The upstream file is a gzip-compressed tar archive.
    TarGzip,
    /// The upstream file is a zip-compressed archive
    Zip,
}

/// Cache behavior for dataset fetches.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum CacheMode {
    /// Reuse an existing cached artifact when present.
    #[default]
    UseCache,
    /// Always redownload the upstream artifact.
    Redownload,
}

/// How compressed datasets should be materialized locally.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ArchiveMode {
    /// Keep the cached archive file as-is.
    #[default]
    KeepCompressed,
    /// Materialize and return a decompressed copy.
    Decompress,
    /// Keep the archive cache and also materialize a decompressed copy.
    KeepBoth,
}

/// Options controlling dataset fetch and cache behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetFetchOptions {
    /// Override the cache directory. When `None`, a per-user cache directory is
    /// selected automatically.
    pub cache_dir: Option<PathBuf>,
    /// Whether to reuse existing cached files or redownload them.
    pub cache_mode: CacheMode,
    /// How compressed datasets should be stored locally.
    pub archive_mode: ArchiveMode,
}

impl Default for DatasetFetchOptions {
    fn default() -> Self {
        Self {
            cache_dir: None,
            cache_mode: CacheMode::UseCache,
            archive_mode: ArchiveMode::KeepCompressed,
        }
    }
}

/// A materialized dataset artifact on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetArtifact {
    pub(crate) dataset_id: &'static str,
    pub(crate) path: PathBuf,
    pub(crate) compressed_path: Option<PathBuf>,
    pub(crate) decompressed_path: Option<PathBuf>,
    pub(crate) was_downloaded: bool,
    pub(crate) was_decompressed: bool,
}

impl DatasetArtifact {
    /// Returns the dataset identifier that produced this artifact.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{DatasetSource, MASS_SPEC_GYM_SMILES};
    ///
    /// let artifact = MASS_SPEC_GYM_SMILES.fetch()?;
    /// assert_eq!(artifact.dataset_id(), "massspecgym-smiles");
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    #[must_use]
    pub fn dataset_id(&self) -> &'static str {
        self.dataset_id
    }

    /// Returns the primary path callers should consume.
    ///
    /// This is the decompressed path when one was requested, otherwise the
    /// cached upstream file path.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{DatasetSource, MASS_SPEC_GYM_SMILES};
    ///
    /// let artifact = MASS_SPEC_GYM_SMILES.fetch()?;
    /// println!("{}", artifact.path().display());
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the cached compressed path, when present.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{ArchiveMode, DatasetFetchOptions, DatasetSource, PUBCHEM_SMILES};
    ///
    /// let artifact = PUBCHEM_SMILES.fetch_with_options(&DatasetFetchOptions {
    ///     archive_mode: ArchiveMode::KeepBoth,
    ///     ..DatasetFetchOptions::default()
    /// })?;
    /// assert!(artifact.compressed_path().is_some());
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    #[must_use]
    pub fn compressed_path(&self) -> Option<&Path> {
        self.compressed_path.as_deref()
    }

    /// Returns the cached decompressed path, when present.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{ArchiveMode, DatasetFetchOptions, DatasetSource, PUBCHEM_SMILES};
    ///
    /// let artifact = PUBCHEM_SMILES.fetch_with_options(&DatasetFetchOptions {
    ///     archive_mode: ArchiveMode::Decompress,
    ///     ..DatasetFetchOptions::default()
    /// })?;
    /// assert!(artifact.decompressed_path().is_some());
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    #[must_use]
    pub fn decompressed_path(&self) -> Option<&Path> {
        self.decompressed_path.as_deref()
    }

    /// Returns whether the upstream artifact was downloaded during this call.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{
    ///     CacheMode, DatasetFetchOptions, DatasetSource, MASS_SPEC_GYM_SMILES,
    /// };
    ///
    /// let artifact = MASS_SPEC_GYM_SMILES.fetch_with_options(&DatasetFetchOptions {
    ///     cache_mode: CacheMode::UseCache,
    ///     ..DatasetFetchOptions::default()
    /// })?;
    /// let _downloaded = artifact.was_downloaded();
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    #[must_use]
    pub fn was_downloaded(&self) -> bool {
        self.was_downloaded
    }

    /// Returns whether a decompressed copy was created during this call.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{ArchiveMode, DatasetFetchOptions, DatasetSource, PUBCHEM_SMILES};
    ///
    /// let artifact = PUBCHEM_SMILES.fetch_with_options(&DatasetFetchOptions {
    ///     archive_mode: ArchiveMode::Decompress,
    ///     ..DatasetFetchOptions::default()
    /// })?;
    /// let _decompressed = artifact.was_decompressed();
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    #[must_use]
    pub fn was_decompressed(&self) -> bool {
        self.was_decompressed
    }
}

/// A materialized multi-file dataset collection on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetCollectionArtifact {
    pub(crate) dataset_id: &'static str,
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) compressed_paths: Vec<PathBuf>,
    pub(crate) was_downloaded: bool,
    pub(crate) was_extracted: bool,
}

impl DatasetCollectionArtifact {
    /// Returns the dataset identifier that produced this artifact collection.
    #[must_use]
    pub fn dataset_id(&self) -> &'static str {
        self.dataset_id
    }

    /// Returns the primary paths callers should consume.
    ///
    /// For archive-based datasets these are extracted directories when
    /// extraction was requested, otherwise the cached archive paths.
    #[must_use]
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Returns the cached compressed archive paths.
    #[must_use]
    pub fn compressed_paths(&self) -> &[PathBuf] {
        &self.compressed_paths
    }

    /// Returns whether any upstream artifact was downloaded during this call.
    #[must_use]
    pub fn was_downloaded(&self) -> bool {
        self.was_downloaded
    }

    /// Returns whether any upstream archive was extracted during this call.
    #[must_use]
    pub fn was_extracted(&self) -> bool {
        self.was_extracted
    }
}

/// Metadata for one file within a downloadable dataset collection.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DatasetFile {
    url: &'static str,
    file_name: &'static str,
    extracted_file_name: &'static str,
    compression: DatasetCompression,
}

impl DatasetFile {
    /// Creates static metadata for one downloadable dataset file.
    #[must_use]
    pub const fn new(
        url: &'static str,
        file_name: &'static str,
        extracted_file_name: &'static str,
        compression: DatasetCompression,
    ) -> Self {
        Self { url, file_name, extracted_file_name, compression }
    }

    /// Returns the upstream URL for this dataset file.
    #[must_use]
    pub fn url(&self) -> &'static str {
        self.url
    }

    /// Returns the cached upstream file name.
    #[must_use]
    pub fn file_name(&self) -> &'static str {
        self.file_name
    }

    /// Returns the extracted file or directory name.
    #[must_use]
    pub fn extracted_file_name(&self) -> &'static str {
        self.extracted_file_name
    }

    /// Returns the compression used by this dataset file.
    #[must_use]
    pub fn compression(&self) -> DatasetCompression {
        self.compression
    }
}

/// Errors raised while fetching and materializing datasets.
#[derive(Debug, Error)]
pub enum DatasetError {
    /// The HTTP download failed.
    #[error("failed to download dataset from {url}: {source}")]
    Download {
        /// The upstream URL.
        url: &'static str,
        /// The underlying transport or HTTP error.
        #[source]
        source: reqwest::Error,
    },
    /// The on-disk materialization step failed.
    #[error("failed to access dataset path {path}: {source}")]
    Io {
        /// The path being read or written.
        path: PathBuf,
        /// The underlying filesystem error.
        #[source]
        source: io::Error,
    },
    /// The dataset contents did not match the expected record layout.
    #[error("failed to parse dataset {dataset_id} at line {line_number}: {message}")]
    Format {
        /// The dataset identifier.
        dataset_id: &'static str,
        /// The 1-based line number within the materialized dataset file.
        line_number: usize,
        /// A human-readable explanation of the malformed record.
        message: String,
    },
    /// The requested dataset subset is not valid.
    #[error("invalid dataset selection for {dataset_id}: {message}")]
    InvalidSelection {
        /// The dataset identifier.
        dataset_id: &'static str,
        /// A human-readable explanation of the invalid selection.
        message: String,
    },
}

/// One SMILES record from a dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSmilesRecord {
    id: String,
    smiles: String,
}

impl DatasetSmilesRecord {
    /// Creates a dataset SMILES record.
    #[must_use]
    pub fn new(id: String, smiles: String) -> Self {
        Self { id, smiles }
    }

    /// Returns the dataset-specific record identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the SMILES string.
    #[must_use]
    pub fn smiles(&self) -> &str {
        &self.smiles
    }

    /// Consumes the record and returns its SMILES string.
    #[must_use]
    pub fn into_smiles(self) -> String {
        self.smiles
    }
}
