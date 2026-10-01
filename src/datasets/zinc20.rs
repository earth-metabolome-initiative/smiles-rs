use alloc::vec::Vec;
use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetCollectionSource, SmilesDatasetRecordSource, SmilesDatasetSource},
    types::{ArchiveMode, DatasetCompression, DatasetError, DatasetFetchOptions, DatasetFile},
};

/// Number of rows reported by the ZINC20-ML `smiles_count.txt` manifest.
pub const ZINC20_EXPECTED_RECORD_COUNT: usize = 1_006_651_037;

macro_rules! zinc20_chunk_file {
    ($chunk:literal) => {
        DatasetFile::new(
            concat!(
                "https://files.docking.org/zinc20-ML/smiles/ZINC20_smiles_chunk_",
                $chunk,
                ".tar.gz"
            ),
            concat!("ZINC20_smiles_chunk_", $chunk, ".tar.gz"),
            concat!("ZINC20_smiles_chunk_", $chunk),
            DatasetCompression::TarGzip,
        )
    };
}

const ZINC20_CHUNK_FILES: [DatasetFile; 20] = [
    zinc20_chunk_file!(1),
    zinc20_chunk_file!(2),
    zinc20_chunk_file!(3),
    zinc20_chunk_file!(4),
    zinc20_chunk_file!(5),
    zinc20_chunk_file!(6),
    zinc20_chunk_file!(7),
    zinc20_chunk_file!(8),
    zinc20_chunk_file!(9),
    zinc20_chunk_file!(10),
    zinc20_chunk_file!(11),
    zinc20_chunk_file!(12),
    zinc20_chunk_file!(13),
    zinc20_chunk_file!(14),
    zinc20_chunk_file!(15),
    zinc20_chunk_file!(16),
    zinc20_chunk_file!(17),
    zinc20_chunk_file!(18),
    zinc20_chunk_file!(19),
    zinc20_chunk_file!(20),
];

/// The ZINC20-ML SMILES dataset distributed as 20 `tar.gz` chunks.
///
/// Source:
/// `https://files.docking.org/zinc20-ML/smiles/`
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Zinc20Smiles {
    first_chunk: u8,
    last_chunk: u8,
}

impl Default for Zinc20Smiles {
    fn default() -> Self {
        Self::all()
    }
}

impl Zinc20Smiles {
    /// First available ZINC20-ML SMILES chunk.
    pub const FIRST_CHUNK: u8 = 1;
    /// Last available ZINC20-ML SMILES chunk.
    pub const LAST_CHUNK: u8 = 20;

    /// Returns a handle spanning all ZINC20-ML SMILES chunks.
    #[must_use]
    pub const fn all() -> Self {
        Self { first_chunk: Self::FIRST_CHUNK, last_chunk: Self::LAST_CHUNK }
    }

    /// Returns a handle for one ZINC20-ML SMILES chunk.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::InvalidSelection`] if `chunk` is outside
    /// `1..=20`.
    pub fn chunk(chunk: u8) -> Result<Self, DatasetError> {
        Self::chunk_range(chunk, chunk)
    }

    /// Returns a handle for an inclusive ZINC20-ML chunk range.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::InvalidSelection`] if the range is empty or
    /// outside `1..=20`.
    pub fn chunk_range(first_chunk: u8, last_chunk: u8) -> Result<Self, DatasetError> {
        if first_chunk < Self::FIRST_CHUNK
            || last_chunk > Self::LAST_CHUNK
            || first_chunk > last_chunk
        {
            return Err(DatasetError::InvalidSelection {
                dataset_id: "zinc20-smiles",
                message: format!(
                    "expected an inclusive chunk range within {}..={}, got {first_chunk}..={last_chunk}",
                    Self::FIRST_CHUNK,
                    Self::LAST_CHUNK
                ),
            });
        }

        Ok(Self { first_chunk, last_chunk })
    }

    /// Returns the first selected chunk.
    #[must_use]
    pub fn first_chunk(&self) -> u8 {
        self.first_chunk
    }

    /// Returns the last selected chunk.
    #[must_use]
    pub fn last_chunk(&self) -> u8 {
        self.last_chunk
    }
}

impl DatasetCollectionSource for Zinc20Smiles {
    fn id(&self) -> &'static str {
        "zinc20-smiles"
    }

    fn files(&self) -> Vec<DatasetFile> {
        let start = usize::from(self.first_chunk - 1);
        let end = usize::from(self.last_chunk);
        ZINC20_CHUNK_FILES[start..end].to_vec()
    }
}

impl SmilesDatasetSource for Zinc20Smiles {
    fn iter_smiles_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesIter, DatasetError> {
        Ok(DatasetSmilesIter::from_records(self.iter_records_with_options(options)?))
    }
}

impl SmilesDatasetRecordSource for Zinc20Smiles {
    fn iter_records_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesRecordIter, DatasetError> {
        let mut options = options.clone();
        if options.archive_mode == ArchiveMode::KeepCompressed {
            options.archive_mode = ArchiveMode::Decompress;
        }
        let artifact = self.fetch_collection_with_options(&options)?;
        DatasetSmilesRecordIter::for_zinc20(&artifact)
    }
}

/// Convenient constant handle for the full ZINC20-ML SMILES dataset.
pub const ZINC20_SMILES: Zinc20Smiles = Zinc20Smiles::all();

pub(crate) fn collect_zinc20_smiles_paths(
    dataset_id: &'static str,
    path: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), DatasetError> {
    if path.is_file() {
        if is_zinc20_smiles_file(path) {
            output.push(path.to_path_buf());
            return Ok(());
        }
        return Err(DatasetError::Format {
            dataset_id,
            line_number: 1,
            message: format!("expected an extracted ZINC20 directory, got {}", path.display()),
        });
    }

    let entries = fs::read_dir(path)
        .map_err(|source| DatasetError::Io { path: path.to_path_buf(), source })?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|source| DatasetError::Io { path: path.to_path_buf(), source })?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_zinc20_smiles_paths(dataset_id, &entry_path, &mut paths)?;
        } else if is_zinc20_smiles_file(&entry_path) {
            paths.push(entry_path);
        }
    }
    paths.sort();
    output.extend(paths);
    Ok(())
}

fn is_zinc20_smiles_file(path: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()).is_some_and(|name| {
        name.starts_with("smiles_all_")
            && Path::new(name)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("txt"))
    })
}
