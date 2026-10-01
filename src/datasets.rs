//! Programmatic dataset downloads with local caching.
//!
//! This module is available behind the `datasets` cargo feature.
//!
//! The download API is intentionally path-oriented:
//!
//! - dataset definitions provide stable metadata
//! - downloads are cached on disk
//! - callers can choose whether gzip archives stay compressed, are
//!   decompressed, or both
//!
//! ```no_run
//! use smiles_rs::datasets::{ArchiveMode, DatasetFetchOptions, DatasetSource, PUBCHEM_SMILES};
//!
//! let artifact = PUBCHEM_SMILES.fetch_with_options(&DatasetFetchOptions {
//!     archive_mode: ArchiveMode::Decompress,
//!     ..DatasetFetchOptions::default()
//! })?;
//!
//! println!("{}", artifact.path().display());
//! # Ok::<(), smiles_rs::DatasetError>(())
//! ```
//!
//! ```no_run
//! use smiles_rs::datasets::{PUBCHEM_SMILES, SmilesDatasetSource};
//!
//! let mut smiles = PUBCHEM_SMILES.iter_smiles()?;
//! if let Some(first) = smiles.next() {
//!     println!("{}", first?);
//! }
//! # Ok::<(), smiles_rs::DatasetError>(())
//! ```
mod coconut;
mod fetch;
mod lotus;
mod massspecgym;
mod progress;
mod pubchem;
mod reader;
mod source;
#[cfg(test)]
mod tests;
mod types;
mod zinc20;

pub use coconut::{COCONUT_SMILES, CoconutSmiles};
pub use fetch::default_dataset_cache_dir;
pub use lotus::{LOTUS_SMILES, LotusSmiles};
pub use massspecgym::{MASS_SPEC_GYM_SMILES, MassSpecGymSmiles};
pub use pubchem::{PUBCHEM_SMILES, PubChemSmiles};
pub use reader::{DatasetSmilesIter, DatasetSmilesRecordIter};
pub use source::{
    DatasetCollectionSource, DatasetSource, SmilesDatasetRecordSource, SmilesDatasetSource,
};
pub use types::{
    ArchiveMode, CacheMode, DatasetArtifact, DatasetCollectionArtifact, DatasetCompression,
    DatasetError, DatasetFetchOptions, DatasetFile, DatasetSmilesRecord,
};
pub use zinc20::{ZINC20_EXPECTED_RECORD_COUNT, ZINC20_SMILES, Zinc20Smiles};
