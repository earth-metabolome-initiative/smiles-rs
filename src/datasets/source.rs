use alloc::vec::Vec;

use super::{
    fetch::{fetch_dataset, fetch_dataset_collection},
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    types::{
        DatasetArtifact, DatasetCollectionArtifact, DatasetCompression, DatasetError,
        DatasetFetchOptions, DatasetFile,
    },
};

/// Metadata for a downloadable dataset source.
pub trait DatasetSource {
    /// Stable dataset identifier used for cache subdirectories and diagnostics.
    fn id(&self) -> &'static str;

    /// Upstream URL for the dataset artifact.
    fn url(&self) -> &'static str;

    /// File name of the cached upstream artifact.
    fn file_name(&self) -> &'static str;

    /// File name of the decompressed artifact.
    ///
    /// For uncompressed sources this should equal [`Self::file_name`].
    fn extracted_file_name(&self) -> &'static str {
        self.file_name()
    }

    /// Compression used by the upstream dataset artifact.
    fn compression(&self) -> DatasetCompression {
        DatasetCompression::None
    }

    /// Fetches the dataset using default options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if the download fails or the cached artifact
    /// cannot be materialized on disk.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{DatasetSource, MASS_SPEC_GYM_SMILES};
    ///
    /// let artifact = MASS_SPEC_GYM_SMILES.fetch()?;
    /// assert!(artifact.path().ends_with("MassSpecGym.tsv"));
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    fn fetch(&self) -> Result<DatasetArtifact, DatasetError> {
        self.fetch_with_options(&DatasetFetchOptions::default())
    }

    /// Fetches the dataset using explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if the download fails or the requested cached
    /// artifact layout cannot be materialized on disk.
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
    /// assert!(artifact.path().ends_with("CID-SMILES"));
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    fn fetch_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetArtifact, DatasetError> {
        fetch_dataset(self, options)
    }
}

/// Metadata for a dataset that is distributed across multiple files.
pub trait DatasetCollectionSource {
    /// Stable dataset identifier used for cache subdirectories and diagnostics.
    fn id(&self) -> &'static str;

    /// Upstream files that make up this dataset collection.
    fn files(&self) -> Vec<DatasetFile>;

    /// Fetches the dataset collection using default options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if any file cannot be downloaded or
    /// materialized on disk.
    fn fetch_collection(&self) -> Result<DatasetCollectionArtifact, DatasetError> {
        self.fetch_collection_with_options(&DatasetFetchOptions::default())
    }

    /// Fetches the dataset collection using explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if any file cannot be downloaded or
    /// materialized on disk.
    fn fetch_collection_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetCollectionArtifact, DatasetError> {
        fetch_dataset_collection(self, options)
    }
}

/// A dataset source that can stream SMILES strings directly.
pub trait SmilesDatasetSource {
    /// Opens a streaming iterator over the dataset SMILES using default fetch
    /// options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if the dataset cannot be fetched or if the
    /// materialized dataset cannot be opened for streaming.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{PUBCHEM_SMILES, SmilesDatasetSource};
    ///
    /// let mut smiles = PUBCHEM_SMILES.iter_smiles()?;
    /// if let Some(first) = smiles.next() {
    ///     let _ = first?;
    /// }
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    fn iter_smiles(&self) -> Result<DatasetSmilesIter, DatasetError> {
        self.iter_smiles_with_options(&DatasetFetchOptions::default())
    }

    /// Opens a streaming iterator over the dataset SMILES using explicit fetch
    /// options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if the dataset cannot be fetched or if the
    /// materialized dataset cannot be opened for streaming.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use smiles_rs::datasets::{
    ///     ArchiveMode, DatasetFetchOptions, PUBCHEM_SMILES, SmilesDatasetSource,
    /// };
    ///
    /// let mut smiles = PUBCHEM_SMILES.iter_smiles_with_options(&DatasetFetchOptions {
    ///     archive_mode: ArchiveMode::KeepCompressed,
    ///     ..DatasetFetchOptions::default()
    /// })?;
    /// if let Some(first) = smiles.next() {
    ///     let _ = first?;
    /// }
    /// # Ok::<(), smiles_rs::DatasetError>(())
    /// ```
    fn iter_smiles_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesIter, DatasetError>;
}

/// A dataset source that can stream SMILES records with dataset identifiers.
pub trait SmilesDatasetRecordSource {
    /// Opens a streaming iterator over dataset records using default fetch
    /// options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if the dataset cannot be fetched or opened.
    fn iter_records(&self) -> Result<DatasetSmilesRecordIter, DatasetError> {
        self.iter_records_with_options(&DatasetFetchOptions::default())
    }

    /// Opens a streaming iterator over dataset records using explicit fetch
    /// options.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError`] if the dataset cannot be fetched or opened.
    fn iter_records_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesRecordIter, DatasetError>;
}
