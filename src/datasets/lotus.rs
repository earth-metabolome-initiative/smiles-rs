use super::{
    DatasetCompression,
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetSource, SmilesDatasetRecordSource, SmilesDatasetSource},
    types::{DatasetError, DatasetFetchOptions},
};

/// The official LOTUS natural products `260413_frozen_metadata.csv.gz` dataset
/// bulk download.
///
/// Source: `https://zenodo.org/records/19360665/files/260413_frozen_metadata.csv.gz`
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct LotusSmiles;

impl DatasetSource for LotusSmiles {
    fn id(&self) -> &'static str {
        "lotus-smiles"
    }

    fn url(&self) -> &'static str {
        "https://zenodo.org/records/19360665/files/260413_frozen_metadata.csv.gz"
    }

    fn file_name(&self) -> &'static str {
        "260413_frozen_metadata.csv.gz"
    }
    fn extracted_file_name(&self) -> &'static str {
        "260413_frozen_metadata.csv"
    }
    fn compression(&self) -> crate::prelude::DatasetCompression {
        DatasetCompression::Gzip
    }
}

impl SmilesDatasetSource for LotusSmiles {
    fn iter_smiles_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesIter, DatasetError> {
        Ok(DatasetSmilesIter::from_records(self.iter_records_with_options(options)?))
    }
}

impl SmilesDatasetRecordSource for LotusSmiles {
    fn iter_records_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesRecordIter, DatasetError> {
        let artifact = self.fetch_with_options(options)?;
        DatasetSmilesRecordIter::for_lotus(&artifact)
    }
}

/// Convenient constant for the Lotus SMILES dataset
pub const LOTUS_SMILES: LotusSmiles = LotusSmiles;
