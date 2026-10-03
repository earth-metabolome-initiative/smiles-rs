use super::{
    DatasetCompression,
    fetch::fetch_zenodo_dataset,
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetSource, SmilesDatasetRecordSource, SmilesDatasetSource, ZenodoDatasetSource},
    types::{DatasetError, DatasetFetchOptions},
};

const LOTUS_ZENODO_CONCEPT_ID: u64 = 22811236;
const LOTUS_ARCHIVE_PREFIX: &str = "lotus-wikidata-smiles-";
const LOTUS_ARCHIVE_SUFFIX: &str = ".tar.gz";

/// The official LOTUS natural products Wikidata SMILES dataset.
///
/// The latest published version is resolved through Zenodo concept record
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct LotusSmiles;

impl DatasetSource for LotusSmiles {
    fn id(&self) -> &'static str {
        "lotus-smiles"
    }

    fn url(&self) -> &'static str {
        "https://doi.org/10.5281/zenodo.22811236"
    }

    fn file_name(&self) -> &'static str {
        "lotus-wikidata-smiles.tar.gz"
    }
    fn extracted_file_name(&self) -> &'static str {
        "lotus-wikidata-smiles.csv"
    }
    fn compression(&self) -> DatasetCompression {
        DatasetCompression::TarGzip
    }
    fn fetch_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<crate::prelude::DatasetArtifact, DatasetError> {
        fetch_zenodo_dataset(self, options)
    }
}

impl ZenodoDatasetSource for LotusSmiles {
    fn zenodo_record_id(&self) -> u64 {
        LOTUS_ZENODO_CONCEPT_ID
    }

    fn zenodo_file_prefix(&self) -> &'static str {
        LOTUS_ARCHIVE_PREFIX
    }

    fn zenodo_file_suffix(&self) -> &'static str {
        LOTUS_ARCHIVE_SUFFIX
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
