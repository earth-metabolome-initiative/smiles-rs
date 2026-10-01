use super::{
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetSource, SmilesDatasetRecordSource, SmilesDatasetSource},
    types::{DatasetCompression, DatasetError, DatasetFetchOptions},
};
/// The official COCONUT natural-products containing a `smiles` column.
///
/// Source:
/// `https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv-08-2026.zip`
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct CoconutSmiles;

impl DatasetSource for CoconutSmiles {
    fn id(&self) -> &'static str {
        "coconut-smiles"
    }

    fn url(&self) -> &'static str {
        "https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv-08-2026.zip"
    }

    fn file_name(&self) -> &'static str {
        "coconut_csv-08-2026.zip"
    }

    fn extracted_file_name(&self) -> &'static str {
        "coconut_csv-08-2026.csv"
    }

    fn compression(&self) -> DatasetCompression {
        DatasetCompression::Zip
    }
}

impl SmilesDatasetSource for CoconutSmiles {
    fn iter_smiles_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesIter, DatasetError> {
        Ok(DatasetSmilesIter::from_records(self.iter_records_with_options(options)?))
    }
}

impl SmilesDatasetRecordSource for CoconutSmiles {
    fn iter_records_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesRecordIter, DatasetError> {
        let artifact = self.fetch_with_options(options)?;
        DatasetSmilesRecordIter::for_coconut(&artifact)
    }
}

/// Convenient constant handle for the COCONUT SMILES dataset.
pub const COCONUT_SMILES: CoconutSmiles = CoconutSmiles;
