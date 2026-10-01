use super::{
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetSource, SmilesDatasetRecordSource, SmilesDatasetSource},
    types::{DatasetError, DatasetFetchOptions},
};

/// The official LOTUS natural products `smiles` dataset, in the plain text
/// `.smi` format containing a `smiles` and `id` column with no headers.
///
/// Source: `https://lotus.naturalproducts.net/download/smiles`
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct LotusSmiles;

impl DatasetSource for LotusSmiles {
    fn id(&self) -> &'static str {
        "lotus-smiles"
    }

    fn url(&self) -> &'static str {
        "https://lotus.naturalproducts.net/download/smiles"
    }

    fn file_name(&self) -> &'static str {
        "Lotus.smi"
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
