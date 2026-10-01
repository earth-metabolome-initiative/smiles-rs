use super::{
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetSource, SmilesDatasetRecordSource, SmilesDatasetSource},
    types::{DatasetCompression, DatasetError, DatasetFetchOptions},
};

/// The official PubChem `CID-SMILES.gz` bulk download.
///
/// Source:
/// `https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz`
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct PubChemSmiles;

impl DatasetSource for PubChemSmiles {
    fn id(&self) -> &'static str {
        "pubchem-smiles"
    }

    fn url(&self) -> &'static str {
        "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
    }

    fn file_name(&self) -> &'static str {
        "CID-SMILES.gz"
    }

    fn extracted_file_name(&self) -> &'static str {
        "CID-SMILES"
    }

    fn compression(&self) -> DatasetCompression {
        DatasetCompression::Gzip
    }
}

impl SmilesDatasetSource for PubChemSmiles {
    fn iter_smiles_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesIter, DatasetError> {
        Ok(DatasetSmilesIter::from_records(self.iter_records_with_options(options)?))
    }
}

impl SmilesDatasetRecordSource for PubChemSmiles {
    fn iter_records_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesRecordIter, DatasetError> {
        let artifact = self.fetch_with_options(options)?;
        DatasetSmilesRecordIter::for_pubchem(&artifact)
    }
}

/// Convenient constant handle for the PubChem SMILES dataset.
pub const PUBCHEM_SMILES: PubChemSmiles = PubChemSmiles;
