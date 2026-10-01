use super::{
    reader::{DatasetSmilesIter, DatasetSmilesRecordIter},
    source::{DatasetSource, SmilesDatasetRecordSource, SmilesDatasetSource},
    types::{DatasetError, DatasetFetchOptions},
};

/// The official MassSpecGym benchmark TSV containing a `smiles` column.
///
/// Source:
/// `https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym.tsv`
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct MassSpecGymSmiles;

impl DatasetSource for MassSpecGymSmiles {
    fn id(&self) -> &'static str {
        "massspecgym-smiles"
    }

    fn url(&self) -> &'static str {
        "https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/main/data/MassSpecGym.tsv"
    }

    fn file_name(&self) -> &'static str {
        "MassSpecGym.tsv"
    }
}

impl SmilesDatasetSource for MassSpecGymSmiles {
    fn iter_smiles_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesIter, DatasetError> {
        Ok(DatasetSmilesIter::from_records(self.iter_records_with_options(options)?))
    }
}

impl SmilesDatasetRecordSource for MassSpecGymSmiles {
    fn iter_records_with_options(
        &self,
        options: &DatasetFetchOptions,
    ) -> Result<DatasetSmilesRecordIter, DatasetError> {
        let artifact = self.fetch_with_options(options)?;
        DatasetSmilesRecordIter::for_mass_spec_gym(&artifact)
    }
}

/// Convenient constant handle for the MassSpecGym SMILES dataset.
pub const MASS_SPEC_GYM_SMILES: MassSpecGymSmiles = MassSpecGymSmiles;
