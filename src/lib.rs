#![no_std]
#![doc = include_str!("../README.md")]

#[allow(unused_imports)]
#[macro_use]
extern crate alloc;
#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(all(feature = "datasets", not(test)))]
extern crate std;

pub mod atom;
pub mod bond;
#[cfg(feature = "datasets")]
pub mod datasets;
pub mod errors;
pub(crate) mod parser;
pub mod smiles;
pub mod token;

#[cfg(feature = "datasets")]
pub use crate::datasets::{
    ArchiveMode, CacheMode, DatasetArtifact, DatasetCollectionArtifact, DatasetCollectionSource,
    DatasetCompression, DatasetError, DatasetFetchOptions, DatasetFile, DatasetSmilesIter,
    DatasetSmilesRecord, DatasetSmilesRecordIter, DatasetSource, LOTUS_SMILES, LotusSmiles,
    MASS_SPEC_GYM_SMILES, MassSpecGymSmiles, PUBCHEM_SMILES, PubChemSmiles,
    SmilesDatasetRecordSource, SmilesDatasetSource, ZINC20_EXPECTED_RECORD_COUNT, ZINC20_SMILES,
    Zinc20Smiles, default_dataset_cache_dir,
};
pub use crate::{
    errors::{RootError, SmilesError, SmilesErrorWithSpan, SubgraphError},
    smiles::{
        AromaticityAssignment, AromaticityAssignmentApplicationError, AromaticityDiagnostic,
        AromaticityModel, AromaticityPerception, AromaticityPolicy, AromaticityRingFamilyKind,
        AromaticityStatus, AtomEnvironment, DoubleBondStereoConfig, Fragment, GraphSimilarities,
        InitialProductVertexOrdering, KekulizationError, KekulizationMode, LargestFragmentMetric,
        McesBuilder, McesResult, McesSearchMode, RdkitDefaultAromaticity, RdkitMdlAromaticity,
        RdkitSimpleAromaticity, RingAtomMembership, RingAtomMembershipScratch, RingMembership,
        Smiles, SmilesComponents, SmilesMces, SymmSssrResult, SymmSssrStatus,
        WildcardAromaticityPerception, WildcardMolecularFormulaConversionError, WildcardSmiles,
        WildcardSmilesComponents,
    },
};

/// Common imports for working with this crate.
pub mod prelude {
    #[cfg(feature = "datasets")]
    pub use crate::{
        ArchiveMode, CacheMode, DatasetArtifact, DatasetCollectionArtifact,
        DatasetCollectionSource, DatasetCompression, DatasetError, DatasetFetchOptions,
        DatasetFile, DatasetSmilesIter, DatasetSmilesRecord, DatasetSmilesRecordIter,
        DatasetSource, LOTUS_SMILES, LotusSmiles, MASS_SPEC_GYM_SMILES, MassSpecGymSmiles,
        PUBCHEM_SMILES, PubChemSmiles, SmilesDatasetRecordSource, SmilesDatasetSource,
        ZINC20_EXPECTED_RECORD_COUNT, ZINC20_SMILES, Zinc20Smiles, default_dataset_cache_dir,
    };
    pub use crate::{
        AromaticityAssignment, AromaticityAssignmentApplicationError, AromaticityDiagnostic,
        AromaticityModel, AromaticityPerception, AromaticityPolicy, AromaticityRingFamilyKind,
        AromaticityStatus, AtomEnvironment, DoubleBondStereoConfig, Fragment, GraphSimilarities,
        InitialProductVertexOrdering, KekulizationError, KekulizationMode, LargestFragmentMetric,
        McesBuilder, McesResult, McesSearchMode, RdkitDefaultAromaticity, RdkitMdlAromaticity,
        RdkitSimpleAromaticity, RingAtomMembership, RingAtomMembershipScratch, RingMembership,
        RootError, Smiles, SmilesComponents, SmilesError, SmilesErrorWithSpan, SmilesMces,
        SubgraphError, SymmSssrResult, SymmSssrStatus, WildcardAromaticityPerception,
        WildcardMolecularFormulaConversionError, WildcardSmiles, WildcardSmilesComponents,
    };
}
