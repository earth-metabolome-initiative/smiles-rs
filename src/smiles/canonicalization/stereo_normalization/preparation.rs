use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrixRef;

use crate::smiles::Smiles;

#[derive(Debug, Clone)]
pub(super) struct StereoNormalizationPreparation {
    // The stereo pass reuses these three component-insensitive views together,
    // so keep them bundled to make the call sites explicit.
    pub(super) new_index_of_old_node: Vec<usize>,
    pub(super) refined_classes: Vec<usize>,
    pub(super) rooted_classes: Vec<usize>,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    pub(super) fn has_stereo_markup_for_normalization(&self) -> bool {
        self.atom_nodes.iter().any(|atom| atom.chirality().is_some())
            || self.bond_matrix().sparse_entries().any(|((_row, _column), entry)| {
                matches!(entry.bond(), crate::bond::Bond::Up | crate::bond::Bond::Down)
            })
    }

    pub(super) fn stereo_normalization_preparation(&self) -> StereoNormalizationPreparation {
        let stereo_neutral_labeling = self.stereo_neutral_canonical_labeling();
        let (refined_classes, rooted_classes) = self.stereo_neutral_preparation_classes();
        StereoNormalizationPreparation {
            new_index_of_old_node: stereo_neutral_labeling.new_index_of_old_node().to_vec(),
            refined_classes,
            rooted_classes,
        }
    }
}
