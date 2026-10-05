use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrixRef;

use super::{Smiles, maybe_collapse_atom_to_organic_subset};
use crate::{
    atom::{Atom, bracketed::chirality::Chirality},
    bond::Bond,
    smiles::{BondMatrixBuilder, StereoNeighbor, render_plan::preserve_raw_directional_single},
};

pub(super) mod chirality;
mod directional_bonds;
mod preparation;

use self::{
    chirality::{atom_with_chirality, canonical_stereo_neighbors_row, normalized_stereo_chirality},
    directional_bonds::{AtomBasedDoubleBondNormalization, atom_based_override_bond},
};

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    pub(crate) fn semantic_tetrahedral_chirality(&self, node_id: usize) -> Option<Chirality> {
        let chirality = self.node_by_id(node_id)?.chirality()?;
        let parsed_neighbors = self.parsed_stereo_neighbors_row(node_id);
        let preparation = self.stereo_normalization_preparation();
        let normalized_neighbors = canonical_stereo_neighbors_row(
            self,
            node_id,
            Some(chirality),
            parsed_neighbors,
            &preparation.new_index_of_old_node,
            &preparation.rooted_classes,
            &preparation.refined_classes,
        );

        match normalized_stereo_chirality(
            self,
            node_id,
            Some(chirality),
            parsed_neighbors,
            &normalized_neighbors,
            &preparation.rooted_classes,
            &preparation.refined_classes,
        ) {
            Some(chirality @ (Chirality::TH(_) | Chirality::AL(_))) => Some(chirality),
            _ => None,
        }
    }

    pub(super) fn stereo_normal_form(&self) -> Self {
        if self.nodes().is_empty() {
            return self.clone();
        }
        if !self.has_stereo_markup_for_normalization() {
            return self.clone();
        }

        let radical_electrons = self.radical_electron_counts();

        let preparation = self.stereo_normalization_preparation();
        let new_index_of_old_node = &preparation.new_index_of_old_node;
        let refined_classes = &preparation.refined_classes;
        let rooted_classes = &preparation.rooted_classes;
        let directional_overrides = self.projected_directional_bond_overrides_with_classes(
            new_index_of_old_node,
            rooted_classes,
            refined_classes,
        );
        let atom_based_double_bond_normalization = self.atom_based_double_bond_normalization(
            new_index_of_old_node,
            rooted_classes,
            refined_classes,
        );
        let normalized_atom_rows = self.stereo_normalized_atom_rows(
            new_index_of_old_node,
            rooted_classes,
            refined_classes,
            &atom_based_double_bond_normalization,
            &radical_electrons,
        );

        let atom_nodes = normalized_atom_rows.iter().map(|(atom, _)| *atom).collect::<Vec<_>>();
        let parsed_stereo_neighbors = normalized_atom_rows
            .into_iter()
            .map(|(_atom, neighbors)| neighbors)
            .collect::<Vec<_>>();

        let mut builder = BondMatrixBuilder::with_capacity(self.number_of_bonds());
        for ((row, column), entry) in self.bond_matrix().sparse_entries() {
            if row >= column {
                continue;
            }

            let mut bond = entry.bond();
            if matches!(bond, Bond::Up | Bond::Down)
                && (!preserve_raw_directional_single(self, &directional_overrides, row, column)
                    || atom_based_double_bond_normalization.semantic_endpoints[row]
                    || atom_based_double_bond_normalization.semantic_endpoints[column])
            {
                bond = Bond::Single;
            }
            if let Some(override_bond) = directional_overrides.get(row, column) {
                bond = override_bond;
            }
            if let Some(override_bond) = atom_based_override_bond(
                &atom_based_double_bond_normalization.override_rows,
                row,
                column,
            ) {
                bond = override_bond;
            }

            builder
                .push_edge_with_descriptor(row, column, entry.descriptor().with_bond(bond), None)
                .unwrap_or_else(|_| unreachable!("stereo normalization preserves a simple graph"));
        }

        Self::from_bond_matrix_parts_with_parsed_stereo_and_radicals(
            atom_nodes,
            builder.finish(self.nodes().len()),
            parsed_stereo_neighbors,
            Some(radical_electrons.to_vec()),
        )
    }

    fn stereo_normalized_atom_rows(
        &self,
        new_index_of_old_node: &[usize],
        rooted_classes: &[usize],
        refined_classes: &[usize],
        atom_based_double_bond_normalization: &AtomBasedDoubleBondNormalization,
        radical_electrons: &[u8],
    ) -> Vec<(Atom, Vec<StereoNeighbor>)> {
        self.atom_nodes
            .iter()
            .copied()
            .enumerate()
            .map(|(node_id, atom)| {
                let chirality = atom.chirality();
                let parsed_neighbors = self.parsed_stereo_neighbors_row(node_id);
                if atom_based_double_bond_normalization.clear_chirality[node_id] {
                    let cleared = atom_with_chirality(atom, None);
                    return (
                        maybe_collapse_atom_to_organic_subset(
                            self,
                            node_id,
                            cleared,
                            radical_electrons,
                        ),
                        Vec::new(),
                    );
                }
                if chirality.is_none() && parsed_neighbors.is_empty() {
                    return (atom, Vec::new());
                }
                let normalized_neighbors = canonical_stereo_neighbors_row(
                    self,
                    node_id,
                    chirality,
                    parsed_neighbors,
                    new_index_of_old_node,
                    rooted_classes,
                    refined_classes,
                );
                let normalized_chirality = normalized_stereo_chirality(
                    self,
                    node_id,
                    chirality,
                    parsed_neighbors,
                    &normalized_neighbors,
                    rooted_classes,
                    refined_classes,
                );
                let normalized_atom = atom_with_chirality(atom, normalized_chirality);
                (
                    if normalized_chirality.is_none() {
                        maybe_collapse_atom_to_organic_subset(
                            self,
                            node_id,
                            normalized_atom,
                            radical_electrons,
                        )
                    } else {
                        normalized_atom
                    },
                    normalized_chirality.map_or_else(Vec::new, |_| normalized_neighbors),
                )
            })
            .collect::<Vec<_>>()
    }
}
