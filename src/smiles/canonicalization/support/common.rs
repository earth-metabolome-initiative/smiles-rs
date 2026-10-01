use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrixRef;

use super::super::{SmilesCanonicalLabeling, remap_parsed_stereo_neighbors_row};
use crate::smiles::{
    BondMatrixBuilder, Smiles, SmilesAtomPolicy,
    canonicalization::{
        state::{
            CanonicalStereoNeighborKey, canonical_stereo_neighbor_key, canonicalization_state_key,
        },
        stereo_normalization::chirality::canonical_stereo_neighbors_row,
    },
};

fn canonicalized_parsed_stereo_neighbor_rows(
    smiles: &Smiles<impl SmilesAtomPolicy>,
) -> Vec<Vec<CanonicalStereoNeighborKey>> {
    let labeling = smiles.stereo_neutral_canonical_labeling();
    let refined_classes = smiles.stereo_neutral_refined_classes();
    let rooted_classes = smiles.stereo_neutral_rooted_classes(&refined_classes);
    smiles
        .nodes()
        .iter()
        .copied()
        .enumerate()
        .map(|(node_id, atom)| {
            canonical_stereo_neighbors_row(
                smiles,
                node_id,
                atom.chirality(),
                smiles.parsed_stereo_neighbors_row(node_id),
                labeling.new_index_of_old_node(),
                &rooted_classes,
                &refined_classes,
            )
            .into_iter()
            .map(canonical_stereo_neighbor_key)
            .collect()
        })
        .collect()
}

pub(crate) fn permute_smiles<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    order: &[usize],
) -> Smiles<AtomPolicy> {
    let labeling = SmilesCanonicalLabeling::new(order.to_vec());
    let atom_nodes =
        order.iter().copied().map(|old_node| smiles.atom_nodes[old_node]).collect::<Vec<_>>();

    let mut builder = BondMatrixBuilder::with_capacity(smiles.number_of_bonds());
    for ((row, column), entry) in smiles.bond_matrix().sparse_entries() {
        if row >= column {
            continue;
        }
        builder
            .push_edge_with_descriptor(
                labeling.new_index_of_old_node()[row],
                labeling.new_index_of_old_node()[column],
                entry.descriptor(),
                None,
            )
            .unwrap_or_else(|_| unreachable!("permuting a simple graph stays simple"));
    }

    let parsed_stereo_neighbors = order
        .iter()
        .copied()
        .map(|old_node| {
            remap_parsed_stereo_neighbors_row(smiles, old_node, labeling.new_index_of_old_node())
        })
        .collect::<Vec<_>>();

    let implicit_hydrogen_cache =
        order.iter().copied().map(|old_node| smiles.implicit_hydrogen_cache[old_node]).collect();

    Smiles::<AtomPolicy>::from_bond_matrix_parts_with_sidecars(
        atom_nodes,
        builder.finish(order.len()),
        parsed_stereo_neighbors,
        implicit_hydrogen_cache,
        None,
    )
}

#[track_caller]
pub(crate) fn same_canonicalization_state(
    left: &Smiles<impl SmilesAtomPolicy>,
    right: &Smiles<impl SmilesAtomPolicy>,
) {
    assert_eq!(left.atom_nodes, right.atom_nodes, "atom_nodes differ");
    assert_eq!(left.bond_matrix, right.bond_matrix, "bond_matrix differs");
    assert_eq!(
        canonicalized_parsed_stereo_neighbor_rows(left),
        canonicalized_parsed_stereo_neighbor_rows(right),
        "parsed_stereo_neighbors differ"
    );
    assert_eq!(
        left.implicit_hydrogen_cache, right.implicit_hydrogen_cache,
        "implicit_hydrogen_cache differs"
    );
    let left_source = left.kekulization_source.as_deref().map(canonicalization_state_key);
    let right_source = right.kekulization_source.as_deref().map(canonicalization_state_key);
    assert_eq!(left_source, right_source, "kekulization_source differs");
}
