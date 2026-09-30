use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrixRef;

use super::{
    super::remap_parsed_stereo_neighbors_row,
    common::{permute_smiles, same_canonicalization_state},
};
use crate::{
    atom::bracketed::chirality::Chirality,
    smiles::{
        AromaticityAssignmentApplicationError, AromaticityPolicy, Smiles, StereoNeighbor,
        canonicalization::{
            SmilesCanonicalLabeling,
            state::{
                CanonicalStereoNeighborKey, canonical_chirality_key, canonical_stereo_neighbor_key,
            },
            stereo_normalization::chirality::{
                canonical_stereo_neighbors_row, stereo_chirality_normal_form,
                tetrahedral_like_is_stereogenic,
            },
        },
        double_bond_stereo::DoubleBondStereoConfig,
        stereo::normalized_tetrahedral_chirality,
    },
};

fn supports_tetrahedral_stereo_normalization(chirality: Option<Chirality>) -> bool {
    matches!(chirality, Some(Chirality::At | Chirality::AtAt | Chirality::TH(_) | Chirality::AL(_)))
}

fn remap_atom_ids(atom_ids: &[usize], new_index_of_old_node: &[usize]) -> Vec<usize> {
    let mut atom_ids = atom_ids
        .iter()
        .copied()
        .map(|old_node| new_index_of_old_node[old_node])
        .collect::<Vec<_>>();
    atom_ids.sort_unstable();
    atom_ids.dedup();
    atom_ids
}

fn remap_bond_edges(bond_edges: &[[usize; 2]], new_index_of_old_node: &[usize]) -> Vec<[usize; 2]> {
    let mut bond_edges = bond_edges
        .iter()
        .copied()
        .map(|[node_a, node_b]| {
            let new_a = new_index_of_old_node[node_a];
            let new_b = new_index_of_old_node[node_b];
            if new_a < new_b { [new_a, new_b] } else { [new_b, new_a] }
        })
        .collect::<Vec<_>>();
    bond_edges.sort_unstable();
    bond_edges.dedup();
    bond_edges
}

fn remap_aromaticity_subgraph(
    atom_ids: &[usize],
    bond_edges: &[[usize; 2]],
    new_index_of_old_node: &[usize],
) -> (Vec<usize>, Vec<[usize; 2]>) {
    (
        remap_atom_ids(atom_ids, new_index_of_old_node),
        remap_bond_edges(bond_edges, new_index_of_old_node),
    )
}

fn remap_aromaticity_application_error(
    error: AromaticityAssignmentApplicationError,
    new_index_of_old_node: &[usize],
) -> AromaticityAssignmentApplicationError {
    match error {
        AromaticityAssignmentApplicationError::AtomIdOutOfBounds { atom_id, atom_count } => {
            AromaticityAssignmentApplicationError::AtomIdOutOfBounds { atom_id, atom_count }
        }
        AromaticityAssignmentApplicationError::BondEdgeAtomOutOfBounds {
            node_a,
            node_b,
            atom_count,
        } => {
            AromaticityAssignmentApplicationError::BondEdgeAtomOutOfBounds {
                node_a,
                node_b,
                atom_count,
            }
        }
        AromaticityAssignmentApplicationError::MissingBondEdge { node_a, node_b } => {
            let new_a = new_index_of_old_node[node_a];
            let new_b = new_index_of_old_node[node_b];
            let (node_a, node_b) = if new_a < new_b { (new_a, new_b) } else { (new_b, new_a) };
            AromaticityAssignmentApplicationError::MissingBondEdge { node_a, node_b }
        }
        AromaticityAssignmentApplicationError::AromaticBondMissingEndpointAtoms {
            node_a,
            node_b,
        } => {
            let new_a = new_index_of_old_node[node_a];
            let new_b = new_index_of_old_node[node_b];
            let (node_a, node_b) = if new_a < new_b { (new_a, new_b) } else { (new_b, new_a) };
            AromaticityAssignmentApplicationError::AromaticBondMissingEndpointAtoms {
                node_a,
                node_b,
            }
        }
        AromaticityAssignmentApplicationError::WouldClearAromaticAtom { atom_id } => {
            AromaticityAssignmentApplicationError::WouldClearAromaticAtom {
                atom_id: new_index_of_old_node[atom_id],
            }
        }
        AromaticityAssignmentApplicationError::WouldClearAromaticBond { node_a, node_b } => {
            let new_a = new_index_of_old_node[node_a];
            let new_b = new_index_of_old_node[node_b];
            let (node_a, node_b) = if new_a < new_b { (new_a, new_b) } else { (new_b, new_a) };
            AromaticityAssignmentApplicationError::WouldClearAromaticBond { node_a, node_b }
        }
    }
}

fn same_aromaticity_application_error_kind(
    left: AromaticityAssignmentApplicationError,
    right: AromaticityAssignmentApplicationError,
) -> bool {
    core::mem::discriminant(&left) == core::mem::discriminant(&right)
}

#[track_caller]
fn assert_same_aromatic_subgraph(
    actual_atom_ids: &[usize],
    actual_bond_edges: &[[usize; 2]],
    expected_atom_ids: &[usize],
    expected_bond_edges: &[[usize; 2]],
    message: &str,
) {
    // Partial-vs-complete aromaticity status and fused-subsystem diagnostics
    // can vary across isomorphic reorderings because the bounded
    // ring-family search is intentionally approximate. The chemical
    // invariant we need here is that canonicalization preserves the
    // aromatic atom/bond assignment itself.
    assert_eq!(actual_atom_ids, expected_atom_ids, "{message}: atom ids differ");
    assert_eq!(actual_bond_edges, expected_bond_edges, "{message}: bond edges differ");
}

fn deterministic_permutations(node_count: usize) -> Vec<Vec<usize>> {
    let mut permutations = Vec::new();
    if node_count >= 2 {
        permutations.push((0..node_count).rev().collect());
    }
    if node_count >= 3 {
        permutations.push((1..node_count).chain(core::iter::once(0)).collect());
    }
    permutations
}

#[track_caller]
fn assert_labeling_is_a_permutation(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    labeling: &SmilesCanonicalLabeling,
) {
    let node_count = smiles.nodes().len();
    assert_eq!(labeling.order().len(), node_count);
    assert_eq!(labeling.new_index_of_old_node().len(), node_count);

    let mut seen_old_nodes = vec![false; node_count];
    for (new_index, old_node) in labeling.order().iter().copied().enumerate() {
        assert!(old_node < node_count, "canonical order contains out-of-bounds node id");
        assert!(!seen_old_nodes[old_node], "canonical order repeats original node id {old_node}");
        seen_old_nodes[old_node] = true;
        assert_eq!(
            labeling.new_index_of_old_node()[old_node],
            new_index,
            "order and inverse permutation disagree",
        );
    }

    let mut seen_new_indices = vec![false; node_count];
    for &new_index in labeling.new_index_of_old_node() {
        assert!(new_index < node_count, "inverse permutation contains out-of-bounds index");
        assert!(
            !seen_new_indices[new_index],
            "inverse permutation repeats canonical index {new_index}",
        );
        seen_new_indices[new_index] = true;
    }
}

#[track_caller]
fn assert_normal_form_chemistry_invariants(
    original: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    normalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
) {
    assert_eq!(
        normalized.nodes().len(),
        original.nodes().len(),
        "normalization changed node count",
    );
    assert_eq!(
        normalized.number_of_bonds(),
        original.number_of_bonds(),
        "normalization changed bond count",
    );
    assert_eq!(
        normalized.ring_membership(),
        original.ring_membership(),
        "normalization changed ring membership",
    );

    for policy in [
        AromaticityPolicy::RdkitDefault,
        AromaticityPolicy::RdkitSimple,
        AromaticityPolicy::RdkitMdl,
    ] {
        let normalized_assignment = normalized.aromaticity_assignment_for(policy);
        let original_assignment = original.aromaticity_assignment_for(policy);
        assert_same_aromatic_subgraph(
            normalized_assignment.atom_ids(),
            normalized_assignment.bond_edges(),
            original_assignment.atom_ids(),
            original_assignment.bond_edges(),
            &format!("normalization changed aromaticity assignment for policy {policy:?}"),
        );
    }
}

type TetrahedralStereoSignature = ((u8, u8), Vec<CanonicalStereoNeighborKey>);
type DoubleBondStereoEntry = ([usize; 2], [(usize, usize); 2], DoubleBondStereoConfig);
type DoubleBondStereoSignature = Vec<Vec<DoubleBondStereoEntry>>;

fn tetrahedral_stereo_signature(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
) -> Vec<TetrahedralStereoSignature> {
    let labeling = smiles.stereo_neutral_canonical_labeling();
    let refined_classes = smiles.stereo_neutral_refined_classes();
    let rooted_classes = smiles.stereo_neutral_rooted_classes(&refined_classes);
    let mut signature = smiles
        .nodes()
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(node_id, atom)| {
            if !supports_tetrahedral_stereo_normalization(atom.chirality()) {
                return None;
            }
            let parsed_neighbors = smiles.parsed_stereo_neighbors_row(node_id);
            let chirality =
                stereo_chirality_normal_form(smiles, node_id, atom.chirality(), parsed_neighbors)?;
            if matches!(chirality, Chirality::At | Chirality::AtAt | Chirality::TH(_))
                && !tetrahedral_like_is_stereogenic(
                    smiles,
                    node_id,
                    chirality,
                    parsed_neighbors,
                    &rooted_classes,
                    &refined_classes,
                )
            {
                return None;
            }
            let canonical_neighbors = canonical_stereo_neighbors_row(
                smiles,
                node_id,
                atom.chirality(),
                parsed_neighbors,
                labeling.new_index_of_old_node(),
                &rooted_classes,
                &refined_classes,
            );
            // Do not anchor tetrahedral invariants on a particular canonical
            // node index. Stereo normalization may change disconnected
            // component ordering in the stereo-neutral labeling while leaving
            // the local stereochemical signature unchanged.
            Some((
                canonical_chirality_key(normalized_tetrahedral_chirality(
                    Some(chirality),
                    parsed_neighbors,
                    &canonical_neighbors,
                )),
                canonical_neighbors
                    .into_iter()
                    .map(canonical_stereo_neighbor_key)
                    .collect::<Vec<_>>(),
            ))
        })
        .collect::<Vec<_>>();
    signature.sort_unstable();
    signature
}

// Component-local ranks, as stereo normalization may reorder components.
fn double_bond_stereo_signature(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
) -> DoubleBondStereoSignature {
    let labeling = smiles.stereo_neutral_canonical_labeling();
    let components = smiles.connected_components();
    let mut next_local_rank = vec![0_usize; components.number_of_components()];
    let mut local_rank = vec![0_usize; smiles.nodes().len()];
    for &node_id in labeling.order() {
        let rank = &mut next_local_rank[components.component_of_node(node_id)];
        local_rank[node_id] = *rank;
        *rank += 1;
    }

    let mut signature = vec![Vec::new(); components.number_of_components()];
    for record in smiles.double_bond_stereo_records() {
        let (side_a, side_b) = (record.side_a(), record.side_b());
        let mut endpoints = [local_rank[side_a.endpoint()], local_rank[side_b.endpoint()]];
        endpoints.sort_unstable();
        let mut sides = [
            (local_rank[side_a.endpoint()], local_rank[side_a.reference_atom()]),
            (local_rank[side_b.endpoint()], local_rank[side_b.reference_atom()]),
        ];
        sides.sort_unstable();
        signature[components.component_of_node(side_a.endpoint())].push((
            endpoints,
            sides,
            record.config(),
        ));
    }
    signature.retain(|entries| !entries.is_empty());
    for entries in &mut signature {
        entries.sort_unstable();
    }
    signature.sort_unstable();
    signature
}

#[track_caller]
fn assert_stereo_normal_form_invariants(
    normalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    stereo_normalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
) {
    assert_eq!(
        tetrahedral_stereo_signature(normalized),
        tetrahedral_stereo_signature(stereo_normalized),
        "tetrahedral stereo changed under stereo normalization",
    );
    assert_eq!(
        double_bond_stereo_signature(normalized),
        double_bond_stereo_signature(stereo_normalized),
        "double-bond stereo changed under stereo normalization",
    );
}

#[track_caller]
fn assert_core_rewrite_invariants(
    normalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    canonicalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    order: &[usize],
    new_index_of_old_node: &[usize],
) {
    let expected_atoms: Vec<_> =
        order.iter().copied().map(|old_node| normalized.atom_nodes[old_node]).collect();
    assert_eq!(canonicalized.atom_nodes, expected_atoms, "atom relabeling mismatch");
    assert_eq!(
        canonicalized.nodes().len(),
        normalized.nodes().len(),
        "canonicalization changed node count",
    );
    assert_eq!(
        canonicalized.number_of_bonds(),
        normalized.number_of_bonds(),
        "canonicalization changed bond count",
    );

    for ((row, column), entry) in normalized.bond_matrix().sparse_entries() {
        if row >= column {
            continue;
        }
        let new_row = new_index_of_old_node[row];
        let new_column = new_index_of_old_node[column];
        let rewritten =
            canonicalized.edge_for_node_pair((new_row, new_column)).unwrap_or_else(|| {
                panic!("missing edge after canonicalization: {new_row}-{new_column}")
            });
        assert_eq!(
            rewritten.bond(),
            entry.bond(),
            "bond kind changed during canonicalization for edge {row}-{column}",
        );
        assert_eq!(
            rewritten.is_aromatic(),
            entry.aromatic(),
            "bond aromaticity changed during canonicalization for edge {row}-{column}",
        );
        assert!(
            rewritten.ring_num().is_none(),
            "canonicalization should clear ring-number provenance"
        );
    }

    let expected_neighbors: Vec<Vec<StereoNeighbor>> = order
        .iter()
        .copied()
        .map(|old_node| {
            remap_parsed_stereo_neighbors_row(normalized, old_node, new_index_of_old_node)
        })
        .collect();
    assert_eq!(
        canonicalized.parsed_stereo_neighbors, expected_neighbors,
        "parsed stereo neighbors were not remapped correctly",
    );

    let expected_hydrogen_counts: Vec<_> = order
        .iter()
        .copied()
        .map(|old_node| normalized.implicit_hydrogen_count(old_node))
        .collect();
    assert_eq!(
        canonicalized.implicit_hydrogen_counts(),
        expected_hydrogen_counts.as_slice(),
        "implicit hydrogen counts were not remapped correctly",
    );

    let expected_cache = order
        .iter()
        .copied()
        .map(|old_node| normalized.implicit_hydrogen_cache[old_node])
        .collect::<Vec<_>>();
    assert_eq!(
        canonicalized.implicit_hydrogen_cache, expected_cache,
        "implicit hydrogen cache was not remapped correctly",
    );
}

#[track_caller]
fn assert_ring_membership_invariants(
    normalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    canonicalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    new_index_of_old_node: &[usize],
) {
    let expected_ring_atoms =
        remap_atom_ids(normalized.ring_membership().atom_ids(), new_index_of_old_node);
    let expected_ring_bonds =
        remap_bond_edges(normalized.ring_membership().bond_edges(), new_index_of_old_node);
    let actual_ring_membership = canonicalized.ring_membership();
    assert_eq!(
        actual_ring_membership.atom_ids(),
        expected_ring_atoms,
        "ring membership atoms were not remapped correctly",
    );
    assert_eq!(
        actual_ring_membership.bond_edges(),
        expected_ring_bonds,
        "ring membership bonds were not remapped correctly",
    );
}

#[track_caller]
fn assert_aromaticity_invariants(
    normalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    canonicalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    new_index_of_old_node: &[usize],
) {
    for policy in [
        AromaticityPolicy::RdkitDefault,
        AromaticityPolicy::RdkitSimple,
        AromaticityPolicy::RdkitMdl,
    ] {
        let expected_assignment = normalized.aromaticity_assignment_for(policy);
        let (expected_atom_ids, expected_bond_edges) = remap_aromaticity_subgraph(
            expected_assignment.atom_ids(),
            expected_assignment.bond_edges(),
            new_index_of_old_node,
        );
        let actual_assignment = canonicalized.aromaticity_assignment_for(policy);
        assert_same_aromatic_subgraph(
            actual_assignment.atom_ids(),
            actual_assignment.bond_edges(),
            &expected_atom_ids,
            &expected_bond_edges,
            &format!("aromaticity assignment changed under canonicalization for policy {policy:?}"),
        );

        match (
            normalized.perceive_aromaticity_for(policy),
            canonicalized.perceive_aromaticity_for(policy),
        ) {
            (Ok(expected), Ok(actual)) => {
                same_canonicalization_state(
                    &expected.into_aromaticized().canonicalize(),
                    &actual.into_aromaticized().canonicalize(),
                );
            }
            (Err(expected), Err(actual)) => {
                let remapped_expected =
                    remap_aromaticity_application_error(expected, new_index_of_old_node);
                assert!(
                    same_aromaticity_application_error_kind(actual, remapped_expected),
                    "aromaticity perception error changed under canonicalization for policy {policy:?}",
                );
            }
            (expected, actual) => {
                panic!(
                    "aromaticity perception success changed under canonicalization for policy {policy:?}: before={expected:?} after={actual:?}"
                );
            }
        }
    }
}

#[track_caller]
fn assert_kekulization_invariants(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    canonicalized: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
) {
    match (smiles.kekulize_standalone(), canonicalized.kekulize_standalone()) {
        (Ok(expected), Ok(actual)) => {
            same_canonicalization_state(&expected.canonicalize(), &actual.canonicalize());
        }
        (Err(expected), Err(actual)) => {
            assert_eq!(
                actual, expected,
                "standalone kekulization error changed under canonicalization",
            );
        }
        (expected, actual) => {
            panic!(
                "standalone kekulization success changed under canonicalization: before={expected:?} after={actual:?}"
            );
        }
    }
}

#[track_caller]
pub(crate) fn assert_canonicalization_invariants(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
) {
    let normalized = smiles.canonicalization_normal_form();
    assert_normal_form_chemistry_invariants(smiles, &normalized);
    let stereo_normalized = normalized.stereo_normal_form();
    assert_stereo_normal_form_invariants(&normalized, &stereo_normalized);

    let labeling = stereo_normalized.exact_canonical_labeling();
    assert_labeling_is_a_permutation(&stereo_normalized, &labeling);

    let pre_kekule_canonical = stereo_normalized.exact_canonicalize();
    let canonicalized = smiles.canonicalize();
    let order = labeling.order();
    let new_index_of_old_node = labeling.new_index_of_old_node();
    let exact_preserves_aromatic_subgraphs =
        super::super::exact_preserves_aromatic_subgraphs(&stereo_normalized, &pre_kekule_canonical);

    if exact_preserves_aromatic_subgraphs {
        assert_core_rewrite_invariants(
            &stereo_normalized,
            &pre_kekule_canonical,
            order,
            new_index_of_old_node,
        );
        assert_ring_membership_invariants(
            &stereo_normalized,
            &pre_kekule_canonical,
            new_index_of_old_node,
        );
        assert_aromaticity_invariants(
            &stereo_normalized,
            &pre_kekule_canonical,
            new_index_of_old_node,
        );
        assert_kekulization_invariants(&stereo_normalized, &canonicalized);
    }

    assert!(
        canonicalized.kekulization_source.is_none(),
        "canonicalization must clear kekulization provenance",
    );

    assert!(canonicalized.is_canonical(), "canonicalize() did not reach a fixpoint canonical form");

    let recanonicalized = canonicalized.canonicalize();
    same_canonicalization_state(&canonicalized, &recanonicalized);

    let recanonicalized_labeling = canonicalized.exact_canonical_labeling();
    assert_labeling_is_a_permutation(&canonicalized, &recanonicalized_labeling);

    for order in deterministic_permutations(smiles.nodes().len()) {
        let permuted = permute_smiles(smiles, &order);
        let permuted_canonicalized = permuted.canonicalize();
        same_canonicalization_state(&canonicalized, &permuted_canonicalized);
    }
}
