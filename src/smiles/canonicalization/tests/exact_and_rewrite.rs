use alloc::{string::ToString, vec::Vec};

use geometric_traits::traits::SparseValuedMatrixRef;

use super::super::{
    Smiles, remap_parsed_stereo_neighbors_row,
    support::{assert_canonicalization_invariants, permute_smiles, same_canonicalization_state},
};
use crate::{parser::smiles_parser::parse_wildcard_smiles, smiles::StereoNeighbor};

#[test]
fn exact_canonicalize_is_idempotent_for_disconnected_cases() {
    for source in ["CC.O", "N[C@H](F)C.O", "c1ccccc1.CC", "CC.CN.O"] {
        let once = Smiles::from_str(source).unwrap().exact_canonicalize();
        let twice = once.exact_canonicalize();
        same_canonicalization_state(&once, &twice);
    }
}

#[test]
fn exact_canonicalize_converges_identical_disconnected_components() {
    let left = Smiles::from_str("CC.CC.O").unwrap().exact_canonicalize();
    let right = Smiles::from_str("O.CC.CC").unwrap().exact_canonicalize();

    same_canonicalization_state(&left, &right);
}

#[test]
fn canonicalize_converges_permuted_symmetric_cage_graph() {
    let original = Smiles::from_str("C12C3C4C1C5C2C3C45").unwrap();
    let permuted = permute_smiles(&original, &[7, 3, 0, 5, 2, 6, 1, 4]);

    assert_eq!(original.canonicalize(), permuted.canonicalize());
}

#[test]
fn canonicalize_converges_permuted_bridged_graph() {
    let original = Smiles::from_str("C1CC2CCC1C2").unwrap();
    let permuted = permute_smiles(&original, &[6, 2, 4, 0, 5, 1, 3]);

    assert_eq!(original.canonicalize(), permuted.canonicalize());
}

#[test]
fn canonicalize_converges_permuted_fused_graph() {
    let original = Smiles::from_str("c1cccc2ccccc12").unwrap();
    let permuted = permute_smiles(&original, &[9, 4, 0, 7, 2, 8, 1, 5, 3, 6]);

    assert_eq!(original.canonicalize(), permuted.canonicalize());
}

#[test]
fn canonicalize_remaps_parsed_stereo_neighbors() {
    let smiles = Smiles::from_str("O[C@H](F)C").unwrap();
    let stereo_normalized = smiles.canonicalization_normal_form().stereo_normal_form();
    let labeling = stereo_normalized.exact_canonical_labeling();
    let canonicalized = smiles.canonicalize();

    let expected: Vec<Vec<StereoNeighbor>> = labeling
        .order()
        .iter()
        .copied()
        .map(|old_node| {
            remap_parsed_stereo_neighbors_row(
                &stereo_normalized,
                old_node,
                labeling.new_index_of_old_node(),
            )
        })
        .collect();

    let actual = (0..canonicalized.nodes().len())
        .map(|node_id| canonicalized.parsed_stereo_neighbors_row(node_id).to_vec())
        .collect::<Vec<_>>();

    assert_eq!(actual, expected);
}

#[test]
fn canonicalize_remaps_implicit_hydrogen_cache_and_clears_provenance() {
    let aromaticized = Smiles::from_str("C1=CC=CC=C1")
        .unwrap()
        .perceive_aromaticity()
        .unwrap()
        .into_aromaticized();
    let labeling = aromaticized.canonical_labeling();
    let expected_cache = aromaticized
        .implicit_hydrogen_counts()
        .iter()
        .copied()
        .enumerate()
        .map(|(old_node, count)| (labeling.new_index_of_old_node()[old_node], count))
        .collect::<Vec<_>>();

    let canonicalized = aromaticized.canonicalize();
    let actual_cache =
        canonicalized.implicit_hydrogen_counts().iter().copied().enumerate().collect::<Vec<_>>();

    let mut expected_cache = expected_cache;
    expected_cache.sort_unstable_by_key(|&(new_node, _)| new_node);

    assert_eq!(actual_cache, expected_cache);
    assert!(canonicalized.kekulization_source.is_none());
    assert!(
        canonicalized
            .bond_matrix()
            .sparse_entries()
            .all(|((row, column), entry)| row >= column || entry.ring_num().is_none())
    );
    assert_canonicalization_invariants(&aromaticized);
}

#[test]
fn canonicalization_spelling_normal_form_keeps_aromatic_wildcards_bracketed() {
    let aromaticized = parse_wildcard_smiles("******#8OOO*c8")
        .unwrap()
        .perceive_aromaticity()
        .unwrap()
        .into_aromaticized();

    let rewritten = aromaticized.canonicalization_spelling_normal_form();

    assert_eq!(rewritten.aromaticity_assignment(), aromaticized.aromaticity_assignment());
    assert!(rewritten.to_string().contains("[*]"));
}

#[test]
fn canonicalize_converges_permuted_graph_with_sidecars() {
    let original = Smiles::from_str("C1=CC=CC=C1[C@H](F)O")
        .unwrap()
        .perceive_aromaticity()
        .unwrap()
        .into_aromaticized();
    let permuted = permute_smiles(&original, &[8, 3, 6, 1, 7, 0, 4, 2, 5]);

    assert_eq!(original.canonicalize(), permuted.canonicalize());
}
