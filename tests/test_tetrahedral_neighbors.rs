//! Tetrahedral orientation relative to caller-named neighbors.

use smiles_rs::{Smiles, WildcardSmiles};

#[test]
fn named_neighbor_orientation_is_independent_of_canonical_ranking() {
    for input in ["C[C@@H](O)CC", "C[C@@H](O)OC"] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        let neighbors = [Some(0), None, Some(2), Some(3)];
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let order = [a, b, c, d];
                        if (0..4).any(|i| order[..i].contains(&order[i])) {
                            continue;
                        }
                        let inversions = (0..4)
                            .map(|i| (i + 1..4).filter(|&j| order[i] > order[j]).count())
                            .sum::<usize>();
                        assert_eq!(
                            smiles.tetrahedral_neighbors_anticlockwise(
                                1,
                                order.map(|i| neighbors[i])
                            ),
                            Some(inversions % 2 == 1),
                            "{input} {order:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn tetrahedral_tags_and_root_hydrogen_follow_written_order() {
    for (input, center, neighbors, expected) in [
        ("F[C@H](Cl)Br", 1, [Some(0), None, Some(2), Some(3)], true),
        ("F[C@@H](Cl)Br", 1, [Some(0), None, Some(2), Some(3)], false),
        ("F[C@TH1H](Cl)Br", 1, [Some(0), None, Some(2), Some(3)], true),
        ("F[C@TH2H](Cl)Br", 1, [Some(0), None, Some(2), Some(3)], false),
        ("[C@H](F)(Cl)Br", 0, [None, Some(1), Some(2), Some(3)], true),
        ("F[C@](Cl)(Br)I", 1, [Some(0), Some(2), Some(3), Some(4)], true),
        ("F[C@@](Cl)(Br)I", 1, [Some(0), Some(2), Some(3), Some(4)], false),
        ("C[C@H](C)O", 1, [Some(0), None, Some(2), Some(3)], true),
    ] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        assert_eq!(
            smiles.tetrahedral_neighbors_anticlockwise(center, neighbors),
            Some(expected),
            "{input}"
        );
    }
}

#[test]
fn ring_neighbors_follow_digit_positions_at_both_ends() {
    for (input, center, neighbors, expected) in [
        ("C[C@@H]1CCCCO1", 1, [Some(0), None, Some(6), Some(2)], false),
        ("C[C@H]1CCCCO1", 1, [Some(0), None, Some(6), Some(2)], true),
        ("O1CCCC[C@@H]1C", 5, [Some(4), None, Some(0), Some(6)], false),
        ("O1CCCC[C@@H](C)1", 5, [Some(4), None, Some(6), Some(0)], false),
    ] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        assert_eq!(
            smiles.tetrahedral_neighbors_anticlockwise(center, neighbors),
            Some(expected),
            "{input}"
        );
    }
}

#[test]
fn neighbor_sets_must_match_exactly() {
    let smiles: Smiles = "C[C@@H](O)CC".parse().expect("valid SMILES");
    for neighbors in [
        [Some(0), None, Some(2), Some(2)],
        [Some(0), None, None, Some(3)],
        [Some(0), None, Some(2), Some(4)],
        [Some(0), None, Some(2), Some(1)],
        [Some(0), None, Some(2), Some(usize::MAX)],
        [Some(0), Some(2), Some(3), Some(4)],
    ] {
        assert_eq!(smiles.tetrahedral_neighbors_anticlockwise(1, neighbors), None);
    }
    assert_eq!(smiles.tetrahedral_neighbors_anticlockwise(usize::MAX, [None; 4]), None);
}

#[test]
fn missing_incomplete_and_non_tetrahedral_stereo_return_none() {
    for (input, center, neighbors) in [
        ("CC(O)CC", 1, [Some(0), None, Some(2), Some(3)]),
        ("F[C@](Cl)Br", 1, [Some(0), None, Some(2), Some(3)]),
        ("FC(Cl)=[C@]=C(Br)I", 3, [Some(0), Some(2), Some(5), Some(6)]),
        ("F[C@AL1](Cl)(Br)I", 1, [Some(0), Some(2), Some(3), Some(4)]),
        ("F[Pt@SP1](Cl)(Br)I", 1, [Some(0), Some(2), Some(3), Some(4)]),
        ("F[P@](Cl)(Br)(I)N", 1, [Some(0), Some(2), Some(3), Some(4)]),
        ("F[P@TB1](Cl)(Br)(I)N", 1, [Some(0), Some(2), Some(3), Some(4)]),
        ("F[Co@OH1](Cl)(Br)(I)(N)O", 1, [Some(0), Some(2), Some(3), Some(4)]),
    ] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        assert_eq!(smiles.tetrahedral_neighbors_anticlockwise(center, neighbors), None, "{input}");
    }
}

#[test]
fn expanded_hydrogen_is_named_by_its_atom_index() {
    let smiles: Smiles = "F[C@H](Cl)Br".parse().expect("valid SMILES");
    let expanded = smiles.with_explicit_hydrogens();
    assert_eq!(
        expanded.tetrahedral_neighbors_anticlockwise(1, [Some(0), Some(4), Some(2), Some(3)]),
        Some(true)
    );
    assert_eq!(
        expanded.tetrahedral_neighbors_anticlockwise(1, [Some(0), None, Some(2), Some(3)]),
        None
    );
}

#[test]
fn wildcard_neighbors_keep_mapped_orientation() {
    for (input, expected) in [("*[C@H](O)CC", true), ("*[C@@H](O)CC", false)] {
        let smiles: WildcardSmiles = input.parse().expect("valid wildcard SMILES");
        assert_eq!(
            smiles.tetrahedral_neighbors_anticlockwise(1, [Some(0), None, Some(2), Some(3)]),
            Some(expected)
        );
    }
}
