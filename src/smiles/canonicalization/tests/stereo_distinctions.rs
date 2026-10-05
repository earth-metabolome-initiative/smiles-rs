use alloc::vec::Vec;

use super::super::{
    Smiles, canonicalization_state_key,
    support::{assert_canonicalization_invariants, same_canonicalization_state},
};

#[track_caller]
fn assert_distinct_canonicalization_state(left: &Smiles, right: &Smiles) {
    assert_ne!(
        canonicalization_state_key(left),
        canonicalization_state_key(right),
        "canonicalization state unexpectedly converged"
    );
}

#[test]
fn canonicalize_converges_source_backed_square_planar_groups() {
    let cis_group = [
        "[NH3][Pt@SP1]([NH3])(Cl)Cl",
        "[NH3][Pt@SP3]([NH3])(Cl)Cl",
        "[NH3][Pt@SP2](Cl)([NH3])Cl",
        "[NH3][Pt@SP1](Cl)(Cl)[NH3]",
    ];
    let trans_group =
        ["[NH3][Pt@SP2]([NH3])(Cl)Cl", "[NH3][Pt@SP1](Cl)([NH3])Cl", "[NH3][Pt@SP3](Cl)(Cl)[NH3]"];

    let cis_canonicalized = cis_group
        .iter()
        .map(|source| Smiles::from_str(source).unwrap().canonicalize())
        .collect::<Vec<_>>();
    for other in &cis_canonicalized[1..] {
        assert_eq!(&cis_canonicalized[0], other);
    }

    let trans_canonicalized = trans_group
        .iter()
        .map(|source| Smiles::from_str(source).unwrap().canonicalize())
        .collect::<Vec<_>>();
    for other in &trans_canonicalized[1..] {
        assert_eq!(&trans_canonicalized[0], other);
    }

    assert_distinct_canonicalization_state(&cis_canonicalized[0], &trans_canonicalized[0]);
}

#[test]
fn canonicalize_preserves_non_tetrahedral_stereo_distinctions() {
    let sp1 = Smiles::from_str("F[Po@SP1](Cl)(Br)I").unwrap().canonicalize();
    let sp2 = Smiles::from_str("F[Po@SP2](Cl)(Br)I").unwrap().canonicalize();
    let sp3 = Smiles::from_str("F[Po@SP3](Cl)(Br)I").unwrap().canonicalize();

    let tb1 = Smiles::from_str("S[As@TB1](F)(Cl)(Br)N").unwrap().canonicalize();
    let tb2 = Smiles::from_str("S[As@TB2](F)(Cl)(Br)N").unwrap().canonicalize();

    let oh1 = Smiles::from_str("C[Co@OH1](F)(Cl)(Br)(I)S").unwrap().canonicalize();
    let oh2 = Smiles::from_str("C[Co@OH2](F)(Cl)(Br)(I)S").unwrap().canonicalize();

    assert_distinct_canonicalization_state(&sp1, &sp2);
    assert_distinct_canonicalization_state(&sp1, &sp3);
    assert_distinct_canonicalization_state(&sp2, &sp3);
    assert_distinct_canonicalization_state(&tb1, &tb2);
    assert_distinct_canonicalization_state(&oh1, &oh2);
}

#[test]
fn canonicalize_clears_orphan_direction_beside_semantic_alkene_in_same_component() {
    let e_group = ["C/C=C/CC", "C/C=C/C/C", "C/C=C/C\\C"];
    let e_canonicalized = e_group
        .iter()
        .map(|source| {
            let smiles = Smiles::from_str(source).unwrap();
            assert_canonicalization_invariants(&smiles);
            smiles.canonicalize()
        })
        .collect::<Vec<_>>();
    for other in &e_canonicalized[1..] {
        same_canonicalization_state(&e_canonicalized[0], other);
    }

    let z_group = ["C/C=C\\CC", "C/C=C\\C\\C"];
    let z_canonicalized = z_group
        .iter()
        .map(|source| {
            let smiles = Smiles::from_str(source).unwrap();
            assert_canonicalization_invariants(&smiles);
            smiles.canonicalize()
        })
        .collect::<Vec<_>>();
    for other in &z_canonicalized[1..] {
        same_canonicalization_state(&z_canonicalized[0], other);
    }

    assert_distinct_canonicalization_state(&e_canonicalized[0], &z_canonicalized[0]);
}

#[test]
fn canonicalize_clears_orphan_direction_in_disconnected_component_beside_semantic_alkene() {
    let orphan_group = ["C/C.C/C=C/C", "C\\C.C/C=C/C", "CC.C/C=C/C"];
    let canonicalized = orphan_group
        .iter()
        .map(|source| {
            let smiles = Smiles::from_str(source).unwrap();
            assert_canonicalization_invariants(&smiles);
            smiles.canonicalize()
        })
        .collect::<Vec<_>>();
    for other in &canonicalized[1..] {
        same_canonicalization_state(&canonicalized[0], other);
    }

    let z_alkene = Smiles::from_str("C/C.C/C=C\\C").unwrap().canonicalize();
    assert_distinct_canonicalization_state(&canonicalized[0], &z_alkene);
}

#[test]
fn canonicalize_clears_partial_and_non_stereogenic_double_bond_directions() {
    let cases = [("CC=C/C", "CC=C-C"), ("CC/C=C", "CC-C=C"), ("C/C=C(C)C", "C-C=C(C)C")];

    for (marked_source, unmarked_source) in cases {
        let marked = Smiles::from_str(marked_source).unwrap();
        assert_canonicalization_invariants(&marked);

        let unmarked = Smiles::from_str(unmarked_source).unwrap();
        same_canonicalization_state(&marked.canonicalize(), &unmarked.canonicalize());
    }
}

#[test]
fn canonicalize_clears_small_ring_alkene_directions_and_keeps_macrocyclic_stereo() {
    let small_ring_group = ["C1CCC/C=C/1", r"C1CCC\C=C\1", "C1CCCC=C1"];
    let canonicalized = small_ring_group
        .iter()
        .map(|source| {
            let smiles = Smiles::from_str(source).unwrap();
            assert_canonicalization_invariants(&smiles);
            smiles.canonicalize()
        })
        .collect::<Vec<_>>();
    for other in &canonicalized[1..] {
        same_canonicalization_state(&canonicalized[0], other);
    }

    let macrocyclic_trans = Smiles::from_str("C1CCCCC/C=C/1").unwrap().canonicalize();
    let macrocyclic_unmarked = Smiles::from_str("C1CCCCCC=C1").unwrap().canonicalize();
    assert_distinct_canonicalization_state(&macrocyclic_trans, &macrocyclic_unmarked);
}

#[test]
fn canonicalize_shared_directional_bond_in_conjugated_diene_keeps_semantic_alkene() {
    let e_group = ["CC/C=C/C=C", r"CC\C=C\C=C"];
    let e_canonicalized = e_group
        .iter()
        .map(|source| {
            let smiles = Smiles::from_str(source).unwrap();
            assert_canonicalization_invariants(&smiles);
            smiles.canonicalize()
        })
        .collect::<Vec<_>>();
    for other in &e_canonicalized[1..] {
        same_canonicalization_state(&e_canonicalized[0], other);
    }

    let z_diene = Smiles::from_str("CC/C=C\\C=C").unwrap().canonicalize();
    assert_distinct_canonicalization_state(&e_canonicalized[0], &z_diene);

    let unassigned_diene = Smiles::from_str("CCC=C/C=C").unwrap().canonicalize();
    assert_distinct_canonicalization_state(&e_canonicalized[0], &unassigned_diene);
}

#[test]
fn canonicalize_clears_orphan_direction_beside_cumulene_stereo_in_mixed_component() {
    let orphan_group =
        ["C/C.F/C=C=C=C/C(C)=C(C)C", "C\\C.F/C=C=C=C/C(C)=C(C)C", "CC.F/C=C=C=C/C(C)=C(C)C"];
    let canonicalized = orphan_group
        .iter()
        .map(|source| {
            let smiles = Smiles::from_str(source).unwrap();
            assert_canonicalization_invariants(&smiles);
            smiles.canonicalize()
        })
        .collect::<Vec<_>>();
    for other in &canonicalized[1..] {
        same_canonicalization_state(&canonicalized[0], other);
    }

    let opposite_cumulene = Smiles::from_str(r"CC.F/C=C=C=C\C(C)=C(C)C").unwrap().canonicalize();
    assert_distinct_canonicalization_state(&canonicalized[0], &opposite_cumulene);
}

#[test]
fn canonicalize_clears_orphan_direction_beside_tetrahedral_stereo() {
    let marked = Smiles::from_str("N[C@](Br)(O)C/C").unwrap();
    assert_canonicalization_invariants(&marked);
    let unmarked = Smiles::from_str("N[C@](Br)(O)CC").unwrap();
    same_canonicalization_state(&marked.canonicalize(), &unmarked.canonicalize());

    let enantiomer = Smiles::from_str("N[C@@](Br)(O)CC").unwrap().canonicalize();
    assert_distinct_canonicalization_state(&unmarked.canonicalize(), &enantiomer);
}

#[test]
fn canonicalize_clears_orphan_direction_beside_atom_based_double_bond_stereo() {
    let marked = Smiles::from_str("F[C@@H](C/C)=[C@H]F").unwrap();
    assert_canonicalization_invariants(&marked);
    let unmarked = Smiles::from_str("F[C@@H](CC)=[C@H]F").unwrap();
    same_canonicalization_state(&marked.canonicalize(), &unmarked.canonicalize());

    let opposite = Smiles::from_str("F[C@H](CC)=[C@H]F").unwrap().canonicalize();
    assert_distinct_canonicalization_state(&unmarked.canonicalize(), &opposite);
}
