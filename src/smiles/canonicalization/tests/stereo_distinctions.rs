use alloc::vec::Vec;

use super::super::{Smiles, canonicalization_state_key};

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
