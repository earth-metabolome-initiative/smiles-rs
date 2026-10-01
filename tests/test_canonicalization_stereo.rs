//! Public canonicalization behavior tests for stereo-sensitive cases.

mod canonicalization_common;

use canonicalization_common::{assert_same_canonical_group, canonical_string};

#[test]
fn canonicalize_preserves_directional_bond_distinction() {
    assert_ne!(canonical_string("F/C=C/F"), canonical_string("F/C=C\\F"));
}

#[test]
fn canonicalize_converges_alkene_stereo_equivalence_groups() {
    let groups = [
        &["F/C=C/F", "F\\C=C\\F"][..],
        &["F/C=C\\F", "F\\C=C/F"][..],
        &["C/C=C/C", "C\\C=C\\C"][..],
        &["Cl/C=C\\Cl", "Cl\\C=C/Cl"][..],
        &["Cl/C=C/I", "Cl\\C=C\\I"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_converges_atom_based_alkene_stereo_equivalence_groups() {
    let groups = [
        &["F[C@@H]=[C@H]F", "F[C@H]=[C@@H]F", "F/C=C/F", "[C@H](F)=[C@H]F"][..],
        &["F[C@H]=[C@H]F", "F[C@@H]=[C@@H]F", "F/C=C\\F", "[C@@H](F)=[C@H]F"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_converges_tetrahedral_stereo_equivalence_groups() {
    let groups = [
        &[
            "N[C@](Br)(O)C",
            "Br[C@](O)(N)C",
            "O[C@](Br)(C)N",
            "Br[C@](C)(O)N",
            "C[C@](Br)(N)O",
            "Br[C@](N)(C)O",
            "C[C@@](Br)(O)N",
            "Br[C@@](N)(O)C",
            "[C@@](C)(Br)(O)N",
            "[C@@](Br)(N)(O)C",
        ][..],
        &["N[C@H](O)C", "O[C@H](C)N", "C[C@@H](O)N"][..],
        &["FC1C[C@](Br)(Cl)CCC1", "[C@]1(Br)(Cl)CCCC(F)C1"][..],
        &["C[C@H]1CCCCO1", "O1CCCC[C@@H]1C"][..],
        &["N[C@](Br)(O)C", "N[C@TH1](Br)(O)C"][..],
        &["N[C@@](Br)(O)C", "N[C@TH2](Br)(O)C"][..],
        &["[C@H](O)(N)C", "O[C@@H](N)C"][..],
        &["OC(Cl)=[C@]=C(C)F", "OC(Cl)=[C@AL1]=C(C)F"][..],
        &["OC(Cl)=[C@@]=C(C)F", "OC(Cl)=[C@AL2]=C(C)F"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_clears_achiral_tetrahedral_markup() {
    assert_same_canonical_group(&["BrC(Br)C", "Br[C@H](Br)C", "Br[C@@H](Br)C"]);
}

#[test]
fn canonicalize_clears_non_stereogenic_alkene_markup() {
    assert_same_canonical_group(&["FC(F)=CF", "F/C(/F)=C/F", "F/C(/F)=C\\F"]);
}

#[test]
fn canonicalize_converges_non_tetrahedral_stereo_equivalence_groups() {
    let groups = [
        &["F[Po@SP1](Cl)(Br)I", "F[Po@SP2](Br)(Cl)I", "F[Po@SP3](Cl)(I)Br"][..],
        &["S[As@TB1](F)(Cl)(Br)N", "S[As@TB2](Br)(Cl)(F)N"][..],
        &["S[As@TB5](F)(N)(Cl)Br", "F[As@TB10](S)(Cl)(N)Br"][..],
        &["F[As@TB15](Cl)(S)(Br)N", "Br[As@TB20](Cl)(S)(F)N"][..],
        &["C[Co@OH1](F)(Cl)(Br)(I)S", "C[Co@OH8](F)(Br)(Cl)(I)S"][..],
        &["C[Co@](F)(Cl)(Br)(I)S", "F[Co@@](S)(I)(C)(Cl)Br"][..],
        &["S[Co@OH5](F)(I)(Cl)(C)Br", "Br[Co@OH9](C)(S)(Cl)(F)I"][..],
        &["Br[Co@OH12](Cl)(I)(F)(S)C", "Cl[Co@OH15](C)(Br)(F)(I)S"][..],
        &["Cl[Co@OH19](C)(I)(F)(S)Br", "I[Co@OH27](Cl)(Br)(F)(S)C"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_converges_square_planar_implicit_hydrogen_spellings() {
    let groups = [
        &["C[Pt@SP1H](Cl)F", "C[Pt@SP1]([H])(Cl)F"][..],
        &["C[Pt@SP1H2]Cl", "C[Pt@SP1]([H])([H])Cl"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_converges_disconnected_ring_closure_stereo_variants() {
    let groups = [
        &["[C@@]1(Cl)(F)(I).Br1", "[C@@](Br)(Cl)(F)(I)"][..],
        &["[C@@](Cl)(F)(I)1.Br1", "[C@@](Cl)(F)(I)Br"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_preserves_semantic_stereo_distinctions() {
    assert_ne!(canonical_string("F/C=C/F"), canonical_string("F/C=C\\F"));
    assert_ne!(canonical_string("F/C=C/F"), canonical_string("FC=CF"));
    assert_ne!(canonical_string("N[C@](Br)(O)C"), canonical_string("N[C@@](Br)(O)C"));
    assert_ne!(canonical_string("OC(Cl)=[C@]=C(C)F"), canonical_string("OC(Cl)=[C@@]=C(C)F"));
}
