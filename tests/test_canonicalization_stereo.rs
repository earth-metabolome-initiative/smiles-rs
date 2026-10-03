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
fn canonicalize_converges_ring_alkene_stereo_equivalence_groups() {
    let trans = ["C1CCCCC/C=C/1", r"C\1CCCCC/C=C1", "C1=C/CCCCCC/1", r"C1CCCCC\C=C\1"];
    let cis =
        [r"C1CCCCC/C=C\1", r"C/1CCCCC/C=C1", r"C1=C\CCCCCC/1", r"C1CCCCC\C=C/1", "C/1=C/CCCCCC1"];
    assert_same_canonical_group(&trans);
    assert_same_canonical_group(&cis);
    assert_ne!(canonical_string(trans[0]), canonical_string(cis[0]));
    assert_ne!(canonical_string(trans[0]), canonical_string("C1CCCCCC=C1"));
    assert_same_canonical_group(&[
        "C1CCC2CCCC/C=C/C2C1",
        r"C1CCC2CCCC\C=C\C2C1",
        "C1=C/C2CCCCC2CCCC/1",
    ]);
}

#[test]
fn canonicalize_converges_macrocyclic_natural_product_spellings() {
    let caryophyllene = [
        r"C/C/1=C\CCC(=C)[C@H]2CC([C@@H]2CC1)(C)C",
        r"[C@@H]12C(=C)CC/C=C(\C)CC[C@H]1C(C)(C2)C",
        r"C=C1CC/C=C(/CC[C@@H]2[C@@H]1CC2(C)C)C",
        r"C1(/C)CC[C@@H]2[C@@H](C(=C)CC\C=1)CC2(C)C",
    ];
    assert_same_canonical_group(&caryophyllene);
    assert_ne!(
        canonical_string(caryophyllene[0]),
        canonical_string(r"C/C/1=C/CCC(=C)[C@H]2CC([C@@H]2CC1)(C)C"),
    );
    assert_same_canonical_group(&[
        r"C/C/1=C\CC(/C=C/C/C(=C/CC1)/C)(C)C",
        r"CC1(C)C/C=C(\C)/CC/C=C(\C)/C/C=C/1",
    ]);
    assert_same_canonical_group(&[r"C1/C=C/CC/C=C\CC/C=C/C1", r"C1C/C=C/CC/C=C/CC/C=C\1"]);
}

#[test]
fn canonicalize_reads_no_alkene_stereo_from_atom_marks_on_cumulated_double_bonds() {
    for input in [r"F[C@@H]=[C@](F)=O", r"F[C@H]=[C@](F)=O", r"O=[C@](F)=[C@@H]F"] {
        let canonical = canonical_string(input);
        assert!(!canonical.contains(['/', '\\']), "{input} canonicalized to {canonical}");
    }
}

#[test]
fn canonicalize_keeps_cumulene_stereo_next_to_non_stereogenic_alkene() {
    for (trans, cis) in [
        (r"F/C=C=C=C/C(C)=C(C)C", r"F/C=C=C=C\C(C)=C(C)C"),
        (r"F/C=C=C=C/C=C(C)C", r"F/C=C=C=C\C=C(C)C"),
    ] {
        assert_ne!(canonical_string(trans), canonical_string(cis), "{trans} and {cis}");
    }
}

#[test]
fn canonicalize_converges_atom_based_alkene_stereo_equivalence_groups() {
    let groups = [
        &[
            "F[C@@H]=[C@H]F",
            "F[C@H]=[C@@H]F",
            "F/C=C/F",
            "[C@H](F)=[C@H]F",
            "F1.F[C@@H]=[C@H]1",
            "F1.[C@H](F)=[C@H]1",
        ][..],
        &[
            "F[C@H]=[C@H]F",
            "F[C@@H]=[C@@H]F",
            "F/C=C\\F",
            "[C@@H](F)=[C@H]F",
            "F1.F[C@H]=[C@H]1",
            "F1.[C@@H](F)=[C@H]1",
        ][..],
        &["C1CCCCC[C@@H]=[C@H]1", "C1CCCCC[C@H]=[C@@H]1", "C1CCCCC/C=C/1"][..],
        &["C1CCCCC[C@H]=[C@H]1", "C1CCCCC[C@@H]=[C@@H]1", r"C1CCCCC/C=C\1"][..],
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
