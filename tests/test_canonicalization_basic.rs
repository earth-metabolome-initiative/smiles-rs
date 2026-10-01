//! Public canonicalization behavior tests for non-stereo cases.

mod canonicalization_common;

use canonicalization_common::{
    assert_same_canonical_group, canonical_string, wildcard_canonical_string,
};
use smiles_rs::smiles::Smiles;

#[test]
fn canonical_labeling_inverse_matches_order() {
    let smiles = Smiles::from_str("NCC(=O)O").unwrap();
    let labeling = smiles.canonical_labeling();

    assert_eq!(labeling.order().len(), smiles.nodes().len());
    let mut seen = labeling.order().to_vec();
    seen.sort_unstable();
    assert_eq!(seen, (0..smiles.nodes().len()).collect::<Vec<_>>());

    for (new_index, old_node) in labeling.order().iter().copied().enumerate() {
        assert_eq!(labeling.new_index_of_old_node()[old_node], new_index);
    }
}

#[test]
fn canonical_labeling_is_deterministic() {
    let smiles = Smiles::from_str("C1CC(C2CC2)CC1").unwrap();
    let first = smiles.canonical_labeling();
    let second = smiles.canonical_labeling();

    assert_eq!(first, second);
}

#[test]
fn canonicalize_is_idempotent_for_linear_case() {
    let smiles = Smiles::from_str("OCC").unwrap();
    let once = smiles.canonicalize();
    let twice = once.canonicalize();

    assert_eq!(once, twice);
    assert!(once.is_canonical());
}

#[test]
fn canonicalize_converges_equivalent_linear_spellings() {
    assert_eq!(canonical_string("CCO"), canonical_string("OCC"));
}

#[test]
fn canonicalize_converges_disconnected_component_permutations() {
    assert_eq!(canonical_string("CC.O"), canonical_string("O.CC"));
}

#[test]
fn canonicalize_marks_result_as_canonical() {
    let original = Smiles::from_str("OCC").unwrap();
    let canonicalized = original.canonicalize();

    assert!(!original.is_canonical());
    assert!(canonicalized.is_canonical());
}

#[test]
fn canonicalize_preserves_isotope_distinction() {
    assert_ne!(canonical_string("CC"), canonical_string("[13CH3]C"));
}

#[test]
fn canonicalize_converges_additional_source_backed_connectivity_groups() {
    let groups = [
        &["OCC", "CCO", "C-C-O", "C(O)C"][..],
        &["C=CCC=CCO", "C=C-C-C=C-C-O", "OCC=CCC=C"][..],
        &["Oc1ccccc1", "c1ccccc1O", "c1(O)ccccc1", "c1(ccccc1)O"][..],
        &["C1=CCCCC1", "C=1CCCCC1", "C1CCCCC=1", "C=1CCCCC=1"][..],
        &["C1CCCCC1C1CCCCC1", "C1CCCCC1C2CCCCC2"][..],
        &["C12CCCCC1CCCC2", "C1CC2CCCCC2CC1"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_converges_standard_form_bracket_and_explicit_hydrogen_groups() {
    let groups = [
        &["CCO", "[CH3]CO", "[CH3][CH2][OH]", "[H]OCC"][..],
        &["CC(=O)O", "[CH3]C(=O)O"][..],
        &["c1ccccc1", "[cH]1[cH][cH][cH][cH][cH]1"][..],
        &["Oc1ccccc1", "[OH][c]1[cH][cH][cH][cH][cH]1", "[H]Oc1ccccc1"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_converges_additional_source_backed_real_world_groups() {
    let groups = [
        &["CC1=C(C=CO1)C(=O)NC2=CC=CC=C2", "Cc1occc1C(=O)Nc2ccccc2"][..],
        &[
            "CC1COC(Cn2cncn2)(c2ccc(Oc3ccc(Cl)cc3)cc2Cl)O1",
            "n1cnn(c1)CC2(OC(C)CO2)c3c(Cl)cc(cc3)Oc4ccc(Cl)cc4",
        ][..],
        &[
            "CC(C)C(Nc1ccc(C(F)(F)F)cc1Cl)C(=O)OC(C#N)c1cccc(Oc2ccccc2)c1",
            "FC(F)(F)c1cc(Cl)c(cc1)NC(C(C)(C))C(=O)OC(C#N)c2cccc(c2)Oc3ccccc3",
        ][..],
        &["NC(Cc1c[nH]c2ccccc12)C(=O)O", "OC(=O)C(N)C-c1c2ccccc2[nH]c1"][..],
        &["c1c(c23)ccc(c34)ccc4ccc2c1", "C1=CC2=C3C(=CC=C4C3=C1C=C4)C=C2"][..],
        &["S[Co@@](F)(Cl)(Br)(I)C=O", "O=C[Co@](F)(Cl)(Br)(I)S"][..],
    ];

    for group in groups {
        assert_same_canonical_group(group);
    }
}

#[test]
fn canonicalize_converges_default_wildcard_parser_spellings() {
    assert_eq!(wildcard_canonical_string("O*O"), wildcard_canonical_string("O[*]O"));
}
