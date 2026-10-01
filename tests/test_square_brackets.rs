//! Test for tokenizing square brackets

use elements_rs::Element;
use smiles_rs::{errors::SmilesError, smiles::Smiles};

/// const for testing square brackets
const SMILES_WITH_BRACKETS: &[&str] = &[
    "[OH2]",
    "[OH3+]",
    "[Ti+4]",
    "[Co+3]",
    "[Ga+]$[As-]",
    "[te]1cccc1",
    "[Na+].[Cl-]",
    "N[C@@H](C)C(=O)O",
    "N[CH](C)C(=O)O",
    "[14cH]1ccccc1",
    "[14c@H]1ccccc1",
    "[2H]C(Cl)(Cl)Cl",
    "[HH]",
    "[HH-]",
    "[3HH]",
    "[C@@H](C)(N)C(=O)O",
    "C[C@H](N)C(=O)O",
    "OC(=O)[C@@H](N)C",
    "[K+].C=C.Cl[Pt-](Cl)Cl.O",
    "CCN1C[C@]2(COC)CC[C@H](O)[C@@]34[C@@H]5C[C@H]6[C@H](OC)[C@@H]5[C@](O)(C[C@@H]6OC)[C@@](O)([C@@H](OC)[C@H]23)[C@@H]14",
    "C[C@@H]1C[C@@]2(O[C@H]2C)C(=O)O[C@@H]2CCN(C)C/C=C(/COC(=O)[C@]1(C)O)C2=O",
    "CC1=C[C@H](O)CC(C)(C)[C@H]1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(\\C)CO",
];
#[test]
fn test_valid_square_brackets() {
    for &s in SMILES_WITH_BRACKETS {
        let _smiles = Smiles::from_str(s).unwrap_or_else(|_| panic!("Failed to parse {s}"));
    }
}

#[test]
fn test_unexpected_right_brackets() {
    // stray ']' becomes UnexpectedCharacter(']') with a span
    let err = Smiles::from_str("[Co+3]]").unwrap_err();

    assert_eq!(err.smiles_error(), SmilesError::UnexpectedCharacter(']'));
    assert_eq!(err.start(), 6);
    assert_eq!(err.end(), 7);
}

#[test]
fn test_unexpected_left_brackets() {
    let err = Smiles::from_str("[[Co+3]").unwrap_err();

    assert_eq!(err.smiles_error(), SmilesError::MissingElement);
    assert_eq!(err.start(), 0);
    assert_eq!(err.end(), 2);
}

#[test]
fn test_unclosed_brackets() {
    let err = Smiles::from_str("[Co+3").unwrap_err();

    assert_eq!(err.smiles_error(), SmilesError::UnclosedBracket);
    assert_eq!(err.start(), 0);
    assert_eq!(err.end(), 5);
}

#[test]
fn test_smiles_tokens_water() {
    let water_line = SMILES_WITH_BRACKETS[0];
    let smiles =
        Smiles::from_str(water_line).unwrap_or_else(|_| panic!("Failed to parse {water_line}"));

    assert_eq!(smiles.nodes().len(), 1);

    let atom = &smiles.nodes()[0];
    assert_eq!(atom.element(), Some(elements_rs::Element::O));
    assert!(!atom.aromatic());
    assert_eq!(atom.hydrogen_count(), 2);
    assert_eq!(atom.charge_value(), 0);
    assert_eq!(atom.class(), 0);
    assert_eq!(atom.chirality(), None);
    assert_eq!(atom.isotope_mass_number(), None);
}

#[test]
fn test_bracket_aromatic_tellurium_parses() {
    let smiles =
        Smiles::from_str("[te]1cccc1").unwrap_or_else(|_| panic!("Failed to parse [te]1cccc1"));

    assert_eq!(smiles.nodes()[0].element(), Some(Element::Te));
    assert!(smiles.nodes()[0].aromatic());
    assert!(smiles.nodes()[0].is_bracket_atom());
}

#[test]
fn test_aromatic_tellurium_requires_brackets() {
    let err = Smiles::from_str("te1cccc1").unwrap_err();
    assert_eq!(err.smiles_error(), SmilesError::InvalidAromaticElement(Element::Te));
}

#[test]
fn test_hydrogen_hcount_compatibility_cases() {
    for source in ["[HH]", "[HH1]", "[HH-]", "[3HH]"] {
        let smiles =
            Smiles::from_str(source).unwrap_or_else(|_| panic!("Failed to parse {source}"));
        assert_eq!(smiles.nodes().len(), 1);
        assert_eq!(smiles.nodes()[0].element(), Some(elements_rs::Element::H));
        assert_eq!(smiles.nodes()[0].hydrogen_count(), 1);
    }
}

#[test]
fn test_hydrogen_hcount_gt_one_stays_invalid() {
    let err = Smiles::from_str("[HH2]").unwrap_err();
    assert_eq!(err.smiles_error(), SmilesError::InvalidHydrogenWithExplicitHydrogensFound);
}

#[test]
fn test_bracket_hydrogen_count_at_cap_parses() {
    let smiles = Smiles::from_str("[CH15]").unwrap_or_else(|_| panic!("Failed to parse [CH15]"));

    assert_eq!(smiles.nodes().len(), 1);
    assert_eq!(smiles.nodes()[0].hydrogen_count(), 15);
}

#[test]
fn test_bracket_hydrogen_count_just_over_cap_is_rejected() {
    let err = Smiles::from_str("[CH16]").unwrap_err();

    assert_eq!(err.smiles_error(), SmilesError::HydrogenCountOverflow(16));
}

#[test]
fn test_bracket_hydrogen_count_far_over_cap_is_rejected() {
    let err = Smiles::from_str("[HoH254]").unwrap_err();

    assert_eq!(err.smiles_error(), SmilesError::HydrogenCountOverflow(254));
}

/// Regression test for the bug where bracket atoms with extreme hydrogen
/// counts parsed successfully and then panicked inside `total_valence` when
/// `explicit_valence + hydrogen_count + implicit_hydrogen_count` exceeded
/// the `u8` ceiling.
#[test]
fn test_extreme_hydrogen_count_does_not_panic_downstream() {
    let result = Smiles::from_str("S85II5OINS8[HoH254]9NN9NCC");

    let err = result.expect_err("input with [HoH254] must not parse successfully");
    assert_eq!(err.smiles_error(), SmilesError::HydrogenCountOverflow(254));
}
