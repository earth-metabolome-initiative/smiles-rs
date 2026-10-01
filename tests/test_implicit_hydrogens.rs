//! Fixture-backed TDD scaffold for implicit hydrogen counting.
//!
//! The immediate target is the local SMILES rule set:
//! - unbracketed organic-subset atoms
//! - bracket atoms defaulting to zero implicit hydrogens
//! - explicit bracket hydrogens never double-counted as implicit
//!
//! Aromatic cases are included separately because they need an aromatic policy
//! layer in addition to the raw valence ranges from `elements-rs`.
//!
//! The fixtures are intentionally split into:
//! - `OpenSMILES` / Daylight local-rule cases
//! - raw-RDKit-compatibility cases for unsanitized SMILES input
//! - aromatic token cases that are still local, but not purely periodic-table
//!   lookups

mod implicit_hydrogen_common;

use implicit_hydrogen_common::{
    AROMATIC_CASES, ImplicitHydrogenCase, ORGANIC_SUBSET_CASES, RAW_RDKIT_COMPAT_CASES, all_cases,
};
use smiles_rs::smiles::{Smiles, WildcardSmiles};

#[test]
fn implicit_hydrogen_fixtures_match_current_parser_node_order() {
    for case in all_cases() {
        let actual_labels = parsed_atom_labels(case);
        let expected_labels = case.atoms.iter().map(|atom| atom.atom).collect::<Vec<_>>();

        assert_eq!(
            actual_labels.len(),
            case.atoms.len(),
            "fixture {} has the wrong atom count for {}\nnote: {}",
            case.name,
            case.smiles,
            case.note
        );

        assert_eq!(
            actual_labels, expected_labels,
            "fixture {} does not match the parser node order for {}\nnote: {}",
            case.name, case.smiles, case.note
        );
    }
}

#[test]
fn organic_subset_implicit_hydrogens_match_fixture() {
    assert_cases_match_expected(ORGANIC_SUBSET_CASES);
}

#[test]
fn raw_rdkit_compat_implicit_hydrogens_match_fixture() {
    assert_cases_match_expected(RAW_RDKIT_COMPAT_CASES);
}

#[test]
fn aromatic_implicit_hydrogens_match_fixture() {
    assert_cases_match_expected(AROMATIC_CASES);
}

fn assert_cases_match_expected(cases: &[ImplicitHydrogenCase]) {
    for case in cases {
        let actual = parsed_implicit_hydrogen_counts(case);
        let expected = case.atoms.iter().map(|atom| atom.implicit_hydrogens).collect::<Vec<_>>();

        assert_eq!(
            actual,
            expected.as_slice(),
            "implicit hydrogen mismatch for fixture {} ({})\nnote: {}",
            case.name,
            case.smiles,
            case.note
        );
    }
}

fn parsed_atom_labels(case: &ImplicitHydrogenCase) -> Vec<String> {
    if case.smiles.contains('*') {
        let smiles = WildcardSmiles::from_str(case.smiles).unwrap_or_else(|err| {
            panic!("fixture {} failed to parse:\n{}", case.name, err.render(case.smiles))
        });
        smiles.nodes().iter().map(ToString::to_string).collect()
    } else {
        let smiles = Smiles::from_str(case.smiles).unwrap_or_else(|err| {
            panic!("fixture {} failed to parse:\n{}", case.name, err.render(case.smiles))
        });
        smiles.nodes().iter().map(ToString::to_string).collect()
    }
}

fn parsed_implicit_hydrogen_counts(case: &ImplicitHydrogenCase) -> Vec<u8> {
    if case.smiles.contains('*') {
        let smiles = WildcardSmiles::from_str(case.smiles).unwrap_or_else(|err| {
            panic!("fixture {} failed to parse:\n{}", case.name, err.render(case.smiles))
        });
        smiles.implicit_hydrogen_counts().to_vec()
    } else {
        let smiles = Smiles::from_str(case.smiles).unwrap_or_else(|err| {
            panic!("fixture {} failed to parse:\n{}", case.name, err.render(case.smiles))
        });
        smiles.implicit_hydrogen_counts().to_vec()
    }
}
