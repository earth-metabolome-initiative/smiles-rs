//! Conversion tests from parsed SMILES graphs into molecular formulas.

use std::io::{BufRead, BufReader, Cursor, Read};

use elements_rs::Element;
use flate2::read::GzDecoder;
use molecular_formulas::{ChargedMolecularFormula, MolecularFormula, prelude::ChemicalFormula};
use smiles_rs::prelude::{Smiles, WildcardMolecularFormulaConversionError, WildcardSmiles};

const RDKIT_MOLECULAR_FORMULA_FIXTURE: &[u8] =
    include_bytes!("fixtures/rdkit_molecular_formula.csv.gz");
const RDKIT_MOLECULAR_FORMULA_HEADER: &str =
    "name,smiles,rdkit_formula,rdkit_fragment_formula,formal_charge";
type TestFormula = ChemicalFormula<u32, i32>;

#[test]
fn methane_smiles_converts_to_formula_with_implicit_hydrogens() {
    let smiles = Smiles::from_str("C").expect("methane skeleton should parse");
    let formula: TestFormula = ChemicalFormula::from(smiles);

    assert_eq!(formula.to_string(), "CH₄");
    assert_eq!(formula.count_of_element::<u32>(Element::C), Ok(1));
    assert_eq!(formula.count_of_element::<u32>(Element::H), Ok(4));
}

#[test]
fn charged_isotopic_smiles_converts_to_formula() {
    let smiles = Smiles::from_str("[13CH3][NH3+]").expect("charged isotopic SMILES should parse");
    let formula: TestFormula = ChemicalFormula::from(&smiles);

    assert_eq!(formula.to_string(), "[¹³C]H₆N⁺");
    assert_eq!(formula.count_of_element::<u32>(Element::C), Ok(1));
    assert_eq!(formula.count_of_element::<u32>(Element::H), Ok(6));
    assert_eq!(formula.count_of_element::<u32>(Element::N), Ok(1));
    assert!((formula.charge() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn disconnected_smiles_converts_to_formula_mixture() {
    let smiles = Smiles::from_str("[Na+].[Cl-]").expect("salt SMILES should parse");
    let formula: TestFormula = ChemicalFormula::from(smiles);

    assert_eq!(formula.to_string(), "Na⁺.Cl⁻");
    assert!(formula.charge().abs() < f64::EPSILON);
}

#[test]
fn wildcard_smiles_returns_formula_conversion_error() {
    let smiles = WildcardSmiles::from_str("*").expect("wildcard SMILES should parse");
    let error = TestFormula::try_from(smiles).expect_err("wildcard cannot produce a formula");

    assert_eq!(error, WildcardMolecularFormulaConversionError::WildcardAtom { atom_id: 0 });
}

#[test]
fn rdkit_molecular_formula_fixture_matches_conversion() {
    let stats = assert_rdkit_molecular_formula_fixture_matches(Cursor::new(
        RDKIT_MOLECULAR_FORMULA_FIXTURE,
    ));

    assert_eq!(stats.total, 10_036);
    assert_eq!(stats.wildcards, 0);
}

#[derive(Debug, Default)]
struct RdkitMolecularFormulaFixtureStats {
    total: usize,
    formulas: usize,
    wildcards: usize,
}

fn assert_rdkit_molecular_formula_fixture_matches<R: Read>(
    reader: R,
) -> RdkitMolecularFormulaFixtureStats {
    let decoder = GzDecoder::new(reader);
    let mut lines = BufReader::new(decoder).lines();

    assert_eq!(
        lines.next().transpose().expect("fixture header should be readable").as_deref(),
        Some(RDKIT_MOLECULAR_FORMULA_HEADER)
    );

    let mut stats = RdkitMolecularFormulaFixtureStats::default();
    for (line_index, line) in lines.enumerate() {
        let line =
            line.unwrap_or_else(|error| panic!("fixture line {} failed: {error}", line_index + 2));
        stats.total += 1;
        let columns = line.split(',').collect::<Vec<_>>();
        assert_eq!(
            columns.len(),
            5,
            "fixture line {} has the wrong column count: {line}",
            line_index + 2
        );

        let name = columns[0];
        let smiles_text = columns[1];
        let rdkit_formula = columns[2];
        let rdkit_fragment_formula = columns[3];
        let formal_charge = columns[4]
            .parse::<i32>()
            .unwrap_or_else(|error| panic!("{name} has invalid formal_charge: {error}"));

        let smiles = Smiles::from_str(smiles_text)
            .unwrap_or_else(|error| panic!("{name} failed to parse {smiles_text}: {error}"));
        if rdkit_fragment_formula.contains('*') {
            stats.wildcards += 1;
            print_rdkit_fixture_progress(&stats);
            continue;
        }

        let actual: TestFormula = ChemicalFormula::from(&smiles);
        let expected = TestFormula::try_from(rdkit_fragment_formula).unwrap_or_else(|error| {
            panic!("{name} has invalid RDKit formula {rdkit_fragment_formula}: {error}")
        });
        stats.formulas += 1;

        assert_eq!(
            actual, expected,
            "{name} formula mismatch for {smiles_text}: actual={actual}, expected={expected}, RDKit aggregate={rdkit_formula}",
        );
        assert!(
            (actual.charge() - f64::from(formal_charge)).abs() < f64::EPSILON,
            "{name} charge mismatch for {smiles_text}: actual={}, expected={formal_charge}",
            actual.charge()
        );

        print_rdkit_fixture_progress(&stats);
    }

    stats
}

fn print_rdkit_fixture_progress(stats: &RdkitMolecularFormulaFixtureStats) {
    if stats.total.is_multiple_of(1_000_000) {
        eprintln!(
            "processed {} RDKit rows: {} molecular formulas, {} wildcards",
            stats.total, stats.formulas, stats.wildcards
        );
    }
}
