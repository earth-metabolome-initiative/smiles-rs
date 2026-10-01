//! Tests on Elements that should or should not be parsed as aromatic

use elements_rs::Element;
use smiles_rs::{
    atom::{Atom, atom_symbol::AtomSymbol},
    errors::SmilesError,
    smiles::Smiles,
};
const SMILES_STR: &[&str] = &[
    "c1ccccc1",          // benzene
    "n1ccccc1",          // pyridine
    "Cc1ccccc1",         // toluene
    "c1cccc2ccccc12",    // naphthalene
    "n1cc[nH]c1",        // imidazole (bracketed aromatic N)
    "C1CCCCC1",          // cyclohexane
    "C1CCC=CC1",         // cyclohexene
    "C1CCC1",            // cyclobutane
    "CCO",               // ethanol
    "C1CCCC2CCCCC12",    // decalin
    "[nH]1cccc1",        // pyrrole-like aromatic N
    "c1[n+]([O-])cccc1", // nitro-substituted aromatic ring
    "[CH3][CH2][OH]",    // ethanol (explicit atoms)
    "[NH4+]",            // ammonium (no aromaticity)
];

#[test]
fn test_aromatic_benzene_from_tokenization() -> Result<(), SmilesError> {
    let smiles = Smiles::from_str(SMILES_STR[0]).map_err(|e| e.smiles_error())?;

    assert_eq!(smiles.nodes().len(), 6);
    assert_eq!(smiles.number_of_bonds(), 6);
    assert!(smiles.nodes().iter().all(Atom::aromatic));
    assert!(smiles.nodes().iter().all(|atom| atom.symbol() == AtomSymbol::Element(Element::C)));

    Ok(())
}

#[test]
fn test_aromatic_imidazole_from_tokenization() -> Result<(), SmilesError> {
    let smiles = Smiles::from_str(SMILES_STR[4]).map_err(|e| e.smiles_error())?;
    let bracketed_n = Atom::builder()
        .with_symbol(AtomSymbol::Element(Element::N))
        .with_aromatic(true)
        .with_hydrogens(1)
        .build();

    assert_eq!(smiles.nodes().len(), 5);
    assert_eq!(smiles.number_of_bonds(), 5);
    assert_eq!(smiles.nodes()[0], Atom::new_organic_subset(AtomSymbol::Element(Element::N), true));
    assert_eq!(smiles.nodes()[3], bracketed_n);
    assert!(smiles.nodes().iter().all(Atom::aromatic));
    Ok(())
}
