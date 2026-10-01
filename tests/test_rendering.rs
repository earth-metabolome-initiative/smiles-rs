//! Rendering-focused tests for SMILES output.

use geometric_traits::traits::SparseValuedMatrixRef;
use smiles_rs::{
    bond::Bond,
    smiles::{Smiles, WildcardSmiles},
};

const SMILES_STR: &[&str] = &[
    "C1=CC=CC=C1",
    "[OH2]",
    "[Ti+4]",
    "[Co+3]",
    "CCO",
    "C#N",
    "[Ga+]$[As-]",
    "[te]1cccc1",
    "[Na+].[Cl-]",
    "C1CCCC2C1CCCC2",
    "C0CCCCC0C0CCCCC0",
    "C1:C:C:C:C:C1",
    "c1ccccc1",
    "c1ccccc1c2ccccc2",
    "COc(c1)cccc1C#N",
    "FC(Br)(Cl)F",
    "CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C",
    "N[C@@H](C)C(=O)O",
    "N[CH](C)C(=O)O",
    "NC(C)C(=O)O",
    "[14cH]1ccccc1",
    "[14c@H]1ccccc1",
    "[2H]C(Cl)(Cl)Cl",
    "[C@@H](C)(N)C(=O)O",
    "C[C@H](N)C(=O)O",
    "OC(=O)[C@@H](N)C",
    "[K+].C=C.Cl[Pt-](Cl)Cl.O",
    "[Ti+4]",
    "CCN1C[C@]2(COC)CC[C@H](O)[C@@]34[C@@H]5C[C@H]6[C@H](OC)[C@@H]5[C@](O)(C[C@@H]6OC)[C@@](O)([C@@H](OC)[C@H]23)[C@@H]14",
    "C[C@@H]1C[C@@]2(O[C@H]2C)C(=O)O[C@@H]2CCN(C)C/C=C(/COC(=O)[C@]1(C)O)C2=O",
    "CC=C(C)C1=C(Cl)C(O)=C(C)C2=C1OC1=CC(O)=C(Cl)C(C)=C1C(=O)O2",
    "CC1=C[C@H](O)CC(C)(C)[C@H]1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(\\C)CO",
    "COC1(C23C14C5=C6C7=C8C5=C9C1=C5C%10=C%11C%12=C%13C%10=C%10C1=C8C1=C%10C8=C%10C%14=C%15C%16=C%17C(=C%12C%12=C%17C%17=C%18C%16=C%16C%15=C%15C%10=C1C7=C%15C1=C%16C(=C%18C7=C2C2=C%10C(=C5C9=C42)C%11=C%12C%10=C%177)C3=C16)C%14=C%138)OC",
    "C1CCCC=1",
    "C1CC2CC3C4C5C6C7C8C9C%10C%11C%12C%13C%14C%15C%16CC%17C%16%16C%15%15C%14%14C%13%13C%12%12C%11%11C%10%10C99C88C77C66C55C44C33C2C2C33C44C55C66C77C88C99C%10%10C%11%11C%12%12C%13%13C%14%14C%15%15C%16%16C%17C%17C%16%16C%15%15C%14%14C%13%13C%12%12C%11%11C%10%10C99C88C77C66C55C44C33C2C2C33C44C55C66C77C88C99C%10%10C%11%11C%12%12C%13%13C%14%14C%15%15C%16%16C%17C%17C%16%16C%15%15C%14%14C%13%13C%12%12C%11%11C%10%10C99C88C77C66C55C44C33C2C2C33C44C55C66C77C88C99C%10%10C%11%11C%12%12C%13%13C%14%14C%15%15C%16%16C%17C%17C%16%16C%15%15C%14%14C%13%13C%12%12C%11%11C%10%10C99C88C77C66C55C44C33C2C(C1)C3C4C5C6C7C8C9C%10C%11C%12C%13C%14C%15C%16C%17",
    "C1[C@H]([C@H]([C@@H](C(O1)(CO)O)O)O)O",
    "C(C(F)(F)I)(CC1=CC=CC=C1N=C=NC2=CC=CC=OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOF)(F)FOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOC2C",
    "CC(=O)OC(CC(=O)[O-])C",
    "CC(=O)OC(CC(=O)[O-])C",
];

fn parse_or_panic(s: &str) -> Smiles {
    Smiles::from_str(s).unwrap_or_else(|e| panic!("Failed to parse:\n{}", e.render(s)))
}

fn bond_count(smiles: &Smiles, bond: Bond) -> usize {
    smiles
        .bond_matrix()
        .sparse_entries()
        .filter(|((row, column), entry)| row < column && entry.bond() == bond)
        .count()
}

fn single_family_count(smiles: &Smiles) -> usize {
    bond_count(smiles, Bond::Single) + bond_count(smiles, Bond::Up) + bond_count(smiles, Bond::Down)
}

fn aromatic_bond_count(smiles: &Smiles) -> usize {
    smiles
        .bond_matrix()
        .sparse_entries()
        .filter(|((row, column), entry)| row < column && entry.aromatic())
        .count()
}

fn assert_render_round_trip_preserves_invariants(original: &str) {
    let parsed = parse_or_panic(original);
    let rendered = parsed.to_string();
    let reparsed = Smiles::from_str(&rendered).unwrap_or_else(|e| {
        panic!(
            "Failed to parse rendered SMILES.\nOriginal:\n{original}\nRendered:\n{rendered}\n{}",
            e.render(&rendered)
        )
    });
    let rerendered = reparsed.to_string();

    assert_eq!(
        rendered, rerendered,
        "Rendered SMILES did not reach a fixed point.\nOriginal:\n{original}\nRendered:\n{rendered}\nRerendered:\n{rerendered}"
    );

    assert_eq!(
        parsed.nodes().len(),
        reparsed.nodes().len(),
        "Node count changed after render round trip.\nOriginal:\n{original}\nRendered:\n{rendered}"
    );

    assert_eq!(
        parsed.number_of_bonds(),
        reparsed.number_of_bonds(),
        "Edge count changed after render round trip.\nOriginal:\n{original}\nRendered:\n{rendered}"
    );

    let parsed_aromatic = parsed.nodes().iter().filter(|n| n.aromatic()).count();
    let reparsed_aromatic = reparsed.nodes().iter().filter(|n| n.aromatic()).count();

    assert_eq!(
        parsed_aromatic, reparsed_aromatic,
        "Aromatic atom count changed after render round trip.\nOriginal:\n{original}\nRendered:\n{rendered}"
    );

    assert_eq!(
        single_family_count(&parsed),
        single_family_count(&reparsed),
        "Single-bond family count changed after render round trip.\nOriginal:\n{original}\nRendered:\n{rendered}"
    );

    for bond in [Bond::Double, Bond::Triple, Bond::Quadruple] {
        assert_eq!(
            bond_count(&parsed, bond),
            bond_count(&reparsed, bond),
            "Bond count for {bond:?} changed after render round trip.\nOriginal:\n{original}\nRendered:\n{rendered}"
        );
    }

    assert_eq!(
        aromatic_bond_count(&parsed),
        aromatic_bond_count(&reparsed),
        "Aromatic-bond count changed after render round trip.\nOriginal:\n{original}\nRendered:\n{rendered}"
    );
}

#[test]
fn test_render_round_trip_all_inputs() {
    for &original in SMILES_STR {
        assert_render_round_trip_preserves_invariants(original);
    }
}

#[test]
fn wildcard_render_round_trip_preserves_invariants() {
    let original = "c1ccccc1*";
    let parsed = WildcardSmiles::from_str(original)
        .unwrap_or_else(|e| panic!("Failed to parse:\n{}", e.render(original)));
    let rendered = parsed.to_string();
    let reparsed = WildcardSmiles::from_str(&rendered).unwrap_or_else(|e| {
        panic!(
            "Failed to parse rendered wildcard SMILES.\nOriginal:\n{original}\nRendered:\n{rendered}\n{}",
            e.render(&rendered)
        )
    });

    assert_eq!(rendered, reparsed.to_string());
    assert_eq!(parsed.nodes().len(), reparsed.nodes().len());
    assert_eq!(parsed.number_of_bonds(), reparsed.number_of_bonds());
}

#[test]
fn benzene_kekule_roundtrip_preserves_bond_inventory() {
    assert_render_round_trip_preserves_invariants("C1=CC=CC=C1");
}
