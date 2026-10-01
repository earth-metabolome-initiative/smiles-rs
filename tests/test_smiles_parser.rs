//! Tests of the parser module for several corner cases.

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
];

#[test]
fn test_parser_all_inputs() {
    for &s in SMILES_STR {
        Smiles::from_str(s).unwrap_or_else(|e| panic!("Failed to parse:\n{}", e.render(s)));
    }
}

#[test]
fn test_parse_benzene_graph_shape() {
    let line = SMILES_STR[0];
    let smiles =
        Smiles::from_str(line).unwrap_or_else(|e| panic!("Failed to parse:\n{}", e.render(line)));

    assert_eq!(smiles.nodes().len(), 6);
    assert_eq!(smiles.number_of_bonds(), 6);
    assert!(smiles.nodes().iter().all(|n| !n.aromatic()));
}

#[test]
fn test_parse_benzene_with_wildcard_graph_shape() {
    let line = "c1ccccc1*";
    let smiles = WildcardSmiles::from_str(line)
        .unwrap_or_else(|e| panic!("Failed to parse:\n{}", e.render(line)));

    assert_eq!(smiles.nodes().len(), 7);
    assert_eq!(smiles.number_of_bonds(), 7);

    let aromatic_count = smiles.nodes().iter().filter(|n| n.aromatic()).count();
    assert_eq!(aromatic_count, 6);
}

#[test]
fn test_parse_tribenzo_annulene_variant() {
    let smiles_string = r"c1=c\c2ccc(cc2)-c2ccccc2-c2ccc/1cc2";
    let smiles = Smiles::from_str(smiles_string)
        .unwrap_or_else(|e| panic!("Failed to parse:\n{}", e.render(smiles_string)));

    assert_eq!(smiles.nodes().len(), 20);
    assert_eq!(smiles.number_of_bonds(), 23);

    assert!(has_edge(&smiles, 0, 1, Bond::Double));
    assert!(has_edge(&smiles, 1, 2, Bond::Down));
    assert!(has_edge(&smiles, 17, 0, Bond::Up));

    assert!(has_edge(&smiles, 5, 8, Bond::Single));
    assert!(has_edge(&smiles, 13, 14, Bond::Single));

    let degrees: Vec<usize> =
        (0..smiles.nodes().len()).map(|node_id| smiles.edge_count_for_node(node_id)).collect();

    assert_eq!(degrees.iter().filter(|&&d| d == 3).count(), 6);
    assert_eq!(degrees.iter().filter(|&&d| d == 2).count(), 14);
}

fn has_edge(smiles: &Smiles, a: usize, b: usize, bond: Bond) -> bool {
    smiles.edge_for_node_pair((a, b)).is_some_and(|edge| edge.bond() == bond)
}

/// Adversarial input: a single carbon with 300 single-bond branches.
///
/// Before the explicit-valence widening, this panicked during parse via a
/// debug-mode `Sum<u8>` overflow inside `explicit_valence`. After the fix,
/// the input must parse and the downstream valence helpers must saturate
/// instead of panicking.
#[test]
fn test_high_degree_single_bond_carbon_does_not_panic() {
    let mut input = String::from("C");
    for _ in 0..300 {
        input.push_str("(C)");
    }

    let smiles =
        Smiles::from_str(&input).expect("300-branch carbon must parse, not panic, after widening");

    assert_eq!(smiles.edge_count_for_node(0), 300);
    assert_eq!(smiles.total_valence(0), u8::MAX);
    assert_eq!(smiles.connectivity_count(0), u8::MAX);
}

/// Adversarial input: a single carbon with many quadruple-bond branches.
///
/// 64 branches at bond order 4 sum to 256, which overflows `u8` by 1 and
/// previously panicked during parse. After the fix, parse succeeds and
/// `total_valence` saturates at `u8::MAX`.
#[test]
fn test_high_degree_quadruple_bond_carbon_does_not_panic() {
    let mut input = String::from("C");
    for _ in 0..64 {
        input.push_str("($C)");
    }

    let smiles = Smiles::from_str(&input)
        .expect("64 quadruple-bonded neighbors must parse, not panic, after widening");

    assert_eq!(smiles.edge_count_for_node(0), 64);
    assert_eq!(smiles.total_valence(0), u8::MAX);
}
