use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrixRef;

use super::super::{
    Smiles, canonicalization_state_key,
    support::{assert_canonicalization_invariants, permute_smiles, same_canonicalization_state},
};
use crate::{bond::Bond, parser::smiles_parser::parse_wildcard_smiles, smiles::WildcardAtoms};

fn wildcard_smiles(source: &str) -> Smiles<WildcardAtoms> {
    parse_wildcard_smiles(source).unwrap()
}

fn hidden_bond_order_digest(smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>) -> u64 {
    use core::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    smiles
        .bond_matrix()
        .sparse_entries()
        .filter(|((row, column), _)| row < column)
        .for_each(|((row, column), entry)| (row, column, entry.order()).hash(&mut hasher));
    hasher.finish()
}

fn bond_count(smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>, bond: Bond) -> usize {
    smiles
        .bond_matrix()
        .sparse_entries()
        .filter(|((row, column), entry)| row < column && entry.bond() == bond)
        .count()
}

#[test]
fn canonicalize_handles_wildcard_quadruple_bond_regression() {
    let smiles = wildcard_smiles("*$*");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn double_bond_stereo_invariant_survives_component_reordering_regression() {
    let smiles = wildcard_smiles("*[*@]*.F/C=*/*");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn kekulization_invariant_holds_for_four_membered_aromatic_rings_regression() {
    for source in ["Co1ccc1O", "ccc1ccc1cc"] {
        assert_canonicalization_invariants(&Smiles::from_str(source).unwrap());
    }
}

#[test]
fn aromaticity_invariant_holds_for_radical_bracket_aromatic_atom_regression() {
    let smiles = Smiles::from_str("cc1ncN[n]1").unwrap();

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_preserves_aromatic_triple_bond_order() {
    let smiles = Smiles::from_str("C1=CC#CC=C1").unwrap();
    let canonicalized = smiles.canonicalize();

    assert_eq!(bond_count(&canonicalized, Bond::Triple), 1);
    assert_canonicalization_invariants(&canonicalized);
}

#[test]
fn canonicalize_handles_aromatic_sulfur_chain_regression() {
    let smiles = Smiles::from_str("scCC").unwrap();

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_mixed_aromatic_phosphorus_wildcard_regression() {
    let smiles = wildcard_smiles("bsC3P1*sC3P1**p1C3P1**OC3p1C3Os*p1C3P1**OC3p1C3Os");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_polywildcard_aromatic_oxygen_regression() {
    let smiles = wildcard_smiles("OO*c8Oc8OOOO**c8OOO*c8OOOO*CO*c8OOOO*c8OOOO**c8OOO*c8OOOO*COOOO");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_aromatic_wildcard_triangle_regression() {
    let smiles = wildcard_smiles("P1*P1");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_fuzz_crash_dfd86928_regression() {
    let smiles =
        wildcard_smiles("CBBBBBBBBBBBBBBBBBBBBBBB.OBBBBBBBBBBBBBBBBBBBB.OBBP1*P1**BBBBBBB");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_fuzz_crash_partial_aromatic_diagnostics_regression() {
    let smiles = wildcard_smiles("bsp1C2P1**C3o1**p1*CPp*O14C3Po1**p1C2P1**C3o1$*p1C4P1**Op1C3Os");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_fuzz_crash_component_order_budget_regression() {
    let smiles = wildcard_smiles(
        "Oooooooooooooooooooo.****.*ccOc**S*nnbsC3P1*soooooooooo.****.*ccOc**S*nnbsC3P1*sC3P1**p1C3P1**OC3p1C3Os*p1C3P2***S*nnnCnsnC3P1**p1C3P1**OC3p1C3Os*p1C3P2***S*nnnCns",
    );

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_fuzz_crash_new_regression_candidate() {
    let smiles = wildcard_smiles("***SCbs=3P1o*sC3P1**p1C3P1**OC3P1*=*OC3p1C3Os");
    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_fuzz_crash_db6568a7_regression() {
    let smiles = wildcard_smiles(
        "*[*@]*.**.******[*@]*.**.**.*.**.**.******.**.****.*.*.**.**.***/**.**.****.**.***.**.**.*.**.**.******N**.****.**.**.**",
    );

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonical_convergence_for_reversed_orphan_directional_fragment() {
    let smiles = wildcard_smiles("C/F\\Cl");

    assert_canonicalization_invariants(&smiles);

    let canonicalized = smiles.canonicalize();
    assert_eq!(
        bond_count(&canonicalized, Bond::Up) + bond_count(&canonicalized, Bond::Down),
        2,
        "orphan directional bonds must survive canonicalization"
    );

    let node_count = smiles.nodes().len();
    let reversed: Vec<usize> = (0..node_count).rev().collect();
    let permuted = permute_smiles(&smiles, &reversed);
    same_canonicalization_state(&canonicalized, &permuted.canonicalize());
}

#[test]
fn canonicalize_handles_fuzz_crash_hex_wildcard_oxygen_regression() {
    let smiles = wildcard_smiles("******#8OOO*c8");

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_fuzz_crash_aromatic_boron_hydrogen_regression() {
    let smiles = Smiles::from_str("[H]bbbbbbbbbb").unwrap();

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_handles_fuzz_crash_square_planar_platinum_regression() {
    let smiles = Smiles::from_str("[NH3][Pt@SP1]([NH3])(Cl)Cl").unwrap();

    assert_canonicalization_invariants(&smiles);
}

#[test]
fn canonicalize_reaches_fixed_point_for_pubchem_cid_10888631_regression() {
    let smiles =
        Smiles::from_str("C1C2[C@@H]3[C@H]4[C@H]5[C@H]6C5CC4[C@@H]6[C@H]2[C@@H]7[C@@H]3C71")
            .unwrap();

    let canonicalized = smiles.canonicalize();
    assert!(canonicalized.is_canonical(), "canonicalize() did not reach a canonical fixed point");

    let recanonicalized = canonicalized.canonicalize();
    same_canonicalization_state(&canonicalized, &recanonicalized);
}

#[test]
fn canonicalize_reaches_fixed_point_for_pubchem_cid_101510359_regression() {
    let smiles = Smiles::from_str("CC(C)(C)C1=CC(=CC(=C1)C2=C3C=CC(=C(C4=NC5=CC6=C(C=C(N6)C(=C7C=CC2=N7)C8=CC(=CC(=C8)C(C)(C)C)C(C)(C)C)C9=C1C=CC(=C(C2=NC(=C(C6=CC=C(N6)C(=C6C=CC9=N6)C6=CC(=CC(=C6)C(C)(C)C)C(C)(C)C)C6=CC(=CC(=C6)C(C)(C)C)C(C)(C)C)C=C2)C2=C6C=C7C(=CC(=N7)C(=C7C=CC(=C(C8=NC(=C(C(=C2)N6)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C=C8)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)N7)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C2=C6C=CC(=C(C5=C4)C4=NC(=C(C5=CC=C(N5)C(=C5C=CC2=N5)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C=C4)N6)N1)C1=CC(=CC(=C1)C(C)(C)C)C(C)(C)C)N3)C(C)(C)C").unwrap();

    let canonicalized = smiles.canonicalize();
    let recanonicalized = canonicalized.canonicalize();

    same_canonicalization_state(&canonicalized, &recanonicalized);
}

#[test]
fn canonicalization_spelling_normal_form_preserves_semantic_implicit_hydrogens() {
    let aromaticized = Smiles::from_str("C1=CC=CC=C1")
        .unwrap()
        .perceive_aromaticity()
        .unwrap()
        .into_aromaticized();
    let rewritten = aromaticized.canonicalization_spelling_normal_form();

    assert_eq!(rewritten.implicit_hydrogen_counts(), aromaticized.implicit_hydrogen_counts());
    assert_eq!(rewritten.implicit_hydrogen_cache, aromaticized.implicit_hydrogen_counts());
}

#[test]
fn canonicalization_step_does_not_hide_bond_order_drift_for_pubchem_cid_101510359() {
    let smiles = Smiles::from_str("CC(C)(C)C1=CC(=CC(=C1)C2=C3C=CC(=C(C4=NC5=CC6=C(C=C(N6)C(=C7C=CC2=N7)C8=CC(=CC(=C8)C(C)(C)C)C(C)(C)C)C9=C1C=CC(=C(C2=NC(=C(C6=CC=C(N6)C(=C6C=CC9=N6)C6=CC(=CC(=C6)C(C)(C)C)C(C)(C)C)C6=CC(=CC(=C6)C(C)(C)C)C(C)(C)C)C=C2)C2=C6C=C7C(=CC(=N7)C(=C7C=CC(=C(C8=NC(=C(C(=C2)N6)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C=C8)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)N7)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C2=C6C=CC(=C(C5=C4)C4=NC(=C(C5=CC=C(N5)C(=C5C=CC2=N5)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C2=CC(=CC(=C2)C(C)(C)C)C(C)(C)C)C=C4)N6)N1)C1=CC(=CC(=C1)C(C)(C)C)C(C)(C)C)N3)C(C)(C)C").unwrap();

    let entry = smiles.canonicalize();
    let direct_step = entry.canonicalization_step();

    assert_eq!(canonicalization_state_key(&entry), canonicalization_state_key(&direct_step));
    assert_eq!(hidden_bond_order_digest(&entry), hidden_bond_order_digest(&direct_step));
}
