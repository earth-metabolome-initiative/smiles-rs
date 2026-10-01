//! Radius-N atom environment tests (RDKit FindAtomEnvironmentOfRadiusN).
use smiles_rs::prelude::Smiles;

fn canon(s: &str) -> String {
    s.parse::<Smiles>().unwrap().canonicalize().render()
}

#[test]
fn empty_shell_rule() {
    let ethanol: Smiles = "CCO".parse().unwrap();
    assert!(ethanol.atom_environment(1, 2).is_none()); // central C, eccentricity 1 < 2
    assert!("CC(=O)O".parse::<Smiles>().unwrap().atom_environment(1, 2).is_none()); // carbonyl
    assert!("C".parse::<Smiles>().unwrap().atom_environment(0, 1).is_none()); // methane
    assert!("[NH4+]".parse::<Smiles>().unwrap().atom_environment(0, 1).is_none());
    assert!(ethanol.atom_environment(0, 0).is_none()); // radius 0
}

#[test]
fn terminal_carbon_radius_two() {
    let m: Smiles = "CCO".parse().unwrap();
    let env = m.atom_environment(0, 2).unwrap();
    assert_eq!(env.center(), 0);
    assert_eq!(env.atom_count(), 3);
    assert_eq!(env.bond_count(), 2);
    assert!(env.contains_atom(0) && env.contains_atom(1) && env.contains_atom(2));
    assert_eq!(canon(&env.rooted_smiles(false).unwrap()), canon("CCO"));
}

#[test]
fn central_carbon_radius_one_is_non_empty() {
    let m: Smiles = "CC(=O)O".parse().unwrap();
    let env = m.atom_environment(1, 1).unwrap();
    assert_eq!(env.center(), 1);
    assert_eq!(env.bond_count(), 3);
}

#[test]
fn benzene_radius_two_is_a_path() {
    let m: Smiles = "c1ccccc1".parse().unwrap();
    let env = m.atom_environment(0, 2).unwrap();
    assert_eq!(env.atom_count(), 5);
    assert_eq!(env.bond_count(), 4);
}

#[test]
fn benzene_radius_three_closes_the_ring() {
    let m: Smiles = "c1ccccc1".parse().unwrap();
    let env = m.atom_environment(0, 3).unwrap();
    assert_eq!(env.atom_count(), 6);
    assert_eq!(env.bond_count(), 6); // ring-closure bond included
    assert_eq!(canon(&env.rooted_smiles(false).unwrap()), canon("c1ccccc1"));
}

#[test]
fn to_fragment_is_centered() {
    let m: Smiles = "CCO".parse().unwrap();
    let env = m.atom_environment(0, 2).unwrap();
    let frag = env.to_fragment().unwrap();
    assert_eq!(frag.atom_count(), 3);
    assert_eq!(frag.local_id(0), Some(0));
}

#[test]
fn iterators_agree_with_counts() {
    let m: Smiles = "c1ccccc1".parse().unwrap();
    let env = m.atom_environment(0, 2).unwrap();
    assert_eq!(env.atoms().count(), env.atom_count());
    assert_eq!(env.bonds().count(), env.bond_count());
}

#[test]
fn small_ring_radius_two_closes_via_frontier_bond() {
    // Cyclopropane: at radius 2 the two radius-1 neighbors close the ring.
    let m: Smiles = "C1CC1".parse().unwrap();
    let env = m.atom_environment(0, 2).unwrap();
    assert_eq!(env.atom_count(), 3);
    assert_eq!(env.bond_count(), 3);
}

#[test]
fn radius_beyond_eccentricity_is_empty() {
    // Cyclopropane has eccentricity 1, so radius 3 exhausts the frontier early.
    let m: Smiles = "C1CC1".parse().unwrap();
    assert!(m.atom_environment(0, 3).is_none());
}

#[test]
fn toluene_labels_match_rdkit_reference() {
    // Semantic parity with the RDKit MAP4 reference strings from the proposal.
    let m: Smiles = "Cc1ccccc1".parse().unwrap();
    assert_eq!(m.rooted_environment_smiles(0, 1, false).unwrap(), "Cc");
    assert_eq!(m.rooted_environment_smiles(1, 2, false).unwrap(), "c(C)(cc)cc");
}

#[test]
#[should_panic(expected = "center")]
fn panics_on_out_of_range_center() {
    let _ = "CCO".parse::<Smiles>().unwrap().atom_environment(99, 1);
}
