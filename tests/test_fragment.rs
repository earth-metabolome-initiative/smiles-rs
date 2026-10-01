//! Fragment construction tests.
use smiles_rs::{
    bond::{Bond, bond_edge::bond_edge},
    prelude::{Smiles, SubgraphError},
};

#[test]
fn from_atoms_recomputes_hydrogens() {
    let m: Smiles = "CCO".parse().unwrap();
    let f = m.fragment_from_atoms([0, 1]).unwrap();
    assert_eq!(f.atom_count(), 2);
    // The central carbon, now terminal, gains hydrogens: both become methyl.
    assert_eq!(f.smiles().render(), "CC");
}

#[test]
fn from_bonds_uses_endpoints_only() {
    let m: Smiles = "CCO".parse().unwrap();
    let f = m.fragment_from_bonds([bond_edge(0, 1, Bond::Single, None)]).unwrap();
    assert_eq!(f.smiles().render(), "CC");
}

#[test]
fn maps_round_trip() {
    let m: Smiles = "CCO".parse().unwrap();
    let f = m.fragment_from_atoms([2, 1]).unwrap();
    assert_eq!(f.local_id(2), Some(0));
    assert_eq!(f.local_id(1), Some(1));
    assert_eq!(f.local_id(0), None);
    assert_eq!(f.parent_id(0), 2);
    assert_eq!(f.parent_id(1), 1);
}

#[test]
fn render_rooted_anchors_at_parent_atom() {
    let m: Smiles = "CCO".parse().unwrap();
    let f = m.fragment_from_atoms([0, 1, 2]).unwrap();
    assert!(f.render_rooted(2, true).unwrap().starts_with('O'));
}

#[test]
fn render_rooted_non_isomeric_drops_stereo() {
    let m: Smiles = "N[C@@H](C)C(=O)O".parse().unwrap();
    let f = m.fragment_from_atoms(0..m.nodes().len()).unwrap();
    assert!(!f.render_rooted(1, false).unwrap().contains('@'));
}

#[test]
fn benzene_fragment_round_trips() {
    let m: Smiles = "c1ccccc1".parse().unwrap();
    let f = m.fragment_from_atoms(0..6).unwrap();
    assert_eq!(f.smiles().canonicalize().render(), m.canonicalize().render());
}

#[test]
fn equivalent_environments_share_label() {
    let m: Smiles = "c1ccccc1".parse().unwrap();
    let a = m.fragment_from_atoms([0, 1, 2]).unwrap().render_rooted(1, false).unwrap();
    let b = m.fragment_from_atoms([1, 2, 3]).unwrap().render_rooted(2, false).unwrap();
    assert_eq!(a, b, "symmetric benzene environments must share a label");
}

#[test]
fn rooted_label_is_numbering_invariant() {
    // Isomorphic environments must collapse to one label regardless of the
    // fragment's internal atom numbering (core MAP4 requirement).
    let m: Smiles = "Cc1ccccc1".parse().unwrap();
    for center in 0..m.nodes().len() {
        let Some(env) = m.atom_environment(center, 2) else { continue };
        let atoms: Vec<usize> = env.atoms().collect();
        let reversed: Vec<usize> = atoms.iter().rev().copied().collect();
        let forward = m.fragment_from_atoms(atoms).unwrap().render_rooted(center, false).unwrap();
        let backward =
            m.fragment_from_atoms(reversed).unwrap().render_rooted(center, false).unwrap();
        assert_eq!(forward, backward, "center {center}");
    }
}

#[test]
fn out_of_range_atom_errors() {
    let m: Smiles = "CCO".parse().unwrap();
    assert!(m.fragment_from_atoms([0, 99]).is_err());
}

#[test]
fn render_rooted_errors_for_absent_atom() {
    let m: Smiles = "CCO".parse().unwrap();
    let f = m.fragment_from_atoms([0, 1]).unwrap();
    assert!(f.render_rooted(2, true).is_err());
}

#[test]
fn into_smiles_returns_owned() {
    let m: Smiles = "CCO".parse().unwrap();
    let owned = m.fragment_from_atoms([0, 1]).unwrap().into_smiles();
    assert_eq!(owned.render(), "CC");
}

#[test]
fn from_bonds_out_of_range_endpoint_errors() {
    let m: Smiles = "CCO".parse().unwrap();
    // Target endpoint out of range.
    let err = m.fragment_from_bonds([bond_edge(0, 99, Bond::Single, None)]);
    assert_eq!(err.unwrap_err(), SubgraphError::AtomOutOfRange(99));
    // Source endpoint out of range.
    let err = m.fragment_from_bonds([bond_edge(99, 0, Bond::Single, None)]);
    assert_eq!(err.unwrap_err(), SubgraphError::AtomOutOfRange(99));
}

#[test]
fn from_bonds_non_edge_errors() {
    // Atoms 0 and 2 of CCO are valid but not bonded to each other.
    let m: Smiles = "CCO".parse().unwrap();
    let err = m.fragment_from_bonds([bond_edge(0, 2, Bond::Single, None)]);
    assert_eq!(err.unwrap_err(), SubgraphError::BondReferencesUnknownAtom(2));
}
