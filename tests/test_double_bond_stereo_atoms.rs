//! Reference atoms for interpreting double-bond geometry.

use smiles_rs::{DoubleBondStereoConfig, Smiles, WildcardSmiles};

#[test]
fn chosen_substituent_geometry_is_independent_of_reference_priority() {
    for (input, expected_atoms, expected_config, fluorine_cis_bromine) in [
        ("F/C(Cl)=C/Br", (2, 4), DoubleBondStereoConfig::Z, false),
        ("F/C(C)=C/Br", (0, 4), DoubleBondStereoConfig::E, false),
        ("F/C(I)=C/Br", (2, 4), DoubleBondStereoConfig::Z, false),
        ("F/C([H])=C/Br", (0, 4), DoubleBondStereoConfig::E, false),
        ("F/C(O)=C/Br", (0, 4), DoubleBondStereoConfig::E, false),
        (r"F/C(Cl)=C\Br", (2, 4), DoubleBondStereoConfig::E, true),
    ] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        let config = smiles.double_bond_stereo_config(1, 3).expect("assigned stereo");
        let (left, right) = smiles.double_bond_stereo_atoms(1, 3).expect("reference atoms");
        assert_eq!((left, right), expected_atoms, "{input}");
        assert_eq!(config, expected_config, "{input}");
        assert_eq!(smiles.double_bond_stereo_atoms(3, 1), Some((right, left)), "{input}");
        for (x, y, expected_cis) in [(0, 4, fluorine_cis_bromine), (2, 4, !fluorine_cis_bromine)] {
            let cis = (config == DoubleBondStereoConfig::Z) ^ (x != left) ^ (y != right);
            assert_eq!(cis, expected_cis, "{input}");
        }
    }
}

#[test]
fn reference_atoms_are_absent_without_semantic_stereo() {
    for (input, a, b) in [
        ("CC=CC", 1, 2),
        ("F/C=C", 1, 2),
        ("F/C(F)=C/Br", 1, 3),
        (r"C\1CCCC/C=C1", 5, 6),
        ("C1CCC/C2=C/1CCCCCCC2", 4, 5),
        ("C/1=C/2CCCC(C1)CC2", 0, 1),
        ("F/C=C/Br", 0, 1),
        ("F/C=C/Br", 1, 1),
        ("F/C=C/Br", 1, usize::MAX),
    ] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        assert_eq!(smiles.double_bond_stereo_config(a, b), None, "{input}");
        assert_eq!(smiles.double_bond_stereo_atoms(a, b), None, "{input}");
        assert_eq!(smiles.double_bond_stereo_atoms(b, a), None, "{input}");
    }
}

#[test]
fn ring_double_bonds_of_eight_or_more_atoms_carry_stereo() {
    for (input, a, b, expected_atoms, expected_config) in [
        ("C1CCCCC/C=C/1", 6, 7, (5, 0), DoubleBondStereoConfig::E),
        (r"C1CCCCC/C=C\1", 6, 7, (5, 0), DoubleBondStereoConfig::Z),
        (r"C\1CCCCC/C=C1", 6, 7, (5, 0), DoubleBondStereoConfig::E),
        ("C1=C/CCCCCC/1", 0, 1, (7, 2), DoubleBondStereoConfig::E),
        ("C1CCCCCC/C=C/1", 7, 8, (6, 0), DoubleBondStereoConfig::E),
        (r"C1CCCCCCCCC/C=C\1", 10, 11, (9, 0), DoubleBondStereoConfig::Z),
        ("C1CCC2CCCC/C=C/C2C1", 8, 9, (7, 10), DoubleBondStereoConfig::E),
        (r"C1CCC2CCCC/C=C\C2C1", 8, 9, (7, 10), DoubleBondStereoConfig::Z),
        ("C1CCCC2=C1CCCCC/C=C/2", 11, 12, (10, 4), DoubleBondStereoConfig::E),
    ] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        assert_eq!(smiles.double_bond_stereo_config(a, b), Some(expected_config), "{input}");
        assert_eq!(smiles.double_bond_stereo_atoms(a, b), Some(expected_atoms), "{input}");
        assert_eq!(
            smiles.double_bond_stereo_atoms(b, a),
            Some((expected_atoms.1, expected_atoms.0)),
            "{input}"
        );
    }
}

#[test]
fn reference_atoms_follow_ring_closure_bond_direction() {
    for (input, a, b, expected) in [
        ("F/C=C/1.Br1", 1, 2, (0, 3)),
        (r"F/C=C1.Br\1", 1, 2, (0, 3)),
        ("Br1.F/C=C/1", 2, 3, (1, 0)),
    ] {
        let smiles: Smiles = input.parse().expect("valid SMILES");
        assert_eq!(smiles.double_bond_stereo_atoms(a, b), Some(expected), "{input}");
        assert_eq!(
            smiles.double_bond_stereo_config(a, b),
            Some(DoubleBondStereoConfig::E),
            "{input}"
        );
    }
}

#[test]
fn wildcard_substituent_geometry_distinguishes_isomers() {
    for (input, expected_cis) in [(r"*/C(Cl)=C/Br", false), (r"*/C(Cl)=C\Br", true)] {
        let smiles: WildcardSmiles = input.parse().expect("valid wildcard SMILES");
        for (a, b, x, y) in [(1, 3, 0, 4), (3, 1, 4, 0)] {
            let (left, right) = smiles.double_bond_stereo_atoms(a, b).expect("reference atoms");
            let expected_atoms = if a == 1 { (2, 4) } else { (4, 2) };
            assert_eq!((left, right), expected_atoms, "{input}");
            let config = smiles.double_bond_stereo_config(a, b).expect("assigned stereo");
            let cis = (config == DoubleBondStereoConfig::Z) ^ (x != left) ^ (y != right);
            assert_eq!(cis, expected_cis, "{input}");
        }
    }
}
