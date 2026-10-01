//! Non-isomeric normalization tests.
use smiles_rs::prelude::Smiles;

fn ni(s: &str) -> String {
    s.parse::<Smiles>().unwrap().non_isomeric().render()
}

#[test]
fn drops_isotopes() {
    assert!(!ni("[13CH3]O").contains("13"));
    assert_eq!(ni("[13CH3]O"), ni("[12CH3]O"));
}

#[test]
fn drops_tetrahedral_chirality() {
    let r = ni("N[C@@H](C)C(=O)O");
    assert!(!r.contains('@'), "got {r}");
    assert_eq!(ni("N[C@@H](C)C(=O)O"), ni("N[C@H](C)C(=O)O"));
}

#[test]
fn drops_double_bond_stereo() {
    let r = ni("F/C=C/F");
    assert!(!r.contains('/') && !r.contains('\\'), "got {r}");
    assert_eq!(ni("F/C=C/F"), ni("F/C=C\\F"));
}

#[test]
fn preserves_skeleton_and_is_idempotent() {
    for s in ["[13CH3]O", "N[C@@H](C)C(=O)O", "F/C=C/F", "c1ccccc1", "CCO"] {
        let once = s.parse::<Smiles>().unwrap().non_isomeric();
        assert_eq!(once.render(), once.non_isomeric().render(), "{s} not idempotent");
        // same heavy-atom count
        assert_eq!(once.nodes().len(), s.parse::<Smiles>().unwrap().nodes().len(), "{s}");
    }
}

#[test]
fn order_independent_with_canonicalize() {
    // non_isomeric() must be order-independent with respect to
    // canonicalization: clearing the isotope must also drop the bracket it
    // forced.
    for s in ["C[13CH3]", "[12CH4]", "C[OH]"] {
        let m: Smiles = s.parse().unwrap();
        assert_eq!(
            m.canonicalize().non_isomeric().render(),
            m.non_isomeric().canonicalize().render(),
            "{s} order-dependent",
        );
    }
}

#[test]
fn collapses_brackets_left_by_cleared_isotope() {
    assert_eq!(ni("C[13CH3]"), "CC");
    assert_eq!(ni("[12CH4]"), "C");
}

#[test]
fn keeps_brackets_that_are_still_required() {
    // charge, aromatic N-H, and an explicit hydrogen atom each force a bracket
    // that survives non-isomeric normalization.
    for s in ["C[NH3+]", "c1cc[nH]c1", "[H]C"] {
        let m: Smiles = s.parse().unwrap();
        assert!(m.non_isomeric().render().contains('['), "{s} lost a required bracket");
    }
}

#[test]
fn fragment_label_collapses_bracket_forced_aromatic_nitrogen() {
    // MAP4 fragment labels go through non_isomeric().render_rooted, so the
    // bracket-forced aromatic nitrogen must collapse in the fragment label.
    let m: Smiles = "[CH3][Ga]([CH3])[n]1cccn1".parse().unwrap();
    for (atom_id, atom) in m.nodes().iter().enumerate() {
        if atom.aromatic() {
            let label = m.rooted_environment_smiles(atom_id, 1, false).unwrap();
            assert!(!label.contains("[n]"), "atom {atom_id} label {label} kept [n]");
        }
    }
}
