//! One-shot MAP4 substructure label + public export checks.
use smiles_rs::prelude::{AtomEnvironment, Fragment, RootError, Smiles, SubgraphError};

fn canon(s: &str) -> String {
    s.parse::<Smiles>().unwrap().canonicalize().render()
}

#[test]
fn empty_shell_folds_to_none() {
    let m: Smiles = "CCO".parse().unwrap();
    assert!(m.rooted_environment_smiles(1, 2, false).is_none());
    assert!(m.rooted_environment_smiles(0, 0, false).is_none());
}

#[test]
fn produces_centered_label() {
    let m: Smiles = "CCO".parse().unwrap();
    let label = m.rooted_environment_smiles(0, 2, false).unwrap();
    assert_eq!(canon(&label), canon("CCO"));
}

#[test]
fn isotopes_collapse_under_non_isomeric() {
    let a: Smiles = "[13CH3]CO".parse().unwrap();
    let b: Smiles = "[12CH3]CO".parse().unwrap();
    assert_eq!(a.rooted_environment_smiles(0, 2, false), b.rooted_environment_smiles(0, 2, false),);
}

// Type-level check that the public surface is exported through the prelude.
#[allow(dead_code)]
fn exports_are_public(
    _env: AtomEnvironment<'_>,
    _frag: Fragment,
    _e: SubgraphError,
    _r: RootError,
) {
}
