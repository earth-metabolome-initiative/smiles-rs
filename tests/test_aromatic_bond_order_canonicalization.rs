//! Aromatic-bond-order representation invariance for canonicalization and
//! rooted environment SMILES.
//!
//! The single/double kekule order retained on an aromatic bond is non-semantic:
//! two representations of the same molecule that differ only in that order must
//! canonicalize identically and produce identical rooted-environment labels,
//! including on broken-ring fragments carved by the MAP4 path.
use smiles_rs::smiles::{AromaticityPolicy, Smiles};

/// Kekulize, then re-perceive RDKit-default aromaticity: an equivalent
/// representation of the molecule that retains kekule orders on aromatic bonds.
fn roundtrip(m: &Smiles) -> Smiles {
    m.kekulize_standalone()
        .unwrap()
        .perceive_aromaticity_for(AromaticityPolicy::RdkitDefault)
        .unwrap()
        .into_aromaticized()
}

#[test]
fn rooted_environment_is_representation_independent() {
    for smiles in ["Cc1ccc(C)cc1", "Cc1ccccc1", "c1ccc2ccccc2c1", "O=C(O)c1ccccc1"] {
        let raw: Smiles = smiles.parse().unwrap();
        let norm = roundtrip(&raw);
        assert_eq!(raw.nodes().len(), norm.nodes().len());
        for i in 0..raw.nodes().len() {
            for r in 1..=3 {
                assert_eq!(
                    raw.rooted_environment_smiles(i, r, false),
                    norm.rooted_environment_smiles(i, r, false),
                    "{smiles} atom {i} radius {r} is representation-dependent",
                );
            }
        }
    }
}

#[test]
fn broken_ring_fragment_canonicalizes_independently_of_kekule_order() {
    let raw: Smiles = "Cc1ccc(C)cc1".parse().unwrap();
    let norm = roundtrip(&raw);
    let raw_frag = raw.atom_environment(6, 2).unwrap().to_fragment().unwrap();
    let norm_frag = norm.atom_environment(6, 2).unwrap().to_fragment().unwrap();
    assert_eq!(
        raw_frag.smiles().canonicalize().render(),
        norm_frag.smiles().canonicalize().render(),
    );
}

#[test]
fn whole_molecule_canonical_form_unchanged() {
    let raw: Smiles = "Cc1ccc(C)cc1".parse().unwrap();
    let norm = roundtrip(&raw);
    assert_eq!(raw.canonicalize().render(), norm.canonicalize().render());
}

#[test]
fn rooted_environment_matches_rdkit_style_value() {
    let raw: Smiles = "Cc1ccc(C)cc1".parse().unwrap();
    let norm = roundtrip(&raw);
    assert_eq!(raw.rooted_environment_smiles(6, 2, false).unwrap(), "c(cc)c(C)c");
    assert_eq!(norm.rooted_environment_smiles(6, 2, false).unwrap(), "c(cc)c(C)c");
}
