use smiles_rs::smiles::{Smiles, WildcardSmiles};

pub fn canonical_string(source: &str) -> String {
    let canonicalized = Smiles::from_str(source).unwrap().canonicalize();
    assert!(canonicalized.is_canonical(), "canonicalized form is not fixed: {source}");
    canonicalized.to_string()
}

pub fn assert_same_canonical_group(group: &[&str]) {
    let expected = canonical_string(group[0]);
    for source in &group[1..] {
        assert_eq!(expected, canonical_string(source), "group did not converge: {group:?}");
    }
}

#[allow(dead_code)]
pub fn wildcard_canonical_string(source: &str) -> String {
    let canonicalized = WildcardSmiles::from_str(source).unwrap().canonicalize();
    assert!(canonicalized.is_canonical(), "canonicalized form is not fixed: {source}");
    canonicalized.to_string()
}
