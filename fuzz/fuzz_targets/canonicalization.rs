#![no_main]

use libfuzzer_sys::fuzz_target;
use smiles_rs::smiles::{Smiles, WildcardSmiles};

fn assert_strict_canonicalization(data: &str) {
    match data.parse::<Smiles>() {
        Ok(smiles) => {
            smiles.debug_assert_canonicalization_invariants();
        }
        Err(err) => {
            let _ = err.render(data);
        }
    }
}

fn assert_wildcard_canonicalization(data: &str) {
    match data.parse::<WildcardSmiles>() {
        Ok(smiles) => {
            smiles.debug_assert_canonicalization_invariants();
        }
        Err(err) => {
            let _ = err.render(data);
        }
    }
}

fuzz_target!(|data: &str| {
    assert_strict_canonicalization(data);
    assert_wildcard_canonicalization(data);
});
