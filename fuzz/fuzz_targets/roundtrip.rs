#![no_main]

use std::string::ToString;

use libfuzzer_sys::fuzz_target;
use smiles_rs::smiles::{Smiles, WildcardSmiles};

fn assert_strict_roundtrip(data: &str) {
    match data.parse::<Smiles>() {
        Ok(smiles) => {
            let rendered = smiles.to_string();
            let reparsed = rendered.parse::<Smiles>().unwrap();
            let rerendered = reparsed.to_string();
            assert_eq!(
                rendered, rerendered,
                "strict rendered forms differ after round-trip: rendered={rendered} rerendered={rerendered}"
            );
        }
        Err(err) => {
            let _ = err.render(data);
        }
    }
}

fn assert_wildcard_roundtrip(data: &str) {
    match data.parse::<WildcardSmiles>() {
        Ok(smiles) => {
            let rendered = smiles.to_string();
            let reparsed = rendered.parse::<WildcardSmiles>().unwrap();
            let rerendered = reparsed.to_string();
            assert_eq!(
                rendered, rerendered,
                "wildcard rendered forms differ after round-trip: rendered={rendered} rerendered={rerendered}"
            );
        }
        Err(err) => {
            let _ = err.render(data);
        }
    }
}

fuzz_target!(|data: &str| {
    assert_strict_roundtrip(data);
    assert_wildcard_roundtrip(data);
});
