#![no_main]

use libfuzzer_sys::fuzz_target;
use smiles_rs::prelude::{AromaticityPolicy, AromaticityStatus, Smiles, WildcardSmiles};

// Keep this target in the small-molecule regime so fuzz time goes into
// roundtrip invariants instead of a few pathological giant graphs.
const MAX_FUZZ_SMILES_BYTES: usize = 256;
const MAX_FUZZ_ATOMS: usize = 80;
const MAX_FUZZ_BONDS: usize = 96;

const POLICIES: [AromaticityPolicy; 3] = [
    AromaticityPolicy::RdkitDefault,
    AromaticityPolicy::RdkitSimple,
    AromaticityPolicy::RdkitMdl,
];

macro_rules! define_policy_roundtrip_runner {
    ($runner:ident, $smiles_type:ty) => {
        fn $runner(data: &str) {
            let Ok(smiles) = data.parse::<$smiles_type>() else {
                return;
            };
            if smiles.nodes().len() > MAX_FUZZ_ATOMS
                || smiles.number_of_bonds() > MAX_FUZZ_BONDS
                || smiles.nodes().iter().any(|atom| atom.aromatic())
                || smiles.nodes().iter().enumerate().any(|(atom_id, _)| {
                    smiles.edges_for_node(atom_id).any(|edge| edge.is_aromatic())
                })
            {
                return;
            }

            for policy in POLICIES {
                let perception = match smiles.perceive_aromaticity_for(policy) {
                    Ok(perception) => perception,
                    Err(_) => return,
                };

                let preserved = perception
                    .kekulize()
                    .expect("preserve-source kekulization should succeed after perception");
                assert_eq!(
                    &preserved,
                    &smiles,
                    "preserve-source kekulization should restore the original graph for policy {policy:?}"
                );

                if perception.status() == AromaticityStatus::Unsupported {
                    continue;
                }

                let expected_atom_ids = perception.assignment().atom_ids().to_vec();
                let expected_bond_edges = perception.assignment().bond_edges().to_vec();

                let Ok(standalone) = perception.kekulize_standalone() else {
                    continue;
                };
                let reperceived = standalone
                    .perceive_aromaticity_for(policy)
                    .expect("reperception should succeed after standalone kekulization");

                assert_eq!(
                    reperceived.assignment().atom_ids(),
                    expected_atom_ids,
                    "standalone kekulization changed aromatic atom ids for policy {policy:?}"
                );
                assert_eq!(
                    reperceived.assignment().bond_edges(),
                    expected_bond_edges,
                    "standalone kekulization changed aromatic bond edges for policy {policy:?}"
                );
            }
        }
    };
}

define_policy_roundtrip_runner!(run_strict_policy_roundtrips, Smiles);
define_policy_roundtrip_runner!(run_wildcard_policy_roundtrips, WildcardSmiles);

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_FUZZ_SMILES_BYTES {
        return;
    }

    let Ok(data) = core::str::from_utf8(data) else {
        return;
    };
    run_strict_policy_roundtrips(data);
    run_wildcard_policy_roundtrips(data);
});
