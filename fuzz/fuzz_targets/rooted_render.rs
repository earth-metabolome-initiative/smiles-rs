#![no_main]

use libfuzzer_sys::fuzz_target;
use smiles_rs::prelude::Smiles;

// Keep this target in the small-molecule regime so fuzz time goes into the
// invariants rather than a few pathological giant graphs. Rooted rendering
// canonicalizes once per atom, so the atom cap is a touch lower than the other
// targets to keep throughput up.
const MAX_FUZZ_SMILES_BYTES: usize = 256;
const MAX_FUZZ_ATOMS: usize = 48;
const MAX_FUZZ_BONDS: usize = 64;
const MAX_RADIUS: usize = 3;

fn check(data: &str) {
    if data.len() > MAX_FUZZ_SMILES_BYTES {
        return;
    }
    let Ok(smiles) = data.parse::<Smiles>() else {
        return;
    };
    let node_count = smiles.nodes().len();
    if node_count == 0 || node_count > MAX_FUZZ_ATOMS || smiles.number_of_bonds() > MAX_FUZZ_BONDS {
        return;
    }

    // Baseline is the molecule that the default `render()` round-trips to, not
    // `canonicalize(original)`. Re-rooting only changes the traversal order, so
    // it must agree with `render()`. Comparing against the original canonical
    // form would instead test whether `render()` itself perfectly preserves
    // stereo, which is a separate (pre-existing) concern.
    let Ok(rendered_base) = smiles.render().parse::<Smiles>() else {
        return;
    };
    let base_canonical = rendered_base.canonicalize().render();

    for root in 0..node_count {
        check_rooted_render_preserves_molecule(&smiles, root, &base_canonical);
        check_canonical_labeling_rooted(&smiles, root, node_count);
    }

    for center in 0..node_count {
        for radius in 1..=MAX_RADIUS {
            check_environment(&smiles, center, radius);
        }
    }

    check_non_isomeric(&smiles);
}

// `render_rooted` only reorders the traversal, so its output must parse back to
// the same molecule as the default `render()`, and the render must be
// deterministic.
fn check_rooted_render_preserves_molecule(smiles: &Smiles, root: usize, base_canonical: &str) {
    let rooted = smiles.render_rooted(root);
    assert_eq!(rooted, smiles.render_rooted(root), "render_rooted not deterministic at {root}");

    let reparsed = rooted
        .parse::<Smiles>()
        .unwrap_or_else(|err| panic!("rooted render did not parse: {rooted} ({err:?})"));
    assert_eq!(
        reparsed.canonicalize().render(),
        base_canonical,
        "rooting at {root} changed the molecule: {rooted}"
    );
}

// The labeling order is a permutation of all atom ids and contains the root.
fn check_canonical_labeling_rooted(smiles: &Smiles, root: usize, node_count: usize) {
    let labeling = smiles.canonical_labeling_rooted(root);
    let order = labeling.order();
    let mut seen = order.to_vec();
    seen.sort_unstable();
    assert_eq!(seen, (0..node_count).collect::<Vec<usize>>(), "order not a permutation");
    assert!(order.contains(&root), "root missing from labeling order");
}

// An environment label is always a valid molecule, is deterministic, and is
// invariant to the fragment's internal atom numbering (the property MAP4 relies
// on).
fn check_environment(smiles: &Smiles, center: usize, radius: usize) {
    let Some(environment) = smiles.atom_environment(center, radius) else {
        return;
    };
    assert_eq!(environment.center(), center);

    let label = smiles.rooted_environment_smiles(center, radius, false);
    assert_eq!(
        label,
        smiles.rooted_environment_smiles(center, radius, false),
        "environment label not deterministic at ({center}, {radius})"
    );
    if let Some(label) = &label {
        label
            .parse::<Smiles>()
            .unwrap_or_else(|err| panic!("environment label did not parse: {label} ({err:?})"));
    }

    let atoms: Vec<usize> = environment.atoms().collect();
    let reversed: Vec<usize> = atoms.iter().rev().copied().collect();
    let forward = smiles
        .fragment_from_atoms(atoms)
        .expect("environment atoms are valid")
        .render_rooted(center, false)
        .expect("center is in its environment");
    let backward = smiles
        .fragment_from_atoms(reversed)
        .expect("environment atoms are valid")
        .render_rooted(center, false)
        .expect("center is in its environment");
    assert_eq!(forward, backward, "label depends on numbering at ({center}, {radius})");
}

// Dropping isomeric features preserves the heavy-atom skeleton, is idempotent,
// and yields a parseable molecule with no stereo tokens.
fn check_non_isomeric(smiles: &Smiles) {
    let plain = smiles.non_isomeric();
    assert_eq!(plain.nodes().len(), smiles.nodes().len(), "non_isomeric changed atom count");

    let rendered = plain.render();
    assert!(
        !rendered.contains('@') && !rendered.contains('/') && !rendered.contains('\\'),
        "non_isomeric left stereo tokens: {rendered}"
    );
    rendered
        .parse::<Smiles>()
        .unwrap_or_else(|err| panic!("non_isomeric render did not parse: {rendered} ({err:?})"));
    assert_eq!(plain.non_isomeric().render(), rendered, "non_isomeric not idempotent");
}

fuzz_target!(|data: &str| {
    check(data);
});
