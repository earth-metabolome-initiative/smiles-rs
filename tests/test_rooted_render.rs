//! Root-anchored rendering tests.
use smiles_rs::prelude::Smiles;

fn canon(s: &str) -> String {
    s.parse::<Smiles>().unwrap().canonicalize().render()
}

fn first_atom(s: &str) -> String {
    format!("{:?}", s.parse::<Smiles>().unwrap().node_by_id(0).unwrap())
}

#[test]
fn deterministic() {
    let m: Smiles = "Cc1ccccc1".parse().unwrap();
    for r in 0..m.nodes().len() {
        assert_eq!(m.render_rooted(r), m.render_rooted(r));
    }
}

#[test]
fn round_trips_same_molecule() {
    for s in ["CCO", "Cc1ccccc1", "CC(=O)O", "c1ccccc1", "FC(Cl)Br", "C1CC1.CC"] {
        let m: Smiles = s.parse().unwrap();
        for r in 0..m.nodes().len() {
            assert_eq!(canon(&m.render_rooted(r)), canon(s), "{s}@{r}");
        }
    }
}

#[test]
fn rooting_is_independent_of_root_for_tricky_stereo() {
    // Pathological input (aromatic atoms outside a ring, adjacent cumulated
    // double bond with directional stereo) from a fuzz finding: every root must
    // render to the same molecule as the default render().
    let m: Smiles = "CCOCNO=FNO=cc/CNC".parse().unwrap();
    let base = m.render().parse::<Smiles>().unwrap().canonicalize().render();
    for r in 0..m.nodes().len() {
        let rooted = m.render_rooted(r).parse::<Smiles>().unwrap().canonicalize().render();
        assert_eq!(rooted, base, "root {r}");
    }
}

#[test]
fn rooting_keeps_directional_bond_that_becomes_ring_closure() {
    for s in ["C1F/C=C1=C", "C2C=CC=C=C/C2=CCC"] {
        let m: Smiles = s.parse().unwrap();
        let base = m.render().parse::<Smiles>().unwrap().canonicalize().render();
        assert!(base.contains('/'), "{s}: baseline lost its directional bond: {base}");
        for r in 0..m.nodes().len() {
            let rooted = m.render_rooted(r).parse::<Smiles>().unwrap().canonicalize().render();
            assert_eq!(rooted, base, "{s} root {r}");
        }
    }
}

#[test]
fn starts_at_root() {
    for s in ["CCO", "Cc1ccccc1", "FC(Cl)Br", "c1ccccc1"] {
        let m: Smiles = s.parse().unwrap();
        for r in 0..m.nodes().len() {
            assert_eq!(
                first_atom(&m.render_rooted(r)),
                format!("{:?}", m.node_by_id(r).unwrap()),
                "{s}@{r}"
            );
        }
    }
}

#[test]
fn matches_default_and_differs_by_root() {
    let m: Smiles = "CCO".parse().unwrap();
    assert_eq!(m.render_rooted(0), m.render());
    assert_ne!(m.render_rooted(0), m.render_rooted(2));
    assert!(m.render_rooted(2).starts_with('O'));
}

#[test]
#[should_panic(expected = "root")]
fn panics_oob() {
    let _ = "CCO".parse::<Smiles>().unwrap().render_rooted(3);
}

#[test]
fn labeling_rooted() {
    for s in ["CCO", "Cc1ccccc1", "FC(Cl)Br"] {
        let m: Smiles = s.parse().unwrap();
        for r in 0..m.nodes().len() {
            let l = m.canonical_labeling_rooted(r);
            assert_eq!(l.order()[0], r);
            assert_eq!(l.new_index_of_old_node()[r], 0);
            let mut seen = l.order().to_vec();
            seen.sort_unstable();
            assert_eq!(seen, (0..m.nodes().len()).collect::<Vec<_>>());
        }
    }
}

#[test]
fn labeling_rooted_multi_component_is_valid_permutation() {
    // With several components, root leads its own run but need not be ordinal
    // 0.
    let m: Smiles = "C1CC1.CC".parse().unwrap();
    let l = m.canonical_labeling_rooted(4);
    let mut seen = l.order().to_vec();
    seen.sort_unstable();
    assert_eq!(seen, (0..m.nodes().len()).collect::<Vec<_>>());
    assert!(l.order().contains(&4));
}

#[test]
#[should_panic(expected = "root")]
fn labeling_panics_oob() {
    let _ = "CCO".parse::<Smiles>().unwrap().canonical_labeling_rooted(3);
}
