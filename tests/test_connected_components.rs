//! Connected-components parity checks against documented `RDKit` fragment
//! behavior.

use smiles_rs::smiles::{Smiles, WildcardSmiles};

struct RdkitFixture {
    smiles: &'static str,
    components: &'static [&'static [usize]],
    component_ids: &'static [usize],
}

const RDKIT_CONNECTED_COMPONENTS_FIXTURES: &[RdkitFixture] = &[
    RdkitFixture {
        // From the RDKit `GetMolFrags()` documentation.
        smiles: "CC(=O)[O-].[NH3+]C",
        components: &[&[0, 1, 2, 3], &[4, 5]],
        component_ids: &[0, 0, 0, 0, 1, 1],
    },
    RdkitFixture {
        smiles: "C.CC.N",
        components: &[&[0], &[1, 2], &[3]],
        component_ids: &[0, 1, 1, 2],
    },
    RdkitFixture {
        smiles: "Cl.C1CC1.O",
        components: &[&[0], &[1, 2, 3], &[4]],
        component_ids: &[0, 1, 1, 1, 2],
    },
    RdkitFixture {
        smiles: "C1CC1.C1CC1",
        components: &[&[0, 1, 2], &[3, 4, 5]],
        component_ids: &[0, 0, 0, 1, 1, 1],
    },
];

#[test]
fn connected_components_match_rdkit_fragment_fixtures() {
    for fixture in RDKIT_CONNECTED_COMPONENTS_FIXTURES {
        let smiles = Smiles::from_str(fixture.smiles)
            .unwrap_or_else(|error| panic!("failed to parse {:?}: {error}", fixture.smiles));
        let components = smiles.connected_components();

        assert_eq!(
            components.number_of_components(),
            fixture.components.len(),
            "{}",
            fixture.smiles
        );
        assert_eq!(
            components.component_identifiers().collect::<Vec<_>>(),
            fixture.component_ids,
            "{}",
            fixture.smiles
        );

        let expected_largest =
            fixture.components.iter().map(|component| component.len()).max().unwrap_or(0);
        let expected_smallest =
            fixture.components.iter().map(|component| component.len()).min().unwrap_or(0);
        assert_eq!(components.largest_component_size(), expected_largest, "{}", fixture.smiles);
        assert_eq!(components.smallest_component_size(), expected_smallest, "{}", fixture.smiles);

        for (component_id, expected_nodes) in fixture.components.iter().enumerate() {
            assert_eq!(
                components.node_ids_of_component(component_id).collect::<Vec<_>>(),
                *expected_nodes,
                "{} component {component_id}",
                fixture.smiles
            );

            for &node_id in *expected_nodes {
                assert_eq!(
                    components.component_of_node(node_id),
                    component_id,
                    "{} node {node_id}",
                    fixture.smiles
                );
            }
        }
    }
}

#[test]
fn connected_components_returns_atoms_for_each_component() {
    let smiles = Smiles::from_str("C.O").unwrap();
    let components = smiles.connected_components();

    let left = components.nodes_of_component(0).collect::<Vec<_>>();
    let right = components.nodes_of_component(1).collect::<Vec<_>>();

    assert_eq!(left, vec![*smiles.node_by_id(0).unwrap()]);
    assert_eq!(right, vec![*smiles.node_by_id(1).unwrap()]);
}

#[test]
fn wildcard_connected_components_keep_wildcard_api_surface() {
    let smiles = WildcardSmiles::from_str("*.CC").unwrap();
    let components = smiles.connected_components();

    assert_eq!(components.number_of_components(), 2);
    assert_eq!(components.component_identifiers().collect::<Vec<_>>(), vec![0, 1, 1]);
    assert!(components.nodes_of_component(0).next().unwrap().symbol().is_wildcard());
}
