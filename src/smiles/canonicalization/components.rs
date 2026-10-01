use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrixRef;

use super::{
    SmilesCanonicalLabeling, canonicalization_state_key, remap_parsed_stereo_neighbors_row,
};
use crate::smiles::{BondMatrixBuilder, Smiles};

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    pub(super) fn canonical_labeling_with(
        &self,
        whole_graph_labeling: impl Fn(&Self) -> SmilesCanonicalLabeling,
    ) -> SmilesCanonicalLabeling {
        if self.connected_components().number_of_components() <= 1 {
            return whole_graph_labeling(self);
        }

        self.componentwise_canonical_labeling(whole_graph_labeling)
    }

    fn componentwise_canonical_labeling(
        &self,
        whole_graph_labeling: impl Fn(&Self) -> SmilesCanonicalLabeling,
    ) -> SmilesCanonicalLabeling {
        let component_count = self.connected_components().number_of_components();
        let mut keyed_component_orders = Vec::with_capacity(component_count);

        self.for_each_component_subgraph(|old_nodes, component| {
            let component_labeling = whole_graph_labeling(&component);
            let component_canonicalized =
                component.exact_canonicalize_with_labeling(&component_labeling);
            // Disconnected components are canonicalized independently, then
            // ordered by their canonical graph state so permutations of whole
            // components collapse to one output ordering.
            let order = component_labeling
                .order()
                .iter()
                .copied()
                .map(|component_node| old_nodes[component_node])
                .collect::<Vec<_>>();
            keyed_component_orders
                .push((canonicalization_state_key(&component_canonicalized), order));
        });

        keyed_component_orders.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        SmilesCanonicalLabeling::new(
            keyed_component_orders.into_iter().flat_map(|(_key, order)| order).collect(),
        )
    }

    fn induced_subgraph(&self, old_nodes: &[usize]) -> Self {
        let mut new_index_of_old_node = vec![usize::MAX; self.nodes().len()];
        for (new_index, old_node) in old_nodes.iter().copied().enumerate() {
            new_index_of_old_node[old_node] = new_index;
        }

        let atom_nodes =
            old_nodes.iter().copied().map(|old_node| self.atom_nodes[old_node]).collect::<Vec<_>>();
        let mut builder = BondMatrixBuilder::default();
        for ((row, column), entry) in self.bond_matrix().sparse_entries() {
            if row >= column
                || new_index_of_old_node[row] == usize::MAX
                || new_index_of_old_node[column] == usize::MAX
            {
                continue;
            }
            builder
                .push_edge_with_descriptor(
                    new_index_of_old_node[row],
                    new_index_of_old_node[column],
                    entry.descriptor(),
                    None,
                )
                .unwrap_or_else(|_| {
                    unreachable!("induced component extraction preserves a simple graph")
                });
        }

        let parsed_stereo_neighbors = old_nodes
            .iter()
            .copied()
            .map(|old_node| {
                remap_parsed_stereo_neighbors_row(self, old_node, &new_index_of_old_node)
            })
            .collect::<Vec<_>>();
        let implicit_hydrogen_cache = old_nodes
            .iter()
            .copied()
            .map(|old_node| self.implicit_hydrogen_cache[old_node])
            .collect();

        Self::from_bond_matrix_parts_with_sidecars(
            atom_nodes,
            builder.finish(old_nodes.len()),
            parsed_stereo_neighbors,
            implicit_hydrogen_cache,
            None,
        )
    }

    fn for_each_component_subgraph(&self, mut visit: impl FnMut(&[usize], Self)) {
        let components = self.connected_components();
        for component_id in 0..components.number_of_components() {
            let old_nodes = components.node_ids_of_component(component_id).collect::<Vec<_>>();
            let component = self.induced_subgraph(&old_nodes);
            visit(&old_nodes, component);
        }
    }

    pub(super) fn stereo_neutral_preparation_classes(&self) -> (Vec<usize>, Vec<usize>) {
        let components = self.connected_components();
        if components.number_of_components() <= 1 {
            let refined_classes = self.stereo_neutral_refined_classes();
            let rooted_classes = self.stereo_neutral_rooted_classes(&refined_classes);
            return (refined_classes, rooted_classes);
        }

        let mut refined_classes = vec![0; self.nodes().len()];
        let mut rooted_classes = vec![0; self.nodes().len()];

        self.for_each_component_subgraph(|old_nodes, component| {
            let component_refined_classes = component.stereo_neutral_refined_classes();
            let component_rooted_classes =
                component.stereo_neutral_rooted_classes(&component_refined_classes);

            // Stereo-neutral refinement does not need information to cross
            // disconnected components, so compute it per component and project
            // the class ids back into the original node numbering.
            for (new_node, old_node) in old_nodes.iter().copied().enumerate() {
                refined_classes[old_node] = component_refined_classes[new_node];
                rooted_classes[old_node] = component_rooted_classes[new_node];
            }
        });

        (refined_classes, rooted_classes)
    }
}
