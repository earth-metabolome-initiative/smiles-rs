use alloc::vec::Vec;

use geometric_traits::traits::{SparseMatrix2D, SparseValuedMatrix2DRef};

use super::{Smiles, SmilesAtomPolicy, invariants::AtomInvariant};
use crate::bond::{
    BondDescriptor,
    bond_edge::{BondEdge, bond_edge_other},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpanningForest {
    roots: Vec<usize>,
    parents: Vec<Option<usize>>,
    parent_bonds: Vec<Option<BondDescriptor>>,
    children: Vec<Vec<usize>>,
    child_bonds: Vec<Vec<BondDescriptor>>,
    closure_edges: Vec<BondEdge>,
}

impl SpanningForest {
    #[must_use]
    pub(crate) fn roots(&self) -> &[usize] {
        &self.roots
    }

    #[must_use]
    pub(crate) fn parent_of(&self, node_id: usize) -> Option<usize> {
        self.parents.get(node_id).copied().flatten()
    }

    #[must_use]
    pub(crate) fn parent_bond_of(&self, node_id: usize) -> Option<BondDescriptor> {
        self.parent_bonds.get(node_id).copied().flatten()
    }

    #[must_use]
    pub(crate) fn children_of(&self, node_id: usize) -> &[usize] {
        self.children.get(node_id).map_or(&[], Vec::as_slice)
    }

    #[must_use]
    pub(crate) fn children(&self) -> &[Vec<usize>] {
        &self.children
    }

    #[must_use]
    pub(crate) fn child_bonds_of(&self, node_id: usize) -> &[BondDescriptor] {
        self.child_bonds.get(node_id).map_or(&[], Vec::as_slice)
    }

    #[must_use]
    pub(crate) fn closure_edges(&self) -> &[BondEdge] {
        &self.closure_edges
    }
}

struct ForestBuildState {
    visited: Vec<bool>,
    parents: Vec<Option<usize>>,
    parent_bonds: Vec<Option<BondDescriptor>>,
    children: Vec<Vec<usize>>,
    child_bonds: Vec<Vec<BondDescriptor>>,
    closure_pairs: Vec<(usize, usize)>,
}

impl ForestBuildState {
    fn new(node_count: usize) -> Self {
        Self {
            visited: vec![false; node_count],
            parents: vec![None; node_count],
            parent_bonds: vec![None; node_count],
            children: vec![Vec::new(); node_count],
            child_bonds: vec![Vec::new(); node_count],
            closure_pairs: Vec::new(),
        }
    }

    fn into_forest<AtomPolicy: SmilesAtomPolicy>(
        self,
        roots: &[usize],
        smiles: &Smiles<AtomPolicy>,
    ) -> SpanningForest {
        let mut closure_keys = self.closure_pairs;
        closure_keys.sort_unstable();
        closure_keys.dedup();
        let closure_edges = closure_keys
            .into_iter()
            .map(|edge| smiles.edge_for_node_pair(edge).unwrap_or_else(|| unreachable!()))
            .collect();

        SpanningForest {
            roots: roots.to_vec(),
            parents: self.parents,
            parent_bonds: self.parent_bonds,
            children: self.children,
            child_bonds: self.child_bonds,
            closure_edges,
        }
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn spanning_forest(&self) -> SpanningForest {
        let invariants = self.atom_invariants();
        let refined = self.refined_atom_classes_from_invariants(&invariants);
        let rooted = self.rooted_symmetry_classes_from_refined(refined.classes());
        let roots = self.component_roots_from_planning(&invariants, &rooted);
        self.spanning_forest_with_planning(&roots, &invariants, refined.classes(), &rooted)
    }

    #[must_use]
    pub(crate) fn spanning_forest_with_planning(
        &self,
        roots: &[usize],
        invariants: &[AtomInvariant],
        refined_classes: &[usize],
        rooted_classes: &[usize],
    ) -> SpanningForest {
        let ordered_neighbors = self.ordered_neighbor_edges_table_with_planning(
            invariants,
            refined_classes,
            rooted_classes,
        );
        self.spanning_forest_with_ordered_neighbors(roots, &ordered_neighbors)
    }

    #[must_use]
    pub(crate) fn spanning_forest_with_parser_neighbor_order(
        &self,
        roots: &[usize],
    ) -> SpanningForest {
        let node_count = self.nodes().len();
        let mut state = ForestBuildState::new(node_count);

        for &root in roots {
            if root >= node_count || state.visited[root] {
                continue;
            }
            state.visited[root] = true;
            self.build_spanning_tree_from_parser_neighbor_order(root, &mut state);
        }

        state.into_forest(roots, self)
    }

    fn spanning_forest_with_ordered_neighbors(
        &self,
        roots: &[usize],
        ordered_neighbors: &[Vec<BondEdge>],
    ) -> SpanningForest {
        let node_count = self.nodes().len();
        let mut state = ForestBuildState::new(node_count);

        for &root in roots {
            if root >= node_count || state.visited[root] {
                continue;
            }
            state.visited[root] = true;
            self.build_spanning_tree_from(root, ordered_neighbors, &mut state);
        }

        state.into_forest(roots, self)
    }

    fn build_spanning_tree_from(
        &self,
        node_id: usize,
        ordered_neighbors: &[Vec<BondEdge>],
        state: &mut ForestBuildState,
    ) {
        let edges = &ordered_neighbors[node_id];
        for edge in edges.iter().copied() {
            self.visit_spanning_edge(node_id, edge, ordered_neighbors, state);
        }
    }

    fn visit_spanning_edge(
        &self,
        node_id: usize,
        edge: BondEdge,
        ordered_neighbors: &[Vec<BondEdge>],
        state: &mut ForestBuildState,
    ) {
        let neighbor_id = bond_edge_other(edge, node_id).unwrap_or_else(|| unreachable!());
        if state.parents[node_id] == Some(neighbor_id) {
            return;
        }

        if state.visited[neighbor_id] {
            state.closure_pairs.push(crate::smiles::edge_key(node_id, neighbor_id));
        } else {
            state.visited[neighbor_id] = true;
            state.parents[neighbor_id] = Some(node_id);
            state.parent_bonds[neighbor_id] = Some(edge_descriptor(edge));
            state.children[node_id].push(neighbor_id);
            state.child_bonds[node_id].push(edge_descriptor(edge));
            self.build_spanning_tree_from(neighbor_id, ordered_neighbors, state);
        }
    }

    fn parser_ordered_neighbor_edges(&self, node_id: usize) -> Vec<BondEdge> {
        let mut ordered_neighbors: Vec<(usize, usize, BondEdge)> = self
            .bond_matrix
            .sparse_row(node_id)
            .zip(self.bond_matrix.sparse_row_values_ref(node_id))
            .map(|(neighbor_id, entry)| {
                (entry.order(), neighbor_id, entry.to_bond_edge(node_id, neighbor_id))
            })
            .collect();
        ordered_neighbors.sort_unstable_by(|left, right| {
            left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1))
        });
        ordered_neighbors.into_iter().map(|(_order, _neighbor_id, edge)| edge).collect()
    }

    fn build_spanning_tree_from_parser_neighbor_order(
        &self,
        node_id: usize,
        state: &mut ForestBuildState,
    ) {
        for edge in self.parser_ordered_neighbor_edges(node_id) {
            let neighbor_id = bond_edge_other(edge, node_id).unwrap_or_else(|| unreachable!());
            if state.parents[node_id] == Some(neighbor_id) {
                continue;
            }

            if state.visited[neighbor_id] {
                state.closure_pairs.push(crate::smiles::edge_key(node_id, neighbor_id));
            } else {
                state.visited[neighbor_id] = true;
                state.parents[neighbor_id] = Some(node_id);
                state.parent_bonds[neighbor_id] = Some(edge_descriptor(edge));
                state.children[node_id].push(neighbor_id);
                state.child_bonds[node_id].push(edge_descriptor(edge));
                self.build_spanning_tree_from_parser_neighbor_order(neighbor_id, state);
            }
        }
    }
}

#[inline]
#[must_use]
fn edge_descriptor(edge: BondEdge) -> BondDescriptor {
    if edge.is_aromatic() { BondDescriptor::aromatic(edge.bond()) } else { edge.bond().into() }
}

#[cfg(test)]
mod tests {
    use super::Smiles;
    use crate::bond::Bond;

    #[test]
    fn spanning_forest_of_empty_graph_is_empty() {
        let forest = Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy().spanning_forest();
        assert!(forest.roots().is_empty());
        assert!(forest.closure_edges().is_empty());
        assert_eq!(forest.parent_of(0), None);
        assert!(forest.children_of(0).is_empty());
    }

    #[test]
    fn spanning_forest_builds_parent_chain_for_acyclic_component() {
        let smiles: Smiles = "CCCO".parse().unwrap();
        let forest = smiles.spanning_forest();

        assert_eq!(forest.roots(), &[0]);
        assert_eq!(forest.parent_of(0), None);
        assert_eq!(forest.parent_of(1), Some(0));
        assert_eq!(forest.parent_of(2), Some(1));
        assert_eq!(forest.parent_of(3), Some(2));
        assert_eq!(forest.children_of(0), &[1]);
        assert_eq!(forest.children_of(1), &[2]);
        assert_eq!(forest.children_of(2), &[3]);
        assert!(forest.closure_edges().is_empty());
    }

    #[test]
    fn spanning_forest_classifies_cycle_edge_as_closure() {
        let smiles: Smiles = "C1CC1".parse().unwrap();
        let forest = smiles.spanning_forest();

        assert_eq!(forest.roots(), &[0]);
        assert_eq!(forest.parent_of(0), None);
        assert_eq!(forest.parent_of(1), Some(0));
        assert_eq!(forest.parent_of(2), Some(1));
        assert_eq!(forest.children_of(0), &[1]);
        assert_eq!(forest.children_of(1), &[2]);
        assert_eq!(forest.closure_edges(), &[smiles.edge_for_node_pair((0, 2)).unwrap()]);
    }

    #[test]
    fn spanning_forest_separates_disconnected_components() {
        let smiles: Smiles = "C1CC1.CC".parse().unwrap();
        let forest = smiles.spanning_forest();

        assert_eq!(forest.roots(), &[0, 3]);
        assert_eq!(forest.parent_of(0), None);
        assert_eq!(forest.parent_of(1), Some(0));
        assert_eq!(forest.parent_of(2), Some(1));
        assert_eq!(forest.parent_of(3), None);
        assert_eq!(forest.parent_of(4), Some(3));
        assert_eq!(forest.children_of(3), &[4]);
        assert_eq!(forest.closure_edges(), &[smiles.edge_for_node_pair((0, 2)).unwrap()]);
    }

    #[test]
    fn spanning_forest_preserves_neighbor_order_when_choosing_tree_children() {
        let smiles: Smiles = "C1CCC2CC2C1".parse().unwrap();
        let forest = smiles.spanning_forest();

        assert_eq!(forest.roots(), &[0]);
        assert_eq!(forest.children_of(0), &[1]);
        assert_eq!(
            forest.closure_edges(),
            &[
                smiles.edge_for_node_pair((0, 6)).unwrap(),
                smiles.edge_for_node_pair((3, 5)).unwrap()
            ]
        );
    }

    #[test]
    fn parser_neighbor_order_forest_ignores_out_of_range_and_duplicate_roots() {
        let smiles: Smiles = "C1CC1.CC".parse().unwrap();
        let forest = smiles.spanning_forest_with_parser_neighbor_order(&[0, 0, 99, 3]);

        assert_eq!(forest.roots(), &[0, 0, 99, 3]);
        assert_eq!(forest.parent_of(0), None);
        assert_eq!(forest.parent_of(1), Some(0));
        assert_eq!(forest.parent_of(2), Some(1));
        assert_eq!(forest.parent_of(3), None);
        assert_eq!(forest.parent_of(4), Some(3));
        assert_eq!(forest.parent_bond_of(1), Some(Bond::Single.into()));
        assert_eq!(forest.child_bonds_of(0), &[Bond::Single.into()]);
        assert_eq!(forest.children(), &[vec![1], vec![2], vec![], vec![4], vec![]]);
    }

    #[test]
    fn parser_neighbor_order_preserves_bond_insertion_order_for_tree_edges() {
        let smiles: Smiles = "C1(C)CC1".parse().unwrap();
        let forest = smiles.spanning_forest_with_parser_neighbor_order(&[0]);

        assert_eq!(forest.children_of(0), &[1, 2]);
        assert_eq!(forest.child_bonds_of(0), &[Bond::Single.into(), Bond::Single.into()]);
        assert_eq!(forest.closure_edges(), &[smiles.edge_for_node_pair((0, 3)).unwrap()]);
    }
}
