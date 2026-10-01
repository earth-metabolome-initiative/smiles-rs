use alloc::vec::Vec;

use geometric_traits::traits::{SparseMatrix2D, SparseValuedMatrix2DRef};
use smallvec::SmallVec;

use super::{Smiles, invariants::bond_entry_code};

type Neighborhood = SmallVec<[(u8, usize); 4]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectedEdge {
    from: usize,
    to: usize,
    reverse: usize,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn rooted_symmetry_classes(&self) -> Vec<usize> {
        let refined = self.refined_atom_classes();
        self.rooted_symmetry_classes_from_refined(refined.classes())
    }

    #[must_use]
    pub(crate) fn rooted_symmetry_classes_from_refined(
        &self,
        initial_classes: &[usize],
    ) -> Vec<usize> {
        let node_count = initial_classes.len();
        if node_count == 0 {
            return Vec::new();
        }
        if classes_are_unique(initial_classes) {
            return initial_classes.to_vec();
        }

        let topology = self.directed_edge_topology();
        let directed_edges = topology.directed_edges();
        if directed_edges.is_empty() {
            return initial_classes.to_vec();
        }

        let mut edge_classes: Vec<usize> =
            directed_edges.iter().map(|edge| initial_classes[edge.to]).collect();
        let edge_node_classes: Vec<usize> =
            directed_edges.iter().map(|edge| initial_classes[edge.to]).collect();
        let mut edge_neighborhoods: Vec<Neighborhood> = directed_edges
            .iter()
            .map(|edge| {
                SmallVec::with_capacity(
                    topology
                        .row_edge_ids(edge.to)
                        .len()
                        .saturating_sub(usize::from(edge.reverse != usize::MAX)),
                )
            })
            .collect();
        let mut node_neighborhoods: Vec<Neighborhood> =
            topology.row_edge_ids.iter().map(|row| SmallVec::with_capacity(row.len())).collect();
        let mut edge_indices: Vec<usize> = (0..directed_edges.len()).collect();
        let mut next_edge_classes = vec![0usize; directed_edges.len()];
        let mut node_indices: Vec<usize> = (0..node_count).collect();
        let mut rooted_classes = vec![0usize; node_count];

        loop {
            for (edge_id, edge) in directed_edges.iter().enumerate() {
                let neighborhood = &mut edge_neighborhoods[edge_id];
                neighborhood.clear();
                neighborhood.extend(
                    topology
                        .row_edge_ids(edge.to)
                        .iter()
                        .copied()
                        .filter(|&neighbor_edge_id| neighbor_edge_id != edge.reverse)
                        .map(|neighbor_edge_id| {
                            (
                                topology.directed_edge_bond_kind(neighbor_edge_id),
                                edge_classes[neighbor_edge_id],
                            )
                        }),
                );
                neighborhood.sort_unstable();
            }

            dense_ranks_by_current_and_neighborhood_reuse(
                &edge_node_classes,
                &edge_neighborhoods,
                &mut edge_indices,
                &mut next_edge_classes,
            );
            if next_edge_classes == edge_classes {
                break;
            }
            core::mem::swap(&mut edge_classes, &mut next_edge_classes);
        }

        for (node_id, neighborhood) in node_neighborhoods.iter_mut().enumerate() {
            neighborhood.clear();
            neighborhood.extend(
                topology.row_edge_ids(node_id).iter().copied().map(|edge_id| {
                    (topology.directed_edge_bond_kind(edge_id), edge_classes[edge_id])
                }),
            );
            neighborhood.sort_unstable();
        }

        dense_ranks_by_current_and_neighborhood_reuse(
            initial_classes,
            &node_neighborhoods,
            &mut node_indices,
            &mut rooted_classes,
        );
        rooted_classes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectedEdgeTopology {
    directed_edges: Vec<DirectedEdge>,
    row_edge_ids: Vec<Vec<usize>>,
    bond_kinds: Vec<u8>,
}

impl DirectedEdgeTopology {
    fn directed_edges(&self) -> &[DirectedEdge] {
        &self.directed_edges
    }

    fn row_edge_ids(&self, node_id: usize) -> &[usize] {
        self.row_edge_ids.get(node_id).map_or(&[], Vec::as_slice)
    }

    fn directed_edge_bond_kind(&self, edge_id: usize) -> u8 {
        self.bond_kinds[edge_id]
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    fn directed_edge_topology(&self) -> DirectedEdgeTopology {
        let mut neighbors_by_node = Vec::with_capacity(self.atom_nodes.len());
        let mut bond_kinds_by_node = Vec::with_capacity(self.atom_nodes.len());

        for from in 0..self.atom_nodes.len() {
            let neighbors: Vec<usize> = self.bond_matrix.sparse_row(from).collect();
            let bond_kinds: Vec<u8> = self
                .bond_matrix
                .sparse_row_values_ref(from)
                .map(|entry| bond_entry_code(*entry))
                .collect();
            neighbors_by_node.push(neighbors);
            bond_kinds_by_node.push(bond_kinds);
        }

        let mut row_edge_ids = Vec::with_capacity(self.atom_nodes.len());
        let mut directed_edges = Vec::with_capacity(self.number_of_bonds() * 2);
        let mut bond_kinds = Vec::with_capacity(self.number_of_bonds() * 2);

        for (from, neighbors) in neighbors_by_node.iter().enumerate() {
            let mut row_ids = Vec::with_capacity(neighbors.len());
            for (position, &to) in neighbors.iter().enumerate() {
                let reverse_position =
                    neighbors_by_node[to].binary_search(&from).unwrap_or_else(|_| unreachable!());
                let reverse_edge_id = row_edge_ids
                    .get(to)
                    .and_then(|row: &Vec<usize>| row.get(reverse_position))
                    .copied();
                let edge_id = directed_edges.len();
                directed_edges.push(DirectedEdge {
                    from,
                    to,
                    reverse: reverse_edge_id.unwrap_or(usize::MAX),
                });
                bond_kinds.push(bond_kinds_by_node[from][position]);
                row_ids.push(edge_id);
                if let Some(reverse_edge_id) = reverse_edge_id {
                    directed_edges[edge_id].reverse = reverse_edge_id;
                    directed_edges[reverse_edge_id].reverse = edge_id;
                }
            }
            row_edge_ids.push(row_ids);
        }

        DirectedEdgeTopology { directed_edges, row_edge_ids, bond_kinds }
    }
}

fn dense_ranks_by_current_and_neighborhood_reuse<N>(
    current_classes: &[usize],
    neighborhoods: &[N],
    indices: &mut [usize],
    ranks: &mut [usize],
) where
    N: Ord,
{
    for (position, index) in indices.iter_mut().enumerate() {
        *index = position;
    }
    indices.sort_unstable_by(|&left_index, &right_index| {
        current_classes[left_index]
            .cmp(&current_classes[right_index])
            .then_with(|| neighborhoods[left_index].cmp(&neighborhoods[right_index]))
            .then_with(|| left_index.cmp(&right_index))
    });

    let mut current_rank = 0usize;

    for (offset, &index) in indices.iter().enumerate() {
        if offset > 0 {
            let previous = indices[offset - 1];
            if current_classes[previous] != current_classes[index]
                || neighborhoods[previous] != neighborhoods[index]
            {
                current_rank += 1;
            }
        }
        ranks[index] = current_rank;
    }
}

fn classes_are_unique(classes: &[usize]) -> bool {
    classes.iter().copied().max().is_none_or(|max_class| max_class + 1 == classes.len())
}

#[cfg(test)]
mod tests {
    use super::Smiles;

    fn parse(smiles: &str) -> Smiles {
        smiles.parse().unwrap()
    }

    fn assert_same_partition(actual: &[usize], expected: &[usize]) {
        assert_eq!(actual.len(), expected.len());
        for left in 0..actual.len() {
            for right in left..actual.len() {
                assert_eq!(
                    actual[left] == actual[right],
                    expected[left] == expected[right],
                    "partition mismatch at pair ({left}, {right}): actual={actual:?} expected={expected:?}",
                );
            }
        }
    }

    #[test]
    fn rooted_symmetry_classes_of_empty_graph_are_empty() {
        assert!(
            Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy()
                .rooted_symmetry_classes()
                .is_empty()
        );
    }

    #[test]
    fn rooted_symmetry_classes_preserve_path_symmetry() {
        let classes = parse("CCC").rooted_symmetry_classes();
        assert_eq!(classes.len(), 3);
        assert_eq!(classes[0], classes[2]);
        assert_ne!(classes[0], classes[1]);
    }

    #[test]
    fn rooted_symmetry_classes_keep_benzene_atoms_equivalent() {
        let classes = parse("c1ccccc1").rooted_symmetry_classes();
        assert_eq!(classes.len(), 6);
        assert!(classes.iter().all(|&class| class == classes[0]));
    }

    #[test]
    fn rooted_symmetry_classes_do_not_merge_refined_classes() {
        let smiles = parse("C=CO");
        let refined = smiles.refined_atom_classes().classes().to_vec();
        let rooted = smiles.rooted_symmetry_classes();

        for left in 0..rooted.len() {
            for right in (left + 1)..rooted.len() {
                if refined[left] != refined[right] {
                    assert_ne!(rooted[left], rooted[right]);
                }
            }
        }
    }

    #[test]
    fn rooted_symmetry_classes_preserve_neighbor_based_separation() {
        let smiles = parse("CCCO");
        let rooted = smiles.rooted_symmetry_classes();
        assert_eq!(rooted.len(), 4);
        assert_ne!(rooted[1], rooted[2]);
    }

    #[test]
    fn rooted_symmetry_classes_match_rdkit_reference_partitions_for_basic_cases() {
        assert_same_partition(&parse("CCC").rooted_symmetry_classes(), &[0, 2, 0]);
        assert_same_partition(&parse("c1ccccc1").rooted_symmetry_classes(), &[0, 0, 0, 0, 0, 0]);
        assert_same_partition(&parse("CCCO").rooted_symmetry_classes(), &[0, 2, 3, 1]);
        assert_same_partition(&parse("C=CO").rooted_symmetry_classes(), &[0, 2, 1]);
        assert_same_partition(
            &parse("C1NCN1.C1NCN1").rooted_symmetry_classes(),
            &[0, 4, 0, 4, 0, 4, 0, 4],
        );
    }

    #[test]
    fn rooted_symmetry_classes_match_rdkit_reference_partitions_for_symmetric_cages() {
        assert_same_partition(
            &parse("C1CCC2CC2C1").rooted_symmetry_classes(),
            &[0, 0, 2, 5, 4, 5, 2],
        );
        assert_same_partition(
            &parse("C1C2CC2C3C1C3").rooted_symmetry_classes(),
            &[0, 3, 1, 5, 5, 3, 1],
        );
        assert_same_partition(
            &parse("C1C2CC3C1C3C2").rooted_symmetry_classes(),
            &[0, 3, 0, 4, 4, 4, 0],
        );
        assert_same_partition(
            &parse("C1CC2(CCC1C2)Cl").rooted_symmetry_classes(),
            &[1, 3, 7, 3, 1, 6, 5, 0],
        );
    }

    #[test]
    fn rooted_symmetry_classes_match_rdkit_reference_partitions_for_bridged_cases() {
        assert_same_partition(
            &parse("B1C2CCCC1CCC2").rooted_symmetry_classes(),
            &[0, 7, 3, 1, 3, 7, 3, 1, 3],
        );
        assert_same_partition(
            &parse("CN1C2CCCC1CNC2").rooted_symmetry_classes(),
            &[0, 9, 7, 2, 1, 2, 7, 4, 6, 4],
        );
        assert_same_partition(
            &parse("C1N2CN3CN1CN(C2)C3").rooted_symmetry_classes(),
            &[0, 6, 0, 6, 0, 6, 0, 6, 0, 0],
        );
    }
}
