use alloc::vec::Vec;

use geometric_traits::traits::{SparseMatrix2D, SparseValuedMatrix2DRef};
use smallvec::SmallVec;

use super::{
    Smiles,
    invariants::{AtomInvariant, bond_entry_code, planning_chirality_key},
};
use crate::{atom::AtomSyntax, bond::Bond};

type Neighborhood = SmallVec<[(u8, usize); 4]>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AtomPartition {
    classes: Vec<usize>,
}

impl AtomPartition {
    #[inline]
    #[must_use]
    pub(crate) fn classes(&self) -> &[usize] {
        &self.classes
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn class_of(&self, node_id: usize) -> Option<usize> {
        self.classes.get(node_id).copied()
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn number_of_classes(&self) -> usize {
        self.classes.iter().copied().max().map_or(0, |max_class| max_class + 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AtomInvariantKey {
    syntax: u8,
    symbol_kind: u8,
    atomic_number: u8,
    isotope_mass_number: Option<u16>,
    aromatic: bool,
    hydrogens: u8,
    charge: i8,
    class: u16,
    chirality_kind: u8,
    chirality_value: u8,
    degree: usize,
    bond_kind_histogram: [usize; 5],
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn refined_atom_classes(&self) -> AtomPartition {
        let invariants = self.atom_invariants();
        self.refined_atom_classes_from_invariants(&invariants)
    }

    #[must_use]
    pub(crate) fn refined_atom_classes_from_invariants(
        &self,
        invariants: &[AtomInvariant],
    ) -> AtomPartition {
        let seed_colors: Vec<AtomInvariantKey> =
            invariants.iter().copied().map(AtomInvariantKey::from).collect();
        let adjacency = self.refinement_adjacency();
        let mut classes = dense_ranks(&seed_colors);
        if classes_are_unique(&classes) {
            return AtomPartition { classes };
        }
        let mut neighborhoods: Vec<Neighborhood> =
            adjacency.iter().map(|neighbors| SmallVec::with_capacity(neighbors.len())).collect();
        let mut indices: Vec<usize> = (0..classes.len()).collect();
        let mut next_classes = vec![0usize; classes.len()];

        loop {
            for (node_id, neighbors) in adjacency.iter().enumerate() {
                let neighborhood = &mut neighborhoods[node_id];
                neighborhood.clear();
                neighborhood.extend(
                    neighbors
                        .iter()
                        .map(|&(neighbor_id, bond_kind)| (bond_kind, classes[neighbor_id])),
                );
                neighborhood.sort_unstable();
            }

            dense_ranks_by_current_and_neighborhood_reuse(
                &classes,
                &neighborhoods,
                &mut indices,
                &mut next_classes,
            );
            if next_classes == classes {
                break;
            }
            core::mem::swap(&mut classes, &mut next_classes);
        }

        AtomPartition { classes }
    }

    fn refinement_adjacency(&self) -> Vec<Vec<(usize, u8)>> {
        let mut adjacency = Vec::with_capacity(self.atom_nodes.len());
        for node_id in 0..self.atom_nodes.len() {
            let neighbors = self
                .bond_matrix
                .sparse_row(node_id)
                .zip(self.bond_matrix.sparse_row_values_ref(node_id))
                .map(|(neighbor_id, entry)| (neighbor_id, bond_entry_code(*entry)))
                .collect();
            adjacency.push(neighbors);
        }
        adjacency
    }
}

impl From<super::invariants::AtomInvariant> for AtomInvariantKey {
    fn from(value: super::invariants::AtomInvariant) -> Self {
        let (symbol_kind, atomic_number) = match value.symbol.element() {
            Some(element) => (0, u8::from(element)),
            None => (1, 0),
        };
        let (chirality_kind, chirality_value) = planning_chirality_key(value.chirality);

        Self {
            syntax: match value.syntax {
                AtomSyntax::OrganicSubset => 0,
                AtomSyntax::Bracket => 1,
            },
            symbol_kind,
            atomic_number,
            isotope_mass_number: value.isotope_mass_number,
            aromatic: value.aromatic,
            hydrogens: value.hydrogens,
            charge: value.charge,
            class: value.class,
            chirality_kind,
            chirality_value,
            degree: value.degree,
            bond_kind_histogram: [
                value.bond_kind_histogram.count(Bond::Single),
                value.bond_kind_histogram.count(Bond::Double),
                value.bond_kind_histogram.count(Bond::Triple),
                value.bond_kind_histogram.count(Bond::Quadruple),
                value.bond_kind_histogram.aromatic_count(),
            ],
        }
    }
}

fn dense_ranks<K>(keys: &[K]) -> Vec<usize>
where
    K: Ord,
{
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    indices.sort_unstable_by(|&left_index, &right_index| {
        keys[left_index].cmp(&keys[right_index]).then_with(|| left_index.cmp(&right_index))
    });

    let mut ranks = vec![0usize; keys.len()];
    let mut current_rank = 0usize;

    for (offset, &index) in indices.iter().enumerate() {
        if offset > 0 && keys[indices[offset - 1]] != keys[indices[offset]] {
            current_rank += 1;
        }
        ranks[index] = current_rank;
    }

    ranks
}

fn classes_are_unique(classes: &[usize]) -> bool {
    classes.iter().copied().max().is_none_or(|max_class| max_class + 1 == classes.len())
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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{AtomPartition, Smiles};

    fn parse(smiles: &str) -> Smiles {
        smiles.parse().unwrap()
    }

    #[test]
    fn partition_refinement_of_empty_graph_is_empty() {
        let partition =
            Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy().refined_atom_classes();
        assert_eq!(partition, AtomPartition { classes: Vec::new() });
    }

    #[test]
    fn partition_refinement_preserves_obvious_symmetry() {
        let partition = parse("CCC").refined_atom_classes();
        assert_eq!(partition.classes().len(), 3);
        assert_eq!(partition.class_of(0), partition.class_of(2));
        assert_ne!(partition.class_of(0), partition.class_of(1));
        assert!(partition.number_of_classes() >= 2);
    }

    #[test]
    fn partition_refinement_separates_atoms_with_same_local_invariant_but_different_neighbors() {
        let smiles = parse("CCCO");

        let local_middle = smiles.atom_invariant(1).unwrap();
        let local_next_to_oxygen = smiles.atom_invariant(2).unwrap();
        assert_eq!(local_middle, local_next_to_oxygen);

        let partition = smiles.refined_atom_classes();
        assert_ne!(partition.class_of(1), partition.class_of(2));
    }

    #[test]
    fn partition_refinement_keeps_benzene_atoms_equivalent() {
        let partition = parse("c1ccccc1").refined_atom_classes();
        let first = partition.class_of(0).unwrap();
        assert!(partition.classes().iter().all(|&class| class == first));
        assert_eq!(partition.number_of_classes(), 1);
    }

    #[test]
    fn partition_refinement_distinguishes_bond_neighborhoods() {
        let smiles = parse("C=CO");
        let partition = smiles.refined_atom_classes();
        assert_ne!(partition.class_of(0), partition.class_of(1));
        assert_ne!(partition.class_of(1), partition.class_of(2));
    }
}
