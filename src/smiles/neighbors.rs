use alloc::vec::Vec;

use geometric_traits::traits::{SparseMatrix2D, SparseValuedMatrix2DRef};

use super::Smiles;
use crate::{
    atom::AtomSyntax,
    bond::bond_edge::BondEdge,
    smiles::invariants::{AtomInvariant, bond_entry_code, planning_chirality_key},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NeighborOrderKey {
    rooted_class: usize,
    refined_class: usize,
    bond_kind: usize,
    degree: usize,
    symbol_kind: u8,
    atomic_number: u8,
    isotope_mass_number: Option<u16>,
    hydrogens: u8,
    charge: i8,
    aromatic: bool,
    syntax: u8,
    chirality_kind: u8,
    chirality_value: u8,
    atom_class: u16,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn ordered_neighbor_edges(&self, node_id: usize) -> Vec<BondEdge> {
        let invariants = self.atom_invariants();
        let refined = self.refined_atom_classes_from_invariants(&invariants);
        let rooted = self.rooted_symmetry_classes_from_refined(refined.classes());
        self.ordered_neighbor_edges_with_planning(node_id, &invariants, refined.classes(), &rooted)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn ordered_neighbor_ids(&self, node_id: usize) -> Vec<usize> {
        let invariants = self.atom_invariants();
        let refined = self.refined_atom_classes_from_invariants(&invariants);
        let rooted = self.rooted_symmetry_classes_from_refined(refined.classes());
        self.ordered_neighbor_edges_with_planning(node_id, &invariants, refined.classes(), &rooted)
            .into_iter()
            .map(|edge| {
                crate::bond::bond_edge::bond_edge_other(edge, node_id)
                    .unwrap_or_else(|| unreachable!())
            })
            .collect()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn ordered_neighbor_edges_with_planning(
        &self,
        node_id: usize,
        invariants: &[AtomInvariant],
        refined_classes: &[usize],
        rooted_classes: &[usize],
    ) -> Vec<BondEdge> {
        if node_id >= self.atom_nodes.len() {
            return Vec::new();
        }

        self.ordered_neighbor_edges_table_with_planning(invariants, refined_classes, rooted_classes)
            .into_iter()
            .nth(node_id)
            .unwrap_or_default()
    }

    #[must_use]
    pub(crate) fn ordered_neighbor_edges_table_with_planning(
        &self,
        invariants: &[AtomInvariant],
        refined_classes: &[usize],
        rooted_classes: &[usize],
    ) -> Vec<Vec<BondEdge>> {
        let mut ordered_rows = Vec::with_capacity(self.atom_nodes.len());
        for node_id in 0..self.atom_nodes.len() {
            ordered_rows.push(self.ordered_neighbor_edges_row_with_planning(
                node_id,
                invariants,
                refined_classes,
                rooted_classes,
            ));
        }
        ordered_rows
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    fn ordered_neighbor_edges_row_with_planning(
        &self,
        node_id: usize,
        invariants: &[AtomInvariant],
        refined_classes: &[usize],
        rooted_classes: &[usize],
    ) -> Vec<BondEdge> {
        let mut neighbors: Vec<(NeighborOrderKey, usize, usize, BondEdge)> = self
            .bond_matrix
            .sparse_row(node_id)
            .zip(self.bond_matrix.sparse_row_values_ref(node_id))
            .map(|(neighbor_id, entry)| {
                let key = neighbor_order_key(
                    invariants[neighbor_id],
                    refined_classes[neighbor_id],
                    rooted_classes[neighbor_id],
                    usize::from(bond_entry_code(*entry)),
                );
                (key, entry.order(), neighbor_id, entry.to_bond_edge(node_id, neighbor_id))
            })
            .collect();

        neighbors.sort_unstable_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        neighbors.into_iter().map(|(_, _, _, edge)| edge).collect()
    }
}

fn neighbor_order_key(
    invariant: AtomInvariant,
    refined_class: usize,
    rooted_class: usize,
    bond_kind: usize,
) -> NeighborOrderKey {
    let (symbol_kind, atomic_number) = match invariant.symbol.element() {
        Some(element) => (0, u8::from(element)),
        None => (1, u8::MAX),
    };
    let (chirality_kind, chirality_value) = planning_chirality_key(invariant.chirality);

    NeighborOrderKey {
        rooted_class,
        refined_class,
        bond_kind,
        degree: invariant.degree,
        symbol_kind,
        atomic_number,
        isotope_mass_number: invariant.isotope_mass_number,
        hydrogens: invariant.hydrogens,
        charge: invariant.charge,
        aromatic: invariant.aromatic,
        syntax: match invariant.syntax {
            AtomSyntax::OrganicSubset => 0,
            AtomSyntax::Bracket => 1,
        },
        chirality_kind,
        chirality_value,
        atom_class: invariant.class,
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use elements_rs::Element;

    use super::Smiles;
    use crate::{
        atom::{Atom, atom_symbol::AtomSymbol},
        bond::{
            Bond,
            bond_edge::{BondEdge, bond_edge},
        },
        smiles::BondMatrixBuilder,
    };

    fn atom(element: Element) -> Atom {
        Atom::new_organic_subset(AtomSymbol::Element(element), false)
    }

    fn smiles_from_edges(atom_nodes: Vec<Atom>, bond_edges: &[BondEdge]) -> Smiles {
        let mut builder = BondMatrixBuilder::with_capacity(bond_edges.len());
        for edge in bond_edges {
            let descriptor = edge.descriptor();
            builder
                .push_edge_with_descriptor(
                    edge.source(),
                    edge.target(),
                    descriptor,
                    edge.ring_num(),
                )
                .unwrap();
        }
        let number_of_nodes = atom_nodes.len();
        Smiles::from_bond_matrix_parts(atom_nodes, builder.finish(number_of_nodes))
    }

    #[test]
    fn ordered_neighbors_of_empty_or_missing_node_are_empty() {
        let smiles = Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy();
        assert!(smiles.ordered_neighbor_edges(0).is_empty());
        assert!(smiles.ordered_neighbor_ids(0).is_empty());
    }

    #[test]
    fn ordered_neighbors_preserve_all_incident_edges() {
        let smiles = smiles_from_edges(
            vec![atom(Element::C), atom(Element::O), atom(Element::N)],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(0, 2, Bond::Double, None)],
        );

        let ordered = smiles.ordered_neighbor_edges(0);
        assert_eq!(ordered.len(), 2);
        assert!(ordered.contains(&bond_edge(0, 1, Bond::Single, None)));
        assert!(ordered.contains(&bond_edge(0, 2, Bond::Double, None)));
    }

    #[test]
    fn ordered_neighbors_break_symmetric_ties_by_node_id_last() {
        let smiles: Smiles = "CCC".parse().unwrap();
        assert_eq!(smiles.ordered_neighbor_ids(1), vec![0, 2]);
    }

    #[test]
    fn ordered_neighbors_use_structural_classes_before_node_id() {
        let smiles: Smiles = "CCCO".parse().unwrap();
        assert_eq!(smiles.ordered_neighbor_ids(1), vec![0, 2]);
        assert_eq!(smiles.ordered_neighbor_ids(2), vec![1, 3]);
    }

    #[test]
    fn ordered_neighbors_keep_equivalent_ring_neighbors_stable() {
        let smiles: Smiles = "C1CCC2CC2C1".parse().unwrap();
        assert_eq!(smiles.ordered_neighbor_ids(0), vec![1, 6]);
    }

    #[test]
    fn ordered_neighbors_reflect_bond_sensitive_structural_classes() {
        let smiles = smiles_from_edges(
            vec![atom(Element::C), atom(Element::O), atom(Element::O)],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(0, 2, Bond::Double, None)],
        );

        assert_eq!(smiles.ordered_neighbor_ids(0), vec![2, 1]);
    }

    #[test]
    fn ordered_neighbors_are_stable_for_symmetric_four_cycle() {
        let smiles: Smiles = "C1NCN1".parse().unwrap();
        assert_eq!(smiles.ordered_neighbor_ids(0), vec![1, 3]);
        assert_eq!(smiles.ordered_neighbor_ids(2), vec![1, 3]);
    }

    #[test]
    fn ordered_neighbors_are_local_to_each_component() {
        let smiles: Smiles = "C1NCN1.C1NCN1".parse().unwrap();
        assert_eq!(smiles.ordered_neighbor_ids(0), vec![1, 3]);
        assert_eq!(smiles.ordered_neighbor_ids(4), vec![5, 7]);
    }

    #[test]
    fn ordered_neighbors_handle_additional_symmetric_cage_fixture() {
        let smiles: Smiles = "C1C2CC2C3C1C3".parse().unwrap();
        assert_eq!(smiles.ordered_neighbor_ids(0), vec![1, 5]);
    }
}
