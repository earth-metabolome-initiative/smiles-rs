use alloc::vec::Vec;

use super::Smiles;
use crate::{
    atom::AtomSyntax,
    smiles::invariants::{AtomInvariant, planning_chirality_key},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ComponentRootKey {
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
    rooted_class: usize,
    rooted_partition_size: usize,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn component_roots(&self) -> Vec<usize> {
        let invariants = self.atom_invariants();
        let rooted_classes = self.rooted_symmetry_classes();
        self.component_roots_from_planning(&invariants, &rooted_classes)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn root_of_component(&self, component_id: usize) -> Option<usize> {
        self.component_roots().get(component_id).copied()
    }

    fn component_root_key(
        invariants: &[AtomInvariant],
        rooted_classes: &[usize],
        rooted_partition_sizes: &[usize],
        node_id: usize,
    ) -> ComponentRootKey {
        let invariant = invariants.get(node_id).copied().unwrap_or_else(|| unreachable!());
        component_root_key(invariant, rooted_classes[node_id], rooted_partition_sizes)
    }

    #[must_use]
    pub(crate) fn component_roots_from_planning(
        &self,
        invariants: &[AtomInvariant],
        rooted_classes: &[usize],
    ) -> Vec<usize> {
        let rooted_partition_sizes = rooted_partition_sizes(rooted_classes);
        let components = self.connected_components();
        let mut roots = Vec::with_capacity(components.number_of_components());

        for component_id in 0..components.number_of_components() {
            let component_nodes: Vec<usize> =
                components.node_ids_of_component(component_id).collect();
            let first = component_nodes[0];
            let rest = &component_nodes[1..];
            let mut best = first;
            let mut best_key =
                Self::component_root_key(invariants, rooted_classes, &rooted_partition_sizes, best);

            for &node_id in rest {
                let candidate_key = Self::component_root_key(
                    invariants,
                    rooted_classes,
                    &rooted_partition_sizes,
                    node_id,
                );
                if candidate_key < best_key || (candidate_key == best_key && node_id < best) {
                    best = node_id;
                    best_key = candidate_key;
                }
            }
            roots.push(best);
        }

        roots
    }
}

fn component_root_key(
    invariant: AtomInvariant,
    rooted_class: usize,
    rooted_partition_sizes: &[usize],
) -> ComponentRootKey {
    let (symbol_kind, atomic_number) = match invariant.symbol.element() {
        Some(element) => (0, u8::from(element)),
        None => (1, u8::MAX),
    };
    let (chirality_kind, chirality_value) = planning_chirality_key(invariant.chirality);

    ComponentRootKey {
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
        rooted_class,
        rooted_partition_size: rooted_partition_sizes[rooted_class],
    }
}

fn rooted_partition_sizes(rooted_classes: &[usize]) -> Vec<usize> {
    let max_class = rooted_classes.iter().copied().max().map_or(0, |class| class + 1);
    let mut sizes = vec![0; max_class];
    for &class in rooted_classes {
        sizes[class] += 1;
    }
    sizes
}

#[cfg(test)]
mod tests {
    use super::Smiles;

    fn parse(smiles: &str) -> Smiles {
        smiles.parse().unwrap()
    }

    fn assert_root_matches_rdkit(smiles: &str, tied_partition: &[usize], rdkit_roots: &[usize]) {
        let parsed = parse(smiles);
        let actual_roots = parsed.component_roots();
        assert_eq!(actual_roots.len(), rdkit_roots.len(), "{smiles}");

        for (actual_root, rdkit_root) in actual_roots.into_iter().zip(rdkit_roots.iter().copied()) {
            assert_eq!(
                tied_partition[actual_root], tied_partition[rdkit_root],
                "root should stay inside the RDKit untied partition for {smiles}: actual={actual_root} rdkit={rdkit_root} tied={tied_partition:?}",
            );

            let expected_partition_size =
                tied_partition.iter().filter(|&&class| class == tied_partition[rdkit_root]).count();
            if expected_partition_size == 1 {
                assert_eq!(
                    actual_root, rdkit_root,
                    "root should match unique RDKit winner for {smiles}"
                );
            }
        }
    }

    #[test]
    fn component_roots_of_empty_graph_are_empty() {
        let smiles = Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy();
        assert!(smiles.component_roots().is_empty());
        assert_eq!(smiles.root_of_component(0), None);
    }

    #[test]
    fn component_roots_are_selected_per_component() {
        let smiles = parse("CC(=O)[O-].[NH3+]C");
        assert_eq!(smiles.component_roots(), vec![0, 5]);
        assert_eq!(smiles.root_of_component(0), Some(0));
        assert_eq!(smiles.root_of_component(1), Some(5));
        assert_eq!(smiles.root_of_component(2), None);
    }

    #[test]
    fn component_roots_match_rdkit_reference_for_basic_cases() {
        assert_root_matches_rdkit("CCC", &[0, 2, 0], &[0]);
        assert_root_matches_rdkit("CCCO", &[0, 2, 3, 1], &[0]);
        assert_root_matches_rdkit("C=CO", &[0, 2, 1], &[0]);
        assert_root_matches_rdkit("c1ccccc1", &[0, 0, 0, 0, 0, 0], &[2]);
        assert_root_matches_rdkit("CC(=O)[O-].[NH3+]C", &[1, 5, 3, 4, 2, 0], &[0, 5]);
        assert_root_matches_rdkit("Cl.C1CC1.O", &[1, 2, 2, 2, 0], &[0, 1, 4]);
    }

    #[test]
    fn component_roots_match_rdkit_reference_for_symmetric_cages() {
        assert_root_matches_rdkit("C1CCC2CC2C1", &[0, 0, 2, 5, 4, 5, 2], &[0]);
        assert_root_matches_rdkit("C1C2CC2C3C1C3", &[0, 3, 1, 5, 5, 3, 1], &[0]);
        assert_root_matches_rdkit("C1C2CC3C1C3C2", &[0, 3, 0, 4, 4, 4, 0], &[0]);
        assert_root_matches_rdkit("C1CC2(CCC1C2)Cl", &[1, 3, 7, 3, 1, 6, 5, 0], &[7]);
    }

    #[test]
    fn component_roots_match_rdkit_reference_for_bridged_cases() {
        assert_root_matches_rdkit("B1C2CCCC1CCC2", &[0, 7, 3, 1, 3, 7, 3, 1, 3], &[0]);
        assert_root_matches_rdkit("CN1C2CCCC1CNC2", &[0, 9, 7, 2, 1, 2, 7, 4, 6, 4], &[0]);
        assert_root_matches_rdkit("C1N2CN3CN1CN(C2)C3", &[0, 6, 0, 6, 0, 6, 0, 6, 0, 0], &[0]);
        assert_root_matches_rdkit("C1NCN1.C1NCN1", &[0, 4, 0, 4, 0, 4, 0, 4], &[0, 4]);
    }
}
