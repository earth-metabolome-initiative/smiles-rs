use alloc::vec::Vec;
use core::cmp::Ordering;

use hashbrown::HashMap;
use smallvec::SmallVec;

use super::{
    Smiles,
    invariants::{AtomInvariant, bond_descriptor_index, planning_chirality_key},
    spanning_tree::SpanningForest,
};

type ChildSignatureVec = SmallVec<[(usize, usize); 4]>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BranchPlan {
    ordered_children: Vec<Vec<usize>>,
    subtree_signatures: Vec<usize>,
}

impl BranchPlan {
    #[must_use]
    pub(crate) fn ordered_children(&self, node_id: usize) -> &[usize] {
        self.ordered_children.get(node_id).map_or(&[], Vec::as_slice)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn continuation_child(&self, node_id: usize) -> Option<usize> {
        self.ordered_children(node_id).last().copied()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn branch_children(&self, node_id: usize) -> &[usize] {
        self.ordered_children(node_id)
            .split_last()
            .map_or(&[], |(_continuation, branches)| branches)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn subtree_signature(&self, node_id: usize) -> Option<usize> {
        self.subtree_signatures.get(node_id).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NodeLocalKey {
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
    rooted_class: usize,
    refined_class: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SubtreeSignatureKey {
    local: NodeLocalKey,
    children: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChildOrderKey {
    bond_kind: usize,
    subtree_signature: usize,
    rooted_class: usize,
    refined_class: usize,
    local: NodeLocalKey,
    node_id: usize,
}

impl Ord for ChildOrderKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bond_kind
            .cmp(&other.bond_kind)
            .then_with(|| self.subtree_signature.cmp(&other.subtree_signature))
            .then_with(|| self.rooted_class.cmp(&other.rooted_class))
            .then_with(|| self.refined_class.cmp(&other.refined_class))
            .then_with(|| self.local.cmp(&other.local))
            .then_with(|| self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for ChildOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn branch_plan(&self) -> BranchPlan {
        let invariants = self.atom_invariants();
        let refined = self.refined_atom_classes_from_invariants(&invariants);
        let refined_classes = refined.classes().to_vec();
        let rooted_classes = self.rooted_symmetry_classes_from_refined(&refined_classes);
        let roots = self.component_roots_from_planning(&invariants, &rooted_classes);
        let forest = self.spanning_forest_with_planning(
            &roots,
            &invariants,
            &refined_classes,
            &rooted_classes,
        );
        self.branch_plan_with_planning(&forest, &invariants, &refined_classes, &rooted_classes)
    }

    #[must_use]
    pub(crate) fn branch_plan_with_planning(
        &self,
        forest: &SpanningForest,
        invariants: &[AtomInvariant],
        refined_classes: &[usize],
        rooted_classes: &[usize],
    ) -> BranchPlan {
        let node_count = self.nodes().len();
        if !forest.children().iter().any(|children| children.len() > 1) {
            return BranchPlan {
                ordered_children: forest.children().to_vec(),
                subtree_signatures: vec![0; node_count],
            };
        }

        let local_keys: Vec<NodeLocalKey> = (0..node_count)
            .map(|node_id| {
                node_local_key(
                    invariants[node_id],
                    rooted_classes[node_id],
                    refined_classes[node_id],
                )
            })
            .collect();

        let postorder = forest_postorder(forest, node_count);
        let mut subtree_signatures = vec![usize::MAX; node_count];
        let mut signature_ids: HashMap<SubtreeSignatureKey, usize> =
            HashMap::with_capacity(node_count);

        for &node_id in &postorder {
            let mut child_signatures =
                ChildSignatureVec::with_capacity(forest.children_of(node_id).len());
            for (&child, bond) in forest
                .children_of(node_id)
                .iter()
                .zip(forest.child_bonds_of(node_id).iter().copied())
            {
                child_signatures.push((bond_descriptor_index(bond), subtree_signatures[child]));
            }
            child_signatures.sort_unstable();

            let signature_key = SubtreeSignatureKey {
                local: local_keys[node_id].clone(),
                children: child_signatures.into_vec(),
            };
            let next_id = signature_ids.len();
            let signature_id = *signature_ids.entry(signature_key).or_insert(next_id);
            subtree_signatures[node_id] = signature_id;
        }

        let mut ordered_children = vec![Vec::new(); node_count];

        for (node_id, ordered_children_row) in ordered_children.iter_mut().enumerate() {
            let mut keyed_children = Vec::with_capacity(forest.children_of(node_id).len());
            for (&child, bond) in forest
                .children_of(node_id)
                .iter()
                .zip(forest.child_bonds_of(node_id).iter().copied())
            {
                keyed_children.push((
                    ChildOrderKey {
                        bond_kind: bond_descriptor_index(bond),
                        subtree_signature: subtree_signatures[child],
                        rooted_class: rooted_classes[child],
                        refined_class: refined_classes[child],
                        local: local_keys[child].clone(),
                        node_id: child,
                    },
                    child,
                ));
            }
            keyed_children.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            *ordered_children_row = keyed_children.into_iter().map(|(_key, child)| child).collect();
        }

        BranchPlan { ordered_children, subtree_signatures }
    }
}

fn node_local_key(
    invariant: AtomInvariant,
    rooted_class: usize,
    refined_class: usize,
) -> NodeLocalKey {
    let (symbol_kind, atomic_number) = match invariant.symbol.element() {
        Some(element) => (0, u8::from(element)),
        None => (1, 0),
    };
    let (chirality_kind, chirality_value) = planning_chirality_key(invariant.chirality);

    NodeLocalKey {
        syntax: match invariant.syntax {
            crate::atom::AtomSyntax::OrganicSubset => 0,
            crate::atom::AtomSyntax::Bracket => 1,
        },
        symbol_kind,
        atomic_number,
        isotope_mass_number: invariant.isotope_mass_number,
        aromatic: invariant.aromatic,
        hydrogens: invariant.hydrogens,
        charge: invariant.charge,
        class: invariant.class,
        chirality_kind,
        chirality_value,
        degree: invariant.degree,
        rooted_class,
        refined_class,
    }
}

fn forest_postorder(forest: &SpanningForest, node_count: usize) -> Vec<usize> {
    fn visit(node_id: usize, forest: &SpanningForest, visited: &mut [bool], out: &mut Vec<usize>) {
        if visited[node_id] {
            return;
        }
        visited[node_id] = true;
        for &child in forest.children_of(node_id) {
            visit(child, forest, visited, out);
        }
        out.push(node_id);
    }

    let mut visited = vec![false; node_count];
    let mut out = Vec::with_capacity(node_count);
    for &root in forest.roots() {
        visit(root, forest, &mut visited, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::cmp::Ordering;

    use elements_rs::Element;

    use super::{
        BranchPlan, ChildOrderKey, NodeLocalKey, Smiles, forest_postorder, node_local_key,
    };
    use crate::{
        atom::{Atom, AtomSyntax, atom_symbol::AtomSymbol, bracketed::chirality::Chirality},
        bond::{
            Bond,
            bond_edge::{BondEdge, bond_edge},
        },
        smiles::{
            BondMatrixBuilder,
            invariants::{AtomInvariant, BondKindHistogram},
        },
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

    fn plan(smiles: &str) -> BranchPlan {
        smiles.parse::<Smiles>().unwrap().branch_plan()
    }

    #[test]
    fn branch_plan_of_empty_graph_is_empty() {
        let plan = Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy().branch_plan();
        assert!(plan.ordered_children(0).is_empty());
        assert_eq!(plan.continuation_child(0), None);
        assert!(plan.branch_children(0).is_empty());
        assert_eq!(plan.subtree_signature(0), None);
    }

    #[test]
    fn branch_plan_keeps_single_child_as_continuation() {
        let plan = plan("CCCO");
        assert_eq!(plan.ordered_children(0), &[1]);
        assert_eq!(plan.continuation_child(0), Some(1));
        assert!(plan.branch_children(0).is_empty());
    }

    #[test]
    fn branch_plan_prefers_larger_subtree_as_continuation() {
        let smiles = smiles_from_edges(
            vec![
                atom(Element::B),
                atom(Element::C),
                atom(Element::C),
                atom(Element::C),
                atom(Element::C),
            ],
            &[
                bond_edge(0, 1, Bond::Single, None),
                bond_edge(0, 2, Bond::Single, None),
                bond_edge(0, 3, Bond::Single, None),
                bond_edge(3, 4, Bond::Single, None),
            ],
        );
        let plan = smiles.branch_plan();
        assert_eq!(plan.ordered_children(0), &[2, 3]);
        assert_eq!(plan.continuation_child(0), Some(3));
        assert_eq!(plan.branch_children(0), &[2]);
        assert_ne!(plan.subtree_signature(2), plan.subtree_signature(3));
    }

    #[test]
    fn branch_plan_uses_node_id_only_as_last_tie_break() {
        let smiles = smiles_from_edges(
            vec![atom(Element::B), atom(Element::C), atom(Element::C), atom(Element::C)],
            &[
                bond_edge(0, 1, Bond::Single, None),
                bond_edge(0, 2, Bond::Single, None),
                bond_edge(0, 3, Bond::Single, None),
            ],
        );
        let plan = smiles.branch_plan();
        assert_eq!(plan.ordered_children(0), &[2, 3]);
        assert_eq!(plan.continuation_child(0), Some(3));
        assert_eq!(plan.branch_children(0), &[2]);
        assert_eq!(plan.subtree_signature(2), plan.subtree_signature(3));
    }

    #[test]
    fn branch_plan_is_bond_sensitive_when_ordering_children() {
        let smiles = smiles_from_edges(
            vec![atom(Element::N), atom(Element::B), atom(Element::C), atom(Element::C)],
            &[
                bond_edge(0, 1, Bond::Single, None),
                bond_edge(0, 2, Bond::Single, None),
                bond_edge(0, 3, Bond::Double, None),
            ],
        );
        let plan = smiles.branch_plan();
        assert_eq!(plan.ordered_children(0), &[2, 3]);
        assert_eq!(plan.continuation_child(0), Some(3));
        assert_eq!(plan.branch_children(0), &[2]);
    }

    #[test]
    fn branch_plan_short_circuits_when_every_node_has_at_most_one_child() {
        let smiles: Smiles = "CCCO".parse().unwrap();
        let invariants = smiles.atom_invariants();
        let refined = smiles.refined_atom_classes_from_invariants(&invariants);
        let rooted = smiles.rooted_symmetry_classes_from_refined(refined.classes());
        let roots = smiles.component_roots_from_planning(&invariants, &rooted);
        let forest =
            smiles.spanning_forest_with_planning(&roots, &invariants, refined.classes(), &rooted);

        let plan =
            smiles.branch_plan_with_planning(&forest, &invariants, refined.classes(), &rooted);
        assert_eq!(plan.ordered_children(0), forest.children_of(0));
        assert_eq!(plan.subtree_signature(0), Some(0));
    }

    #[test]
    fn child_order_key_partial_cmp_delegates_to_total_cmp() {
        let left = ChildOrderKey {
            bond_kind: 0,
            subtree_signature: 1,
            rooted_class: 2,
            refined_class: 3,
            local: super::node_local_key(
                smiles_from_edges(vec![atom(Element::C)], &[]).atom_invariant(0).unwrap(),
                4,
                5,
            ),
            node_id: 6,
        };
        let right = ChildOrderKey { subtree_signature: 2, ..left.clone() };

        assert_eq!(left.partial_cmp(&right), Some(Ordering::Less));
        assert_eq!(right.partial_cmp(&left), Some(Ordering::Greater));
    }

    #[test]
    fn branch_plan_handles_wildcard_local_keys() {
        let smiles = smiles_from_edges(
            vec![
                Atom::builder().with_symbol(AtomSymbol::WildCard).build(),
                atom(Element::C),
                atom(Element::N),
            ],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(0, 2, Bond::Single, None)],
        );
        let plan = smiles.branch_plan();
        assert!(plan.subtree_signature(0).is_some());
        assert!(plan.subtree_signature(1).is_some());
        assert!(plan.subtree_signature(2).is_some());
    }

    #[test]
    fn forest_postorder_skips_duplicate_roots() {
        let smiles: Smiles = "CC".parse().unwrap();
        let forest = smiles.spanning_forest_with_parser_neighbor_order(&[0, 0]);
        assert_eq!(forest.roots(), &[0, 0]);
        assert_eq!(forest_postorder(&forest, smiles.nodes().len()), vec![1, 0]);
    }

    #[test]
    fn node_local_key_uses_wildcard_symbol_marker() {
        let key = node_local_key(
            AtomInvariant {
                syntax: AtomSyntax::Bracket,
                symbol: AtomSymbol::WildCard,
                isotope_mass_number: None,
                aromatic: false,
                hydrogens: 0,
                charge: 0,
                class: 0,
                chirality: Some(Chirality::AtAt),
                degree: 3,
                bond_kind_histogram: BondKindHistogram::default(),
            },
            7,
            11,
        );

        assert_eq!(
            key,
            NodeLocalKey {
                syntax: 1,
                symbol_kind: 1,
                atomic_number: 0,
                isotope_mass_number: None,
                aromatic: false,
                hydrogens: 0,
                charge: 0,
                class: 0,
                chirality_kind: 1,
                chirality_value: 0,
                degree: 3,
                rooted_class: 7,
                refined_class: 11,
            }
        );
    }
}
