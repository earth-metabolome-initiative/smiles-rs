use alloc::{string::String, vec::Vec};
use core::{cell::OnceCell, fmt::Write};

use super::{Smiles, SmilesAtomPolicy, SmilesCanonicalLabeling, render_plan::RenderPlan};

/// Renders a [`Smiles`] graph by first building a [`RenderPlan`] and then
/// emitting text from that plan.
///
/// This is the top-level display helper used by [`Smiles::render`]. All
/// structural choices must already be encoded in the plan; this function is a
/// pure write-only pass.
#[must_use]
pub(crate) fn emit<AtomPolicy: SmilesAtomPolicy>(smiles: &Smiles<AtomPolicy>) -> String {
    orbit_min(smiles, None).0
}

/// Renders `smiles` with one component's traversal forced to start at `root`.
#[must_use]
pub(crate) fn emit_rooted<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    root: usize,
) -> String {
    orbit_min(smiles, Some(root)).0
}

/// Returns the traversal order (original atom ids) backing
/// [`emit_rooted`]'s output.
#[must_use]
pub(crate) fn rooted_traversal_order<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    root: usize,
) -> Vec<usize> {
    orbit_min(smiles, Some(root)).1
}

/// Returns the lexicographically smallest `(string, original_atom_order)`
/// reachable by repeatedly relabeling `smiles` to its exact canonical form
/// (`root`-individualized when given) and rendering that, until the sequence
/// of rendered strings cycles.
///
/// WL refinement (`rooted_classes`) cannot always break ties between atoms
/// that are genuinely interchangeable under a molecular symmetry (a true
/// graph automorphism), and even the exact canonizer's own tie-break for a
/// residual automorphism (one `root` individualization does not touch, or
/// any automorphism at all when there is no `root`) is only consistent up to
/// that automorphism: which orbit member it ranks first is free to differ
/// between two isomorphic copies of the same graph, which is exactly what a
/// plain parse and a reparse of its own rendered text are. A single
/// relabel-and-render pass cannot detect that, so this mirrors
/// `Smiles::canonicalize`'s `canonicalize_orbit_min` exactly: the state
/// sequence is seeded with two rounds of relabeling (never with `smiles`'s
/// own, un-relabeled numbering, whose render can coincide with the first
/// relabeled round purely because both reach for the same exact-labeling
/// primitive, which would wrongly look like an immediate fixed point and
/// stop the search one hop short of the orbit it was meant to explore), and
/// only grows further when those two rounds already disagree. The loop only
/// runs when WL refinement leaves a tie, so molecules without one keep paying
/// for a single render pass. A second round whose state equals the first is
/// returned unrendered, since it renders identically.
fn orbit_min<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    root: Option<usize>,
) -> (String, Vec<usize>) {
    let classes = smiles.planning_classes();
    if !classes.has_ties() {
        let plan = smiles.render_plan_with(classes, root, None);
        return (emit_with_plan(smiles, &plan), plan.traversal_order());
    }

    let identity_order: Vec<usize> = (0..smiles.nodes().len()).collect();
    let first = relabel_round(smiles, &smiles.tie_break_labeling(root), root, &identity_order);
    let first_string = render_round(&first);
    let second = first.next();
    if second.has_same_state(&first) {
        return (first_string, first.order);
    }
    let second_string = render_round(&second);

    if second_string == first_string {
        return (first_string, first.order);
    }

    let mut strings = alloc::vec![first_string, second_string];
    let mut rounds = alloc::vec![first, second];

    loop {
        let previous = rounds.last().unwrap_or_else(|| unreachable!("rounds is non-empty"));
        let next = previous.next();
        let next_string = render_round(&next);

        if let Some(cycle_start) = strings.iter().position(|existing| *existing == next_string) {
            let best = (cycle_start..strings.len())
                .min_by(|&left, &right| strings[left].cmp(&strings[right]))
                .unwrap_or_else(|| unreachable!("cycle slice is non-empty"));
            return (strings.swap_remove(best), rounds.swap_remove(best).order);
        }

        strings.push(next_string);
        rounds.push(next);
    }
}

/// One relabeled state in [`orbit_min`]'s search: the relabeled graph, where
/// `root` (if any) landed in it, the original atom id behind each of its
/// local atom ids, and its own [`Smiles::tie_break_labeling`] for `root`,
/// computed on first use.
struct RelabelRound<AtomPolicy: SmilesAtomPolicy> {
    graph: Smiles<AtomPolicy>,
    root: Option<usize>,
    order: Vec<usize>,
    labeling: OnceCell<SmilesCanonicalLabeling>,
}

impl<AtomPolicy: SmilesAtomPolicy> RelabelRound<AtomPolicy> {
    fn labeling(&self) -> &SmilesCanonicalLabeling {
        self.labeling.get_or_init(|| self.graph.tie_break_labeling(self.root))
    }

    fn next(&self) -> Self {
        relabel_round(&self.graph, self.labeling(), self.root, &self.order)
    }

    /// Whether both rounds hold the same graph state and root, which makes
    /// every deterministic function of a round, its render included, agree.
    fn has_same_state(&self, other: &Self) -> bool {
        self.root == other.root && same_graph_state(&self.graph, &other.graph)
    }
}

/// Compares every field of two graphs, where `PartialEq` compares only atoms
/// and bonds.
fn same_graph_state<AtomPolicy: SmilesAtomPolicy>(
    left: &Smiles<AtomPolicy>,
    right: &Smiles<AtomPolicy>,
) -> bool {
    let Smiles {
        atom_nodes,
        bond_matrix,
        parsed_stereo_neighbors,
        implicit_hydrogen_cache,
        radical_electrons,
        kekulization_source,
        atom_policy: _,
    } = left;
    *atom_nodes == right.atom_nodes
        && *bond_matrix == right.bond_matrix
        && *parsed_stereo_neighbors == right.parsed_stereo_neighbors
        && *implicit_hydrogen_cache == right.implicit_hydrogen_cache
        && *radical_electrons == right.radical_electrons
        && match (kekulization_source, &right.kekulization_source) {
            (None, None) => true,
            (Some(left), Some(right)) => same_graph_state(left, right),
            _ => false,
        }
}

/// Relabels `graph` by `labeling`, its exact canonical labeling with `root`
/// (local to `graph`) individualized when given, and composes the result
/// with `order` (the original atom id behind each of `graph`'s own local atom
/// ids) so the returned round's `order` stays in terms of the original atom
/// ids throughout the search.
fn relabel_round<AtomPolicy: SmilesAtomPolicy>(
    graph: &Smiles<AtomPolicy>,
    labeling: &SmilesCanonicalLabeling,
    root: Option<usize>,
    order: &[usize],
) -> RelabelRound<AtomPolicy> {
    RelabelRound {
        graph: graph.exact_canonicalize_with_labeling(labeling),
        root: root.map(|r| labeling.new_index_of_old_node()[r]),
        order: labeling.order().iter().map(|&local| order[local]).collect(),
        labeling: OnceCell::new(),
    }
}

/// Renders a [`RelabelRound`]'s graph rooted at where `root` landed in it.
fn render_round<AtomPolicy: SmilesAtomPolicy>(round: &RelabelRound<AtomPolicy>) -> String {
    let plan = round.graph.render_plan_with(
        round.graph.planning_classes(),
        round.root,
        Some(round.labeling()),
    );
    emit_with_plan(&round.graph, &plan)
}

/// Emits a SMILES string from a completed render plan.
///
/// No graph search or ordering work happens here. The emitter only walks
/// already-planned components and nodes and writes:
///
/// - atom text
/// - closure bond symbols and ring labels
/// - branch parentheses
/// - continuation edges
/// - `.` between disconnected components
#[must_use]
fn emit_with_plan<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    plan: &RenderPlan,
) -> String {
    let mut rendered = String::with_capacity(plan.estimated_rendered_len(smiles));

    for (index, component) in plan.components().iter().enumerate() {
        if index != 0 {
            rendered.push('.');
        }
        emit_node(smiles, plan, component.root(), &mut rendered);
    }

    rendered
}

/// Emits one planned node recursively.
///
/// The write order mirrors SMILES surface syntax:
///
/// 1. atom text
/// 2. attached closures
/// 3. parenthesized branch children
/// 4. continuation child, if any
fn emit_node<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    plan: &RenderPlan,
    node_id: usize,
    target: &mut String,
) {
    let node_plan = plan.node(node_id).unwrap_or_else(|| unreachable!());
    let atom = smiles.node_by_id(node_id).unwrap_or_else(|| unreachable!());
    atom.write_smiles_with_chirality_to_string(target, node_plan.normalized_chirality());

    for closure in node_plan.closures() {
        if closure.emit_bond_symbol() {
            target.push_str(rendered_bond_text(smiles, node_id, closure.partner(), closure.bond()));
        }
        write_ring_label(target, closure.label());
    }

    for branch_child in node_plan.branch_children() {
        target.push('(');
        target.push_str(rendered_bond_text(
            smiles,
            node_id,
            branch_child.child(),
            branch_child.bond(),
        ));
        emit_node(smiles, plan, branch_child.child(), target);
        target.push(')');
    }

    if let Some(continuation_child) = node_plan.continuation_child() {
        target.push_str(rendered_bond_text(
            smiles,
            node_id,
            continuation_child.child(),
            continuation_child.bond(),
        ));
        emit_node(smiles, plan, continuation_child.child(), target);
    }
}

/// Writes a ring label using the emitter's current label syntax.
fn write_ring_label(target: &mut String, label: u16) {
    if label < 10 {
        target.push(char::from(b'0' + u8::try_from(label).unwrap_or_else(|_| unreachable!())));
    } else if label < 100 {
        target.push('%');
        write!(target, "{label}").unwrap_or_else(|_| unreachable!("writing to String cannot fail"));
    } else {
        target.push_str("%(");
        write!(target, "{label}").unwrap_or_else(|_| unreachable!("writing to String cannot fail"));
        target.push(')');
    }
}

/// Returns the bond token to print between two planned neighbors after
/// aromatic elision rules are applied.
fn rendered_bond_text(
    smiles: &Smiles<impl SmilesAtomPolicy>,
    from: usize,
    to: usize,
    descriptor: crate::bond::BondDescriptor,
) -> &'static str {
    let from_aromatic = smiles.node_by_id(from).unwrap_or_else(|| unreachable!()).aromatic();
    let to_aromatic = smiles.node_by_id(to).unwrap_or_else(|| unreachable!()).aromatic();
    match (descriptor.bond(), descriptor.is_aromatic()) {
        (crate::bond::Bond::Single | crate::bond::Bond::Double, true)
            if from_aromatic && to_aromatic =>
        {
            ""
        }
        (crate::bond::Bond::Single | crate::bond::Bond::Double, true) => ":",
        (crate::bond::Bond::Single, false) if from_aromatic && to_aromatic => "-",
        (crate::bond::Bond::Single, false) => "",
        (other, _) => other.smiles_symbol(),
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::emit;
    use crate::{parser::smiles_parser::parse_wildcard_smiles, smiles::Smiles};

    fn render(smiles: &str) -> String {
        if smiles.contains('*') {
            emit(&parse_wildcard_smiles(smiles).unwrap())
        } else {
            emit(&smiles.parse::<Smiles>().unwrap())
        }
    }

    #[test]
    fn emitter_renders_simple_chain() {
        assert_eq!(render("CC"), "CC");
        assert_eq!(render("C=O"), "C=O");
        assert_eq!(render("CCCO"), "CCCO");
    }

    #[test]
    fn emitter_renders_simple_branch() {
        assert_eq!(render("CC(C)O"), "CC(C)O");
    }

    #[test]
    fn emitter_renders_simple_ring() {
        assert_eq!(render("C1CC1"), "C1CC1");
    }

    #[test]
    fn emitter_renders_disconnected_components() {
        assert_eq!(render("CC.O"), "CC.O");
    }

    #[test]
    fn emitter_stabilizes_explicit_single_between_aromatic_atoms() {
        let rendered = render("*c-c");
        assert!(rendered.contains('-'));
        assert_eq!(render(&rendered), rendered);
    }

    #[test]
    fn emitter_renders_large_ring_labels_with_current_syntax() {
        let mut rendered = String::new();
        super::write_ring_label(&mut rendered, 7);
        super::write_ring_label(&mut rendered, 42);
        super::write_ring_label(&mut rendered, 123);
        assert_eq!(rendered, "7%42%(123)");
    }

    #[test]
    fn rendered_bond_text_keeps_colon_for_non_aromatic_endpoints() {
        let aromatic_mismatch: Smiles = "C:C".parse().unwrap();
        assert_eq!(
            super::rendered_bond_text(
                &aromatic_mismatch,
                0,
                1,
                crate::bond::BondDescriptor::aromatic(crate::bond::Bond::Single)
            ),
            ":"
        );

        let aromatic_pair: Smiles = "c1ccccc1".parse().unwrap();
        assert_eq!(
            super::rendered_bond_text(
                &aromatic_pair,
                0,
                1,
                crate::bond::BondDescriptor::aromatic(crate::bond::Bond::Single)
            ),
            ""
        );
        assert_eq!(
            super::rendered_bond_text(&aromatic_pair, 0, 1, crate::bond::Bond::Single.into()),
            "-"
        );
        assert_eq!(
            super::rendered_bond_text(
                &aromatic_pair,
                0,
                1,
                crate::bond::BondDescriptor::aromatic(crate::bond::Bond::Double)
            ),
            ""
        );
        assert_eq!(
            super::rendered_bond_text(
                &aromatic_pair,
                0,
                1,
                crate::bond::BondDescriptor::aromatic(crate::bond::Bond::Triple)
            ),
            "#"
        );
    }

    /// Asserts that rendering, reparsing, and rendering again is a no-op,
    /// the property the `rooted_render` and `roundtrip` fuzz targets check
    /// continuously; a molecule with a real automorphism exercises the
    /// `orbit_min` relabeling path, and one without exercises its fast path.
    fn assert_render_is_idempotent(data: &str) {
        let first = render(data);
        let reparsed: Smiles = first.parse().unwrap();
        let second = emit(&reparsed);
        assert_eq!(first, second, "render of {data:?} is not stable across a reparse");
    }
    #[test]
    fn render_resolves_a_genuine_automorphism_to_its_lexicographic_minimum() {
        // The degree-2 `[C@]` here has two structurally identical `C`
        // substituents (a true automorphism `orbit_min`'s loop must run past
        // its first two rounds for), so only one of `@`/`@@` is reachable as
        // the stable, lexicographically smallest answer.
        assert_eq!(render("[C@](C)C"), "C[C@@]C");
    }

    #[test]
    fn render_is_idempotent_without_a_tie() {
        for data in ["CCO", "CC(C)O", "c1ccccc1", "N[C@@H](C)C(=O)O"] {
            assert_render_is_idempotent(data);
        }
    }

    #[test]
    fn render_is_idempotent_with_a_genuine_automorphism() {
        for data in ["C1CC1", "[C@](C)C", "C1CCCC2(CC1)CCCCCC2", "C12C3C4C1C5C2C3C45"] {
            assert_render_is_idempotent(data);
        }
    }

    #[test]
    fn render_rooted_is_idempotent_with_a_genuine_automorphism() {
        // `render_rooted` writes its root as the first atom of a connected
        // graph's traversal, so the root is always local atom `0` again
        // after a reparse.
        let smiles: Smiles = "C1CCCC2(CC1)CCCCCC2".parse().unwrap();
        for root in 0..smiles.nodes().len() {
            let first = smiles.render_rooted(root);
            let reparsed: Smiles = first.parse().unwrap();
            let second = reparsed.render_rooted(0);
            assert_eq!(first, second, "render_rooted at {root} is not stable across a reparse");
        }
    }
}
