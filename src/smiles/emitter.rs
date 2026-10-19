use alloc::{string::String, vec::Vec};
use core::fmt::Write;

use super::{Smiles, SmilesAtomPolicy, SmilesCanonicalLabeling, render_plan::RenderPlan};

/// Renders a [`Smiles`] graph by first building a [`RenderPlan`] and then
/// emitting text from that plan.
///
/// This is the top-level display helper used by [`Smiles::render`]. All
/// structural choices must already be encoded in the plan; this function is a
/// pure write-only pass.
#[must_use]
pub(crate) fn emit<AtomPolicy: SmilesAtomPolicy>(smiles: &Smiles<AtomPolicy>) -> String {
    symmetric_render(smiles, None).0
}

/// Renders `smiles` with one component's traversal forced to start at `root`.
#[must_use]
pub(crate) fn emit_rooted<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    root: usize,
) -> String {
    symmetric_render(smiles, Some(root)).0
}

/// Returns the traversal order (original atom ids) backing
/// [`emit_rooted`]'s output.
#[must_use]
pub(crate) fn rooted_traversal_order<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    root: usize,
) -> Vec<usize> {
    symmetric_render(smiles, Some(root)).1
}

/// Returns the render of `smiles` and the original atom ids in its traversal
/// order.
///
/// When WL refinement leaves no tie, the planner's own order is canonical.
/// Otherwise the graph is renumbered by
/// [`Smiles::symmetric_stereo_labeling`], which is the same for every
/// renumbering of the input, and rendered with that numbering as the
/// tie-break. A render needing ring labels above 99 yields to the input
/// numbering's plan when that needs fewer.
fn symmetric_render<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    root: Option<usize>,
) -> (String, Vec<usize>) {
    let classes = smiles.planning_classes();
    if !classes.has_ties() {
        let plan = smiles.render_plan_with(classes, root, None);
        return (emit_with_plan(smiles, &plan), plan.traversal_order());
    }

    let labeling =
        smiles.symmetric_stereo_labeling(root, classes.rooted_classes(), classes.refined_classes());
    let canonical = smiles.exact_canonicalize_with_labeling(&labeling);
    let identity = SmilesCanonicalLabeling::from_order((0..smiles.nodes().len()).collect());
    let plan = canonical.render_plan_with(
        canonical.planning_classes(),
        root.map(|root| labeling.new_index_of_old_node()[root]),
        Some(&identity),
    );
    if plan.max_ring_label() > 99 {
        // relabeling drops the input bond order that the fallback for labels
        // above 99 follows
        let input_plan = smiles.render_plan_with(classes, root, None);
        if input_plan.max_ring_label() < plan.max_ring_label() {
            return (emit_with_plan(smiles, &input_plan), input_plan.traversal_order());
        }
    }
    let order = plan.traversal_order().into_iter().map(|local| labeling.order()[local]).collect();
    (emit_with_plan(&canonical, &plan), order)
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
    use alloc::{string::String, vec::Vec};

    use super::emit;
    use crate::{
        parser::smiles_parser::parse_wildcard_smiles,
        smiles::{Smiles, canonicalization::support::permute_smiles},
    };

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

    /// Asserts that rendering, reparsing, and rendering again is a no-op.
    fn assert_render_is_idempotent(data: &str) {
        let first = render(data);
        let reparsed: Smiles = first.parse().unwrap();
        let second = emit(&reparsed);
        assert_eq!(first, second, "render of {data:?} is not stable across a reparse");
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

    /// Symmetric stereo whose relabelings fall into several cycles still
    /// renders one string.
    #[test]
    fn render_is_idempotent_with_symmetric_stereo() {
        for data in [
            r"C1\C=C/CC/C=C/CC/C=C/CC/C=C/C1",
            r"C1\C=C/[C@H](O)C/C=C/[C@H](O)C/C=C/[C@H](O)C/C=C/[C@H](O)1",
            r"[C@@H]1(C)/C=C/C[C@@H](C)/C=C/C[C@@H](C)/C=C/C1",
        ] {
            assert_render_is_idempotent(data);
        }
    }

    /// Deterministic atom orders, each listing the old id at every new
    /// position.
    fn shuffled_orders(node_count: usize, count: usize) -> Vec<Vec<usize>> {
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        (0..count)
            .map(|_| {
                let mut order: Vec<usize> = (0..node_count).collect();
                for position in (1..node_count).rev() {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let swap_with = usize::try_from(seed >> 33).unwrap() % (position + 1);
                    order.swap(position, swap_with);
                }
                order
            })
            .collect()
    }

    /// Renumbering atoms whose skeleton automorphisms act on stereo changes
    /// neither `render` nor `render_rooted` at the same atom.
    #[test]
    fn render_is_invariant_under_renumbering_with_symmetric_stereo() {
        for data in [
            "[C@](C)C",
            "C[C@](C)(C)[C@](C)(C)[C@](C)(C)[C@](C)(C)C",
            "F[C@H]1C[C@@H](F)C1",
            r"C1\C=C/CC/C=C/CC/C=C/CC/C=C/C1",
            r"C1\C=C/[C@H](O)C/C=C/[C@H](O)C/C=C/[C@H](O)C/C=C/[C@H](O)1",
            r"[C@@H]1(C)/C=C/C[C@@H](C)/C=C/C[C@@H](C)/C=C/C1",
            r"F/C=C=C=C/F",
            r"F/C(/Cl)=C=C=C(\Cl)/F",
            r"C/C=C=C=C/C.C/C=C=C=C/C",
            "C[C@TH1](C)(C)[C@TH2](C)(C)C",
            "CC(C)=[C@AL1]=C(C)C",
            "C[C@AL2](C)=C=C(C)[C@AL1](C)=C=C(C)C",
            r"C=C=C/C=C=C",
            r"CC=C=C/C=C=CC",
            r"F/C(F)=C=C=C(F)/F",
            "F[C@TH1](C)(C)CC[C@](C)(C)F",
            "F[C@TH1](Cl)(Br)CC[C@](Cl)(Br)F",
            "CC(F)=[C@AL1]=C(F)CCC(F)=[C@]=C(F)C",
            "C1[C@TH1](F)(Cl)CC[C@](F)(Cl)C1",
            "C1[C@TH1](F)(Cl)C[C@](F)(Cl)C1",
            "C1C[C@TH1](F)(C)CC[C@](F)(C)CC1",
            "C1[C@TH1](F)(Cl)CC[C@@](F)(Cl)CC[C@TH2](F)(Cl)CC[C@](F)(Cl)C1",
            "C1C=C[C@AL1]=CC=CC=C[C@]=CC=C1",
            "C1C=[C@AL1]=CCC=[C@]=CC1",
            "C1C=[C@AL1]=CCCCC=[C@]=CCCC1",
            "C1C(F)=[C@AL1]=C(Cl)CCC(F)=[C@]=C(Cl)CC1",
            "C1[C@AL1](F)(Cl)CC[C@](F)(Cl)C1",
            "N[C@](C)(O)CC[C@@](C)(O)N",
            "C1[C@H](F)[C@@H](F)[C@H](F)[C@@H]1F",
        ] {
            let smiles: Smiles = data.parse().unwrap();
            let node_count = smiles.nodes().len();
            for order in shuffled_orders(node_count, 12) {
                let renumbered = permute_smiles(&smiles, &order);
                assert_eq!(renumbered.render(), smiles.render(), "{data} renumbered by {order:?}");
                for (new_root, &old_root) in order.iter().enumerate() {
                    assert_eq!(
                        renumbered.render_rooted(new_root),
                        smiles.render_rooted(old_root),
                        "{data} rooted at {old_root} renumbered by {order:?}"
                    );
                }
            }
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
