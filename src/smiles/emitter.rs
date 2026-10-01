use alloc::string::String;
use core::fmt::Write;

use super::{Smiles, SmilesAtomPolicy, render_plan::RenderPlan};

/// Renders a [`Smiles`] graph by first building a [`RenderPlan`] and then
/// emitting text from that plan.
///
/// This is the top-level display helper used by [`Smiles::render`]. All
/// structural choices must already be encoded in the plan; this function is a
/// pure write-only pass.
#[must_use]
pub(crate) fn emit<AtomPolicy: SmilesAtomPolicy>(smiles: &Smiles<AtomPolicy>) -> String {
    let plan = smiles.render_plan();
    emit_with_plan(smiles, &plan)
}

/// Renders `smiles` with one component's traversal forced to start at `root`.
#[must_use]
pub(crate) fn emit_rooted<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    root: usize,
) -> String {
    let plan = smiles.render_plan_with_root(Some(root));
    emit_with_plan(smiles, &plan)
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
}
