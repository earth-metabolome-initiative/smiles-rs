use alloc::vec::Vec;

use geometric_traits::traits::SparseValuedMatrix2DRef;

use super::{
    Smiles, SmilesAtomPolicy,
    branches::BranchPlan,
    spanning_tree::SpanningForest,
    stereo::{
        DirectionalBondOverrides, StereoNeighbor, normalized_bond_for_emit,
        normalized_tetrahedral_chirality,
    },
};
use crate::{
    atom::bracketed::chirality::Chirality,
    bond::{Bond, BondDescriptor, bond_edge::BondEdge},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RenderPlan {
    components: Vec<ComponentRenderPlan>,
    nodes: Vec<NodeRenderPlan>,
}

impl RenderPlan {
    /// Returns the per-component traversal plans in final output order.
    #[must_use]
    pub(crate) fn components(&self) -> &[ComponentRenderPlan] {
        &self.components
    }

    /// Returns the per-node emission plan for a node id, if present.
    #[must_use]
    pub(crate) fn node(&self, node_id: usize) -> Option<&NodeRenderPlan> {
        self.nodes.get(node_id)
    }

    /// Returns all node ids in final traversal order (components concatenated
    /// in output order, each in preorder). Each component root is first in
    /// its run.
    #[must_use]
    pub(crate) fn traversal_order(&self) -> Vec<usize> {
        self.components.iter().flat_map(|component| component.preorder.iter().copied()).collect()
    }

    /// Estimates the final rendered length from the completed plan.
    ///
    /// This is used only to pre-size the output buffer. It mirrors the final
    /// emission structure:
    ///
    /// - component separators
    /// - atom text with normalized chirality
    /// - closure bond symbols and ring labels
    /// - branch parentheses and child bond symbols
    #[must_use]
    pub(crate) fn estimated_rendered_len<AtomPolicy: SmilesAtomPolicy>(
        &self,
        smiles: &Smiles<AtomPolicy>,
    ) -> usize {
        let mut total = self.components.len().saturating_sub(1);

        for (node_id, node_plan) in self.nodes.iter().enumerate() {
            let atom = smiles.node_by_id(node_id).unwrap_or_else(|| unreachable!());
            total += atom.rendered_len_hint_with_chirality(node_plan.normalized_chirality());

            for closure in node_plan.closures() {
                if closure.emit_bond_symbol() {
                    total +=
                        rendered_bond_text_len(smiles, node_id, closure.partner(), closure.bond());
                }
                total += ring_label_len(closure.label());
            }

            for child in node_plan.branch_children() {
                total += 2;
                total += rendered_bond_text_len(smiles, node_id, child.child(), child.bond());
            }

            if let Some(child) = node_plan.continuation_child() {
                total += rendered_bond_text_len(smiles, node_id, child.child(), child.bond());
            }
        }

        total
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComponentRenderPlan {
    root: usize,
    preorder: Vec<usize>,
}

impl ComponentRenderPlan {
    /// Returns the chosen root for this connected component.
    #[must_use]
    pub(crate) fn root(&self) -> usize {
        self.root
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn preorder(&self) -> &[usize] {
        &self.preorder
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChildRenderPlan {
    child: usize,
    bond: BondDescriptor,
}

impl ChildRenderPlan {
    /// Returns the child node reached by this tree edge.
    #[must_use]
    pub(crate) fn child(&self) -> usize {
        self.child
    }

    /// Returns the bond token that should be emitted before descending into the
    /// child.
    #[must_use]
    pub(crate) fn bond(&self) -> BondDescriptor {
        self.bond
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ClosureRenderPlan {
    edge: BondEdge,
    partner: usize,
    label: u16,
    bond: BondDescriptor,
    emit_bond_symbol: bool,
}

impl ClosureRenderPlan {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn edge(self) -> BondEdge {
        self.edge
    }

    /// Returns the partner node for this closure event.
    #[must_use]
    pub(crate) fn partner(self) -> usize {
        self.partner
    }

    /// Returns the allocated ring label to print for this closure event.
    #[must_use]
    pub(crate) fn label(self) -> u16 {
        self.label
    }

    /// Returns the already-normalized bond token to print if this endpoint is
    /// the closing side of the closure.
    #[must_use]
    pub(crate) fn bond(self) -> BondDescriptor {
        self.bond
    }

    /// Returns whether this endpoint must emit the bond symbol together with
    /// the ring label.
    ///
    /// Only the closing side emits the bond symbol. The opening side prints the
    /// label alone so closures follow standard SMILES placement rules.
    #[must_use]
    pub(crate) fn emit_bond_symbol(self) -> bool {
        self.emit_bond_symbol
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeRenderPlan {
    parent: Option<usize>,
    parent_bond: Option<BondDescriptor>,
    ordered_children: Vec<ChildRenderPlan>,
    closures: Vec<ClosureRenderPlan>,
    emitted_stereo_neighbors: Vec<StereoNeighbor>,
    normalized_chirality: Option<Chirality>,
}

impl NodeRenderPlan {
    #[cfg(test)]
    #[must_use]
    pub(crate) fn parent(&self) -> Option<usize> {
        self.parent
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn parent_bond(&self) -> Option<BondDescriptor> {
        self.parent_bond
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn ordered_children(&self) -> &[ChildRenderPlan] {
        &self.ordered_children
    }

    /// Returns the continuation child, if one exists.
    ///
    /// The continuation child is the last child in final order. All earlier
    /// children become parenthesized branches.
    #[must_use]
    pub(crate) fn continuation_child(&self) -> Option<ChildRenderPlan> {
        self.ordered_children.last().copied()
    }

    /// Returns the parenthesized branch children in final output order.
    ///
    /// This is every ordered child except the continuation child.
    #[must_use]
    pub(crate) fn branch_children(&self) -> &[ChildRenderPlan] {
        self.ordered_children.split_last().map_or(&[], |(_continuation, branches)| branches)
    }

    /// Returns the closure events attached to this node in final output order.
    #[must_use]
    pub(crate) fn closures(&self) -> &[ClosureRenderPlan] {
        &self.closures
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn emitted_stereo_neighbors(&self) -> &[StereoNeighbor] {
        &self.emitted_stereo_neighbors
    }

    /// Returns the chirality token to print for this atom after normalizing the
    /// parsed stereo description against the emitted neighbor order.
    #[must_use]
    pub(crate) fn normalized_chirality(&self) -> Option<Chirality> {
        self.normalized_chirality
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClosureDraft {
    closure_id: usize,
    edge: BondEdge,
    partner: usize,
}

struct RenderOrdering {
    forest: SpanningForest,
    branch_plan: BranchPlan,
    components: Vec<ComponentRenderPlan>,
    directional_overrides: DirectionalBondOverrides,
    closures_by_node: Vec<Vec<ClosureRenderPlan>>,
    max_assigned_label: u16,
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    /// Builds the complete traversal-and-emission plan for this graph.
    ///
    /// This is the planning half of display. It computes every ordering and
    /// normalization decision that affects the final string, then hands the
    /// resulting immutable plan to the emitter.
    ///
    /// The live path is:
    ///
    /// 1. atom invariants
    /// 2. seeded WL refinement
    /// 3. rooted symmetry classes
    /// 4. component roots
    /// 5. primary spanning forest from structurally ordered neighbors
    /// 6. branch planning and component preorder assignment
    /// 7. directional bond overrides, tetrahedral normalization, and raw
    ///    directional-single fallback for unsupported double-bond environments
    /// 8. closure scheduling and ring-label assignment
    /// 9. per-node materialization
    ///
    /// If the primary forest would force ring labels above `99`, a fallback
    /// forest based on parser bond order is also evaluated and used only if it
    /// lowers the maximum label. This is a best-effort reduction step, not a
    /// proof that the final plan will always stay below `100`.
    #[must_use]
    pub(crate) fn render_plan(&self) -> RenderPlan {
        self.render_plan_with_root(None)
    }

    /// Replaces the chosen root of `forced`'s connected component with
    /// `forced`, leaving the roots of other components untouched.
    fn force_component_root(&self, roots: &mut [usize], forced: usize) {
        let components = self.connected_components();
        let target = components.component_of_node(forced);
        for root in roots.iter_mut() {
            if components.component_of_node(*root) == target {
                *root = forced;
                break;
            }
        }
    }

    /// Builds a render plan, optionally forcing the traversal of one connected
    /// component to start at `forced_root`. Other components keep their
    /// deterministically chosen roots.
    pub(crate) fn render_plan_with_root(&self, forced_root: Option<usize>) -> RenderPlan {
        let node_count = self.nodes().len();
        let invariants = self.atom_invariants();
        let refined = self.refined_atom_classes_from_invariants(&invariants);
        let refined_classes = refined.classes().to_vec();
        let rooted_classes = self.rooted_symmetry_classes_from_refined(&refined_classes);
        let mut roots = self.component_roots_from_planning(&invariants, &rooted_classes);
        if let Some(forced) = forced_root {
            self.force_component_root(&mut roots, forced);
        }
        let primary = build_render_ordering(
            self,
            self.spanning_forest_with_planning(
                &roots,
                &invariants,
                &refined_classes,
                &rooted_classes,
            ),
            &invariants,
            &refined_classes,
            &rooted_classes,
            node_count,
        );
        let ordering = if primary.max_assigned_label > 99 {
            let fallback = build_render_ordering(
                self,
                self.spanning_forest_with_parser_neighbor_order(&roots),
                &invariants,
                &refined_classes,
                &rooted_classes,
                node_count,
            );
            if fallback.max_assigned_label < primary.max_assigned_label {
                fallback
            } else {
                primary
            }
        } else {
            primary
        };

        let mut nodes = Vec::with_capacity(node_count);
        for (node_id, closures) in ordering.closures_by_node.into_iter().enumerate() {
            let parent = ordering.forest.parent_of(node_id);
            let atom = self.node_by_id(node_id).unwrap_or_else(|| unreachable!());
            let parent_bond =
                parent.zip(ordering.forest.parent_bond_of(node_id)).map(|(parent_id, bond)| {
                    planned_bond_for_emit(
                        self,
                        &ordering.directional_overrides,
                        bond,
                        parent_id,
                        node_id,
                    )
                });

            let ordered_children: Vec<ChildRenderPlan> = ordering
                .branch_plan
                .ordered_children(node_id)
                .iter()
                .map(|&child| {
                    let edge =
                        self.edge_for_node_pair((node_id, child)).unwrap_or_else(|| unreachable!());
                    ChildRenderPlan {
                        child,
                        bond: planned_bond_for_emit(
                            self,
                            &ordering.directional_overrides,
                            edge.descriptor(),
                            node_id,
                            child,
                        ),
                    }
                })
                .collect();

            let emitted_stereo_neighbors =
                emitted_stereo_neighbors(self, node_id, parent, &closures, &ordered_children);
            let normalized_chirality = atom.chirality().and_then(|chirality| {
                let parsed_stereo_neighbors = self.parsed_stereo_neighbors_row(node_id);
                normalized_tetrahedral_chirality(
                    Some(chirality),
                    parsed_stereo_neighbors,
                    &emitted_stereo_neighbors,
                )
            });

            nodes.push(NodeRenderPlan {
                parent,
                parent_bond,
                ordered_children,
                closures,
                emitted_stereo_neighbors,
                normalized_chirality,
            });
        }

        RenderPlan { components: ordering.components, nodes }
    }
}

/// Builds the ordering bundle shared by node materialization.
///
/// This combines the forest, branch plan, component preorder, closure
/// scheduling, and directional override projection so the final node loop does
/// not have to recompute any structural decisions.
fn build_render_ordering<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    forest: SpanningForest,
    invariants: &[crate::smiles::invariants::AtomInvariant],
    refined_classes: &[usize],
    rooted_classes: &[usize],
    node_count: usize,
) -> RenderOrdering {
    let branch_plan =
        smiles.branch_plan_with_planning(&forest, invariants, refined_classes, rooted_classes);
    let (components, preorder_indices, global_preorder) =
        build_component_preorders(&forest, &branch_plan, node_count);
    let directional_overrides = smiles.projected_directional_bond_overrides_with_classes(
        &preorder_indices,
        rooted_classes,
        refined_classes,
    );
    let (closures_by_node, max_assigned_label) = build_labeled_closures(
        smiles,
        &forest,
        &preorder_indices,
        &global_preorder,
        &directional_overrides,
    );

    RenderOrdering {
        forest,
        branch_plan,
        components,
        directional_overrides,
        closures_by_node,
        max_assigned_label,
    }
}

/// Computes the final preorder of every component after child order has been
/// fixed.
///
/// This preorder, not raw DFS discovery order, drives closure scheduling and
/// therefore ring-label lifetimes.
fn build_component_preorders(
    forest: &SpanningForest,
    branch_plan: &BranchPlan,
    node_count: usize,
) -> (Vec<ComponentRenderPlan>, Vec<usize>, Vec<usize>) {
    let mut components = Vec::with_capacity(forest.roots().len());
    let mut preorder_indices = vec![usize::MAX; node_count];
    let mut global_preorder = Vec::with_capacity(node_count);
    let mut next_index = 0;

    for &root in forest.roots() {
        let mut component_preorder = Vec::new();
        assign_component_preorder(
            root,
            branch_plan,
            &mut preorder_indices,
            &mut next_index,
            &mut component_preorder,
            &mut global_preorder,
        );
        components.push(ComponentRenderPlan { root, preorder: component_preorder });
    }

    (components, preorder_indices, global_preorder)
}

/// Recursively assigns preorder indices by walking children in final branch
/// order.
fn assign_component_preorder(
    node_id: usize,
    branch_plan: &BranchPlan,
    preorder_indices: &mut [usize],
    next_index: &mut usize,
    component_preorder: &mut Vec<usize>,
    global_preorder: &mut Vec<usize>,
) {
    if preorder_indices[node_id] != usize::MAX {
        return;
    }

    preorder_indices[node_id] = *next_index;
    *next_index += 1;
    component_preorder.push(node_id);
    global_preorder.push(node_id);

    for &child in branch_plan.ordered_children(node_id) {
        assign_component_preorder(
            child,
            branch_plan,
            preorder_indices,
            next_index,
            component_preorder,
            global_preorder,
        );
    }
}

fn build_labeled_closures<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    forest: &SpanningForest,
    preorder_indices: &[usize],
    global_preorder: &[usize],
    directional_overrides: &DirectionalBondOverrides,
) -> (Vec<Vec<ClosureRenderPlan>>, u16) {
    build_labeled_closures_impl(
        smiles,
        forest,
        preorder_indices,
        global_preorder,
        directional_overrides,
    )
}

/// Schedules all closure events in final preorder and assigns the lowest
/// reusable ring label to each live closure interval.
///
/// The important policy here is:
///
/// - opening and closing endpoints are attached to their owning nodes first
/// - nodes are processed in final global preorder
/// - closings are emitted before openings at the same node
/// - labels are recycled as soon as the closing endpoint is seen
fn build_labeled_closures_impl<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    forest: &SpanningForest,
    preorder_indices: &[usize],
    global_preorder: &[usize],
    directional_overrides: &DirectionalBondOverrides,
) -> (Vec<Vec<ClosureRenderPlan>>, u16) {
    let node_count = smiles.nodes().len();
    let mut drafts_by_node = vec![Vec::new(); node_count];
    for (closure_id, &edge) in forest.closure_edges().iter().enumerate() {
        let (node_a, node_b) = (edge.source(), edge.target());
        drafts_by_node[node_a].push(ClosureDraft { closure_id, edge, partner: node_b });
        drafts_by_node[node_b].push(ClosureDraft { closure_id, edge, partner: node_a });
    }

    let mut labeled_by_node = vec![Vec::new(); node_count];
    let mut active_labels = vec![None; forest.closure_edges().len()];
    let mut used_labels = vec![false; forest.closure_edges().len().saturating_add(1)];
    let mut max_assigned_label = 0;

    for &node_id in global_preorder {
        drafts_by_node[node_id].sort_unstable_by(|left, right| {
            compare_closure_drafts_for_node(node_id, left, right, preorder_indices, &active_labels)
        });

        for draft in &drafts_by_node[node_id] {
            let is_closing = active_labels[draft.closure_id].is_some();
            let label = if let Some(label) = active_labels[draft.closure_id].take() {
                used_labels[label as usize] = false;
                label
            } else {
                let label = lowest_free_label(&used_labels);
                active_labels[draft.closure_id] = Some(label);
                used_labels[label as usize] = true;
                label
            };

            labeled_by_node[node_id].push(ClosureRenderPlan {
                edge: draft.edge,
                partner: draft.partner,
                label,
                bond: planned_bond_for_emit(
                    smiles,
                    directional_overrides,
                    draft.edge.descriptor(),
                    node_id,
                    draft.partner,
                ),
                emit_bond_symbol: is_closing,
            });
            max_assigned_label = max_assigned_label.max(label);
        }
    }

    (labeled_by_node, max_assigned_label)
}

/// Orders the closure events attached to a node.
///
/// Closing events come first so they free labels before new openings are
/// assigned. Openings are then ordered by how soon their partner appears in the
/// final preorder, which tends to shorten label lifetimes.
fn compare_closure_drafts_for_node(
    node_id: usize,
    left: &ClosureDraft,
    right: &ClosureDraft,
    preorder_indices: &[usize],
    active_labels: &[Option<u16>],
) -> core::cmp::Ordering {
    let left_closing = active_labels[left.closure_id].is_some();
    let right_closing = active_labels[right.closure_id].is_some();

    right_closing
        .cmp(&left_closing)
        .then_with(|| {
            if left_closing && right_closing {
                active_labels[left.closure_id].cmp(&active_labels[right.closure_id]).then_with(
                    || preorder_indices[left.partner].cmp(&preorder_indices[right.partner]),
                )
            } else {
                closure_opening_key(node_id, left, preorder_indices).cmp(&closure_opening_key(
                    node_id,
                    right,
                    preorder_indices,
                ))
            }
        })
        .then_with(|| left.partner.cmp(&right.partner))
        .then_with(|| {
            (left.edge.source(), left.edge.target())
                .cmp(&(right.edge.source(), right.edge.target()))
        })
}

/// Returns the ordering key for a closure opening endpoint.
fn closure_opening_key(
    node_id: usize,
    draft: &ClosureDraft,
    preorder_indices: &[usize],
) -> (usize, usize) {
    let node_preorder = preorder_indices[node_id];
    let partner_preorder = preorder_indices[draft.partner];
    (partner_preorder.saturating_sub(node_preorder), partner_preorder)
}

/// Returns the smallest currently-unused ring label.
fn lowest_free_label(used_labels: &[bool]) -> u16 {
    used_labels
        .iter()
        .enumerate()
        .skip(1)
        .find_map(|(label, &used)| {
            (!used).then(|| u16::try_from(label).unwrap_or_else(|_| unreachable!()))
        })
        .unwrap_or_else(|| unreachable!())
}

/// Converts a stored graph bond into the exact bond token that should be
/// emitted between two planned neighbors.
///
/// This is where the render plan combines:
///
/// - lexical normalization of directional single bonds for traversal direction
/// - semantic directional overrides derived from supported double-bond stereo
/// - the fallback policy that preserves raw directional syntax in unsupported
///   environments
fn planned_bond_for_emit<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    directional_overrides: &DirectionalBondOverrides,
    descriptor: BondDescriptor,
    from: usize,
    to: usize,
) -> BondDescriptor {
    let normalized = normalized_bond_for_emit(descriptor.bond(), from, to);
    if matches!(normalized, Bond::Single | Bond::Up | Bond::Down)
        && let Some(override_bond) = directional_overrides.get(from, to)
    {
        return descriptor.with_bond(override_bond);
    }

    let bond = match normalized {
        Bond::Up | Bond::Down
            if preserve_raw_directional_single(smiles, directional_overrides, from, to) =>
        {
            normalized
        }
        Bond::Up | Bond::Down => Bond::Single,
        other => other,
    };
    descriptor.with_bond(bond)
}

/// Preserves raw directions in double-bond environments outside the semantic
/// stereo model.
pub(super) fn preserve_raw_directional_single<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    directional_overrides: &DirectionalBondOverrides,
    from: usize,
    to: usize,
) -> bool {
    [from, to].into_iter().any(|node_id| {
        !directional_overrides.has_semantic_endpoint(node_id)
            && smiles.edges_for_node(node_id).any(|edge| {
                edge.bond() == Bond::Double
                    && !edge.is_aromatic()
                    && !double_bond_supports_semantic_stereo(smiles, edge.source(), edge.target())
            })
    })
}

/// Returns whether a double bond is simple enough for the semantic alkene
/// stereo extractor to own it completely.
fn double_bond_supports_semantic_stereo<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    node_a: usize,
    node_b: usize,
) -> bool {
    non_single_family_bond_count(smiles, node_a) == 1
        && non_single_family_bond_count(smiles, node_b) == 1
}

/// Counts the non-single-family bonds around a node.
///
/// `Single`, `Up`, and `Down` are treated as one family here because the
/// semantic double-bond stereo layer reasons about them together.
fn non_single_family_bond_count<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    node_id: usize,
) -> usize {
    smiles
        .bond_matrix()
        .sparse_row_values_ref(node_id)
        .filter(|entry| {
            entry.aromatic() || !matches!(entry.bond(), Bond::Single | Bond::Up | Bond::Down)
        })
        .count()
}

/// Computes the lexical neighbor order that matters for tetrahedral chirality
/// after all branches and closures have been fixed.
///
/// The order is:
///
/// - parent, if any
/// - explicit hydrogen for chiral `[XH]` centers
/// - closure partners in their node-local closure order
/// - tree children in final branch/continuation order
fn emitted_stereo_neighbors<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    node_id: usize,
    parent: Option<usize>,
    closures: &[ClosureRenderPlan],
    ordered_children: &[ChildRenderPlan],
) -> Vec<StereoNeighbor> {
    let mut neighbors = Vec::new();

    if let Some(parent) = parent {
        neighbors.push(StereoNeighbor::Atom(parent));
    }
    let atom = smiles.node_by_id(node_id).unwrap_or_else(|| unreachable!());
    if atom.hydrogen_count() == 1 && atom.chirality().is_some() {
        neighbors.push(StereoNeighbor::ExplicitHydrogen);
    }
    neighbors.extend(closures.iter().map(|closure| StereoNeighbor::Atom(closure.partner())));
    neighbors.extend(ordered_children.iter().map(|child| StereoNeighbor::Atom(child.child())));

    neighbors
}

/// Returns the rendered width of a ring label.
fn ring_label_len(label: u16) -> usize {
    if label < 10 {
        1
    } else if label < 100 {
        1 + decimal_len_u16(label)
    } else {
        3 + decimal_len_u16(label)
    }
}

/// Returns the decimal width of a `u16`.
fn decimal_len_u16(value: u16) -> usize {
    if value >= 10_000 {
        5
    } else if value >= 1_000 {
        4
    } else if value >= 100 {
        3
    } else if value >= 10 {
        2
    } else {
        1
    }
}

/// Returns the rendered width of a bond token after aromatic elision rules are
/// applied.
fn rendered_bond_text_len<AtomPolicy: SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    from: usize,
    to: usize,
    descriptor: BondDescriptor,
) -> usize {
    let from_aromatic = smiles.node_by_id(from).unwrap_or_else(|| unreachable!()).aromatic();
    let to_aromatic = smiles.node_by_id(to).unwrap_or_else(|| unreachable!()).aromatic();
    let both_aromatic = from_aromatic && to_aromatic;
    let elided = (descriptor.is_aromatic()
        && both_aromatic
        && matches!(descriptor.bond(), Bond::Single | Bond::Double))
        || (descriptor.bond() == Bond::Single && !descriptor.is_aromatic() && !both_aromatic);
    usize::from(!elided)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        ChildRenderPlan, ClosureRenderPlan, ComponentRenderPlan, RenderPlan, Smiles, StereoNeighbor,
    };
    use crate::atom::bracketed::chirality::Chirality;

    fn plan(smiles: &str) -> RenderPlan {
        smiles.parse::<Smiles>().unwrap().render_plan()
    }

    #[test]
    fn render_plan_of_empty_graph_is_empty() {
        let plan = Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy().render_plan();
        assert_eq!(plan.components(), [] as [ComponentRenderPlan; 0]);
        assert_eq!(plan.node(0), None);
    }

    #[test]
    fn render_plan_preserves_component_roots_and_preorder_for_chain() {
        let plan = plan("CCCO");
        assert_eq!(plan.components().len(), 1);
        assert_eq!(plan.components()[0].root(), 0);
        assert_eq!(plan.components()[0].preorder(), &[0, 1, 2, 3]);

        let node_0 = plan.node(0).unwrap();
        assert_eq!(node_0.parent(), None);
        assert_eq!(node_0.continuation_child().map(|child| child.child()), Some(1));
        assert_eq!(node_0.branch_children(), [] as [ChildRenderPlan; 0]);
        assert_eq!(node_0.closures(), [] as [ClosureRenderPlan; 0]);
        assert_eq!(node_0.ordered_children()[0].child(), 1);

        let node_2 = plan.node(2).unwrap();
        assert_eq!(node_2.parent(), Some(1));
        assert_eq!(node_2.parent_bond(), Some(crate::bond::Bond::Single.into()));
        assert_eq!(node_2.continuation_child().map(|child| child.child()), Some(3));
        assert_eq!(node_2.branch_children(), [] as [ChildRenderPlan; 0]);
        assert_eq!(
            node_2.emitted_stereo_neighbors(),
            &[StereoNeighbor::Atom(1), StereoNeighbor::Atom(3)]
        );
    }

    #[test]
    fn render_plan_assigns_ring_labels_from_final_preorder() {
        let smiles: Smiles = "C1CC1".parse().unwrap();
        let plan = smiles.render_plan();

        assert_eq!(plan.components()[0].preorder(), &[0, 1, 2]);

        let node_0 = plan.node(0).unwrap();
        let node_2 = plan.node(2).unwrap();
        assert_eq!(node_0.closures().len(), 1);
        assert_eq!(node_2.closures().len(), 1);
        assert_eq!(node_0.closures()[0].partner(), 2);
        assert_eq!(node_2.closures()[0].partner(), 0);
        assert_eq!(node_0.closures()[0].label(), 1);
        assert_eq!(node_2.closures()[0].label(), 1);
        assert_eq!(node_0.closures()[0].edge(), smiles.edge_for_node_pair((0, 2)).unwrap());
    }

    #[test]
    fn render_plan_carries_branch_and_continuation_structure() {
        let plan = plan("CC(C)O");
        let node_1 = plan.node(1).unwrap();

        assert_eq!(node_1.parent(), Some(0));
        assert_eq!(node_1.ordered_children().len(), 2);
        assert_eq!(node_1.branch_children().len(), 1);
        assert_eq!(node_1.continuation_child().map(|child| child.child()), Some(3));
        assert_eq!(
            node_1.branch_children().iter().map(super::ChildRenderPlan::child).collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(
            node_1.emitted_stereo_neighbors(),
            &[StereoNeighbor::Atom(0), StereoNeighbor::Atom(2), StereoNeighbor::Atom(3)]
        );
    }

    #[test]
    fn render_plan_normalizes_tetrahedral_chirality_from_emitted_neighbor_order() {
        let plan = plan("N[C@@H](C)O");
        let node_1 = plan.node(1).unwrap();

        assert_eq!(
            node_1.emitted_stereo_neighbors(),
            &[
                StereoNeighbor::Atom(2),
                StereoNeighbor::ExplicitHydrogen,
                StereoNeighbor::Atom(0),
                StereoNeighbor::Atom(3),
            ]
        );
        assert_eq!(node_1.normalized_chirality(), Some(Chirality::At));
    }
}
