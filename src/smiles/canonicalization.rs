use alloc::vec::Vec;

use elements_rs::Element;
use geometric_traits::{
    prelude::CanonicalLabeling,
    traits::{SparseMatrix2D, SparseValuedMatrixRef},
};

use super::{
    BondMatrix, BondMatrixBuilder, Smiles, StereoNeighbor,
    implicit_hydrogens::implicit_hydrogens_if_written_unbracketed,
};
use crate::{
    atom::{Atom, AtomSyntax, atom_symbol::AtomSymbol, can_write_unbracketed_aromatic},
    bond::Bond,
};

mod components;
mod state;
mod stereo_normalization;
#[cfg(any(test, feature = "fuzzing"))]
mod support;
#[cfg(test)]
mod tests;
use self::state::{
    CanonicalAtomLabel, CanonicalBondLabel, CanonicalizationStateKey, canonical_atom_label,
    canonical_bond_label, canonicalization_state_key, stereo_neutral_canonical_atom_label,
};

/// Canonical labeling result for a [`Smiles`] graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmilesCanonicalLabeling {
    order: Vec<usize>,
    new_index_of_old_node: Vec<usize>,
}

impl SmilesCanonicalLabeling {
    fn new(order: Vec<usize>) -> Self {
        let mut new_index_of_old_node = vec![usize::MAX; order.len()];
        for (new_index, old_node) in order.iter().copied().enumerate() {
            new_index_of_old_node[old_node] = new_index;
        }
        Self { order, new_index_of_old_node }
    }

    pub(crate) fn from_order(order: Vec<usize>) -> Self {
        Self::new(order)
    }

    /// Returns original node ids in canonical order.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let labeling = "OC".parse::<Smiles>()?.canonical_labeling();
    /// assert_eq!(labeling.order(), &[1, 0]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Returns the new canonical index for each original node id.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let labeling = "OC".parse::<Smiles>()?.canonical_labeling();
    /// assert_eq!(labeling.new_index_of_old_node(), &[1, 0]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn new_index_of_old_node(&self) -> &[usize] {
        &self.new_index_of_old_node
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    pub(super) fn exact_canonical_labeling(&self) -> SmilesCanonicalLabeling {
        self.canonical_labeling_with(Self::exact_canonical_labeling_whole_graph)
    }

    fn exact_canonical_labeling_whole_graph(&self) -> SmilesCanonicalLabeling {
        self.canonical_labeling_whole_graph(canonical_atom_label, canonical_bond_label)
    }

    fn stereo_neutral_canonical_labeling(&self) -> SmilesCanonicalLabeling {
        self.canonical_labeling_with(Self::stereo_neutral_canonical_labeling_whole_graph)
    }

    fn stereo_neutral_canonical_labeling_whole_graph(&self) -> SmilesCanonicalLabeling {
        self.canonical_labeling_whole_graph(stereo_neutral_canonical_atom_label, |entry| {
            canonical_bond_label(entry.with_bond(entry.bond().without_direction()))
        })
    }

    fn canonical_labeling_whole_graph(
        &self,
        atom_label: impl Fn(Atom) -> CanonicalAtomLabel,
        bond_label: impl Fn(super::BondEntry) -> CanonicalBondLabel,
    ) -> SmilesCanonicalLabeling {
        let result = CanonicalLabeling::canonical_labeling(
            self,
            |node_id| atom_label(self.nodes()[node_id]),
            |node_a, node_b| {
                bond_label(
                    self.bond_entry_for_node_pair((node_a, node_b))
                        .unwrap_or_else(|| unreachable!("canonizer only queries existing edges")),
                )
            },
        );
        SmilesCanonicalLabeling::new(result.order)
    }

    fn exact_canonicalize(&self) -> Self {
        let labeling = self.exact_canonical_labeling();
        self.exact_canonicalize_with_labeling(&labeling)
    }

    fn exact_canonicalize_with_labeling(&self, labeling: &SmilesCanonicalLabeling) -> Self {
        let order = labeling.order();
        let new_index_of_old_node = labeling.new_index_of_old_node();

        let atom_nodes: Vec<_> =
            order.iter().copied().map(|old_node| self.atom_nodes[old_node]).collect();

        let mut canonical_edges = Vec::with_capacity(self.number_of_bonds());
        for ((row, column), entry) in self.bond_matrix().sparse_entries() {
            if row >= column {
                continue;
            }
            let new_row = new_index_of_old_node[row];
            let new_column = new_index_of_old_node[column];
            let (new_row, new_column) = crate::smiles::edge_key(new_row, new_column);
            canonical_edges.push((new_row, new_column, entry.descriptor()));
        }
        canonical_edges.sort_unstable_by_key(|(row, column, _descriptor)| (*row, *column));
        let bond_matrix = BondMatrix::from_sorted_upper_triangular_entries(
            atom_nodes.len(),
            canonical_edges.into_iter().enumerate().map(
                |(canonical_order, (row, column, descriptor))| {
                    (
                        row,
                        column,
                        super::BondEntry::from_descriptor(descriptor, None, canonical_order),
                    )
                },
            ),
        )
        .unwrap_or_else(|_| {
            unreachable!("canonical relabeling preserves a simple upper-triangular graph")
        });

        let parsed_stereo_neighbors = order
            .iter()
            .copied()
            .map(|old_node| {
                remap_parsed_stereo_neighbors_row(self, old_node, new_index_of_old_node)
            })
            .collect();

        let implicit_hydrogen_cache =
            order.iter().copied().map(|old_node| self.implicit_hydrogen_cache[old_node]).collect();

        Self::from_bond_matrix_parts_with_sidecars(
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            implicit_hydrogen_cache,
            None,
        )
    }

    fn canonicalization_normal_form(&self) -> Self {
        let input = self.canonicalization_spelling_normal_form();
        if input.nodes().is_empty() {
            return input;
        }
        let Ok(perception) = input.perceive_aromaticity() else {
            return input;
        };
        if perception.aromaticized().aromaticity_assignment() != *perception.assignment() {
            return input;
        }
        perception.into_aromaticized().canonicalization_spelling_normal_form()
    }

    fn canonicalization_step(&self) -> Self {
        let canonicalized = self
            .canonicalization_normal_form()
            .collapse_removable_explicit_hydrogens()
            .canonicalize_from_current_bond_orders();
        let has_aromatic_bonds = canonicalized
            .bond_matrix()
            .sparse_entries()
            .any(|((_row, _column), entry)| entry.aromatic());
        if !has_aromatic_bonds {
            return canonicalized;
        }

        canonicalized
            .kekulize_standalone_in_current_order()
            .ok()
            .map(|kekulized| kekulized.canonicalize_from_current_bond_orders())
            .unwrap_or(canonicalized)
    }

    fn canonicalize_from_current_bond_orders(&self) -> Self {
        self.stereo_normal_form().exact_canonicalize().canonicalization_spelling_normal_form()
    }

    pub(super) fn canonicalization_spelling_normal_form(&self) -> Self {
        let atom_nodes = self
            .atom_nodes
            .iter()
            .copied()
            .enumerate()
            .map(|(node_id, atom)| canonicalization_atom_spelling_normal_form(self, node_id, atom))
            .collect::<Vec<_>>();
        if atom_nodes.is_empty() {
            return self.clone_without_kekulization_source();
        }
        // Preserve the current interpreted implicit-hydrogen counts for atoms
        // whose spelling did not change. Aromaticized graphs can carry
        // sidecar counts that are not recoverable from raw aromatic token
        // syntax alone. Recompute only the atoms whose spelling normalization
        // actually changed their syntax-level hydrogen semantics.
        let implicit_hydrogen_cache = atom_nodes
            .iter()
            .copied()
            .enumerate()
            .map(|(node_id, rewritten_atom)| {
                canonicalization_implicit_hydrogen_count(self, node_id, rewritten_atom)
            })
            .collect::<Vec<_>>();
        Self::from_bond_matrix_parts_with_sidecars(
            atom_nodes,
            self.bond_matrix.clone(),
            self.parsed_stereo_neighbors.clone(),
            implicit_hydrogen_cache,
            None,
        )
    }

    /// Returns the canonical labeling of the current graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let labeling = "OC".parse::<Smiles>()?.canonical_labeling();
    /// assert_eq!(labeling.order().len(), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn canonical_labeling(&self) -> SmilesCanonicalLabeling {
        self.canonicalization_normal_form()
            .collapse_removable_explicit_hydrogens()
            .stereo_normal_form()
            .exact_canonical_labeling()
    }

    /// Returns whether the current graph is already in canonical form.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let canonical = "CO".parse::<Smiles>()?.canonicalize();
    /// assert!(canonical.is_canonical());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn is_canonical(&self) -> bool {
        let canonicalized = self.canonicalize();
        self.atom_nodes == canonicalized.atom_nodes
            && self.bond_matrix == canonicalized.bond_matrix
            && self.parsed_stereo_neighbors == canonicalized.parsed_stereo_neighbors
            && self.implicit_hydrogen_cache == canonicalized.implicit_hydrogen_cache
            && self.kekulization_source == canonicalized.kekulization_source
    }

    /// Returns the graph rewritten into canonical node order.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let canonical = "OC".parse::<Smiles>()?.canonicalize();
    /// assert_eq!(canonical.to_string(), "CO");
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn canonicalize(&self) -> Self {
        self.canonicalize_orbit_min()
    }

    fn canonicalize_orbit_min(&self) -> Self {
        let first = self.canonicalization_step();
        let first_key = canonicalization_state_key(&first);
        let second = first.canonicalization_step();
        let second_key = canonicalization_state_key(&second);

        if second_key == first_key {
            return first;
        }

        let mut states: Vec<Self> = vec![first, second];
        let mut keys: Vec<CanonicalizationStateKey> = vec![first_key, second_key];
        let mut current = states[1].canonicalization_step();

        loop {
            let key = canonicalization_state_key(&current);
            if let Some(cycle_start) = keys.iter().position(|existing| *existing == key) {
                // Orbit-min chooses the lexicographically smallest state from
                // the discovered cycle rather than whichever representative we
                // happened to encounter first.
                let best_relative_index = keys[cycle_start..]
                    .iter()
                    .enumerate()
                    .min_by(|left, right| left.1.cmp(right.1))
                    .map_or_else(|| unreachable!("cycle slice is non-empty"), |(index, _)| index);
                return states[cycle_start + best_relative_index].clone();
            }
            keys.push(key);
            states.push(current.clone());
            current = current.canonicalization_step();
        }
    }

    fn collapse_removable_explicit_hydrogens(&self) -> Self {
        let node_count = self.nodes().len();
        let mut collapsed_parent_of = vec![None; node_count];
        let mut collapsed_count_for_parent = vec![0_u8; node_count];

        for (node_id, collapsed_parent) in collapsed_parent_of.iter_mut().enumerate() {
            let Some(parent) = self.collapsible_explicit_hydrogen_parent(node_id) else {
                continue;
            };
            *collapsed_parent = Some(parent);
            collapsed_count_for_parent[parent] =
                collapsed_count_for_parent[parent].saturating_add(1);
        }

        if collapsed_parent_of.iter().all(Option::is_none) {
            return self.clone();
        }

        let kept_nodes = (0..node_count)
            .filter(|&node_id| collapsed_parent_of[node_id].is_none())
            .collect::<Vec<_>>();
        let mut new_index_of_old_node = vec![usize::MAX; node_count];
        for (new_index, old_node) in kept_nodes.iter().copied().enumerate() {
            new_index_of_old_node[old_node] = new_index;
        }

        let atom_nodes = kept_nodes
            .iter()
            .copied()
            .map(|old_node| {
                atom_with_hydrogen_count(
                    self.atom_nodes[old_node],
                    self.atom_nodes[old_node]
                        .hydrogen_count()
                        .saturating_add(collapsed_count_for_parent[old_node]),
                )
            })
            .collect::<Vec<_>>();

        let mut builder = BondMatrixBuilder::with_capacity(self.number_of_bonds());
        for ((row, column), entry) in self.bond_matrix().sparse_entries() {
            if row >= column
                || collapsed_parent_of[row].is_some()
                || collapsed_parent_of[column].is_some()
            {
                continue;
            }
            builder
                .push_edge_with_descriptor(
                    new_index_of_old_node[row],
                    new_index_of_old_node[column],
                    entry.descriptor(),
                    None,
                )
                .unwrap_or_else(|_| {
                    unreachable!("collapsing terminal hydrogens preserves simplicity")
                });
        }

        let parsed_stereo_neighbors = kept_nodes
            .iter()
            .copied()
            .map(|old_node| {
                self.parsed_stereo_neighbors_row(old_node)
                    .iter()
                    .copied()
                    .filter_map(|neighbor| {
                        match neighbor {
                            StereoNeighbor::ExplicitHydrogen => {
                                Some(StereoNeighbor::ExplicitHydrogen)
                            }
                            StereoNeighbor::Atom(neighbor_id) => {
                                if collapsed_parent_of[neighbor_id] == Some(old_node) {
                                    Some(StereoNeighbor::ExplicitHydrogen)
                                } else if collapsed_parent_of[neighbor_id].is_some() {
                                    None
                                } else {
                                    Some(StereoNeighbor::Atom(new_index_of_old_node[neighbor_id]))
                                }
                            }
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::from_bond_matrix_parts_with_parsed_stereo(
            atom_nodes,
            builder.finish(kept_nodes.len()),
            parsed_stereo_neighbors,
        )
    }

    fn collapsible_explicit_hydrogen_parent(&self, node_id: usize) -> Option<usize> {
        let atom = self.nodes()[node_id];
        if atom.element() != Some(Element::H)
            || atom.syntax() != AtomSyntax::Bracket
            || atom.isotope_mass_number().is_some()
            || atom.aromatic()
            || atom.hydrogen_count() != 0
            || atom.charge_value() != 0
            || atom.class() != 0
            || atom.chirality().is_some()
        {
            return None;
        }

        let mut neighbors = self.bond_matrix().sparse_row(node_id);
        let parent = neighbors.next()?;
        if neighbors.next().is_some() {
            return None;
        }

        let bond = self.edge_for_node_pair((node_id, parent))?.bond();
        if !matches!(bond, Bond::Single | Bond::Up | Bond::Down) {
            return None;
        }

        let parent_atom = self.nodes()[parent];
        if parent_atom.element() == Some(Element::H) {
            return None;
        }
        Some(parent)
    }
}

#[cfg(feature = "fuzzing")]
impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    /// Panics if canonicalization invariants are violated.
    ///
    /// This is intended for fuzzing and other internal validation passes.
    #[doc(hidden)]
    pub fn debug_assert_canonicalization_invariants(&self) {
        support::assert_canonicalization_invariants(self);
    }
}

#[cfg(feature = "fuzzing")]
impl super::WildcardSmiles {
    /// Panics if canonicalization invariants are violated.
    ///
    /// This is intended for fuzzing and other internal validation passes.
    #[doc(hidden)]
    pub fn debug_assert_canonicalization_invariants(&self) {
        self.inner().debug_assert_canonicalization_invariants();
    }
}

#[allow(dead_code)]
fn exact_preserves_aromatic_subgraphs<AtomPolicy: crate::smiles::SmilesAtomPolicy>(
    before: &Smiles<AtomPolicy>,
    after: &Smiles<AtomPolicy>,
) -> bool {
    [
        super::AromaticityPolicy::RdkitDefault,
        super::AromaticityPolicy::RdkitSimple,
        super::AromaticityPolicy::RdkitMdl,
    ]
    .into_iter()
    .all(|policy| {
        let before_assignment = before.aromaticity_assignment_for(policy);
        let after_assignment = after.aromaticity_assignment_for(policy);
        before_assignment.atom_ids() == after_assignment.atom_ids()
            && before_assignment.bond_edges() == after_assignment.bond_edges()
    })
}

fn atom_with_hydrogen_count(atom: Atom, hydrogens: u8) -> Atom {
    match atom.syntax() {
        AtomSyntax::OrganicSubset if hydrogens == atom.hydrogen_count() => atom,
        AtomSyntax::OrganicSubset | AtomSyntax::Bracket => {
            let mut builder = Atom::builder()
                .with_symbol(atom.symbol())
                .with_aromatic(atom.aromatic())
                .with_hydrogens(hydrogens)
                .with_charge(atom.charge())
                .with_class(atom.class());
            if let Some(isotope) = atom.isotope_mass_number() {
                builder = builder.with_isotope(isotope);
            }
            if let Some(chirality) = atom.chirality() {
                builder = builder.with_chirality(chirality);
            }
            builder.build()
        }
    }
}

fn maybe_collapse_atom_to_organic_subset<AtomPolicy: crate::smiles::SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    node_id: usize,
    atom: Atom,
) -> Atom {
    if atom.syntax() != AtomSyntax::Bracket
        || atom.isotope_mass_number().is_some()
        || atom.charge_value() != 0
        || atom.class() != 0
        || atom.chirality().is_some()
        || (atom.aromatic()
            && atom.element().is_none_or(|element| !can_write_unbracketed_aromatic(element)))
        || !canonicalization_valid_unbracketed(atom.symbol())
        || implicit_hydrogens_if_written_unbracketed(smiles, node_id, &atom)
            != atom.hydrogen_count()
    {
        return atom;
    }

    Atom::new_organic_subset(atom.symbol(), atom.aromatic())
}

fn canonicalization_valid_unbracketed(symbol: AtomSymbol) -> bool {
    matches!(
        symbol,
        AtomSymbol::WildCard
            | AtomSymbol::Element(
                Element::B
                    | Element::C
                    | Element::N
                    | Element::O
                    | Element::P
                    | Element::S
                    | Element::F
                    | Element::Cl
                    | Element::Br
                    | Element::I
            )
    )
}

fn canonicalization_atom_spelling_normal_form<AtomPolicy: crate::smiles::SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    node_id: usize,
    atom: Atom,
) -> Atom {
    if atom.syntax() == AtomSyntax::Bracket {
        maybe_collapse_atom_to_organic_subset(smiles, node_id, atom)
    } else {
        atom
    }
}

fn canonicalization_implicit_hydrogen_count(
    smiles: &Smiles<impl crate::smiles::SmilesAtomPolicy>,
    node_id: usize,
    rewritten_atom: Atom,
) -> u8 {
    if rewritten_atom == smiles.nodes()[node_id] {
        return smiles.implicit_hydrogen_count(node_id);
    }
    match rewritten_atom.syntax() {
        AtomSyntax::Bracket => 0,
        AtomSyntax::OrganicSubset => {
            implicit_hydrogens_if_written_unbracketed(smiles, node_id, &rewritten_atom)
        }
    }
}

fn remap_parsed_stereo_neighbors_row<AtomPolicy: crate::smiles::SmilesAtomPolicy>(
    smiles: &Smiles<AtomPolicy>,
    old_node: usize,
    new_index_of_old_node: &[usize],
) -> Vec<StereoNeighbor> {
    smiles
        .parsed_stereo_neighbors
        .get(old_node)
        .into_iter()
        .flatten()
        .copied()
        .map(|neighbor| {
            match neighbor {
                StereoNeighbor::Atom(old_neighbor) => {
                    StereoNeighbor::Atom(new_index_of_old_node[old_neighbor])
                }
                StereoNeighbor::ExplicitHydrogen => StereoNeighbor::ExplicitHydrogen,
            }
        })
        .collect()
}
