use alloc::{boxed::Box, vec::Vec};
use core::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use geometric_traits::{
    impls::{SymmetricCSR2D, ValuedCSR2D},
    traits::{Graph, MonopartiteGraph, MonoplexGraph, SizedSparseMatrix},
};
use hashbrown::HashSet;

use super::{Smiles, WildcardSmiles};
use crate::{
    atom::Atom,
    bond::{
        Bond, BondDescriptor,
        bond_edge::{BondEdge, bond_edge_with_aromaticity},
        ring_num::RingNum,
    },
    errors::SmilesError,
};

/// The symmetric valued sparse matrix storing SMILES bonds.
pub type BondMatrix = SymmetricCSR2D<ValuedCSR2D<usize, usize, usize, BondEntry>>;

#[derive(Clone, Copy, Debug)]
/// Value stored for each bond in the symmetric adjacency matrix.
pub struct BondEntry {
    bond: Bond,
    ring_num: Option<RingNum>,
    aromatic: bool,
    order: usize,
}

impl BondEntry {
    /// Creates a new stored bond value for the adjacency matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::{Bond, ring_num::RingNum},
    ///     smiles::BondEntry,
    /// };
    ///
    /// let entry = BondEntry::new(Bond::Double, Some(RingNum::try_new(1)?), 0);
    /// assert_eq!(entry.bond(), Bond::Double);
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(bond: Bond, ring_num: Option<RingNum>, order: usize) -> Self {
        Self { bond, ring_num, aromatic: false, order }
    }

    /// Creates a new stored bond value from a parsed bond descriptor.
    #[inline]
    #[must_use]
    pub const fn from_descriptor(
        descriptor: BondDescriptor,
        ring_num: Option<RingNum>,
        order: usize,
    ) -> Self {
        Self { bond: descriptor.bond(), ring_num, aromatic: descriptor.is_aromatic(), order }
    }

    /// Returns the bond type stored for this adjacency entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{bond::Bond, smiles::BondEntry};
    ///
    /// let entry = BondEntry::new(Bond::Triple, None, 0);
    /// assert_eq!(entry.bond(), Bond::Triple);
    /// ```
    #[inline]
    #[must_use]
    pub const fn bond(self) -> Bond {
        self.bond
    }

    /// Returns the ring number stored for this adjacency entry, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::{Bond, ring_num::RingNum},
    ///     smiles::BondEntry,
    /// };
    ///
    /// let entry = BondEntry::new(Bond::Single, Some(RingNum::try_new(4)?), 0);
    /// assert_eq!(entry.ring_num().map(|num| num.get()), Some(4));
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    #[inline]
    #[must_use]
    pub const fn ring_num(self) -> Option<RingNum> {
        self.ring_num
    }

    /// Returns whether this bond is aromatic.
    #[inline]
    #[must_use]
    pub const fn aromatic(self) -> bool {
        self.aromatic
    }

    /// Returns this entry's bond order and aromaticity as a descriptor.
    #[inline]
    #[must_use]
    pub const fn descriptor(self) -> BondDescriptor {
        if self.aromatic {
            BondDescriptor::aromatic(self.bond)
        } else {
            BondDescriptor::new(self.bond)
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn order(self) -> usize {
        self.order
    }

    #[inline]
    #[must_use]
    pub(crate) fn to_bond_edge(self, node_a: usize, node_b: usize) -> BondEdge {
        bond_edge_with_aromaticity(node_a, node_b, self.bond, self.ring_num, self.aromatic)
    }

    #[inline]
    #[must_use]
    pub(crate) const fn with_bond(mut self, bond: Bond) -> Self {
        self.bond = bond;
        self
    }

    #[inline]
    #[must_use]
    pub(crate) const fn with_aromatic(mut self, aromatic: bool) -> Self {
        self.aromatic = aromatic;
        self
    }

    #[inline]
    #[must_use]
    pub(crate) const fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }
}

impl PartialEq for BondEntry {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Aromaticity must agree, and the kekule order (single/double) only
        // matters for non-aromatic bonds. This crate stores aromatic rings
        // with an alternating Single/Double kekule order, while RDKit gives
        // every aromatic bond one uniform type, so requiring the kekule order
        // to match would make an aromatic bond stored as Single in one molecule
        // compare unequal to the same bond stored as Double in another. MCES
        // edge compatibility reads this equality, so collapsing the kekule
        // order for aromatic bonds is what lets two aromatic rings
        // match regardless of how their double bonds happen to fall.
        self.aromatic == other.aromatic && (self.aromatic || self.bond == other.bond)
    }
}

impl Eq for BondEntry {}

impl Hash for BondEntry {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Mirror `eq`: the kekule order is only part of the identity for
        // non-aromatic bonds, so it must be dropped from the hash whenever the
        // bond is aromatic to keep equal values hashing equally.
        self.aromatic.hash(state);
        if !self.aromatic {
            self.bond.hash(state);
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BondMatrixBuilder {
    entries: Vec<PendingBond>,
    seen_edges: HashSet<(usize, usize)>,
}

impl BondMatrixBuilder {
    #[inline]
    #[must_use]
    pub(crate) fn with_capacity(number_of_edges: usize) -> Self {
        Self {
            entries: Vec::with_capacity(number_of_edges),
            seen_edges: HashSet::with_capacity(number_of_edges),
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn contains_edge(&self, node_a: usize, node_b: usize) -> bool {
        let (row, column) = crate::smiles::edge_key(node_a, node_b);
        self.seen_edges.contains(&(row, column))
    }

    #[inline]
    pub(crate) fn push_edge_with_descriptor(
        &mut self,
        node_a: usize,
        node_b: usize,
        descriptor: BondDescriptor,
        ring_num: Option<RingNum>,
    ) -> Result<(), SmilesError> {
        let (row, column) = crate::smiles::edge_key(node_a, node_b);
        if row == column {
            return Err(SmilesError::SelfLoopEdge(row));
        }
        if !self.seen_edges.insert((row, column)) {
            return Err(SmilesError::DuplicateEdge(row, column));
        }

        let order = self.entries.len();
        self.entries.push(PendingBond::new(
            row,
            column,
            BondEntry::from_descriptor(descriptor, ring_num, order),
        ));
        Ok(())
    }

    #[inline]
    #[must_use]
    pub(crate) fn finish(self, number_of_nodes: usize) -> BondMatrix {
        build_bond_matrix(number_of_nodes, self.entries)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingBond {
    row: usize,
    column: usize,
    entry: BondEntry,
}

impl PendingBond {
    #[inline]
    #[must_use]
    const fn new(row: usize, column: usize, entry: BondEntry) -> Self {
        Self { row, column, entry }
    }

    #[inline]
    #[must_use]
    const fn row_major_key(self) -> (usize, usize) {
        (self.row, self.column)
    }

    #[inline]
    #[must_use]
    const fn into_entry(self) -> (usize, usize, BondEntry) {
        (self.row, self.column, self.entry)
    }
}

#[inline]
#[must_use]
fn build_bond_matrix(number_of_nodes: usize, mut entries: Vec<PendingBond>) -> BondMatrix {
    reassign_rdkit_bond_orders(&mut entries);
    if !is_row_major_sorted(&entries) {
        entries.sort_unstable_by_key(|bond| bond.row_major_key());
    }
    BondMatrix::from_sorted_upper_triangular_entries(
        number_of_nodes,
        entries.into_iter().map(PendingBond::into_entry),
    )
    .unwrap_or_else(|_| {
        unreachable!("bond entries are unique, upper-triangular, and row-major sorted")
    })
}

#[inline]
#[must_use]
pub(crate) fn build_bond_matrix_from_known_simple_edges(
    number_of_nodes: usize,
    edges: impl IntoIterator<Item = (usize, usize, BondDescriptor, Option<RingNum>)>,
) -> BondMatrix {
    let entries = edges
        .into_iter()
        .enumerate()
        .map(|(order, (row, column, descriptor, ring_num))| {
            debug_assert!(row < column, "known-simple edges must be upper triangular");
            PendingBond::new(row, column, BondEntry::from_descriptor(descriptor, ring_num, order))
        })
        .collect();
    build_bond_matrix(number_of_nodes, entries)
}

#[inline]
fn reassign_rdkit_bond_orders(entries: &mut [PendingBond]) {
    let mut reordered = entries.iter_mut().collect::<Vec<_>>();
    reordered.sort_unstable_by_key(|pending| {
        (pending.entry.ring_num().is_some(), pending.entry.order())
    });
    for (order, pending) in reordered.into_iter().enumerate() {
        pending.entry = pending.entry.with_order(order);
    }
}

#[inline]
fn is_row_major_sorted(entries: &[PendingBond]) -> bool {
    entries.windows(2).all(|window| window[0].row_major_key() <= window[1].row_major_key())
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[inline]
    #[must_use]
    pub(crate) fn from_bond_matrix_parts(atom_nodes: Vec<Atom>, bond_matrix: BondMatrix) -> Self {
        let parsed_stereo_neighbors = vec![Vec::new(); atom_nodes.len()];
        Self::from_bond_matrix_parts_with_parsed_stereo_and_source(
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            None,
        )
    }

    #[inline]
    #[must_use]
    pub(crate) fn from_bond_matrix_parts_with_parsed_stereo(
        atom_nodes: Vec<Atom>,
        bond_matrix: BondMatrix,
        parsed_stereo_neighbors: Vec<Vec<super::StereoNeighbor>>,
    ) -> Self {
        Self::from_bond_matrix_parts_with_parsed_stereo_and_source(
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            None,
        )
    }

    #[inline]
    #[must_use]
    pub(crate) fn from_bond_matrix_parts_with_parsed_stereo_and_source(
        atom_nodes: Vec<Atom>,
        bond_matrix: BondMatrix,
        parsed_stereo_neighbors: Vec<Vec<super::StereoNeighbor>>,
        kekulization_source: Option<Box<Self>>,
    ) -> Self {
        debug_assert_eq!(atom_nodes.len(), parsed_stereo_neighbors.len());
        let mut smiles = Self {
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            implicit_hydrogen_cache: Vec::new(),
            kekulization_source,
            atom_policy: PhantomData,
        };
        smiles.implicit_hydrogen_cache = smiles.recompute_implicit_hydrogen_counts();
        smiles
    }

    #[inline]
    #[must_use]
    pub(crate) fn from_bond_matrix_parts_with_sidecars(
        atom_nodes: Vec<Atom>,
        bond_matrix: BondMatrix,
        parsed_stereo_neighbors: Vec<Vec<super::StereoNeighbor>>,
        implicit_hydrogen_cache: Vec<u8>,
        kekulization_source: Option<Box<Self>>,
    ) -> Self {
        debug_assert_eq!(atom_nodes.len(), parsed_stereo_neighbors.len());
        let smiles = Self {
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            implicit_hydrogen_cache,
            kekulization_source,
            atom_policy: PhantomData,
        };
        assert_eq!(
            smiles.atom_nodes.len(),
            smiles.implicit_hydrogen_cache.len(),
            "implicit hydrogen cache length must match atom count",
        );
        smiles
    }

    /// Returns the symmetric valued sparse matrix storing the graph bonds.
    ///
    /// # Examples
    ///
    /// ```
    /// use geometric_traits::traits::SizedSparseMatrix;
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "C=O".parse()?;
    /// assert_eq!(smiles.bond_matrix().number_of_defined_values(), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn bond_matrix(&self) -> &BondMatrix {
        &self.bond_matrix
    }

    /// Returns the number of unique chemical bonds in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CCO".parse()?;
    /// assert_eq!(smiles.number_of_bonds(), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn number_of_bonds(&self) -> usize {
        self.bond_matrix.number_of_defined_values() / 2
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Graph for Smiles<AtomPolicy> {
    #[inline]
    fn has_nodes(&self) -> bool {
        !self.atom_nodes.is_empty()
    }

    #[inline]
    fn has_edges(&self) -> bool {
        self.number_of_bonds() > 0
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> MonoplexGraph for Smiles<AtomPolicy> {
    type Edge = <BondMatrix as geometric_traits::traits::Edges>::Edge;
    type Edges = BondMatrix;

    #[inline]
    fn edges(&self) -> &Self::Edges {
        &self.bond_matrix
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> MonopartiteGraph for Smiles<AtomPolicy> {
    type NodeId = usize;
    type NodeSymbol = Atom;
    type Nodes = Vec<Atom>;

    #[inline]
    fn nodes_vocabulary(&self) -> &Self::Nodes {
        &self.atom_nodes
    }
}

impl Graph for WildcardSmiles {
    #[inline]
    fn has_nodes(&self) -> bool {
        self.inner().has_nodes()
    }

    #[inline]
    fn has_edges(&self) -> bool {
        self.inner().has_edges()
    }
}

impl MonoplexGraph for WildcardSmiles {
    type Edge = <BondMatrix as geometric_traits::traits::Edges>::Edge;
    type Edges = BondMatrix;

    #[inline]
    fn edges(&self) -> &Self::Edges {
        self.inner().edges()
    }
}

impl MonopartiteGraph for WildcardSmiles {
    type NodeId = usize;
    type NodeSymbol = Atom;
    type Nodes = Vec<Atom>;

    #[inline]
    fn nodes_vocabulary(&self) -> &Self::Nodes {
        self.inner().nodes_vocabulary()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    use elements_rs::Element;
    use geometric_traits::traits::{Graph, GraphSimilarities, McesBuilder};

    use super::*;
    use crate::{atom::atom_symbol::AtomSymbol, bond::ring_num::RingNum};

    /// Builds a two-atom aromatic carbon/nitrogen graph whose single bond is
    /// stored with the given kekule order. The aromatic flag is always set.
    fn aromatic_cn_bond(kekule: Bond) -> Smiles {
        let atoms = alloc::vec![
            Atom::new_organic_subset(AtomSymbol::Element(Element::C), true),
            Atom::new_organic_subset(AtomSymbol::Element(Element::N), true),
        ];
        let mut builder = BondMatrixBuilder::with_capacity(1);
        builder.push_edge_with_descriptor(0, 1, BondDescriptor::aromatic(kekule), None).unwrap();
        Smiles::from_bond_matrix_parts(atoms, builder.finish(2))
    }

    fn assert_similarity_close(actual: impl Into<f64>, expected: f64) {
        let actual = actual.into();
        let difference = (actual - expected).abs();

        assert!(
            difference < 1.0e-6,
            "expected Johnson similarity {expected}, got {actual} (diff {difference})"
        );
    }

    #[test]
    fn mces_matches_aromatic_bonds_across_differing_kekule_orders() {
        // The same aromatic carbon/nitrogen bond stored as Single in one
        // molecule and as Double in another must still be a compatible edge for
        // labeled MCES, since aromatization in this crate leaves an
        // arbitrary kekule order on aromatic bonds while RDKit treats
        // every aromatic bond as one uniform type. Without collapsing
        // the kekule order the two edges would compare unequal and no
        // edge would match.
        let stored_single = aromatic_cn_bond(Bond::Single);
        let stored_double = aromatic_cn_bond(Bond::Double);

        let result = McesBuilder::new(&stored_single, &stored_double).compute_labeled();

        assert_eq!(result.matched_edges().len(), 1);
        assert_similarity_close(result.johnson_similarity(), 1.0);
    }

    #[test]
    fn bond_entry_equality_ignores_ring_digits_and_uses_bond_descriptor() {
        let first = BondEntry::new(Bond::Double, Some(RingNum::try_new(1).unwrap()), 0);
        let second = BondEntry::new(Bond::Double, Some(RingNum::try_new(9).unwrap()), 17);
        let third = BondEntry::new(Bond::Double, None, 99);
        let fourth = BondEntry::new(Bond::Single, Some(RingNum::try_new(1).unwrap()), 0);
        let aromatic = BondEntry::from_descriptor(
            BondDescriptor::aromatic(Bond::Double),
            Some(RingNum::try_new(1).unwrap()),
            0,
        );

        assert_eq!(first, second);
        assert_eq!(first, third);
        assert_ne!(first, fourth);
        assert_ne!(first, aromatic);

        // Aromatic bonds ignore the kekule order, so an aromatic single and an
        // aromatic double compare equal, while non-aromatic bonds still keep
        // the single/double distinction.
        let aromatic_single =
            BondEntry::from_descriptor(BondDescriptor::aromatic(Bond::Single), None, 0);
        let aromatic_double =
            BondEntry::from_descriptor(BondDescriptor::aromatic(Bond::Double), None, 0);
        assert_eq!(aromatic_single, aromatic_double);
        assert_ne!(fourth, aromatic_single);
    }

    #[test]
    fn bond_entry_hash_ignores_ring_digits_and_includes_aromaticity_like_equality() {
        let first = BondEntry::new(Bond::Double, Some(RingNum::try_new(1).unwrap()), 0);
        let second = BondEntry::new(Bond::Double, Some(RingNum::try_new(9).unwrap()), 17);
        let aromatic = BondEntry::from_descriptor(
            BondDescriptor::aromatic(Bond::Double),
            Some(RingNum::try_new(1).unwrap()),
            0,
        );
        let mut first_hasher = DefaultHasher::new();
        let mut second_hasher = DefaultHasher::new();
        let mut aromatic_hasher = DefaultHasher::new();

        first.hash(&mut first_hasher);
        second.hash(&mut second_hasher);
        aromatic.hash(&mut aromatic_hasher);

        assert_eq!(first_hasher.finish(), second_hasher.finish());
        assert_ne!(first_hasher.finish(), aromatic_hasher.finish());

        // Equal aromatic bonds must hash equally even when their kekule order
        // differs, mirroring the equality that ignores it.
        let aromatic_single =
            BondEntry::from_descriptor(BondDescriptor::aromatic(Bond::Single), None, 0);
        let aromatic_double =
            BondEntry::from_descriptor(BondDescriptor::aromatic(Bond::Double), None, 0);
        let mut aromatic_single_hasher = DefaultHasher::new();
        let mut aromatic_double_hasher = DefaultHasher::new();
        aromatic_single.hash(&mut aromatic_single_hasher);
        aromatic_double.hash(&mut aromatic_double_hasher);
        assert_eq!(aromatic_single_hasher.finish(), aromatic_double_hasher.finish());
    }

    #[test]
    fn bond_matrix_builder_rejects_duplicate_edges() {
        let mut builder = BondMatrixBuilder::with_capacity(2);

        builder.push_edge_with_descriptor(0, 1, Bond::Single.into(), None).unwrap();
        let error = builder.push_edge_with_descriptor(1, 0, Bond::Double.into(), None).unwrap_err();

        assert_eq!(error, SmilesError::DuplicateEdge(0, 1));
    }

    #[test]
    fn empty_smiles_reports_no_nodes_or_edges() {
        let smiles = Smiles::<crate::smiles::ConcreteAtoms>::new_for_policy();

        assert_eq!(smiles.number_of_bonds(), 0);
        assert!(!Graph::has_nodes(&smiles));
        assert!(!Graph::has_edges(&smiles));
    }

    #[test]
    fn wildcard_smiles_exposes_graph_traits() {
        let smiles = WildcardSmiles::from_str("*C").unwrap();

        assert!(Graph::has_nodes(&smiles));
        assert!(Graph::has_edges(&smiles));
        assert_eq!(smiles.nodes_vocabulary().len(), 2);
        assert_eq!(smiles.edges().number_of_defined_values(), 2);
    }

    struct DirectionalParityCase {
        name: &'static str,
        smiles1: &'static str,
        smiles2: &'static str,
        raw_edges: usize,
        raw_similarity: f64,
        collapsed_edges: usize,
        collapsed_similarity: f64,
    }

    #[test]
    fn labeled_mces_distinguishes_atom_identity() {
        let ethanol = Smiles::from_str("CCO").unwrap();
        let ethylamine = Smiles::from_str("CCN").unwrap();
        let dimethyl_ether = Smiles::from_str("COC").unwrap();

        let result_one = McesBuilder::new(&ethanol, &ethylamine).compute_labeled();
        let result_two = McesBuilder::new(&ethanol, &dimethyl_ether).compute_labeled();

        assert_eq!(result_one.matched_edges().len(), 1);
        assert_eq!(result_two.matched_edges().len(), 1);
        assert!(result_one.johnson_similarity() < 1.0);
        assert!(result_two.johnson_similarity() < 1.0);
    }

    #[test]
    fn labeled_mces_ignores_bracket_and_explicit_hydrogen_spelling() {
        let implicit = Smiles::from_str("CC").unwrap();
        let explicit = Smiles::from_str("[CH3][CH3]").unwrap();

        let result = McesBuilder::new(&implicit, &explicit).compute_labeled();

        assert_eq!(result.matched_edges().len(), 1);
        assert_similarity_close(result.johnson_similarity(), 1.0);
    }

    #[test]
    fn labeled_mces_rejects_topology_only_ring_false_positive() {
        let benzene = Smiles::from_str("c1ccccc1").unwrap();
        let pyridine = Smiles::from_str("c1ccncc1").unwrap();

        let result = McesBuilder::new(&benzene, &pyridine).compute_labeled();

        assert_eq!(result.matched_edges().len(), 4);
        assert!(result.johnson_similarity() < 1.0);
    }

    #[test]
    fn labeled_mces_ignores_literal_ring_digits_in_bond_labels() {
        let ring_one = Smiles::from_str("C1CCCCC1").unwrap();
        let ring_two = Smiles::from_str("C2CCCCC2").unwrap();

        let result = McesBuilder::new(&ring_one, &ring_two).compute_labeled();

        assert_eq!(result.matched_edges().len(), 6);
        assert_similarity_close(result.johnson_similarity(), 1.0);
    }

    #[test]
    fn directional_bond_collapse_recovers_eight_massspecgym_directional_cases() {
        let cases = [
            DirectionalParityCase {
                name: "massspecgym_default_0012",
                smiles1: "CCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
                smiles2: "CCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCC",
                raw_edges: 42,
                raw_similarity: 0.728_139,
                collapsed_edges: 46,
                collapsed_similarity: 0.868_206,
            },
            DirectionalParityCase {
                name: "massspecgym_default_0015",
                smiles1: "CCCCCCC=CCCCCCCCC(=O)O",
                smiles2: "CCCCCCCC/C=C\\CCCCCCCC(=O)OCC",
                raw_edges: 15,
                raw_similarity: 0.723_588,
                collapsed_edges: 17,
                collapsed_similarity: 0.813_953,
            },
            DirectionalParityCase {
                name: "massspecgym_default_0051",
                smiles1: "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCC(=O)O",
                smiles2: "CCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
                raw_edges: 41,
                raw_similarity: 0.723_406,
                collapsed_edges: 42,
                collapsed_similarity: 0.758_689,
            },
            DirectionalParityCase {
                name: "massspecgym_default_0059",
                smiles1: "CCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC",
                smiles2: "CCCCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCC/C=C\\CCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",
                raw_edges: 45,
                raw_similarity: 0.882_461,
                collapsed_edges: 47,
                collapsed_similarity: 0.960_004,
            },
            DirectionalParityCase {
                name: "massspecgym_default_0069",
                smiles1: "CCCCCC/C=C\\CCCCCCCC(=O)O",
                smiles2: "CCCCCCC=CCCCCCCCCC(=O)NCC(=O)O",
                raw_edges: 14,
                raw_similarity: 0.650_159,
                collapsed_edges: 16,
                collapsed_similarity: 0.733_968,
            },
            DirectionalParityCase {
                name: "massspecgym_default_0090",
                smiles1: "CCCCCCCCCCCC(=O)OC(CCCCCC)C/C=C/CCCCCCCC(=O)O",
                smiles2: "CCCCCCCCCCCCCCCC(=O)OC(CCCCCCC)CCCCCCCCCCC(=O)O",
                raw_edges: 30,
                raw_similarity: 0.745_106,
                collapsed_edges: 32,
                collapsed_similarity: 0.844_350,
            },
            DirectionalParityCase {
                name: "massspecgym_default_0097",
                smiles1: "CCCCCCCC/C=C\\CCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCC/C=C\\CCCCCCCC)COP(=O)(O)OCCN",
                smiles2: "CCCCCCCCCCCCCCCC/C=C\\OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCC",
                raw_edges: 42,
                raw_similarity: 0.720_961,
                collapsed_edges: 46,
                collapsed_similarity: 0.854_829,
            },
            DirectionalParityCase {
                name: "massspecgym_default_0098",
                smiles1: "CCCCC/C=C\\CCCCCCCCCCCCC(=O)O",
                smiles2: "CCCCCC=CCC=CCCCCCCCCC(=O)OCCCCCCCCCCCCCCC(=O)O",
                raw_edges: 19,
                raw_similarity: 0.521_240,
                collapsed_edges: 20,
                collapsed_similarity: 0.546_977,
            },
        ];

        for case in cases {
            let smiles1 = Smiles::from_str(case.smiles1).unwrap();
            let smiles2 = Smiles::from_str(case.smiles2).unwrap();
            let raw = McesBuilder::new(&smiles1, &smiles2).compute_labeled();
            assert_eq!(raw.matched_edges().len(), case.raw_edges, "{}", case.name);
            assert_similarity_close(raw.johnson_similarity(), case.raw_similarity);

            let collapsed1 = smiles1.with_directional_bonds_collapsed();
            let collapsed2 = smiles2.with_directional_bonds_collapsed();
            let collapsed = McesBuilder::new(&collapsed1, &collapsed2).compute_labeled();
            assert_eq!(collapsed.matched_edges().len(), case.collapsed_edges, "{}", case.name);
            assert_similarity_close(collapsed.johnson_similarity(), case.collapsed_similarity);
        }
    }

    #[test]
    fn directional_bond_collapse_improves_but_does_not_fully_resolve_massspecgym_0023() {
        let smiles1 = Smiles::from_str(
            "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCC/C=C\\CCCCCC",
        )
        .unwrap();
        let smiles2 = Smiles::from_str(
            "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCC/C=C\\CCCCCCCC",
        )
        .unwrap();

        let raw = McesBuilder::new(&smiles1, &smiles2).compute_labeled();
        let collapsed1 = smiles1.with_directional_bonds_collapsed();
        let collapsed2 = smiles2.with_directional_bonds_collapsed();
        let collapsed = McesBuilder::new(&collapsed1, &collapsed2).compute_labeled();

        assert_eq!(raw.matched_edges().len(), 43);
        assert_similarity_close(raw.johnson_similarity(), 0.780_422);
        assert_eq!(collapsed.matched_edges().len(), 44);
        assert_similarity_close(collapsed.johnson_similarity(), 0.797_861_065_613_257_5);
        assert!(collapsed.matched_edges().len() > raw.matched_edges().len());
        assert!(collapsed.johnson_similarity() > raw.johnson_similarity());
    }
}
