//! Represents a parsed SMILES graph.
//!
//! A [`Smiles`] value stores atoms as [`Atom`] values and bonds in a
//! symmetric valued sparse matrix.
//!
//! # Examples
//!
//! SMILES strings can be parsed into a graph, inspected, and rendered back
//! into a SMILES string.
//!
//! ```rust
//! use smiles_rs::prelude::Smiles;
//!
//! let source = "CC";
//! let smiles: Smiles = source.parse()?;
//!
//! assert_eq!(smiles.nodes().len(), 2);
//! assert_eq!(smiles.number_of_bonds(), 1);
//! assert_eq!(smiles.to_string(), "CC");
//!
//! # Ok::<(), smiles_rs::errors::SmilesErrorWithSpan>(())
//! ```
use alloc::{boxed::Box, string::String, vec::Vec};
use core::{fmt, marker::PhantomData};

use elements_rs::Element;
use geometric_traits::traits::{
    SizedSparseMatrix2D, SizedSparseValuedMatrixRef, SparseMatrix2D, SparseValuedMatrix2DRef,
    SparseValuedMatrixRef,
};

use crate::{
    atom::{Atom, atom_symbol::AtomSymbol, bracketed::chirality::Chirality},
    bond::bond_edge::BondEdge,
    errors::SmilesError,
};

mod aromaticity;
mod atom_environment;
mod branches;
mod canonicalization;
mod connected_components;
mod double_bond_stereo;
mod emitter;
mod fragment;
mod from_str;
mod geometric_traits_impl;
mod implicit_hydrogens;
mod invariants;
mod kekulization;
mod mces;
mod molecular_formula;
mod neighbors;
mod rdkit_symm_sssr;
mod refinement;
mod render_plan;
mod roots;
mod spanning_tree;
mod stereo;
mod symmetry;

use self::{aromaticity::rdkit_smarts_total_valence, implicit_hydrogens::explicit_valence};
pub use self::{
    aromaticity::{
        AromaticityAssignment, AromaticityAssignmentApplicationError, AromaticityDiagnostic,
        AromaticityModel, AromaticityPerception, AromaticityPolicy, AromaticityRingFamilyKind,
        AromaticityStatus, RdkitDefaultAromaticity, RdkitMdlAromaticity, RdkitSimpleAromaticity,
        WildcardAromaticityPerception,
    },
    atom_environment::AtomEnvironment,
    canonicalization::SmilesCanonicalLabeling,
    connected_components::{SmilesComponents, WildcardSmilesComponents},
    double_bond_stereo::DoubleBondStereoConfig,
    fragment::Fragment,
    geometric_traits_impl::{BondEntry, BondMatrix},
    kekulization::{KekulizationError, KekulizationMode},
    mces::{
        GraphSimilarities, InitialProductVertexOrdering, LargestFragmentMetric, McesBuilder,
        McesResult, McesSearchMode, SmilesMces,
    },
    molecular_formula::WildcardMolecularFormulaConversionError,
};
pub(crate) use self::{
    geometric_traits_impl::{BondMatrixBuilder, build_bond_matrix_from_known_simple_edges},
    stereo::StereoNeighbor,
};

/// Error raised while deriving ring membership from a [`Smiles`] graph.
/// Status flags describing whether symmetrized SSSR completed exactly or
/// required a fallback path.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct SymmSssrStatus {
    used_fallback: bool,
    hit_queue_cutoff: bool,
}

impl SymmSssrStatus {
    /// Returns whether the SSSR search completed without fallback or cutoff.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let result = "c1ccccc1".parse::<Smiles>()?.symm_sssr_result();
    /// assert!(result.status().is_complete());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn is_complete(self) -> bool {
        !self.used_fallback && !self.hit_queue_cutoff
    }

    /// Returns whether the search fell back to the approximate ring finder.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let status = "c1ccccc1".parse::<Smiles>()?.symm_sssr_result().status();
    /// assert!(!status.used_fallback());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn used_fallback(self) -> bool {
        self.used_fallback
    }

    /// Returns whether the search hit the BFS queue cutoff.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let status = "c1ccccc1".parse::<Smiles>()?.symm_sssr_result().status();
    /// assert!(!status.hit_queue_cutoff());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn hit_queue_cutoff(self) -> bool {
        self.hit_queue_cutoff
    }
}

/// Canonicalized symmetrized SSSR cycles together with the search status.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymmSssrResult {
    cycles: Vec<Vec<usize>>,
    status: SymmSssrStatus,
}

impl SymmSssrResult {
    /// Returns the canonicalized cycle set.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let result = "c1ccccc1".parse::<Smiles>()?.symm_sssr_result();
    /// assert_eq!(result.cycles().len(), 1);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn cycles(&self) -> &[Vec<usize>] {
        &self.cycles
    }

    /// Returns the search status for the cycle set.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let result = "c1ccccc1".parse::<Smiles>()?.symm_sssr_result();
    /// assert!(result.status().is_complete());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn status(&self) -> SymmSssrStatus {
        self.status
    }
}

/// Canonicalized ring-membership summary for a [`Smiles`] graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingMembership {
    atom_ids: Vec<usize>,
    bond_edges: Vec<[usize; 2]>,
}

impl RingMembership {
    /// Returns the atom ids that belong to at least one ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_membership();
    /// assert_eq!(ring.atom_ids(), &[0, 1, 2]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn atom_ids(&self) -> &[usize] {
        &self.atom_ids
    }

    /// Returns the bond edges that belong to at least one ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_membership();
    /// assert_eq!(ring.bond_edges(), &[[0, 1], [0, 2], [1, 2]]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn bond_edges(&self) -> &[[usize; 2]] {
        &self.bond_edges
    }

    /// Returns whether the given atom id belongs to at least one ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_membership();
    /// assert!(ring.contains_atom(1));
    /// assert!(!ring.contains_atom(3));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_atom(&self, atom_id: usize) -> bool {
        self.atom_ids.binary_search(&atom_id).is_ok()
    }

    /// Returns whether the given edge belongs to at least one ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_membership();
    /// assert!(ring.contains_edge(0, 1));
    /// assert!(!ring.contains_edge(2, 3));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_edge(&self, node_a: usize, node_b: usize) -> bool {
        let edge = if node_a < node_b { [node_a, node_b] } else { [node_b, node_a] };
        self.bond_edges.binary_search(&edge).is_ok()
    }
}

/// Atom-only ring-membership summary for a [`Smiles`] graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RingAtomMembership {
    atom_flags: Vec<bool>,
}

impl RingAtomMembership {
    /// Returns a flag slice indicating which atom ids belong to at least one
    /// ring.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_atom_membership();
    /// assert_eq!(ring.atom_flags(), &[true, true, true, false]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn atom_flags(&self) -> &[bool] {
        &self.atom_flags
    }

    /// Returns whether the given atom id belongs to at least one ring.
    ///
    /// # Panics
    /// Panics if `atom_id` is not a valid atom index in this summary.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_atom_membership();
    /// assert!(ring.contains_atom(2));
    /// assert!(!ring.contains_atom(3));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn contains_atom(&self, atom_id: usize) -> bool {
        assert!(
            atom_id < self.atom_flags.len(),
            "invalid atom index {atom_id} for ring-atom summary with {} atoms",
            self.atom_flags.len()
        );
        self.atom_flags[atom_id]
    }
}

/// Reusable scratch storage for atom-only ring membership.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RingAtomMembershipScratch {
    discovery_order_u32: Vec<u32>,
    lowlink_u32: Vec<u32>,
    discovery_order: Vec<usize>,
    lowlink: Vec<usize>,
}

mod sealed {
    pub trait Sealed {}
}

/// Type-level policy describing whether a SMILES graph may contain wildcard
/// (`*`) atoms.
///
/// This trait is sealed; use [`ConcreteAtoms`] through the default [`Smiles`]
/// type or use [`WildcardSmiles`] when wildcard atoms are part of the input
/// language.
pub trait SmilesAtomPolicy:
    sealed::Sealed + Copy + Clone + fmt::Debug + PartialEq + Eq + 'static
{
    /// Whether wildcard atoms are accepted by the parser and internal
    /// constructors for this SMILES policy.
    const ALLOW_WILDCARDS: bool;
}

/// Default atom policy for [`Smiles`]: every atom must resolve to a concrete
/// element.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct ConcreteAtoms;

/// Atom policy for [`WildcardSmiles`]: wildcard (`*`) atoms are allowed.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
pub(crate) struct WildcardAtoms;

impl sealed::Sealed for ConcreteAtoms {}
impl sealed::Sealed for WildcardAtoms {}

impl SmilesAtomPolicy for ConcreteAtoms {
    const ALLOW_WILDCARDS: bool = false;
}

impl SmilesAtomPolicy for WildcardAtoms {
    const ALLOW_WILDCARDS: bool = true;
}

/// Represents a parsed SMILES graph.
#[derive(Debug, Clone)]
pub struct Smiles<AtomPolicy = ConcreteAtoms> {
    atom_nodes: Vec<Atom>,
    bond_matrix: BondMatrix,
    parsed_stereo_neighbors: Vec<Vec<StereoNeighbor>>,
    implicit_hydrogen_cache: Vec<u8>,
    kekulization_source: Option<Box<Self>>,
    atom_policy: PhantomData<fn() -> AtomPolicy>,
}

/// A parsed SMILES graph that may contain wildcard (`*`) atoms.
///
/// This type preserves the wildcard-capable behavior that older [`Smiles`]
/// parsing accepted. Use [`Smiles`] when every atom must resolve to a concrete
/// chemical element.
#[derive(Debug, Clone)]
pub struct WildcardSmiles {
    inner: Smiles<WildcardAtoms>,
}

#[inline]
#[must_use]
pub(crate) const fn edge_key(node_a: usize, node_b: usize) -> (usize, usize) {
    if node_a < node_b { (node_a, node_b) } else { (node_b, node_a) }
}

impl Smiles {
    /// Returns a normalized edge key with node IDs in ascending order.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// assert_eq!(Smiles::edge_key(4, 1), (1, 4));
    /// ```
    #[inline]
    #[must_use]
    pub fn edge_key(node_a: usize, node_b: usize) -> (usize, usize) {
        edge_key(node_a, node_b)
    }
}

impl<AtomPolicy: SmilesAtomPolicy> Smiles<AtomPolicy> {
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn new_for_policy() -> Self {
        Self {
            atom_nodes: Vec::new(),
            bond_matrix: BondMatrix::default(),
            parsed_stereo_neighbors: Vec::new(),
            implicit_hydrogen_cache: Vec::new(),
            kekulization_source: None,
            atom_policy: PhantomData,
        }
    }

    /// Returns a slice of all parsed [`Atom`] values.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CO".parse()?;
    /// assert_eq!(smiles.nodes().len(), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn nodes(&self) -> &[Atom] {
        &self.atom_nodes
    }

    /// Returns the atom with the given positional index, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use elements_rs::Element;
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CO".parse()?;
    /// assert_eq!(smiles.node_by_id(1).and_then(|atom| atom.element()), Some(Element::O));
    /// assert!(smiles.node_by_id(99).is_none());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn node_by_id(&self, id: usize) -> Option<&Atom> {
        self.atom_nodes.get(id)
    }

    #[inline]
    #[must_use]
    pub(crate) fn contains_wildcard_atom(&self) -> bool {
        self.atom_nodes.iter().any(|atom| atom.symbol().is_wildcard())
    }

    #[inline]
    #[must_use]
    pub(crate) fn into_atom_policy<TargetPolicy: SmilesAtomPolicy>(self) -> Smiles<TargetPolicy> {
        let Self {
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            implicit_hydrogen_cache,
            kekulization_source,
            atom_policy: _,
        } = self;
        Smiles {
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            implicit_hydrogen_cache,
            kekulization_source: kekulization_source
                .map(|source| Box::new((*source).into_atom_policy())),
            atom_policy: PhantomData,
        }
    }

    /// Returns the bond connecting the given pair of node ids, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::{Bond, bond_edge::BondEdge},
    ///     prelude::Smiles,
    /// };
    ///
    /// let smiles: Smiles = "C=O".parse()?;
    /// assert_eq!(smiles.edge_for_node_pair((0, 1)), Some(BondEdge::new(0, 1, Bond::Double, None)));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn edge_for_node_pair(&self, nodes: (usize, usize)) -> Option<BondEdge> {
        let (row, column) = edge_key(nodes.0, nodes.1);
        let rank = self.bond_matrix.try_rank(row, column)?;
        let entry = *self.bond_matrix.select_value_ref(rank);
        Some(entry.to_bond_edge(row, column))
    }

    #[inline]
    #[must_use]
    pub(crate) fn bond_for_node_pair(&self, nodes: (usize, usize)) -> Option<crate::bond::Bond> {
        let (row, column) = edge_key(nodes.0, nodes.1);
        let rank = self.bond_matrix.try_rank(row, column)?;
        Some(self.bond_matrix.select_value_ref(rank).bond())
    }

    #[inline]
    #[must_use]
    pub(crate) fn bond_entry_for_node_pair(&self, nodes: (usize, usize)) -> Option<BondEntry> {
        let (row, column) = edge_key(nodes.0, nodes.1);
        let rank = self.bond_matrix.try_rank(row, column)?;
        Some(*self.bond_matrix.select_value_ref(rank))
    }

    /// Returns the number of bonds incident to the provided node id.
    ///
    /// # Panics
    /// Panics if `id` is not a valid atom index in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CCO".parse()?;
    /// assert_eq!(smiles.edge_count_for_node(1), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn edge_count_for_node(&self, id: usize) -> usize {
        assert!(
            id < self.atom_nodes.len(),
            "invalid atom index {id} for graph with {} atoms",
            self.atom_nodes.len()
        );
        self.bond_matrix.sparse_row_values_ref(id).count()
    }

    /// Returns the connectivity count for the provided atom id.
    ///
    /// This is the local RDKit-style count used by this crate:
    /// `degree + explicit hydrogens + implicit hydrogens`.
    ///
    /// The result saturates at `u8::MAX` for adversarial inputs whose true
    /// connectivity does not fit into `u8`. Real-chemistry connectivities
    /// always fit comfortably below the saturation point.
    ///
    /// # Panics
    /// Panics if `id` is not a valid atom index in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "C=C".parse()?;
    /// assert_eq!(smiles.connectivity_count(0), 3);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn connectivity_count(&self, id: usize) -> u8 {
        let atom = self.node_by_id(id).unwrap_or_else(|| {
            panic!("invalid atom index {id} for graph with {} atoms", self.atom_nodes.len())
        });
        let connectivity = self
            .edge_count_for_node(id)
            .saturating_add(usize::from(atom.hydrogen_count()))
            .saturating_add(usize::from(self.implicit_hydrogen_count(id)));
        u8::try_from(connectivity).unwrap_or(u8::MAX)
    }

    /// Returns the total valence for the provided atom id.
    ///
    /// This is the local RDKit-style total valence used by this crate:
    /// `bond-order sum + explicit hydrogens + implicit hydrogens`.
    ///
    /// The result saturates at `u8::MAX` for adversarial inputs whose true
    /// valence does not fit into `u8` (e.g., very high-degree atoms with many
    /// quadruple bonds). Real-chemistry valences always fit comfortably below
    /// the saturation point.
    ///
    /// # Panics
    /// Panics if `id` is not a valid atom index in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "C=C".parse()?;
    /// assert_eq!(smiles.total_valence(0), 4);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn total_valence(&self, id: usize) -> u8 {
        let atom = self.node_by_id(id).unwrap_or_else(|| {
            panic!("invalid atom index {id} for graph with {} atoms", self.atom_nodes.len())
        });
        let valence = usize::from(explicit_valence(self, id))
            .saturating_add(usize::from(atom.hydrogen_count()))
            .saturating_add(usize::from(self.implicit_hydrogen_count(id)));
        u8::try_from(valence).unwrap_or(u8::MAX)
    }

    /// Returns the RDKit-style SMARTS total valence for the provided atom id
    /// under the supplied aromaticity assignment.
    ///
    /// This keeps the raw parsed-graph behavior of [`Smiles::total_valence`]
    /// for non-aromatic atoms, but upgrades aromatic atoms to the aromaticity-
    /// aware SMARTS `v` interpretation used by RDKit.
    ///
    /// # Panics
    /// Panics if `id` is not a valid atom index in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::{AromaticityPolicy, Smiles};
    ///
    /// let smiles: Smiles = "c1ccccc1".parse()?;
    /// let aromaticity = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
    /// assert_eq!(smiles.smarts_total_valence(0, &aromaticity), 4);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn smarts_total_valence(&self, id: usize, aromaticity: &AromaticityAssignment) -> u8 {
        assert!(
            id < self.atom_nodes.len(),
            "invalid atom index {id} for graph with {} atoms",
            self.atom_nodes.len()
        );
        rdkit_smarts_total_valence(self, id, aromaticity)
    }

    /// Returns a zero-allocation iterator over the bonds incident to the
    /// provided node id.
    ///
    /// # Panics
    /// Panics if `id` is not a valid atom index in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::{Bond, bond_edge::BondEdge},
    ///     prelude::Smiles,
    /// };
    ///
    /// let smiles: Smiles = "CCO".parse()?;
    /// let edges = smiles.edges_for_node(1).collect::<Vec<_>>();
    ///
    /// assert!(edges.contains(&BondEdge::new(1, 0, Bond::Single, None)));
    /// assert!(edges.contains(&BondEdge::new(1, 2, Bond::Single, None)));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    pub fn edges_for_node(&self, id: usize) -> impl Iterator<Item = BondEdge> + '_ {
        assert!(
            id < self.atom_nodes.len(),
            "invalid atom index {id} for graph with {} atoms",
            self.atom_nodes.len()
        );
        self.bond_matrix
            .sparse_row(id)
            .zip(self.bond_matrix.sparse_row_values_ref(id))
            .map(move |(other, entry)| entry.to_bond_edge(id, other))
    }

    /// Returns semantic tetrahedral or allene-like chirality for SMARTS-style
    /// matching.
    ///
    /// This normalizes parsed `@` / `@@` / `@TH` / `@AL` markup through the
    /// crate's existing stereo-normalization logic, including implicit-
    /// hydrogen handling and stereogenicity checks. Only tetrahedral and
    /// allene-like semantic chirality is exposed here; other parsed stereo
    /// families return `None`.
    ///
    /// # Panics
    /// Panics if `atom_id` is not a valid atom index in this graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{atom::bracketed::chirality::Chirality, prelude::Smiles};
    ///
    /// let left: Smiles = "F[C@H](Cl)Br".parse()?;
    /// let right: Smiles = "F[C@@H](Cl)Br".parse()?;
    ///
    /// assert!(matches!(left.smarts_tetrahedral_chirality(1), Some(Chirality::TH(_))));
    /// assert!(matches!(right.smarts_tetrahedral_chirality(1), Some(Chirality::TH(_))));
    /// assert_ne!(left.smarts_tetrahedral_chirality(1), right.smarts_tetrahedral_chirality(1));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn smarts_tetrahedral_chirality(&self, atom_id: usize) -> Option<Chirality> {
        assert!(
            atom_id < self.atom_nodes.len(),
            "invalid atom index {atom_id} for graph with {} atoms",
            self.atom_nodes.len()
        );
        self.semantic_tetrahedral_chirality(atom_id)
    }

    /// Returns semantic `E`/`Z` stereo for the requested double bond, if any.
    ///
    /// This is the chemistry-facing stereo classification derived from parsed
    /// directional single-bond tokens. It returns `None` when the edge is not
    /// a double bond, when stereo was not specified, or when the surrounding
    /// environment does not support a semantic alkene-stereo assignment.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{DoubleBondStereoConfig, prelude::Smiles};
    ///
    /// let trans: Smiles = "F/C=C/F".parse()?;
    /// let plain: Smiles = "CC=CC".parse()?;
    ///
    /// assert_eq!(trans.double_bond_stereo_config(1, 2), Some(DoubleBondStereoConfig::E));
    /// assert_eq!(plain.double_bond_stereo_config(1, 2), None);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn double_bond_stereo_config(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> Option<DoubleBondStereoConfig> {
        self.semantic_double_bond_stereo_config(node_a, node_b)
    }

    /// Returns the atoms and bonds that belong to at least one ring.
    ///
    /// Ring membership is derived from the cyclic biconnected components of
    /// the graph, so chain attachments and bridges are excluded.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_membership();
    /// assert_eq!(ring.atom_ids(), &[0, 1, 2]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn ring_membership(&self) -> RingMembership {
        let bond_count = self.number_of_bonds();
        if self.atom_nodes.len() < 3 || bond_count < 3 {
            return RingMembership { atom_ids: Vec::new(), bond_edges: Vec::new() };
        }
        if u32::try_from(self.atom_nodes.len()).is_ok() {
            return self.ring_membership_with_packed_bridge_keys(bond_count);
        }

        let mut discovery_order = vec![0_usize; self.atom_nodes.len()];
        let mut lowlink = vec![0_usize; self.atom_nodes.len()];
        let mut time = 0_usize;
        let mut bridge_edges = Vec::<[usize; 2]>::with_capacity(bond_count);

        for start_atom_id in 0..self.atom_nodes.len() {
            if discovery_order[start_atom_id] == 0 {
                self.find_bridge_edges_depth_first(
                    start_atom_id,
                    None,
                    &mut time,
                    &mut discovery_order,
                    &mut lowlink,
                    &mut bridge_edges,
                );
            }
        }
        if bridge_edges.len() == bond_count {
            return RingMembership { atom_ids: Vec::new(), bond_edges: Vec::new() };
        }
        if bridge_edges.is_empty() {
            lowlink.fill(0);
            return self.ring_membership_from_all_unique_edges(bond_count, &mut lowlink);
        }
        bridge_edges.sort_unstable();

        lowlink.fill(0);
        let mut bond_edges = Vec::with_capacity(bond_count.saturating_sub(bridge_edges.len()));
        let mut next_bridge = 0_usize;
        for row in 0..self.atom_nodes.len() {
            for column in self.bond_matrix.sparse_row(row) {
                if row >= column {
                    continue;
                }
                let edge = [row, column];
                while let Some(bridge) = bridge_edges.get(next_bridge) {
                    if *bridge < edge {
                        next_bridge += 1;
                    } else {
                        break;
                    }
                }
                if bridge_edges.get(next_bridge) == Some(&edge) {
                    continue;
                }
                lowlink[row] = 1;
                lowlink[column] = 1;
                bond_edges.push(edge);
            }
        }

        let atom_ids = lowlink
            .into_iter()
            .enumerate()
            .filter_map(|(atom_id, is_ring_atom)| (is_ring_atom != 0).then_some(atom_id))
            .collect();

        RingMembership { atom_ids, bond_edges }
    }

    /// Returns atom-only ring membership for the graph.
    ///
    /// This is derived from the same cyclic biconnected components as
    /// [`Self::ring_membership()`], but it only materializes per-atom boolean
    /// flags and skips ring-bond collection.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let ring = "C1CC1O".parse::<Smiles>()?.ring_atom_membership();
    /// assert_eq!(ring.atom_flags(), &[true, true, true, false]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn ring_atom_membership(&self) -> RingAtomMembership {
        let mut ring_atom_membership = RingAtomMembership::default();
        let mut scratch = RingAtomMembershipScratch::default();
        self.write_ring_atom_membership(&mut ring_atom_membership, &mut scratch);
        ring_atom_membership
    }

    /// Writes atom-only ring membership into caller-provided output and scratch
    /// buffers.
    ///
    /// This computes the same result as [`Self::ring_atom_membership()`], but
    /// it allows callers processing many molecules to reuse allocations across
    /// calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{RingAtomMembership, RingAtomMembershipScratch, prelude::Smiles};
    ///
    /// let smiles: Smiles = "C1CC1O".parse()?;
    /// let mut membership = RingAtomMembership::default();
    /// let mut scratch = RingAtomMembershipScratch::default();
    ///
    /// smiles.write_ring_atom_membership(&mut membership, &mut scratch);
    /// assert_eq!(membership.atom_flags(), &[true, true, true, false]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    pub fn write_ring_atom_membership(
        &self,
        ring_atom_membership: &mut RingAtomMembership,
        scratch: &mut RingAtomMembershipScratch,
    ) {
        let atom_count = self.atom_nodes.len();
        ring_atom_membership.atom_flags.resize(atom_count, false);
        ring_atom_membership.atom_flags.fill(false);

        let bond_count = self.number_of_bonds();
        if atom_count < 3 || bond_count < 3 {
            return;
        }
        if u32::try_from(atom_count).is_ok() {
            scratch.discovery_order_u32.resize(atom_count, 0);
            scratch.discovery_order_u32.fill(0);
            scratch.lowlink_u32.resize(atom_count, 0);
            scratch.lowlink_u32.fill(0);
            let mut time = 0_u32;

            for start_atom_id in 0..atom_count {
                if scratch.discovery_order_u32[start_atom_id] == 0 {
                    self.mark_ring_atom_flags_depth_first_u32(
                        start_atom_id,
                        None,
                        &mut time,
                        &mut scratch.discovery_order_u32,
                        &mut scratch.lowlink_u32,
                        &mut ring_atom_membership.atom_flags,
                    );
                }
            }
        } else {
            scratch.discovery_order.resize(atom_count, 0);
            scratch.discovery_order.fill(0);
            scratch.lowlink.resize(atom_count, 0);
            scratch.lowlink.fill(0);
            let mut time = 0_usize;

            for start_atom_id in 0..atom_count {
                if scratch.discovery_order[start_atom_id] == 0 {
                    self.mark_ring_atom_flags_depth_first(
                        start_atom_id,
                        None,
                        &mut time,
                        &mut scratch.discovery_order,
                        &mut scratch.lowlink,
                        &mut ring_atom_membership.atom_flags,
                    );
                }
            }
        }
    }

    /// Returns the canonicalized symmetric-SSSR-style cycle set together with
    /// search completeness status.
    ///
    /// The current implementation mirrors `RDKit`'s ring-finding pipeline,
    /// including its approximate fallback behavior on some highly fused
    /// systems.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let result = "c1ccccc1".parse::<Smiles>()?.symm_sssr_result();
    /// assert_eq!(result.cycles().len(), 1);
    /// assert!(result.status().is_complete());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn symm_sssr_result(&self) -> SymmSssrResult {
        let ring_membership = self.ring_membership();
        rdkit_symm_sssr::symmetrize_sssr_with_ring_membership(self, &ring_membership)
    }

    /// Returns a graph with directional single bonds collapsed to ordinary
    /// single bonds.
    ///
    /// This is intended for chemistry-facing matching paths where bond order
    /// should be compared without carrying slash/backslash directional notation
    /// through the MCES bond type.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{bond::Bond, prelude::Smiles};
    ///
    /// let raw: Smiles = "C/C=C\\C".parse()?;
    /// let collapsed = raw.with_directional_bonds_collapsed();
    ///
    /// assert_eq!(collapsed.edge_for_node_pair((0, 1)).unwrap().bond(), Bond::Single);
    /// assert_eq!(collapsed.edge_for_node_pair((2, 3)).unwrap().bond(), Bond::Single);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn with_directional_bonds_collapsed(&self) -> Self {
        let bond_matrix = BondMatrix::from_sorted_upper_triangular_entries(
            self.atom_nodes.len(),
            self.bond_matrix.sparse_entries().filter_map(|((row, column), entry)| {
                (row < column).then_some((
                    row,
                    column,
                    entry.with_bond(entry.bond().without_direction()),
                ))
            }),
        )
        .unwrap_or_else(|_| unreachable!("existing bond matrix entries are already valid"));

        Self {
            atom_nodes: self.atom_nodes.clone(),
            bond_matrix,
            parsed_stereo_neighbors: self.parsed_stereo_neighbors.clone(),
            implicit_hydrogen_cache: self.implicit_hydrogen_cache.clone(),
            kekulization_source: self.kekulization_source.clone(),
            atom_policy: PhantomData,
        }
    }

    /// Returns a variant of the graph with all currently implied hydrogens
    /// materialized as terminal `[H]` atoms.
    ///
    /// This mirrors the structural effect of `RDKit`'s `AddHs`: bracket-
    /// explicit hydrogens and current implicit hydrogens are turned into real
    /// hydrogen nodes connected by single bonds, and the parent atom's local
    /// hydrogen count is cleared.
    ///
    /// Existing heavy-atom bonds, charges, classes, and chirality tags are
    /// preserved. Stereo-neighbor rows are rewritten so chiral `[XH]` centers
    /// point at the new hydrogen nodes instead of keeping an
    /// `ExplicitHydrogen` placeholder.
    ///
    /// # Examples
    ///
    /// ```
    /// use elements_rs::Element;
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CO".parse()?;
    /// let explicit = smiles.with_explicit_hydrogens();
    ///
    /// assert_eq!(explicit.nodes().len(), 6);
    /// assert_eq!(explicit.implicit_hydrogen_counts(), &[0, 0, 0, 0, 0, 0]);
    /// assert_eq!(
    ///     explicit.nodes().iter().filter(|atom| atom.element() == Some(Element::H)).count(),
    ///     4
    /// );
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn with_explicit_hydrogens(&self) -> Self {
        let original_atom_count = self.atom_nodes.len();
        let mut added_hydrogen_counts = Vec::with_capacity(original_atom_count);
        let mut hydrogen_start_by_parent = Vec::with_capacity(original_atom_count);
        let mut total_added_hydrogens = 0_usize;
        for (atom_id, atom) in self.atom_nodes.iter().enumerate() {
            hydrogen_start_by_parent.push(original_atom_count + total_added_hydrogens);
            let hydrogen_count = usize::from(atom.hydrogen_count())
                + usize::from(self.implicit_hydrogen_count(atom_id));
            added_hydrogen_counts.push(hydrogen_count);
            total_added_hydrogens += hydrogen_count;
        }
        if total_added_hydrogens == 0 {
            return self.clone();
        }

        let mut atom_nodes = self
            .atom_nodes
            .iter()
            .copied()
            .map(atom_without_explicit_hydrogen_count)
            .collect::<Vec<_>>();
        atom_nodes.reserve(total_added_hydrogens);
        atom_nodes.extend((0..total_added_hydrogens).map(|_| explicit_hydrogen_atom()));

        let mut parsed_stereo_neighbors = self.parsed_stereo_neighbors.clone();
        for atom_id in 0..original_atom_count {
            if added_hydrogen_counts[atom_id] == 0 {
                continue;
            }
            let hydrogen_start = hydrogen_start_by_parent[atom_id];
            let mut next_explicit_hydrogen = 0_usize;
            for neighbor in &mut parsed_stereo_neighbors[atom_id] {
                if matches!(*neighbor, StereoNeighbor::ExplicitHydrogen) {
                    let hydrogen_id = hydrogen_start + next_explicit_hydrogen;
                    *neighbor = StereoNeighbor::Atom(hydrogen_id);
                    next_explicit_hydrogen += 1;
                }
            }
        }
        parsed_stereo_neighbors.extend((0..total_added_hydrogens).map(|_| Vec::new()));

        let bond_matrix = build_bond_matrix_from_known_simple_edges(
            original_atom_count + total_added_hydrogens,
            (0..original_atom_count).flat_map(|parent_atom_id| {
                let hydrogen_start = hydrogen_start_by_parent[parent_atom_id];
                let hydrogen_count = added_hydrogen_counts[parent_atom_id];
                self.bond_matrix
                    .sparse_row(parent_atom_id)
                    .zip(self.bond_matrix.sparse_row_values_ref(parent_atom_id))
                    .filter_map(move |(neighbor_atom_id, entry)| {
                        (parent_atom_id < neighbor_atom_id).then_some((
                            parent_atom_id,
                            neighbor_atom_id,
                            entry.descriptor(),
                            entry.ring_num(),
                        ))
                    })
                    .chain((0..hydrogen_count).map(move |offset| {
                        (
                            parent_atom_id,
                            hydrogen_start + offset,
                            crate::bond::Bond::Single.into(),
                            None,
                        )
                    }))
            }),
        );

        Self::from_bond_matrix_parts_with_sidecars(
            atom_nodes,
            bond_matrix,
            parsed_stereo_neighbors,
            vec![0; original_atom_count + total_added_hydrogens],
            None,
        )
    }

    #[inline]
    #[must_use]
    pub(crate) fn clone_without_kekulization_source(&self) -> Self {
        Self {
            atom_nodes: self.atom_nodes.clone(),
            bond_matrix: self.bond_matrix.clone(),
            parsed_stereo_neighbors: self.parsed_stereo_neighbors.clone(),
            implicit_hydrogen_cache: self.implicit_hydrogen_cache.clone(),
            kekulization_source: None,
            atom_policy: PhantomData,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) fn resolved_kekulization_source(&self) -> Box<Self> {
        self.kekulization_source
            .clone()
            .unwrap_or_else(|| Box::new(self.clone_without_kekulization_source()))
    }

    /// Renders the graph back into a SMILES string.
    ///
    /// This is the main entry point for display. Rendering is split into a
    /// deterministic planning phase followed by a write-only emission phase:
    ///
    /// 1. compute atom-local invariants
    /// 2. run seeded Weisfeiler-Lehman refinement over the labeled molecular
    ///    graph
    /// 3. derive rooted symmetry tie-break classes on top of the refined
    ///    partition
    /// 4. choose one root per connected component
    /// 5. order each atom's incident edges deterministically
    /// 6. build a rooted DFS spanning forest
    /// 7. choose continuation versus branch children from subtree signatures
    /// 8. compute a final preorder for each component
    /// 9. schedule closure events and assign reusable ring labels
    /// 10. normalize stereo relative to the final traversal, with a raw
    ///     directional-single fallback for unsupported double-bond environments
    /// 11. emit a string from the finished plan
    ///
    /// The key design constraint is that ordering decisions come from
    /// graph-derived keys and explicit plan objects. The emitter does not
    /// canonicalize, search, or preview alternate strings while writing.
    ///
    /// [`fmt::Display`] uses this exact pipeline by delegating to `render()`
    /// and then writing the finished string into the formatter.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "C=O".parse()?;
    /// assert_eq!(smiles.render(), "C=O");
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn render(&self) -> String {
        self::emitter::emit(self)
    }

    /// Renders this graph as SMILES with the traversal forced to start at
    /// `root`.
    ///
    /// All ordering decisions other than the root are identical to
    /// [`render`](Self::render), so the output is deterministic and parses back
    /// to the same molecule. Only the connected component containing `root` is
    /// re-rooted, leaving any other components on their chosen roots.
    ///
    /// # Panics
    ///
    /// Panics if `root` is not a valid atom id.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CCO".parse()?;
    /// assert!(smiles.render_rooted(2).starts_with('O'));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn render_rooted(&self, root: usize) -> String {
        assert!(root < self.nodes().len(), "render_rooted: root {root} is out of range");
        self::emitter::emit_rooted(self, root)
    }

    /// Returns the traversal order used by
    /// [`render_rooted`](Self::render_rooted) as a [`SmilesCanonicalLabeling`].
    /// For a connected graph this places `root` at ordinal 0. When the graph
    /// has several components, `root` leads its own component's run and the
    /// components keep their render order.
    ///
    /// # Panics
    ///
    /// Panics if `root` is not a valid atom id.
    #[must_use]
    pub fn canonical_labeling_rooted(&self, root: usize) -> SmilesCanonicalLabeling {
        assert!(root < self.nodes().len(), "canonical_labeling_rooted: root {root} out of range");
        let order = self.render_plan_with_root(Some(root)).traversal_order();
        SmilesCanonicalLabeling::from_order(order)
    }

    /// Returns a copy with isomeric features removed: isotope labels and
    /// tetrahedral chirality are cleared on every atom, and directional
    /// (double-bond stereo) bonds are flattened to plain bonds. Implicit
    /// hydrogen counts are recomputed for the result.
    ///
    /// This mirrors RDKit's `isomericSmiles=False`: two graphs that differ only
    /// in stereochemistry or isotope labeling collapse to the same molecule.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let labeled: Smiles = "[13CH3]O".parse()?;
    /// let plain: Smiles = "[12CH3]O".parse()?;
    /// assert_eq!(labeled.non_isomeric().render(), plain.non_isomeric().render());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn non_isomeric(&self) -> Self {
        let atom_nodes: Vec<_> = self.atom_nodes.iter().map(|atom| atom.non_isomeric()).collect();
        let mut builder = crate::smiles::BondMatrixBuilder::default();
        for ((row, column), entry) in self.bond_matrix.sparse_entries() {
            if row >= column {
                continue;
            }
            let descriptor = entry.descriptor();
            let flattened = descriptor.with_bond(descriptor.bond().without_direction());
            builder
                .push_edge_with_descriptor(row, column, flattened, None)
                .unwrap_or_else(|_| unreachable!("non-isomeric copy preserves a simple graph"));
        }
        let parsed_stereo_neighbors = vec![Vec::new(); atom_nodes.len()];
        let result = Self::from_bond_matrix_parts_with_parsed_stereo_and_source(
            atom_nodes,
            builder.finish(self.atom_nodes.len()),
            parsed_stereo_neighbors,
            None,
        );
        // Clearing isotope and stereo can leave an atom bracketed for a reason
        // that no longer applies (e.g. `[13CH3]` -> `[CH3]`). Collapse such
        // atoms to the bare organic-subset form using the same eligibility rule
        // as canonicalization, so `non_isomeric()` is order-independent with
        // respect to `canonicalize()`.
        result.canonicalization_spelling_normal_form()
    }

    fn ring_membership_with_packed_bridge_keys(&self, bond_count: usize) -> RingMembership {
        let mut discovery_order = vec![0_usize; self.atom_nodes.len()];
        let mut lowlink = vec![0_usize; self.atom_nodes.len()];
        let mut time = 0_usize;
        let mut bridge_keys = Vec::<u64>::with_capacity(bond_count);

        for start_atom_id in 0..self.atom_nodes.len() {
            if discovery_order[start_atom_id] == 0 {
                self.find_bridge_keys_depth_first(
                    start_atom_id,
                    None,
                    &mut time,
                    &mut discovery_order,
                    &mut lowlink,
                    &mut bridge_keys,
                );
            }
        }
        if bridge_keys.len() == bond_count {
            return RingMembership { atom_ids: Vec::new(), bond_edges: Vec::new() };
        }
        if bridge_keys.is_empty() {
            lowlink.fill(0);
            return self.ring_membership_from_all_unique_edges(bond_count, &mut lowlink);
        }
        bridge_keys.sort_unstable();

        lowlink.fill(0);
        let mut bond_edges = Vec::with_capacity(bond_count.saturating_sub(bridge_keys.len()));
        let mut next_bridge = 0_usize;
        for row in 0..self.atom_nodes.len() {
            for column in self.bond_matrix.sparse_row(row) {
                if row >= column {
                    continue;
                }
                let edge_key = Self::packed_edge_key(row, column);
                while let Some(bridge_key) = bridge_keys.get(next_bridge) {
                    if *bridge_key < edge_key {
                        next_bridge += 1;
                    } else {
                        break;
                    }
                }
                if bridge_keys.get(next_bridge) == Some(&edge_key) {
                    continue;
                }
                lowlink[row] = 1;
                lowlink[column] = 1;
                bond_edges.push([row, column]);
            }
        }

        let atom_ids = lowlink
            .into_iter()
            .enumerate()
            .filter_map(|(atom_id, is_ring_atom)| (is_ring_atom != 0).then_some(atom_id))
            .collect();

        RingMembership { atom_ids, bond_edges }
    }

    fn ring_membership_from_all_unique_edges(
        &self,
        bond_count: usize,
        ring_atom_flags: &mut [usize],
    ) -> RingMembership {
        let mut bond_edges = Vec::with_capacity(bond_count);

        for row in 0..self.atom_nodes.len() {
            for column in self.bond_matrix.sparse_row(row) {
                if row >= column {
                    continue;
                }
                ring_atom_flags[row] = 1;
                ring_atom_flags[column] = 1;
                bond_edges.push([row, column]);
            }
        }

        let atom_ids = ring_atom_flags
            .iter()
            .enumerate()
            .filter_map(|(atom_id, &is_ring_atom)| (is_ring_atom != 0).then_some(atom_id))
            .collect();

        RingMembership { atom_ids, bond_edges }
    }

    #[inline]
    fn packed_edge_key(node_a: usize, node_b: usize) -> u64 {
        let node_a =
            u32::try_from(node_a).unwrap_or_else(|_| unreachable!("guarded by ring_membership"));
        let node_b =
            u32::try_from(node_b).unwrap_or_else(|_| unreachable!("guarded by ring_membership"));
        (u64::from(node_a) << 32) | u64::from(node_b)
    }

    fn find_bridge_keys_depth_first(
        &self,
        atom_id: usize,
        parent_atom_id: Option<usize>,
        time: &mut usize,
        discovery_order: &mut [usize],
        lowlink: &mut [usize],
        bridge_keys: &mut Vec<u64>,
    ) {
        *time += 1;
        discovery_order[atom_id] = *time;
        lowlink[atom_id] = *time;

        for neighbor_atom_id in self.bond_matrix.sparse_row(atom_id) {
            if discovery_order[neighbor_atom_id] == 0 {
                self.find_bridge_keys_depth_first(
                    neighbor_atom_id,
                    Some(atom_id),
                    time,
                    discovery_order,
                    lowlink,
                    bridge_keys,
                );
                lowlink[atom_id] = lowlink[atom_id].min(lowlink[neighbor_atom_id]);
                if lowlink[neighbor_atom_id] > discovery_order[atom_id] {
                    let (row, column) = edge_key(atom_id, neighbor_atom_id);
                    bridge_keys.push(Self::packed_edge_key(row, column));
                }
            } else if parent_atom_id != Some(neighbor_atom_id) {
                lowlink[atom_id] = lowlink[atom_id].min(discovery_order[neighbor_atom_id]);
            }
        }
    }

    fn find_bridge_edges_depth_first(
        &self,
        atom_id: usize,
        parent_atom_id: Option<usize>,
        time: &mut usize,
        discovery_order: &mut [usize],
        lowlink: &mut [usize],
        bridge_edges: &mut Vec<[usize; 2]>,
    ) {
        *time += 1;
        discovery_order[atom_id] = *time;
        lowlink[atom_id] = *time;

        for neighbor_atom_id in self.bond_matrix.sparse_row(atom_id) {
            if discovery_order[neighbor_atom_id] == 0 {
                self.find_bridge_edges_depth_first(
                    neighbor_atom_id,
                    Some(atom_id),
                    time,
                    discovery_order,
                    lowlink,
                    bridge_edges,
                );
                lowlink[atom_id] = lowlink[atom_id].min(lowlink[neighbor_atom_id]);
                if lowlink[neighbor_atom_id] > discovery_order[atom_id] {
                    bridge_edges.push(if atom_id < neighbor_atom_id {
                        [atom_id, neighbor_atom_id]
                    } else {
                        [neighbor_atom_id, atom_id]
                    });
                }
            } else if parent_atom_id != Some(neighbor_atom_id) {
                lowlink[atom_id] = lowlink[atom_id].min(discovery_order[neighbor_atom_id]);
            }
        }
    }

    fn mark_ring_atom_flags_depth_first(
        &self,
        atom_id: usize,
        parent_atom_id: Option<usize>,
        time: &mut usize,
        discovery_order: &mut [usize],
        lowlink: &mut [usize],
        atom_flags: &mut [bool],
    ) {
        *time += 1;
        discovery_order[atom_id] = *time;
        lowlink[atom_id] = *time;

        for neighbor_atom_id in self.bond_matrix.sparse_row(atom_id) {
            if discovery_order[neighbor_atom_id] == 0 {
                self.mark_ring_atom_flags_depth_first(
                    neighbor_atom_id,
                    Some(atom_id),
                    time,
                    discovery_order,
                    lowlink,
                    atom_flags,
                );
                lowlink[atom_id] = lowlink[atom_id].min(lowlink[neighbor_atom_id]);
                if lowlink[neighbor_atom_id] <= discovery_order[atom_id] {
                    atom_flags[atom_id] = true;
                    atom_flags[neighbor_atom_id] = true;
                }
            } else if parent_atom_id != Some(neighbor_atom_id) {
                let neighbor_discovery_order = discovery_order[neighbor_atom_id];
                lowlink[atom_id] = lowlink[atom_id].min(neighbor_discovery_order);
                if neighbor_discovery_order < discovery_order[atom_id] {
                    atom_flags[atom_id] = true;
                    atom_flags[neighbor_atom_id] = true;
                }
            }
        }
    }

    fn mark_ring_atom_flags_depth_first_u32(
        &self,
        atom_id: usize,
        parent_atom_id: Option<usize>,
        time: &mut u32,
        discovery_order: &mut [u32],
        lowlink: &mut [u32],
        atom_flags: &mut [bool],
    ) {
        *time += 1;
        discovery_order[atom_id] = *time;
        lowlink[atom_id] = *time;

        for neighbor_atom_id in self.bond_matrix.sparse_row(atom_id) {
            if discovery_order[neighbor_atom_id] == 0 {
                self.mark_ring_atom_flags_depth_first_u32(
                    neighbor_atom_id,
                    Some(atom_id),
                    time,
                    discovery_order,
                    lowlink,
                    atom_flags,
                );
                lowlink[atom_id] = lowlink[atom_id].min(lowlink[neighbor_atom_id]);
                if lowlink[neighbor_atom_id] <= discovery_order[atom_id] {
                    atom_flags[atom_id] = true;
                    atom_flags[neighbor_atom_id] = true;
                }
            } else if parent_atom_id != Some(neighbor_atom_id) {
                let neighbor_discovery_order = discovery_order[neighbor_atom_id];
                lowlink[atom_id] = lowlink[atom_id].min(neighbor_discovery_order);
                if neighbor_discovery_order < discovery_order[atom_id] {
                    atom_flags[atom_id] = true;
                    atom_flags[neighbor_atom_id] = true;
                }
            }
        }
    }
}

fn explicit_hydrogen_atom() -> Atom {
    Atom::builder().with_symbol(AtomSymbol::Element(Element::H)).build()
}

fn atom_without_explicit_hydrogen_count(atom: Atom) -> Atom {
    if atom.hydrogen_count() == 0 {
        return atom;
    }

    let mut builder = Atom::builder()
        .with_symbol(atom.symbol())
        .with_aromatic(atom.aromatic())
        .with_hydrogens(0)
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

fn canonicalize_cycle(cycle: &[usize]) -> Vec<usize> {
    let smallest_atom_id =
        *cycle.iter().min().unwrap_or_else(|| unreachable!("Johnson only yields non-empty cycles"));
    let start_index = cycle
        .iter()
        .position(|&atom_id| atom_id == smallest_atom_id)
        .unwrap_or_else(|| unreachable!("the smallest atom id is taken from the same cycle"));

    let forward =
        cycle[start_index..].iter().chain(cycle[..start_index].iter()).copied().collect::<Vec<_>>();

    let reversed = cycle.iter().rev().copied().collect::<Vec<_>>();
    let reversed_start_index = reversed
        .iter()
        .position(|&atom_id| atom_id == smallest_atom_id)
        .unwrap_or_else(|| unreachable!("the smallest atom id is taken from the same cycle"));
    let backward = reversed[reversed_start_index..]
        .iter()
        .chain(reversed[..reversed_start_index].iter())
        .copied()
        .collect::<Vec<_>>();

    forward.min(backward)
}

fn cycle_edges(cycle: &[usize]) -> Vec<[usize; 2]> {
    let number_of_nodes = cycle.len();
    (0..number_of_nodes)
        .map(|index| {
            let next_index = (index + 1) % number_of_nodes;
            let left = cycle[index];
            let right = cycle[next_index];
            if left <= right { [left, right] } else { [right, left] }
        })
        .collect()
}

impl WildcardSmiles {
    #[inline]
    #[must_use]
    pub(crate) fn from_inner(inner: Smiles<WildcardAtoms>) -> Self {
        Self { inner }
    }

    #[inline]
    #[must_use]
    pub(crate) fn inner(&self) -> &Smiles<WildcardAtoms> {
        &self.inner
    }

    /// Returns a slice of all parsed [`Atom`] values.
    #[inline]
    #[must_use]
    pub fn nodes(&self) -> &[Atom] {
        self.inner.nodes()
    }

    /// Returns the atom with the given positional index, if present.
    #[inline]
    #[must_use]
    pub fn node_by_id(&self, id: usize) -> Option<&Atom> {
        self.inner.node_by_id(id)
    }

    /// Returns the bond connecting the given pair of node ids, if present.
    #[inline]
    #[must_use]
    pub fn edge_for_node_pair(&self, nodes: (usize, usize)) -> Option<BondEdge> {
        self.inner.edge_for_node_pair(nodes)
    }

    /// Returns the number of bonds incident to the provided node id.
    #[inline]
    #[must_use]
    pub fn edge_count_for_node(&self, id: usize) -> usize {
        self.inner.edge_count_for_node(id)
    }

    /// Returns the connectivity count for the provided atom id.
    #[inline]
    #[must_use]
    pub fn connectivity_count(&self, id: usize) -> u8 {
        self.inner.connectivity_count(id)
    }

    /// Returns the total valence for the provided atom id.
    #[inline]
    #[must_use]
    pub fn total_valence(&self, id: usize) -> u8 {
        self.inner.total_valence(id)
    }

    /// Returns the RDKit-style SMARTS total valence for the provided atom id
    /// under the supplied aromaticity assignment.
    #[inline]
    #[must_use]
    pub fn smarts_total_valence(&self, id: usize, aromaticity: &AromaticityAssignment) -> u8 {
        self.inner.smarts_total_valence(id, aromaticity)
    }

    /// Returns a zero-allocation iterator over the bonds incident to the
    /// provided node id.
    #[inline]
    pub fn edges_for_node(&self, id: usize) -> impl Iterator<Item = BondEdge> + '_ {
        self.inner.edges_for_node(id)
    }

    /// Returns semantic tetrahedral or allene-like chirality for SMARTS-style
    /// matching.
    #[inline]
    #[must_use]
    pub fn smarts_tetrahedral_chirality(&self, atom_id: usize) -> Option<Chirality> {
        self.inner.smarts_tetrahedral_chirality(atom_id)
    }

    /// Returns semantic `E`/`Z` stereo for the requested double bond, if any.
    #[inline]
    #[must_use]
    pub fn double_bond_stereo_config(
        &self,
        node_a: usize,
        node_b: usize,
    ) -> Option<DoubleBondStereoConfig> {
        self.inner.double_bond_stereo_config(node_a, node_b)
    }

    /// Returns the atoms and bonds that belong to at least one ring.
    #[inline]
    #[must_use]
    pub fn ring_membership(&self) -> RingMembership {
        self.inner.ring_membership()
    }

    /// Returns atom-only ring membership for the graph.
    #[inline]
    #[must_use]
    pub fn ring_atom_membership(&self) -> RingAtomMembership {
        self.inner.ring_atom_membership()
    }

    /// Writes atom-only ring membership into caller-provided output and
    /// scratch buffers.
    #[inline]
    pub fn write_ring_atom_membership(
        &self,
        ring_atom_membership: &mut RingAtomMembership,
        scratch: &mut RingAtomMembershipScratch,
    ) {
        self.inner.write_ring_atom_membership(ring_atom_membership, scratch);
    }

    /// Returns the canonicalized symmetric-SSSR-style cycle set together with
    /// search completeness status.
    #[inline]
    #[must_use]
    pub fn symm_sssr_result(&self) -> SymmSssrResult {
        self.inner.symm_sssr_result()
    }

    /// Returns the symmetric valued sparse matrix storing the graph bonds.
    #[inline]
    #[must_use]
    pub fn bond_matrix(&self) -> &BondMatrix {
        self.inner.bond_matrix()
    }

    /// Returns the number of unique chemical bonds in the graph.
    #[inline]
    #[must_use]
    pub fn number_of_bonds(&self) -> usize {
        self.inner.number_of_bonds()
    }

    /// Returns the cached implicit hydrogen count for every atom.
    #[inline]
    #[must_use]
    pub fn implicit_hydrogen_counts(&self) -> &[u8] {
        self.inner.implicit_hydrogen_counts()
    }

    /// Returns the implicit hydrogen count for the provided atom id.
    #[inline]
    #[must_use]
    pub fn implicit_hydrogen_count(&self, id: usize) -> u8 {
        self.inner.implicit_hydrogen_count(id)
    }

    /// Returns the canonical labeling of the current graph.
    #[inline]
    #[must_use]
    pub fn canonical_labeling(&self) -> SmilesCanonicalLabeling {
        self.inner.canonical_labeling()
    }

    /// Returns whether the current graph is already in canonical form.
    #[inline]
    #[must_use]
    pub fn is_canonical(&self) -> bool {
        self.inner.is_canonical()
    }

    /// Returns the graph rewritten into canonical node order.
    #[inline]
    #[must_use]
    pub fn canonicalize(&self) -> Self {
        Self::from_inner(self.inner.canonicalize())
    }

    /// Returns a graph with directional single bonds collapsed to ordinary
    /// single bonds.
    #[inline]
    #[must_use]
    pub fn with_directional_bonds_collapsed(&self) -> Self {
        Self::from_inner(self.inner.with_directional_bonds_collapsed())
    }

    /// Returns a variant of the graph with all currently implied hydrogens
    /// materialized as terminal `[H]` atoms.
    #[inline]
    #[must_use]
    pub fn with_explicit_hydrogens(&self) -> Self {
        Self::from_inner(self.inner.with_explicit_hydrogens())
    }

    /// Renders the graph back into a SMILES string.
    #[inline]
    #[must_use]
    pub fn render(&self) -> String {
        self.inner.render()
    }

    /// Returns a localized Kekule form of the current graph.
    ///
    /// # Errors
    /// Returns a [`KekulizationError`] if no valid localized form exists.
    #[inline]
    pub fn kekulize(&self) -> Result<Self, KekulizationError> {
        self.inner.kekulize().map(Self::from_inner)
    }

    /// Returns a localized Kekule form of the current graph using the provided
    /// mode.
    ///
    /// # Errors
    /// Returns a [`KekulizationError`] if no valid localized form exists.
    #[inline]
    pub fn kekulize_with(&self, mode: KekulizationMode) -> Result<Self, KekulizationError> {
        self.inner.kekulize_with(mode).map(Self::from_inner)
    }

    /// Returns a localized Kekule form by solving from the current aromatic
    /// graph alone.
    ///
    /// # Errors
    /// Returns a [`KekulizationError`] if no valid localized form exists.
    #[inline]
    pub fn kekulize_standalone(&self) -> Result<Self, KekulizationError> {
        self.inner.kekulize_standalone().map(Self::from_inner)
    }

    /// Returns a new graph with the provided aromaticity assignment applied.
    ///
    /// # Errors
    /// Returns an [`AromaticityAssignmentApplicationError`] if the assignment
    /// is inconsistent with the graph.
    #[inline]
    pub fn try_with_aromaticity_assignment(
        &self,
        assignment: &AromaticityAssignment,
    ) -> Result<Self, AromaticityAssignmentApplicationError> {
        self.inner.try_with_aromaticity_assignment(assignment).map(Self::from_inner)
    }
}

impl Smiles<WildcardAtoms> {
    fn try_into_smiles(self) -> Result<Smiles, SmilesError> {
        if self.contains_wildcard_atom() {
            Err(SmilesError::WildcardAtomNotAllowed)
        } else {
            Ok(self.into_atom_policy())
        }
    }
}

impl TryFrom<WildcardSmiles> for Smiles {
    type Error = SmilesError;

    fn try_from(smiles: WildcardSmiles) -> Result<Self, Self::Error> {
        smiles.inner.try_into_smiles()
    }
}

impl From<Smiles> for WildcardSmiles {
    fn from(smiles: Smiles) -> Self {
        Self::from_inner(smiles.into_atom_policy())
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RingComponent {
    atom_ids: Vec<usize>,
    bond_edges: Vec<[usize; 2]>,
}

impl<LeftPolicy: SmilesAtomPolicy, RightPolicy: SmilesAtomPolicy> PartialEq<Smiles<RightPolicy>>
    for Smiles<LeftPolicy>
{
    fn eq(&self, other: &Smiles<RightPolicy>) -> bool {
        self.atom_nodes == other.atom_nodes && self.bond_matrix == other.bond_matrix
    }
}

impl PartialEq for WildcardSmiles {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl PartialEq<Smiles> for WildcardSmiles {
    fn eq(&self, other: &Smiles) -> bool {
        self.inner == *other
    }
}

impl PartialEq<WildcardSmiles> for Smiles {
    fn eq(&self, other: &WildcardSmiles) -> bool {
        *self == other.inner
    }
}

impl<AtomPolicy: SmilesAtomPolicy> fmt::Display for Smiles<AtomPolicy> {
    /// Formats the graph by running the full render pipeline and writing the
    /// resulting SMILES string into `f`.
    ///
    /// See [`Smiles::render`] for the full algorithm stack and the rationale
    /// behind the split between planning and emission.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

impl fmt::Display for WildcardSmiles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[cfg(test)]
mod tests {
    use alloc::{string::ToString, vec::Vec};

    use elements_rs::Element;

    use super::{
        AromaticityAssignment, AromaticityAssignmentApplicationError, AromaticityDiagnostic,
        AromaticityModel, AromaticityPolicy, AromaticityStatus, BondMatrixBuilder,
        DoubleBondStereoConfig, RdkitMdlAromaticity, RdkitSimpleAromaticity, RingAtomMembership,
        RingAtomMembershipScratch, RingMembership, Smiles, SmilesAtomPolicy, StereoNeighbor,
        WildcardSmiles,
    };
    use crate::{
        atom::{Atom, AtomSyntax, atom_symbol::AtomSymbol, bracketed::chirality::Chirality},
        bond::{
            Bond,
            bond_edge::{BondEdge, bond_edge},
            ring_num::RingNum,
        },
        errors::SmilesError,
    };

    const LARGE_POLYCYCLE_FRONTIER_CASE: &str = "CC1=C2CC3=C(C4=C5C=C3[C@@H]6C2=CC7=C1CC8=C(C9=C1C=C8[C@@H]7CCC[C@@H]2C3=C7CC8=C2C=C2C%10CCCC%11C%12=CC%13=C%14CC(=C7C)C(=C3)[C@@H]%13CCC[C@H]3C7=C%13CC%15=C3C=C3C(CCCC%16C%17=CC(=C(C4)C(=C%17CC4=C%16C=C%16[C@H](CCC6)C(=C7)C(=C%13C)CC%16=C4C)C)C5CCCC1C1=CC%10=C(CC2=C8C)C(=C1C9)C)C1=CC%11=C(CC%12=C%14C)C(=C1CC3=C%15C)C)C)C";

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

    fn assert_edge_descriptor<AtomPolicy: SmilesAtomPolicy>(
        smiles: &Smiles<AtomPolicy>,
        node_pair: (usize, usize),
        bond: Bond,
        aromatic: bool,
    ) {
        let edge = smiles.edge_for_node_pair(node_pair).expect("expected edge");
        assert_eq!(edge.bond(), bond);
        assert_eq!(edge.is_aromatic(), aromatic);
    }

    fn hydrogen_neighbors(smiles: &Smiles, atom_id: usize) -> Vec<usize> {
        smiles
            .edges_for_node(atom_id)
            .filter_map(|edge| {
                let neighbor = if edge.source() == atom_id { edge.target() } else { edge.source() };
                (smiles.nodes()[neighbor].element() == Some(Element::H)).then_some(neighbor)
            })
            .collect()
    }

    #[test]
    fn bond_matrix_builder_rejects_self_loops() {
        let mut builder = BondMatrixBuilder::with_capacity(1);
        let err = builder
            .push_edge_with_descriptor(0, 0, Bond::Single.into(), None)
            .expect_err("self-loop should fail");
        assert_eq!(err, SmilesError::SelfLoopEdge(0));
    }

    #[test]
    fn edge_key_normalizes_node_order() {
        assert_eq!(crate::smiles::edge_key(1, 4), (1, 4));
        assert_eq!(crate::smiles::edge_key(4, 1), (1, 4));
        assert_eq!(crate::smiles::edge_key(3, 3), (3, 3));
    }

    #[test]
    fn edge_lookup_helpers_work() {
        let ring = RingNum::try_new(1).unwrap();
        let smiles = smiles_from_edges(
            vec![atom(Element::C), atom(Element::O), atom(Element::N)],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(1, 2, Bond::Double, Some(ring))],
        );

        assert_eq!(smiles.number_of_bonds(), 2);
        assert_eq!(smiles.edge_for_node_pair((0, 1)), Some(bond_edge(0, 1, Bond::Single, None)));
        assert_eq!(smiles.edge_for_node_pair((1, 0)), Some(bond_edge(0, 1, Bond::Single, None)));
        assert_eq!(
            smiles.edge_for_node_pair((1, 2)),
            Some(bond_edge(1, 2, Bond::Double, Some(ring)))
        );
        assert_eq!(smiles.edge_for_node_pair((0, 2)), None);

        let edges_for_1 = smiles.edges_for_node(1).collect::<Vec<_>>();
        assert_eq!(edges_for_1.len(), 2);
        assert!(edges_for_1.contains(&bond_edge(1, 0, Bond::Single, None)));
        assert!(edges_for_1.contains(&bond_edge(1, 2, Bond::Double, Some(ring))));

        let edges_for_0 = smiles.edges_for_node(0).collect::<Vec<_>>();
        assert_eq!(edges_for_0, vec![bond_edge(0, 1, Bond::Single, None)]);
    }

    #[test]
    #[should_panic(expected = "invalid atom index 99 for graph with 3 atoms")]
    fn edges_for_node_panics_for_invalid_atom_id() {
        let ring = RingNum::try_new(7).expect("valid ring number");
        let smiles = smiles_from_edges(
            vec![atom(Element::C), atom(Element::O), atom(Element::N)],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(1, 2, Bond::Double, Some(ring))],
        );

        let _ = smiles.edges_for_node(99).collect::<Vec<_>>();
    }

    #[test]
    #[should_panic(expected = "invalid atom index 99 for graph with 3 atoms")]
    fn edge_count_for_node_panics_for_invalid_atom_id() {
        let ring = RingNum::try_new(7).expect("valid ring number");
        let smiles = smiles_from_edges(
            vec![atom(Element::C), atom(Element::O), atom(Element::N)],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(1, 2, Bond::Double, Some(ring))],
        );

        let _ = smiles.edge_count_for_node(99);
    }

    #[test]
    fn connectivity_count_and_total_valence_follow_rdkit_style_local_counts() {
        let alcohol: Smiles = "CO".parse().expect("valid SMILES");
        assert_eq!(alcohol.connectivity_count(0), 4);
        assert_eq!(alcohol.total_valence(0), 4);
        assert_eq!(alcohol.connectivity_count(1), 2);
        assert_eq!(alcohol.total_valence(1), 2);

        let alkene: Smiles = "C=C".parse().expect("valid SMILES");
        assert_eq!(alkene.connectivity_count(0), 3);
        assert_eq!(alkene.total_valence(0), 4);
        assert_eq!(alkene.connectivity_count(1), 3);
        assert_eq!(alkene.total_valence(1), 4);

        let bracketed: Smiles = "[CH3]".parse().expect("valid SMILES");
        assert_eq!(bracketed.connectivity_count(0), 3);
        assert_eq!(bracketed.total_valence(0), 3);
    }

    #[test]
    #[should_panic(expected = "invalid atom index 99 for graph with 2 atoms")]
    fn connectivity_count_panics_for_invalid_atom_id() {
        let smiles: Smiles = "CO".parse().expect("valid SMILES");
        let _ = smiles.connectivity_count(99);
    }

    #[test]
    #[should_panic(expected = "invalid atom index 99 for graph with 2 atoms")]
    fn total_valence_panics_for_invalid_atom_id() {
        let smiles: Smiles = "CO".parse().expect("valid SMILES");
        let _ = smiles.total_valence(99);
    }

    #[test]
    fn smarts_total_valence_preserves_raw_non_aromatic_values() {
        let sodium: Smiles = "[Na+]".parse().expect("valid sodium cation");
        let sodium_aromaticity = sodium.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(sodium.smarts_total_valence(0, &sodium_aromaticity), 0);

        let ammonium: Smiles = "[NH4+]".parse().expect("valid ammonium");
        let ammonium_aromaticity =
            ammonium.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(ammonium.smarts_total_valence(0, &ammonium_aromaticity), 4);

        let phosphoric_acid: Smiles = "P(=O)(O)(O)O".parse().expect("valid phosphoric acid");
        let phosphorus_aromaticity =
            phosphoric_acid.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(phosphoric_acid.smarts_total_valence(0, &phosphorus_aromaticity), 5);
    }

    #[test]
    fn smarts_total_valence_matches_rdkit_aromatic_expectations() {
        let benzene: Smiles = "c1ccccc1".parse().expect("valid aromatic benzene");
        let benzene_aromaticity =
            benzene.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(benzene.total_valence(0), 3);
        assert_eq!(benzene.smarts_total_valence(0, &benzene_aromaticity), 4);

        let pyridine: Smiles = "c1ncccc1".parse().expect("valid aromatic pyridine");
        let pyridine_aromaticity =
            pyridine.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(pyridine.total_valence(1), 2);
        assert_eq!(pyridine.smarts_total_valence(1, &pyridine_aromaticity), 3);

        let pyrrolic_nitrogen: Smiles = "[nH]1cccc1".parse().expect("valid aromatic pyrrole");
        let pyrrolic_aromaticity =
            pyrrolic_nitrogen.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(pyrrolic_nitrogen.total_valence(0), 3);
        assert_eq!(pyrrolic_nitrogen.smarts_total_valence(0, &pyrrolic_aromaticity), 3);
    }

    #[test]
    #[should_panic(expected = "invalid atom index 99 for graph with 1 atoms")]
    fn smarts_total_valence_panics_for_invalid_atom_id() {
        let smiles: Smiles = "C".parse().expect("valid SMILES");
        let aromaticity = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        let _ = smiles.smarts_total_valence(99, &aromaticity);
    }

    #[test]
    fn with_explicit_hydrogens_materializes_implicit_hydrogens_as_terminal_hydrogen_atoms() {
        let smiles: Smiles = "CO".parse().expect("valid SMILES");
        let explicit = smiles.with_explicit_hydrogens();

        assert_eq!(explicit.nodes().len(), 6);
        assert_eq!(explicit.number_of_bonds(), 5);
        assert_eq!(explicit.implicit_hydrogen_counts(), &[0, 0, 0, 0, 0, 0]);
        assert_eq!(explicit.nodes()[0].hydrogen_count(), 0);
        assert_eq!(explicit.nodes()[1].hydrogen_count(), 0);
        assert_eq!(explicit.edge_count_for_node(0), 4);
        assert_eq!(explicit.edge_count_for_node(1), 2);
        assert_eq!(
            explicit.nodes().iter().filter(|atom| atom.element() == Some(Element::H)).count(),
            4
        );
    }

    #[test]
    fn with_explicit_hydrogens_rewrites_bracket_explicit_hydrogens_into_terminal_hydrogen_atoms() {
        let ammonium: Smiles = "[NH4+]".parse().expect("valid ammonium");
        let explicit = ammonium.with_explicit_hydrogens();

        assert_eq!(explicit.nodes().len(), 5);
        assert_eq!(explicit.number_of_bonds(), 4);
        assert_eq!(explicit.nodes()[0].charge_value(), 1);
        assert_eq!(explicit.nodes()[0].hydrogen_count(), 0);
        assert_eq!(explicit.edge_count_for_node(0), 4);
        assert_eq!(explicit.implicit_hydrogen_counts(), &[0, 0, 0, 0, 0]);
        assert!(explicit.nodes()[1..].iter().all(|atom| atom.element() == Some(Element::H)));
    }

    #[test]
    fn with_explicit_hydrogens_preserves_semantic_tetrahedral_chirality() {
        let smiles: Smiles = "F[C@H](Cl)Br".parse().expect("valid SMILES");
        let before = smiles.smarts_tetrahedral_chirality(1);
        let explicit = smiles.with_explicit_hydrogens();

        assert_eq!(explicit.nodes().len(), 5);
        assert_eq!(explicit.edge_for_node_pair((1, 4)).unwrap().bond(), Bond::Single);
        assert_eq!(explicit.smarts_tetrahedral_chirality(1), before);
    }

    #[test]
    fn with_explicit_hydrogens_rewrites_parsed_stereo_neighbors_for_chiral_h_centers() {
        let smiles: Smiles = "F[C@H](Cl)Br".parse().expect("valid SMILES");
        let explicit = smiles.with_explicit_hydrogens();

        assert_eq!(
            explicit.parsed_stereo_neighbors_row(1),
            &[
                StereoNeighbor::Atom(0),
                StereoNeighbor::Atom(4),
                StereoNeighbor::Atom(2),
                StereoNeighbor::Atom(3),
            ]
        );
    }

    #[test]
    fn with_explicit_hydrogens_is_noop_when_no_hydrogens_are_present() {
        let smiles: Smiles = "[Na+]".parse().expect("valid sodium cation");
        assert_eq!(smiles.with_explicit_hydrogens(), smiles);
    }

    #[test]
    fn with_explicit_hydrogens_is_idempotent_after_materialization() {
        let once = "CO".parse::<Smiles>().expect("valid SMILES").with_explicit_hydrogens();
        let twice = once.with_explicit_hydrogens();
        assert_eq!(twice, once);
    }

    #[test]
    fn with_explicit_hydrogens_produces_terminal_bracket_hydrogen_atoms() {
        let smiles: Smiles = "CO".parse().expect("valid SMILES");
        let explicit = smiles.with_explicit_hydrogens();

        for atom_id in 2..explicit.nodes().len() {
            let atom = explicit.nodes()[atom_id];
            assert_eq!(atom.element(), Some(Element::H));
            assert_eq!(atom.syntax(), AtomSyntax::Bracket);
            assert_eq!(atom.hydrogen_count(), 0);
            assert_eq!(atom.charge_value(), 0);
            assert_eq!(atom.class(), 0);
            assert_eq!(atom.chirality(), None);
            assert_eq!(explicit.edge_count_for_node(atom_id), 1);
            assert_eq!(explicit.implicit_hydrogen_count(atom_id), 0);
        }
    }

    #[test]
    fn with_explicit_hydrogens_preserves_ring_membership_on_the_heavy_atom_subgraph() {
        let smiles: Smiles = "C1CCCCC1".parse().expect("valid cyclohexane");
        let explicit = smiles.with_explicit_hydrogens();

        assert_eq!(explicit.nodes().len(), 18);
        assert_eq!(explicit.number_of_bonds(), 18);
        assert_eq!(explicit.ring_membership().atom_ids(), &[0, 1, 2, 3, 4, 5]);
        for atom_id in 0..6 {
            assert_eq!(explicit.nodes()[atom_id].element(), Some(Element::C));
            assert_eq!(explicit.edge_count_for_node(atom_id), 4);
            assert_eq!(hydrogen_neighbors(&explicit, atom_id).len(), 2);
        }
        for atom_id in 6..18 {
            assert!(!explicit.ring_membership().contains_atom(atom_id));
        }
    }

    #[test]
    fn with_explicit_hydrogens_preserves_aromatic_heavy_atoms() {
        let smiles: Smiles = "c1ccccc1".parse().expect("valid benzene");
        let explicit = smiles.with_explicit_hydrogens();

        assert_eq!(explicit.nodes().len(), 12);
        assert_eq!(explicit.number_of_bonds(), 12);
        assert_eq!(explicit.ring_membership().atom_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(explicit.implicit_hydrogen_counts(), &[0; 12]);
        for atom_id in 0..6 {
            assert!(explicit.nodes()[atom_id].aromatic());
            assert_eq!(explicit.nodes()[atom_id].hydrogen_count(), 0);
            assert_eq!(hydrogen_neighbors(&explicit, atom_id).len(), 1);
        }
        for [node_a, node_b] in explicit.aromaticity_assignment().bond_edges() {
            let edge = explicit.edge_for_node_pair((*node_a, *node_b)).unwrap();
            assert!(edge.is_aromatic());
        }
    }

    #[test]
    fn with_explicit_hydrogens_preserves_bracket_atom_metadata_while_clearing_h_count() {
        let smiles: Smiles = "[13CH-:7]".parse().expect("valid bracket atom");
        let explicit = smiles.with_explicit_hydrogens();
        let atom = explicit.nodes()[0];

        assert_eq!(explicit.nodes().len(), 2);
        assert_eq!(explicit.number_of_bonds(), 1);
        assert_eq!(atom.element(), Some(Element::C));
        assert_eq!(atom.isotope_mass_number(), Some(13));
        assert_eq!(atom.charge_value(), -1);
        assert_eq!(atom.class(), 7);
        assert_eq!(atom.syntax(), AtomSyntax::Bracket);
        assert_eq!(atom.hydrogen_count(), 0);
        assert_eq!(explicit.nodes()[1].element(), Some(Element::H));
        assert_eq!(explicit.edge_count_for_node(0), 1);
    }

    #[test]
    fn smarts_tetrahedral_chirality_exposes_semantic_tetrahedral_forms() {
        let left: Smiles = "F[C@H](Cl)Br".parse().expect("valid SMILES");
        let right: Smiles = "F[C@@H](Cl)Br".parse().expect("valid SMILES");
        let achiral: Smiles = "CC(O)N".parse().expect("valid SMILES");

        assert!(matches!(left.smarts_tetrahedral_chirality(1), Some(Chirality::TH(_))));
        assert!(matches!(right.smarts_tetrahedral_chirality(1), Some(Chirality::TH(_))));
        assert_ne!(left.smarts_tetrahedral_chirality(1), right.smarts_tetrahedral_chirality(1));
        assert_eq!(achiral.smarts_tetrahedral_chirality(1), None);
    }

    #[test]
    fn smarts_tetrahedral_chirality_exposes_allene_like_forms() {
        let left: Smiles = "OC(Cl)=[C@]=C(C)F".parse().expect("valid SMILES");
        let right: Smiles = "OC(Cl)=[C@@]=C(C)F".parse().expect("valid SMILES");

        assert!(matches!(left.smarts_tetrahedral_chirality(3), Some(Chirality::AL(_))));
        assert!(matches!(right.smarts_tetrahedral_chirality(3), Some(Chirality::AL(_))));
        assert_ne!(left.smarts_tetrahedral_chirality(3), right.smarts_tetrahedral_chirality(3));
    }

    #[test]
    #[should_panic(expected = "invalid atom index 99 for graph with 4 atoms")]
    fn smarts_tetrahedral_chirality_panics_for_invalid_atom_id() {
        let smiles: Smiles = "F[C@H](Cl)Br".parse().expect("valid SMILES");
        let _ = smiles.smarts_tetrahedral_chirality(99);
    }

    #[test]
    fn double_bond_stereo_config_reports_semantic_e_z_and_absence() {
        let trans: Smiles = "F/C=C/F".parse().expect("valid SMILES");
        let cis: Smiles = "F/C=C\\F".parse().expect("valid SMILES");
        let plain: Smiles = "CC=CC".parse().expect("valid SMILES");
        let ring: Smiles = "C1CC/C=C/CC1".parse().expect("valid SMILES");

        assert_eq!(trans.double_bond_stereo_config(1, 2), Some(DoubleBondStereoConfig::E));
        assert_eq!(cis.double_bond_stereo_config(1, 2), Some(DoubleBondStereoConfig::Z));
        assert_eq!(plain.double_bond_stereo_config(1, 2), None);
        assert_eq!(ring.double_bond_stereo_config(3, 4), None);
        assert_eq!(plain.double_bond_stereo_config(9, 10), None);
    }

    #[test]
    fn render_and_display_work_for_simple_valid_graph() {
        let smiles = smiles_from_edges(
            vec![atom(Element::C), atom(Element::O)],
            &[bond_edge(0, 1, Bond::Double, None)],
        );

        let rendered = smiles.render();
        assert_eq!(rendered, "C=O");
        assert_eq!(format!("{smiles}"), "C=O");
    }

    #[test]
    fn directional_bond_collapse_rewrites_graph_but_not_atoms() {
        let raw: Smiles = "C/C=C\\C".parse().unwrap();
        let collapsed = raw.with_directional_bonds_collapsed();

        assert_eq!(raw.nodes(), collapsed.nodes());
        assert_eq!(raw.number_of_bonds(), collapsed.number_of_bonds());
        assert_eq!(raw.edge_for_node_pair((0, 1)).unwrap().bond(), Bond::Up);
        assert_eq!(raw.edge_for_node_pair((2, 3)).unwrap().bond(), Bond::Down);
        assert_eq!(collapsed.edge_for_node_pair((0, 1)).unwrap().bond(), Bond::Single);
        assert_eq!(collapsed.edge_for_node_pair((1, 2)).unwrap().bond(), Bond::Double);
        assert_eq!(collapsed.edge_for_node_pair((2, 3)).unwrap().bond(), Bond::Single);
    }

    #[test]
    fn ring_membership_is_empty_for_acyclic_graphs() {
        let smiles: Smiles = "CCO".parse().unwrap();
        let ring_membership = smiles.ring_membership();

        assert_eq!(
            ring_membership,
            RingMembership { atom_ids: Vec::new(), bond_edges: Vec::new() }
        );
        assert!(!ring_membership.contains_atom(0));
        assert!(!ring_membership.contains_edge(0, 1));
    }

    #[test]
    fn ring_atom_membership_is_empty_for_acyclic_graphs() {
        let smiles: Smiles = "CCO".parse().unwrap();
        let ring_atom_membership = smiles.ring_atom_membership();

        assert_eq!(
            ring_atom_membership,
            RingAtomMembership { atom_flags: vec![false, false, false] }
        );
        assert!(!ring_atom_membership.contains_atom(0));
        assert!(!ring_atom_membership.contains_atom(1));
        assert!(!ring_atom_membership.contains_atom(2));
    }

    #[test]
    fn ring_atom_membership_captures_only_ring_atoms() {
        let smiles: Smiles = "CC1CCCCC1".parse().unwrap();
        let ring_atom_membership = smiles.ring_atom_membership();

        assert_eq!(ring_atom_membership.atom_flags(), &[false, true, true, true, true, true, true]);
        assert!(!ring_atom_membership.contains_atom(0));
        assert!(ring_atom_membership.contains_atom(4));
    }

    #[test]
    #[should_panic(expected = "invalid atom index 99")]
    fn ring_atom_membership_panics_for_invalid_atom_id() {
        let smiles: Smiles = "C1CCCCC1".parse().unwrap();
        let ring_atom_membership = smiles.ring_atom_membership();

        let _ = ring_atom_membership.contains_atom(99);
    }

    #[test]
    fn ring_atom_membership_matches_ring_membership_on_representative_cases() {
        let cases = [
            "CCO",
            "CC1CCCCC1",
            "C1CCCCC1CC2CCCCC2",
            "c1cccc2ccccc12",
            "C12C3C4C1C5C2C3C45",
            "CC1=C2CC3=C(C4=C5C=C3[C@@H]6C2=CC7=C1CC8=C(C9=C1C=C8[C@@H]7CCC[C@@H]2C3=C7CC8=C2C=C2C%10CCCC%11C%12=CC%13=C%14CC(=C7C)C(=C3)[C@@H]%13CCC[C@H]3C7=C%13CC%15=C3C=C3C(CCCC%16C%17=CC(=C(C4)C(=C%17CC4=C%16C=C%16[C@H](CCC6)C(=C7)C(=C%13C)CC%16=C4C)C)C5CCCC1C1=CC%10=C(CC2=C8C)C(=C1C9)C)C1=CC%11=C(CC%12=C%14C)C(=C1CC3=C%15C)C)C)C",
        ];

        for source in cases {
            let smiles: Smiles = source.parse().unwrap();
            let ring_atom_membership = smiles.ring_atom_membership();
            let ring_membership = smiles.ring_membership();
            let mut expected_flags = vec![false; smiles.nodes().len()];
            for &atom_id in ring_membership.atom_ids() {
                expected_flags[atom_id] = true;
            }
            assert_eq!(ring_atom_membership.atom_flags(), expected_flags, "{source}");
        }
    }

    #[test]
    fn write_ring_atom_membership_matches_one_shot_api_across_reused_buffers() {
        let cases = ["CCO", "CC1CCCCC1", "c1cccc2ccccc12", "C12C3C4C1C5C2C3C45"];
        let mut ring_atom_membership = RingAtomMembership::default();
        let mut scratch = RingAtomMembershipScratch::default();

        for source in cases {
            let smiles: Smiles = source.parse().unwrap();
            smiles.write_ring_atom_membership(&mut ring_atom_membership, &mut scratch);
            assert_eq!(ring_atom_membership, smiles.ring_atom_membership(), "{source}");
        }
    }

    #[test]
    fn ring_membership_captures_ring_atoms_and_bonds() {
        let smiles: Smiles = "C1CCCCC1".parse().unwrap();
        let ring_membership = smiles.ring_membership();

        assert_eq!(ring_membership.atom_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(ring_membership.bond_edges(), &[[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]);
        assert!(ring_membership.contains_atom(3));
        assert!(ring_membership.contains_edge(5, 0));
        assert!(!ring_membership.contains_edge(0, 2));
    }

    #[test]
    fn symm_sssr_returns_two_fused_rings_for_naphthalene() {
        let smiles: Smiles = "c1cccc2ccccc12".parse().unwrap();

        assert_eq!(
            smiles.symm_sssr_result().cycles(),
            &[vec![0, 1, 2, 3, 4, 9], vec![4, 5, 6, 7, 8, 9]]
        );
    }

    #[test]
    fn symm_sssr_returns_cubane_faces() {
        let smiles: Smiles = "C12C3C4C1C5C2C3C45".parse().unwrap();

        assert_eq!(
            smiles.symm_sssr_result().cycles(),
            &[
                vec![0, 1, 2, 3],
                vec![0, 1, 6, 5],
                vec![0, 3, 4, 5],
                vec![1, 2, 7, 6],
                vec![2, 3, 4, 7],
                vec![4, 5, 6, 7],
            ]
        );
    }

    #[test]
    fn symm_sssr_result_reports_complete_benzene_search() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();
        let result = smiles.symm_sssr_result();

        assert!(result.status().is_complete());
        assert!(!result.status().used_fallback());
        assert!(!result.status().hit_queue_cutoff());
        assert_eq!(result.cycles(), &[vec![0, 1, 2, 3, 4, 5]]);
        assert_eq!(result.cycles(), &[vec![0, 1, 2, 3, 4, 5]]);
    }

    #[test]
    fn symm_sssr_result_exposes_fallback_for_large_polycycle_frontier_case() {
        let smiles: Smiles = LARGE_POLYCYCLE_FRONTIER_CASE.parse().unwrap();
        let result = smiles.symm_sssr_result();

        assert!(!result.status().is_complete());
        assert!(result.status().used_fallback());
        assert!(!result.status().hit_queue_cutoff());
        assert_eq!(result.cycles(), smiles.symm_sssr_result().cycles());
    }

    #[test]
    fn aromaticity_assignment_perceives_kekule_benzene() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();
        let perception = smiles.perceive_aromaticity().unwrap();
        let assignment = perception.assignment();
        let aromaticized = perception.aromaticized();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(assignment.bond_edges(), &[[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]);
        assert!(aromaticized.nodes().iter().all(Atom::aromatic));
        assert_edge_descriptor(aromaticized, (0, 1), Bond::Double, true);
        assert_edge_descriptor(aromaticized, (4, 5), Bond::Double, true);
    }

    #[test]
    fn aromaticity_assignment_can_be_provided_by_a_custom_model() {
        struct SingleBondModel;

        impl AromaticityModel for SingleBondModel {
            fn assignment<AtomPolicy: SmilesAtomPolicy>(
                &self,
                _smiles: &Smiles<AtomPolicy>,
            ) -> AromaticityAssignment {
                AromaticityAssignment::new(AromaticityStatus::Complete, vec![0, 1], vec![[0, 1]])
            }
        }

        let smiles: Smiles = "CC".parse().unwrap();
        let perception = smiles.perceive_aromaticity_with(&SingleBondModel).unwrap();
        let assignment = perception.assignment();
        let aromaticized = perception.aromaticized();

        assert_eq!(assignment.atom_ids(), &[0, 1]);
        assert_eq!(assignment.bond_edges(), &[[0, 1]]);
        assert!(aromaticized.nodes().iter().all(Atom::aromatic));
        assert_edge_descriptor(aromaticized, (0, 1), Bond::Single, true);
    }

    #[test]
    fn aromaticity_policy_helpers_match_default_model() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();

        assert_eq!(
            smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault),
            smiles.aromaticity_assignment()
        );

        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault).unwrap();
        assert_eq!(perception.status(), AromaticityStatus::Complete);
        assert_eq!(
            perception.assignment(),
            &smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault)
        );
        assert!(perception.aromaticized().nodes().iter().all(Atom::aromatic));

        let owned_assignment = smiles
            .perceive_aromaticity_for(AromaticityPolicy::RdkitDefault)
            .unwrap()
            .into_assignment();
        assert_eq!(owned_assignment, smiles.aromaticity_assignment());

        let owned_aromaticized = smiles
            .perceive_aromaticity_for(AromaticityPolicy::RdkitDefault)
            .unwrap()
            .into_aromaticized();
        assert!(owned_aromaticized.nodes().iter().all(Atom::aromatic));
    }

    #[test]
    fn aromaticity_policy_helpers_match_mdl_model() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();

        assert_eq!(
            smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl),
            smiles.aromaticity_assignment_with(&RdkitMdlAromaticity)
        );

        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitMdl).unwrap();
        assert_eq!(perception.status(), AromaticityStatus::Complete);
        assert_eq!(
            perception.assignment(),
            &smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl)
        );
        assert!(perception.aromaticized().nodes().iter().all(Atom::aromatic));
    }

    #[test]
    fn aromaticity_policy_helpers_match_simple_model() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();

        assert_eq!(
            smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple),
            smiles.aromaticity_assignment_with(&RdkitSimpleAromaticity)
        );

        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitSimple).unwrap();
        assert_eq!(perception.status(), AromaticityStatus::Complete);
        assert_eq!(
            perception.assignment(),
            &smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple)
        );
        assert!(perception.aromaticized().nodes().iter().all(Atom::aromatic));
    }

    #[test]
    fn mdl_aromaticity_keeps_kekule_benzene_aromatic() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(assignment.bond_edges(), &[[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]);
    }

    #[test]
    fn mdl_aromaticity_rejects_standalone_imidazole() {
        let smiles: Smiles = "C1=CN=CN1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty());
        assert!(assignment.bond_edges().is_empty());
    }

    #[test]
    fn mdl_aromaticity_keeps_only_benzenoid_ring_in_coumarin() {
        let smiles: Smiles = "C1=CC=C2C(=C1)C=CC(=O)O2".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(assignment.bond_edges(), &[[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]);
    }

    #[test]
    fn mdl_aromaticity_rejects_standalone_tropylium_like_ring() {
        let smiles: Smiles = "C1=CC=C[CH+]C=C1.F[P-](F)(F)(F)(F)F".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty(), "{assignment:?}");
        assert!(assignment.bond_edges().is_empty(), "{assignment:?}");
    }

    #[test]
    fn mdl_aromaticity_rejects_non_benzenoid_single_ring_subsystem_in_fused_cation() {
        let smiles: Smiles = "C1CCC2(CC1)C=CC3=C[C+]4C=CC5(C4=CC=C23)CCCCC5".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty(), "{assignment:?}");
        assert!(assignment.bond_edges().is_empty(), "{assignment:?}");
    }

    #[test]
    fn mdl_aromaticity_keeps_large_annulene_aromatic() {
        let smiles: Smiles = "C1=CC=CC=CC=CC=CC=CC=CC=CC=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitMdl);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(
            assignment.atom_ids(),
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        );
        assert_eq!(
            assignment.bond_edges(),
            &[
                [0, 1],
                [0, 17],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [12, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [16, 17],
            ]
        );
    }

    #[test]
    fn simple_aromaticity_keeps_kekule_benzene_aromatic() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(assignment.bond_edges(), &[[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]);
    }

    #[test]
    fn simple_aromaticity_keeps_standalone_imidazole_aromatic() {
        let smiles: Smiles = "C1=CN=CN1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4]);
        assert_eq!(assignment.bond_edges(), &[[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]);
    }

    #[test]
    fn simple_aromaticity_keeps_naphthalene_aromatic() {
        let smiles: Smiles = "C1=CC2=CC=CC=C2C=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            assignment.bond_edges(),
            &[
                [0, 1],
                [0, 9],
                [1, 2],
                [2, 3],
                [2, 7],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9]
            ]
        );
    }

    #[test]
    fn simple_aromaticity_rejects_tropylium() {
        let smiles: Smiles = "[CH+]1C=CC=CC=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty(), "{assignment:?}");
        assert!(assignment.bond_edges().is_empty(), "{assignment:?}");
    }

    #[test]
    fn simple_aromaticity_rejects_azulene() {
        let smiles: Smiles = "C1=CC=C2C=CC=C2C=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty(), "{assignment:?}");
        assert!(assignment.bond_edges().is_empty(), "{assignment:?}");
    }

    #[test]
    fn simple_aromaticity_rejects_large_annulene() {
        let smiles: Smiles = "C1=CC=CC=CC=CC=CC=CC=CC=CC=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitSimple);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty(), "{assignment:?}");
        assert!(assignment.bond_edges().is_empty(), "{assignment:?}");
    }

    #[test]
    fn try_with_aromaticity_assignment_accepts_valid_partial_assignment() {
        let smiles: Smiles = "CCC".parse().unwrap();
        let assignment =
            AromaticityAssignment::new(AromaticityStatus::Complete, vec![0, 1], vec![[0, 1]]);

        let aromaticized = smiles.try_with_aromaticity_assignment(&assignment).unwrap();

        assert!(aromaticized.nodes()[0].aromatic());
        assert!(aromaticized.nodes()[1].aromatic());
        assert!(!aromaticized.nodes()[2].aromatic());
        assert_edge_descriptor(&aromaticized, (0, 1), Bond::Single, true);
        assert_eq!(aromaticized.edge_for_node_pair((1, 2)).unwrap().bond(), Bond::Single);
    }

    #[test]
    fn try_with_aromaticity_assignment_rejects_bonds_without_aromatic_endpoints() {
        let smiles: Smiles = "CC".parse().unwrap();
        let assignment =
            AromaticityAssignment::new(AromaticityStatus::Complete, Vec::new(), vec![[0, 1]]);

        let error = smiles.try_with_aromaticity_assignment(&assignment).unwrap_err();

        assert_eq!(
            error,
            AromaticityAssignmentApplicationError::AromaticBondMissingEndpointAtoms {
                node_a: 0,
                node_b: 1,
            }
        );
    }

    #[test]
    fn try_with_aromaticity_assignment_rejects_out_of_bounds_bond_edges() {
        let smiles: Smiles = "CC".parse().unwrap();
        let assignment =
            AromaticityAssignment::new(AromaticityStatus::Complete, vec![0, 1], vec![[0, 2]]);

        let error = smiles.try_with_aromaticity_assignment(&assignment).unwrap_err();

        assert_eq!(
            error,
            AromaticityAssignmentApplicationError::BondEdgeAtomOutOfBounds {
                node_a: 0,
                node_b: 2,
                atom_count: 2,
            }
        );
    }

    #[test]
    fn try_with_aromaticity_assignment_rejects_nonexistent_bond_edges() {
        let smiles: Smiles = "CCC".parse().unwrap();
        let assignment =
            AromaticityAssignment::new(AromaticityStatus::Complete, vec![0, 2], vec![[0, 2]]);

        let error = smiles.try_with_aromaticity_assignment(&assignment).unwrap_err();

        assert_eq!(
            error,
            AromaticityAssignmentApplicationError::MissingBondEdge { node_a: 0, node_b: 2 }
        );
    }

    #[test]
    fn try_with_aromaticity_assignment_rejects_clearing_existing_aromatic_labels() {
        let smiles: Smiles = "c1ccccc1".parse().unwrap();
        let assignment =
            AromaticityAssignment::new(AromaticityStatus::Complete, Vec::new(), Vec::new());

        let error = smiles.try_with_aromaticity_assignment(&assignment).unwrap_err();

        assert_eq!(
            error,
            AromaticityAssignmentApplicationError::WouldClearAromaticAtom { atom_id: 0 }
        );
    }

    #[test]
    fn try_with_aromaticity_assignment_rejects_out_of_bounds_atom_ids() {
        let smiles: Smiles = "CC".parse().unwrap();
        let assignment =
            AromaticityAssignment::new(AromaticityStatus::Complete, vec![2], Vec::new());

        let error = smiles.try_with_aromaticity_assignment(&assignment).unwrap_err();

        assert_eq!(
            error,
            AromaticityAssignmentApplicationError::AtomIdOutOfBounds { atom_id: 2, atom_count: 2 }
        );
    }

    #[test]
    fn try_with_aromaticity_assignment_rejects_clearing_existing_aromatic_bonds() {
        let smiles: Smiles = "c1ccccc1".parse().unwrap();
        let assignment = AromaticityAssignment::new(
            AromaticityStatus::Complete,
            vec![0, 1, 2, 3, 4, 5],
            Vec::new(),
        );

        let error = smiles.try_with_aromaticity_assignment(&assignment).unwrap_err();

        assert_eq!(
            error,
            AromaticityAssignmentApplicationError::WouldClearAromaticBond { node_a: 0, node_b: 1 }
        );
    }

    #[test]
    fn try_with_aromaticity_assignment_preserves_non_aromatic_edges_not_in_assignment() {
        let smiles: Smiles = "CCC".parse().unwrap();
        let assignment =
            AromaticityAssignment::new(AromaticityStatus::Complete, vec![0, 1, 2], vec![[0, 1]]);

        let aromaticized = smiles.try_with_aromaticity_assignment(&assignment).unwrap();

        assert_edge_descriptor(&aromaticized, (0, 1), Bond::Single, true);
        assert_eq!(aromaticized.edge_for_node_pair((1, 2)).unwrap().bond(), Bond::Single);
    }

    #[test]
    fn aromaticity_assignment_preserves_custom_partial_status() {
        struct PartialModel;

        impl AromaticityModel for PartialModel {
            fn assignment<AtomPolicy: SmilesAtomPolicy>(
                &self,
                _smiles: &Smiles<AtomPolicy>,
            ) -> AromaticityAssignment {
                AromaticityAssignment::new(AromaticityStatus::Partial, vec![0, 1], vec![[0, 1]])
            }
        }

        let smiles: Smiles = "CC".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_with(&PartialModel);

        assert_eq!(assignment.status(), AromaticityStatus::Partial);
        assert_eq!(assignment.atom_ids(), &[0, 1]);
        assert_eq!(assignment.bond_edges(), &[[0, 1]]);
    }

    #[test]
    fn perceive_aromaticity_preserves_custom_partial_status() {
        struct PartialModel;

        impl AromaticityModel for PartialModel {
            fn assignment<AtomPolicy: SmilesAtomPolicy>(
                &self,
                _smiles: &Smiles<AtomPolicy>,
            ) -> AromaticityAssignment {
                AromaticityAssignment::new(AromaticityStatus::Partial, vec![0, 1], vec![[0, 1]])
            }
        }

        let smiles: Smiles = "CC".parse().unwrap();
        let perception = smiles.perceive_aromaticity_with(&PartialModel).unwrap();

        assert_eq!(perception.status(), AromaticityStatus::Partial);
        assert!(perception.diagnostics().is_empty());
        assert_eq!(perception.assignment().atom_ids(), &[0, 1]);
        assert_eq!(perception.assignment().bond_edges(), &[[0, 1]]);
        assert!(perception.aromaticized().nodes().iter().all(Atom::aromatic));
        assert_edge_descriptor(perception.aromaticized(), (0, 1), Bond::Single, true);
    }

    #[test]
    fn perceive_aromaticity_for_rdkit_default_returns_consumable_perception() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();
        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault).unwrap();

        assert_eq!(perception.status(), AromaticityStatus::Complete);
        assert!(perception.diagnostics().is_empty());
        assert_eq!(perception.assignment().atom_ids(), &[0, 1, 2, 3, 4, 5]);

        let assignment = perception.into_assignment();
        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5]);

        let aromaticized = smiles
            .perceive_aromaticity_for(AromaticityPolicy::RdkitDefault)
            .unwrap()
            .into_aromaticized();
        assert!(aromaticized.nodes().iter().all(Atom::aromatic));
    }

    #[test]
    fn perception_exposes_source_bond_for_aromatic_triple_bond() {
        let smiles: Smiles = "C1=CC#CC=C1".parse().unwrap();
        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault).unwrap();

        assert!(perception.assignment().contains_edge(2, 3));
        assert_eq!(perception.source_bond_for_node_pair((2, 3)), Some(Bond::Triple));
        assert_eq!(perception.source_bond_for_node_pair((3, 2)), Some(Bond::Triple));
        assert_edge_descriptor(perception.aromaticized(), (2, 3), Bond::Triple, true);
    }

    #[test]
    fn aromatic_triple_bond_edge_has_named_aromaticity_and_bond_order() {
        let smiles: Smiles = "C1=CC#CC=C1".parse().unwrap();
        let aromatic = smiles
            .perceive_aromaticity_for(AromaticityPolicy::RdkitDefault)
            .unwrap()
            .into_aromaticized();

        let triple = aromatic
            .edges_for_node(2)
            .find(|edge| edge.bond() == Bond::Triple)
            .expect("expected aromatic triple edge");

        assert!(triple.is_aromatic());
    }

    #[test]
    fn wildcard_perception_exposes_source_bond_for_aromatic_triple_bond() {
        let smiles: WildcardSmiles = "C1=CC#CC=C1".parse().unwrap();
        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault).unwrap();

        assert!(perception.assignment().contains_edge(2, 3));
        assert_eq!(perception.source_bond_for_node_pair((2, 3)), Some(Bond::Triple));
        let edge = perception.aromaticized().edge_for_node_pair((2, 3)).expect("expected edge");
        assert_eq!(edge.bond(), Bond::Triple);
        assert!(edge.is_aromatic());
    }

    #[test]
    fn rdkit_documented_fused_system_keeps_aromatic_atom_bridge_bonds_non_aromatic() {
        let smiles: Smiles = "C1=CC2=C(C=C1)C1=CC=CC=C21".parse().unwrap();
        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault).unwrap();
        let assignment = perception.assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(
            assignment.bond_edges(),
            &[
                [0, 1],
                [0, 5],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [6, 7],
                [6, 11],
                [7, 8],
                [8, 9],
                [9, 10],
                [10, 11],
            ]
        );
        assert!(!assignment.contains_edge(3, 6));
        assert!(!assignment.contains_edge(2, 11));
        assert_edge_descriptor(perception.aromaticized(), (3, 6), Bond::Single, false);
        assert_edge_descriptor(perception.aromaticized(), (2, 11), Bond::Single, false);
    }

    #[test]
    fn rdkit_documented_exocyclic_fused_system_keeps_aromatic_atom_double_bridge_non_aromatic() {
        let smiles: Smiles = "O=C1C=CC(=O)C2=C1OC=CO2".parse().unwrap();
        let perception = smiles.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault).unwrap();
        let assignment = perception.assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[1, 2, 3, 4, 6, 7, 8, 9, 10, 11]);
        assert_eq!(
            assignment.bond_edges(),
            &[[1, 2], [1, 7], [2, 3], [3, 4], [4, 6], [6, 11], [7, 8], [8, 9], [9, 10], [10, 11],]
        );
        assert!(!assignment.contains_edge(6, 7));
        assert_edge_descriptor(perception.aromaticized(), (6, 7), Bond::Double, false);
    }

    #[test]
    fn rdkit_documented_radical_aromaticity_special_cases_match_default_model() {
        let hetero_radical: Smiles = "C1=C[N]C=C1".parse().unwrap();
        let hetero_radical_assignment =
            hetero_radical.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(hetero_radical_assignment.status(), AromaticityStatus::Complete);
        assert!(hetero_radical_assignment.atom_ids().is_empty());
        assert!(hetero_radical_assignment.bond_edges().is_empty());

        let charged_carbon_radical: Smiles = "C1=CC=CC=C[C+]1".parse().unwrap();
        let charged_carbon_radical_assignment =
            charged_carbon_radical.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);
        assert_eq!(charged_carbon_radical_assignment.status(), AromaticityStatus::Complete);
        assert!(charged_carbon_radical_assignment.atom_ids().is_empty());
        assert!(charged_carbon_radical_assignment.bond_edges().is_empty());

        let neutral_carbon_radical: Smiles = "C1=[C]NC=C1".parse().unwrap();
        let perception = neutral_carbon_radical
            .perceive_aromaticity_for(AromaticityPolicy::RdkitDefault)
            .unwrap();
        let neutral_carbon_radical_assignment = perception.assignment();
        assert_eq!(neutral_carbon_radical_assignment.status(), AromaticityStatus::Complete);
        assert_eq!(neutral_carbon_radical_assignment.atom_ids(), &[0, 1, 2, 3, 4]);
        assert_eq!(
            neutral_carbon_radical_assignment.bond_edges(),
            &[[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]
        );
        assert_edge_descriptor(perception.aromaticized(), (0, 1), Bond::Double, true);
    }

    #[test]
    fn rdkit_documented_tellurium_aromaticity_extension_matches_default_model() {
        let smiles: Smiles = "OC(=O)c1[te]ccc1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[3, 4, 5, 6, 7]);
        assert_eq!(assignment.bond_edges(), &[[3, 4], [3, 7], [4, 5], [5, 6], [6, 7]]);
    }

    #[test]
    fn rdkit_documented_dummy_atom_aromaticity_matches_wildcard_model() {
        let six_member: WildcardSmiles = "*1=CC=CC=C1".parse().unwrap();
        let six_member_perception =
            six_member.perceive_aromaticity_for(AromaticityPolicy::RdkitDefault).unwrap();
        let six_member_assignment = six_member_perception.assignment();

        assert_eq!(six_member_assignment.status(), AromaticityStatus::Complete);
        assert_eq!(six_member_assignment.atom_ids(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(
            six_member_assignment.bond_edges(),
            &[[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]
        );
        let wildcard_edge =
            six_member_perception.aromaticized().edge_for_node_pair((0, 1)).unwrap();
        assert_eq!(wildcard_edge.bond(), Bond::Double);
        assert!(wildcard_edge.is_aromatic());

        let five_member: WildcardSmiles = "C1=C*C=C1".parse().unwrap();
        let five_member_assignment =
            five_member.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault);

        assert_eq!(five_member_assignment.status(), AromaticityStatus::Complete);
        assert_eq!(five_member_assignment.atom_ids(), &[0, 1, 2, 3, 4]);
        assert_eq!(five_member_assignment.bond_edges(), &[[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]);
    }

    #[test]
    fn aromaticity_assignment_for_rdkit_default_matches_default_entrypoint() {
        let smiles: Smiles = "C1=CC=CC=C1".parse().unwrap();

        assert_eq!(
            smiles.aromaticity_assignment(),
            smiles.aromaticity_assignment_for(AromaticityPolicy::RdkitDefault)
        );
    }

    #[test]
    fn aromaticity_assignment_new_with_diagnostics_canonicalizes_inputs() {
        let assignment = AromaticityAssignment::new_with_diagnostics(
            AromaticityStatus::Partial,
            vec![3, 1, 1, 2],
            vec![[1, 3], [1, 3], [0, 2]],
            vec![
                AromaticityDiagnostic::SymmSssrHitQueueCutoff,
                AromaticityDiagnostic::SymmSssrUsedFallback,
                AromaticityDiagnostic::SymmSssrUsedFallback,
            ],
        );

        assert_eq!(assignment.atom_ids(), &[1, 2, 3]);
        assert_eq!(assignment.bond_edges(), &[[0, 2], [1, 3]]);
        assert_eq!(
            assignment.diagnostics(),
            &[
                AromaticityDiagnostic::SymmSssrUsedFallback,
                AromaticityDiagnostic::SymmSssrHitQueueCutoff,
            ]
        );
    }

    #[test]
    fn aromaticity_assignment_reports_sssr_fallback_diagnostic_for_large_polycycle_case() {
        let smiles: Smiles = LARGE_POLYCYCLE_FRONTIER_CASE.parse().unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Partial);
        assert!(assignment.diagnostics().contains(&AromaticityDiagnostic::SymmSssrUsedFallback));
        assert!(!assignment.diagnostics().contains(&AromaticityDiagnostic::SymmSssrHitQueueCutoff));
    }

    #[test]
    fn aromaticity_assignment_perceives_kekule_pyrrole() {
        let smiles: Smiles = "C1=CNC=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4]);
        assert_eq!(assignment.bond_edges(), &[[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]);
    }

    #[test]
    fn aromaticity_assignment_perceives_keto_pyrimidinone() {
        let smiles: Smiles = "C1=C(NC(=O)N=C1)N".parse().unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 5, 6]);
        assert_eq!(assignment.bond_edges(), &[[0, 1], [0, 6], [1, 2], [2, 3], [3, 5], [5, 6]]);
    }

    #[test]
    fn aromaticity_assignment_handles_charged_pyridinium_systems() {
        let smiles: Smiles = "C1=CC(=C[N+](=C1)C2C(C(C(O2)COP(=O)(O)OP(=O)(O)OCC3C(C(C(O3)N4C=NC5=C(N=CN=C54)N)OP(=O)(O)O)O)O)O)C(=O)N"
            .parse()
            .unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5, 27, 28, 29, 30, 31, 32, 33, 34, 35]);
        assert_eq!(
            assignment.bond_edges(),
            &[
                [0, 1],
                [0, 5],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [27, 28],
                [27, 35],
                [28, 29],
                [29, 30],
                [30, 31],
                [30, 35],
                [31, 32],
                [32, 33],
                [33, 34],
                [34, 35],
            ]
        );
    }

    #[test]
    fn aromaticity_assignment_handles_phosphinine_oxide_after_rdkit_cleanup() {
        let smiles: Smiles = "C1=CC=C2C(=C1)C=CC=P2=O".parse().unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            assignment.bond_edges(),
            &[
                [0, 1],
                [0, 5],
                [1, 2],
                [2, 3],
                [3, 4],
                [3, 9],
                [4, 5],
                [4, 6],
                [6, 7],
                [7, 8],
                [8, 9]
            ]
        );
    }

    #[test]
    fn fused_symm_sssr_groups_indole_into_one_component() {
        let smiles: Smiles = "C1=CC=C2[NH]C=CC2=C1".parse().unwrap();
        let components = smiles.fused_symm_sssr_components();

        assert_eq!(components.len(), 1);
        assert_eq!(components[0].atom_ids, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            components[0].bond_edges,
            vec![[0, 1], [0, 8], [1, 2], [2, 3], [3, 4], [3, 7], [4, 5], [5, 6], [6, 7], [7, 8]]
        );
    }

    #[test]
    fn aromaticity_assignment_keeps_cycloheptatriene_non_aromatic() {
        let smiles: Smiles = "C1=CC=CCC=C1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty());
        assert!(assignment.bond_edges().is_empty());
    }

    #[test]
    fn aromaticity_assignment_keeps_cubane_non_aromatic() {
        let smiles: Smiles = "C12C3C4C1C5C2C3C45".parse().unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert!(assignment.atom_ids().is_empty());
        assert!(assignment.bond_edges().is_empty());
    }

    #[test]
    fn aromaticity_assignment_perceives_kekule_naphthalene() {
        let smiles: Smiles = "C1=CC=C2C=CC=CC2=C1".parse().unwrap();
        let perception = smiles.perceive_aromaticity().unwrap();
        let assignment = perception.assignment();
        let aromaticized = perception.aromaticized();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            assignment.bond_edges(),
            &[
                [0, 1],
                [0, 9],
                [1, 2],
                [2, 3],
                [3, 4],
                [3, 8],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9]
            ]
        );
        assert!(aromaticized.nodes().iter().all(Atom::aromatic));
        assert_edge_descriptor(aromaticized, (3, 8), Bond::Single, true);
    }

    #[test]
    fn aromaticity_assignment_keeps_azulene_fusion_bond_non_aromatic() {
        let smiles: Smiles = "c1cccc2cccc2c1".parse().unwrap();
        let assignment = smiles.aromaticity_assignment();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            assignment.bond_edges(),
            &[[0, 1], [0, 9], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]
        );
        assert!(!assignment.contains_edge(4, 8));
    }

    #[test]
    fn aromaticity_assignment_perceives_kekule_indole() {
        let smiles: Smiles = "C1=CC=C2[NH]C=CC2=C1".parse().unwrap();
        let perception = smiles.perceive_aromaticity().unwrap();
        let assignment = perception.assignment();
        let aromaticized = perception.aromaticized();

        assert_eq!(assignment.status(), AromaticityStatus::Complete);
        assert_eq!(assignment.atom_ids(), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            assignment.bond_edges(),
            &[[0, 1], [0, 8], [1, 2], [2, 3], [3, 4], [3, 7], [4, 5], [5, 6], [6, 7], [7, 8]]
        );
        assert!(aromaticized.nodes().iter().all(Atom::aromatic));
        assert_edge_descriptor(aromaticized, (3, 7), Bond::Single, true);
    }

    #[test]
    fn render_smoke_from_parsed_smiles() {
        let smiles = Smiles::from_str("CC").expect("should parse");
        let rendered = smiles.render();
        assert_eq!(rendered, "CC");
        assert_eq!(format!("{smiles}"), "CC");
    }

    #[test]
    fn invalid_bonds_rejected() {
        assert!("-N".parse::<Smiles>().is_err());
    }

    #[test]
    fn edge_cases_branches() {
        let case_1 = "B(s)s";
        let case_1_smiles =
            case_1.parse::<Smiles>().unwrap_or_else(|_| panic!("This is a valid SMILES"));
        let case_1_smiles_rerendered = case_1_smiles.to_string();
        let case_1_smiles_reparsed = case_1_smiles_rerendered
            .parse::<Smiles>()
            .unwrap_or_else(|_| panic!("Failed to reparse B(s)s"));
        assert_eq!(case_1_smiles_reparsed.to_string(), case_1_smiles_rerendered);
    }

    #[test]
    fn branch_render_regression_non_aromatic_reaches_a_fixed_point() {
        let smiles: Smiles = "C(O)N".parse().unwrap();
        let rendered = smiles.to_string();
        let resmiles: Smiles = rendered.parse().unwrap();
        let rerendered = resmiles.to_string();
        assert_eq!(rendered, rerendered);
    }

    #[test]
    fn parse_b_s_branch_shape_is_correct() {
        let smiles: Smiles = "B(s)s".parse().unwrap();

        assert_eq!(smiles.nodes().len(), 3);
        assert_eq!(smiles.number_of_bonds(), 2);
        assert_eq!(smiles.edge_for_node_pair((0, 1)).unwrap().bond(), Bond::Single);
        assert_eq!(smiles.edge_for_node_pair((0, 2)).unwrap().bond(), Bond::Single);
        assert_eq!(smiles.edge_for_node_pair((1, 2)), None);
    }

    #[test]
    fn render_b_s_branch_reaches_a_fixed_point() {
        let smiles: Smiles = "B(s)s".parse().unwrap();
        let rendered = smiles.to_string();
        let reparsed: Smiles = rendered.parse().unwrap();

        assert_eq!(rendered, reparsed.to_string());
    }

    #[test]
    fn edge_case_rings_is_invalid() {
        let err = "P1P1".parse::<Smiles>().expect_err("P1P1 should be invalid");
        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
        assert_eq!(err.start(), 3);
        assert_eq!(err.end(), 4);
    }

    #[test]
    fn edge_case_wildcard_carbon_bond() {
        let input = "*c-c";
        let smiles: WildcardSmiles = input.parse().unwrap();

        let rendered = smiles.to_string();
        let reparsed: WildcardSmiles = rendered.parse().unwrap();
        let rerendered = reparsed.to_string();

        assert_eq!(rendered, rerendered);
    }

    #[test]
    fn wildcard_aromaticity_perception_preserves_wildcard_wrapper() {
        let smiles: WildcardSmiles = "*C1=CC=CC=C1".parse().unwrap();
        let perception = smiles.perceive_aromaticity().unwrap();

        assert!(perception.assignment().contains_atom(1));

        let aromaticized: WildcardSmiles = perception.into_aromaticized();

        assert!(aromaticized.node_by_id(0).unwrap().symbol().is_wildcard());
        assert!(aromaticized.node_by_id(1).unwrap().aromatic());
    }

    #[test]
    fn strict_and_wildcard_smiles_convert_through_traits() {
        let strict: Smiles = "CO".parse().unwrap();
        let wildcard = WildcardSmiles::from(strict.clone());

        assert_eq!(wildcard.to_string(), strict.to_string());
        assert_eq!(Smiles::try_from(wildcard).unwrap(), strict);

        let wildcard: WildcardSmiles = "*O".parse().unwrap();
        assert_eq!(Smiles::try_from(wildcard).unwrap_err(), SmilesError::WildcardAtomNotAllowed);
    }

    #[test]
    fn explicit_single_ring_closure_between_aromatic_atoms_is_preserved() {
        let smiles: Smiles = "c1-c-c-c-c-c1".parse().unwrap();
        let rendered = smiles.to_string();
        let reparsed: Smiles = rendered.parse().unwrap();
        assert_eq!(rendered, reparsed.to_string());
    }

    #[test]
    fn explicit_single_between_aromatic_atoms_survives_render_roundtrip() {
        let smiles: WildcardSmiles = "*c-c".parse().unwrap();
        let rendered = smiles.to_string();
        let reparsed: WildcardSmiles = rendered.parse().unwrap();

        assert_eq!(rendered, reparsed.to_string());
        assert_eq!(reparsed.edge_for_node_pair((0, 1)).unwrap().bond(), Bond::Single);
    }

    #[test]
    fn parser_rejects_self_loop_ring() {
        let err = "C11".parse::<Smiles>().expect_err("C11 should be invalid");
        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
    }

    #[test]
    fn parser_rejects_repeated_ring_number_on_same_atom() {
        let err = "C88SC88".parse::<Smiles>().expect_err("C88SC88 should be invalid");
        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
    }

    #[test]
    fn parser_rejects_parallel_ring_bond_between_same_atoms() {
        let err = "C12CCCCC12".parse::<Smiles>().expect_err("parallel edge should be invalid");
        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
    }

    #[test]
    fn hydrogen_with_single_explicit_hydrogen_is_accepted_for_compatibility() {
        let smiles: Smiles = "[HH]".parse().expect("[HH] should be accepted for compatibility");
        assert_eq!(smiles.nodes().len(), 1);
        assert_eq!(smiles.number_of_bonds(), 0);
        assert_eq!(smiles.nodes()[0].element(), Some(Element::H));
        assert_eq!(smiles.nodes()[0].hydrogen_count(), 1);
        assert_eq!(smiles.to_string(), "[HH]");
    }

    #[test]
    fn hydrogen_with_more_than_one_explicit_hydrogen_stays_invalid() {
        let invalid =
            "[HH2]".parse::<Smiles>().expect_err("Hydrogens cannot have explicit hydrogens > 1");
        assert_eq!(invalid.smiles_error(), SmilesError::InvalidHydrogenWithExplicitHydrogensFound);
    }
}
