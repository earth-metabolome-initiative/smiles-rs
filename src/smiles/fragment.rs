//! Standalone fragments carved out of a parent [`Smiles`] graph.
//!
//! A [`Fragment`] keeps the parent's atom and bond typing (including aromatic
//! flags) but recomputes implicit hydrogen counts for the fragment's own
//! connectivity, so an atom that becomes terminal gains the hydrogens its
//! reduced valence implies. Atom ids are remapped to a compact local numbering,
//! with the parent correspondence kept private behind [`Fragment::local_id`]
//! and [`Fragment::parent_id`].

use alloc::{string::String, vec::Vec};

use geometric_traits::traits::SparseValuedMatrixRef;

use super::{BondMatrixBuilder, ConcreteAtoms, Smiles, SmilesAtomPolicy};
use crate::{
    bond::bond_edge::BondEdge,
    errors::{RootError, SubgraphError},
};

/// A standalone molecule carved out of a parent graph, with a private mapping
/// back to the parent's atom ids.
#[derive(Debug, Clone)]
pub struct Fragment<AtomPolicy = ConcreteAtoms> {
    smiles: Smiles<AtomPolicy>,
    parent_of_local: Vec<usize>,
    local_of_parent: Vec<usize>,
}

impl<AtomPolicy: SmilesAtomPolicy> Fragment<AtomPolicy> {
    /// Returns the fragment as a standalone [`Smiles`].
    #[inline]
    #[must_use]
    pub fn smiles(&self) -> &Smiles<AtomPolicy> {
        &self.smiles
    }

    /// Consumes the fragment and returns the standalone [`Smiles`].
    #[inline]
    #[must_use]
    pub fn into_smiles(self) -> Smiles<AtomPolicy> {
        self.smiles
    }

    /// Returns the number of atoms in the fragment.
    #[inline]
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.parent_of_local.len()
    }

    /// Returns the fragment-local id of a parent atom, if it is in the
    /// fragment.
    #[inline]
    #[must_use]
    pub fn local_id(&self, parent_atom: usize) -> Option<usize> {
        self.local_of_parent.get(parent_atom).copied().filter(|&local| local != usize::MAX)
    }

    /// Returns the parent id of a fragment atom.
    ///
    /// # Panics
    ///
    /// Panics if `local_atom` is not a valid fragment atom id.
    #[inline]
    #[must_use]
    pub fn parent_id(&self, local_atom: usize) -> usize {
        self.parent_of_local[local_atom]
    }

    /// Renders the fragment anchored at a parent atom. When `isomeric` is
    /// false, isotopes and stereochemistry are dropped (RDKit
    /// `isomericSmiles=False`).
    ///
    /// # Errors
    ///
    /// Returns [`RootError::AtomNotInFragment`] if `parent_atom` is not part of
    /// the fragment.
    pub fn render_rooted(&self, parent_atom: usize, isomeric: bool) -> Result<String, RootError> {
        let local = self.local_id(parent_atom).ok_or(RootError::AtomNotInFragment(parent_atom))?;
        if isomeric {
            Ok(self.smiles.render_rooted(local))
        } else {
            Ok(self.smiles.non_isomeric().render_rooted(local))
        }
    }
}

impl<AtomPolicy: SmilesAtomPolicy> Smiles<AtomPolicy> {
    /// Builds a fragment from an atom set: every bond among the chosen atoms is
    /// included (an induced subgraph). Implicit hydrogen counts are recomputed
    /// for the fragment.
    ///
    /// # Errors
    ///
    /// Returns [`SubgraphError::AtomOutOfRange`] if any id is not a valid atom.
    pub fn fragment_from_atoms(
        &self,
        atoms: impl IntoIterator<Item = usize>,
    ) -> Result<Fragment<AtomPolicy>, SubgraphError> {
        let node_count = self.nodes().len();
        let mut parent_of_local = Vec::new();
        let mut local_of_parent = vec![usize::MAX; node_count];
        for atom in atoms {
            if atom >= node_count {
                return Err(SubgraphError::AtomOutOfRange(atom));
            }
            register_local(&mut parent_of_local, &mut local_of_parent, atom);
        }

        let mut builder = BondMatrixBuilder::default();
        for ((row, column), entry) in self.bond_matrix.sparse_entries() {
            if row >= column {
                continue;
            }
            let (local_row, local_column) = (local_of_parent[row], local_of_parent[column]);
            if local_row == usize::MAX || local_column == usize::MAX {
                continue;
            }
            builder
                .push_edge_with_descriptor(local_row, local_column, entry.descriptor(), None)
                .unwrap_or_else(|_| unreachable!("induced fragment preserves a simple graph"));
        }

        Ok(self.finish_fragment(parent_of_local, local_of_parent, builder))
    }

    /// Builds a fragment from a bond set (RDKit `PathToSubmol`): exactly the
    /// listed bonds are kept and their endpoints become the fragment's atoms.
    /// Implicit hydrogen counts are recomputed for the fragment.
    ///
    /// # Errors
    ///
    /// Returns [`SubgraphError::AtomOutOfRange`] if an endpoint is not a valid
    /// atom, or [`SubgraphError::BondReferencesUnknownAtom`] if a listed bond
    /// is not an edge of the parent graph.
    pub fn fragment_from_bonds(
        &self,
        bonds: impl IntoIterator<Item = BondEdge>,
    ) -> Result<Fragment<AtomPolicy>, SubgraphError> {
        let node_count = self.nodes().len();
        let mut parent_of_local = Vec::new();
        let mut local_of_parent = vec![usize::MAX; node_count];
        let mut builder = BondMatrixBuilder::default();
        for edge in bonds {
            let [source, target] = edge.endpoints();
            if source >= node_count {
                return Err(SubgraphError::AtomOutOfRange(source));
            }
            if target >= node_count {
                return Err(SubgraphError::AtomOutOfRange(target));
            }
            let descriptor = self
                .edge_for_node_pair((source, target))
                .ok_or(SubgraphError::BondReferencesUnknownAtom(target))?
                .descriptor();
            let local_source = register_local(&mut parent_of_local, &mut local_of_parent, source);
            let local_target = register_local(&mut parent_of_local, &mut local_of_parent, target);
            builder
                .push_edge_with_descriptor(local_source, local_target, descriptor, None)
                .map_err(|_| SubgraphError::BondReferencesUnknownAtom(target))?;
        }

        Ok(self.finish_fragment(parent_of_local, local_of_parent, builder))
    }

    fn finish_fragment(
        &self,
        parent_of_local: Vec<usize>,
        local_of_parent: Vec<usize>,
        builder: BondMatrixBuilder,
    ) -> Fragment<AtomPolicy> {
        let atom_nodes: Vec<_> =
            parent_of_local.iter().map(|&parent| self.atom_nodes[parent]).collect();
        let atom_count = atom_nodes.len();
        let parsed_stereo_neighbors = vec![Vec::new(); atom_count];
        let smiles = Self::from_bond_matrix_parts_with_parsed_stereo_and_source(
            atom_nodes,
            builder.finish(atom_count),
            parsed_stereo_neighbors,
            None,
        );
        Fragment { smiles, parent_of_local, local_of_parent }
    }
}

/// Assigns (or returns) the local id of a parent atom, recording it on first
/// sight.
fn register_local(
    parent_of_local: &mut Vec<usize>,
    local_of_parent: &mut [usize],
    parent_atom: usize,
) -> usize {
    let local = &mut local_of_parent[parent_atom];
    if *local == usize::MAX {
        *local = parent_of_local.len();
        parent_of_local.push(parent_atom);
    }
    *local
}
