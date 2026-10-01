//! Radius-N circular atom environments, matching RDKit's
//! `FindAtomEnvironmentOfRadiusN`.
//!
//! The environment of radius `r` around an atom is the cumulative set of bonds
//! reachable within `r` bonds, grown shell by shell. Following RDKit, the
//! environment is *empty* (returned as `None`) when the outermost requested
//! shell adds no new bond, e.g. when the center's eccentricity is below `r` or
//! the center has no incident bonds at all. Ring-closure bonds between two
//! atoms that fall in the same shell are included, so a benzene environment
//! renders as a ring rather than a chain.

use alloc::{string::String, vec::Vec};

use geometric_traits::traits::SparseMatrix2D;

use super::{ConcreteAtoms, Fragment, Smiles, SmilesAtomPolicy};
use crate::{bond::bond_edge::BondEdge, errors::SubgraphError};

/// A borrowed view of the radius-N circular environment of an atom. Always
/// non-empty and always knows its own center.
#[derive(Debug, Clone)]
pub struct AtomEnvironment<'mol, AtomPolicy = ConcreteAtoms> {
    parent: &'mol Smiles<AtomPolicy>,
    center: usize,
    atoms: Vec<usize>,
    bonds: Vec<BondEdge>,
}

impl<AtomPolicy: SmilesAtomPolicy> AtomEnvironment<'_, AtomPolicy> {
    /// Returns the center atom id (in parent coordinates).
    #[inline]
    #[must_use]
    pub fn center(&self) -> usize {
        self.center
    }

    /// Returns the number of atoms in the environment.
    #[inline]
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Returns the number of bonds in the environment.
    #[inline]
    #[must_use]
    pub fn bond_count(&self) -> usize {
        self.bonds.len()
    }

    /// Returns whether a parent atom id is part of the environment.
    #[must_use]
    pub fn contains_atom(&self, atom_id: usize) -> bool {
        self.atoms.contains(&atom_id)
    }

    /// Iterates the parent atom ids in the environment.
    pub fn atoms(&self) -> impl Iterator<Item = usize> + '_ {
        self.atoms.iter().copied()
    }

    /// Iterates the bonds in the environment.
    pub fn bonds(&self) -> impl Iterator<Item = BondEdge> + '_ {
        self.bonds.iter().copied()
    }

    /// Materializes the environment as a standalone [`Fragment`].
    ///
    /// # Errors
    ///
    /// Propagates any [`SubgraphError`] from fragment construction.
    pub fn to_fragment(&self) -> Result<Fragment<AtomPolicy>, SubgraphError> {
        self.parent.fragment_from_bonds(self.bonds.iter().copied())
    }

    /// Renders the environment as a SMILES label rooted at its center. When
    /// `isomeric` is false, isotopes and stereochemistry are dropped (the MAP4
    /// setting).
    ///
    /// # Errors
    ///
    /// Propagates any [`SubgraphError`] from fragment construction.
    pub fn rooted_smiles(&self, isomeric: bool) -> Result<String, SubgraphError> {
        let fragment = self.to_fragment()?;
        Ok(fragment
            .render_rooted(self.center, isomeric)
            .unwrap_or_else(|_| unreachable!("the center always belongs to its own environment")))
    }
}

impl<AtomPolicy: SmilesAtomPolicy> Smiles<AtomPolicy> {
    /// Returns the radius-`radius` circular environment of `center`, or `None`
    /// for the empty-shell case (RDKit `FindAtomEnvironmentOfRadiusN`).
    ///
    /// # Panics
    ///
    /// Panics if `center` is not a valid atom id.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let mol: Smiles = "CCO".parse()?;
    /// assert!(mol.atom_environment(1, 2).is_none()); // central C, eccentricity 1 < 2
    /// let env = mol.atom_environment(0, 2).unwrap();
    /// assert_eq!(env.center(), 0);
    /// assert_eq!(env.atom_count(), 3);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[must_use]
    pub fn atom_environment(
        &self,
        center: usize,
        radius: usize,
    ) -> Option<AtomEnvironment<'_, AtomPolicy>> {
        assert!(center < self.nodes().len(), "atom_environment: center {center} is out of range");
        let (atoms, bonds) = self.grow_radius_environment(center, radius)?;
        Some(AtomEnvironment { parent: self, center, atoms, bonds })
    }

    /// Grows the radius-`radius` environment shell by shell, returning its
    /// atoms and bonds, or `None` if the outermost requested shell adds
    /// nothing.
    fn grow_radius_environment(
        &self,
        center: usize,
        radius: usize,
    ) -> Option<(Vec<usize>, Vec<BondEdge>)> {
        if radius == 0 {
            return None;
        }

        let mut atom_layer = vec![usize::MAX; self.nodes().len()];
        atom_layer[center] = 0;
        let mut atoms = vec![center];
        let mut bonds: Vec<BondEdge> = Vec::new();
        let mut frontier = vec![center];

        for layer in 1..=radius {
            let mut next_frontier = Vec::new();
            let mut bonds_this_layer = 0;
            for &source in &frontier {
                for target in self.bond_matrix.sparse_row(source) {
                    if !self.record_environment_bond(
                        source,
                        target,
                        layer,
                        &mut atom_layer,
                        &mut atoms,
                        &mut next_frontier,
                        &mut bonds,
                    ) {
                        continue;
                    }
                    bonds_this_layer += 1;
                }
            }
            if bonds_this_layer == 0 {
                return None;
            }
            frontier = next_frontier;
            if frontier.is_empty() && layer < radius {
                return None;
            }
        }

        Some((atoms, bonds))
    }

    /// Classifies one frontier edge and, if it belongs to this shell, records
    /// it (and any newly reached atom). Returns whether a bond was added.
    #[allow(clippy::too_many_arguments)]
    fn record_environment_bond(
        &self,
        source: usize,
        target: usize,
        layer: usize,
        atom_layer: &mut [usize],
        atoms: &mut Vec<usize>,
        next_frontier: &mut Vec<usize>,
        bonds: &mut Vec<BondEdge>,
    ) -> bool {
        let target_layer = atom_layer[target];
        let add_bond = if target_layer == usize::MAX {
            atom_layer[target] = layer;
            next_frontier.push(target);
            atoms.push(target);
            true
        } else if target_layer == layer {
            // Discovered earlier in this same shell by another atom.
            true
        } else if target_layer == layer - 1 {
            // Ring closure within the current frontier, recorded once.
            source < target
        } else {
            // Already recorded in an earlier shell.
            false
        };
        if add_bond {
            bonds.push(
                self.edge_for_node_pair((source, target))
                    .unwrap_or_else(|| unreachable!("iterated edge must exist")),
            );
        }
        add_bond
    }

    /// One-shot MAP4 substructure label: environment to fragment to rooted
    /// render. Returns `None` for the empty-shell case and for any fragment
    /// construction failure (both map to MAP4's empty label).
    ///
    /// # Panics
    ///
    /// Panics if `center` is not a valid atom id.
    #[must_use]
    pub fn rooted_environment_smiles(
        &self,
        center: usize,
        radius: usize,
        isomeric: bool,
    ) -> Option<String> {
        self.atom_environment(center, radius)?.rooted_smiles(isomeric).ok()
    }
}
