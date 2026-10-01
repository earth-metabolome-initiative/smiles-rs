use geometric_traits::traits::{
    ConnectedComponents as GeometricConnectedComponents,
    algorithms::connected_components::ConnectedComponentsResult as GeometricConnectedComponentsResult,
};

use super::{ConcreteAtoms, Smiles, SmilesAtomPolicy, WildcardAtoms, WildcardSmiles};
use crate::atom::Atom;

/// Connected-component view over a parsed [`Smiles`] graph.
///
/// This is a thin wrapper around the `geometric-traits` connected-components
/// implementation, exposed here so callers do not need to depend on
/// `geometric-traits` directly just to inspect disconnected fragments.
pub struct SmilesComponents<'a, AtomPolicy: SmilesAtomPolicy = ConcreteAtoms> {
    inner: GeometricConnectedComponentsResult<'a, Smiles<AtomPolicy>, usize>,
}

/// Connected-component view over a [`WildcardSmiles`] graph.
///
/// This mirrors [`SmilesComponents`] while keeping the wildcard-capable public
/// API on [`WildcardSmiles`].
pub struct WildcardSmilesComponents<'a> {
    inner: SmilesComponents<'a, WildcardAtoms>,
}

impl<'a, AtomPolicy: SmilesAtomPolicy> SmilesComponents<'a, AtomPolicy> {
    #[inline]
    pub(crate) const fn new(
        inner: GeometricConnectedComponentsResult<'a, Smiles<AtomPolicy>, usize>,
    ) -> Self {
        Self { inner }
    }

    /// Returns the number of connected components in the graph.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// assert_eq!(smiles.connected_components().number_of_components(), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn number_of_components(&self) -> usize {
        self.inner.number_of_components()
    }

    /// Returns the size of the largest connected component.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// assert_eq!(smiles.connected_components().largest_component_size(), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn largest_component_size(&self) -> usize {
        self.inner.largest_component_size()
    }

    /// Returns the size of the smallest connected component.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// assert_eq!(smiles.connected_components().smallest_component_size(), 1);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn smallest_component_size(&self) -> usize {
        self.inner.smallest_component_size()
    }

    /// Returns the connected-component identifier of the provided node id.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// let components = smiles.connected_components();
    ///
    /// assert_eq!(components.component_of_node(0), components.component_of_node(1));
    /// assert_ne!(components.component_of_node(0), components.component_of_node(2));
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn component_of_node(&self, node: usize) -> usize {
        self.inner.component_of_node(node)
    }

    /// Returns an iterator over the connected-component identifier of each
    /// node.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// let identifiers = smiles.connected_components().component_identifiers().collect::<Vec<_>>();
    ///
    /// assert_eq!(identifiers.len(), smiles.nodes().len());
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    pub fn component_identifiers(&self) -> impl Iterator<Item = usize> + '_ {
        self.inner.component_identifiers()
    }

    /// Returns an iterator over the node ids belonging to the given component.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// let components = smiles.connected_components();
    /// let first_component = components.component_of_node(0);
    ///
    /// assert_eq!(components.node_ids_of_component(first_component).collect::<Vec<_>>(), vec![0, 1]);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    pub fn node_ids_of_component(
        &self,
        component_identifier: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        self.inner.node_ids_of_component(component_identifier)
    }

    /// Returns an iterator over the atoms belonging to the given component.
    ///
    /// # Examples
    ///
    /// ```
    /// use elements_rs::Element;
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// let components = smiles.connected_components();
    /// let oxygen_component = components.component_of_node(2);
    ///
    /// assert_eq!(
    ///     components
    ///         .nodes_of_component(oxygen_component)
    ///         .map(|atom| atom.element())
    ///         .collect::<Vec<_>>(),
    ///     vec![Some(Element::O)]
    /// );
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    pub fn nodes_of_component(
        &self,
        component_identifier: usize,
    ) -> impl Iterator<Item = Atom> + '_ {
        self.inner.nodes_of_component(component_identifier)
    }
}

impl<'a> WildcardSmilesComponents<'a> {
    #[inline]
    pub(crate) const fn new(inner: SmilesComponents<'a, WildcardAtoms>) -> Self {
        Self { inner }
    }

    /// Returns the number of connected components in the graph.
    #[inline]
    #[must_use]
    pub fn number_of_components(&self) -> usize {
        self.inner.number_of_components()
    }

    /// Returns the size of the largest connected component.
    #[inline]
    #[must_use]
    pub fn largest_component_size(&self) -> usize {
        self.inner.largest_component_size()
    }

    /// Returns the size of the smallest connected component.
    #[inline]
    #[must_use]
    pub fn smallest_component_size(&self) -> usize {
        self.inner.smallest_component_size()
    }

    /// Returns the connected-component identifier of the provided node id.
    #[inline]
    #[must_use]
    pub fn component_of_node(&self, node: usize) -> usize {
        self.inner.component_of_node(node)
    }

    /// Returns an iterator over the connected-component identifier of each
    /// node.
    #[inline]
    pub fn component_identifiers(&self) -> impl Iterator<Item = usize> + '_ {
        self.inner.component_identifiers()
    }

    /// Returns an iterator over the node ids belonging to the given component.
    #[inline]
    pub fn node_ids_of_component(
        &self,
        component_identifier: usize,
    ) -> impl Iterator<Item = usize> + '_ {
        self.inner.node_ids_of_component(component_identifier)
    }

    /// Returns an iterator over the atoms belonging to the given component.
    #[inline]
    pub fn nodes_of_component(
        &self,
        component_identifier: usize,
    ) -> impl Iterator<Item = Atom> + '_ {
        self.inner.nodes_of_component(component_identifier)
    }
}

impl<AtomPolicy: crate::smiles::SmilesAtomPolicy> Smiles<AtomPolicy> {
    /// Returns the connected components of the graph.
    ///
    /// This is infallible for [`usize`] component markers, because a graph
    /// cannot contain more components than nodes, and node ids are already
    /// stored as `usize`.
    ///
    /// # Panics
    ///
    /// Panics if the underlying geometric-traits connected-components routine
    /// ever reports marker overflow for `usize` component identifiers. That
    /// would indicate a bug in the graph library or an invalid graph size.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::prelude::Smiles;
    ///
    /// let smiles: Smiles = "CC.O".parse()?;
    /// let components = smiles.connected_components();
    /// assert_eq!(components.number_of_components(), 2);
    /// # Ok::<(), smiles_rs::SmilesErrorWithSpan>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn connected_components(&self) -> SmilesComponents<'_, AtomPolicy> {
        let components = GeometricConnectedComponents::<usize>::connected_components(self)
            .expect("usize markers cannot overflow on a usize-indexed graph");
        SmilesComponents::new(components)
    }
}

impl WildcardSmiles {
    /// Returns the connected components of the wildcard-capable graph.
    ///
    /// This mirrors [`Smiles::connected_components`] while preserving the
    /// [`WildcardSmiles`] API surface.
    #[inline]
    #[must_use]
    pub fn connected_components(&self) -> WildcardSmilesComponents<'_> {
        WildcardSmilesComponents::new(self.inner().connected_components())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use elements_rs::Element;

    use crate::{
        atom::{Atom, atom_symbol::AtomSymbol},
        bond::{
            Bond,
            bond_edge::{BondEdge, bond_edge},
        },
        smiles::{BondMatrixBuilder, Smiles},
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

    #[test]
    fn connected_components_expose_sizes_identifiers_and_nodes() {
        let smiles = smiles_from_edges(
            vec![atom(Element::C), atom(Element::O), atom(Element::N), atom(Element::S)],
            &[bond_edge(0, 1, Bond::Single, None), bond_edge(2, 3, Bond::Double, None)],
        );
        let components = smiles.connected_components();

        assert_eq!(components.number_of_components(), 2);
        assert_eq!(components.largest_component_size(), 2);
        assert_eq!(components.smallest_component_size(), 2);
        assert_eq!(components.component_of_node(0), components.component_of_node(1));
        assert_ne!(components.component_of_node(0), components.component_of_node(2));

        let identifiers = components.component_identifiers().collect::<Vec<_>>();
        assert_eq!(identifiers.len(), 4);

        let first_nodes =
            components.node_ids_of_component(components.component_of_node(0)).collect::<Vec<_>>();
        assert_eq!(first_nodes, vec![0, 1]);

        let second_atoms =
            components.nodes_of_component(components.component_of_node(2)).collect::<Vec<_>>();
        assert_eq!(second_atoms, vec![atom(Element::N), atom(Element::S)]);
    }
}
