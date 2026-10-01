//! Module for bonds stored as graph edge values.

use crate::bond::{Bond, BondDescriptor, ring_num::RingNum};

/// Contains the two atom indices connected via a [`Bond`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BondEdge {
    source: usize,
    target: usize,
    bond: Bond,
    ring_num: Option<RingNum>,
    aromatic: bool,
}

impl BondEdge {
    /// Creates a new non-aromatic bond edge.
    #[inline]
    #[must_use]
    pub const fn new(source: usize, target: usize, bond: Bond, ring_num: Option<RingNum>) -> Self {
        Self::with_aromaticity(source, target, bond, ring_num, false)
    }

    /// Creates a new bond edge with explicit aromaticity.
    #[inline]
    #[must_use]
    pub const fn with_aromaticity(
        source: usize,
        target: usize,
        bond: Bond,
        ring_num: Option<RingNum>,
        aromatic: bool,
    ) -> Self {
        Self { source, target, bond, ring_num, aromatic }
    }

    /// Creates a new bond edge from a bond descriptor.
    #[inline]
    #[must_use]
    pub const fn from_descriptor(
        source: usize,
        target: usize,
        descriptor: BondDescriptor,
        ring_num: Option<RingNum>,
    ) -> Self {
        Self::with_aromaticity(
            source,
            target,
            descriptor.bond(),
            ring_num,
            descriptor.is_aromatic(),
        )
    }

    /// Returns the source atom id.
    #[inline]
    #[must_use]
    pub const fn source(self) -> usize {
        self.source
    }

    /// Returns the target atom id.
    #[inline]
    #[must_use]
    pub const fn target(self) -> usize {
        self.target
    }

    /// Returns both endpoint atom ids in stored order.
    #[inline]
    #[must_use]
    pub const fn endpoints(self) -> [usize; 2] {
        [self.source, self.target]
    }

    /// Returns the underlying bond order or direction.
    #[inline]
    #[must_use]
    pub const fn bond(self) -> Bond {
        self.bond
    }

    /// Returns the underlying bond order or direction.
    #[inline]
    #[must_use]
    pub const fn bond_type(self) -> Bond {
        self.bond
    }

    /// Returns the ring number attached to this edge, if any.
    #[inline]
    #[must_use]
    pub const fn ring_num(self) -> Option<RingNum> {
        self.ring_num
    }

    /// Returns whether this edge is aromatic.
    #[inline]
    #[must_use]
    pub const fn is_aromatic(self) -> bool {
        self.aromatic
    }

    /// Returns this edge's bond order and aromaticity as a descriptor.
    #[inline]
    #[must_use]
    pub const fn descriptor(self) -> BondDescriptor {
        if self.aromatic {
            BondDescriptor::aromatic(self.bond)
        } else {
            BondDescriptor::new(self.bond)
        }
    }

    /// Returns the other endpoint for an incident atom id, if any.
    #[inline]
    #[must_use]
    pub const fn other(self, atom_id: usize) -> Option<usize> {
        if self.source == atom_id {
            Some(self.target)
        } else if self.target == atom_id {
            Some(self.source)
        } else {
            None
        }
    }
}

/// Creates a new non-aromatic edge.
///
/// # Examples
///
/// ```
/// use smiles_rs::bond::{Bond, bond_edge::bond_edge};
///
/// let edge = bond_edge(0, 1, Bond::Double, None);
/// assert_eq!(edge.source(), 0);
/// assert_eq!(edge.target(), 1);
/// assert_eq!(edge.bond(), Bond::Double);
/// assert!(!edge.is_aromatic());
/// ```
#[inline]
#[must_use]
pub const fn bond_edge(
    node_a: usize,
    node_b: usize,
    bond: Bond,
    ring_num: Option<RingNum>,
) -> BondEdge {
    BondEdge::new(node_a, node_b, bond, ring_num)
}

/// Creates a new edge with explicit aromaticity.
#[inline]
#[must_use]
pub const fn bond_edge_with_aromaticity(
    node_a: usize,
    node_b: usize,
    bond: Bond,
    ring_num: Option<RingNum>,
    aromatic: bool,
) -> BondEdge {
    BondEdge::with_aromaticity(node_a, node_b, bond, ring_num, aromatic)
}

/// Creates a new edge from a bond descriptor.
#[inline]
#[must_use]
pub const fn bond_edge_from_descriptor(
    node_a: usize,
    node_b: usize,
    descriptor: BondDescriptor,
    ring_num: Option<RingNum>,
) -> BondEdge {
    BondEdge::from_descriptor(node_a, node_b, descriptor, ring_num)
}

/// Returns the other node id for the provided incident edge, if any.
///
/// # Examples
///
/// ```
/// use smiles_rs::bond::{
///     Bond,
///     bond_edge::{bond_edge, bond_edge_other},
/// };
///
/// let edge = bond_edge(2, 5, Bond::Single, None);
/// assert_eq!(bond_edge_other(edge, 2), Some(5));
/// assert_eq!(bond_edge_other(edge, 99), None);
/// ```
#[inline]
#[must_use]
pub const fn bond_edge_other(edge: BondEdge, node_id: usize) -> Option<usize> {
    edge.other(node_id)
}

/// Returns the ring number value stored in the edge, if any.
///
/// # Examples
///
/// ```
/// use smiles_rs::bond::{
///     Bond,
///     bond_edge::{bond_edge, bond_edge_ring_num_val},
///     ring_num::RingNum,
/// };
///
/// let edge = bond_edge(0, 1, Bond::Single, Some(RingNum::try_new(7)?));
/// assert_eq!(bond_edge_ring_num_val(edge), Some(7));
/// # Ok::<(), smiles_rs::SmilesError>(())
/// ```
#[inline]
#[must_use]
pub fn bond_edge_ring_num_val(edge: BondEdge) -> Option<u8> {
    edge.ring_num().map(|num| num.get())
}

#[cfg(test)]
mod tests {
    use crate::bond::{
        Bond, BondDescriptor,
        bond_edge::{
            BondEdge, bond_edge, bond_edge_from_descriptor, bond_edge_other,
            bond_edge_ring_num_val, bond_edge_with_aromaticity,
        },
        ring_num::RingNum,
    };

    #[test]
    fn bond_edge_named_accessors_expose_all_fields() {
        let edge = BondEdge::with_aromaticity(3, 7, Bond::Triple, None, true);

        assert_eq!(edge.source(), 3);
        assert_eq!(edge.target(), 7);
        assert_eq!(edge.endpoints(), [3, 7]);
        assert_eq!(edge.bond(), Bond::Triple);
        assert_eq!(edge.bond_type(), Bond::Triple);
        assert_eq!(edge.ring_num(), None);
        assert!(edge.is_aromatic());
        assert_eq!(edge.other(3), Some(7));
        assert_eq!(edge.other(7), Some(3));
        assert_eq!(edge.other(99), None);
    }

    #[test]
    fn test_bond_edge_constructors() {
        let edge = bond_edge(3, 7, Bond::Double, None);

        assert_eq!(edge.source(), 3);
        assert_eq!(edge.target(), 7);
        assert_eq!(edge.endpoints(), [3, 7]);
        assert_eq!(edge.bond(), Bond::Double);
        assert_eq!(edge.ring_num(), None);
        assert!(!edge.is_aromatic());

        let aromatic = bond_edge_with_aromaticity(3, 7, Bond::Triple, None, true);
        assert_eq!(aromatic.bond(), Bond::Triple);
        assert!(aromatic.is_aromatic());

        let from_descriptor =
            bond_edge_from_descriptor(3, 7, BondDescriptor::aromatic(Bond::Single), None);
        assert_eq!(from_descriptor, bond_edge_with_aromaticity(3, 7, Bond::Single, None, true));
    }

    #[test]
    fn bond_edge_descriptor_round_trips_aromaticity() {
        let edge = BondEdge::from_descriptor(0, 1, BondDescriptor::aromatic(Bond::Single), None);

        assert_eq!(edge.bond(), Bond::Single);
        assert!(edge.is_aromatic());
        assert_eq!(edge.descriptor(), BondDescriptor::aromatic(Bond::Single));
    }

    #[test]
    fn test_bond_edge_other() {
        let edge = bond_edge(10, 2, Bond::Single, None);
        assert_eq!(bond_edge_other(edge, 10), Some(2));
        assert_eq!(bond_edge_other(edge, 2), Some(10));
        assert_eq!(bond_edge_other(edge, 5), None);
    }

    #[test]
    fn test_bond_edge_ring_num_value() {
        let ring_num = RingNum::try_new(2).unwrap();
        let edge = bond_edge(0, 1, Bond::Single, Some(ring_num));
        assert_eq!(bond_edge_ring_num_val(edge), Some(2));
        assert_eq!(bond_edge_ring_num_val(bond_edge(0, 1, Bond::Single, None)), None);
    }

    #[test]
    fn test_bond_edge_vertices_preserve_order() {
        let edge: BondEdge = bond_edge(10, 2, Bond::Single, None);
        assert_eq!(edge.endpoints(), [10, 2]);
    }
}
