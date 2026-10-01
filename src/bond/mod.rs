//! Module for specifying the bond between two atoms in a `SMILES` string
pub mod bond_edge;
pub mod ring_num;

use core::fmt;

#[derive(Copy, Debug, Default, PartialEq, Clone, Eq, Hash)]
/// Enum used to specify the Bond type, based on SMILES specification
pub enum Bond {
    #[default]
    /// Implicit single bond or explicit with `-`
    Single,
    /// Defined with `=`
    Double,
    /// Defined with `#`
    Triple,
    /// Defined with `$`
    Quadruple,
    /// Represents a stereochemical single bond `/` (up)
    Up,
    /// Represents a stereochemical single bond `\` (down)
    Down,
}

impl fmt::Display for Bond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.smiles_symbol())
    }
}

impl Bond {
    #[inline]
    #[must_use]
    pub(crate) const fn smiles_symbol(self) -> &'static str {
        match self {
            Self::Single => "-",
            Self::Double => "=",
            Self::Triple => "#",
            Self::Quadruple => "$",
            Self::Up => "/",
            Self::Down => "\\",
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn without_direction(self) -> Self {
        match self {
            Self::Up | Self::Down => Self::Single,
            _ => self,
        }
    }
}

/// Parsed or rendered bond syntax with aromaticity carried separately from
/// the underlying bond order.
#[derive(Copy, Debug, Default, PartialEq, Clone, Eq, Hash)]
pub struct BondDescriptor {
    bond: Bond,
    aromatic: bool,
}

impl BondDescriptor {
    /// Creates a non-aromatic bond descriptor for the provided bond order.
    #[inline]
    #[must_use]
    pub const fn new(bond: Bond) -> Self {
        Self { bond, aromatic: false }
    }

    /// Creates an aromatic bond descriptor for the provided underlying bond
    /// order.
    #[inline]
    #[must_use]
    pub const fn aromatic(bond: Bond) -> Self {
        Self { bond, aromatic: true }
    }

    /// Returns the underlying bond order or direction.
    #[inline]
    #[must_use]
    pub const fn bond(self) -> Bond {
        self.bond
    }

    /// Returns whether this bond is aromatic.
    #[inline]
    #[must_use]
    pub const fn is_aromatic(self) -> bool {
        self.aromatic
    }

    #[inline]
    #[must_use]
    pub(crate) const fn with_bond(mut self, bond: Bond) -> Self {
        self.bond = bond;
        self
    }
}

impl From<Bond> for BondDescriptor {
    #[inline]
    fn from(bond: Bond) -> Self {
        Self::new(bond)
    }
}

impl fmt::Display for BondDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.aromatic && self.bond == Bond::Single {
            f.write_str(":")
        } else {
            f.write_str(self.bond.smiles_symbol())
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use crate::bond::{Bond, BondDescriptor};

    #[test]
    fn test_default() {
        assert_eq!(Bond::default(), Bond::Single);
    }
    #[test]
    fn test_bond_fmt_all_arms() {
        let cases = [
            (Bond::Single, "-"),
            (Bond::Double, "="),
            (Bond::Triple, "#"),
            (Bond::Quadruple, "$"),
            (Bond::Up, "/"),
            (Bond::Down, "\\"),
        ];

        for (bond, expected) in cases {
            assert_eq!(expected, bond.to_string());
        }
    }

    #[test]
    fn directional_bonds_collapse_to_single() {
        assert_eq!(Bond::Up.without_direction(), Bond::Single);
        assert_eq!(Bond::Down.without_direction(), Bond::Single);
        assert_eq!(Bond::Double.without_direction(), Bond::Double);
    }

    #[test]
    fn bond_descriptor_carries_aromaticity_separately() {
        let aromatic_single = BondDescriptor::aromatic(Bond::Single);
        let aromatic_triple = BondDescriptor::aromatic(Bond::Triple);

        assert_eq!(aromatic_single.bond(), Bond::Single);
        assert!(aromatic_single.is_aromatic());
        assert_eq!(aromatic_single.to_string(), ":");
        assert_eq!(aromatic_triple.to_string(), "#");
        assert_eq!(BondDescriptor::from(Bond::Double).to_string(), "=");
    }
}
