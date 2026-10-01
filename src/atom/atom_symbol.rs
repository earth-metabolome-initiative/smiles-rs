//! Module for the symbols representing an element in a `SMILES` string
use core::{cmp::Ordering, fmt};

use elements_rs::Element;

#[derive(Copy, Default, Debug, PartialEq, Clone, Eq, Hash)]
/// Enum to allow for standard elements or the `WildCard` variant, represented
/// as `*`
pub enum AtomSymbol {
    /// The explicitly named [`Element`]
    Element(Element),
    /// `WildCard` variant, described [here](http://opensmiles.org/opensmiles.html#inatoms)
    #[default]
    WildCard,
}

impl AtomSymbol {
    /// Creates an atom symbol
    ///
    /// # Examples
    ///
    /// ```
    /// use elements_rs::Element;
    /// use smiles_rs::atom::atom_symbol::AtomSymbol;
    ///
    /// assert_eq!(AtomSymbol::new(Some(Element::C)), AtomSymbol::Element(Element::C));
    /// assert_eq!(AtomSymbol::new(None), AtomSymbol::WildCard);
    /// ```
    #[must_use]
    pub fn new(element_type: Option<Element>) -> Self {
        match element_type {
            Some(element) => AtomSymbol::Element(element),
            None => AtomSymbol::default(),
        }
    }
    /// creates an `AtomSymbol` set as `WildCard`
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::atom_symbol::AtomSymbol;
    ///
    /// assert_eq!(AtomSymbol::new_wildcard(), AtomSymbol::WildCard);
    /// ```
    #[must_use]
    pub fn new_wildcard() -> Self {
        Self::WildCard
    }
    /// Verifies whether the symbol present is a wildcard
    ///
    /// # Examples
    ///
    /// ```
    /// use elements_rs::Element;
    /// use smiles_rs::atom::atom_symbol::AtomSymbol;
    ///
    /// assert!(AtomSymbol::new_wildcard().is_wildcard());
    /// assert!(!AtomSymbol::Element(Element::O).is_wildcard());
    /// ```
    #[must_use]
    pub fn is_wildcard(&self) -> bool {
        matches!(self, AtomSymbol::WildCard)
    }
    /// Returns either the [`Element`] or `None` if wildcard
    ///
    /// # Examples
    ///
    /// ```
    /// use elements_rs::Element;
    /// use smiles_rs::atom::atom_symbol::AtomSymbol;
    ///
    /// assert_eq!(AtomSymbol::Element(Element::N).element(), Some(Element::N));
    /// assert_eq!(AtomSymbol::WildCard.element(), None);
    /// ```
    #[must_use]
    pub fn element(&self) -> Option<Element> {
        match self {
            AtomSymbol::Element(e) => Some(*e),
            AtomSymbol::WildCard => None,
        }
    }
    /// Consumes the `AtomSymbol` and returns the [`Element`] or `None` if
    /// `WildCard`
    ///
    /// # Examples
    ///
    /// ```
    /// use elements_rs::Element;
    /// use smiles_rs::atom::atom_symbol::AtomSymbol;
    ///
    /// assert_eq!(AtomSymbol::Element(Element::S).into_element(), Some(Element::S));
    /// assert_eq!(AtomSymbol::WildCard.into_element(), None);
    /// ```
    #[must_use]
    pub fn into_element(self) -> Option<Element> {
        match self {
            AtomSymbol::Element(e) => Some(e),
            AtomSymbol::WildCard => None,
        }
    }
}

impl fmt::Display for AtomSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Element(e) => f.write_str(e.symbol()),
            Self::WildCard => f.write_str("*"),
        }
    }
}

impl PartialOrd for AtomSymbol {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AtomSymbol {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::WildCard, Self::WildCard) => Ordering::Equal,
            (Self::WildCard, Self::Element(_)) => Ordering::Less,
            (Self::Element(_), Self::WildCard) => Ordering::Greater,
            (Self::Element(left), Self::Element(right)) => left.symbol().cmp(right.symbol()),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use elements_rs::Element;

    use crate::atom::atom_symbol::AtomSymbol;

    #[test]
    fn test_atom_symbols_all() {
        let hydrogen = Element::H;
        let hydro_symbol = AtomSymbol::new(Some(hydrogen));
        assert!(!hydro_symbol.is_wildcard());
        assert_eq!(hydro_symbol.element(), Some(hydrogen));
        let into_hydro = hydro_symbol.into_element();
        assert_eq!(into_hydro, Some(hydrogen));

        let default = AtomSymbol::default();
        assert_eq!(default, AtomSymbol::WildCard);

        let wild = AtomSymbol::new_wildcard();
        assert!(wild.is_wildcard());
    }

    #[test]
    fn test_atom_symbol_fmt_all_arms() {
        let cases = [
            (AtomSymbol::Element(Element::H), "H"),
            (AtomSymbol::WildCard, "*"),
            (AtomSymbol::new(None), "*"),
        ];

        for (symbol, expected) in cases {
            assert_eq!(expected, symbol.to_string());
        }
    }

    #[test]
    fn wildcard_accessors_return_none() {
        let wildcard = AtomSymbol::WildCard;
        assert_eq!(wildcard.element(), None);
        assert_eq!(wildcard.into_element(), None);
    }

    #[test]
    fn atom_symbol_order_places_wildcard_before_elements_and_sorts_by_symbol() {
        let wildcard = AtomSymbol::WildCard;
        let carbon = AtomSymbol::Element(Element::C);
        let oxygen = AtomSymbol::Element(Element::O);

        assert!(wildcard < carbon);
        assert_eq!(wildcard.partial_cmp(&oxygen), Some(core::cmp::Ordering::Less));
        assert_eq!(oxygen.cmp(&carbon), core::cmp::Ordering::Greater);
        assert_eq!(carbon.cmp(&carbon), core::cmp::Ordering::Equal);

        let mut symbols = [oxygen, wildcard, carbon];
        symbols.sort_unstable();
        assert_eq!(symbols, [wildcard, carbon, oxygen]);
    }
}
