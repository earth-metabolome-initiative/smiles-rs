use core::str::FromStr;

use super::{Smiles, SmilesAtomPolicy, WildcardSmiles};
use crate::{
    errors::SmilesErrorWithSpan,
    parser::smiles_parser::{parse_smiles, parse_smiles_with_policy, parse_wildcard_smiles},
};

impl Smiles {
    /// Parses a strict [`Smiles`] graph from text.
    ///
    /// Wildcard (`*`) atoms are rejected. Parse
    /// [`WildcardSmiles`] when wildcard atoms are part
    /// of the expected input language.
    ///
    /// # Errors
    /// Returns a spanned parse error when tokenization or graph construction
    /// fails.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, SmilesErrorWithSpan> {
        parse_smiles(s)
    }
}

impl<AtomPolicy: SmilesAtomPolicy> FromStr for Smiles<AtomPolicy> {
    type Err = SmilesErrorWithSpan;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_smiles_with_policy(s)
    }
}

impl WildcardSmiles {
    /// Parses a wildcard-capable [`WildcardSmiles`] graph from text.
    ///
    /// # Errors
    /// Returns a spanned parse error when tokenization or graph construction
    /// fails.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, SmilesErrorWithSpan> {
        parse_wildcard_smiles(s).map(Self::from_inner)
    }
}

impl FromStr for WildcardSmiles {
    type Err = SmilesErrorWithSpan;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use elements_rs::Element;

    use crate::{
        atom::{Atom, atom_symbol::AtomSymbol},
        bond::{
            Bond,
            bond_edge::{bond_edge, bond_edge_ring_num_val},
            ring_num::RingNum,
        },
        smiles::{Smiles, WildcardSmiles},
    };

    #[test]
    fn parse_benzene_with_ring_nums() {
        let smiles = Smiles::from_str("C1=CC=CC=C1")
            .unwrap_or_else(|e| panic!("Failed to tokenize:\n{}", e.render("C1=CC=CC=C1")));

        let expected_nodes = [
            Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
            Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
            Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
            Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
            Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
            Atom::new_organic_subset(AtomSymbol::Element(Element::C), false),
        ];
        let expected_ring_edge = bond_edge(0, 5, Bond::Single, Some(RingNum::try_new(1).unwrap()));

        assert_eq!(smiles.nodes()[0], expected_nodes[0]);
        assert_eq!(smiles.nodes()[5], expected_nodes[5]);

        assert_eq!(smiles.edge_for_node_pair((0, 5)), Some(expected_ring_edge));
        assert_eq!(smiles.edge_for_node_pair((0, 5)).and_then(bond_edge_ring_num_val), Some(1));
        assert_eq!(smiles.node_by_id(0), Some(&expected_nodes[0]));
        assert_eq!(smiles.node_by_id(5), Some(&expected_nodes[5]));
    }

    #[test]
    fn from_str_propagates_token_iter_error() {
        let err = Smiles::from_str("Ac").expect_err("expected tokenization to fail");

        assert_eq!(
            err.smiles_error(),
            crate::errors::SmilesError::InvalidUnbracketedAtom(AtomSymbol::Element(Element::Ac))
        );
        assert_eq!(err.start(), 0);
        assert_eq!(err.end(), 2);
        assert_eq!(err.to_string(), "Invalid unbracketed atom: Ac at 0..2");
    }

    #[test]
    fn from_str_propagates_parser_error() {
        let err = Smiles::from_str("C(").expect_err("expected parsing to fail");

        assert_eq!(err.smiles_error(), crate::errors::SmilesError::UnclosedBranch);
        assert_eq!(err.start(), 1);
        assert_eq!(err.end(), 2);
        assert_eq!(err.to_string(), "Branch not closed at 1..2");
    }

    #[test]
    fn empty_input_is_not_a_valid_smiles() {
        let err = Smiles::from_str("").expect_err("empty input should not parse");
        assert_eq!(err.smiles_error(), crate::errors::SmilesError::MissingElement);
        assert_eq!((err.start(), err.end()), (0, 0));

        let err = WildcardSmiles::from_str("").expect_err("empty input should not parse");
        assert_eq!(err.smiles_error(), crate::errors::SmilesError::MissingElement);
        assert_eq!((err.start(), err.end()), (0, 0));
    }

    #[test]
    fn strict_smiles_rejects_wildcards() {
        for (source, span) in [
            ("*", (0, 1)),
            ("C(*)C", (2, 3)),
            ("[*]", (0, 3)),
            ("[13*]", (0, 5)),
            ("[*:1]", (0, 5)),
        ] {
            let err = Smiles::from_str(source).expect_err("strict SMILES should reject wildcard");
            assert_eq!(err.smiles_error(), crate::errors::SmilesError::WildcardAtomNotAllowed);
            assert_eq!((err.start(), err.end()), span);
        }
    }

    #[test]
    fn wildcard_smiles_accepts_wildcards() {
        for source in ["*", "C(*)C", "[*]", "[13*]", "[*:1]"] {
            WildcardSmiles::from_str(source)
                .unwrap_or_else(|error| panic!("failed to parse {source}: {error}"));
        }
    }

    #[test]
    fn concrete_isotopes_are_validated_while_parsing() {
        let err = Smiles::from_str("[999C]").expect_err("unknown carbon isotope should be invalid");

        assert_eq!(err.smiles_error(), crate::errors::SmilesError::InvalidIsotope);
        assert_eq!((err.start(), err.end()), (0, 6));

        let err = WildcardSmiles::from_str("[999C]")
            .expect_err("wildcard-capable parsing should still validate concrete isotopes");
        assert_eq!(err.smiles_error(), crate::errors::SmilesError::InvalidIsotope);
        assert_eq!((err.start(), err.end()), (0, 6));
    }
}
