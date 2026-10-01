//! Represents tokens used in parsing SMILES strings.

use core::ops::Range;

use crate::{
    atom::Atom,
    bond::{BondDescriptor, ring_num::RingNum},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
/// Lightweight token category used when only syntax class matters.
pub enum TokenKind {
    /// `.` separator
    NonBond,
    /// Atom token
    Atom,
    /// Explicit bond token
    Bond,
    /// `(`
    LeftParentheses,
    /// `)`
    RightParentheses,
    /// Ring closure token
    RingClosure,
}

#[derive(Copy, Debug, PartialEq, Clone, Eq, Hash)]
/// Represents a token in a molecular formula.
pub enum Token {
    /// Represented with a `.`
    NonBond,
    /// Parsed atom token, preserving whether it came from bracket or
    /// organic-subset syntax inside the stored [`Atom`] value.
    Atom(Atom),
    /// The parsed bond. Single bonds are implicit between atoms if not
    /// explicated as a bond, e.g. `CC` would be `C-C`.
    Bond(BondDescriptor),
    /// Token for left parentheses, used for branching `(`
    LeftParentheses,
    /// Token for right parentheses, used for ending a branch `)`
    RightParentheses,
    /// Ring number markers occur outside of `[]` and may be of type `%` and
    /// `0-99`, and `%` may be omitted.
    RingClosure(RingNum),
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
/// A parsed token and its relative location in the string
pub struct TokenWithSpan {
    /// The parsed token
    token: Token,
    /// The starting to ending points of the token in the string
    span: Range<usize>,
}

impl TokenWithSpan {
    /// Generates a new token with a specified span
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::Bond,
    ///     token::{Token, TokenWithSpan},
    /// };
    ///
    /// let token = TokenWithSpan::new(Token::Bond(Bond::Double.into()), 1, 2);
    /// assert_eq!(token.token(), Token::Bond(Bond::Double.into()));
    /// ```
    #[must_use]
    pub fn new(token: Token, start: usize, end: usize) -> Self {
        Self { token, span: start..end }
    }
    /// Returns the token
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::Bond,
    ///     token::{Token, TokenWithSpan},
    /// };
    ///
    /// let token = TokenWithSpan::new(Token::Bond(Bond::Single.into()), 0, 1);
    /// assert_eq!(token.token(), Token::Bond(Bond::Single.into()));
    /// ```
    #[must_use]
    pub fn token(&self) -> Token {
        self.token
    }
    /// Returns the token kind without the payload.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::token::{Token, TokenKind, TokenWithSpan};
    ///
    /// let token = TokenWithSpan::new(Token::LeftParentheses, 2, 3);
    /// assert_eq!(token.token_kind(), TokenKind::LeftParentheses);
    /// ```
    #[must_use]
    pub fn token_kind(&self) -> TokenKind {
        self.token.kind()
    }
    /// Returns the span as [`Range`]
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::token::{Token, TokenWithSpan};
    ///
    /// let token = TokenWithSpan::new(Token::RightParentheses, 4, 5);
    /// assert_eq!(token.span(), 4..5);
    /// ```
    #[must_use]
    pub fn span(&self) -> Range<usize> {
        self.span.start..self.span.end
    }
    /// Returns the start of the span as [`usize`]
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::token::{Token, TokenWithSpan};
    ///
    /// let token = TokenWithSpan::new(Token::NonBond, 6, 7);
    /// assert_eq!(token.start(), 6);
    /// ```
    #[must_use]
    pub fn start(&self) -> usize {
        self.span.start
    }
    /// Returns the end of the span as [`usize`]
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::token::{Token, TokenWithSpan};
    ///
    /// let token = TokenWithSpan::new(Token::NonBond, 6, 7);
    /// assert_eq!(token.end(), 7);
    /// ```
    #[must_use]
    pub fn end(&self) -> usize {
        self.span.end
    }
    /// Returns whether the token is a bond
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::Bond,
    ///     token::{Token, TokenWithSpan},
    /// };
    ///
    /// let bond = TokenWithSpan::new(Token::Bond(Bond::Triple.into()), 1, 2);
    /// let atom = TokenWithSpan::new(Token::NonBond, 3, 4);
    ///
    /// assert!(bond.is_bond());
    /// assert!(!atom.is_bond());
    /// ```
    #[must_use]
    pub fn is_bond(&self) -> bool {
        self.token_kind() == TokenKind::Bond
    }
}

impl Token {
    /// Returns the payload-free category of the token.
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{
    ///     bond::Bond,
    ///     token::{Token, TokenKind},
    /// };
    ///
    /// assert_eq!(Token::Bond(Bond::Single.into()).kind(), TokenKind::Bond);
    /// assert_eq!(Token::NonBond.kind(), TokenKind::NonBond);
    /// ```
    #[inline]
    #[must_use]
    pub fn kind(self) -> TokenKind {
        match self {
            Self::NonBond => TokenKind::NonBond,
            Self::Atom(_) => TokenKind::Atom,
            Self::Bond(_) => TokenKind::Bond,
            Self::LeftParentheses => TokenKind::LeftParentheses,
            Self::RightParentheses => TokenKind::RightParentheses,
            Self::RingClosure(_) => TokenKind::RingClosure,
        }
    }
}

#[cfg(test)]
mod tests {
    use elements_rs::Element;

    use super::{Token, TokenKind, TokenWithSpan};
    use crate::{
        atom::{Atom, atom_symbol::AtomSymbol},
        bond::{Bond, ring_num::RingNum},
    };

    #[test]
    fn token_variants_can_be_constructed_and_compared() {
        let bracket_atom = Atom::builder().with_symbol(AtomSymbol::Element(Element::C)).build();
        let organic_atom = Atom::new_organic_subset(AtomSymbol::Element(Element::O), false);
        let ring_num = RingNum::try_new(12).unwrap();

        let cases = [
            Token::NonBond,
            Token::Atom(bracket_atom),
            Token::Atom(organic_atom),
            Token::Bond(Bond::Double.into()),
            Token::LeftParentheses,
            Token::RightParentheses,
            Token::RingClosure(ring_num),
        ];

        assert_eq!(cases[0], Token::NonBond);
        assert_eq!(cases[1], Token::Atom(bracket_atom));
        assert_eq!(cases[2], Token::Atom(organic_atom));
        assert_eq!(cases[3], Token::Bond(Bond::Double.into()));
        assert_eq!(cases[4], Token::LeftParentheses);
        assert_eq!(cases[5], Token::RightParentheses);
        assert_eq!(cases[6], Token::RingClosure(ring_num));
    }

    #[test]
    fn token_with_span_new_and_accessors_work() {
        let token = Token::Bond(Bond::Triple.into());
        let token_with_span = TokenWithSpan::new(token, 3, 7);

        assert_eq!(token_with_span.token(), Token::Bond(Bond::Triple.into()));
        assert_eq!(token_with_span.span(), 3..7);
        assert_eq!(token_with_span.start(), 3);
        assert_eq!(token_with_span.end(), 7);
    }

    #[test]
    fn token_with_span_preserves_complex_token_variants() {
        let bracket_atom = Atom::builder().with_symbol(AtomSymbol::Element(Element::N)).build();
        let ring_num = RingNum::try_new(9).unwrap();

        let bracketed = TokenWithSpan::new(Token::Atom(bracket_atom), 0, 3);
        let ring = TokenWithSpan::new(Token::RingClosure(ring_num), 5, 6);
        let non_bond = TokenWithSpan::new(Token::NonBond, 10, 11);

        assert_eq!(bracketed.token(), Token::Atom(bracket_atom));
        assert_eq!(bracketed.span(), 0..3);

        assert_eq!(ring.token(), Token::RingClosure(ring_num));
        assert_eq!(ring.start(), 5);
        assert_eq!(ring.end(), 6);

        assert_eq!(non_bond.token(), Token::NonBond);
        assert_eq!(non_bond.span(), 10..11);
    }

    #[test]
    fn token_with_span_clone_and_eq_behave_as_expected() {
        let original = TokenWithSpan::new(Token::LeftParentheses, 2, 3);
        let cloned = original.clone();

        assert_eq!(original, cloned);
        assert_eq!(original.token(), Token::LeftParentheses);
        assert_eq!(cloned.span(), 2..3);
    }

    #[test]
    fn token_kind_and_is_bond_helpers_cover_all_paths() {
        let bond = TokenWithSpan::new(Token::Bond(Bond::Triple.into()), 1, 2);
        assert_eq!(bond.token_kind(), TokenKind::Bond);
        assert!(bond.is_bond());

        let atom = TokenWithSpan::new(
            Token::Atom(Atom::new_organic_subset(AtomSymbol::Element(Element::C), false)),
            0,
            1,
        );
        assert_eq!(atom.token_kind(), TokenKind::Atom);
        assert!(!atom.is_bond());

        assert_eq!(Token::NonBond.kind(), TokenKind::NonBond);
        assert_eq!(
            Token::Atom(Atom::new_organic_subset(AtomSymbol::Element(Element::N), false)).kind(),
            TokenKind::Atom
        );
        assert_eq!(Token::Bond(Bond::Double.into()).kind(), TokenKind::Bond);
        assert_eq!(Token::LeftParentheses.kind(), TokenKind::LeftParentheses);
        assert_eq!(Token::RightParentheses.kind(), TokenKind::RightParentheses);
        assert_eq!(Token::RingClosure(RingNum::try_new(1).unwrap()).kind(), TokenKind::RingClosure);
    }
}
