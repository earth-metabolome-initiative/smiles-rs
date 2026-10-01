//! Defines errors used in the SMILES parser.

use alloc::{format, string::String};
use core::{fmt, num::TryFromIntError, ops::Range};

use elements_rs::Element;
use thiserror::Error;

use crate::{
    atom::atom_symbol::AtomSymbol,
    bond::{Bond, BondDescriptor},
};

/// The errors that could occur during SMILES parsing.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Error)]
pub enum SmilesError {
    /// Bond Inside Bracket
    #[error("Bond in bracket: {0}")]
    BondInBracket(Bond),
    /// A charge is over the allowed maximum (15)
    #[error("Charge overflow: {0}")]
    ChargeOverflow(i8),
    /// A charge is below allowed minimum (-15)
    #[error("Charge underflow: {0}")]
    ChargeUnderflow(i8),
    /// A duplicate edge between two nodes has been found
    #[error("Node A: {0} has multiple edges with Node B: {1}")]
    DuplicateEdge(usize, usize),
    /// A non bare element found outside of brackets
    #[error("Element requires brackets")]
    ElementRequiresBrackets,
    /// Wrapper for `element_rs` errors
    #[error("Error Parsing Element: {0}")]
    ElementsRs(#[from] elements_rs::errors::Error),
    /// A branch without any nodes has been parsed
    #[error("A branch without any nodes has been found")]
    EmptyBranch,
    /// A bond was not able to bind two atoms
    #[error("Bond: {0} missing atom index(es)")]
    IncompleteBond(BondDescriptor),
    /// Element forbidden to be written as aromatic here
    #[error("Invalid aromatic element: {0}")]
    InvalidAromaticElement(Element),
    /// A bond is not in a valid position, such as outside of a branch start or
    /// next to another bond
    #[error("A bond has been found in a non valid location")]
    InvalidBond,
    /// A branch is invalid, missing an atom
    #[error("An invalid branch has been found")]
    InvalidBranch,
    /// Specified Chirality is not a valid form
    #[error("Invalid chirality")]
    InvalidChirality,
    /// The class is not valid
    #[error("Invalid class")]
    InvalidClass,
    /// Error indicating invalid Element name
    #[error("Invalid element name: {0}")]
    InvalidElementName(char),
    /// A bracket atom's explicit hydrogen count exceeds the maximum supported
    /// value (15).
    ///
    /// The cap mirrors the magnitude cap on bracket-atom charges and prevents
    /// downstream `u8` valence math (`explicit_valence + hydrogen_count +
    /// implicit_hydrogen_count`) from overflowing for adversarial inputs.
    /// Real chemistry SMILES never need an explicit `H` count above this bound.
    #[error("Hydrogen count overflow: {0}")]
    HydrogenCountOverflow(u8),
    /// A hydrogen bracket atom has an unsupported explicit hydrogen count
    #[error("Hydrogen found as bracketed atom with an unsupported explicit hydrogen count")]
    InvalidHydrogenWithExplicitHydrogensFound,
    /// Invalid Isotope value passed
    #[error("Invalid isotope")]
    InvalidIsotope,
    /// Invalid `Token::NonBond`
    #[error("Invalid Non-bond '.' found")]
    InvalidNonBondToken,
    /// Error indicating that an invalid number was encountered.
    #[error("Invalid number")]
    InvalidNumber,
    /// Integer Overflow
    #[error("Integer overflow")]
    IntegerOverflow,
    /// Non organic element found out of bracket
    #[error("Invalid unbracketed atom: {0}")]
    InvalidUnbracketedAtom(AtomSymbol),
    /// An invalid ring number has been found
    #[error("Invalid ring number")]
    InvalidRingNumber,
    /// found `[..]` that did not contain an element
    #[error("Missing element inside brackets")]
    MissingBracketElement,
    /// Missing Element
    #[error("Missing element")]
    MissingElement,
    /// Node id does not point to a valid atom in the graph
    #[error("Invalid atom index: {0}")]
    NodeIdInvalid(usize),
    /// Non Bond in Bracket
    #[error("Non-bond '.' in bracket")]
    NonBondInBracket,
    /// Ring Number Overflow (greater than 99)
    #[error("Ring number overflow: {0}")]
    RingNumberOverflow(u8),
    /// An edge connects a node to itself
    #[error("Node: {0} has an edge that goes from itself and to itself")]
    SelfLoopEdge(usize),
    /// Unexpectedly inside of brackets
    #[error("Unexpected bracketed state")]
    UnexpectedBracketedState,
    /// Unexpected end of string
    #[error("Unexpected end of string")]
    UnexpectedEndOfString,
    /// Error indicating that an unexpected character was encountered.
    #[error("Unexpected character: {0}")]
    UnexpectedCharacter(char),
    /// Error indicating that unexpected unicode input was encountered.
    #[error("Unexpected unicode character")]
    UnexpectedUnicodeCharacter,
    /// An unexpected `:` has been found
    #[error("Unexpected ':'")]
    UnexpectedColon,
    /// An unexpected `-` has been found
    #[error("Unexpected '-'")]
    UnexpectedDash,
    /// An unexpected `%` has been found
    #[error("Unexpected '%'")]
    UnexpectedPercent,
    /// An unexpected left bracket `[` was found
    #[error("Unexpected '['")]
    UnexpectedLeftBracket,
    /// A `(` that wasn't expected has been found
    #[error("Unexpected '('")]
    UnexpectedLeftParentheses,
    /// An unexpected right bracket `]` was found
    #[error("Unexpected ']'")]
    UnexpectedRightBracket,
    /// An unexpected right parentheses `)` was found
    #[error("Unexpected `)`")]
    UnexpectedRightParentheses,
    /// A wildcard atom was parsed where only concrete atoms are allowed.
    #[error("Wildcard atom not allowed")]
    WildcardAtomNotAllowed,
    /// A closing `]` bracket was not found
    #[error("Unclosed '['")]
    UnclosedBracket,
    /// A branch has not been closed with a `)`
    #[error("Branch not closed")]
    UnclosedBranch,
    /// A ring number has been found that was not completed
    #[error("Ring not closed")]
    UnclosedRing,
}

impl From<TryFromIntError> for SmilesError {
    fn from(_: TryFromIntError) -> Self {
        SmilesError::InvalidNumber
    }
}

/// Wraps the `Smiles` error adding the location of where the error was found
#[derive(Debug)]
pub struct SmilesErrorWithSpan {
    /// The [`SmilesError`]
    smiles_error: SmilesError,
    /// The span as `usize`
    span: Range<usize>,
}

impl SmilesErrorWithSpan {
    /// Creates a new error from the [`SmilesError`] and the `span`
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{SmilesError, SmilesErrorWithSpan};
    ///
    /// let err = SmilesErrorWithSpan::new(SmilesError::UnexpectedEndOfString, 2, 3);
    /// assert_eq!(err.start(), 2);
    /// assert_eq!(err.end(), 3);
    /// ```
    #[must_use]
    pub fn new(smiles_error: SmilesError, start: usize, end: usize) -> Self {
        Self { smiles_error, span: Range { start, end } }
    }

    /// Returns the [`SmilesError`]
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{SmilesError, SmilesErrorWithSpan};
    ///
    /// let err = SmilesErrorWithSpan::new(SmilesError::InvalidNumber, 0, 1);
    /// assert_eq!(err.smiles_error(), SmilesError::InvalidNumber);
    /// ```
    #[must_use]
    pub fn smiles_error(&self) -> SmilesError {
        self.smiles_error
    }

    /// Returns the start of the span
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{SmilesError, SmilesErrorWithSpan};
    ///
    /// let err = SmilesErrorWithSpan::new(SmilesError::InvalidClass, 4, 6);
    /// assert_eq!(err.start(), 4);
    /// ```
    #[must_use]
    pub fn start(&self) -> usize {
        self.span.start
    }

    /// Returns the end of the span
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{SmilesError, SmilesErrorWithSpan};
    ///
    /// let err = SmilesErrorWithSpan::new(SmilesError::InvalidClass, 4, 6);
    /// assert_eq!(err.end(), 6);
    /// ```
    #[must_use]
    pub fn end(&self) -> usize {
        self.span.end
    }

    /// Returns the full span for the error
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{SmilesError, SmilesErrorWithSpan};
    ///
    /// let err = SmilesErrorWithSpan::new(SmilesError::UnexpectedLeftBracket, 1, 2);
    /// assert_eq!(err.span(), 1..2);
    /// ```
    #[must_use]
    pub fn span(&self) -> Range<usize> {
        self.span.start..self.span.end
    }

    /// Render the error pointing back to location in the original string
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::{SmilesError, SmilesErrorWithSpan};
    ///
    /// let err = SmilesErrorWithSpan::new(SmilesError::UnexpectedRightBracket, 1, 2);
    /// let rendered = err.render("C]");
    ///
    /// assert!(rendered.contains("C]"));
    /// assert!(rendered.contains("^"));
    /// ```
    #[must_use]
    pub fn render(&self, input: &str) -> String {
        let start = self.start().min(input.len());
        let end = self.end().min(input.len()).max(start + 1).min(input.len());

        let mut underline = String::new();
        underline.push_str(&" ".repeat(start));
        underline.push_str(&"^".repeat(end - start));

        format!("{input}\n{underline}\n{}", self.smiles_error)
    }
}

impl fmt::Display for SmilesErrorWithSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}..{}", self.smiles_error, self.start(), self.end())
    }
}

/// Error returned when carving a [`Fragment`](crate::smiles::Fragment) out of a
/// parent graph fails.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum SubgraphError {
    /// An atom id in the input is not a valid atom of the parent graph.
    #[error("Atom id {0} is out of range for the parent graph")]
    AtomOutOfRange(usize),
    /// A bond in the input references an atom that is not part of the fragment.
    #[error("Bond references atom {0}, which is not in the fragment")]
    BondReferencesUnknownAtom(usize),
}

/// Error returned when rendering a fragment anchored at a chosen parent atom.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum RootError {
    /// The requested root atom is not part of the fragment.
    #[error("Atom {0} is not in the fragment")]
    AtomNotInFragment(usize),
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use std::num::TryFromIntError;

    use elements_rs::Element;

    use crate::{
        atom::atom_symbol::AtomSymbol,
        bond::{Bond, BondDescriptor},
        errors::{SmilesError, SmilesErrorWithSpan},
    };

    #[test]
    fn test_smiles_error_display_monolithic() {
        let elements_rs_error = elements_rs::errors::Error::AtomicNumber(4);

        let cases = [
            (
                SmilesError::BondInBracket(Bond::Double),
                format!("Bond in bracket: {}", Bond::Double),
            ),
            (SmilesError::ChargeOverflow(50), "Charge overflow: 50".to_string()),
            (SmilesError::ChargeUnderflow(-50), "Charge underflow: -50".to_string()),
            (SmilesError::ElementRequiresBrackets, "Element requires brackets".to_string()),
            (
                SmilesError::ElementsRs(elements_rs_error),
                format!("Error Parsing Element: {elements_rs_error}"),
            ),
            (
                SmilesError::IncompleteBond(BondDescriptor::aromatic(Bond::Single)),
                "Bond: : missing atom index(es)".to_string(),
            ),
            (SmilesError::HydrogenCountOverflow(16), "Hydrogen count overflow: 16".to_string()),
            (
                SmilesError::InvalidAromaticElement(Element::Ac),
                format!("Invalid aromatic element: {}", Element::Ac),
            ),
            (SmilesError::InvalidChirality, "Invalid chirality".to_string()),
            (SmilesError::InvalidClass, "Invalid class".to_string()),
            (SmilesError::InvalidElementName('w'), "Invalid element name: w".to_string()),
            (SmilesError::InvalidIsotope, "Invalid isotope".to_string()),
            (SmilesError::InvalidNonBondToken, "Invalid Non-bond '.' found".to_string()),
            (SmilesError::InvalidNumber, "Invalid number".to_string()),
            (SmilesError::IntegerOverflow, "Integer overflow".to_string()),
            (
                SmilesError::InvalidUnbracketedAtom(AtomSymbol::WildCard),
                format!("Invalid unbracketed atom: {}", AtomSymbol::WildCard),
            ),
            (SmilesError::InvalidRingNumber, "Invalid ring number".to_string()),
            (SmilesError::MissingBracketElement, "Missing element inside brackets".to_string()),
            (SmilesError::MissingElement, "Missing element".to_string()),
            (SmilesError::NodeIdInvalid(2), "Invalid atom index: 2".to_string()),
            (SmilesError::NonBondInBracket, "Non-bond '.' in bracket".to_string()),
            (SmilesError::RingNumberOverflow(100), "Ring number overflow: 100".to_string()),
            (SmilesError::UnexpectedBracketedState, "Unexpected bracketed state".to_string()),
            (SmilesError::UnexpectedEndOfString, "Unexpected end of string".to_string()),
            (SmilesError::UnexpectedCharacter('$'), "Unexpected character: $".to_string()),
            (SmilesError::UnexpectedUnicodeCharacter, "Unexpected unicode character".to_string()),
            (SmilesError::UnexpectedColon, "Unexpected ':'".to_string()),
            (SmilesError::UnexpectedDash, "Unexpected '-'".to_string()),
            (SmilesError::UnexpectedPercent, "Unexpected '%'".to_string()),
            (SmilesError::UnexpectedLeftBracket, "Unexpected '['".to_string()),
            (SmilesError::UnexpectedLeftParentheses, "Unexpected '('".to_string()),
            (SmilesError::UnexpectedRightBracket, "Unexpected ']'".to_string()),
            (SmilesError::UnexpectedRightParentheses, "Unexpected `)`".to_string()),
            (SmilesError::WildcardAtomNotAllowed, "Wildcard atom not allowed".to_string()),
            (SmilesError::UnclosedBracket, "Unclosed '['".to_string()),
            (SmilesError::UnclosedBranch, "Branch not closed".to_string()),
            (SmilesError::UnclosedRing, "Ring not closed".to_string()),
            (
                SmilesError::SelfLoopEdge(1),
                "Node: 1 has an edge that goes from itself and to itself".to_string(),
            ),
            (
                SmilesError::DuplicateEdge(0, 1),
                "Node A: 0 has multiple edges with Node B: 1".to_string(),
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected, "wrong Display for {error:?}");
        }
    }

    #[test]
    fn test_smiles_error_from_conversions() {
        let int_error: Result<u8, TryFromIntError> = u8::try_from(300_i16);
        let int_error = int_error.unwrap_err();

        let smiles_error: SmilesError = int_error.into();
        assert_eq!(smiles_error, SmilesError::InvalidNumber);

        let elements_error = elements_rs::errors::Error::AtomicNumber(4);
        let smiles_error: SmilesError = elements_error.into();
        assert_eq!(
            smiles_error,
            SmilesError::ElementsRs(elements_rs::errors::Error::AtomicNumber(4))
        );
    }

    #[test]
    fn test_smiles_error_with_span() {
        let error = SmilesErrorWithSpan::new(SmilesError::UnexpectedCharacter('$'), 2, 3);

        assert_eq!(error.smiles_error(), SmilesError::UnexpectedCharacter('$'));
        assert_eq!(error.start(), 2);
        assert_eq!(error.end(), 3);
        assert_eq!(error.span(), (2..3));

        assert_eq!(error.to_string(), "Unexpected character: $ at 2..3");

        assert_eq!(error.render("CC$O"), "CC$O\n  ^\nUnexpected character: $");
    }

    #[test]
    fn render_marks_zero_width_and_multi_character_spans() {
        // A zero-width span (start == end) must still produce exactly one
        // caret, which only holds because the end is raised to at least
        // start + 1.
        let zero_width = SmilesErrorWithSpan::new(SmilesError::UnexpectedEndOfString, 2, 2);
        assert_eq!(zero_width.render("CCO"), "CCO\n  ^\nUnexpected end of string");

        // A multi-character span draws one caret per covered position, so the
        // caret run width is end minus start rather than any other combination.
        let two_wide = SmilesErrorWithSpan::new(SmilesError::UnexpectedCharacter('x'), 1, 3);
        assert_eq!(two_wide.render("CCCC"), "CCCC\n ^^\nUnexpected character: x");
    }

    #[test]
    fn test_smiles_error_with_unicode_span() {
        let error = SmilesErrorWithSpan::new(SmilesError::UnexpectedUnicodeCharacter, 2, 4);

        assert_eq!(error.smiles_error(), SmilesError::UnexpectedUnicodeCharacter);
        assert_eq!(error.start(), 2);
        assert_eq!(error.end(), 4);
        assert_eq!(error.span(), (2..4));

        assert_eq!(error.to_string(), "Unexpected unicode character at 2..4");
    }
}
