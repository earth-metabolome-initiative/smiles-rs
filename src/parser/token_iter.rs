//! Submodule creating the `TokenIter` struct, which is an iterator over
//! the `Token`s found in a provided string.

use core::str::{FromStr, from_utf8};

use elements_rs::Element;

use crate::{
    atom::{
        Atom,
        atom_symbol::AtomSymbol,
        bracketed::{charge::Charge, chirality::Chirality},
    },
    bond::{Bond, ring_num::RingNum},
    errors::{SmilesError, SmilesErrorWithSpan},
    token::{Token, TokenWithSpan},
};

/// An iterator over the tokens found in a SMILES string.
pub(crate) struct TokenIter<'a> {
    /// Raw input bytes for the ASCII-heavy parsing fast path.
    bytes: &'a [u8],
    /// Current byte offset in the input.
    position: usize,
    /// Denotes whether currently inside brackets
    in_bracket: bool,
    /// The length of the input
    len: usize,
}

impl<'a> From<&'a str> for TokenIter<'a> {
    #[inline]
    fn from(s: &'a str) -> Self {
        TokenIter { bytes: s.as_bytes(), position: 0, in_bracket: false, len: s.len() }
    }
}

impl TokenIter<'_> {
    #[inline]
    fn parse_token(&mut self, current_byte: u8) -> Result<Token, SmilesError> {
        let token = match current_byte {
            b'.' => {
                if self.in_bracket {
                    return Err(SmilesError::NonBondInBracket);
                }
                Token::NonBond
            }
            b'[' => {
                if self.in_bracket {
                    return Err(SmilesError::UnexpectedLeftBracket);
                }
                self.in_bracket = true;
                let isotope_mass_number = if let Some(isotope) = try_fold_number::<u16, 3>(self) {
                    Some(isotope?)
                } else {
                    None
                };
                let (symbol, aromatic) = try_element(self)?;
                let chirality = try_chirality(self)?;
                let hydrogens = hydrogen_count(self)?;
                let charge = try_charge(self)?;
                let class = try_class(self)?;
                let atom = Atom::new_bracket(
                    symbol,
                    isotope_mass_number,
                    aromatic,
                    hydrogens,
                    charge,
                    class,
                    chirality,
                );
                if self.peek_byte() == Some(b']') {
                    self.in_bracket = false;
                    let _ = self.next_byte();
                    Token::Atom(atom)
                } else {
                    return Err(SmilesError::UnclosedBracket);
                }
            }
            c if c.is_ascii_alphabetic() || c == b'*' => {
                if self.in_bracket {
                    return Err(SmilesError::UnexpectedBracketedState);
                }
                let (symbol, aromatic) = if let Some(atom) = try_organic_subset_from_first(self, c)
                {
                    atom?
                } else {
                    let (symbol, aromatic) = try_element_from_first(self, c)?;
                    if !valid_unbracketed(symbol) {
                        return Err(SmilesError::InvalidUnbracketedAtom(symbol));
                    }
                    (symbol, aromatic)
                };
                Token::Atom(Atom::new_organic_subset(symbol, aromatic))
            }

            n if n.is_ascii_digit() || n == b'%' => {
                if n == b'%' {
                    if self.in_bracket {
                        return Err(SmilesError::UnexpectedPercent);
                    }

                    if let Some(num) = try_fold_number::<u8, 2>(self) {
                        let ring_num = RingNum::try_new(num?)?;
                        if ring_num.get() < 10 {
                            return Err(SmilesError::InvalidRingNumber);
                        }
                        Token::RingClosure(ring_num)
                    } else {
                        return Err(SmilesError::InvalidRingNumber);
                    }
                } else {
                    Token::RingClosure(RingNum::try_new(n - b'0')?)
                }
            }
            b'-' | b'=' | b'#' | b'$' | b':' | b'/' | b'\\' => {
                try_bond(current_byte, self.in_bracket)?
            }
            b'(' => {
                if self.in_bracket {
                    return Err(SmilesError::UnexpectedBracketedState);
                }
                Token::LeftParentheses
            }
            b')' => {
                if self.in_bracket {
                    return Err(SmilesError::UnexpectedBracketedState);
                }
                Token::RightParentheses
            }
            _ => return Err(SmilesError::UnexpectedCharacter(char::from(current_byte))),
        };
        Ok(token)
    }

    #[inline]
    fn current_end(&self) -> usize {
        self.position
    }

    #[inline]
    fn peek_byte(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    #[inline]
    fn next_byte(&mut self) -> Option<u8> {
        let byte = self.bytes.get(self.position).copied()?;
        self.position += 1;
        Some(byte)
    }
}

impl Iterator for TokenIter<'_> {
    type Item = Result<TokenWithSpan, SmilesErrorWithSpan>;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.position;
        let current_byte = self.next_byte()?;
        if !current_byte.is_ascii() {
            self.position = (start + utf8_char_width(current_byte)).min(self.len);
            return Some(Err(SmilesErrorWithSpan::new(
                SmilesError::UnexpectedUnicodeCharacter,
                start,
                self.position,
            )));
        }
        match self.parse_token(current_byte) {
            Ok(token) => {
                let end = self.current_end();
                Some(Ok(TokenWithSpan::new(token, start, end)))
            }
            Err(e) => {
                let mut end = self.current_end();
                if end <= start {
                    end = (start + 1).min(self.len);
                }
                Some(Err(SmilesErrorWithSpan::new(e, start, end)))
            }
        }
    }
}

#[inline]
const fn utf8_char_width(first_byte: u8) -> usize {
    match first_byte {
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 1,
    }
}

/// determines whether an aromatic is valid for a given bracketed or unbracketed
/// atom
///
/// # Parameters
/// - `bool` for the status of `in_bracket`
/// - the [`Element`] being passed
#[inline]
fn aromatic_from_element(in_bracket: bool, element: Element) -> Result<bool, SmilesError> {
    let allowed = if in_bracket {
        matches!(
            element,
            Element::B
                | Element::C
                | Element::N
                | Element::O
                | Element::P
                | Element::S
                | Element::Se
                | Element::As
                | Element::Te
                | Element::Si
        )
    } else {
        matches!(
            element,
            Element::B | Element::C | Element::N | Element::O | Element::S | Element::P
        )
    };
    if allowed { Ok(true) } else { Err(SmilesError::InvalidAromaticElement(element)) }
}

#[inline]
fn try_element(stream: &mut TokenIter<'_>) -> Result<(AtomSymbol, bool), SmilesError> {
    let first = stream.next_byte().ok_or(SmilesError::MissingElement)?;
    try_element_from_first(stream, first)
}

#[inline]
fn try_organic_subset_from_first(
    stream: &mut TokenIter<'_>,
    byte_1: u8,
) -> Option<Result<(AtomSymbol, bool), SmilesError>> {
    let element = match byte_1 {
        b'*' => return Some(Ok((AtomSymbol::WildCard, false))),
        b'B' => {
            if stream.peek_byte() == Some(b'r') {
                let _ = stream.next_byte();
                Element::Br
            } else {
                Element::B
            }
        }
        b'C' => {
            if stream.peek_byte() == Some(b'l') {
                let _ = stream.next_byte();
                Element::Cl
            } else {
                Element::C
            }
        }
        b'N' => Element::N,
        b'O' => Element::O,
        b'P' => Element::P,
        b'S' => Element::S,
        b'F' => Element::F,
        b'I' => Element::I,
        b'b' => return Some(Ok((AtomSymbol::Element(Element::B), true))),
        b'c' => return Some(Ok((AtomSymbol::Element(Element::C), true))),
        b'n' => return Some(Ok((AtomSymbol::Element(Element::N), true))),
        b'o' => return Some(Ok((AtomSymbol::Element(Element::O), true))),
        b'p' => return Some(Ok((AtomSymbol::Element(Element::P), true))),
        b's' => return Some(Ok((AtomSymbol::Element(Element::S), true))),
        _ => return None,
    };
    Some(Ok((AtomSymbol::Element(element), false)))
}

#[inline]
fn parse_ascii_element(bytes: &[u8]) -> Option<Element> {
    let symbol = from_utf8(bytes).ok()?;
    Element::from_str(symbol).ok()
}

#[inline]
fn try_element_from_first(
    stream: &mut TokenIter<'_>,
    byte_1: u8,
) -> Result<(AtomSymbol, bool), SmilesError> {
    if byte_1 == b'*' {
        return Ok((AtomSymbol::WildCard, false));
    }
    if !byte_1.is_ascii_alphabetic() {
        return Err(SmilesError::MissingElement);
    }

    let is_aromatic_candidate = byte_1.is_ascii_lowercase();

    if let Some(byte_2) = stream.peek_byte()
        && byte_2.is_ascii_alphabetic()
    {
        if is_aromatic_candidate && byte_2.is_ascii_lowercase() {
            let candidate = [byte_1.to_ascii_uppercase(), byte_2];
            if let Some(element) = parse_ascii_element(&candidate) {
                let _ = stream.next_byte();
                let aromatic = aromatic_from_element(stream.in_bracket, element)?;
                return Ok((AtomSymbol::Element(element), aromatic));
            }
        }
        if !is_aromatic_candidate && byte_2.is_ascii_lowercase() {
            let candidate = [byte_1, byte_2];
            if let Some(element) = parse_ascii_element(&candidate) {
                let _ = stream.next_byte();
                return Ok((AtomSymbol::Element(element), false));
            }
        }
    }

    let one = [if is_aromatic_candidate { byte_1.to_ascii_uppercase() } else { byte_1 }];
    if let Some(element) = parse_ascii_element(&one) {
        let aromatic = if is_aromatic_candidate {
            aromatic_from_element(stream.in_bracket, element)?
        } else {
            false
        };
        return Ok((AtomSymbol::Element(element), aromatic));
    }

    Err(SmilesError::InvalidElementName(char::from(byte_1)))
}

// B, C, N, O, P, S, F, Cl, Br, I,
#[inline]
fn valid_unbracketed(symbol: AtomSymbol) -> bool {
    match symbol {
        AtomSymbol::Element(element) => {
            matches!(
                element,
                Element::B
                    | Element::C
                    | Element::N
                    | Element::O
                    | Element::P
                    | Element::S
                    | Element::F
                    | Element::Cl
                    | Element::Br
                    | Element::I
            )
        }
        AtomSymbol::WildCard => true,
    }
}

#[inline]
fn try_chirality(stream: &mut TokenIter<'_>) -> Result<Option<Chirality>, SmilesError> {
    if stream.peek_byte() != Some(b'@') {
        return Ok(None);
    }
    let _ = stream.next_byte();
    let byte_2 = stream.peek_byte().ok_or(SmilesError::UnexpectedEndOfString)?;
    let chirality = match byte_2 {
        b'@' => {
            let _ = stream.next_byte();
            Chirality::AtAt
        }
        b'T' => {
            let _ = stream.next_byte();
            match stream.peek_byte().ok_or(SmilesError::UnexpectedEndOfString)? {
                b'H' => {
                    let _ = stream.next_byte();
                    let num =
                        try_fold_number::<u8, 1>(stream).ok_or(SmilesError::InvalidChirality)??;
                    Chirality::try_th(num)?
                }
                b'B' => {
                    let _ = stream.next_byte();
                    let num =
                        try_fold_number::<u8, 2>(stream).ok_or(SmilesError::InvalidChirality)??;
                    Chirality::try_tb(num)?
                }
                _ => return Err(SmilesError::InvalidChirality),
            }
        }
        b'A' => {
            let _ = stream.next_byte();
            match stream.peek_byte().ok_or(SmilesError::UnexpectedEndOfString)? {
                b'L' => {
                    let _ = stream.next_byte();
                    let num =
                        try_fold_number::<u8, 1>(stream).ok_or(SmilesError::InvalidChirality)??;
                    Chirality::try_al(num)?
                }
                _ => return Err(SmilesError::InvalidChirality),
            }
        }
        b'S' => {
            let _ = stream.next_byte();
            match stream.peek_byte().ok_or(SmilesError::UnexpectedEndOfString)? {
                b'P' => {
                    let _ = stream.next_byte();
                    let num =
                        try_fold_number::<u8, 1>(stream).ok_or(SmilesError::InvalidChirality)??;
                    Chirality::try_sp(num)?
                }
                _ => return Err(SmilesError::InvalidChirality),
            }
        }
        b'O' => {
            let _ = stream.next_byte();
            match stream.peek_byte().ok_or(SmilesError::UnexpectedEndOfString)? {
                b'H' => {
                    let _ = stream.next_byte();
                    let num =
                        try_fold_number::<u8, 2>(stream).ok_or(SmilesError::InvalidChirality)??;
                    Chirality::try_oh(num)?
                }
                _ => return Err(SmilesError::InvalidChirality),
            }
        }
        b'H' | b'-' | b'+' | b':' | b']' => Chirality::At,
        _ => return Err(SmilesError::InvalidChirality),
    };
    Ok(Some(chirality))
}

#[inline]
fn try_fold_number<B, const MAX_DIGITS: usize>(
    stream: &mut TokenIter<'_>,
) -> Option<Result<B, SmilesError>>
where
    B: TryFrom<u16>,
{
    let mut amount: u16 = 0;
    let mut digits_found = 0;

    let bytes = stream.bytes;
    let len = stream.len;
    let mut position = stream.position;

    while position < len && digits_found < MAX_DIGITS {
        let byte = bytes[position];
        if !byte.is_ascii_digit() {
            break;
        }
        digits_found += 1;
        position += 1;
        let digit = u16::from(byte - b'0');
        match amount.checked_mul(10).and_then(|x| x.checked_add(digit)) {
            Some(val) => amount = val,
            None => return Some(Err(SmilesError::IntegerOverflow)),
        }
    }

    if digits_found == 0 {
        return None;
    }

    stream.position = position;
    Some(B::try_from(amount).map_err(|_| SmilesError::IntegerOverflow))
}

/// Maximum bracket-atom explicit hydrogen count accepted by the parser.
///
/// Chosen to mirror the magnitude cap on bracket-atom charges. Real chemistry
/// SMILES do not require explicit hydrogen counts above this bound, and
/// bounding the value here keeps downstream `u8` valence math from overflowing.
const MAX_HYDROGEN_COUNT: u8 = 15;

#[inline]
fn hydrogen_count(stream: &mut TokenIter<'_>) -> Result<u8, SmilesError> {
    if stream.peek_byte() == Some(b'H') {
        let _ = stream.next_byte();
        let count = match try_fold_number::<u8, 3>(stream) {
            Some(h) => h?,
            None => 1,
        };
        if count > MAX_HYDROGEN_COUNT {
            return Err(SmilesError::HydrogenCountOverflow(count));
        }
        Ok(count)
    } else {
        Ok(0)
    }
}

#[inline]
fn try_charge(stream: &mut TokenIter<'_>) -> Result<Charge, SmilesError> {
    match stream.peek_byte() {
        Some(b'-') => {
            stream.position += 1;
            match stream.peek_byte() {
                Some(b'-') => repeated_sign_charge(stream, ChargeSign::Negative),
                _ => parse_signed_charge_magnitude(stream, ChargeSign::Negative),
            }
        }
        Some(b'+') => {
            stream.position += 1;
            match stream.peek_byte() {
                Some(b'+') => repeated_sign_charge(stream, ChargeSign::Positive),
                _ => parse_signed_charge_magnitude(stream, ChargeSign::Positive),
            }
        }
        Some(byte) if byte.is_ascii_digit() => {
            if let Some(possible_num) = try_fold_number::<i8, 2>(stream) {
                let magnitude = possible_num?;
                match stream.peek_byte() {
                    Some(b'-') => {
                        stream.position += 1;
                        Charge::try_new(-magnitude)
                    }
                    Some(b'+') => {
                        stream.position += 1;
                        Charge::try_new(magnitude)
                    }
                    _ => Err(SmilesError::UnexpectedCharacter(char::from(byte))),
                }
            } else {
                Err(SmilesError::UnexpectedCharacter(char::from(byte)))
            }
        }
        _ => Ok(Charge::default()),
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ChargeSign {
    Positive,
    Negative,
}

impl ChargeSign {
    #[inline]
    fn apply(self, magnitude: i8) -> i8 {
        match self {
            Self::Positive => magnitude,
            Self::Negative => -magnitude,
        }
    }

    #[inline]
    fn sign_byte(self) -> u8 {
        match self {
            Self::Positive => b'+',
            Self::Negative => b'-',
        }
    }

    #[inline]
    fn try_new_charge(self, magnitude: i8) -> Result<Charge, SmilesError> {
        Charge::try_new(self.apply(magnitude))
    }

    #[inline]
    fn overflow_error(self) -> SmilesError {
        match self {
            Self::Positive => SmilesError::ChargeOverflow(16),
            Self::Negative => SmilesError::ChargeUnderflow(-16),
        }
    }
}

#[inline]
fn parse_signed_charge_magnitude(
    stream: &mut TokenIter<'_>,
    sign: ChargeSign,
) -> Result<Charge, SmilesError> {
    if let Some(possible_num) = try_fold_number::<i8, 2>(stream) {
        sign.try_new_charge(possible_num?)
    } else {
        sign.try_new_charge(1)
    }
}

#[inline]
fn repeated_sign_charge(
    stream: &mut TokenIter<'_>,
    sign: ChargeSign,
) -> Result<Charge, SmilesError> {
    const MAX_ABSOLUTE_CHARGE: u16 = 15;

    let mut magnitude: u16 = 1;
    while stream.peek_byte() == Some(sign.sign_byte()) {
        stream.position += 1;
        magnitude = magnitude.saturating_add(1);
    }
    if magnitude > MAX_ABSOLUTE_CHARGE {
        Err(sign.overflow_error())
    } else {
        let magnitude = i8::try_from(magnitude)
            .unwrap_or_else(|_| unreachable!("charge magnitude is checked above"));
        sign.try_new_charge(magnitude)
    }
}

#[inline]
fn try_class(stream: &mut TokenIter<'_>) -> Result<u16, SmilesError> {
    match stream.peek_byte() {
        Some(b':') => {
            stream.position += 1;
            if let Some(possible_num) = try_fold_number::<u16, 3>(stream) {
                possible_num
            } else {
                Err(SmilesError::InvalidClass)
            }
        }
        _ => Ok(0),
    }
}

#[inline]
fn try_bond(byte: u8, bracket: bool) -> Result<Token, SmilesError> {
    let bond = match byte {
        b'-' => {
            if bracket {
                return Err(SmilesError::UnexpectedDash);
            }
            Token::Bond(Bond::Single.into())
        }
        b'=' => {
            if bracket {
                return Err(SmilesError::BondInBracket(Bond::Double));
            }
            Token::Bond(Bond::Double.into())
        }
        b'#' => {
            if bracket {
                return Err(SmilesError::BondInBracket(Bond::Triple));
            }
            Token::Bond(Bond::Triple.into())
        }
        b'$' => {
            if bracket {
                return Err(SmilesError::BondInBracket(Bond::Quadruple));
            }
            Token::Bond(Bond::Quadruple.into())
        }
        b':' => {
            if bracket {
                return Err(SmilesError::UnexpectedColon);
            }
            Token::Bond(crate::bond::BondDescriptor::aromatic(Bond::Single))
        }
        b'/' => {
            if bracket {
                return Err(SmilesError::BondInBracket(Bond::Up));
            }
            Token::Bond(Bond::Up.into())
        }
        b'\\' => {
            if bracket {
                return Err(SmilesError::BondInBracket(Bond::Down));
            }
            Token::Bond(Bond::Down.into())
        }
        _ => return Err(SmilesError::UnexpectedCharacter(char::from(byte))),
    };
    Ok(bond)
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec::Vec};

    use elements_rs::Element;

    use super::*;
    use crate::{
        atom::{
            atom_symbol::AtomSymbol,
            bracketed::{charge::Charge, chirality::Chirality},
        },
        bond::{Bond, ring_num::RingNum},
        errors::SmilesError,
        token::Token,
    };

    fn next_ok(input: &str) -> TokenWithSpan {
        TokenIter::from(input).next().expect("expected one token").expect("expected token ok")
    }

    fn next_err(input: &str) -> SmilesErrorWithSpan {
        TokenIter::from(input)
            .next()
            .expect("expected one token")
            .expect_err("expected token error")
    }

    #[test]
    fn parse_token_direct_bracket_state_errors() {
        let mut iter = TokenIter::from(".");
        iter.in_bracket = true;
        assert_eq!(iter.parse_token(b'.'), Err(SmilesError::NonBondInBracket));

        let mut iter = TokenIter::from("[");
        iter.in_bracket = true;
        assert_eq!(iter.parse_token(b'['), Err(SmilesError::UnexpectedLeftBracket));

        let mut iter = TokenIter::from("C");
        iter.in_bracket = true;
        assert_eq!(iter.parse_token(b'C'), Err(SmilesError::UnexpectedBracketedState));

        let mut iter = TokenIter::from("%");
        iter.in_bracket = true;
        assert_eq!(iter.parse_token(b'%'), Err(SmilesError::UnexpectedPercent));

        let mut iter = TokenIter::from("(");
        iter.in_bracket = true;
        assert_eq!(iter.parse_token(b'('), Err(SmilesError::UnexpectedBracketedState));

        let mut iter = TokenIter::from(")");
        iter.in_bracket = true;
        assert_eq!(iter.parse_token(b')'), Err(SmilesError::UnexpectedBracketedState));
    }

    #[test]
    fn parse_token_bracket_atom_covers_isotope_chirality_hydrogens_charge_and_class() {
        let token = next_ok("[13C@H2+2:12]");
        assert_eq!(token.start(), 0);
        assert_eq!(token.end(), "[13C@H2+2:12]".len());
        assert!(matches!(token.token(), Token::Atom(atom) if atom.is_bracket_atom()));
    }

    #[test]
    fn parse_token_unbracketed_atom_errors() {
        let err = next_err("Ac");
        assert_eq!(
            err.smiles_error(),
            SmilesError::InvalidUnbracketedAtom(AtomSymbol::Element(Element::Ac))
        );
        assert_eq!(err.start(), 0);
        assert_eq!(err.end(), 2);
        assert_eq!(err.span().start, 0);
        assert_eq!(err.span().end, 2);

        let err = next_err("q");
        assert_eq!(err.smiles_error(), SmilesError::InvalidElementName('q'));
        assert_eq!(err.start(), 0);
        assert_eq!(err.end(), 1);
        assert_eq!(err.span().start, 0);
        assert_eq!(err.span().end, 1);
    }

    #[test]
    fn parse_token_ring_number_errors() {
        let err = next_err("%");
        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
        assert_eq!(err.start(), 0);
        assert_eq!(err.end(), 1);
        assert_eq!(err.span().start, 0);
        assert_eq!(err.span().end, 1);

        let err = next_err("%9");
        assert_eq!(err.smiles_error(), SmilesError::InvalidRingNumber);
        assert_eq!(err.start(), 0);
        assert_eq!(err.end(), 2);
        assert_eq!(err.span().start, 0);
        assert_eq!(err.span().end, 2);
    }

    #[test]
    fn parse_token_single_digit_ring_closure_success() {
        let token = next_ok("1");
        assert_eq!(token.token(), Token::RingClosure(RingNum::try_new(1).unwrap()));
        assert_eq!(token.span(), 0..1);
    }

    #[test]
    fn aromatic_from_element_branches() {
        assert_eq!(aromatic_from_element(false, Element::C), Ok(true));
        assert_eq!(aromatic_from_element(true, Element::Se), Ok(true));
        assert_eq!(aromatic_from_element(true, Element::Te), Ok(true));
        assert_eq!(aromatic_from_element(true, Element::Si), Ok(true));

        assert_eq!(
            aromatic_from_element(false, Element::Si),
            Err(SmilesError::InvalidAromaticElement(Element::Si))
        );
        assert_eq!(
            aromatic_from_element(false, Element::Se),
            Err(SmilesError::InvalidAromaticElement(Element::Se))
        );
        assert_eq!(
            aromatic_from_element(false, Element::Te),
            Err(SmilesError::InvalidAromaticElement(Element::Te))
        );

        assert_eq!(
            aromatic_from_element(true, Element::Cl),
            Err(SmilesError::InvalidAromaticElement(Element::Cl))
        );
    }

    #[test]
    fn try_element_from_first_branches() {
        let mut stream = TokenIter::from("");
        assert_eq!(try_element_from_first(&mut stream, b'*'), Ok((AtomSymbol::WildCard, false)));

        let mut stream = TokenIter::from("");
        assert_eq!(
            try_element_from_first(&mut stream, b'c'),
            Ok((AtomSymbol::Element(Element::C), true))
        );

        let mut stream = TokenIter::from("l");
        assert_eq!(
            try_element_from_first(&mut stream, b'C'),
            Ok((AtomSymbol::Element(Element::Cl), false))
        );

        let mut stream = TokenIter::from("e");
        stream.in_bracket = true;
        assert_eq!(
            try_element_from_first(&mut stream, b's'),
            Ok((AtomSymbol::Element(Element::Se), true))
        );

        let mut stream = TokenIter::from("e");
        stream.in_bracket = true;
        assert_eq!(
            try_element_from_first(&mut stream, b't'),
            Ok((AtomSymbol::Element(Element::Te), true))
        );

        let mut stream = TokenIter::from("");
        assert_eq!(try_element_from_first(&mut stream, b'1'), Err(SmilesError::MissingElement));

        let mut stream = TokenIter::from("");
        assert_eq!(
            try_element_from_first(&mut stream, b'q'),
            Err(SmilesError::InvalidElementName('q'))
        );
    }

    #[test]
    fn valid_unbracketed_branches() {
        assert!(valid_unbracketed(AtomSymbol::Element(Element::C)));
        assert!(valid_unbracketed(AtomSymbol::Element(Element::Cl)));
        assert!(valid_unbracketed(AtomSymbol::WildCard));
        assert!(!valid_unbracketed(AtomSymbol::Element(Element::Ac)));
    }

    #[test]
    fn try_chirality_branches() {
        let mut stream = TokenIter::from("H");
        assert_eq!(try_chirality(&mut stream), Ok(None));

        let mut stream = TokenIter::from("@@");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::AtAt)));

        let mut stream = TokenIter::from("@]");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::At)));

        let mut stream = TokenIter::from("@H");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::At)));

        let mut stream = TokenIter::from("@+");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::At)));

        let mut stream = TokenIter::from("@AL1");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::try_al(1).unwrap())));

        let mut stream = TokenIter::from("@SP1");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::try_sp(1).unwrap())));

        let mut stream = TokenIter::from("@OH12");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::try_oh(12).unwrap())));

        let mut stream = TokenIter::from("@");
        assert_eq!(try_chirality(&mut stream), Err(SmilesError::UnexpectedEndOfString));

        let mut stream = TokenIter::from("@Z");
        assert_eq!(try_chirality(&mut stream), Err(SmilesError::InvalidChirality));

        let mut stream = TokenIter::from("@AQ");
        assert_eq!(try_chirality(&mut stream), Err(SmilesError::InvalidChirality));

        let mut stream = TokenIter::from("@AP1");
        assert_eq!(try_chirality(&mut stream), Err(SmilesError::InvalidChirality));

        let mut stream = TokenIter::from("@OQ");
        assert_eq!(try_chirality(&mut stream), Err(SmilesError::InvalidChirality));
    }

    #[test]
    fn try_fold_number_branches_including_overflow() {
        let mut stream = TokenIter::from("123x");
        assert_eq!(try_fold_number::<u16, 3>(&mut stream), Some(Ok(123)));

        let mut stream = TokenIter::from("x");
        assert_eq!(try_fold_number::<u16, 3>(&mut stream), None);

        let mut stream = TokenIter::from("700000");
        assert_eq!(try_fold_number::<u16, 6>(&mut stream), Some(Err(SmilesError::IntegerOverflow)));

        let mut stream = TokenIter::from("300");
        assert_eq!(try_fold_number::<u8, 3>(&mut stream), Some(Err(SmilesError::IntegerOverflow)));
    }

    #[test]
    fn hydrogen_count_branches() {
        let mut stream = TokenIter::from("H2");
        assert_eq!(hydrogen_count(&mut stream), Ok(2));

        let mut stream = TokenIter::from("H");
        assert_eq!(hydrogen_count(&mut stream), Ok(1));

        let mut stream = TokenIter::from("C");
        assert_eq!(hydrogen_count(&mut stream), Ok(0));

        let mut stream = TokenIter::from("H15");
        assert_eq!(hydrogen_count(&mut stream), Ok(15));

        let mut stream = TokenIter::from("H16");
        assert_eq!(hydrogen_count(&mut stream), Err(SmilesError::HydrogenCountOverflow(16)));

        let mut stream = TokenIter::from("H254");
        assert_eq!(hydrogen_count(&mut stream), Err(SmilesError::HydrogenCountOverflow(254)));
    }

    #[test]
    fn try_charge_branches() {
        let mut stream = TokenIter::from("-");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(-1).unwrap()));

        let mut stream = TokenIter::from("--");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(-2).unwrap()));

        let mut stream = TokenIter::from("-15");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(-15).unwrap()));

        let mut stream = TokenIter::from("-16");
        assert_eq!(try_charge(&mut stream), Err(SmilesError::ChargeUnderflow(-16)));

        let mut stream = TokenIter::from("+");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(1).unwrap()));

        let mut stream = TokenIter::from("++");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(2).unwrap()));

        let mut stream = TokenIter::from("+15");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(15).unwrap()));

        let mut stream = TokenIter::from("+16");
        assert_eq!(try_charge(&mut stream), Err(SmilesError::ChargeOverflow(16)));

        let mut stream = TokenIter::from("C");
        assert_eq!(try_charge(&mut stream), Ok(Charge::default()));
    }

    #[test]
    fn try_class_branches() {
        let mut stream = TokenIter::from(":12");
        assert_eq!(try_class(&mut stream), Ok(12));

        let mut stream = TokenIter::from(":");
        assert_eq!(try_class(&mut stream), Err(SmilesError::InvalidClass));

        let mut stream = TokenIter::from("C");
        assert_eq!(try_class(&mut stream), Ok(0));
    }

    #[test]
    fn try_bond_branches() {
        let ok_cases = [
            ('-', Token::Bond(Bond::Single.into())),
            ('=', Token::Bond(Bond::Double.into())),
            ('#', Token::Bond(Bond::Triple.into())),
            ('$', Token::Bond(Bond::Quadruple.into())),
            (':', Token::Bond(crate::bond::BondDescriptor::aromatic(Bond::Single))),
            ('/', Token::Bond(Bond::Up.into())),
            ('\\', Token::Bond(Bond::Down.into())),
        ];

        for (ch, expected) in ok_cases {
            assert_eq!(try_bond(ch as u8, false), Ok(expected));
        }

        assert_eq!(try_bond(b'-', true), Err(SmilesError::UnexpectedDash));
        assert_eq!(try_bond(b':', true), Err(SmilesError::UnexpectedColon));
        assert_eq!(try_bond(b'=', true), Err(SmilesError::BondInBracket(Bond::Double)));
        assert_eq!(try_bond(b'#', true), Err(SmilesError::BondInBracket(Bond::Triple)));
        assert_eq!(try_bond(b'$', true), Err(SmilesError::BondInBracket(Bond::Quadruple)));
        assert_eq!(try_bond(b'/', true), Err(SmilesError::BondInBracket(Bond::Up)));
        assert_eq!(try_bond(b'\\', true), Err(SmilesError::BondInBracket(Bond::Down)));
        assert_eq!(try_bond(b'x', false), Err(SmilesError::UnexpectedCharacter('x')));
    }

    #[test]
    fn iterator_error_span_mapping_smoke_test() {
        let err = next_err("Ac");
        assert_eq!(
            err.smiles_error(),
            SmilesError::InvalidUnbracketedAtom(AtomSymbol::Element(Element::Ac))
        );
        assert_eq!(err.start(), 0);
        assert_eq!(err.end(), 2);
    }

    #[test]
    fn try_chirality_th_form_should_parse() {
        let mut stream = TokenIter::from("@TH1");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::try_th(1).unwrap())));
    }

    #[test]
    fn try_chirality_al_form_should_parse() {
        let mut stream = TokenIter::from("@AL2");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::try_al(2).unwrap())));
    }

    #[test]
    fn try_chirality_tb_form_should_parse() {
        let mut stream = TokenIter::from("@TB10");
        assert_eq!(try_chirality(&mut stream), Ok(Some(Chirality::try_tb(10).unwrap())));
    }
    #[test]
    fn token_iter_parses_bracket_atom_with_th_chirality() {
        let token = next_ok("[C@TH1]");
        assert!(matches!(token.token(), Token::Atom(atom) if atom.is_bracket_atom()));
    }

    #[test]
    fn token_iter_parses_bracket_atom_with_al_chirality() {
        let token = next_ok("[C@AL1](F)=C=C");
        assert!(matches!(token.token(), Token::Atom(atom) if atom.is_bracket_atom()));
    }
    #[test]
    fn test_charge_parsing() {
        // Define test cases with expected charge parsing behavior
        let test_cases = vec![
            ("[Fe+++]", 3), // Fe with +3 charge
            ("[Fe-1]", -1), // Fe with -1 charge
            ("[Fe+]", 1),   // Fe with +1 charge
            ("[Fe2+]", 2),  // Fe with +2 charge
            ("[Fe-2]", -2), // Fe with -2 charge
        ];

        for (smiles, expected_charge) in test_cases {
            let tokens: Vec<Result<TokenWithSpan, SmilesErrorWithSpan>> =
                TokenIter::from(smiles).collect();
            for token in tokens {
                match token {
                    Ok(token_with_span) => {
                        let possible_atom = token_with_span.token();
                        match possible_atom {
                            Token::Atom(atom) => {
                                assert_eq!(atom.charge_value(), expected_charge);
                            }
                            _ => panic!("Token: {possible_atom:?} should be an atom token!"),
                        }
                    }
                    Err(e) => panic!("{e} in {smiles}"),
                }
            }
        }
    }
    #[test]
    fn try_charge_supports_number_before_sign() {
        let mut stream = TokenIter::from("2-");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(-2).unwrap()));

        let mut stream = TokenIter::from("2+");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(2).unwrap()));

        let mut stream = TokenIter::from("15-");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(-15).unwrap()));

        let mut stream = TokenIter::from("15+");
        assert_eq!(try_charge(&mut stream), Ok(Charge::try_new(15).unwrap()));
    }

    #[test]
    fn repeated_charge_runs_fail_cleanly_instead_of_overflowing() {
        let plus_err = next_err(&format!("[s{}P]", "+".repeat(200)));
        assert_eq!(plus_err.smiles_error(), SmilesError::ChargeOverflow(16));

        let minus_err = next_err(&format!("[s{}P]", "-".repeat(200)));
        assert_eq!(minus_err.smiles_error(), SmilesError::ChargeUnderflow(-16));
    }

    #[test]
    fn test_charge_parsing_number_before_sign() {
        let test_cases = vec![("[C2-]", -2), ("[C2+]", 2), ("[Fe15+]", 15), ("[Fe15-]", -15)];

        for (smiles, expected_charge) in test_cases {
            let tokens: Vec<_> = TokenIter::from(smiles).collect();
            for token in tokens {
                match token {
                    Ok(token_with_span) => {
                        match token_with_span.token() {
                            Token::Atom(atom) => {
                                assert_eq!(atom.charge_value(), expected_charge);
                            }
                            other => panic!("Token {other:?} should be an atom token!"),
                        }
                    }
                    Err(e) => panic!("{e} in {smiles}"),
                }
            }
        }
    }
}
