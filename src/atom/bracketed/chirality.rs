//! Module for parsing, validating, and specifying the chirality of an atom
use core::fmt;

use crate::errors::SmilesError;

#[derive(Copy, Debug, PartialEq, Clone, Eq, Hash)]
/// Specifies the chirality if present
pub enum Chirality {
    /// `@`
    At,
    /// `@@`
    AtAt,
    /// `@TH` variants (1-2)
    TH(u8),
    /// `@AL` variants (1-2)
    AL(u8),
    /// `@SP` variants (1-3)
    SP(u8),
    /// `@TB` variants (1-20)
    TB(u8),
    /// `@OH` variants (1-30)
    OH(u8),
}

impl Chirality {
    /// Convert `u8` to `TH`+`U8`
    ///
    /// # Errors
    /// - Will return [`SmilesError::InvalidChirality`] if attempt to parse
    ///   fails
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::bracketed::chirality::Chirality;
    ///
    /// assert_eq!(Chirality::try_th(1)?, Chirality::TH(1));
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    pub fn try_th(num: u8) -> Result<Self, SmilesError> {
        (1..=2).contains(&num).then_some(Self::TH(num)).ok_or(SmilesError::InvalidChirality)
    }
    /// Convert `u8` to `AL`+`U8`
    ///
    /// # Errors
    /// - Will return [`SmilesError::InvalidChirality`] if attempt to parse
    ///   fails
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::bracketed::chirality::Chirality;
    ///
    /// assert_eq!(Chirality::try_al(2)?, Chirality::AL(2));
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    pub fn try_al(num: u8) -> Result<Self, SmilesError> {
        (1..=2).contains(&num).then_some(Self::AL(num)).ok_or(SmilesError::InvalidChirality)
    }
    /// Convert `u8` to `SP`+`U8`
    ///
    /// # Errors
    /// - Will return [`SmilesError::InvalidChirality`] if attempt to parse
    ///   fails
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::bracketed::chirality::Chirality;
    ///
    /// assert_eq!(Chirality::try_sp(3)?, Chirality::SP(3));
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    pub fn try_sp(num: u8) -> Result<Self, SmilesError> {
        (1..=3).contains(&num).then_some(Self::SP(num)).ok_or(SmilesError::InvalidChirality)
    }
    /// Convert `u8` to `TB`+`U8`
    ///
    /// # Errors
    /// - Will return [`SmilesError::InvalidChirality`] if attempt to parse
    ///   fails
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::bracketed::chirality::Chirality;
    ///
    /// assert_eq!(Chirality::try_tb(20)?, Chirality::TB(20));
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    pub fn try_tb(num: u8) -> Result<Self, SmilesError> {
        (1..=20).contains(&num).then_some(Self::TB(num)).ok_or(SmilesError::InvalidChirality)
    }
    /// Convert `u8` to `OH`+`U8`
    ///
    /// # Errors
    /// - Will return [`SmilesError::InvalidChirality`] if attempt to parse
    ///   fails
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::bracketed::chirality::Chirality;
    ///
    /// assert_eq!(Chirality::try_oh(30)?, Chirality::OH(30));
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    pub fn try_oh(num: u8) -> Result<Self, SmilesError> {
        (1..=30).contains(&num).then_some(Self::OH(num)).ok_or(SmilesError::InvalidChirality)
    }

    #[inline]
    #[must_use]
    pub(crate) const fn display_len(self) -> usize {
        match self {
            Self::At => 1,
            Self::AtAt => 2,
            Self::TH(n) | Self::AL(n) | Self::SP(n) | Self::TB(n) | Self::OH(n) => {
                3 + decimal_len_u8(n)
            }
        }
    }
}

impl fmt::Display for Chirality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::At => f.write_str("@"),
            Self::AtAt => f.write_str("@@"),
            Self::TH(n) => write!(f, "@TH{n}"),
            Self::AL(n) => write!(f, "@AL{n}"),
            Self::SP(n) => write!(f, "@SP{n}"),
            Self::TB(n) => write!(f, "@TB{n}"),
            Self::OH(n) => write!(f, "@OH{n}"),
        }
    }
}

const fn decimal_len_u8(value: u8) -> usize {
    if value >= 100 {
        3
    } else if value >= 10 {
        2
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::{Chirality, decimal_len_u8};
    use crate::errors::SmilesError;

    #[test]
    fn try_th_accepts_valid_values() {
        assert_eq!(Chirality::try_th(1), Ok(Chirality::TH(1)));
        assert_eq!(Chirality::try_th(2), Ok(Chirality::TH(2)));
    }

    #[test]
    fn try_th_rejects_out_of_range_values() {
        assert_eq!(Chirality::try_th(0), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_th(3), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_th(u8::MAX), Err(SmilesError::InvalidChirality));
    }

    #[test]
    fn try_al_accepts_valid_values() {
        assert_eq!(Chirality::try_al(1), Ok(Chirality::AL(1)));
        assert_eq!(Chirality::try_al(2), Ok(Chirality::AL(2)));
    }

    #[test]
    fn try_al_rejects_out_of_range_values() {
        assert_eq!(Chirality::try_al(0), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_al(3), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_al(u8::MAX), Err(SmilesError::InvalidChirality));
    }

    #[test]
    fn try_sp_accepts_valid_values() {
        assert_eq!(Chirality::try_sp(1), Ok(Chirality::SP(1)));
        assert_eq!(Chirality::try_sp(2), Ok(Chirality::SP(2)));
        assert_eq!(Chirality::try_sp(3), Ok(Chirality::SP(3)));
    }

    #[test]
    fn try_sp_rejects_out_of_range_values() {
        assert_eq!(Chirality::try_sp(0), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_sp(4), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_sp(u8::MAX), Err(SmilesError::InvalidChirality));
    }

    #[test]
    fn try_tb_accepts_lower_and_upper_bounds() {
        assert_eq!(Chirality::try_tb(1), Ok(Chirality::TB(1)));
        assert_eq!(Chirality::try_tb(20), Ok(Chirality::TB(20)));
    }

    #[test]
    fn try_tb_accepts_some_mid_values() {
        assert_eq!(Chirality::try_tb(2), Ok(Chirality::TB(2)));
        assert_eq!(Chirality::try_tb(10), Ok(Chirality::TB(10)));
        assert_eq!(Chirality::try_tb(19), Ok(Chirality::TB(19)));
    }

    #[test]
    fn try_tb_rejects_out_of_range_values() {
        assert_eq!(Chirality::try_tb(0), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_tb(21), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_tb(u8::MAX), Err(SmilesError::InvalidChirality));
    }

    #[test]
    fn try_oh_accepts_lower_and_upper_bounds() {
        assert_eq!(Chirality::try_oh(1), Ok(Chirality::OH(1)));
        assert_eq!(Chirality::try_oh(30), Ok(Chirality::OH(30)));
    }

    #[test]
    fn try_oh_accepts_some_mid_values() {
        assert_eq!(Chirality::try_oh(2), Ok(Chirality::OH(2)));
        assert_eq!(Chirality::try_oh(15), Ok(Chirality::OH(15)));
        assert_eq!(Chirality::try_oh(29), Ok(Chirality::OH(29)));
    }

    #[test]
    fn try_oh_rejects_out_of_range_values() {
        assert_eq!(Chirality::try_oh(0), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_oh(31), Err(SmilesError::InvalidChirality));
        assert_eq!(Chirality::try_oh(u8::MAX), Err(SmilesError::InvalidChirality));
    }

    #[test]
    fn simple_variants_are_distinct() {
        assert_ne!(Chirality::At, Chirality::AtAt);
        assert_ne!(Chirality::At, Chirality::TH(1));
        assert_ne!(Chirality::AtAt, Chirality::TH(1));
    }

    #[test]
    fn variants_with_same_tag_but_different_numbers_are_distinct() {
        assert_ne!(Chirality::TH(1), Chirality::TH(2));
        assert_ne!(Chirality::SP(1), Chirality::SP(2));
        assert_ne!(Chirality::TB(1), Chirality::TB(2));
        assert_ne!(Chirality::OH(1), Chirality::OH(2));
    }

    #[test]
    fn test_chirality_fmt_all_arms() {
        let chirals = vec![
            Chirality::At,
            Chirality::AtAt,
            Chirality::TH(2),
            Chirality::AL(2),
            Chirality::SP(10),
            Chirality::TB(20),
            Chirality::OH(30),
        ];
        let expected = vec!["@", "@@", "@TH2", "@AL2", "@SP10", "@TB20", "@OH30"];
        for (chiral, expected) in chirals.into_iter().zip(expected) {
            assert_eq!(expected, chiral.to_string());
        }
    }

    #[test]
    fn display_len_covers_all_number_width_paths() {
        assert_eq!(Chirality::At.display_len(), 1);
        assert_eq!(Chirality::AtAt.display_len(), 2);
        assert_eq!(Chirality::TH(2).display_len(), 4);
        assert_eq!(Chirality::AL(2).display_len(), 4);
        assert_eq!(Chirality::SP(3).display_len(), 4);
        assert_eq!(Chirality::TB(20).display_len(), 5);
        assert_eq!(Chirality::OH(30).display_len(), 5);

        assert_eq!(decimal_len_u8(1), 1);
        assert_eq!(decimal_len_u8(10), 2);
        assert_eq!(decimal_len_u8(100), 3);
    }
}
