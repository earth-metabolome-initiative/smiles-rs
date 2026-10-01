//! Module for validating a charge on an atom
use core::fmt;

use crate::errors::SmilesError;

#[derive(Copy, Default, Debug, PartialEq, Clone, Eq, Hash)]
/// Wrapper struct for possible charge to limit upper and lower bounds
pub struct Charge(i8);

impl Charge {
    /// Attempts to set the `Charge`, if outside of bounds returns
    /// [`SmilesError::ChargeUnderflow`] or [`SmilesError::ChargeOverflow`]
    ///
    /// # Errors
    /// - Returns [`SmilesError::ChargeUnderflow`] if `i8` is less than `-15`
    /// - Returns [`SmilesError::ChargeOverflow`] if `i8` is greater than `15`
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::bracketed::charge::Charge;
    ///
    /// assert_eq!(Charge::try_new(2)?.get(), 2);
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    pub fn try_new(num: i8) -> Result<Self, SmilesError> {
        (-15..=15).contains(&num).then_some(Self(num)).ok_or(if num.is_negative() {
            SmilesError::ChargeUnderflow(num)
        } else {
            SmilesError::ChargeOverflow(num)
        })
    }

    /// Returns the `Charge` value as `i8`
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::atom::bracketed::charge::Charge;
    ///
    /// assert_eq!(Charge::try_new(-1)?.get(), -1);
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    #[must_use]
    pub fn get(&self) -> i8 {
        self.0
    }

    #[inline]
    #[must_use]
    pub(crate) fn display_len(self) -> usize {
        match self.0 {
            0 => 0,
            1 | -1 => 1,
            2..=9 | -9..=-2 => 2,
            10..=15 => 3,
            -15..=-10 => 4,
            _ => unreachable!("state not reachable, charges can only be between -15 & 15"),
        }
    }
}

impl fmt::Display for Charge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            0 => Ok(()),
            1 => f.write_str("+"),
            n @ 2..=15 => write!(f, "+{n}"),
            -1 => f.write_str("-"),
            n @ -15..=-2 => write!(f, "{n}"),
            _ => unreachable!("state not reachable, charges can only be between -15 & 15"),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::{String, ToString};
    use std::panic::catch_unwind;

    use super::Charge;
    use crate::errors::SmilesError;

    #[test]
    fn default_is_zero() {
        assert_eq!(Charge::default().get(), 0);
    }

    #[test]
    fn try_new_accepts_in_range_values() {
        assert_eq!(Charge::try_new(-15).map(|c| c.get()), Ok(-15));
        assert_eq!(Charge::try_new(15).map(|c| c.get()), Ok(15));
        assert_eq!(Charge::try_new(-1).map(|c| c.get()), Ok(-1));
        assert_eq!(Charge::try_new(0).map(|c| c.get()), Ok(0));
        assert_eq!(Charge::try_new(1).map(|c| c.get()), Ok(1));
        assert_eq!(Charge::try_new(7).map(|c| c.get()), Ok(7));
        assert_eq!(Charge::try_new(-7).map(|c| c.get()), Ok(-7));
    }

    #[test]
    fn try_new_rejects_negative_underflow() {
        assert_eq!(Charge::try_new(-16), Err(SmilesError::ChargeUnderflow(-16)));
        assert_eq!(Charge::try_new(i8::MIN), Err(SmilesError::ChargeUnderflow(i8::MIN)));
    }

    #[test]
    fn try_new_rejects_positive_overflow() {
        assert_eq!(Charge::try_new(16), Err(SmilesError::ChargeOverflow(16)));
        assert_eq!(Charge::try_new(i8::MAX), Err(SmilesError::ChargeOverflow(i8::MAX)));
    }

    #[test]
    fn try_new_error_type_depends_on_sign() {
        match Charge::try_new(-16) {
            Err(SmilesError::ChargeUnderflow(-16)) => {}
            other => panic!("expected ChargeUnderflow(-16), got {other:?}"),
        }

        match Charge::try_new(16) {
            Err(SmilesError::ChargeOverflow(16)) => {}
            other => panic!("expected ChargeOverflow(16), got {other:?}"),
        }
    }

    #[test]
    fn test_charge_fmt_all_arms() -> Result<(), SmilesError> {
        for i in -15..=15 {
            let charge = Charge::try_new(i)?;
            let expected = match i {
                0 => String::new(),
                1 => "+".to_string(),
                2..=15 => format!("+{i}"),
                -1 => "-".to_string(),
                -15..=-2 => i.to_string(),
                _ => unreachable!(),
            };

            assert_eq!(expected, charge.to_string());
        }
        Ok(())
    }

    #[test]
    fn test_unreachable_fmt() {
        let unreachable_charge = Charge(17);

        let result = catch_unwind(|| {
            let _ = unreachable_charge.to_string();
        });

        assert!(result.is_err());
    }

    #[test]
    fn display_len_covers_multi_digit_magnitudes() -> Result<(), SmilesError> {
        assert_eq!(Charge::try_new(10)?.display_len(), 3);
        assert_eq!(Charge::try_new(15)?.display_len(), 3);
        assert_eq!(Charge::try_new(-10)?.display_len(), 4);
        assert_eq!(Charge::try_new(-15)?.display_len(), 4);
        Ok(())
    }
}
