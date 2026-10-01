//! Module for mapping and validating a ring marker
use core::fmt;

use crate::errors::SmilesError;

#[derive(Copy, Debug, PartialEq, PartialOrd, Clone, Eq, Ord, Hash)]
/// Represents a ring marker and implements tighter bounds for the minimal and
/// maximal value a ring marker can be
pub struct RingNum(u8);
impl RingNum {
    /// Attempts to generate a [`RingNum`] form a [`u8`],
    ///
    /// # Errors
    /// - Returns a [`SmilesError::RingNumberOverflow`] if the value is above
    ///   `99`
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::bond::ring_num::RingNum;
    ///
    /// let ring = RingNum::try_new(12)?;
    /// assert_eq!(ring.get(), 12);
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    pub fn try_new(num: u8) -> Result<Self, SmilesError> {
        (num <= 99).then_some(Self(num)).ok_or(SmilesError::RingNumberOverflow(num))
    }

    /// Returns the value for the [`RingNum`]
    ///
    /// # Examples
    ///
    /// ```
    /// use smiles_rs::bond::ring_num::RingNum;
    ///
    /// assert_eq!(RingNum::try_new(7)?.get(), 7);
    /// # Ok::<(), smiles_rs::SmilesError>(())
    /// ```
    #[must_use]
    pub fn get(&self) -> u8 {
        self.0
    }
}

impl fmt::Display for RingNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.get() > 9 { write!(f, "%{}", self.get()) } else { write!(f, "{}", self.get()) }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use crate::{bond::ring_num::RingNum, errors::SmilesError};
    #[test]
    fn test_ring_num_try_new_bounds() -> Result<(), SmilesError> {
        assert_eq!(RingNum::try_new(0)?.get(), 0);
        assert_eq!(RingNum::try_new(9)?.get(), 9);
        assert_eq!(RingNum::try_new(10)?.get(), 10);
        assert_eq!(RingNum::try_new(99)?.get(), 99);

        assert_eq!(RingNum::try_new(100), Err(SmilesError::RingNumberOverflow(100)));

        assert_eq!(RingNum::try_new(200), Err(SmilesError::RingNumberOverflow(200)));

        Ok(())
    }

    #[test]
    fn test_ring_num_fmt_all_arms() -> Result<(), SmilesError> {
        let cases = [
            (RingNum::try_new(0)?, "0"),
            (RingNum::try_new(3)?, "3"),
            (RingNum::try_new(9)?, "9"),
            (RingNum::try_new(10)?, "%10"),
            (RingNum::try_new(42)?, "%42"),
            (RingNum::try_new(99)?, "%99"),
        ];

        for (ring, expected) in cases {
            assert_eq!(expected, ring.to_string());
        }

        Ok(())
    }
}
