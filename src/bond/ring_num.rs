//! Module for mapping and validating a ring marker
use core::{fmt, num::NonZeroU32};

use crate::errors::SmilesError;

#[derive(Copy, Debug, PartialEq, PartialOrd, Clone, Eq, Ord, Hash)]
/// Represents a ring marker from `0` to [`RingNum::MAX`]
// stored offset by one so `Option<RingNum>` stays four bytes
pub struct RingNum(NonZeroU32);
impl RingNum {
    /// Largest ring marker, the five-digit limit of the `%(N)` spelling.
    pub const MAX: u32 = 99_999;

    /// Attempts to generate a [`RingNum`] from a [`u32`],
    ///
    /// # Errors
    /// - Returns a [`SmilesError::RingNumberOverflow`] if the value is above
    ///   [`RingNum::MAX`]
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
    pub fn try_new(num: u32) -> Result<Self, SmilesError> {
        if num > Self::MAX {
            return Err(SmilesError::RingNumberOverflow(num));
        }
        Ok(Self(NonZeroU32::MIN.saturating_add(num)))
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
    pub const fn get(&self) -> u32 {
        self.0.get() - 1
    }
}

impl fmt::Display for RingNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.get() {
            num @ 0..=9 => write!(f, "{num}"),
            num @ 10..=99 => write!(f, "%{num}"),
            num => write!(f, "%({num})"),
        }
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
        assert_eq!(RingNum::try_new(100)?.get(), 100);
        assert_eq!(RingNum::try_new(RingNum::MAX)?.get(), RingNum::MAX);

        assert_eq!(
            RingNum::try_new(RingNum::MAX + 1),
            Err(SmilesError::RingNumberOverflow(RingNum::MAX + 1))
        );

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
            (RingNum::try_new(100)?, "%(100)"),
            (RingNum::try_new(RingNum::MAX)?, "%(99999)"),
        ];

        for (ring, expected) in cases {
            assert_eq!(expected, ring.to_string());
        }

        Ok(())
    }
}
