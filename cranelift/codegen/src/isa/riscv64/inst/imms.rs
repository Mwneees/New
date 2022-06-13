//! Riscv64 ISA definitions: immediate constants.

// Some variants are never constructed, but we still want them as options in the future.
#[allow(dead_code)]
use std::fmt::{Debug, Display, Formatter, Result};

#[derive(Copy, Clone, Debug)]
pub struct Imm12 {
    pub bits: i16,
}

impl Imm12 {
    pub fn maybe_from_u64(val: u64) -> Option<Imm12> {
        let bit = 1 << 11;
        if val == 0 {
            Some(Imm12 { bits: 0 })
        } else if (val & bit) != 0 && (val >> 12) == 0xffff_ffff_ffff_f {
            Some(Imm12 {
                bits: (val & 0xffff) as i16,
            })
        } else if (val & bit) == 0 && (val >> 12) == 0 {
            Some(Imm12 {
                bits: (val & 0xffff) as i16,
            })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn from_bits(bits: i16) -> Self {
        Self { bits: bits & 0xfff }
    }
    #[inline(always)]
    pub(crate) fn form_bool(b: bool) -> Self {
        if b {
            Self { bits: -1 }
        } else {
            Self { bits: 0 }
        }
    }

    /// Create a zero immediate of this format.
    #[inline(always)]
    pub fn zero() -> Self {
        Imm12 { bits: 0 }
    }
    #[inline(always)]
    pub fn as_i16(self) -> i16 {
        self.bits
    }
    #[inline(always)]
    pub fn as_u32(&self) -> u32 {
        (self.bits as u32) & 0xfff
    }
}

impl Into<i64> for Imm12 {
    fn into(self) -> i64 {
        self.bits as i64
    }
}
impl Display for Imm12 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:+}", self.bits)
    }
}

impl std::ops::Neg for Imm12 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self { bits: -self.bits }
    }
}

// singed
#[derive(Clone, Copy)]
pub struct Umm20 {
    /// The immediate bits.
    pub bits: i32,
}

impl Umm20 {
    #[inline(always)]
    pub fn from_bits(bits: i32) -> Self {
        Self {
            bits: bits & 0xf_ffff,
        }
    }
    #[inline(always)]
    pub fn as_u32(&self) -> u32 {
        (self.bits as u32) & 0xf_ffff
    }
}

impl Debug for Umm20 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.bits)
    }
}

impl Display for Umm20 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.bits)
    }
}

#[derive(Clone, Copy)]
pub struct Uimm5 {
    bits: u8,
}

impl Uimm5 {
    pub fn maybe_from_u64(val: u64) -> Option<Uimm5> {
        if (val >> 5) == 0 {
            Some(Self {
                bits: (val & 0b1_1111) as u8,
            })
        } else {
            None
        }
    }
    pub fn from_bits(bits: u8) -> Self {
        Self { bits }
    }
    /// Create a zero immediate of this format.
    pub fn zero() -> Self {
        Self { bits: 0 }
    }
    pub fn as_u32(&self) -> u32 {
        (self.bits as u32) & 0b1_1111
    }
}

impl Debug for Uimm5 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.bits)
    }
}

impl Display for Uimm5 {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.bits)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_imm12() {
        let x = Imm12::zero();
        assert_eq!(0, x.as_u32());
        Imm12::maybe_from_u64(0xffff_ffff_ffff_ffff).unwrap();
    }
}
