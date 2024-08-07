//! Pulley registers.

use core::hash::Hash;
use core::{fmt, ops::Range};

/// Trait for common register operations.
pub trait Reg: Sized + Copy + Eq + Ord + Hash + Into<AnyReg> + fmt::Debug + fmt::Display {
    /// Range of valid register indices.
    const RANGE: Range<u8>;

    /// Convert a register index to a register, without bounds checking.
    unsafe fn new_unchecked(index: u8) -> Self;

    /// Convert a register index to a register, with bounds checking.
    fn new(index: u8) -> Option<Self> {
        if Self::RANGE.contains(&index) {
            Some(unsafe { Self::new_unchecked(index) })
        } else {
            None
        }
    }

    /// Convert a register to its index.
    fn to_u8(self) -> u8;

    /// Convert a register to its index.
    fn index(self) -> usize {
        self.to_u8().into()
    }
}

macro_rules! impl_reg {
    ($reg_ty:ty, $any:ident, $range:expr) => {
        impl From<$reg_ty> for AnyReg {
            fn from(r: $reg_ty) -> Self {
                AnyReg::$any(r)
            }
        }

        impl fmt::Display for $reg_ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Debug::fmt(&self, f)
            }
        }

        impl Reg for $reg_ty {
            const RANGE: Range<u8> = $range;

            unsafe fn new_unchecked(index: u8) -> Self {
                core::mem::transmute(index)
            }

            fn to_u8(self) -> u8 {
                self as u8
            }
        }
    };
}

/// An `x` register: integers.
#[repr(u8)]
#[derive(Debug,Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[allow(non_camel_case_types, missing_docs)]
#[rustfmt::skip]
pub enum XReg {
    x0,  x1,  x2,  x3,  x4,  x5,  x6,  x7,  x8,  x9,
    x10, x11, x12, x13, x14, x15, x16, x17, x18, x19,
    x20, x21, x22, x23, x24, x25, x26,

    /// The special `sp` stack pointer register.
    sp,

    /// The special `lr` link register.
    lr,

    /// The special `fp` frame pointer register.
    fp,

    /// The special `spilltmp0` scratch register.
    spilltmp0,

    /// The special `spilltmp1` scratch register.
    spilltmp1,
}

impl XReg {
    /// Is this `x` register a special register?
    pub fn is_special(self) -> bool {
        matches!(
            self,
            Self::sp | Self::lr | Self::fp | Self::spilltmp0 | Self::spilltmp1
        )
    }
}

/// An `f` register: floats.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[allow(non_camel_case_types, missing_docs)]
#[rustfmt::skip]
pub enum FReg {
    f0,  f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,
    f10, f11, f12, f13, f14, f15, f16, f17, f18, f19,
    f20, f21, f22, f23, f24, f25, f26, f27, f28, f29,
    f30, f31,
}

/// A `v` register: vectors.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
#[allow(non_camel_case_types, missing_docs)]
#[rustfmt::skip]
pub enum VReg {
    v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,
    v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
    v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
    v30, v31,
}

impl_reg!(XReg, X, 0..32);
impl_reg!(FReg, F, 0..32);
impl_reg!(VReg, V, 0..32);

/// Any register, regardless of class.
///
/// Never appears inside an instruction -- instructions always name a particular
/// class of register -- but this is useful for testing and things like that.
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AnyReg {
    X(XReg),
    F(FReg),
    V(VReg),
}

impl fmt::Display for AnyReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl fmt::Debug for AnyReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AnyReg::X(r) => fmt::Debug::fmt(r, f),
            AnyReg::F(r) => fmt::Debug::fmt(r, f),
            AnyReg::V(r) => fmt::Debug::fmt(r, f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn special_x_regs() {
        assert!(XReg::sp.is_special());
        assert!(XReg::lr.is_special());
        assert!(XReg::fp.is_special());
        assert!(XReg::spilltmp0.is_special());
        assert!(XReg::spilltmp1.is_special());
    }

    #[test]
    fn not_special_x_regs() {
        for i in 0..27 {
            assert!(!XReg::new(i).unwrap().is_special());
        }
    }
}
