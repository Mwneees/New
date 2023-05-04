//! Shims for ControlPlane when chaos mode is disabled. Enables
//! unconditional use of the type and its methods throughout cranelift.

use std::fmt::Display;

/// A shim for ControlPlane when chaos mode is disabled.
/// Please see the [crate-level documentation](crate).
#[derive(Debug, Clone, Default)]
pub struct ControlPlane {
    /// prevent direct instantiation (use `default` instead)
    _private: (),
}

/// A shim for ControlPlane's `Arbitrary` implementation when chaos mode is
/// disabled. It doesn't consume any bytes and always returns a default
/// control plane.
impl arbitrary::Arbitrary<'_> for ControlPlane {
    fn arbitrary(_u: &mut arbitrary::Unstructured<'_>) -> arbitrary::Result<Self> {
        Ok(Self::default())
    }
}

impl ControlPlane {
    /// Set the [fuel limit](crate#fuel-limit). This variant is used when
    /// chaos mode is disabled. It doesn't do anything.
    pub fn set_fuel(&mut self, _fuel: u8) {}

    /// Returns a pseudo-random boolean. This variant is used when chaos
    /// mode is disabled. It always returns `false`.
    #[inline]
    pub fn get_decision(&mut self) -> bool {
        false
    }

    /// Shuffles the items in the slice into a pseudo-random permutation.
    /// This variant is used when chaos mode is disabled. It doesn't do
    /// anything.
    #[inline]
    pub fn shuffle<T>(&mut self, _slice: &mut [T]) {}

    /// Returns a new iterator over the same items as the input iterator in
    /// a pseudo-random order. This variant is used when chaos mode is
    /// disabled. It always returns the same order.
    #[inline]
    pub fn shuffled<T>(&mut self, iter: impl Iterator<Item = T>) -> impl Iterator<Item = T> {
        iter
    }
}

impl Display for ControlPlane {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl TryFrom<&str> for ControlPlane {
    type Error = ();
    fn try_from(_value: &str) -> Result<Self, Self::Error> {
        Ok(Self::default())
    }
}
