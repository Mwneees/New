//! Shims for ControlPlane when chaos mode is disabled. Enables
//! unconditional use of the type and its methods throughout cranelift.

use std::fmt::Display;

use arbitrary::Unstructured;

/// A shim for ControlPlane when chaos mode is disabled.
/// Please see the [crate-level documentation](crate).
#[derive(Debug, Clone, Default)]
pub struct ControlPlane {
    /// prevent direct instantiation (use `default` instead)
    _private: (),
}

impl ControlPlane {
    /// Generate a new control plane. This variant is used when chaos mode is
    /// disabled. It doesn't consume any bytes and always returns a default
    /// control plane.
    pub fn new(_u: &mut Unstructured, _fuel: u8) -> arbitrary::Result<Self> {
        Ok(Self::default())
    }

    /// Returns a pseudo-random boolean. This variant is used when chaos
    /// mode is disabled. It always returns `false`.
    #[inline]
    pub fn get_decision(&mut self) -> bool {
        false
    }
}

impl Display for ControlPlane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "; control plane: ()")
    }
}
