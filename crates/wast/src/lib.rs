//! Implementation of the WAST text format for wasmtime.

#![deny(missing_docs)]
#![warn(unused_import_braces)]

#[cfg(feature = "component-model")]
mod component;
mod core;
mod spectest;
mod wast;

pub use crate::spectest::link_spectest;
pub use crate::wast::WastContext;

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
