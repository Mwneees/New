//! Standalone environment for WebAssembly using Cranelift. Provides functions to translate
//! `get_global`, `set_global`, `memory.size`, `memory.grow`, `call_indirect` that hardcode in
//! the translation the base addresses of regions of memory that will hold the globals, tables and
//! linear memories.

#![deny(missing_docs)]
#![warn(clippy::cast_sign_loss)]
#![no_std]

#[cfg(feature = "std")]
#[macro_use]
extern crate std;
extern crate alloc;

pub use wasmtime_types::prelude;

mod address_map;
mod builtin;
mod demangling;
mod gc;
mod module;
mod module_artifacts;
mod module_types;
pub mod obj;
mod ref_bits;
mod scopevec;
mod stack_map;
mod trap_encoding;
mod tunables;
mod vmoffsets;

pub use crate::address_map::*;
pub use crate::builtin::*;
pub use crate::demangling::*;
pub use crate::gc::*;
pub use crate::module::*;
pub use crate::module_artifacts::*;
pub use crate::module_types::*;
pub use crate::ref_bits::*;
pub use crate::scopevec::ScopeVec;
pub use crate::stack_map::StackMap;
pub use crate::trap_encoding::*;
pub use crate::tunables::*;
pub use crate::vmoffsets::*;
pub use object;

#[cfg(feature = "compile")]
mod compile;
#[cfg(feature = "compile")]
pub use crate::compile::*;

#[cfg(feature = "component-model")]
pub mod component;
#[cfg(all(feature = "component-model", feature = "compile"))]
pub mod fact;

// Reexport all of these type-level since they're quite commonly used and it's
// much easier to refer to everything through one crate rather than importing
// one of three and making sure you're using the right one.
pub use cranelift_entity::*;
pub use wasmtime_types::*;

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
