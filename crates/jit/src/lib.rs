//! JIT-style runtime for WebAssembly using Cranelift.

#![deny(missing_docs, trivial_numeric_casts)]
#![warn(unused_import_braces)]

mod code_memory;
mod debug;
mod demangling;
mod instantiate;
pub mod profiling;
mod unwind;

pub use crate::code_memory::CodeMemory;
#[cfg(feature = "addr2line")]
pub use crate::instantiate::SymbolizeContext;
pub use crate::instantiate::{
    subslice_range, CompiledFunctionInfo, CompiledModule, CompiledModuleInfo, ObjectBuilder,
};
pub use demangling::*;

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
