#![cfg_attr(feature = "bench", feature(test))]
#![feature(proc_macro_hygiene)]

#[macro_use]
extern crate smallvec;
extern crate capstone;
extern crate either;
pub extern crate wasmparser;
#[macro_use]
extern crate memoffset;
extern crate dynasm;
extern crate dynasmrt;
extern crate itertools;
// Just so we can implement `Signature` for `cranelift_codegen::ir::Signature`
extern crate cranelift_codegen;
extern crate multi_mut;

mod backend;
mod disassemble;
mod error;
mod function_body;
mod microwasm;
mod module;
mod translate_sections;

#[cfg(feature = "bench")]
mod benches;

pub use crate::backend::CodeGenSession;
pub use crate::function_body::translate_wasm as translate_function;
pub use crate::module::{
    translate, ExecutableModule, ExecutionError, ModuleContext, Signature, TranslatedModule,
};
