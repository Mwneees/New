//! Compilation process, from AST to Sema to Sequences of Insts.

use crate::error::Result;
use crate::{ast, codegen, sema, trie};

/// Compile the given AST definitions into Rust source code.
pub fn compile(defs: &ast::Defs, options: &codegen::CodegenOptions) -> Result<String> {
    let mut typeenv = sema::TypeEnv::from_ast(defs)?;
    let termenv = sema::TermEnv::from_ast(&mut typeenv, defs)?;

    // As the overlap checker currently finds a lot of overlap errors in the lowerings, require it
    // to be explicitly enabled while we work through them.
    #[cfg(feature = "check-overlap")]
    crate::overlap::check(&mut typeenv, &termenv)?;

    let tries = trie::build_tries(&typeenv, &termenv);
    Ok(codegen::codegen(&typeenv, &termenv, &tries, options))
}
