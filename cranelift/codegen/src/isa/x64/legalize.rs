use crate::flowgraph;
use crate::ir;
use crate::isa::x64::X64Backend;
use crate::legalizer::isle;

// Used by ISLE
use crate::cursor::CursorPosition;
use crate::ir::condcodes::*;
use crate::ir::immediates::*;
use crate::ir::types::*;
use crate::ir::*;
use crate::machinst::isle::*;

#[allow(dead_code, unused_variables, unreachable_patterns)]
mod generated {
    include!(concat!(env!("ISLE_DIR"), "/legalize_x64.rs"));
}

pub(crate) fn run(
    isa: &X64Backend,
    func: &mut ir::Function,
    cfg: &mut flowgraph::ControlFlowGraph,
) {
    crate::legalizer::isle::run(isa, func, cfg, |cx, i| {
        generated::constructor_legalize(cx, i)
    })
}

impl generated::Context for isle::LegalizeContext<'_, X64Backend> {
    crate::isle_common_legalizer_methods!();
}
