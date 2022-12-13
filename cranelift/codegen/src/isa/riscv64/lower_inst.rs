//! Lower a single Cranelift instruction into vcode.

use crate::ir::Inst as IRInst;

use crate::isa::riscv64::settings as riscv64_settings;
use crate::machinst::lower::*;
use crate::settings::Flags;
use crate::CodegenResult;

use crate::isa::riscv64::inst::*;
use target_lexicon::Triple;

/// Actually codegen an instruction's results into registers.
pub(crate) fn lower_insn_to_regs(
    ctx: &mut Lower<Inst>,
    insn: IRInst,
    triple: &Triple,
    flags: &Flags,
    isa_flags: &riscv64_settings::Flags,
) -> CodegenResult<InstOutput> {
    let ty = if ctx.num_outputs(insn) > 0 {
        Some(ctx.output_ty(insn, 0))
    } else {
        None
    };
    if let Some(temp_regs) = super::lower::isle::lower(ctx, flags, triple, isa_flags, insn) {
        return Ok(temp_regs);
    }
    unreachable!(
        "not implemented in ISLE: inst = `{}`, type = `{:?}`",
        ctx.dfg().display_inst(insn),
        ty
    );
}
