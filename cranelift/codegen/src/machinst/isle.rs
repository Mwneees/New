use crate::ir::{types, Inst, Value, ValueList};
use crate::machinst::{get_output_reg, InsnOutput, LowerCtx, Reg, Writable};
use alloc::boxed::Box;
use alloc::vec::Vec;
use smallvec::SmallVec;
use std::cell::Cell;

pub use super::MachLabel;
pub use crate::ir::{ExternalName, FuncRef, GlobalValue, SigRef};
pub use crate::isa::unwind::UnwindInst;
pub use crate::machinst::RelocDistance;

pub type Unit = ();
pub type ValueSlice = (ValueList, usize);
pub type ValueArray2 = [Value; 2];
pub type ValueArray3 = [Value; 3];
pub type WritableReg = Writable<Reg>;
pub type VecReg = Vec<Reg>;
pub type VecWritableReg = Vec<WritableReg>;
pub type ValueRegs = crate::machinst::ValueRegs<Reg>;
pub type InstOutput = SmallVec<[ValueRegs; 2]>;
pub type InstOutputBuilder = Cell<InstOutput>;
pub type VecMachLabel = Vec<MachLabel>;
pub type BoxExternalName = Box<ExternalName>;

/// Helper macro to define methods in `prelude.isle` within `impl Context for
/// ...` for each backend. These methods are shared amongst all backends.
#[macro_export]
#[doc(hidden)]
macro_rules! isle_lower_prelude_methods {
    () => {
        isle_common_prelude_methods!();

        fn avoid_div_traps(&mut self, _: Type) -> Option<()> {
            if self.flags.avoid_div_traps() {
                Some(())
            } else {
                None
            }
        }

        #[inline]
        fn unpack_value_array_2(&mut self, arr: &ValueArray2) -> (Value, Value) {
            let [a, b] = *arr;
            (a, b)
        }

        #[inline]
        fn pack_value_array_2(&mut self, a: Value, b: Value) -> ValueArray2 {
            [a, b]
        }

        #[inline]
        fn unpack_value_array_3(&mut self, arr: &ValueArray3) -> (Value, Value, Value) {
            let [a, b, c] = *arr;
            (a, b, c)
        }

        #[inline]
        fn pack_value_array_3(&mut self, a: Value, b: Value, c: Value) -> ValueArray3 {
            [a, b, c]
        }

        #[inline]
        fn value_reg(&mut self, reg: Reg) -> ValueRegs {
            ValueRegs::one(reg)
        }

        #[inline]
        fn value_regs(&mut self, r1: Reg, r2: Reg) -> ValueRegs {
            ValueRegs::two(r1, r2)
        }

        #[inline]
        fn value_regs_invalid(&mut self) -> ValueRegs {
            ValueRegs::invalid()
        }

        #[inline]
        fn output_none(&mut self) -> InstOutput {
            smallvec::smallvec![]
        }

        #[inline]
        fn output(&mut self, regs: ValueRegs) -> InstOutput {
            smallvec::smallvec![regs]
        }

        #[inline]
        fn output_pair(&mut self, r1: ValueRegs, r2: ValueRegs) -> InstOutput {
            smallvec::smallvec![r1, r2]
        }

        #[inline]
        fn output_builder_new(&mut self) -> InstOutputBuilder {
            std::cell::Cell::new(InstOutput::new())
        }

        #[inline]
        fn output_builder_push(&mut self, builder: &InstOutputBuilder, regs: ValueRegs) -> Unit {
            let mut vec = builder.take();
            vec.push(regs);
            builder.set(vec);
        }

        #[inline]
        fn output_builder_finish(&mut self, builder: &InstOutputBuilder) -> InstOutput {
            builder.take()
        }

        #[inline]
        fn temp_writable_reg(&mut self, ty: Type) -> WritableReg {
            let value_regs = self.lower_ctx.alloc_tmp(ty);
            value_regs.only_reg().unwrap()
        }

        #[inline]
        fn invalid_reg(&mut self) -> Reg {
            use crate::machinst::valueregs::InvalidSentinel;
            Reg::invalid_sentinel()
        }

        #[inline]
        fn invalid_reg_etor(&mut self, reg: Reg) -> Option<()> {
            use crate::machinst::valueregs::InvalidSentinel;
            if reg.is_invalid_sentinel() {
                Some(())
            } else {
                None
            }
        }

        #[inline]
        fn valid_reg(&mut self, reg: Reg) -> Option<()> {
            use crate::machinst::valueregs::InvalidSentinel;
            if !reg.is_invalid_sentinel() {
                Some(())
            } else {
                None
            }
        }

        #[inline]
        fn put_in_reg(&mut self, val: Value) -> Reg {
            self.lower_ctx.put_value_in_regs(val).only_reg().unwrap()
        }

        #[inline]
        fn put_in_regs(&mut self, val: Value) -> ValueRegs {
            self.lower_ctx.put_value_in_regs(val)
        }

        #[inline]
        fn ensure_in_vreg(&mut self, reg: Reg, ty: Type) -> Reg {
            self.lower_ctx.ensure_in_vreg(reg, ty)
        }

        #[inline]
        fn value_regs_get(&mut self, regs: ValueRegs, i: usize) -> Reg {
            regs.regs()[i]
        }

        #[inline]
        fn value_list_slice(&mut self, list: ValueList) -> ValueSlice {
            (list, 0)
        }

        #[inline]
        fn value_slice_empty(&mut self, slice: ValueSlice) -> Option<()> {
            let (list, off) = slice;
            if off >= list.len(&self.lower_ctx.dfg().value_lists) {
                Some(())
            } else {
                None
            }
        }

        #[inline]
        fn value_slice_unwrap(&mut self, slice: ValueSlice) -> Option<(Value, ValueSlice)> {
            let (list, off) = slice;
            if let Some(val) = list.get(off, &self.lower_ctx.dfg().value_lists) {
                Some((val, (list, off + 1)))
            } else {
                None
            }
        }

        #[inline]
        fn value_slice_len(&mut self, slice: ValueSlice) -> usize {
            let (list, off) = slice;
            list.len(&self.lower_ctx.dfg().value_lists) - off
        }

        #[inline]
        fn value_slice_get(&mut self, slice: ValueSlice, idx: usize) -> Value {
            let (list, off) = slice;
            list.get(off + idx, &self.lower_ctx.dfg().value_lists)
                .unwrap()
        }

        #[inline]
        fn writable_reg_to_reg(&mut self, r: WritableReg) -> Reg {
            r.to_reg()
        }

        #[inline]
        fn inst_results(&mut self, inst: Inst) -> ValueSlice {
            (self.lower_ctx.dfg().inst_results_list(inst), 0)
        }

        #[inline]
        fn first_result(&mut self, inst: Inst) -> Option<Value> {
            self.lower_ctx.dfg().inst_results(inst).first().copied()
        }

        #[inline]
        fn inst_data(&mut self, inst: Inst) -> InstructionData {
            self.lower_ctx.dfg()[inst].clone()
        }

        #[inline]
        fn value_type(&mut self, val: Value) -> Type {
            self.lower_ctx.dfg().value_type(val)
        }

        #[inline]
        fn def_inst(&mut self, val: Value) -> Option<Inst> {
            self.lower_ctx.dfg().value_def(val).inst()
        }

        #[inline]
        fn func_ref_data(&mut self, func_ref: FuncRef) -> (SigRef, ExternalName, RelocDistance) {
            let funcdata = &self.lower_ctx.dfg().ext_funcs[func_ref];
            (
                funcdata.signature,
                funcdata.name.clone(),
                funcdata.reloc_distance(),
            )
        }

        #[inline]
        fn symbol_value_data(
            &mut self,
            global_value: GlobalValue,
        ) -> Option<(ExternalName, RelocDistance, i64)> {
            let (name, reloc, offset) = self.lower_ctx.symbol_value_data(global_value)?;
            Some((name.clone(), reloc, offset))
        }

        #[inline]
        fn reloc_distance_near(&mut self, dist: RelocDistance) -> Option<()> {
            if dist == RelocDistance::Near {
                Some(())
            } else {
                None
            }
        }

        #[inline]
        fn emit_u64_le_const(&mut self, value: u64) -> VCodeConstant {
            let data = VCodeConstantData::U64(value.to_le_bytes());
            self.lower_ctx.use_constant(data)
        }
    };
}

/// This structure is used to implement the ISLE-generated `Context` trait and
/// internally has a temporary reference to a machinst `LowerCtx`.
pub(crate) struct IsleContext<'a, C: LowerCtx, F, I, const N: usize>
where
    [(C::I, bool); N]: smallvec::Array,
{
    pub lower_ctx: &'a mut C,
    pub flags: &'a F,
    pub isa_flags: &'a I,
}

/// Shared lowering code amongst all backends for doing ISLE-based lowering.
///
/// The `isle_lower` argument here is an ISLE-generated function for `lower` and
/// then this function otherwise handles register mapping and such around the
/// lowering.
pub(crate) fn lower_common<C, F, I, IF, const N: usize>(
    lower_ctx: &mut C,
    flags: &F,
    isa_flags: &I,
    outputs: &[InsnOutput],
    inst: Inst,
    isle_lower: IF,
) -> Result<(), ()>
where
    C: LowerCtx,
    [(C::I, bool); N]: smallvec::Array<Item = (C::I, bool)>,
    IF: Fn(&mut IsleContext<'_, C, F, I, N>, Inst) -> Option<InstOutput>,
{
    // TODO: reuse the ISLE context across lowerings so we can reuse its
    // internal heap allocations.
    let mut isle_ctx = IsleContext {
        lower_ctx,
        flags,
        isa_flags,
    };

    let temp_regs = isle_lower(&mut isle_ctx, inst).ok_or(())?;

    #[cfg(debug_assertions)]
    {
        debug_assert_eq!(
            temp_regs.len(),
            outputs.len(),
            "the number of temporary values and destination values do \
         not match ({} != {}); ensure the correct registers are being \
         returned.",
            temp_regs.len(),
            outputs.len(),
        );
    }

    // The ISLE generated code emits its own registers to define the
    // instruction's lowered values in. However, other instructions
    // that use this SSA value will be lowered assuming that the value
    // is generated into a pre-assigned, different, register.
    //
    // To connect the two, we set up "aliases" in the VCodeBuilder
    // that apply when it is building the Operand table for the
    // regalloc to use. These aliases effectively rewrite any use of
    // the pre-assigned register to the register that was returned by
    // the ISLE lowering logic.
    for i in 0..outputs.len() {
        let regs = temp_regs[i];
        let dsts = get_output_reg(isle_ctx.lower_ctx, outputs[i]);
        let ty = isle_ctx
            .lower_ctx
            .output_ty(outputs[i].insn, outputs[i].output);
        if ty == types::IFLAGS || ty == types::FFLAGS {
            // Flags values do not occupy any registers.
            assert!(regs.len() == 0);
        } else {
            for (dst, temp) in dsts.regs().iter().zip(regs.regs().iter()) {
                isle_ctx.lower_ctx.set_vreg_alias(dst.to_reg(), *temp);
            }
        }
    }

    Ok(())
}
