//! A pre-legalization rewriting pass.
//!
//! This module provides early-stage optimizations. The optimizations found
//! should be useful for already well-optimized code. More general purpose
//! early-stage optimizations can be found in the preopt crate.

use crate::cursor::{Cursor, FuncCursor};
use crate::divconst_magic_numbers::{magic_s32, magic_s64, magic_u32, magic_u64};
use crate::divconst_magic_numbers::{MS32, MS64, MU32, MU64};
use crate::flowgraph::ControlFlowGraph;
use crate::ir::immediates::Offset32;
use crate::ir::{
    condcodes::{CondCode, IntCC},
    instructions::Opcode,
    types::{I128, I32, I64, INVALID},
    Block, DataFlowGraph, Function, Inst, InstBuilder, InstructionData, Type, Value,
};
use crate::isa::TargetIsa;
use crate::timing;

#[inline]
/// Replaces the unique result of the instruction inst to an alias of the given value, and
/// replaces the instruction with a nop. Can be used only on instructions producing one unique
/// result, otherwise will assert.
fn replace_single_result_with_alias(dfg: &mut DataFlowGraph, inst: Inst, value: Value) {
    // Replace the result value by an alias.
    let results = dfg.detach_results(inst);
    debug_assert!(results.len(&dfg.value_lists) == 1);
    let result = results.get(0, &dfg.value_lists).unwrap();
    dfg.change_to_alias(result, value);

    // Replace instruction by a nop.
    dfg.replace(inst).nop();
}

//----------------------------------------------------------------------
//
// Pattern-match helpers and transformation for div and rem by constants.

// Simple math helpers

/// if `x` is a power of two, or the negation thereof, return the power along
/// with a boolean that indicates whether `x` is negative. Else return None.
#[inline]
fn i32_is_power_of_two(x: i32) -> Option<(bool, u32)> {
    // We have to special-case this because abs(x) isn't representable.
    if x == -0x8000_0000 {
        return Some((true, 31));
    }
    let abs_x = i32::wrapping_abs(x) as u32;
    if abs_x.is_power_of_two() {
        return Some((x < 0, abs_x.trailing_zeros()));
    }
    None
}

/// Same comments as for i32_is_power_of_two apply.
#[inline]
fn i64_is_power_of_two(x: i64) -> Option<(bool, u32)> {
    // We have to special-case this because abs(x) isn't representable.
    if x == -0x8000_0000_0000_0000 {
        return Some((true, 63));
    }
    let abs_x = i64::wrapping_abs(x) as u64;
    if abs_x.is_power_of_two() {
        return Some((x < 0, abs_x.trailing_zeros()));
    }
    None
}

/// Representation of an instruction that can be replaced by a single division/remainder operation
/// between a left Value operand and a right immediate operand.
#[derive(Debug)]
enum DivRemByConstInfo {
    DivU32(Value, u32),
    DivU64(Value, u64),
    DivS32(Value, i32),
    DivS64(Value, i64),
    RemU32(Value, u32),
    RemU64(Value, u64),
    RemS32(Value, i32),
    RemS64(Value, i64),
}

/// Possibly create a DivRemByConstInfo from the given components, by figuring out which, if any,
/// of the 8 cases apply, and also taking care to sanity-check the immediate.
fn package_up_divrem_info(
    value: Value,
    value_type: Type,
    imm_i64: i64,
    is_signed: bool,
    is_rem: bool,
) -> Option<DivRemByConstInfo> {
    let imm_u64 = imm_i64 as u64;

    match (is_signed, value_type) {
        (false, I32) => {
            if imm_u64 < 0x1_0000_0000 {
                if is_rem {
                    Some(DivRemByConstInfo::RemU32(value, imm_u64 as u32))
                } else {
                    Some(DivRemByConstInfo::DivU32(value, imm_u64 as u32))
                }
            } else {
                None
            }
        }

        (false, I64) => {
            // unsigned 64, no range constraint.
            if is_rem {
                Some(DivRemByConstInfo::RemU64(value, imm_u64))
            } else {
                Some(DivRemByConstInfo::DivU64(value, imm_u64))
            }
        }

        (true, I32) => {
            if imm_u64 <= 0x7fff_ffff || imm_u64 >= 0xffff_ffff_8000_0000 {
                if is_rem {
                    Some(DivRemByConstInfo::RemS32(value, imm_u64 as i32))
                } else {
                    Some(DivRemByConstInfo::DivS32(value, imm_u64 as i32))
                }
            } else {
                None
            }
        }

        (true, I64) => {
            // signed 64, no range constraint.
            if is_rem {
                Some(DivRemByConstInfo::RemS64(value, imm_u64 as i64))
            } else {
                Some(DivRemByConstInfo::DivS64(value, imm_u64 as i64))
            }
        }

        _ => None,
    }
}

/// Examine `inst` to see if it is a div or rem by a constant, and if so return the operands,
/// signedness, operation size and div-vs-rem-ness in a handy bundle.
fn get_div_info(inst: Inst, dfg: &DataFlowGraph) -> Option<DivRemByConstInfo> {
    if let InstructionData::BinaryImm64 { opcode, arg, imm } = dfg.insts[inst] {
        let (is_signed, is_rem) = match opcode {
            Opcode::UdivImm => (false, false),
            Opcode::UremImm => (false, true),
            Opcode::SdivImm => (true, false),
            Opcode::SremImm => (true, true),
            _ => return None,
        };
        return package_up_divrem_info(arg, dfg.value_type(arg), imm.into(), is_signed, is_rem);
    }

    None
}

/// Actually do the transformation given a bundle containing the relevant information.
/// `divrem_info` describes a div or rem by a constant, that `pos` currently points at, and `inst`
/// is the associated instruction.  `inst` is replaced by a sequence of other operations that
/// calculate the same result. Note that there are various `divrem_info` cases where we cannot do
/// any transformation, in which case `inst` is left unchanged.
fn do_divrem_transformation(divrem_info: &DivRemByConstInfo, pos: &mut FuncCursor, inst: Inst) {
    let is_rem = match *divrem_info {
        DivRemByConstInfo::DivU32(_, _)
        | DivRemByConstInfo::DivU64(_, _)
        | DivRemByConstInfo::DivS32(_, _)
        | DivRemByConstInfo::DivS64(_, _) => false,
        DivRemByConstInfo::RemU32(_, _)
        | DivRemByConstInfo::RemU64(_, _)
        | DivRemByConstInfo::RemS32(_, _)
        | DivRemByConstInfo::RemS64(_, _) => true,
    };

    match *divrem_info {
        // -------------------- U32 --------------------

        // U32 div, rem by zero: ignore
        DivRemByConstInfo::DivU32(_n1, 0) | DivRemByConstInfo::RemU32(_n1, 0) => {}

        // U32 div by 1: identity
        // U32 rem by 1: zero
        DivRemByConstInfo::DivU32(n1, 1) | DivRemByConstInfo::RemU32(n1, 1) => {
            if is_rem {
                pos.func.dfg.replace(inst).iconst(I32, 0);
            } else {
                replace_single_result_with_alias(&mut pos.func.dfg, inst, n1);
            }
        }

        // U32 div, rem by a power-of-2
        DivRemByConstInfo::DivU32(n1, d) | DivRemByConstInfo::RemU32(n1, d)
            if d.is_power_of_two() =>
        {
            debug_assert!(d >= 2);
            // compute k where d == 2^k
            let k = d.trailing_zeros();
            debug_assert!(k >= 1 && k <= 31);
            if is_rem {
                let mask = (1u64 << k) - 1;
                pos.func.dfg.replace(inst).band_imm(n1, mask as i64);
            } else {
                pos.func.dfg.replace(inst).ushr_imm(n1, k as i64);
            }
        }

        // U32 div, rem by non-power-of-2
        DivRemByConstInfo::DivU32(n1, d) | DivRemByConstInfo::RemU32(n1, d) => {
            debug_assert!(d >= 3);
            let MU32 {
                mul_by,
                do_add,
                shift_by,
            } = magic_u32(d);
            let qf; // final quotient
            let q0 = pos.ins().iconst(I32, mul_by as i64);
            let q1 = pos.ins().umulhi(n1, q0);
            if do_add {
                debug_assert!(shift_by >= 1 && shift_by <= 32);
                let t1 = pos.ins().isub(n1, q1);
                let t2 = pos.ins().ushr_imm(t1, 1);
                let t3 = pos.ins().iadd(t2, q1);
                // I never found any case where shift_by == 1 here.
                // So there's no attempt to fold out a zero shift.
                debug_assert_ne!(shift_by, 1);
                qf = pos.ins().ushr_imm(t3, (shift_by - 1) as i64);
            } else {
                debug_assert!(shift_by >= 0 && shift_by <= 31);
                // Whereas there are known cases here for shift_by == 0.
                if shift_by > 0 {
                    qf = pos.ins().ushr_imm(q1, shift_by as i64);
                } else {
                    qf = q1;
                }
            }
            // Now qf holds the final quotient. If necessary calculate the
            // remainder instead.
            if is_rem {
                let tt = pos.ins().imul_imm(qf, d as i64);
                pos.func.dfg.replace(inst).isub(n1, tt);
            } else {
                replace_single_result_with_alias(&mut pos.func.dfg, inst, qf);
            }
        }

        // -------------------- U64 --------------------

        // U64 div, rem by zero: ignore
        DivRemByConstInfo::DivU64(_n1, 0) | DivRemByConstInfo::RemU64(_n1, 0) => {}

        // U64 div by 1: identity
        // U64 rem by 1: zero
        DivRemByConstInfo::DivU64(n1, 1) | DivRemByConstInfo::RemU64(n1, 1) => {
            if is_rem {
                pos.func.dfg.replace(inst).iconst(I64, 0);
            } else {
                replace_single_result_with_alias(&mut pos.func.dfg, inst, n1);
            }
        }

        // U64 div, rem by a power-of-2
        DivRemByConstInfo::DivU64(n1, d) | DivRemByConstInfo::RemU64(n1, d)
            if d.is_power_of_two() =>
        {
            debug_assert!(d >= 2);
            // compute k where d == 2^k
            let k = d.trailing_zeros();
            debug_assert!(k >= 1 && k <= 63);
            if is_rem {
                let mask = (1u64 << k) - 1;
                pos.func.dfg.replace(inst).band_imm(n1, mask as i64);
            } else {
                pos.func.dfg.replace(inst).ushr_imm(n1, k as i64);
            }
        }

        // U64 div, rem by non-power-of-2
        DivRemByConstInfo::DivU64(n1, d) | DivRemByConstInfo::RemU64(n1, d) => {
            debug_assert!(d >= 3);
            let MU64 {
                mul_by,
                do_add,
                shift_by,
            } = magic_u64(d);
            let qf; // final quotient
            let q0 = pos.ins().iconst(I64, mul_by as i64);
            let q1 = pos.ins().umulhi(n1, q0);
            if do_add {
                debug_assert!(shift_by >= 1 && shift_by <= 64);
                let t1 = pos.ins().isub(n1, q1);
                let t2 = pos.ins().ushr_imm(t1, 1);
                let t3 = pos.ins().iadd(t2, q1);
                // I never found any case where shift_by == 1 here.
                // So there's no attempt to fold out a zero shift.
                debug_assert_ne!(shift_by, 1);
                qf = pos.ins().ushr_imm(t3, (shift_by - 1) as i64);
            } else {
                debug_assert!(shift_by >= 0 && shift_by <= 63);
                // Whereas there are known cases here for shift_by == 0.
                if shift_by > 0 {
                    qf = pos.ins().ushr_imm(q1, shift_by as i64);
                } else {
                    qf = q1;
                }
            }
            // Now qf holds the final quotient. If necessary calculate the
            // remainder instead.
            if is_rem {
                let tt = pos.ins().imul_imm(qf, d as i64);
                pos.func.dfg.replace(inst).isub(n1, tt);
            } else {
                replace_single_result_with_alias(&mut pos.func.dfg, inst, qf);
            }
        }

        // -------------------- S32 --------------------

        // S32 div, rem by zero or -1: ignore
        DivRemByConstInfo::DivS32(_n1, -1)
        | DivRemByConstInfo::RemS32(_n1, -1)
        | DivRemByConstInfo::DivS32(_n1, 0)
        | DivRemByConstInfo::RemS32(_n1, 0) => {}

        // S32 div by 1: identity
        // S32 rem by 1: zero
        DivRemByConstInfo::DivS32(n1, 1) | DivRemByConstInfo::RemS32(n1, 1) => {
            if is_rem {
                pos.func.dfg.replace(inst).iconst(I32, 0);
            } else {
                replace_single_result_with_alias(&mut pos.func.dfg, inst, n1);
            }
        }

        DivRemByConstInfo::DivS32(n1, d) | DivRemByConstInfo::RemS32(n1, d) => {
            if let Some((is_negative, k)) = i32_is_power_of_two(d) {
                // k can be 31 only in the case that d is -2^31.
                debug_assert!(k >= 1 && k <= 31);
                let t1 = if k - 1 == 0 {
                    n1
                } else {
                    pos.ins().sshr_imm(n1, (k - 1) as i64)
                };
                let t2 = pos.ins().ushr_imm(t1, (32 - k) as i64);
                let t3 = pos.ins().iadd(n1, t2);
                if is_rem {
                    // S32 rem by a power-of-2
                    let t4 = pos.ins().band_imm(t3, i32::wrapping_neg(1 << k) as i64);
                    // Curiously, we don't care here what the sign of d is.
                    pos.func.dfg.replace(inst).isub(n1, t4);
                } else {
                    // S32 div by a power-of-2
                    let t4 = pos.ins().sshr_imm(t3, k as i64);
                    if is_negative {
                        pos.func.dfg.replace(inst).irsub_imm(t4, 0);
                    } else {
                        replace_single_result_with_alias(&mut pos.func.dfg, inst, t4);
                    }
                }
            } else {
                // S32 div, rem by a non-power-of-2
                debug_assert!(d < -2 || d > 2);
                let MS32 { mul_by, shift_by } = magic_s32(d);
                let q0 = pos.ins().iconst(I32, mul_by as i64);
                let q1 = pos.ins().smulhi(n1, q0);
                let q2 = if d > 0 && mul_by < 0 {
                    pos.ins().iadd(q1, n1)
                } else if d < 0 && mul_by > 0 {
                    pos.ins().isub(q1, n1)
                } else {
                    q1
                };
                debug_assert!(shift_by >= 0 && shift_by <= 31);
                let q3 = if shift_by == 0 {
                    q2
                } else {
                    pos.ins().sshr_imm(q2, shift_by as i64)
                };
                let t1 = pos.ins().ushr_imm(q3, 31);
                let qf = pos.ins().iadd(q3, t1);
                // Now qf holds the final quotient. If necessary calculate
                // the remainder instead.
                if is_rem {
                    let tt = pos.ins().imul_imm(qf, d as i64);
                    pos.func.dfg.replace(inst).isub(n1, tt);
                } else {
                    replace_single_result_with_alias(&mut pos.func.dfg, inst, qf);
                }
            }
        }

        // -------------------- S64 --------------------

        // S64 div, rem by zero or -1: ignore
        DivRemByConstInfo::DivS64(_n1, -1)
        | DivRemByConstInfo::RemS64(_n1, -1)
        | DivRemByConstInfo::DivS64(_n1, 0)
        | DivRemByConstInfo::RemS64(_n1, 0) => {}

        // S64 div by 1: identity
        // S64 rem by 1: zero
        DivRemByConstInfo::DivS64(n1, 1) | DivRemByConstInfo::RemS64(n1, 1) => {
            if is_rem {
                pos.func.dfg.replace(inst).iconst(I64, 0);
            } else {
                replace_single_result_with_alias(&mut pos.func.dfg, inst, n1);
            }
        }

        DivRemByConstInfo::DivS64(n1, d) | DivRemByConstInfo::RemS64(n1, d) => {
            if let Some((is_negative, k)) = i64_is_power_of_two(d) {
                // k can be 63 only in the case that d is -2^63.
                debug_assert!(k >= 1 && k <= 63);
                let t1 = if k - 1 == 0 {
                    n1
                } else {
                    pos.ins().sshr_imm(n1, (k - 1) as i64)
                };
                let t2 = pos.ins().ushr_imm(t1, (64 - k) as i64);
                let t3 = pos.ins().iadd(n1, t2);
                if is_rem {
                    // S64 rem by a power-of-2
                    let t4 = pos.ins().band_imm(t3, i64::wrapping_neg(1 << k));
                    // Curiously, we don't care here what the sign of d is.
                    pos.func.dfg.replace(inst).isub(n1, t4);
                } else {
                    // S64 div by a power-of-2
                    let t4 = pos.ins().sshr_imm(t3, k as i64);
                    if is_negative {
                        pos.func.dfg.replace(inst).irsub_imm(t4, 0);
                    } else {
                        replace_single_result_with_alias(&mut pos.func.dfg, inst, t4);
                    }
                }
            } else {
                // S64 div, rem by a non-power-of-2
                debug_assert!(d < -2 || d > 2);
                let MS64 { mul_by, shift_by } = magic_s64(d);
                let q0 = pos.ins().iconst(I64, mul_by);
                let q1 = pos.ins().smulhi(n1, q0);
                let q2 = if d > 0 && mul_by < 0 {
                    pos.ins().iadd(q1, n1)
                } else if d < 0 && mul_by > 0 {
                    pos.ins().isub(q1, n1)
                } else {
                    q1
                };
                debug_assert!(shift_by >= 0 && shift_by <= 63);
                let q3 = if shift_by == 0 {
                    q2
                } else {
                    pos.ins().sshr_imm(q2, shift_by as i64)
                };
                let t1 = pos.ins().ushr_imm(q3, 63);
                let qf = pos.ins().iadd(q3, t1);
                // Now qf holds the final quotient. If necessary calculate
                // the remainder instead.
                if is_rem {
                    let tt = pos.ins().imul_imm(qf, d);
                    pos.func.dfg.replace(inst).isub(n1, tt);
                } else {
                    replace_single_result_with_alias(&mut pos.func.dfg, inst, qf);
                }
            }
        }
    }
}

/// Reorder branches to encourage fallthroughs.
///
/// When a block ends with a conditional branch followed by an unconditional
/// branch, this will reorder them if one of them is branching to the next Block
/// layout-wise. The unconditional jump can then become a fallthrough.
fn branch_order(pos: &mut FuncCursor, cfg: &mut ControlFlowGraph, block: Block, inst: Inst) {
    let (term_inst, term_dest, cond_inst, cond_dest, kind, cond_arg) =
        match pos.func.dfg.insts[inst] {
            InstructionData::Jump {
                opcode: Opcode::Jump,
                destination,
            } => {
                let next_block = if let Some(next_block) = pos.func.layout.next_block(block) {
                    next_block
                } else {
                    return;
                };

                if destination.block(&pos.func.dfg.value_lists) == next_block {
                    return;
                }

                let prev_inst = if let Some(prev_inst) = pos.func.layout.prev_inst(inst) {
                    prev_inst
                } else {
                    return;
                };

                match &pos.func.dfg.insts[prev_inst] {
                    &InstructionData::Branch {
                        opcode,
                        arg,
                        destination: cond_dest,
                    } => {
                        if cond_dest.block(&pos.func.dfg.value_lists) != next_block {
                            return;
                        }

                        let kind = match opcode {
                            Opcode::Brz => Opcode::Brnz,
                            Opcode::Brnz => Opcode::Brz,
                            _ => panic!("unexpected opcode"),
                        };

                        (inst, destination, prev_inst, cond_dest, kind, arg)
                    }
                    _ => return,
                }
            }

            _ => return,
        };

    pos.func
        .dfg
        .replace(term_inst)
        .Jump(Opcode::Jump, INVALID, cond_dest);
    pos.func
        .dfg
        .replace(cond_inst)
        .Branch(kind, INVALID, term_dest, cond_arg);

    cfg.recompute_block(pos.func, block);
}

mod simplify {
    use super::*;
    use crate::ir::{
        dfg::ValueDef,
        immediates,
        instructions::Opcode,
        types::{I16, I32, I8},
    };
    use std::marker::PhantomData;

    pub struct PeepholeOptimizer<'a, 'b> {
        phantom: PhantomData<(&'a (), &'b ())>,
    }

    pub fn peephole_optimizer<'a, 'b>(_: &dyn TargetIsa) -> PeepholeOptimizer<'a, 'b> {
        PeepholeOptimizer {
            phantom: PhantomData,
        }
    }

    pub fn apply_all<'a, 'b>(
        _optimizer: &mut PeepholeOptimizer<'a, 'b>,
        pos: &mut FuncCursor<'a>,
        inst: Inst,
        native_word_width: u32,
    ) {
        simplify(pos, inst, native_word_width);
        branch_opt(pos, inst);
    }

    #[inline]
    fn resolve_imm64_value(dfg: &DataFlowGraph, value: Value) -> Option<immediates::Imm64> {
        if let ValueDef::Result(candidate_inst, _) = dfg.value_def(value) {
            if let InstructionData::UnaryImm {
                opcode: Opcode::Iconst,
                imm,
            } = dfg.insts[candidate_inst]
            {
                return Some(imm);
            }
        }
        None
    }

    /// Try to transform [(x << N) >> N] into a (un)signed-extending move.
    /// Returns true if the final instruction has been converted to such a move.
    fn try_fold_extended_move(
        pos: &mut FuncCursor,
        inst: Inst,
        opcode: Opcode,
        arg: Value,
        imm: immediates::Imm64,
    ) -> bool {
        if let ValueDef::Result(arg_inst, _) = pos.func.dfg.value_def(arg) {
            if let InstructionData::BinaryImm64 {
                opcode: Opcode::IshlImm,
                arg: prev_arg,
                imm: prev_imm,
            } = &pos.func.dfg.insts[arg_inst]
            {
                if imm != *prev_imm {
                    return false;
                }

                let dest_ty = pos.func.dfg.ctrl_typevar(inst);
                if dest_ty != pos.func.dfg.ctrl_typevar(arg_inst) || !dest_ty.is_int() {
                    return false;
                }

                let imm_bits: i64 = imm.into();
                let ireduce_ty = match (dest_ty.lane_bits() as i64).wrapping_sub(imm_bits) {
                    8 => I8,
                    16 => I16,
                    32 => I32,
                    _ => return false,
                };
                let ireduce_ty = ireduce_ty.by(dest_ty.lane_count()).unwrap();

                // This becomes a no-op, since ireduce_ty has a smaller lane width than
                // the argument type (also the destination type).
                let arg = *prev_arg;
                let narrower_arg = pos.ins().ireduce(ireduce_ty, arg);

                if opcode == Opcode::UshrImm {
                    pos.func.dfg.replace(inst).uextend(dest_ty, narrower_arg);
                } else {
                    pos.func.dfg.replace(inst).sextend(dest_ty, narrower_arg);
                }
                return true;
            }
        }
        false
    }

    /// Apply basic simplifications.
    ///
    /// This folds constants with arithmetic to form `_imm` instructions, and other minor
    /// simplifications.
    ///
    /// Doesn't apply some simplifications if the native word width (in bytes) is smaller than the
    /// controlling type's width of the instruction. This would result in an illegal instruction that
    /// would likely be expanded back into an instruction on smaller types with the same initial
    /// opcode, creating unnecessary churn.
    fn simplify(pos: &mut FuncCursor, inst: Inst, native_word_width: u32) {
        match pos.func.dfg.insts[inst] {
            InstructionData::Binary { opcode, args } => {
                if let Some(mut imm) = resolve_imm64_value(&pos.func.dfg, args[1]) {
                    let new_opcode = match opcode {
                        Opcode::Iadd => Opcode::IaddImm,
                        Opcode::Imul => Opcode::ImulImm,
                        Opcode::Sdiv => Opcode::SdivImm,
                        Opcode::Udiv => Opcode::UdivImm,
                        Opcode::Srem => Opcode::SremImm,
                        Opcode::Urem => Opcode::UremImm,
                        Opcode::Band => Opcode::BandImm,
                        Opcode::Bor => Opcode::BorImm,
                        Opcode::Bxor => Opcode::BxorImm,
                        Opcode::Rotl => Opcode::RotlImm,
                        Opcode::Rotr => Opcode::RotrImm,
                        Opcode::Ishl => Opcode::IshlImm,
                        Opcode::Ushr => Opcode::UshrImm,
                        Opcode::Sshr => Opcode::SshrImm,
                        Opcode::Isub => {
                            imm = imm.wrapping_neg();
                            Opcode::IaddImm
                        }
                        _ => return,
                    };
                    let ty = pos.func.dfg.ctrl_typevar(inst);
                    if ty.bytes() <= native_word_width {
                        pos.func
                            .dfg
                            .replace(inst)
                            .BinaryImm64(new_opcode, ty, imm, args[0]);

                        // Repeat for BinaryImm simplification.
                        simplify(pos, inst, native_word_width);
                    }
                } else if let Some(imm) = resolve_imm64_value(&pos.func.dfg, args[0]) {
                    let new_opcode = match opcode {
                        Opcode::Iadd => Opcode::IaddImm,
                        Opcode::Imul => Opcode::ImulImm,
                        Opcode::Band => Opcode::BandImm,
                        Opcode::Bor => Opcode::BorImm,
                        Opcode::Bxor => Opcode::BxorImm,
                        Opcode::Isub => Opcode::IrsubImm,
                        _ => return,
                    };
                    let ty = pos.func.dfg.ctrl_typevar(inst);
                    if ty.bytes() <= native_word_width {
                        pos.func
                            .dfg
                            .replace(inst)
                            .BinaryImm64(new_opcode, ty, imm, args[1]);
                    }
                }
            }

            InstructionData::BinaryImm64 { opcode, arg, imm } => {
                let ty = pos.func.dfg.ctrl_typevar(inst);

                let mut arg = arg;
                let mut imm = imm;
                match opcode {
                    Opcode::IaddImm
                    | Opcode::ImulImm
                    | Opcode::BorImm
                    | Opcode::BandImm
                    | Opcode::BxorImm => {
                        // Fold binary_op(C2, binary_op(C1, x)) into binary_op(binary_op(C1, C2), x)
                        if let ValueDef::Result(arg_inst, _) = pos.func.dfg.value_def(arg) {
                            if let InstructionData::BinaryImm64 {
                                opcode: prev_opcode,
                                arg: prev_arg,
                                imm: prev_imm,
                            } = &pos.func.dfg.insts[arg_inst]
                            {
                                if opcode == *prev_opcode
                                    && ty == pos.func.dfg.ctrl_typevar(arg_inst)
                                {
                                    let lhs: i64 = imm.into();
                                    let rhs: i64 = (*prev_imm).into();
                                    let new_imm = match opcode {
                                        Opcode::BorImm => lhs | rhs,
                                        Opcode::BandImm => lhs & rhs,
                                        Opcode::BxorImm => lhs ^ rhs,
                                        Opcode::IaddImm => lhs.wrapping_add(rhs),
                                        Opcode::ImulImm => lhs.wrapping_mul(rhs),
                                        _ => panic!("can't happen"),
                                    };
                                    let new_imm = immediates::Imm64::from(new_imm);
                                    let new_arg = *prev_arg;
                                    pos.func
                                        .dfg
                                        .replace(inst)
                                        .BinaryImm64(opcode, ty, new_imm, new_arg);
                                    imm = new_imm;
                                    arg = new_arg;
                                }
                            }
                        }
                    }

                    Opcode::UshrImm | Opcode::SshrImm => {
                        if pos.func.dfg.ctrl_typevar(inst).bytes() <= native_word_width
                            && try_fold_extended_move(pos, inst, opcode, arg, imm)
                        {
                            return;
                        }
                    }

                    _ => {}
                };

                // Replace operations that are no-ops.
                match (opcode, imm.into(), ty) {
                    (Opcode::IaddImm, 0, _)
                    | (Opcode::ImulImm, 1, _)
                    | (Opcode::SdivImm, 1, _)
                    | (Opcode::UdivImm, 1, _)
                    | (Opcode::BorImm, 0, _)
                    | (Opcode::BandImm, -1, _)
                    | (Opcode::BxorImm, 0, _)
                    | (Opcode::RotlImm, 0, _)
                    | (Opcode::RotrImm, 0, _)
                    | (Opcode::IshlImm, 0, _)
                    | (Opcode::UshrImm, 0, _)
                    | (Opcode::SshrImm, 0, _) => {
                        // Alias the result value with the original argument.
                        replace_single_result_with_alias(&mut pos.func.dfg, inst, arg);
                    }
                    (Opcode::ImulImm, 0, ty) | (Opcode::BandImm, 0, ty) if ty != I128 => {
                        // Replace by zero.
                        pos.func.dfg.replace(inst).iconst(ty, 0);
                    }
                    (Opcode::BorImm, -1, ty) if ty != I128 => {
                        // Replace by minus one.
                        pos.func.dfg.replace(inst).iconst(ty, -1);
                    }
                    _ => {}
                }
            }

            InstructionData::IntCompare { opcode, cond, args } => {
                debug_assert_eq!(opcode, Opcode::Icmp);
                if let Some(imm) = resolve_imm64_value(&pos.func.dfg, args[1]) {
                    if pos.func.dfg.ctrl_typevar(inst).bytes() <= native_word_width {
                        pos.func.dfg.replace(inst).icmp_imm(cond, args[0], imm);
                    }
                }
            }

            _ => {}
        }
    }

    /// Fold comparisons into branch operations when possible.
    ///
    /// This matches against operations which compare against zero, then use the
    /// result in a `brz` or `brnz` branch. It folds those two operations into a
    /// single `brz` or `brnz`.
    fn branch_opt(pos: &mut FuncCursor, inst: Inst) {
        let (cmp_arg, new_opcode) = if let InstructionData::Branch {
            opcode: br_opcode,
            arg: first_arg,
            ..
        } = pos.func.dfg.insts[inst]
        {
            let icmp_inst =
                if let ValueDef::Result(icmp_inst, _) = pos.func.dfg.value_def(first_arg) {
                    icmp_inst
                } else {
                    return;
                };

            if let InstructionData::IntCompareImm {
                opcode: Opcode::IcmpImm,
                arg: cmp_arg,
                cond: cmp_cond,
                imm: cmp_imm,
            } = pos.func.dfg.insts[icmp_inst]
            {
                let cmp_imm: i64 = cmp_imm.into();
                if cmp_imm != 0 {
                    return;
                }

                // icmp_imm returns non-zero when the comparison is true. So, if
                // we're branching on zero, we need to invert the condition.
                let cond = match br_opcode {
                    Opcode::Brz => cmp_cond.inverse(),
                    Opcode::Brnz => cmp_cond,
                    _ => return,
                };

                let new_opcode = match cond {
                    IntCC::Equal => Opcode::Brz,
                    IntCC::NotEqual => Opcode::Brnz,
                    _ => return,
                };

                (cmp_arg, new_opcode)
            } else {
                return;
            }
        } else {
            return;
        };

        if let InstructionData::Branch { opcode, arg, .. } = &mut pos.func.dfg.insts[inst] {
            *opcode = new_opcode;
            *arg = cmp_arg;
        } else {
            unreachable!();
        }
    }
}

/// The main pre-opt pass.
pub fn do_preopt(func: &mut Function, cfg: &mut ControlFlowGraph, isa: &dyn TargetIsa) {
    let _tt = timing::preopt();

    let mut pos = FuncCursor::new(func);
    let native_word_width = isa.pointer_bytes() as u32;
    let mut optimizer = simplify::peephole_optimizer(isa);

    while let Some(block) = pos.next_block() {
        while let Some(inst) = pos.next_inst() {
            simplify::apply_all(&mut optimizer, &mut pos, inst, native_word_width);

            // Try to transform divide-by-constant into simpler operations.
            if let Some(divrem_info) = get_div_info(inst, &pos.func.dfg) {
                do_divrem_transformation(&divrem_info, &mut pos, inst);
                continue;
            }

            branch_order(&mut pos, cfg, block, inst);
        }
    }
    drop(pos);

    try_fold_multi3(func);
}

fn try_fold_multi3(func: &mut Function) -> Option<()> {
    if func.signature.params.len() != 7 {
        return None;
    }

    if func.dfg.num_blocks() != 2 {
        return None;
    }

    use crate::ir::*;

    if &func.signature.params
        != &[
            AbiParam {
                value_type: I64,
                purpose: ArgumentPurpose::VMContext,
                extension: ArgumentExtension::None,
                legalized_to_pointer: false,
            },
            AbiParam {
                value_type: I64,
                purpose: ArgumentPurpose::Normal,
                extension: ArgumentExtension::None,
                legalized_to_pointer: false,
            },
            AbiParam {
                value_type: I32,
                purpose: ArgumentPurpose::Normal,
                extension: ArgumentExtension::None,
                legalized_to_pointer: false,
            },
            AbiParam {
                value_type: I64,
                purpose: ArgumentPurpose::Normal,
                extension: ArgumentExtension::None,
                legalized_to_pointer: false,
            },
            AbiParam {
                value_type: I64,
                purpose: ArgumentPurpose::Normal,
                extension: ArgumentExtension::None,
                legalized_to_pointer: false,
            },
            AbiParam {
                value_type: I64,
                purpose: ArgumentPurpose::Normal,
                extension: ArgumentExtension::None,
                legalized_to_pointer: false,
            },
            AbiParam {
                value_type: I64,
                purpose: ArgumentPurpose::Normal,
                extension: ArgumentExtension::None,
                legalized_to_pointer: false,
            },
        ]
    {
        return None;
    }

    // we will try to match based on forward-analysis,
    // and ABI knowledge: that v3 - v6 inclusive are our i128

    // we expect a preoptimized and somewhat normalized body to avoid peeking too deep
    // into graphs

    let mut pos = FuncCursor::new(func);
    let main_body_block = pos.next_block().unwrap();
    if pos.func.dfg.num_insts() > 40 {
        // bail out
        return None;
    }

    let start_of_block_position = pos.position();

    let main_block_params = pos.func.dfg.block_params(main_body_block);
    let x_low = main_block_params[3];
    let x_high = main_block_params[4];

    let y_low = main_block_params[5];
    let y_high = main_block_params[6];

    // now some heuristics. We are lucky that x and y are summetric, so roughly follow the same path

    fn find_use_in_binop_with_imm(
        pos: &mut FuncCursor,
        value: Value,
        opcode: Opcode,
        imm_value: i64,
    ) -> Option<Inst> {
        while let Some(inst) = pos.next_inst() {
            match pos.func.dfg[inst] {
                InstructionData::BinaryImm64 {
                    opcode: op,
                    arg,
                    imm,
                } if op == opcode && imm.bits() == imm_value && arg == value => return Some(inst),
                _ => continue,
            }
        }

        None
    }

    fn find_binop_with_inputs(
        pos: &mut FuncCursor,
        v0: Value,
        v1: Value,
        opcode: Opcode,
    ) -> Option<Inst> {
        while let Some(inst) = pos.next_inst() {
            match pos.func.dfg[inst] {
                InstructionData::Binary { opcode: op, args } if op == opcode => {
                    if args == [v0, v1] || args == [v1, v0] {
                        return Some(inst);
                    }
                }
                _ => continue,
            }
        }

        None
    }

    fn find_compare_with_inputs(
        pos: &mut FuncCursor,
        v0: Value,
        v1: Value,
        opcode: Opcode,
        condition: IntCC,
    ) -> Option<Inst> {
        while let Some(inst) = pos.next_inst() {
            match pos.func.dfg[inst] {
                InstructionData::IntCompare {
                    opcode: op,
                    args,
                    cond,
                } if op == opcode && cond == condition => {
                    if args == [v0, v1] || args == [v1, v0] {
                        return Some(inst);
                    }
                }
                _ => continue,
            }
        }

        None
    }

    fn find_unary_op(pos: &mut FuncCursor, v0: Value, opcode: Opcode) -> Option<Inst> {
        while let Some(inst) = pos.next_inst() {
            match pos.func.dfg[inst] {
                InstructionData::Unary { opcode: op, arg } if op == opcode && arg == v0 => {
                    return Some(inst)
                }
                _ => continue,
            }
        }

        None
    }

    fn find_store(
        pos: &mut FuncCursor,
        v0: Value,
        offset: Offset32,
        flags: MemFlags,
    ) -> Option<Inst> {
        while let Some(inst) = pos.next_inst() {
            match pos.func.dfg[inst] {
                InstructionData::Store {
                    args,
                    flags: fl,
                    offset: of,
                    opcode: Opcode::Store,
                } if args.len() == 2 && args[0] == v0 && of == offset && fl == flags => {
                    return Some(inst)
                }
                _ => continue,
            }
        }

        None
    }

    // find lowest product
    let x_low_low_inst =
        find_use_in_binop_with_imm(&mut pos, x_low, Opcode::BandImm, 0x0000_0000_ffff_ffff)?;
    pos.set_position(start_of_block_position);
    let y_low_low_inst =
        find_use_in_binop_with_imm(&mut pos, y_low, Opcode::BandImm, 0x0000_0000_ffff_ffff)?;
    pos.set_position(start_of_block_position);
    let x_low_low = pos.func.dfg.first_result(x_low_low_inst);
    let y_low_low = pos.func.dfg.first_result(y_low_low_inst);
    // do not restore position, as it can not be before inputs are materialized
    let result_low_low_product_inst =
        find_binop_with_inputs(&mut pos, x_low_low, y_low_low, Opcode::Imul)?;
    let result_low_low_product = pos.func.dfg.first_result(result_low_low_product_inst);

    // find cross products for low part of result
    pos.set_position(start_of_block_position);
    let x_low_high_inst = find_use_in_binop_with_imm(&mut pos, x_low, Opcode::UshrImm, 32)?;
    let x_low_high = pos.func.dfg.first_result(x_low_high_inst);
    let x_low_high_y_low_low_product_inst =
        find_binop_with_inputs(&mut pos, x_low_high, y_low_low, Opcode::Imul)?;
    let x_low_high_y_low_low_product = pos.func.dfg.first_result(x_low_high_y_low_low_product_inst);

    // 2nd one
    pos.set_position(start_of_block_position);
    let y_low_high_inst = find_use_in_binop_with_imm(&mut pos, y_low, Opcode::UshrImm, 32)?;
    let y_low_high = pos.func.dfg.first_result(y_low_high_inst);
    let pos_x_y_low_high_are_ready = pos.position();

    let x_low_low_y_low_high_product_inst =
        find_binop_with_inputs(&mut pos, x_low_low, y_low_high, Opcode::Imul)?;
    let x_low_low_y_low_high_product = pos.func.dfg.first_result(x_low_low_y_low_high_product_inst);

    // final result of high
    let result_low_high_inst = find_binop_with_inputs(
        &mut pos,
        x_low_high_y_low_low_product,
        x_low_low_y_low_high_product,
        Opcode::Iadd,
    )?;
    let result_low_high = pos.func.dfg.first_result(result_low_high_inst);

    let pos_result_low_high_ready = pos.position();

    // result_low_high is shifted left by 32 and added to the low-low product
    let result_low_high_shl_inst =
        find_use_in_binop_with_imm(&mut pos, result_low_high, Opcode::IshlImm, 32)?;
    let result_low_high_shl = pos.func.dfg.first_result(result_low_high_shl_inst);

    // final addition to get lower libm
    let result_low_limb_inst = find_binop_with_inputs(
        &mut pos,
        result_low_low_product,
        result_low_high_shl,
        Opcode::Iadd,
    )?;
    let result_low_limb = pos.func.dfg.first_result(result_low_limb_inst);

    let pos_lower_limb_ready = pos.position();

    // continue the walk, now we need to find carry propagation and higher multiplication

    // find the cross product
    pos.set_position(pos_x_y_low_high_are_ready);
    let result_high_low_contribution_from_xy_low_high_mul_inst =
        find_binop_with_inputs(&mut pos, x_low_high, y_low_high, Opcode::Imul)?;
    let result_high_low_contribution_from_xy_low_high_mul = pos
        .func
        .dfg
        .first_result(result_high_low_contribution_from_xy_low_high_mul_inst);

    // this carry propagations via comparison and sign extensions
    pos.set_position(pos_x_y_low_high_are_ready);
    let from_low_high_cross_product_carry_inst = if let Some(inst) = find_compare_with_inputs(
        &mut pos,
        result_low_high,
        x_low_high_y_low_low_product,
        Opcode::Icmp,
        IntCC::UnsignedLessThan,
    ) {
        inst
    } else {
        pos.set_position(pos_x_y_low_high_are_ready);
        if let Some(inst) = find_compare_with_inputs(
            &mut pos,
            result_low_high,
            x_low_low_y_low_high_product,
            Opcode::Icmp,
            IntCC::UnsignedLessThan,
        ) {
            inst
        } else {
            return None;
        }
    };

    let from_low_high_cross_product_carry = pos
        .func
        .dfg
        .first_result(from_low_high_cross_product_carry_inst);
    let carry_cast_inst = find_unary_op(&mut pos, from_low_high_cross_product_carry, Opcode::Bint)?;
    let carry_cast = pos.func.dfg.first_result(carry_cast_inst);
    let carry_uext_inst = find_unary_op(&mut pos, carry_cast, Opcode::Uextend)?;
    let carry_uext = pos.func.dfg.first_result(carry_uext_inst);
    let carry_shifted_inst = find_use_in_binop_with_imm(&mut pos, carry_uext, Opcode::IshlImm, 32)?;
    let carry_shifted = pos.func.dfg.first_result(carry_shifted_inst);

    // find result_low_high >> 32
    pos.set_position(pos_result_low_high_ready);
    let result_low_high_ushr32_inst =
        find_use_in_binop_with_imm(&mut pos, result_low_high, Opcode::UshrImm, 32)?;
    let result_low_high_ushr32 = pos.func.dfg.first_result(result_low_high_ushr32_inst);

    let combine_carry_and_result_low_high_inst =
        find_binop_with_inputs(&mut pos, carry_shifted, result_low_high_ushr32, Opcode::Bor)?;
    let combine_carry_and_result_low_high = pos
        .func
        .dfg
        .first_result(combine_carry_and_result_low_high_inst);

    let combine_0_inst = find_binop_with_inputs(
        &mut pos,
        result_high_low_contribution_from_xy_low_high_mul,
        combine_carry_and_result_low_high,
        Opcode::Iadd,
    )?;
    let combine_0 = pos.func.dfg.first_result(combine_0_inst);

    // carry into lower part
    pos.set_position(pos_lower_limb_ready);
    let carry_inst = find_compare_with_inputs(
        &mut pos,
        result_low_low_product,
        result_low_limb,
        Opcode::Icmp,
        IntCC::UnsignedLessThan,
    )?;
    let carry = pos.func.dfg.first_result(carry_inst);
    let carry_cast_inst = find_unary_op(&mut pos, carry, Opcode::Bint)?;
    let carry_cast = pos.func.dfg.first_result(carry_cast_inst);
    let carry_uext_inst = find_unary_op(&mut pos, carry_cast, Opcode::Uextend)?;
    let carry_uext = pos.func.dfg.first_result(carry_uext_inst);

    let combine_1_inst = find_binop_with_inputs(&mut pos, carry_uext, combine_0, Opcode::Iadd)?;
    let combine_1 = pos.func.dfg.first_result(combine_1_inst);

    pos.set_position(start_of_block_position);
    let x_low_y_high_inst = find_binop_with_inputs(&mut pos, x_low, y_high, Opcode::Imul)?;
    let x_low_y_high = pos.func.dfg.first_result(x_low_y_high_inst);

    pos.set_position(start_of_block_position);
    let x_high_y_low_inst = find_binop_with_inputs(&mut pos, x_high, y_low, Opcode::Imul)?;
    let x_high_y_low = pos.func.dfg.first_result(x_high_y_low_inst);

    let combine_2_inst =
        find_binop_with_inputs(&mut pos, x_low_y_high, x_high_y_low, Opcode::Iadd)?;
    let combine_2 = pos.func.dfg.first_result(combine_2_inst);

    let result_high_limb_inst =
        find_binop_with_inputs(&mut pos, combine_1, combine_2, Opcode::Iadd)?;
    let result_high_limb = pos.func.dfg.first_result(result_high_limb_inst);

    let _pos_higher_limb_ready = pos.position();

    // insert before everything

    pos.set_position(start_of_block_position);
    pos.next_inst();

    let x_full = pos.ins().iconcat(x_low, x_high);
    let y_full = pos.ins().iconcat(y_low, y_high);
    let result = pos.ins().imul(x_full, y_full);
    let (result_low, result_high) = pos.ins().isplit(result);

    let mut storage_flags = MemFlags::new();
    storage_flags.set_endianness(Endianness::Little);

    pos.set_position(pos_lower_limb_ready);
    let store_result_low_inst =
        find_store(&mut pos, result_low_limb, Offset32::new(0), storage_flags)?;
    pos.func.dfg.inst_fixed_args_mut(store_result_low_inst)[0] = result_low;
    let store_result_high_inst =
        find_store(&mut pos, result_high_limb, Offset32::new(8), storage_flags)?;
    pos.func.dfg.inst_fixed_args_mut(store_result_high_inst)[0] = result_high;

    return Some(());
}
