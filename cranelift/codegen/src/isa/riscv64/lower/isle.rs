//! ISLE integration glue code for riscv64 lowering.

// Pull in the ISLE generated code.
pub mod generated_code;

use alloc::vec::Vec;

// Types that the generated ISLE code uses via `use super::*`.
use super::{writable_zero_reg, zero_reg, Inst as MInst};

use crate::isa::riscv64::settings::Flags as IsaFlags;
use crate::machinst::{isle::*, MachInst, SmallInstVec};
use crate::settings::Flags;

use crate::machinst::{VCodeConstant, VCodeConstantData};
use crate::{
    ir::{
        immediates::*, types::*, AtomicRmwOp, ExternalName, Inst, InstructionData, MemFlags,
        StackSlot, TrapCode, Value, ValueList,
    },
    isa::riscv64::inst::*,
    machinst::{InsnOutput, LowerCtx},
};

use std::boxed::Box;
use std::convert::TryFrom;

use crate::machinst::Reg;

type BoxCallInfo = Box<CallInfo>;
type BoxCallIndInfo = Box<CallIndInfo>;
type BoxExternalName = Box<ExternalName>;

pub(crate) const F32_POS_1: u64 = 0x3f800000;
pub(crate) const F64_POS_1: u64 = 0x3ff0000000000000;

/// The main entry point for lowering with ISLE.
pub(crate) fn lower<C>(
    lower_ctx: &mut C,
    flags: &Flags,
    isa_flags: &IsaFlags,
    outputs: &[InsnOutput],
    inst: Inst,
) -> Result<(), ()>
where
    C: LowerCtx<I = MInst>,
{
    lower_common(lower_ctx, flags, isa_flags, outputs, inst, |cx, insn| {
        generated_code::constructor_lower(cx, insn)
    })
}

impl<C> generated_code::Context for IsleContext<'_, C, Flags, IsaFlags, 6>
where
    C: LowerCtx<I = MInst>,
{
    isle_prelude_methods!();
    fn vec_regs_to_value_regs(&mut self, val: &VecWritableReg) -> ValueRegs {
        match val.len() {
            1 => ValueRegs::one(val[0].to_reg()),
            2 => ValueRegs::two(val[0].to_reg(), val[1].to_reg()),
            _ => unreachable!(),
        }
    }

    fn valid_bextend_ty(&mut self, from: Type, to: Type) -> Option<Unit> {
        if from.is_bool() && to.is_bool() && to.bits() < 128 && from.bits() < to.bits() {
            Some(())
        } else {
            None
        }
    }
    fn gen_return(&mut self, val: ValueSlice) -> InstOutput {
        // due to owernship error I have to clone ssa_values.
        let ssa_values: Vec<_> = val
            .0
            .as_slice(&self.lower_ctx.dfg().value_lists)
            .iter()
            .map(|v| *v)
            .collect();

        for (i, ssa_value) in ssa_values.iter().enumerate() {
            let src_reg = self.lower_ctx.put_value_in_regs(*ssa_value);
            let retval_reg = self.lower_ctx.retval(i);
            let ty = self.lower_ctx.value_ty(*ssa_value);
            assert!(src_reg.len() == retval_reg.len());
            for (&src, &dst) in src_reg.regs().iter().zip(retval_reg.regs().iter()) {
                let ty = MInst::canonical_type_for_rc(src.class());
                self.emit(&MInst::gen_move(dst, src, ty));
            }
        }
        InstOutput::default()
    }
    fn vec_writable_clone(&mut self, v: &VecWritableReg) -> VecWritableReg {
        v.clone()
    }
    fn value_inst(&mut self, val: Value) -> Option<Inst> {
        let may = self.lower_ctx.get_value_as_source_or_const(val);
        may.inst.as_inst().map(|(insn, index)| insn)
    }

    fn gen_ceil(&mut self, rs: Reg, ty: Type) -> Reg {
        unimplemented!()
    }
    fn gen_floor(&mut self, rs: Reg, ty: Type) -> Reg {
        unimplemented!()
    }
    fn gen_trunc(&mut self, rs: Reg, ty: Type) -> Reg {
        unimplemented!()
    }
    fn gen_nearest(&mut self, rs: Reg, ty: Type) -> Reg {
        unimplemented!()
    }

    fn gen_moves(&mut self, rs: ValueRegs, in_ty: Type, out_ty: Type) -> ValueRegs {
        match (in_ty.is_vector(), out_ty.is_vector()) {
            (true, true) => todo!(),
            (true, false) => todo!(),
            (false, true) => todo!(),
            (false, false) => {
                assert!(in_ty.bits() == out_ty.bits());
                match in_ty.bits() {
                    128 => {
                        // if 128 must not be a float.
                        let low = self.temp_writable_reg(out_ty);
                        let high = self.temp_writable_reg(out_ty);
                        self.emit(&gen_move(low, I64, rs.regs()[0], I64));
                        self.emit(&gen_move(high, I64, rs.regs()[1], I64));
                        ValueRegs::two(low.to_reg(), high.to_reg())
                    }
                    _ => {
                        let rd = self.temp_writable_reg(out_ty);
                        self.emit(&gen_move(rd, out_ty, rs.regs()[0], in_ty));
                        ValueRegs::one(rd.to_reg())
                    }
                }
            }
        }
    }
    fn con_vec_writable(&mut self, ty: Type) -> VecWritableReg {
        if ty.is_int() {
            if ty.bits() <= 64 {
                vec![self.temp_writable_reg(I64)]
            } else {
                vec![self.temp_writable_reg(I64), self.temp_writable_reg(I64)]
            }
        } else if ty.is_float() {
            vec![self.temp_writable_reg(ty)]
        } else {
            unimplemented!("ty:{:?}", ty)
        }
    }

    fn imm(&mut self, ty: Type, mut val: u64) -> Reg {
        /*
        Boolean types
        Boolean values are either true or false.

        The b1 type represents an abstract boolean value. It can only exist as an SSA value, and can't be directly stored in memory. It can, however, be converted into an integer with value 0 or 1 by the bint instruction (and converted back with icmp_imm with 0).

        Several larger boolean types are also defined, primarily to be used as SIMD element types. They can be stored in memory, and are represented as either all zero bits or all one bits.

        b1
        b8
        b16
        b32
        b64
        ///////////////////////////////////////////////////////////
        "represented as either all zero bits or all one bits."
        \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
                        */
        if ty.is_bool() && val != 0 {
            // need all be one
            val = !0;
        }
        let tmp = self.temp_writable_reg(ty);
        self.emit_list(&MInst::load_constant_u64(tmp, val));
        tmp.to_reg()
    }
    #[inline(always)]
    fn emit(&mut self, arg0: &MInst) -> Unit {
        self.lower_ctx.emit(arg0.clone());
    }
    #[inline(always)]
    fn imm12_from_u64(&mut self, arg0: u64) -> Option<Imm12> {
        Imm12::maybe_from_u64(arg0)
    }
    #[inline(always)]
    fn writable_zero_reg(&mut self) -> WritableReg {
        writable_zero_reg()
    }
    #[inline(always)]
    fn neg_imm12(&mut self, arg0: Imm12) -> Imm12 {
        -arg0
    }
    #[inline(always)]
    fn zero_reg(&mut self) -> Reg {
        zero_reg()
    }
    #[inline(always)]
    fn imm_from_bits(&mut self, val: u64) -> Imm12 {
        Imm12::maybe_from_u64(val).unwrap()
    }
    #[inline(always)]
    fn imm_from_neg_bits(&mut self, val: i64) -> Imm12 {
        Imm12::maybe_from_u64(val as u64).unwrap()
    }

    fn gen_default_frm(&mut self) -> OptionFloatRoundingMode {
        None
    }
    fn con_select_reg(&mut self, cc: &IntCC, a: Reg, b: Reg, rs1: Reg, rs2: Reg) -> Reg {
        let rd = self.temp_writable_reg(MInst::canonical_type_for_rc(rs1.class()));
        self.emit(&MInst::SelectReg {
            rd,
            rs1,
            rs2,
            condition: IntegerCompare {
                kind: *cc,
                rs1: a,
                rs2: b,
            },
        });
        rd.to_reg()
    }
    fn load_u64_constant(&mut self, val: u64) -> Reg {
        let rd = self.temp_writable_reg(I64);
        MInst::load_constant_u64(rd, val)
            .iter()
            .for_each(|i| self.emit(i));
        rd.to_reg()
    }
    fn u8_as_i32(&mut self, x: u8) -> i32 {
        x as i32
    }

    fn ext_sign_bit(&mut self, ty: Type, r: Reg) -> Reg {
        assert!(ty.is_int());
        let rd = self.temp_writable_reg(I64);
        self.emit(&MInst::AluRRImm12 {
            alu_op: AluOPRRI::Bexti,
            rd,
            rs: r,
            imm12: Imm12::from_bits((ty.bits() - 1) as i16),
        });
        rd.to_reg()
    }
    fn imm12_const(&mut self, val: i32) -> Imm12 {
        Imm12::maybe_from_u64(val as u64).unwrap()
    }
    fn imm12_const_add(&mut self, val: i32, add: i32) -> Imm12 {
        Imm12::maybe_from_u64((val + add) as u64).unwrap()
    }

    //
    fn con_shamt(&mut self, ty: Type, shamt: Reg) -> ValueRegs {
        let shamt = {
            let tmp = self.temp_writable_reg(I64);
            self.emit(&MInst::AluRRImm12 {
                alu_op: AluOPRRI::Andi,
                rd: tmp,
                rs: shamt,
                imm12: Imm12::from_bits((ty.bits() - 1) as i16),
            });
            tmp.to_reg()
        };
        let len_sub_shamt = {
            let len_sub_shamt = self.temp_writable_reg(I64);
            self.emit(&MInst::load_constant_imm12(
                len_sub_shamt,
                Imm12::from_bits(ty.bits() as i16),
            ));
            self.emit(&MInst::AluRRR {
                alu_op: AluOPRRR::Sub,
                rd: len_sub_shamt,
                rs1: len_sub_shamt.to_reg(),
                rs2: shamt,
            });
            len_sub_shamt.to_reg()
        };
        ValueRegs::two(shamt, len_sub_shamt)
    }

    fn valueregs_2_reg(&mut self, val: Value) -> Reg {
        self.put_in_regs(val).regs()[0]
    }

    fn ifcmp_parameters(&mut self, val: Value) -> Option<(Value, Value, Type)> {
        let inst = self.value_inst(val);
        let inst = match inst {
            Some(x) => x,
            None => return None,
        };
        let opcode = self.lower_ctx.data(inst).opcode();
        if opcode != crate::ir::Opcode::Ifcmp {
            return None;
        }
        let a = self.lower_ctx.input_as_value(inst, 0);
        let b = self.lower_ctx.input_as_value(inst, 1);
        let ty = self.lower_ctx.input_ty(inst, 0);
        Some((a, b, ty))
    }
    fn ffcmp_parameters(&mut self, val: Value) -> Option<(Value, Value, Type)> {
        let inst = self.value_inst(val);
        let inst = match inst {
            Some(x) => x,
            None => return None,
        };
        let opcode = self.lower_ctx.data(inst).opcode();
        if opcode != crate::ir::Opcode::Ffcmp {
            return None;
        }
        let a = self.lower_ctx.input_as_value(inst, 0);
        let b = self.lower_ctx.input_as_value(inst, 1);
        let ty = self.lower_ctx.input_ty(inst, 0);
        Some((a, b, ty))
    }
    fn load_float_const(&mut self, val: u64, ty: Type) -> Reg {
        let result = self.temp_writable_reg(ty);
        if ty == F32 {
            MInst::load_fp_constant32(result, val as u32)
                .into_iter()
                .for_each(|i| self.emit(&i));
        } else if ty == F64 {
            MInst::load_fp_constant64(result, val)
                .into_iter()
                .for_each(|i| self.emit(&i));
        } else {
            unimplemented!()
        }
        result.to_reg()
    }
    fn move_f_to_x(&mut self, r: Reg, ty: Type) -> Reg {
        let result = self.temp_writable_reg(I64);
        self.emit(&gen_move(result, I64, r, ty));
        result.to_reg()
    }
    fn offset32_imm(&mut self, offset: i32) -> Offset32 {
        Offset32::new(offset)
    }
    fn default_memflags(&mut self) -> MemFlags {
        MemFlags::new()
    }
    fn move_x_to_f(&mut self, r: Reg, ty: Type) -> Reg {
        let result = self.temp_writable_reg(ty);
        self.emit(&gen_move(result, ty, r, I64));
        result.to_reg()
    }

    fn pack_float_rounding_mode(&mut self, f: &FRM) -> OptionFloatRoundingMode {
        Some(*f)
    }
    fn float_convert_2_int_op(&mut self, from: Type, is_signed: bool, to: Type) -> FpuOPRR {
        FpuOPRR::float_convert_2_int_op(from, is_signed, to)
    }
    fn int_convert_2_float_op(&mut self, from: Type, is_signed: bool, to: Type) -> FpuOPRR {
        FpuOPRR::int_convert_2_float_op(from, is_signed, to)
    }
    fn con_amode(&mut self, base: Reg, offset: Offset32, ty: Type) -> AMode {
        AMode::RegOffset(base, i64::from(offset), ty)
    }
    fn valid_atomic_transaction(&mut self, ty: Type) -> Option<Type> {
        if ty == I32 || ty == I64 {
            Some(ty)
        } else {
            None
        }
    }

    fn load_gv_addr(&mut self, gv: GlobalValue) -> Reg {
        unimplemented!()
    }
    fn load_op(&mut self, ty: Type) -> LoadOP {
        LoadOP::from_type(ty)
    }
    fn store_op(&mut self, ty: Type) -> StoreOP {
        StoreOP::from_type(ty)
    }
    fn load_ext_name(&mut self, name: ExternalName, offset: i64) -> Reg {
        let tmp = self.temp_writable_reg(I64);
        self.emit(&MInst::LoadExtName {
            rd: tmp,
            name: Box::new(name),
            offset,
        });
        tmp.to_reg()
    }
    fn offset32_add(&mut self, a: Offset32, adden: i64) -> Offset32 {
        a.try_add_i64(adden).expect("offset exceed range.")
    }
    fn type_and_value(&mut self, val: Value) -> (Type, Value) {
        let ty = self.lower_ctx.value_ty(val);
        (ty, val)
    }
    fn gen_stack_addr(&mut self, slot: StackSlot, offset: Offset32) -> Reg {
        let result = self.temp_writable_reg(I64);
        let i = self
            .lower_ctx
            .abi()
            .stackslot_addr(slot, i64::from(offset) as u32, result);
        self.emit(&i);
        result.to_reg()
    }
    fn atomic_amo(&mut self) -> AMO {
        AMO::SeqCst
    }
    fn gen_move(&mut self, r: Reg, ty: Type) -> Reg {
        let tmp = self.temp_writable_reg(ty);
        self.emit(&gen_move(tmp, ty, r, ty));
        tmp.to_reg()
    }

    fn sext_int_if_need(&mut self, val: Value) -> Option<(ValueRegs, Type)> {
        let (ty, val) = self.type_and_value(val);
        if !ty.is_int() {
            return None;
        }
        match ty.bits() {
            128 => {
                let r = self.put_in_regs(val);
                Some((r, ty))
            }
            64 => {
                let r = self.put_in_reg(val);
                Some((ValueRegs::one(r), ty))
            }
            _ => {
                let rs = self.put_in_reg(val);
                let rd = self.temp_writable_reg(I64);
                self.emit(&MInst::Extend {
                    rd,
                    rn: rs,
                    signed: true,
                    from_bits: ty.bits() as u8,
                    to_bits: 64,
                });

                Some((ValueRegs::one(rd.to_reg()), ty))
            }
        }
    }

    fn intcc_is_signed(&mut self, cc: &IntCC) -> Option<(IntCC, bool)> {
        let cc = *cc;
        match cc {
            IntCC::SignedLessThan => Some((cc, true)),
            IntCC::SignedGreaterThanOrEqual => Some((cc, true)),
            IntCC::SignedGreaterThan => Some((cc, true)),
            IntCC::SignedLessThanOrEqual => Some((cc, true)),
            //
            IntCC::UnsignedLessThan => Some((cc, false)),
            IntCC::UnsignedGreaterThanOrEqual => Some((cc, false)),
            IntCC::UnsignedGreaterThan => Some((cc, false)),
            IntCC::UnsignedLessThanOrEqual => Some((cc, false)),
            _ => None,
        }
    }
    fn intcc_is_eq_or_ne(&mut self, cc: &IntCC) -> Option<IntCC> {
        let cc = *cc;
        if cc == IntCC::Equal || cc == IntCC::NotEqual {
            Some(cc)
        } else {
            None
        }
    }

    fn intcc_is_overflow_or_nof(&mut self, cc: &IntCC) -> Option<IntCC> {
        let cc = *cc;
        if cc == IntCC::Overflow || cc == IntCC::NotOverflow {
            Some(cc)
        } else {
            None
        }
    }
}

impl<C> IsleContext<'_, C, Flags, IsaFlags, 6>
where
    C: LowerCtx<I = MInst>,
{
    #[inline(always)]
    fn emit_list(&mut self, list: &SmallInstVec<MInst>) {
        for i in list {
            self.lower_ctx.emit(i.clone());
        }
    }
}
