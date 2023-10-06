//! Proof-carrying code checking for AArch64 VCode.

use crate::ir::pcc::*;
use crate::ir::MemFlags;
use crate::isa::aarch64::inst::args::PairAMode;
use crate::isa::aarch64::inst::Inst;
use crate::isa::aarch64::inst::{AMode, ExtendOp};
use crate::machinst::Reg;
use crate::machinst::VCode;
use crate::trace;

pub(crate) fn check(inst: &Inst, vcode: &VCode<Inst>) -> PccResult<()> {
    trace!("Checking facts on inst: {:?}", inst);

    match inst {
        Inst::Args { .. } => {
            // Defs on the args have "axiomatic facts": we trust the
            // ABI code to pass through the values unharmed, so the
            // facts given to us in the CLIF should still be true.
            Ok(())
        }
        Inst::ULoad8 { mem, flags, .. } | Inst::SLoad8 { mem, flags, .. } => {
            check_addr(*flags, mem, vcode, 1)
        }
        Inst::ULoad16 { mem, flags, .. } | Inst::SLoad16 { mem, flags, .. } => {
            check_addr(*flags, mem, vcode, 2)
        }
        Inst::ULoad32 { mem, flags, .. } | Inst::SLoad32 { mem, flags, .. } => {
            check_addr(*flags, mem, vcode, 4)
        }
        Inst::ULoad64 { mem, flags, .. } => check_addr(*flags, mem, vcode, 8),
        Inst::FpuLoad32 { mem, flags, .. } => check_addr(*flags, mem, vcode, 4),
        Inst::FpuLoad64 { mem, flags, .. } => check_addr(*flags, mem, vcode, 8),
        Inst::FpuLoad128 { mem, flags, .. } => check_addr(*flags, mem, vcode, 16),
        Inst::LoadP64 { mem, flags, .. } => check_addr_pair(*flags, mem, vcode, 16),
        Inst::FpuLoadP64 { mem, flags, .. } => check_addr_pair(*flags, mem, vcode, 16),
        Inst::FpuLoadP128 { mem, flags, .. } => check_addr_pair(*flags, mem, vcode, 32),
        Inst::VecLoadReplicate { rn, flags, .. } => check_scalar_addr(*flags, *rn, vcode, 16),
        Inst::LoadAcquire {
            access_ty,
            rn,
            flags,
            ..
        } => check_scalar_addr(*flags, *rn, vcode, access_ty.bytes() as u8),

        Inst::Store8 { mem, flags, .. } => check_addr(*flags, mem, vcode, 1),
        Inst::Store16 { mem, flags, .. } => check_addr(*flags, mem, vcode, 2),
        Inst::Store32 { mem, flags, .. } => check_addr(*flags, mem, vcode, 4),
        Inst::Store64 { mem, flags, .. } => check_addr(*flags, mem, vcode, 8),
        Inst::FpuStore32 { mem, flags, .. } => check_addr(*flags, mem, vcode, 4),
        Inst::FpuStore64 { mem, flags, .. } => check_addr(*flags, mem, vcode, 8),
        Inst::FpuStore128 { mem, flags, .. } => check_addr(*flags, mem, vcode, 16),
        Inst::StoreP64 { mem, flags, .. } => check_addr_pair(*flags, mem, vcode, 16),
        Inst::FpuStoreP64 { mem, flags, .. } => check_addr_pair(*flags, mem, vcode, 16),
        Inst::FpuStoreP128 { mem, flags, .. } => check_addr_pair(*flags, mem, vcode, 32),
        Inst::StoreRelease {
            access_ty,
            rn,
            flags,
            ..
        } => check_scalar_addr(*flags, *rn, vcode, access_ty.bytes() as u8),

        i => {
            panic!("Fact on unknown inst: {:?}", i);
        }
    }
}

fn amode_extend(value: &Fact, mode: ExtendOp) -> Option<Fact> {
    match mode {
        ExtendOp::UXTB => value.uextend(8, 64),
        ExtendOp::UXTH => value.uextend(16, 64),
        ExtendOp::UXTW => value.uextend(32, 64),
        ExtendOp::UXTX => Some(value.clone()),
        ExtendOp::SXTB => value.sextend(8, 64),
        ExtendOp::SXTH => value.uextend(16, 64),
        ExtendOp::SXTW => value.uextend(32, 64),
        ExtendOp::SXTX => None,
    }
}

fn check_addr(flags: MemFlags, addr: &AMode, vcode: &VCode<Inst>, size: u8) -> PccResult<()> {
    if !flags.checked() {
        return Ok(());
    }

    match addr {
        &AMode::RegReg { rn, rm } => {
            let rn = vcode.vreg_fact(rn.into()).ok_or(PccError::MissingFact)?;
            let rm = vcode.vreg_fact(rm.into()).ok_or(PccError::MissingFact)?;
            let sum = rn.add(&rm, 64, 64).ok_or(PccError::MissingFact)?;
            sum.check_address(size as u32)
        }
        &AMode::RegScaled { rn, rm, ty } => {
            let rn = vcode.vreg_fact(rn.into()).ok_or(PccError::MissingFact)?;
            let rm = vcode.vreg_fact(rm.into()).ok_or(PccError::MissingFact)?;
            let rm_scaled = rm.scale(64, ty.bytes()).ok_or(PccError::Overflow)?;
            let sum = rn.add(&rm_scaled, 64, 64).ok_or(PccError::MissingFact)?;
            sum.check_address(size as u32)
        }
        &AMode::RegScaledExtended {
            rn,
            rm,
            ty,
            extendop,
        } => {
            let rn = vcode.vreg_fact(rn.into()).ok_or(PccError::MissingFact)?;
            let rm = vcode.vreg_fact(rm.into()).ok_or(PccError::MissingFact)?;
            let rm_extended = amode_extend(rm, extendop).ok_or(PccError::MissingFact)?;
            let rm_scaled = rm_extended
                .scale(64, ty.bytes())
                .ok_or(PccError::Overflow)?;
            let sum = rn.add(&rm_scaled, 64, 64).ok_or(PccError::MissingFact)?;
            sum.check_address(size as u32)
        }
        &AMode::RegExtended { rn, rm, extendop } => {
            let rn = vcode.vreg_fact(rn.into()).ok_or(PccError::MissingFact)?;
            let rm = vcode.vreg_fact(rm.into()).ok_or(PccError::MissingFact)?;
            let rm_extended = amode_extend(rm, extendop).ok_or(PccError::MissingFact)?;
            let sum = rn.add(&rm_extended, 64, 64).ok_or(PccError::MissingFact)?;
            sum.check_address(size as u32)
        }
        &AMode::Unscaled { rn, simm9 } => {
            let rn = vcode.vreg_fact(rn.into()).ok_or(PccError::MissingFact)?;
            let sum = rn
                .offset(64, simm9.value as i64)
                .ok_or(PccError::MissingFact)?;
            sum.check_address(size as u32)
        }
        &AMode::UnsignedOffset { rn, uimm12 } => {
            let rn = vcode.vreg_fact(rn.into()).ok_or(PccError::MissingFact)?;
            let offset = (uimm12.value as u64) * (size as u64);
            let sum = rn.offset(64, offset as i64).ok_or(PccError::MissingFact)?;
            sum.check_address(size as u32)
        }
        &AMode::Label { .. } | &AMode::Const { .. } => {
            // Always accept: labels and constants must be within the
            // generated code (else they won't be resolved).
            Ok(())
        }
        &AMode::RegOffset { rn, off, .. } => {
            let rn = vcode.vreg_fact(rn.into()).ok_or(PccError::MissingFact)?;
            let sum = rn.offset(64, off).ok_or(PccError::MissingFact)?;
            sum.check_address(size as u32)
        }
        &AMode::SPOffset { .. }
        | &AMode::FPOffset { .. }
        | &AMode::NominalSPOffset { .. }
        | &AMode::SPPostIndexed { .. }
        | &AMode::SPPreIndexed { .. } => {
            // We trust ABI code (for now!) and no lowering rules
            // lower input value accesses directly to these.
            Ok(())
        }
    }
}

fn check_addr_pair(
    _flags: MemFlags,
    _addr: &PairAMode,
    _vcode: &VCode<Inst>,
    _size: u8,
) -> PccResult<()> {
    Err(PccError::Unimplemented)
}

fn check_scalar_addr(flags: MemFlags, reg: Reg, vcode: &VCode<Inst>, size: u8) -> PccResult<()> {
    if !flags.checked() {
        return Ok(());
    }
    let fact = vcode.vreg_fact(reg.into()).ok_or(PccError::MissingFact)?;
    fact.check_address(size as u32)
}
