use super::regs;
use crate::abi::{ABIArg, ABISig, ABI};
use regalloc2::PReg;
use smallvec::SmallVec;
use wasmtime_environ::WasmType;

/// Helper environment to track argument-register
/// assignment in x64
///
/// The first element tracks the general purpose register index
/// The second element tracks the floating point register index
#[derive(Default)]
struct RegIndexEnv(u8, u8);

impl RegIndexEnv {
    fn next_gpr(&mut self) -> u8 {
        Self::increment(&mut self.0)
    }

    fn next_fpr(&mut self) -> u8 {
        Self::increment(&mut self.1)
    }

    fn increment(index: &mut u8) -> u8 {
        let current = *index;
        *index += 1;
        current
    }
}

#[derive(Default)]
pub(crate) struct X64ABI;

impl ABI for X64ABI {
    // TODO: change to 16 once SIMD is supported
    fn stack_align(&self) -> u8 {
        8
    }

    fn arg_base_offset(&self) -> u8 {
        16
    }

    fn word_bits() -> u32 {
        64
    }

    fn sig(&self, wasm_sig: &wasmtime_environ::WasmFuncType) -> ABISig {
        let mut stack_offset = 0;
        let mut index_env = RegIndexEnv::default();

        let params: SmallVec<[ABIArg; 6]> = wasm_sig
            .params()
            .iter()
            .map(|arg| Self::to_abi_arg(arg, &mut stack_offset, &mut index_env))
            .collect();

        ABISig { params }
    }
}

impl X64ABI {
    fn to_abi_arg(
        wasm_arg: &WasmType,
        stack_offset: &mut u64,
        index_env: &mut RegIndexEnv,
    ) -> ABIArg {
        let (reg, ty) = match wasm_arg {
            ty @ (WasmType::I32 | WasmType::I64) => (Self::int_reg_for(index_env.next_gpr()), ty),

            ty @ (WasmType::F32 | WasmType::F64) => (Self::float_reg_for(index_env.next_fpr()), ty),

            ty => unreachable!("Unsupported argument type {:?}", ty),
        };

        let default = || {
            let arg = ABIArg::stack_offset(*stack_offset, *ty);
            let size = Self::word_bytes();
            *stack_offset += size as u64;
            arg
        };

        reg.map_or_else(default, |reg| ABIArg::reg(reg, *ty))
    }

    fn int_reg_for(index: u8) -> Option<PReg> {
        match index {
            0 => Some(regs::rdi()),
            1 => Some(regs::rsi()),
            2 => Some(regs::rdx()),
            3 => Some(regs::rcx()),
            4 => Some(regs::r8()),
            5 => Some(regs::r9()),
            _ => None,
        }
    }

    fn float_reg_for(index: u8) -> Option<PReg> {
        match index {
            0 => Some(regs::xmm0()),
            1 => Some(regs::xmm1()),
            2 => Some(regs::xmm2()),
            3 => Some(regs::xmm3()),
            4 => Some(regs::xmm4()),
            5 => Some(regs::xmm5()),
            6 => Some(regs::xmm6()),
            7 => Some(regs::xmm7()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RegIndexEnv, X64ABI};
    use crate::{
        abi::{ABIArg, ABI},
        isa::x64::regs,
    };
    use regalloc2::PReg;
    use wasmtime_environ::{
        WasmFuncType,
        WasmType::{self, *},
    };

    #[test]
    fn test_get_next_reg_index() {
        let mut index_env = RegIndexEnv::default();
        assert_eq!(index_env.next_fpr(), 0);
        assert_eq!(index_env.next_gpr(), 0);
        assert_eq!(index_env.next_fpr(), 1);
        assert_eq!(index_env.next_gpr(), 1);
        assert_eq!(index_env.next_fpr(), 2);
        assert_eq!(index_env.next_gpr(), 2);
    }

    #[test]
    fn int_abi_sig() {
        let wasm_sig = WasmFuncType::new(
            Box::new([I32, I64, I32, I64, I32, I32, I64, I32]),
            Box::new([]),
        );

        let abi = X64ABI::default();
        let sig = abi.sig(&wasm_sig);
        let params = sig.params;

        match_reg_arg(params.get(0).unwrap(), I32, regs::rdi());
        match_reg_arg(params.get(1).unwrap(), I64, regs::rsi());
        match_reg_arg(params.get(2).unwrap(), I32, regs::rdx());
        match_reg_arg(params.get(3).unwrap(), I64, regs::rcx());
        match_reg_arg(params.get(4).unwrap(), I32, regs::r8());
        match_reg_arg(params.get(5).unwrap(), I32, regs::r9());
        match_stack_arg(params.get(6).unwrap(), I64, 0u64);
        match_stack_arg(params.get(7).unwrap(), I32, 8u64);
    }

    #[test]
    fn float_abi_sig() {
        let wasm_sig = WasmFuncType::new(
            Box::new([F32, F64, F32, F64, F32, F32, F64, F32, F64]),
            Box::new([]),
        );

        let abi = X64ABI::default();
        let sig = abi.sig(&wasm_sig);
        let params = sig.params;

        match_reg_arg(params.get(0).unwrap(), F32, regs::xmm0());
        match_reg_arg(params.get(1).unwrap(), F64, regs::xmm1());
        match_reg_arg(params.get(2).unwrap(), F32, regs::xmm2());
        match_reg_arg(params.get(3).unwrap(), F64, regs::xmm3());
        match_reg_arg(params.get(4).unwrap(), F32, regs::xmm4());
        match_reg_arg(params.get(5).unwrap(), F32, regs::xmm5());
        match_reg_arg(params.get(6).unwrap(), F64, regs::xmm6());
        match_reg_arg(params.get(7).unwrap(), F32, regs::xmm7());
        match_stack_arg(params.get(8).unwrap(), F64, 0u64);
    }

    #[test]
    fn mixed_abi_sig() {
        let wasm_sig = WasmFuncType::new(
            Box::new([F32, I32, I64, F64, I32, F32, F64, F32, F64]),
            Box::new([]),
        );

        let abi = X64ABI::default();
        let sig = abi.sig(&wasm_sig);
        let params = sig.params;

        match_reg_arg(params.get(0).unwrap(), F32, regs::xmm0());
        match_reg_arg(params.get(1).unwrap(), I32, regs::rdi());
        match_reg_arg(params.get(2).unwrap(), I64, regs::rsi());
        match_reg_arg(params.get(3).unwrap(), F64, regs::xmm1());
        match_reg_arg(params.get(4).unwrap(), I32, regs::rdx());
        match_reg_arg(params.get(5).unwrap(), F32, regs::xmm2());
        match_reg_arg(params.get(6).unwrap(), F64, regs::xmm3());
        match_reg_arg(params.get(7).unwrap(), F32, regs::xmm4());
        match_reg_arg(params.get(8).unwrap(), F64, regs::xmm5());
    }

    fn match_reg_arg(abi_arg: &ABIArg, expected_ty: WasmType, expected_reg: PReg) {
        match abi_arg {
            &ABIArg::Reg { reg, ty } => {
                assert_eq!(reg, expected_reg);
                assert_eq!(ty, expected_ty);
            }
            stack => panic!("Expected reg argument, got {:?}", stack),
        }
    }

    fn match_stack_arg(abi_arg: &ABIArg, expected_ty: WasmType, expected_offset: u64) {
        match abi_arg {
            &ABIArg::Stack { offset, ty } => {
                assert_eq!(offset, expected_offset);
                assert_eq!(ty, expected_ty);
            }
            stack => panic!("Expected stack argument, got {:?}", stack),
        }
    }
}
