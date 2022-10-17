use crate::isa::reg::Reg;
use smallvec::SmallVec;
use std::ops::{Add, BitAnd, Not, Sub};
use wasmtime_environ::{WasmFuncType, WasmType};

pub(crate) mod addressing_mode;
pub(crate) mod local;

/// Trait implemented by a specific ISA and used to provide
/// information about alignment, parameter passing, usage of
/// specific registers, etc
pub(crate) trait ABI {
    /// The required stack alignment
    fn stack_align(&self) -> u8;

    /// The offset to the argument base, relative to the frame pointer
    fn arg_base_offset(&self) -> u8;

    /// Construct the ABI-specific signature from a WebAssembly
    /// function type
    fn sig(&self, wasm_sig: &WasmFuncType) -> ABISig;

    /// Returns the number of bits in a word
    fn word_bits() -> u32;

    /// Returns the number of bytes in a word
    fn word_bytes() -> u32 {
        Self::word_bits() / 8
    }

    /// Returns the designated scratch register
    fn scratch_reg() -> Reg;
}

/// ABI-specific representation of a function argument
#[derive(Debug)]
pub(crate) enum ABIArg {
    /// A register argument
    Reg {
        /// Type of the argument
        ty: WasmType,
        /// Register holding the argument
        reg: Reg,
    },
    /// A stack argument
    Stack {
        /// The type of the argument
        ty: WasmType,
        /// Offset of the argument relative to the frame pointer
        // SpiderMonkey's baseline compiler references arguments relative
        // to the frame pointer and locals relative to the stack pointer
        offset: u32,
    },
}

impl ABIArg {
    /// Allocate a new register abi arg
    pub fn reg(reg: Reg, ty: WasmType) -> Self {
        Self::Reg { reg, ty }
    }

    /// Allocate a new stack abi arg
    pub fn stack_offset(offset: u32, ty: WasmType) -> Self {
        Self::Stack { ty, offset }
    }

    /// Is this abi arg in a register
    pub fn is_reg(&self) -> bool {
        match *self {
            ABIArg::Reg { .. } => true,
            _ => false,
        }
    }

    /// Get the register associated to this arg
    pub fn get_reg(&self) -> Option<Reg> {
        match *self {
            ABIArg::Reg { reg, .. } => Some(reg),
            _ => None,
        }
    }

    /// Get the type associated to this arg
    pub fn ty(&self) -> WasmType {
        match *self {
            ABIArg::Reg { ty, .. } | ABIArg::Stack { ty, .. } => ty,
        }
    }
}

/// ABI-specific representation of the function result
pub(crate) enum ABIResult {
    Reg {
        /// Type of the result
        ty: Option<WasmType>,
        /// Register to hold the result
        reg: Reg,
    },
}

impl ABIResult {
    /// Create a register ABI result
    pub fn reg(ty: Option<WasmType>, reg: Reg) -> Self {
        Self::Reg { ty, reg }
    }

    /// Get the result reg
    pub fn result_reg(&self) -> Reg {
        match self {
            Self::Reg { reg, .. } => *reg,
        }
    }

    /// Checks if the result is void
    pub fn is_void(&self) -> bool {
        match self {
            Self::Reg { ty, .. } => ty.is_none(),
        }
    }
}

/// An ABI-specific representation of a function signature
pub(crate) struct ABISig {
    /// Function parameters
    pub params: SmallVec<[ABIArg; 6]>,
    pub result: ABIResult,
}

/// Returns the size in bytes of a given WebAssembly type
pub(crate) fn ty_size(ty: &WasmType) -> u32 {
    match *ty {
        WasmType::I32 | WasmType::F32 => 4,
        WasmType::I64 | WasmType::F64 => 8,
        _ => panic!(),
    }
}

/// Align a value up to the given power-of-two-alignment
// See https://sites.google.com/site/theoryofoperatingsystems/labs/malloc/align8
pub(crate) fn align_to<N>(value: N, alignment: N) -> N
where
    N: Not<Output = N>
        + BitAnd<N, Output = N>
        + Add<N, Output = N>
        + Sub<N, Output = N>
        + From<u8>
        + Copy,
{
    let alignment_mask = alignment - 1.into();
    (value + alignment_mask) & !alignment_mask
}
