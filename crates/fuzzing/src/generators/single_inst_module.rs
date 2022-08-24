//! Generate Wasm modules that contain a single instruction.

use super::ModuleConfig;
use arbitrary::Unstructured;
use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, Module,
    TypeSection, ValType,
};

/// The name of the function generated by this module.
const FUNCTION_NAME: &'static str = "test";

/// Configure a single instruction module.
///
/// By explicitly defining the parameter and result types (versus generating the
/// module directly), we can more easily generate values of the right type.
#[derive(Clone)]
pub struct SingleInstModule<'a> {
    instruction: Instruction<'a>,
    parameters: &'a [ValType],
    results: &'a [ValType],
    feature: fn(&ModuleConfig) -> bool,
}

impl<'a> SingleInstModule<'a> {
    /// Choose a single-instruction module that matches `config`.
    pub fn new(u: &mut Unstructured<'a>, config: &mut ModuleConfig) -> arbitrary::Result<&'a Self> {
        // To avoid skipping modules unnecessarily during fuzzing, fix up the
        // `ModuleConfig` to match the inherent limits of a single-instruction
        // module.
        config.config.min_funcs = 1;
        config.config.max_funcs = 1;
        config.config.min_tables = 0;
        config.config.max_tables = 0;
        config.config.min_memories = 0;
        config.config.max_memories = 0;

        // Only select instructions that match the `ModuleConfig`.
        let instructions = &INSTRUCTIONS
            .iter()
            .filter(|i| (i.feature)(config))
            .collect::<Vec<_>>();
        u.choose(&instructions[..]).copied()
    }

    /// Encode a binary Wasm module with a single exported function, `test`,
    /// that executes the single instruction.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut module = Module::new();

        // Encode the type section.
        let mut types = TypeSection::new();
        types.function(
            self.parameters.iter().cloned(),
            self.results.iter().cloned(),
        );
        module.section(&types);

        // Encode the function section.
        let mut functions = FunctionSection::new();
        let type_index = 0;
        functions.function(type_index);
        module.section(&functions);

        // Encode the export section.
        let mut exports = ExportSection::new();
        exports.export(FUNCTION_NAME, ExportKind::Func, 0);
        module.section(&exports);

        // Encode the code section.
        let mut codes = CodeSection::new();
        let mut f = Function::new([]);
        for (index, _) in self.parameters.iter().enumerate() {
            f.instruction(&Instruction::LocalGet(index as u32));
        }
        f.instruction(&self.instruction);
        f.instruction(&Instruction::End);
        codes.function(&f);
        module.section(&codes);

        // Extract the encoded Wasm bytes for this module.
        module.finish()
    }
}

// MACROS
//
// These macros make it a bit easier to define the instructions available for
// generation. The idea is that, with these macros, we can define the list of
// instructions compactly and allow for easier changes to the Rust code (e.g.,
// `SingleInstModule`).

macro_rules! valtype {
    (i32) => {
        ValType::I32
    };
    (i64) => {
        ValType::I64
    };
    (f32) => {
        ValType::F32
    };
    (f64) => {
        ValType::F64
    };
}

macro_rules! inst {
    ($inst:ident, ($($arguments_ty:tt),+) -> $result_ty:tt) => {
        inst! { $inst, ($($arguments_ty),+) -> $result_ty, |_| true }
    };
    ($inst:ident, ($($arguments_ty:tt),+) -> $result_ty:tt, $feature:expr) => {
        SingleInstModule {
            instruction: Instruction::$inst,
            parameters: &[$(valtype!($arguments_ty)),+],
            results: &[valtype!($result_ty)],
            feature: $feature,
        }
    };
}

static INSTRUCTIONS: &[SingleInstModule] = &[
    // Integer arithmetic.
    // I32Const
    // I64Const
    // F32Const
    // F64Const
    inst!(I32Clz, (i32) -> i32),
    inst!(I64Clz, (i64) -> i64),
    inst!(I32Ctz, (i32) -> i32),
    inst!(I64Ctz, (i64) -> i64),
    inst!(I32Popcnt, (i32) -> i32),
    inst!(I64Popcnt, (i64) -> i64),
    inst!(I32Add, (i32, i32) -> i32),
    inst!(I64Add, (i64, i64) -> i64),
    inst!(I32Sub, (i32, i32) -> i32),
    inst!(I64Sub, (i64, i64) -> i64),
    inst!(I32Mul, (i32, i32) -> i32),
    inst!(I64Mul, (i64, i64) -> i64),
    inst!(I32DivS, (i32, i32) -> i32),
    inst!(I64DivS, (i64, i64) -> i64),
    inst!(I32DivU, (i32, i32) -> i32),
    inst!(I64DivU, (i64, i64) -> i64),
    inst!(I32RemS, (i32, i32) -> i32),
    inst!(I64RemS, (i64, i64) -> i64),
    inst!(I32RemU, (i32, i32) -> i32),
    inst!(I64RemU, (i64, i64) -> i64),
    // Integer bitwise.
    inst!(I32And, (i32, i32) -> i32),
    inst!(I64And, (i64, i64) -> i64),
    inst!(I32Or, (i32, i32) -> i32),
    inst!(I64Or, (i64, i64) -> i64),
    inst!(I32Xor, (i32, i32) -> i32),
    inst!(I64Xor, (i64, i64) -> i64),
    inst!(I32Shl, (i32, i32) -> i32),
    inst!(I64Shl, (i64, i64) -> i64),
    inst!(I32ShrS, (i32, i32) -> i32),
    inst!(I64ShrS, (i64, i64) -> i64),
    inst!(I32ShrU, (i32, i32) -> i32),
    inst!(I64ShrU, (i64, i64) -> i64),
    inst!(I32Rotl, (i32, i32) -> i32),
    inst!(I64Rotl, (i64, i64) -> i64),
    inst!(I32Rotr, (i32, i32) -> i32),
    inst!(I64Rotr, (i64, i64) -> i64),
    // Integer comparison.
    inst!(I32Eqz, (i32) -> i32),
    inst!(I64Eqz, (i64) -> i32),
    inst!(I32Eq, (i32, i32) -> i32),
    inst!(I64Eq, (i64, i64) -> i32),
    inst!(I32Ne, (i32, i32) -> i32),
    inst!(I64Ne, (i64, i64) -> i32),
    inst!(I32LtS, (i32, i32) -> i32),
    inst!(I64LtS, (i64, i64) -> i32),
    inst!(I32LtU, (i32, i32) -> i32),
    inst!(I64LtU, (i64, i64) -> i32),
    inst!(I32GtS, (i32, i32) -> i32),
    inst!(I64GtS, (i64, i64) -> i32),
    inst!(I32GtU, (i32, i32) -> i32),
    inst!(I64GtU, (i64, i64) -> i32),
    inst!(I32LeS, (i32, i32) -> i32),
    inst!(I64LeS, (i64, i64) -> i32),
    inst!(I32LeU, (i32, i32) -> i32),
    inst!(I64LeU, (i64, i64) -> i32),
    inst!(I32GeS, (i32, i32) -> i32),
    inst!(I64GeS, (i64, i64) -> i32),
    inst!(I32GeU, (i32, i32) -> i32),
    inst!(I64GeU, (i64, i64) -> i32),
    // Floating-point arithmetic.
    inst!(F32Abs, (f32) -> f32),
    inst!(F64Abs, (f64) -> f64),
    inst!(F32Sqrt, (f32) -> f32),
    inst!(F64Sqrt, (f64) -> f64),
    inst!(F32Ceil, (f32) -> f32),
    inst!(F64Ceil, (f64) -> f64),
    inst!(F32Floor, (f32) -> f32),
    inst!(F64Floor, (f64) -> f64),
    inst!(F32Trunc, (f32) -> f32),
    inst!(F64Trunc, (f64) -> f64),
    inst!(F32Nearest, (f32) -> f32),
    inst!(F64Nearest, (f64) -> f64),
    inst!(F32Neg, (f32) -> f32),
    inst!(F64Neg, (f64) -> f64),
    inst!(F32Add, (f32, f32) -> f32),
    inst!(F64Add, (f64, f64) -> f64),
    inst!(F32Sub, (f32, f32) -> f32),
    inst!(F64Sub, (f64, f64) -> f64),
    inst!(F32Mul, (f32, f32) -> f32),
    inst!(F64Mul, (f64, f64) -> f64),
    inst!(F32Div, (f32, f32) -> f32),
    inst!(F64Div, (f64, f64) -> f64),
    inst!(F32Min, (f32, f32) -> f32),
    inst!(F64Min, (f64, f64) -> f64),
    inst!(F32Max, (f32, f32) -> f32),
    inst!(F64Max, (f64, f64) -> f64),
    inst!(F32Copysign, (f32, f32) -> f32),
    inst!(F64Copysign, (f64, f64) -> f64),
    // Floating-point comparison.
    inst!(F32Eq, (f32, f32) -> i32),
    inst!(F64Eq, (f64, f64) -> i32),
    inst!(F32Ne, (f32, f32) -> i32),
    inst!(F64Ne, (f64, f64) -> i32),
    inst!(F32Lt, (f32, f32) -> i32),
    inst!(F64Lt, (f64, f64) -> i32),
    inst!(F32Gt, (f32, f32) -> i32),
    inst!(F64Gt, (f64, f64) -> i32),
    inst!(F32Le, (f32, f32) -> i32),
    inst!(F64Le, (f64, f64) -> i32),
    inst!(F32Ge, (f32, f32) -> i32),
    inst!(F64Ge, (f64, f64) -> i32),
    // Integer conversions ("to integer").
    inst!(I32Extend8S, (i32) -> i32, |c| c.config.sign_extension_enabled),
    inst!(I32Extend16S, (i32) -> i32, |c| c.config.sign_extension_enabled),
    inst!(I64Extend8S, (i64) -> i64, |c| c.config.sign_extension_enabled),
    inst!(I64Extend16S, (i64) -> i64, |c| c.config.sign_extension_enabled),
    inst!(I64Extend32S, (i64) -> i64, |c| c.config.sign_extension_enabled),
    inst!(I32WrapI64, (i64) -> i32),
    inst!(I64ExtendI32S, (i32) -> i64),
    inst!(I64ExtendI32U, (i32) -> i64),
    inst!(I32TruncF32S, (f32) -> i32),
    inst!(I32TruncF32U, (f32) -> i32),
    inst!(I32TruncF64S, (f64) -> i32),
    inst!(I32TruncF64U, (f64) -> i32),
    inst!(I64TruncF32S, (f32) -> i64),
    inst!(I64TruncF32U, (f32) -> i64),
    inst!(I64TruncF64S, (f64) -> i64),
    inst!(I64TruncF64U, (f64) -> i64),
    inst!(I32TruncSatF32S, (f32) -> i32, |c| c.config.saturating_float_to_int_enabled),
    inst!(I32TruncSatF32U, (f32) -> i32, |c| c.config.saturating_float_to_int_enabled),
    inst!(I32TruncSatF64S, (f64) -> i32, |c| c.config.saturating_float_to_int_enabled),
    inst!(I32TruncSatF64U, (f64) -> i32, |c| c.config.saturating_float_to_int_enabled),
    inst!(I64TruncSatF32S, (f32) -> i64, |c| c.config.saturating_float_to_int_enabled),
    inst!(I64TruncSatF32U, (f32) -> i64, |c| c.config.saturating_float_to_int_enabled),
    inst!(I64TruncSatF64S, (f64) -> i64, |c| c.config.saturating_float_to_int_enabled),
    inst!(I64TruncSatF64U, (f64) -> i64, |c| c.config.saturating_float_to_int_enabled),
    inst!(I32ReinterpretF32, (f32) -> i32),
    inst!(I64ReinterpretF64, (f64) -> i64),
    // Floating-point conversions ("to float").
    inst!(F32DemoteF64, (f64) -> f32),
    inst!(F64PromoteF32, (f32) -> f64),
    inst!(F32ConvertI32S, (i32) -> f32),
    inst!(F32ConvertI32U, (i32) -> f32),
    inst!(F32ConvertI64S, (i64) -> f32),
    inst!(F32ConvertI64U, (i64) -> f32),
    inst!(F64ConvertI32S, (i32) -> f64),
    inst!(F64ConvertI32U, (i32) -> f64),
    inst!(F64ConvertI64S, (i64) -> f64),
    inst!(F64ConvertI64U, (i64) -> f64),
    inst!(F32ReinterpretI32, (i32) -> f32),
    inst!(F64ReinterpretI64, (i64) -> f64),
];

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity() {
        let sut = SingleInstModule {
            instruction: Instruction::I32Add,
            parameters: &[ValType::I32, ValType::I32],
            results: &[ValType::I32],
            feature: |_| true,
        };
        let wasm = sut.to_bytes();
        let wat = wasmprinter::print_bytes(wasm).unwrap();
        assert_eq!(
            wat,
            r#"(module
  (type (;0;) (func (param i32 i32) (result i32)))
  (func (;0;) (type 0) (param i32 i32) (result i32)
    local.get 0
    local.get 1
    i32.add
  )
  (export "test" (func 0))
)"#
        )
    }

    #[test]
    fn instructions_encode_to_valid_modules() {
        for inst in INSTRUCTIONS {
            assert!(wat::parse_bytes(&inst.to_bytes()).is_ok());
        }
    }
}
