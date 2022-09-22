use crate::{compilation_env::CompilationEnv, isa::TargetIsa};
use anyhow::Result;
use target_lexicon::Triple;
use wasmtime_environ::{FunctionBodyData, WasmFuncType};

mod abi;
mod masm;
mod regs;

/// Create an ISA from the given triple
pub(crate) fn isa_from(triple: Triple) -> Aarch64 {
    Aarch64::new(triple)
}

pub(crate) struct Aarch64 {
    triple: Triple,
}

impl Aarch64 {
    pub fn new(triple: Triple) -> Self {
        Self { triple }
    }
}

impl TargetIsa for Aarch64 {
    fn name(&self) -> &'static str {
        "x64"
    }

    fn triple(&self) -> &Triple {
        &self.triple
    }

    fn compile_function(
        &self,
        sig: &WasmFuncType,
        body: &mut FunctionBodyData,
    ) -> Result<&'static str> {
        let abi = abi::Aarch64ABI::default();
        let asm = masm::MacroAssembler::default();
        let env = CompilationEnv::new(sig, body, abi, asm)?;

        todo!()
    }
}
