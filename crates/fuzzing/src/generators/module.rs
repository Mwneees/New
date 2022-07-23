//! Generate a Wasm module and the configuration for generating it.

use super::SingleInstModule;
use arbitrary::{Arbitrary, Unstructured};
use wasm_smith::SwarmConfig;

/// A container for the kinds of modules that can be generated during fuzzing.
#[derive(Arbitrary)]
pub enum Module<'a> {
    /// A module generated by `wasm-smith`; this might have complex control
    /// flow, traps, etc.
    BigModule(ModuleConfig),
    /// A module exercising a single Wasm instruction; contains a single
    /// exported function, `test`.
    SingleInstModule(&'a SingleInstModule<'a>),
}

impl Module<'_> {
    /// Generate the actual bytes of a Wasm module given some random `input` and
    /// `default_fuel` (fuel does not apply to single-instruction modules).
    pub fn generate(
        &self,
        input: &mut Unstructured<'_>,
        default_fuel: Option<u32>,
    ) -> arbitrary::Result<Vec<u8>> {
        match self {
            Module::BigModule(config) => {
                let module = config.generate(input, default_fuel)?;
                Ok(module.to_bytes())
            }
            Module::SingleInstModule(config) => Ok(config.to_bytes()),
        }
    }
}

/// Default module-level configuration for fuzzing Wasmtime.
///
/// Internally this uses `wasm-smith`'s own `SwarmConfig` but we further refine
/// the defaults here as well.
#[derive(Debug, Clone)]
pub struct ModuleConfig {
    #[allow(missing_docs)]
    pub config: SwarmConfig,
}

impl<'a> Arbitrary<'a> for ModuleConfig {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<ModuleConfig> {
        let mut config = SwarmConfig::arbitrary(u)?;

        // Allow multi-memory by default.
        config.max_memories = config.max_memories.max(2);

        // Allow multi-table by default.
        config.max_tables = config.max_tables.max(4);

        // Allow enabling some various wasm proposals by default. Note that
        // these are all unconditionally turned off even with
        // `SwarmConfig::arbitrary`.
        config.memory64_enabled = u.arbitrary()?;

        // Allow the threads proposal if memory64 is not already enabled. FIXME:
        // to allow threads and memory64 to coexist, see
        // https://github.com/bytecodealliance/wasmtime/issues/4267.
        config.threads_enabled = !config.memory64_enabled && u.arbitrary()?;

        Ok(ModuleConfig { config })
    }
}

impl ModuleConfig {
    /// Uses this configuration and the supplied source of data to generate a
    /// Wasm module.
    ///
    /// If a `default_fuel` is provided, the resulting module will be configured
    /// to ensure termination; as doing so will add an additional global to the
    /// module, the pooling allocator, if configured, must also have its globals
    /// limit updated.
    pub fn generate(
        &self,
        input: &mut Unstructured<'_>,
        default_fuel: Option<u32>,
    ) -> arbitrary::Result<wasm_smith::Module> {
        let mut module = wasm_smith::Module::new(self.config.clone(), input)?;

        if let Some(default_fuel) = default_fuel {
            module.ensure_termination(default_fuel);
        }

        Ok(module)
    }
}

/// Document the Wasm features necessary for a module to be evaluated.
#[derive(Debug, Default)]
#[allow(missing_docs)]
pub struct ModuleFeatures {
    pub simd: bool,
    pub multi_value: bool,
    pub reference_types: bool,
}
