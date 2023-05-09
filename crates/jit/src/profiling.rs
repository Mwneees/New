#![allow(missing_docs)]

use crate::{demangling::demangle_function_name_or_index, CodeMemory, CompiledModule};
use anyhow::{bail, Result};
use wasmtime_environ::{DefinedFuncIndex, EntityRef};

cfg_if::cfg_if! {
    if #[cfg(all(feature = "jitdump", target_os = "linux"))] {
        mod jitdump;
        pub use jitdump::new as new_jitdump;
    } else {
        pub fn new_jitdump() -> Result<Box<dyn ProfilingAgent>> {
            if cfg!(feature = "jitdump") {
                bail!("jitdump is not supported on this platform");
            } else {
                bail!("jitdump support disabled at compile time");
            }
        }
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_os = "linux")] {
        mod perfmap;
        pub use perfmap::new as new_perfmap;
    } else {
        pub fn new_perfmap() -> Result<Box<dyn ProfilingAgent>> {
            bail!("perfmap support not supported on this platform");
        }
    }
}

cfg_if::cfg_if! {
    // Note: VTune support is disabled on windows mingw because the ittapi crate doesn't compile
    // there; see also https://github.com/bytecodealliance/wasmtime/pull/4003 for rationale.
    if #[cfg(all(feature = "vtune", target_arch = "x86_64", not(all(target_os = "windows", target_env = "gnu"))))] {
        mod vtune;
        pub use vtune::new as new_vtune;
    } else {
        pub fn new_vtune() -> Result<Box<dyn ProfilingAgent>> {
            if cfg!(feature = "vtune") {
                bail!("VTune is not supported on this platform.");
            } else {
                bail!("VTune support disabled at compile time.");
            }
        }
    }
}

/// Common interface for profiling tools.
pub trait ProfilingAgent: Send + Sync + 'static {
    /// Notify the profiler of a new module loaded into memory
    fn module_load(&self, module: &CompiledModule);

    /// Notify the profiler about a single dynamically-generated trampoline (for host function)
    /// that is being loaded now.`
    fn load_single_trampoline(&self, name: &str, addr: *const u8, size: usize);

    fn register_trampolines(&self, code: &CodeMemory) {
        use object::{File, Object as _, ObjectSection, ObjectSymbol, SectionKind, SymbolKind};

        let image = match File::parse(&code.mmap()[..]) {
            Ok(image) => image,
            Err(_) => return,
        };

        let text_base = match image.sections().find(|s| s.kind() == SectionKind::Text) {
            Some(section) => match section.data() {
                Ok(data) => data.as_ptr() as usize,
                Err(_) => return,
            },
            None => return,
        };

        for sym in image.symbols() {
            if !sym.is_definition() {
                continue;
            }
            if sym.kind() != SymbolKind::Text {
                continue;
            }
            let address = sym.address();
            let size = sym.size();
            if address == 0 || size == 0 {
                continue;
            }
            if let Ok(name) = sym.name() {
                let addr = text_base + address as usize;
                self.load_single_trampoline(name, addr as *const u8, size as usize);
            }
        }
    }
}

pub fn new_null() -> Box<dyn ProfilingAgent> {
    Box::new(NullProfilerAgent)
}

#[derive(Debug, Default, Clone, Copy)]
struct NullProfilerAgent;

impl ProfilingAgent for NullProfilerAgent {
    fn module_load(&self, _module: &CompiledModule) {}
    fn load_single_trampoline(&self, _name: &str, _addr: *const u8, _size: usize) {}
    fn register_trampolines(&self, _code: &CodeMemory) {}
}

#[allow(dead_code)]
fn debug_name(module: &CompiledModule, index: DefinedFuncIndex) -> String {
    let index = module.module().func_index(index);
    let mut debug_name = String::new();
    demangle_function_name_or_index(&mut debug_name, module.func_name(index), index.index())
        .unwrap();
    debug_name
}
