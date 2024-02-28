#[cfg(feature = "runtime")]
use crate::runtime::type_registry::TypeRegistry;
use crate::Config;
use anyhow::{Context, Result};
#[cfg(any(feature = "cranelift", feature = "winch"))]
use object::write::{Object, StandardSegment};
use object::SectionKind;
use once_cell::sync::OnceCell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use wasmtime_environ::obj;
use wasmtime_environ::{FlagValue, ObjectKind, Tunables};
#[cfg(feature = "runtime")]
use wasmtime_runtime::GcRuntime;

mod serialization;

/// An `Engine` which is a global context for compilation and management of wasm
/// modules.
///
/// An engine can be safely shared across threads and is a cheap cloneable
/// handle to the actual engine. The engine itself will be deallocated once all
/// references to it have gone away.
///
/// Engines store global configuration preferences such as compilation settings,
/// enabled features, etc. You'll likely only need at most one of these for a
/// program.
///
/// ## Engines and `Clone`
///
/// Using `clone` on an `Engine` is a cheap operation. It will not create an
/// entirely new engine, but rather just a new reference to the existing engine.
/// In other words it's a shallow copy, not a deep copy.
///
/// ## Engines and `Default`
///
/// You can create an engine with default configuration settings using
/// `Engine::default()`. Be sure to consult the documentation of [`Config`] for
/// default settings.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    config: Config,
    tunables: Tunables,
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    compiler: Box<dyn wasmtime_environ::Compiler>,
    #[cfg(feature = "runtime")]
    allocator: Box<dyn wasmtime_runtime::InstanceAllocator + Send + Sync>,
    #[cfg(feature = "runtime")]
    gc_runtime: Arc<dyn GcRuntime>,
    #[cfg(feature = "runtime")]
    profiler: Box<dyn crate::profiling_agent::ProfilingAgent>,
    #[cfg(feature = "runtime")]
    signatures: TypeRegistry,
    #[cfg(feature = "runtime")]
    epoch: AtomicU64,
    #[cfg(feature = "runtime")]
    unique_id_allocator: wasmtime_runtime::CompiledModuleIdAllocator,

    /// One-time check of whether the compiler's settings, if present, are
    /// compatible with the native host.
    compatible_with_native_host: OnceCell<Result<(), String>>,
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new(&Config::default()).unwrap()
    }
}

impl Engine {
    /// Creates a new [`Engine`] with the specified compilation and
    /// configuration settings.
    ///
    /// # Errors
    ///
    /// This method can fail if the `config` is invalid or some
    /// configurations are incompatible.
    ///
    /// For example, feature `reference_types` will need to set
    /// the compiler setting `enable_safepoints` and `unwind_info`
    /// to `true`, but explicitly disable these two compiler settings
    /// will cause errors.
    pub fn new(config: &Config) -> Result<Engine> {
        #[cfg(feature = "runtime")]
        {
            // Ensure that wasmtime_runtime's signal handlers are configured. This
            // is the per-program initialization required for handling traps, such
            // as configuring signals, vectored exception handlers, etc.
            wasmtime_runtime::init_traps(crate::module::get_wasm_trap, config.macos_use_mach_ports);
            #[cfg(feature = "debug-builtins")]
            wasmtime_runtime::debug_builtins::ensure_exported();
        }

        let config = config.clone();
        let tunables = config.validate()?;

        #[cfg(any(feature = "cranelift", feature = "winch"))]
        let (config, compiler) = config.build_compiler(&tunables)?;

        Ok(Engine {
            inner: Arc::new(EngineInner {
                #[cfg(any(feature = "cranelift", feature = "winch"))]
                compiler,
                #[cfg(feature = "runtime")]
                allocator: config.build_allocator(&tunables)?,
                #[cfg(feature = "runtime")]
                gc_runtime: config.build_gc_runtime()?,
                #[cfg(feature = "runtime")]
                profiler: config.build_profiler()?,
                #[cfg(feature = "runtime")]
                signatures: TypeRegistry::new(),
                #[cfg(feature = "runtime")]
                epoch: AtomicU64::new(0),
                #[cfg(feature = "runtime")]
                unique_id_allocator: wasmtime_runtime::CompiledModuleIdAllocator::new(),
                compatible_with_native_host: OnceCell::new(),
                config,
                tunables,
            }),
        })
    }

    /// Returns the configuration settings that this engine is using.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    pub(crate) fn run_maybe_parallel<
        A: Send,
        B: Send,
        E: Send,
        F: Fn(A) -> Result<B, E> + Send + Sync,
    >(
        &self,
        input: Vec<A>,
        f: F,
    ) -> Result<Vec<B>, E> {
        if self.config().parallel_compilation {
            #[cfg(feature = "parallel-compilation")]
            {
                use rayon::prelude::*;
                return input
                    .into_par_iter()
                    .map(|a| f(a))
                    .collect::<Result<Vec<B>, E>>();
            }
        }

        // In case the parallel-compilation feature is disabled or the parallel_compilation config
        // was turned off dynamically fallback to the non-parallel version.
        input
            .into_iter()
            .map(|a| f(a))
            .collect::<Result<Vec<B>, E>>()
    }

    /// Take a weak reference to this engine.
    pub fn weak(&self) -> EngineWeak {
        EngineWeak {
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub(crate) fn tunables(&self) -> &Tunables {
        &self.inner.tunables
    }

    /// Returns whether the engine `a` and `b` refer to the same configuration.
    #[inline]
    pub fn same(a: &Engine, b: &Engine) -> bool {
        Arc::ptr_eq(&a.inner, &b.inner)
    }

    /// Detects whether the bytes provided are a precompiled object produced by
    /// Wasmtime.
    ///
    /// This function will inspect the header of `bytes` to determine if it
    /// looks like a precompiled core wasm module or a precompiled component.
    /// This does not validate the full structure or guarantee that
    /// deserialization will succeed, instead it helps higher-levels of the
    /// stack make a decision about what to do next when presented with the
    /// `bytes` as an input module.
    ///
    /// If the `bytes` looks like a precompiled object previously produced by
    /// [`Module::serialize`](crate::Module::serialize),
    /// [`Component::serialize`](crate::component::Component::serialize),
    /// [`Engine::precompile_module`], or [`Engine::precompile_component`], then
    /// this will return `Some(...)` indicating so. Otherwise `None` is
    /// returned.
    pub fn detect_precompiled(&self, bytes: &[u8]) -> Option<Precompiled> {
        serialization::detect_precompiled_bytes(bytes)
    }

    /// Like [`Engine::detect_precompiled`], but performs the detection on a file.
    pub fn detect_precompiled_file(&self, path: impl AsRef<Path>) -> Result<Option<Precompiled>> {
        serialization::detect_precompiled_file(path)
    }

    /// Returns the target triple which this engine is compiling code for
    /// and/or running code for.
    pub(crate) fn target(&self) -> target_lexicon::Triple {
        // If a compiler is configured, use that target.
        #[cfg(any(feature = "cranelift", feature = "winch"))]
        return self.compiler().triple().clone();

        // ... otherwise it's the native target
        #[cfg(not(any(feature = "cranelift", feature = "winch")))]
        return target_lexicon::Triple::host();
    }

    /// Verify that this engine's configuration is compatible with loading
    /// modules onto the native host platform.
    ///
    /// This method is used as part of `Module::new` to ensure that this
    /// engine can indeed load modules for the configured compiler (if any).
    /// Note that if cranelift is disabled this trivially returns `Ok` because
    /// loaded serialized modules are checked separately.
    pub(crate) fn check_compatible_with_native_host(&self) -> Result<()> {
        self.inner
            .compatible_with_native_host
            .get_or_init(|| self._check_compatible_with_native_host())
            .clone()
            .map_err(anyhow::Error::msg)
    }

    fn _check_compatible_with_native_host(&self) -> Result<(), String> {
        #[cfg(any(feature = "cranelift", feature = "winch"))]
        {
            let compiler = self.compiler();

            // Check to see that the config's target matches the host
            let target = compiler.triple();
            if *target != target_lexicon::Triple::host() {
                return Err(format!(
                    "target '{}' specified in the configuration does not match the host",
                    target
                ));
            }

            // Also double-check all compiler settings
            for (key, value) in compiler.flags().iter() {
                self.check_compatible_with_shared_flag(key, value)?;
            }
            for (key, value) in compiler.isa_flags().iter() {
                self.check_compatible_with_isa_flag(key, value)?;
            }
        }
        Ok(())
    }

    /// Checks to see whether the "shared flag", something enabled for
    /// individual compilers, is compatible with the native host platform.
    ///
    /// This is used both when validating an engine's compilation settings are
    /// compatible with the host as well as when deserializing modules from
    /// disk to ensure they're compatible with the current host.
    ///
    /// Note that most of the settings here are not configured by users that
    /// often. While theoretically possible via `Config` methods the more
    /// interesting flags are the ISA ones below. Typically the values here
    /// represent global configuration for wasm features. Settings here
    /// currently rely on the compiler informing us of all settings, including
    /// those disabled. Settings then fall in a few buckets:
    ///
    /// * Some settings must be enabled, such as `preserve_frame_pointers`.
    /// * Some settings must have a particular value, such as
    ///   `libcall_call_conv`.
    /// * Some settings do not matter as to their value, such as `opt_level`.
    pub(crate) fn check_compatible_with_shared_flag(
        &self,
        flag: &str,
        value: &FlagValue,
    ) -> Result<(), String> {
        let target = self.target();
        let ok = match flag {
            // These settings must all have be enabled, since their value
            // can affect the way the generated code performs or behaves at
            // runtime.
            "libcall_call_conv" => *value == FlagValue::Enum("isa_default".into()),
            "preserve_frame_pointers" => *value == FlagValue::Bool(true),
            "enable_probestack" => *value == FlagValue::Bool(crate::config::probestack_supported(target.architecture)),
            "probestack_strategy" => *value == FlagValue::Enum("inline".into()),

            // Features wasmtime doesn't use should all be disabled, since
            // otherwise if they are enabled it could change the behavior of
            // generated code.
            "enable_llvm_abi_extensions" => *value == FlagValue::Bool(false),
            "enable_pinned_reg" => *value == FlagValue::Bool(false),
            "use_colocated_libcalls" => *value == FlagValue::Bool(false),
            "use_pinned_reg_as_heap_base" => *value == FlagValue::Bool(false),

            // If reference types (or anything that depends on reference types,
            // like typed function references and GC) are enabled this must be
            // enabled, otherwise this setting can have any value.
            "enable_safepoints" => {
                if self.config().features.reference_types {
                    *value == FlagValue::Bool(true)
                } else {
                    return Ok(())
                }
            }

            // Windows requires unwind info as part of its ABI.
            "unwind_info" => {
                if target.operating_system == target_lexicon::OperatingSystem::Windows {
                    *value == FlagValue::Bool(true)
                } else {
                    return Ok(())
                }
            }

            // These settings don't affect the interface or functionality of
            // the module itself, so their configuration values shouldn't
            // matter.
            "enable_heap_access_spectre_mitigation"
            | "enable_table_access_spectre_mitigation"
            | "enable_nan_canonicalization"
            | "enable_jump_tables"
            | "enable_float"
            | "enable_verifier"
            | "enable_pcc"
            | "regalloc_checker"
            | "regalloc_verbose_logs"
            | "is_pic"
            | "bb_padding_log2_minus_one"
            | "machine_code_cfg_info"
            | "tls_model" // wasmtime doesn't use tls right now
            | "opt_level" // opt level doesn't change semantics
            | "enable_alias_analysis" // alias analysis-based opts don't change semantics
            | "probestack_func_adjusts_sp" // probestack above asserted disabled
            | "probestack_size_log2" // probestack above asserted disabled
            | "regalloc" // shouldn't change semantics
            | "enable_incremental_compilation_cache_checks" // shouldn't change semantics
            | "enable_atomics" => return Ok(()),

            // Everything else is unknown and needs to be added somewhere to
            // this list if encountered.
            _ => {
                return Err(format!("unknown shared setting {:?} configured to {:?}", flag, value))
            }
        };

        if !ok {
            return Err(format!(
                "setting {:?} is configured to {:?} which is not supported",
                flag, value,
            ));
        }
        Ok(())
    }

    /// Same as `check_compatible_with_native_host` except used for ISA-specific
    /// flags. This is used to test whether a configured ISA flag is indeed
    /// available on the host platform itself.
    pub(crate) fn check_compatible_with_isa_flag(
        &self,
        flag: &str,
        value: &FlagValue,
    ) -> Result<(), String> {
        match value {
            // ISA flags are used for things like CPU features, so if they're
            // disabled then it's compatible with the native host.
            FlagValue::Bool(false) => return Ok(()),

            // Fall through below where we test at runtime that features are
            // available.
            FlagValue::Bool(true) => {}

            // Only `bool` values are supported right now, other settings would
            // need more support here.
            _ => {
                return Err(format!(
                    "isa-specific feature {:?} configured to unknown value {:?}",
                    flag, value
                ))
            }
        }

        #[allow(unused_assignments)]
        let mut enabled = None;

        #[cfg(target_arch = "aarch64")]
        {
            enabled = match flag {
                "has_lse" => Some(std::arch::is_aarch64_feature_detected!("lse")),
                // No effect on its own, but in order to simplify the code on a
                // platform without pointer authentication support we fail if
                // "has_pauth" is enabled, but "sign_return_address" is not.
                "has_pauth" => Some(std::arch::is_aarch64_feature_detected!("paca")),
                // No effect on its own.
                "sign_return_address_all" => Some(true),
                // The pointer authentication instructions act as a `NOP` when
                // unsupported (but keep in mind "has_pauth" as well), so it is
                // safe to enable them.
                "sign_return_address" => Some(true),
                // No effect on its own.
                "sign_return_address_with_bkey" => Some(true),
                // The `BTI` instruction acts as a `NOP` when unsupported, so it
                // is safe to enable it.
                "use_bti" => Some(true),
                // fall through to the very bottom to indicate that support is
                // not enabled to test whether this feature is enabled on the
                // host.
                _ => None,
            };
        }

        // There is no is_s390x_feature_detected macro yet, so for now
        // we use getauxval from the libc crate directly.
        #[cfg(all(target_arch = "s390x", target_os = "linux"))]
        {
            let v = unsafe { libc::getauxval(libc::AT_HWCAP) };
            const HWCAP_S390X_VXRS_EXT2: libc::c_ulong = 32768;

            enabled = match flag {
                // There is no separate HWCAP bit for mie2, so assume
                // that any machine with vxrs_ext2 also has mie2.
                "has_vxrs_ext2" | "has_mie2" => Some((v & HWCAP_S390X_VXRS_EXT2) != 0),
                // fall through to the very bottom to indicate that support is
                // not enabled to test whether this feature is enabled on the
                // host.
                _ => None,
            }
        }

        #[cfg(target_arch = "riscv64")]
        {
            enabled = match flag {
                // make sure `test_isa_flags_mismatch` test pass.
                "not_a_flag" => None,
                // due to `is_riscv64_feature_detected` is not stable.
                // we cannot use it.
                _ => Some(true),
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            enabled = match flag {
                "has_sse3" => Some(std::is_x86_feature_detected!("sse3")),
                "has_ssse3" => Some(std::is_x86_feature_detected!("ssse3")),
                "has_sse41" => Some(std::is_x86_feature_detected!("sse4.1")),
                "has_sse42" => Some(std::is_x86_feature_detected!("sse4.2")),
                "has_popcnt" => Some(std::is_x86_feature_detected!("popcnt")),
                "has_avx" => Some(std::is_x86_feature_detected!("avx")),
                "has_avx2" => Some(std::is_x86_feature_detected!("avx2")),
                "has_fma" => Some(std::is_x86_feature_detected!("fma")),
                "has_bmi1" => Some(std::is_x86_feature_detected!("bmi1")),
                "has_bmi2" => Some(std::is_x86_feature_detected!("bmi2")),
                "has_avx512bitalg" => Some(std::is_x86_feature_detected!("avx512bitalg")),
                "has_avx512dq" => Some(std::is_x86_feature_detected!("avx512dq")),
                "has_avx512f" => Some(std::is_x86_feature_detected!("avx512f")),
                "has_avx512vl" => Some(std::is_x86_feature_detected!("avx512vl")),
                "has_avx512vbmi" => Some(std::is_x86_feature_detected!("avx512vbmi")),
                "has_lzcnt" => Some(std::is_x86_feature_detected!("lzcnt")),

                // fall through to the very bottom to indicate that support is
                // not enabled to test whether this feature is enabled on the
                // host.
                _ => None,
            };
        }

        #[cfg(target_family = "wasm")]
        {
            let _ = &mut enabled;
        }

        match enabled {
            Some(true) => return Ok(()),
            Some(false) => {
                return Err(format!(
                    "compilation setting {:?} is enabled, but not available on the host",
                    flag
                ))
            }
            // fall through
            None => {}
        }

        Err(format!(
            "cannot test if target-specific flag {:?} is available at runtime",
            flag
        ))
    }
}

#[cfg(any(feature = "cranelift", feature = "winch"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cranelift", feature = "winch"))))]
impl Engine {
    pub(crate) fn compiler(&self) -> &dyn wasmtime_environ::Compiler {
        &*self.inner.compiler
    }

    /// Ahead-of-time (AOT) compiles a WebAssembly module.
    ///
    /// The `bytes` provided must be in one of two formats:
    ///
    /// * A [binary-encoded][binary] WebAssembly module. This is always supported.
    /// * A [text-encoded][text] instance of the WebAssembly text format.
    ///   This is only supported when the `wat` feature of this crate is enabled.
    ///   If this is supplied then the text format will be parsed before validation.
    ///   Note that the `wat` feature is enabled by default.
    ///
    /// This method may be used to compile a module for use with a different target
    /// host. The output of this method may be used with
    /// [`Module::deserialize`](crate::Module::deserialize) on hosts compatible
    /// with the [`Config`](crate::Config) associated with this [`Engine`].
    ///
    /// The output of this method is safe to send to another host machine for later
    /// execution. As the output is already a compiled module, translation and code
    /// generation will be skipped and this will improve the performance of constructing
    /// a [`Module`](crate::Module) from the output of this method.
    ///
    /// [binary]: https://webassembly.github.io/spec/core/binary/index.html
    /// [text]: https://webassembly.github.io/spec/core/text/index.html
    pub fn precompile_module(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        crate::CodeBuilder::new(self)
            .wasm(bytes, None)?
            .compile_module_serialized()
    }

    /// Same as [`Engine::precompile_module`] except for a
    /// [`Component`](crate::component::Component)
    #[cfg(feature = "component-model")]
    #[cfg_attr(docsrs, doc(cfg(feature = "component-model")))]
    pub fn precompile_component(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        crate::CodeBuilder::new(self)
            .wasm(bytes, None)?
            .compile_component_serialized()
    }

    /// Produces a blob of bytes by serializing the `engine`'s configuration data to
    /// be checked, perhaps in a different process, with the `check_compatible`
    /// method below.
    ///
    /// The blob of bytes is inserted into the object file specified to become part
    /// of the final compiled artifact.
    pub(crate) fn append_compiler_info(&self, obj: &mut Object<'_>) {
        serialization::append_compiler_info(self, obj, &serialization::Metadata::new(&self))
    }

    pub(crate) fn append_bti(&self, obj: &mut Object<'_>) {
        let section = obj.add_section(
            obj.segment_name(StandardSegment::Data).to_vec(),
            obj::ELF_WASM_BTI.as_bytes().to_vec(),
            SectionKind::ReadOnlyData,
        );
        let contents = if self.compiler().is_branch_protection_enabled() {
            1
        } else {
            0
        };
        obj.append_section_data(section, &[contents], 1);
    }
}

/// Return value from the [`Engine::detect_precompiled`] API.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Precompiled {
    /// The input bytes look like a precompiled core wasm module.
    Module,
    /// The input bytes look like a precompiled wasm component.
    Component,
}

#[cfg(feature = "runtime")]
#[cfg_attr(docsrs, doc(cfg(feature = "runtime")))]
impl Engine {
    /// Eagerly initialize thread-local functionality shared by all [`Engine`]s.
    ///
    /// Wasmtime's implementation on some platforms may involve per-thread
    /// setup that needs to happen whenever WebAssembly is invoked. This setup
    /// can take on the order of a few hundred microseconds, whereas the
    /// overhead of calling WebAssembly is otherwise on the order of a few
    /// nanoseconds. This setup cost is paid once per-OS-thread. If your
    /// application is sensitive to the latencies of WebAssembly function
    /// calls, even those that happen first on a thread, then this function
    /// can be used to improve the consistency of each call into WebAssembly
    /// by explicitly frontloading the cost of the one-time setup per-thread.
    ///
    /// Note that this function is not required to be called in any embedding.
    /// Wasmtime will automatically initialize thread-local-state as necessary
    /// on calls into WebAssembly. This is provided for use cases where the
    /// latency of WebAssembly calls are extra-important, which is not
    /// necessarily true of all embeddings.
    pub fn tls_eager_initialize() {
        wasmtime_runtime::tls_eager_initialize();
    }

    pub(crate) fn allocator(&self) -> &dyn wasmtime_runtime::InstanceAllocator {
        self.inner.allocator.as_ref()
    }

    pub(crate) fn gc_runtime(&self) -> &Arc<dyn GcRuntime> {
        &self.inner.gc_runtime
    }

    pub(crate) fn profiler(&self) -> &dyn crate::profiling_agent::ProfilingAgent {
        self.inner.profiler.as_ref()
    }

    #[cfg(feature = "cache")]
    pub(crate) fn cache_config(&self) -> &wasmtime_cache::CacheConfig {
        &self.config().cache_config
    }

    pub(crate) fn signatures(&self) -> &TypeRegistry {
        &self.inner.signatures
    }

    pub(crate) fn epoch_counter(&self) -> &AtomicU64 {
        &self.inner.epoch
    }

    pub(crate) fn current_epoch(&self) -> u64 {
        self.epoch_counter().load(Ordering::Relaxed)
    }

    /// Increments the epoch.
    ///
    /// When using epoch-based interruption, currently-executing Wasm
    /// code within this engine will trap or yield "soon" when the
    /// epoch deadline is reached or exceeded. (The configuration, and
    /// the deadline, are set on the `Store`.) The intent of the
    /// design is for this method to be called by the embedder at some
    /// regular cadence, for example by a thread that wakes up at some
    /// interval, or by a signal handler.
    ///
    /// See [`Config::epoch_interruption`](crate::Config::epoch_interruption)
    /// for an introduction to epoch-based interruption and pointers
    /// to the other relevant methods.
    ///
    /// When performing `increment_epoch` in a separate thread, consider using
    /// [`Engine::weak`] to hold an [`EngineWeak`](crate::EngineWeak) and
    /// performing [`EngineWeak::upgrade`](crate::EngineWeak::upgrade) on each
    /// tick, so that the epoch ticking thread does not keep an [`Engine`] alive
    /// longer than any of its consumers.
    ///
    /// ## Signal Safety
    ///
    /// This method is signal-safe: it does not make any syscalls, and
    /// performs only an atomic increment to the epoch value in
    /// memory.
    pub fn increment_epoch(&self) {
        self.inner.epoch.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn unique_id_allocator(&self) -> &wasmtime_runtime::CompiledModuleIdAllocator {
        &self.inner.unique_id_allocator
    }

    /// Returns a [`std::hash::Hash`] that can be used to check precompiled WebAssembly compatibility.
    ///
    /// The outputs of [`Engine::precompile_module`] and [`Engine::precompile_component`]
    /// are compatible with a different [`Engine`] instance only if the two engines use
    /// compatible [`Config`]s. If this Hash matches between two [`Engine`]s then binaries
    /// from one are guaranteed to deserialize in the other.
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "cranelift")))] // see build.rs
    pub fn precompile_compatibility_hash(&self) -> impl std::hash::Hash + '_ {
        crate::compile::HashedEngineCompileEnv(self)
    }

    /// Executes `f1` and `f2` in parallel if parallel compilation is enabled at
    /// both runtime and compile time, otherwise runs them synchronously.
    #[allow(dead_code)] // only used for the component-model feature right now
    pub(crate) fn join_maybe_parallel<T, U>(
        &self,
        f1: impl FnOnce() -> T + Send,
        f2: impl FnOnce() -> U + Send,
    ) -> (T, U)
    where
        T: Send,
        U: Send,
    {
        if self.config().parallel_compilation {
            #[cfg(feature = "parallel-compilation")]
            return rayon::join(f1, f2);
        }
        (f1(), f2())
    }

    /// Loads a `CodeMemory` from the specified in-memory slice, copying it to a
    /// uniquely owned mmap.
    ///
    /// The `expected` marker here is whether the bytes are expected to be a
    /// precompiled module or a component.
    pub(crate) fn load_code_bytes(
        &self,
        bytes: &[u8],
        expected: ObjectKind,
    ) -> Result<Arc<crate::CodeMemory>> {
        self.load_code(wasmtime_runtime::MmapVec::from_slice(bytes)?, expected)
    }

    /// Like `load_code_bytes`, but creates a mmap from a file on disk.
    pub(crate) fn load_code_file(
        &self,
        path: &Path,
        expected: ObjectKind,
    ) -> Result<Arc<crate::CodeMemory>> {
        self.load_code(
            wasmtime_runtime::MmapVec::from_file(path).with_context(|| {
                format!("failed to create file mapping for: {}", path.display())
            })?,
            expected,
        )
    }

    pub(crate) fn load_code(
        &self,
        mmap: wasmtime_runtime::MmapVec,
        expected: ObjectKind,
    ) -> Result<Arc<crate::CodeMemory>> {
        serialization::check_compatible(self, &mmap, expected)?;
        let mut code = crate::CodeMemory::new(mmap)?;
        code.publish()?;
        Ok(Arc::new(code))
    }
}

/// A weak reference to an [`Engine`].
#[derive(Clone)]
pub struct EngineWeak {
    inner: std::sync::Weak<EngineInner>,
}

impl EngineWeak {
    /// Upgrade this weak reference into an [`Engine`]. Returns `None` if
    /// strong references (the [`Engine`] type itself) no longer exist.
    pub fn upgrade(&self) -> Option<Engine> {
        std::sync::Weak::upgrade(&self.inner).map(|inner| Engine { inner })
    }
}
