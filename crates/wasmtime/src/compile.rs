use std::mem;

use anyhow::{Context, Result};
use object::write::{Object, StandardSegment};
use object::SectionKind;
#[cfg(feature = "component-model")]
use wasmtime_environ::component::Translator;
use wasmtime_environ::{
    obj, CompiledModuleInfo, FinishedObject, ModuleEnvironment, ModuleTypes, ObjectKind,
};

use crate::compiler::CompileInputs;
#[cfg(feature = "component-model")]
use crate::component_artifacts::{CompiledComponentInfo, ComponentArtifacts};
use crate::{Engine, Metadata, ModuleVersionStrategy, WasmFeatures, VERSION};

/// Converts an input binary-encoded WebAssembly module to compilation
/// artifacts and type information.
///
/// This is where compilation actually happens of WebAssembly modules and
/// translation/parsing/validation of the binary input occurs. The binary
/// artifact represented in the `MmapVec` returned here is an in-memory ELF
/// file in an owned area of virtual linear memory where permissions (such
/// as the executable bit) can be applied.
///
/// Additionally compilation returns an `Option` here which is always
/// `Some`, notably compiled metadata about the module in addition to the
/// type information found within.
pub(crate) fn build_artifacts<T: FinishedObject>(
    engine: &Engine,
    wasm: &[u8],
) -> Result<(T, Option<(CompiledModuleInfo, ModuleTypes)>)> {
    let tunables = &engine.config().tunables;

    // First a `ModuleEnvironment` is created which records type information
    // about the wasm module. This is where the WebAssembly is parsed and
    // validated. Afterwards `types` will have all the type information for
    // this module.
    let mut validator = wasmparser::Validator::new_with_features(engine.config().features.clone());
    let parser = wasmparser::Parser::new(0);
    let mut types = Default::default();
    let mut translation = ModuleEnvironment::new(tunables, &mut validator, &mut types)
        .translate(parser, wasm)
        .context("failed to parse WebAssembly module")?;
    let functions = mem::take(&mut translation.function_body_inputs);

    let compile_inputs = CompileInputs::for_module(&types, &translation, functions);
    let unlinked_compile_outputs = compile_inputs.compile(engine)?;
    let types = types.finish();
    let (compiled_funcs, function_indices) = unlinked_compile_outputs.pre_link();

    // Emplace all compiled functions into the object file with any other
    // sections associated with code as well.
    let mut object = engine.compiler().object(ObjectKind::Module)?;
    // Insert `Engine` and type-level information into the compiled
    // artifact so if this module is deserialized later it contains all
    // information necessary.
    //
    // Note that `append_compiler_info` and `append_types` here in theory
    // can both be skipped if this module will never get serialized.
    // They're only used during deserialization and not during runtime for
    // the module itself. Currently there's no need for that, however, so
    // it's left as an exercise for later.
    engine.append_compiler_info(&mut object);
    engine.append_bti(&mut object);

    let (mut object, compilation_artifacts) = function_indices.link_and_append_code(
        object,
        engine,
        compiled_funcs,
        std::iter::once(translation).collect(),
    )?;

    let info = compilation_artifacts.unwrap_as_module_info();
    object.serialize_info(&(&info, &types));
    let result = T::finish_object(object)?;

    Ok((result, Some((info, types))))
}

/// Performs the compilation phase for a component, translating and
/// validating the provided wasm binary to machine code.
///
/// This method will compile all nested core wasm binaries in addition to
/// any necessary extra functions required for operation with components.
/// The output artifact here is the serialized object file contained within
/// an owned mmap along with metadata about the compilation itself.
#[cfg(feature = "component-model")]
pub(crate) fn build_component_artifacts<T: FinishedObject>(
    engine: &Engine,
    binary: &[u8],
) -> Result<(T, ComponentArtifacts)> {
    use wasmtime_environ::ScopeVec;

    let tunables = &engine.config().tunables;
    let compiler = engine.compiler();

    let scope = ScopeVec::new();
    let mut validator = wasmparser::Validator::new_with_features(engine.config().features.clone());
    let mut types = Default::default();
    let (component, mut module_translations) =
        Translator::new(tunables, &mut validator, &mut types, &scope)
            .translate(binary)
            .context("failed to parse WebAssembly module")?;

    let compile_inputs = CompileInputs::for_component(
        &types,
        &component,
        module_translations.iter_mut().map(|(i, translation)| {
            let functions = mem::take(&mut translation.function_body_inputs);
            (i, &*translation, functions)
        }),
    );
    let unlinked_compile_outputs = compile_inputs.compile(&engine)?;
    let types = types.finish();
    let (compiled_funcs, function_indices) = unlinked_compile_outputs.pre_link();

    let mut object = compiler.object(ObjectKind::Component)?;
    engine.append_compiler_info(&mut object);
    engine.append_bti(&mut object);

    let (mut object, compilation_artifacts) = function_indices.link_and_append_code(
        object,
        engine,
        compiled_funcs,
        module_translations,
    )?;

    let info = CompiledComponentInfo {
        component: component.component,
        trampolines: compilation_artifacts.trampolines,
        resource_drop_wasm_to_native_trampoline: compilation_artifacts
            .resource_drop_wasm_to_native_trampoline,
    };
    let artifacts = ComponentArtifacts {
        info,
        types,
        static_modules: compilation_artifacts.modules,
    };
    object.serialize_info(&artifacts);

    let result = T::finish_object(object)?;
    Ok((result, artifacts))
}

impl Engine {
    pub(crate) fn compiler(&self) -> &dyn wasmtime_environ::Compiler {
        &*self.inner.compiler
    }

    pub(crate) fn append_compiler_info(&self, obj: &mut Object<'_>) {
        append_compiler_info(self, obj);
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
    #[cfg_attr(nightlydoc, doc(cfg(any(feature = "cranelift", feature = "winch"))))]
    pub fn precompile_module(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "wat")]
        let bytes = wat::parse_bytes(&bytes)?;
        let (v, _) = crate::compile::build_artifacts::<Vec<u8>>(self, &bytes)?;
        Ok(v)
    }

    /// Same as [`Engine::precompile_module`] except for a
    /// [`Component`](crate::component::Component)
    #[cfg_attr(nightlydoc, doc(cfg(any(feature = "cranelift", feature = "winch"))))]
    #[cfg(feature = "component-model")]
    #[cfg_attr(nightlydoc, doc(cfg(feature = "component-model")))]
    pub fn precompile_component(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "wat")]
        let bytes = wat::parse_bytes(&bytes)?;
        let (v, _) = crate::compile::build_component_artifacts::<Vec<u8>>(self, &bytes)?;
        Ok(v)
    }
}

/// Produces a blob of bytes by serializing the `engine`'s configuration data to
/// be checked, perhaps in a different process, with the `check_compatible`
/// method below.
///
/// The blob of bytes is inserted into the object file specified to become part
/// of the final compiled artifact.
pub(crate) fn append_compiler_info(engine: &Engine, obj: &mut Object<'_>) {
    let section = obj.add_section(
        obj.segment_name(StandardSegment::Data).to_vec(),
        obj::ELF_WASM_ENGINE.as_bytes().to_vec(),
        SectionKind::ReadOnlyData,
    );
    let mut data = Vec::new();
    data.push(VERSION);
    let version = match &engine.config().module_version {
        ModuleVersionStrategy::WasmtimeVersion => env!("CARGO_PKG_VERSION"),
        ModuleVersionStrategy::Custom(c) => c,
        ModuleVersionStrategy::None => "",
    };
    // This precondition is checked in Config::module_version:
    assert!(
        version.len() < 256,
        "package version must be less than 256 bytes"
    );
    data.push(version.len() as u8);
    data.extend_from_slice(version.as_bytes());
    bincode::serialize_into(&mut data, &Metadata::new(engine)).unwrap();
    obj.set_section_data(section, data, 1);
}

impl Metadata<'_> {
    pub(crate) fn new(engine: &Engine) -> Metadata<'static> {
        let wasmparser::WasmFeatures {
            reference_types,
            multi_value,
            bulk_memory,
            component_model,
            simd,
            threads,
            tail_call,
            multi_memory,
            exceptions,
            memory64,
            relaxed_simd,
            extended_const,
            memory_control,
            function_references,
            gc,
            component_model_values,
            component_model_nested_names,

            // Always on; we don't currently have knobs for these.
            mutable_global: _,
            saturating_float_to_int: _,
            sign_extension: _,
            floats: _,
        } = engine.config().features;

        assert!(!memory_control);
        assert!(!gc);
        assert!(!component_model_values);
        assert!(!component_model_nested_names);

        Metadata {
            target: engine.compiler().triple().to_string(),
            shared_flags: engine.compiler().flags(),
            isa_flags: engine.compiler().isa_flags(),
            tunables: engine.config().tunables.clone(),
            features: WasmFeatures {
                reference_types,
                multi_value,
                bulk_memory,
                component_model,
                simd,
                threads,
                tail_call,
                multi_memory,
                exceptions,
                memory64,
                relaxed_simd,
                extended_const,
                function_references,
            },
        }
    }
}
