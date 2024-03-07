//! Definitions of runtime structures and metadata which are serialized into ELF
//! with `bincode` as part of a module's compilation process.

use crate::{
    obj, DefinedFuncIndex, FuncIndex, FunctionLoc, MemoryInitialization, Module, ModuleTranslation,
    PrimaryMap, Tunables, WasmFunctionInfo,
};
use anyhow::{bail, Result};
use object::write::{Object, SectionId, StandardSegment, WritableBuffer};
use object::SectionKind;
use serde_derive::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::ops::Range;
use std::str;
use wasmtime_types::ModuleInternedTypeIndex;

/// Secondary in-memory results of function compilation.
#[derive(Serialize, Deserialize)]
pub struct CompiledFunctionInfo {
    /// The [`WasmFunctionInfo`] for this function.
    pub wasm_func_info: WasmFunctionInfo,
    /// The [`FunctionLoc`] indicating the location of this function in the text
    /// section of the compition artifact.
    pub wasm_func_loc: FunctionLoc,
    /// A trampoline for array callers (e.g. `Func::new`) calling into this function (if needed).
    pub array_to_wasm_trampoline: Option<FunctionLoc>,
    /// A trampoline for native callers (e.g. `Func::wrap`) calling into this function (if needed).
    pub native_to_wasm_trampoline: Option<FunctionLoc>,
}

/// Secondary in-memory results of module compilation.
///
/// This opaque structure can be optionally passed back to
/// `CompiledModule::from_artifacts` to avoid decoding extra information there.
#[derive(Serialize, Deserialize)]
pub struct CompiledModuleInfo {
    /// Type information about the compiled WebAssembly module.
    pub module: Module,

    /// Metadata about each compiled function.
    pub funcs: PrimaryMap<DefinedFuncIndex, CompiledFunctionInfo>,

    /// Sorted list, by function index, of names we have for this module.
    pub func_names: Vec<FunctionName>,

    /// Metadata about wasm-to-native trampolines. Used when exposing a native
    /// callee (e.g. `Func::wrap`) to a Wasm caller. Sorted by signature index.
    pub wasm_to_native_trampolines: Vec<(ModuleInternedTypeIndex, FunctionLoc)>,

    /// General compilation metadata.
    pub meta: Metadata,
}

/// The name of a function stored in the
/// [`ELF_NAME_DATA`](crate::obj::ELF_NAME_DATA) section.
#[derive(Serialize, Deserialize)]
pub struct FunctionName {
    /// The Wasm function index of this function.
    pub idx: FuncIndex,
    /// The offset of the name in the
    /// [`ELF_NAME_DATA`](crate::obj::ELF_NAME_DATA) section.
    pub offset: u32,
    /// The length of the name in bytes.
    pub len: u32,
}

/// Metadata associated with a compiled ELF artifact.
#[derive(Serialize, Deserialize)]
pub struct Metadata {
    /// Whether or not native debug information is available in `obj`
    pub native_debug_info_present: bool,

    /// Whether or not the original wasm module contained debug information that
    /// we skipped and did not parse.
    pub has_unparsed_debuginfo: bool,

    /// Offset in the original wasm file to the code section.
    pub code_section_offset: u64,

    /// Whether or not custom wasm-specific dwarf sections were inserted into
    /// the ELF image.
    ///
    /// Note that even if this flag is `true` sections may be missing if they
    /// weren't found in the original wasm module itself.
    pub has_wasm_debuginfo: bool,

    /// Dwarf sections and the offsets at which they're stored in the
    /// ELF_WASMTIME_DWARF
    pub dwarf: Vec<(u8, Range<u64>)>,
}

/// Helper structure to create an ELF file as a compilation artifact.
///
/// This structure exposes the process which Wasmtime will encode a core wasm
/// module into an ELF file, notably managing data sections and all that good
/// business going into the final file.
pub struct ObjectBuilder<'a> {
    /// The `object`-crate-defined ELF file write we're using.
    obj: Object<'a>,

    /// General compilation configuration.
    tunables: &'a Tunables,

    /// The section identifier for "rodata" which is where wasm data segments
    /// will go.
    data: SectionId,

    /// The section identifier for function name information, or otherwise where
    /// the `name` custom section of wasm is copied into.
    ///
    /// This is optional and lazily created on demand.
    names: Option<SectionId>,

    /// The section identifier for dwarf information copied from the original
    /// wasm files.
    ///
    /// This is optional and lazily created on demand.
    dwarf: Option<SectionId>,
}

impl<'a> ObjectBuilder<'a> {
    /// Creates a new builder for the `obj` specified.
    pub fn new(mut obj: Object<'a>, tunables: &'a Tunables) -> ObjectBuilder<'a> {
        let data = obj.add_section(
            obj.segment_name(StandardSegment::Data).to_vec(),
            obj::ELF_WASM_DATA.as_bytes().to_vec(),
            SectionKind::ReadOnlyData,
        );
        ObjectBuilder {
            obj,
            tunables,
            data,
            names: None,
            dwarf: None,
        }
    }

    /// Completes compilation of the `translation` specified, inserting
    /// everything necessary into the `Object` being built.
    ///
    /// This function will consume the final results of compiling a wasm module
    /// and finish the ELF image in-progress as part of `self.obj` by appending
    /// any compiler-agnostic sections.
    ///
    /// The auxiliary `CompiledModuleInfo` structure returned here has also been
    /// serialized into the object returned, but if the caller will quickly
    /// turn-around and invoke `CompiledModule::from_artifacts` after this then
    /// the information can be passed to that method to avoid extra
    /// deserialization. This is done to avoid a serialize-then-deserialize for
    /// API calls like `Module::new` where the compiled module is immediately
    /// going to be used.
    ///
    /// The various arguments here are:
    ///
    /// * `translation` - the core wasm translation that's being completed.
    ///
    /// * `funcs` - compilation metadata about functions within the translation
    ///   as well as where the functions are located in the text section and any
    ///   associated trampolines.
    ///
    /// * `wasm_to_native_trampolines` - list of all trampolines necessary for
    ///   Wasm callers calling native callees (e.g. `Func::wrap`). One for each
    ///   function signature in the module. Must be sorted by `SignatureIndex`.
    ///
    /// Returns the `CompiledModuleInfo` corresponding to this core Wasm module
    /// as a result of this append operation. This is then serialized into the
    /// final artifact by the caller.
    pub fn append(
        &mut self,
        translation: ModuleTranslation<'_>,
        funcs: PrimaryMap<DefinedFuncIndex, CompiledFunctionInfo>,
        wasm_to_native_trampolines: Vec<(ModuleInternedTypeIndex, FunctionLoc)>,
    ) -> Result<CompiledModuleInfo> {
        let ModuleTranslation {
            mut module,
            debuginfo,
            has_unparsed_debuginfo,
            data,
            data_align,
            passive_data,
            ..
        } = translation;

        // Place all data from the wasm module into a section which will the
        // source of the data later at runtime. This additionally keeps track of
        // the offset of
        let mut total_data_len = 0;
        let data_offset = self
            .obj
            .append_section_data(self.data, &[], data_align.unwrap_or(1));
        for (i, data) in data.iter().enumerate() {
            // The first data segment has its alignment specified as the alignment
            // for the entire section, but everything afterwards is adjacent so it
            // has alignment of 1.
            let align = if i == 0 { data_align.unwrap_or(1) } else { 1 };
            self.obj.append_section_data(self.data, data, align);
            total_data_len += data.len();
        }
        for data in passive_data.iter() {
            self.obj.append_section_data(self.data, data, 1);
        }

        // If any names are present in the module then the `ELF_NAME_DATA` section
        // is create and appended.
        let mut func_names = Vec::new();
        if debuginfo.name_section.func_names.len() > 0 {
            let name_id = *self.names.get_or_insert_with(|| {
                self.obj.add_section(
                    self.obj.segment_name(StandardSegment::Data).to_vec(),
                    obj::ELF_NAME_DATA.as_bytes().to_vec(),
                    SectionKind::ReadOnlyData,
                )
            });
            let mut sorted_names = debuginfo.name_section.func_names.iter().collect::<Vec<_>>();
            sorted_names.sort_by_key(|(idx, _name)| *idx);
            for (idx, name) in sorted_names {
                let offset = self.obj.append_section_data(name_id, name.as_bytes(), 1);
                let offset = match u32::try_from(offset) {
                    Ok(offset) => offset,
                    Err(_) => bail!("name section too large (> 4gb)"),
                };
                let len = u32::try_from(name.len()).unwrap();
                func_names.push(FunctionName {
                    idx: *idx,
                    offset,
                    len,
                });
            }
        }

        // Data offsets in `MemoryInitialization` are offsets within the
        // `translation.data` list concatenated which is now present in the data
        // segment that's appended to the object. Increase the offsets by
        // `self.data_size` to account for any previously added module.
        let data_offset = u32::try_from(data_offset).unwrap();
        match &mut module.memory_initialization {
            MemoryInitialization::Segmented(list) => {
                for segment in list {
                    segment.data.start = segment.data.start.checked_add(data_offset).unwrap();
                    segment.data.end = segment.data.end.checked_add(data_offset).unwrap();
                }
            }
            MemoryInitialization::Static { map } => {
                for (_, segment) in map {
                    if let Some(segment) = segment {
                        segment.data.start = segment.data.start.checked_add(data_offset).unwrap();
                        segment.data.end = segment.data.end.checked_add(data_offset).unwrap();
                    }
                }
            }
        }

        // Data offsets for passive data are relative to the start of
        // `translation.passive_data` which was appended to the data segment
        // of this object, after active data in `translation.data`. Update the
        // offsets to account prior modules added in addition to active data.
        let data_offset = data_offset + u32::try_from(total_data_len).unwrap();
        for (_, range) in module.passive_data_map.iter_mut() {
            range.start = range.start.checked_add(data_offset).unwrap();
            range.end = range.end.checked_add(data_offset).unwrap();
        }

        // Insert the wasm raw wasm-based debuginfo into the output, if
        // requested. Note that this is distinct from the native debuginfo
        // possibly generated by the native compiler, hence these sections
        // getting wasm-specific names.
        let mut dwarf = Vec::new();
        if self.tunables.parse_wasm_debuginfo {
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_abbrev);
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_addr);
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_aranges);
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_info);
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_line);
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_line_str);
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_str);
            self.push_debug(&mut dwarf, &debuginfo.dwarf.debug_str_offsets);
            self.push_debug(&mut dwarf, &debuginfo.debug_ranges);
            self.push_debug(&mut dwarf, &debuginfo.debug_rnglists);

            if let Some(dwarf_package) = debuginfo.dwarf_package {
                self.push_debug(&mut dwarf, &dwarf_package.debug_abbrev);
                self.push_debug(&mut dwarf, &dwarf_package.debug_info);
                self.push_debug(&mut dwarf, &dwarf_package.debug_line);
                self.push_debug(&mut dwarf, &dwarf_package.debug_str);
                self.push_debug(&mut dwarf, &dwarf_package.debug_str_offsets);
                self.push_debug(&mut dwarf, &dwarf_package.debug_rnglists);
            }
        }
        // Sort this for binary-search-lookup later in `symbolize_context`.
        dwarf.sort_by_key(|(id, _)| *id);

        Ok(CompiledModuleInfo {
            module,
            funcs,
            wasm_to_native_trampolines,
            func_names,
            meta: Metadata {
                native_debug_info_present: self.tunables.generate_native_debuginfo,
                has_unparsed_debuginfo,
                code_section_offset: debuginfo.wasm_file.code_section_offset,
                has_wasm_debuginfo: self.tunables.parse_wasm_debuginfo,
                dwarf,
            },
        })
    }

    fn push_debug<'b, T>(&mut self, dwarf: &mut Vec<(u8, Range<u64>)>, section: &T)
    where
        T: gimli::Section<gimli::EndianSlice<'b, gimli::LittleEndian>>,
    {
        let data = section.reader().slice();
        if data.is_empty() {
            return;
        }
        let section_id = *self.dwarf.get_or_insert_with(|| {
            self.obj.add_section(
                self.obj.segment_name(StandardSegment::Debug).to_vec(),
                obj::ELF_WASMTIME_DWARF.as_bytes().to_vec(),
                SectionKind::Debug,
            )
        });
        let offset = self.obj.append_section_data(section_id, data, 1);
        dwarf.push((T::id() as u8, offset..offset + data.len() as u64));
    }

    /// Creates the `ELF_WASMTIME_INFO` section from the given serializable data
    /// structure.
    pub fn serialize_info<T>(&mut self, info: &T)
    where
        T: serde::Serialize,
    {
        let section = self.obj.add_section(
            self.obj.segment_name(StandardSegment::Data).to_vec(),
            obj::ELF_WASMTIME_INFO.as_bytes().to_vec(),
            SectionKind::ReadOnlyData,
        );
        let data = bincode::serialize(info).unwrap();
        self.obj.set_section_data(section, data, 1);
    }

    /// Serializes `self` into a buffer. This can be used for execution as well
    /// as serialization.
    pub fn finish<T: WritableBuffer>(self, t: &mut T) -> Result<()> {
        self.obj.emit(t).map_err(|e| e.into())
    }
}

/// A type which can be the result of serializing an object.
pub trait FinishedObject: Sized {
    /// Emit the object as `Self`.
    fn finish_object(obj: ObjectBuilder<'_>) -> Result<Self>;
}

impl FinishedObject for Vec<u8> {
    fn finish_object(obj: ObjectBuilder<'_>) -> Result<Self> {
        let mut result = ObjectVec::default();
        obj.finish(&mut result)?;
        return Ok(result.0);

        #[derive(Default)]
        struct ObjectVec(Vec<u8>);

        impl WritableBuffer for ObjectVec {
            fn len(&self) -> usize {
                self.0.len()
            }

            fn reserve(&mut self, additional: usize) -> Result<(), ()> {
                assert_eq!(self.0.len(), 0, "cannot reserve twice");
                self.0 = Vec::with_capacity(additional);
                Ok(())
            }

            fn resize(&mut self, new_len: usize) {
                if new_len <= self.0.len() {
                    self.0.truncate(new_len)
                } else {
                    self.0.extend(vec![0; new_len - self.0.len()])
                }
            }

            fn write_bytes(&mut self, val: &[u8]) {
                self.0.extend(val);
            }
        }
    }
}
