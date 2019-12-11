use crate::r#ref::HostRef;
use crate::runtime::Store;
use crate::types::{
    ExportType, ExternType, FuncType, GlobalType, ImportType, Limits, MemoryType, Mutability,
    TableType, ValType,
};
use anyhow::{Error, Result};
use wasmparser::{
    validate, ExternalKind, ImportSectionEntryType, ModuleReader, OperatorValidatorConfig,
    SectionCode, ValidatingParserConfig,
};
use std::cell::Ref;

fn into_memory_type(mt: wasmparser::MemoryType) -> MemoryType {
    assert!(!mt.shared);
    MemoryType::new(Limits::new(mt.limits.initial, mt.limits.maximum))
}

fn into_global_type(gt: &wasmparser::GlobalType) -> GlobalType {
    let mutability = if gt.mutable {
        Mutability::Var
    } else {
        Mutability::Const
    };
    GlobalType::new(into_valtype(&gt.content_type), mutability)
}

fn into_valtype(ty: &wasmparser::Type) -> ValType {
    use wasmparser::Type::*;
    match ty {
        I32 => ValType::I32,
        I64 => ValType::I64,
        F32 => ValType::F32,
        F64 => ValType::F64,
        V128 => ValType::V128,
        AnyFunc => ValType::FuncRef,
        AnyRef => ValType::AnyRef,
        _ => unimplemented!("types in into_valtype"),
    }
}

fn into_func_type(mt: wasmparser::FuncType) -> FuncType {
    assert_eq!(mt.form, wasmparser::Type::Func);
    let params = mt.params.iter().map(into_valtype).collect::<Vec<_>>();
    let returns = mt.returns.iter().map(into_valtype).collect::<Vec<_>>();
    FuncType::new(params.into_boxed_slice(), returns.into_boxed_slice())
}

fn into_table_type(tt: wasmparser::TableType) -> TableType {
    assert!(
        tt.element_type == wasmparser::Type::AnyFunc || tt.element_type == wasmparser::Type::AnyRef
    );
    let ty = into_valtype(&tt.element_type);
    let limits = Limits::new(tt.limits.initial, tt.limits.maximum);
    TableType::new(ty, limits)
}

fn read_imports_and_exports(binary: &[u8]) -> Result<(Box<[ImportType]>, Box<[ExportType]>)> {
    let mut reader = ModuleReader::new(binary)?;
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut memories = Vec::new();
    let mut tables = Vec::new();
    let mut func_sig = Vec::new();
    let mut sigs = Vec::new();
    let mut globals = Vec::new();
    while !reader.eof() {
        let section = reader.read()?;
        match section.code {
            SectionCode::Memory => {
                let section = section.get_memory_section_reader()?;
                memories.reserve_exact(section.get_count() as usize);
                for entry in section {
                    memories.push(into_memory_type(entry?));
                }
            }
            SectionCode::Type => {
                let section = section.get_type_section_reader()?;
                sigs.reserve_exact(section.get_count() as usize);
                for entry in section {
                    sigs.push(into_func_type(entry?));
                }
            }
            SectionCode::Function => {
                let section = section.get_function_section_reader()?;
                func_sig.reserve_exact(section.get_count() as usize);
                for entry in section {
                    func_sig.push(entry?);
                }
            }
            SectionCode::Global => {
                let section = section.get_global_section_reader()?;
                globals.reserve_exact(section.get_count() as usize);
                for entry in section {
                    globals.push(into_global_type(&entry?.ty));
                }
            }
            SectionCode::Table => {
                let section = section.get_table_section_reader()?;
                tables.reserve_exact(section.get_count() as usize);
                for entry in section {
                    tables.push(into_table_type(entry?))
                }
            }
            SectionCode::Import => {
                let section = section.get_import_section_reader()?;
                imports.reserve_exact(section.get_count() as usize);
                for entry in section {
                    let entry = entry?;
                    let r#type = match entry.ty {
                        ImportSectionEntryType::Function(index) => {
                            func_sig.push(index);
                            let sig = &sigs[index as usize];
                            ExternType::Func(sig.clone())
                        }
                        ImportSectionEntryType::Table(tt) => {
                            let table = into_table_type(tt);
                            tables.push(table.clone());
                            ExternType::Table(table)
                        }
                        ImportSectionEntryType::Memory(mt) => {
                            let memory = into_memory_type(mt);
                            memories.push(memory.clone());
                            ExternType::Memory(memory)
                        }
                        ImportSectionEntryType::Global(gt) => {
                            let global = into_global_type(&gt);
                            globals.push(global.clone());
                            ExternType::Global(global)
                        }
                    };
                    imports.push(ImportType::new(entry.module, entry.field, r#type));
                }
            }
            SectionCode::Export => {
                let section = section.get_export_section_reader()?;
                exports.reserve_exact(section.get_count() as usize);
                for entry in section {
                    let entry = entry?;
                    let r#type = match entry.kind {
                        ExternalKind::Function => {
                            let sig_index = func_sig[entry.index as usize] as usize;
                            let sig = &sigs[sig_index];
                            ExternType::Func(sig.clone())
                        }
                        ExternalKind::Table => {
                            ExternType::Table(tables[entry.index as usize].clone())
                        }
                        ExternalKind::Memory => {
                            ExternType::Memory(memories[entry.index as usize].clone())
                        }
                        ExternalKind::Global => {
                            ExternType::Global(globals[entry.index as usize].clone())
                        }
                    };
                    exports.push(ExportType::new(entry.field, r#type));
                }
            }
            _ => {
                // skip other sections
            }
        }
    }
    Ok((imports.into_boxed_slice(), exports.into_boxed_slice()))
}

/// A Module wraps around some WASM source and manages its imports and exports.
#[derive(Clone)]
pub struct Module {
    inner: HostRef<ModuleInner>
}
impl Module {
    /// Validate and decode the raw wasm data in `binary` and create a new
    /// `Module` in the given `store`.
    #[inline]
    pub fn new(store: &HostRef<Store>, binary: &[u8]) -> Result<Module> {
        Ok(Self {
            inner: HostRef::new(ModuleInner::new(store, binary)?),
        })
    }
    /// Similar to `new`, but does not perform any validation. Only use this
    /// on modules which are known to have been validated already!
    #[inline]
    pub fn new_unchecked(store: &HostRef<Store>, binary: &[u8]) -> Result<Module> {
        Ok(Self {
            inner: HostRef::new(ModuleInner::new_unchecked(store, binary)?),
        })
    }
    pub fn from_exports(store: &HostRef<Store>, exports: Box<[ExportType]>) -> Self {
        Self {
            inner: HostRef::new(ModuleInner::from_exports(store, exports)),
        }
    }

    /// Ensure that a given WASM store has valid syntax and semantics.
    pub fn validate(_store: &HostRef<Store>, binary: &[u8]) -> Result<()> {
        let config = ValidatingParserConfig {
            operator_config: OperatorValidatorConfig {
                enable_threads: false,
                enable_reference_types: false,
                enable_bulk_memory: false,
                enable_simd: false,
                enable_multi_value: true,
            },
        };
        validate(binary, Some(config)).map_err(Error::new)
    }

    pub(crate) fn binary(&self) -> Option<Ref<[u8]>> {
        self.inner.borrow().binary().map(|_| {
            // note that the variable that's passed to this closure
            // cannot be used as the binary that's returned with the ref,
            // otherwise the mapped borrow lives to long.
            // one borrow is needed to check if it's binary,
            // another borrow is needed to return to the user.
            Ref::map(self.inner.borrow(), |i: &ModuleInner| -> &[u8] { &i.binary().unwrap() })
        })
    }
    pub fn imports(&self) -> Ref<[ImportType]> {
        Ref::map(self.inner.borrow(), |i| -> &[ImportType] { &i.imports() })
    }
    pub fn exports(&self) -> Ref<[ExportType]> {
        Ref::map(self.inner.borrow(), |i| -> &[ExportType] { &i.exports() })
    }
}

#[derive(Clone)]
pub(crate) enum ModuleCodeSource {
    Binary(Box<[u8]>),
    Unknown,
}

#[derive(Clone)]
struct ModuleInner {
    store: HostRef<Store>,
    source: ModuleCodeSource,
    pub(crate) imports: Box<[ImportType]>,
    pub(crate) exports: Box<[ExportType]>,
}

impl ModuleInner {
    /// See documentation on ModuleInner.
    fn new(store: &HostRef<Store>, binary: &[u8]) -> Result<Self> {
        Module::validate(store, binary)?;
        Self::new_unchecked(store, binary)
    }
    /// See documentation on ModuleInner.
    fn new_unchecked(store: &HostRef<Store>, binary: &[u8]) -> Result<Self> {
        let (imports, exports) = read_imports_and_exports(binary)?;
        Ok(ModuleInner {
            store: store.clone(),
            source: ModuleCodeSource::Binary(binary.into()),
            imports,
            exports,
        })
    }
    fn from_exports(store: &HostRef<Store>, exports: Box<[ExportType]>) -> Self {
        ModuleInner {
            store: store.clone(),
            source: ModuleCodeSource::Unknown,
            imports: Box::new([]),
            exports,
        }
    }
    pub(crate) fn binary(&self) -> Option<&[u8]> {
        match &self.source {
            ModuleCodeSource::Binary(b) => Some(b),
            _ => None,
        }
    }
    pub fn imports(&self) -> &[ImportType] {
        &self.imports
    }
    pub fn exports(&self) -> &[ExportType] {
        &self.exports
    }
}
