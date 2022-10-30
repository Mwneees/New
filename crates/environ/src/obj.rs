//! Utilities for working with object files that operate as Wasmtime's
//! serialization and intermediate format for compiled modules.

use crate::{EntityRef, FuncIndex, SignatureIndex};

const FUNCTION_PREFIX: &str = "_wasm_function_";
const TRAMPOLINE_PREFIX: &str = "_trampoline_";

/// Returns the symbol name in an object file for the corresponding wasm
/// function index in a module.
pub fn func_symbol_name(index: FuncIndex) -> String {
    format!("{}{}", FUNCTION_PREFIX, index.index())
}

/// Returns the symbol name in an object file for the corresponding trampoline
/// for the given signature in a module.
pub fn trampoline_symbol_name(index: SignatureIndex) -> String {
    format!("{}{}", TRAMPOLINE_PREFIX, index.index())
}

/// A custom Wasmtime-specific section of our compilation image which stores
/// mapping data from offsets in the image to offset in the original wasm
/// binary.
///
/// This section has a custom binary encoding. Currently its encoding is:
///
/// * The section starts with a 32-bit little-endian integer. This integer is
///   how many entries are in the following two arrays.
/// * Next is an array with the previous count number of 32-bit little-endian
///   integers. This array is a sorted list of relative offsets within the text
///   section. This is intended to be a lookup array to perform a binary search
///   on an offset within the text section on this array.
/// * Finally there is another array, with the same count as before, also of
///   32-bit little-endian integers. These integers map 1:1 with the previous
///   array of offsets, and correspond to what the original offset was in the
///   wasm file.
///
/// Decoding this section is intentionally simple, it only requires loading a
/// 32-bit little-endian integer plus some bounds checks. Reading this section
/// is done with the `lookup_file_pos` function below. Reading involves
/// performing a binary search on the first array using the index found for the
/// native code offset to index into the second array and find the wasm code
/// offset.
///
/// At this time this section has an alignment of 1, which means all reads of it
/// are unaligned. Additionally at this time the 32-bit encodings chosen here
/// mean that >=4gb text sections are not supported.
pub const ELF_WASMTIME_ADDRMAP: &str = ".wasmtime.addrmap";

/// A custom binary-encoded section of wasmtime compilation artifacts which
/// encodes the ability to map an offset in the text section to the trap code
/// that it corresponds to.
///
/// This section is used at runtime to determine what flavor fo trap happened to
/// ensure that embedders and debuggers know the reason for the wasm trap. The
/// encoding of this section is custom to Wasmtime and managed with helpers in
/// the `object` crate:
///
/// * First the section has a 32-bit little endian integer indicating how many
///   trap entries are in the section.
/// * Next is an array, of the same length as read before, of 32-bit
///   little-endian integers. These integers are offsets into the text section
///   of the compilation image.
/// * Finally is the same count number of bytes. Each of these bytes corresponds
///   to a trap code.
///
/// This section is decoded by `lookup_trap_code` below which will read the
/// section count, slice some bytes to get the various arrays, and then perform
/// a binary search on the offsets array to find the an index corresponding to
/// the pc being looked up. If found the same index in the trap array (the array
/// of bytes) is the trap code for that offset.
///
/// Note that at this time this section has an alignment of 1. Additionally due
/// to the 32-bit encodings for offsets this doesn't support images >=4gb.
pub const ELF_WASMTIME_TRAPS: &str = ".wasmtime.traps";

/// A custom section which consists of just 1 byte which is either 0 or 1 as to
/// whether BTI is enabled.
pub const ELF_WASM_BTI: &str = ".wasmtime.bti";

/// A bincode-encoded section containing `ModuleTypes` type information.
pub const ELF_WASM_TYPES: &str = ".wasmtime.types";

/// A bincode-encoded section containing engine-specific metadata used to
/// double-check that an artifact can be loaded into the current host.
pub const ELF_WASM_ENGINE: &str = ".wasmtime.engine";

/// This is the name of the section in the final ELF image which contains
/// concatenated data segments from the original wasm module.
///
/// This section is simply a list of bytes and ranges into this section are
/// stored within a `Module` for each data segment. Memory initialization and
/// passive segment management all index data directly located in this section.
///
/// Note that this implementation does not afford any method of leveraging the
/// `data.drop` instruction to actually release the data back to the OS. The
/// data section is simply always present in the ELF image. If we wanted to
/// release the data it's probably best to figure out what the best
/// implementation is for it at the time given a particular set of constraints.
pub const ELF_WASM_DATA: &'static str = ".rodata.wasm";

/// This is the name of the section in the final ELF image which contains a
/// `bincode`-encoded `CompiledModuleInfo`.
///
/// This section is optionally decoded in `CompiledModule::from_artifacts`
/// depending on whether or not a `CompiledModuleInfo` is already available. In
/// cases like `Module::new` where compilation directly leads into consumption,
/// it's available. In cases like `Module::deserialize` this section must be
/// decoded to get all the relevant information.
pub const ELF_WASMTIME_INFO: &'static str = ".wasmtime.info";

/// This is the name of the section in the final ELF image which contains a
/// concatenated list of all function names.
///
/// This section is optionally included in the final artifact depending on
/// whether the wasm module has any name data at all (or in the future if we add
/// an option to not preserve name data). This section is a concatenated list of
/// strings where `CompiledModuleInfo::func_names` stores offsets/lengths into
/// this section.
///
/// Note that the goal of this section is to avoid having to decode names at
/// module-load time if we can. Names are typically only used for debugging or
/// things like backtraces so there's no need to eagerly load all of them. By
/// storing the data in a separate section the hope is that the data, which is
/// sometimes quite large (3MB seen for spidermonkey-compiled-to-wasm), can be
/// paged in lazily from an mmap and is never paged in if we never reference it.
pub const ELF_NAME_DATA: &'static str = ".name.wasm";
