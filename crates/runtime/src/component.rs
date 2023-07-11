//! Runtime support for the component model in Wasmtime
//!
//! Currently this runtime support includes a `VMComponentContext` which is
//! similar in purpose to `VMContext`. The context is read from
//! cranelift-generated trampolines when entering the host from a wasm module.
//! Eventually it's intended that module-to-module calls, which would be
//! cranelift-compiled adapters, will use this `VMComponentContext` as well.

use self::resources::ResourceTables;
use crate::{
    SendSyncPtr, Store, VMArrayCallFunction, VMFuncRef, VMGlobalDefinition, VMMemoryDefinition,
    VMNativeCallFunction, VMOpaqueContext, VMSharedSignatureIndex, VMWasmCallFunction, ValRaw,
};
use anyhow::Result;
use memoffset::offset_of;
use sptr::Strict;
use std::alloc::{self, Layout};
use std::any::Any;
use std::marker;
use std::mem;
use std::ops::Deref;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use wasmtime_environ::component::*;
use wasmtime_environ::HostPtr;

const INVALID_PTR: usize = 0xdead_dead_beef_beef_u64 as usize;

mod resources;
mod transcode;

/// Runtime representation of a component instance and all state necessary for
/// the instance itself.
///
/// This type never exists by-value, but rather it's always behind a pointer.
/// The size of the allocation for `ComponentInstance` includes the trailing
/// `VMComponentContext` which is variably sized based on the `offsets`
/// contained within.
#[repr(C)]
pub struct ComponentInstance {
    /// Size and offset information for the trailing `VMComponentContext`.
    offsets: VMComponentOffsets<HostPtr>,

    /// For more information about this see the documentation on
    /// `Instance::vmctx_self_reference`.
    vmctx_self_reference: SendSyncPtr<VMComponentContext>,

    /// Runtime type information about this component.
    runtime_info: Arc<dyn ComponentRuntimeInfo>,

    /// TODO
    resource_tables: ResourceTables,

    /// TODO:
    /// TODO: Any is bad
    resource_types: Arc<dyn Any + Send + Sync>,

    /// A zero-sized field which represents the end of the struct for the actual
    /// `VMComponentContext` to be allocated behind.
    vmctx: VMComponentContext,
}

/// Type signature for host-defined trampolines that are called from
/// WebAssembly.
///
/// This function signature is invoked from a cranelift-compiled trampoline that
/// adapts from the core wasm System-V ABI into the ABI provided here:
///
/// * `vmctx` - this is the first argument to the wasm import, and should always
///   end up being a `VMComponentContext`.
/// * `data` - this is the data pointer associated with the `VMLowering` for
///   which this function pointer was registered.
/// * `ty` - the type index, relative to the tables in `vmctx`, that is the
///   type of the function being called.
/// * `flags` - the component flags for may_enter/leave corresponding to the
///   component instance that the lowering happened within.
/// * `opt_memory` - this nullable pointer represents the memory configuration
///   option for the canonical ABI options.
/// * `opt_realloc` - this nullable pointer represents the realloc configuration
///   option for the canonical ABI options.
/// * `string_encoding` - this is the configured string encoding for the
///   canonical ABI this lowering corresponds to.
/// * `args_and_results` - pointer to stack-allocated space in the caller where
///   all the arguments are stored as well as where the results will be written
///   to. The size and initialized bytes of this depends on the core wasm type
///   signature that this callee corresponds to.
/// * `nargs_and_results` - the size, in units of `ValRaw`, of
///   `args_and_results`.
//
// FIXME: 9 arguments is probably too many. The `data` through `string-encoding`
// parameters should probably get packaged up into the `VMComponentContext`.
// Needs benchmarking one way or another though to figure out what the best
// balance is here.
pub type VMLoweringCallee = extern "C" fn(
    vmctx: *mut VMOpaqueContext,
    data: *mut u8,
    ty: TypeFuncIndex,
    flags: InstanceFlags,
    opt_memory: *mut VMMemoryDefinition,
    opt_realloc: *mut VMFuncRef,
    string_encoding: StringEncoding,
    args_and_results: *mut mem::MaybeUninit<ValRaw>,
    nargs_and_results: usize,
);

/// Structure describing a lowered host function stored within a
/// `VMComponentContext` per-lowering.
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VMLowering {
    /// The host function pointer that is invoked when this lowering is
    /// invoked.
    pub callee: VMLoweringCallee,
    /// The host data pointer (think void* pointer) to get passed to `callee`.
    pub data: *mut u8,
}

/// This is a marker type to represent the underlying allocation of a
/// `VMComponentContext`.
///
/// This type is similar to `VMContext` for core wasm and is allocated once per
/// component instance in Wasmtime. While the static size of this type is 0 the
/// actual runtime size is variable depending on the shape of the component that
/// this corresponds to. This structure always trails a `ComponentInstance`
/// allocation and the allocation/liftetime of this allocation is managed by
/// `ComponentInstance`.
#[repr(C)]
// Set an appropriate alignment for this structure where the most-aligned value
// internally right now `VMGlobalDefinition` which has an alignment of 16 bytes.
#[repr(align(16))]
pub struct VMComponentContext {
    /// For more information about this see the equivalent field in `VMContext`
    _marker: marker::PhantomPinned,
}

impl ComponentInstance {
    /// TODO
    pub unsafe fn from_vmctx<R>(
        vmctx: *mut VMComponentContext,
        f: impl FnOnce(&mut ComponentInstance) -> R,
    ) -> R {
        let ptr = vmctx
            .cast::<u8>()
            .sub(mem::size_of::<ComponentInstance>())
            .cast::<ComponentInstance>();
        f(&mut *ptr)
    }

    /// Returns the layout corresponding to what would be an allocation of a
    /// `ComponentInstance` for the `offsets` provided.
    ///
    /// The returned layout has space for both the `ComponentInstance` and the
    /// trailing `VMComponentContext`.
    fn alloc_layout(offsets: &VMComponentOffsets<HostPtr>) -> Layout {
        let size = mem::size_of::<Self>()
            .checked_add(usize::try_from(offsets.size_of_vmctx()).unwrap())
            .unwrap();
        let align = mem::align_of::<Self>();
        Layout::from_size_align(size, align).unwrap()
    }

    /// Initializes an uninitialized pointer to a `ComponentInstance` in
    /// addition to its trailing `VMComponentContext`.
    ///
    /// The `ptr` provided must be valid for `alloc_size` bytes and will be
    /// entirely overwritten by this function call. The `offsets` correspond to
    /// the shape of the component being instantiated and `store` is a pointer
    /// back to the Wasmtime store for host functions to have access to.
    unsafe fn new_at(
        ptr: NonNull<ComponentInstance>,
        alloc_size: usize,
        offsets: VMComponentOffsets<HostPtr>,
        runtime_info: Arc<dyn ComponentRuntimeInfo>,
        resource_types: Arc<dyn Any + Send + Sync>,
        store: *mut dyn Store,
    ) {
        assert!(alloc_size >= Self::alloc_layout(&offsets).size());

        ptr::write(
            ptr.as_ptr(),
            ComponentInstance {
                offsets,
                vmctx_self_reference: SendSyncPtr::new(
                    NonNull::new(
                        ptr.as_ptr()
                            .cast::<u8>()
                            .add(mem::size_of::<ComponentInstance>())
                            .cast(),
                    )
                    .unwrap(),
                ),
                resource_tables: ResourceTables::new(runtime_info.component().num_resource_tables),
                runtime_info,
                resource_types,
                vmctx: VMComponentContext {
                    _marker: marker::PhantomPinned,
                },
            },
        );

        (*ptr.as_ptr()).initialize_vmctx(store);
    }

    fn vmctx(&self) -> *mut VMComponentContext {
        let addr = std::ptr::addr_of!(self.vmctx);
        Strict::with_addr(self.vmctx_self_reference.as_ptr(), Strict::addr(addr))
    }

    unsafe fn vmctx_plus_offset<T>(&self, offset: u32) -> *const T {
        self.vmctx()
            .cast::<u8>()
            .add(usize::try_from(offset).unwrap())
            .cast()
    }

    unsafe fn vmctx_plus_offset_mut<T>(&mut self, offset: u32) -> *mut T {
        self.vmctx()
            .cast::<u8>()
            .add(usize::try_from(offset).unwrap())
            .cast()
    }

    /// Returns a pointer to the "may leave" flag for this instance specified
    /// for canonical lowering and lifting operations.
    pub fn instance_flags(&self, instance: RuntimeComponentInstanceIndex) -> InstanceFlags {
        unsafe {
            let ptr = self
                .vmctx_plus_offset::<VMGlobalDefinition>(self.offsets.instance_flags(instance))
                .cast_mut();
            InstanceFlags(SendSyncPtr::new(NonNull::new(ptr).unwrap()))
        }
    }

    /// Returns the store that this component was created with.
    pub fn store(&self) -> *mut dyn Store {
        unsafe {
            let ret = *self.vmctx_plus_offset::<*mut dyn Store>(self.offsets.store());
            assert!(!ret.is_null());
            ret
        }
    }

    /// Returns the runtime memory definition corresponding to the index of the
    /// memory provided.
    ///
    /// This can only be called after `idx` has been initialized at runtime
    /// during the instantiation process of a component.
    pub fn runtime_memory(&self, idx: RuntimeMemoryIndex) -> *mut VMMemoryDefinition {
        unsafe {
            let ret = *self.vmctx_plus_offset(self.offsets.runtime_memory(idx));
            debug_assert!(ret as usize != INVALID_PTR);
            ret
        }
    }

    /// Returns the realloc pointer corresponding to the index provided.
    ///
    /// This can only be called after `idx` has been initialized at runtime
    /// during the instantiation process of a component.
    pub fn runtime_realloc(&self, idx: RuntimeReallocIndex) -> NonNull<VMFuncRef> {
        unsafe {
            let ret = *self.vmctx_plus_offset::<NonNull<_>>(self.offsets.runtime_realloc(idx));
            debug_assert!(ret.as_ptr() as usize != INVALID_PTR);
            ret
        }
    }

    /// Returns the post-return pointer corresponding to the index provided.
    ///
    /// This can only be called after `idx` has been initialized at runtime
    /// during the instantiation process of a component.
    pub fn runtime_post_return(&self, idx: RuntimePostReturnIndex) -> NonNull<VMFuncRef> {
        unsafe {
            let ret = *self.vmctx_plus_offset::<NonNull<_>>(self.offsets.runtime_post_return(idx));
            debug_assert!(ret.as_ptr() as usize != INVALID_PTR);
            ret
        }
    }

    /// Returns the host information for the lowered function at the index
    /// specified.
    ///
    /// This can only be called after `idx` has been initialized at runtime
    /// during the instantiation process of a component.
    pub fn lowering(&self, idx: LoweredIndex) -> VMLowering {
        unsafe {
            let ret = *self.vmctx_plus_offset::<VMLowering>(self.offsets.lowering(idx));
            debug_assert!(ret.callee as usize != INVALID_PTR);
            debug_assert!(ret.data as usize != INVALID_PTR);
            ret
        }
    }

    /// Returns the core wasm function pointer corresponding to the lowering
    /// index specified.
    ///
    /// The returned function is suitable to pass directly to a wasm module
    /// instantiation and the function is a cranelift-compiled trampoline.
    ///
    /// This can only be called after `idx` has been initialized at runtime
    /// during the instantiation process of a component.
    pub fn lowering_func_ref(&self, idx: LoweredIndex) -> NonNull<VMFuncRef> {
        unsafe { self.func_ref(self.offsets.lowering_func_ref(idx)) }
    }

    /// Same as `lowering_func_ref` except for the functions that always trap.
    pub fn always_trap_func_ref(&self, idx: RuntimeAlwaysTrapIndex) -> NonNull<VMFuncRef> {
        unsafe { self.func_ref(self.offsets.always_trap_func_ref(idx)) }
    }

    /// Same as `lowering_func_ref` except for the transcoding functions.
    pub fn transcoder_func_ref(&self, idx: RuntimeTranscoderIndex) -> NonNull<VMFuncRef> {
        unsafe { self.func_ref(self.offsets.transcoder_func_ref(idx)) }
    }

    /// Same as `lowering_func_ref` except for the transcoding functions.
    pub fn resource_new_func_ref(&self, idx: RuntimeResourceNewIndex) -> NonNull<VMFuncRef> {
        unsafe { self.func_ref(self.offsets.resource_new_func_ref(idx)) }
    }

    /// Same as `lowering_func_ref` except for the transcoding functions.
    pub fn resource_rep_func_ref(&self, idx: RuntimeResourceRepIndex) -> NonNull<VMFuncRef> {
        unsafe { self.func_ref(self.offsets.resource_rep_func_ref(idx)) }
    }

    /// Same as `lowering_func_ref` except for the transcoding functions.
    pub fn resource_drop_func_ref(&self, idx: RuntimeResourceDropIndex) -> NonNull<VMFuncRef> {
        unsafe { self.func_ref(self.offsets.resource_drop_func_ref(idx)) }
    }

    unsafe fn func_ref(&self, offset: u32) -> NonNull<VMFuncRef> {
        let ret = self.vmctx_plus_offset::<VMFuncRef>(offset);
        debug_assert!(
            mem::transmute::<Option<NonNull<VMWasmCallFunction>>, usize>((*ret).wasm_call)
                != INVALID_PTR
        );
        debug_assert!((*ret).vmctx as usize != INVALID_PTR);
        NonNull::new(ret.cast_mut()).unwrap()
    }

    /// Stores the runtime memory pointer at the index specified.
    ///
    /// This is intended to be called during the instantiation process of a
    /// component once a memory is available, which may not be until part-way
    /// through component instantiation.
    ///
    /// Note that it should be a property of the component model that the `ptr`
    /// here is never needed prior to it being configured here in the instance.
    pub fn set_runtime_memory(&mut self, idx: RuntimeMemoryIndex, ptr: *mut VMMemoryDefinition) {
        unsafe {
            debug_assert!(!ptr.is_null());
            let storage = self.vmctx_plus_offset_mut(self.offsets.runtime_memory(idx));
            debug_assert!(*storage as usize == INVALID_PTR);
            *storage = ptr;
        }
    }

    /// Same as `set_runtime_memory` but for realloc function pointers.
    pub fn set_runtime_realloc(&mut self, idx: RuntimeReallocIndex, ptr: NonNull<VMFuncRef>) {
        unsafe {
            let storage = self.vmctx_plus_offset_mut(self.offsets.runtime_realloc(idx));
            debug_assert!(*storage as usize == INVALID_PTR);
            *storage = ptr.as_ptr();
        }
    }

    /// Same as `set_runtime_memory` but for post-return function pointers.
    pub fn set_runtime_post_return(
        &mut self,
        idx: RuntimePostReturnIndex,
        ptr: NonNull<VMFuncRef>,
    ) {
        unsafe {
            let storage = self.vmctx_plus_offset_mut(self.offsets.runtime_post_return(idx));
            debug_assert!(*storage as usize == INVALID_PTR);
            *storage = ptr.as_ptr();
        }
    }

    /// Configures a lowered host function with all the pieces necessary.
    ///
    /// * `idx` - the index that's being configured
    /// * `lowering` - the host-related closure information to get invoked when
    ///   the lowering is called.
    /// * `{wasm,native,array}_call` - the cranelift-compiled trampolines which will
    ///   read the `VMComponentContext` and invoke `lowering` provided.
    /// * `type_index` - the signature index for the core wasm type
    ///   registered within the engine already.
    pub fn set_lowering(
        &mut self,
        idx: LoweredIndex,
        lowering: VMLowering,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            debug_assert!(
                *self.vmctx_plus_offset::<usize>(self.offsets.lowering_callee(idx)) == INVALID_PTR
            );
            debug_assert!(
                *self.vmctx_plus_offset::<usize>(self.offsets.lowering_data(idx)) == INVALID_PTR
            );
            *self.vmctx_plus_offset_mut(self.offsets.lowering(idx)) = lowering;
            self.set_func_ref(
                self.offsets.lowering_func_ref(idx),
                wasm_call,
                native_call,
                array_call,
                type_index,
            );
        }
    }

    /// Same as `set_lowering` but for the "always trap" functions.
    pub fn set_always_trap(
        &mut self,
        idx: RuntimeAlwaysTrapIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.set_func_ref(
                self.offsets.always_trap_func_ref(idx),
                wasm_call,
                native_call,
                array_call,
                type_index,
            );
        }
    }

    /// Same as `set_lowering` but for the transcoder functions.
    pub fn set_transcoder(
        &mut self,
        idx: RuntimeTranscoderIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.set_func_ref(
                self.offsets.transcoder_func_ref(idx),
                wasm_call,
                native_call,
                array_call,
                type_index,
            );
        }
    }

    /// Same as `set_lowering` but for the transcoder functions.
    pub fn set_resource_new(
        &mut self,
        idx: RuntimeResourceNewIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.set_func_ref(
                self.offsets.resource_new_func_ref(idx),
                wasm_call,
                native_call,
                array_call,
                type_index,
            );
        }
    }

    /// Same as `set_lowering` but for the transcoder functions.
    pub fn set_resource_rep(
        &mut self,
        idx: RuntimeResourceRepIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.set_func_ref(
                self.offsets.resource_rep_func_ref(idx),
                wasm_call,
                native_call,
                array_call,
                type_index,
            );
        }
    }

    /// Same as `set_lowering` but for the transcoder functions.
    pub fn set_resource_drop(
        &mut self,
        idx: RuntimeResourceDropIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.set_func_ref(
                self.offsets.resource_drop_func_ref(idx),
                wasm_call,
                native_call,
                array_call,
                type_index,
            );
        }
    }

    unsafe fn set_func_ref(
        &mut self,
        offset: u32,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        debug_assert!(*self.vmctx_plus_offset::<usize>(offset) == INVALID_PTR);
        let vmctx = VMOpaqueContext::from_vmcomponent(self.vmctx());
        *self.vmctx_plus_offset_mut(offset) = VMFuncRef {
            wasm_call: Some(wasm_call),
            native_call,
            array_call,
            type_index,
            vmctx,
        };
    }

    /// TODO
    pub fn set_resource_destructor(
        &mut self,
        idx: ResourceIndex,
        dtor: Option<NonNull<VMFuncRef>>,
    ) {
        unsafe {
            let offset = self.offsets.resource_destructor(idx);
            debug_assert!(*self.vmctx_plus_offset::<usize>(offset) == INVALID_PTR);
            *self.vmctx_plus_offset_mut(offset) = dtor;
        }
    }

    /// TODO
    pub fn resource_destructor(&mut self, idx: ResourceIndex) -> Option<NonNull<VMFuncRef>> {
        unsafe {
            let offset = self.offsets.resource_destructor(idx);
            debug_assert!(*self.vmctx_plus_offset::<usize>(offset) != INVALID_PTR);
            *self.vmctx_plus_offset(offset)
        }
    }

    unsafe fn initialize_vmctx(&mut self, store: *mut dyn Store) {
        *self.vmctx_plus_offset_mut(self.offsets.magic()) = VMCOMPONENT_MAGIC;
        *self.vmctx_plus_offset_mut(self.offsets.libcalls()) =
            &transcode::VMComponentLibcalls::INIT;
        *self.vmctx_plus_offset_mut(self.offsets.store()) = store;
        *self.vmctx_plus_offset_mut(self.offsets.limits()) = (*store).vmruntime_limits();

        for i in 0..self.offsets.num_runtime_component_instances {
            let i = RuntimeComponentInstanceIndex::from_u32(i);
            let mut def = VMGlobalDefinition::new();
            *def.as_i32_mut() = FLAG_MAY_ENTER | FLAG_MAY_LEAVE;
            *self.instance_flags(i).as_raw() = def;
        }

        // In debug mode set non-null bad values to all "pointer looking" bits
        // and pices related to lowering and such. This'll help detect any
        // erroneous usage and enable debug assertions above as well to prevent
        // loading these before they're configured or setting them twice.
        if cfg!(debug_assertions) {
            for i in 0..self.offsets.num_lowerings {
                let i = LoweredIndex::from_u32(i);
                let offset = self.offsets.lowering_callee(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
                let offset = self.offsets.lowering_data(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
                let offset = self.offsets.lowering_func_ref(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_always_trap {
                let i = RuntimeAlwaysTrapIndex::from_u32(i);
                let offset = self.offsets.always_trap_func_ref(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_transcoders {
                let i = RuntimeTranscoderIndex::from_u32(i);
                let offset = self.offsets.transcoder_func_ref(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_resource_new {
                let i = RuntimeResourceNewIndex::from_u32(i);
                let offset = self.offsets.resource_new_func_ref(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_resource_rep {
                let i = RuntimeResourceRepIndex::from_u32(i);
                let offset = self.offsets.resource_rep_func_ref(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_resource_drop {
                let i = RuntimeResourceDropIndex::from_u32(i);
                let offset = self.offsets.resource_drop_func_ref(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_runtime_memories {
                let i = RuntimeMemoryIndex::from_u32(i);
                let offset = self.offsets.runtime_memory(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_runtime_reallocs {
                let i = RuntimeReallocIndex::from_u32(i);
                let offset = self.offsets.runtime_realloc(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_runtime_post_returns {
                let i = RuntimePostReturnIndex::from_u32(i);
                let offset = self.offsets.runtime_post_return(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
            for i in 0..self.offsets.num_resources {
                let i = ResourceIndex::from_u32(i);
                let offset = self.offsets.resource_destructor(i);
                *self.vmctx_plus_offset_mut(offset) = INVALID_PTR;
            }
        }
    }

    /// TODO
    pub fn component(&self) -> &Component {
        self.runtime_info.component()
    }

    /// Returns the type information that this instance is instantiated with.
    pub fn component_types(&self) -> &Arc<ComponentTypes> {
        self.runtime_info.component_types()
    }

    /// TODO
    pub fn resource_types(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.resource_types
    }

    /// TODO
    pub fn resource_new32(&mut self, resource: TypeResourceTableIndex, rep: u32) -> u32 {
        self.resource_tables.resource_new(resource, rep)
    }

    /// TODO
    pub fn resource_lower_own(&mut self, ty: TypeResourceTableIndex, rep: u32) -> u32 {
        self.resource_tables.resource_lower_own(ty, rep)
    }

    /// TODO
    pub fn resource_lift_own(
        &mut self,
        ty: TypeResourceTableIndex,
        idx: u32,
    ) -> Result<(u32, Option<NonNull<VMFuncRef>>, Option<InstanceFlags>)> {
        let rep = self.resource_tables.resource_lift_own(ty, idx)?;
        let resource = self.component_types()[ty].ty;
        let dtor = self.resource_destructor(resource);
        let component = self.component();
        let flags = component.defined_resource_index(resource).map(|i| {
            let instance = component.defined_resource_instances[i];
            self.instance_flags(instance)
        });
        Ok((rep, dtor, flags))
    }

    /// TODO
    pub fn resource_lift_borrow(&mut self, ty: TypeResourceTableIndex, idx: u32) -> Result<u32> {
        self.resource_tables.resource_lift_borrow(ty, idx)
    }

    /// TODO
    pub fn resource_lower_borrow(&mut self, ty: TypeResourceTableIndex, rep: u32) -> Result<u32> {
        // Implement `lower_borrow`'s special case here where if a borrow is
        // inserted into a table owned by the instance which implemented the
        // original resource then no borrow tracking is employed and instead the
        // `rep` is returned "raw".
        //
        // This check is performed by comparing the owning instance of `ty`
        // against the owning instance of the resource that `ty` is working
        // with.
        let resource = &self.component_types()[ty];
        let component = self.component();
        if let Some(idx) = component.defined_resource_index(resource.ty) {
            if resource.instance == component.defined_resource_instances[idx] {
                return Ok(rep);
            }
        }

        // ... failing that though borrow tracking enuses and is delegated to
        // the resource tables implementation.
        self.resource_tables.resource_lower_borrow(ty, rep)
    }

    /// TODO
    pub fn resource_rep32(&mut self, resource: TypeResourceTableIndex, idx: u32) -> Result<u32> {
        self.resource_tables.resource_rep(resource, idx)
    }

    /// TODO
    pub fn resource_drop(
        &mut self,
        resource: TypeResourceTableIndex,
        idx: u32,
    ) -> Result<Option<u32>> {
        self.resource_tables.resource_drop(resource, idx)
    }

    /// TODO
    pub fn enter_call(&mut self) {
        self.resource_tables.enter_call();
    }

    /// TODO
    pub fn exit_call(&mut self) -> Result<()> {
        self.resource_tables.exit_call()
    }
}

impl VMComponentContext {
    /// Moves the `self` pointer backwards to the `ComponentInstance` pointer
    /// that this `VMComponentContext` trails.
    pub fn instance(&self) -> *mut ComponentInstance {
        unsafe {
            (self as *const Self as *mut u8)
                .offset(-(offset_of!(ComponentInstance, vmctx) as isize))
                as *mut ComponentInstance
        }
    }
}

/// An owned version of `ComponentInstance` which is akin to
/// `Box<ComponentInstance>`.
///
/// This type can be dereferenced to `ComponentInstance` to access the
/// underlying methods.
pub struct OwnedComponentInstance {
    ptr: SendSyncPtr<ComponentInstance>,
}

impl OwnedComponentInstance {
    /// Allocates a new `ComponentInstance + VMComponentContext` pair on the
    /// heap with `malloc` and configures it for the `component` specified.
    pub fn new(
        runtime_info: Arc<dyn ComponentRuntimeInfo>,
        resource_types: Arc<dyn Any + Send + Sync>,
        store: *mut dyn Store,
    ) -> OwnedComponentInstance {
        let component = runtime_info.component();
        let offsets = VMComponentOffsets::new(HostPtr, component);
        let layout = ComponentInstance::alloc_layout(&offsets);
        unsafe {
            // Technically it is not required to `alloc_zeroed` here. The
            // primary reason for doing this is because a component context
            // start is a "partly initialized" state where pointers and such are
            // configured as the instantiation process continues. The component
            // model should guarantee that we never access uninitialized memory
            // in the context, but to help protect against possible bugs a
            // zeroed allocation is done here to try to contain
            // use-before-initialized issues.
            let ptr = alloc::alloc_zeroed(layout) as *mut ComponentInstance;
            let ptr = NonNull::new(ptr).unwrap();

            ComponentInstance::new_at(
                ptr,
                layout.size(),
                offsets,
                runtime_info,
                resource_types,
                store,
            );

            let ptr = SendSyncPtr::new(ptr);
            OwnedComponentInstance { ptr }
        }
    }

    // Note that this is technically unsafe due to the fact that it enables
    // `mem::swap`-ing two component instances which would get all the offsets
    // mixed up and cause issues. This is scoped to just this module though as a
    // convenience to forward to `&mut` methods on `ComponentInstance`.
    unsafe fn instance_mut(&mut self) -> &mut ComponentInstance {
        &mut *self.ptr.as_ptr()
    }

    /// TODO
    pub fn instance_ptr(&self) -> *mut ComponentInstance {
        self.ptr.as_ptr()
    }

    /// See `ComponentInstance::set_runtime_memory`
    pub fn set_runtime_memory(&mut self, idx: RuntimeMemoryIndex, ptr: *mut VMMemoryDefinition) {
        unsafe { self.instance_mut().set_runtime_memory(idx, ptr) }
    }

    /// See `ComponentInstance::set_runtime_realloc`
    pub fn set_runtime_realloc(&mut self, idx: RuntimeReallocIndex, ptr: NonNull<VMFuncRef>) {
        unsafe { self.instance_mut().set_runtime_realloc(idx, ptr) }
    }

    /// See `ComponentInstance::set_runtime_post_return`
    pub fn set_runtime_post_return(
        &mut self,
        idx: RuntimePostReturnIndex,
        ptr: NonNull<VMFuncRef>,
    ) {
        unsafe { self.instance_mut().set_runtime_post_return(idx, ptr) }
    }

    /// See `ComponentInstance::set_lowering`
    pub fn set_lowering(
        &mut self,
        idx: LoweredIndex,
        lowering: VMLowering,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.instance_mut().set_lowering(
                idx,
                lowering,
                wasm_call,
                native_call,
                array_call,
                type_index,
            )
        }
    }

    /// See `ComponentInstance::set_always_trap`
    pub fn set_always_trap(
        &mut self,
        idx: RuntimeAlwaysTrapIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.instance_mut()
                .set_always_trap(idx, wasm_call, native_call, array_call, type_index)
        }
    }

    /// See `ComponentInstance::set_transcoder`
    pub fn set_transcoder(
        &mut self,
        idx: RuntimeTranscoderIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.instance_mut()
                .set_transcoder(idx, wasm_call, native_call, array_call, type_index)
        }
    }

    /// See `ComponentInstance::set_resource_new`
    pub fn set_resource_new(
        &mut self,
        idx: RuntimeResourceNewIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.instance_mut().set_resource_new(
                idx,
                wasm_call,
                native_call,
                array_call,
                type_index,
            )
        }
    }

    /// See `ComponentInstance::set_resource_rep`
    pub fn set_resource_rep(
        &mut self,
        idx: RuntimeResourceRepIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.instance_mut().set_resource_rep(
                idx,
                wasm_call,
                native_call,
                array_call,
                type_index,
            )
        }
    }

    /// See `ComponentInstance::set_resource_drop`
    pub fn set_resource_drop(
        &mut self,
        idx: RuntimeResourceDropIndex,
        wasm_call: NonNull<VMWasmCallFunction>,
        native_call: NonNull<VMNativeCallFunction>,
        array_call: VMArrayCallFunction,
        type_index: VMSharedSignatureIndex,
    ) {
        unsafe {
            self.instance_mut().set_resource_drop(
                idx,
                wasm_call,
                native_call,
                array_call,
                type_index,
            )
        }
    }

    /// See `ComponentInstance::set_resource_destructor`
    pub fn set_resource_destructor(
        &mut self,
        idx: ResourceIndex,
        dtor: Option<NonNull<VMFuncRef>>,
    ) {
        unsafe { self.instance_mut().set_resource_destructor(idx, dtor) }
    }

    /// TODO
    pub fn resource_types_mut(&mut self) -> &mut Arc<dyn Any + Send + Sync> {
        unsafe { &mut (*self.ptr.as_ptr()).resource_types }
    }
}

impl Deref for OwnedComponentInstance {
    type Target = ComponentInstance;
    fn deref(&self) -> &ComponentInstance {
        unsafe { &*self.ptr.as_ptr() }
    }
}

impl Drop for OwnedComponentInstance {
    fn drop(&mut self) {
        let layout = ComponentInstance::alloc_layout(&self.offsets);
        unsafe {
            ptr::drop_in_place(self.ptr.as_ptr());
            alloc::dealloc(self.ptr.as_ptr().cast(), layout);
        }
    }
}

impl VMComponentContext {
    /// Helper function to cast between context types using a debug assertion to
    /// protect against some mistakes.
    #[inline]
    pub unsafe fn from_opaque(opaque: *mut VMOpaqueContext) -> *mut VMComponentContext {
        // See comments in `VMContext::from_opaque` for this debug assert
        debug_assert_eq!((*opaque).magic, VMCOMPONENT_MAGIC);
        opaque.cast()
    }
}

impl VMOpaqueContext {
    /// Helper function to clearly indicate the cast desired
    #[inline]
    pub fn from_vmcomponent(ptr: *mut VMComponentContext) -> *mut VMOpaqueContext {
        ptr.cast()
    }
}

#[allow(missing_docs)]
#[repr(transparent)]
pub struct InstanceFlags(SendSyncPtr<VMGlobalDefinition>);

#[allow(missing_docs)]
impl InstanceFlags {
    #[inline]
    pub unsafe fn may_leave(&self) -> bool {
        *(*self.as_raw()).as_i32() & FLAG_MAY_LEAVE != 0
    }

    #[inline]
    pub unsafe fn set_may_leave(&mut self, val: bool) {
        if val {
            *(*self.as_raw()).as_i32_mut() |= FLAG_MAY_LEAVE;
        } else {
            *(*self.as_raw()).as_i32_mut() &= !FLAG_MAY_LEAVE;
        }
    }

    #[inline]
    pub unsafe fn may_enter(&self) -> bool {
        *(*self.as_raw()).as_i32() & FLAG_MAY_ENTER != 0
    }

    #[inline]
    pub unsafe fn set_may_enter(&mut self, val: bool) {
        if val {
            *(*self.as_raw()).as_i32_mut() |= FLAG_MAY_ENTER;
        } else {
            *(*self.as_raw()).as_i32_mut() &= !FLAG_MAY_ENTER;
        }
    }

    #[inline]
    pub unsafe fn needs_post_return(&self) -> bool {
        *(*self.as_raw()).as_i32() & FLAG_NEEDS_POST_RETURN != 0
    }

    #[inline]
    pub unsafe fn set_needs_post_return(&mut self, val: bool) {
        if val {
            *(*self.as_raw()).as_i32_mut() |= FLAG_NEEDS_POST_RETURN;
        } else {
            *(*self.as_raw()).as_i32_mut() &= !FLAG_NEEDS_POST_RETURN;
        }
    }

    #[inline]
    pub fn as_raw(&self) -> *mut VMGlobalDefinition {
        self.0.as_ptr()
    }
}

/// Runtime information about a component stored locally for reflection.
pub trait ComponentRuntimeInfo: Send + Sync + 'static {
    /// Returns the type information about the compiled component.
    fn component(&self) -> &Component;
    /// Returns a handle to the tables of type information for this component.
    fn component_types(&self) -> &Arc<ComponentTypes>;
}
