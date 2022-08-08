//! Compilation support for the component model.

use crate::compiler::{Compiler, CompilerContext};
use crate::obj::ModuleTextBuilder;
use crate::CompiledFunction;
use anyhow::Result;
use cranelift_codegen::ir::{self, InstBuilder, MemFlags};
use cranelift_frontend::FunctionBuilder;
use object::write::Object;
use std::any::Any;
use std::ops::Range;
use wasmtime_environ::component::{
    AlwaysTrapInfo, CanonicalOptions, Component, ComponentCompiler, ComponentTypes, FunctionInfo,
    LowerImport, LoweredIndex, RuntimeAlwaysTrapIndex, VMComponentOffsets,
};
use wasmtime_environ::{PrimaryMap, PtrSize, SignatureIndex, Trampoline, TrapCode, WasmFuncType};

impl ComponentCompiler for Compiler {
    fn compile_lowered_trampoline(
        &self,
        component: &Component,
        lowering: &LowerImport,
        types: &ComponentTypes,
    ) -> Result<Box<dyn Any + Send>> {
        let ty = &types[lowering.canonical_abi];
        let isa = &*self.isa;
        let pointer_type = isa.pointer_type();
        let offsets = VMComponentOffsets::new(isa.pointer_bytes(), component);

        let CompilerContext {
            mut func_translator,
            codegen_context: mut context,
        } = self.take_context();

        context.func = ir::Function::with_name_signature(
            ir::ExternalName::user(0, 0),
            crate::indirect_signature(isa, ty),
        );

        let mut builder = FunctionBuilder::new(&mut context.func, func_translator.context());
        let block0 = builder.create_block();

        // Start off by spilling all the wasm arguments into a stack slot to be
        // passed to the host function.
        let (values_vec_ptr_val, values_vec_len) =
            self.wasm_to_host_spill_args(ty, &mut builder, block0);
        let vmctx = builder.func.dfg.block_params(block0)[0];

        // Save the exit FP and return address for stack walking purposes.
        //
        // First we need to get the `VMRuntimeLimits`.
        let limits = builder.ins().load(
            pointer_type,
            MemFlags::trusted(),
            vmctx,
            i32::try_from(offsets.limits()).unwrap(),
        );
        // Then save the exit Wasm FP to the limits. We dereference the current
        // FP to get the previous FP because the current FP is the trampoline's
        // FP, and we want the Wasm function's FP, which is the caller of this
        // trampoline.
        let trampoline_fp = builder.ins().get_frame_pointer(pointer_type);
        let wasm_fp = builder.ins().load(
            pointer_type,
            MemFlags::trusted(),
            trampoline_fp,
            // The FP always points to the next older FP for all supported
            // targets. See assertion in
            // `crates/runtime/src/traphandlers/backtrace.rs`.
            0,
        );
        builder.ins().store(
            MemFlags::trusted(),
            wasm_fp,
            limits,
            offsets.ptr.vmruntime_limits_last_wasm_exit_fp(),
        );
        // Finally save the Wasm return address to the limits.
        let wasm_pc = builder.ins().get_return_address(pointer_type);
        builder.ins().store(
            MemFlags::trusted(),
            wasm_pc,
            limits,
            offsets.ptr.vmruntime_limits_last_wasm_exit_pc(),
        );

        // Below this will incrementally build both the signature of the host
        // function we're calling as well as the list of arguments since the
        // list is somewhat long.
        let mut callee_args = Vec::new();
        let mut host_sig = ir::Signature::new(crate::wasmtime_call_conv(isa));

        let CanonicalOptions {
            instance,
            memory,
            realloc,
            post_return,
            string_encoding,
        } = lowering.options;

        // vmctx: *mut VMComponentContext
        host_sig.params.push(ir::AbiParam::new(pointer_type));
        callee_args.push(vmctx);

        // data: *mut u8,
        host_sig.params.push(ir::AbiParam::new(pointer_type));
        callee_args.push(builder.ins().load(
            pointer_type,
            MemFlags::trusted(),
            vmctx,
            i32::try_from(offsets.lowering_data(lowering.index)).unwrap(),
        ));

        // flags: *mut VMGlobalDefinition
        host_sig.params.push(ir::AbiParam::new(pointer_type));
        callee_args.push(
            builder
                .ins()
                .iadd_imm(vmctx, i64::from(offsets.instance_flags(instance))),
        );

        // memory: *mut VMMemoryDefinition
        host_sig.params.push(ir::AbiParam::new(pointer_type));
        callee_args.push(match memory {
            Some(idx) => builder.ins().load(
                pointer_type,
                MemFlags::trusted(),
                vmctx,
                i32::try_from(offsets.runtime_memory(idx)).unwrap(),
            ),
            None => builder.ins().iconst(pointer_type, 0),
        });

        // realloc: *mut VMCallerCheckedAnyfunc
        host_sig.params.push(ir::AbiParam::new(pointer_type));
        callee_args.push(match realloc {
            Some(idx) => builder.ins().load(
                pointer_type,
                MemFlags::trusted(),
                vmctx,
                i32::try_from(offsets.runtime_realloc(idx)).unwrap(),
            ),
            None => builder.ins().iconst(pointer_type, 0),
        });

        // A post-return option is only valid on `canon.lift`'d functions so no
        // valid component should have this specified for a lowering which this
        // trampoline compiler is interested in.
        assert!(post_return.is_none());

        // string_encoding: StringEncoding
        host_sig.params.push(ir::AbiParam::new(ir::types::I8));
        callee_args.push(
            builder
                .ins()
                .iconst(ir::types::I8, i64::from(string_encoding as u8)),
        );

        // storage: *mut ValRaw
        host_sig.params.push(ir::AbiParam::new(pointer_type));
        callee_args.push(values_vec_ptr_val);

        // storage_len: usize
        host_sig.params.push(ir::AbiParam::new(pointer_type));
        callee_args.push(
            builder
                .ins()
                .iconst(pointer_type, i64::from(values_vec_len)),
        );

        // Load host function pointer from the vmcontext and then call that
        // indirect function pointer with the list of arguments.
        let host_fn = builder.ins().load(
            pointer_type,
            MemFlags::trusted(),
            vmctx,
            i32::try_from(offsets.lowering_callee(lowering.index)).unwrap(),
        );
        let host_sig = builder.import_signature(host_sig);
        builder.ins().call_indirect(host_sig, host_fn, &callee_args);

        // After the host function has returned the results are loaded from
        // `values_vec_ptr_val` and then returned.
        self.wasm_to_host_load_results(ty, &mut builder, values_vec_ptr_val);

        let func: CompiledFunction = self.finish_trampoline(&mut context, isa)?;
        self.save_context(CompilerContext {
            func_translator,
            codegen_context: context,
        });
        Ok(Box::new(func))
    }

    fn compile_always_trap(&self, ty: &WasmFuncType) -> Result<Box<dyn Any + Send>> {
        let isa = &*self.isa;
        let CompilerContext {
            mut func_translator,
            codegen_context: mut context,
        } = self.take_context();
        context.func = ir::Function::with_name_signature(
            ir::ExternalName::user(0, 0),
            crate::indirect_signature(isa, ty),
        );
        let mut builder = FunctionBuilder::new(&mut context.func, func_translator.context());
        let block0 = builder.create_block();
        builder.append_block_params_for_function_params(block0);
        builder.switch_to_block(block0);
        builder.seal_block(block0);
        builder
            .ins()
            .trap(ir::TrapCode::User(super::ALWAYS_TRAP_CODE));
        builder.finalize();

        let func: CompiledFunction = self.finish_trampoline(&mut context, isa)?;
        self.save_context(CompilerContext {
            func_translator,
            codegen_context: context,
        });
        Ok(Box::new(func))
    }

    fn emit_obj(
        &self,
        lowerings: PrimaryMap<LoweredIndex, Box<dyn Any + Send>>,
        always_trap: PrimaryMap<RuntimeAlwaysTrapIndex, Box<dyn Any + Send>>,
        trampolines: Vec<(SignatureIndex, Box<dyn Any + Send>)>,
        obj: &mut Object<'static>,
    ) -> Result<(
        PrimaryMap<LoweredIndex, FunctionInfo>,
        PrimaryMap<RuntimeAlwaysTrapIndex, AlwaysTrapInfo>,
        Vec<Trampoline>,
    )> {
        let module = Default::default();
        let mut text = ModuleTextBuilder::new(obj, &module, &*self.isa);

        let range2info = |range: Range<u64>| FunctionInfo {
            start: u32::try_from(range.start).unwrap(),
            length: u32::try_from(range.end - range.start).unwrap(),
        };
        let ret_lowerings = lowerings
            .iter()
            .map(|(i, lowering)| {
                let lowering = lowering.downcast_ref::<CompiledFunction>().unwrap();
                assert!(lowering.traps.is_empty());
                let range = text.named_func(
                    &format!("_wasm_component_lowering_trampoline{}", i.as_u32()),
                    &lowering,
                );
                range2info(range)
            })
            .collect();
        let ret_always_trap = always_trap
            .iter()
            .map(|(i, func)| {
                let func = func.downcast_ref::<CompiledFunction>().unwrap();
                assert_eq!(func.traps.len(), 1);
                assert_eq!(func.traps[0].trap_code, TrapCode::AlwaysTrapAdapter);
                let name = format!("_wasmtime_always_trap{}", i.as_u32());
                let range = text.named_func(&name, func);
                AlwaysTrapInfo {
                    info: range2info(range),
                    trap_offset: func.traps[0].code_offset,
                }
            })
            .collect();

        let ret_trampolines = trampolines
            .iter()
            .map(|(i, func)| {
                let func = func.downcast_ref::<CompiledFunction>().unwrap();
                assert!(func.traps.is_empty());
                text.trampoline(*i, func)
            })
            .collect();

        text.finish()?;

        Ok((ret_lowerings, ret_always_trap, ret_trampolines))
    }
}
