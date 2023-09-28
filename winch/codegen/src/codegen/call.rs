//! Function call emission.  For more details around the ABI and
//! calling convention, see [ABI].
use crate::{
    abi::{ABIArg, ABISig, ABI},
    codegen::{BuiltinFunction, CodeGenContext},
    masm::{CalleeKind, MacroAssembler, OperandSize},
    reg::Reg,
};
use wasmtime_environ::FuncIndex;

/// All the information needed to emit a function call.
pub(crate) struct FnCall<'a> {
    /// The stack space consumed by the function call; that is,
    /// the sum of:
    ///
    /// 1. The amount of stack space created by saving any live
    ///    registers at the callsite.
    /// 2. The amount of space used by any memory entries in the value
    ///    stack present at the callsite, that will be used as
    ///    arguments for the function call. Any memory values in the
    ///    value stack that are needed as part of the function
    ///    arguments, will be consumed by the function call (either by
    ///    assigning those values to a register or by storing those
    ///    values to a memory location if the callee argument is on
    ///    the stack), so we track that stack space to reclaim it once
    ///    the function call has ended. This could also be done in
    ///    `assign_args` everytime a memory entry needs to be assigned
    ///    to a particular location, but doing so, will incur in more
    ///    instructions (e.g. a pop per argument that needs to be
    ///    assigned); it's more efficient to track the space needed by
    ///    those memory values and reclaim it at once.
    ///
    /// The machine stack throghout the function call is as follows:
    /// ┌──────────────────────────────────────────────────┐
    /// │                                                  │
    /// │                  1                               │
    /// │  Stack space created by any previous spills      │
    /// │  from the value stack; and which memory values   │
    /// │  are used as function arguments.                 │
    /// │                                                  │
    /// ├──────────────────────────────────────────────────┤ ---> The Wasm value stack at this point in time would look like:
    /// │                                                  │      [ Reg | Reg | Mem(offset) | Mem(offset) ]
    /// │                   2                              │
    /// │   Stack space created by saving                  │
    /// │   any live registers at the callsite.            │
    /// │                                                  │
    /// │                                                  │
    /// ├─────────────────────────────────────────────────┬┤ ---> The Wasm value stack at this point in time would look like:
    /// │                                                  │      [ Mem(offset) | Mem(offset) | Mem(offset) | Mem(offset) ]
    /// │                                                  │      Assuming that the callee takes 4 arguments, we calculate
    /// │                                                  │      2 spilled registers + 2 memory values; all of which will be used
    /// │   Stack space allocated for                      │      as arguments to the call via `assign_args`, thus the memory they represent is
    /// │   the callee function arguments in the stack;    │      is considered to be consumed by the call.
    /// │   represented by `arg_stack_space`               │
    /// │                                                  │
    /// │                                                  │
    /// │                                                  │
    /// └──────────────────────────────────────────────────┘ ------> Stack pointer when emitting the call
    ///
    call_stack_space: Option<u32>,
    /// The total stack space needed for the callee arguments on the
    /// stack, including any adjustments to the function's frame and
    /// aligned to to the required ABI alignment.
    arg_stack_space: u32,
    /// The ABI-specific signature of the callee.
    pub abi_sig: &'a ABISig,
}

impl<'a> FnCall<'a> {
    /// Allocate and setup a new function call.
    ///
    /// The setup process, will first save all the live registers in
    /// the value stack, tracking down those spilled for the function
    /// arguments(see comment below for more details) it will also
    /// track all the memory entries consumed by the function
    /// call. Then, it will calculate any adjustments needed to ensure
    /// the alignment of the caller's frame.  It's important to note
    /// that the order of operations in the setup is important, as we
    /// want to calculate any adjustments to the caller's frame, after
    /// having saved any live registers, so that we can account for
    /// any pushes generated by register spilling.
    pub fn new<M: MacroAssembler>(
        callee_sig: &'a ABISig,
        context: &mut CodeGenContext,
        masm: &mut M,
    ) -> Self {
        let arg_stack_space = callee_sig.stack_bytes;
        let mut call = Self {
            abi_sig: &callee_sig,
            arg_stack_space,
            call_stack_space: None,
        };
        // When all the information is known upfront, we can optimize and
        // calculate the stack space needed by the call right away.
        call.save_live_registers(context, masm);
        call
    }

    /// Creates a new [`FnCall`] from an [`ABISIg`] without checking if the stack has the correct
    /// amount of values for the call. This happens in situations in which not all the information
    /// is known upfront in order to fulfill the call, like for example with dealing with libcalls.
    /// Libcalls don't necessarily match 1-to-1 to WebAssembly instructions, so it's possible that
    /// before emittiing the libcall we need to adjust the value stack with the right values to be
    /// used as parameters. We can't preemptively adjust the stack since in some cases we might
    /// need to ensure that the stack is balanced right until we emit the function call because
    /// there's a dependency on certain values on the stack. A good example of this is when lazily
    /// initializing funcrefs: in order to correctly get the value of a function reference we need
    /// to determine if a libcall is needed, in order to do so we preemptively prepare the compiler
    /// to emit one since we can't know ahead-of-time of time if one will be required or not. That
    /// is the main reason why this function is unchecked, it's the caller's responsibility to
    /// ensure -- depending on the libcall -- that the value stack is correctly balanaced before
    /// the call.
    pub fn new_unchecked(sig: &'a ABISig) -> Self {
        Self {
            abi_sig: sig,
            call_stack_space: None,
            arg_stack_space: sig.stack_bytes,
        }
    }

    fn save_live_registers<M: MacroAssembler>(
        &mut self,
        context: &mut CodeGenContext,
        masm: &mut M,
    ) {
        let callee_params = &self.abi_sig.params;
        let stack = &context.stack;
        let call_stack_space = match callee_params.len() {
            0 => {
                let _ = context.save_live_registers_and_calculate_sizeof(masm, ..);
                0u32
            }
            _ => {
                // Here we perform a "spill" of the register entries
                // in the Wasm value stack, we also count any memory
                // values that will be used used as part of the callee
                // arguments.  Saving the live registers is done by
                // emitting push operations for every `Reg` entry in
                // the Wasm value stack. We do this to be compliant
                // with Winch's internal ABI, in which all registers
                // are treated as caller-saved. For more details, see
                // [ABI].
                //
                // The next few lines, partition the value stack into
                // two sections:
                // +------------------+--+--- (Stack top)
                // |                  |  |
                // |                  |  | 1. The top `n` elements, which are used for
                // |                  |  |    function arguments; for which we save any
                // |                  |  |    live registers, keeping track of the amount of registers
                // +------------------+  |    saved plus the amount of memory values consumed by the function call;
                // |                  |  |    with this information we can later reclaim the space used by the function call.
                // |                  |  |
                // +------------------+--+---
                // |                  |  | 2. The rest of the items in the stack, for which
                // |                  |  |    we only save any live registers.
                // |                  |  |
                // +------------------+  |
                assert!(stack.len() >= callee_params.len());
                let partition = stack.len() - callee_params.len();
                let _ = context.save_live_registers_and_calculate_sizeof(masm, 0..partition);
                context.save_live_registers_and_calculate_sizeof(masm, partition..)
            }
        };

        self.call_stack_space = Some(call_stack_space);
    }

    /// Used to calculate the stack space just before emitting the function
    /// call, which is when all the information to emit the call is expected to
    /// be available.
    fn calculate_call_stack_space(&mut self, context: &mut CodeGenContext) {
        let params_len = self.abi_sig.params.len();
        assert!(context.stack.len() >= params_len);

        let stack_len = context.stack.len();
        let call_stack_space = if params_len == 0 {
            0
        } else {
            context.stack.sizeof((stack_len - params_len)..)
        };
        self.call_stack_space = Some(call_stack_space);
    }

    /// Emit a direct function call, to a locally defined function.
    pub fn direct<M: MacroAssembler>(
        &mut self,
        masm: &mut M,
        context: &mut CodeGenContext,
        callee: FuncIndex,
    ) {
        if self.call_stack_space.is_none() {
            self.calculate_call_stack_space(context);
        }
        let reserved_stack = masm.call(self.arg_stack_space, |masm| {
            self.assign_args(context, masm, <M::ABI as ABI>::scratch_reg());
            CalleeKind::direct(callee.as_u32())
        });
        self.post_call::<M>(masm, context, reserved_stack);
    }

    /// Emit an indirect function call, using a register.
    pub fn reg<M: MacroAssembler>(&mut self, masm: &mut M, context: &mut CodeGenContext, reg: Reg) {
        if self.call_stack_space.is_none() {
            self.calculate_call_stack_space(context);
        }

        let reserved_stack = masm.call(self.arg_stack_space, |masm| {
            let scratch = <M::ABI as ABI>::scratch_reg();
            self.assign_args(context, masm, scratch);
            CalleeKind::indirect(reg)
        });
        context.free_reg(reg);
        self.post_call::<M>(masm, context, reserved_stack);
    }

    /// Emit an indirect function call, using a an address.
    /// This function will load the provided address into a unallocatable
    /// scratch register.
    pub fn addr<M: MacroAssembler>(
        &mut self,
        masm: &mut M,
        context: &mut CodeGenContext,
        callee: M::Address,
    ) {
        if self.call_stack_space.is_none() {
            self.calculate_call_stack_space(context);
        }

        let reserved_stack = masm.call(self.arg_stack_space, |masm| {
            let scratch = <M::ABI as ABI>::scratch_reg();
            self.assign_args(context, masm, scratch);
            masm.load(callee, scratch, OperandSize::S64);
            CalleeKind::indirect(scratch)
        });

        self.post_call::<M>(masm, context, reserved_stack);
    }

    /// Prepares the compiler to call a built-in function (libcall).
    /// This fuction, saves all the live registers and loads the callee
    /// address into a non-argument register which is then passed to the
    /// caller through the provided callback.
    ///
    /// It is the caller's responsibility to finalize the function call
    /// by calling `FnCall::reg` once all the information is known.
    pub fn with_lib<M: MacroAssembler, F>(
        &mut self,
        masm: &mut M,
        context: &mut CodeGenContext,
        func: &BuiltinFunction,
        mut f: F,
    ) where
        F: FnMut(&mut CodeGenContext, &mut M, &mut Self, Reg),
    {
        // When dealing with libcalls, we don't have all the information
        // upfront (all necessary arguments in the stack) in order to optimize
        // saving the live registers, so we save all the values available in
        // the value stack.
        context.spill(masm);
        let vmctx = <M::ABI as ABI>::vmctx_reg();
        let scratch = <M::ABI as ABI>::scratch_reg();

        let builtins_base = masm.address_at_reg(vmctx, func.base);
        masm.load(builtins_base, scratch, OperandSize::S64);
        let builtin_func_addr = masm.address_at_reg(scratch, func.offset);
        context.without::<(), M, _>(
            // Do not free the result registers if any as the function call will
            // push them onto the stack as a result of the call.
            self.abi_sig.regs(),
            self.abi_sig.param_regs(),
            masm,
            |cx, masm| {
                let callee = cx.any_gpr(masm);
                masm.load_ptr(builtin_func_addr, callee);
                f(cx, masm, self, callee);
                cx.free_reg(callee);
            },
        );
    }

    fn post_call<M: MacroAssembler>(&self, masm: &mut M, context: &mut CodeGenContext, size: u32) {
        masm.free_stack(self.call_stack_space.unwrap() + size);
        // Only account for registers given that any memory entries
        // consumed by the call (assigned to a register or to a stack
        // slot) were freed by the previous call to
        // `masm.free_stack`, so we only care about dropping them
        // here.
        //
        // NOTE / TODO there's probably a path to getting rid of
        // `save_live_registers_and_calculate_sizeof` and
        // `call_stack_space`, making it a bit more obvious what's
        // happening here. We could:
        //
        // * Modify the `spill` implementation so that it takes a
        // filtering callback, to control which values the caller is
        // interested in saving (e.g. save all if no function is provided)
        // * Rely on the new implementation of `drop_last` to calcuate
        // the stack memory entries consumed by the call and then free
        // the calculated stack space.
        context.drop_last(self.abi_sig.params.len(), |regalloc, v| {
            if v.is_reg() {
                regalloc.free(v.get_reg().into());
            }
        });
        context.push_abi_results(&self.abi_sig.result, masm);
    }

    fn assign_args<M: MacroAssembler>(
        &self,
        context: &mut CodeGenContext,
        masm: &mut M,
        scratch: Reg,
    ) {
        let arg_count = self.abi_sig.params.len();
        let stack = &context.stack;
        let mut stack_values = stack.peekn(arg_count);
        for arg in &self.abi_sig.params {
            let val = stack_values
                .next()
                .unwrap_or_else(|| panic!("expected stack value for function argument"));
            match &arg {
                &ABIArg::Reg { ty: _, reg } => {
                    context.move_val_to_reg(&val, *reg, masm);
                }
                &ABIArg::Stack { ty, offset } => {
                    let addr = masm.address_at_sp(*offset);
                    let size: OperandSize = (*ty).into();
                    context.move_val_to_reg(val, scratch, masm);
                    masm.store(scratch.into(), addr, size);
                }
            }
        }
    }
}
