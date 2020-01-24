//! WebAssembly trap handling, which is built on top of the lower-level
//! signalhandling mechanisms.

use crate::instance::{InstanceHandle, SignalHandler};
use crate::trap_registry::get_trap_registry;
use crate::trap_registry::TrapDescription;
use crate::vmcontext::{VMContext, VMFunctionBody};
use backtrace::Backtrace;
use std::any::Any;
use std::cell::Cell;
use std::error::Error;
use std::fmt;
use std::ptr;
use wasmtime_environ::ir;

extern "C" {
    fn WasmtimeCallTrampoline(
        jmp_buf: *mut *const u8,
        vmctx: *mut u8,
        caller_vmctx: *mut u8,
        callee: *const VMFunctionBody,
        values_vec: *mut u8,
    ) -> i32;
    fn WasmtimeCall(
        jmp_buf: *mut *const u8,
        vmctx: *mut u8,
        caller_vmctx: *mut u8,
        callee: *const VMFunctionBody,
    ) -> i32;
    fn Unwind(jmp_buf: *const u8) -> !;
}

cfg_if::cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "macos"))] {
        #[no_mangle]
        pub unsafe extern "C" fn HandleTrap(
            pc: *mut u8,
            signum: libc::c_int,
            siginfo: *mut libc::siginfo_t,
            context: *mut libc::c_void,
        ) -> *const u8 {
            tls::with(|info| {
                match info {
                    Some(info) => info.handle_trap(pc, false, |handler| handler(signum, siginfo, context)),
                    None => ptr::null(),
                }
            })
        }
    } else if #[cfg(target_os = "windows")] {
        use winapi::um::winnt::PEXCEPTION_POINTERS;
        use winapi::um::minwinbase::EXCEPTION_STACK_OVERFLOW;

        #[no_mangle]
        pub unsafe extern "C" fn HandleTrap(
            pc: *mut u8,
            exception_info: PEXCEPTION_POINTERS
        ) -> *const u8 {
            tls::with(|info| {
                let reset_guard_page = (*(*exception_info).ExceptionRecord).ExceptionCode == EXCEPTION_STACK_OVERFLOW;
                match info {
                    Some(info) => info.handle_trap(pc, reset_guard_page, |handler| handler(exception_info)),
                    None => ptr::null(),
                }
            })
        }
    }
}

/// Raises a user-defined trap immediately.
///
/// This function performs as-if a wasm trap was just executed, only the trap
/// has a dynamic payload associated with it which is user-provided. This trap
/// payload is then returned from `wasmtime_call` an `wasmtime_call_trampoline`
/// below.
///
/// # Safety
///
/// Only safe to call when wasm code is on the stack, aka `wasmtime_call` or
/// `wasmtime_call_trampoline` must have been previously called.
pub unsafe fn raise_user_trap(data: Box<dyn Error + Send + Sync>) -> ! {
    let trap = Trap::User(data);
    tls::with(|info| info.unwrap().unwind_with(UnwindReason::Trap(trap)))
}

/// Carries a Rust panic across wasm code and resumes the panic on the other
/// side.
///
/// # Safety
///
/// Only safe to call when wasm code is on the stack, aka `wasmtime_call` or
/// `wasmtime_call_trampoline` must have been previously called.
pub unsafe fn resume_panic(payload: Box<dyn Any + Send>) -> ! {
    tls::with(|info| info.unwrap().unwind_with(UnwindReason::Panic(payload)))
}

#[cfg(target_os = "windows")]
fn reset_guard_page() {
    extern "C" {
        fn _resetstkoflw() -> winapi::ctypes::c_int;
    }

    // We need to restore guard page under stack to handle future stack overflows properly.
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/resetstkoflw?view=vs-2019
    if unsafe { _resetstkoflw() } == 0 {
        panic!("failed to restore stack guard page");
    }
}

#[cfg(not(target_os = "windows"))]
fn reset_guard_page() {}

/// Stores trace message with backtrace.
#[derive(Debug)]
pub enum Trap {
    /// A user-raised trap through `raise_user_trap`.
    User(Box<dyn Error + Send + Sync>),
    /// A wasm-originating trap from wasm code itself.
    Wasm {
        /// What sort of trap happened, as well as where in the original wasm module
        /// it happened.
        desc: TrapDescription,
        /// Native stack backtrace at the time the trap occurred
        backtrace: Backtrace,
    },
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Trap::User(user) => user.fmt(f),
            Trap::Wasm { desc, .. } => desc.fmt(f),
        }
    }
}

impl std::error::Error for Trap {}

/// Call the wasm function pointed to by `callee`. `values_vec` points to
/// a buffer which holds the incoming arguments, and to which the outgoing
/// return values will be written.
#[no_mangle]
pub unsafe extern "C" fn wasmtime_call_trampoline(
    vmctx: *mut VMContext,
    caller_vmctx: *mut VMContext,
    callee: *const VMFunctionBody,
    values_vec: *mut u8,
) -> Result<(), Trap> {
    CallThreadState::new(vmctx).with(|cx| {
        WasmtimeCallTrampoline(
            cx.jmp_buf.as_ptr(),
            vmctx as *mut u8,
            caller_vmctx as *mut u8,
            callee,
            values_vec,
        )
    })
}

/// Call the wasm function pointed to by `callee`, which has no arguments or
/// return values.
#[no_mangle]
pub unsafe extern "C" fn wasmtime_call(
    vmctx: *mut VMContext,
    caller_vmctx: *mut VMContext,
    callee: *const VMFunctionBody,
) -> Result<(), Trap> {
    CallThreadState::new(vmctx).with(|cx| {
        WasmtimeCall(
            cx.jmp_buf.as_ptr(),
            vmctx as *mut u8,
            caller_vmctx as *mut u8,
            callee,
        )
    })
}

/// Temporary state stored on the stack which is registered in the `tls` module
/// below for calls into wasm.
pub struct CallThreadState {
    unwind: Cell<UnwindReason>,
    jmp_buf: Cell<*const u8>,
    reset_guard_page: Cell<bool>,
    prev: Option<*const CallThreadState>,
    vmctx: *mut VMContext,
}

enum UnwindReason {
    None,
    Panic(Box<dyn Any + Send>),
    Trap(Trap),
}

impl CallThreadState {
    fn new(vmctx: *mut VMContext) -> CallThreadState {
        CallThreadState {
            unwind: Cell::new(UnwindReason::None),
            vmctx,
            jmp_buf: Cell::new(ptr::null()),
            reset_guard_page: Cell::new(false),
            prev: None,
        }
    }

    fn with(mut self, closure: impl FnOnce(&CallThreadState) -> i32) -> Result<(), Trap> {
        tls::with(|prev| {
            self.prev = prev.map(|p| p as *const _);
            let ret = tls::set(&self, || closure(&self));
            match self.unwind.replace(UnwindReason::None) {
                UnwindReason::None => {
                    debug_assert_eq!(ret, 1);
                    Ok(())
                }
                UnwindReason::Trap(trap) => {
                    debug_assert_eq!(ret, 0);
                    Err(trap)
                }
                UnwindReason::Panic(panic) => {
                    debug_assert_eq!(ret, 0);
                    std::panic::resume_unwind(panic)
                }
            }
        })
    }

    fn any_instance(&self, func: impl Fn(&InstanceHandle) -> bool) -> bool {
        unsafe {
            if func(&InstanceHandle::from_vmctx(self.vmctx)) {
                return true;
            }
            match self.prev {
                Some(prev) => (*prev).any_instance(func),
                None => false,
            }
        }
    }

    fn unwind_with(&self, reason: UnwindReason) -> ! {
        self.unwind.replace(reason);
        unsafe {
            Unwind(self.jmp_buf.get());
        }
    }

    /// Trap handler using our thread-local state.
    ///
    /// * `pc` - the program counter the trap happened at
    /// * `reset_guard_page` - whether or not to reset the guard page,
    ///   currently Windows specific
    /// * `call_handler` - a closure used to invoke the platform-specific
    ///   signal handler for each instance, if available.
    ///
    /// Attempts to handle the trap if it's a wasm trap. Returns a few
    /// different things:
    ///
    /// * null - the trap didn't look like a wasm trap and should continue as a
    ///   trap
    /// * 1 as a pointer - the trap was handled by a custom trap handler on an
    ///   instance, and the trap handler should quickly return.
    /// * a different pointer - a jmp_buf buffer to longjmp to, meaning that
    ///   the wasm trap was succesfully handled.
    fn handle_trap(
        &self,
        pc: *const u8,
        reset_guard_page: bool,
        call_handler: impl Fn(&SignalHandler) -> bool,
    ) -> *const u8 {
        // First up see if any instance registered has a custom trap handler,
        // in which case run them all. If anything handles the trap then we
        // return that the trap was handled.
        if self.any_instance(|i| {
            let handler = match i.instance().signal_handler.replace(None) {
                Some(handler) => handler,
                None => return false,
            };
            let result = call_handler(&handler);
            i.instance().signal_handler.set(Some(handler));
            return result;
        }) {
            return 1 as *const _;
        }

        // TODO: stack overflow can happen at any random time (i.e. in malloc()
        // in memory.grow) and it's really hard to determine if the cause was
        // stack overflow and if it happened in WebAssembly module.
        //
        // So, let's assume that any untrusted code called from WebAssembly
        // doesn't trap. Then, if we have called some WebAssembly code, it
        // means the trap is stack overflow.
        if self.jmp_buf.get().is_null() {
            return ptr::null();
        }

        let registry = get_trap_registry();
        let trap = Trap::Wasm {
            desc: registry
                .get_trap(pc as usize)
                .unwrap_or_else(|| TrapDescription {
                    source_loc: ir::SourceLoc::default(),
                    trap_code: ir::TrapCode::StackOverflow,
                }),
            backtrace: Backtrace::new_unresolved(),
        };

        self.reset_guard_page.set(reset_guard_page);
        self.unwind.replace(UnwindReason::Trap(trap));
        self.jmp_buf.get()
    }
}

impl Drop for CallThreadState {
    fn drop(&mut self) {
        if self.reset_guard_page.get() {
            reset_guard_page();
        }
    }
}

// A private inner module for managing the TLS state that we require across
// calls in wasm. The WebAssembly code is called from C++ and then a trap may
// happen which requires us to read some contextual state to figure out what to
// do with the trap. This `tls` module is used to persist that information from
// the caller to the trap site.
mod tls {
    use super::CallThreadState;
    use std::cell::Cell;
    use std::ptr;

    thread_local!(static PTR: Cell<*const CallThreadState> = Cell::new(ptr::null()));

    /// Configures thread local state such that for the duration of the
    /// execution of `closure` any call to `with` will yield `ptr`, unless this
    /// is recursively called again.
    pub fn set<R>(ptr: &CallThreadState, closure: impl FnOnce() -> R) -> R {
        struct Reset<'a, T: Copy>(&'a Cell<T>, T);

        impl<T: Copy> Drop for Reset<'_, T> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }

        PTR.with(|p| {
            let _r = Reset(p, p.replace(ptr));
            closure()
        })
    }

    /// Returns the last pointer configured with `set` above. Panics if `set`
    /// has not been previously called.
    pub fn with<R>(closure: impl FnOnce(Option<&CallThreadState>) -> R) -> R {
        PTR.with(|ptr| {
            let p = ptr.get();
            unsafe { closure(if p.is_null() { None } else { Some(&*p) }) }
        })
    }
}
