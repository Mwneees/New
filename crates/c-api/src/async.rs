use std::ffi::c_void;
use std::future::Future;
use std::mem::{self, MaybeUninit};
use std::pin::Pin;
use std::str;
use std::task::{Context, Poll};

use wasmtime::{AsContextMut, Caller, Func, Instance, Result, Trap, Val};

use crate::{
    bad_utf8, handle_result, to_str, translate_args, wasm_functype_t, wasm_trap_t,
    wasmtime_caller_t, wasmtime_error_t, wasmtime_linker_t, wasmtime_module_t, wasmtime_val_t,
    wasmtime_val_union, CStoreContextMut, WASMTIME_I32,
};

pub type wasmtime_func_async_callback_t = extern "C" fn(
    *mut c_void,
    *mut wasmtime_caller_t,
    *const wasmtime_val_t,
    usize,
    *mut wasmtime_val_t,
    usize,
) -> Box<wasmtime_async_continuation_t>;

#[repr(C)]
pub struct wasmtime_async_continuation_t {
    pub callback: wasmtime_func_async_continuation_callback_t,
    pub env: *mut c_void,
    pub finalizer: Option<extern "C" fn(*mut c_void)>,
}

pub type wasmtime_func_async_continuation_callback_t = extern "C" fn(
    *mut c_void,
    *mut wasmtime_caller_t,
    trap_ret: *mut Option<Box<wasm_trap_t>>,
) -> bool;

struct CHostCallFuture<'a> {
    pub callback: wasmtime_func_async_continuation_callback_t,
    pub env: crate::ForeignData,
    pub caller: wasmtime_caller_t<'a>,
    pub param_count: usize,
    pub results: &'a mut [Val],
    pub hostcall_val_storage: Vec<wasmtime_val_t>,
}

unsafe impl Send for CHostCallFuture<'_> {}

impl Future for CHostCallFuture<'_> {
    type Output = Result<()>;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut trap = None;
        let done = {
            let cb = this.callback;
            cb(this.env.data, &mut this.caller, &mut trap)
        };
        if let Some(trap) = trap {
            Poll::Ready(Err(trap.error))
        } else if !done {
            Poll::Pending
        } else {
            // The contract is that once `Ready` is returned `poll` should never be called again.
            unsafe {
                // Translate the `wasmtime_val_t` results into the `results` space
                let (_, out_results) = this.hostcall_val_storage.split_at_mut(this.param_count);
                for (i, result) in out_results.iter().enumerate() {
                    this.results[i] = result.to_val();
                }

                // Move our `vals` storage back into the store now that we no longer
                // need it. This'll get picked up by the next hostcall and reuse our
                // same storage.
                let mut v = mem::take(&mut this.hostcall_val_storage);
                v.truncate(0);
                this.caller.caller.data_mut().hostcall_val_storage = v;

                Poll::Ready(Ok(()))
            }
        }
    }
}

unsafe fn c_async_callback_to_rust_fn(
    callback: wasmtime_func_async_callback_t,
    data: *mut c_void,
    finalizer: Option<extern "C" fn(*mut std::ffi::c_void)>,
) -> impl for<'a> Fn(
    Caller<'a, crate::StoreData>,
    &'a [Val],
    &'a mut [Val],
) -> Box<dyn Future<Output = Result<()>> + Send + 'a>
       + Send
       + Sync
       + 'static {
    let foreign = crate::ForeignData { data, finalizer };
    move |mut caller, params, results| {
        let _ = &foreign; // move entire foreign into this closure

        // Convert `params/results` to `wasmtime_val_t`. Use the previous
        // storage in `hostcall_val_storage` to help avoid allocations all the
        // time.
        let mut hostcall_val_storage = mem::take(&mut caller.data_mut().hostcall_val_storage);
        debug_assert!(hostcall_val_storage.is_empty());
        hostcall_val_storage.reserve(params.len() + results.len());
        hostcall_val_storage.extend(params.iter().cloned().map(|p| wasmtime_val_t::from_val(p)));
        hostcall_val_storage.extend((0..results.len()).map(|_| wasmtime_val_t {
            kind: WASMTIME_I32,
            of: wasmtime_val_union { i32: 0 },
        }));
        let (params, out_results) = hostcall_val_storage.split_at_mut(params.len());

        // Invoke the C function pointer.
        // The result will be a continutation which we will wrap in a Future.
        // The future will take ownership of the vals, and caller as well as the continutation.
        let mut caller = wasmtime_caller_t { caller };
        let continuation = callback(
            foreign.data,
            &mut caller,
            params.as_ptr(),
            params.len(),
            out_results.as_mut_ptr(),
            out_results.len(),
        );

        let param_count = params.len();
        return Box::new(CHostCallFuture {
            callback: continuation.callback,
            env: crate::ForeignData {
                data: continuation.env,
                finalizer: continuation.finalizer,
            },
            caller,
            param_count,
            results,
            hostcall_val_storage,
        });
    }
}

pub enum CallFutureState<'a> {
    Undefined,
    Called(Pin<Box<dyn Future<Output = Result<()>> + 'a>>),
    Finished(Result<()>),
}

impl Default for CallFutureState<'_> {
    fn default() -> Self {
        CallFutureState::Undefined
    }
}

#[repr(transparent)]
pub struct wasmtime_call_future_t<'a> {
    pub state: CallFutureState<'a>,
}

#[no_mangle]
pub extern "C" fn wasmtime_call_future_delete(_future: Box<wasmtime_call_future_t>) {}

#[no_mangle]
pub extern "C" fn wasmtime_call_future_poll(future: &mut wasmtime_call_future_t) -> bool {
    let mut fut = match mem::take(&mut future.state) {
        CallFutureState::Called(fut) => fut,
        _ => panic!("wasmtime_call_future_poll on a completed function"),
    };
    let w = futures::task::noop_waker_ref();
    match fut.as_mut().poll(&mut Context::from_waker(w)) {
        Poll::Ready(result) => {
            future.state = CallFutureState::Finished(result);
            true
        }
        Poll::Pending => {
            future.state = CallFutureState::Called(fut);
            false
        }
    }
}

#[no_mangle]
pub extern "C" fn wasmtime_call_future_get_results(
    future: &mut wasmtime_call_future_t,
    trap_ret: &mut *mut wasm_trap_t,
) -> Option<Box<wasmtime_error_t>> {
    let r = match mem::take(&mut future.state) {
        CallFutureState::Finished(r) => r,
        _ => panic!("wasmtime_call_future_get_results on a completed function"),
    };
    match r {
        Ok(()) => None,
        Err(err) => {
            if err.is::<Trap>() {
                *trap_ret = Box::into_raw(Box::new(wasm_trap_t::new(err)));
                None
            } else {
                Some(Box::new(wasmtime_error_t::from(err)))
            }
        }
    }
}

async fn do_func_call_async(
    mut store: CStoreContextMut<'_>,
    func: &Func,
    args: impl ExactSizeIterator<Item = Val>,
    results: &mut [MaybeUninit<wasmtime_val_t>],
) -> Result<()> {
    let mut store = store.as_context_mut();
    let mut params = mem::take(&mut store.data_mut().wasm_val_storage);
    let (wt_params, wt_results) = translate_args(
        &mut params,
        args,
        results.len(),
    );
    func.call_async(&mut store, wt_params, wt_results).await?;
    for (slot, val) in results.iter_mut().zip(wt_results.iter()) {
        crate::initialize(slot, wasmtime_val_t::from_val(val.clone()));
    }
    params.truncate(0);
    store.data_mut().wasm_val_storage = params;
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_func_call_async<'a>(
    store: CStoreContextMut<'a>,
    func: &'a Func,
    args: *const wasmtime_val_t,
    nargs: usize,
    results: *mut MaybeUninit<wasmtime_val_t>,
    nresults: usize,
) -> Box<wasmtime_call_future_t<'a>> {
    let args = crate::slice_from_raw_parts(args, nargs).iter().map(|i| i.to_val());
    let results = crate::slice_from_raw_parts_mut(results, nresults);
    let fut = Box::pin(do_func_call_async(
        store, func, args, results,
    ));
    Box::new(wasmtime_call_future_t {
        state: CallFutureState::Called(fut),
    })
}

#[no_mangle]
pub unsafe extern "C" fn wasmtime_linker_define_async_func(
    linker: &mut wasmtime_linker_t,
    module: *const u8,
    module_len: usize,
    name: *const u8,
    name_len: usize,
    ty: &wasm_functype_t,
    callback: crate::wasmtime_func_async_callback_t,
    data: *mut c_void,
    finalizer: Option<extern "C" fn(*mut std::ffi::c_void)>,
) -> Option<Box<wasmtime_error_t>> {
    let ty = ty.ty().ty.clone();
    let module = to_str!(module, module_len);
    let name = to_str!(name, name_len);
    let cb = c_async_callback_to_rust_fn(callback, data, finalizer);

    handle_result(
        linker.linker.func_new_async(module, name, ty, cb),
        |_linker| (),
    )
}

async fn do_linker_instantiate_async(
    linker: &wasmtime_linker_t,
    store: CStoreContextMut<'_>,
    module: &wasmtime_module_t,
    instance_ptr: &mut Instance,
) -> Result<()> {
    let instance = linker
        .linker
        .instantiate_async(store, &module.module)
        .await?;
    *instance_ptr = instance;
    Ok(())
}

#[no_mangle]
pub extern "C" fn wasmtime_linker_instantiate_async<'a>(
    linker: &'a wasmtime_linker_t,
    store: CStoreContextMut<'a>,
    module: &'a wasmtime_module_t,
    instance_ptr: &'a mut Instance,
) -> Box<crate::wasmtime_call_future_t<'a>> {
    let fut = Box::pin(do_linker_instantiate_async(
        linker,
        store,
        module,
        instance_ptr,
    ));
    Box::new(crate::wasmtime_call_future_t {
        state: crate::CallFutureState::Called(fut),
    })
}