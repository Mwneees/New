use crate::{sig_registry::SignatureRegistry, trampoline::StoreInstanceHandle};
use crate::{Config, Extern, ExternRef, FuncType, Store, Trap, Val, ValType};
use anyhow::{bail, ensure, Context as _, Result};
use smallvec::{smallvec, SmallVec};
use std::any::Any;
use std::cmp::max;
use std::fmt;
use std::future::Future;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::ptr::{self, NonNull};
use std::task::{Context, Poll};
use wasmtime_environ::wasm::{EntityIndex, FuncIndex};
use wasmtime_runtime::{
    raise_user_trap, ExportFunction, InstanceAllocator, InstanceHandle, OnDemandInstanceAllocator,
    VMCallerCheckedAnyfunc, VMContext, VMExternRef, VMFunctionBody, VMFunctionImport,
    VMSharedSignatureIndex, VMTrampoline,
};

/// Represents a host function.
///
/// This differs from `Func` in that it is not associated with a `Store`.
/// Host functions are associated with a `Config`.
pub(crate) struct HostFunc {
    ty: FuncType,
    instance: InstanceHandle,
    trampoline: VMTrampoline,
}

impl HostFunc {
    /// Creates a new host function from a callback.
    ///
    /// This is analogous to [`Func::new`].
    pub fn new(
        config: &Config,
        ty: FuncType,
        func: impl Fn(Caller<'_>, &[Val], &mut [Val]) -> Result<(), Trap> + Send + Sync + 'static,
    ) -> Self {
        let ty_clone = ty.clone();

        // Create a trampoline that converts raw u128 values to `Val`
        let func = Box::new(move |caller_vmctx, values_vec: *mut u128| {
            // Lookup the last registered store as host functions have no associated store
            let store = wasmtime_runtime::with_last_info(|last| {
                last.and_then(Any::downcast_ref::<Store>)
                    .cloned()
                    .expect("Host function called without thread state")
            });

            Func::invoke(&store, &ty_clone, caller_vmctx, values_vec, &func)
        });

        let (instance, trampoline) = crate::trampoline::create_function(&ty, func, config, None)
            .expect("failed to create host function");

        Self {
            ty,
            instance,
            trampoline,
        }
    }

    /// Creates a new host function from wrapping a closure.
    ///
    /// This is analogous to [`Func::wrap`].
    pub fn wrap<Params, Results>(
        allocator: &dyn InstanceAllocator,
        func: impl IntoFunc<Params, Results> + Send + Sync,
    ) -> Self {
        let (ty, instance, trampoline) = func.into_func(allocator, None);

        Self {
            ty,
            instance,
            trampoline,
        }
    }

    /// Converts a `HostFunc` to a `Func`.
    ///
    /// # Safety
    ///
    /// This is unsafe as the caller must ensure that the store's config defines this `HostFunc`.
    pub unsafe fn to_func(&self, store: &Store) -> Func {
        let instance = StoreInstanceHandle {
            store: store.clone(),
            // This clone of the instance handle should be safe because it should not be deallocated
            // until all configs that reference it are dropped.
            // A config will not drop until all stores referencing the config are dropped.
            handle: self.instance.clone(),
        };

        let export = ExportFunction {
            anyfunc: std::ptr::NonNull::new_unchecked(store.get_host_anyfunc(
                &self.instance,
                &self.ty,
                self.trampoline,
            )),
        };

        Func {
            instance,
            trampoline: self.trampoline,
            export,
        }
    }
}

impl Drop for HostFunc {
    fn drop(&mut self) {
        // Host functions are always allocated with the default (on-demand) allocator
        unsafe { OnDemandInstanceAllocator::new(None).deallocate(&self.instance) }
    }
}

// A note about thread safety of host function instance handles:
// Host functions must be `Send+Sync` because `Module` must be `Send+Sync`.
// However, the underlying runtime `Instance` is not `Send` or `Sync`.
// For this to be safe, we must ensure that the runtime instance's state is not mutated for host functions.
// Additionally, we add the `Send+Sync` bounds to `define_host_func` and `wrap_host_func` so
// that the underlying closure stored in the instance's host state is safe to call from any thread.
// Therefore, these impls should be safe because the underlying instance is not mutated and
// the closures backing the host functions are required to be `Send+Sync`.
unsafe impl Send for HostFunc {}
unsafe impl Sync for HostFunc {}

/// A WebAssembly function which can be called.
///
/// This type can represent a number of callable items, such as:
///
/// * An exported function from a WebAssembly module.
/// * A user-defined function used to satisfy an import.
///
/// These types of callable items are all wrapped up in this `Func` and can be
/// used to both instantiate an [`Instance`] as well as be extracted from an
/// [`Instance`].
///
/// [`Instance`]: crate::Instance
///
/// # `Func` and `Clone`
///
/// Functions are internally reference counted so you can `clone` a `Func`. The
/// cloning process only performs a shallow clone, so two cloned `Func`
/// instances are equivalent in their functionality.
///
/// # `Func` and `async`
///
/// Functions from the perspective of WebAssembly are always synchronous. You
/// might have an `async` function in Rust, however, which you'd like to make
/// available from WebAssembly. Wasmtime supports asynchronously calling
/// WebAssembly through native stack switching. You can get some more
/// information about [asynchronous configs](Config::new_async), but from the
/// perspective of `Func` it's important to know that whether or not your
/// [`Store`] is asynchronous will dictate whether you call functions through
/// [`Func::call`] or [`Func::call_async`] (or the wrappers such as
/// [`Func::get0`] vs [`Func::get0_async`]).
///
/// Note that asynchronous function APIs here are a bit trickier than their
/// synchronous brethren. For example [`Func::new_async`] and
/// [`Func::wrapN_async`](Func::wrap1_async) take explicit state parameters to
/// allow you to close over the state in the returned future. It's recommended
/// that you pass state via these parameters instead of through the closure's
/// environment, which may give Rust lifetime errors. Additionally unlike
/// synchronous functions which can all get wrapped through [`Func::wrap`]
/// asynchronous functions need to explicitly wrap based on the number of
/// parameters that they have (e.g. no wasm parameters gives you
/// [`Func::wrap0_async`], one wasm parameter you'd use [`Func::wrap1_async`],
/// etc). Be sure to consult the documentation for [`Func::wrap`] for how the
/// wasm type signature is inferred from the Rust type signature.
///
/// # To `Func::call` or to `Func::getN`
///
/// There are four ways to call a `Func`. Half are asynchronous and half are
/// synchronous, corresponding to the type of store you're using. Within each
/// half you've got two choices:
///
/// * Dynamically typed - if you don't statically know the signature of the
///   function that you're calling you'll be using [`Func::call`] or
///   [`Func::call_async`]. These functions take a variable-length slice of
///   "boxed" arguments in their [`Val`] representation. Additionally the
///   results are returned as an owned slice of [`Val`]. These methods are not
///   optimized due to the dynamic type checks that must occur, in addition to
///   some dynamic allocations for where to put all the arguments. While this
///   allows you to call all possible wasm function signatures, if you're
///   looking for a speedier alternative you can also use...
///
/// * Statically typed - if you statically know the type signature of the wasm
///   function you're calling then you can use the [`Func::getN`](Func::get1)
///   family of functions where `N` is the number of wasm arguments the function
///   takes. Asynchronous users can use [`Func::getN_async`](Func::get1_async).
///   These functions will perform type validation up-front and then return a
///   specialized closure which can be used to invoke the wasm function. This
///   route involves no dynamic allocation and does not type-checks during
///   runtime, so it's recommended to use this where possible for maximal speed.
///
/// Unfortunately a limitation of the code generation backend right now means
/// that the statically typed `getN` methods only work with wasm functions that
/// return 0 or 1 value. If 2 or more values are returned you'll need to use the
/// dynamic `call` API. We hope to fix this in the future, though!
///
/// # Examples
///
/// One way to get a `Func` is from an [`Instance`] after you've instantiated
/// it:
///
/// ```
/// # use wasmtime::*;
/// # fn main() -> anyhow::Result<()> {
/// let engine = Engine::default();
/// let store = Store::new(&engine);
/// let module = Module::new(&engine, r#"(module (func (export "foo")))"#)?;
/// let instance = Instance::new(&store, &module, &[])?;
/// let foo = instance.get_func("foo").expect("export wasn't a function");
///
/// // Work with `foo` as a `Func` at this point, such as calling it
/// // dynamically...
/// match foo.call(&[]) {
///     Ok(result) => { /* ... */ }
///     Err(trap) => {
///         panic!("execution of `foo` resulted in a wasm trap: {}", trap);
///     }
/// }
/// foo.call(&[])?;
///
/// // ... or we can make a static assertion about its signature and call it.
/// // Our first call here can fail if the signatures don't match, and then the
/// // second call can fail if the function traps (like the `match` above).
/// let foo = foo.get0::<()>()?;
/// foo()?;
/// # Ok(())
/// # }
/// ```
///
/// You can also use the [`wrap` function](Func::wrap) to create a
/// `Func`
///
/// ```
/// # use wasmtime::*;
/// # fn main() -> anyhow::Result<()> {
/// let store = Store::default();
///
/// // Create a custom `Func` which can execute arbitrary code inside of the
/// // closure.
/// let add = Func::wrap(&store, |a: i32, b: i32| -> i32 { a + b });
///
/// // Next we can hook that up to a wasm module which uses it.
/// let module = Module::new(
///     store.engine(),
///     r#"
///         (module
///             (import "" "" (func $add (param i32 i32) (result i32)))
///             (func (export "call_add_twice") (result i32)
///                 i32.const 1
///                 i32.const 2
///                 call $add
///                 i32.const 3
///                 i32.const 4
///                 call $add
///                 i32.add))
///     "#,
/// )?;
/// let instance = Instance::new(&store, &module, &[add.into()])?;
/// let call_add_twice = instance.get_func("call_add_twice").expect("export wasn't a function");
/// let call_add_twice = call_add_twice.get0::<i32>()?;
///
/// assert_eq!(call_add_twice()?, 10);
/// # Ok(())
/// # }
/// ```
///
/// Or you could also create an entirely dynamic `Func`!
///
/// ```
/// # use wasmtime::*;
/// # fn main() -> anyhow::Result<()> {
/// let store = Store::default();
///
/// // Here we need to define the type signature of our `Double` function and
/// // then wrap it up in a `Func`
/// let double_type = wasmtime::FuncType::new(
///     [wasmtime::ValType::I32].iter().cloned(),
///     [wasmtime::ValType::I32].iter().cloned(),
/// );
/// let double = Func::new(&store, double_type, |_, params, results| {
///     let mut value = params[0].unwrap_i32();
///     value *= 2;
///     results[0] = value.into();
///     Ok(())
/// });
///
/// let module = Module::new(
///     store.engine(),
///     r#"
///         (module
///             (import "" "" (func $double (param i32) (result i32)))
///             (func $start
///                 i32.const 1
///                 call $double
///                 drop)
///             (start $start))
///     "#,
/// )?;
/// let instance = Instance::new(&store, &module, &[double.into()])?;
/// // .. work with `instance` if necessary
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Func {
    instance: StoreInstanceHandle,
    trampoline: VMTrampoline,
    export: ExportFunction,
}

macro_rules! for_each_function_signature {
    ($mac:ident) => {
        $mac!(0);
        $mac!(1 A1);
        $mac!(2 A1 A2);
        $mac!(3 A1 A2 A3);
        $mac!(4 A1 A2 A3 A4);
        $mac!(5 A1 A2 A3 A4 A5);
        $mac!(6 A1 A2 A3 A4 A5 A6);
        $mac!(7 A1 A2 A3 A4 A5 A6 A7);
        $mac!(8 A1 A2 A3 A4 A5 A6 A7 A8);
        $mac!(9 A1 A2 A3 A4 A5 A6 A7 A8 A9);
        $mac!(10 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10);
        $mac!(11 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11);
        $mac!(12 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12);
        $mac!(13 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13);
        $mac!(14 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14);
        $mac!(15 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15);
        $mac!(16 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15 A16);
    };
}

macro_rules! generate_get_methods {
    ($num:tt $($args:ident)*) => (paste::paste! {
        /// Extracts a natively-callable object from this `Func`, if the
        /// signature matches.
        ///
        /// See the [`Func`] structure for more documentation. Returns an error
        /// if the type parameters and return parameter provided don't match the
        /// actual function's type signature.
        ///
        /// # Panics
        ///
        /// Panics if this is called on a function in an asynchronous store.
        #[allow(non_snake_case)]
        pub fn [<get $num>]<$($args,)* R>(&self)
            -> Result<impl Fn($($args,)*) -> Result<R, Trap>>
        where
            $($args: WasmTy,)*
            R: WasmTy,
        {
            assert!(!self.store().is_async(), concat!("must use `get", $num, "_async` on synchronous stores"));
            self.[<_get $num>]::<$($args,)* R>()
        }

        /// Extracts a natively-callable object from this `Func`, if the
        /// signature matches.
        ///
        /// See the [`Func`] structure for more documentation. Returns an error
        /// if the type parameters and return parameter provided don't match the
        /// actual function's type signature.
        ///
        /// # Panics
        ///
        /// Panics if this is called on a function in a synchronous store.
        #[allow(non_snake_case, warnings)]
        #[cfg(feature = "async")]
        #[cfg_attr(nightlydoc, doc(cfg(feature = "async")))]
        pub fn [<get $num _async>]<$($args,)* R>(&self)
            -> Result<impl Fn($($args,)*) -> WasmCall<R>>
        where
            $($args: WasmTy + 'static,)*
            R: WasmTy + 'static,
        {
            assert!(self.store().is_async(), concat!("must use `get", $num, "` on synchronous stores"));

            // TODO: ideally we wouldn't box up the future here but could
            // instead name the future returned by `on_fiber`.  Unfortunately
            // that would require a bunch of inter-future borrows which are safe
            // but gnarly to implement by hand.
            //
            // Otherwise the implementation here is pretty similar to
            // `call_async` where we're just calling the `on_fiber` method to
            // run the blocking work. Most of the goop here is juggling
            // lifetimes and making sure everything lives long enough and
            // closures have the right signatures.
            //
            // This is... less readable than ideal.
            let func = self.[<_get $num>]::<$($args,)* R>()?;
            let store = self.store().clone();
            Ok(move |$($args),*| {
                let func = func.clone();
                let store = store.clone();
                WasmCall {
                    inner: Pin::from(Box::new(async move {
                        match store.on_fiber(|| func($($args,)*)).await {
                            Ok(result) => result,
                            Err(trap) => Err(trap),
                        }
                    }))
                }
            })
        }

        // Internal helper to share between `getN` and `getN_async`
        #[allow(non_snake_case)]
        fn [<_get $num>]<$($args,)* R>(&self)
            -> Result<impl Fn($($args,)*) -> Result<R, Trap> + Clone>
        where
            $($args: WasmTy,)*
            R: WasmTy,
        {
            // Verify all the paramers match the expected parameters, and that
            // there are no extra parameters...
            let ty = self.ty();
            let mut params = ty.params();
            let n = 0;
            $(
                let n = n + 1;
                $args::matches(&mut params)
                    .with_context(|| format!("Type mismatch in argument {}", n))?;
            )*
            ensure!(params.next().is_none(), "Type mismatch: too many arguments (expected {})", n);

            // ... then do the same for the results...
            let mut results = ty.results();
            R::matches(&mut results)
                .context("Type mismatch in return type")?;
            ensure!(results.next().is_none(), "Type mismatch: too many return values (expected 1)");

            // Pass the instance into the closure so that we keep it live for
            // the lifetime of the closure. Pass the `anyfunc` in so that we can
            // call it.
            let instance = self.instance.clone();
            let anyfunc = self.export.anyfunc;

            // ... and then once we've passed the typechecks we can hand out our
            // object since our `transmute` below should be safe!
            Ok(move |$($args: $args),*| -> Result<R, Trap> {
                unsafe {
                    let fnptr = mem::transmute::<
                        *const VMFunctionBody,
                        unsafe extern "C" fn(
                            *mut VMContext,
                            *mut VMContext,
                            $( $args::Abi, )*
                        ) -> R::Abi,
                    >(anyfunc.as_ref().func_ptr.as_ptr());

                    let mut ret = None;

                    $(
                        // Because this returned closure is not marked `unsafe`,
                        // we have to check that incoming values are compatible
                        // with our store.
                        if !$args.compatible_with_store(&instance.store) {
                            return Err(Trap::new(
                                "attempt to pass cross-`Store` value to Wasm as function argument"
                            ));
                        }

                        let $args = $args.into_abi_for_arg(&instance.store);
                    )*

                    invoke_wasm_and_catch_traps(&instance.store, || {
                        ret = Some(fnptr(
                            anyfunc.as_ref().vmctx,
                            ptr::null_mut(),
                            $( $args, )*
                        ));
                    })?;

                    Ok(R::from_abi(ret.unwrap(), &instance.store))
                }
            })
        }
    })
}

macro_rules! generate_wrap_async_func {
    ($num:tt $($args:ident)*) => (paste::paste!{
        /// Same as [`Func::wrap`], except the closure asynchronously produces
        /// its result. For more information see the [`Func`] documentation.
        ///
        /// # Panics
        ///
        /// This function will panic if called with a non-asynchronous store.
        #[allow(non_snake_case)]
        #[cfg(feature = "async")]
        #[cfg_attr(nightlydoc, doc(cfg(feature = "async")))]
        pub fn [<wrap $num _async>]<T, $($args,)* R>(
            store: &Store,
            state: T,
            func: impl for<'a> Fn(Caller<'a>, &'a T, $($args),*) -> Box<dyn Future<Output = R> + 'a> + 'static,
        ) -> Func
        where
            T: 'static,
            $($args: WasmTy,)*
            R: WasmRet,
        {
            assert!(store.is_async());
            Func::wrap(store, move |caller: Caller<'_>, $($args: $args),*| {
                let store = caller.store().clone();
                let mut future = Pin::from(func(caller, &state, $($args),*));
                match store.block_on(future.as_mut()) {
                    Ok(ret) => ret.into_result(),
                    Err(e) => Err(e),
                }
            })
        }
    })
}

impl Func {
    /// Creates a new `Func` with the given arguments, typically to create a
    /// user-defined function to pass as an import to a module.
    ///
    /// * `store` - a cache of data where information is stored, typically
    ///   shared with a [`Module`](crate::Module).
    ///
    /// * `ty` - the signature of this function, used to indicate what the
    ///   inputs and outputs are, which must be WebAssembly types.
    ///
    /// * `func` - the native code invoked whenever this `Func` will be called.
    ///   This closure is provided a [`Caller`] as its first argument to learn
    ///   information about the caller, and then it's passed a list of
    ///   parameters as a slice along with a mutable slice of where to write
    ///   results.
    ///
    /// Note that the implementation of `func` must adhere to the `ty`
    /// signature given, error or traps may occur if it does not respect the
    /// `ty` signature.
    ///
    /// Additionally note that this is quite a dynamic function since signatures
    /// are not statically known. For a more performant `Func` it's recommended
    /// to use [`Func::wrap`] if you can because with statically known
    /// signatures the engine can optimize the implementation much more.
    pub fn new(
        store: &Store,
        ty: FuncType,
        func: impl Fn(Caller<'_>, &[Val], &mut [Val]) -> Result<(), Trap> + 'static,
    ) -> Self {
        let ty_clone = ty.clone();

        // Create a trampoline that converts raw u128 values to `Val`
        let func = Box::new(move |caller_vmctx, values_vec: *mut u128| {
            // Lookup the last registered store as host functions have no associated store
            let store = wasmtime_runtime::with_last_info(|last| {
                last.and_then(Any::downcast_ref::<Store>)
                    .cloned()
                    .expect("function called without thread state")
            });

            Func::invoke(&store, &ty_clone, caller_vmctx, values_vec, &func)
        });

        let (instance, trampoline) = crate::trampoline::create_function(
            &ty,
            func,
            store.engine().config(),
            Some(&mut store.signatures().borrow_mut()),
        )
        .expect("failed to create function");

        let idx = EntityIndex::Function(FuncIndex::from_u32(0));
        let (instance, export) = match instance.lookup_by_declaration(&idx) {
            wasmtime_runtime::Export::Function(f) => {
                (unsafe { store.add_instance(instance, true) }, f)
            }
            _ => unreachable!(),
        };

        Func {
            instance,
            trampoline,
            export,
        }
    }

    /// Creates a new host-defined WebAssembly function which, when called,
    /// will run the asynchronous computation defined by `func` to completion
    /// and then return the result to WebAssembly.
    ///
    /// This function is the asynchronous analogue of [`Func::new`] and much of
    /// that documentation applies to this as well. There are a few key
    /// differences (besides being asynchronous) that are worth pointing out:
    ///
    /// * The state parameter `T` is passed to the provided function `F` on
    ///   each invocation. This is done so you can use the state in `T` in the
    ///   computation of the output future (the future can close over this
    ///   value). Unfortunately due to limitations of async-in-Rust right now
    ///   you **cannot** close over the captured variables in `F` itself in the
    ///   returned future. This means that you likely won't close over much
    ///   state in `F` and will instead use `T`.
    ///
    /// * The closure here returns a *boxed* future, not something that simply
    ///   implements a future. This is also unfortunately due to limitations in
    ///   Rust right now.
    ///
    /// Overall we're not super happy with this API signature and would love to
    /// change it to make it more ergonomic. Despite this, however, you should
    /// be able to still hook into asynchronous computations and plug them into
    /// wasm. Improvements are always welcome with PRs!
    ///
    /// # Panics
    ///
    /// This function will panic if `store` is not associated with an [async
    /// config](Config::new_async).
    ///
    /// # Examples
    ///
    /// ```
    /// # use wasmtime::*;
    /// # fn main() -> anyhow::Result<()> {
    /// // Simulate some application-specific state as well as asynchronous
    /// // functions to query that state.
    /// struct MyDatabase {
    ///     // ...
    /// }
    ///
    /// impl MyDatabase {
    ///     async fn get_row_count(&self) -> u32 {
    ///         // ...
    /// #       100
    ///     }
    /// }
    ///
    /// let my_database = MyDatabase {
    ///     // ...
    /// };
    ///
    /// // Using `new_async` we can hook up into calling our async
    /// // `get_row_count` function.
    /// let store = Store::new(&Engine::new(&Config::new_async()));
    /// let get_row_count_type = wasmtime::FuncType::new(
    ///     None,
    ///     Some(wasmtime::ValType::I32),
    /// );
    /// let double = Func::new_async(&store, get_row_count_type, my_database, |_, database, params, results| {
    ///     Box::new(async move {
    ///         let count = database.get_row_count().await;
    ///         results[0] = Val::I32(count as i32);
    ///         Ok(())
    ///     })
    /// });
    /// // ...
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    #[cfg_attr(nightlydoc, doc(cfg(feature = "async")))]
    pub fn new_async<T, F>(store: &Store, ty: FuncType, state: T, func: F) -> Func
    where
        T: 'static,
        F: for<'a> Fn(
                Caller<'a>,
                &'a T,
                &'a [Val],
                &'a mut [Val],
            ) -> Box<dyn Future<Output = Result<(), Trap>> + 'a>
            + 'static,
    {
        assert!(store.is_async());
        Func::new(store, ty, move |caller, params, results| {
            let store = caller.store().clone();
            let mut future = Pin::from(func(caller, &state, params, results));
            match store.block_on(future.as_mut()) {
                Ok(Ok(())) => Ok(()),
                Ok(Err(trap)) | Err(trap) => Err(trap),
            }
        })
    }

    pub(crate) unsafe fn from_caller_checked_anyfunc(
        store: &Store,
        anyfunc: *mut VMCallerCheckedAnyfunc,
    ) -> Option<Self> {
        let anyfunc = NonNull::new(anyfunc)?;
        debug_assert!(anyfunc.as_ref().type_index != VMSharedSignatureIndex::default());
        let export = ExportFunction { anyfunc };
        let f = Func::from_wasmtime_function(&export, store);
        Some(f)
    }

    /// Creates a new `Func` from the given Rust closure.
    ///
    /// This function will create a new `Func` which, when called, will
    /// execute the given Rust closure. Unlike [`Func::new`] the target
    /// function being called is known statically so the type signature can
    /// be inferred. Rust types will map to WebAssembly types as follows:
    ///
    /// | Rust Argument Type  | WebAssembly Type |
    /// |---------------------|------------------|
    /// | `i32`               | `i32`            |
    /// | `u32`               | `i32`            |
    /// | `i64`               | `i64`            |
    /// | `u64`               | `i64`            |
    /// | `f32`               | `f32`            |
    /// | `f64`               | `f64`            |
    /// | (not supported)     | `v128`           |
    /// | `Option<Func>`      | `funcref`        |
    /// | `Option<ExternRef>` | `externref`      |
    ///
    /// Any of the Rust types can be returned from the closure as well, in
    /// addition to some extra types
    ///
    /// | Rust Return Type  | WebAssembly Return Type | Meaning           |
    /// |-------------------|-------------------------|-------------------|
    /// | `()`              | nothing                 | no return value   |
    /// | `Result<T, Trap>` | `T`                     | function may trap |
    ///
    /// At this time multi-value returns are not supported, and supporting this
    /// is the subject of [#1178].
    ///
    /// [#1178]: https://github.com/bytecodealliance/wasmtime/issues/1178
    ///
    /// Finally you can also optionally take [`Caller`] as the first argument of
    /// your closure. If inserted then you're able to inspect the caller's
    /// state, for example the [`Memory`](crate::Memory) it has exported so you
    /// can read what pointers point to.
    ///
    /// Note that when using this API, the intention is to create as thin of a
    /// layer as possible for when WebAssembly calls the function provided. With
    /// sufficient inlining and optimization the WebAssembly will call straight
    /// into `func` provided, with no extra fluff entailed.
    ///
    /// # Examples
    ///
    /// First up we can see how simple wasm imports can be implemented, such
    /// as a function that adds its two arguments and returns the result.
    ///
    /// ```
    /// # use wasmtime::*;
    /// # fn main() -> anyhow::Result<()> {
    /// # let store = Store::default();
    /// let add = Func::wrap(&store, |a: i32, b: i32| a + b);
    /// let module = Module::new(
    ///     store.engine(),
    ///     r#"
    ///         (module
    ///             (import "" "" (func $add (param i32 i32) (result i32)))
    ///             (func (export "foo") (param i32 i32) (result i32)
    ///                 local.get 0
    ///                 local.get 1
    ///                 call $add))
    ///     "#,
    /// )?;
    /// let instance = Instance::new(&store, &module, &[add.into()])?;
    /// let foo = instance.get_func("foo").unwrap().get2::<i32, i32, i32>()?;
    /// assert_eq!(foo(1, 2)?, 3);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// We can also do the same thing, but generate a trap if the addition
    /// overflows:
    ///
    /// ```
    /// # use wasmtime::*;
    /// # fn main() -> anyhow::Result<()> {
    /// # let store = Store::default();
    /// let add = Func::wrap(&store, |a: i32, b: i32| {
    ///     match a.checked_add(b) {
    ///         Some(i) => Ok(i),
    ///         None => Err(Trap::new("overflow")),
    ///     }
    /// });
    /// let module = Module::new(
    ///     store.engine(),
    ///     r#"
    ///         (module
    ///             (import "" "" (func $add (param i32 i32) (result i32)))
    ///             (func (export "foo") (param i32 i32) (result i32)
    ///                 local.get 0
    ///                 local.get 1
    ///                 call $add))
    ///     "#,
    /// )?;
    /// let instance = Instance::new(&store, &module, &[add.into()])?;
    /// let foo = instance.get_func("foo").unwrap().get2::<i32, i32, i32>()?;
    /// assert_eq!(foo(1, 2)?, 3);
    /// assert!(foo(i32::max_value(), 1).is_err());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// And don't forget all the wasm types are supported!
    ///
    /// ```
    /// # use wasmtime::*;
    /// # fn main() -> anyhow::Result<()> {
    /// # let store = Store::default();
    /// let debug = Func::wrap(&store, |a: i32, b: u32, c: f32, d: i64, e: u64, f: f64| {
    ///
    ///     println!("a={}", a);
    ///     println!("b={}", b);
    ///     println!("c={}", c);
    ///     println!("d={}", d);
    ///     println!("e={}", e);
    ///     println!("f={}", f);
    /// });
    /// let module = Module::new(
    ///     store.engine(),
    ///     r#"
    ///         (module
    ///             (import "" "" (func $debug (param i32 i32 f32 i64 i64 f64)))
    ///             (func (export "foo")
    ///                 i32.const -1
    ///                 i32.const 1
    ///                 f32.const 2
    ///                 i64.const -3
    ///                 i64.const 3
    ///                 f64.const 4
    ///                 call $debug))
    ///     "#,
    /// )?;
    /// let instance = Instance::new(&store, &module, &[debug.into()])?;
    /// let foo = instance.get_func("foo").unwrap().get0::<()>()?;
    /// foo()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Finally if you want to get really fancy you can also implement
    /// imports that read/write wasm module's memory
    ///
    /// ```
    /// use std::str;
    ///
    /// # use wasmtime::*;
    /// # fn main() -> anyhow::Result<()> {
    /// # let store = Store::default();
    /// let log_str = Func::wrap(&store, |caller: Caller<'_>, ptr: i32, len: i32| {
    ///     let mem = match caller.get_export("memory") {
    ///         Some(Extern::Memory(mem)) => mem,
    ///         _ => return Err(Trap::new("failed to find host memory")),
    ///     };
    ///
    ///     // We're reading raw wasm memory here so we need `unsafe`. Note
    ///     // though that this should be safe because we don't reenter wasm
    ///     // while we're reading wasm memory, nor should we clash with
    ///     // any other memory accessors (assuming they're well-behaved
    ///     // too).
    ///     unsafe {
    ///         let data = mem.data_unchecked()
    ///             .get(ptr as u32 as usize..)
    ///             .and_then(|arr| arr.get(..len as u32 as usize));
    ///         let string = match data {
    ///             Some(data) => match str::from_utf8(data) {
    ///                 Ok(s) => s,
    ///                 Err(_) => return Err(Trap::new("invalid utf-8")),
    ///             },
    ///             None => return Err(Trap::new("pointer/length out of bounds")),
    ///         };
    ///         assert_eq!(string, "Hello, world!");
    ///         println!("{}", string);
    ///     }
    ///     Ok(())
    /// });
    /// let module = Module::new(
    ///     store.engine(),
    ///     r#"
    ///         (module
    ///             (import "" "" (func $log_str (param i32 i32)))
    ///             (func (export "foo")
    ///                 i32.const 4   ;; ptr
    ///                 i32.const 13  ;; len
    ///                 call $log_str)
    ///             (memory (export "memory") 1)
    ///             (data (i32.const 4) "Hello, world!"))
    ///     "#,
    /// )?;
    /// let instance = Instance::new(&store, &module, &[log_str.into()])?;
    /// let foo = instance.get_func("foo").unwrap().get0::<()>()?;
    /// foo()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn wrap<Params, Results>(store: &Store, func: impl IntoFunc<Params, Results>) -> Func {
        let (_, instance, trampoline) = func.into_func(
            &store.engine().config().default_instance_allocator,
            Some(&mut store.signatures().borrow_mut()),
        );

        let (instance, export) = unsafe {
            let idx = EntityIndex::Function(FuncIndex::from_u32(0));
            match instance.lookup_by_declaration(&idx) {
                wasmtime_runtime::Export::Function(f) => (store.add_instance(instance, true), f),
                _ => unreachable!(),
            }
        };

        Func {
            instance,
            export,
            trampoline,
        }
    }

    for_each_function_signature!(generate_wrap_async_func);

    pub(crate) fn sig_index(&self) -> VMSharedSignatureIndex {
        unsafe { self.export.anyfunc.as_ref().type_index }
    }

    /// Returns the underlying wasm type that this `Func` has.
    pub fn ty(&self) -> FuncType {
        // Signatures should always be registered in the store's registry of
        // shared signatures, so we should be able to unwrap safely here.
        let signatures = self.instance.store.signatures().borrow();
        let (wft, _) = signatures
            .lookup_shared(self.sig_index())
            .expect("signature should be registered");

        // This is only called with `Export::Function`, and since it's coming
        // from wasmtime_runtime itself we should support all the types coming
        // out of it, so assert such here.
        FuncType::from_wasm_func_type(&wft)
    }

    /// Returns the number of parameters that this function takes.
    pub fn param_arity(&self) -> usize {
        let signatures = self.instance.store.signatures().borrow();
        let (sig, _) = signatures
            .lookup_shared(self.sig_index())
            .expect("signature should be registered");
        sig.params.len()
    }

    /// Returns the number of results this function produces.
    pub fn result_arity(&self) -> usize {
        let signatures = self.instance.store.signatures().borrow();
        let (sig, _) = signatures
            .lookup_shared(self.sig_index())
            .expect("signature should be registered");
        sig.returns.len()
    }

    /// Invokes this function with the `params` given, returning the results and
    /// any trap, if one occurs.
    ///
    /// The `params` here must match the type signature of this `Func`, or a
    /// trap will occur. If a trap occurs while executing this function, then a
    /// trap will also be returned.
    ///
    /// # Panics
    ///
    /// This function will panic if called on a function belonging to an async
    /// store. Asynchronous stores must always use `call_async`.
    /// initiates a panic.
    pub fn call(&self, params: &[Val]) -> Result<Box<[Val]>> {
        assert!(
            !self.store().is_async(),
            "must use `call_async` on asynchronous stores",
        );
        self._call(params)
    }

    /// Invokes this function with the `params` given, returning the results
    /// asynchronously.
    ///
    /// This function is the same as [`Func::call`] except that it is
    /// asynchronous. This is only compatible with stores associated with an
    /// [asynchronous config](Config::new_async).
    ///
    /// It's important to note that the execution of WebAssembly will happen
    /// synchronously in the `poll` method of the future returned from this
    /// function. Wasmtime does not manage its own thread pool or similar to
    /// execute WebAssembly in. Future `poll` methods are generally expected to
    /// resolve quickly, so it's recommended that you run or poll this future
    /// in a "blocking context".
    ///
    /// For more information see the documentation on [asynchronous
    /// configs](Config::new_async).
    ///
    /// # Panics
    ///
    /// Panics if this is called on a function in a synchronous store. This
    /// only works with functions defined within an asynchronous store.
    #[cfg(feature = "async")]
    #[cfg_attr(nightlydoc, doc(cfg(feature = "async")))]
    pub async fn call_async(&self, params: &[Val]) -> Result<Box<[Val]>> {
        assert!(
            self.store().is_async(),
            "must use `call` on synchronous stores",
        );
        let result = self.store().on_fiber(|| self._call(params)).await??;
        Ok(result)
    }

    fn _call(&self, params: &[Val]) -> Result<Box<[Val]>> {
        // We need to perform a dynamic check that the arguments given to us
        // match the signature of this function and are appropriate to pass to
        // this function. This involves checking to make sure we have the right
        // number and types of arguments as well as making sure everything is
        // from the same `Store`.
        let my_ty = self.ty();
        if my_ty.params().len() != params.len() {
            bail!(
                "expected {} arguments, got {}",
                my_ty.params().len(),
                params.len()
            );
        }

        let mut values_vec = vec![0; max(params.len(), my_ty.results().len())];

        // Store the argument values into `values_vec`.
        let param_tys = my_ty.params();
        for ((arg, slot), ty) in params.iter().cloned().zip(&mut values_vec).zip(param_tys) {
            if arg.ty() != ty {
                bail!(
                    "argument type mismatch: found {} but expected {}",
                    arg.ty(),
                    ty
                );
            }
            if !arg.comes_from_same_store(&self.instance.store) {
                bail!("cross-`Store` values are not currently supported");
            }
            unsafe {
                arg.write_value_to(&self.instance.store, slot);
            }
        }

        // Call the trampoline.
        unsafe {
            let anyfunc = self.export.anyfunc.as_ref();
            invoke_wasm_and_catch_traps(&self.instance.store, || {
                (self.trampoline)(
                    anyfunc.vmctx,
                    ptr::null_mut(),
                    anyfunc.func_ptr.as_ptr(),
                    values_vec.as_mut_ptr(),
                )
            })?;
        }

        // Load the return values out of `values_vec`.
        let mut results = Vec::with_capacity(my_ty.results().len());
        for (index, ty) in my_ty.results().enumerate() {
            unsafe {
                let ptr = values_vec.as_ptr().add(index);
                results.push(Val::read_value_from(&self.instance.store, ptr, ty));
            }
        }

        Ok(results.into())
    }

    pub(crate) fn caller_checked_anyfunc(&self) -> NonNull<VMCallerCheckedAnyfunc> {
        self.export.anyfunc
    }

    pub(crate) unsafe fn from_wasmtime_function(export: &ExportFunction, store: &Store) -> Self {
        // Each function signature in a module should have a trampoline stored
        // on that module as well, so unwrap the result here since otherwise
        // it's a bug in wasmtime.
        let anyfunc = export.anyfunc.as_ref();
        let trampoline = store
            .signatures()
            .borrow()
            .lookup_shared(anyfunc.type_index)
            .expect("failed to retrieve trampoline from module")
            .1;

        Func {
            instance: store.existing_vmctx(anyfunc.vmctx),
            export: export.clone(),
            trampoline,
        }
    }

    for_each_function_signature!(generate_get_methods);

    /// Get a reference to this function's store.
    pub fn store(&self) -> &Store {
        &self.instance.store
    }

    pub(crate) fn vmimport(&self) -> VMFunctionImport {
        unsafe {
            let f = self.caller_checked_anyfunc();
            VMFunctionImport {
                body: f.as_ref().func_ptr,
                vmctx: f.as_ref().vmctx,
            }
        }
    }

    pub(crate) fn wasmtime_export(&self) -> &ExportFunction {
        &self.export
    }

    pub(crate) fn invoke(
        store: &Store,
        ty: &FuncType,
        caller_vmctx: *mut VMContext,
        values_vec: *mut u128,
        func: &dyn Fn(Caller<'_>, &[Val], &mut [Val]) -> Result<(), Trap>,
    ) -> Result<(), Trap> {
        // We have a dynamic guarantee that `values_vec` has the right
        // number of arguments and the right types of arguments. As a result
        // we should be able to safely run through them all and read them.
        const STACK_ARGS: usize = 4;
        const STACK_RETURNS: usize = 2;
        let mut args: SmallVec<[Val; STACK_ARGS]> = SmallVec::with_capacity(ty.params().len());
        for (i, ty) in ty.params().enumerate() {
            unsafe {
                let val = Val::read_value_from(store, values_vec.add(i), ty);
                args.push(val);
            }
        }

        let mut returns: SmallVec<[Val; STACK_RETURNS]> =
            smallvec![Val::null(); ty.results().len()];

        func(
            Caller {
                store,
                caller_vmctx,
            },
            &args,
            &mut returns,
        )?;

        // Unlike our arguments we need to dynamically check that the return
        // values produced are correct. There could be a bug in `func` that
        // produces the wrong number, wrong types, or wrong stores of
        // values, and we need to catch that here.
        for (i, (ret, ty)) in returns.into_iter().zip(ty.results()).enumerate() {
            if ret.ty() != ty {
                return Err(Trap::new(
                    "function attempted to return an incompatible value",
                ));
            }
            if !ret.comes_from_same_store(store) {
                return Err(Trap::new(
                    "cross-`Store` values are not currently supported",
                ));
            }
            unsafe {
                ret.write_value_to(store, values_vec.add(i));
            }
        }

        Ok(())
    }
}

impl fmt::Debug for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Func")
    }
}

pub(crate) fn invoke_wasm_and_catch_traps(
    store: &Store,
    closure: impl FnMut(),
) -> Result<(), Trap> {
    unsafe {
        let canary = 0;
        let _auto_reset_canary = store
            .externref_activations_table()
            .set_stack_canary(&canary);

        wasmtime_runtime::catch_traps(store, closure).map_err(|e| Trap::from_runtime(store, e))
    }
}

/// A trait implemented for types which can be arguments to closures passed to
/// [`Func::wrap`] and friends.
///
/// This trait should not be implemented by user types. This trait may change at
/// any time internally. The types which implement this trait, however, are
/// stable over time.
///
/// For more information see [`Func::wrap`]
pub unsafe trait WasmTy {
    // The raw ABI representation of this type inside Wasm.
    #[doc(hidden)]
    type Abi: Copy;

    // Is this value compatible with the given store?
    #[doc(hidden)]
    fn compatible_with_store(&self, store: &Store) -> bool;

    // Convert this value into its ABI representation, when passing a value into
    // Wasm as an argument.
    #[doc(hidden)]
    fn into_abi_for_arg(self, store: &Store) -> Self::Abi;

    // Convert from the raw ABI representation back into `Self`, when receiving
    // a value from Wasm.
    //
    // Safety: The abi value *must* have be valid for this type (e.g. for
    // `externref`, it must be a valid raw `VMExternRef` pointer, not some
    // random, dangling pointer).
    #[doc(hidden)]
    unsafe fn from_abi(abi: Self::Abi, store: &Store) -> Self;

    // Add this type to the given vec of expected valtypes.
    #[doc(hidden)]
    fn valtype() -> Option<ValType>;

    // Does the next valtype(s) match this type?
    #[doc(hidden)]
    fn matches(tys: impl Iterator<Item = ValType>) -> anyhow::Result<()>;

    // Load this type's raw ABI representation from an args array.
    #[doc(hidden)]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi;

    // Store this type's raw ABI representation into an args array.
    #[doc(hidden)]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128);
}

/// A trait implemented for types which can be returned from closures passed to
/// [`Func::wrap`] and friends.
///
/// This trait should not be implemented by user types. This trait may change at
/// any time internally. The types which implement this trait, however, are
/// stable over time.
///
/// For more information see [`Func::wrap`]
pub unsafe trait WasmRet {
    // Same as `WasmTy::Abi`.
    #[doc(hidden)]
    type Abi: Copy;

    // The "ok" version of this, meaning that which is returned if there is no
    // error.
    #[doc(hidden)]
    type Ok: WasmTy;

    // Same as `WasmTy::compatible_with_store`.
    #[doc(hidden)]
    fn compatible_with_store(&self, store: &Store) -> bool;

    // Similar to `WasmTy::into_abi_for_arg` but used when host code is
    // returning a value into Wasm, rather than host code passing an argument to
    // a Wasm call. Unlike `into_abi_for_arg`, implementors of this method can
    // raise traps, which means that callers must ensure that
    // `invoke_wasm_and_catch_traps` is on the stack, and therefore this method
    // is unsafe.
    #[doc(hidden)]
    unsafe fn into_abi_for_ret(self, store: &Store) -> Self::Abi;

    // Same as `WasmTy::from_abi`.
    #[doc(hidden)]
    unsafe fn from_abi(abi: Self::Abi, store: &Store) -> Self;

    // Same as `WasmTy::push`.
    #[doc(hidden)]
    fn valtype() -> Option<ValType>;

    // Same as `WasmTy::matches`.
    #[doc(hidden)]
    fn matches(tys: impl Iterator<Item = ValType>) -> anyhow::Result<()>;

    // Same as `WasmTy::load_from_args`.
    #[doc(hidden)]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi;

    // Same as `WasmTy::store_to_args`.
    #[doc(hidden)]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128);

    // Converts this result into an explicit `Result` to match on.
    #[doc(hidden)]
    fn into_result(self) -> Result<Self::Ok, Trap>;
}

unsafe impl WasmTy for () {
    type Abi = Self;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {}

    #[inline]
    unsafe fn from_abi(_abi: Self::Abi, _store: &Store) -> Self {}

    fn valtype() -> Option<ValType> {
        None
    }

    fn matches(_tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        Ok(())
    }

    #[inline]
    unsafe fn load_from_args(_ptr: &mut *const u128) -> Self::Abi {}

    #[inline]
    unsafe fn store_to_args(_abi: Self::Abi, _ptr: *mut u128) {}
}

unsafe impl WasmTy for i32 {
    type Abi = Self;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {
        self
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &Store) -> Self {
        abi
    }

    fn valtype() -> Option<ValType> {
        Some(ValType::I32)
    }

    fn matches(mut tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        let next = tys.next();
        ensure!(
            next == Some(ValType::I32),
            "Type mismatch, expected i32, got {:?}",
            next
        );
        Ok(())
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        let ret = *(*ptr).cast::<Self>();
        *ptr = (*ptr).add(1);
        return ret;
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        *ptr.cast::<Self>() = abi;
    }
}

unsafe impl WasmTy for u32 {
    type Abi = <i32 as WasmTy>::Abi;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {
        self as i32
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &Store) -> Self {
        abi as Self
    }

    fn valtype() -> Option<ValType> {
        <i32 as WasmTy>::valtype()
    }

    fn matches(tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        <i32 as WasmTy>::matches(tys)
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        <i32 as WasmTy>::load_from_args(ptr)
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        <i32 as WasmTy>::store_to_args(abi, ptr)
    }
}

unsafe impl WasmTy for i64 {
    type Abi = Self;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {
        self
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &Store) -> Self {
        abi
    }

    fn valtype() -> Option<ValType> {
        Some(ValType::I64)
    }

    fn matches(mut tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        let next = tys.next();
        ensure!(
            next == Some(ValType::I64),
            "Type mismatch, expected i64, got {:?}",
            next
        );
        Ok(())
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        let ret = *(*ptr).cast::<Self>();
        *ptr = (*ptr).add(1);
        return ret;
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        *ptr.cast::<Self>() = abi;
    }
}

unsafe impl WasmTy for u64 {
    type Abi = <i64 as WasmTy>::Abi;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {
        self as i64
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &Store) -> Self {
        abi as Self
    }

    fn valtype() -> Option<ValType> {
        <i64 as WasmTy>::valtype()
    }

    fn matches(tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        <i64 as WasmTy>::matches(tys)
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        <i64 as WasmTy>::load_from_args(ptr)
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        <i64 as WasmTy>::store_to_args(abi, ptr)
    }
}

unsafe impl WasmTy for f32 {
    type Abi = Self;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {
        self
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &Store) -> Self {
        abi
    }

    fn valtype() -> Option<ValType> {
        Some(ValType::F32)
    }

    fn matches(mut tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        let next = tys.next();
        ensure!(
            next == Some(ValType::F32),
            "Type mismatch, expected f32, got {:?}",
            next
        );
        Ok(())
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        let ret = f32::from_bits(*(*ptr).cast::<u32>());
        *ptr = (*ptr).add(1);
        return ret;
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        *ptr.cast::<u32>() = abi.to_bits();
    }
}

unsafe impl WasmTy for f64 {
    type Abi = Self;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {
        self
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &Store) -> Self {
        abi
    }

    fn valtype() -> Option<ValType> {
        Some(ValType::F64)
    }

    fn matches(mut tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        let next = tys.next();
        ensure!(
            next == Some(ValType::F64),
            "Type mismatch, expected f64, got {:?}",
            next
        );
        Ok(())
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        let ret = f64::from_bits(*(*ptr).cast::<u64>());
        *ptr = (*ptr).add(1);
        return ret;
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        *ptr.cast::<u64>() = abi.to_bits();
    }
}

unsafe impl WasmTy for Option<ExternRef> {
    type Abi = *mut u8;

    #[inline]
    fn compatible_with_store(&self, _store: &Store) -> bool {
        true
    }

    #[inline]
    fn into_abi_for_arg(self, store: &Store) -> Self::Abi {
        if let Some(x) = self {
            let abi = x.inner.as_raw();
            unsafe {
                store
                    .externref_activations_table()
                    .insert_with_gc(x.inner, store.stack_map_registry());
            }
            abi
        } else {
            ptr::null_mut()
        }
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &Store) -> Self {
        if abi.is_null() {
            None
        } else {
            Some(ExternRef {
                inner: VMExternRef::clone_from_raw(abi),
            })
        }
    }

    fn valtype() -> Option<ValType> {
        Some(ValType::ExternRef)
    }

    fn matches(mut tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        let next = tys.next();
        ensure!(
            next == Some(ValType::ExternRef),
            "Type mismatch, expected externref, got {:?}",
            next
        );
        Ok(())
    }

    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        let ret = *(*ptr).cast::<usize>() as *mut u8;
        *ptr = (*ptr).add(1);
        ret
    }

    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        ptr::write(ptr.cast::<usize>(), abi as usize);
    }
}

unsafe impl WasmTy for Option<Func> {
    type Abi = *mut VMCallerCheckedAnyfunc;

    #[inline]
    fn compatible_with_store(&self, store: &Store) -> bool {
        if let Some(f) = self {
            Store::same(store, f.store())
        } else {
            true
        }
    }

    #[inline]
    fn into_abi_for_arg(self, _store: &Store) -> Self::Abi {
        if let Some(f) = self {
            f.caller_checked_anyfunc().as_ptr()
        } else {
            ptr::null_mut()
        }
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, store: &Store) -> Self {
        Func::from_caller_checked_anyfunc(store, abi)
    }

    fn valtype() -> Option<ValType> {
        Some(ValType::FuncRef)
    }

    fn matches(mut tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        let next = tys.next();
        ensure!(
            next == Some(ValType::FuncRef),
            "Type mismatch, expected funcref, got {:?}",
            next
        );
        Ok(())
    }

    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        let ret = *(*ptr).cast::<usize>() as *mut VMCallerCheckedAnyfunc;
        *ptr = (*ptr).add(1);
        ret
    }

    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        ptr::write(ptr.cast::<usize>(), abi as usize);
    }
}

unsafe impl<T> WasmRet for T
where
    T: WasmTy,
{
    type Abi = <T as WasmTy>::Abi;
    type Ok = T;

    #[inline]
    fn compatible_with_store(&self, store: &Store) -> bool {
        <Self as WasmTy>::compatible_with_store(self, store)
    }

    #[inline]
    unsafe fn into_abi_for_ret(self, store: &Store) -> Self::Abi {
        <Self as WasmTy>::into_abi_for_arg(self, store)
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, store: &Store) -> Self {
        <Self as WasmTy>::from_abi(abi, store)
    }

    fn valtype() -> Option<ValType> {
        <Self as WasmTy>::valtype()
    }

    #[inline]
    fn matches(tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        <Self as WasmTy>::matches(tys)
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        <Self as WasmTy>::load_from_args(ptr)
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        <Self as WasmTy>::store_to_args(abi, ptr)
    }

    #[inline]
    fn into_result(self) -> Result<T, Trap> {
        Ok(self)
    }
}

unsafe impl<T> WasmRet for Result<T, Trap>
where
    T: WasmTy,
{
    type Abi = <T as WasmTy>::Abi;
    type Ok = T;

    #[inline]
    fn compatible_with_store(&self, store: &Store) -> bool {
        match self {
            Ok(x) => <T as WasmTy>::compatible_with_store(x, store),
            Err(_) => true,
        }
    }

    #[inline]
    unsafe fn into_abi_for_ret(self, store: &Store) -> Self::Abi {
        match self {
            Ok(val) => return <T as WasmTy>::into_abi_for_arg(val, store),
            Err(trap) => handle_trap(trap),
        }

        unsafe fn handle_trap(trap: Trap) -> ! {
            raise_user_trap(trap.into())
        }
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, store: &Store) -> Self {
        Ok(<T as WasmTy>::from_abi(abi, store))
    }

    fn valtype() -> Option<ValType> {
        <T as WasmTy>::valtype()
    }

    fn matches(tys: impl Iterator<Item = ValType>) -> anyhow::Result<()> {
        <T as WasmTy>::matches(tys)
    }

    #[inline]
    unsafe fn load_from_args(ptr: &mut *const u128) -> Self::Abi {
        <T as WasmTy>::load_from_args(ptr)
    }

    #[inline]
    unsafe fn store_to_args(abi: Self::Abi, ptr: *mut u128) {
        <T as WasmTy>::store_to_args(abi, ptr);
    }

    #[inline]
    fn into_result(self) -> Result<T, Trap> {
        self
    }
}

/// Internal trait implemented for all arguments that can be passed to
/// [`Func::wrap`] and [`Config::wrap_host_func`](crate::Config::wrap_host_func).
///
/// This trait should not be implemented by external users, it's only intended
/// as an implementation detail of this crate.
pub trait IntoFunc<Params, Results> {
    #[doc(hidden)]
    fn into_func(
        self,
        allocator: &dyn InstanceAllocator,
        registry: Option<&mut SignatureRegistry>,
    ) -> (FuncType, InstanceHandle, VMTrampoline);
}

/// A structure representing the *caller's* context when creating a function
/// via [`Func::wrap`].
///
/// This structure can be taken as the first parameter of a closure passed to
/// [Func::wrap], and it can be used to learn information about the caller of
/// the function, such as the calling module's memory, exports, etc.
///
/// The primary purpose of this structure is to provide access to the
/// caller's information, namely it's exported memory and exported functions. This
/// allows functions which take pointers as arguments to easily read the memory the
/// pointers point into, or if a function is expected to call malloc in the wasm
/// module to reserve space for the output you can do that.
///
/// Note that this Caller type a pretty temporary mechanism for accessing the
/// caller's information until interface types has been fully standardized and
/// implemented. The interface types proposal will obsolete this type and this will
/// be removed in the future at some point after interface types is implemented. If
/// you're relying on this Caller type it's recommended to become familiar with
/// interface types to ensure that your use case is covered by the proposal.
pub struct Caller<'a> {
    store: &'a Store,
    caller_vmctx: *mut VMContext,
}

impl Caller<'_> {
    /// Looks up an export from the caller's module by the `name` given.
    ///
    /// Note that this function is only implemented for the `Extern::Memory`
    /// and the `Extern::Func` types currently. No other exported structures
    /// can be acquired through this just yet, but this may be implemented
    /// in the future!
    ///
    /// Note that when accessing and calling exported functions, one should adhere
    /// to the guidelines of the interface types proposal.
    ///
    /// # Return
    ///
    /// If a memory or function export with the `name` provided was found, then it is
    /// returned as a `Memory`. There are a number of situations, however, where
    /// the memory or function may not be available:
    ///
    /// * The caller instance may not have an export named `name`
    /// * The export named `name` may not be an exported memory
    /// * There may not be a caller available, for example if `Func` was called
    ///   directly from host code.
    ///
    /// It's recommended to take care when calling this API and gracefully
    /// handling a `None` return value.
    pub fn get_export(&self, name: &str) -> Option<Extern> {
        unsafe {
            if self.caller_vmctx.is_null() {
                return None;
            }
            let instance = InstanceHandle::from_vmctx(self.caller_vmctx);
            let handle = self.store.existing_instance_handle(instance);
            let index = handle.module().exports.get(name)?;
            match index {
                // Only allow memory/functions for now to emulate what interface
                // types will once provide
                EntityIndex::Memory(_) | EntityIndex::Function(_) => {
                    Some(Extern::from_wasmtime_export(
                        &handle.lookup_by_declaration(&index),
                        &handle.store,
                    ))
                }
                _ => None,
            }
        }
    }

    /// Get a reference to the caller's store.
    pub fn store(&self) -> &Store {
        self.store
    }
}

#[inline(never)]
#[cold]
unsafe fn raise_cross_store_trap() -> ! {
    #[derive(Debug)]
    struct CrossStoreError;

    impl std::error::Error for CrossStoreError {}

    impl fmt::Display for CrossStoreError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "host function attempted to return cross-`Store` \
                 value to Wasm",
            )
        }
    }

    raise_user_trap(Box::new(CrossStoreError));
}

macro_rules! impl_into_func {
    ($num:tt $($args:ident)*) => {
        // Implement for functions without a leading `&Caller` parameter,
        // delegating to the implementation below which does have the leading
        // `Caller` parameter.
        #[allow(non_snake_case)]
        impl<F, $($args,)* R> IntoFunc<($($args,)*), R> for F
        where
            F: Fn($($args),*) -> R + 'static,
            $($args: WasmTy,)*
            R: WasmRet,
        {
            fn into_func(self, allocator: &dyn InstanceAllocator, registry: Option<&mut SignatureRegistry>) -> (FuncType, InstanceHandle, VMTrampoline) {
                let f = move |_: Caller<'_>, $($args:$args),*| {
                    self($($args),*)
                };

                f.into_func(allocator, registry)
            }
        }

        #[allow(non_snake_case)]
        impl<F, $($args,)* R> IntoFunc<(Caller<'_>, $($args,)*), R> for F
        where
            F: Fn(Caller<'_>, $($args),*) -> R + 'static,
            $($args: WasmTy,)*
            R: WasmRet,
        {
            fn into_func(self, allocator: &dyn InstanceAllocator, registry: Option<&mut SignatureRegistry>) -> (FuncType, InstanceHandle, VMTrampoline) {
                /// This shim is called by Wasm code, constructs a `Caller`,
                /// calls the wrapped host function, and returns the translated
                /// result back to Wasm.
                ///
                /// Note that this shim's ABI must *exactly* match that expected
                /// by Cranelift, since Cranelift is generating raw function
                /// calls directly to this function.
                unsafe extern "C" fn wasm_to_host_shim<F, $($args,)* R>(
                    vmctx: *mut VMContext,
                    caller_vmctx: *mut VMContext,
                    $( $args: $args::Abi, )*
                ) -> R::Abi
                where
                    F: Fn(Caller<'_>, $( $args ),*) -> R + 'static,
                    $( $args: WasmTy, )*
                    R: WasmRet,
                {
                    let state = (*vmctx).host_state();
                    // Double-check ourselves in debug mode, but we control
                    // the `Any` here so an unsafe downcast should also
                    // work.
                    debug_assert!(state.is::<F>());
                    let func = &*(state as *const _ as *const F);

                    let store = wasmtime_runtime::with_last_info(|last| {
                        last.and_then(Any::downcast_ref::<Store>)
                            .cloned()
                            .expect("function called without thread state")
                    });

                    let ret = {
                        panic::catch_unwind(AssertUnwindSafe(|| {
                            func(
                                Caller { store: &store, caller_vmctx },
                                $( $args::from_abi($args, &store), )*
                            )
                        }))
                    };

                    // Note that we need to be careful when dealing with traps
                    // here. Traps are implemented with longjmp/setjmp meaning
                    // that it's not unwinding and consequently no Rust
                    // destructors are run. We need to be careful to ensure that
                    // nothing on the stack needs a destructor when we exit
                    // abnormally from this `match`, e.g. on `Err`, on
                    // cross-store-issues, or if `Ok(Err)` is raised.
                    match ret {
                        Err(panic) => wasmtime_runtime::resume_panic(panic),
                        Ok(ret) => {
                            // Because the wrapped function is not `unsafe`, we
                            // can't assume it returned a value that is
                            // compatible with this store.
                            if !ret.compatible_with_store(&store) {
                                // Explicitly drop all locals with destructors prior to raising the trap
                                drop(store);
                                drop(ret);
                                raise_cross_store_trap();
                            }

                            ret.into_abi_for_ret(&store)
                        }
                    }
                }

                /// This trampoline allows host code to indirectly call the
                /// wrapped function (e.g. via `Func::call` on a `funcref` that
                /// happens to reference our wrapped function).
                ///
                /// It reads the arguments out of the incoming `args` array,
                /// calls the given function pointer, and then stores the result
                /// back into the `args` array.
                unsafe extern "C" fn host_trampoline<$($args,)* R>(
                    callee_vmctx: *mut VMContext,
                    caller_vmctx: *mut VMContext,
                    ptr: *const VMFunctionBody,
                    args: *mut u128,
                )
                where
                    $($args: WasmTy,)*
                    R: WasmRet,
                {
                    let ptr = mem::transmute::<
                        *const VMFunctionBody,
                        unsafe extern "C" fn(
                            *mut VMContext,
                            *mut VMContext,
                            $( $args::Abi, )*
                        ) -> R::Abi,
                    >(ptr);

                    let mut _next = args as *const u128;
                    $( let $args = $args::load_from_args(&mut _next); )*
                    let ret = ptr(callee_vmctx, caller_vmctx, $( $args ),*);
                    R::store_to_args(ret, args);
                }

                let ty = FuncType::new(
                    None::<ValType>.into_iter()
                        $(.chain($args::valtype()))*
                    ,
                    R::valtype(),
                );

                let trampoline = host_trampoline::<$($args,)* R>;

                // If not given a registry, use a default signature index that is guaranteed to trap
                // if the function is called indirectly without first being associated with a store (a bug condition).
                let shared_signature_id = registry
                    .map(|r| r.register(ty.as_wasm_func_type(), trampoline))
                    .unwrap_or(VMSharedSignatureIndex::default());

                let instance = unsafe {
                    crate::trampoline::create_raw_function(
                        allocator,
                        std::slice::from_raw_parts_mut(
                            wasm_to_host_shim::<F, $($args,)* R> as *mut _,
                            0,
                        ),
                        Box::new(self),
                        shared_signature_id
                    )
                    .expect("failed to create raw function")
                };

                (ty, instance, trampoline)
            }
        }
    }
}

for_each_function_signature!(impl_into_func);

/// Returned future from the [`Func::get1_async`] family of methods, used to
/// represent an asynchronous call into WebAssembly.
pub struct WasmCall<R> {
    inner: Pin<Box<dyn Future<Output = Result<R, Trap>>>>,
}

impl<R> Future for WasmCall<R> {
    type Output = Result<R, Trap>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

#[test]
fn wasm_ty_roundtrip() -> Result<(), anyhow::Error> {
    use crate::*;
    let store = Store::default();
    let debug = Func::wrap(&store, |a: i32, b: u32, c: f32, d: i64, e: u64, f: f64| {
        assert_eq!(a, -1);
        assert_eq!(b, 1);
        assert_eq!(c, 2.0);
        assert_eq!(d, -3);
        assert_eq!(e, 3);
        assert_eq!(f, 4.0);
    });
    let module = Module::new(
        store.engine(),
        r#"
             (module
                 (import "" "" (func $debug (param i32 i32 f32 i64 i64 f64)))
                 (func (export "foo") (param i32 i32 f32 i64 i64 f64)
                    (if (i32.ne (local.get 0) (i32.const -1))
                        (then unreachable)
                    )
                    (if (i32.ne (local.get 1) (i32.const 1))
                        (then unreachable)
                    )
                    (if (f32.ne (local.get 2) (f32.const 2))
                        (then unreachable)
                    )
                    (if (i64.ne (local.get 3) (i64.const -3))
                        (then unreachable)
                    )
                    (if (i64.ne (local.get 4) (i64.const 3))
                        (then unreachable)
                    )
                    (if (f64.ne (local.get 5) (f64.const 4))
                        (then unreachable)
                    )
                    local.get 0
                    local.get 1
                    local.get 2
                    local.get 3
                    local.get 4
                    local.get 5
                    call $debug
                )
            )
         "#,
    )?;
    let instance = Instance::new(&store, &module, &[debug.into()])?;
    let foo = instance
        .get_func("foo")
        .unwrap()
        .get6::<i32, u32, f32, i64, u64, f64, ()>()?;
    foo(-1, 1, 2.0, -3, 3, 4.0)?;
    Ok(())
}
