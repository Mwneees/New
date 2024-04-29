use crate::{
    store::{AutoAssertNoGc, StoreOpaque},
    HeapType, Ref, RefType, Result, Uninhabited, Val, ValRaw, ValType, WasmTy,
};

/// A reference to the abstract `noextern` heap value.
///
/// The are no instances of `(ref noextern)`: it is an uninhabited type.
///
/// There is precisely one instance of `(ref null noextern)`, aka `nullexternref`:
/// the null reference.
///
/// This `NoExtern` Rust type's sole purpose is for use with
/// [`Func::wrap`][crate::Func::wrap]- and
/// [`Func::typed`][crate::Func::typed]-style APIs for statically typing a
/// function as taking or returning a `(ref null noextern)` (aka
/// `Option<NoExtern>`) which is always `None`.
///
/// # Example
///
/// ```
/// # use wasmtime::*;
/// # fn _foo() -> Result<()> {
/// let mut config = Config::new();
/// config.wasm_function_references(true);
/// config.wasm_gc(true);
/// let engine = Engine::new(&config)?;
///
/// let module = Module::new(
///     &engine,
///     r#"
///         (module
///             (func (export "f") (param (ref null noextern))
///                 ;; If the reference is null, return.
///                 local.get 0
///                 ref.is_null noextern
///                 br_if 0
///
///                 ;; If the reference was not null (which is impossible)
///                 ;; then raise a trap.
///                 unreachable
///             )
///         )
///     "#,
/// )?;
///
/// let mut store = Store::new(&engine, ());
/// let instance = Instance::new(&mut store, &module, &[])?;
/// let f = instance.get_func(&mut store, "f").unwrap();
///
/// // We can cast a `(ref null noextern)`-taking function into a typed function that
/// // takes an `Option<NoExtern>` via the `Func::typed` method.
/// let f = f.typed::<Option<NoExtern>, ()>(&store)?;
///
/// // We can call the typed function, passing the null `noextern` reference.
/// let result = f.call(&mut store, NoExtern::null());
///
/// // The function should not have trapped, because the reference we gave it was
/// // null (as it had to be, since `NoExtern` is uninhabited).
/// assert!(result.is_ok());
/// # Ok(())
/// # }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NoExtern {
    _inner: Uninhabited,
}

impl NoExtern {
    /// Get the null `(ref null noextern)` (aka `nullexternref`) reference.
    #[inline]
    pub fn null() -> Option<Self> {
        None
    }

    /// Get the null `(ref null noextern)` (aka `nullexternref`) reference as a
    /// [`Ref`].
    #[inline]
    pub fn null_ref() -> Ref {
        Ref::Extern(None)
    }

    /// Get the null `(ref null noextern)` (aka `nullexternref`) reference as a
    /// [`Val`].
    #[inline]
    pub fn null_val() -> Val {
        Val::ExternRef(None)
    }
}

unsafe impl WasmTy for NoExtern {
    type Abi = NoExtern;

    #[inline]
    fn valtype() -> ValType {
        ValType::Ref(RefType::new(false, HeapType::NoExtern))
    }

    #[inline]
    fn compatible_with_store(&self, _store: &StoreOpaque) -> bool {
        match self._inner {}
    }

    #[inline]
    fn dynamic_concrete_type_check(&self, _: &StoreOpaque, _: bool, _: &HeapType) -> Result<()> {
        match self._inner {}
    }

    #[inline]
    fn is_vmgcref_and_points_to_object(&self) -> bool {
        match self._inner {}
    }

    #[inline]
    unsafe fn abi_from_raw(_raw: *mut ValRaw) -> Self::Abi {
        unreachable!("NoExtern is uninhabited")
    }

    #[inline]
    unsafe fn abi_into_raw(abi: Self::Abi, _raw: *mut ValRaw) {
        match abi._inner {}
    }

    #[inline]
    fn into_abi(self, _store: &mut AutoAssertNoGc<'_>) -> Result<Self::Abi> {
        unreachable!("NoExtern is uninhabited")
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &mut AutoAssertNoGc<'_>) -> Self {
        match abi._inner {}
    }
}

unsafe impl WasmTy for Option<NoExtern> {
    type Abi = u32;

    #[inline]
    fn valtype() -> ValType {
        ValType::Ref(RefType::new(true, HeapType::NoExtern))
    }

    #[inline]
    fn compatible_with_store(&self, _store: &StoreOpaque) -> bool {
        true
    }

    #[inline]
    fn dynamic_concrete_type_check(
        &self,
        _store: &StoreOpaque,
        _nullable: bool,
        _ty: &HeapType,
    ) -> Result<()> {
        unreachable!()
    }

    #[inline]
    unsafe fn abi_from_raw(_raw: *mut ValRaw) -> Self::Abi {
        0
    }

    #[inline]
    unsafe fn abi_into_raw(abi: Self::Abi, raw: *mut ValRaw) {
        debug_assert_eq!(abi, 0);
        *raw = ValRaw::externref(0);
    }

    #[inline]
    fn into_abi(self, _store: &mut AutoAssertNoGc<'_>) -> Result<Self::Abi> {
        Ok(0)
    }

    #[inline]
    unsafe fn from_abi(abi: Self::Abi, _store: &mut AutoAssertNoGc<'_>) -> Self {
        debug_assert_eq!(abi, 0);
        None
    }
}
