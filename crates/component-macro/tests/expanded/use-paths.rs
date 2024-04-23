pub struct D {}
const _: () = {
    #[allow(unused_imports)]
    use wasmtime::component::__internal::anyhow;
    impl D {
        pub fn add_to_linker<T, U>(
            linker: &mut wasmtime::component::Linker<T>,
            get: impl Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
        ) -> wasmtime::Result<()>
        where
            U: foo::foo::a::Host + foo::foo::b::Host + foo::foo::c::Host + d::Host,
        {
            foo::foo::a::add_to_linker(linker, get)?;
            foo::foo::b::add_to_linker(linker, get)?;
            foo::foo::c::add_to_linker(linker, get)?;
            d::add_to_linker(linker, get)?;
            Ok(())
        }
        /// Instantiates the provided `module` using the specified
        /// parameters, wrapping up the result in a structure that
        /// translates between wasm and the host.
        pub fn instantiate<T>(
            mut store: impl wasmtime::AsContextMut<Data = T>,
            component: &wasmtime::component::Component,
            linker: &wasmtime::component::Linker<T>,
        ) -> wasmtime::Result<(Self, wasmtime::component::Instance)> {
            let instance = linker.instantiate(&mut store, component)?;
            Ok((Self::new(store, &instance)?, instance))
        }
        /// Instantiates a pre-instantiated module using the specified
        /// parameters, wrapping up the result in a structure that
        /// translates between wasm and the host.
        pub fn instantiate_pre<T>(
            mut store: impl wasmtime::AsContextMut<Data = T>,
            instance_pre: &wasmtime::component::InstancePre<T>,
        ) -> wasmtime::Result<(Self, wasmtime::component::Instance)> {
            let instance = instance_pre.instantiate(&mut store)?;
            Ok((Self::new(store, &instance)?, instance))
        }
        /// Low-level creation wrapper for wrapping up the exports
        /// of the `instance` provided in this structure of wasm
        /// exports.
        ///
        /// This function will extract exports from the `instance`
        /// defined within `store` and wrap them all up in the
        /// returned structure which can be used to interact with
        /// the wasm module.
        pub fn new(
            mut store: impl wasmtime::AsContextMut,
            instance: &wasmtime::component::Instance,
        ) -> wasmtime::Result<Self> {
            let mut store = store.as_context_mut();
            let mut exports = instance.exports(&mut store);
            let mut __exports = exports.root();
            Ok(D {})
        }
    }
};
pub mod foo {
    pub mod foo {
        #[allow(clippy::all)]
        pub mod a {
            #[allow(unused_imports)]
            use wasmtime::component::__internal::anyhow;
            #[derive(wasmtime::component::ComponentType)]
            #[derive(wasmtime::component::Lift)]
            #[derive(wasmtime::component::Lower)]
            #[component(record)]
            #[derive(Clone, Copy)]
            pub struct Foo {}
            impl core::fmt::Debug for Foo {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    f.debug_struct("Foo").finish()
                }
            }
            const _: () = {
                assert!(0 == < Foo as wasmtime::component::ComponentType >::SIZE32);
                assert!(1 == < Foo as wasmtime::component::ComponentType >::ALIGN32);
            };
            pub trait Host {
                fn a(&mut self) -> Foo;
            }
            pub trait GetHost<T>: Send + Sync + Copy + 'static {
                fn get_host<'a>(&self, data: &'a mut T) -> impl Host;
            }
            pub fn add_to_linker_get_host<T>(
                linker: &mut wasmtime::component::Linker<T>,
                host_getter: impl GetHost<T>,
            ) -> wasmtime::Result<()> {
                let mut inst = linker.instance("foo:foo/a")?;
                inst.func_wrap(
                    "a",
                    move |mut caller: wasmtime::StoreContextMut<'_, T>, (): ()| {
                        let host = &mut host_getter.get_host(caller.data_mut());
                        let r = Host::a(host);
                        Ok((r,))
                    },
                )?;
                Ok(())
            }
            impl<T, U, F> GetHost<T> for F
            where
                U: Host,
                F: Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
            {
                fn get_host<'a>(&self, data: &'a mut T) -> impl Host {
                    self(data)
                }
            }
            pub fn add_to_linker<T, U>(
                linker: &mut wasmtime::component::Linker<T>,
                get: impl Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
            ) -> wasmtime::Result<()>
            where
                U: Host,
            {
                add_to_linker_get_host(linker, get)
            }
            impl<_T: Host + ?Sized> Host for &mut _T {
                fn a(&mut self) -> Foo {
                    Host::a(*self)
                }
            }
        }
        #[allow(clippy::all)]
        pub mod b {
            #[allow(unused_imports)]
            use wasmtime::component::__internal::anyhow;
            pub type Foo = super::super::super::foo::foo::a::Foo;
            const _: () = {
                assert!(0 == < Foo as wasmtime::component::ComponentType >::SIZE32);
                assert!(1 == < Foo as wasmtime::component::ComponentType >::ALIGN32);
            };
            pub trait Host {
                fn a(&mut self) -> Foo;
            }
            pub trait GetHost<T>: Send + Sync + Copy + 'static {
                fn get_host<'a>(&self, data: &'a mut T) -> impl Host;
            }
            pub fn add_to_linker_get_host<T>(
                linker: &mut wasmtime::component::Linker<T>,
                host_getter: impl GetHost<T>,
            ) -> wasmtime::Result<()> {
                let mut inst = linker.instance("foo:foo/b")?;
                inst.func_wrap(
                    "a",
                    move |mut caller: wasmtime::StoreContextMut<'_, T>, (): ()| {
                        let host = &mut host_getter.get_host(caller.data_mut());
                        let r = Host::a(host);
                        Ok((r,))
                    },
                )?;
                Ok(())
            }
            impl<T, U, F> GetHost<T> for F
            where
                U: Host,
                F: Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
            {
                fn get_host<'a>(&self, data: &'a mut T) -> impl Host {
                    self(data)
                }
            }
            pub fn add_to_linker<T, U>(
                linker: &mut wasmtime::component::Linker<T>,
                get: impl Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
            ) -> wasmtime::Result<()>
            where
                U: Host,
            {
                add_to_linker_get_host(linker, get)
            }
            impl<_T: Host + ?Sized> Host for &mut _T {
                fn a(&mut self) -> Foo {
                    Host::a(*self)
                }
            }
        }
        #[allow(clippy::all)]
        pub mod c {
            #[allow(unused_imports)]
            use wasmtime::component::__internal::anyhow;
            pub type Foo = super::super::super::foo::foo::b::Foo;
            const _: () = {
                assert!(0 == < Foo as wasmtime::component::ComponentType >::SIZE32);
                assert!(1 == < Foo as wasmtime::component::ComponentType >::ALIGN32);
            };
            pub trait Host {
                fn a(&mut self) -> Foo;
            }
            pub trait GetHost<T>: Send + Sync + Copy + 'static {
                fn get_host<'a>(&self, data: &'a mut T) -> impl Host;
            }
            pub fn add_to_linker_get_host<T>(
                linker: &mut wasmtime::component::Linker<T>,
                host_getter: impl GetHost<T>,
            ) -> wasmtime::Result<()> {
                let mut inst = linker.instance("foo:foo/c")?;
                inst.func_wrap(
                    "a",
                    move |mut caller: wasmtime::StoreContextMut<'_, T>, (): ()| {
                        let host = &mut host_getter.get_host(caller.data_mut());
                        let r = Host::a(host);
                        Ok((r,))
                    },
                )?;
                Ok(())
            }
            impl<T, U, F> GetHost<T> for F
            where
                U: Host,
                F: Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
            {
                fn get_host<'a>(&self, data: &'a mut T) -> impl Host {
                    self(data)
                }
            }
            pub fn add_to_linker<T, U>(
                linker: &mut wasmtime::component::Linker<T>,
                get: impl Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
            ) -> wasmtime::Result<()>
            where
                U: Host,
            {
                add_to_linker_get_host(linker, get)
            }
            impl<_T: Host + ?Sized> Host for &mut _T {
                fn a(&mut self) -> Foo {
                    Host::a(*self)
                }
            }
        }
    }
}
#[allow(clippy::all)]
pub mod d {
    #[allow(unused_imports)]
    use wasmtime::component::__internal::anyhow;
    pub type Foo = super::foo::foo::c::Foo;
    const _: () = {
        assert!(0 == < Foo as wasmtime::component::ComponentType >::SIZE32);
        assert!(1 == < Foo as wasmtime::component::ComponentType >::ALIGN32);
    };
    pub trait Host {
        fn b(&mut self) -> Foo;
    }
    pub trait GetHost<T>: Send + Sync + Copy + 'static {
        fn get_host<'a>(&self, data: &'a mut T) -> impl Host;
    }
    pub fn add_to_linker_get_host<T>(
        linker: &mut wasmtime::component::Linker<T>,
        host_getter: impl GetHost<T>,
    ) -> wasmtime::Result<()> {
        let mut inst = linker.instance("d")?;
        inst.func_wrap(
            "b",
            move |mut caller: wasmtime::StoreContextMut<'_, T>, (): ()| {
                let host = &mut host_getter.get_host(caller.data_mut());
                let r = Host::b(host);
                Ok((r,))
            },
        )?;
        Ok(())
    }
    impl<T, U, F> GetHost<T> for F
    where
        U: Host,
        F: Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
    {
        fn get_host<'a>(&self, data: &'a mut T) -> impl Host {
            self(data)
        }
    }
    pub fn add_to_linker<T, U>(
        linker: &mut wasmtime::component::Linker<T>,
        get: impl Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
    ) -> wasmtime::Result<()>
    where
        U: Host,
    {
        add_to_linker_get_host(linker, get)
    }
    impl<_T: Host + ?Sized> Host for &mut _T {
        fn b(&mut self) -> Foo {
            Host::b(*self)
        }
    }
}
