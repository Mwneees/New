use crate::{bindings, WasiHttpView};
use wasmtime_wasi::preview2;

wasmtime::component::bindgen!({
    world: "wasi:http/proxy@0.2.0",
    tracing: true,
    async: true,
    with: {
        "wasi:cli/stderr@0.2.0": preview2::bindings::cli::stderr,
        "wasi:cli/stdin@0.2.0": preview2::bindings::cli::stdin,
        "wasi:cli/stdout@0.2.0": preview2::bindings::cli::stdout,
        "wasi:clocks/monotonic-clock@0.2.0": preview2::bindings::clocks::monotonic_clock,
        "wasi:clocks/timezone@0.2.0": preview2::bindings::clocks::timezone,
        "wasi:clocks/wall-clock@0.2.0": preview2::bindings::clocks::wall_clock,
        "wasi:http/incoming-handler@0.2.0": bindings::http::incoming_handler,
        "wasi:http/outgoing-handler@0.2.0": bindings::http::outgoing_handler,
        "wasi:http/types@0.2.0": bindings::http::types,
        "wasi:io/streams@0.2.0": preview2::bindings::io::streams,
        "wasi:io/poll@0.2.0": preview2::bindings::io::poll,
        "wasi:random/random@0.2.0": preview2::bindings::random::random,
    },
});

pub fn add_to_linker<T>(l: &mut wasmtime::component::Linker<T>) -> anyhow::Result<()>
where
    T: WasiHttpView + preview2::WasiView + bindings::http::types::Host,
{
    // TODO: this shouldn't be required, but the adapter unconditionally pulls in all of these
    // dependencies.
    preview2::command::add_to_linker(l)?;

    bindings::http::outgoing_handler::add_to_linker(l, |t| t)?;
    bindings::http::types::add_to_linker(l, |t| t)?;

    Ok(())
}

pub mod sync {
    use crate::{bindings, WasiHttpView};
    use wasmtime_wasi::preview2;

    wasmtime::component::bindgen!({
        world: "wasi:http/proxy@0.2.0",
        tracing: true,
        async: false,
        with: {
            "wasi:cli/stderr@0.2.0": preview2::bindings::cli::stderr,
            "wasi:cli/stdin@0.2.0": preview2::bindings::cli::stdin,
            "wasi:cli/stdout@0.2.0": preview2::bindings::cli::stdout,
            "wasi:clocks/monotonic-clock@0.2.0": preview2::bindings::clocks::monotonic_clock,
            "wasi:clocks/timezone@0.2.0": preview2::bindings::clocks::timezone,
            "wasi:clocks/wall-clock@0.2.0": preview2::bindings::clocks::wall_clock,
            "wasi:http/incoming-handler@0.2.0": bindings::http::incoming_handler,
            "wasi:http/outgoing-handler@0.2.0": bindings::http::outgoing_handler,
            "wasi:http/types@0.2.0": bindings::http::types,
            "wasi:io/streams@0.2.0": preview2::bindings::io::streams,
            "wasi:poll/poll@0.2.0": preview2::bindings::poll::poll,
            "wasi:random/random@0.2.0": preview2::bindings::random::random,
        },
    });

    pub fn add_to_linker<T>(l: &mut wasmtime::component::Linker<T>) -> anyhow::Result<()>
    where
        T: WasiHttpView + preview2::WasiView + bindings::http::types::Host,
    {
        // TODO: this shouldn't be required, but the adapter unconditionally pulls in all of these
        // dependencies.
        preview2::command::sync::add_to_linker(l)?;

        add_only_http_to_linker(l)?;

        Ok(())
    }

    #[doc(hidden)]
    // TODO: This is temporary solution until the preview2 command functions can be removed
    pub fn add_only_http_to_linker<T>(l: &mut wasmtime::component::Linker<T>) -> anyhow::Result<()>
    where
        T: WasiHttpView + preview2::WasiView + bindings::http::types::Host,
    {
        bindings::http::outgoing_handler::add_to_linker(l, |t| t)?;
        bindings::http::types::add_to_linker(l, |t| t)?;

        Ok(())
    }
}
