//! ## The `WasiFile` and `WasiDir` traits
//!
//! The WASI specification only defines one `handle` type, `fd`, on which all
//! operations on both files and directories (aka dirfds) are defined. We
//! believe this is a design mistake, and are architecting wasi-common to make
//! this straightforward to correct in future snapshots of WASI.  Wasi-common
//! internally treats files and directories as two distinct resource types in
//! the table - `Box<dyn WasiFile>` and `Box<dyn WasiDir>`. The snapshot 0 and
//! 1 interfaces via `fd` will attempt to downcast a table element to one or
//! both of these interfaces depending on what is appropriate - e.g.
//! `fd_close` operates on both files and directories, `fd_read` only operates
//! on files, and `fd_readdir` only operates on directories.

//! The `WasiFile` and `WasiDir` traits are defined by `wasi-common` in terms
//! of types defined directly in the crate's source code (I decided it should
//! NOT those generated by the `wiggle` proc macros, see snapshot architecture
//! below), as well as the `cap_std::time` family of types.  And, importantly,
//! `wasi-common` itself provides no implementation of `WasiDir`, and only two
//! trivial implementations of `WasiFile` on the `crate::pipe::{ReadPipe,
//! WritePipe}` types, which in turn just delegate to `std::io::{Read,
//! Write}`. In order for `wasi-common` to access the local filesystem at all,
//! you need to provide `WasiFile` and `WasiDir` impls through either the new
//! `wasi-cap-std-sync` crate found at `crates/wasi-common/cap-std-sync` - see
//! the section on that crate below - or by providing your own implementation
//! from elsewhere.
//!
//! This design makes it possible for `wasi-common` embedders to statically
//! reason about access to the local filesystem by examining what impls are
//! linked into an application. We found that this separation of concerns also
//! makes it pretty enjoyable to write alternative implementations, e.g. a
//! virtual filesystem (which will land in a future PR).
//!
//! ## Traits for the rest of WASI's features
//!
//! Other aspects of a WASI implementation are not yet considered resources
//! and accessed by `handle`. We plan to correct this design deficiency in
//! WASI in the future, but for now we have designed the following traits to
//! provide embedders with the same sort of implementation flexibility they
//! get with WasiFile/WasiDir:
//!
//! * Timekeeping: `WasiSystemClock` and `WasiMonotonicClock` provide the two
//! interfaces for a clock. `WasiSystemClock` represents time as a
//! `cap_std::time::SystemTime`, and `WasiMonotonicClock` represents time as
//! `cap_std::time::Instant`.  * Randomness: we re-use the `cap_rand::RngCore`
//! trait to represent a randomness source. A trivial `Deterministic` impl is
//! provided.  * Scheduling: The `WasiSched` trait abstracts over the
//! `sched_yield` and `poll_oneoff` functions.
//!
//! Users can provide implementations of each of these interfaces to the
//! `WasiCtx::builder(...)` function. The
//! `wasi_cap_std_sync::WasiCtxBuilder::new()` function uses this public
//! interface to plug in its own implementations of each of these resources.

#![warn(clippy::cast_sign_loss)]

pub mod clocks;
mod ctx;
pub mod dir;
mod error;
pub mod file;
pub mod pipe;
pub mod random;
pub mod sched;
pub mod snapshots;
mod string_array;
pub mod table;

pub use cap_rand::RngCore;
pub use clocks::{SystemTimeSpec, WasiClocks, WasiMonotonicClock, WasiSystemClock};
pub use ctx::WasiCtx;
pub use dir::WasiDir;
pub use error::{Error, ErrorExt, I32Exit};
pub use file::WasiFile;
pub use sched::{Poll, WasiSched};
pub use string_array::{StringArray, StringArrayError};
pub use table::Table;
