include!(concat!(env!("OUT_DIR"), "/gen.rs"));

use std::io::IsTerminal;

/// The wasi-tests binaries use these environment variables to determine their
/// expected behavior.
/// Used by all of the tests/ which execute the wasi-tests binaries.
pub fn wasi_tests_environment() -> &'static [(&'static str, &'static str)] {
    #[cfg(windows)]
    {
        &[
            ("ERRNO_MODE_WINDOWS", "1"),
            // Windows does not support dangling links or symlinks in the filesystem.
            ("NO_DANGLING_FILESYSTEM", "1"),
            // Windows does not support renaming a directory to an empty directory -
            // empty directory must be deleted.
            ("NO_RENAME_DIR_TO_EMPTY_DIR", "1"),
            // cap-std-sync does not support the sync family of fdflags
            ("NO_FDFLAGS_SYNC_SUPPORT", "1"),
        ]
    }
    #[cfg(all(unix, all(not(target_os = "macos"), not(target_os = "ios"))))]
    {
        &[
            ("ERRNO_MODE_UNIX", "1"),
            // cap-std-sync does not support the sync family of fdflags
            ("NO_FDFLAGS_SYNC_SUPPORT", "1"),
        ]
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        &[
            ("ERRNO_MODE_MACOS", "1"),
            // cap-std-sync does not support the sync family of fdflags
            ("NO_FDFLAGS_SYNC_SUPPORT", "1"),
        ]
    }
}

pub fn stdio_is_terminal() -> bool {
    std::io::stdin().is_terminal()
        && std::io::stdout().is_terminal()
        && std::io::stderr().is_terminal()
}
