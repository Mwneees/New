//! Low-level abstraction for allocating and managing zero-filled pages
//! of memory.

use anyhow::{bail, Context, Result};
use more_asserts::assert_le;
use std::fs::File;
use std::io;
use std::ptr;
use std::slice;

/// Round `size` up to the nearest multiple of `page_size`.
fn round_up_to_page_size(size: usize, page_size: usize) -> usize {
    (size + (page_size - 1)) & !(page_size - 1)
}

/// A simple struct consisting of a page-aligned pointer to page-aligned
/// and initially-zeroed memory and a length.
#[derive(Debug)]
pub struct Mmap {
    // Note that this is stored as a `usize` instead of a `*const` or `*mut`
    // pointer to allow this structure to be natively `Send` and `Sync` without
    // `unsafe impl`. This type is sendable across threads and shareable since
    // the coordination all happens at the OS layer.
    ptr: usize,
    len: usize,
    file: Option<File>,
}

impl Mmap {
    /// Construct a new empty instance of `Mmap`.
    pub fn new() -> Self {
        // Rust's slices require non-null pointers, even when empty. `Vec`
        // contains code to create a non-null dangling pointer value when
        // constructed empty, so we reuse that here.
        let empty = Vec::<u8>::new();
        Self {
            ptr: empty.as_ptr() as usize,
            len: 0,
            file: None,
        }
    }

    /// Create a new `Mmap` pointing to at least `size` bytes of page-aligned accessible memory.
    pub fn with_at_least(size: usize) -> Result<Self> {
        let page_size = region::page::size();
        let rounded_size = round_up_to_page_size(size, page_size);
        Self::accessible_reserved(rounded_size, rounded_size)
    }

    /// Creates a new `Mmap` from the specified open `file` with the `len`
    /// specified.
    ///
    /// The memory is mapped in read-only mode for the entire file. If portions
    /// of the file need to be modified then the `region` crate can be use to
    /// alter permissions of each page.
    ///
    /// Note that `len` doesn't have to be page-aligned, but rather it's
    /// expected to be the length of the `file` provided.
    pub fn new_file(file: File, len: usize) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::prelude::*;
            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr as isize == -1_isize {
                return Err(io::Error::last_os_error())
                    .context(format!("mmap failed to allocate {:#x} bytes", len));
            }

            Ok(Self {
                ptr: ptr as usize,
                len,
                file: Some(file),
            })
        }
        #[cfg(windows)]
        {
            use std::os::windows::prelude::*;
            use winapi::um::handleapi::*;
            use winapi::um::memoryapi::*;
            use winapi::um::winnt::*;
            unsafe {
                let mapping = CreateFileMappingW(
                    file.as_raw_handle().cast(),
                    ptr::null_mut(),
                    PAGE_READONLY,
                    0,
                    0,
                    ptr::null(),
                );
                if mapping.is_null() {
                    return Err(io::Error::last_os_error())
                        .context(format!("failed to create file mapping of {:#x} bytes", len));
                }
                let ptr = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len);
                CloseHandle(mapping);
                if ptr.is_null() {
                    return Err(io::Error::last_os_error())
                        .context(format!("failed to create map view of {:#x} bytes", len));
                }
                Ok(Self {
                    ptr: ptr as usize,
                    len,
                    file: Some(file),
                })
            }
        }
    }

    /// Create a new `Mmap` pointing to `accessible_size` bytes of page-aligned accessible memory,
    /// within a reserved mapping of `mapping_size` bytes. `accessible_size` and `mapping_size`
    /// must be native page-size multiples.
    #[cfg(not(target_os = "windows"))]
    pub fn accessible_reserved(accessible_size: usize, mapping_size: usize) -> Result<Self> {
        let page_size = region::page::size();
        assert_le!(accessible_size, mapping_size);
        assert_eq!(mapping_size & (page_size - 1), 0);
        assert_eq!(accessible_size & (page_size - 1), 0);

        // Mmap may return EINVAL if the size is zero, so just
        // special-case that.
        if mapping_size == 0 {
            return Ok(Self::new());
        }

        Ok(if accessible_size == mapping_size {
            // Allocate a single read-write region at once.
            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    mapping_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if ptr as isize == -1_isize {
                bail!(
                    "mmap failed to allocate {:#x} bytes: {}",
                    mapping_size,
                    io::Error::last_os_error()
                );
            }

            Self {
                ptr: ptr as usize,
                len: mapping_size,
                file: None,
            }
        } else {
            // Reserve the mapping size.
            let ptr = unsafe {
                libc::mmap(
                    ptr::null_mut(),
                    mapping_size,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if ptr as isize == -1_isize {
                bail!(
                    "mmap failed to allocate {:#x} bytes: {}",
                    mapping_size,
                    io::Error::last_os_error()
                );
            }

            let mut result = Self {
                ptr: ptr as usize,
                len: mapping_size,
                file: None,
            };

            if accessible_size != 0 {
                // Commit the accessible size.
                result.make_accessible(0, accessible_size)?;
            }

            result
        })
    }

    /// Create a new `Mmap` pointing to `accessible_size` bytes of page-aligned accessible memory,
    /// within a reserved mapping of `mapping_size` bytes. `accessible_size` and `mapping_size`
    /// must be native page-size multiples.
    #[cfg(target_os = "windows")]
    pub fn accessible_reserved(accessible_size: usize, mapping_size: usize) -> Result<Self> {
        use winapi::um::memoryapi::VirtualAlloc;
        use winapi::um::winnt::{MEM_COMMIT, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE};

        if mapping_size == 0 {
            return Ok(Self::new());
        }

        let page_size = region::page::size();
        assert_le!(accessible_size, mapping_size);
        assert_eq!(mapping_size & (page_size - 1), 0);
        assert_eq!(accessible_size & (page_size - 1), 0);

        Ok(if accessible_size == mapping_size {
            // Allocate a single read-write region at once.
            let ptr = unsafe {
                VirtualAlloc(
                    ptr::null_mut(),
                    mapping_size,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            if ptr.is_null() {
                bail!("VirtualAlloc failed: {}", io::Error::last_os_error());
            }

            Self {
                ptr: ptr as usize,
                len: mapping_size,
                file: None,
            }
        } else {
            // Reserve the mapping size.
            let ptr =
                unsafe { VirtualAlloc(ptr::null_mut(), mapping_size, MEM_RESERVE, PAGE_NOACCESS) };
            if ptr.is_null() {
                bail!("VirtualAlloc failed: {}", io::Error::last_os_error());
            }

            let mut result = Self {
                ptr: ptr as usize,
                len: mapping_size,
                file: None,
            };

            if accessible_size != 0 {
                // Commit the accessible size.
                result.make_accessible(0, accessible_size)?;
            }

            result
        })
    }

    /// Make the memory starting at `start` and extending for `len` bytes accessible.
    /// `start` and `len` must be native page-size multiples and describe a range within
    /// `self`'s reserved memory.
    #[cfg(not(target_os = "windows"))]
    pub fn make_accessible(&mut self, start: usize, len: usize) -> Result<()> {
        let page_size = region::page::size();
        assert_eq!(start & (page_size - 1), 0);
        assert_eq!(len & (page_size - 1), 0);
        assert_le!(len, self.len);
        assert_le!(start, self.len - len);

        // Commit the accessible size.
        let ptr = self.ptr as *const u8;
        unsafe {
            region::protect(ptr.add(start), len, region::Protection::READ_WRITE)?;
        }

        Ok(())
    }

    /// Make the memory starting at `start` and extending for `len` bytes accessible.
    /// `start` and `len` must be native page-size multiples and describe a range within
    /// `self`'s reserved memory.
    #[cfg(target_os = "windows")]
    pub fn make_accessible(&mut self, start: usize, len: usize) -> Result<()> {
        use winapi::ctypes::c_void;
        use winapi::um::memoryapi::VirtualAlloc;
        use winapi::um::winnt::{MEM_COMMIT, PAGE_READWRITE};
        let page_size = region::page::size();
        assert_eq!(start & (page_size - 1), 0);
        assert_eq!(len & (page_size - 1), 0);
        assert_le!(len, self.len);
        assert_le!(start, self.len - len);

        // Commit the accessible size.
        let ptr = self.ptr as *const u8;
        if unsafe {
            VirtualAlloc(
                ptr.add(start) as *mut c_void,
                len,
                MEM_COMMIT,
                PAGE_READWRITE,
            )
        }
        .is_null()
        {
            bail!("VirtualAlloc failed: {}", io::Error::last_os_error());
        }

        Ok(())
    }

    /// Return the allocated memory as a slice of u8.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// Return the allocated memory as a mutable slice of u8.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr as *mut u8, self.len) }
    }

    /// Return the allocated memory as a pointer to u8.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// Return the allocated memory as a mutable pointer to u8.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr as *mut u8
    }

    /// Return the length of the allocated memory.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether any memory has been allocated.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for Mmap {
    #[cfg(not(target_os = "windows"))]
    fn drop(&mut self) {
        if self.len != 0 {
            let r = unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
            assert_eq!(r, 0, "munmap failed: {}", io::Error::last_os_error());
        }
    }

    #[cfg(target_os = "windows")]
    fn drop(&mut self) {
        if self.len != 0 {
            use winapi::ctypes::c_void;
            use winapi::um::memoryapi::*;
            use winapi::um::winnt::MEM_RELEASE;
            if self.file.is_none() {
                let r = unsafe { VirtualFree(self.ptr as *mut c_void, 0, MEM_RELEASE) };
                assert_ne!(r, 0);
            } else {
                let r = unsafe { UnmapViewOfFile(self.ptr as *mut c_void) };
                assert_ne!(r, 0);
            }
        }
    }
}

fn _assert() {
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<Mmap>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up_to_page_size() {
        assert_eq!(round_up_to_page_size(0, 4096), 0);
        assert_eq!(round_up_to_page_size(1, 4096), 4096);
        assert_eq!(round_up_to_page_size(4096, 4096), 4096);
        assert_eq!(round_up_to_page_size(4097, 4096), 8192);
    }
}
