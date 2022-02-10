//! memfd support: creation of backing images for modules, and logic
//! to support mapping these backing images into memory.

use crate::InstantiationError;
use crate::MmapVec;
use anyhow::Result;
use libc::c_void;
use rustix::fd::AsRawFd;
use std::fs::File;
use std::sync::Arc;
use std::{convert::TryFrom, ops::Range};
use wasmtime_environ::{DefinedMemoryIndex, MemoryInitialization, Module, PrimaryMap};

/// MemFDs containing backing images for certain memories in a module.
///
/// This is meant to be built once, when a module is first
/// loaded/constructed, and then used many times for instantiation.
pub struct ModuleMemFds {
    memories: PrimaryMap<DefinedMemoryIndex, Option<Arc<MemoryMemFd>>>,
}

impl ModuleMemFds {
    /// Get the MemoryMemFd for a given memory.
    pub fn get_memory_image(&self, defined_index: DefinedMemoryIndex) -> Option<&Arc<MemoryMemFd>> {
        self.memories[defined_index].as_ref()
    }
}

/// One backing image for one memory.
#[derive(Debug)]
pub struct MemoryMemFd {
    /// The file descriptor source of this image.
    ///
    /// This might be an mmaped `*.cwasm` file or on Linux it could also be a
    /// `Memfd` as an anonymous file in memory. In either case this is used as
    /// the backing-source for the CoW image.
    fd: MemFdSource,

    /// Length of image, in bytes.
    ///
    /// Note that initial memory size may be larger; leading and trailing zeroes
    /// are truncated (handled by backing fd).
    ///
    /// Must be a multiple of the system page size.
    len: usize,

    /// Image starts this many bytes into `fd` source.
    ///
    /// This is 0 for anonymous-backed memfd files and is the offset of the data
    /// section in a `*.cwasm` file for `*.cwasm`-backed images.
    ///
    /// Must be a multiple of the system page size.
    fd_offset: u64,

    /// Image starts this many bytes into heap space.
    ///
    /// Must be a multiple of the system page size.
    linear_memory_offset: usize,
}

#[derive(Debug)]
enum MemFdSource {
    Mmap(Arc<File>),
    #[cfg(target_os = "linux")]
    Memfd(memfd::Memfd),
}

impl MemFdSource {
    fn as_file(&self) -> &File {
        match self {
            MemFdSource::Mmap(file) => file,
            #[cfg(target_os = "linux")]
            MemFdSource::Memfd(memfd) => memfd.as_file(),
        }
    }
}

impl MemoryMemFd {
    fn new(
        page_size: u32,
        offset: u32,
        data: &[u8],
        mmap: Option<&MmapVec>,
    ) -> Result<Option<MemoryMemFd>> {
        // Sanity-check that various parameters are page-aligned.
        let len = data.len();
        assert_eq!(offset % page_size, 0);
        assert_eq!((len as u32) % page_size, 0);
        let linear_memory_offset = usize::try_from(offset).unwrap();

        // If a backing `mmap` is present then `data` should be a sub-slice of
        // the `mmap`. The sanity-checks here double-check that. Additionally
        // compilation should have ensured that the `data` section is
        // page-aligned within `mmap`, so that's also all double-checked here.
        //
        // Finally if the `mmap` itself comes from a backing file on disk, such
        // as a `*.cwasm` file, then that's a valid source of data for the
        // memory image so we simply return referencing that.
        //
        // Note that this path is platform-agnostic in the sense of all
        // platforms we support support memory mapping copy-on-write data from
        // files, but for now this is still a Linux-specific region of Wasmtime.
        // Some work will be needed to get this file compiling for macOS and
        // Windows.
        if let Some(mmap) = mmap {
            let start = mmap.as_ptr() as usize;
            let end = start + mmap.len();
            let data_start = data.as_ptr() as usize;
            let data_end = data_start + data.len();
            assert!(start <= data_start && data_end <= end);
            assert_eq!((start as u32) % page_size, 0);
            assert_eq!((data_start as u32) % page_size, 0);
            assert_eq!((data_end as u32) % page_size, 0);
            assert_eq!((mmap.original_offset() as u32) % page_size, 0);

            if let Some(file) = mmap.original_file() {
                return Ok(Some(MemoryMemFd {
                    fd: MemFdSource::Mmap(file.clone()),
                    fd_offset: u64::try_from(mmap.original_offset() + (data_start - start))
                        .unwrap(),
                    linear_memory_offset,
                    len,
                }));
            }
        }

        // If `mmap` doesn't come from a file then platform-specific mechanisms
        // may be used to place the data in a form that's amenable to an mmap.

        cfg_if::cfg_if! {
            if #[cfg(target_os = "linux")] {
                // On Linux `memfd_create` is used to create an anonymous
                // in-memory file to represent the heap image. This anonymous
                // file is then used as the basis for further mmaps.

                use std::io::Write;

                let memfd = create_memfd()?;
                memfd.as_file().write_all(data)?;

                // Seal the memfd's data and length.
                //
                // This is a defense-in-depth security mitigation. The
                // memfd will serve as the starting point for the heap of
                // every instance of this module. If anything were to
                // write to this, it could affect every execution. The
                // memfd object itself is owned by the machinery here and
                // not exposed elsewhere, but it is still an ambient open
                // file descriptor at the syscall level, so some other
                // vulnerability that allowed writes to arbitrary fds
                // could modify it. Or we could have some issue with the
                // way that we map it into each instance. To be
                // extra-super-sure that it never changes, and because
                // this costs very little, we use the kernel's "seal" API
                // to make the memfd image permanently read-only.
                memfd.add_seal(memfd::FileSeal::SealGrow)?;
                memfd.add_seal(memfd::FileSeal::SealShrink)?;
                memfd.add_seal(memfd::FileSeal::SealWrite)?;
                memfd.add_seal(memfd::FileSeal::SealSeal)?;

                Ok(Some(MemoryMemFd {
                    fd: MemFdSource::Memfd(memfd),
                    fd_offset: 0,
                    linear_memory_offset,
                    len,
                }))
            } else {
                // Other platforms don't have an easily available way of
                // representing the heap image as an mmap-source right now. We
                // could theoretically create a file and immediately unlink it
                // but that means that data may likely be preserved to disk
                // which isn't what we want here.
                Ok(None)
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn create_memfd() -> Result<memfd::Memfd> {
    // Create the memfd. It needs a name, but the
    // documentation for `memfd_create()` says that names can
    // be duplicated with no issues.
    memfd::MemfdOptions::new()
        .allow_sealing(true)
        .create("wasm-memory-image")
        .map_err(|e| e.into())
}

impl ModuleMemFds {
    /// Create a new `ModuleMemFds` for the given module. This can be
    /// passed in as part of a `InstanceAllocationRequest` to speed up
    /// instantiation and execution by using memfd-backed memories.
    pub fn new(
        module: &Module,
        wasm_data: &[u8],
        mmap: Option<&MmapVec>,
    ) -> Result<Option<ModuleMemFds>> {
        let map = match &module.memory_initialization {
            MemoryInitialization::Static { map } => map,
            _ => return Ok(None),
        };
        let mut memories = PrimaryMap::with_capacity(map.len());
        let page_size = region::page::size() as u32;
        for (memory_index, init) in map {
            // mmap-based-initialization only works for defined memories with a
            // known starting point of all zeros, so bail out if the mmeory is
            // imported.
            let defined_memory = match module.defined_memory_index(memory_index) {
                Some(idx) => idx,
                None => return Ok(None),
            };

            // If there's no initialization for this memory known then we don't
            // need an image for the memory so push `None` and move on.
            let (offset, data_range) = match init {
                Some(pair) => pair,
                None => {
                    memories.push(None);
                    continue;
                }
            };

            // Get the image for this wasm module  as a subslice of `wasm_data`,
            // and then use that to try to create the `MemoryMemFd`. If this
            // creation files then we fail creating `ModuleMemFds` since this
            // memory couldn't be represented.
            let data = &wasm_data[data_range.start as usize..data_range.end as usize];
            let memfd = match MemoryMemFd::new(page_size, *offset, data, mmap)? {
                Some(memfd) => memfd,
                None => return Ok(None),
            };

            let idx = memories.push(Some(Arc::new(memfd)));
            assert_eq!(idx, defined_memory);
        }

        Ok(Some(ModuleMemFds { memories }))
    }
}

/// A single slot handled by the memfd instance-heap mechanism.
///
/// The mmap scheme is:
///
/// base ==> (points here)
/// - (image.offset bytes)   anonymous zero memory, pre-image
/// - (image.len bytes)      CoW mapping of memfd heap image
/// - (up to static_size)    anonymous zero memory, post-image
///
/// The ordering of mmaps to set this up is:
///
/// - once, when pooling allocator is created:
///   - one large mmap to create 8GiB * instances * memories slots
///
/// - per instantiation of new image in a slot:
///   - mmap of anonymous zero memory, from 0 to max heap size
///     (static_size)
///   - mmap of CoW'd memfd image, from `image.offset` to
///     `image.offset + image.len`. This overwrites part of the
///     anonymous zero memory, potentially splitting it into a pre-
///     and post-region.
///   - mprotect(PROT_NONE) on the part of the heap beyond the initial
///     heap size; we re-mprotect it with R+W bits when the heap is
///     grown.
#[derive(Debug)]
pub struct MemFdSlot {
    /// The base of the actual heap memory. Bytes at this address are
    /// what is seen by the Wasm guest code.
    base: usize,
    /// The maximum static memory size, plus post-guard.
    static_size: usize,
    /// The memfd image that backs this memory. May be `None`, in
    /// which case the memory is all zeroes.
    pub(crate) image: Option<Arc<MemoryMemFd>>,
    /// The initial heap size.
    initial_size: usize,
    /// The current heap size. All memory above `base + cur_size`
    /// should be PROT_NONE (mapped inaccessible).
    cur_size: usize,
    /// Whether this slot may have "dirty" pages (pages written by an
    /// instantiation). Set by `instantiate()` and cleared by
    /// `clear_and_remain_ready()`, and used in assertions to ensure
    /// those methods are called properly.
    ///
    /// Invariant: if !dirty, then this memory slot contains a clean
    /// CoW mapping of `image`, if `Some(..)`, and anonymous-zero
    /// memory beyond the image up to `static_size`. The addresses
    /// from offset 0 to `initial_size` are accessible R+W and the
    /// rest of the slot is inaccessible.
    dirty: bool,
    /// Whether this MemFdSlot is responsible for mapping anonymous
    /// memory (to hold the reservation while overwriting mappings
    /// specific to this slot) in place when it is dropped. Default
    /// on, unless the caller knows what they are doing.
    clear_on_drop: bool,
}

impl MemFdSlot {
    /// Create a new MemFdSlot. Assumes that there is an anonymous
    /// mmap backing in the given range to start.
    pub(crate) fn create(base_addr: *mut c_void, initial_size: usize, static_size: usize) -> Self {
        let base = base_addr as usize;
        MemFdSlot {
            base,
            static_size,
            initial_size,
            cur_size: initial_size,
            image: None,
            dirty: false,
            clear_on_drop: true,
        }
    }

    /// Inform the MemFdSlot that it should *not* clear the underlying
    /// address space when dropped. This should be used only when the
    /// caller will clear or reuse the address space in some other
    /// way.
    pub(crate) fn no_clear_on_drop(&mut self) {
        self.clear_on_drop = false;
    }

    pub(crate) fn set_heap_limit(&mut self, size_bytes: usize) -> Result<()> {
        // mprotect the relevant region.
        self.set_protection(self.cur_size..size_bytes, region::Protection::READ_WRITE)?;
        self.cur_size = size_bytes;

        Ok(())
    }

    pub(crate) fn instantiate(
        &mut self,
        initial_size_bytes: usize,
        maybe_image: Option<&Arc<MemoryMemFd>>,
    ) -> Result<(), InstantiationError> {
        assert!(!self.dirty);
        assert_eq!(self.cur_size, self.initial_size);

        // Fast-path: previously instantiated with the same image, or
        // no image but the same initial size, so the mappings are
        // already correct; there is no need to mmap anything. Given
        // that we asserted not-dirty above, any dirty pages will have
        // already been thrown away by madvise() during the previous
        // termination. The `clear_and_remain_ready()` path also
        // mprotects memory above the initial heap size back to
        // PROT_NONE, so we don't need to do that here.
        if (self.image.is_none()
            && maybe_image.is_none()
            && self.initial_size == initial_size_bytes)
            || (self.image.is_some()
                && maybe_image.is_some()
                && self.image.as_ref().unwrap().fd.as_file().as_raw_fd()
                    == maybe_image.as_ref().unwrap().fd.as_file().as_raw_fd())
        {
            self.dirty = true;
            return Ok(());
        }
        // Otherwise, we need to transition from the previous state to the
        // state now requested. An attempt is made here to minimize syscalls to
        // the kernel to ideally reduce the overhead of this as it's fairly
        // performance sensitive with memories. Note that the "previous state"
        // is assumed to be post-initialization (e.g. after an mmap on-demand
        // memory was created) or after `clear_and_remain_ready` was called
        // which notably means that `madvise` has reset all the memory back to
        // its original state.
        //
        // Security/audit note: we map all of these MAP_PRIVATE, so
        // all instance data is local to the mapping, not propagated
        // to the backing fd. We throw away this CoW overlay with
        // madvise() below, from base up to static_size (which is the
        // whole slot) when terminating the instance.

        if self.image.is_some() {
            // In this case the state of memory at this time is that the memory
            // from `0..self.initial_size` is reset back to its original state,
            // but this memory contians a CoW image that is different from the
            // one specified here. To reset state we first reset the mapping
            // of memory to anonymous PROT_NONE memory, and then afterwards the
            // heap is made visible with an mprotect.
            self.reset_with_anon_memory()
                .map_err(|e| InstantiationError::Resource(e.into()))?;
            self.set_protection(0..initial_size_bytes, region::Protection::READ_WRITE)
                .map_err(|e| InstantiationError::Resource(e.into()))?;
        } else if initial_size_bytes < self.initial_size {
            // In this case the previous module had now CoW image which means
            // that the memory at `0..self.initial_size` is all zeros and
            // read-write, everything afterwards being PROT_NONE.
            //
            // Our requested heap size is smaller than the previous heap size
            // so all that's needed now is to shrink the heap further to
            // `initial_size_bytes`.
            //
            // So we come in with:
            // - anon-zero memory, R+W,  [0, self.initial_size)
            // - anon-zero memory, none, [self.initial_size, self.static_size)
            // and we want:
            // - anon-zero memory, R+W,  [0, initial_size_bytes)
            // - anon-zero memory, none, [initial_size_bytes, self.static_size)
            //
            // so given initial_size_bytes < self.initial_size we
            // mprotect(NONE) the zone from the first to the second.
            self.set_protection(
                initial_size_bytes..self.initial_size,
                region::Protection::NONE,
            )
            .map_err(|e| InstantiationError::Resource(e.into()))?;
        } else if initial_size_bytes > self.initial_size {
            // In this case, like the previous one, the previous module had no
            // CoW image but had a smaller heap than desired for this module.
            // That means that here `mprotect` is used to make the new pages
            // read/write, and since they're all reset from before they'll be
            // made visible as zeros.
            self.set_protection(
                self.initial_size..initial_size_bytes,
                region::Protection::READ_WRITE,
            )
            .map_err(|e| InstantiationError::Resource(e.into()))?;
        } else {
            // The final case here is that the previous module has no CoW image
            // so the previous heap is all zeros. The previous heap is the exact
            // same size as the requested heap, so no syscalls are needed to do
            // anything else.
        }

        // The memory image, at this point, should have `initial_size_bytes` of
        // zeros starting at `self.base` followed by inaccessible memory to
        // `self.static_size`. Update sizing fields to reflect this.
        self.initial_size = initial_size_bytes;
        self.cur_size = initial_size_bytes;

        // The initial memory image, if given. If not, we just get a
        // memory filled with zeroes.
        if let Some(image) = maybe_image.as_ref() {
            assert!(
                image.linear_memory_offset.checked_add(image.len).unwrap() <= initial_size_bytes
            );
            if image.len > 0 {
                unsafe {
                    let ptr = rustix::io::mmap(
                        (self.base + image.linear_memory_offset) as *mut c_void,
                        image.len,
                        rustix::io::ProtFlags::READ | rustix::io::ProtFlags::WRITE,
                        rustix::io::MapFlags::PRIVATE | rustix::io::MapFlags::FIXED,
                        image.fd.as_file(),
                        image.fd_offset,
                    )
                    .map_err(|e| InstantiationError::Resource(e.into()))?;
                    assert_eq!(ptr as usize, self.base + image.linear_memory_offset);
                }
            }
        }

        self.image = maybe_image.cloned();
        self.dirty = true;

        Ok(())
    }

    #[allow(dead_code)] // ignore warnings as this is only used in some cfgs
    pub(crate) fn clear_and_remain_ready(&mut self) -> Result<()> {
        assert!(self.dirty);

        cfg_if::cfg_if! {
            if #[cfg(target_os = "linux")] {
                // On Linux we can use `madvise` to reset the virtual memory
                // back to its original state. This means back to all zeros for
                // anonymous-backed pages and back to the original contents for
                // CoW memory (the initial heap image). This has the precise
                // semantics we want for reuse between instances, so it's all we
                // need to do.
                unsafe {
                    rustix::io::madvise(
                        self.base as *mut c_void,
                        self.cur_size,
                        rustix::io::Advice::LinuxDontNeed,
                    )?;
                }
            } else {
                // If we're not on Linux, however, then there's no generic
                // platform way to reset memory back to its original state, so
                // instead this is "feigned" by resetting memory back to
                // entirely zeros with an anonymous backing.
                //
                // Additionally the previous image, if any, is dropped here
                // since it's no longer applicable to this mapping.
                self.reset_with_anon_memory()?;
                self.image = None;
            }
        }

        // mprotect the initial heap region beyond the initial heap size back to PROT_NONE.
        self.set_protection(self.initial_size..self.cur_size, region::Protection::NONE)?;
        self.cur_size = self.initial_size;
        self.dirty = false;
        Ok(())
    }

    fn set_protection(&self, range: Range<usize>, flags: region::Protection) -> Result<()> {
        assert!(range.start <= range.end);
        assert!(range.end <= self.static_size);
        let mprotect_start = self.base.checked_add(range.start).unwrap();
        if range.len() > 0 {
            unsafe {
                region::protect(mprotect_start as *mut _, range.len(), flags)?;
            }
        }

        Ok(())
    }

    pub(crate) fn has_image(&self) -> bool {
        self.image.is_some()
    }

    #[allow(dead_code)] // ignore warnings as this is only used in some cfgs
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Map anonymous zeroed memory across the whole slot,
    /// inaccessible. Used both during instantiate and during drop.
    fn reset_with_anon_memory(&self) -> Result<()> {
        unsafe {
            let ptr = rustix::io::mmap_anonymous(
                self.base as *mut c_void,
                self.static_size,
                rustix::io::ProtFlags::empty(),
                rustix::io::MapFlags::PRIVATE | rustix::io::MapFlags::FIXED,
            )?;
            assert_eq!(ptr as usize, self.base);
        }
        Ok(())
    }
}

impl Drop for MemFdSlot {
    fn drop(&mut self) {
        // The MemFdSlot may be dropped if there is an error during
        // instantiation: for example, if a memory-growth limiter
        // disallows a guest from having a memory of a certain size,
        // after we've already initialized the MemFdSlot.
        //
        // We need to return this region of the large pool mmap to a
        // safe state (with no module-specific mappings). The
        // MemFdSlot will not be returned to the MemoryPool, so a new
        // MemFdSlot will be created and overwrite the mappings anyway
        // on the slot's next use; but for safety and to avoid
        // resource leaks it's better not to have stale mappings to a
        // possibly-otherwise-dead module's image.
        //
        // To "wipe the slate clean", let's do a mmap of anonymous
        // memory over the whole region, with PROT_NONE. Note that we
        // *can't* simply munmap, because that leaves a hole in the
        // middle of the pooling allocator's big memory area that some
        // other random mmap may swoop in and take, to be trampled
        // over by the next MemFdSlot later.
        //
        // Since we're in drop(), we can't sanely return an error if
        // this mmap fails. Let's ignore the failure if so; the next
        // MemFdSlot to be created for this slot will try to overwrite
        // the existing stale mappings, and return a failure properly
        // if we still cannot map new memory.
        //
        // The exception to all of this is if the `unmap_on_drop` flag
        // (which is set by default) is false. If so, the owner of
        // this MemFdSlot has indicated that it will clean up in some
        // other way.
        if self.clear_on_drop {
            let _ = self.reset_with_anon_memory();
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod test {
    use std::sync::Arc;

    use super::create_memfd;
    use super::MemFdSlot;
    use super::MemoryMemFd;
    use crate::mmap::Mmap;
    use anyhow::Result;
    use std::io::Write;

    fn create_memfd_with_data(offset: usize, data: &[u8]) -> Result<MemoryMemFd> {
        // Offset must be page-aligned.
        let page_size = region::page::size();
        assert_eq!(offset & (page_size - 1), 0);
        let memfd = create_memfd()?;
        memfd.as_file().write_all(data)?;

        // The image length is rounded up to the nearest page size
        let image_len = (data.len() + page_size - 1) & !(page_size - 1);
        memfd.as_file().set_len(image_len as u64)?;

        Ok(MemoryMemFd {
            fd: memfd,
            len: image_len,
            offset,
        })
    }

    #[test]
    fn instantiate_no_image() {
        // 4 MiB mmap'd area, not accessible
        let mut mmap = Mmap::accessible_reserved(0, 4 << 20).unwrap();
        // Create a MemFdSlot on top of it
        let mut memfd = MemFdSlot::create(mmap.as_mut_ptr() as *mut _, 0, 4 << 20);
        memfd.no_clear_on_drop();
        assert!(!memfd.is_dirty());
        // instantiate with 64 KiB initial size
        memfd.instantiate(64 << 10, None).unwrap();
        assert!(memfd.is_dirty());
        // We should be able to access this 64 KiB (try both ends) and
        // it should consist of zeroes.
        let slice = mmap.as_mut_slice();
        assert_eq!(0, slice[0]);
        assert_eq!(0, slice[65535]);
        slice[1024] = 42;
        assert_eq!(42, slice[1024]);
        // grow the heap
        memfd.set_heap_limit(128 << 10).unwrap();
        let slice = mmap.as_slice();
        assert_eq!(42, slice[1024]);
        assert_eq!(0, slice[131071]);
        // instantiate again; we should see zeroes, even as the
        // reuse-anon-mmap-opt kicks in
        memfd.clear_and_remain_ready().unwrap();
        assert!(!memfd.is_dirty());
        memfd.instantiate(64 << 10, None).unwrap();
        let slice = mmap.as_slice();
        assert_eq!(0, slice[1024]);
    }

    #[test]
    fn instantiate_image() {
        // 4 MiB mmap'd area, not accessible
        let mut mmap = Mmap::accessible_reserved(0, 4 << 20).unwrap();
        // Create a MemFdSlot on top of it
        let mut memfd = MemFdSlot::create(mmap.as_mut_ptr() as *mut _, 0, 4 << 20);
        memfd.no_clear_on_drop();
        // Create an image with some data.
        let image = Arc::new(create_memfd_with_data(4096, &[1, 2, 3, 4]).unwrap());
        // Instantiate with this image
        memfd.instantiate(64 << 10, Some(&image)).unwrap();
        assert!(memfd.has_image());
        let slice = mmap.as_mut_slice();
        assert_eq!(&[1, 2, 3, 4], &slice[4096..4100]);
        slice[4096] = 5;
        // Clear and re-instantiate same image
        memfd.clear_and_remain_ready().unwrap();
        memfd.instantiate(64 << 10, Some(&image)).unwrap();
        let slice = mmap.as_slice();
        // Should not see mutation from above
        assert_eq!(&[1, 2, 3, 4], &slice[4096..4100]);
        // Clear and re-instantiate no image
        memfd.clear_and_remain_ready().unwrap();
        memfd.instantiate(64 << 10, None).unwrap();
        assert!(!memfd.has_image());
        let slice = mmap.as_slice();
        assert_eq!(&[0, 0, 0, 0], &slice[4096..4100]);
        // Clear and re-instantiate image again
        memfd.clear_and_remain_ready().unwrap();
        memfd.instantiate(64 << 10, Some(&image)).unwrap();
        let slice = mmap.as_slice();
        assert_eq!(&[1, 2, 3, 4], &slice[4096..4100]);
        // Create another image with different data.
        let image2 = Arc::new(create_memfd_with_data(4096, &[10, 11, 12, 13]).unwrap());
        memfd.clear_and_remain_ready().unwrap();
        memfd.instantiate(128 << 10, Some(&image2)).unwrap();
        let slice = mmap.as_slice();
        assert_eq!(&[10, 11, 12, 13], &slice[4096..4100]);
        // Instantiate the original image again; we should notice it's
        // a different image and not reuse the mappings.
        memfd.clear_and_remain_ready().unwrap();
        memfd.instantiate(64 << 10, Some(&image)).unwrap();
        let slice = mmap.as_slice();
        assert_eq!(&[1, 2, 3, 4], &slice[4096..4100]);
    }
}
