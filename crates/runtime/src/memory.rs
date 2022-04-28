//! Memory management for linear memories.
//!
//! `RuntimeLinearMemory` is to WebAssembly linear memories what `Table` is to WebAssembly tables.

use crate::mmap::Mmap;
use crate::vmcontext::VMMemoryDefinition;
use crate::MemoryImage;
use crate::MemoryImageSlot;
use crate::Store;
use anyhow::Error;
use anyhow::{bail, format_err, Result};
use more_asserts::{assert_ge, assert_le};
use std::convert::TryFrom;
use std::sync::Arc;
use wasmtime_environ::{MemoryPlan, MemoryStyle, WASM32_MAX_PAGES, WASM64_MAX_PAGES};

const WASM_PAGE_SIZE: usize = wasmtime_environ::WASM_PAGE_SIZE as usize;
const WASM_PAGE_SIZE_U64: u64 = wasmtime_environ::WASM_PAGE_SIZE as u64;

/// A memory allocator
pub trait RuntimeMemoryCreator: Send + Sync {
    /// Create new RuntimeLinearMemory
    fn new_memory(
        &self,
        plan: &MemoryPlan,
        minimum: usize,
        maximum: Option<usize>,
        // Optionally, a memory image for CoW backing.
        memory_image: Option<&Arc<MemoryImage>>,
    ) -> Result<Box<dyn RuntimeLinearMemory>>;
}

/// A default memory allocator used by Wasmtime
pub struct DefaultMemoryCreator;

impl RuntimeMemoryCreator for DefaultMemoryCreator {
    /// Create new MmapMemory
    fn new_memory(
        &self,
        plan: &MemoryPlan,
        minimum: usize,
        maximum: Option<usize>,
        memory_image: Option<&Arc<MemoryImage>>,
    ) -> Result<Box<dyn RuntimeLinearMemory>> {
        Ok(Box::new(MmapMemory::new(
            plan,
            minimum,
            maximum,
            memory_image,
        )?))
    }
}

/// A linear memory
pub trait RuntimeLinearMemory: Send + Sync {
    /// Returns the number of allocated bytes.
    fn byte_size(&self) -> usize;

    /// Returns the maximum number of bytes the memory can grow to.
    /// Returns `None` if the memory is unbounded.
    fn maximum_byte_size(&self) -> Option<usize>;

    /// Grow memory to the specified amount of bytes.
    ///
    /// Returns an error if memory can't be grown by the specified amount
    /// of bytes.
    fn grow_to(&mut self, size: usize) -> Result<()>;

    /// Return a `VMMemoryDefinition` for exposing the memory to compiled wasm
    /// code.
    fn vmmemory(&mut self) -> VMMemoryDefinition;

    /// Does this memory need initialization? It may not if it already
    /// has initial contents courtesy of the `MemoryImage` passed to
    /// `RuntimeMemoryCreator::new_memory()`.
    fn needs_init(&self) -> bool;

    /// For the pooling allocator, we must be able to downcast this trait to its
    /// underlying structure; this trampoline function allows us to use
    /// `Box::downcast` which only takes a `Box<dyn Any>`.
    #[cfg(feature = "pooling-allocator")]
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any>;
}

/// A linear memory instance.
#[derive(Debug)]
pub struct MmapMemory {
    // The underlying allocation.
    mmap: Mmap,

    // The number of bytes that are accessible in `mmap` and available for
    // reading and writing.
    //
    // This region starts at `pre_guard_size` offset from the base of `mmap`.
    accessible: usize,

    // The optional maximum accessible size, in bytes, for this linear memory.
    //
    // Note that this maximum does not factor in guard pages, so this isn't the
    // maximum size of the linear address space reservation for this memory.
    maximum: Option<usize>,

    // The amount of extra bytes to reserve whenever memory grows. This is
    // specified so that the cost of repeated growth is amortized.
    extra_to_reserve_on_growth: usize,

    // Size in bytes of extra guard pages before the start and after the end to
    // optimize loads and stores with constant offsets.
    pre_guard_size: usize,
    offset_guard_size: usize,

    // An optional CoW mapping that provides the initial content of this
    // MmapMemory, if mapped.
    memory_image: Option<MemoryImageSlot>,
}

impl MmapMemory {
    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages.
    pub fn new(
        plan: &MemoryPlan,
        minimum: usize,
        mut maximum: Option<usize>,
        memory_image: Option<&Arc<MemoryImage>>,
    ) -> Result<Self> {
        // It's a programmer error for these two configuration values to exceed
        // the host available address space, so panic if such a configuration is
        // found (mostly an issue for hypothetical 32-bit hosts).
        let offset_guard_bytes = usize::try_from(plan.offset_guard_size).unwrap();
        let pre_guard_bytes = usize::try_from(plan.pre_guard_size).unwrap();

        let (alloc_bytes, extra_to_reserve_on_growth) = match plan.style {
            // Dynamic memories start with the minimum size plus the `reserve`
            // amount specified to grow into.
            MemoryStyle::Dynamic { reserve } => (minimum, usize::try_from(reserve).unwrap()),

            // Static memories will never move in memory and consequently get
            // their entire allocation up-front with no extra room to grow into.
            // Note that the `maximum` is adjusted here to whatever the smaller
            // of the two is, the `maximum` given or the `bound` specified for
            // this memory.
            MemoryStyle::Static { bound } => {
                assert_ge!(bound, plan.memory.minimum);
                let bound_bytes =
                    usize::try_from(bound.checked_mul(WASM_PAGE_SIZE_U64).unwrap()).unwrap();
                maximum = Some(bound_bytes.min(maximum.unwrap_or(usize::MAX)));
                (bound_bytes, 0)
            }
        };
        let request_bytes = pre_guard_bytes
            .checked_add(alloc_bytes)
            .and_then(|i| i.checked_add(extra_to_reserve_on_growth))
            .and_then(|i| i.checked_add(offset_guard_bytes))
            .ok_or_else(|| format_err!("cannot allocate {} with guard regions", minimum))?;

        let mut mmap = Mmap::accessible_reserved(0, request_bytes)?;
        if minimum > 0 {
            mmap.make_accessible(pre_guard_bytes, minimum)?;
        }

        // If a memory image was specified, try to create the MemoryImageSlot on
        // top of our mmap.
        let memory_image = match memory_image {
            Some(image) => {
                let base = unsafe { mmap.as_mut_ptr().add(pre_guard_bytes) };
                let mut slot = MemoryImageSlot::create(
                    base.cast(),
                    minimum,
                    alloc_bytes + extra_to_reserve_on_growth,
                );
                slot.instantiate(minimum, Some(image))?;
                // On drop, we will unmap our mmap'd range that this slot was
                // mapped on top of, so there is no need for the slot to wipe
                // it with an anonymous mapping first.
                slot.no_clear_on_drop();
                Some(slot)
            }
            None => None,
        };

        Ok(Self {
            mmap,
            accessible: minimum,
            maximum,
            pre_guard_size: pre_guard_bytes,
            offset_guard_size: offset_guard_bytes,
            extra_to_reserve_on_growth,
            memory_image,
        })
    }
}

impl RuntimeLinearMemory for MmapMemory {
    fn byte_size(&self) -> usize {
        self.accessible
    }

    fn maximum_byte_size(&self) -> Option<usize> {
        self.maximum
    }

    fn grow_to(&mut self, new_size: usize) -> Result<()> {
        if new_size > self.mmap.len() - self.offset_guard_size - self.pre_guard_size {
            // If the new size of this heap exceeds the current size of the
            // allocation we have, then this must be a dynamic heap. Use
            // `new_size` to calculate a new size of an allocation, allocate it,
            // and then copy over the memory from before.
            let request_bytes = self
                .pre_guard_size
                .checked_add(new_size)
                .and_then(|s| s.checked_add(self.extra_to_reserve_on_growth))
                .and_then(|s| s.checked_add(self.offset_guard_size))
                .ok_or_else(|| format_err!("overflow calculating size of memory allocation"))?;

            let mut new_mmap = Mmap::accessible_reserved(0, request_bytes)?;
            new_mmap.make_accessible(self.pre_guard_size, new_size)?;

            new_mmap.as_mut_slice()[self.pre_guard_size..][..self.accessible]
                .copy_from_slice(&self.mmap.as_slice()[self.pre_guard_size..][..self.accessible]);

            // Now drop the MemoryImageSlot, if any. We've lost the CoW
            // advantages by explicitly copying all data, but we have
            // preserved all of its content; so we no longer need the
            // mapping. We need to do this before we (implicitly) drop the
            // `mmap` field by overwriting it below.
            drop(self.memory_image.take());

            self.mmap = new_mmap;
        } else if let Some(image) = self.memory_image.as_mut() {
            // MemoryImageSlot has its own growth mechanisms; defer to its
            // implementation.
            image.set_heap_limit(new_size)?;
        } else {
            // If the new size of this heap fits within the existing allocation
            // then all we need to do is to make the new pages accessible. This
            // can happen either for "static" heaps which always hit this case,
            // or "dynamic" heaps which have some space reserved after the
            // initial allocation to grow into before the heap is moved in
            // memory.
            assert!(new_size > self.accessible);
            self.mmap.make_accessible(
                self.pre_guard_size + self.accessible,
                new_size - self.accessible,
            )?;
        }

        self.accessible = new_size;

        Ok(())
    }

    fn vmmemory(&mut self) -> VMMemoryDefinition {
        VMMemoryDefinition {
            base: unsafe { self.mmap.as_mut_ptr().add(self.pre_guard_size) },
            current_length: self.accessible,
        }
    }

    fn needs_init(&self) -> bool {
        // If we're using a CoW mapping, then no initialization
        // is needed.
        self.memory_image.is_none()
    }

    #[cfg(feature = "pooling-allocator")]
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// A "static" memory where the lifetime of the backing memory is managed
/// elsewhere. Currently used with the pooling allocator.
struct ExternalMemory {
    /// The memory in the host for this wasm memory. The length of this
    /// slice is the maximum size of the memory that can be grown to.
    base: &'static mut [u8],

    /// The current size, in bytes, of this memory.
    size: usize,

    /// A callback which makes portions of `base` accessible for when memory
    /// is grown. Otherwise it's expected that accesses to `base` will
    /// fault.
    make_accessible: Option<fn(*mut u8, usize) -> Result<()>>,

    /// The image management, if any, for this memory. Owned here and
    /// returned to the pooling allocator when termination occurs.
    memory_image: Option<MemoryImageSlot>,
}

impl ExternalMemory {
    fn new(
        base: &'static mut [u8],
        initial_size: usize,
        maximum_size: Option<usize>,
        make_accessible: Option<fn(*mut u8, usize) -> Result<()>>,
        memory_image: Option<MemoryImageSlot>,
    ) -> Result<Self> {
        if base.len() < initial_size {
            bail!(
                "initial memory size of {} exceeds the pooling allocator's \
                 configured maximum memory size of {} bytes",
                initial_size,
                base.len(),
            );
        }

        // Only use the part of the slice that is necessary.
        let base = match maximum_size {
            Some(max) if max < base.len() => &mut base[..max],
            _ => base,
        };

        if let Some(make_accessible) = make_accessible {
            if initial_size > 0 {
                make_accessible(base.as_mut_ptr(), initial_size)?;
            }
        }

        Ok(Self {
            base,
            size: initial_size,
            make_accessible,
            memory_image,
        })
    }
}

impl Default for ExternalMemory {
    fn default() -> Self {
        Self {
            base: &mut [],
            size: 0,
            make_accessible: Some(|_, _| unreachable!()),
            memory_image: None,
        }
    }
}

impl RuntimeLinearMemory for ExternalMemory {
    fn byte_size(&self) -> usize {
        self.size
    }

    fn maximum_byte_size(&self) -> Option<usize> {
        Some(self.base.len())
    }

    fn grow_to(&mut self, new_byte_size: usize) -> Result<()> {
        // Never exceed the static memory size; this check should have been made
        // prior to arriving here.
        assert!(new_byte_size > self.base.len());

        // Actually grow the memory.
        if let Some(image) = &mut self.memory_image {
            image.set_heap_limit(new_byte_size)?;
        } else {
            let make_accessible = self
                .make_accessible
                .expect("make_accessible must be Some if this is not a CoW memory");

            // Operating system can fail to make memory accessible.
            let old_byte_size = self.byte_size();
            make_accessible(
                unsafe { self.base.as_mut_ptr().add(old_byte_size) },
                new_byte_size - old_byte_size,
            )?;
        }

        // Update our accounting of the available size.
        self.size = new_byte_size;
        Ok(())
    }

    fn vmmemory(&mut self) -> VMMemoryDefinition {
        VMMemoryDefinition {
            base: self.base.as_mut_ptr().cast(),
            current_length: self.size,
        }
    }

    fn needs_init(&self) -> bool {
        if let Some(slot) = &self.memory_image {
            !slot.has_image()
        } else {
            true
        }
    }

    #[cfg(feature = "pooling-allocator")]
    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any> {
        self
    }
}

/// Representation of a runtime wasm linear memory.
pub struct Memory(Box<dyn RuntimeLinearMemory>);

impl Memory {
    /// Create a new dynamic (movable) memory instance for the specified plan.
    pub fn new_dynamic(
        plan: &MemoryPlan,
        creator: &dyn RuntimeMemoryCreator,
        store: &mut dyn Store,
        memory_image: Option<&Arc<MemoryImage>>,
    ) -> Result<Self> {
        let (minimum, maximum) = Self::limit_new(plan, store)?;
        Ok(Memory(creator.new_memory(
            plan,
            minimum,
            maximum,
            memory_image,
        )?))
    }

    /// Create a new static (immovable) memory instance for the specified plan.
    pub fn new_static(
        plan: &MemoryPlan,
        base: &'static mut [u8],
        make_accessible: Option<fn(*mut u8, usize) -> Result<()>>,
        memory_image: Option<MemoryImageSlot>,
        store: &mut dyn Store,
    ) -> Result<Self> {
        let (minimum, maximum) = Self::limit_new(plan, store)?;
        let pooled_memory =
            ExternalMemory::new(base, minimum, maximum, make_accessible, memory_image)?;
        Ok(Memory(Box::new(pooled_memory)))
    }

    /// Calls the `store`'s limiter to optionally prevent a memory from being allocated.
    ///
    /// Returns the minimum size and optional maximum size of the memory, in
    /// bytes.
    fn limit_new(plan: &MemoryPlan, store: &mut dyn Store) -> Result<(usize, Option<usize>)> {
        // Sanity-check what should already be true from wasm module validation.
        let absolute_max = if plan.memory.memory64 {
            WASM64_MAX_PAGES
        } else {
            WASM32_MAX_PAGES
        };
        assert_le!(plan.memory.minimum, absolute_max);
        assert!(plan.memory.maximum.is_none() || plan.memory.maximum.unwrap() <= absolute_max);

        // This is the absolute possible maximum that the module can try to
        // allocate, which is our entire address space minus a wasm page. That
        // shouldn't ever actually work in terms of an allocation because
        // presumably the kernel wants *something* for itself, but this is used
        // to pass to the `store`'s limiter for a requested size
        // to approximate the scale of the request that the wasm module is
        // making. This is necessary because the limiter works on `usize` bytes
        // whereas we're working with possibly-overflowing `u64` calculations
        // here. To actually faithfully represent the byte requests of modules
        // we'd have to represent things as `u128`, but that's kinda
        // overkill for this purpose.
        let absolute_max = 0usize.wrapping_sub(WASM_PAGE_SIZE);

        // If the minimum memory size overflows the size of our own address
        // space, then we can't satisfy this request, but defer the error to
        // later so the `store` can be informed that an effective oom is
        // happening.
        let minimum = plan
            .memory
            .minimum
            .checked_mul(WASM_PAGE_SIZE_U64)
            .and_then(|m| usize::try_from(m).ok());

        // The plan stores the maximum size in units of wasm pages, but we
        // use units of bytes. Unlike for the `minimum` size we silently clamp
        // the effective maximum size to `absolute_max` above if the maximum is
        // too large. This should be ok since as a wasm runtime we get to
        // arbitrarily decide the actual maximum size of memory, regardless of
        // what's actually listed on the memory itself.
        let mut maximum = plan.memory.maximum.map(|max| {
            usize::try_from(max)
                .ok()
                .and_then(|m| m.checked_mul(WASM_PAGE_SIZE))
                .unwrap_or(absolute_max)
        });

        // If this is a 32-bit memory and no maximum is otherwise listed then we
        // need to still specify a maximum size of 4GB. If the host platform is
        // 32-bit then there's no need to limit the maximum this way since no
        // allocation of 4GB can succeed, but for 64-bit platforms this is
        // required to limit memories to 4GB.
        if !plan.memory.memory64 && maximum.is_none() {
            maximum = usize::try_from(1u64 << 32).ok();
        }

        // Inform the store's limiter what's about to happen. This will let the limiter
        // reject anything if necessary, and this also guarantees that we should
        // call the limiter for all requested memories, even if our `minimum`
        // calculation overflowed. This means that the `minimum` we're informing
        // the limiter is lossy and may not be 100% accurate, but for now the
        // expected uses of limiter means that's ok.
        if !store.memory_growing(0, minimum.unwrap_or(absolute_max), maximum)? {
            bail!(
                "memory minimum size of {} pages exceeds memory limits",
                plan.memory.minimum
            );
        }

        // At this point we need to actually handle overflows, so bail out with
        // an error if we made it this far.
        let minimum = minimum.ok_or_else(|| {
            format_err!(
                "memory minimum size of {} pages exceeds memory limits",
                plan.memory.minimum
            )
        })?;
        Ok((minimum, maximum))
    }

    /// Returns the number of allocated wasm pages.
    pub fn byte_size(&self) -> usize {
        self.0.byte_size()
    }

    /// Returns the maximum number of pages the memory can grow to at runtime.
    ///
    /// Returns `None` if the memory is unbounded.
    ///
    /// The runtime maximum may not be equal to the maximum from the linear memory's
    /// Wasm type when it is being constrained by an instance allocator.
    pub fn maximum_byte_size(&self) -> Option<usize> {
        self.0.maximum_byte_size()
    }

    /// Returns whether or not this memory needs initialization. It
    /// may not if it already has initial content thanks to a CoW
    /// mechanism.
    pub(crate) fn needs_init(&self) -> bool {
        self.0.needs_init()
    }

    /// Grow memory by the specified amount of wasm pages.
    ///
    /// Returns `None` if memory can't be grown by the specified amount
    /// of wasm pages. Returns `Some` with the old size of memory, in bytes, on
    /// successful growth.
    ///
    /// # Safety
    ///
    /// Resizing the memory can reallocate the memory buffer for dynamic memories.
    /// An instance's `VMContext` may have pointers to the memory's base and will
    /// need to be fixed up after growing the memory.
    ///
    /// Generally, prefer using `InstanceHandle::memory_grow`, which encapsulates
    /// this unsafety.
    ///
    /// Ensure that the provided Store is not used to get access any Memory
    /// which lives inside it.
    pub unsafe fn grow(
        &mut self,
        delta_pages: u64,
        store: &mut dyn Store,
    ) -> Result<Option<usize>, Error> {
        let old_byte_size = self.byte_size();

        // Wasm spec: when growing by 0 pages, always return the current size.
        if delta_pages == 0 {
            return Ok(Some(old_byte_size));
        }

        // largest wasm-page-aligned region of memory it is possible to
        // represent in a usize. This will be impossible for the system to
        // actually allocate.
        let absolute_max = 0usize.wrapping_sub(WASM_PAGE_SIZE);
        // calculate byte size of the new allocation. Let it overflow up to
        // usize::MAX, then clamp it down to absolute_max.
        let new_byte_size = usize::try_from(delta_pages)
            .unwrap_or(usize::MAX)
            .saturating_mul(WASM_PAGE_SIZE)
            .saturating_add(old_byte_size);
        let new_byte_size = if new_byte_size > absolute_max {
            absolute_max
        } else {
            new_byte_size
        };

        let maximum = self.maximum_byte_size();
        // Store limiter gets first chance to reject memory_growing.
        if !store.memory_growing(old_byte_size, new_byte_size, maximum)? {
            return Ok(None);
        }

        // Never exceed maximum, even if limiter permitted it.
        if let Some(max) = maximum {
            if new_byte_size > max {
                store.memory_grow_failed(&format_err!("Memory maximum size exceeded"));
                return Ok(None);
            }
        }

        match self.0.grow_to(new_byte_size) {
            Ok(_) => Ok(Some(old_byte_size)),
            Err(e) => {
                store.memory_grow_failed(&e);
                Ok(None)
            }
        }
    }

    /// Return a `VMMemoryDefinition` for exposing the memory to compiled wasm code.
    pub fn vmmemory(&mut self) -> VMMemoryDefinition {
        self.0.vmmemory()
    }

    /// Check if the inner implementation of [`Memory`] is indeed an
    /// `ExternalMemory`.
    #[cfg(feature = "pooling-allocator")]
    pub fn is_external(&self) -> bool {
        let as_any = &self.0 as &dyn std::any::Any;
        as_any.downcast_ref::<ExternalMemory>().is_some()
    }

    /// Consume the memory, returning its [`MemoryImageSlot`] if any is present.
    /// This implicitly checks that the memory is an external memory.
    #[cfg(feature = "pooling-allocator")]
    pub fn unwrap_image_slot(self) -> Option<MemoryImageSlot> {
        let as_any = self.0.into_any();
        if let Ok(m) = as_any.downcast::<ExternalMemory>() {
            m.memory_image
        } else {
            None
        }
    }
}

// The default memory representation is an empty memory that cannot grow.
impl Default for Memory {
    fn default() -> Self {
        Memory(Box::new(ExternalMemory::default()))
    }
}
