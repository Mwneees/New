//! Vectors allocated in arenas, with small per-vector overhead.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Range;

/// A vector of `T` stored within a `BumpArena`.
///
/// This is something like a normal `Vec`, except that all accesses
/// and updates require a separate borrow of the `BumpArena`. This, in
/// turn, makes the Vec itself very compact: only three `u32`s (12
/// bytes).
///
/// The `BumpVec` does *not* implement `Clone` or `Copy`; it
/// represents unique ownership of a range of indices in the arena. If
/// dropped, those indices will be unavailable until the arena is
/// freed. This is "fine" (it is normally how arena allocation
/// works). To explicitly free and make available for single-slot
/// allocations, a very rudimentary reuse mechanism exists via
/// `BumpVec::free(arena)`.
///
/// The type `T` must not have a `Drop` implementation. This typically
/// means that it does not own any boxed memory, sub-collections, or
/// other resources. This is essential for the efficiency of the data
/// structure (otherwise the arena needs to track which indices are
/// live or dead; the BumpVec itself cannot do the drop because it
/// does not retain a reference to the arena).
#[derive(Debug)]
pub struct BumpVec<T> {
    base: u32,
    len: u32,
    cap: u32,
    _phantom: PhantomData<T>,
}

#[derive(Default)]
pub struct BumpArena<T> {
    vec: Vec<MaybeUninit<T>>,
    freelist: Vec<Range<u32>>,
}

impl<T> BumpArena<T> {
    pub fn new() -> Self {
        Self {
            vec: vec![],
            freelist: vec![],
        }
    }

    pub fn with_capacity(&mut self, cap: usize) -> BumpVec<T> {
        let cap = u32::try_from(cap).unwrap();
        if let Some(range) = self.maybe_freelist_alloc(cap) {
            BumpVec {
                base: range.start,
                len: 0,
                cap,
                _phantom: PhantomData,
            }
        } else {
            let base = self.vec.len() as u32;
            for _ in 0..cap {
                self.vec.push(MaybeUninit::uninit());
            }
            BumpVec {
                base,
                len: 0,
                cap,
                _phantom: PhantomData,
            }
        }
    }

    pub fn single(&mut self, t: T) -> BumpVec<T> {
        let mut vec = self.with_capacity(1);
        unsafe {
            self.write_into_index(vec.base, t);
        }
        vec.len = 1;
        vec
    }

    pub fn from_iter<I: Iterator<Item = T>>(&mut self, i: I) -> BumpVec<T> {
        let base = self.vec.len() as u32;
        self.vec.extend(i.map(|item| MaybeUninit::new(item)));
        let len = self.vec.len() as u32 - base;
        BumpVec {
            base,
            len,
            cap: len,
            _phantom: PhantomData,
        }
    }

    pub fn append(&mut self, a: BumpVec<T>, b: BumpVec<T>) -> BumpVec<T> {
        if (a.cap - a.len) >= b.len {
            self.append_into_cap(a, b)
        } else {
            self.append_into_new(a, b)
        }
    }

    unsafe fn read_out_of_index(&self, index: u32) -> T {
        self.vec[index as usize].as_ptr().read()
    }

    unsafe fn write_into_index(&mut self, index: u32, t: T) {
        self.vec[index as usize].as_mut_ptr().write(t);
    }

    unsafe fn move_item(&mut self, from: u32, to: u32) {
        let item = self.read_out_of_index(from);
        self.write_into_index(to, item);
    }

    unsafe fn push_item(&mut self, from: u32) -> u32 {
        let index = self.vec.len() as u32;
        let item = self.read_out_of_index(from);
        self.vec.push(MaybeUninit::new(item));
        index
    }

    fn append_into_cap(&mut self, mut a: BumpVec<T>, b: BumpVec<T>) -> BumpVec<T> {
        debug_assert!(a.cap - a.len >= b.len);
        for i in 0..b.len {
            // Safety: initially, the indices in `b` are initialized;
            // the indices in `a`'s cap, beyond its length, are
            // uninitialized. We move the initialized contents from
            // `b` to the tail beyond `a`, and we consume `b` (so it
            // no longer exists), and we update `a`'s length to cover
            // the initialized contents in their new location.
            unsafe {
                self.move_item(b.base + i, a.base + a.cap + i);
            }
        }
        a.len += b.len;
        b.free(self);
        a
    }

    fn maybe_freelist_alloc(&mut self, len: u32) -> Option<Range<u32>> {
        if let Some(entry) = self.freelist.last_mut() {
            if entry.len() >= len as usize {
                let base = entry.start;
                entry.start += len;
                if entry.start == entry.end {
                    self.freelist.pop();
                }
                return Some(base..(base + len));
            }
        }
        None
    }

    fn append_into_new(&mut self, a: BumpVec<T>, b: BumpVec<T>) -> BumpVec<T> {
        // New capacity: round up to a power of two.
        let len = a.len + b.len;
        let cap = round_up_power_of_two(len);

        if let Some(range) = self.maybe_freelist_alloc(cap) {
            for i in 0..a.len {
                // Safety: the indices in `a` must be initialized. We read
                // out the item and copy it to a new index; the old index
                // is no longer covered by a BumpVec, because we consume
                // `a`.
                unsafe {
                    self.move_item(a.base + i, range.start + i);
                }
            }
            for i in 0..b.len {
                // Safety: the indices in `b` must be initialized. We read
                // out the item and copy it to a new index; the old index
                // is no longer covered by a BumpVec, because we consume
                // `b`.
                unsafe {
                    self.move_item(b.base + i, range.start + a.len + i);
                }
            }

            a.free(self);
            b.free(self);

            BumpVec {
                base: range.start,
                len,
                cap,
                _phantom: PhantomData,
            }
        } else {
            self.vec.reserve(cap as usize);
            let base = self.vec.len() as u32;
            for i in 0..a.len {
                // Safety: the indices in `a` must be initialized. We read
                // out the item and copy it to a new index; the old index
                // is no longer covered by a BumpVec, because we consume
                // `a`.
                unsafe {
                    self.push_item(a.base + i);
                }
            }
            for i in 0..b.len {
                // Safety: the indices in `b` must be initialized. We read
                // out the item and copy it to a new index; the old index
                // is no longer covered by a BumpVec, because we consume
                // `b`.
                unsafe {
                    self.push_item(b.base + i);
                }
            }
            let len = self.vec.len() as u32 - base;

            for _ in len..cap {
                self.vec.push(MaybeUninit::uninit());
            }

            a.free(self);
            b.free(self);

            BumpVec {
                base,
                len,
                cap,
                _phantom: PhantomData,
            }
        }
    }

    /// Returns the size of the backing `Vec`.
    pub fn size(&self) -> usize {
        self.vec.len()
    }
}

fn round_up_power_of_two(x: u32) -> u32 {
    debug_assert!(x > 0);
    debug_assert!(x < 0x8000_0000);
    let log2 = 32 - (x - 1).leading_zeros();
    1 << log2
}

impl<T> BumpVec<T> {
    pub fn as_slice<'a>(&'a self, arena: &'a BumpArena<T>) -> &'a [T] {
        let maybe_uninit_slice =
            &arena.vec[(self.base as usize)..((self.base + self.len) as usize)];
        // Safety: the index range we represent must be initialized.
        unsafe { std::mem::transmute(maybe_uninit_slice) }
    }

    pub fn as_mut_slice<'a>(&'a mut self, arena: &'a mut BumpArena<T>) -> &'a mut [T] {
        let maybe_uninit_slice =
            &mut arena.vec[(self.base as usize)..((self.base + self.len) as usize)];
        // Safety: the index range we represent must be initialized.
        unsafe { std::mem::transmute(maybe_uninit_slice) }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn cap(&self) -> usize {
        self.cap as usize
    }

    pub fn reserve(&mut self, extra_len: usize, arena: &mut BumpArena<T>) {
        let extra_len = u32::try_from(extra_len).unwrap();
        if self.cap - self.len < extra_len {
            if self.base + self.cap == arena.vec.len() as u32 {
                for _ in 0..extra_len {
                    arena.vec.push(MaybeUninit::uninit());
                }
                self.cap += extra_len;
            } else {
                let new_cap = self.cap + extra_len;
                let new = arena.with_capacity(new_cap as usize);
                unsafe {
                    for i in 0..self.len {
                        arena.move_item(self.base + i, new.base + i);
                    }
                }
                self.base = new.base;
                self.cap = new.cap;
            }
        }
    }

    /// Push an item, growing the capacity if needed.
    pub fn push(&mut self, t: T, arena: &mut BumpArena<T>) {
        if self.cap > self.len {
            unsafe {
                arena.write_into_index(self.base + self.len, t);
            }
            self.len += 1;
        } else if (self.base + self.cap) as usize == arena.vec.len() {
            arena.vec.push(MaybeUninit::new(t));
            self.cap += 1;
            self.len += 1;
        } else {
            let new_cap = round_up_power_of_two(self.cap + 1);
            let extra = new_cap - self.cap;
            self.reserve(extra as usize, arena);
            unsafe {
                arena.write_into_index(self.base + self.len, t);
            }
            self.len += 1;
        }
    }

    /// Clone, if `T` is cloneable.
    pub fn clone(&self, arena: &mut BumpArena<T>) -> BumpVec<T>
    where
        T: Clone,
    {
        let mut new = arena.with_capacity(self.len as usize);
        for i in 0..self.len {
            let item = self.as_slice(arena)[i as usize].clone();
            new.push(item, arena);
        }
        new
    }

    /// Truncate the length to a smaller-or-equal length.
    pub fn truncate(&mut self, len: usize) {
        let len = len as u32;
        assert!(len <= self.len);
        self.len = len;
    }

    /// Consume the BumpVec and return its indices to a free pool in
    /// the arena.
    pub fn free(self, arena: &mut BumpArena<T>) {
        arena.freelist.push(self.base..(self.base + self.cap));
    }
}

impl<T> std::default::Default for BumpVec<T> {
    fn default() -> Self {
        BumpVec {
            base: 0,
            len: 0,
            cap: 0,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_round_up() {
        assert_eq!(1, round_up_power_of_two(1));
        assert_eq!(2, round_up_power_of_two(2));
        assert_eq!(4, round_up_power_of_two(3));
        assert_eq!(4, round_up_power_of_two(4));
        assert_eq!(32, round_up_power_of_two(24));
        assert_eq!(0x8000_0000, round_up_power_of_two(0x7fff_ffff));
    }

    #[test]
    fn test_basic() {
        let mut arena: BumpArena<u32> = BumpArena::new();

        let a = arena.single(1);
        let b = arena.single(2);
        let c = arena.single(3);
        let ab = arena.append(a, b);
        assert_eq!(ab.as_slice(&arena), &[1, 2]);
        assert_eq!(ab.cap(), 2);
        let abc = arena.append(ab, c);
        assert_eq!(abc.len(), 3);
        assert_eq!(abc.cap(), 4);
        assert_eq!(abc.as_slice(&arena), &[1, 2, 3]);
        assert_eq!(arena.size(), 9);
        let mut d = arena.single(4);
        // Should have reused the freelist.
        assert_eq!(arena.size(), 9);
        assert_eq!(d.len(), 1);
        assert_eq!(d.cap(), 1);
        assert_eq!(d.as_slice(&arena), &[4]);
        d.as_mut_slice(&mut arena)[0] = 5;
        assert_eq!(d.as_slice(&arena), &[5]);
        abc.free(&mut arena);
        let d2 = d.clone(&mut arena);
        let dd = arena.append(d, d2);
        // Should have reused the freelist.
        assert_eq!(arena.size(), 9);
        assert_eq!(dd.as_slice(&arena), &[5, 5]);
        let mut e = arena.from_iter([10, 11, 12].into_iter());
        e.push(13, &mut arena);
        assert_eq!(arena.size(), 13);
        e.reserve(4, &mut arena);
        assert_eq!(arena.size(), 17);
        let _f = arena.from_iter([1, 2, 3, 4, 5, 6, 7, 8].into_iter());
        assert_eq!(arena.size(), 25);
        e.reserve(8, &mut arena);
        assert_eq!(e.cap(), 16);
        assert_eq!(e.as_slice(&arena), &[10, 11, 12, 13]);
        // `e` must have been copied now that `f` is at the end of the
        // arena.
        assert_eq!(arena.size(), 41);
    }
}
