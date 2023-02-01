//! Index/slot allocator policies for the pooling allocator.

use crate::CompiledModuleId;
use std::collections::hash_map::{Entry, HashMap};
use std::mem;
use std::sync::Mutex;

/// A slot index. The job of this allocator is to hand out these
/// indices.
#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotId(pub u32);
impl SlotId {
    /// The index of this slot.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug)]
pub struct IndexAllocator(Mutex<Inner>);

#[derive(Debug)]
struct Inner {
    /// Maximum  number of "unused warm slots" which will be allowed during
    /// allocation.
    ///
    /// This is a user-configurable knob which can be used to influence the
    /// maximum number of unused slots at any one point in time. A "warm slot"
    /// is one that's considered having been previously allocated.
    max_unused_warm_slots: u32,

    /// Current count of "warm slots", or those that were previously allocated
    /// which are now no longer in use.
    ///
    /// This is the size of the `warm` list.
    unused_warm_slots: u32,

    /// A linked list (via indices) which enumerates all "warm and unused"
    /// slots, or those which have previously been allocated and then free'd.
    warm: List,

    /// Last slot that was allocated for the first time ever.
    ///
    /// This is initially 0 and is incremented during `pick_cold`. If this
    /// matches `max_cold`, there are no more cold slots left.
    last_cold: u32,

    /// The state of any given slot.
    ///
    /// Records indices in the above list (empty) or two lists (with affinity),
    /// and these indices are kept up-to-date to allow fast removal.
    slot_state: Vec<SlotState>,

    /// Affine slot management which tracks which slots are free and were last
    /// used with the specified `CompiledModuleId`.
    ///
    /// The `List` here is appended to during deallocation and removal happens
    /// from the tail during allocation.
    module_affine: HashMap<CompiledModuleId, List>,
}

/// A helper "linked list" data structure which is based on indices.
#[derive(Default, Debug)]
struct List {
    head: Option<SlotId>,
    tail: Option<SlotId>,
}

/// A helper data structure for an intrusive linked list, coupled with the
/// `List` type.
#[derive(Default, Debug, Copy, Clone)]
struct Link {
    prev: Option<SlotId>,
    next: Option<SlotId>,
}

#[derive(Clone, Debug)]
enum SlotState {
    /// This slot is currently in use and is affine to the specified module.
    Used(Option<CompiledModuleId>),

    /// This slot is not currently used, and has never been used.
    UnusedCold,

    /// This slot is not currently used, but was previously allocated.
    ///
    /// The payload here is metadata about the lists that this slot is contained
    /// within.
    UnusedWarm(Unused),
}

impl SlotState {
    fn unwrap_unused(&mut self) -> &mut Unused {
        match self {
            SlotState::UnusedWarm(u) => u,
            _ => unreachable!(),
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
struct Unused {
    /// Which module this slot was historically affine to, if any.
    affinity: Option<CompiledModuleId>,

    /// Metadata about the linked list for all slots affine to `affinity`.
    affine: Link,

    /// Metadata within the `warm` list of the main allocator.
    unused: Link,
}

enum AllocMode {
    ForceAffineAndClear,
    AnySlot,
}

impl IndexAllocator {
    /// Create the default state for this strategy.
    pub fn new(max_instances: u32, max_unused_warm_slots: u32) -> Self {
        IndexAllocator(Mutex::new(Inner {
            last_cold: 0,
            max_unused_warm_slots,
            unused_warm_slots: 0,
            module_affine: HashMap::new(),
            slot_state: (0..max_instances).map(|_| SlotState::UnusedCold).collect(),
            warm: List::default(),
        }))
    }

    /// Allocate a new index from this allocator optionally using `id` as an
    /// affinity request if the allocation strategy supports it.
    ///
    /// Returns `None` if no more slots are available.
    pub fn alloc(&self, module_id: Option<CompiledModuleId>) -> Option<SlotId> {
        self._alloc(module_id, AllocMode::AnySlot)
    }

    /// Attempts to allocate a guaranteed-affine slot to the module `id`
    /// specified.
    ///
    /// Returns `None` if there are no slots affine to `id`. The allocation of
    /// this slot will not record the affinity to `id`, instead simply listing
    /// it as taken. This is intended to be used for clearing out all affine
    /// slots to a module.
    pub fn alloc_affine_and_clear_affinity(&self, module_id: CompiledModuleId) -> Option<SlotId> {
        self._alloc(Some(module_id), AllocMode::ForceAffineAndClear)
    }

    fn _alloc(&self, module_id: Option<CompiledModuleId>, mode: AllocMode) -> Option<SlotId> {
        let mut inner = self.0.lock().unwrap();
        let inner = &mut *inner;

        // As a first-pass always attempt an affine allocation. This will
        // succeed if any slots are considered affine to `module_id` (if it's
        // specified). Failing that something else is attempted to be chosen.
        let slot_id = inner.pick_affine(module_id).or_else(|| {
            match mode {
                // If any slot is requested then this is a normal instantiation
                // looking for an index. Without any affine candidates there are
                // two options here:
                //
                // 1. Pick a slot amongst previously allocated slots
                // 2. Pick a slot that's never been used before
                //
                // The choice here is guided by the initial configuration of
                // `max_unused_warm_slots`. If our unused warm slots, which are
                // likely all affine, is below this threshold then the affinity
                // of the warm slots isn't tampered with and first a cold slot
                // is chosen. If the cold slot allocation fails, however, a warm
                // slot is evicted.
                //
                // The opposite happens when we're above our threshold for the
                // maximum number of warm slots, meaning that a warm slot is
                // attempted to be picked from first with a cold slot following
                // that.
                AllocMode::AnySlot => {
                    if inner.unused_warm_slots < inner.max_unused_warm_slots {
                        inner.pick_cold().or_else(|| inner.pick_warm())
                    } else {
                        inner.pick_warm().or_else(|| inner.pick_cold())
                    }
                }

                // In this mode an affinity-based allocation is always performed
                // as the purpose here is to clear out slots relevant to
                // `module_id` during module teardown. This means that there's
                // no consulting non-affine slots in this path.
                AllocMode::ForceAffineAndClear => None,
            }
        })?;

        inner.slot_state[slot_id.index()] = SlotState::Used(match mode {
            AllocMode::ForceAffineAndClear => None,
            AllocMode::AnySlot => module_id,
        });

        Some(slot_id)
    }

    pub(crate) fn free(&self, index: SlotId) {
        let mut inner = self.0.lock().unwrap();
        let inner = &mut *inner;
        let module = match inner.slot_state[index.index()] {
            SlotState::Used(module) => module,
            _ => unreachable!(),
        };

        // Bump the number of warm slots since this slot is now considered
        // previously used. Afterwards append it to the linked list of all
        // unused and warm slots.
        inner.unused_warm_slots += 1;
        let unused = inner
            .warm
            .append(index, &mut inner.slot_state, |s| &mut s.unused);

        let affine = match module {
            // If this slot is affine to a particular module then append this
            // index to the linked list for the affine module. Otherwise insert
            // a new one-element linked list.
            Some(module) => match inner.module_affine.entry(module) {
                Entry::Occupied(mut e) => e
                    .get_mut()
                    .append(index, &mut inner.slot_state, |s| &mut s.affine),
                Entry::Vacant(v) => {
                    v.insert(List::new(index));
                    Link::default()
                }
            },

            // If this slot has no affinity then the affine link is empty.
            None => Link::default(),
        };

        inner.slot_state[index.index()] = SlotState::UnusedWarm(Unused {
            affinity: module,
            affine,
            unused,
        });
    }

    /// For testing only, we want to be able to assert what is on the
    /// single freelist, for the policies that keep just one.
    #[cfg(test)]
    pub(crate) fn testing_freelist(&self) -> Vec<SlotId> {
        let inner = self.0.lock().unwrap();
        inner.warm.iter(&inner.slot_state, |s| &s.unused).collect()
    }

    /// For testing only, get the list of all modules with at least
    /// one slot with affinity for that module.
    #[cfg(test)]
    pub(crate) fn testing_module_affinity_list(&self) -> Vec<CompiledModuleId> {
        let inner = self.0.lock().unwrap();
        inner.module_affine.keys().copied().collect()
    }
}

impl Inner {
    /// Attempts to allocate a slot already affine to `id`, returning `None` if
    /// `id` is `None` or if there are no affine slots.
    fn pick_affine(&mut self, module_id: Option<CompiledModuleId>) -> Option<SlotId> {
        // Note that the `tail` is chosen here of the affine list as it's the
        // most recently used, which for affine allocations is what we want --
        // maximizing temporal reuse.
        let ret = self.module_affine.get(&module_id?)?.tail?;
        self.remove(ret);
        Some(ret)
    }

    fn pick_warm(&mut self) -> Option<SlotId> {
        // Insertions into the `unused` list happen at the `tail`, so the
        // least-recently-used item will be at the head. That's our goal here,
        // pick the least-recently-used slot since something "warm" is being
        // evicted anyway.
        let head = self.warm.head?;
        self.remove(head);
        Some(head)
    }

    fn remove(&mut self, slot: SlotId) {
        // Decrement the size of the warm list, and additionally remove it from
        // the `warm` linked list.
        self.unused_warm_slots -= 1;
        self.warm
            .remove(slot, &mut self.slot_state, |u| &mut u.unused);

        // If this slot is affine to a module then additionally remove it from
        // that module's affinity linked list. Note that if the module's affine
        // list is empty then the module's entry in the map is completely
        // removed as well.
        let module = self.slot_state[slot.index()].unwrap_unused().affinity;
        if let Some(module) = module {
            let mut list = match self.module_affine.entry(module) {
                Entry::Occupied(e) => e,
                Entry::Vacant(_) => unreachable!(),
            };
            list.get_mut()
                .remove(slot, &mut self.slot_state, |u| &mut u.affine);

            if list.get_mut().head.is_none() {
                list.remove();
            }
        }
    }

    fn pick_cold(&mut self) -> Option<SlotId> {
        if (self.last_cold as usize) == self.slot_state.len() {
            None
        } else {
            let ret = Some(SlotId(self.last_cold));
            self.last_cold += 1;
            ret
        }
    }
}

impl List {
    /// Creates a new one-element list pointing at `id`.
    fn new(id: SlotId) -> List {
        List {
            head: Some(id),
            tail: Some(id),
        }
    }

    /// Appends the `id` to this list whose links are determined by `link`.
    fn append(
        &mut self,
        id: SlotId,
        states: &mut [SlotState],
        link: fn(&mut Unused) -> &mut Link,
    ) -> Link {
        // This `id` is the new tail...
        let tail = mem::replace(&mut self.tail, Some(id));

        // If the tail was present, then update its `next` field to ourselves as
        // we've been appended, otherwise update the `head` since the list was
        // previously empty.
        match tail {
            Some(tail) => link(states[tail.index()].unwrap_unused()).next = Some(id),
            None => self.head = Some(id),
        }
        Link {
            prev: tail,
            next: None,
        }
    }

    /// Removes `id` from this list whose links are determined by `link`.
    fn remove(
        &mut self,
        id: SlotId,
        slot_state: &mut [SlotState],
        link: fn(&mut Unused) -> &mut Link,
    ) -> Unused {
        let mut state = *slot_state[id.index()].unwrap_unused();
        let next = link(&mut state).next;
        let prev = link(&mut state).prev;

        // If a `next` node is present for this link, then its previous was our
        // own previous now. Otherwise we are the tail so the new tail is our
        // previous.
        match next {
            Some(next) => link(slot_state[next.index()].unwrap_unused()).prev = prev,
            None => self.tail = prev,
        }

        // Same as the `next` node, except everything is in reverse.
        match prev {
            Some(prev) => link(slot_state[prev.index()].unwrap_unused()).next = next,
            None => self.head = next,
        }
        state
    }

    #[cfg(test)]
    fn iter<'a>(
        &self,
        states: &'a [SlotState],
        link: fn(&Unused) -> &Link,
    ) -> impl Iterator<Item = SlotId> + 'a {
        let mut cur = self.head;
        std::iter::from_fn(move || {
            let ret = cur?;
            match &states[ret.index()] {
                SlotState::UnusedWarm(u) => cur = link(u).next,
                _ => unreachable!(),
            }
            Some(ret)
        })
    }
}

#[cfg(test)]
mod test {
    use super::{IndexAllocator, SlotId};
    use crate::CompiledModuleIdAllocator;

    #[test]
    fn test_next_available_allocation_strategy() {
        for size in 0..20 {
            let state = IndexAllocator::new(size, 0);
            for i in 0..size {
                assert_eq!(state.alloc(None).unwrap().index(), i as usize);
            }
            assert!(state.alloc(None).is_none());
        }
    }

    #[test]
    fn test_affinity_allocation_strategy() {
        let id_alloc = CompiledModuleIdAllocator::new();
        let id1 = id_alloc.alloc();
        let id2 = id_alloc.alloc();
        let state = IndexAllocator::new(100, 100);

        let index1 = state.alloc(Some(id1)).unwrap();
        assert_eq!(index1.index(), 0);
        let index2 = state.alloc(Some(id2)).unwrap();
        assert_eq!(index2.index(), 1);
        assert_ne!(index1, index2);

        state.free(index1);
        let index3 = state.alloc(Some(id1)).unwrap();
        assert_eq!(index3, index1);
        state.free(index3);

        state.free(index2);

        // Both id1 and id2 should have some slots with affinity.
        let affinity_modules = state.testing_module_affinity_list();
        assert_eq!(2, affinity_modules.len());
        assert!(affinity_modules.contains(&id1));
        assert!(affinity_modules.contains(&id2));

        // Now there is 1 free instance for id2 and 1 free instance
        // for id1, and 98 empty. Allocate 100 for id2. The first
        // should be equal to the one we know was previously used for
        // id2. The next 99 are arbitrary.

        let mut indices = vec![];
        for _ in 0..100 {
            indices.push(state.alloc(Some(id2)).unwrap());
        }
        assert!(state.alloc(None).is_none());
        assert_eq!(indices[0], index2);

        for i in indices {
            state.free(i);
        }

        // Now there should be no slots left with affinity for id1.
        let affinity_modules = state.testing_module_affinity_list();
        assert_eq!(1, affinity_modules.len());
        assert!(affinity_modules.contains(&id2));

        // Allocate an index we know previously had an instance but
        // now does not (list ran empty).
        let index = state.alloc(Some(id1)).unwrap();
        state.free(index);
    }

    #[test]
    fn clear_affine() {
        let id_alloc = CompiledModuleIdAllocator::new();
        let id = id_alloc.alloc();

        for max_unused_warm_slots in [0, 1, 2] {
            let state = IndexAllocator::new(100, max_unused_warm_slots);

            let index1 = state.alloc(Some(id)).unwrap();
            let index2 = state.alloc(Some(id)).unwrap();
            state.free(index2);
            state.free(index1);
            assert!(state.alloc_affine_and_clear_affinity(id).is_some());
            assert!(state.alloc_affine_and_clear_affinity(id).is_some());
            assert_eq!(state.alloc_affine_and_clear_affinity(id), None);
        }
    }

    #[test]
    fn test_affinity_allocation_strategy_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let id_alloc = CompiledModuleIdAllocator::new();
        let ids = std::iter::repeat_with(|| id_alloc.alloc())
            .take(10)
            .collect::<Vec<_>>();
        let state = IndexAllocator::new(1000, 1000);
        let mut allocated: Vec<SlotId> = vec![];
        let mut last_id = vec![None; 1000];

        let mut hits = 0;
        for _ in 0..100_000 {
            loop {
                if !allocated.is_empty() && rng.gen_bool(0.5) {
                    let i = rng.gen_range(0..allocated.len());
                    let to_free_idx = allocated.swap_remove(i);
                    state.free(to_free_idx);
                } else {
                    let id = ids[rng.gen_range(0..ids.len())];
                    let index = match state.alloc(Some(id)) {
                        Some(id) => id,
                        None => continue,
                    };
                    if last_id[index.index()] == Some(id) {
                        hits += 1;
                    }
                    last_id[index.index()] = Some(id);
                    allocated.push(index);
                }
                break;
            }
        }

        // 10% reuse would be random chance (because we have 10 module
        // IDs). Check for at least double that to ensure some sort of
        // affinity is occurring.
        assert!(
            hits > 20000,
            "expected at least 20000 (20%) ID-reuses but got {}",
            hits
        );
    }

    #[test]
    fn test_affinity_threshold() {
        let id_alloc = CompiledModuleIdAllocator::new();
        let id1 = id_alloc.alloc();
        let id2 = id_alloc.alloc();
        let id3 = id_alloc.alloc();
        let state = IndexAllocator::new(10, 2);

        // Set some slot affinities
        assert_eq!(state.alloc(Some(id1)), Some(SlotId(0)));
        state.free(SlotId(0));
        assert_eq!(state.alloc(Some(id2)), Some(SlotId(1)));
        state.free(SlotId(1));

        // Only 2 slots are allowed to be unused and warm, so we're at our
        // threshold, meaning one must now be evicted.
        assert_eq!(state.alloc(Some(id3)), Some(SlotId(0)));
        state.free(SlotId(0));

        // pickup `id2` again, it should be affine.
        assert_eq!(state.alloc(Some(id2)), Some(SlotId(1)));

        // with only one warm slot available allocation for `id1` should pick a
        // fresh slot
        assert_eq!(state.alloc(Some(id1)), Some(SlotId(2)));

        state.free(SlotId(1));
        state.free(SlotId(2));

        // ensure everything stays affine
        assert_eq!(state.alloc(Some(id1)), Some(SlotId(2)));
        assert_eq!(state.alloc(Some(id2)), Some(SlotId(1)));
        assert_eq!(state.alloc(Some(id3)), Some(SlotId(0)));

        state.free(SlotId(1));
        state.free(SlotId(2));
        state.free(SlotId(0));

        // LRU is 1, so that should be picked
        assert_eq!(state.alloc(Some(id_alloc.alloc())), Some(SlotId(1)));

        // Pick another LRU entry, this time 2
        assert_eq!(state.alloc(Some(id_alloc.alloc())), Some(SlotId(2)));

        // This should preserve slot `0` and pick up something new
        assert_eq!(state.alloc(Some(id_alloc.alloc())), Some(SlotId(3)));

        state.free(SlotId(1));
        state.free(SlotId(2));
        state.free(SlotId(3));

        // for good measure make sure id3 is still affine
        assert_eq!(state.alloc(Some(id3)), Some(SlotId(0)));
    }
}
