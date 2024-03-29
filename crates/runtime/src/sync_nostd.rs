#![allow(missing_docs)]

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

pub struct OnceLock<T> {
    val: UnsafeCell<MaybeUninit<T>>,
    state: AtomicU8,
}

unsafe impl<T: Send> Send for OnceLock<T> {}
unsafe impl<T: Sync> Sync for OnceLock<T> {}

const UNINITIALIZED: u8 = 0;
const INITIALIZING: u8 = 1;
const INITIALIZED: u8 = 2;

impl<T> OnceLock<T> {
    pub const fn new() -> OnceLock<T> {
        OnceLock {
            state: AtomicU8::new(UNINITIALIZED),
            val: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    pub fn get_or_init(&self, f: impl FnOnce() -> T) -> &T {
        if let Some(ret) = self.get() {
            return ret;
        }
        self.try_init::<()>(|| Ok(f())).unwrap()
    }

    pub fn get_or_try_init<E>(&self, f: impl FnOnce() -> Result<T, E>) -> Result<&T, E> {
        if let Some(ret) = self.get() {
            return Ok(ret);
        }
        self.try_init(f)
    }

    fn get(&self) -> Option<&T> {
        match self.state.load(Ordering::Acquire) {
            INITIALIZED => Some(unsafe { (*self.val.get()).assume_init_ref() }),
            _ => None,
        }
    }

    #[cold]
    fn try_init<E>(&self, f: impl FnOnce() -> Result<T, E>) -> Result<&T, E> {
        match self.state.compare_exchange(
            UNINITIALIZED,
            INITIALIZING,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(UNINITIALIZED) => match f() {
                Ok(val) => {
                    let ret = unsafe { &*(*self.val.get()).write(val) };
                    let prev = self.state.swap(INITIALIZED, Ordering::Release);
                    assert_eq!(prev, INITIALIZING);
                    Ok(ret)
                }
                Err(e) => match self.state.swap(UNINITIALIZED, Ordering::Relaxed) {
                    INITIALIZING => Err(e),
                    _ => unreachable!(),
                },
            },
            Err(INITIALIZING) => panic!("concurrent initialization only allowed with `std`"),
            Err(INITIALIZED) => Ok(self.get().unwrap()),
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Default)]
pub struct RwLock<T> {
    val: UnsafeCell<T>,
    state: AtomicU32,
}

unsafe impl<T: Send> Send for RwLock<T> {}
unsafe impl<T: Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    pub const fn new(val: T) -> RwLock<T> {
        RwLock {
            val: UnsafeCell::new(val),
            state: AtomicU32::new(0),
        }
    }

    pub fn read(&self) -> impl Deref<Target = T> + '_ {
        const READER_LIMIT: u32 = u32::MAX / 2;
        match self
            .state
            .fetch_update(Ordering::Acquire, Ordering::Acquire, |x| match x {
                u32::MAX => None,
                n => {
                    let next = n + 1;
                    if next < READER_LIMIT {
                        Some(next)
                    } else {
                        None
                    }
                }
            }) {
            Ok(_) => RwLockReadGuard { lock: self },
            Err(_) => panic!(
                "concurrent read request while locked for writing, must use `std` to avoid panic"
            ),
        }
    }

    pub fn write(&self) -> impl DerefMut<Target = T> + '_ {
        match self
            .state
            .compare_exchange(0, u32::MAX, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(0) => RwLockWriteGuard { lock: self },
            _ => panic!("concurrent write request, must use `std` to avoid panicking"),
        }
    }
}

struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.val.get() }
    }
}

impl<T> Drop for RwLockReadGuard<'_, T> {
    fn drop(&mut self) {
        self.lock.state.fetch_sub(1, Ordering::Release);
    }
}

struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.val.get() }
    }
}

impl<T> DerefMut for RwLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.val.get() }
    }
}

impl<T> Drop for RwLockWriteGuard<'_, T> {
    fn drop(&mut self) {
        match self.lock.state.swap(0, Ordering::Release) {
            u32::MAX => {}
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_once_lock() {
        let lock = OnceLock::new();
        assert!(lock.get().is_none());
        assert_eq!(*lock.get_or_init(|| 1), 1);
        assert_eq!(*lock.get_or_init(|| 2), 1);
        assert_eq!(*lock.get_or_init(|| 3), 1);
        assert_eq!(lock.get_or_try_init::<()>(|| Ok(3)), Ok(&1));

        let lock = OnceLock::new();
        assert_eq!(lock.get_or_try_init::<()>(|| Ok(3)), Ok(&3));
        assert_eq!(*lock.get_or_init(|| 1), 3);

        let lock = OnceLock::new();
        assert_eq!(lock.get_or_try_init(|| Err(())), Err(()));
        assert_eq!(*lock.get_or_init(|| 1), 1);
    }

    #[test]
    fn smoke_rwlock() {
        let lock = RwLock::new(1);
        assert_eq!(*lock.read(), 1);

        let a = lock.read();
        let b = lock.read();
        assert_eq!(*a, 1);
        assert_eq!(*b, 1);
        drop((a, b));

        assert_eq!(*lock.write(), 1);

        *lock.write() = 4;
        assert_eq!(*lock.read(), 4);
        assert_eq!(*lock.write(), 4);

        let a = lock.read();
        let b = lock.read();
        assert_eq!(*a, 4);
        assert_eq!(*b, 4);
        drop((a, b));
    }

    #[test]
    #[should_panic(expected = "concurrent write request")]
    fn rwlock_panic_read_then_write() {
        let lock = RwLock::new(1);
        let _a = lock.read();
        let _b = lock.write();
    }

    #[test]
    #[should_panic(expected = "concurrent read request")]
    fn rwlock_panic_write_then_read() {
        let lock = RwLock::new(1);
        let _a = lock.write();
        let _b = lock.read();
    }

    #[test]
    #[should_panic(expected = "concurrent write request")]
    fn rwlock_panic_write_then_write() {
        let lock = RwLock::new(1);
        let _a = lock.write();
        let _b = lock.write();
    }
}
