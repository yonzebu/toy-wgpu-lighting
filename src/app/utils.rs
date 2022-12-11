use std::cell::Cell;
use std::hash::Hash;
use std::ops::Deref;

#[derive(Debug)]
pub struct Dirtiable<T> {
    dirty: Cell<bool>,
    value: T,
}

/// A type to mark items as dirty/changed for when e.g. you want to only do expensive I/O
/// using something if it's been changed. At any time is either marked "dirty" and has been
/// changed, or is "clean" and has not been changed since the last action that marked it clean.
///
/// As a consequence, almost every function that reads from this is distinguished partially
/// based on whether it leaves the "dirtiness" as it was or marks the value as clean.
impl<T> Dirtiable<T> {
    /// Makes a new dirty value.
    pub fn new(value: T) -> Self {
        Dirtiable {
            dirty: Cell::new(true),
            value,
        }
    }

    // TODO: decide if this should exist
    // /// Makes a new clean value
    // pub fn new_clean(value: T) -> Self {
    //     Dirtiable {
    //         dirty: false.into(),
    //         value,
    //     }
    // }

    /// Consumes this to produce the contained value
    pub fn _into_inner(self) -> T {
        self.value
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty.get()
    }
    pub fn _is_clean(&self) -> bool {
        !self.is_dirty()
    }

    /// Returns a reference to the contained value.
    /// Unlike the [`Deref`] implementation, this explicitly marks the value as clean.
    #[must_use]
    pub fn clean(&self) -> &T {
        self.dirty.set(false);
        &self.value
    }

    pub fn clean_with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        f(self.clean())
    }

    /// Calls the provided closure only if dirty, marking clean when it's done.
    pub fn if_dirty<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&T) -> R,
    {
        if self.is_dirty() {
            let result = f(&self.value);
            self.dirty.set(false);
            Some(result)
        } else {
            None
        }
    }

    /// This automatically marks this as dirty, even if you never write to the returned reference.
    #[must_use]
    pub fn _get_mut(&mut self) -> &mut T {
        self.dirty.set(true);
        &mut self.value
    }

    /// Calls the provided function-like object and marks this as dirty.
    pub fn modify<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        self.dirty.set(true);
        f(&mut self.value)
    }

    /// Overwrites the contained value and marks this as dirty.
    pub fn _write(&mut self, new_value: T) {
        self.dirty.set(true);
        self.value = new_value;
    }

    #[must_use]
    pub fn _replace(&mut self, mut new_value: T) -> T {
        std::mem::swap(&mut self.value, &mut new_value);
        self.dirty.set(true);
        new_value
    }

    pub fn mark_clean(&mut self) {
        self.dirty.set(false);
    }
    pub fn _mark_dirty(&mut self) {
        self.dirty.set(true);
    }
}

impl<T: Copy> Dirtiable<T> {
    /// Copies the contained value out.
    #[must_use]
    pub fn _get(&self) -> T {
        **self
    }

    /// Copies the contained value out and marks as clean.
    #[must_use]
    pub fn _get_and_clean(&self) -> T {
        *self.clean()
    }
}

impl<T: Default> Dirtiable<T> {
    /// Takes the contained value, replacing with a default and marking this dirty
    #[must_use]
    pub fn _take(&mut self) -> T {
        self._mark_dirty();
        self._replace(T::default())
    }

    /// Gets the contained value and replaces it with the default value, marking this clean when done.
    #[must_use]
    pub fn _flush(&mut self) -> T {
        let old = self._replace(T::default());
        self.mark_clean();
        old
    }

    /// Calls a closure on the contained value after replacing it with the default value
    /// and marking clean.
    pub fn _flush_with<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(T) -> R,
    {
        f(self._flush())
    }
}

impl<T: Clone> Clone for Dirtiable<T> {
    fn clone(&self) -> Self {
        Self::new(self.value.clone())
    }
}

impl<T: Default> Default for Dirtiable<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: PartialEq> PartialEq for Dirtiable<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
    fn ne(&self, other: &Self) -> bool {
        self.value != other.value
    }
}
impl<T: Eq> Eq for Dirtiable<T> {}

impl<T: PartialOrd> PartialOrd for Dirtiable<T> {
    fn ge(&self, other: &Self) -> bool {
        self.value >= other.value
    }
    fn gt(&self, other: &Self) -> bool {
        self.value > other.value
    }
    fn le(&self, other: &Self) -> bool {
        self.value <= other.value
    }
    fn lt(&self, other: &Self) -> bool {
        self.value < other.value
    }
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}
impl<T: Ord> Ord for Dirtiable<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T: Hash> Hash for Dirtiable<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state)
    }
}

impl<T> From<T> for Dirtiable<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> Deref for Dirtiable<T> {
    type Target = T;
    /// Does not implicitly mark clean
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

// pub trait IsId
// where
//     Self: Into<Self::Underlying>,
//     // usize: Into<Self>,
//     Self::Underlying: Into<Self>,
// {
//     type Underlying: Eq + Hash + Copy + Default;
//     fn succ(&self) -> Option<Self>;
//     fn pred(&self) -> Option<Self>;
// }

// pub trait IsId : Into<usize> + From<usize> + Copy
// {}

// #[derive(Default, Clone, Debug, PartialEq, Eq)]
// pub struct IdMap<I, V> {
//     internal: HashMap<usize, V>,
//     next_id: I,
// }
// type Keys<'a, I: IsId, V> = Map<std::collections::hash_map::Keys<'a, usize, V>, fn(&usize) -> I>;

// impl<I: IsId, V> IdMap<I, V> {

//     pub fn new() -> Self {
//         Self {
//             internal: HashMap::new(),
//             next_id: 0.into(),
//         }
//     }

//     pub fn get(&self, id: I) -> Option<&V> {
//         self.internal.get(&id.into())
//     }

//     pub fn get_mut(&mut self, id: I) -> Option<&mut V> {
//         self.internal.get_mut(&id.into())
//     }

//     pub fn insert(&mut self, value: V) -> Option<I> {
//         let orig_id: usize = self.next_id.into();
//         let mut next_id = orig_id.wrapping_add(1);
//         while next_id != orig_id {
//             if !self.internal.contains_key(&next_id) {
//                 self.internal.insert(next_id, value);
//                 self.next_id = next_id.into();
//                 return Some(next_id.into());
//             }
//             next_id = next_id.wrapping_add(1);
//         }
//         None
//     }

//     pub fn remove(&mut self, id: I) -> Option<V> {
//         self.internal.remove(&id.into())
//     }

//     pub fn keys<'a>(&'a self) -> Keys<'a, I, V> {
//         self.internal.keys().map(|&id| id.into())
//     }

//     // pub fn values<'a>(&'a self)
// }

// pub struct QueuedUpdate<T> {
//     queued: T,
//     value: T,
// }

// impl<T: Copy> QueuedUpdate<T> {
//     pub fn queue_replace(&mut self, to_queue: T) {
//         self.queued = to_queue;
//     }

//     pub fn apply_queue(&mut self) {
//         self.value = self.queued;
//     }
// }
// impl<T: Default> QueuedUpdate<T> {
//     pub fn flush_queue(&mut self) {
//         self.value = std::mem::replace(&mut self.queued, Default::default());
//     }
// }

// pub struct Instant {
//     #[cfg(not(target_arch = "wasm32"))]
//     internal: ::std::time::Instant,
//     #[cfg(target_arch = "wasm32")]

// }
