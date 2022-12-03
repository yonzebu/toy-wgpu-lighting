use std::cell::Cell;
use std::ops::Deref;

#[derive(Debug)]
pub struct Dirtiable<T> {
    dirty: Cell<bool>,
    value: T,
}

/// A type to mark items as dirty/changed for when e.g. you want to only do expensive I/O 
/// using something if it's been changed.
impl<T> Dirtiable<T> {
    /// Makes a new dirty value
    pub fn new(value: T) -> Self {
        Dirtiable {
            dirty: true.into(),
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

    pub fn into_inner(self) -> T {
        self.value
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty.get()
    }

    /// Returns a reference to the contained value.
    /// Unlike the [`Deref`] implementation, this explicitly marks the value as clean.
    pub fn clean(&self) -> &T {
        self.dirty.set(false);
        &self.value
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
    pub fn get_mut(&mut self) -> &mut T {
        self.dirty.set(true);
        &mut self.value
    }

    /// Calls the provided function-like object and marks this as dirty
    pub fn modify<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R,
    {
        self.dirty.set(true);
        f(&mut self.value)
    }

    /// Overwrites the contained value and marks this as dirty
    pub fn write(&mut self, new_value: T) {
        self.dirty.set(true);
        self.value = new_value;
    }

    pub fn replace(&mut self, mut new_value: T) -> T {
        std::mem::swap(&mut self.value, &mut new_value);
        self.dirty.set(true);
        new_value
    }

    pub fn mark_clean(&mut self) {
        self.dirty.set(false);
    }
    pub fn mark_dirty(&mut self) {
        self.dirty.set(true);
    }
}

impl<T: Copy> Dirtiable<T> {
    fn get(&self) -> T {
        self.value
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
