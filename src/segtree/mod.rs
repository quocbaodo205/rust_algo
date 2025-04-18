mod dynamic;
mod lazy;
mod persistent;

pub use dynamic::*;
pub use lazy::*;
pub use persistent::*;
use std::{
    any::type_name,
    fmt::Debug,
    ops::{Bound, Range, RangeBounds},
};
fn convert_range(len: usize, range: impl RangeBounds<usize>) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(l) => *l,
        Bound::Unbounded => 0,
        _ => unreachable!(),
    };
    let end = match range.end_bound() {
        Bound::Included(r) => r + 1,
        Bound::Excluded(r) => *r,
        Bound::Unbounded => len,
    };
    Range { start, end }
}

/// If the Link-Cut Tree does not require any operations, this type can be used as a dummy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DefaultZST;

impl MapMonoid for DefaultZST {
    type M = ();
    type Act = ();

    fn e() -> Self::M {}
    fn op(_: &Self::M, _: &Self::M) -> Self::M {}
    fn map(_: &Self::M, _: &Self::Act) -> Self::M {}
    fn id() -> Self::Act {}
    fn composite(_: &Self::Act, _: &Self::Act) -> Self::Act {}
}

pub trait ZeroOne {
    fn zero() -> Self;
    fn one() -> Self;
}

macro_rules! impl_zero_one {
    ( $zero:expr, $one:expr, $( $t:ty ),* ) => {
        $(
            impl ZeroOne for $t {
                fn zero() -> Self { $zero }
                fn one() -> Self { $one }
            }
        )*
    };
}

impl_zero_one!(0, 1, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
impl_zero_one!(0.0, 1.0, f32, f64);

fn convert_range_isize(min: isize, max: isize, range: impl RangeBounds<isize>) -> Range<isize> {
    let l = match range.start_bound() {
        Bound::Included(l) => *l,
        Bound::Unbounded => min,
        Bound::Excluded(l) => l - 1,
    };
    let r = match range.end_bound() {
        Bound::Included(r) => r + 1,
        Bound::Excluded(r) => *r,
        Bound::Unbounded => max,
    };
    Range { start: l, end: r }
}

pub struct SegmentTree<T: Monoid> {
    t: Vec<T::M>,
}

impl<T: Monoid> SegmentTree<T> {
    /// Create new `SegmentTree` filled with `M::id`.
    pub fn new(size: usize) -> Self {
        Self {
            t: (0..size * 2).map(|_| T::id()).collect(),
        }
    }

    pub fn from_vec(v: Vec<T::M>) -> Self {
        let size = v.len();
        let mut t = (0..size).map(|_| T::id()).chain(v).collect::<Vec<_>>();

        for i in (1..size).rev() {
            t[i] = T::op(&t[i << 1], &t[(i << 1) | 1]);
        }

        Self { t }
    }

    pub fn len(&self) -> usize {
        self.t.len() >> 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get `index`-th element.
    ///
    /// # Panics
    /// - `index < self.len()` must be satisfied.
    pub fn get(&self, index: usize) -> &T::M {
        assert!(index < self.len());
        &self.t[index + self.len()]
    }

    /// Set `val` to `index`-th element.
    ///
    /// # Panics
    /// - `index < self.len()` must be satisfied.
    pub fn set(&mut self, mut index: usize, val: T::M) {
        assert!(index < self.len());

        index += self.len();
        self.t[index] = val;
        while index > 1 {
            let (l, r) = (index.min(index ^ 1), index.max(index ^ 1));
            self.t[index >> 1] = T::op(&self.t[l], &self.t[r]);
            index >>= 1;
        }
    }

    /// Update `index`-th element by `f`.
    ///
    /// This method is equivalent to `self.set(index, f(self.get(index)))`.
    ///
    /// # Panics
    /// - `index < self.len()` must be satisfied.
    pub fn update_by<F>(&mut self, index: usize, f: F)
    where
        F: Fn(&T::M) -> T::M,
    {
        assert!(index < self.len());
        let new = f(&self.t[index + self.len()]);
        self.set(index, new);
    }

    /// Apply `M::op` to the elements within `range` and return its result.
    ///
    /// # Panics
    /// - The head of `range` must be smaller than or equal to the tail of `range`.
    /// - `range` must not contain a range greater than `self.len()`.
    ///
    /// # Examples
    /// ```rust
    /// use ds::{SegmentTree, Monoid};
    ///
    /// struct I32Sum;
    /// impl Monoid for I32Sum {
    ///     type M = i32;
    ///     fn id() -> i32 { 0 }
    ///     fn op(l: &i32, r: &i32) -> i32 { l + r }
    /// }
    ///
    /// let mut st = SegmentTree::<I32Sum>::from_vec(vec![0, 1, 2, 3]);
    /// assert_eq!(st.fold(1..3), 3);
    /// assert_eq!(st.fold(..), 6);
    /// st.set(2, 5);
    /// assert_eq!(st.fold(..), 9);
    /// // Panics !!! (range.start > range.end)
    /// // st.fold(3..1);
    /// // Panics !!! (index out of range)
    /// // st.fold(1..5);
    /// ```
    pub fn fold(&self, range: impl RangeBounds<usize>) -> T::M {
        let Range { start, end } = convert_range(self.len(), range);
        assert!(start <= end);
        assert!(end <= self.len());

        let (mut l, mut r) = (start + self.len(), end + self.len());
        let (mut lf, mut rf) = (T::id(), T::id());
        while l < r {
            if l & 1 != 0 {
                lf = T::op(&lf, &self.t[l]);
                l += 1;
            }
            if r & 1 != 0 {
                rf = T::op(&self.t[r - 1], &rf);
            }
            l >>= 1;
            r >>= 1;
        }

        T::op(&lf, &rf)
    }
}

impl<T: Monoid> Clone for SegmentTree<T>
where
    T::M: Clone,
{
    fn clone(&self) -> Self {
        SegmentTree { t: self.t.clone() }
    }
}

impl<T: Monoid> Debug for SegmentTree<T>
where
    T::M: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(type_name::<Self>())
            .field("t", &self.t)
            .finish()
    }
}

impl<M: Monoid> FromIterator<M::M> for SegmentTree<M> {
    fn from_iter<T: IntoIterator<Item = M::M>>(iter: T) -> Self {
        let mut t = iter.into_iter().collect::<Vec<M::M>>();
        let size = t.len();
        t.resize_with(size * 2, M::id);
        for i in 0..size {
            t.swap(i, i + size);
        }
        for i in (1..size).rev() {
            t[i] = M::op(&t[i << 1], &t[(i << 1) | 1]);
        }

        Self { t }
    }
}

// Some useful Monoid
#[derive(Debug, Clone)]
struct I32Sum;
impl Monoid for I32Sum {
    type M = i32;
    fn id() -> i32 {
        0
    }
    fn op(l: &i32, r: &i32) -> i32 {
        l + r
    }
}

#[derive(Debug, Clone)]
pub struct Reversible<T: Monoid + Clone> {
    pub forward: T,
    pub reverse: T,
}

impl<T: Monoid + Clone> Reversible<T> {
    pub fn new(val: T) -> Self {
        Self {
            forward: val.clone(),
            reverse: val,
        }
    }
}

impl<T: Monoid<M = T> + Clone> Monoid for Reversible<T> {
    type M = Self;
    fn id() -> Self::M {
        Self {
            forward: T::id(),
            reverse: T::id(),
        }
    }
    fn op(l: &Self::M, r: &Self::M) -> Self::M {
        Self {
            forward: T::op(&l.forward, &r.forward),
            reverse: T::op(&r.reverse, &l.reverse),
        }
    }
}
