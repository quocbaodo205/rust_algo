use std::{
    cmp::{max, min},
    ops::{Add, AddAssign},
};

#[allow(dead_code)]
pub struct SegmentTreeMin {
    len: usize,
    tree_min: Vec<usize>,
}

#[allow(dead_code)]
impl SegmentTreeMin {
    pub fn new(n: usize) -> Self {
        SegmentTreeMin {
            len: n,
            tree_min: vec![1000000000; 4 * n + 1],
        }
    }

    fn query_internal_min(&self, node: usize, tl: usize, tr: usize, ql: usize, qr: usize) -> usize {
        if ql > qr {
            return 1000000000;
        }
        if tl == ql && tr == qr {
            return self.tree_min[node];
        }
        let mid = (tl + tr) / 2;
        min(
            self.query_internal_min(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal_min(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        )
    }

    // Query the inclusive range [l..r]
    pub fn query_min(&self, l: usize, r: usize) -> usize {
        self.query_internal_min(1, 0, self.len - 1, l, r)
    }

    fn update_internal_min(&mut self, node: usize, tl: usize, tr: usize, pos: usize, val: usize) {
        if tl == tr {
            self.tree_min[node] = val;
            // println!("tree_min at {tl},{tr} = {}", self.tree_min[node]);
        } else {
            let mid = (tl + tr) / 2;
            if pos <= mid {
                self.update_internal_min(node * 2, tl, mid, pos, val);
            } else {
                self.update_internal_min(node * 2 + 1, mid + 1, tr, pos, val);
            }
            self.tree_min[node] = min(self.tree_min[node * 2], self.tree_min[node * 2 + 1]);
            // println!("tree_min at {tl},{tr} = {}", self.tree_min[node]);
        }
    }

    pub fn update_min(&mut self, pos: usize, val: usize) {
        self.update_internal_min(1, 0, self.len - 1, pos, val);
    }
}

#[allow(dead_code)]
pub struct SegmentTreeMax {
    len: usize,
    tree_max: Vec<usize>,
}

#[allow(dead_code)]
impl SegmentTreeMax {
    pub fn new(n: usize) -> Self {
        SegmentTreeMax {
            len: n,
            tree_max: vec![0; 4 * n + 1],
        }
    }

    fn query_internal_max(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> usize {
        if ql > qr {
            return 0;
        }
        if tl == ql && tr == qr {
            return self.tree_max[node];
        }
        let mid = (tl + tr) / 2;
        max(
            self.query_internal_max(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal_max(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        )
    }

    // Query the inclusive range [l..r]
    pub fn query_max(&mut self, l: usize, r: usize) -> usize {
        self.query_internal_max(1, 0, self.len - 1, l, r)
    }

    fn update_internal_max(&mut self, node: usize, tl: usize, tr: usize, pos: usize, val: usize) {
        if tl == tr {
            self.tree_max[node] = val;
            // println!("tree_max at {tl},{tr} = {}", self.tree_max[node]);
        } else {
            let mid = (tl + tr) / 2;
            if pos <= mid {
                self.update_internal_max(node * 2, tl, mid, pos, val);
            } else {
                self.update_internal_max(node * 2 + 1, mid + 1, tr, pos, val);
            }
            self.tree_max[node] = max(self.tree_max[node * 2], self.tree_max[node * 2 + 1]);
            // println!("tree_max at {tl},{tr} = {}", self.tree_max[node]);
        }
    }

    pub fn update_max(&mut self, pos: usize, val: usize) {
        self.update_internal_max(1, 0, self.len - 1, pos, val);
    }
}

// Allow you to check if a range [l,r] is increasing after single change query.
#[allow(dead_code)]
pub struct SegmentTreeIsInc {
    len: usize,
    tree_min: Vec<usize>,
    tree_max: Vec<usize>,
    is_inc: Vec<bool>,
}

#[allow(dead_code)]
impl SegmentTreeIsInc {
    pub fn new(n: usize) -> Self {
        SegmentTreeIsInc {
            len: n,
            tree_min: vec![1000000000; 4 * n + 1],
            tree_max: vec![0; 4 * n + 1],
            is_inc: vec![true; 4 * n + 1],
        }
    }

    fn query_internal_min(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> usize {
        if ql > qr {
            return 1000000000;
        }
        if tl == ql && tr == qr {
            return self.tree_min[node];
        }
        let mid = (tl + tr) / 2;
        let mmin = min(
            self.query_internal_min(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal_min(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        );
        mmin
    }

    fn query_internal_max(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> usize {
        if ql > qr {
            return 0;
        }
        if tl == ql && tr == qr {
            return self.tree_max[node];
        }
        let mid = (tl + tr) / 2;
        let mmax = max(
            self.query_internal_max(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal_max(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        );
        mmax
    }

    fn query_internal_is_inc(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> bool {
        if ql > qr {
            return true;
        }
        if tl == ql && tr == qr {
            return self.is_inc[node];
        }
        let mid = (tl + tr) / 2;
        let max_left = self.query_internal_max(node * 2, tl, mid, ql, min(qr, mid));
        let min_right = self.query_internal_min(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr);
        if max_left <= min_right {
            return self.query_internal_is_inc(node * 2, tl, mid, ql, min(qr, mid))
                & self.query_internal_is_inc(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr);
        } else {
            return false;
        }
    }

    // Query the inclusive range [l..r]
    pub fn query_min(&mut self, l: usize, r: usize) -> usize {
        self.query_internal_min(1, 0, self.len - 1, l, r)
    }

    pub fn query_is_inc(&mut self, l: usize, r: usize) -> bool {
        self.query_internal_is_inc(1, 0, self.len - 1, l, r)
    }

    fn update_internal(&mut self, node: usize, tl: usize, tr: usize, pos: usize, val: usize) {
        if tl == tr {
            self.tree_min[node] = val;
            self.tree_max[node] = val;
            self.is_inc[node] = true;
        } else {
            let mid = (tl + tr) / 2;
            if pos <= mid {
                self.update_internal(node * 2, tl, mid, pos, val);
            } else {
                self.update_internal(node * 2 + 1, mid + 1, tr, pos, val);
            }
            self.tree_min[node] = min(self.tree_min[node * 2], self.tree_min[node * 2 + 1]);
            self.tree_max[node] = max(self.tree_max[node * 2], self.tree_max[node * 2 + 1]);
            if self.tree_max[node * 2] <= self.tree_min[node * 2 + 1] {
                // println!(
                //     "Updating inc of {tl},{tr} = {:?}",
                //     self.is_inc[node * 2] & self.is_inc[node * 2 + 1]
                // );
                self.is_inc[node] = self.is_inc[node * 2] & self.is_inc[node * 2 + 1];
            } else {
                // println!(
                //     "Updating inc of {tl},{tr} to false since it {} > {}",
                //     self.tree_max[node * 2],
                //     self.tree_min[node * 2 + 1]
                // );
                self.is_inc[node] = false;
            }
        }
    }

    pub fn update(&mut self, pos: usize, val: usize) {
        self.update_internal(1, 0, self.len - 1, pos, val);
    }
}

// =============================================================================
// Template for segment tree that allow flexible Lazy and Value definition

// Lazy operation that allow range add or range set
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegtreeLazy {
    pub val: i64,
    // Is increase, if false then it's a set operation.
    // Remember to find all is_inc and change to false should needed.
    pub is_inc: bool,
}

impl AddAssign for SegtreeLazy {
    fn add_assign(&mut self, rhs: Self) {
        if !rhs.is_inc {
            self.val = 0;
            self.is_inc = false;
        }
        self.val += rhs.val;
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegtreeValue {
    sum: i64,
    mx: i64,
}

#[allow(dead_code)]
impl SegtreeValue {
    // Update from lazy operation
    pub fn update(&mut self, lz: SegtreeLazy, sz: usize) {
        if !lz.is_inc {
            self.sum = 0;
            self.mx = 0;
        }
        self.sum += lz.val * (sz as i64);
        self.mx += lz.val;
    }
}

impl Add for SegtreeValue {
    type Output = SegtreeValue;

    fn add(self, rhs: Self) -> Self::Output {
        SegtreeValue {
            sum: self.sum + rhs.sum,
            mx: max(self.mx, rhs.mx),
        }
    }
}

pub const LID: SegtreeLazy = SegtreeLazy {
    val: 0,
    is_inc: true,
};
pub const VID: SegtreeValue = SegtreeValue {
    sum: 0,
    mx: -1000000000000000000,
};

// ============================== Main structure ====================

#[allow(dead_code)]
pub struct SegmentTreeLazy {
    len: usize,
    tree: Vec<SegtreeValue>,
    lazy: Vec<SegtreeLazy>,
}

#[allow(dead_code)]
impl SegmentTreeLazy {
    pub fn new(n: usize) -> Self {
        SegmentTreeLazy {
            len: n,
            tree: vec![VID; 4 * n + 10],
            lazy: vec![LID; 4 * n + 10],
        }
    }

    fn build_internal(&mut self, node: usize, tl: usize, tr: usize, a: &[SegtreeValue]) {
        if tl == tr {
            self.tree[node] = a[tl];
        } else {
            let mid = (tl + tr) / 2;
            self.build_internal(node * 2, tl, mid, a);
            self.build_internal(node * 2 + 1, mid + 1, tr, a);
            self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1];
        }
    }

    pub fn build(a: &[SegtreeValue]) -> Self {
        let mut segment_tree = SegmentTreeLazy {
            len: a.len(),
            tree: vec![VID; 4 * a.len() + 10],
            lazy: vec![LID; 4 * a.len() + 10],
        };
        segment_tree.build_internal(1, 0, a.len() - 1, a);
        segment_tree
    }

    // Update current node and push lazy to children
    fn push(&mut self, node: usize, _tl: usize, _tr: usize) {
        // Update tree
        let sz = _tr + 1 - _tl;
        let lzd = self.lazy[node];
        // println!("Pushing lz = {lzd:?} to node {node}, _tl = {_tl}, tr = {_tr}");
        // Reset current lazy
        self.lazy[node] = LID;

        self.tree[node].update(lzd, sz);

        // Update lazy
        if node * 2 + 1 >= self.tree.len() {
            return;
        }
        self.lazy[node * 2] += lzd;
        self.lazy[node * 2 + 1] += lzd;
    }

    fn query_internal(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> SegtreeValue {
        if ql > qr {
            return VID;
        }
        if tl == ql && tr == qr {
            self.push(node, tl, tr);
            return self.tree[node];
        }
        self.push(node, tl, tr);
        let mid = (tl + tr) / 2;
        let ans = self.query_internal(node * 2, tl, mid, ql, min(qr, mid))
            + self.query_internal(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr);
        println!("Got ans for {tl} {tr} {ql} {qr} = {ans:?}");
        ans
    }

    // Query the inclusive range [l..r]
    pub fn query(&mut self, l: usize, r: usize) -> SegtreeValue {
        self.query_internal(1, 0, self.len - 1, l, r)
    }

    // When needed, please apply transformation before passing [val]
    fn update_internal(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
        lz: SegtreeLazy,
    ) {
        if ql > qr {
            return;
        }
        if tl == ql && tr == qr {
            self.lazy[node] += lz;
            self.push(node, tl, tr);
        } else {
            self.push(node, tl, tr);
            let mid = (tl + tr) / 2;
            self.update_internal(node * 2, tl, mid, ql, min(qr, mid), lz);
            self.update_internal(node * 2 + 1, mid + 1, tr, max(ql, mid + 1), qr, lz);
            self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1];
            println!("After update, tree node {node} = {:?}", self.tree[node]);
        }
    }

    // Update range [l,r] with lazy
    pub fn update(&mut self, ql: usize, qr: usize, lz: SegtreeLazy) {
        self.update_internal(1, 0, self.len - 1, ql, qr, lz);
    }
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn test_segment_tree() {
        // Testing range add + range set / sum + max query in 1 tree:
        let a = [1, 2, 3, 4, 5, 6];
        let tree_val: Vec<SegtreeValue> =
            a.iter().map(|&x| SegtreeValue { sum: x, mx: x }).collect();
        let mut st = SegmentTreeLazy::build(&tree_val);
        assert_eq!(st.query(0, 0), SegtreeValue { sum: 1, mx: 1 });
        assert_eq!(st.query(0, 1), SegtreeValue { sum: 3, mx: 2 });
        assert_eq!(
            st.query(2, 5),
            SegtreeValue {
                sum: 3 + 4 + 5 + 6,
                mx: 6
            }
        );

        // Range [1,2] += 1, arr = [1,3,4,4,5,6]
        st.update(
            1,
            2,
            SegtreeLazy {
                val: 1,
                is_inc: true,
            },
        );
        assert_eq!(
            st.query(0, 5),
            SegtreeValue {
                sum: 1 + 3 + 4 + 4 + 5 + 6,
                mx: 6
            }
        );
        assert_eq!(st.query(2, 3), SegtreeValue { sum: 4 + 4, mx: 4 });

        // Range set [3,4] = 0, arr = [1,3,4,0,0,6]
        st.update(
            3,
            4,
            SegtreeLazy {
                val: 0,
                is_inc: false,
            },
        );
        assert_eq!(
            st.query(0, 5),
            SegtreeValue {
                sum: 1 + 3 + 4 + 6,
                mx: 6
            }
        );
        assert_eq!(st.query(2, 3), SegtreeValue { sum: 4, mx: 4 });

        // Range set [3,4] = 0, arr = [1,3,4,0,0,6]
        st.update(
            0,
            0,
            SegtreeLazy {
                val: -1,
                is_inc: false,
            },
        );
        assert_eq!(
            st.query(0, 5),
            SegtreeValue {
                sum: 3 - 1 + 4 + 6,
                mx: 6
            }
        );
        assert_eq!(st.query(2, 3), SegtreeValue { sum: 4, mx: 4 });
        assert_eq!(st.query(5, 5), SegtreeValue { sum: 6, mx: 6 });
        st.update(
            5,
            5,
            SegtreeLazy {
                val: 15,
                is_inc: false,
            },
        );
        assert_eq!(st.query(5, 5), SegtreeValue { sum: 15, mx: 15 });
    }

    #[test]
    fn test_segment_tree_limit() {
        let a = vec![0; 200000];
        let tree_val: Vec<SegtreeValue> =
            a.iter().map(|&x| SegtreeValue { sum: x, mx: x }).collect();
        let mut st = SegmentTreeLazy::build(&tree_val);
        st.update(
            0,
            0,
            SegtreeLazy {
                val: 1000000000,
                is_inc: false,
            },
        );
        assert_eq!(
            st.query(0, 200000 - 1),
            SegtreeValue {
                sum: 1000000000,
                mx: 1000000000
            }
        );
        st.update(
            200000 - 1,
            200000 - 1,
            SegtreeLazy {
                val: 1000000000,
                is_inc: false,
            },
        );
        assert_eq!(
            st.query(0, 200000 - 1),
            SegtreeValue {
                sum: 2000000000,
                mx: 1000000000
            }
        );
        assert_eq!(
            st.query(200000 - 1, 200000 - 1),
            SegtreeValue {
                sum: 1000000000,
                mx: 1000000000
            }
        );
    }
}

// ============== Lib checker segment tree that I use for speed ================

// https://judge.yosupo.jp/problem/range_affine_range_sum
// https://judge.yosupo.jp/submission/214491
// Update i in [l..r): a[i] = a[i] * b + c
// st.apply(l, r, affine(b, c));
// let ans = st.sum(l, r);
// Init:
// let a: Vec<(i32, i32)> = v.iter().map(|&x| (x, 1));
// let mut st = LazySegTree::from(a);
use lazy_segtree::*;
impl Monoid for (i32, i32) {
    fn id() -> Self {
        (0, 0)
    }
    fn op(&self, other: &Self) -> Self {
        (self.0 + other.0, self.1 + other.1)
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Affine {
    a: i32,
    b: i32,
}
fn affine(a: i32, b: i32) -> Affine {
    Affine { a, b }
}
impl Monoid for Affine {
    fn id() -> Self {
        affine(1, 0)
    }
    fn op(&self, other: &Self) -> Self {
        affine(self.a * other.a, self.a * other.b + self.b)
    }
}
impl Map<(i32, i32)> for Affine {
    fn map(&self, x: &(i32, i32)) -> (i32, i32) {
        (self.a * x.0 + self.b * x.1, x.1)
    }
}
pub mod lazy_segtree {
    use std::mem;
    pub trait Monoid {
        fn id() -> Self;
        fn op(&self, other: &Self) -> Self;
    }
    pub trait Map<T> {
        fn map(&self, value: &T) -> T;
    }
    pub struct LazySegTree<T, F> {
        a: Vec<T>,
        f: Vec<F>,
    }
    impl<T: Monoid, F: Map<T> + Monoid> LazySegTree<T, F> {
        pub fn new(len: usize) -> Self {
            Self {
                a: (0..2 * len).map(|_| T::id()).collect(),
                f: (0..len).map(|_| F::id()).collect(),
            }
        }
        pub fn len(&self) -> usize {
            self.a.len() / 2
        }
        pub fn sum(&self, l: usize, r: usize) -> T {
            if l == r {
                return T::id();
            }
            let mut l = l + self.len();
            l >>= l.trailing_zeros();
            let mut r = r + self.len();
            r >>= r.trailing_zeros();
            let mut sum_l = T::id();
            let mut sum_r = T::id();
            loop {
                if l >= r {
                    let mut i = l / 2;
                    sum_l = sum_l.op(&self.a[l]);
                    l += 1;
                    l >>= l.trailing_zeros();
                    while i > l / 2 {
                        sum_l = self.f[i].map(&sum_l);
                        i /= 2;
                    }
                } else {
                    let mut i = r / 2;
                    r -= 1;
                    sum_r = self.a[r].op(&sum_r);
                    r >>= r.trailing_zeros();
                    while i > r / 2 {
                        sum_r = self.f[i].map(&sum_r);
                        i /= 2;
                    }
                }
                if l == r {
                    break;
                }
            }
            let mut sum = sum_l.op(&sum_r);
            let mut i = l / 2;
            while i > 0 {
                sum = self.f[i].map(&sum);
                i /= 2;
            }
            sum
        }
        pub fn apply(&mut self, l: usize, r: usize, f: F) {
            if l == r {
                return;
            }
            let mut l = l + self.len();
            l >>= l.trailing_zeros();
            let mut r = r + self.len();
            r >>= r.trailing_zeros();
            let (l_orig, r_orig) = (l, r);
            self.propagate(l / 2);
            self.propagate((r - 1) / 2);
            loop {
                if l >= r {
                    self.a[l] = f.map(&self.a[l]);
                    if let Some(fl) = self.f.get_mut(l) {
                        *fl = f.op(fl);
                    }
                    l += 1;
                    l >>= l.trailing_zeros();
                } else {
                    r -= 1;
                    self.a[r] = f.map(&self.a[r]);
                    if let Some(fr) = self.f.get_mut(r) {
                        *fr = f.op(fr);
                    }
                    r >>= r.trailing_zeros();
                }
                if l == r {
                    break;
                }
            }
            self.update_path(l_orig / 2);
            self.update_path((r_orig - 1) / 2);
        }
        fn propagate(&mut self, i: usize) {
            let h = usize::BITS - i.leading_zeros();
            for k in (0..h).rev() {
                let p = i >> k;
                let l = 2 * p;
                let r = 2 * p + 1;
                let f = mem::replace(&mut self.f[p], F::id());
                self.a[l] = f.map(&self.a[l]);
                if let Some(fl) = self.f.get_mut(l) {
                    *fl = f.op(fl);
                }
                self.a[r] = f.map(&self.a[r]);
                if let Some(fr) = self.f.get_mut(r) {
                    *fr = f.op(fr);
                }
            }
        }
        fn update_path(&mut self, mut i: usize) {
            while i > 0 {
                self.a[i] = self.a[2 * i].op(&self.a[2 * i + 1]);
                i /= 2;
            }
        }
    }
    impl<T: Monoid, F: Monoid> From<Vec<T>> for LazySegTree<T, F> {
        fn from(mut a: Vec<T>) -> Self {
            let len = a.len();
            a.reserve(len);
            let ptr = a.as_mut_ptr();
            unsafe {
                ptr.copy_to(ptr.add(len), len);
                for i in (1..len).rev() {
                    ptr.add(i)
                        .write(T::op(&*ptr.add(2 * i), &*ptr.add(2 * i + 1)));
                }
                ptr.write(T::id());
                a.set_len(2 * len);
            }
            let f = (0..len).map(|_| F::id()).collect();
            Self { a, f }
        }
    }
}

pub mod lazy_seg_tree {
    pub trait Monoid {
        fn id() -> Self;
        fn op(&self, other: &Self) -> Self;
    }
    pub trait Map<T> {
        fn map(&self, x: T) -> T;
    }
    pub struct LazySegTree<T, F> {
        ss: Box<[T]>,
        fs: Box<[F]>,
    }
    impl<T: Monoid, F: Monoid + Map<T>> LazySegTree<T, F> {
        pub fn new(n: usize) -> Self {
            use std::iter::repeat_with;
            let len = 2 * n.next_power_of_two();
            Self {
                ss: repeat_with(T::id).take(len).collect(),
                fs: repeat_with(F::id).take(len).collect(),
            }
        }
        fn len(&self) -> usize {
            self.ss.len() / 2
        }
        fn propagate(&mut self, i: usize) {
            let h = 8 * std::mem::size_of::<usize>() as u32 - i.leading_zeros();
            for k in (1..h).rev() {
                let p = i >> k;
                let l = 2 * p;
                let r = 2 * p + 1;
                self.ss[l] = self.fs[p].map(std::mem::replace(&mut self.ss[l], T::id()));
                self.ss[r] = self.fs[p].map(std::mem::replace(&mut self.ss[r], T::id()));
                self.fs[l] = self.fs[p].op(&self.fs[l]);
                self.fs[r] = self.fs[p].op(&self.fs[r]);
                self.fs[p] = F::id();
            }
        }
        pub fn prod(&mut self, l: usize, r: usize) -> T {
            assert!(l <= r);
            assert!(r <= self.len());
            let mut l = l + self.len();
            let mut r = r + self.len();
            self.propagate(l >> l.trailing_zeros());
            self.propagate((r >> r.trailing_zeros()) - 1);
            let mut lv = T::id();
            let mut rv = T::id();
            while l < r {
                if l % 2 == 1 {
                    lv = lv.op(&self.ss[l]);
                    l += 1;
                }
                if r % 2 == 1 {
                    r -= 1;
                    rv = self.ss[r].op(&rv);
                }
                l /= 2;
                r /= 2;
            }
            lv.op(&rv)
        }
        pub fn set(&mut self, i: usize, v: T) {
            let mut i = i + self.len();
            self.propagate(i);
            self.ss[i] = v;
            while i > 1 {
                i /= 2;
                self.ss[i] = self.ss[2 * i].op(&self.ss[2 * i + 1]);
            }
        }
        pub fn apply(&mut self, l: usize, r: usize, f: &F) {
            assert!(l <= r);
            assert!(r <= self.len());
            let mut li = l + self.len();
            let mut ri = r + self.len();
            let ln = li >> li.trailing_zeros();
            let rn = ri >> ri.trailing_zeros();
            self.propagate(ln);
            self.propagate(rn - 1);
            while li < ri {
                if li % 2 == 1 {
                    self.fs[li] = f.op(&self.fs[li]);
                    self.ss[li] = f.map(std::mem::replace(&mut self.ss[li], T::id()));
                    li += 1;
                }
                if ri % 2 == 1 {
                    ri -= 1;
                    self.fs[ri] = f.op(&self.fs[ri]);
                    self.ss[ri] = f.map(std::mem::replace(&mut self.ss[ri], T::id()));
                }
                li /= 2;
                ri /= 2;
            }
            let mut l = (l + self.len()) / 2;
            let mut r = (r + self.len() - 1) / 2;
            while l > 0 {
                if l < ln {
                    self.ss[l] = self.ss[2 * l].op(&self.ss[2 * l + 1]);
                }
                if r < rn - 1 {
                    self.ss[r] = self.ss[2 * r].op(&self.ss[2 * r + 1]);
                }
                l /= 2;
                r /= 2;
            }
        }
    }
    impl<T: Monoid, F: Monoid + Map<T>> std::iter::FromIterator<T> for LazySegTree<T, F> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            let mut ss: Vec<_> = iter.into_iter().collect();
            let iter_n = ss.len();
            let n = iter_n.next_power_of_two();
            ss.splice(..0, std::iter::repeat_with(T::id).take(n));
            ss.extend(std::iter::repeat_with(T::id).take(n - iter_n));
            debug_assert_eq!(ss.len(), 2 * n);
            for i in (1..n).rev() {
                ss[i] = ss[2 * i].op(&ss[2 * i + 1]);
            }
            Self {
                ss: ss.into(),
                fs: std::iter::repeat_with(F::id).take(2 * n).collect(),
            }
        }
    }
}

// ============================ The only trustable segment tree code lol ===================
// Tracking max suffix sum (useful for tree based problem)
// Also a good template for changing the lazy prop + merge node.

// let mut st: Vec<Node> = vec![
//     Node {
//         max_suf_sum: -1000000000000000000,
//         sum: 0,
//         lz: false
//     };
//     4 * n + 20
// ];
// build(&mut st, 1, 1, n);
// to update: update_plus(&mut st, 1, 1, n, label[u], 1);
// update_set(&mut st, 1, 1, n, label[u], max_label[u]);
// let st_ans = tree_query(st, 1, 1, n, label[cur], label[cur]);

#[derive(Copy, Clone, Debug)]
struct Node {
    max_suf_sum: i64, // value
    sum: i64,         // size
    lz: bool,         // is lazy
}

// Build for node [ql..qr]
fn build(tree: &mut Vec<Node>, cc: usize, tl: usize, tr: usize) {
    if tl > tr {
        return;
    }
    tree[cc].max_suf_sum = -1;
    tree[cc].sum = -(tr as i64 - tl as i64 + 1);
    tree[cc].lz = false;
    if tl == tr {
        return;
    }
    let mid = (tl + tr) / 2;
    build(tree, cc * 2, tl, mid);
    build(tree, cc * 2 + 1, mid + 1, tr);
}

fn push(tree: &mut Vec<Node>, cc: usize, ql: usize, qr: usize) {
    // Has lazy val
    if ql > qr {
        return;
    }
    if tree[cc].lz {
        let mid = (ql + qr) / 2;
        tree[cc].lz = false;
        tree[cc * 2].lz = true;
        tree[cc * 2 + 1].lz = true;

        // Update the tree correctly: when set -1 for bunch of nodes (has lz), always update correct sum.
        tree[cc * 2].max_suf_sum = -1;
        tree[cc * 2].sum = -(mid as i64 - ql as i64 + 1);

        tree[cc * 2 + 1].max_suf_sum = -1;
        tree[cc * 2 + 1].sum = -(qr as i64 - mid as i64);
    }
}

fn merge(lc: Node, rc: Node) -> Node {
    Node {
        max_suf_sum: max(lc.max_suf_sum + rc.sum, rc.max_suf_sum),
        sum: lc.sum + rc.sum,
        lz: false,
    }
}

// Only allow +val in a single position (no lazy).
fn update_plus(tree: &mut Vec<Node>, cc: usize, ql: usize, qr: usize, position: usize, val: i64) {
    if ql == qr {
        tree[cc].max_suf_sum += val;
        tree[cc].sum += val;
        return;
    }
    push(tree, cc, ql, qr);
    let mid = (ql + qr) / 2;
    if position <= mid {
        update_plus(tree, cc * 2, ql, mid, position, val);
    } else {
        update_plus(tree, cc * 2 + 1, mid + 1, qr, position, val);
    }
    tree[cc] = merge(tree[cc * 2], tree[cc * 2 + 1]);
}

// Set with lazy: only allow set -1 so no val.
fn update_set(tree: &mut Vec<Node>, cc: usize, ql: usize, qr: usize, l: usize, r: usize) {
    if l > r {
        return;
    }
    if ql == l && qr == r {
        tree[cc].lz = true;
        tree[cc].max_suf_sum = -1;
        tree[cc].sum = -(qr as i64 - ql as i64 + 1);
        return;
    }
    push(tree, cc, ql, qr);
    let mid = (ql + qr) / 2;
    if r <= mid {
        update_set(tree, cc * 2, ql, mid, l, r);
    } else if l > mid {
        update_set(tree, cc * 2 + 1, mid + 1, qr, l, r);
    } else {
        update_set(tree, cc * 2, ql, mid, l, mid);
        update_set(tree, cc * 2 + 1, mid + 1, qr, mid + 1, r);
    }
    tree[cc] = merge(tree[cc * 2], tree[cc * 2 + 1]);
}

fn tree_query(tree: &mut Vec<Node>, cc: usize, ql: usize, qr: usize, l: usize, r: usize) -> Node {
    if ql == l && qr == r {
        return tree[cc];
    }
    push(tree, cc, ql, qr);
    let mid = (ql + qr) / 2;
    if r <= mid {
        return tree_query(tree, cc * 2, ql, mid, l, r);
    } else if l > mid {
        return tree_query(tree, cc * 2 + 1, mid + 1, qr, l, r);
    } else {
        let left_ans = tree_query(tree, cc * 2, ql, mid, l, mid);
        let right_ans = tree_query(tree, cc * 2 + 1, mid + 1, qr, mid + 1, r);
        return merge(left_ans, right_ans);
    }
}
