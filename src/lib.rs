use std::cmp::{max, min};

#[allow(dead_code)]
fn get_log_2(x: u32) -> u32 {
    let mut count = 0;
    let mut x = x;
    while x > 0 {
        x /= 2;
        count += 1;
    }
    count
}

pub mod comb;
pub mod dsu;
pub mod fenw;
pub mod game;
pub mod modint;
pub mod number_theory;
pub mod prime;
pub mod rng;
pub mod root_tree;
pub mod segment_tree;
pub mod string;
pub mod trie;

// Segment tree implementation that support single place modification
// All input range need to be inclusive and 0-indexed
// Accept simple type that implement Copy and Clone
// Like i32, u32, i64, ...
#[allow(dead_code)]
struct SegmentTree<'a, T: Copy + Clone> {
    len: usize,
    tree: Vec<T>,
    default: T,
    // Function to tranform a[i] into something before insert in the tree
    transform: &'a dyn Fn(T) -> T,
    // Function to combine left and right branch (ex: sum, max, min, ...)
    combine: &'a dyn Fn(T, T) -> T,
    // Update function
    // Can be set val, add val, ...
    update_fn: &'a dyn Fn(T, T) -> T,
}

#[allow(dead_code)]
impl<'a, T: Copy + Clone> SegmentTree<'a, T> {
    pub fn new(
        n: usize,
        default: T,
        transform: &'a dyn Fn(T) -> T,
        combine: &'a dyn Fn(T, T) -> T,
        update_fn: &'a dyn Fn(T, T) -> T,
    ) -> Self {
        SegmentTree {
            len: n,
            default,
            tree: vec![default; 4 * n + 1],
            transform,
            combine,
            update_fn,
        }
    }

    fn build_internal(&mut self, node: usize, tl: usize, tr: usize, a: &[T]) {
        if tl == tr {
            self.tree[node] = (*self.transform)(a[tl]);
        } else {
            let mid = (tl + tr) / 2;
            self.build_internal(node * 2, tl, mid, a);
            self.build_internal(node * 2 + 1, mid + 1, tr, a);
            self.tree[node] = (*self.combine)(self.tree[node * 2], self.tree[node * 2 + 1]);
        }
    }

    pub fn build(
        a: &[T],
        default: T,
        transform: &'a dyn Fn(T) -> T,
        combine: &'a dyn Fn(T, T) -> T,
        update_fn: &'a dyn Fn(T, T) -> T,
    ) -> Self {
        let mut segment_tree = SegmentTree {
            len: a.len(),
            default,
            tree: vec![default; 4 * a.len() + 1],
            transform,
            combine,
            update_fn,
        };
        segment_tree.build_internal(1, 0, a.len() - 1, a);
        segment_tree
    }

    fn query_internal(&mut self, node: usize, tl: usize, tr: usize, ql: usize, qr: usize) -> T {
        if ql > qr {
            return self.default;
        }
        if tl == ql && tr == qr {
            return self.tree[node];
        }
        let mid = (tl + tr) / 2;
        (*self.combine)(
            self.query_internal(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        )
    }

    // Query the inclusive range [l..r]
    pub fn query(&mut self, l: usize, r: usize) -> T {
        self.query_internal(1, 0, self.len - 1, l, r)
    }

    // When needed, please apply transformation before passing [val]
    fn update_internal(&mut self, node: usize, tl: usize, tr: usize, pos: usize, val: T) {
        if tl == tr {
            self.tree[node] = (*self.update_fn)(self.tree[node], val);
        } else {
            let mid = (tl + tr) / 2;
            if pos <= mid {
                self.update_internal(node * 2, tl, mid, pos, val);
            } else {
                self.update_internal(node * 2 + 1, mid + 1, tr, pos, val);
            }
            self.tree[node] = (*self.combine)(self.tree[node * 2], self.tree[node * 2 + 1]);
        }
    }

    // Update the array at position [pos] into [val]
    pub fn update(&mut self, pos: usize, val: T) {
        self.update_internal(1, 0, self.len - 1, pos, val);
    }
}

#[allow(dead_code)]
struct SegmentTreeLazy<'a, T: Copy + Clone> {
    len: usize,
    tree: Vec<T>,
    lazy: Vec<T>,
    default: T,
    lazy_default: T,
    // Function to tranform a[i] into something before insert in the tree
    transform: &'a dyn Fn(T) -> T,
    // Function to combine left and right branch (ex: sum, max, min, ...)
    combine: &'a dyn Fn(T, T) -> T,
    // Combine [current value] with the [lazy value]
    // Input: (current, lazy)
    // ex: sum, assign, ...
    lazy_combine: &'a dyn Fn(T, T) -> T,
    // Update function
    // Can be set val, add val, ...
    update_fn: &'a dyn Fn(T, T) -> T,
}

#[allow(dead_code)]
impl<'a, T: Copy + Clone> SegmentTreeLazy<'a, T> {
    pub fn new(
        n: usize,
        default: T,
        lazy_default: T,
        transform: &'a dyn Fn(T) -> T,
        combine: &'a dyn Fn(T, T) -> T,
        lazy_combine: &'a dyn Fn(T, T) -> T,
        update_fn: &'a dyn Fn(T, T) -> T,
    ) -> Self {
        SegmentTreeLazy {
            len: n,
            tree: vec![default; 4 * n + 1],
            lazy: vec![lazy_default; 4 * n + 1],
            default,
            lazy_default,
            transform,
            combine,
            lazy_combine,
            update_fn,
        }
    }

    fn build_internal(&mut self, node: usize, tl: usize, tr: usize, a: &[T]) {
        if tl == tr {
            self.tree[node] = (*self.transform)(a[tl]);
        } else {
            let mid = (tl + tr) / 2;
            self.build_internal(node * 2, tl, mid, a);
            self.build_internal(node * 2 + 1, mid + 1, tr, a);
            self.tree[node] = (*self.combine)(self.tree[node * 2], self.tree[node * 2 + 1]);
        }
    }

    pub fn build(
        a: &[T],
        default: T,
        lazy_default: T,
        transform: &'a dyn Fn(T) -> T,
        combine: &'a dyn Fn(T, T) -> T,
        lazy_combine: &'a dyn Fn(T, T) -> T,
        update_fn: &'a dyn Fn(T, T) -> T,
    ) -> Self {
        let mut segment_tree = SegmentTreeLazy {
            len: a.len(),
            default,
            lazy_default,
            tree: vec![default; 4 * a.len() + 1],
            lazy: vec![lazy_default; 4 * a.len() + 1],
            transform,
            combine,
            lazy_combine,
            update_fn,
        };
        segment_tree.build_internal(1, 0, a.len() - 1, a);
        segment_tree
    }

    // Push all updates to chilren nodes
    // Doesn't mess with the current tree node
    // Only reset the current lazy node
    fn push(&mut self, node: usize, _tl: usize, _tr: usize) {
        // Update tree
        self.tree[node * 2] = (*self.lazy_combine)(self.tree[node * 2], self.lazy[node]);
        self.tree[node * 2 + 1] = (*self.lazy_combine)(self.tree[node * 2 + 1], self.lazy[node]);

        // Update lazy
        self.lazy[node * 2] = (*self.update_fn)(self.lazy[node * 2], self.lazy[node]);
        self.lazy[node * 2 + 1] = (*self.update_fn)(self.lazy[node * 2 + 1], self.lazy[node]);

        // Reset current lazy
        self.lazy[node] = self.lazy_default;
    }

    fn query_internal(&mut self, node: usize, tl: usize, tr: usize, ql: usize, qr: usize) -> T {
        if ql > qr {
            return self.default;
        }
        if tl == ql && tr == qr {
            return self.tree[node];
        }
        self.push(node, tl, tr);
        let mid = (tl + tr) / 2;
        (*self.combine)(
            self.query_internal(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        )
    }

    // Query the inclusive range [l..r]
    pub fn query(&mut self, l: usize, r: usize) -> T {
        self.query_internal(1, 0, self.len - 1, l, r)
    }

    // When needed, please apply transformation before passing [val]
    fn update_internal(&mut self, node: usize, tl: usize, tr: usize, ql: usize, qr: usize, val: T) {
        eprintln!("Range is {},{} | {},{}", tl, tr, ql, qr);
        if ql > qr {
            eprintln!("free");
            return;
        }
        if tl == ql && tr == qr {
            eprintln!("Update perfect range");
            self.tree[node] = (*self.update_fn)(self.tree[node], val);
            self.lazy[node] = (*self.update_fn)(self.lazy[node], val);
        } else {
            self.push(node, tl, tr);
            let mid = (tl + tr) / 2;
            self.update_internal(node * 2, tl, mid, ql, min(qr, mid), val);
            self.update_internal(node * 2 + 1, mid + 1, tr, max(ql, mid + 1), qr, val);
            self.tree[node] = (*self.combine)(self.tree[node * 2], self.tree[node * 2 + 1]);
        }
    }

    // Update the array at position [pos] into [val]
    pub fn update(&mut self, ql: usize, qr: usize, val: T) {
        self.update_internal(1, 0, self.len - 1, ql, qr, val);
    }
}

#[cfg(test)]
mod test {
    use std::cmp::max;

    use crate::SegmentTreeLazy;

    use super::SegmentTree;

    #[test]
    fn test_segment_tree() {
        let a = [1, 2, 3, 4, 5, 6];

        // Func to combine 2 range
        fn sum(x: i32, y: i32) -> i32 {
            x + y
        }

        // Func to transform a value
        fn trans(x: i32) -> i32 {
            x
        }

        // Func to update (set)
        fn update(_x: i32, y: i32) -> i32 {
            y
        }

        let mut sum_tree = SegmentTree::<i32>::build(&a[..], 0, &trans, &sum, &update);

        // Check query for sum
        assert_eq!(sum_tree.query(0, 2), 1 + 2 + 3);
        assert_eq!(sum_tree.query(3, 5), 4 + 5 + 6);
        assert_eq!(sum_tree.query(2, 3), 3 + 4);

        // Check query after change val
        sum_tree.update(2, 10);
        assert_eq!(sum_tree.query(0, 2), 1 + 2 + 10);
        assert_eq!(sum_tree.query(3, 5), 4 + 5 + 6);
        assert_eq!(sum_tree.query(2, 3), 10 + 4);

        let a = [3, 1, 2, 5, 4, 6];
        let mut max_tree = SegmentTree::<i32>::build(&a[..], 0, &trans, &max::<i32>, &update);

        // Check query for max
        assert_eq!(max_tree.query(0, 2), 3);
        assert_eq!(max_tree.query(3, 5), 6);
        assert_eq!(max_tree.query(2, 3), 5);

        // Check query after change val
        max_tree.update(2, 7);
        assert_eq!(max_tree.query(0, 2), 7);
        assert_eq!(max_tree.query(3, 5), 6);
        assert_eq!(max_tree.query(2, 3), 7);
        assert_eq!(max_tree.query(0, 1), 3);
        assert_eq!(max_tree.query(4, 5), 6);

        // Segment tree test for counting problem
        // In this problem, we will be counting 0s
        fn t(x: i32) -> i32 {
            if x == 0 {
                1
            } else {
                0
            }
        }

        let a = [1, 2, 0, 4, 4, 0, 5, 0, 0];
        let mut zero_tree = SegmentTree::<i32>::build(&a[..], 0, &t, &sum, &update);

        // Check query
        assert_eq!(zero_tree.query(0, 2), 1);
        assert_eq!(zero_tree.query(1, 5), 2);
        assert_eq!(zero_tree.query(0, 8), 4);
        assert_eq!(zero_tree.query(6, 8), 2);

        // Check query after change val
        // Change a[4] = 0, but transform before putting inside
        // Better logic control when updating value instead of fixing the transform.
        zero_tree.update(4, t(0));
        // Check query
        assert_eq!(zero_tree.query(0, 2), 1);
        assert_eq!(zero_tree.query(1, 5), 3);
        assert_eq!(zero_tree.query(0, 8), 5);
        assert_eq!(zero_tree.query(6, 8), 2);

        // Change a[2] = 10
        zero_tree.update(2, t(10));
        assert_eq!(zero_tree.query(0, 2), 0);
        assert_eq!(zero_tree.query(1, 5), 2);
        assert_eq!(zero_tree.query(0, 8), 4);
        assert_eq!(zero_tree.query(6, 8), 2);

        // Example tree with add single, get max
        let a = [1, 2, 3, 4, 5, 6];
        let mut add_max = SegmentTree::<i32>::build(&a[..], 0, &trans, &max::<i32>, &sum);

        // Check query for sum
        assert_eq!(add_max.query(0, 2), 3);
        assert_eq!(add_max.query(3, 5), 6);
        assert_eq!(add_max.query(2, 3), 4);

        // Check query after change val
        add_max.update(2, 10);
        assert_eq!(add_max.query(0, 2), 13);
        assert_eq!(add_max.query(3, 5), 6);
        assert_eq!(add_max.query(2, 3), 13);
    }

    #[test]
    fn test_segment_tree_lazy() {
        let a = [1, 2, 3, 4, 5, 6];

        fn sum(x: i32, y: i32) -> i32 {
            x + y
        }

        fn trans(x: i32) -> i32 {
            x
        }

        fn update(_x: i32, y: i32) -> i32 {
            y
        }

        // Example segment tree lazy for set range sum range
        let mut set_sum_tree_lazy =
            SegmentTreeLazy::<i32>::build(&a[..], 0, 0, &trans, &sum, &sum, &update);

        // Check query for sum
        assert_eq!(set_sum_tree_lazy.query(0, 2), 1 + 2 + 3);
        assert_eq!(set_sum_tree_lazy.query(3, 5), 4 + 5 + 6);
        assert_eq!(set_sum_tree_lazy.query(2, 3), 3 + 4);

        // Check query after change val (single)
        set_sum_tree_lazy.update(2, 2, 10);
        assert_eq!(set_sum_tree_lazy.query(0, 2), 1 + 2 + 10);
        assert_eq!(set_sum_tree_lazy.query(3, 5), 4 + 5 + 6);
        assert_eq!(set_sum_tree_lazy.query(2, 3), 10 + 4);
        set_sum_tree_lazy.update(2, 2, 3);
        assert_eq!(set_sum_tree_lazy.query(0, 2), 1 + 2 + 3);
        assert_eq!(set_sum_tree_lazy.query(3, 5), 4 + 5 + 6);
        assert_eq!(set_sum_tree_lazy.query(2, 3), 3 + 4);

        // Check query after change val (range)
        set_sum_tree_lazy.update(2, 3, 10);
        // Array become [1,2,10,10,5,6]
        assert_eq!(set_sum_tree_lazy.query(0, 2), 1 + 2 + 10);
        assert_eq!(set_sum_tree_lazy.query(3, 5), 10 + 5 + 6);
        assert_eq!(set_sum_tree_lazy.query(2, 3), 10 + 10);

        // Example segment tree for add range max range
        let a = [1, 2, 3, 4, 5, 6];
        let mut add_max_tree_lazy =
            SegmentTreeLazy::<i32>::build(&a[..], 0, 0, &trans, &max::<i32>, &sum, &sum);
        assert_eq!(add_max_tree_lazy.query(0, 2), 3);
        assert_eq!(add_max_tree_lazy.query(3, 5), 6);
        assert_eq!(add_max_tree_lazy.query(2, 3), 4);

        // Check query after add val (single)
        add_max_tree_lazy.update(2, 2, 2);
        // Array become [1,2,5,4,5,6]
        assert_eq!(add_max_tree_lazy.query(0, 2), 5);
        assert_eq!(add_max_tree_lazy.query(3, 5), 6);
        assert_eq!(add_max_tree_lazy.query(2, 3), 5);
        add_max_tree_lazy.update(2, 2, -2);
        assert_eq!(add_max_tree_lazy.query(0, 2), 3);
        assert_eq!(add_max_tree_lazy.query(3, 5), 6);
        assert_eq!(add_max_tree_lazy.query(2, 3), 4);

        // Check query after add val (range)
        add_max_tree_lazy.update(2, 3, 3);
        // Array become [1,2,6,7,5,6]
        assert_eq!(add_max_tree_lazy.query(0, 2), 6);
        assert_eq!(add_max_tree_lazy.query(3, 5), 7);
        assert_eq!(add_max_tree_lazy.query(2, 3), 7);
        add_max_tree_lazy.update(2, 3, -3);
        assert_eq!(add_max_tree_lazy.query(0, 2), 3);
        assert_eq!(add_max_tree_lazy.query(3, 5), 6);
        assert_eq!(add_max_tree_lazy.query(2, 3), 4);
    }
}
