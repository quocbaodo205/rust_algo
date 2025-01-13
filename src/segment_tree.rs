use std::{
    cmp::{max, min},
    ops::{Add, AddAssign},
};

#[allow(dead_code)]
struct SegmentTreeMin {
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
        min(
            self.query_internal_min(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal_min(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        )
    }

    // Query the inclusive range [l..r]
    pub fn query_min(&mut self, l: usize, r: usize) -> usize {
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
struct SegmentTreeMax {
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
struct SegmentTreeIsInc {
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

// Template for segment tree that allow crazy shit

// Lazy operation that allow range add or range set
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SegtreeLazy {
    val: i64,
    // Is increase, if false then it's a set operation.
    // Remember to find all is_inc and change to false should needed.
    is_inc: bool,
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
struct SegtreeValue {
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

const LID: SegtreeLazy = SegtreeLazy {
    val: 0,
    is_inc: true,
};
const VID: SegtreeValue = SegtreeValue {
    sum: 0,
    mx: -1000000000000000000,
};

// ============================== Main structure ====================

#[allow(dead_code)]
struct SegmentTreeLazy {
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
