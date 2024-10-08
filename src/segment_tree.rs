use std::cmp::{max, min};

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

// TODO: Change range function to accept a RangeBound like fenw
