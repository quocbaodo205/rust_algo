// Simple Fenwick tree for sum case
// Copy + modify should needed

#[allow(dead_code)]
struct FenwickTree {
    len: usize,
    bit: Vec<i64>,
}

#[allow(dead_code)]
impl FenwickTree {
    pub fn new(n: usize) -> Self {
        FenwickTree {
            len: n,
            bit: vec![0; n],
        }
    }

    // Sum range [0..r]
    pub fn sum_full(&self, r: i32) -> i64 {
        let mut r: i32 = r as i32;
        let mut ret = 0;
        while r >= 0 {
            ret += self.bit[r as usize];
            r = (r & (r + 1)) - 1;
        }
        ret
    }

    // Sum range [l..r]
    // Usage: sum(1..=3) or sum(..10) or (7..)
    pub fn sum<R: std::ops::RangeBounds<i32>>(&self, range: R) -> i64 {
        let start: i32 = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end: i32 = match range.end_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x - 1,
            std::ops::Bound::Unbounded => self.len as i32,
        };
        self.sum_full(end) - self.sum_full(start - 1)
    }

    // Single add
    pub fn add(&mut self, i: i32, delta: i64) {
        let mut i = i;
        while i < self.len as i32 {
            self.bit[i as usize] += delta;
            i = i | (i + 1);
        }
    }
}

// TODO: Add test cases
