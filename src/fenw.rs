// Simple Fenwick tree for sum case
// Copy + modify should needed

#[allow(dead_code)]
struct FenwickTree {
    len: usize,
    bit: Vec<i32>,
}

#[allow(dead_code)]
impl FenwickTree {
    pub fn new(n: usize) -> Self {
        FenwickTree {
            len: n,
            bit: vec![0; n + 10],
        }
    }

    // Sum range [0..r]
    pub fn sum_full(&self, r: usize) -> i32 {
        let mut r = r as i32;
        let mut ret = 0;
        while r >= 0 {
            ret += self.bit[r as usize];
            r = (r & (r + 1)) - 1;
        }
        ret
    }

    // Sum range [l..r]
    // Usage: sum(1..=3) or sum(..10) or (7..)
    pub fn sum<R: std::ops::RangeBounds<usize>>(&self, range: R) -> i32 {
        let start: usize = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end: usize = match range.end_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x - 1,
            std::ops::Bound::Unbounded => self.len,
        };
        self.sum_full(end)
            - if start == 0 {
                0
            } else {
                self.sum_full(start - 1)
            }
    }

    // Single add
    pub fn add(&mut self, i: usize, delta: i32) {
        let mut i = i;
        while i <= self.len {
            self.bit[i] += delta;
            i = i | (i + 1);
        }
    }
}

#[allow(dead_code)]
struct FenwickTreeRangeAdd {
    len: usize,
    bit1: Vec<i32>,
    bit2: Vec<i32>,
}

#[allow(dead_code)]
impl FenwickTreeRangeAdd {
    pub fn new(n: usize) -> Self {
        FenwickTreeRangeAdd {
            len: n,
            bit1: vec![0; n + 10],
            bit2: vec![0; n + 10],
        }
    }

    // Single add
    fn add(n: usize, b: &mut Vec<i32>, i: usize, delta: i32) {
        let mut i = i as i32;
        while i <= n as i32 {
            b[i as usize] += delta;
            i += i & -i;
        }
    }

    // Range add
    pub fn range_add<R: std::ops::RangeBounds<usize>>(&mut self, range: R, x: i32) {
        let start: usize = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end: usize = match range.end_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x - 1,
            std::ops::Bound::Unbounded => self.len,
        };
        Self::add(self.len, &mut self.bit1, start, x);
        Self::add(self.len, &mut self.bit1, end + 1, -x);
        Self::add(self.len, &mut self.bit2, start, x * (start as i32 - 1));
        Self::add(self.len, &mut self.bit2, end + 1, -x * (end as i32));
    }

    // Sum range [0..r]
    fn sum_full(b: &Vec<i32>, r: usize) -> i32 {
        let mut r = r as i32;
        let mut ret = 0;
        while r > 0 {
            ret += b[r as usize];
            r -= r & -r;
        }
        ret
    }

    pub fn prefix_sum(&self, idx: usize) -> i32 {
        Self::sum_full(&self.bit1, idx) * (idx as i32) - Self::sum_full(&self.bit2, idx)
    }

    // Sum range [l..r]
    // Usage: sum(1..=3) or sum(..10) or (7..)
    pub fn sum<R: std::ops::RangeBounds<usize>>(&self, range: R) -> i32 {
        let start: usize = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end: usize = match range.end_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x - 1,
            std::ops::Bound::Unbounded => self.len,
        };
        return self.prefix_sum(end)
            - if start == 0 {
                0
            } else {
                self.prefix_sum(start - 1)
            };
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_fw_single() {
        let mut fw = FenwickTree::new(10);
        fw.add(1, 1);
        fw.add(2, 2);
        fw.add(10, 1);
        assert_eq!(fw.sum(..), 4);
        assert_eq!(fw.sum(..=2), 3);
        assert_eq!(fw.sum(..2), 1);
        assert_eq!(fw.sum(..1), 0);
        assert_eq!(fw.sum(1..=2), 3);
        assert_eq!(fw.sum(2..=2), 2);
        assert_eq!(fw.sum(2..), 3);
    }

    #[test]
    fn test_fw_range() {
        let mut fw = FenwickTreeRangeAdd::new(10);
        fw.range_add(2..=2, 1);
        assert_eq!(fw.prefix_sum(5), 1);
        assert_eq!(fw.prefix_sum(2), 1);
        fw.range_add(3..=3, 1);
        assert_eq!(fw.prefix_sum(5), 2);
        assert_eq!(fw.prefix_sum(2), 1);
        assert_eq!(fw.sum(..), 2);
        assert_eq!(fw.sum(2..=3), 2);
        assert_eq!(fw.sum(2..=2), 1);
        fw.range_add(3..=4, 2);
        assert_eq!(fw.sum(..), 6);
        assert_eq!(fw.sum(3..=3), 3);
        fw.range_add(7..=10, 2);
        assert_eq!(fw.sum(..), 14);
        assert_eq!(fw.sum(3..=3), 3);
    }
}
