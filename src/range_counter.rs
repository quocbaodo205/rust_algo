use std::collections::VecDeque;

/// A collection of useful range counter.
/// Support update the val when shrink/extend the counted range 1 by 1.
/// Very useful for many D&C DP problems, as range moving + D&C DP approach make the cost funtion amortized O(1).

/// A range counter base on positions.
/// Often have to store these position in a deque to allow calculation of value.
/// Currently, val is calculated as sum(all x | last_pos(x)-first_pos(x))
pub struct RangePosCounter {
    pos_stores: Vec<VecDeque<usize>>,
    a: Vec<usize>,
    l: usize,
    r: usize,
    val: usize,
}

impl RangePosCounter {
    pub fn new(
        pos_stores: Vec<VecDeque<usize>>,
        a: Vec<usize>,
        l: usize,
        r: usize,
        val: usize,
    ) -> Self {
        RangePosCounter {
            pos_stores,
            a,
            l,
            r,
            val,
        }
    }

    pub fn move_left_left(&mut self) {
        self.l -= 1;
        unsafe {
            let x = *self.a.get_unchecked(self.l);
            if let Some(&old_first_pos) = self.pos_stores.get_unchecked(x).front() {
                self.val += old_first_pos - self.l;
            }
            self.pos_stores.get_unchecked_mut(x).push_front(self.l);
        }
    }

    pub fn move_left_right(&mut self) {
        unsafe {
            let x = *self.a.get_unchecked(self.l);
            self.pos_stores.get_unchecked_mut(x).pop_front();
            if let Some(new_first_pos) = self.pos_stores.get_unchecked(x).front() {
                self.val -= new_first_pos - self.l;
            }
        }
        self.l += 1;
    }

    pub fn move_right_left(&mut self) {
        unsafe {
            let x = *self.a.get_unchecked(self.r);
            self.pos_stores.get_unchecked_mut(x).pop_back();
            if let Some(new_last_pos) = self.pos_stores.get_unchecked(x).back() {
                self.val -= self.r - new_last_pos;
            }
        }
        self.r -= 1;
    }

    pub fn move_right_right(&mut self) {
        self.r += 1;
        unsafe {
            let x = *self.a.get_unchecked(self.r);
            if let Some(&old_last_pos) = self.pos_stores.get_unchecked(x).back() {
                self.val += self.r - old_last_pos;
            }
            self.pos_stores.get_unchecked_mut(x).push_back(self.r);
        }
    }

    pub fn to_range(&mut self, to_l: usize, to_r: usize) {
        while to_l < self.l {
            self.move_left_left();
        }
        while to_l > self.l {
            if self.l == self.r {
                self.move_right_right();
            }
            self.move_left_right();
        }
        while to_r > self.r {
            self.move_right_right();
        }
        while to_r < self.r {
            if self.l == self.r {
                self.move_left_left();
            }
            self.move_right_left();
        }
    }

    pub fn get_val(&self) -> usize {
        self.val
    }
}

/// Range counter based on the count of values.
/// Currently, val is calculated as sum(all x | (count[x] * count[x]-1) / 2)
pub struct RangeCounter {
    counter: Vec<usize>,
    a: Vec<usize>,
    l: usize,
    r: usize,
    val: u64,
}

impl RangeCounter {
    pub fn new(counter: Vec<usize>, a: Vec<usize>, l: usize, r: usize, val: u64) -> Self {
        RangeCounter {
            counter,
            a,
            l,
            r,
            val,
        }
    }

    pub fn get_val(&self) -> u64 {
        self.val
    }

    pub fn move_left_left(&mut self) {
        self.l -= 1;
        unsafe {
            let t = self
                .counter
                .get_unchecked_mut(*self.a.get_unchecked(self.l));
            self.val += *t as u64;
            *t += 1;
        }
    }

    pub fn move_left_right(&mut self) {
        unsafe {
            let t = self
                .counter
                .get_unchecked_mut(*self.a.get_unchecked(self.l));
            *t -= 1;
            self.val -= *t as u64;
        }
        self.l += 1;
    }

    pub fn move_right_left(&mut self) {
        unsafe {
            let t = self
                .counter
                .get_unchecked_mut(*self.a.get_unchecked(self.r));
            *t -= 1;
            self.val -= *t as u64;
        }
        self.r -= 1;
    }

    pub fn move_right_right(&mut self) {
        self.r += 1;
        unsafe {
            let t = self
                .counter
                .get_unchecked_mut(*self.a.get_unchecked(self.r));
            self.val += *t as u64;
            *t += 1;
        }
    }

    /// Move the current range to the correct new range
    pub fn to_range(&mut self, to_l: usize, to_r: usize) {
        while to_l < self.l {
            self.move_left_left();
        }
        while to_l > self.l {
            if self.l == self.r {
                self.move_right_right();
            }
            self.move_left_right();
        }
        while to_r > self.r {
            self.move_right_right();
        }
        while to_r < self.r {
            if self.l == self.r {
                self.move_left_left();
            }
            self.move_right_left();
        }
    }
}
