use std::ops::{Add, AddAssign};

/// Structure that represent the basic of the vector space Z(d,2)
/// Also keep track of the size and added size to count various things.
#[derive(Clone)]
pub struct XorBasic {
    pub bases: Vec<u64>,
    pub sz: usize,
    pub set_sz: usize,
}

fn power_mod(a: u64, b: u64, m: u64) -> u64 {
    let mut res = 1u64;
    let mut x = a;
    let mut cur = b;
    while cur > 0 {
        if cur % 2 == 1 {
            res = (res * x) % m;
        }
        x = (x * x) % m;
        cur /= 2;
    }
    res
}

impl XorBasic {
    /// Init basic with the number of dimension. Usually is the number of bit.
    pub fn new(d: usize) -> Self {
        XorBasic {
            bases: vec![0; d],
            sz: 0,
            set_sz: 0,
        }
    }

    /// Insert a vector (represent by mask) into the basic.
    pub fn insert(&mut self, mask: u64) {
        self.set_sz += 1;
        if self.sz == self.bases.len() {
            return; // Cannot add anymore.
        }
        let mut mask = mask;
        // f(mask) is the last position that has a bit set
        for i in (0..self.bases.len()).rev() {
            if (mask & (1 << i)) == 0 {
                // i != f(mask) case
                continue;
            }

            if self.bases[i] == 0 {
                // No basic vector with i-th bit set
                self.bases[i] = mask;
                self.sz += 1;
            }

            mask ^= self.bases[i]; // Subtract the basic vector from this vector.
        }
    }

    /// Insert the vector into the basic, with the step along the way on how to create this base.
    /// If you sumxor all of them, you will get back the mask.
    pub fn insert_with_steps(&mut self, mask: u64) -> Vec<u64> {
        let mut steps = Vec::new();
        self.set_sz += 1;
        if self.sz == self.bases.len() {
            return steps;
        }
        let mut mask = mask;
        // f(mask) is the last position that has a bit set
        for i in (0..self.bases.len()).rev() {
            if (mask & (1 << i)) == 0 {
                // i != f(mask) case
                continue;
            }

            if self.bases[i] == 0 {
                // No basic vector with i-th bit set
                self.bases[i] = mask;
                self.sz += 1;
            }

            steps.push(self.bases[i]);
            mask ^= self.bases[i]; // Subtract the basic vector from this vector.
        }
        steps
    }

    /// Check if a vector (represent by mask) is representable using the current basic.
    pub fn is_representable(&self, mask: u64) -> bool {
        if self.sz == self.bases.len() {
            return true;
        }
        let mut mask = mask;
        for i in (0..self.bases.len()).rev() {
            if (mask & (1 << i)) == 0 {
                // i != f(mask) case
                continue;
            }

            if self.bases[i] == 0 {
                // No basic vector with i-th bit set, then cannot be represent
                return false;
            }

            mask ^= self.bases[i]; // Subtract the basic vector from this vector.
        }
        return true;
    }

    /// Count the number of way to represent mask using the curent basic and set size.
    /// If it's representable, then the ans is 2^(set_sz - sz).
    pub fn count_representable(&self, mask: u64) -> u64 {
        if !self.is_representable(mask) {
            return 0;
        }
        return 1u64 << (self.set_sz - self.sz);
    }

    /// Count the number of way to represent mask using the curent basic and set size.
    /// If it's representable, then the ans is 2^(set_sz - sz). Input a modulo for fast calculate.
    pub fn count_representable_mod(&self, mask: u64, md: u64) -> u64 {
        if !self.is_representable(mask) {
            return 0;
        }
        return power_mod(2u64, (self.set_sz - self.sz) as u64, md);
    }

    /// Given a representable mask, recreate mask using sumxor of some basic.
    pub fn xor_steps(&mut self, mask: u64) -> Vec<u64> {
        let mut steps = Vec::new();
        let mut mask = mask;
        for i in (0..self.bases.len()).rev() {
            // that bit is 0, nothing to do.
            if (mask & (1 << i)) == 0 {
                // i != f(mask) case
                continue;
            }

            steps.push(self.bases[i]); // Use base i
            mask ^= self.bases[i]; // Subtract the basic vector from this vector.
        }
        steps
    }

    /// Get the maximum vector that is representable by the basic
    pub fn get_max(&self) -> u64 {
        let mut ans = 0;
        for i in (0..self.bases.len()).rev() {
            if self.bases[i] == 0 {
                continue; // No basic with this bit on.
            }
            if (ans & (1 << i)) == 0 {
                continue; // this bit is already on, no need.
            }

            ans ^= self.bases[i];
        }
        ans
    }

    /// Gauss-Jordan Elimination to turn all vector in the basic to single form,
    /// in which every key bit (the highest bit of a vector) on appear in 1 vector.
    /// This is useful for other DP calculation, since every bit is only present once
    /// and allow for sumxor to only inc the number of on bit.
    pub fn eliminate(&self) -> Vec<u64> {
        let mut op = self.bases.clone();
        for j in 0..self.sz {
            for i in 0..j {
                op[i] = op[i].min(op[i] ^ op[j]);
            }
        }
        op.sort();
        op.reverse();
        op
    }
}

// ============================== Utility calculator =======================

/// Count the number of subset of a so that sumxor is 0.
/// Go one by one and update the basic, base calculation on count_rep(a[i]) and count_0.
pub fn num_subset_zero(a: &[u64]) -> u64 {
    let md = 1000000007;
    let mut basic = XorBasic::new(32); // Assume 32 bit number
    let mut count_0 = 0u64;
    let mut ans = 0u64;
    for &mask in a.iter() {
        if mask == 0 {
            ans *= 2; // Combine with all previous 0 subset
            ans += 1; // Also itself
            ans %= md;
            count_0 += 1;
        } else {
            let c = basic.count_representable_mod(mask, md);
            let total = c * power_mod(2u64, count_0, md); // 2^count_0 subset with sumxor = 0
            ans += total; // Combine with anything that represent itself to make 0
            ans %= md;
            basic.insert(mask);
        }
    }
    ans
}

/// Count #ways to pick a subset, which sumxor has x bit.
pub fn count_representable_with_x_bit(a: &[u64]) -> Vec<u64> {
    let md = 1000000007;
    let m = 32;
    let mut basic = XorBasic::new(m);
    for &x in a.iter() {
        basic.insert(x);
    }

    // Elimination
    let el = basic.eliminate();
    // f[i]: how many ways (using the basic) to make a number with i bits
    let mut f = vec![0u64; m + 1];
    f[0] = 1;
    let mut cur = 0u64;
    for i in 1u64..(1 << basic.sz) {
        cur ^= el[i.trailing_zeros() as usize] as u64;
        f[cur.count_ones() as usize] += 1;
        f[cur.count_ones() as usize] %= md;
    }
    // #way to represent any number possible number using the set a
    let c = power_mod(2u64, (basic.set_sz - basic.sz) as u64, md);
    let mut ans = vec![c; m + 1];
    for i in 0..=m {
        ans[i] = (ans[i] * f[i]) % md;
    }
    ans
}
