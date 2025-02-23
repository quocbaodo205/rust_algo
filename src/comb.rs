use std::ops;

#[derive(Clone, Copy, Debug)]
pub struct ModInt {
    val: u64,
    m: u64,
}

impl ops::Add for ModInt {
    type Output = ModInt;

    fn add(self, rhs: ModInt) -> Self::Output {
        ModInt {
            val: (self.val + rhs.val) % self.m,
            m: self.m,
        }
    }
}

impl ops::Add<u64> for ModInt {
    type Output = ModInt;

    fn add(self, rhs: u64) -> Self::Output {
        ModInt {
            val: (self.val + rhs) % self.m,
            m: self.m,
        }
    }
}

impl ops::Sub for ModInt {
    type Output = ModInt;

    fn sub(self, rhs: ModInt) -> Self::Output {
        ModInt {
            val: (self.val + self.m - rhs.val) % self.m,
            m: self.m,
        }
    }
}

impl ops::Mul for ModInt {
    type Output = ModInt;

    fn mul(self, rhs: ModInt) -> Self::Output {
        ModInt {
            val: (self.val * rhs.val) % self.m,
            m: self.m,
        }
    }
}

impl ops::MulAssign for ModInt {
    fn mul_assign(&mut self, rhs: Self) {
        self.val = (self.val * rhs.val) % self.m;
    }
}

impl ops::Mul<u64> for ModInt {
    type Output = ModInt;

    fn mul(self, rhs: u64) -> Self::Output {
        ModInt {
            val: (self.val * rhs) % self.m,
            m: self.m,
        }
    }
}

impl ops::Div for ModInt {
    type Output = ModInt;

    // Mul with the inverse
    fn div(self, rhs: Self) -> Self::Output {
        rhs * power(self, self.m - 2)
    }
}

impl ops::Div<ModInt> for u64 {
    type Output = ModInt;

    fn div(self, rhs: ModInt) -> Self::Output {
        ModInt {
            val: self * power(rhs, rhs.m - 2).val,
            m: rhs.m,
        }
    }
}

pub fn power(a: ModInt, b: u64) -> ModInt {
    let mut res = ModInt { val: 1, m: a.m };
    let mut x = a;
    let mut cur = b;
    while cur > 0 {
        if cur % 2 == 1 {
            res *= x;
        }
        x *= x;
        cur /= 2;
    }
    res
}

pub struct CombMod {
    fac: Vec<ModInt>,
    invfac: Vec<ModInt>,
    _inv: Vec<ModInt>,
    m: u64,
}

impl CombMod {
    pub fn new(n: usize, m: u64) -> Self {
        let mut f: Vec<ModInt> = vec![ModInt { val: 1, m }; n + 1];
        let mut inv: Vec<ModInt> = vec![ModInt { val: 1, m }; n + 1];
        let mut invf: Vec<ModInt> = vec![ModInt { val: 0, m }; n + 1];
        (1..f.len()).for_each(|i| f[i] = f[i - 1] * i as u64);
        invf[n] = 1 / f[n];
        (1..inv.len()).rev().for_each(|i| {
            invf[i - 1] = invf[i] * i as u64;
            inv[i] = invf[i] * f[i - 1];
        });
        CombMod {
            fac: f,
            invfac: invf,
            _inv: inv,
            m,
        }
    }

    pub fn binom(&self, n: usize, k: usize) -> ModInt {
        match n < k {
            true => ModInt { val: 0, m: self.m },
            false => self.fac[n] * self.invfac[k] * self.invfac[n - k],
        }
    }
}

// ================================================

pub struct Combinatoric {
    fac: Vec<u64>,
    invfac: Vec<u64>,
    _inv: Vec<u64>,
    m: u64,
}

impl Combinatoric {
    pub fn pow(a: u64, b: u64, m: u64) -> u64 {
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

    pub fn inverse_mod(x: u64, m: u64) -> u64 {
        Self::pow(x, m - 2, m)
    }

    pub fn new(n: usize, m: u64) -> Self {
        let mut f: Vec<u64> = vec![1; n + 1];
        let mut inv: Vec<u64> = vec![1; n + 1];
        let mut invf: Vec<u64> = vec![0; n + 1];
        (1..f.len()).for_each(|i| f[i] = (f[i - 1] * i as u64) % m);
        invf[n] = Self::inverse_mod(f[n], m);
        (1..inv.len()).rev().for_each(|i| {
            invf[i - 1] = (invf[i] * i as u64) % m;
            inv[i] = (invf[i] * f[i - 1]) % m;
        });
        Combinatoric {
            fac: f,
            invfac: invf,
            _inv: inv,
            m,
        }
    }

    pub fn binom(&self, n: usize, k: usize) -> u64 {
        match n < k {
            true => 0,
            false => (((self.fac[n] * self.invfac[k]) % self.m) * self.invfac[n - k]) % self.m,
        }
    }
}

#[allow(dead_code)]
// Find combination of <num> positions. Useful for include / exclude.
fn combination(
    v: &Vec<usize>,
    idx: usize,
    num: usize,
    cur_st: &mut Vec<usize>,
    cur_ans: usize,
    ans: &mut Vec<usize>,
) {
    if cur_st.len() == num {
        ans.push(cur_ans);
        return;
    }
    (idx..v.len()).for_each(|i| {
        cur_st.push(i);
        combination(v, i + 1, num, cur_st, cur_ans * v[i], ans);
        cur_st.pop();
    });
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pow() {
        let m = 1000000007u64;
        assert_eq!(Combinatoric::pow(2, 3, m), 8u64);
        assert_eq!(Combinatoric::pow(2, 10, m), 1024u64);
        assert_eq!(Combinatoric::pow(3, 15, m), 14348907u64);

        assert_eq!(power(ModInt { val: 2, m }, 3).val, 8u64);
        assert_eq!(power(ModInt { val: 2, m }, 10).val, 1024u64);
        assert_eq!(power(ModInt { val: 3, m }, 15).val, 14348907u64);
    }

    #[test]
    fn test_fac() {
        let comb = Combinatoric::new(200000, 1000000007);
        assert_eq!(comb.fac[3], 3u64 * 2);
        assert_eq!(comb.fac[5], 5u64 * 4 * 3 * 2);

        let combm = CombMod::new(200000, 1000000007);
        assert_eq!(combm.fac[3].val, 3u64 * 2);
        assert_eq!(combm.fac[5].val, 5u64 * 4 * 3 * 2);
    }
}
