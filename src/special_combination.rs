// Prime mod is allowed for Mint and FPS

use numeric_traits::Integer;
use static_modint::ModInt998244353 as Mint;
pub type FPS = formal_power_series::FormalPowerSeries998244353;

pub struct BinomialPrime {
    fact: Vec<Mint>,
    fact_inv: Vec<Mint>,
    inv: Vec<Mint>,
}

impl Default for BinomialPrime {
    fn default() -> Self {
        Self::new()
    }
}

impl BinomialPrime {
    pub fn new() -> Self {
        Self {
            fact: vec![Mint::from_raw(1); 2],
            fact_inv: vec![Mint::from_raw(1); 2],
            inv: vec![Mint::from_raw(1); 2],
        }
    }

    pub fn expand(&mut self, n: usize) {
        let prev_n = self.fact.len() - 1;
        if prev_n >= n {
            return;
        }

        let new_n = n.ceil_pow2().min(Mint::modulus() as usize - 1);
        if prev_n >= new_n {
            return;
        }

        self.fact.resize(new_n + 1, Mint::from_raw(0));
        self.fact_inv.resize(new_n + 1, Mint::from_raw(0));
        self.inv.resize(new_n + 1, Mint::from_raw(0));

        for i in prev_n + 1..=new_n {
            self.fact[i] = self.fact[i - 1] * Mint::from_raw(i as _);
        }
        self.fact_inv[new_n] = self.fact[new_n].recip();
        self.inv[new_n] = self.fact_inv[new_n] * self.fact[new_n - 1];
        for i in (prev_n + 1..new_n).rev() {
            self.fact_inv[i] = self.fact_inv[i + 1] * Mint::from_raw((i + 1) as _);
            self.inv[i] = self.fact_inv[i] * self.fact[i - 1];
        }
    }

    pub fn fact(&mut self, n: usize) -> Mint {
        self.expand(n);
        if n >= self.fact.len() {
            Mint::from_raw(0)
        } else {
            self.fact[n]
        }
    }

    pub fn fact_inv(&mut self, n: usize) -> Mint {
        self.expand(n);
        assert!(n < self.fact_inv.len(), "n! is 0");
        self.fact_inv[n]
    }

    pub fn inv(&mut self, n: usize) -> Mint {
        self.expand(n);
        let n = n % Mint::modulus() as usize;
        assert!(n != 0, "n is multiple of modulus");
        self.inv[n]
    }

    pub fn nck(&mut self, mut n: usize, mut k: usize) -> Mint {
        if n < k {
            return Mint::from_raw(0);
        }

        let p = Mint::modulus() as usize;
        let mut res = Mint::from_raw(1);
        while n > 0 || k > 0 {
            res *= self.fact(n % p) * self.fact_inv(k % p) * self.fact_inv((n - k) % p);
            n /= p;
            k /= p;
        }
        res
    }

    /// n 個から k 個選んで並べる場合の数
    pub fn npk(&mut self, n: usize, k: usize) -> Mint {
        if n < k {
            Mint::from(0)
        } else {
            self.expand(n);
            self.fact[n] * self.fact_inv[n - k]
        }
    }

    /// 重複を許して n 個から k 個選ぶ場合の数  
    /// または、 [x^k] (1-x)^{-n}
    pub fn nhk(&mut self, n: usize, k: usize) -> Mint {
        if n == 0 && k == 0 {
            Mint::from(1)
        } else {
            self.nck(n + k - 1, k)
        }
    }

    /// カタラン数
    /// n 個の +1 と n 個の -1 を、累積和がすべて非負となるように並べる場合の数
    pub fn catalan(&mut self, n: usize) -> Mint {
        self.expand(n * 2);
        self.fact[n * 2] * self.fact_inv[n + 1] * self.fact_inv[n]
    }

    // Calculate sum(xCk) for x in set A and every k in [0..=max_val],
    pub fn xck(&mut self, a: &[usize]) -> Vec<Mint> {
        // First turn it into count[i]
        let max_val = *a.iter().max().unwrap();
        self.expand(max_val + 1);
        let mut count = vec![0; max_val + 1];
        for &x in a.iter() {
            count[x] += 1;
        }

        let mut f1_val = vec![Mint::new(0); max_val + 2];
        for i in 0..=max_val {
            if count[i] != 0 {
                f1_val[i] = Mint::new(count[i]) * self.fact[i];
            }
        }
        // dbg!(f1_val.clone());
        let f1: FPS = f1_val.into();

        // f2: 1 / (- i)!
        // This function is undefined for i > 0. let's shift everything by max_val:
        // f2(max_val + i) = 1 / (-i)! => f2(i) = 1 / (max_val - i)!
        let mut f2_val = vec![Mint::new(0); 2 * max_val + 2];
        for i in 0..=max_val {
            f2_val[i] = self.fact_inv[max_val - i];
        }
        // dbg!(f2_val.clone());
        let f2: FPS = f2_val.into();

        // finally, to find xCn = 1/n! * sum(i>=0)[ count[i]*(i!) * 1/(i-n)! ]
        // = 1/n! * sum(n1+n2=n)[ f1(n1) * f2(max_val + n2) ]
        // 1/n! * [F1(x) * F2(x)] and read off coefficient x^(n + max_val)
        let f3 = f1 * f2;
        // dbg!(f3.clone());
        // We can use f3 to calculate quickly for all n = 0 -> max_val.
        let mut ans = vec![Mint::new(0); max_val + 1];
        for i in 0..=max_val {
            ans[i] = self.fact_inv[i] * f3[max_val + i];
        }
        ans
    }
}
