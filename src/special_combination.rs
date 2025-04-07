use std::cell::RefCell;

// Prime mod is allowed for Mint and FPS
pub type Mint = modint::StaticModInt<924844033>;
pub type FPS = formal_power_series::FormalPowerSeries<924844033>;

pub struct SpecialCombination {
    inv: RefCell<Vec<Mint>>,
    fact: RefCell<Vec<Mint>>,
    fact_inv: RefCell<Vec<Mint>>,
}

impl SpecialCombination {
    /// 初期化
    pub fn new() -> Self {
        Self {
            inv: RefCell::new(vec![Mint::from(0), Mint::from(1)]),
            fact: RefCell::new(vec![Mint::from(1); 2]),
            fact_inv: RefCell::new(vec![Mint::from(1); 2]),
        }
    }

    pub fn expand(&self, n: usize) {
        let mut inv = self.inv.borrow_mut();
        let mut fact = self.fact.borrow_mut();
        let mut fact_inv = self.fact_inv.borrow_mut();
        let m = inv.len();
        let mut nn = m;
        while nn <= n {
            nn *= 2;
        }
        inv.resize(nn, Mint::default());
        fact.resize(nn, Mint::default());
        fact_inv.resize(nn, Mint::default());
        let p = Mint::modulus() as usize;
        for i in m..nn {
            inv[i] = -inv[p % i] * Mint::from((p / i) as u32);
            fact[i] = fact[i - 1] * Mint::from(i);
            fact_inv[i] = fact_inv[i - 1] * inv[i];
        }
    }

    /// n の逆元
    pub fn inv(&self, n: usize) -> Mint {
        self.expand(n);
        self.inv.borrow()[n]
    }

    /// n!
    pub fn fact(&self, n: usize) -> Mint {
        self.expand(n);
        self.fact.borrow()[n]
    }

    /// n! の逆元
    pub fn fact_inv(&self, n: usize) -> Mint {
        self.expand(n);
        self.fact_inv.borrow()[n]
    }

    /// n 個から k 個選ぶ場合の数
    pub fn nck(&self, n: usize, k: usize) -> Mint {
        if n < k {
            Mint::from(0)
        } else {
            self.expand(n);
            self.fact.borrow()[n] * self.fact_inv.borrow()[k] * self.fact_inv.borrow()[n - k]
        }
    }

    /// n 個から k 個選んで並べる場合の数
    // pub fn npk(&self, n: usize, k: usize) -> Mint {
    //     if n < k {
    //         Mint::from(0)
    //     } else {
    //         self.expand(n);
    //         self.fact.borrow()[n] * self.fact_inv.borrow()[n - k]
    //     }
    // }

    /// 重複を許して n 個から k 個選ぶ場合の数  
    /// または、 [x^k] (1-x)^{-n}
    // pub fn nhk(&self, n: usize, k: usize) -> Mint {
    //     if n == 0 && k == 0 {
    //         Mint::from(1)
    //     } else {
    //         self.nck(n + k - 1, k)
    //     }
    // }

    /// カタラン数
    /// n 個の +1 と n 個の -1 を、累積和がすべて非負となるように並べる場合の数
    // pub fn catalan(&self, n: usize) -> Mint {
    //     self.expand(n * 2);
    //     self.fact.borrow()[n * 2] * self.fact_inv.borrow()[n + 1] * self.fact_inv.borrow()[n]
    // }

    // Calculate sum(xCk) for x in set A and every k in [0..=max_val],
    pub fn xck(&self, a: &[usize]) -> Vec<Mint> {
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
                f1_val[i] = Mint::new(count[i]) * self.fact.borrow()[i];
            }
        }
        // dbg!(f1_val.clone());
        let f1: FPS = f1_val.into();

        // f2: 1 / (- i)!
        // This function is undefined for i > 0. let's shift everything by max_val:
        // f2(max_val + i) = 1 / (-i)! => f2(i) = 1 / (max_val - i)!
        let mut f2_val = vec![Mint::new(0); 2 * max_val + 2];
        for i in 0..=max_val {
            f2_val[i] = self.fact_inv.borrow()[max_val - i];
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
            ans[i] = self.fact_inv.borrow()[i] * f3[max_val + i];
        }
        ans
    }
}
