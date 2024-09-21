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
