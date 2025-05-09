use std::collections::HashMap;

// List primes <= n
pub fn linear_sieve(n: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut lp: Vec<usize> = vec![0; n + 1];
    let mut pr: Vec<usize> = Vec::new();
    let mut idx: Vec<usize> = vec![0; n + 1];
    let c = n as u64;
    unsafe {
        (2..=n).for_each(|i| {
            if lp.get_unchecked(i) == &0 {
                lp[i] = i;
                pr.push(i);
            }
            let mut j = 0;
            while j < pr.len()
                && *pr.get_unchecked(j) <= *lp.get_unchecked(i)
                && (i as u64) * (*pr.get_unchecked(j) as u64) <= c
            {
                lp[i * *pr.get_unchecked(j)] = *pr.get_unchecked(j);
                j += 1;
            }
        });
    }
    // Mapping: prime -> index
    pr.iter().enumerate().for_each(|(i, &prime)| {
        idx[prime] = i + 1;
    });
    // Lowest prime factor
    // list of prime number
    // prime -> index mapping
    (lp, pr, idx)
}

pub fn is_prime(n: usize, lp: &Vec<usize>, pr: &Vec<usize>) -> bool {
    if n < lp.len() {
        return lp[n] == n;
    }
    let r = (n as f64).sqrt() as usize;
    for &p in pr.iter() {
        if n % p == 0 {
            return false;
        }
        if p >= r {
            break;
        }
    }
    true
}

pub fn prime_list(a: usize, lp: &Vec<usize>, pr: &Vec<usize>) -> Vec<usize> {
    let mut x = a;
    let mut f: Vec<usize> = Vec::new();
    let mut pr_idx = 0;
    while x > 1 {
        let q = {
            if x < lp.len() {
                lp[x]
            } else {
                let r = (x as f32).sqrt() as usize;
                while x % pr[pr_idx] != 0 {
                    pr_idx += 1;
                    if pr[pr_idx] >= r {
                        // Is a prime number
                        break;
                    }
                }
                if pr[pr_idx] >= r {
                    x
                } else {
                    pr[pr_idx]
                }
            }
        };
        f.push(q);
        while x % q == 0 {
            x /= q;
        }
    }
    f
}

pub fn prime_factorize(a: usize, lp: &Vec<usize>, pr: &Vec<usize>) -> Vec<(usize, usize)> {
    let mut ans: Vec<(usize, usize)> = Vec::new();
    let mut x = a;
    let mut pr_idx = 0;
    while x > 1 {
        let q = {
            if x < lp.len() {
                lp[x]
            } else {
                let r = (x as f32).sqrt() as usize;
                while x % pr[pr_idx] != 0 {
                    pr_idx += 1;
                    if pr[pr_idx] >= r {
                        // Is a prime number
                        break;
                    }
                }
                if pr[pr_idx] >= r {
                    x
                } else {
                    pr[pr_idx]
                }
            }
        };
        let mut count = 0;
        while x % q == 0 {
            x /= q;
            count += 1;
        }
        ans.push((q, count));
    }

    ans
}

pub fn mobius(n: usize) -> Vec<i8> {
    let mut is_composite: Vec<bool> = vec![false; n + 1];
    let mut pr: Vec<usize> = Vec::new();
    let mut mu: Vec<i8> = vec![0; n + 1];
    mu[1] = 1;
    let c = n as u64;
    (2..=n).for_each(|i| {
        if !is_composite[i] {
            pr.push(i);
            mu[i] = -1;
        }
        let mut j = 0;
        while j < pr.len() && (i as u64) * (pr[j] as u64) <= c {
            is_composite[i * pr[j]] = true;
            if i % pr[j] != 0 {
                mu[i * pr[j]] = -mu[i];
            } else {
                break;
            }
            j += 1;
        }
    });
    mu
}

pub fn big_mobius(n: usize, mu: &Vec<i8>, pr: &Vec<usize>) -> i8 {
    if n < mu.len() {
        return mu[n];
    }
    let mut ans = 1;
    let mut x = n;
    let r = (n as f32).sqrt() as usize;
    for &p in pr.iter() {
        if p > r {
            break;
        }
        let mut k = 1;
        while x % p == 0 {
            x /= p;
            k *= p;
            if mu[k] == 0 {
                return 0;
            }
        }
        ans *= mu[k];
        if x < mu.len() {
            return ans * mu[x];
        }
    }
    // If not returned, last part has to be a prime with mu[p] = -1.
    ans * -1
}

// To be used with the mobius function,
// so only care p1*p2,... not p1^2, p1^3... since mobius(d) = 0 anyway.
// mu[x in factors_mu] is the Inclusion-Exclusion of all prime factor of a.
pub fn factors_mu(a: usize, lp: &Vec<usize>, pr: &Vec<usize>) -> Vec<usize> {
    let mut x = a;
    let mut f: Vec<usize> = Vec::new();
    f.push(1);
    let mut pr_idx = 0;
    while x > 1 {
        let q = {
            if x < lp.len() {
                lp[x]
            } else {
                let r = (x as f32).sqrt() as usize;
                while x % pr[pr_idx] != 0 {
                    pr_idx += 1;
                    if pr[pr_idx] >= r {
                        // Is a prime number
                        break;
                    }
                }
                if pr[pr_idx] >= r {
                    x
                } else {
                    pr[pr_idx]
                }
            }
        };
        while x % q == 0 {
            x /= q;
        }
        let len = f.len();
        (0..len).for_each(|i| {
            f.push(f[i] * q);
        });
    }
    f
}

pub fn divisors(n: usize) -> Vec<usize> {
    let mut res = Vec::new();
    let r = (n as f32).sqrt() as usize;
    for d in 1..=r {
        if n % d == 0 {
            res.push(d);
            if n / d != d {
                res.push(n / d);
            }
        }
    }
    res.sort();
    res
}

fn power(a: u64, b: u64, m: u64) -> u64 {
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

fn sum_n(n: u64) -> u128 {
    let r = n as u128;
    (r * (r + 1)) / 2
}

fn sum_n_square(n: u64, m: u128) -> u64 {
    let r = n as u128;
    let x = (r * (r + 1)) / 2;
    if x % 3 == 0 {
        return ((((x / 3) % m) * ((2 * r + 1) % m)) % m) as u64;
    }
    let y = (2 * r + 1) / 3;
    (((x % m) * (y % m)) % m) as u64
}

#[allow(dead_code)]
// cal psum of divisor function(2) (1 shot not a prefix array)
fn psum_divisor_func_sq(n: u64, m: u64) -> u64 {
    let sn = (n as f64).sqrt() as usize;
    let mut res = 0u64;
    (1..=sn).for_each(|i| res = (res + sum_n_square(n / (i as u64), m as u128)) % m);
    (1..=sn).for_each(|i| {
        let x = i as u64;
        res = (res + ((x * x) % m) * ((n / x) % m)) % m;
    });
    res = res + m - ((sn as u64) % m * sum_n_square(sn as u64, m as u128)) % m;
    res % m
}

#[allow(dead_code)]
// framework to cal psum of (f*g)(n)
// https://codeforces.com/blog/entry/117635
fn dirichlet_convolution(
    n: u64,
    a: f64,
    b: f64,
    f: fn(u64) -> u64,
    g: fn(u64) -> u64,
    F: fn(u64) -> u64,
    G: fn(u64) -> u64,
) -> u64 {
    // Step 1: cal split by the expected time to cal F(n) and G(n).
    let nf = n as f64;
    let p = (1.0 - b) / (2.0 - a - b);
    let kx = nf.powf(p);
    let k = kx as u64;
    let l = n / k;
    // Step 2: cal the result
    let mut res = 0;
    (1..=k).for_each(|i| {
        // First part
        res += f(i) * G(n / i);
    });
    (1..=l).for_each(|i| {
        // Second part
        res += g(i) * F(n / i);
    });
    // Overlap part
    res -= F(k) * G(l);
    res
}

#[allow(dead_code)]
// Given f(x) and f^-1(x) (denote as g), find f^a(x) = y
// f(x) generate a cyclic group of order n
// For example: f(x) = x*k % n -> f^a(x) = k^a % n.
fn baby_step_giant_step(
    f: &dyn Fn(u64) -> u64,
    g: &dyn Fn(u64) -> u64,
    n: usize,
    f0: u64,
    x: u64,
    y: u64,
) -> Option<usize> {
    let m = (n as f64).sqrt().ceil() as usize;

    // Calculate table for f0 -> f^(m-1)
    let mut mp: HashMap<u64, usize> = HashMap::new();
    let mut c = f0;
    for i in 0..m {
        mp.insert(c, i);
        c = f(c);
    }

    // Calculate f^-m (via g)
    let mut gm = f0;
    for _ in 0..m {
        gm = g(gm);
    }

    let mut y = y;
    for i in 0..m {
        if let Some(&j) = mp.get(&y) {
            return Some(i * m + j);
        }
        y *= gm;
    }
    None
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_d2() {
        let m = 1000000000u64;
        assert_eq!(sum_n_square(6u64, m as u128), 91);
        assert_eq!(sum_n_square(7u64, m as u128), 140);
        assert_eq!(psum_divisor_func_sq(6u64, m), 113);
        assert_eq!(psum_divisor_func_sq(7u64, m), 163);
        assert_eq!(psum_divisor_func_sq(8u64, m), 248);
        assert_eq!(psum_divisor_func_sq(9u64, m), 339);
        assert_eq!(psum_divisor_func_sq(15u64, m), 1481);
        assert_eq!(psum_divisor_func_sq(1000000000000000u64, m), 281632621);
    }

    #[test]
    fn test_dirichlet() {
        fn f(n: u64) -> u64 {
            1
        }

        fn g(n: u64) -> u64 {
            n * n
        }

        fn F(n: u64) -> u64 {
            n
        }

        fn G(n: u64) -> u64 {
            (n * (n + 1) * (2 * n + 1)) / 6
        }

        assert_eq!(dirichlet_convolution(6, 0.0, 0.0, f, g, F, G), 113);
        assert_eq!(dirichlet_convolution(7, 0.0, 0.0, f, g, F, G), 163);
        assert_eq!(dirichlet_convolution(8, 0.0, 0.0, f, g, F, G), 248);
        assert_eq!(dirichlet_convolution(9, 0.0, 0.0, f, g, F, G), 339);
        assert_eq!(dirichlet_convolution(15, 0.0, 0.0, f, g, F, G), 1481);
    }

    #[test]
    fn test_dirichlet_divisor_fn() {
        fn f(n: u64) -> u64 {
            1
        }

        fn g(n: u64) -> u64 {
            let mut res = 0u64;
            let sn = ((n as f64).sqrt()) as u64;
            (1..=sn).for_each(|d| {
                if n % d == 0 {
                    res += d * d;
                    if n / d != d {
                        res += (n / d) * (n / d);
                    }
                }
            });
            res
        }

        fn F(n: u64) -> u64 {
            n
        }

        fn G(n: u64) -> u64 {
            psum_divisor_func_sq(n, 1000000000u64)
        }

        fn simple(n: u64) -> u64 {
            let mut res = 0u64;
            (1..=n).for_each(|x| {
                (1..=x).for_each(|d| {
                    if x % d == 0 {
                        res += g(d);
                    }
                });
            });
            res
        }

        for x in 1..100 {
            assert_eq!(dirichlet_convolution(x, 0.0, 0.5, f, g, F, G), simple(x));
        }
    }
}
