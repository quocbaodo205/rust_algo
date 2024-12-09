#[allow(dead_code)]
// List primes <= n
fn linear_sieve(n: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
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

#[allow(dead_code)]
fn prime_list(a: usize, lp: &Vec<usize>) -> Vec<usize> {
    let mut x = a;
    let mut f: Vec<usize> = Vec::new();
    if lp[x] == x {
        // Is a prime number
        f.push(x);
    } else {
        while x > 1 {
            let q = lp[x];
            f.push(q);
            while x % q == 0 {
                x /= q;
            }
        }
    }
    f
}

#[allow(dead_code)]
fn prime_factorize(x: usize, lp: &Vec<usize>) -> Vec<(usize, usize)> {
    let mut ans: Vec<(usize, usize)> = Vec::new();

    let mut x = x;
    while x > 1 {
        let k = lp[x];
        let mut count = 0;
        while x % k == 0 {
            x /= k;
            count += 1;
        }
        ans.push((k, count));
    }

    ans
}

#[allow(dead_code)]
fn mobius(n: usize) -> Vec<i8> {
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

#[allow(dead_code)]
// To be used with the mobius function,
// so only care p1*p2,... not p1^2, p1^3... since modibus(d) = 0 anyway.
fn factors_mu(a: usize, lp: &Vec<usize>) -> Vec<usize> {
    let mut x = a;
    let mut f: Vec<usize> = Vec::new();
    f.push(1);
    if lp[x] == x {
        // Is a prime number
        f.push(x);
    } else {
        while x > 1 {
            let q = lp[x];
            while x % q == 0 {
                x /= q;
            }
            let len = f.len();
            (0..len).for_each(|i| {
                f.push(f[i] * q);
            });
        }
    }
    f
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
