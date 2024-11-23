use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet},
    fmt::write,
    fmt::Debug,
    io::{self, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    iter::zip,
    mem,
    mem::swap,
    ops::Bound::*,
    ops::RangeBounds,
    str::FromStr,
};

#[allow(dead_code)]
fn read_line_str_as_vec_template(line: &mut String, reader: &mut BufReader<Stdin>) -> Vec<u8> {
    line.clear();
    reader.read_line(line).unwrap();
    line.trim().as_bytes().iter().cloned().collect()
}

#[allow(dead_code)]
fn read_line_binary_template(line: &mut String, reader: &mut BufReader<Stdin>) -> Vec<u8> {
    line.clear();
    reader.read_line(line).unwrap();
    line.trim()
        .as_bytes()
        .iter()
        .cloned()
        .map(|x| x - b'0')
        .collect()
}

#[allow(dead_code)]
fn read_vec_template<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> Vec<T> {
    line.clear();
    reader.read_line(line).unwrap();
    Vec::from_iter(line.split_whitespace().map(|x| match x.parse() {
        Ok(data) => data,
        Err(_) => default,
    }))
}

#[allow(dead_code)]
fn read_1_number_<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> T {
    let v = read_vec_template(line, reader, default);
    v[0]
}

#[allow(dead_code)]
fn read_2_number_<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (T, T) {
    let v = read_vec_template(line, reader, default);
    (v[0], v[1])
}

#[allow(dead_code)]
fn read_3_number_<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (T, T, T) {
    let v = read_vec_template(line, reader, default);
    (v[0], v[1], v[2])
}

#[allow(dead_code)]
fn read_vec_string_template(line: &mut String, reader: &mut BufReader<Stdin>) -> Vec<String> {
    line.clear();
    reader.read_line(line).unwrap();
    line.split_whitespace().map(|x| x.to_string()).collect()
}

#[allow(unused_macros)]
macro_rules! isOn {
    ($S:expr, $b:expr) => {
        ($S & (1u64 << $b)) > 0
    };
}

#[allow(unused_macros)]
macro_rules! turnOn {
    ($S:ident, $b:expr) => {
        $S |= (1u64 << $b)
    };
    ($S:expr, $b:expr) => {
        $S | (164 << $b)
    };
}

#[allow(dead_code)]
fn gcd(mut n: usize, mut m: usize) -> usize {
    if n == 0 || m == 0 {
        return n + m;
    }
    while m != 0 {
        if m < n {
            let t = m;
            m = n;
            n = t;
        }
        m = m % n;
    }
    n
}

#[allow(dead_code)]
fn true_distance_sq(a: (i64, i64), b: (i64, i64)) -> i64 {
    let x = (a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1);
    // println!("Dist sq {:?} - {:?} = {}", a, b, x);
    return x;
}

#[allow(dead_code)]
fn lower_bound_pos<T: Ord + PartialOrd>(a: &Vec<T>, search_value: T) -> usize {
    a.binary_search_by(|e| match e.cmp(&search_value) {
        Ordering::Equal => Ordering::Greater,
        ord => ord,
    })
    .unwrap_err()
}

#[allow(dead_code)]
fn upper_bound_pos<T: Ord + PartialOrd>(a: &Vec<T>, search_value: T) -> usize {
    a.binary_search_by(|e| match e.cmp(&search_value) {
        Ordering::Equal => Ordering::Less,
        ord => ord,
    })
    .unwrap_err()
}

#[allow(dead_code)]
fn neighbors<T>(tree: &BTreeSet<T>, val: T) -> (Option<&T>, Option<&T>)
where
    T: Ord + Copy,
{
    let mut before = tree.range((Unbounded, Excluded(val)));
    let mut after = tree.range((Excluded(val), Unbounded));

    (before.next_back(), after.next())
}

// Copy this as needed, this is too restricted.
#[allow(dead_code)]
fn bin_search_template(l: usize, r: usize, f: &dyn Fn(usize) -> bool) -> usize {
    let mut l = l;
    let mut r = r;

    let mut ans = l; // Change acordingly.
    while l <= r {
        let mid = (l + r) / 2;
        if f(mid) {
            ans = mid;
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return ans;
}

pub fn next_permutation<T>(arr: &mut [T]) -> bool
where
    T: std::cmp::Ord,
{
    use std::cmp::Ordering;

    // find 1st pair (x, y) from back which satisfies x < y
    let last_ascending = match arr.windows(2).rposition(|w| w[0] < w[1]) {
        Some(i) => i,
        None => {
            arr.reverse();
            return false;
        }
    };

    // In the remaining later segment, find the one which is just
    // larger that the index found above.
    // SAFETY: unwrap_err whill not panic since binary search will
    // will never succeed since we never return equal ordering
    let swap_with = arr[last_ascending + 1..]
        .binary_search_by(|n| match arr[last_ascending].cmp(n) {
            Ordering::Equal => Ordering::Greater,
            ord => ord,
        })
        .unwrap_err();
    arr.swap(last_ascending, last_ascending + swap_with);
    arr[last_ascending + 1..].reverse();
    true
}

#[allow(dead_code)]
fn num_digit(x: u64) -> u64 {
    let mut c = 0;
    let mut rx = x;
    while rx > 0 {
        rx /= 10;
        c += 1;
    }
    return c;
}

// =========================== End template here =======================

type V<T> = Vec<T>;
type V2<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;

fn mobius(n: usize) -> Vec<i8> {
    let mut tag: Vec<bool> = vec![false; n + 1];
    let mut pr: Vec<usize> = Vec::new();
    let mut mu: Vec<i8> = vec![0; n + 1];
    mu[1] = 1;
    let c = n as u64;
    (2..=n).for_each(|i| {
        if !tag[i] {
            pr.push(i);
            mu[i] = -1;
        }
        let mut j = 0;
        while j < pr.len() && (i as u64) * (pr[j] as u64) <= c {
            tag[i * pr[j]] = true;
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
// List primes <= 10^7
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

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    // let t = read_1_number_(line, reader, 0);
    // (0..t).for_each(|_te| {});
    let n = read_1_number_(line, reader, 0usize);
    let p = read_vec_template(line, reader, 0usize);
    let (lp, _, _) = linear_sieve(n);
    let mu = mobius(n);
    let nmu: V<(usize, i8)> = mu
        .iter()
        .cloned()
        .enumerate()
        .filter(|&(_, x)| x != 0)
        .collect();
    let mut c1 = vec![0i64; n + 1];
    let mut c2 = vec![0i64; n + 1];
    let mut c3: V<BTreeMap<usize, i64>> = vec![BTreeMap::new(); n + 1];
    (1..=n).for_each(|i| {
        let p1 = factors_mu(i, &lp);
        p1.iter().for_each(|&x| c1[x] += 1);
        let p2 = factors_mu(p[i - 1], &lp);
        p2.iter().for_each(|&x| c2[x] += 1);
        p1.iter().for_each(|&a| {
            if mu[a] == 0 {
                return;
            }
            p2.iter().for_each(|&b| {
                if mu[b] == 0 {
                    return;
                }
                if !c3[a].contains_key(&b) {
                    c3[a].insert(b, 0);
                }
                *(c3[a].get_mut(&b).unwrap()) += 1;
            });
        });
    });
    c1.iter_mut().for_each(|x| *x = (*x * ((*x) + 1)) / 2);
    c2.iter_mut().for_each(|x| *x = (*x * ((*x) + 1)) / 2);

    let mut ans = ((n as i64) * (n as i64 + 1)) / 2;
    nmu.iter().for_each(|&(d, md)| {
        let v1 = md as i64 * c1[d];
        let v2 = md as i64 * c2[d];
        ans = ans - v1 - v2;
    });

    nmu.iter().for_each(|&(a, md)| {
        c3[a]
            .iter()
            .for_each(|(&b, &c)| ans += md as i64 * mu[b] as i64 * ((c * (c + 1)) / 2));
    });

    writeln!(out, "{ans}").unwrap();
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
