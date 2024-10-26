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
            bit: vec![0; n],
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
            std::ops::Bound::Unbounded => self.len - 1,
        };
        self.sum_full(end)
            - if start == 0 {
                0
            } else {
                self.sum_full(start - 1)
            }
    }

    // Single add
    pub fn add(&mut self, i: i32, delta: i32) {
        let mut i = i;
        while i < self.len as i32 {
            self.bit[i as usize] += delta;
            i = i | (i + 1);
        }
    }
}

// =========================== End template here =======================

type V<T> = Vec<T>;
type V2<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    // let t = read_1_number_(line, reader, 0);
    // (0..t).for_each(|_te| {
    // });
    let (n, m, q) = read_3_number_(line, reader, 0usize);
    let mut s = read_line_binary_template(line, reader);

    let mut v: V<usize> = V::new();
    let mut used: V<bool> = vec![false; n];
    // Jump to the next unused position fast.
    let mut next_not_use: V<usize> = (1..n + 1).collect();

    (0..m).for_each(|_| {
        let (l, r) = read_2_number_(line, reader, 0usize);
        let (l, r) = (l - 1, r - 1);

        let mut stk: V<usize> = V::new();
        let mut i = l;
        while i <= r {
            if !used[i] {
                used[i] = true;
                v.push(i);
                stk.push(i);
            }
            i = next_not_use[i];
        }

        while !stk.is_empty() {
            let lt = stk.pop().unwrap();
            let nx = next_not_use[lt];
            if nx < n && used[nx] {
                next_not_use[lt] = next_not_use[nx];
            }
        }
    });

    let mut ord_s: V<u8> = v.iter().map(|&i| s[i]).collect();
    let mut mp: V<usize> = vec![n; n];
    v.iter().enumerate().for_each(|(i, &c)| mp[c] = i);
    let mut num_spare: usize = (0..n)
        .filter_map(|i| {
            if mp[i] != n {
                return None;
            }
            Some(s[i] as usize)
        })
        .sum();

    // println!("s = {s:?}, v = {v:?}, ord_s = {ord_s:?}, mp = {mp:?}, spare = {num_spare:?}");

    let mut fw = FenwickTree::new(ord_s.len());
    ord_s
        .iter()
        .enumerate()
        .for_each(|(i, &c)| fw.add(i as i32, c as i32));

    (0..q).for_each(|qq| {
        let p = read_1_number_(line, reader, 0usize);
        let p = p - 1;
        // println!("Case {qq} flip {p}, s[{p}] = {}", s[p]);

        if mp[p] != n {
            // Is inside
            let in_pos = mp[p];
            if ord_s[in_pos] == 1 {
                fw.add(in_pos as i32, -1);
            } else {
                fw.add(in_pos as i32, 1);
            }
            ord_s[in_pos] = 1 - ord_s[in_pos];
        } else {
            // A spare
            if s[p] == 1 {
                num_spare -= 1;
            } else {
                num_spare += 1;
            }
            s[p] = 1 - s[p];
        }
        // println!("After flip, s = {s:?}, ord_s = {ord_s:?}");

        let num1 = fw.sum(..);
        let num0 = ord_s.len() - num1 as usize;
        // println!("After flip, num spare = {num_spare}, num1 = {num1}, num0 = {num0}");
        if num0 < num_spare {
            // println!("Use all spare to fill up all 0");
            writeln!(out, "{}", num0).unwrap();
        } else {
            let prx = num1 as usize + num_spare;
            let prx1 = fw.sum(..prx);
            // println!("Can make maximum {prx} 1s, prefix have {prx1} 1s");
            writeln!(out, "{}", prx - prx1 as usize).unwrap();
        }
    });
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
