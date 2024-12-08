use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{write, Debug},
    io::{self, read_to_string, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    iter::zip,
    mem::{self, swap},
    ops::{self, Bound::*, RangeBounds},
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

// Template for grid movement
#[derive(Copy, Clone, Debug)]
struct Point((i32, i32));

impl ops::Add for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point((self.0 .0 + rhs.0 .0, self.0 .1 + rhs.0 .1))
    }
}

impl ops::Sub for Point {
    type Output = Point;

    fn sub(self, rhs: Point) -> Self::Output {
        Point((self.0 .0 - rhs.0 .0, self.0 .1 - rhs.0 .1))
    }
}

#[allow(dead_code)]
fn is_valid_point(x: Point, n: usize, m: usize) -> bool {
    return x.0 .0 >= 0 && x.0 .0 < n as i32 && x.0 .1 >= 0 && x.0 .1 < m as i32;
}

#[allow(dead_code)]
// Translate a character to a directional Point
fn translate(c: u8) -> Point {
    match c as char {
        'U' => Point((-1, 0)),
        'D' => Point((1, 0)),
        'L' => Point((0, -1)),
        'R' => Point((0, 1)),
        _ => Point((0, 0)),
    }
}

#[allow(dead_code)]
// Simple DFS from a point, using a map of direction LRUD
fn dfs_grid(s: Point, mp: &V<V<u8>>, checked: &mut V<V<bool>>) {
    let n = mp.len();
    let m = mp[0].len();

    if checked[s.0 .0 as usize][s.0 .1 as usize] {
        return;
    }
    let mut s = Some(s);
    while let Some(u) = s {
        checked[u.0 .0 as usize][u.0 .1 as usize] = true;
        let v = u + translate(mp[u.0 .0 as usize][u.0 .1 as usize]);
        s = match is_valid_point(v, n, m) && !checked[v.0 .0 as usize][v.0 .1 as usize] {
            true => Some(v),
            false => None,
        }
    }
}

#[allow(dead_code)]
// Simple BFS from a point, using a blocker map
fn flood_grid(s: Point, blocked: &V<V<bool>>, checked: &mut V<V<bool>>) {
    let n = blocked.len();
    let m = blocked[0].len();

    if checked[s.0 .0 as usize][s.0 .1 as usize] {
        return;
    }
    let mut q: VecDeque<Point> = VecDeque::new();
    let direction = "LRUD".as_bytes();
    q.push_back(s);
    while let Some(u) = q.pop_front() {
        direction.iter().for_each(|&d| {
            let v = u + translate(d);
            if is_valid_point(v, n, m)
                && !checked[v.0 .0 as usize][v.0 .1 as usize]
                && !blocked[v.0 .0 as usize][v.0 .1 as usize]
            {
                checked[v.0 .0 as usize][v.0 .1 as usize] = true;
                q.push_back(v);
            }
        });
    }
}

// =========================== End template here =======================

type V<T> = Vec<T>;
type VV<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;
type Map<K, V> = BTreeMap<K, V>;
type US = usize;
type UU = (US, US);

fn linear_sieve() -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut lp: Vec<usize> = vec![0; 1000001];
    let mut pr: Vec<usize> = Vec::new();
    let mut idx: Vec<usize> = vec![0; 1000001];
    let c = 1000000u64;
    unsafe {
        (2..=1000000).for_each(|i| {
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

fn solve(re: &mut BufReader<Stdin>, li: &mut String, out: &mut BufWriter<Stdout>) {
    let t = read_1_number_(li, re, 0);
    let df = 0usize;
    let (lp, pr, _) = linear_sieve();
    let count_pr: V<US> = (0..=100000)
        .map(|i| {
            if i < 2 {
                return 1;
            }
            let ft = prime_factorize(i, &lp);
            ft.iter().map(|(_, q)| q).sum()
        })
        .collect();

    (0..t).for_each(|_te| {
        let (n, m) = read_2_number_(li, re, df);
        let mut a = read_vec_template(li, re, df);
        a.sort();
        a.reverse();

        let mut ans = vec![0; n + 1];
        ans[1] = a[0];
        for i in 2..=n {
            // println!("prime factors of {i}: {:?}", prime_factorize(i, &lp));
            if count_pr[i] >= a.len() {
                writeln!(out, "-1").unwrap();
                return;
            }
            ans[i] = a[count_pr[i]];
        }

        ans[1..].iter().for_each(|&x| {
            write!(out, "{x} ").unwrap();
        });
        writeln!(out).unwrap();
    });
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
