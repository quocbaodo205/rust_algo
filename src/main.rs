use std::{
    borrow::BorrowMut,
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::{write, Debug, Display},
    io::{self, read_to_string, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    iter::zip,
    mem::{self, swap},
    ops::{self, Bound::*, DerefMut, RangeBounds},
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
fn read_1_number<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> T {
    let v = read_vec_template(line, reader, default);
    v[0]
}

#[allow(dead_code)]
fn read_2_number<T: FromStr + Copy>(
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
        ($S & (1 << $b)) > 0
    };
}

#[allow(unused_macros)]
macro_rules! turnOn {
    ($S:ident, $b:expr) => {
        $S |= (1 << $b)
    };
    ($S:expr, $b:expr) => {
        $S | (1 << $b)
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

type V<T> = Vec<T>;
type VV<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;
type Map<K, V> = BTreeMap<K, V>;
type US = usize;
type UU = (US, US);

#[allow(dead_code)]
fn compress<T>(a: &V<T>) -> (V<US>, V<T>)
where
    T: Ord + PartialOrd + Clone + Copy,
{
    let unique_val: Set<T> = a.iter().cloned().collect();
    let unique_val_arr: V<T> = unique_val.iter().cloned().collect();
    (
        a.iter()
            .map(|x| lower_bound_pos(&unique_val_arr, *x) + 1)
            .collect(),
        unique_val_arr,
    )
}

#[allow(dead_code)]
fn compress_pair<T>(a: &V<(T, T)>) -> (V<UU>, V<T>, V<T>)
where
    T: Ord + PartialOrd + Clone + Copy,
{
    let mut final_arr = vec![(0, 0); a.len()];
    let unique_x: Set<T> = a.iter().map(|&p| p.0).collect();
    let unique_x_v: V<T> = unique_x.iter().cloned().collect();
    (0..a.len()).for_each(|i| {
        final_arr[i].0 = lower_bound_pos(&unique_x_v, a[i].0) + 1;
    });
    let unique_y: Set<T> = a.iter().map(|&p| p.1).collect();
    let unique_y_v: V<T> = unique_y.iter().cloned().collect();
    (0..a.len()).for_each(|i| {
        final_arr[i].1 = lower_bound_pos(&unique_y_v, a[i].1) + 1;
    });
    (final_arr, unique_x_v, unique_y_v)
}

// =========================== IO for classic problems =======================

#[allow(dead_code)]
fn better_array_debug<T>(a: &V<T>)
where
    T: Debug,
{
    a.iter()
        .enumerate()
        .for_each(|(i, x)| println!("{i}: {x:?}"));
}

// 2 array need to have the same length
#[allow(dead_code)]
fn better_2_array_debug<T, D>(a: &V<T>, b: &V<D>)
where
    T: Debug,
    D: Debug,
{
    (0..a.len()).for_each(|i| {
        println!("{i}: {:?} -- {:?}", a[i], b[i]);
    })
}

#[allow(dead_code)]
fn read_n_and_array<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (US, V<T>) {
    let n = read_1_number(line, reader, 0usize);
    let v = read_vec_template(line, reader, default);
    (n, v)
}

#[allow(dead_code)]
fn read_n_and_array_of_pair<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (US, V<(T, T)>) {
    let n = read_1_number(line, reader, 0usize);
    let mut v: V<(T, T)> = V::new();
    v.reserve(n);
    (0..n).for_each(|_| {
        let x = read_2_number(line, reader, default);
        v.push(x);
    });
    (n, v)
}

#[allow(dead_code)]
fn read_n_m_and_2_array<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (T, T, V<T>, V<T>) {
    let (n, m) = read_2_number(line, reader, default);
    let a = read_vec_template(line, reader, default);
    let b = read_vec_template(line, reader, default);
    (n, m, a, b)
}

#[allow(dead_code)]
fn read_graph_edge_list(
    g: &mut VV<US>,
    m: US,
    is_bidirectional: bool,
    line: &mut String,
    reader: &mut BufReader<Stdin>,
) {
    (0..m).for_each(|_| {
        let (u, v) = read_2_number(line, reader, 0usize);
        let (u, v) = (u - 1, v - 1);
        g[u].push(v);
        if is_bidirectional {
            g[v].push(u);
        }
    });
}

#[allow(dead_code)]
fn read_tree_parent_list(g: &mut VV<US>, n: US, line: &mut String, reader: &mut BufReader<Stdin>) {
    // Rooted tree at 0, parent of [1..n-1]
    let parent = read_vec_template(line, reader, 0usize);
    // Example: 1 1 2 2 4 -> parent of 1 is 0, parent of 2 is 0, ...
    (0..n - 2).for_each(|i| g[parent[i] - 1].push(i + 1));
}

#[allow(dead_code)]
fn array_output<T>(a: &V<T>, out: &mut BufWriter<Stdout>)
where
    T: Display,
{
    a.iter().for_each(|x| {
        write!(out, "{x} ").unwrap();
    });
    writeln!(out).unwrap();
}

// =========================== End template here =======================

fn solve(re: &mut BufReader<Stdin>, li: &mut String, out: &mut BufWriter<Stdout>) {
    let t = read_1_number(li, re, 0);
    let df = 0usize;
    (0..t).for_each(|_te| {});
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
