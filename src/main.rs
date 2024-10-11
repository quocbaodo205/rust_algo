use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet},
    fmt::write,
    fmt::Debug,
    io::{self, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    iter::zip,
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

type V<T> = Vec<T>;
type V2<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;

// =========================== End template here =======================

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let t = read_1_number_(line, reader, 0);
    (0..t).for_each(|_te| {
        let (n, m) = read_2_number_(line, reader, 0usize);
        let mut g: V2<usize> = vec![V::new(); n];
        let mut revg: V2<usize> = vec![V::new(); n];
        (0..m).for_each(|_| {
            let (mut u, mut v) = read_2_number_(line, reader, 0usize);
            u -= 1;
            v -= 1;
            if u > v {
                swap(&mut u, &mut v);
            }
            g[u].push(v);
            revg[v].push(u);
        });

        let mut d: V<usize> = (0..n).collect();
        (0..n).for_each(|u| {
            if u > 0 {
                d[u] = min(d[u], d[u - 1] + 1);
            }
            g[u].iter().for_each(|&v| d[v] = min(d[v], d[u] + 1));
        });

        // println!("Case {_te}, d = {d:?}");

        let mut gt = 0;
        let mut max_good_right: Set<(usize, usize)> = Set::new();
        let mut to_erase: V2<(usize, usize)> = vec![V::new(); n];

        let mut ans: V<bool> = vec![false; n - 1];
        (0..n).rev().for_each(|s| {
            revg[s].iter().for_each(|&v| {
                // edge v -> u (it's the reverse graph)
                // + GT for multiset
                let mdv = (s - d[v], gt);
                gt += 1;
                max_good_right.insert(mdv);
                to_erase[v].push(mdv);
            });
            // println!("u = {u}, current set = {max_good_right:?}");

            // Delete added bridge that start at u
            to_erase[s].iter().for_each(|e| {
                max_good_right.remove(e);
            });

            if s < n - 1 {
                let mdv = if max_good_right.is_empty() {
                    -1
                } else {
                    max_good_right.last().unwrap().0 as i32
                };
                if s as i32 >= mdv - 1 {
                    ans[s] = true;
                }
            }
        });

        ans.iter()
            .for_each(|&valid| write!(out, "{}", if valid { 1 } else { 0 }).unwrap());
        writeln!(out).unwrap();
    });
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
