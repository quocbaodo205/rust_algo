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
// https://cp-algorithms.com/string/z-function.html
fn z_function(s: &Vec<u8>) -> Vec<usize> {
    let n = s.len();
    let mut z = vec![0; n];
    let mut l = 0;
    let mut r = 0;
    (1..n).for_each(|i| {
        if i < r {
            z[i] = min(r - i, z[i - l]);
        }
        while (i + z[i] < n) && (s[z[i]] == s[i + z[i]]) {
            z[i] += 1;
        }
        if i + z[i] > r {
            l = i;
            r = i + z[i];
        }
    });
    z
}

// =========================== End template here =======================

type V<T> = Vec<T>;
type V2<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let t = read_1_number_(line, reader, 0);
    (0..t).for_each(|_te| {
        let (n, l, r) = read_3_number_(line, reader, 0usize);
        let s = read_line_str_as_vec_template(line, reader);
        let z = z_function(&s);

        let mut ans: V<usize> = vec![0; n + 1];
        ans[1] = n;
        let E: usize = (n as f32).sqrt() as usize + 1;
        // Check each prefix to find the max split for it
        (1..=E).for_each(|t| {
            let mut cur_split = 1;
            let mut j = t;
            while j < n {
                while j < n && z[j] < t {
                    j += 1;
                }
                if j == n {
                    break;
                }
                cur_split += 1;
                ans[cur_split] = max(ans[cur_split], t);
                j += t;
            }
        });
        // println!("Case {_te}, z = {z:?}");
        // Divide s into k substrings
        (2..=E).for_each(|k| {
            // println!("k = {k}");
            // Binary search to find the max LCP
            let mut l = 1;
            let mut r = n;

            let mut lp = 0;
            while l <= r {
                let mid = (l + r) / 2;
                // println!("l = {l}, r = {r}, mid = {mid}, lp = {lp}");

                // Find if can k split
                let mut cur_split = 1;
                let mut j = mid;
                while j < n && cur_split < k {
                    while j < n && z[j] < mid {
                        j += 1;
                    }
                    // println!("j = {j}, cur = {cur_split}");
                    if j == n {
                        break;
                    }
                    cur_split += 1;
                    j += mid;
                }

                // println!("Lastly cur = {cur_split}");

                if cur_split >= k {
                    lp = mid;
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }

            // Found the max LCP to split into k substrings
            if lp != 0 {
                ans[k] = max(ans[k], lp);
            }
        });

        (l..=r).for_each(|i| {
            write!(out, "{} ", ans[i]).unwrap();
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
