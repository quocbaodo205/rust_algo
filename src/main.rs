use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet},
    fmt::write,
    io::{self, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    iter::zip,
    str::FromStr,
};

#[allow(dead_code)]
fn read_line_template<T: FromStr>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> T {
    line.clear();
    reader.read_line(line).unwrap();
    match line.trim().parse::<T>() {
        Ok(data) => data,
        Err(_) => default,
    }
}

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
fn gcd(mut n: u64, mut m: u64) -> u64 {
    assert!(n != 0 && m != 0);
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

// =========================== End template here =======================

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let v = read_vec_template(line, reader, 0);
    let t = v[0];
    for _te in 0..t {
        let v = read_vec_template(line, reader, 0);
        let n: usize = v[0];
        let a = read_vec_template(line, reader, 0);
        let mut curs = 0;
        (0..n).rev().for_each(|i| {
            if i == n - 1 {
                curs = a[i];
            } else {
                if a[i] <= a[i + 1] {
                    // a[i] need to wait until a[i+1] = a[i], then 1 more move to reduce together.
                    curs += 1;
                } else {
                    // a[i] > a[i+1], a[i] reduced separatedly into a[i+1]
                    if curs < a[i] {
                        // With all previous move, still can't reduce a[i] fully
                        curs = a[i]; // After a[i+1] = 0, a[i] still remain some!
                    } else {
                        // a[i] become a[i+1] at some point, only to wait
                        // Since a[i] = a[i+1], take one more move to reduce a[i] to 0.
                        curs += 1;
                    }
                }
            }
        });
        writeln!(out, "{curs}").unwrap();
    }
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
