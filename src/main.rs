use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet},
    fmt::write,
    io::{self, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    iter::zip,
    mem::swap,
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

// =========================== End template here =======================

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let t = read_1_number_(line, reader, 0);
    (0..t).for_each(|_te| {
        let n: usize = read_1_number_(line, reader, 0);
        let mut a: Vec<usize> = read_vec_template(line, reader, 0);
        let g: usize = a.iter().fold(0, |g, &x| gcd(g, x));
        a.iter_mut().for_each(|x| *x /= g);

        // Take until gcd = 1
        let mut used: Vec<bool> = vec![false; n];
        let mut cur_sum = 0;
        let mut cur_gcd = 0;
        loop {
            let min_g = (0..n)
                .filter(|&i| !used[i])
                .map(|i| gcd(cur_gcd, a[i]))
                .min()
                .unwrap();
            let pos = (0..n)
                .filter(|&i| !used[i] && gcd(cur_gcd, a[i]) == min_g)
                .next()
                .unwrap();
            cur_gcd = min_g;
            cur_sum += min_g as u64;
            used[pos] = true;
            if min_g == 1 {
                break;
            }
        }
        // The rest: += 1;
        cur_sum += used.iter().filter(|&&u| !u).count() as u64;
        cur_sum *= g as u64;
        writeln!(out, "{cur_sum}").unwrap();
    });
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
