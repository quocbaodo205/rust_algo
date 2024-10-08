use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet},
    fmt::write,
    fmt::Debug,
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

// =========================== End template here =======================

#[allow(dead_code)]
struct SegmentTreeIsInc {
    len: usize,
    tree_min: Vec<usize>,
    tree_max: Vec<usize>,
    is_inc: Vec<bool>,
}

#[allow(dead_code)]
impl SegmentTreeIsInc {
    pub fn new(n: usize) -> Self {
        SegmentTreeIsInc {
            len: n,
            tree_min: vec![1000000000; 4 * n + 1],
            tree_max: vec![0; 4 * n + 1],
            is_inc: vec![true; 4 * n + 1],
        }
    }

    fn query_internal_min(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> usize {
        if ql > qr {
            return 1000000000;
        }
        if tl == ql && tr == qr {
            return self.tree_min[node];
        }
        let mid = (tl + tr) / 2;
        let mmin = min(
            self.query_internal_min(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal_min(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        );
        mmin
    }

    fn query_internal_max(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> usize {
        if ql > qr {
            return 0;
        }
        if tl == ql && tr == qr {
            return self.tree_max[node];
        }
        let mid = (tl + tr) / 2;
        let mmax = max(
            self.query_internal_max(node * 2, tl, mid, ql, min(qr, mid)),
            self.query_internal_max(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr),
        );
        mmax
    }

    fn query_internal_is_inc(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> bool {
        if ql > qr {
            return true;
        }
        if tl == ql && tr == qr {
            return self.is_inc[node];
        }
        let mid = (tl + tr) / 2;
        let max_left = self.query_internal_max(node * 2, tl, mid, ql, min(qr, mid));
        let min_right = self.query_internal_min(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr);
        if max_left <= min_right {
            return self.query_internal_is_inc(node * 2, tl, mid, ql, min(qr, mid))
                & self.query_internal_is_inc(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr);
        } else {
            return false;
        }
    }

    // Query the inclusive range [l..r]
    pub fn query_min(&mut self, l: usize, r: usize) -> usize {
        self.query_internal_min(1, 0, self.len - 1, l, r)
    }

    pub fn query_is_inc(&mut self, l: usize, r: usize) -> bool {
        self.query_internal_is_inc(1, 0, self.len - 1, l, r)
    }

    fn update_internal(&mut self, node: usize, tl: usize, tr: usize, pos: usize, val: usize) {
        if tl == tr {
            self.tree_min[node] = val;
            self.tree_max[node] = val;
            self.is_inc[node] = true;
        } else {
            let mid = (tl + tr) / 2;
            if pos <= mid {
                self.update_internal(node * 2, tl, mid, pos, val);
            } else {
                self.update_internal(node * 2 + 1, mid + 1, tr, pos, val);
            }
            self.tree_min[node] = min(self.tree_min[node * 2], self.tree_min[node * 2 + 1]);
            self.tree_max[node] = max(self.tree_max[node * 2], self.tree_max[node * 2 + 1]);
            if self.tree_max[node * 2] <= self.tree_min[node * 2 + 1] {
                // println!(
                //     "Updating inc of {tl},{tr} = {:?}",
                //     self.is_inc[node * 2] & self.is_inc[node * 2 + 1]
                // );
                self.is_inc[node] = self.is_inc[node * 2] & self.is_inc[node * 2 + 1];
            } else {
                // println!(
                //     "Updating inc of {tl},{tr} to false since it {} > {}",
                //     self.tree_max[node * 2],
                //     self.tree_min[node * 2 + 1]
                // );
                self.is_inc[node] = false;
            }
        }
    }

    pub fn update(&mut self, pos: usize, val: usize) {
        self.update_internal(1, 0, self.len - 1, pos, val);
    }
}

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let t = read_1_number_(line, reader, 0);
    (0..t).for_each(|_te| {
        // println!("Case {_te}");
        let (n, m, q) = read_3_number_::<usize>(line, reader, 0);

        let mut mp: Vec<usize> = vec![0; n];
        let a: Vec<usize> = read_vec_template(line, reader, 0);
        (0..n).for_each(|i| mp[a[i] - 1] = i);

        let b: Vec<usize> = read_vec_template(line, reader, 0);
        let mut b: Vec<usize> = b.iter().map(|&x| mp[x - 1]).collect();
        // println!("mp = {mp:?}, b = {b:?}");

        // Create a segment tree, s[i]: first position of b[i].
        // Help answer the following: if all first position is increasing.
        let mut stree = SegmentTreeIsInc::new(n);
        let mut occ: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

        (0..n).for_each(|i| stree.update(i, m));
        b.iter().enumerate().for_each(|(i, &x)| {
            occ[x].insert(i);
            if occ[x].len() == 1 {
                // println!("Updating first occ of {x}: {i}");
                stree.update(x, i);
            }
        });

        writeln!(
            out,
            "{}",
            if stree.query_is_inc(0, n - 1) {
                "YA"
            } else {
                "TIDAK"
            }
        )
        .unwrap();

        (0..q).for_each(|_| {
            let (s, t) = read_2_number_(line, reader, 0);
            let (s, t) = (s - 1, mp[t - 1]);

            // println!("Changing b[{s}] to {t}");

            let old_val = b[s];
            // Changing the first occurence
            if occ[old_val].first().unwrap() == &s {
                // println!("Some change the occ[{old_val}]");
                occ[old_val].remove(&s);
                match occ[old_val].first() {
                    Some(&x) => {
                        stree.update(old_val, x);
                    }
                    None => {
                        stree.update(old_val, m);
                    }
                }
            } else {
                occ[old_val].remove(&s);
            }

            match occ[t].first() {
                Some(&x) => {
                    // New position is the smallest
                    if s < x {
                        // println!("occ[{t}] updating to smaller postion");
                        stree.update(t, s);
                    }
                }
                None => {
                    // Nothing, this is first
                    // println!("occ[{t}] updating to first");
                    stree.update(t, s);
                }
            }
            occ[t].insert(s);
            b[s] = t;

            writeln!(
                out,
                "{}",
                if stree.query_is_inc(0, n - 1) {
                    "YA"
                } else {
                    "TIDAK"
                }
            )
            .unwrap();
        });
    });
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
