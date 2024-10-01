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

#[allow(dead_code)]
struct DSU {
    n: usize,
    parent: Vec<usize>,
    size: Vec<usize>,
}

#[allow(dead_code)]
impl DSU {
    pub fn new(sz: usize) -> Self {
        DSU {
            n: sz,
            parent: (0..sz).collect(),
            size: vec![0; sz],
        }
    }

    pub fn find_parent(&mut self, u: usize) -> usize {
        if self.parent[u] == u {
            return u;
        }
        self.parent[u] = self.find_parent(self.parent[u]);
        return self.parent[u];
    }

    pub fn union(&mut self, u: usize, v: usize) {
        let mut pu = self.find_parent(u);
        let mut pv = self.find_parent(v);
        if pu == pv {
            return;
        }
        if self.size[pu] > self.size[pv] {
            swap(&mut pu, &mut pv);
        }
        self.size[pu] += self.size[pv];
        self.parent[pv] = pu;
    }
}

// =========================== End template here =======================
fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let v = read_vec_template(line, reader, 0);
    let t = v[0];
    for _te in 0..t {
        let v = read_vec_template(line, reader, 0);
        let (n, m) = (v[0], v[1]);

        let mut dsu = DSU::new(n + 1);

        // Slip start / end of each query by categories
        let mut start_cnt: Vec<Vec<i32>> = vec![vec![0; 11]; n + 1];
        let mut end_cnt: Vec<Vec<i32>> = vec![vec![0; 11]; n + 1];

        let mut dp: Vec<Vec<i32>> = vec![vec![0; 11]; n + 1];
        let mut id: Vec<Vec<usize>> = vec![vec![0; 11]; n + 1];

        // Initiate ID
        (0..=n).for_each(|i| {
            (1..=10).for_each(|j| {
                id[i][j] = i;
            });
        });

        (0..m).for_each(|_| {
            let v: Vec<usize> = read_vec_template(line, reader, 0);
            let (a, d, k) = (v[0], v[1], v[2]);
            start_cnt[a][d] += 1;
            end_cnt[a + k * d][d] += 1;
        });

        (1..=n).for_each(|i| {
            (1..=10).for_each(|j| {
                dp[i][j] = start_cnt[i][j] - end_cnt[i][j];
                if i < j + 1 {
                    return;
                }
                if dp[i - j][j] > 0 {
                    dsu.union(id[i - j][j], i);
                    id[i][j] = id[i - j][j];
                    dp[i][j] += dp[i - j][j];
                }
            });
        });

        let ans = (1..=n).filter(|&u| dsu.find_parent(u) == u).count();
        writeln!(out, "{ans}").unwrap();
    }
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
