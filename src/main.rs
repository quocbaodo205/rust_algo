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

// Compressed trie to use when string is just too long.
// Label are just reference so it's even more efficient.
type CTrieLink<'a> = Option<Box<CTrieNode<'a>>>;

struct CTrieNode<'a> {
    val: i32,
    children: [CTrieLink<'a>; 26],
    label: &'a [u8],
}

const CREPEAT: Option<Box<CTrieNode>> = None;

impl<'a> CTrieNode<'a> {
    pub fn new(val: i32, label: &'a [u8]) -> Self {
        CTrieNode {
            val,
            children: [CREPEAT; 26],
            label,
        }
    }
}

pub struct CTrie<'a> {
    head: CTrieNode<'a>,
}

fn prefix_pos(label: &[u8], st: &[u8]) -> usize {
    let mut pos = st.len();
    for (i, &c) in st.iter().enumerate() {
        if label.len() <= i {
            pos = i;
            break;
        }
        if c as u8 != label[i] {
            pos = i;
            break;
        }
    }
    pos
}

impl<'a> CTrie<'a> {
    pub fn new(x: &'a [u8]) -> Self {
        CTrie {
            head: CTrieNode::new(0, x),
        }
    }

    pub fn insert(&mut self, st: &'a [u8]) {
        let mut cur = &mut self.head;
        let mut rst = st;

        loop {
            let pos = prefix_pos(&cur.label, rst);
            // Case 1: rst is a prefix of label
            if pos == rst.len() {
                if cur.label.len() == rst.len() {
                    cur.val += 1;
                    break;
                }

                let c_old = (cur.label[pos] - b'a') as usize;
                let old_child_label = &cur.label[pos..];

                // When break, combine all other children...
                // Dance!
                let mut new_node = Box::new(CTrieNode::new(cur.val, old_child_label));
                (0..26).for_each(
                    |c_idx| match mem::replace(&mut cur.children[c_idx], CREPEAT) {
                        Some(nd) => {
                            new_node.children[c_idx] = Some(nd);
                        }
                        None => {}
                    },
                );
                cur.children[c_old] = Some(new_node);
                cur.label = &cur.label[..pos];
                cur.val += 1;
                break;
            }
            // Case 2: same prefix < label -> break up the label and add 2 childrens
            if cur.label.len() > pos {
                let c_old = (cur.label[pos] - b'a') as usize;
                let c_new = (rst[pos] - b'a') as usize;

                let old_label = &cur.label[..pos];
                let old_child_label = &cur.label[pos..];
                let new_child_label = &rst[pos..];

                let mut new_node = Box::new(CTrieNode::new(cur.val, old_child_label));
                (0..26).for_each(
                    |c_idx| match mem::replace(&mut cur.children[c_idx], CREPEAT) {
                        Some(nd) => {
                            new_node.children[c_idx] = Some(nd);
                        }
                        None => {}
                    },
                );
                cur.children[c_old] = Some(new_node);

                // New stuff always 1
                cur.children[c_new] = Some(Box::new(CTrieNode::new(1, new_child_label)));

                cur.val += 1; // count # prefix
                cur.label = old_label;
                break;
            } else if cur.label.len() <= pos {
                // Case 3: same prefix > label -> Keep comparing with chilren with a reduced rst
                let c_new = (rst[pos] - b'a') as usize;
                if cur.children[c_new].is_none() {
                    // Add the whole (new so it's 1)
                    cur.val += 1;
                    cur.children[c_new] = Some(Box::new(CTrieNode::new(1, &rst[pos..])));
                    break;
                } else {
                    cur.val += 1;
                    cur = cur.children[c_new].as_mut().unwrap();
                    rst = &rst[pos..];
                }
            }
        }
    }

    pub fn get(&mut self, st: &[u8]) -> i32 {
        let mut cur = &mut self.head;
        let mut rst = st;

        loop {
            let pos = prefix_pos(&cur.label, rst);
            // Case 1: rst is a prefix of label
            if pos == rst.len() {
                // println!("get case 1");
                return cur.val;
            }
            // Case 2: same prefix < label -> Cannot be found
            if cur.label.len() > pos {
                return 0;
            } else if cur.label.len() <= pos {
                // Case 3: same prefix > label -> Keep comparing with chilren with a reduced rst
                let c_new = (rst[pos] - b'a') as usize;
                if cur.children[c_new].is_none() {
                    return 0;
                } else {
                    cur = cur.children[c_new].as_mut().unwrap();
                    rst = &rst[pos..];
                }
            }
        }
    }

    // Emulate 1 by 1 child move for more intuitive search
    pub fn get_1b1(&mut self, st: &[u8]) -> i32 {
        let mut cur = &mut self.head;
        let mut j = 0;
        for &c in st.iter() {
            let idx = (c - b'a') as usize;
            if j < cur.label.len() {
                if cur.label[j] != c {
                    return 0;
                }
                j += 1;
            } else {
                if cur.children[idx].is_none() {
                    return 0;
                }
                cur = cur.children[idx].as_mut().unwrap();
                // First char is guarantee match alr, j = 1
                j = 1;
            }
        }
        return cur.val;
    }

    pub fn get_ans(&mut self, st: &[u8]) -> u64 {
        let mut ans: u64 = 0;
        let mut cur = &mut self.head;
        let mut j = 0;
        for i in (0..st.len()).rev() {
            let idx = (st[i] - b'a') as usize;
            // Count how many other nodes != c
            let mut not_c = cur.val;
            if j < cur.label.len() {
                // Equivilant to no children
                if cur.label[j] != st[i] {
                    ans += (not_c as u64) * (i + 1) as u64;
                    break;
                }
                // Has children = idx
                j += 1;
                not_c -= cur.val;
                ans += (not_c as u64) * (i + 1) as u64;
            } else {
                if cur.children[idx].is_none() {
                    ans += (not_c as u64) * (i + 1) as u64;
                    break;
                }
                cur = cur.children[idx].as_mut().unwrap();
                // First char is guarantee match alr, j = 1
                j = 1;
                not_c -= cur.val;
                ans += (not_c as u64) * (i + 1) as u64;
            }
        }
        ans
    }
}

// =========================== End template here =======================

type V<T> = Vec<T>;
type V2<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    // let t = read_1_number_(line, reader, 0);
    // (0..t).for_each(|_te| {});
    let m = read_1_number_(line, reader, 0usize);
    let mut v: V<V<u8>> = V::new();
    let emp: V<u8> = V::new();
    let mut ctrie = CTrie::new(&emp);
    (0..m).for_each(|_| {
        let s = read_line_str_as_vec_template(line, reader);
        v.push(s);
    });
    for s in v.iter() {
        ctrie.insert(&s);
    }
    let ans: u64 = v.iter().map(|s| ctrie.get_ans(s)).sum();
    writeln!(out, "{}", ans * 2).unwrap();
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
