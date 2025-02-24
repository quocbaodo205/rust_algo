use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    fmt::{write, Debug, Display},
    io::{self, read_to_string, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    mem::{self, swap},
    ops::{self, Add, AddAssign, Bound::*, DerefMut, RangeBounds},
    rc::Rc,
    str::FromStr,
};

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckState {
    NOTCHECK,
    CHECKING,
    CHECKED,
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
fn read_3_number<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (T, T, T) {
    let v = read_vec_template(line, reader, default);
    (v[0], v[1], v[2])
}

#[allow(dead_code)]
fn read_4_number<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (T, T, T, T) {
    let v = read_vec_template(line, reader, default);
    (v[0], v[1], v[2], v[3])
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
            if mid == 0 {
                break;
            }
            r = mid - 1;
        }
    }
    return ans;
}

#[allow(dead_code)]
fn ter_search_template(l: usize, r: usize, f: &dyn Fn(usize) -> i32) -> usize {
    let mut l = l;
    let mut r = r;

    while l <= r {
        if r - l < 3 {
            // Can no longer get mid1 and mid2
            // Check all ans in [l..r]
            let mut ans = f(r);
            let mut pos = r;
            for i in l..r {
                if f(i) > ans {
                    ans = f(i);
                    pos = i;
                }
            }
            return pos;
        }
        let mid1 = l + (r - l) / 3;
        let mid2 = r - (r - l) / 3;

        let f1 = f(mid1);
        let f2 = f(mid2);

        if f1 < f2 {
            l = mid1;
        } else {
            r = mid2;
        }
    }
    return l;
}

// Template for 2 pointer: function f will act on a range [l,r]:
// - If satisfy then advance r
// - Else advance l until satisfy
#[allow(dead_code)]
fn two_pointer_template(a: &[i32], f: &dyn Fn(usize, usize) -> bool) {
    // Range representation: [l,r)
    let mut l = 0;
    let mut r = 0;

    while l < a.len() {
        while r < a.len() && f(l, r) {
            // TODO: Process + r
            r += 1;
        }
        // Full range satisfy
        if r == a.len() {
            break;
        }

        while !f(l, r) {
            // TODO: Process - l
            l += 1;
        }
    }
}

// Template for sliding window of size d
#[allow(dead_code)]
fn sliding_windows_d(s: &[u8], d: usize, f: &dyn Fn(usize) -> usize) {
    // Process first substr
    let mut start = 0;
    let mut end = start + d - 1;
    let mut contrib = 0; // Contribution of each position
                         // TODO: Calculate first contrib of [start..=end]
    (start..=end).for_each(|i| {
        contrib += 1;
    });

    // Move up and process each substr
    while end + 1 < s.len() {
        // Minus old contrib
        contrib -= f(start);

        // Plus new contrib
        start += 1;
        end += 1;
        contrib += f(end);
    }
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

type V<T> = Vec<T>;
type VV<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;
type Map<K, V> = BTreeMap<K, V>;
type US = usize;
type UU = (US, US);
type UUU = (US, US, US);

// ========================= Math ops =================================

#[allow(dead_code)]
fn to_digit_array(a: u64) -> V<US> {
    let mut ans: V<US> = V::new();
    let mut a = a;
    while a > 0 {
        ans.push((a % 10) as usize);
        a /= 10;
    }
    ans.reverse();
    ans
}

// Calculate ceil(a/b) in int
#[allow(dead_code)]
fn ceil_int(a: u64, b: u64) -> u64 {
    let mut r = a / b;
    if a % b != 0 {
        r += 1;
    }
    r
}

// Calculate sumxor 1->n
#[allow(dead_code)]
fn sumxor(n: u64) -> u64 {
    let md = n % 4;
    match md {
        0 => n,
        1 => 1,
        2 => n + 1,
        3 => 0,
        _ => 0,
    }
}

#[allow(dead_code)]
fn sumxor_range(l: u64, r: u64) -> u64 {
    if l == 0 {
        sumxor(r)
    } else {
        sumxor(l - 1) ^ sumxor(r)
    }
}

// How many time that x === k (mod m) appear in range [l -> r]? (k < m obviously)
#[allow(dead_code)]
fn mod_in_range(l: u64, r: u64, k: u64, m: u64) -> (u64, u64, u64) {
    // Number form: x = k + a*m
    // Consider [l -> l+m): appear only once
    // First oc: l <= k + a*m < l+m
    // l - k <= a*m < l+m - k
    // (l - k) / m <= a < (l+m-k)/k
    let first_oc = if l <= k {
        k
    } else {
        k + (ceil_int(l - k, m)) * m
    };
    if first_oc > r {
        // No occurrence in range
        return (0, 0, 0);
    }
    // firstoc + a*m <= r
    // a <= (r - firstoc) / m
    let last_oc = first_oc + ((r - first_oc) / m) * m;
    // firstoc: k + a*m
    // lastoc: k + b*m
    let oc = ((last_oc - k) - (first_oc - k)) / m + 1;
    // println!("x === {k} (mod {m}) in range [{l},{r}], first_oc = {first_oc}, last_oc = {last_oc}, # = {oc}");
    (oc, first_oc, last_oc)
}

// =========================== IO for classic problems =======================

#[allow(dead_code)]
fn better_array_debug<T>(a: &V<T>)
where
    T: Debug,
{
    a.iter()
        .enumerate()
        .for_each(|(i, x)| println!("{i:4}: {x:?}"));
}

// 2 array need to have the same length
#[allow(dead_code)]
fn better_2_array_debug<T, D>(a: &V<T>, b: &V<D>)
where
    T: Debug,
    D: Debug,
{
    (0..a.len()).for_each(|i| {
        println!("{i:4}: {:?} -- {:4?}", a[i], b[i]);
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
fn read_n_and_2_array<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (T, V<T>, V<T>) {
    let n = read_1_number(line, reader, default);
    let a = read_vec_template(line, reader, default);
    let b = read_vec_template(line, reader, default);
    (n, a, b)
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
fn read_n_m_k_and_2_array<T: FromStr + Copy>(
    line: &mut String,
    reader: &mut BufReader<Stdin>,
    default: T,
) -> (T, T, T, V<T>, V<T>) {
    let (n, m, k) = read_3_number(line, reader, default);
    let a = read_vec_template(line, reader, default);
    let b = read_vec_template(line, reader, default);
    (n, m, k, a, b)
}

#[allow(dead_code)]
fn read_graph_from_edge_list(
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
fn read_graph_from_edge_list_keep(
    g: &mut VV<UU>,
    m: US,
    is_bidirectional: bool,
    line: &mut String,
    reader: &mut BufReader<Stdin>,
) {
    (0..m).for_each(|idx| {
        let (u, v) = read_2_number(line, reader, 0usize);
        let (u, v) = (u - 1, v - 1);
        g[u].push((v, idx));
        if is_bidirectional {
            g[v].push((u, idx));
        }
    });
}

// Weight first to allow sort by weight, very useful!
#[allow(dead_code)]
fn read_edge_list_with_weight(
    m: US,
    line: &mut String,
    reader: &mut BufReader<Stdin>,
) -> V<(US, US, US)> {
    let mut edges: V<(US, US, US)> = V::new();
    edges.reserve(m);
    (0..m).for_each(|_| {
        let (u, v, w) = read_3_number(line, reader, 0usize);
        let (u, v) = (u - 1, v - 1);
        edges.push((w, u, v));
    });
    edges
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

#[allow(dead_code)]
fn array_output_with_size<T>(a: &V<T>, out: &mut BufWriter<Stdout>)
where
    T: Display,
{
    write!(out, "{} ", a.len()).unwrap();
    a.iter().for_each(|x| {
        write!(out, "{x} ").unwrap();
    });
    writeln!(out).unwrap();
}

#[allow(dead_code)]
fn write_vertex_list_add1(a: &V<US>, out: &mut BufWriter<Stdout>) {
    a.iter().for_each(|x| {
        write!(out, "{} ", x + 1).unwrap();
    });
    writeln!(out).unwrap();
}

// =========================== Interactive queries =======================

#[allow(dead_code)]
fn query(l: u64, r: u64, re: &mut BufReader<Stdin>, li: &mut String) -> u64 {
    println!("? {l} {r}");
    let ans = read_1_number(li, re, 0u64);
    ans
}

// =============================================================================
// Template for segment tree that allow flexible Lazy and Value definition

// Lazy operation that allow range add or range set
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegtreeLazy {
    pub val: i32,
    // Is increase, if false then it's a set operation.
    // Remember to find all is_inc and change to false should needed.
    pub is_inc: bool,
}

impl AddAssign for SegtreeLazy {
    fn add_assign(&mut self, rhs: Self) {
        if !rhs.is_inc {
            self.val = 0;
            self.is_inc = false;
        }
        self.val += rhs.val;
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegtreeValue {
    mx: i32,
}

#[allow(dead_code)]
impl SegtreeValue {
    // Update from lazy operation
    pub fn update(&mut self, lz: SegtreeLazy, sz: usize) {
        if !lz.is_inc {
            self.mx = 0;
        }
        self.mx += lz.val;
    }
}

impl Add for SegtreeValue {
    type Output = SegtreeValue;

    fn add(self, rhs: Self) -> Self::Output {
        SegtreeValue {
            mx: max(self.mx, rhs.mx),
        }
    }
}

pub const LID: SegtreeLazy = SegtreeLazy {
    val: 0,
    is_inc: true,
};
pub const VID: SegtreeValue = SegtreeValue { mx: -1000000000 };

// ============================== Main structure ====================

#[allow(dead_code)]
pub struct SegmentTreeLazy {
    len: usize,
    tree: Vec<SegtreeValue>,
    lazy: Vec<SegtreeLazy>,
}

#[allow(dead_code)]
impl SegmentTreeLazy {
    pub fn new(n: usize) -> Self {
        SegmentTreeLazy {
            len: n,
            tree: vec![VID; 4 * n + 10],
            lazy: vec![LID; 4 * n + 10],
        }
    }

    fn build_internal(&mut self, node: usize, tl: usize, tr: usize, a: &[SegtreeValue]) {
        if tl == tr {
            self.tree[node] = a[tl];
        } else {
            let mid = (tl + tr) / 2;
            self.build_internal(node * 2, tl, mid, a);
            self.build_internal(node * 2 + 1, mid + 1, tr, a);
            self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1];
        }
    }

    pub fn build(a: &[SegtreeValue]) -> Self {
        let mut segment_tree = SegmentTreeLazy {
            len: a.len(),
            tree: vec![VID; 4 * a.len() + 10],
            lazy: vec![LID; 4 * a.len() + 10],
        };
        segment_tree.build_internal(1, 0, a.len() - 1, a);
        segment_tree
    }

    // Update current node and push lazy to children
    fn push(&mut self, node: usize, _tl: usize, _tr: usize) {
        // Update tree
        let sz = _tr + 1 - _tl;
        let lzd = self.lazy[node];
        // println!("Pushing lz = {lzd:?} to node {node}, _tl = {_tl}, tr = {_tr}");
        // Reset current lazy
        self.lazy[node] = LID;

        self.tree[node].update(lzd, sz);

        // Update lazy
        if node * 2 + 1 >= self.tree.len() {
            return;
        }
        self.lazy[node * 2] += lzd;
        self.lazy[node * 2 + 1] += lzd;
    }

    fn query_internal(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
    ) -> SegtreeValue {
        if ql > qr {
            return VID;
        }
        if tl == ql && tr == qr {
            self.push(node, tl, tr);
            return self.tree[node];
        }
        self.push(node, tl, tr);
        let mid = (tl + tr) / 2;
        let ans = self.query_internal(node * 2, tl, mid, ql, min(qr, mid))
            + self.query_internal(node * 2 + 1, mid + 1, tr, max(mid + 1, ql), qr);
        // println!("Got ans for {tl} {tr} {ql} {qr} = {ans:?}");
        ans
    }

    // Query the inclusive range [l..r]
    pub fn query(&mut self, l: usize, r: usize) -> SegtreeValue {
        self.query_internal(1, 0, self.len - 1, l, r)
    }

    // When needed, please apply transformation before passing [val]
    fn update_internal(
        &mut self,
        node: usize,
        tl: usize,
        tr: usize,
        ql: usize,
        qr: usize,
        lz: SegtreeLazy,
    ) {
        if ql > qr {
            return;
        }
        if tl == ql && tr == qr {
            self.lazy[node] += lz;
            self.push(node, tl, tr);
        } else {
            self.push(node, tl, tr);
            let mid = (tl + tr) / 2;
            self.update_internal(node * 2, tl, mid, ql, min(qr, mid), lz);
            self.update_internal(node * 2 + 1, mid + 1, tr, max(ql, mid + 1), qr, lz);
            self.tree[node] = self.tree[node * 2] + self.tree[node * 2 + 1];
            // println!("After update, tree node {node} = {:?}", self.tree[node]);
        }
    }

    // Update range [l,r] with lazy
    pub fn update(&mut self, ql: usize, qr: usize, lz: SegtreeLazy) {
        self.update_internal(1, 0, self.len - 1, ql, qr, lz);
    }
}

// =========================== End template here =======================

// Turn into a farmiliar rooted tree structure
#[allow(dead_code)]
pub fn dfs_root(
    u: usize,
    p: usize,
    l: usize,
    g: &Vec<Vec<usize>>,
    parent: &mut Vec<usize>,
    children: &mut Vec<Vec<usize>>,
    level: &mut Vec<usize>,
    time_in: &mut Vec<usize>,
    time_out: &mut Vec<usize>,
    global_time: &mut usize,
    up: &mut VV<US>,
    lg: usize,
) {
    level[u] = l;
    time_in[u] = *global_time;
    time_out[u] = *global_time;
    *global_time += 1;

    // Updating 2^i parent
    up[u][0] = p;
    for i in 1..=lg {
        up[u][i] = up[up[u][i - 1]][i - 1];
    }

    g[u].iter().for_each(|&v| {
        if v == p {
            return;
        }
        dfs_root(
            v,
            u,
            l + 1,
            g,
            parent,
            children,
            level,
            time_in,
            time_out,
            global_time,
            up,
            lg,
        );
        parent[v] = u;
        children[u].push(v);
        time_out[u] = time_out[v];
    });
}

#[allow(dead_code)]
fn dfs_tree_size(u: usize, children: &Vec<Vec<usize>>, tsize: &mut Vec<usize>) -> usize {
    tsize[u] = 1;
    children[u].iter().for_each(|&v| {
        tsize[u] += dfs_tree_size(v, children, tsize);
    });
    tsize[u]
}

#[allow(dead_code)]
fn dfs_hld_label(
    u: usize,
    children: &Vec<Vec<usize>>,
    tsize: &Vec<usize>,
    label: &mut Vec<usize>,
    global_label: &mut usize,
) {
    label[u] = *global_label;
    *global_label += 1;
    // Find heavy vertex
    let mut hv_vertex: Option<usize> = None;
    let mut best_st = 0;
    for &v in children[u].iter() {
        if tsize[v] > best_st {
            best_st = tsize[v];
            hv_vertex = Some(v);
        }
    }
    if let Some(hv) = hv_vertex {
        dfs_hld_label(hv, children, tsize, label, global_label);
        // dfs for others
        for &v in children[u].iter() {
            if v == hv {
                continue;
            }
            dfs_hld_label(v, children, tsize, label, global_label);
        }
    }
}

#[allow(dead_code)]
fn dfs_hld_parent(
    u: usize,
    children: &Vec<Vec<usize>>,
    tsize: &Vec<usize>,
    parent: &mut Vec<usize>,
) {
    // Find heavy vertex
    let mut hv_vertex: Option<usize> = None;
    let mut best_st = 0;
    for &v in children[u].iter() {
        if tsize[v] > best_st {
            best_st = tsize[v];
            hv_vertex = Some(v);
        }
    }

    if let Some(hv) = hv_vertex {
        parent[hv] = parent[u];
        dfs_hld_parent(hv, children, tsize, parent);
        // dfs for others
        for &v in children[u].iter() {
            if v == hv {
                continue;
            }
            parent[v] = v;
            dfs_hld_parent(v, children, tsize, parent);
        }
    }
}

// Query HLD but only from u to parent px
#[allow(dead_code)]
fn query_hld_to_parent(
    u: usize,
    px: usize,
    parent: &Vec<usize>,
    hld_parent: &Vec<usize>,
    label: &Vec<usize>,
    level: &Vec<usize>,
    st: &mut SegmentTreeLazy,
    is_count_last: bool,
) -> SegtreeValue {
    // println!("query hld from {u} to {px}");
    let mut ans = VID;
    let mut cur = u;
    while cur != px {
        // Explore the heavy chain
        let hp = hld_parent[cur];
        // println!("cur = {cur}, hp = {hp}");
        if hp == cur {
            // Current u -> p is a light edge, goes up light normal
            // Also query from the segtree cuz why not.
            let st_ans = st.query(label[cur], label[cur]);
            // println!("ans for {} is {:?}", label[cur], st_ans);
            ans = ans + st_ans;
            cur = parent[cur];
        } else {
            // Have a heavy chain:
            if level[hp] >= level[px] {
                // Deeper than px, can use full st[hp+1..=cur]
                // Use the +1 to avoid counting for data[hp], it will cal in the next up edge.
                let st_ans = st.query(label[hp] + 1, label[cur]);
                // println!(
                //     "ans from {} -> {} is {:?}",
                //     label[hp] + 1,
                //     label[cur],
                //     st_ans
                // );
                ans = ans + st_ans;
                cur = hp;
            } else {
                // Have to stop at px
                let st_ans = st.query(label[px] + 1, label[cur]);
                // println!(
                //     "ans from {} -> {} is {:?}",
                //     label[px] + 1,
                //     label[cur],
                //     st_ans
                // );
                ans = ans + st_ans;
                cur = px;
            }
        }
    }
    // Consider the last edge: cur -> px:
    if is_count_last {
        let st_ans = st.query(label[px], label[px]);
        // println!("ans for {} is {:?}", label[px], st_ans);
        ans = ans + st_ans;
    }
    ans
}

// Check if u is parent of v
#[allow(dead_code)]
pub fn is_parent(u: usize, v: usize, time_in: &Vec<usize>, time_out: &Vec<usize>) -> bool {
    return time_in[u] <= time_in[v] && time_out[u] >= time_out[v];
}

#[allow(dead_code)]
pub fn get_lca(u: US, v: US, time_in: &Vec<usize>, time_out: &Vec<usize>, up: &VV<US>) -> usize {
    if is_parent(u, v, time_in, time_out) {
        return u;
    }
    if is_parent(v, u, time_in, time_out) {
        return v;
    }
    let mut u = u;
    for i in (0..up[0].len()).rev() {
        if !is_parent(up[u][i], v, time_in, time_out) {
            u = up[u][i];
        }
    }
    up[u][0]
}

#[allow(dead_code)]
fn query_hld(
    u: usize,
    v: usize,
    parent: &Vec<usize>,
    hld_parent: &Vec<usize>,
    time_in: &Vec<usize>,
    time_out: &Vec<usize>,
    up: &VV<US>,
    label: &Vec<usize>,
    level: &Vec<usize>,
    st: &mut SegmentTreeLazy,
) -> SegtreeValue {
    let lca = get_lca(u, v, time_in, time_out, up);
    if lca == u {
        // Only from v to lca
        return query_hld_to_parent(v, lca, parent, hld_parent, label, level, st, true);
    }
    if lca == v {
        return query_hld_to_parent(u, lca, parent, hld_parent, label, level, st, true);
    }
    let ans_u_lca = query_hld_to_parent(u, lca, parent, hld_parent, label, level, st, true);
    let ans_v_lca_without_lca =
        query_hld_to_parent(v, lca, parent, hld_parent, label, level, st, false);
    ans_u_lca + ans_v_lca_without_lca
}

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let default = 0usize;
    // let t = read_1_number(line, reader, 0);
    // (0..t).for_each(|_te| {
    let (n, q) = read_2_number(line, reader, default);
    let data = read_vec_template(line, reader, default);

    let mut g: VV<US> = vec![V::new(); n];
    read_graph_from_edge_list(&mut g, n - 1, true, line, reader);
    // Turn tree into rooted structure
    let mut parent = vec![0; n];
    let mut children: VV<US> = vec![Vec::new(); n];
    let mut level = vec![0; n];
    let mut time_in = vec![0; n];
    let mut time_out = vec![0; n];
    let mut global_time = 0;
    let lg = (n as f32).log2() as usize;
    let mut up: VV<US> = vec![vec![0; lg + 1]; n];
    dfs_root(
        0,
        0,
        0,
        &g,
        &mut parent,
        &mut children,
        &mut level,
        &mut time_in,
        &mut time_out,
        &mut global_time,
        &mut up,
        lg,
    );

    // println!("children = {children:?}");

    // Get the tree size
    let mut tsize = vec![0; n];
    dfs_tree_size(0, &children, &mut tsize);
    // println!("got the size for each node: {tsize:?}");

    // DFS labeling: so that chain of heavy edges have concescutive label (for segment tree).
    let mut label = vec![0; n];
    let mut global_label = 1;
    dfs_hld_label(0, &children, &tsize, &mut label, &mut global_label);
    // println!("label = {label:?}");

    // DFS to find for a vertex, its longest hld chain parent.
    // All the way to the top until you hit a light edge
    // if (u,v) is a light edge, hld_parent[v] = v.
    let mut hld_parent = vec![0; n];
    hld_parent[0] = 0;
    dfs_hld_parent(0, &children, &tsize, &mut hld_parent);
    // println!("hld_parent = {hld_parent:?}");

    let mut st = SegmentTreeLazy::new(global_label + 5); // cover all the labels
    let data: V<SegtreeLazy> = data
        .into_iter()
        .map(|x| SegtreeLazy {
            val: x as i32,
            is_inc: false,
        })
        .collect();
    for (u, &x) in data.iter().enumerate() {
        st.update(label[u], label[u], x);
    }
    for _ in 0..q {
        let query = read_vec_template(line, reader, default);
        if query[0] == 1 {
            let (u, x) = (query[1] - 1, query[2]);
            st.update(
                label[u],
                label[u],
                SegtreeLazy {
                    val: x as i32,
                    is_inc: false,
                },
            );
        } else {
            let (u, v) = (query[1] - 1, query[2] - 1);
            let ans = query_hld(
                u,
                v,
                &parent,
                &hld_parent,
                &time_in,
                &time_out,
                &up,
                &label,
                &level,
                &mut st,
            );
            writeln!(out, "{}", ans.mx).unwrap();
        }
    }
    // });
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
