use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    fmt::{write, Debug, Display},
    io::{self, read_to_string, BufRead, BufReader, BufWriter, Stdin, Stdout, Write},
    mem::{self, swap},
    ops::{self, Bound::*, DerefMut, RangeBounds},
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

// =========================== End template here =======================

// Check if u is parent of v
#[allow(dead_code)]
pub fn is_parent(u: usize, v: usize, time_in: &Vec<usize>, time_out: &Vec<usize>) -> bool {
    return time_in[u] <= time_in[v] && time_out[u] >= time_out[v];
}

// Find the list of span edge, these edges form a spanning tree.
// All other edges are called back edges, the back edge always point from u to it's sub tree.
#[allow(dead_code)]
fn dfs_tree(
    u: US,
    p: US,
    g: &VV<UU>,
    state: &mut V<CheckState>,
    span_edges: &mut V<UUU>,
    back_edges: &mut V<UUU>,
) {
    state[u] = CheckState::CHECKED;
    for &(v, idx) in g[u].iter() {
        if v == p {
            continue;
        }
        if state[v] != CheckState::CHECKED {
            span_edges.push((u, v, idx));
            dfs_tree(v, u, g, state, span_edges, back_edges);
        } else {
            back_edges.push((u, v, idx));
        }
    }
}

// Turn into a farmiliar rooted tree structure
#[allow(dead_code)]
pub fn dfs_root(
    u: usize,
    p: usize,
    l: usize,
    g: &Vec<Vec<UU>>,
    parent: &mut Vec<UU>,
    children: &mut Vec<Vec<usize>>,
    level: &mut Vec<usize>,
    time_in: &mut Vec<usize>,
    time_out: &mut Vec<usize>,
    global_time: &mut usize,
) {
    level[u] = l;
    time_in[u] = *global_time;
    time_out[u] = *global_time;
    *global_time += 1;

    g[u].iter().for_each(|&(v, idx)| {
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
        );
        parent[v] = (u, idx);
        children[u].push(v);
        time_out[u] = time_out[v];
    });
}

// Get path from u -> v via backward tracking.
// u must be parent of v, else will be bogus
#[allow(dead_code)]
pub fn get_path_parent(u: US, v: US, parent: &Vec<usize>) -> V<US> {
    let mut cur = v;
    let mut ans: V<US> = V::new();
    ans.push(v);
    if u == v {
        return ans;
    }
    while parent[cur] != u {
        let p = parent[cur];
        cur = p;
        ans.push(cur);
    }
    ans.push(u);
    ans.reverse();
    ans
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
            size: vec![1; sz],
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

    pub fn count_set(&mut self) -> usize {
        (0..self.n).filter(|&u| self.find_parent(u) == u).count()
    }
}

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<Stdout>) {
    let default = 0usize;
    // let t = read_1_number(line, reader, 0);
    // (0..t).for_each(|_te| {
    let (n, m) = read_2_number(line, reader, default);
    let mut g: VV<UU> = vec![V::new(); n];
    read_graph_from_edge_list_keep(&mut g, m, true, line, reader);
    // A DSU of edges: and edges with idx has DSU[idx] = x, meaning it has x cycle containing it.
    let mut dsu = DSU::new(m + 1);
    let mut index = vec![0; n];
    let mut ans: V<US> = V::new();
    let mut state = vec![CheckState::NOTCHECK; n];
    // Graph could be un-connected.
    for st in 0..n {
        if state[st] == CheckState::CHECKED {
            continue;
        }
        let mut span_edges: V<UUU> = V::new();
        let mut back_edges: V<UUU> = V::new();
        dfs_tree(st, st, &g, &mut state, &mut span_edges, &mut back_edges);
        // Turn it into a tree structure with parent and shit (very often use!!)
        // Get a list of explored vertex and compress them.
        let mut unique_val: Set<US> = Set::new();
        for &(u, v, _) in span_edges.iter() {
            unique_val.insert(u);
            unique_val.insert(v);
        }
        for &(u, v, _) in back_edges.iter() {
            unique_val.insert(u);
            unique_val.insert(v);
        }
        let unique_val_arr: V<US> = unique_val.iter().cloned().collect();
        let nn = unique_val_arr.len();
        if nn == 0 {
            // No value, no edge, nothing to do!
            continue;
        }
        let span_edges: V<UUU> = span_edges
            .iter()
            .map(|&(u, v, idx)| {
                (
                    lower_bound_pos(&unique_val_arr, u),
                    lower_bound_pos(&unique_val_arr, v),
                    idx,
                )
            })
            .collect();
        let back_edges: V<UUU> = back_edges
            .iter()
            .map(|&(u, v, idx)| {
                (
                    lower_bound_pos(&unique_val_arr, u),
                    lower_bound_pos(&unique_val_arr, v),
                    idx,
                )
            })
            .collect();

        let st = lower_bound_pos(&unique_val_arr, st);
        let mut gp: VV<UU> = vec![V::new(); nn];
        for &(u, v, idx) in span_edges.iter() {
            gp[u].push((v, idx));
            gp[v].push((u, idx));
        }
        let mut parent = vec![(0, 0); nn];
        let mut children: VV<US> = vec![Vec::new(); nn];
        let mut level = vec![0; nn];
        let mut time_in = vec![0; nn];
        let mut time_out = vec![0; nn];
        let mut global_time = 1;
        dfs_root(
            st,
            st,
            0,
            &gp,
            &mut parent,
            &mut children,
            &mut level,
            &mut time_in,
            &mut time_out,
            &mut global_time,
        );
        // Fix the back edge list, since they also contain duplicates.
        let mut back_edges: V<UUU> = back_edges
            .iter()
            .cloned()
            .filter(|&(u, v, _)| level[u] < level[v])
            .collect();
        // println!("span edges: {span_edges:?}");
        back_edges.sort_by_key(|(u, _, _)| level[*u]);
        // println!("back edges: {back_edges:?}");
        for &(u, v, idx) in back_edges.iter() {
            // For each back-edge (x,y), goes from y upward, each generate and edge (x',y').
            // obviously (x',y') can't be a bridge. Mark it as unable.
            let mut cur = v;
            index[unique_val_arr[v]] = idx + 1;
            while parent[cur].0 != u {
                let p = parent[cur].0;
                if index[unique_val_arr[p]] != 0 {
                    // This chain is already explored before, no need to go further!
                    dsu.union(idx + 1, index[unique_val_arr[p]]);
                    break;
                }
                index[unique_val_arr[p]] = idx + 1;
                cur = p;
            }
            // last edge: cur -> p (before break or can be u)
            if parent[cur].0 != cur {
                // Either index is fine
                index[unique_val_arr[parent[cur].0]] = idx + 1;
            }
        }
        for &(u, v, idx) in back_edges.iter() {
            let group_p = dsu.find_parent(idx + 1);
            if dsu.size[group_p] == 1 {
                // Only 1 cycle passed through
                let mut cur = v;
                while parent[cur].0 != u {
                    let (p, edge_idx) = parent[cur];
                    ans.push(edge_idx + 1);
                    cur = p;
                }
                // last edge: cur -> p (before break or can be u)
                if parent[cur].0 != cur {
                    let (p, edge_idx) = parent[cur];
                    ans.push(edge_idx + 1);
                }
                ans.push(idx + 1);
            }
        }
    }
    // println!("DSU size: {:?}", dsu.size);
    writeln!(out, "{}", ans.len()).unwrap();
    ans.sort();
    array_output(&ans, out);
    // });
}

fn main() {
    let mut reader = BufReader::new(io::stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(io::stdout());

    solve(&mut reader, &mut line, &mut out);
}
