use crate::{
    basic_graph::{Graph, UndirectedGraph},
    utils,
};

type VV<T> = Vec<Vec<T>>;
type US = usize;
type UU = (US, US);

// Turn into a farmiliar rooted tree structure
pub fn dfs_root<V, T, const DIRECTED: bool>(
    u: usize,
    p: usize,
    l: usize,
    g: &Graph<V, T, DIRECTED>,
    parent: &mut Vec<usize>,
    children: &mut Vec<Vec<usize>>,
    level: &mut Vec<usize>,
    time_in: &mut Vec<usize>,
    time_out: &mut Vec<usize>,
    global_time: &mut usize,
    up: &mut VV<US>,
    lg: usize,
) where
    V: Clone,
    T: Clone + Ord + Default,
{
    level[u] = l;
    time_in[u] = *global_time;
    time_out[u] = *global_time;
    *global_time += 1;

    // Updating 2^i parent
    up[u][0] = p;
    for i in 1..=lg {
        up[u][i] = up[up[u][i - 1]][i - 1];
    }

    for &(v, _) in &g[u] {
        if v == p {
            continue;
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
    }
}

// Fast template to run for a graph, level start at 0
fn runner() {
    let n = 10;
    let uelist: Vec<(US, US)> = vec![(0, 1), (1, 2)];
    let g = UndirectedGraph::from_unweighted_edges(n, &uelist);
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
}

// Check if u is parent of v
pub fn is_parent(u: usize, v: usize, time_in: &Vec<usize>, time_out: &Vec<usize>) -> bool {
    return time_in[u] <= time_in[v] && time_out[u] >= time_out[v];
}

// Get path from u -> v via backward tracking.
// u must be parent of v, else will be bogus
pub fn get_path_parent(u: US, v: US, parent: &Vec<usize>) -> Vec<US> {
    let mut cur = v;
    let mut ans: Vec<US> = Vec::new();
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

// Get path from u -> v via LCA
// via u -> LCA, then -> v.
pub fn get_path_lca(
    u: US,
    v: US,
    parent: &Vec<usize>,
    time_in: &Vec<usize>,
    time_out: &Vec<usize>,
    up: &VV<US>,
) -> Vec<US> {
    if is_parent(u, v, time_in, time_out) {
        return get_path_parent(u, v, parent);
    }
    if is_parent(v, u, time_in, time_out) {
        return get_path_parent(v, u, parent);
    }
    let lca = get_lca(u, v, time_in, time_out, up);
    let mut path = get_path_parent(lca, u, parent); // lca -> u
    path.reverse(); // u -> lca
    path.pop();
    path.extend(get_path_parent(lca, v, parent).into_iter()); // Add lca -> v
    path
}

// ======================== Technical analysis ==============================

// ================== Utils ====================
pub fn dfs_tree_size(u: usize, children: &Vec<Vec<usize>>, tsize: &mut Vec<usize>) -> usize {
    tsize[u] = 1;
    children[u].iter().for_each(|&v| {
        tsize[u] += dfs_tree_size(v, children, tsize);
    });
    tsize[u]
}

pub fn dfs_max_level(
    u: usize,
    children: &Vec<Vec<usize>>,
    level: &Vec<usize>,
    max_level: &mut Vec<usize>,
) -> usize {
    max_level[u] = level[u];
    children[u].iter().for_each(|&v| {
        max_level[u] = max_level[u].max(dfs_max_level(v, children, level, max_level));
    });
    max_level[u]
}

// Dummy combine structure for hl_combine.
// Should be copy and modify with the hl_combine to be useful.
struct CombineStructure {}

impl CombineStructure {
    fn new() -> Self {
        CombineStructure {}
    }

    fn add(&mut self) {}

    // This should take ownership of the rhs and drop it.
    fn combine(&mut self, rhs: CombineStructure) {}

    fn get(&self) -> i32 {
        0
    }
}

// Heavy-light combination.
// Use when need to combine result from all children c in childen[v].
// Very often use for dynamic programming.
fn hl_combine(
    u: usize,
    children: &Vec<Vec<usize>>,
    tsize: &Vec<usize>,
    combine_structure: &mut CombineStructure,
    dp: &mut Vec<i32>,
) {
    // Leaf logic: usually only need to add something.
    if children[u].is_empty() {
        combine_structure.add();
        return;
    }

    // Figure out heavy child
    let mut hv = children[u][0];
    for &v in children[u].iter() {
        if tsize[v] > tsize[hv] {
            hv = v;
        }
    }

    // Always work for heavy first.
    hl_combine(hv, children, tsize, combine_structure, dp);

    for &v in children[u].iter() {
        if v == hv {
            continue;
        } else {
            // Work for light later to ensure result from light does not affect heavy.
            let mut ns = CombineStructure::new();
            hl_combine(v, children, tsize, &mut ns, dp);
            combine_structure.combine(ns);
        }
    }

    dp[u] = combine_structure.get();
    combine_structure.add();
}

// ================== Counter ====================

/// Count the number of non-empty set for a rooted tree,
/// where there is no pair (u,v) that u is the ancestor of v.
/// Meaning every node is pair-wise separated.
pub fn dfs_count_set_separated_pair(
    u: usize,
    children: &Vec<Vec<usize>>,
    count: &mut Vec<usize>,
) -> usize {
    let mut ans = 1;
    children[u].iter().for_each(|&v| {
        ans *= dfs_count_set_separated_pair(v, children, count) + 1;
    });
    count[u] = ans;
    ans
}

// Count number of set S size k, so that sum distance from u to each element node in set is minimum.
pub fn count_set_with_min_distance_to_u(
    u: usize,
    children: &Vec<Vec<usize>>,
    tsize: &mut Vec<usize>,
    k: usize,
) -> usize {
    let mut ans = 0;
    if k == 1 {
        // Only possible set is the set contain u
        ans += 1;
    } else {
        // Pick k branches, each branch pick 1 element to make set
        // = given an array, calculate sum( product of k elements )
        let sub_state: Vec<usize> = children[u].iter().map(|&v| tsize[v]).collect();
        ans += utils::sum_of_product(&sub_state, k);
        // Pick u inside the set, and pick k-1 branches
        ans += utils::sum_of_product(&sub_state, k - 1);
    }
    ans
}

// Count for each vertex w, number of pair (u,v) so that w is the lca of u and v.
pub fn pair_lca(tsize: &mut Vec<usize>, n: usize, children: &mut Vec<Vec<usize>>) {
    // count_pair(w) = Sum( u = child[w] | tsize[u] * (tsize[w] - 1 - tsize[u]) );
    let count_pair: Vec<u64> = (0..n)
        .map(|w| {
            children[w]
                .iter()
                .map(|&u| tsize[u] as u64 * (tsize[w] as u64 - 1 - tsize[u] as u64))
                .sum::<u64>()
        })
        .collect();
}

// ================== Sumation ====================

// Calculating Sum(u < v | min(level[u], level[v]))
pub fn sum_min(level: &mut Vec<usize>, tsize: &mut Vec<usize>, n: usize) {
    // Count how many node with level = l, put into an array for suffix sum
    let maxl = *level.iter().max().unwrap();
    let mut count_level = vec![0; maxl + 1];
    level.iter().for_each(|&l| count_level[l] += 1);
    let mut suffix_sum_l: Vec<u64> = count_level
        .iter()
        .rev()
        .scan(0, |ssum, &c| {
            *ssum += c;
            Some(*ssum)
        })
        .collect();
    suffix_sum_l.reverse();

    let mut ans = 0u64;
    // 2*min(lu, lv): lv >= lu and v is not decendant of u
    (0..n).for_each(|u| {
        ans += 2 * level[u] as u64 * (suffix_sum_l[level[u]] - tsize[u] as u64);
    });

    // Minus out lu == lv being counted twice
    (1..=maxl).for_each(|k| {
        ans -= 2 * k as u64 * ((count_level[k] * (count_level[k] - 1)) / 2);
    });
}

// Calculating Sum(u < v | level[lca(u,v)])
#[allow(dead_code)]
fn sum_lca(
    level: &mut Vec<usize>,
    tsize: &mut Vec<usize>,
    n: usize,
    children: &mut Vec<Vec<usize>>,
) {
    // count_pair(w) = Sum( u = child[w] | tsize[u] * (tsize[w] - 1 - tsize[u]) );
    let count_pair: Vec<u64> = (0..n)
        .map(|w| {
            children[w]
                .iter()
                .map(|&u| tsize[u] as u64 * (tsize[w] as u64 - 1 - tsize[u] as u64))
                .sum::<u64>()
        })
        .collect();
    // (2*l[w] + 1) * (#pair with w as lca)
    let mut snd = 0;
    (0..n).for_each(|w| {
        snd += (2 * level[w] as u64 + 1) * count_pair[w];
    });
    // snd / 2 due to over count (u,v) and (v,u)
    snd /= 2;
}

// ============================ Other problems ============================

// Position based check to see if it's a good dfs order
pub fn check_dfs_order(
    pos: usize,
    dfs_order: &Vec<usize>,
    parent: &Vec<usize>,
    time_in: &Vec<usize>,
    time_out: &Vec<usize>,
) -> bool {
    let u = dfs_order[pos];
    if pos == 0 {
        if u != 0 {
            return false;
        }
    }
    if u == 0 {
        if pos != 0 {
            return false;
        }
    }
    let v = dfs_order[pos + 1];
    // u need to be in the subtree of parent[v]
    let pa = parent[v];
    if pa == u {
        return true;
    }
    if time_in[u] != time_out[u] {
        return false;
    }

    return is_parent(pa, u, &time_in, &time_out);
    // return !(time_in[u] < time_in[pa] || time_in[u] > time_out[pa]);
}

pub fn dfs_reroot<V, T, const DIRECTED: bool>(
    u: usize,
    p: usize,
    g: &Graph<V, T, DIRECTED>,
    state: &mut Vec<usize>,
) -> usize
where
    V: Clone,
    T: Clone + Ord + Default,
{
    let mut ans = 0;
    // TODO: Calculate ans if root at u
    for &(v, _) in &g[u] {
        if v == p {
            continue;
        }
        let old_state_u = state[u];
        let old_state_v = state[v];

        // TODO: Making root at v
        let state_u_without_v = state[u] - state[u];
        let state_v_with_u = state[v] + state_u_without_v; // TODO: What?

        // Rerooting at v and process
        state[u] = state_u_without_v;
        state[v] = state_v_with_u;
        ans += dfs_reroot(v, u, g, state);
        // Return old state
        state[u] = old_state_u;
        state[v] = old_state_v;
    }
    ans
}
