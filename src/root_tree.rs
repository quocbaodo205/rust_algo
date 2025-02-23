type V<T> = Vec<T>;
type VV<T> = V<V<T>>;
type US = usize;
type UU = (US, US);

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

// Fast template to run for a graph, level start at 0
#[allow(dead_code)]
fn runner() {
    let n = 10;
    let mut g: VV<US> = vec![V::new(); n];
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
#[allow(dead_code)]
pub fn is_parent(u: usize, v: usize, time_in: &Vec<usize>, time_out: &Vec<usize>) -> bool {
    return time_in[u] <= time_in[v] && time_out[u] >= time_out[v];
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

// Get path from u -> v via LCA
// via u -> LCA, then -> v.
#[allow(dead_code)]
pub fn get_lca(u: US, v: US, time_in: &Vec<usize>, time_out: &Vec<usize>, up: &VV<US>) -> usize {
    if is_parent(u, v, time_in, time_out) {
        return u;
    }
    if is_parent(v, u, time_in, time_out) {
        return v;
    }
    let mut u = u;
    for i in (0..up.len()).rev() {
        if !is_parent(up[u][i], v, time_in, time_out) {
            u = up[u][i];
        }
    }
    up[u][0]
}

// Get path from u -> v via LCA
// via u -> LCA, then -> v.
#[allow(dead_code)]
pub fn get_path_lca(
    u: US,
    v: US,
    parent: &Vec<usize>,
    time_in: &Vec<usize>,
    time_out: &Vec<usize>,
    up: &VV<US>,
) -> V<US> {
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

#[allow(dead_code)]
// Position based check to see if it's a good dfs order
fn check_dfs_order(
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

#[allow(dead_code)]
fn dfs_tree_size(u: usize, children: &Vec<Vec<usize>>, tsize: &mut Vec<usize>) -> usize {
    tsize[u] = 1;
    children[u].iter().for_each(|&v| {
        tsize[u] += dfs_tree_size(v, children, tsize);
    });
    tsize[u]
}

// ======================== Technical analysis ==============================

// Calculating Sum(u < v | min(level[u], level[v]))
#[allow(dead_code)]
fn sum_min(level: &mut Vec<usize>, tsize: &mut Vec<usize>, n: usize) {
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

// Count for each vertex w, number of pair (u,v) so that w is the lca of u and v.
#[allow(dead_code)]
fn pair_lca(tsize: &mut Vec<usize>, n: usize, children: &mut Vec<Vec<usize>>) {
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

#[allow(dead_code)]
fn dfs_reroot(u: usize, p: usize, g: &Vec<Vec<usize>>, state: &mut Vec<usize>) -> usize {
    let mut ans = 0;
    // TODO: Calculate ans if root at u
    g[u].iter().for_each(|&v| {
        if v == p {
            return;
        }
        let old_state_u = state[u];
        let old_state_v = state[v];

        // TODO: Making root at v
        let mut state_u_without_v = state[u];
        let mut state_v_with_u = 10; // TODO: What?

        // Rerooting at v and process
        state[u] = state_u_without_v;
        state[v] = state_v_with_u;
        ans += dfs_reroot(v, u, g, state);
        // Return old state
        state[u] = old_state_u;
        state[v] = old_state_v;
    });
    ans
}
