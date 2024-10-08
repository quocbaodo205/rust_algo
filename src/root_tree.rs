// Turn into a farmiliar rooted tree structure
#[allow(dead_code)]
fn dfs_root(
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
) {
    level[u] = l;
    time_in[u] = *global_time;
    time_out[u] = *global_time;
    *global_time += 1;
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
        );
        parent[v] = u;
        children[u].push(v);
        time_out[u] = time_out[v];
    });
}

// Check if u is parent of v
#[allow(dead_code)]
fn is_parent(u: usize, v: usize, time_in: &Vec<usize>, time_out: &Vec<usize>) -> bool {
    return time_in[u] <= time_in[v] && time_out[u] >= time_out[v];
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
