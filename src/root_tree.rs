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
) {
    level[u] = l;
    g[u].iter().for_each(|&v| {
        if v == p {
            return;
        }
        parent[v] = u;
        children[u].push(v);
        dfs_root(v, u, l + 1, g, parent, children, level);
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
