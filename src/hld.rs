use crate::{
    basic_graph::{Graph, UndirectedGraph},
    // segment_tree::{SegmentTreeLazy, SegtreeLazy, SegtreeValue, VID},
};

// ======================== Heavy-light decomposition ==============================
// Useful for vertex / edge update and queries involve paths

#[allow(dead_code)]
fn dfs_hld_label(
    u: usize,
    children: &Vec<Vec<usize>>,
    tsize: &Vec<usize>,
    label: &mut Vec<usize>,
    max_label: &mut Vec<usize>,
    global_label: &mut usize,
) {
    label[u] = *global_label;
    max_label[u] = *global_label;
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
        dfs_hld_label(hv, children, tsize, label, max_label, global_label);
        max_label[u] = max(max_label[u], max_label[hv]);
        // dfs for others
        for &v in children[u].iter() {
            if v == hv {
                continue;
            }
            dfs_hld_label(v, children, tsize, label, max_label, global_label);
            max_label[u] = max(max_label[u], max_label[v]);
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

// Update a chain from parent to u
// For edge: label[u] is for edge(u -> parent[u]).
#[allow(dead_code)]
fn update_hld_to_parent(
    u: usize,
    px: usize,
    parent: &Vec<usize>,
    hld_parent: &Vec<usize>,
    label: &Vec<usize>,
    level: &Vec<usize>,
    st: &mut SegmentTreeLazy,
    is_count_last: bool,
    update_val: SegtreeLazy,
) {
    // println!("update hld from {u} to {px}");
    let mut cur = u;
    while cur != px {
        // Explore the heavy chain
        let hp = hld_parent[cur];
        // println!("cur = {cur}, hp = {hp}");
        if hp == cur {
            // Current u -> p is a light edge, goes up light normal
            st.update(label[cur], label[cur], update_val);
            cur = parent[cur];
        } else {
            // Have a heavy chain:
            if level[hp] >= level[px] {
                // Deeper than px, can use full st[hp+1..=cur]
                // Use the +1 to avoid counting for data[hp], it will cal in the next up edge.
                st.update(label[hp] + 1, label[cur], update_val);
                // println!(
                //     "update from {} -> {}",
                //     label[hp] + 1,
                //     label[cur],
                // );
                cur = hp;
            } else {
                // Have to stop at px
                let st_ans = st.update(label[px] + 1, label[cur], update_val);
                cur = px;
            }
        }
    }
    // Consider the last edge: cur -> px:
    if is_count_last {
        let st_ans = st.update(label[px], label[px], update_val);
        // println!("ans for {} is {:?}", label[px], st_ans);
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

#[allow(dead_code)]
fn hld() {
    let n = 10;
    let uelist: Vec<(US, US)> = vec![(0, 1), (1, 2)];
    let g = UndirectedGraph::from_unweighted_edges(n, &uelist);
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

    // Get the tree size
    let mut tsize = vec![0; n];
    dfs_tree_size(0, &children, &mut tsize);

    // DFS labeling: so that chain of heavy edges have concescutive label (for segment tree).
    // Also get the max label of the subchild of v: useful to update the whole subtree at once!
    let mut label = vec![0; n];
    let mut max_label = vec![0; n];
    let mut global_label = 0;
    dfs_hld_label(
        0,
        &children,
        &tsize,
        &mut label,
        &mut max_label,
        &mut global_label,
    );

    // DFS to find for a vertex, its longest hld chain parent.
    // All the way to the top until you hit a light edge
    // if (u,v) is a light edge, hld_parent[v] = v.
    let mut hld_parent = vec![0; n];
    hld_parent[0] = 0;
    dfs_hld_parent(0, &children, &tsize, &mut hld_parent);

    // Build a segment tree from a list of data for vertex v
    let data = vec![0; n];
    let mut st = SegmentTreeLazy::new(global_label + 5); // cover all the labels
    for (u, &x) in data.iter().enumerate() {
        st.update(
            label[u],
            label[u],
            SegtreeLazy {
                val: x,
                is_inc: true,
            },
        );
    }
    // Sample update:
    let (u, x) = (5, 100);
    st.update(
        label[u],
        label[u],
        SegtreeLazy {
            val: x as i64,
            is_inc: false,
        },
    );
    // Sample query:
    let u = 0;
    let v = 7;
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
}
