use std::collections::VecDeque;

use crate::basic_graph::{Graph, UndirectedGraph};

type VV<T> = Vec<Vec<T>>;
type US = usize;
type UU = (US, US);

// ============================== Basic shallowest decomposition ====================

/// Structure that resembled a rooted tree.
/// Follow from root via children, guarantee the chain from root to any leaf is the shallowest.
/// Tree height is bounded by O(log n).
pub struct Arborescence {
    pub root: usize,
    pub next_root_decomp: VV<US>,
}

fn extract_chain(labels: usize, u: usize, decomp_tree: &mut VV<US>, stacks: &mut VV<US>) {
    let mut labels = labels;
    let mut u = u;
    while labels > 0 {
        let label = labels.ilog2() as usize;
        labels ^= 1 << label;
        if let Some(v) = stacks[label].pop() {
            decomp_tree[u].push(v);
            u = v;
        } else {
            break;
        }
    }
}

fn dfs_label<V, T, const DIRECTED: bool>(
    u: usize,
    p: usize,
    g: &Graph<V, T, DIRECTED>,
    forbid: &mut Vec<usize>,
    decomp_tree: &mut VV<US>,
    stacks: &mut VV<US>,
) where
    V: Clone,
    T: Clone + Ord + Default,
{
    // label is forbidden if:
    // i is in any of the children (forbid_v + 1)
    // or there exist j > i, such that j is in any 2 of (forbid_v + 1).
    let mut forbid_1 = 0;
    let mut forbid_2 = 0;
    for &(v, _) in g[u].iter() {
        if v == p {
            continue;
        }
        dfs_label(v, u, g, forbid, decomp_tree, stacks);
        let forbid_by_v = forbid[v] + 1;
        forbid_2 |= forbid_1 & forbid_by_v;
        forbid_1 |= forbid_by_v;
    }
    let bit_length = (2 * forbid_2 + 1).ilog2();
    forbid[u] = forbid_1 | ((1 << bit_length) - 1);

    let label_u = (forbid[u] + 1).trailing_zeros() as usize;
    stacks[label_u].push(u);
    for &(v, _) in g[u].iter().rev() {
        if v == p {
            continue;
        }
        extract_chain(
            (forbid[v] + 1) & ((1 << label_u) - 1),
            u,
            decomp_tree,
            stacks,
        );
    }
}

pub fn shallowest_decomposition_tree<V, T, const DIRECTED: bool>(
    root: usize,
    g: &Graph<V, T, DIRECTED>,
) -> Arborescence
where
    V: Clone,
    T: Clone + Ord + Default,
{
    let n = g.len();
    let lg = n.ilog2() as usize;
    let mut decomp_tree: VV<US> = vec![Vec::new(); n];
    let mut stacks: VV<US> = vec![Vec::new(); lg + 1];

    let mut forbid: Vec<usize> = vec![0; n];
    dfs_label(root, root, g, &mut forbid, &mut decomp_tree, &mut stacks);
    let max_label = (forbid[root] + 1).ilog2() as usize;
    let decomposition_root = stacks[max_label].pop().unwrap();
    extract_chain(
        (forbid[root] + 1) & ((1 << max_label) - 1),
        decomposition_root,
        &mut decomp_tree,
        &mut stacks,
    );
    Arborescence {
        root: decomposition_root,
        next_root_decomp: decomp_tree,
    }
}

/// BFS style break down of the decomposition.
pub fn bfs_list(tree: &Arborescence) -> Vec<usize> {
    let mut bfs: Vec<usize> = Vec::new();
    let mut cur: VecDeque<usize> = VecDeque::new();
    cur.push_back(tree.root);
    while let Some(u) = cur.pop_front() {
        bfs.push(u);
        for &v in tree.next_root_decomp[u].iter() {
            cur.push_back(v);
        }
    }
    bfs
}

/// Decomposition style travling and calculating.
/// Useful for problems asking to calculate the answer for every node as if it's root,
/// and involve combination of 2 paths from 2 different subtree.
pub fn decomposition_traveling() {
    let n = 10;

    let uelist: Vec<(US, US)> = vec![(0, 1), (1, 2)];
    let g = UndirectedGraph::from_unweighted_edges(n, &uelist);
    let tree = shallowest_decomposition_tree(0, &g);
    let bfs_list = bfs_list(&tree);

    // Needed storage for traveling.
    // TODO: Add others for storing single path label / counts.
    let mut parent = vec![usize::MAX; n];
    let mut ans = vec![1u64; n]; // Always have 1 palindrome of 1 char.
    let mut dp = vec![0u64; n];

    // Turn into d_graph for fast delete after each decomposition.
    let mut d_graph: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        for &(v, _) in g[u].iter() {
            d_graph[u].push(v);
        }
    }
    // Check each decomposition node in bfs manner:
    for &decomp_node in bfs_list.iter() {
        // Effectively delete the decomp_node (decompose at this node)
        parent[decomp_node] = decomp_node;
        for v in d_graph[decomp_node].clone().into_iter() {
            // Find, swap last and remove.
            if let Some(idx) = d_graph[v].iter().position(|&x| x == decomp_node) {
                d_graph[v].swap_remove(idx);
            }
        }

        // Get BFS order traversal for every subtree after decomposition
        let mut all_bfss: VV<US> = Vec::new();
        // With decomp node, flaten out but still in order of every child
        let mut all_bfss_flat: Vec<US> = vec![decomp_node];
        for &child in d_graph[decomp_node].iter() {
            parent[child] = decomp_node;
            let mut bfs_at_child = Vec::<US>::new();
            let mut cur: VecDeque<US> = VecDeque::new();
            cur.push_back(child);
            while let Some(u) = cur.pop_front() {
                bfs_at_child.push(u);
                for &v in d_graph[u].iter() {
                    if parent[v] == usize::MAX {
                        parent[v] = u;
                        cur.push_back(v);
                    }
                }
            }
            // Clone out and put to flat list
            all_bfss_flat.extend(bfs_at_child.clone().into_iter());
            all_bfss.push(bfs_at_child);
        }

        // TODO: Main part: calculate answer as with decomp_node as a root tree.
        // Almost always single path calculation from decomp_node to every node.
        // Usually involve combining u and parent[u] with some storage and combination.
        // For complicated problems, might invole seperation for up and down path along with height.
        for &u in all_bfss_flat.iter() {
            // Example:
            // bitmask[u] = bitmask[parent[u]] ^ a[u]; // bitmask[u]: all mask from decomp_node to u.
            // bitmask_count[bitmask[u]] += 1;
        }

        // TODO: Combined path: Work for each subtree and combine it with all calculated single path.
        for bfs in all_bfss.iter() {
            // Minus out everything for this subtree.
            // So whatever count we currently have is for all other node without this subtree.
            // Example:
            // for &u in bfs.iter() {
            //     bitmask_count[bitmask[u]] -= 1;
            // }

            // Calculate dp[u] in reverse manner: from leave to top (almost decomp node).
            // DP[u] store the result of the subtree under u for this decomposition.
            // For height, it's important to note that decomposition does not guarantee height to be <= log(n).
            // It's best to base the calculation on the height of the node u, rather have a count and check
            // for all possible height of v.
            for &u in bfs.iter().rev() {
                // Example:
                // let base = bitmask[u] ^ a[decomp_node]; // From u to before decomp_node
                // for &p in palindrome.iter() {
                //     // Find all combination that gives a palindrome
                //     // from this branch to all other branches (combination path count)
                //     dp[u] += bitmask_count[base ^ p];
                // }

                // Propagate up to parent, important since we're working backward from leaves.
                dp[parent[u]] += dp[u];
            }

            // Plus back to get ready for other subtree.
            // Example:
            // for &u in bfs.iter() {
            //     bitmask_count[bitmask[u]] += 1;
            // }
        }

        // Special calculation for the decomp_node: The single path count.
        // Example:
        // bitmask_count[a[decomp_node]] -= 1; // Minus out to avoid only one node case: already accounted for.
        // for &p in palindrome.iter() {
        //     dp[decomp_node] += bitmask_count[p];
        // }
        // dp[decomp_node] /= 2; // Avoiding double count for decomp_node since 2 branches are used twice.

        // Put to answer and reset all to prepare for the next decomposition.
        for &u in all_bfss_flat.iter() {
            ans[u] += dp[u];
            dp[u] = 0;
            parent[u] = usize::MAX;
        }
    }
}

/// Decomposition style travling and calculating, but for label on edges.
pub fn decomposition_traveling_edge() {
    let n = 10;
    let mut uelist: Vec<(US, US, US)> = Vec::new();

    let g = UndirectedGraph::from_edges(n, &uelist);
    let tree = shallowest_decomposition_tree(0, &g);
    let bfs_list = bfs_list(&tree);

    // Needed storage for traveling.
    let mut parent = vec![usize::MAX; n];
    let mut ans = vec![0; n];
    let mut dp = vec![0u64; n];
    // Turn into d_graph for fast delete after each decomposition.
    let mut d_graph: Vec<Vec<(US, US)>> = vec![Vec::new(); n];
    for u in 0..n {
        for &(v, d) in g[u].iter() {
            d_graph[u].push((v, d));
        }
    }
    // Check each decomposition node in bfs manner:
    for &decomp_node in bfs_list.iter() {
        // Effectively delete the decomp_node (decompose at this node)
        parent[decomp_node] = decomp_node;
        for (v, _) in d_graph[decomp_node].clone().into_iter() {
            // Find, swap last and remove.
            if let Some(idx) = d_graph[v].iter().position(|&(x, _)| x == decomp_node) {
                d_graph[v].swap_remove(idx);
            }
        }

        // println!("decomp = {decomp_node}");
        // Get BFS order traversal for every subtree after decomposition
        let mut all_bfss: VV<(US, US)> = Vec::new();
        // With decomp node, flaten out but still in order of every child
        let mut all_bfss_flat: Vec<(US, US)> = vec![(decomp_node, 0)];
        for &(child, d) in d_graph[decomp_node].iter() {
            parent[child] = decomp_node;
            let mut bfs_at_child = Vec::<(US, US)>::new();
            let mut cur: VecDeque<(US, US)> = VecDeque::new();
            cur.push_back((child, d));
            while let Some(uw) = cur.pop_front() {
                bfs_at_child.push(uw);
                for &(v, d) in d_graph[uw.0].iter() {
                    if parent[v] == usize::MAX {
                        parent[v] = uw.0;
                        cur.push_back((v, d));
                    }
                }
            }
            // Clone out and put to flat list
            // println!("Subtree: {bfs_at_child:?}");
            all_bfss_flat.extend(bfs_at_child.clone().into_iter());
            all_bfss.push(bfs_at_child);
        }

        // TODO: Main part: calculate answer as with decomp_node as a root tree.
        // Almost always single path calculation from decomp_node to every node.
        // Usually involve combining u and parent[u] with some storage and combination.
        // For complicated problems, might invole seperation for up and down path along with height.
        for &(u, d) in all_bfss_flat.iter() {
            // Example:
            // bitmask[u] = bitmask[parent[u]] ^ a[u]; // bitmask[u]: all mask from decomp_node to u.
            // bitmask_count[bitmask[u]] += 1;
        }

        // TODO: Combined path: Work for each subtree and combine it with all calculated single path.
        for bfs in all_bfss.iter() {
            // Minus out everything for this subtree.
            // So whatever count we currently have is for all other node without this subtree.
            // Example:
            // for &(u,_) in bfs.iter() {
            //     bitmask_count[bitmask[u]] -= 1;
            // }

            // Calculate dp[u] in reverse manner: from leave to top (almost decomp node).
            // DP[u] store the result of the subtree under u for this decomposition.
            // For height, it's important to note that decomposition does not guarantee height to be <= log(n).
            // It's best to base the calculation on the height of the node u, rather have a count and check
            // for all possible height of v.
            for &(u, _) in bfs.iter().rev() {
                // Example:
                // let base = bitmask[u] ^ a[decomp_node]; // From u to before decomp_node
                // for &p in palindrome.iter() {
                //     // Find all combination that gives a palindrome
                //     // from this branch to all other branches (combination path count)
                //     dp[u] += bitmask_count[base ^ p];
                // }

                // Propagate up to parent, important since we're working backward from leaves.
                dp[parent[u]] += dp[u];
            }

            // Plus back to get ready for other subtree.
            // Example:
            // for &(u,_) in bfs.iter() {
            //     bitmask_count[bitmask[u]] += 1;
            // }
        }

        // Special calculation for the decomp_node: The single path count.
        // Example:
        // bitmask_count[a[decomp_node]] -= 1; // Minus out to avoid only one node case: already accounted for.
        // for &p in palindrome.iter() {
        //     dp[decomp_node] += bitmask_count[p];
        // }
        // dp[decomp_node] /= 2; // Avoiding double count for decomp_node since 2 branches are used twice.

        // Put to answer and reset all to prepare for the next decomposition.
        for &(u, _) in all_bfss_flat.iter() {
            ans[u] += dp[u];
            dp[u] = 0;
            parent[u] = usize::MAX;
        }
    }
}

// ==================================== DFS single update single get ===========================

pub fn dfs_decomp(
    u: usize,
    p: usize,
    tree: &Arborescence,
    time_in: &mut Vec<usize>,
    time_out: &mut Vec<usize>,
    global_time: &mut usize,
    up: &mut VV<US>,
    lg: usize,
) {
    time_in[u] = *global_time;
    time_out[u] = *global_time;
    *global_time += 1;
    // Updating 2^i parent
    up[u][0] = p;
    for i in 1..=lg {
        up[u][i] = up[up[u][i - 1]][i - 1];
    }

    for &v in tree.next_root_decomp[u].iter() {
        dfs_decomp(v, u, tree, time_in, time_out, global_time, up, lg);
        time_out[u] = time_out[v];
    }
}

/// Check if u is parent of v
pub fn is_parent_decomp(u: usize, v: usize, time_in: &Vec<usize>, time_out: &Vec<usize>) -> bool {
    return time_in[u] <= time_in[v] && time_out[u] >= time_out[v];
}

pub fn get_lca_decomp(
    u: US,
    v: US,
    time_in: &Vec<usize>,
    time_out: &Vec<usize>,
    up: &VV<US>,
) -> usize {
    if is_parent_decomp(u, v, time_in, time_out) {
        return u;
    }
    if is_parent_decomp(v, u, time_in, time_out) {
        return v;
    }
    let mut u = u;
    for i in (0..up[0].len()).rev() {
        if !is_parent_decomp(up[u][i], v, time_in, time_out) {
            u = up[u][i];
        }
    }
    up[u][0]
}

// Fast cp to main

// /// Update a node relative to all decomposition.
// /// Assume we have a function f(u,root) that give the result.
// /// then we want to store it into dp[root]. This will be later use for calculation
// pub fn update_decomp(
//     u: usize,
//     root: usize,
//     tree: &tree_decomp::Arborescence,
//     time_in_decomp: &Vec<usize>,
//     time_out_decomp: &Vec<usize>,
//     up_decomp: &VV<US>,
//     dp: &mut Vec<US>,
// ) {
//     // println!("Updating {u}, with root_decomp = {root}");
//     if u == root {
//         dp[root] = 0; // TODO: logic here
//         return;
//     }
//     let d = 1; // TODO: f(u,root)
//     dp[root] = dp[root].min(d); // Length from d to root.

//     for &next_root in tree.next_root_decomp[root].iter() {
//         // Only update the correct subtree in decomposition.
//         if tree_decomp::is_parent_decomp(next_root, u, time_in_decomp, time_out_decomp) {
//             update_decomp(
//                 u,
//                 next_root,
//                 tree,
//                 time_in_decomp,
//                 time_out_decomp,
//                 up_decomp,
//                 dp,
//             );
//         }
//     }
// }

// /// Get the value of a node via all it decomposition.
// /// root will keep getting closer to u.
// pub fn get_decomp(
//     u: usize,
//     root: usize,
//     tree: &tree_decomp::Arborescence,
//     time_in_decomp: &Vec<usize>,
//     time_out_decomp: &Vec<usize>,
//     up_decomp: &VV<US>,
//     dp: &mut Vec<US>,
// ) -> US {
//     // println!("Query for {u}, with root_decomp = {root}");
//     if u == root {
//         return dp[root];
//     }
//     let mut ans = usize::MAX;
//     // Use the root as a possible ans, even if it can be wrong.
//     // If this root is not set, then next root related can't be an ans as well.
//     if dp[root] < usize::MAX {
//         let d = 1; // TODO: f(u,root)
//         ans = ans.min(d + dp[root]);
//         for &next_root in tree.next_root_decomp[root].iter() {
//             if tree_decomp::is_parent_decomp(next_root, u, time_in_decomp, time_out_decomp) {
//                 // println!("{u} in same subtree with {next_root}, go deeper...");
//                 ans = ans.min(get_decomp(
//                     u,
//                     next_root,
//                     tree,
//                     time_in_decomp,
//                     time_out_decomp,
//                     up_decomp,
//                     dp,
//                 ));
//             }
//         }
//     }
//     ans
// }

/// Update a node relative to all decomposition.
/// Assume we have a function f(u,root) that give the result.
/// then we want to store it into dp[root]. This will be later use for calculation
pub fn update_decomp(
    u: usize,
    root: usize,
    tree: &Arborescence,
    time_in_decomp: &Vec<usize>,
    time_out_decomp: &Vec<usize>,
    up_decomp: &VV<US>,
    dp: &mut Vec<US>,
) {
    // println!("Updating {u}, with root_decomp = {root}");
    if u == root {
        dp[root] = 0;
        return;
    }
    let d = 1; // TODO: f(u,root).
    dp[root] = dp[root].min(d); // Length from d to root.

    for &next_root in tree.next_root_decomp[root].iter() {
        // Only update the correct subtree in decomposition.
        if is_parent_decomp(next_root, u, time_in_decomp, time_out_decomp) {
            update_decomp(
                u,
                next_root,
                tree,
                time_in_decomp,
                time_out_decomp,
                up_decomp,
                dp,
            );
        }
    }
}

/// Get the value of a node via all it decomposition.
/// root will keep getting closer to u.
pub fn get_decomp(
    u: usize,
    root: usize,
    tree: &Arborescence,
    time_in_decomp: &Vec<usize>,
    time_out_decomp: &Vec<usize>,
    up_decomp: &VV<US>,
    dp: &mut Vec<US>,
) -> US {
    // println!("Query for {u}, with root_decomp = {root}");
    if u == root {
        return dp[root];
    }
    let mut ans = usize::MAX;
    // Use the root as a possible ans, even if it can be wrong.
    // If this root is not set, then next root related can't be an ans as well.
    if dp[root] < usize::MAX {
        let d = 1; // TODO: f(u, root)
        ans = ans.min(d + dp[root]);
        for &next_root in tree.next_root_decomp[root].iter() {
            if is_parent_decomp(next_root, u, time_in_decomp, time_out_decomp) {
                // println!("{u} in same subtree with {next_root}, go deeper...");
                ans = ans.min(get_decomp(
                    u,
                    next_root,
                    tree,
                    time_in_decomp,
                    time_out_decomp,
                    up_decomp,
                    dp,
                ));
            }
        }
    }
    ans
}

/// Single edit and modify using decomposition.
pub fn single_edit() {
    let n = 10;

    let uelist: Vec<(US, US)> = vec![(0, 1), (1, 2)];
    let g = UndirectedGraph::from_unweighted_edges(n, &uelist);
    let tree = shallowest_decomposition_tree(0, &g);
    // First: Need to know which subtree contains the node to quickly go there
    // Make use of the LCA pattern: if lca(u, node) == node, then it's in that subtree.
    let mut time_in_decomp = vec![0; n];
    let mut time_out_decomp = vec![0; n];
    let mut global_time_decomp = 0;
    let lg = (n as f32).log2() as usize;
    let mut up_decomp: VV<US> = vec![vec![0; lg + 1]; n];
    dfs_decomp(
        tree.root,
        tree.root,
        &tree,
        &mut time_in_decomp,
        &mut time_out_decomp,
        &mut global_time_decomp,
        &mut up_decomp,
        lg,
    );

    // Storage for the ans at every decomp_root;
    let mut dp = vec![0; n];
    // Update and get as follow, copy and modify:
    update_decomp(
        10,
        tree.root,
        &tree,
        &time_in_decomp,
        &time_out_decomp,
        &up_decomp,
        &mut dp,
    );
    get_decomp(
        10,
        tree.root,
        &tree,
        &time_in_decomp,
        &time_out_decomp,
        &up_decomp,
        &mut dp,
    );
}
