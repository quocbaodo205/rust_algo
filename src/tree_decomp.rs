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
    root: usize,
    chilren: VV<US>,
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
        chilren: decomp_tree,
    }
}

/// BFS style break down of the decomposition.
pub fn bfs_list(tree: &Arborescence) -> Vec<usize> {
    let mut bfs: Vec<usize> = Vec::new();
    let mut cur: VecDeque<usize> = VecDeque::new();
    cur.push_back(tree.root);
    while let Some(u) = cur.pop_front() {
        bfs.push(u);
        for &v in tree.chilren[u].iter() {
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
