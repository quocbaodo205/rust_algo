use std::{cmp::Ordering, collections::BTreeSet};

use crate::{
    basic_graph::{CheckState, DirectedGraph, Graph, UndirectedGraph},
    root_tree,
};

type VV<T> = Vec<Vec<T>>;
type Set<T> = BTreeSet<T>;
type US = usize;
type UU = (US, US);

#[allow(dead_code)]
fn lower_bound_pos<T: Ord + PartialOrd>(a: &Vec<T>, search_value: T) -> usize {
    a.binary_search_by(|e| match e.cmp(&search_value) {
        Ordering::Equal => Ordering::Greater,
        ord => ord,
    })
    .unwrap_err()
}

// ============================ dfs tree magic =========================

// Find the list of span edge, these edges form a spanning tree.
// All other edges are called back edges, the back edge always point from u to it's sub tree.
#[allow(dead_code)]
fn dfs_tree<V, T, const DIRECTED: bool>(
    u: US,
    p: US,
    g: &Graph<V, T, DIRECTED>,
    state: &mut Vec<CheckState>,
    span_edges: &mut Vec<UU>,
    back_edges: &mut Vec<UU>,
) where
    V: Clone,
    T: Clone + Ord + Default,
{
    state[u] = CheckState::CHECKED;
    for &(v, _) in &g[u] {
        if v == p {
            continue;
        }
        if state[v] != CheckState::CHECKED {
            span_edges.push((u, v));
            dfs_tree(v, u, g, state, span_edges, back_edges);
        } else {
            back_edges.push((u, v));
        }
    }
}

// A span-edge (u,v) can be a bridge, if no back-edge connect ancestor of (u,v) to decendant of (u,v).
// Assume u is parent of v (in dfs tree), that means no back-edge from ancestors(u) connect to subchildren(v)
// Also a template for other stuff, so it's kinda messy right now.
#[allow(dead_code)]
fn find_bridges() {
    let n = 10;
    let uelist: Vec<(US, US)> = vec![(0, 1), (1, 2)];
    let g = UndirectedGraph::from_unweighted_edges(n, &uelist);
    let mut state = vec![CheckState::NOTCHECK; n];
    // Graph could be un-connected.
    for st in 0..n {
        if state[st] == CheckState::CHECKED {
            continue;
        }
        let mut span_edges: Vec<UU> = Vec::new();
        let mut back_edges: Vec<UU> = Vec::new();
        dfs_tree(0, 0, &g, &mut state, &mut span_edges, &mut back_edges);
        // Turn it into a tree structure with parent and shit (very often use!!)
        // Get a list of explored vertex and compress them.
        let mut unique_val: Set<US> = Set::new();
        for &(u, v) in span_edges.iter() {
            unique_val.insert(u);
            unique_val.insert(v);
        }
        for &(u, v) in back_edges.iter() {
            unique_val.insert(u);
            unique_val.insert(v);
        }
        let unique_val_arr: Vec<US> = unique_val.iter().cloned().collect();
        let nn = unique_val_arr.len();
        if nn == 0 {
            // No value, no edge, nothing to do!
            continue;
        }
        let span_edges: Vec<UU> = span_edges
            .iter()
            .map(|&(u, v)| {
                (
                    lower_bound_pos(&unique_val_arr, u),
                    lower_bound_pos(&unique_val_arr, v),
                )
            })
            .collect();
        let back_edges: Vec<UU> = back_edges
            .iter()
            .map(|&(u, v)| {
                (
                    lower_bound_pos(&unique_val_arr, u),
                    lower_bound_pos(&unique_val_arr, v),
                )
            })
            .collect();

        let st = lower_bound_pos(&unique_val_arr, st);
        let gp = UndirectedGraph::from_unweighted_edges(nn, &span_edges);
        let mut parent = vec![0; nn];
        let mut children: VV<US> = vec![Vec::new(); nn];
        let mut level = vec![0; nn];
        let mut time_in = vec![0; nn];
        let mut time_out = vec![0; nn];
        let mut global_time = 1;
        let lg = (nn as f32).log2() as usize;
        let mut up: VV<US> = vec![vec![0; lg + 1]; nn];
        root_tree::dfs_root(
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
            &mut up,
            lg,
        );
        // Fix the back edge list, since they also contain duplicates.
        let mut back_edges: Vec<UU> = back_edges
            .iter()
            .cloned()
            .filter(|&(u, v)| level[u] < level[v])
            .collect();
        println!("span edges: {span_edges:?}");
        // Back edges are sorted by level of u,
        // So when exploring other chain with lower level start, we can be sure that the previous chain is already explored.
        back_edges.sort_by_key(|(u, _)| level[*u]);
        println!("back edges: {back_edges:?}");
        // Group back edges by start and end (very useful, but not used in this func)
        let mut be_start: VV<US> = vec![Vec::new(); nn];
        let mut be_end: VV<US> = vec![Vec::new(); nn];
        for &(u, v) in back_edges.iter() {
            be_start[u].push(v);
            be_end[v].push(u);
        }

        // Turn gp into a V<Set>, useful for deleting
        let mut gp: Vec<Set<US>> = vec![Set::new(); nn];
        for &(u, v) in span_edges.iter() {
            gp[u].insert(v);
            gp[v].insert(u);
        }
        // iterate from bottom to top, considering #children and #be_end[u] (useful as hell!)
        let mut cd: Vec<US> = (0..nn).collect();
        cd.sort_by_key(|&x| level[x]);
        cd.reverse();
        for &u in cd.iter() {
            // TODO: Process every u here
        }

        // ============================= End of all useful stuff =============================

        // Start finding bridges via impossible span edges:
        let mut impossible_span_edge: Set<UU> = Set::new();
        for &(u, v) in back_edges.iter() {
            // For each back-edge (x,y), goes from y upward, each generate and edge (x',y').
            // obviously (x',y') can't be a bridge. Mark it as unable.
            let mut cur = v;
            while parent[cur] != u {
                let p = parent[cur];
                if impossible_span_edge.contains(&(p, cur)) {
                    // This chain is already explored before, no need to go further!
                    break;
                }
                impossible_span_edge.insert((p, cur));
                cur = p;
            }
            // last edge: cur -> u
            if parent[cur] != cur {
                impossible_span_edge.insert((parent[cur], cur));
            }
        }
        println!("impossible span edge: {impossible_span_edge:?}");
        for &(u, v) in span_edges.iter() {
            if !impossible_span_edge.contains(&(u, v)) {
                println!(
                    "{}, {} is a possible bridge",
                    unique_val_arr[u], unique_val_arr[v]
                );
            }
        }
    }
}

// dfs tree for directed graph
// can split into 3 types: span-edges, back-edges and cross-edges.
// Cross edges always direct from vertex that was explore later
// to vertex that was explore earlier.
// TODO: Turn this into unconnected form like previously.
#[allow(dead_code)]
fn find_dfs_tree_directed() {
    let n = 10;
    let uelist: Vec<(US, US)> = vec![(0, 1), (1, 2)];
    let g = DirectedGraph::from_unweighted_edges(n, &uelist);
    let mut state = vec![CheckState::NOTCHECK; n];
    // Assume connected graph, find dfs tree at 0
    let mut span_edges: Vec<UU> = Vec::new();
    let mut back_edges: Vec<UU> = Vec::new();
    dfs_tree(0, 0, &g, &mut state, &mut span_edges, &mut back_edges);
    // Turn it into a tree structure with parent and shit (very often use!!)
    let gp = DirectedGraph::from_unweighted_edges(n, &span_edges);
    let mut parent = vec![0; n];
    let mut children: VV<US> = vec![Vec::new(); n];
    let mut level = vec![0; n];
    let mut time_in = vec![0; n];
    let mut time_out = vec![0; n];
    let mut global_time = 0;
    let lg = (n as f32).log2() as usize;
    let mut up: VV<US> = vec![vec![0; lg + 1]; n];
    root_tree::dfs_root(
        0,
        0,
        0,
        &gp,
        &mut parent,
        &mut children,
        &mut level,
        &mut time_in,
        &mut time_out,
        &mut global_time,
        &mut up,
        lg,
    );
    // Start dividing back-edges into cross-edges
    let mut cross_edges: Vec<UU> = back_edges
        .iter()
        .cloned()
        .filter(|&(u, v)| {
            !(root_tree::is_parent(u, v, &time_in, &time_out)
                || root_tree::is_parent(v, u, &time_in, &time_out))
        })
        .collect();
    // True back edge: In the same subtree
    let mut back_edges: Vec<UU> = back_edges
        .iter()
        .cloned()
        .filter(|&(u, v)| {
            root_tree::is_parent(u, v, &time_in, &time_out)
                || root_tree::is_parent(v, u, &time_in, &time_out)
        })
        .collect();
}
