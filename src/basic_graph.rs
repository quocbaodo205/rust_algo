use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    ops,
};

use crate::root_tree;

#[allow(dead_code)]
fn lower_bound_pos<T: Ord + PartialOrd>(a: &Vec<T>, search_value: T) -> usize {
    a.binary_search_by(|e| match e.cmp(&search_value) {
        Ordering::Equal => Ordering::Greater,
        ord => ord,
    })
    .unwrap_err()
}

type V<T> = Vec<T>;
type VV<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;
type Map<K, V> = BTreeMap<K, V>;
type US = usize;
type UU = (US, US);

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckState {
    NOTCHECK,
    CHECKING,
    CHECKED,
}

// ============================ basic =====================================

// Visual: https://visualgo.net/en/graphds

// simple DFS
#[allow(dead_code)]
fn dfs_temp(u: US, g: &VV<US>, state: &mut V<CheckState>) {
    state[u] = CheckState::CHECKED;
    g[u].iter().for_each(|&v| {
        if state[v] == CheckState::NOTCHECK {
            dfs_temp(v, g, state);
        }
    });
}

// simple BFS
#[allow(dead_code)]
fn bfs_temp(n: US) {
    let mut g: VV<US> = vec![V::new(); n];
    let mut state = vec![CheckState::NOTCHECK; n];

    // main bfs part
    let mut q: VecDeque<US> = VecDeque::new();

    (0..n).for_each(|u| {
        if state[u] == CheckState::NOTCHECK {
            state[u] = CheckState::CHECKED;
            q.push_back(u);
        }
        while let Some(u) = q.pop_front() {
            g[u].iter().for_each(|&v| {
                if state[v] == CheckState::NOTCHECK {
                    state[v] = CheckState::CHECKED;
                    q.push_back(v);
                }
            });
        }
    });
}

// ============================ shortest path =====================================

// simple BFS
#[allow(dead_code)]
fn bfs_shpath(n: US) {
    let mut g: VV<US> = vec![V::new(); n];
    let mut d = vec![1000000009; n];

    // Init some d[u] = 0 and push to Q
    let mut q: VecDeque<US> = VecDeque::new();
    (0..n).for_each(|u| {
        if d[u] == 0 {
            q.push_back(u);
        }
    });

    while let Some(u) = q.pop_front() {
        g[u].iter().for_each(|&v| {
            if d[v] > d[u] + 1 {
                d[v] = d[u] + 1;
                q.push_back(v);
            }
        });
    }
}

#[allow(dead_code)]
fn dijkstra(n: US) {
    let mut g: VV<UU> = vec![V::new(); n];
    let mut d = vec![1000000009; n];

    // Init some d[u] = 0 and push to Q
    let mut q: Set<UU> = (0..n).map(|u| (d[u], u)).collect();

    while let Some((du, u)) = q.pop_first() {
        g[u].iter().for_each(|&(v, w)| {
            if d[v] > du + w {
                q.remove(&(d[v], v));
                d[v] = du + w;
                q.insert((d[v], v));
            }
        });
    }
}

// ============================ topo sort =====================================

#[allow(dead_code)]
fn dfs_has_cycle(u: US, g: &VV<US>, state: &mut V<CheckState>) -> bool {
    state[u] = CheckState::CHECKING;
    for &v in g[u].iter() {
        let has_cycle_v = match state[v] {
            CheckState::NOTCHECK => dfs_has_cycle(v, g, state),
            CheckState::CHECKING => true,
            CheckState::CHECKED => false,
        };
        if has_cycle_v {
            return true;
        }
    }
    state[u] = CheckState::CHECKED;
    false
}

#[allow(dead_code)]
fn dfs_topo(u: US, g: &VV<US>, state: &mut V<CheckState>, ts: &mut V<US>) {
    state[u] = CheckState::CHECKED;
    g[u].iter().for_each(|&v| {
        if state[v] == CheckState::NOTCHECK {
            dfs_topo(v, g, state, ts);
        }
    });
    ts.push(u);
}

#[allow(dead_code)]
fn find_topo(g: &VV<US>, state: &mut V<CheckState>) -> V<US> {
    let mut ts: V<US> = V::new();
    (0..state.len()).for_each(|u| {
        if state[u] == CheckState::NOTCHECK {
            dfs_topo(u, g, state, &mut ts)
        }
    });
    ts.reverse();
    ts
}

// ============================ dfs tree magic =========================

// Find the list of span edge, these edges form a spanning tree.
// All other edges are called back edges, the back edge always point from u to it's sub tree.
#[allow(dead_code)]
fn dfs_tree(
    u: US,
    p: US,
    g: &VV<US>,
    state: &mut V<CheckState>,
    span_edges: &mut V<UU>,
    back_edges: &mut V<UU>,
) {
    state[u] = CheckState::CHECKED;
    for &v in g[u].iter() {
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
    let mut g: VV<US> = vec![V::new(); n];
    let mut state = vec![CheckState::NOTCHECK; n];
    // Graph could be un-connected.
    for st in 0..n {
        if state[st] == CheckState::CHECKED {
            continue;
        }
        let mut span_edges: V<UU> = V::new();
        let mut back_edges: V<UU> = V::new();
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
        let unique_val_arr: V<US> = unique_val.iter().cloned().collect();
        let nn = unique_val_arr.len();
        if nn == 0 {
            // No value, no edge, nothing to do!
            continue;
        }
        let span_edges: V<UU> = span_edges
            .iter()
            .map(|&(u, v)| {
                (
                    lower_bound_pos(&unique_val_arr, u),
                    lower_bound_pos(&unique_val_arr, v),
                )
            })
            .collect();
        let back_edges: V<UU> = back_edges
            .iter()
            .map(|&(u, v)| {
                (
                    lower_bound_pos(&unique_val_arr, u),
                    lower_bound_pos(&unique_val_arr, v),
                )
            })
            .collect();

        let st = lower_bound_pos(&unique_val_arr, st);
        let mut gp: VV<US> = vec![V::new(); nn];
        for &(u, v) in span_edges.iter() {
            gp[u].push(v);
            gp[v].push(u);
        }
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
        let mut back_edges: V<UU> = back_edges
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
        let mut be_start: VV<US> = vec![V::new(); nn];
        let mut be_end: VV<US> = vec![V::new(); nn];
        for &(u, v) in back_edges.iter() {
            be_start[u].push(v);
            be_end[v].push(u);
        }

        // Turn gp into a V<Set>, useful for deleting
        let mut gp: V<Set<US>> = vec![Set::new(); nn];
        for &(u, v) in span_edges.iter() {
            gp[u].insert(v);
            gp[v].insert(u);
        }
        // iterate from bottom to top, considering #children and #be_end[u] (useful as hell!)
        let mut cd: V<US> = (0..nn).collect();
        cd.sort_by_key(|&x| level[x]);
        cd.reverse();
        for &u in cd.iter() {}

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
#[allow(dead_code)]
fn find_dfs_tree_directed(g: &VV<US>, state: &mut V<CheckState>) {
    let n = g.len();
    // Assume connected graph, find dfs tree at 0
    let mut span_edges: V<UU> = V::new();
    let mut back_edges: V<UU> = V::new();
    dfs_tree(0, 0, g, state, &mut span_edges, &mut back_edges);
    // Turn it into a tree structure with parent and shit (very often use!!)
    let mut gp: VV<US> = vec![V::new(); n];
    for &(u, v) in span_edges.iter() {
        gp[u].push(v);
    }
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
    let mut cross_edges: V<UU> = back_edges
        .iter()
        .cloned()
        .filter(|&(u, v)| {
            !(root_tree::is_parent(u, v, &time_in, &time_out)
                || root_tree::is_parent(v, u, &time_in, &time_out))
        })
        .collect();
    // True back edge: In the same subtree
    let mut back_edges: V<UU> = back_edges
        .iter()
        .cloned()
        .filter(|&(u, v)| {
            root_tree::is_parent(u, v, &time_in, &time_out)
                || root_tree::is_parent(v, u, &time_in, &time_out)
        })
        .collect();
}

// ============================ 2d =====================================

// Template for grid movement
#[derive(Copy, Clone, Debug)]
struct Point(i32, i32);

impl ops::Add for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl ops::Sub for Point {
    type Output = Point;

    fn sub(self, rhs: Point) -> Self::Output {
        Point(self.0 - rhs.0, self.1 - rhs.1)
    }
}

#[allow(dead_code)]
fn is_valid_point(x: Point, n: usize, m: usize) -> bool {
    return x.0 >= 0 && x.0 < n as i32 && x.1 >= 0 && x.1 < m as i32;
}

#[allow(dead_code)]
// Translate a character to a directional Point
fn translate(c: u8) -> Point {
    match c as char {
        'U' => Point(-1, 0),
        'D' => Point(1, 0),
        'L' => Point(0, -1),
        'R' => Point(0, 1),
        _ => Point(0, 0),
    }
}

#[allow(dead_code)]
// Simple DFS from a point, using a map of direction LRUD
fn dfs_grid(s: Point, mp: &V<V<u8>>, checked: &mut V<V<bool>>) {
    let n = mp.len();
    let m = mp[0].len();

    if checked[s.0 as usize][s.1 as usize] {
        return;
    }
    let mut s = Some(s);
    while let Some(u) = s {
        checked[u.0 as usize][u.1 as usize] = true;
        let v = u + translate(mp[u.0 as usize][u.1 as usize]);
        s = match is_valid_point(v, n, m) && !checked[v.0 as usize][v.1 as usize] {
            true => Some(v),
            false => None,
        }
    }
}

#[allow(dead_code)]
// Simple BFS from a point, using a blocker map
fn flood_grid(s: Point, blocked: &V<V<bool>>, checked: &mut V<V<bool>>) {
    let n = blocked.len();
    let m = blocked[0].len();

    if checked[s.0 as usize][s.1 as usize] {
        return;
    }
    let mut q: VecDeque<Point> = VecDeque::new();
    let direction = [Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0)];
    q.push_back(s);
    while let Some(u) = q.pop_front() {
        direction.iter().for_each(|&d| {
            let v = u + d;
            if is_valid_point(v, n, m)
                && !checked[v.0 as usize][v.1 as usize]
                && !blocked[v.0 as usize][v.1 as usize]
            {
                checked[v.0 as usize][v.1 as usize] = true;
                q.push_back(v);
            }
        });
    }
}
