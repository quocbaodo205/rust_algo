use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ops,
};

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

// ============================ 2d =====================================

// Template for grid movement
#[derive(Copy, Clone, Debug)]
struct Point((i32, i32));

impl ops::Add for Point {
    type Output = Point;

    fn add(self, rhs: Point) -> Self::Output {
        Point((self.0 .0 + rhs.0 .0, self.0 .1 + rhs.0 .1))
    }
}

impl ops::Sub for Point {
    type Output = Point;

    fn sub(self, rhs: Point) -> Self::Output {
        Point((self.0 .0 - rhs.0 .0, self.0 .1 - rhs.0 .1))
    }
}

#[allow(dead_code)]
fn is_valid_point(x: Point, n: usize, m: usize) -> bool {
    return x.0 .0 >= 0 && x.0 .0 < n as i32 && x.0 .1 >= 0 && x.0 .1 < m as i32;
}

#[allow(dead_code)]
// Translate a character to a directional Point
fn translate(c: u8) -> Point {
    match c as char {
        'U' => Point((-1, 0)),
        'D' => Point((1, 0)),
        'L' => Point((0, -1)),
        'R' => Point((0, 1)),
        _ => Point((0, 0)),
    }
}

#[allow(dead_code)]
// Simple DFS from a point, using a map of direction LRUD
fn dfs_grid(s: Point, mp: &V<V<u8>>, checked: &mut V<V<bool>>) {
    let n = mp.len();
    let m = mp[0].len();

    if checked[s.0 .0 as usize][s.0 .1 as usize] {
        return;
    }
    let mut s = Some(s);
    while let Some(u) = s {
        checked[u.0 .0 as usize][u.0 .1 as usize] = true;
        let v = u + translate(mp[u.0 .0 as usize][u.0 .1 as usize]);
        s = match is_valid_point(v, n, m) && !checked[v.0 .0 as usize][v.0 .1 as usize] {
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

    if checked[s.0 .0 as usize][s.0 .1 as usize] {
        return;
    }
    let mut q: VecDeque<Point> = VecDeque::new();
    let direction = "LRUD".as_bytes();
    q.push_back(s);
    while let Some(u) = q.pop_front() {
        direction.iter().for_each(|&d| {
            let v = u + translate(d);
            if is_valid_point(v, n, m)
                && !checked[v.0 .0 as usize][v.0 .1 as usize]
                && !blocked[v.0 .0 as usize][v.0 .1 as usize]
            {
                checked[v.0 .0 as usize][v.0 .1 as usize] = true;
                q.push_back(v);
            }
        });
    }
}
