use std::{
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
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
pub enum CheckState {
    NOTCHECK,
    CHECKING,
    CHECKED,
}

use std::ops::Index;

/// CSR 形式の二次元配列
#[derive(Clone)]
pub struct CSRArray<T> {
    pos: Vec<usize>,
    data: Vec<T>,
}

impl<T: Clone> CSRArray<T> {
    /// a の各要素 (i, x) について、 i 番目の配列に x が格納される
    pub fn new(n: usize, a: &[(usize, T)]) -> Self {
        let mut pos = vec![0; n + 1];
        for &(i, _) in a {
            pos[i] += 1;
        }
        for i in 0..n {
            pos[i + 1] += pos[i];
        }
        let mut ord = vec![0; a.len()];
        for j in (0..a.len()).rev() {
            let (i, _) = a[j];
            pos[i] -= 1;
            ord[pos[i]] = j;
        }
        let data = ord.into_iter().map(|i| a[i].1.clone()).collect();
        Self { pos, data }
    }
}

impl<T> CSRArray<T> {
    pub fn len(&self) -> usize {
        self.pos.len() - 1
    }

    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        (0..self.len()).map(|i| &self[i])
    }
}

impl<T> Index<usize> for CSRArray<T> {
    type Output = [T];
    fn index(&self, i: usize) -> &Self::Output {
        let start = self.pos[i];
        let end = self.pos[i + 1];
        &self.data[start..end]
    }
}

/// 隣接リスト
#[derive(Clone)]
pub struct Graph<V, E, const DIRECTED: bool>
where
    V: Clone,
    E: Clone,
{
    vertices: Vec<V>,
    edges: CSRArray<(usize, E)>,
}

pub type DirectedGraph<V, E> = Graph<V, E, true>;
pub type UndirectedGraph<V, E> = Graph<V, E, false>;

/// グリッドグラフの 4 近傍  
/// 上, 左, 下, 右
pub const GRID_NEIGHBOURS_4: [(usize, usize); 4] = [(!0, 0), (0, !0), (1, 0), (0, 1)];

/// グリッドグラフの 8 近傍  
/// 上, 左, 下, 右, 左上, 左下, 右下, 右上
pub const GRID_NEIGHBOURS_8: [(usize, usize); 8] = [
    (!0, 0),
    (0, !0),
    (1, 0),
    (0, 1),
    (!0, !0),
    (1, !0),
    (1, 1),
    (!0, 1),
];

impl<V, E> DirectedGraph<V, E>
where
    V: Clone,
    E: Clone,
{
    /// 頂点重みと重み付き有向辺からグラフを構築する
    pub fn from_vertices_and_edges(vertices: &[V], edges: &[(usize, usize, E)]) -> Self {
        let edges = edges
            .iter()
            .map(|(u, v, w)| (*u, (*v, w.clone())))
            .collect::<Vec<_>>();

        Self {
            vertices: vertices.to_vec(),
            edges: CSRArray::new(vertices.len(), &edges),
        }
    }

    /// グリッドからグラフを構築する
    ///
    /// # 引数
    ///
    /// * `grid` - グリッド
    /// * `neighbours` - 近傍
    /// * `cost` - grid の値から重みを計算する関数
    pub fn from_grid(
        grid: &Vec<Vec<V>>,
        neighbours: &[(usize, usize)],
        cost: impl Fn(&V, &V) -> Option<E>,
    ) -> Self {
        let h = grid.len();
        let w = grid[0].len();
        let mut edges = vec![];
        for i in 0..h {
            for j in 0..w {
                for &(di, dj) in neighbours {
                    let ni = i.wrapping_add(di);
                    let nj = j.wrapping_add(dj);
                    if ni >= h || nj >= w {
                        continue;
                    }
                    if let Some(c) = cost(&grid[i][j], &grid[ni][nj]) {
                        edges.push((i * w + j, ni * w + nj, c));
                    }
                }
            }
        }
        Self::from_vertices_and_edges(
            &grid.into_iter().flatten().cloned().collect::<Vec<_>>(),
            &edges,
        )
    }
}

impl<V, E> UndirectedGraph<V, E>
where
    V: Clone,
    E: Clone,
{
    /// 頂点重みと重み付き無向辺からグラフを構築する
    pub fn from_vertices_and_edges(vertices: &[V], edges: &[(usize, usize, E)]) -> Self {
        let edges = edges
            .iter()
            .map(|(u, v, w)| [(*u, (*v, w.clone())), (*v, (*u, w.clone()))])
            .flatten()
            .collect::<Vec<_>>();

        Self {
            vertices: vertices.to_vec(),
            edges: CSRArray::new(vertices.len(), &edges),
        }
    }

    /// グリッドからグラフを構築する
    ///
    /// # 引数
    ///
    /// * `grid` - グリッド
    /// * `neighbours` - 近傍
    /// * `cost` - grid の値から重みを計算する関数
    pub fn from_grid(
        grid: &Vec<Vec<V>>,
        neighbours: &[(usize, usize)],
        cost: impl Fn(&V, &V) -> Option<E>,
    ) -> Self {
        let h = grid.len();
        let w = grid[0].len();
        let mut edges = vec![];
        for i in 0..h {
            for j in 0..w {
                for &(di, dj) in neighbours {
                    let ni = i.wrapping_add(di);
                    let nj = j.wrapping_add(dj);
                    if ni >= h || nj >= w {
                        continue;
                    }
                    let u = i * w + j;
                    let v = ni * w + nj;
                    if u > v {
                        continue;
                    }
                    if let Some(c) = cost(&grid[i][j], &grid[ni][nj]) {
                        edges.push((u, v, c));
                    }
                }
            }
        }
        Self::from_vertices_and_edges(
            &grid.into_iter().flatten().cloned().collect::<Vec<_>>(),
            &edges,
        )
    }
}

impl<V, E, const DIRECTED: bool> Graph<V, E, DIRECTED>
where
    V: Clone,
    E: Clone,
{
    /// 頂点数を返す
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// 頂点重みを返す
    pub fn vertex(&self, v: usize) -> &V {
        &self.vertices[v]
    }

    /// 頂点 v から出る辺を返す  
    /// g\[v\] と同じ
    pub fn edges(&self, v: usize) -> &[(usize, E)] {
        &self.edges[v]
    }
}

impl<V, E, const DIRECTED: bool> Index<usize> for Graph<V, E, DIRECTED>
where
    V: Clone,
    E: Clone,
{
    type Output = [(usize, E)];

    fn index(&self, v: usize) -> &[(usize, E)] {
        self.edges(v)
    }
}

impl<V> DirectedGraph<V, ()>
where
    V: Clone,
{
    /// 頂点重みと重みなし有向辺からグラフを構築する
    pub fn from_vertices_and_unweighted_edges(vertices: &[V], edges: &[(usize, usize)]) -> Self {
        Self::from_vertices_and_edges(
            vertices,
            &edges.iter().map(|&(u, v)| (u, v, ())).collect::<Vec<_>>(),
        )
    }
}

impl<V> UndirectedGraph<V, ()>
where
    V: Clone,
{
    /// 頂点重みと重みなし無向辺からグラフを構築する
    pub fn from_vertices_and_unweighted_edges(vertices: &[V], edges: &[(usize, usize)]) -> Self {
        Self::from_vertices_and_edges(
            vertices,
            &edges.iter().map(|&(u, v)| (u, v, ())).collect::<Vec<_>>(),
        )
    }
}

impl<E> DirectedGraph<(), E>
where
    E: Clone,
{
    /// 重み付き有向辺からグラフを構築する
    pub fn from_edges(n: usize, edges: &[(usize, usize, E)]) -> Self {
        Self::from_vertices_and_edges(&vec![(); n], edges)
    }
}

impl<E> UndirectedGraph<(), E>
where
    E: Clone,
{
    /// 重み付き無向辺からグラフを構築する
    pub fn from_edges(n: usize, edges: &[(usize, usize, E)]) -> Self {
        Self::from_vertices_and_edges(&vec![(); n], edges)
    }
}

impl DirectedGraph<(), ()> {
    /// 重みなし有向辺からグラフを構築する
    pub fn from_unweighted_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        Self::from_edges(
            n,
            &edges.iter().map(|&(u, v)| (u, v, ())).collect::<Vec<_>>(),
        )
    }
}

impl UndirectedGraph<(), ()> {
    /// 重みなし無向辺からグラフを構築する
    pub fn from_unweighted_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        Self::from_edges(
            n,
            &edges.iter().map(|&(u, v)| (u, v, ())).collect::<Vec<_>>(),
        )
    }
}

// ============================ shortest path =====================================

pub struct ZeroOneBFSResult<T>
where
    T: Clone + Ord + Add<Output = T> + Default,
{
    pub dist: Vec<T>,
    pub prev: Vec<usize>,
}

/// 0-1 BFS  
/// 辺の重みが 0 か 1 のグラフ上で、始点から各頂点への最短距離を求める
///
/// # 戻り値
///
/// ZeroOneBFSResult
/// - dist: 始点から各頂点への最短距離
/// - prev: 始点から各頂点への最短経路における前の頂点
pub fn zero_one_bfs<V, T, const DIRECTED: bool>(
    g: &Graph<V, T, DIRECTED>,
    starts: &[usize],
    inf: T,
) -> ZeroOneBFSResult<T>
where
    V: Clone,
    T: Clone + Ord + Add<Output = T> + Default + From<u8>,
{
    assert!(starts.len() > 0);
    let zero = T::from(0);
    let one = T::from(1);
    let mut dist = vec![inf.clone(); g.len()];
    let mut prev = vec![!0; g.len()];
    let mut dq = VecDeque::new();
    for &s in starts {
        dist[s] = T::default();
        dq.push_back((zero.clone(), s));
    }
    while let Some((s, v)) = dq.pop_front() {
        if dist[v] < s {
            continue;
        }
        for (u, w) in &g[v] {
            assert!(*w == zero || *w == one);
            let t = dist[v].clone() + w.clone();
            if dist[*u] > t {
                dist[*u] = t.clone();
                prev[*u] = v;
                if *w == zero {
                    dq.push_front((t.clone(), *u));
                } else {
                    dq.push_back((t.clone(), *u));
                }
            }
        }
    }
    ZeroOneBFSResult { dist, prev }
}

impl<T> ZeroOneBFSResult<T>
where
    T: Clone + Ord + Add<Output = T> + Default,
{
    /// 始点から頂点 v への最短経路を求める  
    /// 経路が存在しない場合は None を返す
    pub fn path(&self, mut v: usize) -> Option<Vec<usize>> {
        if self.dist[v].clone() != T::default() && self.prev[v] == !0 {
            return None;
        }
        let mut path = vec![];
        while v != !0 {
            path.push(v);
            v = self.prev[v];
        }
        path.reverse();
        Some(path)
    }
}

use std::{cmp::Reverse, collections::BinaryHeap, ops::Add};

pub struct DijkstraResult<T>
where
    T: Clone + Ord + Add<Output = T> + Default,
{
    pub dist: Vec<T>,
    pub prev: Vec<usize>,
}

/// ダイクストラ法  
/// 始点集合からの最短距離を求める。
pub fn dijkstra<V, T, const DIRECTED: bool>(
    g: &Graph<V, T, DIRECTED>,
    starts: &[usize],
    inf: T,
) -> DijkstraResult<T>
where
    V: Clone,
    T: Clone + Ord + Add<Output = T> + Default,
{
    assert!(starts.len() > 0);
    let mut dist = vec![inf.clone(); g.len()];
    let mut prev = vec![!0; g.len()];
    let mut pq = BinaryHeap::new();
    for &s in starts {
        dist[s] = T::default();
        pq.push(Reverse((T::default(), s)));
    }
    while let Some(Reverse((s, v))) = pq.pop() {
        if dist[v] < s {
            continue;
        }
        for (u, w) in &g[v] {
            assert!(w.clone() >= T::default());
            if dist[*u] > dist[v].clone() + w.clone() {
                dist[*u] = dist[v].clone() + w.clone();
                prev[*u] = v;
                pq.push(Reverse((dist[*u].clone(), *u)));
            }
        }
    }
    DijkstraResult { dist, prev }
}

impl<T> DijkstraResult<T>
where
    T: Clone + Ord + Add<Output = T> + Default,
{
    /// 終点 v までの最短経路を求める。
    /// 終点に到達できない場合は None を返す。
    pub fn path(&self, mut v: usize) -> Option<Vec<usize>> {
        if self.dist[v].clone() != T::default() && self.prev[v] == !0 {
            return None;
        }
        let mut path = vec![];
        while v != !0 {
            path.push(v);
            v = self.prev[v];
        }
        path.reverse();
        Some(path)
    }
}

// ============================ topo sort =====================================

#[allow(dead_code)]
fn dfs_has_cycle<V, T, const DIRECTED: bool>(
    u: US,
    g: &Graph<V, T, DIRECTED>,
    state: &mut Vec<CheckState>,
) -> bool
where
    V: Clone,
    T: Clone + Ord + Add<Output = T> + Default + From<u8>,
{
    state[u] = CheckState::CHECKING;
    for (v, _) in &g[u] {
        let has_cycle_v = match state[*v] {
            CheckState::NOTCHECK => dfs_has_cycle(*v, g, state),
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
fn dfs_topo<V, T, const DIRECTED: bool>(
    u: US,
    g: &Graph<V, T, DIRECTED>,
    state: &mut Vec<CheckState>,
    ts: &mut Vec<US>,
) where
    V: Clone,
    T: Clone + Ord + Add<Output = T> + Default + From<u8>,
{
    state[u] = CheckState::CHECKED;
    for (v, _) in &g[u] {
        if state[*v] == CheckState::NOTCHECK {
            dfs_topo(*v, g, state, ts);
        }
    }
    ts.push(u);
}

#[allow(dead_code)]
fn find_topo<V, T, const DIRECTED: bool>(
    g: &Graph<V, T, DIRECTED>,
    state: &mut Vec<CheckState>,
) -> Vec<US>
where
    V: Clone,
    T: Clone + Ord + Add<Output = T> + Default + From<u8>,
{
    let mut ts: Vec<US> = Vec::new();
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
