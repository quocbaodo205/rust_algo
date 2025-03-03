use std::collections::{BTreeSet, VecDeque};

type V<T> = Vec<T>;
type VV<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;
// type Map<K, V> = BTreeMap<K, V>;
type US = usize;
type UU = (US, US);
type UUU = (US, US, US);

// A vector show the shortest path in ring from u -> v
// Return the list of position that it pass through
#[allow(dead_code)]
pub fn shortest_in_ring(
    u: usize,
    v: usize,
    ring_size: usize,
    pos_in_ring: &Vec<usize>,
) -> (Vec<usize>, usize) {
    let i = pos_in_ring[u];
    let j = pos_in_ring[v];
    // Respect direction
    if i > j {
        // 4 -> 1 for example
        // 2 choices: 4 -> 5 -> ... -> [0] -> 1
        // Or to go backward 4 -> 3 -> 2 -> 1
        if i - j == (ring_size - (i - j)) {
            // Either way is fine, check if is last
            if i == ring_size - 1 {
                return (vec![i, ring_size - 1, 0, j], 1);
            }
            return (vec![i, j], 0);
        }
        if i - j < (ring_size - (i - j)) {
            // Going backward is faster
            return (vec![i, j], 0);
        }
        return (vec![i, ring_size - 1, 0, j], 1);
    } else {
        // 1 -> 4 for example
        // go 1 -> 2 -> 3 -> 4...
        // for backward 1 -> 0 -> ... -> 4
        if j - i == (ring_size - (j - i)) {
            // Check if first, then have to go forward
            if i == 0 {
                return (vec![i, j], 1);
            }
            return (vec![i, 0, ring_size - 1, j], 0);
        }
        if j - i < (ring_size - (j - i)) {
            // Going forward is faster
            return (vec![i, j], 1);
        }
        return (vec![i, 0, ring_size - 1, j], 0);
    }
}

#[allow(dead_code)]
fn find_ring(g: &VV<US>) {
    let n = 10;
    let mut g_ring: V<Set<US>> = g
        .iter()
        .map(|v| Set::from_iter(v.iter().cloned()))
        .collect();
    let mut q: VecDeque<US> = VecDeque::new();
    for u in 0..n {
        if g_ring[u].len() == 1 {
            q.push_back(u);
        }
    }
    while let Some(u) = q.pop_front() {
        let ch: V<US> = g_ring[u].iter().cloned().collect();
        for &v in ch.iter() {
            g_ring[v].remove(&u);
            if g_ring[v].len() == 1 {
                q.push_back(v);
            }
        }
        g_ring[u].clear();
    }
    let mut st = 0;
    for u in 0..n {
        if !g_ring[u].is_empty() {
            st = u;
            break;
        }
    }
    let mut cur = st;
    let mut nx = *g_ring[st].first().unwrap();
    let mut ring: V<US> = vec![st];
    while nx != st {
        ring.push(nx);
        for &v in g_ring[nx].iter() {
            if v != cur {
                cur = nx;
                nx = v;
                break;
            }
        }
    }
    let mut pos_in_ring = vec![n; n];
    for (i, &u) in ring.iter().enumerate() {
        pos_in_ring[u] = i;
    }

    println!("Ring vertex: {ring:?}");
}
