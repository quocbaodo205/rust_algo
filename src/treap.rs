// Support range ops like Segment tree like but with Insert / Erase array!
// Super useful when key of the segtree is big (but only small #key)
// Also have the ability of an ordered stat tree.
// If no need for any of these, you can just use fw / sg they are faster.

use std::{
    cmp::{max, min},
    ops::{Add, AddAssign},
};

pub type SmallRng = Xoshiro256PlusPlus;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    /// Construct a new RNG from a 64-bit seed.
    pub fn new(mut state: u64) -> Self {
        const PHI: u64 = 0x9e3779b97f4a7c15;
        let mut seed = <[u64; 4]>::default();
        for chunk in &mut seed {
            state = state.wrapping_add(PHI);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z = z ^ (z >> 31);
            *chunk = z;
        }
        Self { s: seed }
    }

    /// Generate a random `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generate a random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result_plusplus = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;

        self.s[3] = self.s[3].rotate_left(45);

        result_plusplus
    }
}

// ==================== Key-value based treap and range ops =================== //
// https://caterpillow.github.io/byot

// Lazy operation that allow range add or range set
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TreapLazy {
    val: i64,
    // Is increase, if false then it's a set operation.
    // Remember to find all is_inc and change to false should needed.
    is_inc: bool,
}

impl AddAssign for TreapLazy {
    fn add_assign(&mut self, rhs: Self) {
        if !rhs.is_inc {
            self.val = 0;
            self.is_inc = false;
        }
        self.val += rhs.val;
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TreapValue {
    sum: i64,
    mx: i64,
}

#[allow(dead_code)]
impl TreapValue {
    // Update from lazy operation
    pub fn update(&mut self, lz: TreapLazy, sz: usize) {
        if !lz.is_inc {
            self.sum = 0;
            self.mx = 0;
        }
        self.sum += lz.val * (sz as i64);
        self.mx += lz.val;
    }
}

impl Add for TreapValue {
    type Output = TreapValue;

    fn add(self, rhs: Self) -> Self::Output {
        TreapValue {
            sum: self.sum + rhs.sum,
            mx: max(self.mx, rhs.mx),
        }
    }
}

const LID: TreapLazy = TreapLazy {
    val: 0,
    is_inc: true,
};
const VID: TreapValue = TreapValue {
    sum: 0,
    mx: -1000000000000000000,
};

// =================== Main structure ====================== //

type TreapLink = Option<Box<TreapNode>>;

#[allow(dead_code)]
struct TreapNode {
    val: TreapValue, // Value of this single node
    agg: TreapValue, // Value of this node + children
    lz: TreapLazy,

    key: i64,
    priority: u32,
    size: usize,
    left: TreapLink,
    right: TreapLink,
}

#[allow(dead_code)]
impl TreapNode {
    pub fn new(key: i64, value: TreapValue, agg: TreapValue, rng: &mut SmallRng) -> Self {
        let p = rng.next_u32();
        TreapNode {
            val: value,
            agg,
            lz: LID,

            key,
            priority: p,
            size: 1,
            left: None,
            right: None,
        }
    }
}

#[allow(dead_code)]
struct Treap {
    head: TreapLink,
}

#[allow(dead_code)]
impl Treap {
    pub fn new(key: i64, val: TreapValue, rng: &mut SmallRng) -> Self {
        Treap {
            head: Some(Box::new(TreapNode::new(key, val, val, rng))),
        }
    }

    // ================================ Functions with links ====================== //

    fn sz_link(link: Option<&Box<TreapNode>>) -> usize {
        if let Some(node) = link {
            node.size
        } else {
            0
        }
    }

    fn agg_link(link: Option<&Box<TreapNode>>) -> TreapValue {
        if let Some(node) = link {
            node.agg
        } else {
            VID
        }
    }

    fn prio(link: Option<&Box<TreapNode>>) -> u32 {
        if let Some(node) = link {
            node.priority
        } else {
            0
        }
    }

    fn key(link: Option<&Box<TreapNode>>) -> i64 {
        if let Some(node) = link {
            node.key
        } else {
            0
        }
    }

    fn push(link: TreapLink) -> TreapLink {
        match link {
            Some(mut node) => {
                // Update via lz
                node.val.update(node.lz, 1);
                node.agg.update(node.lz, node.size);
                // Push lz to child
                if let Some(mut lnode) = node.left.take() {
                    lnode.lz += node.lz;
                    node.left = Some(lnode);
                }
                if let Some(mut rnode) = node.right.take() {
                    rnode.lz += node.lz;
                    node.right = Some(rnode);
                }
                // Reset lazy
                node.lz = LID;
                Some(node)
            }
            None => None,
        }
    }

    fn pull(link: TreapLink) -> TreapLink {
        if let Some(mut node) = link {
            // Push to left and right first
            node.left = Treap::push(node.left.take());
            node.right = Treap::push(node.right.take());

            node.size =
                Treap::sz_link(node.left.as_ref()) + 1 + Treap::sz_link(node.right.as_ref());

            // Aggregate children
            node.agg = Treap::agg_link(node.left.as_ref())
                + node.val
                + Treap::agg_link(node.right.as_ref());
            return Some(node);
        }
        None
    }

    fn merge_link(l: TreapLink, r: TreapLink) -> TreapLink {
        if l.is_none() || r.is_none() {
            if l.is_none() {
                return r;
            } else {
                return l;
            }
        }

        let lnew = Treap::push(l);
        let rnew = Treap::push(r);

        let (lprio, rprio) = (Treap::prio(lnew.as_ref()), Treap::prio(rnew.as_ref()));
        if lprio > rprio {
            if let Some(mut lnode) = lnew {
                let new_right = Treap::merge_link(lnode.right.take(), rnew);
                lnode.right = new_right;
                return Treap::pull(Some(lnode));
            }
        } else {
            if let Some(mut rnode) = rnew {
                let new_left = Treap::merge_link(lnew, rnode.left.take());
                rnode.left = new_left;
                return Treap::pull(Some(rnode));
            }
        }

        None
    }

    // Split at key = k, right will contain k,
    // (-inf, k) and [k, inf)
    fn split_link(link: TreapLink, k: i64) -> (TreapLink, TreapLink) {
        match link {
            Some(mut node) => {
                if k <= node.key {
                    let (l, r) = Treap::split_link(node.left.take(), k);
                    node.left = r;
                    return (l, Treap::pull(Some(node)));
                } else {
                    let (l, r) = Treap::split_link(node.right.take(), k);
                    node.right = l;
                    return (Treap::pull(Some(node)), r);
                }
            }
            None => (None, None),
        }
    }

    // Find the key by index
    fn findi_link(link: TreapLink, i: usize) -> (TreapLink, Option<i64>) {
        match link {
            Some(node) => {
                let mut new_node = Treap::push(Some(node)).unwrap();
                let k = new_node.key;
                let left_sz = Treap::sz_link(new_node.left.as_ref());
                match i.cmp(&left_sz) {
                    std::cmp::Ordering::Less => {
                        let (nleft, v) = Treap::findi_link(new_node.left.take(), i);
                        new_node.left = nleft;
                        return (Some(new_node), v);
                    }
                    std::cmp::Ordering::Equal => (Some(new_node), Some(k)),
                    std::cmp::Ordering::Greater => {
                        let (nright, v) = Treap::findi_link(new_node.right.take(), i - left_sz - 1);
                        new_node.right = nright;
                        return (Some(new_node), v);
                    }
                }
            }
            None => (None, None),
        }
    }

    // Add value to a link via lazy prop
    fn add_link(link: TreapLink, lz: TreapLazy) -> TreapLink {
        match link {
            Some(mut node) => {
                node.lz += lz;
                Some(node)
            }
            None => None,
        }
    }

    // ================================ Functions with treaps ====================== //

    pub fn build(keys: &[i64], values: &[TreapValue], rng: &mut SmallRng) -> Self {
        let mut head = Some(Box::new(TreapNode::new(keys[0], values[0], values[0], rng)));
        for i in 1..keys.len() {
            head = Treap::merge_link(
                head,
                Some(Box::new(TreapNode::new(keys[i], values[i], values[i], rng))),
            );
        }
        Treap { head }
    }

    pub fn size(&self) -> usize {
        Treap::sz_link(self.head.as_ref())
    }

    // Merge this treap with another treap.
    // When merged, the other treap will not be usable anymore.
    pub fn merge(&mut self, rhs: Treap) {
        let new_head = Treap::merge_link(self.head.take(), rhs.head);
        self.head = new_head;
    }

    // Split into this and the new right treap, right treap will contain k
    pub fn split(&mut self, k: i64) -> Treap {
        let (l, r) = Treap::split_link(self.head.take(), k);
        self.head = l;
        Treap { head: r }
    }

    pub fn single_add(&mut self, k: i64, val: i64) {
        // ..k and k..
        let (l, r) = Treap::split_link(self.head.take(), k);
        // k and k+1..
        let (rs, ri) = Treap::split_link(r, k + 1);

        // Add the lazy val to k and merge all
        self.head = Treap::merge_link(
            l,
            Treap::merge_link(Treap::add_link(rs, TreapLazy { val, is_inc: true }), ri),
        );
    }

    pub fn single_query(&mut self, k: i64) -> TreapValue {
        // ..k and k..
        let (l, r) = Treap::split_link(self.head.take(), k);
        // k and k+1..
        let (rs, ri) = Treap::split_link(r, k + 1);
        let ans = Treap::agg_link(rs.as_ref());

        self.head = Treap::merge_link(l, Treap::merge_link(rs, ri));
        return ans;
    }

    // Add val to range [lo..hi)
    pub fn range_add(&mut self, lo: i64, hi: i64, val: i64) {
        // ..hi and hi..
        let (l, r) = Treap::split_link(self.head.take(), hi);
        // ..lo and lo..hi
        let (ll, mid) = Treap::split_link(l, lo);

        // Add the lazy val to k and merge all
        self.head = Treap::merge_link(
            Treap::merge_link(ll, Treap::add_link(mid, TreapLazy { val, is_inc: true })),
            r,
        );
    }

    pub fn range_query(&mut self, lo: i64, hi: i64) -> TreapValue {
        // ..hi and hi..
        let (l, r) = Treap::split_link(self.head.take(), hi);
        // ..lo and lo..hi
        let (ll, mid) = Treap::split_link(l, lo);

        let ans = Treap::agg_link(mid.as_ref());

        self.head = Treap::merge_link(Treap::merge_link(ll, mid), r);
        return ans;
    }

    // Insert key = k into the tree
    pub fn insert(&mut self, k: i64, val: TreapValue, rng: &mut SmallRng) {
        let rtree = self.split(k);
        self.merge(Treap::new(k, val, rng));
        self.merge(rtree);
    }

    // Delete key = k
    pub fn remove(&mut self, k: i64) {
        // ..k and k..
        let mut rtree = self.split(k);
        // k and k+1..
        let rtree2 = rtree.split(k + 1);
        self.merge(rtree2);
    }

    // Find the key at index i (0 based), useful for order stat set operations.
    // For #key <= x, can use range query.
    pub fn findi(&mut self, i: usize) -> Option<i64> {
        let (nhead, v) = Treap::findi_link(self.head.take(), i);
        self.head = nhead;
        v
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn test_treap_simple() {
        let mut rng = SmallRng::new(42);
        let mut treap = Treap::build(&vec![1, 2, 3, 4], &vec![VID; 4], &mut rng);
        assert_eq!(treap.size(), 4);

        // Split out
        let rtreap = treap.split(2);
        assert_eq!(treap.size(), 1);
        assert_eq!(rtreap.size(), 3);
        // Merge back
        treap.merge(rtreap);
        assert_eq!(treap.size(), 4);

        // Splitting out of range test
        let rtreap = treap.split(5);
        assert_eq!(treap.size(), 4);
        assert_eq!(rtreap.head.is_none(), true);

        // Find key at index 1 (0 based)
        assert_eq!(treap.findi(1), Some(2));
    }

    #[test]
    fn test_treap_range_ops() {
        let mut rng = SmallRng::new(42);
        let ranges: Vec<i64> = (1..=4).collect();
        let mut treap = Treap::build(&ranges, &vec![TreapValue { sum: 0, mx: 0 }; 4], &mut rng);
        assert_eq!(treap.size(), 4);

        // Single add at 3
        treap.single_add(3, -1);
        assert_eq!(treap.size(), 4);
        assert_eq!(treap.single_query(3), TreapValue { sum: -1, mx: -1 });
        assert_eq!(treap.single_query(1), TreapValue { sum: 0, mx: 0 });

        // Range add 2 to [2..4), arr = [0 3 2 0 0]
        treap.range_add(2, 4, 3);
        assert_eq!(treap.range_query(2, 4), TreapValue { sum: 3 + 2, mx: 3 });
        assert_eq!(treap.range_query(2, 5), TreapValue { sum: 3 + 2, mx: 3 });
        assert_eq!(treap.range_query(1, 5), TreapValue { sum: 3 + 2, mx: 3 });
        treap.range_add(4, 5, -2); // [0 3 2 -2 0]
        assert_eq!(treap.range_query(2, 4), TreapValue { sum: 3 + 2, mx: 3 });
        assert_eq!(treap.range_query(2, 5), TreapValue { sum: 3, mx: 3 });
        assert_eq!(treap.range_query(1, 5), TreapValue { sum: 3, mx: 3 });
        treap.range_add(1, 3, 7); // [7 10 2 -2 0]
        assert_eq!(treap.range_query(2, 4), TreapValue { sum: 12, mx: 10 });
        assert_eq!(treap.range_query(3, 5), TreapValue { sum: 0, mx: 2 });
        assert_eq!(treap.range_query(1, 5), TreapValue { sum: 17, mx: 10 });
    }

    #[test]
    fn test_treap_dynamic() {
        // Dynamic array ability
        let mut rng = SmallRng::new(42);
        let mut treap = Treap::build(&vec![1], &vec![TreapValue { sum: 0, mx: 0 }], &mut rng);

        // Usually can use a separated key set to check if key exist
        let mut keys: HashSet<i64> = HashSet::new();
        keys.insert(1);

        // Adding key 7, now is [(1:0), (7:1)]
        keys.insert(7);
        treap.insert(7, TreapValue { sum: 1, mx: 1 }, &mut rng);
        assert_eq!(treap.range_query(2, 4), VID); // Does not exist
        assert_eq!(treap.range_query(1, 4), TreapValue { sum: 0, mx: 0 });
        assert_eq!(treap.range_query(5, 8), TreapValue { sum: 1, mx: 1 });
        assert_eq!(treap.range_query(1, 10), TreapValue { sum: 1, mx: 1 });

        // Adding key 4, now is [(1:0), (4:9), (7:1)]
        keys.insert(4);
        treap.insert(4, TreapValue { sum: 9, mx: 9 }, &mut rng);
        assert_eq!(treap.range_query(2, 4), VID); // Does not exist
        assert_eq!(treap.range_query(1, 5), TreapValue { sum: 9, mx: 9 });
        assert_eq!(treap.range_query(5, 8), TreapValue { sum: 1, mx: 1 });
        assert_eq!(treap.range_query(1, 10), TreapValue { sum: 10, mx: 9 });

        // Find key at indexes (0 based)
        assert_eq!(treap.findi(0), Some(1));
        assert_eq!(treap.findi(1), Some(4));
        assert_eq!(treap.findi(2), Some(7));
        assert_eq!(treap.findi(3), None);

        // Remove key 7, now is [(1:0), (4:9)]
        keys.remove(&7);
        treap.remove(7);
        assert_eq!(treap.range_query(2, 4), VID); // Does not exist
        assert_eq!(treap.range_query(1, 3), TreapValue { sum: 0, mx: 0 });
        assert_eq!(treap.range_query(5, 8), VID);
        assert_eq!(treap.range_query(1, 10), TreapValue { sum: 9, mx: 9 });

        // Find key at indexes (0 based)
        assert_eq!(treap.findi(0), Some(1));
        assert_eq!(treap.findi(1), Some(4));
        assert_eq!(treap.findi(2), None);
        assert_eq!(treap.findi(3), None);
    }
}
