// 1d (number line) and 2d coord algo
// For 1d, dealing with segments
// For 2d, add line sweep segment tree

use crate::fenw::FenwickTreeSingleAdd;
use std::{
    cmp::{max, min, Ordering},
    collections::{BTreeMap, BTreeSet},
    ops::{self, Bound::*, DerefMut, RangeBounds},
};

type V<T> = Vec<T>;
type VV<T> = V<V<T>>;
type Set<T> = BTreeSet<T>;
type Map<K, V> = BTreeMap<K, V>;
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

#[allow(dead_code)]
fn neighbors<T>(tree: &BTreeSet<T>, val: T) -> (Option<&T>, Option<&T>)
where
    T: Ord + Copy,
{
    let mut before = tree.range((Unbounded, Excluded(val)));
    let mut after = tree.range((Excluded(val), Unbounded));

    (before.next_back(), after.next())
}
// ============================ 1d ================================== //

#[allow(dead_code)]
fn compress<T>(a: &V<T>) -> (V<US>, V<T>)
where
    T: Ord + PartialOrd + Clone + Copy,
{
    let unique_val: Set<T> = a.iter().cloned().collect();
    let unique_val_arr: V<T> = unique_val.iter().cloned().collect();
    (
        a.iter()
            .map(|x| lower_bound_pos(&unique_val_arr, *x) + 1)
            .collect(),
        unique_val_arr,
    )
}

#[allow(dead_code)]
fn compress_segments<T>(a: &V<(T, T)>) -> (V<(US, US)>, V<T>)
where
    T: Ord + PartialOrd + Clone + Copy,
{
    let mut unique_val: Set<T> = a.iter().map(|&p| p.0).collect();
    unique_val.extend(a.iter().map(|&p| p.1));
    let unique_val_arr: V<T> = unique_val.iter().cloned().collect();
    (
        a.iter()
            .map(|&p| {
                (
                    (lower_bound_pos(&unique_val_arr, p.0) + 1),
                    (lower_bound_pos(&unique_val_arr, p.1) + 1),
                )
            })
            .collect(),
        unique_val_arr,
    )
}

#[allow(dead_code)]
fn is_valid_segment<T>(p: (T, T)) -> bool
where
    T: Ord + PartialOrd,
{
    return p.0 <= p.1;
}

#[allow(dead_code)]
fn overlap_segment<T>(a: (T, T), b: (T, T)) -> Option<(T, T)>
where
    T: Ord + PartialOrd,
{
    if a.0 > b.1 || b.0 > a.1 {
        return None;
    }
    Some((max(a.0, b.0), min(a.1, b.1)))
}

// https://www.geeksforgeeks.org/scheduling-in-greedy-algorithms/
// Select the next possible interval with smallest end
// Return the min number of even to delete to make schedule works.
#[allow(dead_code)]
fn scheduling() -> usize {
    let v = [];
    let mut rl: Vec<(i32, i32)> = v.iter().map(|&(l, r)| (r, l)).collect();
    rl.sort_unstable();

    let mut end = -1000000000;
    let mut count = 0;

    rl.iter().for_each(|&(r, l)| {
        // Segment is currently [l, r), modify should needed...
        if l >= end {
            end = r;
        } else {
            count += 1;
        }
    });

    count
}

#[allow(dead_code)]
fn pointsweep() {
    let n = 10;
    let v = [];
    // println!("Case {_te}, v = {v:?}");

    // Find all left bound of i.
    let mut left_bound = vec![0; n];
    // By r ->, if r eq then by l ->
    let mut rli: V<(US, US, US)> = v.iter().enumerate().map(|(i, &(l, r))| (r, l, i)).collect();
    rli.sort();
    let mut all_left: Set<UU> = v.iter().enumerate().map(|(i, &(l, _))| (l, i)).collect();

    // println!("rli = {rli:?}");

    let mut prev_r = 0;
    rli.iter().enumerate().for_each(|(c, &(r, l, i))| {
        // Remove all previously prev_r < r
        if prev_r < r {
            let mut cc = c;
            while cc >= 1 && rli[cc - 1].0 == prev_r {
                all_left.remove(&(rli[cc - 1].1, rli[cc - 1].2));
                cc -= 1;
            }
        }
        prev_r = r;

        // TODO: Process here, current code is finding the maximum l' <= l that r' >= r
        // Find maximum <= l (after remove the current)
        let (bf, af) = neighbors(&all_left, (l, i));
        // println!("i = {i}, l = {l}, bf = {bf:?}, af = {af:?}");
        // Check if there's some full equal
        if let Some(x) = af {
            if x.0 == l {
                left_bound[i] = l;
                return;
            }
        }
        if bf.is_none() {
            left_bound[i] = l;
        } else {
            left_bound[i] = bf.unwrap().0;
        }
    });
}

// =============================== 2d ======================== //

#[allow(dead_code)]
fn compress_pair<T>(a: &V<(T, T)>) -> (V<UU>, V<T>, V<T>)
where
    T: Ord + PartialOrd + Clone + Copy,
{
    let mut final_arr = vec![(0, 0); a.len()];
    let unique_x: Set<T> = a.iter().map(|&p| p.0).collect();
    let unique_x_v: V<T> = unique_x.iter().cloned().collect();
    (0..a.len()).for_each(|i| {
        final_arr[i].0 = lower_bound_pos(&unique_x_v, a[i].0) + 1;
    });
    let unique_y: Set<T> = a.iter().map(|&p| p.1).collect();
    let unique_y_v: V<T> = unique_y.iter().cloned().collect();
    (0..a.len()).for_each(|i| {
        final_arr[i].1 = lower_bound_pos(&unique_y_v, a[i].1) + 1;
    });
    (final_arr, unique_x_v, unique_y_v)
}

// Template for line sweep
#[allow(dead_code)]
pub fn linesweep() {
    // Turn all point into ayx to sort by y first.
    let a = vec![(1, 1)];
    let (a, unique_x_v, unique_y_v) = compress_pair(&a);
    let mut ayx: V<UU> = a.iter().map(|&p| (p.1 as US, p.0 as US)).collect();
    ayx.sort();
    // Line sweep fw tree
    let mut up = FenwickTreeSingleAdd::new(100);
    let mut down = FenwickTreeSingleAdd::new(100);

    ayx.iter().for_each(|&(_, x)| {
        // All everything to up first
        up.add(x, 1);
    });
    // Add the max point to process the last time.
    ayx.push((unique_y_v.len() + 1, unique_x_v.len() + 1));
    let mut prev_y = 0;
    let mut ans = 0;
    let mut ans_point = (1, 1);
    ayx.iter().enumerate().for_each(|(i, &(y, x))| {
        if y > prev_y {
            // println!("line sweep at y = {prev_y}");

            // TODO: Process here

            // Move all to down
            let mut c = i;
            while c >= 1 && ayx[c - 1].0 == prev_y {
                up.add(ayx[c - 1].1, -1);
                down.add(ayx[c - 1].1, 1);
                // println!("Moving {} {} to down", ayx[c - 1].1, ayx[c - 1].0);
                c -= 1;
            }
        }
        prev_y = y;
    });
}
