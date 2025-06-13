/// Deal with [a + k*d] repeating points, ranges and other modulo collision structure.
// =============================== Repeating ranges ==================================
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound::*;

use crate::utils;

/// Given a repeating range of [x..x+d] % k, find if a point s is in this range.
/// Often you are given a lot of repeating ranges [x1..x1+d]... but a constant d and k.
/// r = s % k => r-d <= x <= r.
/// Given r, if any x satisfy this eq, then it's good. You can find this via range sum.
pub fn check_in_repeating_ranges(all_x: &Vec<usize>, d: usize, k: usize, s: usize) -> bool {
    // Preprocess all x and put into an array of k
    let mut p = vec![0; k];
    for &x in all_x.iter() {
        p[x % k] += 1;
    }
    let psum: Vec<usize> = p
        .iter()
        .scan(0, |ssum, &px| {
            *ssum += px;
            Some(*ssum)
        })
        .collect();

    // Check if a point s is in any repeating range.
    let r = s % k;
    let range_sum = psum[r] - if r > d { psum[r - d - 1] } else { 0 };
    range_sum > 0
}

// =============================== Periodic points ==================================

/// Given some points on the number line with a periodic property: p[i] change state for a moment at every t === d[i] mod k point in time.
/// Find the closest collision: If we're at time t, point x, and moving in the positive direction at a rate 1,
/// find the next point p[i] so that t + (p[i] - x) === d[i] mod k.
/// We will assume p[i] is sorted inc.
pub fn positive_move_collision(
    p: &Vec<u64>,
    k: u64,
    x: u64,
    t: u64,
    positive_groups: &BTreeMap<u64, Vec<usize>>,
) -> Option<usize> {
    // Group every points by (d[i] - p[i]) % k.
    // let mut positive_groups = BTreeMap::<u64, Vec<usize>>::new();
    // for i in 0..p.len() {
    //     let g = (k + (d[i] % k) - (p[i] % k)) % k;
    //     positive_groups
    //         .entry(g)
    //         .and_modify(|v| (*v).push(i))
    //         .or_insert(vec![i]);
    // }
    // Find the closest p[i] to the strict right of x, ans has to be in range [next_i..].
    let next_i = utils::upper_bound_pos(&p, x);
    if next_i == p.len() {
        // There is no next point.
        return None;
    }
    // (x,t) will collide with all points in group = (t - x) % k
    let xt_positive_group = (k + (t % k) - (x % k)) % k;
    if let Some(v) = positive_groups.get(&xt_positive_group) {
        let v_pos = utils::lower_bound_pos(v, next_i);
        if v_pos < v.len() {
            return Some(v[v_pos]);
        }
    }
    None
}

/// Same as above but with negative direction movement.
pub fn negative_move_collision(
    p: &Vec<u64>,
    k: u64,
    x: u64,
    t: u64,
    negative_groups: &BTreeMap<u64, Vec<usize>>,
) -> Option<usize> {
    // Group every points by (d[i] + p[i]) % k.
    // let mut negative_groups = BTreeMap::<u64, Vec<usize>>::new();
    // for i in 0..p.len() {
    //     let g = ((d[i] % k) + (p[i] % k)) % k;
    //     negative_groups
    //         .entry(g)
    //         .and_modify(|v| (*v).push(i))
    //         .or_insert(vec![i]);
    // }
    // Find the closest p[i] to the strict left of x, ans has to be in range [..=prev_i].
    let prev_i = utils::lower_bound_pos(&p, x);
    if prev_i == 0 {
        // There is no previous point.
        return None;
    }
    let prev_i = prev_i - 1;
    // (x,t) will collide with all points in group = (t + x) % k
    let xt_negative_group = ((t % k) + (x % k)) % k;
    if let Some(v) = negative_groups.get(&xt_negative_group) {
        let v_pos = utils::lower_bound_pos(v, prev_i);
        if v_pos < v.len() {
            if v[v_pos] == prev_i {
                return Some(prev_i);
            }
            // v[v_pos] > prev_i, need back 1 space
            if v_pos > 0 {
                return Some(v[v_pos - 1]);
            }
        }
    }
    None
}
