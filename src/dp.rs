// ================ D&C optimization ==================
// D&C DP take this form:
// dp[i][j] = min{ k <= j | dp[i-1][k-1] + cost(k,j) }
// Column j depend on all val from previous column and its relation ship with j.
// Let opt(i,j) is the best k that minimize the dp[i][j].
// Only apply when opt(i,j') <= opt(i,j) for all i and j.
// Meaning for a fixed row i, opt(i,j) follow monotonicity.

use std::cmp::min;

// Template for cost function.
pub fn cost_fn(j: usize, k: usize) -> usize {
    0
}

// Calculate all dp_cur[l..=r] in D&C manner.
pub fn dc_compute(
    dp_cur: &mut Vec<usize>,
    dp_before: &Vec<usize>,
    l: usize,
    r: usize,
    optl: usize,
    optr: usize,
) {
    if l > r {
        return;
    }

    let mid = (l + r) / 2;
    // Find which k is best
    let mut best = (usize::MAX, -1);
    for k in optl..=min(mid, optr) {
        let c = if k > 0 { dp_before[k - 1] } else { 0 } + cost_fn(k, mid);
        if c < best.0 {
            best.0 = c;
            best.1 = k as i32;
        }
    }
    dp_cur[mid] = best.0;
    // D&C for other range
    let opt = best.1 as usize;
    if mid > l {
        dc_compute(dp_cur, dp_before, l, mid - 1, optl, opt);
    }
    if mid + 1 <= r {
        dc_compute(dp_cur, dp_before, mid + 1, r, opt, optr);
    }
}

pub fn solve_dc() {
    let n = 10;
    let m = 10;

    let mut dp_before = vec![0; n];
    for j in 0..n {
        dp_before[j] = cost_fn(0, j);
    }
    for _ in 1..m {
        let mut dp_cur = vec![0; n];
        dc_compute(&mut dp_cur, &dp_before, 0, n - 1, 0, n - 1);
        dp_before = dp_cur;
    }

    // Have the whole dp_before to use afterward.
}
