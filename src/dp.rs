// ================ D&C optimization ==================
// D&C DP take this form:
// dp[i][j] = min{ k <= j | dp[i-1][k-1] + cost(k,j) }
// Column j depend on all val from previous column and its relation ship with j.
// Let opt(i,j) is the best k that minimize the dp[i][j].
// Only apply when opt(i,j') <= opt(i,j) for all i and j.
// Meaning for a fixed row i, opt(i,j) follow monotonicity.

use std::cmp::min;

// Template for cost function for range [i..=j]
pub fn cost_fn(i: usize, j: usize) -> usize {
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
    let mut best = (usize::MAX, 0);
    for k in optl..=min(mid, optr) {
        let c = if k > 0 { dp_before[k - 1] } else { 0 } + cost_fn(k, mid);
        if c < best.0 {
            best.0 = c;
            best.1 = k;
        }
    }
    dp_cur[mid] = best.0;
    // D&C for other range
    let opt = best.1;
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
    // println!("dp[0] = {dp_before:?}");
    for _i in 1..m {
        let mut dp_cur = vec![0; n];
        dc_compute(&mut dp_cur, &dp_before, 0, n - 1, 0, n - 1);
        dp_before = dp_cur;
        // println!("dp[{_i}] = {dp_before:?}");
    }
    // println!("{}", dp_before[n - 1]);
}

macro_rules! isOn {
    ($S:expr, $b:expr) => {
        ($S & (1 << $b)) > 0
    };
}

macro_rules! turnOn {
    ($S:ident, $b:expr) => {
        $S |= (1 << $b)
    };
    ($S:expr, $b:expr) => {
        $S | (1 << $b)
    };
}

// https://codeforces.com/blog/entry/90841
// Currently solve the fill domino 2x1 in grid n * m problem.
fn plug_dp() {
    let n = 10;
    let m = 20;
    let max_mask = 1usize << (m + 1);
    // dp[j][state]: for each row and j is -> plug
    let mut dp_before: Vec<Vec<usize>> = vec![vec![0; max_mask << 1]; m + 1];
    dp_before[m][0] = 1; // Base case all 0 row as processed and off.
    for _i in 0..n {
        let mut dp_cur: Vec<Vec<usize>> = vec![vec![0; max_mask << 1]; m + 1];

        // Transfer the -> plug from the end of previous to the start of current.
        for mask in 0..max_mask {
            dp_cur[0][mask << 1] = dp_before[m][mask];
        }

        for j in 0..m {
            // j is -> plug, j + 1 is down plug
            for mask in 0..max_mask {
                let x = dp_cur[j][mask];
                let is_right_on = isOn!(mask, j);
                let is_down_on = isOn!(mask, j + 1);
                if !is_right_on && !is_down_on {
                    // Need to add a domino at i,j
                    // Horizontal domino:
                    // - at j now a down plug, and it's not down (same)
                    // - at j+1 is now a -> plug, and it has some (flip to 1)
                    dp_cur[j + 1][mask ^ (1 << (j + 1))] += x;
                    // Vertical domino:
                    // - at j is now a down plug, and it has some (flip to 1)
                    // - at j+1 is now a -> plug, and it has no -> (same)
                    dp_cur[j + 1][mask ^ (1 << j)] += x;
                } else if is_right_on && !is_down_on {
                    // Already have a domino -> over from j to j+1,
                    // - at j now a down plug, and it's not down (flip)
                    // - at j+1 is now a -> plug, and it has none (same)
                    dp_cur[j + 1][mask ^ (1 << j)] += x;
                } else if !is_right_on && is_down_on {
                    // Already have a domino down from (i,j+1) to (i+1,j+1),
                    // - at j now a down plug, and it's not down (same)
                    // - at j+1 is now a -> plug, and it has none (flip)
                    dp_cur[j + 1][mask ^ (1 << (j + 1))] += x;
                }
            }
        }

        dp_before = dp_cur;
    }

    // Last row -> at m, the entire grid is filled so the plug mask should be 0.
    let ans = dp_before[m][0];
}
