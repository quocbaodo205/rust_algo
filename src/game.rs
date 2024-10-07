use std::{collections::BTreeSet, fmt::Debug};

#[allow(dead_code)]
fn gcd(mut n: usize, mut m: usize) -> usize {
    if n == 0 || m == 0 {
        return n + m;
    }
    while m != 0 {
        if m < n {
            let t = m;
            m = n;
            n = t;
        }
        m = m % n;
    }
    n
}

#[allow(dead_code)]
fn find_mex_with_set(all_sub_g: &BTreeSet<usize>) -> usize {
    for mx in 0..=all_sub_g.len() {
        if !all_sub_g.contains(&mx) {
            return mx;
        }
    }
    return all_sub_g.len();
}

// Calculate grundy number for a state
// https://cp-algorithms.com/game_theory/sprague-grundy-nim.html
// 1: Identify all transition (move)
// 2: Each move can spawn some independent game state (s)
//  - Calculate grundy number for each of these child states (s) and xor them -> (x)
// 3: calculate mex of all these (x) -> (g)
#[allow(dead_code)]
fn grundy_number(max_state: usize) -> Vec<usize> {
    let mut g: Vec<usize> = vec![0; max_state + 1];
    // First case: no stone is an auto lost
    g[0] = 0;
    (1..=max_state).for_each(|state| {
        let mut all_sub_g: BTreeSet<usize> = BTreeSet::new();
        (1..=state).for_each(|mv| {
            if gcd(mv, state) == 1 {
                // Only spawn 1 new independent game
                let new_state = state - mv;
                all_sub_g.insert(g[new_state]);
            }
        });
        g[state] = find_mex_with_set(&all_sub_g);
    });

    g
}

#[allow(dead_code)]
fn print_down<T>(v: &Vec<T>)
where
    T: Debug,
{
    v.iter().enumerate().for_each(|(i, x)| {
        println!("{i:4} {:4?}", *x);
    })
}
