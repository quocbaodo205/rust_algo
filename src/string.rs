use std::cmp::{max, min};

// ========================= simple search ====================

// Searching all substring by length first and sliding windows
#[allow(dead_code)]
fn all_sub_str_template(s: &Vec<u8>, f: &dyn Fn(usize) -> usize) {
    // Any substring of length d
    (1..=s.len()).for_each(|d| {
        // Process first substr
        let mut start = 0;
        let mut end = start + d - 1;
        let mut contrib = 0; // Contribution of each position

        // Move up and process each substr
        while end < s.len() - 1 {
            // Minus old contrib
            contrib -= f(start);

            // Plus new contrib
            start += 1;
            end += 1;
            contrib += f(end);
        }
    });
}

// ========================= complex algo =====================

#[allow(dead_code)]
fn manacher_odd(st: &str) -> Vec<usize> {
    let n = st.len();

    let mut s = "$".to_owned();
    s.push_str(st);
    s.push('^');
    let s = s.as_bytes();

    let mut p = vec![0; n + 2];
    let mut l: usize = 1;
    let mut r: usize = 1;

    (1..=n).for_each(|i| {
        p[i] = max(0, min(r - i, p[l + (r - i)]));
        while s[i - p[i]] == s[i + p[i]] {
            p[i] += 1;
        }
        if i + p[i] > r {
            l = i - p[i];
            r = i + p[i];
        }
    });

    p[1..p.len() - 1].to_vec()
}

#[allow(dead_code)]
fn manacher(st: &str) -> Vec<usize> {
    let mut t = "".to_owned();
    st.chars().for_each(|c| {
        t.push('#');
        t.push(c);
    });
    t.push('#');
    let p = manacher_odd(&t);
    p[1..p.len() - 1].to_vec()
}

#[allow(dead_code)]
// https://cp-algorithms.com/string/z-function.html
fn z_function(s: &Vec<u8>) -> Vec<usize> {
    let n = s.len();
    let mut z = vec![0; n];
    let mut l = 0;
    let mut r = 0;
    (1..n).for_each(|i| {
        if i < r {
            z[i] = min(r - i, z[i - l]);
        }
        while (i + z[i] < n) && (s[z[i]] == s[i + z[i]]) {
            z[i] += 1;
        }
        if i + z[i] > r {
            l = i;
            r = i + z[i];
        }
    });
    z
}

#[allow(dead_code)]
fn is_palindrome(s: &[u8]) -> bool {
    let mut i = 0;
    let mut j = s.len() - 1;
    while i < j {
        if s[i] != s[j] {
            return false;
        }
        i += 1;
        j -= 1;
    }
    true
}
