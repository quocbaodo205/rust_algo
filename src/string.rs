use std::cmp::{max, min};

// ========================= simple search and misc ====================

// Template for sliding window of size d
#[allow(dead_code)]
fn sliding_windows_d(s: &[u8], d: usize, f: &dyn Fn(usize) -> usize) {
    // Process first substr
    let mut start = 0;
    let mut end = start + d - 1;
    let mut contrib = 0; // Contribution of each position
                         // TODO: Calculate first contrib of [start..=end]
    (start..=end).for_each(|i| {
        contrib += 1;
    });

    // Move up and process each substr
    while end + 1 < s.len() {
        // Minus old contrib
        contrib -= f(start);

        // Plus new contrib
        start += 1;
        end += 1;
        contrib += f(end);
    }
}

// Searching all substring by length first and sliding windows
#[allow(dead_code)]
fn all_sub_str_template(s: &[u8], f: &dyn Fn(usize) -> usize) {
    // Any substring of length d
    (1..=s.len()).for_each(|d| {
        sliding_windows_d(s, d, f);
    });
}

// Compress binary string into consecutive (1, len) and (0, len)
#[allow(dead_code)]
fn compress_01(s: &[u8]) -> Vec<(u8, usize)> {
    let mut cur = s[0];
    let mut count = 1;

    let mut ans = Vec::new();
    (1..s.len()).for_each(|i| {
        if s[i] != cur {
            // Break down
            ans.push((cur, count));
            count = 0;
        }
        cur = s[i];
        count += 1;
    });
    // Last one
    ans.push((cur, count));

    ans
}

// ========================= palindrome algo =====================

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
fn manacher(st: &str) -> (String, Vec<usize>) {
    let mut t = "".to_owned();
    st.chars().for_each(|c| {
        t.push('#');
        t.push(c);
    });
    t.push('#');
    let p = manacher_odd(&t);
    // at i: t[i - (p[i]-1) ..= i + (p[i]-1)] is the longest palindrome
    // you can use the returned t and remove some characters from it.
    (t, p)
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

// ========================= Z-functions related algo =====================

#[allow(dead_code)]
// https://cp-algorithms.com/string/z-function.html
fn z_function(s: &[u8]) -> Vec<usize> {
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
fn z_search(t: &[u8], p: &[u8]) {
    // s = p + # + t
    let mut s: Vec<u8> = Vec::from_iter(p.iter().cloned());
    s.push(b'#');
    s.extend_from_slice(t);

    let z = z_function(&s);
    (0..t.len()).for_each(|i| {
        let k = z[i + p.len() + 1];
        if k == p.len() {
            // string p occured at position i of t
            // TODO: ...
        }
    });
}

#[allow(dead_code)]
// Smallest compression of string s = t + t + ... + t. Return the size of t.
fn z_compress(s: &[u8]) -> usize {
    let z = z_function(&s);
    for i in 0..s.len() {
        if s.len() % (i + 1) != 0 {
            continue;
        }
        if i + 1 + z[i] == s.len() {
            return i + 1;
        }
    }
    s.len()
}

// ========================= Classic DP algo =====================

#[allow(dead_code)]
fn longest_common_subsequence(s1: &[u8], s2: &[u8]) -> usize {
    let n = s1.len();
    let m = s2.len();

    // LCS of s1[..i] and s2[..j]
    let mut dp: Vec<Vec<usize>> = vec![vec![0; m + 1]; n + 1];
    (0..n).for_each(|i| {
        (0..m).for_each(|j| {
            if s1[i] == s2[j] {
                // inc the longest LCS
                dp[i + 1][j + 1] = max(dp[i + 1][j + 1], dp[i][j] + 1);
            } else {
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
            }
        });
    });

    dp[n][m]
}

#[allow(dead_code)]
fn edit_distance(s1: &[u8], s2: &[u8]) -> usize {
    let n = s1.len();
    let m = s2.len();

    // Edit distance of s1[..i] and s2[..j]
    let mut dp: Vec<Vec<usize>> = vec![vec![0; m + 1]; n + 1];
    (0..n).for_each(|i| dp[i + 1][0] = i + 1); // All delete
    (0..m).for_each(|i| dp[0][i + 1] = i + 1); // All insert
    (0..n).for_each(|i| {
        (0..m).for_each(|j| {
            dp[i + 1][j + 1] = min(
                dp[i][j + 1] + 1, // Delete this character from s1[..i]
                min(
                    dp[i + 1][j] + 1,                              // Insert s2[j] to s1[..i]
                    dp[i][j] + if s1[i] == s2[j] { 0 } else { 1 }, // Match / replace
                ),
            );
        });
    });

    dp[n][m]
}
