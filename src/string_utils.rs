use std::cmp::{max, min};

// ========================= simple search and misc ====================

// Template for sliding window of size d.
// Copy and modify.
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

/// Search all (prefix, revsuffix) pairs efficiently.
/// Copy inner code and change accordingly.
pub fn search_prefix_revsuffix(p: &Vec<u8>) {
    // Usually prefix is the input so no need to clone
    // let mut prefix: Vec<u8> = a;

    // First style: Check prefix first and then proceed with suffix creation.
    // Faster in case prefix condition is kinda hard to meet, so run in reverse.
    let pn = p.len();
    for i in (1..pn - 1).rev() {
        let prefix = &p[0..i];
        if true {
            let suffix: Vec<u8> = p[i..].iter().rev().cloned().collect();
        }
    }

    // 2nd style: Prefix and suffix calculate independent.
    // Much faster to just go and create both suffix and prefix.
    let mut prefix: Vec<u8> = p.iter().cloned().collect();
    let mut suffix: Vec<u8> = Vec::new();
    suffix.reserve(pn);
    for i in 1..pn {
        suffix.push(prefix.pop().unwrap());
        // TODO: Work on the prefix and suffix
    }
}

pub fn unique_concaternation(a: &Vec<Vec<u8>>) -> (Vec<u8>, Vec<usize>) {
    // Use these 39 chars and cycle them to create a maybe unique seperator for all input strings.
    let all_chars = [
        '-', '+', '_', '=', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', '`', '~',
        '[', ']', ':', ';', ',', '<', '>', '.', '?', '/', '|', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', '0',
    ];
    let mut ans = Vec::<u8>::new();
    let mut all_length: usize = a.iter().map(|s| s.len()).sum();
    all_length += a.len() - 1;
    ans.reserve(all_length);
    // All position will be map back to the index of the original string, except the seperator is a.len().
    let mut remap = vec![a.len(); all_length];
    let mut cur_sep = 0;
    for i in 0..a.len() {
        for &c in a[i].iter() {
            remap[ans.len()] = i;
            ans.push(c);
        }
        if i == a.len() - 1 {
            break;
        }
        ans.push(all_chars[cur_sep] as u8);
        cur_sep = (cur_sep + 1) % all_chars.len();
    }
    (ans, remap)
}

pub fn print_string_with_index(s: &Vec<u8>) {
    for (i, &c) in s.iter().enumerate() {
        println!("{i:2} : {}", c as char);
    }
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

// https://cp-algorithms.com/string/z-function.html
pub fn z_function(s: &[u8]) -> Vec<usize> {
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

/// Find all occurences of pattern p in a text t.
/// Both pattern and text is known beforehand.
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

// ============================== Suffix array stuff ===========================

use suffix_array::SuffixArray;

use crate::utils;

/// Find first different char positon
pub fn first_mismatch(a: &[u8], b: &[u8]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Find all occurences of pattern p in a text t.
/// Text is known beforehand, and can be kept by inputing the sa.
/// Return the first and last occ in sa. Complexity O(|p| log |t|) for binsearch and compare.
pub fn sa_search(t: &[u8], sa: &SuffixArray, p: &[u8]) -> (usize, usize) {
    // sa.suffix_array contains all start pos of the suffix in order.
    // Binary search and compare all suffix to a pattern.
    let mut l = 0;
    let mut r = sa.suffix_array().len() - 1;
    let pn = p.len();

    // println!("Finding first occ, p = {p:?}");
    let mut first_sa_occ = r + 1;
    while l <= r {
        let mid = (l + r) / 2;
        let suffix = &t[sa.suffix_array()[mid]..min(t.len(), sa.suffix_array()[mid] + pn)];
        let mm = first_mismatch(suffix, p);
        // println!(
        //     "mid = {mid}, sa[{mid}] = {}, range = {:?}, suffix = {suffix:?}, mm = {mm}",
        //     sa.suffix_array()[mid],
        //     sa.suffix_array()[mid]..min(t.len(), sa.suffix_array()[mid] + pn)
        // );
        // is an occurence: part of the suffix match all of p
        if mm == pn {
            first_sa_occ = mid;
            // println!("update first occ = sa[{first_occ}]");
            if mid == 0 {
                break;
            }
            r = mid - 1; // Check smaller strings
        } else {
            if mm == suffix.len() {
                // Used all the suffix, has to check bigger strings
                l = mid + 1; // Check bigger strings
            } else if suffix[mm] < p[mm] {
                // suffix is smaller, check bigger strings
                l = mid + 1;
            } else {
                // suffix is bigger, check smaller strings
                if mid == 0 {
                    break;
                }
                r = mid - 1; // Check smaller strings
            }
        }
    }

    // println!("=====================================");
    // println!("Finding last occ, p = {p:?}");
    l = 0;
    r = sa.suffix_array().len() - 1;
    let mut last_sa_occ = r + 1;
    while l <= r {
        let mid = (l + r) / 2;
        let suffix = &t[sa.suffix_array()[mid]..min(t.len(), sa.suffix_array()[mid] + pn)];
        let mm = first_mismatch(suffix, p);
        // println!(
        //     "mid = {mid}, sa[{mid}] = {}, range = {:?}, suffix = {suffix:?}, mm = {mm}",
        //     sa.suffix_array()[mid],
        //     sa.suffix_array()[mid]..min(t.len(), sa.suffix_array()[mid] + pn)
        // );
        // is an occurence: part of the suffix match all of p
        if mm == pn {
            last_sa_occ = mid;
            // println!("update last occ = sa[{last_occ}]");
            l = mid + 1; // Check bigger strings
        } else {
            if mm == suffix.len() {
                // Used all the suffix, has to check bigger strings
                l = mid + 1; // Check bigger strings
            } else if suffix[mm] < p[mm] {
                // suffix is smaller, check bigger strings
                l = mid + 1;
            } else {
                // suffix is bigger, check smaller strings
                if mid == 0 {
                    break;
                }
                r = mid - 1; // Check smaller strings
            }
        }
    }
    // First true occurence in the string t can be found using either search or RMQ base on the query type.
    // Usually search is faster for long patterns and creating RMQ is costly.
    (first_sa_occ, last_sa_occ)
}

/// Given an index i of the suffix array with a len, find the maximum range [l..=r]
/// so that the lcp of the range >= len. Works in O(log n)
pub fn get_range_with_lcp_i(sa: &SuffixArray, i: usize, len: usize) -> (usize, usize) {
    // Not even full suffix at sa[i] can fit
    if sa.suffix_array().len() - sa.suffix_array()[i] < len {
        return (sa.suffix_array().len(), sa.suffix_array().len());
    }
    // Find the right bound first
    let mut l = i;
    let mut r = sa.suffix_array().len() - 1;

    let mut right_bound = l;
    while l <= r {
        let mid = (l + r) / 2;
        if sa.lcp(sa.suffix_array()[i], sa.suffix_array()[mid]) >= len {
            right_bound = mid;
            l = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            r = mid - 1;
        }
    }

    let mut l = 0;
    let mut r = i;

    let mut left_bound = r;
    while l <= r {
        let mid = (l + r) / 2;
        if sa.lcp(sa.suffix_array()[i], sa.suffix_array()[mid]) >= len {
            left_bound = mid;
            if mid == 0 {
                break;
            }
            r = mid - 1;
        } else {
            l = mid + 1;
        }
    }

    (left_bound, right_bound)
}

/// Template to explore all ranges with the LCP(l..=r) with all len
/// Should be copy out and modify exploration function. Works in O(n) to explore all.
/// Do note that LCP always deal with atleast 2 strings, so need another run for 1 string.
pub fn all_range_with_lcp(sa: &SuffixArray) {
    let (left_pos_smaller, right_pos_smaller) = utils::first_smaller_pos(sa.lcp_array());
    // lcp[i]: lcp(sa[i], sa[i+1])
    for (i, &lcp_val) in sa.lcp_array().iter().enumerate() {
        let r = right_pos_smaller[i];
        let l = left_pos_smaller[i];
        // TODO: Work on range [l..=r]
    }
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
