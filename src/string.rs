use std::cmp::{max, min};

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
