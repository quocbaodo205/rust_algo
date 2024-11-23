#[allow(dead_code)]

fn mobius() -> Vec<i8> {
    let mut tag: Vec<bool> = vec![false; 1000001];
    let mut pr: Vec<usize> = Vec::new();
    let mut mu: Vec<i8> = vec![0; 1000001];
    mu[1] = 1;
    let c = 1000000u64;
    (2..=1000000).for_each(|i| {
        if !tag[i] {
            pr.push(i);
            mu[i] = -1;
        }
        let mut j = 0;
        while j < pr.len() && (i as u64) * (pr[j] as u64) <= c {
            tag[i * pr[j]] = true;
            if i % pr[j] != 0 {
                mu[i * pr[j]] = -mu[i];
            } else {
                break;
            }
            j += 1;
        }
    });
    mu
}
