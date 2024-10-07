#[allow(dead_code)]
// List primes <= 10^7
fn linear_sieve() -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut lp: Vec<usize> = vec![0; 10000001];
    let mut pr: Vec<usize> = Vec::new();
    let mut idx: Vec<usize> = vec![0; 10000001];
    let c = 10000000u64;
    unsafe {
        (2..=10000000).for_each(|i| {
            if lp.get_unchecked(i) == &0 {
                lp[i] = i;
                pr.push(i);
            }
            let mut j = 0;
            while j < pr.len()
                && *pr.get_unchecked(j) <= *lp.get_unchecked(i)
                && (i as u64) * (*pr.get_unchecked(j) as u64) <= c
            {
                lp[i * *pr.get_unchecked(j)] = *pr.get_unchecked(j);
                j += 1;
            }
        });
    }
    // Mapping: prime -> index
    pr.iter().enumerate().for_each(|(i, &prime)| {
        idx[prime] = i + 1;
    });
    // Lowest prime factor
    // list of prime number
    // prime -> index mapping
    (lp, pr, idx)
}
