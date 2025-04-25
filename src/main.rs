use std::io::{stdin, stdout, BufReader, BufWriter, Stdin, StdoutLock, Write};

// pub mod basic_graph;
// pub mod special_combination;
// pub mod root_tree;
// pub mod number_theory;
// use basic_graph::Graph;
// use binomial::BinomialPrime;
// use static_modint::ModInt998244353 as Mint;
// pub mod dp;
// pub mod range_counter;
// pub mod segtree;
// use segtree::LiChaoTree;
// use segtree::RangeAffineRangeSum;
pub mod utils;

type VV<T> = Vec<Vec<T>>;
type US = usize;

fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<StdoutLock>) {
    let default = 0usize;
    let t = utils::read_1_number(line, reader, default);
    (0..t).for_each(|_te| {});
}

fn main() {
    let mut reader = BufReader::new(stdin());
    let mut line = String::new();
    let mut out = BufWriter::new(stdout().lock());

    solve(&mut reader, &mut line, &mut out);
}
