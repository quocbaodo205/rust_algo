use std::io::{stdin, stdout, BufReader, BufWriter, Stdin, StdoutLock, Write};

// pub mod basic_graph;
// pub mod special_combination;
// pub mod root_tree;
// pub mod tree_decomp;
// pub mod number_theory;
// use basic_graph::UndirectedGraph;
// pub mod dfs_tree;
// pub mod hld;
// use binomial::BinomialPrime;
// use static_modint::ModInt998244353 as Mint;
// pub mod dp;
// pub mod range_counter;
// pub mod segtree;
// use segtree::LiChaoTree;
// use segtree::RangeAffineRangeSum;
// pub mod rng;
// use rng::SmallRng;
pub mod utils;
// pub mod xor_basic;
// use xor_basic::XorBasic;
// pub mod string_utils;
// use range_minimum_query::RangeMinimumQuery;
// use suffix_array::SuffixArray;

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
