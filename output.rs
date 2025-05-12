// use std::{
//     collections::VecDeque,
//     io::{stdin, stdout, BufReader, BufWriter, Stdin, StdoutLock, Write},
// };

// pub mod basic_graph;
// // pub mod special_combination;
// // pub mod root_tree;
// pub mod tree_decomp;
// // pub mod number_theory;
// use basic_graph::UndirectedGraph;
// // use binomial::BinomialPrime;
// // use static_modint::ModInt998244353 as Mint;
// // pub mod dp;
// // pub mod range_counter;
// // pub mod segtree;
// // use segtree::LiChaoTree;
// // use segtree::RangeAffineRangeSum;
// pub mod utils;

// type VV<T> = Vec<Vec<T>>;
// type US = usize;

// fn solve(reader: &mut BufReader<Stdin>, line: &mut String, out: &mut BufWriter<StdoutLock>) {
//     let default = 0usize;
//     // let t = utils::read_1_number(line, reader, default);
//     // (0..t).for_each(|_te| {});
//     let n = utils::read_1_number(line, reader, default);
//     let uelist = utils::read_edge_list(n - 1, line, reader);
//     let s = utils::read_line_str_as_vec_template(line, reader);
//     let a: Vec<usize> = s.iter().map(|&c| 1 << (c - b'a')).collect();

//     let g = UndirectedGraph::from_unweighted_edges(n, &uelist);
//     let tree = tree_decomp::shallowest_decomposition_tree(0, &g);
//     let bfs_list = tree_decomp::bfs_list(&tree);

//     // All palindrome bitmask for 20 characters
//     let mut palindrome: Vec<usize> = (0..21).map(|i| 1 << i).collect();
//     palindrome[20] = 0; // Case for 0 odd palindrome.
//     let mut parent = vec![usize::MAX; n];
//     let mut bitmask = vec![0; n];
//     let mut bitmask_count = vec![0u64; 1 << 20];
//     let mut ans = vec![1u64; n]; // Always have 1 palindrome of 1 char.
//     let mut dp = vec![0u64; n];

//     let mut d_graph: Vec<Vec<usize>> = vec![Vec::new(); n];
//     for u in 0..n {
//         for &(v, _) in g[u].iter() {
//             d_graph[u].push(v);
//         }
//     }
//     // Check each decomposition node in bfs manner:
//     for &decomp_node in bfs_list.iter() {
//         // Effectively delete the decomp_node (decompose at this node)
//         parent[decomp_node] = decomp_node;
//         for v in d_graph[decomp_node].clone().into_iter() {
//             // Find, swap last and remove.
//             if let Some(idx) = d_graph[v].iter().position(|&x| x == decomp_node) {
//                 d_graph[v].swap_remove(idx);
//             }
//         }

//         // Get BFS order traversal for every subtree after decomposition
//         let mut all_bfss: VV<US> = Vec::new();
//         // With decomp node, flaten out but still in order of every child
//         let mut all_bfss_flat: Vec<US> = vec![decomp_node];
//         for &child in d_graph[decomp_node].iter() {
//             parent[child] = decomp_node;
//             let mut bfs_at_child = Vec::<US>::new();
//             let mut cur: VecDeque<US> = VecDeque::new();
//             cur.push_back(child);
//             while let Some(u) = cur.pop_front() {
//                 bfs_at_child.push(u);
//                 for &v in d_graph[u].iter() {
//                     if parent[v] == usize::MAX {
//                         parent[v] = u;
//                         cur.push_back(v);
//                     }
//                 }
//             }
//             // Clone out and put to flat list
//             all_bfss_flat.extend(bfs_at_child.clone().into_iter());
//             all_bfss.push(bfs_at_child);
//         }

//         // Cal single path bitmask from decomp_node to every other node
//         for &u in all_bfss_flat.iter() {
//             bitmask[u] = bitmask[parent[u]] ^ a[u]; // bitmask[u]: all mask from decomp_node to u.
//             bitmask_count[bitmask[u]] += 1;
//         }

//         for bfs in all_bfss.iter() {
//             // Minus out everything for this subtree.
//             // So whatever count we currently have is for all other node without this subtree.
//             for &u in bfs.iter() {
//                 bitmask_count[bitmask[u]] -= 1;
//             }

//             // Calculate dp[u] in reverse manner: from leave to top (almost decomp node).
//             // DP[u] store the result of the subtree under u for this decomposition.
//             for &u in bfs.iter().rev() {
//                 let base = bitmask[u] ^ a[decomp_node]; // From u to before decomp_node
//                 for &p in palindrome.iter() {
//                     // Find all combination that gives a palindrome
//                     // from this branch to all other branches (combination path count)
//                     dp[u] += bitmask_count[base ^ p];
//                 }
//                 // Propagate up to parent.
//                 dp[parent[u]] += dp[u];
//             }

//             // Plus back to get ready for other subtree.
//             for &u in bfs.iter() {
//                 bitmask_count[bitmask[u]] += 1;
//             }
//         }

//         // Special calculation for the decomp_node: The single path count.
//         bitmask_count[a[decomp_node]] -= 1; // Minus out to avoid only one node case: already accounted for.
//         for &p in palindrome.iter() {
//             dp[decomp_node] += bitmask_count[p];
//         }
//         dp[decomp_node] /= 2; // Avoiding double count for decomp_node, as its parent is itself.

//         // Put to answer and reset all to prepare for the next decomposition.
//         for &u in all_bfss_flat.iter() {
//             ans[u] += dp[u];
//             dp[u] = 0;
//             parent[u] = usize::MAX;
//             bitmask_count[bitmask[u]] = 0;
//             bitmask[u] = 0;
//         }
//     }
//     for &x in ans.iter() {
//         write!(out, "{x} ").unwrap();
//     }
//     writeln!(out).unwrap();
// }

// fn main() {
//     let mut reader = BufReader::new(stdin());
//     let mut line = String::new();
//     let mut out = BufWriter::new(stdout().lock());

//     solve(&mut reader, &mut line, &mut out);
// }

pub use __cargo_equip::prelude::*;use std::{collections::{BTreeSet,VecDeque},io::{stdin,stdout,BufReader,BufWriter,Stdin,StdoutLock,Write},};pub mod basic_graph{pub use crate::__cargo_equip::prelude::*;use std::{collections::{BTreeMap,BTreeSet,HashSet,VecDeque},ops,};type V<T> =Vec<T>;type VV<T> =V<V<T>>;type Set<T> =BTreeSet<T>;type Map<K,V> =BTreeMap<K,V>;type US=usize;type UU=(US,US);#[allow(dead_code)]#[derive(Clone,Copy,PartialEq,Eq)]pub enum CheckState{NOTCHECK,CHECKING,CHECKED,}use std::ops::Index;#[doc=" CSR 形式の二次元配列"]#[derive(Clone)]pub struct CSRArray<T>{pos:Vec<usize>,data:Vec<T>,}impl<T:Clone>CSRArray<T>{#[doc=" a の各要素 (i, x) について、 i 番目の配列に x が格納される"]pub fn new(n:usize,a:&[(usize,T)])->Self{let mut pos=vec![0;n+1];for&(i,_)in a{pos[i]+=1;}for i in 0..n{pos[i+1]+=pos[i];}let mut ord=vec![0;a.len()];for j in(0..a.len()).rev(){let(i,_)=a[j];pos[i]-=1;ord[pos[i]]=j;}let data=ord.into_iter().map(|i|a[i].1.clone()).collect();Self{pos,data}}}impl<T>CSRArray<T>{pub fn len(&self)->usize{self.pos.len()-1}pub fn iter(&self)->impl Iterator<Item=&[T]>{(0..self.len()).map(|i|&self[i])}}impl<T>Index<usize>for CSRArray<T>{type Output=[T];fn index(&self,i:usize)->&Self::Output{let start=self.pos[i];let end=self.pos[i+1];&self.data[start..end]}}#[doc=" 隣接リスト"]#[derive(Clone)]pub struct Graph<V,E,const DIRECTED:bool>where V:Clone,E:Clone,{vertices:Vec<V>,edges:CSRArray<(usize,E)>,}pub type DirectedGraph<V,E> =Graph<V,E,true>;pub type UndirectedGraph<V,E> =Graph<V,E,false>;#[doc=" グリッドグラフの 4 近傍  "]#[doc=" 上, 左, 下, 右"]pub const GRID_NEIGHBOURS_4:[(usize,usize);4]=[(!0,0),(0,!0),(1,0),(0,1)];#[doc=" グリッドグラフの 8 近傍  "]#[doc=" 上, 左, 下, 右, 左上, 左下, 右下, 右上"]pub const GRID_NEIGHBOURS_8:[(usize,usize);8]=[(!0,0),(0,!0),(1,0),(0,1),(!0,!0),(1,!0),(1,1),(!0,1),];impl<V,E>DirectedGraph<V,E>where V:Clone,E:Clone,{#[doc=" 頂点重みと重み付き有向辺からグラフを構築する"]pub fn from_vertices_and_edges(vertices:&[V],edges:&[(usize,usize,E)])->Self{let edges=edges.iter().map(|(u,v,w)|(*u,(*v,w.clone()))).collect::<Vec<_>>();Self{vertices:vertices.to_vec(),edges:CSRArray::new(vertices.len(),&edges),}}#[doc=" グリッドからグラフを構築する"]#[doc=""]#[doc=" # 引数"]#[doc=""]#[doc=" * `grid` - グリッド"]#[doc=" * `neighbours` - 近傍"]#[doc=" * `cost` - grid の値から重みを計算する関数"]pub fn from_grid(grid:&Vec<Vec<V>>,neighbours:&[(usize,usize)],cost:impl Fn(&V,&V)->Option<E>,)->Self{let h=grid.len();let w=grid[0].len();let mut edges=vec![];for i in 0..h{for j in 0..w{for&(di,dj)in neighbours{let ni=i.wrapping_add(di);let nj=j.wrapping_add(dj);if ni>=h||nj>=w{continue;}if let Some(c)=cost(&grid[i][j],&grid[ni][nj]){edges.push((i*w+j,ni*w+nj,c));}}}}Self::from_vertices_and_edges(&grid.into_iter().flatten().cloned().collect::<Vec<_>>(),&edges,)}}impl<V,E>UndirectedGraph<V,E>where V:Clone,E:Clone,{#[doc=" 頂点重みと重み付き無向辺からグラフを構築する"]pub fn from_vertices_and_edges(vertices:&[V],edges:&[(usize,usize,E)])->Self{let edges=edges.iter().map(|(u,v,w)|[(*u,(*v,w.clone())),(*v,(*u,w.clone()))]).flatten().collect::<Vec<_>>();Self{vertices:vertices.to_vec(),edges:CSRArray::new(vertices.len(),&edges),}}#[doc=" グリッドからグラフを構築する"]#[doc=""]#[doc=" # 引数"]#[doc=""]#[doc=" * `grid` - グリッド"]#[doc=" * `neighbours` - 近傍"]#[doc=" * `cost` - grid の値から重みを計算する関数"]pub fn from_grid(grid:&Vec<Vec<V>>,neighbours:&[(usize,usize)],cost:impl Fn(&V,&V)->Option<E>,)->Self{let h=grid.len();let w=grid[0].len();let mut edges=vec![];for i in 0..h{for j in 0..w{for&(di,dj)in neighbours{let ni=i.wrapping_add(di);let nj=j.wrapping_add(dj);if ni>=h||nj>=w{continue;}let u=i*w+j;let v=ni*w+nj;if u>v{continue;}if let Some(c)=cost(&grid[i][j],&grid[ni][nj]){edges.push((u,v,c));}}}}Self::from_vertices_and_edges(&grid.into_iter().flatten().cloned().collect::<Vec<_>>(),&edges,)}}impl<V,E,const DIRECTED:bool>Graph<V,E,DIRECTED>where V:Clone,E:Clone,{#[doc=" 頂点数を返す"]pub fn len(&self)->usize{self.vertices.len()}#[doc=" 頂点重みを返す"]pub fn vertex(&self,v:usize)->&V{&self.vertices[v]}#[doc=" 頂点 v から出る辺を返す  "]#[doc=" g\\[v\\] と同じ"]pub fn edges(&self,v:usize)->&[(usize,E)]{&self.edges[v]}}impl<V,E,const DIRECTED:bool>Index<usize>for Graph<V,E,DIRECTED>where V:Clone,E:Clone,{type Output=[(usize,E)];fn index(&self,v:usize)->&[(usize,E)]{self.edges(v)}}impl<V>DirectedGraph<V,()>where V:Clone,{#[doc=" 頂点重みと重みなし有向辺からグラフを構築する"]pub fn from_vertices_and_unweighted_edges(vertices:&[V],edges:&[(usize,usize)])->Self{Self::from_vertices_and_edges(vertices,&edges.iter().map(|&(u,v)|(u,v,())).collect::<Vec<_>>(),)}}impl<V>UndirectedGraph<V,()>where V:Clone,{#[doc=" 頂点重みと重みなし無向辺からグラフを構築する"]pub fn from_vertices_and_unweighted_edges(vertices:&[V],edges:&[(usize,usize)])->Self{Self::from_vertices_and_edges(vertices,&edges.iter().map(|&(u,v)|(u,v,())).collect::<Vec<_>>(),)}}impl<E>DirectedGraph<(),E>where E:Clone,{#[doc=" 重み付き有向辺からグラフを構築する"]pub fn from_edges(n:usize,edges:&[(usize,usize,E)])->Self{Self::from_vertices_and_edges(&vec![();n],edges)}}impl<E>UndirectedGraph<(),E>where E:Clone,{#[doc=" 重み付き無向辺からグラフを構築する"]pub fn from_edges(n:usize,edges:&[(usize,usize,E)])->Self{Self::from_vertices_and_edges(&vec![();n],edges)}}impl DirectedGraph<(),()>{#[doc=" 重みなし有向辺からグラフを構築する"]pub fn from_unweighted_edges(n:usize,edges:&[(usize,usize)])->Self{Self::from_edges(n,&edges.iter().map(|&(u,v)|(u,v,())).collect::<Vec<_>>(),)}}impl UndirectedGraph<(),()>{#[doc=" 重みなし無向辺からグラフを構築する"]pub fn from_unweighted_edges(n:usize,edges:&[(usize,usize)])->Self{Self::from_edges(n,&edges.iter().map(|&(u,v)|(u,v,())).collect::<Vec<_>>(),)}}pub struct ZeroOneBFSResult<T>where T:Clone+Ord+Add<Output=T>+Default,{pub dist:Vec<T>,pub prev:Vec<usize>,}#[doc=" 0-1 BFS  "]#[doc=" 辺の重みが 0 か 1 のグラフ上で、始点から各頂点への最短距離を求める"]#[doc=""]#[doc=" # 戻り値"]#[doc=""]#[doc=" ZeroOneBFSResult"]#[doc=" - dist: 始点から各頂点への最短距離"]#[doc=" - prev: 始点から各頂点への最短経路における前の頂点"]pub fn zero_one_bfs<V,T,const DIRECTED:bool>(g:&Graph<V,T,DIRECTED>,starts:&[usize],inf:T,)->ZeroOneBFSResult<T>where V:Clone,T:Clone+Ord+Add<Output=T>+Default+From<u8>,{assert!(starts.len()>0);let zero=T::from(0);let one=T::from(1);let mut dist=vec![inf.clone();g.len()];let mut prev=vec![!0;g.len()];let mut dq=VecDeque::new();for&s in starts{dist[s]=T::default();dq.push_back((zero.clone(),s));}while let Some((s,v))=dq.pop_front(){if dist[v]<s{continue;}for(u,w)in&g[v]{assert!(*w==zero||*w==one);let t=dist[v].clone()+w.clone();if dist[*u]>t{dist[*u]=t.clone();prev[*u]=v;if*w==zero{dq.push_front((t.clone(),*u));}else{dq.push_back((t.clone(),*u));}}}}ZeroOneBFSResult{dist,prev}}impl<T>ZeroOneBFSResult<T>where T:Clone+Ord+Add<Output=T>+Default,{#[doc=" 始点から頂点 v への最短経路を求める  "]#[doc=" 経路が存在しない場合は None を返す"]pub fn path(&self,mut v:usize)->Option<Vec<usize>>{if self.dist[v].clone()!=T::default()&&self.prev[v]==!0{return None;}let mut path=vec![];while v!=!0{path.push(v);v=self.prev[v];}path.reverse();Some(path)}}use std::{cmp::Reverse,collections::BinaryHeap,ops::Add};pub struct DijkstraResult<T>where T:Clone+Ord+Add<Output=T>+Default,{pub dist:Vec<T>,pub prev:Vec<usize>,}#[doc=" ダイクストラ法  "]#[doc=" 始点集合からの最短距離を求める。"]pub fn dijkstra<V,T,const DIRECTED:bool>(g:&Graph<V,T,DIRECTED>,starts:&[usize],inf:T,)->DijkstraResult<T>where V:Clone,T:Clone+Ord+Add<Output=T>+Default,{assert!(starts.len()>0);let mut dist=vec![inf.clone();g.len()];let mut prev=vec![!0;g.len()];let mut pq=BinaryHeap::new();for&s in starts{dist[s]=T::default();pq.push(Reverse((T::default(),s)));}while let Some(Reverse((s,v)))=pq.pop(){if dist[v]<s{continue;}for(u,w)in&g[v]{assert!(w.clone()>=T::default());if dist[*u]>dist[v].clone()+w.clone(){dist[*u]=dist[v].clone()+w.clone();prev[*u]=v;pq.push(Reverse((dist[*u].clone(),*u)));}}}DijkstraResult{dist,prev}}impl<T>DijkstraResult<T>where T:Clone+Ord+Add<Output=T>+Default,{#[doc=" 終点 v までの最短経路を求める。"]#[doc=" 終点に到達できない場合は None を返す。"]pub fn path(&self,mut v:usize)->Option<Vec<usize>>{if self.dist[v].clone()!=T::default()&&self.prev[v]==!0{return None;}let mut path=vec![];while v!=!0{path.push(v);v=self.prev[v];}path.reverse();Some(path)}}#[allow(dead_code)]fn dfs_has_cycle<V,T,const DIRECTED:bool>(u:US,g:&Graph<V,T,DIRECTED>,state:&mut Vec<CheckState>,)->bool where V:Clone,T:Clone+Ord+Add<Output=T>+Default+From<u8>,{state[u]=CheckState::CHECKING;for(v,_)in&g[u]{let has_cycle_v=match state[*v]{CheckState::NOTCHECK=>dfs_has_cycle(*v,g,state),CheckState::CHECKING=>true,CheckState::CHECKED=>false,};if has_cycle_v{return true;}}state[u]=CheckState::CHECKED;false}#[allow(dead_code)]fn dfs_topo<V,T,const DIRECTED:bool>(u:US,g:&Graph<V,T,DIRECTED>,state:&mut Vec<CheckState>,ts:&mut Vec<US>,)where V:Clone,T:Clone+Ord+Add<Output=T>+Default+From<u8>,{state[u]=CheckState::CHECKED;for(v,_)in&g[u]{if state[*v]==CheckState::NOTCHECK{dfs_topo(*v,g,state,ts);}}ts.push(u);}#[allow(dead_code)]fn find_topo<V,T,const DIRECTED:bool>(g:&Graph<V,T,DIRECTED>,state:&mut Vec<CheckState>,)->Vec<US>where V:Clone,T:Clone+Ord+Add<Output=T>+Default+From<u8>,{let mut ts:Vec<US> =Vec::new();(0..state.len()).for_each(|u|{if state[u]==CheckState::NOTCHECK{dfs_topo(u,g,state,&mut ts)}});ts.reverse();ts}#[derive(Copy,Clone,Debug)]struct Point(i32,i32);impl ops::Add for Point{type Output=Point;fn add(self,rhs:Point)->Self::Output{Point(self.0+rhs.0,self.1+rhs.1)}}impl ops::Sub for Point{type Output=Point;fn sub(self,rhs:Point)->Self::Output{Point(self.0-rhs.0,self.1-rhs.1)}}#[allow(dead_code)]fn is_valid_point(x:Point,n:usize,m:usize)->bool{return x.0>=0&&x.0<n as i32&&x.1>=0&&x.1<m as i32;}#[allow(dead_code)]fn translate(c:u8)->Point{match c as char{'U'=>Point(-1,0),'D'=>Point(1,0),'L'=>Point(0,-1),'R'=>Point(0,1),_=>Point(0,0),}}#[allow(dead_code)]fn dfs_grid(s:Point,mp:&V<V<u8>>,checked:&mut V<V<bool>>){let n=mp.len();let m=mp[0].len();if checked[s.0 as usize][s.1 as usize]{return;}let mut s=Some(s);while let Some(u)=s{checked[u.0 as usize][u.1 as usize]=true;let v=u+translate(mp[u.0 as usize][u.1 as usize]);s=match is_valid_point(v,n,m)&&!checked[v.0 as usize][v.1 as usize]{true=>Some(v),false=>None,}}}#[allow(dead_code)]fn flood_grid(s:Point,blocked:&V<V<bool>>,checked:&mut V<V<bool>>){let n=blocked.len();let m=blocked[0].len();if checked[s.0 as usize][s.1 as usize]{return;}let mut q:VecDeque<Point> =VecDeque::new();let direction=[Point(0,1),Point(0,-1),Point(1,0),Point(-1,0)];q.push_back(s);while let Some(u)=q.pop_front(){direction.iter().for_each(|&d|{let v=u+d;if is_valid_point(v,n,m)&&!checked[v.0 as usize][v.1 as usize]&&!blocked[v.0 as usize][v.1 as usize]{checked[v.0 as usize][v.1 as usize]=true;q.push_back(v);}});}}}pub mod tree_decomp{pub use crate::__cargo_equip::prelude::*;use std::collections::VecDeque;use crate::{basic_graph::{Graph,UndirectedGraph},utils,};type VV<T> =Vec<Vec<T>>;type US=usize;type UU=(US,US);#[doc=" Structure that resembled a rooted tree."]#[doc=" Follow from root via children, guarantee the chain from root to any leaf is the shallowest."]#[doc=" Tree height is bounded by O(log n)."]pub struct Arborescence{root:usize,chilren:VV<US>,}fn extract_chain(labels:usize,u:usize,decomp_tree:&mut VV<US>,stacks:&mut VV<US>){let mut labels=labels;let mut u=u;while labels>0{let label=labels.ilog2()as usize;labels^=1<<label;if let Some(v)=stacks[label].pop(){decomp_tree[u].push(v);u=v;}else{break;}}}fn dfs_label<V,T,const DIRECTED:bool>(u:usize,p:usize,g:&Graph<V,T,DIRECTED>,forbid:&mut Vec<usize>,decomp_tree:&mut VV<US>,stacks:&mut VV<US>,)where V:Clone,T:Clone+Ord+Default,{let mut forbid_1=0;let mut forbid_2=0;for&(v,_)in g[u].iter(){if v==p{continue;}dfs_label(v,u,g,forbid,decomp_tree,stacks);let forbid_by_v=forbid[v]+1;forbid_2|=forbid_1&forbid_by_v;forbid_1|=forbid_by_v;}let bit_length=(2*forbid_2+1).ilog2();forbid[u]=forbid_1|((1<<bit_length)-1);let label_u=(forbid[u]+1).trailing_zeros()as usize;stacks[label_u].push(u);for&(v,_)in g[u].iter().rev(){if v==p{continue;}extract_chain((forbid[v]+1)&((1<<label_u)-1),u,decomp_tree,stacks,);}}pub fn shallowest_decomposition_tree<V,T,const DIRECTED:bool>(root:usize,g:&Graph<V,T,DIRECTED>,)->Arborescence where V:Clone,T:Clone+Ord+Default,{let n=g.len();let lg=n.ilog2()as usize;let mut decomp_tree:VV<US> =vec![Vec::new();n];let mut stacks:VV<US> =vec![Vec::new();lg+1];let mut forbid:Vec<usize> =vec![0;n];dfs_label(root,root,g,&mut forbid,&mut decomp_tree,&mut stacks);let max_label=(forbid[root]+1).ilog2()as usize;let decomposition_root=stacks[max_label].pop().unwrap();extract_chain((forbid[root]+1)&((1<<max_label)-1),decomposition_root,&mut decomp_tree,&mut stacks,);Arborescence{root:decomposition_root,chilren:decomp_tree,}}#[doc=" BFS style break down of the decomposition."]pub fn bfs_list(tree:&Arborescence)->Vec<usize>{let mut bfs:Vec<usize> =Vec::new();let mut cur:VecDeque<usize> =VecDeque::new();cur.push_back(tree.root);while let Some(u)=cur.pop_front(){bfs.push(u);for&v in tree.chilren[u].iter(){cur.push_back(v);}}bfs}}use basic_graph::UndirectedGraph;pub mod utils{pub use crate::__cargo_equip::prelude::*;use std::{cmp::Ordering,collections::BTreeSet,fmt::Debug,io::{BufRead,BufReader,Stdin},ops::Bound::*,str::FromStr,};pub fn read_line_str_as_vec_template(line:&mut String,reader:&mut BufReader<Stdin>)->Vec<u8>{line.clear();reader.read_line(line).unwrap();line.trim().as_bytes().iter().cloned().collect()}pub fn read_line_binary_template(line:&mut String,reader:&mut BufReader<Stdin>)->Vec<u8>{line.clear();reader.read_line(line).unwrap();line.trim().as_bytes().iter().cloned().map(|x|x-b'0').collect()}pub fn read_vec_template<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->Vec<T>{line.clear();reader.read_line(line).unwrap();Vec::from_iter(line.split_whitespace().map(|x|match x.parse(){Ok(data)=>data,Err(_)=>default,}))}pub fn read_1_number<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->T{let v=read_vec_template(line,reader,default);v[0]}pub fn read_2_number<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(T,T){let v=read_vec_template(line,reader,default);(v[0],v[1])}pub fn read_3_number<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(T,T,T){let v=read_vec_template(line,reader,default);(v[0],v[1],v[2])}pub fn read_4_number<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(T,T,T,T){let v=read_vec_template(line,reader,default);(v[0],v[1],v[2],v[3])}pub fn read_vec_string_template(line:&mut String,reader:&mut BufReader<Stdin>)->Vec<String>{line.clear();reader.read_line(line).unwrap();line.split_whitespace().map(|x|x.to_string()).collect()}macro_rules!isOn{($S:expr,$b:expr)=>{($S&(1<<$b))>0};}macro_rules!turnOn{($S:ident,$b:expr)=>{$S|=(1<<$b)};($S:expr,$b:expr)=>{$S|(1<<$b)};}pub fn gcd(mut n:usize,mut m:usize)->usize{if n==0||m==0{return n+m;}while m!=0{if m<n{let t=m;m=n;n=t;}m=m%n;}n}pub fn true_distance_sq(a:(i64,i64),b:(i64,i64))->i64{let x=(a.0-b.0)*(a.0-b.0)+(a.1-b.1)*(a.1-b.1);return x;}pub fn lower_bound_pos<T:Ord+PartialOrd>(a:&Vec<T>,search_value:T)->usize{a.binary_search_by(|e|match e.cmp(&search_value){Ordering::Equal=>Ordering::Greater,ord=>ord,}).unwrap_err()}pub fn upper_bound_pos<T:Ord+PartialOrd>(a:&Vec<T>,search_value:T)->usize{a.binary_search_by(|e|match e.cmp(&search_value){Ordering::Equal=>Ordering::Less,ord=>ord,}).unwrap_err()}pub fn neighbors<T>(tree:&BTreeSet<T>,val:T)->(Option<&T>,Option<&T>)where T:Ord+Copy,{let mut before=tree.range((Unbounded,Excluded(val)));let mut after=tree.range((Excluded(val),Unbounded));(before.next_back(),after.next())}pub fn bin_search_template(l:usize,r:usize,f:&dyn Fn(usize)->bool)->usize{let mut l=l;let mut r=r;let mut ans=l;while l<=r{let mid=(l+r)/2;if f(mid){ans=mid;l=mid+1;}else{if mid==0{break;}r=mid-1;}}return ans;}pub fn ter_search_template(l:usize,r:usize,f:&dyn Fn(usize)->i32)->usize{let mut l=l;let mut r=r;while l<=r{if r-l<3{let mut ans=f(r);let mut pos=r;for i in l..r{if f(i)>ans{ans=f(i);pos=i;}}return pos;}let mid1=l+(r-l)/3;let mid2=r-(r-l)/3;let f1=f(mid1);let f2=f(mid2);if f1<f2{l=mid1;}else{r=mid2;}}return l;}pub fn two_pointer_template(a:&[i32],f:&dyn Fn(usize,usize)->bool){let mut l=0;let mut r=0;while l<a.len(){while r<a.len()&&f(l,r){r+=1;}if r==a.len(){break;}while!f(l,r){l+=1;}}}pub fn sliding_windows_d(s:&[u8],d:usize,f:&dyn Fn(usize)->usize){let mut start=0;let mut end=start+d-1;let mut contrib=0;(start..=end).for_each(|i|{contrib+=1;});while end+1<s.len(){contrib-=f(start);start+=1;end+=1;contrib+=f(end);}}pub fn next_permutation<T>(arr:&mut[T])->bool where T:std::cmp::Ord,{use std::cmp::Ordering;let last_ascending=match arr.windows(2).rposition(|w|w[0]<w[1]){Some(i)=>i,None=>{arr.reverse();return false;}};let swap_with=arr[last_ascending+1..].binary_search_by(|n|match arr[last_ascending].cmp(n){Ordering::Equal=>Ordering::Greater,ord=>ord,}).unwrap_err();arr.swap(last_ascending,last_ascending+swap_with);arr[last_ascending+1..].reverse();true}type V<T> =Vec<T>;type VV<T> =V<V<T>>;type Set<T> =BTreeSet<T>;type US=usize;type UU=(US,US);type UUU=(US,US,US);pub fn to_digit_array(a:u64)->V<US>{let mut ans:V<US> =V::new();let mut a=a;while a>0{ans.push((a%10)as usize);a/=10;}ans.reverse();ans}pub fn ceil_int(a:u64,b:u64)->u64{let mut r=a/b;if a%b!=0{r+=1;}r}pub fn sumxor(n:u64)->u64{let md=n%4;match md{0=>n,1=>1,2=>n+1,3=>0,_=>0,}}pub fn sumxor_range(l:u64,r:u64)->u64{if l==0{sumxor(r)}else{sumxor(l-1)^sumxor(r)}}pub fn mod_in_range(l:u64,r:u64,k:u64,m:u64)->(u64,u64,u64){let first_oc=if l<=k{k}else{k+(ceil_int(l-k,m))*m};if first_oc>r{return(0,0,0);}let last_oc=first_oc+((r-first_oc)/m)*m;let oc=((last_oc-k)-(first_oc-k))/m+1;(oc,first_oc,last_oc)}#[doc=" https://www.geeksforgeeks.org/sum-of-products-of-all-possible-k-size-subsets-of-the-given-array/"]#[doc=" In O(n*k)"]pub fn sum_of_product(arr:&Vec<US>,k:usize)->US{let n=arr.len();let mut dp:Vec<US> =vec![0;n+1];let mut cur_sum=0;for i in 1..=n{dp[i]=arr[i-1];cur_sum+=arr[i-1];}for _ in 2..=k{let mut temp_sum=0;for j in 1..=n{cur_sum-=dp[j];dp[j]=arr[j-1]*cur_sum;temp_sum+=dp[j];}cur_sum=temp_sum;}cur_sum}pub fn better_array_debug<T>(a:&V<T>)where T:Debug,{a.iter().enumerate().for_each(|(i,x)|println!("{i:4}: {x:?}"));}pub fn better_2_array_debug<T,D>(a:&V<T>,b:&V<D>)where T:Debug,D:Debug,{(0..a.len()).for_each(|i|{println!("{i:4}: {:?} -- {:4?}",a[i],b[i]);})}pub fn read_n_and_array<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(US,V<T>){let n=read_1_number(line,reader,0usize);let v=read_vec_template(line,reader,default);(n,v)}pub fn read_n_m_and_array<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(US,US,V<T>){let(n,m)=read_2_number(line,reader,0usize);let v=read_vec_template(line,reader,default);(n,m,v)}pub fn read_n_and_array_of_pair<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(US,V<(T,T)>){let n=read_1_number(line,reader,0usize);let mut v:V<(T,T)> =V::new();v.reserve(n);(0..n).for_each(|_|{let x=read_2_number(line,reader,default);v.push(x);});(n,v)}pub fn read_n_and_2_array<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(T,V<T>,V<T>){let n=read_1_number(line,reader,default);let a=read_vec_template(line,reader,default);let b=read_vec_template(line,reader,default);(n,a,b)}pub fn read_n_m_and_2_array<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(T,T,V<T>,V<T>){let(n,m)=read_2_number(line,reader,default);let a=read_vec_template(line,reader,default);let b=read_vec_template(line,reader,default);(n,m,a,b)}pub fn read_n_m_k_and_2_array<T:FromStr+Copy>(line:&mut String,reader:&mut BufReader<Stdin>,default:T,)->(T,T,T,V<T>,V<T>){let(n,m,k)=read_3_number(line,reader,default);let a=read_vec_template(line,reader,default);let b=read_vec_template(line,reader,default);(n,m,k,a,b)}pub fn read_edge_list(m:usize,line:&mut String,reader:&mut BufReader<Stdin>)->V<(UU)>{let mut el:V<UU> =V::new();el.reserve(m);(0..m).for_each(|_|{let(u,v)=read_2_number(line,reader,0usize);el.push((u-1,v-1));});el}pub fn query(l:u64,r:u64,re:&mut BufReader<Stdin>,li:&mut String)->u64{println!("? {l} {r}");let ans=read_1_number(li,re,0u64);ans}}type VV<T> =Vec<Vec<T>>;type US=usize;fn solve(reader:&mut BufReader<Stdin>,line:&mut String,out:&mut BufWriter<StdoutLock>){let default=0usize;let n=utils::read_1_number(line,reader,default);let uelist=utils::read_edge_list(n-1,line,reader);let s=utils::read_line_str_as_vec_template(line,reader);let a:Vec<usize> =s.iter().map(|&c|1<<(c-b'a')).collect();let g=UndirectedGraph::from_unweighted_edges(n,&uelist);let tree=tree_decomp::shallowest_decomposition_tree(0,&g);let bfs_list=tree_decomp::bfs_list(&tree);let mut palindrome:Vec<usize> =(0..21).map(|i|1<<i).collect();palindrome[20]=0;let mut parent=vec![usize::MAX;n];let mut bitmask=vec![0;n];let mut bitmask_count=vec![0u64;1<<20];let mut ans=vec![1u64;n];let mut dp=vec![0u64;n];let mut d_graph:Vec<Vec<usize>> =vec![Vec::new();n];for u in 0..n{for&(v,_)in g[u].iter(){d_graph[u].push(v);}}for&decomp_node in bfs_list.iter(){parent[decomp_node]=decomp_node;for v in d_graph[decomp_node].clone().into_iter(){if let Some(idx)=d_graph[v].iter().position(|&x|x==decomp_node){d_graph[v].swap_remove(idx);}}let mut all_bfss:VV<US> =Vec::new();let mut all_bfss_flat:Vec<US> =vec![decomp_node];for&child in d_graph[decomp_node].iter(){parent[child]=decomp_node;let mut bfs_at_child=Vec::<US>::new();let mut cur:VecDeque<US> =VecDeque::new();cur.push_back(child);while let Some(u)=cur.pop_front(){bfs_at_child.push(u);for&v in d_graph[u].iter(){if parent[v]==usize::MAX{parent[v]=u;cur.push_back(v);}}}all_bfss_flat.extend(bfs_at_child.clone().into_iter());all_bfss.push(bfs_at_child);}for&u in all_bfss_flat.iter(){bitmask[u]=bitmask[parent[u]]^a[u];bitmask_count[bitmask[u]]+=1;}for bfs in all_bfss.iter(){for&u in bfs.iter(){bitmask_count[bitmask[u]]-=1;}for&u in bfs.iter().rev(){let base=bitmask[u]^a[decomp_node];for&p in palindrome.iter(){dp[u]+=bitmask_count[base^p];}dp[parent[u]]+=dp[u];}for&u in bfs.iter(){bitmask_count[bitmask[u]]+=1;}}bitmask_count[a[decomp_node]]-=1;for&p in palindrome.iter(){dp[decomp_node]+=bitmask_count[p];}dp[decomp_node]/=2;for&u in all_bfss_flat.iter(){ans[u]+=dp[u];dp[u]=0;parent[u]=usize::MAX;bitmask_count[bitmask[u]]=0;bitmask[u]=0;}}for&x in ans.iter(){write!(out,"{x} ").unwrap();}writeln!(out).unwrap();}fn main(){let mut reader=BufReader::new(stdin());let mut line=String::new();let mut out=BufWriter::new(stdout().lock());solve(&mut reader,&mut line,&mut out);}#[doc="  # Bundled libraries"]#[doc=" "]#[doc="  - `path+file:///home/quocbaodo205/Documents/cp_rust#0.1.0` published in **missing** licensed under **missing** as `crate::__cargo_equip::crates::cp_rust`"]#[allow(unused)]mod __cargo_equip{pub(crate)mod crates{pub mod cp_rust{}}pub(crate)mod macros{pub mod cp_rust{}}pub(crate)mod prelude{pub use crate::__cargo_equip::crates::*;}mod preludes{pub mod cp_rust{}}}
