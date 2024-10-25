// Trie implementation
// Mutable, clonable

use std::mem;
use std::rc::Rc;

// Using Box, when transfer ownership, won't copy
type BinTrieLink = [Option<Box<BinTrieNode>>; 2];

#[derive(Clone)]
pub struct BinTrieNode {
    val: i32,
    children: BinTrieLink,
}

impl BinTrieNode {
    fn new(val: i32) -> Self {
        BinTrieNode {
            val,
            children: [None, None],
        }
    }

    fn insert(&mut self, pos: usize, val: i32) {
        if self.children[pos].is_none() {
            // Create on stack
            let node = BinTrieNode::new(val);
            // Copy to heap
            self.children[pos] = Some(Box::new(node));
        }
    }
}

pub struct BinTrie {
    head: BinTrieNode,
}

impl BinTrie {
    pub fn new() -> Self {
        BinTrie {
            head: BinTrieNode::new(0),
        }
    }

    pub fn insert(&mut self, tree: u32, val: i32) {
        // Borrowed head, not move. Head is still valid on its own
        let mut k = &mut self.head;

        for bit in (0..32).rev() {
            let pos = if tree & (1 << bit) > 0 { 1 } else { 0 };
            k.insert(pos, 0);
            k = k.children[pos].as_mut().unwrap();
        }

        // Finally, set value
        k.val = val;
    }

    pub fn get(&mut self, tree: u32) -> i32 {
        let mut k = &mut self.head;

        for bit in (0..32).rev() {
            let pos = if tree & (1 << bit) > 0 { 1 } else { 0 };
            if k.children[pos].as_ref().is_none() {
                return 0;
            }
            // Borrow k.children[pos] mutable, child is still valid
            k = k.children[pos].as_mut().unwrap();
        }

        // Finally, set value
        k.val
    }
}

impl Default for BinTrie {
    fn default() -> Self {
        BinTrie::new()
    }
}

// Character Trie
type TrieLink = Option<Box<TrieNode>>;

struct TrieNode {
    val: i32,
    children: [TrieLink; 26],
}
const ARRAY_REPEAT_VALUE: Option<Box<TrieNode>> = None;

impl TrieNode {
    pub fn new(val: i32) -> Self {
        TrieNode {
            val,
            children: [ARRAY_REPEAT_VALUE; 26],
        }
    }

    pub fn add(&mut self, idx: usize) {
        if self.children[idx].is_none() {
            self.children[idx] = Some(Box::new(TrieNode::new(0)));
        }
    }
}

pub struct Trie {
    head: TrieNode,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            head: TrieNode::new(0),
        }
    }

    pub fn insert(&mut self, st: &[u8]) {
        let mut cur = &mut self.head;
        for c in st {
            let idx = (*c - b'a') as usize;
            cur.add(idx);
            cur = cur.children[idx].as_mut().unwrap().as_mut();
        }
        cur.val += 1;
    }

    pub fn get(&mut self, st: &[u8]) -> i32 {
        let mut cur = &mut self.head;
        for c in st {
            let idx = (*c - b'a') as usize;
            if cur.children[idx].is_none() {
                return 0;
            }
            cur = cur.children[idx].as_mut().unwrap();
        }
        cur.val
    }
}

impl Default for Trie {
    fn default() -> Self {
        Trie::new()
    }
}

type PersistentTrieLink = Option<Rc<PersistentTrieNode>>;
const ARRAY_REPEAT_VALUE_P: Option<Rc<PersistentTrieNode>> = None;

#[allow(dead_code)]
struct PersistentTrieNode {
    val: i32,
    children: [PersistentTrieLink; 26],
}

#[allow(dead_code)]
impl PersistentTrieNode {
    pub fn new(val: i32) -> Self {
        PersistentTrieNode {
            val,
            children: [ARRAY_REPEAT_VALUE_P; 26],
        }
    }
}

#[allow(dead_code)]
struct PersistentTrie {
    history: Vec<PersistentTrieNode>,
}

#[allow(dead_code)]
impl PersistentTrie {
    pub fn new() -> Self {
        PersistentTrie {
            history: vec![PersistentTrieNode::new(0)],
        }
    }

    pub fn insert_f(st: &[u8], idx: usize) -> PersistentTrieNode {
        let mut node = PersistentTrieNode::new(0);
        match idx == st.len() {
            true => {
                node.val += 1;
                node
            }
            false => {
                let c = (st[idx] - b'a') as usize;
                node.children[c] = Some(Rc::new(PersistentTrie::insert_f(st, idx + 1)));
                node
            }
        }
    }

    pub fn insert_r(st: &[u8], idx: usize, old_head: &PersistentTrieNode) -> PersistentTrieNode {
        // Clone every children
        let mut node = PersistentTrieNode::new(old_head.val);

        for (c, v) in old_head.children.iter().enumerate() {
            if let Some(n) = v.as_ref() {
                node.children[c] = Some(Rc::clone(n));
            }
        }

        match idx == st.len() {
            true => {
                node.val += 1;
                node
            }
            false => {
                //
                let c = (st[idx] - b'a') as usize;
                // Nothing at old head, add everything without having to clone
                match &old_head.children[c] {
                    Some(t) => {
                        node.children[c] =
                            Some(Rc::new(PersistentTrie::insert_r(st, idx + 1, t.as_ref())));
                    }
                    None => {
                        node.children[c] = Some(Rc::new(PersistentTrie::insert_f(st, idx + 1)));
                    }
                }
                node
            }
        }
    }

    pub fn insert(&mut self, st: &[u8]) {
        let old_head = self.history.last().unwrap();
        self.history.push(PersistentTrie::insert_r(st, 0, old_head));
    }

    pub fn get_r(st: &[u8], idx: usize, node: &PersistentTrieNode) -> i32 {
        match idx == st.len() {
            true => node.val,
            false => {
                let c = (st[idx] - b'a') as usize;
                match &node.children[c] {
                    Some(t) => PersistentTrie::get_r(st, idx + 1, t.as_ref()),
                    None => 0,
                }
            }
        }
    }

    pub fn get(&self, st: &[u8]) -> i32 {
        match self.history.last() {
            Some(t) => PersistentTrie::get_r(st, 0, t),
            None => 0,
        }
    }

    pub fn undo(&mut self) {
        self.history.pop();
    }
}

impl Default for PersistentTrie {
    fn default() -> Self {
        PersistentTrie::new()
    }
}

// Compressed trie to use when string is just too long.
// Label are just reference so it's even more efficient.
type CTrieLink<'a> = Option<Box<CTrieNode<'a>>>;

struct CTrieNode<'a> {
    val: i32,
    children: [CTrieLink<'a>; 26],
    label: &'a [u8],
}

const CREPEAT: Option<Box<CTrieNode>> = None;

impl<'a> CTrieNode<'a> {
    pub fn new(val: i32, label: &'a [u8]) -> Self {
        CTrieNode {
            val,
            children: [CREPEAT; 26],
            label,
        }
    }
}

pub struct CTrie<'a> {
    head: CTrieNode<'a>,
}

fn prefix_pos(label: &[u8], st: &[u8]) -> usize {
    let mut pos = st.len();
    for (i, &c) in st.iter().enumerate() {
        if label.len() <= i {
            pos = i;
            break;
        }
        if c as u8 != label[i] {
            pos = i;
            break;
        }
    }
    pos
}

impl<'a> CTrie<'a> {
    pub fn new(x: &'a [u8]) -> Self {
        CTrie {
            head: CTrieNode::new(0, x),
        }
    }

    pub fn insert(&mut self, st: &'a [u8]) {
        let mut cur = &mut self.head;
        let mut rst = st;

        loop {
            let pos = prefix_pos(&cur.label, rst);
            println!(
                "insert label = {:?}, rst = {:?}, pos = {pos}",
                cur.label, rst
            );
            // Case 1: rst is a prefix of label
            if pos == rst.len() {
                println!("insert case 1, cur.val = {}", cur.val);
                if cur.label.len() == rst.len() {
                    cur.val += 1;
                    break;
                }

                let c_old = (cur.label[pos] - b'a') as usize;
                let old_child_label = &cur.label[pos..];

                // When break, combine all other children...
                // Dance!
                let mut new_node = Box::new(CTrieNode::new(cur.val, old_child_label));
                (0..26).for_each(
                    |c_idx| match mem::replace(&mut cur.children[c_idx], CREPEAT) {
                        Some(nd) => {
                            new_node.children[c_idx] = Some(nd);
                        }
                        None => {}
                    },
                );
                cur.children[c_old] = Some(new_node);
                cur.label = &cur.label[..pos];
                cur.val += 1;
                break;
            }
            // Case 2: same prefix < label -> break up the label and add 2 childrens
            if cur.label.len() > pos {
                println!("insert case 2, cur.val = {}", cur.val);
                let c_old = (cur.label[pos] - b'a') as usize;
                let c_new = (rst[pos] - b'a') as usize;

                let old_label = &cur.label[..pos];
                let old_child_label = &cur.label[pos..];
                let new_child_label = &rst[pos..];

                let mut new_node = Box::new(CTrieNode::new(cur.val, old_child_label));
                (0..26).for_each(
                    |c_idx| match mem::replace(&mut cur.children[c_idx], CREPEAT) {
                        Some(nd) => {
                            new_node.children[c_idx] = Some(nd);
                        }
                        None => {}
                    },
                );
                cur.children[c_old] = Some(new_node);

                // New stuff always 1
                cur.children[c_new] = Some(Box::new(CTrieNode::new(1, new_child_label)));

                cur.val += 1; // count # prefix
                cur.label = old_label;
                break;
            } else if cur.label.len() <= pos {
                // Case 3: same prefix > label -> Keep comparing with chilren with a reduced rst
                println!("insert case 3, cur.val = {}", cur.val);
                let c_new = (rst[pos] - b'a') as usize;
                if cur.children[c_new].is_none() {
                    // Add the whole (new so it's 1)
                    cur.val += 1;
                    cur.children[c_new] = Some(Box::new(CTrieNode::new(1, &rst[pos..])));
                    break;
                } else {
                    cur.val += 1;
                    cur = cur.children[c_new].as_mut().unwrap();
                    rst = &rst[pos..];
                }
            }
        }
        println!("Done....................................");
    }

    pub fn get(&mut self, st: &[u8]) -> i32 {
        let mut cur = &mut self.head;
        let mut rst = st;

        loop {
            let pos = prefix_pos(&cur.label, rst);
            // Case 1: rst is a prefix of label
            if pos == rst.len() {
                // println!("get case 1");
                return cur.val;
            }
            // Case 2: same prefix < label -> Cannot be found
            if cur.label.len() > pos {
                return 0;
            } else if cur.label.len() <= pos {
                // Case 3: same prefix > label -> Keep comparing with chilren with a reduced rst
                let c_new = (rst[pos] - b'a') as usize;
                if cur.children[c_new].is_none() {
                    return 0;
                } else {
                    cur = cur.children[c_new].as_mut().unwrap();
                    rst = &rst[pos..];
                }
            }
        }
    }

    // Emulate 1 by 1 child move for more intuitive search
    pub fn get_1b1(&mut self, st: &[u8]) -> i32 {
        let mut cur = &mut self.head;
        let mut j = 0;
        for &c in st.iter() {
            let idx = (c - b'a') as usize;
            if j < cur.label.len() {
                if cur.label[j] != c {
                    // Equivilant to no children
                    return 0;
                }
                j += 1;
            } else {
                if cur.children[idx].is_none() {
                    return 0;
                }
                cur = cur.children[idx].as_mut().unwrap();
                // First char is guarantee match alr, j = 1
                j = 1;
            }
        }
        cur.val
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bin_trie_test() {
        let mut t = BinTrie::new();

        t.insert(0b1101, 10);
        assert_eq!(t.get(0b1101), 10);
        // Check get failed
        assert_eq!(t.get(0b1100), 0);
        assert_eq!(t.get(0b0101), 0);
        assert_eq!(t.get(0b1001), 0);
        assert_eq!(t.get(0b1111), 0);

        t.insert(0b11101, 11);
        assert_eq!(t.get(0b1101), 10);
        assert_eq!(t.get(0b1100), 0);
        assert_eq!(t.get(0b11101), 11);
        assert_eq!(t.get(0b11100), 0);
    }

    #[test]
    fn ctrie_test() {
        let emp: Vec<u8> = Vec::new();
        let mut t = CTrie::new(&emp);
        t.insert("abc".as_bytes());
        assert_eq!(t.get("abc".as_bytes()), 1);
        assert_eq!(t.get_1b1("abc".as_bytes()), 1);
        t.insert("abc".as_bytes());
        assert_eq!(t.get("abc".as_bytes()), 2);
        assert_eq!(t.get_1b1("abc".as_bytes()), 2);
        t.insert("abce".as_bytes());
        assert_eq!(t.get("abcd".as_bytes()), 0);
        assert_eq!(t.get("abce".as_bytes()), 1);
        assert_eq!(t.get("abc".as_bytes()), 3);
        assert_eq!(t.get("ab".as_bytes()), 3);
        assert_eq!(t.get_1b1("abcd".as_bytes()), 0);
        assert_eq!(t.get_1b1("abce".as_bytes()), 1);
        assert_eq!(t.get_1b1("abc".as_bytes()), 3);
        assert_eq!(t.get_1b1("ab".as_bytes()), 3);

        t.insert("zzz".as_bytes());
        assert_eq!(t.get("z".as_bytes()), 1);
        assert_eq!(t.get("zz".as_bytes()), 1);
        assert_eq!(t.get("zzz".as_bytes()), 1);
        assert_eq!(t.get_1b1("z".as_bytes()), 1);
        assert_eq!(t.get_1b1("zz".as_bytes()), 1);
        assert_eq!(t.get_1b1("zzz".as_bytes()), 1);
    }

    #[test]
    fn ctrie_test_2() {
        let emp: Vec<u8> = Vec::new();
        let mut t = CTrie::new(&emp);
        t.insert("aba".as_bytes());
        assert_eq!(t.get("aba".as_bytes()), 1);
        assert_eq!(t.get_1b1("aba".as_bytes()), 1);
        t.insert("ab".as_bytes());
        assert_eq!(t.get("ab".as_bytes()), 2);
        assert_eq!(t.get_1b1("ab".as_bytes()), 2);
        assert_eq!(t.get("aba".as_bytes()), 1);
        assert_eq!(t.get_1b1("aba".as_bytes()), 1);

        t.insert("aaaaaaa".as_bytes());
        t.insert("aaaaaa".as_bytes());
        t.insert("aaaaa".as_bytes());
        assert_eq!(t.get_1b1("aaaaaaa".as_bytes()), 1);
        assert_eq!(t.get_1b1("aaaaaa".as_bytes()), 2);
        assert_eq!(t.get_1b1("aaaaa".as_bytes()), 3);
        assert_eq!(t.get_1b1("a".as_bytes()), 5);
    }

    #[test]
    fn trie_test() {
        let mut t = Trie::new();
        t.insert("abc".as_bytes());
        assert_eq!(t.get("abc".as_bytes()), 1);
        t.insert("abc".as_bytes());
        assert_eq!(t.get("abc".as_bytes()), 2);
        t.insert("abce".as_bytes());
        assert_eq!(t.get("abcd".as_bytes()), 0);
        assert_eq!(t.get("abce".as_bytes()), 1);

        t.insert("zzz".as_bytes());
        assert_eq!(t.get("z".as_bytes()), 0);
        assert_eq!(t.get("zz".as_bytes()), 0);
        assert_eq!(t.get("zzz".as_bytes()), 1);
    }

    #[test]
    fn persistent_trie_test() {
        let mut t = PersistentTrie::new();
        t.insert("abc".as_bytes());
        assert_eq!(t.get("abc".as_bytes()), 1);
        t.insert("abc".as_bytes());
        assert_eq!(t.get("abc".as_bytes()), 2);
        t.insert("abce".as_bytes());
        assert_eq!(t.get("abcd".as_bytes()), 0);
        assert_eq!(t.get("abce".as_bytes()), 1);

        t.insert("zzz".as_bytes());
        assert_eq!(t.get("z".as_bytes()), 0);
        assert_eq!(t.get("zz".as_bytes()), 0);
        assert_eq!(t.get("zzz".as_bytes()), 1);

        t.insert("ab".as_bytes());
        assert_eq!(t.get("abc".as_bytes()), 2);
        assert_eq!(t.get("abcd".as_bytes()), 0);
        assert_eq!(t.get("abce".as_bytes()), 1);
        assert_eq!(t.get("z".as_bytes()), 0);
        assert_eq!(t.get("zz".as_bytes()), 0);
        assert_eq!(t.get("zzz".as_bytes()), 1);
        assert_eq!(t.get("ab".as_bytes()), 1);

        t.undo();
        assert_eq!(t.get("abc".as_bytes()), 2);
        assert_eq!(t.get("abcd".as_bytes()), 0);
        assert_eq!(t.get("abce".as_bytes()), 1);
        assert_eq!(t.get("z".as_bytes()), 0);
        assert_eq!(t.get("zz".as_bytes()), 0);
        assert_eq!(t.get("zzz".as_bytes()), 1);
        assert_eq!(t.get("ab".as_bytes()), 0);
        t.undo();
        assert_eq!(t.get("abc".as_bytes()), 2);
        assert_eq!(t.get("abcd".as_bytes()), 0);
        assert_eq!(t.get("abce".as_bytes()), 1);
        assert_eq!(t.get("ab".as_bytes()), 0);
        assert_eq!(t.get("zzz".as_bytes()), 0);
    }
}
