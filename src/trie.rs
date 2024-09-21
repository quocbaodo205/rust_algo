// Trie implementation
// Mutable, clonable

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
