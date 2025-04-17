pub trait PersistentMonoid {
    type M: Clone;
    fn id() -> Self::M;
    fn op(l: &Self::M, r: &Self::M) -> Self::M;
}

use std::rc::Rc;

struct Node<T: PersistentMonoid> {
    data: T::M,
    left: Option<Rc<Node<T>>>,
    right: Option<Rc<Node<T>>>,
}

impl<T: PersistentMonoid> Node<T> {
    fn new(data: T::M) -> Self {
        Node {
            data,
            left: None,
            right: None,
        }
    }
    fn build(l: usize, r: usize) -> Self {
        if l + 1 >= r {
            Node::new(T::id())
        } else {
            Node {
                data: T::id(),
                left: Some(Rc::new(Node::build(l, (l + r) >> 1))),
                right: Some(Rc::new(Node::build((l + r) >> 1, r))),
            }
        }
    }
    fn set(&self, i: usize, x: T::M, l: usize, r: usize) -> Self {
        assert!(l <= i && i < r);
        if i == l && i + 1 == r {
            Node::new(x)
        } else if l <= i && i < ((l + r) >> 1) {
            let left = Some(Rc::new(self.left.as_ref().unwrap().set(
                i,
                x,
                l,
                (l + r) >> 1,
            )));
            let right = self.right.clone();
            Node {
                data: T::op(
                    &match left.as_ref() {
                        Some(n) => n.data.clone(),
                        None => T::id(),
                    },
                    &match right.as_ref() {
                        Some(n) => n.data.clone(),
                        None => T::id(),
                    },
                ),
                left,
                right,
            }
        } else {
            let left = self.left.clone();
            let right = Some(Rc::new(self.right.as_ref().unwrap().set(
                i,
                x,
                (l + r) >> 1,
                r,
            )));
            Node {
                data: T::op(
                    &match left.as_ref() {
                        Some(n) => n.data.clone(),
                        None => T::id(),
                    },
                    &match right.as_ref() {
                        Some(n) => n.data.clone(),
                        None => T::id(),
                    },
                ),
                left,
                right,
            }
        }
    }
    fn fold(&self, a: usize, b: usize, l: usize, r: usize) -> T::M {
        if a <= l && r <= b {
            self.data.clone()
        } else if r <= a || b <= l {
            T::id()
        } else {
            T::op(
                &match self.left.as_ref() {
                    Some(n) => n.fold(a, b, l, (l + r) >> 1),
                    None => T::id(),
                },
                &match self.right.as_ref() {
                    Some(n) => n.fold(a, b, (l + r) >> 1, r),
                    None => T::id(),
                },
            )
        }
    }
}

impl<T: PersistentMonoid> Drop for Node<T> {
    fn drop(&mut self) {
        if let Some(left) = self.left.take() {
            if let Ok(_) = Rc::try_unwrap(left) {}
        }
        if let Some(right) = self.right.take() {
            if let Ok(_) = Rc::try_unwrap(right) {}
        }
    }
}

pub struct PersistentSegmentTree<T: PersistentMonoid> {
    root: Node<T>,
    sz: usize,
}

// let seg = PersistentSegmentTree::<I32Sum>::new(3);
// let seg2 = seg.set(0, 3);
// assert_eq!(seg2.fold(0, 2), 3);
impl<T: PersistentMonoid> PersistentSegmentTree<T> {
    pub fn new(n: usize) -> Self {
        Self {
            root: Node::build(0, n),
            sz: n,
        }
    }
    pub fn update(&self, i: usize, x: T::M) -> Self {
        Self {
            root: self.root.set(i, x, 0, self.sz),
            sz: self.sz,
        }
    }
    pub fn fold(&self, l: usize, r: usize) -> T::M {
        self.root.fold(l, r, 0, self.sz)
    }
}
