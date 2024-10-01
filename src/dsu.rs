use std::mem::swap;

#[allow(dead_code)]
struct DSU {
    n: usize,
    parent: Vec<usize>,
    size: Vec<usize>,
}

#[allow(dead_code)]
impl DSU {
    pub fn new(sz: usize) -> Self {
        DSU {
            n: sz,
            parent: (0..sz).collect(),
            size: vec![0; sz],
        }
    }

    pub fn find_parent(&mut self, u: usize) -> usize {
        if self.parent[u] == u {
            return u;
        }
        self.parent[u] = self.find_parent(self.parent[u]);
        return self.parent[u];
    }

    pub fn union(&mut self, u: usize, v: usize) {
        let mut pu = self.find_parent(u);
        let mut pv = self.find_parent(v);
        if pu == pv {
            return;
        }
        if self.size[pu] > self.size[pv] {
            swap(&mut pu, &mut pv);
        }
        self.size[pu] += self.size[pv];
        self.parent[pv] = pu;
    }

    pub fn count_set(&mut self) -> usize {
        (0..self.n).filter(|&u| self.find_parent(u) == u).count()
    }
}
