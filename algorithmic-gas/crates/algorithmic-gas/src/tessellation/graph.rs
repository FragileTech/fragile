//! Symmetric walker neighbor graph in CSR form. Edge slots are ordered by
//! (source, destination), so per-edge arrays have one canonical layout and
//! per-node sums always run in the same order.
use crate::{GasError, Result, error::require};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeighborGraph {
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
    /// Slot of the edge (j, i) for the slot holding (i, j).
    reverse: Vec<u32>,
}
impl NeighborGraph {
    pub fn empty(nodes: usize) -> Self {
        Self {
            offsets: vec![0; nodes + 1],
            neighbors: Vec::new(),
            reverse: Vec::new(),
        }
    }
    /// Build from undirected pairs. Self loops are dropped and duplicates merged.
    pub fn from_undirected(nodes: usize, pairs: &[[u32; 2]]) -> Result<Self> {
        require(nodes <= i32::MAX as usize, "neighbor graph node count")?;
        let mut directed = Vec::with_capacity(pairs.len() * 2);
        for &[a, b] in pairs {
            require(
                (a as usize) < nodes && (b as usize) < nodes,
                "neighbor graph edge endpoint out of range",
            )?;
            if a != b {
                directed.push((a, b));
                directed.push((b, a));
            }
        }
        directed.sort_unstable();
        directed.dedup();
        if directed.len() > i32::MAX as usize {
            return Err(GasError::Capability(
                "neighbor graph exceeds the edge index range".into(),
            ));
        }
        let mut offsets = vec![0u32; nodes + 1];
        for &(a, _) in &directed {
            offsets[a as usize + 1] += 1;
        }
        for i in 0..nodes {
            offsets[i + 1] += offsets[i];
        }
        let neighbors: Vec<u32> = directed.iter().map(|e| e.1).collect();
        let mut graph = Self {
            offsets,
            neighbors,
            reverse: Vec::new(),
        };
        graph.reverse = directed
            .iter()
            .map(|&(a, b)| {
                graph
                    .slot(b as usize, a)
                    .expect("symmetric edge list contains every reverse edge")
                    as u32
            })
            .collect();
        Ok(graph)
    }
    pub fn nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    /// Number of directed edge slots (twice the undirected edge count).
    pub fn edges(&self) -> usize {
        self.neighbors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }
    pub fn neighbors(&self) -> &[u32] {
        &self.neighbors
    }
    pub fn reverse(&self) -> &[u32] {
        &self.reverse
    }
    pub fn range(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i] as usize..self.offsets[i + 1] as usize
    }
    pub fn row(&self, i: usize) -> &[u32] {
        &self.neighbors[self.range(i)]
    }
    pub fn degree(&self, i: usize) -> usize {
        self.range(i).len()
    }
    pub fn slot(&self, i: usize, j: u32) -> Option<usize> {
        let range = self.range(i);
        self.neighbors[range.clone()]
            .binary_search(&j)
            .ok()
            .map(|k| range.start + k)
    }
    /// Source node of every edge slot.
    pub fn sources(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.edges());
        for i in 0..self.nodes() {
            out.extend(std::iter::repeat_n(i as u32, self.degree(i)));
        }
        out
    }
    /// Directed [E, 2] edge list in slot order.
    pub fn coo(&self) -> Vec<[u32; 2]> {
        self.sources()
            .into_iter()
            .zip(&self.neighbors)
            .map(|(a, &b)| [a, b])
            .collect()
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.nodes();
        require(
            self.offsets.first() == Some(&0)
                && self.offsets.last().copied() == Some(self.neighbors.len() as u32)
                && self.offsets.windows(2).all(|w| w[0] <= w[1])
                && self.reverse.len() == self.neighbors.len(),
            "neighbor graph CSR layout",
        )?;
        for i in 0..n {
            let row = self.row(i);
            require(
                row.windows(2).all(|w| w[0] < w[1])
                    && row.iter().all(|&j| (j as usize) < n && j as usize != i),
                "neighbor graph rows must be sorted, loop-free and in range",
            )?;
            for e in self.range(i) {
                let back = self.reverse[e] as usize;
                require(
                    back < self.neighbors.len()
                        && self.neighbors[back] as usize == i
                        && self.range(self.neighbors[e] as usize).contains(&back),
                    "neighbor graph reverse index",
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csr_is_symmetric_sorted_and_reversible() {
        let g =
            NeighborGraph::from_undirected(5, &[[3, 1], [1, 3], [0, 1], [2, 2], [4, 0], [1, 4]])
                .unwrap();
        g.validate().unwrap();
        assert_eq!(g.edges(), 8);
        assert_eq!(g.row(1), &[0, 3, 4]);
        assert_eq!(g.degree(2), 0);
        assert_eq!(g.coo()[0], [0, 1]);
        for (e, [a, b]) in g.coo().into_iter().enumerate() {
            assert_eq!(g.coo()[g.reverse()[e] as usize], [b, a]);
        }
        assert!(NeighborGraph::from_undirected(2, &[[0, 2]]).is_err());
        NeighborGraph::empty(3).validate().unwrap();
    }
}
