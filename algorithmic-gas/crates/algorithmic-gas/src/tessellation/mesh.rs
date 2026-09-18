//! Site-level simplicial mesh and the tessellator role. A tessellator receives
//! distinct sites in general affine position (full rank); duplicates and
//! rank-deficient swarms are resolved before it runs.
use crate::{GasError, Result, error::require};
use serde::{Deserialize, Serialize};

/// Delaunay complex of `sites` points in `dimension` coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SiteMesh {
    pub dimension: usize,
    /// Coordinates that were triangulated, `sites * dimension`.
    pub coords: Vec<f64>,
    /// `dimension + 1` site indices per simplex.
    pub simplices: Vec<u32>,
    /// Undirected edges `[a, b]` with `a < b`, sorted.
    pub edges: Vec<[u32; 2]>,
    /// Sites on the convex hull (their Voronoi cells are unbounded).
    pub hull: Vec<bool>,
    /// Leading real sites; the remaining sites are images added by the domain.
    pub real: usize,
    /// Real site every site is an image of (the identity on real sites).
    pub base: Vec<u32>,
}
impl SiteMesh {
    pub fn sites(&self) -> usize {
        self.hull.len()
    }
    pub fn site(&self, s: usize) -> &[f64] {
        &self.coords[s * self.dimension..(s + 1) * self.dimension]
    }
    pub fn simplex_count(&self) -> usize {
        self.simplices.len() / (self.dimension + 1)
    }
    /// A mesh without simplices over `n` real sites.
    pub fn disconnected(dimension: usize, coords: Vec<f64>, n: usize) -> Self {
        Self {
            dimension,
            coords,
            hull: vec![true; n],
            real: n,
            base: (0..n as u32).collect(),
            ..Self::default()
        }
    }
    pub fn simplex(&self, k: usize) -> &[u32] {
        let w = self.dimension + 1;
        &self.simplices[k * w..(k + 1) * w]
    }
    /// Edges of every simplex, sorted and deduplicated.
    pub fn edges_from_simplices(simplices: &[u32], width: usize) -> Vec<[u32; 2]> {
        let mut edges = Vec::with_capacity(simplices.len() / width * width * (width - 1) / 2);
        for s in simplices.chunks_exact(width) {
            for a in 0..width {
                for b in a + 1..width {
                    edges.push([s[a].min(s[b]), s[a].max(s[b])]);
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        edges
    }
    /// Sites ordered along a line: consecutive sites are neighbors.
    pub fn path(coords: Vec<f64>, dimension: usize, order: &[u32]) -> Self {
        let n = order.len();
        let mut edges: Vec<[u32; 2]> = order
            .windows(2)
            .map(|w| [w[0].min(w[1]), w[0].max(w[1])])
            .collect();
        edges.sort_unstable();
        let mut hull = vec![false; n];
        if let (Some(&a), Some(&b)) = (order.first(), order.last()) {
            hull[a as usize] = true;
            hull[b as usize] = true;
        }
        Self {
            dimension,
            coords,
            simplices: order.windows(2).flatten().copied().collect(),
            edges,
            hull,
            real: n,
            base: (0..n as u32).collect(),
        }
    }
}

pub trait Tessellator {
    /// `coords` holds `n` distinct, affinely spanning sites of `dimension` coordinates.
    fn tessellate(&self, coords: &[f64], n: usize, dimension: usize) -> Result<SiteMesh>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TessellatorKind {
    /// Pick by dimension: sorted path (1), Spade (2), Bowyer-Watson (3).
    #[default]
    Auto,
    Path1,
    Spade2,
    BowyerWatson3,
}
impl TessellatorKind {
    pub fn supports(self, dimension: usize) -> bool {
        match self {
            Self::Auto => (1..=3).contains(&dimension),
            Self::Path1 => dimension == 1,
            Self::Spade2 => dimension == 2,
            Self::BowyerWatson3 => dimension == 3,
        }
    }
}
impl Tessellator for TessellatorKind {
    fn tessellate(&self, coords: &[f64], n: usize, dimension: usize) -> Result<SiteMesh> {
        require(
            coords.len() == n * dimension,
            "tessellator coordinate buffer shape",
        )?;
        if !self.supports(dimension) {
            return Err(GasError::Capability(format!(
                "tessellator {self:?} does not support {dimension} spatial dimensions; \
                 choose a projection onto at most three coordinates"
            )));
        }
        match dimension {
            1 => {
                let mut order: Vec<u32> = (0..n as u32).collect();
                order.sort_unstable_by(|&a, &b| {
                    coords[a as usize]
                        .total_cmp(&coords[b as usize])
                        .then(a.cmp(&b))
                });
                Ok(SiteMesh::path(coords.to_vec(), 1, &order))
            }
            2 => super::delaunay2::triangulate(coords, n),
            _ => super::delaunay3::tetrahedralize(coords, n),
        }
    }
}
