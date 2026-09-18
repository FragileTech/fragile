//! Degenerate swarms still have a Delaunay graph. Coincident walkers share a
//! site and become mutual neighbors; a swarm confined to an affine subspace is
//! triangulated inside that subspace after a distance-preserving projection.
//! Particles are never perturbed.
use super::{
    domain::{TessellationDomain, extend, initial_margin},
    graph::NeighborGraph,
    linalg::hestenes_svd,
    mesh::{SiteMesh, Tessellator, TessellatorKind},
    sites::SiteSet,
};
use crate::{GasError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DuplicatePolicy {
    /// Coincident walkers form a clique and every site edge lifts to all
    /// walker pairs across its two groups.
    #[default]
    LiftCliques,
    /// The lowest-index walker of a group carries the site edges; the others
    /// attach to it only. Linear in the number of walkers.
    Representative,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    #[default]
    Error,
    /// Continue with an empty graph: no graph forces and zero curvature.
    EmptyGraph,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DegeneracyPolicy {
    pub duplicates: DuplicatePolicy,
    /// Triangulate rank-deficient swarms inside their affine span.
    pub rank_projection: bool,
    pub on_failure: FailurePolicy,
}
impl Default for DegeneracyPolicy {
    fn default() -> Self {
        Self {
            duplicates: DuplicatePolicy::LiftCliques,
            rank_projection: true,
            on_failure: FailurePolicy::Error,
        }
    }
}

/// Walker-level tessellation: sites, their mesh and the lifted neighbor graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Tessellation {
    pub sites: SiteSet,
    pub mesh: SiteMesh,
    /// Affine rank of the sites; equals `sites.dimension` for a generic swarm.
    pub rank: usize,
    pub graph: NeighborGraph,
    /// Set when `FailurePolicy::EmptyGraph` absorbed a tessellation failure.
    pub failure: Option<String>,
}
impl Tessellation {
    /// True when the mesh lives in the projected coordinates themselves, so
    /// dual (Voronoi) constructions are meaningful.
    pub fn full_rank(&self) -> bool {
        self.failure.is_none()
            && self.rank == self.sites.dimension
            && self.mesh.dimension == self.sites.dimension
    }
}

/// Affine rank and principal frame of the sites. Returns centered coordinates,
/// singular values and the right singular basis (rows).
fn affine_rank(coords: &[f64], n: usize, d: usize) -> (Vec<f64>, usize, Vec<f64>) {
    let mut mean = vec![0.; d];
    for p in coords.chunks_exact(d) {
        for (m, x) in mean.iter_mut().zip(p) {
            *m += x;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let centered: Vec<f64> = coords
        .chunks_exact(d)
        .flat_map(|p| p.iter().zip(&mean).map(|(x, m)| x - m))
        .collect();
    let (sigma, basis) = hestenes_svd(&centered, n, d);
    let tolerance = sigma[0] * n.max(d) as f64 * f64::EPSILON;
    let rank = sigma.iter().filter(|&&s| s > tolerance).count();
    (centered, rank, basis)
}

/// Mesh of the sites in an open domain: path, projected or full tessellation.
fn open_mesh(
    coords: &[f64],
    n: usize,
    d: usize,
    tessellator: &dyn Tessellator,
    policy: &DegeneracyPolicy,
) -> Result<(SiteMesh, usize)> {
    if n < 2 {
        return Ok((SiteMesh::disconnected(d, coords.to_vec(), n), 0));
    }
    let (centered, rank, basis) = affine_rank(coords, n, d);
    if rank == d {
        return Ok((tessellator.tessellate(coords, n, d)?, rank));
    }
    if !policy.rank_projection {
        return Err(GasError::Numerical(format!(
            "swarm spans {rank} of {d} tessellation dimensions and rank projection is disabled"
        )));
    }
    let projected: Vec<f64> = (0..n)
        .flat_map(|s| {
            let (centered, basis) = (&centered, &basis);
            (0..rank).map(move |r| {
                (0..d)
                    .map(|k| centered[s * d + k] * basis[r * d + k])
                    .sum::<f64>()
            })
        })
        .collect();
    Ok((TessellatorKind::Auto.tessellate(&projected, n, rank)?, rank))
}
/// True when every simplex touching a real site has its circumsphere inside
/// the box grown by `margin`: no missing image can alter a real cell.
fn images_suffice(mesh: &SiteMesh, domain: &TessellationDomain, margin: f64) -> bool {
    let Some(bounds) = domain.bounds() else {
        return true;
    };
    let d = mesh.dimension;
    (0..mesh.simplex_count()).all(|k| {
        let simplex = mesh.simplex(k);
        if simplex.iter().all(|&v| v as usize >= mesh.real) {
            return true;
        }
        let Some((center, radius)) = super::voronoi::circumsphere(mesh, simplex) else {
            return false;
        };
        (0..d).all(|a| {
            center[a] - radius >= bounds.lower[a] - margin
                && center[a] + radius <= bounds.upper[a] + margin
        })
    })
}
fn site_mesh(
    sites: &SiteSet,
    tessellator: &dyn Tessellator,
    policy: &DegeneracyPolicy,
    domain: &TessellationDomain,
) -> Result<(SiteMesh, usize)> {
    let (n, d) = (sites.sites(), sites.dimension);
    if n == 0 || domain.bounds().is_none() {
        return open_mesh(&sites.coords, n, d, tessellator, policy);
    }
    let mut margin = initial_margin(n, d, domain);
    loop {
        let extended = extend(&sites.coords, n, d, domain, margin)?;
        let total = extended.base.len();
        let (mut mesh, rank) = open_mesh(&extended.coords, total, d, tessellator, policy)?;
        mesh.real = extended.real;
        mesh.base = extended.base;
        if extended.complete || (rank == d && images_suffice(&mesh, domain, margin)) {
            return Ok((mesh, rank));
        }
        margin *= 2.;
    }
}
/// Undirected edges between real sites. A periodic domain folds image
/// endpoints back onto their real site; a clip box drops image edges.
fn real_edges(mesh: &SiteMesh, domain: &TessellationDomain) -> Vec<[u32; 2]> {
    let mut out: Vec<[u32; 2]> = mesh
        .edges
        .iter()
        .filter_map(|&[a, b]| {
            let (ra, rb) = ((a as usize) < mesh.real, (b as usize) < mesh.real);
            let keep = (ra && rb) || (domain.periodic() && (ra || rb));
            let (a, b) = (mesh.base[a as usize], mesh.base[b as usize]);
            (keep && a != b).then_some([a.min(b), a.max(b)])
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn lifted_edge_count(sites: &SiteSet, edges: &[[u32; 2]], policy: DuplicatePolicy) -> u128 {
    let size = |s: u32| sites.group(s as usize).len() as u128;
    match policy {
        DuplicatePolicy::LiftCliques => {
            edges.iter().map(|&[a, b]| size(a) * size(b)).sum::<u128>()
                + (0..sites.sites() as u32)
                    .map(|s| size(s) * (size(s) - 1) / 2)
                    .sum::<u128>()
        }
        DuplicatePolicy::Representative => {
            edges.len() as u128 + (sites.walkers - sites.sites().min(sites.walkers)) as u128
        }
    }
}

/// Tessellate the sites and lift the site graph to walkers. `max_edges` bounds
/// the number of undirected walker edges before anything is materialized.
pub fn tessellate(
    sites: SiteSet,
    tessellator: &dyn Tessellator,
    policy: &DegeneracyPolicy,
    domain: &TessellationDomain,
    max_edges: usize,
) -> Result<Tessellation> {
    domain.validate(sites.dimension)?;
    let (mesh, rank) = match site_mesh(&sites, tessellator, policy, domain) {
        Ok(v) => v,
        Err(e) if policy.on_failure == FailurePolicy::EmptyGraph => {
            let graph = NeighborGraph::empty(sites.walkers);
            let mesh = SiteMesh::disconnected(sites.dimension, sites.coords.clone(), sites.sites());
            return Ok(Tessellation {
                sites,
                mesh,
                rank: 0,
                graph,
                failure: Some(e.to_string()),
            });
        }
        Err(e) => return Err(e),
    };
    let edges = real_edges(&mesh, domain);
    let count = lifted_edge_count(&sites, &edges, policy.duplicates);
    if count > max_edges as u128 {
        return Err(GasError::Capability(format!(
            "lifting coincident walkers needs {count} neighbor edges, above the budget of \
             {max_edges}; use the representative duplicate policy or raise max_batch_elements"
        )));
    }
    let mut pairs: Vec<[u32; 2]> = Vec::with_capacity(count as usize);
    match policy.duplicates {
        DuplicatePolicy::LiftCliques => {
            for &[a, b] in &edges {
                for &i in sites.group(a as usize) {
                    for &j in sites.group(b as usize) {
                        pairs.push([i, j]);
                    }
                }
            }
            for s in 0..sites.sites() {
                let g = sites.group(s);
                for (k, &i) in g.iter().enumerate() {
                    for &j in &g[k + 1..] {
                        pairs.push([i, j]);
                    }
                }
            }
        }
        DuplicatePolicy::Representative => {
            for &[a, b] in &edges {
                pairs.push([sites.group(a as usize)[0], sites.group(b as usize)[0]]);
            }
            for s in 0..sites.sites() {
                let g = sites.group(s);
                for &j in &g[1..] {
                    pairs.push([g[0], j]);
                }
            }
        }
    }
    let graph = NeighborGraph::from_undirected(sites.walkers, &pairs)?;
    Ok(Tessellation {
        sites,
        mesh,
        rank,
        graph,
        failure: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorBatch;
    fn run(d: usize, x: Vec<f64>, policy: DegeneracyPolicy) -> Result<Tessellation> {
        let n = x.len() / d;
        let x = TensorBatch::vectors(n, d, x).unwrap();
        let axes: Vec<usize> = (0..d).collect();
        let sites = SiteSet::build(&x, &axes, &vec![true; n], None)?;
        tessellate(
            sites,
            &TessellatorKind::Auto,
            &policy,
            &TessellationDomain::Open,
            1 << 20,
        )
    }
    #[test]
    fn coincident_swarm_is_a_clique_and_tiny_swarms_are_empty() {
        let t = run(3, [0.; 12].to_vec(), DegeneracyPolicy::default()).unwrap();
        assert_eq!((t.rank, t.graph.edges()), (0, 12));
        let t = run(2, vec![1., 2.], DegeneracyPolicy::default()).unwrap();
        assert!(t.graph.is_empty() && t.graph.nodes() == 1);
    }
    #[test]
    fn collinear_sites_form_a_path_in_geometric_order() {
        let t = run(
            2,
            vec![3., 6., 0., 0., 1., 2., -2., -4.],
            DegeneracyPolicy::default(),
        )
        .unwrap();
        assert_eq!(t.rank, 1);
        assert_eq!(
            t.graph.coo(),
            [[0, 2], [1, 2], [1, 3], [2, 0], [2, 1], [3, 1]]
        );
        assert!(!t.full_rank());
    }
    #[test]
    fn coplanar_sites_in_three_dimensions_triangulate_in_their_plane() {
        // Unit square in the plane z = x + y, plus its center.
        let p = |x: f64, y: f64| [x, y, x + y];
        let x: Vec<f64> = [p(0., 0.), p(1., 0.), p(0., 1.), p(1., 1.), p(0.5, 0.5)]
            .into_iter()
            .flatten()
            .collect();
        let t = run(3, x.clone(), DegeneracyPolicy::default()).unwrap();
        assert_eq!((t.rank, t.mesh.dimension), (2, 2));
        assert_eq!(t.graph.degree(4), 4);
        assert_eq!(t.graph.edges(), 16);
        let strict = DegeneracyPolicy {
            rank_projection: false,
            ..DegeneracyPolicy::default()
        };
        assert!(run(3, x.clone(), strict).is_err());
        let absorbed = run(
            3,
            x,
            DegeneracyPolicy {
                on_failure: FailurePolicy::EmptyGraph,
                ..strict
            },
        )
        .unwrap();
        assert!(absorbed.graph.is_empty() && absorbed.failure.is_some());
    }
    #[test]
    fn duplicate_groups_lift_to_products_and_cliques() {
        // Triangle sites with multiplicities 2, 1, 3.
        let x = vec![0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1.];
        let t = run(2, x.clone(), DegeneracyPolicy::default()).unwrap();
        // Undirected: 2*1 + 2*3 + 1*3 cross pairs + 1 + 0 + 3 clique pairs = 15.
        assert_eq!(t.graph.edges(), 30);
        assert_eq!(t.graph.row(0), &[1, 2, 3, 4, 5]);
        t.graph.validate().unwrap();
        let rep = DegeneracyPolicy {
            duplicates: DuplicatePolicy::Representative,
            ..DegeneracyPolicy::default()
        };
        let t = run(2, x.clone(), rep).unwrap();
        assert_eq!(t.graph.edges(), 2 * (3 + 3));
        assert_eq!(t.graph.row(3), &[0]);
        let axes = [0, 1];
        let batch = TensorBatch::vectors(6, 2, x).unwrap();
        let sites = SiteSet::build(&batch, &axes, &[true; 6], None).unwrap();
        let e = tessellate(
            sites,
            &TessellatorKind::Auto,
            &DegeneracyPolicy::default(),
            &TessellationDomain::Open,
            14,
        )
        .unwrap_err();
        assert!(e.to_string().contains("15 neighbor edges"));
    }
}
