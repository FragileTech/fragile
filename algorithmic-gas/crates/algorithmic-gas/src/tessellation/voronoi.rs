//! Voronoi cells as the dual of the Delaunay mesh, for any tessellator that
//! returns simplices. A facet is the convex polygon of circumcenters around a
//! Delaunay edge; because every facet lies in a bisector at half the edge
//! length from the site, the cell volume is the sum of the pyramids
//! V_i = sum_j A_ij |x_j - x_i| / (2 d). Bounded domains need no clipping:
//! their image sites close the cells. Every real site is processed
//! independently, so the stage parallelizes over sites.
use super::{
    degenerate::Tessellation,
    domain::TessellationDomain,
    linalg::lu_solve_multi,
    mesh::SiteMesh,
    par::{Parallelism, map_indexed},
    sites::NO_SITE,
};
use crate::{Real, Result, error::require};
use serde::{Deserialize, Serialize};

/// Boundary classes: 0 open or touching the boundary, 1 adjacent to class 0,
/// 2 interior.
pub const TIER_BOUNDARY: u8 = 0;
pub const TIER_ADJACENT: u8 = 1;
pub const TIER_INTERIOR: u8 = 2;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoronoiCellsConfig {
    /// Distance to a clip-box face below which a cell counts as boundary.
    pub boundary_tolerance: f64,
}

/// Walker-level Voronoi data. Coincident walkers share a cell: its volume and
/// facet areas are split evenly among them.
#[derive(Clone, Debug, PartialEq)]
pub struct VoronoiCells<T: Real> {
    pub dimension: usize,
    pub volume: Vec<T>,
    pub bounded: Vec<bool>,
    pub tier: Vec<u8>,
    /// Facet area per directed edge slot of the walker graph.
    pub facet_area: Vec<T>,
    pub vertex_offsets: Vec<u32>,
    /// Cell vertices, `dimension` coordinates each, in the frame of the walker
    /// (unwrapped across a periodic boundary).
    pub vertices: Vec<T>,
}
impl<T: Real> VoronoiCells<T> {
    pub fn walkers(&self) -> usize {
        self.volume.len()
    }
    pub fn cell_vertices(&self, i: usize) -> &[T] {
        &self.vertices[self.vertex_offsets[i] as usize * self.dimension
            ..self.vertex_offsets[i + 1] as usize * self.dimension]
    }
}

/// Circumcenter and circumradius of a simplex, `None` when it is numerically flat.
pub fn circumsphere(mesh: &SiteMesh, simplex: &[u32]) -> Option<(Vec<f64>, f64)> {
    let d = mesh.dimension;
    let origin = mesh.site(simplex[0] as usize);
    let mut a = vec![0.; d * d];
    let mut b = vec![0.; d];
    for k in 0..d {
        let p = mesh.site(simplex[k + 1] as usize);
        for c in 0..d {
            let e = p[c] - origin[c];
            a[k * d + c] = 2. * e;
            b[k] += e * e;
        }
    }
    lu_solve_multi(&mut a, &mut b, d, 1).ok()?;
    let radius = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !radius.is_finite() {
        return None;
    }
    Some((b.iter().zip(origin).map(|(o, p)| o + p).collect(), radius))
}

struct SiteCell {
    bounded: bool,
    volume: f64,
    vertices: Vec<f64>,
    /// (neighbor site in the extended mesh, facet area), ascending by neighbor.
    facets: Vec<(u32, f64)>,
}
fn unbounded() -> SiteCell {
    SiteCell {
        bounded: false,
        volume: 0.,
        vertices: Vec::new(),
        facets: Vec::new(),
    }
}
/// Area of the convex polygon `points` (3D, cyclic order) by its vector area.
fn polygon_area(points: &[[f64; 3]]) -> f64 {
    let o = points[0];
    let mut v = [0.; 3];
    for w in points[1..].windows(2) {
        let a = [w[0][0] - o[0], w[0][1] - o[1], w[0][2] - o[2]];
        let b = [w[1][0] - o[0], w[1][1] - o[1], w[1][2] - o[2]];
        v[0] += a[1] * b[2] - a[2] * b[1];
        v[1] += a[2] * b[0] - a[0] * b[2];
        v[2] += a[0] * b[1] - a[1] * b[0];
    }
    0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
fn site_cell(
    mesh: &SiteMesh,
    s: usize,
    incident: &[u32],
    centers: &[Option<Vec<f64>>],
) -> SiteCell {
    let d = mesh.dimension;
    if mesh.hull[s]
        || incident.is_empty()
        || incident.iter().any(|&k| centers[k as usize].is_none())
    {
        return unbounded();
    }
    let x = mesh.site(s);
    let centroid = |k: u32| -> Vec<f64> {
        let simplex = mesh.simplex(k as usize);
        (0..d)
            .map(|c| {
                simplex
                    .iter()
                    .map(|&v| mesh.site(v as usize)[c])
                    .sum::<f64>()
                    / simplex.len() as f64
            })
            .collect()
    };
    // (neighbor, simplex) pairs grouped by neighbor.
    let mut around: Vec<(u32, u32)> = incident
        .iter()
        .flat_map(|&k| {
            mesh.simplex(k as usize)
                .iter()
                .filter(move |&&v| v as usize != s)
                .map(move |&v| (v, k))
        })
        .collect();
    around.sort_unstable();
    let mut cell = SiteCell {
        bounded: true,
        volume: 0.,
        vertices: incident
            .iter()
            .flat_map(|&k| centers[k as usize].clone().unwrap_or_default())
            .collect(),
        facets: Vec::new(),
    };
    let mut start = 0;
    while start < around.len() {
        let neighbor = around[start].0;
        let run = around[start..]
            .iter()
            .take_while(|p| p.0 == neighbor)
            .count();
        let ring: Vec<u32> = around[start..start + run].iter().map(|p| p.1).collect();
        start += run;
        let y = mesh.site(neighbor as usize);
        let axis: Vec<f64> = y.iter().zip(x).map(|(a, b)| a - b).collect();
        let length = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
        let area = if d == 2 {
            // An interior edge has exactly two triangles.
            if ring.len() != 2 {
                return unbounded();
            }
            let (a, b) = (&centers[ring[0] as usize], &centers[ring[1] as usize]);
            match (a, b) {
                (Some(a), Some(b)) => ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt(),
                _ => return unbounded(),
            }
        } else {
            if ring.len() < 3 {
                return unbounded();
            }
            // Order the tetrahedra around the edge by the angle of their
            // centroids, which lie strictly inside the fan.
            let u0 = {
                let k = (0..3)
                    .min_by(|&a, &b| axis[a].abs().total_cmp(&axis[b].abs()))
                    .unwrap_or(0);
                let mut e = [0.; 3];
                e[k] = 1.;
                let dot = axis[k] / (length * length);
                [
                    e[0] - dot * axis[0],
                    e[1] - dot * axis[1],
                    e[2] - dot * axis[2],
                ]
            };
            let u1 = [
                axis[1] * u0[2] - axis[2] * u0[1],
                axis[2] * u0[0] - axis[0] * u0[2],
                axis[0] * u0[1] - axis[1] * u0[0],
            ];
            let mut ordered: Vec<(f64, u32)> = ring
                .iter()
                .map(|&k| {
                    let c = centroid(k);
                    let r = [c[0] - x[0], c[1] - x[1], c[2] - x[2]];
                    let (p, q) = (
                        r[0] * u0[0] + r[1] * u0[1] + r[2] * u0[2],
                        r[0] * u1[0] + r[1] * u1[1] + r[2] * u1[2],
                    );
                    (q.atan2(p), k)
                })
                .collect();
            ordered.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let polygon: Vec<[f64; 3]> = ordered
                .iter()
                .filter_map(|&(_, k)| centers[k as usize].as_ref().map(|c| [c[0], c[1], c[2]]))
                .collect();
            polygon_area(&polygon)
        };
        cell.volume += area * length / (2 * d) as f64;
        cell.facets.push((neighbor, area));
    }
    cell
}

/// Voronoi cells of a tessellation. A swarm that does not span the tessellated
/// space has no dual cells: every cell is reported unbounded.
pub fn cells<T: Real>(
    tessellation: &Tessellation,
    domain: &TessellationDomain,
    config: &VoronoiCellsConfig,
    graph_sources: &[u32],
    par: Parallelism,
) -> Result<VoronoiCells<T>> {
    require(
        config.boundary_tolerance.is_finite() && config.boundary_tolerance >= 0.,
        "Voronoi boundary tolerance must be nonnegative",
    )?;
    let (sites, mesh, graph) = (&tessellation.sites, &tessellation.mesh, &tessellation.graph);
    let d = sites.dimension;
    let n = sites.walkers;
    let mut out = VoronoiCells {
        dimension: d,
        volume: vec![T::ZERO; n],
        bounded: vec![false; n],
        tier: vec![TIER_BOUNDARY; n],
        facet_area: vec![T::ZERO; graph.edges()],
        vertex_offsets: vec![0; n + 1],
        vertices: Vec::new(),
    };
    if !tessellation.full_rank() || !(2..=3).contains(&d) {
        return Ok(out);
    }
    let simplices = mesh.simplex_count();
    let centers: Vec<Option<Vec<f64>>> = map_indexed(simplices, par, |k| {
        circumsphere(mesh, mesh.simplex(k)).map(|c| c.0)
    });
    // Simplices incident to every real site, in CSR form.
    let mut offsets = vec![0u32; mesh.real + 1];
    for &v in &mesh.simplices {
        if (v as usize) < mesh.real {
            offsets[v as usize + 1] += 1;
        }
    }
    for s in 0..mesh.real {
        offsets[s + 1] += offsets[s];
    }
    let mut incident = vec![0u32; offsets[mesh.real] as usize];
    let mut cursor = offsets.clone();
    for k in 0..simplices {
        for &v in mesh.simplex(k) {
            if (v as usize) < mesh.real {
                incident[cursor[v as usize] as usize] = k as u32;
                cursor[v as usize] += 1;
            }
        }
    }
    let site_cells = map_indexed(mesh.real, par, |s| {
        site_cell(
            mesh,
            s,
            &incident[offsets[s] as usize..offsets[s + 1] as usize],
            &centers,
        )
    });
    // Site classes, then the lift to walkers.
    let touches_boundary = |s: usize| match domain {
        TessellationDomain::Open => !site_cells[s].bounded,
        TessellationDomain::Periodic { .. } => false,
        TessellationDomain::ClipBox { bounds } => {
            !site_cells[s].bounded
                || site_cells[s]
                    .facets
                    .iter()
                    .any(|f| f.0 as usize >= mesh.real)
                || (0..d).any(|k| {
                    let v = mesh.site(s)[k];
                    v - bounds.lower[k] <= config.boundary_tolerance
                        || bounds.upper[k] - v <= config.boundary_tolerance
                })
        }
    };
    let mut tier = vec![TIER_INTERIOR; mesh.real];
    for (s, t) in tier.iter_mut().enumerate() {
        if touches_boundary(s) {
            *t = TIER_BOUNDARY;
        }
    }
    let boundary = tier.clone();
    for &[a, b] in &mesh.edges {
        let (a, b) = (
            mesh.base[a as usize] as usize,
            mesh.base[b as usize] as usize,
        );
        if !domain.periodic() && a != b {
            if boundary[a] == TIER_BOUNDARY && tier[b] == TIER_INTERIOR {
                tier[b] = TIER_ADJACENT;
            }
            if boundary[b] == TIER_BOUNDARY && tier[a] == TIER_INTERIOR {
                tier[a] = TIER_ADJACENT;
            }
        }
    }
    for i in 0..n {
        let s = sites.site_of_walker[i];
        if s == NO_SITE {
            out.vertex_offsets[i + 1] = out.vertex_offsets[i];
            continue;
        }
        let cell = &site_cells[s as usize];
        let share = sites.group(s as usize).len() as f64;
        out.bounded[i] = cell.bounded;
        out.tier[i] = tier[s as usize];
        out.volume[i] = T::from_f64(cell.volume / share);
        out.vertices
            .extend(cell.vertices.iter().map(|&v| T::from_f64(v)));
        out.vertex_offsets[i + 1] = out.vertex_offsets[i] + (cell.vertices.len() / d) as u32;
    }
    let areas = map_indexed(graph.edges(), par, |e| {
        let (i, j) = (graph_sources[e] as usize, graph.neighbors()[e] as usize);
        let (a, b) = (sites.site_of_walker[i], sites.site_of_walker[j]);
        if a == b || a == NO_SITE || b == NO_SITE {
            return T::ZERO;
        }
        // All images of b adjacent to a contribute in a small periodic box. A
        // facet shared with an open cell is read from the bounded side.
        let facet = |from: u32, to: u32| -> f64 {
            site_cells[from as usize]
                .facets
                .iter()
                .filter(|f| mesh.base[f.0 as usize] == to)
                .map(|f| f.1)
                .sum()
        };
        let total = if site_cells[a as usize].bounded {
            facet(a, b)
        } else {
            facet(b, a)
        };
        let split = (sites.group(a as usize).len() * sites.group(b as usize).len()) as f64;
        T::from_f64(total / split)
    });
    out.facet_area = areas;
    Ok(out)
}
