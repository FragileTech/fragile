//! Planar Delaunay triangulation through Spade's exact-predicate bulk loader.
//! `bulk_load_stable` keeps vertex order, so vertex index equals site index.
use super::mesh::SiteMesh;
use crate::{GasError, Result};
use spade::{DelaunayTriangulation, HasPosition, Point2, Triangulation};

struct Site(Point2<f64>);
impl HasPosition for Site {
    type Scalar = f64;
    fn position(&self) -> Point2<f64> {
        self.0
    }
}

pub fn triangulate(coords: &[f64], n: usize) -> Result<SiteMesh> {
    let sites = (0..n)
        .map(|i| Site(Point2::new(coords[2 * i], coords[2 * i + 1])))
        .collect();
    let t = DelaunayTriangulation::<Site>::bulk_load_stable(sites).map_err(|e| {
        GasError::Numerical(format!("planar Delaunay rejected a coordinate: {e:?}"))
    })?;
    if t.num_vertices() != n {
        return Err(GasError::Numerical(
            "planar Delaunay merged sites that must be distinct".into(),
        ));
    }
    let mut simplices = Vec::with_capacity(t.num_inner_faces() * 3);
    for face in t.inner_faces() {
        let mut v = face.vertices().map(|h| h.index() as u32);
        // Canonical rotation: smallest index first, orientation preserved.
        let k = (0..3).min_by_key(|&k| v[k]).unwrap_or(0);
        v.rotate_left(k);
        simplices.extend(v);
    }
    let mut hull = vec![false; n];
    for e in t.convex_hull() {
        hull[e.from().index()] = true;
    }
    let mut triangles: Vec<[u32; 3]> = simplices
        .chunks_exact(3)
        .map(|s| [s[0], s[1], s[2]])
        .collect();
    triangles.sort_unstable();
    let simplices: Vec<u32> = triangles.into_iter().flatten().collect();
    Ok(SiteMesh {
        dimension: 2,
        coords: coords.to_vec(),
        edges: SiteMesh::edges_from_simplices(&simplices, 3),
        simplices,
        hull,
        real: n,
        base: (0..n as u32).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cloud(n: usize) -> Vec<f64> {
        // Deterministic generic-position points (irrational rotations).
        (0..n)
            .flat_map(|i| {
                let t = i as f64 + 1.;
                [(t * 0.754877666).fract(), (t * 0.569840291).fract()]
            })
            .collect()
    }
    #[test]
    fn triangulation_is_delaunay_and_euler_consistent() {
        let n = 60;
        let c = cloud(n);
        let m = triangulate(&c, n).unwrap();
        let p = |i: u32| robust::Coord {
            x: c[2 * i as usize],
            y: c[2 * i as usize + 1],
        };
        for s in m.simplices.chunks_exact(3) {
            assert!(robust::orient2d(p(s[0]), p(s[1]), p(s[2])) > 0.);
            for q in 0..n as u32 {
                if !s.contains(&q) {
                    assert!(robust::incircle(p(s[0]), p(s[1]), p(s[2]), p(q)) <= 0.);
                }
            }
        }
        // V - E + F = 1 for a triangulated disc.
        assert_eq!(
            n as i64 - m.edges.len() as i64 + m.simplex_count() as i64,
            1
        );
        let h = m.hull.iter().filter(|&&b| b).count();
        assert_eq!(m.edges.len(), 3 * n - 3 - h);
    }
    #[test]
    fn three_sites_make_one_triangle() {
        let m = triangulate(&[0., 0., 1., 0., 0., 1.], 3).unwrap();
        assert_eq!(m.simplices, [0, 1, 2]);
        assert_eq!(m.edges, [[0, 1], [0, 2], [1, 2]]);
        assert!(m.hull.iter().all(|&b| b));
    }
}
