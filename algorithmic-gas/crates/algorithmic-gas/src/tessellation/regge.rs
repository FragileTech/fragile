//! Regge calculus on the Delaunay complex.
//!
//! Curvature concentrates on the codimension-2 hinges: vertices in 2D, edges
//! in 3D. With deficit angle delta_h = 2 pi - (angles around h), the
//! Einstein-Hilbert action is  integral R dV = 2 sum_h |h| delta_h. Assigning
//! every hinge to its sites and dividing by the barycentric dual volume gives
//! R_i = 2 delta_i / A_i in 2D and R_i = sum_{e ni i} L_e delta_e / V_i in 3D.
//!
//! All angles and volumes derive from edge lengths alone, so `curvature`
//! accepts any length per edge. Given exact geodesic lengths the action
//! converges to the continuum one. A deficit angle is O(h^2), the same order as
//! the relative error of a length built from the endpoint metrics
//! (`EdgeLengths`), so with those lengths the result is a curvature indicator
//! with an O(1) discretization bias, not a convergent estimator. Euclidean
//! lengths of a flat triangulation have zero interior deficit. Hull sites have
//! open angle fans and are reported invalid with zero curvature.
use super::{degenerate::Tessellation, par::Parallelism};
use crate::Real;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReggeLengths {
    /// Edge lengths of the walker metric (mean of the endpoint metrics): a
    /// biased indicator, see the module documentation.
    #[default]
    Geodesic,
    /// Coordinate edge lengths.
    Euclidean,
}

pub struct ReggeCurvature<T: Real> {
    pub scalar: Vec<T>,
    pub deficit: Vec<T>,
    pub dual_volume: Vec<T>,
    pub valid: Vec<bool>,
    /// integral R dV over the valid sites.
    pub action: T,
}

/// Angle opposite side `c` in a triangle with sides a, b, c (Kahan's formula,
/// accurate for needle triangles). Lengths violating the triangle inequality
/// give the degenerate angle 0 or pi.
fn angle(a: f64, b: f64, c: f64) -> f64 {
    let (a, b) = if a >= b { (a, b) } else { (b, a) };
    let mu = if b >= c { c - (a - b) } else { b - (a - c) };
    let ratio = (((a - b) + c) * mu) / ((a + (b + c)) * ((a - c) + b));
    if ratio.is_nan() || ratio < 0. {
        return if c >= a + b { std::f64::consts::PI } else { 0. };
    }
    2. * ratio.sqrt().atan()
}
fn heron(a: f64, b: f64, c: f64) -> f64 {
    let s = 0.5 * (a + b + c);
    (s * (s - a) * (s - b) * (s - c)).max(0.).sqrt()
}
/// Volume of a tetrahedron from its edge lengths (Cayley-Menger), zero when
/// the lengths are not realizable.
fn tet_volume(l: [[f64; 4]; 4]) -> f64 {
    let (u, v, w) = (l[0][1], l[0][2], l[0][3]);
    let (big_u, big_v, big_w) = (l[2][3], l[1][3], l[1][2]);
    let (u2, v2, w2) = (u * u, v * v, w * w);
    let (a, b, c) = (
        v2 + w2 - big_u * big_u,
        w2 + u2 - big_v * big_v,
        u2 + v2 - big_w * big_w,
    );
    let det = 4. * u2 * v2 * w2 - u2 * a * a - v2 * b * b - w2 * c * c + a * b * c;
    det.max(0.).sqrt() / 12.
}

/// `slot_length` holds one length per directed edge slot of the walker graph.
pub fn curvature<T: Real>(
    tessellation: &Tessellation,
    slot_length: &[T],
    _par: Parallelism,
) -> ReggeCurvature<T> {
    let (sites, mesh, graph) = (&tessellation.sites, &tessellation.mesh, &tessellation.graph);
    let n = sites.walkers;
    let mut out = ReggeCurvature {
        scalar: vec![T::ZERO; n],
        deficit: vec![T::ZERO; n],
        dual_volume: vec![T::ZERO; n],
        valid: vec![false; n],
        action: T::ZERO,
    };
    let d = mesh.dimension;
    if !tessellation.full_rank() || !(2..=3).contains(&d) {
        return out;
    }
    // Length of a mesh edge through the representative walkers of its sites.
    let length = |a: u32, b: u32| -> Option<f64> {
        let (a, b) = (mesh.base[a as usize], mesh.base[b as usize]);
        if a == b {
            return None;
        }
        let (i, j) = (sites.group(a as usize)[0], sites.group(b as usize)[0]);
        graph
            .slot(i as usize, j)
            .map(|e| slot_length[e].to_f64())
            .filter(|l| l.is_finite() && *l > 0.)
    };
    let real = mesh.real;
    let mut around = vec![0f64; real];
    let mut dual = vec![0f64; real];
    // 3D: dihedral sum and length per mesh edge, keyed by the sorted edge list.
    let mut hinge_angle = vec![0f64; mesh.edges.len()];
    let mut broken = vec![false; real];
    for k in 0..mesh.simplex_count() {
        let s = mesh.simplex(k);
        if s.iter().all(|&v| v as usize >= real) {
            continue;
        }
        let mut l = [[0f64; 4]; 4];
        let mut ok = true;
        for a in 0..=d {
            for b in a + 1..=d {
                match length(s[a], s[b]) {
                    Some(v) => {
                        l[a][b] = v;
                        l[b][a] = v;
                    }
                    None => ok = false,
                }
            }
        }
        if !ok {
            for &v in s {
                if (v as usize) < real {
                    broken[v as usize] = true;
                }
            }
            continue;
        }
        if d == 2 {
            let area = heron(l[0][1], l[0][2], l[1][2]);
            for a in 0..3 {
                let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                if (s[a] as usize) < real {
                    around[s[a] as usize] += angle(l[a][b], l[a][c], l[b][c]);
                    dual[s[a] as usize] += area / 3.;
                }
            }
        } else {
            let volume = tet_volume(l);
            for a in 0..4 {
                if (s[a] as usize) < real {
                    dual[s[a] as usize] += volume / 4.;
                }
            }
            for a in 0..4 {
                for b in a + 1..4 {
                    let others: Vec<usize> = (0..4).filter(|&k| k != a && k != b).collect();
                    let (c, e) = (others[0], others[1]);
                    // Dihedral angle on edge ab with p = ab, q = ac, r = ae:
                    // (p x q).(p x r) = p^2 (q.r) - (p.q)(p.r) and
                    // |p| p.(q x r) = 6 V |p| share the factor 4 A_abc A_abe.
                    let dot = |x: usize, y: usize| {
                        0.5 * (l[a][x] * l[a][x] + l[a][y] * l[a][y] - l[x][y] * l[x][y])
                    };
                    let cosine = l[a][b] * l[a][b] * dot(c, e) - dot(b, c) * dot(b, e);
                    let dihedral = (6. * volume * l[a][b]).atan2(cosine);
                    let key = [s[a].min(s[b]), s[a].max(s[b])];
                    if let Ok(slot) = mesh.edges.binary_search(&key) {
                        hinge_angle[slot] += dihedral;
                    }
                }
            }
        }
    }
    let tau = std::f64::consts::TAU;
    let mut curvature_volume = vec![0f64; real];
    if d == 2 {
        for s in 0..real {
            curvature_volume[s] = 2. * (tau - around[s]);
        }
    } else {
        for (slot, &[a, b]) in mesh.edges.iter().enumerate() {
            // An edge on the hull has an open fan: it is no hinge.
            if mesh.hull[a as usize] && mesh.hull[b as usize] {
                continue;
            }
            let Some(len) = length(a, b) else { continue };
            let weight = len * (tau - hinge_angle[slot]);
            for v in [a, b] {
                if (v as usize) < real {
                    curvature_volume[v as usize] += weight;
                }
            }
        }
    }
    let mut action = 0f64;
    for s in 0..real {
        let valid = !mesh.hull[s] && !broken[s] && dual[s] > 0.;
        if !valid {
            continue;
        }
        action += curvature_volume[s];
        let group = sites.group(s);
        let share = group.len() as f64;
        for &i in group {
            let i = i as usize;
            out.valid[i] = true;
            out.scalar[i] = T::from_f64(curvature_volume[s] / dual[s]);
            out.deficit[i] = T::from_f64(if d == 2 {
                tau - around[s]
            } else {
                curvature_volume[s]
            });
            out.dual_volume[i] = T::from_f64(dual[s] / share);
        }
    }
    out.action = T::from_f64(action);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn regular_tetrahedron_volume_and_angles() {
        let l = [
            [0., 1., 1., 1.],
            [1., 0., 1., 1.],
            [1., 1., 0., 1.],
            [1., 1., 1., 0.],
        ];
        let v = tet_volume(l);
        assert!((v - 2f64.sqrt() / 12.).abs() < 1e-15);
        assert!((angle(1., 1., 1.) - std::f64::consts::FRAC_PI_3).abs() < 1e-15);
        // Needle: the angle opposite a tiny side, and impossible lengths.
        assert!((angle(1., 1., 1e-9) - 1e-9).abs() < 1e-24);
        assert_eq!(angle(1., 1., 3.), std::f64::consts::PI);
        assert_eq!(angle(3., 1., 1.), 0.);
        assert!((heron(3., 4., 5.) - 6.).abs() < 1e-14);
    }
}
