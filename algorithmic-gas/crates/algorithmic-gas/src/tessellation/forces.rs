//! Forces carried by the tessellation graph.
//!
//! The viscous force is the weighted graph Laplacian of the velocities,
//! F_i = nu sum_j w_ij (v_j - v_i), restricted to tessellation neighbors. Its
//! spatial Jacobian, fitted per walker by weighted least squares over the same
//! neighbors, has an antisymmetric part: the curl 2-form of the emergent
//! connection. The Boris step rotates velocities by the Cayley transform of
//! that 2-form, which is exactly norm preserving.
use super::{
    graph::NeighborGraph,
    linalg::{cayley_apply, lu_solve_multi, skew_rate},
    par::{Parallelism, fill_rows, try_map_indexed},
};
use crate::{Real, Result, physics::geometry::symmetric_eigen};

pub struct GraphField<'a, T: Real> {
    pub graph: &'a NeighborGraph,
    /// One weight per directed edge slot.
    pub weights: &'a [T],
    /// `[walkers, dimension]` ambient positions.
    pub positions: &'a [T],
    pub dimension: usize,
    pub eligible: &'a [bool],
    /// (axis, length) of every periodic coordinate.
    pub wrap: &'a [(usize, T)],
}
impl<T: Real> GraphField<'_, T> {
    fn delta(&self, i: usize, j: usize) -> Vec<T> {
        let d = self.dimension;
        let mut out: Vec<T> = (0..d)
            .map(|a| self.positions[j * d + a] - self.positions[i * d + a])
            .collect();
        for &(axis, length) in self.wrap {
            let v = out[axis];
            out[axis] = v - length * (v / length + T::from_f64(0.5)).floor();
        }
        out
    }
}
/// F_i = nu sum_j w_ij (v_j - v_i) over eligible neighbors.
pub fn viscous_force<T: Real>(
    field: &GraphField<'_, T>,
    velocities: &[T],
    nu: T,
    par: Parallelism,
) -> Vec<T> {
    let d = field.dimension;
    let mut out = vec![T::ZERO; velocities.len()];
    fill_rows(&mut out, d, par, |i, row| {
        if !field.eligible[i] {
            return;
        }
        for e in field.graph.range(i) {
            let j = field.graph.neighbors()[e] as usize;
            if !field.eligible[j] {
                continue;
            }
            let w = nu * field.weights[e];
            for a in 0..d {
                row[a] = row[a] + w * (velocities[j * d + a] - velocities[i * d + a]);
            }
        }
    });
    out
}
/// Curl 2-form `[walkers, d, d]` of a walker vector field: the antisymmetric
/// part of its least-squares Jacobian. With A = sum w dF (x) dx and
/// B = sum w dx (x) dx the Jacobian solves J (B + ridge I) = A, where the ridge
/// is sqrt(eps) times the mean eigenvalue of B.
pub fn curl<T: Real>(field: &GraphField<'_, T>, force: &[T], par: Parallelism) -> Result<Vec<T>> {
    let d = field.dimension;
    let rows = try_map_indexed(field.eligible.len(), par, |i| {
        let mut a = vec![T::ZERO; d * d];
        let mut b = vec![T::ZERO; d * d];
        if !field.eligible[i] {
            return Ok(a);
        }
        for e in field.graph.range(i) {
            let j = field.graph.neighbors()[e] as usize;
            if !field.eligible[j] {
                continue;
            }
            let w = field.weights[e];
            let dx = field.delta(i, j);
            for r in 0..d {
                let df = w * (force[j * d + r] - force[i * d + r]);
                let wx = w * dx[r];
                for c in 0..d {
                    a[r * d + c] = a[r * d + c] + df * dx[c];
                    b[r * d + c] = b[r * d + c] + wx * dx[c];
                }
            }
        }
        let trace = (0..d).fold(T::ZERO, |s, k| s + b[k * d + k]) / T::from_f64(d as f64);
        let ridge = (trace * T::EPSILON.sqrt()).max(T::MIN_POSITIVE);
        for k in 0..d {
            b[k * d + k] = b[k * d + k] + ridge;
        }
        // J^T = solve(B, A^T): the right-hand sides are the columns of A^T.
        let mut jt = vec![T::ZERO; d * d];
        for r in 0..d {
            for c in 0..d {
                jt[r * d + c] = a[c * d + r];
            }
        }
        lu_solve_multi(&mut b, &mut jt, d, d)?;
        let mut out = vec![T::ZERO; d * d];
        for r in 0..d {
            for c in 0..d {
                out[r * d + c] = (jt[c * d + r] - jt[r * d + c]) / T::from_f64(2.);
            }
        }
        Ok(out)
    })?;
    Ok(rows.into_iter().flatten().collect())
}
/// Largest rotation rate of a skew matrix.
fn spectral_norm_skew<T: Real>(a: &[T], d: usize) -> Result<T> {
    if d <= 3 {
        return Ok(skew_rate(a));
    }
    let mut gram = vec![T::ZERO; d * d];
    for r in 0..d {
        for c in r..d {
            let v = (0..d).fold(T::ZERO, |s, k| s + a[k * d + r] * a[k * d + c]);
            gram[r * d + c] = v;
            gram[c * d + r] = v;
        }
    }
    let (l, _) = symmetric_eigen(&gram, d)?;
    Ok(l.into_iter().fold(T::ZERO, |m, v| m.max(v)).sqrt())
}
/// Cayley rotation v -> (I - A)^{-1} (I + A) v with A = scale * curl. Returns
/// the rotated velocities and the rotation angle 2 atan |A| of every walker.
pub fn boris_rotate<T: Real>(
    velocities: &[T],
    curl: &[T],
    dimension: usize,
    scale: T,
    eligible: &[bool],
    par: Parallelism,
) -> Result<(Vec<T>, Vec<T>)> {
    let d = dimension;
    let rows = try_map_indexed(eligible.len(), par, |i| {
        let v = &velocities[i * d..(i + 1) * d];
        if !eligible[i] {
            return Ok((v.to_vec(), T::ZERO));
        }
        let a: Vec<T> = curl[i * d * d..(i + 1) * d * d]
            .iter()
            .map(|&c| scale * c)
            .collect();
        let angle = T::from_f64(2.) * spectral_norm_skew(&a, d)?.atan();
        Ok((cayley_apply(&a, v)?, angle))
    })?;
    let mut rotated = Vec::with_capacity(velocities.len());
    let mut angles = Vec::with_capacity(eligible.len());
    for (v, angle) in rows {
        rotated.extend(v);
        angles.push(angle);
    }
    Ok((rotated, angles))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ring() -> (NeighborGraph, Vec<f64>) {
        // Four walkers on the unit circle, joined in a cycle.
        let g = NeighborGraph::from_undirected(4, &[[0, 1], [1, 2], [2, 3], [3, 0]]).unwrap();
        (g, vec![1., 0., 0., 1., -1., 0., 0., -1.])
    }
    #[test]
    fn viscous_force_conserves_momentum_for_symmetric_weights() {
        let (graph, x) = ring();
        let w = vec![0.25; graph.edges()];
        let field = GraphField {
            graph: &graph,
            weights: &w,
            positions: &x,
            dimension: 2,
            eligible: &[true; 4],
            wrap: &[],
        };
        let v = [1., 2., -0.5, 0.25, 3., -1., 0., 0.75];
        let f = viscous_force(&field, &v, 2., Parallelism::Serial);
        assert!((0..2).all(|a| (0..4).map(|i| f[i * 2 + a]).sum::<f64>().abs() < 1e-14));
        assert!((f[0] - 2. * 0.25 * ((-0.5 - 1.) + (0. - 1.))).abs() < 1e-15);
    }
    #[test]
    fn curl_of_a_rigid_rotation_is_its_generator() {
        let (graph, x) = ring();
        let w = vec![1.; graph.edges()];
        let field = GraphField {
            graph: &graph,
            weights: &w,
            positions: &x,
            dimension: 2,
            eligible: &[true; 4],
            wrap: &[],
        };
        // F = Omega x with Omega = [[0, -w], [w, 0]] plus a symmetric strain.
        let omega = 0.7;
        let f: Vec<f64> = x
            .chunks_exact(2)
            .flat_map(|p| [0.3 * p[0] - omega * p[1], omega * p[0] - 0.3 * p[1]])
            .collect();
        let c = curl(&field, &f, Parallelism::Serial).unwrap();
        for i in 0..4 {
            assert!((c[i * 4 + 1] + omega).abs() < 1e-7 && (c[i * 4 + 2] - omega).abs() < 1e-7);
            assert!(c[i * 4].abs() < 1e-15 && c[i * 4 + 3].abs() < 1e-15);
        }
        let v = [1., 0., 0., 2., -1., 1., 0.5, 0.5];
        let (rotated, angle) =
            boris_rotate(&v, &c, 2, 0.1, &[true; 4], Parallelism::Serial).unwrap();
        for i in 0..4 {
            let n = |s: &[f64]| s[i * 2] * s[i * 2] + s[i * 2 + 1] * s[i * 2 + 1];
            assert!((n(&rotated) - n(&v)).abs() < 1e-14);
            assert!((angle[i] - 2. * (0.1 * omega).atan()).abs() < 1e-8);
        }
        // Counterclockwise for positive omega.
        assert!(rotated[1] > 0.);
    }
}
