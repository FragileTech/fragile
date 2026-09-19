//! Hessian of a scalar walker field estimated on the neighbor graph.
//!
//! The diagonal estimator is a three-point second difference along every
//! coordinate axis, using the closest neighbor inside a 30 degree cone on each
//! side at its own spacing. The full estimator is the quadratic part of a local
//! least-squares fit.
//! Axes without a neighbor on both sides contribute zero.
use super::{
    frame::GeometryFrame,
    par::{Parallelism, map_indexed},
    ricci::fit_local_quadratic,
};
use crate::{Real, Result};

const MAX_ANGLE_DEGREES: f64 = 30.;

/// `[walkers, d, d]` Hessian estimates of `values`.
pub fn estimate<T: Real>(
    frame: &GeometryFrame<'_, T>,
    values: &[T],
    full: bool,
    par: Parallelism,
) -> Result<Vec<T>> {
    let d = frame.dimension;
    if full {
        return Ok(fit_local_quadratic(frame, values, None, 1e-6, par)?.1);
    }
    let cos_threshold = T::from_f64(MAX_ANGLE_DEGREES.to_radians().cos());
    let tiny = T::from_f64(1e-10);
    let rows = map_indexed(frame.walkers(), par, |i| {
        let mut h = vec![T::ZERO; d * d];
        // Closest aligned neighbor on the positive and negative side per axis.
        let mut side: Vec<[Option<(T, usize)>; 2]> = vec![[None, None]; d];
        for &j in frame.graph.row(i) {
            let dx = frame.delta(i, j as usize);
            let norm = dx.iter().fold(T::ZERO, |s, &v| s + v * v).sqrt() + tiny;
            for a in 0..d {
                if (dx[a] / norm).abs() < cos_threshold {
                    continue;
                }
                let slot = &mut side[a][usize::from(dx[a] < T::ZERO)];
                if slot.is_none_or(|(best, _)| dx[a].abs() < best) {
                    *slot = Some((dx[a].abs(), j as usize));
                }
            }
        }
        for a in 0..d {
            if let [Some((hp, jp)), Some((hn, jn))] = side[a] {
                // Neighbors are never equidistant in a swarm. The equal-spacing
                // stencil would leak the gradient as f' (hp - hn) / h², which
                // diverges under refinement; this one is exact for a quadratic.
                // Both spacings are positive: a coincident neighbor is outside
                // every cone.
                h[a * d + a] = T::from_f64(2.)
                    * ((values[jp] - values[i]) / hp + (values[jn] - values[i]) / hn)
                    / (hp + hn);
            }
        }
        h
    });
    Ok(rows.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObservationBatch, TensorBatch, tessellation::graph::NeighborGraph};

    #[test]
    fn unequal_spacings_do_not_leak_the_gradient() {
        // Center with axis neighbors at +0.1 / -0.2 and +0.05 / -0.3.
        let x = vec![0., 0., 0.1, 0., -0.2, 0., 0., 0.05, 0., -0.3];
        let positions = TensorBatch::vectors(5, 2, x.clone()).unwrap();
        let observations = ObservationBatch::positions(positions.clone());
        let graph = NeighborGraph::from_undirected(5, &[[0, 1], [0, 2], [0, 3], [0, 4]]).unwrap();
        let eligible = [true; 5];
        let frame = GeometryFrame::new(&positions, &[0, 1], &eligible, &graph, &observations, None)
            .unwrap();
        let field = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> {
            x.chunks_exact(2).map(|p| f(p[0], p[1])).collect()
        };
        let linear = estimate(
            &frame,
            &field(&|x, y| 3. * x - 7. * y + 2.),
            false,
            Parallelism::Serial,
        )
        .unwrap();
        assert!(linear[..4].iter().all(|h| h.abs() < 1e-12), "{linear:?}");
        let quadratic = estimate(
            &frame,
            &field(&|x, y| 1.5 * x * x - 2. * y * y + 3. * x - 7. * y),
            false,
            Parallelism::Serial,
        )
        .unwrap();
        assert!((quadratic[0] - 3.).abs() < 1e-12 && (quadratic[3] + 4.).abs() < 1e-12);
        assert_eq!((quadratic[1], quadratic[2]), (0., 0.));
    }
}
