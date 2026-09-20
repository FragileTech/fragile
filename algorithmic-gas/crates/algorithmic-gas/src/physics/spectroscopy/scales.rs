//! Geodesic scales on the recorded graph: calibration, gating and smearing of
//! multiscale channel copies. A scale is a shortest-path length over recorded
//! edge lengths, a measurement convention. It lies between the Euclidean graph
//! distance times the smallest and the largest recorded edge-length ratio;
//! no convergence to a distance on the fitness manifold is claimed.
use super::{
    config::{EdgeLength, ScaleSelection},
    contract::{Element, FrameState, Topology},
};
use crate::{
    GasError, Result,
    error::require,
    physics::{numerics::shortest_paths, qft::math::C},
    tessellation::{GraphSnapshot, Parallelism},
};

/// Total number of pair distances a warm-up contributes to `calibrate`; a
/// frame contributes `CALIBRATION_SAMPLES / max(warmup, 1)`, at least one.
pub const CALIBRATION_SAMPLES: usize = 500_000;
/// The smearing kernel ends at `SMEAR_CUTOFF · scale`, where its weight is
/// `exp(−32)`. A distance table truncated at `SMEAR_CUTOFF` times the largest
/// scale therefore smears exactly as the untruncated one.
pub const SMEAR_CUTOFF: f64 = 8.;

/// Shortest-path distances `[n, n]` over the chosen edge length, truncated at
/// `radius` (`f64::INFINITY` beyond). Periodic wraps are those of the snapshot.
/// The dense table is admitted against `max_bytes` by
/// `numerics::shortest_paths::all_pairs`; the accumulator passes what remains
/// of `MeasurementBudget::max_bytes`. Warm-up frames use an infinite radius,
/// gated frames the largest scale and smeared frames `SMEAR_CUTOFF` times it.
///
/// A snapshot the recording carried over from an earlier step is declined: the
/// walkers of this frame moved away from the graph that would gate and smear
/// them, so the table is not this frame's geometry. Scales need a geometry
/// schedule that tessellates every step.
pub fn distances(
    graph: &GraphSnapshot<f64>,
    length: EdgeLength,
    radius: f64,
    max_bytes: usize,
    par: Parallelism,
) -> Result<Vec<f64>> {
    if graph.stale_steps > 0 {
        return Err(GasError::Capability(
            "scale gating needs a tessellation refreshed every step".into(),
        ));
    }
    let length = match length {
        EdgeLength::Geodesic => &graph.geodesic_length,
        EdgeLength::Euclidean => &graph.euclidean_length,
    };
    shortest_paths::all_pairs(&graph.graph, length, radius, max_bytes, par)
}

/// At most `cap` finite positive distances of the pairs `i < j` of a table
/// `[n, n]`, in row-major order at the smallest constant stride that spans
/// the whole triangle, so no tail of walkers is left out. The choice is a
/// function of the frame alone, so a warm-up sample does not depend on how
/// the run is chunked. Coincident walkers at distance zero are no sample.
///
/// A table whose flattened pair sequence carries an arithmetic period
/// commensurate with the stride would be read at one phase of that period;
/// shortest-path lengths carry no such period, and on geodesic tables the
/// quantiles of the strided sample deviate from the full ones by about as
/// much as those of an independent uniform subsample of the same size, which
/// is ordinary quantile sampling noise rather than aliasing.
pub fn pair_samples(distances: &[f64], n: usize, cap: usize) -> Vec<f64> {
    debug_assert_eq!(distances.len(), n * n);
    let pairs = || {
        (0..n)
            .flat_map(move |i| distances[i * n + i + 1..(i + 1) * n].iter().copied())
            .filter(|d| d.is_finite() && *d > 0.)
    };
    let stride = pairs().count().div_ceil(cap.max(1)).max(1);
    pairs().step_by(stride).take(cap).collect()
}

/// Linear-interpolation quantile of an ascending nonempty sample.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = p * (sorted.len() - 1) as f64;
    let k = h.floor() as usize;
    let next = sorted[(k + 1).min(sorted.len() - 1)];
    sorted[k] + (h - k as f64) * (next - sorted[k])
}
/// Ascending scales from warm-up samples of finite pair distances. Only the
/// warm-up contributes, so no measured frame enters its own scales.
/// `Quantiles` is the geometric ladder `q_low (q_high / q_low)^(k / (count − 1))`
/// between two quantiles of the finite positive samples, with both ends exact
/// and `sqrt(q_low q_high)` for a single scale. Equal rungs are merged, never
/// separated by an invented offset, so the ladder can be shorter than `count`.
/// `Fixed` needs no sample.
pub fn calibrate(samples: &[f64], selection: &ScaleSelection) -> Result<Vec<f64>> {
    selection.validate()?;
    let (count, low, high) = match selection {
        ScaleSelection::Fixed { values } => return Ok(values.clone()),
        ScaleSelection::Quantiles { count, low, high } => (*count, *low, *high),
    };
    let mut sorted: Vec<f64> = samples
        .iter()
        .copied()
        .filter(|d| d.is_finite() && *d > 0.)
        .collect();
    if sorted.is_empty() {
        return Err(GasError::Numerical(
            "no finite pair distance during warm-up".into(),
        ));
    }
    sorted.sort_by(f64::total_cmp);
    let (bottom, top) = (quantile(&sorted, low), quantile(&sorted, high));
    let mut scales: Vec<f64> = Vec::with_capacity(count);
    for k in 0..count {
        let scale = if count == 1 {
            (bottom * top).sqrt()
        } else if k == 0 {
            bottom
        } else if k + 1 == count {
            top
        } else {
            (bottom * (top / bottom).powf(k as f64 / (count - 1) as f64)).min(top)
        };
        if scales.last().is_none_or(|&last| scale > last) {
            scales.push(scale);
        }
    }
    require(
        scales.iter().all(|s| s.is_finite() && *s > 0.),
        "scale ladder overflow",
    )?;
    Ok(scales)
}

/// Largest pairwise distance among the walkers of an element; 0 for a site.
/// `distances` is `[n, n]` and the element names walkers below `n`
/// (`Topology::validate`). A NaN distance makes the diameter NaN.
pub fn diameter(element: &Element, distances: &[f64], n: usize) -> f64 {
    debug_assert_eq!(distances.len(), n * n);
    let [i, j, k] = element.walkers.map(|w| w as usize);
    match element.kind.arity() {
        1 => 0.,
        2 => distances[i * n + j],
        // `f64::max` would drop a NaN side.
        _ => [distances[i * n + k], distances[j * n + k]]
            .into_iter()
            .fold(distances[i * n + j], |m, d| {
                if d > m || d.is_nan() { d } else { m }
            }),
    }
}

/// `[elements]`: the element's diameter is within `scale` (inclusive) and
/// finite. The recorded graph is post-clone and places a cloned walker on its
/// donor, so a pair or triplet holding a walker with `cloned[w]` (`[n]`,
/// `FrameState::cloned` of the element's source frame) is outside every
/// scale. A site has diameter 0 and passes. The gate only removes elements of
/// a topology; on a bit-symmetric table both directions of a mutual pair
/// share their fate.
pub fn gate(
    topology: &Topology,
    distances: &[f64],
    n: usize,
    scale: f64,
    cloned: &[bool],
) -> Vec<bool> {
    debug_assert_eq!(cloned.len(), n);
    topology
        .elements
        .iter()
        .map(|element| {
            let arity = element.kind.arity() as usize;
            let reach = diameter(element, distances, n);
            arity == 1
                || (reach.is_finite()
                    && reach <= scale
                    && element.walkers[..arity]
                        .iter()
                        .all(|&w| !cloned[w as usize]))
        })
        .collect()
}

/// Colours averaged with the kernel `exp(−d²/2 scale²)` over valid walkers and
/// renormalized; `[n, d]` and the validity of each smeared colour. A walker
/// with `state.cloned` sits on its donor in the post-clone graph: it neither
/// contributes to a smeared colour nor receives one (`valid` is false).
/// The walker itself enters with weight 1, so a vanishing scale is the
/// identity, and the kernel ends at `SMEAR_CUTOFF · scale`. The result has
/// unit norm, which makes any normalization of the kernel irrelevant; a sum
/// of norm at most `1e-12` is invalid. `scale` is finite and positive. There
/// is no parallel transport: the smeared field is covariant under a global
/// rotation of the colour frame only, not under a rephasing of each walker.
pub fn smear(state: &FrameState, distances: &[f64], scale: f64) -> (Vec<C>, Vec<bool>) {
    let (n, d) = (state.n, state.d);
    debug_assert!(distances.len() == n * n && state.color.len() == n * d && scale > 0.);
    debug_assert!(state.color_valid.len() == n && state.cloned.len() == n);
    let carrier: Vec<bool> = (0..n)
        .map(|i| state.color_valid[i] && !state.cloned[i])
        .collect();
    let mut color = vec![C::ZERO; n * d];
    let mut valid = vec![false; n];
    for i in (0..n).filter(|&i| carrier[i]) {
        let sum = &mut color[i * d..(i + 1) * d];
        for j in (0..n).filter(|&j| carrier[j]) {
            let reach = distances[i * n + j];
            if reach.is_finite() && reach <= SMEAR_CUTOFF * scale {
                let z = reach / scale;
                let weight = (-0.5 * z * z).exp();
                for (s, &c) in sum.iter_mut().zip(&state.color[j * d..(j + 1) * d]) {
                    *s = *s + c * weight;
                }
            }
        }
        let norm = sum.iter().map(|c| c.abs2()).sum::<f64>().sqrt();
        if norm > 1e-12 {
            sum.iter_mut().for_each(|c| *c = *c / norm);
            valid[i] = true;
        } else {
            sum.fill(C::ZERO);
        }
    }
    (color, valid)
}
