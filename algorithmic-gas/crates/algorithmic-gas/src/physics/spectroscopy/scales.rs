//! Geodesic scales on the recorded graph: calibration, gating and smearing of
//! multiscale channel copies.
use super::{
    config::{EdgeLength, ScaleSelection},
    contract::{Element, FrameState, Topology},
};
use crate::{
    GasError, Result,
    physics::qft::math::C,
    tessellation::{GraphSnapshot, Parallelism},
};

/// Shortest-path distances `[n, n]` over the chosen edge length, truncated at
/// `radius` (`f64::INFINITY` beyond). Periodic wraps are those of the snapshot.
/// The dense table is admitted against `max_bytes` by
/// `numerics::shortest_paths::all_pairs`; the accumulator passes what remains
/// of `MeasurementBudget::max_bytes`.
pub fn distances(
    graph: &GraphSnapshot<f64>,
    length: EdgeLength,
    radius: f64,
    max_bytes: usize,
    par: Parallelism,
) -> Result<Vec<f64>> {
    let _ = (graph, length, radius, max_bytes, par);
    Err(GasError::Capability("pending: spectroscopy::scales".into()))
}

/// Ascending scales from warm-up samples of finite pair distances.
pub fn calibrate(samples: &[f64], selection: &ScaleSelection) -> Result<Vec<f64>> {
    selection.validate()?;
    let _ = samples;
    Err(GasError::Capability("pending: spectroscopy::scales".into()))
}

/// Largest pairwise distance among the walkers of an element; 0 for a site.
pub fn diameter(element: &Element, distances: &[f64], n: usize) -> f64 {
    let _ = (element, distances, n);
    f64::INFINITY
}

/// `[elements]`: the element's diameter is within `scale`. The recorded graph
/// is post-clone and places a cloned walker on its donor, so a pair or triplet
/// holding a walker with `cloned[w]` (`[n]`, `FrameState::cloned` of the
/// element's source frame) is outside every scale. A site has diameter 0 and
/// passes.
pub fn gate(
    topology: &Topology,
    distances: &[f64],
    n: usize,
    scale: f64,
    cloned: &[bool],
) -> Vec<bool> {
    let _ = (distances, n, scale, cloned);
    vec![false; topology.elements.len()]
}

/// Colours averaged with the kernel `exp(−d²/2 scale²)` over valid walkers and
/// renormalized; `[n, d]` and the validity of each smeared colour. A walker
/// with `state.cloned` sits on its donor in the post-clone graph: it neither
/// contributes to a smeared colour nor receives one (`valid` is false).
pub fn smear(state: &FrameState, distances: &[f64], scale: f64) -> (Vec<C>, Vec<bool>) {
    let _ = (distances, scale);
    (vec![C::ZERO; state.color.len()], vec![false; state.n])
}
