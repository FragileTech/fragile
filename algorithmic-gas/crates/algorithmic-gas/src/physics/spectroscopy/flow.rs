//! Graph smoothing diagnostic: convex neighbour averaging of the colour field
//! on the recorded graph. It makes no length-scale claim: the colour is a site
//! field without links, a sweep is not covariant under a rephasing of single
//! walkers, the sweep count times the mixing weight has no unit, and the
//! roughness is bounded by 2 and need not decrease, so no threshold crossing
//! of the curve sets a scale.
use super::{config::FlowConfig, contract::FrameState};
use crate::{
    Result,
    error::require,
    physics::qft::math::{C, dot},
    tessellation::NeighborGraph,
};

/// Mean of `1 − Re c_i† c_j` over the undirected edges between carriers, in
/// CSR slot order; `edges` is their number.
fn roughness(graph: &NeighborGraph, carrier: &[bool], color: &[C], d: usize, edges: usize) -> f64 {
    let mut sum = 0.;
    for i in (0..graph.nodes()).filter(|&i| carrier[i]) {
        for &j in graph.row(i) {
            let j = j as usize;
            if i < j && carrier[j] {
                sum += 1. - dot(&color[i * d..(i + 1) * d], &color[j * d..(j + 1) * d]).re;
            }
        }
    }
    sum / edges as f64
}

/// Mean neighbour mismatch `1 − Re q_ij` over edges between valid colours
/// after `0, 1, …, config.steps` smoothing steps; `None` when no edge is valid.
/// The recorded graph is post-clone: walkers with `state.cloned` are invalid.
/// A step replaces every valid colour at once by the renormalized blend
/// `(1 − step_size) c_i + step_size m_i`, `m_i` the uniform mean over its valid
/// graph neighbours; a colour without a valid neighbour, or whose blend has
/// norm at most `1e-12`, stays. Valid colours have unit norm.
pub fn smooth(
    state: &FrameState,
    graph: &NeighborGraph,
    config: &FlowConfig,
) -> Result<Vec<Option<f64>>> {
    config.validate()?;
    let (n, d) = (state.n, state.d);
    require(
        graph.nodes() == n
            && state.color.len() == n * d
            && state.color_valid.len() == n
            && state.cloned.len() == n,
        "graph smoothing needs one colour, validity and clone flag per graph node",
    )?;
    graph.validate()?;
    let carrier: Vec<bool> = (0..n)
        .map(|i| state.color_valid[i] && !state.cloned[i])
        .collect();
    let edges = (0..n)
        .filter(|&i| carrier[i])
        .flat_map(|i| graph.row(i).iter().map(move |&j| (i, j as usize)))
        .filter(|&(i, j)| i < j && carrier[j])
        .count();
    if edges == 0 {
        return Ok(vec![None; config.steps + 1]);
    }
    let a = config.step_size;
    let mut color = state.color.clone();
    let mut next = color.clone();
    let mut mean = vec![C::ZERO; d];
    let mut curve = vec![roughness(graph, &carrier, &color, d, edges)];
    for _ in 0..config.steps {
        for i in (0..n).filter(|&i| carrier[i]) {
            mean.fill(C::ZERO);
            let mut count = 0usize;
            for j in graph.row(i).iter().map(|&j| j as usize) {
                if carrier[j] {
                    count += 1;
                    for (m, &c) in mean.iter_mut().zip(&color[j * d..(j + 1) * d]) {
                        *m = *m + c;
                    }
                }
            }
            if count == 0 {
                continue;
            }
            let own = &color[i * d..(i + 1) * d];
            for ((m, &c), u) in mean.iter().zip(own).zip(&mut next[i * d..(i + 1) * d]) {
                *u = c * (1. - a) + *m / count as f64 * a;
            }
            let blend = &mut next[i * d..(i + 1) * d];
            let norm = blend.iter().map(|c| c.abs2()).sum::<f64>().sqrt();
            if norm > 1e-12 {
                blend.iter_mut().for_each(|c| *c = *c / norm);
            } else {
                blend.copy_from_slice(own);
            }
        }
        color.copy_from_slice(&next);
        curve.push(roughness(graph, &carrier, &color, d, edges));
    }
    require(
        curve.iter().all(|r| r.is_finite()),
        "colour roughness overflow",
    )?;
    Ok(curve.into_iter().map(Some).collect())
}
