//! Radius-truncated shortest paths on a CSR neighbor graph.
use crate::{
    GasError, Result,
    tessellation::{NeighborGraph, Parallelism},
};

/// Dijkstra distances from `source` along edges of CSR-slot lengths `length`;
/// `f64::INFINITY` beyond `radius` or when unreachable. `[nodes]`.
pub fn from_source(
    graph: &NeighborGraph,
    length: &[f64],
    source: usize,
    radius: f64,
) -> Result<Vec<f64>> {
    let _ = (graph, length, source, radius);
    Err(GasError::Capability(
        "pending: numerics::shortest_paths".into(),
    ))
}

/// All sources through `tessellation::par::map_indexed`; `[nodes, nodes]`.
/// The dense table is admitted against `max_bytes` with
/// `memory::enforce(nodes · nodes · 8, max_bytes)` before it is allocated.
/// Lengths must be finite and nonnegative, one per CSR slot.
pub fn all_pairs(
    graph: &NeighborGraph,
    length: &[f64],
    radius: f64,
    max_bytes: usize,
    par: Parallelism,
) -> Result<Vec<f64>> {
    let _ = (graph, length, radius, max_bytes, par);
    Err(GasError::Capability(
        "pending: numerics::shortest_paths".into(),
    ))
}
