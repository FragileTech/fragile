//! Graph smoothing diagnostic: convex neighbour averaging of the colour field
//! on the recorded graph. It makes no length-scale claim.
use super::{config::FlowConfig, contract::FrameState};
use crate::{GasError, Result, tessellation::NeighborGraph};

/// Mean neighbour mismatch `1 − Re q_ij` over edges between valid colours
/// after `0, 1, …, config.steps` smoothing steps; `None` when no edge is valid.
/// The recorded graph is post-clone: walkers with `state.cloned` are invalid.
pub fn smooth(
    state: &FrameState,
    graph: &NeighborGraph,
    config: &FlowConfig,
) -> Result<Vec<Option<f64>>> {
    config.validate()?;
    let _ = (state, graph);
    Err(GasError::Capability("pending: spectroscopy::flow".into()))
}
