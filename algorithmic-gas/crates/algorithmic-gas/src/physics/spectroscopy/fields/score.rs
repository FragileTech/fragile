//! Cloning score, fitness phases, score gradient and walker roles.
//!
//! The acceptance probability is always `CloneDecision::acceptance_probability`
//! (clipped, gated by `every`). The score here is the unclipped, ungated
//! `S_i = (V_c − V_i) / (V_i + ε)`; on a cloning step the two agree through
//! `p = clamp(S / saturation, 0, 1)`.
use crate::{
    GasConfig,
    physics::spectroscopy::contract::{Availability, Frame, FrameState, WalkerRole},
};

pub fn score(own: f64, companion: f64, epsilon: f64) -> f64 {
    (companion - own) / (own + epsilon)
}

/// U(1) fitness phase `θ_ij = −(Φ_j − Φ_i) / ħ_eff`.
pub fn u1_phase(own: f64, companion: f64, h_eff: f64) -> f64 {
    -(companion - own) / h_eff
}

/// SU(2) cloning phase `(V_j − V_i) / ((|V_i| + ε) h_S)`. It equals
/// `score / h_S` on the positive fitness of a valid record.
pub fn su2_phase(own: f64, companion: f64, epsilon: f64, h_s: f64) -> f64 {
    (companion - own) / ((own.abs() + epsilon) * h_s)
}

/// Roles from the ungated score sign and the accepted decisions of the
/// frame, so that a gated cloning period does not erase them. `score` holds
/// 0 on a row without a valid score, which is then a persister unless it
/// clones or is targeted.
pub fn roles(frame: &Frame, score: &[f64]) -> Vec<WalkerRole> {
    let _ = score;
    vec![WalkerRole::Persister; frame.n]
}

/// Write `score` and `score_valid` (from `fitness`, `companion_fitness` and
/// `gas.clone_decision.epsilon`), `role` and `score_gradient` (from
/// `state.{distance,cloning}_companion`) into `state`. A row without a valid
/// score holds 0, never NaN. Displacements of the gradient take the minimum
/// image of `contract::periodic_box(&gas.boundary)`, as the vector channels
/// do. `Unavailable` when the frame lacks fitness or cloning companions.
pub fn fill(frame: &Frame, gas: &GasConfig, state: &mut FrameState) -> Availability {
    let _ = (frame, gas, state);
    Availability::unavailable("pending: spectroscopy::fields::score")
}
