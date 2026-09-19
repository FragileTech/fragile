//! Colour state `c_a = F_a e^{i κ v_a} / max(|F|, δ)` of the viscous force under
//! the alignment arms, and the colour determinant. Only `κ` is observable:
//! mass, length and action scale enter through it alone.
use crate::physics::{
    qft::math::C,
    spectroscopy::{
        config::{ColorAlignment, StepAlignment},
        contract::{Availability, Frame, FrameState},
    },
};

/// Colour of one walker into `out` (`[d]`); false, with `out` zeroed, when
/// `|F| ≤ threshold` or an input is not finite. Agrees bit for bit with the
/// colour of the recorded-colour lecture experiment under `MatchedKick { B1 }`.
pub fn color_vector(
    force: &[f64],
    velocity: &[f64],
    kappa: f64,
    threshold: f64,
    out: &mut [C],
) -> bool {
    let _ = (force, velocity, kappa, threshold);
    out.fill(C::ZERO);
    false
}

/// `det[a, b, c]` of three colour vectors in three dimensions.
pub fn det3(a: &[C], b: &[C], c: &[C]) -> C {
    let _ = (a, b, c);
    C::ZERO
}

/// Colour of every walker from the viscous-force records `alignment` names,
/// with `force`, `force_valid` and `phase_velocity` of the same records.
/// `PrecedingKick` and `PrecedingForce` read the B2 record of `previous` and
/// mask row `i` iff its `generation[i] != frame.generation[i]`: that record is
/// post-clone, so walkers cloned at `t − 1` stay valid. `MatchedKick` and
/// `ReferenceOffset` read a stage of `frame` and mask `frame.cloned[i]`.
/// Ineligible rows and rows the record does not cover are invalid.
/// `Unavailable` without the record an arm reads (no `previous`, no B stage).
pub(super) fn viscous_force(
    alignment: ColorAlignment,
    threshold: f64,
    previous: Option<&Frame>,
    frame: &Frame,
    kappa: f64,
    state: &mut FrameState,
) -> Availability {
    let _ = (alignment, threshold, previous, frame, kappa, state);
    Availability::unavailable("pending: spectroscopy::fields::color")
}

/// Colour from `recorded_color`: its `force` as amplitude, its `velocity` as
/// phase. `StepAlignment::Preceding` reads `previous.recorded_color` with the
/// generation mask, `Matched` reads `frame.recorded_color` and masks
/// `frame.cloned[i]`.
pub(super) fn recorded_field(
    alignment: StepAlignment,
    threshold: f64,
    previous: Option<&Frame>,
    frame: &Frame,
    kappa: f64,
    state: &mut FrameState,
) -> Availability {
    let _ = (alignment, threshold, previous, frame, kappa, state);
    Availability::unavailable("pending: spectroscopy::fields::color")
}
