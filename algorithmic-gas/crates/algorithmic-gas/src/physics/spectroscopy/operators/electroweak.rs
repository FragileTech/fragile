//! U(1) fitness phases on distance pairs, SU(2) cloning phases on cloning
//! pairs, their mixed triplet product and the symmetry-breaking site scalars.
//! Interaction ranges are never the clone regulariser and self-companions are
//! masked.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::{U1, Su2, ElectroweakMixed, FitnessPhase, CloneIndicator,
/// ParityVelocity}`. U(1) requires `Fitness`; SU(2) adds `ClonePlan`. A
/// `Range::FromKernel` whose role has no Gaussian kernel width in the
/// capabilities is `GasError::Capability`. Complex values have the two
/// components `(Re, Im)` and are `Mixed`: the imaginary part of a U(1) phase
/// is exchange odd. `ElectroweakMixed` sets `degenerate`; `CloneIndicator`
/// fixes `normalization: Some(FixedN)`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::electroweak".into(),
    ))
}

/// The doublet modes are two-hop, `a_i ± a_{k(i)}`, the second entry built
/// on `state.cloning_companion[k(i)]` and masked at `NO_COMPANION`. Every hop
/// phase is `su2_phase(F_a, F_b, ε_clone, h_S)` from `state.fitness` of the two
/// walkers of the hop at the evaluated time, with `ε_clone =
/// context.gas.clone_decision.epsilon` and `h_S =
/// context.measurement.electroweak.h_s.unwrap_or(h_eff)`; `state.score` is
/// never read, because at a sink it belongs to the sink-time companion and
/// not to the frozen pair. `directed` replaces every hop phase by its absolute
/// value; a zero phase gives the value 1 and masks nothing. The amplitude
/// distance follows `ElectroweakScales::distance`.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let _ = (spec, element, state, context, out);
    false
}
