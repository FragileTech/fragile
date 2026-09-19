//! Colour singlet pair operators of `q_ij = c_i† c_j`: `Re q` (exchange even,
//! parity even) and `Im q` (exchange odd, parity odd), with the score-directed,
//! score-weighted, `γ5`-diagonal and `|q|²` arms.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Meson` on a distance or cloning pair. Requires `Color`; the
/// score modes add `Fitness` and `CloningCompanions`. The standard
/// pseudoscalar is `ExchangeParity::Odd`, the directed arms are `Mixed`.
/// `Pseudoscalar` with `Abs2` duplicates the scalar `|q|²` and is
/// `GasError::Capability`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::meson".into(),
    ))
}

/// Invalid colours mask the element. A directed arm orients the pair from
/// `state.score` and masks a tie or a missing score.
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
