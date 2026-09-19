//! Baryon operators of the colour determinant `b_ijk = det[c_i, c_j, c_k]` on
//! companion triplets. Phase-invariant modes are labelled as such; `|b|` is
//! not a Volume II mode.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Baryon` on triplets, three dimensions only. Requires
/// `Color`; `ScoreOrdered` adds `Fitness` and `CloningCompanions`. `Real`,
/// `Imag` and `Complex` (two components) are odd under relabelling of the
/// companions, so `ExchangeParity::Odd`; `Abs2`, `Abs`, `ScoreOrdered` and
/// `FluxWeighted` are `Even`. `Abs`, `ScoreOrdered` and `FluxWeighted` carry
/// an empty `book_label`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::baryon".into(),
    ))
}

/// Invalid colours mask the element. `ScoreOrdered` orders the walkers by
/// `state.score`, ties keeping the sampled order.
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
