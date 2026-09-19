//! Chirality of the walker roles on full-frame series with masks: a gappy
//! series of cloning frames is never correlated as if it were uniform.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Chirality`: `Chi` and `LeftFraction` on sites, so that every
/// walker of `L_t` counts whatever its cloning entry is; `LeftRightCoupling`
/// on cloning pairs. Requires `Fitness`, `CloningCompanions` and `ClonePlan`.
/// Every arm fixes `normalization: Some(FixedN)` and is
/// `ExchangeParity::Even`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::chirality".into(),
    ))
}

/// Reads `state.role`; a walker without a role masks the element.
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
