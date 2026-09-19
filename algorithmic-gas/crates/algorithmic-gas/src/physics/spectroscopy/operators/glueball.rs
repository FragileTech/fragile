//! Plaquette operators of `Π_ijk = q_ij q_jk q_ki` on companion triplets and
//! the viscous force norm on sites, optionally projected on a momentum mode.
//! `Π` is invariant under per-site rephasing and a common unitary rotation;
//! the colour encoding is not locally covariant under more.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Glueball`: plaquette observables on triplets, `ForceNorm`
/// on sites. Requires `Color`; a momentum projection adds `PeriodicBox` and
/// is `GasError::Capability` when its axis is outside the position
/// coordinates. All arms are `ExchangeParity::Even`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::glueball".into(),
    ))
}

/// Invalid colours mask a plaquette; the phase observables mask `Π = 0`.
/// `ForceNorm` reads `state.force` where `state.force_valid`, whatever the
/// colour validity. The momentum weight reads the anchor position.
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
