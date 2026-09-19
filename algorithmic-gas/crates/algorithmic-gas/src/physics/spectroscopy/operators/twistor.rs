//! Twistor observables of the two effective edge twistors of a companion
//! triplet: the spinor contraction `τ` and the Pauli-vector bilinear `W`. The
//! edge vector `(Δt, Δx)` mixes units with `c = 1` implicit.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Twistor` on triplets, three dimensions only. Requires
/// `Velocities` and a time step in the capabilities; no formula reads the
/// colour. `Vector` and `Axial` keep 3 components, `Tensor` the 5 spin-2
/// components, the others 1. Every arm fixes `normalization: Some(FixedN)`,
/// reports no spatial parity and states the velocity scale in its note.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::twistor".into(),
    ))
}

/// A null edge masks the element. The dominant-column tie selects column 0.
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
