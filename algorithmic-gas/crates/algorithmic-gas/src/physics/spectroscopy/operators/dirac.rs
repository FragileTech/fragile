//! Dirac bilinears `ψ̄_i Γ ψ_j` of the spinor lift of the colour state, with
//! the chiral projectors, role classes and electroweak phase links. Channels
//! are labelled by their algebra and verified parities, never by a `J^PC` name.
//! The lift is a convention of the reference code and is neither norm
//! preserving nor rotation equivariant; the descriptor note says so.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Dirac` on a distance or cloning pair, three dimensions
/// only. Requires `Color`; a role class adds `Fitness`, `CloningCompanions`
/// and `ClonePlan`, a phase link the records of its phase. `Vector`, `Axial`,
/// `Tensor` and `TensorTime` keep 3 components; `Pseudoscalar` keeps the
/// imaginary part. Parities come from the verified table of the family.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::dirac".into(),
    ))
}

/// A colour with a vanishing real or imaginary part has no lift and masks
/// the element, as does a walker outside the role class.
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
