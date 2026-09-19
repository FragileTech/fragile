//! Antisymmetric colour bilinear `Re(c̄_i × c_j)` on pairs, the tensor channel
//! of the reference code. Volume II does not define it: the descriptor has an
//! empty `book_label`. The RMS envelope is a diagnostic that never enters a
//! correlator.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Tensor` on a distance or cloning pair, three dimensions
/// only. Requires `Color`. `Components` has the three components `(01)`,
/// `(02)`, `(12)` and is `ExchangeParity::Odd`; `Envelope` has one component,
/// is `Even` and has `correlatable: false`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::tensor".into(),
    ))
}

/// `Envelope` writes the sum of squared components; the consumer takes the
/// square root of its frame mean.
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
