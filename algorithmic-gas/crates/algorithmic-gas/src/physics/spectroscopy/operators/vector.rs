//! Vector and axial pair operators `Re q_ij · D` and `Im q_ij · D` for a
//! displacement `D`. Components are kept and contracted in the correlator;
//! the projection applies in every displacement mode.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::ChannelSpec,
        contract::{Element, ElementKind, FrameState, OperatorContext, Signature},
    },
};

/// `ChannelSpec::Vector` on a distance or cloning pair, `d` components.
/// Requires `Color`; `ScoreGradient` and every projection other than `Full`
/// add `Fitness` and `CloningCompanions` and are `Mixed`. The standard vector
/// is `ExchangeParity::Odd`, the standard axial `Even`. `ColorGamma` needs
/// `d ≥ 3` (`GasError::Capability` below) and swaps the parities: its vector
/// is even, its axial odd.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    let _ = (spec, kind, context);
    Err(GasError::Capability(
        "pending: spectroscopy::operators::vector".into(),
    ))
}

/// Displacements use the minimum image of a periodic box. Projections are
/// along or across `state.score_gradient` of the anchor and mask a zero
/// gradient; a zero displacement masks `Unit`.
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
