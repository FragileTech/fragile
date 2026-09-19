//! Antisymmetric colour bilinear `Re(c̄_i × c_j)` on pairs, the tensor channel
//! of the reference code: three components dual to a vector, even under
//! inversion and odd under exchange, with no spin-two part. Its frame mean
//! cancels on a mutual pairing; the source-frozen propagator, whose summand is
//! even under exchange, is the observable. The envelope is a diagnostic that
//! never enters a correlator.
use crate::{
    GasError, Result,
    physics::{
        qft::math::C,
        spectroscopy::{
            config::{ChannelSpec, TensorMode},
            contract::{
                Descriptor, Element, ElementKind, ExchangeParity, FrameState, OperatorContext,
                Record, Requirements, Signature, SpatialParity,
            },
        },
    },
};

const NOTE: &str = "antisymmetric colour bilinear of the reference code, dual to the vector \
                    Re(conj(c_i) x c_j): three components and no spin-two part; invariant under a \
                    common phase and covariant under a common real rotation of the colour \
                    components, not invariant under SU(3) or under per-walker phases";

/// `ChannelSpec::Tensor` on a distance or cloning pair, three dimensions
/// only. Requires `Color`. `Components` has the three components `(01)`,
/// `(02)`, `(12)` and is `ExchangeParity::Odd`; `Envelope` has one component,
/// is `Even` and has `correlatable: false`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    _context: &OperatorContext<'_>,
) -> Result<Signature> {
    let ChannelSpec::Tensor { mode } = spec else {
        return Err(GasError::Capability("not a tensor channel".into()));
    };
    if kind.arity() != 2 {
        return Err(GasError::Capability(
            "tensor channels are measured on pairs".into(),
        ));
    }
    let requires = Requirements::new([Record::Color]).in_dimension(3);
    let signature = match mode {
        TensorMode::Components => Signature {
            descriptor: Descriptor {
                definition: concat!(
                    r"O^{\mu\nu}_{ij} = \operatorname{Re}(\bar c_i^{\mu} c_j^{\nu})",
                    r" - \operatorname{Re}(\bar c_i^{\nu} c_j^{\mu}),",
                    r"\quad (\mu\nu) = (01), (02), (12)"
                )
                .into(),
                book_label: "prop-qft-color-gamma-parities".into(),
                spatial_parity: Some(SpatialParity::Even),
                note: format!(
                    "{NOTE}; exchange odd: on a pairing that is not mutual its frame mean \
                     measures the asymmetry of the companion law, so the source-frozen \
                     propagator is the observable"
                ),
            },
            ..Signature::new(requires, 3, ExchangeParity::Odd)
        },
        TensorMode::Envelope => Signature {
            correlatable: false,
            descriptor: Descriptor {
                definition: concat!(
                    r"\sum_{\mu<\nu} (O^{\mu\nu}_{ij})^2",
                    r" = |\operatorname{Re}(\bar c_i \times c_j)|^2"
                )
                .into(),
                book_label: String::new(),
                spatial_parity: Some(SpatialParity::Even),
                note: format!(
                    "{NOTE}; squared norm over the kept components, at most 1 for unit colours; \
                     the envelope is the square root of its frame mean, an equal-time magnitude"
                ),
            },
            ..Signature::new(requires, 1, ExchangeParity::Even)
        },
    };
    Ok(signature)
}

/// Valid finite colour of `walker` in three dimensions.
pub(super) fn color(state: &FrameState, walker: usize) -> Option<&[C]> {
    if state.d != 3 || state.color_valid.get(walker) != Some(&true) {
        return None;
    }
    state
        .color
        .get(walker * 3..walker * 3 + 3)
        .filter(|c| c.iter().all(|z| z.re.is_finite() && z.im.is_finite()))
}

/// `Re(ā^μ b^ν) − Re(ā^ν b^μ)`. Each bracket adds the real product before the
/// imaginary one, so that exchanging `a` and `b` swaps the two brackets bit
/// for bit and the value changes sign exactly.
fn wedge(a: &[C], b: &[C], mu: usize, nu: usize) -> f64 {
    (a[mu].re * b[nu].re + a[mu].im * b[nu].im) - (a[nu].re * b[mu].re + a[nu].im * b[mu].im)
}

/// `Envelope` writes the sum of squared components; the consumer takes the
/// square root of its frame mean.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    _context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let ChannelSpec::Tensor { mode } = spec else {
        return false;
    };
    let (i, j) = (element.walkers[0] as usize, element.walkers[1] as usize);
    if i == j {
        return false;
    }
    let (Some(a), Some(b)) = (color(state, i), color(state, j)) else {
        return false;
    };
    let components = [wedge(a, b, 0, 1), wedge(a, b, 0, 2), wedge(a, b, 1, 2)];
    match mode {
        TensorMode::Components => out[..3].copy_from_slice(&components),
        TensorMode::Envelope => out[0] = components.iter().map(|o| o * o).sum(),
    }
    true
}
