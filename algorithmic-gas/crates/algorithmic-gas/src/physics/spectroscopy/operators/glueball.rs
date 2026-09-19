//! Plaquette operators of `Π_ijk = q_ij q_jk q_ki` on companion triplets and
//! the viscous force norm on sites, optionally projected on a momentum mode.
//! `Π` is invariant under per-site rephasing and a common unitary rotation;
//! the colour encoding is not locally covariant under more.
use crate::{
    GasError, Result,
    physics::{
        qft::math::{C, dot},
        spectroscopy::{
            config::{ChannelSpec, GlueballObservable, Momentum, MomentumPhase},
            contract::{
                Descriptor, Element, ElementKind, ExchangeParity, FrameState, OperatorContext,
                Record, Requirements, Signature, SpatialParity, periodic_box,
            },
        },
    },
};
use std::f64::consts::TAU;

/// `ChannelSpec::Glueball`: plaquette observables on triplets, `ForceNorm`
/// on sites. Requires `Color`; a momentum projection adds `PeriodicBox` and
/// is `GasError::Capability` when its axis is outside the position
/// coordinates. All arms are `ExchangeParity::Even`. The plaquette exists in
/// every dimension.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    spec.validate()?;
    let ChannelSpec::Glueball {
        observable,
        momentum,
    } = spec
    else {
        return Err(GasError::Capability("not a glueball specification".into()));
    };
    let measured = match observable {
        GlueballObservable::ForceNorm => ElementKind::Site,
        _ => ElementKind::Triplet,
    };
    if kind != measured {
        return Err(GasError::Capability(format!(
            "glueball/{} is not defined on {} elements",
            observable.name(),
            kind.name()
        )));
    }
    let mut requires = Requirements::new([Record::Color]);
    let mut spatial_parity = SpatialParity::Even;
    if let Some(momentum) = momentum {
        let dimension = context.capabilities.dimension;
        if momentum.axis >= dimension {
            return Err(GasError::Capability(format!(
                "momentum axis {} is outside the {dimension} position coordinates",
                momentum.axis
            )));
        }
        requires = requires.with(Record::PeriodicBox);
        if momentum.phase == MomentumPhase::Sin {
            spatial_parity = SpatialParity::Odd;
        }
    }
    let plaquette = r"\Pi_{ijk} = q_{ij} q_{jk} q_{ki},\; q_{ab} = c_a^\dagger c_b";
    let (definition, label, note) = match observable {
        GlueballObservable::RePlaquette => (
            format!(r"\operatorname{{Re}}\, \Pi_{{ijk}},\quad {plaquette}"),
            "def-sm-direct-color-contractions",
            "product of rank-one colour projectors on the triangle anchor, distance companion, \
             cloning companion; invariant under per-site rephasing and a common unitary rotation \
             only, so it is no locally covariant Wilson loop; bounded below by -1/8",
        ),
        GlueballObservable::OneMinusRe => (
            format!(r"1 - \operatorname{{Re}}\, \Pi_{{ijk}},\quad {plaquette}"),
            "def-sm-direct-color-contractions",
            "its connected correlator equals that of the real part of the plaquette: the two \
             channels are one piece of evidence",
        ),
        GlueballObservable::OneMinusCos => (
            format!(r"1 - \cos(\arg \Pi_{{ijk}}),\quad {plaquette}"),
            "def-sm-direct-color-contractions",
            "evaluated without trigonometry as one minus the real part of the plaquette over its \
             modulus; a triplet with a link overlap of modulus at most 1e-12 is masked because \
             its phase is undefined",
        ),
        GlueballObservable::Sin2 => (
            format!(r"\sin^2(\arg \Pi_{{ijk}}),\quad {plaquette}"),
            "def-sm-direct-color-contractions",
            "evaluated without trigonometry as the squared imaginary part of the plaquette over its \
             modulus; a triplet with a link overlap of modulus at most 1e-12 is masked because \
             its phase is undefined; it cannot tell a phase from its supplement",
        ),
        GlueballObservable::ForceNorm => (
            r"\lVert F^{\mathrm{visc}}_i \rVert^2".to_string(),
            "sec-qft-calibration-channel-derivations",
            "squared norm of the recorded force behind the colour, as a frame mean over covered \
             walkers instead of a sum; a zero force is a value, not a mask; the recorded force \
             carries the normalisation of the viscous kernel, so amplitudes are not comparable \
             across variants",
        ),
    };
    let descriptor = match momentum {
        None => Descriptor {
            definition,
            book_label: label.into(),
            spatial_parity: Some(spatial_parity),
            note: note.into(),
        },
        Some(momentum) => Descriptor {
            definition: format!(
                r"\left({definition}\right) \{}(2\pi\, {}\, x_i^{{({})}} / L)",
                momentum.phase.name(),
                momentum.mode,
                momentum.axis
            ),
            book_label: String::new(),
            spatial_parity: Some(spatial_parity),
            note: format!(
                "{note}; Fourier weight at the raw anchor coordinate, L the periodic box length, \
                 not a Volume II definition; a single cosine or sine correlator depends on the \
                 origin of the axis, only their sum at equal mode is translation invariant, and \
                 the cosine mode 0 duplicates the unprojected channel"
            ),
        },
    };
    Ok(Signature {
        descriptor,
        ..Signature::new(requires, 1, ExchangeParity::Even)
    })
}

/// Invalid colours mask a plaquette; the phase observables also mask a
/// triplet with a link overlap of modulus at most `1e-12`, where `arg Π` is
/// undefined. `ForceNorm` reads `state.force` where `state.force_valid`,
/// whatever the colour validity, and masks a norm that overflows. The
/// momentum weight reads the anchor position of `state` and masks the element
/// without a periodic box of positive length or with a nonfinite coordinate.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let ChannelSpec::Glueball {
        observable,
        momentum,
    } = spec
    else {
        return false;
    };
    let value = match observable {
        GlueballObservable::ForceNorm => force_norm(element, state),
        GlueballObservable::RePlaquette => colors(element, state).map(|c| plaquette(c).re),
        GlueballObservable::OneMinusRe => colors(element, state).map(|c| 1. - plaquette(c).re),
        GlueballObservable::OneMinusCos => colors(element, state)
            .and_then(plaquette_phase)
            .map(|u| 1. - u.re),
        GlueballObservable::Sin2 => colors(element, state)
            .and_then(plaquette_phase)
            .map(|u| u.im * u.im),
    };
    // The weight is read after the value exists, so its anchor is a walker of `state`.
    let projected = value.and_then(|value| match momentum {
        None => Some(value),
        Some(momentum) => Some(value * fourier_weight(momentum, element, state, context)?),
    });
    match projected {
        Some(projected) => {
            out[0] = projected;
            true
        }
        None => false,
    }
}

/// Colours `[c_i, c_j, c_k]` of a triplet read from `state`; `None` when a
/// walker is ineligible or has no valid colour, or when two slots coincide.
pub(super) fn colors<'a>(element: &Element, state: &'a FrameState) -> Option<[&'a [C]; 3]> {
    let [i, j, k] = element.walkers.map(|w| w as usize);
    if element.kind != ElementKind::Triplet || i == j || i == k || j == k {
        return None;
    }
    let d = state.d;
    let color = |w: usize| {
        (state.eligible.get(w) == Some(&true) && state.color_valid.get(w) == Some(&true))
            .then(|| state.color.get(w * d..(w + 1) * d))
            .flatten()
    };
    Some([color(i)?, color(j)?, color(k)?])
}

/// `Π = q_ab q_bc q_ca` with `q_ab = a† b`.
fn plaquette([a, b, c]: [&[C]; 3]) -> C {
    dot(a, b) * dot(b, c) * dot(c, a)
}

/// `Π / |Π|`; `None` when a link overlap has modulus at most `1e-12`. Only a
/// square root enters, so the phase observables need no trigonometry.
pub(super) fn plaquette_phase([a, b, c]: [&[C]; 3]) -> Option<C> {
    let floor = 1e-12;
    let links = [dot(a, b), dot(b, c), dot(c, a)];
    links.iter().all(|q| q.abs() > floor).then(|| {
        let product = links[0] * links[1] * links[2];
        product / product.abs()
    })
}
fn force_norm(element: &Element, state: &FrameState) -> Option<f64> {
    let (i, d) = (element.walkers[0] as usize, state.d);
    if element.kind != ElementKind::Site
        || state.eligible.get(i) != Some(&true)
        || state.force_valid.get(i) != Some(&true)
    {
        return None;
    }
    let row = state.force.as_ref()?.get(i * d..(i + 1) * d)?;
    let norm = row.iter().map(|f| f * f).sum::<f64>();
    norm.is_finite().then_some(norm)
}
fn fourier_weight(
    momentum: &Momentum,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
) -> Option<f64> {
    let (i, axis) = (element.walkers[0] as usize, momentum.axis);
    let domain = periodic_box(&context.gas.boundary)?;
    let length = domain.upper.get(axis)? - domain.lower.get(axis)?;
    if axis >= state.d || !(length.is_finite() && length > 0.) {
        return None;
    }
    let angle = TAU * f64::from(momentum.mode) * state.x.get(i * state.d + axis)? / length;
    angle.is_finite().then(|| match momentum.phase {
        MomentumPhase::Cos => angle.cos(),
        MomentumPhase::Sin => angle.sin(),
    })
}
