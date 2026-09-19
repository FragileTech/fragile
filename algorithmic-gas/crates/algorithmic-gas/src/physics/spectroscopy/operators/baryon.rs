//! Baryon operators of the colour determinant `b_ijk = det[c_i, c_j, c_k]` on
//! companion triplets, with the columns in the sampled order: anchor,
//! distance companion, cloning companion. Phase-invariant modes are labelled
//! as such; `|b|` is not a Volume II mode.
use super::glueball::{colors, plaquette_phase};
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::{BaryonMode, ChannelSpec},
        contract::{
            Descriptor, Element, ElementKind, ExchangeParity, FrameState, OperatorContext, Record,
            Requirements, Signature, SpatialParity,
        },
        fields::color::det3,
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
    _: &OperatorContext<'_>,
) -> Result<Signature> {
    spec.validate()?;
    let ChannelSpec::Baryon { mode, flux_alpha } = spec else {
        return Err(GasError::Capability("not a baryon specification".into()));
    };
    if kind != ElementKind::Triplet {
        return Err(GasError::Capability(format!(
            "baryon/{} is not defined on {} elements",
            mode.name(),
            kind.name()
        )));
    }
    let mut requires = Requirements::new([Record::Color]).in_dimension(3);
    if *mode == BaryonMode::ScoreOrdered {
        requires = requires
            .with(Record::Fitness)
            .with(Record::CloningCompanions);
    }
    let components = if *mode == BaryonMode::Complex { 2 } else { 1 };
    let exchange = match mode {
        BaryonMode::Real | BaryonMode::Imag | BaryonMode::Complex => ExchangeParity::Odd,
        _ => ExchangeParity::Even,
    };
    let determinant = r"b_{ijk} = \det[c_i, c_j, c_k]";
    let (re, im) = (
        r"\operatorname{Re}\, b_{ijk}",
        r"\operatorname{Im}\, b_{ijk}",
    );
    let odd = "antisymmetric under exchange of the distance and cloning roles: with identically \
               distributed independent roles the frame mean has zero expectation at every lag, \
               and the signal is in the source-frozen propagator; not invariant under per-site \
               rephasing or a common velocity boost, which rotate the phase of the determinant";
    let neutral = "invariant under per-site rephasing and a common unitary rotation, so it \
                   carries vacuum quantum numbers and a large mean: only its connected \
                   correlator is meaningful";
    let descriptor = match mode {
        BaryonMode::Real => Descriptor {
            definition: format!(r"{re},\quad {determinant}"),
            book_label: "def-sm-direct-color-contractions".into(),
            spatial_parity: Some(SpatialParity::Odd),
            note: odd.into(),
        },
        BaryonMode::Imag => Descriptor {
            definition: format!(r"{im},\quad {determinant}"),
            book_label: "def-sm-direct-color-contractions".into(),
            spatial_parity: Some(SpatialParity::Even),
            note: odd.into(),
        },
        BaryonMode::Complex => Descriptor {
            definition: format!(r"({re},\, {im}),\quad {determinant}"),
            book_label: "prop-sm-baryon-exterior-correlator".into(),
            spatial_parity: None,
            note: format!(
                "the contracted correlator is the real part of the conjugate source determinant \
                 times the sink determinant, which a time-independent determinant phase leaves \
                 fixed; the real part is parity odd and the imaginary part parity even; {odd}"
            ),
        },
        BaryonMode::Abs2 => Descriptor {
            definition: format!(r"\lvert b_{{ijk}} \rvert^2,\quad {determinant}"),
            book_label: "thm-sm-direct-color-invariants".into(),
            spatial_parity: Some(SpatialParity::Even),
            note: format!(
                "{neutral}; for unit colours it equals one minus the three squared link overlaps \
                 plus twice the real part of the plaquette, so it holds no information beyond \
                 the pair and plaquette channels"
            ),
        },
        BaryonMode::Abs => Descriptor {
            definition: format!(r"\lvert b_{{ijk}} \rvert,\quad {determinant}"),
            book_label: String::new(),
            spatial_parity: Some(SpatialParity::Even),
            note: format!(
                "not a Volume II mode, kept for comparison with the reference code; not smooth \
                 at a vanishing determinant; {neutral}"
            ),
        },
        BaryonMode::ScoreOrdered => Descriptor {
            definition: format!(
                "\\operatorname{{sgn}}(\\sigma)\\, {re},\\quad S_{{\\sigma(1)}} < \
                 S_{{\\sigma(2)}} < S_{{\\sigma(3)}},\\quad {determinant}"
            ),
            book_label: String::new(),
            spatial_parity: Some(SpatialParity::Odd),
            note: "not a Volume II mode; the columns are ordered by the cloning score of each \
                   walker at the evaluated time, so a sink reads the sink-time order and the \
                   series mixes colour decorrelation with score crossings; a tie or a walker \
                   without a score masks the triplet; parity odd for parity-even scores; not \
                   invariant under per-site rephasing or a common velocity boost"
                .into(),
        },
        BaryonMode::FluxWeighted => Descriptor {
            definition: format!(
                "\\lvert b_{{ijk}} \\rvert\\, \\exp\\left({flux_alpha}\\, (1 - \\cos \\arg \
                 \\Pi_{{ijk}})\\right),\\quad \\Pi_{{ijk}} = q_{{ij}} q_{{jk}} q_{{ki}},\\quad \
                 {determinant}"
            ),
            book_label: String::new(),
            spatial_parity: Some(SpatialParity::Even),
            note: format!(
                "not a Volume II mode; reads colours only; a triplet with a link overlap of \
                 modulus at most 1e-12 is masked because the plaquette phase is undefined, and \
                 so is a weight that overflows; {neutral}"
            ),
        },
    };
    Ok(Signature {
        descriptor,
        ..Signature::new(requires, components, exchange)
    })
}

/// Invalid colours mask the element, a vanishing determinant does not.
/// `ScoreOrdered` orders the walkers by `state.score` and masks a tie or a
/// walker without `state.score_valid`. `FluxWeighted` masks an undefined
/// plaquette phase and a weight that overflows.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    _: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let ChannelSpec::Baryon { mode, flux_alpha } = spec else {
        return false;
    };
    if state.d != 3 {
        return false;
    }
    let Some([a, b, c]) = colors(element, state) else {
        return false;
    };
    let det = det3(a, b, c);
    match mode {
        BaryonMode::Real => out[0] = det.re,
        BaryonMode::Imag => out[0] = det.im,
        BaryonMode::Complex => {
            out[0] = det.re;
            out[1] = det.im;
        }
        BaryonMode::Abs2 => out[0] = det.abs2(),
        BaryonMode::Abs => out[0] = det.abs(),
        BaryonMode::ScoreOrdered => {
            let Some(sign) = order_sign(element, state) else {
                return false;
            };
            out[0] = sign * det.re;
        }
        BaryonMode::FluxWeighted => {
            let Some(phase) = plaquette_phase([a, b, c]) else {
                return false;
            };
            let weighted = det.abs() * (flux_alpha * (1. - phase.re)).exp();
            if !weighted.is_finite() {
                return false;
            }
            out[0] = weighted;
        }
    }
    true
}

/// Sign of the permutation that sorts the three walkers by ascending score:
/// the determinant with its columns in that order is this sign times
/// `b_ijk`. `None` on a tie or a walker without a valid score.
fn order_sign(element: &Element, state: &FrameState) -> Option<f64> {
    let scores = state.score.as_ref()?;
    let score = |w: u32| {
        let w = w as usize;
        (state.score_valid.get(w) == Some(&true))
            .then(|| scores.get(w).copied())
            .flatten()
    };
    let [i, j, k] = element.walkers;
    let s = [score(i)?, score(j)?, score(k)?];
    let mut sign = 1.;
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        if s[a] == s[b] {
            return None;
        }
        if s[a] > s[b] {
            sign = -sign;
        }
    }
    Some(sign)
}
