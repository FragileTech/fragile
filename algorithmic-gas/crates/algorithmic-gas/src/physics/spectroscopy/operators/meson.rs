//! Pair operators of the colour overlap `q_ij = c_i† c_j`: `Re q` (exchange
//! even, parity even) and `Im q` (exchange odd, parity odd), with the
//! score-directed, score-weighted, `γ5`-diagonal and `|q|²` arms. The arms of
//! `q` are colour singlets; the `γ5`-diagonal contraction is not.
use crate::{
    GasError, Result,
    physics::{
        qft::math::{C, dot},
        spectroscopy::{
            config::{ChannelSpec, MesonMode, MesonQuantum},
            contract::{
                Auxiliary, Descriptor, Element, ElementKind, ExchangeParity, FrameState,
                OperatorContext, Record, Requirements, Signature, SpatialParity,
            },
        },
    },
};

/// The two valid colours of a pair element. Both contractions accumulate the
/// components in order without fused products, so exchanging the walkers
/// conjugates them bit for bit and the exchange-odd part of a mirrored pair
/// cancels exactly.
pub(super) struct ColorPair<'a> {
    pub(super) i: usize,
    pub(super) j: usize,
    pub(super) anchor: &'a [C],
    pub(super) companion: &'a [C],
}
impl<'a> ColorPair<'a> {
    /// `None` masks the element: not a pair, a walker paired with itself, a
    /// walker outside the state or an invalid colour.
    pub(super) fn of(element: &Element, state: &'a FrameState) -> Option<Self> {
        let (i, j) = (element.walkers[0] as usize, element.walkers[1] as usize);
        let d = state.d;
        let valid = |w: usize| state.color_valid.get(w).copied().unwrap_or(false);
        if element.kind.arity() != 2 || i == j || !valid(i) || !valid(j) {
            return None;
        }
        Some(Self {
            i,
            j,
            anchor: state.color.get(i * d..(i + 1) * d)?,
            companion: state.color.get(j * d..(j + 1) * d)?,
        })
    }
    /// `q_ij = c_i† c_j`.
    pub(super) fn overlap(&self) -> C {
        dot(self.anchor, self.companion)
    }
    /// `g_ij = Σ_a (−1)^a conj(c_i^a) c_j^a`.
    fn gamma5(&self) -> C {
        let terms = self.anchor.iter().zip(self.companion).enumerate();
        terms.fold(C::ZERO, |s, (a, (&x, &y))| {
            if a.is_multiple_of(2) {
                s + x.conj() * y
            } else {
                s - x.conj() * y
            }
        })
    }
}

/// `S_j − S_i`; `NaN` where either walker has no valid score, which is missing
/// data and not the tie that a difference of 0 states.
fn score_difference(state: &FrameState, i: usize, j: usize) -> f64 {
    let gap = || {
        let score = state.score.as_ref()?;
        let valid = |w: usize| state.score_valid.get(w).copied().unwrap_or(false);
        (valid(i) && valid(j)).then_some(score.get(j)? - score.get(i)?)
    };
    gap().unwrap_or(f64::NAN)
}

/// `ChannelSpec::Meson` on a distance or cloning pair. Requires `Color`; the
/// score modes add `Fitness` and `CloningCompanions`. `Im q`, `|ΔS| Im q` and
/// `Im g` are `ExchangeParity::Odd`. The directed pseudoscalar is `Even`: the
/// orientation sign and `Im q` both change under exchange and a tie is masked
/// at both ends. `Pseudoscalar` with `Abs2` duplicates the scalar `|q|²` and
/// is `GasError::Capability`, as is `Gamma5Diagonal` on one colour component,
/// where `g = q`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    spec.validate()?;
    let (ChannelSpec::Meson { quantum, mode }, 2) = (spec, kind.arity()) else {
        return Err(GasError::Capability(
            "meson operators act on the pairs of a meson specification".into(),
        ));
    };
    let exchange = match (quantum, mode) {
        (MesonQuantum::Pseudoscalar, MesonMode::Abs2) => {
            return Err(GasError::Capability(
                "the squared modulus of the overlap has no imaginary part".into(),
            ));
        }
        (_, MesonMode::Gamma5Diagonal) if context.capabilities.dimension < 2 => {
            return Err(GasError::Capability(
                "the diagonal colour gamma5 is the identity on one colour component".into(),
            ));
        }
        (MesonQuantum::Scalar, _) | (_, MesonMode::ScoreDirected) => ExchangeParity::Even,
        (MesonQuantum::Pseudoscalar, _) => ExchangeParity::Odd,
    };
    let requires = match mode {
        MesonMode::ScoreDirected | MesonMode::ScoreWeighted => {
            Requirements::new([Record::Color, Record::Fitness, Record::CloningCompanions])
        }
        _ => Requirements::new([Record::Color]),
    };
    let part = match quantum {
        MesonQuantum::Scalar => r"\operatorname{Re}",
        MesonQuantum::Pseudoscalar => r"\operatorname{Im}",
    };
    let scored = "Scores are those of each walker against its own cloning companion, read once \
        at the source time of the element and frozen with it: a sink of the propagator reapplies \
        the source-time factor instead of the order of its own frame.";
    let (definition, book_label, note) = match mode {
        MesonMode::Standard => (
            format!(r"{part}\, c_i^\dagger c_j"),
            "def-sm-direct-color-contractions",
            String::new(),
        ),
        MesonMode::ScoreDirected => (
            format!(r"{part}\, c_a^\dagger c_b,\ \{{a, b\}} = \{{i, j\}},\ S_a < S_b"),
            "",
            format!(
                "The pair is read from its lower to its higher score and masked on a tie. \
                 {scored} On a mutual cloning pair the orientation runs from the fitter to the \
                 less fit walker. Because the orientation is frozen with the element it enters \
                 the lag product squared, so the source-frozen propagator of this arm is the one \
                 of meson/{}/standard over the oriented elements and only the frame mean is a \
                 channel of its own.{}",
                quantum.name(),
                match quantum {
                    MesonQuantum::Scalar => {
                        " The real part ignores the orientation: the series equals \
                         meson/scalar/standard off ties and never shares a fit basis with it."
                    }
                    MesonQuantum::Pseudoscalar => "",
                }
            ),
        ),
        MesonMode::ScoreWeighted => (
            format!(r"|S_j - S_i|\, {part}\, c_i^\dagger c_j"),
            "",
            format!(
                "The score difference multiplies the value, not the element weight: the frame \
                 denominator stays the valid weight sum, and a tied pair keeps the element with \
                 the value 0. A frame whose weighted score dispersion Σ_I w_I |S_j − S_i| \
                 vanishes has no average at all and carries the weight 0, so a series of exact \
                 zeros is never presented as a measurement. {scored} The factor is exchange even, \
                 so it does not lift the cancellation of the imaginary part on a mutual pairing."
            ),
        ),
        MesonMode::Gamma5Diagonal => (
            format!(
                r"{part}\, c_i^\dagger \Gamma_5 c_j,\ \Gamma_5 = \operatorname{{diag}}((-1)^a)"
            ),
            "def-qft-color-gamma-operators",
            "Invariant only under the colour frames that commute with Γ5, not under SU(d). \
             The real part is even under inversion: a second scalar channel."
                .into(),
        ),
        MesonMode::Abs2 => (
            r"|c_i^\dagger c_j|^2".into(),
            "cor-sm-direct-exchange-parity",
            "Invariant under a separate rephasing of each walker.".into(),
        ),
    };
    let auxiliary = match mode {
        MesonMode::ScoreDirected => Some(Auxiliary::ScoreOrientation),
        MesonMode::ScoreWeighted => Some(Auxiliary::ScoreDispersion),
        _ => None,
    };
    let directed = *mode == MesonMode::ScoreDirected;
    Ok(Signature {
        propagatable: !directed,
        descriptor: Descriptor {
            definition,
            book_label: book_label.into(),
            spatial_parity: Some(match quantum {
                MesonQuantum::Scalar => SpatialParity::Even,
                MesonQuantum::Pseudoscalar => SpatialParity::Odd,
            }),
            note,
        },
        auxiliary,
        ..Signature::new(requires, 1, exchange)
    })
}

/// Invalid colours mask the element. A score mode leaves the contraction in
/// the sampled order and writes the factor the score contributes beside it:
/// the orientation sign of the directed arm, the dispersion `|S_j − S_i|` of
/// the weighted one, `0` on a tie and not a number where a walker has no valid
/// score. The measurement freezes that factor with the element. The arms
/// without an operator mask every element.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    _: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let ChannelSpec::Meson { quantum, mode } = spec else {
        return false;
    };
    let Some(pair) = ColorPair::of(element, state) else {
        return false;
    };
    let part = |z: C| match quantum {
        MesonQuantum::Scalar => z.re,
        MesonQuantum::Pseudoscalar => z.im,
    };
    let (value, factor) = match mode {
        MesonMode::Standard => (part(pair.overlap()), None),
        // The real part is invariant under the orientation, so the factor of
        // the directed scalar carries no sign: it only states that the pair
        // has an order at all.
        MesonMode::ScoreDirected => {
            let ds = score_difference(state, pair.i, pair.j);
            let factor = match quantum {
                _ if ds == 0. || ds.is_nan() => ds,
                MesonQuantum::Scalar => 1.,
                MesonQuantum::Pseudoscalar => ds.signum(),
            };
            (part(pair.overlap()), Some(factor))
        }
        MesonMode::ScoreWeighted => (
            part(pair.overlap()),
            Some(score_difference(state, pair.i, pair.j).abs()),
        ),
        MesonMode::Gamma5Diagonal if state.d < 2 => return false,
        MesonMode::Gamma5Diagonal => (part(pair.gamma5()), None),
        MesonMode::Abs2 => match quantum {
            MesonQuantum::Scalar => (pair.overlap().abs2(), None),
            MesonQuantum::Pseudoscalar => return false,
        },
    };
    if out.len() != 1 + usize::from(factor.is_some()) {
        return false;
    }
    out[0] = value;
    if let (Some(slot), Some(factor)) = (out.get_mut(1), factor) {
        *slot = factor;
    }
    value.is_finite()
}
