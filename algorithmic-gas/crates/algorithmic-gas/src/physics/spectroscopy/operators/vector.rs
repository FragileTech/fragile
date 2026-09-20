//! Vector and axial pair operators `Re q_ij · D` and `Im q_ij · D` for a
//! displacement `D`. Components are kept and contracted in the correlator;
//! the projection applies in every displacement mode.
use super::meson::ColorPair;
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::{ChannelSpec, Displacement, VectorProjection, VectorQuantum},
        contract::{
            Descriptor, Element, ElementKind, ExchangeParity, FrameState, OperatorContext, Record,
            Requirements, Signature, SpatialParity, periodic_box,
        },
    },
};

/// Euclidean norm scaled by the largest entry, so that neither a tiny nor a
/// huge vector loses its direction. Zero for the zero vector.
fn norm(v: &[f64]) -> f64 {
    let scale = v.iter().map(|x| x.abs()).fold(0., f64::max);
    if scale > 0. && scale.is_finite() {
        scale * v.iter().map(|x| (x / scale).powi(2)).sum::<f64>().sqrt()
    } else {
        scale
    }
}

/// Score gradient of the anchor `i`. `None` where the walker has no valid
/// score or no gradient: a zero gradient is the record of a walker without a
/// usable companion and has no direction.
fn score_gradient(state: &FrameState, i: usize) -> Option<&[f64]> {
    let d = state.d;
    let gradient = state.score_gradient.as_ref()?.get(i * d..(i + 1) * d)?;
    let valid = state.score_valid.get(i).copied().unwrap_or(false);
    (valid && norm(gradient) > 0.).then_some(gradient)
}

/// `out = x_j − x_i`, the minimum image on a periodic box. `false` where the
/// state lacks a position or the box has another dimension.
fn separation(
    state: &FrameState,
    context: &OperatorContext<'_>,
    i: usize,
    j: usize,
    out: &mut [f64],
) -> bool {
    let d = state.d;
    let (Some(own), Some(other)) = (
        state.x.get(i * d..(i + 1) * d),
        state.x.get(j * d..(j + 1) * d),
    ) else {
        return false;
    };
    let domain = periodic_box(&context.gas.boundary);
    if domain.is_some_and(|b| b.lower.len() != d || b.upper.len() != d) {
        return false;
    }
    for (axis, (r, (a, b))) in out.iter_mut().zip(own.iter().zip(other)).enumerate() {
        let delta = b - a;
        *r = domain.map_or(delta, |b| b.minimum_image(delta, axis));
    }
    true
}

/// `h^μ = i (conj(c_i^μ) c_j^ν − conj(c_i^ν) c_j^μ)`, `ν = μ + 1 mod d`:
/// `out[μ]` is `Re h^μ` for the vector and `Im h^μ` for the axial. The two
/// products are subtracted part by part, so `h_ji = conj(h_ij)` bit for bit.
fn color_gamma(pair: &ColorPair<'_>, quantum: VectorQuantum, out: &mut [f64]) {
    let (ci, cj, d) = (pair.anchor, pair.companion, pair.anchor.len());
    for (mu, h) in out.iter_mut().enumerate() {
        let nu = (mu + 1) % d;
        let (forward, backward) = (ci[mu].conj() * cj[nu], ci[nu].conj() * cj[mu]);
        *h = match quantum {
            VectorQuantum::Vector => -(forward.im - backward.im),
            VectorQuantum::Axial => forward.re - backward.re,
        };
    }
}

/// `ChannelSpec::Vector` on a distance or cloning pair, `d` components.
/// Requires `Color`; `ScoreGradient` and every projection other than `Full`
/// add `Fitness` and `CloningCompanions` and are `Mixed`, because they read
/// the score gradient of the anchor alone. The standard vector is
/// `ExchangeParity::Odd`, the standard axial `Even`. `ColorGamma` needs
/// `d ≥ 3` (`GasError::Capability` below) and swaps the parities: its vector
/// is even, its axial odd. The part of the score gradient across itself
/// vanishes identically and is `GasError::Capability`, as is every part
/// across the gradient in one dimension.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    spec.validate()?;
    let (
        ChannelSpec::Vector {
            quantum,
            projection,
            displacement,
        },
        2,
    ) = (spec, kind.arity())
    else {
        return Err(GasError::Capability(
            "vector operators act on the pairs of a vector specification".into(),
        ));
    };
    let d = context.capabilities.dimension;
    let exchange = match (displacement, projection, quantum) {
        (Displacement::ColorGamma, ..) if d < 3 => {
            return Err(GasError::Capability(
                "colour-space gamma operators need at least 3 colour components".into(),
            ));
        }
        (Displacement::ScoreGradient, VectorProjection::Transverse, _) => {
            return Err(GasError::Capability(
                "the score gradient has no part across itself".into(),
            ));
        }
        (_, VectorProjection::Transverse, _) if d < 2 => {
            return Err(GasError::Capability(
                "a displacement in one dimension has no part across the score gradient".into(),
            ));
        }
        (Displacement::ColorGamma, _, VectorQuantum::Vector)
        | (Displacement::Raw | Displacement::Unit, VectorProjection::Full, VectorQuantum::Axial) => {
            ExchangeParity::Even
        }
        (Displacement::ColorGamma, _, VectorQuantum::Axial)
        | (Displacement::Raw | Displacement::Unit, VectorProjection::Full, VectorQuantum::Vector) => {
            ExchangeParity::Odd
        }
        _ => ExchangeParity::Mixed,
    };
    let requires = if exchange == ExchangeParity::Mixed {
        Requirements::new([Record::Color, Record::Fitness, Record::CloningCompanions])
    } else {
        Requirements::new([Record::Color])
    };
    let part = match quantum {
        VectorQuantum::Vector => r"\operatorname{Re}",
        VectorQuantum::Axial => r"\operatorname{Im}",
    };
    let (symbol, note) = match displacement {
        Displacement::Raw => (
            "r_{ij}",
            "r_ij = x_j − x_i, the minimum image on a periodic box; the components are those of \
             the recorded positions.",
        ),
        Displacement::Unit => (
            r"\hat r_{ij}",
            "r̂_ij = r_ij / |r_ij| with r_ij = x_j − x_i, the minimum image on a periodic box; a \
             zero displacement is masked.",
        ),
        Displacement::ScoreGradient => (
            "G_i",
            "G_i is the mean of (S_j − S_i) r̂_ij / |r_ij| over the first distance and the first \
             cloning companion of the anchor, with the scores of each walker against its own \
             cloning companion at the evaluated time: a finite-difference quotient of the score \
             field, of the units of a score over a length, which scales as 1/λ under x → λx; an \
             anchor without a gradient is masked.",
        ),
        Displacement::ColorGamma => (
            "",
            "The matrices (Γ_μ)_ab = i (δ_aμ δ_bν − δ_aν δ_bμ), ν = μ + 1 mod d, act on the \
             colour index, not on space: no displacement enters, the components are not \
             invariant under SU(d), and the exchange behaviour is opposite to that of the \
             displacement vector.",
        ),
    };
    let mut note = String::from(note);
    let definition = match (displacement, projection) {
        (Displacement::ColorGamma, _) => format!(r"{part}\, c_i^\dagger \Gamma_\mu c_j"),
        (_, VectorProjection::Full) => format!(r"{part}(c_i^\dagger c_j)\, {symbol}"),
        (_, VectorProjection::Longitudinal) => {
            format!(r"{part}(c_i^\dagger c_j)\, ({symbol} \cdot \hat n_i)\, \hat n_i")
        }
        (_, VectorProjection::Transverse) => {
            format!(r"{part}(c_i^\dagger c_j)\, [{symbol} - ({symbol} \cdot \hat n_i)\, \hat n_i]")
        }
    };
    match (displacement, projection) {
        (_, VectorProjection::Full) => {}
        (Displacement::ScoreGradient, _) => note.push_str(
            " The gradient lies along itself: the series equals the full projection element by \
             element and never shares a fit basis with it.",
        ),
        _ => note.push_str(
            " n̂_i = G_i / |G_i| is the direction of the score gradient of the anchor, the mean \
             of (S_j − S_i) r̂_ij / |r_ij| over its first distance and first cloning companion; \
             an anchor without a gradient is masked.",
        ),
    }
    if *displacement == Displacement::ScoreGradient {
        note.push_str(
            " The gradient of a pair whose two companion maps agree is the same vector at both \
             ends, because both the score difference and the displacement change sign together: \
             on such a frame this arm inherits the exchange parity of its colour part, the \
             imaginary one cancels element by element over a mutual pairing, and only the \
             source-frozen propagator carries a signal.",
        );
    }
    let book_label = match (displacement, projection) {
        (Displacement::ColorGamma, _) => "def-qft-color-gamma-operators",
        (Displacement::Raw | Displacement::Unit, VectorProjection::Full) => {
            "def-sm-direct-color-contractions"
        }
        _ => "",
    };
    Ok(Signature {
        descriptor: Descriptor {
            definition,
            book_label: book_label.into(),
            spatial_parity: Some(match quantum {
                VectorQuantum::Vector => SpatialParity::Odd,
                VectorQuantum::Axial => SpatialParity::Even,
            }),
            note,
        },
        ..Signature::new(requires, d, exchange)
    })
}

/// Displacements use the minimum image of a periodic box. Projections are
/// along or across `state.score_gradient` of the anchor and mask a zero
/// gradient; a zero displacement masks `Unit` and has the value 0 under `Raw`.
/// The arms without an operator mask every element.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let ChannelSpec::Vector {
        quantum,
        projection,
        displacement,
    } = spec
    else {
        return false;
    };
    let Some(pair) = ColorPair::of(element, state) else {
        return false;
    };
    if out.len() != state.d || (state.d < 2 && *projection == VectorProjection::Transverse) {
        return false;
    }
    let directed =
        *projection != VectorProjection::Full || *displacement == Displacement::ScoreGradient;
    let gradient = directed.then(|| score_gradient(state, pair.i)).flatten();
    match displacement {
        Displacement::ColorGamma => {
            if state.d < 3 || *projection != VectorProjection::Full {
                return false;
            }
            color_gamma(&pair, *quantum, out);
            return out.iter().all(|h| h.is_finite());
        }
        Displacement::Raw | Displacement::Unit => {
            if !separation(state, context, pair.i, pair.j, out) {
                return false;
            }
            if *displacement == Displacement::Unit {
                let length = norm(out);
                if length > 0. {
                    out.iter_mut().for_each(|r| *r /= length);
                } else {
                    return false;
                }
            }
        }
        Displacement::ScoreGradient => match gradient {
            Some(g) if *projection != VectorProjection::Transverse => out.copy_from_slice(g),
            _ => return false,
        },
    }
    if *projection != VectorProjection::Full {
        let Some(g) = gradient else {
            return false;
        };
        let length = norm(g);
        let along = out
            .iter()
            .zip(g)
            .map(|(r, g)| r * (g / length))
            .sum::<f64>();
        for (r, g) in out.iter_mut().zip(g) {
            let parallel = along * (g / length);
            *r = match projection {
                VectorProjection::Longitudinal => parallel,
                _ => *r - parallel,
            };
        }
    }
    let q = pair.overlap();
    let weight = match quantum {
        VectorQuantum::Vector => q.re,
        VectorQuantum::Axial => q.im,
    };
    out.iter_mut().for_each(|r| *r *= weight);
    out.iter().all(|r| r.is_finite())
}
