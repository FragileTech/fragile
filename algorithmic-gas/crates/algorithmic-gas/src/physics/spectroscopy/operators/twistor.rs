//! Twistor observables of the two effective edge twistors of a companion
//! triplet: the spinor contraction `τ` and the Pauli-vector bilinear `W`. The
//! edge vector `(Δt, Δx)` mixes units with `c = 1` implicit, so the velocity
//! scale is a time. The first edge runs to the distance companion, the second
//! to the cloning companion; the exchange parity of an arm is its sign under
//! the swap of these two roles. It is exact on one element, and the frame mean
//! of an odd arm has zero expectation only when the two companion laws are
//! exchangeable. No arm has a definite spatial parity.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::{ChannelSpec, FrameNormalization, TwistorObservable},
        contract::{
            Descriptor, Element, ElementKind, ExchangeParity, FrameState, OperatorContext, Record,
            Requirements, Signature, periodic_box,
        },
        fields::spinor::{contraction, edge, pauli_bilinear, spin_two},
    },
};
use std::array::from_fn;

/// The five spin-2 components of `W Wᵀ` in the markup of the report page.
const SPIN_TWO: &str = concat!(
    r"\operatorname{Re}\bigl(W^xW^y,\;W^xW^z,\;W^yW^z,\;\tfrac{1}{\sqrt2}(W^xW^x-W^yW^y),\;",
    r"\tfrac{1}{\sqrt6}(2W^zW^z-W^xW^x-W^yW^y)\bigr),\quad ",
    r"W^a=\lambda_{ij}^\dagger\sigma^a\lambda_{ik}",
);
const SPIN_TWO_MEAN: &str = concat!(
    r"\tfrac15\operatorname{Re}\bigl(W^xW^y+W^xW^z+W^yW^z+\tfrac{1}{\sqrt2}(W^xW^x-W^yW^y)+",
    r"\tfrac{1}{\sqrt6}(2W^zW^z-W^xW^x-W^yW^y)\bigr),\quad ",
    r"W^a=\lambda_{ij}^\dagger\sigma^a\lambda_{ik}",
);
fn components(observable: TwistorObservable) -> usize {
    match observable {
        TwistorObservable::Vector | TwistorObservable::Axial => 3,
        TwistorObservable::Tensor => 5,
        _ => 1,
    }
}
fn descriptor(observable: TwistorObservable, velocity_scale: f64) -> Descriptor {
    let (definition, caveat) = match observable {
        TwistorObservable::Scalar => (
            r"\operatorname{Re}\,\epsilon^{AB}\lambda_{ij,A}\lambda_{ik,B}",
            "mixes with twistor/pseudoscalar under rotations about the third axis; odd under \
             exchange of the two companion roles",
        ),
        TwistorObservable::Pseudoscalar => (
            r"\operatorname{Im}\,\epsilon^{AB}\lambda_{ij,A}\lambda_{ik,B}",
            "mixes with twistor/scalar under rotations about the third axis; odd under exchange \
             of the two companion roles; odd only under the mirror of the second axis combined \
             with velocity reversal",
        ),
        TwistorObservable::Abs2 => (
            r"|\epsilon^{AB}\lambda_{ij,A}\lambda_{ik,B}|^2",
            "squared mass of the two-twistor momentum of unit energy; the only arm free of the \
             spinor phase convention",
        ),
        TwistorObservable::Vector => (
            r"\operatorname{Re}\,\lambda_{ij}^\dagger\sigma^a\lambda_{ik}",
            "three components contracted by the dot product; rotates into twistor/axial when \
             the two edges select different columns",
        ),
        TwistorObservable::Axial => (
            r"\operatorname{Im}\,\lambda_{ij}^\dagger\sigma^a\lambda_{ik}",
            "three components contracted by the dot product; rotates into twistor/vector when \
             the two edges select different columns; odd under exchange of the two companion \
             roles",
        ),
        TwistorObservable::Tensor => (
            SPIN_TWO,
            "five components kept separate; the basis is not orthonormal (the off-diagonal \
             components lack a factor of the square root of 2), so the contracted correlator is \
             not rotation invariant",
        ),
        TwistorObservable::TensorMean => (
            SPIN_TWO_MEAN,
            "adds spin-2 components of different azimuthal number; not rotation invariant",
        ),
    };
    Descriptor {
        definition: definition.into(),
        book_label: "def-effective-twistor-operators".into(),
        spatial_parity: None,
        note: format!(
            "velocity scale {velocity_scale} in time units with c = 1; the time component of an \
             edge is one integrator step and its displacement is the minimum image on a \
             periodic box; the dominant-column rule fixes the spinor phases and singles out the \
             third axis; no arm has a definite spatial parity, all are even up to the ratio of \
             the time step to the edge length; {caveat}"
        ),
    }
}

/// `ChannelSpec::Twistor` on triplets, three dimensions only. Requires
/// `Velocities` and a time step in the capabilities; no formula reads the
/// colour. `Vector` and `Axial` keep 3 components, `Tensor` the 5 spin-2
/// components, the others 1. Every arm fixes `normalization: Some(FixedN)`,
/// reports no spatial parity and states the velocity scale in its note. The
/// contraction and the imaginary part of `W` are odd under the swap of the two
/// companion roles, every other arm is even.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    spec.validate()?;
    let ChannelSpec::Twistor {
        observable,
        velocity_scale,
    } = *spec
    else {
        return Err(GasError::Configuration(
            "twistor signature of another channel family".into(),
        ));
    };
    if kind != ElementKind::Triplet {
        return Err(GasError::Capability(
            "twistor observables are defined on triplets".into(),
        ));
    }
    if context.capabilities.time_step.is_none() {
        return Err(GasError::Capability(
            "twistor edges require the time step of the kinetic integrator".into(),
        ));
    }
    let exchange = match observable {
        TwistorObservable::Scalar | TwistorObservable::Pseudoscalar | TwistorObservable::Axial => {
            ExchangeParity::Odd
        }
        _ => ExchangeParity::Even,
    };
    Ok(Signature {
        normalization: Some(FrameNormalization::FixedN),
        descriptor: descriptor(observable, velocity_scale),
        ..Signature::new(
            Requirements::new([Record::Velocities]).in_dimension(3),
            components(observable),
            exchange,
        )
    })
}

/// A null edge masks the element. The dominant-column tie selects column 0.
/// Walkers are read at the evaluated time: an ineligible walker, a repeated
/// slot, a nonfinite coordinate or an `out` of another width masks too.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let ChannelSpec::Twistor {
        observable,
        velocity_scale,
    } = *spec
    else {
        return false;
    };
    let (Some(dt), Some(v)) = (context.capabilities.time_step, state.v.as_deref()) else {
        return false;
    };
    let [i, j, k] = element.walkers.map(|w| w as usize);
    let domain = periodic_box(&context.gas.boundary);
    if element.kind != ElementKind::Triplet
        || state.d != 3
        || out.len() != components(observable)
        || i == j
        || i == k
        || j == k
        || [i, j, k]
            .iter()
            .any(|&w| state.eligible.get(w) != Some(&true))
        || domain.is_some_and(|b| b.lower.len() != 3 || b.upper.len() != 3)
    {
        return false;
    }
    let row = |field: &[f64], w: usize| -> Option<[f64; 3]> {
        field.chunks_exact(3).nth(w)?.try_into().ok()
    };
    let spinor = |w: usize| {
        let (x0, x1, v0, v1) = (row(&state.x, i)?, row(&state.x, w)?, row(v, i)?, row(v, w)?);
        let dx = from_fn(|a| {
            let delta = x1[a] - x0[a];
            domain.map_or(delta, |b| b.minimum_image(delta, a))
        });
        edge(dx, from_fn(|a| v1[a] - v0[a]), dt, velocity_scale)
    };
    let (Some(a), Some(b)) = (spinor(j), spinor(k)) else {
        return false;
    };
    match observable {
        TwistorObservable::Scalar => out[0] = contraction(&a, &b).re,
        TwistorObservable::Pseudoscalar => out[0] = contraction(&a, &b).im,
        TwistorObservable::Abs2 => out[0] = contraction(&a, &b).abs2(),
        TwistorObservable::Vector => out.copy_from_slice(&pauli_bilinear(&a, &b).map(|w| w.re)),
        TwistorObservable::Axial => out.copy_from_slice(&pauli_bilinear(&a, &b).map(|w| w.im)),
        TwistorObservable::Tensor => out.copy_from_slice(&spin_two(&pauli_bilinear(&a, &b))),
        TwistorObservable::TensorMean => {
            out[0] = spin_two(&pauli_bilinear(&a, &b)).iter().sum::<f64>() / 5.
        }
    }
    true
}
