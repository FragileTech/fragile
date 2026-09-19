//! Chirality of the walker roles on full-frame series with masks: a gappy
//! series of cloning frames is never correlated as if it were uniform.
use crate::{
    GasError, Result,
    physics::{
        qft::math::C,
        spectroscopy::{
            config::{ChannelSpec, ChiralityObservable, FrameNormalization},
            contract::{
                Descriptor, Element, ElementKind, ExchangeParity, FrameState, OperatorContext,
                Record, Requirements, Signature, WalkerRole,
            },
            fields::score::walker_fitness,
        },
    },
};

/// Cloners and strong resisters are left-handed, weak resisters and
/// persisters right-handed.
fn left(role: WalkerRole) -> bool {
    matches!(role, WalkerRole::Cloner | WalkerRole::StrongResister)
}
/// Role of walker `i` when the row is eligible and the roles are filled.
fn role(state: &FrameState, i: usize) -> Option<WalkerRole> {
    let role = *state.role.as_ref()?.get(i)?;
    state.eligible.get(i)?.then_some(role)
}

/// `ChannelSpec::Chirality`: `Chi` and `LeftFraction` on sites, so that every
/// walker of `L_t` counts whatever its cloning entry is; `LeftRightCoupling`
/// on cloning pairs. Requires `Fitness`, `CloningCompanions` and `ClonePlan`.
/// Every arm fixes `normalization: Some(FixedN)` and is
/// `ExchangeParity::Even`. The left-right coupling is `GasError::Capability`:
/// its support, the cloners with a right-handed cloning companion, is empty
/// on every frame, because the companion of a cloner is a cloner or a strong
/// resister.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    spec.validate()?;
    let ChannelSpec::Chirality { observable } = *spec else {
        return Err(GasError::Capability(format!(
            "{} is not a chirality channel",
            spec.id()
        )));
    };
    if !spec.kinds(context.measurement.pairs).contains(&kind) {
        return Err(GasError::Capability(format!(
            "{} is not defined on {} elements",
            spec.id(),
            kind.name()
        )));
    }
    let comb = if context.gas.clone_decision.every > 1 {
        "; with a cloning period above 1 nobody is left-handed off the period and the series \
         is a comb without a decay rate"
    } else {
        ""
    };
    let (definition, note) = match observable {
        ChiralityObservable::Chi => (
            "\\chi_i = \\mathbf{1}[i \\in \\Delta_t \\cup SR_t] - \
             \\mathbf{1}[i \\in WR_t \\cup P_t]",
            format!(
                "fixed 1/N frame mean in which dead walkers contribute 0, so that it equals \
                 twice the left fraction minus the living fraction; roles from the accepted \
                 decisions and the ungated score sign{comb}"
            ),
        ),
        ChiralityObservable::LeftFraction => (
            "\\mathbf{1}[i \\in \\Delta_t \\cup SR_t]",
            format!("fixed 1/N frame mean |L_t|/N over all walkers{comb}"),
        ),
        ChiralityObservable::LeftRightCoupling => {
            return Err(GasError::Capability(
                "identically zero: the cloning companion of a cloner is a cloner or a strong \
                 resister, never right-handed"
                    .into(),
            ));
        }
    };
    Ok(Signature {
        normalization: Some(FrameNormalization::FixedN),
        descriptor: Descriptor {
            definition: definition.into(),
            book_label: "def-sm-walker-chirality".into(),
            spatial_parity: None,
            note,
        },
        ..Signature::new(
            Requirements::new([
                Record::Fitness,
                Record::CloningCompanions,
                Record::ClonePlan,
            ]),
            1,
            ExchangeParity::Even,
        )
    })
}

/// Reads `state.role`; a walker without a role masks the element. The
/// left-right coupling writes `e^{i (F_k − F_i) / ħ_eff}` as `(Re, Im)` on a
/// cloner whose cloning companion is right-handed.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let ChannelSpec::Chirality { observable } = *spec else {
        return false;
    };
    let [i, k, _] = element.walkers.map(|w| w as usize);
    match observable {
        ChiralityObservable::Chi | ChiralityObservable::LeftFraction => {
            let (Some(role), true, false) = (
                role(state, i),
                element.kind == ElementKind::Site,
                out.is_empty(),
            ) else {
                return false;
            };
            out[0] = match (observable, left(role)) {
                (_, true) => 1.,
                (ChiralityObservable::Chi, false) => -1.,
                _ => 0.,
            };
            true
        }
        ChiralityObservable::LeftRightCoupling => {
            if element.kind != ElementKind::CloningPair
                || i == k
                || out.len() < 2
                || role(state, i) != Some(WalkerRole::Cloner)
                || role(state, k).is_none_or(left)
            {
                return false;
            }
            let (Some(own), Some(companion)) = (walker_fitness(state, i), walker_fitness(state, k))
            else {
                return false;
            };
            let z = C::phase((companion - own) / context.measurement.electroweak.h_eff);
            out[0] = z.re;
            out[1] = z.im;
            z.re.is_finite() && z.im.is_finite()
        }
    }
}
