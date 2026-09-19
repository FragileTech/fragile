//! Dirac bilinears `ψ̄_i Γ ψ_j` of the spinor lift of the colour state, with
//! the chiral projectors, role classes and electroweak phase links. Channels
//! are labelled by their algebra and verified parities, never by a `J^PC` name.
//! The kept part is the one even under exchange: the real part when `γ0 Γ` is
//! Hermitian, the imaginary part for `γ5`. The lift is neither norm preserving
//! nor rotation equivariant; the descriptor note says so.
use super::tensor::color;
use crate::{
    GasError, Result,
    physics::{
        qft::math::C,
        spectroscopy::{
            config::{ChannelSpec, ChiralProjector, DiracGamma, PhaseLink, RoleClass},
            contract::{
                Descriptor, Element, ElementKind, ExchangeParity, FrameState, OperatorContext,
                Record, Requirements, Signature, SpatialParity, WalkerRole,
            },
            fields::{
                dirac::{Spinor, apply, bilinear, gamma, gamma5, lift, overlap},
                score::{su2_phase, u1_phase},
            },
        },
    },
};

const LIFT_NOTE: &str = "spinor lift psi = (E(Im c), E(Re c)) with E(w) = (w1 + i w2, w3) / \
                         sqrt|w|: odd, not norm preserving, covariant only under rotations about \
                         the third colour axis and not invariant under a common colour phase; the \
                         colour encoding is componentwise; a colour whose real or imaginary part \
                         has norm at most 1e-12 is masked";
/// Index pairs `(jk)` of the spatial `σ^{jk}` components, in kept order.
const SPATIAL: [(usize, usize); 3] = [(1, 2), (1, 3), (2, 3)];

fn components(gamma: DiracGamma) -> usize {
    match gamma {
        DiracGamma::Scalar | DiracGamma::Pseudoscalar => 1,
        _ => 3,
    }
}
/// Sign `η` of `γ0 Γ γ0 = η Γ`: the behaviour under `c → −c*`, which acts on
/// the lift as `γ0`.
fn inversion(gamma: DiracGamma) -> SpatialParity {
    match gamma {
        DiracGamma::Scalar | DiracGamma::Axial | DiracGamma::Tensor => SpatialParity::Even,
        DiracGamma::Pseudoscalar | DiracGamma::Vector | DiracGamma::TensorTime => {
            SpatialParity::Odd
        }
    }
}
/// The kept part of `γ0 Γ` alone is even. `γ0 Γ γ5` is Hermitian together with
/// `γ0 Γ` only for the currents, so a projector leaves the other arms with an
/// odd part. The cloning phase has no exchange symmetry, and the mirror of a
/// cross-class pair belongs to the opposite class.
fn exchange(
    gamma: DiracGamma,
    projector: ChiralProjector,
    pairs: RoleClass,
    link: PhaseLink,
) -> ExchangeParity {
    let current = matches!(gamma, DiracGamma::Vector | DiracGamma::Axial);
    if link == PhaseLink::Su2
        || matches!(pairs, RoleClass::LeftRight | RoleClass::RightLeft)
        || (projector != ChiralProjector::None && !current)
    {
        ExchangeParity::Mixed
    } else {
        ExchangeParity::Even
    }
}
fn definition(
    gamma: DiracGamma,
    projector: ChiralProjector,
    pairs: RoleClass,
    link: PhaseLink,
) -> String {
    let part = if gamma == DiracGamma::Pseudoscalar {
        "Im"
    } else {
        "Re"
    };
    let phase = match link {
        PhaseLink::None => "",
        PhaseLink::U1 => r"e^{i\theta_{ij}}\,",
        PhaseLink::Su2 => r"e^{i\vartheta_{ij}}\,",
    };
    let (matrix, range) = match gamma {
        DiracGamma::Scalar => ("", ""),
        DiracGamma::Pseudoscalar => (r"\gamma^5", ""),
        DiracGamma::Vector => (r"\gamma^k", r",\quad k = 1, 2, 3"),
        DiracGamma::Axial => (r"\gamma^5\gamma^k", r",\quad k = 1, 2, 3"),
        DiracGamma::Tensor => (r"\sigma^{jk}", r",\quad (jk) = (12), (13), (23)"),
        DiracGamma::TensorTime => (r"\sigma^{0k}", r",\quad k = 1, 2, 3"),
    };
    let chirality = match projector {
        ChiralProjector::None => "",
        ChiralProjector::Left => " P_L",
        ChiralProjector::Right => " P_R",
    };
    let class = match pairs {
        RoleClass::Any => "",
        RoleClass::LeftLeft => r",\quad (i, j) \in LL",
        RoleClass::RightRight => r",\quad (i, j) \in RR",
        RoleClass::LeftRight => r",\quad (i, j) \in LR",
        RoleClass::RightLeft => r",\quad (i, j) \in RL",
    };
    format!(r"\operatorname{{{part}}}\,{phase}\bar\psi_i {matrix}{chirality}\psi_j{range}{class}")
}
fn note(
    gamma: DiracGamma,
    projector: ChiralProjector,
    pairs: RoleClass,
    link: PhaseLink,
) -> String {
    let analogue = match gamma {
        DiracGamma::Scalar => "0++",
        DiracGamma::Pseudoscalar => "0-+",
        DiracGamma::Vector | DiracGamma::TensorTime => "1--",
        DiracGamma::Axial => "1++",
        DiracGamma::Tensor => "1+-",
    };
    let mut note = format!(
        "{LIFT_NOTE}; {analogue} is the continuum analogue of the fermion bilinear, not a \
         property of the channel: spin and charge conjugation are not defined on the record"
    );
    if components(gamma) == 3 {
        note.push_str("; the three components are not those of a rotation vector");
    }
    if gamma == DiracGamma::Pseudoscalar {
        note.push_str("; identically minus the third tensor_time component");
    }
    if projector != ChiralProjector::None {
        note.push_str(
            "; the chiral projector mixes both inversion parities, and the upper and lower \
             spinor components are parity blocks, not chirality eigenspaces",
        );
        note.push_str(match gamma {
            DiracGamma::Vector => "",
            DiracGamma::Axial => {
                "; identically the projected vector current, with the opposite sign under the \
                 right projector: not an independent channel"
            }
            _ => {
                "; the part odd under exchange cancels on a mutual pairing, where the left and \
                 the right projection have the same frame mean"
            }
        });
    }
    if pairs != RoleClass::Any {
        note.push_str(
            "; the role class reads the walker roles of the evaluated frame (left = cloner or \
             strong resister); a pair mask is not a spinor projector",
        );
    }
    match link {
        PhaseLink::None => {}
        PhaseLink::U1 => note.push_str(
            "; theta_ij = -(F_j - F_i) / h_eff of the fitness; the reference code multiplies by \
             the conjugate phase, which changes the kept part and not only its sign",
        ),
        PhaseLink::Su2 => note.push_str(
            "; vartheta_ij = (F_j - F_i) / ((|F_i| + epsilon_clone) h_S): a scalar modulation, \
             not a matrix-valued SU(2) link; the reference code uses a phase of |F_j - F_i| \
             that is symmetric under exchange",
        ),
    }
    note
}

/// `ChannelSpec::Dirac` on a distance or cloning pair, three dimensions
/// only. Requires `Color`; a role class adds `Fitness`, `CloningCompanions`
/// and `ClonePlan`, a phase link the records of its phase. `Vector`, `Axial`,
/// `Tensor` and `TensorTime` keep 3 components; `Pseudoscalar` keeps the
/// imaginary part. Parities come from the verified table of the family.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    _context: &OperatorContext<'_>,
) -> Result<Signature> {
    let &ChannelSpec::Dirac {
        gamma,
        projector,
        pairs,
        link,
    } = spec
    else {
        return Err(GasError::Capability("not a Dirac channel".into()));
    };
    if kind.arity() != 2 {
        return Err(GasError::Capability(
            "Dirac channels are measured on pairs".into(),
        ));
    }
    let mut requires = Requirements::new([Record::Color]).in_dimension(3);
    if pairs != RoleClass::Any {
        requires = requires
            .with(Record::Fitness)
            .with(Record::CloningCompanions)
            .with(Record::ClonePlan);
    }
    if link != PhaseLink::None {
        requires = requires.with(Record::Fitness);
    }
    if link == PhaseLink::Su2 {
        requires = requires.with(Record::ClonePlan);
    }
    let plain =
        projector == ChiralProjector::None && pairs == RoleClass::Any && link == PhaseLink::None;
    Ok(Signature {
        descriptor: Descriptor {
            definition: definition(gamma, projector, pairs, link),
            book_label: if plain {
                "def-qft-dirac-lift"
            } else {
                "thm-sm-ew-operator-layers"
            }
            .into(),
            spatial_parity: (projector == ChiralProjector::None).then(|| inversion(gamma)),
            note: note(gamma, projector, pairs, link),
        },
        ..Signature::new(
            requires,
            components(gamma),
            exchange(gamma, projector, pairs, link),
        )
    })
}

/// Left-handed role of a walker; `None` without roles.
fn left(state: &FrameState, walker: usize) -> Option<bool> {
    let role = state.role.as_ref()?.get(walker)?;
    Some(matches!(
        role,
        WalkerRole::Cloner | WalkerRole::StrongResister
    ))
}
fn admitted(pairs: RoleClass, state: &FrameState, i: usize, j: usize) -> bool {
    let class = match pairs {
        RoleClass::Any => return true,
        RoleClass::LeftLeft => (true, true),
        RoleClass::RightRight => (false, false),
        RoleClass::LeftRight => (true, false),
        RoleClass::RightLeft => (false, true),
    };
    left(state, i).zip(left(state, j)) == Some(class)
}
/// Unit phase of the link from the fitness of the evaluated frame; `None`
/// masks the element.
fn phase(
    link: PhaseLink,
    state: &FrameState,
    context: &OperatorContext<'_>,
    i: usize,
    j: usize,
) -> Option<C> {
    let fitness = |walker: usize| state.fitness.as_ref()?.get(walker).copied();
    let scales = &context.measurement.electroweak;
    let angle = match link {
        PhaseLink::None => return Some(C::ONE),
        PhaseLink::U1 => u1_phase(fitness(i)?, fitness(j)?, scales.h_eff),
        PhaseLink::Su2 => su2_phase(
            fitness(i)?,
            fitness(j)?,
            context.gas.clone_decision.epsilon,
            scales.h_s.unwrap_or(scales.h_eff),
        ),
    };
    angle.is_finite().then(|| C::phase(angle))
}
/// `P ψ` with `P_{L,R} = (1 ∓ γ5) / 2`.
fn projected(projector: ChiralProjector, spinor: &Spinor) -> Spinor {
    let sign = match projector {
        ChiralProjector::None => return *spinor,
        ChiralProjector::Left => -1.,
        ChiralProjector::Right => 1.,
    };
    let image = apply(&gamma5(), spinor);
    std::array::from_fn(|r| (spinor[r] + image[r] * sign) * 0.5)
}
/// Component `k` of `ψ̄_a Γ ψ_b`. `γ5 γ^k` multiplies in this order, and
/// `σ^{μν} = i γ^μ γ^ν` for `μ ≠ ν` is applied factor by factor, so that no
/// matrix product is formed per element.
fn component(gamma_kind: DiracGamma, k: usize, a: &Spinor, b: &Spinor) -> C {
    let i = C::new(0., 1.);
    match gamma_kind {
        DiracGamma::Scalar => overlap(a, b),
        DiracGamma::Pseudoscalar => bilinear(a, &gamma5(), b),
        DiracGamma::Vector => bilinear(a, &gamma(k + 1), b),
        DiracGamma::Axial => bilinear(a, &gamma5(), &apply(&gamma(k + 1), b)),
        DiracGamma::TensorTime => i * bilinear(a, &gamma(0), &apply(&gamma(k + 1), b)),
        DiracGamma::Tensor => {
            let (mu, nu) = SPATIAL[k];
            i * bilinear(a, &gamma(mu), &apply(&gamma(nu), b))
        }
    }
}

/// An ineligible walker masks the element in every arm, as does a colour
/// with a vanishing real or imaginary part, which has no lift, a walker
/// outside the role class and a link without a finite fitness at both ends.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    let &ChannelSpec::Dirac {
        gamma,
        projector,
        pairs,
        link,
    } = spec
    else {
        return false;
    };
    let (i, j) = (element.walkers[0] as usize, element.walkers[1] as usize);
    let eligible = |walker: usize| state.eligible.get(walker) == Some(&true);
    if i == j || !eligible(i) || !eligible(j) || !admitted(pairs, state, i, j) {
        return false;
    }
    let spinor = |walker| color(state, walker).and_then(lift);
    let (Some(a), Some(b), Some(u)) = (spinor(i), spinor(j), phase(link, state, context, i, j))
    else {
        return false;
    };
    let b = projected(projector, &b);
    for (k, value) in out.iter_mut().take(components(gamma)).enumerate() {
        let z = u * component(gamma, k, &a, &b);
        *value = if gamma == DiracGamma::Pseudoscalar {
            z.im
        } else {
            z.re
        };
    }
    true
}
