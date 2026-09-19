//! U(1) fitness phases on distance pairs, SU(2) cloning phases on cloning
//! pairs, their mixed triplet product and the symmetry-breaking site scalars.
//! Interaction ranges are never the clone regulariser and self-companions are
//! masked.
use crate::{
    GasError, Result,
    physics::{
        qft::math::C,
        spectroscopy::{
            config::{ChannelSpec, FrameNormalization, PairDistance, Range, Su2Mode, U1Mode},
            contract::{
                Descriptor, Element, ElementKind, ExchangeParity, FrameState, NO_COMPANION,
                OperatorContext, Record, Requirements, Signature, WalkerRole,
            },
            fields::score::{AmplitudeDistance, amplitude, su2_phase, u1_phase, walker_fitness},
        },
    },
};

/// Gaussian amplitude `exp(−D²/4ε²)` of one companion role.
#[derive(Clone, Copy)]
struct Dressing<'a> {
    distance: AmplitudeDistance<'a>,
    range: f64,
}
impl Dressing<'_> {
    fn value(&self, state: &FrameState, i: usize, j: usize) -> Option<f64> {
        Some(amplitude(self.distance.squared(state, i, j)?, self.range))
    }
}
/// Amplitude of the companion role of `kind`, a distance or a cloning pair.
/// The range is the fixed value or the Gaussian kernel width of the role.
fn dressing<'a>(context: &OperatorContext<'a>, kind: ElementKind) -> Result<Dressing<'a>> {
    let scales = &context.measurement.electroweak;
    let (name, range, width, donors) = match kind {
        ElementKind::DistancePair => (
            "epsilon_d",
            scales.epsilon_d,
            context.capabilities.distance_kernel_width,
            &context.gas.distance_donors,
        ),
        _ => (
            "epsilon_c",
            scales.epsilon_c,
            context.capabilities.cloning_kernel_width,
            &context.gas.cloning_donors,
        ),
    };
    let range = match range {
        Range::Fixed { value } => value,
        Range::FromKernel => width.ok_or_else(|| {
            GasError::Capability(format!(
                "range {name} is the width of a Gaussian companion kernel and the kernel of \
                 this role has none; set electroweak.{name} to a fixed value"
            ))
        })?,
    };
    Ok(Dressing {
        distance: AmplitudeDistance::new(context.gas, &donors.distance, scales)?,
        range,
    })
}

/// Hop amplitudes `a_ab = r_c(a, b) e^{iϑ_ab}` of the SU(2) family.
#[derive(Clone, Copy)]
struct Hops<'a> {
    epsilon: f64,
    h_s: f64,
    directed: bool,
    dressing: Option<Dressing<'a>>,
}
impl<'a> Hops<'a> {
    fn new(context: &OperatorContext<'a>, directed: bool, dressed: bool) -> Result<Self> {
        let scales = &context.measurement.electroweak;
        Ok(Self {
            epsilon: context.gas.clone_decision.epsilon,
            h_s: scales.h_s.unwrap_or(scales.h_eff),
            directed,
            dressing: dressed
                .then(|| dressing(context, ElementKind::CloningPair))
                .transpose()?,
        })
    }
    fn value(&self, state: &FrameState, a: usize, b: usize) -> Option<C> {
        if a == b {
            return None;
        }
        let phase = su2_phase(
            walker_fitness(state, a)?,
            walker_fitness(state, b)?,
            self.epsilon,
            self.h_s,
        );
        let phase = if self.directed { phase.abs() } else { phase };
        let r = match &self.dressing {
            Some(dressing) => dressing.value(state, a, b)?,
            None => 1.,
        };
        Some(C::phase(phase) * r)
    }
}
fn descriptor(
    definition: impl Into<String>,
    book_label: &str,
    note: impl Into<String>,
) -> Descriptor {
    Descriptor {
        definition: definition.into(),
        book_label: book_label.into(),
        spatial_parity: None,
        note: note.into(),
    }
}
fn distance_note(context: &OperatorContext<'_>) -> &'static str {
    match context.measurement.electroweak.distance {
        PairDistance::Raw => "amplitude distance on the raw recorded coordinates",
        PairDistance::Configured => "amplitude distance of the role's donor module",
    }
}

/// `ChannelSpec::{U1, Su2, ElectroweakMixed, FitnessPhase, CloneIndicator,
/// ParityVelocity}`. U(1) requires `Fitness`; SU(2) adds `ClonePlan`. A
/// `Range::FromKernel` whose role has no Gaussian kernel width in the
/// capabilities is `GasError::Capability`. Complex values have the two
/// components `(Re, Im)` and are `Mixed`: the imaginary part of a U(1) phase
/// is exchange odd. `ElectroweakMixed` sets `degenerate`; `CloneIndicator`
/// fixes `normalization: Some(FixedN)`.
pub(super) fn signature(
    spec: &ChannelSpec,
    kind: ElementKind,
    context: &OperatorContext<'_>,
) -> Result<Signature> {
    spec.validate()?;
    let scales = &context.measurement.electroweak;
    scales.validate()?;
    if !spec.kinds(context.measurement.pairs).contains(&kind) {
        return Err(GasError::Capability(format!(
            "{} is not defined on {} elements",
            spec.id(),
            kind.name()
        )));
    }
    let velocities = |requires: Requirements, needed: bool| {
        if needed {
            requires.with(Record::Velocities)
        } else {
            requires
        }
    };
    let scale_note = if scales.h_s.is_some() {
        "score phase on its own action scale h_S"
    } else {
        "score phase on the fitness action scale: h_S = h_eff"
    };
    Ok(match *spec {
        ChannelSpec::U1 { mode, charge } => {
            let dressed = match mode {
                U1Mode::Phase => None,
                U1Mode::Dressed => Some(dressing(context, kind)?),
            };
            let requires = velocities(
                Requirements::new([Record::Fitness]),
                dressed.is_some_and(|d| d.distance.needs_velocities()),
            );
            let q = if charge == 1 {
                String::new()
            } else {
                charge.to_string()
            };
            let phase = format!(
                "e^{{i {q}\\theta_{{ij}}}},\\; \\theta_{{ij}} = -(\\Phi_j - \\Phi_i)/\
                 \\hbar_{{\\mathrm{{eff}}}}"
            );
            let odd = "imaginary part is exchange odd: on a mutual pairing the frame-mean \
                       observable is the cosine";
            let charged = if charge == 1 {
                ""
            } else {
                "; charges other than 1 are not defined in the book"
            };
            Signature {
                descriptor: match (mode, charge) {
                    (U1Mode::Phase, 1) => descriptor(phase, "def-fractal-set-phase-potential", odd),
                    (U1Mode::Phase, _) => descriptor(phase, "", format!("{odd}{charged}")),
                    (U1Mode::Dressed, _) => descriptor(
                        format!("e^{{-D_{{ij}}^2/4\\varepsilon_d^2}}\\, {phase}"),
                        if charge == 1 {
                            "def-sm-direct-companion-doublet"
                        } else {
                            ""
                        },
                        format!(
                            "{odd}; {}; the amplitude is not raised to the charge{charged}",
                            distance_note(context)
                        ),
                    ),
                },
                ..Signature::new(requires, 2, ExchangeParity::Mixed)
            }
        }
        ChannelSpec::Su2 { mode, directed } => {
            let hops = Hops::new(context, directed, mode != Su2Mode::Phase)?;
            let exchange = match mode {
                Su2Mode::Phase | Su2Mode::Component => ExchangeParity::Mixed,
                Su2Mode::Doublet => ExchangeParity::Even,
                Su2Mode::DoubletDiff => ExchangeParity::Odd,
            };
            let requires = velocities(
                Requirements::new([Record::Fitness, Record::ClonePlan]),
                hops.dressing.is_some_and(|d| d.distance.needs_velocities()),
            );
            let phase =
                "\\vartheta_{ik} = (F_k - F_i)/((|F_i| + \\varepsilon_{\\mathrm{clone}})\\, h_S)";
            let hop = "a_{ik} = e^{-D_{ik}^2/4\\varepsilon_c^2} e^{i\\vartheta_{ik}}";
            let (definition, label, remark) = match mode {
                Su2Mode::Phase => (
                    format!("e^{{i\\vartheta_{{ik}}}},\\; {phase}"),
                    "def-fractal-set-cloning-score",
                    "the phase is only weight antisymmetric, so its imaginary part survives a \
                     mutual pairing",
                ),
                Su2Mode::Component => (
                    format!("{hop},\\; {phase}"),
                    "def-sm-direct-companion-doublet",
                    "the phase is only weight antisymmetric, so its imaginary part survives a \
                     mutual pairing",
                ),
                Su2Mode::Doublet => (
                    format!("a_{{i k(i)}} + a_{{k(i) k(k(i))}},\\; {hop},\\; {phase}"),
                    "def-sm-direct-companion-doublet",
                    "the second entry follows the companion's own cloning companion at the \
                     evaluated time; on a bijective companion map the frame mean is twice that \
                     of the component",
                ),
                Su2Mode::DoubletDiff => (
                    format!("a_{{i k(i)}} - a_{{k(i) k(k(i))}},\\; {hop},\\; {phase}"),
                    "def-sm-direct-companion-doublet",
                    "the second entry follows the companion's own cloning companion at the \
                     evaluated time; exchange odd on a mutual pairing, and the frame mean \
                     vanishes identically on every bijective companion map while the \
                     source-frozen propagator does not",
                ),
            };
            let ranged = if mode == Su2Mode::Phase {
                String::new()
            } else {
                format!("; {}", distance_note(context))
            };
            Signature {
                descriptor: if directed {
                    descriptor(
                        definition,
                        "",
                        format!(
                            "{remark}; {scale_note}{ranged}; directed replaces every hop phase \
                             by its absolute value and is not defined in the book"
                        ),
                    )
                } else {
                    descriptor(definition, label, format!("{remark}; {scale_note}{ranged}"))
                },
                ..Signature::new(requires, 2, exchange)
            }
        }
        ChannelSpec::ElectroweakMixed => {
            let distance = dressing(context, ElementKind::DistancePair)?;
            let cloning = dressing(context, ElementKind::CloningPair)?;
            let requires = velocities(
                Requirements::new([Record::Fitness, Record::ClonePlan]),
                distance.distance.needs_velocities() || cloning.distance.needs_velocities(),
            );
            Signature {
                degenerate: true,
                descriptor: descriptor(
                    "e^{-D_{ij}^2/4\\varepsilon_d^2}\\, e^{-D_{ik}^2/4\\varepsilon_c^2}\\, \
                     e^{i(\\theta_{ij} + \\vartheta_{ik})}",
                    "",
                    format!(
                        "mixed proxy of the calibration chapter without a labelled definition; \
                         defined also where the two companions coincide; {scale_note}; {}",
                        distance_note(context)
                    ),
                ),
                ..Signature::new(requires, 2, ExchangeParity::Mixed)
            }
        }
        ChannelSpec::FitnessPhase => Signature {
            descriptor: descriptor(
                "\\theta_i = -\\Phi_i/\\hbar_{\\mathrm{eff}}",
                "",
                "site phase of the fitness potential; the book defines the phase but no \
                 channel of it",
            ),
            ..Signature::new(
                Requirements::new([Record::Fitness]),
                1,
                ExchangeParity::Even,
            )
        },
        ChannelSpec::CloneIndicator => {
            let comb = if context.gas.clone_decision.every > 1 {
                "; with a cloning period above 1 the series is zero off the period, a comb \
                 without a decay rate"
            } else {
                ""
            };
            Signature {
                normalization: Some(FrameNormalization::FixedN),
                descriptor: descriptor(
                    "\\mathbf{1}[i \\in \\Delta_t]",
                    "",
                    format!(
                        "accepted clone decisions of living walkers, revivals excluded; not a \
                         channel of the book{comb}"
                    ),
                ),
                ..Signature::new(
                    Requirements::new([Record::ClonePlan]),
                    1,
                    ExchangeParity::Even,
                )
            }
        }
        ChannelSpec::ParityVelocity { role } => {
            let gated = matches!(role, WalkerRole::Cloner | WalkerRole::StrongResister);
            let comb = if gated && context.gas.clone_decision.every > 1 {
                "; with a cloning period above 1 the role is empty off the period, a comb \
                 without a decay rate"
            } else {
                ""
            };
            Signature {
                descriptor: descriptor(
                    format!(
                        "\\lVert v_i \\rVert,\\; i \\in \\mathrm{{{}}}",
                        role.name().replace('_', "\\_")
                    ),
                    "",
                    format!(
                        "speed of the walkers of one role, from the accepted decisions and the \
                         ungated score sign; it compares speed distributions between roles and \
                         is not a parity-odd observable; not a channel of the book{comb}"
                    ),
                ),
                ..Signature::new(
                    Requirements::new([
                        Record::Velocities,
                        Record::Fitness,
                        Record::CloningCompanions,
                        Record::ClonePlan,
                    ]),
                    1,
                    ExchangeParity::Even,
                )
            }
        }
        _ => {
            return Err(GasError::Capability(format!(
                "{} is not an electroweak channel",
                spec.id()
            )));
        }
    })
}

/// Walkers of an element of `kind`; `None` for any other element.
fn walkers(element: &Element, kind: ElementKind) -> Option<[usize; 3]> {
    (element.kind == kind).then(|| element.walkers.map(|w| w as usize))
}
fn pair_value(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
) -> Option<C> {
    let scales = &context.measurement.electroweak;
    match *spec {
        ChannelSpec::U1 { mode, charge } => {
            let [i, j, _] = walkers(element, ElementKind::DistancePair)?;
            if i == j {
                return None;
            }
            let theta = u1_phase(
                walker_fitness(state, i)?,
                walker_fitness(state, j)?,
                scales.h_eff,
            );
            let r = match mode {
                U1Mode::Phase => 1.,
                U1Mode::Dressed => dressing(context, ElementKind::DistancePair)
                    .ok()?
                    .value(state, i, j)?,
            };
            Some(C::phase(f64::from(charge) * theta) * r)
        }
        ChannelSpec::Su2 { mode, directed } => {
            let [i, k, _] = walkers(element, ElementKind::CloningPair)?;
            let hops = Hops::new(context, directed, mode != Su2Mode::Phase).ok()?;
            let own = hops.value(state, i, k)?;
            if matches!(mode, Su2Mode::Phase | Su2Mode::Component) {
                return Some(own);
            }
            let next = *state.cloning_companion.as_ref()?.get(k)?;
            if next == NO_COMPANION {
                return None;
            }
            let companion = hops.value(state, k, next as usize)?;
            Some(if mode == Su2Mode::Doublet {
                own + companion
            } else {
                own - companion
            })
        }
        ChannelSpec::ElectroweakMixed => {
            let [i, j, k] = walkers(element, ElementKind::Triplet)?;
            if i == j {
                return None;
            }
            let theta = u1_phase(
                walker_fitness(state, i)?,
                walker_fitness(state, j)?,
                scales.h_eff,
            );
            let r = dressing(context, ElementKind::DistancePair)
                .ok()?
                .value(state, i, j)?;
            let hop = Hops::new(context, false, true).ok()?.value(state, i, k)?;
            Some(C::phase(theta) * r * hop)
        }
        _ => None,
    }
}
fn site_value(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
) -> Option<f64> {
    let [i, ..] = walkers(element, ElementKind::Site)?;
    if !*state.eligible.get(i)? {
        return None;
    }
    match *spec {
        ChannelSpec::FitnessPhase => {
            Some(-walker_fitness(state, i)? / context.measurement.electroweak.h_eff)
        }
        ChannelSpec::CloneIndicator => Some(f64::from(*state.cloned.get(i)?)),
        ChannelSpec::ParityVelocity { role } => {
            if *state.role.as_ref()?.get(i)? != role {
                return None;
            }
            let v = state.v.as_ref()?.get(i * state.d..(i + 1) * state.d)?;
            Some(v.iter().map(|a| a * a).sum::<f64>().sqrt())
        }
        _ => None,
    }
}

/// The doublet modes are two-hop, `a_i ± a_{k(i)}`, the second entry built
/// on `state.cloning_companion[k(i)]` and masked at `NO_COMPANION`. Every hop
/// phase is `su2_phase(F_a, F_b, ε_clone, h_S)` from `state.fitness` of the two
/// walkers of the hop at the evaluated time, with `ε_clone =
/// context.gas.clone_decision.epsilon` and `h_S =
/// context.measurement.electroweak.h_s.unwrap_or(h_eff)`; `state.score` is
/// never read, because at a sink it belongs to the sink-time companion and
/// not to the frozen pair. `directed` replaces every hop phase by its absolute
/// value; a zero phase gives the value 1 and masks nothing. The amplitude
/// distance follows `ElectroweakScales::distance`.
pub(super) fn evaluate(
    spec: &ChannelSpec,
    element: &Element,
    state: &FrameState,
    context: &OperatorContext<'_>,
    out: &mut [f64],
) -> bool {
    match spec {
        ChannelSpec::U1 { .. } | ChannelSpec::Su2 { .. } | ChannelSpec::ElectroweakMixed => {
            match pair_value(spec, element, state, context) {
                Some(z) if out.len() >= 2 && z.re.is_finite() && z.im.is_finite() => {
                    out[0] = z.re;
                    out[1] = z.im;
                    true
                }
                _ => false,
            }
        }
        _ => match site_value(spec, element, state, context) {
            Some(value) if !out.is_empty() && value.is_finite() => {
                out[0] = value;
                true
            }
            _ => false,
        },
    }
}
