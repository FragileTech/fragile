//! Cloning score, fitness phases, companion amplitudes, score gradient and
//! walker roles.
//!
//! The acceptance probability is always `CloneDecision::acceptance_probability`
//! (clipped, gated by `every`). The score here is the unclipped, ungated
//! `S_i = (V_c − V_i) / (V_i + ε)`; on a cloning step the two agree through
//! `p = clamp(S / saturation, 0, 1)`. The regulariser `ε` of the score is never
//! an interaction range: a companion amplitude takes its range from the
//! companion kernel of its role.
use crate::{
    GasConfig, GasError, Result,
    boundary::BoxDomain,
    error::require,
    geometry::Distance,
    kinetic::KineticKind,
    physics::spectroscopy::{
        config::{ElectroweakScales, PairDistance},
        contract::{Availability, Frame, FrameState, NO_COMPANION, WalkerRole, periodic_box},
        frame::position_field,
    },
};

pub fn score(own: f64, companion: f64, epsilon: f64) -> f64 {
    (companion - own) / (own + epsilon)
}

/// U(1) fitness phase `θ_ij = −(Φ_j − Φ_i) / ħ_eff`.
pub fn u1_phase(own: f64, companion: f64, h_eff: f64) -> f64 {
    -(companion - own) / h_eff
}

/// SU(2) cloning phase `(V_j − V_i) / ((|V_i| + ε) h_S)`. It equals
/// `score / h_S` on the positive fitness of a valid record.
pub fn su2_phase(own: f64, companion: f64, epsilon: f64, h_s: f64) -> f64 {
    (companion - own) / ((own.abs() + epsilon) * h_s)
}

/// Companion amplitude `exp(−D²/4ε²)`: the square root of the Gaussian kernel
/// weight `exp(−D²/2ε²)` of range `ε`.
pub fn amplitude(squared_distance: f64, range: f64) -> f64 {
    (-squared_distance / (4. * range * range)).exp()
}

/// Fitness of walker `i` when the row is eligible and the value finite; a
/// fitness phase masks every other walker.
pub(crate) fn walker_fitness(state: &FrameState, i: usize) -> Option<f64> {
    let fitness = *state.fitness.as_ref()?.get(i)?;
    (*state.eligible.get(i)? && fitness.is_finite()).then_some(fitness)
}

/// Distance `D` inside the companion amplitude of one role, following
/// `ElectroweakScales::{distance, lambda}`: the raw `|Δx|² + λ|Δv|²` of the
/// recorded coordinates, or the distance of the role's donor module, which
/// is the one a kernel width refers to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmplitudeDistance<'a> {
    configured: Option<&'a Distance>,
    lambda: f64,
}
impl<'a> AmplitudeDistance<'a> {
    /// `role` is the distance of the donor module of the companion role.
    /// `scales.lambda` replaces its velocity weight, which is 0 for a
    /// distance without velocities. `GasError::Capability` when the
    /// configured distance is not one of the recorded positions and
    /// velocities.
    pub fn new(gas: &'a GasConfig, role: &'a Distance, scales: &ElectroweakScales) -> Result<Self> {
        scales.validate()?;
        let own = match role {
            Distance::SquashedPhaseSpace { lambda, .. } | Distance::PhaseSpace { lambda, .. } => {
                *lambda
            }
            _ => 0.,
        };
        let lambda = scales.lambda.unwrap_or(own);
        require(
            lambda.is_finite() && lambda >= 0.,
            "amplitude velocity weight must be finite and nonnegative",
        )?;
        if scales.distance == PairDistance::Raw {
            return Ok(Self {
                configured: None,
                lambda,
            });
        }
        let recorded = match role {
            Distance::Euclidean { field, .. } => field == position_field(gas),
            Distance::PhaseSpace {
                positions,
                velocities,
                ..
            }
            | Distance::SquashedPhaseSpace {
                positions,
                velocities,
                ..
            } => {
                positions == position_field(gas)
                    && (lambda == 0.
                        || matches!(&gas.kinetic.integrator,
                            KineticKind::Baoab { velocities: v, .. } if v == velocities))
            }
            Distance::Cosine { .. } => false,
        };
        if !recorded {
            return Err(GasError::Capability(
                "configured companion distance is not a distance of the recorded positions \
                 and velocities"
                    .into(),
            ));
        }
        Ok(Self {
            configured: Some(role),
            lambda,
        })
    }
    pub fn needs_velocities(&self) -> bool {
        self.lambda > 0.
    }
    /// `D²` between walkers `i` and `j` of `state`; `None` when a coordinate
    /// it reads is absent or the value is not finite.
    pub fn squared(&self, state: &FrameState, i: usize, j: usize) -> Option<f64> {
        let d = state.d;
        let (xi, xj) = (row(&state.x, i, d)?, row(&state.x, j, d)?);
        let velocities = if self.lambda > 0. {
            let v = state.v.as_deref()?;
            Some((row(v, i, d)?, row(v, j, d)?))
        } else {
            None
        };
        let (positions, velocities) = match self.configured {
            None => (
                gap(xi, xj, 1., 1.),
                velocities.map_or(0., |(a, b)| gap(a, b, 1., 1.)),
            ),
            Some(Distance::SquashedPhaseSpace {
                position_radius,
                velocity_radius,
                ..
            }) => (
                gap(
                    xi,
                    xj,
                    squash(xi, *position_radius),
                    squash(xj, *position_radius),
                ),
                velocities.map_or(0., |(a, b)| {
                    gap(
                        a,
                        b,
                        squash(a, *velocity_radius),
                        squash(b, *velocity_radius),
                    )
                }),
            ),
            Some(Distance::Euclidean {
                scales, periodic, ..
            }) => (
                image_gap(xi, xj, periodic.as_ref(), |k| {
                    scales.get(k).copied().unwrap_or(1.)
                })?,
                velocities.map_or(0., |(a, b)| gap(a, b, 1., 1.)),
            ),
            Some(Distance::PhaseSpace {
                position_scale,
                velocity_scale,
                periodic,
                ..
            }) => (
                image_gap(xi, xj, periodic.as_ref(), |_| *position_scale)?,
                velocities.map_or(0., |(a, b)| {
                    gap(a, b, 1. / velocity_scale, 1. / velocity_scale)
                }),
            ),
            Some(Distance::Cosine { .. }) => return None,
        };
        let squared = positions + self.lambda * velocities;
        squared.is_finite().then_some(squared)
    }
}
fn row(values: &[f64], i: usize, d: usize) -> Option<&[f64]> {
    values.get(i.checked_mul(d)?..i.checked_add(1)?.checked_mul(d)?)
}
/// `Σ_k (a_k f_a − b_k f_b)²`.
fn gap(a: &[f64], b: &[f64], fa: f64, fb: f64) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x * fa - y * fb) * (x * fa - y * fb))
        .sum()
}
/// `Σ_k (image(a_k − b_k) / s_k)²`; `None` when the box is not one of the coordinates.
fn image_gap(
    a: &[f64],
    b: &[f64],
    periodic: Option<&BoxDomain>,
    scale: impl Fn(usize) -> f64,
) -> Option<f64> {
    let periodic = periodic_in(periodic, a.len())?;
    Some(
        a.iter()
            .zip(b)
            .enumerate()
            .map(|(k, (x, y))| {
                let delta = periodic.map_or(x - y, |p| p.minimum_image(x - y, k)) / scale(k);
                delta * delta
            })
            .sum(),
    )
}
/// The box when it has `d` axes; the outer `None` when it has another number.
fn periodic_in(periodic: Option<&BoxDomain>, d: usize) -> Option<Option<&BoxDomain>> {
    match periodic {
        Some(p) if p.lower.len() != d || p.upper.len() != d => None,
        other => Some(other),
    }
}
/// Factor `1 / (1 + |u| / R)` of the bounded feature `u / (1 + |u| / R)`, in
/// the overflow-safe form of the companion distance.
fn squash(u: &[f64], radius: f64) -> f64 {
    let scale = u.iter().fold(0., |m: f64, x| m.max(x.abs()));
    if scale == 0. {
        return 1.;
    }
    let norm = u
        .iter()
        .map(|x| (x / scale) * (x / scale))
        .sum::<f64>()
        .sqrt();
    if scale <= radius {
        1. / (1. + scale / radius * norm)
    } else {
        let r = radius / scale;
        r / (r + norm)
    }
}

/// Roles from the ungated score sign and the accepted decisions of the
/// frame, so that a gated cloning period does not erase them. `score` holds
/// 0 on a row without a valid score, which is then a persister unless it
/// clones or is targeted. A revived row is no cloner and targets nobody, a
/// historical companion is no walker of the frame, and an ineligible row
/// keeps the default role: every operator masks it.
pub fn roles(frame: &Frame, score: &[f64]) -> Vec<WalkerRole> {
    let cloner: Vec<bool> = frame
        .eligible
        .iter()
        .zip(&frame.cloned)
        .zip(&frame.revived)
        .map(|((&eligible, &cloned), &revived)| eligible && cloned && !revived)
        .collect();
    let mut targeted = vec![false; frame.n];
    if let Some(companions) = &frame.cloning {
        for i in (0..cloner.len()).filter(|&i| cloner[i]) {
            // The decision of row `i` was taken against its first entry.
            let e = i * companions.count;
            if companions.valid.get(e) == Some(&true)
                && companions.historical.get(e) == Some(&false)
                && let Some(target) = companions
                    .slot
                    .get(e)
                    .and_then(|&slot| targeted.get_mut(slot as usize))
            {
                *target = true;
            }
        }
    }
    let mut roles = vec![WalkerRole::default(); frame.n];
    for (i, role) in roles.iter_mut().enumerate() {
        if !frame.eligible.get(i).is_some_and(|e| *e) {
            continue;
        }
        *role = if cloner.get(i).is_some_and(|c| *c) {
            WalkerRole::Cloner
        } else if targeted[i] {
            WalkerRole::StrongResister
        } else if score.get(i).is_some_and(|s| *s > 0.) {
            WalkerRole::WeakResister
        } else {
            WalkerRole::Persister
        };
    }
    roles
}

/// Write `score` and `score_valid` (from `fitness`, `companion_fitness` and
/// `gas.clone_decision.epsilon`), `role` and `score_gradient` (from
/// `state.{distance,cloning}_companion`) into `state`. The gradient of a
/// walker is the mean of `(S_j − S_i) r̂_ij / |r_ij|` over its pairs, the
/// finite-difference quotient of the score field: it has the units of a score
/// over a length and scales as `1/λ` under `x → λx`. A row without a valid
/// score holds 0, never NaN, and a nonfinite score is not valid. Displacements
/// of the gradient take the minimum image of
/// `contract::periodic_box(&gas.boundary)`, as the vector channels do; a box
/// of another dimension leaves the gradient zero. The score of a row reads
/// its first cloning entry, the one `companion_fitness` belongs to.
/// `Unavailable`, with `state` untouched, when the frame lacks fitness or
/// cloning companions.
pub fn fill(frame: &Frame, gas: &GasConfig, state: &mut FrameState) -> Availability {
    let (n, d) = (frame.n, frame.d);
    let (Some(fitness), Some(companions), Some(companion_fitness)) =
        (&frame.fitness, &frame.cloning, &frame.companion_fitness)
    else {
        return Availability::unavailable("frame records no fitness or no cloning companions");
    };
    if fitness.len() != n
        || companion_fitness.len() != n
        || frame.eligible.len() != n
        || companions.count == 0
        || companions.valid.len() != n * companions.count
        || frame.x.len() != n * d
    {
        return Availability::unavailable("score frame shape");
    }
    let epsilon = gas.clone_decision.epsilon;
    let mut values = vec![0.; n];
    let mut valid = vec![false; n];
    for i in 0..n {
        // `companion_fitness` belongs to the first cloning entry of the row.
        let sampled = companions.valid[i * companions.count];
        let value = score(fitness[i], companion_fitness[i], epsilon);
        if frame.eligible[i] && sampled && value.is_finite() {
            values[i] = value;
            valid[i] = true;
        }
    }
    // A box of another dimension gives no direction, as in the vector channels.
    let periodic = periodic_box(&gas.boundary);
    let directed = periodic.is_none_or(|p| p.lower.len() == d && p.upper.len() == d);
    let mut gradient = vec![0.; n * d];
    let mut direction = vec![0.; d];
    for i in (0..n).filter(|&i| directed && valid[i]) {
        let mut pairs = 0.;
        for map in [&state.distance_companion, &state.cloning_companion]
            .into_iter()
            .flatten()
        {
            let Some(j) = map
                .get(i)
                .filter(|&&j| j != NO_COMPANION)
                .map(|&j| j as usize)
                .filter(|&j| j < n && valid[j])
            else {
                continue;
            };
            for (k, r) in direction.iter_mut().enumerate() {
                let delta = frame.x[j * d + k] - frame.x[i * d + k];
                *r = periodic.map_or(delta, |p| p.minimum_image(delta, k));
            }
            let length = direction.iter().map(|r| r * r).sum::<f64>().sqrt();
            if !(length.is_finite() && length > 0.) {
                continue;
            }
            for (g, r) in gradient[i * d..(i + 1) * d].iter_mut().zip(&direction) {
                // (ΔS / |r|) r̂ : a difference quotient, not a difference.
                *g += (values[j] - values[i]) * r / (length * length);
            }
            pairs += 1.;
        }
        if pairs > 0. {
            for g in &mut gradient[i * d..(i + 1) * d] {
                *g /= pairs;
            }
        }
    }
    state.role = Some(roles(frame, &values));
    state.score = Some(values);
    state.score_valid = valid;
    state.score_gradient = Some(gradient);
    Availability::Available
}
