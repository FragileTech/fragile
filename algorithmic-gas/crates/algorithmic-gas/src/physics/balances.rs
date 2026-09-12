//! Exact finite-step balances for the recorded algorithm, with unit particle mass.
use super::fitness::Objective;
use crate::{
    GasConfig, GasError, Real, Result,
    error::require,
    kinetic::KineticKind,
    noise::{InnovationLaw, NoiseGeometry},
    tracking::{RecordedStep, StageSnapshot},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T:Real")]
pub struct ThermostatMoments<T: Real> {
    pub decay: T,
    pub innovation_scale_squared: T,
    pub energy_mean_delta: T,
    pub energy_variance: T,
    pub momentum_mean_delta: Vec<T>,
    pub momentum_covariance: Vec<T>,
}
/// B has row-major shape [rows,dimension,rank]; independent standardized
/// innovations across both rows and rank. No boundary/revival is included.
#[allow(clippy::too_many_arguments)] // Explicit tensor shape, clock and noise-law contract.
pub fn thermostat_moments<T: Real>(
    velocities: &[T],
    factor: &[T],
    dimension: usize,
    rank: usize,
    alive: &[bool],
    dt: T,
    friction: T,
    law: InnovationLaw,
) -> Result<ThermostatMoments<T>> {
    let d = dimension;
    let n = alive.len();
    require(
        n > 0
            && d > 0
            && d <= 16
            && rank > 0
            && rank <= 16
            && velocities.len() == n * d
            && factor.len() == n * d * rank,
        "thermostat shape/capacity",
    )?;
    require(
        dt.is_finite()
            && dt > T::ZERO
            && friction.is_finite()
            && friction >= T::ZERO
            && factor.iter().all(|x| x.is_finite())
            && velocities
                .chunks_exact(d)
                .zip(alive)
                .all(|(v, &a)| !a || v.iter().all(|x| x.is_finite())),
        "finite thermostat inputs",
    )?;
    let c = (-friction * dt).exp();
    let two = T::from_f64(2.);
    let half = T::from_f64(0.5);
    let s2 = if friction == T::ZERO {
        dt
    } else {
        -(-two * friction * dt).exp_m1() / (two * friction)
    };
    let mut mean = T::ZERO;
    let mut variance = T::ZERO;
    let mut momentum = vec![T::ZERO; d];
    let mut covariance = vec![T::ZERO; d * d];
    for row in 0..n {
        if !alive[row] {
            continue;
        }
        let v = &velocities[row * d..(row + 1) * d];
        let b = &factor[row * d * rank..(row + 1) * d * rank];
        let kinetic = v.iter().fold(T::ZERO, |s, &x| s + x * x) * half;
        let mut a = vec![T::ZERO; d * d];
        for i in 0..d {
            for j in 0..d {
                for k in 0..rank {
                    a[i * d + j] = a[i * d + j] + b[i * rank + k] * b[j * rank + k];
                }
            }
        }
        let trace = (0..d).fold(T::ZERO, |s, i| s + a[i * d + i]);
        let mut va = T::ZERO;
        for i in 0..d {
            momentum[i] = momentum[i] + (c - T::ONE) * v[i];
            for j in 0..d {
                va = va + v[i] * a[i * d + j] * v[j];
                covariance[i * d + j] = covariance[i * d + j] + s2 * a[i * d + j];
            }
        }
        mean = mean + (c * c - T::ONE) * kinetic + half * s2 * trace;
        let frobenius = a.iter().fold(T::ZERO, |s, &x| s + x * x);
        let mut var = c * c * s2 * va + half * s2 * s2 * frobenius;
        if law == InnovationLaw::StandardizedUniform {
            let mut column_norm4 = T::ZERO;
            for k in 0..rank {
                let norm = (0..d).fold(T::ZERO, |s, i| s + b[i * rank + k] * b[i * rank + k]);
                column_norm4 = column_norm4 + norm * norm;
            }
            var = var - T::from_f64(0.3) * s2 * s2 * column_norm4;
        }
        variance = variance + var;
    }
    require(
        mean.is_finite()
            && variance.is_finite()
            && variance >= T::ZERO
            && momentum.iter().chain(&covariance).all(|x| x.is_finite()),
        "thermostat moment overflow",
    )?;
    Ok(ThermostatMoments {
        decay: c,
        innovation_scale_squared: s2,
        energy_mean_delta: mean,
        energy_variance: variance,
        momentum_mean_delta: momentum,
        momentum_covariance: covariance,
    })
}

/// Translation of the independent innovation coordinates. The centered noise
/// law and its fourth moment remain unchanged; the conditional output mean moves.
#[allow(clippy::too_many_arguments)]
pub fn shifted_thermostat_moments<T: Real>(
    velocities: &[T],
    factor: &[T],
    dimension: usize,
    rank: usize,
    alive: &[bool],
    dt: T,
    friction: T,
    law: InnovationLaw,
    innovation_mean: &[T],
) -> Result<ThermostatMoments<T>> {
    let mut out = thermostat_moments(
        velocities, factor, dimension, rank, alive, dt, friction, law,
    )?;
    let d = dimension;
    require(
        innovation_mean.len() == alive.len() * rank
            && innovation_mean.iter().all(|v| v.is_finite()),
        "innovation mean shape/finiteness",
    )?;
    for row in 0..alive.len() {
        if !alive[row] {
            continue;
        }
        let v = &velocities[row * d..(row + 1) * d];
        let b = &factor[row * d * rank..(row + 1) * d * rank];
        let u: Vec<_> = (0..d)
            .map(|i| {
                (0..rank).fold(T::ZERO, |s, k| {
                    s + b[i * rank + k] * innovation_mean[row * rank + k]
                }) * out.innovation_scale_squared.sqrt()
            })
            .collect();
        for i in 0..d {
            out.momentum_mean_delta[i] = out.momentum_mean_delta[i] + u[i];
            out.energy_mean_delta =
                out.energy_mean_delta + out.decay * v[i] * u[i] + T::from_f64(0.5) * u[i] * u[i];
            for j in 0..d {
                let a = (0..rank).fold(T::ZERO, |s, k| s + b[i * rank + k] * b[j * rank + k]);
                out.energy_variance = out.energy_variance
                    + out.innovation_scale_squared
                        * (T::from_f64(2.) * out.decay * v[i] * a * u[j] + u[i] * a * u[j]);
            }
        }
    }
    require(
        out.energy_mean_delta.is_finite()
            && out.energy_variance.is_finite()
            && out.energy_variance >= T::ZERO
            && out.momentum_mean_delta.iter().all(|x| x.is_finite()),
        "shifted thermostat moment overflow",
    )?;
    Ok(out)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticleMoments {
    pub count: usize,
    pub kinetic_energy: f64,
    pub potential_energy: Option<f64>,
    pub momentum: Vec<f64>,
    /// Sum of centered vv^T. Divide by declared volume for a stress density.
    pub kinetic_stress: Vec<f64>,
}
fn moments(
    stage: &StageSnapshot,
    positions: &str,
    velocities: &str,
    include_truncated: bool,
    potential: Option<&Objective>,
) -> Result<ParticleMoments> {
    let v = stage
        .fields
        .get(velocities)
        .ok_or_else(|| GasError::MissingField(velocities.into()))?;
    let d = v.item_shape.iter().product::<usize>();
    let n = stage.validity.len();
    require(
        d > 0 && d <= 16 && v.values.len() == n * d,
        "balance velocity shape",
    )?;
    let x = stage
        .fields
        .get(positions)
        .ok_or_else(|| GasError::MissingField(positions.into()))?;
    require(x.values.len() == n * d, "balance position shape")?;
    let mut out = ParticleMoments {
        count: 0,
        kinetic_energy: 0.,
        potential_energy: potential.map(|_| 0.),
        momentum: vec![0.; d],
        kinetic_stress: vec![0.; d * d],
    };
    for row in 0..n {
        if !stage.validity[row].eligible(include_truncated) {
            continue;
        }
        let vel = &v.values[row * d..(row + 1) * d];
        require(
            vel.iter().all(|x| x.is_finite()),
            "nonfinite eligible balance velocity",
        )?;
        out.count += 1;
        for (i, &velocity) in vel.iter().enumerate() {
            out.momentum[i] += velocity;
            out.kinetic_energy += 0.5 * velocity * velocity;
        }
        if let Some(objective) = potential {
            let p = objective.value(&x.values[row * d..(row + 1) * d])?;
            out.potential_energy = out.potential_energy.map(|u| u + p);
        }
    }
    if out.count > 0 {
        for row in 0..n {
            if stage.validity[row].eligible(include_truncated) {
                for i in 0..d {
                    for j in 0..d {
                        out.kinetic_stress[i * d + j] += (v.values[row * d + i]
                            - out.momentum[i] / out.count as f64)
                            * (v.values[row * d + j] - out.momentum[j] / out.count as f64);
                    }
                }
            }
        }
    }
    Ok(out)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperatorBalance {
    pub from: String,
    pub to: String,
    pub count_delta: i64,
    pub kinetic_delta: f64,
    pub potential_delta: Option<f64>,
    pub momentum_delta: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BalanceReport {
    pub step: u64,
    pub initial: ParticleMoments,
    pub final_state: ParticleMoments,
    pub operators: Vec<OperatorBalance>,
    pub kinetic_telescoping_residual: f64,
    pub momentum_telescoping_residual: Vec<f64>,
    pub thermostat: Option<ThermostatMoments<f64>>,
    pub thermostat_realized_energy_delta: Option<f64>,
    pub unavailable: Vec<String>,
}
/// Budgets compare actual operator endpoints. Killed/revived slots enter through
/// the alive-count observable, rather than being silently omitted from transfers.
pub fn analyze_step<T: Real>(
    step: &RecordedStep<T>,
    config: &GasConfig,
    potential: Option<&Objective>,
) -> Result<BalanceReport> {
    let (positions, velocities, thermostat) = match &config.kinetic.integrator {
        KineticKind::Baoab {
            positions,
            velocities,
            dt,
            friction,
        } => (
            positions.as_str(),
            velocities.as_str(),
            Some((*dt, *friction)),
        ),
        _ => ("positions", "velocities", None),
    };
    require(step.stages.len() >= 2, "balance stage coverage")?;
    let mut states = Vec::new();
    for stage in &step.stages {
        states.push(moments(
            stage,
            positions,
            velocities,
            config.include_truncated,
            potential,
        )?);
    }
    let initial = states.first().unwrap().clone();
    let final_state = states.last().unwrap().clone();
    let d = initial.momentum.len();
    let mut operators = Vec::new();
    let mut dk = 0.;
    let mut dp = vec![0.; d];
    for i in 1..states.len() {
        let a = &states[i - 1];
        let b = &states[i];
        let momentum: Vec<_> = a
            .momentum
            .iter()
            .zip(&b.momentum)
            .map(|(&a, &b)| b - a)
            .collect();
        let delta = b.kinetic_energy - a.kinetic_energy;
        dk += delta;
        for j in 0..d {
            dp[j] += momentum[j];
        }
        operators.push(OperatorBalance {
            from: step.stages[i - 1].stage.clone(),
            to: step.stages[i].stage.clone(),
            count_delta: b.count as i64 - a.count as i64,
            kinetic_delta: delta,
            potential_delta: a
                .potential_energy
                .zip(b.potential_energy)
                .map(|(a, b)| b - a),
            momentum_delta: momentum,
        });
    }
    let mut unavailable = Vec::new();
    let mut prediction = None;
    let mut realized = None;
    if let Some((dt, friction)) = thermostat {
        let before = step.stages.iter().position(|s| s.stage == "A1");
        let after = step
            .stages
            .iter()
            .position(|s| s.stage == "O_before_boundary");
        let sample = step.noise.iter().rev().find(|s| {
            s.stream == crate::random::Stream::Kinetic
                && s.substep == 2
                && s.factor.is_some()
                && s.geometry.is_some()
                && s.innovation_law.is_some()
        });
        if let (Some(before), Some(after), Some(sample)) = (before, after, sample) {
            let v = &step.stages[before].fields[velocities].values;
            let n = sample.rows;
            let compact = sample.factor.as_ref().unwrap();
            let (rank, width) = match &sample.geometry {
                Some(NoiseGeometry::LowRank { rank, .. }) => (*rank, d * rank),
                Some(NoiseGeometry::Isotropic { .. }) => (d, 1),
                Some(NoiseGeometry::Diagonal { .. }) => (d, d),
                _ => (d, d * d),
            };
            require(compact.len() == n * width, "recorded factor coverage")?;
            let mut factor = vec![0.; n * d * rank];
            for row in 0..n {
                for i in 0..d {
                    for j in 0..rank {
                        factor[(row * d + i) * rank + j] = match &sample.geometry {
                            Some(NoiseGeometry::Isotropic { .. }) => {
                                if i == j {
                                    compact[row]
                                } else {
                                    0.
                                }
                            }
                            Some(NoiseGeometry::Diagonal { .. }) => {
                                if i == j {
                                    compact[row * d + i]
                                } else {
                                    0.
                                }
                            }
                            _ => compact[row * width + i * rank + j],
                        };
                    }
                }
            }
            let alive: Vec<_> = step.stages[before]
                .validity
                .iter()
                .map(|v| v.eligible(config.include_truncated))
                .collect();
            let mut innovation_mean = vec![0.; n * rank];
            for shift in &sample.applied_source_shifts {
                shift.validate(n, rank)?;
                innovation_mean[shift.walker * rank + shift.coordinate] += shift.shift;
            }
            prediction = Some(shifted_thermostat_moments(
                v,
                &factor,
                d,
                rank,
                &alive,
                dt,
                friction,
                sample.innovation_law.unwrap(),
                &innovation_mean,
            )?);
            realized = Some(states[after].kinetic_energy - states[before].kinetic_energy);
        } else {
            unavailable.push("thermostat prediction requires A1, O_before_boundary, and a recorded diffusion factor; recorded coverage is incomplete".into());
        }
    }
    Ok(BalanceReport {
        step: step.report.step,
        kinetic_telescoping_residual: final_state.kinetic_energy - initial.kinetic_energy - dk,
        momentum_telescoping_residual: (0..d)
            .map(|i| final_state.momentum[i] - initial.momentum[i] - dp[i])
            .collect(),
        initial,
        final_state,
        operators,
        thermostat: prediction,
        thermostat_realized_energy_delta: realized,
        unavailable,
    })
}
