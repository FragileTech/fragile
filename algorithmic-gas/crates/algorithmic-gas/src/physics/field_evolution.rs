//! Finite-step weak field equations for the recorded BAOAB gas.
//!
//! Every field is the eligible empirical sum divided by the fixed slot count N.
//! Cloning and boundary loss therefore appear as sources, without renormalizing
//! them away. The O term is integrated analytically under the executed innovation
//! law, conditional on the complete A1 state. Its residual is a martingale
//! increment. Later forces/boundaries are retained as executed sources; they are
//! not claimed to be predictable before the O innovation is drawn.
use crate::{
    GasConfig, GasError, Real, Result,
    error::require,
    kinetic::KineticKind,
    noise::InnovationLaw,
    random::Stream,
    tracking::{RecordedStep, StageSnapshot},
};
use serde::{Deserialize, Serialize};

/// Fourier testing resolves a field spatially without a binning kernel. Stress
/// means the raw second velocity moment, not a postulated constitutive stress.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WeakFieldObservable {
    PhaseSpaceCharacteristic { k: Vec<f64>, l: Vec<f64> },
    Density { k: Vec<f64> },
    Momentum { k: Vec<f64>, component: usize },
    Stress { k: Vec<f64>, a: usize, b: usize },
    KineticEnergy { k: Vec<f64> },
}
impl WeakFieldObservable {
    fn k(&self) -> &[f64] {
        match self {
            Self::PhaseSpaceCharacteristic { k, .. }
            | Self::Density { k }
            | Self::Momentum { k, .. }
            | Self::Stress { k, .. }
            | Self::KineticEnergy { k } => k,
        }
    }
    fn validate(&self, d: usize) -> Result<()> {
        require(
            d > 0 && d <= 16 && self.k().len() == d && self.k().iter().all(|x| x.is_finite()),
            "weak field wavevector shape/finiteness",
        )?;
        match self {
            Self::PhaseSpaceCharacteristic { l, .. } => require(
                l.len() == d && l.iter().all(|x| x.is_finite()),
                "velocity wavevector shape/finiteness",
            ),
            Self::Momentum { component, .. } => require(*component < d, "momentum component"),
            Self::Stress { a, b, .. } => require(*a < d && *b < d, "stress components"),
            _ => Ok(()),
        }
    }
    fn value(&self, x: &[f64], v: &[f64]) -> [f64; 2] {
        let phase = dot(self.k(), x)
            + match self {
                Self::PhaseSpaceCharacteristic { l, .. } => dot(l, v),
                _ => 0.,
            };
        let magnitude = match self {
            Self::Momentum { component, .. } => v[*component],
            Self::Stress { a, b, .. } => v[*a] * v[*b],
            Self::KineticEnergy { .. } => 0.5 * dot(v, v),
            _ => 1.,
        };
        [magnitude * phase.cos(), magnitude * phase.sin()]
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeakStageSource {
    pub from: String,
    pub to: String,
    pub kind: String,
    pub increment: [f64; 2],
    /// Present only for B/A maps independently reconstructed from dt, the
    /// recorded potential gradient and the configured viscous interaction.
    pub deterministic_law_residual: Option<[f64; 2]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeakFieldBalance {
    pub step: u64,
    pub slots: usize,
    pub eligible_at_thermostat: usize,
    pub observable: WeakFieldObservable,
    pub initial: [f64; 2],
    pub final_value: [f64; 2],
    pub stage_sources: Vec<WeakStageSource>,
    pub thermostat_conditional_increment: [f64; 2],
    pub thermostat_realized_increment: [f64; 2],
    pub martingale_increment: [f64; 2],
    /// Row-major real/imaginary conditional covariance, scaled by N^-2.
    /// Built-in innovations are independent across walker and factor indices.
    pub martingale_covariance: [f64; 4],
    pub field_equation_residual: [f64; 2],
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn fields<'a>(
    s: &'a StageSnapshot,
    positions: &str,
    velocities: &str,
) -> Result<(&'a [f64], &'a [f64], usize)> {
    let x = s
        .fields
        .get(positions)
        .ok_or_else(|| GasError::MissingField(positions.into()))?;
    let v = s
        .fields
        .get(velocities)
        .ok_or_else(|| GasError::MissingField(velocities.into()))?;
    require(
        x.item_shape.len() == 1
            && x.item_shape == v.item_shape
            && x.item_shape[0] > 0
            && x.values.len() == s.validity.len() * x.item_shape[0]
            && v.values.len() == x.values.len(),
        "weak field stage shape",
    )?;
    Ok((&x.values, &v.values, x.item_shape[0]))
}
fn evaluate(
    s: &StageSnapshot,
    observable: &WeakFieldObservable,
    positions: &str,
    velocities: &str,
    include_truncated: bool,
) -> Result<[f64; 2]> {
    let (x, v, d) = fields(s, positions, velocities)?;
    observable.validate(d)?;
    let n = s.validity.len();
    require(n > 0, "weak field empty slots")?;
    let mut sum = [0.; 2];
    for i in 0..n {
        if !s.validity[i].eligible(include_truncated) {
            continue;
        }
        let (x, v) = (&x[i * d..(i + 1) * d], &v[i * d..(i + 1) * d]);
        require(
            x.iter().chain(v).all(|a| a.is_finite()),
            "nonfinite eligible weak field input",
        )?;
        let z = observable.value(x, v);
        for j in 0..2 {
            sum[j] += z[j] / n as f64;
        }
    }
    require(sum.iter().all(|x| x.is_finite()), "weak field overflow")?;
    Ok(sum)
}
fn same_recorded_scalar(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}
fn same_snapshot_state(a: &StageSnapshot, b: &StageSnapshot) -> bool {
    a.version == b.version
        && a.generations == b.generations
        && a.validity == b.validity
        && a.fields.len() == b.fields.len()
        && a.fields.iter().all(|(name, field)| {
            b.fields.get(name).is_some_and(|other| {
                field.item_shape == other.item_shape
                    && field.values.len() == other.values.len()
                    && field
                        .values
                        .iter()
                        .zip(&other.values)
                        .all(|(&x, &y)| same_recorded_scalar(x, y))
            })
        })
}
fn required_stage<'a>(
    stages: &'a [StageSnapshot],
    cursor: &mut usize,
    name: &str,
) -> Result<&'a StageSnapshot> {
    let stage = stages
        .get(*cursor)
        .ok_or_else(|| GasError::Capability(format!("required field stage {name} unavailable")))?;
    require(
        stage.stage == name,
        format!("field stage order: expected {name}, found {}", stage.stage),
    )?;
    *cursor += 1;
    Ok(stage)
}
fn clone_stage_coverage<T: Real>(step: &RecordedStep<T>) -> Result<()> {
    require(
        step.stages.len() >= 2
            && step.stages[0].stage == "pre_clone"
            && step.stages[1].stage == "literal_clone"
            && step
                .stages
                .iter()
                .filter(|s| s.stage == "pre_clone")
                .count()
                == 1
            && step
                .stages
                .iter()
                .filter(|s| s.stage == "literal_clone")
                .count()
                == 1,
        "clone field stage order/coverage",
    )?;
    require(
        same_snapshot_state(
            &step.stages[0],
            &StageSnapshot::capture("pre_clone", &step.before),
        ) && step.stages[1].version > step.stages[0].version,
        "clone field input/output provenance",
    )
}
/// Validate the complete recorded built-in BAOAB path. A terminal boundary may
/// end the path early; omitted operations are allowed only after that recorded
/// boundary has actually removed every eligible row.
fn weak_stage_coverage<T: Real>(
    config: &GasConfig,
    step: &RecordedStep<T>,
    positions: &str,
    velocities: &str,
) -> Result<()> {
    clone_stage_coverage(step)?;
    let mut cursor = 2;
    required_stage(&step.stages, &mut cursor, "post_transform")?;
    let mut input = required_stage(&step.stages, &mut cursor, "post_clone")?;
    let has_eligible = |s: &StageSnapshot| {
        s.validity
            .iter()
            .any(|v| v.eligible(config.include_truncated))
    };
    for (name, raw_name, input_name) in [
        ("B1", "B1_before_boundary", Some("B1_input")),
        ("A1", "A1_before_boundary", None),
        ("O", "O_before_boundary", None),
        ("A2", "A2_before_boundary", None),
        ("B2", "B2_before_boundary", Some("B2_input")),
    ] {
        if !has_eligible(input) {
            break;
        }
        if let Some(input_name) = input_name {
            let barrier = required_stage(&step.stages, &mut cursor, input_name)?;
            require(
                same_snapshot_state(input, barrier),
                "read-only gradient evaluation changed recorded population state",
            )?;
            input = barrier;
        }
        let raw = required_stage(&step.stages, &mut cursor, raw_name)?;
        let (x, v, d) = fields(input, positions, velocities)?;
        let (rx, rv, rd) = fields(raw, positions, velocities)?;
        require(
            d == rd
                && input.validity == raw.validity
                && input.generations == raw.generations
                && raw.version > input.version,
            "raw kinetic stage provenance",
        )?;
        for i in 0..input.validity.len() {
            let inactive = !input.validity[i].eligible(config.include_truncated);
            for a in 0..d {
                let offset = i * d + a;
                require(
                    (name.starts_with('A') && !inactive
                        || same_recorded_scalar(x[offset], rx[offset]))
                        && (!name.starts_with('A') && !inactive
                            || same_recorded_scalar(v[offset], rv[offset])),
                    "raw kinetic stage changed an inactive row or its untouched coordinate field",
                )?;
            }
        }
        let boundary = required_stage(&step.stages, &mut cursor, name)?;
        require(
            boundary.version >= raw.version
                && boundary.generations == raw.generations
                && boundary.validity.len() == raw.validity.len(),
            "kinetic boundary provenance",
        )?;
        if config.kinetic.boundary_schedule == crate::kinetic::KineticBoundarySchedule::EndOfStep {
            require(
                same_snapshot_state(raw, boundary),
                "unexpected intermediate boundary change",
            )?;
        }
        input = boundary;
    }
    // Additional canonical stages occur only when BAOAB reaches B2. An
    // actual early terminal boundary returns before these operations.
    if input.stage == "B2" {
        for (enabled, name, raw_name, moves_positions) in [
            (
                config.kinetic.position_diffusion > 0.,
                "position_diffusion",
                "position_diffusion_before_boundary",
                true,
            ),
            (
                config.kinetic.velocity_cap.is_some(),
                "velocity_cap",
                "velocity_cap_before_boundary",
                false,
            ),
        ] {
            if !enabled || !has_eligible(input) {
                continue;
            }
            let raw = required_stage(&step.stages, &mut cursor, raw_name)?;
            let (x, v, d) = fields(input, positions, velocities)?;
            let (rx, rv, rd) = fields(raw, positions, velocities)?;
            require(
                d == rd
                    && input.validity == raw.validity
                    && input.generations == raw.generations
                    && raw.version > input.version,
                "final kinetic stage provenance",
            )?;
            for i in 0..input.validity.len() {
                let inactive = !input.validity[i].eligible(config.include_truncated);
                for a in 0..d {
                    let k = i * d + a;
                    require(
                        (moves_positions && !inactive || same_recorded_scalar(x[k], rx[k]))
                            && (!moves_positions && !inactive || same_recorded_scalar(v[k], rv[k])),
                        "final kinetic stage changed inactive/untouched coordinates",
                    )?;
                }
            }
            let boundary = required_stage(&step.stages, &mut cursor, name)?;
            require(
                boundary.version >= raw.version
                    && boundary.generations == raw.generations
                    && boundary.validity.len() == raw.validity.len(),
                "final kinetic boundary provenance",
            )?;
            if config.kinetic.boundary_schedule
                == crate::kinetic::KineticBoundarySchedule::EndOfStep
            {
                require(
                    same_snapshot_state(raw, boundary),
                    "unexpected intermediate boundary change",
                )?;
            }
            input = boundary;
        }
        if config.kinetic.boundary_schedule == crate::kinetic::KineticBoundarySchedule::EndOfStep {
            let raw = required_stage(&step.stages, &mut cursor, "terminal_before_boundary")?;
            require(
                same_snapshot_state(input, raw),
                "terminal boundary input differs from completed kinetic state",
            )?;
            let boundary = required_stage(&step.stages, &mut cursor, "terminal")?;
            require(
                boundary.version >= raw.version
                    && boundary.generations == raw.generations
                    && boundary.validity.len() == raw.validity.len(),
                "terminal boundary provenance",
            )?;
        }
    }
    let final_stage = required_stage(&step.stages, &mut cursor, "post_kinetic")?;
    require(
        cursor == step.stages.len()
            && same_snapshot_state(
                final_stage,
                &StageSnapshot::capture("post_kinetic", &step.final_population),
            )
            && step
                .stages
                .windows(2)
                .all(|pair| pair[1].version >= pair[0].version),
        "complete field stage coverage/final provenance",
    )
}
fn characteristic(law: InnovationLaw, q: &[f64]) -> f64 {
    match law {
        InnovationLaw::Gaussian => (-0.5 * dot(q, q)).exp(),
        InnovationLaw::StandardizedUniform => q
            .iter()
            .map(|q| {
                let z = 3_f64.sqrt() * q;
                if z.abs() < 1e-5 {
                    1. - z * z / 6. + z.powi(4) / 120.
                } else {
                    z.sin() / z
                }
            })
            .product(),
    }
}
/// Conditional mean and scalar variance of a polynomial in m+L xi, where xi
/// has independent standardized Gaussian or uniform components. The fourth
/// cumulant is 0 or -6/5 respectively; no Gaussian substitution is used.
fn polynomial_moments(
    observable: &WeakFieldObservable,
    m: &[f64],
    factor: &[f64],
    rank: usize,
    law: InnovationLaw,
) -> (f64, f64) {
    let d = m.len();
    let mut linear = vec![0.; d];
    let mut q = vec![0.; d * d];
    let constant = match observable {
        WeakFieldObservable::Density { .. } => 1.,
        WeakFieldObservable::Momentum { component, .. } => {
            linear[*component] = 1.;
            0.
        }
        WeakFieldObservable::Stress { a, b, .. } => {
            q[a * d + b] += 0.5;
            q[b * d + a] += 0.5;
            0.
        }
        WeakFieldObservable::KineticEnergy { .. } => {
            for a in 0..d {
                q[a * d + a] = 0.5;
            }
            0.
        }
        _ => unreachable!("characteristic uses its exact Fourier factor"),
    };
    let mut covariance = vec![0.; d * d];
    for a in 0..d {
        for b in 0..d {
            covariance[a * d + b] = dot(
                &factor[a * rank..(a + 1) * rank],
                &factor[b * rank..(b + 1) * rank],
            );
        }
    }
    let mut mean = constant + dot(&linear, m);
    let mut effective = linear;
    for a in 0..d {
        for b in 0..d {
            mean += q[a * d + b] * (m[a] * m[b] + covariance[a * d + b]);
            effective[a] += 2. * q[a * d + b] * m[b];
        }
    }
    let mut variance = 0.;
    for a in 0..d {
        for b in 0..d {
            variance += effective[a] * covariance[a * d + b] * effective[b];
            for c in 0..d {
                for e in 0..d {
                    variance += 2.
                        * q[a * d + b]
                        * covariance[b * d + c]
                        * q[c * d + e]
                        * covariance[e * d + a];
                }
            }
        }
    }
    if law == InnovationLaw::StandardizedUniform {
        for r in 0..rank {
            let mut column = 0.;
            for a in 0..d {
                for b in 0..d {
                    column += factor[a * rank + r] * q[a * d + b] * factor[b * rank + r];
                }
            }
            variance -= 1.2 * column * column;
        }
    }
    (mean, variance.max(0.))
}
fn deterministic_prediction<T: Real>(
    config: &GasConfig,
    step: &RecordedStep<T>,
    before: &StageSnapshot,
    after: &StageSnapshot,
    positions: &str,
    velocities: &str,
    dt: f64,
) -> Result<Option<StageSnapshot>> {
    let stage = after.stage.as_str();
    let drift = matches!(stage, "A1_before_boundary" | "A2_before_boundary");
    let kick = matches!(stage, "B1_before_boundary" | "B2_before_boundary");
    if stage == "velocity_cap_before_boundary" {
        let radius = config
            .kinetic
            .velocity_cap
            .ok_or_else(|| GasError::Capability("unconfigured velocity cap".into()))?;
        let (_, v, d) = fields(before, positions, velocities)?;
        let mut predicted = before.clone();
        let target = predicted.fields.get_mut(velocities).unwrap();
        for i in 0..before.validity.len() {
            if !before.validity[i].eligible(config.include_truncated) {
                continue;
            }
            let row = &v[i * d..(i + 1) * d];
            let scale = row.iter().fold(0_f64, |m, x| m.max(x.abs()));
            if scale > 0. {
                let length = row.iter().map(|x| (x / scale).powi(2)).sum::<f64>().sqrt();
                // The engine's two branches: neither ratio can overflow.
                let factor = if scale <= radius {
                    1. / (1. + (scale / radius) * length)
                } else {
                    (radius / scale) / (radius / scale + length)
                };
                for (a, &value) in row.iter().enumerate() {
                    target.values[i * d + a] = value * factor;
                }
            }
        }
        return Ok(Some(predicted));
    }
    if !drift && !kick {
        return Ok(None);
    }
    let (x, v, d) = fields(before, positions, velocities)?;
    let n = before.validity.len();
    let active: Vec<_> = before
        .validity
        .iter()
        .map(|s| s.eligible(config.include_truncated))
        .collect();
    let mut change = vec![0.; n * d];
    if drift {
        change.copy_from_slice(v);
    } else {
        let label = if stage.starts_with("B1") { "B1" } else { "B2" };
        let gradient = step
            .field_evaluations
            .iter()
            .find(|f| f.stage == label && f.field == "potential_gradient")
            .ok_or_else(|| {
                GasError::Capability("recorded potential gradient unavailable".into())
            })?;
        require(
            gradient.version == before.version && gradient.values.len() == n * d,
            "kick gradient provenance",
        )?;
        for (change, gradient) in change.iter_mut().zip(&gradient.values) {
            *change = -gradient;
        }
        if let Some(viscosity) = &config.qft.viscosity {
            for i in 0..n {
                if !active[i] {
                    continue;
                }
                let mut weights = vec![0.; n];
                for j in 0..n {
                    if i != j && active[j] {
                        let distance = (0..d)
                            .map(|a| (x[i * d + a] - x[j * d + a]).powi(2))
                            .sum::<f64>();
                        weights[j] = (-distance / (2. * viscosity.bandwidth.powi(2))).exp();
                    }
                }
                let normalization = if viscosity.row_normalized {
                    weights.iter().sum()
                } else {
                    active.iter().filter(|a| **a).count() as f64
                };
                if normalization > 0. {
                    for a in 0..d {
                        for j in 0..n {
                            if weights[j] > 0. {
                                change[i * d + a] += viscosity.coefficient
                                    * weights[j]
                                    * (v[j * d + a] - v[i * d + a])
                                    / normalization;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut predicted = before.clone();
    let target = predicted
        .fields
        .get_mut(if drift { positions } else { velocities })
        .unwrap();
    for i in 0..n {
        if active[i] {
            for a in 0..d {
                target.values[i * d + a] += 0.5 * dt * change[i * d + a];
            }
        }
    }
    Ok(Some(predicted))
}

/// Compute an exact finite-step field balance and an analytic conditional O
/// prediction. Requires built-in BAOAB noise coverage. Non-O sources are retained
/// individually, including complete clone transforms and every boundary update.
pub fn weak_field_balance<T: Real>(
    config: &GasConfig,
    step: &RecordedStep<T>,
    observable: &WeakFieldObservable,
) -> Result<WeakFieldBalance> {
    let KineticKind::Baoab {
        positions,
        velocities,
        dt,
        friction,
    } = &config.kinetic.integrator
    else {
        return Err(GasError::Capability(
            "weak field equations require BAOAB".into(),
        ));
    };
    require(
        dt.is_finite() && *dt > 0. && friction.is_finite() && *friction >= 0.,
        "weak field kinetic clock",
    )?;
    weak_stage_coverage(config, step, positions, velocities)?;
    let before = step
        .stages
        .iter()
        .find(|s| s.stage == "A1")
        .ok_or_else(|| GasError::Capability("A1 stage unavailable".into()))?;
    let after = step
        .stages
        .iter()
        .find(|s| s.stage == "O_before_boundary")
        .ok_or_else(|| GasError::Capability("raw O stage unavailable".into()))?;
    let (x, v, d) = fields(before, positions, velocities)?;
    observable.validate(d)?;
    let n = before.validity.len();
    require(
        n > 0 && after.validity == before.validity,
        "raw O eligibility changed",
    )?;
    let noise = step
        .noise
        .iter()
        .find(|s| s.stream == Stream::Kinetic && s.substep == 2)
        .ok_or_else(|| GasError::Capability("thermostat noise unavailable".into()))?;
    let law = noise
        .innovation_law
        .ok_or_else(|| GasError::Capability("thermostat innovation law unavailable".into()))?;
    require(
        noise.rows == n
            && noise.dimension == d
            && noise.step == step.report.step
            && noise.sample.len() == n * d,
        "thermostat noise shape/address",
    )?;
    let executed = step
        .field_evaluations
        .iter()
        .find(|f| f.stage == "O" && f.field == "executed_noise")
        .ok_or_else(|| GasError::Capability("executed thermostat noise unavailable".into()))?;
    require(
        executed.version == before.version
            && executed.values == noise.sample
            && executed.rows == n
            && executed.item_shape == [d]
            && executed.available
                == before
                    .validity
                    .iter()
                    .map(|s| s.eligible(config.include_truncated))
                    .collect::<Vec<_>>(),
        "thermostat provider differs from recorded conditional law",
    )?;
    let decay = (-friction * dt).exp();
    let scale = if *friction == 0. {
        dt.sqrt()
    } else {
        (-(-2. * friction * dt).exp_m1() / (2. * friction)).sqrt()
    };
    let (out_x, out_v, _) = fields(after, positions, velocities)?;
    let tolerance = if T::PRECISION == crate::Precision::F32 {
        2e-5
    } else {
        1e-10
    };
    require(
        step.stages
            .windows(2)
            .filter(|s| s[0].stage == "A1" && s[1].stage == "O_before_boundary")
            .count()
            == 1,
        "thermostat stage order/coverage",
    )?;
    let mut predicted = [0.; 2];
    let mut covariance = [0.; 4];
    let mut eligible = 0;
    for i in 0..n {
        if !before.validity[i].eligible(config.include_truncated) {
            continue;
        }
        eligible += 1;
        let mut factor = noise.dense_factor(i)?;
        require(
            factor.iter().all(|x| x.is_finite()),
            "nonfinite thermostat factor",
        )?;
        let rank = factor.len() / d;
        require(rank > 0, "thermostat innovation rank must be positive")?;
        let mut shift = vec![0.; rank];
        for source in &noise.applied_source_shifts {
            source.validate(n, rank)?;
            require(
                source.step == noise.step
                    && source.stream == noise.stream
                    && source.substep == noise.substep,
                "innovation source address mismatch",
            )?;
            if source.walker == i {
                shift[source.coordinate] += source.shift;
            }
        }
        let raw = noise
            .raw_innovation
            .as_ref()
            .ok_or_else(|| GasError::Capability("raw thermostat innovations unavailable".into()))?;
        require(raw.len() == n * rank, "raw thermostat innovation shape")?;
        for a in 0..d {
            let sample = dot(
                &factor[a * rank..(a + 1) * rank],
                &raw[i * rank..(i + 1) * rank],
            );
            require(
                (sample - noise.sample[i * d + a]).abs() <= tolerance * (1. + sample.abs()),
                "recorded factor does not generate executed innovation",
            )?;
            require(
                out_x[i * d + a] == x[i * d + a],
                "raw thermostat changed positions",
            )?;
        }
        for f in &mut factor {
            *f *= scale;
        }
        let mean: Vec<_> = (0..d)
            .map(|a| decay * v[i * d + a] + dot(&factor[a * rank..(a + 1) * rank], &shift))
            .collect();
        for a in 0..d {
            let expected = decay * v[i * d + a] + scale * noise.sample[i * d + a];
            require(
                (out_v[i * d + a] - expected).abs() <= tolerance * (1. + expected.abs()),
                "executed O velocity differs from configured BAOAB law",
            )?;
        }
        let phase = dot(observable.k(), &x[i * d..(i + 1) * d]);
        let (row_mean, row_covariance) =
            if let WeakFieldObservable::PhaseSpaceCharacteristic { l, .. } = observable {
                let q: Vec<_> = (0..rank)
                    .map(|r| (0..d).map(|a| l[a] * factor[a * rank + r]).sum())
                    .collect();
                let phi = characteristic(law, &q);
                let phi2 = characteristic(law, &q.iter().map(|x| 2. * x).collect::<Vec<_>>());
                let angle = phase + dot(l, &mean);
                let m = [angle.cos() * phi, angle.sin() * phi];
                let cross = 0.5 * (2. * angle).sin() * phi2 - m[0] * m[1];
                (
                    m,
                    [
                        (0.5 * (1. + (2. * angle).cos() * phi2) - m[0] * m[0]).max(0.),
                        cross,
                        cross,
                        (0.5 * (1. - (2. * angle).cos() * phi2) - m[1] * m[1]).max(0.),
                    ],
                )
            } else {
                let (m, var) = polynomial_moments(observable, &mean, &factor, rank, law);
                let (c, s) = (phase.cos(), phase.sin());
                (
                    [c * m, s * m],
                    [c * c * var, c * s * var, c * s * var, s * s * var],
                )
            };
        for j in 0..2 {
            predicted[j] += row_mean[j] / n as f64;
        }
        for j in 0..4 {
            covariance[j] += row_covariance[j] / (n as f64 * n as f64);
        }
    }
    let input = evaluate(
        before,
        observable,
        positions,
        velocities,
        config.include_truncated,
    )?;
    let output = evaluate(
        after,
        observable,
        positions,
        velocities,
        config.include_truncated,
    )?;
    let conditional = sub(predicted, input);
    let realized = sub(output, input);
    let martingale = sub(output, predicted);
    let mut snapshots = vec![StageSnapshot::capture("initial", &step.before)];
    snapshots.extend(step.stages.iter().cloned());
    snapshots.push(StageSnapshot::capture("final", &step.final_population));
    let initial = evaluate(
        &snapshots[0],
        observable,
        positions,
        velocities,
        config.include_truncated,
    )?;
    let final_value = evaluate(
        snapshots.last().unwrap(),
        observable,
        positions,
        velocities,
        config.include_truncated,
    )?;
    let mut sources = Vec::new();
    let mut rhs = conditional;
    for pair in snapshots.windows(2) {
        if pair[0].stage == "A1" && pair[1].stage == "O_before_boundary" {
            continue;
        }
        let a = evaluate(
            &pair[0],
            observable,
            positions,
            velocities,
            config.include_truncated,
        )?;
        let b = evaluate(
            &pair[1],
            observable,
            positions,
            velocities,
            config.include_truncated,
        )?;
        let increment = sub(b, a);
        for j in 0..2 {
            rhs[j] += increment[j];
        }
        let predicted =
            deterministic_prediction(config, step, &pair[0], &pair[1], positions, velocities, *dt)?;
        let residual = predicted
            .as_ref()
            .map(|p| {
                evaluate(
                    p,
                    observable,
                    positions,
                    velocities,
                    config.include_truncated,
                )
                .map(|p| sub(b, p))
            })
            .transpose()?;
        let kind = if pair[1].stage == "position_diffusion_before_boundary" {
            "position_diffusion"
        } else if pair[1].stage == "velocity_cap_before_boundary" {
            "velocity_cap"
        } else if pair[0].stage.ends_with("before_boundary") {
            "boundary"
        } else if pair[1].stage == "literal_clone" {
            "clone_replacement"
        } else if pair[1].stage == "post_transform" {
            "clone_transform"
        } else if residual.is_some() {
            "deterministic_kinetic"
        } else {
            "observation_or_eligibility"
        };
        sources.push(WeakStageSource {
            from: pair[0].stage.clone(),
            to: pair[1].stage.clone(),
            kind: kind.into(),
            increment,
            deterministic_law_residual: residual,
        });
    }
    let residual = sub(sub(sub(final_value, initial), rhs), martingale);
    require(
        predicted
            .iter()
            .chain(&covariance)
            .chain(&martingale)
            .all(|x| x.is_finite()),
        "weak field prediction overflow",
    )?;
    Ok(WeakFieldBalance {
        step: step.report.step,
        slots: n,
        eligible_at_thermostat: eligible,
        observable: observable.clone(),
        initial,
        final_value,
        stage_sources: sources,
        thermostat_conditional_increment: conditional,
        thermostat_realized_increment: realized,
        martingale_increment: martingale,
        martingale_covariance: covariance,
        field_equation_residual: residual,
    })
}

/// Clone gate equation conditional on the complete source pool, donor candidates,
/// rescored fitnesses, and (for revival) the selected donor. This integrates the
/// independent acceptance gates; it does not integrate donor selection itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloneFieldBalance {
    pub step: u64,
    pub slots: usize,
    pub candidate_historical_sources: usize,
    pub accepted_historical_sources: usize,
    pub revival_sources: usize,
    pub conditional_increment: [f64; 2],
    pub realized_increment: [f64; 2],
    pub martingale_increment: [f64; 2],
    pub martingale_covariance: [f64; 4],
    pub literal_copy_residual: [f64; 2],
    pub max_acceptance_probability_residual: f64,
}

pub fn clone_field_balance<T: Real>(
    archive: &crate::tracking::RunArchive<T>,
    step: &RecordedStep<T>,
    observable: &WeakFieldObservable,
) -> Result<CloneFieldBalance> {
    let config = &archive.gas_config;
    let KineticKind::Baoab {
        positions,
        velocities,
        ..
    } = &config.kinetic.integrator
    else {
        return Err(GasError::Capability(
            "clone field equation requires named phase-space fields".into(),
        ));
    };
    clone_stage_coverage(step)?;
    let before = step
        .stages
        .iter()
        .find(|s| s.stage == "pre_clone")
        .ok_or_else(|| GasError::Capability("pre-clone stage unavailable".into()))?;
    let after = step
        .stages
        .iter()
        .find(|s| s.stage == "literal_clone")
        .ok_or_else(|| GasError::Capability("literal clone stage unavailable".into()))?;
    let (x, v, d) = fields(before, positions, velocities)?;
    observable.validate(d)?;
    let n = before.validity.len();
    let plan = &step.report.clone_plan;
    require(
        n > 0
            && plan.choices.len() == n
            && plan.population_version == before.version
            && step.report.pre_clone_fitness.fitness.len() == n
            && step.donor_fitness.len() == plan.sources.len(),
        "clone field plan shape/provenance",
    )?;
    let mut result = CloneFieldBalance {
        step: step.report.step,
        slots: n,
        candidate_historical_sources: 0,
        accepted_historical_sources: 0,
        revival_sources: 0,
        conditional_increment: [0.; 2],
        realized_increment: [0.; 2],
        martingale_increment: [0.; 2],
        martingale_covariance: [0.; 4],
        literal_copy_residual: [0.; 2],
        max_acceptance_probability_residual: 0.,
    };
    let tolerance = if T::PRECISION == crate::Precision::F32 {
        2e-5
    } else {
        1e-10
    };
    let mut expected_realized = [0.; 2];
    for (i, choice) in plan.choices.iter().enumerate() {
        let recorded_probability = choice.probability.ok_or_else(|| {
            GasError::Capability("clone acceptance probability unavailable".into())
        })?;
        require(
            recorded_probability.is_finite() && (0. ..=1.).contains(&recorded_probability),
            "clone probability outside [0,1]",
        )?;
        if choice.donors.is_empty() {
            require(
                !choice.accepted && !choice.revival && recorded_probability == 0.,
                "empty clone candidate has nonzero acceptance",
            )?;
            continue;
        }
        require(
            choice.donors.len() == 1 && choice.donors[0].weight == 1.,
            "clone field law requires literal unit-weight donor",
        )?;
        let pool_index = choice.donors[0].pool_index as usize;
        let source = plan
            .sources
            .get(pool_index)
            .ok_or_else(|| GasError::Shape("clone source pool index".into()))?;
        require(
            source.frame < step.report.step,
            "clone source is not causal",
        )?;
        let source_population = archive
            .steps
            .iter()
            .filter(|s| s.epoch == step.epoch && s.report.step.checked_sub(1) == Some(source.frame))
            .map(|s| &s.before)
            .chain(
                archive
                    .anchors
                    .iter()
                    .filter(|a| a.epoch == step.epoch && a.step == source.frame)
                    .map(|a| &a.population),
            )
            .find(|p| {
                p.version == source.version
                    && (source.slot as usize) < p.len()
                    && p.generations[source.slot as usize] == source.generation
            })
            .ok_or_else(|| {
                GasError::Capability(
                    "exact historical donor incarnation outside archive coverage".into(),
                )
            })?;
        let slot = source.slot as usize;
        require(
            source_population.validity[slot].eligible(config.include_truncated),
            "clone source is ineligible",
        )?;
        let source_x = source_population.observations.field(positions)?;
        let source_v = source_population.observations.field(velocities)?;
        require(
            source_x.width() == d && source_v.width() == d,
            "clone source field shape",
        )?;
        let sx: Vec<_> = source_x.row(slot)?.iter().map(|v| v.to_f64()).collect();
        let sv: Vec<_> = source_v.row(slot)?.iter().map(|v| v.to_f64()).collect();
        require(
            sx.iter().chain(&sv).all(|x| x.is_finite()),
            "clone source field nonfinite",
        )?;
        let source_value = observable.value(&sx, &sv);
        let alive = before.validity[i].eligible(config.include_truncated);
        let recipient = if alive {
            observable.value(&x[i * d..(i + 1) * d], &v[i * d..(i + 1) * d])
        } else {
            [0.; 2]
        };
        let delta = sub(source_value, recipient);
        let probability = if !alive {
            require(
                choice.revival && choice.accepted,
                "ineligible recipient lacks mandatory revival",
            )?;
            result.revival_sources += 1;
            1.
        } else {
            require(
                !choice.revival,
                "eligible recipient incorrectly labeled revival",
            )?;
            let f = step.report.pre_clone_fitness.fitness[i];
            let donor = step.donor_fitness[pool_index];
            require(
                f.is_finite() && f > T::ZERO && donor.is_finite() && donor > T::ZERO,
                "clone conditional fitness positivity",
            )?;
            // The executed acceptance law, including its cloning period, in
            // the same scalar precision.
            config
                .clone_decision
                .acceptance_probability(step.report.step, f, donor)
                .to_f64()
        };
        let probability_error = (probability - recorded_probability).abs();
        result.max_acceptance_probability_residual = result
            .max_acceptance_probability_residual
            .max(probability_error);
        require(
            probability_error <= tolerance,
            "recorded acceptance probability differs from configured fitness law",
        )?;
        // Verify the address used by the built-in gate. This is a local replay,
        // not a draw from or mutation of the running engine random state.
        if alive {
            let mut rng = crate::random::RandomStream::new(
                config.seed,
                step.report.step,
                Stream::Accept,
                i as u64,
                0,
            );
            require(
                (rng.uniform::<T>().to_f64() < probability) == choice.accepted,
                "clone acceptance does not match independent addressed gate",
            )?;
        }
        let historical = source.frame < step.report.step - 1;
        result.candidate_historical_sources += usize::from(historical);
        result.accepted_historical_sources += usize::from(historical && choice.accepted);
        for a in 0..2 {
            result.conditional_increment[a] += probability * delta[a] / n as f64;
            if choice.accepted {
                expected_realized[a] += delta[a] / n as f64;
            }
            for b in 0..2 {
                result.martingale_covariance[a * 2 + b] +=
                    probability * (1. - probability) * delta[a] * delta[b] / (n as f64 * n as f64);
            }
        }
    }
    result.realized_increment = sub(
        evaluate(
            after,
            observable,
            positions,
            velocities,
            config.include_truncated,
        )?,
        evaluate(
            before,
            observable,
            positions,
            velocities,
            config.include_truncated,
        )?,
    );
    result.martingale_increment = sub(result.realized_increment, result.conditional_increment);
    result.literal_copy_residual = sub(result.realized_increment, expected_realized);
    require(
        result
            .literal_copy_residual
            .iter()
            .all(|r| r.abs() <= tolerance),
        "literal clone output differs from exact recorded donor copies",
    )?;
    Ok(result)
}

/// A component-collision law conditions on the realized donor/gate graph and
/// the already drawn position jitter. Its only remaining innovations are one
/// independent shared Haar O(d) rotation per connected component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollisionFieldBalance {
    pub step: u64,
    pub slots: usize,
    pub component_slots: Vec<Vec<usize>>,
    pub conditioning: String,
    pub conditional_increment: [f64; 2],
    pub realized_increment: [f64; 2],
    pub martingale_increment: [f64; 2],
    /// Real/imaginary conditional covariance, including every pair of walkers
    /// in the same component. Distinct components use independent rotations.
    pub martingale_covariance: [f64; 4],
    pub maximum_momentum_conservation_residual: f64,
    pub maximum_relative_energy_residual: f64,
}

fn sphere_characteristic(d: usize, q: f64) -> Result<f64> {
    match d {
        1 => Ok(q.cos()),
        3 => Ok(if q.abs() < 1e-5 {
            1. - q * q / 6. + q.powi(4) / 120.
        } else {
            q.sin() / q
        }),
        _ => Err(GasError::Capability(
            "Haar phase-space characteristic currently requires dimension 1 or 3; no Gaussian substitute is used".into(),
        )),
    }
}
fn collision_polynomial_mean(
    observable: &WeakFieldObservable,
    u: &[f64],
    r: &[f64],
    e: f64,
) -> f64 {
    let d = u.len() as f64;
    match observable {
        WeakFieldObservable::Density { .. } => 1.,
        WeakFieldObservable::Momentum { component, .. } => u[*component],
        WeakFieldObservable::Stress { a, b, .. } => {
            u[*a] * u[*b] + if a == b { e * e * dot(r, r) / d } else { 0. }
        }
        WeakFieldObservable::KineticEnergy { .. } => 0.5 * (dot(u, u) + e * e * dot(r, r)),
        _ => unreachable!("phase-space fields use the spherical characteristic"),
    }
}
/// Haar fourth moments for one shared rotation. With z_i=R r_i,
/// E[z_ia z_ib z_jc z_jd]=A δ_ab δ_cd+B(δ_ac δ_bd+δ_ad δ_bc).
/// The d=1 quadratic term is deterministic and is handled without division by d-1.
fn collision_polynomial_covariance(
    observable: &WeakFieldObservable,
    u: &[f64],
    ri: &[f64],
    rj: &[f64],
    e: f64,
) -> f64 {
    let d = u.len() as f64;
    let t = dot(ri, rj);
    match observable {
        WeakFieldObservable::Density { .. } => 0.,
        WeakFieldObservable::Momentum { .. } => e * e * t / d,
        WeakFieldObservable::KineticEnergy { .. } => e * e * t * dot(u, u) / d,
        WeakFieldObservable::Stress { a, b, .. } => {
            let diagonal = a == b;
            let linear = e * e * t / d
                * (u[*a].powi(2) + u[*b].powi(2) + if diagonal { 2. * u[*a] * u[*b] } else { 0. });
            if u.len() == 1 {
                return linear;
            }
            let ab = dot(ri, ri) * dot(rj, rj);
            let denominator = d * (d - 1.) * (d + 2.);
            let a4 = ((d + 1.) * ab - 2. * t * t) / denominator;
            let b4 = (d * t * t - ab) / denominator;
            let fourth = if diagonal {
                a4 + 2. * b4 - ab / (d * d)
            } else {
                b4
            };
            linear + e.powi(4) * fourth
        }
        _ => unreachable!("phase-space fields use the spherical characteristic"),
    }
}

pub fn collision_field_balance<T: Real>(
    config: &GasConfig,
    step: &RecordedStep<T>,
    observable: &WeakFieldObservable,
) -> Result<CollisionFieldBalance> {
    let e = config
        .clone_transform
        .restitution
        .ok_or_else(|| GasError::Capability("component collision is not enabled".into()))?;
    require(
        e.is_finite() && (0. ..=1.).contains(&e),
        "collision restitution",
    )?;
    let KineticKind::Baoab {
        positions,
        velocities,
        ..
    } = &config.kinetic.integrator
    else {
        return Err(GasError::Capability(
            "collision field requires named phase-space coordinates".into(),
        ));
    };
    clone_stage_coverage(step)?;
    let literal = &step.stages[1];
    let mut cursor = 2;
    let transformed = required_stage(&step.stages, &mut cursor, "post_transform")?;
    let (x, w, d) = fields(transformed, positions, velocities)?;
    let (_, literal_v, literal_d) = fields(literal, positions, velocities)?;
    let (_, old_v, old_d) = fields(&step.stages[0], positions, velocities)?;
    observable.validate(d)?;
    let n = transformed.validity.len();
    require(
        n > 0
            && n == step.before.len()
            && old_d == d
            && literal_d == d
            && step.report.clone_plan.choices.len() == n
            && literal.validity == transformed.validity,
        "raw collision shape/eligibility",
    )?;
    // Construct the graph independently of its numerical recording. Accepted
    // historical edges are not this current-population collision law.
    let mut adjacency = vec![Vec::new(); n];
    for (i, choice) in step.report.clone_plan.choices.iter().enumerate() {
        if !choice.accepted {
            continue;
        }
        require(
            choice.donors.len() == 1 && choice.donors[0].weight == 1.,
            "collision donor shape",
        )?;
        let source = step
            .report
            .clone_plan
            .sources
            .get(choice.donors[0].pool_index as usize)
            .ok_or_else(|| GasError::Shape("collision donor source".into()))?;
        let j = source.slot as usize;
        require(
            j < n
                && source.frame.checked_add(1) == Some(step.report.step)
                && source.version == step.before.version
                && source.generation == step.before.generations[j],
            "collision requires exact current donor",
        )?;
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for i in 0..n {
        if seen[i] || adjacency[i].is_empty() {
            continue;
        }
        let mut members = vec![i];
        seen[i] = true;
        let mut next = 0;
        while next < members.len() {
            for &j in &adjacency[members[next]] {
                if !seen[j] {
                    seen[j] = true;
                    members.push(j);
                }
            }
            next += 1;
        }
        members.sort_unstable();
        components.push(members);
    }
    let mut predicted = [0.; 2];
    let mut baseline = [0.; 2];
    let mut observed = [0.; 2];
    let mut covariance = [0.; 4];
    let mut momentum_error: f64 = 0.;
    let mut energy_error: f64 = 0.;
    for i in 0..n {
        if !transformed.validity[i].eligible(config.include_truncated) {
            continue;
        }
        let point = &x[i * d..(i + 1) * d];
        let b = observable.value(point, &literal_v[i * d..(i + 1) * d]);
        let o = observable.value(point, &w[i * d..(i + 1) * d]);
        for a in 0..2 {
            baseline[a] += b[a] / n as f64;
            observed[a] += o[a] / n as f64;
            if !seen[i] {
                predicted[a] += o[a] / n as f64;
            }
        }
    }
    let field = |name: &str, shape: &[usize]| -> Result<&crate::tracking::FieldEvaluation> {
        let values = step
            .field_evaluations
            .iter()
            .filter(|f| f.stage == "component_collision" && f.field == name)
            .collect::<Vec<_>>();
        require(
            values.len() == 1,
            format!("collision field {name} coverage"),
        )?;
        let f = values[0];
        require(
            f.version == step.before.version
                && f.rows == n
                && f.item_shape == shape
                && f.available == seen
                && f.values.len() == n * shape.iter().product::<usize>(),
            format!("collision field {name} provenance"),
        )?;
        Ok(f)
    };
    let ids = field("collision_component_id", &[1])?;
    let sizes = field("collision_component_size", &[1])?;
    let input = field("collision_input_velocity", &[d])?;
    let centers = field("collision_center_of_mass", &[d])?;
    let rotations = field("collision_rotation", &[d, d])?;
    let output = field("collision_output_velocity", &[d])?;
    let tolerance = if T::PRECISION == crate::Precision::F32 {
        3e-5
    } else {
        2e-9
    };
    for members in &components {
        let mut u = vec![0.; d];
        for &i in members {
            require(
                transformed.validity[i].eligible(config.include_truncated),
                "component includes unavailable output",
            )?;
            for a in 0..d {
                require(
                    old_v[i * d + a].is_finite(),
                    "nonfinite frozen collision velocity",
                )?;
                u[a] += old_v[i * d + a] / members.len() as f64;
            }
        }
        let relative = members
            .iter()
            .map(|&i| (0..d).map(|a| old_v[i * d + a] - u[a]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let rotation = &rotations.values[members[0] * d * d..(members[0] + 1) * d * d];
        for a in 0..d {
            for b in 0..d {
                let inner = (0..d)
                    .map(|c| rotation[c * d + a] * rotation[c * d + b])
                    .sum::<f64>();
                require(
                    (inner - if a == b { 1. } else { 0. }).abs() <= tolerance,
                    "collision rotation is not orthogonal",
                )?;
            }
        }
        let mut initial_energy = 0.;
        let mut final_energy = 0.;
        let mut before_momentum = vec![0.; d];
        let mut after_momentum = vec![0.; d];
        let mut means = Vec::new();
        let mut phases = Vec::new();
        for (index, &i) in members.iter().enumerate() {
            require(
                ids.values[i] == members[0] as f64 && sizes.values[i] == members.len() as f64,
                "recorded component differs from accepted graph",
            )?;
            require(
                rotations.values[i * d * d..(i + 1) * d * d] == *rotation,
                "component must share one rotation",
            )?;
            for a in 0..d {
                let expected = u[a] + e * dot(&rotation[a * d..(a + 1) * d], &relative[index]);
                require(
                    (input.values[i * d + a] - old_v[i * d + a]).abs() <= tolerance
                        && (centers.values[i * d + a] - u[a]).abs()
                            <= tolerance * (1. + u[a].abs())
                        && (output.values[i * d + a] - w[i * d + a]).abs()
                            <= tolerance * (1. + w[i * d + a].abs())
                        && (expected - w[i * d + a]).abs() <= tolerance * (1. + expected.abs()),
                    "collision output differs from frozen shared-rotation law",
                )?;
                before_momentum[a] += old_v[i * d + a];
                after_momentum[a] += w[i * d + a];
                initial_energy += 0.5 * relative[index][a].powi(2);
                final_energy += 0.5 * (w[i * d + a] - u[a]).powi(2);
            }
            let phase = dot(observable.k(), &x[i * d..(i + 1) * d]);
            let m = if let WeakFieldObservable::PhaseSpaceCharacteristic { l, .. } = observable {
                let phi = sphere_characteristic(
                    d,
                    e * dot(l, l).sqrt() * dot(&relative[index], &relative[index]).sqrt(),
                )?;
                let angle = phase + dot(l, &u);
                phases.push(angle);
                [angle.cos() * phi, angle.sin() * phi]
            } else {
                phases.push(phase);
                let mean = collision_polynomial_mean(observable, &u, &relative[index], e);
                [phase.cos() * mean, phase.sin() * mean]
            };
            for a in 0..2 {
                predicted[a] += m[a] / n as f64;
            }
            means.push(m);
        }
        for a in 0..d {
            momentum_error = momentum_error.max((after_momentum[a] - before_momentum[a]).abs());
        }
        energy_error = energy_error.max((final_energy - e * e * initial_energy).abs());
        for (i, ri) in relative.iter().enumerate() {
            for (j, rj) in relative.iter().enumerate() {
                let pair =
                    if let WeakFieldObservable::PhaseSpaceCharacteristic { l, .. } = observable {
                        let plus = ri
                            .iter()
                            .zip(rj)
                            .map(|(a, b)| (a + b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let minus = ri
                            .iter()
                            .zip(rj)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let scale = e * dot(l, l).sqrt();
                        let p = sphere_characteristic(d, scale * plus)?;
                        let m = sphere_characteristic(d, scale * minus)?;
                        let sum = phases[i] + phases[j];
                        let diff = phases[i] - phases[j];
                        [
                            0.5 * (sum.cos() * p + diff.cos() * m) - means[i][0] * means[j][0],
                            0.5 * (sum.sin() * p - diff.sin() * m) - means[i][0] * means[j][1],
                            0.5 * (sum.sin() * p + diff.sin() * m) - means[i][1] * means[j][0],
                            0.5 * (diff.cos() * m - sum.cos() * p) - means[i][1] * means[j][1],
                        ]
                    } else {
                        let v = collision_polynomial_covariance(observable, &u, ri, rj, e);
                        let (ci, si) = (phases[i].cos(), phases[i].sin());
                        let (cj, sj) = (phases[j].cos(), phases[j].sin());
                        [v * ci * cj, v * ci * sj, v * si * cj, v * si * sj]
                    };
                for a in 0..4 {
                    covariance[a] += pair[a] / (n as f64 * n as f64);
                }
            }
        }
    }
    for i in 0..n {
        if !seen[i] {
            for a in 0..d {
                require(
                    same_recorded_scalar(w[i * d + a], literal_v[i * d + a]),
                    "collision changed an unconnected row",
                )?;
            }
        }
    }
    let scale = covariance.iter().map(|x| x.abs()).fold(1_f64, f64::max);
    require(
        covariance[0] >= -tolerance * scale && covariance[3] >= -tolerance * scale,
        "invalid component field covariance",
    )?;
    covariance[0] = covariance[0].max(0.);
    covariance[3] = covariance[3].max(0.);
    let cross = 0.5 * (covariance[1] + covariance[2]);
    covariance[1] = cross;
    covariance[2] = cross;
    require(
        predicted
            .iter()
            .chain(&baseline)
            .chain(&observed)
            .chain(&covariance)
            .all(|x| x.is_finite()),
        "collision field moments nonfinite",
    )?;
    Ok(CollisionFieldBalance {
        step:step.report.step,slots:n,component_slots:components,
        conditioning:"realized accepted current-donor graph and post-jitter positions; one independent shared Haar O(d) rotation per component; prior acceptance gates are not averaged in this stratum".into(),
        conditional_increment:sub(predicted,baseline),realized_increment:sub(observed,baseline),
        martingale_increment:sub(observed,predicted),martingale_covariance:covariance,
        maximum_momentum_conservation_residual:momentum_error,
        maximum_relative_energy_residual:energy_error,
    })
}
