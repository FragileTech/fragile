use crate::{
    ComputeBackend, ExecutionContext, GasError, Population, Real, Result, TensorBatch,
    boundary::BoundaryPolicy,
    compute::{Binary, Expression},
    domain::{DomainAdapter, GradientProvider},
    error::require,
    noise::{Noise, NoiseRequest, NoiseSource},
    random::Stream,
};
use serde::{Deserialize, Serialize};

/// Optional execution features used by recorded QFT experiments.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QftExecutionConfig {
    pub viscosity: Option<ViscousForceConfig>,
    pub innovation_shifts: Vec<crate::noise::InnovationShift>,
}
/// Gaussian spatial graph: w_ij = exp(-|x_i-x_j|²/(2 bandwidth²)).
/// Symmetric normalization divides by the eligible population, including self;
/// row normalization divides by off-diagonal row mass and need not conserve momentum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViscousForceConfig {
    pub coefficient: f64,
    pub bandwidth: f64,
    #[serde(default)]
    pub row_normalized: bool,
}
impl QftExecutionConfig {
    pub fn validate(&self, rows: usize, dimension: usize) -> Result<()> {
        if let Some(v) = &self.viscosity {
            require(
                v.coefficient.is_finite()
                    && v.coefficient >= 0.
                    && v.bandwidth.is_finite()
                    && v.bandwidth > 0.,
                "invalid viscous coefficient/bandwidth",
            )?;
        }
        for s in &self.innovation_shifts {
            s.validate(rows, dimension)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KineticKind {
    DirectJump {
        field: String,
        amplitude: f64,
    },
    Brownian {
        field: String,
        amplitude: f64,
        dt: f64,
    },
    Baoab {
        positions: String,
        velocities: String,
        dt: f64,
        friction: f64,
    },
    Environment,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KineticOperator {
    pub integrator: KineticKind,
    pub noise: Noise,
}
impl Default for KineticOperator {
    fn default() -> Self {
        Self {
            integrator: KineticKind::DirectJump {
                field: "positions".into(),
                amplitude: 0.05,
            },
            noise: Noise::default(),
        }
    }
}

pub(crate) fn check_boundary<T: Real>(
    p: &mut Population<T>,
    boundary: &BoundaryPolicy,
    domain: &dyn DomainAdapter<T>,
) -> Result<()> {
    let changed = boundary.apply(p)?;
    if !changed.is_empty() {
        p.version = p
            .version
            .checked_add(1)
            .ok_or_else(|| GasError::Numerical("population version overflow".into()))?;
        domain.reconcile(p, &changed)?;
        domain.refresh_observations(p)?;
    }
    Ok(())
}

/// Batched a*x + b*y using the selected compute adapter. Only eligible rows
/// commit, so a dead row cannot continue moving during later BAOAB substeps.
#[allow(clippy::too_many_arguments)]
async fn update<T: Real>(
    p: &mut Population<T>,
    target: &str,
    source: &TensorBatch<T>,
    a: T,
    b: T,
    eligible: &[bool],
    cx: &mut ExecutionContext,
    domain: &dyn DomainAdapter<T>,
) -> Result<()> {
    let old = p.observations.field(target)?;
    require(
        old.rows() == source.rows() && old.width() == source.width(),
        "kinetic batch shape",
    )?;
    let mut e = Expression::default();
    let x = e.input(0);
    let y = e.input(1);
    let an = e.scalar(a.to_f64());
    let bn = e.scalar(b.to_f64());
    let ax = e.binary(Binary::Multiply, x, an);
    let by = e.binary(Binary::Multiply, y, bn);
    e.binary(Binary::Add, ax, by);
    let result = cx.evaluate(&e, &[old.clone(), source.clone()]).await?;
    let field = p.observations.field_mut(target)?;
    for (i, &alive) in eligible.iter().enumerate() {
        if alive {
            field.replace_row(i, result.row(i)?)?;
        }
    }
    p.version = p
        .version
        .checked_add(1)
        .ok_or_else(|| GasError::Numerical("population version overflow".into()))?;
    domain.reconcile(p, &[target.into()])?;
    domain.refresh_observations(p)?;
    Ok(())
}
pub struct KineticContext<'a, T: Real> {
    pub gradient: Option<&'a dyn GradientProvider<T>>,
    pub domain: &'a dyn DomainAdapter<T>,
    pub boundary: &'a BoundaryPolicy,
    pub include_truncated: bool,
    pub seed: u64,
    pub step: u64,
    pub operators: Option<&'a dyn crate::operators::GasOperators<T>>,
    pub frozen_fitness: Option<crate::operators::FrozenFitnessContext<'a, T>>,
}
impl<T: Real> KineticContext<'_, T> {
    fn has_eligible(&self, p: &Population<T>) -> bool {
        p.validity
            .iter()
            .any(|s| s.eligible(self.include_truncated))
    }
    fn boundary(&self, p: &mut Population<T>) -> Result<()> {
        if let Some(operators) = self.operators {
            operators.boundary(self.boundary, p, self.domain)
        } else {
            check_boundary(p, self.boundary, self.domain)
        }
    }
}
/// Evaluate the actual BAOAB force and record its input stage. This is an
/// acceleration convention (unit mass), matching the potential-gradient kick.
fn recorded_force<T: Real>(
    p: &Population<T>,
    positions: &str,
    velocities: &str,
    gradient: &TensorBatch<T>,
    k: &KineticContext<'_, T>,
    stage: &str,
    cx: &mut ExecutionContext,
) -> Result<Option<TensorBatch<T>>> {
    let eligible = p.eligible(k.include_truncated);
    let v = p.observations.field(velocities)?;
    let x = p.observations.field(positions)?;
    require(
        gradient.rows() == p.len() && gradient.width() == v.width(),
        "force gradient shape",
    )?;
    let n = p.len();
    let d = v.width();
    let config = k
        .frozen_fitness
        .as_ref()
        .and_then(|f| f.config.qft.viscosity.as_ref());
    if config.is_none_or(|c| c.coefficient == 0.) && cx.recorded_fields.is_none() {
        return Ok(None);
    }
    let force_bytes = crate::memory::checked_mul(
        crate::memory::checked_mul(n, d)?,
        std::mem::size_of::<T>() * 4,
    )?;
    let edge_bytes =
        if config.is_some_and(|c| c.coefficient > 0.) && cx.recorded_influences.is_some() {
            crate::memory::checked_mul(crate::memory::checked_mul(n, n.saturating_sub(1))?, 256)?
        } else {
            0
        };
    crate::memory::enforce(
        crate::memory::checked_add(force_bytes, edge_bytes)?,
        cx.max_memory_bytes,
    )?;
    let mut viscous = vec![T::ZERO; n * d];
    if let Some(config) = config.filter(|c| c.coefficient > 0.) {
        let count = eligible.iter().filter(|&&a| a).count();
        let bandwidth = T::from_f64(config.bandwidth);
        let denominator = T::from_f64(2.) * bandwidth * bandwidth;
        let coefficient = T::from_f64(config.coefficient);
        for i in 0..n {
            if !eligible[i] {
                continue;
            }
            let mut weights = vec![T::ZERO; n];
            let mut mass = T::ZERO;
            for j in 0..n {
                if i == j || !eligible[j] {
                    continue;
                }
                let mut distance = T::ZERO;
                for a in 0..d {
                    let delta = x.values()[i * d + a] - x.values()[j * d + a];
                    distance = distance + delta * delta;
                }
                weights[j] = (-distance / denominator).exp();
                mass = mass + weights[j];
            }
            let normalizer = if config.row_normalized {
                mass
            } else {
                T::from_f64(count as f64)
            };
            if normalizer == T::ZERO {
                continue;
            }
            for j in 0..n {
                if i == j || !eligible[j] {
                    continue;
                }
                let weight = coefficient * weights[j] / normalizer;
                for a in 0..d {
                    viscous[i * d + a] = viscous[i * d + a]
                        + weight * (v.values()[j * d + a] - v.values()[i * d + a]);
                }
                cx.record_influence(
                    stage,
                    "viscous_force",
                    i as u32,
                    crate::donor::SourceRef {
                        frame: k.step,
                        slot: j as u32,
                        generation: p.generations[j],
                        version: p.version,
                    },
                    weight.to_f64(),
                );
            }
        }
    }
    let potential: Vec<T> = gradient.values().iter().map(|&g| -g).collect();
    let total: Vec<T> = potential
        .iter()
        .zip(&viscous)
        .map(|(&a, &b)| a + b)
        .collect();
    let viscous = TensorBatch::vectors(n, d, viscous)?;
    let total = TensorBatch::vectors(n, d, total)?;
    cx.record_field_with_coverage(stage, "force_input_velocity", p.version, v, &eligible);
    cx.record_field_with_coverage(
        stage,
        "potential_force",
        p.version,
        &TensorBatch::vectors(n, d, potential)?,
        &eligible,
    );
    cx.record_field_with_coverage(stage, "viscous_force", p.version, &viscous, &eligible);
    cx.record_field_with_coverage(stage, "total_force", p.version, &total, &eligible);
    // Preserve the precise arithmetic path of every existing zero-viscosity run.
    Ok(config.filter(|c| c.coefficient > 0.).map(|_| total))
}

impl KineticOperator {
    pub fn validate<T: Real>(&self, p: &Population<T>, has_gradient: bool) -> Result<()> {
        match &self.integrator {
            KineticKind::DirectJump { field, amplitude }
            | KineticKind::Brownian {
                field, amplitude, ..
            } => {
                require(
                    amplitude.is_finite() && *amplitude >= 0.,
                    "jump amplitude must be finite and nonnegative",
                )?;
                let x = p.observations.field(field)?;
                self.noise.validate(&p.observations, p.len(), x.width())?;
                if let KineticKind::Brownian { dt, .. } = &self.integrator {
                    require(dt.is_finite() && *dt > 0., "Brownian dt must be positive")?;
                }
            }
            KineticKind::Baoab {
                positions,
                velocities,
                dt,
                friction,
            } => {
                require(
                    has_gradient,
                    "BAOAB requires an explicit potential gradient provider",
                )?;
                require(
                    dt.is_finite() && *dt > 0. && friction.is_finite() && *friction >= 0.,
                    "invalid BAOAB dt/friction",
                )?;
                let x = p.observations.field(positions)?;
                let v = p.observations.field(velocities)?;
                require(
                    x.item_shape().len() == 1 && x.item_shape() == v.item_shape(),
                    "BAOAB requires matching position/velocity vectors",
                )?;
                self.noise.validate(&p.observations, p.len(), x.width())?;
            }
            KineticKind::Environment => {}
        }
        Ok(())
    }
    pub async fn advance<T: Real>(
        &self,
        p: &mut Population<T>,
        k: KineticContext<'_, T>,
        cx: &mut ExecutionContext,
    ) -> Result<()> {
        if let Some(operators) = k.operators {
            let noise = crate::operators::HookNoise {
                operators,
                config: &self.noise,
                frozen: k.frozen_fitness,
            };
            self.advance_with_noise(p, k, &noise, cx).await
        } else {
            self.advance_with_noise(p, k, &self.noise, cx).await
        }
    }
    /// Custom noise implementations can be injected without changing the
    /// integrator. They must obey the no-temporal-scaling contract.
    pub async fn advance_with_noise<T: Real, N: NoiseSource<T>>(
        &self,
        p: &mut Population<T>,
        k: KineticContext<'_, T>,
        noise: &N,
        cx: &mut ExecutionContext,
    ) -> Result<()> {
        if !k.has_eligible(p) {
            return Ok(());
        }
        self.validate(p, k.gradient.is_some())?;
        match &self.integrator {
            KineticKind::DirectJump { field, amplitude }
            | KineticKind::Brownian {
                field, amplitude, ..
            } => {
                let d = p.observations.field(field)?.width();
                let eta = noise
                    .sample(
                        &p.observations,
                        NoiseRequest {
                            rows: p.len(),
                            dimension: d,
                            seed: k.seed,
                            step: k.step,
                            stream: Stream::Kinetic,
                            substep: 0,
                        },
                        cx,
                    )
                    .await?;
                let temporal = match self.integrator {
                    KineticKind::Brownian { dt, .. } => T::from_f64(dt).sqrt(),
                    _ => T::ONE,
                };
                let alive = p.eligible(k.include_truncated);
                update(
                    p,
                    field,
                    &eta,
                    T::ONE,
                    T::from_f64(*amplitude) * temporal,
                    &alive,
                    cx,
                    k.domain,
                )
                .await?;
                cx.record_boundary_input("jump_before_boundary", p);
                k.boundary(p)?;
            }
            KineticKind::Baoab {
                positions,
                velocities,
                dt,
                friction,
            } => {
                let h = T::from_f64(*dt);
                let gamma = T::from_f64(*friction);
                let half = h * T::from_f64(0.5);
                let gradient = k
                    .gradient
                    .ok_or_else(|| GasError::Capability("missing gradient provider".into()))?;
                let grad = gradient.gradient(p, cx).await?;
                cx.record_boundary_input("B1_input", p);
                cx.record_field("B1", "potential_gradient", p.version, &grad);
                let force = recorded_force(p, positions, velocities, &grad, &k, "B1", cx)?;
                let alive = p.eligible(k.include_truncated);
                if let Some(force) = force {
                    update(p, velocities, &force, T::ONE, half, &alive, cx, k.domain).await?;
                } else {
                    update(p, velocities, &grad, T::ONE, -half, &alive, cx, k.domain).await?;
                }
                cx.record_boundary_input("B1_before_boundary", p);
                k.boundary(p)?;
                cx.trace_population("B1", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                let velocity = p.observations.field(velocities)?.clone();
                let alive = p.eligible(k.include_truncated);
                update(p, positions, &velocity, T::ONE, half, &alive, cx, k.domain).await?;
                cx.record_boundary_input("A1_before_boundary", p);
                k.boundary(p)?;
                cx.trace_population("A1", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                let d = p.observations.field(velocities)?.width();
                let noise_eligible = p.eligible(k.include_truncated);
                let eta = noise
                    .sample_masked(
                        &p.observations,
                        NoiseRequest {
                            rows: p.len(),
                            dimension: d,
                            seed: k.seed,
                            step: k.step,
                            stream: Stream::Kinetic,
                            substep: 2,
                        },
                        &noise_eligible,
                        cx,
                    )
                    .await?;
                cx.record_field_with_coverage(
                    "O",
                    "executed_noise",
                    p.version,
                    &eta,
                    &noise_eligible,
                );
                let c = (-gamma * h).exp();
                // Stable near gamma=0. B is the diffusion factor in dv=B dW.
                let scale = if gamma == T::ZERO {
                    h.sqrt()
                } else {
                    (-(-T::from_f64(2.) * gamma * h).exp_m1() / (T::from_f64(2.) * gamma)).sqrt()
                };
                let alive = p.eligible(k.include_truncated);
                update(p, velocities, &eta, c, scale, &alive, cx, k.domain).await?;
                cx.record_boundary_input("O_before_boundary", p);
                k.boundary(p)?;
                cx.trace_population("O", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                let velocity = p.observations.field(velocities)?.clone();
                let alive = p.eligible(k.include_truncated);
                update(p, positions, &velocity, T::ONE, half, &alive, cx, k.domain).await?;
                cx.record_boundary_input("A2_before_boundary", p);
                k.boundary(p)?;
                cx.trace_population("A2", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                // Deliberately recompute at the post-A positions; no stale cache.
                let grad = gradient.gradient(p, cx).await?;
                cx.record_boundary_input("B2_input", p);
                cx.record_field("B2", "potential_gradient", p.version, &grad);
                let force = recorded_force(p, positions, velocities, &grad, &k, "B2", cx)?;
                let alive = p.eligible(k.include_truncated);
                if let Some(force) = force {
                    update(p, velocities, &force, T::ONE, half, &alive, cx, k.domain).await?;
                } else {
                    update(p, velocities, &grad, T::ONE, -half, &alive, cx, k.domain).await?;
                }
                cx.record_boundary_input("B2_before_boundary", p);
                k.boundary(p)?;
                cx.trace_population("B2", p);
            }
            KineticKind::Environment => {
                let alive = p.eligible(k.include_truncated);
                k.domain.transition(p, &alive, k.seed, k.step)?;
                p.version = p
                    .version
                    .checked_add(1)
                    .ok_or_else(|| GasError::Numerical("population version overflow".into()))?;
                k.domain.refresh_observations(p)?;
                cx.record_boundary_input("environment_before_boundary", p);
                k.boundary(p)?;
            }
        }
        Ok(())
    }
}
