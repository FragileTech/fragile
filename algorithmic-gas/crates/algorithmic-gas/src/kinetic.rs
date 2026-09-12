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
                cx.record_field("B1", "potential_gradient", p.version, &grad);
                let alive = p.eligible(k.include_truncated);
                update(p, velocities, &grad, T::ONE, -half, &alive, cx, k.domain).await?;
                k.boundary(p)?;
                cx.trace_population("B1", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                let velocity = p.observations.field(velocities)?.clone();
                let alive = p.eligible(k.include_truncated);
                update(p, positions, &velocity, T::ONE, half, &alive, cx, k.domain).await?;
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
                let c = (-gamma * h).exp();
                // Stable near gamma=0. B is the diffusion factor in dv=B dW.
                let scale = if gamma == T::ZERO {
                    h.sqrt()
                } else {
                    (-(-T::from_f64(2.) * gamma * h).exp_m1() / (T::from_f64(2.) * gamma)).sqrt()
                };
                let alive = p.eligible(k.include_truncated);
                update(p, velocities, &eta, c, scale, &alive, cx, k.domain).await?;
                k.boundary(p)?;
                cx.trace_population("O", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                let velocity = p.observations.field(velocities)?.clone();
                let alive = p.eligible(k.include_truncated);
                update(p, positions, &velocity, T::ONE, half, &alive, cx, k.domain).await?;
                k.boundary(p)?;
                cx.trace_population("A2", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                // Deliberately recompute at the post-A positions; no stale cache.
                let grad = gradient.gradient(p, cx).await?;
                cx.record_field("B2", "potential_gradient", p.version, &grad);
                let alive = p.eligible(k.include_truncated);
                update(p, velocities, &grad, T::ONE, -half, &alive, cx, k.domain).await?;
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
                k.boundary(p)?;
            }
        }
        Ok(())
    }
}
