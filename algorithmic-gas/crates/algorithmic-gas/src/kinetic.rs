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
    /// Viscous coupling over the tessellation graph of the geometry stage.
    pub graph_viscosity: Option<GraphViscosityConfig>,
    /// Boris rotation of the B steps by the curl of the graph viscous force.
    pub curl: Option<CurlRotationConfig>,
}
/// F_i = coefficient * sum_j w_ij (v_j - v_i) over tessellation neighbors, with
/// the named edge weights of the geometry pipeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphViscosityConfig {
    pub coefficient: f64,
    pub weights: String,
}
/// Each B step becomes quarter kick, Cayley rotation by
/// A = beta_curl * (dt / 4) * curl(F_viscous), quarter kick.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurlRotationConfig {
    pub beta_curl: f64,
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
        if let Some(v) = &self.graph_viscosity {
            require(
                v.coefficient.is_finite() && v.coefficient >= 0. && !v.weights.is_empty(),
                "invalid graph viscosity coefficient/weights",
            )?;
            require(
                self.viscosity.is_none(),
                "dense and graph viscosity are mutually exclusive",
            )?;
        }
        if let Some(c) = &self.curl {
            require(
                c.beta_curl.is_finite() && c.beta_curl >= 0.,
                "invalid curl rotation strength",
            )?;
            require(
                self.graph_viscosity.is_some(),
                "curl rotation requires graph viscosity",
            )?;
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
#[serde(default, deny_unknown_fields)]
pub struct KineticOperator {
    pub integrator: KineticKind,
    pub noise: Noise,
    /// Independent position Brownian diffusion after BAOAB, in length/sqrt(time).
    pub position_diffusion: f64,
    /// Final radial map v -> V v/(V+|v|), applied once per full update.
    pub velocity_cap: Option<f64>,
    pub boundary_schedule: KineticBoundarySchedule,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KineticBoundarySchedule {
    #[default]
    Substeps,
    EndOfStep,
}
impl Default for KineticOperator {
    fn default() -> Self {
        Self {
            integrator: KineticKind::DirectJump {
                field: "positions".into(),
                amplitude: 0.05,
            },
            noise: Noise::default(),
            position_diffusion: 0.,
            velocity_cap: None,
            boundary_schedule: KineticBoundarySchedule::Substeps,
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
        p.observations.provenance.population_version = p.version;
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
    p.observations.provenance.population_version = p.version;
    Ok(())
}
pub struct KineticContext<'a, T: Real> {
    pub gradient: Option<&'a dyn GradientProvider<T>>,
    pub domain: &'a dyn DomainAdapter<T>,
    pub boundary: &'a BoundaryPolicy,
    pub include_truncated: bool,
    pub seed: u64,
    pub step: u64,
    /// Neighbor graph and edge weights of the geometry stage, if any.
    pub graph: Option<&'a crate::tessellation::GraphSnapshot<T>>,
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
        // Reused per row: both loops below skip the same entries.
        let mut weights = vec![T::ZERO; n];
        for i in 0..n {
            if !eligible[i] {
                continue;
            }
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

/// One B step over the tessellation graph: quarter kick, Cayley rotation by
/// the curl of the viscous force, quarter kick with the viscous force
/// re-evaluated at the rotated velocities. `duration` is the length of the B
/// step (dt / 2). Returns `None` when graph viscosity is not configured.
#[allow(clippy::too_many_arguments)]
fn graph_kick<T: Real>(
    p: &Population<T>,
    positions: &str,
    velocities: &str,
    gradient: &TensorBatch<T>,
    k: &KineticContext<'_, T>,
    stage: &str,
    duration: T,
    cx: &mut ExecutionContext,
) -> Result<Option<TensorBatch<T>>> {
    use crate::tessellation::{
        Parallelism,
        forces::{GraphField, boris_rotate, curl, viscous_force},
    };
    let Some(qft) = k.frozen_fitness.as_ref().map(|f| &f.config.qft) else {
        return Ok(None);
    };
    let Some(config) = &qft.graph_viscosity else {
        return Ok(None);
    };
    let snapshot = k.graph.ok_or_else(|| {
        GasError::Capability("graph viscosity needs the geometry stage's neighbor graph".into())
    })?;
    let weights = snapshot.weights.get(&config.weights).ok_or_else(|| {
        GasError::Configuration(format!(
            "graph viscosity edge weights {:?} are not produced by the geometry stage",
            config.weights
        ))
    })?;
    let eligible = p.eligible(k.include_truncated);
    let x = p.observations.field(positions)?;
    let v = p.observations.field(velocities)?;
    let (n, d) = (p.len(), v.width());
    require(
        snapshot.graph.nodes() == n
            && gradient.rows() == n
            && gradient.width() == d
            && x.width() == d,
        "graph force shapes",
    )?;
    crate::memory::enforce(
        crate::memory::checked_mul(
            crate::memory::checked_mul(n, d * d + 8 * d + 2)?,
            std::mem::size_of::<T>(),
        )?,
        cx.max_memory_bytes,
    )?;
    let wrap: Vec<(usize, T)> = snapshot
        .wrap
        .iter()
        .map(|&(axis, length)| (axis, T::from_f64(length)))
        .collect();
    let field = GraphField {
        graph: &snapshot.graph,
        weights,
        positions: x.values(),
        dimension: d,
        eligible: &eligible,
        wrap: &wrap,
    };
    let par = Parallelism::Auto;
    let nu = T::from_f64(config.coefficient);
    let quarter = duration * T::from_f64(0.5);
    let viscous = viscous_force(&field, v.values(), nu, par);
    let kicked: Vec<T> = (0..n * d)
        .map(|a| v.values()[a] + quarter * (viscous[a] - gradient.values()[a]))
        .collect();
    let beta = qft.curl.as_ref().map_or(0., |c| c.beta_curl);
    let (rotated, curl_field, angle) = if beta > 0. {
        let c = curl(&field, &viscous, par)?;
        let scale = T::from_f64(0.5 * beta) * duration;
        let (rotated, angle) = boris_rotate(&kicked, &c, d, scale, &eligible, par)?;
        (rotated, c, angle)
    } else {
        (kicked, vec![T::ZERO; n * d * d], vec![T::ZERO; n])
    };
    let viscous_rotated = viscous_force(&field, &rotated, nu, par);
    let result: Vec<T> = (0..n * d)
        .map(|a| rotated[a] + quarter * (viscous_rotated[a] - gradient.values()[a]))
        .collect();
    if cx.recorded_fields.is_some() {
        let total: Vec<T> = (0..n * d)
            .map(|a| viscous[a] - gradient.values()[a])
            .collect();
        cx.record_field_with_coverage(stage, "force_input_velocity", p.version, v, &eligible);
        cx.record_field_with_coverage(
            stage,
            "viscous_force",
            p.version,
            &TensorBatch::vectors(n, d, viscous)?,
            &eligible,
        );
        cx.record_field_with_coverage(
            stage,
            "total_force",
            p.version,
            &TensorBatch::vectors(n, d, total)?,
            &eligible,
        );
        cx.record_field_with_coverage(
            stage,
            "curl_field",
            p.version,
            &TensorBatch::new(n, vec![d, d], curl_field)?,
            &eligible,
        );
        cx.record_field_with_coverage(
            stage,
            "boris_rotation_angle",
            p.version,
            &TensorBatch::scalars(angle)?,
            &eligible,
        );
    }
    Ok(Some(TensorBatch::vectors(n, d, result)?))
}

impl KineticOperator {
    pub fn validate<T: Real>(&self, p: &Population<T>, has_gradient: bool) -> Result<()> {
        require(
            self.position_diffusion.is_finite() && self.position_diffusion >= 0.,
            "position diffusion must be finite and nonnegative",
        )?;
        require(
            self.velocity_cap.is_none_or(|v| v.is_finite() && v > 0.),
            "velocity cap must be finite and positive",
        )?;
        require(
            matches!(self.integrator, KineticKind::Baoab { .. })
                || (self.position_diffusion == 0.
                    && self.velocity_cap.is_none()
                    && self.boundary_schedule == KineticBoundarySchedule::Substeps),
            "position diffusion, velocity cap and terminal boundary schedule require BAOAB",
        )?;
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
        // Sampling hooks receive ObservationBatch rather than Population. Its
        // provenance must identify the actual post-clone/substep state.
        p.observations.provenance.population_version = p.version;
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
                let alive = p.eligible(k.include_truncated);
                let kicked = graph_kick(p, positions, velocities, &grad, &k, "B1", half, cx)?;
                let force = match &kicked {
                    Some(_) => None,
                    None => recorded_force(p, positions, velocities, &grad, &k, "B1", cx)?,
                };
                if let Some(next) = kicked {
                    update(p, velocities, &next, T::ZERO, T::ONE, &alive, cx, k.domain).await?;
                } else if let Some(force) = force {
                    update(p, velocities, &force, T::ONE, half, &alive, cx, k.domain).await?;
                } else {
                    update(p, velocities, &grad, T::ONE, -half, &alive, cx, k.domain).await?;
                }
                cx.record_boundary_input("B1_before_boundary", p);
                if self.boundary_schedule == KineticBoundarySchedule::Substeps {
                    k.boundary(p)?;
                }
                cx.trace_population("B1", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                let velocity = p.observations.field(velocities)?.clone();
                let alive = p.eligible(k.include_truncated);
                update(p, positions, &velocity, T::ONE, half, &alive, cx, k.domain).await?;
                cx.record_boundary_input("A1_before_boundary", p);
                if self.boundary_schedule == KineticBoundarySchedule::Substeps {
                    k.boundary(p)?;
                }
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
                if self.boundary_schedule == KineticBoundarySchedule::Substeps {
                    k.boundary(p)?;
                }
                cx.trace_population("O", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                let velocity = p.observations.field(velocities)?.clone();
                let alive = p.eligible(k.include_truncated);
                update(p, positions, &velocity, T::ONE, half, &alive, cx, k.domain).await?;
                cx.record_boundary_input("A2_before_boundary", p);
                if self.boundary_schedule == KineticBoundarySchedule::Substeps {
                    k.boundary(p)?;
                }
                cx.trace_population("A2", p);
                if !k.has_eligible(p) {
                    return Ok(());
                }
                // Deliberately recompute at the post-A positions; no stale cache.
                let grad = gradient.gradient(p, cx).await?;
                cx.record_boundary_input("B2_input", p);
                cx.record_field("B2", "potential_gradient", p.version, &grad);
                let alive = p.eligible(k.include_truncated);
                let kicked = graph_kick(p, positions, velocities, &grad, &k, "B2", half, cx)?;
                let force = match &kicked {
                    Some(_) => None,
                    None => recorded_force(p, positions, velocities, &grad, &k, "B2", cx)?,
                };
                if let Some(next) = kicked {
                    update(p, velocities, &next, T::ZERO, T::ONE, &alive, cx, k.domain).await?;
                } else if let Some(force) = force {
                    update(p, velocities, &force, T::ONE, half, &alive, cx, k.domain).await?;
                } else {
                    update(p, velocities, &grad, T::ONE, -half, &alive, cx, k.domain).await?;
                }
                cx.record_boundary_input("B2_before_boundary", p);
                if self.boundary_schedule == KineticBoundarySchedule::Substeps {
                    k.boundary(p)?;
                }
                cx.trace_population("B2", p);
                if self.position_diffusion > 0. && k.has_eligible(p) {
                    let alive = p.eligible(k.include_truncated);
                    let eta = Noise::default()
                        .sample_masked(
                            &p.observations,
                            NoiseRequest {
                                rows: p.len(),
                                dimension: d,
                                seed: k.seed,
                                step: k.step,
                                stream: Stream::Kinetic,
                                substep: 5,
                            },
                            &alive,
                            cx,
                        )
                        .await?;
                    cx.record_field_with_coverage(
                        "position_diffusion",
                        "executed_noise",
                        p.version,
                        &eta,
                        &alive,
                    );
                    update(
                        p,
                        positions,
                        &eta,
                        T::ONE,
                        h.sqrt() * T::from_f64(self.position_diffusion),
                        &alive,
                        cx,
                        k.domain,
                    )
                    .await?;
                    cx.record_boundary_input("position_diffusion_before_boundary", p);
                    if self.boundary_schedule == KineticBoundarySchedule::Substeps {
                        k.boundary(p)?;
                    }
                    cx.trace_population("position_diffusion", p);
                }
                if let Some(radius) = self.velocity_cap
                    && k.has_eligible(p)
                {
                    let alive = p.eligible(k.include_truncated);
                    let mut capped = p.observations.field(velocities)?.clone();
                    let radius = T::from_f64(radius);
                    for row in capped.values_mut().chunks_mut(d) {
                        // Scaled norm avoids overflow in the sum of squares.
                        let scale = row.iter().fold(T::ZERO, |m, x| m.max(x.abs()));
                        if scale > T::ZERO {
                            let norm_scaled = row
                                .iter()
                                .fold(T::ZERO, |s, x| s + (*x / scale) * (*x / scale))
                                .sqrt();
                            let factor = if scale <= radius {
                                T::ONE / (T::ONE + (scale / radius) * norm_scaled)
                            } else {
                                (radius / scale) / (radius / scale + norm_scaled)
                            };
                            for x in row {
                                *x = *x * factor;
                            }
                        }
                    }
                    update(
                        p,
                        velocities,
                        &capped,
                        T::ZERO,
                        T::ONE,
                        &alive,
                        cx,
                        k.domain,
                    )
                    .await?;
                    cx.record_boundary_input("velocity_cap_before_boundary", p);
                    if self.boundary_schedule == KineticBoundarySchedule::Substeps {
                        k.boundary(p)?;
                    }
                    cx.trace_population("velocity_cap", p);
                }
                if self.boundary_schedule == KineticBoundarySchedule::EndOfStep {
                    cx.record_boundary_input("terminal_before_boundary", p);
                    k.boundary(p)?;
                    cx.trace_population("terminal", p);
                }
            }
            KineticKind::Environment => {
                let alive = p.eligible(k.include_truncated);
                k.domain.transition(p, &alive, k.seed, k.step)?;
                p.version = p
                    .version
                    .checked_add(1)
                    .ok_or_else(|| GasError::Numerical("population version overflow".into()))?;
                k.domain.refresh_observations(p)?;
                p.observations.provenance.population_version = p.version;
                cx.record_boundary_input("environment_before_boundary", p);
                k.boundary(p)?;
            }
        }
        Ok(())
    }
}
