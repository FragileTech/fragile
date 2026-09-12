//! General-dimensional conditional metric provider. Packed jets and eigensolves
//! are host kernels; application of diffusion factors uses the selected backend.
use crate::Benchmark;
use algorithmic_gas::{
    ExecutionContext, GasConfig, GasError, ObservationBatch, Population, Real, Result, TensorBatch,
    domain::OperatorFuture,
    donor::CompanionReducer,
    fitness::{PositiveMap, Standardizer},
    geometry::Distance,
    kinetic::KineticKind,
    noise::{FactorValues, Noise, NoiseGeometry, NoiseRequest, NoiseSource},
    operators::{FrozenFitnessContext, GasOperators},
    partv_geometry::MetricPolicy,
    physics::{
        fitness::{
            ConditionalFitnessCache, Objective, companion_distance, local_log_weights,
            pipeline_from_measurement_jets,
        },
        geometry::{FitnessJet, fitness_curvature, metric_spectrum},
        jet::JetSpace,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhysicsMetricConfig {
    pub epsilon: f64,
    pub temperature: f64,
    pub policy: MetricPolicy,
    pub curvature: bool,
    /// Absolute fitness-Hessian eigenvalue threshold for curvature availability.
    pub clipping_threshold: f64,
}
impl Default for PhysicsMetricConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            temperature: 1.,
            policy: MetricPolicy::Clipped,
            curvature: true,
            clipping_threshold: 1e-8,
        }
    }
}
#[derive(Clone)]
pub struct PhysicsMetricOperators {
    pub config: PhysicsMetricConfig,
    pub benchmark: Benchmark,
    pub reward_shift: Vec<f64>,
}
impl Benchmark {
    pub fn physics_objective(self, dimension: usize) -> Objective {
        match self {
            Self::Sphere => Objective::Sphere,
            Self::Quadratic => Objective::Quadratic {
                matrix: (0..dimension * dimension)
                    .map(|i| {
                        if i / dimension == i % dimension {
                            1.
                        } else {
                            0.
                        }
                    })
                    .collect(),
            },
            Self::Rastrigin => Objective::Rastrigin,
            Self::Rosenbrock => Objective::Rosenbrock,
            Self::StyblinskiTang => Objective::StyblinskiTang,
        }
    }
}
impl<T: Real> GasOperators<T> for PhysicsMetricOperators {
    fn id(&self) -> String {
        format!(
            "conditional-fitness-nd/v1/{:?}/{:?}/{:?}",
            self.benchmark, self.config, self.reward_shift
        )
    }
    fn validate(&self, c: &GasConfig, p: &Population<T>) -> Result<()> {
        let error = |s: &str| GasError::Capability(s.into());
        let d = p.observations.field("positions")?.width();
        JetSpace::new(d, if self.config.curvature { 4 } else { 2 })?;
        if !(self.config.epsilon.is_finite()
            && self.config.epsilon > 0.
            && self.config.temperature.is_finite()
            && self.config.temperature >= 0.
            && self.config.clipping_threshold.is_finite()
            && self.config.clipping_threshold >= 0.)
        {
            return Err(error("invalid conditional physics metric parameters"));
        }
        if !matches!(&c.kinetic.integrator,KineticKind::Baoab{positions,friction,..} if positions=="positions"&&*friction>=0.)
        {
            return Err(error(
                "conditional physics metric requires BAOAB with nonnegative friction",
            ));
        }
        if c.distance_donors.count != 1 || !matches!(c.reducer, CompanionReducer::Mean) {
            return Err(error(
                "conditional physics metric requires one immutable distance companion and mean reduction",
            ));
        }
        match &c.distance_donors.distance {
            Distance::Euclidean {
                field,
                scales,
                squared: false,
                periodic: None,
            } if field == "positions" && (scales.is_empty() || scales.iter().all(|x| *x == 1.)) => {
            }
            Distance::PhaseSpace {
                positions,
                position_scale: 1.,
                velocity_scale: 1.,
                periodic: None,
                ..
            } if positions == "positions" => {}
            _ => {
                return Err(error(
                    "conditional physics metric needs unscaled Euclidean/phase distance without periodic seams",
                ));
            }
        }
        for (standardizer, map) in [
            (&c.fitness.reward_standardizer, &c.fitness.reward_map),
            (&c.fitness.diversity_standardizer, &c.fitness.diversity_map),
        ] {
            if !matches!(
                standardizer,
                Standardizer::Global { .. } | Standardizer::Local { .. }
            ) || !matches!(map, PositiveMap::Logistic { .. })
            {
                return Err(error(
                    "conditional physics metric requires smooth global/local standardizers and logistic maps",
                ));
            }
            let space = JetSpace::new(d, 2)?;
            let query: Vec<_> = (0..d)
                .map(|a| space.variable(T::ZERO, a))
                .collect::<Result<_>>()?;
            local_log_weights(
                &query,
                &[vec![T::ZERO; d]],
                &[vec![T::ZERO; d]],
                &[true],
                0,
                standardizer,
            )?;
        }
        Ok(())
    }
    fn conditioned_noise<'a>(
        &'a self,
        noise: &'a Noise,
        observations: &'a ObservationBatch<T>,
        request: NoiseRequest,
        frozen: Option<&'a FrozenFitnessContext<'a, T>>,
        eligible: Option<&'a [bool]>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>> {
        Box::pin(async move {
            let f = frozen.ok_or_else(|| {
                GasError::Capability("missing conditional selection state".into())
            })?;
            let (friction, velocities) = match &f.config.kinetic.integrator {
                KineticKind::Baoab {
                    friction,
                    velocities,
                    ..
                } => (*friction, velocities),
                _ => {
                    return Err(GasError::Capability(
                        "conditional metric outside O stage".into(),
                    ));
                }
            };
            let d = request.dimension;
            let n = request.rows;
            let positions = f.population.observations.field("positions")?;
            let actual = observations.field("positions")?;
            let velocity_field = match &f.config.distance_donors.distance {
                Distance::PhaseSpace { velocities, .. } => velocities,
                _ => velocities,
            };
            let velocity = f.population.observations.field(velocity_field)?;
            let weight = match f.config.distance_donors.distance {
                Distance::PhaseSpace { lambda, .. } => T::from_f64(lambda),
                _ => T::ZERO,
            };
            let shift = |i: usize| T::from_f64(self.reward_shift.get(i).copied().unwrap_or(0.));
            let points: Vec<Vec<T>> = positions
                .values()
                .chunks_exact(d)
                .map(|x| x.iter().enumerate().map(|(i, &v)| v - shift(i)).collect())
                .collect();
            // Companion indices address immutable pool rows, not current population slots.
            // Repeated slot numbers in historical frames must retain their own coordinates.
            let donor_positions = f.pool.population.observations.field("positions")?;
            let donor_velocities = f.pool.population.observations.field(velocity_field)?;
            let mut companion_points = Vec::with_capacity(n);
            let mut dv = Vec::with_capacity(n);
            for (i, point) in points.iter().enumerate() {
                if f.companions.valid[i] {
                    let pool_index = f.companions.indices[i] as usize;
                    let source_point = donor_positions
                        .row(pool_index)?
                        .iter()
                        .enumerate()
                        .map(|(a, &v)| v - shift(a))
                        .collect::<Vec<_>>();
                    let source_velocity = donor_velocities.row(pool_index)?;
                    let difference = (0..d).fold(T::ZERO, |s, a| {
                        let delta = velocity.values()[i * d + a] - source_velocity[a];
                        s + weight * delta * delta
                    });
                    companion_points.push(source_point);
                    dv.push(difference);
                } else {
                    companion_points.push(point.clone());
                    dv.push(T::ZERO);
                }
            }
            let objective = self.benchmark.physics_objective(d);
            let floor = T::from_f64(f.config.fitness.distance_floor);
            let constants = JetSpace::new(d, 0)?;
            let mut rewards = Vec::with_capacity(n);
            let mut distances = Vec::with_capacity(n);
            for i in 0..n {
                if !f.alive[i] {
                    rewards.push(T::ZERO);
                    distances.push(floor);
                    continue;
                }
                let x: Vec<_> = points[i].iter().map(|&v| constants.constant(v)).collect();
                rewards.push(objective.evaluate(&x)?.value());
                distances.push(if f.companions.valid[i] {
                    companion_distance(&x, &companion_points[i], dv[i], floor)?.value()
                } else {
                    floor
                });
            }
            let cache = if matches!(
                f.config.fitness.reward_standardizer,
                Standardizer::Global { .. }
            ) && matches!(
                f.config.fitness.diversity_standardizer,
                Standardizer::Global { .. }
            ) {
                Some(ConditionalFitnessCache::new(
                    &rewards,
                    &distances,
                    f.alive,
                    &f.config.fitness,
                )?)
            } else {
                None
            };
            let record_curvature = self.config.curvature && cx.recording_enabled();
            let space = JetSpace::new(d, if record_curvature { 3 } else { 2 })?;
            let mut space4 = None;
            let available = eligible
                .map(|v| v.to_vec())
                .unwrap_or_else(|| vec![true; n]);
            let mut curvature_available = vec![false; n];
            let mut factors = vec![0.; n * d * d];
            let mut hessians = vec![T::ZERO; n * d * d];
            let mut metrics = hessians.clone();
            let mut roots = hessians.clone();
            let mut ricci = hessians.clone();
            let mut scalar = vec![T::ZERO; n];
            let mut targets = scalar.clone();
            let mut extension = scalar.clone();
            let mut condition = scalar.clone();
            let mut logdet = scalar.clone();
            let amplitude = T::from_f64((2. * friction * self.config.temperature).sqrt());
            for i in 0..n {
                if !available[i] {
                    continue;
                }
                let target = if f.alive[i] {
                    i
                } else {
                    let donor = f.clone_plan.choices[i].donors.first().ok_or_else(|| {
                        GasError::Capability("revival missing conditional donor field".into())
                    })?;
                    let source = f
                        .clone_plan
                        .sources
                        .get(donor.pool_index as usize)
                        .ok_or_else(|| {
                            GasError::Capability("revival source identity unavailable".into())
                        })?;
                    let slot = source.slot as usize;
                    if source.frame != f.pool.current_frame
                        || source.version != f.population.version
                        || f.population.generations.get(slot) != Some(&source.generation)
                        || !f.alive.get(slot).copied().unwrap_or(false)
                    {
                        return Err(GasError::Capability("revival conditional field requires its exact eligible current-frame source; a historical slot cannot substitute".into()));
                    }
                    slot
                };
                let evaluate = |space: &std::sync::Arc<JetSpace>| -> Result<_> {
                    let x: Vec<_> = (0..d)
                        .map(|a| space.variable(actual.values()[i * d + a] - shift(a), a))
                        .collect::<Result<_>>()?;
                    let reward = objective.evaluate(&x)?;
                    let distance = if f.companions.valid[target] {
                        companion_distance(&x, &companion_points[target], dv[target], floor)?
                    } else {
                        space.constant(floor)
                    };
                    if let Some(cache) = &cache {
                        cache.evaluate(target, &reward, &distance)
                    } else {
                        let mut rs: Vec<_> = rewards.iter().map(|&v| space.constant(v)).collect();
                        let mut ds: Vec<_> = distances.iter().map(|&v| space.constant(v)).collect();
                        rs[target] = reward;
                        ds[target] = distance;
                        let logs = |standardizer: &Standardizer| -> Result<_> {
                            let rows = if let Standardizer::Local {
                                distance: Distance::PhaseSpace { velocities, .. },
                                ..
                            } = standardizer
                            {
                                f.population
                                    .observations
                                    .field(velocities)?
                                    .values()
                                    .chunks_exact(d)
                                    .map(|v| v.to_vec())
                                    .collect::<Vec<_>>()
                            } else {
                                Vec::new()
                            };
                            local_log_weights(&x, &points, &rows, f.alive, target, standardizer)
                        };
                        let reward_logs = logs(&f.config.fitness.reward_standardizer)?;
                        let distance_logs = logs(&f.config.fitness.diversity_standardizer)?;
                        pipeline_from_measurement_jets(
                            &rs,
                            &ds,
                            f.alive,
                            target,
                            &f.config.fitness,
                            reward_logs.as_deref(),
                            distance_logs.as_deref(),
                        )
                    }
                };
                let jet = evaluate(&space)?;
                let mut h = vec![T::ZERO; d * d];
                for a in 0..d {
                    for b in 0..d {
                        h[a * d + b] = jet.derivative(&[a, b])?;
                    }
                }
                let m =
                    metric_spectrum(&h, d, T::from_f64(self.config.epsilon), self.config.policy)?;
                let range = i * d * d..(i + 1) * d * d;
                hessians[range.clone()].copy_from_slice(&h);
                metrics[range.clone()].copy_from_slice(&m.metric);
                roots[range.clone()].copy_from_slice(&m.inverse_sqrt);
                for a in 0..d * d {
                    factors[i * d * d + a] = (m.inverse_sqrt[a] * amplitude).to_f64();
                }
                condition[i] = m.condition_number;
                logdet[i] = m.log_determinant;
                targets[i] = T::from_f64(target as f64);
                extension[i] = if f.alive[i] { T::ZERO } else { T::ONE };
                if record_curvature {
                    let packed = FitnessJet::from_jet(&jet)?;
                    let get = |j: &FitnessJet<T>| {
                        fitness_curvature(
                            j,
                            T::from_f64(self.config.epsilon),
                            self.config.policy,
                            T::from_f64(self.config.clipping_threshold),
                            false,
                        )
                    };
                    let result = match get(&packed) {
                        Err(GasError::Capability(message))
                            if message.contains("fourth fitness") =>
                        {
                            if space4.is_none() {
                                space4 = Some(JetSpace::new(d, 4)?);
                            }
                            get(&FitnessJet::from_jet(&evaluate(space4.as_ref().unwrap())?)?)
                        }
                        other => other,
                    };
                    match result {
                        Ok(curvature) => {
                            curvature_available[i] = true;
                            ricci[range].copy_from_slice(&curvature.ricci);
                            scalar[i] = curvature.scalar;
                        }
                        Err(GasError::Capability(_)) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            let version = observations.provenance.population_version;
            for (name, values) in [
                ("fitness_hessian", hessians),
                ("fitness_metric", metrics),
                ("metric_inverse_sqrt", roots),
            ] {
                cx.record_field_with_coverage(
                    "O",
                    name,
                    version,
                    &TensorBatch::new(n, vec![d, d], values)?,
                    &available,
                );
            }
            for (name, values) in [
                ("conditional_field_target_slot", targets),
                ("revival_donor_field_extension", extension),
                ("metric_condition_number", condition),
                ("metric_log_determinant", logdet),
            ] {
                cx.record_field_with_coverage(
                    "O",
                    name,
                    version,
                    &TensorBatch::vectors(n, 1, values)?,
                    &available,
                );
            }
            if record_curvature {
                cx.record_field_with_coverage(
                    "O",
                    "fitness_ricci",
                    version,
                    &TensorBatch::new(n, vec![d, d], ricci)?,
                    &curvature_available,
                );
                cx.record_field_with_coverage(
                    "O",
                    "fitness_scalar_curvature",
                    version,
                    &TensorBatch::vectors(n, 1, scalar)?,
                    &curvature_available,
                );
            }
            let adapted = Noise {
                innovation: noise.innovation,
                geometry: NoiseGeometry::Full {
                    factor: FactorValues::PerWalker {
                        values: factors.clone(),
                    },
                },
            };
            let sample = adapted.sample(observations, request, cx).await?;
            cx.record_noise("O", request, &sample, Some(factors));
            Ok(sample)
        })
    }
}
