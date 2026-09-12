//! Immutable conditional-fitness diffusion provider for planar lecture runs.
use crate::Benchmark;
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasConfig, GasError, ObservationBatch, Population, Precision,
    Real, Result, TensorBatch,
    domain::OperatorFuture,
    donor::CompanionReducer,
    fitness::{PositiveMap, Standardizer},
    geometry::Distance,
    kinetic::KineticKind,
    noise::{FactorValues, Noise, NoiseGeometry, NoiseRequest, NoiseSource},
    operators::{FrozenFitnessContext, GasOperators},
    partv_geometry::{
        FitnessInput, MetricPolicy, Objective, conditional_fitness_pipeline, metric_from_hessian,
    },
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveMetricConfig {
    pub epsilon: f64,
    pub temperature: f64,
    #[serde(default)]
    pub policy: MetricPolicy,
}
#[derive(Clone)]
pub struct AdaptiveMetricOperators {
    pub config: AdaptiveMetricConfig,
    pub benchmark: Benchmark,
    pub reward_shift: Vec<f64>,
}
impl<T: Real> GasOperators<T> for AdaptiveMetricOperators {
    fn id(&self) -> String {
        format!(
            "conditional-fitness-2d/v1/{:?}/{:?}/{:?}",
            self.benchmark, self.config, self.reward_shift
        )
    }
    fn validate(&self, c: &GasConfig, p: &Population<T>) -> Result<()> {
        let error = |s: &str| GasError::Capability(s.into());
        if T::PRECISION != Precision::F64
            || c.backend != BackendKind::Cpu
            || p.observations.field("positions")?.width() != 2
        {
            return Err(error(
                "adaptive fitness metric requires CPU/WASM f64 with two position coordinates",
            ));
        }
        if !self.config.epsilon.is_finite()
            || self.config.epsilon <= 0.
            || !self.config.temperature.is_finite()
            || self.config.temperature < 0.
        {
            return Err(error(
                "adaptive metric requires positive epsilon and nonnegative temperature",
            ));
        }
        if !matches!(c.kinetic.integrator,KineticKind::Baoab{ref positions,friction,..} if positions=="positions" && friction>0.)
        {
            return Err(error(
                "adaptive fitness metric requires BAOAB and positive friction",
            ));
        }
        if c.distance_donors.history_window != 0
            || c.distance_donors.count != 1
            || !matches!(c.reducer, CompanionReducer::Mean)
        {
            return Err(error(
                "adaptive fitness metric requires one same-frame companion and mean reduction",
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
                    "adaptive metric supports unscaled Euclidean/phase distance without periodic seams",
                ));
            }
        }
        if !matches!(c.fitness.reward_standardizer, Standardizer::Global { .. })
            || !matches!(
                c.fitness.diversity_standardizer,
                Standardizer::Global { .. }
            )
            || !matches!(c.fitness.reward_map, PositiveMap::Logistic { .. })
            || !matches!(c.fitness.diversity_map, PositiveMap::Logistic { .. })
        {
            return Err(error(
                "adaptive metric requires smooth global/logistic fitness",
            ));
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
                GasError::Capability("adaptive metric missing frozen selection context".into())
            })?;
            let (friction, velocities) = match &f.config.kinetic.integrator {
                KineticKind::Baoab {
                    friction,
                    velocities,
                    ..
                } => (*friction, velocities),
                _ => {
                    return Err(GasError::Capability(
                        "adaptive metric requires O stage".into(),
                    ));
                }
            };
            let frozen_positions = f.population.observations.field("positions")?;
            let actual = observations.field("positions")?;
            let shift = |axis: usize| self.reward_shift.get(axis).copied().unwrap_or(0.);
            let points: Vec<_> = frozen_positions
                .values()
                .chunks_exact(2)
                .map(|p| [p[0].to_f64() - shift(0), p[1].to_f64() - shift(1)])
                .collect();
            let velocity_field = match &f.config.distance_donors.distance {
                Distance::PhaseSpace { velocities, .. } => velocities,
                _ => velocities,
            };
            let vs = f
                .population
                .observations
                .field(velocity_field)?
                .values()
                .chunks_exact(2)
                .map(|v| [v[0].to_f64(), v[1].to_f64()])
                .collect();
            let companions: Vec<_> = (0..points.len())
                .map(|i| {
                    if f.companions.valid[i] {
                        f.pool.sources[f.companions.indices[i] as usize].slot as usize
                    } else {
                        i
                    }
                })
                .collect();
            let velocity_weight = match f.config.distance_donors.distance {
                Distance::PhaseSpace { lambda, .. } => lambda,
                _ => 0.,
            };
            let objective = match self.benchmark {
                Benchmark::Sphere => Objective::Sphere,
                Benchmark::Quadratic => Objective::Quadratic {
                    curvature: [[1., 0.], [0., 1.]],
                },
                Benchmark::Rastrigin => Objective::Rastrigin,
                Benchmark::Rosenbrock => Objective::Rosenbrock,
                Benchmark::StyblinskiTang => Objective::StyblinskiTang,
            };
            let mut input = FitnessInput {
                points,
                velocities: vs,
                alive: f.alive.to_vec(),
                companions,
                companion_valid: f.companions.valid.clone(),
                target: 0,
                query: [0.; 2],
                objective,
                sigma_min: 0.001,
                distance_floor: f.config.fitness.distance_floor,
                velocity_weight,
                reward_exponent: 1.,
                distance_exponent: 1.,
                map_amplitude: 2.,
                map_floor: 1e-6,
                maximize: false,
                metric_epsilon: self.config.epsilon,
                metric_policy: self.config.policy,
            };
            let mut factors = Vec::new();
            let mut hessians = Vec::new();
            let mut metrics = Vec::new();
            let mut targets = Vec::new();
            let mut extensions = Vec::new();
            let mut inverse_roots = Vec::new();
            let factor = (2. * friction * self.config.temperature).sqrt();
            let available = eligible
                .map(|m| m.to_vec())
                .unwrap_or_else(|| vec![true; request.rows]);
            for (i, &is_available) in available.iter().enumerate() {
                if !is_available {
                    factors.extend([0.; 4]);
                    hessians.extend([T::ZERO; 4]);
                    metrics.extend([T::ZERO; 4]);
                    inverse_roots.extend([T::ZERO; 4]);
                    targets.push(T::ZERO);
                    extensions.push(T::ZERO);
                    continue;
                }
                let revived = !f.alive[i];
                let target = if revived {
                    let choice = &f.clone_plan.choices[i];
                    let donor = choice.donors.first().ok_or_else(|| {
                        GasError::Capability("revived adaptive row lacks donor field".into())
                    })?;
                    f.clone_plan.sources[donor.pool_index as usize].slot as usize
                } else {
                    i
                };
                input.target = target;
                input.query = [
                    actual.values()[2 * i].to_f64() - shift(0),
                    actual.values()[2 * i + 1].to_f64() - shift(1),
                ];
                let jet = conditional_fitness_pipeline(&input, &f.config.fitness)?;
                let m = metric_from_hessian(jet.hessian, self.config.epsilon, self.config.policy)?;
                factors.extend(m.inverse_sqrt.iter().flatten().map(|x| x * factor));
                hessians.extend(jet.hessian.iter().flatten().map(|&x| T::from_f64(x)));
                metrics.extend(m.metric.iter().flatten().map(|&x| T::from_f64(x)));
                inverse_roots.extend(m.inverse_sqrt.iter().flatten().map(|&x| T::from_f64(x)));
                targets.push(T::from_f64(target as f64));
                extensions.push(T::from_f64(if revived { 1. } else { 0. }));
            }
            let version = observations.provenance.population_version;
            cx.record_field_with_coverage(
                "O",
                "fitness_hessian",
                version,
                &TensorBatch::new(request.rows, vec![2, 2], hessians)?,
                &available,
            );
            cx.record_field_with_coverage(
                "O",
                "fitness_metric",
                version,
                &TensorBatch::new(request.rows, vec![2, 2], metrics)?,
                &available,
            );
            cx.record_field_with_coverage(
                "O",
                "metric_inverse_sqrt",
                version,
                &TensorBatch::new(request.rows, vec![2, 2], inverse_roots)?,
                &available,
            );
            cx.record_field_with_coverage(
                "O",
                "conditional_field_target_slot",
                version,
                &TensorBatch::vectors(request.rows, 1, targets)?,
                &available,
            );
            cx.record_field_with_coverage(
                "O",
                "revival_donor_field_extension",
                version,
                &TensorBatch::vectors(request.rows, 1, extensions)?,
                &available,
            );
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
