use algorithmic_gas::{
    boundary::BoundaryPolicy,
    domain::{
        GradientProvider, NamedObservationExtractor, NamedRewardExtractor, NumericalDomain,
        OperatorFuture,
    },
    donor::{CompanionBatch, CompanionRequest, DonorModule},
    extraction::ExtractionPipeline,
    kinetic::{KineticContext, KineticKind, KineticOperator},
    noise::{FactorValues, Noise, NoiseGeometry, NoiseRequest},
    random::Stream,
    *,
};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel, RunConfig};
use futures_lite::future::block_on;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

struct ReplaceNoise {
    invalid_companions: bool,
}
impl GasOperators<f64> for ReplaceNoise {
    fn id(&self) -> String {
        format!("test-operators/{}/v1", self.invalid_companions)
    }
    fn noise<'a>(
        &'a self,
        _: &'a Noise,
        _: &'a ObservationBatch<f64>,
        r: NoiseRequest,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            TensorBatch::vectors(r.rows, r.dimension, vec![1.; r.rows * r.dimension])
        })
    }
    fn companions<'a>(
        &'a self,
        _: &'a DonorModule,
        r: CompanionRequest<'a, f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, CompanionBatch> {
        Box::pin(async move {
            assert!(matches!(r.stream, Stream::Distance | Stream::Cloning));
            let indices = r
                .pool
                .sources
                .iter()
                .enumerate()
                .filter(|(_, s)| s.frame == r.pool.current_frame)
                .map(|(j, _)| j as u32)
                .collect::<Vec<_>>();
            Ok(CompanionBatch {
                rows: r.population.len(),
                count: 1,
                valid: vec![true; indices.len()],
                indices: if self.invalid_companions {
                    vec![]
                } else {
                    indices
                },
                mutual: false,
            })
        })
    }
}
fn model() -> BenchmarkModel {
    BenchmarkModel {
        benchmark: Benchmark::Sphere,
        field: "positions".into(),
        direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
    }
}
fn population() -> Population<f64> {
    Population::new(ObservationBatch::positions(
        TensorBatch::vectors(2, 2, vec![1., 2., 3., 4.]).unwrap(),
    ))
    .unwrap()
}
fn config() -> GasConfig {
    GasConfig {
        precision: Precision::F64,
        ..Default::default()
    }
}

#[test]
fn custom_noise_and_companion_hooks_work_and_checkpoint_identity_is_checked() {
    block_on(async {
        let mut gas = GasBuilder::new(population(), model())
            .config(config())
            .operators(ReplaceNoise {
                invalid_companions: false,
            })
            .build()
            .await
            .unwrap();
        let saved = gas.checkpoint();
        let report = gas.step().await.unwrap();
        assert_eq!(report.clones, 0);
        assert_eq!(report.distance_companions.indices, vec![0, 1]);
        assert_eq!(
            gas.population()
                .observations
                .field("positions")
                .unwrap()
                .values(),
            &[1.05, 2.05, 3.05, 4.05]
        );
        gas.restore(saved.clone()).unwrap();
        assert_eq!(gas.step().await.unwrap(), report);
        let mut other = GasBuilder::new(population(), model())
            .config(config())
            .build()
            .await
            .unwrap();
        assert!(other.restore(saved).is_err());
    });
}
#[test]
fn malformed_custom_operator_output_is_rejected_before_commit() {
    block_on(async {
        let mut gas = GasBuilder::new(population(), model())
            .config(config())
            .operators(ReplaceNoise {
                invalid_companions: true,
            })
            .build()
            .await
            .unwrap();
        let original = gas.population().clone();
        assert!(matches!(gas.step().await, Err(GasError::Shape(_))));
        assert_eq!(gas.population(), &original);
        assert_eq!(gas.step_number(), 0);
    });
}
struct SharedEvaluation(Arc<AtomicUsize>);
impl DerivedFieldProvider<f64> for SharedEvaluation {
    fn evaluate(
        &self,
        input: &InputBatch<f64>,
        context: &ExtractionContext<'_, f64>,
    ) -> Result<BTreeMap<String, TensorBatch<f64>>> {
        self.0.fetch_add(1, Ordering::Relaxed);
        let ram = &input.numerical["ram"];
        // A shared function evaluation depends on external inputs AND the
        // immutable algorithm state. Both extractors receive its result.
        let mut values = ram.values().to_vec();
        for (x, old) in values
            .iter_mut()
            .zip(context.population.observations.field("positions")?.values())
        {
            *x += old;
        }
        let raw = values
            .chunks(2)
            .map(|r| r[0] * r[0] + r[1] * r[1])
            .collect();
        Ok(BTreeMap::from([
            ("features".into(), TensorBatch::vectors(2, 2, values)?),
            ("objective".into(), TensorBatch::scalars(raw)?),
        ]))
    }
}
#[test]
fn external_input_derives_observations_and_scalar_rewards_from_shared_state_once() {
    block_on(async {
        let count = Arc::new(AtomicUsize::new(0));
        let pipeline = ExtractionPipeline {
            observations: Box::new(NamedObservationExtractor {
                fields: vec![("features".into(), "positions".into())],
            }),
            rewards: Box::new(NamedRewardExtractor {
                field: "objective".into(),
            }),
            derived: Some(Box::new(SharedEvaluation(count.clone()))),
        };
        let input = InputBatch {
            rows: 2,
            version: 12,
            numerical: BTreeMap::from([(
                "ram".into(),
                TensorBatch::vectors(2, 2, vec![1., 1., 1., 1.]).unwrap(),
            )]),
            bytes: BTreeMap::new(),
        };
        let mut gas = GasBuilder::new(population(), model())
            .config(config())
            .operators(ReplaceNoise {
                invalid_companions: false,
            })
            .build()
            .await
            .unwrap();
        let report = gas.step_with_extraction(&input, &pipeline).await.unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 1);
        assert_eq!(report.pre_clone_rewards.raw, vec![13., 41.]);
        assert_eq!(report.pre_clone_rewards.provenance.input_version, 12);
        assert_eq!(
            gas.population()
                .observations
                .field("positions")
                .unwrap()
                .values(),
            &[2.05, 3.05, 4.05, 5.05]
        );
        for i in 0..2 {
            assert!(
                (gas.population().rewards.raw[i]
                    - Benchmark::Sphere
                        .value(
                            gas.population()
                                .observations
                                .field("positions")
                                .unwrap()
                                .row(i)
                                .unwrap()
                        )
                        .unwrap())
                .abs()
                    < 1e-12
            );
        }
    });
}
struct ZeroGradient;
impl GradientProvider<f64> for ZeroGradient {
    fn id(&self) -> String {
        "zero/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move { TensorBatch::vectors(p.len(), 1, vec![0.; p.len()]) })
    }
}
#[test]
fn isolated_gaussian_thermostat_scaling_including_zero_friction() {
    block_on(async {
        let n = 16_384;
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        for (dt, friction) in [(0.02_f64, 0.0_f64), (0.2, 1e-12), (0.4, 1.3)] {
            let mut p = Population::new(ObservationBatch::positions(
                TensorBatch::vectors(n, 1, vec![0.; n]).unwrap(),
            ))
            .unwrap();
            p.observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(n, 1, vec![0.; n]).unwrap(),
            );
            let kinetic = KineticOperator {
                integrator: KineticKind::Baoab {
                    positions: "positions".into(),
                    velocities: "velocities".into(),
                    dt,
                    friction,
                },
                noise: Noise {
                    geometry: NoiseGeometry::Isotropic {
                        scale: FactorValues::Constant { values: vec![1.7] },
                    },
                    ..Default::default()
                },
            };
            kinetic
                .advance(
                    &mut p,
                    KineticContext {
                        gradient: Some(&ZeroGradient),
                        domain: &NumericalDomain,
                        boundary: &BoundaryPolicy::Unbounded,
                        include_truncated: false,
                        seed: 7,
                        step: 1,
                        operators: None,
                        frozen_fitness: None,
                    },
                    &mut cx,
                )
                .await
                .unwrap();
            let v = p.observations.field("velocities").unwrap().values();
            let mean = v.iter().sum::<f64>() / n as f64;
            let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let expected = 1.7_f64.powi(2)
                * if friction == 0. {
                    dt
                } else {
                    -(-2. * friction * dt).exp_m1() / (2. * friction)
                };
            // Fixed seed, 16384 Gaussian rows: 5% is >4 variance SEs.
            assert!(
                (variance / expected - 1.).abs() < 0.05,
                "dt={dt}, gamma={friction}: {variance} vs {expected}"
            );
            assert!(mean.abs() < 0.04 * expected.sqrt());
        }
    });
}
#[test]
fn configuration_rejects_narrowing_memory_and_unsupported_history() {
    block_on(async {
        let mut c = RunConfig::default();
        c.gas.fitness.distance_floor = 1e-80;
        assert!(matches!(
            c.build::<f32>().await,
            Err(GasError::Configuration(_))
        ));
        let mut c = config();
        c.max_batch_elements = 2;
        assert!(
            GasBuilder::new(population(), model())
                .config(c)
                .build()
                .await
                .is_err()
        );
        let mut c = config();
        c.cloning_donors.history_window = 1;
        let mut gas = GasBuilder::new(population(), model())
            .config(c)
            .build()
            .await
            .unwrap();
        let input = InputBatch {
            rows: 2,
            version: 0,
            numerical: BTreeMap::new(),
            bytes: BTreeMap::new(),
        };
        assert!(
            gas.step_with_input(Some(&input))
                .await
                .unwrap_err()
                .to_string()
                .contains("source-aligned")
        );
    });
}
#[test]
fn binary_checkpoint_preserves_invalid_nonfinite_observations() {
    block_on(async {
        let gas = GasBuilder::new(population(), model())
            .config(config())
            .build()
            .await
            .unwrap();
        let mut checkpoint = gas.checkpoint();
        checkpoint
            .population
            .observations
            .field_mut("positions")
            .unwrap()
            .values_mut()[0] = f64::NAN;
        checkpoint.population.validity[0].invalid = true;
        let decoded: Checkpoint<f64> =
            Checkpoint::from_bytes(&checkpoint.to_bytes().unwrap()).unwrap();
        assert!(
            decoded
                .population
                .observations
                .field("positions")
                .unwrap()
                .values()[0]
                .is_nan()
        );
        assert!(decoded.population.validity[0].invalid);
    });
}

struct InvalidReward;
impl algorithmic_gas::domain::RewardSource<f64> for InvalidReward {
    fn id(&self) -> String {
        "invalid-test-reward/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<f64>> {
        Box::pin(async move {
            Ok(RewardBatch::new(
                vec![f64::NAN, 1.],
                Provenance {
                    population_version: p.version,
                    input_version: 0,
                    stage: stage.into(),
                },
            ))
        })
    }
}
#[test]
fn nonfinite_reward_policy_is_explicit() {
    block_on(async {
        assert!(
            GasBuilder::new(population(), InvalidReward)
                .config(config())
                .build()
                .await
                .err()
                .unwrap()
                .to_string()
                .contains("reward evaluation failed")
        );
        let c = GasConfig {
            invalid_reward: InvalidRewardPolicy::Exclude,
            ..config()
        };
        let gas = GasBuilder::new(population(), InvalidReward)
            .config(c)
            .build()
            .await
            .unwrap();
        assert_eq!(gas.population().eligible(false), vec![false, true]);
        assert!(!gas.population().rewards.valid[0]);
    });
}
