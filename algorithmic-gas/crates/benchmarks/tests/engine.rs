use algorithmic_gas::{
    boundary::{BoundaryPolicy, BoxDomain},
    domain::{DomainAdapter, OperatorFuture, RewardSource},
    donor::{DonorModule, SamplingLaw},
    fitness::ObjectiveDirection,
    geometry::{Distance, Kernel},
    kinetic::{KineticKind, KineticOperator},
    noise::{FactorValues, Noise, NoiseGeometry},
    *,
};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel, RunConfig};
use futures_lite::future::block_on;

fn config(precision: Precision) -> RunConfig {
    let mut c = RunConfig {
        walkers: 32,
        ..Default::default()
    };
    c.gas.precision = precision;
    c
}
#[test]
fn analytic_benchmark_values_and_derivatives() {
    block_on(async {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        for benchmark in [
            Benchmark::Sphere,
            Benchmark::Rastrigin,
            Benchmark::Rosenbrock,
            Benchmark::StyblinskiTang,
            Benchmark::Quadratic,
        ] {
            let points =
                TensorBatch::vectors(2, 3, vec![0.3_f64, -0.7, 1.2, -1.4, 0.9, 0.1]).unwrap();
            let p = Population::new(ObservationBatch::positions(points.clone())).unwrap();
            let model = BenchmarkModel {
                benchmark,
                field: "positions".into(),
                direction: ObjectiveDirection::Minimize,
            };
            let rewards = model.evaluate(&p, None, "test", &mut cx).await.unwrap();
            let gradient = algorithmic_gas::domain::GradientProvider::gradient(&model, &p, &mut cx)
                .await
                .unwrap();
            for i in 0..2 {
                let row = points.row(i).unwrap();
                assert!((benchmark.value(row).unwrap() - rewards.raw[i]).abs() < 1e-10);
                for j in 0..3 {
                    let mut plus = row.to_vec();
                    let mut minus = row.to_vec();
                    plus[j] += 1e-6;
                    minus[j] -= 1e-6;
                    let finite =
                        (benchmark.value(&plus).unwrap() - benchmark.value(&minus).unwrap()) / 2e-6;
                    assert!(
                        (finite - gradient.row(i).unwrap()[j]).abs() < 2e-6,
                        "{benchmark:?}, coordinate {j}"
                    );
                }
            }
        }
    });
}
#[test]
fn checkpoint_replay_f32_and_f64() {
    async fn replay<T: Real>() {
        let config = config(T::PRECISION);
        let mut gas = config.build::<T>().await.unwrap();
        for _ in 0..3 {
            gas.step().await.unwrap();
        }
        let bytes = gas.checkpoint().to_bytes().unwrap();
        let saved = Checkpoint::from_bytes(&bytes).unwrap();
        let first = gas.step().await.unwrap();
        let final_population = gas.population().clone();
        gas.restore(saved).unwrap();
        let second = gas.step().await.unwrap();
        assert_eq!(first, second);
        assert_eq!(gas.population(), &final_population);
    }
    block_on(replay::<f32>());
    block_on(replay::<f64>());
}
#[test]
fn cancellation_does_not_commit_a_step() {
    block_on(async {
        let mut gas = config(Precision::F32).build::<f32>().await.unwrap();
        let before = gas.population().clone();
        gas.cancellation_token().cancel();
        assert!(matches!(gas.step().await, Err(GasError::Cancelled)));
        assert_eq!(gas.step_number(), 0);
        assert_eq!(gas.population(), &before);
        gas.cancellation_token().reset();
        gas.step().await.unwrap();
        assert_eq!(gas.step_number(), 1);
    });
}
#[test]
fn singletons_and_dead_revival() {
    block_on(async {
        let mut c = config(Precision::F64);
        c.walkers = 1;
        let mut gas = c.build::<f64>().await.unwrap();
        let report = gas.step().await.unwrap();
        assert_eq!(report.distance_companions.indices, vec![0]);
        assert_eq!(
            report.pre_clone_fitness.separation,
            vec![c.gas.fitness.distance_floor]
        );
        c.walkers = 4;
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: ObjectiveDirection::Minimize,
        };
        let mut population = c.initial_population::<f64>().unwrap();
        population.validity[1].terminated = true;
        population.validity[2].truncated = true;
        let mut gas = GasBuilder::new(population, model.clone())
            .gradient(model)
            .config(c.gas)
            .build()
            .await
            .unwrap();
        let report = gas.step().await.unwrap();
        assert_eq!(report.revivals, 2);
        assert_eq!(report.eligible, 4);
    });
}
#[test]
fn extinction_is_explicit() {
    block_on(async {
        let c = config(Precision::F32);
        let mut p = c.initial_population::<f32>().unwrap();
        p.validity.iter_mut().for_each(|s| s.terminated = true);
        let m = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: ObjectiveDirection::Minimize,
        };
        assert!(matches!(
            GasBuilder::new(p, m).config(c.gas).build().await,
            Err(GasError::Extinction)
        ));
    });
}
#[test]
fn baoab_constant_noise_zero_matches_verlet() {
    block_on(async {
        let mut c = config(Precision::F64);
        c.walkers = 1;
        c.dimensions = 1;
        c.benchmark = Benchmark::Quadratic;
        c.gas.boundary = BoundaryPolicy::Unbounded;
        c.gas.kinetic = KineticOperator {
            integrator: KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.1,
                friction: 0.,
            },
            noise: Noise {
                geometry: NoiseGeometry::Isotropic {
                    scale: FactorValues::Constant { values: vec![0.] },
                },
                ..Default::default()
            },
        };
        let mut p = c.initial_population::<f64>().unwrap();
        p.observations
            .field_mut("positions")
            .unwrap()
            .replace_row(0, &[1.])
            .unwrap();
        let m = BenchmarkModel {
            benchmark: Benchmark::Quadratic,
            field: "positions".into(),
            direction: ObjectiveDirection::Minimize,
        };
        let mut gas = GasBuilder::new(p, m.clone())
            .gradient(m)
            .config(c.gas)
            .build()
            .await
            .unwrap();
        gas.step().await.unwrap();
        let p = gas.population();
        assert!((p.observations.field("positions").unwrap().values()[0] - 0.995).abs() < 1e-12);
        assert!((p.observations.field("velocities").unwrap().values()[0] + 0.09975).abs() < 1e-12);
    });
}
#[test]
fn boundary_stops_later_baoab_substeps() {
    block_on(async {
        let mut c = config(Precision::F64);
        c.walkers = 1;
        c.dimensions = 1;
        c.benchmark = Benchmark::Sphere;
        c.gas.boundary = BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-0.01],
                upper: vec![0.01],
            },
        };
        c.gas.kinetic.integrator = KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 0.1,
            friction: 1.,
        };
        let mut p = c.initial_population::<f64>().unwrap();
        p.observations
            .field_mut("positions")
            .unwrap()
            .replace_row(0, &[0.])
            .unwrap();
        p.observations
            .field_mut("velocities")
            .unwrap()
            .replace_row(0, &[1.])
            .unwrap();
        let m = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: ObjectiveDirection::Minimize,
        };
        let mut gas = GasBuilder::new(p, m.clone())
            .gradient(m)
            .config(c.gas)
            .build()
            .await
            .unwrap();
        let report = gas.step().await.unwrap();
        assert_eq!(report.eligible, 0);
        assert_eq!(
            gas.population()
                .observations
                .field("positions")
                .unwrap()
                .values(),
            &[0.05]
        );
        assert_eq!(
            gas.population()
                .observations
                .field("velocities")
                .unwrap()
                .values(),
            &[1.]
        );
        assert!(matches!(gas.step().await, Err(GasError::Extinction)));
    });
}
#[test]
fn phase_space_anisotropic_configuration_runs() {
    block_on(async {
        let mut c = config(Precision::F64);
        c.gas.boundary = BoundaryPolicy::Unbounded;
        c.gas.kinetic.integrator = KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 0.01,
            friction: 1.,
        };
        c.gas.kinetic.noise.geometry = NoiseGeometry::Full {
            factor: FactorValues::Constant {
                values: vec![0.2, 0., 0.1, 0.3],
            },
        };
        let module = DonorModule {
            distance: Distance::PhaseSpace {
                positions: "positions".into(),
                velocities: "velocities".into(),
                position_scale: 1.,
                velocity_scale: 1.,
                lambda: 1.,
                periodic: None,
            },
            kernel: Kernel::Uniform,
            law: SamplingLaw::FisherYates,
            ..Default::default()
        };
        c.gas.distance_donors = module.clone();
        c.gas.cloning_donors = module;
        c.gas.clone_transform.restitution = Some(0.8);
        c.gas.clone_transform.velocity_field = Some("velocities".into());
        let mut gas = c.build::<f64>().await.unwrap();
        for _ in 0..5 {
            gas.step().await.unwrap();
        }
        assert_eq!(gas.step_number(), 5);
        assert!(
            gas.population()
                .observations
                .field("velocities")
                .unwrap()
                .values()
                .iter()
                .all(|v| v.is_finite())
        );
    });
}
#[test]
fn multi_clone_and_overlapping_restitution_rejected_at_prepare() {
    block_on(async {
        let mut c = config(Precision::F32);
        c.gas.cloning_donors.count = 2;
        assert!(c.build::<f32>().await.is_err());
        c.gas.cloning_donors.count = 1;
        c.gas.clone_transform.restitution = Some(0.8);
        c.gas.clone_transform.velocity_field = Some("velocities".into());
        assert!(c.build::<f32>().await.is_err());
    });
}
#[test]
fn historical_donors_rescore_and_replay() {
    block_on(async {
        let mut c = config(Precision::F64);
        c.gas.cloning_donors.history_window = 2;
        let mut gas = c.build::<f64>().await.unwrap();
        gas.step().await.unwrap();
        gas.step().await.unwrap();
        let saved = gas.checkpoint();
        let report = gas.step().await.unwrap();
        assert!(report.clone_plan.sources.iter().any(|s| s.frame < 2));
        let result = gas.population().clone();
        gas.restore(saved).unwrap();
        gas.step().await.unwrap();
        assert_eq!(gas.population(), &result);
    });
}

struct MockDomain;
impl<T: Real> DomainAdapter<T> for MockDomain {
    fn id(&self) -> String {
        "deterministic-counter/v1".into()
    }
    fn refresh_observations(&self, p: &mut Population<T>) -> Result<()> {
        let states = p.states.as_ref().unwrap();
        for (i, bytes) in states.snapshots.iter().enumerate() {
            let counter = bytes[0] as f64;
            p.observations
                .field_mut("features")?
                .replace_row(i, &[T::from_f64(counter), T::from_f64(counter / 4.)])?;
            p.rewards.raw[i] = T::from_f64(counter);
            p.rewards.valid[i] = true;
        }
        Ok(())
    }
    fn reconcile(&self, _: &mut Population<T>, changed: &[String]) -> Result<()> {
        if changed.is_empty() {
            Ok(())
        } else {
            Err(GasError::Domain(
                "mock simulator does not support coordinate repair".into(),
            ))
        }
    }
    fn transition(&self, p: &mut Population<T>, alive: &[bool], _: u64, _: u64) -> Result<()> {
        let states = p.states.as_mut().unwrap();
        for (i, s) in states.snapshots.iter_mut().enumerate() {
            if alive[i] {
                s[0] += 1;
            }
        }
        Ok(())
    }
}
struct StoredReward;
impl<T: Real> RewardSource<T> for StoredReward {
    fn id(&self) -> String {
        "counter-reward/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<T>,
        _: Option<&'a InputBatch<T>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<T>> {
        Box::pin(async move {
            let mut r = p.rewards.clone();
            r.provenance = Provenance {
                stage: stage.into(),
                population_version: p.version,
                input_version: 0,
            };
            Ok(r)
        })
    }
}
#[test]
fn opaque_simulator_features_and_rewards_stay_consistent() {
    block_on(async {
        let mut p = Population::new(ObservationBatch {
            fields: std::collections::BTreeMap::from([(
                "features".into(),
                TensorBatch::vectors(4, 2, vec![0_f32; 8]).unwrap(),
            )]),
            provenance: Provenance::default(),
        })
        .unwrap();
        p.states = Some(StateStore {
            snapshots: vec![vec![1], vec![2], vec![3], vec![4]],
            codec: "counter/v1".into(),
        });
        p.validity[1].terminated = true;
        let module = DonorModule {
            distance: Distance::Cosine {
                field: "features".into(),
                zero_tolerance: 0.,
            },
            kernel: Kernel::Exponential { temperature: 1. },
            ..Default::default()
        };
        let mut config = GasConfig {
            distance_donors: module.clone(),
            cloning_donors: module,
            ..Default::default()
        };
        config.kinetic.integrator = KineticKind::Environment;
        config.fitness.direction = ObjectiveDirection::Maximize;
        let mut gas = GasBuilder::new(p, StoredReward)
            .domain(MockDomain)
            .config(config)
            .build()
            .await
            .unwrap();
        for _ in 0..4 {
            gas.step().await.unwrap();
            let p = gas.population();
            for i in 0..p.len() {
                let counter = p.states.as_ref().unwrap().snapshots[i][0] as f32;
                assert_eq!(
                    p.observations.field("features").unwrap().row(i).unwrap()[0],
                    counter
                );
                assert_eq!(p.rewards.raw[i], counter);
            }
        }
        let before = gas.checkpoint();
        gas.step().await.unwrap();
        let result = gas.population().clone();
        gas.restore(before).unwrap();
        gas.step().await.unwrap();
        assert_eq!(gas.population(), &result);
    });
}
