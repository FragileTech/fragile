use algorithmic_gas::{
    boundary::{BoundaryPolicy, BoxDomain},
    domain::{GradientProvider, OperatorFuture, RewardSource},
    fitness::ObjectiveDirection,
    geometry::Kernel,
    kinetic::KineticKind,
    noise::{Noise, NoiseRequest},
    *,
};
use futures_lite::future::block_on;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn population(values: &[f64]) -> Population<f64> {
    Population::new(ObservationBatch::positions(
        TensorBatch::vectors(values.len(), 1, values.to_vec()).unwrap(),
    ))
    .unwrap()
}
fn config() -> GasConfig {
    let mut c = GasConfig {
        precision: Precision::F64,
        ..Default::default()
    };
    c.distance_donors.kernel = Kernel::Uniform;
    c.cloning_donors.kernel = Kernel::Uniform;
    c.kinetic.integrator = KineticKind::DirectJump {
        field: "positions".into(),
        amplitude: 0.,
    };
    c.boundary = BoundaryPolicy::AbsorbingBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![0.],
            upper: vec![1.],
        },
    };
    c
}
struct BoundedReward;
impl RewardSource<f64> for BoundedReward {
    fn id(&self) -> String {
        "bounded-review/v1".into()
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
                p.observations
                    .field("positions")?
                    .values()
                    .iter()
                    .map(|&x| if (0. ..=1.).contains(&x) { x } else { f64::NAN })
                    .collect(),
                Provenance {
                    population_version: p.version,
                    stage: stage.into(),
                    ..Default::default()
                },
            ))
        })
    }
}

#[test]
fn clone_jitter_is_classified_before_bounded_reward_and_revives_next_step() {
    block_on(async {
        let mut c = config();
        c.fitness.direction = ObjectiveDirection::Maximize;
        c.fitness.diversity_exponent = 0.;
        c.clone_decision.saturation = 1e-6;
        c.clone_transform.position_field = Some("positions".into());
        c.clone_transform.jitter = Some(Noise::default());
        c.clone_transform.jitter_amplitude = 1000.;
        let mut gas = GasBuilder::new(population(&[0.1, 0.9]), BoundedReward)
            .config(c)
            .build()
            .await
            .unwrap();
        let report = gas.step().await.unwrap();
        assert_eq!(report.eligible, 1);
        assert!(gas.population().validity[0].out_of_bounds);
        assert!(!gas.population().rewards.valid[0]);
        let saved = gas.checkpoint();
        saved.validate().unwrap();
        let next = gas.step().await.unwrap();
        assert_eq!(next.revivals, 1);
        assert_eq!(next.eligible, 2);
        let expected = gas.checkpoint().to_bytes().unwrap();
        gas.restore(saved).unwrap();
        gas.step().await.unwrap();
        assert_eq!(gas.checkpoint().to_bytes().unwrap(), expected);
    });
}

struct Counts {
    gradients: Arc<AtomicUsize>,
    noise: Arc<AtomicUsize>,
}
impl GradientProvider<f64> for Counts {
    fn id(&self) -> String {
        "zero-gradient/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            assert!(p.eligible(false).iter().any(|&a| a));
            self.gradients.fetch_add(1, Ordering::Relaxed);
            TensorBatch::vectors(p.len(), 1, vec![0.; p.len()])
        })
    }
}
impl GasOperators<f64> for Counts {
    fn id(&self) -> String {
        "noise-counter/v1".into()
    }
    fn noise<'a>(
        &'a self,
        _: &'a Noise,
        _: &'a ObservationBatch<f64>,
        request: NoiseRequest,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            self.noise.fetch_add(1, Ordering::Relaxed);
            TensorBatch::vectors(
                request.rows,
                request.dimension,
                vec![0.; request.rows * request.dimension],
            )
        })
    }
}
#[test]
fn baoab_extinction_skips_all_remaining_providers_and_commits() {
    block_on(async {
        let gradients = Arc::new(AtomicUsize::new(0));
        let noise = Arc::new(AtomicUsize::new(0));
        let mut p = population(&[0.5]);
        p.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(1, 1, vec![10.]).unwrap(),
        );
        let mut c = config();
        c.kinetic.integrator = KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 1.,
            friction: 0.,
        };
        let counters = || Counts {
            gradients: gradients.clone(),
            noise: noise.clone(),
        };
        let mut gas = GasBuilder::new(p, BoundedReward)
            .config(c)
            .gradient(counters())
            .operators(counters())
            .build()
            .await
            .unwrap();
        let report = gas.step().await.unwrap();
        assert_eq!(report.eligible, 0);
        assert_eq!(gas.step_number(), 1);
        assert_eq!(gradients.load(Ordering::Relaxed), 1);
        assert_eq!(noise.load(Ordering::Relaxed), 0);
        gas.checkpoint().validate().unwrap();
        assert!(matches!(gas.step().await, Err(GasError::Extinction)));
        assert_eq!(gradients.load(Ordering::Relaxed), 1);
    });
}

#[test]
fn replacement_sets_consistent_provenance_before_reward_evaluation() {
    block_on(async {
        let mut gas = GasBuilder::new(population(&[0.1, 0.9]), BoundedReward)
            .config(config())
            .build()
            .await
            .unwrap();
        gas.step().await.unwrap();
        let old = gas.population().version;
        let mut replacement = population(&[0.2, 0.8]);
        replacement.version = 77;
        gas.replace_population(replacement).await.unwrap();
        let p = gas.population();
        assert_eq!(p.version, old + 1);
        assert_eq!(p.rewards.provenance.population_version, p.version);
        assert_eq!(p.observations.provenance.population_version, p.version);
        assert_eq!(p.rewards.provenance.stage, "external_replace");
        let saved = gas.checkpoint();
        saved.validate().unwrap();
        gas.restore(saved).unwrap();
    });
}

#[test]
fn restore_rejects_nested_corruption_without_changing_live_state() {
    block_on(async {
        let mut gas = GasBuilder::new(population(&[0.1, 0.9]), BoundedReward)
            .config(config())
            .build()
            .await
            .unwrap();
        gas.step().await.unwrap();
        let good = gas.checkpoint();
        let mutations: Vec<fn(&mut Checkpoint<f64>)> = vec![
            |s| {
                s.last_report
                    .as_mut()
                    .unwrap()
                    .pre_clone_rewards
                    .raw
                    .clear()
            },
            |s| s.last_report.as_mut().unwrap().distance_companions.indices[0] = u32::MAX,
            |s| s.last_report.as_mut().unwrap().distance_sources[0].frame = 1000,
            |s| s.last_report.as_mut().unwrap().distance_sources[0].version = u64::MAX,
            |s| s.last_report.as_mut().unwrap().distance_sources[0].generation = u64::MAX,
            |s| s.last_report.as_mut().unwrap().pre_clone_eligible.clear(),
            |s| {
                s.last_report
                    .as_mut()
                    .unwrap()
                    .pre_clone_fitness
                    .reward_stats
                    .global_fallback
                    .clear()
            },
            |s| {
                s.last_report.as_mut().unwrap().clone_plan.choices[0].donors[0].pool_index =
                    u32::MAX
            },
            |s| s.last_report.as_mut().unwrap().clone_plan.choices[0].donors[0].weight = f64::NAN,
            |s| s.last_report.as_mut().unwrap().final_rewards.raw[0] += 1.,
            |s| s.last_report.as_mut().unwrap().result_version += 1,
            |s| s.last_report.as_mut().unwrap().clones += 1,
            |s| s.last_report.as_mut().unwrap().reward_evaluations += 1,
            |s| s.population.observations.provenance.population_version += 1,
            |s| s.schema_version = 1,
        ];
        for mutate in mutations {
            let mut bad = good.clone();
            mutate(&mut bad);
            assert!(gas.restore(bad).is_err());
            assert_eq!(gas.population(), &good.population);
            assert_eq!(gas.step_number(), good.step);
        }
        let mut trailing = good.to_bytes().unwrap();
        trailing.push(0);
        assert!(Checkpoint::<f64>::from_bytes(&trailing).is_err());
        // Deterministic malformed-input smoke corpus, supplemented by the
        // coverage-guided fuzz target. These must return Result, never panic.
        let bytes = good.to_bytes().unwrap();
        for n in (0..bytes.len()).step_by(31) {
            assert!(Checkpoint::<f64>::from_bytes(&bytes[..n]).is_err());
        }
        gas.restore(good).unwrap();
    });
}

struct AbsorbAll;
impl GasOperators<f64> for AbsorbAll {
    fn id(&self) -> String {
        "absorb-all/v1".into()
    }
    fn transform<'a>(
        &'a self,
        _: &'a algorithmic_gas::cloning::CloneTransform,
        _: algorithmic_gas::operators::TransformRequest<'a, f64>,
        p: &'a mut Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, Vec<String>> {
        Box::pin(async move {
            p.observations.field_mut("positions")?.values_mut().fill(2.);
            Ok(vec!["positions".into()])
        })
    }
    fn kinetic<'a>(
        &'a self,
        _: &'a algorithmic_gas::kinetic::KineticOperator,
        _: &'a mut Population<f64>,
        _: algorithmic_gas::kinetic::KineticContext<'a, f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, ()> {
        Box::pin(async {
            Err(GasError::Execution(
                "kinetics called after extinction".into(),
            ))
        })
    }
}
#[test]
fn engine_skips_custom_kinetics_after_post_clone_extinction() {
    block_on(async {
        let mut gas = GasBuilder::new(population(&[0.1, 0.9]), BoundedReward)
            .config(config())
            .operators(AbsorbAll)
            .build()
            .await
            .unwrap();
        let report = gas.step().await.unwrap();
        assert_eq!(report.eligible, 0);
        assert_eq!(gas.step_number(), 1);
        gas.checkpoint().validate().unwrap();
    });
}
