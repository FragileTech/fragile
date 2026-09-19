use algorithmic_gas::{
    domain::{OperatorFuture, RewardSource},
    tracking::RecordingConfig,
    *,
};
use futures_lite::future::block_on;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Objective(Arc<AtomicBool>);
impl RewardSource<f64> for Objective {
    fn id(&self) -> String {
        "elite-test/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<f64>> {
        Box::pin(async move {
            if stage == "post_kinetic" && self.0.load(Ordering::Relaxed) {
                return Err(GasError::Numerical("test failure".into()));
            }
            Ok(RewardBatch::new(
                p.observations
                    .field("positions")?
                    .values()
                    .iter()
                    .map(|x| x * x)
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
fn build(n_elite: usize, fail: Arc<AtomicBool>) -> AlgorithmicGas<f64> {
    let population = Population::new(ObservationBatch::positions(
        TensorBatch::scalars(vec![1., 2., 3., 4.]).unwrap(),
    ))
    .unwrap();
    let mut config = GasConfig {
        n_elite,
        precision: Precision::F64,
        ..Default::default()
    };
    config.distance_donors.history_window = 2;
    config.cloning_donors.history_window = 2;
    block_on(
        GasBuilder::new(population, Objective(fail))
            .config(config)
            .build(),
    )
    .unwrap()
}
#[test]
fn elites_checkpoint_recording_motion_rollback_and_reset() {
    block_on(async {
        let failure = Arc::new(AtomicBool::new(false));
        let mut gas = build(2, failure.clone());
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let saved = gas.checkpoint();
        let bank = saved.elites.as_ref().unwrap().population.as_ref().unwrap();
        let positions = bank
            .observations
            .field("positions")
            .unwrap()
            .values()
            .to_vec();
        let report = gas.step().await.unwrap();
        assert!(
            report.clone_plan.choices[..2]
                .iter()
                .all(|c| !c.accepted && c.probability == Some(0.))
        );
        assert!(
            report
                .clone_plan
                .sources
                .iter()
                .any(|s| s.frame == 1 && s.slot < 2)
        );
        let record = gas.recording().unwrap().steps.last().unwrap();
        assert_eq!(
            &record
                .before
                .observations
                .field("positions")
                .unwrap()
                .values()[..2],
            &positions
        );
        let moved = record
            .stages
            .iter()
            .find(|s| s.stage == "elite_selection")
            .unwrap();
        assert_ne!(&moved.fields["positions"].values[..2], &positions);
        assert_eq!(
            &gas.population()
                .observations
                .field("positions")
                .unwrap()
                .values()[..2],
            gas.checkpoint()
                .elites
                .unwrap()
                .population
                .unwrap()
                .observations
                .field("positions")
                .unwrap()
                .values()
        );
        gas.recording().unwrap().validate().unwrap();
        let mut resumed = build(2, failure.clone());
        resumed
            .restore(Checkpoint::from_bytes(&saved.to_bytes().unwrap()).unwrap())
            .unwrap();
        assert_eq!(resumed.step().await.unwrap(), report);
        assert_eq!(resumed.population(), gas.population());
        // Historical donor windows must remain valid after repeated reinjection.
        for _ in 0..4 {
            gas.step().await.unwrap();
        }
        gas.checkpoint().validate().unwrap();
        let before = gas.checkpoint().to_bytes().unwrap();
        failure.store(true, Ordering::Relaxed);
        assert!(gas.step().await.is_err());
        assert_eq!(before, gas.checkpoint().to_bytes().unwrap());
        failure.store(false, Ordering::Relaxed);
        let token = gas.cancellation_token();
        token.cancel();
        assert!(gas.step().await.is_err());
        assert_eq!(before, gas.checkpoint().to_bytes().unwrap());
        token.reset();
        gas.replace_population(gas.population().clone())
            .await
            .unwrap();
        assert!(gas.checkpoint().elites.unwrap().population.is_none());
    });
}
#[test]
fn disabled_legacy_validation_full_population_and_memory() {
    block_on(async {
        let fail = Arc::new(AtomicBool::new(false));
        let mut gas = build(0, fail.clone());
        let mut legacy = serde_json::to_value(gas.checkpoint()).unwrap();
        legacy.as_object_mut().unwrap().remove("elites");
        legacy["config"].as_object_mut().unwrap().remove("n_elite");
        let checkpoint: Checkpoint<f64> = serde_json::from_value(legacy).unwrap();
        let mut restored = build(0, fail.clone());
        restored.restore(checkpoint).unwrap();
        assert_eq!(gas.step().await.unwrap(), restored.step().await.unwrap());
        let mut full = build(4, fail.clone());
        full.step().await.unwrap();
        let report = full.step().await.unwrap();
        assert_eq!(report.clones, 0);
        let mut cp = full.checkpoint();
        cp.elites = None;
        assert!(cp.validate().is_err());
        let mut invalid = full.config().clone();
        invalid.n_elite = 5;
        assert!(invalid.validate(full.population(), false).is_err());
        let enabled_budget = full
            .config()
            .working_set_bytes(full.population(), &[], None)
            .unwrap();
        invalid.n_elite = 0;
        let disabled_budget = invalid
            .working_set_bytes(full.population(), &[], None)
            .unwrap();
        assert!(enabled_budget > disabled_budget);
        invalid.n_elite = 4;
        invalid.max_memory_bytes = disabled_budget;
        assert!(invalid.validate(full.population(), false).is_err());
        let input = InputBatch {
            rows: 4,
            version: 0,
            numerical: Default::default(),
            bytes: Default::default(),
        };
        let before = full.checkpoint().to_bytes().unwrap();
        assert!(
            full.step_with_input(Some(&input))
                .await
                .unwrap_err()
                .to_string()
                .contains("source-aligned")
        );
        assert_eq!(before, full.checkpoint().to_bytes().unwrap());
    });
}

struct InvalidatingObjective(Arc<AtomicBool>);
impl RewardSource<f64> for InvalidatingObjective {
    fn id(&self) -> String {
        "invalidating/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<f64>> {
        Box::pin(async move {
            let value = if stage == "post_kinetic" && self.0.load(Ordering::Relaxed) {
                f64::NAN
            } else {
                1.
            };
            Ok(RewardBatch::new(
                vec![value; p.len()],
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
fn retained_elites_recover_an_extinct_population() {
    block_on(async {
        let invalid = Arc::new(AtomicBool::new(false));
        let p = Population::new(ObservationBatch::positions(
            TensorBatch::scalars(vec![1., 2., 3., 4.]).unwrap(),
        ))
        .unwrap();
        let config = GasConfig {
            n_elite: 2,
            precision: Precision::F64,
            invalid_reward: engine::InvalidRewardPolicy::Exclude,
            ..Default::default()
        };
        let mut gas = GasBuilder::new(p, InvalidatingObjective(invalid.clone()))
            .config(config)
            .build()
            .await
            .unwrap();
        assert_eq!(gas.elite_count(), 0);
        gas.step().await.unwrap();
        invalid.store(true, Ordering::Relaxed);
        assert_eq!(gas.step().await.unwrap().eligible, 2);
        assert_eq!(gas.elite_count(), 2);
        let saved = gas.checkpoint().to_bytes().unwrap();
        gas.restore(Checkpoint::from_bytes(&saved).unwrap())
            .unwrap();
        invalid.store(false, Ordering::Relaxed);
        assert_eq!(gas.step().await.unwrap().eligible, 4);
        assert_eq!(gas.elite_count(), 2);
    });
}

#[test]
fn twenty_elites_move_then_best_walkers_are_copied_back_after_every_step() {
    block_on(async {
        for direction in [
            fitness::ObjectiveDirection::Minimize,
            fitness::ObjectiveDirection::Maximize,
        ] {
            let initial: Vec<f64> = (1..=64).map(f64::from).collect();
            let p = Population::new(ObservationBatch::positions(
                TensorBatch::scalars(initial).unwrap(),
            ))
            .unwrap();
            let mut config = GasConfig {
                n_elite: 20,
                precision: Precision::F64,
                ..Default::default()
            };
            config.fitness.direction = direction;
            config.kinetic.integrator = kinetic::KineticKind::DirectJump {
                field: "positions".into(),
                amplitude: 10.,
            };
            let mut gas = GasBuilder::new(p, Objective(Arc::new(AtomicBool::new(false))))
                .config(config)
                .build()
                .await
                .unwrap();
            gas.start_recording(RecordingConfig::default()).unwrap();
            let mut best: Vec<f64> = vec![];
            for step in 0..12 {
                let report = gas.step().await.unwrap();
                if step > 0 {
                    assert!(report.clone_plan.choices[..20].iter().all(|c| !c.accepted));
                    assert!(report.clone_plan.sources.iter().any(|s| s.slot < 20));
                }
                let record = gas.recording().unwrap().steps.last().unwrap();
                let moved = record
                    .stages
                    .iter()
                    .find(|s| s.stage == "elite_selection")
                    .unwrap();
                let positions = &moved.fields["positions"].values;
                assert_ne!(
                    &positions[..20],
                    &record
                        .before
                        .observations
                        .field("positions")
                        .unwrap()
                        .values()[..20]
                );
                best.extend_from_slice(positions);
                // Match C++: prior bank first on ties, then population order;
                // no deduplication or freezing of the kinetic candidates.
                best.sort_by(|a, b| {
                    direction
                        .orient(b * b)
                        .partial_cmp(&direction.orient(a * a))
                        .unwrap()
                });
                best.truncate(20);
                assert_eq!(
                    &gas.population()
                        .observations
                        .field("positions")
                        .unwrap()
                        .values()[..20],
                    &best
                );
            }
            gas.checkpoint().validate().unwrap();
            gas.recording().unwrap().validate().unwrap();
        }
    });
}

#[test]
fn tied_elites_return_to_saved_positions_after_moving() {
    block_on(async {
        let p = Population::new(ObservationBatch::positions(
            TensorBatch::scalars((1..=20).map(f64::from).collect()).unwrap(),
        ))
        .unwrap();
        let config = GasConfig {
            n_elite: 20,
            precision: Precision::F64,
            ..Default::default()
        };
        let mut gas = GasBuilder::new(p, InvalidatingObjective(Arc::new(AtomicBool::new(false))))
            .config(config)
            .build()
            .await
            .unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let saved = gas
            .population()
            .observations
            .field("positions")
            .unwrap()
            .values()
            .to_vec();
        for _ in 0..5 {
            assert_eq!(gas.step().await.unwrap().clones, 0);
            let record = gas.recording().unwrap().steps.last().unwrap();
            let moved = record
                .stages
                .iter()
                .find(|s| s.stage == "elite_selection")
                .unwrap();
            assert_ne!(moved.fields["positions"].values, saved);
            assert_eq!(
                gas.population()
                    .observations
                    .field("positions")
                    .unwrap()
                    .values(),
                saved
            );
        }
        gas.recording().unwrap().validate().unwrap();
    });
}
