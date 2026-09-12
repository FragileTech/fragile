use algorithmic_gas::{
    Precision, RecordingConfig, boundary::BoundaryPolicy, kinetic::KineticKind,
    physics::balances::analyze_step,
};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig, physics_metric::PhysicsMetricConfig};
fn config(d: usize) -> RunConfig {
    let mut c = RunConfig {
        dimensions: d,
        walkers: 12,
        benchmark: Benchmark::Quadratic,
        physics_metric: Some(PhysicsMetricConfig::default()),
        ..Default::default()
    };
    c.gas.precision = Precision::F64;
    c.gas.boundary = BoundaryPolicy::Unbounded;
    c.gas.kinetic.integrator = KineticKind::Baoab {
        positions: "positions".into(),
        velocities: "velocities".into(),
        dt: 0.02,
        friction: 1.,
    };
    c
}
#[test]
fn generalized_metric_recording_is_observational_and_balances_actual_stages() {
    futures_lite::future::block_on(async {
        for d in [1, 2, 3, 4, 8] {
            let c = config(d);
            let mut plain = c.build::<f64>().await.unwrap();
            let mut recorded = c.build::<f64>().await.unwrap();
            recorded
                .start_recording(RecordingConfig::default())
                .unwrap();
            for _ in 0..3 {
                assert_eq!(plain.step().await.unwrap(), recorded.step().await.unwrap());
                assert_eq!(plain.population(), recorded.population());
                let archive = recorded.recording().unwrap();
                archive.validate().unwrap();
                let step = archive.steps.last().unwrap();
                let b =
                    analyze_step(step, &c.gas, Some(&c.benchmark.physics_objective(d))).unwrap();
                assert!(b.kinetic_telescoping_residual.abs() < 1e-12);
                assert!(b.thermostat.is_some(), "{:?}", b.unavailable);
                let recorded_moments = step
                    .thermostat_moments("velocities", 1., 0.02, 1., true)
                    .unwrap();
                assert!(
                    (recorded_moments.predicted_energy_change
                        - b.thermostat.as_ref().unwrap().energy_mean_delta)
                        .abs()
                        < 1e-12
                );
                let curvature = step
                    .field_evaluations
                    .iter()
                    .find(|f| f.field == "fitness_scalar_curvature")
                    .unwrap();
                assert_eq!(curvature.rows, c.walkers);
                assert!(curvature.available.iter().any(|v| *v));
            }
            recorded.stop_recording();
            let a = plain.checkpoint();
            let b = recorded.checkpoint();
            // Last report execution counters are equal already; recording has no RNG state.
            assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap());
        }
    });
}
#[test]
fn checkpoint_replicas_preserve_history_and_change_only_future_randomness() {
    futures_lite::future::block_on(async {
        let c = config(3);
        let mut run = c.build::<f64>().await.unwrap();
        run.step().await.unwrap();
        let checkpoint = run.checkpoint();
        assert!(checkpoint.with_future_seed(checkpoint.config.seed).is_err());
        let replicate = |seed| checkpoint.with_future_seed(seed).unwrap();
        let a = replicate(991);
        let b = replicate(992);
        assert_eq!(a.population, b.population);
        assert_eq!(a.history, b.history);
        assert_eq!(a.step, b.step);
        assert_eq!(a.operator_set, b.operator_set);
        let mut ac = c.clone();
        ac.gas = a.config.clone();
        let mut bc = c;
        bc.gas = b.config.clone();
        let mut ar = ac.build::<f64>().await.unwrap();
        let mut br = bc.build::<f64>().await.unwrap();
        ar.restore(a.clone()).unwrap();
        br.restore(b).unwrap();
        ar.step().await.unwrap();
        br.step().await.unwrap();
        assert_ne!(ar.population(), br.population());
        let result = ar.checkpoint().to_bytes().unwrap();
        ar.restore(a).unwrap();
        ar.step().await.unwrap();
        assert_eq!(ar.checkpoint().to_bytes().unwrap(), result);
    });
}
#[test]
fn checkpoint_and_archive_require_the_supported_schema_and_explicit_coverage() {
    futures_lite::future::block_on(async {
        let c = config(3);
        let mut run = c.build::<f64>().await.unwrap();
        run.start_recording(RecordingConfig::default()).unwrap();
        run.step().await.unwrap();
        let checkpoint = run.checkpoint();
        let bytes = checkpoint.to_bytes().unwrap();
        let decoded = algorithmic_gas::Checkpoint::<f64>::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_bytes().unwrap(), bytes);
        let archive = checkpoint.recording.as_ref().unwrap();
        let json = serde_json::to_string(archive).unwrap();
        let decoded = algorithmic_gas::RunArchive::<f64>::from_json(&json).unwrap();
        assert_eq!(serde_json::to_string(&decoded).unwrap(), json);
        let archive_bytes = archive.to_bytes().unwrap();
        assert_eq!(
            algorithmic_gas::RunArchive::<f64>::from_bytes(&archive_bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            archive_bytes
        );

        for unsupported in [0, 1, 2, 3, 5, u32::MAX] {
            let mut invalid = checkpoint.clone();
            invalid.schema_version = unsupported;
            let mut bytes = Vec::new();
            ciborium::ser::into_writer(&invalid, &mut bytes).unwrap();
            assert!(algorithmic_gas::Checkpoint::<f64>::from_bytes(&bytes).is_err());
            assert!(run.restore(invalid).is_err());
        }
        for unsupported in [0, 1, 3, u32::MAX] {
            let mut invalid = archive.clone();
            invalid.schema_version = unsupported;
            assert!(
                algorithmic_gas::RunArchive::<f64>::from_json(
                    &serde_json::to_string(&invalid).unwrap()
                )
                .is_err()
            );
            let mut bytes = Vec::new();
            ciborium::ser::into_writer(&invalid, &mut bytes).unwrap();
            assert!(algorithmic_gas::RunArchive::<f64>::from_bytes(&bytes).is_err());
        }
        let mut incomplete = archive.clone();
        incomplete.steps[0].field_evaluations[0].available.clear();
        assert!(incomplete.validate().is_err());
        let mut missing = serde_json::to_value(archive).unwrap();
        missing["steps"][0]["field_evaluations"][0]
            .as_object_mut()
            .unwrap()
            .remove("available");
        assert!(algorithmic_gas::RunArchive::<f64>::from_json(&missing.to_string()).is_err());
        assert_eq!(
            run.checkpoint().to_bytes().unwrap(),
            checkpoint.to_bytes().unwrap()
        );
    });
}

#[test]
fn local_normalization_preserves_recording_replay_in_three_dimensions() {
    use algorithmic_gas::{
        fitness::Standardizer,
        geometry::{Distance, Kernel},
    };
    futures_lite::future::block_on(async {
        for width in [0.3, 1., 3.] {
            let mut c = config(3);
            c.gas.fitness.reward_standardizer = Standardizer::Local {
                sigma_min: 0.03,
                distance: Distance::default(),
                kernel: Kernel::Gaussian { width },
                include_self: true,
            };
            c.gas.fitness.diversity_standardizer = c.gas.fitness.reward_standardizer.clone();
            let mut plain = c.build::<f64>().await.unwrap();
            let mut recorded = c.build::<f64>().await.unwrap();
            recorded
                .start_recording(RecordingConfig::default())
                .unwrap();
            for _ in 0..3 {
                assert_eq!(plain.step().await.unwrap(), recorded.step().await.unwrap());
                assert_eq!(plain.population(), recorded.population());
            }
            recorded.stop_recording();
            assert_eq!(
                plain.checkpoint().to_bytes().unwrap(),
                recorded.checkpoint().to_bytes().unwrap()
            );
        }
    });
}

#[test]
fn three_dimensional_f32_metric_preserves_recording_and_finite_curvature() {
    futures_lite::future::block_on(async {
        let mut c = config(3);
        c.gas.precision = Precision::F32;
        let mut plain = c.build::<f32>().await.unwrap();
        let mut recorded = c.build::<f32>().await.unwrap();
        recorded
            .start_recording(RecordingConfig::default())
            .unwrap();
        for _ in 0..4 {
            assert_eq!(plain.step().await.unwrap(), recorded.step().await.unwrap());
            assert_eq!(plain.population(), recorded.population());
            let archive = recorded.recording().unwrap();
            archive.validate().unwrap();
            let curvature = archive
                .steps
                .last()
                .unwrap()
                .field_evaluations
                .iter()
                .find(|f| f.field == "fitness_scalar_curvature")
                .unwrap();
            assert!(curvature.available.iter().any(|v| *v));
            assert!(
                curvature
                    .values
                    .iter()
                    .zip(&curvature.available)
                    .all(|(v, available)| !available || v.is_finite())
            );
        }
        recorded.stop_recording();
        assert_eq!(
            plain.checkpoint().to_bytes().unwrap(),
            recorded.checkpoint().to_bytes().unwrap()
        );
    });
}
