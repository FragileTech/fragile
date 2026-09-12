use algorithmic_gas::{
    Precision, boundary::BoundaryPolicy, fitness::ObjectiveDirection, kinetic::KineticKind,
};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig};

#[test]
fn trace_records_real_stages_without_changing_checkpoint() {
    futures_lite::future::block_on(async {
        let mut c = RunConfig {
            walkers: 8,
            benchmark: Benchmark::Quadratic,
            ..RunConfig::default()
        };
        c.gas.precision = Precision::F64;
        c.gas.boundary = BoundaryPolicy::Unbounded;
        c.gas.kinetic.integrator = KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 0.02,
            friction: 1.,
        };
        let mut plain = c.build::<f64>().await.unwrap();
        let mut traced = c.build::<f64>().await.unwrap();
        traced.set_trace(true).unwrap();
        assert_eq!(plain.step().await.unwrap(), traced.step().await.unwrap());
        assert_eq!(
            plain.checkpoint().to_bytes().unwrap(),
            traced.checkpoint().to_bytes().unwrap()
        );
        let names: Vec<_> = traced
            .stage_trace()
            .unwrap()
            .iter()
            .map(|v| v["stage"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "pre_clone",
                "literal_clone",
                "post_transform",
                "B1",
                "A1",
                "O",
                "A2",
                "B2",
                "post_kinetic"
            ]
        );
        let trace = traced.stage_trace().unwrap().to_vec();
        traced.cancellation_token().cancel();
        assert!(traced.step().await.is_err());
        assert_eq!(traced.stage_trace().unwrap(), trace);
    });
}

#[test]
fn translated_reward_and_independent_gradient_have_stable_identity() {
    futures_lite::future::block_on(async {
        let mut c = RunConfig {
            walkers: 4,
            benchmark: Benchmark::Quadratic,
            potential: Some(Benchmark::Sphere),
            reward_shift: vec![1., 0.],
            ..RunConfig::default()
        };
        c.gas.precision = Precision::F64;
        let run = c.build::<f64>().await.unwrap();
        let x = run.population().observations.field("positions").unwrap();
        for i in 0..4 {
            let p = x.row(i).unwrap();
            assert!(
                (run.population().rewards.raw[i] - 0.5 * ((p[0] - 1.).powi(2) + p[1] * p[1])).abs()
                    < 1e-12
            );
        }
        let cp = run.checkpoint();
        let mut restored = c.build::<f64>().await.unwrap();
        restored.restore(cp.clone()).unwrap();
        c.reward_shift = vec![0., 0.];
        let mut different = c.build::<f64>().await.unwrap();
        assert!(different.restore(cp).is_err());
        c.reward_shift = vec![f64::MAX, 0.];
        c.gas.precision = Precision::F32;
        assert!(c.build::<f32>().await.is_err());
    });
}

#[test]
fn failed_fixture_reward_preserves_checkpoint_and_stage_trace() {
    futures_lite::future::block_on(async {
        let mut c = RunConfig {
            walkers: 4,
            benchmark: Benchmark::Sphere,
            ..RunConfig::default()
        };
        c.gas.precision = Precision::F32;
        c.gas.boundary = BoundaryPolicy::Unbounded;
        let mut run = c.build::<f32>().await.unwrap();
        run.set_trace(true).unwrap();
        run.step().await.unwrap();
        let checkpoint = run.checkpoint().to_bytes().unwrap();
        let trace = run.stage_trace().unwrap().to_vec();
        let mut invalid_reward = run.population().clone();
        // Finite coordinates pass structural validation but overflow the reward.
        invalid_reward
            .observations
            .field_mut("positions")
            .unwrap()
            .values_mut()
            .fill(1e30);
        assert!(run.replace_population(invalid_reward).await.is_err());
        assert_eq!(run.checkpoint().to_bytes().unwrap(), checkpoint);
        assert_eq!(run.stage_trace().unwrap(), trace);
        let mut replacement = run.population().clone();
        replacement
            .observations
            .field_mut("positions")
            .unwrap()
            .values_mut()
            .fill(0.);
        run.replace_population(replacement).await.unwrap();
        assert_eq!(run.step_number(), 1);
        assert!(run.stage_trace().unwrap().is_empty());
        assert!(run.population().rewards.raw.iter().all(|&r| r == 0.));
    });
}

#[test]
fn translated_coordinate_overflow_cannot_become_a_zero_reward() {
    futures_lite::future::block_on(async {
        let mut c = RunConfig {
            walkers: 4,
            benchmark: Benchmark::Sphere,
            ..RunConfig::default()
        };
        c.gas.precision = Precision::F32;
        c.gas.boundary = BoundaryPolicy::Unbounded;
        c.initial_lower = 2.9e38;
        c.initial_upper = 3e38;
        c.reward_shift = vec![-3e38, -3e38];
        // Both input and shift fit in f32, but their difference does not.
        assert!(c.build::<f32>().await.is_err());
    });
}

#[test]
fn explicit_potential_force_is_independent_of_reward_direction() {
    futures_lite::future::block_on(async {
        for independent in [false, true] {
            let mut c = RunConfig {
                walkers: 4,
                benchmark: Benchmark::Quadratic,
                potential: independent.then_some(Benchmark::Quadratic),
                ..RunConfig::default()
            };
            c.gas.precision = Precision::F64;
            c.gas.boundary = BoundaryPolicy::Unbounded;
            c.gas.fitness.direction = ObjectiveDirection::Maximize;
            c.gas.fitness.reward_exponent = 0.;
            c.gas.fitness.diversity_exponent = 0.;
            c.gas.kinetic.integrator = KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.1,
                friction: 1.,
            };
            let mut run = c.build::<f64>().await.unwrap();
            let mut fixture = run.population().clone();
            fixture
                .observations
                .field_mut("positions")
                .unwrap()
                .values_mut()
                .fill(1.);
            run.replace_population(fixture).await.unwrap();
            run.set_trace(true).unwrap();
            run.step().await.unwrap();
            let b1 = run
                .stage_trace()
                .unwrap()
                .iter()
                .find(|state| state["stage"] == "B1")
                .unwrap();
            let values = b1["population"]["observations"]["fields"]["velocities"]["values"]
                .as_array()
                .unwrap();
            let expected = if independent { -0.05 } else { 0.05 };
            assert!(
                values
                    .iter()
                    .all(|value| { (value.as_f64().unwrap() - expected).abs() < 1e-12 })
            );
        }
    });
}
