use algorithmic_gas::{
    boundary::BoundaryPolicy,
    kinetic::{KineticKind, KineticOperator},
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry},
    physics::partvi::ExperimentRequest,
};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig, qft_experiments};
use futures_lite::future::block_on;
use serde_json::json;
fn config() -> RunConfig {
    let mut config = RunConfig {
        benchmark: Benchmark::Sphere,
        walkers: 4,
        dimensions: 3,
        ..Default::default()
    };
    config.gas.boundary = BoundaryPolicy::Unbounded;
    config.gas.kinetic = KineticOperator {
        integrator: KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 0.04,
            friction: 1.,
        },
        noise: Noise {
            innovation: InnovationLaw::Gaussian,
            geometry: NoiseGeometry::Isotropic {
                scale: FactorValues::Constant { values: vec![1.] },
            },
        },
    };
    config
}
#[test]
fn true_source_reruns_and_independent_reweighting_agree_with_sampling_error() {
    block_on(async {
        let result = qft_experiments::run(
            &config(),
            &ExperimentRequest {
                experiment: 19,
                parameters: json!({"replicas":64,"horizon":2,"warmup":1,"theta":0.15}),
            },
        )
        .await
        .unwrap();
        let metric = |name: &str| {
            result
                .metrics
                .iter()
                .find(|m| m.label == name)
                .unwrap()
                .value
                .unwrap()
        };
        assert!(
            metric("Direct minus reweighted").abs()
                < 6. * metric("Independent comparison standard error")
        );
        assert_eq!(result.details["samples"].as_array().unwrap().len(), 64);
        let sample = &result.details["samples"][0];
        assert_ne!(sample["weight_seed"], sample["direct_seed"]);
        assert_ne!(sample["plus"], sample["baseline_direct_group"]);
        assert_eq!(result.details["source_address"]["step"], 2);
    });
}
#[test]
fn conditional_drift_validation_has_disjoint_replica_keys_and_real_stage_budgets() {
    block_on(async {
        let request = ExperimentRequest {
            experiment: 22,
            parameters: json!({"replicas":4,"horizon":1,"warmup":1}),
        };
        let a = qft_experiments::run(&config(), &request).await.unwrap();
        let b = qft_experiments::run(&config(), &request).await.unwrap();
        assert_eq!(a.details, b.details);
        let report = &a.details["report"];
        for key in report["calibration_keys"].as_array().unwrap() {
            assert!(!report["validation_keys"].as_array().unwrap().contains(key));
        }
        let ledgers = a.details["sample_stage_ledgers"].as_array().unwrap();
        assert_eq!(ledgers.len(), 2);
        assert!(
            ledgers[0]
                .as_array()
                .unwrap()
                .iter()
                .any(|b| b["to"] == "O_before_boundary")
        );
    });
}
#[test]
fn continuation_configuration_rejects_non_gaussian_sources_and_excessive_budgets() {
    block_on(async {
        let mut c = config();
        c.gas.kinetic.noise.innovation = InnovationLaw::StandardizedUniform;
        assert!(
            qft_experiments::run(
                &c,
                &ExperimentRequest {
                    experiment: 19,
                    parameters: json!({})
                }
            )
            .await
            .is_err()
        );
        assert!(
            qft_experiments::run(
                &config(),
                &ExperimentRequest {
                    experiment: 22,
                    parameters: json!({"replicas":129})
                }
            )
            .await
            .is_err()
        );
    });
}
