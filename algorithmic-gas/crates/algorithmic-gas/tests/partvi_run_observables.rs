use algorithmic_gas::physics::{
    partvi::{ExperimentRequest, ExperimentResult},
    qft,
};
use serde_json::json;
fn metric(r: &ExperimentResult, name: &str) -> f64 {
    r.metrics
        .iter()
        .find(|m| m.label == name)
        .unwrap()
        .value
        .unwrap()
}
#[test]
fn mechanical_demo_contains_only_executed_stage_measurements() {
    let a = archive();
    let result = algorithmic_gas::physics::fields::analyze(
        &ExperimentRequest {
            experiment: 51,
            parameters: json!({}),
        },
        Some(&a),
    )
    .unwrap();
    let expected = a
        .steps
        .last()
        .unwrap()
        .mechanical_budgets("velocities", 1., a.gas_config.include_truncated)
        .unwrap();
    assert_eq!(result.details["stage_budgets"], json!(expected));
    assert!(metric(&result, "Kinetic telescoping residual").abs() < 1e-12);
    assert!(result.details.get("pair_fixture").is_none());
    assert!(result.details.get("constitutive_reference").is_none());
    assert!(result.metrics.iter().all(|m| !matches!(
        m.label.as_str(),
        "Ricci contraction"
            | "Einstein contraction residual"
            | "Restitution pair energy change"
            | "Restitution identity residual"
            | "Pair momentum residual"
    )));
    assert!(
        result
            .plots
            .iter()
            .all(|p| p.title != "Do not double-count the literal clone")
    );
}
fn archive() -> algorithmic_gas::RunArchive<f64> {
    use algorithmic_gas::{
        domain::{GradientProvider, OperatorFuture, RewardSource},
        kinetic::{KineticKind, KineticOperator, QftExecutionConfig, ViscousForceConfig},
        *,
    };
    struct Zero;
    impl RewardSource<f64> for Zero {
        fn id(&self) -> String {
            "qft-test-zero".into()
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
                    vec![0.; p.len()],
                    Provenance {
                        population_version: p.version,
                        stage: stage.into(),
                        ..Default::default()
                    },
                ))
            })
        }
    }
    impl GradientProvider<f64> for Zero {
        fn id(&self) -> String {
            "qft-test-zero-gradient".into()
        }
        fn gradient<'a>(
            &'a self,
            p: &'a Population<f64>,
            _: &'a mut ExecutionContext,
        ) -> OperatorFuture<'a, TensorBatch<f64>> {
            Box::pin(async move { TensorBatch::vectors(p.len(), 3, vec![0.; p.len() * 3]) })
        }
    }
    futures_lite::future::block_on(async {
        let mut obs = ObservationBatch::positions(
            TensorBatch::vectors(3, 3, vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap(),
        );
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(3, 3, vec![1., 0.2, 0.3, -0.5, 0.1, 0.4, 0., 0.7, -0.3]).unwrap(),
        );
        let config = GasConfig {
            precision: Precision::F64,
            qft: QftExecutionConfig {
                viscosity: Some(ViscousForceConfig {
                    coefficient: 0.3,
                    bandwidth: 1.,
                    row_normalized: false,
                }),
                ..Default::default()
            },
            kinetic: KineticOperator {
                integrator: KineticKind::Baoab {
                    positions: "positions".into(),
                    velocities: "velocities".into(),
                    dt: 0.04,
                    friction: 0.2,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut gas = GasBuilder::new(Population::new(obs).unwrap(), Zero)
            .config(config)
            .gradient(Zero)
            .build()
            .await
            .unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..96 {
            gas.step().await.unwrap();
        }
        gas.recording().unwrap().clone()
    })
}
#[test]
fn all_additional_qft_measurements_consume_executed_records() {
    let a = archive();
    for id in [1, 7, 11, 15, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33] {
        let result = qft::analyze(
            &ExperimentRequest {
                experiment: id,
                parameters: json!({}),
            },
            Some(&a),
        )
        .unwrap();
        assert_eq!(result.experiment, id);
        assert_eq!(result.details["source"], "executed_rust_euclidean_gas");
        assert_eq!(result.details["recorded_steps"], 96);
        assert!(
            result
                .plots
                .iter()
                .any(|p| p.series.iter().any(|s| !s.points.is_empty())),
            "empty plot for {id}"
        );
        for value in result
            .plots
            .iter()
            .flat_map(|p| &p.series)
            .flat_map(|s| &s.points)
            .flatten()
        {
            assert!(value.is_finite(), "nonfinite plot in {id}");
        }
    }
}
#[test]
fn measured_graph_transport_obeys_orientation_and_gauge_laws() {
    let a = archive();
    let result = qft::analyze(
        &ExperimentRequest {
            experiment: 24,
            parameters: json!({"angle":1.3}),
        },
        Some(&a),
    )
    .unwrap();
    assert!(metric(&result, "Resolved interaction triangles") > 0.);
    assert!(metric(&result, "Orientation residual") < 1e-12);
    assert!(metric(&result, "Local phase covariance residual") < 1e-12);
    assert!(metric(&result, "Action variation residual") < 1e-8);
}
#[test]
fn real_run_finite_step_chain_rule_and_density_translation_are_exact() {
    let a = archive();
    let result = qft::analyze(
        &ExperimentRequest {
            experiment: 7,
            parameters: json!({}),
        },
        Some(&a),
    )
    .unwrap();
    assert!(metric(&result, "Discrete chain-rule residual") < 1e-10);
    let result = qft::analyze(
        &ExperimentRequest {
            experiment: 30,
            parameters: json!({"amplitude":2.7,"wavenumber":1.9}),
        },
        Some(&a),
    )
    .unwrap();
    assert!(metric(&result, "Density translation residual") < 1e-12);
}
#[test]
fn fits_keep_candidates_distinct_from_identified_rates() {
    let a = archive();
    let result = qft::analyze(
        &ExperimentRequest {
            experiment: 36,
            parameters: json!({"channels":1,"max_lag":6}),
        },
        Some(&a),
    )
    .unwrap();
    assert_eq!(result.details["readout"], "phase_space");
    for fit in result.details["fits"].as_array().unwrap() {
        assert_eq!(fit["readout"], "phase_space");
        assert!(fit["mass"].is_null());
        if fit["status"] == "inconclusive" {
            assert!(fit["decay_rate"].is_null());
            assert!(fit["frequency"].is_null());
        }
        assert!(fit["heldout_zero_baseline_rmse"].is_number());
    }
}
