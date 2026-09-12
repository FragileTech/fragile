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
fn executed_algorithms_define_modes_channels_and_companion_algebra() {
    let a = archive();
    let run = |id, parameters| {
        qft::analyze(
            &ExperimentRequest {
                experiment: id,
                parameters,
            },
            Some(&a),
        )
        .unwrap()
    };
    let g = run(3, json!({"modes":3}));
    assert_eq!(metric(&g, "Observed rows"), 96.);
    assert_eq!(metric(&g, "Raw contributing walker rows"), 288.);
    assert!(metric(&g, "Measured-mode CAR residual") < 1e-9);
    assert!(metric(&g, "Replica determinant residual") < 1e-9);
    let channel = run(4, json!({"modes":2,"lag":1}));
    assert!(metric(&channel, "Unital residual") < 1e-10);
    assert!(metric(&channel, "Linear CAR contraction residual") < 1e-10);
    assert!(metric(&channel, "Held-out pairs") > 10.);
    assert!(channel.details["recorded_law"]["source_whitening"].is_array());
    assert!(
        !channel
            .plots
            .iter()
            .any(|p| p.title == "Top exterior sector through time")
    );
    let clones = run(2, json!({}));
    assert!(metric(&clones, "Recorded acceptance probability residual") < 1e-12);
    assert!(metric(&clones, "Weighted antisymmetry residual") < 1e-12);
    let invariants = run(9, json!({"angle":1.2}));
    assert_eq!(metric(&invariants, "Valid triples"), 1.);
    assert!(metric(&invariants, "Common SU(3) and ray-phase residual") < 1e-10);
    assert!(metric(&invariants, "Projector trace residual") < 1e-10);
    assert!(metric(&invariants, "Baryon Gram determinant residual") < 1e-10);
    let doublets = run(10, json!({"length":0.7,"phase_scale":1.3}));
    assert_eq!(metric(&doublets, "Exact conditional scalar rows"), 288.);
    assert!(metric(&doublets, "Conditional pairwise variance residual") < 1e-12);
    assert!(
        doublets.details["rows"]
            .as_array()
            .unwrap()
            .iter()
            .any(|x| x["valid_doublet"].as_bool() == Some(true))
    );
    let writes = run(14, json!({}));
    assert!(metric(&writes, "Literal clone write residual") < 1e-12);
}
#[test]
fn measured_channel_insufficient_and_zero_rank_are_unavailable() {
    let mut a = archive();
    a.steps.truncate(3);
    let req = ExperimentRequest {
        experiment: 4,
        parameters: json!({}),
    };
    let out = qft::analyze(&req, Some(&a)).unwrap();
    assert_eq!(out.details["status"], "unavailable");
    assert!(
        !out.metrics
            .iter()
            .any(|m| m.label == "Vacuum multiplicative defect")
    );
}
#[test]
fn finite_reference_fock_space_quotients_dependent_readouts() {
    let out = qft::analyze(
        &ExperimentRequest {
            experiment: 3,
            parameters: json!({"modes":6}),
        },
        None,
    )
    .unwrap();
    assert_eq!(metric(&out, "Gram rank"), 2.);
    assert_eq!(metric(&out, "Fock dimension"), 4.);
}
