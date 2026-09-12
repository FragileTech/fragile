use algorithmic_gas::{
    domain::{GradientProvider, OperatorFuture, RewardSource},
    kinetic::{KineticKind, KineticOperator, QftExecutionConfig, ViscousForceConfig},
    noise::{FactorValues, InnovationLaw, InnovationShift, Noise, NoiseGeometry},
    physics::{fields, partvi::ExperimentRequest},
    random::Stream,
    tracking::RecordingConfig,
    *,
};
use futures_lite::future::block_on;
use serde_json::json;
struct Zero;
impl RewardSource<f64> for Zero {
    fn id(&self) -> String {
        "zero/v1".into()
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
        "zero-gradient/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move { TensorBatch::vectors(p.len(), 3, vec![0.; p.len() * 3]) })
    }
}
async fn executed(seed: u64, law: InnovationLaw) -> AlgorithmicGas<f64> {
    let mut observations = ObservationBatch::positions(
        TensorBatch::vectors(
            4,
            3,
            vec![-0.3, 0., 0., 0.1, 0.2, 0., 0.4, -0.2, 0., 0.8, 0.1, 0.],
        )
        .unwrap(),
    );
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(
            4,
            3,
            vec![1., 0.3, 0., -1., 0., 0.1, 0.2, -0.5, 0., 0.6, 0., -0.2],
        )
        .unwrap(),
    );
    let config = GasConfig {
        seed,
        precision: Precision::F64,
        backend: BackendKind::Cpu,
        qft: QftExecutionConfig {
            viscosity: Some(ViscousForceConfig {
                coefficient: 0.3,
                bandwidth: 1.,
                row_normalized: false,
            }),
            innovation_shifts: vec![InnovationShift {
                step: 1,
                stream: Stream::Kinetic,
                substep: 2,
                walker: 0,
                coordinate: 1,
                shift: 0.7,
            }],
        },
        kinetic: KineticOperator {
            integrator: KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.08,
                friction: 0.7,
            },
            noise: Noise {
                innovation: law,
                geometry: NoiseGeometry::LowRank {
                    rank: 2,
                    factor: FactorValues::Constant {
                        values: vec![0.7, 0.2, 0.1, 0.8, 0.3, -0.1],
                    },
                },
            },
        },
        ..Default::default()
    };
    let mut gas = GasBuilder::new(Population::new(observations).unwrap(), Zero)
        .config(config)
        .gradient(Zero)
        .build()
        .await
        .unwrap();
    gas.start_recording(RecordingConfig::default()).unwrap();
    for _ in 0..24 {
        gas.step().await.unwrap();
    }
    gas
}

fn readout(
    gas: &AlgorithmicGas<f64>,
    id: u32,
    parameters: serde_json::Value,
) -> algorithmic_gas::physics::partvi::ExperimentResult {
    fields::analyze(
        &ExperimentRequest {
            experiment: id,
            parameters,
        },
        gas.recording(),
    )
    .unwrap_or_else(|e| panic!("archive experiment {id}: {e}"))
}
fn metric(result: &algorithmic_gas::physics::partvi::ExperimentResult, name: &str) -> f64 {
    result
        .metrics
        .iter()
        .find(|m| m.label == name)
        .unwrap()
        .value
        .unwrap()
}
#[test]
fn all_non_geometric_field_defaults_measure_the_executed_archive() {
    block_on(async {
        for seed in [0, 7, 516] {
            let gas = executed(seed, InnovationLaw::Gaussian).await;
            let bytes = gas.recording().unwrap().to_bytes().unwrap();
            for id in [
                41, 42, 43, 44, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65,
                66,
            ] {
                let result = readout(&gas, id, json!({}));
                assert_eq!(result.details["source"], "recorded_engine_fields", "{id}");
                assert_eq!(result.details["executed_steps"], 24, "{id}");
                assert!(!result.plots.is_empty(), "{id}");
                assert!(
                    result
                        .plots
                        .iter()
                        .flat_map(|p| &p.series)
                        .flat_map(|s| &s.points)
                        .flatten()
                        .all(|x| x.is_finite()),
                    "{id}"
                );
                assert_eq!(
                    serde_json::to_value(&result).unwrap(),
                    serde_json::to_value(readout(&gas, id, json!({}))).unwrap(),
                    "{id}"
                );
            }
            assert_eq!(
                bytes,
                gas.recording().unwrap().to_bytes().unwrap(),
                "measurement mutated archive"
            );
        }
    });
}
#[test]
fn independent_recorded_stage_and_graph_identities_hold() {
    block_on(async {
        let gas = executed(7, InnovationLaw::StandardizedUniform).await;
        for (id, label, tolerance) in [
            (41, "Log-volume telescope residual", 1e-10),
            (44, "Graph Euler identity residual", 0.),
            (46, "Virial product-rule residual", 1e-10),
            (49, "Parseval energy residual", 1e-10),
            (56, "Discrete cut first-variation residual", 0.),
            (58, "Source derivative residual", 1e-7),
            (60, "Max-flow min-cut residual", 0.),
            (64, "Information chain-rule residual", 1e-12),
        ] {
            let result = readout(&gas, id, json!({}));
            assert!(metric(&result, label) <= tolerance, "{id}: {result:?}");
        }
    });
}
#[test]
fn field_measurements_respond_to_run_and_analysis_controls() {
    block_on(async {
        let gas = executed(0, InnovationLaw::Gaussian).await;
        let other = executed(516, InnovationLaw::Gaussian).await;
        for id in [41, 47, 49, 50, 53, 57, 58, 61, 63, 65, 66] {
            assert_ne!(
                serde_json::to_value(readout(&gas, id, json!({}))).unwrap()["plots"],
                serde_json::to_value(readout(&other, id, json!({}))).unwrap()["plots"],
                "{id} ignored trajectory"
            );
        }
        assert_ne!(
            serde_json::to_value(readout(&gas, 47, json!({"wave_number":0.3}))).unwrap()["plots"],
            serde_json::to_value(readout(&gas, 47, json!({"wave_number":2.}))).unwrap()["plots"]
        );
        assert_ne!(
            serde_json::to_value(readout(&gas, 58, json!({"source":-1.}))).unwrap()["plots"],
            serde_json::to_value(readout(&gas, 58, json!({"source":1.}))).unwrap()["plots"]
        );
        assert_ne!(
            serde_json::to_value(readout(&gas, 61, json!({"bins":2}))).unwrap()["plots"],
            serde_json::to_value(readout(&gas, 61, json!({"bins":12}))).unwrap()["plots"]
        );
    });
}
#[test]
fn missing_archive_and_invalid_graph_terminals_error() {
    for id in 37..=66 {
        assert!(
            fields::analyze(
                &ExperimentRequest {
                    experiment: id,
                    parameters: json!({})
                },
                None
            )
            .is_err()
        );
    }
    block_on(async {
        let gas = executed(7, InnovationLaw::Gaussian).await;
        assert!(
            fields::analyze(
                &ExperimentRequest {
                    experiment: 60,
                    parameters: json!({"source_slot":1,"sink_slot":1})
                },
                gas.recording()
            )
            .is_err()
        );
    });
}
