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
    gas.step().await.unwrap();
    gas.step().await.unwrap();
    gas
}
#[test]
fn executed_heating_uses_source_shift_and_uniform_fourth_moment() {
    block_on(async {
        for law in [InnovationLaw::Gaussian, InnovationLaw::StandardizedUniform] {
            let mut sums = vec![];
            let mut predicted = 0.;
            for seed in 0..64 {
                let gas = executed(seed, law).await;
                let r = fields::analyze(
                    &ExperimentRequest {
                        experiment: 48,
                        parameters: json!({}),
                    },
                    gas.recording(),
                )
                .unwrap();
                sums.push(
                    r.metrics
                        .iter()
                        .find(|m| m.label == "Cumulative centered O-stage energy")
                        .unwrap()
                        .value
                        .unwrap(),
                );
                predicted += r
                    .metrics
                    .iter()
                    .find(|m| m.label == "Predictable cumulative energy variance")
                    .unwrap()
                    .value
                    .unwrap();
                let first = &r.details["steps"][0];
                let tracked = first["measured"]["predicted_energy_change"]
                    .as_f64()
                    .unwrap();
                let variance_mean = first["conditional_moments"]["energy_mean_delta"]
                    .as_f64()
                    .unwrap();
                assert!((tracked - variance_mean).abs() < 1e-12);
                assert!(
                    first["conditional_moments"]["energy_variance"]
                        .as_f64()
                        .unwrap()
                        > 0.
                );
            }
            let residual = sums.iter().sum::<f64>();
            assert!(
                residual.abs() < 5. * predicted.sqrt(),
                "{law:?}: sum={residual}, predictable variance={predicted}"
            );
            let measured_square = sums.iter().map(|v| v * v).sum::<f64>();
            eprintln!(
                "HEATING law={law:?} independent_runs=64 stages_per_run=2 centered_sum={residual} predictable_variance_sum={predicted} standardized_sum={} second_moment_ratio={}",
                residual / predicted.sqrt(),
                measured_square / predicted
            );
            assert!(
                (measured_square / predicted - 1.).abs() < 0.65,
                "{law:?}: observed/predicted second moment={}",
                measured_square / predicted
            );
        }
    })
}
#[test]
fn executed_kicks_and_weak_momentum_preserve_all_operator_transfers() {
    block_on(async {
        let gas = executed(9, InnovationLaw::Gaussian).await;
        let r = fields::analyze(
            &ExperimentRequest {
                experiment: 51,
                parameters: json!({}),
            },
            gas.recording(),
        )
        .unwrap();
        for row in r.details["weak_momentum"]["stages"].as_array().unwrap() {
            assert!(row["residual"].as_f64().unwrap() < 1e-12);
        }
        let kicks = r.details["kick_work"].as_array().unwrap();
        assert_eq!(kicks.len(), 2);
        for row in kicks {
            assert!(row["velocity_residual"].as_f64().unwrap() < 1e-12);
            assert!(
                (row["measured_energy_increment"].as_f64().unwrap()
                    - row["predicted_energy_increment"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
        }
        assert!(
            r.details["full_mechanical_report"]["kinetic_telescoping_residual"]
                .as_f64()
                .unwrap()
                .abs()
                < 1e-12
        );
    })
}
#[test]
fn archive_thermostat_rejects_an_unidentified_change_to_the_noise() {
    block_on(async {
        let gas = executed(4, InnovationLaw::Gaussian).await;
        let mut a = gas.recording().unwrap().clone();
        a.steps[0]
            .field_evaluations
            .iter_mut()
            .find(|f| f.stage == "O" && f.field == "executed_noise")
            .unwrap()
            .values[0] += 1.;
        assert!(
            fields::analyze(
                &ExperimentRequest {
                    experiment: 48,
                    parameters: json!({})
                },
                Some(&a)
            )
            .is_err()
        );
    })
}
#[test]
fn fastest_advertised_density_mode_preserves_positive_decay() {
    let r=fields::analyze(&ExperimentRequest{experiment:47,parameters:json!({"wave_number":6,"length":0.5,"diffusion":0.2,"rate":2.,"resolution":256})},None).unwrap();
    let points = &r.plots[0].series[0].points;
    assert!(points.iter().all(|p| p[1] >= 0. && p[1] <= 1.));
    assert!(points.windows(2).all(|p| p[1][1] <= p[0][1]));
    assert!(r.details["explicit_euler_multiplier"].as_f64().unwrap() >= 0.5 - 1e-12);
}

#[test]
fn recorded_selected_graph_is_separate_from_the_reconstructed_gaussian_graph() {
    block_on(async {
        let gas = executed(3, InnovationLaw::Gaussian).await;
        let a = gas.recording().unwrap();
        let r = fields::analyze(
            &ExperimentRequest {
                experiment: 52,
                parameters: json!({}),
            },
            Some(a),
        )
        .unwrap();
        let graph = algorithmic_gas::fractal_set::FractalSet::from_archive(a);
        use algorithmic_gas::fractal_set::EdgeKind;
        let kinds = [
            EdgeKind::IgDistance,
            EdgeKind::IgCloning,
            EdgeKind::HistoricalDistance,
            EdgeKind::HistoricalCloning,
        ];
        let rows = r.details["recorded_interaction_cut"]["channels"]
            .as_array()
            .unwrap();
        for (kind, row) in kinds.iter().zip(rows) {
            assert_eq!(
                row["observed_edges"].as_u64().unwrap() as usize,
                graph.edges.iter().filter(|e| e.kind == *kind).count()
            );
        }
        assert_eq!(r.details["gaussian_reference_graph"], true);
    })
}
