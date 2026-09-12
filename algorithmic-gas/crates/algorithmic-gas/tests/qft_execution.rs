use algorithmic_gas::{
    domain::{GradientProvider, OperatorFuture, RewardSource},
    kinetic::{KineticKind, KineticOperator, QftExecutionConfig, ViscousForceConfig},
    noise::{FactorValues, InnovationLaw, InnovationShift, Noise, NoiseGeometry},
    random::Stream,
    tracking::RecordingConfig,
    *,
};
use futures_lite::future::block_on;
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
async fn run(qft: QftExecutionConfig, amplitude: f64) -> AlgorithmicGas<f64> {
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(2, 3, vec![0.; 6]).unwrap());
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(2, 3, vec![1., 0., 0., -1., 0., 0.]).unwrap(),
    );
    let config = GasConfig {
        precision: Precision::F64,
        qft,
        kinetic: KineticOperator {
            integrator: KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.1,
                friction: 0.,
            },
            noise: Noise {
                innovation: InnovationLaw::Gaussian,
                geometry: NoiseGeometry::Isotropic {
                    scale: FactorValues::Constant {
                        values: vec![amplitude],
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
    gas
}
#[test]
fn zero_viscosity_preserves_seeded_execution_exactly() {
    block_on(async {
        let a = run(Default::default(), 0.7).await;
        let b = run(
            QftExecutionConfig {
                viscosity: Some(ViscousForceConfig {
                    coefficient: 0.,
                    bandwidth: 1.,
                    row_normalized: false,
                }),
                ..Default::default()
            },
            0.7,
        )
        .await;
        assert_eq!(a.population(), b.population());
        assert_eq!(a.recording().unwrap().steps, b.recording().unwrap().steps);
    });
}
#[test]
fn both_kicks_use_recorded_forces_and_symmetric_damping_preserves_momentum() {
    block_on(async {
        let gas = run(
            QftExecutionConfig {
                viscosity: Some(ViscousForceConfig {
                    coefficient: 2.,
                    bandwidth: 1.,
                    row_normalized: false,
                }),
                ..Default::default()
            },
            0.,
        )
        .await;
        let record = &gas.recording().unwrap().steps[0];
        for stage in ["B1", "B2"] {
            let input = record
                .stages
                .iter()
                .find(|s| s.stage == format!("{stage}_input"))
                .unwrap();
            let output = record
                .stages
                .iter()
                .find(|s| s.stage == format!("{stage}_before_boundary"))
                .unwrap();
            let force = record
                .field_evaluations
                .iter()
                .find(|f| f.stage == stage && f.field == "total_force")
                .unwrap();
            assert_eq!(input.version, force.version);
            for j in 0..6 {
                assert!(
                    (output.fields["velocities"].values[j]
                        - input.fields["velocities"].values[j]
                        - 0.05 * force.values[j])
                        .abs()
                        < 1e-14
                );
            }
            for j in 0..3 {
                assert!((force.values[j] + force.values[3 + j]).abs() < 1e-14);
            }
            assert_eq!(
                record
                    .influences
                    .iter()
                    .filter(|i| i.stage == stage && i.field == "viscous_force")
                    .count(),
                2
            );
        }
        let v = gas.population().observations.field("velocities").unwrap();
        assert!(v.values()[0] > 0. && v.values()[0] < 1.);
        let budgets = record.mechanical_budgets("velocities", 1., false).unwrap();
        assert!(
            (budgets
                .iter()
                .map(|b| b.energy_after - b.energy_before)
                .sum::<f64>()
                - (v.values()[0].powi(2) - 1.))
                .abs()
                < 1e-14
        );
        for b in budgets {
            for j in 0..3 {
                assert!(
                    (b.momentum_after[j]
                        - b.momentum_before[j]
                        - b.common_momentum_change[j]
                        - b.eligibility_momentum_change[j])
                        .abs()
                        < 1e-14
                );
            }
        }
    });
}
#[test]
fn addressed_source_changes_actual_outcome_and_records_conditional_law() {
    block_on(async {
        let base = run(Default::default(), 0.7).await;
        let shifted = run(
            QftExecutionConfig {
                innovation_shifts: vec![InnovationShift {
                    step: 1,
                    stream: Stream::Kinetic,
                    substep: 2,
                    walker: 0,
                    coordinate: 1,
                    shift: 0.25,
                }],
                ..Default::default()
            },
            0.7,
        )
        .await;
        let record = &shifted.recording().unwrap().steps[0];
        let baseline = &base.recording().unwrap().steps[0];
        let a = base.population().observations.field("velocities").unwrap();
        let b = shifted
            .population()
            .observations
            .field("velocities")
            .unwrap();
        assert!((b.values()[1] - a.values()[1] - 0.1_f64.sqrt() * 0.7 * 0.25).abs() < 1e-14);
        assert_eq!(a.values()[3..], b.values()[3..]);
        let noise = record
            .noise
            .iter()
            .find(|s| s.substep == 2 && s.raw_innovation.is_some())
            .unwrap();
        let original = baseline
            .noise
            .iter()
            .find(|s| s.substep == 2 && s.raw_innovation.is_some())
            .unwrap();
        assert_eq!(noise.applied_source_shifts.len(), 1);
        assert!(
            (noise
                .conditional_gaussian_log_density(&[true, true])
                .unwrap()
                - original
                    .conditional_gaussian_log_density(&[true, true])
                    .unwrap())
            .abs()
                < 1e-14
        );
        let moments = record
            .thermostat_moments("velocities", 1., 0.1, 0., false)
            .unwrap();
        assert!((moments.predicted_momentum_change[1] - 0.1_f64.sqrt() * 0.7 * 0.25).abs() < 1e-14);
        assert!((moments.momentum_covariance[0] - 2. * 0.1 * 0.7_f64.powi(2)).abs() < 1e-14);
    });
}
#[test]
fn unsupported_and_singular_noise_densities_are_explicit() {
    block_on(async {
        let gas = run(Default::default(), 0.).await;
        let record = &gas.recording().unwrap().steps[0];
        let noise = record
            .noise
            .iter()
            .find(|s| s.substep == 2 && s.geometry.is_some())
            .unwrap();
        assert!(matches!(
            noise.conditional_gaussian_log_density(&[true, true]),
            Err(GasError::Capability(_))
        ));
        let mut unknown = noise.clone();
        unknown.innovation_law = None;
        assert!(matches!(
            unknown.conditional_gaussian_log_density(&[true, true]),
            Err(GasError::Capability(_))
        ));
    });
}

#[test]
fn gaussian_density_includes_full_factor_determinant_and_rejects_low_rank() {
    let mut sample = algorithmic_gas::tracking::NoiseSnapshot {
        stage: "raw_noise".into(),
        step: 1,
        stream: Stream::Kinetic,
        substep: 2,
        rows: 1,
        dimension: 2,
        sample: vec![0.4, -1.],
        raw_innovation: Some(vec![0.2, -0.4]),
        factor: Some(vec![2., 0., 1., 3.]),
        geometry: Some(NoiseGeometry::Full {
            factor: FactorValues::Constant {
                values: vec![2., 0., 1., 3.],
            },
        }),
        innovation_law: Some(InnovationLaw::Gaussian),
        applied_source_shifts: vec![],
    };
    let prediction = -std::f64::consts::TAU.ln() - 6_f64.ln() - 0.1;
    assert!((sample.conditional_gaussian_log_density(&[true]).unwrap() - prediction).abs() < 1e-14);
    sample.geometry = Some(NoiseGeometry::LowRank {
        rank: 1,
        factor: FactorValues::Constant {
            values: vec![1., 0.],
        },
    });
    sample.factor = Some(vec![1., 0.]);
    sample.raw_innovation = Some(vec![0.4]);
    assert!(matches!(
        sample.conditional_gaussian_log_density(&[true]),
        Err(GasError::Capability(_))
    ));
}
