use algorithmic_gas::{
    boundary::BoundaryPolicy,
    domain::{OperatorFuture, RewardSource},
    donor::{CompanionBatch, DonorModule, OddPolicy, SamplingLaw, SourceRef},
    geometry::Kernel,
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry},
    physics::{
        partvi::ExperimentRequest,
        path_action::{analyze, likelihood, uniform_assignment_log_probability},
    },
    *,
};
use futures_lite::future::block_on;
use serde_json::json;
fn sources(n: usize) -> Vec<SourceRef> {
    (0..n)
        .map(|i| SourceRef {
            frame: 0,
            slot: i as u32,
            generation: 0,
            version: 0,
        })
        .collect()
}
fn batch(n: usize, k: usize, indices: Vec<u32>, mutual: bool) -> CompanionBatch {
    CompanionBatch {
        rows: n,
        count: k,
        valid: vec![true; n * k],
        indices,
        mutual,
    }
}
#[test]
fn uniform_independent_joint_assignments_normalize() {
    let module = DonorModule {
        kernel: Kernel::Uniform,
        ..Default::default()
    };
    let src = sources(3);
    let mut total = 0.;
    for a in [1, 2] {
        for b in [0, 2] {
            for c in [0, 1] {
                let b = batch(3, 1, vec![a, b, c], false);
                total += uniform_assignment_log_probability(&module, &b, &src, 0, &[true; 3])
                    .unwrap()
                    .exp();
            }
        }
    }
    assert!((total - 1.).abs() < 1e-12);
}
#[test]
fn without_replacement_preserves_order_and_normalization() {
    let module = DonorModule {
        kernel: Kernel::Uniform,
        count: 2,
        replacement: false,
        ..Default::default()
    };
    let src = sources(3);
    let mut total = 0.;
    for a in [false, true] {
        for b in [false, true] {
            for c in [false, true] {
                let mut rows = vec![];
                for (mut row, flip) in [(vec![1, 2], a), (vec![0, 2], b), (vec![0, 1], c)] {
                    if flip {
                        row.reverse();
                    }
                    rows.extend(row);
                }
                total += uniform_assignment_log_probability(
                    &module,
                    &batch(3, 2, rows, false),
                    &src,
                    0,
                    &[true; 3],
                )
                .unwrap()
                .exp();
            }
        }
    }
    assert!((total - 1.).abs() < 1e-12);
    assert!(
        uniform_assignment_log_probability(
            &module,
            &batch(3, 2, vec![1, 1, 0, 2, 0, 1], false),
            &src,
            0,
            &[true; 3]
        )
        .is_err()
    );
}
#[test]
fn matching_counts_shuffle_multiplicity_and_odd_fixed_points() {
    let module = DonorModule {
        kernel: Kernel::Uniform,
        law: SamplingLaw::FisherYates,
        ..Default::default()
    };
    let total = [vec![1, 0, 3, 2], vec![2, 3, 0, 1], vec![3, 2, 1, 0]]
        .into_iter()
        .map(|v| {
            uniform_assignment_log_probability(
                &module,
                &batch(4, 1, v, true),
                &sources(4),
                0,
                &[true; 4],
            )
            .unwrap()
            .exp()
        })
        .sum::<f64>();
    assert!((total - 1.).abs() < 1e-12);
    let total = [vec![0, 2, 1], vec![2, 1, 0], vec![1, 0, 2]]
        .into_iter()
        .map(|v| {
            uniform_assignment_log_probability(
                &module,
                &batch(3, 1, v, true),
                &sources(3),
                0,
                &[true; 3],
            )
            .unwrap()
            .exp()
        })
        .sum::<f64>();
    assert!((total - 1.).abs() < 1e-12);
    let unmatched = DonorModule {
        odd: OddPolicy::Unmatched,
        ..module
    };
    let mut b = batch(3, 1, vec![0, 2, 1], true);
    b.valid[0] = false;
    assert!(
        (uniform_assignment_log_probability(&unmatched, &b, &sources(3), 0, &[true; 3])
            .unwrap()
            .exp()
            - 1. / 3.)
            .abs()
            < 1e-12
    );
}
struct Reward;
impl RewardSource<f64> for Reward {
    fn id(&self) -> String {
        "deterministic-test-reward".into()
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
async fn archive(kernel: Kernel, amplitude: f64) -> RunArchive<f64> {
    archive_history(kernel, amplitude, 0).await
}
async fn archive_history(kernel: Kernel, amplitude: f64, history_window: usize) -> RunArchive<f64> {
    let module = DonorModule {
        kernel,
        history_window,
        ..Default::default()
    };
    let mut config = GasConfig {
        precision: Precision::F64,
        boundary: BoundaryPolicy::Unbounded,
        distance_donors: module.clone(),
        cloning_donors: module,
        ..Default::default()
    };
    config.kinetic.noise = Noise {
        innovation: InnovationLaw::Gaussian,
        geometry: NoiseGeometry::Isotropic {
            scale: FactorValues::Constant {
                values: vec![amplitude],
            },
        },
    };
    let p = Population::new(ObservationBatch::positions(
        TensorBatch::vectors(3, 1, vec![-1., 0.2, 1.]).unwrap(),
    ))
    .unwrap();
    let mut gas = GasBuilder::new(p, Reward)
        .config(config)
        .build()
        .await
        .unwrap();
    gas.start_recording(Default::default()).unwrap();
    for _ in 0..4 {
        gas.step().await.unwrap();
    }
    gas.recording().unwrap().clone()
}
#[test]
fn actual_archive_likelihood_replays_and_keeps_noise_determinant() {
    block_on(async {
        let a = archive(Kernel::Uniform, 0.7).await;
        let report = likelihood(&a).unwrap();
        assert!(report.complete);
        let mut expected = 0.;
        for s in &a.steps {
            expected -= 6. * 2_f64.ln();
            for c in &s.report.clone_plan.choices {
                let p = c.probability.unwrap();
                expected += if c.accepted { p.ln() } else { (-p).ln_1p() };
            }
            for noise in &s.noise {
                let raw = noise.raw_innovation.as_ref().unwrap();
                expected += -0.5 * raw.len() as f64 * std::f64::consts::TAU.ln()
                    - raw.len() as f64 * 0.7_f64.ln()
                    - 0.5 * raw.iter().map(|x| x * x).sum::<f64>();
            }
        }
        assert!((report.log_density.unwrap() - expected).abs() < 1e-11);
        let b = RunArchive::<f64>::from_bytes(&a.to_bytes().unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&report).unwrap(),
            serde_json::to_value(likelihood(&b).unwrap()).unwrap()
        );
        for experiment in [16, 17, 21] {
            let r = analyze(
                &ExperimentRequest {
                    experiment,
                    parameters: json!({}),
                },
                &a,
            )
            .unwrap();
            serde_json::to_vec(&r).unwrap();
            assert!(!r.plots.is_empty());
        }
    });
}
#[test]
fn gaussian_weighted_probability_uses_recorded_coordinates() {
    block_on(async {
        let a = archive(Kernel::Gaussian { width: 0.8 }, 0.5).await;
        let report = likelihood(&a).unwrap();
        assert!(report.complete, "{report:?}");
        let s = &a.steps[0];
        let x = [-1_f64, 0.2, 1.];
        let mut expected = 0.;
        for b in [&s.report.distance_companions, &s.report.cloning_companions] {
            for i in 0..3 {
                let j = b.indices[i] as usize;
                let numerator = (-(x[i] - x[j]).powi(2) / (2. * 0.8_f64.powi(2))).exp();
                let z = (0..3)
                    .filter(|&j| j != i)
                    .map(|j| (-(x[i] - x[j]).powi(2) / (2. * 0.8_f64.powi(2))).exp())
                    .sum::<f64>();
                expected += (numerator / z).ln();
            }
        }
        let measured = report.steps[0].components[..2]
            .iter()
            .map(|c| match c.likelihood {
                algorithmic_gas::physics::path_action::Likelihood::Available { log_density } => {
                    log_density
                }
                _ => panic!(),
            })
            .sum::<f64>();
        assert!((measured - expected).abs() < 1e-12);
    });
}
#[test]
fn singular_factors_retain_latent_law_and_unknown_providers_remain_unavailable() {
    block_on(async {
        let mut a = archive(Kernel::Uniform, 0.).await;
        let r = likelihood(&a).unwrap();
        assert!(r.complete);
        assert!(r.log_density.is_some());
        assert!(
            r.steps
                .iter()
                .flat_map(|s| &s.components)
                .any(|c| c.carrier == "standardized_innovation_lebesgue")
        );
        assert!(r.steps.iter().any(|s| s.available_log_density != 0.));
        a.providers
            .insert("operators".into(), "custom/random-law".into());
        let out = analyze(
            &ExperimentRequest {
                experiment: 17,
                parameters: json!({}),
            },
            &a,
        )
        .unwrap();
        assert!(
            out.metrics
                .iter()
                .any(|m| m.label == "Conditional primitive action" && m.value.is_none())
        );
        serde_json::to_vec(&out).unwrap();
    });
}
#[test]
fn empirical_partition_and_disintegration_use_own_normalization() {
    block_on(async {
        let a = archive(Kernel::Uniform, 0.7).await;
        let r = analyze(
            &ExperimentRequest {
                experiment: 16,
                parameters: json!({"angle":0.4}),
            },
            &a,
        )
        .unwrap();
        let residual = r
            .metrics
            .iter()
            .find(|m| m.label == "Source derivative residual")
            .unwrap()
            .value
            .unwrap();
        assert!(residual < 1e-9);
        let r = analyze(
            &ExperimentRequest {
                experiment: 21,
                parameters: json!({"bins":4}),
            },
            &a,
        )
        .unwrap();
        let bins = r.details["empirical_disintegration"]["occupied_bins"]
            .as_array()
            .unwrap();
        assert!(
            (bins
                .iter()
                .map(|b| b["joint"].as_f64().unwrap())
                .sum::<f64>()
                - 1.)
                .abs()
                < 1e-12
        );
        for b in bins {
            assert!(
                (b["joint_action"].as_f64().unwrap()
                    - b["marginal_action"].as_f64().unwrap()
                    - b["conditional_action"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
        }
    });
}

#[test]
fn legacy_permutations_and_death_fallback_have_different_laws() {
    let module = DonorModule {
        kernel: Kernel::Uniform,
        law: SamplingLaw::LegacyPermutation,
        ..Default::default()
    };
    let full = batch(3, 1, vec![0, 2, 1], false);
    assert!(
        (uniform_assignment_log_probability(&module, &full, &sources(3), 0, &[true; 3])
            .unwrap()
            .exp()
            - 1. / 6.)
            .abs()
            < 1e-12
    );
    let mut partial = batch(3, 1, vec![0, 0, 0], false);
    partial.valid[1] = false;
    let src = vec![sources(3)[0], sources(3)[2]];
    assert!(
        (uniform_assignment_log_probability(&module, &partial, &src, 0, &[true, false, true])
            .unwrap()
            .exp()
            - 1. / 4.)
            .abs()
            < 1e-12
    );
}
#[test]
fn weighted_historical_sources_use_retained_coordinates() {
    block_on(async {
        let a = archive_history(Kernel::Gaussian { width: 0.8 }, 0.5, 2).await;
        assert!(a.steps.iter().any(|s| {
            s.report
                .clone_plan
                .sources
                .iter()
                .any(|x| x.frame + 1 < s.report.step)
        }));
        let report = likelihood(&a).unwrap();
        assert!(report.complete, "{report:?}");
        let roundtrip = RunArchive::<f64>::from_bytes(&a.to_bytes().unwrap()).unwrap();
        assert_eq!(
            report.log_density,
            likelihood(&roundtrip).unwrap().log_density
        );
    });
}
