use algorithmic_gas::{
    boundary::BoundaryPolicy,
    kinetic::KineticKind,
    noise::{FactorValues, InnovationLaw, NoiseGeometry},
    physics::partvi::{self, ExperimentRequest},
    tracking::RecordingConfig,
};
use algorithmic_gas_benchmarks::{RunConfig, qft_experiments};
use futures_lite::future::block_on;
use serde_json::json;

fn config(seed: u64) -> RunConfig {
    let mut c = RunConfig {
        walkers: 8,
        dimensions: 3,
        ..Default::default()
    };
    c.gas.seed = seed;
    c.gas.precision = algorithmic_gas::Precision::F64;
    c.gas.qft.viscosity = Some(algorithmic_gas::kinetic::ViscousForceConfig {
        coefficient: 0.15,
        bandwidth: 1.,
        row_normalized: false,
    });
    c.gas.boundary = BoundaryPolicy::Unbounded;
    c.gas.distance_donors.history_window = 2;
    c.gas.cloning_donors.history_window = 2;
    c.gas.kinetic.integrator = KineticKind::Baoab {
        positions: "positions".into(),
        velocities: "velocities".into(),
        dt: 0.04,
        friction: 0.7,
    };
    c.gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
        scale: FactorValues::Constant { values: vec![0.8] },
    };
    c
}
#[test]
fn gaussian_source_identity_agrees_across_independent_engine_seed_groups() {
    block_on(async {
        let mut residual = 0.;
        let mut variance = 0.;
        let mut samples = 0;
        for seed in [31, 407, 9021] {
            let r = qft_experiments::run(
                &config(seed),
                &ExperimentRequest {
                    experiment: 19,
                    parameters: json!({"replicas":96,"horizon":3,"warmup":3,"theta":0.3}),
                },
            )
            .await
            .unwrap();
            let metric = |label: &str| {
                r.metrics
                    .iter()
                    .find(|m| m.label == label)
                    .unwrap()
                    .value
                    .unwrap()
            };
            residual += metric("Direct minus reweighted");
            variance += metric("Independent comparison standard error").powi(2);
            for row in r.details["samples"].as_array().unwrap() {
                assert_ne!(row["weight_seed"], row["direct_seed"]);
                let xi = row["innovation"].as_f64().unwrap();
                let expected = (0.3 * xi - 0.045).exp();
                assert!((row["weight"].as_f64().unwrap() - expected).abs() < 1e-12);
                samples += 1;
            }
        }
        eprintln!(
            "source identity: residual sum={residual}, independent SE={}, samples={samples}",
            variance.sqrt()
        );
        assert!(
            residual.abs() < 4.5 * variance.sqrt(),
            "independent engine source identity fails"
        );
    });
}
#[test]
fn actual_noether_thermostat_oracle_predicts_independent_replica_fluctuations() {
    block_on(async {
        let mut momentum = 0.;
        let mut energy = 0.;
        let mut momentum_var = 0.;
        let mut energy_var = 0.;
        let mut second_momentum = 0.;
        let mut expected_second = 0.;
        let mut all = 0;
        for seed in [71, 809, 1217] {
            let r = qft_experiments::run(
                &config(seed),
                &ExperimentRequest {
                    experiment: 22,
                    parameters: json!({"replicas":64,"horizon":3,"warmup":3}),
                },
            )
            .await
            .unwrap();
            let p = &r.details["analytic_o_prediction"];
            assert_eq!(p["status"], "available");
            momentum += p["momentum_mean"].as_f64().unwrap();
            energy += p["energy_mean"].as_f64().unwrap();
            momentum_var += p["momentum_oracle_standard_error"]
                .as_f64()
                .unwrap()
                .powi(2);
            energy_var += p["energy_oracle_standard_error"].as_f64().unwrap().powi(2);
            for sample in p["samples"].as_array().unwrap() {
                assert_eq!(sample["covered_o_steps"], 3);
                second_momentum += sample["momentum_martingale"].as_f64().unwrap().powi(2);
                expected_second += sample["momentum_predictable_variation"].as_f64().unwrap();
                all += 1;
            }
        }
        eprintln!(
            "Noether O oracle: mean momentum sum={momentum}, SE={}; mean energy sum={energy}, SE={}; martingale variance ratio={}, replicas={all}",
            momentum_var.sqrt(),
            energy_var.sqrt(),
            second_momentum / expected_second
        );
        assert!(momentum.abs() < 4.5 * momentum_var.sqrt());
        assert!(energy.abs() < 4.5 * energy_var.sqrt());
        assert!((second_momentum / expected_second - 1.).abs() < 0.3);
    });
}
#[test]
fn recorded_twistors_retain_historical_sources_and_frame_spectra_match_direct_sums() {
    block_on(async {
        let mut gas = config(516).build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig {
            max_steps: 100,
            max_bytes: 128 * 1024 * 1024,
        })
        .unwrap();
        for _ in 0..96 {
            gas.step().await.unwrap();
        }
        let a = gas.recording().unwrap();
        let r = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 34,
                parameters: json!({}),
            },
            Some(a),
        )
        .unwrap();
        let history_count = r.details["donor_sources"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|v| v["donors"].as_array().into_iter().flatten())
            .filter(|d| d["age_steps"].as_u64().unwrap_or(0) > 0)
            .count();
        eprintln!(
            "twistor provenance: historical donors={history_count}, valid readouts={}",
            r.details["readouts"].as_array().unwrap().len()
        );
        assert!(!r.details["readouts"].as_array().unwrap().is_empty());
        assert!(
            r.details["donor_sources"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["donors"]
                    .as_array()
                    .is_some_and(|ds| ds.iter().any(|d| d["age_steps"].as_u64().unwrap_or(0) > 0)))
        );
        let spectral = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 36,
                parameters: json!({"channels":1,"max_lag":6}),
            },
            Some(a),
        )
        .unwrap();
        assert_eq!(spectral.details["aggregation"], "frame_mean");
        assert_eq!(spectral.details["status"], "available");
        let cut = (96_f64 * 0.6).floor() as usize;
        let means: Vec<_> = a.steps[..cut]
            .iter()
            .map(|step| {
                let s = step.stages.iter().find(|s| s.stage == "pre_clone").unwrap();
                let x = &s.fields["positions"].values;
                let v = &s.fields["velocities"].values;
                (
                    s.validity
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.eligible(false))
                        .map(|(i, _)| x[3 * i])
                        .sum::<f64>()
                        / 8.,
                    s.validity
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| a.eligible(false))
                        .map(|(i, _)| v[3 * i])
                        .sum::<f64>()
                        / 8.,
                )
            })
            .collect();
        let mx = means.iter().map(|p| p.0).sum::<f64>() / cut as f64;
        let mv = means.iter().map(|p| p.1).sum::<f64>() / cut as f64;
        let c0 = means
            .iter()
            .map(|p| (p.0 - mx).powi(2) + (p.1 - mv).powi(2))
            .sum::<f64>()
            / cut as f64;
        let measured = spectral.details["complex_correlations"][0]["real"][0]
            .as_f64()
            .unwrap();
        eprintln!(
            "frame spectral Gram: direct={c0}, measured={measured}, fits={}",
            spectral.details["fits"]
        );
        assert!((c0 - measured).abs() < 1e-11 * (1. + c0));
        assert_eq!(spectral.details["lag_counts"][0]["training_pairs"], cut);
        assert!(
            spectral.details["fits"]
                .as_array()
                .unwrap()
                .iter()
                .all(|f| f["mass"].is_null())
        );
        let pairs = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 36,
                parameters: json!({"channels":1,"max_lag":6,"aggregation":"source_pairs"}),
            },
            Some(a),
        )
        .unwrap();
        assert!(
            pairs.details["lag_counts"][0]["training_pairs"]
                .as_u64()
                .unwrap()
                > cut as u64
        );
    });
}
#[test]
fn low_rank_uniform_kinetic_moments_use_the_actual_innovation_law() {
    block_on(async {
        let mut c = config(805);
        c.gas.kinetic.noise.innovation = InnovationLaw::StandardizedUniform;
        c.gas.kinetic.noise.geometry = NoiseGeometry::LowRank {
            rank: 1,
            factor: FactorValues::Constant {
                values: vec![1., 2., 0.5],
            },
        };
        let mut gas = c.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig {
            max_steps: 100,
            max_bytes: 128 * 1024 * 1024,
        })
        .unwrap();
        for _ in 0..96 {
            gas.step().await.unwrap();
        }
        let r = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 18,
                parameters: json!({}),
            },
            gas.recording(),
        )
        .unwrap();
        assert_eq!(r.details["status"], "available");
        let likelihood =
            algorithmic_gas::physics::path_action::likelihood(gas.recording().unwrap()).unwrap();
        assert!(likelihood.complete);
        let latent_expected = -8. * (2. * 3_f64.sqrt()).ln();
        for step in &likelihood.steps {
            let latent = step
                .components
                .iter()
                .find(|c| c.carrier == "standardized_innovation_lebesgue")
                .unwrap();
            match latent.likelihood {
                algorithmic_gas::physics::path_action::Likelihood::Available { log_density } => {
                    assert!((log_density - latent_expected).abs() < 1e-12)
                }
                _ => panic!("missing uniform latent density"),
            }
        }
        let metric = |label: &str| {
            r.metrics
                .iter()
                .find(|m| m.label == label)
                .unwrap()
                .value
                .unwrap()
        };
        let second = (1. + 4. + 0.25) / 3.;
        let fourth = 1.8 * (1. + 16. + 0.0625) / 3.;
        assert!((metric("Recorded predicted second moment") - second).abs() < 1e-12);
        assert!((metric("Recorded predicted fourth moment") - fourth).abs() < 1e-12);
        eprintln!(
            "low-rank uniform O: second={} vs {second}; fourth={} vs {fourth}",
            metric("Recorded mean squared factor innovation"),
            metric("Recorded fourth moment")
        );
        assert!((metric("Recorded fourth moment") / fourth - 1.).abs() < 0.25);
    });
}

#[test]
fn gaussian_source_reweighting_retains_killing_and_revival_in_the_execution_map() {
    block_on(async {
        let mut residual = 0.;
        let mut variance = 0.;
        let mut killed = 0;
        for seed in [159, 9183] {
            let mut c = config(seed);
            c.walkers = 4;
            c.initial_lower = -0.03;
            c.initial_upper = 0.03;
            c.gas.boundary = BoundaryPolicy::AbsorbingBox {
                field: "positions".into(),
                domain: algorithmic_gas::boundary::BoxDomain {
                    lower: vec![-0.06; 3],
                    upper: vec![0.06; 3],
                },
            };
            c.gas.kinetic.integrator = KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.1,
                friction: 0.7,
            };
            c.gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
                scale: FactorValues::Constant { values: vec![2.] },
            };
            let r=qft_experiments::run(&c,&ExperimentRequest{experiment:19,parameters:json!({"replicas":96,"horizon":4,"warmup":0,"theta":0.6,"observable":"alive_fraction"})}).await.unwrap();
            let metric = |label: &str| {
                r.metrics
                    .iter()
                    .find(|m| m.label == label)
                    .unwrap()
                    .value
                    .unwrap()
            };
            residual += metric("Direct minus reweighted");
            variance += metric("Independent comparison standard error").powi(2);
            killed += 96 - r.details["survived"][2].as_u64().unwrap();
        }
        eprintln!(
            "killed source identity: residual sum={residual}, independent SE={}, extinct direct replicas={killed}/192",
            variance.sqrt()
        );
        assert!(killed > 0);
        assert!(residual.abs() < 4.5 * variance.sqrt() + 1e-12);
    });
}

#[test]
fn source_shift_changes_second_moment_without_changing_factor_covariance() {
    block_on(async {
        let mut c = config(92);
        c.gas.kinetic.noise.geometry = NoiseGeometry::LowRank {
            rank: 1,
            factor: FactorValues::Constant {
                values: vec![1., 2., 0.5],
            },
        };
        c.gas.qft.innovation_shifts = vec![algorithmic_gas::noise::InnovationShift {
            step: 1,
            stream: algorithmic_gas::random::Stream::Kinetic,
            substep: 2,
            walker: 0,
            coordinate: 0,
            shift: 2.,
        }];
        let mut gas = c.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig {
            max_steps: 4,
            max_bytes: 16 * 1024 * 1024,
        })
        .unwrap();
        for _ in 0..3 {
            gas.step().await.unwrap();
        }
        let r = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 18,
                parameters: json!({}),
            },
            gas.recording(),
        )
        .unwrap();
        let value = |label: &str| {
            r.metrics
                .iter()
                .find(|m| m.label == label)
                .unwrap()
                .value
                .unwrap()
        };
        assert!((value("Recorded mean factor covariance diagonal") - 1.75).abs() < 1e-12);
        assert!(
            r.metrics
                .iter()
                .any(|m| m.label.contains("second") && m.value.is_some_and(|v| v > 1.75))
        );
    });
}

#[test]
fn extinction_before_thermostat_is_not_reported_as_a_validated_zero_residual() {
    block_on(async {
        let mut c = config(4);
        c.initial_lower = 0.8;
        c.initial_upper = 0.9;
        c.gas.kinetic.integrator = KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 4.,
            friction: 0.7,
        };
        c.gas.boundary = BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: algorithmic_gas::boundary::BoxDomain {
                lower: vec![-1.; 3],
                upper: vec![1.; 3],
            },
        };
        let r = qft_experiments::run(
            &c,
            &ExperimentRequest {
                experiment: 22,
                parameters: json!({"warmup":0,"horizon":1,"replicas":4}),
            },
        )
        .await
        .unwrap();
        assert_eq!(r.details["analytic_o_prediction"]["status"], "unexercised");
        assert_eq!(r.details["analytic_o_prediction"]["covered_o_stages"], 0);
        assert!(
            !r.metrics
                .iter()
                .any(|m| m.label == "Analytic O momentum martingale mean")
        );
    });
}
