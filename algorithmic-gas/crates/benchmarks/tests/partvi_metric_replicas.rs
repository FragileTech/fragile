use algorithmic_gas::{
    BackendKind, Precision,
    boundary::BoundaryPolicy,
    geometry::Distance,
    kinetic::{KineticKind, KineticOperator},
    noise::Noise,
    partv_geometry::MetricPolicy,
    physics::partvi::{ExperimentRequest, analyze_archive},
    tracking::RecordingConfig,
};
use algorithmic_gas_benchmarks::{
    Benchmark, RunConfig, physics_metric::PhysicsMetricConfig, qft_experiments,
};
use futures_lite::future::block_on;
use serde_json::json;
fn config() -> RunConfig {
    let mut config = RunConfig {
        benchmark: Benchmark::Sphere,
        walkers: 8,
        dimensions: 3,
        physics_metric: Some(PhysicsMetricConfig {
            epsilon: 0.5,
            policy: MetricPolicy::Clipped,
            curvature: false,
            ..Default::default()
        }),
        ..Default::default()
    };
    config.gas.backend = BackendKind::Cpu;
    config.gas.precision = Precision::F64;
    config.gas.boundary = BoundaryPolicy::Unbounded;
    config.gas.distance_donors.distance = Distance::Euclidean {
        field: "positions".into(),
        scales: vec![],
        squared: false,
        periodic: None,
    };
    config.gas.kinetic = KineticOperator {
        integrator: KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 0.04,
            friction: 1.,
        },
        noise: Noise::default(),
    };
    config
}
#[test]
fn actual_metric_replicas_have_disjoint_keys_fixed_probes_and_unbiased_covariance() {
    block_on(async {
        let request = ExperimentRequest {
            experiment: 45,
            parameters: json!({"replicas":4,"horizon":1,"warmup":1,"readout":"fixed_probe"}),
        };
        let a = qft_experiments::run(&config(), &request).await.unwrap();
        let b = qft_experiments::run(&config(), &request).await.unwrap();
        assert_eq!(a.details, b.details);
        let report = &a.details["report"];
        let ca = report["calibration_keys"].as_array().unwrap();
        let va = report["validation_keys"].as_array().unwrap();
        assert!(ca.iter().all(|x| !va.contains(x)));
        assert_eq!(report["increment_covariance"].as_array().unwrap().len(), 36);
        assert_eq!(a.details["available_counts"], json!([4, 4]));
        for sample in a.details["samples"].as_array().unwrap() {
            assert_eq!(sample["context"]["query"], a.details["probe"]);
            assert_eq!(sample["metric"].as_array().unwrap().len(), 6);
        }
    });
}
#[test]
fn archived_three_dimensional_field_matches_executed_hessian_on_the_same_probe() {
    block_on(async {
        let cfg = config();
        let mut gas = cfg.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig {
            max_steps: 2,
            max_bytes: 32 * 1024 * 1024,
        })
        .unwrap();
        gas.step().await.unwrap();
        let archive = gas.recording().unwrap();
        let record = archive.steps.last().unwrap();
        let stage = record.stages.iter().find(|s| s.stage == "A1").unwrap();
        let query = &stage.fields["positions"].values[..3];
        let (jet, _) =
            algorithmic_gas::physics::fields::archive_fitness_jet(archive, 0, 0, query, 4).unwrap();
        let measured = record
            .field_evaluations
            .iter()
            .find(|f| f.field == "fitness_hessian")
            .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (jet.derivative(&[i, j]).unwrap() - measured.values[i * 3 + j]).abs() < 1e-9
                );
            }
        }
        let result = analyze_archive(
            &ExperimentRequest {
                experiment: 39,
                parameters: json!({}),
            },
            Some(archive),
        )
        .unwrap();
        assert!(result.details.get("archive_field").is_some());
        assert!(result.model.contains("Actual recorded"));
    });
}
#[test]
fn metric_runner_requires_real_three_dimensional_provider_and_completed_context() {
    block_on(async {
        let mut c = config();
        c.dimensions = 2;
        assert!(
            qft_experiments::run(
                &c,
                &ExperimentRequest {
                    experiment: 45,
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
                    experiment: 45,
                    parameters: json!({"warmup":0})
                }
            )
            .await
            .is_err()
        );
    });
}

#[test]
fn archived_local_fitness_reconstruction_differentiates_the_neighborhood_weights() {
    block_on(async {
        let mut cfg = config();
        let standardizer = algorithmic_gas::fitness::Standardizer::Local {
            sigma_min: 0.05,
            distance: Distance::Euclidean {
                field: "positions".into(),
                scales: vec![],
                squared: false,
                periodic: None,
            },
            kernel: algorithmic_gas::geometry::Kernel::Gaussian { width: 0.7 },
            include_self: true,
        };
        cfg.gas.fitness.reward_standardizer = standardizer.clone();
        cfg.gas.fitness.diversity_standardizer = standardizer;
        let mut gas = cfg.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig {
            max_steps: 1,
            max_bytes: 32 * 1024 * 1024,
        })
        .unwrap();
        gas.step().await.unwrap();
        let archive = gas.recording().unwrap();
        let record = archive.steps.last().unwrap();
        let stage = record.stages.iter().find(|s| s.stage == "A1").unwrap();
        let (jet, _) = algorithmic_gas::physics::fields::archive_fitness_jet(
            archive,
            0,
            0,
            &stage.fields["positions"].values[..3],
            4,
        )
        .unwrap();
        let measured = record
            .field_evaluations
            .iter()
            .find(|f| f.field == "fitness_hessian")
            .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (jet.derivative(&[i, j]).unwrap() - measured.values[i * 3 + j]).abs() < 1e-9
                );
            }
        }
    });
}

#[test]
fn material_metric_uses_final_positions_and_retains_component_cross_covariances() {
    block_on(async {
        for seed in [11, 29, 43] {
            let mut cfg = config();
            cfg.gas.seed = seed;
            let r = qft_experiments::run(
                &cfg,
                &ExperimentRequest {
                    experiment: 45,
                    parameters: json!({"replicas":4,"horizon":1,"warmup":1}),
                },
            )
            .await
            .unwrap();
            assert_eq!(r.details["readout"], "material");
            let samples = r.details["samples"].as_array().unwrap();
            assert!(
                samples
                    .iter()
                    .any(|s| s["future_probe"] != r.details["probe"])
            );
            assert!(samples.iter().any(|s| {
                s["field_clone_motion_increments"].as_array().unwrap()[12..]
                    .iter()
                    .any(|v| v.as_f64().unwrap().abs() > 1e-10)
            }));
            for s in samples {
                assert_eq!(s["context"]["query"], s["future_probe"]);
                for i in 0..6 {
                    let terms = s["field_clone_motion_increments"].as_array().unwrap();
                    let sum = terms[i].as_f64().unwrap()
                        + terms[i + 6].as_f64().unwrap()
                        + terms[i + 12].as_f64().unwrap();
                    let total = s["increment"][i].as_f64().unwrap();
                    assert!((sum - total).abs() < 1e-11 * (1. + total.abs()));
                }
            }
            eprintln!(
                "MATERIAL seed={seed} covariance_telescoping_residual={}",
                r.details["component_covariance_residual"]
            );
            let covariance = r.details["report"]["increment_covariance"]
                .as_array()
                .unwrap();
            for (sum, total) in r.details["component_covariance_sum"]
                .as_array()
                .unwrap()
                .iter()
                .zip(covariance)
            {
                let x = sum.as_f64().unwrap();
                let y = total.as_f64().unwrap();
                assert!((x - y).abs() < 1e-10 * (1. + x.abs() + y.abs()));
            }
        }
    });
}

#[test]
fn historical_distance_sources_match_archived_global_and_local_phase_space_jets() {
    block_on(async {
        for local in [false, true] {
            let mut cfg = config();
            cfg.gas.distance_donors.history_window = 2;
            cfg.gas.cloning_donors.history_window = 2;
            cfg.gas.distance_donors.distance = Distance::PhaseSpace {
                positions: "positions".into(),
                velocities: "velocities".into(),
                position_scale: 1.,
                velocity_scale: 1.,
                periodic: None,
                lambda: 0.7,
            };
            if local {
                // The configured engine supports historical cloning rescore for global
                // normalization; this case tests local distance memory with current cloning.
                cfg.gas.cloning_donors.history_window = 0;
                let s = algorithmic_gas::fitness::Standardizer::Local {
                    sigma_min: 0.05,
                    distance: cfg.gas.distance_donors.distance.clone(),
                    kernel: algorithmic_gas::geometry::Kernel::Gaussian { width: 0.7 },
                    include_self: true,
                };
                cfg.gas.fitness.reward_standardizer = s.clone();
                cfg.gas.fitness.diversity_standardizer = s;
            }
            let mut gas = cfg.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig {
                max_steps: 5,
                max_bytes: 64 * 1024 * 1024,
            })
            .unwrap();
            for _ in 0..5 {
                gas.step().await.unwrap();
            }
            let archive = gas.recording().unwrap();
            let mut historical = 0;
            let mut max_error = 0_f64;
            for (record_index, record) in archive.steps.iter().enumerate() {
                let stage = record.stages.iter().find(|s| s.stage == "A1").unwrap();
                let hessian = record
                    .field_evaluations
                    .iter()
                    .find(|f| f.field == "fitness_hessian")
                    .unwrap();
                for slot in 0..cfg.walkers {
                    let source = record.report.distance_sources
                        [record.report.distance_companions.indices[slot] as usize];
                    if source.frame >= record.report.step - 1 {
                        continue;
                    }
                    historical += 1;
                    let query = &stage.fields["positions"].values[slot * 3..slot * 3 + 3];
                    let (jet, context) = algorithmic_gas::physics::fields::archive_fitness_jet(
                        archive,
                        record_index,
                        slot,
                        query,
                        2,
                    )
                    .unwrap();
                    assert_eq!(context["source"], serde_json::to_value(source).unwrap());
                    for i in 0..3 {
                        for j in 0..3 {
                            max_error = max_error.max(
                                (jet.derivative(&[i, j]).unwrap()
                                    - hessian.values[slot * 9 + i * 3 + j])
                                    .abs(),
                            );
                        }
                    }
                }
            }
            eprintln!(
                "HISTORY local={local} historical_queries={historical} max_hessian_error={max_error}"
            );
            assert!(historical > 0);
            assert!(max_error < 1e-8);
        }
    });
}

#[test]
fn metric_zero_friction_is_defined_and_its_programmed_noise_is_zero() {
    block_on(async {
        let mut cfg = config();
        if let KineticKind::Baoab { friction, .. } = &mut cfg.gas.kinetic.integrator {
            *friction = 0.;
        }
        let mut gas = cfg.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let step = &gas.recording().unwrap().steps[0];
        let metric = step
            .field_evaluations
            .iter()
            .find(|f| f.field == "fitness_metric")
            .unwrap();
        assert!(metric.values.iter().any(|x| x.abs() > 0.));
        let noise = step.noise.iter().find(|n| n.substep == 2).unwrap();
        assert!(noise.factor.as_ref().unwrap().iter().all(|x| *x == 0.));
    })
}

#[test]
fn revival_metric_uses_the_exact_current_source_field_with_memory_enabled() {
    block_on(async {
        let mut cfg = config();
        cfg.gas.distance_donors.history_window = 2;
        cfg.gas.cloning_donors.history_window = 2;
        let mut gas = cfg.build::<f64>().await.unwrap();
        let mut checkpoint = gas.checkpoint();
        checkpoint.population.validity[0].terminated = true;
        gas.restore(checkpoint).unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let archive = gas.recording().unwrap();
        let record = &archive.steps[0];
        assert!(record.report.clone_plan.choices[0].revival);
        let query = &record
            .stages
            .iter()
            .find(|s| s.stage == "A1")
            .unwrap()
            .fields["positions"]
            .values[..3];
        let (jet, context) =
            algorithmic_gas::physics::fields::archive_fitness_jet(archive, 0, 0, query, 2).unwrap();
        assert_eq!(context["revival_donor_field_extension"], true);
        let measured = record
            .field_evaluations
            .iter()
            .find(|f| f.field == "fitness_hessian")
            .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (jet.derivative(&[i, j]).unwrap() - measured.values[i * 3 + j]).abs() < 1e-9
                );
            }
        }
    })
}

#[test]
fn absorbing_kills_and_revivals_retain_historical_distance_context() {
    block_on(async {
        let mut checked = 0;
        let mut historical = 0;
        let mut maximum = 0_f64;
        for seed in [3, 17, 37] {
            let mut cfg = config();
            cfg.gas.seed = seed;
            cfg.initial_lower = -0.4;
            cfg.initial_upper = 0.4;
            cfg.gas.distance_donors.history_window = 2;
            cfg.gas.cloning_donors.history_window = 2;
            cfg.physics_metric.as_mut().unwrap().temperature = 2.;
            cfg.gas.boundary = BoundaryPolicy::AbsorbingBox {
                field: "positions".into(),
                domain: algorithmic_gas::boundary::BoxDomain {
                    lower: vec![-0.8; 3],
                    upper: vec![0.8; 3],
                },
            };
            if let KineticKind::Baoab { dt, .. } = &mut cfg.gas.kinetic.integrator {
                *dt = 0.3;
            }
            let mut gas = cfg.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig {
                max_steps: 8,
                max_bytes: 64 * 1024 * 1024,
            })
            .unwrap();
            for _ in 0..8 {
                match gas.step().await {
                    Ok(_) => {}
                    Err(algorithmic_gas::GasError::Extinction) => break,
                    Err(e) => panic!("{e}"),
                }
            }
            let archive = gas.recording().unwrap();
            for (index, record) in archive.steps.iter().enumerate() {
                let stage = record.stages.iter().find(|s| s.stage == "A1").unwrap();
                let hessian = record
                    .field_evaluations
                    .iter()
                    .find(|f| f.field == "fitness_hessian")
                    .unwrap();
                for slot in 0..cfg.walkers {
                    let choice = &record.report.clone_plan.choices[slot];
                    if !choice.revival || !stage.validity[slot].eligible(false) {
                        continue;
                    }
                    let source =
                        record.report.clone_plan.sources[choice.donors[0].pool_index as usize];
                    assert_eq!(source.frame, record.report.step - 1);
                    let donor_slot = source.slot as usize;
                    let distance_source = record.report.distance_sources
                        [record.report.distance_companions.indices[donor_slot] as usize];
                    historical += usize::from(distance_source.frame < record.report.step - 1);
                    let (jet, context) = algorithmic_gas::physics::fields::archive_fitness_jet(
                        archive,
                        index,
                        slot,
                        &stage.fields["positions"].values[slot * 3..slot * 3 + 3],
                        2,
                    )
                    .unwrap();
                    assert_eq!(context["target_slot"], donor_slot);
                    for i in 0..3 {
                        for j in 0..3 {
                            maximum = maximum.max(
                                (jet.derivative(&[i, j]).unwrap()
                                    - hessian.values[slot * 9 + i * 3 + j])
                                    .abs(),
                            );
                        }
                    }
                    checked += 1;
                }
            }
        }
        eprintln!(
            "KILLED_MEMORY checked_revivals={checked} historical_distance_sources={historical} maximum_hessian_error={maximum}"
        );
        assert!(checked > 0);
        assert!(historical > 0);
        assert!(maximum < 1e-8);
    })
}

#[test]
fn metric_replicas_resolve_sources_retained_before_continuation_recording() {
    block_on(async {
        let mut cfg = config();
        cfg.gas.distance_donors.history_window = 2;
        cfg.gas.cloning_donors.history_window = 2;
        let r = qft_experiments::run(
            &cfg,
            &ExperimentRequest {
                experiment: 45,
                parameters: json!({"replicas":4,"warmup":2,"horizon":2}),
            },
        )
        .await
        .unwrap();
        assert_eq!(r.details["historical_frames"], 2);
        assert_eq!(r.details["available_counts"], json!([4, 4]));
        assert!(
            r.details["samples"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["context"]["source"]["frame"].as_u64().unwrap()
                    < s["context"]["step"].as_u64().unwrap() - 1)
        );
    })
}
