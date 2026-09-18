use algorithmic_gas::{
    RecordingConfig,
    boundary::{BoundaryPolicy, BoxDomain},
    noise::{FactorValues, InnovationLaw, InnovationShift, NoiseGeometry},
    physics::field_evolution::{WeakFieldObservable, clone_field_balance, weak_field_balance},
    random::Stream,
};
use algorithmic_gas_benchmarks::lecture;
use serde_json::json;

fn observables() -> Vec<WeakFieldObservable> {
    let k = vec![0.7, -0.4, 0.2];
    vec![
        WeakFieldObservable::PhaseSpaceCharacteristic {
            k: k.clone(),
            l: vec![4., 2., -3.],
        },
        WeakFieldObservable::Density { k: k.clone() },
        WeakFieldObservable::Momentum {
            k: k.clone(),
            component: 1,
        },
        WeakFieldObservable::Stress {
            k: k.clone(),
            a: 0,
            b: 1,
        },
        WeakFieldObservable::Stress {
            k: k.clone(),
            a: 0,
            b: 0,
        },
        WeakFieldObservable::KineticEnergy { k },
    ]
}
#[test]
fn analytic_field_predictions_match_full_interacting_runs_for_both_innovation_laws() {
    futures_lite::future::block_on(async {
        for law in [InnovationLaw::Gaussian, InnovationLaw::StandardizedUniform] {
            let observables = observables();
            let mut sums = vec![[0.; 2]; observables.len()];
            let mut variances = vec![[0.; 2]; observables.len()];
            let mut realized_squares = vec![[0.; 2]; observables.len()];
            let mut nonzero_clone = false;
            let mut clone_sum = [0.; 2];
            let mut clone_variance = [0.; 2];
            let mut clone_squares = [0.; 2];
            let mut cloned_history = 0;
            let mut historical_donors = false;
            for seed in 0..48 {
                let mut config = lecture::gas_config("VI-51", &json!({"walkers": 12,"engine_dt":0.12,"engine_memory":3,"engine_viscosity":0.7}), seed).unwrap();
                config.physics_metric = None;
                config.gas.kinetic.noise.innovation = law;
                config.gas.kinetic.noise.geometry = NoiseGeometry::Full {
                    factor: FactorValues::Constant {
                        values: vec![0.8, 0.2, 0., 0.1, 0.7, 0.1, 0., 0.2, 0.6],
                    },
                };
                config.gas.qft.innovation_shifts = (0..8)
                    .map(|step| InnovationShift {
                        step,
                        stream: Stream::Kinetic,
                        substep: 2,
                        walker: 0,
                        coordinate: 1,
                        shift: 0.4,
                    })
                    .collect();
                config.gas.qft.viscosity.as_mut().unwrap().row_normalized = seed % 2 == 0;
                let mut gas = config.build::<f64>().await.unwrap();
                gas.start_recording(RecordingConfig {
                    max_steps: 8,
                    ..Default::default()
                })
                .unwrap();
                for _ in 0..8 {
                    gas.step().await.unwrap();
                }
                for step in &gas.recording().unwrap().steps {
                    let clone =
                        clone_field_balance(gas.recording().unwrap(), step, &observables[0])
                            .unwrap();
                    cloned_history += clone.accepted_historical_sources;
                    for a in 0..2 {
                        clone_sum[a] += clone.martingale_increment[a];
                        clone_variance[a] += clone.martingale_covariance[a * 2 + a];
                        clone_squares[a] += clone.martingale_increment[a].powi(2);
                    }
                    assert!(clone.literal_copy_residual.iter().all(|r| r.abs() < 2e-13));
                    historical_donors |= step
                        .report
                        .clone_plan
                        .sources
                        .iter()
                        .any(|source| source.frame < step.report.step - 1);
                    for (index, observable) in observables.iter().enumerate() {
                        let result = weak_field_balance(&config.gas, step, observable).unwrap();
                        assert!(
                            result
                                .field_equation_residual
                                .iter()
                                .all(|v| v.abs() < 2e-13)
                        );
                        assert_eq!(result.slots, 12);
                        assert_eq!(
                            result
                                .stage_sources
                                .iter()
                                .filter(|s| s.deterministic_law_residual.is_some())
                                .count(),
                            4
                        );
                        for source in &result.stage_sources {
                            if let Some(residual) = source.deterministic_law_residual {
                                assert!(
                                    residual.iter().all(|v| v.abs() < 2e-13),
                                    "{} -> {} {:?}",
                                    source.from,
                                    source.to,
                                    residual
                                );
                            }
                            if source.kind == "clone_replacement"
                                && source.increment.iter().any(|v| v.abs() > 1e-6)
                            {
                                nonzero_clone = true;
                            }
                        }
                        for a in 0..2 {
                            sums[index][a] += result.martingale_increment[a];
                            variances[index][a] += result.martingale_covariance[a * 2 + a];
                            realized_squares[index][a] += result.martingale_increment[a].powi(2);
                        }
                    }
                }
            }
            assert!(historical_donors, "must exercise actual donor memory");
            assert!(cloned_history > 0);
            for a in 0..2 {
                let z = clone_sum[a] / clone_variance[a].sqrt();
                let ratio = clone_squares[a] / clone_variance[a];
                println!(
                    "{law:?} clone channel={a} z={z:.4} variance_ratio={ratio:.4} historical_copies={cloned_history}"
                );
                assert!((0.5..1.5).contains(&ratio));
                assert!(z.abs() < 4.5);
            }
            assert!(nonzero_clone, "must retain the actual cloning source");
            for index in 0..observables.len() {
                for a in 0..2 {
                    let variance = variances[index][a];
                    if variance < 1e-15 {
                        assert!(sums[index][a].abs() < 1e-12);
                        continue;
                    }
                    let z = sums[index][a] / variance.sqrt();
                    let ratio = realized_squares[index][a] / variance;
                    println!(
                        "{law:?} {:?} channel={a} martingale_z={z:.4} variance_ratio={ratio:.4}",
                        observables[index]
                    );
                    assert!(z.abs() < 4.5, "martingale drift {z}");
                    assert!(
                        (0.65..1.4).contains(&ratio),
                        "predicted quadratic variation differs: {ratio}"
                    );
                }
            }
        }
    });
}
#[test]
fn periodic_boundary_and_cloning_sources_are_retained_in_the_field_equation() {
    futures_lite::future::block_on(async {
        let mut config =
            lecture::gas_config("VI-51", &json!({"walkers":16,"engine_dt":0.5}), 7).unwrap();
        config.physics_metric = None;
        config.gas.boundary = BoundaryPolicy::PeriodicBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-0.3; 3],
                upper: vec![0.3; 3],
            },
        };
        for module in [
            &mut config.gas.distance_donors,
            &mut config.gas.cloning_donors,
        ] {
            match &mut module.distance {
                algorithmic_gas::geometry::Distance::Euclidean { periodic, .. }
                | algorithmic_gas::geometry::Distance::PhaseSpace { periodic, .. } => {
                    *periodic = Some(BoxDomain {
                        lower: vec![-0.3; 3],
                        upper: vec![0.3; 3],
                    })
                }
                _ => panic!("expected spatial distance"),
            }
        }
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..12 {
            gas.step().await.unwrap();
        }
        let mut boundary_mass = 0.;
        for step in &gas.recording().unwrap().steps {
            let result = weak_field_balance(&config.gas, step, &observables()[0]).unwrap();
            assert!(
                result
                    .field_equation_residual
                    .iter()
                    .all(|r| r.abs() < 2e-13)
            );
            boundary_mass += result
                .stage_sources
                .iter()
                .filter(|s| s.kind == "boundary")
                .map(|s| s.increment[0].abs() + s.increment[1].abs())
                .sum::<f64>();
        }
        assert!(
            boundary_mass > 0.1,
            "boundary wrapping must remain an explicit weak-field source"
        );
    });
}
#[test]
fn field_prediction_rejects_changed_clock_noise_and_missing_stage_coverage() {
    futures_lite::future::block_on(async {
        let mut config = lecture::gas_config("VI-51", &json!({"walkers":8}), 7).unwrap();
        config.physics_metric = None;
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let step = &gas.recording().unwrap().steps[0];
        let observable = &observables()[0];
        weak_field_balance(&config.gas, step, observable).unwrap();
        let mut bad = config.gas.clone();
        if let algorithmic_gas::kinetic::KineticKind::Baoab { friction, .. } =
            &mut bad.kinetic.integrator
        {
            *friction += 2.;
        }
        assert!(weak_field_balance(&bad, step, observable).is_err());
        let mut bad = step.clone();
        bad.noise[0].sample[0] += 1.;
        assert!(weak_field_balance(&config.gas, &bad, observable).is_err());
        let mut bad = step.clone();
        bad.stages.retain(|s| s.stage != "O_before_boundary");
        assert!(weak_field_balance(&config.gas, &bad, observable).is_err());
        let mut bad = step.clone();
        bad.noise[0].factor.as_mut().unwrap()[0] += 0.5;
        assert!(weak_field_balance(&config.gas, &bad, observable).is_err());
        clone_field_balance(gas.recording().unwrap(), step, observable).unwrap();
        let selected = step
            .report
            .clone_plan
            .choices
            .iter()
            .position(|choice| !choice.donors.is_empty())
            .unwrap();
        let pool = step.report.clone_plan.choices[selected].donors[0].pool_index as usize;
        let mut bad = step.clone();
        bad.report.clone_plan.sources[pool].generation += 1;
        assert!(clone_field_balance(gas.recording().unwrap(), &bad, observable).is_err());
        let mut bad = step.clone();
        let probability = bad.report.clone_plan.choices[selected].probability.unwrap();
        bad.report.clone_plan.choices[selected].probability = Some(if probability < 0.5 {
            probability + 0.1
        } else {
            probability - 0.1
        });
        assert!(clone_field_balance(gas.recording().unwrap(), &bad, observable).is_err());
    });
}

#[test]
fn conditional_fitness_metric_factor_obeys_the_same_derived_field_law() {
    futures_lite::future::block_on(async {
        let config = lecture::gas_config("VI-51", &json!({"walkers":8}), 516).unwrap();
        assert!(config.physics_metric.is_some());
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..5 {
            gas.step().await.unwrap();
        }
        let mut nonconstant_factor = false;
        let mut previous_factor: Option<Vec<f64>> = None;
        for step in &gas.recording().unwrap().steps {
            let noise = step
                .noise
                .iter()
                .find(|n| n.stream == Stream::Kinetic && n.substep == 2)
                .unwrap();
            let factor = noise.factor.as_ref().unwrap();
            if let Some(previous) = &previous_factor {
                nonconstant_factor |= previous
                    .iter()
                    .zip(factor)
                    .any(|(a, b)| (a - b).abs() > 1e-5);
            }
            previous_factor = Some(factor.clone());
            for observable in observables() {
                let result = weak_field_balance(&config.gas, step, &observable).unwrap();
                let clone =
                    clone_field_balance(gas.recording().unwrap(), step, &observable).unwrap();
                assert!(
                    result
                        .field_equation_residual
                        .iter()
                        .chain(&clone.literal_copy_residual)
                        .all(|v| v.abs() < 1e-12)
                );
                assert!(
                    result.martingale_covariance[0] >= 0. && result.martingale_covariance[3] >= 0.
                );
            }
        }
        assert!(nonconstant_factor);
    });
}

#[test]
fn absorbing_boundary_loss_and_revival_remain_in_fixed_slot_density() {
    futures_lite::future::block_on(async {
        let mut config = lecture::gas_config(
            "VI-51",
            &json!({"walkers":32,"engine_dt":0.25,"engine_memory":3}),
            516,
        )
        .unwrap();
        config.physics_metric = None;
        config.gas.boundary = BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-0.8; 3],
                upper: vec![0.8; 3],
            },
        };
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..16 {
            gas.step().await.unwrap();
        }
        let observable = WeakFieldObservable::Density { k: vec![0.; 3] };
        let mut loss = 0.;
        let mut revivals = 0;
        for step in &gas.recording().unwrap().steps {
            let field = weak_field_balance(&config.gas, step, &observable).unwrap();
            let clone = clone_field_balance(gas.recording().unwrap(), step, &observable).unwrap();
            loss += field
                .stage_sources
                .iter()
                .filter(|s| s.kind == "boundary")
                .map(|s| s.increment[0])
                .sum::<f64>();
            revivals += clone.revival_sources;
            assert_eq!(field.thermostat_conditional_increment, [0., 0.]);
            assert!(
                field
                    .field_equation_residual
                    .iter()
                    .all(|r| r.abs() < 1e-13)
            );
            assert!(
                (clone.realized_increment[0] - clone.revival_sources as f64 / 32.).abs() < 1e-13
            );
        }
        assert!(loss < -0.01);
        assert!(revivals > 0);
    });
}

#[test]
fn vi51_registry_exposes_algorithm_derived_fields_for_all_observables_and_wave_numbers() {
    use algorithmic_gas::{
        lecture::LectureRequest,
        physics::field_evolution::{CloneFieldBalance, WeakFieldBalance},
    };
    futures_lite::future::block_on(async {
        for observable in ["density", "momentum", "stress", "energy", "phase_space"] {
            // None exercises registry resolution of the actual default (1).
            for wave_number in (0..=16).map(|i| if i == 4 { None } else { Some(i as f64 / 4.) }) {
                let mut parameters = json!({"walkers":8,"field_observable":observable});
                if let Some(k) = wave_number {
                    parameters["wave_number"] = json!(k);
                }
                let mut session = lecture::LectureSession::create(LectureRequest {
                    id: "VI-51".into(),
                    seed: 516,
                    steps: 8,
                    parameters,
                })
                .await
                .unwrap();
                session.advance(8).await.unwrap();
                assert!(session.done());
                let snapshot = session.snapshot().unwrap();
                let details = &snapshot["result"]["details"];
                assert_eq!(details["calculation_origin"], "executed_algorithm_archive");
                assert_eq!(details["run_steps"], json!([8]));
                let weak = &details["weak_field_equation"];
                let clone = &details["clone_field_equation"];
                for report in [weak, clone] {
                    assert_eq!(
                        report["status"], "available",
                        "{observable} {wave_number:?}: {report}"
                    );
                    assert_eq!(report["analyzed_updates"], 8);
                    assert_eq!(report["unsupported"], json!([]));
                }
                assert_eq!(weak["fixed_capacity_normalization"], true);
                assert_eq!(
                    weak["observable"]["k"],
                    json!([wave_number.unwrap_or(1.), 0., 0.])
                );
                let expected_kind = match observable {
                    "energy" => "kinetic_energy",
                    "phase_space" => "phase_space_characteristic",
                    name => name,
                };
                assert_eq!(weak["observable"]["kind"], expected_kind);
                let weak_reports: Vec<WeakFieldBalance> =
                    serde_json::from_value(weak["reports"].clone()).unwrap();
                let clone_reports: Vec<CloneFieldBalance> =
                    serde_json::from_value(clone["reports"].clone()).unwrap();
                let evidence = session.evidence();
                assert!(evidence.configs[0].physics_metric.is_some());
                let archive = &evidence.archives[0];
                for ((field, cloning), step) in
                    weak_reports.iter().zip(&clone_reports).zip(&archive.steps)
                {
                    assert_eq!(field.step, step.report.step);
                    assert_eq!(cloning.step, step.report.step);
                    assert_eq!(field.slots, 8);
                    assert!(
                        field
                            .field_equation_residual
                            .iter()
                            .chain(&cloning.literal_copy_residual)
                            .all(|x| x.is_finite() && x.abs() < 1e-12)
                    );
                    assert!(
                        field
                            .martingale_increment
                            .iter()
                            .chain(&field.martingale_covariance)
                            .chain(&cloning.martingale_increment)
                            .chain(&cloning.martingale_covariance)
                            .all(|x| x.is_finite())
                    );
                    // The displayed series are the exact core calculation on
                    // exported actual-run evidence, not a substitute fixture.
                    let independent =
                        weak_field_balance(&archive.gas_config, step, &field.observable).unwrap();
                    let independent_clone =
                        clone_field_balance(archive, step, &field.observable).unwrap();
                    assert_eq!(
                        serde_json::to_value(field).unwrap(),
                        serde_json::to_value(independent).unwrap()
                    );
                    assert_eq!(
                        serde_json::to_value(cloning).unwrap(),
                        serde_json::to_value(independent_clone).unwrap()
                    );
                    let deterministic: Vec<_> = field
                        .stage_sources
                        .iter()
                        .filter_map(|s| s.deterministic_law_residual)
                        .collect();
                    assert_eq!(deterministic.len(), 4);
                    assert!(
                        deterministic
                            .iter()
                            .flatten()
                            .all(|r| r.is_finite() && r.abs() < 1e-12)
                    );
                    if observable == "density" {
                        assert_eq!(field.thermostat_conditional_increment, [0., 0.]);
                        assert_eq!(field.martingale_covariance, [0.; 4]);
                    }
                }
                if observable != "density" {
                    assert!(weak_reports.iter().any(|r| r.martingale_covariance[0] > 0.));
                }
                let plots = snapshot["result"]["plots"].as_array().unwrap();
                assert!(
                    plots
                        .iter()
                        .any(|p| p["title"] == "Conditional field equation: cloning term")
                );
                assert!(
                    plots
                        .iter()
                        .any(|p| p["title"] == "Conditional field equation: thermostat term")
                );
            }
        }
    });
}

#[test]
fn conditional_polynomial_fields_match_independent_quadrature_of_actual_noise_inputs() {
    use algorithmic_gas::kinetic::KineticKind;
    futures_lite::future::block_on(async {
        for law in [InnovationLaw::Gaussian, InnovationLaw::StandardizedUniform] {
            // Three-point quadrature integrates fourth-order velocity products
            // exactly without calling the predictor's moment formulas.
            let (nodes, weights) = match law {
                InnovationLaw::Gaussian => (
                    [-3_f64.sqrt(), 0., 3_f64.sqrt()],
                    [1. / 6., 2. / 3., 1. / 6.],
                ),
                InnovationLaw::StandardizedUniform => (
                    [-(9_f64 / 5.).sqrt(), 0., (9_f64 / 5.).sqrt()],
                    [5. / 18., 4. / 9., 5. / 18.],
                ),
            };
            for (friction, metric_enabled) in [0., 1e-12, 1.3]
                .into_iter()
                .flat_map(|gamma| [(gamma, false), (gamma, true)])
            {
                let mut config = lecture::gas_config(
                    "VI-51",
                    &json!({"walkers":8,"engine_memory":3,"engine_viscosity":0.7}),
                    903,
                )
                .unwrap();
                if !metric_enabled {
                    config.physics_metric = None;
                }
                config.gas.kinetic.noise.innovation = law;
                config.gas.kinetic.noise.geometry = NoiseGeometry::LowRank {
                    rank: 2,
                    factor: FactorValues::Constant {
                        values: vec![0.8, 0.2, 0.1, 0.7, 0.2, -0.4],
                    },
                };
                let KineticKind::Baoab {
                    friction: gamma,
                    dt,
                    ..
                } = &mut config.gas.kinetic.integrator
                else {
                    unreachable!()
                };
                *gamma = friction;
                let dt = *dt;
                config.gas.qft.innovation_shifts = (1..=4)
                    .flat_map(|step| {
                        [
                            InnovationShift {
                                step,
                                stream: Stream::Kinetic,
                                substep: 2,
                                walker: 0,
                                coordinate: 1,
                                shift: 0.6,
                            },
                            InnovationShift {
                                step,
                                stream: Stream::Kinetic,
                                substep: 2,
                                walker: 0,
                                coordinate: 1,
                                shift: -0.2,
                            },
                        ]
                    })
                    .collect();
                let mut gas = config.build::<f64>().await.unwrap();
                gas.start_recording(RecordingConfig::default()).unwrap();
                for _ in 0..4 {
                    gas.step().await.unwrap();
                }
                let mut largest_cross_covariance: f64 = 0.;
                for step in &gas.recording().unwrap().steps {
                    let input = step.stages.iter().find(|s| s.stage == "A1").unwrap();
                    let noise = step
                        .noise
                        .iter()
                        .find(|s| s.stream == Stream::Kinetic && s.substep == 2)
                        .unwrap();
                    let n = input.validity.len();
                    let positions = &input.fields["positions"].values;
                    let velocities = &input.fields["velocities"].values;
                    let decay = (-friction * dt).exp();
                    let scale = if friction == 0. {
                        dt.sqrt()
                    } else {
                        (-(-2. * friction * dt).exp_m1() / (2. * friction)).sqrt()
                    };
                    for observable in observables().into_iter().skip(1) {
                        let predicted = weak_field_balance(&config.gas, step, &observable).unwrap();
                        let mut mean_increment = [0.; 2];
                        let mut covariance = [0.; 4];
                        let magnitude = |v: &[f64]| match observable {
                            WeakFieldObservable::Density { .. } => 1.,
                            WeakFieldObservable::Momentum { component, .. } => v[component],
                            WeakFieldObservable::Stress { a, b, .. } => v[a] * v[b],
                            WeakFieldObservable::KineticEnergy { .. } => {
                                0.5 * v.iter().map(|v| v * v).sum::<f64>()
                            }
                            _ => unreachable!(),
                        };
                        for i in 0..n {
                            if !input.validity[i].eligible(config.gas.include_truncated) {
                                continue;
                            }
                            let k = [0.7, -0.4, 0.2];
                            let angle: f64 = (0..3).map(|a| k[a] * positions[3 * i + a]).sum();
                            let phase = [angle.cos(), angle.sin()];
                            let factor = noise.dense_factor(i).unwrap();
                            let rank = factor.len() / 3;
                            assert_eq!(rank, if metric_enabled { 3 } else { 2 });
                            let mut shift = vec![0.; rank];
                            for source in &noise.applied_source_shifts {
                                if source.walker == i {
                                    shift[source.coordinate] += source.shift;
                                }
                            }
                            let mut samples = Vec::new();
                            let mut mean = [0.; 2];
                            for index in 0..3_usize.pow(rank as u32) {
                                let mut address = index;
                                let mut weight = 1.;
                                let innovation: Vec<_> = (0..rank)
                                    .map(|r| {
                                        let digit = address % 3;
                                        address /= 3;
                                        weight *= weights[digit];
                                        nodes[digit] + shift[r]
                                    })
                                    .collect();
                                let velocity: Vec<_> = (0..3)
                                    .map(|a| {
                                        decay * velocities[3 * i + a]
                                            + scale
                                                * (0..rank)
                                                    .map(|r| factor[a * rank + r] * innovation[r])
                                                    .sum::<f64>()
                                    })
                                    .collect();
                                let value = magnitude(&velocity);
                                let value = [phase[0] * value, phase[1] * value];
                                for a in 0..2 {
                                    mean[a] += weight * value[a];
                                }
                                samples.push((weight, value));
                            }
                            let initial = magnitude(&velocities[3 * i..3 * i + 3]);
                            for a in 0..2 {
                                mean_increment[a] += (mean[a] - phase[a] * initial) / n as f64;
                            }
                            for (weight, value) in samples {
                                for a in 0..2 {
                                    for b in 0..2 {
                                        covariance[2 * a + b] +=
                                            weight * (value[a] - mean[a]) * (value[b] - mean[b])
                                                / (n * n) as f64;
                                    }
                                }
                            }
                        }
                        for (a, b) in mean_increment
                            .iter()
                            .zip(predicted.thermostat_conditional_increment)
                        {
                            assert!(
                                (a - b).abs() < 2e-12,
                                "{law:?} gamma={friction} {observable:?}"
                            );
                        }
                        for (a, b) in covariance.iter().zip(predicted.martingale_covariance) {
                            assert!(
                                (a - b).abs() < 2e-12,
                                "quadrature covariance {law:?} gamma={friction}: {covariance:?} != {:?}",
                                predicted.martingale_covariance
                            );
                        }
                        largest_cross_covariance =
                            largest_cross_covariance.max(covariance[1].abs());
                    }
                }
                assert!(
                    (metric_enabled && friction < 1e-10) || largest_cross_covariance > 1e-6,
                    "must exercise complex cross covariance"
                );
                println!(
                    "Quadrature {law:?} friction={friction} metric={metric_enabled}: conditional mean and full covariance agree; max cross covariance={largest_cross_covariance:.6}"
                );
            }
        }
    });
}

#[test]
fn clone_mean_and_full_covariance_match_all_gate_outcomes_on_actual_historical_contexts() {
    futures_lite::future::block_on(async {
        let config = lecture::gas_config(
            "VI-51",
            &json!({"walkers":8,"engine_memory":3,"engine_viscosity":0.7}),
            20011,
        )
        .unwrap();
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..6 {
            gas.step().await.unwrap();
        }
        let archive = gas.recording().unwrap();
        let mut historical = 0;
        let mut nonzero_cross = false;
        for step in &archive.steps {
            let input = step.stages.iter().find(|s| s.stage == "pre_clone").unwrap();
            let n = input.validity.len();
            assert!(
                input
                    .validity
                    .iter()
                    .all(|v| v.eligible(config.gas.include_truncated))
            );
            let plan = &step.report.clone_plan;
            for observable in observables() {
                let evaluate = |x: &[f64], v: &[f64]| {
                    let mut angle: f64 = [0.7, -0.4, 0.2].iter().zip(x).map(|(a, b)| a * b).sum();
                    let magnitude = match &observable {
                        WeakFieldObservable::PhaseSpaceCharacteristic { l, .. } => {
                            angle += l.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
                            1.
                        }
                        WeakFieldObservable::Density { .. } => 1.,
                        WeakFieldObservable::Momentum { component, .. } => v[*component],
                        WeakFieldObservable::Stress { a, b, .. } => v[*a] * v[*b],
                        WeakFieldObservable::KineticEnergy { .. } => {
                            0.5 * v.iter().map(|v| v * v).sum::<f64>()
                        }
                    };
                    [magnitude * angle.cos(), magnitude * angle.sin()]
                };
                let mut current = Vec::new();
                let mut donor = Vec::new();
                let mut probabilities = Vec::new();
                for (i, choice) in plan.choices.iter().enumerate() {
                    let value = evaluate(
                        &input.fields["positions"].values[3 * i..3 * i + 3],
                        &input.fields["velocities"].values[3 * i..3 * i + 3],
                    );
                    current.push(value);
                    probabilities.push(choice.probability.unwrap());
                    if choice.donors.is_empty() {
                        donor.push(value);
                        continue;
                    }
                    let source = &plan.sources[choice.donors[0].pool_index as usize];
                    historical += usize::from(source.frame < step.report.step - 1);
                    let population = archive
                        .anchors
                        .iter()
                        .filter(|a| a.epoch == step.epoch && a.step == source.frame)
                        .map(|a| &a.population)
                        .chain(
                            archive
                                .steps
                                .iter()
                                .filter(|s| {
                                    s.epoch == step.epoch
                                        && s.report.step.checked_sub(1) == Some(source.frame)
                                })
                                .map(|s| &s.before),
                        )
                        .find(|p| {
                            p.version == source.version
                                && p.generations[source.slot as usize] == source.generation
                        })
                        .unwrap();
                    donor.push(evaluate(
                        population
                            .observations
                            .field("positions")
                            .unwrap()
                            .row(source.slot as usize)
                            .unwrap(),
                        population
                            .observations
                            .field("velocities")
                            .unwrap()
                            .row(source.slot as usize)
                            .unwrap(),
                    ));
                }
                // Enumerate the conditional gate law and measure complete
                // swarm increments, avoiding the analytic p(1-p) formula.
                let mut outcomes = Vec::new();
                let mut mean = [0.; 2];
                let mut total_weight = 0.;
                for gates in 0..(1_usize << n) {
                    let mut weight = 1.;
                    let mut increment = [0.; 2];
                    for i in 0..n {
                        if gates & (1 << i) != 0 {
                            weight *= probabilities[i];
                            for (a, value) in increment.iter_mut().enumerate() {
                                *value += (donor[i][a] - current[i][a]) / n as f64;
                            }
                        } else {
                            weight *= 1. - probabilities[i];
                        }
                    }
                    if weight == 0. {
                        continue;
                    }
                    total_weight += weight;
                    for a in 0..2 {
                        mean[a] += weight * increment[a];
                    }
                    outcomes.push((weight, increment));
                }
                assert!((total_weight - 1.).abs() < 1e-12);
                let mut covariance = [0.; 4];
                for (weight, value) in outcomes {
                    for a in 0..2 {
                        for b in 0..2 {
                            covariance[a * 2 + b] +=
                                weight * (value[a] - mean[a]) * (value[b] - mean[b]);
                        }
                    }
                }
                let analytic = clone_field_balance(archive, step, &observable).unwrap();
                for (a, b) in mean.iter().zip(analytic.conditional_increment) {
                    assert!((a - b).abs() < 2e-12);
                }
                for (a, b) in covariance.iter().zip(analytic.martingale_covariance) {
                    assert!((a - b).abs() < 2e-12);
                }
                nonzero_cross |= covariance[1].abs() > 1e-8;
            }
        }
        assert!(historical > 0);
        assert!(nonzero_cross);
    });
}

#[test]
fn field_equations_cover_single_precision_recorded_execution() {
    futures_lite::future::block_on(async {
        let mut config = lecture::gas_config(
            "VI-51",
            &json!({"walkers":8,"engine_memory":3,"engine_viscosity":0.7}),
            516,
        )
        .unwrap();
        config.gas.precision = algorithmic_gas::Precision::F32;
        let mut gas = config.build::<f32>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..6 {
            gas.step().await.unwrap();
        }
        let archive = gas.recording().unwrap();
        for step in &archive.steps {
            for observable in observables() {
                let weak = weak_field_balance(&config.gas, step, &observable).unwrap();
                let cloning = clone_field_balance(archive, step, &observable).unwrap();
                assert!(
                    weak.field_equation_residual
                        .iter()
                        .chain(&cloning.literal_copy_residual)
                        .all(|x| x.abs() < 1e-12)
                );
                for stage in &weak.stage_sources {
                    if let Some(residual) = stage.deterministic_law_residual {
                        assert!(residual.iter().all(|x| x.abs() < 2e-5));
                    }
                }
            }
        }
    });
}
