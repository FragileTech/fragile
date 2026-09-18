use algorithmic_gas::{
    RecordingConfig,
    physics::field_evolution::{WeakFieldObservable, collision_field_balance},
};
use algorithmic_gas_benchmarks::lecture;
use serde_json::json;

// Private exact degree-four Haar cubature. These rotations integrate the
// conditional field law on genuine archived gas inputs; they are not demo data.
fn rotations(d: usize) -> Vec<(Vec<f64>, f64)> {
    if d == 1 {
        return vec![(vec![-1.], 0.5), (vec![1.], 0.5)];
    }
    let mut out = vec![];
    if d == 2 {
        for k in 0..16 {
            let (s, c) = (std::f64::consts::TAU * k as f64 / 16.).sin_cos();
            for sign in [-1., 1.] {
                out.push((vec![sign * c, -s, sign * s, c], 1. / 32.));
            }
        }
    } else {
        for a in 0..9 {
            for b in 0..9 {
                let (sa, ca) = (std::f64::consts::TAU * a as f64 / 9.).sin_cos();
                let (sg, cg) = (std::f64::consts::TAU * b as f64 / 9.).sin_cos();
                for (cb, weight) in [
                    (-(3_f64 / 5.).sqrt(), 5. / 18.),
                    (0., 4. / 9.),
                    ((3_f64 / 5.).sqrt(), 5. / 18.),
                ] {
                    let sb = (1. - cb * cb).sqrt();
                    let r = vec![
                        ca * cb * cg - sa * sg,
                        -ca * cb * sg - sa * cg,
                        ca * sb,
                        sa * cb * cg + ca * sg,
                        -sa * cb * sg + ca * cg,
                        sa * sb,
                        -sb * cg,
                        sb * sg,
                        cb,
                    ];
                    for sign in [-1., 1.] {
                        out.push((r.iter().map(|x| sign * x).collect(), weight / 162.));
                    }
                }
            }
        }
    }
    out
}
fn value(observable: &WeakFieldObservable, x: &[f64], v: &[f64]) -> [f64; 2] {
    let (k, magnitude) = match observable {
        WeakFieldObservable::Density { k } => (k, 1.),
        WeakFieldObservable::Momentum { k, component } => (k, v[*component]),
        WeakFieldObservable::Stress { k, a, b } => (k, v[*a] * v[*b]),
        WeakFieldObservable::KineticEnergy { k } => (k, 0.5 * v.iter().map(|x| x * x).sum::<f64>()),
        _ => unreachable!(),
    };
    let phase = k.iter().zip(x).map(|(a, b)| a * b).sum::<f64>();
    [magnitude * phase.cos(), magnitude * phase.sin()]
}
#[test]
fn component_fields_match_exact_haar_cubature_on_actual_gas_inputs_in_dimensions_one_two_three() {
    futures_lite::future::block_on(async {
        for d in [1, 2, 3] {
            let mut config =
                lecture::gas_config("VI-51", &json!({"walkers":12,"engine_memory":0}), 7).unwrap();
            config.dimensions = d;
            config.physics_metric = None;
            config.gas.clone_transform.restitution = Some(0.7);
            config.gas.clone_transform.velocity_field = Some("velocities".into());
            let mut gas = config.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig::default()).unwrap();
            for _ in 0..4 {
                gas.step().await.unwrap();
            }
            let mut nonzero_variance = false;
            let mut components_checked = 0;
            for step in &gas.recording().unwrap().steps {
                let before = step
                    .before
                    .observations
                    .field("velocities")
                    .unwrap()
                    .values();
                let literal = step
                    .stages
                    .iter()
                    .find(|s| s.stage == "literal_clone")
                    .unwrap();
                let after = step
                    .stages
                    .iter()
                    .find(|s| s.stage == "post_transform")
                    .unwrap();
                let positions = &after.fields["positions"].values;
                let n = after.validity.len();
                let mut k = vec![0.; d];
                k[0] = 0.7;
                let mut observables = vec![
                    WeakFieldObservable::Density { k: k.clone() },
                    WeakFieldObservable::Momentum {
                        k: k.clone(),
                        component: 0,
                    },
                    WeakFieldObservable::Stress {
                        k: k.clone(),
                        a: 0,
                        b: 0,
                    },
                    WeakFieldObservable::KineticEnergy { k: k.clone() },
                ];
                if d > 1 {
                    observables.push(WeakFieldObservable::Stress {
                        k: k.clone(),
                        a: 0,
                        b: 1,
                    });
                }
                for observable in observables {
                    let report = collision_field_balance(&config.gas, step, &observable).unwrap();
                    assert!(report.maximum_momentum_conservation_residual < 1e-12);
                    assert!(report.maximum_relative_energy_residual < 1e-12);
                    let mut covered = vec![false; n];
                    let mut reference_mean = [0.; 2];
                    let mut reference_cov = [0.; 4];
                    for members in &report.component_slots {
                        components_checked += 1;
                        let mut u = vec![0.; d];
                        for &i in members {
                            covered[i] = true;
                            for a in 0..d {
                                u[a] += before[i * d + a] / members.len() as f64;
                            }
                        }
                        let mut mean = [0.; 2];
                        let mut second = [0.; 4];
                        for (r, weight) in rotations(d) {
                            let mut f = [0.; 2];
                            for &i in members {
                                let velocity = (0..d)
                                    .map(|a| {
                                        u[a] + 0.7
                                            * (0..d)
                                                .map(|b| r[a * d + b] * (before[i * d + b] - u[b]))
                                                .sum::<f64>()
                                    })
                                    .collect::<Vec<_>>();
                                let z =
                                    value(&observable, &positions[i * d..(i + 1) * d], &velocity);
                                for a in 0..2 {
                                    f[a] += z[a] / n as f64;
                                }
                            }
                            for a in 0..2 {
                                mean[a] += weight * f[a];
                                for b in 0..2 {
                                    second[2 * a + b] += weight * f[a] * f[b];
                                }
                            }
                        }
                        for a in 0..2 {
                            reference_mean[a] += mean[a];
                            for b in 0..2 {
                                reference_cov[2 * a + b] += second[2 * a + b] - mean[a] * mean[b];
                            }
                        }
                    }
                    let mut baseline = [0.; 2];
                    for i in 0..n {
                        let z = value(
                            &observable,
                            &positions[i * d..(i + 1) * d],
                            &literal.fields["velocities"].values[i * d..(i + 1) * d],
                        );
                        for a in 0..2 {
                            baseline[a] += z[a] / n as f64;
                            if !covered[i] {
                                reference_mean[a] += z[a] / n as f64;
                            }
                        }
                    }
                    for a in 0..2 {
                        assert!(
                            (report.conditional_increment[a] - (reference_mean[a] - baseline[a]))
                                .abs()
                                < 2e-12,
                            "d={d} mean {observable:?}"
                        );
                    }
                    for a in 0..4 {
                        assert!(
                            (report.martingale_covariance[a] - reference_cov[a]).abs() < 2e-12,
                            "d={d} covariance {observable:?}: {:?} vs {:?}",
                            report.martingale_covariance,
                            reference_cov
                        );
                    }
                    nonzero_variance |= report.martingale_covariance[0] > 1e-10;
                }
                // At zero wave number shared rotations conserve total momentum
                // and total relative kinetic energy; independent-walker noise
                // would give a spurious positive variance here.
                for observable in [
                    WeakFieldObservable::Momentum {
                        k: vec![0.; d],
                        component: 0,
                    },
                    WeakFieldObservable::KineticEnergy { k: vec![0.; d] },
                ] {
                    let report = collision_field_balance(&config.gas, step, &observable).unwrap();
                    assert!(report.martingale_covariance.iter().all(|x| x.abs() < 2e-12));
                    assert!(report.martingale_increment.iter().all(|x| x.abs() < 2e-12));
                }
                let phase = WeakFieldObservable::PhaseSpaceCharacteristic {
                    k: vec![0.4; d],
                    l: vec![0.75; d],
                };
                if d == 2 {
                    assert!(collision_field_balance(&config.gas, step, &phase).is_err());
                } else {
                    let report = collision_field_balance(&config.gas, step, &phase).unwrap();
                    assert!(
                        report.martingale_covariance[0] >= 0.
                            && report.martingale_covariance[3] >= 0.
                    );
                }
            }
            assert!(components_checked > 0);
            assert!(
                nonzero_variance,
                "spatial field must exercise nonzero Haar fluctuations in d={d}"
            );
        }
    });
}

#[test]
fn revived_rows_enter_frozen_component_moments_and_graph_conditioning() {
    futures_lite::future::block_on(async {
        for d in [1, 2, 3] {
            let mut config =
                lecture::gas_config("VI-51", &json!({"walkers":24,"engine_memory":0}), 7).unwrap();
            config.dimensions = d;
            config.physics_metric = None;
            config.gas.clone_transform.restitution = Some(0.7);
            config.gas.clone_transform.velocity_field = Some("velocities".into());
            config.gas.boundary = algorithmic_gas::boundary::BoundaryPolicy::AbsorbingBox {
                field: "velocities".into(),
                domain: algorithmic_gas::boundary::BoxDomain {
                    lower: vec![-0.35; d],
                    upper: vec![0.35; d],
                },
            };
            let mut gas = config.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig::default()).unwrap();
            for _ in 0..8 {
                gas.step().await.unwrap();
            }
            let mut revival_count = 0;
            for step in &gas.recording().unwrap().steps {
                let report = collision_field_balance(
                    &config.gas,
                    step,
                    &WeakFieldObservable::Momentum {
                        k: vec![0.; d],
                        component: 0,
                    },
                )
                .unwrap();
                for (i, choice) in step.report.clone_plan.choices.iter().enumerate() {
                    if choice.revival {
                        revival_count += 1;
                        assert!(
                            report
                                .component_slots
                                .iter()
                                .any(|component| component.contains(&i))
                        );
                    }
                }
                assert!(report.maximum_momentum_conservation_residual < 1e-12);
                assert!(report.maximum_relative_energy_residual < 1e-12);
                assert!(report.martingale_increment.iter().all(|x| x.abs() < 1e-12));
                assert!(report.martingale_covariance.iter().all(|x| x.abs() < 1e-12));
            }
            assert!(revival_count > 0, "must execute dead-row revival in d={d}");
        }
    });
}
