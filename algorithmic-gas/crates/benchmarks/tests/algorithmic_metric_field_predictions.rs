//! Ensemble validation under the actual conditional-fitness metric provider.
//! Every statistical sampling unit is a complete independent trajectory. The
//! six adaptive within-trajectory increments are never treated as six replicas.
use algorithmic_gas::{
    RecordingConfig,
    noise::InnovationLaw,
    physics::field_evolution::{WeakFieldObservable, clone_field_balance, weak_field_balance},
    random::Stream,
};
use algorithmic_gas_benchmarks::lecture;
use serde_json::json;

#[derive(Clone, Default)]
struct Trajectory {
    total: [f64; 2],
    predictable: [f64; 4],
    realized: [f64; 4],
}
impl Trajectory {
    fn accumulate(&mut self, increment: [f64; 2], covariance: [f64; 4]) {
        for (a, total) in self.total.iter_mut().enumerate() {
            *total += increment[a];
        }
        for a in 0..2 {
            for b in 0..2 {
                self.predictable[a * 2 + b] += covariance[a * 2 + b];
                self.realized[a * 2 + b] += increment[a] * increment[b];
            }
        }
    }
}
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
fn project(v: [f64; 2], direction: [f64; 2]) -> f64 {
    v[0] * direction[0] + v[1] * direction[1]
}
fn quadratic(matrix: [f64; 4], direction: [f64; 2]) -> f64 {
    direction[0] * direction[0] * matrix[0]
        + direction[0] * direction[1] * (matrix[1] + matrix[2])
        + direction[1] * direction[1] * matrix[3]
}
fn mean_standard_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.);
    (mean, (variance / n).sqrt())
}
fn standardized(mean: f64, standard_error: f64) -> f64 {
    if standard_error == 0. {
        assert_eq!(mean, 0., "deterministic field residual must vanish");
        0.
    } else {
        mean / standard_error
    }
}

#[test]
fn adaptive_metric_fields_predict_independent_trajectory_means_and_complex_covariance() {
    futures_lite::future::block_on(async {
        const STEPS: usize = 6;
        let observables = observables();
        let projections = [
            ("real", [1., 0.]),
            ("imaginary", [0., 1.]),
            ("real_plus_imaginary", [1., 1.]),
            ("real_minus_imaginary", [1., -1.]),
        ];
        let mut comparisons = Vec::new();
        let mut batches = Vec::new();
        let mut failures = Vec::new();
        // Fixed disjoint precision batches retain every earlier result. The
        // batch sizes, seed ranges and regression thresholds never adapt to outcomes.
        for (law, seed_start, replicas) in [
            (InnovationLaw::Gaussian, 0, 128),
            (InnovationLaw::StandardizedUniform, 0, 128),
            (InnovationLaw::StandardizedUniform, 10_000, 128),
            (InnovationLaw::StandardizedUniform, 20_000, 512),
        ] {
            let mut ensembles = vec![Vec::<Trajectory>::new(); observables.len() * 2];
            let mut accepted_historical = 0;
            let mut adaptive_factor_changes = 0;
            let mut nonzero_cross_covariances = 0;
            let mut nonzero_clone_cross_covariances = 0;
            let mut metric_inverse_residual_max: f64 = 0.;
            let mut configuration = None;
            for seed in 0..replicas {
                let mut config =
                    lecture::gas_config("VI-51", &json!({"walkers":8}), (seed_start + seed) as u64)
                        .unwrap();
                assert!(config.physics_metric.is_some());
                assert!(config.gas.cloning_donors.history_window > 0);
                assert!(config.gas.qft.viscosity.as_ref().unwrap().coefficient > 0.);
                config.gas.kinetic.noise.innovation = law;
                if seed == 0 {
                    configuration = Some(serde_json::to_value(&config).unwrap());
                }
                let mut gas = config.build::<f64>().await.unwrap();
                gas.start_recording(RecordingConfig {
                    max_steps: STEPS,
                    ..Default::default()
                })
                .unwrap();
                for _ in 0..STEPS {
                    gas.step().await.unwrap();
                }
                let archive = gas.recording().unwrap();
                assert_eq!(archive.steps.len(), STEPS);
                let mut trajectories = vec![Trajectory::default(); observables.len() * 2];
                let mut previous_factor: Option<&Vec<f64>> = None;
                for step in &archive.steps {
                    let noise = step
                        .noise
                        .iter()
                        .find(|n| n.stream == Stream::Kinetic && n.substep == 2)
                        .unwrap();
                    assert_eq!(noise.innovation_law, Some(law));
                    let factor = noise.factor.as_ref().unwrap();
                    let metric = step
                        .field_evaluations
                        .iter()
                        .find(|f| f.stage == "O" && f.field == "fitness_metric")
                        .unwrap();
                    let hessian = step
                        .field_evaluations
                        .iter()
                        .find(|f| f.stage == "O" && f.field == "fitness_hessian")
                        .unwrap();
                    let metric_config = config.physics_metric.as_ref().unwrap();
                    let friction = match config.gas.kinetic.integrator {
                        algorithmic_gas::kinetic::KineticKind::Baoab { friction, .. } => friction,
                        _ => unreachable!(),
                    };
                    let amplitude2 = 2. * friction * metric_config.temperature;
                    for walker in 0..8 {
                        let range = walker * 9..(walker + 1) * 9;
                        let reconstructed = algorithmic_gas::physics::geometry::metric_spectrum(
                            &hessian.values[range.clone()],
                            3,
                            metric_config.epsilon,
                            metric_config.policy,
                        )
                        .unwrap();
                        assert!(
                            reconstructed
                                .metric
                                .iter()
                                .zip(&metric.values[range])
                                .all(|(a, b)| (a - b).abs() < 1e-10 * (1. + b.abs()))
                        );
                        let b = noise.dense_factor(walker).unwrap();
                        assert_eq!(b.len(), 9);
                        // An independent matrix product checks BB^T=2 gamma T g^-1;
                        // no recorded inverse/factor is reused as its own answer.
                        for a in 0..3 {
                            for c in 0..3 {
                                let product = (0..3)
                                    .map(|j| {
                                        metric.values[walker * 9 + a * 3 + j]
                                            * (0..3)
                                                .map(|r| b[j * 3 + r] * b[c * 3 + r])
                                                .sum::<f64>()
                                    })
                                    .sum::<f64>();
                                let target = if a == c { amplitude2 } else { 0. };
                                let residual = (product - target).abs() / (1. + amplitude2);
                                metric_inverse_residual_max =
                                    metric_inverse_residual_max.max(residual);
                                assert!(
                                    residual < 1e-8,
                                    "actual diffusion/metric inverse relation: {residual}"
                                );
                            }
                        }
                    }
                    if previous_factor.is_some_and(|previous| previous != factor) {
                        adaptive_factor_changes += 1;
                    }
                    previous_factor = Some(factor);
                    accepted_historical += step
                        .report
                        .clone_plan
                        .choices
                        .iter()
                        .filter(|choice| {
                            choice.accepted
                                && choice.donors.iter().any(|donor| {
                                    step.report.clone_plan.sources[donor.pool_index as usize].frame
                                        < step.report.step - 1
                                })
                        })
                        .count();
                    for (index, observable) in observables.iter().enumerate() {
                        let result = weak_field_balance(&config.gas, step, observable).unwrap();
                        assert!(
                            result
                                .field_equation_residual
                                .iter()
                                .all(|r| r.is_finite() && r.abs() < 1e-12)
                        );
                        trajectories[index]
                            .accumulate(result.martingale_increment, result.martingale_covariance);
                        let clone = clone_field_balance(archive, step, observable).unwrap();
                        trajectories[observables.len() + index]
                            .accumulate(clone.martingale_increment, clone.martingale_covariance);
                        assert!(clone.literal_copy_residual.iter().all(|r| r.abs() < 1e-12));
                        if clone.martingale_covariance[1].abs() > 1e-12 {
                            nonzero_clone_cross_covariances += 1;
                        }
                        if result.martingale_covariance[1].abs() > 1e-12 {
                            nonzero_cross_covariances += 1;
                        }
                    }
                }
                for (ensemble, trajectory) in ensembles.iter_mut().zip(trajectories) {
                    ensemble.push(trajectory);
                }
            }
            assert!(
                accepted_historical > 0,
                "must execute historical donor replacement"
            );
            assert!(
                adaptive_factor_changes > 0,
                "must exercise the state-dependent metric, not a constant factor"
            );
            assert!(
                nonzero_cross_covariances > 0,
                "complex off-diagonal covariance must be exercised"
            );
            assert!(
                nonzero_clone_cross_covariances > 0,
                "clone cross covariance must be exercised"
            );
            println!(
                "{law:?}: seed_start={seed_start}, independent_trajectories={replicas}, stages_per_trajectory={STEPS}, accepted_historical={accepted_historical}, adaptive_factor_changes={adaptive_factor_changes}, nonzero_cross_covariances={nonzero_cross_covariances}, nonzero_clone_cross_covariances={nonzero_clone_cross_covariances}, metric_inverse_residual_max={metric_inverse_residual_max:.8e}"
            );
            batches.push(json!({"innovation":law,"seed_start":seed_start,"seed_end_exclusive":seed_start+replicas,"independent_trajectories":replicas,"stages_per_trajectory":STEPS,"configuration_at_first_seed":configuration,"accepted_historical":accepted_historical,"adaptive_factor_changes":adaptive_factor_changes,"nonzero_thermostat_cross_covariances":nonzero_cross_covariances,"nonzero_clone_cross_covariances":nonzero_clone_cross_covariances,"metric_inverse_residual_max":metric_inverse_residual_max}));
            for (index, ensemble) in ensembles.iter().enumerate() {
                let observable = &observables[index % observables.len()];
                let term = if index < observables.len() {
                    "thermostat"
                } else {
                    "cloning"
                };
                for (name, direction) in projections {
                    let totals: Vec<_> = ensemble
                        .iter()
                        .map(|t| project(t.total, direction))
                        .collect();
                    let predicted: Vec<_> = ensemble
                        .iter()
                        .map(|t| quadratic(t.predictable, direction))
                        .collect();
                    let realized: Vec<_> = ensemble
                        .iter()
                        .map(|t| quadratic(t.realized, direction))
                        .collect();
                    let terminal_squares: Vec<_> = totals.iter().map(|x| x * x).collect();
                    let bracket_error: Vec<_> = realized
                        .iter()
                        .zip(&predicted)
                        .map(|(observed, predicted)| observed - predicted)
                        .collect();
                    let terminal_error: Vec<_> = terminal_squares
                        .iter()
                        .zip(&predicted)
                        .map(|(observed, predicted)| observed - predicted)
                        .collect();
                    let (mean, empirical_se) = mean_standard_error(&totals);
                    let (predicted_mean, _) = mean_standard_error(&predicted);
                    let (realized_mean, _) = mean_standard_error(&realized);
                    let (terminal_mean, _) = mean_standard_error(&terminal_squares);
                    let (bracket_mean, bracket_se) = mean_standard_error(&bracket_error);
                    let (terminal_difference, terminal_se) = mean_standard_error(&terminal_error);
                    let mean_z = standardized(mean, empirical_se);
                    let bracket_z = standardized(bracket_mean, bracket_se);
                    let terminal_z = standardized(terminal_difference, terminal_se);
                    if predicted_mean <= 1e-20 {
                        assert!(totals.iter().chain(&realized).all(|v| v.abs() < 1e-12));
                        comparisons.push(json!({"innovation":law,"seed_start":seed_start,"replicas":replicas,"term":term,"observable":observable,"projection":name,"direction":direction,"deterministic":true,"mean":mean,"empirical_standard_error":empirical_se,"predicted_quadratic_variation":predicted_mean,"realized_quadratic_variation":realized_mean,"terminal_square_mean":terminal_mean}));
                        continue;
                    }
                    let analytic_se = (predicted_mean / replicas as f64).sqrt();
                    let analytic_mean_z = mean / analytic_se;
                    let bracket_ratio = realized_mean / predicted_mean;
                    let terminal_ratio = terminal_mean / predicted_mean;
                    println!(
                        "{law:?} seed_start={seed_start} term={term} {observable:?} projection={name} mean={mean:.8e} empirical_se={empirical_se:.8e} analytic_se={analytic_se:.8e} mean_z={mean_z:.4} analytic_mean_z={analytic_mean_z:.4} quadratic_variation_ratio={bracket_ratio:.4} quadratic_variation_z={bracket_z:.4} terminal_square_ratio={terminal_ratio:.4} terminal_square_z={terminal_z:.4}"
                    );
                    // Independent trajectory-based uncertainty is used for both
                    // mean and variance comparisons. Projecting along real±imag
                    // also tests the nonzero complex covariance contribution.
                    comparisons.push(json!({"innovation":law,"seed_start":seed_start,"replicas":replicas,"term":term,"observable":observable,"projection":name,"direction":direction,"deterministic":false,"mean":mean,"empirical_standard_error":empirical_se,"analytic_standard_error":analytic_se,"mean_z":mean_z,"analytic_mean_z":analytic_mean_z,"predicted_quadratic_variation":predicted_mean,"realized_quadratic_variation":realized_mean,"quadratic_variation_difference":bracket_mean,"quadratic_variation_difference_standard_error":bracket_se,"quadratic_variation_ratio":bracket_ratio,"quadratic_variation_z":bracket_z,"terminal_square_mean":terminal_mean,"terminal_square_difference":terminal_difference,"terminal_square_difference_standard_error":terminal_se,"terminal_square_ratio":terminal_ratio,"terminal_square_z":terminal_z}));
                    for (comparison, z) in [
                        ("trajectory_mean", mean_z),
                        ("analytic_trajectory_mean", analytic_mean_z),
                        ("quadratic_variation", bracket_z),
                        ("terminal_covariance", terminal_z),
                    ] {
                        if z.abs() >= 4.5 {
                            failures.push(format!("{comparison}: {law:?} seed_start={seed_start} term={term} {observable:?} {name} z={z}"));
                        }
                    }
                }
            }
        }
        let report = json!({"sampling_unit":"complete independent trajectory","fixed_standard_error_threshold":4.5,"batches":batches,"comparisons":comparisons,"failures":failures});
        if let Some(path) = std::env::var_os("ALGORITHMIC_GAS_FIELD_REPORT") {
            std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        }
        assert!(
            failures.is_empty(),
            "independent metric field validation failures: {failures:?}"
        );
    });
}
