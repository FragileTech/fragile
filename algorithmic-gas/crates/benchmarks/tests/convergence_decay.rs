use algorithmic_gas::{GasBuilder, GasConfig, ObservationBatch, Population, TensorBatch};
use algorithmic_gas_benchmarks::convergence_decay::{
    DecayProfile, DecayValidationConfig, PairRandomness, canonical_decay_population,
    canonicalize_decay_engine, decay_observables, diffusion_from_well_spacing,
    linear_decay_constants, refresh_decay_summary, validate_decay,
};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel};

fn population(x: &[f64], v: &[f64], dead: &[usize]) -> Population<f64> {
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(x.len(), 1, x.to_vec()).unwrap());
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(v.len(), 1, v.to_vec()).unwrap(),
    );
    let mut population = Population::new(observations).unwrap();
    for &i in dead {
        population.validity[i].invalid = true;
    }
    population
}

#[test]
fn exact_linear_lyapunov_rate_is_positive_and_independent_of_n() {
    let c = linear_decay_constants(0.04, 1., 1., 0.2, 0.01).unwrap();
    assert!(c.lambda_min > 0.);
    assert!(c.contraction_factor > 0. && c.contraction_factor < 1.);
    assert!(c.squared_error_decay_rate > 0.);
    assert!(c.lyapunov_identity_residual < 1e-12);
    assert_eq!(c.lyapunov_matrix[0][0], 1.);
    assert!(linear_decay_constants(0., 1., 1., 0.2, 0.01).is_err());
    assert!(linear_decay_constants(5., 1., 1., 0., 0.).is_err());
}

#[test]
fn optimal_marked_error_and_components_ignore_independent_storage_permutations() {
    let p = linear_decay_constants(0.04, 1., 1., 0., 0.)
        .unwrap()
        .lyapunov_matrix;
    let left = population(&[-0.4, 0.3, 2.5], &[0.1, -0.1, 0.], &[2]);
    let right = population(&[-0.2, 0.4, 2.6], &[0.2, -0.2, 0.], &[2]);
    let a = decay_observables(&left, &right, p, 1.).unwrap();
    let left = population(&[2.5, -0.4, 0.3], &[0., 0.1, -0.1], &[0]);
    let right = population(&[0.4, 2.6, -0.2], &[-0.2, 0., 0.2], &[1]);
    let b = decay_observables(&left, &right, p, 1.).unwrap();
    assert_eq!(
        serde_json::to_value(a).unwrap(),
        serde_json::to_value(b).unwrap()
    );
    let identical = decay_observables(
        &left,
        &population(&[0.3, 2.5, -0.4], &[-0.1, 0., 0.1], &[1]),
        p,
        1.,
    )
    .unwrap();
    assert_eq!(identical.full_marked_error, 0.);
    assert_eq!(identical.alive_wasserstein_squared, Some(0.));
}

#[test]
fn unequal_alive_supports_and_empty_alive_measure_remain_explicit() {
    let p = [[1., 0.], [0., 1.]];
    let left = population(&[-1., 0., 1.], &[0.; 3], &[2]);
    let right = population(&[-1., 0., 1.], &[0.; 3], &[]);
    let observed = decay_observables(&left, &right, p, 1.).unwrap();
    assert_eq!(observed.alive_counts, [2, 3]);
    assert!((observed.alive_wasserstein_squared.unwrap() - 0.5).abs() < 1e-12);
    let empty = population(&[-1., 0., 1.], &[0.; 3], &[0, 1, 2]);
    let observed = decay_observables(&empty, &right, p, 1.).unwrap();
    assert_eq!(observed.alive_wasserstein_squared, None);
    assert!(observed.full_marked_error.is_finite());
}

#[test]
fn assignment_matches_an_exhaustive_marked_phase_space_oracle() {
    let p = [[1., 0.25], [0.25, 2.]];
    let x = [-1., 0.2, 0.3, 2.];
    let v = [2., -1., 1., 0.];
    let y = [-0.9, 0.4, 0.2, 1.8];
    let w = [1.9, 0.9, -0.8, 0.1];
    let left = population(&x, &v, &[1, 3]);
    let right = population(&y, &w, &[0, 2]);
    let observed = decay_observables(&left, &right, p, 1.).unwrap();
    let mut minimum = f64::INFINITY;
    let mut physical_minimum = f64::INFINITY;
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let permutation = [a, b, c, d];
                    if (0..4).any(|i| (0..i).any(|j| permutation[i] == permutation[j])) {
                        continue;
                    }
                    let physical = permutation
                        .iter()
                        .enumerate()
                        .map(|(i, &j)| {
                            let dx = x[i] - y[j];
                            let dv = v[i] - w[j];
                            dx * dx + 0.5 * dx * dv + 2. * dv * dv
                        })
                        .sum::<f64>()
                        / 4.;
                    let mismatch = permutation
                        .iter()
                        .enumerate()
                        .filter(|(i, j)| [1, 3].contains(i) != [0, 2].contains(j))
                        .count() as f64
                        / 4.;
                    minimum = minimum.min(physical + mismatch);
                    physical_minimum = physical_minimum.min(physical);
                }
            }
        }
    }
    assert!((observed.full_marked_error - minimum).abs() < 1e-12);
    assert!((observed.full_physical_wasserstein_squared - physical_minimum).abs() < 1e-12);
}

#[test]
fn native_controls_decrease_error_with_fixed_n_independent_prediction() {
    let config = DecayValidationConfig {
        walkers: vec![4, 16],
        dimensions: vec![1],
        seeds: vec![7, 173],
        steps: 32,
        fit_end_step: 16,
        profiles: vec![
            DecayProfile::LinearQuadraticNoiseless,
            DecayProfile::LinearSphereNoiseless,
        ],
        ..Default::default()
    };
    let report = futures_lite::future::block_on(validate_decay(&config)).unwrap();
    for case in &report.cases {
        assert!(
            case.checks.iter().all(|check| check.passed),
            "{:?}",
            case.checks
        );
        assert!(case.linear_control_theorem_applicable);
        assert!(case.empirical_rate.estimated_squared_error_rate.unwrap() > 0.);
        let points = &case.ensemble_trajectory;
        assert!(
            points[32].full_marked_error.as_ref().unwrap().mean
                < points[0].full_marked_error.as_ref().unwrap().mean
        );
    }
    let first = &report.cases[0];
    let second = &report.cases[1];
    assert_eq!(
        first.reference_linear_constants.contraction_factor,
        second.reference_linear_constants.contraction_factor
    );
    assert!(
        (first.ensemble_trajectory[32]
            .full_marked_error
            .as_ref()
            .unwrap()
            .mean
            - second.ensemble_trajectory[32]
                .full_marked_error
                .as_ref()
                .unwrap()
                .mean)
            .abs()
            < 1e-12
    );
}

#[test]
fn independent_noise_floor_and_canonical_revival_are_not_hidden() {
    let config = DecayValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7, 173, 1729, 7919, 104729, 31337, 65537, 99991],
        steps: 24,
        fit_end_step: 12,
        profiles: vec![
            DecayProfile::LinearQuadraticIndependentNoise,
            DecayProfile::CanonicalQuadratic,
            DecayProfile::CanonicalRastrigin,
        ],
        ..Default::default()
    };
    let report = futures_lite::future::block_on(validate_decay(&config)).unwrap();
    let control = &report.cases[0];
    assert!(
        control.ensemble_trajectory[24]
            .exact_barycenter_noise_floor
            .unwrap()
            > 0.
    );
    assert!(control.statistical_checks.iter().all(|check| check.status
        != algorithmic_gas_benchmarks::convergence_kinetic::KineticCheckStatus::Violated));
    for case in &report.cases[1..] {
        assert!(!case.linear_control_theorem_applicable);
        assert!(case.total_revivals >= 16);
        assert!(
            case.ensemble_trajectory
                .iter()
                .all(|p| p.exact_barycenter_expectation.is_none())
        );
        assert!(
            case.empirical_rate
                .observable
                .contains("alive empirical physical q")
        );
    }
}

#[test]
fn invalid_decay_axes_and_duplicate_seeds_are_rejected() {
    let config = DecayValidationConfig {
        walkers: vec![65],
        ..Default::default()
    };
    assert!(config.validate().is_err());
    let config = DecayValidationConfig {
        seeds: vec![7, 7],
        ..Default::default()
    };
    assert!(config.validate().is_err());
    let config = DecayValidationConfig {
        fit_end_step: 129,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn diffusion_design_and_baseline_seed_addresses_are_explicit() {
    assert_eq!(diffusion_from_well_spacing(0.04, 1., 0.25).unwrap(), 1.25);
    assert!(diffusion_from_well_spacing(0., 1., 0.25).is_err());
    let config = DecayValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7, 173],
        steps: 4,
        fit_end_step: 2,
        calibration_position_diffusion: 0.1,
        profiles: vec![
            DecayProfile::CanonicalRastrigin,
            DecayProfile::RastriginDiffusionCalibration,
        ],
        ..Default::default()
    };
    let report = futures_lite::future::block_on(validate_decay(&config)).unwrap();
    let baseline = &report.cases[0];
    let calibrated = &report.cases[1];
    assert_eq!(baseline.native_config, calibrated.native_config);
    assert_eq!(
        serde_json::to_value(&baseline.seed_trajectories).unwrap(),
        serde_json::to_value(&calibrated.seed_trajectories).unwrap()
    );
    let mechanism = calibrated.diffusion_calibration.as_ref().unwrap();
    assert_eq!(mechanism.reference_position_diffusion, 1.25);
    assert_eq!(mechanism.selected_position_diffusion, 0.1);
    assert!(!calibrated.linear_control_theorem_applicable);
    let invalid = DecayValidationConfig {
        calibration_position_diffusion: -1.,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn shared_gaussian_control_has_zero_difference_floor_and_fixed_contraction() {
    let config = DecayValidationConfig {
        walkers: vec![4, 16],
        dimensions: vec![1],
        seeds: vec![7, 173, 1729, 7919],
        steps: 16,
        fit_end_step: 8,
        profiles: vec![DecayProfile::LinearQuadraticIndependentNoise],
        pair_randomness: PairRandomness::Shared,
        ..Default::default()
    };
    let report = futures_lite::future::block_on(validate_decay(&config)).unwrap();
    for case in &report.cases {
        assert!(
            case.checks.iter().all(|check| check.passed),
            "{:?}",
            case.checks
        );
        assert!(case.statistical_checks.iter().all(|check| check.status
            != algorithmic_gas_benchmarks::convergence_kinetic::KineticCheckStatus::Violated));
        for point in &case.ensemble_trajectory {
            assert_eq!(point.exact_barycenter_noise_floor, Some(0.));
            assert!(
                (point.full_marked_error.as_ref().unwrap().mean
                    - point.exact_barycenter_expectation.unwrap())
                .abs()
                    < 1e-12
            );
        }
        assert!(
            case.seed_trajectories
                .iter()
                .all(|t| t.native_engine_seeds[0] == t.native_engine_seeds[1])
        );
        assert_eq!(case.endpoint_decline.empirical_mean_decreased, Some(true));
    }
}

#[test]
fn canonical_shared_native_step_is_permutation_invariant_and_preserves_counters() {
    futures_lite::future::block_on(async {
        let mut config = GasConfig::euclidean(1, 0.04).unwrap();
        config.seed = 12345;
        let model = BenchmarkModel {
            benchmark: Benchmark::Rastrigin,
            field: "positions".into(),
            direction: config.fitness.direction,
        };
        let original = population(&[0.7, -0.4, 3., 0.1], &[0.2, -0.1, 0., 0.3], &[2]);
        let reordered = population(&[3., 0.1, 0.7, -0.4], &[0., 0.3, 0.2, -0.1], &[0]);
        let mut left = GasBuilder::new(original, model.clone())
            .gradient(model.clone())
            .config(config.clone())
            .build()
            .await
            .unwrap();
        let mut right = GasBuilder::new(reordered, model.clone())
            .gradient(model)
            .config(config)
            .build()
            .await
            .unwrap();
        for _ in 0..3 {
            let before = left.checkpoint();
            canonicalize_decay_engine(&mut left).unwrap();
            canonicalize_decay_engine(&mut right).unwrap();
            let after = left.checkpoint();
            assert_eq!(before.step, after.step);
            assert_eq!(before.population.version, after.population.version);
            assert_eq!(before.reward_evaluations, after.reward_evaluations);
            assert_eq!(
                serde_json::to_value(before.execution).unwrap(),
                serde_json::to_value(after.execution).unwrap()
            );
            assert_eq!(
                serde_json::to_value(before.population.rewards.provenance).unwrap(),
                serde_json::to_value(after.population.rewards.provenance).unwrap()
            );
            left.step().await.unwrap();
            right.step().await.unwrap();
            let observable = decay_observables(
                left.population(),
                right.population(),
                [[1., 0.], [0., 1.]],
                1.,
            )
            .unwrap();
            assert_eq!(observable.full_marked_error, 0.);
        }
    });
}

#[test]
fn canonical_population_gathers_all_atom_fields_and_tracking_metadata() {
    let mut original = population(&[0.7, -0.4, 3., 0.1], &[0.2, -0.1, 0., 0.3], &[2]);
    original.rewards.raw = vec![7., -4., 30., 1.];
    original.generations = vec![70, 40, 300, 10];
    let canonical = canonical_decay_population(&original).unwrap();
    assert_eq!(
        canonical.observations.field("positions").unwrap().values(),
        &[-0.4, 0.1, 0.7, 3.]
    );
    assert_eq!(
        canonical.observations.field("velocities").unwrap().values(),
        &[-0.1, 0.3, 0.2, 0.]
    );
    assert_eq!(canonical.rewards.raw, vec![-4., 1., 7., 30.]);
    assert_eq!(canonical.generations, vec![40, 10, 70, 300]);
    assert!(canonical.validity[3].invalid);
}

#[test]
fn cloning_calibration_changes_only_copy_jitter_and_retains_baseline_addresses() {
    let config = DecayValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7, 173],
        steps: 2,
        fit_end_step: 2,
        calibration_jitter_amplitude: 0.5,
        profiles: vec![
            DecayProfile::CanonicalRastrigin,
            DecayProfile::RastriginCloningCalibration,
        ],
        pair_randomness: PairRandomness::Shared,
        ..Default::default()
    };
    let report = futures_lite::future::block_on(validate_decay(&config)).unwrap();
    let baseline = &report.cases[0];
    let calibrated = &report.cases[1];
    let mut expected = baseline.native_config.clone();
    expected.clone_transform.jitter_amplitude = 0.5;
    assert_eq!(expected, calibrated.native_config);
    assert_eq!(
        baseline.seed_trajectories[0].native_engine_seeds,
        calibrated.seed_trajectories[0].native_engine_seeds
    );
    assert_eq!(calibrated.native_config.kinetic.position_diffusion, 0.1);
    assert!(calibrated.diffusion_calibration.is_none());
    let mechanism = calibrated.cloning_calibration.as_ref().unwrap();
    assert_eq!(mechanism.baseline_jitter_amplitude, 0.1);
    assert_eq!(mechanism.selected_jitter_amplitude, 0.5);
    assert_eq!(
        mechanism.gaussian_jitter_variance_per_accepted_coordinate,
        Some(0.25)
    );
    assert!(calibrated.total_revivals >= 4);
    let invalid = DecayValidationConfig {
        calibration_jitter_amplitude: f64::NAN,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn matching_dead_storage_atom_does_not_dilute_alive_theory_error_or_endpoint_ratio() {
    let p = [[1., 0.], [0., 1.]];
    let all_alive = decay_observables(
        &population(&[-0.5; 3], &[0.; 3], &[]),
        &population(&[0.5; 3], &[0.; 3], &[]),
        p,
        1.,
    )
    .unwrap();
    let with_matching_dead = decay_observables(
        &population(&[-0.5, -0.5, -0.5, 3.], &[0.; 4], &[3]),
        &population(&[0.5, 0.5, 0.5, 3.], &[0.; 4], &[3]),
        p,
        1.,
    )
    .unwrap();
    assert_eq!(all_alive.alive_wasserstein_squared, Some(1.));
    assert_eq!(with_matching_dead.alive_wasserstein_squared, Some(1.));
    assert_eq!(with_matching_dead.full_marked_error, 0.75);

    let config = DecayValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7, 173],
        steps: 2,
        fit_end_step: 2,
        profiles: vec![DecayProfile::CanonicalRastrigin],
        pair_randomness: PairRandomness::Shared,
        canonical_initial_dead_atom: true,
        ..Default::default()
    };
    let report = futures_lite::future::block_on(validate_decay(&config)).unwrap();
    let mut refreshed = report.clone();
    let original_samples = serde_json::to_value(&report.cases[0].seed_trajectories).unwrap();
    refreshed.cases[0].endpoint_decline.mean_initial_error = Some(-123.);
    refreshed.cases[0].empirical_rate.observable = "stale full storage error".into();
    refreshed.cases[0].terminal_window_mean_error = None;
    refresh_decay_summary(&mut refreshed).unwrap();
    assert_eq!(
        serde_json::to_value(&refreshed).unwrap(),
        serde_json::to_value(&report).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&refreshed.cases[0].seed_trajectories).unwrap(),
        original_samples
    );
    let case = &report.cases[0];
    let endpoint = &case.endpoint_decline;
    let initial = case.ensemble_trajectory[0]
        .alive_error
        .as_ref()
        .unwrap()
        .mean;
    let final_error = case.ensemble_trajectory[2]
        .alive_error
        .as_ref()
        .unwrap()
        .mean;
    let storage_initial = case.ensemble_trajectory[0]
        .full_marked_error
        .as_ref()
        .unwrap()
        .mean;
    assert!((initial - 1.).abs() < 1e-12);
    assert!((storage_initial - 0.75).abs() < 1e-12);
    assert_eq!(endpoint.mean_initial_error, Some(initial));
    assert_eq!(endpoint.mean_final_error, Some(final_error));
    assert_eq!(
        endpoint.terminal_to_initial_ratio,
        Some(final_error / initial)
    );
    assert!(endpoint.observable.contains("alive empirical physical q"));
    assert_eq!(endpoint.requested_seed_pairs, 2);
    assert_eq!(endpoint.complete_seed_pairs, 2);
    assert_eq!(endpoint.missing_endpoint_seed_pairs, 0);
    assert_eq!(endpoint.empty_alive_endpoint_seed_pairs, 0);
    assert_eq!(endpoint.extinct_seed_pairs, 0);
    assert_eq!(case.empirical_rate.complete_window_seed_pairs, 2);
    assert_eq!(case.empirical_rate.missing_window_seed_pairs, 0);
    assert_eq!(case.empirical_rate.empty_alive_window_seed_pairs, 0);
    assert_eq!(
        case.terminal_window_mean_error.as_ref().unwrap().mean,
        (case.ensemble_trajectory[1]
            .alive_error
            .as_ref()
            .unwrap()
            .mean
            + final_error)
            / 2.
    );
}

#[test]
fn refreshed_alive_summaries_count_missing_extinct_and_empty_pairs() {
    let config = DecayValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7, 173],
        steps: 2,
        fit_end_step: 2,
        profiles: vec![DecayProfile::CanonicalQuadratic],
        ..Default::default()
    };
    let report = futures_lite::future::block_on(validate_decay(&config)).unwrap();
    let mut missing = report.clone();
    missing.cases[0].seed_trajectories[0].points.pop();
    missing.cases[0].seed_trajectories[0].termination =
        Some("extinction prevented retained paired update".into());
    refresh_decay_summary(&mut missing).unwrap();
    let case = &missing.cases[0];
    assert_eq!(case.endpoint_decline.requested_seed_pairs, 2);
    assert_eq!(case.endpoint_decline.complete_seed_pairs, 1);
    assert_eq!(case.endpoint_decline.missing_endpoint_seed_pairs, 1);
    assert_eq!(case.endpoint_decline.empty_alive_endpoint_seed_pairs, 0);
    assert_eq!(case.endpoint_decline.extinct_seed_pairs, 1);
    assert_eq!(case.empirical_rate.complete_window_seed_pairs, 1);
    assert_eq!(case.empirical_rate.missing_window_seed_pairs, 1);
    assert_eq!(case.empirical_rate.estimated_squared_error_rate, None);
    assert_eq!(
        case.terminal_window_mean_error
            .as_ref()
            .unwrap()
            .independent_pairs,
        1
    );

    let mut empty = report;
    let stored_full = empty.cases[0].seed_trajectories[0].points[2]
        .observables
        .full_marked_error;
    empty.cases[0].seed_trajectories[0].points[2]
        .observables
        .alive_wasserstein_squared = None;
    refresh_decay_summary(&mut empty).unwrap();
    let case = &empty.cases[0];
    assert_eq!(case.endpoint_decline.complete_seed_pairs, 1);
    assert_eq!(case.endpoint_decline.missing_endpoint_seed_pairs, 0);
    assert_eq!(case.endpoint_decline.empty_alive_endpoint_seed_pairs, 1);
    assert_eq!(case.endpoint_decline.extinct_seed_pairs, 0);
    assert_eq!(case.empirical_rate.complete_window_seed_pairs, 1);
    assert_eq!(case.empirical_rate.empty_alive_window_seed_pairs, 1);
    assert_eq!(case.empirical_rate.estimated_squared_error_rate, None);
    assert_eq!(
        case.seed_trajectories[0].points[2]
            .observables
            .full_marked_error,
        stored_full
    );
}
