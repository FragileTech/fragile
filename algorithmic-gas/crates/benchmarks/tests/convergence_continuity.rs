use algorithmic_gas::{ObservationBatch, TensorBatch, geometry::Distance, noise::InnovationLaw};
use algorithmic_gas_benchmarks::convergence_continuity::{
    continuity_validation_cases, finite_line_transport, fixed_factor_moments, generic_revival_case,
    generic_revival_validation_cases, optimal_swarm_displacement, perturbation_coefficients,
    population_scaling_validation_cases, scalar_empirical_transport_squared,
};

#[test]
fn finite_support_transport_noise_and_continuity_fixtures_pass() {
    let report = continuity_validation_cases().unwrap();
    assert!(report.checks.len() > 20_000);
    assert!(report.computed.len() > 400);
    assert_eq!(report.noise_experiments.len(), 33);
    assert_eq!(report.tail_experiments.len(), 48);
    for id in [
        "single_walker_fixed_law_positional_error",
        "single_walker_changed_support_structural_error",
        "single_walker_own_status_error",
        "sasaki_stable_full_sum_with_support_change",
    ] {
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.id == id && check.passed),
            "{id}"
        );
    }
    for experiment in &report.noise_experiments {
        assert_eq!(experiment.status, "not_rejected", "{experiment:?}");
    }
    for experiment in &report.tail_experiments {
        assert_eq!(experiment.status, "not_rejected", "{experiment:?}");
    }
    for check in report.checks {
        assert!(check.passed, "{check:?}");
    }
}

#[test]
fn normalized_empirical_measurement_laws_have_uniform_constants_and_population_decay() {
    let report = population_scaling_validation_cases().unwrap();
    assert_eq!(report.cases.len(), 10);
    for check in &report.checks {
        assert!(check.passed, "{check:?}");
    }
    for id in [
        "empirical_fixed_support_sharp_lipschitz",
        "empirical_arbitrary_support_uniform_four",
        "empirical_unequal_alive_law_sharp_lipschitz",
        "empirical_standardized_law_permutation_invariance",
        "scalar_unequal_empirical_quantile_vs_general_transport",
    ] {
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.id == id && check.passed),
            "{id}"
        );
    }
    for case in &report.cases {
        assert!((case.empirical_lipschitz_squared - 100.).abs() < 1e-12);
        assert_eq!(case.universal_squared_error_bound, 4.);
        assert!(case.exact_standardized_normalized_empirical_error <= 4.);
        assert!(
            case.exact_standardized_normalized_empirical_error
                <= case.empirical_lipschitz_squared * case.exact_raw_normalized_empirical_error
                    + 1e-12
        );
        assert!(case.exact_raw_error_to_expected_law <= case.raw_expected_law_decay_bound);
        assert!(
            case.exact_standardized_error_to_expected_law
                <= case.standardized_expected_law_decay_bound
        );
        assert!(case.raw_sample_standard_error > 0.);
        assert!(case.standardized_sample_standard_error > 0.);
        assert_eq!(case.status, "not_rejected", "{case:?}");
        assert!(
            (case.observed_raw_normalized_empirical_error
                - case.exact_raw_normalized_empirical_error)
                .abs()
                <= case.raw_hoeffding_margin
        );
        assert!(
            (case.observed_standardized_normalized_empirical_error
                - case.exact_standardized_normalized_empirical_error)
                .abs()
                <= case.standardized_hoeffding_margin
        );
        assert_eq!(
            case.measurement_baseline.expected_operator_squared_error,
            0.
        );
        assert!(
            case.measurement_baseline
                .independent_raw_variance_contribution
                > 0.
        );
    }
    for law in ["uniform", "canonical_gaussian_width_2"] {
        let cases: Vec<_> = report
            .cases
            .iter()
            .filter(|case| case.companion_law == law)
            .collect();
        assert_eq!(cases.len(), 5);
        let first = cases[0];
        let last = cases[4];
        assert!(
            last.exact_raw_normalized_empirical_error < first.exact_raw_normalized_empirical_error
        );
        assert!(
            last.exact_standardized_normalized_empirical_error
                < first.exact_standardized_normalized_empirical_error
        );
        assert!(last.exact_raw_error_to_expected_law < first.exact_raw_error_to_expected_law);
        assert!(
            last.exact_standardized_error_to_expected_law
                < first.exact_standardized_error_to_expected_law
        );
        assert_eq!(
            last.raw_expected_law_decay_bound,
            first.raw_expected_law_decay_bound / 4.
        );
        assert_eq!(
            last.standardized_expected_law_decay_bound,
            first.standardized_expected_law_decay_bound / 4.
        );
        assert!(
            last.measurement_baseline.exact_raw_summed_mean_square
                > 10. * first.measurement_baseline.exact_raw_summed_mean_square
        );
    }
}

#[test]
fn generic_revival_keeps_strict_applicability_separate_from_uniform_acceptance() {
    let report = generic_revival_validation_cases().unwrap();
    let sufficient = &report.cases[0];
    assert!((sufficient.revival_ratio.unwrap() - 2.5).abs() < 1e-12);
    assert!(sufficient.strict_condition_holds);
    assert_eq!(sufficient.exact_uniform_acceptance, Some(1.));
    assert_eq!(sufficient.status, "applicable");
    let insufficient = &report.cases[1];
    assert!(!insufficient.strict_condition_holds);
    assert_eq!(insufficient.exact_uniform_acceptance, Some(0.25));
    assert_eq!(insufficient.status, "not_applicable");
    let borderline = &report.cases[2];
    assert!(!borderline.strict_condition_holds);
    assert_eq!(borderline.revival_ratio, Some(1.));
    assert_eq!(borderline.exact_uniform_acceptance, Some(1.));
    assert_eq!(borderline.status, "not_applicable");
    assert!(report.cases[3].fitness_floor.is_none());
    assert!(report.cases[3].log_fitness_floor.is_finite());
    assert!(report.cases[3].revival_ratio.is_some());
    assert!(report.cases[4].revival_ratio.is_none());
    assert!(report.cases[4].strict_condition_holds);
    assert_eq!(report.cases[4].exact_uniform_acceptance, Some(1.));
    assert!(report.checks.iter().all(|check| check.passed));
    assert!(generic_revival_case("bad", 0., 1., 1., 0.1, 1.).is_err());
    assert!(generic_revival_case("bad", 0.1, 0., 0., 0.1, 1.).is_err());
    assert!(generic_revival_case("bad", 0.1, -1., 2., 0.1, 1.).is_err());
    assert!(generic_revival_case("bad", 0.1, 1., 1., 0., 1.).is_err());
}

#[test]
fn finite_transport_computes_optimal_and_common_part_couplings_separately() {
    let result = finite_line_transport(&[0., 1., 2.], &[0.5, 0.5, 0.], &[0., 0.5, 0.5]).unwrap();
    assert!((result.wasserstein_squared - 1.).abs() < 1e-12);
    assert!((result.common_part_coupling_cost - 2.).abs() < 1e-12);
    assert!((result.total_variation - 0.5).abs() < 1e-12);
    assert!(result.maximum_marginal_residual < 1e-12);
    assert!(finite_line_transport(&[1., 0.], &[0.5, 0.5], &[0.5, 0.5]).is_err());
    assert!(finite_line_transport(&[0., 1.], &[1., 1.], &[0.5, 0.5]).is_err());
    assert!(finite_line_transport(&[0., 1.], &[-0.1, 1.1], &[0.5, 0.5]).is_err());
}

#[test]
fn factor_moments_account_for_native_uniform_kurtosis() {
    let factor = [0.2, 0.1, -0.3, 0.4];
    let gaussian = fixed_factor_moments(2, 2, &factor, InnovationLaw::Gaussian).unwrap();
    let uniform = fixed_factor_moments(2, 2, &factor, InnovationLaw::StandardizedUniform).unwrap();
    assert!((gaussian.expected_squared_norm - 0.3).abs() < 1e-12);
    assert!((gaussian.covariance_trace_squared - 0.0658).abs() < 1e-12);
    assert!((gaussian.squared_norm_variance - 0.1316).abs() < 1e-12);
    assert!((uniform.squared_norm_variance - 0.07664).abs() < 1e-12);
    assert!(fixed_factor_moments(2, 0, &[], InnovationLaw::Gaussian).is_err());
    assert!(fixed_factor_moments(2, 2, &[1.; 3], InnovationLaw::Gaussian).is_err());
    assert!(fixed_factor_moments(1, 1, &[f64::MAX], InnovationLaw::Gaussian).is_err());
}

#[test]
fn concentration_constants_reject_invalid_or_unrepresentable_inputs() {
    let c = perturbation_coefficients(8, 2., 0.5, 0.05).unwrap();
    assert_eq!(c.coordinate_bounded_difference, 0.5);
    assert_eq!(c.sum_bounded_differences_squared, 2.);
    assert_eq!(c.mean_bound, 4.);
    assert!((c.fluctuation_bound - 4. * (4. * 40_f64.ln()).sqrt()).abs() < 1e-12);
    assert!(perturbation_coefficients(0, 2., 0.5, 0.05).is_err());
    assert!(perturbation_coefficients(8, 0., 0.5, 0.05).is_err());
    assert!(perturbation_coefficients(8, 2., -0.5, 0.05).is_err());
    assert!(perturbation_coefficients(8, 2., 0.5, 1.).is_err());
    assert!(perturbation_coefficients(8, f64::MAX, 0.5, 0.05).is_err());
    assert!(perturbation_coefficients(8, f64::MIN_POSITIVE, 0., 0.05).is_err());
    assert!(perturbation_coefficients(8, 2., 0.5, f64::MIN_POSITIVE).is_ok());
}

#[test]
fn empirical_swarm_displacement_ignores_independent_storage_permutations() {
    let points = [0., 1., 2.];
    let alive = [true, false, true];
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for squashed in [false, true] {
        let metric = if squashed {
            Distance::SquashedPhaseSpace {
                positions: "positions".into(),
                velocities: "velocities".into(),
                position_radius: 1.,
                velocity_radius: 1.,
                lambda: 0.,
            }
        } else {
            Distance::default()
        };
        for order1 in permutations {
            for order2 in permutations {
                let observe = |order: [usize; 3]| {
                    let mut obs = ObservationBatch::positions(
                        TensorBatch::vectors(3, 1, order.iter().map(|i| points[*i]).collect())
                            .unwrap(),
                    );
                    obs.fields.insert(
                        "velocities".into(),
                        TensorBatch::vectors(3, 1, vec![0.; 3]).unwrap(),
                    );
                    obs
                };
                let marks1: Vec<bool> = order1.iter().map(|i| alive[*i]).collect();
                let marks2: Vec<bool> = order2.iter().map(|i| alive[*i]).collect();
                let result = optimal_swarm_displacement(
                    &metric,
                    &observe(order1),
                    &observe(order2),
                    &marks1,
                    &marks2,
                    2.,
                )
                .unwrap();
                assert!(result.metric_squared.abs() < 1e-12, "{result:?}");
                assert!(result.positional_sum.abs() < 1e-12);
                assert_eq!(result.status_mismatch_count, 0.);
                assert!(result.metric_squared <= result.candidate_coupling_cost + 1e-12);
            }
        }
    }
}

#[test]
fn marked_transport_tie_components_use_a_storage_independent_optimizer() {
    let observe =
        |points: Vec<f64>| ObservationBatch::positions(TensorBatch::vectors(2, 1, points).unwrap());
    let reference = optimal_swarm_displacement(
        &Distance::default(),
        &observe(vec![0., 1.]),
        &observe(vec![0., 1.]),
        &[true, false],
        &[false, true],
        1.,
    )
    .unwrap();
    assert!((reference.metric_squared - 1.).abs() < 1e-12);
    for swap_left in [false, true] {
        for swap_right in [false, true] {
            let (left, left_alive) = if swap_left {
                (vec![1., 0.], [false, true])
            } else {
                (vec![0., 1.], [true, false])
            };
            let (right, right_alive) = if swap_right {
                (vec![1., 0.], [true, false])
            } else {
                (vec![0., 1.], [false, true])
            };
            let result = optimal_swarm_displacement(
                &Distance::default(),
                &observe(left),
                &observe(right),
                &left_alive,
                &right_alive,
                1.,
            )
            .unwrap();
            assert!((result.metric_squared - reference.metric_squared).abs() < 1e-12);
            assert!((result.positional_sum - reference.positional_sum).abs() < 1e-12);
            assert_eq!(
                result.status_mismatch_count,
                reference.status_mismatch_count
            );
            assert!(
                (result.metric_squared
                    - (result.positional_sum + result.status_mismatch_count) / 2.)
                    .abs()
                    < 1e-12
            );
        }
    }
    let squared_metric = Distance::Euclidean {
        field: "positions".into(),
        scales: vec![],
        squared: true,
        periodic: None,
    };
    let result = optimal_swarm_displacement(
        &squared_metric,
        &observe(vec![0., 2.]),
        &observe(vec![2., 4.]),
        &[true; 2],
        &[true; 2],
        1.,
    )
    .unwrap();
    assert!((result.metric_squared - 4.).abs() < 1e-12);
    assert!(
        optimal_swarm_displacement(
            &squared_metric,
            &observe(vec![0., 2.]),
            &observe(vec![2., 4.]),
            &[true; 2],
            &[true; 2],
            0.
        )
        .is_err()
    );
}

#[test]
fn scalar_empirical_transport_handles_unequal_counts_and_independent_permutations() {
    let expected = scalar_empirical_transport_squared(&[0., 1.], &[-0.2, 0.4, 1.3]).unwrap();
    assert!(
        (expected - scalar_empirical_transport_squared(&[1., 0.], &[1.3, -0.2, 0.4]).unwrap())
            .abs()
            < 1e-12
    );
    assert_eq!(
        scalar_empirical_transport_squared(&[0., 0., 1.], &[1., 0., 0.]).unwrap(),
        0.
    );
    assert!(scalar_empirical_transport_squared(&[], &[0.]).is_err());
    assert!(scalar_empirical_transport_squared(&[f64::NAN], &[0.]).is_err());
}
