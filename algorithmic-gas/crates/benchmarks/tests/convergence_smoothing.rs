use algorithmic_gas_benchmarks::convergence_smoothing::{
    SmoothingInputs, ball_interval_death_probability, heat_interval_death_probability,
    interval_swarm_diagnostic, log_unit_ball_volume, smoothing_constants,
    smoothing_validation_cases,
};
use std::f64::consts::PI;

fn inputs() -> SmoothingInputs {
    SmoothingInputs {
        dimensions: 2,
        sigma: 0.1,
        perimeter: 1.,
        inverse_distance_factor: 9.,
        walkers: 16,
    }
}

#[test]
fn normalized_kernel_constants_have_dimensional_scaling_and_explicit_inverse_factor() {
    let mut input = inputs();
    let first = smoothing_constants(&input).unwrap();
    assert!((first.values["uniform_ball_universal"] - 10.).abs() < 1e-12);
    assert!((first.values["uniform_ball_perimeter"] - 100. / PI).abs() < 1e-12);
    assert!((first.values["heat_perimeter"] - 25. / PI).abs() < 1e-12);
    assert!(
        (first.values["heat_matched_pair"] / first.values["heat_pointwise"] - 36.).abs() < 1e-12
    );
    assert!(
        (first.values["heat_empirical_mean"] / first.values["heat_pointwise"] - 9.).abs() < 1e-12
    );
    input.sigma *= 10.;
    let second = smoothing_constants(&input).unwrap();
    assert!((first.values["heat_universal"] / second.values["heat_universal"] - 10.).abs() < 1e-12);
    assert!(
        (first.values["heat_perimeter"] / second.values["heat_perimeter"] - 100.).abs() < 1e-10
    );
    assert!((log_unit_ball_volume(3).unwrap().exp() - 4. * PI / 3.).abs() < 1e-12);
}

#[test]
fn logs_preserve_density_and_metric_constants_beyond_linear_representability() {
    let mut input = inputs();
    input.dimensions = 256;
    input.sigma = 10.;
    let constants = smoothing_constants(&input).unwrap();
    assert!(!constants.values.contains_key("heat_density_max"));
    assert!(constants.natural_logs["heat_density_max"] < -745.);
    assert!(constants.natural_logs["heat_perimeter"].is_finite());
    input.sigma = 1e-300;
    input.inverse_distance_factor = 1e300;
    let constants = smoothing_constants(&input).unwrap();
    assert!(!constants.values.contains_key("heat_matched_pair"));
    assert!(constants.natural_logs["heat_matched_pair"] > 700.);
    let extreme = interval_swarm_diagnostic(&[0.74], &[0.76], 0.25, 2., 1e-320).unwrap();
    assert!(extreme.checks.iter().all(|check| check.passed));
    assert!(
        extreme
            .checks
            .iter()
            .any(|check| check.id.ends_with("log1p_comparison"))
    );
}

#[test]
fn exact_interval_laws_use_heat_variance_two_sigma_squared_and_uniform_overlap() {
    let death = heat_interval_death_probability(-1., 1., 0., 1.).unwrap();
    // erf(1/2), independently tabulated to fifteen decimal places.
    assert!((death - 0.4795001221869535).abs() < 2e-13);
    assert!((heat_interval_death_probability(-100., 100., 0., 1.).unwrap()).abs() < 2e-13);
    assert_eq!(
        heat_interval_death_probability(-1., 1., 100., 1.).unwrap(),
        1.
    );
    assert_eq!(
        ball_interval_death_probability(-1., 1., 0., 0.5).unwrap(),
        0.
    );
    assert_eq!(
        ball_interval_death_probability(-1., 1., 1., 1.).unwrap(),
        0.5
    );
    assert_eq!(
        ball_interval_death_probability(-1., 1., 3., 1.).unwrap(),
        1.
    );
}

#[test]
fn invalid_parameters_are_rejected_and_zero_perimeter_has_zero_branches() {
    let mut input = inputs();
    for invalid in [0., -1., f64::NAN, f64::INFINITY] {
        input.sigma = invalid;
        assert!(smoothing_constants(&input).is_err());
    }
    input = inputs();
    input.inverse_distance_factor = 0.;
    assert!(smoothing_constants(&input).is_err());
    input = inputs();
    input.walkers = 0;
    assert!(smoothing_constants(&input).is_err());
    input = inputs();
    input.perimeter = -1.;
    assert!(smoothing_constants(&input).is_err());
    input.perimeter = 0.;
    let constants = smoothing_constants(&input).unwrap();
    assert_eq!(constants.values["heat_pointwise"], 0.);
    assert_eq!(constants.values["uniform_ball_matched_pair"], 0.);
    assert!(!constants.natural_logs.contains_key("heat_pointwise"));
    assert!(heat_interval_death_probability(1., -1., 0., 1.).is_err());
    assert!(ball_interval_death_probability(-1., 1., f64::NAN, 1.).is_err());
}

#[test]
fn swarm_metric_transfer_is_invariant_to_independent_storage_permutations() {
    let left = [0.74, -0.8, 0.1, 1.];
    let right = [0.75, -0.8, 0.1, 1.];
    let original = interval_swarm_diagnostic(&left, &right, 0.25, 2., 0.1).unwrap();
    let permuted = interval_swarm_diagnostic(
        &[1., 0.1, -0.8, 0.74],
        &[0.1, 0.75, 1., -0.8],
        0.25,
        2.,
        0.1,
    )
    .unwrap();
    assert_eq!(original.observations, permuted.observations);
    assert!(original.checks.iter().all(|check| check.passed));
    let identical =
        interval_swarm_diagnostic(&left, &[1., 0.1, -0.8, 0.74], 0.25, 2., 0.1).unwrap();
    assert_eq!(identical.observations["d_N"], 0.);
    assert_eq!(
        identical.observations["heat_max_matched_probability_change"],
        0.
    );
    assert_eq!(original.constants.values["H"], 81.);
    assert_eq!(original.constants.values["sqrt_N"], 2.);
}

#[test]
fn all_scale_families_and_directional_integral_checks_pass() {
    let report = smoothing_validation_cases().unwrap();
    assert_eq!(report.fixtures.len(), 60);
    assert_eq!(report.swarm_cases.len(), 12);
    let failed: Vec<_> = report
        .checks
        .iter()
        .chain(report.swarm_cases.iter().flat_map(|case| &case.checks))
        .filter(|check| !check.passed)
        .collect();
    assert!(failed.is_empty(), "{failed:?}");
    assert!(
        report
            .fixtures
            .iter()
            .any(|fixture| fixture.constants.values["heat_perimeter"]
                < fixture.constants.values["heat_universal"])
    );
    assert!(
        report
            .fixtures
            .iter()
            .any(|fixture| fixture.constants.values["heat_perimeter"]
                > fixture.constants.values["heat_universal"])
    );
}
