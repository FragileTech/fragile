use algorithmic_gas_benchmarks::convergence_coefficients::{
    CompositeInputs, coefficient_validation_cases, composite_coefficients,
    standardization_coefficients,
};

#[test]
fn native_pipeline_bounds_cover_floor_support_and_exponent_sweeps() {
    let report = coefficient_validation_cases().unwrap();
    assert!(report.checks.len() > 10000);
    assert!(report.computed.len() > 200);
    assert!(!report.unavailable.is_empty());
    for check in report.checks {
        assert!(check.passed, "{check:?}");
    }
}
#[test]
fn coefficient_inputs_reject_invalid_and_overflowing_values() {
    assert!(standardization_coefficients(0, 1, 0, 1., 0.1).is_err());
    assert!(standardization_coefficients(2, 1, 2, 1., 0.1).is_err());
    assert!(standardization_coefficients(2, 1, 1, 1., 0.).is_err());
    assert!(standardization_coefficients(2, 1, 1, f64::NAN, 0.1).is_err());
    assert!(standardization_coefficients(2, 1, 1, 1., 1e-200).is_err());
    let mut input = CompositeInputs {
        walkers: 8,
        alive1: 5,
        diameter: 4.,
        status_penalty: 1.,
        death_lipschitz: 0.7,
        boundary_exponent: 0.5,
        perturbation_moment_squared: 0.02,
        failure_probability: 0.05,
        potential_error_coefficients: [0.4, 0.2, 0.1, 0.03],
        clone_value_lipschitz: 0.6,
    };
    assert!(composite_coefficients(&input).is_ok());
    input.failure_probability = 1.;
    assert!(composite_coefficients(&input).is_err());
    input.failure_probability = 0.05;
    input.boundary_exponent = 1.1;
    assert!(composite_coefficients(&input).is_err());
    input.boundary_exponent = 0.5;
    input.potential_error_coefficients[0] = -1.;
    assert!(composite_coefficients(&input).is_err());
}
#[test]
fn empirical_value_coefficients_are_population_independent() {
    let small = standardization_coefficients(2, 1, 1, 1., 0.1).unwrap();
    let large = standardization_coefficients(128, 64, 64, 1., 0.1).unwrap();
    assert!((small.value_total - large.value_total).abs() < 1e-6);
    assert!((small.value_direct - 100.).abs() < 1e-12);
    assert!((small.value_scale - 16_000_000.).abs() < 1e-7);
}

#[test]
fn sasaki_structural_coefficient_uses_second_alive_count_and_its_own_moduli() {
    // Vmax=1 and floor=1: chapter01 mu_S=2, sigma_S=3, while
    // chapter02 k_min=1 gives mu_S=3 and sigma_S=4.5.
    let death = standardization_coefficients(2, 1, 1, 1., 1.).unwrap();
    assert_eq!(death.structural_indirect_split, 152.);
    assert_eq!(death.mean_structural_lipschitz_sasaki, 3.);
    assert_eq!(death.second_structural_lipschitz_sasaki, 3.);
    assert_eq!(death.scale_structural_lipschitz_sasaki, 4.5);
    assert_eq!(death.structural_indirect_split_sasaki, 180.);
    let birth = standardization_coefficients(1, 2, 1, 1., 1.).unwrap();
    assert_eq!(birth.structural_indirect_split, 20.);
    assert_eq!(birth.structural_indirect_split_sasaki, 342.);
}

#[test]
fn sasaki_actual_component_and_composite_checks_cover_unequal_alive_counts() {
    let report = coefficient_validation_cases().unwrap();
    for id in [
        "sasaki_structural_orthogonal_decomposition",
        "sasaki_direct_structural_error_bound",
        "sasaki_indirect_structural_error_bound",
        "sasaki_structural_total_bound",
        "sasaki_composite_standardization_bound",
    ] {
        let cases = report
            .checks
            .iter()
            .filter(|c| c.id == id)
            .collect::<Vec<_>>();
        assert!(!cases.is_empty(), "{id}");
        assert!(cases.iter().all(|c| c.passed), "{id}");
        assert!(
            cases.iter().any(|c| c.scope.contains("k1=2,k2=1")),
            "{id}: no death fixture"
        );
        assert!(
            cases.iter().any(|c| c.scope.contains("k1=1,k2=2")),
            "{id}: no birth fixture"
        );
    }
    for id in [
        "sasaki_direct_structural_error_bound",
        "sasaki_indirect_structural_error_bound",
    ] {
        assert!(
            report
                .checks
                .iter()
                .any(|c| c.id == id && c.observed > 1e-4),
            "{id}: only zero errors tested"
        );
    }
    assert!(
        report
            .computed
            .iter()
            .any(|c| c.name == "structural_indirect_split_sasaki")
    );
    let chapter = include_str!(
        "../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"
    );
    for check in report.checks.iter().filter(|c| c.id.starts_with("sasaki_")) {
        for label in &check.source_labels {
            assert!(chapter.contains(&format!(":label: {label}")), "{label}");
        }
    }
}
