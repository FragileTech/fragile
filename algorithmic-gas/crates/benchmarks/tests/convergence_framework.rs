use algorithmic_gas_benchmarks::convergence_framework::{
    hermite_patch, stable_distance_regressions, validate_framework,
};

#[test]
fn framework_bounds_hold_for_population_and_floor_sweep() {
    for n in [2, 3, 8, 32] {
        for d in [1, 2, 5] {
            for floor in [0.01, 0.1, 1.] {
                let report = validate_framework(n, d, floor, 1729).unwrap();
                for check in report.checks {
                    assert!(check.passed, "{check:?}");
                }
            }
        }
    }
}

#[test]
fn hermite_coefficients_endpoints_and_extrema_are_computed_from_fixed_zmax() {
    for zmax in [1.000001, 1.1, 2., 10., 100.] {
        let patch = hermite_patch(zmax).unwrap();
        let [a, b, c, d] = patch.coefficients;
        assert!((patch.normalized_endpoint_matrix_determinant - 1.).abs() < 1e-14);
        for (formula, solved) in patch
            .coefficients
            .iter()
            .zip(patch.independently_solved_coefficients)
        {
            assert!((formula - solved).abs() < 1e-12);
        }
        assert!((d - zmax.ln() - 1.).abs() < 1e-12);
        assert!((c - 1. / zmax).abs() < 1e-12);
        assert!((a + b + c + d - zmax.ln_1p() - 1.).abs() < 1e-12);
        assert!((3. * a + 2. * b + c).abs() < 1e-12);
        assert!((0. ..=1.).contains(&patch.derivative_maximizer));
        assert!(patch.exact_derivative_minimum >= -1e-12);
        assert!(patch.exact_rescale_lipschitz <= patch.uniform_derivative_bound + 1e-12);
        for step in 0..=1000 {
            let s = step as f64 / 1000.;
            let derivative = (3. * a * s + 2. * b) * s + c;
            assert!(derivative <= patch.exact_derivative_maximum + 1e-12);
        }
    }
    let large = hermite_patch(10.).unwrap();
    assert!(large.exact_derivative_maximum < 1.);
    assert_eq!(large.exact_rescale_lipschitz, 1.);
    let near_one = hermite_patch(1.000001).unwrap();
    assert!(near_one.exact_rescale_lipschitz > 1.);
    assert!(near_one.exact_rescale_lipschitz < near_one.uniform_derivative_bound);
    for zmax in [1e100, 1e308] {
        let patch = hermite_patch(zmax).unwrap();
        for (formula, solved) in patch
            .coefficients
            .iter()
            .zip(patch.independently_solved_coefficients)
        {
            assert!(
                ((formula - solved) / formula).abs() < 1e-12,
                "zmax={zmax}, formula={formula}, solved={solved}"
            );
        }
        assert!(patch.derivative_vertex.is_finite());
        assert!(patch.exact_derivative_maximum > patch.coefficients[2]);
        assert_eq!(patch.exact_rescale_lipschitz, 1.);
    }
    assert!(hermite_patch(1.).is_err());
    assert!(hermite_patch(f64::INFINITY).is_err());
    assert!(hermite_patch(f64::NAN).is_err());
}

#[test]
fn repaired_sasaki_bound_handles_changed_measurement_support() {
    let checks = stable_distance_regressions().unwrap();
    assert!((checks[0].observed - 1. / 18.).abs() < 1e-14);
    for check in checks {
        assert!(check.passed);
        assert!(check.observed > 0.);
    }
}

#[test]
fn invalid_or_overflowing_framework_parameters_are_rejected() {
    assert!(validate_framework(1, 2, 0.1, 7).is_err());
    assert!(validate_framework(4, 0, 0.1, 7).is_err());
    assert!(validate_framework(4, 2, f64::NAN, 7).is_err());
    assert!(validate_framework(4, 2, 1e-200, 7).is_err());
}
