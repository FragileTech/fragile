use algorithmic_gas_benchmarks::convergence_chapter11_completion::*;
#[test]
fn mass_shape_and_entropy_handle_zero_without_false_smoothing() {
    assert_eq!(hellinger_squared(&[1., 0.], &[0., 1.]).unwrap(), 2.);
    assert_eq!(
        relative_entropy(&[1., 0.], &[0., 1.]).unwrap(),
        f64::INFINITY
    );
    assert!(hellinger_squared(&[-1., 1.], &[1., 0.]).is_err());
    assert!(relative_entropy(&[0.5], &[1.]).is_err());
}
#[test]
fn exact_transport_is_permutation_and_duplication_invariant() {
    let x = [-1., 0., 2.];
    let p = [0.25, 0.5, 0.25];
    let y = [0., 1.];
    let q = [0.5, 0.5];
    let w = wasserstein_1d_squared(&x, &p, &y, &q).unwrap();
    assert!((w - 0.75).abs() < 1e-12);
    assert_eq!(
        w,
        wasserstein_1d_squared(&[2., -1., 0.], &[0.25, 0.25, 0.5], &y, &q).unwrap()
    );
    assert_eq!(
        w,
        wasserstein_1d_squared(
            &[-1., -1., 0., 0., 2., 2.],
            &[0.125, 0.125, 0.25, 0.25, 0.125, 0.125],
            &y,
            &q
        )
        .unwrap()
    );
}
#[test]
fn heterogeneous_survival_keeps_exact_probability_law_and_n_scaling() {
    let law = poisson_binomial(&[0.1, 0.5, 0.9]).unwrap();
    assert!((law.iter().sum::<f64>() - 1.).abs() < 1e-14);
    assert!((law[0] - 0.045).abs() < 1e-14);
    assert!((law[3] - 0.045).abs() < 1e-14);
    assert!(poisson_binomial(&[1.1]).is_err());
}
#[test]
fn gaussian_dimension_enters_moments_without_a_population_factor() {
    let a = gaussian_curve(1, 1., 0.5, 1., 1., 0.7).unwrap();
    let b = gaussian_curve(4, 1., 0.5, 1., 4., 0.7).unwrap();
    assert!((b["entropy"].as_f64().unwrap() - 4. * a["entropy"].as_f64().unwrap()).abs() < 1e-14);
    assert!(
        (a["wasserstein_squared"].as_f64().unwrap() - 2. * a["entropy"].as_f64().unwrap()).abs()
            < 1e-14
    );
}
#[test]
fn all_reference_estimates_have_matched_operands_and_pass() {
    let rows = reference_suite().unwrap();
    assert!(rows.len() > 5000);
    for row in rows {
        assert_eq!(row["passed"], true, "{row}");
        assert!(row["operands"].is_object());
    }
}
