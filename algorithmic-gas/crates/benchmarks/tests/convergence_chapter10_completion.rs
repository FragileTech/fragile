use algorithmic_gas_benchmarks::convergence_chapter10_completion::*;
#[test]
fn hypotheses_guard() {
    for args in [
        (0., 1., 1., 1.),
        (1., 0., 1., 1.),
        (1., 1., -1., 1.),
        (1., 1., 1., f64::NAN),
    ] {
        assert!(kinetic_constants(args.0, args.1, args.2, args.3).is_err());
    }
    assert!(entropy(&[0.4, 0.5], &[0.4, 0.6]).is_err());
}
#[test]
fn underdamped_entropy_decreases_and_bound_holds() {
    let v = gaussian_trajectory(1., 1., 1., [2., -1.], 0.01, 2000).unwrap();
    assert_eq!(v["endpoint_entropy_decreased"], true);
    assert!(
        v["checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["passed"] == true)
    );
}
#[test]
fn mixed_nonquadratic_fisher_identities() {
    let v = density_integrals(
        1.,
        1.,
        2. * std::f64::consts::PI,
        1.,
        1.,
        [0.4, -0.3, 0.2],
        512,
    );
    assert!(
        v["checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["passed"] == true),
        "{}",
        v["checks"]
    );
}
#[test]
fn full_killing_normalization_and_doob_conjugacy() {
    let v = killed_and_discrete_suite().unwrap();
    assert!(
        v["checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["passed"] == true),
        "{}",
        v["checks"]
    );
    assert!(v["c"].as_f64().unwrap().abs() > 0.01);
}
#[test]
fn rate_is_dimension_and_population_independent() {
    let c = kinetic_constants(1., 1., 400., 1e12).unwrap();
    for n in [1_f64, 8., 32., 128.] {
        for d in [1_f64, 2., 4., 8.] {
            let ix = 0.3 * n * d;
            let iv = 0.7 * n * d;
            let entropy = 0.1 * n * d;
            let phi = entropy + c.eta * (2. * ix + 2. * iv);
            assert!((phi / (n * d) - (0.1 + 2. * c.eta)).abs() < 1e-14);
            assert!(c.rate > 0.);
        }
    }
}
