use algorithmic_gas_benchmarks::convergence_chapter12_completion::*;
#[test]
fn exact_collision_and_population_normalization() {
    assert_eq!(collision_probability(8, 1).unwrap(), 0.);
    assert!((collision_probability(8, 2).unwrap() - 0.125).abs() < 1e-14);
    for n in [8, 32, 128] {
        for k in 1..=n {
            assert!(collision_probability(n, k).unwrap() <= union_bound(n, k).unwrap() + 1e-14);
        }
    }
    assert!(collision_probability(0, 1).is_err());
}
#[test]
fn covariance_retains_diagonal() {
    let v = distinct_covariance(8, 0.125, 1.).unwrap();
    assert_eq!(v, 0.);
    assert_eq!(distinct_covariance(8, 1., 1.).unwrap(), 1.);
    assert!(distinct_covariance(1, 0., 0.).is_err());
}
#[test]
fn bounded_tilt_exact_entropy_and_covariance() {
    for n in [2, 8, 32, 128] {
        let x = tilted_bernoulli(n, 0.3, 1.).unwrap();
        let f = |k: &str| x[k].as_f64().unwrap();
        assert!(
            (distinct_covariance(n, f("empirical_variance"), f("diagonal_covariance")).unwrap()
                - f("distinct_covariance"))
            .abs()
                < 1e-12
        );
        assert!(f("mse") <= entropy_mse_bound(n, 1., f("total_relative_entropy")).unwrap());
        assert!(f("total_relative_entropy") <= 1.);
        assert!(f("product_square_exponential_moment") <= 2_f64.sqrt());
    }
}
#[test]
fn gaussian_constant_independent_of_population() {
    assert_eq!(gaussian_lsi_constant(0.5, 0.25).unwrap(), 2.);
    assert!(gaussian_lsi_constant(0., 1.).is_err());
    assert!(entropy_mse_bound(8, 1., -1.).is_err());
}
