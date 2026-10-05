use algorithmic_gas_benchmarks::convergence_chapter08_completion::*;

#[test]
fn graph_bound_retains_population_independent_constants_and_logarithms() {
    let a = component_bounds(32., 2., 0.75, 12).unwrap();
    assert!((a.kappa - (-4_f64).exp()).abs() < 1e-15);
    assert_eq!(a.minimum_population, 3);
    for r in 1..12 {
        assert!(
            (a.radius_log_upper[r]
                - a.radius_log_upper[r - 1]
                - (2. * a.edge_constant / r as f64).ln())
            .abs()
                < 1e-10
        );
    }
    assert!(component_bounds(32., 2., 0., 2).is_err());
}
#[test]
fn terminal_probability_uses_each_coordinate_and_actual_noise() {
    let p = alive_probability(&[0.], 1., 1.).unwrap();
    assert!((p - 0.6826894921370859).abs() < 1e-9);
    let q = alive_probability(&[0., 0.], 1., 1.).unwrap();
    assert!((q - p * p).abs() < 1e-14);
    assert!(alive_probability(&[0.5], 1., 1.).unwrap() < p);
    assert!(alive_probability(&[0.], 1., 0.).is_err());
}
#[test]
fn finite_moment_envelope_uses_dimension_and_actual_configuration() {
    let a = finite_moment_budget(1, 4, 0.5, 0.04, 1., 1., 0., 2., 0.5, 0.1, 1., 0.1).unwrap();
    let b = finite_moment_budget(4, 4, 0.5, 0.04, 1., 1., 0., 2., 0.5, 0.1, 1., 0.1).unwrap();
    assert_eq!(a.gaussian_norm_moment, 3.);
    assert_eq!(b.gaussian_norm_moment, 24.);
    assert_eq!(a.coefficient, b.coefficient);
    assert!(b.additive > a.additive);
    assert!(finite_moment_budget(1, 3, 0.5, 0.04, 1., 1., 0., 2., 0.5, 0.1, 1., 0.1).is_err());
}
#[test]
fn zero_variance_does_not_create_unavailable_uncertainty() {
    let (m, v) = mean_variance(&[[1.; 7], [1.; 7]]);
    assert_eq!(m, [1.; 7]);
    assert_eq!(v, [0.; 7]);
    assert!(
        Comparison::identity(
            "native",
            "def-baoab-update-rule",
            1.,
            1.,
            serde_json::json!({})
        )
        .passed
    );
    assert!(
        !Comparison::identity(
            "native",
            "def-baoab-update-rule",
            1.,
            2.,
            serde_json::json!({})
        )
        .passed
    );
}

#[test]
fn shared_rotation_preserves_negative_cross_covariance_per_draw() {
    let report = covariance_experiment(&[vec![-1., 0.], vec![1., 0.]], 0.5, 103, 512).unwrap();
    for sample in report["samples"].as_array().unwrap() {
        let out = sample["centered_output"].as_array().unwrap();
        for (a, b) in out[0]
            .as_array()
            .unwrap()
            .iter()
            .zip(out[1].as_array().unwrap())
        {
            assert!((a.as_f64().unwrap() + b.as_f64().unwrap()).abs() < 1e-14);
        }
    }
    let row = report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == "cross-0-1-0-0")
        .unwrap();
    assert_eq!(row["operands"]["expected"], -0.125);
    assert!(row["operands"]["observed"].as_f64().unwrap() < -0.08);
}

#[test]
fn bounded_marked_observables_are_permutation_invariant_and_n_normalized() {
    use algorithmic_gas::{ObservationBatch, Population, TensorBatch};
    let make = |xs: Vec<f64>| {
        let n = xs.len();
        Population::<f64>::new(ObservationBatch::positions(
            TensorBatch::vectors(n, 1, xs).unwrap(),
        ))
        .unwrap()
    };
    let p = make(vec![-0.3, 0.1, 0.7]);
    let q = make(vec![0.7, -0.3, 0.1]);
    let r = make(vec![-0.3, 0.1, 0.7, -0.3, 0.1, 0.7]);
    let a = bounded_observables(&p).unwrap();
    let b = bounded_observables(&q).unwrap();
    let c = bounded_observables(&r).unwrap();
    for j in 0..7 {
        assert!((a[j] - b[j]).abs() < 1e-14);
        assert!((a[j] - c[j]).abs() < 1e-14);
    }
}
