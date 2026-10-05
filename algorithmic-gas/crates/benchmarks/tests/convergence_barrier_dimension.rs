use algorithmic_gas_benchmarks::convergence_barrier_dimension::{
    dimension_barrier_bounds, independent_seed_allowance, prepared_barrier_bounds,
    prepared_swarm_barrier_bounds,
};
#[test]
fn linear_and_layer_cake_bounds_track_dimension_without_population() {
    let base = dimension_barrier_bounds(1, 2., 0.02).unwrap();
    assert!((base.linear_bound - 48.9666254140098).abs() < 1e-10);
    assert!((base.sharp_bound - 4.689371503696138).abs() < 1e-12);
    for d in [1, 2, 4, 8] {
        let row = dimension_barrier_bounds(d, 2., 0.02).unwrap();
        assert!((row.linear_bound - d as f64 * base.linear_bound).abs() < 1e-10);
        assert!((row.sharp_bound - d as f64 * base.sharp_bound).abs() < 1e-10);
        assert!(row.sharp_bound <= row.old_bound.unwrap());
    }
    assert!(
        (dimension_barrier_bounds(2, 2., 0.02).unwrap().linear_bound - 97.9332508280196).abs()
            < 1e-10
    );
}
#[test]
fn high_noise_keeps_coordinate_survival_and_small_box_scaling() {
    let first = dimension_barrier_bounds(1, 0.2, 0.2).unwrap();
    for d in [1, 2, 4, 8] {
        let bound = dimension_barrier_bounds(d, 0.2, 0.2).unwrap();
        let expected = d as f64 * first.sharp_bound * first.density_ratio.powf((d - 1) as f64);
        assert!((bound.sharp_bound - expected).abs() < 1e-12);
        assert!((bound.sharp_bound - bound.old_bound.unwrap()).abs() < 1e-12);
    }
}
#[test]
fn state_aware_bounds_handle_unbounded_means_and_permutations() {
    let near = prepared_barrier_bounds(&[0., 0.1], 2., 0.02).unwrap();
    let swapped = prepared_barrier_bounds(&[0.1, 0.], 2., 0.02).unwrap();
    assert!((near.mean_upper - swapped.mean_upper).abs() < 1e-14);
    assert!(near.mean_upper < 0.02);
    assert!(near.second_moment_upper < 0.001);
    let far = prepared_barrier_bounds(&[100., -100.], 2., 0.02).unwrap();
    assert!(far.mean_upper < 1e-100);
    assert!(far.mean_upper > 0.);
    let underflow = dimension_barrier_bounds(4096, 0.001, 1.).unwrap();
    assert!(underflow.log_sharp_bound < -1000.);
    assert!(underflow.sharp_bound_positive_floor_applied);
    assert!(underflow.sharp_bound > 0.);
    let repeated = prepared_swarm_barrier_bounds(&vec![vec![0., 0.1]; 64], 2., 0.02).unwrap();
    let single = prepared_swarm_barrier_bounds(&[vec![0., 0.1]], 2., 0.02).unwrap();
    assert!((repeated.0 - single.0).abs() < 1e-14);
    assert!((repeated.1 - single.1).abs() < 1e-14);
}
#[test]
fn midpoint_integration_is_inside_analytic_mean_and_second_moment_envelopes() {
    // Deterministic independent numerical check; the certificate itself is analytic.
    for mu in [0., 0.3, 1.8, 2., 2.4] {
        let bound = prepared_barrier_bounds(&[mu], 2., 0.2).unwrap();
        let bins = 200_000;
        let dx = 4. / bins as f64;
        let mut first = 0.;
        let mut second = 0.;
        for j in 0..bins {
            let x = -2. + (j as f64 + 0.5) * dx;
            let phi = -(1. - (x / 2.).powi(2)).ln();
            let density =
                (-0.5 * ((x - mu) / 0.2).powi(2)).exp() / (2. * std::f64::consts::PI).sqrt() / 0.2;
            first += phi * density * dx;
            second += phi * phi * density * dx;
        }
        assert!(
            first <= bound.mean_upper + 1e-10,
            "mu={mu}: {first} > {}",
            bound.mean_upper
        );
        assert!(second <= bound.second_moment_upper + 1e-10);
    }
}
#[test]
fn cantelli_uses_independent_swarm_seeds_and_rejects_missing_gaussian_hypotheses() {
    let a = independent_seed_allowance(32., 32, 0.05).unwrap();
    let b = independent_seed_allowance(128., 128, 0.05).unwrap();
    assert!((a / b - 2.).abs() < 1e-12);
    assert!(dimension_barrier_bounds(0, 2., 0.02).is_err());
    assert!(dimension_barrier_bounds(2, 2., 0.).is_err());
    assert!(prepared_barrier_bounds(&[f64::NAN], 2., 0.02).is_err());
    assert!(prepared_swarm_barrier_bounds(&[], 2., 0.02).is_err());
    assert!(independent_seed_allowance(1., 0, 0.05).is_err());
}
