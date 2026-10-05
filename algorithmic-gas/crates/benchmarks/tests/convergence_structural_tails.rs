use algorithmic_gas_benchmarks::convergence_structural_tails::{
    CoordinateInterval, baoab_position_center, coupled_tail_cost_upper, gaussian_region_budget,
    gaussian_tail_budget, moment_tail_budget, nonquadratic_gaussian_example,
};

#[test]
fn gaussian_landing_accounts_for_dimension_and_unbounded_noise() {
    for d in [1, 2, 4, 8] {
        let centers = vec![
            CoordinateInterval {
                lower: -0.1,
                upper: 0.1
            };
            d
        ];
        let target = vec![
            CoordinateInterval {
                lower: -1.,
                upper: 1.
            };
            d
        ];
        let result = gaussian_region_budget(&centers, &target, 0.2, 0.2).unwrap();
        let union = 2. * d as f64 * (-0.5_f64 * (0.9_f64 / 0.2).powi(2)).exp();
        assert!(result.landing_lower >= 1. - union - 1e-15);
        assert!(result.escape_upper <= union * (1. + 1e-14));
        assert!(result.escape_upper > 0.);
        assert_eq!(result.dimension, d);
    }
    let extreme = gaussian_tail_budget(8, 0., 0.01, 100., 2.).unwrap();
    assert!(extreme.escape_upper > 0.);
    assert!(extreme.log_escape_upper < -1_000_000.);
    assert!(gaussian_region_budget(&[], &[], 1., 1.).is_err());
}

#[test]
fn gaussian_weighted_tail_bounds_the_exact_one_dimensional_gaussian_integral() {
    // Deterministic quadrature of the actual unbounded Gaussian, independent of
    // the chi-square exponential-envelope implementation.
    let bins = 200_000;
    let step = 24. / bins as f64;
    for cutoff in [0.1, 1., 2., 4.] {
        for r in [0., 0.5, 1., 2., 3., 4.] {
            let bound = gaussian_tail_budget(1, 0., 1., cutoff, r).unwrap();
            let mut actual = 0.;
            for i in 0..bins {
                let x = -12. + (i as f64 + 0.5) * step;
                if x.abs() > cutoff {
                    actual += x.abs().powf(r) * (-0.5 * x * x).exp()
                        / (2. * std::f64::consts::PI).sqrt()
                        * step;
                }
            }
            assert!(
                actual <= bound.tail_cost_upper + 1e-7,
                "cutoff={cutoff}, r={r}: {actual} > {}",
                bound.tail_cost_upper
            );
        }
    }
}

#[test]
fn coupled_moment_tail_handles_correlated_samples_and_normalization() {
    let x = [0.1_f64, 2., 5., -4.];
    let y = [6.0_f64, 0.2, -2., 1.];
    let mx = x.iter().map(|v| v.powi(4)).sum::<f64>() / 4.;
    let my = y.iter().map(|v| v.powi(4)).sum::<f64>() / 4.;
    let bound = coupled_tail_cost_upper(4., mx, my, 3., 2., 1., 0.5).unwrap();
    let observed: f64 = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.abs().max(b.abs()) > 3.)
        .map(|(a, b)| (1. + 0.5 * (a.powi(2) + b.powi(2))) / 4.)
        .sum();
    assert!(observed <= bound);
    let repeated_mx = x.iter().cycle().take(64).map(|v| v.powi(4)).sum::<f64>() / 64.;
    // Duplication leaves the exact normalized moment unchanged. The two f64
    // summation orders differ by at most a few ulps, including the subsequent
    // log/exp envelope evaluation; the mathematical formula is unchanged.
    let roundoff = 8. * f64::EPSILON;
    assert!((mx - repeated_mx).abs() <= roundoff * mx.abs().max(1.));
    let original = moment_tail_budget(4., mx, 3., 2.).unwrap().tail_cost_upper;
    let repeated = moment_tail_budget(4., repeated_mx, 3., 2.)
        .unwrap()
        .tail_cost_upper;
    assert!((original - repeated).abs() <= roundoff * original.abs().max(1.));
    assert!(moment_tail_budget(2., mx, 3., 2.).is_err());
}

#[test]
fn nonquadratic_force_centers_and_region_curvature_are_distinct() {
    let d = 4;
    let example = nonquadratic_gaussian_example(d).unwrap();
    let regions = example["regions"].as_array().unwrap();
    assert!((regions[0]["local_hessian"].as_f64().unwrap() - 0.2).abs() < 1e-14);
    assert!((regions[1]["local_hessian"].as_f64().unwrap() - 1.8).abs() < 1e-14);
    assert!((example["force_perturbation_norm_upper"].as_f64().unwrap() - 0.8).abs() < 1e-14);
    let center =
        baoab_position_center(&[0.3], &[0.4], &[0.3 - 0.4 * (0.6_f64).sin()], 0.04, 1.).unwrap();
    let harmonic = baoab_position_center(&[0.3], &[0.4], &[0.3], 0.04, 1.).unwrap();
    assert!(center[0] > harmonic[0]);
    for region in regions {
        assert!(region["gaussian_tail"]["escape_upper"].as_f64().unwrap() > 0.);
        assert!(region["landing"]["landing_lower"].as_f64().unwrap() > 0.999);
    }
}
