use algorithmic_gas_benchmarks::convergence_structural_rates::{
    RegionalErrorBudget, nonquadratic_force_budget, polynomial_tail, rastrigin_force_defect,
};

fn budget() -> RegionalErrorBudget {
    RegionalErrorBudget {
        dimensions: 4,
        dt: 0.04,
        block_steps: 2,
        regions: vec!["core".into(), "slow".into(), "tail".into()],
        transfer: vec![
            vec![0.70, 0.02, 0.],
            vec![0.01, 0.96, 0.01],
            vec![0., 0.005, 0.90],
        ],
        defects: vec![0., 0., 0.001],
        weights: vec![1., 2., 4.],
        analytic_references: vec![],
    }
}
#[test]
fn slow_zone_controls_the_rate_and_missing_proofs_gate_it() {
    let base = budget().compile().unwrap();
    assert!((base["rho_upper"].as_f64().unwrap() - 0.98).abs() < 1e-14);
    assert_eq!(
        base["certificate_status"],
        "diagnostic_only_missing_regional_proofs"
    );
    let mut slow = budget();
    slow.transfer[1][1] = 0.999;
    assert!(!slow.compile().unwrap()["contracts"].as_bool().unwrap());
    let mut different_dimension = budget();
    different_dimension.dimensions = 8;
    assert_eq!(
        different_dimension.compile().unwrap()["rho_upper"],
        base["rho_upper"]
    );
}

#[test]
fn positive_weights_recover_a_valid_rate_with_the_metric_conversion() {
    let mut b = budget();
    b.regions = vec!["core".into(), "slow".into()];
    b.transfer = vec![vec![0.5, 0.01], vec![0.8, 0.9]];
    b.weights = vec![1., 1.];
    b.defects = vec![0., 0.001];
    assert!(!b.compile().unwrap()["contracts"].as_bool().unwrap());
    let result = b.optimize_weights(256).unwrap();
    let optimized = &result["optimized"];
    assert!(optimized["contracts"].as_bool().unwrap());
    let rho = optimized["rho_upper"].as_f64().unwrap();
    // Independent characteristic equation, rather than replaying the iteration.
    let spectral_radius = (1.4_f64 + (0.4_f64.powi(2) + 4. * 0.008).sqrt()) / 2.;
    assert!(rho >= spectral_radius && rho - spectral_radius < 1e-10);
    assert!(optimized["unweighted_prefactor_upper"].as_f64().unwrap() > 1.9);
    assert_eq!(
        optimized["certificate_status"],
        "diagnostic_only_missing_regional_proofs"
    );
    assert!(b.optimize_weights(100_001).is_err());
}
#[test]
fn tail_growth_and_cutoff_change_the_error_floor() {
    let a = polynomial_tail(8., 256., 4., 2.).unwrap();
    let b = polynomial_tail(8., 256., 8., 2.).unwrap();
    assert!(
        (a["tail_cost_upper"].as_f64().unwrap() / b["tail_cost_upper"].as_f64().unwrap() - 64.)
            .abs()
            < 1e-12
    );
    assert!(polynomial_tail(2., 1., 4., 2.).is_err());
    let mut bigger_tail = budget();
    bigger_tail.defects[2] *= 2.;
    assert!(
        (bigger_tail.compile().unwrap()["error_floor_upper"]
            .as_f64()
            .unwrap()
            / budget().compile().unwrap()["error_floor_upper"]
                .as_f64()
                .unwrap()
            - 2.)
            .abs()
            < 1e-12
    );
}
#[test]
fn nonquadratic_budget_preserves_harmonic_limit_and_dimension_cost() {
    let harmonic = nonquadratic_force_budget(0.04, 1., 2., 0.04, 0.00289, 0., 0.).unwrap();
    assert_eq!(harmonic["additive_defect_upper"], 0.);
    let a = rastrigin_force_defect(1, None).unwrap();
    let b = rastrigin_force_defect(4, None).unwrap();
    let d1 = a["residual_difference_upper"].as_f64().unwrap();
    let d4 = b["residual_difference_upper"].as_f64().unwrap();
    let qa = nonquadratic_force_budget(0.04, 1., 2., 0.04, 0.00289, d1, d1).unwrap();
    let qb = nonquadratic_force_budget(0.04, 1., 2., 0.04, 0.00289, d4, d4).unwrap();
    assert!(
        (qb["error_floor_upper"].as_f64().unwrap() / qa["error_floor_upper"].as_f64().unwrap()
            - 4.)
            .abs()
            < 1e-12
    );
    assert_eq!(qa["rho_upper"], qb["rho_upper"]);
    assert!(
        rastrigin_force_defect(1, Some(0.001)).unwrap()["residual_difference_upper"]
            .as_f64()
            .unwrap()
            < d1
    );
}
#[test]
fn invalid_signed_or_overflowing_budget_is_rejected() {
    let mut b = budget();
    b.transfer[0][0] = -0.1;
    assert!(b.compile().is_err());
    let mut b = budget();
    b.weights[2] = f64::MAX;
    b.transfer[2][0] = 2.;
    assert!(b.compile().is_err());
    assert!(nonquadratic_force_budget(0.04, 1., 2., 1., 0.00289, 1., 1.).is_err());
}

#[test]
fn regional_well_modulus_closes_only_inside_the_certified_region() {
    use algorithmic_gas_benchmarks::convergence_structural_rates::local_force_rate_metric;
    let omega = 2. + 40. * std::f64::consts::PI.powi(2);
    let alpha = 1. - omega * 0.02f64.powi(2);
    // Conservative independently checked harmonic endpoint deficit for this metric.
    let beta = 0.025;
    let delta = 0.038;
    let rate = |radius: f64| {
        let modulus =
            40. * std::f64::consts::PI.powi(2) * (1. - (std::f64::consts::TAU * radius).cos());
        local_force_rate_metric(0.04, 1., omega, alpha, beta, delta, modulus).unwrap()
    };
    assert!(rate(0.01)["contracts"].as_bool().unwrap());
    assert!(!rate(0.15)["contracts"].as_bool().unwrap());
    assert!(local_force_rate_metric(0.04, 1., omega, beta * beta, beta, delta, 0.).is_err());
}
