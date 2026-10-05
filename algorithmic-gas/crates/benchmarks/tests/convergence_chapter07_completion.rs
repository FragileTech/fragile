use algorithmic_gas_benchmarks::convergence_chapter07_completion::*;
#[test]
fn all_reference_formulas() {
    let rows = reference_suite().unwrap();
    assert!(rows.len() > 700);
    for r in rows {
        assert_eq!(r["passed"], true, "{r}");
    }
}
#[test]
fn reject_missing_hypotheses() {
    assert!(alive_mass(0., 1.).is_err());
    assert!(poisson_reference(0, 1., 1.).is_err());
    assert!(ou_reference(2, 0., 1., 1., 0.).is_err());
    assert!(residual_reference(1., 2., 1., 0.5).is_err());
}
#[test]
fn exact_gamma_special_values() {
    assert!((gamma(0.5).unwrap() - std::f64::consts::PI.sqrt()).abs() < 1e-13);
    assert!((gamma(5.).unwrap() - 24.).abs() < 1e-12);
}
