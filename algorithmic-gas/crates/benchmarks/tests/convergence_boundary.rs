use algorithmic_gas_benchmarks::convergence_boundary::{
    ResetInputs, logarithmic_barrier, reset_constants, validate_boundary,
};

fn inputs() -> ResetInputs {
    ResetInputs {
        lower: vec![-2.; 2],
        upper: vec![2.; 2],
        dt: 0.04,
        friction: 1.,
        jitter: 0.1,
        velocity_cap: 2.,
        restitution: 0.5,
        force_lipschitz: 1.,
        force_offset: 0.,
        thermostat_scale: 1.,
        position_diffusion: 0.1,
        velocity_weight: 1.,
    }
}

#[test]
fn log_barrier_integral_and_reset_constants_are_explicit() {
    let mut p = inputs();
    let c = reset_constants(&p).unwrap();
    assert!((c["barrier_L1"] - 64.).abs() < 1e-12);
    assert_eq!(c["R_D_squared"], 8.);
    assert!((c["M"] - (1. + c["M_x"] + 4.)).abs() < 1e-12);
    let factor_product = c["terminal_density_max"] * c["barrier_L1"];
    assert!((c["M_b"] / factor_product - 1.).abs() < 1e-12);
    p.friction = 0.;
    assert_eq!(reset_constants(&p).unwrap()["s_h_squared"], p.dt);
    p.position_diffusion = 0.;
    assert!(reset_constants(&p).is_err());
    p = inputs();
    p.upper[0] = p.lower[0];
    assert!(reset_constants(&p).is_err());
}

#[test]
fn representable_boundary_product_survives_density_factor_underflow() {
    let mut p = inputs();
    p.lower = vec![-2.; 256];
    p.upper = vec![2.; 256];
    p.dt = 1.;
    p.position_diffusion = 10.;
    let c = reset_constants(&p).unwrap();
    assert!(!c.contains_key("terminal_density_max"));
    assert!(c["log_terminal_density_max"] < -745.);
    assert!((c["M_b"] / 4.673094797152113e-202 - 1.).abs() < 1e-10);
    assert_eq!(c["M_b"], c["log_M_b"].exp());
}

#[test]
fn boundary_observable_uses_an_integrable_log_and_terminal_eligibility() {
    let lower = [-2., -2.];
    let upper = [2., 2.];
    let center = logarithmic_barrier(&[0., 0.], &lower, &upper).unwrap();
    assert!((center - 2. * 4_f64.ln()).abs() < 1e-12);
    assert_eq!(logarithmic_barrier(&[2., 0.], &lower, &upper).unwrap(), 0.);
    assert_eq!(logarithmic_barrier(&[3., 0.], &lower, &upper).unwrap(), 0.);
    assert!(logarithmic_barrier(&[1.999999, 0.], &lower, &upper).unwrap() > center);
    assert!(logarithmic_barrier(&[f64::NAN, 0.], &lower, &upper).is_err());
}

#[test]
fn complete_native_reset_overwrites_unrestricted_dead_positions_before_kinetics() {
    let report = futures_lite::future::block_on(validate_boundary(128, 5_900_000)).unwrap();
    assert_eq!(report.ensembles.len(), 5);
    assert!(report.checks.iter().all(|check| check.passed));
    for ensemble in &report.ensembles {
        assert!(ensemble.checks.iter().all(|check| check.passed));
        assert!(
            ensemble
                .estimates
                .iter()
                .all(|estimate| estimate.status == "not_rejected"),
            "{:?}",
            ensemble.estimates
        );
        assert!(ensemble.entering_dead_coordinate_magnitude > 100.);
    }
    assert_eq!(report.ensembles[0].constants, report.ensembles[1].constants);
    assert_eq!(report.ensembles[4].entering_alive, 1);
    assert!(futures_lite::future::block_on(validate_boundary(1, 0)).is_err());
}
