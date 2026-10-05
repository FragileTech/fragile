use algorithmic_gas_benchmarks::convergence_chapter06_completion::*;

fn parameters(d: usize, p: f64) -> NativeTailParameters {
    NativeTailParameters {
        dimension: d,
        moment_order: p,
        timestep: 0.04,
        friction: 1.,
        harmonic_curvature: 2.,
        residual_per_coordinate: 20. * std::f64::consts::PI,
        clone_jitter_standard_deviation: 0.02,
        ou_standard_deviation: 0.196,
        final_position_standard_deviation: 0.02,
        frozen_velocity_norm_bound: 2.,
        restitution: 0.5,
    }
}

#[test]
fn native_and_theorem_hazards_are_distinct_and_log_survives_underflow() {
    let h = uniform_final_gaussian_box_hazard(2, 2, 0.2, 0.2).unwrap();
    let q = h.per_row_death_lower_float;
    assert!((h.native_hazard_lower_float - q * q).abs() < 1e-14);
    assert!((h.chapter_hazard_lower_float - (2. * q - q * q)).abs() < 1e-14);
    assert!(h.chapter_hazard_lower_float > h.native_hazard_lower_float);
    let rare = uniform_final_gaussian_box_hazard(8, 64, 2., 0.02).unwrap();
    assert!(rare.log_native_zero_alive_hazard_lower.is_finite());
    assert!(rare.log_native_zero_alive_hazard_lower < -300_000.);
    assert_eq!(rare.native_hazard_lower_float, 0.);
    assert!(
        rare.log_chapter_fewer_two_alive_hazard_lower > rare.log_native_zero_alive_hazard_lower
    );
}

#[test]
fn gaussian_norm_moments_include_dimension_for_arbitrary_orders() {
    for d in [1, 2, 4, 8] {
        let second = native_selected_source_tail_interface(&parameters(d, 2.), 1.).unwrap();
        assert!((second.gaussian_norm_lp_upper.powi(2) - d as f64).abs() < 1e-12);
        let fourth = native_selected_source_tail_interface(&parameters(d, 4.), 1.).unwrap();
        assert!((fourth.gaussian_norm_lp_upper.powi(4) - (d * (d + 2)) as f64).abs() < 1e-10);
        let fractional = native_selected_source_tail_interface(&parameters(d, 3.5), 1.).unwrap();
        assert!((fractional.gaussian_norm_lp_upper - fourth.gaussian_norm_lp_upper).abs() < 1e-12);
    }
}

#[test]
fn moment_closure_requires_actual_source_bound_and_can_fail() {
    let interface = native_selected_source_tail_interface(&parameters(2, 4.), 3.).unwrap();
    let conservative = compose_proved_source_moment(
        &interface,
        1.,
        0.,
        "Cloning disabled, all alive: the zero-jitter source array equals input.",
    )
    .unwrap();
    assert!(conservative.closes);
    let unsupported = compose_proved_source_moment(
        &interface,
        2.,
        0.,
        "Declared bound with gain2; it does not absorb the native kinetic remainder.",
    )
    .unwrap();
    assert!(!unsupported.closes);
    assert!(unsupported.invariant_moment_upper.is_none());
    assert!(compose_proved_source_moment(&interface, 1., 0., "").is_err());
}

#[test]
fn normalized_duplication_and_young_bound_do_not_introduce_population_factors() {
    for p in [1., 2., 3.5, 4., 8.] {
        let interface = native_selected_source_tail_interface(&parameters(4, p), 7.).unwrap();
        assert!(
            interface.one_step_moment_upper
                <= interface.source_moment_coefficient * 7.
                    + interface.drift_additive_moment
                    + 1e-10
        );
        // Moment of a replicated empirical swarm is still 7; no N appears.
        let replicated = native_selected_source_tail_interface(&parameters(4, p), 7.).unwrap();
        assert_eq!(
            interface.one_step_moment_upper,
            replicated.one_step_moment_upper
        );
    }
}

#[test]
fn invalid_native_domains_are_rejected() {
    assert!(uniform_final_gaussian_box_hazard(0, 2, 2., 0.02).is_err());
    assert!(uniform_final_gaussian_box_hazard(2, 1, 2., 0.02).is_err());
    let mut p = parameters(2, 0.5);
    assert!(native_selected_source_tail_interface(&p, 1.).is_err());
    p.moment_order = 4.;
    p.restitution = 1.1;
    assert!(native_selected_source_tail_interface(&p, 1.).is_err());
}

#[test]
fn native_displacement_uses_first_kick_and_dimension_not_terminal_cap() {
    use algorithmic_gas_benchmarks::convergence_chapter06_completion::native_prepared_displacement_budget;
    for d in [1, 2, 4, 8] {
        let b =
            native_prepared_displacement_budget(4, 4, d, 0.04, 1., 0.3, 0.02, 100., 75.).unwrap();
        let c = 0.04 * (1. + (-0.04_f64).exp()) / 2.;
        assert!(
            (b.squared_displacement
                - (c * c * 100. + d as f64 * (0.006_f64.powi(2) + 0.02_f64.powi(2))))
            .abs()
                < 1e-14
        );
        assert!(b.squared_displacement > 0.04_f64.powi(2) * 4.);
        let repeated =
            native_prepared_displacement_budget(64, 64, d, 0.04, 1., 0.3, 0.02, 100., 75.).unwrap();
        assert_eq!(b.squared_displacement, repeated.squared_displacement);
        assert!(b.centered_squared_displacement <= repeated.centered_squared_displacement);
        let singleton =
            native_prepared_displacement_budget(64, 1, d, 0.04, 1., 0.3, 0.02, 100., 0.).unwrap();
        assert_eq!(singleton.centered_squared_displacement, 0.);
    }
    assert!(native_prepared_displacement_budget(4, 4, 1, 0.04, 1., 0.3, 0.02, 1., 2.).is_err());
    assert!(native_prepared_displacement_budget(4, 0, 1, 0.04, 1., 0.3, 0.02, 1., 0.).is_err());
}

#[test]
fn state_budget_coefficient_is_not_dropped_from_position_rate() {
    use algorithmic_gas_benchmarks::convergence_chapter06_completion::compose_native_position_budget;
    let (r, b, closes) = compose_native_position_budget(0.1, 0.8, 0.01, 0.02, 0.03).unwrap();
    assert!((r - 1.1).abs() < 1e-14);
    assert!((b - 0.341).abs() < 1e-14);
    assert!(!closes);
    assert!(
        compose_native_position_budget(0.1, 0.8, 0.01, 0.005, 0.03)
            .unwrap()
            .2
    );
    assert!(compose_native_position_budget(0., 0.8, 0., 0., 0.).is_err());
}
