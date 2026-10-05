use algorithmic_gas_benchmarks::convergence_chapter05_remaining::*;

#[test]
fn finite_horizon_bound_controls_ballistic_transient_without_equilibrium_substitution() {
    // x=(-2,2), v=(-3,3): the complete centered position increment is exact.
    let h = 0.1;
    let observed = (2_f64 + 3. * h).powi(2) - 4.;
    let mv = transient_velocity_envelope(9., 2., 2.).unwrap();
    assert_eq!(mv, 9.);
    let correct = finite_horizon_position_bound(4., mv, h, 0.5).unwrap();
    assert!((correct["exact_increment_bound"].as_f64().unwrap() - observed).abs() < 1e-14);
    assert!(observed <= correct["horizon_increment_bound"].as_f64().unwrap());
    let invalid_equilibrium_only = finite_horizon_position_bound(4., 1., h, 0.5).unwrap();
    assert!(
        observed
            > invalid_equilibrium_only["horizon_increment_bound"]
                .as_f64()
                .unwrap()
    );
    assert!(finite_horizon_position_bound(4., 9., 0.6, 0.5).is_err());
}
#[test]
fn native_conditional_moments_match_exact_degree_two_gaussian_cubature() {
    let x = [-0.2, 0.7, 0.4, -0.1];
    let v = [0.6, -0.3, 0.2, 0.5];
    let expected = native_position_moments(&x, &v, 2, 0.04, 1., 0.3, 0.02).unwrap();
    let m: Vec<f64> = serde_json::from_value(expected["mean_positions"].clone()).unwrap();
    let sd = expected["tau_squared"].as_f64().unwrap().sqrt();
    let mut total = 0.;
    let mut variance = 0.;
    let mut bary = 0.;
    // Eight nodes integrate any quadratic of four independent Gaussian coordinates.
    for coordinate in 0..4 {
        for sign in [-1., 1.] {
            let mut output = m.clone();
            output[coordinate] += sign * 2. * sd;
            let moments = second_moments(&output, 2).unwrap();
            total += moments["total"].as_f64().unwrap() / 8.;
            variance += moments["variance"].as_f64().unwrap() / 8.;
            bary += moments["barycenter"].as_f64().unwrap() / 8.;
        }
    }
    for (actual, key) in [
        (total, "total_prediction"),
        (variance, "variance_prediction"),
        (bary, "barycenter_prediction"),
    ] {
        assert!((actual - expected[key].as_f64().unwrap()).abs() < 1e-14);
    }
    assert!(
        expected["conditional_increment"].as_f64().unwrap()
            <= expected["conditional_increment_bound"].as_f64().unwrap()
    );
    let doubled_x = [x, x].concat();
    let doubled_v = [v, v].concat();
    let doubled = native_position_moments(&doubled_x, &doubled_v, 2, 0.04, 1., 0.3, 0.02).unwrap();
    assert_eq!(expected["total_prediction"], doubled["total_prediction"]);
    assert_eq!(
        expected["uniform_variance_noise_source"],
        doubled["uniform_variance_noise_source"]
    );
    let px = [0.4, -0.1, -0.2, 0.7];
    let pv = [0.2, 0.5, 0.6, -0.3];
    let permuted = native_position_moments(&px, &pv, 2, 0.04, 1., 0.3, 0.02).unwrap();
    assert!((permuted["variance_prediction"].as_f64().unwrap() - variance).abs() < 1e-14);
}
#[test]
fn continuum_generator_keeps_force_work_noise_and_cap_transfer_gate() {
    let x = [-2., 2.];
    let v = [-3., 3.];
    let f = [2., -2.];
    let r = component_generators(&x, &v, &f, 1, 1., 1., 1., true).unwrap();
    assert_eq!(r["velocity_generator"].as_f64().unwrap(), -29.5);
    assert_eq!(r["barycenter_generator"].as_f64().unwrap(), 0.5);
    assert_eq!(r["position_generator"].as_f64().unwrap(), 12.);
    assert_eq!(r["native_continuum_transfer_applicable"], false);
    assert!(r["velocity_generator"].as_f64().unwrap() <= r["velocity_bound"].as_f64().unwrap());
    assert!(r["barycenter_generator"].as_f64().unwrap() <= r["barycenter_bound"].as_f64().unwrap());
    assert!(component_generators(&x, &v, &f, 1, 1., 1., 2., true).is_err());
}
#[test]
fn joint_minorization_requires_position_noise_and_nonsingular_second_kick() {
    let p = CompactKineticParameters {
        dimension: 1,
        timestep: 0.04,
        friction: 1.,
        ou_standard_deviation: 0.2,
        position_standard_deviation: 0.02,
        velocity_cap: 2.,
        force_lipschitz: 1.,
        force_at_origin: 0.,
        input_position_radius: 0.1,
        input_velocity_radius: 0.2,
        target_position_radius: 0.05,
        target_velocity_radius: 0.1,
        target_center_norm: 0.,
    };
    let bound = compact_kinetic_minorization(&p).unwrap();
    for x in [-0.1, 0., 0.1] {
        for v in [-0.2, 0., 0.2] {
            for xx in [-0.05, 0., 0.05] {
                for vv in [-0.1, 0., 0.1] {
                    let density = quadratic_output_log_density(
                        &[x],
                        &[v],
                        &[xx],
                        &[vv],
                        p.timestep,
                        p.friction,
                        1.,
                        p.ou_standard_deviation,
                        p.position_standard_deviation,
                        p.velocity_cap,
                    )
                    .unwrap();
                    assert!(density >= bound["log_density_lower"].as_f64().unwrap() - 1e-12);
                }
            }
        }
    }
    let mut degenerate = p.clone();
    degenerate.position_standard_deviation = 0.;
    assert!(compact_kinetic_minorization(&degenerate).is_err());
    degenerate = p;
    degenerate.timestep = 2.;
    assert!(compact_kinetic_minorization(&degenerate).is_err());
    assert!(
        quadratic_output_log_density(&[0.], &[0.], &[0.], &[0.], 2., 1., 1., 0.2, 0.02, 2.)
            .is_err()
    );
}
