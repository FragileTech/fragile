use algorithmic_gas_benchmarks::convergence_structural_mixing::*;
fn fixtures(d: usize) -> (KineticParameters, SelectionParameters, KernelHypotheses) {
    let k = KineticParameters {
        dimension: d,
        h: 1.,
        friction: 0.,
        velocity_noise: 1.,
        position_noise: 1.,
        velocity_cap: 1.,
        collision_alpha: 0.5,
        position_minorization_radius: 1.,
        velocity_minorization_radius: 1.,
    };
    let channel = FitnessChannel {
        floor: 1.,
        amplitude: 1.,
        regularizer: 1.,
        exponent: 0.,
    };
    let s = SelectionParameters {
        reward: channel.clone(),
        diversity: channel,
        comparison_feature_diameter: 8_f64.sqrt(),
        measurement_scale: 4.,
        cloning_scale: 4.,
        distance_regularizer: 0.001,
        cloning_gate_scale: 1.,
        cloning_gate_floor: 1.,
    };
    let h = KernelHypotheses {
        conservative_all_alive: true,
        current_frame: true,
        nonviscous: true,
        actual_sampled_global_normalization: true,
        normalized_gaussian_companions: true,
        simultaneous_copying_and_component_haar_collision: true,
        actual_radial_velocity_cap: true,
        profiles_proved_for_configured_force_and_reward: true,
    };
    (k, s, h)
}
#[test]
fn rastrigin_center_profile_and_hypothesis_gates_are_actual_parameter_dependent() {
    let (mut k, s, mut hypotheses) = fixtures(2);
    let p = rastrigin_profiles(&k);
    let center = p.discrete_center_bound.unwrap();
    // The center is attained at x_j=1/4; no sampled finite maximum substitutes for a supremum.
    let x = 0.25;
    let force = -2. * x - 20. * std::f64::consts::PI * (2. * std::f64::consts::PI * x).sin();
    assert!(((2_f64).sqrt() * (x + 0.5 * force).abs() - center).abs() < 1e-13);
    assert!((p.force_lipschitz - (2. + 40. * std::f64::consts::PI.powi(2))).abs() < 1e-13);
    k.h = 0.04;
    k.friction = 1.;
    let invalid = evaluate(&k, &s, &rastrigin_profiles(&k), &hypotheses).unwrap();
    assert_eq!(invalid["applicable"], false);
    assert_eq!(invalid["qw"], serde_json::Value::Null);
    k.h = 1.;
    k.friction = 0.;
    hypotheses.conservative_all_alive = false;
    let invalid = evaluate(&k, &s, &rastrigin_profiles(&k), &hypotheses).unwrap();
    assert_eq!(invalid["raw_reward"]["applicable"], false);
    assert_eq!(
        invalid["raw_reward"]["rate_per_physical_time"],
        serde_json::Value::Null
    );
}
#[test]
fn source_minorization_and_weighted_register_preserve_underflow_without_positive_fake_q() {
    for d in [1, 2, 4, 8] {
        let (k, s, h) = fixtures(d);
        let p = rastrigin_profiles(&k);
        let report = evaluate(&k, &s, &p, &h).unwrap();
        assert!(report["kinetic"]["log_epsilon2"].as_f64().unwrap() < -745.);
        assert_eq!(report["kinetic"]["epsilon2"], serde_json::Value::Null);
        assert_eq!(report["raw_reward"]["coefficient"], serde_json::Value::Null);
        assert_eq!(
            report["raw_reward"]["rate_per_physical_time"],
            serde_json::Value::Null
        );
        assert!(
            report["raw_reward"]["log_rate_lower"]
                .as_f64()
                .unwrap()
                .is_finite()
        );
        assert_eq!(report["bounded_reward"]["applicable"], false);
    }
}
#[test]
fn normalization_feedback_gate_rejects_unproved_global_contraction() {
    let (k, mut s, h) = fixtures(1);
    s.reward.exponent = 0.01;
    s.diversity.exponent = 0.01;
    let report = evaluate(&k, &s, &rastrigin_profiles(&k), &h).unwrap();
    assert_eq!(report["raw_reward"]["applicable"], false);
    assert_eq!(report["raw_reward"]["coefficient"], serde_json::Value::Null);
    assert!(
        report["raw_reward"]["log_feedback"].as_f64().unwrap()
            > report["raw_reward"]["log_gain"].as_f64().unwrap()
    );
    // A declared finite profile with no nonconvex envelope recovers the scalar
    // two-step source formula independently of any recorded trajectory.
    let p = LandscapeProfiles {
        discrete_center_bound: Some(0.),
        force_lipschitz: 2.,
        bounded_reward_oscillation: Some(1.),
        raw_reward_quadratic_growth: Some(1.),
        derivation: "F=-2x,eta=.5 exact cancellation".into(),
    };
    s.reward.exponent = 0.;
    s.diversity.exponent = 0.;
    let r = evaluate(&k, &s, &p, &h).unwrap();
    let r0 = 12_f64.sqrt();
    let r1 = r0 / 2. + 0.5;
    let mv = 1. + r0;
    let q = 2. * (1. + r1);
    let expected = (3_f64 / 4.).ln() + 2. * 2_f64.ln()
        - std::f64::consts::TAU.ln()
        - (q + mv).powi(2) / 2.
        - 1.5_f64.ln()
        - (1. + r1 + q / 2.).powi(2) / 2.;
    assert!((r["kinetic"]["log_epsilon2"].as_f64().unwrap() - expected).abs() < 1e-12);
    assert_eq!(r["bounded_reward"]["applicable"], true);
    assert_eq!(r["raw_reward"]["applicable"], true);
    s.reward.exponent = 1e-100;
    s.diversity.exponent = 1e-100;
    let with_feedback = evaluate(&k, &s, &p, &h).unwrap();
    assert!(with_feedback["selection"]["a_star"].as_f64().unwrap() > 0.);
    assert_eq!(with_feedback["bounded_reward"]["applicable"], true);
    assert_eq!(with_feedback["raw_reward"]["applicable"], true);
    assert!(
        with_feedback["raw_reward"]["log_feedback"]
            .as_f64()
            .unwrap()
            < with_feedback["raw_reward"]["log_gain"].as_f64().unwrap()
    );

    let gain = expected.exp();
    assert!(
        (r["bounded_reward"]["rate_per_physical_time"]
            .as_f64()
            .unwrap()
            / (gain / 2.)
            - 1.)
            .abs()
            < 1e-12
    );
    assert!(
        (r["raw_reward"]["rate_per_physical_time"].as_f64().unwrap() / (gain / 4.) - 1.).abs()
            < 1e-12
    );
}
