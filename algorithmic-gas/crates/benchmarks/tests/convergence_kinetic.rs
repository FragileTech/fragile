use algorithmic_gas_benchmarks::convergence_kinetic::{
    ConfiningLandscape, KineticCheckStatus, KineticInput, KineticValidationConfig,
    default_validation_cases, geometry_reward_constants, kinetic_constants,
    quadratic_nondeception_bound, quadratic_uniform_ball_variance, validate_kinetic,
};

#[test]
fn constants_cover_zero_friction_and_the_critical_timestep() {
    let mut config = KineticValidationConfig {
        friction: 0.,
        ..Default::default()
    };
    let constants = kinetic_constants(&config).unwrap();
    assert_eq!(constants.friction_factor, 1.);
    assert_eq!(constants.thermal_variance, config.dt);
    assert_eq!(constants.position_flow_coefficient, config.dt);
    assert!(constants.phase_space_nondegeneracy_hypotheses_hold);
    config.friction = 1e-12;
    let limiting = kinetic_constants(&config).unwrap();
    assert!((limiting.thermal_variance / constants.thermal_variance - 1.).abs() < 1e-12);

    config.dt = 2.;
    config.friction = 1.;
    let critical = kinetic_constants(&config).unwrap();
    assert_eq!(critical.phase_space_invertibility_ratio, 1.);
    assert!(!critical.phase_space_nondegeneracy_hypotheses_hold);
    assert_eq!(critical.positional_condition_number, Some(1.));

    config.velocity_diffusion = 0.;
    config.position_diffusion = 0.;
    let deterministic = kinetic_constants(&config).unwrap();
    assert_eq!(deterministic.position_variance, 0.);
    assert_eq!(deterministic.death_probability_lipschitz, None);
    assert_eq!(deterministic.positional_condition_number, None);
}

#[test]
fn nonconvex_landscape_has_an_analytic_gradient_and_global_confining_bounds() {
    let landscape = ConfiningLandscape {
        name: "anisotropic_cosine".into(),
        curvature: vec![0.5, 4., 1.],
        center: vec![0.3, -0.2, 0.4],
        ripple_amplitude: 0.4,
        ripple_frequency: 3.,
    };
    let bounds = landscape.assumptions().unwrap();
    assert!((bounds.force_lipschitz - 7.6).abs() < 1e-12);
    assert!(bounds.hessian_lower_bound < 0.);
    for point in [
        vec![0., 0., 0.],
        vec![0.3, -0.2, 0.4],
        vec![3., -4., 7.],
        vec![1e3, -2e3, 5e3],
    ] {
        let gradient = landscape.analytic_gradient(&point).unwrap();
        let centered_norm_sq = point
            .iter()
            .zip(&landscape.center)
            .map(|(x, center)| (x - center).powi(2))
            .sum::<f64>();
        assert!(
            landscape.potential(&point).unwrap() + 1e-12
                >= bounds.quadratic_potential_lower_coefficient * centered_norm_sq
        );
        let dissipation = point
            .iter()
            .zip(&landscape.center)
            .zip(&gradient)
            .map(|((x, center), g)| (x - center) * g)
            .sum::<f64>();
        assert!(
            dissipation + 1e-12
                >= bounds.centered_dissipativity_coefficient * centered_norm_sq
                    - bounds.centered_dissipativity_offset
        );
        let force_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        let x_norm = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(force_norm <= bounds.force_at_zero + bounds.force_lipschitz * x_norm + 1e-12);
        if x_norm < 10. {
            for j in 0..3 {
                let mut plus = point.clone();
                let mut minus = point.clone();
                plus[j] += 1e-6;
                minus[j] -= 1e-6;
                let difference = (landscape.potential(&plus).unwrap()
                    - landscape.potential(&minus).unwrap())
                    / 2e-6;
                assert!((difference - gradient[j]).abs() < 5e-7);
            }
        }
    }
}

#[test]
fn invalid_numerics_and_shapes_fail_before_a_simulation() {
    let valid = KineticValidationConfig::default();
    for invalid in [
        KineticValidationConfig {
            dt: 0.,
            ..valid.clone()
        },
        KineticValidationConfig {
            friction: -1.,
            ..valid.clone()
        },
        KineticValidationConfig {
            velocity_diffusion: f64::NAN,
            ..valid.clone()
        },
        KineticValidationConfig {
            metric_weight: 0.,
            ..valid.clone()
        },
        KineticValidationConfig {
            samples: 1,
            ..valid.clone()
        },
        KineticValidationConfig {
            box_half_width: Some(-2.),
            ..valid.clone()
        },
        KineticValidationConfig {
            coupling_position_shift: 0.,
            coupling_velocity_shift: 0.,
            ..valid.clone()
        },
        KineticValidationConfig {
            inputs: vec![KineticInput {
                positions: vec![0.],
                velocities: vec![0.],
            }],
            ..valid.clone()
        },
        KineticValidationConfig {
            landscape: ConfiningLandscape {
                curvature: vec![1., -1.],
                ..valid.landscape.clone()
            },
            ..valid.clone()
        },
    ] {
        assert!(kinetic_constants(&invalid).is_err(), "{}", invalid.name);
    }
}

#[test]
fn native_conditional_experiments_match_chapter_two_and_retain_the_obstruction() {
    futures_lite::future::block_on(async {
        for config in default_validation_cases(2048, 619_020) {
            let report = validate_kinetic(&config).await.unwrap();
            assert_eq!(report.source_mapped_constants.len(), 24);
            assert!(
                report
                    .reference_landscape_validation
                    .checks
                    .iter()
                    .all(|c| c.status != KineticCheckStatus::Violated)
            );
            for experiment in &report.experiments {
                for check in &experiment.checks {
                    assert_ne!(
                        check.status,
                        KineticCheckStatus::Violated,
                        "{} / {:?} / {}: {:?}",
                        config.name,
                        experiment.input,
                        check.item,
                        check
                    );
                }
                assert!(experiment.maximum_b2_to_final_cap_error < 1e-12);
                assert!(experiment.maximum_velocity_norm < config.velocity_cap);
                if report.constants.phase_space_nondegeneracy_hypotheses_hold {
                    assert!(experiment.full_covariance_min_eigenvalue.unwrap() > 0.);
                }
                if config.dt == 2. {
                    assert!(experiment.velocity_variance.iter().all(|v| v.mean < 1e-24));
                    assert!(
                        experiment
                            .checks
                            .iter()
                            .any(|c| c.item == "full_phase_space_covariance_positivity"
                                && c.status == KineticCheckStatus::NotApplicable)
                    );
                }
                assert!(
                    experiment
                        .energy_drift
                        .interpretation
                        .contains("cannot certify")
                );
            }
            let json = serde_json::to_string(&report).unwrap();
            let restored: algorithmic_gas_benchmarks::convergence_kinetic::KineticValidationReport =
                serde_json::from_str(&json).unwrap();
            assert_eq!(report, restored);
            let chapter = include_str!(
                "../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"
            );
            for constant in &report.source_mapped_constants {
                let label = constant.source.split('#').next_back().unwrap();
                assert!(
                    chapter.contains(&format!(":label: {label}")),
                    "{}",
                    constant.source
                );
            }
        }
    });
}

#[test]
fn compact_reward_constants_include_the_inverse_projection_factors() {
    let config = KineticValidationConfig {
        dimensions: 1,
        landscape: ConfiningLandscape::quadratic(1),
        inputs: vec![KineticInput {
            positions: vec![0.],
            velocities: vec![0.],
        }],
        ..Default::default()
    };
    let constants = geometry_reward_constants(&config).unwrap();
    assert_eq!(constants.compact_physical_position_radius, Some(2.));
    assert_eq!(constants.inverse_position_projection_lipschitz, Some(4.));
    assert_eq!(constants.inverse_velocity_projection_lipschitz, 4.);
    assert_eq!(constants.reward_position_lipschitz_physical, Some(2.));
    assert_eq!(constants.reward_lipschitz_squashed_sasaki, Some(8.));
    assert_eq!(constants.reward_absolute_bound, Some(2.));
    let (left, right) = (1.9_f64, 2.0_f64);
    let reward_difference = 0.5 * (right * right - left * left);
    let projected_difference = 2. * right / (2. + right) - 2. * left / (2. + left);
    // Regression: the old untransformed physical constant is insufficient in
    // the squashed metric, even within the canonical alive interval.
    assert!(
        reward_difference
            > constants.reward_position_lipschitz_physical.unwrap() * projected_difference
    );
    assert!(
        reward_difference
            <= constants.reward_lipschitz_squashed_sasaki.unwrap() * projected_difference
    );
    let unbounded = geometry_reward_constants(&KineticValidationConfig {
        box_half_width: None,
        ..config
    })
    .unwrap();
    assert_eq!(unbounded.inverse_position_projection_lipschitz, None);
    assert_eq!(unbounded.reward_lipschitz_squashed_sasaki, None);
    assert_eq!(unbounded.reward_absolute_bound, None);
}

#[test]
fn quadratic_reference_variance_and_nondeception_have_the_declared_scope() {
    for d in [1, 2, 3, 8] {
        let mut quadratic = ConfiningLandscape::quadratic(d);
        quadratic.center = (0..d).map(|j| 0.1 * (j + 1) as f64).collect();
        let center_squared = quadratic.center.iter().map(|x| x * x).sum::<f64>();
        let r = 0.7_f64;
        let expected = center_squared * r * r / (d as f64 + 2.)
            + d as f64 * r.powi(4) / ((d as f64 + 4.) * (d as f64 + 2.).powi(2));
        assert!((quadratic_uniform_ball_variance(&quadratic, r).unwrap() - expected).abs() < 1e-14);
        assert!(
            (quadratic_nondeception_bound(&quadratic, 0.5).unwrap() - 0.5_f64.powi(2) / 12.).abs()
                < 1e-15
        );
    }
    let anisotropic = ConfiningLandscape {
        curvature: vec![0.5, 4.],
        center: vec![0., 0.],
        ..ConfiningLandscape::quadratic(2)
    };
    assert!(
        (quadratic_nondeception_bound(&anisotropic, 0.5).unwrap() - 0.5_f64.powi(4) / 12.).abs()
            < 1e-15
    );
    let ripples = ConfiningLandscape {
        ripple_amplitude: 0.4,
        ripple_frequency: 3.,
        ..anisotropic
    };
    assert!(quadratic_uniform_ball_variance(&ripples, 1.).is_err());
    assert!(quadratic_nondeception_bound(&ripples, 0.5).is_err());
}

#[test]
fn conditional_sampling_supports_different_dimensions_and_reports_standard_errors() {
    futures_lite::future::block_on(async {
        for d in [1, 3] {
            let config = KineticValidationConfig {
                dimensions: d,
                landscape: ConfiningLandscape::quadratic(d),
                samples: 1024,
                seed: 618_044 + d as u64,
                inputs: vec![KineticInput {
                    positions: vec![0.4; d],
                    velocities: vec![0.2; d],
                }],
                ..Default::default()
            };
            let report = validate_kinetic(&config).await.unwrap();
            let experiment = &report.experiments[0];
            assert_eq!(experiment.position_drift.len(), d);
            assert_eq!(experiment.position_covariance.len(), d * d);
            for estimate in &experiment.position_drift {
                let expected_se =
                    (report.constants.position_variance / config.samples as f64).sqrt();
                assert!((estimate.standard_error / expected_se - 1.).abs() < 0.1);
                assert_eq!(estimate.samples, config.samples);
            }
            assert!(
                experiment
                    .checks
                    .iter()
                    .all(|c| c.status != KineticCheckStatus::Violated)
            );
        }
    });
}
