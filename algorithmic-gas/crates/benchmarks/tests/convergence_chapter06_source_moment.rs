use algorithmic_gas_benchmarks::{
    convergence_chapter06_completion::*, convergence_chapter06_source_moment::*,
};
fn source(exponent: f64, history: usize) -> SourceMomentParameters {
    SourceMomentParameters {
        position_feature_radius: 2.,
        velocity_feature_radius: 2.,
        algorithmic_velocity_weight: 1.,
        cloning_bandwidth: 3.,
        gate_saturation: 1.,
        gate_epsilon: 1e-6,
        historical_window: history,
        component_collision_enabled: history == 0,
        fitness_channels: vec![
            FitnessChannelRange {
                amplitude: 2.,
                positive_floor: 0.1,
                exponent
            };
            2
        ],
    }
}
fn native(d: usize, p: f64) -> NativeTailParameters {
    NativeTailParameters {
        dimension: d,
        moment_order: p,
        timestep: 0.04,
        friction: 1.,
        harmonic_curvature: 2.,
        residual_per_coordinate: 20. * std::f64::consts::PI,
        clone_jitter_standard_deviation: 0.1,
        ou_standard_deviation: 0.196,
        final_position_standard_deviation: 0.02,
        frozen_velocity_norm_bound: 2.,
        restitution: 0.5,
    }
}
#[test]
fn nonempty_actual_weak_exponent_interval_closes_p4_and_p8_with_history() {
    for d in [1, 2, 4, 8] {
        for p in [4., 8.] {
            for h in [0, 1, 4] {
                let mut kinetic = native(d, p);
                if h > 0 {
                    kinetic.restitution = 0.;
                }
                let k = native_selected_source_tail_interface(&kinetic, 1.).unwrap();
                let c = selected_native_moment_closure(&k, &source(1e-5, h)).unwrap();
                assert!(c.moment.closes);
                assert!(c.moment.invariant_moment_upper.unwrap().is_finite());
                assert_eq!(c.history_block_length, h + 1);
            }
        }
    }
}
#[test]
fn native_historical_collision_mode_is_rejected_even_at_zero_restitution() {
    let mut p = source(1e-5, 1);
    p.component_collision_enabled = true;
    assert!(source_moment_envelope(&p).is_err());
    p.component_collision_enabled = false;
    let k = native_selected_source_tail_interface(&native(1, 4.), 1.).unwrap();
    assert!(selected_native_moment_closure(&k, &p).is_err());
    let mut k = k;
    k.parameters.restitution = 0.;
    assert!(
        selected_native_moment_closure(&k, &p)
            .unwrap()
            .moment
            .closes
    );
}
#[test]
fn strong_default_selection_is_not_promoted_to_a_closed_drift() {
    let k = native_selected_source_tail_interface(&native(2, 4.), 1.).unwrap();
    let c = selected_native_moment_closure(&k, &source(1., 0)).unwrap();
    assert!(!c.moment.closes);
    assert!(c.moment.invariant_moment_upper.is_none());
}
#[test]
fn actual_squared_feature_diameter_and_gate_epsilon_are_accounted_for() {
    let c = source_moment_envelope(&source(1e-5, 0)).unwrap();
    assert!((c.algorithmic_feature_diameter_squared - 32.).abs() < 1e-14);
    assert!((c.kernel_lower_float - (-32_f64 / 18.).exp()).abs() < 1e-14);
    let mut eps = source(1e-5, 0);
    eps.gate_epsilon = 10.;
    assert!(source_moment_envelope(&eps).unwrap().acceptance_upper < c.acceptance_upper);
    assert_eq!(
        source_moment_envelope(&source(0., 0))
            .unwrap()
            .current_frame_source_moment_coefficient,
        1.
    );
}
#[test]
fn history_incoming_load_bound_is_uniform_in_population_and_warmup() {
    for n in [2usize, 4, 16, 64, 200] {
        for h in 1..5 {
            let ratio = (n * (h + 1)) as f64 / (n * (h + 1) - 1) as f64;
            assert!(ratio <= 4. / 3. + 1e-14);
        }
    }
}

#[test]
fn accepted_jitter_and_component_energy_refine_floor_without_faking_native_parameters() {
    for d in [1, 2, 4, 8] {
        for p in [1., 2., 4., 8.] {
            let parameters = native(d, p);
            let k = native_selected_source_tail_interface(&parameters, 1.).unwrap();
            let refined =
                selected_native_refined_moment_closure(&k, &source(1e-5, 0), true).unwrap();
            assert!(refined.refined_additive_lp_budget < k.additive_lp_budget);
            assert!(refined.refined_moment.additive < refined.original.moment.additive);
            assert_eq!(
                refined.refined_moment.coefficient,
                refined.original.moment.coefficient
            );
            assert_eq!(
                k.parameters.frozen_velocity_norm_bound,
                parameters.frozen_velocity_norm_bound
            );
            assert_eq!(
                k.parameters.clone_jitter_standard_deviation,
                parameters.clone_jitter_standard_deviation
            );
            if p <= 2. {
                assert_eq!(
                    refined.collision_normalized_lp_upper,
                    parameters.frozen_velocity_norm_bound
                );
            }
            if refined.refined_moment.closes {
                assert!(
                    refined.refined_moment.invariant_moment_upper.unwrap()
                        < refined.original.moment.invariant_moment_upper.unwrap()
                );
            }
            assert!(selected_native_refined_moment_closure(&k, &source(1e-5, 0), false).is_err());
        }
    }
}

#[test]
fn root_minkowski_closes_beyond_young_and_improves_floor() {
    for d in [1, 2, 4, 8] {
        for p in [4., 8.] {
            let k = native_selected_source_tail_interface(&native(d, p), 1.).unwrap();
            let root = selected_native_root_moment_closure(&k, &source(1e-5, 0), true).unwrap();
            assert!(root.closes);
            assert!(
                root.invariant_moment_upper.unwrap()
                    < root.refined.refined_moment.invariant_moment_upper.unwrap()
            );
            let extended =
                selected_native_root_moment_closure(&k, &source(0.00015, 0), true).unwrap();
            assert!(extended.closes);
            if p == 4. {
                assert!(!extended.refined.original.moment.closes);
            }
            let strong = selected_native_root_moment_closure(&k, &source(1., 0), true).unwrap();
            assert!(!strong.closes);
            assert!(strong.invariant_moment_upper.is_none());
            let initial = 10.;
            let mut value = initial;
            for n in 1..20 {
                value = root.root_moment_coefficient * value + root.additive_root_budget;
                let bound = root.root_moment_coefficient.powi(n) * initial
                    + root.invariant_root_moment_upper.unwrap()
                        * (1. - root.root_moment_coefficient.powi(n));
                assert!((value - bound).abs() < 1e-10 * bound.max(1.));
            }
        }
    }
}

#[test]
fn root_selection_interval_retains_actual_gate_saturation() {
    let k = native_selected_source_tail_interface(&native(2, 4.), 1.).unwrap();
    let mut p = source(0.00015, 0);
    p.gate_saturation = 0.1;
    assert!(
        !selected_native_root_moment_closure(&k, &p, true)
            .unwrap()
            .closes
    );
    p.gate_saturation = 10.;
    assert!(
        selected_native_root_moment_closure(&k, &p, true)
            .unwrap()
            .closes
    );
    let saturated = source_moment_envelope(&p).unwrap();
    p.gate_saturation = 0.1;
    let unsaturated = source_moment_envelope(&p).unwrap();
    assert!((unsaturated.acceptance_upper / saturated.acceptance_upper - 100.).abs() < 1e-10);
}
