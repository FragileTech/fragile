use algorithmic_gas_benchmarks::convergence_chapter09_completion::*;
fn p() -> ConstantsInput {
    ConstantsInput {
        minimum_alive_fraction: 0.5,
        measurement_weight_lower: 0.8,
        clone_weight_lower: 0.8,
        separation_upper: 2.,
        separation_scale_floor: 0.2,
        reward_map_floor: 0.1,
        reward_map_amplitude: 2.,
        separation_map_floor: 0.1,
        separation_map_amplitude: 2.,
        reward_exponent: 1.,
        separation_exponent: 1.,
        acceptance_epsilon: 1e-6,
        acceptance_saturation: 1.,
    }
}
#[test]
fn finite_population_rates_use_probability_normalization() {
    let c = quantitative_constants(p()).unwrap();
    for n in [4, 16, 64, 256] {
        let v = c.variance_upper(n, 1.).unwrap();
        let v4 = c.variance_upper(4 * n, 1.).unwrap();
        assert!((v.log_upper - v4.log_upper - 4_f64.ln()).abs() < 1e-10);
        let b = c.bias_upper(n, 1.).unwrap();
        let b4 = c.bias_upper(4 * n, 1.).unwrap();
        assert!((b.log_upper - b4.log_upper - 2_f64.ln()).abs() < 1e-10);
        assert!(c.mean_square_upper(n, 1.).unwrap().log_upper >= v.log_upper);
    }
}
#[test]
fn zero_exponent_and_large_configuration_constants_are_kept() {
    let mut q = p();
    q.separation_exponent = 0.;
    let c = quantitative_constants(q).unwrap();
    assert_eq!(c.h_s, 0.);
    assert_eq!(c.l_t, 0.);
    assert!(c.log_b_star.is_finite());
    let mut q = p();
    q.clone_weight_lower = 0.001;
    let c = quantitative_constants(q).unwrap();
    assert!(c.variance_upper(64, 1.).unwrap().upper_float.is_none());
    assert!(c.log_a_phi_unit.is_finite());
    assert!(
        (log_component_moment(0.8, 2).unwrap().exp() - (1. + 1.6) * (3.2_f64).exp()).abs() < 1e-10
    );
}
#[test]
fn sampling_and_alive_ratios_preserve_permutation_invariance() {
    for n in [2, 4, 16, 64] {
        for l in 0..=n.min(6) {
            let exact = repeated_address_probability(n, l).unwrap();
            let union = (l * l.saturating_sub(1)) as f64 / (2. * n as f64);
            assert!(exact <= union + 1e-12);
        }
    }
    for (u, v, x, m) in [
        (0.1, 0.2, -0.3, 0.5),
        (-0.4, 0.4, 0.2, 0.8),
        (0., 0., 0.3, 0.5),
    ] {
        let bound = normalized_alive_ratio_bound(u, v, x, m).unwrap();
        let observed = if v > 0. {
            (u / v - x / m).abs()
        } else {
            1. + (x / m).abs()
        };
        assert!(observed <= bound + 1e-12);
    }
    assert!(normalized_alive_ratio_bound(0.5, 0.1, 0., 0.2).is_err());
}
#[test]
fn survival_transfer_never_replaces_finite_timestep_by_continuous_time() {
    let c = survival_constants(0.2, 0.3, 1024).unwrap();
    assert!(c.delta < 0.5);
    assert!(
        survival_constants(0.2, 0.3, c.n_surv as usize)
            .unwrap()
            .delta
            <= 0.5
    );
    for h in [0., 0.001, 0.2, 1.] {
        for n in [0, 1, 16, 128] {
            let b = full_path_extinction_upper(h, n).unwrap();
            assert!(b <= h * n as f64 + 1e-12);
            assert!((0. ..=1.).contains(&b));
        }
    }
    assert_eq!(full_path_extinction_upper(0., 128).unwrap(), 0.);
    assert_eq!(full_path_extinction_upper(1., 0).unwrap(), 0.);
}
#[test]
fn invalid_hypotheses_are_rejected() {
    let mut q = p();
    q.minimum_alive_fraction = 0.;
    assert!(quantitative_constants(q).is_err());
    let mut q = p();
    q.acceptance_saturation = 0.;
    assert!(quantitative_constants(q).is_err());
    let mut q = p();
    q.separation_scale_floor = f64::NAN;
    assert!(quantitative_constants(q).is_err());
    assert!(repeated_address_probability(0, 2).is_err());
    assert!(log_component_moment(-1., 2).is_err());
    assert!(from_log(f64::INFINITY).is_err());
}

#[test]
fn arbitrary_component_moment_uses_complete_factorial_series() {
    let c = 0.8_f64;
    let fourth = (1. + 14. * c + 18. * c * c + 4. * c * c * c).ln() + 16. * c;
    assert!((log_component_moment(c, 4).unwrap() - fourth).abs() < 1e-12);
    for order in [1, 2, 3, 4, 8, 16, 128] {
        assert_eq!(log_component_moment(0., order).unwrap(), 0.);
        assert!(log_component_moment(c, order).unwrap().is_finite());
    }
    assert!(log_component_moment(c, 0).is_err());
    assert!(log_component_moment(c, 129).is_err());
}

#[test]
fn native_bridge_uses_actual_kernel_and_rejects_other_sampling_programs() {
    let cfg = algorithmic_gas::GasConfig::euclidean(1, 0.04).unwrap();
    let constants = constants_from_native(&cfg, 1.).unwrap();
    assert!((constants.c - 2. * 4_f64.exp()).abs() < 1e-10);
    let higher = algorithmic_gas::GasConfig::euclidean(4, 0.04).unwrap();
    assert_eq!(constants_from_native(&higher, 1.).unwrap().c, constants.c);
    let mut unsupported = cfg.clone();
    unsupported.cloning_donors.history_window = 1;
    assert!(constants_from_native(&unsupported, 1.).is_err());
    let mut unsupported = cfg.clone();
    unsupported.distance_donors.allow_self = true;
    assert!(constants_from_native(&unsupported, 1.).is_err());
    let mut unsupported = cfg;
    unsupported.clone_decision.revival_from_companion = false;
    assert!(constants_from_native(&unsupported, 1.).is_err());
}
