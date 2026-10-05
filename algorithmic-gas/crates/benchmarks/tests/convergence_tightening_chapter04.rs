use algorithmic_gas::kinetic::ViscousForceConfig;
use algorithmic_gas_benchmarks::convergence_tightening_chapter04::{
    PotentialProfile, count_hilbert_derivative, dimension_reset, gaussian_log_budget,
    occupation_certificate, propagated_moment_budget, regional_landscape_profile,
    row_continuity_budget, row_uniform_derivative, viscous_certificate,
};

fn rastrigin(d: usize) -> PotentialProfile {
    PotentialProfile {
        dimension: d,
        quadratic_curvature: 2.,
        cosine_amplitude: 10.,
        frequency: std::f64::consts::TAU,
    }
}

#[test]
fn dead_storage_and_particle_permutation_leave_reset_unchanged() {
    let mut x = vec![vec![1., 0.], vec![-1., 0.], vec![1e6, -1e6]];
    let a = vec![vec![0.; 3], vec![0.; 3], vec![0.5, 0.5, 0.]];
    let c = occupation_certificate(&x, &[true, true, false], &a, 0.1).unwrap();
    x[2] = vec![-1e12, 1e12];
    let other = occupation_certificate(&x, &[true, true, false], &a, 0.1).unwrap();
    assert_eq!(c.expected_variance, other.expected_variance);
    assert_eq!(c.coordinate_span_dimension, 1);
    assert!((c.jitter_contribution - (1. - 1. / 3.) * 2. * 0.01 / 3.).abs() < 1e-14);
    assert!(c.expected_variance <= c.dimension_upper);
    let permutation = [2, 0, 1];
    let perm_x: Vec<_> = permutation.iter().map(|&i| x[i].clone()).collect();
    let perm_a: Vec<_> = permutation
        .iter()
        .map(|&i| permutation.iter().map(|&j| a[i][j]).collect())
        .collect();
    let perm = occupation_certificate(&perm_x, &[false, true, true], &perm_a, 0.1).unwrap();
    assert!((perm.expected_variance - c.expected_variance).abs() < 1e-14);
    assert!(occupation_certificate(&x, &[false; 3], &a, 0.1).is_err());
    let mut bad = a;
    bad[0][2] = 0.1;
    assert!(occupation_certificate(&x, &[true, true, false], &bad, 0.1).is_err());
}

#[test]
fn jung_geometry_is_sharp_and_noise_retains_ambient_dimension() {
    let x = vec![
        vec![1., 0.],
        vec![-0.5, 3_f64.sqrt() / 2.],
        vec![-0.5, -3_f64.sqrt() / 2.],
    ];
    let c = occupation_certificate(&x, &[true; 3], &vec![vec![0.; 3]; 3], 0.).unwrap();
    assert!((c.expected_variance - c.dimension_upper).abs() < 1e-14);
    assert_eq!(c.coordinate_span_dimension, 2);
    assert_eq!(
        dimension_reset(10, 1, 4., 0.1, 1., 100).unwrap(),
        1. + 0.99 * 10. * 0.01
    );
    assert!(dimension_reset(2, 0, 1., 0., 1., 10).is_err());
    assert!(dimension_reset(2, 3, 1., 0., 1., 10).is_err());
}

#[test]
fn global_and_regional_rastrigin_profiles_certify_different_restoration() {
    for d in [1, 2, 3, 8] {
        let p = rastrigin(d);
        let core = regional_landscape_profile(&p, &vec![-0.125; d], &vec![0.125; d], 0.25, 1., 2.)
            .unwrap();
        let barrier =
            regional_landscape_profile(&p, &vec![0.375; d], &vec![0.625; d], 0.25, 1., 2.).unwrap();
        assert!(core.certified_restoring_region);
        assert!(!barrier.certified_restoring_region);
        assert_eq!(core.pairwise_defect_upper, 0.);
        assert!(barrier.pairwise_defect_upper > 0.);
        assert!(
            (core.bounded_perturbation_amplitude - 20. * std::f64::consts::PI * (d as f64).sqrt())
                .abs()
                < 1e-12
        );
        assert_eq!(core.reward_constant_growth, 20. * d as f64);
        assert!(
            regional_landscape_profile(&p, &vec![-2.; d], &vec![2.; d], 1., 2., 2.)
                .unwrap()
                .global_radial_defect_upper
                .is_none()
        );
    }
    let p = rastrigin(1);
    let prof = regional_landscape_profile(&p, &[-2.], &[2.], 0.8, 1., 2.).unwrap();
    for i in 0..101 {
        for j in 0..101 {
            let x = -2. + 4. * i as f64 / 100.;
            let y = -2. + 4. * j as f64 / 100.;
            let dx = x - y;
            let f =
                |z: f64| -2. * z - 20. * std::f64::consts::PI * (std::f64::consts::TAU * z).sin();
            assert!((x * f(x) + x * x).max(0.) <= prof.radial_defect_upper + 1e-12);
            if dx.abs() <= 0.8 {
                assert!(
                    (dx * (f(x) - f(y)) + dx * dx).max(0.) <= prof.pairwise_defect_upper + 1e-12
                );
            }
        }
    }
}

#[test]
fn exact_box_derivative_and_global_moment_reward_bounds_are_valid() {
    let p = rastrigin(2);
    let exact = p.reward_lipschitz_on_box(2.).unwrap();
    let crude = 2_f64.sqrt() * (4. + 20. * std::f64::consts::PI);
    assert!(exact < crude);
    for i in 0..1001 {
        let x = -2. + 4. * i as f64 / 1000.;
        let g = 2. * x + 20. * std::f64::consts::PI * (std::f64::consts::TAU * x).sin();
        assert!(2_f64.sqrt() * g.abs() <= exact + 1e-12);
    }
    let x: [f64; 2] = [1e5, -3.];
    let y: [f64; 2] = [1e5 + 0.1, -2.8];
    let m2 = |z: &[f64]| z.iter().map(|z| z * z).sum::<f64>();
    let cost = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
    assert!(
        (p.value(&x).unwrap() - p.value(&y).unwrap()).abs()
            <= p.reward_mean_bound(m2(&x), m2(&y), cost).unwrap()
    );
    let coordinate_cost: Vec<_> = x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).collect();
    let separated = p
        .reward_separated_moment_bound(m2(&x), m2(&y), &coordinate_cost)
        .unwrap();
    assert!((p.value(&x).unwrap() - p.value(&y).unwrap()).abs() <= separated);
    assert!(separated < p.reward_mean_bound(m2(&x), m2(&y), cost).unwrap());
    let p = rastrigin(8);
    let cost = [0.75, 0., 0., 0., 0., 0., 0., 0.];
    assert!(p.reward_separated_moment_bound(0.75, 0., &cost).unwrap() < 22.);
}

#[test]
fn normalized_moment_and_full_gaussian_budgets_do_not_need_source_maxima() {
    let p = rastrigin(3);
    let c =
        propagated_moment_budget(&p, 1e6, 0.4, 0.1, 4., 2., 0.04, 1., 0.0384, 0.02, 0.3).unwrap();
    assert!(c.final_position_second_moment >= c.ou_position_second_moment);
    assert!(p.cosine_amplitude * p.frequency * 3_f64.sqrt() > 100.);
    assert!(propagated_moment_budget(&p, 1., 1., 0.1, 4., 2., 0.04, 1., 0.04, 0.02, 100.).is_err());
    for d in [1, 3, 10] {
        let sharp = gaussian_log_budget(d, 0.5, 4., 0.03).unwrap();
        let crude = 0.5 * d as f64 * 2_f64.ln() + 2. * 0.5 * 16.;
        assert!(sharp < crude);
    }
    assert!(gaussian_log_budget(3, 1., 4., 0.5).is_err());
    let row = row_continuity_budget(3, 12_f64.sqrt(), 0.1, 4., 0.04, 1., 0.03844, 1.).unwrap();
    assert!(row.sharp_log_m < row.old_log_m);
    assert!(row.sharp_h < row.old_h);
    assert!(row.sharp_log_k < row.old_log_k);
    assert!(row_continuity_budget(3, -2., 0.1, 4., 0.04, 1., 0.04, 1.).is_err());
}

#[test]
fn derivative_moments_match_provider_differences_and_hilbert_bound() {
    let x: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![(i as f64 * 1.7).sin(), (i as f64 * 0.8).cos()])
        .collect();
    let v: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![(i as f64).sin(), (i as f64).cos()])
        .collect();
    let h: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![(i as f64 * 0.7).sin(), (i as f64 * 0.2).cos()])
        .collect();
    for row in [false, true] {
        let config = ViscousForceConfig {
            coefficient: 0.3,
            bandwidth: 1.,
            row_normalized: row,
        };
        let c = viscous_certificate(&x, &v, &h, &config).unwrap();
        let shifted = |sign: f64| {
            x.iter()
                .zip(&h)
                .map(|(x, h)| x.iter().zip(h).map(|(x, h)| x + sign * 1e-6 * h).collect())
                .collect::<Vec<_>>()
        };
        let a = viscous_certificate(&shifted(1.), &v, &h, &config).unwrap();
        let b = viscous_certificate(&shifted(-1.), &v, &h, &config).unwrap();
        for ((a, b), j) in a
            .force
            .concat()
            .iter()
            .zip(b.force.concat())
            .zip(c.jacobian_action.concat())
        {
            assert!(((a - b) / 2e-6 - j).abs() < 1e-9);
        }
        assert!(
            c.action_max_row_norm <= c.max_row_moment_derivative * c.direction_max_row_norm + 1e-12
        );
        if !row {
            let bound = count_hilbert_derivative(16, 0.3, 1., 1.).unwrap();
            assert!(c.action_hilbert_norm <= bound * c.direction_hilbert_norm);
        } else {
            let bound = row_uniform_derivative(16, 0.3, 1., 2_f64.sqrt(), 1.).unwrap();
            assert!(c.action_max_row_norm <= bound * c.direction_max_row_norm);
        }
    }
    let cfg = ViscousForceConfig {
        coefficient: 0.3,
        bandwidth: 1.,
        row_normalized: true,
    };
    let pair = viscous_certificate(&x[..2], &v[..2], &h[..2], &cfg).unwrap();
    assert!(pair.action_max_row_norm < 1e-16);
    assert_eq!(row_uniform_derivative(2, 0.3, 1., 1., 1.).unwrap(), 0.);
}
