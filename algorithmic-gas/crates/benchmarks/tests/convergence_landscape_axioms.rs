use algorithmic_gas::{ExecutionContext, Precision, TensorBatch, compute::BackendKind};
use algorithmic_gas_benchmarks::convergence_landscape_axioms::*;

#[test]
fn quadratic_limit_has_the_exact_all_segment_floor() {
    let c = segment_gradient_budget(3., 0., 2.).unwrap();
    assert!((c.gradient_energy_lower - 3.).abs() < 1e-13);
    assert!(c.positive_nondeception && c.population_independent);
}

#[test]
fn shorter_segments_do_not_receive_positive_nondeception() {
    let c = segment_gradient_budget(2., 10., 1.).unwrap();
    assert_eq!(c.gradient_energy_lower, 0.);
    assert!(!c.positive_nondeception);
}

#[test]
fn dimension_is_explicit_and_there_is_no_population_factor() {
    for d in [1, 2, 4, 8] {
        let c = rastrigin_gradient_budget(d).unwrap();
        let expected = 400. * std::f64::consts::PI.powi(2) * d as f64;
        assert!((c.gradient_energy_lower / expected - 1.).abs() < 1e-14);
    }
}

#[test]
fn analytic_integral_matches_native_gradient_at_constant_and_tiny_segments() {
    let mut cx =
        futures_lite::future::block_on(ExecutionContext::new(BackendKind::Cpu, Precision::F64))
            .unwrap();
    for gap in [0., 1e-12, 1e-6] {
        let x = [0.17, -0.32];
        let y = [x[0] + gap, x[1] - gap];
        let (exact, bound) = rastrigin_segment_energy(&x, &y, 128).unwrap();
        let numerical = (0..=128)
            .map(|i| {
                let t = i as f64 / 128.;
                let z: Vec<_> = x.iter().zip(y).map(|(a, b)| a + t * (b - a)).collect();
                let input = TensorBatch::vectors(1, 2, z).unwrap();
                let g = futures_lite::future::block_on(native_rastrigin_gradient(&mut cx, &input))
                    .unwrap();
                let w = if i == 0 || i == 128 {
                    1.
                } else if i % 2 == 0 {
                    2.
                } else {
                    4.
                };
                w * g.values().iter().map(|v| v * v).sum::<f64>()
            })
            .sum::<f64>()
            / 384.;
        assert!((exact - numerical).abs() <= bound + 1e-10 * exact.abs().max(1.));
    }
}

#[test]
fn invalid_hypotheses_and_quadratures_are_rejected() {
    for p in [
        (0., 1., 2.),
        (1., -1., 2.),
        (1., 1., 0.),
        (f64::NAN, 1., 2.),
    ] {
        assert!(segment_gradient_budget(p.0, p.1, p.2).is_err());
    }
    assert!(rastrigin_gradient_budget(0).is_err());
    assert!(rastrigin_segment_energy(&[0.], &[1.], 3).is_err());
}

#[test]
fn short_coordinate_certificate_covers_full_period_and_critical_other_coordinates() {
    use algorithmic_gas_benchmarks::convergence_landscape_axioms::{
        rastrigin_coordinate_energy_lower, rastrigin_segment_energy,
        rastrigin_short_gradient_budget,
    };
    for d in [1usize, 2, 4, 8] {
        let b = rastrigin_short_gradient_budget(d).unwrap();
        assert!(b.gradient_energy_lower > 1207.);
        assert_eq!(b.minimum_segment_length, (d as f64).sqrt());
        let mut x = vec![0.; d];
        let mut y = x.clone();
        x[0] = -b.minimum_segment_length / 2.;
        y[0] = b.minimum_segment_length / 2.;
        let energy = rastrigin_segment_energy(&x, &y, 8192).unwrap().0;
        assert!(energy >= b.gradient_energy_lower);
        assert!(rastrigin_coordinate_energy_lower(d, b.minimum_segment_length * 0.99).is_err());
    }
    assert!(rastrigin_short_gradient_budget(0).is_err());
}
