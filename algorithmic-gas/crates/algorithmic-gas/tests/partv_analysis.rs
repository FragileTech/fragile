use algorithmic_gas::partv_analysis::{
    AnalysisRequest, AnalysisResponse, analyze, ordering_fraction, product_interval_volume,
    spin2_decode, spin2_encode,
};
fn request(kind: &str) -> AnalysisRequest {
    AnalysisRequest {
        kind: kind.into(),
        ..Default::default()
    }
}
fn metric(r: &AnalysisResponse, name: &str) -> f64 {
    r.metrics.iter().find(|m| m.name == name).unwrap().value
}
#[test]
fn all_reference_modes_are_seeded_and_finite() {
    for kind in [
        "spin2",
        "transport",
        "kernel",
        "manufactured",
        "integration",
        "counts",
        "dimension",
        "curvature",
    ] {
        let r = request(kind);
        let a = analyze(r.clone()).unwrap();
        let b = analyze(r).unwrap();
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap()
        );
        assert!(a.metrics.iter().all(|m| m.value.is_finite()));
    }
}
#[test]
fn spin2_roundtrip_and_lift_are_exact_to_precision() {
    for v in [
        [0., 0.],
        [1., 0.],
        [-1., 0.],
        [0., -3.],
        [1e200, -2e200],
        [1e-200, 2e-200],
    ] {
        let z = spin2_encode(v).unwrap();
        let w = spin2_decode(z);
        for i in 0..2 {
            assert!((w[i] - v[i]).abs() <= v[0].hypot(v[1]) * 1e-14 + 1e-300);
        }
    }
    assert!(spin2_encode([f64::NAN, 0.]).is_err());
    let a = analyze(request("spin2")).unwrap();
    assert!(metric(&a, "decoder equivariance error") < 1e-14);
    assert!((metric(&a, "2pi spinor overlap") + 1.).abs() < 1e-14);
    assert!((metric(&a, "4pi spinor overlap") - 1.).abs() < 1e-14);
}
#[test]
fn gauge_transformations_preserve_actual_loop_products() {
    let a = analyze(request("transport")).unwrap();
    for m in &a.metrics {
        if let Some(reference) = m.reference {
            assert!((m.value - reference).abs() < 1e-13, "{}", m.name);
        }
    }
    let b = analyze(AnalysisRequest {
        phase: 0.,
        ..request("transport")
    })
    .unwrap();
    assert_eq!(metric(&b, "SU2 normalized Wilson trace"), 1.);
    assert!(metric(&a, "SU2 normalized Wilson trace") < 0.99);
}
#[test]
fn independently_verified_timelike_moments_and_manufactured_order() {
    let a = analyze(request("manufactured")).unwrap();
    assert!(metric(&a, "zeroth moment").abs() < 1e-4);
    assert!((metric(&a, "time second moment") + 2.).abs() < 1e-4);
    assert!((metric(&a, "each spatial second moment") - 2.).abs() < 1e-4);
    assert!(metric(&a, "positive kernel mass") > 0. && metric(&a, "negative kernel mass") < 0.);
    let bias = a.series.iter().find(|s| s.name == "absolute bias").unwrap();
    let observed_order = (bias.y[0] / bias.y[2]).ln() / (bias.x[0] / bias.x[2]).ln();
    assert!((observed_order - 2.).abs() < 0.02, "order={observed_order}");
}
#[test]
fn known_density_correction_matches_analytic_integral() {
    let a = analyze(AnalysisRequest {
        samples: 1024,
        replicas: 64,
        density_contrast: 0.9,
        ..request("integration")
    })
    .unwrap();
    for m in a.metrics.iter().filter(|m| m.standard_error.is_some()) {
        assert!(
            (m.value - m.reference.unwrap()).abs() < 5. * m.standard_error.unwrap(),
            "{}",
            m.name
        );
    }
    assert!(metric(&a, "uncorrected integral") - metric(&a, "density corrected integral") > 0.4);
}
#[test]
fn fixed_total_and_poisson_thinning_are_distinct() {
    let fixed = analyze(AnalysisRequest {
        count_model: "fixed".into(),
        ..request("counts")
    })
    .unwrap();
    assert_eq!(metric(&fixed, "total variance"), 0.);
    let poisson = analyze(AnalysisRequest {
        replicas: 128,
        ..request("counts")
    })
    .unwrap();
    assert!((metric(&poisson, "total Fano factor") - 1.).abs() < 0.4);
    assert!((metric(&poisson, "region variance") - 64.).abs() < 30.);
}
#[test]
fn myrheim_meyer_normalization_and_diamond_sampling() {
    for (d, expected) in [(2., 0.5), (3., 8. / 35.), (4., 0.1)] {
        assert!((ordering_fraction(d) - expected).abs() < 1e-13);
    }
    for d in 2..=4 {
        let a = analyze(AnalysisRequest {
            samples: 256,
            replicas: 32,
            spacetime_dimension: d,
            count_model: "fixed".into(),
            ..request("dimension")
        })
        .unwrap();
        let m = &a.metrics[0];
        assert!((m.value - m.reference.unwrap()).abs() < 5. * m.standard_error.unwrap());
        for p in a.points {
            let r = p[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(r <= 0.5 - p[0].abs() + 1e-14);
        }
    }
}
#[test]
fn product_curvature_quadrature_has_correct_flat_volume_and_linear_coefficient() {
    let flat = product_interval_volume(0., 1.);
    assert!((flat - std::f64::consts::PI / 12.).abs() < 1e-13);
    let a = analyze(AnalysisRequest {
        bandwidth: 1.,
        curvature: 0.05,
        ..request("curvature")
    })
    .unwrap();
    let calibration = a
        .metrics
        .iter()
        .find(|m| m.name == "independent linear calibration")
        .unwrap();
    assert!((calibration.value / calibration.reference.unwrap() - 1.).abs() < 1e-7);
    assert!((metric(&a, "quadrature finite-radius curvature") - 0.1).abs() < 0.001);
    let m = &a.metrics[0];
    assert!((m.value - m.reference.unwrap()).abs() < 5. * m.standard_error.unwrap());
}
#[test]
fn request_defaults_and_work_limits_are_explicit() {
    let r: AnalysisRequest = serde_json::from_str("{\"kind\":\"kernel\"}").unwrap();
    assert_eq!(r.samples, 256);
    assert!(serde_json::from_str::<AnalysisRequest>("{\"unknown\":1}").is_err());
    assert!(
        analyze(AnalysisRequest {
            samples: 16384,
            replicas: 128,
            ..request("counts")
        })
        .is_err()
    );
    assert!(
        analyze(AnalysisRequest {
            samples: 1024,
            replicas: 128,
            ..request("dimension")
        })
        .is_err()
    );
    assert!(
        analyze(AnalysisRequest {
            spacetime_dimension: 4,
            ..request("curvature")
        })
        .is_err()
    );
}

#[test]
fn manufactured_sampling_uses_population_size_and_measures_mse() {
    let small = analyze(AnalysisRequest {
        samples: 256,
        replicas: 64,
        bandwidth: 1.,
        ..request("manufactured")
    })
    .unwrap();
    let large = analyze(AnalysisRequest {
        samples: 2048,
        replicas: 64,
        bandwidth: 1.,
        ..request("manufactured")
    })
    .unwrap();
    let ratio =
        metric(&small, "sampled operator variance") / metric(&large, "sampled operator variance");
    assert!(ratio > 3. && ratio < 16., "variance ratio={ratio}");
    let m = large
        .metrics
        .iter()
        .find(|m| m.name == "sampled manufactured operator")
        .unwrap();
    assert!((m.value - m.reference.unwrap()).abs() < 5. * m.standard_error.unwrap());
    assert!(metric(&large, "mean support count") > 7. * metric(&small, "mean support count"));
    let values = &large
        .series
        .iter()
        .find(|s| s.name == "largest-bandwidth replicas")
        .unwrap()
        .y;
    let mse = values.iter().map(|x| (x - 4.).powi(2)).sum::<f64>() / values.len() as f64;
    assert!((mse - metric(&large, "sampled operator MSE")).abs() < 1e-12);
}
#[test]
fn integration_ess_detects_importance_weight_dispersion() {
    let uniform = analyze(AnalysisRequest {
        density_contrast: 0.,
        ..request("integration")
    })
    .unwrap();
    assert!((metric(&uniform, "effective sample fraction") - 1.).abs() < 1e-14);
    let nonuniform = analyze(AnalysisRequest {
        density_contrast: 0.9,
        ..request("integration")
    })
    .unwrap();
    assert!(metric(&nonuniform, "effective sample fraction") < 0.85);
}

#[test]
fn independent_curvature_kernel_moments_predict_heldout_curvatures() {
    for k in [-0.7, 0., 0.3] {
        let a = analyze(AnalysisRequest {
            curvature: k,
            bandwidth: 0.2,
            samples: 1024,
            replicas: 32,
            ..request("curvature")
        })
        .unwrap();
        assert!(
            (metric(&a, "kernel flat calibration") - 187. * std::f64::consts::PI / 630.).abs()
                < 1e-13
        );
        assert!(
            (metric(&a, "kernel curvature moment MR") + 359. * std::f64::consts::PI / 41580.).abs()
                < 1e-13
        );
        assert!((metric(&a, "kernel quadrature finite-radius curvature") - 2. * k).abs() < 0.003);
        let m = a
            .metrics
            .iter()
            .find(|m| m.name == "kernel weighted volume")
            .unwrap();
        assert!((m.value - m.reference.unwrap()).abs() < 5. * m.standard_error.unwrap());
        assert!(
            (metric(&a, "estimated compact curvature action")
                - metric(&a, "kernel estimated scalar curvature")
                    * metric(&a, "compact geometric volume"))
            .abs()
                < 1e-12
        );
    }
}
#[test]
fn nonuniform_pair_sampling_is_corrected_by_pair_weights() {
    let a = analyze(AnalysisRequest {
        density_contrast: 0.9,
        samples: 256,
        replicas: 64,
        count_model: "fixed".into(),
        ..request("dimension")
    })
    .unwrap();
    let m = a
        .metrics
        .iter()
        .find(|m| m.name == "ordering fraction")
        .unwrap();
    assert!((m.value - 8. / 35.).abs() < 5. * m.standard_error.unwrap());
    assert!(metric(&a, "mean importance effective sample size") < 220.);
    assert!(
        a.metadata["normalization"]
            .as_str()
            .unwrap()
            .contains("w_i*w_j")
    );
    let uniform = analyze(AnalysisRequest {
        density_contrast: 0.,
        ..request("dimension")
    })
    .unwrap();
    assert_eq!(
        metric(&uniform, "ordering fraction"),
        metric(&uniform, "unweighted ordering fraction")
    );
}

#[test]
fn timelike_support_counts_use_double_cone_volume() {
    let a = analyze(AnalysisRequest {
        samples: 1024,
        replicas: 64,
        bandwidth: 1.,
        ..request("manufactured")
    })
    .unwrap();
    let m = a
        .metrics
        .iter()
        .find(|m| m.name == "mean support count")
        .unwrap();
    let p = std::f64::consts::PI / 12.;
    let se = (1024. * p * (1. - p) / 64.).sqrt();
    assert!((m.value - m.reference.unwrap()).abs() < 5. * se);
    let b = analyze(AnalysisRequest {
        samples: 1024,
        replicas: 64,
        ..request("curvature")
    })
    .unwrap();
    let m = b
        .metrics
        .iter()
        .find(|m| m.name == "kernel mean support count")
        .unwrap();
    assert!((m.value - m.reference.unwrap()).abs() < 5. * se);
}
