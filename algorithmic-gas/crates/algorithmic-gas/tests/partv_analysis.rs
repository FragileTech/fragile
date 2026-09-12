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

#[test]
fn kernel_calibration_matches_exact_polynomial_integrals() {
    let a = analyze(request("kernel")).unwrap();
    // Integrate monomials independently: r=t*u, dV=4*pi*t²*u dt du.
    // Coefficients here are exact rational values from that polynomial system.
    for (i, numerator) in [10395. / 32., -155925. / 32., 31185. / 2.]
        .iter()
        .enumerate()
    {
        let actual = a.metadata["coefficients"][i].as_f64().unwrap();
        assert!((actual * std::f64::consts::PI / numerator - 1.).abs() < 1e-12);
    }
}

#[test]
fn zero_gradient_manufactured_variance_has_sharper_bandwidth_rate() {
    let a = analyze(AnalysisRequest {
        bandwidth: 0.01,
        samples: 8,
        replicas: 2,
        ..request("manufactured")
    })
    .unwrap();
    let variance = a
        .series
        .iter()
        .find(|s| s.name == "Predicted single-run variance")
        .unwrap();
    let standard_error = a
        .series
        .iter()
        .find(|s| s.name == "Predicted standard error of mean")
        .unwrap();
    let leading_integral = 86574609. / (33592. * std::f64::consts::PI);
    for i in 0..variance.x.len() {
        let h = variance.x[i];
        // Exact quadratic-field angular integral of k²(t²+x²+2y²)².
        let predicted_leading = 8. * leading_integral / (8. * h.powi(3));
        assert!((variance.y[i] / predicted_leading - 1.).abs() < 1e-4);
        assert!((standard_error.y[i].powi(2) / variance.y[i] - 0.5).abs() < 1e-12);
    }
    // Empty observed supports do not mean the underlying estimator has zero variance.
    assert_eq!(metric(&a, "sampled operator variance"), 0.);
    assert!(variance.y[0] > 1e6);
    let measured = a
        .series
        .iter()
        .find(|s| s.name == "Monte Carlo single-run variance")
        .unwrap();
    assert!(measured.y.iter().all(|v| *v == 0.));
}

#[test]
fn predicted_operator_variance_tracks_independent_replicas_and_inverse_sample_count() {
    let mut predictions = vec![];
    for n in [1024, 4096] {
        let a = analyze(AnalysisRequest {
            bandwidth: 1.,
            samples: n,
            replicas: 128,
            ..request("manufactured")
        })
        .unwrap();
        let v = a
            .metrics
            .iter()
            .find(|m| m.name == "sampled operator variance")
            .unwrap();
        predictions.push(v.reference.unwrap());
        assert!((v.value / v.reference.unwrap() - 1.).abs() < 0.4);
        let mean = a
            .metrics
            .iter()
            .find(|m| m.name == "sampled manufactured operator")
            .unwrap();
        let predicted_se = (v.reference.unwrap() / 128.).sqrt();
        assert!((mean.value - mean.reference.unwrap()).abs() < 5. * predicted_se);
        eprintln!(
            "V16 N={n}: variance={} prediction={}, mean={} expected={}, predicted SEM={predicted_se}",
            v.value,
            v.reference.unwrap(),
            mean.value,
            mean.reference.unwrap()
        );
    }
    assert!((predictions[0] / predictions[1] - 4.).abs() < 1e-12);
}

#[test]
fn shrinking_cube_curvature_variance_matches_exact_flat_moment() {
    let pi = std::f64::consts::PI;
    let flat = 187. * pi / 630.;
    let mr = -359. * pi / 41580.;
    // Independent exact integration of (1-t²+r²)^6 over the cone.
    let second = 8. * 8774. * pi / 45045.;
    let mut variances = vec![];
    for h in [1., 0.5] {
        let a = analyze(AnalysisRequest {
            bandwidth: h,
            curvature: 0.,
            samples: 4096,
            replicas: 128,
            ..request("curvature")
        })
        .unwrap();
        let v = a
            .metrics
            .iter()
            .find(|m| m.name == "kernel single-run curvature variance")
            .unwrap();
        let exact = (second - flat * flat) / (4096. * h.powi(4) * mr * mr);
        assert!((v.reference.unwrap() / exact - 1.).abs() < 1e-12);
        assert!((v.value / exact - 1.).abs() < 0.4);
        let mean = metric(&a, "kernel estimated scalar curvature");
        let predicted_se = metric(&a, "kernel predicted standard error of mean");
        assert!(mean.abs() < 5. * predicted_se);
        eprintln!(
            "V20 eps={h}: variance={} exact={exact}, R={mean}, predicted SEM={predicted_se}",
            v.value
        );
        variances.push(exact);
    }
    assert!((variances[1] / variances[0] - 16.).abs() < 1e-12);
}

#[test]
fn product_interval_volume_matches_closed_forms_for_both_curvature_signs() {
    let tau = 1.4;
    let a = tau / 2.;
    let k: f64 = 0.8;
    let root = k.sqrt();
    let positive = 4. * std::f64::consts::PI * (a / k - (root * a).sin() / root.powi(3));
    let negative = 4. * std::f64::consts::PI * ((root * a).sinh() / root.powi(3) - a / k);
    assert!((product_interval_volume(k, tau) - positive).abs() < 1e-12);
    assert!((product_interval_volume(-k, tau) - negative).abs() < 1e-12);
}

#[test]
fn count_sample_variance_uncertainty_uses_fourth_moments() {
    for model in ["fixed", "poisson"] {
        let a = analyze(AnalysisRequest {
            count_model: model.into(),
            replicas: 128,
            ..request("counts")
        })
        .unwrap();
        for m in a.metrics.iter().filter(|m| m.name.ends_with("variance")) {
            let variance = m.reference.unwrap();
            let cumulant = variance * if model == "fixed" { -0.125 } else { 1. };
            let expected_error = ((cumulant + 256. / 127. * variance * variance) / 128.).sqrt();
            assert!((m.standard_error.unwrap() - expected_error).abs() < 1e-12);
            assert!((m.value - variance).abs() <= 5. * expected_error);
        }
    }
}

#[test]
fn unresolved_mm_inverses_preserve_original_replica_indices() {
    let a = analyze(AnalysisRequest {
        samples: 8,
        replicas: 128,
        spacetime_dimension: 4,
        count_model: "fixed".into(),
        ..request("dimension")
    })
    .unwrap();
    let fractions = a
        .series
        .iter()
        .find(|s| s.name == "ordering fraction replicas")
        .unwrap();
    let dimensions = a
        .series
        .iter()
        .find(|s| s.name == "MM dimensions with resolved inverse")
        .unwrap();
    assert!(dimensions.y.len() < 128);
    assert!(
        dimensions
            .x
            .windows(2)
            .any(|indices| indices[1] > indices[0] + 1.)
    );
    for (&replica, &dimension) in dimensions.x.iter().zip(&dimensions.y) {
        let i = fractions
            .x
            .iter()
            .position(|index| *index == replica)
            .unwrap();
        assert!((ordering_fraction(dimension) - fractions.y[i]).abs() < 1e-12);
    }
    assert_eq!(
        a.metadata["replica_coverage"]["with_resolved_inverse"]
            .as_u64()
            .unwrap(),
        dimensions.y.len() as u64
    );
}
