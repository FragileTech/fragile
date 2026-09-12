use crate::physics::{
    partvi::{self, ExperimentRequest, ExperimentResult},
    qft::{edge_spinor, math::*},
};
use serde_json::json;
fn run(id: u32, p: serde_json::Value) -> ExperimentResult {
    super::analyze_dispatch(
        &ExperimentRequest {
            experiment: id,
            parameters: p,
        },
        None,
    )
    .unwrap()
}
fn metric(r: &ExperimentResult, label: &str) -> f64 {
    r.metrics
        .iter()
        .find(|x| x.label == label)
        .unwrap_or_else(|| panic!("missing metric {label}"))
        .value
        .unwrap()
}
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() < t,
        "{a} differs from {b} by {}",
        (a - b).abs()
    );
}
#[test]
fn every_qft_workbench_has_finite_serializable_outputs() {
    for id in 1..=36 {
        let r = run(id, json!({}));
        assert_eq!(r.experiment, id);
        assert!(!r.plots.is_empty(), "experiment {id} lacks a plot");
        assert!(
            r.plots
                .iter()
                .flat_map(|p| &p.series)
                .flat_map(|s| &s.points)
                .flatten()
                .all(|x| x.is_finite()),
            "experiment {id}"
        );
        assert!(!r.model.is_empty());
        serde_json::to_vec(&r).unwrap();
    }
}
#[test]
fn qft_controls_remain_finite_at_endpoints() {
    for amplitude in [0., 2.] {
        for id in 1..=36 {
            let r = run(
                id,
                json!({"amplitude":amplitude,"angle":std::f64::consts::PI,"rate":2.,"n":32,"steps":2,"seed":19}),
            );
            assert!(
                r.plots
                    .iter()
                    .flat_map(|p| &p.series)
                    .flat_map(|s| &s.points)
                    .flatten()
                    .all(|x| x.is_finite()),
                "experiment {id} amplitude {amplitude}"
            );
        }
    }
}
#[test]
fn weighted_clone_identity_retains_nonzero_raw_score_sum() {
    let r = run(2, json!({"amplitude":0.7}));
    close(metric(&r, "Weighted antisymmetry residual"), 0., 1e-12);
    close(metric(&r, "Opposing acceptance product"), 0., 1e-12);
    assert!(r.plots[0].series[0].points.iter().any(|p| p[1] > 0.1));
}
#[test]
fn car_and_choi_identities_are_independent() {
    let r = run(3, json!({"modes":4}));
    close(metric(&r, "CAR residual"), 0., 1e-12);
    close(metric(&r, "Gram rank"), 2., 1e-12);
    let r = run(4, json!({"rate":0.7}));
    close(metric(&r, "Unital residual"), 0., 1e-12);
    assert!(metric(&r, "Minimum Choi eigenvalue") >= -1e-12);
    close(
        metric(&r, "Vacuum multiplicative defect"),
        metric(&r, "Predicted defect"),
        1e-12,
    );
}
#[test]
fn complex_hermitian_solver_preserves_imaginary_offdiagonals() {
    let a = vec![C::from(2.), C::new(0., 1.), C::new(0., -1.), C::from(2.)];
    let w = hermitian_eigenvalues(&a, 2);
    close(w[0], 1., 1e-10);
    close(w[1], 3., 1e-10);
}
#[test]
fn direct_color_threshold_has_raw_half_norm_and_zero_extension() {
    let r = run(8, json!({}));
    let raw = &r.plots[0].series[0].points;
    let extended = &r.plots[0].series[1].points;
    close(raw.last().unwrap()[1], 0.25, 1e-12);
    close(extended.last().unwrap()[1], 0., 1e-12);
    assert_eq!(
        r.details["valid_mask"].as_array().unwrap().last().unwrap(),
        &json!(false)
    );
}
#[test]
fn su3_orbits_and_scalar_projector_triangles_match() {
    let r = run(9, json!({"angle":1.3,"amplitude":1.1}));
    for key in [
        "Gram invariance residual",
        "Baryon invariance residual",
        "Gram determinant residual",
        "Triangle trace residual",
    ] {
        close(metric(&r, key), 0., 1e-12);
    }
}
#[test]
fn hidden_mode_memory_is_nonzero_and_lumpable_limit_vanishes() {
    let r = run(12, json!({"amplitude":1.}));
    close(metric(&r, "Two-step memory defect"), 0.01, 1e-12);
    close(metric(&r, "Memory recurrence residual"), 0., 1e-12);
    let r = run(12, json!({"amplitude":0.}));
    close(metric(&r, "Two-step memory defect"), 0., 1e-12);
}
#[test]
fn gaussian_response_agrees_with_shifted_quadrature() {
    let r = run(19, json!({"amplitude":1.,"n":4096,"seed":31}));
    let pred = normal_cdf(0.5);
    assert!((metric(&r, "Shifted mean") - pred).abs() < 6. * metric(&r, "Shifted SEM"));
    assert!((metric(&r, "Reweighted mean") - pred).abs() < 6. * metric(&r, "Reweighted SEM"));
}
#[test]
fn joint_collision_covariance_and_product_terms_survive() {
    let r = run(26, json!({"amplitude":1.}));
    close(
        metric(&r, "Cross-walker increment covariance"),
        -0.64,
        1e-12,
    );
    close(metric(&r, "Product increment residual"), 0., 1e-12);
    close(metric(&r, "Total momentum increment"), 0., 1e-12);
}
#[test]
fn reflection_counterexample_has_the_predicted_negative_sign() {
    let r = run(28, json!({"amplitude":0.9,"rate":0.4}));
    close(metric(&r, "Reflected quadratic value"), -0.49, 1e-12);
    assert!(metric(&r, "Minimum Hermitian-part eigenvalue") < 0.);
    close(metric(&r, "Quadratic identity residual"), 0., 1e-12);
}
#[test]
fn scalar_outer_faces_telescope_while_triangles_do_not() {
    let r = run(29, json!({"angle":1.2}));
    close(metric(&r, "Outer plaquette defect"), 0., 1e-12);
    assert!(metric(&r, "Triangle readout defect") > 0.1);
}
#[test]
fn spinor_tie_rule_and_null_mass_identity() {
    let (l, _, col) = edge_spinor([0.; 3], [0.; 3], 1., 1.).unwrap();
    assert_eq!(col, 0);
    close(l[0].re, 1., 1e-12);
    assert!(edge_spinor([0.; 3], [0.; 3], 0., 1.).is_none());
    let r = run(33, json!({"angle":1.2}));
    close(metric(&r, "Determinant–Lorentz residual"), 0., 1e-12);
    close(metric(&r, "Two-spinor mass identity residual"), 0., 1e-12);
}
#[test]
fn fixed_source_slots_give_distinct_lag_readouts() {
    let r = run(35, json!({"steps":24}));
    let a = &r.plots[0].series[0].points;
    let b = &r.plots[0].series[2].points;
    close(a[0][1], b[0][1], 1e-12);
    assert!(
        a.iter()
            .zip(b)
            .skip(1)
            .any(|(x, y)| (x[1] - y[1]).abs() > 1e-4)
    );
}
#[test]
fn whitening_uses_positive_rank_and_correct_spectrum() {
    for amplitude in [0., 0.2, 1., 2.] {
        let r = run(36, json!({"amplitude":amplitude,"rate":0.3,"steps":20}));
        close(
            metric(&r, "Retained covariance rank"),
            if amplitude == 0. { 1. } else { 2. },
            1e-12,
        );
        close(metric(&r, "GEVP eigenvalue residual"), 0., 1e-9);
    }
}

#[test]
fn archive_color_and_twistors_use_executed_stages_and_companion_slots() {
    use crate::{
        domain::{GradientProvider, OperatorFuture, RewardSource},
        kinetic::{KineticKind, KineticOperator, QftExecutionConfig, ViscousForceConfig},
        *,
    };
    struct Zero;
    impl RewardSource<f64> for Zero {
        fn id(&self) -> String {
            "qft-test-zero".into()
        }
        fn evaluate<'a>(
            &'a self,
            p: &'a Population<f64>,
            _: Option<&'a InputBatch<f64>>,
            stage: &'a str,
            _: &'a mut ExecutionContext,
        ) -> OperatorFuture<'a, RewardBatch<f64>> {
            Box::pin(async move {
                Ok(RewardBatch::new(
                    vec![0.; p.len()],
                    Provenance {
                        population_version: p.version,
                        stage: stage.into(),
                        ..Default::default()
                    },
                ))
            })
        }
    }
    impl GradientProvider<f64> for Zero {
        fn id(&self) -> String {
            "qft-test-zero-gradient".into()
        }
        fn gradient<'a>(
            &'a self,
            p: &'a Population<f64>,
            _: &'a mut ExecutionContext,
        ) -> OperatorFuture<'a, TensorBatch<f64>> {
            Box::pin(async move { TensorBatch::vectors(p.len(), 3, vec![0.; p.len() * 3]) })
        }
    }
    futures_lite::future::block_on(async {
        let mut obs = ObservationBatch::positions(
            TensorBatch::vectors(3, 3, vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap(),
        );
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(3, 3, vec![1., 0.2, 0.3, -0.5, 0.1, 0.4, 0., 0.7, -0.3]).unwrap(),
        );
        let config = GasConfig {
            precision: Precision::F64,
            qft: QftExecutionConfig {
                viscosity: Some(ViscousForceConfig {
                    coefficient: 0.3,
                    bandwidth: 1.,
                    row_normalized: false,
                }),
                ..Default::default()
            },
            kinetic: KineticOperator {
                integrator: KineticKind::Baoab {
                    positions: "positions".into(),
                    velocities: "velocities".into(),
                    dt: 0.04,
                    friction: 0.2,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut gas = GasBuilder::new(Population::new(obs).unwrap(), Zero)
            .config(config)
            .gradient(Zero)
            .build()
            .await
            .unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..4 {
            gas.step().await.unwrap();
        }
        let archive = gas.recording().unwrap();
        let mut tampered = archive.clone();
        let influence = tampered
            .steps
            .last_mut()
            .unwrap()
            .influences
            .iter_mut()
            .find(|i| i.field == "viscous_force")
            .unwrap();
        influence.source.generation += 1;
        assert!(
            tampered.validate().is_err(),
            "mutated force-stage source incarnation must be rejected"
        );
        let mut tampered = archive.clone();
        let influence = tampered
            .steps
            .last_mut()
            .unwrap()
            .influences
            .iter_mut()
            .find(|i| i.field == "viscous_force")
            .unwrap();
        influence.source.version += 1;
        assert!(
            tampered.validate().is_err(),
            "mutated force-stage source version must be rejected"
        );
        let color = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 8,
                parameters: json!({}),
            },
            Some(archive),
        )
        .unwrap();
        assert!(metric(&color, "Valid colors") > 0.);
        assert!(color.details["stage"].as_str().unwrap().contains("B1"));
        let record = archive.steps.last().unwrap();
        let force = record
            .field_evaluations
            .iter()
            .find(|x| x.stage == "B1" && x.field == "viscous_force")
            .unwrap();
        close(
            color.details["force"][0][0].as_f64().unwrap(),
            force.values[0],
            1e-12,
        );
        let twistor = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 34,
                parameters: json!({}),
            },
            Some(archive),
        )
        .unwrap();
        assert!(metric(&twistor, "Valid triplets") > 0.);
        assert_eq!(twistor.details["stage"], "pre_clone");
        let lag = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 35,
                parameters: json!({}),
            },
            Some(archive),
        )
        .unwrap();
        assert!(!lag.plots[0].series[0].points.is_empty());
        assert!(
            lag.details["lag_counts"][0]["fixed_valid"]
                .as_u64()
                .unwrap()
                > 0
        );
        let insufficient = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 36,
                parameters: json!({}),
            },
            Some(archive),
        )
        .unwrap();
        assert_eq!(insufficient.details["status"], "unavailable");
        assert!(
            !insufficient
                .metrics
                .iter()
                .any(|m| m.label.contains("decay") || m.label.contains("mass"))
        );
        for _ in 0..92 {
            gas.step().await.unwrap();
        }
        let long_archive = gas.recording().unwrap();
        let gram = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 3,
                parameters: json!({}),
            },
            Some(long_archive),
        )
        .unwrap();
        assert!(metric(&gram, "Gram rank") > 0.);
        assert_eq!(
            gram.details["source_epoch_step_version_slot"]
                .as_array()
                .unwrap()
                .len(),
            288
        );
        for id in [12, 13, 32] {
            let native = partvi::analyze_archive(
                &ExperimentRequest {
                    experiment: id,
                    parameters: json!({"bins":3,"lag":3}),
                },
                Some(long_archive),
            )
            .unwrap();
            assert_eq!(
                native.details["status"], "available",
                "{id}: {:?}",
                native.details
            );
            let train_end = native.details["training_frames"][1].as_u64().unwrap();
            let test_start = native.details["heldout_frames"][0].as_u64().unwrap();
            assert!(test_start > train_end + 3);
            assert!(metric(&native, "Training transitions") > 0.);
        }
        let spectral = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 36,
                parameters: json!({"max_lag":6,"fit_end":4,"channels":3}),
            },
            Some(long_archive),
        )
        .unwrap();
        assert_eq!(spectral.details["status"], "available");
        assert!(metric(&spectral, "Retained covariance rank") > 0.);
        assert!(
            spectral.details["lag_counts"][6]["heldout_pairs"]
                .as_u64()
                .unwrap()
                > 0
        );
        let no_fit = partvi::analyze_archive(
            &ExperimentRequest {
                experiment: 36,
                parameters: json!({"max_lag":6,"fit_start":5,"fit_end":6}),
            },
            Some(long_archive),
        )
        .unwrap_err();
        assert!(no_fit.to_string().contains("at least three lags"));
    });
}

#[test]
fn qsd_window_comparison_enumerates_the_conditioned_path_law() {
    for amplitude in [0., 0.7, 2.] {
        let r = run(27, json!({"amplitude":amplitude,"steps":128}));
        close(metric(&r, "Window bound excess"), 0., 1e-12);
        assert_eq!(r.plots[0].series[3].points.len(), 129);
        assert!(r.plots[0].series[3].points[0][1] > 0.);
        assert!(r.plots[0].series[3].points.last().unwrap()[1] < 1e-6);
    }
}

#[test]
fn effective_twistor_incidence_matches_sigma_matrix_action() {
    let (lambda, mu, _) = edge_spinor([0.2, 0.3, 0.4], [0.5, 0.6, 0.7], 0.1, 0.8).unwrap();
    let expected = [
        C::from(0.5) * lambda[0] + C::new(0.2, -0.3) * lambda[1],
        C::new(0.2, 0.3) * lambda[0] + C::from(-0.3) * lambda[1],
    ];
    close((mu[0] - expected[0]).abs(), 0., 1e-12);
    close((mu[1] - expected[1]).abs(), 0., 1e-12);
    assert!(edge_spinor([f64::NAN, 0., 0.], [0.; 3], 1., 1.).is_none());
}

#[test]
fn multimode_fermionic_channel_has_correct_graded_signs() {
    for modes in 1..=3 {
        let result = run(4, json!({"modes":modes,"rate":0.7}));
        close(
            metric(&result, "Linear CAR contraction residual"),
            0.,
            1e-12,
        );
        close(metric(&result, "Unital residual"), 0., 1e-12);
        assert!(metric(&result, "Minimum Choi eigenvalue") >= -1e-12);
        for p in &result.plots[1].series[0].points {
            close(p[1], 0., 1e-12);
        }
    }
}
#[test]
fn general_complex_spectrum_retains_oscillation_and_repeated_modes() {
    let a = vec![C::ZERO, C::from(-1.), C::ONE, C::ZERO];
    let w = general_eigenvalues(&a, 2);
    assert!(w.iter().all(|z| z.re.abs() < 1e-12));
    assert!(w.iter().any(|z| (z.im - 1.).abs() < 1e-12));
    assert!(w.iter().any(|z| (z.im + 1.).abs() < 1e-12));
    let a = vec![
        C::from(2.),
        C::new(0., 1.),
        C::ZERO,
        C::new(0., -1.),
        C::from(2.),
        C::ZERO,
        C::ZERO,
        C::ZERO,
        C::ONE,
    ];
    let (e, v) = hermitian_modes(&a, 3);
    assert_eq!(e.len(), 3);
    for i in 0..3 {
        for j in 0..3 {
            close(
                (dot(&v[i], &v[j]) - C::from(f64::from(i == j))).abs(),
                0.,
                1e-9,
            );
        }
        let product: Vec<C> = (0..3)
            .map(|row| (0..3).fold(C::ZERO, |s, k| s + a[row * 3 + k] * v[i][k]))
            .collect();
        for k in 0..3 {
            close((product[k] - v[i][k] * e[i]).abs(), 0., 1e-9);
        }
    }
}

#[test]
fn sampled_density_operator_matches_independent_quadrature_with_sampling_error() {
    let mut standardized = vec![];
    for seed in [7, 19, 41, 83] {
        let r = run(7, json!({"amplitude":1.2,"n":4096,"seed":seed}));
        let rows = r.details["sampling_errors"].as_array().unwrap();
        for index in [12, 18, 24] {
            let row = &rows[index];
            let estimate = row["estimate"].as_f64().unwrap();
            let oracle = row["quadrature"].as_f64().unwrap();
            let se = row["standard_error"].as_f64().unwrap();
            assert!(se > 0.);
            standardized.push((estimate - oracle) / se);
        }
        let quadrature = &r.plots[0].series[3].points;
        let target = metric(&r, "Unnormalized continuum target");
        assert!((quadrature[0][1] - target).abs() < (quadrature[24][1] - target).abs());
    }
    // These checks use independent point sets, not a prediction reused as data.
    assert!(standardized.iter().all(|z| z.abs() < 5.));
    let a = run(7, json!({"seed":7,"n":512}));
    let b = run(7, json!({"seed":83,"n":512}));
    assert_ne!(a.plots[0].series[0].points, b.plots[0].series[0].points);
    assert_eq!(a.plots[0].series[3].points, b.plots[0].series[3].points);
}
#[test]
fn euler_gap_bias_refines_while_exact_semigroup_rate_is_fixed() {
    for rate in [0.1, 0.7, 2.] {
        let r = run(25, json!({"rate":rate,"dt":0.2}));
        let target = 2. * rate;
        let euler = &r.plots[1].series[0].points;
        let exact = &r.plots[1].series[1].points;
        assert!(euler.iter().all(|p| p[1] > target));
        assert!(euler.windows(2).all(|w| w[1][1] < w[0][1]));
        assert!((euler.last().unwrap()[1] - target) < 0.02 * (euler[0][1] - target));
        assert!(exact.iter().all(|p| (p[1] - target).abs() < 1e-10));
        let i = euler.len() - 2;
        let error_ratio = (euler[i + 1][1] - target) / (euler[i][1] - target);
        close(error_ratio, 0.8, 0.003);
        for (a, b) in r.plots[0].series[0]
            .points
            .iter()
            .zip(&r.plots[0].series[1].points)
        {
            close(a[1], b[1], 1e-12);
        }
    }
}

#[test]
fn finite_reference_fock_space_quotients_dependent_readouts() {
    let out = super::analyze_dispatch(
        &ExperimentRequest {
            experiment: 3,
            parameters: json!({"modes":6}),
        },
        None,
    )
    .unwrap();
    assert_eq!(metric(&out, "Gram rank"), 2.);
    assert_eq!(metric(&out, "Fock dimension"), 4.);
}
