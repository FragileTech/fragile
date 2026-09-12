use algorithmic_gas::physics::partvi::{ExperimentRequest, ExperimentResult, analyze};
use serde_json::{Value, json};
fn run(experiment: u32, parameters: Value) -> ExperimentResult {
    analyze(&ExperimentRequest {
        experiment,
        parameters,
    })
    .unwrap_or_else(|e| panic!("experiment {experiment}: {e}"))
}
fn metric(r: &ExperimentResult, label: &str) -> f64 {
    r.metrics
        .iter()
        .find(|m| m.label == label)
        .unwrap()
        .value
        .unwrap()
}
#[test]
fn all_thirty_experiments_return_finite_scientific_plots() {
    for id in 37..=66 {
        let r = run(id, json!({}));
        assert!(!r.plots.is_empty(), "{id}");
        assert!(!r.model.is_empty());
        assert!(
            r.plots
                .iter()
                .flat_map(|p| &p.series)
                .flat_map(|s| &s.points)
                .flatten()
                .all(|x| x.is_finite())
        );
    }
}
#[test]
fn slab_transport_preserves_lorentzian_norm_and_converges() {
    let r = run(37, json!({"resolution":64}));
    assert!(metric(&r, "Transport error") < 1e-10);
    assert!(metric(&r, "Metric norm residual") < 1e-10);
    let flat = run(37, json!({"expansion":0}));
    assert_eq!(metric(&flat, "Ambient time component"), 0.);
}
#[test]
fn integrated_sphere_holonomy_recovers_gaussian_curvature() {
    let r = run(38, json!({}));
    assert!((metric(&r, "Finest curvature estimate") - 1.).abs() < 1e-6);
}
#[test]
fn packed_conditional_and_clipped_curvature_match_independent_metric_differences() {
    for mode in ["conditional", "coupled", "separable", "mixed"] {
        let r = run(39, json!({"mode":mode}));
        assert!(
            metric(&r, "Independent residual") < 2e-4,
            "{mode} {}",
            metric(&r, "Independent residual")
        );
        if mode == "separable" {
            assert!(metric(&r, "Packed scalar curvature").abs() < 1e-13);
        }
    }
}
#[test]
fn connection_design_detects_single_direction_nonidentifiability() {
    let r = run(40, json!({"mode":"single_velocity"}));
    assert_eq!(r.details["last_design"]["status"], "rank_deficient");
    assert!(r.details["last_design"].get("coefficients").is_none());
    let eigenvalues = r.details["last_design"]["gram_eigenvalues"]
        .as_array()
        .unwrap();
    assert!(
        eigenvalues
            .iter()
            .any(|x| x.as_f64().unwrap().abs() < 1e-12)
    );
    assert!(
        r.plots
            .iter()
            .all(|p| p.series.iter().any(|s| !s.points.is_empty()))
    );
    let full = run(40, json!({}));
    let points = &full.plots[0].series[0].points;
    assert!(points.last().unwrap()[1] < points[0][1] / 20.);
}
#[test]
fn anisotropic_congruence_retains_the_shear_term() {
    let r = run(41, json!({"anisotropy":0.5}));
    assert!(r.details["shear_squared"].as_f64().unwrap() > 0.1);
    assert!(metric(&r, "Raychaudhuri residual") < 1e-12);
}
#[test]
fn material_volume_refinement_reduces_expansion_error() {
    let r = run(42, json!({}));
    let p = &r.plots[0].series[0].points;
    assert!(p.last().unwrap()[1] < p[0][1] / 100.);
}
#[test]
fn focusing_solution_stays_below_the_comparison() {
    let r = run(43, json!({"ricci":0.3}));
    for (a, b) in r.plots[0].series[0]
        .points
        .iter()
        .zip(&r.plots[0].series[1].points)
    {
        assert!(a[1] <= b[1] + 1e-10);
    }
}
#[test]
fn planar_topology_and_three_dimensional_scaling_are_distinct() {
    let r = run(44, json!({"scale":1}));
    let doubled = run(44, json!({"scale":2}));
    assert!(metric(&r, "Gauss–Bonnet residual") < 1e-12);
    assert!(
        (metric(&doubled, "3D integrated hinge curvature")
            - 2. * metric(&r, "3D integrated hinge curvature"))
        .abs()
            < 1e-12
    );
}
#[test]
fn correlation_energy_has_the_claimed_cubic_remainder() {
    let r = run(45, json!({}));
    assert!(metric(&r, "Free energy") >= 0.);
    assert!(metric(&r, "Quadratic remainder") <= metric(&r, "Taylor remainder bound"));
    let mean = metric(&r, "Independent drift discrepancy");
    assert!(mean.abs() < 6. * metric(&r, "Predicted discrepancy SE"));
}
#[test]
fn dilation_pressure_sign_changes_with_energy_sign() {
    let p = run(46, json!({"mode":"positive"}));
    let a = run(46, json!({"mode":"attractive"}));
    assert!(metric(&p, "Analytic pressure") > 0.);
    assert!((metric(&p, "Analytic pressure") + metric(&a, "Analytic pressure")).abs() < 1e-12);
    assert!(metric(&p, "Derivative residual") < 1e-7);
}
#[test]
fn grid_dispersion_converges_to_the_gaussian_multiplier() {
    let a = run(47, json!({"resolution":32}));
    let b = run(47, json!({"resolution":128}));
    let err = |r: &ExperimentResult| {
        (metric(r, "Continuum Fourier rate") - metric(r, "Grid operator rate")).abs()
    };
    assert!(err(&b) < err(&a) / 10.);
}
#[test]
fn thermostat_gaussian_uniform_and_zero_friction_match_conditional_energy() {
    for (mode, gamma) in [("gaussian", 1.), ("uniform", 1.), ("gaussian", 0.)] {
        let r = run(48, json!({"mode":mode,"friction":gamma}));
        let error = (metric(&r, "Measured mean energy change")
            - metric(&r, "Predicted mean energy change"))
        .abs();
        assert!(error < 6. * metric(&r, "Mean standard error"));
        let expected = r.details["expected_variance"].as_f64().unwrap();
        assert!(metric(&r, "Variance residual").abs() < 0.08 * expected);
    }
}
#[test]
fn mode_pressure_is_a_fixed_mode_set_volume_derivative() {
    let r = run(49, json!({}));
    assert!(metric(&r, "Volume derivative error") < 1e-7);
}
#[test]
fn literal_clone_is_subtracted_from_restitution_stage() {
    let r = run(51, json!({}));
    let p = &r.details["pair_fixture"];
    let literal = p["literal_delta"].as_f64().unwrap();
    let transform = p["transform_delta"].as_f64().unwrap();
    assert!((literal + transform - metric(&r, "Restitution pair energy change")).abs() < 1e-12);
    assert!(metric(&r, "Einstein contraction residual") < 1e-12);
}
#[test]
fn network_flow_equals_the_returned_partition_capacity() {
    let r = run(52, json!({}));
    assert!(metric(&r, "Max-flow / min-cut residual") < 1e-10);
}
#[test]
fn path_kl_checks_both_supported_and_singular_reversal() {
    let r = run(53, json!({}));
    let p = &r.details["path"];
    assert!(
        (p["expected_analytic_kl"].as_f64().unwrap() - p["enumerated_kl"].as_f64().unwrap()).abs()
            < 1e-10
    );
    assert!((metric(&r, "Integral fluctuation expectation") - 1.).abs() < 1e-10);
    assert!(metric(&r, "Modular identity residual") < 1e-12);
    let singular = run(53, json!({"reverse":0}));
    assert_eq!(
        singular.details["path"]["status"],
        "infinite_support_mismatch"
    );
}
#[test]
fn torus_perimeter_retains_two_boundaries_and_density_squared() {
    let r = run(54, json!({"density_contrast":0,"resolution":256}));
    assert!(
        (metric(&r, "Weighted perimeter prediction") - 2. * (2. * std::f64::consts::PI).sqrt())
            .abs()
            < 1e-12
    );
    assert!(
        (metric(&r, "Finest quadrature") / metric(&r, "Weighted perimeter prediction") - 1.).abs()
            < 0.02
    );
}
#[test]
fn independent_population_cut_expectation_has_n_n_minus_one_factor() {
    let r = run(55, json!({}));
    let error =
        (metric(&r, "Replica mean cut") - metric(&r, "Exact pair-integral expectation")).abs();
    assert!(error < 6. * metric(&r, "Replica mean SE"));
    assert_eq!(r.details["finite_pair_factor"], 64 * 63);
}
#[test]
fn first_variation_retains_density_gradient() {
    let r = run(56, json!({}));
    assert!(metric(&r, "Shape derivative residual") < 1e-6);
}
#[test]
fn gibbs_qsd_error_is_bounded_by_normalized_residual() {
    for c in [0., 0.3] {
        let r = run(57, json!({"killing_contrast":c}));
        assert!(metric(&r, "Verified contraction bound") < 1.);
        assert!(metric(&r, "Candidate TV error") <= metric(&r, "Residual-based TV bound") + 1e-12);
    }
}
#[test]
fn poisson_response_and_fishers_match_exact_two_state_formulas() {
    let r = run(58, json!({}));
    let expected = metric(&r, "Exact response");
    assert!((metric(&r, "Poisson response") - expected).abs() < 1e-12);
    assert!((metric(&r, "Independent stationary finite difference") - expected).abs() < 1e-5);
    assert!(
        (metric(&r, "Stationary Fisher") - r.details["exact_stationary_fisher"].as_f64().unwrap())
            .abs()
            < 1e-12
    );
    assert!(
        (metric(&r, "Transition Fisher") - r.details["exact_transition_fisher"].as_f64().unwrap())
            .abs()
            < 1e-12
    );
}
#[test]
fn constrained_minimum_matches_independent_vertical_competitors() {
    let r = run(60, json!({}));
    assert!(
        (metric(&r, "Network minimum") - metric(&r, "Independent straight-interface minimum"))
            .abs()
            < 1e-9
    );
}
#[test]
fn centered_qsd_observable_is_invariant_to_energy_zero() {
    let a = run(61, json!({"energy_shift":0}));
    let b = run(61, json!({"energy_shift":3}));
    assert!(metric(&a, "Centered QSD expectation").abs() < 1e-15);
    for (p, q) in a.plots[0].series[0]
        .points
        .iter()
        .zip(&b.plots[0].series[0].points)
    {
        assert!((p[1] - q[1]).abs() < 1e-14);
    }
}
#[test]
fn hidden_bit_has_one_bit_of_predictive_information() {
    let r = run(64, json!({}));
    assert!((metric(&r, "Measured discarded predictive information") - 2_f64.ln()).abs() < 0.002);
    assert_eq!(metric(&r, "Macro causal states"), 1.);
    let no = run(64, json!({"reveal_probability":0}));
    assert_eq!(metric(&no, "Measured discarded predictive information"), 0.);
}
#[test]
fn dynkin_residual_vanishes_for_lumpable_projection_and_obeys_bound() {
    for delta in [0., 0.5] {
        let r = run(65, json!({"rate_perturbation":delta}));
        let bound = metric(&r, "Generator residual bound");
        for s in &r.plots[0].series[..2] {
            for p in &s.points {
                assert!(p[1].abs() <= p[0] * bound + 1e-10);
            }
        }
    }
}
#[test]
fn vacuum_unit_conversion_preserves_the_c_squared_factor() {
    let r = run(66, json!({}));
    let c = 299_792_458_f64;
    assert!(
        (metric(&r, "Vacuum energy density") / metric(&r, "Vacuum mass density") / c.powi(2) - 1.)
            .abs()
            < 1e-14
    );
    let zero = run(66, json!({"omega":0}));
    assert!(
        zero.metrics
            .iter()
            .find(|m| m.label == "Equation-of-state w")
            .unwrap()
            .value
            .is_none()
    );
}

#[test]
fn genealogical_separator_rejects_simultaneously_observed_ancestor_and_child() {
    let root = run(52, json!({"separator":"roots"}));
    let terminal = run(52, json!({"separator":"terminal_episodes"}));
    assert_eq!(
        root.details["genealogical_separator"]["is_separating_antichain"],
        true
    );
    assert_eq!(
        terminal.details["genealogical_separator"]["is_separating_antichain"],
        false
    );
    assert_eq!(
        terminal.details["genealogical_separator"]["chain_intersection_counts"],
        json!([1, 2, 2])
    );
}
#[test]
fn all_three_variation_calculations_are_present() {
    let r = run(56, json!({}));
    assert!(metric(&r, "Density-matrix first-law residual") < 1e-6);
    assert!(
        r.details["density_matrix_variation"]["noncommuting"]
            .as_bool()
            .unwrap()
    );
    let d = &r.details["density_variation"];
    assert!(
        (d["analytic_derivative"].as_f64().unwrap() - d["difference_derivative"].as_f64().unwrap())
            .abs()
            < 1e-10
    );
}

#[test]
fn every_advertised_field_control_endpoint_has_a_reviewable_result() {
    let controls: Value =
        serde_json::from_str(include_str!("../src/physics/fields-controls.json")).unwrap();
    for (id, rows) in controls.as_object().unwrap() {
        let id = id.parse::<u32>().unwrap();
        for row in rows.as_array().unwrap() {
            let key = row["key"].as_str().unwrap();
            let values = if row["type"] == "select" {
                row["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|o| o["value"].clone())
                    .collect::<Vec<_>>()
            } else {
                vec![row["min"].clone(), row["max"].clone()]
            };
            for value in values {
                let mut parameters = json!({});
                parameters[key] = value.clone();
                let result = analyze(&ExperimentRequest {
                    experiment: id,
                    parameters: parameters.clone(),
                })
                .unwrap_or_else(|e| panic!("experiment {id}, {parameters}: {e}"));
                assert!(!result.plots.is_empty(), "{id} {key} {value}");
            }
        }
    }
}
#[test]
fn indefinite_metric_endpoint_reports_raw_spectrum_without_inventing_curvature() {
    let r = run(39, json!({"epsilon":0.1}));
    if r.details["status"] == "curvature_unavailable" {
        assert_eq!(r.details["raw_eigenvalues"].as_array().unwrap().len(), 3);
        assert!(
            r.metrics
                .iter()
                .find(|m| m.label == "Packed scalar curvature")
                .unwrap()
                .value
                .is_none()
        );
    }
}

#[test]
fn rigid_rotation_retains_twist_and_acceleration_divergence() {
    for dimension in [2, 3] {
        for speed in [0., 0.4, 0.8] {
            let r = run(
                41,
                json!({"mode":"rotation","dimension":dimension,"rotation_speed":speed,"radius":0.2}),
            );
            assert!(metric(&r, "Raychaudhuri residual") < 2e-6, "{}", r.details);
            assert!(metric(&r, "Vorticity contraction residual") < 2e-6);
            assert!(r.details["shear_squared"].as_f64().unwrap().abs() < 1e-15);
            assert!(r.details["theta"].as_f64().unwrap().abs() < 1e-12);
            let expected = 2. * (speed / 0.2_f64).powi(2) / (1. - speed * speed).powi(2);
            assert!(
                (r.details["acceleration_divergence"].as_f64().unwrap() + expected).abs() < 2e-6
            );
            let u = r.details["unit_velocity"].as_array().unwrap();
            let norm = -u[0].as_f64().unwrap().powi(2)
                + u[1..]
                    .iter()
                    .map(|v| v.as_f64().unwrap().powi(2))
                    .sum::<f64>();
            assert!((norm + 1.).abs() < 1e-12);
        }
    }
}

#[test]
fn prescribed_shear_shortens_the_constant_forcing_focusing_time() {
    let free = run(43, json!({"ricci":0.,"shear_squared":0.}));
    let sheared = run(43, json!({"ricci":0.,"shear_squared":0.7}));
    assert!(
        metric(&sheared, "Constant-forcing focusing time")
            < metric(&free, "Constant-forcing focusing time")
    );
    for (measured, bound) in sheared.plots[0].series[0]
        .points
        .iter()
        .zip(&sheared.plots[0].series[1].points)
    {
        assert!(measured[1] <= bound[1] + 1e-10);
        let d = 3_f64;
        let forcing = 0.7_f64;
        let theta0 = -1_f64;
        let exact = (d * forcing).sqrt()
            * ((theta0 / (d * forcing).sqrt()).atan() - (forcing / d).sqrt() * measured[0]).tan();
        assert!((measured[1] - exact).abs() < 1e-6);
    }
}

#[test]
fn closed_sphere_has_no_boundary_and_total_curvature_four_pi_at_every_refinement() {
    for refinement in 0..=3 {
        for scale in [0.1, 4.] {
            let r = run(
                44,
                json!({"surface":"sphere","refinement":refinement,"scale":scale}),
            );
            assert_eq!(metric(&r, "Euler characteristic"), 2.);
            assert_eq!(metric(&r, "Boundary edge count"), 0.);
            assert!(metric(&r, "Gauss–Bonnet residual") < 2e-11);
            assert!(
                (metric(&r, "Total intrinsic deficit") - 4. * std::f64::consts::PI).abs() < 2e-11
            );
            assert_eq!(r.details["all_edges_have_two_incident_faces"], true);
            assert!(
                r.details["deficits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|x| x.as_f64().unwrap() > 0.)
            );
        }
    }
}

#[test]
fn curvature_reports_the_effective_spectral_parameters_in_every_domain() {
    for mode in ["conditional", "mixed"] {
        for epsilon in [0.1, 3.] {
            let r = run(39, json!({"mode":mode,"epsilon":epsilon}));
            assert_eq!(r.details["epsilon"], epsilon);
            assert_eq!(r.details["clipping_threshold"], 1e-8);
            assert_eq!(
                r.details["policy"],
                if mode == "mixed" { "clipped" } else { "strict" }
            );
        }
    }
}
