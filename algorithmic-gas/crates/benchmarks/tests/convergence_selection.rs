use algorithmic_gas_benchmarks::convergence_selection::{
    DriftAssemblyInputs, analyze_conditional_selection, analyze_fixed_rescaling,
    assemble_diagonal_drift, cloning_assembly_offsets, complete_linkage, log_gap_bounds,
    optimal_centered_transport_1d, validate_selection,
};

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
fn log_mean(values: &[f64]) -> f64 {
    mean(&values.iter().map(|x| x.ln()).collect::<Vec<_>>())
}
fn all_pass(checks: &[algorithmic_gas_benchmarks::convergence_framework::BoundCheck]) {
    let failures: Vec<_> = checks.iter().filter(|check| !check.passed).collect();
    assert!(failures.is_empty(), "failed checks: {failures:#?}");
}

#[test]
fn complete_linkage_retains_small_clusters_and_respects_complete_diameter() {
    // Single linkage would join all first three rows through the middle row.
    let x: [f64; 4] = [0., 0.09, 0.18, 2.];
    let distances: Vec<Vec<f64>> = x
        .iter()
        .map(|a| x.iter().map(|b| (a - b).abs()).collect())
        .collect();
    let groups = complete_linkage(&distances, 0.1).unwrap();
    assert_eq!(groups, vec![vec![0, 1], vec![2], vec![3]]);
    for group in &groups {
        for &i in group {
            for &j in group {
                assert!(distances[i][j] <= 0.1);
            }
        }
    }
    let mut asymmetric = distances.clone();
    asymmetric[1][0] += 0.1;
    assert!(complete_linkage(&asymmetric, 0.1).is_err());
    assert!(complete_linkage(&[vec![0., 1.], vec![0.]], 0.1).is_err());
    assert!(complete_linkage(&distances, 0.).is_err());
}

#[test]
fn shared_rescale_propagates_variance_but_constant_input_remains_constant() {
    let varying = analyze_fixed_rescaling(&[-0.8, -0.2, 0.1, 0.7], 0.1, 0.1).unwrap();
    all_pass(&varying.checks);
    assert!(varying.constants["variance_lower"] > 0.);
    assert!(varying.observations["rescaled_variance"].as_f64().unwrap() > 0.);
    let constant = analyze_fixed_rescaling(&[0.5, 0.5, 0.5], 0.1, 0.1).unwrap();
    all_pass(&constant.checks);
    assert_eq!(constant.observations["rescaled_variance"], 0.);
    assert_eq!(constant.constants["variance_lower"], 0.);
    assert!(
        !constant
            .checks
            .iter()
            .any(|c| c.id == "variance_to_raw_gap")
    );
    assert!(analyze_fixed_rescaling(&[0., 1.], 0., 0.1).is_err());
}

#[test]
fn mean_gap_log_bounds_allow_negative_lower_and_equal_mean_log_discrepancy() {
    let equal_mean = log_gap_bounds(0.1, 2.1, 0.).unwrap();
    assert!(equal_mean.lower < 0.);
    assert!(equal_mean.upper > 0.);
    // A mean alone does not determine log expectation. This equal-mean pair
    // exercises the Jensen defect instead of assuming logarithms commute.
    let endpoints = [0.1, 2.1];
    let constant = [1.1, 1.1];
    let actual = log_mean(&constant) - log_mean(&endpoints);
    assert!(actual > 0.);
    assert!(actual <= equal_mean.upper + 1e-12);
    assert!(-actual >= equal_mean.lower - 1e-12);
    for a in [0.1, 0.5, 1.] {
        let b = a + 2.;
        for i in 0..=20 {
            let kappa = 2. * i as f64 / 20.;
            let bound = log_gap_bounds(a, b, kappa).unwrap();
            for j in 0..=200 {
                let u = a + (b - a - kappa) * j as f64 / 200.;
                let mixture_upper =
                    (u + kappa).ln() - ((b - u) * a.ln() + (u - a) * b.ln()) / (b - a);
                let mixture_lower =
                    ((b - u - kappa) * a.ln() + (u + kappa - a) * b.ln()) / (b - a) - u.ln();
                assert!(mixture_lower >= bound.lower - 1e-12);
                assert!(mixture_upper <= bound.upper + 1e-12);
            }
        }
    }
    assert!(log_gap_bounds(0., 2., 0.1).is_err());
    assert!(log_gap_bounds(0.1, 2.1, 2.2).is_err());
}

#[test]
fn log_gap_bounds_keep_the_tiny_maximal_gap_endpoint() {
    let a = 1e-8_f64;
    let b = 1e-6_f64;
    assert!(b - (b - a) < a);
    let bounds = log_gap_bounds(a, b, b - a).unwrap();
    let exact = (b / a).ln();
    assert!((bounds.lower - exact).abs() < 1e-12);
    assert!((bounds.upper - exact).abs() < 1e-12);
}

#[test]
fn retained_pressure_respects_clipping_nonself_kernel_and_common_labels() {
    let fitness = [0.2, 0.9, 1.2, 1.8];
    let kernel = vec![
        vec![0., 0.6, 0.2, 0.2],
        vec![0.1, 0., 0.2, 0.7],
        vec![0.4, 0.2, 0., 0.4],
        vec![0.2, 0.3, 0.5, 0.],
    ];
    let report = analyze_conditional_selection(
        &fitness,
        &kernel,
        &[true, true, false, false],
        &[false, true, true, true],
        &[4., 1., 0.5, 0.1],
        1.8,
        0.3,
        1e-6,
    )
    .unwrap();
    all_pass(&report.checks);
    // Lowest-fitness row clips to acceptance one for every eligible donor.
    assert!(
        (report.observations["acceptance_probabilities"][0]
            .as_f64()
            .unwrap()
            - 1.)
            .abs()
            < 1e-12
    );
    assert_eq!(report.observations["acceptance_probabilities"][3], 0.);
    assert_eq!(report.observations["target"], serde_json::json!([1]));
    assert!(report.constants["p_u"] > 0.);
    let tie = analyze_conditional_selection(
        &[1.; 4],
        &kernel,
        &[true, true, false, false],
        &[true; 4],
        &[1.; 4],
        1.,
        1.,
        1e-6,
    )
    .unwrap();
    all_pass(&tie.checks);
    assert_eq!(tie.constants["p_u"], 0.);
    assert_eq!(
        tie.observations["acceptance_probabilities"],
        serde_json::json!([0., 0., 0., 0.])
    );
    assert!(!tie.checks.iter().any(|c| c.id == "unfit_fraction"));
    assert!(
        !tie.checks
            .iter()
            .any(|c| c.id.starts_with("target_pressure_"))
    );
    let mut self_kernel = kernel.clone();
    self_kernel[0][0] = 0.01;
    assert!(
        analyze_conditional_selection(
            &fitness,
            &self_kernel,
            &[true; 4],
            &[true; 4],
            &[1.; 4],
            1.8,
            1.,
            0.
        )
        .is_err()
    );
}

fn drift_inputs() -> DriftAssemblyInputs {
    DriftAssemblyInputs {
        observables: vec![[1., 0.2, 0.1, 0.1], [2., 0.8, 0.6, 0.5]],
        cloning: vec![vec![0.8, 0.2], vec![0.3, 0.7]],
        kinetic: vec![vec![0.6, 0.2], vec![0.1, 0.6]],
        cloning_rates: [0.2, 0.4],
        kinetic_rates: [0.3, 0.5],
        cloning_offsets: [1., 1., 1., 1.],
        kinetic_offsets: [0.8, 0.8, 0.8, 0.8],
        variance_weight: 0.7,
        boundary_weight: 1.3,
        steps: 5,
        clone_velocity_dissipation: None,
    }
}

#[test]
fn finite_composition_retains_stage_order_offsets_defects_and_survival() {
    let input = drift_inputs();
    let report = assemble_diagonal_drift(&input).unwrap();
    all_pass(&report.checks);
    assert!((report.composed_transition[0][0] - 0.5).abs() < 1e-12);
    assert!((report.composed_transition[0][1] - 0.28).abs() < 1e-12);
    // PK*PC gives .54 in the first entry, so this detects the wrong stage order.
    assert!((report.composed_transition[0][0] - 0.54).abs() > 1e-3);
    let explicit_c = 0.7 * 1. + 0.8 + 0.7 * (1. + 0.8 + 0.5 * 1. + 0.8) + 1.3 * (1. + 0.8);
    assert!((report.c_star - explicit_c).abs() < 1e-12);
    assert!((report.kappa_star - 0.2).abs() < 1e-12);
    for state in 0..2 {
        assert!(report.surviving_mass[state] > 0. && report.surviving_mass[state] < 1.);
        let conditional = report.conditioned_moments[state].unwrap();
        assert!(
            (conditional * report.surviving_mass[state]
                - report.iterated_unnormalized_moments[state])
                .abs()
                < 1e-12
        );
    }
    let mut weighted = input.clone();
    weighted.variance_weight *= 5.;
    weighted.boundary_weight *= 2.;
    let weighted_report = assemble_diagonal_drift(&weighted).unwrap();
    all_pass(&weighted_report.checks);
    assert_eq!(weighted_report.kappa_star, report.kappa_star);
    assert!(weighted_report.c_star > report.c_star);
}

#[test]
fn submarkov_cloning_includes_lost_constant_and_extinction_is_unconditioned() {
    let mut input = drift_inputs();
    input.cloning[0].iter_mut().for_each(|p| *p *= 0.6);
    input.cloning[1].iter_mut().for_each(|p| *p *= 0.8);
    let report = assemble_diagonal_drift(&input).unwrap();
    all_pass(&report.checks);
    assert!(
        report
            .checks
            .iter()
            .any(|c| c.id == "composed_component_defect_state_0_0" && c.observed < 1e-12)
    );
    input.kinetic = vec![vec![0.; 2]; 2];
    let extinct = assemble_diagonal_drift(&input).unwrap();
    all_pass(&extinct.checks);
    assert_eq!(extinct.surviving_mass, vec![0.; 2]);
    assert_eq!(extinct.iterated_unnormalized_moments, vec![0.; 2]);
    assert_eq!(extinct.conditioned_moments, vec![None; 2]);
}

#[test]
fn false_component_hypotheses_are_reported_as_failures_and_invalid_rates_rejected() {
    let mut input = drift_inputs();
    input.cloning_offsets = [0.; 4];
    input.kinetic_offsets = [0.; 4];
    let report = assemble_diagonal_drift(&input).unwrap();
    assert!(
        report
            .checks
            .iter()
            .any(|c| c.id.starts_with("clone_component_") && !c.passed)
    );
    input.cloning_rates[0] = 0.;
    assert!(assemble_diagonal_drift(&input).is_err());
    input.cloning_rates[0] = 0.2;
    input.kinetic[0][0] = 1.1;
    assert!(assemble_diagonal_drift(&input).is_err());
}

#[test]
fn positive_survival_underflow_is_not_reported_as_extinction() {
    let mut input = drift_inputs();
    input.cloning = vec![vec![1., 0.], vec![0., 1.]];
    input.kinetic = vec![vec![1e-300, 0.], vec![0., 1e-300]];
    input.steps = 2;
    let error = assemble_diagonal_drift(&input).unwrap_err().to_string();
    assert!(error.contains("surviving mass underflowed"));
    input.cloning = vec![vec![1e-300, 0.], vec![0., 1e-300]];
    let error = assemble_diagonal_drift(&input).unwrap_err().to_string();
    assert!(error.contains("transition probability underflowed"));
}

#[test]
fn uniform_clone_offsets_keep_revival_velocity_term() {
    let offsets = cloning_assembly_offsets(3, 2., 0.1, 0.5, 2., 0.4, 0.2).unwrap();
    assert!((offsets[1] - 4.06).abs() < 1e-12);
    assert_eq!(offsets[2], 16.);
    assert!(cloning_assembly_offsets(3, 2., 0.1, 0., 2., 0.4, 0.2).is_err());
}

#[test]
fn empirical_transport_and_conditional_aggregate_ignore_storage_permutations() {
    let original = [0., 1., 1., 1.];
    let reflected = [0., -1., -1., -1.];
    let cost = optimal_centered_transport_1d(&original, &reflected).unwrap();
    assert!((cost.squared_wasserstein - 0.25).abs() < 1e-12);
    let permuted = optimal_centered_transport_1d(&[1., 0., 1., 1.], &[-1., -1., 0., -1.]).unwrap();
    assert_eq!(permuted.squared_wasserstein, cost.squared_wasserstein);
    let same_swarm = optimal_centered_transport_1d(&original, &[1., 1., 0., 1.]).unwrap();
    assert_eq!(same_swarm.squared_wasserstein, 0.);
    let fitness = [0.2, 0.7, 1.3, 1.9];
    let kernel = vec![
        vec![0., 0.2, 0.3, 0.5],
        vec![0.4, 0., 0.3, 0.3],
        vec![0.2, 0.6, 0., 0.2],
        vec![0.5, 0.1, 0.4, 0.],
    ];
    let high = [true, true, false, false];
    let common = [true, false, true, true];
    let errors = [0.1, 0.8, 0.3, 0.7];
    let original_report =
        analyze_conditional_selection(&fitness, &kernel, &high, &common, &errors, 1.9, 1., 1e-6)
            .unwrap();
    let permutation = [2_usize, 0, 3, 1];
    let reordered_kernel: Vec<Vec<f64>> = permutation
        .iter()
        .map(|&i| permutation.iter().map(|&j| kernel[i][j]).collect())
        .collect();
    let reordered_report = analyze_conditional_selection(
        &permutation.map(|i| fitness[i]),
        &reordered_kernel,
        &permutation.map(|i| high[i]),
        &permutation.map(|i| common[i]),
        &permutation.map(|i| errors[i]),
        1.9,
        1.,
        1e-6,
    )
    .unwrap();
    all_pass(&original_report.checks);
    all_pass(&reordered_report.checks);
    for key in ["target_error", "weighted_pressure_error"] {
        assert!(
            (original_report.observations[key].as_f64().unwrap()
                - reordered_report.observations[key].as_f64().unwrap())
            .abs()
                < 1e-12
        );
    }
    for (key, value) in original_report.constants {
        assert!((value - reordered_report.constants[&key]).abs() < 1e-12);
    }
}

#[test]
fn complete_report_retains_invalid_and_tie_outcomes_and_applies_only_matching_mass_scope() {
    let report = validate_selection(20261002).unwrap();
    all_pass(&report.checks);
    assert_eq!(report.fixtures.len(), 6);
    let geometry = report
        .fixtures
        .iter()
        .find(|f| f.name == "finite_cluster_geometry_and_target")
        .unwrap();
    let clusters = geometry.observations["clusters"].as_array().unwrap();
    let mut sizes: Vec<_> = clusters
        .iter()
        .map(|g| g.as_array().unwrap().len())
        .collect();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![3, 5, 5, 5]);
    let high = geometry.observations["high"].as_array().unwrap();
    for group in clusters.iter().filter(|g| g.as_array().unwrap().len() < 5) {
        for index in group.as_array().unwrap() {
            assert_eq!(high[index.as_u64().unwrap() as usize], true);
        }
    }
    let averaged = report
        .fixtures
        .iter()
        .find(|f| f.name == "exact_measurement_averaged_small_clusters")
        .unwrap();
    assert!(
        (averaged.observations["measurement_probability_sum"]
            .as_f64()
            .unwrap()
            - 1.)
            .abs()
            < 1e-12
    );
    assert!(
        averaged.observations["equal_fitness_outcome_probability"]
            .as_f64()
            .unwrap()
            > 0.
    );
    assert!(averaged.constants["p_star"] > 0.);
    assert_eq!(
        averaged.constants["optimal_positional_transport_cost"],
        0.25
    );
    assert_eq!(
        averaged.observations["clusters"][0]["eligible_for_AP5"],
        false
    );
    assert_eq!(averaged.observations["row_pressure_certificates"][0], 0.);
    assert!(
        !averaged
            .checks
            .iter()
            .any(|c| c.id == "averaged_small_cluster_pressure_0")
    );
    assert!(!averaged.checks.iter().any(|c| c.id.contains("valid_mass")));
    assert!(
        averaged
            .checks
            .iter()
            .any(|c| c.id == "explicit_uncovered_error")
    );
    let labels: Vec<_> = report
        .checks
        .iter()
        .flat_map(|c| &c.source_labels)
        .collect();
    for expected in [
        "prop-fixed-rescale-variance-bound",
        "thm-stability-condition-final-corrected",
        "lem-error-concentration-target-set",
        "thm-keystone-averaged-error-capture",
        "cor-cloning-weighted-assembly-input",
    ] {
        assert!(
            labels.iter().any(|label| label.as_str() == expected),
            "missing {expected}"
        );
    }
}
