use algorithmic_gas::{
    GasBuilder, GasConfig, ObservationBatch, Population, TensorBatch, cloning::CloneDecision,
};
use algorithmic_gas_benchmarks::{
    Benchmark, BenchmarkModel, RunConfig,
    convergence_cloning::{
        analyze_cloning_step, canonical_cloning_constants, canonical_keystone_constants,
        canonical_two_cluster_acceptance_floor, exact_balanced_two_site_fixture,
        exact_cloning_moments, exact_four_walker_fixture, validate_independent_cloning,
    },
};

#[test]
fn canonical_constants_match_the_explicit_balanced_keystone_rate() {
    let constants = canonical_cloning_constants();
    assert!((constants.fitness_min - 0.01).abs() < 1e-15);
    assert!((constants.fitness_max - 4.41).abs() < 1e-15);
    assert!((constants.kappa_c - (-4_f64).exp()).abs() < 1e-15);
    assert!((constants.balanced_two_cluster_chi_0 - 0.30251915319).abs() < 1e-11);
    for index in 0..150 {
        assert!(
            canonical_two_cluster_acceptance_floor(0.5 + index as f64 / 100.)
                >= canonical_two_cluster_acceptance_floor(0.5)
        );
    }
}

#[test]
fn complete_keystone_constants_retain_small_positive_rates_in_logarithms() {
    let small = canonical_keystone_constants(1, 0.1, 2.).unwrap();
    assert!(small.values["omega_f"] >= small.values["omega_f_derivative_lower_bound"]);
    assert!(small.values["r"] <= small.values["h_f"] / 2.);
    assert!(small.values["chi_0"] > 0.);
    let large = canonical_keystone_constants(8, 0.1, 2. * 8_f64.sqrt()).unwrap();
    assert!(large.natural_logs["chi_0"].is_finite());
    assert!(large.natural_logs["chi_0"] < (-1000.));
    assert!(!large.values.contains_key("chi_0"));
    assert!(canonical_keystone_constants(1, 0., 2.).is_err());
}

#[test]
fn four_walker_full_measurement_enumeration_preserves_positive_drift_regression() {
    let no_jitter = exact_four_walker_fixture(0.).unwrap();
    let jitter = exact_four_walker_fixture(0.1).unwrap();
    assert!((no_jitter.expected_position_variance - 0.002009271535689).abs() < 1e-13);
    assert!((jitter.expected_position_variance - 0.00297011744615).abs() < 1e-13);
    assert!(jitter.expected_position_drift > 0.);
    assert!(
        (jitter.expected_position_variance
            - no_jitter.expected_position_variance
            - jitter.jitter_variance_contribution)
            .abs()
            < 1e-14
    );
    assert!(jitter.measurement_barycenter_variance > 0.);
    assert!(
        (jitter.total_barycenter_variance
            - jitter.measurement_barycenter_variance
            - jitter.conditional_barycenter_variance)
            .abs()
            < 1e-15
    );
    assert!(jitter.checks.iter().all(|check| check.passed));
}

#[test]
fn singleton_revival_discards_arbitrary_dead_positions_and_keeps_noise_contribution() {
    let result = exact_cloning_moments(
        &[vec![1e100], vec![2.], vec![-1e100]],
        &[0., 1., 0.],
        &[vec![0., 1., 0.], vec![0., 1., 0.], vec![0., 1., 0.]],
        &[false, true, false],
        &CloneDecision::default(),
        1,
        0.1,
    )
    .unwrap();
    assert_eq!(result.acceptance_probabilities, vec![1., 0., 1.]);
    assert_eq!(result.output_means, vec![vec![2.], vec![2.], vec![2.]]);
    assert_eq!(result.entering_alive_variance, 0.);
    assert!((result.expected_position_variance - 4. / 9. * 0.01).abs() < 1e-15);
    assert!(result.checks.iter().all(|check| check.passed));
}

#[test]
fn frozen_gate_singleton_ties_and_disabled_period_are_separate_branches() {
    let singleton = exact_cloning_moments(
        &[vec![3., 4.]],
        &[1.],
        &[vec![1.]],
        &[true],
        &CloneDecision::default(),
        1,
        10.,
    )
    .unwrap();
    assert_eq!(singleton.expected_position_variance, 0.);
    assert_eq!(singleton.conditional_barycenter_variance, 0.);
    assert!(singleton.checks.iter().all(|check| check.passed));
    let disabled = exact_cloning_moments(
        &[vec![0.], vec![2.]],
        &[1., 2.],
        &[vec![0., 1.], vec![1., 0.]],
        &[true, true],
        &CloneDecision {
            every: 2,
            ..Default::default()
        },
        1,
        0.1,
    )
    .unwrap();
    assert_eq!(disabled.acceptance_probabilities, vec![0., 0.]);
    assert_eq!(disabled.expected_position_drift, 0.);
    assert!(disabled.checks.iter().all(|check| check.passed));
}

#[test]
fn live_conditional_flux_identity_includes_the_moving_barycenter() {
    let result = exact_cloning_moments(
        &[vec![-1.], vec![0.], vec![2.]],
        &[1., 4., 2.],
        &[vec![0., 0.5, 0.5], vec![0.5, 0., 0.5], vec![0.5, 0.5, 0.]],
        &[true, true, true],
        &CloneDecision::default(),
        1,
        0.1,
    )
    .unwrap();
    assert!(result.collective_flux_identity_residual.unwrap().abs() < 1e-14);
    assert!(result.checks.iter().all(|check| check.passed));
}

#[test]
fn real_engine_step_supplies_matching_fitness_gate_moments_and_collision_checks() {
    futures_lite::future::block_on(async {
        let config = RunConfig {
            walkers: 8,
            dimensions: 2,
            gas: GasConfig::euclidean(2, 0.04).unwrap(),
            ..RunConfig::euclidean().unwrap()
        };
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(Default::default()).unwrap();
        gas.step().await.unwrap();
        let report = analyze_cloning_step(gas.recording().unwrap(), 0).unwrap();
        assert!(report.moments.is_some());
        assert!(report.unavailable.is_empty(), "{:?}", report.unavailable);
        assert!(
            report.checks.iter().all(|check| check.passed),
            "{:?}",
            report
                .checks
                .iter()
                .filter(|c| !c.passed)
                .collect::<Vec<_>>()
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| check.id == "component_energy_restitution")
        );
        assert!(report.checks.iter().all(|check| {
            !check
                .source_labels
                .iter()
                .any(|label| label == "lem-dead-walker-clone-prob")
        }));
    });
}

#[test]
fn malformed_probabilities_and_empty_alive_pool_are_rejected() {
    assert!(
        exact_cloning_moments(
            &[vec![1.]],
            &[1.],
            &[vec![0.5]],
            &[true],
            &CloneDecision::default(),
            1,
            0.
        )
        .is_err()
    );
    assert!(
        exact_cloning_moments(
            &[vec![1.]],
            &[0.],
            &[vec![1.]],
            &[false],
            &CloneDecision::default(),
            1,
            0.
        )
        .is_err()
    );
}

#[test]
fn exact_balanced_measurement_law_obeys_its_pressure_and_noise_bounds() {
    for walkers in [4, 8] {
        for radius in [0.5, 1., 1.9] {
            let result = exact_balanced_two_site_fixture(walkers, radius, 0.1).unwrap();
            assert!(
                result.checks.iter().all(|check| check.passed),
                "{:?}",
                result.checks
            );
            assert!(result.acceptance_probabilities.iter().all(|p| *p > 0.));
            assert!(result.output_means.iter().map(|v| v[0]).sum::<f64>().abs() < 1e-13);
        }
    }
}

#[test]
fn independent_actual_engine_ensembles_match_exact_measurement_moments() {
    let result =
        futures_lite::future::block_on(validate_independent_cloning(128, 8_700_000)).unwrap();
    assert_eq!(result.cases.len(), 8);
    assert!(
        result.all_consistent,
        "{:?}",
        result
            .cases
            .iter()
            .filter(|case| !case.all_consistent)
            .collect::<Vec<_>>()
    );
    assert!(
        result.cases[1]
            .positional_variance
            .exact_expectation
            .unwrap()
            > result.cases[1].input_variance
    );
    for case in &result.cases {
        let balanced = case.name.starts_with("balanced_");
        assert_eq!(
            case.acceptance_pressure
                .source_labels
                .iter()
                .any(|label| label == "cor-keystone-canonical-balanced-structural"),
            balanced,
        );
        assert_eq!(
            case.positional_variance
                .source_labels
                .iter()
                .any(|label| label == "prop-cloning-two-cluster-noise-balance"),
            balanced,
        );
        assert_eq!(
            case.positional_variance
                .source_labels
                .iter()
                .any(|label| label == "ex-cloning-position-spreading"),
            !balanced,
        );
    }
}

#[test]
fn balanced_fixture_below_half_radius_does_not_claim_the_balanced_structural_rate() {
    let report = exact_balanced_two_site_fixture(4, 0.1, 0.1).unwrap();
    assert!(report.checks.iter().all(|check| check.passed));
    assert!(report.checks.iter().all(|check| {
        !check
            .source_labels
            .iter()
            .any(|label| label == "cor-keystone-canonical-balanced-structural")
    }));
}

#[test]
fn actual_absorbing_engine_revives_dead_rows_from_the_live_weighted_kernel() {
    futures_lite::future::block_on(async {
        let repetitions = 512;
        let mut donor_one = 0;
        let mut expected = None;
        for seed in 0..repetitions {
            let mut config = GasConfig::euclidean(1, 0.0001).unwrap();
            config.seed = seed as u64 + 900_000;
            let mut observations = ObservationBatch::positions(
                TensorBatch::vectors(4, 1, vec![-10., -0.5, 0.5, 10.]).unwrap(),
            );
            observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(4, 1, vec![0.; 4]).unwrap(),
            );
            let population = Population::new(observations).unwrap();
            let model = BenchmarkModel {
                benchmark: Benchmark::Quadratic,
                field: "positions".into(),
                direction: config.fitness.direction,
            };
            let mut gas = GasBuilder::new(population, model.clone())
                .gradient(model)
                .config(config)
                .build()
                .await
                .unwrap();
            gas.start_recording(Default::default()).unwrap();
            gas.step().await.unwrap();
            let archive = gas.recording().unwrap();
            let step = &archive.steps[0];
            assert_eq!(
                step.report.pre_clone_eligible,
                vec![false, true, true, false]
            );
            assert_eq!(step.report.revivals, 2);
            for row in [0, 3] {
                let choice = &step.report.clone_plan.choices[row];
                assert!(choice.accepted && choice.revival);
                assert_eq!(choice.probability, Some(1.));
                let source = &step.report.clone_plan.sources[choice.donors[0].pool_index as usize];
                assert_eq!(source.frame, 0);
                assert!(source.slot == 1 || source.slot == 2);
            }
            let choice = &step.report.clone_plan.choices[0];
            let source = &step.report.clone_plan.sources[choice.donors[0].pool_index as usize];
            donor_one += usize::from(source.slot == 1);
            let analysis = analyze_cloning_step(archive, 0).unwrap();
            assert!(analysis.checks.iter().all(|c| c.passed));
            for row in 0..4 {
                let probability_check = analysis
                    .checks
                    .iter()
                    .find(|check| check.id == format!("acceptance_probability_row_{row}"))
                    .unwrap();
                assert_eq!(
                    probability_check
                        .source_labels
                        .iter()
                        .any(|label| label == "lem-dead-walker-clone-prob"),
                    row == 0 || row == 3,
                );
            }
            if seed == 0 {
                let mut wrong_flags = archive.clone();
                wrong_flags.steps[0].report.clone_plan.choices[0].accepted = false;
                let failed = analyze_cloning_step(&wrong_flags, 0).unwrap();
                assert!(
                    failed
                        .checks
                        .iter()
                        .any(|check| check.id == "mandatory_revival_row_0" && !check.passed)
                );
            }
            let moments = analysis.moments.unwrap();
            assert_eq!(moments.alive, 2);
            assert_eq!(moments.acceptance_probabilities[0], 1.);
            assert_eq!(moments.acceptance_probabilities[3], 1.);
            assert!((moments.entering_alive_variance - 0.125).abs() < 1e-14);
            expected = Some(moments.accepted_edge_probabilities[0][1]);
            let post_clone = step
                .stages
                .iter()
                .find(|s| s.stage == "post_transform")
                .unwrap();
            assert!(post_clone.validity.iter().all(|v| v.eligible(false)));
            assert!(
                step.final_population
                    .validity
                    .iter()
                    .all(|v| v.eligible(false))
            );
        }
        let probability = expected.unwrap();
        let standard_error = (probability * (1. - probability) / repetitions as f64).sqrt();
        assert!((donor_one as f64 / repetitions as f64 - probability).abs() < 5. * standard_error);
    });
}
