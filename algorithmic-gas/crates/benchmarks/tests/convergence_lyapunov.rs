use algorithmic_gas_benchmarks::{
    convergence_kinetic::{KineticCheckStatus, KineticInput, KineticValidationConfig},
    convergence_lyapunov::{
        LyapunovSwarm, LyapunovValidationConfig, LyapunovValidationReport,
        boundary_probability_separation, default_lyapunov_cases, logarithmic_box_barrier_values,
        lyapunov_validation_cases, uniform_transport, validate_lyapunov,
    },
};

fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}

fn reorder(swarm: &LyapunovSwarm, permutation: &[usize]) -> LyapunovSwarm {
    LyapunovSwarm {
        positions: permutation
            .iter()
            .map(|&i| swarm.positions[i].clone())
            .collect(),
        velocities: permutation
            .iter()
            .map(|&i| swarm.velocities[i].clone())
            .collect(),
        alive: permutation.iter().map(|&i| swarm.alive[i]).collect(),
        barrier_values: swarm
            .barrier_values
            .as_ref()
            .map(|values| permutation.iter().map(|&i| values[i]).collect()),
    }
}

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-11 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

fn assert_intrinsic_observables_equal(a: &LyapunovValidationReport, b: &LyapunovValidationReport) {
    for (a, b) in [
        (
            a.physical_wasserstein_squared,
            b.physical_wasserstein_squared,
        ),
        (
            a.centered_euclidean_wasserstein_squared,
            b.centered_euclidean_wasserstein_squared,
        ),
        (a.location_error, b.location_error),
        (a.structural_error, b.structural_error),
        (a.candidate_coupling_cost, b.candidate_coupling_cost),
        (a.intra_swarm_variance, b.intra_swarm_variance),
        (
            a.augmented_lyapunov_without_boundary,
            b.augmented_lyapunov_without_boundary,
        ),
        (
            a.projected_positional_displacement_sum,
            b.projected_positional_displacement_sum,
        ),
        (
            a.physical_phase_displacement_sum,
            b.physical_phase_displacement_sum,
        ),
        (a.dispersion_squared, b.dispersion_squared),
    ] {
        close(a, b);
    }
    assert_eq!(a.status_difference_count, b.status_difference_count);
    for (a, b) in [
        (a.boundary_potential, b.boundary_potential),
        (a.augmented_lyapunov, b.augmented_lyapunov),
    ] {
        match (a, b) {
            (Some(a), Some(b)) => close(a, b),
            (None, None) => {}
            _ => panic!("barrier availability changed on reordering"),
        }
    }
    for (a, b) in a
        .position_barycenters
        .iter()
        .chain(&a.velocity_barycenters)
        .flatten()
        .zip(
            b.position_barycenters
                .iter()
                .chain(&b.velocity_barycenters)
                .flatten(),
        )
    {
        close(*a, *b);
    }
    for (a, b) in a.variance_conversions.iter().zip(&b.variance_conversions) {
        assert_eq!(a.alive, b.alive);
        for (a, b) in [
            (a.sum_squares_position, b.sum_squares_position),
            (a.sum_squares_velocity, b.sum_squares_velocity),
            (a.physical_position_variance, b.physical_position_variance),
            (a.physical_velocity_variance, b.physical_velocity_variance),
            (a.pairwise_sasaki_variance, b.pairwise_sasaki_variance),
        ] {
            close(a, b);
        }
    }
}

fn assert_marked_plan_reindexes(
    old: &LyapunovValidationReport,
    new: &LyapunovValidationReport,
    left: &[usize],
    right: &[usize],
) {
    for (i, &old_i) in left.iter().enumerate() {
        for (j, &old_j) in right.iter().enumerate() {
            close(
                new.dispersion_transport_plan[i][j],
                old.dispersion_transport_plan[old_i][old_j],
            );
        }
    }
}

fn assert_companion_laws_reindex(
    old: &LyapunovValidationReport,
    new: &LyapunovValidationReport,
    permutation: &[usize],
) {
    for row in &new.companion_laws {
        let old_query = permutation[row.query_slot];
        let before = old
            .companion_laws
            .iter()
            .find(|r| r.query_slot == old_query)
            .unwrap();
        assert_eq!(row.candidate_count, before.candidate_count);
        close(row.log_normalizer, before.log_normalizer);
        for (&new_donor, &probability) in row.donor_slots.iter().zip(&row.probabilities) {
            let old_donor = permutation[new_donor];
            let before_index = before
                .donor_slots
                .iter()
                .position(|&i| i == old_donor)
                .unwrap();
            close(probability, before.probabilities[before_index]);
        }
    }
    // Native random addresses use storage rows. Equivariance concerns these
    // probability laws, not equality of sampled trajectories at a fixed seed.
}

#[test]
fn discrete_transport_matches_permutation_enumeration_and_handles_unequal_weights() {
    let left = vec![vec![0., 1.], vec![1., -2.], vec![2., 0.4]];
    let right = vec![vec![1.8, 0.6], vec![-0.3, 1.4], vec![0.7, -1.6]];
    let (actual, plan) = uniform_transport(&left, &right, squared_distance).unwrap();
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let exact = permutations
        .iter()
        .map(|p| {
            (0..3)
                .map(|i| squared_distance(&left[i], &right[p[i]]))
                .sum::<f64>()
                / 3.
        })
        .fold(f64::INFINITY, f64::min);
    assert!((actual - exact).abs() < 1e-14);
    for row in &plan {
        assert!((row.iter().sum::<f64>() - 1. / 3.).abs() < 1e-14);
    }
    for j in 0..3 {
        assert!((plan.iter().map(|row| row[j]).sum::<f64>() - 1. / 3.).abs() < 1e-14);
    }
    let (unequal, plan) = uniform_transport(
        &[vec![0.]],
        &[vec![-1.], vec![1.], vec![3.]],
        squared_distance,
    )
    .unwrap();
    assert!((unequal - 11. / 3.).abs() < 1e-14);
    assert_eq!(plan, vec![vec![1. / 3.; 3]]);

    // For 2-versus-3 empirical measures, each left atom supplies 3 units
    // and each right atom receives 2. Enumerating every feasible first row
    // gives an independent optimal-cost oracle, including split couplings.
    for (left, right) in [
        (
            vec![vec![0., 0.], vec![1., 0.]],
            vec![vec![0.4, 0.], vec![0.3, 0.], vec![2., 0.]],
        ),
        (
            vec![vec![-2., 0.5], vec![3., -1.]],
            vec![vec![1., 2.], vec![-0.4, -2.], vec![0.5, 1.]],
        ),
    ] {
        let mut oracle = f64::INFINITY;
        for a in 0..=2 {
            for b in 0..=2 {
                for c in 0..=2 {
                    if a + b + c != 3 {
                        continue;
                    }
                    let units = [a, b, c];
                    let objective = (0..3)
                        .map(|j| {
                            units[j] as f64 * squared_distance(&left[0], &right[j])
                                + (2 - units[j]) as f64 * squared_distance(&left[1], &right[j])
                        })
                        .sum::<f64>()
                        / 6.;
                    oracle = oracle.min(objective);
                }
            }
        }
        let (forward, _) = uniform_transport(&left, &right, squared_distance).unwrap();
        let (reverse, _) = uniform_transport(&right, &left, squared_distance).unwrap();
        assert!((forward - oracle).abs() < 1e-13);
        assert!((reverse - oracle).abs() < 1e-13);
    }
}

#[test]
fn permutation_regression_preserves_zero_transport_cost_with_positive_candidate_cost() {
    futures_lite::future::block_on(async {
        let case = default_lyapunov_cases(128, 530_701).remove(0);
        let report = validate_lyapunov(&case.config, &case.left, &case.right)
            .await
            .unwrap();
        assert!(report.physical_wasserstein_squared.abs() < 1e-14);
        assert!(report.structural_error.abs() < 1e-14);
        assert!(report.centered_euclidean_wasserstein_squared.abs() < 1e-14);
        assert!(report.candidate_coupling_cost > 0.1);
        assert!(report.dispersion_squared.abs() < 1e-14);
        assert!(report.projected_positional_displacement_sum.abs() < 1e-14);
        assert!(report.physical_phase_displacement_sum.abs() < 1e-14);
        assert_eq!(report.status_difference_count, 0);
        assert!(report.intra_swarm_variance > 0.);
        assert!(report.augmented_lyapunov.unwrap() > 0.);
        assert!(report.checks.iter().all(|c| c.passed));
    });
}

#[test]
fn independent_and_simultaneous_reorderings_preserve_intrinsic_observables_and_companion_laws() {
    futures_lite::future::block_on(async {
        let left_permutation = [2, 7, 0, 5, 3, 1, 6, 4];
        let independent_right_permutation = [6, 1, 7, 0, 4, 2, 5, 3];
        for case in default_lyapunov_cases(128, 530_711) {
            let baseline = validate_lyapunov(&case.config, &case.left, &case.right)
                .await
                .unwrap();
            for right_permutation in [&left_permutation, &independent_right_permutation] {
                let left = reorder(&case.left, &left_permutation);
                let right = reorder(&case.right, right_permutation);
                let permuted = validate_lyapunov(&case.config, &left, &right)
                    .await
                    .unwrap();
                assert_intrinsic_observables_equal(&baseline, &permuted);
                assert_marked_plan_reindexes(
                    &baseline,
                    &permuted,
                    &left_permutation,
                    right_permutation,
                );
                assert_companion_laws_reindex(&baseline, &permuted, &left_permutation);
                assert!(permuted.checks.iter().all(|c| c.passed));
            }
        }
    });
}

#[test]
fn tied_marked_costs_use_intrinsic_atoms_to_select_permutation_invariant_components() {
    futures_lite::future::block_on(async {
        let left = LyapunovSwarm {
            positions: vec![vec![-1.], vec![1.]],
            velocities: vec![vec![0.], vec![0.]],
            alive: vec![true, false],
            barrier_values: None,
        };
        let mut right = left.clone();
        right.alive.reverse();
        let config = LyapunovValidationConfig {
            status_weight: 16. / 9.,
            companion_replicates: 64,
            ..Default::default()
        };
        let baseline = validate_lyapunov(&config, &left, &right).await.unwrap();
        // Keeping positions paired pays only mark mismatch; swapping the two
        // atoms pays only positional cost. Both total costs are 16/9.
        close(baseline.dispersion_squared, 16. / 9.);
        for (left_order, right_order) in [([0, 1], [1, 0]), ([1, 0], [0, 1]), ([1, 0], [1, 0])] {
            let after = validate_lyapunov(
                &config,
                &reorder(&left, &left_order),
                &reorder(&right, &right_order),
            )
            .await
            .unwrap();
            assert_intrinsic_observables_equal(&baseline, &after);
            assert_marked_plan_reindexes(&baseline, &after, &left_order, &right_order);
            assert_companion_laws_reindex(&baseline, &after, &left_order);
        }
    });
}

#[test]
fn supplied_candidate_plan_is_an_admissible_alive_coupling_and_reindexes_with_its_supports() {
    futures_lite::future::block_on(async {
        let case = default_lyapunov_cases(64, 530_719).remove(0);
        let mut config = case.config.clone();
        config.candidate_transport_plan = Some(
            (0..8)
                .map(|i| (0..8).map(|j| if i == j { 1. / 8. } else { 0. }).collect())
                .collect(),
        );
        let original = validate_lyapunov(&config, &case.left, &case.right)
            .await
            .unwrap();
        assert!(original.candidate_coupling_cost > 0.1);
        let left_order = [2, 7, 0, 5, 3, 1, 6, 4];
        let right_order = [6, 1, 7, 0, 4, 2, 5, 3];
        let old_plan = config.candidate_transport_plan.clone().unwrap();
        config.candidate_transport_plan = Some(
            left_order
                .iter()
                .map(|&i| right_order.iter().map(|&j| old_plan[i][j]).collect())
                .collect(),
        );
        let permuted = validate_lyapunov(
            &config,
            &reorder(&case.left, &left_order),
            &reorder(&case.right, &right_order),
        )
        .await
        .unwrap();
        assert_intrinsic_observables_equal(&original, &permuted);
        config.candidate_transport_plan.as_mut().unwrap()[0][0] = -0.1;
        assert!(
            validate_lyapunov(&config, &case.left, &case.right)
                .await
                .is_err()
        );
        config.candidate_transport_plan = Some(vec![vec![0.; 8]; 8]);
        assert!(
            validate_lyapunov(&config, &case.left, &case.right)
                .await
                .is_err()
        );
        config.candidate_transport_plan = Some(vec![vec![1.; 1]; 8]);
        assert!(
            validate_lyapunov(&config, &case.left, &case.right)
                .await
                .is_err()
        );
    });
}

#[test]
fn native_metric_companion_and_variance_diagnostics_obey_their_scopes() {
    futures_lite::future::block_on(async {
        let reports = lyapunov_validation_cases(512, 530_703).await.unwrap();
        assert_eq!(reports.len(), 3);
        for report in reports {
            assert!(
                report.checks.iter().all(|c| c.passed),
                "{:?}",
                report
                    .checks
                    .iter()
                    .filter(|c| !c.passed)
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.source_mapped_constants.len(), 6);
            assert_eq!(report.boundary_probability_diagnostics.len(), 1);
            for probe in &report.boundary_probability_diagnostics {
                assert!(probe.checks.iter().all(|c| c.passed));
                assert!(probe.gaussian_total_variation > 0.);
                assert!(probe.compact_inverse_projection_lipschitz >= 1.);
            }
            assert!(report.coercivity_min_eigenvalue > 0.);
            assert!(report.coercivity_max_eigenvalue >= report.coercivity_min_eigenvalue);
            let alive_left = report.variance_conversions[0].alive;
            let alive_right = report.variance_conversions[1].alive;
            for row in &report.transport_plan {
                assert!((row.iter().sum::<f64>() - 1. / alive_left as f64).abs() < 1e-14);
            }
            for j in 0..alive_right {
                assert!(
                    (report.transport_plan.iter().map(|row| row[j]).sum::<f64>()
                        - 1. / alive_right as f64)
                        .abs()
                        < 1e-14
                );
            }
            for row in &report.companion_laws {
                assert_eq!(row.status, KineticCheckStatus::NotRejected);
                assert!((row.probabilities.iter().sum::<f64>() - 1.).abs() < 1e-14);
                assert!(!row.donor_slots.contains(&row.query_slot));
                assert!(
                    row.frequency_status
                        .iter()
                        .all(|s| *s == KineticCheckStatus::NotRejected)
                );
                assert!(row.minimum_probability_lower_bound.unwrap() > 0.);
                assert!(row.log_normalizer >= row.log_normalizer_lower_bound - 1e-14);
                assert!(row.log_normalizer <= row.log_normalizer_upper_bound + 1e-14);
            }
            let json = serde_json::to_value(&report).unwrap();
            let restored:algorithmic_gas_benchmarks::convergence_lyapunov::LyapunovValidationReport=
                serde_json::from_value(json.clone()).unwrap();
            assert_eq!(json, serde_json::to_value(restored).unwrap());
            let chapters = [
                include_str!(
                    "../../../docs/source/2_fractal_gas/convergence_program/01_fragile_gas_framework.md"
                ),
                include_str!(
                    "../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"
                ),
                include_str!(
                    "../../../docs/source/2_fractal_gas/convergence_program/03_cloning.md"
                ),
            ];
            for label in report.checks.iter().flat_map(|c| c.source_labels.iter()) {
                assert!(
                    chapters
                        .iter()
                        .any(|chapter| chapter.contains(&format!(":label: {label}"))),
                    "{label}"
                );
            }
        }
    });
}

#[test]
fn dead_coordinates_do_not_enter_alive_statistics_and_singletons_have_fallback_donors() {
    futures_lite::future::block_on(async {
        let left = LyapunovSwarm {
            positions: vec![vec![0.2], vec![1e6]],
            velocities: vec![vec![0.1], vec![-1e5]],
            alive: vec![true, false],
            barrier_values: None,
        };
        let mut right = left.clone();
        right.positions[0][0] += 0.3;
        let report = validate_lyapunov(
            &LyapunovValidationConfig {
                companion_replicates: 64,
                ..Default::default()
            },
            &left,
            &right,
        )
        .await
        .unwrap();
        assert_eq!(report.variance_conversions[0].alive, 1);
        assert_eq!(report.variance_conversions[0].sum_squares_position, 0.);
        assert_eq!(report.structural_error, 0.);
        assert_eq!(report.boundary_potential, None);
        assert_eq!(report.augmented_lyapunov, None);
        assert_eq!(report.companion_laws[0].donor_slots, vec![0]);
        assert_eq!(report.companion_laws[0].probabilities, vec![1.]);
        assert_eq!(report.companion_laws[0].measured_probabilities, vec![1.]);
        assert!(report.checks.iter().all(|c| c.passed));
    });
}

#[test]
fn boundary_probability_conversion_is_local_and_rejects_degenerate_noise() {
    let config = KineticValidationConfig::default();
    let left = KineticInput {
        positions: vec![1.9, 0.],
        velocities: vec![0.2, 0.],
    };
    let right = KineticInput {
        positions: vec![1.91, 0.],
        velocities: vec![0.21, 0.],
    };
    let report = boundary_probability_separation(&config, &left, &right).unwrap();
    assert!(report.checks.iter().all(|c| c.passed));
    assert!(report.compact_inverse_projection_lipschitz > 1.);
    assert!(report.projected_distance < report.physical_distance);
    assert!(report.gaussian_total_variation > 0.);
    let same = boundary_probability_separation(&config, &left, &left).unwrap();
    assert!(same.gaussian_total_variation < 1.5e-7);
    assert!(
        boundary_probability_separation(
            &KineticValidationConfig {
                velocity_diffusion: 0.,
                position_diffusion: 0.,
                ..config
            },
            &left,
            &right
        )
        .is_err()
    );
}

#[test]
fn logarithmic_barrier_is_finite_only_on_alive_interior_rows() {
    let middle = logarithmic_box_barrier_values(&[vec![0.]], &[true], 2.).unwrap()[0];
    let near = logarithmic_box_barrier_values(&[vec![1.9999]], &[true], 2.).unwrap()[0];
    assert_eq!(middle, 1.);
    assert!(near > 8.);
    assert!(logarithmic_box_barrier_values(&[vec![2.]], &[true], 2.).is_err());
    assert_eq!(
        logarithmic_box_barrier_values(&[vec![1e6]], &[false], 2.).unwrap(),
        vec![0.]
    );
}

#[test]
fn noncoercive_parameters_are_rejected() {
    assert!(
        LyapunovValidationConfig {
            companion_replicates: 1_000_000,
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
    assert!(
        LyapunovValidationConfig {
            companion_replicates: 1_000_001,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LyapunovValidationConfig {
            cross_coefficient: 2.,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LyapunovValidationConfig {
            lambda_v: 0.,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        LyapunovValidationConfig {
            companion_width: 0.,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}
