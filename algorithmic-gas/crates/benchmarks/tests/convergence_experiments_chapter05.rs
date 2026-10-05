use algorithmic_gas_benchmarks::{
    convergence_experiments::{ArchiveStore, ExperimentConfig},
    convergence_experiments_chapter05::{
        estimate_direction_strata, exact_quadratic_transition, optimal_qcost_points, run,
        run_cubature,
    },
};

#[test]
fn fixed_directions_do_not_add_noise_uncertainty() {
    let (mean, se) = estimate_direction_strata(&[1., 4., 9., 1., 4., 9., 1.]).unwrap();
    assert!((mean - 29. / 7.).abs() < 1e-15);
    assert_eq!(se, 0.);
    let (_, se) = estimate_direction_strata(&[2., 10., 100., 4., 14., 106.]).unwrap();
    assert!((se * se - 56. / 36.).abs() < 1e-14);
    assert!(estimate_direction_strata(&[1., 4., 9.]).is_err());
    assert!(estimate_direction_strata(&[1., 4., 9., 1., f64::NAN, 9.]).is_err());
}

#[test]
fn continuum_metric_has_a_strict_uniform_lyapunov_certificate() {
    let a = [[0., 1.], [-1., -1.]];
    let p = [[1., 1. / 3.], [1. / 3., 2. / 3.]];
    let determinant = p[0][0] * p[1][1] - p[0][1] * p[1][0];
    assert!(p[0][0] > 0. && determinant > 0.);
    let kappa = 4. / (5. + 5_f64.sqrt());
    let mut positive_gap = [[0.; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let drift = (0..2)
                .map(|k| a[k][i] * p[k][j] + p[i][k] * a[k][j])
                .sum::<f64>();
            let negative_identity = if i == j { -2. / 3. } else { 0. };
            assert!((drift - negative_identity).abs() < 1e-15);
            positive_gap[i][j] = -drift - kappa * p[i][j];
        }
    }
    assert!(positive_gap[0][0] > 0. && positive_gap[1][1] > 0.);
    assert!(
        positive_gap[0][0] * positive_gap[1][1] - positive_gap[0][1] * positive_gap[1][0] >= -1e-15
    );
    let points = [[0.3, -0.2], [-1.1, 0.4], [0.8, 0.7], [0.1, -0.6]];
    let cost = |n: usize| {
        points
            .iter()
            .cycle()
            .take(n)
            .map(|z| z[0] * z[0] + 2. / 3. * z[0] * z[1] + 2. / 3. * z[1] * z[1])
            .sum::<f64>()
            / n as f64
    };
    assert!((cost(4) - cost(64)).abs() < 1e-15);
    // The former b=2, lambda_v=1+epsilon metric is positive, but its
    // unit-quadratic drift has a positive eigenvalue for 0<epsilon<1.
    for epsilon in [0.25, 0.5, 0.75] {
        let old = [[1., 1.], [1., 1. + epsilon]];
        let drift: [[f64; 2]; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..2)
                    .map(|k| a[k][i] * old[k][j] + old[i][k] * a[k][j])
                    .sum()
            })
        });
        assert!(old[0][0] * old[1][1] - old[0][1] * old[1][0] > 0.);
        assert!(drift[0][0] * drift[1][1] - drift[0][1] * drift[1][0] < 0.);
    }
}

#[test]
fn q_transport_is_independent_of_both_storage_permutations() {
    let left = vec![
        vec![-1., 0.2],
        vec![0.1, -0.4],
        vec![1.5, 0.7],
        vec![0.8, -0.1],
    ];
    let right = vec![
        vec![0.9, -0.2],
        vec![-0.8, 0.5],
        vec![1.4, 0.6],
        vec![0.2, -0.6],
    ];
    let expected = optimal_qcost_points(&left, &right, 0.9996).unwrap();
    let permuted_left = vec![
        left[2].clone(),
        left[0].clone(),
        left[3].clone(),
        left[1].clone(),
    ];
    let permuted_right = vec![
        right[1].clone(),
        right[3].clone(),
        right[0].clone(),
        right[2].clone(),
    ];
    assert!(
        (optimal_qcost_points(&permuted_left, &permuted_right, 0.9996).unwrap() - expected).abs()
            < 1e-14
    );
    let existing = algorithmic_gas_benchmarks::convergence_lyapunov::uniform_transport(
        &left,
        &right,
        |a, b| 0.9996 * (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2),
    )
    .unwrap()
    .0;
    assert!((existing - expected).abs() < 1e-14);
    let large_left: Vec<_> = left.iter().cycle().take(16).cloned().collect();
    let large_right: Vec<_> = right.iter().cycle().take(16).cloned().collect();
    assert!(
        (optimal_qcost_points(&large_left, &large_right, 0.9996).unwrap() - expected).abs() < 1e-14
    );
}

#[test]
fn exact_sde_semigroup_and_stationary_covariance() {
    let (a, c) = exact_quadratic_transition(0.2, 1., 1., 1.).unwrap();
    let (b, d) = exact_quadratic_transition(0.4, 1., 1., 1.).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let a2 = (0..2).map(|k| a[i][k] * a[k][j]).sum::<f64>();
            assert!((a2 - b[i][j]).abs() < 1e-13);
            let propagated = (0..2)
                .flat_map(|k| (0..2).map(move |l| a[i][k] * c[k][l] * a[j][l]))
                .sum::<f64>()
                + c[i][j];
            assert!((propagated - d[i][j]).abs() < 1e-13);
        }
    }
    let (_, stationary) = exact_quadratic_transition(100., 1., 1., 1.).unwrap();
    assert!((stationary[0][0] - 0.5).abs() < 1e-12);
    assert!((stationary[1][1] - 0.5).abs() < 1e-12);
    assert!(exact_quadratic_transition(0.2, 0., 1., 1.).is_err());
    assert!(exact_quadratic_transition(0.2, 1., -1., 1.).is_err());
}

#[test]
fn native_cap_rate_refinement_and_reusable_archive() {
    futures_lite::future::block_on(async {
        let path = std::env::temp_dir().join(format!(
            "chapter05-native-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut store = ArchiveStore::new(&path).unwrap();
        let cfg = ExperimentConfig {
            samples: 32,
            steps: 2,
            compact: true,
            archive_chunk_steps: 2,
            ..Default::default()
        };
        let report = run(&cfg, &mut store).await.unwrap();
        assert_eq!(report["summary"]["comparisons_failed"], 0);
        assert!(
            report["capped_cases"][0]["last_mean_ratio"]
                .as_f64()
                .unwrap()
                < 1.
        );
        let levels = report["refinement"]["levels"].as_array().unwrap();
        assert!(
            levels[1]["strong_MSE"].as_f64().unwrap() < levels[0]["strong_MSE"].as_f64().unwrap()
        );
        assert!(
            levels[2]["strong_MSE"].as_f64().unwrap() < levels[1]["strong_MSE"].as_f64().unwrap()
        );
        let ratio = levels[0]["predicted_weak_bias"].as_f64().unwrap()
            / levels[1]["predicted_weak_bias"].as_f64().unwrap();
        assert!(ratio > 3.8 && ratio < 4.2);
        store.finish("completed").unwrap();
        assert_eq!(store.verify(true).unwrap()["checksums_verified"], true);
        std::fs::remove_dir_all(path).unwrap();
    });
}

#[test]
fn exact_cubature_resolves_native_refinement_bias_and_strong_error() {
    futures_lite::future::block_on(async {
        let path = std::env::temp_dir().join(format!(
            "chapter05-cubature-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut store = ArchiveStore::new(&path).unwrap();
        let report = run_cubature(&mut store).await.unwrap();
        assert_eq!(report["summary"]["comparisons_failed"], 0);
        assert_eq!(report["summary"]["sampling_standard_error"], 0.);
        for ratio in report["refinement_ratios"].as_array().unwrap() {
            assert!((ratio["strong_MSE_ratio"].as_f64().unwrap() - 4.).abs() < 3e-5);
            assert!((ratio["weak_bias_ratio"].as_f64().unwrap() - 4.).abs() < 0.03);
        }
        for level in report["levels"].as_array().unwrap() {
            let observed = level["cubature_weak_bias"].as_f64().unwrap();
            let predicted = level["predicted_weak_bias"].as_f64().unwrap();
            assert!((observed - predicted).abs() < 1e-13);
            assert!(observed.abs() > 1e-8);
        }
        store.finish("completed").unwrap();
        assert_eq!(store.verify(true).unwrap()["native_recorded_steps"], 2688);
        std::fs::remove_dir_all(path).unwrap();
    });
}
