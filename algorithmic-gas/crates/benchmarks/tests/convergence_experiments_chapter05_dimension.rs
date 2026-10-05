use algorithmic_gas::{ObservationBatch, Population, TensorBatch};
use algorithmic_gas_benchmarks::{
    convergence_experiments::{ArchiveStore, ExperimentConfig},
    convergence_experiments_chapter05::dimension,
};

#[test]
fn chi_cap_enclosures_contain_independent_references_and_reject_invalid_parameters() {
    // Independently integrated 60-digit radial expectations, not training data.
    for (d, reference) in [
        (1, 0.2394795755),
        (2, 0.1992723331),
        (4, 0.2802143266),
        (8, 0.3733421120),
    ] {
        let certificate = dimension::constants(d, 0.04, 1., 1., 2., 1., 1024).unwrap();
        let eta = &certificate["eta_certificate"];
        assert!(eta["eta_lower"].as_f64().unwrap() < reference);
        assert!(eta["eta_upper"].as_f64().unwrap() > reference);
        assert!(eta["enclosure_width"].as_f64().unwrap() < 0.001);
        assert!(
            certificate["delta_exact_spectrum_lower"].as_f64().unwrap()
                > certificate["same_eta_D_over_T"].as_f64().unwrap()
        );
        assert!(
            certificate["delta_exact_spectrum_lower"].as_f64().unwrap()
                > 3. * certificate["old_delta_D_over_T"].as_f64().unwrap()
        );
    }
    for d in [0, 257] {
        assert!(dimension::cap_eta_enclosure(d, 0.2, 2., 1024).is_err());
    }
    for sigma in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(dimension::cap_eta_enclosure(2, sigma, 2., 1024).is_err());
    }
    assert!(dimension::cap_eta_enclosure(2, 0.2, 2., 1000).is_err());
    assert!(dimension::constants(2, 2., 1., 1., 2., 1., 1024).is_err());
    assert!(dimension::sector_constants(2., 1., 1.).is_err());
    assert!(dimension::sector_constants(0.04, 0., 1.).is_err());
}

#[test]
fn native_stage_matrix_and_sector_lmis_have_strict_certificates() {
    for (omega, reference_delta) in [
        (0.5, 0.0006337469971),
        (1., 0.0015674487258),
        (2., 0.0028934762666),
    ] {
        let certificate = dimension::sector_constants(0.04, 1., omega).unwrap();
        let beta = certificate["beta"].as_f64().unwrap();
        let delta = certificate["delta_pathwise_lower"].as_f64().unwrap();
        assert!(delta >= reference_delta);
        let h = 0.04_f64;
        let a = (-h).exp();
        // Reconstruct the native B-A-O-A-B stage by independent impulse propagation.
        let stage = |x: f64, v: f64| {
            let v = v - h / 2. * omega * x;
            let x = x + h / 2. * v;
            let v = a * v;
            let x = x + h / 2. * v;
            let v = v - h / 2. * omega * x;
            [x, v]
        };
        let first = stage(1. / omega.sqrt(), 0.);
        let second = stage(0., 1.);
        let map = [
            [omega.sqrt() * first[0], omega.sqrt() * second[0]],
            [first[1], second[1]],
        ];
        for i in 0..2 {
            for j in 0..2 {
                let interval = &certificate["scaled_native_matrix_interval"][i][j];
                assert!(interval[0].as_f64().unwrap() <= map[i][j] + 4. * f64::EPSILON);
                assert!(interval[1].as_f64().unwrap() >= map[i][j] - 4. * f64::EPSILON);
            }
        }
        // Independent direct SPD check of both endpoint deficits minus delta*G.
        for endpoint in [0., 1.] {
            let m = [
                [map[0][0], map[0][1]],
                [endpoint * map[1][0], endpoint * map[1][1]],
            ];
            let g = [[1., beta], [beta, 1.]];
            let deficit: [[f64; 2]; 2] = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    (1. - delta) * g[i][j]
                        - (0..2)
                            .flat_map(|k| (0..2).map(move |l| m[k][i] * g[k][l] * m[l][j]))
                            .sum::<f64>()
                })
            });
            assert!(deficit[0][0] >= -1e-14 && deficit[1][1] >= -1e-14);
            assert!(deficit[0][0] * deficit[1][1] - deficit[0][1] * deficit[1][0] >= -1e-14);
        }
        assert!(
            certificate["metric_equivalence_to_Q_diagonal"]["lower"]
                .as_f64()
                .unwrap()
                > 0.
        );
    }
}

fn population(rows: &[[f64; 2]]) -> Population<f64> {
    let mut obs = ObservationBatch::positions(
        TensorBatch::vectors(rows.len(), 1, rows.iter().map(|r| r[0]).collect()).unwrap(),
    );
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(rows.len(), 1, rows.iter().map(|r| r[1]).collect()).unwrap(),
    );
    Population::new(obs).unwrap()
}

#[test]
fn sector_cost_is_permutation_and_population_duplication_invariant() {
    let left = [[-1., 0.2], [0.4, -0.1], [0.8, 0.5], [1.1, -0.8]];
    let right = [[0.3, -0.2], [1.3, -0.7], [-0.9, 0.1], [0.9, 0.6]];
    let cost = dimension::optimal_sector_cost(&population(&left), &population(&right), 2., 0.0555)
        .unwrap();
    let permuted = [left[2], left[0], left[3], left[1]];
    let permuted_right = [right[3], right[1], right[0], right[2]];
    assert!(
        (dimension::optimal_sector_cost(
            &population(&permuted),
            &population(&permuted_right),
            2.,
            0.0555
        )
        .unwrap()
            - cost)
            .abs()
            < 1e-14
    );
    let l: Vec<_> = left.into_iter().cycle().take(16).collect();
    let r: Vec<_> = right.into_iter().cycle().take(16).collect();
    assert!(
        (dimension::optimal_sector_cost(&population(&l), &population(&r), 2., 0.0555).unwrap()
            - cost)
            .abs()
            < 1e-14
    );
}

#[test]
fn native_dimension_probes_preserve_noise_and_force_records() {
    futures_lite::future::block_on(async {
        let path = std::env::temp_dir().join(format!(
            "chapter05-dimension-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut store = ArchiveStore::new(&path).unwrap();
        let config = ExperimentConfig {
            samples: 12,
            steps: 2,
            compact: true,
            archive_chunk_steps: 1,
            ..Default::default()
        };
        let report = dimension::run(&config, &mut store).await.unwrap();
        assert_eq!(report["summary"]["comparisons_failed"], 0);
        for case in report["cases"].as_array().unwrap() {
            assert!(case["final_optimal_Q_ratio"].as_f64().unwrap() < 1.);
            assert!(case["maximum_matrix_residual"].as_f64().unwrap() < 1e-12);
            assert!(case["maximum_force_residual"].as_f64().unwrap() < 1e-12);
        }
        store.finish("completed").unwrap();
        assert_eq!(store.verify(true).unwrap()["native_recorded_steps"], 96);
        std::fs::remove_dir_all(path).unwrap();
    });
}

#[test]
fn high_well_curvature_uses_a_verified_adapted_metric_and_keeps_nonlinear_scope() {
    let omega = 2. + 40. * std::f64::consts::PI.powi(2);
    let certificate = dimension::sector_curvature_profile(0.04, 1., omega).unwrap();
    let alpha = certificate["alpha"].as_f64().unwrap();
    let beta = certificate["beta"].as_f64().unwrap();
    let delta = certificate["delta_pathwise_lower"].as_f64().unwrap();
    assert!((alpha - (1. - omega * 0.02_f64.powi(2))).abs() < 1e-14);
    assert!(alpha > beta * beta && delta > 0.038 && delta < 0.04);
    for endpoint in certificate["endpoint_LMI_certificate"].as_array().unwrap() {
        assert!(
            endpoint["generalized_minimum_eigenvalue"][0]
                .as_f64()
                .unwrap()
                >= delta
        );
    }
    assert!(
        certificate["scope"]
            .as_str()
            .unwrap()
            .contains("force remainder")
    );
    assert!(dimension::sector_curvature_profile(0.04, 1., 3000.).is_err());
    assert!(dimension::sector_curvature_profile(0.04, 0., omega).is_err());
}
