use algorithmic_gas::{
    GasConfig, RecordingConfig, Result, TensorBatch,
    boundary::BoundaryPolicy,
    geometry::AlgorithmicDistance,
    mean_field::step_diagnostics,
    noise::{FactorValues, NoiseGeometry},
};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig};

#[test]
fn bounded_comparison_does_not_cap_physical_positions() {
    let c = GasConfig::euclidean(2, 0.04).unwrap();
    let mut p = RunConfig {
        dimensions: 2,
        walkers: 2,
        gas: c.clone(),
        ..Default::default()
    }
    .initial_population::<f64>()
    .unwrap();
    p.observations.fields.insert(
        "positions".into(),
        TensorBatch::vectors(2, 2, vec![1e12, 0., -1e12, 0.]).unwrap(),
    );
    let dist = c
        .distance_donors
        .distance
        .compare(&p.observations, 0, &p.observations, 1)
        .unwrap();
    assert!(dist <= 4. && dist > 3.99);
    assert_eq!(p.observations.field("positions").unwrap().values()[0], 1e12);
}

#[test]
fn canonical_stage_order_and_cap_match_executed_baoab() {
    futures_lite::future::block_on(async {
        let c = RunConfig {
            benchmark: Benchmark::Quadratic,
            walkers: 16,
            dimensions: 2,
            gas: GasConfig::euclidean(2, 0.04).unwrap(),
            ..Default::default()
        };
        let mut gas = c.build::<f64>().await.unwrap();
        let mut population = gas.population().clone();
        population.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(16, 2, vec![0.9; 32]).unwrap(),
        );
        for i in 0..4 {
            population.validity[i].terminated = true;
        }
        gas.replace_population(population).await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let a = gas.recording().unwrap();
        let s = &a.steps[0];
        assert_eq!(s.report.revivals, 4);
        let names: Vec<_> = s.stages.iter().map(|s| s.stage.as_str()).collect();
        for (before, after) in [
            ("B2", "position_diffusion"),
            ("position_diffusion", "velocity_cap"),
            ("velocity_cap", "terminal"),
        ] {
            assert!(
                names.iter().position(|&n| n == before).unwrap()
                    < names.iter().position(|&n| n == after).unwrap()
            );
        }
        let source = s
            .stages
            .iter()
            .find(|s| s.stage == "position_diffusion")
            .unwrap();
        let capped = s.stages.iter().find(|s| s.stage == "velocity_cap").unwrap();
        for (v, w) in source.fields["velocities"]
            .values
            .chunks(2)
            .zip(capped.fields["velocities"].values.chunks(2))
        {
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            for j in 0..2 {
                assert!((w[j] - 2. * v[j] / (2. + norm)).abs() < 1e-12);
            }
        }
        let diag = step_diagnostics(s).unwrap();
        assert_eq!(diag.component_sizes.iter().sum::<usize>(), 16);
        assert!(diag.component_sizes.iter().any(|&s| s > 1));
    });
}

#[test]
fn terminal_boundary_allows_a_full_kinetic_update_after_clone_jitter() {
    futures_lite::future::block_on(async {
        let mut c = RunConfig {
            benchmark: Benchmark::Quadratic,
            walkers: 16,
            dimensions: 2,
            gas: GasConfig::euclidean(2, 0.04).unwrap(),
            ..Default::default()
        };
        c.gas.clone_transform.jitter_amplitude = 100.;
        let mut gas = c.build::<f64>().await.unwrap();
        let mut p = gas.population().clone();
        for i in 1..16 {
            p.validity[i].terminated = true;
        }
        gas.replace_population(p).await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let s = &gas.recording().unwrap().steps[0];
        let post = s.stages.iter().find(|s| s.stage == "post_clone").unwrap();
        assert!(post.validity.iter().all(|v| v.eligible(false)));
        assert!(post.fields["positions"].values.iter().any(|x| x.abs() > 2.));
        assert!(s.stages.iter().any(|s| s.stage == "B2"));
        assert!(s.report.eligible < 16);
    });
}

#[test]
fn final_gaussian_position_prediction_is_calibrated_on_actual_runs() {
    futures_lite::future::block_on(async {
        let mut sum = 0.;
        let mut predicted_variance = 0.;
        for seed in 0..32 {
            let mut c = RunConfig {
                benchmark: Benchmark::Quadratic,
                walkers: 32,
                dimensions: 2,
                gas: GasConfig::euclidean(2, 0.04).unwrap(),
                ..Default::default()
            };
            c.gas.seed = seed;
            c.gas.boundary = BoundaryPolicy::Unbounded;
            c.gas.kinetic.position_diffusion = 0.7;
            c.gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
                scale: FactorValues::Constant { values: vec![1.] },
            };
            let mut gas = c.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig::default()).unwrap();
            for _ in 0..4 {
                gas.step().await.unwrap();
            }
            for s in &gas.recording().unwrap().steps {
                let prior = s.stages.iter().find(|s| s.stage == "B2").unwrap();
                let post = s
                    .stages
                    .iter()
                    .find(|s| s.stage == "position_diffusion")
                    .unwrap();
                let var = 0.7f64.powi(2) * 0.04;
                for (x, y) in prior.fields["positions"]
                    .values
                    .chunks(2)
                    .zip(post.fields["positions"].values.chunks(2))
                {
                    let expected = (-var / 2.).exp() * x[0].cos();
                    sum += y[0].cos() - expected;
                    predicted_variance +=
                        0.5 * (1. + (-2. * var).exp() * (2. * x[0]).cos()) - expected * expected;
                }
            }
        }
        assert!(
            sum.abs() < 4.5 * predicted_variance.sqrt(),
            "Gaussian position residual {} SE",
            sum / predicted_variance.sqrt()
        );
    });
}

#[test]
fn independent_run_size_sweep_keeps_the_same_timestep() -> Result<()> {
    let p = serde_json::json!({"walkers":64,"replicas":4,"h":0.04});
    let cs = algorithmic_gas_benchmarks::lecture_meanfield::configs("III-03", &p, 7)?;
    assert_eq!(cs.len(), 12);
    assert_eq!(
        cs.iter()
            .map(|c| c.walkers)
            .collect::<std::collections::BTreeSet<_>>(),
        [16, 32, 64].into_iter().collect()
    );
    Ok(())
}

#[test]
fn canonical_path_likelihood_covers_weighted_revival_and_shared_rotation() {
    futures_lite::future::block_on(async {
        let c = RunConfig {
            benchmark: Benchmark::Quadratic,
            walkers: 12,
            dimensions: 2,
            gas: GasConfig::euclidean(2, 0.04).unwrap(),
            ..Default::default()
        };
        let mut gas = c.build::<f64>().await.unwrap();
        let mut p = gas.population().clone();
        for i in 0..4 {
            p.validity[i].terminated = true;
        }
        gas.replace_population(p).await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let a = gas.stop_recording().unwrap();
        let likelihood = algorithmic_gas::physics::path_action::likelihood(&a).unwrap();
        assert!(likelihood.complete, "{likelihood:#?}");
        assert!(
            likelihood.steps[0]
                .components
                .iter()
                .any(|c| c.carrier == "normalized_component_Haar")
        );
    });
}

#[test]
fn executed_baoab_joint_covariance_has_the_derived_resonance() {
    futures_lite::future::block_on(async {
        let c = RunConfig {
            benchmark: Benchmark::Quadratic,
            walkers: 1,
            dimensions: 1,
            gas: GasConfig::euclidean(1, 2.).unwrap(),
            ..Default::default()
        };
        let mut gas = c.build::<f64>().await.unwrap();
        let mut p = gas.population().clone();
        p.observations.fields.insert(
            "positions".into(),
            TensorBatch::vectors(1, 1, vec![0.2]).unwrap(),
        );
        p.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(1, 1, vec![0.3]).unwrap(),
        );
        gas.replace_population(p).await.unwrap();
        for n in 1..=8 {
            let report = gas.step().await.unwrap();
            let expected = if n % 2 == 0 { 1. } else { -1. } * 2. * 0.3 / (2. + n as f64 * 0.3);
            let observed = gas
                .population()
                .observations
                .field("velocities")
                .unwrap()
                .values()[0];
            assert!((observed - expected).abs() < 1e-12);
            if report.eligible == 0 {
                break;
            }
        }

        for h in [0.04, 2.] {
            let mut samples = Vec::new();
            for seed in 0..16 {
                let mut c = RunConfig {
                    benchmark: Benchmark::Quadratic,
                    walkers: 128,
                    dimensions: 1,
                    gas: GasConfig::euclidean(1, h).unwrap(),
                    ..Default::default()
                };
                c.gas.seed = seed + 90000;
                let mut gas = c.build::<f64>().await.unwrap();
                let mut p = gas.population().clone();
                p.observations.fields.insert(
                    "positions".into(),
                    TensorBatch::vectors(128, 1, vec![0.2; 128]).unwrap(),
                );
                p.observations.fields.insert(
                    "velocities".into(),
                    TensorBatch::vectors(128, 1, vec![0.3; 128]).unwrap(),
                );
                gas.replace_population(p).await.unwrap();
                gas.start_recording(Default::default()).unwrap();
                gas.step().await.unwrap();
                let s = &gas.recording().unwrap().steps[0];
                assert_eq!(s.report.clones, 0);
                let stage = s
                    .stages
                    .iter()
                    .find(|s| s.stage == "position_diffusion")
                    .unwrap();
                samples.extend(
                    stage.fields["positions"]
                        .values
                        .iter()
                        .copied()
                        .zip(stage.fields["velocities"].values.iter().copied()),
                );
            }
            let q2 = (1. - (-2. * h).exp()) / 2.;
            let s2 = 0.01 * h;
            let b = 1. - h * h / 4.;
            let expected = [h * h * q2 / 4. + s2, h * q2 * b / 2., q2 * b * b];
            let count = samples.len() as f64;
            let mx = samples.iter().map(|s| s.0).sum::<f64>() / count;
            let mv = samples.iter().map(|s| s.1).sum::<f64>() / count;
            let actual = [
                samples.iter().map(|s| (s.0 - mx).powi(2)).sum::<f64>() / (count - 1.),
                samples.iter().map(|s| (s.0 - mx) * (s.1 - mv)).sum::<f64>() / (count - 1.),
                samples.iter().map(|s| (s.1 - mv).powi(2)).sum::<f64>() / (count - 1.),
            ];
            let se = [
                (2. * expected[0].powi(2) / (count - 1.)).sqrt(),
                ((expected[0] * expected[2] + expected[1].powi(2)) / (count - 1.)).sqrt(),
                (2. * expected[2].powi(2) / (count - 1.)).sqrt(),
            ];
            for i in 0..3 {
                assert!(
                    (actual[i] - expected[i]).abs() < 5. * se[i] + 1e-12,
                    "h={h}, covariance {i}: {:?} vs {:?}",
                    actual,
                    expected
                );
            }
            if h == 2. {
                assert!(actual[2] < 1e-26);
            } else {
                assert!(expected[0] * expected[2] - expected[1] * expected[1] > 0.);
            }
        }
    });
}
