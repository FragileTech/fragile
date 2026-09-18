use algorithmic_gas::{GasBuilder, GasConfig, RunArchive, TensorBatch};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel, RunConfig, lecture_meanfield};
use futures_lite::future::block_on;
async fn archive(seed: u64, steps: usize, nonzero_velocity: bool) -> RunArchive<f64> {
    let config = GasConfig::euclidean(2, 0.04).unwrap();
    let setup = RunConfig {
        walkers: 32,
        dimensions: 2,
        benchmark: Benchmark::Quadratic,
        gas: config.clone(),
        ..Default::default()
    };
    let mut population = setup.initial_population::<f64>().unwrap();
    if nonzero_velocity {
        population.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(
                32,
                2,
                (0..64).map(|j| (j as f64 * 0.73).sin() * 0.8).collect(),
            )
            .unwrap(),
        );
    }
    let model = BenchmarkModel {
        benchmark: Benchmark::Quadratic,
        field: "positions".into(),
        direction: config.fitness.direction,
    };
    let mut gas = GasBuilder::new(population, model.clone())
        .gradient(model)
        .config(GasConfig { seed, ..config })
        .build()
        .await
        .unwrap();
    gas.start_recording(Default::default()).unwrap();
    for _ in 0..steps {
        gas.step().await.unwrap();
    }
    gas.recording().unwrap().clone()
}
#[test]
fn reference_does_not_condition_on_realized_companions_or_accepted_graph() {
    block_on(async {
        let a = archive(11, 1, true).await;
        let b = archive(91, 1, true).await;
        assert_eq!(a.steps[0].before, b.steps[0].before);
        assert_ne!(
            a.steps[0].report.cloning_companions,
            b.steps[0].report.cloning_companions
        );
        let x = lecture_meanfield::rooted_clone_reference(&a, 256, 4491).unwrap();
        let y = lecture_meanfield::rooted_clone_reference(&b, 256, 4491).unwrap();
        assert_eq!(x["status"], "available");
        assert_eq!(x["independent_root_draws"], 256);
        assert_eq!(x["measurement_type_count"], 1024);
        for (u, v) in x["estimates"]
            .as_array()
            .unwrap()
            .iter()
            .zip(y["estimates"].as_array().unwrap())
        {
            assert_eq!(u["reference_mean"], v["reference_mean"]);
            assert_eq!(
                u["reference_mc_standard_error"],
                v["reference_mc_standard_error"]
            );
            assert!(
                u["reference_mc_standard_error"]
                    .as_f64()
                    .unwrap()
                    .is_finite()
            );
        }
        assert!(
            x["estimates"]
                .as_array()
                .unwrap()
                .iter()
                .zip(y["estimates"].as_array().unwrap())
                .any(|(u, v)| u["finite_population_value"] != v["finite_population_value"])
        );
        assert!(lecture_meanfield::rooted_clone_reference(&a, 255, 4491).is_err());
    });
}
#[test]
fn interactive_analysis_adds_single_independent_reference_with_slot_weighted_components() {
    block_on(async {
        let archive = archive(19, 3, false).await;
        let result = lecture_meanfield::analyze("III-04", std::slice::from_ref(&archive)).unwrap();
        let reference = &result.details["rooted_mean_field_reference"];
        assert_eq!(reference["step"], 2);
        assert_eq!(reference["independent_root_draws"], 512);
        assert_eq!(reference["stage"], "post_transform");
        assert_eq!(reference["random_stream"], "MeanFieldReference");
        assert_eq!(
            result
                .plots
                .iter()
                .filter(|p| p.title.starts_with("Rooted population prediction"))
                .count(),
            2
        );
        assert!(
            result
                .plots
                .iter()
                .any(|p| p.title == "Rooted component-size distribution")
        );
        let rows = reference["component_size_distribution"].as_array().unwrap();
        for name in ["reference_probability", "finite_tagged_walker_probability"] {
            let total = rows.iter().map(|r| r[name].as_f64().unwrap()).sum::<f64>();
            assert!((total - 1.).abs() < 1e-12);
        }
        let selected = &archive.steps[1];
        let actual = selected
            .stages
            .iter()
            .find(|s| s.stage == "post_transform")
            .unwrap();
        let measured = actual.fields["velocities"]
            .values
            .chunks(2)
            .map(|v| v[0] * v[0])
            .sum::<f64>()
            / 32.;
        let stress = &reference["estimates"][3];
        assert!((measured - stress["finite_population_value"].as_f64().unwrap()).abs() < 1e-12);
        assert!(stress["reference_mc_standard_error"].as_f64().unwrap() > 0.);
    });
}
