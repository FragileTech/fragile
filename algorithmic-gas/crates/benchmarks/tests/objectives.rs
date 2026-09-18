//! Parity classics: scalar values against the Optimization Lab goldens, tensor graphs
//! against scalar values, analytic gradients against finite differences, and serde stability.
use algorithmic_gas::{
    domain::{GradientProvider, RewardSource},
    fitness::ObjectiveDirection,
    *,
};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel, RunConfig, mixture::Mixture};
use futures_lite::future::block_on;
use serde_json::{Value, json};

fn goldens() -> Value {
    let path = format!(
        "{}/tests/fixtures/objective-goldens.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// The Optimization Lab's flat config → the Rust benchmark (`quadratic` is `quadratic_well`).
fn from_lab_config(config: &Value) -> (Benchmark, usize) {
    let id = config["benchmark"].as_str().unwrap();
    let mut spec = serde_json::Map::new();
    spec.insert(
        "id".into(),
        json!(if id == "quadratic" {
            "quadratic_well"
        } else {
            id
        }),
    );
    for (key, value) in config.as_object().unwrap() {
        if key != "benchmark" && key != "dimensions" {
            spec.insert(key.clone(), value.clone());
        }
    }
    let benchmark: Benchmark = serde_json::from_value(Value::Object(spec)).unwrap();
    let d = benchmark
        .fixed_dimension()
        .unwrap_or(config["dimensions"].as_u64().unwrap() as usize);
    (benchmark, d)
}

#[test]
fn classic_values_match_the_optimization_lab() {
    let goldens = goldens();
    let cases = goldens["classics"].as_array().unwrap();
    assert!(cases.len() >= 20);
    for case in cases {
        let (benchmark, d) = from_lab_config(&case["config"]);
        if let Benchmark::GaussianMixture {
            n_gaussians,
            benchmark_seed,
        } = benchmark
        {
            let mixture = Mixture::seeded(benchmark_seed, n_gaussians as usize, d, -10., 10.);
            assert_eq!(mixture.centers, numbers(&case["centers"]), "{benchmark:?}");
            assert_eq!(mixture.stds, numbers(&case["stds"]), "{benchmark:?}");
            assert_eq!(mixture.weights, numbers(&case["weights"]), "{benchmark:?}");
        }
        let evaluator = benchmark.evaluator(d).unwrap();
        for (point, expected) in case["points"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["values"].as_array().unwrap())
        {
            let actual = evaluator.value(&numbers(point));
            let Some(expected) = expected.as_f64() else {
                // Coincident atoms: the lab returns +inf, the port clamps r² and stays finite.
                assert!(matches!(benchmark, Benchmark::LennardJones { .. }));
                assert!(actual.is_finite() && actual > 1e20, "{actual}");
                continue;
            };
            assert!(
                (actual - expected).abs() <= 1e-10 * expected.abs().max(1.),
                "{benchmark:?}: {actual} != {expected}"
            );
        }
    }
}

fn parity_classics() -> Vec<(Benchmark, usize, Vec<f64>)> {
    let lab = |id: &str| Benchmark::from_id(id).unwrap();
    vec![
        (
            lab("quadratic_well"),
            3,
            vec![0.3, -0.7, 1.2, -1.4, 0.9, 0.1],
        ),
        (lab("mexican_hat"), 3, vec![0.3, -0.7, 1.2, -1.4, 0.9, 0.1]),
        (
            Benchmark::MexicanHat {
                lambda_h: 0.5,
                vev: 300.,
                field_scale: 100.,
                tilt: 0.3,
            },
            1,
            vec![0.8, -2.5],
        ),
        (Benchmark::Eggholder, 2, vec![101.3, -37.2, -310.5, 420.25]),
        (Benchmark::Easom, 2, vec![2.9, 3.4, 3.6, 2.2]),
        (Benchmark::HolderTable, 2, vec![7.3, 8.9, -4.2, 1.7]),
        (
            Benchmark::LennardJones { n_atoms: 3 },
            9,
            vec![
                0., 0., 0., 1.1, 0.1, -0.2, 0.4, 1.2, 0.3, -0.5, 0.2, 0.1, 0.9, -0.3, 0.6, 0.1,
                1.0, -0.8,
            ],
        ),
        (Benchmark::Constant, 3, vec![0.3, -0.7, 1.2, -1.4, 0.9, 0.1]),
        (lab("gaussian_mixture"), 2, vec![0.3, -0.7, -6.5, 8.25]),
        (
            Benchmark::GaussianMixture {
                n_gaussians: 5,
                benchmark_seed: 7,
            },
            3,
            vec![0.3, -0.7, 1.2, -9.4, 9.9, 0.1],
        ),
    ]
}

#[test]
fn classic_graphs_match_scalar_values_and_finite_differences() {
    async fn check<T: Real>(value_tolerance: f64, gradient_tolerance: f64) {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, T::PRECISION)
            .await
            .unwrap();
        for (benchmark, d, flat) in parity_classics() {
            let rows = flat.len() / d;
            let points =
                TensorBatch::vectors(rows, d, flat.iter().map(|&v| T::from_f64(v)).collect())
                    .unwrap();
            let p = Population::new(ObservationBatch::positions(points)).unwrap();
            let model = BenchmarkModel {
                benchmark,
                field: "positions".into(),
                direction: ObjectiveDirection::Minimize,
            };
            let rewards = model.evaluate(&p, None, "test", &mut cx).await.unwrap();
            let gradient = GradientProvider::gradient(&model, &p, &mut cx)
                .await
                .unwrap();
            let evaluator = benchmark.evaluator(d).unwrap();
            for i in 0..rows {
                let row = &flat[i * d..(i + 1) * d];
                let expected = evaluator.value(row);
                let scale = expected.abs().max(1.);
                assert!(
                    (rewards.raw[i].to_f64() - expected).abs() <= value_tolerance * scale,
                    "{benchmark:?} value {} != {expected}",
                    rewards.raw[i].to_f64()
                );
                for k in 0..d {
                    let (mut plus, mut minus) = (row.to_vec(), row.to_vec());
                    let h = 1e-6 * row[k].abs().max(1.);
                    plus[k] += h;
                    minus[k] -= h;
                    let finite = (evaluator.value(&plus) - evaluator.value(&minus)) / (2. * h);
                    let analytic = gradient.row(i).unwrap()[k].to_f64();
                    assert!(
                        (finite - analytic).abs() <= gradient_tolerance * finite.abs().max(1.),
                        "{benchmark:?} row {i} coordinate {k}: {analytic} != {finite}"
                    );
                }
            }
        }
    }
    block_on(async {
        check::<f64>(1e-10, 2e-5).await;
        check::<f32>(2e-4, 5e-3).await;
    });
}

#[test]
fn known_minima_are_attained() {
    for benchmark in Benchmark::all() {
        let d = benchmark.fixed_dimension().unwrap_or(5);
        let (Some(minimum), Some(minimizer)) =
            (benchmark.known_minimum(d), benchmark.known_minimizer(d))
        else {
            continue;
        };
        let value = benchmark.evaluator(d).unwrap().value(&minimizer);
        assert!(
            (value - minimum).abs() <= 2e-5 * minimum.abs().max(1.),
            "{benchmark:?}: f(x*) = {value}, catalog minimum {minimum}"
        );
    }
    // Two and three atoms at the pair equilibrium distance 2^(1/6).
    let r = 2f64.powf(1. / 6.);
    let dimer = Benchmark::LennardJones { n_atoms: 2 };
    let value = dimer.evaluator(6).unwrap().value(&[0., 0., 0., r, 0., 0.]);
    assert!((value - dimer.known_minimum(6).unwrap()).abs() < 1e-12);
    let trimer = Benchmark::LennardJones { n_atoms: 3 };
    let triangle = [0., 0., 0., r, 0., 0., r / 2., r * 3f64.sqrt() / 2., 0.];
    let value = trimer.evaluator(9).unwrap().value(&triangle);
    assert!((value - trimer.known_minimum(9).unwrap()).abs() < 1e-12);
    // The far field of the mixture stays finite in both precisions.
    let mixture = Benchmark::from_id("gaussian_mixture").unwrap();
    assert!(mixture.value(&[10f64; 4]).unwrap().is_finite());
    assert!(mixture.value(&[10f32; 4]).unwrap().is_finite());
}

#[test]
fn serialization_is_stable_and_round_trips() {
    // Parameter-free objectives keep their historical bare-string encoding.
    let default = serde_json::to_value(RunConfig::default()).unwrap();
    assert_eq!(default["benchmark"], json!("rastrigin"));
    assert_eq!(default["potential"], Value::Null);
    for (benchmark, debug) in [
        (Benchmark::Sphere, "Sphere"),
        (Benchmark::Rastrigin, "Rastrigin"),
        (Benchmark::Rosenbrock, "Rosenbrock"),
        (Benchmark::StyblinskiTang, "StyblinskiTang"),
        (Benchmark::Quadratic, "Quadratic"),
    ] {
        assert_eq!(format!("{benchmark:?}"), debug);
        assert_eq!(
            serde_json::to_value(benchmark).unwrap(),
            json!(benchmark.id())
        );
        let model = BenchmarkModel {
            benchmark,
            field: "positions".into(),
            direction: ObjectiveDirection::Minimize,
        };
        assert_eq!(
            <BenchmarkModel as RewardSource<f64>>::id(&model),
            format!("benchmark/{debug}/positions/v1")
        );
        assert_eq!(
            <BenchmarkModel as GradientProvider<f64>>::id(&model),
            format!("analytic-gradient/{debug}/Minimize/v1")
        );
    }
    for benchmark in Benchmark::all() {
        let id = benchmark.id();
        assert_eq!(Benchmark::from_id(&id), Some(benchmark));
        let bare: Benchmark = serde_json::from_value(json!(id)).unwrap();
        assert_eq!(bare, benchmark);
        let json = serde_json::to_value(benchmark).unwrap();
        assert_eq!(
            serde_json::from_value::<Benchmark>(json).unwrap(),
            benchmark
        );
        let mut bytes = vec![];
        ciborium::into_writer(&benchmark, &mut bytes).unwrap();
        let cbor: Benchmark = ciborium::from_reader(bytes.as_slice()).unwrap();
        assert_eq!(cbor, benchmark);
    }
    let instance: Benchmark =
        serde_json::from_value(json!({"id": "bbob_21", "coco_instance": 3})).unwrap();
    assert_eq!(
        instance,
        Benchmark::Bbob {
            function: 21,
            instance: 3
        }
    );
    assert_eq!(
        serde_json::to_value(instance).unwrap(),
        json!({"id": "bbob_21", "coco_instance": 3})
    );
    for invalid in [
        json!("bbob_25"),
        json!("bbob_01"),
        json!("unknown"),
        json!({"id": "sphere", "alpha": 1.0}),
        json!({"id": "bbob_3", "coco_instance": 0}),
        json!({"id": "lennard_jones", "n_atoms": 1}),
        json!({"id": "quadratic_well", "alpha": -1.0}),
        json!({"id": "quadratic_well", "curvature": 1.0}),
    ] {
        assert!(
            serde_json::from_value::<Benchmark>(invalid.clone()).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn dimension_rules() {
    assert!(Benchmark::Eggholder.validate(2).is_ok());
    assert!(Benchmark::Eggholder.validate(3).is_err());
    assert!(Benchmark::LennardJones { n_atoms: 4 }.validate(12).is_ok());
    assert!(Benchmark::LennardJones { n_atoms: 4 }.validate(10).is_err());
    let bbob = Benchmark::Bbob {
        function: 7,
        instance: 1,
    };
    assert!(bbob.validate(10).is_ok());
    assert!(bbob.validate(4).is_err());
    let config = RunConfig {
        benchmark: bbob,
        dimensions: 5,
        physics_metric: Some(Default::default()),
        ..Default::default()
    };
    assert!(config.validate().is_err());
}
