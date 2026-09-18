//! The objective catalog against the Optimization Lab catalog, and engine-level behaviour
//! of host-evaluated and stochastic objectives.
use algorithmic_gas::{kinetic::KineticKind, *};
use algorithmic_gas_benchmarks::{
    Benchmark, RunConfig,
    catalog::{catalog, objective_info},
};
use futures_lite::future::block_on;
use serde_json::{Value, json};

fn lab_catalog() -> Vec<Value> {
    let path = format!(
        "{}/tests/fixtures/objective-goldens.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let goldens: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    goldens["catalog"]["benchmarks"].as_array().unwrap().clone()
}

/// JSON equality up to the integer/float spelling of numbers.
fn normalized(value: &Value) -> Value {
    match value {
        Value::Number(n) => json!(n.as_f64().unwrap()),
        Value::Array(items) => Value::Array(items.iter().map(normalized).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), normalized(value)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[test]
fn catalog_mirrors_the_optimization_lab() {
    let ours = catalog();
    let ours = ours["benchmarks"].as_array().unwrap();
    let lab = lab_catalog();
    // Same entries in the same order; the unit lecture quadratic is appended.
    assert_eq!(ours.len(), lab.len() + 1);
    assert_eq!(ours.last().unwrap()["id"], json!("quadratic"));
    for (ours, lab) in ours.iter().zip(&lab) {
        let lab_id = lab["id"].as_str().unwrap();
        let expected_id = if lab_id == "quadratic" {
            "quadratic_well"
        } else {
            lab_id
        };
        assert_eq!(ours["id"], json!(expected_id));
        for key in [
            "name",
            "bounds",
            "dimension",
            "dimensions",
            "minDimension",
            "maxDimension",
            "dimensionRule",
            "parameters",
            "function",
        ] {
            assert_eq!(
                normalized(&ours[key]),
                normalized(&lab[key]),
                "{lab_id}.{key}"
            );
        }
        if !lab["minimum"].is_null() {
            assert_eq!(
                normalized(&ours["minimum"]),
                normalized(&lab["minimum"]),
                "{lab_id}.minimum"
            );
        }
        if lab["suite"] == json!("bbob") {
            assert_eq!(ours["group"], lab["group"]);
            assert_eq!(ours["objective_execution"], json!("host"));
            assert_eq!(ours["gradient_execution"], json!("host_central_difference"));
        }
        // Every declared default parameter has an input field.
        let fields = ours["parameterFields"].as_array().unwrap();
        let declared = ours["parameters"]
            .as_object()
            .map_or(0, serde_json::Map::len);
        assert_eq!(fields.len(), declared, "{lab_id} parameter fields");
        let id = ours["id"].as_str().unwrap();
        assert_eq!(Benchmark::from_id(id).unwrap().id(), id);
    }
}

fn run_config(benchmark: Benchmark, dimensions: usize) -> RunConfig {
    let (low, high) = benchmark.bounds();
    let mut config = RunConfig {
        benchmark,
        dimensions,
        walkers: 24,
        initial_lower: low / 2.,
        initial_upper: high / 2.,
        ..Default::default()
    };
    config.gas.boundary = boundary::BoundaryPolicy::AbsorbingBox {
        field: "positions".into(),
        domain: boundary::BoxDomain {
            lower: vec![low; dimensions],
            upper: vec![high; dimensions],
        },
    };
    config
}

#[test]
fn objective_info_resolves_instances() {
    let mut config = run_config(
        Benchmark::Bbob {
            function: 21,
            instance: 3,
        },
        5,
    );
    config.reward_shift = vec![0.25; 5];
    let info = objective_info(&config).unwrap();
    assert_eq!(info["coco_problem_id"], json!("bbob_f021_i03_d05"));
    assert_eq!(info["objective_execution"], json!("host"));
    let minimizer: Vec<f64> = info["minimizer"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let unshifted: Vec<f64> = minimizer.iter().map(|x| x - 0.25).collect();
    let value = config.benchmark.evaluator(5).unwrap().value(&unshifted);
    assert!((value - info["minimum"].as_f64().unwrap()).abs() < 1e-9);
    let molecule = objective_info(&run_config(Benchmark::LennardJones { n_atoms: 4 }, 12)).unwrap();
    assert_eq!(molecule["molecule"], json!(true));
    assert_eq!(molecule["minimum"], json!(-6.0));
}

#[test]
fn host_objectives_replay_and_account_evaluations() {
    async fn replay<T: Real>(mut config: RunConfig, host: bool) {
        config.gas.precision = T::PRECISION;
        let mut gas = config.build::<T>().await.unwrap();
        for _ in 0..2 {
            gas.step().await.unwrap();
        }
        let saved = Checkpoint::from_bytes(&gas.checkpoint().to_bytes().unwrap()).unwrap();
        let first = gas.step().await.unwrap();
        let population = gas.population().clone();
        gas.restore(saved).unwrap();
        let second = gas.step().await.unwrap();
        assert_eq!(first, second);
        assert_eq!(&population, gas.population());
        let stats = gas.execution_stats();
        let n = config.walkers as u64;
        // Initial evaluation plus three per step, exactly as for graph objectives.
        assert_eq!(gas.reward_evaluations(), n + 3 * n * 3);
        if host {
            assert_eq!(stats.host_reward_evaluations, gas.reward_evaluations());
        } else {
            assert_eq!(stats.host_reward_evaluations, 0);
        }
    }
    block_on(async {
        let bbob = Benchmark::Bbob {
            function: 15,
            instance: 2,
        };
        replay::<f64>(run_config(bbob, 3), true).await;
        replay::<f32>(run_config(bbob, 3), true).await;
        replay::<f64>(run_config(Benchmark::Rastrigin, 3), false).await;
        let noisy = Benchmark::StochasticGaussian { std: 1. };
        replay::<f64>(run_config(noisy, 2), false).await;
        replay::<f32>(run_config(noisy, 2), false).await;

        // A BAOAB run spends 2·d host evaluations per eligible walker and gradient call.
        let mut config = run_config(bbob, 3);
        config.gas.precision = Precision::F64;
        config.gas.kinetic.integrator = KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 0.01,
            friction: 1.,
        };
        let mut gas = config.build::<f64>().await.unwrap();
        gas.step().await.unwrap();
        let spent = gas.execution_stats().host_gradient_evaluations;
        assert!(spent > 0 && spent % (2 * 3) == 0, "{spent}");
    });
}

#[test]
fn stochastic_rewards_are_fresh_each_stage_and_step() {
    block_on(async {
        let mut config = run_config(Benchmark::StochasticGaussian { std: 2. }, 2);
        config.gas.precision = Precision::F64;
        let mut gas = config.build::<f64>().await.unwrap();
        let initial = gas.population().rewards.raw.clone();
        assert!(initial.iter().any(|&r| r != 0.));
        let first = gas.step().await.unwrap();
        let second = gas.step().await.unwrap();
        assert_ne!(first.pre_clone_rewards.raw, second.pre_clone_rewards.raw);
        assert_ne!(first.pre_clone_rewards.raw, first.final_rewards.raw);
        let spread = initial.iter().map(|r| r * r).sum::<f64>() / initial.len() as f64;
        assert!(spread > 0.5 && spread < 16., "{spread}");
    });
}
