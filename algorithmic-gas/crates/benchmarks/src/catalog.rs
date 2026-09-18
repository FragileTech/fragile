//! Machine-readable objective catalog shared by the browser laboratory and the CLI.
//!
//! Entries mirror the Optimization Lab catalog (identifiers, names, groups, domains,
//! dimension rules, parameters) and add where each objective and gradient executes.
use crate::{Benchmark, GradientExecution, RunConfig, bbob};
use algorithmic_gas::Result;
use serde_json::{Map, Value, json};

pub const VERSION: &str = "ag-objectives-1";

fn name(benchmark: Benchmark) -> String {
    match benchmark {
        Benchmark::Sphere => "Sphere".into(),
        Benchmark::Rastrigin => "Rastrigin".into(),
        Benchmark::Rosenbrock => "Rosenbrock".into(),
        Benchmark::StyblinskiTang => "Styblinski–Tang".into(),
        Benchmark::Quadratic => "Unit Quadratic".into(),
        Benchmark::QuadraticWell { .. } => "Quadratic Well".into(),
        Benchmark::MexicanHat { .. } => "Mexican Hat".into(),
        Benchmark::Eggholder => "EggHolder".into(),
        Benchmark::Easom => "Easom".into(),
        Benchmark::HolderTable => "Holder Table".into(),
        Benchmark::LennardJones { .. } => "Lennard–Jones".into(),
        Benchmark::Constant => "Constant".into(),
        Benchmark::StochasticGaussian { .. } => "Stochastic Gaussian".into(),
        Benchmark::GaussianMixture { .. } => "Mixture of Gaussians".into(),
        Benchmark::Bbob { function, .. } => format!(
            "BBOB f{function} · {}",
            bbob::NAMES[usize::from(function) - 1]
        ),
    }
}

fn field(key: &str, label: &str, integer: bool, min: f64, max: f64, default: f64) -> Value {
    json!({
        "key": key,
        "label": label,
        "kind": if integer { "integer" } else { "number" },
        "min": min,
        "max": max,
        "default": default,
    })
}

fn parameter_fields(benchmark: Benchmark) -> Vec<Value> {
    let default = |key: &str| {
        benchmark
            .parameters()
            .iter()
            .find(|(k, _)| *k == key)
            .map_or(0., |(_, v)| *v)
    };
    let number = |key: &str, label: &str, min: f64, max: f64| {
        field(key, label, false, min, max, default(key))
    };
    let integer = |key: &str, label: &str, min: f64, max: f64| {
        field(key, label, true, min, max, default(key))
    };
    match benchmark {
        Benchmark::QuadraticWell { .. } => vec![number("alpha", "Curvature α", 0., 1e6)],
        Benchmark::MexicanHat { .. } => vec![
            number("lambda_h", "Quartic coupling λ", 0., 1e6),
            number("vev", "Vacuum expectation value", 0., 1e6),
            number("field_scale", "Field scale", 1e-12, 1e12),
            number("tilt", "Tilt", -1e6, 1e6),
        ],
        Benchmark::LennardJones { .. } => vec![integer(
            "n_atoms",
            "Atoms",
            2.,
            f64::from(Benchmark::MAX_ATOMS),
        )],
        Benchmark::StochasticGaussian { .. } => {
            vec![number("std", "Noise deviation", 0., 1e6)]
        }
        Benchmark::GaussianMixture { .. } => vec![
            integer(
                "n_gaussians",
                "Components",
                1.,
                f64::from(Benchmark::MAX_GAUSSIANS),
            ),
            integer("benchmark_seed", "Benchmark seed", 0., 2147483647.),
        ],
        Benchmark::Bbob { .. } => vec![integer(
            "coco_instance",
            "COCO instance",
            1.,
            f64::from(bbob::MAX_INSTANCE),
        )],
        _ => vec![],
    }
}

fn gradient(benchmark: Benchmark) -> &'static str {
    match benchmark {
        Benchmark::Eggholder | Benchmark::HolderTable => "analytic; regularized at cusps",
        Benchmark::Constant => "zero",
        Benchmark::StochasticGaussian { .. } => "disabled",
        Benchmark::Bbob { .. } => "central difference (2d objective evaluations)",
        _ => "analytic",
    }
}

fn reference(benchmark: Benchmark) -> Option<&'static str> {
    Some(match benchmark {
        Benchmark::MexicanHat { .. } => "Ring minima only when tilt is zero",
        Benchmark::StyblinskiTang => "Minimum approximately -39.16617 × dimension",
        Benchmark::LennardJones { .. } => {
            "Known energies: 2 atoms -1; 3 atoms -3; 4 atoms -6; 10 atoms -28.422532. \
             Squared pair distances are clamped at 1e-4."
        }
        Benchmark::StochasticGaussian { .. } => "Expected value 0; no spatial optimum",
        Benchmark::GaussianMixture { .. } => {
            "Component centers are reference points, not guaranteed minima"
        }
        Benchmark::Bbob { .. } => {
            "Shifted/rotated BBOB instance; reference minimum resolved on reset. \
             Dimensions: 2, 3, 5, 10, 20, 40."
        }
        Benchmark::Quadratic => "Unit-curvature well of the lecture experiments",
        _ => return None,
    })
}

fn entry(benchmark: Benchmark) -> Value {
    let (low, high) = benchmark.bounds();
    let mut entry = Map::new();
    let mut put = |key: &str, value: Value| {
        entry.insert(key.into(), value);
    };
    put("id", json!(benchmark.id()));
    put("name", json!(name(benchmark)));
    let (suite, group) = match benchmark {
        Benchmark::Bbob { function, .. } => ("bbob", bbob::group(function)),
        Benchmark::Quadratic => ("classic", "Lecture"),
        _ => ("classic", "Classic benchmarks"),
    };
    put("suite", json!(suite));
    put("group", json!(group));
    put("bounds", json!([low, high]));
    match benchmark {
        Benchmark::Eggholder | Benchmark::Easom | Benchmark::HolderTable => {
            put("dimension", json!(2));
        }
        Benchmark::LennardJones { .. } => put("dimensionRule", json!("3 × n_atoms")),
        Benchmark::Bbob { function, .. } => {
            put("minDimension", json!(2));
            put("maxDimension", json!(40));
            put("dimensions", json!(bbob::DIMENSIONS));
            put("function", json!(function));
            put("provider", json!("COCO 2.8.2 (Rust port)"));
            put(
                "source",
                json!("https://coco-platform.org/testsuites/bbob/overview.html"),
            );
        }
        Benchmark::Rosenbrock => put("minDimension", json!(2)),
        _ => put("minDimension", json!(1)),
    }
    let dimension_free_minimum = match benchmark {
        Benchmark::StyblinskiTang | Benchmark::LennardJones { .. } | Benchmark::Bbob { .. } => None,
        other => other.known_minimum(other.fixed_dimension().unwrap_or(2)),
    };
    if let Some(minimum) = dimension_free_minimum {
        put("minimum", json!(minimum));
    }
    if let Some(reference) = reference(benchmark) {
        put("reference", json!(reference));
    }
    let parameters: Map<String, Value> = benchmark
        .parameters()
        .into_iter()
        .map(|(key, value)| (key.into(), json!(value)))
        .collect();
    if !parameters.is_empty() {
        put("parameters", Value::Object(parameters));
    }
    put("parameterFields", json!(parameter_fields(benchmark)));
    put("gradient", json!(gradient(benchmark)));
    put("objective_execution", json!(benchmark.execution()));
    put("gradient_execution", json!(benchmark.gradient_execution()));
    if benchmark.gradient_execution() == GradientExecution::HostCentralDifference {
        put("gradient_cost", json!("2d objective evaluations"));
    }
    if benchmark.is_stochastic() {
        put("stochastic", json!(true));
    }
    put("physics_metric", json!(benchmark.supports_physics_metric()));
    put(
        "molecule",
        json!(matches!(benchmark, Benchmark::LennardJones { .. })),
    );
    Value::Object(entry)
}

/// Every selectable objective with its default parameters, in display order.
pub fn catalog() -> Value {
    json!({
        "version": VERSION,
        "benchmarks": Benchmark::all().into_iter().map(entry).collect::<Vec<_>>(),
    })
}

/// Facts that depend on the configured parameters, dimension and reward shift.
pub fn objective_info(config: &RunConfig) -> Result<Value> {
    config.validate()?;
    let benchmark = config.benchmark;
    let d = config.dimensions;
    let (low, high) = benchmark.bounds();
    let minimizer = benchmark.known_minimizer(d).map(|mut x| {
        for (x, shift) in x.iter_mut().zip(&config.reward_shift) {
            *x += shift;
        }
        x
    });
    let mut info = json!({
        "id": benchmark.id(),
        "name": name(benchmark),
        "bounds": [low, high],
        "dimensions": d,
        "minimum": benchmark.known_minimum(d),
        "minimizer": minimizer,
        "objective_execution": benchmark.execution(),
        "gradient_execution": benchmark.gradient_execution(),
        "stochastic": benchmark.is_stochastic(),
        "molecule": matches!(benchmark, Benchmark::LennardJones { .. }),
    });
    if let Benchmark::Bbob { function, instance } = benchmark {
        let problem = bbob::problem(function, d, instance)
            .map_err(algorithmic_gas::GasError::Configuration)?;
        info["coco_problem_id"] = json!(problem.problem_id());
        info["coco_version"] = json!("2.8.2");
    }
    Ok(info)
}
