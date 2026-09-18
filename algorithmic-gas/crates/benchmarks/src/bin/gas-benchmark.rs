use algorithmic_gas::{BackendKind, Precision, Real, Result};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig};
use std::{env, time::Instant};

async fn run<T: Real>(config: RunConfig, steps: usize) -> Result<()> {
    let initialization = Instant::now();
    let mut gas = config.build::<T>().await?;
    let initialization_seconds = initialization.elapsed().as_secs_f64();
    let initial = gas
        .population()
        .rewards
        .raw
        .iter()
        .map(|x| x.to_f64())
        .fold(f64::INFINITY, f64::min);
    let start = Instant::now();
    for _ in 0..steps {
        gas.step().await?;
    }
    let best = gas
        .population()
        .rewards
        .raw
        .iter()
        .zip(gas.population().eligible(config.gas.include_truncated))
        .filter_map(|(x, alive)| alive.then_some(x.to_f64()))
        .reduce(|a, b| {
            if config.gas.fitness.direction
                == algorithmic_gas::fitness::ObjectiveDirection::Minimize
            {
                a.min(b)
            } else {
                a.max(b)
            }
        });
    let output = serde_json::json!({"config":config,"steps":gas.step_number(),"initialization_seconds":initialization_seconds,"elapsed_seconds":start.elapsed().as_secs_f64(),"initial_minimum":initial,"final_best":best,"reward_evaluations":gas.reward_evaluations(),"execution":gas.execution_stats(),"execution_model":"host-orchestrated Burn batches; accelerator transfers included"});
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
    Ok(())
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut config = RunConfig::default();
    let mut steps = 100usize;
    let mut from_file = false;
    let mut dimensions_given = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "gas-benchmark [--config FILE] [--steps N] [--precision f32|f64] [--backend cpu|wgpu|cuda] [--benchmark ID] [--instance N] [--param KEY=VALUE] [--walkers N] [--dimensions D] [--seed N]\n--list-benchmarks prints the objective catalog (ids, domains, dimension rules, parameters).\nGPU profiles are explicitly host-orchestrated and report transfers."
            );
            return Ok(());
        }
        if arg == "--list-benchmarks" {
            println!(
                "{}",
                serde_json::to_string_pretty(&algorithmic_gas_benchmarks::catalog::catalog())?
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--config" => {
                config = serde_json::from_str(&std::fs::read_to_string(value)?)?;
                from_file = true;
            }
            "--steps" => steps = value.parse()?,
            "--walkers" => config.walkers = value.parse()?,
            "--dimensions" => {
                config.dimensions = value.parse()?;
                dimensions_given = true;
            }
            "--seed" => config.gas.seed = value.parse()?,
            "--precision" => {
                config.gas.precision = match value.as_str() {
                    "f32" => Precision::F32,
                    "f64" => Precision::F64,
                    _ => return Err("unknown precision".into()),
                }
            }
            "--backend" => {
                config.gas.backend = match value.as_str() {
                    "cpu" => BackendKind::Cpu,
                    "wgpu" => BackendKind::Wgpu,
                    "cuda" => BackendKind::Cuda,
                    _ => return Err("unknown backend".into()),
                }
            }
            "--benchmark" => {
                config.benchmark = Benchmark::from_id(&value)
                    .ok_or_else(|| format!("unknown benchmark {value}; see --list-benchmarks"))?;
            }
            "--instance" => config
                .benchmark
                .set_parameter("coco_instance", value.parse()?)?,
            "--param" => {
                let (key, number) = value.split_once('=').ok_or("--param expects key=value")?;
                config.benchmark.set_parameter(key, number.parse()?)?;
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    if !from_file
        && !dimensions_given
        && let Some(fixed) = config.benchmark.fixed_dimension()
    {
        config.dimensions = fixed;
    }
    if !from_file
        && let algorithmic_gas::boundary::BoundaryPolicy::AbsorbingBox { domain, .. } =
            &mut config.gas.boundary
    {
        let (lo, hi) = config.benchmark.bounds();
        domain.lower = vec![lo; config.dimensions];
        domain.upper = vec![hi; config.dimensions];
    }
    match config.gas.precision {
        Precision::F32 => futures_lite::future::block_on(run::<f32>(config, steps))?,
        Precision::F64 => futures_lite::future::block_on(run::<f64>(config, steps))?,
    }
    Ok(())
}
