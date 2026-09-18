use algorithmic_gas::{BackendKind, Precision, Real, RecordingConfig, Result};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig};
use std::{env, time::Instant};

/// Durable recording of the run through the engine's `RunArchive`.
struct Recording {
    path: String,
    config: RecordingConfig,
}
async fn run<T: Real>(config: RunConfig, steps: usize, recording: Option<Recording>) -> Result<()> {
    let initialization = Instant::now();
    let mut gas = config.build::<T>().await?;
    if let Some(recording) = &recording {
        gas.start_recording(recording.config.clone())?;
    }
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
    // Population geometry of the final state, when the gas has a geometry stage.
    let geometry = gas.graph().map(|graph| {
        let fields = &gas.population().observations.fields;
        let column = |name: &str| -> Vec<f64> {
            fields
                .get(name)
                .map(|f| f.values().iter().map(|v| v.to_f64()).collect())
                .unwrap_or_default()
        };
        let volume = column(algorithmic_gas::tessellation::stage::VOLUME_FIELD);
        let mut summary = serde_json::json!({
            "neighbor_edges": graph.graph.edges(),
            "mean_volume_element": volume.iter().sum::<f64>() / volume.len().max(1) as f64,
        });
        for (name, field) in fields {
            if let Some(curvature) = name.strip_prefix("geometry.curvature.") {
                let r: Vec<f64> = field.values().iter().map(|v| v.to_f64()).collect();
                summary[format!("mean_{curvature}")] =
                    (r.iter().sum::<f64>() / r.len().max(1) as f64).into();
                summary[format!("action_{curvature}")] = r
                    .iter()
                    .zip(&volume)
                    .map(|(r, v)| r * v)
                    .sum::<f64>()
                    .into();
            }
        }
        summary
    });
    if let (Some(recording), Some(archive)) = (&recording, gas.stop_recording()) {
        let bytes = if recording.path.ends_with(".json") {
            serde_json::to_vec(&archive)
                .map_err(|e| algorithmic_gas::GasError::Checkpoint(e.to_string()))?
        } else {
            archive.to_bytes()?
        };
        std::fs::write(&recording.path, bytes)
            .map_err(|e| algorithmic_gas::GasError::Execution(e.to_string()))?;
    }
    let output = serde_json::json!({"config":config,"geometry":geometry,"steps":gas.step_number(),"initialization_seconds":initialization_seconds,"elapsed_seconds":start.elapsed().as_secs_f64(),"initial_minimum":initial,"final_best":best,"reward_evaluations":gas.reward_evaluations(),"execution":gas.execution_stats(),"execution_model":"host-orchestrated Burn batches; accelerator transfers included"});
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
    Ok(())
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut config = RunConfig::default();
    let mut steps = 100usize;
    let mut from_file = false;
    let mut dimensions_given = false;
    let mut record_path: Option<String> = None;
    let mut recording = RecordingConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "gas-benchmark [--config FILE] [--einstein-hilbert] [--steps N] [--precision f32|f64] [--backend cpu|wgpu|cuda] [--benchmark ID] [--instance N] [--param KEY=VALUE] [--walkers N] [--dimensions D] [--seed N] [--record FILE.cbor|FILE.json] [--record-steps N] [--record-bytes N] [--record-graph]\n--list-benchmarks prints the objective catalog (ids, domains, dimension rules, parameters).\n--einstein-hilbert selects the free gas rewarded with the Einstein-Hilbert action of its tessellation geometry.\n--record writes the run archive; --record-graph adds the tessellation graph of every step.\nGPU profiles are explicitly host-orchestrated and report transfers."
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
        if arg == "--einstein-hilbert" {
            config = RunConfig::einstein_hilbert()?;
            from_file = true;
            continue;
        }
        if arg == "--record-graph" {
            recording.graph = true;
            continue;
        }
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--config" => {
                config = serde_json::from_str(&std::fs::read_to_string(value)?)?;
                from_file = true;
            }
            "--steps" => steps = value.parse()?,
            "--record" => record_path = Some(value),
            "--record-steps" => recording.max_steps = value.parse()?,
            "--record-bytes" => recording.max_bytes = value.parse()?,
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
    let recording = record_path.map(|path| Recording {
        path,
        config: RecordingConfig {
            max_steps: recording.max_steps.max(steps),
            ..recording
        },
    });
    match config.gas.precision {
        Precision::F32 => futures_lite::future::block_on(run::<f32>(config, steps, recording))?,
        Precision::F64 => futures_lite::future::block_on(run::<f64>(config, steps, recording))?,
    }
    Ok(())
}
