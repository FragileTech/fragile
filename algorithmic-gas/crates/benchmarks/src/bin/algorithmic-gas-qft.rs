// Reproducible Part VI runner. All scientific calculations live in algorithmic-gas.
use algorithmic_gas::physics::partvi::{
    ExperimentRequest, analyze, analyze_archive, supports_archive,
};
use algorithmic_gas::{BackendKind, Precision, RecordingConfig, RunArchive};
use algorithmic_gas_benchmarks::RunConfig;
use serde_json::{Value, json};
use std::{
    env, fs,
    io::{self, Read},
    path::Path,
};
fn read_json(path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let mut bytes = String::new();
    if path == "-" {
        io::stdin().read_to_string(&mut bytes)?;
    } else {
        bytes = fs::read_to_string(path)?;
    }
    if bytes.len() > 128 * 1024 * 1024 {
        return Err("input exceeds 128 MiB".into());
    }
    Ok(serde_json::from_str(&bytes)?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    futures_lite::future::block_on(run())
}
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".into());
    if ["help", "--help", "-h"].contains(&command.as_str()) {
        println!(
            "algorithmic-gas-qft analyze REQUEST.json [--archive ARCHIVE.json]\nalgorithmic-gas-qft run REQUEST.json [--config RUN.json] [--steps 32] [--save-archive ARCHIVE.json]\nalgorithmic-gas-qft sweep SWEEP.json [--archive ARCHIVE.json]\nAll commands: [--output RESULTS.json]. Use - for JSON stdin.\nRequest: {{\"experiment\":8,\"parameters\":{{\"seed\":7}}}}\nSweep: {{\"requests\":[REQUEST,...]}} or {{\"experiment\":7,\"parameters\":{{}},\"parameter\":\"n\",\"values\":[64,128,256]}}."
        );
        return Ok(());
    }
    let input = args.next().ok_or("missing request file")?;
    let mut output = None;
    let mut archive_path = None;
    let mut config_path = None;
    let mut save_archive = None;
    let mut steps = 32usize;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--output" => output = Some(value),
            "--archive" => archive_path = Some(value),
            "--config" => config_path = Some(value),
            "--save-archive" => save_archive = Some(value),
            "--steps" => steps = value.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if steps > 100_000 {
        return Err("run step budget exceeds 100000".into());
    }
    let data = read_json(&input)?;
    let mut archive: Option<RunArchive<f64>> = archive_path
        .as_deref()
        .map(|p| read_json(p).and_then(|v| Ok(serde_json::from_value(v)?)))
        .transpose()?;
    let result = match command.as_str() {
        "run" => {
            if archive.is_some() {
                return Err(
                    "run uses its executed trajectory; use analyze for an imported archive".into(),
                );
            }
            let request: ExperimentRequest = serde_json::from_value(data)?;
            request.validate()?;
            if !supports_archive(request.experiment) && !matches!(request.experiment, 19 | 22 | 45)
            {
                return Err(format!("VI-{:02} has no executed-run measurement; use analyze for its declared finite model", request.experiment).into());
            }
            let mut config: RunConfig = match &config_path {
                Some(p) => serde_json::from_value(read_json(p)?)?,
                None => {
                    let mut c = RunConfig {
                        walkers: 32,
                        dimensions: 3,
                        benchmark: algorithmic_gas_benchmarks::Benchmark::Quadratic,
                        initial_lower: -1.,
                        initial_upper: 1.,
                        ..Default::default()
                    };
                    c.gas.boundary = algorithmic_gas::boundary::BoundaryPolicy::Unbounded;
                    c.gas.kinetic = algorithmic_gas::kinetic::KineticOperator {
                        integrator: algorithmic_gas::kinetic::KineticKind::Baoab {
                            positions: "positions".into(),
                            velocities: "velocities".into(),
                            dt: 0.04,
                            friction: 1.,
                        },
                        noise: algorithmic_gas::noise::Noise {
                            innovation: algorithmic_gas::noise::InnovationLaw::Gaussian,
                            geometry: algorithmic_gas::noise::NoiseGeometry::Isotropic {
                                scale: algorithmic_gas::noise::FactorValues::Constant {
                                    values: vec![0.8_f64.sqrt()],
                                },
                            },
                        },
                    };
                    c.gas.qft.viscosity = Some(algorithmic_gas::kinetic::ViscousForceConfig {
                        coefficient: 1.,
                        bandwidth: 1.,
                        row_normalized: false,
                    });
                    c
                }
            };
            if config_path.is_none() && matches!(request.experiment, 39 | 45 | 48) {
                config.physics_metric = Some(
                    algorithmic_gas_benchmarks::physics_metric::PhysicsMetricConfig::default(),
                );
            }
            config.gas.precision = Precision::F64;
            config.gas.backend = BackendKind::Cpu;
            config.gas.seed = request.usize("seed", 7) as u64;
            let ensemble = matches!(request.experiment, 19 | 22 | 45);
            if ensemble && save_archive.is_some() {
                return Err("ensemble experiments export replica estimates in results; --save-archive requires a single trajectory experiment".into());
            }
            let measured = if ensemble {
                algorithmic_gas_benchmarks::qft_experiments::run(&config, &request).await?
            } else {
                let mut gas = config.build::<f64>().await?;
                gas.start_recording(RecordingConfig {
                    max_steps: steps.max(1),
                    max_bytes: 512 * 1024 * 1024,
                })?;
                for _ in 0..steps {
                    gas.step().await?;
                }
                archive = gas.recording().cloned();
                analyze_archive(&request, archive.as_ref())?
            };
            json!({"schema":"fragile-partvi-results-v1","command":"run","config":config,"results":[measured]})
        }
        "analyze" => {
            let request: ExperimentRequest = serde_json::from_value(data)?;
            let measured = if archive.is_some() {
                analyze_archive(&request, archive.as_ref())?
            } else {
                analyze(&request)?
            };
            json!({"schema":"fragile-partvi-results-v1","command":"analyze","results":[measured]})
        }
        "sweep" => {
            let requests: Vec<ExperimentRequest> = if let Some(v) = data.get("requests") {
                serde_json::from_value(v.clone())?
            } else {
                let id = data
                    .get("experiment")
                    .and_then(Value::as_u64)
                    .ok_or("sweep experiment required")?;
                let key = data
                    .get("parameter")
                    .and_then(Value::as_str)
                    .ok_or("sweep parameter required")?;
                if data.get("parameters").is_some_and(|p| !p.is_object()) {
                    return Err("sweep parameters must be an object".into());
                }
                if !(1..=66).contains(&id) {
                    return Err("sweep experiment must be in 1..=66".into());
                }
                let values = data
                    .get("values")
                    .and_then(Value::as_array)
                    .ok_or("sweep values required")?;
                values
                    .iter()
                    .map(|v| {
                        let mut parameters =
                            data.get("parameters").cloned().unwrap_or_else(|| json!({}));
                        if let Some(map) = parameters.as_object_mut() {
                            map.insert(key.to_string(), v.clone());
                        }
                        ExperimentRequest {
                            experiment: id as u32,
                            parameters,
                        }
                    })
                    .collect()
            };
            if requests.len() > 4096 {
                return Err("sweep exceeds 4096 requests".into());
            }
            let mut results = vec![];
            for (i, request) in requests.iter().enumerate() {
                results.push(analyze_archive(request, archive.as_ref())?);
                eprintln!("completed {}/{}", i + 1, requests.len());
            }
            json!({"schema":"fragile-partvi-results-v1","command":"sweep","results":results})
        }
        _ => return Err("command must be run, analyze or sweep".into()),
    };
    if let Some(path) = save_archive {
        fs::write(
            Path::new(&path),
            serde_json::to_string(&archive.ok_or("no executed archive to save")?)?,
        )?;
    }
    let text = serde_json::to_string_pretty(&result)?;
    if let Some(path) = output {
        fs::write(path, text + "\n")?;
    } else {
        println!("{text}");
    }
    Ok(())
}
