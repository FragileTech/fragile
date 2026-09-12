//! Reproducible research sweep. Each completed job is atomically saved; rerunning
//! the same command resumes missing jobs. Interrupt between jobs with Ctrl-C.
use algorithmic_gas::{
    Precision, RecordingConfig,
    boundary::BoundaryPolicy,
    fitness::Standardizer,
    kinetic::KineticKind,
    physics::{
        balances::analyze_step,
        evolution::{ReplicaSample, metric_evolution},
    },
};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig, physics_metric::PhysicsMetricConfig};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
type Error = Box<dyn std::error::Error>;
fn atomic(path: &Path, value: &Value) -> Result<(), Error> {
    atomic_bytes(path, &serde_json::to_vec_pretty(value)?)
}
fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}
fn exclusive_lock(path: &Path) -> Result<std::fs::File, Error> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    file.lock()?;
    Ok(file)
}
fn mean_metric(step: &algorithmic_gas::tracking::RecordedStep<f64>) -> Result<Vec<f64>, Error> {
    let f = step
        .field_evaluations
        .iter()
        .find(|f| f.field == "fitness_metric")
        .ok_or("missing O metric")?;
    let width = f.item_shape.iter().product::<usize>();
    let mut mean = vec![0.; width];
    let mut count = 0;
    for i in 0..f.rows {
        if f.available[i] {
            count += 1;
            for (j, m) in mean.iter_mut().enumerate() {
                *m += f.values[i * width + j];
            }
        }
    }
    if count == 0 {
        return Err("no available O metric".into());
    }
    for m in &mut mean {
        *m /= count as f64;
    }
    Ok(mean)
}
async fn job(
    c: &RunConfig,
    steps: usize,
    replicas: usize,
    job_id: usize,
    output: &Path,
) -> Result<Value, Error> {
    let started = Instant::now();
    let mut run = c.build::<f64>().await?;
    let mut records = Vec::new();
    let checkpoints = [1, steps.div_ceil(2), steps];
    for number in 1..=steps {
        let record = checkpoints.contains(&number);
        if record {
            run.start_recording(RecordingConfig {
                max_steps: 1,
                max_bytes: 128 * 1024 * 1024,
            })?;
        }
        run.step().await?;
        if record {
            let archive = run.stop_recording().unwrap();
            let step = &archive.steps[0];
            let metric = mean_metric(step)?;
            let balance = analyze_step(
                step,
                &c.gas,
                Some(&c.benchmark.physics_objective(c.dimensions)),
            )?;
            let curvature = step
                .field_evaluations
                .iter()
                .find(|f| f.field == "fitness_scalar_curvature")
                .ok_or("missing curvature field")?;
            let values: Vec<_> = curvature
                .values
                .iter()
                .zip(&curvature.available)
                .filter_map(|(&v, &available)| available.then_some(v))
                .collect();
            let archive_name = format!("job_{job_id:04}_step_{number:04}.cbor");
            atomic_bytes(&output.join(&archive_name), &archive.to_bytes()?)?;
            let evolution = if replicas > 0 {
                let checkpoint = run.checkpoint();
                let mut samples = Vec::new();
                for replica in 0..2 * replicas {
                    let key = 10_000_000_000
                        + job_id as u64 * 1_000_000_000
                        + number as u64 * 100_000
                        + replica as u64;
                    let cp = checkpoint.with_future_seed(key)?;
                    let mut rc = c.clone();
                    rc.gas = cp.config.clone();
                    let mut independent = rc.build::<f64>().await?;
                    independent.restore(cp)?;
                    independent.start_recording(RecordingConfig {
                        max_steps: 1,
                        max_bytes: 128 * 1024 * 1024,
                    })?;
                    independent.step().await?;
                    let next = mean_metric(&independent.recording().unwrap().steps[0])?;
                    samples.push(ReplicaSample {
                        key,
                        increment: next.iter().zip(&metric).map(|(a, b)| a - b).collect(),
                    });
                }
                Some(metric_evolution(
                    &samples[..replicas],
                    &samples[replicas..],
                )?)
            } else {
                None
            };
            records.push(json!({"step":number,"archive":archive_name,"balance":balance,"mean_O_metric":metric,"mean_scalar_curvature":if values.is_empty(){None}else{Some(values.iter().sum::<f64>()/values.len() as f64)},"curvature_available":values.len(),"curvature_queries":curvature.rows,"conditional_metric_evolution":evolution}));
        }
    }
    Ok(
        json!({"schema_version":1,"configuration":c,"steps":steps,"replicas_per_pool":replicas,"records":records,"elapsed_seconds":started.elapsed().as_secs_f64(),"interpretation":"Actual fixed-step generalized-metric gas. Curvature is computed per O query before averaging. Conditional increments of the population-average O metric use fresh future RNG keys with physical/history state fixed.","execution":run.execution_stats()}),
    )
}
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().collect();
    let research = args.iter().any(|a| a == "--research");
    let local = args.iter().any(|a| a == "--local");
    let integer_arg = |name: &str, default: usize| -> Result<usize, Error> {
        args.windows(2)
            .find(|v| v[0] == name)
            .map_or(Ok(default), |v| Ok(v[1].parse()?))
    };
    let shards = integer_arg("--shards", 1)?;
    let shard = integer_arg("--shard", 0)?;
    if shards == 0 || shard >= shards {
        return Err("require shards > 0 and 0 <= shard < shards".into());
    }
    let output = args
        .windows(2)
        .find(|v| v[0] == "--output")
        .map(|v| v[1].as_str())
        .unwrap_or("outputs/physics-research");
    let output = Path::new(output);
    std::fs::create_dir_all(output)?;
    let seeds = if research { 32 } else { 2 };
    let steps: usize = if research {
        if local { 256 } else { 1024 }
    } else {
        32
    };
    let replicas = if research { 1024 } else { 16 };
    let mut jobs = Vec::new();
    for d in [2, 3] {
        for n in if research {
            vec![32, 128, 512]
        } else {
            vec![16]
        } {
            for seed in 0..seeds {
                let mut c = RunConfig {
                    benchmark: Benchmark::Quadratic,
                    dimensions: d,
                    walkers: n,
                    physics_metric: Some(PhysicsMetricConfig::default()),
                    ..Default::default()
                };
                c.gas.precision = Precision::F64;
                c.gas.seed = 7 + seed;
                c.gas.boundary = BoundaryPolicy::Unbounded;
                c.gas.kinetic.integrator = KineticKind::Baoab {
                    positions: "positions".into(),
                    velocities: "velocities".into(),
                    dt: 0.02,
                    friction: 1.,
                };
                jobs.push(("base".to_string(), c, if seed == 0 { replicas } else { 0 }));
            }
        }
    }
    if local {
        let base = jobs
            .iter()
            .find(|(_, c, _)| c.dimensions == 3)
            .unwrap()
            .1
            .clone();
        jobs.clear();
        for width in [0.3, 1., 3.] {
            for seed in 0..seeds {
                let mut c = base.clone();
                c.gas.seed = 7 + seed;
                c.gas.fitness.reward_standardizer = Standardizer::Local {
                    sigma_min: 0.03,
                    distance: algorithmic_gas::geometry::Distance::default(),
                    kernel: algorithmic_gas::geometry::Kernel::Gaussian { width },
                    include_self: true,
                };
                c.gas.fitness.diversity_standardizer = c.gas.fitness.reward_standardizer.clone();
                jobs.push((format!("local_width_{width}"), c, 0));
            }
        }
    } else if research {
        // One-factor controls at N=128,d=3; each has 32 independent trajectories.
        let base = jobs
            .iter()
            .find(|(_, c, _)| c.dimensions == 3 && c.walkers == 128)
            .unwrap()
            .1
            .clone();
        for variant in [
            "epsilon_small",
            "epsilon_large",
            "temperature_low",
            "temperature_high",
            "friction_low",
            "friction_high",
            "regularizer_large",
            "half_step",
        ] {
            for seed in 0..seeds {
                let mut c = base.clone();
                c.gas.seed = 7 + seed;
                match variant {
                    "epsilon_small" => c.physics_metric.as_mut().unwrap().epsilon = 0.03,
                    "epsilon_large" => c.physics_metric.as_mut().unwrap().epsilon = 0.3,
                    "temperature_low" => c.physics_metric.as_mut().unwrap().temperature = 0.25,
                    "temperature_high" => c.physics_metric.as_mut().unwrap().temperature = 4.,
                    "regularizer_large" => {
                        c.gas.fitness.reward_standardizer = Standardizer::Global { sigma_min: 0.3 };
                        c.gas.fitness.diversity_standardizer =
                            Standardizer::Global { sigma_min: 0.3 };
                    }
                    _ => {
                        if let KineticKind::Baoab { dt, friction, .. } =
                            &mut c.gas.kinetic.integrator
                        {
                            match variant {
                                "friction_low" => *friction = 0.3,
                                "friction_high" => *friction = 3.,
                                "half_step" => *dt = 0.01,
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                jobs.push((variant.into(), c, 0));
            }
        }
    }
    let manifest_lock = exclusive_lock(&output.join("manifest.lock"))?;
    atomic(
        &output.join("manifest.json"),
        &json!({"schema_version":1,"preset":if research{"research"}else{"smoke"},"normalization_study":if local{"local_gaussian_width"}else{"global"},"jobs":jobs.len(),"seeds_per_configuration":seeds,"steps":steps,"replicas_per_pool_for_seed_7_base_cases":if local{0}else{replicas},"checkpoint_steps":[1,steps.div_ceil(2),steps],"backend":"cpu","precision":"f64","resumption":"completed JSON jobs are validated against configuration before reuse","design":["step-size comparison changes BAOAB time step with the clone operator fixed","unbounded harmonic force with programmed fitness-dependent noise","conditional replicas at seed 7 global base checkpoints; other seeds estimate trajectory variability"]}),
    )?;
    drop(manifest_lock);
    futures_lite::future::block_on(async {
        for (id, (variant, c, replicas)) in jobs.iter().enumerate() {
            if id % shards != shard {
                continue;
            }
            let path = output.join(format!("job_{id:04}.json"));
            // The file holds its lock through archive writes and JSON commit.
            // Closing it also releases the lock after interruption or failure.
            let _job_lock = exclusive_lock(&path.with_extension("lock"))?;
            if path.exists() {
                let saved: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                if saved["configuration"] != serde_json::to_value(c)?
                    || saved["steps"] != steps
                    || saved["replicas_per_pool"] != *replicas
                {
                    return Err("existing research job configuration differs; select a new output directory".into());
                }
                continue;
            }
            eprintln!(
                "job {}/{}: {variant}, d={}, N={}, seed={}",
                id + 1,
                jobs.len(),
                c.dimensions,
                c.walkers,
                c.gas.seed
            );
            let result = job(c, steps, *replicas, id, output).await?;
            atomic(&path, &result)?;
        }
        Ok::<_, Error>(())
    })?;
    Ok(())
}
