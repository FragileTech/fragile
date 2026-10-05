//! Reproducible EH operator estimates from native stage records and replicas.
use algorithmic_gas::{Precision, RecordingConfig};
use algorithmic_gas_benchmarks::{
    RunConfig,
    convergence_einstein_hilbert::{EhStepEstimate, analyze_step},
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
type Error = Box<dyn std::error::Error>;
#[derive(Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Config {
    walkers: Vec<usize>,
    seeds: Vec<u64>,
    steps: usize,
    replica_walkers: Vec<usize>,
    replicas: usize,
    warmup: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            walkers: vec![32, 64, 128, 256],
            seeds: vec![7, 1729, 991],
            steps: 400,
            replica_walkers: vec![32, 64],
            replicas: 128,
            warmup: 39,
        }
    }
}
#[derive(Serialize)]
struct Run {
    config: RunConfig,
    estimates: Vec<EhStepEstimate>,
}
#[derive(Serialize)]
struct ReplicaReport {
    config: RunConfig,
    warmup: usize,
    samples: usize,
    first_future_seed: u64,
    clone_mean_residual: f64,
    clone_standard_error: f64,
    clone_standardized_residual: f64,
    kinetic_mean_residual: f64,
    kinetic_standard_error: f64,
    kinetic_standardized_residual: f64,
    mean_predicted_kinetic_variance: f64,
    exact_checks_passed: bool,
}
fn run_config(n: usize, seed: u64) -> Result<RunConfig, Error> {
    let mut c = RunConfig::einstein_hilbert()?;
    c.walkers = n;
    c.gas.seed = seed;
    c.gas.precision = Precision::F64;
    Ok(c)
}
fn record() -> RecordingConfig {
    RecordingConfig {
        graph: true,
        ..RecordingConfig::default()
    }
}
fn stats(x: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let se = (x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt();
    (mean, se, if se > 0. { mean / se } else { 0. })
}
async fn execute(config: Config) -> Result<serde_json::Value, Error> {
    if config
        .walkers
        .iter()
        .chain(&config.replica_walkers)
        .any(|n| *n < 4)
        || config.steps == 0
        || config.replicas < 2
    {
        return Err("need N>=4, steps>0 and replicas>=2".into());
    }
    let mut runs = vec![];
    let mut failures = 0;
    for &n in &config.walkers {
        for &seed in &config.seeds {
            eprintln!("EH trajectory N={n} seed={seed}");
            let c = run_config(n, seed)?;
            let mut gas = c.build::<f64>().await?;
            let mut estimates = vec![];
            for _ in 0..config.steps {
                gas.start_recording(record())?;
                gas.step().await?;
                let archive = gas.stop_recording().ok_or("missing recording")?;
                let estimate =
                    analyze_step(&archive.steps[0], 0.002, 3., 0.33, &c.gas.clone_decision)?;
                failures += usize::from(!estimate.exact_checks_passed);
                estimates.push(estimate);
            }
            runs.push(Run {
                config: c,
                estimates,
            });
        }
    }
    let mut replicas = vec![];
    for &n in &config.replica_walkers {
        eprintln!(
            "EH independent future replicas N={n}, count={}",
            config.replicas
        );
        let c = run_config(n, 7)?;
        let mut gas = c.build::<f64>().await?;
        for _ in 0..config.warmup {
            gas.step().await?;
        }
        let checkpoint = gas.checkpoint();
        let mut clone = vec![];
        let mut kinetic = vec![];
        let mut variance = 0.;
        let mut passed = true;
        for k in 0..config.replicas {
            let cp = checkpoint.with_future_seed(100_000 + k as u64)?;
            let mut rc = c.clone();
            rc.gas = cp.config.clone();
            let mut replica = rc.build::<f64>().await?;
            replica.restore(cp)?;
            replica.start_recording(record())?;
            replica.step().await?;
            let archive = replica.stop_recording().ok_or("missing recording")?;
            let estimate = analyze_step(&archive.steps[0], 0.002, 3., 0.33, &c.gas.clone_decision)?;
            if !estimate.step.is_multiple_of(c.gas.clone_decision.every) {
                return Err("replica warmup must end just before an open cloning gate".into());
            }
            clone.push(estimate.clone_residual);
            kinetic.push(estimate.kinetic_residual);
            variance += estimate.kinetic_position_variance;
            passed &= estimate.exact_checks_passed;
        }
        failures += usize::from(!passed);
        let (cm, cs, cz) = stats(&clone);
        let (km, ks, kz) = stats(&kinetic);
        replicas.push(ReplicaReport {
            config: c,
            warmup: config.warmup,
            samples: config.replicas,
            first_future_seed: 100_000,
            clone_mean_residual: cm,
            clone_standard_error: cs,
            clone_standardized_residual: cz,
            kinetic_mean_residual: km,
            kinetic_standard_error: ks,
            kinetic_standardized_residual: kz,
            mean_predicted_kinetic_variance: variance / config.replicas as f64,
            exact_checks_passed: passed,
        });
    }
    Ok(
        serde_json::json!({"scope":"EH.M1--M6 and EH.K1--K6: exact native operator identities; independent future-seed replicas support conditional expectations, not a proof of long-time convergence", "configuration":config,"runs":runs,"replicas":replicas,"exact_failures":failures}),
    )
}
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "config") {
        println!("{}", serde_json::to_string_pretty(&Config::default())?);
        return Ok(());
    }
    if args.len() != 2 {
        return Err("Usage: gas-eh-proof config | CONFIG.json OUTPUT.json".into());
    }
    let config = serde_json::from_slice(&fs::read(&args[0])?)?;
    let report = futures_lite::future::block_on(execute(config))?;
    let path = Path::new(&args[1]);
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    if report["exact_failures"].as_u64().unwrap_or(1) > 0 {
        return Err("EH algebraic checks failed; inspect report".into());
    }
    Ok(())
}
