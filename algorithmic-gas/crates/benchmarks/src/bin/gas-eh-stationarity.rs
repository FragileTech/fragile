//! Long native EH trajectories with signed cycle budgets and empirical snapshots.
use algorithmic_gas::{Precision, RecordingConfig, tracking::RecordedStep};
use algorithmic_gas_benchmarks::{
    RunConfig,
    convergence_einstein_hilbert::{analyze_step, mean},
};
use serde::Serialize;
use std::{fs, path::Path};
type Error = Box<dyn std::error::Error>;

#[derive(Default, Serialize)]
struct Cycle {
    step: u64,
    entering_spread: f64,
    final_spread: f64,
    final_energy: f64,
    cloning_drift: f64,
    transport_drift: f64,
    ballistic_drift: f64,
    thermal_drift: f64,
    cloning_residual: f64,
    kinetic_residual: f64,
    residual_variance_bound: f64,
    clones: usize,
}

#[derive(Serialize)]
struct Snapshot {
    step: u64,
    centered_positions: Vec<f64>,
    velocities: Vec<f64>,
    final_curvature: Vec<f64>,
    final_volume: Vec<f64>,
    final_rewards: Vec<f64>,
    consumed_graph_degree: Vec<usize>,
    consumed_graph_geodesic_length: Vec<f64>,
}
fn snapshot(r: &RecordedStep<f64>) -> Result<Snapshot, Error> {
    let fields = &r.final_population.observations.fields;
    let get = |name: &str| -> Result<Vec<f64>, Error> {
        Ok(fields
            .get(name)
            .ok_or_else(|| format!("missing {name}"))?
            .values()
            .to_vec())
    };
    let mut x = get("positions")?;
    let mx = mean(&x, 3);
    for row in x.chunks_exact_mut(3) {
        for a in 0..3 {
            row[a] -= mx[a];
        }
    }
    let graph = r.graph.as_ref().ok_or("missing graph")?;
    Ok(Snapshot {
        step: r.report.step,
        centered_positions: x,
        velocities: get("velocities")?,
        final_curvature: get("geometry.curvature.ricci_scalar")?,
        final_volume: get("geometry.volume_element")?,
        final_rewards: r.report.final_rewards.raw.clone(),
        consumed_graph_degree: (0..graph.graph.nodes())
            .map(|i| graph.graph.range(i).len())
            .collect(),
        consumed_graph_geodesic_length: graph.geodesic_length.clone(),
    })
}

async fn run(n: usize, seed: u64, steps: usize, stride: usize, path: &Path) -> Result<(), Error> {
    if n < 4 || steps == 0 || !steps.is_multiple_of(20) || stride == 0 || !stride.is_multiple_of(20)
    {
        return Err("need N>=4 and positive steps/stride divisible by 20".into());
    }
    let mut config = RunConfig::einstein_hilbert()?;
    config.walkers = n;
    config.gas.seed = seed;
    config.gas.precision = Precision::F64;
    let mut gas = config.build::<f64>().await?;
    let (mut cycles, mut snapshots) = (vec![], vec![]);
    let (mut failures, mut max_residual) = (0, 0_f64);
    let mut cycle = Cycle::default();
    let b = 0.001 * (1. + (-0.002_f64).exp());
    let thermal = 3. * (1. - 1. / n as f64) * 1e-6 * 0.33 * (1. - (-0.004_f64).exp());
    for k in 1..=steps {
        gas.start_recording(RecordingConfig {
            graph: true,
            ..Default::default()
        })?;
        gas.step().await?;
        let archive = gas.stop_recording().ok_or("missing recording")?;
        let r = &archive.steps[0];
        let e = analyze_step(r, 0.002, 3., 0.33, &config.gas.clone_decision)?;
        failures += usize::from(!e.exact_checks_passed);
        max_residual = max_residual.max(e.exact_max_relative_residual);
        if k % 20 == 1 {
            cycle.entering_spread = e.cloning.entering_variance;
        }
        cycle.cloning_drift += e.cloning.expected_variance - e.cloning.entering_variance;
        cycle.transport_drift += 2. * b * e.b1_position_velocity_covariance;
        cycle.ballistic_drift += b * b * e.b1_velocity_variance;
        cycle.thermal_drift += thermal;
        cycle.cloning_residual += e.clone_residual;
        cycle.kinetic_residual += e.kinetic_residual;
        cycle.residual_variance_bound +=
            e.cloning.spread_variance_bound + e.kinetic_position_variance;
        cycle.clones += r.report.clones;
        if k % 20 == 0 {
            cycle.step = e.step;
            cycle.final_spread = e.final_position_variance;
            cycle.final_energy = e.final_velocity_energy;
            let accounted = cycle.cloning_drift
                + cycle.transport_drift
                + cycle.ballistic_drift
                + cycle.thermal_drift
                + cycle.cloning_residual
                + cycle.kinetic_residual;
            let actual = cycle.final_spread - cycle.entering_spread;
            let residual = (actual - accounted).abs() / (1. + actual.abs() + accounted.abs());
            failures += usize::from(residual > 2e-10);
            max_residual = max_residual.max(residual);
            cycles.push(cycle);
            cycle = Cycle::default();
        }
        if k.is_multiple_of(stride) {
            snapshots.push(snapshot(r)?);
        }
        if k.is_multiple_of(5000) || k == steps {
            eprintln!(
                "EH N={n} seed={seed} step={k}/{steps} W={:.6} E={:.6}",
                e.final_position_variance, e.final_velocity_energy
            );
        }
    }
    let report = serde_json::json!({
        "scope": "Native reference EH, unscaled centered positions, phase-20 budgets; snapshots are dependent time samples, not independent replicas.",
        "config": config, "steps": steps, "stride": stride,
        "exact_failures": failures, "max_relative_identity_residual": max_residual,
        "cycles": cycles, "snapshots": snapshots,
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec(&report)?)?;
    if failures > 0 {
        return Err("native identity check failed".into());
    }
    Ok(())
}
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("Usage: gas-eh-stationarity N SEED STEPS STRIDE OUTPUT.json".into());
    }
    futures_lite::future::block_on(run(
        args[0].parse()?,
        args[1].parse()?,
        args[2].parse()?,
        args[3].parse()?,
        Path::new(&args[4]),
    ))
}
