//! Saved independent native ensembles for Chapters 10--12.
use algorithmic_gas::{
    GasConfig, GasError, RecordingConfig, TensorBatch, boundary::BoundaryPolicy,
    boundary::BoxDomain, geometry::Kernel, kinetic::KineticKind,
};
use algorithmic_gas_benchmarks::{
    Benchmark, RunConfig,
    convergence_chapter08_completion::bounded_observables,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::json;
use std::{fs, path::PathBuf};

async fn run(
    path: &str,
    replicas: usize,
    horizon: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if replicas == 0 || horizon == 0 {
        return Err("positive replica count and horizon required".into());
    }
    let mut store = ArchiveStore::new(path)?;
    store.save_json("execution-source", &json!({
        "runner":include_str!("gas-entropy-native-trajectories.rs"),
        "engine":include_str!("../../../algorithmic-gas/src/engine.rs"),
        "kinetic":include_str!("../../../algorithmic-gas/src/kinetic.rs"),
        "cloning":include_str!("../../../algorithmic-gas/src/cloning.rs"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,
        "scope":"Actual native independent trajectory ensembles. Storage row indices only record unordered physical swarms. Error observables use empirical probability or subprobability normalization. Independent sides are not a same-stream coupling. Extinction is retained without survival conditioning or replacement."}))?;
    let mut cases = vec![];
    let mut updates = 0usize;
    let mut extinct = 0usize;
    let mut checks = 0usize;
    let mut failed = 0usize;
    for (profile, landscape, gamma, sigma, center, absorbing) in [
        (
            "quadratic-unbounded",
            Benchmark::Quadratic,
            1.,
            0.1,
            0.,
            false,
        ),
        ("rastrigin-well", Benchmark::Rastrigin, 1., 0.1, 0., false),
        (
            "rastrigin-saddle",
            Benchmark::Rastrigin,
            2.,
            0.1,
            0.5,
            false,
        ),
        ("rastrigin-tail", Benchmark::Rastrigin, 1., 0.1, 4.1, false),
        (
            "quadratic-boundary-revival",
            Benchmark::Quadratic,
            1.,
            0.7,
            0.,
            true,
        ),
        (
            "rastrigin-boundary-revival",
            Benchmark::Rastrigin,
            2.,
            0.7,
            0.,
            true,
        ),
    ] {
        for d in [1, 2, 4] {
            for n in [8, 32, 128] {
                let id = format!("{profile}-d{d}-N{n}");
                let mut gas = GasConfig::euclidean(d, 0.04)?;
                gas.boundary = if absorbing {
                    BoundaryPolicy::AbsorbingBox {
                        field: "positions".into(),
                        domain: BoxDomain {
                            lower: vec![-0.3; d],
                            upper: vec![0.3; d],
                        },
                    }
                } else {
                    BoundaryPolicy::Unbounded
                };
                gas.cloning_donors.kernel = Kernel::Gaussian { width: 4. };
                gas.distance_donors.kernel = Kernel::Gaussian { width: 4. };
                gas.kinetic.position_diffusion = sigma;
                gas.kinetic.velocity_cap = Some(2.);
                if let KineticKind::Baoab { friction, .. } = &mut gas.kinetic.integrator {
                    *friction = gamma;
                }
                let cfg = RunConfig {
                    benchmark: landscape,
                    walkers: n,
                    dimensions: d,
                    initial_lower: -0.1,
                    initial_upper: 0.1,
                    gas,
                    ..Default::default()
                };
                store.save_json(&format!("{id}-configuration"), &serde_json::to_value(&cfg)?)?;
                let mut trajectories = vec![];
                for side in ["left", "right"] {
                    let shift = if side == "right" {
                        if absorbing { 0.06 } else { 0.12 }
                    } else {
                        0.
                    };
                    let dead = if absorbing {
                        if side == "left" { n / 4 } else { n / 2 }
                    } else {
                        0
                    };
                    let mut base = cfg.build::<f64>().await?;
                    let mut p = base.population().clone();
                    let x = (0..n)
                        .flat_map(|i| {
                            (0..d).map(move |a| {
                                if i < dead && a == 0 {
                                    0.5
                                } else {
                                    center
                                        + shift
                                        + 0.12
                                            * (2. * ((i * 17 + a * 11) % n) as f64 / (n - 1) as f64
                                                - 1.)
                                }
                            })
                        })
                        .collect();
                    let v = (0..n)
                        .flat_map(|i| {
                            (0..d).map(move |a| {
                                0.1 * (2. * ((i * 13 + a * 7) % n) as f64 / (n - 1) as f64 - 1.)
                                    / (d as f64).sqrt()
                            })
                        })
                        .collect();
                    p.observations
                        .fields
                        .insert("positions".into(), TensorBatch::vectors(n, d, x)?);
                    p.observations
                        .fields
                        .insert("velocities".into(), TensorBatch::vectors(n, d, v)?);
                    base.replace_population(p).await?;
                    let frozen = base.population().clone();
                    let initial = bounded_observables(&frozen)?;
                    store.save_checkpoint(&format!("{id}-{side}-initial"), &base.checkpoint())?;
                    for rep in 0..replicas {
                        let seed = 202610051000000u64
                            + (cases.len() as u64) * 1_000_000
                            + if side == "right" { 500_000 } else { 0 }
                            + rep as u64;
                        let mut actual_cfg = cfg.clone();
                        actual_cfg.gas.seed = seed;
                        let mut actual = actual_cfg.build::<f64>().await?;
                        actual.replace_population(frozen.clone()).await?;
                        actual.start_recording(RecordingConfig {
                            max_steps: horizon,
                            ..Default::default()
                        })?;
                        let mut extinct_before_step = None;
                        let mut first_extinction_after_step = None;
                        for t in 0..horizon {
                            match actual.step().await {
                                Ok(_) => {
                                    updates += 1;
                                    if !actual.population().eligible(false).iter().any(|&a| a) {
                                        first_extinction_after_step = Some(t + 1);
                                        extinct += 1;
                                        break;
                                    }
                                }
                                Err(GasError::Extinction) => {
                                    extinct_before_step = Some(t + 1);
                                    extinct += 1;
                                    break;
                                }
                                Err(err) => return Err(err.into()),
                            }
                        }
                        let archive = actual.stop_recording().ok_or("missing native recording")?;
                        let terminal = bounded_observables(actual.population())?;
                        let velocities = actual.population().observations.field("velocities")?;
                        let mut max_speed = 0_f64;
                        for row in velocities.values().chunks_exact(d) {
                            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                            max_speed = max_speed.max(norm);
                            checks += 1;
                            failed += usize::from(!norm.is_finite() || norm > 2. + 1e-10);
                        }
                        for a in initial.iter().chain(terminal.iter()) {
                            checks += 1;
                            failed += usize::from(!a.is_finite() || a.abs() > 1. + 1e-10);
                        }
                        let tag = format!("{id}-rep{rep}-{side}");
                        let archive_path = store.save_archive(&tag, &archive)?;
                        let checkpoint = store
                            .save_checkpoint(&format!("{tag}-terminal"), &actual.checkpoint())?;
                        trajectories.push(json!({"tag":tag,"side":side,"replica":rep,"seed":seed,
                            "archive":archive_path,"checkpoint":checkpoint,"recorded_steps":archive.steps.len(),
                            "extinction_before_step":extinct_before_step,"initial_observables":initial,
                            "first_extinction_after_step":first_extinction_after_step,
                            "terminal_observables":terminal,"max_terminal_speed":max_speed}));
                    }
                }
                println!(
                    "{id}: {} trajectories; {updates} updates accumulated",
                    trajectories.len()
                );
                cases.push(json!({"id":id,"profile":profile,"N":n,"d":d,"h":0.04,
                    "friction":gamma,"position_diffusion":sigma,"center":center,
                    "absorbing":absorbing,"config":cfg,"trajectories":trajectories}));
            }
        }
    }
    let report = json!({"chapters":[10,11,12],"cases":cases,
        "summary":{"cases":cases.len(),"trajectories":cases.len()*2*replicas,
            "independent_replicas_per_side":replicas,"requested_steps":horizon,
            "new_complete_native_updates":updates,"extinctions":extinct,
            "comparisons":checks,"comparisons_failed":failed},
        "normalization":"Empirical probabilities and alive subprobabilities divide by N. Complete alive conditional probability comparisons require positive mass; extinction records are preserved.",
        "scope":"Recorded native trajectories provide observed finite-time errors; reference continuous entropy and global native QSD/LSI statements require their explicit analytic hypotheses."});
    store.save_json("completion-report", &report)?;
    store.finish("complete")?;
    let verification = store.verify(true)?;
    fs::write(
        PathBuf::from(path).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        PathBuf::from(path).join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!("{}", report["summary"]);
    if failed > 0 {
        return Err("native sanity checks failed; full data retained".into());
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 3 {
        return Err("gas-entropy-native-trajectories EMPTY_OUTPUT [REPLICAS=24] [STEPS=32]".into());
    }
    let replicas = args.get(1).map(|a| a.parse()).transpose()?.unwrap_or(24);
    let steps = args.get(2).map(|a| a.parse()).transpose()?.unwrap_or(32);
    futures_lite::future::block_on(run(&args[0], replicas, steps))
}
