//! Native resonant BAOAB experiment; preserve the full prepared state and raw noises.
use algorithmic_gas::{
    GasConfig, RecordingConfig, TensorBatch,
    boundary::BoundaryPolicy,
    geometry::Kernel,
    noise::{FactorValues, NoiseGeometry},
    tracking::RecordedStep,
};
use algorithmic_gas_benchmarks::{
    Benchmark, RunConfig,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::json;
use std::{fs, path::PathBuf};

fn stage<'a>(
    s: &'a RecordedStep<f64>,
    name: &str,
    field: &str,
) -> Result<&'a [f64], Box<dyn std::error::Error>> {
    Ok(&s
        .stages
        .iter()
        .find(|a| a.stage == name)
        .ok_or("missing native stage")?
        .fields
        .get(field)
        .ok_or("missing native field")?
        .values)
}
async fn run(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = ArchiveStore::new(path)?;
    store.save_json("execution-source", &json!({"runner":include_str!("gas-native-kinetic-resonance.rs"),
        "native_kinetic":include_str!("../../../algorithmic-gas/src/kinetic.rs"),
        "chapter":include_str!("../../../../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,
        "scope":"Exact native quadratic h=2 resonant conditional velocity identity, not a global mixing experiment. Full marked population, probability-normalized errors."}))?;
    let mut cases = vec![];
    let mut updates = 0usize;
    let mut checks = 0usize;
    let mut failed = 0usize;
    for d in [1, 2, 4] {
        for n in [8, 32] {
            for speed in [0., 0.2] {
                for ou_scale in [0.1, 1., 3.] {
                    for sigma_x in [0.1, 0.7] {
                        let id = format!(
                            "resonance-d{d}-N{n}-speed{speed}-ou{ou_scale}-position{sigma_x}"
                        );
                        let mut gas = GasConfig::euclidean(d, 2.)?;
                        gas.boundary = BoundaryPolicy::Unbounded;
                        gas.cloning_donors.kernel = Kernel::Gaussian { width: 4. };
                        gas.distance_donors.kernel = Kernel::Gaussian { width: 4. };
                        gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
                            scale: FactorValues::Constant {
                                values: vec![ou_scale],
                            },
                        };
                        gas.kinetic.position_diffusion = sigma_x;
                        let cfg = RunConfig {
                            benchmark: Benchmark::Quadratic,
                            walkers: n,
                            dimensions: d,
                            initial_lower: -0.3,
                            initial_upper: 0.3,
                            gas,
                            ..Default::default()
                        };
                        let mut base = cfg.build::<f64>().await?;
                        let mut p = base.population().clone();
                        let x = (0..n)
                            .flat_map(|i| {
                                (0..d).map(move |a| {
                                    0.2 * ((i * 17 + a * 11) % n) as f64 / (n - 1) as f64 - 0.1
                                })
                            })
                            .collect();
                        p.observations
                            .fields
                            .insert("positions".into(), TensorBatch::vectors(n, d, x)?);
                        p.observations.fields.insert(
                            "velocities".into(),
                            TensorBatch::vectors(n, d, vec![speed / (d as f64).sqrt(); n * d])?,
                        );
                        base.replace_population(p).await?;
                        let frozen = base.population().clone();
                        store.save_checkpoint(&format!("{id}-frozen"), &base.checkpoint())?;
                        let mut rows = vec![];
                        let mut maximum_identity_error = 0_f64;
                        let mut maximum_cap_error = 0_f64;
                        let mut expected_speed_sum = 0.;
                        let mut measured_speed_sum = 0.;
                        for rep in 0..16 {
                            let mut actual_cfg = cfg.clone();
                            actual_cfg.gas.seed =
                                202610049700 + updates as u64 * 10000 + rep as u64;
                            let mut actual = actual_cfg.build::<f64>().await?;
                            actual.replace_population(frozen.clone()).await?;
                            actual.start_recording(RecordingConfig {
                                max_steps: 1,
                                ..Default::default()
                            })?;
                            actual.step().await?;
                            let archive = actual.stop_recording().ok_or("missing recording")?;
                            let s = &archive.steps[0];
                            let prepared = stage(s, "B1_input", "velocities")?;
                            let uncapped = stage(s, "B2", "velocities")?;
                            let out = s
                                .final_population
                                .observations
                                .field("velocities")?
                                .values();
                            let archive_path =
                                store.save_archive(&format!("{id}-rep{rep}"), &archive)?;
                            for i in 0..n {
                                let v = &prepared[i * d..(i + 1) * d];
                                let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                                let factor = 2. / (2. + norm);
                                let observed_norm = out[i * d..(i + 1) * d]
                                    .iter()
                                    .map(|a| a * a)
                                    .sum::<f64>()
                                    .sqrt();
                                measured_speed_sum += observed_norm;
                                expected_speed_sum += factor * norm;
                                for a in 0..d {
                                    let identity_error = (uncapped[i * d + a] + v[a]).abs();
                                    let cap_error = (out[i * d + a] + factor * v[a]).abs();
                                    let allowance =
                                        1e-12 * (1. + v[a].abs() + uncapped[i * d + a].abs());
                                    maximum_identity_error =
                                        maximum_identity_error.max(identity_error);
                                    maximum_cap_error = maximum_cap_error.max(cap_error);
                                    checks += 2;
                                    failed += usize::from(identity_error > allowance)
                                        + usize::from(cap_error > allowance);
                                    rows.push(json!({"storage_row":i,"coordinate":a,"prepared_velocity":v[a],
                            "uncapped_output":uncapped[i*d+a],"physical_capped_output":out[i*d+a],
                            "expected_capped_output":-factor*v[a],"identity_error":identity_error,
                            "cap_error":cap_error,"allowance":allowance,"archive":archive_path,"replica":rep}));
                                }
                            }
                            updates += 1;
                        }
                        let raw = store.save_json(
                            &format!("{id}-resonance-operands"),
                            &json!({"rows":rows}),
                        )?;
                        cases.push(json!({"id":id,"N":n,"d":d,"h":2.,"omega":1.,"ou_noise_scale":ou_scale,
                "position_diffusion":sigma_x,"input_speed":speed,"independent_native_replicas":16,
                "mean_predicted_output_speed":expected_speed_sum/(16*n) as f64,
                "mean_measured_output_speed":measured_speed_sum/(16*n) as f64,
                "maximum_identity_error":maximum_identity_error,"maximum_cap_error":maximum_cap_error,
                "raw_operands":raw,"source_label":"lem-chaos-kinetic-memory-observable"}));
                    }
                }
            }
        }
    }
    let report = json!({"chapter":9,"cases":cases,"summary":{"cases":cases.len(),
        "new_complete_native_updates":updates,"comparisons":checks,"comparisons_failed":failed},
        "scope":"Native exact quadratic resonant identity at h=2: uncapped terminal velocity = -prepared V for every OU scale and final position diffusion. Full records include sampled fitness, cloning, components, jitter, force stages and innovations. Analytic singular-support obstruction requires the chapter's real-arithmetic argument; floating-point zeros are not claimed to prove a TV lower bound."});
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
        return Err("native resonance comparisons failed; raw evidence retained".into());
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("gas-native-kinetic-resonance EMPTY_OUTPUT".into());
    }
    futures_lite::future::block_on(run(&args[0]))
}
