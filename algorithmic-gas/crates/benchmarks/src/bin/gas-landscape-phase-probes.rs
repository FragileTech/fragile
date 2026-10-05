//! Native within-core KUR probes with positive sampled cloning and both viscosities.
#![recursion_limit = "256"]
use algorithmic_gas::{
    GasBuilder, GasConfig, ObservationBatch, Population, RecordingConfig, TensorBatch,
    boundary::BoundaryPolicy, kinetic::ViscousForceConfig, tracking::RecordedStep,
};
use algorithmic_gas_benchmarks::{
    Benchmark, BenchmarkModel,
    convergence_experiments::{ArchiveStore, sha256_file},
    convergence_landscape_phase as phase,
};
use serde_json::{Value, json};
use std::{fs, path::Path};
type Error = Box<dyn std::error::Error>;
fn population(n: usize, d: usize) -> algorithmic_gas::Result<Population<f64>> {
    let x = (0..n)
        .flat_map(|i| (0..d).map(move |j| 0.01 * ((i * 7 + j * 3) % 11) as f64 / 5. - 0.01))
        .collect();
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, d, x)?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, d, vec![0.; n * d])?,
    );
    Population::new(obs)
}
fn stage<'a>(r: &'a RecordedStep<f64>, name: &str, field: &str) -> Result<&'a [f64], Error> {
    Ok(&r
        .stages
        .iter()
        .find(|s| s.stage == name)
        .ok_or("Missing complete stage")?
        .fields
        .get(field)
        .ok_or("Missing stage field")?
        .values)
}
fn variance(x: &[f64], n: usize, d: usize) -> f64 {
    let means: Vec<_> = (0..d)
        .map(|j| x.chunks(d).map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    x.chunks(d)
        .map(|r| {
            r.iter()
                .zip(&means)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64
}
fn estimate(x: &[f64]) -> (f64, f64) {
    let m = x.len() as f64;
    let mean = x.iter().sum::<f64>() / m;
    let se = (x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m - 1.) / m).sqrt();
    (mean, se)
}
fn exact_conditional_position(
    mu: &[f64],
    accepted: &[bool],
    n: usize,
    d: usize,
    p: &phase::RegionalParameters,
) -> f64 {
    let t = p.timestep / 2.;
    let eta = t * t * (1. + (-p.friction * p.timestep).exp());
    let ell = 1. - 2. * eta;
    let amp = 20. * std::f64::consts::PI * eta;
    let omega = std::f64::consts::TAU;
    let mut means = vec![];
    let mut varsum = 0.;
    for (i, row) in mu.chunks(d).enumerate() {
        let sigma = if accepted[i] { p.clone_jitter } else { 0. };
        let s2 = sigma * sigma;
        let a = (-omega * omega * s2 / 2.).exp();
        let b = (-2. * omega * omega * s2).exp();
        for &x in row {
            let sin = (omega * x).sin();
            let cos = (omega * x).cos();
            means.push(ell * x - amp * a * sin);
            if sigma > 0. {
                varsum += ell * ell * s2 - 2. * ell * amp * omega * s2 * a * cos
                    + amp * amp * ((1. - b * (2. * omega * x).cos()) / 2. - a * a * sin * sin);
            }
        }
    }
    variance(&means, n, d)
        + (1. - 1. / n as f64) * varsum / n as f64
        + (1. - 1. / n as f64)
            * d as f64
            * (t * t * p.ou_amplitude * p.ou_amplitude
                + p.position_amplitude * p.position_amplitude)
}
async fn run(store: &mut ArchiveStore, samples: usize) -> Result<Value, Error> {
    let mut cases = vec![];
    let mut comparisons = vec![];
    for d in [1, 2, 3, 4, 8] {
        for n in [4, 64] {
            for row in [false, true] {
                for jitter in [0., 0.1] {
                    let tag = format!("d{d}_n{n}_{}_j{jitter}", if row { "row" } else { "count" });
                    eprintln!("KUR native {tag}: {samples} complete independent replicas");
                    let mut observations = vec![];
                    let mut paths = vec![];
                    let mut config = GasConfig::euclidean(d, 0.04)?;
                    config.boundary = BoundaryPolicy::Unbounded;
                    config.clone_transform.jitter_amplitude = jitter;
                    config.kinetic.velocity_cap = Some(2.);
                    config.kinetic.position_diffusion = 0.1;
                    config.qft.viscosity = Some(ViscousForceConfig {
                        coefficient: 0.3,
                        bandwidth: 1.,
                        row_normalized: row,
                    });
                    let params = phase::native_parameters(&serde_json::to_value(&config)?, d)?;
                    let old = phase::regional_bound(&params)?;
                    let new = phase::tightened_regional_bound(&params)?;
                    if old["applicable"] != true || new["applicable"] != true {
                        return Err(
                            "Native probe parameters must satisfy the proved regional premises"
                                .into(),
                        );
                    }
                    let t = params.timestep / 2.;
                    let b = t * (1. + (-params.friction * params.timestep).exp());
                    let vc = (1. + 2. * params.restitution.abs()) * params.velocity_cap;
                    let bound_for = |c: &Value, state: bool| {
                        let floor = c["conservative_floor"].as_f64().unwrap();
                        let eps = c["epsilon"].as_f64().unwrap();
                        if state {
                            floor - (1. + 1. / eps) * b * b * vc * vc
                        } else {
                            floor
                        }
                    };
                    let lambda_old = old["positional_coefficient"].as_f64().unwrap();
                    let lambda_new = new["positional_coefficient"].as_f64().unwrap();
                    for rep in 0..samples {
                        config.seed = 0x81000000
                            + d as u64 * 1_000_000
                            + n as u64 * 1000
                            + u64::from(row) * 100
                            + u64::from(jitter > 0.) * 10_000
                            + rep as u64;
                        let initial = population(n, d)?;
                        if phase::source_core_eligibility(
                            initial.observations.field("positions")?.values(),
                            d,
                            &params,
                        )?["all_sources_in_one_core"]
                            != true
                        {
                            return Err("Unconditional probe source-core premise failed".into());
                        }
                        let model = BenchmarkModel {
                            benchmark: Benchmark::Rastrigin,
                            field: "positions".into(),
                            direction: config.fitness.direction,
                        };
                        let mut gas = GasBuilder::new(initial, model.clone())
                            .gradient(model)
                            .config(config.clone())
                            .build()
                            .await?;
                        gas.start_recording(RecordingConfig {
                            max_steps: 2,
                            max_bytes: 16 * 1024 * 1024,
                            graph: false,
                        })?;
                        let report = gas.step().await?;
                        let archive = gas.stop_recording().ok_or("Missing full native archive")?;
                        let recorded = &archive.steps[0];
                        let mu = stage(recorded, "literal_clone", "positions")?;
                        let out = stage(recorded, "terminal", "positions")?;
                        if stage(recorded, "B1_input", "velocities")?
                            .iter()
                            .any(|v| *v != 0.)
                            || report.revivals != 0
                        {
                            return Err(
                                "State-aware zero prepared velocity/no-revival premise failed"
                                    .into(),
                            );
                        }
                        let accepted: Vec<_> = report
                            .clone_plan
                            .choices
                            .iter()
                            .map(|c| c.accepted)
                            .collect();
                        let source_var = variance(mu, n, d);
                        let output_var = variance(out, n, d);
                        let exact = exact_conditional_position(mu, &accepted, n, d, &params);
                        let a = store
                            .save_archive(&format!("kur/{tag}/rep{rep}/left_through1"), &archive)?;
                        let checkpoint = store.save_checkpoint(
                            &format!("kur/{tag}/rep{rep}/checkpoint1"),
                            &gas.checkpoint(),
                        )?;
                        observations.push(json!({"replicate":rep,"seed":config.seed,"zero_jitter_source_variance":source_var,"output_positional_variance":output_var,"exact_conditional_position_variance":exact,"accepted_clones":report.clones,"accepted_plan":accepted,"archive":a,"checkpoint":checkpoint}));
                        paths.push(a);
                    }
                    for (name, c, lambda) in [
                        ("original", &old, lambda_old),
                        ("refined", &new, lambda_new),
                    ] {
                        let residuals: Vec<_> = observations
                            .iter()
                            .map(|o| {
                                o["output_positional_variance"].as_f64().unwrap()
                                    - lambda * o["zero_jitter_source_variance"].as_f64().unwrap()
                            })
                            .collect();
                        let (mean, se) = estimate(&residuals);
                        for state in [false, true] {
                            let bound = bound_for(c, state);
                            comparisons.push(json!({"case":tag,"variant":name,"state_aware_zero_velocity":state,"observed":mean,"bound":bound,"standard_error":se,"relation":"less_equal","passed":mean<=bound+6.*se+2e-12,"scope":"Complete independent native replicas; actual sampled zero-jitter source variance retained. All frozen eligible sources are in one core. State-aware case uses prepared collision velocities exactly zero, with configured cap2/viscosity.3 unchanged."}));
                        }
                    }
                    let residuals: Vec<_> = observations
                        .iter()
                        .map(|o| {
                            o["output_positional_variance"].as_f64().unwrap()
                                - o["exact_conditional_position_variance"].as_f64().unwrap()
                        })
                        .collect();
                    let (mean, se) = estimate(&residuals);
                    comparisons.push(json!({"case":tag,"variant":"exact_KUL8_KUL9_complete_position","observed":mean,"bound":0.,"standard_error":se,"relation":"equal","passed":mean.abs()<=6.*se+2e-12,"scope":"Exact unrestricted Gaussian recipient-jitter moments conditional on actual accepted source plan, plus native OU/final-position centered Gaussian variance. Both viscosities operate with prepared velocity0; B2/cap do not alter completed positions."}));
                    let plan=store.save_json(&format!("kur/{tag}/complete_observations"),&json!({"parameters":params,"original":old,"refined":new,"observations":observations,"archives":paths,"population_normalization":"W_N=N^-1sum_i|x_i-mean(x)|²; no particle labels"}))?;
                    cases.push(json!({"case":tag,"N":n,"d":d,"normalization":params.normalization,"jitter":jitter,"samples":samples,"observations":plan,"original":old,"refined":new,"original_state_aware_floor":bound_for(&old,true),"refined_state_aware_floor":bound_for(&new,true),"mean_accepted_clones":observations.iter().map(|o|o["accepted_clones"].as_u64().unwrap() as f64).sum::<f64>()/samples as f64}));
                }
            }
        }
    }
    let failed = comparisons.iter().filter(|c| c["passed"] != true).count();
    Ok(
        json!({"chapter":5,"cases":cases,"comparisons":comparisons,"summary":{"cases":cases.len(),"native_steps":cases.len()*samples,"comparisons":comparisons.len(),"comparisons_failed":failed},"scope":"Unchanged native positive-fitness Rastrigin cloning/Haar/two-viscosity BAOAB with full Gaussian jitter and terminal diffusion, unbounded domain; one-step within-core source-variance and exact conditional Gaussian moment checks. No residence/whole-law iteration is inferred."}),
    )
}
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 2 {
        return Err("gas-landscape-phase-probes EMPTY_OUTPUT_DIRECTORY [SAMPLES]".into());
    }
    let samples = if args.len() == 2 {
        args[1].parse()?
    } else {
        32
    };
    if samples < 8 {
        return Err("Use at least8 independent replicas".into());
    }
    let mut store = ArchiveStore::new(&args[0])?;
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let mut sources = vec![];
    for relative in [
        "docs/source/2_fractal_gas/convergence_program/18a_keystone_uniform_coupled.md",
        "docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md",
        "algorithmic-gas/crates/benchmarks/src/bin/gas-landscape-phase-probes.rs",
        "algorithmic-gas/crates/benchmarks/src/convergence_landscape_phase.rs",
        "algorithmic-gas/crates/algorithmic-gas/src/kinetic.rs",
        "algorithmic-gas/crates/algorithmic-gas/src/cloning.rs",
        "algorithmic-gas/crates/algorithmic-gas/src/engine.rs",
        "algorithmic-gas/crates/algorithmic-gas/src/fitness.rs",
        "algorithmic-gas/crates/benchmarks/src/lib.rs",
    ] {
        let path = workspace.join(relative);
        sources.push(
            json!({"path":relative,"sha256":sha256_file(&path)?,"text":fs::read_to_string(path)?}),
        );
    }
    let provenance=store.save_json("kur/provenance",&json!({"sources":sources,"executable_sha256":sha256_file(&std::env::current_exe()?)?,"command":std::env::args().collect::<Vec<_>>(),"samples":samples,"Cargo_lock":fs::read_to_string(workspace.join("algorithmic-gas/Cargo.lock"))?}))?;
    let mut report = futures_lite::future::block_on(run(&mut store, samples))?;
    report["provenance_archive"] = json!(provenance);
    fs::write(
        store.root().join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    store.save_json("kur/report", &report)?;
    let failed = report["summary"]["comparisons_failed"].as_u64().unwrap();
    store.finish(if failed == 0 {
        "completed"
    } else {
        "discrepancy"
    })?;
    let verification = store.verify(true)?;
    fs::write(
        store.root().join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!("{}", report["summary"]);
    if failed > 0 {
        return Err("Native KUR discrepancies retained".into());
    }
    Ok(())
}
