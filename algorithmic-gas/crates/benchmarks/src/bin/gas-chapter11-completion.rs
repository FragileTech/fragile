//! Chapter 11 exact identities, reference rates and native alive-law operands.
use algorithmic_gas_benchmarks::{
    convergence_chapter08_completion::normal_cdf,
    convergence_chapter11_completion::{
        hellinger_squared, poisson_binomial, reference_suite, wasserstein_1d_squared,
    },
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs};

const CUTS: [f64; 11] = [-4., -2., -1., -0.5, -0.25, 0., 0.25, 0.5, 1., 2., 4.];
fn observation(p: &algorithmic_gas::Population<f64>) -> Result<Value, Box<dyn std::error::Error>> {
    let field = p.observations.field("positions")?;
    let d = field.width();
    let alive = p.eligible(false);
    let mut bins = vec![0.; 12];
    let mut coordinates = vec![];
    for (i, &is_alive) in alive.iter().enumerate() {
        if is_alive {
            let x = field.values()[i * d];
            coordinates.push(x);
            let bin = CUTS.partition_point(|cut| x >= *cut);
            bins[bin] += 1. / p.len() as f64;
        }
    }
    coordinates.sort_by(f64::total_cmp);
    let m = coordinates.len() as f64 / p.len() as f64;
    Ok(
        json!({"N":p.len(),"d":d,"mass":m,"alive_count":coordinates.len(),"bins":bins,"projected_coordinates":coordinates}),
    )
}
fn floats(v: &Value) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    v.as_array()
        .ok_or("array missing")?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| "float missing".into()))
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "gas-chapter11-completion EMPTY_OUTPUT COMPLETE_NATIVE_DATASET [DATASET ...]".into(),
        );
    }
    let mut out = ArchiveStore::new(&args[0])?;
    out.save_json("source",&json!({"chapter":include_str!("../../../../../docs/source/2_fractal_gas/convergence_program/11_hk_convergence.md"),"inventory":include_str!("../../../../proof-validation/chapter11_inventory.json"),"module":include_str!("../convergence_chapter11_completion.rs"),"runner":include_str!("gas-chapter11-completion.rs"),"binary_sha256":sha256_file(&std::env::current_exe()?)?,"new_native_updates":0,"grid_cuts":CUTS,"native_scope":"Actual independently seeded complete native trajectories. Hellinger compares the declared common-grid first-coordinate pushforwards; W2 is exact transport of normalized first-coordinate alive empirical laws. Neither identifies true ambient entropy or native QSD convergence."}))?;
    let references = reference_suite()?;
    let reference_failures = references.iter().filter(|r| r["passed"] != true).count();
    out.save_json("reference-operands", &json!({"checks":references}))?;
    let (mut trajectories, mut updates, mut checks, mut failures) =
        (0_usize, 0_usize, 0_usize, 0_usize);
    let mut provenance = vec![];
    let mut seen = BTreeSet::new();
    let mut native_paths = vec![];
    let mut survival_batches = vec![];
    let mut max_identity = 0_f64;
    for root in &args[1..] {
        let input = ArchiveStore::open(root)?;
        if input.status() != "complete" && input.status() != "completed" {
            return Err("complete immutable input required".into());
        }
        let dataset = input
            .root()
            .file_name()
            .ok_or("dataset missing")?
            .to_string_lossy()
            .to_string();
        provenance.push(json!({"dataset":root,"index_sha256":sha256_file(&input.root().join("archive-index.json"))?}));
        for entry in input
            .entries()
            .iter()
            .filter(|e| e.kind == "native_run_archive")
        {
            let archive = input.load_archive(&entry.path)?;
            if archive.steps.is_empty() {
                return Err("native trajectory empty".into());
            }
            let cfg = serde_json::to_value(&archive.gas_config)?;
            let h = cfg["kinetic"]["integrator"]["dt"]
                .as_f64()
                .ok_or("physical dt missing")?;
            let (case, suffix) = entry
                .tag
                .split_once("-rep")
                .ok_or("independent replicate tag missing")?;
            let side = if suffix.contains("-left") {
                "left"
            } else if suffix.contains("-right") {
                "right"
            } else {
                "single"
            };
            let signature = format!("{dataset}/{case}/{side}");
            if !seen.insert((signature.clone(), archive.gas_config.seed)) {
                return Err("duplicate independent seed in own law".into());
            }
            let initial = observation(&archive.steps[0].before)?;
            let mut observations = vec![];
            let n = archive.steps[0].before.len();
            let d = initial["d"].as_u64().unwrap() as usize;
            let mut survival = vec![];
            for record in &archive.steps {
                let observed = observation(&record.final_population)?;
                let old = floats(&initial["bins"])?;
                let new = floats(&observed["bins"])?;
                let m = initial["mass"].as_f64().unwrap();
                let k = observed["mass"].as_f64().unwrap();
                let raw_h = hellinger_squared(&old, &new)?;
                let mut value = json!({"step":record.report.step,"physical_time":record.report.step as f64*h,"observation":observed,"grid_H_squared_to_initial":raw_h});
                if m > 0. && k > 0. {
                    let p: Vec<_> = old.iter().map(|v| v / m).collect();
                    let q: Vec<_> = new.iter().map(|v| v / k).collect();
                    let shape = hellinger_squared(&p, &q)?;
                    let identity = (m.sqrt() - k.sqrt()).powi(2) + (m * k).sqrt() * shape;
                    let residual = (raw_h - identity).abs();
                    max_identity = max_identity.max(residual);
                    checks += 1;
                    failures += usize::from(residual > 2e-10);
                    let x = floats(&initial["projected_coordinates"])?;
                    let y = floats(&observed["projected_coordinates"])?;
                    let w = wasserstein_1d_squared(
                        &x,
                        &vec![1. / x.len() as f64; x.len()],
                        &y,
                        &vec![1. / y.len() as f64; y.len()],
                    )?;
                    value["grid_shape_H_squared_to_initial"] = json!(shape);
                    value["projected_W2_squared_to_initial"] = json!(w);
                    value["grid_mass_shape_identity_residual"] = json!(residual);
                }
                if cfg["kinetic"]["boundary_schedule"] == "end_of_step"
                    && cfg["boundary"]["kind"] == "absorbing_box"
                    && let Some(pre) = record.stages.iter().find(|s| s.stage == "B2")
                {
                    let sigma = cfg["kinetic"]["position_diffusion"]
                        .as_f64()
                        .ok_or("position noise missing")?
                        * h.sqrt();
                    let lower = floats(&cfg["boundary"]["domain"]["lower"])?;
                    let upper = floats(&cfg["boundary"]["domain"]["upper"])?;
                    let centers = &pre.fields["positions"].values;
                    if sigma > 0. && pre.validity.iter().all(|v| v.eligible(false)) {
                        let probabilities: Vec<_> = centers
                            .chunks(d)
                            .map(|center| {
                                center
                                    .iter()
                                    .enumerate()
                                    .map(|(axis, x)| {
                                        let lo = lower[axis % lower.len()];
                                        let hi = upper[axis % upper.len()];
                                        (normal_cdf((hi - x) / sigma)
                                            - normal_cdf((lo - x) / sigma))
                                        .clamp(0., 1.)
                                    })
                                    .product::<f64>()
                            })
                            .collect();
                        let mean = probabilities.iter().sum::<f64>() / n as f64;
                        let variance = probabilities.iter().map(|p| p * (1. - p)).sum::<f64>()
                            / (n * n) as f64;
                        let law = poisson_binomial(&probabilities)?;
                        let b = mean / 2.;
                        let exact_tail = law
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (*i as f64 / n as f64) < b)
                            .map(|(_, p)| p)
                            .sum::<f64>();
                        let bound = (-2. * n as f64 * (mean - b).powi(2)).exp();
                        let actual = observed["mass"].as_f64().unwrap();
                        let tolerance = d as f64 * n as f64 * 1.5e-7 + 1e-11;
                        checks += 3;
                        failures += usize::from(variance > 1. / (4. * n as f64) + 1e-14)
                            + usize::from(exact_tail > bound + tolerance)
                            + usize::from((law.iter().sum::<f64>() - 1.).abs() > 1e-10);
                        survival.push(json!({"step":record.report.step,"h":h,"N":n,"d":d,"centers_B2":centers,"sigma":sigma,"lower":lower,"upper":upper,"probabilities":probabilities,"poisson_binomial_law":law,"conditional_mean":mean,"conditional_variance":variance,"variance_upper":1./(4.*n as f64),"lower_mass_threshold":b,"lower_tail_probability":exact_tail,"Hoeffding_upper":bound,"actual_alive_fraction":actual,"centered_mass_innovation":actual-mean,"normal_CDF_absolute_error_allowance":tolerance,"conditioning":"complete post-B2 state and all earlier shared/native choices; final Gaussian position innovations are independent by row"}));
                    }
                }
                // Whole kinetic-stage conditional law after successful revival/collision.
                // The first force is evaluated at the fixed prepared state; the second
                // force and radial cap do not change position. No empirical Gaussian fit.
                if cfg["kinetic"]["boundary_schedule"] == "end_of_step"
                    && cfg["kinetic"]["integrator"]["kind"] == "baoab"
                    && cfg["kinetic"]["noise"]["innovation"] == "gaussian"
                    && cfg["kinetic"]["noise"]["geometry"]["kind"] == "isotropic"
                    && cfg["kinetic"]["noise"]["geometry"]["scale"]["kind"] == "constant"
                    && cfg["geometry"].is_null()
                    && cfg["qft"]["viscosity"].is_null()
                    && cfg["qft"]["graph_viscosity"].is_null()
                    && cfg["qft"]["curl"].is_null()
                    && cfg["qft"]["innovation_shifts"]
                        .as_array()
                        .is_some_and(|v| v.is_empty())
                    && (cfg["boundary"]["kind"] == "absorbing_box"
                        || cfg["boundary"]["kind"] == "unbounded")
                    && let Some(pre) = record.stages.iter().find(|stage| stage.stage == "B1_input")
                    && pre.validity.iter().all(|v| v.eligible(false))
                {
                    let gamma = cfg["kinetic"]["integrator"]["friction"]
                        .as_f64()
                        .ok_or("friction missing")?;
                    let decay = (-gamma * h).exp();
                    let variance_time = if gamma == 0. {
                        h
                    } else {
                        -(-2. * gamma * h).exp_m1() / (2. * gamma)
                    };
                    let scale = variance_time.sqrt();
                    let noise_factor = cfg["kinetic"]["noise"]["geometry"]["scale"]["values"][0]
                        .as_f64()
                        .ok_or("OU factor missing")?;
                    let sigma_x = cfg["kinetic"]["position_diffusion"].as_f64().unwrap_or(0.);
                    let sigma = (h * h / 4. * noise_factor * noise_factor * variance_time
                        + h * sigma_x * sigma_x)
                        .sqrt();
                    let grad = &record
                        .field_evaluations
                        .iter()
                        .find(|field| field.stage == "B1" && field.field == "potential_gradient")
                        .ok_or("actual first-kick force missing")?
                        .values;
                    let x = &pre.fields["positions"].values;
                    let v = &pre.fields["velocities"].values;
                    let centers: Vec<_> = x
                        .iter()
                        .zip(v)
                        .zip(grad)
                        .map(|((&x, &v), &g)| x + h * (1. + decay) / 2. * (v - h * g / 2.))
                        .collect();
                    let ou = &record
                        .field_evaluations
                        .iter()
                        .find(|field| field.stage == "O" && field.field == "executed_noise")
                        .ok_or("actual OU innovations missing")?
                        .values;
                    let position_noise = record.field_evaluations.iter().find(|field| {
                        field.stage == "position_diffusion" && field.field == "executed_noise"
                    });
                    let final_positions = record
                        .final_population
                        .observations
                        .field("positions")?
                        .values();
                    let mut max_residual = 0_f64;
                    for axis in 0..n * d {
                        let noise = position_noise.map_or(0., |field| field.values[axis]);
                        let predicted =
                            centers[axis] + h * scale * ou[axis] / 2. + h.sqrt() * sigma_x * noise;
                        let residual = (predicted - final_positions[axis]).abs();
                        max_residual = max_residual.max(residual);
                        checks += 1;
                        failures += usize::from(residual > 2e-10 * (1. + predicted.abs()));
                    }
                    let (probabilities, lower, upper) = if cfg["boundary"]["kind"] == "unbounded" {
                        (vec![1.; n], vec![], vec![])
                    } else {
                        let lower = floats(&cfg["boundary"]["domain"]["lower"])?;
                        let upper = floats(&cfg["boundary"]["domain"]["upper"])?;
                        if sigma <= 0. {
                            return Err(
                                "positive native whole-step position variance required".into()
                            );
                        }
                        let probabilities: Vec<f64> = centers
                            .chunks(d)
                            .map(|center| {
                                center
                                    .iter()
                                    .enumerate()
                                    .map(|(axis, x)| {
                                        (normal_cdf((upper[axis % upper.len()] - x) / sigma)
                                            - normal_cdf((lower[axis % lower.len()] - x) / sigma))
                                        .clamp(0., 1.)
                                    })
                                    .product()
                            })
                            .collect();
                        (probabilities, lower, upper)
                    };
                    let mean = probabilities.iter().sum::<f64>() / n as f64;
                    let variance =
                        probabilities.iter().map(|p| p * (1. - p)).sum::<f64>() / (n * n) as f64;
                    let law = poisson_binomial(&probabilities)?;
                    let threshold = mean / 2.;
                    let tail = law
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i as f64 / n as f64) < threshold)
                        .map(|(_, p)| p)
                        .sum::<f64>();
                    let bound = (-2. * n as f64 * (mean - threshold).powi(2)).exp();
                    checks += 3;
                    failures += usize::from(variance > 1. / (4. * n as f64) + 1e-14)
                        + usize::from(tail > bound + d as f64 * n as f64 * 1.5e-7)
                        + usize::from((law.iter().sum::<f64>() - 1.).abs() > 1e-10);
                    survival.push(json!({"step":record.report.step,"h":h,"N":n,"d":d,"prepared_positions":x,"prepared_velocities":v,"native_B1_gradient":grad,"centers_whole_kinetic":centers,"sigma":sigma,"noise_factor":noise_factor,"gamma":gamma,"OU_variance_time":variance_time,"position_diffusion":sigma_x,"lower":lower,"upper":upper,"probabilities":probabilities,"poisson_binomial_law":law,"conditional_mean":mean,"conditional_variance":variance,"variance_upper":1./(4.*n as f64),"lower_mass_threshold":threshold,"lower_tail_probability":tail,"Hoeffding_upper":bound,"actual_alive_fraction":observed["mass"],"centered_mass_innovation":observed["mass"].as_f64().unwrap()-mean,"maximum_full_native_position_identity_residual":max_residual,"conditioning":"complete prepared post-revival/collision state B1_input; all full BAOAB row OU and final-position Gaussian innovations integrated, no shared kinetic interactions; terminal boundary only","expression_ids":["chapter11-expression-0030","chapter11-expression-0035","chapter11-expression-0053"]}));
                }
                observations.push(value);
                updates += 1;
            }
            let path=out.save_json(&format!("native-{}",trajectories),&json!({"group":signature,"dataset":root,"source_path":entry.path,"source_sha256":entry.sha256,"case":case,"side":side,"seed":archive.gas_config.seed,"N":n,"d":d,"h":h,"native_configuration":cfg,"initial":initial,"observations":observations,"terminal":observations.last().ok_or("terminal missing")?["observation"],"grid_cuts":CUTS}))?;
            native_paths.push(path);
            if !survival.is_empty() {
                let path=out.save_json(&format!("conditional-survival-{}",trajectories),&json!({"source_path":entry.path,"source_sha256":entry.sha256,"group":signature,"seed":archive.gas_config.seed,"frames":survival}))?;
                survival_batches.push(path);
            }
            trajectories += 1;
        }
    }
    out.save_json("provenance", &json!(provenance))?;
    out.finish("complete")?;
    let verification = out.verify(true)?;
    fs::write(
        out.root().join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    let report = json!({"chapter":11,"summary":{"reference_checks":references.len(),"reference_failures":reference_failures,"native_checks":checks,"native_failures":failures,"native_trajectories":trajectories,"retained_native_updates":updates,"new_native_updates":0,"maximum_mass_shape_identity_residual":max_identity},"native_operands":native_paths,"conditional_survival_operands":survival_batches,"reference_operands_tag":"reference-operands","upstream":provenance,"verification":verification,"scope":"All native metrics are probability-normalized and permutation invariant. Discrete grid KL, if later computed, is only the exact KL of explicitly declared pushforward laws. No native LSI, QSD or unconditional mass contraction inferred from these finite trajectories."});
    fs::write(
        out.root().join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    if reference_failures + failures > 0 {
        return Err("Chapter 11 numerical comparison failed".into());
    }
    Ok(())
}
