//! Reanalyze complete native moment stages; does not generate trajectories.
use algorithmic_gas::{GasError, Result, tracking::RecordedStep};
use algorithmic_gas_benchmarks::{
    convergence_chapter05_remaining as estimates,
    convergence_experiments::{ArchiveStore, sha256_file},
    convergence_landscape_phase,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};
fn fail(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn stage<'a>(step: &'a RecordedStep<f64>, name: &str, field: &str) -> Result<&'a [f64]> {
    step.stages
        .iter()
        .find(|s| s.stage == name)
        .and_then(|s| s.fields.get(field))
        .map(|f| f.values.as_slice())
        .ok_or_else(|| fail("Missing native moment stage"))
}
fn estimate(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let se = if values.len() > 1 {
        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64
            / values.len() as f64)
            .sqrt()
    } else {
        0.
    };
    (mean, se)
}
fn source_expression(tag: &str) -> String {
    let source = include_str!(
        "../../../../../docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md"
    );
    source
        .split("$$")
        .skip(1)
        .step_by(2)
        .find(|formula| formula.contains(&format!("\\tag{{{tag}}}")))
        .unwrap()
        .trim()
        .into()
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err("gas-chapter05-remaining OUTPUT.json ARCHIVE_ROOT...".into());
    }
    if Path::new(&args[0]).exists() {
        return Err("Derived output exists; retain immutable evidence".into());
    }
    let mut observations = vec![];
    let mut artifacts = vec![];
    let mut sources = vec![];
    let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut skipped = vec![];
    for root in &args[1..] {
        let store = ArchiveStore::open(root)?;
        let index = Path::new(root).join("archive-index.json");
        sources
            .push(json!({"root":root,"index_sha256":sha256_file(&index)?,"status":store.status()}));
        for entry in store.entries() {
            if entry.kind != "native_run_archive" || !entry.tag.contains("/left_through") {
                continue;
            }
            let (case, reptext) = entry
                .tag
                .rsplit_once("/rep")
                .ok_or("Missing complete replica tag")?;
            let replicate: usize = reptext
                .split('/')
                .next()
                .ok_or("Missing replica")?
                .parse()?;
            let archive = store.load_archive(&entry.path)?;
            let cfg = serde_json::to_value(&archive.gas_config)?;
            let d = archive.steps[0]
                .before
                .observations
                .field("positions")?
                .width();
            let p = match convergence_landscape_phase::native_parameters(&cfg, d) {
                Ok(p) => p,
                Err(e) => {
                    skipped.push(json!({"tag":entry.tag,"reason":e.to_string()}));
                    continue;
                }
            };
            artifacts.push(json!({"root":root,"path":entry.path,"sha256":entry.sha256,"tag":entry.tag,"full_decode_validated":true}));
            for step in &archive.steps {
                let x = stage(step, "B1_input", "positions")?;
                let v = stage(step, "B1_input", "velocities")?;
                let v1 = stage(step, "B1", "velocities")?;
                let out = stage(step, "terminal", "positions")?;
                let vp = stage(step, "terminal", "velocities")?;
                let n = x.len() / d;
                let prediction = estimates::native_position_moments(
                    x,
                    v1,
                    d,
                    p.timestep,
                    p.friction,
                    p.ou_amplitude,
                    p.position_amplitude,
                )?;
                let actual = estimates::second_moments(out, d)?;
                let capped = estimates::second_moments(vp, d)?;
                let force = step
                    .field_evaluations
                    .iter()
                    .find(|f| f.stage == "B1" && f.field == "total_force")
                    .ok_or("Missing actual total force")?;
                let ou_time = if p.friction == 0. {
                    p.timestep
                } else {
                    -(-2. * p.friction * p.timestep).exp_m1() / (2. * p.friction)
                };
                let sigma = p.ou_amplitude / ou_time.sqrt();
                let generator = estimates::component_generators(
                    x,
                    v,
                    &force.values,
                    d,
                    p.friction,
                    sigma,
                    p.friction,
                    true,
                )?;
                let residuals = json!({"variance":actual["variance"].as_f64().unwrap()-prediction["variance_prediction"].as_f64().unwrap(),"total":actual["total"].as_f64().unwrap()-prediction["total_prediction"].as_f64().unwrap(),"barycenter":actual["barycenter"].as_f64().unwrap()-prediction["barycenter_prediction"].as_f64().unwrap()});
                let first_force_residual = x
                    .iter()
                    .zip(v)
                    .zip(v1)
                    .zip(&force.values)
                    .map(|(((_x, v), v1), f)| (v1 - v - p.timestep / 2. * f).abs())
                    .fold(0., f64::max);
                let checks = json!({"first_kick_identity":first_force_residual,"parallel_axis":(actual["total"].as_f64().unwrap()-actual["variance"].as_f64().unwrap()-actual["barycenter"].as_f64().unwrap()).abs(),"postcap_total":capped["total"],"postcap_barycenter":capped["barycenter"],"postcap_variance":capped["variance"],"postcap_bound":p.velocity_cap*p.velocity_cap,"native_increment_residual":prediction["conditional_increment"].as_f64().unwrap()-prediction["conditional_increment_bound"].as_f64().unwrap(),"generator_velocity_residual":generator["velocity_generator"].as_f64().unwrap()-generator["velocity_bound"].as_f64().unwrap(),"generator_barycenter_residual":generator["barycenter_generator"].as_f64().unwrap()-generator["barycenter_bound"].as_f64().unwrap(),"generator_position_residual":generator["position_generator"].as_f64().unwrap().abs()-generator["position_bound"].as_f64().unwrap()});
                let mut density = Value::Null;
                if let Some(tail) = case.strip_prefix("dimension/omega") {
                    let omega: f64 = tail.split('_').next().ok_or("Missing curvature")?.parse()?;
                    if cfg["qft"]["viscosity"].is_null() && cfg["qft"]["graph_viscosity"].is_null()
                    {
                        let params = estimates::CompactKineticParameters {
                            dimension: d,
                            timestep: p.timestep,
                            friction: p.friction,
                            ou_standard_deviation: p.ou_amplitude,
                            position_standard_deviation: p.position_amplitude,
                            velocity_cap: p.velocity_cap,
                            force_lipschitz: omega,
                            force_at_origin: 0.,
                            input_position_radius: 2.,
                            input_velocity_radius: 2.,
                            target_position_radius: 2.,
                            target_velocity_radius: 1.,
                            target_center_norm: 0.,
                        };
                        let certificate = estimates::compact_kinetic_minorization(&params)?;
                        let force_defect = x
                            .iter()
                            .zip(&force.values)
                            .map(|(x, f)| (f + omega * x).abs())
                            .fold(0., f64::max);
                        let mut checked = 0usize;
                        let mut maximum = f64::NEG_INFINITY;
                        for i in 0..n {
                            let slice = |a: &[f64]| a[i * d..(i + 1) * d].to_vec();
                            let norm = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
                            let xx = slice(x);
                            let vv = slice(v);
                            let ox = slice(out);
                            let ov = slice(vp);
                            if norm(&xx) <= params.input_position_radius
                                && norm(&vv) <= params.input_velocity_radius
                                && norm(&ox) <= params.target_position_radius
                                && norm(&ov) <= params.target_velocity_radius
                            {
                                let actual = estimates::quadratic_output_log_density(
                                    &xx,
                                    &vv,
                                    &ox,
                                    &ov,
                                    p.timestep,
                                    p.friction,
                                    omega,
                                    p.ou_amplitude,
                                    p.position_amplitude,
                                    p.velocity_cap,
                                )?;
                                maximum = maximum.max(
                                    certificate["log_density_lower"].as_f64().unwrap() - actual,
                                );
                                checked += 1;
                            }
                        }
                        density = json!({"certificate":certificate,"declared_harmonic_curvature":omega,"force_defect":force_defect,"interior_rows_checked":checked,"maximum_log_density_residual":if checked>0{Some(maximum)}else{None},"scope":"Pointwise harmonic native joint density including inverse-cap Jacobian checked only where declared input and target bounds hold; no selected data class is substituted for a global hypothesis."});
                    }
                }
                let row = json!({"case":case,"replicate":replicate,"epoch":step.epoch,"step":step.report.step,"N":n,"d":d,"seed":archive.gas_config.seed,"prediction":prediction,"actual":actual,"residuals":residuals,"checks":checks,"generator":generator,"density":density,"archive":entry.path,"root":root,"archive_sha256":entry.sha256});
                let group = format!("{root}/{case}/epoch{}/step{}", step.epoch, step.report.step);
                groups.entry(group).or_default().push(row.clone());
                observations.push(row);
            }
        }
    }
    let mut comparisons = vec![];
    for (group, rows) in groups {
        let mut replicas = std::collections::BTreeSet::new();
        for row in &rows {
            if !replicas.insert(row["replicate"].as_u64().unwrap()) {
                return Err("Duplicate retained replica/step cannot supply uncertainty".into());
            }
        }
        for component in ["variance", "total", "barycenter"] {
            let r: Vec<_> = rows
                .iter()
                .map(|r| r["residuals"][component].as_f64().unwrap())
                .collect();
            let (mean, se) = estimate(&r);
            comparisons.push(json!({"id":format!("{group}/conditional_{component}"),"source_labels":["cor-kinetic-native-positional-moments"],"whole_theorem_validated":false,"observed":mean,"bound":0.,"standard_error":se,"residual":mean,"relation":"equal","passed":mean.abs()<=6.*se+3e-11,"independent_complete_replicas":rows.len(),"hypotheses":rows[0]["prediction"]["hypotheses"],"scope":"Conditional native position expectation residual, averaged over independent complete left replicas at one fixed epoch/step. No rows or dependent steps are treated as additional replicas."}));
        }
        let mut maximum_standardized_residual = 0_f64;
        let mut moment_components = vec![];
        for component in ["variance", "total", "barycenter"] {
            let values: Vec<_> = rows
                .iter()
                .map(|r| r["residuals"][component].as_f64().unwrap())
                .collect();
            let (mean, se) = estimate(&values);
            maximum_standardized_residual =
                maximum_standardized_residual.max(mean.abs() / (se + 5e-12));
            moment_components.push(json!({"component":component,"observed_residual":mean,"standard_error":se,"bound":0.}));
        }
        comparisons.push(json!({"id":format!("{group}/all_conditional_position_moments"),"source_labels":["cor-kinetic-native-positional-moments"],"source_formula":source_expression("5.X3"),"observed":maximum_standardized_residual,"bound":6.,"relation":"less_equal","passed":maximum_standardized_residual<=6.,"components":moment_components,"independent_complete_replicas":rows.len(),"hypotheses":rows[0]["prediction"]["hypotheses"],"scope":"All three whole-formula 5.X3 expectations checked jointly by maximum absolute standardized residual; each component retains its complete-replica SE, with 5e-12 arithmetic allowance. No new data or global-domain certificate."}));
        let density: Vec<_> = rows
            .iter()
            .filter(|r| {
                r["density"]["interior_rows_checked"]
                    .as_u64()
                    .is_some_and(|n| n > 0)
            })
            .collect();
        if !density.is_empty() {
            let max = density
                .iter()
                .map(|r| {
                    r["density"]["maximum_log_density_residual"]
                        .as_f64()
                        .unwrap()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            let force = density
                .iter()
                .map(|r| r["density"]["force_defect"].as_f64().unwrap())
                .fold(0., f64::max);
            comparisons.push(json!({"id":format!("{group}/joint_density_lower"),"source_labels":["lem-kinetic-minorization"],"observed":max,"bound":0.,"residual":max,"standard_error":0.,"relation":"less_equal","passed":max<=3e-11 && force<=3e-11,"force_defect":force,"certificate":density[0]["density"]["certificate"],"rows_checked":density.iter().map(|r|r["density"]["interior_rows_checked"].as_u64().unwrap()).sum::<u64>(),"whole_theorem_validated":false,"scope":"Analytically known exact quadratic native joint density versus constructive declared compact lower bound at retained interior outputs; cap Jacobian included. Interacting force rows remain gated."}));
        }
        for (name, bound, label) in [
            ("first_kick_identity", 0., "def-baoab-integrator"),
            ("parallel_axis", 0., "def-positional-variance-recall"),
            (
                "postcap_total",
                rows[0]["checks"]["postcap_bound"].as_f64().unwrap(),
                "cor-net-barycenter-drift",
            ),
            (
                "postcap_barycenter",
                rows[0]["checks"]["postcap_bound"].as_f64().unwrap(),
                "cor-net-barycenter-drift",
            ),
            (
                "postcap_variance",
                rows[0]["checks"]["postcap_bound"].as_f64().unwrap(),
                "thm-velocity-variance-contraction-kinetic",
            ),
            (
                "native_increment_residual",
                0.,
                "cor-kinetic-native-positional-moments",
            ),
            (
                "generator_velocity_residual",
                0.,
                "thm-velocity-variance-contraction-kinetic",
            ),
            (
                "generator_barycenter_residual",
                0.,
                "thm-velocity-barycenter-dissipation",
            ),
            (
                "generator_position_residual",
                0.,
                "thm-positional-variance-bounded-expansion",
            ),
        ] {
            let max = rows
                .iter()
                .map(|r| r["checks"][name].as_f64().unwrap())
                .fold(f64::NEG_INFINITY, f64::max);
            let source_formula = if name == "native_increment_residual" {
                Some(source_expression("5.X4"))
            } else {
                None
            };
            comparisons.push(json!({"id":format!("{group}/{name}"),"source_labels":[label],"source_formula":source_formula,"observed":max,"bound":bound,"residual":max-bound,"relation":"less_equal","standard_error":0.,"passed":max<=bound+3e-11,"scope":if name.starts_with("generator") {"Pointwise continuum generator algebra on retained states only. Native fixed-cap continuum transfer is gated; no native rate credit."}else{"Actual retained native stage identity or pathwise normalized moment bound; no statistical uncertainty required."},"whole_theorem_validated":false}));
        }
    }
    let failed = comparisons.iter().filter(|r| r["passed"] != true).count();
    let source =
        Path::new("../docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md");
    let report = json!({"chapter":5,"title":"Remaining native conditional moment and continuum algebra checks","new_native_steps":0,"source_path":source,"current_source_sha256":sha256_file(source)?,"input_indexes":sources,"artifacts":artifacts,"observations":observations,"comparisons":comparisons,"skipped":skipped,"summary":{"retained_steps":observations.len(),"independent_archive_replicas":artifacts.len(),"comparisons":comparisons.len(),"comparisons_failed":failed},"closure":{"conditional_native_positional_moments":"Exact per-state conditional expectations validated with retained unrestricted Gaussian innovations.","continuum_transfer":"Canonical repeated radial cap remains gated; algebraic generator checks do not imply native rates.","regional_source_pressure":"Existing Keystone pressure is retained with its signed donor/barycenter/collision/force/cap residual; no unconditional residual upper bound for current positive-fitness unbounded Rastrigin has been established.","phase_transfer":"No empirical phase transitions are substituted for proved discrepancy transfer."}});
    fs::write(&args[0], serde_json::to_vec_pretty(&report)?)?;
    eprintln!(
        "{} retained complete steps, {} comparisons, {failed} failures",
        observations.len(),
        comparisons.len()
    );
    if failed > 0 {
        return Err("Remaining native estimates rejected; inspect retained residuals".into());
    }
    Ok(())
}
