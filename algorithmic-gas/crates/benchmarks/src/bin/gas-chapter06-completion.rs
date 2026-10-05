//! Recompute complete-step source/tail moments from immutable native histories.
use algorithmic_gas_benchmarks::{
    convergence_chapter06_completion::{
        NativeTailParameters, native_selected_source_tail_interface,
        uniform_final_gaussian_box_hazard,
    },
    convergence_experiments::{ArchiveStore, sha256_file},
    convergence_landscape_phase::native_parameters,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

fn moment(rows: &[f64], d: usize, order: f64, alive: Option<&[bool]>) -> f64 {
    let n = rows.len() / d;
    rows.chunks_exact(d)
        .enumerate()
        .filter(|(i, _)| alive.is_none_or(|a| a[*i]))
        .map(|(_, x)| x.iter().map(|v| v * v).sum::<f64>().powf(order / 2.))
        .sum::<f64>()
        / n as f64
}
fn group(tag: &str) -> String {
    if let Some((case, rest)) = tag.split_once("/rep") {
        return format!(
            "{case}-{}",
            if rest.contains("/left_") {
                "left"
            } else if rest.contains("/right_") {
                "right"
            } else {
                "unclassified"
            }
        );
    }
    if let Some((case, rest)) = tag.split_once("-distribution") {
        return format!("{case}-distribution{}", rest.split('-').next().unwrap());
    }
    if let Some((case, rest)) = tag.split_once("-rep") {
        let side = if rest.contains("-left-") {
            "left"
        } else if rest.contains("-right-") {
            "right"
        } else {
            "unclassified"
        };
        return format!("{case}-{side}");
    }
    tag.to_string()
}
#[derive(Default)]
struct Aggregate {
    samples: usize,
    observed: f64,
    bound: f64,
    second: f64,
    source: f64,
    before: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("gas-chapter06-completion INPUT_DATASET EMPTY_OUTPUT_DIR".into());
    }
    let input = ArchiveStore::open(&args[0])?;
    if !["complete", "completed"].contains(&input.status()) {
        return Err("reanalyze only complete committed native datasets".into());
    }
    let mut output = ArchiveStore::new(&args[1])?;
    output.save_json("completion-source", &json!({
        "module":include_str!("../convergence_chapter06_completion.rs"),
        "runner":include_str!("gas-chapter06-completion.rs"),
        "native_parameter_bridge":include_str!("../convergence_landscape_phase.rs"),
        "chapter06":include_str!("../../../../../docs/source/2_fractal_gas/convergence_program/06_convergence.md"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,
        "input_index_sha256":sha256_file(&input.root().join("archive-index.json"))?,
        "input_dataset":input.root(),"new_native_steps":0,
        "scope":"Retained raw native source/jitter/kinetic histories; current one-step moment interfaces; no new simulations or QSD certificate."}))?;
    let mut rows = vec![];
    let mut groups: BTreeMap<(String, u64, u64, usize, usize), Aggregate> = BTreeMap::new();
    let mut consumed = vec![];
    let mut gaps = vec![];
    let mut steps = 0usize;
    let mut raw_paths = vec![];
    let mut hazard_profiles = BTreeMap::new();
    let mut initial_seed_units: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut seen_updates = BTreeSet::new();
    for entry in input
        .entries()
        .iter()
        .filter(|e| e.kind == "native_run_archive")
    {
        let archive = input.load_archive(&entry.path)?;
        let law = group(&entry.tag);
        initial_seed_units
            .entry(law.clone())
            .or_default()
            .insert(archive.gas_config.seed);
        consumed.push(json!({"path":entry.path,"sha256":entry.sha256,"tag":entry.tag,"records":archive.steps.len()}));
        let gradient = archive
            .providers
            .get("gradient")
            .map(String::as_str)
            .unwrap_or("");
        let force = if gradient.contains("/Quadratic/") {
            (1., 0.)
        } else if gradient.contains("/Sphere/") {
            (2., 0.)
        } else if gradient.contains("/Rastrigin/") {
            (2., 20. * std::f64::consts::PI)
        } else {
            gaps.push(json!({"archive":entry.path,"provider":gradient,"missing":"Global harmonic-plus-bounded-residual force profile for this actual gradient."}));
            continue;
        };
        let cfg = serde_json::to_value(&archive.gas_config)?;
        for record in &archive.steps {
            let Some(source) = record.stages.iter().find(|s| s.stage == "literal_clone") else {
                gaps.push(json!({"archive":entry.path,"step":record.report.step,"missing":"literal_clone source array"}));
                continue;
            };
            let positions = &source.fields["positions"];
            let d = positions.item_shape[0];
            let n = positions.values.len() / d;
            let params = match native_parameters(&cfg, d) {
                Ok(p) => p,
                Err(e) => {
                    gaps.push(json!({"archive":entry.path,"step":record.report.step,"missing":e.to_string()}));
                    continue;
                }
            };
            if params.timestep * params.viscosity / 2. > 1. {
                return Err("actual first-kick stochasticity premise fails".into());
            }
            let before_x = record.before.observations.field("positions")?.values();
            let before_v = record.before.observations.field("velocities")?.values();
            let max_velocity = before_v
                .chunks_exact(d)
                .chain(source.fields["velocities"].values.chunks_exact(d))
                .map(|x| x.iter().map(|v| v * v).sum::<f64>().sqrt())
                .fold(0., f64::max);
            if max_velocity > params.velocity_cap + 1e-10 {
                return Err(
                    "frozen velocity cap premise fails, including dead collision slots".into(),
                );
            }
            let live = record.final_population.eligible(false);
            let final_x = record
                .final_population
                .observations
                .field("positions")?
                .values();
            if !seen_updates.insert((law.clone(), archive.gas_config.seed, record.report.step)) {
                return Err("duplicate seed/time record in law group".into());
            }
            if cfg["boundary"]["kind"] == "absorbing_box" {
                let lower = cfg["boundary"]["domain"]["lower"]
                    .as_array()
                    .ok_or("missing box lower")?;
                let upper = cfg["boundary"]["domain"]["upper"]
                    .as_array()
                    .ok_or("missing box upper")?;
                let l = upper[0].as_f64().ok_or("box upper")?;
                if lower.iter().all(|x| x.as_f64() == Some(-l))
                    && upper.iter().all(|x| x.as_f64() == Some(l))
                {
                    hazard_profiles.entry((law.clone(), n, d)).or_insert(
                        uniform_final_gaussian_box_hazard(d, n, l, params.position_amplitude)?,
                    );
                }
            }
            for order in [4usize, 8] {
                let source_m = moment(&positions.values, d, order as f64, None);
                let before_m = moment(
                    before_x,
                    d,
                    order as f64,
                    Some(&record.before.eligible(false)),
                );
                let observed = moment(final_x, d, order as f64, Some(&live));
                let mut p = NativeTailParameters {
                    dimension: d,
                    moment_order: order as f64,
                    timestep: params.timestep,
                    friction: params.friction,
                    harmonic_curvature: force.0,
                    residual_per_coordinate: force.1,
                    clone_jitter_standard_deviation: params.clone_jitter,
                    ou_standard_deviation: params.ou_amplitude,
                    final_position_standard_deviation: params.position_amplitude,
                    frozen_velocity_norm_bound: params.velocity_cap,
                    restitution: params.restitution,
                };
                let bound = native_selected_source_tail_interface(&p, source_m)?;
                p.moment_order = 2. * order as f64;
                let second = native_selected_source_tail_interface(
                    &p,
                    moment(&positions.values, d, 2. * order as f64, None),
                )?
                .one_step_moment_upper;
                let state_velocity_bound = {
                    p.moment_order = order as f64;
                    p.frozen_velocity_norm_bound = max_velocity;
                    native_selected_source_tail_interface(&p, source_m)?.one_step_moment_upper
                };
                let a = groups
                    .entry((law.clone(), record.report.step, order as u64, n, d))
                    .or_default();
                a.samples += 1;
                a.observed += observed;
                a.bound += bound.one_step_moment_upper;
                a.second += second;
                a.source += source_m;
                a.before += before_m;
                rows.push(json!({"law_group":law,"seed":archive.gas_config.seed,"step":record.report.step,
                    "N":n,"d":d,"moment_order":order,"alive":live.iter().filter(|a|**a).count(),
                    "before_alive_N_moment":before_m,"literal_source_N_moment":source_m,
                    "observed_completed_alive_N_moment":observed,
                    "conditional_moment_upper":bound.one_step_moment_upper,
                    "conditional_second_moment_upper":second,
                    "state_velocity_conditional_upper":state_velocity_bound,
                    "frozen_velocity_maximum":max_velocity,"native_parameters":p,
                    "source_archive":entry.path,"source_sha256":entry.sha256,
                    "scope":"One realized conditional preparation; observable is N-normalized and zero for dead rows. Bound is conditional expectation, not a pathwise comparison or survival-conditioned bound."}));
            }
            steps += 1;
            if rows.len() >= 4096 {
                raw_paths.push(output.save_json(
                    &format!("moments-batch{}", raw_paths.len()),
                    &json!({"rows":rows}),
                )?);
                rows.clear();
            }
        }
        if steps.is_multiple_of(16384) {
            eprintln!("processed {steps} retained updates");
        }
    }
    if !rows.is_empty() {
        raw_paths.push(output.save_json(
            &format!("moments-batch{}", raw_paths.len()),
            &json!({"rows":rows}),
        )?);
    }
    let delta = 0.01 / groups.len().max(1) as f64;
    let mut comparisons = vec![];
    for ((law, step, order, n, d), a) in &groups {
        let m = initial_seed_units[law].len() as f64;
        let allowance = ((1. - delta) / delta * a.second).sqrt() / m;
        let observed = a.observed / m;
        let bound = a.bound / m;
        comparisons.push(json!({"id":format!("{law}-step{step}-p{order}"),"law_group":law,"step":step,
            "N":n,"d":d,"moment_order":order,"independent_seed_units":m as usize,
            "surviving_preparation_seed_units":a.samples,
            "absorbed_zero_seed_units":m as usize-a.samples,
            "observed":observed,"bound":bound,"cantelli_allowance":allowance,
            "signed_residual":observed-bound,"passed":observed<=bound+allowance,
            "mean_literal_source_moment":a.source/m,"mean_before_alive_N_moment":a.before/m,
            "empirical_source_increment":(a.source-a.before)/m,
            "family_failure_budget":0.01,"per_comparison_failure_budget":delta,
            "hypotheses":"Condition simultaneously on actual pre-jitter source/acceptance/rotation preparations of independent seed units. Previously native-absorbed trajectory units contribute exactly zero to observed and conditional-bound sums; retain each initial law's original seed denominator. E[Y²|preparations] bounded by normalized 2p moment. No independence between timesteps or rows; paired left/right laws grouped separately. Source increments are diagnostics, not uniform suprema.",
            "relation":"conditional expectation upper bound with family-adjusted Cantelli uncertainty"}));
    }
    let failed = comparisons.iter().filter(|c| c["passed"] != true).count();
    let hazard_rows: Vec<Value> = hazard_profiles
        .into_iter()
        .map(|((law, n, d), h)| json!({"law_group":law,"N":n,"d":d,"hazard":h}))
        .collect();
    let report = json!({"chapter":6,"title":"Retained full-native selected-source and positional tail interfaces",
        "consumed_archives":consumed,"raw_moment_batches":raw_paths,"comparisons":comparisons,"uniform_box_hazards":hazard_rows,
        "gaps":gaps,"summary":{"retained_complete_updates":steps,"conditional_moment_rows":steps*2,"comparisons":groups.len(),"comparisons_failed":failed,"new_native_steps":0},
        "scope":"Global force envelopes and complete selected-source arrays, including historical donors, are reconstructed from recorded native literal_clone stages. Moments use N normalization; dead positions are excluded. Conditional expectation checks use own law and time independent-seed groups. No equilibrium/QSD/global selected-source drift is asserted."});
    output.save_json("completion-report", &report)?;
    output.finish("complete")?;
    fs::write(
        PathBuf::from(&args[1]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    if failed > 0 {
        return Err("conditional moment comparison failure".into());
    }
    Ok(())
}
