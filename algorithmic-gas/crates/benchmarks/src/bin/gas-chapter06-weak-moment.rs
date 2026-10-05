//! Full native weak-selection source and complete-update moment expectations.
use algorithmic_gas_benchmarks::{
    convergence_chapter06_completion::{
        NativeTailParameters, native_selected_source_tail_interface,
    },
    convergence_chapter06_source_moment::{
        FitnessChannelRange, SourceMomentParameters, selected_native_moment_closure,
        selected_native_refined_moment_closure, selected_native_root_moment_closure,
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
fn get(v: &Value, p: &str) -> Result<f64, Box<dyn std::error::Error>> {
    v.pointer(p)
        .and_then(Value::as_f64)
        .filter(|x| x.is_finite())
        .ok_or_else(|| format!("Missing finite actual native parameter {p}").into())
}
fn moment(x: &[f64], d: usize, p: f64) -> f64 {
    x.chunks_exact(d)
        .map(|r| r.iter().map(|v| v * v).sum::<f64>().powf(p / 2.))
        .sum::<f64>()
        / (x.len() / d) as f64
}
fn law(tag: &str) -> String {
    let Some((prefix, suffix)) = tag.split_once("-rep") else {
        return tag.into();
    };
    format!(
        "{prefix}-{}",
        if suffix.contains("-left-") {
            "left"
        } else if suffix.contains("-right-") {
            "right"
        } else {
            "unspecified"
        }
    )
}
#[derive(Default)]
struct Aggregate {
    units: usize,
    observed: f64,
    bound: f64,
    second: f64,
    accepted: usize,
    walkers: usize,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("gas-chapter06-weak-moment COMPLETE_NATIVE_DATASET EMPTY_OUTPUT_DIR".into());
    }
    let input = ArchiveStore::open(&args[0])?;
    if !["complete", "completed"].contains(&input.status()) {
        return Err("complete immutable input required".into());
    }
    let mut output = ArchiveStore::new(&args[1])?;
    output.save_json("weak-moment-source",&json!({"runner":include_str!("gas-chapter06-weak-moment.rs"),
        "source_interface":include_str!("../convergence_chapter06_source_moment.rs"),
        "kinetic_interface":include_str!("../convergence_chapter06_completion.rs"),
        "proof_note":include_str!("../../../../proof-validation/chapter06-global-tail-closure.md"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,
        "input_index_sha256":sha256_file(&input.root().join("archive-index.json"))?,
        "new_native_steps":0,"scope":"Actual complete source/measurement/acceptance/Haar/jitter/kinetic expectations in the certified conservative weak-selection moment regime."}))?;
    let mut groups: BTreeMap<(String, u64, usize, String), Aggregate> = BTreeMap::new();
    let mut profiles = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut rows = vec![];
    let mut consumed = vec![];
    let mut raw_paths = vec![];
    let mut updates = 0;
    let mut accepted = 0;
    let mut gate_residual = f64::NEG_INFINITY;
    for e in input
        .entries()
        .iter()
        .filter(|e| e.kind == "native_run_archive")
    {
        let a = input.load_archive(&e.path)?;
        let cfg = serde_json::to_value(&a.gas_config)?;
        if cfg["boundary"]["kind"] != "unbounded"
            || cfg["cloning_donors"]["history_window"] != 0
            || cfg["cloning_donors"]["distance"]["kind"] != "squashed_phase_space"
            || cfg["cloning_donors"]["kernel"]["kind"] != "gaussian"
            || cfg["cloning_donors"]["law"] != "independent"
            || cfg["cloning_donors"]["allow_self"] != false
        {
            return Err(
                "actual conservative current-frame squashed Gaussian source hypotheses fail".into(),
            );
        }
        let channel = |name: &str| -> Result<FitnessChannelRange, Box<dyn std::error::Error>> {
            if cfg["fitness"][format!("{name}_map")]["kind"] != "logistic" {
                return Err("actual logistic channel required".into());
            }
            Ok(FitnessChannelRange {
                amplitude: get(&cfg, &format!("/fitness/{name}_map/amplitude"))?,
                positive_floor: get(&cfg, &format!("/fitness/{name}_map/floor"))?,
                exponent: get(&cfg, &format!("/fitness/{name}_exponent"))?,
            })
        };
        let source = SourceMomentParameters {
            position_feature_radius: get(&cfg, "/cloning_donors/distance/position_radius")?,
            velocity_feature_radius: get(&cfg, "/cloning_donors/distance/velocity_radius")?,
            algorithmic_velocity_weight: get(&cfg, "/cloning_donors/distance/lambda")?,
            cloning_bandwidth: get(&cfg, "/cloning_donors/kernel/width")?,
            gate_saturation: get(&cfg, "/clone_decision/saturation")?,
            gate_epsilon: get(&cfg, "/clone_decision/epsilon")?,
            fitness_channels: vec![channel("reward")?, channel("diversity")?],
            historical_window: 0,
            component_collision_enabled: !cfg["clone_transform"]["restitution"].is_null(),
        };
        let gradient = a
            .providers
            .get("gradient")
            .map(String::as_str)
            .unwrap_or("");
        let (omega, residual) = if gradient.contains("/Rastrigin/") {
            (2., 20. * std::f64::consts::PI)
        } else if gradient.contains("/Quadratic/") {
            (1., 0.)
        } else {
            return Err("unsupported analytic force profile".into());
        };
        consumed.push(json!({"path":e.path,"sha256":e.sha256,"tag":e.tag}));
        for r in &a.steps {
            let key = law(&e.tag);
            if !seen.insert((key.clone(), a.gas_config.seed, r.report.step)) {
                return Err("duplicate native seed/time in law".into());
            }
            let n = r.before.len();
            let x = r.before.observations.field("positions")?;
            let d = x.width();
            if r.before.eligible(false).iter().any(|a| !*a)
                || r.final_population.eligible(false).iter().any(|a| !*a)
                || r.elite_injection.is_some()
                || r.report.revivals != 0
            {
                return Err("all-alive no-external-source hypothesis fails".into());
            }
            let literal = r
                .stages
                .iter()
                .find(|s| s.stage == "literal_clone")
                .ok_or("literal source missing")?;
            let params = native_parameters(&cfg, d)?;
            let frozen_max = r
                .before
                .observations
                .field("velocities")?
                .values()
                .chunks_exact(d)
                .chain(literal.fields["velocities"].values.chunks_exact(d))
                .map(|v| v.iter().map(|z| z * z).sum::<f64>().sqrt())
                .fold(0., f64::max);
            if frozen_max > params.velocity_cap + 1e-10
                || params.timestep * params.viscosity / 2. > 1.
            {
                return Err("native cap/stochastic first matrix premise fails".into());
            }
            for p in [4usize, 8] {
                let before = moment(x.values(), d, p as f64);
                let before_second = moment(x.values(), d, 2. * p as f64);
                let source_observed = moment(&literal.fields["positions"].values, d, p as f64);
                let completed = moment(
                    r.final_population.observations.field("positions")?.values(),
                    d,
                    p as f64,
                );
                let mut native = NativeTailParameters {
                    dimension: d,
                    moment_order: p as f64,
                    timestep: params.timestep,
                    friction: params.friction,
                    harmonic_curvature: omega,
                    residual_per_coordinate: residual,
                    clone_jitter_standard_deviation: params.clone_jitter,
                    ou_standard_deviation: params.ou_amplitude,
                    final_position_standard_deviation: params.position_amplitude,
                    frozen_velocity_norm_bound: params.velocity_cap,
                    restitution: params.restitution,
                };
                let interface = native_selected_source_tail_interface(&native, before)?;
                let closure = selected_native_moment_closure(&interface, &source)?;
                let refined = selected_native_refined_moment_closure(
                    &interface,
                    &source,
                    params.viscosity == 0. || params.normalization == "count",
                )?;
                let root = selected_native_root_moment_closure(
                    &interface,
                    &source,
                    params.viscosity == 0. || params.normalization == "count",
                )?;
                if !root.closes {
                    return Err("actual weak profile does not absorb source moment gain".into());
                }
                native.moment_order = 2. * p as f64;
                let second = selected_native_moment_closure(
                    &native_selected_source_tail_interface(&native, before_second)?,
                    &source,
                )?;
                let refined_second = selected_native_refined_moment_closure(
                    &native_selected_source_tail_interface(&native, before_second)?,
                    &source,
                    params.viscosity == 0. || params.normalization == "count",
                )?;
                let root_second = selected_native_root_moment_closure(
                    &native_selected_source_tail_interface(&native, before_second)?,
                    &source,
                    params.viscosity == 0. || params.normalization == "count",
                )?;
                let source_bound = closure.source.current_frame_source_moment_coefficient * before;
                let source_second =
                    closure.source.current_frame_source_moment_coefficient * before_second;
                let full_bound = closure.moment.coefficient * before + closure.moment.additive;
                let full_second =
                    second.moment.coefficient * before_second + second.moment.additive;
                let refined_bound =
                    refined.refined_moment.coefficient * before + refined.refined_moment.additive;
                let refined_second_bound = refined_second.refined_moment.coefficient
                    * before_second
                    + refined_second.refined_moment.additive;
                let root_bound = (root.root_moment_coefficient * before.powf(1. / p as f64)
                    + root.additive_root_budget)
                    .powi(p as i32);
                let root_second_bound = (root_second.root_moment_coefficient
                    * before_second.powf(1. / (2. * p as f64))
                    + root_second.additive_root_budget)
                    .powi(2 * p as i32);
                for choice in &r.report.clone_plan.choices {
                    let probability = choice
                        .probability
                        .ok_or("recorded native gate probability missing")?;
                    gate_residual =
                        gate_residual.max(probability - closure.source.acceptance_upper);
                    if probability > closure.source.acceptance_upper + 1e-12 {
                        return Err("actual acceptance exceeds global profile".into());
                    }
                }
                profiles.entry((key.clone(),p)).or_insert(json!({"law_group":key,"N":n,"d":d,"moment_order":p,
                    "actual_native_viscosity":params.viscosity,"actual_native_normalization":params.normalization,
                    "parameters":cfg,"closure":closure,"refined_closure":refined,"root_closure":root}));
                for (kind, observed, bound, second_moment) in [
                    ("source", source_observed, source_bound, source_second),
                    ("complete", completed, full_bound, full_second),
                    (
                        "complete_refined",
                        completed,
                        refined_bound,
                        refined_second_bound,
                    ),
                    ("complete_root", completed, root_bound, root_second_bound),
                ] {
                    let group = groups
                        .entry((key.clone(), r.report.step, p, kind.into()))
                        .or_default();
                    group.units += 1;
                    group.observed += observed;
                    group.bound += bound;
                    group.second += second_moment;
                    group.accepted += r.report.clones;
                    group.walkers += n;
                }
                rows.push(json!({"law_group":key,"seed":a.gas_config.seed,"step":r.report.step,"N":n,"d":d,
                    "moment_order":p,"before":before,"before_2p":before_second,"literal_source":source_observed,
                    "completed":completed,"source_bound":source_bound,"source_second_bound":source_second,
                    "complete_bound":full_bound,"complete_second_bound":full_second,"refined_complete_bound":refined_bound,
                    "refined_complete_second_bound":refined_second_bound,"root_complete_bound":root_bound,
                    "root_complete_second_bound":root_second_bound,"accepted_clones":r.report.clones,
                    "source_archive":e.path,"source_sha256":e.sha256}));
            }
            updates += 1;
            accepted += r.report.clones;
            if rows.len() >= 4096 {
                raw_paths.push(output.save_json(
                    &format!("weak-moment-batch{}", raw_paths.len()),
                    &json!({"rows":rows}),
                )?);
                rows.clear();
            }
        }
    }
    if !rows.is_empty() {
        raw_paths.push(output.save_json(
            &format!("weak-moment-batch{}", raw_paths.len()),
            &json!({"rows":rows}),
        )?);
    }
    let delta = 0.01 / groups.len().max(1) as f64;
    let mut comparisons = vec![];
    for ((law, step, p, kind), g) in groups {
        let m = g.units as f64;
        let allowance = ((1. - delta) / delta * g.second).sqrt() / m;
        let observed = g.observed / m;
        let bound = g.bound / m;
        comparisons.push(json!({"id":format!("{law}-{kind}-p{p}-step{step}"),"law_group":law,"step":step,
            "moment_order":p,"kind":kind,"observed":observed,"bound":bound,"cantelli_allowance":allowance,
            "signed_residual":observed-bound,"passed":observed<=bound+allowance,
            "independent_seed_units":g.units,"accepted_clones":g.accepted,"processed_walkers":g.walkers,
            "hypotheses":"Conditional on independent entering native swarms; source/moment expectations average complete measurement/donor/acceptance/Haar/jitter/kinetic laws. N-normalized 2p bound controls the conditional second moment, without within-swarm independence or sampled maxima. Paired left/right laws and times remain separate.",
            "scope":"Complete actual native weak-selection conservative source or complete-update expectation. This is a moment test, not a law-convergence fit."}));
    }
    let failures = comparisons.iter().filter(|c| c["passed"] != true).count();
    let profile_rows: Vec<_> = profiles.into_values().collect();
    let report = json!({"chapter":6,"comparisons":comparisons,"profiles":profile_rows,"raw_batches":raw_paths,
        "consumed_archives":consumed,"maximum_acceptance_upper_residual":gate_residual,
        "summary":{"complete_native_updates":updates,"accepted_clones":accepted,"comparisons":comparisons.len(),
            "comparisons_failed":failures,"native_steps_added":0},
        "family_failure_budget":0.01,"scope":"Full actual source preparation and complete native weak-selection p4/p8 moment closure; globally certified coefficients are reconstructed from actual gc, while independent-seed observations retain uncertainty. No equilibrium/QSD/joint-law LSI or pure cross-phase error contraction claim."});
    output.save_json("weak-moment-report", &report)?;
    output.finish("complete")?;
    fs::write(
        PathBuf::from(&args[1]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    if failures > 0 {
        return Err("weak native moment discrepancy".into());
    }
    Ok(())
}
