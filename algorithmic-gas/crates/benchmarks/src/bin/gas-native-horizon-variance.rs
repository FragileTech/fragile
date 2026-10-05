//! Independent-replica, permutation-invariant full native finite-horizon diagnostics.
//! This consumes immutable actual updates; it does not fit a theorem constant.
use algorithmic_gas_benchmarks::{
    convergence_chapter08_completion::bounded_observables,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

type Key = (String, usize, usize, String, String, u64);
type MatchKey = (String, usize, String, String, u64);

fn mean_variance(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (mean, variance)
}

fn mean_variance_se(values: &[f64]) -> (f64, f64, f64, f64) {
    let (mean, variance) = mean_variance(values);
    let leave_one_out: Vec<f64> = (0..values.len())
        .map(|skip| {
            let omitted: Vec<f64> = values
                .iter()
                .enumerate()
                .filter_map(|(i, value)| (i != skip).then_some(*value))
                .collect();
            mean_variance(&omitted).1
        })
        .collect();
    let average = leave_one_out.iter().sum::<f64>() / values.len() as f64;
    let jackknife_variance = (values.len() - 1) as f64 / values.len() as f64
        * leave_one_out
            .iter()
            .map(|value| (value - average).powi(2))
            .sum::<f64>();
    (
        mean,
        variance,
        (variance / values.len() as f64).sqrt(),
        jackknife_variance.sqrt(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("gas-native-horizon-variance EMPTY_OUTPUT PROFILE_DATASET PROFILE_DATASET PROFILE_DATASET".into());
    }
    let mut out = ArchiveStore::new(&args[0])?;
    out.save_json("source", &json!({
        "runner": include_str!("gas-native-horizon-variance.rs"),
        "observables": include_str!("../convergence_chapter08_completion.rs"),
        "chapter09": include_str!("../../../../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"),
        "binary_sha256": sha256_file(&std::env::current_exe()?)?,
        "new_native_updates": 0,
        "statistical_unit": "one independently seeded complete native trajectory, within fixed profile/dimension/N/zone/side. Left and right are never combined.",
    }))?;
    let mut groups: BTreeMap<Key, Vec<[f64; 7]>> = BTreeMap::new();
    let mut seeds: BTreeMap<Key, BTreeSet<u64>> = BTreeMap::new();
    let mut provenance = vec![];
    let mut recorded_steps = 0usize;
    let mut trajectories = 0usize;
    let mut source_configs = BTreeMap::new();
    for path in &args[1..] {
        let input = ArchiveStore::open(path)?;
        if input.status() != "complete" {
            return Err("complete immutable native archives required".into());
        }
        let profile = input
            .root()
            .file_name()
            .ok_or("profile missing")?
            .to_str()
            .ok_or("profile not UTF8")?
            .to_owned();
        provenance.push(json!({"profile":profile,"dataset":path,
            "index_sha256":sha256_file(&input.root().join("archive-index.json"))?}));
        for entry in input
            .entries()
            .iter()
            .filter(|entry| entry.kind == "native_run_archive")
        {
            let archive = input.load_archive(&entry.path)?;
            let cfg = serde_json::to_value(&archive.gas_config)?;
            for donor in ["distance_donors", "cloning_donors"] {
                if cfg[donor]["history_window"] != 0
                    || cfg[donor]["count"] != 1
                    || cfg[donor]["law"] != "independent"
                    || cfg[donor]["allow_self"] != false
                {
                    return Err(
                        "canonical independent current one-donor configuration required".into(),
                    );
                }
            }
            if cfg["boundary"]["kind"] != "unbounded"
                || !cfg["qft"]["viscosity"].is_null()
                || !cfg["qft"]["graph_viscosity"].is_null()
            {
                return Err("full-alive unbounded row-local kinetic trajectory required".into());
            }
            if archive.steps.len() != 16 {
                return Err("expected retained16-step trajectory".into());
            }
            let (case, suffix) = entry.tag.split_once("-rep").ok_or("case tag missing")?;
            let side = if suffix.contains("-left-") {
                "left"
            } else if suffix.contains("-right-") {
                "right"
            } else {
                return Err("paired side missing".into());
            };
            let replicate = suffix
                .split('-')
                .next()
                .ok_or("replicate missing")?
                .parse::<usize>()?;
            let zone = case.rsplit('-').next().ok_or("zone missing")?.to_owned();
            let n = archive.steps[0].before.len();
            let d = archive.steps[0]
                .before
                .observations
                .field("positions")?
                .width();
            if ![4, 64].contains(&n) || ![1, 2, 4].contains(&d) {
                return Err("expected N4/64, d1/2/4 retained grid".into());
            }
            let mut signature = cfg.clone();
            signature["seed"] = Value::Null;
            let config_key = (profile.clone(), d, zone.clone(), side.to_string());
            if let Some(previous) = source_configs.insert(config_key, signature.clone())
                && previous != signature
            {
                return Err("matched population grid changed native configuration".into());
            }
            let mut observations = vec![];
            for step in &archive.steps {
                if step.before.eligible(false).iter().any(|alive| !alive)
                    || step
                        .final_population
                        .eligible(false)
                        .iter()
                        .any(|alive| !alive)
                {
                    return Err(
                        "physical alive law must have exact mass1 in these unbounded inputs".into(),
                    );
                }
                let values = bounded_observables(&step.final_population)?;
                if values[3] != 1. {
                    return Err("marked alive mass changed".into());
                }
                let key = (
                    profile.clone(),
                    d,
                    n,
                    zone.clone(),
                    side.to_string(),
                    step.report.step,
                );
                if !seeds
                    .entry(key.clone())
                    .or_default()
                    .insert(archive.gas_config.seed)
                {
                    return Err("duplicate seed within independent-replica group".into());
                }
                groups.entry(key).or_default().push(values);
                observations.push(json!({"step":step.report.step,"values":values}));
                recorded_steps += 1;
            }
            out.save_json(&format!("{profile}-{case}-rep{replicate}-{side}"), &json!({
                "profile":profile,"case":case,"side":side,"replicate":replicate,
                "N":n,"d":d,"zone":zone,"seed":archive.gas_config.seed,
                "native_configuration":cfg,"upstream_dataset":path,
                "upstream_path":entry.path,"upstream_sha256":entry.sha256,
                "initial":bounded_observables(&archive.steps[0].before)?,"observations":observations,
            }))?;
            trajectories += 1;
        }
    }
    let mut rows = vec![];
    let mut matched: BTreeMap<MatchKey, BTreeMap<usize, Value>> = BTreeMap::new();
    for ((profile, d, n, zone, side, time), values) in groups {
        if values.len() != 16 {
            return Err("exactly16 independent trajectories per conditional group required".into());
        }
        let summaries: Vec<_> = (0..7)
            .map(|test| {
                let measurements: Vec<_> = values.iter().map(|v| v[test]).collect();
                let (mean, variance, mean_se, variance_se) = mean_variance_se(&measurements);
                json!({"test":test,"mean":mean,"variance":variance,
                "N_times_variance":n as f64*variance,"mean_standard_error":mean_se,
                "variance_jackknife_standard_error":variance_se})
            })
            .collect();
        let row = json!({"profile":profile,"d":d,"N":n,"zone":zone,"side":side,
            "time_step":time,"independent_trajectories":16,"tests":summaries});
        matched
            .entry((profile, d, zone, side, time))
            .or_default()
            .insert(n, row.clone());
        rows.push(row);
    }
    let mut slopes = vec![];
    for ((profile, d, zone, side, time), pair) in matched {
        let low = pair.get(&4).ok_or("matched N4 missing")?;
        let high = pair.get(&64).ok_or("matched N64 missing")?;
        for test in 0..7 {
            let v4 = low["tests"][test]["variance"]
                .as_f64()
                .ok_or("variance4 missing")?;
            let v64 = high["tests"][test]["variance"]
                .as_f64()
                .ok_or("variance64 missing")?;
            let slope = (v4 > 0. && v64 > 0.).then(|| (v64 / v4).ln() / 16_f64.ln());
            slopes.push(json!({"profile":profile,"d":d,"zone":zone,"side":side,
                "time_step":time,"test":test,"variance_N4":v4,"variance_N64":v64,
                "empirical_log_variance_slope":slope,"empirical_N_times_variance_N4":4.*v4,
                "empirical_N_times_variance_N64":64.*v64,
                "scope":"Two retained sizes; independent full-run replication within each side, with16 seeds. This diagnostic does not estimate an iterated mean-field bias or prove an N-uniform finite-horizon theorem constant."}));
        }
    }
    let report = json!({"summary":{"trajectories":trajectories,"retained_native_updates":recorded_steps,
        "new_native_updates":0,"conditional_groups":rows.len(),"paired_N_diagnostics":slopes.len()},
        "upstream":provenance,"observables":["sin(x1)","cos(x1)","|x|²/(1+|x|²)","alive mass","alive sin submass","alive cos submass","alive tail submass"],
        "scope":"Empirical finite-horizon native variance only. Exact full-alive unbounded law uses1/N=1/k. All times along a trajectory are correlated; means and variances use16 independently seeded complete runs. No conditional-on-random-current-state one-step variance is inferred. C_T and iterated mean-field bias remain analytic conditions.",
        "groups":rows,"N_diagnostics":slopes});
    out.save_json("completion-report", &report)?;
    fs::write(
        out.root().join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    out.finish("complete")?;
    let verification = out.verify(true)?;
    fs::write(
        out.root().join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!("{}", report["summary"]);
    Ok(())
}
