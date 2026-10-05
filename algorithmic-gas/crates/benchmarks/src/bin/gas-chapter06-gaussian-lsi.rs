//! Fixed-checkpoint conditional Gaussian LSI tests from actual saved innovations.
//! A conditional candidate law is not a stationary/killed alive law.
use algorithmic_gas_benchmarks::convergence_experiments::{ArchiveStore, sha256_file};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

fn law(tag: &str) -> String {
    if let Some((case, rest)) = tag.split_once("-distribution") {
        return format!("{case}-distribution{}", rest.split('-').next().unwrap());
    }
    if let Some((case, rest)) = tag.split_once("-rep") {
        return format!(
            "{case}-{}",
            if rest.contains("-left-") {
                "left"
            } else if rest.contains("-right-") {
                "right"
            } else {
                "unspecified"
            }
        );
    }
    if let Some((case, rest)) = tag.split_once("/rep") {
        return format!(
            "{case}-{}",
            if rest.contains("/left_") {
                "left"
            } else if rest.contains("/right_") {
                "right"
            } else {
                "unspecified"
            }
        );
    }
    tag.into()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err(
            "gas-chapter06-gaussian-lsi INPUT_DATASET EMPTY_OUTPUT_DIR FIXED_CHECKPOINT".into(),
        );
    }
    let checkpoint: u64 = args[2].parse()?;
    if checkpoint == 0 {
        return Err("fixed positive update checkpoint required".into());
    }
    let input = ArchiveStore::open(&args[0])?;
    if !["complete", "completed"].contains(&input.status()) {
        return Err("input dataset must be complete".into());
    }
    let mut output = ArchiveStore::new(&args[1])?;
    output.save_json("gaussian-lsi-source",&json!({"runner":include_str!("gas-chapter06-gaussian-lsi.rs"),
        "chapter":include_str!("../../../../../docs/source/2_fractal_gas/convergence_program/06_convergence.md"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,"input_index_sha256":sha256_file(&input.root().join("archive-index.json"))?,
        "checkpoint":checkpoint,"native_steps_added":0,
        "scope":"Gaussian LSI rho=1/sigma² conditional on complete pre-final-noise preparation, tested on a bounded N-normalized candidate observable; no stationary/QSD or alive-status LSI claim."}))?;
    let mut units: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut selected = vec![];
    for e in input
        .entries()
        .iter()
        .filter(|e| e.kind == "native_run_archive")
    {
        let seed = e.metadata["native_config"]["seed"]
            .as_u64()
            .ok_or("indexed native seed missing")?;
        units.entry(law(&e.tag)).or_default().insert(seed);
        let first = e.metadata["first_step"]
            .as_u64()
            .ok_or("first recorded step missing")?;
        let last = e.metadata["last_step"]
            .as_u64()
            .ok_or("last recorded step missing")?;
        if first <= checkpoint && checkpoint <= last {
            selected.push(e);
        }
    }
    let mut rows = vec![];
    let mut consumed = vec![];
    let mut seen = BTreeSet::new();
    let mut events: BTreeMap<(String, u32), (usize, usize)> = BTreeMap::new();
    for e in selected {
        let a = input.load_archive(&e.path)?;
        let r = a
            .steps
            .iter()
            .find(|r| r.report.step == checkpoint)
            .ok_or("checkpoint native record missing")?;
        let group = law(&e.tag);
        if !seen.insert((group.clone(), a.gas_config.seed)) {
            return Err("duplicate fixed-checkpoint trajectory".into());
        }
        let Some(noise) = r
            .field_evaluations
            .iter()
            .find(|f| f.stage == "position_diffusion" && f.field == "executed_noise")
        else {
            continue;
        };
        let n = r.before.len();
        let d = noise.item_shape[0];
        let cfg = serde_json::to_value(&a.gas_config)?;
        let h = cfg["kinetic"]["integrator"]["dt"]
            .as_f64()
            .ok_or("actual h missing")?;
        let scale = cfg["kinetic"]["position_diffusion"]
            .as_f64()
            .ok_or("actual final noise missing")?;
        let sigma = scale * h.sqrt();
        if sigma <= 0. {
            return Err("positive final Gaussian variance required for LSI".into());
        }
        let k = noise.available.iter().filter(|a| **a).count();
        let mean_noise = noise
            .available
            .iter()
            .enumerate()
            .filter(|(_, available)| **available)
            .map(|(i, _)| noise.values[i * d])
            .sum::<f64>()
            / n as f64;
        let observed = (sigma * mean_noise).clamp(-1., 1.);
        consumed.push(json!({"path":e.path,"sha256":e.sha256,"tag":e.tag}));
        for multiple in [1u32, 2, 3] {
            let threshold = multiple as f64 * sigma / (n as f64).sqrt();
            let event = observed.abs() >= threshold;
            let counts = events.entry((group.clone(), multiple)).or_default();
            counts.0 += usize::from(event);
            counts.1 += 1;
            rows.push(json!({"law_group":group,"seed":a.gas_config.seed,"step":checkpoint,"N":n,"d":d,
                "gaussian_coordinates_used":k,"rho_conditional_gaussian_lsi":1./sigma.powi(2),
                "observable":"clip(N^-1 sum_i (X_i,1−prepared_mean_i,1),[-1,1]), only Gaussian-covered rows",
                "L":1.,"gradient_squared_upper":k as f64/(n*n) as f64,
                "expected_observable_conditional":0.,"observed":observed,"threshold":threshold,
                "event":event,"concentration_upper":(2.*(-0.5*(multiple*multiple) as f64).exp()).min(1.),
                "source_labels":["thm-convergence-lsi-concentration"],
                "source_archive":e.path,"source_sha256":e.sha256,
                "scope":"Actual final candidate Gaussian law given its unrestricted prepared mean; observable is bounded and permutation invariant. Alive restriction/conditioning and global stationary/QSD LSI remain separate."}));
        }
    }
    let delta = 0.01 / events.len().max(1) as f64;
    let log = (1. / delta).ln();
    let mut comparisons = vec![];
    for ((group, multiple), (count, prepared)) in events {
        let total = units[&group].len();
        let m = total as f64;
        let per_seed = (2. * (-0.5 * (multiple * multiple) as f64).exp()).min(1.);
        let prediction = prepared as f64 * per_seed / m;
        let allowance = (2. * prepared as f64 * per_seed * log).sqrt() / m + 2. * log / (3. * m);
        let observed = count as f64 / m;
        comparisons.push(json!({"id":format!("{group}-c{multiple}"),"law_group":group,
            "independent_seed_units":total,"prepared_seed_units":prepared,"absorbed_zero_seed_units":total-prepared,
            "observed":observed,"bound":prediction,"bernstein_allowance":allowance,
            "passed":observed<=prediction+allowance,"signed_residual":observed-prediction,
            "source_labels":["thm-convergence-lsi-concentration"],
            "scope":"Conditional Gaussian concentration under actual full native preparation at a fixed externally chosen checkpoint; own initial-law denominator restored with absorbed-zero units. Does not certify a full-swarm stationary/QSD LSI."}));
    }
    let failures = comparisons.iter().filter(|c| c["passed"] != true).count();
    output.save_json("gaussian-lsi-raw-observations", &json!({"rows":rows}))?;
    let report = json!({"chapter":6,"checkpoint":checkpoint,"comparisons":comparisons,"consumed_archives":consumed,
        "family_failure_budget":0.01,"summary":{"independent_law_groups":units.len(),"comparisons":comparisons.len(),"comparisons_failed":failures,"native_steps_added":0},
        "scope":"Analytic conditional Gaussian LSI=1/sigma² with full native recorded innovations, bounded candidate observable, correct N normalization, fixed time, independent seed units and each own denominator. No stationary/QSD joint LSI claim."});
    output.save_json("gaussian-lsi-report", &report)?;
    output.finish("complete")?;
    fs::write(
        PathBuf::from(&args[1]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    if failures > 0 {
        return Err("conditional Gaussian concentration comparison failed".into());
    }
    Ok(())
}
