//! Chapter 7 reference profiles and exact saved native companion/OU laws.
use algorithmic_gas::geometry::{AlgorithmicDistance, ComparisonKind, Distance, InteractionKernel};
use algorithmic_gas_benchmarks::{
    convergence_chapter07_completion::reference_suite,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

#[derive(Default)]
struct Aggregate {
    count: usize,
    residual: f64,
    variance: f64,
}
fn law(tag: &str) -> String {
    if let Some((case, rep)) = tag.split_once("-rep") {
        return format!(
            "{case}-{}",
            if rep.contains("-left-") {
                "left"
            } else if rep.contains("-right-") {
                "right"
            } else {
                "other"
            }
        );
    }
    if let Some((case, rep)) = tag.split_once("/rep") {
        return format!(
            "{case}-{}",
            if rep.contains("/left_") {
                "left"
            } else if rep.contains("/right_") {
                "right"
            } else {
                "other"
            }
        );
    }
    tag.into()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "gas-chapter07-completion EMPTY_OUTPUT COMPLETE_NATIVE_DATASET [DATASET ...]".into(),
        );
    }
    let mut out = ArchiveStore::new(&args[0])?;
    out.save_json("chapter07-source",&json!({
        "chapter":include_str!("../../../../docs/source/2_fractal_gas/convergence_program/07_discrete_qsd.md"),
        "inventory":include_str!("../../../../proof-validation/chapter07_inventory.json"),
        "module":include_str!("../convergence_chapter07_completion.rs"),"runner":include_str!("gas-chapter07-completion.rs"),
        "native_kinetic":include_str!("../../../algorithmic-gas/src/kinetic.rs"),
        "native_distance":include_str!("../../../algorithmic-gas/src/geometry.rs"),
        "native_companion":include_str!("../../../algorithmic-gas/src/donor.rs"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,"new_native_updates":0}))?;
    let references = reference_suite()?;
    let reference_failures = references.iter().filter(|r| r["passed"] != true).count();
    out.save_json("reference-law-fixtures", &json!({"checks":references}))?;
    let mut rows = vec![];
    let mut batches = vec![];
    let mut consumed = vec![];
    let mut groups: BTreeMap<(String, u64), Aggregate> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut frames = 0usize;
    let mut checked_coordinates = 0usize;
    let mut finite_pair_terms = 0usize;
    let mut exact_checks = 0usize;
    let mut exact_failures = 0usize;
    let mut max_ou_residual = 0_f64;
    let mut max_distance_residual = 0_f64;
    let mut max_distance_gap = 0_f64;
    for root in &args[1..] {
        let input = ArchiveStore::open(root)?;
        if !["complete", "completed"].contains(&input.status()) {
            return Err("only complete immutable input datasets".into());
        }
        out.save_json(&format!("input-index-{}", consumed.len()), &json!({
            "dataset":root,"index_sha256":sha256_file(&input.root().join("archive-index.json"))?,
            "status":input.status(),"retained_entries":input.entries().len(),"new_native_updates":0
        }))?;
        for entry in input
            .entries()
            .iter()
            .filter(|e| e.kind == "native_run_archive")
        {
            let archive = input.load_archive(&entry.path)?;
            let cfg = &archive.gas_config;
            let cv = serde_json::to_value(cfg)?;
            if cv["distance_donors"]["history_window"] != 0
                || cv["distance_donors"]["law"] != "independent"
                || cv["distance_donors"]["count"] != 1
                || cv["distance_donors"]["allow_self"] != false
            {
                return Err("current independent one-companion no-self law required".into());
            }
            let h = cv["kinetic"]["integrator"]["dt"]
                .as_f64()
                .ok_or("native h missing")?;
            let gamma = cv["kinetic"]["integrator"]["friction"]
                .as_f64()
                .ok_or("native gamma missing")?;
            let noise_scale = cv["kinetic"]["noise"]["geometry"]["scale"]["values"][0]
                .as_f64()
                .ok_or("native noise scale missing")?;
            if cv["kinetic"]["noise"]["innovation"] != "gaussian"
                || cv["kinetic"]["noise"]["geometry"]["kind"] != "isotropic"
            {
                return Err("standard native Gaussian OU noise required".into());
            }
            let c = (-gamma * h).exp();
            let variance_time = if gamma == 0. {
                h
            } else {
                -(-2. * gamma * h).exp_m1() / (2. * gamma)
            };
            let scale = variance_time.sqrt();
            let variance = noise_scale.powi(2) * variance_time;
            consumed.push(
                json!({"dataset":root,"path":entry.path,"sha256":entry.sha256,"native_config":cfg}),
            );
            for r in &archive.steps {
                let group = format!("{root}/{}", law(&entry.tag));
                if !seen.insert((group.clone(), cfg.seed, r.report.step)) {
                    return Err("duplicate law/seed/time input".into());
                }
                if r.before.eligible(false).iter().any(|a| !*a)
                    || r.report.distance_sources.len() != r.before.len()
                {
                    return Err("all alive, full current companion source pool required".into());
                }
                let n = r.before.len();
                let obs = &r.before.observations;
                let distance = &cfg.distance_donors.distance;
                let kind = <Distance as AlgorithmicDistance<f64>>::comparison_kind(distance);
                let mut expectation = 0.;
                let mut nearest = 0.;
                let mut realized = 0.;
                let mut maximum = 0.;
                let mut max_measurement_defect = 0_f64;
                let floor = cv["fitness"]["distance_floor"]
                    .as_f64()
                    .ok_or("actual distance floor missing")?;
                for i in 0..n {
                    let mut total = 0.;
                    let mut numerator = 0.;
                    let mut minimum = f64::INFINITY;
                    let mut largest = 0_f64;
                    for s in &r.report.distance_sources {
                        let j = s.slot as usize;
                        if i == j {
                            continue;
                        }
                        let raw = distance.compare(obs, i, obs, j)?;
                        let dist = if kind == ComparisonKind::SquaredDistance {
                            raw.sqrt()
                        } else {
                            raw
                        };
                        let weight = cfg.distance_donors.kernel.log_weight(raw, kind)?.exp();
                        total += weight;
                        numerator += dist * weight;
                        minimum = minimum.min(dist);
                        largest = largest.max(dist);
                        finite_pair_terms += 1;
                    }
                    if total <= 0. || !total.is_finite() {
                        return Err("finite companion denominator failed".into());
                    }
                    let mean = numerator / total;
                    exact_checks += 2;
                    exact_failures +=
                        usize::from(mean < minimum - 1e-11) + usize::from(mean > largest + 1e-11);
                    let pool = r
                        .report
                        .distance_companions
                        .row(i)
                        .next()
                        .ok_or("companion index missing")? as usize;
                    let j = r.report.distance_sources[pool].slot as usize;
                    let raw = distance.compare(obs, i, obs, j)?;
                    let dist = if kind == ComparisonKind::SquaredDistance {
                        raw.sqrt()
                    } else {
                        raw
                    };
                    let expected_separation = dist.hypot(floor);
                    max_measurement_defect = max_measurement_defect.max(
                        (expected_separation - r.report.pre_clone_fitness.separation[i]).abs(),
                    );
                    expectation += mean / n as f64;
                    nearest += minimum / n as f64;
                    realized += dist / n as f64;
                    maximum += largest / n as f64;
                }
                exact_checks += 1;
                exact_failures += usize::from(max_measurement_defect > 1e-9);
                max_distance_residual = max_distance_residual.max(max_measurement_defect);
                max_distance_gap = max_distance_gap.max(expectation - nearest);
                let stage = r
                    .stages
                    .iter()
                    .find(|s| s.stage == "A1")
                    .ok_or("OU input A1 absent")?;
                let output = r
                    .stages
                    .iter()
                    .find(|s| s.stage == "O_before_boundary")
                    .ok_or("OU output absent")?;
                let input_v = &stage.fields["velocities"];
                let output_v = &output.fields["velocities"];
                let noise = r
                    .field_evaluations
                    .iter()
                    .find(|f| f.stage == "O" && f.field == "executed_noise")
                    .ok_or("actual OU draw absent")?;
                let d = input_v.item_shape[0];
                let mut input_moment = 0.;
                let mut output_moment = 0.;
                let mut max_residual = 0_f64;
                for (i, ((&v, &w), &eta)) in input_v
                    .values
                    .iter()
                    .zip(&output_v.values)
                    .zip(&noise.values)
                    .enumerate()
                {
                    if !noise.available[i / d] {
                        return Err("conservative all-row Gaussian coverage missing".into());
                    }
                    let error = (w - (c * v + scale * eta)).abs();
                    max_residual = max_residual.max(error);
                    exact_checks += 1;
                    exact_failures += usize::from(error > 2e-10 * (1. + w.abs()));
                    checked_coordinates += 1;
                    input_moment += v * v / n as f64;
                    output_moment += w * w / n as f64;
                }
                max_ou_residual = max_ou_residual.max(max_residual);
                let conditional_second = c * c * input_moment + d as f64 * variance;
                let conditional_variance = (4. * c * c * variance * input_moment
                    + 2. * d as f64 * variance * variance)
                    / n as f64;
                let a = groups.entry((group.clone(), r.report.step)).or_default();
                a.count += 1;
                a.residual += output_moment - conditional_second;
                a.variance += conditional_variance;
                let stage_moment = |name: &str| -> Result<f64, Box<dyn std::error::Error>> {
                    let snapshot = r
                        .stages
                        .iter()
                        .find(|s| s.stage == name)
                        .ok_or("velocity balance stage missing")?;
                    Ok(snapshot.fields["velocities"]
                        .values
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        / n as f64)
                };
                let pre_clone_moment = stage_moment("pre_clone")?;
                let post_clone_moment = stage_moment("post_clone")?;
                let second_kick_moment = stage_moment("B2")?;
                let terminal_moment = stage_moment("terminal")?;
                let cap = cv["kinetic"]["velocity_cap"]
                    .as_f64()
                    .ok_or("actual velocity cap missing")?;
                exact_checks += 1;
                exact_failures += usize::from(terminal_moment > cap * cap + 1e-10);
                rows.push(json!({"dataset":root,"source_archive":entry.path,"source_sha256":entry.sha256,"law_group":group,"seed":cfg.seed,"step":r.report.step,"N":n,"d":d,
                    "companion_kernel_expected_distance":expectation,"nearest_neighbor_distance":nearest,"realized_kernel_distance":realized,"largest_candidate_distance":maximum,
                    "measurement_distance_defect":max_measurement_defect,"kernel_parameters":cfg.distance_donors,
                    "ou_input_N_second_moment":input_moment,"ou_output_N_second_moment":output_moment,
                    "ou_conditional_expected_N_second_moment":conditional_second,"ou_conditional_variance_of_N_moment":conditional_variance,
                    "ou_exact_coordinate_defect":max_residual,"ou_multiplier":c,"ou_variance_per_coordinate":variance,"h":h,"friction":gamma,
                    "reference_temperature":if gamma>0.{Some(noise_scale.powi(2)/(2.*gamma))}else{None},
                    "complete_velocity_balance":{"entering_moment":pre_clone_moment,"cloning_and_collision_increment":post_clone_moment-pre_clone_moment,
                        "first_force_kick_increment":input_moment-post_clone_moment,"OU_increment":output_moment-input_moment,
                        "second_force_kick_increment":second_kick_moment-output_moment,"cap_increment":terminal_moment-second_kick_moment,
                        "complete_output_N_second_moment":terminal_moment,"complete_output_upper":cap*cap,
                        "scope":"Exact observed full velocity moment decomposition; components are diagnostics with realized native interactions, not reset-rate parameters or stationary balance claims."},
                    "scope":"Actual current companion kernel and uncapped O substep conditional law. All N-normalized sums are permutation invariant. B kicks, collisions, cloning and final caps prevent identifying this conditional OU law with the completed stationary velocity marginal."}));
                frames += 1;
                if rows.len() >= 1024 {
                    batches.push(out.save_json(
                        &format!("native-law-batch{}", batches.len()),
                        &json!({"rows":rows}),
                    )?);
                    rows.clear();
                }
            }
        }
        eprintln!("Chapter 7 retained frames processed: {frames}");
    }
    if !rows.is_empty() {
        batches.push(out.save_json(
            &format!("native-law-batch{}", batches.len()),
            &json!({"rows":rows}),
        )?);
    }
    let delta = 0.01 / groups.len().max(1) as f64;
    let statistical:Vec<_>=groups.into_iter().map(|((law,step),a)| {
        let m=a.count as f64;let allowance=(((2.-delta)/delta)*a.variance).sqrt()/m;
        json!({"id":format!("{law}/step{step}/OU-second-moment"),"law_group":law,"step":step,
            "independent_seed_units":a.count,"mean_conditional_residual":a.residual/m,
            "cantelli_allowance":allowance,"passed":(a.residual/m).abs()<=allowance,
            "relation":"conditional expectation equality; two-sided family-adjusted Cantelli validation",
            "family_failure_budget":0.01,"per_comparison_failure_budget":delta,
            "hypotheses":"Each own-law seed is independent. Conditional exact Gaussian moments precede boundary/cap. Rows and timesteps are not independent replicas; each checkpoint is compared separately."})
    }).collect();
    let statistical_failures = statistical.iter().filter(|r| r["passed"] != true).count();
    let report = json!({"chapter":7,"summary":{"reference_checks":references.len(),"reference_failures":reference_failures,
        "retained_native_updates":frames,"new_native_updates":0,"finite_native_companion_pair_terms":finite_pair_terms,
        "exact_native_checks":exact_checks,"exact_native_failures":exact_failures,"OU_coordinates":checked_coordinates,
        "conditional_moment_checks":statistical.len(),"conditional_moment_failures":statistical_failures,
        "maximum_OU_coordinate_defect":max_ou_residual,"maximum_measured_distance_defect":max_distance_residual,
        "maximum_kernel_minus_nearest_distance":max_distance_gap},"statistical_comparisons":statistical,
        "consumed_sources":consumed,"raw_native_batches":batches,
        "scope":"Every Chapter 7 specified reference formula and conditional native counterpart is evaluated. Full selected native equilibrium/QSD shape, global contraction constants and mean-field consistency remain analytic hypotheses; no long finite sample is certified as stationary."});
    out.save_json("completion-report", &report)?;
    out.finish("complete")?;
    let verify = out.verify(true)?;
    fs::write(
        PathBuf::from(&args[0]).join("verification.json"),
        serde_json::to_vec_pretty(&verify)?,
    )?;
    fs::write(
        PathBuf::from(&args[0]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    if reference_failures + exact_failures + statistical_failures > 0 {
        return Err("Chapter 7 comparison failure; evidence retained".into());
    }
    Ok(())
}
