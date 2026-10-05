//! Exact conditional native one-donor/acceptance source moment integration.
use algorithmic_gas::geometry::{AlgorithmicDistance, Distance, InteractionKernel};
use algorithmic_gas_benchmarks::{
    convergence_chapter06_source_moment::{
        FitnessChannelRange, SourceMomentParameters, source_moment_envelope,
    },
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
fn number(v: &Value, p: &str) -> Result<f64, Box<dyn std::error::Error>> {
    v.pointer(p)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("Missing actual parameter {p}").into())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("gas-chapter06-source-integration COMPLETE_DATASET EMPTY_OUTPUT".into());
    }
    let input = ArchiveStore::open(&args[0])?;
    if !["complete", "completed"].contains(&input.status()) {
        return Err("committed complete input required".into());
    }
    let mut out = ArchiveStore::new(&args[1])?;
    out.save_json(
        "source-integration-source",
        &json!({"runner":include_str!("gas-chapter06-source-integration.rs"),
        "interface":include_str!("../convergence_chapter06_source_moment.rs"),
        "native_distance":include_str!("../../../algorithmic-gas/src/geometry.rs"),
        "native_gate":include_str!("../../../algorithmic-gas/src/cloning.rs"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,
        "input_index_sha256":sha256_file(&input.root().join("archive-index.json"))?}),
    )?;
    let mut rows = vec![];
    let mut batches = vec![];
    let mut consumed = vec![];
    let mut frames = 0usize;
    let mut positive = 0usize;
    let mut accepted = 0usize;
    let mut comparisons = 0usize;
    let mut max_incoming_residual = f64::NEG_INFINITY;
    let mut max_moment_residual = f64::NEG_INFINITY;
    let mut max_scaled_moment_residual = f64::NEG_INFINITY;
    let mut max_expected_acceptances = 0_f64;
    for e in input
        .entries()
        .iter()
        .filter(|e| e.kind == "native_run_archive")
    {
        let archive = input.load_archive(&e.path)?;
        let c = &archive.gas_config;
        let cfg = serde_json::to_value(c)?;
        if cfg["cloning_donors"]["history_window"] != 0
            || cfg["cloning_donors"]["law"] != "independent"
            || cfg["cloning_donors"]["count"] != 1
            || cfg["cloning_donors"]["allow_self"] != false
        {
            return Err("current independent no-self one-donor source required".into());
        }
        let channels = ["reward", "diversity"]
            .into_iter()
            .map(|name| -> Result<_, Box<dyn std::error::Error>> {
                if cfg["fitness"][format!("{name}_map")]["kind"] != "logistic" {
                    return Err("actual logistic fitness required".into());
                }
                Ok(FitnessChannelRange {
                    amplitude: number(&cfg, &format!("/fitness/{name}_map/amplitude"))?,
                    positive_floor: number(&cfg, &format!("/fitness/{name}_map/floor"))?,
                    exponent: number(&cfg, &format!("/fitness/{name}_exponent"))?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let envelope = source_moment_envelope(&SourceMomentParameters {
            position_feature_radius: number(&cfg, "/cloning_donors/distance/position_radius")?,
            velocity_feature_radius: number(&cfg, "/cloning_donors/distance/velocity_radius")?,
            algorithmic_velocity_weight: number(&cfg, "/cloning_donors/distance/lambda")?,
            cloning_bandwidth: number(&cfg, "/cloning_donors/kernel/width")?,
            gate_saturation: c.clone_decision.saturation,
            gate_epsilon: c.clone_decision.epsilon,
            historical_window: 0,
            component_collision_enabled: c.clone_transform.restitution.is_some(),
            fitness_channels: channels,
        })?;
        consumed.push(json!({"path":e.path,"sha256":e.sha256,"tag":e.tag}));
        for r in &archive.steps {
            let n = r.before.len();
            if n < 2
                || r.before.eligible(false).iter().any(|x| !*x)
                || r.elite_injection.is_some()
                || r.report.revivals != 0
            {
                return Err("all-alive conservative source input required".into());
            }
            let source = &r.report.clone_plan.sources;
            if source.len() != n || source.iter().any(|s| s.frame != source[0].frame) {
                return Err("current complete source pool required".into());
            }
            let mut slots = source.iter().map(|s| s.slot as usize).collect::<Vec<_>>();
            slots.sort_unstable();
            if slots != (0..n).collect::<Vec<_>>() {
                return Err("current pool slot permutation invalid".into());
            }
            let obs = &r.before.observations;
            let kind =
                <Distance as AlgorithmicDistance<f64>>::comparison_kind(&c.cloning_donors.distance);
            let own = &r.report.pre_clone_fitness.fitness;
            let donor = &r.donor_fitness;
            for (j, s) in source.iter().enumerate() {
                if donor[j] != own[s.slot as usize] {
                    return Err(
                        "current donor fitness is not its actual frozen measurement fitness".into(),
                    );
                }
            }
            let mut accepted_transfer = vec![0.; n * n];
            let mut incoming = vec![0.; n];
            let mut expectation = 0.;
            for i in 0..n {
                let mut weights = vec![0.; n];
                let mut total = 0.;
                for (pool, s) in source.iter().enumerate() {
                    let j = s.slot as usize;
                    if i == j {
                        continue;
                    }
                    let distance = c.cloning_donors.distance.compare(obs, i, obs, j)?;
                    let weight = c.cloning_donors.kernel.log_weight(distance, kind)?.exp();
                    weights[pool] = weight;
                    total += weight;
                }
                if !total.is_finite() || total <= 0. {
                    return Err("actual donor normalization invalid".into());
                }
                for (pool, s) in source.iter().enumerate() {
                    let j = s.slot as usize;
                    if i == j {
                        continue;
                    }
                    let a =
                        c.clone_decision
                            .acceptance_probability(r.report.step, own[i], donor[pool]);
                    if a > envelope.acceptance_upper + 1e-12 {
                        return Err("actual pair gate exceeds global range".into());
                    }
                    let q = weights[pool] / total * a;
                    if q > 0. {
                        positive += 1;
                    }
                    accepted_transfer[i * n + j] = q;
                    incoming[j] += q;
                    expectation += q;
                }
                for pool in r.report.cloning_companions.row(i) {
                    let computed = c.clone_decision.acceptance_probability(
                        r.report.step,
                        own[i],
                        donor[pool as usize],
                    );
                    if (r.report.clone_plan.choices[i].probability.unwrap() - computed).abs()
                        > 1e-12
                    {
                        return Err(
                            "sampled recorded gate differs from native integrated gate".into()
                        );
                    }
                }
            }
            let gain = envelope.current_frame_source_moment_coefficient - 1.;
            let mut source_moments = vec![];
            for p in [4usize, 8] {
                let norms = obs
                    .field("positions")?
                    .values()
                    .chunks_exact(obs.field("positions")?.width())
                    .map(|x| x.iter().map(|v| v * v).sum::<f64>().powf(p as f64 / 2.))
                    .collect::<Vec<_>>();
                let before = norms.iter().sum::<f64>() / n as f64;
                let mut expected = before;
                for i in 0..n {
                    for j in 0..n {
                        expected += accepted_transfer[i * n + j] * (norms[j] - norms[i]) / n as f64;
                    }
                }
                let bound = (1. + gain) * before;
                let residual = expected - bound;
                max_moment_residual = max_moment_residual.max(residual);
                max_scaled_moment_residual =
                    max_scaled_moment_residual.max(residual / bound.max(1.));
                comparisons += 1;
                if residual > 1e-11 * bound.max(1.) {
                    return Err(
                        "exact conditional native source expectation violates uniform moment gain"
                            .into(),
                    );
                }
                source_moments.push(json!({"p":p,"input":before,"exact_expected_source":expected,"bound":bound,"signed_residual":residual,"passed":true}));
            }
            for load in &incoming {
                let residual = load - gain;
                max_incoming_residual = max_incoming_residual.max(residual);
                comparisons += 1;
                if residual > 1e-12 {
                    return Err("actual incoming accepted-source load exceeds amax/kappa".into());
                }
            }
            rows.push(json!({"seed":c.seed,"step":r.report.step,"N":n,"source_archive":e.path,"source_sha256":e.sha256,
                "actual_accepted_clones":r.report.clones,"exact_expected_accepted_clones_given_fitness":expectation,
                "actual_frozen_fitness":own,"incoming_accepted_load":incoming,"incoming_uniform_upper":gain,
                "source_moments":source_moments}));
            max_expected_acceptances = max_expected_acceptances.max(expectation);
            frames += 1;
            accepted += r.report.clones;
            if rows.len() >= 1024 {
                batches.push(out.save_json(
                    &format!("integration-batch{}", batches.len()),
                    &json!({"rows":rows}),
                )?);
                rows.clear();
            }
        }
    }
    if !rows.is_empty() {
        batches.push(out.save_json(
            &format!("integration-batch{}", batches.len()),
            &json!({"rows":rows}),
        )?);
    }
    let report = json!({"chapter":6,"comparisons":[
        {"id":"exact_native_incoming_load_max_residual","observed":max_incoming_residual,"bound":1e-12,"relation":"upper","passed":max_incoming_residual<=1e-12,
         "scope":"Maximum over explicitly retained incoming-load comparisons; global analytic upper supplied by actual kernel/fitness ranges, not inferred from sampled maximum."},
        {"id":"exact_native_source_p4_p8_max_scaled_residual","observed":max_scaled_moment_residual,"bound":1e-11,"relation":"upper","passed":max_scaled_moment_residual<=1e-11,
         "scope":"Maximum of exact conditional finite-sum p4/p8 moment residuals; each native frame checked at its own scale. No new Gaussian or donor samples."}],
        "summary":{"native_frames":frames,"exact_conditional_comparisons":comparisons,"comparisons_failed":0,
        "strictly_positive_pair_acceptance_contributions":positive,"sampled_accepted_clones":accepted,"new_native_steps":0},
        "maximum_incoming_load_residual":max_incoming_residual,"maximum_moment_residual":max_moment_residual,
        "maximum_exact_conditional_expected_acceptances":max_expected_acceptances,"raw_batches":batches,"consumed_archives":consumed,
        "arithmetic_scope":"Binary64 finite native API sums, with explicit 1e-12 incoming/gate and 1e-11 relative moment roundoff tolerances; not directed interval arithmetic.",
        "scope":"Exact finite sums using native Distance/Kernel/CloneDecision APIs conditional on actual sampled complete fitness. Donor/acceptance integration averages every i!=j, retaining full nonlinear measurement before conditioning. Incoming-load/moment comparisons deterministic; no statistical allowance, unobserved event assertion or whole-measurement marginalization. Swarm sums invariant under permutation; storage row indices have no particle identity."});
    out.save_json("source-integration-report", &report)?;
    out.finish("complete")?;
    fs::write(
        PathBuf::from(&args[1]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    Ok(())
}
