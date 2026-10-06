//! Analyze the existing regional KUR constants on retained native source frames.
#![recursion_limit = "256"]
use algorithmic_gas_benchmarks::{
    convergence_experiments::sha256_file, convergence_landscape_phase as phase,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};
fn source_statement(label: &str) -> String {
    let s = include_str!(
        "../../../../docs/source/2_fractal_gas/convergence_program/18a_keystone_uniform_coupled.md"
    );
    statement(s, label)
}
fn statement(s: &str, label: &str) -> String {
    let start = s.find(&format!(":label: {label}")).unwrap();
    let begin = s[..start].rfind(":::{prf:").unwrap();
    let end = start + s[start..].find("\n:::").unwrap() + 4;
    s[begin..end].to_string()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 3 {
        return Err("gas-landscape-phase FIXTURES.json RETAINED_FRAMES.json OUTPUT.json".into());
    }
    let fixtures: Value = serde_json::from_str(&fs::read_to_string(&a[0])?)?;
    let frames: Value = serde_json::from_str(&fs::read_to_string(&a[1])?)?;
    let mut comparisons = vec![];
    for f in fixtures["cases"]
        .as_array()
        .ok_or("Missing independent Python fixtures")?
    {
        let p = serde_json::from_value(f["parameters"].clone())?;
        let observed = phase::regional_bound(&p)?;
        for key in [
            "mean_map_squared",
            "accepted_coordinate_variance",
            "gate_refresh_bound",
            "positional_coefficient",
            "conservative_floor",
        ] {
            let x = observed[key]
                .as_f64()
                .ok_or("Unexpected failed Python fixture")?;
            let y = f["expected"][key]
                .as_f64()
                .ok_or("Missing Python expected value")?;
            comparisons.push(json!({"fixture":f["id"],"constant":key,"observed":x,"bound":y,"relation":"equal","numerical_tolerance":2e-12*(1.+x.abs()+y.abs()),"passed":(x-y).abs()<=2e-12*(1.+x.abs()+y.abs()),"scope":"Independent existing Python function vs Rust port; formula equality, no empirical convergence inferred."}));
        }
        let tightened = phase::tightened_regional_bound(&p)?;
        for (index, q) in f["full_gaussian_quadrature"]
            .as_array()
            .ok_or("Missing independent full-Gaussian quadrature")?
            .iter()
            .enumerate()
        {
            let x = tightened["exact_variance_endpoint_values"][1 - index]
                .as_f64()
                .ok_or("Failed refined fixture")?;
            let y = q["variance"]
                .as_f64()
                .ok_or("Missing quadrature variance")?;
            comparisons.push(json!({"fixture":f["id"],"constant":"exact_KUL8_endpoint_variance","endpoint":index,"observed":x,"bound":y,"relation":"equal","numerical_tolerance":3e-12,"passed":(x-y).abs()<=3e-12,"scope":"Independent SciPy integration over the entire Gaussian line vs exact native-force variance; cubature/MonteCarlo not used."}));
        }
        let lambda = tightened["positional_coefficient"].as_f64().unwrap();
        let old = observed["positional_coefficient"].as_f64().unwrap();
        comparisons.push(json!({"fixture":f["id"],"constant":"nested_analytic_geometry_multiplier","observed":lambda,"bound":old,"relation":"less_equal","passed":lambda<=old,"scope":"Nested analytic curvature intervals, without using sampled root locations. Floor is recomputed and not assumed monotone."}));
    }
    let mut profile_checks = vec![];
    let profile = phase::rastrigin_regional_profile(-2, 2)?;
    for (key, expected) in [
        ("stable_roots", &fixtures["profile"]["stable_roots"]),
        ("barriers", &fixtures["profile"]["barriers"]),
    ] {
        for (index, y) in expected
            .as_array()
            .ok_or("Missing Python roots")?
            .iter()
            .enumerate()
        {
            let x = profile[key][index]["value"].as_f64().unwrap();
            let y = y.as_f64().unwrap();
            profile_checks.push(json!({"kind":key,"index":index,"observed":x,"bound":y,"relation":"equal","passed":(x-y).abs()<2e-12,"scope":"Existing SciPy brentq vs Rust bisection root diagnostic inside proved source brackets."}));
        }
    }
    let mut rows = vec![];
    for frame in frames["frames"]
        .as_array()
        .ok_or("Missing retained source frames")?
    {
        let d = frame["dimension"].as_u64().ok_or("Missing dimension")? as usize;
        let p = phase::native_parameters(&frame["gas_config"], d)?;
        let positions: Vec<f64> = serde_json::from_value(frame["frozen_source_positions"].clone())?;
        let eligibility = phase::source_core_eligibility(&positions, d, &p)?;
        let velocities: Vec<f64> =
            serde_json::from_value(frame["frozen_source_velocities"].clone())?;
        if velocities.len() != positions.len() || velocities.iter().any(|x| !x.is_finite()) {
            return Err("Incomplete finite source velocity records".into());
        }
        let maximum_source_speed = velocities
            .chunks(d)
            .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt())
            .fold(0., f64::max);
        let velocity_cap_premise = maximum_source_speed <= p.velocity_cap;
        let mut maximum = 0_f64;
        for force in frame["force_records"]
            .as_array()
            .ok_or("Missing actual force records")?
        {
            let x: Vec<f64> = serde_json::from_value(force["positions"].clone())?;
            let g: Vec<f64> = serde_json::from_value(force["gradient"].clone())?;
            if x.len() != g.len() || x.is_empty() {
                return Err("Incomplete force query".into());
            }
            for (x, g) in x.into_iter().zip(g) {
                maximum = maximum.max(
                    (g - 2. * x - 20. * std::f64::consts::PI * (std::f64::consts::TAU * x).sin())
                        .abs(),
                );
            }
        }
        let force_matches = maximum <= 1e-10
            && frame["providers"]["gradient"] == "analytic-gradient/Rastrigin/Minimize/v1";
        let eligible = eligibility["all_sources_in_one_core"] == true
            && frame["all_entering_rows_eligible"] == true;
        let mut bound = phase::regional_bound(&p)?;
        let mut tightened = phase::tightened_regional_bound(&p)?;
        let applicable =
            bound["applicable"] == true && eligible && force_matches && velocity_cap_premise;
        if !applicable {
            bound["physical_multiplier_rate"] = Value::Null;
            tightened["physical_multiplier_rate"] = Value::Null;
        }
        rows.push(json!({"group":frame["group"],"replicate":frame["replicate"],"first_recorded_step":frame["first_recorded_step"],"zero_jitter_source_variance":frame["zero_jitter_source_variance"],"completed_positional_variance":frame["completed_positional_variance"],"id":frame["id"],"artifact":frame["artifact"],"config_parameters":p,"source_eligibility":eligibility,"actual_force_residual":maximum,"force_matches_native_Rastrigin":force_matches,"maximum_source_speed":maximum_source_speed,"entering_velocity_cap_premise":velocity_cap_premise,"source_bound_applicable_to_recorded_frame":applicable,"bound":bound,"tightened_bound":tightened,"scope":"Complete eligible frozen source rows must lie in one core before sampled preparation. Actual force evaluated at both retained kick inputs; crossings remain unrestricted, and terminal/source-pressure accounting is separate."}));
    }
    let mut groups: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for row in &rows {
        if row["first_recorded_step"] == 1 {
            groups
                .entry(row["group"].as_str().ok_or("Missing group")?.to_string())
                .or_default()
                .push(row);
        }
    }
    let mut empirical = vec![];
    for (name, group) in groups {
        let applicable = group.len() >= 2
            && group
                .iter()
                .all(|r| r["source_bound_applicable_to_recorded_frame"] == true);
        if !applicable {
            empirical.push(json!({"group":name,"independent_left_replicas":group.len(),"applicable":false,"reason":"All complete entering frames must satisfy the actual force, single-core and capped-speed premises; at least two independent left replicas required."}));
            continue;
        }
        for kind in ["bound", "tightened_bound"] {
            let lambda = group[0][kind]["positional_coefficient"].as_f64().unwrap();
            let floor = group[0][kind]["conservative_floor"].as_f64().unwrap();
            if group.iter().any(|r| {
                r[kind]["positional_coefficient"].as_f64() != Some(lambda)
                    || r[kind]["conservative_floor"].as_f64() != Some(floor)
            }) {
                return Err("Inconsistent native ensemble parameters".into());
            }
            let residuals: Vec<_> = group
                .iter()
                .map(|r| {
                    r["completed_positional_variance"].as_f64().unwrap()
                        - lambda * r["zero_jitter_source_variance"].as_f64().unwrap()
                })
                .collect();
            let m = residuals.len() as f64;
            let mean = residuals.iter().sum::<f64>() / m;
            let se =
                (residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (m - 1.) / m).sqrt();
            let passed = mean <= floor + 6. * se + 2e-12;
            empirical.push(json!({"group":name,"variant":kind,"independent_left_replicas":group.len(),"applicable":true,"observed":mean,"bound":floor,"standard_error":se,"relation":"less_equal","passed":passed,"positional_coefficient":lambda,"scope":"Unconditioned complete first-step replicas with deterministic eligible entering core; compares E[W_N(X+) - lambda W_N(mu)] to Creg, retains actual sampled source variance and full Gaussian draws. Paired right sides and repeated later frames do not add observations."}));
        }
    }
    let failed = comparisons
        .iter()
        .chain(profile_checks.iter())
        .filter(|x| x["passed"] != true)
        .count()
        + empirical
            .iter()
            .filter(|r| r["applicable"] == true && r["passed"] != true)
            .count();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../docs/source/2_fractal_gas/convergence_program/18a_keystone_uniform_coupled.md",
    );
    let refined_source=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md");
    let refined_binding = json!({"source_path":refined_source,"source_sha256":sha256_file(&refined_source)?,"source_labels":["cor-slc-exact-jitter-regional-refinement"],"source_quotes":[statement(&fs::read_to_string(&refined_source)?,"cor-slc-exact-jitter-regional-refinement")]});
    let out = json!({"empirical_KUR_comparisons":empirical,"refined_current_source":refined_binding,"native_steps":0,"source_path":source,"source_sha256":sha256_file(&source)?,"source_labels":["thm-kur-evaluated-regional-bound","lem-klq-rastrigin-profiles"],"source_quotes":[source_statement("thm-kur-evaluated-regional-bound"),source_statement("lem-klq-rastrigin-profiles")],"python_fixture_path":a[0],"python_fixture_sha256":sha256_file(Path::new(&a[0]))?,"retained_frames_path":a[1],"retained_frames_sha256":sha256_file(Path::new(&a[1]))?,"profile":profile,"comparisons":comparisons,"profile_comparisons":profile_checks,"frames":rows,"summary":{"fixture_comparisons":comparisons.len()+profile_checks.len(),"comparisons_failed":failed,"retained_source_frames":rows.len(),"applicable_frames":rows.iter().filter(|x|x["source_bound_applicable_to_recorded_frame"]==true).count()},"scope":"Existing structural landscape KUR.1/KUR.2 parameterization. No phase-probability fit, closed-core iteration or population-uniform whole-swarm trapping is inferred."});
    fs::write(&a[2], serde_json::to_vec_pretty(&out)?)?;
    println!("{}", out["summary"]);
    if failed != 0 {
        return Err("Existing landscape formula discrepancies retained".into());
    }
    Ok(())
}
