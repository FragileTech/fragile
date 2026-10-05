//! Independent full Poisson point clouds, not inverse-nearest-CDF samples.
use algorithmic_gas::random::{RandomStream, Stream};
use algorithmic_gas_benchmarks::{
    convergence_chapter07_completion::poisson_reference,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("gas-chapter07-poisson EMPTY_OUTPUT".into());
    }
    let mut out = ArchiveStore::new(&args[0])?;
    out.save_json("poisson-source",&json!({"runner":include_str!("gas-chapter07-poisson.rs"),
        "formula_module":include_str!("../convergence_chapter07_completion.rs"),
        "rng":include_str!("../../../algorithmic-gas/src/random.rs"),
        "chapter":include_str!("../../../../../docs/source/2_fractal_gas/convergence_program/07_discrete_qsd.md"),
        "binary_sha256":sha256_file(&std::env::current_exe()?)?,"new_native_updates":0,
        "scope":"Specified homogeneous PPP reference, with native Rust RandomStream randomness. No native Fractal Gas PPP hypothesis inferred."}))?;
    let replicates = 8192usize;
    let failure_budget = 0.01;
    let checks_per_case = 5usize;
    let delta = failure_budget / (15 * checks_per_case) as f64;
    let mut comparisons: Vec<Value> = vec![];
    let mut cases = vec![];
    let mut points = 0usize;
    for (dimension_index, d) in [1usize, 2, 3, 4, 8].into_iter().enumerate() {
        for (intensity_index, intensity) in [0.5, 2., 8.].into_iter().enumerate() {
            let p = poisson_reference(d, intensity, 1.)?;
            let omega = p["omega_D"].as_f64().unwrap();
            let radius = (32. / (intensity * omega)).powf(1. / d as f64);
            let expected = p["mean"].as_f64().unwrap();
            let censor_tail = radius * (-32_f64).exp() / (32. * d as f64);
            let mut raw = vec![];
            let mut minima = vec![];
            let mut empty = 0usize;
            for replicate in 0..replicates {
                let mut rng = RandomStream::new(
                    2026100471,
                    dimension_index as u64,
                    Stream::Kinetic,
                    (intensity_index * replicates + replicate) as u64,
                    37,
                );
                // Knuth count generation: PPP restriction to the declared ball
                // has Poisson(lambda volume)=Poisson(32) point count.
                let mut product = 1_f64;
                let mut count = 0usize;
                loop {
                    product *= rng.uniform::<f64>();
                    if product <= (-32_f64).exp() {
                        break;
                    }
                    count += 1;
                }
                let mut coordinates = Vec::with_capacity(count * d);
                let mut radii = vec![];
                let mut nearest = radius;
                for _ in 0..count {
                    let radial = radius * rng.uniform::<f64>().powf(1. / d as f64);
                    let direction: Vec<_> = (0..d).map(|_| rng.gaussian::<f64>()).collect();
                    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm == 0. {
                        return Err("zero Gaussian direction".into());
                    }
                    let row: Vec<_> = direction.iter().map(|x| radial * x / norm).collect();
                    let measured_norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                    nearest = nearest.min(measured_norm);
                    radii.push(measured_norm);
                    coordinates.extend(row);
                }
                points += count;
                empty += usize::from(count == 0);
                minima.push(nearest);
                raw.push(json!({"replicate":replicate,"point_count":count,"coordinates":coordinates,
                    "radii_from_coordinates":radii,"nearest_norm_censored_at_ball":nearest,"empty":count==0}));
            }
            let mean = minima.iter().sum::<f64>() / replicates as f64;
            let variance =
                minima.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (replicates - 1) as f64;
            let se = (variance / replicates as f64).sqrt();
            let allowance = radius * ((2_f64 / delta).ln() / (2. * replicates as f64)).sqrt();
            comparisons.push(json!({"id":format!("d{d}-lambda{intensity}-mean"),"D":d,"intensity":intensity,
                "observed":mean,"prediction":expected,"standard_error":se,"bounded_mean_hoeffding_allowance":allowance,
                "censor_bias_upper":censor_tail,"passed":(mean-expected).abs()<=allowance+censor_tail,
                "independent_clouds":replicates,"per_comparison_failure_budget":delta,"relation":"mean with two-sided Hoeffding plus analytic censor-tail error"}));
            for multiple in [0.25, 0.5, 1., 1.5] {
                let test_radius = expected * multiple;
                if test_radius >= radius {
                    return Err("CDF radius must be interior".into());
                }
                let predicted = (-intensity * omega * test_radius.powf(d as f64)).exp();
                let observed =
                    minima.iter().filter(|x| **x > test_radius).count() as f64 / replicates as f64;
                let allowance = ((2_f64 / delta).ln() / (2. * replicates as f64)).sqrt();
                comparisons.push(json!({"id":format!("d{d}-lambda{intensity}-survival-c{multiple}"),"D":d,"intensity":intensity,
                    "radius":test_radius,"observed":observed,"prediction":predicted,
                    "hoeffding_allowance":allowance,"passed":(observed-predicted).abs()<=allowance,
                    "independent_clouds":replicates,"per_comparison_failure_budget":delta}));
            }
            let raw_path=out.save_json(&format!("point-clouds-d{d}-lambda{intensity}"),&json!({
                "D":d,"intensity":intensity,"ball_radius":radius,"poisson_count_mean":32.,"clouds":raw}))?;
            cases.push(json!({"D":d,"intensity":intensity,"observed_mean":mean,"prediction_mean":expected,
                "mean_standard_error":se,"ball_radius":radius,"empty_clouds":empty,"raw_archive":raw_path,
                "independent_clouds":replicates,"hoeffding_allowance":allowance,"censor_tail_upper":censor_tail}));
        }
    }
    let mut rates = vec![];
    for d in [1usize, 2, 3, 4, 8] {
        let own: Vec<_> = cases.iter().filter(|r| r["D"] == d).collect();
        let first = own[0];
        let last = own[2];
        let denominator = (16_f64).ln();
        let slope = (last["observed_mean"].as_f64().unwrap()
            / first["observed_mean"].as_f64().unwrap())
        .ln()
            / denominator;
        let standard_error = ((last["mean_standard_error"].as_f64().unwrap()
            / last["observed_mean"].as_f64().unwrap())
        .powi(2)
            + (first["mean_standard_error"].as_f64().unwrap()
                / first["observed_mean"].as_f64().unwrap())
            .powi(2))
        .sqrt()
            / denominator;
        let first_mean = first["observed_mean"].as_f64().unwrap();
        let last_mean = last["observed_mean"].as_f64().unwrap();
        let first_allowance = first["hoeffding_allowance"].as_f64().unwrap()
            + first["censor_tail_upper"].as_f64().unwrap();
        let last_allowance = last["hoeffding_allowance"].as_f64().unwrap()
            + last["censor_tail_upper"].as_f64().unwrap();
        if first_mean <= first_allowance || last_mean <= last_allowance {
            return Err("rate confidence endpoints are not positive".into());
        }
        let lower =
            ((last_mean - last_allowance) / (first_mean + first_allowance)).ln() / denominator;
        let upper =
            ((last_mean + last_allowance) / (first_mean - first_allowance)).ln() / denominator;
        let predicted = -1. / d as f64;
        rates.push(json!({"D":d,"observed_log_density_slope":slope,"prediction":predicted,"delta_method_standard_error":standard_error,
            "simultaneous_hoeffding_interval":[lower,upper],"passed":lower<=predicted&&predicted<=upper,
            "scope":"Independent PPP cloud laws at intensity .5 and 8; interval inherited from family-adjusted bounded mean checks. SE is descriptive, not substituted for the interval."}));
    }
    let failed = comparisons
        .iter()
        .chain(&rates)
        .filter(|r| r["passed"] != true)
        .count();
    let report = json!({"chapter":7,"summary":{"point_cloud_laws":15,"independent_point_clouds":replicates*15,
        "generated_points":points,"comparisons":comparisons.len()+rates.len(),"comparisons_failed":failed,"new_native_updates":0},
        "cases":cases,"comparisons":comparisons,"density_rates":rates,"family_failure_budget":failure_budget,
        "hypotheses":"Independent homogeneous PPP restricted to balls of mean occupancy 32. All point coordinates retained. Empty clouds are censored at radius R, with analytic mean tail bound R exp(-32)/(32D); no empty cloud is dropped. Native selected companion laws are a separate experiment."});
    out.save_json("poisson-report", &report)?;
    out.finish("complete")?;
    fs::write(
        PathBuf::from(&args[0]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        PathBuf::from(&args[0]).join("verification.json"),
        serde_json::to_vec_pretty(&out.verify(true)?)?,
    )?;
    println!("{}", report["summary"]);
    if failed > 0 {
        return Err("PPP comparison failed; raw clouds retained".into());
    }
    Ok(())
}
