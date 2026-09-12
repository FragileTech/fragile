//! Summarize saved trajectories and compare a declared spatial moment closure
//! on disjoint seeds, preserving each seed across configuration comparisons.
use algorithmic_gas::{
    RunArchive,
    physics::{
        balances::analyze_step,
        closure::{ClosureSample, test_linear_closure},
    },
};
use algorithmic_gas_benchmarks::RunConfig;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
type Error = Box<dyn std::error::Error>;
fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut x = values.to_vec();
    x.sort_by(f64::total_cmp);
    Some(x[((x.len() - 1) as f64 * q).round() as usize])
}
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("../outputs/physics-research"),
    );
    let manifest: Value = serde_json::from_slice(&std::fs::read(path.join("manifest.json"))?)?;
    let mut files = std::fs::read_dir(path)?
        .map(|p| p.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    files.retain(|p| {
        p.file_name().unwrap().to_string_lossy().starts_with("job_")
            && p.extension().is_some_and(|e| e == "json")
    });
    files.sort();
    let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut trajectories = 0;
    let mut steps = 0;
    let mut replicas = 0;
    let mut rows = 0;
    let mut curvature = Vec::new();
    let mut max_balance: f64 = 0.;
    let mut standardized = Vec::new();
    let mut thermostat_residual = 0.;
    let mut thermostat_variance = 0.;
    let mut missing_curvature = 0;
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut validated_archives = 0;
    let mut validated_queries = 0;
    let mut replica_keys = BTreeSet::new();
    let mut mismatches = Vec::new();
    for (job_id, file) in files.iter().enumerate() {
        let job: Value = serde_json::from_slice(&std::fs::read(file)?)?;
        let config = &job["configuration"];
        let run_config: RunConfig = serde_json::from_value(config.clone())?;
        run_config.validate()?;
        let mut last_archive = None;
        let n = config["walkers"].as_u64().ok_or("walker count")?;
        let d = config["dimensions"].as_u64().ok_or("dimension")? as usize;
        let seed = config["gas"]["seed"].as_u64().ok_or("seed")?;
        let mut job_thermostat_residual = 0.;
        let mut job_thermostat_variance = 0.;
        let mut job_thermostat_residual_squared = 0.;
        let mut job_thermostat_records = 0;
        trajectories += 1;
        steps += job["steps"].as_u64().unwrap();
        rows += n * job["steps"].as_u64().unwrap();
        for record in job["records"].as_array().ok_or("records")? {
            let archive = RunArchive::<f64>::from_bytes(&std::fs::read(
                path.join(record["archive"].as_str().ok_or("archive path")?),
            )?)?;
            if archive.gas_config != run_config.gas
                || archive.steps.len() != 1
                || archive.steps[0].report.step != record["step"].as_u64().ok_or("record step")?
            {
                return Err(format!("archive/configuration mismatch in {}", file.display()).into());
            }
            let computed = analyze_step(
                &archive.steps[0],
                &run_config.gas,
                Some(
                    &run_config
                        .potential
                        .unwrap_or(run_config.benchmark)
                        .physics_objective(d),
                ),
            )?;
            if serde_json::to_value(computed)? != record["balance"] {
                mismatches.push(format!(
                    "{} step {}: archived balance differs",
                    file.display(),
                    record["step"]
                ));
            }
            let field = archive.steps[0]
                .field_evaluations
                .iter()
                .find(|f| f.field == "fitness_scalar_curvature")
                .ok_or("missing archived curvature")?;
            let values: Vec<_> = field
                .values
                .iter()
                .zip(&field.available)
                .filter_map(|(&v, &available)| available.then_some(v))
                .collect();
            let mean = if values.is_empty() {
                None
            } else {
                Some(values.iter().sum::<f64>() / values.len() as f64)
            };
            if serde_json::to_value(mean)? != record["mean_scalar_curvature"]
                || values.len() as u64
                    != record["curvature_available"]
                        .as_u64()
                        .ok_or("curvature count")?
                || field.rows as u64 != record["curvature_queries"].as_u64().ok_or("query count")?
            {
                mismatches.push(format!(
                    "{} step {}: reported curvature {}, archived {:?}; coverage {}/{}",
                    file.display(),
                    record["step"],
                    record["mean_scalar_curvature"],
                    mean,
                    values.len(),
                    field.rows
                ));
            }
            validated_archives += 1;
            validated_queries += field.rows;
            last_archive = Some(archive);
            if let Some(v) = record["mean_scalar_curvature"].as_f64() {
                curvature.push(v);
            }
            missing_curvature += record["curvature_queries"].as_u64().unwrap()
                - record["curvature_available"].as_u64().unwrap();
            let balance = &record["balance"];
            max_balance = max_balance.max(
                balance["kinetic_telescoping_residual"]
                    .as_f64()
                    .unwrap()
                    .abs(),
            );
            if let (Some(predicted), Some(actual), Some(variance)) = (
                balance["thermostat"]["energy_mean_delta"].as_f64(),
                balance["thermostat_realized_energy_delta"].as_f64(),
                balance["thermostat"]["energy_variance"].as_f64(),
            ) {
                thermostat_residual += actual - predicted;
                thermostat_variance += variance;
                job_thermostat_residual += actual - predicted;
                job_thermostat_variance += variance;
                job_thermostat_residual_squared += (actual - predicted).powi(2);
                job_thermostat_records += 1;
            }
            let evolution = &record["conditional_metric_evolution"];
            if !evolution.is_null() {
                for field in ["calibration_keys", "validation_keys"] {
                    let keys = evolution[field].as_array().ok_or("replica keys")?;
                    if keys.len() as u64
                        != job["replicas_per_pool"]
                            .as_u64()
                            .ok_or("replica pool size")?
                    {
                        return Err("replica pool coverage mismatch".into());
                    }
                    for key in keys {
                        if !replica_keys.insert(key.as_u64().ok_or("replica key")?) {
                            return Err("reused replica random key across research records".into());
                        }
                    }
                }
            }
            if let (Some(residual), Some(se)) = (
                evolution["validation_residual"].as_array(),
                evolution["validation_standard_error"].as_array(),
            ) {
                replicas += 2 * job["replicas_per_pool"].as_u64().unwrap();
                for (r, s) in residual.iter().zip(se) {
                    let (r, s) = (r.as_f64().unwrap(), s.as_f64().unwrap());
                    if s > 0. {
                        standardized.push(r.abs() / s);
                    }
                }
            }
        }
        let last = job["records"]
            .as_array()
            .unwrap()
            .last()
            .ok_or("final record")?;
        let key = serde_json::to_string(
            &json!({"dimension":d,"walkers":n,"metric":config["physics_metric"],"kinetic":config["gas"]["kinetic"]["integrator"],"fitness":config["gas"]["fitness"]}),
        )?;
        groups.entry(key).or_default().push(json!({"seed":seed,"curvature":last["mean_scalar_curvature"],"kinetic_energy":last["balance"]["final_state"]["kinetic_energy"],"count":last["balance"]["final_state"]["count"],"elapsed_seconds":job["elapsed_seconds"],"thermostat_residual_sum":job_thermostat_residual,"thermostat_predictable_variance_sum":job_thermostat_variance,"thermostat_residual_squared_sum":job_thermostat_residual_squared,"thermostat_records":job_thermostat_records}));
        if d == 3 {
            let archive = last_archive.ok_or("missing final archive")?;
            let step = &archive.steps[0];
            let field = |name: &str| {
                step.field_evaluations
                    .iter()
                    .find(|f| f.field == name)
                    .ok_or_else(|| format!("missing {name}"))
            };
            let g = field("fitness_metric")?;
            let ric = field("fitness_ricci")?;
            let scalar = field("fitness_scalar_curvature")?;
            let input = step
                .stages
                .iter()
                .find(|s| s.stage == "A1")
                .ok_or("O input stage")?;
            let velocity = &input
                .fields
                .get("velocities")
                .ok_or("O input velocities")?
                .values;
            let mut geometric = [0.; 9];
            let mut mean_metric = [0.; 9];
            let mut available = 0;
            let mut momentum = [0.; 3];
            let mut stress = [0.; 9];
            for row in 0..g.rows {
                if !scalar.available[row] || !ric.available[row] {
                    continue;
                }
                available += 1;
                for ij in 0..9 {
                    geometric[ij] += ric.values[row * 9 + ij]
                        - 0.5 * scalar.values[row] * g.values[row * 9 + ij];
                    mean_metric[ij] += g.values[row * 9 + ij];
                    stress[ij] += velocity[row * 3 + ij / 3] * velocity[row * 3 + ij % 3];
                }
                for i in 0..3 {
                    momentum[i] += velocity[row * 3 + i];
                }
            }
            let count = available as f64;
            if available > 0 {
                for i in 0..3 {
                    for j in i..3 {
                        let ij = i * 3 + j;
                        let sample = ClosureSample {
                            key: (job_id * 9 + ij) as u64,
                            predictors: vec![
                                stress[ij] / count - momentum[i] * momentum[j] / (count * count),
                                mean_metric[ij] / count,
                            ],
                            observed: geometric[ij] / count,
                        };
                        if seed < 23 {
                            training.push(sample);
                        } else {
                            validation.push(sample);
                        }
                    }
                }
            }
        }
    }
    if !mismatches.is_empty() {
        std::fs::write(
            path.join("archive-audit.json"),
            serde_json::to_vec_pretty(&json!({"consistent":false,"mismatches":mismatches}))?,
        )?;
        return Err(format!(
            "{} archive/report mismatches; see archive-audit.json",
            mismatches.len()
        )
        .into());
    }
    std::fs::write(
        path.join("archive-audit.json"),
        serde_json::to_vec_pretty(
            &json!({"consistent":true,"validated_archives":validated_archives,"validated_curvature_queries":validated_queries,"unique_replica_keys":replica_keys.len()}),
        )?,
    )?;
    let closure = if training.len() > 2 && !validation.is_empty() {
        // Training-only scale normalization; no held-out target informs the fit.
        let scales: Vec<_> = (0..2)
            .map(|i| {
                (training
                    .iter()
                    .map(|s| s.predictors[i].powi(2))
                    .sum::<f64>()
                    / training.len() as f64)
                    .sqrt()
                    .max(1e-12)
            })
            .collect();
        for sample in training.iter_mut().chain(&mut validation) {
            for (i, &scale) in scales.iter().enumerate() {
                sample.predictors[i] /= scale;
            }
        }
        let fit = test_linear_closure(&training, &validation, 1e-10)?;
        let mut shuffled = training.clone();
        let predictors: Vec<_> = training.iter().map(|s| s.predictors.clone()).collect();
        let length = shuffled.len();
        for (i, sample) in shuffled.iter_mut().enumerate() {
            sample.predictors = predictors[(i + 6 * ((length / 6) / 3)) % length].clone();
        }
        Some(
            json!({"candidate":"population mean of spatial (Ricci - R*g/2) = kappa * centered kinetic vv tensor per particle + lambda * population mean g; common coefficients across tested regimes","fit":fit,"shuffled_training_control":test_linear_closure(&shuffled,&validation,1e-10)?,"training_predictor_rms":scales,"split":"seeds7..22 calibration;23..38 validation; tensor components of each seed stay together","measurement":"geometry and kinetic moments use the same available rows at the actual O query before thermostat noise","definition":"spatial curvature tensors are evaluated per query and then averaged; the candidate uses two common coefficients across the tested regimes"}),
        )
    } else {
        None
    };
    let grouped:Vec<_>=groups.into_iter().map(|(key,values)|{
        let curv:Vec<_>=values.iter().filter_map(|v|v["curvature"].as_f64()).collect();
        let sum = |name: &str| values.iter().filter_map(|v|v[name].as_f64()).sum::<f64>();
        let residual = sum("thermostat_residual_sum");
        let variance = sum("thermostat_predictable_variance_sum");
        json!({"configuration":serde_json::from_str::<Value>(&key).unwrap(),"trajectories":values.len(),"final_mean_scalar_curvature_median":quantile(&curv,0.5),"final_mean_scalar_curvature_p95":quantile(&curv,0.95),"thermostat_records":sum("thermostat_records"),"thermostat_residual_sum":residual,"thermostat_predictable_variance_sum":variance,"thermostat_residual_over_predicted_standard_deviation":if variance>0.{Some(residual/variance.sqrt())}else{None},"thermostat_squared_residual_over_variance":if variance>0.{Some(sum("thermostat_residual_squared_sum")/variance)}else{None},"runs":values})
    }).collect();
    let report = json!({"schema_version":1,"complete":trajectories==manifest["jobs"].as_u64().unwrap(),"expected_trajectories":manifest["jobs"],"completed_trajectories":trajectories,"validated_archives":validated_archives,"validated_curvature_queries":validated_queries,"unique_replica_keys":replica_keys.len(),"trajectory_updates":steps,"walker_updates":rows,"conditional_replica_updates":replicas,"maximum_kinetic_telescoping_residual":max_balance,"recorded_curvature_unavailable_queries":missing_curvature,"mean_query_curvature_median":quantile(&curvature,0.5),"mean_query_curvature_p95":quantile(&curvature,0.95),"thermostat_sum_residual":thermostat_residual,"thermostat_sum_of_individual_predictable_variances":thermostat_variance,"thermostat_uncertainty":"standardized thermostat residuals are computed within each configuration across independent seeds; shared random addresses correlate configurations, so individual variances cannot be summed to standardize the pooled residual","metric_validation_absolute_standardized_residual_median":quantile(&standardized,0.5),"metric_validation_absolute_standardized_residual_p95":quantile(&standardized,0.95),"interpretation":"per-configuration uncertainty and seed-separated constitutive evaluation","spatial_moment_closure":closure,"groups":grouped});
    std::fs::write(
        path.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"complete":report["complete"],"trajectories":trajectories,"summary":path.join("summary.json")})
        )?
    );
    Ok(())
}
