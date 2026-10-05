//! Reproducible chapter 1--3 validation across native Rust runs.
use crate::{Benchmark, RunConfig, convergence_cloning, convergence_framework};
use algorithmic_gas::{
    GasConfig, GasError, Result,
    fitness::{PositiveMap, Standardizer},
    geometry::Kernel,
    kinetic::KineticKind,
    noise::{FactorValues, NoiseGeometry},
    tracking::RecordingConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ValidationConfig {
    pub walkers: Vec<usize>,
    pub dimensions: Vec<usize>,
    pub seeds: Vec<u64>,
    pub landscapes: Vec<Benchmark>,
    pub profiles: Vec<String>,
    pub steps: usize,
    pub kinetic_samples: usize,
    pub cloning_samples: usize,
    /// Independent paired complete-update error trajectories and fixed-rate controls.
    pub decay: Option<crate::convergence_decay::DecayValidationConfig>,
    /// A law-valid native common-stream coupling on intrinsic representatives.
    pub shared_decay: Option<crate::convergence_decay::DecayValidationConfig>,
    /// Exact complete configurations take precedence over the Cartesian matrix.
    pub runs: Vec<RunConfig>,
}
impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            walkers: vec![8, 32],
            dimensions: vec![2],
            seeds: vec![7, 1729],
            landscapes: vec![
                Benchmark::Quadratic,
                Benchmark::Sphere,
                Benchmark::Rastrigin,
                Benchmark::Constant,
            ],
            profiles: vec![
                "canonical".into(),
                "collapsed".into(),
                "selection".into(),
                "regularization".into(),
                "thermostat".into(),
                "companions".into(),
            ],
            steps: 24,
            kinetic_samples: 1024,
            cloning_samples: 1024,
            decay: Some(crate::convergence_decay::DecayValidationConfig::default()),
            shared_decay: Some(crate::convergence_decay::DecayValidationConfig {
                pair_randomness: crate::convergence_decay::PairRandomness::Shared,
                ..crate::convergence_decay::DecayValidationConfig::default()
            }),
            runs: vec![],
        }
    }
}
impl ValidationConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(decay) = &self.decay {
            decay.validate()?;
        }
        if let Some(decay) = &self.shared_decay {
            decay.validate()?;
        }
        if !(1..=4096).contains(&self.steps)
            || !(32..=1_000_000).contains(&self.kinetic_samples)
            || !(64..=100_000).contains(&self.cloning_samples)
        {
            return Err(GasError::Configuration(
                "steps must be 1..4096, kinetic samples 32..1000000 and cloning samples 64..100000"
                    .into(),
            ));
        }
        if !self.runs.is_empty() {
            for config in &self.runs {
                config.validate()?;
            }
            return Ok(());
        }
        if self.walkers.is_empty()
            || self.dimensions.is_empty()
            || self.seeds.is_empty()
            || self.landscapes.is_empty()
            || self.profiles.is_empty()
            || self.walkers.iter().any(|n| !(2..=4096).contains(n))
            || self.dimensions.iter().any(|d| !(1..=256).contains(d))
        {
            return Err(GasError::Configuration(
                "nonempty matrix axes, walkers 2..4096, dimensions 1..256 required".into(),
            ));
        }
        for landscape in &self.landscapes {
            for &dimension in &self.dimensions {
                landscape.validate(dimension)?;
            }
        }
        for profile in &self.profiles {
            if ![
                "canonical",
                "collapsed",
                "selection",
                "regularization",
                "thermostat",
                "companions",
            ]
            .contains(&profile.as_str())
            {
                return Err(GasError::Configuration(format!(
                    "unknown validation profile {profile}"
                )));
            }
        }
        let cases = self
            .walkers
            .len()
            .checked_mul(self.dimensions.len())
            .and_then(|n| n.checked_mul(self.seeds.len()))
            .and_then(|n| n.checked_mul(self.landscapes.len()))
            .and_then(|n| n.checked_mul(self.profiles.len()));
        if cases.is_none_or(|n| n > 10000) {
            return Err(GasError::Configuration(
                "validation matrix exceeds 10000 cases".into(),
            ));
        }
        Ok(())
    }
    pub fn matrix(&self) -> Result<Vec<(String, RunConfig)>> {
        self.validate()?;
        if !self.runs.is_empty() {
            return Ok(self
                .runs
                .iter()
                .enumerate()
                .map(|(i, c)| (format!("custom_{i}"), c.clone()))
                .collect());
        }
        let mut result = vec![];
        for &n in &self.walkers {
            for &d in &self.dimensions {
                for &seed in &self.seeds {
                    for &landscape in &self.landscapes {
                        for profile in &self.profiles {
                            let mut config = RunConfig {
                                benchmark: landscape,
                                walkers: n,
                                dimensions: d,
                                gas: GasConfig::euclidean(d, 0.04)?,
                                ..RunConfig::euclidean()?
                            };
                            config.gas.seed = seed;
                            match profile.as_str() {
                                "collapsed" => {
                                    config.initial_lower = 0.;
                                    config.initial_upper = 0.;
                                }
                                "selection" => {
                                    config.gas.fitness.reward_exponent = 2.;
                                    config.gas.fitness.diversity_exponent = 0.5;
                                    config.gas.clone_decision.saturation = 0.5;
                                }
                                "regularization" => {
                                    config.gas.fitness.reward_standardizer =
                                        Standardizer::Global { sigma_min: 0.01 };
                                    config.gas.fitness.diversity_standardizer =
                                        Standardizer::Global { sigma_min: 0.01 };
                                    config.gas.fitness.reward_map = PositiveMap::Logistic {
                                        amplitude: 2.,
                                        floor: 0.01,
                                    };
                                    config.gas.fitness.diversity_map =
                                        config.gas.fitness.reward_map.clone();
                                    config.gas.fitness.distance_floor = 0.01;
                                    config.gas.clone_decision.epsilon = 1e-4;
                                }
                                "thermostat" => {
                                    config.gas.kinetic.integrator = KineticKind::Baoab {
                                        positions: "positions".into(),
                                        velocities: "velocities".into(),
                                        dt: 0.08,
                                        friction: 2.,
                                    };
                                    config.gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
                                        scale: FactorValues::Constant { values: vec![0.5] },
                                    };
                                    config.gas.kinetic.position_diffusion = 0.05;
                                    config.gas.kinetic.velocity_cap = Some(1.);
                                    config.gas.clone_transform.jitter_amplitude = 0.03;
                                    config.gas.clone_transform.restitution = Some(0.8);
                                }
                                "companions" => {
                                    config.gas.distance_donors.kernel =
                                        Kernel::Gaussian { width: 0.5 };
                                    config.gas.cloning_donors.kernel =
                                        Kernel::Gaussian { width: 4. };
                                }
                                _ => {}
                            }
                            result.push((
                                format!("{}_{}_n{n}_d{d}_seed{seed}", landscape.id(), profile),
                                config,
                            ));
                        }
                    }
                }
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunValidation {
    pub id: String,
    pub config: RunConfig,
    pub completed_steps: usize,
    pub outcome: String,
    pub error: Option<String>,
    pub checks_evaluated: usize,
    pub checks_failed: usize,
    pub failed_checks: Vec<Value>,
    pub evaluated_labels: Vec<String>,
    pub unavailable: Vec<String>,
    pub trajectory: Vec<Value>,
    pub first_cloning_diagnostics: Option<Value>,
    pub final_cloning_diagnostics: Option<Value>,
    pub calibration: Value,
}
fn population_moments(p: &algorithmic_gas::Population<f64>) -> Result<Value> {
    let eligible = p.eligible(false);
    let x = p.observations.field("positions")?;
    let rows: Vec<_> = x
        .values()
        .chunks(x.width())
        .zip(&eligible)
        .filter_map(|(r, a)| a.then_some(r))
        .collect();
    if rows.is_empty() {
        return Ok(json!({"alive":0}));
    }
    let n = rows.len() as f64;
    let mut center = vec![0.; x.width()];
    for row in &rows {
        for (m, v) in center.iter_mut().zip(*row) {
            *m += *v / n;
        }
    }
    let position_variance = rows
        .iter()
        .map(|r| {
            r.iter()
                .zip(&center)
                .map(|(x, m)| (x - m).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n;
    let position_second_moment = rows
        .iter()
        .map(|r| r.iter().map(|x| x * x).sum::<f64>())
        .sum::<f64>()
        / n;
    let velocity_second_moment = p.observations.field("velocities").ok().map(|v| {
        v.values()
            .chunks(v.width())
            .zip(&eligible)
            .filter(|(_, a)| **a)
            .map(|(r, _)| r.iter().map(|v| v * v).sum::<f64>())
            .sum::<f64>()
            / n
    });
    Ok(
        json!({"alive":rows.len(),"position_variance":position_variance,"position_second_moment":position_second_moment,"velocity_second_moment":velocity_second_moment}),
    )
}
async fn validate_run(id: String, config: RunConfig, steps: usize) -> Result<RunValidation> {
    let mut gas = config.build::<f64>().await?;
    gas.start_recording(RecordingConfig {
        max_steps: steps,
        max_bytes: 128 * 1024 * 1024,
        graph: false,
    })?;
    let mut trajectory = vec![json!({"step":0,"moments":population_moments(gas.population())?})];
    let mut outcome = "completed".to_string();
    let mut error = None;
    for step in 0..steps {
        match gas.step().await {
            Ok(report) => trajectory.push(json!({"step":step+1,"moments":population_moments(gas.population())?,"clones":report.clones,"revivals":report.revivals})),
            Err(GasError::Extinction) => { outcome = "extinction".into(); break; }
            Err(e) => { outcome = "engine_error".into(); error = Some(e.to_string()); break; }
        }
    }
    let archive = gas
        .stop_recording()
        .ok_or_else(|| GasError::Configuration("validation archive absent".into()))?;
    let mut result = RunValidation {
        id,
        config,
        completed_steps: archive.steps.len(),
        outcome,
        error,
        checks_evaluated: 0,
        checks_failed: 0,
        failed_checks: vec![],
        evaluated_labels: vec![],
        unavailable: vec![],
        trajectory,
        first_cloning_diagnostics: None,
        final_cloning_diagnostics: None,
        calibration: json!({}),
    };
    if let Err(error) = archive.validate() {
        result.outcome = "diagnostic_error".into();
        result.error = Some(error.to_string());
        return Ok(result);
    }
    let mut labels = BTreeSet::new();
    let mut unavailable = BTreeSet::new();
    let mut predictions = vec![];
    let mut residuals = vec![];
    for index in 0..archive.steps.len() {
        let diagnostic = match convergence_cloning::analyze_cloning_step(&archive, index) {
            Ok(diagnostic) => diagnostic,
            Err(error) => {
                result.outcome = "diagnostic_error".into();
                result.error = Some(format!("cloning diagnostic step {index}: {error}"));
                break;
            }
        };
        for check in &diagnostic.checks {
            result.checks_evaluated += 1;
            labels.extend(check.source_labels.iter().cloned());
            if !check.passed {
                result.checks_failed += 1;
                result
                    .failed_checks
                    .push(json!({"step":index,"check":check}));
            }
        }
        if let Some(moments) = &diagnostic.moments {
            predictions.push(moments.expected_position_variance);
            residuals
                .push(diagnostic.realized_position_variance - moments.expected_position_variance);
        }
        unavailable.extend(diagnostic.unavailable.iter().cloned());
        if index == 0 {
            result.first_cloning_diagnostics = Some(
                serde_json::to_value(&diagnostic)
                    .map_err(|e| GasError::Configuration(e.to_string()))?,
            );
        }
        if index + 1 == archive.steps.len() {
            result.final_cloning_diagnostics = Some(
                serde_json::to_value(&diagnostic)
                    .map_err(|e| GasError::Configuration(e.to_string()))?,
            );
        }
    }
    result.evaluated_labels = labels.into_iter().collect();
    result.unavailable = unavailable.into_iter().collect();
    // Within-run residuals are a martingale diagnostic, not independent replicates.
    result.calibration = json!({"conditional_predictions":predictions,"realized_minus_conditional_mean":residuals,
        "scope":"Within-run conditional cloning means retain sampled fitness. No IID standard error or global contraction fit is inferred from serial trajectory points."});
    Ok(result)
}

pub fn inventory() -> Result<Vec<Value>> {
    [
        include_str!("../../../proof-validation/chapter01_inventory.json"),
        include_str!("../../../proof-validation/chapter02_inventory.json"),
        include_str!("../../../proof-validation/chapter03_inventory.json"),
    ]
    .iter()
    .map(|s| serde_json::from_str(s).map_err(|e| GasError::Configuration(e.to_string())))
    .collect()
}

/// Gives each formal claim and quantitative expression a disposition. Merely
/// appearing in the inventory never gives a claim a successful numerical status.
pub fn coverage(catalog: &[Value], evaluated: &BTreeSet<String>) -> Value {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut chapters = vec![];
    for chapter in catalog {
        let mut items = vec![];
        for item in chapter["formal_items"].as_array().into_iter().flatten() {
            let label = item["label"].as_str().unwrap_or("");
            let classification = item["computability_kind"].as_str().unwrap_or("conditional");
            let disposition = if evaluated.contains(label) {
                "diagnostic_exercised_under_stated_scope"
            } else if classification == "not_numerically_testable" {
                "analytical_only"
            } else {
                "not_yet_numerically_validated"
            };
            *counts.entry(disposition.into()).or_default() += 1;
            let mut indexed = item.clone();
            if let Some(object) = indexed.as_object_mut() {
                object.insert("disposition".into(), json!(disposition));
            }
            items.push(indexed);
        }
        let mut indexed_chapter = chapter.clone();
        if let Some(object) = indexed_chapter.as_object_mut() {
            object.insert("formal_items".into(), json!(items));
            object.insert("quantitative_expression_disposition".into(), json!("Tracked individually with formula and hypotheses in inventory; unlinked expressions remain unevaluated."));
        }
        chapters.push(indexed_chapter);
    }
    json!({"formal_claim_counts":counts,"chapters":chapters})
}

/// All simulation errors remain in the report; successful runs are never selected
/// after observing results. Every step is diagnosed before dropping its archive.
pub async fn run_validation(config: &ValidationConfig) -> Result<Value> {
    let cases = config.matrix()?;
    let total_cases = cases.len();
    let mut runs = vec![];
    let mut evaluated = BTreeSet::new();
    let mut framework_reports = vec![];
    let mut phase_errors = vec![];
    let mut fixture_axes = BTreeSet::new();
    let mut auxiliary_fixture_unavailable = BTreeSet::new();
    for (_, c) in &cases {
        if let Standardizer::Global { sigma_min } = c.gas.fitness.reward_standardizer {
            if (2..=4096).contains(&c.walkers) && (1..=256).contains(&c.dimensions) {
                fixture_axes.insert((c.walkers, c.dimensions, sigma_min.to_bits()));
            } else {
                auxiliary_fixture_unavailable.insert(format!("Auxiliary bounded framework fixture skips N={}, d={}; actual engine run and supported cloning diagnostics remain included.",c.walkers,c.dimensions));
            }
        }
    }
    let mut exact_failed = 0;
    let mut exact_checks = 0;
    for (n, d, regularizer) in fixture_axes {
        let report = match convergence_framework::validate_framework(
            n,
            d,
            f64::from_bits(regularizer),
            7,
        ) {
            Ok(report) => report,
            Err(error) => {
                phase_errors.push(json!({"phase":"auxiliary_framework","walkers":n,"dimensions":d,"sigma_min":f64::from_bits(regularizer),"error":error.to_string()}));
                continue;
            }
        };
        for check in &report.checks {
            exact_checks += 1;
            exact_failed += usize::from(!check.passed);
            evaluated.extend(check.source_labels.iter().cloned());
        }
        framework_reports.push(report);
    }
    let repaired_bound_regressions = convergence_framework::stable_distance_regressions()?;
    for check in &repaired_bound_regressions {
        exact_checks += 1;
        exact_failed += usize::from(!check.passed);
        evaluated.extend(check.source_labels.iter().cloned());
    }
    for (index, (id, c)) in cases.into_iter().enumerate() {
        if index.is_multiple_of(16) || index + 1 == total_cases {
            eprintln!("run {}/{total_cases}: {id}", index + 1);
        }
        match validate_run(id.clone(), c.clone(), config.steps).await {
            Ok(r) => {
                exact_checks += r.checks_evaluated;
                exact_failed += r.checks_failed;
                evaluated.extend(r.evaluated_labels.iter().cloned());
                runs.push(
                    serde_json::to_value(r).map_err(|e| GasError::Configuration(e.to_string()))?,
                );
            }
            Err(GasError::Extinction) => runs.push(json!({"id":id,"config":c,"outcome":"extinction","completed_steps":0,"error":null,"trajectory":[],"checks_evaluated":0,"checks_failed":0,"evaluated_labels":[],"unavailable":["All-dead input has no current live donor pool; no cloning diagnostic or revival guarantee applies."]})),
            Err(e) => runs.push(
                json!({"id":id,"config":c,"outcome":"diagnostic_error","error":e.to_string()}),
            ),
        }
    }
    let mut kinetic = vec![];
    for case in crate::convergence_kinetic::default_validation_cases(config.kinetic_samples, 50000)
    {
        match crate::convergence_kinetic::validate_kinetic(&case).await {
            Ok(report) => kinetic.push(
                serde_json::to_value(report).map_err(|e| GasError::Configuration(e.to_string()))?,
            ),
            Err(error) => phase_errors.push(
                json!({"phase":"independent_kinetic","config":case,"error":error.to_string()}),
            ),
        }
    }
    // Kinetic source links are exposed by its own diagnostic schema.
    fn gather_labels(v: &Value, output: &mut BTreeSet<String>) {
        match v {
            Value::Object(map) => {
                let diagnostic = map.get("passed").is_some_and(Value::is_boolean)
                    || map
                        .get("consistent_with_theory")
                        .is_some_and(Value::is_boolean)
                    || matches!(
                        map.get("status").and_then(Value::as_str),
                        Some("not_rejected" | "violated")
                    );
                if diagnostic && let Some(Value::Array(labels)) = map.get("source_labels") {
                    output.extend(labels.iter().filter_map(Value::as_str).map(str::to_string));
                }
                for child in map.values() {
                    gather_labels(child, output);
                }
            }
            Value::Array(items) => {
                for child in items {
                    gather_labels(child, output);
                }
            }
            _ => {}
        }
    }
    fn gather_kinetic_checks(v: &Value, output: &mut BTreeSet<String>, counts: &mut [usize; 3]) {
        match v {
            Value::Object(map) => {
                if let Some(source) = map.get("source").and_then(Value::as_str)
                    && matches!(
                        map.get("status").and_then(Value::as_str),
                        Some("not_rejected" | "violated")
                    )
                    && let Some((_, label)) = source.split_once('#')
                {
                    output.insert(label.to_string());
                }
                if let Some(status) = map.get("status").and_then(Value::as_str) {
                    match status {
                        "not_rejected" => counts[0] += 1,
                        "violated" => counts[1] += 1,
                        "not_applicable" => counts[2] += 1,
                        _ => {}
                    }
                }
                for child in map.values() {
                    gather_kinetic_checks(child, output, counts);
                }
            }
            Value::Array(items) => {
                for child in items {
                    gather_kinetic_checks(child, output, counts);
                }
            }
            _ => {}
        }
    }
    let mut kinetic_check_counts = [0; 3];
    for report in &kinetic {
        gather_labels(report, &mut evaluated);
        gather_kinetic_checks(report, &mut evaluated, &mut kinetic_check_counts);
    }
    let four_walker = [
        convergence_cloning::exact_four_walker_fixture(0.)?,
        convergence_cloning::exact_four_walker_fixture(0.1)?,
    ];
    for report in &four_walker {
        for check in &report.checks {
            evaluated.extend(check.source_labels.iter().cloned());
            exact_checks += 1;
            exact_failed += usize::from(!check.passed);
        }
    }
    let coefficient_report = match crate::convergence_coefficients::coefficient_validation_cases() {
        Ok(report) => {
            for check in &report.checks {
                exact_checks += 1;
                exact_failed += usize::from(!check.passed);
                evaluated.extend(check.source_labels.iter().cloned());
            }
            json!(report)
        }
        Err(error) => {
            phase_errors.push(json!({"phase":"continuity_coefficients","error":error.to_string()}));
            Value::Null
        }
    };
    let independent_value =
        match convergence_cloning::validate_independent_cloning(config.cloning_samples, 700_000)
            .await
        {
            Ok(report) => json!(report),
            Err(error) => {
                phase_errors.push(json!({"phase":"independent_cloning","error":error.to_string()}));
                Value::Null
            }
        };
    gather_labels(&independent_value, &mut evaluated);
    fn count_cloning_statistics(value: &Value, counts: &mut [usize; 2]) {
        match value {
            Value::Object(map) => {
                if let Some(consistent) = map.get("consistent_with_theory").and_then(Value::as_bool)
                {
                    counts[usize::from(!consistent)] += 1;
                }
                for child in map.values() {
                    count_cloning_statistics(child, counts);
                }
            }
            Value::Array(items) => {
                for item in items {
                    count_cloning_statistics(item, counts);
                }
            }
            _ => {}
        }
    }
    let mut cloning_check_counts = [0; 2];
    count_cloning_statistics(&independent_value, &mut cloning_check_counts);
    let boundary_report =
        match crate::convergence_boundary::validate_boundary(config.cloning_samples, 9_900_000)
            .await
        {
            Ok(report) => json!(report),
            Err(error) => {
                phase_errors
                    .push(json!({"phase":"complete_boundary_reset","error":error.to_string()}));
                Value::Null
            }
        };
    let lyapunov_report = match crate::convergence_lyapunov::lyapunov_validation_cases(
        config.kinetic_samples,
        1_900_000,
    )
    .await
    {
        Ok(report) => json!(report),
        Err(error) => {
            phase_errors.push(json!({"phase":"lyapunov_geometry","error":error.to_string()}));
            Value::Null
        }
    };
    let continuity_report = match crate::convergence_continuity::continuity_validation_cases() {
        Ok(report) => json!(report),
        Err(error) => {
            phase_errors.push(json!({"phase":"framework_continuity","error":error.to_string()}));
            Value::Null
        }
    };
    let smoothing_report = match crate::convergence_smoothing::smoothing_validation_cases() {
        Ok(report) => json!(report),
        Err(error) => {
            phase_errors.push(json!({"phase":"boundary_smoothing","error":error.to_string()}));
            Value::Null
        }
    };
    let decay_report = if let Some(decay) = &config.decay {
        match crate::convergence_decay::validate_decay(decay).await {
            Ok(report) => json!(report),
            Err(error) => {
                phase_errors
                    .push(json!({"phase":"complete_update_decay","error":error.to_string()}));
                Value::Null
            }
        }
    } else {
        Value::Null
    };
    let shared_decay_report = if let Some(decay) = &config.shared_decay {
        match crate::convergence_decay::validate_decay(decay).await {
            Ok(report) => json!(report),
            Err(error) => {
                phase_errors.push(
                    json!({"phase":"shared_complete_update_decay","error":error.to_string()}),
                );
                Value::Null
            }
        }
    } else {
        Value::Null
    };
    let selection_report = match crate::convergence_selection::validate_selection(2_900_000) {
        Ok(report) => {
            // The top-level checks already include every fixture check. The
            // nested copies retain context but must not inflate the count.
            for check in &report.checks {
                exact_checks += 1;
                exact_failed += usize::from(!check.passed);
                evaluated.extend(check.source_labels.iter().cloned());
            }
            json!(report)
        }
        Err(error) => {
            phase_errors.push(json!({"phase":"selection_geometry","error":error.to_string()}));
            Value::Null
        }
    };
    fn count_auxiliary_checks(
        value: &Value,
        checks: &mut usize,
        failures: &mut usize,
        statistics: &mut [usize; 3],
    ) {
        match value {
            Value::Object(map) => {
                if map.contains_key("bound")
                    && map.contains_key("source_labels")
                    && let Some(passed) = map.get("passed").and_then(Value::as_bool)
                {
                    *checks += 1;
                    *failures += usize::from(!passed);
                }
                match map.get("status").and_then(Value::as_str) {
                    Some("not_rejected") => statistics[0] += 1,
                    Some("violated") => statistics[1] += 1,
                    Some("not_applicable") => statistics[2] += 1,
                    _ => {}
                }
                for child in map.values() {
                    count_auxiliary_checks(child, checks, failures, statistics);
                }
            }
            Value::Array(items) => {
                for item in items {
                    count_auxiliary_checks(item, checks, failures, statistics);
                }
            }
            _ => {}
        }
    }
    let mut auxiliary_statistics = [0; 3];
    for report in [
        &boundary_report,
        &lyapunov_report,
        &continuity_report,
        &smoothing_report,
        &decay_report,
        &shared_decay_report,
    ] {
        count_auxiliary_checks(
            report,
            &mut exact_checks,
            &mut exact_failed,
            &mut auxiliary_statistics,
        );
        gather_labels(report, &mut evaluated);
    }
    let decay_case_lists = [
        decay_report["cases"].as_array(),
        shared_decay_report["cases"].as_array(),
    ];
    let decay_case_count: usize = decay_case_lists
        .iter()
        .flatten()
        .map(|cases| cases.len())
        .sum();
    let decay_pairs: usize = decay_case_lists
        .iter()
        .flatten()
        .flat_map(|cases| cases.iter())
        .map(|case| case["seed_trajectories"].as_array().map_or(0, Vec::len))
        .sum();
    let decay_updates: usize = decay_case_lists
        .iter()
        .flatten()
        .flat_map(|cases| cases.iter())
        .flat_map(|case| case["seed_trajectories"].as_array().into_iter().flatten())
        .map(|trajectory| {
            trajectory["points"]
                .as_array()
                .map_or(0, |points| points.len().saturating_sub(1))
        })
        .sum();
    let errors = runs
        .iter()
        .filter(|r| {
            matches!(
                r["outcome"].as_str(),
                Some("diagnostic_error" | "engine_error")
            )
        })
        .count();
    let extinct = runs.iter().filter(|r| r["outcome"] == "extinction").count();
    let mut keystone_constants = vec![];
    for dimension in cases_dimensions(&runs) {
        for threshold in [0.01, 0.1, 1.] {
            match convergence_cloning::canonical_keystone_constants(
                dimension,
                threshold,
                2. * (dimension as f64).sqrt(),
            ) {
                Ok(constants) => keystone_constants.push(constants),
                Err(error) => phase_errors.push(json!({"phase":"canonical_keystone_constants","dimensions":dimension,"threshold":threshold,"error":error.to_string()})),
            }
        }
    }
    Ok(
        json!({"schema_version":1,"config":config,"engine":"native algorithmic-gas f64; configured backends recorded per run",
        "summary":{"runs":runs.len(),"exact_checks":exact_checks,"exact_checks_failed":exact_failed,"run_errors":errors,"phase_errors":phase_errors.len(),"extinctions":extinct,"repaired_bound_regressions":repaired_bound_regressions.len(),"kinetic_statistical_checks_not_rejected":kinetic_check_counts[0],"kinetic_statistical_checks_violated":kinetic_check_counts[1],"kinetic_hypotheses_not_applicable":kinetic_check_counts[2],"cloning_statistical_checks_not_rejected":cloning_check_counts[0],"cloning_statistical_checks_violated":cloning_check_counts[1],"auxiliary_statistical_checks_not_rejected":auxiliary_statistics[0],"auxiliary_statistical_checks_violated":auxiliary_statistics[1],"auxiliary_hypotheses_not_applicable":auxiliary_statistics[2],"paired_decay_cases":decay_case_count,"paired_decay_trajectories":decay_pairs,"paired_complete_update_observations":decay_updates},
        "framework":framework_reports,"kinetic":kinetic,"cloning_four_walker_exact":four_walker,
        "independent_cloning":independent_value,
        "canonical_keystone_constants":keystone_constants,
        "canonical_keystone_reference":{"landscape":"quadratic","objective":"U(x)=|x|²/2, minimized","reward":"R=-U; no velocity penalty","physical_region":"(-2,2)^d with entering velocity norm<=2","profile":"fixed GasConfig::euclidean(d,0.04)","analytic_physical_reward_lipschitz":"2*sqrt(d)","application":"Reference calculation only; these constants are not assigned to other landscapes or altered parameter profiles."},
        "continuity_coefficients":coefficient_report,
        "complete_boundary_reset":boundary_report,
        "lyapunov_geometry":lyapunov_report,
        "framework_continuity":continuity_report,
        "boundary_smoothing":smoothing_report,
        "complete_update_decay":decay_report,
        "shared_update_decay":shared_decay_report,
        "selection_geometry":selection_report,
        "auxiliary_fixture_unavailable":auxiliary_fixture_unavailable,
        "phase_errors":phase_errors,
        "repaired_bound_regressions":repaired_bound_regressions,"runs":runs,
        "coverage":coverage(&inventory()?, &evaluated),
        "interpretation":"Numerical agreement is scoped evidence. Uncomputed, existential, global, auxiliary-law and conditional quantities are explicitly tracked; this report does not certify all convergence hypotheses or geometric ergodicity."}),
    )
}

fn cases_dimensions(runs: &[Value]) -> BTreeSet<usize> {
    runs.iter()
        .filter_map(|run| run["config"]["dimensions"].as_u64().map(|n| n as usize))
        .collect()
}
