//! Conditional experiments on the executed gas, with complete checkpoint replicas.
use crate::RunConfig;
use algorithmic_gas::{
    AlgorithmicGas, BackendKind, Checkpoint, GasError, Population, Precision, Result,
    kinetic::KineticKind,
    noise::{InnovationLaw, InnovationShift},
    physics::{
        evolution::{ReplicaSample, metric_evolution},
        partvi::{ExperimentRequest, ExperimentResult, Series},
    },
    random::{RandomStream, Stream},
    tracking::{MechanicalStageBudget, RecordingConfig},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn configuration(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn moments(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    (mean, (variance / values.len() as f64).sqrt())
}
fn readout(
    p: &Population<f64>,
    name: &str,
    walker: usize,
    coordinate: usize,
    include_truncated: bool,
) -> Result<f64> {
    let eligible = p.eligible(include_truncated);
    let active = eligible.iter().filter(|&&a| a).count();
    if name == "alive_fraction" {
        return Ok(active as f64 / p.len() as f64);
    }
    let field = if name == "mean_position" {
        "positions"
    } else {
        "velocities"
    };
    let values = p.observations.field(field)?;
    let d = values.width();
    if coordinate >= d || walker >= p.len() {
        return Err(configuration(
            "observable coordinate/slot outside population",
        ));
    }
    match name {
        "tagged_velocity" => Ok(if eligible[walker] {
            values.values()[walker * d + coordinate].tanh()
        } else {
            0.
        }),
        "momentum" => Ok((0..p.len())
            .filter(|&i| eligible[i])
            .map(|i| values.values()[i * d + coordinate])
            .sum()),
        "mean_position" => Ok((0..p.len())
            .filter(|&i| eligible[i])
            .map(|i| values.values()[i * d + coordinate])
            .sum::<f64>()
            / p.len() as f64),
        "kinetic_energy" => Ok((0..p.len())
            .filter(|&i| eligible[i])
            .flat_map(|i| i * d..(i + 1) * d)
            .map(|j| 0.5 * values.values()[j].powi(2))
            .sum()),
        _ => Err(configuration(
            "observable must be tagged_velocity, momentum, mean_position, kinetic_energy, or alive_fraction",
        )),
    }
}
struct Continuation {
    value: f64,
    survived: bool,
    gas: AlgorithmicGas<f64>,
}
#[allow(clippy::too_many_arguments)]
async fn continuation(
    config: &RunConfig,
    checkpoint: &Checkpoint<f64>,
    seed: u64,
    horizon: usize,
    shift: Option<InnovationShift>,
    observable: &str,
    walker: usize,
    coordinate: usize,
) -> Result<Continuation> {
    let mut checkpoint = checkpoint.with_future_seed(seed)?;
    if let Some(shift) = shift {
        checkpoint.config.qft.innovation_shifts.push(shift);
    }
    let mut config = config.clone();
    config.gas = checkpoint.config.clone();
    let mut gas = config.build::<f64>().await?;
    gas.restore(checkpoint)?;
    gas.start_recording(RecordingConfig {
        max_steps: horizon,
        max_bytes: config.gas.max_memory_bytes.min(128 * 1024 * 1024),
    })?;
    let mut survived = true;
    for _ in 0..horizon {
        match gas.step().await {
            Ok(_) => {}
            Err(GasError::Extinction) => {
                survived = false;
                break;
            }
            Err(error) => return Err(error),
        }
    }
    let value = if survived {
        readout(
            gas.population(),
            observable,
            walker,
            coordinate,
            config.gas.include_truncated,
        )?
    } else {
        0.
    };
    Ok(Continuation {
        value,
        survived,
        gas,
    })
}
/// Replay material is bounded independently of the simulation budget. Every
/// continuation retains its complete random schedule and archive fingerprint;
/// the first archives fitting this byte budget are additionally embedded.
const ARCHIVE_EVIDENCE_BUDGET: usize = 1024 * 1024;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayEvidence {
    run_config: RunConfig,
    request: ExperimentRequest,
    frozen_checkpoint_cbor: Vec<u8>,
    runs: Vec<Value>,
}
#[derive(Default)]
struct EvidenceCollector {
    runs: Vec<Value>,
    retained_bytes: usize,
}
fn fingerprint(bytes: &[u8]) -> String {
    format!(
        "{:016x}",
        bytes
            .iter()
            .fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64)
                .wrapping_mul(0x100000001b3))
    )
}
impl EvidenceCollector {
    fn record(
        &mut self,
        c: &Continuation,
        group: &str,
        replica: usize,
        horizon: usize,
    ) -> Result<()> {
        let archive = c
            .gas
            .recording()
            .ok_or_else(|| configuration("Continuation recording unavailable"))?;
        let bytes = archive.to_bytes()?;
        let hash = fingerprint(&bytes);
        let embedded = self
            .retained_bytes
            .checked_add(bytes.len())
            .is_some_and(|n| n <= ARCHIVE_EVIDENCE_BUDGET);
        let payload = if embedded {
            self.retained_bytes += bytes.len();
            Some(bytes)
        } else {
            None
        };
        self.runs.push(json!({"group":group,"replica":replica,"future_seed":c.gas.config().seed,"horizon":horizon,"recorded_steps":archive.steps.len(),"innovation_shifts":c.gas.config().qft.innovation_shifts,"survived":c.survived,"terminal_readout":c.value,"archive_fingerprint_fnv1a64":hash,"archive_cbor":payload}));
        Ok(())
    }
    fn attach(
        self,
        result: &mut ExperimentResult,
        config: &RunConfig,
        request: &ExperimentRequest,
        checkpoint: &Checkpoint<f64>,
    ) -> Result<()> {
        let executed_steps: usize = self
            .runs
            .iter()
            .filter_map(|r| r["recorded_steps"].as_u64())
            .map(|n| n as usize)
            .sum();
        let evidence = ReplayEvidence {
            run_config: config.clone(),
            request: request.clone(),
            frozen_checkpoint_cbor: checkpoint.to_bytes()?,
            runs: self.runs,
        };
        result.details["continuation_executed_steps"] = json!(executed_steps);
        result.details["continuation_embedded_archive_bytes"] = json!(self.retained_bytes);
        result.details["replay_evidence"] =
            serde_json::to_value(evidence).map_err(|e| configuration(&e.to_string()))?;
        Ok(())
    }
}
/// Recompute results from the exported complete checkpoint and exact future
/// schedules. Embedded archives and every regenerated fingerprint are checked.
pub async fn replay_evidence(value: &Value) -> Result<ExperimentResult> {
    let evidence: ReplayEvidence =
        serde_json::from_value(value.clone()).map_err(|e| configuration(&e.to_string()))?;
    evidence.request.validate()?;
    evidence.run_config.validate()?;
    if evidence.frozen_checkpoint_cbor.len() > 128 * 1024 * 1024 || evidence.runs.len() > 512 {
        return Err(configuration("Continuation replay evidence exceeds limits"));
    }
    let checkpoint = Checkpoint::from_bytes(&evidence.frozen_checkpoint_cbor)?;
    if checkpoint.config != evidence.run_config.gas {
        return Err(configuration(
            "Frozen checkpoint configuration differs from replay configuration",
        ));
    }
    let mut retained_bytes = 0usize;
    for entry in &evidence.runs {
        if !entry["archive_cbor"].is_null() {
            let bytes: Vec<u8> = serde_json::from_value(entry["archive_cbor"].clone())
                .map_err(|e| configuration(&e.to_string()))?;
            retained_bytes = retained_bytes
                .checked_add(bytes.len())
                .ok_or_else(|| configuration("Replay archive byte overflow"))?;
            if retained_bytes > ARCHIVE_EVIDENCE_BUDGET {
                return Err(configuration(
                    "Replay archive payload exceeds evidence budget",
                ));
            }
            let archive = algorithmic_gas::RunArchive::<f64>::from_bytes(&bytes)?;
            if entry["archive_fingerprint_fnv1a64"] != fingerprint(&bytes)
                || entry["future_seed"] != archive.gas_config.seed
            {
                return Err(configuration(
                    "Embedded continuation archive does not match its manifest",
                ));
            }
        }
    }
    let result = run_internal(&evidence.run_config, &evidence.request, Some(&checkpoint)).await?;
    let regenerated: ReplayEvidence =
        serde_json::from_value(result.details["replay_evidence"].clone())
            .map_err(|e| configuration(&e.to_string()))?;
    if !json_numeric_equal(&json!(regenerated.runs), &json!(evidence.runs)) {
        return Err(configuration(
            "Regenerated continuation records differ from exported evidence",
        ));
    }
    Ok(result)
}
// JavaScript JSON encodes integral floating-point readouts as integers.
// Compare their numeric values while retaining exact integer identities.
fn json_numeric_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            match (x.as_u64(), y.as_u64(), x.as_i64(), y.as_i64()) {
                (Some(x), Some(y), _, _) => x == y,
                (_, _, Some(x), Some(y)) => x == y,
                _ => x.as_f64() == y.as_f64(),
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| json_numeric_equal(x, y))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_numeric_equal(v, w)))
        }
        _ => a == b,
    }
}

fn seed(checkpoint: &Checkpoint<f64>, group: u64, index: usize) -> u64 {
    checkpoint
        .config
        .seed
        .wrapping_add(1 + group * 10000 + index as u64)
}
fn innovation(seed: u64, step: u64, walker: usize, coordinate: usize) -> f64 {
    let mut rng = RandomStream::new(seed, step, Stream::Kinetic, walker as u64, 2);
    let mut value = 0.;
    for _ in 0..=coordinate {
        value = rng.gaussian::<f64>();
    }
    value
}
/// Run actual native gas continuations. Browser bindings invoke this same function.
pub async fn run(config: &RunConfig, request: &ExperimentRequest) -> Result<ExperimentResult> {
    run_internal(config, request, None).await
}
/// Measure continuations from the exact completed state displayed by a session.
pub async fn run_from_checkpoint(
    config: &RunConfig,
    request: &ExperimentRequest,
    checkpoint: &Checkpoint<f64>,
) -> Result<ExperimentResult> {
    if config.gas != checkpoint.config {
        return Err(configuration(
            "Continuation checkpoint differs from the executed configuration",
        ));
    }
    // Recording is audit data, not the Markov state. The complete baseline
    // archive is exported by the session; avoid duplicating it in each replica.
    let mut conditioning = checkpoint.clone();
    if request.experiment == 45 {
        let archive = conditioning.recording.as_mut().ok_or_else(|| {
            configuration("Metric conditioning requires its final recorded update")
        })?;
        let keep = conditioning
            .config
            .distance_donors
            .history_window
            .max(conditioning.config.cloning_donors.history_window)
            + 2;
        let start = archive.steps.len().saturating_sub(keep);
        archive.steps = archive.steps.split_off(start);
        let first = archive
            .steps
            .first()
            .ok_or_else(|| configuration("Metric conditioning has no recorded update"))?;
        archive.epoch = first.epoch;
        archive.anchors = vec![algorithmic_gas::tracking::ArchiveAnchor {
            epoch: first.epoch,
            step: first.report.step - 1,
            reason: "conditioning_window".into(),
            population: first.before.clone(),
        }];
    } else {
        conditioning.recording = None;
    }
    conditioning.validate()?;
    run_internal(config, request, Some(&conditioning)).await
}
async fn run_internal(
    config: &RunConfig,
    request: &ExperimentRequest,
    frozen: Option<&Checkpoint<f64>>,
) -> Result<ExperimentResult> {
    request.validate()?;
    if request.experiment == 45 {
        let mut result = metric_prediction(config, request, frozen).await?;
        algorithmic_gas::physics::partvi::annotate_result(
            request,
            &mut result,
            "independent_algorithm_continuations",
        )?;
        return Ok(result);
    }
    if !matches!(request.experiment, 19 | 22) {
        return Err(GasError::Capability(
            "native continuation runner supports experiments 19, 22, and 45".into(),
        ));
    }
    let replicas = request.usize("replicas", 32);
    let horizon = request.usize("horizon", 4);
    let warmup = request.usize("warmup", 2);
    if !(2..=128).contains(&replicas) || !(1..=16).contains(&horizon) || warmup > 16 {
        return Err(configuration(
            "continuation budgets: replicas 2..128, horizon 1..16, warmup 0..16",
        ));
    }
    if config.walkers > 256 {
        return Err(configuration(
            "interactive continuation experiments support at most 256 walkers",
        ));
    }
    let mut config = config.clone();
    config.gas.precision = Precision::F64;
    config.gas.backend = BackendKind::Cpu;
    if !matches!(&config.gas.kinetic.integrator, KineticKind::Baoab { positions, velocities, .. } if positions == "positions" && velocities == "velocities")
    {
        return Err(configuration(
            "native continuation experiments require BAOAB with positions/velocities fields",
        ));
    }
    if request.experiment == 19 && config.gas.kinetic.noise.innovation != InnovationLaw::Gaussian {
        return Err(configuration(
            "Gaussian source response requires Gaussian innovations",
        ));
    }
    if !config.gas.qft.innovation_shifts.is_empty() {
        return Err(configuration(
            "continuation reference run requires an empty source schedule",
        ));
    }
    let walker = request.usize("walker", 0);
    let coordinate = request.usize("coordinate", 0);
    if walker >= config.walkers || coordinate >= config.dimensions {
        return Err(configuration(
            "source walker/coordinate outside configured population",
        ));
    }
    let mut gas = config.build::<f64>().await?;
    if let Some(checkpoint) = frozen {
        gas.restore(checkpoint.clone())?;
    } else {
        for _ in 0..warmup {
            gas.step().await?;
        }
    }
    let checkpoint = gas.checkpoint();
    let observable = request.text(
        "observable",
        if request.experiment == 19 {
            "tagged_velocity"
        } else {
            "momentum"
        },
    );
    let mut result = if request.experiment == 19 {
        source_response(
            &config,
            &checkpoint,
            request,
            replicas,
            horizon,
            walker,
            coordinate,
            observable,
        )
        .await?
    } else {
        noether(
            &config,
            &checkpoint,
            request,
            replicas,
            horizon,
            walker,
            coordinate,
            observable,
        )
        .await?
    };
    result.details["request"] =
        serde_json::to_value(request).map_err(|e| configuration(&e.to_string()))?;
    result.details["run_config"] =
        serde_json::to_value(&config).map_err(|e| configuration(&e.to_string()))?;
    result.details["frozen_step"] = json!(checkpoint.step);
    result.details["frozen_population_version"] = json!(checkpoint.population.version);
    result.details["historical_frames"] = json!(checkpoint.history.len());
    result.details["source"] = json!("native_checkpoint_continuations");
    result.details["precision"] = json!("f64");
    algorithmic_gas::physics::partvi::annotate_result(
        request,
        &mut result,
        "independent_algorithm_continuations",
    )?;
    result.note("Every replica starts from the same complete population, provider configuration, donor history, and step counter. Independent future seeds separate calibration from validation.");
    Ok(result)
}
#[allow(clippy::too_many_arguments)]
async fn source_response(
    config: &RunConfig,
    checkpoint: &Checkpoint<f64>,
    request: &ExperimentRequest,
    replicas: usize,
    horizon: usize,
    walker: usize,
    coordinate: usize,
    observable: &str,
) -> Result<ExperimentResult> {
    let theta = request.number("theta", 0.25);
    if !theta.is_finite() || theta.abs() > 1.5 || theta.abs() < 0.01 {
        return Err(configuration("source theta must have magnitude 0.01..1.5"));
    }
    let source_step = checkpoint.step + 1;
    let mut weighted = vec![];
    let mut direct = vec![];
    let mut first = vec![];
    let mut second = vec![];
    let mut finite_first = vec![];
    let mut finite_second = vec![];
    let mut weights = vec![];
    let mut paired_delta = vec![];
    let mut rows = vec![];
    let mut survived = [0usize; 4];
    let mut evidence = EvidenceCollector::default();
    for k in 0..replicas {
        let aseed = seed(checkpoint, 0, k);
        let bseed = seed(checkpoint, 1, k);
        let a = continuation(
            config, checkpoint, aseed, horizon, None, observable, walker, coordinate,
        )
        .await?;
        let b = continuation(
            config, checkpoint, bseed, horizon, None, observable, walker, coordinate,
        )
        .await?;
        let address = |shift| {
            Some(InnovationShift {
                step: source_step,
                stream: Stream::Kinetic,
                substep: 2,
                walker,
                coordinate,
                shift,
            })
        };
        let plus = continuation(
            config,
            checkpoint,
            bseed,
            horizon,
            address(theta),
            observable,
            walker,
            coordinate,
        )
        .await?;
        let minus = continuation(
            config,
            checkpoint,
            bseed,
            horizon,
            address(-theta),
            observable,
            walker,
            coordinate,
        )
        .await?;
        for (name, c) in [
            ("weight_baseline", &a),
            ("direct_baseline", &b),
            ("direct_plus", &plus),
            ("direct_minus", &minus),
        ] {
            evidence.record(c, name, k, horizon)?;
        }
        let xi = innovation(aseed, source_step, walker, coordinate);
        let weight = (theta * xi - 0.5 * theta * theta).exp();
        weighted.push(a.value * weight);
        weights.push(weight);
        direct.push(plus.value);
        first.push(a.value * xi);
        second.push(a.value * (xi * xi - 1.));
        finite_first.push((plus.value - minus.value) / (2. * theta));
        finite_second.push((plus.value - 2. * b.value + minus.value) / (theta * theta));
        paired_delta.push(plus.value - b.value);
        for (j, c) in [&a, &b, &plus, &minus].iter().enumerate() {
            survived[j] += usize::from(c.survived);
        }
        rows.push(json!({"replica":k,"weight_seed":aseed,"direct_seed":bseed,"innovation":xi,"weight":weight,"baseline_weight_group":a.value,"baseline_direct_group":b.value,"plus":plus.value,"minus":minus.value}));
    }
    let (w, wse) = moments(&weighted);
    let (d, dse) = moments(&direct);
    let (h1, h1se) = moments(&first);
    let (h2, h2se) = moments(&second);
    let (fd1, fd1se) = moments(&finite_first);
    let (fd2, fd2se) = moments(&finite_second);
    let (delta, deltase) = moments(&paired_delta);
    let ess = weights.iter().sum::<f64>().powi(2) / weights.iter().map(|w| w * w).sum::<f64>();
    let mut out = ExperimentResult::new(
        19,
        "Gaussian source response on executed gas histories",
        "Independent checkpoint replicas; actual source-shifted BAOAB and full selection/cloning reevaluation",
    );
    out.metric("Direct shifted mean", d, observable)
        .metric("Reweighted mean", w, observable)
        .metric("Direct minus reweighted", d - w, observable)
        .metric(
            "Independent comparison standard error",
            (dse * dse + wse * wse).sqrt(),
            observable,
        )
        .metric("Common-random source change", delta, observable)
        .metric("Source-change standard error", deltase, observable)
        .metric("Importance effective sample size", ess, "replicas");
    out.plot(
        "Running independent estimates",
        "replicas",
        observable,
        vec![
            running("Direct source rerun", &direct),
            running("Exponential reweighting", &weighted),
        ],
    );
    out.plot(
        "First source derivative",
        "replicas",
        observable,
        vec![
            running("Hermite score", &first),
            running("Symmetric finite difference", &finite_first),
        ],
    );
    out.details = json!({"replicas":replicas,"horizon":horizon,"source_address":{"step":source_step,"stream":"Kinetic","substep":2,"walker":walker,"coordinate":coordinate,"theta":theta},"observable":observable,"samples":rows,"survived":survived,"direct_standard_error":dse,"weighted_standard_error":wse,"first_hermite":{"mean":h1,"standard_error":h1se},"first_finite_difference":{"mean":fd1,"standard_error":fd1se},"second_hermite":{"mean":h2,"standard_error":h2se},"second_finite_difference":{"mean":fd2,"standard_error":fd2se}});
    out.note("The exponential identity compares the finite source exactly. Hermite means are derivatives at zero; symmetric finite differences retain finite-theta bias and can cross cloning or validity thresholds.");
    out.note("Killed continuations have zero terminal readout. Reserved Gaussian addresses are evaluated even when that innovation is unused, so survival selection remains part of the observable.");
    out.note("tagged_velocity means tanh of the selected slot's velocity component; slot identity is fixed and cloning remains active.");
    evidence.attach(&mut out, config, request, checkpoint)?;
    Ok(out)
}
fn running(name: &str, values: &[f64]) -> Series {
    let mut total = 0.;
    Series::line(
        name,
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                total += v;
                [(i + 1) as f64, total / (i + 1) as f64]
            })
            .collect(),
    )
}
#[allow(clippy::too_many_arguments)]
async fn noether(
    config: &RunConfig,
    checkpoint: &Checkpoint<f64>,
    request: &ExperimentRequest,
    replicas: usize,
    horizon: usize,
    walker: usize,
    coordinate: usize,
    observable: &str,
) -> Result<ExperimentResult> {
    let initial = readout(
        &checkpoint.population,
        observable,
        walker,
        coordinate,
        config.gas.include_truncated,
    )?;
    let mut calibration = vec![];
    let mut validation = vec![];
    let mut ledgers: Vec<Vec<MechanicalStageBudget>> = vec![];
    let mut survives = [0usize; 2];
    let mut oracle_samples = vec![];
    let mut evidence = EvidenceCollector::default();
    let mut momentum_residuals = vec![];
    let mut momentum_variations = vec![];
    let mut energy_residuals = vec![];
    let mut energy_variations = vec![];
    let (dt, friction) = match config.gas.kinetic.integrator {
        KineticKind::Baoab { dt, friction, .. } => (dt, friction),
        _ => unreachable!(),
    };
    for k in 0..replicas {
        for group in 0..2 {
            let key = seed(checkpoint, group, k);
            let c = continuation(
                config, checkpoint, key, horizon, None, observable, walker, coordinate,
            )
            .await?;
            evidence.record(
                &c,
                if group == 0 {
                    "calibration"
                } else {
                    "validation"
                },
                k,
                horizon,
            )?;
            survives[group as usize] += usize::from(c.survived);
            let mut mr = 0.;
            let mut mv = 0.;
            let mut er = 0.;
            let mut ev = 0.;
            let mut covered = 0usize;
            let mut unavailable = vec![];
            if let Some(a) = c.gas.recording() {
                for step in &a.steps {
                    // A termination before O contributes no O-stage increment.
                    if !step.stages.iter().any(|s| s.stage == "O_before_boundary") {
                        continue;
                    }
                    if step
                        .stages
                        .iter()
                        .find(|s| s.stage == "A1")
                        .is_some_and(|s| {
                            !s.validity
                                .iter()
                                .any(|v| v.eligible(config.gas.include_truncated))
                        })
                    {
                        continue;
                    }
                    let prediction = step.thermostat_moments(
                        "velocities",
                        1.,
                        dt,
                        friction,
                        config.gas.include_truncated,
                    );
                    let budget =
                        algorithmic_gas::physics::balances::analyze_step(step, &config.gas, None);
                    match (prediction, budget) {
                        (Ok(p), Ok(b)) => {
                            if let Some(t) = b.thermostat {
                                let dimension = p.measured_momentum_change.len();
                                mr += p.measured_momentum_change[coordinate]
                                    - p.predicted_momentum_change[coordinate];
                                mv += p.momentum_covariance[coordinate * dimension + coordinate];
                                er += p.measured_energy_change - p.predicted_energy_change;
                                ev += t.energy_variance;
                                covered += 1;
                            } else {
                                unavailable.push("conditional energy variance absent".to_string());
                            }
                        }
                        (Err(e), _) | (_, Err(e)) => unavailable.push(e.to_string()),
                    }
                }
            }
            if unavailable.is_empty() {
                momentum_residuals.push(mr);
                momentum_variations.push(mv);
                energy_residuals.push(er);
                energy_variations.push(ev);
            }
            oracle_samples.push(json!({"seed":key,"group":group,"replica":k,"covered_o_steps":covered,"momentum_martingale":mr,"momentum_predictable_variation":mv,"energy_martingale":er,"energy_predictable_variation":ev,"unavailable":unavailable}));
            let sample = ReplicaSample {
                key,
                increment: vec![c.value - initial],
            };
            if group == 0 {
                calibration.push(sample);
            } else {
                validation.push(sample);
            }
            if k == 0
                && let Some(archive) = c.gas.recording()
            {
                for step in &archive.steps {
                    ledgers.push(step.mechanical_budgets(
                        "velocities",
                        1.,
                        config.gas.include_truncated,
                    )?);
                }
            }
        }
    }
    let report = metric_evolution(&calibration, &validation)?;
    let mut out = ExperimentResult::new(
        22,
        "Conditional Noether balance in the executed gas",
        "Independent calibration and validation of finite-horizon conditional increments from a complete checkpoint",
    );
    out.metric(
        "Calibrated conditional increment",
        report.conditional_increment_mean[0],
        observable,
    )
    .metric(
        "Calibration standard error",
        report.mean_standard_error[0],
        observable,
    )
    .metric(
        "Validation increment",
        report.validation_increment_mean[0],
        observable,
    )
    .metric(
        "Validation drift residual",
        report.validation_residual[0],
        observable,
    )
    .metric(
        "Independent residual standard error",
        report.validation_standard_error[0],
        observable,
    );
    out.plot(
        "Conditional drift and independent validation",
        "replicas",
        observable,
        vec![
            running(
                "Calibrated drift",
                &calibration
                    .iter()
                    .map(|s| s.increment[0])
                    .collect::<Vec<_>>(),
            ),
            running(
                "Independent increment",
                &validation
                    .iter()
                    .map(|s| s.increment[0])
                    .collect::<Vec<_>>(),
            ),
        ],
    );
    out.details = json!({"replicas_per_group":replicas,"horizon":horizon,"observable":observable,"initial_readout":initial,"report":report,"sample_stage_ledgers":ledgers,"survived":survives});
    let covered_o_stages: u64 = oracle_samples
        .iter()
        .filter_map(|s| s["covered_o_steps"].as_u64())
        .sum();
    if covered_o_stages == 0 {
        out.details["analytic_o_prediction"] = json!({"status":"unexercised","covered_o_stages":0,"samples":oracle_samples,"reason":"No eligible walker reached a covered O stage; no thermostat comparison was exercised"});
    } else if momentum_residuals.len() == 2 * replicas {
        let (momentum_mean, momentum_sem) = moments(&momentum_residuals);
        let (energy_mean, energy_sem) = moments(&energy_residuals);
        let momentum_oracle_se =
            (momentum_variations.iter().sum::<f64>()).sqrt() / momentum_residuals.len() as f64;
        let energy_oracle_se =
            (energy_variations.iter().sum::<f64>()).sqrt() / energy_residuals.len() as f64;
        out.metric(
            "Analytic O momentum martingale mean",
            momentum_mean,
            "momentum",
        )
        .metric(
            "Analytic O momentum mean standard error",
            momentum_oracle_se,
            "momentum",
        )
        .metric("Analytic O energy martingale mean", energy_mean, "energy")
        .metric(
            "Analytic O energy mean standard error",
            energy_oracle_se,
            "energy",
        );
        out.plot(
            "Executed O-stage analytic martingale check",
            "independent replicas",
            "residual",
            vec![
                running(
                    "Momentum increment minus exact conditional drift",
                    &momentum_residuals,
                ),
                running(
                    "Energy increment minus exact conditional drift",
                    &energy_residuals,
                ),
                Series::line(
                    "Analytic zero mean",
                    vec![[1., 0.], [momentum_residuals.len() as f64, 0.]],
                ),
            ],
        );
        out.details["analytic_o_prediction"] = json!({"status":"available","covered_o_stages":covered_o_stages,"samples":oracle_samples,"momentum_mean":momentum_mean,"energy_mean":energy_mean,"momentum_oracle_standard_error":momentum_oracle_se,"energy_oracle_standard_error":energy_oracle_se,"momentum_replica_sem":momentum_sem,"energy_replica_sem":energy_sem,"prediction":"Conditional BAOAB moments from each actual A1 state and recorded factor before its O innovation; all earlier clone, donor, force and boundary effects remain in that state.","sampling_unit":"A full independently seeded continuation; predictable quadratic variations sum over its O stages. Independent replicas supply the comparison of realized second moments with predicted variation."});
    } else {
        out.details["analytic_o_prediction"] = json!({"status":"unavailable","covered_o_stages":covered_o_stages,"samples":oracle_samples,"reason":"At least one sampled O stage lacks its actual conditional law"});
    }
    out.note("A symmetry does not require the raw increment to vanish: the independently calibrated conditional drift is subtracted to form the martingale residual. The displayed horizon is a discrete conditional transition.");
    out.note("The recorded sample ledgers separate changes of persistent active rows from eligibility changes at every raw kinetic and boundary stage; cloning is included in the executed histories.");
    evidence.attach(&mut out, config, request, checkpoint)?;
    Ok(out)
}

/// A fixed-chart, fixed-probe metric reconstructed from the last completed
/// selection context. The context is part of the retained extended state;
/// source identities and source coordinates are frozen during differentiation.
fn conditional_metric_readout(
    archive: &algorithmic_gas::RunArchive<f64>,
    walker: usize,
    probe: &[f64],
    metric: &crate::physics_metric::PhysicsMetricConfig,
    history: &[(u64, Population<f64>)],
) -> Result<(Vec<f64>, bool, serde_json::Value)> {
    let calculate = || -> Result<(Vec<f64>, serde_json::Value)> {
        let (jet, context) = algorithmic_gas::physics::fields::archive_fitness_jet_with_history(
            archive,
            archive.steps.len().saturating_sub(1),
            walker,
            probe,
            2,
            history,
        )?;
        let d = probe.len();
        let mut hessian = vec![0.; d * d];
        for i in 0..d {
            for j in 0..d {
                hessian[i * d + j] = jet.derivative(&[i, j])?;
            }
        }
        let geometry = algorithmic_gas::physics::geometry::metric_spectrum(
            &hessian,
            d,
            metric.epsilon,
            metric.policy,
        )?;
        let mut packed = vec![];
        for i in 0..d {
            for j in i..d {
                packed.push(geometry.metric[i * d + j]);
            }
        }
        Ok((packed, context))
    };
    match calculate() {
        Ok((value, context)) => Ok((value, true, context)),
        Err(error)
            if matches!(&error,
                GasError::Configuration(message) if message == "metric is not positive definite"
            ) || matches!(&error,
                GasError::Capability(message) if message == "conditional target was ineligible at the recorded pre-clone stage"
            ) =>
        {
            Ok((
                vec![0.; probe.len() * (probe.len() + 1) / 2],
                false,
                json!({"status":"undefined_metric_zero_extension","reason":error.to_string()}),
            ))
        }
        Err(error) => Err(error),
    }
}
async fn metric_prediction(
    config: &RunConfig,
    request: &ExperimentRequest,
    frozen: Option<&Checkpoint<f64>>,
) -> Result<ExperimentResult> {
    let replicas = request.usize("replicas", 16);
    let material = match request.text("readout", "material") {
        "material" => true,
        "fixed_probe" => false,
        _ => {
            return Err(configuration(
                "metric readout must be material or fixed_probe",
            ));
        }
    };
    let horizon = request.usize("horizon", 1);
    let warmup = request.usize("warmup", 1);
    if !(2..=32).contains(&replicas)
        || !(1..=4).contains(&horizon)
        || !(1..=4).contains(&warmup)
        || !(2..=64).contains(&config.walkers)
        || config.dimensions != 3
    {
        return Err(configuration(
            "metric continuation budgets: actual dimension 3, walkers 2..64, replicas 2..32 per group, horizon 1..4, warmup 1..4",
        ));
    }
    let walker = request.usize("walker", 0);
    if walker >= config.walkers {
        return Err(configuration("metric probe target outside population"));
    }
    let offset = request.number("probe_offset", 0.);
    if !offset.is_finite() || offset.abs() > 1. {
        return Err(configuration("fixed probe offset must be finite in [-1,1]"));
    }
    let mut config = config.clone();
    config.gas.precision = Precision::F64;
    config.gas.backend = BackendKind::Cpu;
    let metric = config
        .physics_metric
        .clone()
        .ok_or_else(|| configuration("actual metric replicas require RunConfig.physics_metric"))?;
    if !matches!(&config.gas.kinetic.integrator,KineticKind::Baoab{positions,velocities,..}if positions=="positions"&&velocities=="velocities")
    {
        return Err(configuration(
            "actual metric replicas require BAOAB with positions/velocities fields",
        ));
    }
    let mut gas = config.build::<f64>().await?;
    if let Some(checkpoint) = frozen {
        gas.restore(checkpoint.clone())?;
    } else {
        gas.start_recording(RecordingConfig {
            max_steps: warmup,
            max_bytes: config.gas.max_memory_bytes.min(128 * 1024 * 1024),
        })?;
        for _ in 0..warmup {
            gas.step().await?;
        }
    }
    let mut probe = gas
        .population()
        .observations
        .field("positions")?
        .row(walker)?
        .to_vec();
    probe[0] += offset;
    let archive = gas
        .recording()
        .ok_or_else(|| configuration("metric baseline archive unavailable"))?;
    let (initial, initial_available, baseline_context) =
        if material && !gas.population().validity[walker].eligible(config.gas.include_truncated) {
            (
                vec![0.; 6],
                false,
                json!({"status":"inactive_material_probe_zero_extension"}),
            )
        } else {
            conditional_metric_readout(archive, walker, &probe, &metric, &[])?
        };
    let checkpoint = gas.checkpoint();
    let mut calibration = vec![];
    let mut validation = vec![];
    let mut component_calibration = vec![];
    let mut component_validation = vec![];
    let mut available = [0usize; 2];
    let mut survived = [0usize; 2];
    let mut rows = vec![];
    let mut evidence = EvidenceCollector::default();
    for k in 0..replicas {
        for group in 0..2 {
            let key = seed(&checkpoint, group, k);
            let continuation = continuation(
                &config,
                &checkpoint,
                key,
                horizon,
                None,
                "alive_fraction",
                walker,
                0,
            )
            .await?;
            evidence.record(
                &continuation,
                if group == 0 {
                    "calibration"
                } else {
                    "validation"
                },
                k,
                horizon,
            )?;
            survived[group as usize] += usize::from(continuation.survived);
            let mut future_probe = probe.clone();
            let mut literal_probe = probe.clone();
            let mut fixed_value = vec![0.; 6];
            let mut literal_value = vec![0.; 6];
            let (value, valid, context) = if continuation.survived {
                if let Some(a) = continuation.gas.recording() {
                    fixed_value = conditional_metric_readout(
                        a,
                        walker,
                        &probe,
                        &metric,
                        &checkpoint.history,
                    )?
                    .0;
                    if material {
                        future_probe = continuation
                            .gas
                            .population()
                            .observations
                            .field("positions")?
                            .row(walker)?
                            .to_vec();
                        future_probe[0] += offset;
                        let step = a.steps.last().ok_or_else(|| {
                            configuration("metric continuation stage record unavailable")
                        })?;
                        let literal = step
                            .stages
                            .iter()
                            .find(|s| s.stage == "literal_clone")
                            .ok_or_else(|| {
                                configuration(
                                    "literal clone stage unavailable for metric decomposition",
                                )
                            })?;
                        literal_probe =
                            literal.fields["positions"].values[walker * 3..walker * 3 + 3].to_vec();
                        literal_probe[0] += offset;
                        literal_value = conditional_metric_readout(
                            a,
                            walker,
                            &literal_probe,
                            &metric,
                            &checkpoint.history,
                        )?
                        .0;
                        if continuation.gas.population().validity[walker]
                            .eligible(config.gas.include_truncated)
                        {
                            conditional_metric_readout(
                                a,
                                walker,
                                &future_probe,
                                &metric,
                                &checkpoint.history,
                            )?
                        } else {
                            (
                                vec![0.; 6],
                                false,
                                json!({"status":"inactive_material_probe_zero_extension"}),
                            )
                        }
                    } else {
                        literal_value = fixed_value.clone();
                        conditional_metric_readout(a, walker, &probe, &metric, &checkpoint.history)?
                    }
                } else {
                    return Err(configuration("metric continuation archive unavailable"));
                }
            } else {
                (
                    vec![0.; 6],
                    false,
                    json!({"status":"extinction_zero_extension"}),
                )
            };
            available[group as usize] += usize::from(valid);
            let increment = value
                .iter()
                .zip(&initial)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>();
            let mut components = Vec::with_capacity(18);
            components.extend(fixed_value.iter().zip(&initial).map(|(a, b)| a - b));
            components.extend(literal_value.iter().zip(&fixed_value).map(|(a, b)| a - b));
            components.extend(value.iter().zip(&literal_value).map(|(a, b)| a - b));
            let component_sample = ReplicaSample {
                key,
                increment: components.clone(),
            };
            let sample = ReplicaSample {
                key,
                increment: increment.clone(),
            };
            if group == 0 {
                calibration.push(sample);
                component_calibration.push(component_sample);
            } else {
                validation.push(sample);
                component_validation.push(component_sample);
            }
            rows.push(json!({"replica":k,"group":group,"future_seed":key,"survived":continuation.survived,"metric_available":valid,"metric":value,"increment":increment,"context":context,"future_probe":future_probe,"literal_probe":literal_probe,"fixed_probe_metric":fixed_value,"literal_probe_metric":literal_value,"field_clone_motion_increments":components}));
        }
    }
    let report = metric_evolution(&calibration, &validation)?;
    let component_report = metric_evolution(&component_calibration, &component_validation)?;
    let mut cross_covariance_sum = vec![0.; 36];
    for i in 0..6 {
        for j in 0..6 {
            for a in 0..3 {
                for b in 0..3 {
                    cross_covariance_sum[i * 6 + j] +=
                        component_report.increment_covariance[(a * 6 + i) * 18 + b * 6 + j];
                }
            }
        }
    }
    let covariance_residual = cross_covariance_sum
        .iter()
        .zip(&report.increment_covariance)
        .map(|(a, b)| (a - b).abs())
        .fold(0_f64, f64::max);
    let mut out = ExperimentResult::new(
        45,
        "Conditional metric evolution in the executed three-dimensional gas",
        if material {
            "Complete-checkpoint independent replicas; metric evaluated at the final slot position with exact field, clone-probe and motion decomposition"
        } else {
            "Complete-checkpoint independent replicas; fixed chart probe and last completed pre-clone conditional fitness context"
        },
    );
    out.metric(
        "Calibration metric availability",
        available[0] as f64 / replicas as f64,
        "probability",
    )
    .metric(
        "Validation metric availability",
        available[1] as f64 / replicas as f64,
        "probability",
    )
    .metric(
        "Conditional g11 increment",
        report.conditional_increment_mean[0],
        "metric",
    )
    .metric(
        "Independent g11 increment",
        report.validation_increment_mean[0],
        "metric",
    )
    .metric(
        "Independent g11 residual standard error",
        report.validation_standard_error[0],
        "metric",
    )
    .metric(
        "Validation extinction probability",
        1. - survived[1] as f64 / replicas as f64,
        "probability",
    );
    out.plot(
        if material {
            "Material metric conditional increment"
        } else {
            "Fixed-probe conditional metric drift"
        },
        "packed component (11,12,13,22,23,33)",
        "increment",
        vec![
            Series::line(
                "Independent calibration",
                report
                    .conditional_increment_mean
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
            Series::line(
                "Independent validation",
                report
                    .validation_increment_mean
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
        ],
    );
    out.plot(
        "Independent residual and uncertainty",
        "packed component",
        "increment",
        vec![
            Series::line(
                "Validation minus calibration",
                report
                    .validation_residual
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
            Series::line(
                "Estimated +2 standard errors",
                report
                    .validation_standard_error
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, 2. * x])
                    .collect(),
            ),
            Series::line(
                "Estimated -2 standard errors",
                report
                    .validation_standard_error
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, -2. * x])
                    .collect(),
            ),
        ],
    );
    out.plot(
        "Conditional metric increment covariance",
        "flattened 6 by 6 entry",
        "metric squared",
        vec![Series::line(
            "Unbiased calibration covariance",
            report
                .increment_covariance
                .iter()
                .enumerate()
                .map(|(i, &x)| [i as f64, x])
                .collect(),
        )],
    );
    out.metric(
        "Component covariance telescoping residual",
        covariance_residual,
        "metric squared",
    );
    out.plot(
        "Sources of the conditional material metric increment",
        "packed field (0..5), clone probe (6..11), motion (12..17)",
        "increment",
        vec![
            Series::line(
                "Calibration component means",
                component_report
                    .conditional_increment_mean
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect(),
            ),
            Series::line(
                "Independent component means",
                component_report
                    .validation_increment_mean
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect(),
            ),
        ],
    );
    out.details = json!({"readout":if material {"material"} else {"fixed_probe"},"component_report":component_report,"component_covariance_sum":cross_covariance_sum,"component_covariance_residual":covariance_residual,"decomposition":"new field at old probe minus old field; new field at last literal-clone probe minus new field at old probe; new field at final probe minus new field at last literal-clone probe. At horizon1 the middle term is literal-clone probe replacement; longer horizons also include earlier motion. Cross covariances are retained.","request":request,"run_config":config,"replicas_per_group":replicas,"horizon":horizon,"conditioning_steps":checkpoint.step,"probe":probe,"target_slot":walker,"baseline_metric":initial,"baseline_metric_available":initial_available,"baseline_context":baseline_context,"report":report,"samples":rows,"available_counts":available,"survived_counts":survived,"source":"native_checkpoint_continuations","frozen_step":checkpoint.step,"historical_frames":checkpoint.history.len(),"cemetery_extension":"Every unavailable metric or extinct outcome has the full packed zero metric; all replicas remain in both mean and covariance.","probe_convention":if material {"fixed slot, final population position plus constant first-coordinate offset; last completed pre-clone conditional context"} else {"last completed step pre-clone population and its sampled immutable donor context, evaluated at one unchanged chart point"}});
    out.note("The complete extended checkpoint, including donor memory and provider configuration, is fixed before resampling future seeds. Warmup supplies the baseline recorded companion context.");
    out.note("The material readout retains same-step clone replacement, clone transforms, kinetics and boundary motion through the final slot position. Its field-versus-probe decomposition is an exact component evaluation identity in one chart, without assuming a differentiable clone pullback. The fixed-probe alternative is a lagged selection-context observable.");
    out.note("Metric availability and extinction probabilities are measured explicitly. The displayed observable uses a zero metric on undefined outcomes; no replica is discarded or survivor-renormalized.");
    evidence.attach(&mut out, &config, request, &checkpoint)?;
    Ok(out)
}
