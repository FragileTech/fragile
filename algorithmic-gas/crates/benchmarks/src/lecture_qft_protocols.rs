//! Paired complete-engine symmetry and intervention experiments.
//! Each pair shares random addresses; distinct pairs have independent seeds.
use crate::{BenchmarkModel, RunConfig, ShiftedReward};
use algorithmic_gas::{
    AlgorithmicGas, ExecutionContext, GasBuilder, GasError, Population, Result, RunArchive,
    TensorBatch,
    boundary::BoundaryPolicy,
    domain::{GradientProvider, OperatorFuture},
    fitness::ObjectiveDirection,
    noise::{FactorValues, NoiseGeometry},
    physics::partvi::{ExperimentResult, Series},
};
use serde_json::{Value, json};
fn err(s: &str) -> GasError {
    GasError::Configuration(s.into())
}
fn number(p: &Value, k: &str, d: f64) -> f64 {
    p.get(k).and_then(Value::as_f64).unwrap_or(d)
}
fn translated_system(p: &Value) -> bool {
    p.get("translation_mode")
        .and_then(Value::as_str)
        .unwrap_or("translated_system")
        == "translated_system"
}
pub fn supports(id: &str) -> bool {
    matches!(id, "VI-15" | "VI-30" | "VI-31")
}
fn translate_boundary(b: &mut BoundaryPolicy, shift: f64) {
    match b {
        BoundaryPolicy::AbsorbingBox { domain, .. }
        | BoundaryPolicy::PeriodicBox { domain, .. } => {
            domain.lower[0] += shift;
            domain.upper[0] += shift;
        }
        BoundaryPolicy::Composed { policies } => {
            for b in policies {
                translate_boundary(b, shift);
            }
        }
        _ => {}
    }
}
pub fn configs(id: &str, p: &Value, seed: u64) -> Result<Vec<RunConfig>> {
    if !supports(id) {
        return Err(err("Unsupported paired QFT protocol"));
    }
    let replicas = number(p, "replicas", 4.) as usize;
    if !(1..=32).contains(&replicas) {
        return Err(err("QFT protocol replicas must be 1..32"));
    }
    if id == "VI-30"
        && !["translated_system", "fixed_objective"].contains(
            &p.get("translation_mode")
                .and_then(Value::as_str)
                .unwrap_or("translated_system"),
        )
    {
        return Err(err(
            "translation_mode must be translated_system or fixed_objective",
        ));
    }
    let base = crate::lecture::gas_config(id, p, seed)?;
    let mut out = vec![];
    for pair in 0..replicas {
        let mut baseline = base.clone();
        baseline.gas.seed = seed.wrapping_add((pair as u64).wrapping_mul(104729));
        let mut transformed = baseline.clone();
        if id == "VI-15" {
            // Antithetic innovation factor realizes the parity-transformed noise
            // with the same symmetric law and exactly matched random addresses.
            transformed.gas.kinetic.noise.geometry = NoiseGeometry::Diagonal {
                factor: FactorValues::Constant {
                    values: vec![-0.8; transformed.dimensions],
                },
            };
        }
        if id == "VI-30" && translated_system(p) {
            let shift = number(p, "amplitude", 0.5);
            transformed.reward_shift = vec![0.; transformed.dimensions];
            transformed.reward_shift[0] = shift;
            translate_boundary(&mut transformed.gas.boundary, shift);
        }
        baseline.validate()?;
        transformed.validate()?;
        out.extend([baseline, transformed]);
    }
    Ok(out)
}
struct TranslatedGradient {
    model: BenchmarkModel,
    shift: Vec<f64>,
}
impl GradientProvider<f64> for TranslatedGradient {
    fn id(&self) -> String {
        format!(
            "lecture-translated-gradient/{:?}/{:?}",
            self.model.benchmark, self.shift
        )
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            let mut q = p.clone();
            let x = q.observations.field_mut("positions")?;
            let d = x.width();
            for row in x.values_mut().chunks_mut(d) {
                for (x, s) in row.iter_mut().zip(&self.shift) {
                    *x -= s;
                }
            }
            self.model.gradient(&q, cx).await
        })
    }
}
pub async fn build(
    id: &str,
    p: &Value,
    run_index: usize,
    c: &RunConfig,
) -> Result<AlgorithmicGas<f64>> {
    if id != "VI-30" || run_index.is_multiple_of(2) || !translated_system(p) {
        return c.build().await;
    }
    let model = BenchmarkModel {
        benchmark: c.benchmark,
        field: "positions".into(),
        direction: c.gas.fitness.direction,
    };
    let gradient = BenchmarkModel {
        benchmark: c.potential.unwrap_or(c.benchmark),
        direction: if c.potential.is_some() {
            ObjectiveDirection::Minimize
        } else {
            model.direction
        },
        ..model.clone()
    };
    // Translate before provider and boundary initialization, so even a large
    // coordinate shift cannot kill a valid baseline row before recording.
    let mut population = c.initial_population()?;
    let x = population.observations.field_mut("positions")?;
    let d = x.width();
    for row in x.values_mut().chunks_mut(d) {
        row[0] += c.reward_shift[0];
    }
    GasBuilder::new(
        population,
        ShiftedReward {
            model,
            shift: c.reward_shift.clone(),
        },
    )
    .config(c.gas.clone())
    .gradient(TranslatedGradient {
        model: gradient,
        shift: c.reward_shift.clone(),
    })
    .build()
    .await
}
pub async fn initialize(
    id: &str,
    p: &Value,
    run_index: usize,
    gas: &mut AlgorithmicGas<f64>,
) -> Result<()> {
    if run_index.is_multiple_of(2) || (id == "VI-30" && translated_system(p)) {
        return Ok(());
    }
    let mut pop = gas.population().clone();
    let d = pop.observations.field("positions")?.width();
    match id {
        "VI-15" => {
            for name in ["positions", "velocities"] {
                for x in pop.observations.field_mut(name)?.values_mut() {
                    *x = -*x;
                }
            }
        }
        "VI-30" => {
            for x in pop
                .observations
                .field_mut("positions")?
                .values_mut()
                .chunks_mut(d)
            {
                x[0] += number(p, "amplitude", 0.5);
            }
        }
        "VI-31" => {
            let slot = number(p, "source_slot", 0.) as usize;
            if slot >= pop.len() {
                return Err(err("Intervention source slot is outside the population"));
            }
            pop.observations.field_mut("positions")?.values_mut()[slot * d] +=
                number(p, "amplitude", 0.1);
        }
        _ => return Err(err("Unsupported paired QFT initialization")),
    }
    gas.replace_population(pop).await
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
fn summarize(v: &[f64]) -> (f64, Option<f64>) {
    let m = mean(v);
    let se = (v.len() >= 4).then(|| {
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((v.len() - 1) * v.len()) as f64).sqrt()
    });
    (m, se)
}
pub fn analyze_ensemble(
    id: &str,
    p: &Value,
    archives: &[RunArchive<f64>],
) -> Result<ExperimentResult> {
    if archives.len() < 2 || !archives.len().is_multiple_of(2) {
        return Err(err(
            "Paired QFT analysis requires complete baseline/transformed pairs",
        ));
    }
    let steps = archives.iter().map(|a| a.steps.len()).min().unwrap();
    if steps == 0 {
        return Err(err("Advance paired gas runs before analysis"));
    }
    let experiment = id
        .strip_prefix("VI-")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| err("Invalid QFT ID"))?;
    let title = match id {
        "VI-15" => "Complete algorithm parity and color-conjugation experiment",
        "VI-30" => "Complete algorithm translation covariance and response",
        _ => "Complete algorithm local intervention propagation",
    };
    let mut result = ExperimentResult::new(
        experiment,
        title,
        "Paired independently seeded full Euclidean Gas runs with matched random addresses, actual donor memory, clipping, cloning, viscosity and finite-step kinetics",
    );
    let mut error_curve = vec![];
    let mut response_curve = vec![];
    let mut uncertainty = vec![];
    let mut footprint_curve = vec![];
    let mut rows = vec![];
    for t in 0..steps {
        let mut differences = vec![];
        let mut squared = vec![];
        let mut footprints = vec![];
        let mut gate_changes = 0;
        let mut velocity_squared = Vec::new();
        let mut eligibility_changes = 0;
        let mut donor_changes = 0;
        for (pair, aa) in archives.chunks_exact(2).enumerate() {
            let b = &aa[0].steps[t];
            let c = &aa[1].steps[t];
            if b.report.step != c.report.step {
                return Err(err("Paired archive update times differ"));
            }
            let xb = b.final_population.observations.field("positions")?;
            let xc = c.final_population.observations.field("positions")?;
            if xb.rows() != xc.rows() || xb.item_shape() != xc.item_shape() {
                return Err(err("Paired population shapes differ"));
            }
            let n = b.final_population.len();
            let d = xb.width();
            let mut delta = 0.;
            let mut mse = 0.;
            let mut footprint = 0;
            for i in 0..n {
                let row_b = xb.row(i)?;
                let row_c = xc.row(i)?;
                let mut local = 0.;
                for j in 0..d {
                    let expected = match id {
                        "VI-15" => -row_b[j],
                        "VI-30" if translated_system(p) => {
                            row_b[j]
                                + if j == 0 {
                                    number(p, "amplitude", 0.5)
                                } else {
                                    0.
                                }
                        }
                        _ => row_b[j],
                    };
                    let residual = row_c[j] - expected;
                    if j == 0 {
                        delta += residual / n as f64;
                    }
                    mse += residual * residual / (n * d) as f64;
                    local += residual * residual;
                }
                if local > 1e-16 {
                    footprint += 1;
                }
                if b.report.clone_plan.choices[i].accepted
                    != c.report.clone_plan.choices[i].accepted
                {
                    gate_changes += 1;
                }
            }
            let vb = b.final_population.observations.field("velocities")?;
            let vc = c.final_population.observations.field("velocities")?;
            velocity_squared.push(
                vb.values()
                    .iter()
                    .zip(vc.values())
                    .map(|(&b, &c)| (c - if id == "VI-15" { -b } else { b }).powi(2))
                    .sum::<f64>()
                    / (n * d) as f64,
            );
            for i in 0..n {
                eligibility_changes += usize::from(
                    b.final_population.validity[i].eligible(aa[0].gas_config.include_truncated)
                        != c.final_population.validity[i]
                            .eligible(aa[1].gas_config.include_truncated),
                );
                let labels = |s: &algorithmic_gas::tracking::RecordedStep<f64>| {
                    s.report.clone_plan.choices[i]
                        .donors
                        .iter()
                        .map(|d| {
                            let src = s.report.clone_plan.sources[d.pool_index as usize];
                            (src.frame, src.slot, src.generation, d.weight.to_bits())
                        })
                        .collect::<Vec<_>>()
                };
                donor_changes += usize::from(labels(b) != labels(c));
            }
            differences.push(delta);
            squared.push(mse);
            footprints.push(footprint as f64 / n as f64);
            rows.push(json!({"pair":pair,"step":b.report.step,"seed":aa[0].gas_config.seed,"position_mean_residual":delta,"position_mse":mse,"affected_fraction":footprint as f64/n as f64,"baseline_eligible":b.report.eligible,"transformed_eligible":c.report.eligible}));
        }
        let (m, se) = summarize(&differences);
        response_curve.push([t as f64 + 1., m]);
        error_curve.push([t as f64 + 1., mean(&squared).sqrt()]);
        footprint_curve.push([t as f64 + 1., mean(&footprints)]);
        if let Some(se) = se {
            uncertainty.push([t as f64 + 1., se]);
        }
        if t + 1 == steps {
            result
                .metric("Final mean-position residual", m, "")
                .metric("Final position RMS residual", mean(&squared).sqrt(), "")
                .metric(
                    "Changed clone gates at final update",
                    gate_changes as f64,
                    "gates",
                );
            result
                .metric(
                    "Final velocity RMS residual",
                    mean(&velocity_squared).sqrt(),
                    "",
                )
                .metric(
                    "Changed eligibility at final update",
                    eligibility_changes as f64,
                    "slots",
                )
                .metric(
                    "Changed donor sets at final update",
                    donor_changes as f64,
                    "recipients",
                );
            if let Some(se) = se {
                result.metric("Independent-pair standard error", se, "");
            }
        }
    }
    result.plot(
        "Full algorithm paired evolution",
        "executed update",
        "residual",
        vec![
            Series::line("Mean position difference", response_curve),
            Series::line("RMS position difference", error_curve),
            Series::line("Independent-pair standard error", uncertainty),
        ],
    );
    result.plot(
        "Observed intervention footprint",
        "executed update",
        "fraction of slots",
        vec![Series::line(
            "Position residual above 1e-8 norm",
            footprint_curve,
        )],
    );
    result.details = json!({"status":"available","source":"executed_rust_euclidean_gas_ensemble","paired_runs":archives.len()/2,"rows":rows,"gas_configs":archives.iter().map(|a|&a.gas_config).collect::<Vec<_>>(),"providers":archives.iter().map(|a|&a.providers).collect::<Vec<_>>(),"transformation":match id{"VI-15"=>"x and v reversed; isotropic Gaussian or uniform innovations reflected through a negative diagonal factor with the same covariance. Even objectives predict parity covariance; asymmetric objectives can fail it. Force-based colors consequently conjugate up to the common minus sign.","VI-30" if translated_system(p)=>"Shift x coordinate 0, reward optimum, force potential and box boundaries by the same amount; compare shifted full trajectories with baseline trajectories.","VI-30"=>"Shift only initial x coordinate 0 while keeping objective and potential fixed; measured differences are physical algorithm responses, not symmetry residuals.",_=>"Shift one initial walker coordinate, then execute all subsequent donor selection, cloning decisions, history, force and noise operations without freezing gates."},"uncertainty":"Pairs use common random addresses. Pair seeds are independent; standard errors use the across-pair sample variance and require at least four pairs. Within-run walkers and times are never counted as independent replicas.","validation":"Symmetry residuals compare complete executed trajectories, including changes in clone decisions and eligibility. For local interventions, the measured footprint includes endogenous selection and kinetic propagation."});
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(id: &str, p: Value) -> ExperimentResult {
        futures_lite::future::block_on(async {
            let cs = configs(id, &p, 516).unwrap();
            let mut archives = vec![];
            for (i, c) in cs.iter().enumerate() {
                let mut gas = build(id, &p, i, c).await.unwrap();
                initialize(id, &p, i, &mut gas).await.unwrap();
                gas.start_recording(Default::default()).unwrap();
                for _ in 0..12 {
                    gas.step().await.unwrap();
                }
                archives.push(gas.recording().unwrap().clone());
            }
            analyze_ensemble(id, &p, &archives).unwrap()
        })
    }
    fn metric(r: &ExperimentResult, label: &str) -> f64 {
        r.metrics
            .iter()
            .find(|m| m.label == label)
            .unwrap()
            .value
            .unwrap()
    }
    #[test]
    fn even_objective_obeys_complete_algorithm_parity_with_memory() {
        let r = run("VI-15", json!({"walkers":8,"replicas":4}));
        assert!(metric(&r, "Final position RMS residual") < 1e-8);
        assert!(metric(&r, "Independent-pair standard error") < 1e-8);
    }
    #[test]
    fn translating_both_objective_and_force_preserves_full_dynamics() {
        let r = run("VI-30", json!({"walkers":8,"replicas":4,"amplitude":0.5}));
        assert!(metric(&r, "Final position RMS residual") < 1e-8);
    }
    #[test]
    fn fixed_objective_and_local_interventions_have_actual_responses() {
        let r = run(
            "VI-30",
            json!({"walkers":8,"replicas":4,"amplitude":0.5,"translation_mode":"fixed_objective"}),
        );
        assert!(metric(&r, "Final position RMS residual") > 1e-4);
        let r = run("VI-31", json!({"walkers":8,"replicas":4,"amplitude":0.2}));
        assert!(metric(&r, "Final position RMS residual") > 1e-6);
    }
    #[test]
    fn asymmetric_objective_exhibits_parity_breaking() {
        let result = run(
            "VI-15",
            json!({"walkers":8,"replicas":4,"benchmark":"styblinski_tang","engine_dt":0.01}),
        );
        assert!(metric(&result, "Final position RMS residual") > 1e-6);
    }
    #[test]
    fn translated_boundary_is_applied_after_initial_coordinate_shift() {
        let result = run(
            "VI-30",
            json!({"walkers":8,"replicas":4,"amplitude":10.,"engine_boundary":"absorbing_box"}),
        );
        assert!(metric(&result, "Final position RMS residual") < 1e-7);
        assert_eq!(metric(&result, "Changed eligibility at final update"), 0.);
    }
}
