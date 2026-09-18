//! Fixed-assignment and exact companion-law sensitivities on an executed swarm.
use crate::{Benchmark, lecture_early::config};
use algorithmic_gas::{
    GasError, ObservationBatch, Provenance, Result, RewardBatch, RunArchive, TensorBatch,
    donor::SamplingLaw,
    fitness::FitnessPipeline,
    geometry::{AlgorithmicDistance, InteractionKernel, Kernel},
    physics::{
        fitness::{local_log_weights, pipeline_from_measurement_jets},
        jet::{Jet, JetSpace},
        partvi::{ExperimentResult, Series},
    },
};
use serde_json::{Value, json};

fn error(s: &str) -> GasError {
    GasError::Configuration(s.into())
}
fn number(p: &Value, key: &str, default: f64) -> f64 {
    p.get(key).and_then(Value::as_f64).unwrap_or(default)
}
struct Context {
    points: Vec<Vec<f64>>,
    rewards: Vec<f64>,
    alive: Vec<bool>,
    target: usize,
    assignment: Vec<usize>,
    benchmark: Benchmark,
    pipeline: FitnessPipeline,
    donor: algorithmic_gas::donor::DonorModule,
}
impl Context {
    fn new(p: &Value, a: &RunArchive<f64>) -> Result<Self> {
        let step = a
            .steps
            .last()
            .ok_or_else(|| error("Taylor experiment needs a recorded step"))?;
        let module = &a.gas_config.distance_donors;
        if module.history_window != 0 || module.count != 1 || module.law != SamplingLaw::Independent
        {
            return Err(error(
                "Taylor companion-law experiment requires one independent current-frame donor per row",
            ));
        }
        if module.distance != algorithmic_gas::geometry::Distance::default() {
            return Err(error(
                "Taylor experiment requires unscaled Euclidean position distance",
            ));
        }
        let points = step.before.observations.field("positions")?;
        let alive = step.report.pre_clone_eligible.clone();
        let target = alive
            .iter()
            .position(|&v| v)
            .ok_or_else(|| error("no eligible target"))?;
        let batch = &step.report.distance_companions;
        let assignment = (0..alive.len())
            .map(|i| {
                if batch.valid[i] {
                    step.report.distance_sources[batch.indices[i] as usize].slot as usize
                } else {
                    i
                }
            })
            .collect();
        Ok(Self {
            points: points
                .values()
                .chunks(points.width())
                .map(|p| p.to_vec())
                .collect(),
            rewards: step.report.pre_clone_rewards.raw.clone(),
            alive,
            target,
            assignment,
            benchmark: config("IV-07", p, a.gas_config.seed)?.benchmark,
            pipeline: a.gas_config.fitness.clone(),
            donor: module.clone(),
        })
    }
    fn positions(&self, delta: f64) -> Vec<Vec<f64>> {
        let mut x = self.points.clone();
        x[self.target][0] += delta;
        x
    }
    // Independent reference: production scalar objective, distance and fitness APIs.
    fn scalar(&self, delta: f64, assignment: &[usize]) -> Result<f64> {
        let x = self.positions(delta);
        let obs = ObservationBatch::positions(TensorBatch::vectors(
            x.len(),
            x[0].len(),
            x.iter().flatten().copied().collect(),
        )?);
        let mut rewards = self.rewards.clone();
        rewards[self.target] = self.benchmark.value(&x[self.target])?;
        let rewards = RewardBatch::new(rewards, Provenance::default());
        let distances = (0..x.len())
            .map(|i| self.donor.distance.compare(&obs, i, &obs, assignment[i]))
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .pipeline
            .evaluate(&rewards, &distances, &self.alive, &obs, 0)?
            .fitness[self.target])
    }
    fn coordinates(&self, delta: f64, order: usize) -> Result<Vec<Vec<Jet<f64>>>> {
        let space = JetSpace::new(1, order)?;
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                p.iter()
                    .enumerate()
                    .map(|(axis, &v)| {
                        if i == self.target && axis == 0 {
                            if order > 0 {
                                space.variable(v + delta, 0)
                            } else {
                                Ok(space.constant(v + delta))
                            }
                        } else {
                            Ok(space.constant(v))
                        }
                    })
                    .collect()
            })
            .collect()
    }
    fn jet(&self, delta: f64, order: usize, assignment: &[usize]) -> Result<Jet<f64>> {
        let x = self.coordinates(delta, order)?;
        let space = &x[0][0].space;
        let mut rewards: Vec<_> = self.rewards.iter().map(|&v| space.constant(v)).collect();
        rewards[self.target] = self
            .benchmark
            .physics_objective(x[0].len())?
            .evaluate(&x[self.target])?;
        // All incoming edges respond when their selected source is the target.
        let distances: Vec<_> = (0..x.len())
            .map(|i| {
                x[i].iter()
                    .zip(&x[assignment[i]])
                    .fold(
                        space.constant(self.pipeline.distance_floor.powi(2)),
                        |s, (u, v)| s.add(&u.sub(v).pow(2.)),
                    )
                    .pow(0.5)
            })
            .collect();
        let points = self.positions(delta);
        let rlogs = local_log_weights(
            &x[self.target],
            &points,
            &[],
            &self.alive,
            self.target,
            &self.pipeline.reward_standardizer,
        )?;
        let dlogs = local_log_weights(
            &x[self.target],
            &points,
            &[],
            &self.alive,
            self.target,
            &self.pipeline.diversity_standardizer,
        )?;
        pipeline_from_measurement_jets(
            &rewards,
            &distances,
            &self.alive,
            self.target,
            &self.pipeline,
            rlogs.as_deref(),
            dlogs.as_deref(),
        )
    }
    fn probabilities(&self, delta: f64, order: usize) -> Result<Vec<Vec<Jet<f64>>>> {
        let x = self.coordinates(delta, order)?;
        let s = &x[0][0].space;
        (0..x.len())
            .map(|i| {
                if !self.alive[i] {
                    return Ok((0..x.len())
                        .map(|j| s.constant(if i == j { 1. } else { 0. }))
                        .collect());
                }
                let mut logs = Vec::new();
                for j in 0..x.len() {
                    if self.alive[j] && (self.donor.allow_self || i != j) {
                        let d2 = x[i]
                            .iter()
                            .zip(&x[j])
                            .fold(s.constant(0.), |a, (u, v)| a.add(&u.sub(v).pow(2.)));
                        let log = match self.donor.kernel {
                            Kernel::Uniform => s.constant(0.),
                            Kernel::Gaussian { width } => d2.scale(-0.5 / (width * width)),
                            _ => {
                                return Err(error(
                                    "Taylor companion law requires a Gaussian or uniform kernel",
                                ));
                            }
                        };
                        logs.push((j, log));
                    }
                }
                if logs.is_empty() {
                    return Ok((0..x.len())
                        .map(|j| s.constant(if i == j { 1. } else { 0. }))
                        .collect());
                }
                let peak = logs
                    .iter()
                    .map(|(_, v)| v.value())
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut row = vec![s.constant(0.); x.len()];
                let mut mass = s.constant(0.);
                for (j, l) in logs {
                    row[j] = l.sub(&s.constant(peak)).exp();
                    mass = mass.add(&row[j]);
                }
                Ok(row.into_iter().map(|v| v.mul(&mass.pow(-1.))).collect())
            })
            .collect()
    }
    fn assignments(&self) -> Option<Vec<Vec<usize>>> {
        let mut out = vec![Vec::new()];
        for i in 0..self.points.len() {
            let mut candidates: Vec<_> = (0..self.points.len())
                .filter(|&j| self.alive[i] && self.alive[j] && (self.donor.allow_self || i != j))
                .collect();
            if candidates.is_empty() {
                candidates.push(i);
            }
            if out.len() * candidates.len() > 4096 {
                return None;
            }
            out = out
                .into_iter()
                .flat_map(|row| {
                    candidates.iter().map(move |&j| {
                        let mut r = row.clone();
                        r.push(j);
                        r
                    })
                })
                .collect();
        }
        Some(out)
    }
    fn production_probabilities(&self, delta: f64) -> Result<Vec<Vec<f64>>> {
        let x = self.positions(delta);
        let obs = ObservationBatch::positions(TensorBatch::vectors(
            x.len(),
            x[0].len(),
            x.iter().flatten().copied().collect(),
        )?);
        (0..x.len()).map(|i| {
            let mut logs = vec![f64::NEG_INFINITY; x.len()];
            for (j,l) in logs.iter_mut().enumerate() {
                if self.alive[i] && self.alive[j] && (self.donor.allow_self || i != j) {
                    let d = self.donor.distance.compare(&obs, i, &obs, j)?;
                    *l = self.donor.kernel.log_weight(d, <algorithmic_gas::geometry::Distance as AlgorithmicDistance<f64>>::comparison_kind(&self.donor.distance))?;
                }
            }
            let peak = logs.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            if !peak.is_finite() { return Ok((0..x.len()).map(|j|if i==j {1.} else {0.}).collect()); }
            let w: Vec<_> = logs.iter().map(|l|(l-peak).exp()).collect();
            let total: f64 = w.iter().sum(); Ok(w.iter().map(|v|v/total).collect())
        }).collect()
    }
}
fn polynomial(c: &[f64], delta: f64) -> f64 {
    c.iter().rev().fold(0., |s, v| s * delta + v)
}
fn line(name: &str, points: Vec<[f64; 2]>) -> Series {
    Series::line(name, points)
}

pub fn analyze(p: &Value, a: &RunArchive<f64>, r: &mut ExperimentResult) -> Result<()> {
    let cx = Context::new(p, a)?;
    let order = (number(p, "order", 6.) as usize).clamp(1, 12);
    let requested = number(p, "radius", 0.2).clamp(1e-6, 0.5);
    let tolerance = number(p, "tolerance", 1e-5).clamp(1e-10, 0.1);
    let jet = cx.jet(0., order, &cx.assignment)?;
    let coefficients = &jet.coefficients;
    let check = |radius: f64, stagger: bool| -> Result<f64> {
        let mut worst: f64 = 0.;
        for k in 0..65 {
            let t = if stagger {
                (k as f64 + 0.5) / 65.
            } else {
                k as f64 / 64.
            };
            let delta = radius * (2. * t - 1.);
            let actual = cx.scalar(delta, &cx.assignment)?;
            worst =
                worst.max((actual - polynomial(coefficients, delta)).abs() / (1. + actual.abs()));
        }
        Ok(worst)
    };
    let mut radius = requested;
    let mut accepted = false;
    for _ in 0..30 {
        if check(radius, false)? <= tolerance && check(radius, true)? <= tolerance {
            accepted = true;
            break;
        }
        radius *= 0.5;
    }
    if !accepted {
        return Err(error(
            "No displacement window passed the production fitness comparison",
        ));
    }
    let mut direct = vec![];
    let mut taylor = vec![];
    let mut errors = vec![];
    for k in 0..65 {
        let delta = radius * (2. * k as f64 / 64. - 1.);
        let actual = cx.scalar(delta, &cx.assignment)?;
        direct.push([delta, actual]);
        taylor.push([delta, polynomial(coefficients, delta)]);
        errors.push([delta, (actual - polynomial(coefficients, delta)).abs()]);
    }
    r.plot(
        "Move one walker; retain all selected companion identities",
        "displacement",
        "fitness",
        vec![
            line("Production fitness, all affected rows", direct),
            line("Taylor prediction", taylor),
        ],
    );
    r.plot(
        "Taylor error inside the checked window",
        "displacement",
        "absolute error",
        vec![line("Production comparison", errors)],
    );
    let mut by_order = vec![];
    for m in 1..=order {
        let mut worst: f64 = 0.;
        for k in 0..65 {
            let delta = radius * (2. * k as f64 / 64. - 1.);
            worst = worst.max(
                (cx.scalar(delta, &cx.assignment)? - polynomial(&coefficients[..=m], delta)).abs(),
            );
        }
        by_order.push([m as f64, worst]);
    }
    r.plot(
        "How polynomial order changes the approximation",
        "Taylor order",
        "maximum sampled absolute error",
        vec![line("Same checked window", by_order)],
    );
    let h = 1e-5;
    let fd = (cx.scalar(h, &cx.assignment)? - cx.scalar(-h, &cx.assignment)?) / (2. * h);
    r.metric("Requested radius", requested, "")
        .metric("Checked radius", radius, "")
        .metric(
            "Maximum scaled error, staggered checks",
            check(radius, true)?,
            "",
        )
        .metric("Error tolerance", tolerance, "")
        .metric("Analytic first derivative", coefficients[1], "")
        .metric("Production finite-difference derivative", fd, "")
        .metric("Derivative residual", coefficients[1] - fd, "");
    // Each coefficient is checked by an independently assembled real finite-difference stencil.
    // Report conditioning explicitly: high orders can be unresolved in f64.
    let mut checks = vec![];
    for (k, &coefficient) in coefficients.iter().enumerate().take(order + 1).skip(1) {
        let span = order.max(3);
        let base = (radius / span as f64).max(1e-4);
        let mut best = (0., f64::INFINITY, base);
        for scale in [0.5, 1., 2., 4.] {
            let step = base * scale;
            let coarse = coefficient_reference(&cx, &cx.assignment, k, span, step)?;
            let fine = coefficient_reference(&cx, &cx.assignment, k, span, step / 2.)?;
            let uncertainty = (fine.0 - coarse.0).abs() + fine.1 + coarse.1;
            if uncertainty < best.1 {
                best = (fine.0, uncertainty, step / 2.);
            }
        }
        let (reference, uncertainty, step) = best;
        let resolved = uncertainty < 1e-3 * (1. + reference.abs());
        let agrees =
            (coefficient - reference).abs() <= 5. * uncertainty + 1e-6 * (1. + reference.abs());
        checks.push(json!({"order":k,"automatic":coefficient,"independent_coefficient":reference,"step":step,"step_change_and_roundoff":uncertainty,"resolved":resolved,"agrees_within_estimated_error":agrees,"residual":coefficient-reference}));
    }
    r.metric(
        "Resolved coefficient disagreements",
        checks
            .iter()
            .filter(|v| v["resolved"] == true && v["agrees_within_estimated_error"] == false)
            .count() as f64,
        "",
    );
    let unresolved = checks.iter().filter(|v| v["resolved"] == false).count();
    r.metric(
        "Unresolved higher-order coefficient checks",
        unresolved as f64,
        "",
    );
    r.plot(
        "Independent coefficient checks",
        "derivative order",
        "scaled discrepancy / estimated numerical uncertainty",
        vec![
            line(
                "Absolute coefficient discrepancy / (1 + coefficient magnitude)",
                checks
                    .iter()
                    .map(|v| {
                        [
                            v["order"].as_f64().unwrap(),
                            v["residual"].as_f64().unwrap().abs()
                                / (1. + v["automatic"].as_f64().unwrap().abs()),
                        ]
                    })
                    .collect(),
            ),
            line(
                "Step-change and roundoff estimate / (1 + coefficient magnitude)",
                checks
                    .iter()
                    .map(|v| {
                        [
                            v["order"].as_f64().unwrap(),
                            v["step_change_and_roundoff"].as_f64().unwrap()
                                / (1. + v["automatic"].as_f64().unwrap().abs()),
                        ]
                    })
                    .collect(),
            ),
        ],
    );
    let raw = cx
        .benchmark
        .physics_objective(cx.points[0].len())?
        .evaluate(&cx.coordinates(0., 2)?[cx.target])?;
    r.metric("Local reward slope", raw.coefficients[1], "")
        .metric("Local reward curvature", 2. * raw.coefficients[2], "");
    r.plot(
        "Reward landscape along the same displacement",
        "displacement",
        "raw objective",
        vec![line(
            "Production objective",
            (0..65)
                .map(|k| {
                    let delta = radius * (2. * k as f64 / 64. - 1.);
                    Ok([delta, cx.benchmark.value(&cx.positions(delta)[cx.target])?])
                })
                .collect::<Result<Vec<_>>>()?,
        )],
    );
    r.details["coefficient_checks"] = json!(checks);
    r.details["jet_coefficients"] = json!(coefficients);
    r.details["jet_multi_indices"] = json!(jet.space.indices);
    r.details["fixed_assignment"] = json!(cx.assignment);
    r.details["target"] = json!(cx.target);
    r.details["window_validation"] = json!({"kind":"production scalar comparison on 65 grid and 65 staggered points","requested_radius":requested,"accepted_radius":radius,"scaled_tolerance":tolerance,"analytic_remainder_bound":null});
    r.note("The checked window measures approximation accuracy at two interleaved sets of points. It is not the theorem's analytic radius or a bound between sampled points. Every incoming companion distance and the complete normalization respond to the displaced walker.");
    r.note("Higher-order coefficients have independent real finite-difference checks at two step sizes. The coefficient table marks checks unresolved when truncation changes or floating-point cancellation are too large.");
    if let Some(assignments) = cx.assignments() {
        ensemble(&cx, &assignments, radius, r)?;
    } else {
        r.note("Exact companion averaging requires at most 4096 assignments. Choose at most five walkers to display it; the fixed-assignment experiment remains available for larger populations.");
        r.details["companion_average"] =
            json!({"available":false,"reason":"assignment count exceeds 4096"});
    }
    Ok(())
}

fn coefficient_reference(
    cx: &Context,
    c: &[usize],
    order: usize,
    span: usize,
    h: f64,
) -> Result<(f64, f64)> {
    let n = 2 * span + 1;
    // Solve moment equations in nodes normalized to [-1,1], using pivoting.
    let nodes: Vec<_> = (0..n)
        .map(|j| (j as f64 - span as f64) / span as f64)
        .collect();
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            let mut row: Vec<_> = nodes.iter().map(|x| x.powi(k as i32)).collect();
            row.push(if k == order { 1. } else { 0. });
            row
        })
        .collect();
    for i in 0..n {
        let pivot = (i..n)
            .max_by(|&u, &v| a[u][i].abs().total_cmp(&a[v][i].abs()))
            .unwrap();
        a.swap(i, pivot);
        let d = a[i][i];
        if d.abs() < 1e-18 {
            return Err(error("coefficient stencil is singular"));
        }
        for value in &mut a[i][i..=n] {
            *value /= d;
        }
        let pivot_row = a[i].clone();
        for (k, row) in a.iter_mut().enumerate() {
            if k != i {
                let q = row[i];
                for (value, pivot) in row[i..=n].iter_mut().zip(&pivot_row[i..=n]) {
                    *value -= q * pivot;
                }
            }
        }
    }
    let radius = h * span as f64;
    let mut value = 0.;
    let mut absolute = 0.;
    for i in 0..n {
        let f = cx.scalar(radius * nodes[i], c)?;
        let term = a[i][n] * f / radius.powi(order as i32);
        value += term;
        absolute += term.abs();
    }
    Ok((value, 128. * f64::EPSILON * absolute))
}
fn ensemble(
    cx: &Context,
    assignments: &[Vec<usize>],
    radius: f64,
    r: &mut ExperimentResult,
) -> Result<()> {
    let rows = cx.probabilities(0., 1)?;
    let mut conditional = 0.;
    let mut law = 0.;
    let mut normalization = 0.;
    let mut center_weights = vec![];
    let mut values = vec![];
    for c in assignments {
        let mut p = rows[0][0].constant(1.);
        for (i, &j) in c.iter().enumerate() {
            p = p.mul(&rows[i][j]);
        }
        let f = cx.jet(0., 1, c)?;
        conditional += p.value() * f.coefficients[1];
        law += p.coefficients[1] * f.value();
        normalization += p.value();
        center_weights.push(p.value());
        values.push(f.value());
    }
    let mean: f64 = center_weights.iter().zip(&values).map(|(p, f)| p * f).sum();
    let var: f64 = center_weights
        .iter()
        .zip(&values)
        .map(|(p, f)| p * (f - mean).powi(2))
        .sum();
    let evaluate = |delta: f64| -> Result<(f64, f64)> {
        let p = cx.production_probabilities(delta)?;
        let mut updated = 0.;
        let mut frozen = 0.;
        for (k, c) in assignments.iter().enumerate() {
            let weight: f64 = c.iter().enumerate().map(|(i, &j)| p[i][j]).product();
            let f = cx.scalar(delta, c)?;
            updated += weight * f;
            frozen += center_weights[k] * f;
        }
        Ok((updated, frozen))
    };
    let h = 1e-5;
    let fd = (evaluate(h)?.0 - evaluate(-h)?.0) / (2. * h);
    let mut updated = vec![];
    let mut frozen = vec![];
    let mut tangent = vec![];
    for k in 0..33 {
        let delta = radius * (2. * k as f64 / 32. - 1.);
        let (a, b) = evaluate(delta)?;
        updated.push([delta, a]);
        frozen.push([delta, b]);
        tangent.push([delta, mean + (conditional + law) * delta]);
    }
    r.plot(
        "Average over complete companion assignments",
        "displacement",
        "mean fitness",
        vec![
            line("Exact mean with updated companion probabilities", updated),
            line("Mean with probabilities held at the center", frozen),
            line("Tangent including both derivative terms", tangent),
        ],
    );
    let mut distribution: Vec<_> = values
        .iter()
        .copied()
        .zip(center_weights.iter().copied())
        .collect();
    distribution.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut mass = 0.;
    let cdf = distribution
        .iter()
        .map(|(f, p)| {
            mass += p;
            [*f, mass]
        })
        .collect();
    r.plot(
        "Companion fluctuations at the same recorded swarm",
        "fitness",
        "cumulative probability",
        vec![line("Exact assignment distribution", cdf)],
    );
    r.plot(
        "Two contributions to the derivative of mean fitness",
        "term: 0 fixed assignments, 1 changing law, 2 total",
        "derivative",
        vec![line(
            "Exact finite-law calculation",
            vec![[0., conditional], [1., law], [2., conditional + law]],
        )],
    );
    r.metric("Enumerated assignments", assignments.len() as f64, "")
        .metric("Assignment probability sum", normalization, "")
        .metric("Companion mean fitness", mean, "")
        .metric(
            "Minimum fitness across assignments",
            values.iter().copied().fold(f64::INFINITY, f64::min),
            "",
        )
        .metric(
            "Maximum fitness across assignments",
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            "",
        )
        .metric("Companion fitness standard deviation", var.sqrt(), "")
        .metric("Averaged fixed-assignment derivative", conditional, "")
        .metric("Companion-law derivative contribution", law, "")
        .metric("Mean derivative, production finite difference", fd, "")
        .metric("Mean derivative residual", conditional + law - fd, "");
    r.details["companion_average"] = json!({"available":true,"method":"enumeration of all assignments under the configured independent current-frame donor law","count":assignments.len(),"mean":mean,"variance":var,"fixed_assignment_derivative":conditional,"law_derivative":law,"production_finite_difference":fd});
    r.note("Companion randomness is measured at one fixed swarm. Displacement changes deterministic rewards, all affected distances, normalization and companion probabilities. The mean averages the completed nonlinear fitness, not fitness evaluated at average distances.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use algorithmic_gas::RecordingConfig;
    async fn archive(seed: u64, landscape: &str) -> RunArchive<f64> {
        let p = json!({"walkers":4,"sigma":0.15,"landscape":landscape});
        let mut gas = config("IV-07", &p, seed)
            .unwrap()
            .build::<f64>()
            .await
            .unwrap();
        gas.start_recording(RecordingConfig {
            max_steps: 4,
            max_bytes: 32 * 1024 * 1024,
        })
        .unwrap();
        for _ in 0..3 {
            gas.step().await.unwrap();
        }
        gas.recording().unwrap().clone()
    }
    #[test]
    fn coupled_jet_and_assignment_law_match_production() {
        futures_lite::future::block_on(async {
            for seed in [0, 7, 516] {
                for landscape in ["quadratic", "multiwell"] {
                    let a = archive(seed, landscape).await;
                    let p = json!({"walkers":4,"sigma":0.15,"landscape":landscape});
                    let cx = Context::new(&p, &a).unwrap();
                    let assignments = cx.assignments().unwrap();
                    assert_eq!(assignments.len(), 81);
                    let rows = cx.probabilities(0., 1).unwrap();
                    let production = cx.production_probabilities(0.).unwrap();
                    for (a, b) in rows.iter().zip(&production) {
                        for (u, v) in a.iter().zip(b) {
                            assert!((u.value() - v).abs() < 1e-12);
                        }
                    }
                    // Deliberately include all assignments: incoming edges cannot be frozen.
                    for c in &assignments {
                        let j = cx.jet(0., 3, c).unwrap();
                        assert!((j.value() - cx.scalar(0., c).unwrap()).abs() < 1e-11);
                        let h = 1e-5;
                        let plus = cx.scalar(h, c).unwrap();
                        let minus = cx.scalar(-h, c).unwrap();
                        let fd = (plus - minus) / (2. * h);
                        assert!(
                            (fd - j.coefficients[1]).abs() < 2e-5 * (1. + fd.abs()),
                            "{landscape} {seed} {c:?}"
                        );
                        let second = (plus - 2. * j.value() + minus) / (2. * h * h);
                        assert!((second - j.coefficients[2]).abs() < 2e-3 * (1. + second.abs()));
                    }
                    let mut r = ExperimentResult::new(7, "test", "test");
                    ensemble(&cx, &assignments, 0.01, &mut r).unwrap();
                    let stats = &r.details["companion_average"];
                    let exact = stats["fixed_assignment_derivative"].as_f64().unwrap()
                        + stats["law_derivative"].as_f64().unwrap();
                    let fd = stats["production_finite_difference"].as_f64().unwrap();
                    assert!((exact - fd).abs() < 2e-5 * (1. + fd.abs()));
                }
            }
        });
    }
    #[test]
    fn adaptive_window_checks_unseen_displacements_and_preserves_archive() {
        futures_lite::future::block_on(async {
            let a = archive(7, "quadratic").await;
            let snapshot = serde_json::to_value(&a).unwrap();
            let p = json!({"walkers":4,"sigma":0.15,"radius":0.2,"order":6,"tolerance":1e-5});
            let cx = Context::new(&p, &a).unwrap();
            let mut r = ExperimentResult::new(7, "test", "test");
            analyze(&p, &a, &mut r).unwrap();
            let radius = r.details["window_validation"]["accepted_radius"]
                .as_f64()
                .unwrap();
            let jet = cx.jet(0., 6, &cx.assignment).unwrap();
            for i in 0..257 {
                let delta = radius * (2. * (i as f64 + 0.314159) / 257. - 1.);
                let actual = cx.scalar(delta, &cx.assignment).unwrap();
                assert!(
                    (actual - polynomial(&jet.coefficients, delta)).abs()
                        <= 1e-5 * (1. + actual.abs())
                );
            }
            assert_eq!(r.details["coefficient_checks"].as_array().unwrap().len(), 6);
            assert_eq!(serde_json::to_value(&a).unwrap(), snapshot);
        });
    }
}
