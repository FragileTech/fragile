//! New native Chapter 4 experiments with complete, reusable operator archives.
//! Fixed-state trials restore physical empirical inputs between addressed draws.
//! Their statistics concern the recorded proposal, never QSD transition laws.
use crate::{
    Benchmark, BenchmarkModel,
    convergence_cloning::analyze_cloning_step,
    convergence_decay::canonical_decay_population,
    convergence_experiments::{ArchiveStore, ExperimentConfig},
    convergence_lyapunov::uniform_transport,
};
use algorithmic_gas::{
    AlgorithmicGas, GasBuilder, GasConfig, GasError, ObservationBatch, Population, Result,
    RunArchive, TensorBatch,
    fitness::PositiveMap,
    kinetic::{KineticKind, ViscousForceConfig},
    tracking::{RecordingConfig, StageSnapshot},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const SOURCE: &str = include_str!(
    "../../../../docs/source/2_fractal_gas/convergence_program/04_wasserstein_contraction.md"
);
fn error(s: impl Into<String>) -> GasError {
    GasError::Configuration(s.into())
}
fn sq(x: &[f64]) -> f64 {
    x.iter().map(|a| a * a).sum()
}
fn dist(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum()
}
fn mean(x: &[Vec<f64>]) -> Vec<f64> {
    (0..x[0].len())
        .map(|a| x.iter().map(|p| p[a]).sum::<f64>() / x.len() as f64)
        .collect()
}
fn centered(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = mean(x);
    x.iter()
        .map(|p| p.iter().zip(&m).map(|(a, b)| a - b).collect())
        .collect()
}
fn variance(x: &[Vec<f64>]) -> f64 {
    let m = mean(x);
    x.iter().map(|p| dist(p, &m)).sum::<f64>() / x.len() as f64
}
fn phase(x: &[Vec<f64>], v: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .zip(v)
        .map(|(x, v)| [x.clone(), v.clone()].concat())
        .collect()
}
fn qcost(a: &[f64], z: &[f64]) -> f64 {
    let d = a.len() / 2;
    let dx: Vec<_> = a[..d].iter().zip(&z[..d]).map(|(x, y)| x - y).collect();
    let dv: Vec<_> = a[d..].iter().zip(&z[d..]).map(|(x, y)| x - y).collect();
    sq(&dx) + sq(&dv) + 0.5 * dx.iter().zip(&dv).map(|(x, y)| x * y).sum::<f64>()
}
fn stage_points(stage: &StageSnapshot, name: &str) -> Result<Vec<Vec<f64>>> {
    let field = stage
        .fields
        .get(name)
        .ok_or_else(|| error(format!("Missing recorded {name}")))?;
    let d = field.item_shape.iter().product::<usize>();
    Ok(field.values.chunks(d).map(<[f64]>::to_vec).collect())
}
fn population(x: &[Vec<f64>], v: &[Vec<f64>], alive: &[bool]) -> Result<Population<f64>> {
    let mut obs =
        ObservationBatch::positions(TensorBatch::vectors(x.len(), x[0].len(), x.concat())?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(v.len(), v[0].len(), v.concat())?,
    );
    let mut p = Population::new(obs)?;
    for (status, &a) in p.validity.iter_mut().zip(alive) {
        status.terminated = !a;
    }
    Ok(p)
}
async fn engine(
    mut c: GasConfig,
    b: Benchmark,
    p: Population<f64>,
    seed: u64,
) -> Result<AlgorithmicGas<f64>> {
    c.seed = seed;
    c.max_memory_bytes = 1024 * 1024 * 1024;
    let model = BenchmarkModel {
        benchmark: b,
        field: "positions".into(),
        direction: c.fitness.direction,
    };
    GasBuilder::new(p, model.clone())
        .gradient(model)
        .config(c)
        .build()
        .await
}
fn recording(chunk: usize) -> RecordingConfig {
    RecordingConfig {
        max_steps: chunk,
        max_bytes: 192 * 1024 * 1024,
        graph: true,
    }
}
fn source_formula(label: &str, marker: &str) -> String {
    if marker.is_empty() {
        return String::new();
    }
    let key = format!(":label: {label}");
    let tail = SOURCE
        .split_once(&key)
        .map(|(_, tail)| tail)
        .unwrap_or(SOURCE);
    let owned = tail.split_once("\n:label:").map_or(tail, |(head, _)| head);
    owned
        .split("$$")
        .enumerate()
        .find_map(|(i, s)| (i % 2 == 1 && s.contains(marker)).then(|| s.trim().to_owned()))
        .unwrap_or_default()
}
fn comparison(
    id: &str,
    label: &str,
    marker: &str,
    values: &[(f64, f64)],
    relation: &str,
    scope: &str,
) -> Value {
    let n = values.len() as f64;
    let mean = |f: fn(&(f64, f64)) -> f64| values.iter().map(f).sum::<f64>() / n;
    let residuals: Vec<_> = values.iter().map(|(a, b)| a - b).collect();
    let r = residuals.iter().sum::<f64>() / n;
    let se = if n > 1. {
        (residuals.iter().map(|x| (x - r).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt()
    } else {
        0.
    };
    let tol = 2e-10
        * (1.
            + values
                .iter()
                .map(|(a, b)| a.abs().max(b.abs()))
                .fold(0., f64::max));
    let mut passed = if relation == "equal" {
        r.abs() <= 6. * se + tol
    } else {
        r <= 6. * se + tol
    };
    let reference = id.starts_with("reference-");
    let exact = reference
        || id.contains(":target_")
        || id.contains("barycenter_split")
        || id.contains("centered_positional_variance")
        || id.contains("permutation_exact")
        || id.contains("mandatory_revival")
        || id.contains("actual_dense")
        || id.contains("actual_harmonic")
        || id.contains("momentum_conservation");
    let sampling_hypothesis = if reference {
        "Continuous native reference trajectory; every recorded primitive or stage identity is checked separately without an independence assumption"
    } else {
        "Independent addressed innovations after fixed-state restore; exception: stated common-source permutation probe"
    };
    if exact {
        passed = values.iter().all(|(a, b)| {
            if relation == "equal" {
                (a - b).abs() <= tol
            } else {
                a - b <= tol
            }
        });
    }
    let mut v = json!({"id":id,"source_labels":[label],"observed":mean(|x|x.0),"bound":mean(|x|x.1),
        "relation":relation,"samples":values.len(),"residual":r,"standard_error":se,"passed":passed,
        "maximum_absolute_residual":residuals.iter().map(|x|x.abs()).fold(0.,f64::max),
        "six_standard_error_interval":[r-6.*se,r+6.*se],"scope":scope,
        "kind":if exact {"every_retained_sample_exact"} else {"empirical_conditional_expectation"},
        "status":if passed {"not_rejected"} else {"violated"},
        "hypotheses":["Actual native immutable-source copying and sampled fitness retained",
        "Physical empirical probability measures normalized by their own alive counts",
        sampling_hypothesis]});
    if exact {
        let witness = values
            .iter()
            .max_by(|(a, b), (c, d)| {
                let x = if relation == "equal" {
                    (a - b).abs()
                } else {
                    a - b
                };
                let y = if relation == "equal" {
                    (c - d).abs()
                } else {
                    c - d
                };
                x.total_cmp(&y)
            })
            .unwrap();
        v["observed"] = json!(witness.0);
        v["bound"] = json!(witness.1);
        v["standard_error"] = json!(0.);
        v["residual"] = json!(witness.0 - witness.1);
        v["ensemble_observed_mean"] = json!(mean(|x| x.0));
        v["ensemble_bound_mean"] = json!(mean(|x| x.1));
    }
    let quote = source_formula(label, marker);
    if !quote.is_empty() {
        v["source_formula"] = json!(quote);
    }
    v
}
fn push(m: &mut BTreeMap<String, Vec<(f64, f64)>>, key: &str, x: f64, y: f64) {
    m.entry(key.into()).or_default().push((x, y));
}

fn logistic_bounds(m: &PositiveMap, power: f64) -> Result<(f64, f64)> {
    match m {
        PositiveMap::Logistic { amplitude, floor } if *floor > 0. && power >= 0. => {
            Ok((floor.powf(power), (floor + amplitude).powf(power)))
        }
        _ => Err(error(
            "Chapter 4 empirical fitness certificates require positive bounded logistic maps",
        )),
    }
}

/// Event-specific target certificate. H is chosen from geometry before fitness.
/// A nonpositive realized gap excludes only the optional target theorem; its
/// complete measurement outcome remains in the averaged pressure experiment.
#[allow(clippy::too_many_arguments)]
fn target_certificate(
    c: &GasConfig,
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    fitness: &[f64],
    p: &[f64],
    left_p: &[f64],
    structural: f64,
    paired_error: &[f64],
) -> Result<Value> {
    let n = x.len();
    let nf = n as f64;
    let cx = centered(x);
    let cy = centered(y);
    let total = variance(x);
    let h: Vec<_> = cx.iter().map(|p| sq(p) > total).collect();
    let hc = h.iter().filter(|&&h| h).count();
    let lc = n - hc;
    if hc == 0 || lc == 0 {
        return Ok(
            json!({"applicable":false,"reason":"Geometric high/complement sets must both be nonempty","H":h}),
        );
    }
    let high = fitness
        .iter()
        .zip(&h)
        .filter_map(|(&f, &h)| h.then_some(f))
        .sum::<f64>()
        / hc as f64;
    let low = fitness
        .iter()
        .zip(&h)
        .filter_map(|(&f, &h)| (!h).then_some(f))
        .sum::<f64>()
        / lc as f64;
    let gap = low - high;
    let global = fitness.iter().sum::<f64>() / nf;
    let actual_range = fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - fitness.iter().copied().fold(f64::INFINITY, f64::min);
    let u: Vec<_> = fitness.iter().map(|&f| f <= global).collect();
    let t: Vec<_> = h.iter().zip(&u).map(|(&h, &u)| h && u).collect();
    if gap <= 0. {
        return Ok(
            json!({"applicable":false,"reason":"Retained arithmetic fitness gap is nonpositive","H":h,"U":u,"T":t,"high_mean":high,"low_mean":low,"gap":gap}),
        );
    }
    let (rmin, rmax) = logistic_bounds(&c.fitness.reward_map, c.fitness.reward_exponent)?;
    let (dmin, dmax) = logistic_bounds(&c.fitness.diversity_map, c.fitness.diversity_exponent)?;
    let vmin = rmin * dmin;
    let vmax = rmax * dmax;
    let range = vmax - vmin;
    let delta = gap / 2.;
    let fh = hc as f64 / nf;
    let fl = lc as f64 / nf;
    let sstar = fh * fl * delta * delta;
    let svar = fitness.iter().map(|f| (f - global).powi(2)).sum::<f64>() / nf;
    let a_star = (-4_f64).exp();
    let bacc = range.max(c.clone_decision.saturation * (vmax + c.clone_decision.epsilon));
    let pu = a_star * sstar / (2. * range * bacc);
    let overlap = fh * fl * delta / range;
    let fuf = sstar / (2. * range * range);
    let ch = cx
        .iter()
        .zip(&h)
        .filter_map(|(x, &h)| h.then_some(sq(x)))
        .sum::<f64>()
        / (nf * total);
    // These generated comparisons have equal intrinsic velocity vectors in a
    // fixed admissible plan and y either collapsed or the permuted same cloud.
    // Use the recorded structural value and the explicit nonnegative remainder.
    let ax = 0.5;
    let bx = 0.;
    if ax * structural > total + 2e-10 * (1. + total) {
        return Err(error(
            "Generated target family failed its analytic positional/structural hypothesis",
        ));
    }
    let mj = cy
        .iter()
        .zip(&h)
        .filter_map(|(y, &h)| h.then_some(sq(y)))
        .sum::<f64>()
        / nf;
    let bt = paired_error
        .iter()
        .zip(h.iter().zip(&t))
        .filter_map(|(&e, (&h, &t))| (h && !t).then_some(e))
        .sum::<f64>()
        / nf;
    let cerr = ch * ax / 2.;
    let gerr = ch * bx / 2. + mj + bt;
    let chi = pu * cerr;
    let spread2 = 0.01;
    let gmax = (pu * gerr).max(chi * spread2);
    let target_error = paired_error
        .iter()
        .zip(&t)
        .filter_map(|(&e, &t)| t.then_some(e))
        .sum::<f64>()
        / nf;
    let pressure = paired_error
        .iter()
        .zip(p)
        .map(|(&e, &p)| e * p)
        .sum::<f64>()
        / nf;
    let minp = left_p
        .iter()
        .zip(&t)
        .filter_map(|(&p, &t)| t.then_some(p))
        .fold(1., f64::min);
    let countu = u.iter().filter(|&&u| u).count();
    let countt = t.iter().filter(|&&t| t).count();
    Ok(
        json!({"applicable":true,"H":h,"U":u,"T":t,"high_mean":high,"low_mean":low,"gap":gap,
        "fitness_bounds":[vmin,vmax],"R_star":range,"Delta_V":delta,"f_H":fh,"f_L":fl,
        "f_UH":overlap,"f_U_F":fuf,"s_star_squared":sstar,"fitness_variance":svar,
        "a_star":a_star,"B_acc":bacc,"p_u":pu,"c_H":ch,"a_x":ax,"b_x":bx,"M_j":mj,"B_T":bt,
        "c_err":cerr,"g_err":gerr,"chi":chi,"g_max":gmax,"structural_error":structural,
        "target_error":target_error,"target_error_lower":cerr*structural-gerr,"Q":pressure,"Q_lower":chi*structural-gmax,
        "minimum_target_pressure":minp,"target_fraction":countt as f64/nf,"definition_overlap_lower":fh*fl*gap/actual_range,"R_V":actual_range,"unfit_fraction":countu as f64/nf,
        "fit_fraction":1.-countu as f64/nf,"positive_Q_lower":chi*structural-gmax>0.,
        "scope":"Conditional certificate for this actual sampled fitness vector, including excluded-target error. These event-dependent constants are not averaged into a state-uniform realized-target rate."}),
    )
}

fn clouds(n: usize, d: usize, profile: &str) -> Result<(Population<f64>, Population<f64>)> {
    let alive: Vec<_> = (0..n)
        .map(|i| match profile {
            "revival" => i < n / 2,
            "singleton" => i == 0,
            _ => true,
        })
        .collect();
    let x: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..d)
                .map(|a| {
                    if !alive[i] {
                        if i % 2 == 0 { 1e6 } else { -1e6 }
                    } else if a == 0 {
                        if i < n / 4 { -1.5 } else { 0.5 }
                    } else {
                        0.
                    }
                })
                .collect()
        })
        .collect();
    let y: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..d)
                .map(|_| {
                    if alive[i] {
                        0.
                    } else {
                        if i % 2 == 0 { -1e6 } else { 1e6 }
                    }
                })
                .collect()
        })
        .collect();
    let v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..d)
                .map(|a| {
                    0.2 * ((i + 1) as f64 * std::f64::consts::SQRT_2 + a as f64).sin()
                        / (d as f64).sqrt()
                })
                .collect()
        })
        .collect();
    let left = population(&x, &v, &alive)?;
    if profile == "permutation" {
        let px: Vec<_> = x.iter().rev().cloned().collect();
        let pv: Vec<_> = v.iter().rev().cloned().collect();
        let pm: Vec<_> = alive.iter().rev().copied().collect();
        return Ok((
            canonical_decay_population(&left)?,
            canonical_decay_population(&population(&px, &pv, &pm)?)?,
        ));
    }
    Ok((left, population(&y, &v, &alive)?))
}

#[allow(clippy::too_many_arguments)]
async fn proposal_case(
    config: &ExperimentConfig,
    store: &mut ArchiveStore,
    id: usize,
    n: usize,
    d: usize,
    benchmark: Benchmark,
    profile: &str,
) -> Result<Value> {
    let mut c = GasConfig::euclidean(d, 0.04)?;
    if profile == "selection" {
        c.fitness.reward_exponent = 2.;
        c.fitness.diversity_exponent = 0.5;
        c.clone_decision.saturation = 0.5;
        c.clone_transform.jitter_amplitude = 0.2;
    }
    let (left, right) = clouds(n, d, profile)?;
    let entering_alive = left.eligible(false);
    let all_alive = entering_alive.iter().all(|x| *x);
    let extract = |p: &Population<f64>, name: &str| -> Result<Vec<Vec<f64>>> {
        let f = p.observations.field(name)?;
        Ok(f.values().chunks(d).map(<[f64]>::to_vec).collect())
    };
    let ix = extract(&left, "positions")?;
    let iv = extract(&left, "velocities")?;
    let iy = extract(&right, "positions")?;
    let iw = extract(&right, "velocities")?;
    let live_x: Vec<_> = ix
        .iter()
        .zip(&entering_alive)
        .filter_map(|(x, &a)| a.then_some(x.clone()))
        .collect();
    let live_y: Vec<_> = iy
        .iter()
        .zip(&entering_alive)
        .filter_map(|(x, &a)| a.then_some(x.clone()))
        .collect();
    let dx2 = live_x
        .iter()
        .flat_map(|x| live_x.iter().map(move |y| dist(x, y)))
        .fold(0., f64::max)
        .max(
            live_y
                .iter()
                .flat_map(|x| live_y.iter().map(move |y| dist(x, y)))
                .fold(0., f64::max),
        );
    let reset =
        dx2 + 2. * (1. - 1. / n as f64) * d as f64 * c.clone_transform.jitter_amplitude.powi(2);
    let s1 = config.seed.wrapping_add(4_000_000 + id as u64 * 10_000);
    let s2 = if profile == "permutation" {
        s1
    } else {
        s1.wrapping_add(2_000_000_000)
    };
    let mut a = engine(c.clone(), benchmark, left.clone(), s1).await?;
    let mut b = engine(c.clone(), benchmark, right.clone(), s2).await?;
    let chunk = config.archive_chunk_steps.min(64);
    let mut paths = vec![];
    let mut frame_paths = vec![];
    let mut measurements: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut target_applicable = 0;
    let mut positive_target_lower = 0;
    for start in (0..config.samples).step_by(chunk) {
        let end = (start + chunk).min(config.samples);
        a.start_recording(recording(chunk))?;
        b.start_recording(recording(chunk))?;
        let mut frames = vec![];
        for repetition in start..end {
            a.replace_population(left.clone()).await?;
            b.replace_population(right.clone()).await?;
            a.step().await?;
            b.step().await?;
            let index = repetition - start;
            let aa = a.recording().unwrap();
            let bb = b.recording().unwrap();
            let sa = &aa.steps[index];
            let sb = &bb.steps[index];
            let ma = analyze_cloning_step(aa, index)?
                .moments
                .ok_or_else(|| error("Missing exact native cloning moments"))?;
            let mb = analyze_cloning_step(bb, index)?
                .moments
                .ok_or_else(|| error("Missing exact native right cloning moments"))?;
            let oa = sa
                .stages
                .iter()
                .find(|s| s.stage == "post_transform")
                .ok_or_else(|| error("Missing proposal stage"))?;
            let ob = sb
                .stages
                .iter()
                .find(|s| s.stage == "post_transform")
                .ok_or_else(|| error("Missing right proposal stage"))?;
            let x = stage_points(oa, "positions")?;
            let v = stage_points(oa, "velocities")?;
            let y = stage_points(ob, "positions")?;
            let w = stage_points(ob, "velocities")?;
            let za = phase(&x, &v);
            let zb = phase(&y, &w);
            let (full, full_plan) = uniform_transport(&za, &zb, qcost)?;
            let (structural, shape_plan) =
                uniform_transport(&centered(&za), &centered(&zb), qcost)?;
            let location = qcost(&mean(&za), &mean(&zb));
            let (xshape, x_plan) = uniform_transport(&centered(&x), &centered(&y), dist)?;
            let proxy = variance(&x) + variance(&y);
            push(
                &mut measurements,
                "proposal_q_barycenter_split",
                full,
                location + structural,
            );
            push(
                &mut measurements,
                "proposal_centered_positional_variance",
                xshape,
                proxy,
            );
            push(&mut measurements, "proposal_proxy_reset", proxy, reset);
            push(&mut measurements, "proposal_centered_reset", xshape, reset);
            push(
                &mut measurements,
                "conditional_proxy_moment",
                proxy,
                ma.expected_position_variance + mb.expected_position_variance,
            );
            if profile == "permutation" {
                push(
                    &mut measurements,
                    "permutation_exact_empirical_transport",
                    full,
                    0.,
                );
            }
            let post_alive = oa
                .validity
                .iter()
                .filter(|s| !s.terminated && !s.truncated)
                .count();
            push(
                &mut measurements,
                "mandatory_revival_mass",
                post_alive as f64,
                n as f64,
            );
            let mut target = Value::Null;
            if all_alive {
                let ia = phase(&ix, &iv);
                let ib = phase(&iy, &iw);
                let input_struct = uniform_transport(&centered(&ia), &centered(&ib), qcost)?.0;
                let cx = centered(&ix);
                let cy = centered(&iy);
                let errs: Vec<_> = cx.iter().zip(&cy).map(|(x, y)| dist(x, y)).collect();
                let pressure: Vec<_> = ma
                    .acceptance_probabilities
                    .iter()
                    .zip(&mb.acceptance_probabilities)
                    .map(|(a, b)| a + b)
                    .collect();
                target = target_certificate(
                    &c,
                    &ix,
                    &iy,
                    &sa.report.pre_clone_fitness.fitness,
                    &pressure,
                    &ma.acceptance_probabilities,
                    input_struct,
                    &errs,
                )?;
                if target["applicable"] == true {
                    target_applicable += 1;
                    if target["positive_Q_lower"] == true {
                        positive_target_lower += 1;
                    }
                    let get = |name: &str| target[name].as_f64().unwrap();
                    push(
                        &mut measurements,
                        "target_overlap",
                        -get("target_fraction"),
                        -get("definition_overlap_lower"),
                    );
                    push(
                        &mut measurements,
                        "target_fitness_variance",
                        -get("fitness_variance"),
                        -get("s_star_squared"),
                    );
                    push(
                        &mut measurements,
                        "target_unfit_fraction",
                        -get("unfit_fraction"),
                        -get("f_U_F"),
                    );
                    push(
                        &mut measurements,
                        "target_fit_fraction",
                        -get("fit_fraction"),
                        -get("f_U_F"),
                    );
                    push(
                        &mut measurements,
                        "target_row_pressure",
                        -get("minimum_target_pressure"),
                        -get("p_u"),
                    );
                    push(
                        &mut measurements,
                        "target_error_capture",
                        -get("target_error"),
                        -get("target_error_lower"),
                    );
                    push(
                        &mut measurements,
                        "target_keystone",
                        -get("Q"),
                        -get("Q_lower"),
                    );
                }
            }
            frames.push(json!({"repetition":repetition,"native_steps":[sa.report.step,sb.report.step],
                "before":{"positions":[ix,iy],"velocities":[iv,iw],"alive":[sa.report.pre_clone_eligible,sb.report.pre_clone_eligible]},
                "proposal":{"positions":[x,y],"velocities":[v,w],"validity":[oa.validity,ob.validity]},
                "sampled_fitness":[sa.report.pre_clone_fitness,sb.report.pre_clone_fitness],
                "distance_companions":[sa.report.distance_companions,sb.report.distance_companions],
                "distance_sources":[sa.report.distance_sources,sb.report.distance_sources],
                "cloning_companions":[sa.report.cloning_companions,sb.report.cloning_companions],
                "clone_plans":[sa.report.clone_plan,sb.report.clone_plan],
                "conditional_moments":[ma,mb],"full_q_transport":full,"centered_q_transport":structural,"q_location":location,
                "full_q_plan":full_plan,"centered_q_plan":shape_plan,"centered_positional_transport":xshape,"centered_positional_plan":x_plan,
                "positional_proxy":proxy,"C_reset":reset,"realized_target_certificate":target,
                "component_graph_location":"Native field_evaluations collision_component_id/size/rotation/input/output fields and accepted clone_plan edges in corresponding full CBOR archives"}));
        }
        let ap = store.save_archive(
            &format!("chapter04-case{id}-left-{start}"),
            a.recording().unwrap(),
        )?;
        let bp = store.save_archive(
            &format!("chapter04-case{id}-right-{start}"),
            b.recording().unwrap(),
        )?;
        store.save_checkpoint(&format!("chapter04-case{id}-left-{end}"), &a.checkpoint())?;
        store.save_checkpoint(&format!("chapter04-case{id}-right-{end}"), &b.checkpoint())?;
        let fp = store.save_json(
            &format!("chapter04-case{id}-frames-{start}"),
            &json!({"archive_left":ap,"archive_right":bp,"frames":frames,"first_repetition":start}),
        )?;
        paths.push(json!({"left":ap,"right":bp,"repetitions":[start,end]}));
        frame_paths.push(fp);
    }
    let specs = [
        (
            "proposal_q_barycenter_split",
            "thm-full-w2-split",
            "\\|\\bar{z}_1",
            "equal",
        ),
        (
            "proposal_centered_positional_variance",
            "lem-centered-w2-variance-bound",
            "V_{\\text{x,struct}} =",
            "upper",
        ),
        (
            "proposal_proxy_reset",
            "thm-positional-variance-proxy",
            "\\mathbb E V_{\\text{x,proxy}}'",
            "upper",
        ),
        (
            "proposal_centered_reset",
            "prop-centered-w2-control",
            "\\mathbb E V_{\\text{x,struct}}",
            "upper",
        ),
        (
            "conditional_proxy_moment",
            "thm-positional-variance-proxy",
            "",
            "equal",
        ),
        (
            "permutation_exact_empirical_transport",
            "prop-w2-prescribed-coupling-scope",
            "",
            "equal",
        ),
        (
            "mandatory_revival_mass",
            "rem-all-alive-normalization",
            "",
            "equal",
        ),
        (
            "target_overlap",
            "def-target-complement",
            "f_I\\geq",
            "upper",
        ),
        (
            "target_fitness_variance",
            "prop-w2-realized-target-keystone",
            "",
            "upper",
        ),
        (
            "target_unfit_fraction",
            "prop-w2-realized-target-keystone",
            "",
            "upper",
        ),
        (
            "target_fit_fraction",
            "prop-w2-realized-target-keystone",
            "",
            "upper",
        ),
        (
            "target_row_pressure",
            "prop-w2-realized-target-keystone",
            "",
            "upper",
        ),
        (
            "target_error_capture",
            "prop-w2-realized-target-keystone",
            "",
            "upper",
        ),
        (
            "target_keystone",
            "prop-w2-realized-target-keystone",
            "Q\\geq\\chi",
            "upper",
        ),
    ];
    let mut comparisons = vec![];
    for (key, label, marker, relation) in specs {
        if let Some(vals) = measurements.get(key) {
            let mut check = comparison(
                &format!("case{id}:{key}"),
                label,
                marker,
                vals,
                relation,
                "Native post-transform proposal before kinetics. Target comparisons are conditional on recorded positive arithmetic fitness gap; all excluded measurement outcomes remain saved. Direct optimal normalized physical transport has no permanent particle labels.",
            );
            if !all_alive
                && matches!(
                    key,
                    "proposal_proxy_reset" | "proposal_centered_reset" | "conditional_proxy_moment"
                )
            {
                check["source_labels"] = json!(["thm-positional-variance-contraction"]);
                check.as_object_mut().unwrap().remove("source_formula");
            }
            comparisons.push(check);
        }
    }
    Ok(
        json!({"case":id,"benchmark":benchmark,"profile":profile,"N":n,"d":d,"config":c,
        "seeds":[s1,s2],"origin_law":"Declared fixed empirical states; replace_population anchors before each trial; fresh native step-addressed primitive draws. Permutation case uses the same intrinsic canonical representative and identical seed.",
        "initial_positions":[ix,iy],"initial_velocities":[iv,iw],"entering_alive":entering_alive,
        "C_reset":reset,"comparisons":comparisons,"raw_archives":paths,"operator_frame_archives":frame_paths,
        "target_applicable_measurements":target_applicable,"target_excluded_measurements":config.samples-target_applicable,
        "target_positive_lower_bound_measurements":positive_target_lower,"samples":config.samples}),
    )
}

fn primitive_reference(c: &GasConfig) -> Result<Value> {
    let KineticKind::Baoab {
        dt: h,
        friction: gamma,
        ..
    } = c.kinetic.integrator
    else {
        return Err(error("Reference needs native BAOAB"));
    };
    let visc = c
        .qft
        .viscosity
        .as_ref()
        .ok_or_else(|| error("Reference must retain dense viscosity"))?;
    let t = h / 2.;
    let decay = (-gamma * h).exp();
    let q2 = if gamma > 0. {
        -(-2. * gamma * h).exp_m1() / (2. * gamma)
    } else {
        h
    };
    let s = c.kinetic.position_diffusion * h.sqrt();
    let sj = c.clone_transform.jitter_amplitude;
    let vmax = c.kinetic.velocity_cap.unwrap();
    let vc = (1. + 2. * c.clone_transform.restitution.unwrap().abs()) * vmax;
    let rd = 2. * 3_f64.sqrt();
    let b1 = (1. + 2. * t * visc.coefficient) * vc + t * (rd + sj);
    let a = 2. + sj + t * (1. + decay) * b1;
    let sigma = (t * t * q2 + s * s).sqrt();
    // A rigorous Gaussian density interval lower bound, not CDF subtraction
    // that would round the tiny survival floor to zero.
    let z = (a - 2.) / sigma + 1. / 13.;
    let log_p0 = (4. * std::f64::consts::PI / 3.).ln() - 1.5 * (std::f64::consts::TAU).ln() - 0.5;
    let log_af_lower =
        log_p0 + 3. * (-0.5 * z * z - 0.5 * std::f64::consts::TAU.ln() - 13_f64.ln());
    let rstar = (6. * ((12_f64 * 3. * 200.).ln() - 2. * log_af_lower)).sqrt();
    let j = sj * rstar;
    let cx = if visc.row_normalized {
        16. * visc.coefficient * vc * (rd + j) / visc.bandwidth.powi(2)
    } else {
        4. * visc.coefficient * vc * (-0.5_f64).exp() / visc.bandwidth
    };
    Ok(
        json!({"h":h,"t":t,"gamma":gamma,"c":decay,"q_squared":q2,"s":s,"sigma_J":sj,"V":vmax,"V_c":vc,
        "nu":visc.coefficient,"rho":visc.bandwidth,"row_normalized":visc.row_normalized,
        "L_F":1.,"B_F":0.,"R_D":rd,"B_1":b1,"A":a,"sigma_h":sigma,
        "log_survival_floor_lower":log_af_lower,"r_star_from_survival_lower":rstar,"J":j,"C_x":cx,
        "kappa_F":1.-t*t-if visc.row_normalized {2.*t*visc.coefficient} else {t*visc.coefficient},
        "beta_F":t*t*(1.+cx),"scope":"Actual harmonic dense-viscous kernel. Analytic force and rigorous Gaussian interval/ball lower bounds provide primitive positivity certificates. No QSD spectral rate is inferred from this trajectory."}),
    )
}

fn dense_force_checks(archive: &RunArchive<f64>, index: usize) -> Result<Vec<Value>> {
    let step = &archive.steps[index];
    let cfg = archive.gas_config.qft.viscosity.as_ref().unwrap();
    let mut checks = vec![];
    for name in ["B1", "B2"] {
        let stage = step
            .stages
            .iter()
            .find(|s| s.stage == format!("{name}_input"))
            .ok_or_else(|| error("Missing dense force input stage"))?;
        let x = stage_points(stage, "positions")?;
        let v = stage_points(stage, "velocities")?;
        let n = x.len();
        let d = x[0].len();
        let eligible: Vec<_> = stage
            .validity
            .iter()
            .map(|s| !s.terminated && !s.truncated)
            .collect();
        let k = eligible.iter().filter(|&&a| a).count();
        let actual = step
            .field_evaluations
            .iter()
            .find(|f| f.stage == name && f.field == "viscous_force")
            .ok_or_else(|| error("Missing actual dense force field"))?;
        let potential = step
            .field_evaluations
            .iter()
            .find(|f| f.stage == name && f.field == "potential_force")
            .ok_or_else(|| error("Missing actual potential force"))?;
        let mut max_residual = 0_f64;
        let mut potential_residual = 0_f64;
        let mut total_force = vec![0.; d];
        for i in 0..n {
            if !eligible[i] {
                continue;
            }
            let weights: Vec<_> = (0..n)
                .map(|j| {
                    if i == j || !eligible[j] {
                        0.
                    } else {
                        (-dist(&x[i], &x[j]) / (2. * cfg.bandwidth.powi(2))).exp()
                    }
                })
                .collect();
            let denominator = if cfg.row_normalized {
                weights.iter().sum::<f64>()
            } else {
                k as f64
            };
            for a in 0..d {
                let expected = if denominator > 0. {
                    cfg.coefficient
                        * weights
                            .iter()
                            .enumerate()
                            .map(|(j, w)| w * (v[j][a] - v[i][a]))
                            .sum::<f64>()
                        / denominator
                } else {
                    0.
                };
                max_residual = max_residual.max((actual.values[i * d + a] - expected).abs());
                potential_residual =
                    potential_residual.max((potential.values[i * d + a] + x[i][a]).abs());
                total_force[a] += actual.values[i * d + a];
            }
        }
        checks.push(json!({"stage":name,"viscous_formula_residual":max_residual,"harmonic_force_residual":potential_residual,
            "summed_viscous_force_norm":sq(&total_force).sqrt(),"influence_edges":step.influences.iter().filter(|r|r.stage==name&&r.field=="viscous_force").count(),
            "expected_dense_influence_edges":k*(k.saturating_sub(1)),"normalization":if cfg.row_normalized {"off-diagonal row mass"} else {"eligible count"}}));
    }
    Ok(checks)
}

async fn reference_case(
    config: &ExperimentConfig,
    store: &mut ArchiveStore,
    row: bool,
) -> Result<Value> {
    let c = GasConfig::viscous_euclidean(
        3,
        0.04,
        ViscousForceConfig {
            coefficient: 0.3,
            bandwidth: 1.,
            row_normalized: row,
        },
    )?;
    let x: Vec<Vec<f64>> = (0..200)
        .map(|i| {
            (0..3)
                .map(|a| ((i + 1) as f64 * std::f64::consts::SQRT_2 + a as f64).sin())
                .collect()
        })
        .collect();
    let v: Vec<Vec<f64>> = (0..200)
        .map(|i| {
            (0..3)
                .map(|a| 0.2 * ((i + 1) as f64 * 1.73 + a as f64).cos() / 3_f64.sqrt())
                .collect()
        })
        .collect();
    let seed = config.seed.wrapping_add(40_000_000 + u64::from(row));
    let mut gas = engine(
        c.clone(),
        Benchmark::Quadratic,
        population(&x, &v, &[true; 200])?,
        seed,
    )
    .await?;
    let chunk = config.archive_chunk_steps.min(8);
    let mut raw = vec![];
    let mut force = vec![];
    let mut numerical = BTreeMap::new();
    for start in (0..config.steps).step_by(chunk) {
        let end = (start + chunk).min(config.steps);
        gas.start_recording(recording(chunk))?;
        for step in start..end {
            gas.step().await?;
            let archive = gas.recording().unwrap();
            let index = step - start;
            let checks = dense_force_checks(archive, index)?;
            for ck in &checks {
                push(
                    &mut numerical,
                    "actual_dense_force",
                    ck["viscous_formula_residual"].as_f64().unwrap(),
                    0.,
                );
                push(
                    &mut numerical,
                    "actual_harmonic_force",
                    ck["harmonic_force_residual"].as_f64().unwrap(),
                    0.,
                );
                push(
                    &mut numerical,
                    "actual_dense_edges",
                    ck["influence_edges"].as_f64().unwrap(),
                    ck["expected_dense_influence_edges"].as_f64().unwrap(),
                );
                if !row {
                    push(
                        &mut numerical,
                        "count_momentum_conservation",
                        ck["summed_viscous_force_norm"].as_f64().unwrap(),
                        0.,
                    );
                }
            }
            force.push(json!({"step":step,"force_checks":checks,"alive":archive.steps[index].report.eligible,
                "revivals":archive.steps[index].report.revivals,"clones":archive.steps[index].report.clones}));
        }
        let name = if row { "row" } else { "count" };
        let path = store.save_archive(
            &format!("chapter04-reference-{name}-{start}"),
            gas.recording().unwrap(),
        )?;
        let checkpoint = store.save_checkpoint(
            &format!("chapter04-reference-{name}-{end}"),
            &gas.checkpoint(),
        )?;
        raw.push(json!({"archive":path,"checkpoint":checkpoint,"steps":[start,end]}));
    }
    let primitive = primitive_reference(&c)?;
    let kappa = primitive["kappa_F"].as_f64().unwrap();
    let beta = primitive["beta_F"].as_f64().unwrap();
    let mut comparisons = vec![];
    for (id, vals) in &numerical {
        comparisons.push(comparison(&format!("reference-{row}:{id}"),"def-w2-finite-population-qsd-regime","",vals,"equal",
        "Native N=200,d=3 full reference trajectory. Force formula reconstructed from each actual B1/B2 physical input stage, both dense influence arrays retained; force-provider identity is separate from QSD law convergence."));
    }
    comparisons.push(comparison(&format!("reference-{row}:B2_coercivity"),"cor-w2-reference-fitness-degeneracy","",&[(kappa,if row {0.9876} else {0.9936})],"equal",
        "Actual native reference parameters with declared count/row dense normalization; positive harmonic B2 margin computed analytically."));
    comparisons.push(comparison(&format!("reference-{row}:beta_F_margin"),"cor-w2-reference-fitness-degeneracy","",&[(beta,if row {0.071} else {0.002})],"upper",
        "Actual native reference and rigorous Gaussian survival lower bound; its event radius supplies a conservative beta_F, not a fitted mixing rate."));
    let frame = store.save_json(
        &format!("chapter04-reference-{row}-force-ledger"),
        &json!({"primitive":primitive,"force_frames":force,"raw_archives":raw}),
    )?;
    Ok(
        json!({"reference":true,"N":200,"d":3,"native_config":c,"seed":seed,"row_normalized":row,"steps":config.steps,
        "raw_archives":raw,"force_ledger":frame,"primitive":primitive,"comparisons":comparisons,
        "QSD_prediction_status":"Kernel/primitive prerequisites measured. Full minorization/eigenfunction/QSD laws belong to Chapter 6 law ensembles, never paired trajectory error."}),
    )
}

pub async fn run(config: &ExperimentConfig, store: &mut ArchiveStore) -> Result<Value> {
    config.validate()?;
    let populations: &[usize] = if config.compact { &[4] } else { &[4, 16, 64] };
    let dimensions: &[usize] = if config.compact { &[1] } else { &[1, 2] };
    let landscapes: &[Benchmark] = if config.compact {
        &[Benchmark::Quadratic]
    } else {
        &[
            Benchmark::Quadratic,
            Benchmark::Sphere,
            Benchmark::Rastrigin,
            Benchmark::Constant,
        ]
    };
    let mut cases = vec![];
    for &n in populations {
        for &d in dimensions {
            for &landscape in landscapes {
                let id = cases.len();
                cases.push(proposal_case(config, store, id, n, d, landscape, "canonical").await?);
            }
            for profile in ["selection", "revival", "singleton", "permutation"] {
                let id = cases.len();
                cases.push(
                    proposal_case(config, store, id, n, d, Benchmark::Quadratic, profile).await?,
                );
            }
        }
    }
    let mut references = vec![];
    for row in [false, true] {
        references.push(reference_case(config, store, row).await?);
    }
    let comparisons: Vec<_> = cases
        .iter()
        .chain(&references)
        .flat_map(|c| c["comparisons"].as_array().unwrap().iter().cloned())
        .collect();
    let failed = comparisons.iter().filter(|c| c["passed"] == false).count();
    Ok(
        json!({"chapter":4,"title":"Native Chapter 4 proposal and dense-reference validation","config":config,
        "cases":cases,"references":references,"comparisons":comparisons,
        "summary":{"cases":cases.len(),"reference_cases":references.len(),"native_steps":2*config.samples*cases.len()+2*config.steps,
            "comparisons":comparisons.len(),"comparisons_failed":failed,"new_engine_steps":2*config.samples*cases.len()+2*config.steps},
        "scope":"Complete immutable native archives, donor plans, sampled fitness, status masks, noise factors, component rotations and full physical stage states. Empirical proposal transport is permutation invariant and N-normalized. No QSD law rate inferred from pair errors."}),
    )
}

#[cfg(test)]
mod exact_conditional_tests {
    use super::comparison;
    #[test]
    fn a_conditional_violation_cannot_be_hidden_by_other_samples_or_standard_error() {
        let result = comparison(
            "case0:target_row_pressure",
            "prop-w2-realized-target-keystone",
            "",
            &[(1., 0.), (0., 100.)],
            "upper",
            "Retained exact conditional witnesses",
        );
        assert_eq!(result["passed"], false);
        assert_eq!(result["observed"], 1.);
        assert_eq!(result["bound"], 0.);
        assert_eq!(result["standard_error"], 0.);
    }

    #[test]
    fn a_bad_reference_timestep_is_rejected_without_a_replication_assumption() {
        let result = comparison(
            "reference-count:one_step_probe",
            "def-w2-finite-population-qsd-regime",
            "",
            &[(1., 0.), (0., 100.)],
            "upper",
            "Continuous reference trajectory",
        );
        assert_eq!(result["passed"], false);
        assert_eq!(result["observed"], 1.);
        assert_eq!(result["bound"], 0.);
        assert_eq!(result["standard_error"], 0.);
        let hypothesis = result["hypotheses"][2].as_str().unwrap();
        assert!(hypothesis.contains("Continuous native reference trajectory"));
        assert!(!hypothesis.contains("fixed-state restore"));
    }
}
