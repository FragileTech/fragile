//! Measurements of executed phase-space records and their selected interaction graph.
//! Geometric constructions are explicitly readouts, never substituted dynamics.
use super::{
    math::*,
    native::{Frame, frames},
};
use crate::{
    GasError, Result, RunArchive,
    fractal_set::{EventRef, FractalSet},
    physics::partvi::{ExperimentRequest, ExperimentResult, Series},
};
use serde_json::json;
use std::collections::BTreeMap;

pub(super) fn analyze(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    if a.steps.is_empty() {
        return Err(GasError::Capability(
            "Record at least one Euclidean Gas update".into(),
        ));
    }
    match r.experiment {
        1 | 11 | 15 | 23 | 24 | 29 => transport(r, a),
        26 => mechanics(r, a),
        31 => influences(r, a),
        _ => temporal(r, a),
    }
}
fn output(id: u32, title: &str, definition: &str, a: &RunArchive<f64>) -> ExperimentResult {
    let mut o = ExperimentResult::new(id, title, definition);
    o.details = json!({"status":"available","source":"executed_rust_euclidean_gas","recorded_steps":a.steps.len(),"epoch":a.epoch,"gas_config":a.gas_config});
    o
}
fn fields(a: &RunArchive<f64>) -> Result<(&str, &str, f64)> {
    match &a.gas_config.kinetic.integrator {
        crate::kinetic::KineticKind::Baoab {
            positions,
            velocities,
            dt,
            ..
        } => Ok((positions, velocities, *dt)),
        _ => Err(GasError::Capability(
            "This phase-space readout requires BAOAB recording".into(),
        )),
    }
}
fn transport(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let (xn, vn, _) = fields(a)?;
    let graph = FractalSet::from_archive(a);
    let mut rays = BTreeMap::<EventRef, Vec<C>>::new();
    for s in &a.steps {
        for (step, p) in [
            (s.report.step - 1, &s.before),
            (s.report.step, &s.final_population),
        ] {
            let x = p.observations.field(xn)?;
            let v = p.observations.field(vn)?;
            for i in 0..p.len() {
                if !p.validity[i].eligible(a.gas_config.include_truncated) {
                    continue;
                }
                let row: Vec<C> = x
                    .row(i)?
                    .iter()
                    .zip(v.row(i)?)
                    .map(|(&x, &v)| C::new(x, v))
                    .collect();
                let norm = dot(&row, &row).re.sqrt();
                if norm > 1e-12 && norm.is_finite() {
                    rays.insert(
                        EventRef {
                            epoch: s.epoch,
                            step,
                            slot: i as u32,
                            generation: p.generations[i],
                            version: p.version,
                        },
                        row.iter().map(|&z| z / norm).collect(),
                    );
                }
            }
        }
    }
    let overlap = |u: &EventRef, v: &EventRef| -> Option<C> {
        let z = dot(rays.get(u)?, rays.get(v)?);
        (z.abs() > 1e-10).then(|| z / z.abs())
    };
    let gauge =
        |e: &EventRef| C::phase(r.number("angle", 0.7) * (e.slot as f64 + 0.37 * e.step as f64));
    let mut loop_rows = vec![];
    let mut phase = vec![];
    let mut action = vec![];
    let mut derivative = vec![];
    let mut reverse_error: f64 = 0.;
    let mut gauge_error: f64 = 0.;
    let mut derivative_error: f64 = 0.;
    let mut uncovered = 0;
    for (i, t) in graph.triangles.iter().enumerate() {
        let [u, v, w] = t.vertices;
        let Some((uv, vw, wu)) = overlap(&u, &v)
            .zip(overlap(&v, &w))
            .zip(overlap(&w, &u))
            .map(|((x, y), z)| (x, y, z))
        else {
            uncovered += 1;
            continue;
        };
        let z = uv * vw * wu;
        let reversed = wu.conj() * vw.conj() * uv.conj();
        let changed = (gauge(&u).conj() * uv * gauge(&v))
            * (gauge(&v).conj() * vw * gauge(&w))
            * (gauge(&w).conj() * wu * gauge(&u));
        reverse_error = reverse_error.max((reversed - z.conj()).abs());
        gauge_error = gauge_error.max((changed - z).abs());
        let h = 1e-5;
        let fd = ((1. - (z * C::phase(h)).re) - (1. - (z * C::phase(-h)).re)) / (2. * h);
        derivative_error = derivative_error.max((fd - z.im).abs());
        phase.push([i as f64, z.im.atan2(z.re)]);
        action.push([i as f64, 1. - z.re]);
        derivative.push([i as f64, fd]);
        loop_rows.push(json!({"vertices":t.vertices,"boundary_edges":t.boundary_edges,"holonomy":[z.re,z.im],"conjugate_holonomy":[reversed.re,reversed.im],"edge_phase_derivative":z.im}));
    }
    let title = match r.experiment {
        1 => "Oriented transport on executed interaction triangles",
        11 => "Recorded phase-space ray holonomy",
        15 => "Conjugation-odd phase of recorded interaction loops",
        23 => "Local basis Ward identity on recorded ray transport",
        24 => "Measured loop action and phase variation",
        _ => "Executed interaction faces and oriented boundary transport",
    };
    let mut o = output(
        r.experiment,
        title,
        "Unit phase of Hermitian overlaps of normalized x+i v rays at actual graph events, multiplied around recorded interaction triangles",
        a,
    );
    o.metric(
        "Resolved interaction triangles",
        loop_rows.len() as f64,
        "triangles",
    )
    .metric("Unavailable ray triangles", uncovered as f64, "triangles")
    .metric("Orientation residual", reverse_error, "")
    .metric("Local phase covariance residual", gauge_error, "")
    .metric("Action variation residual", derivative_error, "");
    o.plot(
        "Measured oriented loop readouts",
        "recorded triangle",
        "value",
        vec![
            Series::line("Holonomy phase", phase),
            Series::line("1 - real holonomy", action),
            Series::line("Action edge-phase derivative", derivative),
        ],
    );
    o.details["loops"] = json!(loop_rows);
    o.details["graph_unresolved_sources"] = json!(graph.unresolved_sources);
    o.details["construction"] = json!(
        "U(1) connection derived from nonorthogonal recorded phase-space rays, with independent local frame rephasing. This construction does not assert SU(3) parallel transport or a Yang–Mills transition density."
    );
    o.details["interpretation"] = json!(if r.experiment == 15 {
        "Conjugating recorded rays reverses loop phase exactly. Comparing a transformed complete algorithm law requires separate transformed runs; a nonzero phase is not evidence of algorithmic CP violation."
    } else {
        "The action 1-Re(loop) is a measured geometric readout. Its phase derivative and local basis invariance are exact algebra; identification with the algorithm path action is a separate empirical hypothesis."
    });
    if loop_rows.is_empty() {
        o.details["status"] = json!("empty_support");
    }
    Ok(o)
}
fn mean(f: &Frame, squared: bool) -> f64 {
    f.eligible
        .iter()
        .enumerate()
        .filter(|(_, v)| **v)
        .map(|(i, _)| {
            if squared {
                f.x[i].iter().map(|v| v * v).sum()
            } else {
                f.x[i][0]
            }
        })
        .sum::<f64>()
        / f.eligible.len().max(1) as f64
}
fn temporal(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let f = frames(a)?;
    let (_, _, dt) = fields(a)?;
    let values: Vec<f64> = f.iter().map(|f| mean(f, false)).collect();
    let mut o = output(
        r.experiment,
        "Recorded algorithmic field measurement",
        "Pre-clone whole-swarm observables with fixed population normalization",
        a,
    );
    match r.experiment {
        7 => {
            o.title = "Finite-step weak generator on actual gas updates".into();
            let mut rows = vec![];
            let mut linear = vec![];
            let mut quadratic = vec![];
            let mut residual: f64 = 0.;
            let (xn, _, _) = fields(a)?;
            for s in &a.steps {
                let before = s.before.observations.field(xn)?;
                let after = s.final_population.observations.field(xn)?;
                let n = s.before.len() as f64;
                let mut dl = 0.;
                let mut dq = 0.;
                let mut prediction = 0.;
                for i in 0..s.before.len() {
                    if s.before.validity[i].eligible(a.gas_config.include_truncated)
                        && s.final_population.validity[i].eligible(a.gas_config.include_truncated)
                    {
                        for (&x, &y) in before.row(i)?.iter().zip(after.row(i)?) {
                            let dx = y - x;
                            dl += dx / n;
                            dq += (y * y - x * x) / n;
                            prediction += (2. * x * dx + dx * dx) / n;
                        }
                    }
                }
                residual = residual.max((dq - prediction).abs());
                linear.push([s.report.step as f64, dl / dt]);
                quadratic.push([s.report.step as f64, dq / dt]);
                rows.push(json!({"step":s.report.step,"quadratic_increment":dq,"drift_plus_quadratic_variation":prediction}));
            }
            o.metric("Discrete chain-rule residual", residual, "");
            o.plot(
                "Measured complete-step increments",
                "step",
                "increment / dt",
                vec![
                    Series::line("Linear observable", linear),
                    Series::line("Quadratic observable", quadratic),
                ],
            );
            o.details["rows"] = json!(rows);
            o.details["interpretation"] = json!(
                "Realized finite-step weak-generator samples on slots eligible at both endpoints; cloning jumps and finite-step quadratic variation are retained. Conditional expectations require continuation ensembles; a continuum generator is not assumed."
            );
        }
        20 | 25 => {
            let bins = r.usize("bins", 4).clamp(2, 12);
            let cut = (f.len() * 3 / 5).max(1);
            let train = &values[..cut];
            let lo = train.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = train.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let bin = |x: f64| {
                (((x - lo) / (hi - lo).max(1e-12) * bins as f64)
                    .floor()
                    .max(0.) as usize)
                    .min(bins - 1)
            };
            let mut counts = vec![0.; bins * bins];
            let mut held = vec![0.; bins * bins];
            for i in 1..values.len() {
                if f[i].epoch != f[i - 1].epoch || f[i].step != f[i - 1].step + 1 {
                    continue;
                }
                let k = bin(values[i - 1]) * bins + bin(values[i]);
                if i < cut {
                    counts[k] += 1.;
                } else {
                    held[k] += 1.;
                }
            }
            let mut p = vec![0.; bins * bins];
            let mut support = vec![false; bins];
            for i in 0..bins {
                let n: f64 = counts[i * bins..(i + 1) * bins].iter().sum();
                support[i] = n > 0.;
                if n > 0. {
                    for j in 0..bins {
                        p[i * bins + j] = counts[i * bins + j] / n;
                    }
                }
            }
            let mut delta: f64 = 0.;
            for i in 0..bins {
                for j in 0..bins {
                    if support[i] && support[j] {
                        delta = delta.max(
                            (0..bins)
                                .map(|k| (p[i * bins + k] - p[j * bins + k]).abs())
                                .sum::<f64>()
                                / 2.,
                        );
                    }
                }
            }
            let total = counts.iter().sum::<f64>();
            let mut kl = 0.;
            let alpha = 0.5;
            let denom = total + alpha * (bins * bins) as f64;
            for i in 0..bins {
                for j in 0..bins {
                    let q = (counts[i * bins + j] + alpha) / denom;
                    let rev = (counts[j * bins + i] + alpha) / denom;
                    kl += q * (q / rev).ln();
                }
            }
            let mut brier = 0.;
            let mut seen = 0.;
            for i in 0..bins {
                if support[i] {
                    for j in 0..bins {
                        brier += held[i * bins + j]
                            * (0..bins)
                                .map(|k| (p[i * bins + k] - f64::from(j == k)).powi(2))
                                .sum::<f64>();
                        seen += held[i * bins + j];
                    }
                }
            }
            o.title = if r.experiment == 20 {
                "Recorded transition information and time reversal"
            } else {
                "Measured descriptor transition contraction"
            }
            .into();
            o.metric("Empirical forward/reverse KL", kl, "nats")
                .metric("Supported-row Dobrushin coefficient", delta, "")
                .metric("Held-out transitions", seen, "transitions");
            if seen > 0. {
                o.metric("Held-out Brier score", brier / seen, "");
            }
            o.plot(
                "Recorded descriptor transition counts",
                "source × bins + target",
                "count",
                vec![
                    Series::line(
                        "Training",
                        counts
                            .iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64, v])
                            .collect(),
                    ),
                    Series::line(
                        "Held out",
                        held.iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64, v])
                            .collect(),
                    ),
                ],
            );
            o.details["transition"] = json!({"counts":counts,"heldout_counts":held,"matrix":p,"supported_rows":support,"bin_interval":[lo,hi],"training_frames":cut,"pseudocount_for_joint_kl_only":alpha});
            o.details["interpretation"] = json!(
                "Transition rows use training-only mean-position bins. The finite empirical descriptor channel need not be Markov for the full gas. Its contraction is not a full-chain spectral gap. Joint KL uses explicit Jeffreys smoothing and is not a path-space entropy-production estimate."
            );
        }
        27 => {
            o.title = "Survival and conditional population moments".into();
            let mut alive = vec![];
            let mut conditional = vec![];
            let mut increments = vec![];
            for (j, fr) in f.iter().enumerate() {
                let n = fr.eligible.iter().filter(|v| **v).count();
                alive.push([fr.step as f64, n as f64 / fr.eligible.len().max(1) as f64]);
                if n > 0 {
                    conditional.push([
                        fr.step as f64,
                        mean(fr, true) * fr.eligible.len() as f64 / n as f64,
                    ]);
                }
                if j > 0 && f[j - 1].epoch == fr.epoch {
                    increments.push([fr.step as f64, alive[j][1] - alive[j - 1][1]]);
                }
            }
            o.plot(
                "Actual eligible population and conditional radius",
                "step",
                "value",
                vec![
                    Series::line("Eligible fraction", alive),
                    Series::line("Conditional mean squared radius", conditional),
                    Series::line("Net eligible-fraction increment", increments),
                ],
            );
            o.details["interpretation"] = json!(
                "Measured conditional population moments along the recorded run. Population eligibility is not the survival probability of an independent killed trajectory; a QSD claim requires independent surviving-run ensembles and temporal stability."
            );
        }
        28 => {
            o.title = "Empirical temporal reflection matrix".into();
            let m = r
                .usize("modes", 3)
                .clamp(2, 6)
                .min((values.len() / 3).max(2));
            if values.len() <= 2 * m {
                return Err(GasError::Capability(
                    "Reflection matrix needs at least twice modes plus one recorded frames".into(),
                ));
            }
            let center = values.iter().sum::<f64>() / values.len() as f64;
            let mut h = vec![0.; m * m];
            let mut count = 0;
            for origin in m..f.len() - m {
                if (origin - m..=origin + m).any(|i| {
                    f[i].epoch != f[origin].epoch
                        || f[i].step != f[origin].step + i as u64 - origin as u64
                }) {
                    continue;
                }
                count += 1;
                for i in 0..m {
                    for j in 0..m {
                        h[i * m + j] +=
                            (values[origin - i - 1] - center) * (values[origin + j + 1] - center);
                    }
                }
            }
            if count == 0 {
                return Err(GasError::Capability(
                    "No consecutive reflection windows".into(),
                ));
            }
            for z in &mut h {
                *z /= count as f64;
            }
            let sym: Vec<_> = (0..m * m)
                .map(|k| (h[k] + h[k % m * m + k / m]) / 2.)
                .collect();
            let (e, _) = symmetric_eigen(&sym, m);
            o.metric(
                "Minimum reflection eigenvalue",
                e.iter().copied().fold(f64::INFINITY, f64::min),
                "",
            )
            .metric("Reflection windows", count as f64, "windows");
            o.plot(
                "Measured reflection spectrum",
                "mode",
                "eigenvalue",
                vec![Series::line(
                    "Symmetric reflection matrix",
                    e.iter().enumerate().map(|(i, &v)| [i as f64, v]).collect(),
                )],
            );
            o.details["matrix"] = json!(h);
            o.details["interpretation"] = json!(
                "Time-reflection bilinear form of centered frame means, symmetrized explicitly. A negative empirical eigenvalue is a diagnostic requiring independent-run uncertainty; covariance positivity does not imply reflection positivity."
            );
        }
        30 => {
            o.title = "Translation covariance of measured density modes".into();
            let k = r.number("wavenumber", 1.);
            let shift = r.number("amplitude", 0.5);
            let mut actual = vec![];
            let mut predicted = vec![];
            let mut error: f64 = 0.;
            for fr in &f {
                let mut z = C::ZERO;
                let mut moved = C::ZERO;
                for (i, &valid) in fr.eligible.iter().enumerate() {
                    if valid {
                        z = z + C::phase(k * fr.x[i][0]);
                        moved = moved + C::phase(k * (fr.x[i][0] + shift));
                    }
                }
                z = z / fr.eligible.len() as f64;
                moved = moved / fr.eligible.len() as f64;
                let p = z * C::phase(k * shift);
                error = error.max((moved - p).abs());
                actual.push([fr.step as f64, moved.re]);
                predicted.push([fr.step as f64, p.re]);
            }
            o.metric("Density translation residual", error, "");
            o.plot(
                "Actual-coordinate Fourier translation",
                "step",
                "real density mode",
                vec![
                    Series::line("Shifted coordinate readout", actual),
                    Series::line("Phase-covariance prediction", predicted),
                ],
            );
            o.details["interpretation"] = json!(
                "Exact observable transformation on recorded coordinates. This does not assume translation invariance of the objective or complete transition kernel."
            );
        }
        33 => {
            o.title = "Measured momentum bispinors".into();
            let mut dets = vec![];
            let mut energies = vec![];
            let mut error: f64 = 0.;
            let mut n = 0;
            for fr in &f {
                for (i, &valid) in fr.eligible.iter().enumerate() {
                    if valid {
                        let p: [f64; 3] =
                            std::array::from_fn(|j| fr.v[i].get(j).copied().unwrap_or(0.));
                        let e = p.iter().map(|v| v * v).sum::<f64>().sqrt();
                        let matrix = [
                            C::from(e + p[2]),
                            C::new(p[0], -p[1]),
                            C::new(p[0], p[1]),
                            C::from(e - p[2]),
                        ];
                        let det = determinant(&matrix, 2);
                        error = error.max(det.abs() / (1. + e * e));
                        dets.push([n as f64, det.re]);
                        energies.push([n as f64, e]);
                        n += 1;
                    }
                }
            }
            o.metric("Relative null-bispinor determinant residual", error, "");
            o.plot(
                "Bispinors from actual recorded velocities",
                "walker record",
                "value",
                vec![
                    Series::line("Velocity norm", energies),
                    Series::line("Bispinor determinant", dets),
                ],
            );
            o.details["interpretation"] = json!(
                "Three recorded velocity components are embedded as the explicitly null four-vector (|v|,v); missing spatial components are zero. This algebraic readout does not measure a particle dispersion relation or infer a mass."
            );
            o.details["mass"] = json!(null);
        }
        _ => {
            return Err(GasError::Configuration(
                "Unknown executed QFT experiment".into(),
            ));
        }
    }
    Ok(o)
}
fn mechanics(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let (_, vn, _) = fields(a)?;
    let mut o = output(
        r.experiment,
        "Executed cloning and restitution mechanical ledger",
        "All-slot unit-mass momentum and kinetic energy at each recorded clone stage",
        a,
    );
    let mut rows = vec![];
    let mut momentum = vec![];
    let mut energy = vec![];
    let mut telescope: f64 = 0.;
    for s in &a.steps {
        let mut stages = vec![];
        for name in ["pre_clone", "literal_clone", "post_transform", "post_clone"] {
            if let Some(stage) = s.stages.iter().find(|s| s.stage == name) {
                let v = stage
                    .fields
                    .get(vn)
                    .ok_or_else(|| GasError::MissingField(vn.into()))?;
                let d = v.item_shape[0];
                let p: Vec<f64> = (0..d)
                    .map(|j| v.values.chunks_exact(d).map(|v| v[j]).sum())
                    .collect();
                let e = v.values.iter().map(|v| 0.5 * v * v).sum::<f64>();
                stages.push((name, p, e));
            }
        }
        if let (Some(first), Some(last)) = (stages.first(), stages.last()) {
            let sum = stages.windows(2).map(|w| w[1].2 - w[0].2).sum::<f64>();
            telescope = telescope.max((sum - (last.2 - first.2)).abs());
            momentum.push([s.report.step as f64, last.1[0] - first.1[0]]);
            energy.push([s.report.step as f64, last.2 - first.2]);
        }
        rows.push(json!({"step":s.report.step,"stages":stages,"clone_transform":a.gas_config.clone_transform}));
    }
    o.metric("Energy ledger telescoping residual", telescope, "");
    o.plot(
        "Recorded clone-stage transfers",
        "step",
        "increment",
        vec![
            Series::line("Total momentum x", momentum),
            Series::line("Total kinetic energy", energy),
        ],
    );
    o.details["rows"] = json!(rows);
    o.details["interpretation"] = json!(
        "Literal donor copying, restitution and subsequent clone operations may transfer total momentum and energy. Conservation is tested only for configured operations with the corresponding paired identity; changes of eligibility do not remove slots from this ledger."
    );
    Ok(o)
}
fn influences(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let mut o = output(
        r.experiment,
        "Recorded conditional intervention footprint",
        "Exact linear sensitivity of recorded weighted-copy writes with donor selections and gates held fixed",
        a,
    );
    let selected = r.usize("source_slot", 0);
    let amplitude = r.number("amplitude", 0.1);
    let mut rows = vec![];
    let mut footprint = vec![];
    for s in &a.steps {
        let mut total = 0.;
        for (i, c) in s.report.clone_plan.choices.iter().enumerate() {
            let sensitivity = if c.accepted {
                c.donors
                    .iter()
                    .filter(|d| {
                        let src = s.report.clone_plan.sources[d.pool_index as usize];
                        src.slot as usize == selected
                            && src.frame + 1 == s.report.step
                            && src.version == s.before.version
                    })
                    .map(|d| d.weight)
                    .sum::<f64>()
            } else {
                f64::from(i == selected)
            };
            total += sensitivity.abs();
            rows.push(json!({"step":s.report.step,"recipient":i,"source_slot":selected,"conditional_derivative":sensitivity,"finite_intervention_increment":amplitude*sensitivity,"accepted":c.accepted}));
        }
        footprint.push([s.report.step as f64, total]);
    }
    o.plot(
        "Observed copy-map intervention propagation",
        "step",
        "sum absolute sensitivity",
        vec![Series::line("Recorded donor-copy Jacobian", footprint)],
    );
    o.details["rows"] = json!(rows);
    o.details["interpretation"] = json!(
        "The Jacobian conditions on the executed random choices and applies only to literal copying. Endogenous changes in donor selection, gates, restitution and later kinetics require rerunning complete checkpoints and are not replaced by this conditional derivative."
    );
    Ok(o)
}
