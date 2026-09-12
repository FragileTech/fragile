//! Archive-derived observables, strictly chronological prediction and complex spectra.
use super::{math::*, param};
use crate::{
    GasError, Result, RunArchive,
    physics::partvi::{ExperimentRequest, ExperimentResult, Series},
};
use serde_json::json;
struct Frame {
    step: u64,
    epoch: u64,
    version: u64,
    x: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    eligible: Vec<bool>,
}
fn frames(a: &RunArchive<f64>) -> Result<Vec<Frame>> {
    let (pos, vel) = match &a.gas_config.kinetic.integrator {
        crate::kinetic::KineticKind::Baoab {
            positions,
            velocities,
            ..
        } => (positions.as_str(), velocities.as_str()),
        _ => {
            return Err(GasError::Capability(
                "native QFT observables require recorded BAOAB positions and velocities".into(),
            ));
        }
    };
    let mut data = vec![];
    for s in &a.steps {
        let b = s
            .stages
            .iter()
            .find(|b| b.stage == "pre_clone")
            .ok_or_else(|| GasError::Capability("pre_clone stage is unavailable".into()))?;
        let x = b
            .fields
            .get(pos)
            .ok_or_else(|| GasError::MissingField(pos.into()))?;
        let v = b
            .fields
            .get(vel)
            .ok_or_else(|| GasError::MissingField(vel.into()))?;
        if x.item_shape.len() != 1 || x.item_shape != v.item_shape {
            return Err(GasError::Shape(
                "native QFT needs matching position and velocity vectors".into(),
            ));
        }
        let d = x.item_shape[0];
        let xx: Vec<Vec<f64>> = x.values.chunks_exact(d).map(|r| r.to_vec()).collect();
        let vv: Vec<Vec<f64>> = v.values.chunks_exact(d).map(|r| r.to_vec()).collect();
        let valid = b
            .validity
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.eligible(a.gas_config.include_truncated)
                    && xx[i].iter().chain(&vv[i]).all(|x| x.is_finite())
            })
            .collect();
        data.push(Frame {
            step: s.report.step,
            epoch: s.epoch,
            version: b.version,
            x: xx,
            v: vv,
            eligible: valid,
        });
    }
    Ok(data)
}
fn scalar(f: &Frame, i: usize, feature: &str) -> f64 {
    match feature {
        "position_x" | "mean_position_x" => f.x[i][0],
        "speed_squared" | "mean_speed_squared" => f.v[i].iter().map(|x| x * x).sum(),
        _ => f.v[i][0],
    }
}
fn missing(id: u32, title: &str, why: &str) -> ExperimentResult {
    let mut o = ExperimentResult::new(
        id,
        title,
        "Recorded archive with unavailable requested estimate",
    );
    o.note(why);
    o.details = json!({"status":"unavailable","reason":why,"stage":"pre_clone"});
    o
}
pub(super) fn analyze(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let f = frames(a)?;
    match r.experiment {
        3 | 6 => gram(r, &f),
        4 => measured_channel(r, &f),
        5 => regional(r, &f),
        12 | 13 | 32 => prediction(r, &f),
        36 => spectrum(r, &f, a),
        _ => Err(GasError::Capability(
            "unknown archive QFT calculation".into(),
        )),
    }
}
fn gram(r: &ExperimentRequest, frames: &[Frame]) -> Result<ExperimentResult> {
    let fields = r.text("fields", "phase_space");
    if !["phase_space", "positions", "velocities"].contains(&fields) {
        return Err(GasError::Configuration(
            "fields must be phase_space, positions or velocities".into(),
        ));
    }
    let mut rows = vec![];
    let mut sources = vec![];
    let mut frame_counts = vec![];
    for f in frames {
        let mut current = vec![];
        for (i, &valid) in f.eligible.iter().enumerate() {
            if valid {
                let row = match fields {
                    "positions" => f.x[i].iter().copied().take(6).collect::<Vec<_>>(),
                    "velocities" => f.v[i].iter().copied().take(6).collect(),
                    _ => f.x[i]
                        .iter()
                        .take(3)
                        .chain(f.v[i].iter().take(3))
                        .copied()
                        .collect(),
                };
                current.push(row);
                sources.push([f.epoch, f.step, f.version, i as u64]);
            }
        }
        frame_counts.push([f.epoch, f.step, current.len() as u64]);
        if let Some(first) = current.first() {
            rows.push(
                (0..first.len())
                    .map(|j| current.iter().map(|x| x[j]).sum::<f64>() / current.len() as f64)
                    .collect::<Vec<_>>(),
            );
        }
    }
    if rows.len() < 2 {
        return Ok(missing(
            r.experiment,
            "Recorded centered-observable Gram space",
            "At least two frames with eligible contributors are required.",
        ));
    }
    let d = rows[0].len();
    let center: Vec<f64> = (0..d)
        .map(|j| rows.iter().map(|x| x[j]).sum::<f64>() / rows.len() as f64)
        .collect();
    let centered: Vec<Vec<f64>> = rows
        .iter()
        .map(|x| x.iter().zip(&center).map(|(x, m)| x - m).collect())
        .collect();
    let g: Vec<f64> = (0..d * d)
        .map(|k| centered.iter().map(|x| x[k / d] * x[k % d]).sum::<f64>() / rows.len() as f64)
        .collect();
    let (e, v) = symmetric_eigen(&g, d);
    let threshold = e.iter().copied().fold(0., f64::max) * 1e-10;
    let mut keep: Vec<usize> = e
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (x > threshold.max(1e-24)).then_some(i))
        .collect();
    keep.sort_by(|&i, &j| e[j].total_cmp(&e[i]));
    let rank = keep.len();
    keep.truncate(r.usize("modes", 6).clamp(1, 6));
    let m = keep.len();
    let n = 1usize << m;
    let coordinates: Vec<Vec<f64>> = (0..d)
        .map(|i| keep.iter().map(|&k| v[i * d + k] * e[k].sqrt()).collect())
        .collect();
    let projected: Vec<f64> = (0..d * d)
        .map(|k| {
            coordinates[k / d]
                .iter()
                .zip(&coordinates[k % d])
                .map(|(x, y)| x * y)
                .sum()
        })
        .collect();
    let mut car_residual: f64 = 0.;
    if r.experiment == 3 && m > 0 {
        let basis: Vec<Vec<C>> = (0..m).map(|k| creator(m, k)).collect();
        let creators: Vec<Vec<C>> = coordinates
            .iter()
            .map(|c| {
                (0..n * n)
                    .map(|i| (0..m).fold(C::ZERO, |s, k| s + basis[k][i] * c[k]))
                    .collect()
            })
            .collect();
        for i in 0..d {
            for j in 0..d {
                let a = adjoint(&creators[i], n);
                let ac = mul(&a, &creators[j], n);
                let ca = mul(&creators[j], &a, n);
                for k in 0..n * n {
                    let expected = if k / n == k % n {
                        projected[i * d + j]
                    } else {
                        0.
                    };
                    car_residual = car_residual.max((ac[k] + ca[k] - C::from(expected)).abs());
                }
            }
        }
    }
    let mut wedge = 0.;
    let mut same = 0.;
    let mut determinant = 0.;
    if d >= 2 {
        same = g[1];
        determinant = g[0] * g[d + 1] - g[1] * g[d];
        for a in &centered {
            for b in &centered {
                wedge +=
                    (a[0] * b[1] - a[1] * b[0]).powi(2) / (2. * (rows.len() * rows.len()) as f64);
            }
        }
    }
    let mut o = ExperimentResult::new(
        r.experiment,
        if r.experiment == 6 {
            "Recorded products and independent replica wedges"
        } else {
            "Recorded Gram quotient and CAR operators"
        },
        "Whole-swarm frame observables: eligible means of declared pre-clone fields, centered under the finite empirical frame law",
    );
    o.metric("Gram rank", rank as f64, "modes")
        .metric("Observed rows", rows.len() as f64, "frames")
        .metric(
            "Raw contributing walker rows",
            sources.len() as f64,
            "walker frames",
        )
        .metric("Nullity", (d - rank) as f64, "modes")
        .metric("Retained CAR modes", m as f64, "modes")
        .metric("Fock dimension", n as f64, "states")
        .metric("Measured-mode CAR residual", car_residual, "")
        .metric(
            "Discarded covariance norm",
            g.iter()
                .zip(&projected)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),
            "",
        )
        .metric("Same-frame product", same, "")
        .metric("Independent-replica wedge norm squared", wedge, "")
        .metric("Empirical Gram determinant", determinant, "")
        .metric(
            "Replica determinant residual",
            (wedge - determinant).abs(),
            "",
        );
    o.plot(
        "Measured covariance spectrum",
        "mode",
        "eigenvalue",
        vec![Series::line(
            "Empirical whole-frame Gram",
            e.iter().enumerate().map(|(i, &x)| [i as f64, x]).collect(),
        )],
    );
    o.plot(
        "Two distinct recorded products",
        "product",
        "empirical value",
        vec![Series::line(
            "Same-frame, replica wedge, Gram determinant",
            vec![[0., same], [1., wedge], [2., determinant]],
        )],
    );
    o.details = json!({"status":"available","stage":"pre_clone","fields":fields,"aggregation":"eligible whole-swarm means per frame; walkers are contributors, not independent Fock modes","column_means":center,"centered_frame_readouts":centered,"gram":g,"projected_gram":projected,"gram_dimension":d,"eigenvalues":e,"rank_threshold":threshold,"retained_eigenmode_indices":keep,"feature_coordinates_in_quotient":coordinates,"source_epoch_step_version_slot":sources,"frame_eligible_counts":frame_counts,"normalization":"Equal weight per nonempty recorded frame; independent replica law is the product of this finite empirical law","law":"finite empirical record law, no stationarity imposed","theory_labels":["def-lqft-record-fock-space","thm-lqft-record-fock-reconstruction","prop-lqft-finite-record-transfer","thm-lqft-replica-isomorphism"],"validation":"Coordinate CAR multiplication and double empirical-replica summation are independent of the Gram determinant evaluation; these validate finite-law algebra, not population sampling accuracy."});
    Ok(o)
}
fn measured_channel(r: &ExperimentRequest, frames: &[Frame]) -> Result<ExperimentResult> {
    let lag = r.usize("lag", 1).clamp(1, 32);
    let modes = r.usize("modes", 2).clamp(1, 3);
    let fraction = param(r, "train_fraction", 0.6, 0.25, 0.8);
    let cut = (frames.len() as f64 * fraction) as usize;
    let read = |f: &Frame| -> Option<Vec<f64>> {
        let slots: Vec<usize> = f
            .eligible
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some(i))
            .collect();
        if slots.is_empty() {
            return None;
        }
        Some(
            (0..modes)
                .map(|j| {
                    slots.iter().map(|&i| f.v[i][j % f.v[i].len()]).sum::<f64>()
                        / slots.len() as f64
                })
                .collect(),
        )
    };
    let pairs = |start: usize, end: usize| -> Vec<(Vec<f64>, Vec<f64>)> {
        (start..end.saturating_sub(lag))
            .filter_map(|i| {
                let a = &frames[i];
                let b = &frames[i + lag];
                if a.epoch != b.epoch || b.step != a.step + lag as u64 {
                    return None;
                }
                Some((read(a)?, read(b)?))
            })
            .collect()
    };
    let train = pairs(0, cut);
    let test = pairs(cut, frames.len());
    if train.len() < modes + 2 || test.len() < 2 {
        return Ok(missing(
            4,
            "Measured two-time CAR channel",
            "Insufficient chronological training/held-out frame pairs.",
        ));
    }
    let mean = |side: usize| -> Vec<f64> {
        (0..modes)
            .map(|j| {
                train
                    .iter()
                    .map(|p| if side == 0 { p.0[j] } else { p.1[j] })
                    .sum::<f64>()
                    / train.len() as f64
            })
            .collect()
    };
    let ms = mean(0);
    let mt = mean(1);
    let cov = |s: usize, t: usize| -> Vec<f64> {
        (0..modes * modes)
            .map(|k| {
                train
                    .iter()
                    .map(|p| {
                        let (a, ma) = if s == 0 { (&p.0, &ms) } else { (&p.1, &mt) };
                        let (b, mb) = if t == 0 { (&p.0, &ms) } else { (&p.1, &mt) };
                        (a[k / modes] - ma[k / modes]) * (b[k % modes] - mb[k % modes])
                    })
                    .sum::<f64>()
                    / train.len() as f64
            })
            .collect()
    };
    let gs = cov(0, 0);
    let gt = cov(1, 1);
    let cross = cov(0, 1);
    let basis = |g: &[f64]| -> Vec<Vec<f64>> {
        let (e, v) = symmetric_eigen(g, modes);
        let tol = e.iter().copied().fold(0., f64::max) * 1e-10;
        (0..modes)
            .filter(|&k| e[k] > tol.max(1e-24))
            .map(|k| (0..modes).map(|i| v[i * modes + k] / e[k].sqrt()).collect())
            .collect()
    };
    let ws = basis(&gs);
    let wt = basis(&gt);
    let ns = ws.len();
    let nt = wt.len();
    if ns == 0 || nt == 0 {
        return Ok(missing(
            4,
            "Measured two-time CAR channel",
            "The observed source or target covariance has zero rank.",
        ));
    }
    let transfer: Vec<f64> = ws
        .iter()
        .flat_map(|s| {
            wt.iter().map(|t| {
                (0..modes)
                    .map(|i| {
                        (0..modes)
                            .map(|j| s[i] * cross[i * modes + j] * t[j])
                            .sum::<f64>()
                    })
                    .sum()
            })
        })
        .collect();
    let square: Vec<f64> = (0..nt * nt)
        .map(|k| {
            (0..ns)
                .map(|i| transfer[i * nt + k / nt] * transfer[i * nt + k % nt])
                .sum()
        })
        .collect();
    let (e, _) = symmetric_eigen(&square, nt);
    let mut eta: Vec<f64> = e.iter().map(|v| v.max(0.).sqrt()).collect();
    eta.sort_by(|a, b| b.total_cmp(a));
    eta.truncate(ns.min(nt));
    if eta.iter().any(|&x| x > 1. + 1e-7) {
        return Err(GasError::Configuration(
            "Empirical conditional transfer violates the contraction bound".into(),
        ));
    }
    for x in &mut eta {
        *x = x.min(1.);
    }
    let mut heldout = vec![0.; ns * nt];
    for (a, b) in &test {
        for i in 0..ns {
            for j in 0..nt {
                let x = (0..modes).map(|k| ws[i][k] * (a[k] - ms[k])).sum::<f64>();
                let y = (0..modes).map(|k| wt[j][k] * (b[k] - mt[k])).sum::<f64>();
                heldout[i * nt + j] += x * y / test.len() as f64;
            }
        }
    }
    let mut o = super::channel::analyze_contraction(r, &eta);
    o.title = "Actual recorded two-time contraction and CAR channel".into();
    o.model="Conditional transfer of the finite joint empirical whole-swarm record law, with distinct source and target marginal whitening".into();
    // A measured pair law defines one transfer; its powers are not measured Markov evolution.
    o.plots
        .retain(|p| p.title != "Top exterior sector through time");
    o.metric("Training pairs", train.len() as f64, "pairs")
        .metric("Held-out pairs", test.len() as f64, "pairs")
        .metric(
            "Held-out cross-moment error",
            heldout
                .iter()
                .zip(&transfer)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),
            "",
        )
        .metric(
            "Source-target covariance drift",
            gs.iter()
                .zip(&gt)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),
            "",
        );
    o.plot(
        "Chronological out-of-sample cross moments",
        "matrix entry",
        "moment",
        vec![
            Series::line(
                "Training transfer",
                transfer
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
            Series::line(
                "Held-out measured",
                heldout
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
        ],
    );
    o.details["recorded_law"] = json!({"stage":"pre_clone","features":"eligible whole-swarm velocity coordinate means","lag":lag,"train_frame_indices":[0,cut],"heldout_frame_indices":[cut,frames.len()],"source_mean":ms,"target_mean":mt,"source_covariance":gs,"target_covariance":gt,"cross_covariance":cross,"source_whitening":ws,"target_whitening":wt,"transfer":transfer,"heldout_cross_moment":heldout,"source_rank":ns,"target_rank":nt,"frame_epoch_step_version":frames.iter().map(|f|[f.epoch,f.step,f.version]).collect::<Vec<_>>(),"theory":"prop-lqft-finite-record-transfer","dynamics":"All executed copying, clipping, force, noise and eligibility events contribute through their recorded whole-swarm endpoints; no compressed-state Markov assumption is imposed","validation":"CP and exterior identities are finite-law algebra; held-out drift measures separate temporal generalization, without an iid error-bar assumption"});
    Ok(o)
}

fn regional(r: &ExperimentRequest, frames: &[Frame]) -> Result<ExperimentResult> {
    let split = param(r, "region_split", 0., -10., 10.);
    let feature = r.text("feature", "velocity_x");
    if !["velocity_x", "position_x", "speed_squared"].contains(&feature) {
        return Err(GasError::Configuration("Unknown regional feature".into()));
    }
    let mut pairs = vec![];
    let mut counts = vec![];
    for f in frames {
        let mut left = vec![];
        let mut right = vec![];
        for (i, &valid) in f.eligible.iter().enumerate() {
            if valid {
                if f.x[i][0] <= split {
                    left.push(scalar(f, i, feature));
                } else {
                    right.push(scalar(f, i, feature));
                }
            }
        }
        counts.push(json!({"step":f.step,"left":left.len(),"right":right.len()}));
        if !left.is_empty() && !right.is_empty() {
            pairs.push([f.step as f64, mean(&left), mean(&right)]);
        }
    }
    if pairs.len() < 3 {
        return Ok(missing(
            5,
            "Recorded regional covariance",
            "Both regions must contain eligible walkers in at least three recorded frames; move the region split or record longer.",
        ));
    }
    let ml = pairs.iter().map(|p| p[1]).sum::<f64>() / pairs.len() as f64;
    let mr = pairs.iter().map(|p| p[2]).sum::<f64>() / pairs.len() as f64;
    let vl = pairs.iter().map(|p| (p[1] - ml).powi(2)).sum::<f64>() / pairs.len() as f64;
    let vr = pairs.iter().map(|p| (p[2] - mr).powi(2)).sum::<f64>() / pairs.len() as f64;
    let cov = pairs.iter().map(|p| (p[1] - ml) * (p[2] - mr)).sum::<f64>() / pairs.len() as f64;
    let rho = if vl * vr > 1e-24 {
        Some((cov / (vl * vr).sqrt()).clamp(-1., 1.))
    } else {
        None
    };
    let mut o = ExperimentResult::new(
        5,
        "Recorded regional covariance and CAR locality",
        "Centered framewise region averages with disjoint position-defined regions",
    );
    o.metric("Cross covariance", cov, "")
        .metric("Left variance", vl, "")
        .metric("Right variance", vr, "");
    if let Some(rho) = rho {
        let c0 = creator(2, 0);
        let c1 = creator(2, 1);
        let g: Vec<C> = c0
            .iter()
            .zip(&c1)
            .map(|(&a, &b)| a * rho + b * (1. - rho * rho).max(0.).sqrt())
            .collect();
        let ac = mul(&adjoint(&c0, 4), &g, 4);
        let ca = mul(&g, &adjoint(&c0, 4), 4);
        let residual = ac
            .iter()
            .zip(ca)
            .enumerate()
            .map(|(i, (&a, b))| (a + b) - identity(4)[i] * rho)
            .map(C::abs)
            .fold(0., f64::max);
        o.metric("Whitened cross singular value", rho.abs(), "")
            .metric("Recorded CAR locality residual", residual, "");
    } else {
        o.note("A region has zero variance, so normalized CAR covariance is unavailable.");
    }
    o.plot(
        "Regional measurements",
        "recorded step",
        "regional average",
        vec![
            Series::line("Left", pairs.iter().map(|p| [p[0], p[1]]).collect()),
            Series::line("Right", pairs.iter().map(|p| [p[0], p[2]]).collect()),
        ],
    );
    o.details = json!({"stage":"pre_clone","region_split":split,"feature":feature,"frame_counts":counts,"paired_frames":pairs.len(),"normalized_covariance":rho,"centering":[ml,mr],"regions":"left x[0] <= split; right x[0] > split","sampling":"Averages retain temporal correlation; covariance is descriptive for the selected record."});
    Ok(o)
}
fn prediction(r: &ExperimentRequest, frames: &[Frame]) -> Result<ExperimentResult> {
    let feature = r.text("descriptor", "mean_speed_squared");
    if !["mean_velocity_x", "mean_position_x", "mean_speed_squared"].contains(&feature) {
        return Err(GasError::Configuration(
            "Unknown whole-swarm descriptor".into(),
        ));
    }
    let t = frames.len();
    let lag = r.usize("lag", 2).clamp(1, 8);
    let fraction = param(r, "train_fraction", 0.6, 0.5, 0.8);
    let cut = (fraction * t as f64).floor() as usize;
    let test_start = cut + lag;
    if t < 12 || cut < 5 || test_start + lag >= t {
        return Ok(missing(
            r.experiment,
            "Recorded descriptor prediction",
            "Record at least twelve frames and leave a held-out segment longer than the selected lag after the chronological guard.",
        ));
    }
    let mut values = vec![];
    for f in frames {
        let x: Vec<f64> = f
            .eligible
            .iter()
            .enumerate()
            .filter(|(_, valid)| **valid)
            .map(|(i, _)| scalar(f, i, feature))
            .collect();
        if x.is_empty() {
            return Ok(missing(
                r.experiment,
                "Recorded descriptor prediction",
                "Every training and held-out frame needs an eligible descriptor. Missing frames are not bridged.",
            ));
        }
        values.push(mean(&x));
    }
    let requested = r.usize("bins", 4).clamp(2, 8);
    let mut sorted = values[..cut - 1].to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut edges = vec![];
    for j in 1..requested {
        let index = (j * sorted.len() / requested).clamp(1, sorted.len() - 1);
        let low = sorted[index - 1];
        let high = sorted[index];
        if high > low {
            let edge = (low + high) / 2.;
            if edges.last().is_none_or(|last| edge > *last) {
                edges.push(edge);
            }
        }
    }
    let n = edges.len() + 1;
    let state = |v: f64| edges.partition_point(|&e| v > e);
    let states: Vec<usize> = values.iter().map(|&x| state(x)).collect();
    let mut counts = vec![0usize; n * n];
    let mut origin = vec![0usize; n];
    let mut skipped = 0;
    for i in 0..cut - 1 {
        if frames[i + 1].epoch != frames[i].epoch || frames[i + 1].step != frames[i].step + 1 {
            skipped += 1;
            continue;
        }
        counts[states[i] * n + states[i + 1]] += 1;
        origin[states[i]] += 1;
    }
    if origin.contains(&0) {
        return Ok(missing(
            r.experiment,
            "Recorded descriptor prediction",
            "A fitted descriptor state has no observed training departure. Increase the recording length or reduce the bins; unknown transitions are not filled in.",
        ));
    }
    let transition: Vec<f64> = counts
        .iter()
        .enumerate()
        .map(|(i, &x)| x as f64 / origin[i / n] as f64)
        .collect();
    let training_count = origin.iter().sum::<usize>();
    let occupancy: Vec<f64> = origin
        .iter()
        .map(|&x| x as f64 / training_count as f64)
        .collect();
    let mut brier = vec![];
    let mut baseline = vec![];
    let mut errors = vec![];
    let mut lag_counts = vec![];
    let mut matrices = vec![];
    for l in 1..=lag {
        let p = rpower(&transition, n, l);
        let mut losses = vec![];
        let mut base = vec![];
        let mut actual = vec![0usize; n * n];
        let mut actual_rows = vec![0usize; n];
        for i in test_start..t - l {
            if frames[i + l].epoch != frames[i].epoch
                || frames[i + l].step != frames[i].step + l as u64
            {
                continue;
            }
            let start = states[i];
            let end = states[i + l];
            losses.push(
                (0..n)
                    .map(|j| (p[start * n + j] - f64::from(j == end)).powi(2))
                    .sum::<f64>(),
            );
            base.push(
                (0..n)
                    .map(|j| (occupancy[j] - f64::from(j == end)).powi(2))
                    .sum::<f64>(),
            );
            actual[start * n + end] += 1;
            actual_rows[start] += 1;
        }
        if !losses.is_empty() {
            brier.push([l as f64, mean(&losses)]);
            baseline.push([l as f64, mean(&base)]);
            let mut residual = 0.;
            let mut entries = 0;
            let empirical: Vec<Option<f64>> = actual
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if actual_rows[i / n] > 0 {
                        Some(x as f64 / actual_rows[i / n] as f64)
                    } else {
                        None
                    }
                })
                .collect();
            for (i, x) in empirical.iter().enumerate() {
                if let Some(x) = x {
                    residual += (x - p[i]).powi(2);
                    entries += 1;
                }
            }
            errors.push([l as f64, (residual / entries.max(1) as f64).sqrt()]);
            matrices.push(json!({"lag":l,"predicted_transition":p,"heldout_transition":empirical,"heldout_origin_counts":actual_rows}));
        }
        lag_counts.push([l, losses.len()]);
    }
    let pushed: Vec<f64> = (0..n)
        .map(|j| (0..n).map(|i| occupancy[i] * transition[i * n + j]).sum())
        .collect();
    let stationarity_defect = pushed
        .iter()
        .zip(&occupancy)
        .map(|(x, y)| (x - y).abs())
        .sum::<f64>();
    let groups = if n >= 3 { 2 } else { 1 };
    let membership: Vec<usize> = (0..n).map(|i| i * groups / n).collect();
    let mut group_mass = vec![0.; groups];
    for i in 0..n {
        group_mass[membership[i]] += occupancy[i];
    }
    let project = |p: &[f64]| {
        let mut a = vec![0.; groups * groups];
        for i in 0..n {
            for j in 0..n {
                a[membership[i] * groups + membership[j]] +=
                    occupancy[i] / group_mass[membership[i]] * p[i * n + j];
            }
        }
        a
    };
    let coarse = project(&transition);
    let direct = project(&rpower(&transition, n, 2));
    let compressed = rpower(&coarse, groups, 2);
    let defect: Vec<f64> = direct.iter().zip(&compressed).map(|(a, b)| a - b).collect();
    let mut o = ExperimentResult::new(
        r.experiment,
        match r.experiment {
            12 => "Recorded descriptor memory",
            13 => "Held-out multitime descriptor prediction",
            _ => "Recorded QFT predictive pipeline",
        },
        "Training-only quantile partitions and transition counts with a chronological guard before validation",
    );
    o.metric("Training transitions", training_count as f64, "transitions")
        .metric("Effective descriptor states", n as f64, "states")
        .metric(
            "Training projected two-step memory defect",
            defect.iter().map(|x| x * x).sum::<f64>().sqrt(),
            "",
        );
    o.metric(
        "Training occupancy stationarity defect",
        stationarity_defect,
        "L1",
    );
    if let Some(x) = brier.first() {
        o.metric("Held-out one-step Brier score", x[1], "");
    }
    o.plot(
        "Held-out multitime prediction",
        "lag",
        "Brier score",
        vec![
            Series::line("Training transition powers", brier),
            Series::line("Training occupancy baseline", baseline),
        ],
    );
    o.plot(
        "Held-out conditional distribution error",
        "lag",
        "matrix RMSE",
        vec![Series::line(
            "Empirical held-out law minus training prediction",
            errors,
        )],
    );
    o.plot(
        "Observed descriptor across the split",
        "recorded step",
        "descriptor",
        vec![
            Series::line(
                "Training",
                (0..cut)
                    .map(|i| [frames[i].step as f64, values[i]])
                    .collect(),
            ),
            Series::line(
                "Guard",
                (cut..test_start)
                    .map(|i| [frames[i].step as f64, values[i]])
                    .collect(),
            ),
            Series::line(
                "Held out",
                (test_start..t)
                    .map(|i| [frames[i].step as f64, values[i]])
                    .collect(),
            ),
        ],
    );
    o.details = json!({"status":"available","stage":"pre_clone","descriptor":feature,"edges_fitted_on_training_origins":edges,"training_frames":[frames[0].step,frames[cut-1].step],"guard_frames":[frames[cut].step,frames[test_start-1].step],"heldout_frames":[frames[test_start].step,frames[t-1].step],"training_counts":counts,"training_origin_counts":origin,"training_transition":transition,"training_occupancy":occupancy,"pushed_training_occupancy":pushed,"stationarity_defect":stationarity_defect,"partition_scope":"One training-fitted quantile partition. No nested-refinement or prediction-completion claim is inferred from its fitted transition.","heldout_lag_counts":lag_counts,"heldout_matrices":matrices,"coarse_groups":membership,"projected_two_step":direct,"compressed_two_step":compressed,"projection_memory_defect":defect,"skipped_nonconsecutive_training_pairs":skipped,"interpretation":"Held-out errors include finite-sample error and closure error. The projection defect is an exact algebraic calculation on the fitted transition, not a population-theorem estimate.","sampling":"Held-out temporal rows can be correlated. No independent-draw confidence band is assigned."});
    Ok(o)
}
fn read_pair(
    frames: &[Frame],
    twistors: Option<&[super::TwistorFrame]>,
    source: usize,
    sink: usize,
    slot: usize,
    channels: usize,
    alpha: f64,
) -> Option<(Vec<C>, Vec<C>)> {
    if !frames[source].eligible[slot] || !frames[sink].eligible[slot] {
        return None;
    }
    if let Some(tw) = twistors {
        let tri = tw[source].triples.get(slot).copied().flatten()?;
        let (a, wa, ..) = super::twistor_value(&tw[source], tri, alpha)?;
        let (b, wb, ..) = super::twistor_at_sink(&tw[source], &tw[sink], tri, alpha)?;
        Some((
            vec![a, wa[0], wa[1]][..channels].to_vec(),
            vec![b, wb[0], wb[1]][..channels].to_vec(),
        ))
    } else {
        if frames[source].x[slot].len() < channels {
            return None;
        }
        Some((
            (0..channels)
                .map(|j| C::new(frames[source].x[slot][j], frames[source].v[slot][j]))
                .collect(),
            (0..channels)
                .map(|j| C::new(frames[sink].x[slot][j], frames[sink].v[slot][j]))
                .collect(),
        ))
    }
}
fn frame_average(
    frames: &[Frame],
    twistors: Option<&[super::TwistorFrame]>,
    frame: usize,
    channels: usize,
    alpha: f64,
) -> Vec<C> {
    let mut mean = vec![C::ZERO; channels];
    for i in 0..frames[frame].eligible.len() {
        if let Some((x, _)) = read_pair(frames, twistors, frame, frame, i, channels, alpha) {
            for j in 0..channels {
                mean[j] = mean[j] + x[j];
            }
        }
    }
    // The population size is fixed; invalid local readouts are zero-extended.
    for x in &mut mean {
        *x = *x / frames[frame].eligible.len() as f64;
    }
    mean
}
#[allow(clippy::too_many_arguments)]
fn correlation(
    frames: &[Frame],
    twistors: Option<&[super::TwistorFrame]>,
    range: std::ops::Range<usize>,
    lag: usize,
    channels: usize,
    center: &[C],
    alpha: f64,
    frame_mean: bool,
) -> (Vec<C>, usize) {
    let mut c = vec![C::ZERO; channels * channels];
    let mut count = 0;
    if range.len() <= lag {
        return (c, 0);
    }
    for t in range.start..range.end - lag {
        if frames[t + lag].epoch != frames[t].epoch
            || frames[t + lag].step != frames[t].step + lag as u64
        {
            continue;
        }
        let pairs = if frame_mean {
            vec![(
                frame_average(frames, twistors, t, channels, alpha),
                frame_average(frames, twistors, t + lag, channels, alpha),
            )]
        } else {
            (0..frames[t].eligible.len())
                .filter_map(|i| read_pair(frames, twistors, t, t + lag, i, channels, alpha))
                .collect()
        };
        for (a, b) in pairs {
            for j in 0..channels {
                for k in 0..channels {
                    c[j * channels + k] =
                        c[j * channels + k] + (a[j] - center[j]).conj() * (b[k] - center[k]);
                }
            }
            count += 1;
        }
    }
    if count > 0 {
        for x in &mut c {
            *x = *x / count as f64;
        }
    }
    (c, count)
}
fn whiten(c: &[C], dimension: usize, values: &[f64], vectors: &[Vec<C>], keep: &[usize]) -> Vec<C> {
    let rank = keep.len();
    let mut result = vec![C::ZERO; rank * rank];
    for (i, &vi) in keep.iter().enumerate() {
        for (j, &vj) in keep.iter().enumerate() {
            let cv: Vec<C> = (0..dimension)
                .map(|row| {
                    (0..dimension).fold(C::ZERO, |s, k| s + c[row * dimension + k] * vectors[vj][k])
                })
                .collect();
            result[i * rank + j] = dot(&vectors[vi], &cv) / (values[vi] * values[vj]).sqrt();
        }
    }
    result
}
fn ordered_modes(c: &[C], rank: usize, previous: Option<&[C]>) -> Vec<C> {
    let values = general_eigenvalues(c, rank);
    if let Some(previous) = previous {
        let mut unused = values;
        previous
            .iter()
            .map(|p| {
                let index = (0..unused.len())
                    .min_by(|&a, &b| (unused[a] - *p).abs2().total_cmp(&(unused[b] - *p).abs2()))
                    .unwrap();
                unused.remove(index)
            })
            .collect()
    } else {
        let mut v = values;
        v.sort_by(|a, b| b.abs().total_cmp(&a.abs()));
        v
    }
}
fn regression(points: &[[f64; 2]]) -> Option<(f64, f64, f64)> {
    if points.len() < 3 {
        return None;
    }
    let x = points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64;
    let y = points.iter().map(|p| p[1]).sum::<f64>() / points.len() as f64;
    let xx = points.iter().map(|p| (p[0] - x).powi(2)).sum::<f64>();
    if xx <= 1e-20 {
        return None;
    }
    let slope = points.iter().map(|p| (p[0] - x) * (p[1] - y)).sum::<f64>() / xx;
    let intercept = y - slope * x;
    let mse = points
        .iter()
        .map(|p| (p[1] - intercept - slope * p[0]).powi(2))
        .sum::<f64>()
        / points.len() as f64;
    Some((intercept, slope, mse.sqrt()))
}
fn spectrum(
    r: &ExperimentRequest,
    frames: &[Frame],
    archive: &RunArchive<f64>,
) -> Result<ExperimentResult> {
    let channels = r.usize("channels", 2).clamp(1, 3);
    let maxlag = r.usize("max_lag", 12).clamp(2, 24);
    let fit_start = r.usize("fit_start", 1).clamp(1, 8);
    let fit_end = r.usize("fit_end", 6).clamp(2, 16).min(maxlag);
    let t = frames.len();
    let cut = (0.6 * t as f64).floor() as usize;
    let held_start = cut + maxlag;
    let mode = r.text("readout", "phase_space");
    if !matches!(mode, "phase_space" | "twistor") {
        return Err(GasError::Configuration(
            "spectral readout must be phase_space or twistor".into(),
        ));
    }
    let aggregation = r.text("aggregation", "frame_mean");
    if !matches!(aggregation, "frame_mean" | "source_pairs") {
        return Err(GasError::Configuration(
            "spectral aggregation must be frame_mean or source_pairs".into(),
        ));
    }
    let frame_mean = aggregation == "frame_mean";
    if frames
        .iter()
        .any(|f| f.eligible.len() != frames[0].eligible.len())
    {
        return Err(GasError::Capability(
            "spectral observable requires a fixed population size across frames".into(),
        ));
    }
    let alpha = 0.1 + super::amp(r);
    if t < 16 || cut <= maxlag || held_start + maxlag >= t {
        return Ok(missing(
            36,
            "Recorded complex correlators and spectra",
            "The chronological training, lag guard, and held-out segments need more frames. For max_lag=12, record at least 96 frames, or reduce max_lag.",
        ));
    }
    let tw = if mode == "twistor" {
        Some(super::twistor_frames(r, Some(archive))?)
    } else {
        None
    };
    let tws = tw.as_deref();
    let mut center = vec![C::ZERO; channels];
    let mut center_count = 0;
    for f in 0..cut {
        let observations = if frame_mean {
            vec![frame_average(frames, tws, f, channels, alpha)]
        } else {
            (0..frames[f].eligible.len())
                .filter_map(|i| read_pair(frames, tws, f, f, i, channels, alpha).map(|(x, _)| x))
                .collect()
        };
        for x in observations {
            for j in 0..channels {
                center[j] = center[j] + x[j];
            }
            center_count += 1;
        }
    }
    if center_count < channels + 2 {
        return Ok(missing(
            36,
            "Recorded complex correlators and spectra",
            "Insufficient valid training observations for the requested channels.",
        ));
    }
    for x in &mut center {
        *x = *x / center_count as f64;
    }
    let (c0, _) = correlation(frames, tws, 0..cut, 0, channels, &center, alpha, frame_mean);
    let (e, v) = hermitian_modes(&c0, channels);
    let threshold = e.iter().copied().fold(0., f64::max) * 1e-9;
    let keep: Vec<usize> = e
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (x > threshold.max(1e-20)).then_some(i))
        .collect();
    let rank = keep.len();
    if rank == 0 {
        return Ok(missing(
            36,
            "Recorded complex correlators and spectra",
            "Centered covariance has no positive observable mode.",
        ));
    }
    let dt = match archive.gas_config.kinetic.integrator {
        crate::kinetic::KineticKind::Baoab { dt, .. } => dt,
        _ => 1.,
    };
    let mut train_modes: Vec<Vec<C>> = vec![];
    let mut test_modes: Vec<Vec<C>> = vec![];
    let mut counts = vec![];
    let mut matrices = vec![];
    let mut imaginary = vec![];
    let mut hermitian = vec![];
    let mut eigen_residual: f64 = 0.;
    for lag in 0..=maxlag {
        let (c, n) = correlation(
            frames,
            tws,
            0..cut,
            lag,
            channels,
            &center,
            alpha,
            frame_mean,
        );
        let (test, nt) = correlation(
            frames,
            tws,
            held_start..t,
            lag,
            channels,
            &center,
            alpha,
            frame_mean,
        );
        if n == 0 || nt == 0 {
            return Ok(missing(
                36,
                "Recorded complex correlators and spectra",
                "A requested training or held-out lag has no same-epoch consecutive observation pairs.",
            ));
        }
        let z = whiten(&c, channels, &e, &v, &keep);
        let zt = whiten(&test, channels, &e, &v, &keep);
        let roots = ordered_modes(&z, rank, train_modes.last().map(Vec::as_slice));
        let test_roots = ordered_modes(&zt, rank, Some(&roots));
        for (matrix, spectrum) in [(&z, &roots), (&zt, &test_roots)] {
            let scale = 1. + matrix.iter().map(|z| z.abs()).sum::<f64>();
            for &lambda in spectrum {
                if !lambda.re.is_finite() || !lambda.im.is_finite() {
                    return Ok(missing(
                        36,
                        "Recorded complex correlators and spectra",
                        "The selected generalized spectral problem has no numerically resolved finite roots.",
                    ));
                }
                let mut shifted = matrix.clone();
                for i in 0..rank {
                    shifted[i * rank + i] = shifted[i * rank + i] - lambda;
                }
                eigen_residual =
                    eigen_residual.max(determinant(&shifted, rank).abs() / scale.powi(rank as i32));
            }
        }
        if eigen_residual > 1e-6 {
            return Ok(missing(
                36,
                "Recorded complex correlators and spectra",
                "The characteristic-equation residual exceeds tolerance; reduce channels or improve covariance support.",
            ));
        }
        train_modes.push(roots);
        test_modes.push(test_roots);
        counts.push(json!({"lag":lag,"training_pairs":n,"heldout_pairs":nt}));
        imaginary.push([lag as f64 * dt, c[channels.saturating_sub(1)].im]);
        hermitian.push([lag as f64 * dt, difference(&c, &adjoint(&c, channels))]);
        matrices.push(json!({"lag":lag,"real":c.iter().map(|z|z.re).collect::<Vec<_>>(),"imaginary":c.iter().map(|z|z.im).collect::<Vec<_>>(),"heldout_real":test.iter().map(|z|z.re).collect::<Vec<_>>(),"heldout_imaginary":test.iter().map(|z|z.im).collect::<Vec<_>>() }));
    }
    let mut fits = vec![];
    let mut series = vec![];
    for j in 0..rank {
        let mut log_points = vec![];
        let mut phase_points = vec![];
        let mut last_phase = 0.;
        let mut phase = 0.;
        let mut first = true;
        let mut complex = false;
        for (lag, modes) in train_modes
            .iter()
            .enumerate()
            .take(fit_end + 1)
            .skip(fit_start)
        {
            let z = modes[j];
            if z.abs() <= 1e-10 || !z.abs().is_finite() {
                continue;
            }
            let arg = z.im.atan2(z.re);
            if first {
                phase = arg;
                first = false;
            } else {
                let mut d = arg - last_phase;
                while d > std::f64::consts::PI {
                    d -= std::f64::consts::TAU;
                }
                while d < -std::f64::consts::PI {
                    d += std::f64::consts::TAU;
                }
                phase += d;
            }
            last_phase = arg;
            log_points.push([lag as f64 * dt, z.abs().ln()]);
            phase_points.push([lag as f64 * dt, phase]);
            complex |= z.im.abs() > 1e-7 * z.abs() || z.re <= 0.;
        }
        if let (Some((intercept, slope, rmse)), Some((phase0, frequency, phase_rmse))) =
            (regression(&log_points), regression(&phase_points))
        {
            let predicted: Vec<C> = (0..=maxlag)
                .map(|lag| {
                    C::phase(phase0 + frequency * lag as f64 * dt)
                        * (intercept + slope * lag as f64 * dt).exp()
                })
                .collect();
            let held_error = (fit_start..=maxlag)
                .map(|lag| (test_modes[lag][j] - predicted[lag]).abs2())
                .sum::<f64>()
                / (maxlag - fit_start + 1) as f64;
            fits.push(json!({"mode":j,"status":"available","decay_rate":-slope,"frequency":frequency,"log_fit_rmse":rmse,"phase_fit_rmse":phase_rmse,"heldout_complex_rmse":held_error.sqrt(),"complex_or_signed_mode":complex,"mass":null,"positive_real_decay_rate":if frame_mean&&!complex&&slope<0.{Some(-slope)}else{None},"fit_lags":[fit_start,fit_end],"fit_points":log_points.len()}));
            series.push(Series::line(
                format!("Mode {} training fit magnitude", j + 1),
                predicted
                    .iter()
                    .enumerate()
                    .map(|(i, z)| [i as f64 * dt, z.abs()])
                    .collect(),
            ));
        } else {
            fits.push(json!({"mode":j,"status":"unavailable","reason":"At least three nonzero mode values in the selected fit interval are required","mass":null}));
        }
        series.push(Series::line(
            format!("Mode {} training real", j + 1),
            train_modes
                .iter()
                .enumerate()
                .map(|(i, z)| [i as f64 * dt, z[j].re])
                .collect(),
        ));
        series.push(Series::line(
            format!("Mode {} training imaginary", j + 1),
            train_modes
                .iter()
                .enumerate()
                .map(|(i, z)| [i as f64 * dt, z[j].im])
                .collect(),
        ));
        series.push(Series::line(
            format!("Mode {} held-out magnitude", j + 1),
            test_modes
                .iter()
                .enumerate()
                .map(|(i, z)| [i as f64 * dt, z[j].abs()])
                .collect(),
        ));
    }
    let mut o = ExperimentResult::new(
        36,
        "Recorded complex correlators and rank-aware spectral fits",
        "Recorded frame means or source-pair diagnostics, training-only covariance whitening, complex temporal fits",
    );
    o.metric("Generalized spectral root residual", eigen_residual, "");
    o.metric("Retained covariance rank", rank as f64, "modes")
        .metric(
            "Training observations",
            center_count as f64,
            if frame_mean {
                "frames"
            } else {
                "walker frames"
            },
        );
    for fit in &fits {
        if let Some(v) = fit["decay_rate"].as_f64() {
            o.metric(
                format!("Mode {} magnitude decay", fit["mode"].as_u64().unwrap() + 1),
                v,
                "per algorithmic time",
            );
        }
    }
    o.plot(
        "Training fit and held-out complex modes",
        "algorithmic lag time",
        "mode value",
        series,
    );
    o.plot(
        "Measured complex matrix diagnostics",
        "algorithmic lag time",
        "value",
        vec![
            Series::line("Off-diagonal imaginary part", imaginary),
            Series::line("Hermitian defect", hermitian),
        ],
    );
    o.details = json!({"status":"available","stage":"pre_clone","readout":mode,"aggregation":aggregation,"channel_definition":if mode=="twistor"{if frame_mean{"Fixed 1/N sum of tau, Pauli W_x, Pauli W_y from each record's own immutable donor triplets"}else{"tau, Pauli W_x, Pauli W_y from source-frozen triplets"}}else{if frame_mean{"Fixed 1/N sum of eligible x[j] + i v[j] per frame"}else{"x[j] + i v[j], numerical walker slot frozen across lag"}},"training_frames":[frames[0].step,frames[cut-1].step],"guard_frames":[frames[cut].step,frames[held_start-1].step],"heldout_frames":[frames[held_start].step,frames[t-1].step],"training_channel_means_real":center.iter().map(|z|z.re).collect::<Vec<_>>(),"training_channel_means_imaginary":center.iter().map(|z|z.im).collect::<Vec<_>>(),"covariance_eigenvalues":e,"rank_threshold":threshold,"lag_counts":counts,"complex_correlations":matrices,"fits":fits,"normalization":if frame_mean{"Each local observable is zero-extended and summed with fixed 1/N weight; each lag averages same-epoch consecutive frame pairs. Training alone selects centering, whitening and fits."}else{"Source-frozen pair diagnostic divides by its lag-specific valid pair count. Training alone selects centering, whitening and fits."},"interpretation":"The measured finite-window decay and frequency characterize this recorded observable. A particle mass requires an identified positive transfer representation and asymptotic spectral control; mass is not inferred from a positive fitted decay alone."});
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metric(r: &ExperimentResult, label: &str) -> f64 {
        r.metrics
            .iter()
            .find(|m| m.label == label)
            .unwrap()
            .value
            .unwrap()
    }
    #[test]
    fn exact_region_readouts_give_the_predicted_covariance() {
        let data: Vec<Frame> = (0..20)
            .map(|t| Frame {
                step: t,
                epoch: 0,
                version: t,
                x: vec![vec![-1.], vec![1.]],
                v: vec![vec![t as f64], vec![2. * t as f64]],
                eligible: vec![true; 2],
            })
            .collect();
        let r = regional(
            &ExperimentRequest {
                experiment: 5,
                parameters: json!({}),
            },
            &data,
        )
        .unwrap();
        assert!((metric(&r, "Whitened cross singular value") - 1.).abs() < 1e-12);
        assert!(metric(&r, "Recorded CAR locality residual") < 1e-12);
    }
    #[test]
    fn deterministic_chain_predicts_independent_chronological_segment() {
        let data: Vec<Frame> = (0..96)
            .map(|t| Frame {
                step: t,
                epoch: 0,
                version: t,
                x: vec![vec![0.]],
                v: vec![vec![(t % 2) as f64]],
                eligible: vec![true],
            })
            .collect();
        let r = prediction(
            &ExperimentRequest {
                experiment: 13,
                parameters: json!({"descriptor":"mean_velocity_x","lag":4}),
            },
            &data,
        )
        .unwrap();
        assert_eq!(r.details["status"], "available");
        assert!(metric(&r, "Held-out one-step Brier score") < 1e-12);
        assert!(r.plots[0].series[0].points.iter().all(|p| p[1] < 1e-12));
        let end = r.details["training_frames"][1].as_u64().unwrap();
        let start = r.details["heldout_frames"][0].as_u64().unwrap();
        assert!(start > end + 4);
    }
    #[test]
    fn partition_edges_ignore_heldout_extremes() {
        let mut data: Vec<Frame> = (0..96)
            .map(|t| Frame {
                step: t,
                epoch: 0,
                version: t,
                x: vec![vec![0.]],
                v: vec![vec![(t % 7) as f64]],
                eligible: vec![true],
            })
            .collect();
        let request = ExperimentRequest {
            experiment: 13,
            parameters: json!({"descriptor":"mean_velocity_x"}),
        };
        let a = prediction(&request, &data).unwrap();
        for f in data.iter_mut().skip(60) {
            f.v[0][0] += 10000.;
        }
        let b = prediction(&request, &data).unwrap();
        assert_eq!(
            a.details["edges_fitted_on_training_origins"],
            b.details["edges_fitted_on_training_origins"]
        );
        assert_eq!(
            a.details["training_transition"],
            b.details["training_transition"]
        );
    }
}
