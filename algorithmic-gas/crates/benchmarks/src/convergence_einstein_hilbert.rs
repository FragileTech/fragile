//! Quantitative centered identities for the native Einstein--Hilbert operators.
//! Matchings are integrated as matchings, not as independent donor rows.
use algorithmic_gas::{GasError, Result, cloning::CloneDecision, tracking::RecordedStep};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchingMoments {
    pub entering_variance: f64,
    pub signed_radial_increment: f64,
    pub mean_center_displacement: Vec<f64>,
    pub center_displacement_second_moment: f64,
    pub center_variance: f64,
    pub center_variance_bound: f64,
    pub expected_variance: f64,
    pub expected_cross_covariance: f64,
    /// Conditional variance bound for centered spread, with the actual matching.
    pub spread_variance_bound: f64,
}

pub fn mean(x: &[f64], d: usize) -> Vec<f64> {
    let n = x.len() / d;
    (0..d)
        .map(|a| x.chunks_exact(d).map(|r| r[a]).sum::<f64>() / n as f64)
        .collect()
}
pub fn covariance(x: &[f64], y: &[f64], d: usize) -> f64 {
    let (mx, my) = (mean(x, d), mean(y, d));
    x.chunks_exact(d)
        .zip(y.chunks_exact(d))
        .map(|(x, y)| (0..d).map(|a| (x[a] - mx[a]) * (y[a] - my[a])).sum::<f64>())
        .sum::<f64>()
        / (x.len() / d) as f64
}
pub fn energy(v: &[f64], d: usize) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>() / (v.len() / d) as f64
}

/// Exact finite matching and gate expectation (EH.M1--M4). The fitness is
/// frozen after the independent diversity matching. All rows must be live.
pub fn matching_moments(
    x: &[f64],
    v: &[f64],
    fitness: &[f64],
    d: usize,
    decision: &CloneDecision,
    step: u64,
) -> Result<MatchingMoments> {
    let n = fitness.len();
    if n < 4
        || d == 0
        || x.len() != n * d
        || v.len() != x.len()
        || x.iter().chain(v).any(|a| !a.is_finite())
        || fitness.iter().any(|f| !f.is_finite() || *f <= 0.)
    {
        return Err(GasError::Configuration(
            "EH moments require N >= 4, finite rows and positive fitness".into(),
        ));
    }
    decision.validate()?;
    let nf = n as f64;
    let (q, r) = if n.is_multiple_of(2) {
        (1. / (nf - 1.), 1. / ((nf - 1.) * (nf - 3.)))
    } else {
        (1. / nf, 1. / (nf * (nf - 2.)))
    };
    let (mx, mv) = (mean(x, d), mean(v, d));
    let entering_variance = covariance(x, x, d);
    if !step.is_multiple_of(decision.every) {
        return Ok(MatchingMoments {
            entering_variance,
            signed_radial_increment: 0.,
            mean_center_displacement: vec![0.; d],
            center_displacement_second_moment: 0.,
            center_variance: 0.,
            center_variance_bound: 0.,
            expected_variance: entering_variance,
            expected_cross_covariance: covariance(x, v, d),
            spread_variance_bound: 0.,
        });
    }
    let mut h = vec![0.; d];
    let mut incident = vec![vec![0.; d]; n];
    let (mut k, mut h2, mut radial, mut cross) = (0., 0., 0., 0.);
    for i in 0..n {
        for j in i + 1..n {
            let aij = decision.acceptance_probability(step, fitness[i], fitness[j]);
            let aji = decision.acceptance_probability(step, fitness[j], fitness[i]);
            for a in 0..d {
                let dx = x[j * d + a] - x[i * d + a];
                let he = (aij - aji) * dx;
                h[a] += he;
                incident[i][a] += he;
                incident[j][a] += he;
                k += (aij + aji) * dx * dx;
                h2 += he * he;
                radial +=
                    (aij - aji) * ((x[j * d + a] - mx[a]).powi(2) - (x[i * d + a] - mx[a]).powi(2));
                cross += dx * (aij * (v[i * d + a] - mv[a]) - aji * (v[j * d + a] - mv[a]));
            }
        }
    }
    let hs: f64 = h.iter().map(|a| a * a).sum();
    let is: f64 = incident.iter().flatten().map(|a| a * a).sum();
    let second = (q * k + r * (hs + h2 - is)) / (nf * nf);
    let center_mean: Vec<_> = h.iter().map(|a| q * a / nf).collect();
    let center_variance = second - q * q * hs / (nf * nf);
    let fourth = x
        .chunks_exact(d)
        .map(|row| {
            row.iter()
                .zip(&mx)
                .map(|(a, m)| (a - m).powi(2))
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>()
        / nf;
    Ok(MatchingMoments {
        entering_variance,
        signed_radial_increment: q * radial / nf,
        mean_center_displacement: center_mean,
        center_displacement_second_moment: second,
        center_variance,
        center_variance_bound: (q + r) * entering_variance,
        expected_variance: entering_variance + q * radial / nf - second,
        expected_cross_covariance: covariance(x, v, d) + q * cross / nf,
        spread_variance_bound: (q + r)
            * ((fourth - entering_variance.powi(2)).max(0.).sqrt() + 2. * entering_variance)
                .powi(2),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EhStepEstimate {
    pub step: u64,
    pub cloning: MatchingMoments,
    pub realized_clone_variance: f64,
    pub clone_residual: f64,
    pub kinetic_position_mean: f64,
    pub kinetic_position_variance: f64,
    pub kinetic_residual: f64,
    pub final_position_variance: f64,
    pub final_velocity_energy: f64,
    pub b1_velocity_variance: f64,
    pub b1_position_velocity_covariance: f64,
    pub kappa: f64,
    pub velocity_drift_factor_bound: f64,
    pub exact_max_relative_residual: f64,
    pub exact_checks_passed: bool,
}

/// Source-mapped stage checks, including the exact Gaussian position law before
/// the second B kick, which leaves positions unchanged. No independence between
/// interacting rows is assumed when computing simulation error bars.
pub fn analyze_step(
    step: &RecordedStep<f64>,
    h: f64,
    nu: f64,
    temperature: f64,
    decision: &CloneDecision,
) -> Result<EhStepEstimate> {
    let field = |stage: &str, name: &str| -> Result<&[f64]> {
        step.stages
            .iter()
            .find(|s| s.stage == stage)
            .and_then(|s| s.fields.get(name))
            .map(|f| f.values.as_slice())
            .ok_or_else(|| GasError::Configuration(format!("missing EH {stage}/{name}")))
    };
    let n = step.report.pre_clone_fitness.fitness.len();
    let x = field("pre_clone", "positions")?;
    let v = field("pre_clone", "velocities")?;
    let d = x.len() / n;
    let moments = matching_moments(
        x,
        v,
        &step.report.pre_clone_fitness.fitness,
        d,
        decision,
        step.report.step,
    )?;
    let xc = field("post_clone", "positions")?;
    let vc = field("post_clone", "velocities")?;
    let u = field("B1", "velocities")?;
    let w = field("O", "velocities")?;
    let y = field("A2", "positions")?;
    let xp = field("B2", "positions")?;
    let vp = field("B2", "velocities")?;
    let (t, c) = (h / 2., (-h).exp());
    let b = t * (1. + c);
    let noise_variance = temperature * (1. - c * c);
    let m: Vec<_> = xc.iter().zip(u).map(|(x, v)| x + b * v).collect();
    let base = covariance(&m, &m, d);
    let tau2 = t * t * noise_variance;
    let position_mean = base + (1. - 1. / n as f64) * d as f64 * tau2;
    let position_variance =
        4. * tau2 * base / n as f64 + 2. * d as f64 * (n - 1) as f64 * tau2 * tau2 / (n * n) as f64;
    let graph = step
        .graph
        .as_ref()
        .ok_or_else(|| GasError::Configuration("EH diagnostics require recorded graph".into()))?;
    let bounds = algorithmic_gas::variants::einstein_hilbert::graph_kick_bounds(
        &graph.graph,
        &graph.weights["riemannian_kernel_volume"],
        nu,
        h,
    )?;
    let kappa = bounds.quarter_kick_moment_factor;
    let mut residual: f64 = 0.;
    let mut equal = |left: f64, right: f64| {
        residual = residual.max((left - right).abs() / (1. + left.abs() + right.abs()));
    };
    for a in 0..x.len() {
        equal(vc[a], v[a]);
        equal(xp[a], y[a]);
        equal(y[a], xc[a] + t * (u[a] + w[a]));
    }
    let wc = covariance(xc, xc, d);
    let vu = covariance(u, u, d);
    let cxu = covariance(xc, u, d);
    equal(base, wc + 2. * b * cxu + b * b * vu);
    // Retain the full noise/B2 cross terms in the centered quadratic identities.
    let eta: Vec<_> = w.iter().zip(u).map(|(w, u)| w - c * u).collect();
    let e: Vec<_> = vp.iter().zip(w).map(|(v, w)| v - w).collect();
    equal(
        covariance(y, vp, d),
        c * cxu
            + b * c * vu
            + covariance(xc, &eta, d)
            + (b + t * c) * covariance(u, &eta, d)
            + t * covariance(&eta, &eta, d)
            + covariance(y, &e, d),
    );
    equal(
        covariance(vp, vp, d),
        c * c * vu
            + 2. * c * covariance(u, &eta, d)
            + covariance(&eta, &eta, d)
            + 2. * covariance(w, &e, d)
            + covariance(&e, &e, d),
    );
    let max_energy = |v: &[f64]| {
        v.chunks_exact(d)
            .map(|r| r.iter().map(|a| a * a).sum::<f64>())
            .fold(0., f64::max)
    };
    let mut passed = residual < 2e-11 && wc <= 2. * moments.entering_variance + 1e-11;
    for (before, after) in [(vc, u), (w, vp)] {
        passed &= max_energy(after) <= max_energy(before) * (1. + 2e-11) + 1e-12;
        passed &= energy(after, d) <= kappa * kappa * energy(before, d) * (1. + 2e-11) + 1e-12;
    }
    let output = covariance(y, y, d);
    Ok(EhStepEstimate {
        step: step.report.step,
        realized_clone_variance: wc,
        clone_residual: wc - moments.expected_variance,
        cloning: moments,
        kinetic_position_mean: position_mean,
        kinetic_position_variance: position_variance,
        kinetic_residual: output - position_mean,
        final_position_variance: output,
        final_velocity_energy: energy(vp, d),
        b1_velocity_variance: vu,
        b1_position_velocity_covariance: cxu,
        kappa,
        velocity_drift_factor_bound: c * c * kappa.powi(4),
        exact_max_relative_residual: residual,
        exact_checks_passed: passed,
    })
}
