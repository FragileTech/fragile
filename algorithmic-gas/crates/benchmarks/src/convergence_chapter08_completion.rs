//! Source-scoped interfaces for the canonical marked population law.
//! Array addresses identify recorded operands, never a labelled swarm metric.
use algorithmic_gas::{GasError, Population, Result};
use serde::{Deserialize, Serialize};

pub const SOURCE: &str =
    include_str!("../../../docs/source/2_fractal_gas/convergence_program/08_mean_field.md");

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub id: String,
    pub source_label: String,
    pub lhs: f64,
    pub rhs: f64,
    pub allowance: f64,
    pub relation: String,
    pub passed: bool,
    pub operands: serde_json::Value,
    pub hypotheses: String,
}
impl Comparison {
    pub fn identity(
        id: &str,
        label: &str,
        lhs: f64,
        rhs: f64,
        operands: serde_json::Value,
    ) -> Self {
        let allowance = 1e-9 * (1. + lhs.abs() + rhs.abs());
        Self { id: id.into(), source_label: label.into(), lhs, rhs, allowance,
            relation: "equality in real arithmetic; recorded binary64 tolerance".into(),
            passed: lhs.is_finite() && rhs.is_finite() && (lhs-rhs).abs() <= allowance,
            operands, hypotheses: "Actual native recorded canonical complete update; same preparation and realized innovations.".into() }
    }
    pub fn upper(
        id: &str,
        label: &str,
        lhs: f64,
        rhs: f64,
        allowance: f64,
        operands: serde_json::Value,
    ) -> Self {
        Self { id:id.into(), source_label:label.into(), lhs,rhs,allowance,
            relation:"upper bound with stated independent-replica uncertainty".into(),
            passed:lhs.is_finite() && rhs.is_finite() && lhs<=rhs+allowance,
            operands,hypotheses:"N-normalized observable; conditional preparations remain part of the evidence. No row independence is assumed for collision observables.".into() }
    }
}

/// Bounded full marked tests and alive subprobability tests. Physical alive
/// conditional tests divide these by their own alive mass outside this API.
pub fn bounded_observables(p: &Population<f64>) -> Result<[f64; 7]> {
    let x = p.observations.field("positions")?;
    let alive = p.eligible(false);
    let mut out = [0.; 7];
    for (i, row) in x.values().chunks_exact(x.width()).enumerate() {
        let s = row[0].sin();
        let c = row[0].cos();
        let r = row.iter().map(|v| v * v).sum::<f64>();
        let tail = r / (1. + r);
        let a = f64::from(alive[i]);
        for (o, v) in out.iter_mut().zip([s, c, tail, a, a * s, a * c, a * tail]) {
            *o += v / p.len() as f64;
        }
    }
    Ok(out)
}
pub fn root_observables(x: &[f64], alive: bool) -> [f64; 7] {
    let a = f64::from(alive);
    let s = x[0].sin();
    let c = x[0].cos();
    let r = x.iter().map(|v| v * v).sum::<f64>();
    let t = r / (1. + r);
    [s, c, t, a, a * s, a * c, a * t]
}

/// Configuration-only graph constants. The logarithmic presentation retains
/// valid bounds even when exp(2C) overflows binary64.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComponentBounds {
    pub kappa: f64,
    pub alive_fraction_lower: f64,
    pub edge_constant: f64,
    pub log_expected_component_upper: f64,
    pub radius_log_upper: Vec<f64>,
    pub minimum_population: usize,
}
pub fn component_bounds(
    diameter_squared: f64,
    width: f64,
    m: f64,
    rmax: usize,
) -> Result<ComponentBounds> {
    if !diameter_squared.is_finite()
        || diameter_squared < 0.
        || !width.is_finite()
        || width <= 0.
        || !m.is_finite()
        || m <= 0.
        || m > 1.
    {
        return Err(GasError::Configuration(
            "positive width and alive lower fraction required".into(),
        ));
    }
    let kappa = (-diameter_squared / (2. * width * width)).exp();
    let c = 2. / (kappa * m);
    if !c.is_finite() {
        return Err(GasError::Configuration("graph constant overflow".into()));
    }
    let mut log_factorial = 0.;
    let mut radius = vec![0.];
    for r in 1..=rmax {
        log_factorial += (r as f64).ln();
        radius.push(r as f64 * (2. * c).ln() - log_factorial);
    }
    Ok(ComponentBounds {
        kappa,
        alive_fraction_lower: m,
        edge_constant: c,
        log_expected_component_upper: 2. * c,
        radius_log_upper: radius,
        minimum_population: (2. / m).ceil() as usize,
    })
}

/// Explicit finite-horizon q-moment interface in the proof, with the exact
/// configured kernel lower bound. q here is a positive even integer so the
/// Gaussian norm moment is an exact finite product.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FiniteMomentBudget {
    pub order: usize,
    pub copying_coefficient: f64,
    pub coefficient: f64,
    pub additive: f64,
    pub gaussian_norm_moment: f64,
}
#[allow(clippy::too_many_arguments)]
pub fn finite_moment_budget(
    d: usize,
    q: usize,
    kappa: f64,
    h: f64,
    gamma: f64,
    lu: f64,
    bu: f64,
    cap: f64,
    alpha: f64,
    jitter: f64,
    ou_factor: f64,
    position_diffusion: f64,
) -> Result<FiniteMomentBudget> {
    if d == 0
        || q == 0
        || !q.is_multiple_of(2)
        || q > 32
        || ![
            kappa,
            h,
            gamma,
            lu,
            bu,
            cap,
            alpha,
            jitter,
            ou_factor,
            position_diffusion,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.)
        || kappa <= 0.
        || kappa > 1.
        || h <= 0.
        || cap <= 0.
        || alpha > 1.
    {
        return Err(GasError::Configuration(
            "invalid finite moment operands".into(),
        ));
    }
    let ch = (-gamma * h).exp();
    let sh = if gamma == 0. {
        h.sqrt()
    } else {
        (-(-2. * gamma * h).exp_m1() / (2. * gamma)).sqrt()
    };
    let drift = h * (1. + ch) / 2.;
    let eta = drift * h / 2.;
    let gaussian = (0..q / 2)
        .map(|j| d as f64 + 2. * j as f64)
        .product::<f64>();
    let copy = 1. + 2. / kappa;
    let a = 1. + eta * lu;
    let constant = drift * (1. + 2. * alpha) * cap + eta * bu;
    // X'=a Xcopy + a*jitter*GJ + constant + (h/2 sh b)*GO + sigma_x sqrt(h)*GX.
    let factor = 5_f64.powi(q as i32 - 1);
    let coefficient = factor * a.powi(q as i32) * copy;
    let additive = factor
        * ((a * jitter).powi(q as i32) * gaussian
            + constant.powi(q as i32)
            + (h / 2. * sh * ou_factor).powi(q as i32) * gaussian
            + (position_diffusion * h.sqrt()).powi(q as i32) * gaussian);
    Ok(FiniteMomentBudget {
        order: q,
        copying_coefficient: copy,
        coefficient,
        additive,
        gaussian_norm_moment: gaussian,
    })
}

/// Normal CDF evaluated by Simpson on the density for |x|<=12; the
/// fourth-derivative remainder is charged separately in terminal comparisons.
pub fn normal_cdf(x: f64) -> f64 {
    if x >= 12. {
        return 1.;
    }
    if x <= -12. {
        return 0.;
    }
    let n = 1024;
    let dx = x.abs() / n as f64;
    let f = |t: f64| (-t * t / 2.).exp() / (2. * std::f64::consts::PI).sqrt();
    let sum = f(0.)
        + f(x.abs())
        + (1..n)
            .map(|j| {
                if j % 2 == 0 {
                    2. * f(j as f64 * dx)
                } else {
                    4. * f(j as f64 * dx)
                }
            })
            .sum::<f64>();
    0.5 + x.signum() * sum * dx / 3.
}

pub fn alive_probability(center: &[f64], half_width: f64, sigma: f64) -> Result<f64> {
    if sigma <= 0.
        || !sigma.is_finite()
        || half_width <= 0.
        || !half_width.is_finite()
        || center.is_empty()
        || !center.iter().all(|v| v.is_finite())
    {
        return Err(GasError::Configuration(
            "positive final Gaussian and actual box required".into(),
        ));
    }
    Ok(center
        .iter()
        .map(|x| {
            (normal_cdf((half_width - x) / sigma) - normal_cdf((-half_width - x) / sigma))
                .clamp(0., 1.)
        })
        .product())
}

pub fn mean_variance(rows: &[[f64; 7]]) -> ([f64; 7], [f64; 7]) {
    let mut mean = [0.; 7];
    let mut var = [0.; 7];
    for row in rows {
        for j in 0..7 {
            mean[j] += row[j] / rows.len() as f64;
        }
    }
    if rows.len() > 1 {
        for row in rows {
            for j in 0..7 {
                var[j] += (row[j] - mean[j]).powi(2) / (rows.len() - 1) as f64;
            }
        }
    }
    (mean, var)
}

/// Conditional Haar cross-covariance experiment for an actual frozen component.
/// No graph resampling, average-fitness substitution, or independent per-row
/// rotations is introduced. Every component output shares each sampled matrix.
pub fn covariance_experiment(
    velocities: &[Vec<f64>],
    alpha: f64,
    seed: u64,
    draws: usize,
) -> Result<serde_json::Value> {
    use algorithmic_gas::random::{RandomStream, Stream};
    if velocities.len() < 2
        || draws < 2
        || velocities[0].is_empty()
        || velocities
            .iter()
            .any(|v| v.len() != velocities[0].len() || !v.iter().all(|x| x.is_finite()))
        || !(0. ..=1.).contains(&alpha)
    {
        return Err(GasError::Configuration(
            "actual nontrivial finite matched component required".into(),
        ));
    }
    let d = velocities[0].len();
    let m = velocities.len();
    let mut center = vec![0.; d];
    for v in velocities {
        for (c, x) in center.iter_mut().zip(v) {
            *c += x / m as f64;
        }
    }
    let centered = velocities
        .iter()
        .map(|v| {
            v.iter()
                .zip(&center)
                .map(|(x, c)| x - c)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut samples = vec![];
    let mut outputs = vec![];
    for draw in 0..draws {
        let rotation = RandomStream::new(seed, draw as u64, Stream::CollisionRotation, 0, 0)
            .haar_orthogonal(d)?;
        let out = centered
            .iter()
            .map(|u| {
                (0..d)
                    .map(|a| alpha * (0..d).map(|b| rotation[a * d + b] * u[b]).sum::<f64>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        samples.push(
            serde_json::json!({"draw":draw,"seed":seed,"rotation":rotation,"centered_output":out}),
        );
        outputs.push(out);
    }
    let mut checks = vec![];
    // All pairs are retained, including negative cross-covariance. Their total
    // sums vanish but each individual pair comparison remains informative.
    for i in 0..m {
        for j in 0..m {
            let dot = centered[i]
                .iter()
                .zip(&centered[j])
                .map(|(u, w)| u * w)
                .sum::<f64>();
            let ni = centered[i].iter().map(|u| u * u).sum::<f64>();
            let nj = centered[j].iter().map(|u| u * u).sum::<f64>();
            for a in 0..d {
                for b in 0..d {
                    let observed = outputs
                        .iter()
                        .map(|out| out[i][a] * out[j][b] / draws as f64)
                        .sum::<f64>();
                    let expected = if a == b {
                        alpha * alpha * dot / d as f64
                    } else {
                        0.
                    };
                    let allowance =
                        (alpha.powi(4) * ni * nj / draws as f64 / 0.000001).sqrt() + 1e-12;
                    checks.push(Comparison::upper(&format!("cross-{i}-{j}-{a}-{b}"),"thm-mean-field-component-identities",(observed-expected).abs(),0.,allowance,serde_json::json!({"observed":observed,"expected":expected,"i":i,"j":j,"a":a,"b":b,"alpha":alpha,"dimension":d,"draws":draws,"dot_centered":dot,"variance_upper":alpha.powi(4)*ni*nj,"failure_budget":0.000001})));
                }
            }
        }
    }
    for i in 0..m {
        for a in 0..d {
            let observed = outputs
                .iter()
                .map(|out| out[i][a] / draws as f64)
                .sum::<f64>();
            let variance =
                alpha * alpha * centered[i].iter().map(|u| u * u).sum::<f64>() / d as f64;
            checks.push(Comparison::upper(&format!("conditional-mean-{i}-{a}"),"thm-mean-field-component-identities",observed.abs(),0.,(variance/draws as f64/0.000001).sqrt()+1e-12,serde_json::json!({"observed_centered_mean":observed,"expected_centered_mean":0.,"conditional_variance":variance,"draws":draws,"failure_budget":0.000001})));
        }
    }
    Ok(
        serde_json::json!({"input_component_velocities":velocities,"center":center,"restitution":alpha,"samples":samples,"checks":checks,"scope":"Native addressed Haar API, conditioned on an actually recorded graph component; independent shared matrices across draws, complete per-pair covariance tests."}),
    )
}
