//! Exact normalized moment estimates for Chapter 5, with explicit transfer gates.
//! Native fixed-cap output estimates and continuum generator evaluations differ.
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};

fn invalid(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn nonnegative(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_finite() && *x >= 0.)
}
fn mean(x: &[f64], d: usize) -> Result<Vec<f64>> {
    if d == 0 || x.is_empty() || !x.len().is_multiple_of(d) || x.iter().any(|x| !x.is_finite()) {
        return Err(invalid("Expected a finite nonempty swarm of dimension d"));
    }
    let n = x.len() / d;
    Ok((0..d)
        .map(|j| x.chunks(d).map(|r| r[j]).sum::<f64>() / n as f64)
        .collect())
}
pub fn second_moments(x: &[f64], d: usize) -> Result<Value> {
    let m = mean(x, d)?;
    let n = x.len() / d;
    let total = x.iter().map(|x| x * x).sum::<f64>() / n as f64;
    let barycenter = m.iter().map(|x| x * x).sum::<f64>();
    let variance = x
        .chunks(d)
        .map(|r| r.iter().zip(&m).map(|(x, m)| (x - m).powi(2)).sum::<f64>())
        .sum::<f64>()
        / n as f64;
    Ok(json!({"N":n,"d":d,"total":total,"barycenter":barycenter,"variance":variance,"mean":m}))
}
/// Actual transient envelope from a certified continuum generator drift.
pub fn transient_velocity_envelope(initial: f64, rho: f64, source: f64) -> Result<f64> {
    if !nonnegative(&[initial, source]) || !rho.is_finite() || rho <= 0. {
        return Err(invalid(
            "Transient envelope requires rho>0 and finite nonnegative moments",
        ));
    }
    Ok(initial.max(source / rho))
}
/// Exact finite-horizon Jensen estimate; no OU autocovariance is assumed.
pub fn finite_horizon_position_bound(mx: f64, mv: f64, h: f64, horizon: f64) -> Result<Value> {
    if !nonnegative(&[mx, mv, h, horizon]) || horizon <= 0. || h > horizon {
        return Err(invalid("Positive horizon and 0<=h<=H required"));
    }
    let c1 = 2. * (mx * mv).sqrt();
    let c2 = horizon * mv;
    Ok(
        json!({"C1":c1,"C2":c2,"C_kin_x":c1+c2,"exact_increment_bound":c1*h+mv*h*h,"horizon_increment_bound":(c1+c2)*h,"population_independent":true,"hypotheses":"Actual transient E Var_x(0)<=Mx and sup_[0,H] E Var_v<=Mv, fixed averaging set, dx=v dt; no position diffusion/status changes."}),
    )
}
/// Exact native conditional moments after the actual first kick, before OU noise.
pub fn native_position_moments(
    x: &[f64],
    v1: &[f64],
    d: usize,
    h: f64,
    gamma: f64,
    q: f64,
    s: f64,
) -> Result<Value> {
    if x.len() != v1.len() || !nonnegative(&[h, gamma, q, s]) || h <= 0. {
        return Err(invalid("Invalid native stage/noise parameters"));
    }
    let input = second_moments(x, d)?;
    let velocity = second_moments(v1, d)?;
    let n = x.len() / d;
    let c = h / 2.;
    let b = c * (1. + (-gamma * h).exp());
    let m: Vec<_> = x.iter().zip(v1).map(|(x, v)| x + b * v).collect();
    let deterministic = second_moments(&m, d)?;
    let tau2 = c * c * q * q + s * s;
    let trace = d as f64 * tau2;
    let old = input["variance"].as_f64().unwrap();
    let vv = velocity["variance"].as_f64().unwrap();
    let centered = deterministic["variance"].as_f64().unwrap() + (1. - 1. / n as f64) * trace;
    let increment = 2. * b * (old * vv).sqrt() + b * b * vv + trace;
    Ok(
        json!({"N":n,"d":d,"mean_positions":m,"b":b,"tau_squared":tau2,"input":input,"first_kick_velocity":velocity,"deterministic":deterministic,
        "variance_prediction":centered,"total_prediction":deterministic["total"].as_f64().unwrap()+trace,"barycenter_prediction":deterministic["barycenter"].as_f64().unwrap()+trace/n as f64,
        "conditional_increment_bound":increment,"conditional_increment":centered-old,"variance_noise_source":(1.-1./n as f64)*trace,"uniform_variance_noise_source":trace,
        "hypotheses":"All prepared rows included, independent constant isotropic OU and position innovations, no intermediate position projection, physical dead coordinates retained; actual deterministic first kick including viscosity."}),
    )
}
/// Generator evaluated at a retained state for the stated *continuum* extension.
/// Force-square envelopes are actual row averages, not a bounded-domain proxy.
#[allow(clippy::too_many_arguments)]
pub fn component_generators(
    x: &[f64],
    v: &[f64],
    force: &[f64],
    d: usize,
    gamma: f64,
    sigma: f64,
    epsilon: f64,
    fixed_cap: bool,
) -> Result<Value> {
    if x.len() != v.len()
        || v.len() != force.len()
        || !nonnegative(&[gamma, sigma])
        || gamma <= 0.
        || !epsilon.is_finite()
        || epsilon <= 0.
        || epsilon >= 2. * gamma
    {
        return Err(invalid(
            "Generator requires finite aligned states and 0<epsilon<2 gamma",
        ));
    }
    let xm = second_moments(x, d)?;
    let vm = second_moments(v, d)?;
    let fm = second_moments(force, d)?;
    let n = x.len() / d;
    let xbar: Vec<f64> =
        serde_json::from_value(xm["mean"].clone()).map_err(|_| invalid("Mean decode"))?;
    let vbar: Vec<f64> =
        serde_json::from_value(vm["mean"].clone()).map_err(|_| invalid("Mean decode"))?;
    let fbar: Vec<f64> =
        serde_json::from_value(fm["mean"].clone()).map_err(|_| invalid("Mean decode"))?;
    let vf = v
        .chunks(d)
        .zip(force.chunks(d))
        .map(|(v, f)| {
            v.iter()
                .zip(&vbar)
                .zip(f)
                .map(|((v, m), f)| (v - m) * f)
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64;
    let xv = x
        .chunks(d)
        .zip(v.chunks(d))
        .map(|(x, v)| {
            x.iter()
                .zip(&xbar)
                .zip(v.iter().zip(&vbar))
                .map(|((x, m), (v, vm))| (x - m) * (v - vm))
                .sum::<f64>()
        })
        .sum::<f64>()
        / n as f64;
    let bv = vm["barycenter"].as_f64().unwrap();
    let vv = vm["variance"].as_f64().unwrap();
    let noise = d as f64 * sigma * sigma;
    let generator = 2. * vf - 2. * gamma * vv + (1. - 1. / n as f64) * noise;
    let bound = -(2. * gamma - epsilon) * vv + fm["total"].as_f64().unwrap() / epsilon + noise;
    let bary = 2. * vbar.iter().zip(&fbar).map(|(v, f)| v * f).sum::<f64>() - 2. * gamma * bv
        + noise / n as f64;
    let bary_bound = -gamma * bv + fm["total"].as_f64().unwrap() / gamma + noise / n as f64;
    Ok(
        json!({"velocity_generator":generator,"velocity_bound":bound,"velocity_rho":2.*gamma-epsilon,"force_square":fm["total"],"barycenter_generator":bary,"barycenter_bound":bary_bound,"position_generator":2.*xv,"position_bound":2.*(xm["variance"].as_f64().unwrap()*vv).sqrt(),"force_work":2.*vf,"independent_noise_source":(1.-1./n as f64)*noise,
        "native_continuum_transfer_applicable":false,"native_fixed_cap_obstruction":fixed_cap,"native_transfer_reason":if fixed_cap {"Repeated fixed radial cap is not a generator-consistent approximation"} else {"Uncapped extension: observable-specific weak error and timestep hypotheses still required"},"scope":"Pointwise algebraic evaluation of an independent-noise fixed-set continuum generator on the retained physical state; it is not an estimate of native finite-step drift."}),
    )
}

/// Explicit row minorization class; no interacting force or implicit truncation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactKineticParameters {
    pub dimension: usize,
    pub timestep: f64,
    pub friction: f64,
    pub ou_standard_deviation: f64,
    pub position_standard_deviation: f64,
    pub velocity_cap: f64,
    pub force_lipschitz: f64,
    pub force_at_origin: f64,
    pub input_position_radius: f64,
    pub input_velocity_radius: f64,
    pub target_position_radius: f64,
    pub target_velocity_radius: f64,
    pub target_center_norm: f64,
}
fn log_ball_volume(d: usize, r: f64) -> f64 {
    let gamma = if d.is_multiple_of(2) {
        (1..=d / 2).map(|j| (j as f64).ln()).sum::<f64>()
    } else {
        std::f64::consts::PI.ln() / 2. - 2_f64.ln()
            + (1..=d / 2).map(|j| (j as f64 + 0.5).ln()).sum::<f64>()
    };
    d as f64 / 2. * std::f64::consts::PI.ln() + d as f64 * r.ln() - gamma
}
pub fn compact_kinetic_minorization(p: &CompactKineticParameters) -> Result<Value> {
    if p.dimension == 0
        || p.dimension > 256
        || !nonnegative(&[
            p.friction,
            p.force_lipschitz,
            p.force_at_origin,
            p.input_position_radius,
            p.input_velocity_radius,
            p.target_center_norm,
        ])
        || ![
            p.timestep,
            p.ou_standard_deviation,
            p.position_standard_deviation,
            p.velocity_cap,
            p.target_position_radius,
            p.target_velocity_radius,
        ]
        .iter()
        .all(|x| x.is_finite() && *x > 0.)
        || p.target_velocity_radius >= p.velocity_cap
    {
        return Err(invalid(
            "Compact kinetic density requires q,s>0 and 0<target velocity radius<V",
        ));
    }
    let c = p.timestep / 2.;
    let a = (-p.friction * p.timestep).exp();
    let ell = c * c * p.force_lipschitz;
    if ell >= 1. {
        return Err(invalid("Compact density inverse requires c² LF<1"));
    }
    let av = p.input_velocity_radius
        + c * (p.force_at_origin + p.force_lipschitz * p.input_position_radius);
    let ax = p.input_position_radius + c * av;
    let b3 =
        p.velocity_cap * p.target_velocity_radius / (p.velocity_cap - p.target_velocity_radius);
    let w = (b3 + c * (p.force_at_origin + p.force_lipschitz * ax)) / (1. - ell);
    let x2 = ax + c * w;
    let q = p.ou_standard_deviation;
    let s = p.position_standard_deviation;
    let log_density = -(p.dimension as f64) * (std::f64::consts::TAU * q * s).ln()
        - (w + a * av).powi(2) / (2. * q * q)
        - (p.target_center_norm + p.target_position_radius + x2).powi(2) / (2. * s * s)
        - p.dimension as f64 * ell.ln_1p();
    let log_epsilon = log_density
        + log_ball_volume(p.dimension, p.target_position_radius)
        + log_ball_volume(p.dimension, p.target_velocity_radius);
    if !log_epsilon.is_finite() || log_epsilon > 0. {
        return Err(invalid(
            "Nonfinite or invalid compact probability calculation",
        ));
    }
    Ok(
        json!({"parameters":p,"A_v":av,"A_x":ax,"B3":b3,"W":w,"X2":x2,"ell":ell,"log_density_lower":log_density,"log_epsilon":log_epsilon,"epsilon_representable":if log_epsilon>=f64::MIN_POSITIVE.ln(){Some(log_epsilon.exp())}else{None},"N_independent_row":true,"whole_swarm_log_coefficient":"N * log_epsilon; not an N-uniform law coefficient","hypotheses":"Declared bounded prepared input class, global C1 force bounds, no row-coupling viscosity/graph force, two independent positive Gaussian amplitudes, target position ball inside valid interior; binary64 formula evaluation, not a directed numerical interval certificate."}),
    )
}
/// Exact density for uncoupled native isotropic harmonic rows, including cap Jacobian.
#[allow(clippy::too_many_arguments)]
pub fn quadratic_output_log_density(
    x: &[f64],
    v: &[f64],
    out_x: &[f64],
    out_v: &[f64],
    h: f64,
    gamma: f64,
    omega: f64,
    q: f64,
    s: f64,
    cap: f64,
) -> Result<f64> {
    let d = x.len();
    if d == 0
        || [v.len(), out_x.len(), out_v.len()].iter().any(|n| *n != d)
        || x.iter()
            .chain(v)
            .chain(out_x)
            .chain(out_v)
            .any(|x| !x.is_finite())
        || ![h, q, s, cap, omega]
            .iter()
            .all(|x| x.is_finite() && *x > 0.)
        || !gamma.is_finite()
        || gamma < 0.
    {
        return Err(invalid("Invalid harmonic row density inputs"));
    }
    let c = h / 2.;
    let a = (-gamma * h).exp();
    let k = 1. - omega * c * c;
    let r = out_v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if k <= 0. || r >= cap {
        return Err(invalid("Degenerate second-kick map or outside cap image"));
    }
    let mut qerror = 0.;
    let mut serror = 0.;
    for j in 0..d {
        let v1 = v[j] - omega * c * x[j];
        let x1 = x[j] + c * v1;
        let precap = cap * out_v[j] / (cap - r);
        let z = (precap + omega * c * x1) / k;
        qerror += (z - a * v1).powi(2);
        serror += (out_x[j] - x1 - c * z).powi(2);
    }
    let log = -(d as f64) * (std::f64::consts::TAU * q * s).ln()
        - qerror / (2. * q * q)
        - serror / (2. * s * s)
        - d as f64 * k.ln()
        + (d + 1) as f64 * (cap / (cap - r)).ln();
    if !log.is_finite() {
        return Err(invalid("Harmonic density arithmetic overflow"));
    }
    Ok(log)
}
