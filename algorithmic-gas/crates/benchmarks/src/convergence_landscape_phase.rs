//! Existing KUR regional landscape constants, with actual kernel parameters.
//! A source-variance multiplier does not certify a closed regional Markov law.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionalParameters {
    pub dimension: usize,
    pub timestep: f64,
    pub friction: f64,
    pub clone_jitter: f64,
    /// Actual OU standard deviation B sqrt((1-exp(-2 gamma h))/(2 gamma)).
    pub ou_amplitude: f64,
    /// Actual terminal position standard deviation sigma_x sqrt(h).
    pub position_amplitude: f64,
    pub velocity_cap: f64,
    /// Native restitution coefficient, constrained to the closed interval [0, 1].
    pub restitution: f64,
    pub viscosity: f64,
    pub bandwidth: f64,
    pub normalization: String,
    pub core_radius: f64,
    pub largest_well: usize,
}
fn invalid(s: &str) -> GasError {
    GasError::Configuration(s.into())
}
fn validate(p: &RegionalParameters) -> Result<()> {
    if p.dimension == 0
        || p.largest_well > 21
        || ![p.timestep, p.velocity_cap, p.bandwidth, p.core_radius]
            .iter()
            .all(|v| v.is_finite() && *v > 0.)
        || ![
            p.friction,
            p.clone_jitter,
            p.ou_amplitude,
            p.position_amplitude,
            p.viscosity,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.)
        || !p.restitution.is_finite()
        || !(0. ..=1.).contains(&p.restitution)
        || !["count", "row"].contains(&p.normalization.as_str())
    {
        return Err(invalid(
            "Invalid native regional parameters or viscous normalization",
        ));
    }
    Ok(())
}
/// Port of the existing Python rastrigin_regional_bound, preserving its floor.
pub fn regional_bound(p: &RegionalParameters) -> Result<Value> {
    validate(p)?;
    let m = 2. + 20. * 2_f64.sqrt() * std::f64::consts::PI.powi(2);
    regional_bound_radius(p, p.core_radius + 2. * p.largest_well as f64 / m)
}
fn regional_bound_radius(p: &RegionalParameters, radius: f64) -> Result<Value> {
    let pi = std::f64::consts::PI;
    let omega = 2. * pi;
    let m = 2. + 20. * 2_f64.sqrt() * pi * pi;
    let t = p.timestep / 2.;
    let mut failed = vec![];
    if radius >= 0.125 {
        failed.push("The enlarged root core is outside the proved curvature interval r_*<1/8");
    }
    if t * p.viscosity > 1. {
        failed.push("The actual first viscous matrix need not be stochastic: t*nu>1");
    }
    let c = (-p.friction * p.timestep).exp();
    let b = t * (1. + c);
    let eta = t * b;
    let ell = 1. - 2. * eta;
    let amplitude = 20. * pi * eta;
    let a = (-omega * omega * p.clone_jitter * p.clone_jitter / 2.).exp();
    let cos_min = (omega * radius).cos();
    let rho = (ell - amplitude * omega * a)
        .abs()
        .max((ell - amplitude * omega * a * cos_min).abs());
    if !(rho > 0. && rho < 1. && ell >= 0.) {
        failed.push("The displayed regional attraction criterion 0<rho_J<1 and ell>=0 fails");
    }
    if !failed.is_empty() {
        return Ok(
            json!({"parameters":p,"applicable":false,"failed_hypotheses":failed,"enlarged_core_radius":radius,"mean_map_squared":rho*rho,"positional_coefficient":null,"conservative_floor":null,"physical_multiplier_rate":null,"population_independent":true,"scope":"No positive multiplier is supplied outside the analytic regional premises."}),
        );
    }
    let kj = ell * ell * p.clone_jitter * p.clone_jitter
        - 2. * ell * amplitude * omega * p.clone_jitter * p.clone_jitter * a * cos_min
        + amplitude * amplitude / 2.
            * (1.
                - (-2. * omega * omega * p.clone_jitter * p.clone_jitter).exp()
                    * (2. * omega * radius).cos());
    let refresh = amplitude * (1. - a) * (omega * radius).sin();
    let coefficient = (1. + rho * rho) / 2.;
    let epsilon = (coefficient / (rho * rho)).sqrt() - 1.;
    let vc = (1. + 2. * p.restitution.abs()) * p.velocity_cap;
    let d = p.dimension as f64;
    let floor = (1. + epsilon) * ((1. + 1. / epsilon) * d * refresh * refresh + d * kj)
        + (1. + 1. / epsilon) * b * b * vc * vc
        + d * (t * t * p.ou_amplitude * p.ou_amplitude
            + p.position_amplitude * p.position_amplitude);
    if ![kj, refresh, coefficient, epsilon, floor]
        .iter()
        .all(|x| x.is_finite())
    {
        return Err(invalid("Regional arithmetic overflow"));
    }
    Ok(
        json!({"parameters":p,"applicable":true,"failed_hypotheses":[],"enlarged_core_radius":radius,"core_curvature_lower":m,"global_curvature_upper":2.+40.*pi*pi,"mean_map_squared":rho*rho,"accepted_coordinate_variance":kj,"gate_refresh_bound":refresh,"positional_coefficient":coefficient,"epsilon":epsilon,"conservative_floor":floor,"physical_multiplier_rate":-coefficient.ln()/p.timestep,"formal_iteration_floor":floor/(1.-coefficient),"population_independent":true,"source_labels":["thm-kur-evaluated-regional-bound"],"source_formula":"E W_N(X+) <= lambda E W_N(mu) + C_reg","scope":"One-step complete positional source-variance bound with all eligible frozen sources in one declared root core; full Gaussian jitter/OU/position tails retained. A rate of this multiplier can be iterated only with proved source-pressure/residence/phase-flux accounting. It is not a coupled-swarm or conditioned closed-core convergence rate.","arithmetic":"Binary64 evaluation of the existing analytic formulas; independent Python fixture comparison, not a numerical interval certificate."}),
    )
}
/// Exact KUL.8 variance envelope and analytic root-displacement bootstrap.
/// Keeps the original conservative Python-port result available separately.
pub fn tightened_regional_bound(p: &RegionalParameters) -> Result<Value> {
    validate(p)?;
    let pi = std::f64::consts::PI;
    let mut displacement = 2. * p.largest_well as f64 / (2. + 20. * 2_f64.sqrt() * pi * pi);
    let mut history = vec![displacement];
    if displacement <= 0.125 {
        for _ in 0..32 {
            let curvature = 2. + 40. * pi * pi * (std::f64::consts::TAU * displacement).cos();
            displacement = displacement.min(2. * p.largest_well as f64 / curvature);
            history.push(displacement);
        }
    }
    let radius = p.core_radius + displacement;
    let mut out = regional_bound_radius(p, radius)?;
    out["analytic_root_displacement_history"] = json!(history);
    out["geometry_iterations"] = json!(32);
    out["tightening_scope"] = json!(
        "Every root-displacement update uses the analytic Hessian lower bound along the integer-to-root segment; sampled root locations are not used. Exact KUL.8 accepted variance is bounded before first graph weights, which remain handled by pathwise stochasticity and Young's inequality."
    );
    if out["applicable"] != true {
        return Ok(out);
    }
    let omega = std::f64::consts::TAU;
    let t = p.timestep / 2.;
    let c = (-p.friction * p.timestep).exp();
    let b = t * (1. + c);
    let eta = t * b;
    let ell = 1. - 2. * eta;
    let amplitude = 20. * pi * eta;
    let variance = p.clone_jitter * p.clone_jitter;
    let a = (-omega * omega * variance / 2.).exp();
    let fourth = a.powi(4);
    let cos_min = (omega * radius).cos();
    let psi = |z: f64| {
        ell * ell * variance - 2. * ell * amplitude * omega * variance * a * z
            + amplitude * amplitude * ((1. + fourth) / 2. - a * a + (a * a - fourth) * z * z)
    };
    let endpoints = [psi(cos_min), psi(1.)];
    let kj = endpoints[0].max(endpoints[1]).max(0.);
    let refresh = out["gate_refresh_bound"].as_f64().unwrap();
    let eps = out["epsilon"].as_f64().unwrap();
    let coeff = out["positional_coefficient"].as_f64().unwrap();
    let vc = (1. + 2. * p.restitution.abs()) * p.velocity_cap;
    let d = p.dimension as f64;
    let floor = (1. + eps) * ((1. + 1. / eps) * d * refresh * refresh + d * kj)
        + (1. + 1. / eps) * b * b * vc * vc
        + d * (t * t * p.ou_amplitude * p.ou_amplitude
            + p.position_amplitude * p.position_amplitude);
    out["accepted_coordinate_variance"] = json!(kj);
    out["exact_variance_endpoint_values"] = json!(endpoints);
    out["conservative_floor"] = json!(floor);
    out["formal_iteration_floor"] = json!(floor / (1. - coeff));
    out["source_labels"] = json!([
        "cor-slc-exact-jitter-regional-refinement",
        "lem-kul-jitter-periodic-profile",
        "thm-kur-evaluated-regional-bound"
    ]);
    Ok(out)
}
fn gradient(x: f64) -> f64 {
    2. * x + 20. * std::f64::consts::PI * (std::f64::consts::TAU * x).sin()
}
fn root(mut lo: f64, mut hi: f64) -> Result<Value> {
    let mut fl = gradient(lo);
    let fh = gradient(hi);
    if fl.signum() == fh.signum() {
        return Err(invalid("Root bracket has no sign change"));
    }
    for _ in 0..80 {
        let mid = lo + (hi - lo) / 2.;
        if mid == lo || mid == hi {
            break;
        }
        let fm = gradient(mid);
        if fm == 0. {
            lo = mid;
            hi = mid;
            break;
        }
        if fm.signum() == fl.signum() {
            lo = mid;
            fl = fm;
        } else {
            hi = mid;
        }
    }
    Ok(
        json!({"value":lo+(hi-lo)/2.,"numeric_bracket":[lo,hi],"arithmetic":"Binary64 bisection diagnostic inside analytic integer/half-integer source brackets; not interval root certification."}),
    )
}
pub fn rastrigin_regional_profile(first: i32, last: i32) -> Result<Value> {
    if first > last || first < -21 || last > 21 {
        return Err(invalid("Declared integer wells must be within[-21,21]"));
    }
    let stable = (first..=last)
        .map(|k| root(k as f64 - 0.125, k as f64 + 0.125))
        .collect::<Result<Vec<_>>>()?;
    let barriers = (first..last)
        .map(|k| root(k as f64 + 0.375, k as f64 + 0.625))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"integer_centers":(first..=last).collect::<Vec<_>>(),"stable_roots":stable,"barriers":barriers,"core_curvature_lower":2.+20.*2_f64.sqrt()*std::f64::consts::PI.powi(2),"global_curvature_upper":2.+40.*std::f64::consts::PI.powi(2),"force":"F=-2x-20pi sin(2pi x)","source_labels":["lem-klq-rastrigin-profiles"],"scope":"Distinct stable-root cores, intervening unstable barriers and exterior remain separate phases."}),
    )
}
fn get(v: &Value, path: &str) -> Result<f64> {
    v.pointer(path)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(&format!("Missing finite native parameter {path}")))
}
fn isotropic_amplitude(v: &Value) -> Result<f64> {
    if v["innovation"] != "gaussian"
        || v["geometry"]["kind"] != "isotropic"
        || v["geometry"]["scale"]["kind"] != "constant"
    {
        return Err(invalid(
            "Regional Gaussian formulas require actual constant isotropic Gaussian innovations",
        ));
    }
    let values = v["geometry"]["scale"]["values"]
        .as_array()
        .ok_or_else(|| invalid("Missing native noise amplitude"))?;
    let a = values
        .first()
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("Empty native noise amplitude"))?;
    if values.iter().any(|x| x.as_f64() != Some(a)) || !a.is_finite() || a < 0. {
        return Err(invalid("Nonuniform/invalid native noise amplitude"));
    }
    Ok(a)
}
/// Native archived GasConfig bridge. No defaults replace missing physical fields.
pub fn native_parameters(c: &Value, dimension: usize) -> Result<RegionalParameters> {
    if c["kinetic"]["integrator"]["kind"] != "baoab"
        || c["kinetic"]["boundary_schedule"] != "end_of_step"
        || !c["qft"]["graph_viscosity"].is_null()
        || !c["qft"]["curl"].is_null()
        || c["qft"]["innovation_shifts"]
            .as_array()
            .is_none_or(|x| !x.is_empty())
    {
        return Err(invalid(
            "Unsupported actual native operator: require unshifted BAOAB, no graph/curl and terminal classification",
        ));
    }
    let h = get(c, "/kinetic/integrator/dt")?;
    let gamma = get(c, "/kinetic/integrator/friction")?;
    if h <= 0. || gamma < 0. {
        return Err(invalid("Invalid native h/gamma"));
    }
    let b = isotropic_amplitude(&c["kinetic"]["noise"])?;
    let jitter = get(c, "/clone_transform/jitter_amplitude")?;
    let jitter_scale = if jitter == 0. {
        0.
    } else {
        isotropic_amplitude(&c["clone_transform"]["jitter"])?
    };
    let visc = &c["qft"]["viscosity"];
    let (nu, bandwidth, mode) = if visc.is_null() {
        (0., 1., "count".to_string())
    } else {
        (
            get(visc, "/coefficient")?,
            get(visc, "/bandwidth")?,
            if visc["row_normalized"]
                .as_bool()
                .ok_or_else(|| invalid("Missing native viscous normalization"))?
            {
                "row".to_string()
            } else {
                "count".to_string()
            },
        )
    };
    let p = RegionalParameters {
        dimension,
        timestep: h,
        friction: gamma,
        clone_jitter: jitter * jitter_scale,
        ou_amplitude: b
            * (if gamma == 0. {
                h
            } else {
                -(-2. * gamma * h).exp_m1() / (2. * gamma)
            })
            .sqrt(),
        position_amplitude: get(c, "/kinetic/position_diffusion")? * h.sqrt(),
        velocity_cap: get(c, "/kinetic/velocity_cap")?,
        restitution: get(c, "/clone_transform/restitution")?,
        viscosity: nu,
        bandwidth,
        normalization: mode,
        core_radius: 1. / 16.,
        largest_well: 2,
    };
    validate(&p)?;
    Ok(p)
}
/// Frozen eligible source inputs, not recipient post-jitter or displayed row IDs.
pub fn source_core_eligibility(
    positions: &[f64],
    dimension: usize,
    p: &RegionalParameters,
) -> Result<Value> {
    validate(p)?;
    if dimension != p.dimension
        || positions.is_empty()
        || !positions.len().is_multiple_of(dimension)
        || positions.iter().any(|x| !x.is_finite())
    {
        return Err(invalid(
            "Finite source positions must have complete [N,d] shape",
        ));
    }
    let profile = rastrigin_regional_profile(-(p.largest_well as i32), p.largest_well as i32)?;
    let roots = profile["stable_roots"].as_array().unwrap();
    let values: Vec<_> = roots.iter().map(|x| x["value"].as_f64().unwrap()).collect();
    let mut max = 0_f64;
    let mut labels = vec![];
    let mut eligible = 0;
    for row in positions.chunks(dimension) {
        let mut label = vec![];
        let mut inside = true;
        for x in row {
            let (index, distance) = values
                .iter()
                .enumerate()
                .map(|(i, r)| (i, (x - r).abs()))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            max = max.max(distance);
            inside &= distance <= p.core_radius;
            label.push(index as i32 - p.largest_well as i32);
        }
        if inside {
            eligible += 1;
        }
        labels.push((label, inside));
    }
    let same_core = labels.iter().all(|x| x.0 == labels[0].0);
    Ok(
        json!({"N":positions.len()/dimension,"eligible_source_rows":eligible,"all_in_declared_cores":eligible==labels.len(),"all_sources_in_one_core":eligible==labels.len() && same_core,"maximum_distance_to_nearest_stable_root":max,"row_core_observations":labels,"scope":"Storage-row phase observations only; no algorithm labels, and no residence or source-integral conditioning claim is inferred."}),
    )
}
