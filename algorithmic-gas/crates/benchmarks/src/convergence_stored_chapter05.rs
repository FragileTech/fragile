//! Reanalysis of stored native kinetic experiments. Never advances an engine.
use crate::convergence_decay::linear_decay_constants;
use crate::convergence_kinetic::KineticValidationConfig;
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};

const PATH: &str = "docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md";
const SOURCE: &str = include_str!(
    "../../../docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md"
);

fn formula(marker: &str) -> &'static str {
    SOURCE
        .split("$$")
        .enumerate()
        .find_map(|(i, body)| (i % 2 == 1 && body.contains(marker)).then_some(body.trim()))
        .expect("source equation exists")
}

fn configured_cap(v: &[f64], radius: f64) -> Vec<f64> {
    let scale = v.iter().fold(0f64, |m, x| m.max(x.abs()));
    let factor = if scale == 0. {
        1.
    } else {
        let norm_scaled = v.iter().map(|x| (x / scale).powi(2)).sum::<f64>().sqrt();
        if scale <= radius {
            1. / (1. + scale / radius * norm_scaled)
        } else {
            (radius / scale) / (radius / scale + norm_scaled)
        }
    };
    v.iter().map(|x| x * factor).collect()
}

fn number(v: &Value, p: &str) -> Result<f64> {
    v.pointer(p)
        .and_then(Value::as_f64)
        .filter(|x| x.is_finite())
        .ok_or_else(|| GasError::Configuration(format!("Chapter 5 stored record lacks finite {p}")))
}
fn rows<'a>(v: &'a Value, p: &str) -> &'a [Value] {
    v.pointer(p)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}
#[allow(clippy::too_many_arguments)]
fn add(
    out: &mut Vec<Value>,
    id: String,
    label: &str,
    quote: Option<&str>,
    observed: f64,
    bound: f64,
    se: f64,
    relation: &str,
    report: &str,
    scope: &str,
    hypotheses: Value,
) {
    let tolerance = 2e-10 * (1. + observed.abs() + bound.abs());
    let residual = observed - bound;
    let passed = observed.is_finite()
        && bound.is_finite()
        && se.is_finite()
        && se >= 0.
        && if relation == "equal" {
            residual.abs() <= 6. * se + tolerance
        } else {
            residual <= 6. * se + tolerance
        };
    out.push(
        json!({"id":id,"source_labels":[label],"source_formula":quote,
        "observed":observed,"bound":bound,"relation":relation,"passed":passed,
        "status":if passed {"not_rejected"}else{"rejected"},"standard_error":se,
        "residual":residual,"scope":scope,"report_path":report,"hypotheses":hypotheses}),
    );
}
#[allow(clippy::too_many_arguments)]
fn exact(
    out: &mut Vec<Value>,
    id: String,
    label: &str,
    quote: Option<&str>,
    observed: f64,
    bound: f64,
    report: &str,
    h: Value,
) {
    add(
        out,
        id,
        label,
        quote,
        observed,
        bound,
        0.,
        "equal",
        report,
        "Independent reconstruction from native configuration; no new trajectories",
        h,
    );
}
fn estimate(v: &Value) -> Result<(f64, f64)> {
    Ok((number(v, "/mean")?, number(v, "/standard_error")?))
}

/// Exact variance for a summed Gaussian quadratic cost with common covariance.
/// `mean_gram` is the sum of coordinate-wise mean outer products; it need not
/// represent identically oriented or labelled walkers.
pub fn gaussian_quadratic_cost_variance(
    p: [[f64; 2]; 2],
    covariance: [[f64; 2]; 2],
    mean_gram: [[f64; 2]; 2],
    dimensions: usize,
) -> f64 {
    let mut pc = [[0.; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            pc[i][j] = (0..2).map(|k| p[i][k] * covariance[k][j]).sum();
        }
    }
    let central =
        2. * dimensions as f64 * (pc[0][0].powi(2) + 2. * pc[0][1] * pc[1][0] + pc[1][1].powi(2));
    let noncentral = 4.
        * (0..2)
            .flat_map(|i| {
                (0..2).flat_map(move |j| (0..2).map(move |k| pc[i][k] * p[k][j] * mean_gram[j][i]))
            })
            .sum::<f64>();
    central + noncentral
}

/// Finite-step quadratic cap constants, independent of N and d.
pub fn quadratic_cap_constants(h: f64, gamma: f64, diffusion: f64, cap: f64) -> Result<Value> {
    if ![h, gamma, diffusion, cap]
        .iter()
        .all(|x| x.is_finite() && *x > 0.)
        || h >= 2.
    {
        return Err(GasError::Configuration(
            "Quadratic cap coupling needs 0<h<2 and positive friction, noise and cap".into(),
        ));
    }
    let c = h / 2.;
    let k = 1. - c * c;
    let a = (-gamma * h).exp();
    let q = diffusion * (-(-2. * gamma * h).exp_m1() / (2. * gamma)).sqrt();
    let p0 = (2. / std::f64::consts::PI).sqrt() * (-2f64).exp();
    let eta = p0 * (1. - (cap / (cap + k * q)).powi(2));
    let trace = 1. - a * a + eta * (a * a + c * c * (1. - a * a));
    let determinant = eta * c * c * (1. - a * a);
    Ok(
        json!({"c":c,"k":k,"a":a,"q":q,"p0":p0,"eta":eta,"T":trace,"D":determinant,"delta":determinant/trace}),
    )
}

fn kinetic(out: &mut Vec<Value>, report: &str, record: &Value, index: usize) -> Result<()> {
    let cfg: KineticValidationConfig = serde_json::from_value(record["config"].clone())
        .map_err(|e| GasError::Configuration(e.to_string()))?;
    cfg.validate()?;
    let h = cfg.dt;
    let c = h / 2.;
    let a = (-cfg.friction * h).exp();
    let q2 = cfg.velocity_diffusion.powi(2)
        * if cfg.friction == 0. {
            h
        } else {
            -(-2. * cfg.friction * h).exp_m1() / (2. * cfg.friction)
        };
    let assumptions = cfg.landscape.assumptions()?;
    let lf = assumptions.force_lipschitz;
    let hy = json!({"fixed_input_repetitions":cfg.samples,"sample_unit":"independent fixed-input native Gaussian experiment, not interacting walkers","native_config":cfg,"L_F":lf,"L_Sigma":0.,"sigma_min_squared":cfg.velocity_diffusion.powi(2),"sigma_max_squared":cfg.velocity_diffusion.powi(2),"stratonovich_correction":0.});
    let min_curvature = cfg
        .landscape
        .curvature
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_curvature = cfg.landscape.curvature.iter().copied().fold(0., f64::max);
    let force_offset = cfg.landscape.ripple_amplitude * cfg.landscape.ripple_frequency;
    let center_sq = cfg.landscape.center.iter().map(|x| x * x).sum::<f64>();
    let alpha = min_curvature / 2.;
    let r_u = max_curvature.powi(2) * center_sq / min_curvature
        + cfg.dimensions as f64 * force_offset.powi(2) / min_curvature;
    exact(
        out,
        format!("kinetic{index}/analytic_L_F"),
        "axiom-confining-potential",
        None,
        number(record, "/landscape_assumptions/force_lipschitz")?,
        lf,
        report,
        hy.clone(),
    );
    for (j, input) in cfg.inputs.iter().enumerate() {
        let capped = configured_cap(&input.velocities, cfg.velocity_cap);
        let input_norm = input.velocities.iter().map(|v| v * v).sum::<f64>().sqrt();
        let capped_norm = capped.iter().map(|v| v * v).sum::<f64>().sqrt();
        let predicted_norm = cfg.velocity_cap * input_norm / (cfg.velocity_cap + input_norm);
        let map_residual = capped
            .iter()
            .zip(&input.velocities)
            .map(|(actual, v)| {
                (actual - cfg.velocity_cap * v / (cfg.velocity_cap + input_norm)).abs()
            })
            .sum::<f64>();
        exact(
            out,
            format!("kinetic{index}/input{j}/configured_radial_cap"),
            "axiom-friction-timestep",
            Some(formula(r"S(v)=\frac{v_{\max}v}")),
            map_residual
                + (capped_norm - predicted_norm).abs()
                + (capped_norm - cfg.velocity_cap).max(0.),
            0.,
            report,
            json!({"retained_velocity":input.velocities,"native_cap_radius":cfg.velocity_cap,
                   "scope":"Stable scaled-norm cap algorithm from native kinetic.rs independently compared to analytic radial formula on retained inputs; native endpoint reconstruction residual is checked separately"}),
        );
        if input_norm > 0. {
            add(
                out,
                format!("kinetic{index}/input{j}/strict_cap_shrinkage"),
                "axiom-friction-timestep",
                None,
                capped_norm,
                input_norm,
                0.,
                "upper",
                report,
                "Nonzero retained physical input; cap changes every nonzero velocity, including arbitrarily small inputs",
                hy.clone(),
            );
        }
        for (jj, other) in cfg.inputs.iter().enumerate() {
            let other_cap = configured_cap(&other.velocities, cfg.velocity_cap);
            let observed = capped
                .iter()
                .zip(other_cap)
                .map(|(v, w)| (v - w).powi(2))
                .sum::<f64>()
                .sqrt();
            let bound = input
                .velocities
                .iter()
                .zip(&other.velocities)
                .map(|(v, w)| (v - w).powi(2))
                .sum::<f64>()
                .sqrt();
            add(
                out,
                format!("kinetic{index}/input{j}_{jj}/cap_nonexpansion"),
                "axiom-friction-timestep",
                None,
                observed,
                bound,
                0.,
                "upper",
                report,
                "Stable native cap algebra on pairs of retained inputs; physical coordinates, no walker labels",
                hy.clone(),
            );
        }
        let g = cfg.landscape.analytic_gradient(&input.positions)?;
        let dot = input
            .positions
            .iter()
            .zip(&g)
            .map(|(x, g)| x * g)
            .sum::<f64>();
        let x_sq = input.positions.iter().map(|x| x * x).sum::<f64>();
        add(
            out,
            format!("kinetic{index}/input{j}/coercivity"),
            "axiom-confining-potential",
            Some(formula(r"\langle x, \nabla U(x) \rangle \geq")),
            alpha * x_sq - r_u,
            dot,
            0.,
            "upper",
            report,
            "Global analytic quadratic-plus-cosine coercivity constants alpha=min(k)/2 and R=max(k)^2|center|²/min(k)+d(aw)²/min(k), checked against retained input positions; follows by two Young bounds",
            json!({"alpha_U":alpha,"R_U":r_u,"native_config":cfg}),
        );
        if let Some(radius) = cfg.box_half_width
            && input.positions.iter().all(|x| x.abs() <= radius)
        {
            let fmax = cfg
                .landscape
                .curvature
                .iter()
                .zip(&cfg.landscape.center)
                .map(|(k, center)| (k * (radius + center.abs()) + force_offset).powi(2))
                .sum::<f64>()
                .sqrt();
            add(
                out,
                format!("kinetic{index}/input{j}/valid_domain_Fmax"),
                "axiom-confining-potential",
                Some(formula(r"\|F(x)\| = \|\nabla U(x)\| \leq F_{\max}")),
                g.iter().map(|x| x * x).sum::<f64>().sqrt(),
                fmax,
                0.,
                "upper",
                report,
                "Analytic box-domain force bound; not used as a bound on Gaussian excursions outside the valid domain",
                json!({"box_half_width":radius,"F_max":fmax,"native_config":cfg}),
            );
        }
    }
    for (name, observed, predicted) in [
        (
            "friction_factor",
            number(record, "/constants/friction_factor")?,
            a,
        ),
        (
            "thermal_variance",
            number(record, "/constants/thermal_variance")?,
            q2,
        ),
    ] {
        exact(
            out,
            format!("kinetic{index}/{name}"),
            "def-baoab-integrator",
            None,
            observed,
            predicted,
            report,
            hy.clone(),
        );
    }
    // Reconstruct the actual algebra, not recorded 'passed' booleans.
    for (j, e) in rows(record, "/experiments").iter().enumerate() {
        let tag = format!("kinetic{index}/input{j}");
        let gradient = cfg.landscape.analytic_gradient(&cfg.inputs[j].positions)?;
        let v1: Vec<f64> = cfg.inputs[j]
            .velocities
            .iter()
            .zip(&gradient)
            .map(|(v, g)| v - c * g)
            .collect();
        let means: Vec<f64> = cfg.inputs[j]
            .positions
            .iter()
            .zip(&v1)
            .map(|(x, v)| x + c * (1. + a) * v)
            .collect();
        for (axis, mean) in means.iter().enumerate() {
            exact(
                out,
                format!("{tag}/position_mean{axis}"),
                "def-baoab-integrator",
                None,
                number(e, &format!("/conditional_position_mean/{axis}"))?,
                *mean,
                report,
                hy.clone(),
            );
            let (m, se) = estimate(&e["position_drift"][axis])?;
            add(
                out,
                format!("{tag}/native_position_drift{axis}"),
                "def-baoab-integrator",
                None,
                m,
                *mean - cfg.inputs[j].positions[axis],
                se,
                "equal",
                report,
                "Stored independent fixed-input repetitions; full physical outputs, terminally dead rows retained",
                hy.clone(),
            );
        }
        for ij in 0..cfg.dimensions * cfg.dimensions {
            let diag = ij / cfg.dimensions == ij % cfg.dimensions;
            let (m, se) = estimate(&e["thermostat_innovation_covariance"][ij])?;
            add(
                out,
                format!("{tag}/OU_covariance{ij}"),
                "def-baoab-integrator",
                Some(
                    r"v^{(2)} = e^{-\gamma \tau} v^{(1)} + \sqrt{\frac{1 - e^{-2\gamma\tau}}{2\gamma}} \, \Sigma \xi",
                ),
                m,
                if diag { q2 } else { 0. },
                se,
                "equal",
                report,
                "Actual native OU innovations; constant isotropic diffusion",
                hy.clone(),
            );
            let (m, se) = estimate(&e["position_covariance"][ij])?;
            add(
                out,
                format!("{tag}/position_covariance{ij}"),
                "def-baoab-integrator",
                None,
                m,
                if diag {
                    c * c * q2 + h * cfg.position_diffusion.powi(2)
                } else {
                    0.
                },
                se,
                "equal",
                report,
                "Actual native BAOAB plus final position noise; fixed input",
                hy.clone(),
            );
        }
        for axis in 0..cfg.dimensions {
            let (m, se) = estimate(&e["thermostat_innovation_mean"][axis])?;
            add(
                out,
                format!("{tag}/OU_zero_mean{axis}"),
                "def-baoab-integrator",
                None,
                m,
                0.,
                se,
                "equal",
                report,
                "Actual native OU innovations",
                hy.clone(),
            );
        }
        add(
            out,
            format!("{tag}/cap_radius"),
            "thm-kinetic-exact-baoab-cap-coupling",
            None,
            number(e, "/maximum_velocity_norm")?,
            cfg.velocity_cap,
            0.,
            "upper",
            report,
            "Pathwise cap bound; applicable also outside terminal domain",
            hy.clone(),
        );
        add(
            out,
            format!("{tag}/native_precap_identity"),
            "def-baoab-integrator",
            None,
            number(e, "/maximum_b2_to_final_cap_error")?,
            0.,
            0.,
            "equal",
            report,
            "Native B2 endpoint independently reconstructed through actual radial cap",
            hy.clone(),
        );
        if cfg.landscape.ripple_amplitude == 0.
            && cfg
                .landscape
                .curvature
                .iter()
                .all(|x| *x == cfg.landscape.curvature[0])
        {
            let kval = 1. - c * c * cfg.landscape.curvature[0];
            let (observed, se) = e["velocity_variance"].as_array().unwrap().iter().try_fold(
                (0., 0.),
                |(m, s), v| {
                    let (a, b) = estimate(v)?;
                    Ok::<_, GasError>((m + a, s + b))
                },
            )?;
            add(
                out,
                format!("{tag}/postcap_variance"),
                "thm-velocity-variance-contraction-kinetic",
                None,
                observed,
                cfg.dimensions as f64 * kval * kval * q2,
                se,
                "upper",
                report,
                "Fixed-input quadratic pre-cap variance d(1-h² curvature/4)²q²; radial cap is 1-Lipschitz. Sum of per-coordinate SEs is a conservative uncertainty bound",
                hy.clone(),
            );
        }
        if cfg.position_diffusion > 0.
            && cfg.box_half_width.is_some()
            && cfg.landscape.ripple_amplitude == 0.
        {
            let delta_pre = cfg
                .landscape
                .curvature
                .iter()
                .map(|curvature| {
                    ((1. - c * c * (1. + a) * curvature) * cfg.coupling_position_shift
                        + c * (1. + a) * cfg.coupling_velocity_shift)
                        .abs()
                })
                .sum::<f64>();
            let bound = (2. * delta_pre
                / (cfg.position_diffusion * h.sqrt() * (2. * std::f64::consts::PI).sqrt()))
            .min(1.);
            let (m, se) = estimate(&e["synchronous_death_probability_difference"])?;
            add(
                out,
                format!("{tag}/terminal_status_probability_difference"),
                "lem-kinetic-terminal-status-coupling",
                None,
                m.abs(),
                bound,
                se,
                "upper",
                report,
                "Absolute difference of death probabilities is bounded by coupling mismatch; stored statistic is not the mismatch event itself",
                hy.clone(),
            );
        }
    }
    if cfg.landscape.ripple_amplitude == 0.
        && cfg.landscape.curvature.iter().all(|x| *x == 1.)
        && h < 2.
        && cfg.friction > 0.
        && cfg.velocity_diffusion > 0.
    {
        let constants =
            quadratic_cap_constants(h, cfg.friction, cfg.velocity_diffusion, cfg.velocity_cap)?;
        let k = number(&constants, "/k")?;
        let eta = number(&constants, "/eta")?;
        let bx = -c * (1. + a) * k;
        let bv = a - c * c * (1. + a);
        let matrix = [[1. - c * c * (1. + a), c * (1. + a)], [bx, bv]];
        let p0 = (2. / std::f64::consts::PI).sqrt() * (-2f64).exp();
        exact(
            out,
            format!("kinetic{index}/p0"),
            "thm-kinetic-exact-baoab-cap-coupling",
            None,
            number(&constants, "/p0")?,
            p0,
            report,
            hy.clone(),
        );
        let diss = [
            [
                k * (1. - a * a) * c * c + eta * bx * bx,
                (-k * (1. - a * a) * c + eta * bx * bv) / k.sqrt(),
            ],
            [
                -k * (1. - a * a) * c + eta * bx * bv,
                k * (1. - a * a) + eta * bv * bv,
            ],
        ];
        let trace = diss[0][0] / k + diss[1][1];
        let determinant =
            (diss[0][0] * diss[1][1] - (-k * (1. - a * a) * c + eta * bx * bv).powi(2)) / k;
        let delta = number(&constants, "/delta")?;
        let smallest_eigenvalue = 0.5 * (trace - (trace * trace - 4. * determinant).sqrt());
        let full_residual = (trace - number(&constants, "/T")?).abs()
            + (determinant - number(&constants, "/D")?).abs()
            + (delta - determinant / trace).abs();
        exact(
            out,
            format!("kinetic{index}/T_D_delta_complete_equation"),
            "thm-kinetic-exact-baoab-cap-coupling",
            Some(formula("T=1-a^2+")),
            full_residual,
            0.,
            report,
            hy.clone(),
        );
        add(
            out,
            format!("kinetic{index}/spectral_delta_lower_bound"),
            "thm-kinetic-exact-baoab-cap-coupling",
            None,
            delta,
            smallest_eigenvalue,
            0.,
            "upper",
            report,
            "Independent symmetric dissipation-matrix eigensystem; verifies delta=D/T is a lower bound on minimum eigenvalue",
            hy.clone(),
        );
        if h == 0.04 && cfg.friction == 1. && cfg.velocity_diffusion == 1. && cfg.velocity_cap == 2.
        {
            add(
                out,
                format!("kinetic{index}/canonical_eta"),
                "cor-kinetic-canonical-coupling",
                Some(r"\eta>0.0184"),
                0.0184,
                eta,
                0.,
                "upper",
                report,
                "Exact corollary substitution",
                hy.clone(),
            );
            add(
                out,
                format!("kinetic{index}/canonical_delta"),
                "cor-kinetic-canonical-coupling",
                Some(r"\delta>6.03\times10^{-6}"),
                6.03e-6,
                delta,
                0.,
                "upper",
                report,
                "Exact corollary substitution",
                hy.clone(),
            );
        }
        exact(
            out,
            format!("kinetic{index}/cap_dissipation_trace"),
            "thm-kinetic-exact-baoab-cap-coupling",
            None,
            trace,
            number(&constants, "/T")?,
            report,
            hy.clone(),
        );
        exact(
            out,
            format!("kinetic{index}/cap_dissipation_determinant"),
            "thm-kinetic-exact-baoab-cap-coupling",
            None,
            determinant,
            number(&constants, "/D")?,
            report,
            hy.clone(),
        );
        for i in 0..2 {
            for j in 0..2 {
                let aq = matrix[0][i] * k * matrix[0][j] + matrix[1][i] * matrix[1][j];
                let w = [-c, 1.];
                let expected = if i == j {
                    if i == 0 { k } else { 1. }
                } else {
                    0.
                } - k * (1. - a * a) * w[i] * w[j];
                exact(
                    out,
                    format!("kinetic{index}/quadratic_energy_matrix{i}{j}"),
                    "thm-kinetic-exact-baoab-cap-coupling",
                    None,
                    aq,
                    expected,
                    report,
                    hy.clone(),
                );
            }
        }
        add(
            out,
            format!("kinetic{index}/strict_delta"),
            "thm-kinetic-exact-baoab-cap-coupling",
            None,
            -number(&constants, "/delta")?,
            0.,
            0.,
            "upper",
            report,
            "Exact finite-step rate computation only: full paired post-cap costs were not retained in stored fixed-input kinetic experiment",
            json!({"constants":constants,"analytic_hypotheses":hy}),
        );
    }
    Ok(())
}

/// Applicability is reconstructed from actual parameters, never old status flags.
pub fn linear_control_applicable(case: &Value) -> bool {
    let cfg = &case["native_config"];
    case["profile"].as_str().is_some_and(|p| {
        matches!(
            p,
            "linear_quadratic_noiseless"
                | "linear_sphere_noiseless"
                | "linear_quadratic_independent_noise"
        )
    }) && cfg.pointer("/boundary/kind").and_then(Value::as_str) == Some("unbounded")
        && cfg
            .pointer("/kinetic/velocity_cap")
            .is_some_and(Value::is_null)
        && cfg
            .pointer("/fitness/reward_exponent")
            .and_then(Value::as_f64)
            == Some(0.)
        && cfg
            .pointer("/fitness/diversity_exponent")
            .and_then(Value::as_f64)
            == Some(0.)
        && cfg
            .pointer("/clone_transform/jitter_amplitude")
            .and_then(Value::as_f64)
            == Some(0.)
}
fn decay(out: &mut Vec<Value>, report: &str, case: &Value, index: usize) -> Result<()> {
    let applicable = linear_control_applicable(case);
    exact(
        out,
        format!("decay{index}/applicability"),
        "thm-inter-swarm-contraction-kinetic",
        None,
        f64::from(
            case["linear_control_theorem_applicable"]
                .as_bool()
                .unwrap_or(false),
        ),
        f64::from(applicable),
        report,
        json!({"actual_native_parameters":case["native_config"],"scope":"linear quadratic control; general theorem explicitly deferred"}),
    );
    if !applicable {
        return Ok(());
    }
    let cfg = &case["native_config"];
    let dt = number(cfg, "/kinetic/integrator/dt")?;
    let gamma = number(cfg, "/kinetic/integrator/friction")?;
    let curvature = if case["profile"] == "linear_sphere_noiseless" {
        2.
    } else {
        1.
    };
    let noise = number(cfg, "/kinetic/noise/geometry/scale/values/0")?;
    let sigmax = number(cfg, "/kinetic/position_diffusion")?;
    let reconstructed = linear_decay_constants(dt, gamma, curvature, noise, sigmax)?;
    let d = number(case, "/dimensions")?;
    let n = number(case, "/walkers")?;
    let shared = case["pair_randomness"] == "shared";
    let stored = &case["reference_linear_constants"];
    let h = json!({"native_config":cfg,"quadratic_curvature":curvature,"population":n,"dimensions":d,"pair_noise":case["pair_randomness"],"no_cap_no_selection_no_boundary":true,"not_canonical_theorem":true});
    for (name, value) in [
        ("contraction_factor", reconstructed.contraction_factor),
        (
            "squared_error_decay_rate",
            reconstructed.squared_error_decay_rate,
        ),
        ("lambda_min", reconstructed.lambda_min),
        ("lambda_max", reconstructed.lambda_max),
        ("lyapunov_identity_rhs", reconstructed.lyapunov_identity_rhs),
    ] {
        exact(
            out,
            format!("decay{index}/{name}"),
            "def-hypocoercive-norm",
            None,
            number(stored, &format!("/{name}"))?,
            value,
            report,
            h.clone(),
        );
    }
    for i in 0..2 {
        for j in 0..2 {
            exact(
                out,
                format!("decay{index}/BAOAB_matrix{i}{j}"),
                "def-baoab-integrator",
                None,
                number(stored, &format!("/map/{i}/{j}"))?,
                reconstructed.map[i][j],
                report,
                h.clone(),
            );
            exact(
                out,
                format!("decay{index}/noise_covariance{i}{j}"),
                "def-baoab-integrator",
                None,
                number(stored, &format!("/innovation_covariance/{i}/{j}"))?,
                reconstructed.innovation_covariance[i][j],
                report,
                h.clone(),
            );
        }
    }
    let p = reconstructed.lyapunov_matrix;
    let a = reconstructed.map;
    let noise_trace = p[0][0] * reconstructed.innovation_covariance[0][0]
        + 2. * p[0][1] * reconstructed.innovation_covariance[0][1]
        + p[1][1] * reconstructed.innovation_covariance[1][1];
    let injection = if shared { 0. } else { 2. * d * noise_trace };
    let rho = reconstructed.contraction_factor;
    let mut deterministic = [1., 0.];
    let mut covariance = [[0.; 2]; 2];
    let initial = number(&case["ensemble_trajectory"][0], "/alive_error/mean")?;
    for (t, point) in rows(case, "/ensemble_trajectory").iter().enumerate() {
        if t > 0 {
            deterministic = [
                a[0][0] * deterministic[0] + a[0][1] * deterministic[1],
                a[1][0] * deterministic[0] + a[1][1] * deterministic[1],
            ];
            let old = covariance;
            for i in 0..2 {
                for j in 0..2 {
                    covariance[i][j] = (0..2)
                        .flat_map(|k| (0..2).map(move |l| a[i][k] * old[k][l] * a[j][l]))
                        .sum::<f64>()
                        + if shared {
                            0.
                        } else {
                            2. * reconstructed.innovation_covariance[i][j]
                        };
                }
            }
        }
        let mean_cost = p[0][0] * deterministic[0].powi(2)
            + 2. * p[0][1] * deterministic[0] * deterministic[1]
            + p[1][1] * deterministic[1].powi(2);
        // Initial translation has total q-cost one and zero velocity difference.
        let floor = d
            * (p[0][0] * covariance[0][0]
                + 2. * p[0][1] * covariance[0][1]
                + p[1][1] * covariance[1][1])
            / n;
        let (observed, recorded_se) = estimate(&point["barycenter_error"])?;
        // The law is explicitly Gaussian. With only eight seed pairs a sample
        // variance can sharply underestimate the quadratic-form variance. Use
        // its independently known variance, not a larger fitted error bar.
        let normalized_covariance = covariance.map(|row| row.map(|v| v / n));
        let mean_gram = [
            [
                deterministic[0].powi(2),
                deterministic[0] * deterministic[1],
            ],
            [
                deterministic[0] * deterministic[1],
                deterministic[1].powi(2),
            ],
        ];
        let exact_variance =
            gaussian_quadratic_cost_variance(p, normalized_covariance, mean_gram, d as usize);
        let seeds = number(point, "/barycenter_error/independent_pairs")?;
        let se = (exact_variance.max(0.) / seeds).sqrt();
        let mut statistical_hypotheses = h.clone();
        statistical_hypotheses["recorded_sample_standard_error"] = json!(recorded_se);
        statistical_hypotheses["exact_Gaussian_quadratic_variance"] = json!(exact_variance);
        statistical_hypotheses["independent_seed_pairs"] = json!(seeds);
        statistical_hypotheses["uncertainty_formula"] = json!(
            "Var(ZᵀPZ)=2 tr((PΣ)^2)+4 μᵀPΣPμ; independent d coordinates, Σ=finite-time pair covariance/N; SE=sqrt(Var/independent seed pairs)"
        );
        add(
            out,
            format!("decay{index}/step{t}/barycenter_exact"),
            "def-velocity-barycenter-energy",
            None,
            observed,
            mean_cost + floor,
            se,
            "equal",
            report,
            "Independent seed-pair barycenters, exact finite-time BAOAB covariance and Gaussian quadratic standard error; normalized q-cost, no survival filtering. Recorded sample SE is retained separately",
            statistical_hypotheses,
        );
        let envelope =
            rho.powi(t as i32) * initial + injection * (1. - rho.powi(t as i32)) / (1. - rho);
        let (observed, se) = estimate(&point["alive_error"])?;
        add(
            out,
            format!("decay{index}/step{t}/N_uniform_transport_envelope"),
            "thm-inter-swarm-contraction-kinetic",
            None,
            observed,
            envelope,
            se,
            "upper",
            report,
            "Exact quadratic linear-control envelope derived from AᵀPA=P-rI; each engine marginal remains native; empirical optimal transport costs are normalized and permutation invariant",
            h.clone(),
        );
        exact(
            out,
            format!("decay{index}/step{t}/barycenter_noise_floor"),
            "thm-velocity-barycenter-dissipation",
            None,
            number(point, "/exact_barycenter_noise_floor")?,
            floor,
            report,
            h.clone(),
        );
    }
    Ok(())
}

/// Reuse recorded runs to validate each estimate whose necessary observables survive serialization.
pub fn validate(reports: &[(String, Value)]) -> Result<Value> {
    let mut comparisons = Vec::new();
    let mut kinetic_cases = 0;
    let mut decay_cases = 0;
    let mut capped_run_points = 0;
    for (path, report) in reports {
        for record in rows(report, "/kinetic") {
            kinetic(&mut comparisons, path, record, kinetic_cases)?;
            kinetic_cases += 1;
        }
        if path.contains("decay") {
            for case in rows(report, "/cases") {
                decay(&mut comparisons, path, case, decay_cases)?;
                decay_cases += 1;
            }
        }
        for (i, run) in rows(report, "/runs").iter().enumerate() {
            if let Some(cap) = run
                .pointer("/config/gas/kinetic/velocity_cap")
                .and_then(Value::as_f64)
            {
                for (j, point) in rows(run, "/trajectory").iter().enumerate() {
                    if number(point, "/moments/alive")? <= 0. {
                        continue;
                    }
                    add(
                        &mut comparisons,
                        format!("run{i}/step{j}/velocity_second_moment_cap"),
                        "thm-velocity-variance-contraction-kinetic",
                        None,
                        number(point, "/moments/velocity_second_moment")?,
                        cap * cap,
                        0.,
                        "upper",
                        path,
                        "Pathwise final-cap second moment on eligible physical atoms. This validates a uniform moment bound, not the uncapped continuous-time variance rate",
                        json!({"native_cap":cap,"population":run["config"]["walkers"],"dimensions":run["config"]["dimensions"]}),
                    );
                    capped_run_points += 1;
                }
            }
        }
    }
    let gaps = vec![
        json!({"id":"strict_capped_quadratic_coupling_empirical","source_labels":["thm-kinetic-exact-baoab-cap-coupling","cor-kinetic-canonical-coupling"],"reason":"Constants and matrix identities are reconstructed; stored capped kinetic experiments lack full paired post-cap Q-costs. Uncapped linear control is not substituted."}),
        json!({"id":"bounded_transport_smoothing_empirical","source_labels":["thm-kinetic-bounded-transport-smoothing","cor-kinetic-full-cluster-smoothing"],"reason":"Stored records do not retain row output laws/maximal-noise couplings for the bounded marked metric; terminal probability consequence only is measured."}),
        json!({"id":"weak_error_exact_reference","source_labels":["thm-discretization","prop-weak-error-variance","prop-weak-error-boundary","prop-weak-error-wasserstein","prop-explicit-constants"],"reason":"No exact-SDE reference trajectory or timestep refinement is stored. K_integ,K_V,C_LM and weak-error rates are unidentified; matching transfer is explicitly deferred in source."}),
        json!({"id":"general_hypocoercive_rates","source_labels":["thm-inter-swarm-contraction-kinetic","lem-location-error-drift-kinetic","lem-structural-error-drift-kinetic"],"reason":"These general W2 results are in Part II (Deferred). L_F alone does not identify a certified positive nonconvex contraction rate. Canonical endpoint decay is descriptive; exact control has distinct premises."}),
        json!({"id":"component_generator_rates","source_labels":["thm-velocity-variance-contraction-kinetic","thm-velocity-barycenter-dissipation","cor-net-velocity-contraction","cor-net-barycenter-drift","thm-positional-variance-bounded-expansion","assump-uniform-variance-bounds","cor-net-positional-contraction"],"reason":"Stored interacting traces serialize final moments only, not pre/post-kinetic full-stage variances, force covariance or exact-SDE generator. Cap bounds, fixed-input conditional noise variance and control barycenter noise are measured, but C1,C2,Ckin,x and uncapped generator rates are not inferred."}),
        json!({"id":"barrier_and_minorization","source_labels":["thm-boundary-potential-contraction-kinetic","cor-total-boundary-safety","lem-kinetic-minorization"],"reason":"Aligned barrier derivatives, alpha_align,K_phi,K_curv,boundary layer and density/minorization witnesses are absent. Stored finite moments cannot certify kappa_pot,C_pot or epsilon_K."}),
        json!({"id":"stationary_law_pde","source_labels":["prop-fokker-planck-kinetic","rem-formal-invariant-measure"],"reason":"No stationary density or PDE reference is stored; cap and killing change the unmodified Langevin stationary-law premises."}),
    ];
    let source_labels: Vec<&str> = SOURCE
        .lines()
        .filter_map(|s| s.strip_prefix(":label: "))
        .collect();
    let failed = comparisons.iter().filter(|v| v["passed"] == false).count();
    Ok(
        json!({"chapter":5,"title":"Hypocoercivity and Convergence of the Euclidean Gas","source_path":PATH,
        "comparisons":comparisons,"gaps":gaps,"inventory_labels":source_labels,
        "summary":{"comparisons":comparisons.len(),"comparisons_failed":failed,"kinetic_cases":kinetic_cases,"decay_cases":decay_cases,"capped_run_points":capped_run_points,"complete_required_estimates":false},
        "scope":"Stored data only, no new trajectories. Gaussian fixed-input experiments and independent seed pairs supply uncertainty. Linear controls are explicitly distinguished from canonical selected/capped/killed swarms."}),
    )
}
