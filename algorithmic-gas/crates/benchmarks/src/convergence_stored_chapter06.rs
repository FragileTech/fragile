//! Reanalysis of retained native experiments for Chapter 6. No simulation is run.
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};
use std::collections::BTreeSet;

const SOURCE: &str =
    include_str!("../../../../docs/source/2_fractal_gas/convergence_program/06_convergence.md");
const SOURCE_PATH: &str = "docs/source/2_fractal_gas/convergence_program/06_convergence.md";
type Matrix = [[f64; 2]; 2];

fn num(v: &Value, path: &str) -> Result<f64> {
    v.pointer(path)
        .and_then(Value::as_f64)
        .filter(|x| x.is_finite())
        .ok_or_else(|| {
            GasError::Configuration(format!("Chapter 6 retained report lacks finite {path}"))
        })
}
fn matrix(v: &Value, path: &str) -> Result<Matrix> {
    let mut out = [[0.; 2]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = num(v, &format!("{path}/{i}/{j}"))?;
        }
    }
    Ok(out)
}
fn mul(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..2).map(|k| a[i][k] * b[k][j]).sum()))
}
fn tr(a: Matrix, b: Matrix) -> f64 {
    (0..2)
        .flat_map(|i| (0..2).map(move |j| a[i][j] * b[j][i]))
        .sum()
}
fn trans(a: Matrix) -> Matrix {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}
fn maximum_difference(a: Matrix, b: Matrix) -> f64 {
    (0..2)
        .flat_map(|i| (0..2).map(move |j| (a[i][j] - b[i][j]).abs()))
        .fold(0., f64::max)
}
fn quote(label: &str, expression: usize) -> String {
    let start = SOURCE
        .find(&format!(":label: {label}\n"))
        .expect("Chapter 6 label");
    let block = SOURCE[start..].split(":::\n").next().unwrap();
    block
        .split("$$")
        .skip(1)
        .step_by(2)
        .nth(expression)
        .unwrap_or("")
        .trim()
        .to_owned()
}
#[allow(clippy::too_many_arguments)]
fn check(
    out: &mut Vec<Value>,
    id: String,
    labels: &[&str],
    source_quotes: &[String],
    observed: f64,
    bound: f64,
    relation: &str,
    tolerance: f64,
    scope: &str,
    report_path: &str,
    hypotheses: Value,
) {
    let passed = observed.is_finite()
        && bound.is_finite()
        && if relation == "equal" {
            (observed - bound).abs() <= tolerance
        } else {
            observed <= bound + tolerance
        };
    let mut exact_quotes = source_quotes.to_vec();
    if id.starts_with("physical_time_rate_") {
        exact_quotes.push("-h^{-1}\\log r".into());
    }
    if id.starts_with("measured_transport_geometric_") {
        exact_quotes.push(quote("prop-wasserstein-rate-explicit", 1));
        exact_quotes.push(quote("lem-convergence-drift-iteration", 0));
        exact_quotes.push(quote("lem-convergence-drift-iteration", 1));
    }
    out.push(json!({"id":id,"source_labels":labels,"source_quotes":exact_quotes,"observed":observed,"bound":bound,"relation":relation,"tolerance":tolerance,"passed":passed,"status":if passed{"supported_with_recorded_tolerance"}else{"violated"},"scope":scope,"report_path":report_path,"hypotheses":hypotheses}));
}
fn cap_runs(path: &str, report: &Value, checks: &mut Vec<Value>) -> Result<()> {
    let Some(runs) = report["runs"].as_array() else {
        return Ok(());
    };
    for (ri, run) in runs.iter().enumerate() {
        let Some(cap) = run
            .pointer("/config/gas/kinetic/velocity_cap")
            .and_then(Value::as_f64)
        else {
            continue;
        };
        let n = num(run, "/config/walkers")?;
        let Some(points) = run["trajectory"].as_array() else {
            continue;
        };
        for (pi, point) in points
            .iter()
            .enumerate()
            .filter(|(_, p)| p["step"].as_u64().unwrap_or(0) > 0)
        {
            let k = num(point, "/moments/alive")?;
            if k == 0. {
                continue;
            }
            let raw = num(point, "/moments/velocity_second_moment")?;
            let e = raw * k / n;
            let source = "NV_v(S')\\le\\sum_{i\\in B}\\|v_i'\\|^2\\le Nv_{\\max}^2";
            check(
                checks,
                format!("capped_velocity_moment_run{ri}_point{pi}"),
                &["lem-convergence-capped-displacement"],
                &[],
                e,
                cap * cap,
                "upper",
                1e-10,
                "Actual complete-update alive velocity second moment, converted from retained 1/k to chapter 1/N normalization; this implies V_v<=vmax². It does not reconstruct E_v without a saved velocity barycenter.",
                &format!("{path}#/runs/{ri}/trajectory/{pi}"),
                json!({"velocity_cap":cap,"N":n,"alive":k,"retained_normalization":"1/k","chapter_normalization":"1/N","partial_source_expression":source}),
            );
            if let Some(domain) = run.pointer("/config/gas/boundary/domain")
                && let (Some(lower), Some(upper)) =
                    (domain["lower"].as_array(), domain["upper"].as_array())
            {
                let radius2: f64 = lower
                    .iter()
                    .zip(upper)
                    .map(|(l, u)| {
                        l.as_f64()
                            .unwrap()
                            .abs()
                            .max(u.as_f64().unwrap().abs())
                            .powi(2)
                    })
                    .sum();
                let x = num(point, "/moments/position_second_moment")? * k / n;
                check(
                    checks,
                    format!("terminal_box_moment_run{ri}_point{pi}"),
                    &["def-tv-lyapunov"],
                    &[],
                    x,
                    radius2,
                    "upper",
                    1e-10,
                    "Terminal alive positions remain in the actual absorbing box; phi(x)=|x|² has an N-uniform finite moment. This is a terminal moment bound, not a singular-wall-barrier or invariant-law estimate.",
                    &format!("{path}#/runs/{ri}/trajectory/{pi}"),
                    json!({"N":n,"alive":k,"box_radius_squared":radius2,"barrier":"|x|²","converted_normalization":true}),
                );
            }
        }
    }
    Ok(())
}
fn applicable(case: &Value) -> bool {
    let cfg = &case["native_config"];
    let profile = case["profile"].as_str().unwrap_or("");
    let config_ok = [
        "linear_quadratic_noiseless",
        "linear_sphere_noiseless",
        "linear_quadratic_independent_noise",
    ]
    .contains(&profile)
        && cfg.pointer("/boundary/kind").and_then(Value::as_str) == Some("unbounded")
        && cfg
            .pointer("/fitness/reward_exponent")
            .and_then(Value::as_f64)
            == Some(0.)
        && cfg
            .pointer("/fitness/diversity_exponent")
            .and_then(Value::as_f64)
            == Some(0.)
        && cfg
            .pointer("/kinetic/integrator/kind")
            .and_then(Value::as_str)
            == Some("baoab")
        && cfg.pointer("/kinetic/velocity_cap") == Some(&Value::Null)
        && cfg
            .pointer("/clone_transform/jitter_amplitude")
            .and_then(Value::as_f64)
            == Some(0.)
        && cfg.pointer("/clone_transform/restitution") == Some(&Value::Null)
        && cfg.pointer("/qft/viscosity") == Some(&Value::Null)
        && cfg.pointer("/qft/graph_viscosity") == Some(&Value::Null);
    config_ok
        && case["seed_trajectories"].as_array().is_some_and(|seeds| {
            !seeds.is_empty()
                && seeds.iter().all(|s| {
                    s["points"].as_array().is_some_and(|ps| {
                        ps.iter().all(|p| {
                            p["observables"]["alive_counts"]
                                .as_array()
                                .is_some_and(|a| {
                                    a.len() == 2
                                        && a.iter().all(|x| x.as_u64() == case["walkers"].as_u64())
                                })
                                && p["cloned"]
                                    .as_array()
                                    .is_some_and(|a| a.iter().all(|x| x.as_u64() == Some(0)))
                                && p["revived"]
                                    .as_array()
                                    .is_some_and(|a| a.iter().all(|x| x.as_u64() == Some(0)))
                        })
                    })
                })
        })
}
fn controls(
    path: &str,
    report: &Value,
    checks: &mut Vec<Value>,
    gaps: &mut Vec<Value>,
) -> Result<()> {
    let Some(cases) = report["cases"].as_array() else {
        return Ok(());
    };
    // Only decay reports have paired complete-update trajectories.
    for (ci, c) in cases
        .iter()
        .enumerate()
        .filter(|(_, c)| c.get("ensemble_trajectory").is_some())
    {
        let rp = format!("{path}#/cases/{ci}");
        if !applicable(c) {
            gaps.push(json!({"source_labels":["prop-wasserstein-rate-explicit","thm-total-rate-explicit"],"report_path":rp,"reason":"No independently reconstructed complete-update contracting coupling certificate for this selected/capped/killed case; endpoint decline remains a measurement."}));
            continue;
        }
        let cfg = &c["native_config"];
        let h = num(cfg, "/kinetic/integrator/dt")?;
        let gamma = num(cfg, "/kinetic/integrator/friction")?;
        let curvature = if c["profile"] == "linear_sphere_noiseless" {
            2.
        } else {
            1.
        };
        let co = (-gamma * h).exp();
        let bb = h * (1. + co) / 2.;
        let aa = 1. - bb * h * curvature / 2.;
        let a = [
            [aa, bb],
            [
                -h * curvature * (co + aa) / 2.,
                co - h * curvature * bb / 2.,
            ],
        ];
        let reference = &c["reference_linear_constants"];
        let p = matrix(reference, "/lyapunov_matrix")?;
        let eigen_span = ((p[0][0] - p[1][1]).powi(2) + 4. * p[0][1].powi(2)).sqrt();
        let lmax = (p[0][0] + p[1][1] + eigen_span) / 2.;
        let lmin = (p[0][0] + p[1][1] - eigen_span) / 2.;
        let evolved = mul(mul(trans(a), p), a);
        let decrement = p[0][0] - evolved[0][0];
        let rho = 1. - decrement / lmax;
        let scale = num(cfg, "/kinetic/noise/geometry/scale/values/0")?;
        let sigma_pos = num(cfg, "/kinetic/position_diffusion")?;
        let q2 = scale * scale * (-(-2. * gamma * h).exp_m1()) / (2. * gamma);
        let nx = h / 2.;
        let nv = 1. - h * h * curvature / 4.;
        let q = [
            [nx * nx * q2 + sigma_pos * sigma_pos * h, nx * nv * q2],
            [nx * nv * q2, nv * nv * q2],
        ];
        let d = num(c, "/dimensions")?;
        let n = num(c, "/walkers")?;
        let shared = c["pair_randomness"] == "shared";
        let noise_factor = if shared { 0. } else { 2. };
        let source_scope = "Conservative cloning-disabled quadratic native BAOAB kernel with fixed positive Lyapunov cost, zero accepted clones/revivals, no cap or collisions, and recorded pair noise law. Empirical-measure transport is not TV or QSD law distance.";
        let hyp = json!({"linear_control_reconstructed":true,"kernel_kind":"conservative","pair_survival_probability":1.0,"initial_difference":"uniform position translation 1/sqrt(d) per coordinate; zero velocity difference, prescribed by retained native DecayProfile generator","all_rows_alive":true,"cloning_disabled_by_zero_exponents":true,"lyapunov_positive":lmin>0.,"rho":rho,"lambda_min":lmin,"lambda_max":lmax,"pair_randomness":c["pair_randomness"],"N":n,"d":d,"h":h});
        check(
            checks,
            format!("map_reconstruction_case{ci}"),
            &["prop-wasserstein-rate-explicit"],
            &[],
            maximum_difference(a, matrix(reference, "/map")?),
            0.,
            "equal",
            1e-12,
            source_scope,
            &rp,
            hyp.clone(),
        );
        check(
            checks,
            format!("noise_reconstruction_case{ci}"),
            &["prop-wasserstein-rate-explicit"],
            &[],
            maximum_difference(q, matrix(reference, "/innovation_covariance")?),
            0.,
            "equal",
            1e-12,
            source_scope,
            &rp,
            hyp.clone(),
        );
        let target = [
            [p[0][0] - decrement, p[0][1]],
            [p[1][0], p[1][1] - decrement],
        ];
        check(
            checks,
            format!("lyapunov_identity_case{ci}"),
            &["prop-wasserstein-rate-explicit"],
            &[],
            maximum_difference(evolved, target),
            0.,
            "equal",
            1e-12,
            source_scope,
            &rp,
            hyp.clone(),
        );
        check(
            checks,
            format!("positive_lyapunov_and_rate_case{ci}"),
            &["prop-wasserstein-rate-explicit"],
            &[],
            if lmin > 0. && (0.0..1.0).contains(&rho) && decrement > 0. {
                0.
            } else {
                1.
            },
            0.,
            "equal",
            0.,
            source_scope,
            &rp,
            hyp.clone(),
        );
        check(
            checks,
            format!("contraction_factor_case{ci}"),
            &["prop-wasserstein-rate-explicit"],
            &[],
            num(reference, "/contraction_factor")?,
            rho,
            "equal",
            1e-12,
            source_scope,
            &rp,
            hyp.clone(),
        );
        check(
            checks,
            format!("physical_time_rate_case{ci}"),
            &["thm-total-rate-explicit"],
            &[],
            num(reference, "/squared_error_decay_rate")?,
            -rho.ln() / h,
            "equal",
            1e-12,
            source_scope,
            &rp,
            hyp.clone(),
        );
        // Certify the one-step inequality for every displacement using the actual split map.
        let g = std::array::from_fn::<_, 2, _>(|i| {
            std::array::from_fn::<_, 2, _>(|j| rho * p[i][j] - evolved[i][j])
        });
        let gmin = (g[0][0] + g[1][1]
            - ((g[0][0] - g[1][1]).powi(2) + 4. * g[0][1] * g[1][0]).sqrt())
            / 2.;
        let b_d = noise_factor * d * tr(p, q);
        check(
            checks,
            format!("global_one_step_coupling_case{ci}"),
            &["prop-wasserstein-rate-explicit"],
            &[quote("prop-wasserstein-rate-explicit", 0)],
            -gmin,
            0.,
            "upper",
            1e-12,
            source_scope,
            &rp,
            json!({"premises":hyp,"b_D":b_d,"matrix_difference":g,"nonnegative_source":b_d>=0.,"positive_metric":lmin>0.,"observable":"average normalized P-quadratic displacement under an admissible paired coupling"}),
        );
        check(
            checks,
            format!("coupling_rate_domain_case{ci}"),
            &["lem-convergence-drift-iteration", "thm-total-rate-explicit"],
            &["0\\le r<1".into(), "0<r<1".into(), "h>0".into()],
            if h > 0. && rho > 0. && rho < 1. {
                0.
            } else {
                1.
            },
            0.,
            "equal",
            0.,
            source_scope,
            &rp,
            hyp.clone(),
        );
        check(
            checks,
            format!("conservative_survival_domain_case{ci}"),
            &["lem-convergence-drift-iteration"],
            &["\\mu Q^n1>0".into()],
            1.,
            1.,
            "equal",
            0.,
            "Actual unbounded linear-control native kernel: every row survives, hence the pair kernel is conservative and its survival normalization is one at all times. This does not certify any killed QSD.",
            &rp,
            hyp.clone(),
        );
        if b_d > 0. {
            check(
                checks,
                format!("positive_pair_noise_source_case{ci}"),
                &["prop-wasserstein-rate-explicit"],
                &["b_D>0".into()],
                -b_d,
                0.,
                "upper",
                0.,
                source_scope,
                &rp,
                hyp.clone(),
            );
        }
        let points = c["ensemble_trajectory"]
            .as_array()
            .ok_or_else(|| GasError::Configuration("missing control trajectory".into()))?;
        let initial = num(&points[0], "/barycenter_error/mean")?;
        let mut delta = [(initial / p[0][0]).sqrt(), 0.];
        let mut covariance = [[0.; 2]; 2];
        let mut previous_step = 0;
        let increment = noise_factor * d * tr(p, q);
        let floor = increment / (1. - rho);
        for (pi, point) in points.iter().enumerate() {
            let step = point["step"]
                .as_u64()
                .ok_or_else(|| GasError::Configuration("missing integer step".into()))?;
            for _ in previous_step..step {
                delta = [
                    a[0][0] * delta[0] + a[0][1] * delta[1],
                    a[1][0] * delta[0] + a[1][1] * delta[1],
                ];
                let aq = mul(mul(a, covariance), trans(a));
                covariance = std::array::from_fn(|i| std::array::from_fn(|j| aq[i][j] + q[i][j]));
            }
            previous_step = step;
            let deterministic = delta[0] * (p[0][0] * delta[0] + p[0][1] * delta[1])
                + delta[1] * (p[1][0] * delta[0] + p[1][1] * delta[1]);
            let bary_floor = noise_factor * d * tr(p, covariance) / n;
            let transport_upper = deterministic + bary_floor * n;
            let geometric = rho.powf(step as f64) * initial + floor * (1. - rho.powf(step as f64));
            // Exact Gaussian quadratic-form uncertainty, not a noisy estimate from eight pairs.
            let mean_covariance = covariance.map(|row| row.map(|x| noise_factor * x / n));
            let pc = mul(p, mean_covariance);
            let pm = [
                p[0][0] * delta[0] + p[0][1] * delta[1],
                p[1][0] * delta[0] + p[1][1] * delta[1],
            ];
            let cm = [
                mean_covariance[0][0] * pm[0] + mean_covariance[0][1] * pm[1],
                mean_covariance[1][0] * pm[0] + mean_covariance[1][1] * pm[1],
            ];
            let variance = 2. * d * tr(pc, pc) + 4. * (pm[0] * cm[0] + pm[1] * cm[1]);
            let replicates = c["seed_trajectories"].as_array().unwrap().len() as f64;
            let barycenter_standard_error = (variance.max(0.) / replicates).sqrt();
            let point_path = format!("{rp}/ensemble_trajectory/{pi}");
            for (id, observed, bound, relation, se, labels) in [
                (
                    "retained_noise_floor",
                    num(point, "/exact_barycenter_noise_floor")?,
                    bary_floor,
                    "equal",
                    0.,
                    vec!["prop-wasserstein-rate-explicit"],
                ),
                (
                    "retained_exact_barycenter",
                    num(point, "/exact_barycenter_expectation")?,
                    deterministic + bary_floor,
                    "equal",
                    0.,
                    vec!["prop-wasserstein-rate-explicit"],
                ),
                (
                    "retained_n_uniform_envelope",
                    num(point, "/n_independent_geometric_noise_envelope")?,
                    geometric,
                    "equal",
                    0.,
                    vec![
                        "prop-wasserstein-rate-explicit",
                        "lem-convergence-drift-iteration",
                    ],
                ),
                (
                    "measured_barycenter_expectation",
                    num(point, "/barycenter_error/mean")?,
                    deterministic + bary_floor,
                    "equal",
                    barycenter_standard_error,
                    vec!["prop-wasserstein-rate-explicit"],
                ),
                (
                    "measured_transport_exact_upper",
                    num(point, "/alive_error/mean")?,
                    transport_upper,
                    "upper",
                    num(point, "/alive_error/standard_error")?,
                    vec!["prop-wasserstein-rate-explicit"],
                ),
                (
                    "measured_transport_geometric",
                    num(point, "/alive_error/mean")?,
                    geometric,
                    "upper",
                    num(point, "/alive_error/standard_error")?,
                    vec![
                        "prop-wasserstein-rate-explicit",
                        "lem-convergence-drift-iteration",
                    ],
                ),
            ] {
                check(
                    checks,
                    format!("{id}_case{ci}_step{step}"),
                    &labels,
                    &[],
                    observed,
                    bound,
                    relation,
                    6. * se + 1e-10 * (1. + bound.abs()),
                    source_scope,
                    &point_path,
                    json!({"premises":hyp,"uncertainty_units":"independent complete swarm pairs","standard_error":se,"barycenter_gaussian_exact_variance":variance,"barycenter_analytic_standard_error":barycenter_standard_error,"replicates":replicates,"six_standard_errors":true,"transient":deterministic,"barycenter_floor":bary_floor,"population_uniform_noise_floor":floor}),
                );
            }
        }
    }
    Ok(())
}

/// Validate reconstructible Chapter 6 predictions using retained native JSON only.
pub fn validate(reports: &[(String, Value)]) -> Result<Value> {
    let mut comparisons = Vec::new();
    let mut gaps = Vec::new();
    for (path, report) in reports {
        cap_runs(path, report, &mut comparisons)?;
        controls(path, report, &mut comparisons, &mut gaps)?;
    }
    let missing = [
        (
            "def-cemetery-state",
            "Native saved trajectories continue singleton swarms; chapter uses k<2 cemetery. No killed-kernel/QSD certificate is transferred across this distinction.",
        ),
        (
            "thm-convergence-conservative-harris",
            "Same-kernel minorization measure, epsilon, small-set radius and weighted coupling are not saved.",
        ),
        (
            "thm-main-convergence",
            "No two-sided surviving-block bounds c_N,C_N,m,eta_N for the chapter killed kernel; QSD rho_N and alpha_N cannot be inferred from paired empirical transport.",
        ),
        (
            "thm-equilibrium-variance-bounds",
            "No certified QSD eigenmeasure/eigenvalue, integrability or alpha_N>rho(M) survival gap.",
        ),
        (
            "prop-convergence-survival-bound",
            "No retained uniform joint subset-failure product bound or certified QSD safe-set mass a_N; empirical extinction counts do not supply these hypotheses.",
        ),
        (
            "rem-extinction-inevitable",
            "No uniform positive block hazard ell,delta is saved.",
        ),
        (
            "thm-convergence-entropy-to-tv",
            "No actual full-law density, invariant/QSD reference entropy or certified entropy dissipation coefficient.",
        ),
        (
            "thm-convergence-lsi-concentration",
            "No joint-swarm-law LSI rho_N. Interacting rows are not independent replicate units.",
        ),
        (
            "prop-mixing-time-explicit",
            "No certified same-kernel Harris or surviving-block contraction rate; moment decay is not a TV mixing certificate.",
        ),
        (
            "thm-explicit-rate-sensitivity",
            "No differentiable analytic component comparison matrices A_C(p),A_K(p) are saved for selected canonical cases.",
        ),
        (
            "thm-svd-rate-matrix",
            "No certified rate sensitivity matrix; several finite parameter profiles cannot supply its derivatives.",
        ),
        (
            "prop-phase-space-pairing",
            "Chapter formula uses raw Euclidean softmax on a fixed candidate set; retained native law uses squashed features and lacks full candidate edge tables in complete-update history.",
        ),
        (
            "prop-jitter-cloning-coupling",
            "No certificate of the declared scalar model 1-c lambda for competitive native cloning. Jitter profile comparisons cannot establish this model.",
        ),
        (
            "alg-adaptive-tuning",
            "No saved adaptive tuning experiment or declared constrained objective; stored validation reuses fixed runs.",
        ),
    ];
    for (label, reason) in missing {
        gaps.push(json!({"source_labels":[label],"reason":reason,"status":"unavailable_premises"}));
    }
    let covered: BTreeSet<String> = comparisons
        .iter()
        .flat_map(|c| {
            c["source_labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_str().unwrap().to_owned())
        })
        .collect();
    let inventory:Vec<Value>=SOURCE.lines().filter_map(|l|l.strip_prefix(":label: ")).map(|label|json!({"label":label,"status":if covered.contains(label){"partially_exercised"}else{"not_validated_from_retained_runs"},"complete":false})).collect();
    let failed = comparisons.iter().filter(|c| c["passed"] == false).count();
    let evidence = vec![
        json!({"source_labels":["thm-total-rate-explicit"],"source_quotes":["For $h>0$ and $0<r<1$, the corresponding physical-time exponential rate of\nthis moment bound is $-h^{-1}\\log r$."],"scope":"Only physical-time rate conversion is reconstructed; full component matrix not inferred."}),
        json!({"source_labels":["prop-wasserstein-rate-explicit"],"source_quotes":[quote("prop-wasserstein-rate-explicit",1)],"scope":"Complete conservative linear-control transport envelope; b_D=2d tr(PQ) under independent pair noise, zero under common noise, with fixed positive Lyapunov cost."}),
    ];
    Ok(
        json!({"chapter":6,"title":"Convergence, Survival, and Parameter Dependence","source_path":SOURCE_PATH,"scope":"Only existing stored native experiments; no new simulations. This is finite-experiment validation of applicable moment/coupling predictions, not proof of mixing or a QSD.","comparisons":comparisons,"source_evidence":evidence,"gaps":gaps,"applicability_audit":{"label_inventory":inventory,"partially_exercised_labels":covered},"coverage":{"label_inventory":inventory,"partially_exercised_labels":covered,"gaps":gaps,"complete_required_estimates":false},"summary":{"comparisons":comparisons.len(),"comparisons_failed":failed,"complete_required_estimates":false,"new_simulations":0,"reports":reports.len()}}),
    )
}
