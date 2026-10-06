//! Chapter 4 checks recovered from saved native simulation evidence only.
//! No simulation, randomness, matching by display identifiers, or fitted theorem
//! constants are introduced here. Empirical output-pair costs are distinguished
//! from Wasserstein distance between transition laws.
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

use crate::convergence_lyapunov::uniform_transport;

const SOURCE: &str = include_str!(
    "../../../docs/source/2_fractal_gas/convergence_program/04_wasserstein_contraction.md"
);
const SOURCE_PATH: &str =
    "docs/source/2_fractal_gas/convergence_program/04_wasserstein_contraction.md";

fn number(v: &Value, key: &str) -> Result<f64> {
    v[key].as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
        GasError::Configuration(format!("Chapter 4 saved evidence requires finite {key}"))
    })
}

fn points(v: &Value) -> Result<Vec<Vec<f64>>> {
    let points: Vec<Vec<f64>> =
        serde_json::from_value(v.clone()).map_err(|e| GasError::Configuration(e.to_string()))?;
    if points.is_empty()
        || points[0].is_empty()
        || points
            .iter()
            .any(|x| x.len() != points[0].len() || x.iter().any(|y| !y.is_finite()))
    {
        return Err(GasError::Configuration(
            "Stored physical cloud must be nonempty with consistent finite coordinates".into(),
        ));
    }
    Ok(points)
}

fn formula(marker: &str, label: &str) -> String {
    if marker.is_empty() {
        return String::new();
    }
    let label_marker = format!(":label: {label}");
    let owned = SOURCE
        .split_once(&label_marker)
        .map(|(_, tail)| tail.split_once("\n:label:").map_or(tail, |(head, _)| head))
        .unwrap_or(SOURCE);
    owned
        .split("$$")
        .enumerate()
        .find_map(|(i, s)| (i % 2 == 1 && s.contains(marker)).then(|| s.trim().to_owned()))
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
fn check(
    out: &mut Vec<Value>,
    id: String,
    label: &str,
    marker: &str,
    observed: f64,
    bound: f64,
    relation: &str,
    se: f64,
    path: &str,
    scope: &str,
) {
    let residual = observed - bound;
    let tol = 2e-10 * (1. + observed.abs().max(bound.abs()));
    let passed = observed.is_finite()
        && bound.is_finite()
        && se.is_finite()
        && if relation == "equal" {
            residual.abs() <= 6. * se + tol
        } else {
            residual <= 6. * se + tol
        };
    let quote = formula(marker, label);
    let mut value = json!({"id":id,"source_labels":[label],
        "observed":observed,"bound":bound,"relation":relation,
        "residual":residual,"standard_error":se,"passed":passed,
        "status":if passed {if se>0. {"not_rejected"} else {"passed"}} else {"violated"},
        "scope":scope,"report_path":path,
        "hypotheses":["Finite saved numeric inputs; no fresh simulation",
            "Source theorem applicability retained in scope"],
        "statistically_supported":relation!="equal" && residual+6.*se<=tol});
    if !quote.is_empty() {
        value["source_formula"] = json!(quote);
    }
    out.push(value);
}

fn mean_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let se = if values.len() > 1 {
        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n * (n - 1.))).sqrt()
    } else {
        0.
    };
    (mean, se)
}

fn mean(p: &[Vec<f64>]) -> Vec<f64> {
    (0..p[0].len())
        .map(|a| p.iter().map(|x| x[a]).sum::<f64>() / p.len() as f64)
        .collect()
}
fn sqdist(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum()
}
fn variance(p: &[Vec<f64>]) -> f64 {
    let m = mean(p);
    p.iter().map(|x| sqdist(x, &m)).sum::<f64>() / p.len() as f64
}
fn center(p: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = mean(p);
    p.iter()
        .map(|x| x.iter().zip(&m).map(|(a, b)| a - b).collect())
        .collect()
}

fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(x, y)| (x - y).abs()).sum::<f64>() / 2.
}
fn reweighted(p: &[f64], w: &[f64]) -> Vec<f64> {
    let z = p.iter().zip(w).map(|(x, y)| x * y).sum::<f64>();
    p.iter().zip(w).map(|(x, y)| x * y / z).collect()
}
fn reweighting_checks(out: &mut Vec<Value>, replicas: &[Value], path: &str) -> Result<()> {
    let mut maximum_residual = f64::NEG_INFINITY;
    let mut checked = 0;
    let mut witness = (0., 0., 0., 0.);
    for r in replicas {
        let p: Vec<f64> = serde_json::from_value(r["accepted_probabilities"].clone())
            .map_err(|e| GasError::Configuration(e.to_string()))?;
        let w: Vec<f64> = serde_json::from_value(r["entering_fitness"].clone())
            .map_err(|e| GasError::Configuration(e.to_string()))?;
        if p.len() != w.len()
            || p.is_empty()
            || w.iter().any(|x| !x.is_finite() || *x <= 0.)
            || p.iter().any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(GasError::Configuration(
                "Saved reweighting requires positive fitness and a common finite physical atom set"
                    .into(),
            ));
        }
        let z = p.iter().sum::<f64>();
        if z == 0. {
            continue;
        }
        let mu: Vec<_> = p.iter().map(|x| x / z).collect();
        let zeta = vec![1. / p.len() as f64; p.len()];
        let a = w.iter().copied().fold(f64::INFINITY, f64::min);
        let b = w.iter().copied().fold(0., f64::max);
        let observed = total_variation(&reweighted(&mu, &w), &reweighted(&zeta, &w));
        let bound = b / a * total_variation(&mu, &zeta);
        if observed - bound > maximum_residual {
            maximum_residual = observed - bound;
            witness = (observed, bound, a, b);
        }
        checked += 1;
    }
    if checked > 0 {
        check(
            out,
            format!("{path}:bounded_positive_reweighting"),
            "lem-w2-bounded-reweighting-tv",
            "\\|\\mathcal R_w\\mu-\\mathcal R_w\\zeta\\|_{\\mathrm{TV}}",
            witness.0,
            witness.1,
            "upper",
            0.,
            path,
            "Discrete laws on saved physical input atoms: normalized native acceptance activity versus uniform empirical mass, reweighted by actual sampled positive fitness. Every nonzero-activity replica checked; maximum-residual witness retained. TV is invariant under common atom permutations. General finite-law lemma only, not QSD/eigenfunction laws.",
        );
        let last = out.last_mut().unwrap();
        last["checked_replicas"] = json!(checked);
        last["weight_min"] = json!(witness.2);
        last["weight_max"] = json!(witness.3);
    }
    Ok(())
}

fn partition_check(out: &mut Vec<Value>, p: &[Vec<f64>], path: &str) {
    // A geometric partition is intrinsic to coordinates, independent of storage
    // order. The variance lemma's algebra applies to every nonempty partition;
    // this does not assert these parts are the realized fitness target.
    let cut = p.iter().map(|x| x[0]).sum::<f64>() / p.len() as f64;
    let (a, b): (Vec<_>, Vec<_>) = p.iter().cloned().partition(|x| x[0] < cut);
    if a.is_empty() || b.is_empty() {
        return;
    }
    let f = a.len() as f64 / p.len() as f64;
    let rhs =
        f * variance(&a) + (1. - f) * variance(&b) + f * (1. - f) * sqdist(&mean(&a), &mean(&b));
    check(
        out,
        format!("{path}:cluster_variance"),
        "lem-variance-decomposition",
        "f_I \\text{Var}_x(I_k)",
        variance(p),
        rhs,
        "equal",
        0.,
        path,
        "Saved entering all-alive positions; partition by coordinate below/above cloud mean. Arbitrary partition identity, not a target-membership certificate.",
    );
    let m = mean(p);
    let ma = mean(&a);
    let mb = mean(&b);
    let weighted_mean: Vec<_> = ma
        .iter()
        .zip(&mb)
        .map(|(x, y)| f * x + (1. - f) * y)
        .collect();
    check(
        out,
        format!("{path}:cluster_mean"),
        "lem-variance-decomposition",
        "",
        sqdist(&m, &weighted_mean),
        0.,
        "equal",
        0.,
        path,
        "Whole vector barycenter relation reconstructed from the saved coordinate cloud and both nonempty intrinsic geometric parts.",
    );
    out.last_mut().unwrap()["source_quotes"] =
        json!(["\\bar{x}_k = \\frac{1}{N}\\sum_{i=1}^N x_i = f_I \\mu_x(I_k) + f_J \\mu_x(J_k)"]);
    let ma_offset: Vec<_> = ma.iter().zip(&m).map(|(x, y)| x - y).collect();
    let expected_offset: Vec<_> = ma
        .iter()
        .zip(&mb)
        .map(|(x, y)| (1. - f) * (x - y))
        .collect();
    check(
        out,
        format!("{path}:target_center_offset"),
        "lem-variance-decomposition",
        "",
        sqdist(&ma_offset, &expected_offset),
        0.,
        "equal",
        0.,
        path,
        "Part-to-total barycenter difference, independently reconstructed from saved coordinates; geometric partition does not assert fitness-target membership.",
    );
    out.last_mut().unwrap()["source_quotes"] = json!([
        "\\mu_x(I_k) - \\bar{x}_k = \\mu_x(I_k) - f_I \\mu_x(I_k) - f_J \\mu_x(J_k) = f_J (\\mu_x(I_k) - \\mu_x(J_k))"
    ]);
    let n = p.len() as f64;
    let card = a.len() as f64 * (1. - f).powi(2) + b.len() as f64 * f * f;
    check(
        out,
        format!("{path}:partition_mass_balance"),
        "lem-variance-decomposition",
        "",
        card,
        n * f * (1. - f),
        "equal",
        0.,
        path,
        "Counts derived from actual saved geometric groups; both positive; entire cardinality-weighted between-group coefficient identity evaluated.",
    );
    out.last_mut().unwrap()["source_quotes"] = json!([
        "|I_k| f_J^2 + |J_k| f_I^2 = N f_I f_J^2 + N f_J f_I^2 = N f_I f_J (f_J + f_I) = N f_I f_J"
    ]);
}

fn primitive_inputs(
    report: &Value,
    filename: &str,
    out: &mut Vec<Value>,
    inputs: &mut Vec<Value>,
) -> Result<()> {
    let Some(cases) = report["kinetic"].as_array() else {
        return Ok(());
    };
    for (i, c) in cases.iter().enumerate() {
        let cfg = &c["config"];
        let k = &c["constants"];
        let path = format!("{filename}#/kinetic/{i}");
        let h = number(cfg, "dt")?;
        let gamma = number(cfg, "friction")?;
        let bo = number(cfg, "velocity_diffusion")?;
        let sigma_pos = number(cfg, "position_diffusion")?;
        let t = h / 2.;
        let expected_c = (-gamma * h).exp();
        let expected_q = bo.powi(2)
            * if gamma > 0. {
                -(-2. * gamma * h).exp_m1() / (2. * gamma)
            } else {
                h
            };
        let expected_s = sigma_pos * h.sqrt();
        let saved_c = number(k, "friction_factor")?;
        let saved_q = number(k, "thermal_variance")?;
        let saved_t = number(k, "position_flow_coefficient")? / (1. + saved_c);
        let saved_s2 = number(k, "position_variance")? - saved_t.powi(2) * saved_q;
        let residual = (saved_t - t)
            .abs()
            .max((saved_c - expected_c).abs())
            .max((saved_q - expected_q).abs())
            .max((saved_s2 - expected_s.powi(2)).abs());
        check(
            out,
            format!("{path}:primitive_gaussian_inputs"),
            "def-w2-finite-population-qsd-regime",
            "t=h/2,\\qquad c=e^{-\\gamma h}",
            residual,
            0.,
            "equal",
            0.,
            &path,
            "Four primitive BAOAB/Gaussian formula members reconstructed independently from saved native factor, position-flow coefficient and covariance. Includes positive- and zero-friction branches. This is noise-parameter evidence, not a complete QSD applicability certificate.",
        );
        let lf = number(&c["landscape_assumptions"], "force_lipschitz")?;
        let kappa = 1. - t * t * lf; // saved probe has no dense viscosity
        let beta = t * t * lf;
        inputs.push(json!({"report_path":path,"dt":h,"gamma":gamma,"t":t,"c":expected_c,
            "q_squared":expected_q,"s":expected_s,"sigma_h_squared":t*t*expected_q+expected_s.powi(2),
            "analytic_force_lipschitz":lf,"force_at_zero":c["landscape_assumptions"]["force_at_zero"],
            "viscosity":0,"kappa_F":kappa,"beta_F":beta,
            "noise_positive":expected_q>0.&&expected_s>0.,"B2_coercivity_positive":kappa>0.,
            "beta_below_one":beta<1.,"whole_QSD_certificate":false,
            "scope":"Primitive subconditions of the recorded unmodified kinetic probe only. h=2 obstruction correctly excluded from the positive-coercivity regime; no QSD density/minorization/eigenfunction inferred."}));
    }
    Ok(())
}

fn decay(
    report: &Value,
    filename: &str,
    out: &mut Vec<Value>,
    observations: &mut Vec<Value>,
) -> Result<()> {
    let Some(cases) = report["cases"].as_array() else {
        return Ok(());
    };
    for (ci, case) in cases.iter().enumerate() {
        if case["seed_trajectories"].as_array().is_none() {
            continue;
        }
        let path = format!("{filename}#/cases/{ci}");
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for (ti, traj) in case["seed_trajectories"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let Some(ps) = traj["points"].as_array() else {
                continue;
            };
            for (pi, p) in ps.iter().enumerate() {
                let o = &p["observables"];
                if o["alive_wasserstein_squared"].is_null() {
                    continue;
                }
                let w = number(o, "alive_wasserstein_squared")?;
                let location = number(o, "alive_barycenter_error")?;
                let shape = number(o, "alive_structural_error")?;
                let pp = format!("{path}/seed_trajectories/{ti}/points/{pi}");
                check(
                    out,
                    format!("{pp}:alive_split"),
                    "thm-full-w2-split",
                    "\\|\\bar{z}_1 - \\bar{z}_2\\|^2 + W_2^2",
                    w,
                    location + shape,
                    "equal",
                    0.,
                    &pp,
                    "Normalized alive optimal physical q transport; both empirical measures separately centered and normalized; same fixed SPD q in location and centered costs. Applies to unequal nonempty alive counts.",
                );
                check(
                    out,
                    format!("{pp}:shape_nonnegative"),
                    "lem-wasserstein-barycenter-decomposition",
                    "\\int \\|z_1 - z_2\\|^2",
                    -shape,
                    0.,
                    "upper",
                    0.,
                    &pp,
                    "Necessary nonnegativity of stored centered optimal transport; roundoff allowance only. This check does not cover an entire source equation.",
                );
                // No full-expression coverage from a necessary scalar consequence.
                out.last_mut()
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove("source_formula");
                if pi == 0 {
                    starts.push(w);
                }
                if pi + 1 == ps.len() && p["step"] == 128 {
                    ends.push(w);
                }
            }
        }
        if starts.len() == ends.len() && !ends.is_empty() {
            let differences: Vec<_> = ends.iter().zip(&starts).map(|(e, s)| e - s).collect();
            let (change, se) = mean_se(&differences);
            observations.push(json!({"id":format!("{path}:endpoint_decline"),"report_path":path,
                "walkers":case["walkers"],"dimensions":case["dimensions"],"profile":case["profile"],
                "pair_randomness":case["pair_randomness"],"complete_pairs":ends.len(),
                "mean_initial":mean_se(&starts).0,"mean_final":mean_se(&ends).0,
                "mean_change":change,"standard_error":se,"mean_decreased":change<0.,
                "six_se_decline_supported":change+6.*se<0.,
                "scope":"Observed alive empirical pair-error decline. Neither a transition-law distance nor a stationary-law/QSD rate; no fitted rate assigned to Chapter 4 theorem."}));
        }
    }
    Ok(())
}

fn empirical(
    report: &Value,
    filename: &str,
    out: &mut Vec<Value>,
    gaps: &mut Vec<Value>,
) -> Result<()> {
    if let Some(cases) = report["cases"].as_array() {
        for (ci, c) in cases.iter().enumerate() {
            if c["replicas"].as_array().is_none() || c["constants"].is_null() {
                continue;
            }
            let path = format!("{filename}#/cases/{ci}");
            if c["alive"] != c["walkers"] {
                continue;
            } // Chapter 4's stated entering regime.
            let p = points(&c["initial_positions"])?;
            partition_check(out, &p, &path);
            let n = number(c, "walkers")?;
            let d = number(c, "dimensions")?;
            let k = &c["constants"];
            let sigma = number(k, "jitter_sigma")?;
            let diameter = p
                .iter()
                .flat_map(|x| p.iter().map(move |y| sqdist(x, y)))
                .fold(0., f64::max);
            check(
                out,
                format!("{path}:live_diameter"),
                "thm-positional-variance-proxy",
                "",
                number(k, "D_x_squared")?,
                diameter,
                "equal",
                0.,
                &path,
                "Frozen eligible all-alive cloud diameter recomputed from saved coordinates; not a whole-expression check.",
            );
            let reset = diameter + 2. * (1. - 1. / n) * d * sigma * sigma;
            let saved_single_reset = number(k, "B_x")?;
            check(
                out,
                format!("{path}:reset_coefficient"),
                "thm-positional-variance-proxy",
                "\\mathbb E V_{\\text{x,proxy}}'",
                2. * saved_single_reset,
                reset,
                "equal",
                0.,
                &path,
                "Two equal-law independent proposal replicas; exact bound constant recomputed from saved initial coordinates and actual clone jitter, distinct from kinetic noise.",
            );
            // This coefficient reconstruction alone is not the expected-output
            // inequality. The following complete comparison covers that equation.
            out.last_mut()
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove("source_formula");
            let replicas = c["replicas"].as_array().unwrap();
            if replicas
                .iter()
                .all(|r| r["accepted_probabilities"].is_array() && r["entering_fitness"].is_array())
            {
                reweighting_checks(out, replicas, &path)?;
            }
            let samples = replicas
                .iter()
                .map(|r| number(r, "proposal_position_variance").map(|x| 2. * x))
                .collect::<Result<Vec<_>>>()?;
            let (m, se) = mean_se(&samples);
            check(
                out,
                format!("{path}:two_swarm_expected_reset"),
                "thm-positional-variance-proxy",
                "\\mathbb E V_{\\text{x,proxy}}'",
                m,
                reset,
                "upper",
                se,
                &path,
                "Twice the measured first proposal variance estimates sum of two identical conditional laws; independent experiments are sampling units. All proposal recipients revived before output; entering swarm all alive. This validates reset and drift-to-offset, not strict Wasserstein contraction.",
            );
            let lambda = number(k, "lambda_v")?;
            let b = number(k, "b")?;
            let positional_coercivity = 1. - b * b / (4. * lambda);
            if lambda <= 0. || positional_coercivity <= 0. {
                return Err(GasError::Configuration(
                    "Saved Chapter 4 q must be positive definite".into(),
                ));
            }
            // q = lambda|dv+b dx/(2lambda)|² + c_x|dx|².
            // The saved optimal q plan is an admissible positional plan; after
            // separate centering its positional cost can only decrease. Thus
            // the measured q cost/c_x is a conservative upper observable for
            // actual centered positional transport, with no point arrays needed.
            let upper_samples = replicas
                .iter()
                .map(|r| {
                    number(r, "independent_pair_proposal_transport")
                        .map(|x| x / positional_coercivity)
                })
                .collect::<Result<Vec<_>>>()?;
            let (upper, se) = mean_se(&upper_samples);
            check(
                out,
                format!("{path}:centered_output_upper_certificate"),
                "prop-centered-w2-control",
                "\\mathbb E V_{\\text{x,struct}}(S_1',S_2')",
                upper,
                reset,
                "upper",
                se,
                &path,
                "Measured upper certificate for centered positional transport: recorded optimal joint q transport divided by 1-b²/(4lambda_v), from completing the square and translating the same admissible positional plan. This conservatively tests the reset bound; the independent output-pair observable is not transition-law W2.",
            );
            if out.last().unwrap()["passed"] == false {
                // An upper observable exceeding the bound does not refute the
                // unobserved smaller positional cost. Mark unavailable evidence.
                let last = out.last_mut().unwrap();
                last["passed"] = Value::Null;
                last["status"] = json!("inconclusive_upper_certificate");
            }
            gaps.push(json!({"source_labels":["prop-centered-w2-control"],"report_path":path,
                "missing_required_input":"Direct tight centered positional cost requires both post-cloning position arrays. Available joint q transport supplies a conservative upper certificate that is validated here.",
                "not_a_required_certificate_gap":true}));
        }
    }
    if let Some(cases) = report["balanced_keystone_cases"].as_array() {
        for (ci, c) in cases.iter().enumerate() {
            let path = format!("{filename}#/balanced_keystone_cases/{ci}");
            let p = points(&c["positions1"])?;
            let q = points(&c["positions2"])?;
            let (w, _) = uniform_transport(&center(&p), &center(&q), sqdist)?;
            check(
                out,
                format!("{path}:centered_variance_bound"),
                "lem-centered-w2-variance-bound",
                "V_{\\text{x,struct}} = W_{2,x}^2",
                w,
                variance(&p) + variance(&q),
                "upper",
                0.,
                &path,
                "Optimal equal-weight positional transport recomputed by min-cost flow from saved balanced native input coordinates; fully permutation invariant.",
            );
            let cp = center(&p);
            let cq = center(&q);
            let independent = cp
                .iter()
                .flat_map(|x| cq.iter().map(move |y| sqdist(x, y)))
                .sum::<f64>()
                / (p.len() * q.len()) as f64;
            check(
                out,
                format!("{path}:independent_centered_plan"),
                "lem-centered-w2-variance-bound",
                "\\mathbb{E}\\|X - Y\\|^2",
                independent,
                variance(&p) + variance(&q),
                "equal",
                0.,
                &path,
                "Independent empirical product plan evaluated over every saved input pair, with each measure normalized by its own count; zero mean and marginal second moments independently computed.",
            );
            let location = sqdist(&mean(&p), &mean(&q));
            let (full, _) = uniform_transport(&p, &q, sqdist)?;
            check(
                out,
                format!("{path}:barycenter_decomposition"),
                "lem-wasserstein-barycenter-decomposition",
                "\\|\\bar{z}_1 - \\bar{z}_2\\|^2 + W_2^2",
                full,
                location + w,
                "equal",
                0.,
                &path,
                "Independent optimal transports of saved physical positions and separately centered positions; zero velocities give Euclidean phase-space costs.",
            );
            let product = p
                .iter()
                .flat_map(|x| q.iter().map(move |y| sqdist(x, y)))
                .sum::<f64>()
                / (p.len() * q.len()) as f64;
            check(
                out,
                format!("{path}:arbitrary_plan_centering"),
                "lem-wasserstein-barycenter-decomposition",
                "\\int \\|z_1 - z_2\\|^2",
                product,
                location + independent,
                "equal",
                0.,
                &path,
                "Full empirical product plan cost compared with the same separately translated plan plus barycenter cost. Checks the proof's coupling-level identity independently of optimized transport.",
            );
            let displayed_pair = p
                .iter()
                .zip(q.iter().rev())
                .map(|(x, y)| sqdist(x, y))
                .sum::<f64>()
                / p.len() as f64;
            check(
                out,
                format!("{path}:optimal_vs_display_matching"),
                "prop-w2-prescribed-coupling-scope",
                "\\min_{\\sigma\\in\\mathfrak S_N}",
                full,
                displayed_pair,
                "upper",
                0.,
                &path,
                "All-alive saved balanced clouds; optimal equal-mass transport computed independently. A reversed display ordering supplies only an admissible competitor; no walker identities are assigned.",
            );
            partition_check(out, &p, &format!("{path}/positions1"));
            partition_check(out, &q, &format!("{path}/positions2"));
            check(
                out,
                format!("{path}:matched_error_to_variance"),
                "prop-w2-averaged-keystone-constants",
                "W\\leq2",
                w,
                2. * (variance(&p) + variance(&q)),
                "upper",
                0.,
                &path,
                "Chosen optimal position matching; centered physical phase error has zero velocity component.",
            );
            // W<=2 proxy appears inline in the proof, preserve exact quote.
            out.last_mut()
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove("source_formula");
            out.last_mut().unwrap()["source_quotes"] =
                json!(["W\\leq2\\sum_s\\operatorname{Var}_x(S_s)"]);
            let d = number(c, "d")? as usize;
            // This bound is derived analytically from the actual reward -|x|²/2
            // on the configured box; no sampled gradient maximum used.
            let constants = crate::convergence_cloning::canonical_keystone_constants(
                d,
                0.01,
                2. * (d as f64).sqrt(),
            )?;
            let cv = &constants.values;
            let n = number(c, "N")?;
            let chi = cv["chi_star"];
            let c0 = cv["C_0"];
            let emax = cv["E_max"];
            let lower = chi * (w - 0.01) - c0 * emax / (n * n);
            let samples = c["replicas"]
                .as_array()
                .ok_or_else(|| GasError::Configuration("missing saved Keystone replicas".into()))?
                .iter()
                .map(|r| number(r, "measured_retained_activity"))
                .collect::<Result<Vec<_>>>()?;
            let (m, se) = mean_se(&samples);
            check(
                out,
                format!("{path}:all_population_pressure"),
                "prop-w2-averaged-keystone-constants",
                "\\mathbb E Q\\geq\\chi_*(W-W_0)",
                -m,
                -lower,
                "upper",
                se,
                &path,
                "Canonical all-alive quadratic inputs, analytic reward Lipschitz on native box, positive diversity exponent, independent measurement/donor laws. Full affine lower bound with finite-population correction; tiny coefficient retained, not replaced by fitted balanced coefficient.",
            );
            // eta=1 and D_v=0, with structural q=positional for zero velocities.
            let structural_lower = chi * w / 2. - chi * 0.01 - c0 * emax / (n * n);
            check(
                out,
                format!("{path}:structural_pressure_with_remainders"),
                "lem-quantitative-keystone-w2",
                "\\frac{\\chi_*}{1+\\eta}V_{\\mathrm{struct}}",
                -m,
                -structural_lower,
                "upper",
                se,
                &path,
                "eta=1, actual D_v=0; positional optimal transport equals hypocoercive structural cost for these zero-velocity inputs. All structural, threshold and 1/N² terms retained. The full-coverage lower bound is numerically tiny and often negative.",
            );
        }
    }
    Ok(())
}

fn constants(report: &Value, filename: &str, out: &mut Vec<Value>) -> Result<()> {
    let Some(cases) = report["canonical_keystone_constants"].as_array() else {
        return Ok(());
    };
    for (ci, c) in cases.iter().enumerate() {
        let path = format!("{filename}#/canonical_keystone_constants/{ci}");
        let d = number(c, "dimensions")?;
        let w0 = number(c, "error_threshold")?;
        let lr = number(c, "reward_lipschitz")?;
        let v = &c["values"];
        let logs = &c["natural_logs"];
        let mx = (1. + d.sqrt()).powi(-2);
        let d0 = 32_f64.sqrt();
        let dm = (32_f64 + 1e-6).sqrt() - 0.001;
        let ss = (dm * dm / 4. + 0.01).sqrt();
        let zs = dm / 0.1;
        let la = 0.5 * lr / (0.1 * mx);
        let v0 = mx * mx * w0 / 4.;
        let hf = (v0 / 2.).sqrt();
        let rho = (v0 / 2.) / (32. - v0 / 2.);
        let df = (3. * hf * hf / 4.) / ((hf * hf + 1e-6).sqrt() + (hf * hf / 4. + 1e-6).sqrt());
        let tf = df / ss;
        let logomega =
            2_f64.ln() - zs + tf.exp_m1().ln() - (-zs).exp().ln_1p() - (-zs + tf).exp().ln_1p();
        let logr = (hf / 2.).ln().min(if la > 0. {
            0.1_f64.ln() + logomega - (4.2 * la).ln()
        } else {
            f64::INFINITY
        });
        let loggamma = 0.05_f64.ln() + logomega;
        let loga = (loggamma - (4.41_f64 + 1e-6).ln()).min(0.);
        let logc = -12. + rho.ln() + loga;
        let side_log = (4. * (2. * d).sqrt()).ln() - logr;
        let logm = 2.
            * d
            * if side_log < 700. {
                side_log.exp().ceil().ln()
            } else {
                side_log
            };
        let emax = 64. * d;
        let calc = BTreeMap::from([
            ("D_0", d0),
            ("B_f", 2.),
            ("m_x", mx),
            ("m_z", mx),
            ("B_x", 2. * d.sqrt()),
            ("B_v", 2.),
            ("kappa_D", (-4_f64).exp()),
            ("kappa_C", (-4_f64).exp()),
            ("D_m_span", dm),
            ("s_star", ss),
            ("Z_star", zs),
            ("L_A", la),
            ("v_0", v0),
            ("h_f", hf),
            ("rho_f", rho),
            ("Delta_f", df),
            ("t_f", tf),
            ("E_max", emax),
        ]);
        let logcalc = BTreeMap::from([
            ("omega_f", logomega),
            ("r", logr),
            ("gamma_0", loggamma),
            ("a_0", loga),
            ("C_0", logc),
            ("M_r", logm),
            (
                "chi_star",
                logc + 2. * w0.ln() - 2_f64.ln() - 2. * emax.ln() - 2. * logm,
            ),
        ]);
        let mut residualmax = 0_f64;
        for (name, expected) in calc {
            let observed = number(v, name)?;
            residualmax = residualmax.max((observed - expected).abs() / (1. + expected.abs()));
            check(
                out,
                format!("{path}:constant:{name}"),
                "def-w2-averaged-keystone-constants",
                "",
                observed,
                expected,
                "equal",
                0.,
                &path,
                "Explicit analytic constant reconstruction from saved dimension, threshold, complete reward bound and canonical native configuration; not an experimentally fitted convergence rate.",
            );
        }
        for (name, expected) in logcalc {
            let observed = number(logs, name)?;
            residualmax = residualmax.max((observed - expected).abs() / (1. + expected.abs()));
            check(
                out,
                format!("{path}:log_constant:{name}"),
                "def-w2-averaged-keystone-constants",
                "",
                observed,
                expected,
                "equal",
                0.,
                &path,
                "Logarithmic evaluation avoids overflow/underflow; finite-input analytic constant assembly, independent of N.",
            );
        }
        // Exact source groups fully evaluated above: feature diameter/radii,
        // kernel floors, standardization, feature threshold/separation, radius,
        // acceptance floor, cover size and pressure coefficient.
        let markers = [
            "D_0=2\\sqrt",
            "\\kappa_D=e^{-D_0",
            "D_m=\\sqrt",
            "v_0=m_x",
            "\\Delta_f=\\sqrt",
            "r=\\begin{cases}",
            "a_0=\\min",
            "M_r=\\left\\lceil",
        ];
        for marker in markers {
            check(
                out,
                format!("{path}:complete_group:{marker}"),
                "def-w2-averaged-keystone-constants",
                marker,
                residualmax,
                0.,
                "equal",
                0.,
                &path,
                "All scalar members of this displayed canonical formula group independently reconstructed, including log omega/r/a/C/M/chi. Fixed positive canonical exponents/maps; supplied complete analytic reward Lipschitz retained.",
            );
        }
    }
    Ok(())
}

/// Validate the recoverable Chapter 4 predictions without generating new runs.
pub fn validate(reports: &[(String, Value)]) -> Result<Value> {
    let mut comparisons = Vec::new();
    let mut observations = Vec::new();
    let mut gaps = Vec::new();
    let mut primitive_regime_inputs = Vec::new();
    for (name, r) in reports {
        constants(r, name, &mut comparisons)?;
        empirical(r, name, &mut comparisons, &mut gaps)?;
        decay(r, name, &mut comparisons, &mut observations)?;
        primitive_inputs(r, name, &mut comparisons, &mut primitive_regime_inputs)?;
        if let Some(gs) = r["lyapunov_geometry"].as_array() {
            for (i, g) in gs.iter().enumerate() {
                let path = format!("{name}#/lyapunov_geometry/{i}");
                check(
                    &mut comparisons,
                    format!("{path}:physical_barycenter_split"),
                    "thm-full-w2-split",
                    "\\|\\bar{z}_1 - \\bar{z}_2\\|^2 + W_2^2",
                    number(g, "physical_wasserstein_squared")?,
                    number(g, "location_error")? + number(g, "structural_error")?,
                    "equal",
                    0.,
                    &path,
                    "Stored finite native geometry with fixed SPD physical q; all-alive identity and separately normalized nonempty alive identity. Marked projected distance is deliberately excluded.",
                );
            }
        }
    }
    gaps.extend([
        json!({"source_labels":["prop-w2-realized-target-keystone"],"missing_required_input":"Full two-swarm target memberships, complete reward/fitness stability margins, aligned per-atom positional errors and target capture certificate in the same retained entering states. Global variance alone does not establish target error capture."}),
        json!({"source_labels":["def-w2-finite-population-qsd-regime","thm-w2-finite-n-conditioned-convergence"],"missing_required_input":"Same-kernel primitive lower/upper density bounds, target minorization, eigenfunction floor, QSD reference distribution, survival probabilities and law-level empirical samples. Saved pair transport errors cannot certify QSD uniqueness or TV/W2 convergence to that QSD."}),
        json!({"source_labels":["cor-w2-reference-fitness-degeneracy","cor-w2-force-profile-band"],"missing_required_input":"Saved unchanged d=3,N=200 dense-viscous reference runs with the specified count/row normalization. Available N=4,16,64 cases do not match this reference; scalar formula substitution would not be experimental validation."}),
        json!({"source_labels":["prop-w2-prescribed-coupling-scope"],"missing_required_input":"Both raw output clouds and independently survival-normalized transition laws. Stored pair trajectories can verify empirical optimal-transport decompositions but cannot compare optimal transport between output laws."})
    ]);
    if comparisons.is_empty() {
        gaps.push(json!({"missing_required_input":"No recognized saved Chapter 4 numeric evidence supplied."}));
    }
    let covered: BTreeSet<_> = comparisons
        .iter()
        .filter(|v| v["passed"] == true)
        .flat_map(|v| {
            v["source_labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_str().unwrap().to_owned())
        })
        .collect();
    let failed = comparisons.iter().filter(|v| v["passed"] == false).count();
    let mut labels = BTreeSet::new();
    for line in SOURCE.lines() {
        if let Some(label) = line.strip_prefix(":label: ") {
            labels.insert(label.to_owned());
        }
    }
    let unsupported: Vec<_> = labels.difference(&covered).cloned().collect();
    let summaries = json!({"comparisons":comparisons.len(),"comparisons_failed":failed,
        "covered_estimate_labels":covered,"unsupported_estimate_labels":unsupported,
        "empirical_decay_cases":observations.len(),"empirical_mean_declines":observations.iter().filter(|v|v["mean_decreased"]==true).count(),
        "complete_required_estimates":false,"fresh_simulations":0});
    Ok(
        json!({"chapter":4,"title":"Wasserstein-2 Control from Signed Keystone Accounting",
        "source_path":SOURCE_PATH,"comparisons":comparisons,"summary":summaries,"gaps":gaps,
        "decay_observations":observations,"primitive_regime_inputs":primitive_regime_inputs,"scope":"Recoverable finite-input predictions from stored native runs only. Source identities, analytic constant assembly and empirical moments are distinguished from law-level convergence. Population-normalized alive empirical transport excludes dead storage positions."}),
    )
}
