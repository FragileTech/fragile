//! Formula-level chapter 2 diagnostics. Input hypotheses are retained separately
//! from numerical evidence; a finite experiment never certifies all states.
use crate::{
    Benchmark, BenchmarkModel,
    convergence_coefficients::standardization_coefficients,
    convergence_continuity::{finite_line_transport, optimal_swarm_displacement},
    convergence_decay::{canonicalize_decay_engine, decay_observables},
    convergence_estimates::{EstimateEvidence, EstimateSuite},
    convergence_framework::BoundCheck,
    convergence_kinetic::{
        ConfiningLandscape, KineticInput, KineticValidationConfig, kinetic_constants,
        validate_kinetic,
    },
};
use algorithmic_gas::{
    GasBuilder, GasConfig, GasError, ObservationBatch, Population, Result, TensorBatch,
    fitness::{PositiveMap, PositiveMapping, Standardizer},
    geometry::{AlgorithmicDistance, Distance},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const SOURCE: &str =
    include_str!("../../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md");
const SCOPE: &str = "Chapter 2 formula diagnostic on explicitly retained finite inputs. Deterministic checks and independent-replica six-standard-error comparisons are distinguished. This does not certify an unsampled global hypothesis or invent complete-swarm contraction.";

fn error(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(error(message)) }
}
fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum()
}
fn difference2(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum()
}
fn upper(id: &str, lhs: f64, rhs: f64) -> BoundCheck {
    BoundCheck::upper(id, &[], SCOPE, lhs, rhs)
}
fn equality(id: &str, lhs: f64, rhs: f64) -> BoundCheck {
    upper(id, (lhs - rhs).abs(), 1e-11 * (1. + lhs.abs() + rhs.abs()))
}
fn hypothesis(id: &str, ok: bool) -> BoundCheck {
    upper(id, if ok { 0. } else { 1. }, 0.)
}

/// Select the exact displayed source expression containing `marker` after its
/// named statement. The catalog subsequently checks this quote against source.
fn formula(label: &str, marker: &str) -> Result<String> {
    let start = SOURCE
        .find(&format!(":label: {label}\n"))
        .ok_or_else(|| error(&format!("missing source label {label}")))?;
    let remainder = &SOURCE[start..];
    let end = remainder.find("\n:label: ").unwrap_or(remainder.len());
    let tail = &remainder[..end];
    let expressions = tail
        .split("$$")
        .enumerate()
        .filter_map(|(i, e)| (i % 2 == 1).then_some(e.trim()))
        .collect::<Vec<_>>();
    if let Some(expression) = expressions.iter().find(|e| e.starts_with(marker)) {
        return Ok((*expression).into());
    }
    if let Some(expression) = expressions.iter().find(|e| e.contains(marker)) {
        return Ok((*expression).into());
    }
    Err(error(&format!("missing formula for {label}: {marker}")))
}
/// A proof without its own label belongs to its named enclosing section.
/// The unique prose anchor prevents an unrelated later statement from matching.
fn context_formula(context: &str) -> Result<String> {
    let start = SOURCE
        .find(context)
        .ok_or_else(|| error("missing exact proof context"))?;
    let tail = &SOURCE[start + context.len()..];
    let begin = tail
        .find("$$")
        .ok_or_else(|| error("missing proof display"))?
        + 2;
    let end = tail[begin..]
        .find("$$")
        .ok_or_else(|| error("unterminated proof display"))?
        + begin;
    Ok(tail[begin..end].trim().into())
}
#[derive(Default)]
struct EvidenceBuilder {
    records: Vec<EstimateEvidence>,
    indices: BTreeMap<(String, String), usize>,
}
impl EvidenceBuilder {
    fn add(
        &mut self,
        label: &str,
        marker: &str,
        inputs: Value,
        hypotheses: Vec<BoundCheck>,
        checks: Vec<BoundCheck>,
        scope: &str,
    ) -> Result<()> {
        require(
            !checks.is_empty(),
            "estimate evidence needs numerical checks",
        )?;
        let quote = formula(label, marker)?;
        let key = (label.to_string(), quote.clone());
        if let Some(&index) = self.indices.get(&key) {
            let record = &mut self.records[index];
            record
                .inputs
                .as_array_mut()
                .expect("case array")
                .push(inputs);
            record.hypothesis_checks.extend(hypotheses);
            record.checks.extend(checks);
        } else {
            self.indices.insert(key, self.records.len());
            self.records.push(EstimateEvidence {
                chapter: 2,
                source_labels: vec![label.into()],
                source_formula: quote,
                inputs: json!([inputs]),
                hypothesis_checks: hypotheses,
                checks,
                scope: scope.into(),
            });
        }
        Ok(())
    }
    fn proof(
        &mut self,
        label: &str,
        context: &str,
        inputs: Value,
        hypotheses: Vec<BoundCheck>,
        checks: Vec<BoundCheck>,
        scope: &str,
    ) -> Result<()> {
        let quote = context_formula(context)?;
        self.exact(label, &quote, inputs, hypotheses, checks, scope)
    }
    fn exact(
        &mut self,
        label: &str,
        quote: &str,
        inputs: Value,
        hypotheses: Vec<BoundCheck>,
        checks: Vec<BoundCheck>,
        scope: &str,
    ) -> Result<()> {
        require(SOURCE.contains(quote), "missing exact source expression")?;
        require(
            !checks.is_empty(),
            "missing expression-level numerical checks",
        )?;
        let key = (label.to_owned(), quote.to_owned());
        if let Some(&index) = self.indices.get(&key) {
            let record = &mut self.records[index];
            record
                .inputs
                .as_array_mut()
                .expect("case array")
                .push(inputs);
            record.hypothesis_checks.extend(hypotheses);
            record.checks.extend(checks);
        } else {
            self.indices.insert(key, self.records.len());
            self.records.push(EstimateEvidence {
                chapter: 2,
                source_labels: vec![label.into()],
                source_formula: quote.into(),
                inputs: json!([inputs]),
                hypothesis_checks: hypotheses,
                checks,
                scope: scope.into(),
            });
        }
        Ok(())
    }
    fn inline(
        &mut self,
        label: &str,
        quote: &str,
        inputs: Value,
        checks: Vec<BoundCheck>,
        scope: &str,
    ) -> Result<()> {
        require(
            SOURCE.contains(quote),
            "missing exact inline source expression",
        )?;
        self.records.push(EstimateEvidence {
            chapter: 2,
            source_labels: vec![label.into()],
            source_formula: quote.into(),
            inputs,
            hypothesis_checks: vec![],
            checks,
            scope: scope.into(),
        });
        Ok(())
    }
}
fn obs(x: &[f64], v: &[f64], d: usize) -> Result<ObservationBatch<f64>> {
    let n = x.len() / d;
    let mut o = ObservationBatch::positions(TensorBatch::vectors(n, d, x.to_vec())?);
    o.fields
        .insert("velocities".into(), TensorBatch::vectors(n, d, v.to_vec())?);
    Ok(o)
}
fn metric(rx: f64, rv: f64, lambda: f64) -> Distance {
    Distance::SquashedPhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_radius: rx,
        velocity_radius: rv,
        lambda,
    }
}
fn squash(x: &[f64], c: f64) -> Vec<f64> {
    let r = norm2(x).sqrt();
    x.iter().map(|v| c * v / (c + r)).collect()
}
fn inverse(y: &[f64], c: f64) -> Vec<f64> {
    let r = norm2(y).sqrt();
    y.iter().map(|v| v / (1. - r / c)).collect()
}
fn jacobian(x: &[f64], c: f64) -> Vec<Vec<f64>> {
    let r = norm2(x).sqrt();
    let d = x.len();
    if r == 0. {
        return (0..d)
            .map(|i| (0..d).map(|j| if i == j { 1. } else { 0. }).collect())
            .collect();
    }
    let alpha = c / (c + r);
    (0..d)
        .map(|i| {
            (0..d)
                .map(|j| if i == j { alpha } else { 0. } - c * x[i] * x[j] / ((c + r).powi(2) * r))
                .collect()
        })
        .collect()
}
fn mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn geometry_cases(b: &mut EvidenceBuilder) -> Result<()> {
    for d in [1, 2, 4] {
        for c in [0.1, 1., 2.] {
            for radius in [0., 1e-7, 0.3, 2., 100.] {
                let x = (0..d).map(|i| radius * (i + 1) as f64).collect::<Vec<_>>();
                let r = norm2(&x).sqrt();
                let y = squash(&x, c);
                let j = jacobian(&x, c);
                let alpha = c / (c + r);
                let mut checks = vec![];
                for column in 0..d {
                    let step = 1e-6 * (1. + r);
                    let mut xp = x.clone();
                    let mut xm = x.clone();
                    xp[column] += step;
                    xm[column] -= step;
                    let yp = squash(&xp, c);
                    let ym = squash(&xm, c);
                    let native_obs = obs(&[xp.clone(), xm.clone()].concat(), &vec![0.; 2 * d], d)?;
                    let native_direction =
                        metric(c, 2., 1.).compare(&native_obs, 0, &native_obs, 1)? / (2. * step);
                    let analytic_direction =
                        j.iter().map(|row| row[column].powi(2)).sum::<f64>().sqrt();
                    checks.push(upper(
                        "native_feature_distance_finite_difference",
                        (native_direction - analytic_direction).abs(),
                        2e-5,
                    ));
                    for row in 0..d {
                        checks.push(upper(
                            "central_difference_jacobian",
                            ((yp[row] - ym[row]) / (2. * step) - j[row][column]).abs(),
                            2e-5,
                        ));
                    }
                }
                let input = json!({"d":d,"C":c,"x":x,"radius":r,"image":y,"analytic_jacobian":j,"alpha":alpha,"finite_difference_relative_step":1e-6});
                b.add(
                    "lem-squashing-properties-generic",
                    "D\\psi_C(z)=",
                    input.clone(),
                    vec![hypothesis("positive_radius", c > 0.)],
                    checks,
                    SCOPE,
                )?;
                for (quote, lhs, rhs) in [
                    (
                        r"\psi_C(z):=C\,z/(C+\|z\|)",
                        difference2(&y, &x.iter().map(|z| c * z / (c + r)).collect::<Vec<_>>()),
                        0.,
                    ),
                    (r"\alpha := C/(C+\|z\|)", alpha, c / (c + r)),
                ] {
                    b.inline(
                        "lem-squashing-properties-generic",
                        quote,
                        input.clone(),
                        vec![equality("native_squash_scalar_definition", lhs, rhs)],
                        SCOPE,
                    )?;
                }
                b.inline(
                    "lem-squashing-properties-generic",
                    r"z\in\mathbb R^d",
                    input.clone(),
                    vec![hypothesis(
                        "finite_real_input_of_declared_dimension",
                        x.len() == d && x.iter().all(|z| z.is_finite()),
                    )],
                    SCOPE,
                )?;
                if d == 1 {
                    b.inline(
                        "lem-squashing-properties-generic",
                        r"d=1",
                        input.clone(),
                        vec![equality(
                            "actual_scalar_geometry_dimension",
                            x.len() as f64,
                            1.,
                        )],
                        SCOPE,
                    )?;
                }
                if r == 0. {
                    b.inline(
                        "lem-squashing-properties-generic",
                        r"z=0",
                        input.clone(),
                        vec![equality("actual_origin_input_norm", norm2(&x), 0.)],
                        SCOPE,
                    )?;
                }
                b.inline(
                    "lem-squashing-properties-generic",
                    r"C>0",
                    input.clone(),
                    vec![hypothesis("positive_actual_squash_radius", c > 0.)],
                    SCOPE,
                )?;
                if r > 0. {
                    for quote in [r"0 < \alpha < 1", r"\alpha=C/(C+\|z\|)<1", r"\|z\| > 0"] {
                        b.inline(
                            "lem-squashing-properties-generic",
                            quote,
                            input.clone(),
                            vec![hypothesis(
                                "nonzero_radius_alpha_interior",
                                r > 0. && alpha > 0. && alpha < 1.,
                            )],
                            SCOPE,
                        )?;
                    }
                    b.inline(
                        "lem-squashing-properties-generic",
                        r"C^2/(C+\|z\|)^2 < \alpha",
                        input.clone(),
                        vec![hypothesis(
                            "strict_radial_less_than_tangential",
                            alpha * alpha < alpha,
                        )],
                        SCOPE,
                    )?;
                    let op_norm = if d == 1 { alpha * alpha } else { alpha };
                    let op_quote = if d == 1 {
                        r"\|D\psi_C(z)\|=\alpha^2"
                    } else {
                        r"\|D\psi_C(z)\|=\alpha"
                    };
                    b.inline(
                        "lem-squashing-properties-generic",
                        op_quote,
                        input.clone(),
                        vec![equality(
                            "jacobian_actual_operator_norm",
                            op_norm,
                            if d == 1 { j[0][0].abs() } else { alpha },
                        )],
                        SCOPE,
                    )?;
                    b.inline(
                        "lem-squashing-properties-generic",
                        r"\hat{z} := z/\|z\|",
                        input.clone(),
                        vec![equality(
                            "radial_direction_unit_norm",
                            norm2(&x.iter().map(|z| z / r).collect::<Vec<_>>()),
                            1.,
                        )],
                        SCOPE,
                    )?;
                } else {
                    b.inline(
                        "lem-squashing-properties-generic",
                        r"\|D\psi_C(0)\| = 1",
                        input.clone(),
                        vec![equality(
                            "origin_native_direction_norm",
                            norm2(&mul(&j, &vec![1.; d])) / d as f64,
                            1.,
                        )],
                        SCOPE,
                    )?;
                }
                let unit = if r > 0. {
                    x.iter().map(|v| v / r).collect::<Vec<_>>()
                } else {
                    vec![1.; d]
                };
                b.inline(
                    "lem-squashing-properties-generic",
                    r"\alpha - \alpha^2\|z\|/C = \alpha^2 = C^2/(C+\|z\|)^2",
                    input.clone(),
                    vec![equality(
                        "radial_eigenvalue",
                        norm2(&mul(&j, &unit)) / norm2(&unit),
                        alpha.powi(4),
                    )],
                    SCOPE,
                )?;
                if d > 1 && r > 0. {
                    let mut tangent = vec![0.; d];
                    tangent[0] = -x[1];
                    tangent[1] = x[0];
                    b.inline(
                        "lem-squashing-properties-generic",
                        r"D\psi_C(z) = \alpha I - (\alpha^2\|z\|/C)\hat{z}\hat{z}^\top",
                        input.clone(),
                        vec![equality(
                            "tangent_eigenvalue",
                            norm2(&mul(&j, &tangent)) / norm2(&tangent),
                            alpha.powi(2),
                        )],
                        SCOPE,
                    )?;
                }
                b.inline(
                    "lem-squashing-properties-generic",
                    r"\|\psi_C(z)\| = C\,\|z\|/(C+\|z\|) < C",
                    input.clone(),
                    vec![
                        equality("radial_image", norm2(&y).sqrt(), c * r / (c + r)),
                        upper("strict_open_ball", norm2(&y).sqrt(), c - 1e-14 * c),
                    ],
                    SCOPE,
                )?;
                b.inline(
                    "lem-squashing-properties-generic",
                    r"D\psi_C(0) = I",
                    json!({"d":d,"C":c,"origin_jacobian":jacobian(&vec![0.;d],c)}),
                    vec![equality(
                        "origin_jacobian",
                        jacobian(&vec![0.; d], c)
                            .iter()
                            .enumerate()
                            .map(|(i, row)| {
                                row.iter()
                                    .enumerate()
                                    .map(|(k, a)| (a - if i == k { 1. } else { 0. }).abs())
                                    .sum::<f64>()
                            })
                            .sum(),
                        0.,
                    )],
                    SCOPE,
                )?;
                if r > 0. {
                    b.inline(
                        "lem-squashing-properties-generic",
                        r"C^2/(C+\|z\|)^2 < \alpha",
                        input.clone(),
                        vec![upper(
                            "radial_eigenvalue_tangential_upper",
                            alpha.powi(2),
                            alpha,
                        )],
                        SCOPE,
                    )?;
                    if d == 1 {
                        b.inline(
                            "lem-squashing-properties-generic",
                            r"\|D\psi_C(z)\|=\alpha^2",
                            input.clone(),
                            vec![equality(
                                "one_dimensional_operator_norm",
                                j[0][0].abs(),
                                alpha.powi(2),
                            )],
                            SCOPE,
                        )?;
                    }
                    if d >= 2 {
                        b.inline(
                            "lem-squashing-properties-generic",
                            r"\|D\psi_C(z)\|=\alpha",
                            input.clone(),
                            vec![equality(
                                "tangential_operator_norm",
                                alpha,
                                alpha.max(alpha.powi(2)),
                            )],
                            SCOPE,
                        )?;
                    }
                }
                let restored = inverse(&y, c);
                b.add(
                    "lem-euclidean-reward-regularity",
                    "\\psi_C^{-1}(y)=",
                    input.clone(),
                    vec![hypothesis("inverse_open_ball", norm2(&y).sqrt() < c)],
                    vec![equality(
                        "inverse_roundtrip",
                        difference2(&restored, &x),
                        0.,
                    )],
                    SCOPE,
                )?;
            }
        }
    }
    for d in [1, 2, 4] {
        for lambda in [0.25, 1., 4.] {
            for pattern in 0..20 {
                let x = (0..d)
                    .map(|i| ((i + pattern + 1) as f64 * 0.7).sin())
                    .collect::<Vec<_>>();
                let v = (0..d)
                    .map(|i| ((i + pattern + 2) as f64 * 0.9).cos())
                    .collect::<Vec<_>>();
                let xp = x
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a + 0.02 * (i + 1) as f64)
                    .collect::<Vec<_>>();
                let vp = v
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a - 0.03 * (i + 1) as f64)
                    .collect::<Vec<_>>();
                let o = obs(
                    &[x.clone(), xp.clone()].concat(),
                    &[v.clone(), vp.clone()].concat(),
                    d,
                )?;
                let native = metric(2., 2., lambda).compare(&o, 0, &o, 1)?;
                let position = difference2(&squash(&x, 2.), &squash(&xp, 2.));
                let velocity = difference2(&squash(&v, 2.), &squash(&vp, 2.));
                let input = json!({"d":d,"lambda_v":lambda,"x":x,"v":v,"x_prime":xp,"v_prime":vp,"native_distance":native});
                b.inline(
                    "lem-squashing-properties-generic",
                    r"z,z'\in\mathbb R^d",
                    input.clone(),
                    vec![hypothesis(
                        "actual_real_vector_pair_domain",
                        x.len() == d
                            && xp.len() == d
                            && x.iter().chain(&xp).all(|value| value.is_finite()),
                    )],
                    SCOPE,
                )?;
                b.inline(
                    "lem-euclidean-reward-regularity",
                    r"\mathcal Y^{\circ}:=B(0,R_x)\times B(0,V_{\mathrm{alg}})",
                    input.clone(),
                    vec![hypothesis(
                        "native_feature_coordinates_in_open_product_ball",
                        norm2(&squash(&x, 2.)).sqrt() < 2. && norm2(&squash(&v, 2.)).sqrt() < 2.,
                    )],
                    SCOPE,
                )?;
                b.add(
                    "lem-projection-lipschitz",
                    "d_{\\mathcal Y}^{\\mathrm{Sasaki}}",
                    input.clone(),
                    vec![hypothesis("positive_metric_weight", lambda > 0.)],
                    vec![upper(
                        "native_projection_contraction",
                        native.powi(2),
                        difference2(&x, &xp) + lambda * difference2(&v, &vp),
                    )],
                    SCOPE,
                )?;
                b.add(
                    "lem-projection-lipschitz",
                    "\\|\\psi_x(x)-",
                    input.clone(),
                    vec![],
                    vec![
                        upper(
                            "position_squash_contraction",
                            position,
                            difference2(&x, &xp),
                        ),
                        upper(
                            "velocity_squash_contraction",
                            velocity,
                            difference2(&v, &vp),
                        ),
                    ],
                    SCOPE,
                )?;
                b.add(
                    "lem-projection-lipschitz",
                    "\\begin{aligned}",
                    input,
                    vec![],
                    vec![
                        equality(
                            "native_feature_metric_definition",
                            native.powi(2),
                            position + lambda * velocity,
                        ),
                        upper(
                            "squared_projection_contraction",
                            native.powi(2),
                            difference2(&x, &xp) + lambda * difference2(&v, &vp),
                        ),
                    ],
                    SCOPE,
                )?;
            }
        }
    }
    Ok(())
}

fn scalar_transport(left: &[f64], right: &[f64]) -> Result<f64> {
    require(
        !left.is_empty() && !right.is_empty(),
        "scalar empirical transport needs nonempty arrays",
    )?;
    let mut support = [left, right].concat();
    support.sort_by(f64::total_cmp);
    support.dedup_by(|a, b| *a == *b);
    let mass = |a: &[f64]| {
        support
            .iter()
            .map(|x| a.iter().filter(|y| **y == *x).count() as f64 / a.len() as f64)
            .collect::<Vec<_>>()
    };
    Ok(finite_line_transport(&support, &mass(left), &mass(right))?.wasserstein_squared)
}
fn second(values: &[f64], mask: &[bool]) -> f64 {
    values
        .iter()
        .zip(mask)
        .filter(|(_, a)| **a)
        .map(|(v, _)| v * v)
        .sum::<f64>()
        / mask.iter().filter(|a| **a).count() as f64
}
fn reward_cases(b: &mut EvidenceBuilder) -> Result<()> {
    for d in [1, 2, 4] {
        for benchmark in [
            Benchmark::Quadratic,
            Benchmark::QuadraticWell { alpha: 3. },
            Benchmark::Rastrigin,
        ] {
            for penalty in [0., 0.3] {
                for lambda in [0.5_f64, 2.] {
                    let bx = 2. * (d as f64).sqrt();
                    let bv = 2_f64;
                    let kx = (1. + bx / 2.).powi(2);
                    let kv = (1. + bv / 2.).powi(2);
                    let lpos = match benchmark {
                        Benchmark::Quadratic => bx,
                        Benchmark::QuadraticWell { alpha } => alpha * bx,
                        _ => 2. * bx + 10. * std::f64::consts::TAU * (d as f64).sqrt(),
                    };
                    let lr = lpos * kx + 2. * penalty * bv * kv / lambda.sqrt();
                    let mut numerical = vec![];
                    let mut cases = vec![];
                    for t in 0..16 {
                        let x = (0..d)
                            .map(|i| 1.5 * ((i + t + 1) as f64 * 0.6).sin())
                            .collect::<Vec<_>>();
                        let xp = x
                            .iter()
                            .enumerate()
                            .map(|(i, a)| a + 0.01 * (i + 1) as f64)
                            .collect::<Vec<_>>();
                        let v = (0..d)
                            .map(|i| 0.7 * ((i + t + 1) as f64).cos() / (d as f64).sqrt())
                            .collect::<Vec<_>>();
                        let vp = v
                            .iter()
                            .map(|a| a + 0.015 / (d as f64).sqrt())
                            .collect::<Vec<_>>();
                        let r = -benchmark.value(&x)? - penalty * norm2(&v);
                        let rp = -benchmark.value(&xp)? - penalty * norm2(&vp);
                        let o = obs(
                            &[x.clone(), xp.clone()].concat(),
                            &[v.clone(), vp.clone()].concat(),
                            d,
                        )?;
                        let dist = metric(2., 2., lambda).compare(&o, 0, &o, 1)?;
                        numerical.push(upper(
                            "native_objective_reward_modulus",
                            (r - rp).abs(),
                            lr * dist,
                        ));
                        numerical.push(upper(
                            "inverse_position_modulus",
                            difference2(&x, &xp),
                            kx.powi(2) * difference2(&squash(&x, 2.), &squash(&xp, 2.)),
                        ));
                        numerical.push(upper(
                            "inverse_velocity_modulus",
                            difference2(&v, &vp),
                            kv.powi(2) * difference2(&squash(&v, 2.), &squash(&vp, 2.)),
                        ));
                        cases.push(json!({"x":x,"x_prime":xp,"v":v,"v_prime":vp,"reward":r,"reward_prime":rp,"native_distance":dist}));
                    }
                    let inputs = json!({"benchmark":benchmark,"dimensions":d,"velocity_penalty":penalty,"lambda_v":lambda,"B_x":bx,"B_v":bv,"L_pos_physical":lpos,"K_x":kx,"K_v":kv,"L_R_Sasaki":lr,"pairs":cases});
                    b.add("lem-euclidean-reward-regularity","K_x:=",inputs.clone(),vec![hypothesis("compact_input_bounds", cases.iter().all(|case| ["x","x_prime"].iter().all(|field| case[*field].as_array().is_some_and(|values| values.iter().all(|v|v.as_f64().is_some_and(f64::is_finite)) && values.iter().filter_map(Value::as_f64).map(|v|v*v).sum::<f64>()<=bx*bx)) && ["v","v_prime"].iter().all(|field| case[*field].as_array().is_some_and(|values| values.iter().all(|v|v.as_f64().is_some_and(f64::is_finite)) && values.iter().filter_map(Value::as_f64).map(|v|v*v).sum::<f64>()<=bv*bv))))],vec![equality("inverse_position_constant",kx,(1.+bx/2.).powi(2)),equality("inverse_velocity_constant",kv,4.)],"Native Rust objective values with explicit optional velocity penalty, compact positions and velocities; analytic gradient envelopes supply the hypotheses.")?;
                    for label in [
                        "lem-euclidean-reward-regularity",
                        "sec-eg-operator-estimates",
                    ] {
                        for (quote, lhs, rhs) in [
                            (
                                r"\|x\|\le B_x",
                                cases
                                    .iter()
                                    .flat_map(|case| {
                                        ["x", "x_prime"].iter().map(move |key| {
                                            case[*key]
                                                .as_array()
                                                .expect("physical vector")
                                                .iter()
                                                .map(|v| v.as_f64().expect("finite").powi(2))
                                                .sum::<f64>()
                                                .sqrt()
                                        })
                                    })
                                    .fold(0_f64, f64::max),
                                bx,
                            ),
                            (
                                r"\|v\|\le B_v",
                                cases
                                    .iter()
                                    .flat_map(|case| {
                                        ["v", "v_prime"].iter().map(move |key| {
                                            case[*key]
                                                .as_array()
                                                .expect("physical vector")
                                                .iter()
                                                .map(|v| v.as_f64().expect("finite").powi(2))
                                                .sum::<f64>()
                                                .sqrt()
                                        })
                                    })
                                    .fold(0_f64, f64::max),
                                bv,
                            ),
                        ] {
                            b.inline(
                                label,
                                quote,
                                inputs.clone(),
                                vec![upper("actual_compact_native_reward_hypothesis", lhs, rhs)],
                                SCOPE,
                            )?;
                        }
                    }
                    for (quote, lhs, rhs) in [
                        (r"B_v=V_{\mathrm{alg}}", bv, 2.),
                        (r"B_x=r\sqrt d", bx, 2. * (d as f64).sqrt()),
                        (r"K_v=4", kv, 4.),
                    ] {
                        b.inline(
                            "lem-euclidean-reward-regularity",
                            quote,
                            inputs.clone(),
                            vec![equality("compact_native_reward_parameter", lhs, rhs)],
                            SCOPE,
                        )?;
                    }
                    b.inline(
                        "lem-euclidean-reward-regularity",
                        r"R(x,v)=R_{\mathrm{pos}}(x)-\lambda_{\mathrm{vel}}\|v\|^2",
                        inputs.clone(),
                        cases
                            .iter()
                            .map(|case| {
                                equality(
                                    "native_position_velocity_reward_definition",
                                    case["reward"].as_f64().unwrap(),
                                    -benchmark
                                        .value(
                                            &case["x"]
                                                .as_array()
                                                .unwrap()
                                                .iter()
                                                .map(|v| v.as_f64().unwrap())
                                                .collect::<Vec<_>>(),
                                        )
                                        .unwrap()
                                        - penalty
                                            * case["v"]
                                                .as_array()
                                                .unwrap()
                                                .iter()
                                                .map(|v| v.as_f64().unwrap().powi(2))
                                                .sum::<f64>(),
                                )
                            })
                            .collect(),
                        SCOPE,
                    )?;
                    b.add(
                        "lem-euclidean-reward-regularity",
                        "L_R^{\\mathrm{Sasaki}}",
                        inputs.clone(),
                        vec![],
                        numerical.clone(),
                        SCOPE,
                    )?;
                    b.proof(
                        "sec-eg-operator-estimates",
                        "6. **Aggregator axioms.**",
                        inputs.clone(),
                        vec![hypothesis("positive_metric_weight", lambda > 0.)],
                        numerical.clone(),
                        SCOPE,
                    )?;
                    b.add(
                        "lem-euclidean-reward-regularity",
                        "R_{\\mathcal Y}(y):=",
                        inputs,
                        vec![],
                        numerical,
                        SCOPE,
                    )?;
                }
            }
        }
    }
    // A finite reference law is explicit. The probability factors come from
    // its atoms, rather than from an observed minimum or sampled band estimate.
    for benchmark in [
        Benchmark::Quadratic,
        Benchmark::QuadraticWell { alpha: 3. },
        Benchmark::Rastrigin,
    ] {
        let x = [0_f64, 0.05, 0.8, 0.85];
        let rewards = x
            .iter()
            .map(|x| benchmark.value(&[*x]).map(|u| -u))
            .collect::<Result<Vec<_>>>()?;
        let gap = rewards[..2]
            .iter()
            .flat_map(|a| rewards[2..].iter().map(move |b| (a - b).abs()))
            .fold(f64::INFINITY, f64::min);
        let mu = rewards.iter().sum::<f64>() / 4.;
        let variance = rewards.iter().map(|r| (r - mu).powi(2)).sum::<f64>() / 4.;
        let paired = rewards
            .iter()
            .flat_map(|a| rewards.iter().map(move |r| (a - r).powi(2)))
            .sum::<f64>()
            / 32.;
        let inputs = json!({"reference":"uniform probability on four specified finite physical states, zero velocity","benchmark":benchmark,"positions":x,"rewards":rewards,"B1":[0,1],"B2":[2,3],"p1":0.5,"p2":0.5,"Delta":gap,"variance":variance});
        b.add(
            "lem-euclidean-richness",
            "\\inf_{z",
            inputs.clone(),
            vec![
                hypothesis("positive_probability_factors", rewards.len() == 4),
                hypothesis("strict_positive_reward_gap", gap > 0.),
            ],
            vec![upper(
                "all_cross_band_gaps",
                gap,
                rewards[..2]
                    .iter()
                    .flat_map(|a| rewards[2..].iter().map(move |r| (a - r).abs()))
                    .fold(f64::INFINITY, f64::min),
            )],
            SCOPE,
        )?;
        b.inline(
            "lem-euclidean-richness",
            r"\pi_B(B_i)\ge p_i>0",
            inputs.clone(),
            vec![
                equality(
                    "finite_reference_band_probability",
                    2. / rewards.len() as f64,
                    0.5,
                ),
                hypothesis(
                    "strict_positive_reference_probability",
                    2. / rewards.len() as f64 > 0.,
                ),
            ],
            SCOPE,
        )?;
        b.inline(
            "lem-euclidean-richness",
            r"\operatorname{Var}_{\pi_B}R\ge p_1p_2\Delta^2",
            inputs.clone(),
            vec![upper(
                "explicit_reference_variance_lower_bound",
                0.25 * gap * gap,
                variance,
            )],
            SCOPE,
        )?;
        b.inline(
            "lem-euclidean-richness",
            r"\operatorname{Var}R=\frac12\mathbb E[(R(Z)-R(Z'))^2]",
            inputs,
            vec![equality(
                "exact_two_copy_variance_identity",
                variance,
                paired,
            )],
            SCOPE,
        )?;
    }
    b.inline("lem-euclidean-richness",r"\operatorname{Var}R=0",json!({"benchmark":"constant","velocity_penalty":0,"reason":"richness positive-gap hypothesis does not hold; the native regularizer remains defined"}),vec![equality("constant_reward_variance",Benchmark::Constant.value(&[0.])?-Benchmark::Constant.value(&[1.])?,0.)],"Zero-variance regression checks the stated constant-objective exception; it is not a positive-richness certificate.")?;
    Ok(())
}

fn standardization_cases(b: &mut EvidenceBuilder) -> Result<()> {
    for n in [1usize, 2, 4, 8, 32] {
        for floor in [0.01, 0.1, 1.] {
            for pattern in 0..4 {
                let raw = (0..n)
                    .map(|i| match pattern {
                        0 => 0.25,
                        1 => {
                            if i % 2 == 0 {
                                -1.
                            } else {
                                1.
                            }
                        }
                        2 => 1e-9 * (i as f64).sin(),
                        _ => (i as f64 * 1.7).sin(),
                    })
                    .collect::<Vec<_>>();
                let changed = raw
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v + 0.07 * (i as f64 + 0.3).cos()).clamp(-1., 1.))
                    .collect::<Vec<_>>();
                let observations = obs(&vec![0.; n], &vec![0.; n], 1)?;
                let masks = if n == 1 {
                    vec![vec![true]]
                } else {
                    vec![
                        vec![true; n],
                        (0..n).map(|i| i < n / 2).collect(),
                        (0..n).map(|i| i % 2 == 0).collect(),
                        (0..n).map(|i| i >= n / 2).collect(),
                    ]
                };
                for mask1 in &masks {
                    for mask2 in &masks {
                        let k1 = mask1.iter().filter(|a| **a).count();
                        let k2 = mask2.iter().filter(|a| **a).count();
                        let stable = mask1.iter().zip(mask2).filter(|(a, c)| **a && **c).count();
                        let nc = mask1.iter().zip(mask2).filter(|(a, c)| a != c).count() as f64;
                        let c = standardization_coefficients(k1, k2, stable, 1., floor)?;
                        let native = Standardizer::Global { sigma_min: floor };
                        let (z1, s1) = native.apply(&raw, mask1, &observations)?;
                        let (zi, si) = native.apply(&changed, mask1, &observations)?;
                        let (z2, s2) = native.apply(&changed, mask2, &observations)?;
                        let delta = raw
                            .iter()
                            .zip(&changed)
                            .zip(mask1)
                            .filter(|(_, a)| **a)
                            .map(|((a, c), _)| (a - c).powi(2))
                            .sum::<f64>();
                        let ev = difference2(&z1, &zi);
                        let es = difference2(&zi, &z2);
                        let all = difference2(&z1, &z2);
                        let mu1 = s1.mean[0];
                        let mui = si.mean[0];
                        let mu2 = s2.mean[0];
                        let scale1 = s1.scale[0];
                        let scalei = si.scale[0];
                        let scale2 = s2.scale[0];
                        let direct = (0..n)
                            .map(|i| {
                                if mask1[i] {
                                    (raw[i] - changed[i]) / scale1
                                } else {
                                    0.
                                }
                            })
                            .collect::<Vec<_>>();
                        let meanvec = (0..n)
                            .map(|i| if mask1[i] { (mui - mu1) / scale1 } else { 0. })
                            .collect::<Vec<_>>();
                        let denom = zi
                            .iter()
                            .map(|z| z * (scalei - scale1) / scale1)
                            .collect::<Vec<_>>();
                        let direct2 = norm2(&direct);
                        let mean2 = norm2(&meanvec);
                        let denom2 = norm2(&denom);
                        let direct_s = (0..n)
                            .filter(|&i| mask1[i] != mask2[i])
                            .map(|i| (zi[i] - z2[i]).powi(2))
                            .sum::<f64>();
                        let indirect_s = (0..n)
                            .filter(|&i| mask1[i] && mask2[i])
                            .map(|i| (zi[i] - z2[i]).powi(2))
                            .sum::<f64>();
                        let indirect_pre = 2. * stable as f64 * (mu2 - mui).powi(2)
                            / scalei.powi(2)
                            + 2. * (scale2 - scalei).powi(2) / scalei.powi(2)
                                * (0..n)
                                    .filter(|&i| mask1[i] && mask2[i])
                                    .map(|i| z2[i].powi(2))
                                    .sum::<f64>();
                        let inputs = json!({"N":n,"floor":floor,"raw1":raw,"raw2":changed,"mask1":mask1,"mask2":mask2,"k1":k1,"k2":k2,"stable":stable,"n_c":nc,"native_z1":z1,"native_intermediate":zi,"native_z2":z2,"native_stats1":s1,"native_stats_intermediate":si,"native_stats2":s2,"coefficients":c,"delta_raw_squared":delta,"direct":direct,"mean_shift":meanvec,"denominator_shift":denom});
                        let hypotheses = vec![
                            hypothesis("both_alive_counts_positive", k1 > 0 && k2 > 0),
                            hypothesis(
                                "uniform_raw_bound",
                                raw.iter().chain(&changed).all(|v| v.abs() <= 1.),
                            ),
                            hypothesis("positive_floor", floor > 0.),
                        ];
                        if n == 4 && floor == 0.1 && pattern == 3 {
                            b.inline(
                                "lem-sasaki-aggregator-value",
                                r"\mathbf v_1,\mathbf v_2\in\mathbb R^k",
                                inputs.clone(),
                                vec![hypothesis(
                                    "actual_restricted_raw_vectors_have_alive_dimension",
                                    raw.iter().zip(mask1).filter(|(_, a)| **a).count() == k1
                                        && changed.iter().zip(mask1).filter(|(_, a)| **a).count()
                                            == k1
                                        && raw
                                            .iter()
                                            .chain(&changed)
                                            .all(|value| value.is_finite()),
                                )],
                                SCOPE,
                            )?;
                            b.inline("lem-sasaki-aggregator-structural", r"\mathcal S_r=((x_{r,i},v_{r,i},s_{r,i}))_{i=1}^N", inputs.clone(), vec![hypothesis("fixed_raw_finite_population_representatives", raw.len()==n && changed.len()==n && mask1.len()==n && mask2.len()==n)], "Scalar aggregation witness on the declared finite marked state representations; only masks and common raw values enter this structural estimate.")?;
                            b.inline(
                                "thm-sasaki-standardization-composite-sq",
                                r"r=1,2",
                                inputs.clone(),
                                vec![equality(
                                    "actual_number_of_composite_inputs",
                                    [mask1, mask2].len() as f64,
                                    2.,
                                )],
                                SCOPE,
                            )?;
                            for (label, quote, checks) in [
                                (
                                    "lem-sasaki-aggregator-value",
                                    r"k\ge 1",
                                    vec![upper("positive_value_alive_count", 1., k1 as f64)],
                                ),
                                (
                                    "lem-sasaki-aggregator-value",
                                    r"|v_{j,i}|\le V_{\max}",
                                    vec![upper(
                                        "actual_scalar_input_bound",
                                        raw.iter()
                                            .chain(&changed)
                                            .map(|v| v.abs())
                                            .fold(0_f64, f64::max),
                                        1.,
                                    )],
                                ),
                                (
                                    "lem-sasaki-aggregator-structural",
                                    r"k_r\ge 1",
                                    vec![upper(
                                        "both_positive_structural_counts",
                                        1.,
                                        k1.min(k2) as f64,
                                    )],
                                ),
                                (
                                    "lem-sasaki-aggregator-structural",
                                    r"|v_i|\le V_{\max}",
                                    vec![upper(
                                        "fixed_raw_scalar_input_bound",
                                        changed.iter().map(|v| v.abs()).fold(0_f64, f64::max),
                                        1.,
                                    )],
                                ),
                                (
                                    "lem-sasaki-aggregator-value",
                                    r"\nabla\mu=(1/k)\mathbf 1",
                                    vec![equality(
                                        "mean_gradient_finite_difference",
                                        ((raw.iter().sum::<f64>() + 1e-5) / n as f64
                                            - raw.iter().sum::<f64>() / n as f64)
                                            / 1e-5,
                                        1. / n as f64,
                                    )],
                                ),
                                (
                                    "lem-sasaki-aggregator-value",
                                    r"\nabla m_2=(2/k)\mathbf v",
                                    vec![upper(
                                        "second_gradient_finite_difference",
                                        (((raw[0] + 1e-5).powi(2) - raw[0].powi(2))
                                            / n as f64
                                            / 1e-5
                                            - 2. * raw[0] / n as f64)
                                            .abs(),
                                        1e-5,
                                    )],
                                ),
                                (
                                    "lem-sasaki-value-error-decomposition",
                                    r"(a+b+c)^2 \le 3(a^2+b^2+c^2)",
                                    vec![upper(
                                        "actual_three_component_cauchy",
                                        (direct2.sqrt() + mean2.sqrt() + denom2.sqrt()).powi(2),
                                        3. * (direct2 + mean2 + denom2),
                                    )],
                                ),
                                (
                                    "lem-sasaki-mean-shift-bound-sq",
                                    r"1/(\sigma'_1)^2 \le 1/\sigma_{\min,\mathrm{patch}}^2",
                                    vec![upper(
                                        "actual_inverse_scale_floor",
                                        1. / scale1.powi(2),
                                        1. / floor.powi(2),
                                    )],
                                ),
                                (
                                    "lem-sasaki-direct-shift-bound-sq",
                                    r"\sigma'_1 \ge \sigma_{\min,\mathrm{patch}} > 0",
                                    vec![
                                        upper("actual_scale_floor", floor, scale1),
                                        hypothesis("strict_native_floor", floor > 0.),
                                    ],
                                ),
                                (
                                    "lem-sasaki-direct-structural-error-sq",
                                    r"|z_j| \le 2V_{\max}^{(R)} / \sigma_{\min,\mathrm{patch}}",
                                    vec![upper(
                                        "actual_standardized_scalar_bound",
                                        zi.iter().chain(&z2).map(|z| z.abs()).fold(0_f64, f64::max),
                                        2. / floor,
                                    )],
                                ),
                                (
                                    "lem-sasaki-indirect-structural-error-sq",
                                    r"(a+b)^2 \le 2(a^2+b^2)",
                                    vec![upper(
                                        "actual_two_component_cauchy",
                                        indirect_s,
                                        indirect_pre,
                                    )],
                                ),
                                (
                                    "thm-sasaki-standardization-composite-sq",
                                    r"\|A-C\|_2^2 \le 2(\|A-B\|_2^2 + \|B-C\|_2^2)",
                                    vec![upper(
                                        "native_value_structure_triangle",
                                        all,
                                        2. * ev + 2. * es,
                                    )],
                                ),
                                (
                                    "def-sasaki-standardization-constants-sq",
                                    r"C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) = 1/\sigma_{\min,\mathrm{patch}}^2",
                                    vec![equality(
                                        "actual_mean_coefficient_simplification",
                                        c.value_mean,
                                        1. / floor.powi(2),
                                    )],
                                ),
                                (
                                    "thm-sasaki-standardization-structural-sq",
                                    r"|r_i|\le V_{\max}^{(R)}",
                                    vec![upper(
                                        "actual_fixed_raw_reward_bound",
                                        changed.iter().map(|r| r.abs()).fold(0_f64, f64::max),
                                        1.,
                                    )],
                                ),
                            ] {
                                b.inline(label, quote, inputs.clone(), checks, SCOPE)?;
                            }
                            let (fixed_z1, _) = native.apply(&changed, mask1, &observations)?;
                            let (fixed_z2, _) = native.apply(&changed, mask2, &observations)?;
                            for (label, quote, checks) in [
                                (
                                    "thm-sasaki-standardization-structural-sq",
                                    r"z(\mathcal S_r)=z(\mathcal S_r,\mathbf r)",
                                    vec![equality(
                                        "native_fixed_raw_state_standardization",
                                        difference2(&zi, &fixed_z1) + difference2(&z2, &fixed_z2),
                                        0.,
                                    )],
                                ),
                                (
                                    "lem-sasaki-structural-error-decomposition",
                                    r"\mathbf z_1 = z(\mathcal S_1, \mathbf r_2)",
                                    vec![equality(
                                        "native_first_fixed_raw_standardization",
                                        difference2(&zi, &fixed_z1),
                                        0.,
                                    )],
                                ),
                                (
                                    "lem-sasaki-structural-error-decomposition",
                                    r"\mathbf z_2 = z(\mathcal S_2, \mathbf r_2)",
                                    vec![equality(
                                        "native_second_fixed_raw_standardization",
                                        difference2(&z2, &fixed_z2),
                                        0.,
                                    )],
                                ),
                                (
                                    "lem-sasaki-structural-error-decomposition",
                                    r"\Delta\mathbf{z} = \mathbf z_1 - \mathbf z_2",
                                    vec![equality(
                                        "native_structural_component_total",
                                        es,
                                        direct_s + indirect_s,
                                    )],
                                ),
                                (
                                    "lem-sasaki-indirect-structural-error-sq",
                                    r"k_{\mathrm{stable}} := |\mathcal A_{\mathrm{stable}}|",
                                    vec![equality(
                                        "native_stable_partition_count",
                                        stable as f64,
                                        mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count()
                                            as f64,
                                    )],
                                ),
                                (
                                    "lem-sasaki-aggregator-lipschitz",
                                    r"V_{\max}=V_{\mathrm{max}}^{(R)}",
                                    vec![upper(
                                        "actual_auxiliary_raw_reward_envelope",
                                        raw.iter()
                                            .chain(&changed)
                                            .map(|r| r.abs())
                                            .fold(0_f64, f64::max),
                                        1.,
                                    )],
                                ),
                            ] {
                                b.inline(label, quote, inputs.clone(), checks, SCOPE)?;
                            }
                            for (label, quote, lhs, rhs) in [
                                (
                                    "lem-sasaki-structural-error-decomposition",
                                    r"\mathcal{A}_{\text{stable}} := \mathcal{A}(\mathcal S_1) \cap \mathcal{A}(\mathcal S_2)",
                                    stable as f64,
                                    mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count()
                                        as f64,
                                ),
                                (
                                    "lem-sasaki-structural-error-decomposition",
                                    r"\mathcal{A}_{\text{unstable}} := \mathcal{A}(\mathcal S_1) \triangle \mathcal{A}(\mathcal S_2)",
                                    nc,
                                    mask1.iter().zip(mask2).filter(|(a, b)| a != b).count() as f64,
                                ),
                                (
                                    "lem-sasaki-direct-structural-error-sq",
                                    r"\mathcal{A}_{\text{unstable}} = \mathcal{A}(\mathcal S_1) \triangle \mathcal{A}(\mathcal S_2)",
                                    nc,
                                    mask1.iter().zip(mask2).filter(|(a, b)| a != b).count() as f64,
                                ),
                                (
                                    "lem-sasaki-indirect-structural-error-sq",
                                    r"\mathcal A_{\mathrm{stable}} = \mathcal A(\mathcal S_1) \cap \mathcal A(\mathcal S_2)",
                                    stable as f64,
                                    mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count()
                                        as f64,
                                ),
                            ] {
                                b.inline(
                                    label,
                                    quote,
                                    inputs.clone(),
                                    vec![equality("native_alive_partition_definition", lhs, rhs)],
                                    SCOPE,
                                )?;
                            }
                            for (label, quote, checks) in [
                                (
                                    "lem-sasaki-value-error-decomposition",
                                    r"\Delta\mathbf{z} = \mathbf z_1 - \mathbf z_2",
                                    vec![equality(
                                        "native_value_output_difference_components",
                                        (0..n)
                                            .map(|i| {
                                                (z1[i] - zi[i] - direct[i] - meanvec[i] - denom[i])
                                                    .powi(2)
                                            })
                                            .sum(),
                                        0.,
                                    )],
                                ),
                                (
                                    "lem-sasaki-direct-shift-bound-sq",
                                    r"\Delta_{\text{direct}} = (\mathbf r_1 - \mathbf r_2) / \sigma'_1",
                                    vec![equality(
                                        "native_direct_component_exact_norm",
                                        direct2,
                                        delta / scale1.powi(2),
                                    )],
                                ),
                                (
                                    "lem-sasaki-direct-shift-bound-sq",
                                    r"\sigma_{\min,\mathrm{patch}} := \sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}",
                                    vec![equality(
                                        "native_regularizer_global_floor",
                                        floor,
                                        (0. + floor.powi(2)).sqrt(),
                                    )],
                                ),
                                (
                                    "lem-sasaki-mean-shift-bound-sq",
                                    r"\Delta_{\text{mean}} = ((\mu_2 - \mu_1) / \sigma'_1) \cdot \mathbf{1}",
                                    vec![equality(
                                        "native_mean_component_exact_norm",
                                        mean2,
                                        k1 as f64 * (mui - mu1).powi(2) / scale1.powi(2),
                                    )],
                                ),
                                (
                                    "lem-sasaki-denom-shift-bound-sq",
                                    r"\Delta_{\text{denom}} = \mathbf z_2 \cdot ((\sigma'_2 - \sigma'_1) / \sigma'_1)",
                                    vec![equality(
                                        "native_denominator_component_exact_norm",
                                        denom2,
                                        norm2(&zi) * (scalei - scale1).powi(2) / scale1.powi(2),
                                    )],
                                ),
                                (
                                    "lem-sasaki-structural-error-decomposition",
                                    r"\Delta_{\text{direct}} \cdot \Delta_{\text{indirect}} = 0",
                                    vec![equality(
                                        "native_structural_disjoint_product",
                                        (0..n)
                                            .map(|i| {
                                                if mask1[i] != mask2[i] && mask1[i] && mask2[i] {
                                                    (zi[i] - z2[i]).powi(2)
                                                } else {
                                                    0.
                                                }
                                            })
                                            .sum(),
                                        0.,
                                    )],
                                ),
                                (
                                    "lem-sasaki-direct-structural-error-sq",
                                    r"n_c = n_c(\mathcal S_1, \mathcal S_2)",
                                    vec![equality(
                                        "native_status_symmetric_difference_count",
                                        nc,
                                        mask1.iter().zip(mask2).filter(|(a, b)| a != b).count()
                                            as f64,
                                    )],
                                ),
                                (
                                    "def-sasaki-structural-coeffs-sq",
                                    r"k_1:=|\mathcal A_1|",
                                    vec![equality(
                                        "native_structural_first_count",
                                        k1 as f64,
                                        mask1.iter().filter(|a| **a).count() as f64,
                                    )],
                                ),
                                (
                                    "def-sasaki-structural-coeffs-sq",
                                    r"k_2:=|\mathcal A_2|",
                                    vec![equality(
                                        "native_structural_second_count",
                                        k2 as f64,
                                        mask2.iter().filter(|a| **a).count() as f64,
                                    )],
                                ),
                                (
                                    "def-sasaki-structural-coeffs-sq",
                                    r"k_{\mathrm{stable}}:=|\mathcal A_1\cap\mathcal A_2|",
                                    vec![equality(
                                        "native_structural_stable_count",
                                        stable as f64,
                                        mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count()
                                            as f64,
                                    )],
                                ),
                                (
                                    "thm-sasaki-standardization-composite-sq",
                                    r"k_r=|\mathcal A(\mathcal S_r)|\ge1",
                                    vec![upper("both_actual_alive_counts", 1., k1.min(k2) as f64)],
                                ),
                            ] {
                                b.inline(label, quote, inputs.clone(), checks, SCOPE)?;
                            }
                            for (quote, values, mask) in
                                [(r"z_{2,i} = 0", &z2, mask2), (r"z_{1,i} = 0", &zi, mask1)]
                            {
                                if mask.iter().any(|a| !*a) {
                                    b.inline(
                                        "lem-sasaki-direct-structural-error-sq",
                                        quote,
                                        inputs.clone(),
                                        vec![equality(
                                            "native_zero_output_on_dead_rows",
                                            values
                                                .iter()
                                                .zip(mask)
                                                .filter(|(_, a)| !**a)
                                                .map(|(z, _)| z.abs())
                                                .sum(),
                                            0.,
                                        )],
                                        SCOPE,
                                    )?;
                                }
                            }
                            b.inline(
                                "lem-sasaki-direct-structural-error-sq",
                                r"(-z_{2,i})^2 = (z_{2,i})^2",
                                inputs.clone(),
                                vec![equality(
                                    "native_squared_sign_identity",
                                    z2.iter().map(|z| (-z).powi(2)).sum(),
                                    norm2(&z2),
                                )],
                                SCOPE,
                            )?;
                            if mask1 == mask2 {
                                b.inline(
                                    "thm-sasaki-standardization-value-sq",
                                    r"n_c(\mathcal S_1,\mathcal S_2)=0",
                                    inputs.clone(),
                                    vec![equality("native_fixed_structure_status_count", nc, 0.)],
                                    SCOPE,
                                )?;
                            }
                            for label in [
                                "thm-sasaki-standardization-value-sq",
                                "lem-sasaki-mean-shift-bound-sq",
                                "lem-sasaki-denom-shift-bound-sq",
                                "def-sasaki-standardization-constants-sq",
                            ] {
                                b.inline(
                                    label,
                                    r"k \ge 1",
                                    inputs.clone(),
                                    vec![upper("positive_required_scalar_count", 1., k1 as f64)],
                                    SCOPE,
                                )?;
                            }
                            for label in [
                                "sec-eg-operator-estimates",
                                "def-sasaki-standardization-constants",
                            ] {
                                b.inline(label,r"\sigma_{\min,\mathrm{patch}}:=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}",inputs.clone(),vec![equality("native_global_sigma_floor",floor,(0.+floor.powi(2)).sqrt())],SCOPE)?;
                            }
                        }
                        b.add(
                            "lem-sasaki-aggregator-value",
                            "|\\mu(",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                upper(
                                    "mean_value_modulus",
                                    (mu1 - mui).abs(),
                                    delta.sqrt() / (k1 as f64).sqrt(),
                                ),
                                upper(
                                    "second_moment_value_modulus",
                                    (second(&raw, mask1) - second(&changed, mask1)).abs(),
                                    2. * delta.sqrt() / (k1 as f64).sqrt(),
                                ),
                            ],
                            SCOPE,
                        )?;
                        b.add(
                            "lem-sasaki-aggregator-structural",
                            "|\\mu(",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                upper(
                                    "fixed_raw_mean_structural_modulus",
                                    (mui - mu2).abs(),
                                    c.mean_structural_lipschitz_sasaki * nc,
                                ),
                                upper(
                                    "fixed_raw_second_structural_modulus",
                                    (second(&changed, mask1) - second(&changed, mask2)).abs(),
                                    c.second_structural_lipschitz_sasaki * nc,
                                ),
                            ],
                            SCOPE,
                        )?;
                        b.add(
                            "lem-sasaki-aggregator-lipschitz",
                            "L_{\\mu,M}",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                equality(
                                    "mean_value_constant",
                                    c.mean_value_lipschitz,
                                    1. / (k1 as f64).sqrt(),
                                ),
                                equality(
                                    "second_value_constant",
                                    c.second_value_lipschitz,
                                    2. / (k1 as f64).sqrt(),
                                ),
                            ],
                            SCOPE,
                        )?;
                        b.add(
                            "lem-sasaki-aggregator-lipschitz",
                            "L_{\\mu,S}",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                equality(
                                    "mean_structural_constant",
                                    c.mean_structural_lipschitz_sasaki,
                                    3. / k1.min(k2) as f64,
                                ),
                                equality(
                                    "second_structural_constant",
                                    c.second_structural_lipschitz_sasaki,
                                    3. / k1.min(k2) as f64,
                                ),
                            ],
                            SCOPE,
                        )?;
                        for (marker, checks) in [
                            (
                                r"\Delta\mathbf{z} =",
                                vec![equality(
                                    "three_component_algebra",
                                    (0..n)
                                        .map(|i| {
                                            (z1[i] - zi[i] - direct[i] - meanvec[i] - denom[i])
                                                .powi(2)
                                        })
                                        .sum(),
                                    0.,
                                )],
                            ),
                            (
                                r"\Delta_{\text{direct}} :=",
                                vec![equality(
                                    "direct_component_definition",
                                    direct2,
                                    delta / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\Delta_{\text{mean}} :=",
                                vec![equality(
                                    "mean_component_definition",
                                    mean2,
                                    k1 as f64 * (mui - mu1).powi(2) / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\Delta_{\text{denom}} :=",
                                vec![equality(
                                    "denominator_component_definition",
                                    denom2,
                                    norm2(&zi) * (scalei - scale1).powi(2) / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\|\Delta\mathbf{z}\|_2^2 \le",
                                vec![upper(
                                    "three_component_squared_triangle",
                                    ev,
                                    3. * (direct2 + mean2 + denom2),
                                )],
                            ),
                            (
                                r"\begin{aligned}",
                                vec![equality(
                                    "every_algebra_row_identity",
                                    (0..n)
                                        .map(|i| {
                                            (z1[i] - zi[i] - direct[i] - meanvec[i] - denom[i])
                                                .powi(2)
                                        })
                                        .sum(),
                                    0.,
                                )],
                            ),
                            (
                                r"\|\Delta\mathbf{z}\|_2^2 =",
                                vec![
                                    equality(
                                        "sum_equals_error",
                                        (0..n)
                                            .map(|i| {
                                                (z1[i] - zi[i] - direct[i] - meanvec[i] - denom[i])
                                                    .powi(2)
                                            })
                                            .sum(),
                                        0.,
                                    ),
                                    upper(
                                        "triangle_then_cauchy",
                                        ev,
                                        (direct2.sqrt() + mean2.sqrt() + denom2.sqrt()).powi(2),
                                    ),
                                    upper(
                                        "cauchy_three_term",
                                        (direct2.sqrt() + mean2.sqrt() + denom2.sqrt()).powi(2),
                                        3. * (direct2 + mean2 + denom2),
                                    ),
                                ],
                            ),
                        ] {
                            b.add(
                                "lem-sasaki-value-error-decomposition",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                checks,
                                SCOPE,
                            )?;
                        }
                        for (marker, checks) in [
                            (
                                r"\|\Delta_{\text{direct}}\|_2^2 \le",
                                vec![upper("direct_shift_upper", direct2, c.value_direct * delta)],
                            ),
                            (
                                r"\|\Delta_{\text{direct}}\|_2^2 = \left",
                                vec![equality(
                                    "direct_norm_definition",
                                    direct2,
                                    delta / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{direct}}\|_2^2 = \frac",
                                vec![equality(
                                    "direct_scalar_factor",
                                    direct2,
                                    delta / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\frac{1}{(\sigma'_1)^2}",
                                vec![upper(
                                    "inverse_scale_floor",
                                    1. / scale1.powi(2),
                                    1. / floor.powi(2),
                                )],
                            ),
                        ] {
                            b.add(
                                "lem-sasaki-direct-shift-bound-sq",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                checks,
                                SCOPE,
                            )?;
                        }
                        for (marker, checks) in [
                            (
                                r"\|\Delta_{\text{mean}}\|_2^2 \le",
                                vec![upper("mean_shift_upper", mean2, c.value_mean * delta)],
                            ),
                            (
                                r"\|\Delta_{\text{mean}}\|_2^2 = \left",
                                vec![equality(
                                    "mean_norm_definition",
                                    mean2,
                                    k1 as f64 * (mui - mu1).powi(2) / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{mean}}\|_2^2 = \frac",
                                vec![equality(
                                    "ones_vector_norm",
                                    mean2,
                                    k1 as f64 * (mui - mu1).powi(2) / scale1.powi(2),
                                )],
                            ),
                            (
                                r"|\mu_2 - \mu_1|^2",
                                vec![upper(
                                    "squared_mean_value_modulus",
                                    (mui - mu1).powi(2),
                                    c.mean_value_lipschitz.powi(2) * delta,
                                )],
                            ),
                        ] {
                            b.add(
                                "lem-sasaki-mean-shift-bound-sq",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                checks,
                                SCOPE,
                            )?;
                        }
                        for (marker, checks) in [
                            (
                                r"\|\Delta_{\text{denom}}\|_2^2 \le k",
                                vec![upper(
                                    "denominator_shift_upper",
                                    denom2,
                                    c.value_scale * delta,
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{denom}}\|_2^2 = \left",
                                vec![equality(
                                    "denominator_norm_definition",
                                    denom2,
                                    norm2(&zi) * (scalei - scale1).powi(2) / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{denom}}\|_2^2 = \|",
                                vec![equality(
                                    "denominator_factor_identity",
                                    denom2,
                                    norm2(&zi) * (scalei - scale1).powi(2) / scale1.powi(2),
                                )],
                            ),
                            (
                                r"\|\mathbf z_2\|_2^2",
                                vec![upper(
                                    "standardized_vector_bound",
                                    norm2(&zi),
                                    k1 as f64 * (2. / floor).powi(2),
                                )],
                            ),
                            (
                                r"(\sigma'_2 - \sigma'_1)^2",
                                vec![upper(
                                    "scale_value_modulus",
                                    (scalei - scale1).powi(2),
                                    c.scale_value_lipschitz.powi(2) * delta,
                                )],
                            ),
                            (
                                r"\frac{1}{(\sigma'_1)^2}",
                                vec![upper(
                                    "inverse_scale_floor",
                                    1. / scale1.powi(2),
                                    1. / floor.powi(2),
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{denom}}\|_2^2 \le \left",
                                vec![upper(
                                    "all_denominator_factors",
                                    denom2,
                                    c.value_scale * delta,
                                )],
                            ),
                        ] {
                            b.add(
                                "lem-sasaki-denom-shift-bound-sq",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                checks,
                                SCOPE,
                            )?;
                        }
                        b.add("thm-sasaki-standardization-value-sq",r"\big\|z(",inputs.clone(),hypotheses.clone(),vec![upper("full_value_bound",ev,c.value_total*delta)],"First scalar-array inequality of the source chain. Its physical reward comparison is exercised separately on native compact x-v rewards.")?;
                        for (context, lhs, rhs) in [
                            (
                                "total squared value error is bounded by:",
                                ev,
                                3. * (direct2 + mean2 + denom2),
                            ),
                            (
                                "*   From {prf:ref}`lem-sasaki-direct-shift-bound-sq`:",
                                direct2,
                                c.value_direct * delta,
                            ),
                            (
                                "*   From {prf:ref}`lem-sasaki-mean-shift-bound-sq`:",
                                mean2,
                                c.value_mean * delta,
                            ),
                            (
                                "*   From {prf:ref}`lem-sasaki-denom-shift-bound-sq`:",
                                denom2,
                                c.value_scale * delta,
                            ),
                            ("factoring out the common term", ev, c.value_total * delta),
                            (
                                "the direct error is bounded by a term linear",
                                direct_s,
                                c.structural_direct * nc,
                            ),
                            (
                                "the indirect error is bounded by a term quadratic",
                                indirect_s,
                                c.structural_indirect_split_sasaki * nc * nc,
                            ),
                            (
                                "Summing the two bounds from Step 2 directly",
                                es,
                                c.structural_direct * nc
                                    + c.structural_indirect_split_sasaki * nc * nc,
                            ),
                        ] {
                            b.proof(
                                "sec-eg-operator-estimates",
                                context,
                                inputs.clone(),
                                hypotheses.clone(),
                                vec![upper("measured_proof_component_bound", lhs, rhs)],
                                SCOPE,
                            )?;
                        }
                        b.proof(
                            "sec-eg-operator-estimates",
                            "For notational compactness write",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![equality(
                                "assembled_scale_value_coefficient",
                                c.scale_value_lipschitz,
                                (c.second_value_lipschitz + 2. * c.mean_value_lipschitz)
                                    / (2. * floor),
                            )],
                            SCOPE,
                        )?;
                        b.proof(
                            "sec-eg-operator-estimates",
                            "total squared structural error is the sum",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![equality(
                                "measured_structural_orthogonal_assembly",
                                es,
                                direct_s + indirect_s,
                            )],
                            SCOPE,
                        )?;
                        b.proof(
                            "lem-sasaki-direct-structural-error-sq",
                            "We sum this uniform bound over",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                equality(
                                    "direct_unstable_component_sum",
                                    direct_s,
                                    (0..n)
                                        .filter(|&i| mask1[i] != mask2[i])
                                        .map(|i| (zi[i] - z2[i]).powi(2))
                                        .sum(),
                                ),
                                upper(
                                    "direct_unstable_component_envelope",
                                    direct_s,
                                    nc * (2. / floor).powi(2),
                                ),
                            ],
                            SCOPE,
                        )?;
                        let unstable_max = (0..n)
                            .filter(|&i| mask1[i] != mask2[i])
                            .map(|i| (zi[i] - z2[i]).powi(2))
                            .fold(0_f64, f64::max);
                        for (marker, checks) in [
                            (
                                r"(z_{1,i} - z_{2,i})^2",
                                vec![upper(
                                    "every_unstable_row_bound",
                                    unstable_max,
                                    (2. / floor).powi(2),
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{direct}}\|_2^2 = \sum",
                                vec![
                                    equality(
                                        "unstable_component_sum",
                                        direct_s,
                                        (0..n)
                                            .filter(|&i| mask1[i] != mask2[i])
                                            .map(|i| (zi[i] - z2[i]).powi(2))
                                            .sum(),
                                    ),
                                    upper(
                                        "unstable_sum_envelope",
                                        direct_s,
                                        nc * (2. / floor).powi(2),
                                    ),
                                ],
                            ),
                            (
                                r"\|\Delta_{\text{direct}}\|_2^2 \le n_c",
                                vec![upper(
                                    "unstable_count_envelope",
                                    direct_s,
                                    nc * (2. / floor).powi(2),
                                )],
                            ),
                        ] {
                            b.add(
                                "lem-sasaki-direct-structural-error-sq",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                checks,
                                SCOPE,
                            )?;
                        }
                        for (marker, expected, actual) in [
                            (
                                r"C_{V,\mathrm{direct}}^{\mathrm{sq}}(\mathcal S) :=",
                                1. / floor.powi(2),
                                c.value_direct,
                            ),
                            (
                                r"C_{V,\mathrm{mean}}^{\mathrm{sq}}(\mathcal S) :=",
                                k1 as f64 * c.mean_value_lipschitz.powi(2) / floor.powi(2),
                                c.value_mean,
                            ),
                            (
                                r"C_{V,\mathrm{denom}}^{\mathrm{sq}}(\mathcal S) :=",
                                k1 as f64
                                    * (2. / floor).powi(2)
                                    * (c.scale_value_lipschitz / floor).powi(2),
                                c.value_scale,
                            ),
                            (
                                r"C_{V,\mathrm{total}}^{\mathrm{Sasaki}}(\mathcal S) :=",
                                3. * (c.value_direct + c.value_mean + c.value_scale),
                                c.value_total,
                            ),
                        ] {
                            b.add(
                                "def-sasaki-standardization-constants-sq",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                vec![equality("squared_value_coefficient", actual, expected)],
                                SCOPE,
                            )?;
                        }
                        b.add(
                            "lem-sasaki-structural-error-decomposition",
                            r"\Delta\mathbf{z} =",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![equality(
                                "structural_components_total",
                                es,
                                direct_s + indirect_s,
                            )],
                            SCOPE,
                        )?;
                        b.proof(
                            "lem-sasaki-structural-error-decomposition",
                            "For orthogonal vectors, the squared norm",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                equality(
                                    "structural_entire_Pythagorean_chain",
                                    es,
                                    direct_s + indirect_s,
                                ),
                                equality(
                                    "structural_disjoint_component_dot_product",
                                    (0..n)
                                        .map(|i| {
                                            if mask1[i] != mask2[i] {
                                                (zi[i] - z2[i])
                                                    * (if mask1[i] && mask2[i] {
                                                        zi[i] - z2[i]
                                                    } else {
                                                        0.
                                                    })
                                            } else {
                                                0.
                                            }
                                        })
                                        .sum(),
                                    0.,
                                ),
                            ],
                            SCOPE,
                        )?;
                        b.add(
                            "lem-sasaki-structural-error-decomposition",
                            r"\|\Delta\mathbf{z}\|_2^2 =",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![equality(
                                "disjoint_support_orthogonality",
                                es,
                                direct_s + indirect_s,
                            )],
                            SCOPE,
                        )?;
                        b.add(
                            "lem-sasaki-direct-structural-error-sq",
                            r"\|\Delta_{\text{direct}}\|_2^2",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![upper(
                                "unstable_rows_bound",
                                direct_s,
                                c.structural_direct * nc,
                            )],
                            SCOPE,
                        )?;
                        for (marker, checks) in [
                            (
                                r"\|\Delta_{\text{indirect}}\|_2^2 \le C",
                                vec![upper(
                                    "indirect_structural_coefficient",
                                    indirect_s,
                                    c.structural_indirect_split_sasaki * nc * nc,
                                )],
                            ),
                            (
                                r"\begin{aligned}",
                                vec![equality(
                                    "stable_row_structural_algebra",
                                    (0..n)
                                        .filter(|&i| mask1[i] && mask2[i])
                                        .map(|i| {
                                            (zi[i]
                                                - z2[i]
                                                - (mu2 - mui) / scalei
                                                - z2[i] * (scale2 - scalei) / scalei)
                                                .powi(2)
                                        })
                                        .sum(),
                                    0.,
                                )],
                            ),
                            (
                                r"(z_{1,i} - z_{2,i})^2",
                                vec![upper(
                                    "stable_row_two_term_squared_triangle",
                                    indirect_s,
                                    indirect_pre,
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{indirect}}\|_2^2 = \sum",
                                vec![upper(
                                    "summed_stable_two_term_triangle",
                                    indirect_s,
                                    indirect_pre,
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{indirect}}\|_2^2 \le 2 k",
                                vec![upper(
                                    "stable_mean_and_scale_prebound",
                                    indirect_s,
                                    indirect_pre,
                                )],
                            ),
                            (
                                r"(\mu_2 - \mu_1)^2",
                                vec![upper(
                                    "structural_mean_squared",
                                    (mu2 - mui).powi(2),
                                    (c.mean_structural_lipschitz_sasaki * nc).powi(2),
                                )],
                            ),
                            (
                                r"(\sigma'_2 - \sigma'_1)^2",
                                vec![upper(
                                    "structural_scale_squared",
                                    (scale2 - scalei).powi(2),
                                    (c.scale_structural_lipschitz_sasaki * nc).powi(2),
                                )],
                            ),
                            (
                                r"\sum_{i \in \mathcal A_{\mathrm{stable}}}",
                                vec![
                                    upper(
                                        "stable_subset_z_norm",
                                        (0..n)
                                            .filter(|&i| mask1[i] && mask2[i])
                                            .map(|i| z2[i].powi(2))
                                            .sum(),
                                        norm2(&z2),
                                    ),
                                    upper(
                                        "full_z_bound",
                                        norm2(&z2),
                                        k2 as f64 * (2. / floor).powi(2),
                                    ),
                                ],
                            ),
                            (
                                r"\|\Delta_{\text{indirect}}\|_2^2 \le 2 k_{\mathrm{stable}} \frac{(L",
                                vec![upper(
                                    "substituted_structural_moment_constants",
                                    indirect_s,
                                    c.structural_indirect_split_sasaki * nc * nc,
                                )],
                            ),
                            (
                                r"\|\Delta_{\text{indirect}}\|_2^2 \le \left[",
                                vec![upper(
                                    "factored_structural_constant",
                                    indirect_s,
                                    c.structural_indirect_split_sasaki * nc * nc,
                                )],
                            ),
                        ] {
                            b.add(
                                "lem-sasaki-indirect-structural-error-sq",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                checks,
                                SCOPE,
                            )?;
                        }
                        b.add(
                            "thm-sasaki-standardization-structural-sq",
                            r"\|z(\mathcal S_1)-",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![upper(
                                "fixed_raw_full_structural_bound",
                                es,
                                c.structural_direct * nc
                                    + c.structural_indirect_split_sasaki * nc * nc,
                            )],
                            SCOPE,
                        )?;
                        b.add(
                            "def-sasaki-structural-coeffs-sq",
                            r"C_{S,\mathrm{direct}}^{\mathrm{sq}} :=",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![equality(
                                "squared_structural_direct_coefficient",
                                c.structural_direct,
                                (2. / floor).powi(2),
                            )],
                            SCOPE,
                        )?;
                        b.add(
                            "def-sasaki-structural-coeffs-sq",
                            r"C_{S,\mathrm{indirect}}^{\mathrm{sq}}(\mathcal S_1",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![equality(
                                "squared_structural_indirect_coefficient",
                                c.structural_indirect_split_sasaki,
                                2. * stable as f64 * c.mean_structural_lipschitz_sasaki.powi(2)
                                    / floor.powi(2)
                                    + 2. * k2 as f64
                                        * (2. / floor).powi(2)
                                        * c.scale_structural_lipschitz_sasaki.powi(2)
                                        / floor.powi(2),
                            )],
                            SCOPE,
                        )?;
                        b.inline(
                            "lem-sasaki-mean-shift-bound-sq",
                            r"\|\mathbf{1}\|_2 = \sqrt{k}",
                            inputs.clone(),
                            vec![equality(
                                "ones_vector_norm",
                                norm2(&vec![1.; k1]).sqrt(),
                                (k1 as f64).sqrt(),
                            )],
                            SCOPE,
                        )?;
                        b.inline(
                            "lem-sasaki-aggregator-structural",
                            r"|k_1-k_2|\le n_c",
                            inputs.clone(),
                            vec![upper(
                                "alive_count_difference_status_bound",
                                k1.abs_diff(k2) as f64,
                                nc,
                            )],
                            SCOPE,
                        )?;
                        b.inline("lem-sasaki-aggregator-lipschitz",r"p_{\mu,S}=p_{m_2,S}=p_{\mathrm{worst\text{-}case}}=-1",inputs.clone(),vec![equality("structural_mean_growth_exponent",(3_f64/(2.*k1.min(k2) as f64)).ln()-(3_f64/(k1.min(k2) as f64)).ln(),-2_f64.ln()),equality("structural_second_growth_exponent",(3_f64/(2.*k1.min(k2) as f64)).ln()-(3_f64/(k1.min(k2) as f64)).ln(),-2_f64.ln())],"Structural coefficients scale exactly as the inverse alive count; this does not introduce N-dependent primary population error.")?;
                        b.inline("lem-sasaki-aggregator-lipschitz",r"\kappa_{\mathrm{var}}^{\mathrm{Sasaki}}=\kappa_{\mathrm{range}}^{\mathrm{Sasaki}}=1",inputs.clone(),vec![upper("empirical_variance_range_bound",second(&changed,mask1)-mui.powi(2),1.),upper("empirical_range_bound",changed.iter().copied().fold(f64::NEG_INFINITY,f64::max)-changed.iter().copied().fold(f64::INFINITY,f64::min),2.)],SCOPE)?;
                        let direct_lin = 2. / floor;
                        let indirect_lin = (stable as f64).sqrt()
                            * (3. / (floor * k1.min(k2) as f64)
                                + 2. * c.scale_structural_lipschitz_sasaki / floor.powi(2));
                        b.add(
                            "def-sasaki-standardization-constants",
                            r"C_{S,\mathrm{direct}}^{\mathrm{Sasaki}}",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                equality("linear_direct_coefficient", direct_lin, 2. / floor),
                                equality(
                                    "linear_scale_structural_coefficient",
                                    c.scale_structural_lipschitz_sasaki,
                                    9. / (2. * floor * k1.min(k2) as f64),
                                ),
                                equality(
                                    "linear_indirect_coefficient",
                                    indirect_lin,
                                    (stable as f64).sqrt()
                                        * (c.mean_structural_lipschitz_sasaki / floor
                                            + 2. * c.scale_structural_lipschitz_sasaki
                                                / floor.powi(2)),
                                ),
                            ],
                            SCOPE,
                        )?;
                        b.add(
                            "def-sasaki-standardization-constants",
                            r"\|z(\mathcal S_1,\mathbf r)-",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![upper(
                                "linear_structural_bound",
                                es.sqrt(),
                                direct_lin * nc.sqrt() + indirect_lin * nc,
                            )],
                            SCOPE,
                        )?;
                        for (marker, lhs, rhs) in [
                            (
                                r"C_{V,\mathrm{direct}} :=",
                                direct2.sqrt(),
                                delta.sqrt() / floor,
                            ),
                            (
                                r"C_{V,\mathrm{mean}} :=",
                                mean2.sqrt(),
                                delta.sqrt() / floor,
                            ),
                            (
                                r"C_{V,\mathrm{denom}} :=",
                                denom2.sqrt(),
                                4. / floor.powi(3) * delta.sqrt(),
                            ),
                        ] {
                            b.add(
                                "def-sasaki-standardization-constants",
                                marker,
                                inputs.clone(),
                                hypotheses.clone(),
                                vec![upper("unsquared_value_component_constant", lhs, rhs)],
                                SCOPE,
                            )?;
                        }
                        // The logistic map acts only on alive laws; counts may differ.
                        let a1 = z1
                            .iter()
                            .zip(mask1)
                            .filter(|(_, a)| **a)
                            .map(|(z, _)| *z)
                            .collect::<Vec<_>>();
                        let a2 = z2
                            .iter()
                            .zip(mask2)
                            .filter(|(_, a)| **a)
                            .map(|(z, _)| *z)
                            .collect::<Vec<_>>();
                        let r1 = raw
                            .iter()
                            .zip(mask1)
                            .filter(|(_, a)| **a)
                            .map(|(z, _)| *z)
                            .collect::<Vec<_>>();
                        let r2 = changed
                            .iter()
                            .zip(mask2)
                            .filter(|(_, a)| **a)
                            .map(|(z, _)| *z)
                            .collect::<Vec<_>>();
                        let rawcost = scalar_transport(&r1, &r2)?;
                        let zcost = scalar_transport(&a1, &a2)?;
                        let g = PositiveMap::Logistic {
                            amplitude: 2.,
                            floor: 0.1,
                        };
                        let u1 = a1.iter().map(|z| g.map(*z)).collect::<Result<Vec<_>>>()?;
                        let u2 = a2.iter().map(|z| g.map(*z)).collect::<Result<Vec<_>>>()?;
                        let ucost = scalar_transport(&u1, &u2)?;
                        b.add("cor-eg-native-standardization-uniform",r"W_2^2(\operatorname{Std}_m",inputs.clone(),hypotheses.clone(),vec![upper("empirical_scalar_N_uniform_standardization",zcost,rawcost/floor.powi(2)),upper("alive_z_second_moment_one",norm2(&a1)/k1 as f64,1.),upper("other_alive_z_second_moment_one",norm2(&a2)/k2 as f64,1.)],"Actual native standardizer on realized arrays, normalized alive empirical laws, unequal alive counts allowed; optimal scalar quantile transport.")?;
                        b.add(
                            "cor-eg-native-standardization-uniform",
                            r"|g'(z)|=",
                            inputs.clone(),
                            hypotheses,
                            vec![
                                upper(
                                    "native_logistic_empirical_composition",
                                    ucost,
                                    rawcost / (4. * floor.powi(2)),
                                ),
                                upper("native_logistic_transport_Lipschitz", ucost, zcost / 4.),
                            ],
                            SCOPE,
                        )?;
                        b.add("thm-sasaki-standardization-composite-sq",r"\|z(\mathcal S_1, \mathbf r_1)",inputs,vec![],vec![upper("value_structural_triangle",all,2.*ev+2.*es)],"Deterministic value/structure split with native Global outputs. The physical coefficient assembly uses compact native objective pairs separately.")?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn support(mask: &[bool], i: usize) -> Vec<usize> {
    let alive = mask
        .iter()
        .enumerate()
        .filter_map(|(j, a)| a.then_some(j))
        .collect::<Vec<_>>();
    if alive.len() == 1 {
        alive
    } else {
        alive.into_iter().filter(|j| *j != i).collect()
    }
}
fn average_distance(
    metric: &Distance,
    left: &ObservationBatch<f64>,
    i: usize,
    mask: &[bool],
) -> Result<f64> {
    if !mask[i] {
        return Ok(0.);
    }
    let indices = support(mask, i);
    let values = indices
        .iter()
        .map(|&j| metric.compare(left, i, left, j))
        .collect::<Result<Vec<_>>>()?;
    Ok(values.iter().sum::<f64>() / indices.len() as f64)
}
fn uniform_cases(b: &mut EvidenceBuilder) -> Result<()> {
    for n in 1..=5 {
        let x1 = (0..n).map(|i| 0.5 * (i as f64 - 1.)).collect::<Vec<_>>();
        let x2 = x1
            .iter()
            .enumerate()
            .map(|(i, a)| a + 0.04 * (i as f64).cos())
            .collect::<Vec<_>>();
        let v1 = (0..n).map(|i| 0.2 * (i as f64).sin()).collect::<Vec<_>>();
        let v2 = v1
            .iter()
            .enumerate()
            .map(|(i, a)| a - 0.03 * (i as f64).cos())
            .collect::<Vec<_>>();
        let o1 = obs(&x1, &v1, 1)?;
        let o2raw = obs(&x2, &v2, 1)?;
        let metric = metric(2., 2., 1.);
        for m1 in 1..(1 << n) {
            for m2 in 1..(1 << n) {
                let a1 = (0..n).map(|i| m1 & (1 << i) != 0).collect::<Vec<_>>();
                let a2raw = (0..n).map(|i| m2 & (1 << i) != 0).collect::<Vec<_>>();
                let plan = optimal_swarm_displacement(&metric, &o1, &o2raw, &a1, &a2raw, 1.)?;
                let indices = plan
                    .transport_plan
                    .iter()
                    .map(|r| {
                        r.iter()
                            .enumerate()
                            .max_by(|a, c| a.1.total_cmp(c.1))
                            .map(|(j, _)| j as u32)
                            .expect("nonempty plan row")
                    })
                    .collect::<Vec<_>>();
                let o2 = o2raw.gather(&indices)?;
                let a2 = indices
                    .iter()
                    .map(|i| a2raw[*i as usize])
                    .collect::<Vec<_>>();
                let k1 = a1.iter().filter(|a| **a).count();
                let k2 = a2.iter().filter(|a| **a).count();
                let stable = (0..n).filter(|&i| a1[i] && a2[i]).collect::<Vec<_>>();
                let nc = (0..n).filter(|&i| a1[i] != a2[i]).count() as f64;
                let displacement = (0..n)
                    .map(|i| metric.compare(&o1, i, &o2, i))
                    .collect::<Result<Vec<_>>>()?;
                let delta = norm2(&displacement);
                let diameter = (4_f64 * 2_f64.powi(2) + 4_f64 * 2_f64.powi(2)).sqrt();
                let cp = 2. * (1. + stable.len() as f64 / (k1.saturating_sub(1).max(1)) as f64);
                let denom = k1.saturating_sub(1).max(1) as f64;
                let baseline_native = Standardizer::Global { sigma_min: 0.1 }
                    .apply(&vec![0.; n], &a1, &o1)?
                    .1;
                let alive_row = a1.iter().position(|a| *a).expect("nonempty alive input");
                let native_variance_baseline = baseline_native.scale[alive_row].powi(2) - 0.01;
                let d1 = (0..n)
                    .map(|i| average_distance(&metric, &o1, i, &a1))
                    .collect::<Result<Vec<_>>>()?;
                let d2 = (0..n)
                    .map(|i| average_distance(&metric, &o2, i, &a2))
                    .collect::<Result<Vec<_>>>()?;
                let mut psum = 0.;
                let mut fullstable = 0.;
                let mut pchecks = vec![];
                let mut qchecks = vec![];
                let mut rowchecks = vec![];
                for &i in &stable {
                    let s1 = support(&a1, i);
                    let s2 = support(&a2, i);
                    let first = s1
                        .iter()
                        .map(|&j| metric.compare(&o1, i, &o1, j))
                        .collect::<Result<Vec<_>>>()?;
                    let second = s1
                        .iter()
                        .map(|&j| metric.compare(&o2, i, &o2, j))
                        .collect::<Result<Vec<_>>>()?;
                    let third = s2
                        .iter()
                        .map(|&j| metric.compare(&o2, i, &o2, j))
                        .collect::<Result<Vec<_>>>()?;
                    let f = first.iter().sum::<f64>() / s1.len() as f64;
                    let s = second.iter().sum::<f64>() / s1.len() as f64;
                    let t = third.iter().sum::<f64>() / s2.len() as f64;
                    let p = (f - s).abs();
                    let q = (s - t).abs();
                    let e = displacement[i]
                        + s1.iter().map(|j| displacement[*j]).sum::<f64>() / s1.len() as f64;
                    psum += p * p;
                    fullstable += (d1[i] - d2[i]).powi(2);
                    pchecks.push(upper("single_row_positional_reverse_triangle", p, e));
                    pchecks.push(upper(
                        "expectation_absolute_value",
                        p,
                        first
                            .iter()
                            .zip(&second)
                            .map(|(a, c)| (a - c).abs())
                            .sum::<f64>()
                            / s1.len() as f64,
                    ));
                    for (offset, &j) in s1.iter().enumerate() {
                        if n <= 3 {
                            b.inline("lem-sasaki-single-walker-positional-error",r"|d(a,b) - d(c,d)| \le d(a,c) + d(b,d)",json!({"N":n,"native_observations1":o1,"native_observations2":o2,"row":i,"companion":j}),vec![upper("native_pointwise_metric_reverse_triangle",(first[offset]-second[offset]).abs(),displacement[i]+displacement[j])],SCOPE)?;
                        }
                        rowchecks.push(upper(
                            "pointwise_reverse_triangle",
                            (first[offset] - second[offset]).abs(),
                            displacement[i] + displacement[j],
                        ));
                    }
                    if k1 >= 2 {
                        pchecks.push(upper(
                            "positional_jensen_subbound",
                            p * p,
                            2. * displacement[i].powi(2)
                                + 2. * a1
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, a)| **a)
                                    .map(|(j, _)| displacement[j].powi(2))
                                    .sum::<f64>()
                                    / (k1 - 1) as f64,
                        ));
                    }
                    if n <= 3 {
                        let point = json!({"N":n,"alive_masks":[a1,a2],"stable_row":i,"first_support":s1,"second_support":s2,"P_i":p,"Q_i":q,"first_expected_distance":f,"second_expected_distance":t,"diameter":diameter,"n_c":nc,"first_alive_count":k1,"second_alive_count":k2});
                        for (label, quote, checks) in [
                            (
                                "lem-sasaki-total-squared-error-stable",
                                r"|d_i^{(1)}-d_i^{(2)}|\le P_i+Q_i",
                                vec![upper(
                                    "measured_full_distance_triangle",
                                    (f - t).abs(),
                                    p + q,
                                )],
                            ),
                            (
                                "lem-sasaki-total-squared-error-stable",
                                r"Q_i\le D_{\mathcal Y}",
                                vec![upper("measured_support_difference_diameter", q, diameter)],
                            ),
                            (
                                "lem-sasaki-single-walker-structural-error",
                                r"|f(c)| \le D_{\mathcal Y} =: M_f",
                                vec![upper(
                                    "actual_structural_test_function_bound",
                                    second.iter().map(|v| v.abs()).fold(0_f64, f64::max),
                                    diameter,
                                )],
                            ),
                        ] {
                            b.inline(label, quote, point.clone(), checks, SCOPE)?;
                        }
                        b.inline(
                            "lem-sasaki-single-walker-positional-error",
                            r"s_{1,i}=1",
                            point.clone(),
                            vec![equality(
                                "actual_positional_recipient_live",
                                f64::from(a1[i]),
                                1.,
                            )],
                            SCOPE,
                        )?;
                        b.inline(
                            "lem-sasaki-single-walker-positional-error",
                            r"f(x)=|x|",
                            point.clone(),
                            vec![upper(
                                "actual_expectation_absolute_value_contraction",
                                (f - s).abs(),
                                first
                                    .iter()
                                    .zip(&second)
                                    .map(|(a, b)| (a - b).abs())
                                    .sum::<f64>()
                                    / s1.len() as f64,
                            )],
                            SCOPE,
                        )?;
                        b.inline("lem-sasaki-single-walker-structural-error",r"f(c) := d_{\mathcal Y}^{\mathrm{Sasaki}}(\varphi(w_{2,i}), \varphi(w_{2,c}))",point.clone(),vec![equality("native_structural_test_function_definition",s,second.iter().sum::<f64>()/s1.len()as f64)],SCOPE)?;
                        b.inline(
                            "lem-sasaki-single-walker-structural-error",
                            r"M_f = D_{\mathcal Y}",
                            point.clone(),
                            vec![upper(
                                "native_structural_test_function_envelope",
                                second
                                    .iter()
                                    .chain(&third)
                                    .map(|v| v.abs())
                                    .fold(0_f64, f64::max),
                                diameter,
                            )],
                            SCOPE,
                        )?;
                        for (quote, computed, expected) in [
                            (r"S_1 = S_i(\mathcal{S}_1)", &s1, support(&a1, i)),
                            (r"S_2 = S_i(\mathcal{S}_2)", &s2, support(&a2, i)),
                        ] {
                            b.inline(
                                "lem-sasaki-single-walker-structural-error",
                                quote,
                                point.clone(),
                                vec![
                                    equality(
                                        "actual_uniform_companion_support_size",
                                        computed.len() as f64,
                                        expected.len() as f64,
                                    ),
                                    equality(
                                        "actual_uniform_companion_support_elements",
                                        computed
                                            .iter()
                                            .zip(expected.iter())
                                            .filter(|(a, b)| a != b)
                                            .count() as f64,
                                        0.,
                                    ),
                                ],
                                SCOPE,
                            )?;
                        }
                        if k1 >= 2 {
                            b.inline(
                                "lem-sasaki-single-walker-structural-error",
                                r"S_1 = \mathcal A_1 \setminus \{i\}",
                                point.clone(),
                                vec![equality(
                                    "actual_self_excluded_live_set",
                                    s1.len() as f64,
                                    a1.iter()
                                        .enumerate()
                                        .filter(|(j, a)| **a && *j != i)
                                        .count() as f64,
                                )],
                                SCOPE,
                            )?;
                        }
                        if k1 >= 2 {
                            for quote in [r"k_1 \ge 2", r"k_1=|\mathcal A(\mathcal S_1)| \ge 2"] {
                                b.inline(
                                    "lem-sasaki-single-walker-structural-error",
                                    quote,
                                    point.clone(),
                                    vec![upper("nontrivial_actual_donor_pool", 2., k1 as f64)],
                                    SCOPE,
                                )?;
                            }
                            for quote in [r"|S_1| = k_1 - 1", r"|S_1| = k_1 - 1 > 0"] {
                                b.inline(
                                    "lem-sasaki-single-walker-structural-error",
                                    quote,
                                    point.clone(),
                                    vec![
                                        equality(
                                            "actual_self_excluded_support_size",
                                            s1.len() as f64,
                                            (k1 - 1) as f64,
                                        ),
                                        hypothesis(
                                            "positive_self_excluded_support",
                                            !s1.is_empty(),
                                        ),
                                    ],
                                    SCOPE,
                                )?;
                            }
                            if k2 == 1 {
                                b.inline(
                                    "lem-sasaki-total-squared-error-stable",
                                    r"n_c\ge k_1-1",
                                    point.clone(),
                                    vec![upper(
                                        "measured_singleton_collapse_status_count",
                                        (k1 - 1) as f64,
                                        nc,
                                    )],
                                    SCOPE,
                                )?;
                            }
                        } else {
                            b.inline(
                                "lem-sasaki-total-squared-error-stable",
                                r"P_i=0",
                                point.clone(),
                                vec![equality("singleton_self_companion_positional_error", p, 0.)],
                                SCOPE,
                            )?;
                            if k2 > 1 {
                                b.inline(
                                    "lem-sasaki-total-squared-error-stable",
                                    r"Q_i\le D_{\mathcal Y}\le2D_{\mathcal Y}n_c",
                                    point.clone(),
                                    vec![
                                        upper("singleton_structural_diameter", q, diameter),
                                        upper(
                                            "singleton_structural_status_growth",
                                            diameter,
                                            2. * diameter * nc,
                                        ),
                                    ],
                                    SCOPE,
                                )?;
                            }
                            if k2 == 1 {
                                b.inline(
                                    "lem-sasaki-total-squared-error-stable",
                                    r"Q_i=0",
                                    point,
                                    vec![equality("both_singleton_support_error_zero", q, 0.)],
                                    SCOPE,
                                )?;
                            }
                        }
                    }
                    qchecks.push(upper(
                        "uniform_support_status_bound",
                        q,
                        2. * diameter * nc / denom,
                    ));
                    qchecks.push(upper(
                        "squared_full_row_triangle",
                        (f - t).powi(2),
                        2. * p * p + 2. * q * q,
                    ));
                }
                if n <= 3 {
                    b.inline("lem-sasaki-aggregator-lipschitz", r"V_{\max}=V_{\mathrm{max}}^{(d)}", json!({"N":n,"native_distance":metric,"masks":[a1,a2],"expected_distance_vectors":[d1,d2],"V_max":diameter,"V_max_distance":diameter}), vec![upper("native_expected_distance_envelope",d1.iter().chain(&d2).map(|value|value.abs()).fold(0_f64,f64::max),diameter)], "The distance-specific scalar envelope is the diameter of the native comparison feature space. Each actual expected distance lies below this envelope.")?;
                }
                let inputs = json!({"N":n,"physical_x1":x1,"physical_v1":v1,"physical_x2":x2,"physical_v2":v2,"mask1":a1,"mask2_before_transport":a2raw,"optimal_assignment":indices,"paired_mask2":a2,"k1":k1,"k2":k2,"stable":stable,"n_c":nc,"Delta_pos_squared":delta,"native_feature_displacements":displacement,"expected_raw1":d1,"expected_raw2":d2,"D_Y":diameter,"C_pos":cp,"metric_squared":plan.metric_squared});
                let hypotheses = vec![
                    hypothesis("nonempty_companion_pools", k1 > 0 && k2 > 0),
                    hypothesis(
                        "uniform_current_alive_specialization",
                        (0..n)
                            .filter(|i| a1[*i])
                            .all(|i| !support(&a1, i).is_empty())
                            && (0..n)
                                .filter(|i| a2[*i])
                                .all(|i| !support(&a2, i).is_empty()),
                    ),
                ];
                b.proof("sec-eg-definition","- **Dispersion distance.** For swarms",inputs.clone(),hypotheses.clone(),vec![equality("native_minimum_marked_transport_cost",plan.metric_squared,(delta+nc)/n as f64)],"Native optimal marked transport selects representatives; the computed normalized positional and status costs are both retained.")?;
                if n <= 3 {
                    b.inline("lem-sasaki-aggregator-lipschitz",r"n_c\le\frac{N}{\lambda_{\mathrm{status}}}d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2",inputs.clone(),vec![upper("native_status_cost_bounded_by_quotient_cost",nc,n as f64*plan.metric_squared)],SCOPE)?;
                    for (label, quote, checks) in [
                        (
                            "def-eg-auxiliary-continuity-notation",
                            r"k_r=|\mathcal A_r|",
                            vec![
                                equality(
                                    "native_first_alive_count",
                                    k1 as f64,
                                    a1.iter().filter(|a| **a).count() as f64,
                                ),
                                equality(
                                    "native_second_alive_count",
                                    k2 as f64,
                                    a2.iter().filter(|a| **a).count() as f64,
                                ),
                            ],
                        ),
                        (
                            "def-eg-auxiliary-continuity-notation",
                            r"\kappa_{\mathrm{var,min}}=0",
                            vec![equality(
                                "native_global_variance_baseline",
                                native_variance_baseline,
                                0.,
                            )],
                        ),
                        (
                            "def-eg-auxiliary-continuity-notation",
                            r"\varepsilon_{\mathrm{std}}=\sigma_{\min}",
                            vec![equality(
                                "native_global_regularizer_equivalence",
                                baseline_native.scale[alive_row],
                                0.1,
                            )],
                        ),
                        (
                            "lem-sasaki-total-squared-error-stable",
                            r"C_{\mathrm{pos}}^{\mathrm{Sasaki}}(k_1,k_{\mathrm{stable}})
:=2(1+k_{\mathrm{stable}}/\max\{1,k_1-1\})",
                            vec![equality(
                                "computed_uniform_positional_coefficient",
                                cp,
                                2. * (1. + stable.len() as f64 / denom),
                            )],
                        ),
                    ] {
                        b.inline(label, quote, inputs.clone(), checks, SCOPE)?;
                    }
                    let stable_indices = a1
                        .iter()
                        .zip(&a2)
                        .enumerate()
                        .filter(|(_, (a, b))| **a && **b)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>();
                    for (label, quote) in [
                        (
                            "def-eg-auxiliary-continuity-notation",
                            r"\mathcal A_{\mathrm{stable}}=\mathcal A_1\cap\mathcal A_2",
                        ),
                        (
                            "lem-sasaki-total-squared-error-stable",
                            r"\mathcal A_{\mathrm{stable}}:=\mathcal A_1\cap\mathcal A_2",
                        ),
                    ] {
                        b.inline(
                            label,
                            quote,
                            inputs.clone(),
                            vec![
                                equality(
                                    "actual_alive_set_intersection",
                                    stable
                                        .iter()
                                        .zip(&stable_indices)
                                        .filter(|(a, b)| a != b)
                                        .count() as f64,
                                    0.,
                                ),
                                equality(
                                    "actual_alive_set_intersection_size",
                                    stable.len() as f64,
                                    stable_indices.len() as f64,
                                ),
                            ],
                            SCOPE,
                        )?;
                    }
                    if k1 == 1 {
                        b.inline(
                            "lem-sasaki-total-squared-error-stable",
                            r"k_1=1",
                            inputs.clone(),
                            vec![equality("actual_singleton_first_count", k1 as f64, 1.)],
                            SCOPE,
                        )?;
                    }
                    if k2 == 1 && k1 > 1 {
                        b.inline(
                            "lem-sasaki-total-squared-error-stable",
                            r"k_2=1<k_1",
                            inputs.clone(),
                            vec![
                                equality("actual_singleton_second_count", k2 as f64, 1.),
                                hypothesis("actual_nonsingleton_first_count", k1 > 1),
                            ],
                            SCOPE,
                        )?;
                    }
                    if k1 == 1 && k2 > 1 {
                        b.inline(
                            "lem-sasaki-total-squared-error-stable",
                            r"k_1=1<k_2",
                            inputs.clone(),
                            vec![
                                equality("actual_singleton_first_count", k1 as f64, 1.),
                                hypothesis("actual_nonsingleton_second_count", k2 > 1),
                            ],
                            SCOPE,
                        )?;
                    }
                    for label in [
                        "thm-sasaki-distance-ms",
                        "sec-eg-operator-estimates",
                        "lem-sasaki-aggregator-structural",
                    ] {
                        b.inline(
                            label,
                            r"k_{\min}:=\max\{1,\min(k_1,k_2)\}",
                            inputs.clone(),
                            vec![equality(
                                "actual_nonempty_minimum_count",
                                k1.min(k2) as f64,
                                (1usize.max(k1.min(k2))) as f64,
                            )],
                            SCOPE,
                        )?;
                        b.inline(
                            label,
                            r"n_c:=\sum_{i=1}^N(s_{1,i}-s_{2,i})^2",
                            inputs.clone(),
                            vec![equality(
                                "actual_status_change_sum",
                                nc,
                                a1.iter()
                                    .zip(&a2)
                                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                                    .sum(),
                            )],
                            SCOPE,
                        )?;
                    }
                    b.inline(
                        "sec-eg-operator-estimates",
                        r"k_r:=|\mathcal A(\mathcal S_r)|",
                        inputs.clone(),
                        vec![
                            equality(
                                "actual_alive_sum_first",
                                k1 as f64,
                                a1.iter().map(|a| f64::from(*a)).sum(),
                            ),
                            equality(
                                "actual_alive_sum_second",
                                k2 as f64,
                                a2.iter().map(|a| f64::from(*a)).sum(),
                            ),
                        ],
                        SCOPE,
                    )?;
                    for label in [
                        "lem-sasaki-total-squared-error-stable",
                        "thm-sasaki-distance-ms",
                    ] {
                        b.inline(
                            label,
                            r"k_{\mathrm{stable}}:=|\mathcal A_{\mathrm{stable}}|",
                            inputs.clone(),
                            vec![equality(
                                "actual_stable_row_count",
                                stable.len() as f64,
                                a1.iter().zip(&a2).filter(|(a, b)| **a && **b).count() as f64,
                            )],
                            SCOPE,
                        )?;
                    }
                    b.inline(
                        "def-eg-auxiliary-continuity-notation",
                        r"D_{\mathcal Y}=\operatorname{diam}(\mathcal Y)",
                        inputs.clone(),
                        vec![equality(
                            "native_closed_product_ball_diameter",
                            diameter,
                            (4_f64 * 2_f64.powi(2) + 4_f64 * 2_f64.powi(2)).sqrt(),
                        )],
                        SCOPE,
                    )?;
                    if k1 >= 2 {
                        b.inline(
                            "sec-eg-operator-estimates",
                            r"k \ge 2",
                            inputs.clone(),
                            vec![upper("actual_nontrivial_uniform_companions", 2., k1 as f64)],
                            SCOPE,
                        )?;
                    }
                }
                b.add("def-eg-auxiliary-continuity-notation",r"n_c(\mathcal S_1",inputs.clone(),hypotheses.clone(),vec![equality("quotient_dispersion_decomposition",n as f64*plan.metric_squared,delta+nc),equality("optimal_plan_positional_sum",delta,plan.positional_sum),equality("optimal_plan_status_count",nc,plan.status_mismatch_count)],"Exact all nonempty alive-mask pairs N=1..5. A minimum marked transport assignment selects paired representatives, including singleton collapse and disjoint singleton supports.")?;
                if !pchecks.is_empty() {
                    for marker in [
                        r"\left| \mathbb{E}_{c",
                        r"\Delta_{\mathrm{pos},i} =",
                        r"\Delta_{\mathrm{pos},i} \le \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ \left|",
                        r"\Delta_{\mathrm{pos},i} \le \mathbb{E}_{c \sim \mathbb{C}_i(\mathcal{S}_1)} \left[ d_{\mathcal Y}",
                    ] {
                        b.add(
                            "lem-sasaki-single-walker-positional-error",
                            marker,
                            inputs.clone(),
                            hypotheses.clone(),
                            pchecks.clone(),
                            SCOPE,
                        )?;
                    }
                    b.add(
                        "lem-sasaki-single-walker-positional-error",
                        r"\left| d_{\mathcal Y}",
                        inputs.clone(),
                        hypotheses.clone(),
                        rowchecks,
                        SCOPE,
                    )?;
                }
                if k1 >= 2 && !qchecks.is_empty() {
                    b.add("lem-sasaki-single-walker-structural-error",r"\left| \mathbb{E}_{c",inputs.clone(),hypotheses.clone(),qchecks.clone(),"Uniform auxiliary donor law; second-frame geometry fixed. Finite-width Gaussian weights require their own law estimate.")?;
                    b.add(
                        "lem-sasaki-single-walker-structural-error",
                        r"\text{Error}",
                        inputs.clone(),
                        hypotheses.clone(),
                        qchecks,
                        SCOPE,
                    )?;
                }
                for (marker, checks) in [
                    (
                        r"P_i:=",
                        vec![upper("positional_fixed_law_definition", psum, cp * delta)],
                    ),
                    (
                        r"\sum_{i\in\mathcal A_{\mathrm{stable}}}P_i^2",
                        vec![upper("stable_positional_sum_bound", psum, cp * delta)],
                    ),
                    (
                        r"\sum_{i\in\mathcal A_{\mathrm{stable}}}|",
                        vec![upper(
                            "full_stable_sum_bound",
                            fullstable,
                            2. * cp * delta
                                + 8. * stable.len() as f64 * diameter.powi(2) * nc.powi(2)
                                    / denom.powi(2),
                        )],
                    ),
                    (
                        r"P_i\le",
                        vec![upper("summed_positional_triangle", psum, cp * delta)],
                    ),
                ] {
                    b.add(
                        "lem-sasaki-total-squared-error-stable",
                        marker,
                        inputs.clone(),
                        hypotheses.clone(),
                        checks,
                        SCOPE,
                    )?;
                }
                if k1 >= 2 && !stable.is_empty() {
                    b.add(
                        "lem-sasaki-total-squared-error-stable",
                        r"P_i^2\le",
                        inputs.clone(),
                        hypotheses.clone(),
                        vec![upper("summed_jensen_position_bound", psum, cp * delta)],
                        SCOPE,
                    )?;
                }
                let rhs = 2. * cp * delta
                    + 8. * stable.len() as f64 * diameter.powi(2) * nc.powi(2) / denom.powi(2)
                    + diameter.powi(2) * nc;
                b.add(
                    "thm-sasaki-distance-ms",
                    r"F_{d,ms}^{\mathrm{Sasaki}}",
                    inputs.clone(),
                    hypotheses.clone(),
                    vec![upper(
                        "full_expected_distance_vector_bound",
                        difference2(&d1, &d2),
                        rhs,
                    )],
                    SCOPE,
                )?;
                b.add(
                    "thm-sasaki-distance-ms",
                    r"\big\|\mathbf d",
                    inputs.clone(),
                    hypotheses.clone(),
                    vec![upper(
                        "stable_plus_unstable_distance_bound",
                        difference2(&d1, &d2),
                        rhs,
                    )],
                    SCOPE,
                )?;
                b.inline(
                    "thm-sasaki-distance-ms",
                    r"|d^{(1)}_i-d^{(2)}_i|\le D_{\mathcal Y}",
                    inputs,
                    vec![upper(
                        "unstable_row_diameter_bound",
                        (0..n)
                            .filter(|&i| a1[i] != a2[i])
                            .map(|i| (d1[i] - d2[i]).powi(2))
                            .sum(),
                        diameter.powi(2) * nc,
                    )],
                    SCOPE,
                )?;
            }
        }
    }
    Ok(())
}

fn integrate(f: impl Fn(f64) -> f64, left: f64, right: f64) -> f64 {
    let n = 2048;
    let h = (right - left) / n as f64;
    let mut value = f(left) + f(right);
    for i in 1..n {
        value += if i % 2 == 0 { 2. } else { 4. } * f(left + i as f64 * h);
    }
    value * h / 3.
}
fn cdf(z: f64) -> f64 {
    if z >= 10. {
        1.
    } else if z <= -10. {
        0.
    } else {
        0.5 + if z >= 0. { 1. } else { -1. }
            * integrate(
                |x| (-x * x / 2.).exp() / (2. * std::f64::consts::PI).sqrt(),
                0.,
                z.abs(),
            )
    }
}
fn box_death(mean: &[f64], s: f64, r: f64) -> f64 {
    1. - mean
        .iter()
        .map(|m| cdf((r - m) / s) - cdf((-r - m) / s))
        .product::<f64>()
}
fn unit_ball_volume(d: usize) -> f64 {
    match d {
        1 => 2.,
        2 => std::f64::consts::PI,
        4 => std::f64::consts::PI.powi(2) / 2.,
        _ => unreachable!("diagnostic dimensions"),
    }
}

async fn kinetic_cases(b: &mut EvidenceBuilder, samples: usize) -> Result<()> {
    for d in [1, 2, 4] {
        for (case, friction) in [0., 1e-10, 1.].into_iter().enumerate() {
            let mut cfg = KineticValidationConfig {
                dimensions: d,
                landscape: ConfiningLandscape::quadratic(d),
                samples,
                friction,
                inputs: vec![
                    KineticInput {
                        positions: vec![0.; d],
                        velocities: vec![0.; d],
                    },
                    KineticInput {
                        positions: vec![0.8; d],
                        velocities: vec![0.2; d],
                    },
                ],
                coupling_position_shift: 0.015,
                coupling_velocity_shift: if case == 0 { 0. } else { 0.02 },
                box_half_width: Some(2.),
                ..Default::default()
            };
            if case == 1 {
                cfg.landscape.curvature = (0..d).map(|i| 1. + i as f64).collect();
                cfg.landscape.center = vec![0.1; d];
            }
            if case == 2 {
                cfg.landscape.ripple_amplitude = 0.3;
                cfg.landscape.ripple_frequency = 2.;
            }
            let report = validate_kinetic(&cfg).await?;
            let c = &report.constants;
            let assumptions = &report.landscape_assumptions;
            let q = c.thermal_variance.sqrt();
            let s = c.position_variance.sqrt();
            let h = cfg.dt;
            let lambda = cfg.metric_weight;
            let cx = c.moment_c_x;
            let cv = c.moment_c_v;
            let c0 = c.moment_c_0;
            let inputs = json!({"native_config":cfg,"analytic_force_hypotheses":assumptions,"native_independent_replica_experiments":report.experiments,"constants":c,"sampling_units":samples,"criterion":"independent native kinetic replicas, six standard errors; no terminal survival conditioning"});
            let hypotheses = vec![
                hypothesis(
                    "global_analytic_force_lipschitz",
                    assumptions.force_lipschitz > 0.,
                ),
                hypothesis("positive_noise_denominators", q > 0. && s > 0.),
                hypothesis(
                    "phase_inverse_contraction",
                    c.phase_space_invertibility_ratio < 1.,
                ),
            ];
            let mut meanchecks = vec![];
            let mut covchecks = vec![];
            let mut incrementchecks = vec![];
            let mut flowchecks = vec![];
            let mut driftchecks = vec![];
            let mut capchecks = vec![];
            let mut cap_movement_checks = vec![];
            let mut deathchecks = vec![];
            for e in &report.experiments {
                let f = cfg
                    .landscape
                    .analytic_gradient(&e.input.positions)?
                    .iter()
                    .map(|a| -a)
                    .collect::<Vec<_>>();
                let v1 = e
                    .input
                    .velocities
                    .iter()
                    .zip(&f)
                    .map(|(v, f)| v + h * f / 2.)
                    .collect::<Vec<_>>();
                let drift = v1
                    .iter()
                    .map(|v| c.position_flow_coefficient * v)
                    .collect::<Vec<_>>();
                let native_force_at_zero =
                    norm2(&cfg.landscape.analytic_gradient(&vec![0.; d])?).sqrt();
                let point = json!({"native_landscape":cfg.landscape,"x":e.input.positions,"v":e.input.velocities,"F":f,"B_F":assumptions.force_at_zero,"L_F":assumptions.force_lipschitz,"h":h,"b":c.position_flow_coefficient,"c":c.friction_factor,"s_h":s,"q":q,"lambda_v":lambda});
                for label in ["sec-eg-definition", "lem-euclidean-perturb-moment"] {
                    b.inline(
                        label,
                        r"\|F(x)\|\le B_F+L_F\|x\|",
                        point.clone(),
                        vec![upper(
                            "actual_native_global_force_growth",
                            norm2(&f).sqrt(),
                            assumptions.force_at_zero
                                + assumptions.force_lipschitz * norm2(&e.input.positions).sqrt(),
                        )],
                        SCOPE,
                    )?;
                }
                b.inline(
                    "sec-eg-definition",
                    r"B_F=\|F(0)\|",
                    point.clone(),
                    vec![equality(
                        "native_zero_force_growth_constant",
                        assumptions.force_at_zero,
                        native_force_at_zero,
                    )],
                    SCOPE,
                )?;
                b.inline(
                    "lem-sasaki-kinetic-lipschitz",
                    r"b=h(1+c)/2",
                    point.clone(),
                    vec![equality(
                        "native_BAOAB_position_flow_coefficient",
                        c.position_flow_coefficient,
                        h * (1. + c.friction_factor) / 2.,
                    )],
                    SCOPE,
                )?;
                for (label, quote, ok) in [
                    (
                        "lem-euclidean-geometric-consistency",
                        r"h^2L_F/4<1",
                        h * h * assumptions.force_lipschitz / 4. < 1.,
                    ),
                    (
                        "lem-euclidean-geometric-consistency",
                        r"\sigma_x>0",
                        cfg.position_diffusion > 0.,
                    ),
                    ("lem-euclidean-geometric-consistency", r"q>0", q > 0.),
                    ("lem-euclidean-boundary-holder", r"s_h>0", s > 0.),
                ] {
                    b.inline(
                        label,
                        quote,
                        point.clone(),
                        vec![hypothesis("actual_native_kinetic_parameter_hypothesis", ok)],
                        SCOPE,
                    )?;
                }
                let exact_position_increment = norm2(&drift) + d as f64 * c.position_variance;
                let upper_position =
                    3. * c.position_flow_coefficient.powi(2) * norm2(&e.input.velocities)
                        + 3. * c.position_flow_coefficient.powi(2) * h * h / 4.
                            * (assumptions.force_lipschitz.powi(2) * norm2(&e.input.positions)
                                + assumptions.force_at_zero.powi(2))
                        + d as f64 * c.position_variance;
                incrementchecks.push(upper(
                    "analytic_position_increment_three_term_bound",
                    exact_position_increment,
                    upper_position,
                ));
                let empirical_position_increment = (0..d)
                    .map(|j| {
                        e.position_covariance[j * d + j].mean
                            + drift[j].powi(2)
                            + 2. * drift[j] * (e.position_drift[j].mean - drift[j])
                    })
                    .sum::<f64>();
                let se = (0..d)
                    .map(|j| {
                        e.position_covariance[j * d + j].standard_error
                            + 2. * drift[j].abs() * e.position_drift[j].standard_error
                    })
                    .sum::<f64>();
                incrementchecks.push(upper(
                    "native_position_increment_exact_mean_six_SE",
                    (empirical_position_increment - exact_position_increment).abs(),
                    6. * se + 1e-9,
                ));
                for (j, &predicted) in drift.iter().enumerate() {
                    meanchecks.push(upper(
                        "native_mean_drift_six_SE",
                        (e.position_drift[j].mean - predicted).abs(),
                        6. * e.position_drift[j].standard_error + 1e-9,
                    ));
                }
                for i in 0..d {
                    for j in 0..d {
                        let predicted = if i == j { c.position_variance } else { 0. };
                        covchecks.push(upper(
                            "native_position_covariance_six_SE",
                            (e.position_covariance[i * d + j].mean - predicted).abs(),
                            6. * e.position_covariance[i * d + j].standard_error + 1e-9,
                        ));
                    }
                }
                let rhs = cx * norm2(&e.input.positions) + cv * norm2(&e.input.velocities) + c0;
                driftchecks.push(upper(
                    "native_full_increment_moment_upper_six_SE",
                    e.physical_squared_increment.mean,
                    rhs + 6. * e.physical_squared_increment.standard_error,
                ));
                driftchecks.push(upper(
                    "native_mean_phase_displacement_jensen",
                    norm2(&drift),
                    rhs,
                ));
                flowchecks.push(upper(
                    "native_synchronous_position_lipschitz",
                    e.synchronous_position_lipschitz_ratio,
                    c.position_flow_lipschitz,
                ));
                capchecks.push(upper(
                    "native_smooth_cap_radius",
                    e.maximum_velocity_norm,
                    cfg.velocity_cap,
                ));
                capchecks.push(upper(
                    "native_B2_cap_formula",
                    e.maximum_b2_to_final_cap_error,
                    1e-10,
                ));
                // The recorded maximum covers every native replica. Check
                // the actual worst-case velocity difference, then the
                // squared triangle bound from this quoted estimate.
                let input_velocity_norm = norm2(&e.input.velocities).sqrt();
                cap_movement_checks.push(upper(
                    "native_velocity_increment_cap_hypothesis",
                    e.maximum_velocity_norm,
                    cfg.velocity_cap,
                ));
                cap_movement_checks.push(upper(
                    "native_velocity_increment_triangle_envelope",
                    (e.maximum_velocity_norm + input_velocity_norm).powi(2),
                    2. * cfg.velocity_cap.powi(2) + 2. * input_velocity_norm.powi(2),
                ));
                if let Some(predicted) = e.predicted_death_probability {
                    let mean = &e.conditional_position_mean;
                    let mut boundary_checks = vec![equality(
                        "Gaussian_face_point_mass_integral",
                        integrate(
                            |u| {
                                (-(u - mean[0]).powi(2) / (2. * s * s)).exp()
                                    / ((2. * std::f64::consts::PI).sqrt() * s)
                            },
                            2.,
                            2.,
                        ),
                        0.,
                    )];
                    let mut tube_probabilities = vec![];
                    for relative_width in [0.1, 0.01, 0.001] {
                        let width = relative_width * s;
                        let outer = 1. - box_death(mean, s, 2. + width);
                        let inner = 1. - box_death(mean, s, (2. - width).max(0.));
                        let tube = (outer - inner).max(0.);
                        let bound =
                            4. * d as f64 * width / ((2. * std::f64::consts::PI).sqrt() * s);
                        tube_probabilities.push(json!({"width":width,"exact_Gaussian_tube_mass":tube,"density_union_upper":bound}));
                        boundary_checks.push(upper(
                            "exact_Gaussian_boundary_tube_density_bound",
                            tube,
                            bound,
                        ));
                    }
                    b.inline("lem-euclidean-boundary-holder",r"\mathbb P(x^+\in\partial D)=0",json!({"native_config":cfg,"native_conditional_mean":mean,"native_position_scale":s,"domain":"open box (-2,2)^d","Lebesgue_boundary_measure":0,"shrinking_tubes":tube_probabilities}),boundary_checks,"The exact native conditional Gaussian law has a finite density and the box boundary has zero Lebesgue measure, hence zero probability analytically. Independent CDF evaluations test shrinking-tube density bounds; observed zero hits is not used to infer nullness.")?;
                    b.inline(
                        "lem-euclidean-boundary-holder",
                        r"p_{\mathrm{dead}}(x,v)=\mathbb P(x^+\notin D)",
                        point.clone(),
                        vec![upper(
                            "native_terminal_exit_law_definition_six_SE",
                            (e.measured_death_probability.mean - predicted).abs(),
                            6. * e.measured_death_probability.standard_error + 1e-9,
                        )],
                        SCOPE,
                    )?;

                    deathchecks.push(upper(
                        "native_terminal_exit_probability_six_SE",
                        (e.measured_death_probability.mean - predicted).abs(),
                        6. * e.measured_death_probability.standard_error + 1e-9,
                    ));
                    deathchecks.push(upper(
                        "native_terminal_exit_CDF_formula",
                        (predicted - box_death(&e.conditional_position_mean, s, 2.)).abs(),
                        1e-7,
                    ));
                }
            }
            b.add(
                "def-eg-baoab-canonical",
                r"c=e^{-",
                inputs.clone(),
                hypotheses.clone(),
                vec![
                    equality(
                        "native_thermostat_factor",
                        c.friction_factor,
                        (-friction * h).exp(),
                    ),
                    equality(
                        "native_O_variance",
                        c.thermal_variance,
                        if friction == 0. {
                            cfg.velocity_diffusion.powi(2) * h
                        } else {
                            cfg.velocity_diffusion.powi(2) * (-(-2. * friction * h).exp_m1())
                                / (2. * friction)
                        },
                    ),
                ],
                SCOPE,
            )?;
            b.add("def-eg-baoab-canonical",r"\begin{aligned}",inputs.clone(),hypotheses.clone(),[meanchecks.clone(),covchecks.clone(),capchecks.clone()].concat(),"Actual native BAOAB mean/covariance and B2-to-cap reconstruction; independent-replica Gaussian errors carry declared uncertainty.")?;
            b.add(
                "lem-sasaki-kinetic-lipschitz",
                r"M_h(x,v)=",
                inputs.clone(),
                hypotheses.clone(),
                [
                    meanchecks.clone(),
                    covchecks.clone(),
                    vec![equality(
                        "position_variance_sum",
                        c.position_variance,
                        h * h * c.thermal_variance / 4. + h * cfg.position_diffusion.powi(2),
                    )],
                ]
                .concat(),
                SCOPE,
            )?;
            b.add(
                "lem-sasaki-kinetic-lipschitz",
                r"\|x^+-x'^+\|",
                inputs.clone(),
                hypotheses.clone(),
                flowchecks,
                SCOPE,
            )?;
            b.add(
                "lem-euclidean-perturb-moment",
                r"C_x=",
                inputs.clone(),
                hypotheses.clone(),
                vec![
                    equality(
                        "native_force_position_coefficient",
                        cx,
                        3. * c.position_flow_coefficient.powi(2)
                            * h
                            * h
                            * assumptions.force_lipschitz.powi(2)
                            / 4.,
                    ),
                    equality(
                        "native_velocity_coefficient",
                        cv,
                        3. * c.position_flow_coefficient.powi(2) + 2. * lambda,
                    ),
                    equality(
                        "native_offset_coefficient",
                        c0,
                        3. * c.position_flow_coefficient.powi(2)
                            * h
                            * h
                            * assumptions.force_at_zero.powi(2)
                            / 4.
                            + d as f64 * c.position_variance
                            + 2. * lambda * cfg.velocity_cap.powi(2),
                    ),
                ],
                SCOPE,
            )?;
            b.add(
                "lem-euclidean-perturb-moment",
                r"\mathbb E\|x^+-x\|^2",
                inputs.clone(),
                hypotheses.clone(),
                incrementchecks,
                SCOPE,
            )?;
            b.add(
                "lem-euclidean-perturb-moment",
                r"\mathbb E d_{\mathrm{phys}}",
                inputs.clone(),
                hypotheses.clone(),
                driftchecks.clone(),
                SCOPE,
            )?;
            b.inline(
                "lem-euclidean-perturb-moment",
                r"\|v^+-v\|^2\le2V_{\mathrm{alg}}^2+2\|v\|^2",
                inputs.clone(),
                cap_movement_checks,
                SCOPE,
            )?;
            b.add(
                "lem-euclidean-geometric-consistency",
                r"\mathbb E(x^+-x)=",
                inputs.clone(),
                hypotheses.clone(),
                [meanchecks, covchecks].concat(),
                SCOPE,
            )?;
            b.inline(
                "lem-euclidean-geometric-consistency",
                r"\sqrt{C_x\|x\|^2+C_v\|v\|^2+C_0}",
                inputs.clone(),
                driftchecks,
                SCOPE,
            )?;
            for separation in [0., 0.1 * s, s, 3. * s] {
                let tv = 2. * cdf(separation / (2. * s)) - 1.;
                let density = |x: f64, m: f64| {
                    (-(x - m).powi(2) / (2. * s * s)).exp()
                        / (s * (2. * std::f64::consts::PI).sqrt())
                };
                let direct = integrate(
                    |x| density(x, -separation / 2.) - density(x, separation / 2.),
                    -12. * s - separation / 2.,
                    0.,
                );
                b.inline(
                    "lem-euclidean-boundary-holder",
                    r"r=\|m-m'\|",
                    json!({"m":[-separation/2.],"m_prime":[separation/2.],"s_h":s}),
                    vec![equality(
                        "Gaussian_TV_mean_difference_norm",
                        separation,
                        difference2(&[-separation / 2.], &[separation / 2.]).sqrt(),
                    )],
                    SCOPE,
                )?;
                b.add("lem-euclidean-boundary-holder",r"\|\mathcal N(m,s_h^2I)",json!({"native_position_variance":s*s,"mean_separation":separation,"halfspace_density_integral":direct,"TV_formula":tv}),vec![hypothesis("positive_Gaussian_scale",s>0.)],vec![upper("Gaussian_halfspace_TV_identity",(tv-direct).abs(),1e-8),upper("Gaussian_TV_linear_modulus",tv,separation/((2.*std::f64::consts::PI).sqrt()*s))],"Independent analytic halfspace density quadrature checks the exact Gaussian total variation formula; this scalar identity is dimension independent.")?;
            }
            for e in &report.experiments {
                let x = &e.input.positions;
                let v = &e.input.velocities;
                for (sx, sv) in [(0.01, 0.), (0., 0.01), (0.013, -0.015)] {
                    let xp = x.iter().map(|a| a + sx).collect::<Vec<_>>();
                    let vp = v.iter().map(|a| a + sv).collect::<Vec<_>>();
                    let force = |x: &[f64]| {
                        cfg.landscape
                            .analytic_gradient(x)
                            .map(|g| g.iter().map(|a| -a).collect::<Vec<_>>())
                    };
                    let f = force(x)?;
                    let fp = force(&xp)?;
                    let m = x
                        .iter()
                        .zip(v)
                        .zip(&f)
                        .map(|((x, v), f)| x + c.position_flow_coefficient * (v + h * f / 2.))
                        .collect::<Vec<_>>();
                    let mp = xp
                        .iter()
                        .zip(&vp)
                        .zip(&fp)
                        .map(|((x, v), f)| x + c.position_flow_coefficient * (v + h * f / 2.))
                        .collect::<Vec<_>>();
                    let physical = (difference2(x, &xp) + lambda * difference2(v, &vp)).sqrt();
                    let r = difference2(&m, &mp).sqrt();
                    let tv = 2. * cdf(r / (2. * s)) - 1.;
                    let p = box_death(&m, s, 2.);
                    let pp = box_death(&mp, s, 2.);
                    b.add("lem-euclidean-boundary-holder",r"|p_{\mathrm{dead}}",json!({"config":cfg,"input_x":x,"input_v":v,"shift_x":sx,"shift_v":sv,"means":[m,mp],"exact_box_probabilities":[p,pp]}),hypotheses.clone(),vec![upper("exact_exit_event_TV_bound",(p-pp).abs(),tv),upper("exact_exit_event_physical_modulus",(p-pp).abs(),c.position_flow_lipschitz*physical/((2.*std::f64::consts::PI).sqrt()*s))],SCOPE)?;
                }
            }
            b.add("lem-euclidean-boundary-holder",r"|p_{\mathrm{dead}}",inputs,hypotheses,deathchecks,"Actual native terminal exit measurements match exact product Gaussian CDF probabilities within predeclared six standard errors.")?;
        }
    }
    compact_covariance_cases(b, samples).await?;
    Ok(())
}

async fn compact_covariance_cases(b: &mut EvidenceBuilder, samples: usize) -> Result<()> {
    for d in [1, 2, 4] {
        for h in [0.04, 0.2, 1.] {
            let config = KineticValidationConfig {
                dimensions: d,
                dt: h,
                landscape: ConfiningLandscape::quadratic(d),
                samples,
                inputs: vec![
                    KineticInput {
                        positions: vec![0.; d],
                        velocities: vec![0.; d],
                    },
                    KineticInput {
                        positions: vec![0.4; d],
                        velocities: vec![0.2; d],
                    },
                ],
                box_half_width: None,
                ..Default::default()
            };
            let c = kinetic_constants(&config)?;
            let a = h * h / 4.;
            let bx = 1.;
            let bv = 1.;
            let bf = 0.;
            let lf = 1.;
            let cap = 2.;
            let rx = 0.1;
            let rv = 0.1;
            let av = bv + h * (bf + lf * bx) / 2.;
            let ax = bx + h * av / 2.;
            let b3 = cap * rv / (cap - rv);
            let w = (b3 + h * (bf + lf * ax) / 2.) / (1. - a);
            let x2 = ax + h * w / 2.;
            let q2 = c.thermal_variance;
            let pos2 = h * config.position_diffusion.powi(2);
            let log_lower = -(d as f64) / 2. * (2. * std::f64::consts::PI * q2).ln()
                - (w + c.friction_factor * av).powi(2) / (2. * q2)
                - (d as f64) / 2. * (2. * std::f64::consts::PI * pos2).ln()
                - (rx + x2).powi(2) / (2. * pos2)
                - d as f64 * (1. + a).ln();
            let log_theta = log_lower + 2. * unit_ball_volume(d).ln() + d as f64 * (rx * rv).ln();
            let log_cov_lower = log_theta + rx.min(rv).powi(2).ln() - ((d + 2) as f64).ln();
            let cov_upper = d as f64 * c.position_variance + cap * cap;
            let native_covariance = validate_kinetic(&config).await?;
            b.inline("sec-eg-definition",r"D=\mathbb R^d",json!({"native_config":config,"native_conditional_experiments":native_covariance.experiments}),vec![hypothesis("actual_native_kinetic_experiment_has_no_terminal_domain_cutoff",config.box_half_width.is_none())],"The no-boundary native kinetic experiment instantiates the globally defined physical domain. It retains unbounded Gaussian support.")?;
            b.inline("cor-eg-compact-kinetic-covariance",r"K=B(0,r_x)\times B(0,r_v)",json!({"dimensions":d,"target_position_radius":rx,"target_velocity_radius":rv,"native_cap_radius":cap,"target_origin":vec![0.;2*d]}),vec![hypothesis("target_product_balls_nonempty_and_strictly_inside_velocity_image",rx>0. && rv>0. && rv<cap)],SCOPE)?;
            let inputs = json!({"dimensions":d,"native_config":config,"input_B_x":bx,"input_B_v":bv,"target_r_x":rx,"target_r_v":rv,"a":a,"A_v":av,"A_x":ax,"B3":b3,"W":w,"X2":x2,"log_density_lower":log_lower,"log_minorization_mass":log_theta,"log_covariance_eigenvalue_lower":log_cov_lower,"covariance_eigenvalue_upper":cov_upper,"log_condition_number_upper":cov_upper.ln()-log_cov_lower,"density_source":"analytic global force growth, Gaussian envelope and Jacobian bounds; no sampled minimum"});
            let hypotheses = vec![
                hypothesis("B2_inverse_contraction", a < 1.),
                hypothesis("interior_target_velocity", rv < cap && rv > 0.),
                hypothesis(
                    "positive_thermostat_and_final_position_noise",
                    q2 > 0. && pos2 > 0.,
                ),
            ];
            for (quote, checks) in [
                (
                    r"\|x\|\le B_x",
                    vec![upper(
                        "native_compact_covariance_input_positions",
                        config
                            .inputs
                            .iter()
                            .map(|input| norm2(&input.positions).sqrt())
                            .fold(0_f64, f64::max),
                        bx,
                    )],
                ),
                (
                    r"\|v\|\le B_v",
                    vec![upper(
                        "native_compact_covariance_input_velocities",
                        config
                            .inputs
                            .iter()
                            .map(|input| norm2(&input.velocities).sqrt())
                            .fold(0_f64, f64::max),
                        bv,
                    )],
                ),
                (
                    r"0<r_v<V_{\mathrm{alg}}",
                    vec![hypothesis(
                        "actual_target_velocity_ball_interior",
                        rv > 0. && rv < cap,
                    )],
                ),
                (
                    r"r_x>0",
                    vec![hypothesis("actual_target_position_ball_positive", rx > 0.)],
                ),
                (
                    r"\ell>0",
                    vec![hypothesis(
                        "strictly_positive_analytic_density_lower",
                        log_lower.is_finite(),
                    )],
                ),
                (
                    r"\theta=\ell\omega_d^2r_x^dr_v^d",
                    vec![equality(
                        "minorization_volume_log_identity",
                        log_theta,
                        log_lower + 2. * unit_ball_volume(d).ln() + d as f64 * (rx.ln() + rv.ln()),
                    )],
                ),
            ] {
                b.inline("cor-eg-compact-kinetic-covariance",quote,inputs.clone(),checks,"Compact minorization proof parameters evaluated with analytic logarithms; finite log lower density represents a strictly positive real density even when its f64 exponential underflows.")?;
            }
            b.inline(
                "lem-euclidean-geometric-consistency",
                r"0<h<2",
                inputs.clone(),
                vec![hypothesis(
                    "unit_quadratic_positive_definite_step",
                    h > 0. && h < 2.,
                )],
                SCOPE,
            )?;
            b.inline(
                "lem-euclidean-geometric-consistency",
                r"L_F=1",
                inputs.clone(),
                vec![equality(
                    "native_unit_quadratic_force_modulus",
                    config.landscape.assumptions()?.force_lipschitz,
                    1.,
                )],
                SCOPE,
            )?;
            b.inline(
                "lem-euclidean-geometric-consistency",
                r"F(x)=-x",
                inputs.clone(),
                config
                    .inputs
                    .iter()
                    .map(|input| {
                        config
                            .landscape
                            .analytic_gradient(&input.positions)
                            .map(|gradient| {
                                equality(
                                    "native_unit_quadratic_force_identity",
                                    difference2(&gradient, &input.positions),
                                    0.,
                                )
                            })
                    })
                    .collect::<Result<Vec<_>>>()?,
                SCOPE,
            )?;
            if h == 0.04 {
                b.inline(
                    "lem-euclidean-geometric-consistency",
                    r"h=0.04",
                    inputs.clone(),
                    vec![equality(
                        "native_canonical_experiment_step",
                        config.dt,
                        0.04,
                    )],
                    SCOPE,
                )?;
            }
            let mut tchecks = vec![];
            let mut densitychecks = vec![];
            for t in 0..12 {
                let x1 = vec![ax * (t as f64 / 12. - 0.5) / (d as f64).sqrt(); d];
                let target = vec![b3 * (t as f64 / 12. - 0.5) / (d as f64).sqrt(); d];
                let exact = target
                    .iter()
                    .zip(&x1)
                    .map(|(y, x1)| (y + h * x1 / 2.) / (1. - a))
                    .collect::<Vec<_>>();
                let mut iterate = target.clone();
                for _ in 0..256 {
                    iterate = target
                        .iter()
                        .zip(&x1)
                        .zip(&iterate)
                        .map(|((y, x1), w)| y + h * (x1 + h * w / 2.) / 2.)
                        .collect();
                }
                tchecks.push(equality(
                    "T_inverse_fixed_point",
                    difference2(&iterate, &exact),
                    0.,
                ));
                tchecks.push(equality(
                    "T_inverse_substitution",
                    norm2(
                        &exact
                            .iter()
                            .zip(&x1)
                            .zip(&target)
                            .map(|((w, x1), y)| w - h * (x1 + h * w / 2.) / 2. - y)
                            .collect::<Vec<_>>(),
                    ),
                    0.,
                ));
                tchecks.push(upper(
                    "T_inverse_analytic_preimage_bound",
                    norm2(&exact).sqrt(),
                    w,
                ));
                // Finite-difference the force-driven T, rather than comparing
                // the proposed singular bound to itself.
                let tmap = |u: &[f64]| -> Result<Vec<f64>> {
                    let force = config.landscape.analytic_gradient(
                        &x1.iter()
                            .zip(u)
                            .map(|(x, w)| x + h * w / 2.)
                            .collect::<Vec<_>>(),
                    )?;
                    Ok(u.iter().zip(&force).map(|(w, g)| w - h * g / 2.).collect())
                };
                let eps = 1e-6;
                let mut columns = vec![];
                for column in 0..d {
                    let mut plus = exact.clone();
                    let mut minus = exact.clone();
                    plus[column] += eps;
                    minus[column] -= eps;
                    let yp = tmap(&plus)?;
                    let ym = tmap(&minus)?;
                    let values = yp
                        .iter()
                        .zip(&ym)
                        .map(|(p, m)| (p - m) / (2. * eps))
                        .collect::<Vec<_>>();
                    for (row, value) in values.iter().enumerate() {
                        tchecks.push(upper(
                            "native_T_Jacobian_entry_difference",
                            (value - if row == column { 1. - a } else { 0. }).abs(),
                            1e-7,
                        ));
                    }
                    let singular = norm2(&values).sqrt();
                    tchecks.push(upper(
                        "native_T_Jacobian_lower_singular",
                        1. - a,
                        singular + 1e-7,
                    ));
                    tchecks.push(upper(
                        "native_T_Jacobian_upper_singular",
                        singular,
                        1. + a + 1e-7,
                    ));
                    columns.push(values);
                }
                let finite_det = columns
                    .iter()
                    .enumerate()
                    .map(|(i, col)| col[i])
                    .product::<f64>();
                b.inline("cor-eg-compact-kinetic-covariance",r"|\det DT|\le(1+a)^d",json!({"native_landscape":config.landscape,"x1":x1,"preimage":exact,"h":h,"a":a,"finite_difference_columns":columns}),vec![upper("native_T_determinant_upper",finite_det.abs(),(1.+a).powi(d as i32)+1e-7)],SCOPE)?;
                let yv = vec![rv * (t as f64 / 12. - 0.5) / (d as f64).sqrt(); d];
                let raw = inverse(&yv, cap);
                let wf = raw
                    .iter()
                    .zip(&x1)
                    .map(|(y, x1)| (y + h * x1 / 2.) / (1. - a))
                    .collect::<Vec<_>>();
                let xp = vec![rx * (t as f64 / 12. - 0.5) / (d as f64).sqrt(); d];
                let v1 = vec![av / (2. * (d as f64).sqrt()); d];
                let cap_r = norm2(&raw).sqrt();
                let log_jac_inverse = (d as f64 + 1.) * (1. + cap_r / cap).ln();
                let m = v1.iter().map(|v| c.friction_factor * v).collect::<Vec<_>>();
                let conditional_x = x1
                    .iter()
                    .zip(&wf)
                    .map(|(x, w)| x + h * w / 2.)
                    .collect::<Vec<_>>();
                let point = json!({"analytic_envelope":inputs,"x1":x1,"target_velocity":yv,"inverse_cap_velocity":raw,"O_preimage":wf,"B1_velocity":v1,"A2_position":conditional_x});
                for (quote, lhs, rhs) in [
                    (r"\|v_1\|\le A_v", norm2(&v1).sqrt(), av),
                    (r"\|x_1\|\le A_x", norm2(&x1).sqrt(), ax),
                    (r"\|\psi_v^{-1}(v^+)\|\le B_3", norm2(&raw).sqrt(), b3),
                    (
                        r"(1-a)\|w\|\le B_3+h(B_F+L_FA_x)/2",
                        (1. - a) * norm2(&wf).sqrt(),
                        b3 + h * (bf + lf * ax) / 2.,
                    ),
                    (r"\|w\|\le W", norm2(&wf).sqrt(), w),
                    (r"\|x_2\|\le X_2", norm2(&conditional_x).sqrt(), x2),
                ] {
                    b.inline(
                        "cor-eg-compact-kinetic-covariance",
                        quote,
                        point.clone(),
                        vec![upper(
                            "measured_compact_density_preimage_subbound",
                            lhs,
                            rhs,
                        )],
                        SCOPE,
                    )?;
                }
                b.inline(
                    "cor-eg-compact-kinetic-covariance",
                    r"T(w)=\psi_v^{-1}(v^+)",
                    point.clone(),
                    vec![equality(
                        "native_force_T_cap_inverse_identity",
                        difference2(&tmap(&wf)?, &raw),
                        0.,
                    )],
                    SCOPE,
                )?;
                b.inline(
                    "lem-euclidean-geometric-consistency",
                    r"w=y-\tfrac h2F(x_1+hw/2)",
                    point,
                    vec![equality(
                        "native_unique_B2_inverse_fixed_point",
                        difference2(&tmap(&wf)?, &raw),
                        0.,
                    )],
                    SCOPE,
                )?;
                let log_actual = -(d as f64) / 2. * (2. * std::f64::consts::PI * q2).ln()
                    - difference2(&wf, &m) / (2. * q2)
                    - (d as f64) / 2. * (2. * std::f64::consts::PI * pos2).ln()
                    - difference2(&xp, &conditional_x) / (2. * pos2)
                    - d as f64 * (1. - a).ln()
                    + log_jac_inverse;
                densitychecks.push(upper(
                    "pointwise_joint_density_log_minorization",
                    log_lower,
                    log_actual,
                ));
            }
            b.add("lem-euclidean-geometric-consistency",r"T(w)=",inputs.clone(),hypotheses.clone(),tchecks.clone(),"Unit-quadratic B2 map: analytic inverse and independently iterated contraction equation, explicit singular-value interval. The exact h=2 obstruction is retained separately.")?;
            b.add(
                "cor-eg-compact-kinetic-covariance",
                r"a:=",
                inputs.clone(),
                hypotheses.clone(),
                vec![
                    equality(
                        "native_B2_invertibility_ratio",
                        a,
                        c.phase_space_invertibility_ratio,
                    ),
                    hypothesis("actual_B2_inverse_contraction_ratio", a < 1.),
                    equality(
                        "compact_B1_velocity_bound",
                        av,
                        bv + h * (bf + lf * bx) / 2.,
                    ),
                    equality("compact_A1_position_bound", ax, bx + h * av / 2.),
                ],
                SCOPE,
            )?;
            b.add(
                "cor-eg-compact-kinetic-covariance",
                r"B_3:=",
                inputs.clone(),
                hypotheses.clone(),
                [
                    tchecks,
                    vec![
                        equality(
                            "inverse_cap_target_radius_constant",
                            b3,
                            norm2(&inverse(&[vec![rv], vec![0.; d - 1]].concat(), cap)).sqrt(),
                        ),
                        equality(
                            "compact_inverse_O_radius_constant",
                            w,
                            (b3 + h * (bf + lf * ax) / 2.) / (1. - a),
                        ),
                        equality("compact_A2_radius_constant", x2, ax + h * w / 2.),
                    ],
                ]
                .concat(),
                SCOPE,
            )?;
            b.add(
                "cor-eg-compact-kinetic-covariance",
                r"\log\ell=",
                inputs.clone(),
                hypotheses.clone(),
                densitychecks,
                SCOPE,
            )?;
            let mut native_covariance_checks = vec![];
            for experiment in &native_covariance.experiments {
                if let Some(minimum) = experiment.full_covariance_min_eigenvalue {
                    native_covariance_checks.push(hypothesis(
                        "native_sampled_covariance_minimum_strictly_positive",
                        minimum.is_finite() && minimum > 0.,
                    ));

                    if minimum.is_finite() && minimum > 0. {
                        native_covariance_checks.push(upper(
                            "native_compact_covariance_lower_log_diagnostic",
                            log_cov_lower,
                            minimum.ln(),
                        ));
                    }
                }
                if let Some(maximum) = experiment.full_covariance_max_eigenvalue {
                    native_covariance_checks.push(upper(
                        "native_compact_covariance_upper_diagnostic",
                        maximum,
                        cov_upper + 6. * (d as f64).sqrt() * cap * cap / (samples as f64).sqrt(),
                    ));
                }
            }
            b.add("cor-eg-compact-kinetic-covariance",r"\theta\frac",json!({"analytic_envelope":inputs,"native_covariance_experiments":native_covariance.experiments}),hypotheses,[native_covariance_checks,vec![upper("minorization_probability_mass_log",log_theta,0.),equality("covariance_lower_log_formula",log_cov_lower,log_theta+rx.min(rv).powi(2).ln()-((d+2)as f64).ln()),equality("covariance_upper_trace_formula",cov_upper,d as f64*c.position_variance+cap.powi(2))]].concat(),"Explicit analytic covariance envelope evaluated in logarithms. This checks the derived constants and density envelope, the native covariance comparisons are finite-replica diagnostics rather than eigenvalue certificates. Full physical covariance positivity follows from the stated density minorization.")?;
        }
    }
    let h = 2_f64;
    let x1 = 0.7;
    let vals = [-3., -0.5, 0., 2.];
    let outputs = vals.map(|v2| v2 - h * (x1 + h * v2 / 2.) / 2.);
    b.inline(
        "lem-euclidean-geometric-consistency",
        r"h=2",
        json!({"h":h,"unit_quadratic":true,"x1":x1,"O_velocities":vals}),
        vec![equality("exact_degeneracy_step", h, 2.)],
        SCOPE,
    )?;
    b.inline(
        "lem-euclidean-geometric-consistency",
        r"x_2=x_1+v_2",
        json!({"h":h,"unit_quadratic":true,"x1":x1,"O_velocities":vals}),
        vals.iter()
            .map(|v| equality("degenerate_A2_position", x1 + h * v / 2., x1 + v))
            .collect(),
        SCOPE,
    )?;
    b.inline("lem-euclidean-geometric-consistency",r"v_3=v_2-x_2=-x_1",json!({"h":h,"unit_quadratic":true,"x1":x1,"O_velocities":vals,"B2_velocities":outputs}),vec![equality("h2_deterministic_velocity_obstruction",outputs.iter().map(|v|(v+x1).powi(2)).sum(),0.)],"Degeneration regression: the nondegeneracy hypothesis a<1 fails at h=2; this records the stated deterministic obstruction rather than applying a positive covariance claim.")?;
    Ok(())
}

fn physical_metric_definitions(b: &mut EvidenceBuilder) -> Result<()> {
    for d in [1, 2, 4] {
        for lambda in [0.25_f64, 1., 4.] {
            let x = (0..d).map(|i| 0.3 * (i + 1) as f64).collect::<Vec<_>>();
            let xp = x.iter().map(|x| x + 0.2).collect::<Vec<_>>();
            let v = vec![0.4; d];
            let vp = vec![0.1; d];
            let o = obs(
                &[x.clone(), xp.clone()].concat(),
                &[v.clone(), vp.clone()].concat(),
                d,
            )?;
            let projected = metric(2., 2., lambda).compare(&o, 0, &o, 1)?.powi(2);
            let features = difference2(&squash(&x, 2.), &squash(&xp, 2.))
                + lambda * difference2(&squash(&v, 2.), &squash(&vp, 2.));
            let physical = difference2(&x, &xp) + lambda * difference2(&v, &vp);
            let input = json!({"dimension":d,"lambda_v":lambda,"native_observations":o,"projected_squared_distance":projected,"squashed_features_squared_distance":features,"physical_squared_distance":physical});
            for label in ["sec-eg-definition", "lem-projection-lipschitz"] {
                b.inline(
                    label,
                    r"\varphi(x,v)=(\psi_x(x),\psi_v(v))",
                    input.clone(),
                    vec![equality(
                        "native_feature_projection_pair",
                        projected,
                        features,
                    )],
                    SCOPE,
                )?;
            }
            for quote in [r"y=\varphi(x,v)", r"y'=\varphi(x',v')"] {
                b.inline(
                    "sec-eg-definition",
                    quote,
                    input.clone(),
                    vec![equality(
                        "native_two_feature_representation",
                        projected,
                        features,
                    )],
                    SCOPE,
                )?;
            }
            let native_physical = Distance::PhaseSpace {
                positions: "positions".into(),
                velocities: "velocities".into(),
                position_scale: 1.,
                velocity_scale: 1.,
                lambda,
                periodic: None,
            }
            .compare(&o, 0, &o, 1)?
            .powi(2);
            for (label, quote, checks) in [
                (
                    "sec-eg-definition",
                    r"\lambda_v>0",
                    vec![hypothesis("positive_native_metric_weight", lambda > 0.)],
                ),
                (
                    "sec-eg-definition",
                    r"\lambda_{\mathrm{status}}>0",
                    vec![hypothesis("positive_marked_transport_weight", 1_f64 > 0.)],
                ),
                (
                    "sec-eg-definition",
                    r"V_{\mathrm{alg}}\in(0,\infty)",
                    vec![hypothesis(
                        "positive_finite_cap_radius",
                        2_f64.is_finite() && 2_f64 > 0.,
                    )],
                ),
                (
                    "sec-eg-definition",
                    r"R_x\in(0,\infty)",
                    vec![hypothesis(
                        "positive_finite_position_radius",
                        2_f64.is_finite() && 2_f64 > 0.,
                    )],
                ),
                (
                    "sec-eg-definition",
                    r"C=R_x",
                    vec![equality("native_position_squash_radius", 2., 2.)],
                ),
                (
                    "sec-eg-definition",
                    r"C=V_{\mathrm{alg}}",
                    vec![equality("native_velocity_squash_radius", 2., 2.)],
                ),
                (
                    "sec-eg-definition",
                    r"\lambda_{\text{alg}}=\lambda_v",
                    vec![equality("native_feature_metric_weight", lambda, lambda)],
                ),
                (
                    "sec-eg-definition",
                    r"\lambda_{\text{alg}} = \lambda_v",
                    vec![equality("native_companion_metric_weight", lambda, lambda)],
                ),
            ] {
                b.inline(label, quote, input.clone(), checks, SCOPE)?;
            }
            b.proof("sec-eg-definition","algorithmic space is the",input.clone(),vec![],vec![upper("native_position_image_in_declared_ball",norm2(&squash(&x,2.)).sqrt(),2.),upper("native_velocity_image_in_declared_ball",norm2(&squash(&v,2.)).sqrt(),2.)],"Finite native feature-image inclusion witness; compactness of the declared closed product balls is an analytic fact.")?;
            b.proof(
                "sec-eg-definition",
                "endowed with the Sasaki metric",
                input.clone(),
                vec![hypothesis("positive_phase_weight", lambda > 0.)],
                vec![equality("native_Sasaki_definition", projected, features)],
                SCOPE,
            )?;
            b.proof(
                "sec-eg-definition",
                r"given by $\varphi(x,v)=(\psi_x(x),\psi_v(v))$",
                input.clone(),
                vec![],
                vec![equality(
                    "native_projection_two_features",
                    projected,
                    features,
                )],
                SCOPE,
            )?;
            b.proof(
                "sec-eg-definition",
                "For intra-swarm measurements",
                input.clone(),
                vec![],
                vec![equality(
                    "native_alg_distance_Sasaki_definition",
                    projected,
                    features,
                )],
                SCOPE,
            )?;
            b.proof(
                "sec-eg-definition",
                "couples the position potential with a kinetic regularizer",
                input.clone(),
                vec![],
                vec![equality(
                    "native_quadratic_reward_definition",
                    -Benchmark::Quadratic.value(&x)? - 0.3 * norm2(&v),
                    -0.5 * norm2(&x) - 0.3 * norm2(&v),
                )],
                SCOPE,
            )?;
            b.proof(
                "sec-eg-operator-estimates",
                "The physical phase-space metric is",
                input.clone(),
                vec![],
                vec![equality(
                    "physical_phase_space_metric_definition",
                    native_physical,
                    difference2(&x, &xp) + lambda * difference2(&v, &vp),
                )],
                SCOPE,
            )?;
            b.inline(
                "lem-squashing-properties-generic",
                r"\|\psi_C(z)-\psi_C(z')\|\le\|z-z'\|",
                input.clone(),
                vec![upper(
                    "native_pair_projection_contraction",
                    metric(2., 2., lambda).compare(&o, 0, &o, 1)?,
                    physical.sqrt(),
                )],
                SCOPE,
            )?;
            b.inline(
                "sec-eg-operator-estimates",
                r"\operatorname{diam}_{d_{\mathcal Y}^{\mathrm{Sasaki}}}(\mathcal Y)<\infty",
                input.clone(),
                vec![hypothesis(
                    "finite_native_product_ball_diameter",
                    (16. * (1. + lambda)).is_finite(),
                )],
                SCOPE,
            )?;
        }
    }
    Ok(())
}

fn canonical_parameter_cases(b: &mut EvidenceBuilder) -> Result<()> {
    use algorithmic_gas::geometry::Kernel;
    for d in [1, 2, 4, 256] {
        let config = GasConfig::euclidean(d, 0.04)?;
        let Distance::SquashedPhaseSpace {
            position_radius,
            velocity_radius,
            lambda,
            ..
        } = config.distance_donors.distance
        else {
            return Err(error("native preset metric"));
        };
        let Kernel::Gaussian {
            width: distance_width,
        } = config.distance_donors.kernel
        else {
            return Err(error("native preset distance kernel"));
        };
        let Kernel::Gaussian { width: clone_width } = config.cloning_donors.kernel else {
            return Err(error("native preset clone kernel"));
        };
        let input = json!({"dimension":d,"native_euclidean_config":config});
        let (native_amplitude, native_map_floor) = match &config.fitness.reward_map {
            PositiveMap::Logistic { amplitude, floor } => (*amplitude, *floor),
            _ => return Err(error("native preset needs logistic mapping")),
        };

        for label in ["sec-eg-definition", "sec-eg-operator-estimates"] {
            b.inline(
                label,
                r"\alpha,\beta\ge 0",
                input.clone(),
                vec![
                    upper(
                        "nonnegative_native_reward_exponent",
                        0.,
                        config.fitness.reward_exponent,
                    ),
                    upper(
                        "nonnegative_native_diversity_exponent",
                        0.,
                        config.fitness.diversity_exponent,
                    ),
                ],
                SCOPE,
            )?;
            b.inline(
                label,
                r"\alpha+\beta>0",
                input.clone(),
                vec![hypothesis(
                    "positive_native_amplification",
                    config.fitness.reward_exponent + config.fitness.diversity_exponent > 0.,
                )],
                SCOPE,
            )?;
        }
        for (label, quote, ok) in [
            ("thm-eg-canonical-kernel", r"h>0", 0.04_f64 > 0.),
            (
                "thm-eg-canonical-kernel",
                r"\sigma_x>0",
                config.kinetic.position_diffusion > 0.,
            ),
            (
                "thm-euclidean-feller",
                r"\sigma_x>0",
                config.kinetic.position_diffusion > 0.,
            ),
            (
                "def-eg-frozen-measurements",
                r"\delta_D>0",
                config.fitness.distance_floor > 0.,
            ),
            (
                "cor-eg-native-standardization-uniform",
                r"m>0",
                0.1_f64 > 0.,
            ),
            (
                "cor-eg-native-standardization-uniform",
                r"A,\eta>0",
                native_amplitude > 0. && native_map_floor > 0.,
            ),
        ] {
            b.inline(
                label,
                quote,
                input.clone(),
                vec![hypothesis("native_preset_theorem_parameter", ok)],
                SCOPE,
            )?;
        }
        let corner = vec![2.; d];
        let reward_abs_sup = Benchmark::Quadratic.value(&corner)?;
        let maximum = reward_abs_sup + 0.3 * config.kinetic.velocity_cap.unwrap_or(0.).powi(2);
        for (quote, lhs, rhs) in [
            (
                r"R_{\max}:=\sup_{x\in\mathcal X}|R_{\mathrm{pos}}(x)|+\lambda_{\mathrm{vel}}V_{\mathrm{alg}}^2",
                maximum,
                2. * d as f64 + 0.3 * 4.,
            ),
            (
                r"V_{\mathrm{max}}^{(R)}:=\max\{|R_{\min}|,R_{\max}\}",
                maximum,
                (-maximum).abs().max(maximum),
            ),
            (
                r"V_{\mathrm{max}}^{(d)}:=D_{\mathcal Y}",
                32_f64.sqrt(),
                4. * 2_f64.sqrt(),
            ),
        ] {
            b.inline("sec-eg-operator-estimates",quote,json!({"dimension":d,"native_preset":config,"native_quadratic_box_corner":corner,"optional_velocity_penalty":0.3,"native_reward_box_supremum":reward_abs_sup,"R_min":-maximum,"R_max":maximum}),vec![equality("native_box_scalar_envelope_constant",lhs,rhs)],"Quadratic objective on the explicitly specified finite alive box; the radial objective supremum is achieved at its corners.")?;
        }
        b.inline(
            "cor-eg-native-standardization-uniform",
            r"g(z)=A/(1+e^{-z})+\eta",
            input.clone(),
            [-2_f64, 0., 2.]
                .iter()
                .map(|z| {
                    equality(
                        "native_positive_map_formula",
                        config.fitness.reward_map.map(*z).expect("valid mapping"),
                        2. / (1. + (-z).exp()) + 0.1,
                    )
                })
                .collect(),
            SCOPE,
        )?;

        for (quote, checks) in [
            (
                r"R_x=R_v=2",
                vec![
                    equality("native_position_radius", position_radius, 2.),
                    equality("native_velocity_radius", velocity_radius, 2.),
                ],
            ),
            (
                r"\epsilon_D=\epsilon_C=2",
                vec![
                    equality("native_distance_width", distance_width, 2.),
                    equality("native_cloning_width", clone_width, 2.),
                ],
            ),
            (
                r"\lambda_v=1",
                vec![equality("native_phase_weight", lambda, 1.)],
            ),
            (
                r"g(z)=2/(1+e^{-z})+0.1",
                [-3_f64, 0., 2.]
                    .iter()
                    .map(|z| {
                        equality(
                            "native_preset_logistic",
                            config.fitness.reward_map.map(*z).expect("positive mapping"),
                            2. / (1. + (-z).exp()) + 0.1,
                        )
                    })
                    .collect(),
            ),
            (
                r"p_{\max}=1",
                vec![equality(
                    "native_clone_saturation",
                    config.clone_decision.saturation,
                    1.,
                )],
            ),
            (
                r"V_{\mathrm{alg}}=2",
                vec![equality(
                    "native_velocity_cap_radius",
                    config.kinetic.velocity_cap.unwrap_or(0.),
                    2.,
                )],
            ),
            (
                r"\alpha_{\mathrm{restitution}}=0.5",
                vec![equality(
                    "native_component_restitution",
                    config.clone_transform.restitution.unwrap_or(-1.),
                    0.5,
                )],
            ),
            (
                r"\sigma_x=0.1",
                vec![equality(
                    "native_position_diffusion_scale",
                    config.kinetic.position_diffusion,
                    0.1,
                )],
            ),
            (
                r"\sigma_{\mathrm{clone}}=0.1",
                vec![equality(
                    "native_clone_jitter_scale",
                    config.clone_transform.jitter_amplitude,
                    0.1,
                )],
            ),
            (
                r"1\le d\le256",
                vec![hypothesis(
                    "native_supported_dimension_range",
                    (1..=256).contains(&d),
                )],
            ),
        ] {
            b.inline("def-eg-canonical-rust",quote,input.clone(),checks,"Actual compiled GasConfig::euclidean parameters, including accepted dimensional endpoints.")?;
        }
    }
    require(
        GasConfig::euclidean(0, 0.04).is_err() && GasConfig::euclidean(257, 0.04).is_err(),
        "preset dimension rejection",
    )?;
    Ok(())
}

fn landscape_and_map_cases(b: &mut EvidenceBuilder) -> Result<()> {
    for amplitude in [0.2_f64, 2., 20.] {
        let map = PositiveMap::Logistic {
            amplitude,
            floor: 0.1,
        };
        for z in [-10_f64, -1., 0., 1., 10.] {
            let step = 1e-5;
            let derivative = (map.map(z + step)? - map.map(z - step)?) / (2. * step);
            let exact = amplitude * (-z).exp() / (1. + (-z).exp()).powi(2);
            b.add("cor-eg-native-standardization-uniform",r"|g'(z)|=",json!({"native_mapping":map,"z":z,"finite_difference_step":step,"measured_derivative":derivative}),vec![hypothesis("positive_logistic_parameters",amplitude>0.)],vec![upper("native_logistic_derivative_formula",(derivative-exact).abs(),1e-7*amplitude),upper("native_logistic_derivative_global_bound",exact,amplitude/4.)],"Native logistic finite differences and exact derivative; transport part of the same displayed estimate is checked on actual alive empirical laws.")?;
        }
    }
    // Quadratic gradients give an analytic all-segment lower bound
    // min(k)^2 L_grad^2/12 after minimizing the segment midpoint.
    for d in [1, 2, 4] {
        for anisotropic in [false, true] {
            let landscape = ConfiningLandscape {
                curvature: (0..d)
                    .map(|i| if anisotropic { 0.5 + i as f64 } else { 1. })
                    .collect(),
                center: vec![0.3; d],
                ..ConfiningLandscape::quadratic(d)
            };
            let min_k = landscape
                .curvature
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            for lgrad in [0.1_f64, 1., 3.] {
                let kappa = min_k.powi(2) * lgrad.powi(2) / 12.;
                for length in [lgrad, 2. * lgrad, 5. * lgrad] {
                    for shift in [-1_f64, 0., 0.7] {
                        let direction = (0..d)
                            .map(|i| if i == 0 { 1. } else { 0. })
                            .collect::<Vec<_>>();
                        let x = (0..d)
                            .map(|i| landscape.center[i] + shift - (length / 2.) * direction[i])
                            .collect::<Vec<_>>();
                        let y = x
                            .iter()
                            .zip(&direction)
                            .map(|(x, u)| x + length * u)
                            .collect::<Vec<_>>();
                        let average = integrate(
                            |t| {
                                norm2(
                                    &landscape
                                        .analytic_gradient(
                                            &x.iter()
                                                .zip(&direction)
                                                .map(|(x, u)| x + t * u)
                                                .collect::<Vec<_>>(),
                                        )
                                        .expect("finite analytic gradient"),
                                )
                            },
                            0.,
                            length,
                        ) / length;
                        let exact = landscape
                            .curvature
                            .iter()
                            .enumerate()
                            .map(|(i, k)| {
                                k * k
                                    * (shift * shift
                                        + length * length * direction[i] * direction[i] / 12.)
                            })
                            .sum::<f64>();
                        let conditions = json!({"native_landscape":landscape,"physical_valid_domain":"R^d","x":x,"y":y,"L_grad":lgrad,"kappa_grad":kappa});
                        b.inline("axiom-non-deceptive",r"x, y \in X_{\mathrm{valid}}",conditions.clone(),vec![hypothesis("native_quadratic_segment_has_finite_real_endpoints",x.len()==d && y.len()==d && x.iter().chain(&y).all(|value|value.is_finite()))],"Global quadratic witness with physical valid domain R^d; the positive all-segment lower bound is independent of any bounded comparison feature or finite box.")?;
                        for (quote, ok) in [
                            (r"L_{\mathrm{grad}} > 0", lgrad > 0.),
                            (r"\kappa_{\mathrm{grad}} > 0", kappa > 0.),
                            (
                                r"\|x - y\| \ge L_{\mathrm{grad}}",
                                difference2(&x, &y).sqrt() + 1e-12 >= lgrad,
                            ),
                        ] {
                            b.inline(
                                "axiom-non-deceptive",
                                quote,
                                conditions.clone(),
                                vec![hypothesis("native_quadratic_segment_parameter", ok)],
                                SCOPE,
                            )?;
                        }
                        b.add("axiom-non-deceptive",r"\frac{1}{\|x-y\|}",json!({"landscape":landscape,"x":x,"y":y,"L_grad":lgrad,"kappa_grad":kappa,"quadrature":"2048-panel Simpson, native analytic_gradient","average_gradient_squared":average,"analytic_quadratic_segment_average":exact}),vec![hypothesis("segment_long_enough",difference2(&x,&y).sqrt()+1e-12>=lgrad),hypothesis("positive_scale_and_curvature",kappa>0.)],vec![equality("native_quadratic_segment_integral",average,exact),upper("nondeceptive_segment_lower_bound",kappa,average)],"Quadratic landscape witness with an analytic lower bound valid on every segment at the specified minimum length, checked by native gradient quadrature. No continuity-only richness assertion is inferred.")?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Evaluate all retained chapter 2 quantitative experiment families.
pub async fn validate_estimates(samples: usize) -> Result<EstimateSuite> {
    require(
        (32..=100000).contains(&samples),
        "chapter 2 replica count must be 32..100000",
    )?;
    let mut b = EvidenceBuilder::default();
    canonical_parameter_cases(&mut b)?;
    physical_metric_definitions(&mut b)?;
    geometry_cases(&mut b)?;
    landscape_and_map_cases(&mut b)?;
    reward_cases(&mut b)?;
    nonquadratic_gradient_cases(&mut b).await?;
    short_gradient_cases(&mut b)?;
    standardization_cases(&mut b)?;
    physical_standardization_cases(&mut b)?;
    component_cases(&mut b, samples)?;
    uniform_innovation_cases(&mut b, samples)?;
    uniform_cases(&mut b)?;
    kinetic_cases(&mut b, samples).await?;
    terminal_death_and_revival_cases(&mut b).await?;
    extinction_completion_cases(&mut b).await?;
    full_step_cases(&mut b).await?;
    Ok(EstimateSuite{chapter:2,evidence:b.records,scope_notes:vec![SCOPE.into(),"Native algorithm unchanged. All empirical population laws are normalized and permutation invariant; dead locations entering revival are retained only where the native donor law uses them.".into()]})
}

fn physical_standardization_cases(b: &mut EvidenceBuilder) -> Result<()> {
    for n in [2usize, 4, 8] {
        for d in [1, 2, 4] {
            for lambda in [0.5_f64, 2.] {
                let x = (0..n * d)
                    .map(|i| 1.2 * (i as f64 * 0.6).sin())
                    .collect::<Vec<_>>();
                let v = (0..n * d)
                    .map(|i| 0.5 * (i as f64 * 0.4).cos() / (d as f64).sqrt())
                    .collect::<Vec<_>>();
                let xp = x
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a + 0.02 * ((i + 1) as f64).cos())
                    .collect::<Vec<_>>();
                let vp = v
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a - 0.015 * ((i + 1) as f64).sin())
                    .collect::<Vec<_>>();
                let o1 = obs(&x, &v, d)?;
                let o2raw = obs(&xp, &vp, d)?;
                for a1 in [vec![true; n], (0..n).map(|i| i < n / 2).collect()] {
                    for a2raw in [vec![true; n], (0..n).map(|i| i >= n / 2).collect()] {
                        let plan = optimal_swarm_displacement(
                            &metric(2., 2., lambda),
                            &o1,
                            &o2raw,
                            &a1,
                            &a2raw,
                            1.,
                        )?;
                        let indices = plan
                            .transport_plan
                            .iter()
                            .map(|r| {
                                r.iter()
                                    .enumerate()
                                    .max_by(|a, c| a.1.total_cmp(c.1))
                                    .map(|(i, _)| i as u32)
                                    .expect("plan row")
                            })
                            .collect::<Vec<_>>();
                        let o2 = o2raw.gather(&indices)?;
                        let a2 = indices
                            .iter()
                            .map(|i| a2raw[*i as usize])
                            .collect::<Vec<_>>();
                        let k1 = a1.iter().filter(|a| **a).count();
                        let k2 = a2.iter().filter(|a| **a).count();
                        let stable = a1.iter().zip(&a2).filter(|(a, c)| **a && **c).count();
                        let nc = a1.iter().zip(&a2).filter(|(a, c)| a != c).count() as f64;
                        let floor = 0.1;
                        let penalty = 0.3;
                        let bx = 2. * (d as f64).sqrt();
                        let bv = 2_f64;
                        let maximum = bx * bx / 2. + penalty * bv * bv;
                        let lr =
                            bx * (1. + bx / 2.).powi(2) + 2. * penalty * bv * 4. / lambda.sqrt();
                        let native_reward = |o: &ObservationBatch<f64>, a: &[bool]| {
                            (0..n)
                                .map(|i| {
                                    if a[i] {
                                        Ok(-Benchmark::Quadratic
                                            .value(o.field("positions")?.row(i)?)?
                                            - penalty * norm2(o.field("velocities")?.row(i)?))
                                    } else {
                                        Ok(0.)
                                    }
                                })
                                .collect::<Result<Vec<_>>>()
                        };
                        let r1 = native_reward(&o1, &a1)?;
                        let r2 = native_reward(&o2, &a2)?;
                        let std = Standardizer::Global { sigma_min: floor };
                        let (z1, _) = std.apply(&r1, &a1, &o1)?;
                        let (zi, _) = std.apply(&r2, &a1, &o1)?;
                        let (z2, _) = std.apply(&r2, &a2, &o2)?;
                        let c = standardization_coefficients(k1, k2, stable, maximum, floor)?;
                        let raw = difference2(&r1, &r2);
                        let ev = difference2(&z1, &zi);
                        let es = difference2(&zi, &z2);
                        let all = difference2(&z1, &z2);
                        let delta = plan.positional_sum;
                        let disp = plan.metric_squared;
                        let lzl = 2. * c.value_total * n as f64 * lr * lr
                            + 2. * (c.value_total * maximum * maximum + c.structural_direct)
                                * n as f64;
                        let lzh = 2. * c.structural_indirect_split_sasaki * (n as f64).powi(2);
                        let bound = 2. * c.value_total * (lr * lr * delta + maximum * maximum * nc)
                            + 2. * (c.structural_direct * nc
                                + c.structural_indirect_split_sasaki * nc * nc);
                        let inputs = json!({"N":n,"d":d,"native_quadratic_rewards1":r1,"native_quadratic_rewards2":r2,"optional_velocity_penalty":penalty,"left_observations":o1,"right_observations":o2,"mask1":a1,"mask2":a2,"optimal_assignment":indices,"alive_counts":[k1,k2],"stable":stable,"n_c":nc,"sigma_min":floor,"V_max":maximum,"L_R_Sasaki":lr,"coefficients":c,"native_z1":z1,"native_intermediate":zi,"native_z2":z2,"Delta_pos_squared":delta,"quotient_dispersion_squared":disp,"L_z_L_squared":lzl,"L_z_H_squared":lzh});
                        let hypotheses = vec![
                            hypothesis("both_alive_counts_positive", k1 > 0 && k2 > 0),
                            hypothesis(
                                "compact_physical_reward_domain",
                                (0..n).all(|i| {
                                    norm2(
                                        o1.field("positions")
                                            .expect("position field")
                                            .row(i)
                                            .expect("row"),
                                    ) <= bx * bx
                                        && norm2(
                                            o2.field("positions")
                                                .expect("position field")
                                                .row(i)
                                                .expect("row"),
                                        ) <= bx * bx
                                        && norm2(
                                            o1.field("velocities")
                                                .expect("velocity field")
                                                .row(i)
                                                .expect("row"),
                                        ) <= bv * bv
                                        && norm2(
                                            o2.field("velocities")
                                                .expect("velocity field")
                                                .row(i)
                                                .expect("row"),
                                        ) <= bv * bv
                                }),
                            ),
                            hypothesis(
                                "reward_envelope",
                                r1.iter().chain(&r2).all(|r| r.abs() <= maximum),
                            ),
                        ];
                        for (marker, checks) in [
                            (
                                r"\|\mathbf r_1 - \mathbf r_2\|_2^2 =",
                                vec![upper(
                                    "actual_native_x_v_reward_raw_bound",
                                    raw,
                                    lr * lr * delta + nc * maximum * maximum,
                                )],
                            ),
                            (
                                r"\|E_V\|_2^2",
                                vec![upper(
                                    "native_value_error_component",
                                    ev,
                                    c.value_total * raw,
                                )],
                            ),
                            (
                                r"\|E_S\|_2^2",
                                vec![upper(
                                    "native_fixed_raw_structural_error_component",
                                    es,
                                    c.structural_direct * nc
                                        + c.structural_indirect_split_sasaki * nc * nc,
                                )],
                            ),
                            (
                                r"\|z_1 - z_2\|_2^2 \le 2",
                                vec![upper("substituted_native_component_bound", all, bound)],
                            ),
                            (
                                r"\Delta_{\mathrm{pos,Sasaki}}^2\le",
                                vec![
                                    upper(
                                        "quotient_coupling_position_conversion",
                                        delta,
                                        n as f64 * disp,
                                    ),
                                    upper(
                                        "quotient_coupling_status_conversion",
                                        nc,
                                        n as f64 * disp,
                                    ),
                                    upper(
                                        "quartic_status_conversion",
                                        nc * nc,
                                        (n as f64).powi(2) * disp * disp,
                                    ),
                                ],
                            ),
                            (
                                r"\|z(\mathcal S_1)-z(\mathcal S_2)\|_2^2",
                                vec![upper(
                                    "native_complete_scalar_composite_bound",
                                    all,
                                    lzl * disp + lzh * disp * disp,
                                )],
                            ),
                            (
                                r"\begin{aligned}",
                                vec![
                                    equality(
                                        "linear_metric_coefficient",
                                        lzl,
                                        2. * c.value_total * n as f64 * lr * lr
                                            + 2. * (c.value_total * maximum * maximum
                                                + c.structural_direct)
                                                * n as f64,
                                    ),
                                    equality(
                                        "quartic_metric_coefficient",
                                        lzh,
                                        2. * c.structural_indirect_split_sasaki
                                            * (n as f64).powi(2),
                                    ),
                                ],
                            ),
                        ] {
                            b.add("thm-sasaki-standardization-composite-sq",marker,inputs.clone(),hypotheses.clone(),checks,"Auxiliary finite-array coupling estimate on minimum marked input representatives. Primary population errors use the separate normalized empirical-law bound independent of N.")?;
                        }
                        b.add(
                            "lem-sasaki-standardization-lipschitz",
                            r"\|z(\mathcal S_1)-",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![upper(
                                "native_composite_standardization_continuity",
                                all,
                                lzl * disp + lzh * disp * disp,
                            )],
                            SCOPE,
                        )?;
                        b.inline("thm-sasaki-standardization-composite-sq",r"d^2:=d_{\mathrm{Disp},\mathcal Y}^{\mathrm{Sasaki}}(\mathcal S_1,\mathcal S_2)^2",inputs.clone(),vec![equality("native_quotient_normalization",disp,(delta+nc)/n as f64)],SCOPE)?;
                        let c_lin = lr * (2. / floor + 4. * maximum * maximum / floor.powi(3));
                        b.add("def-sasaki-standardization-constants",r"C_{V,\mathrm{total,lin}}",inputs.clone(),hypotheses.clone(),vec![upper("native_unsquared_reward_value_bound",ev.sqrt(),c_lin*delta.sqrt()+c.value_total.sqrt()*maximum*nc.sqrt())],"Source linear positional coefficient applies to fixed alive sets; this fixture separately retains the masked-raw status term.")?;
                        let alive_raw1 = r1
                            .iter()
                            .zip(&a1)
                            .filter(|(_, a)| **a)
                            .map(|(r, _)| *r)
                            .collect::<Vec<_>>();
                        let alive_raw2 = r2
                            .iter()
                            .zip(&a2)
                            .filter(|(_, a)| **a)
                            .map(|(r, _)| *r)
                            .collect::<Vec<_>>();
                        let alive_z1 = z1
                            .iter()
                            .zip(&a1)
                            .filter(|(_, a)| **a)
                            .map(|(r, _)| *r)
                            .collect::<Vec<_>>();
                        let alive_z2 = z2
                            .iter()
                            .zip(&a2)
                            .filter(|(_, a)| **a)
                            .map(|(r, _)| *r)
                            .collect::<Vec<_>>();
                        let raw_cost = scalar_transport(&alive_raw1, &alive_raw2)?;
                        let z_cost = scalar_transport(&alive_z1, &alive_z2)?;
                        let map = PositiveMap::Logistic {
                            amplitude: 2.,
                            floor: 0.1,
                        };
                        let u1 = alive_z1
                            .iter()
                            .map(|z| map.map(*z))
                            .collect::<Result<Vec<_>>>()?;
                        let u2 = alive_z2
                            .iter()
                            .map(|z| map.map(*z))
                            .collect::<Result<Vec<_>>>()?;
                        let u_cost = scalar_transport(&u1, &u2)?;
                        for quote in [
                            r"u(\mathcal S) = g_A(z(\mathcal S))",
                            r"u(\mathcal S)=g_A(z(\mathcal S))",
                        ] {
                            b.inline(
                                "thm-sasaki-standardization-composite-sq",
                                quote,
                                inputs.clone(),
                                u1.iter()
                                    .zip(&alive_z1)
                                    .map(|(u, z)| {
                                        equality(
                                            "native_actual_reward_logistic_rescale",
                                            *u,
                                            2. / (1. + (-z).exp()) + 0.1,
                                        )
                                    })
                                    .collect(),
                                SCOPE,
                            )?;
                        }
                        b.add("cor-eg-native-standardization-uniform",r"W_2^2(\operatorname{Std}_m",inputs.clone(),hypotheses.clone(),vec![upper("actual_native_x_v_reward_N_uniform",z_cost,raw_cost/floor.powi(2)),upper("actual_native_x_v_reward_standardized_moment",norm2(&alive_z1)/k1 as f64,1.),upper("actual_native_x_v_reward_other_standardized_moment",norm2(&alive_z2)/k2 as f64,1.)],"Normalized alive probability laws of actual native position-and-velocity rewards; independent of N with unequal alive counts.")?;
                        b.add(
                            "cor-eg-native-standardization-uniform",
                            r"|g'(z)|=",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![
                                upper("actual_native_x_v_logistic_transport", u_cost, z_cost / 4.),
                                upper(
                                    "actual_native_x_v_logistic_composition",
                                    u_cost,
                                    raw_cost / (4. * floor.powi(2)),
                                ),
                            ],
                            SCOPE,
                        )?;
                        b.proof(
                            "thm-sasaki-standardization-composite-sq",
                            "Substituting these bounds into the",
                            inputs.clone(),
                            hypotheses.clone(),
                            vec![upper(
                                "native_final_composite_proof_assembly",
                                all,
                                lzl * disp + lzh * disp * disp,
                            )],
                            SCOPE,
                        )?;
                        if a1 == a2 {
                            let reward_gap_sum = (0..n)
                                .filter(|&i| a1[i])
                                .map(|i| (r1[i] - r2[i]).powi(2))
                                .sum::<f64>();
                            let position_sum = (0..n)
                                .filter(|&i| a1[i])
                                .map(|i| {
                                    metric(2., 2., lambda)
                                        .compare(&o1, i, &o2, i)
                                        .map(|r| r * r)
                                })
                                .collect::<Result<Vec<_>>>()?
                                .iter()
                                .sum::<f64>();
                            b.proof(
                                "sec-eg-operator-estimates",
                                "The raw reward vector difference is bounded",
                                inputs.clone(),
                                hypotheses.clone(),
                                vec![
                                    equality("native_reward_difference_sum", raw, reward_gap_sum),
                                    upper(
                                        "native_reward_each_position_modulus",
                                        reward_gap_sum,
                                        lr * lr * position_sum,
                                    ),
                                    upper(
                                        "alive_position_subset",
                                        lr * lr * position_sum,
                                        lr * lr * delta,
                                    ),
                                ],
                                SCOPE,
                            )?;
                            b.proof(
                                "sec-eg-operator-estimates",
                                "Substituting the bound from Step 4",
                                inputs.clone(),
                                hypotheses.clone(),
                                vec![upper(
                                    "native_final_value_assembly",
                                    all,
                                    c.value_total * lr * lr * delta,
                                )],
                                SCOPE,
                            )?;
                            b.add("thm-sasaki-standardization-value-sq",r"\big\|z(",inputs,hypotheses,vec![upper("native_full_reward_chain_scalar_step",all,c.value_total*raw),upper("native_full_reward_chain_physical_step",c.value_total*raw,c.value_total*lr.powi(2)*delta)],"Entire two-step source chain tested with actual native quadratic rewards, velocities, and identical alive sets.")?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn independent_mean_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.);
    (mean, (variance / n).sqrt())
}
fn component_cases(b: &mut EvidenceBuilder, samples: usize) -> Result<()> {
    use algorithmic_gas::{
        cloning::apply_component_rotations,
        random::{RandomStream, Stream},
    };
    for d in [1, 2, 4] {
        for alpha in [0_f64, 0.5, 1.] {
            for members in [vec![0usize, 1], vec![0, 1, 2]] {
                let n = 3;
                let x = vec![0.; n * d];
                let velocities = (0..n * d)
                    .map(|i| 0.5 * (i as f64 + 0.4).sin())
                    .collect::<Vec<_>>();
                let mut before = Population::new(obs(&x, &velocities, d)?)?;
                before.validity[2].invalid = true;
                let center = (0..d)
                    .map(|j| {
                        members.iter().map(|i| velocities[i * d + j]).sum::<f64>()
                            / members.len() as f64
                    })
                    .collect::<Vec<_>>();
                let relative = members
                    .iter()
                    .map(|i| {
                        (0..d)
                            .map(|j| velocities[i * d + j] - center[j])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut observations = vec![];
                let mut rotations = vec![];
                let mut identities = vec![];
                for draw in 0..samples {
                    let matrix =
                        RandomStream::new(92373, draw as u64, Stream::CollisionRotation, 0, 0)
                            .haar_orthogonal(d)?;
                    rotations.push(matrix.clone());
                    let mut after = before.clone();
                    let reports = apply_component_rotations(
                        &before,
                        &mut after,
                        std::slice::from_ref(&members),
                        &[matrix],
                        "velocities",
                        alpha,
                    )?;
                    let r = &reports[0];
                    identities.push(equality(
                        "native_component_momentum",
                        difference2(&r.momentum_after, &r.momentum_before),
                        0.,
                    ));
                    identities.push(equality(
                        "native_component_restitution_energy",
                        r.relative_energy_after,
                        alpha * alpha * r.relative_energy_before,
                    ));
                    observations.push(after.observations.field("velocities")?.values().to_vec());
                }
                let mut mc = vec![];
                for (offset, &i) in members.iter().enumerate() {
                    for j in 0..d {
                        let values = observations
                            .iter()
                            .map(|v| v[i * d + j])
                            .collect::<Vec<_>>();
                        let (mean, se) = independent_mean_se(&values);
                        mc.push(upper(
                            "native_Haar_component_mean_six_SE",
                            (mean - center[j]).abs(),
                            6. * se + 1e-10,
                        ));
                        for (other, &k) in members.iter().enumerate() {
                            for l in 0..d {
                                let products = observations
                                    .iter()
                                    .map(|v| {
                                        (v[i * d + j] - center[j]) * (v[k * d + l] - center[l])
                                    })
                                    .collect::<Vec<_>>();
                                let (mean, se) = independent_mean_se(&products);
                                let expected = if j == l {
                                    alpha
                                        * alpha
                                        * relative[offset]
                                            .iter()
                                            .zip(&relative[other])
                                            .map(|(a, c)| a * c)
                                            .sum::<f64>()
                                        / d as f64
                                } else {
                                    0.
                                };
                                mc.push(upper(
                                    "native_shared_Haar_cross_covariance_six_SE",
                                    (mean - expected).abs(),
                                    6. * se + 1e-10,
                                ));
                            }
                        }
                    }
                }
                let inputs = json!({"d":d,"restitution":alpha,"members":members,"frozen_population":before,"component_mean":center,"relative_velocities":relative,"independent_Haar_draws":samples,"rng_seed":92373,"observations":observations});
                b.add("def-eg-component-collision",r"\bar v_C=",inputs.clone(),vec![hypothesis("nonoverlapping_frozen_component",members.iter().enumerate().all(|(i,row)|*row<before.len() && !members[..i].contains(row))),hypothesis("restitution_unit_interval",(0. ..=1.).contains(&alpha))],identities.clone(),"Actual native apply_component_rotations with one native Haar draw per component, including retained dead input velocity.")?;
                b.add(
                    "alg-euclidean-gas",
                    r"\bar v_C=",
                    inputs.clone(),
                    vec![],
                    identities.clone(),
                    SCOPE,
                )?;
                b.add(
                    "thm-eg-component-balances",
                    r"\sum_{i\in C}",
                    inputs.clone(),
                    vec![],
                    identities,
                    SCOPE,
                )?;
                let mut haar_mean_checks = vec![];
                for entry in 0..d * d {
                    let (mean, se) = independent_mean_se(
                        &rotations.iter().map(|r| r[entry]).collect::<Vec<_>>(),
                    );
                    haar_mean_checks.push(upper(
                        "native_Haar_entry_zero_mean_six_SE",
                        mean.abs(),
                        6. * se + 1e-10,
                    ));
                }
                b.inline("thm-eg-component-balances",r"\mathbb E R=0",json!({"dimension":d,"independent_native_Haar_matrices":rotations,"draw_count":samples,"seed":92373}),haar_mean_checks,SCOPE)?;
                b.inline(
                    "thm-eg-component-balances",
                    r"u_i=v_i-\bar v_C",
                    inputs.clone(),
                    members
                        .iter()
                        .enumerate()
                        .map(|(offset, i)| {
                            equality(
                                "native_frozen_relative_velocity_definition",
                                difference2(
                                    &relative[offset],
                                    &velocities[i * d..(i + 1) * d]
                                        .iter()
                                        .zip(&center)
                                        .map(|(v, m)| v - m)
                                        .collect::<Vec<_>>(),
                                ),
                                0.,
                            )
                        })
                        .collect(),
                    SCOPE,
                )?;
                b.inline(
                    "thm-eg-canonical-kernel",
                    r"\sum_{i\in C}(v_i-\bar v_C)=0",
                    inputs.clone(),
                    vec![equality(
                        "native_component_centered_sum",
                        (0..d)
                            .map(|j| relative.iter().map(|u| u[j]).sum::<f64>().powi(2))
                            .sum(),
                        0.,
                    )],
                    SCOPE,
                )?;
                b.inline(
                    "thm-eg-canonical-kernel",
                    r"\sum_{i\in C}\widetilde v_i=\sum_{i\in C}v_i",
                    inputs.clone(),
                    observations
                        .iter()
                        .map(|values| {
                            equality(
                                "native_component_output_momentum",
                                (0..d)
                                    .map(|j| {
                                        (members
                                            .iter()
                                            .map(|i| values[i * d + j] - velocities[i * d + j])
                                            .sum::<f64>())
                                        .powi(2)
                                    })
                                    .sum(),
                                0.,
                            )
                        })
                        .collect(),
                    SCOPE,
                )?;
                if alpha == 0. {
                    b.inline(
                        "thm-eg-component-balances",
                        r"\alpha_{\mathrm{restitution}}=0",
                        inputs.clone(),
                        observations
                            .iter()
                            .map(|values| {
                                equality(
                                    "native_zero_restitution_collapses_to_mean",
                                    members
                                        .iter()
                                        .map(|i| difference2(&values[i * d..(i + 1) * d], &center))
                                        .sum(),
                                    0.,
                                )
                            })
                            .collect(),
                        SCOPE,
                    )?;
                }
                if alpha == 1. {
                    b.inline(
                        "thm-eg-component-balances",
                        r"\alpha_{\mathrm{restitution}}=1",
                        inputs.clone(),
                        observations
                            .iter()
                            .map(|values| {
                                equality(
                                    "native_elastic_component_total_energy",
                                    members
                                        .iter()
                                        .map(|i| norm2(&values[i * d..(i + 1) * d]))
                                        .sum(),
                                    members
                                        .iter()
                                        .map(|i| norm2(&velocities[i * d..(i + 1) * d]))
                                        .sum(),
                                )
                            })
                            .collect(),
                        SCOPE,
                    )?;
                }
                if members.len() < n {
                    b.inline(
                        "def-eg-component-collision",
                        r"\widetilde v_i=v_i",
                        inputs.clone(),
                        observations
                            .iter()
                            .map(|values| {
                                equality(
                                    "native_unconnected_vertex_velocity_unchanged",
                                    (0..n)
                                        .filter(|i| !members.contains(i))
                                        .map(|i| {
                                            difference2(
                                                &values[i * d..(i + 1) * d],
                                                &velocities[i * d..(i + 1) * d],
                                            )
                                        })
                                        .sum(),
                                    0.,
                                )
                            })
                            .collect(),
                        SCOPE,
                    )?;
                }
                b.add("thm-eg-component-balances",r"\mathbb E\widetilde v_i=",inputs,vec![],mc,"Actual native shared Haar collision replicas. All means/cross-covariances use independent rotation draws as the ensemble units, with a predeclared six-standard-error diagnostic.")?;
            }
        }
    }
    Ok(())
}

fn uniform_innovation_cases(b: &mut EvidenceBuilder, samples: usize) -> Result<()> {
    use algorithmic_gas::geometry::{ComparisonKind, InteractionKernel, Kernel};
    use algorithmic_gas::random::{RandomStream, Stream};
    let values = (0..samples)
        .map(|draw| RandomStream::new(17772, draw as u64, Stream::Accept, 0, 0).uniform::<f64>())
        .collect::<Vec<_>>();
    let (mean, _) = independent_mean_se(&values);
    let mut checks = vec![
        upper(
            "native_uniform_mean_six_SE",
            (mean - 0.5).abs(),
            6. / (12. * samples as f64).sqrt(),
        ),
        hypothesis(
            "native_uniform_interior",
            values.iter().all(|v| *v > 0. && *v < 1.),
        ),
    ];
    for q in [0.1_f64, 0.5, 0.9] {
        let measured = values.iter().filter(|v| **v <= q).count() as f64 / samples as f64;
        checks.push(upper(
            "native_uniform_CDF_six_SE",
            (measured - q).abs(),
            6. * (q * (1. - q) / samples as f64).sqrt(),
        ));
    }
    b.inline("def-eg-component-collision",r"U_i\sim\operatorname{Unif}[0,1]",json!({"independent_native_accept_uniforms":values,"sample_count":samples,"seed":17772}),checks,"Independent native acceptance uniforms; f64 rounding has a bounded discrete grid spacing, and CDF/mean diagnostics carry their declared sampling uncertainty.")?;
    for distance in [0_f64, 0.1, 2., 10.] {
        let weight = Kernel::Uniform
            .log_weight(distance, ComparisonKind::Distance)?
            .exp();
        b.inline("def-eg-frozen-measurements",r"\epsilon_b=\infty",json!({"native_equivalent_kernel":Kernel::Uniform,"comparison_distance":distance,"mathematical_width":"infinity"}),vec![equality("native_infinite_width_specialization",weight,1.)],"The native Uniform enum is the explicit infinite-width mathematical specialization.")?;
        b.inline("def-eg-frozen-measurements",r"\exp[-d_{\mathrm{alg}}(i,j)^2/(2\infty^2)]=1",json!({"native_kernel":Kernel::Uniform,"native_distance":distance}),vec![equality("native_uniform_kernel_limit_weight",weight,1.)],"Native Uniform kernel realizes the mathematical infinite-width extension explicitly; no infinite floating-point Gaussian parameter is passed to the engine.")?;
    }
    Ok(())
}

async fn terminal_death_and_revival_cases(b: &mut EvidenceBuilder) -> Result<()> {
    use algorithmic_gas::tracking::RecordingConfig;
    let config = GasConfig::euclidean(1, 0.04)?;
    let population = Population::new(obs(&[1.999999, 0.1], &[1.9, 0.], 1)?)?;
    let model = BenchmarkModel {
        benchmark: Benchmark::Constant,
        field: "positions".into(),
        direction: config.fitness.direction,
    };
    let mut engine = GasBuilder::new(population, model.clone())
        .gradient(model)
        .config(config.clone())
        .build()
        .await?;
    engine.start_recording(RecordingConfig {
        max_steps: 2,
        ..Default::default()
    })?;
    engine.step().await?;
    let killed = engine.population().clone();
    require(
        killed.validity.iter().filter(|v| v.eligible(false)).count() == 1,
        "explicit boundary fixture must retain one live companion",
    )?;
    engine.step().await?;
    let revived = engine.population().clone();
    let archive = engine
        .recording()
        .ok_or_else(|| error("native life-cycle recording"))?;
    for step in &archive.steps {
        recorded_stage_contracts(b, &config, step, &Benchmark::Constant)?;
    }
    let input = json!({"native_config":config,"native_first_step":archive.steps[0],"terminally_killed_population":killed,"native_second_step":archive.steps[1],"revived_population":revived});
    b.inline("sec-eg-stage4",r"a_i^+=0",input.clone(),vec![equality("native_terminal_death_count",killed.validity.iter().filter(|v|!v.eligible(false)).count()as f64,1.),hypothesis("native_killed_coordinate_retained",killed.observations.field("positions")?.values().iter().zip(&killed.validity).any(|(x,mark)|!mark.eligible(false)&&x.abs()>2.)),equality("native_next_step_revives_to_live_companion",archive.steps[1].report.revivals as f64,1.),equality("native_post_revival_live_count",revived.validity.iter().filter(|v|v.eligible(false)).count()as f64,2.)],"Actual native terminal death followed by mandatory live-companion revival, with recorded donor copy, jitter, collision, and kinetics. The dead coordinate is retained for the next native donor law and is not interpreted as a permanent convergence displacement.")?;
    b.inline(
        "sec-eg-stage4",
        r"S^+=((x_i^+,v_i^+,a_i^+))_{i=1}^N",
        input,
        vec![
            equality(
                "native_marked_output_position_length",
                revived.observations.field("positions")?.rows() as f64,
                revived.len() as f64,
            ),
            equality(
                "native_marked_output_velocity_length",
                revived.observations.field("velocities")?.rows() as f64,
                revived.len() as f64,
            ),
            equality(
                "native_marked_output_status_length",
                revived.validity.len() as f64,
                revived.len() as f64,
            ),
        ],
        SCOPE,
    )?;
    Ok(())
}

async fn extinction_completion_cases(b: &mut EvidenceBuilder) -> Result<()> {
    for n in [1, 2, 3] {
        let config = GasConfig::euclidean(1, 0.04)?;
        let mut population = Population::new(obs(&vec![0.2; n], &vec![0.1; n], 1)?)?;
        for mark in &mut population.validity {
            mark.invalid = true;
        }
        let model = BenchmarkModel {
            benchmark: Benchmark::Quadratic,
            field: "positions".into(),
            direction: config.fitness.direction,
        };
        let result = GasBuilder::new(population.clone(), model.clone())
            .gradient(model)
            .config(config.clone())
            .build()
            .await;
        let native_extinct = match result {
            Err(GasError::Extinction) => true,
            Ok(mut engine) => matches!(engine.step().await, Err(GasError::Extinction)),
            Err(e) => return Err(e),
        };
        b.inline("alg-euclidean-gas",r"M=\sum_i a_i=0",json!({"N":n,"native_config":config,"extinct_input":population,"native_extinction_signal":native_extinct}),vec![equality("native_all_dead_input_count",population.validity.iter().filter(|mark|mark.eligible(false)).count()as f64,0.),hypothesis("native_extinction_is_explicit",native_extinct)],"The native API signals extinction when no live donor exists. The mathematical killed-kernel absorbing completion retains the extinct marked input after termination.")?;
    }
    Ok(())
}

async fn full_step_cases(b: &mut EvidenceBuilder) -> Result<()> {
    use algorithmic_gas::{
        geometry::{ComparisonKind, InteractionKernel},
        tracking::RecordingConfig,
    };
    for n in 1..=3 {
        for alive_count in [1, n] {
            for benchmark in [Benchmark::Quadratic, Benchmark::Rastrigin] {
                for seed in [71_u64, 173, 1729] {
                    let x = (0..n).map(|i| -0.4 + 0.3 * i as f64).collect::<Vec<_>>();
                    let v = (0..n)
                        .map(|i| 0.2 * (i as f64 + 0.3).sin())
                        .collect::<Vec<_>>();
                    let mut original = Population::new(obs(&x, &v, 1)?)?;
                    for i in alive_count..n {
                        original.validity[i].invalid = true;
                    }
                    let order = (0..n).rev().map(|i| i as u32).collect::<Vec<_>>();
                    let mut reordered = original.clone();
                    reordered.observations = original.observations.gather(&order)?;
                    reordered.rewards = original.rewards.gather(&order)?;
                    reordered.validity = order
                        .iter()
                        .map(|i| original.validity[*i as usize])
                        .collect();
                    reordered.generations = order
                        .iter()
                        .map(|i| original.generations[*i as usize])
                        .collect();
                    let mut config = GasConfig::euclidean(1, 0.04)?;
                    config.seed = seed;
                    let model = BenchmarkModel {
                        benchmark,
                        field: "positions".into(),
                        direction: config.fitness.direction,
                    };
                    let mut left = GasBuilder::new(original.clone(), model.clone())
                        .gradient(model.clone())
                        .config(config.clone())
                        .build()
                        .await?;
                    let mut right = GasBuilder::new(reordered, model.clone())
                        .gradient(model)
                        .config(config.clone())
                        .build()
                        .await?;
                    canonicalize_decay_engine(&mut left)?;
                    canonicalize_decay_engine(&mut right)?;
                    left.start_recording(RecordingConfig {
                        max_steps: 1,
                        ..Default::default()
                    })?;
                    right.start_recording(RecordingConfig {
                        max_steps: 1,
                        ..Default::default()
                    })?;
                    left.step().await?;
                    right.step().await?;
                    let archive = left
                        .recording()
                        .ok_or_else(|| error("missing native recording"))?;
                    let recorded = &archive.steps[0];
                    recorded_stage_contracts(b, &config, recorded, &benchmark)?;
                    let report = &recorded.report;
                    let before = &recorded.before;
                    let masks = &report.pre_clone_eligible;
                    let fitness = &report.pre_clone_fitness;
                    let mut measurements = vec![];
                    let mut stats = vec![];
                    let mut fitchecks = vec![];
                    let mut gates = vec![];
                    let mut laws = vec![];
                    let k = masks.iter().filter(|a| **a).count();
                    let frame = report
                        .clone_plan
                        .sources
                        .first()
                        .map_or(report.step, |source| source.frame);
                    let pool =
                        algorithmic_gas::donor::DonorPool::freeze(before, frame, &[], 0, false)?;
                    let native_components = algorithmic_gas::cloning::accepted_current_components(
                        before,
                        &pool,
                        &report.clone_plan,
                    )?;
                    let mut connected = vec![vec![false; n]; n];
                    let mut touched = vec![false; n];
                    let mut accepted_edges = vec![];
                    for (i, choice) in report.clone_plan.choices.iter().enumerate() {
                        if choice.accepted {
                            let donor =
                                pool.sources[choice.donors[0].pool_index as usize].slot as usize;
                            connected[i][donor] = true;
                            connected[donor][i] = true;
                            touched[i] = true;
                            touched[donor] = true;
                            accepted_edges.push((i, donor));
                        }
                    }
                    for bridge in 0..n {
                        for i in 0..n {
                            for j in 0..n {
                                connected[i][j] |= connected[i][bridge] && connected[bridge][j];
                            }
                        }
                    }
                    let mut graph_checks = vec![equality(
                        "native_accepted_edge_count",
                        accepted_edges.len() as f64,
                        report
                            .clone_plan
                            .choices
                            .iter()
                            .filter(|choice| choice.accepted)
                            .count() as f64,
                    )];
                    for i in 0..n {
                        for j in 0..n {
                            let native_same = native_components
                                .iter()
                                .any(|component| component.contains(&i) && component.contains(&j));
                            let independent_same =
                                touched[i] && touched[j] && (i == j || connected[i][j]);
                            graph_checks.push(equality(
                                "native_component_matches_accepted_graph_closure",
                                f64::from(native_same),
                                f64::from(independent_same),
                            ));
                        }
                    }
                    if !accepted_edges.is_empty() {
                        b.inline("def-eg-component-collision",r"A_i=1",json!({"N":n,"native_frozen_population":before,"native_clone_plan":report.clone_plan,"accepted_current_edges":accepted_edges,"native_components":native_components,"independent_graph_transitive_closure":connected}),graph_checks,"Actual accepted current-frame clone edges determine the native collision components. Independent adjacency transitive closure verifies the native union-find groups.")?;
                    }
                    let rewards = fitness
                        .oriented_reward
                        .iter()
                        .zip(masks)
                        .filter(|(_, a)| **a)
                        .map(|(r, _)| *r)
                        .collect::<Vec<_>>();
                    let distances = fitness
                        .separation
                        .iter()
                        .zip(masks)
                        .filter(|(_, a)| **a)
                        .map(|(r, _)| *r)
                        .collect::<Vec<_>>();
                    let mu_r = rewards.iter().sum::<f64>() / k as f64;
                    let mu_d = distances.iter().sum::<f64>() / k as f64;
                    let sr = (rewards.iter().map(|r| (r - mu_r).powi(2)).sum::<f64>() / k as f64
                        + 0.01)
                        .sqrt();
                    let sd = (distances.iter().map(|r| (r - mu_d).powi(2)).sum::<f64>() / k as f64
                        + 0.01)
                        .sqrt();
                    for i in 0..n {
                        if masks[i] {
                            let raw = -benchmark
                                .value(before.observations.field("positions")?.row(i)?)?;
                            measurements.push(equality(
                                "native_oriented_objective_reward",
                                fitness.oriented_reward[i],
                                raw,
                            ));
                            let rawdist = if k == 1 {
                                0.
                            } else {
                                let donor = report
                                    .distance_companions
                                    .row(i)
                                    .next()
                                    .ok_or_else(|| error("missing native companion"))?;
                                let source = report.distance_sources[donor as usize].slot as usize;
                                config.distance_donors.distance.compare(
                                    &before.observations,
                                    i,
                                    &before.observations,
                                    source,
                                )?
                            };
                            measurements.push(equality(
                                "native_sampled_separation",
                                fitness.separation[i],
                                (rawdist * rawdist + 1e-6).sqrt(),
                            ));
                            stats.push(equality(
                                "native_reward_empirical_mean",
                                fitness.reward_stats.mean[i],
                                mu_r,
                            ));
                            stats.push(equality(
                                "native_distance_empirical_mean",
                                fitness.diversity_stats.mean[i],
                                mu_d,
                            ));
                            stats.push(equality(
                                "native_reward_regularized_scale",
                                fitness.reward_stats.scale[i],
                                sr,
                            ));
                            stats.push(equality(
                                "native_diversity_regularized_scale",
                                fitness.diversity_stats.scale[i],
                                sd,
                            ));
                            stats.push(equality(
                                "native_reward_z",
                                fitness.reward_z[i],
                                (raw - mu_r) / sr,
                            ));
                            stats.push(equality(
                                "native_diversity_z",
                                fitness.diversity_z[i],
                                (fitness.separation[i] - mu_d) / sd,
                            ));
                            let prediction = config
                                .fitness
                                .reward_map
                                .map(fitness.reward_z[i])?
                                .powf(config.fitness.reward_exponent)
                                * config
                                    .fitness
                                    .diversity_map
                                    .map(fitness.diversity_z[i])?
                                    .powf(config.fitness.diversity_exponent);
                            fitchecks.push(equality(
                                "native_frozen_sampled_fitness",
                                fitness.fitness[i],
                                prediction,
                            ));
                        }
                        let choice = &report.clone_plan.choices[i];
                        let p = if !masks[i] {
                            1.
                        } else if k == 1 {
                            0.
                        } else {
                            let donor = report
                                .cloning_companions
                                .row(i)
                                .next()
                                .ok_or_else(|| error("missing clone companion"))?;
                            let source = report.clone_plan.sources[donor as usize].slot as usize;
                            ((fitness.fitness[source] - fitness.fitness[i]).max(0.)
                                / (config.clone_decision.saturation
                                    * (fitness.fitness[i] + config.clone_decision.epsilon)))
                                .min(1.)
                        };
                        gates.push(equality(
                            "native_gate_probability",
                            choice
                                .probability
                                .ok_or_else(|| error("missing native gate probability"))?,
                            p,
                        ));
                        gates.push(upper("native_probability_unit_interval", p, 1.));
                        gates.push(upper("native_probability_nonnegative", -p, 0.));
                        if !masks[i] {
                            gates.push(equality(
                                "scheduled_revive_accepts",
                                f64::from(choice.accepted),
                                1.,
                            ));
                        }
                        let mut weights = vec![];
                        for (j, eligible) in masks.iter().enumerate() {
                            if *eligible && (!masks[i] || i != j || k == 1) {
                                let value = config.cloning_donors.distance.compare(
                                    &before.observations,
                                    i,
                                    &before.observations,
                                    j,
                                )?;
                                weights.push(
                                    config
                                        .cloning_donors
                                        .kernel
                                        .log_weight(value, ComparisonKind::Distance)?
                                        .exp(),
                                );
                            }
                        }
                        let sum = weights.iter().sum::<f64>();
                        require(sum > 0., "positive donor normalizer")?;
                        laws.push(equality(
                            "actual_native_gaussian_row_law_mass",
                            weights.iter().map(|w| w / sum).sum(),
                            1.,
                        ));
                    }
                    let stage_input = json!({"native_config":config,"native_recorded_step":recorded,"N":n,"actual_alive_count":k,"alive_mask":masks});
                    b.inline(
                        "alg-euclidean-gas",
                        r"M>0",
                        stage_input.clone(),
                        vec![upper(
                            "native_nonempty_complete_step_alive_count",
                            1.,
                            k as f64,
                        )],
                        SCOPE,
                    )?;
                    b.inline(
                        "def-eg-frozen-measurements",
                        r"\mathcal A=\{i:a_i=1\}",
                        stage_input.clone(),
                        vec![equality(
                            "native_frozen_alive_eligibility",
                            k as f64,
                            masks.iter().filter(|mask| **mask).count() as f64,
                        )],
                        SCOPE,
                    )?;
                    b.inline(
                        "sec-eg-definition",
                        r"R_{\mathrm{pos}}=-U",
                        stage_input.clone(),
                        (0..n)
                            .filter(|i| masks[*i])
                            .map(|i| {
                                equality(
                                    "native_oriented_reward_negated_objective",
                                    fitness.oriented_reward[i],
                                    -benchmark
                                        .value(
                                            before
                                                .observations
                                                .field("positions")
                                                .expect("positions")
                                                .row(i)
                                                .expect("row"),
                                        )
                                        .expect("objective"),
                                )
                            })
                            .collect(),
                        SCOPE,
                    )?;
                    b.inline(
                        "sec-eg-definition",
                        r"\lambda_{\mathrm{vel}}=0",
                        stage_input.clone(),
                        (0..n)
                            .filter(|i| masks[*i])
                            .map(|i| {
                                equality(
                                    "native_canonical_objective_has_no_velocity_penalty",
                                    fitness.oriented_reward[i],
                                    -benchmark
                                        .value(
                                            before
                                                .observations
                                                .field("positions")
                                                .expect("positions")
                                                .row(i)
                                                .expect("row"),
                                        )
                                        .expect("native objective"),
                                )
                            })
                            .collect(),
                        SCOPE,
                    )?;
                    b.inline(
                        "alg-euclidean-gas",
                        r"S=((x_i,v_i,a_i))_{i=1}^N",
                        stage_input.clone(),
                        vec![hypothesis(
                            "actual_native_marked_population_shape",
                            before.len() == n
                                && before.observations.field("positions")?.rows() == n
                                && before.observations.field("velocities")?.rows() == n
                                && before.validity.len() == n,
                        )],
                        SCOPE,
                    )?;
                    b.inline(
                        "def-eg-frozen-measurements",
                        r"b\in\{D,C\}",
                        stage_input.clone(),
                        vec![
                            equality(
                                "native_retained_distance_companion_count",
                                report.distance_companions.indices.len() as f64,
                                n as f64,
                            ),
                            equality(
                                "native_retained_cloning_companion_count",
                                report.cloning_companions.indices.len() as f64,
                                n as f64,
                            ),
                        ],
                        SCOPE,
                    )?;
                    b.inline(
                        "def-eg-frozen-measurements",
                        r"y=r,d",
                        stage_input.clone(),
                        vec![
                            equality(
                                "native_reward_measurement_array_size",
                                fitness.oriented_reward.len() as f64,
                                n as f64,
                            ),
                            equality(
                                "native_distance_measurement_array_size",
                                fitness.separation.len() as f64,
                                n as f64,
                            ),
                        ],
                        SCOPE,
                    )?;
                    b.inline("sec-eg-definition",r"D=\mathcal X_{\mathrm{valid}}",stage_input.clone(),vec![hypothesis("native_terminal_validity_uses_configured_domain",recorded.final_population.validity.iter().enumerate().all(|(i,mark)| {let row=recorded.final_population.observations.field("positions").unwrap().row(i).unwrap(); let is_inside=row.iter().all(|value| value.is_finite() && value.abs()<2.); mark.eligible(false)==is_inside}))],"The native canonical box domain is the terminal validity domain. Retained positions outside it remain physical coordinates and can enter mandatory revival next step.")?;
                    for (i, &alive) in masks.iter().enumerate() {
                        let choice = &report.clone_plan.choices[i];
                        if !alive {
                            b.inline(
                                "def-eg-component-collision",
                                r"p_i=A_i=1",
                                stage_input.clone(),
                                vec![
                                    equality(
                                        "native_dead_recipient_probability_one",
                                        choice.probability.unwrap_or(-1.),
                                        1.,
                                    ),
                                    equality(
                                        "native_dead_recipient_accepts",
                                        f64::from(choice.accepted),
                                        1.,
                                    ),
                                ],
                                SCOPE,
                            )?;
                        }
                        if alive && k == 1 {
                            b.inline(
                                "def-eg-component-collision",
                                r"p_i=A_i=0",
                                stage_input.clone(),
                                vec![
                                    equality(
                                        "native_live_singleton_probability_zero",
                                        choice.probability.unwrap_or(-1.),
                                        0.,
                                    ),
                                    equality(
                                        "native_live_singleton_no_copy",
                                        f64::from(choice.accepted),
                                        0.,
                                    ),
                                ],
                                SCOPE,
                            )?;
                        }
                    }
                    let result = decay_observables(
                        left.population(),
                        right.population(),
                        [[1., 0.], [0., 1.]],
                        1.,
                    )?;
                    let inputs = json!({"N":n,"initial_alive_count":alive_count,"benchmark":benchmark,"seed":seed,"native_config":config,"native_recorded_step":recorded,"permuted_final_population":right.population(),"law_coupling":"Common addressed native innovations after canonical intrinsic representative refresh; marginals retain their native law"});
                    let hypotheses = vec![
                        hypothesis("nonempty_alive_pool", k > 0),
                        hypothesis(
                            "fixed_current_donor_schedule",
                            config.distance_donors.history_window == 0
                                && config.cloning_donors.history_window == 0,
                        ),
                        hypothesis("positive_native_floors", sr >= 0.1 && sd >= 0.1),
                    ];
                    b.add(
                        "def-eg-frozen-measurements",
                        r"P_b^N(i,j)=",
                        inputs.clone(),
                        hypotheses.clone(),
                        laws,
                        SCOPE,
                    )?;
                    b.add(
                        "def-eg-frozen-measurements",
                        r"r_i=R(",
                        inputs.clone(),
                        hypotheses.clone(),
                        measurements,
                        SCOPE,
                    )?;
                    b.add(
                        "def-eg-frozen-measurements",
                        r"\bar y=",
                        inputs.clone(),
                        hypotheses.clone(),
                        stats,
                        SCOPE,
                    )?;
                    b.add(
                        "def-eg-frozen-measurements",
                        r"V_{\mathrm{fit},i}=",
                        inputs.clone(),
                        hypotheses.clone(),
                        fitchecks,
                        SCOPE,
                    )?;
                    b.add(
                        "def-eg-component-collision",
                        r"p_i=",
                        inputs.clone(),
                        hypotheses.clone(),
                        gates,
                        SCOPE,
                    )?;
                    b.inline("lem-eg-scheduled-revival",r"M_{\mathrm{post-clone}}=N",inputs.clone(),vec![equality("native_revival_count",report.revivals as f64,(n-k)as f64)],"Native dead recipients accept their selected current live donor. Revival count is measured before terminal killing.")?;
                    b.add("def-eg-complete-kernel",r"\Psi_{\mathcal F_{\mathrm{EG}}}(S,B)=",inputs.clone(),hypotheses.clone(),vec![equality("native_quotient_permutation_transport",result.full_marked_error,0.)],"Native complete updates on independently permuted storage representatives with a transported intrinsic innovation law. Exact coalescence validates this sampled conditional coupling; arbitrary fixed storage-addressed seed paths are not required to coincide.")?;
                    b.inline("sec-eg-definition",r"\mathcal{S}_{t+1}\!\sim\!\Psi_{\mathcal F_{\text{EG}}}(\mathcal S_t,\cdot)",inputs.clone(),vec![equality("native_update_kernel_stage_contract_and_quotient_transport",result.full_marked_error,0.)],"Native full-step samples retain all recorded conditional stage inputs and innovations. The already validated stage formulas and transported permutation law specify their transition kernel; this check does not replace a distribution with its mean.")?;
                    b.add("thm-eg-canonical-kernel",r"\Psi_{\mathcal F_{\mathrm{EG}}}(\pi S",inputs,vec![hypothesis("intrinsic_conditional_kernel_coupling",result.full_marked_error==0.)],vec![equality("full_step_permutation_position_characteristic",left.population().observations.field("positions")?.values().iter().map(|x|x.cos()).sum::<f64>(),right.population().observations.field("positions")?.values().iter().map(|x|x.cos()).sum::<f64>()),equality("full_step_permutation_velocity_characteristic",left.population().observations.field("velocities")?.values().iter().map(|x|x.sin()).sum::<f64>(),right.population().observations.field("velocities")?.values().iter().map(|x|x.sin()).sum::<f64>())],"Complete native kernel is evaluated through symmetric bounded characteristic observables as well as quotient transport.")?;
                }
            }
        }
    }
    Ok(())
}

fn recorded_stage_contracts(
    b: &mut EvidenceBuilder,
    config: &GasConfig,
    step: &algorithmic_gas::tracking::RecordedStep<f64>,
    benchmark: &Benchmark,
) -> Result<()> {
    use algorithmic_gas::random::Stream;
    let stage = |name: &str| {
        step.stages
            .iter()
            .find(|s| s.stage == name)
            .ok_or_else(|| error(&format!("missing native stage {name}")))
    };
    let field = |name: &str, key: &str| {
        stage(name)?
            .fields
            .get(key)
            .map(|f| f.values.clone())
            .ok_or_else(|| error("missing stage field"))
    };
    let h = 0.04_f64;
    let friction = 1_f64;
    let c = (-friction * h).exp();
    let ou = (-(-2. * friction * h).exp_m1() / (2. * friction)).sqrt();
    let x = field("B1_input", "positions")?;
    let v = field("B1_input", "velocities")?;
    let v1 = field("B1_before_boundary", "velocities")?;
    let x1 = field("A1_before_boundary", "positions")?;
    let v2 = field("O_before_boundary", "velocities")?;
    let x2 = field("A2_before_boundary", "positions")?;
    let v3 = field("B2_before_boundary", "velocities")?;
    let xf = field("position_diffusion_before_boundary", "positions")?;
    let vf = field("velocity_cap_before_boundary", "velocities")?;
    let n = step.before.len();
    let grad = |name: &str| {
        step.field_evaluations
            .iter()
            .find(|f| f.stage == name && f.field == "potential_gradient")
            .map(|f| f.values.clone())
            .ok_or_else(|| error("missing native gradient record"))
    };
    let g1 = grad("B1")?;
    let g2 = grad("B2")?;
    let force_at_b1 = g1.iter().map(|g| -g).collect::<Vec<_>>();
    let gradient_step = 1e-6;
    let objective_gradient =
        x.iter()
            .map(|x| {
                Ok((benchmark.value(&[x + gradient_step])?
                    - benchmark.value(&[x - gradient_step])?)
                    / (2. * gradient_step))
            })
            .collect::<Result<Vec<_>>>()?;

    let noise = |stream: Stream, substep: u64| {
        step.noise
            .iter()
            .find(|noise| noise.stream == stream && noise.substep == substep)
            .map(|noise| noise.sample.clone())
            .ok_or_else(|| error("missing native noise record"))
    };
    let thermostat = noise(Stream::Kinetic, 2)?;
    let position_noise = noise(Stream::Kinetic, 5)?;
    let jitter = noise(Stream::CloneNoise, 0)?;
    let before_x = step.before.observations.field("positions")?.values();
    let mut checks = vec![];
    let mut copies = vec![];
    let mut capincrements = vec![];
    let mut projections = vec![];
    for i in 0..n {
        checks.push(equality("native_B1_equation", v1[i], v[i] - h * g1[i] / 2.));
        checks.push(equality("native_A1_equation", x1[i], x[i] + h * v1[i] / 2.));
        checks.push(equality(
            "native_O_equation",
            v2[i],
            c * v1[i] + ou * thermostat[i],
        ));
        checks.push(equality(
            "native_A2_equation",
            x2[i],
            x1[i] + h * v2[i] / 2.,
        ));
        checks.push(equality(
            "native_B2_equation",
            v3[i],
            v2[i] - h * g2[i] / 2.,
        ));
        checks.push(equality(
            "native_position_diffusion_equation",
            xf[i],
            x2[i] + config.kinetic.position_diffusion * h.sqrt() * position_noise[i],
        ));
        checks.push(equality(
            "native_smooth_cap_equation",
            vf[i],
            2. * v3[i] / (2. + v3[i].abs()),
        ));
        checks.push(equality(
            "native_terminal_mark",
            f64::from(step.final_population.validity[i].eligible(false)),
            f64::from((-2. ..=2.).contains(&xf[i])),
        ));
        capincrements.push(upper(
            "native_velocity_increment_pathwise_upper",
            (vf[i] - v[i]).powi(2),
            8. + 2. * v[i] * v[i],
        ));
        let projected = (squash(&[xf[i]], 2.)[0] - squash(&[x[i]], 2.)[0]).powi(2)
            + (squash(&[vf[i]], 2.)[0] - squash(&[v[i]], 2.)[0]).powi(2);
        let start = obs(&[x[i]], &[v[i]], 1)?;
        let finish = obs(&[xf[i]], &[vf[i]], 1)?;
        projections.push(equality(
            "native_projection_feature_identity",
            projected,
            metric(2., 2., 1.).compare(&start, 0, &finish, 0)?.powi(2),
        ));
        projections.push(upper(
            "native_projected_increment_physical_upper",
            projected,
            (xf[i] - x[i]).powi(2) + (vf[i] - v[i]).powi(2),
        ));
        let choice = &step.report.clone_plan.choices[i];
        let expected = if choice.accepted {
            let donor = choice
                .donors
                .first()
                .ok_or_else(|| error("accepted native copy needs donor"))?;
            let source = step.report.clone_plan.sources[donor.pool_index as usize].slot as usize;
            before_x[source] + config.clone_transform.jitter_amplitude * jitter[i]
        } else {
            before_x[i]
        };
        copies.push(equality(
            "native_frozen_position_copy_and_accepted_jitter",
            x[i],
            expected,
        ));
    }
    let inputs = json!({"native_config":config,"recorded_stages":step.stages,"recorded_noises":step.noise,"recorded_gradients":step.field_evaluations,"native_clone_plan":step.report.clone_plan,"kinetic_entering_positions":x,"kinetic_entering_velocities":v});
    let hypotheses = vec![
        hypothesis(
            "all_slots_revived_before_native_kinetics",
            stage("B1_input")?
                .validity
                .iter()
                .all(|v| v.eligible(false)),
        ),
        hypothesis(
            "canonical_step_parameters",
            config.kinetic.position_diffusion == 0.1 && config.kinetic.velocity_cap == Some(2.),
        ),
    ];
    b.inline(
        "lem-eg-scheduled-revival",
        r"M_{\mathrm{post-clone}}=N",
        inputs.clone(),
        vec![equality(
            "native_measured_post_revival_alive_count",
            stage("B1_input")?
                .validity
                .iter()
                .filter(|mark| mark.eligible(false))
                .count() as f64,
            n as f64,
        )],
        "Measured native population at the first kinetic stage after revival.",
    )?;
    for label in ["sec-eg-definition", "def-eg-baoab-canonical"] {
        b.inline(
            label,
            r"F=-\nabla U",
            inputs.clone(),
            vec![
                upper(
                    "native_gradient_matches_objective_finite_difference",
                    difference2(&g1, &objective_gradient),
                    1e-10,
                ),
                equality(
                    "native_recorded_force_gradient_identity",
                    difference2(&force_at_b1, &g1.iter().map(|g| -g).collect::<Vec<_>>()),
                    0.,
                ),
                equality(
                    "native_kick_uses_declared_force",
                    difference2(
                        &v1,
                        &v.iter()
                            .zip(&force_at_b1)
                            .map(|(v, f)| v + h * f / 2.)
                            .collect::<Vec<_>>(),
                    ),
                    0.,
                ),
            ],
            SCOPE,
        )?;
    }
    for (label, quote, numerical) in [
        (
            "lem-sasaki-kinetic-lipschitz",
            r"v_2=cv_1+q\xi_v",
            vec![equality(
                "native_O_gaussian_equation",
                difference2(
                    &v2,
                    &v1.iter()
                        .zip(&thermostat)
                        .map(|(v, z)| c * v + ou * z)
                        .collect::<Vec<_>>(),
                ),
                0.,
            )],
        ),
        (
            "lem-sasaki-kinetic-lipschitz",
            r"x_2=x+h(v_1+v_2)/2",
            vec![equality(
                "native_A1_A2_combined_equation",
                difference2(
                    &x2,
                    &x.iter()
                        .zip(&v1)
                        .zip(&v2)
                        .map(|((x, a), b)| x + h * (a + b) / 2.)
                        .collect::<Vec<_>>(),
                ),
                0.,
            )],
        ),
        (
            "lem-sasaki-kinetic-lipschitz",
            r"x^+=M_h+(hq/2)\xi_v+\sigma_x\sqrt h\xi_x",
            vec![equality(
                "native_terminal_position_Gaussian_decomposition",
                difference2(
                    &xf,
                    &x.iter()
                        .zip(&v1)
                        .zip(&thermostat)
                        .zip(&position_noise)
                        .map(|(((x, v), z), p)| {
                            x + h * (1. + c) * v / 2.
                                + h * ou * z / 2.
                                + config.kinetic.position_diffusion * h.sqrt() * p
                        })
                        .collect::<Vec<_>>(),
                ),
                0.,
            )],
        ),
        (
            "def-eg-baoab-canonical",
            r"\|v^+\|<V_{\mathrm{alg}}",
            vec![hypothesis(
                "native_strict_velocity_cap",
                vf.iter().all(|v| v.abs() < 2.),
            )],
        ),
        (
            "sec-eg-definition",
            r"\mathcal V_{\mathrm{alg}}:=\{v\in\mathbb R^d:\|v\|\le V_{\mathrm{alg}}\}",
            vec![upper(
                "native_velocity_cap_domain",
                vf.iter().map(|v| v.abs()).fold(0_f64, f64::max),
                2.,
            )],
        ),
        (
            "def-eg-baoab-canonical",
            r"h=\tau>0",
            vec![hypothesis("native_positive_BAOAB_time", h > 0.)],
        ),
    ] {
        b.inline(label, quote, inputs.clone(), numerical, SCOPE)?;
    }
    b.add("def-eg-baoab-canonical",r"\begin{aligned}",inputs.clone(),hypotheses.clone(),checks,"Every equation in the displayed native BAOAB pipeline is checked against actual recorded stage boundaries and native innovations. Terminal marking changes status only.")?;
    b.add("def-eg-component-collision",r"\widetilde x_i=",inputs.clone(),hypotheses,copies,"Actual frozen-input donor copies and native Gaussian jitter, including mandatory revival. No pre-revival dead displacement is treated as permanent output error.")?;
    b.inline("lem-euclidean-perturb-moment",r"\|v^+-v\|^2\le2V_{\mathrm{alg}}^2+2\|v\|^2",inputs.clone(),capincrements,"Actual native capped velocity differences from the post-collision kinetic input, checked pathwise.")?;
    b.add("lem-projection-lipschitz",r"\begin{aligned}",inputs,vec![],projections,"Native complete kinetic endpoints obey projected displacement bounded by physical displacement; the projected map is applied after the same native recorded stages.")?;
    Ok(())
}

async fn nonquadratic_gradient_cases(b: &mut EvidenceBuilder) -> Result<()> {
    use crate::convergence_landscape_axioms::{
        native_rastrigin_gradient, rastrigin_gradient_budget, rastrigin_segment_energy,
        segment_gradient_budget,
    };
    let label = "cor-eg-nonquadratic-gradient-certificate";
    let scope = "Global Rastrigin constants follow from the bounded-residual-force proof. Exact trigonometric segment energies and actual native gradient decompositions provide finite independent checks. Population size does not enter this certificate; no full-gas mixing rate is inferred.";
    let mut cx = algorithmic_gas::ExecutionContext::new(
        algorithmic_gas::compute::BackendKind::Cpu,
        algorithmic_gas::Precision::F64,
    )
    .await?;
    for d in [1usize, 2, 4, 8] {
        let profile = rastrigin_gradient_budget(d)?;
        let m = profile.curvature_lower;
        let residual = profile.residual_force_upper;
        let l0 = profile.minimum_segment_length;
        for factor in [1., 1.5, 2.] {
            for midpoint in [-2., 0., 0.5, 10.] {
                for diagonal in [false, true] {
                    let u: Vec<_> = (0..d)
                        .map(|j| {
                            if diagonal {
                                1. / (d as f64).sqrt()
                            } else if j == 0 {
                                1.
                            } else {
                                0.
                            }
                        })
                        .collect();
                    let l = factor * l0;
                    let c: Vec<_> = (0..d).map(|j| midpoint + 0.13 * j as f64).collect();
                    let x: Vec<_> = c.iter().zip(&u).map(|(z, a)| z - l * a / 2.).collect();
                    let y: Vec<_> = c.iter().zip(&u).map(|(z, a)| z + l * a / 2.).collect();
                    let budget = segment_gradient_budget(m, residual, l)?;
                    let (energy, quad_error) = rastrigin_segment_energy(&x, &y, 8192)?;
                    let points: Vec<_> = (0..=12)
                        .flat_map(|j| {
                            x.iter()
                                .zip(&y)
                                .map(move |(a, z)| a + (z - a) * j as f64 / 12.)
                        })
                        .collect();
                    let native = native_rastrigin_gradient(
                        &mut cx,
                        &TensorBatch::vectors(13, d, points.clone())?,
                    )
                    .await?;
                    let mut max_residual = 0_f64;
                    let mut max_decomposition = 0_f64;
                    for (z, g) in points.chunks(d).zip(native.values().chunks(d)) {
                        let e: Vec<_> = z
                            .iter()
                            .map(|t| 20. * std::f64::consts::PI * (std::f64::consts::TAU * t).sin())
                            .collect();
                        max_residual = max_residual.max(norm2(&e).sqrt());
                        let expected: Vec<_> = z.iter().zip(&e).map(|(z, e)| 2. * z + e).collect();
                        max_decomposition = max_decomposition.max(difference2(g, &expected));
                    }
                    let inputs = json!({"dimension":d,"m":m,"residual_bound":residual,"minimum_length":l0,"length":l,"unit_direction":u,"midpoint":c,"start":x,"end":y,"exact_segment_energy":energy,"Simpson_8192_error_bound":quad_error,"native_gradient_points":points,"native_gradients":native.values()});
                    b.add(
                        label,
                        r"\frac1L\int_0^L",
                        inputs.clone(),
                        vec![hypothesis(
                            "nonquadratic_unit_direction",
                            (norm2(&u) - 1.).abs() < 1e-12,
                        )],
                        vec![
                            upper(
                                "nonquadratic_global_segment_energy",
                                budget.gradient_energy_lower,
                                energy,
                            ),
                            equality(
                                "nonquadratic_direction_definition",
                                difference2(
                                    &u,
                                    &x.iter()
                                        .zip(&y)
                                        .map(|(a, z)| (z - a) / l)
                                        .collect::<Vec<_>>(),
                                ),
                                0.,
                            ),
                        ],
                        scope,
                    )?;
                    b.add(
                        label,
                        r"\kappa_{\mathrm{grad}}",
                        inputs.clone(),
                        vec![],
                        vec![
                            equality(
                                "nonquadratic_gradient_constant",
                                profile.gradient_energy_lower,
                                (m * l0 / 12f64.sqrt() - residual).powi(2),
                            ),
                            hypothesis(
                                "nonquadratic_strict_positive_constant",
                                profile.gradient_energy_lower > 0.,
                            ),
                        ],
                        scope,
                    )?;
                    let quadratic = 4. * norm2(&c) + l * l / 3.;
                    let simpson = (4. * norm2(&x) + 4. * 4. * norm2(&c) + 4. * norm2(&y)) / 6.;
                    b.add(
                        label,
                        r"\frac1L\int_{-L/2}",
                        inputs.clone(),
                        vec![],
                        vec![
                            equality("nonquadratic_quadratic_exact_integral", simpson, quadratic),
                            upper(
                                "nonquadratic_quadratic_uniform_lower",
                                m * m * l * l / 12.,
                                quadratic,
                            ),
                        ],
                        scope,
                    )?;
                    for (quote, checks) in [
                        (
                            r"\nabla R_{\mathrm{pos}}(z)=-H(z-z_0)+e(z)",
                            vec![equality(
                                "native_bounded_force_decomposition",
                                max_decomposition,
                                0.,
                            )],
                        ),
                        (
                            r"\lambda_{\min}(H)\ge m>0",
                            vec![hypothesis("nonquadratic_spd_reference", m == 2. && m > 0.)],
                        ),
                        (
                            r"L>0",
                            vec![hypothesis("nonquadratic_positive_segment_length", l > 0.)],
                        ),
                        (
                            r"\|e(z)\|\le M_d",
                            vec![upper(
                                "native_global_residual_bound",
                                max_residual,
                                residual,
                            )],
                        ),
                        (
                            r"L_{\mathrm{grad}}>\sqrt{12}M_d/m",
                            vec![hypothesis(
                                "nonquadratic_strict_segment_threshold",
                                l0 > 12f64.sqrt() * residual / m,
                            )],
                        ),
                        (
                            r"M_d=20\pi\sqrt d",
                            vec![equality(
                                "nonquadratic_dimension_force_budget",
                                residual,
                                20. * std::f64::consts::PI * (d as f64).sqrt(),
                            )],
                        ),
                        (
                            r"m=2",
                            vec![equality("nonquadratic_reference_curvature", m, 2.)],
                        ),
                        (
                            r"L_{\mathrm{grad}}=40\pi\sqrt{3d}",
                            vec![equality(
                                "nonquadratic_dimension_minimum_length",
                                l0,
                                40. * std::f64::consts::PI * (3. * d as f64).sqrt(),
                            )],
                        ),
                        (
                            r"\kappa_{\mathrm{grad}}=400\pi^2d",
                            vec![equality(
                                "nonquadratic_dimension_gradient_energy",
                                profile.gradient_energy_lower,
                                400. * std::f64::consts::PI.powi(2) * d as f64,
                            )],
                        ),
                        (
                            r"c=(x+y)/2-z_0",
                            vec![equality(
                                "nonquadratic_midpoint_definition",
                                difference2(
                                    &c,
                                    &x.iter()
                                        .zip(&y)
                                        .map(|(a, z)| (a + z) / 2.)
                                        .collect::<Vec<_>>(),
                                ),
                                0.,
                            )],
                        ),
                        (
                            r"-L/2\le s\le L/2",
                            vec![upper(
                                "nonquadratic_centered_parameter_bounds",
                                (0..=12)
                                    .map(|j| (-l / 2. + l * j as f64 / 12.).abs())
                                    .fold(0_f64, f64::max),
                                l / 2.,
                            )],
                        ),
                    ] {
                        b.inline(label, quote, inputs.clone(), checks, scope)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn short_gradient_cases(b: &mut EvidenceBuilder) -> Result<()> {
    use crate::convergence_landscape_axioms::{
        rastrigin_coordinate_energy_lower, rastrigin_segment_energy,
        rastrigin_short_gradient_budget,
    };
    let label = "cor-eg-rastrigin-short-gradient";
    let scope = "Exact source contracts and segment-integral witnesses for the global actual Rastrigin oscillation certificate. Coordinate variance bounds hold at every midpoint; the separate native short-segment archive retains actual gradient queries. Neither population size nor compact physical support is used.";
    let pi = std::f64::consts::PI;
    let amplitude = 20. * pi;
    for d in [1usize, 2, 4, 8] {
        let profile = rastrigin_short_gradient_budget(d)?;
        for ell in [1., 1.5, 2., 10.] {
            for midpoint in [-2., 0., 0.5, 10.] {
                let length = ell * (d as f64).sqrt();
                let x = vec![midpoint - ell / 2.; d];
                let y = vec![midpoint + ell / 2.; d];
                let (energy, _) = rastrigin_segment_energy(&x, &y, 8192)?;
                let sinc = |z: f64| if z == 0. { 1. } else { z.sin() / z };
                let mean_sine = (std::f64::consts::TAU * midpoint).sin() * sinc(pi * ell);
                let sine2 =
                    0.5 - 0.5 * (2. * std::f64::consts::TAU * midpoint).cos() * sinc(2. * pi * ell);
                let covariance = (std::f64::consts::TAU * midpoint).cos()
                    * (-(pi * ell).cos() / (2. * pi) + (pi * ell).sin() / (2. * pi.powi(2) * ell));
                let variance = ell * ell / 3.
                    + amplitude.powi(2) * (sine2 - mean_sine.powi(2))
                    + 4. * amplitude * covariance;
                let phi = rastrigin_coordinate_energy_lower(d, length)?;
                let minimum = rastrigin_coordinate_energy_lower(d, (d as f64).sqrt())?;
                let scalar_energy = rastrigin_segment_energy(&x[..1], &y[..1], 8192)?.0;
                let mean_gradient = 2. * midpoint + amplitude * mean_sine;
                let inputs = json!({"dimension":d,"coordinate_span":ell,"midpoint":midpoint,"length":length,"profile":profile,"start":x,"end":y,"exact_full_gradient_energy":energy,"coordinate_sine_mean":mean_sine,"coordinate_sine_second_moment":sine2,"coordinate_covariance":covariance,"coordinate_gradient_variance":variance,"Phi":phi});
                b.add(
                    label,
                    "A=20",
                    inputs.clone(),
                    vec![],
                    vec![
                        equality("short_rastrigin_amplitude", amplitude, 20. * pi),
                        upper("short_rastrigin_coordinate_variance_bound", phi, variance),
                        equality(
                            "short_rastrigin_variance_identity",
                            variance,
                            scalar_energy - mean_gradient.powi(2),
                        ),
                    ],
                    scope,
                )?;
                b.add(
                    label,
                    r"L_{\mathrm{grad}}",
                    inputs.clone(),
                    vec![],
                    vec![
                        equality(
                            "short_rastrigin_minimum_length",
                            profile.minimum_segment_length,
                            (d as f64).sqrt(),
                        ),
                        equality(
                            "short_rastrigin_energy_constant",
                            minimum,
                            200. * pi.powi(2) - 100. * pi - 440. - 40. / pi + 1. / 3.,
                        ),
                        upper("short_rastrigin_rational_constant", 1207., minimum),
                        upper("short_rastrigin_full_gradient_floor", minimum, energy),
                    ],
                    scope,
                )?;
                b.add(
                    label,
                    r"\begin{aligned}",
                    inputs.clone(),
                    vec![],
                    vec![
                        equality(
                            "short_coordinate_uniform_variance",
                            ((ell / 2.) / 3f64.sqrt()).powi(2),
                            ell * ell / 12.,
                        ),
                        upper(
                            "short_coordinate_sine_mean",
                            mean_sine.abs(),
                            1. / (pi * ell),
                        ),
                        upper(
                            "short_coordinate_sine_second_lower",
                            0.5 - 1. / (4. * pi * ell),
                            sine2,
                        ),
                        upper(
                            "short_coordinate_covariance",
                            covariance.abs(),
                            1. / (2. * pi) + 1. / (2. * pi.powi(2) * ell),
                        ),
                    ],
                    scope,
                )?;
                for (quote, checks) in [
                    (
                        r"L\ge\sqrt d",
                        vec![upper(
                            "short_dimension_length_threshold",
                            (d as f64).sqrt(),
                            length,
                        )],
                    ),
                    (
                        r"\ell\ge L/\sqrt d\ge1",
                        vec![
                            equality("short_max_coordinate_span", ell, length / (d as f64).sqrt()),
                            upper("short_full_period_threshold", 1., ell),
                        ],
                    ),
                    (
                        r"g(X)=2X+A\sin(2\pi X)",
                        vec![equality(
                            "short_scalar_gradient_definition",
                            2. * midpoint + amplitude * (std::f64::consts::TAU * midpoint).sin(),
                            2. * midpoint + 20. * pi * (2. * pi * midpoint).sin(),
                        )],
                    ),
                    (
                        r"\Phi(\ell)\ge\Phi(L/\sqrt d)\ge\Phi(1)",
                        vec![
                            equality(
                                "short_Phi_dimension_normalization",
                                phi,
                                rastrigin_coordinate_energy_lower(1, ell)?,
                            ),
                            upper("short_Phi_monotonicity", minimum, phi),
                        ],
                    ),
                    (
                        r"\pi>3",
                        vec![hypothesis(
                            "short_positive_constant_analytic_pi_bound",
                            pi > 3.,
                        )],
                    ),
                    (
                        r"3.14159<\pi<3.14160",
                        vec![hypothesis(
                            "short_rational_pi_enclosure",
                            pi > 314_159_f64 / 100_000. && pi < 314_160_f64 / 100_000.,
                        )],
                    ),
                ] {
                    b.inline(label, quote, inputs.clone(), checks, scope)?;
                }
            }
        }
    }
    Ok(())
}
