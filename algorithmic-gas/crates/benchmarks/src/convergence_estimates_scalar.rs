//! Exact chapter 1 scalar proof identities, evaluated independently of label coverage.
use crate::{
    convergence_coefficients::standardization_coefficients,
    convergence_estimates::{EstimateEvidence, EstimateSuite},
    convergence_framework::{BoundCheck, hermite_patch},
};
use algorithmic_gas::{
    GasError, ObservationBatch, Result, TensorBatch,
    fitness::{PositiveMap, PositiveMapping, Standardizer},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const SOURCE: &str = include_str!(
    "../../../docs/source/2_fractal_gas/convergence_program/01_fragile_gas_framework.md"
);
const SCOPE: &str = "Chapter 1 scalar identity witness on retained finite inputs. Native Global statistics are evaluated before taking expectations. Auxiliary Hermite formulas retain their own scope; no complete-swarm contraction or universal analytic hypothesis follows from these finite checks.";

fn error(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn upper(id: &str, lhs: f64, rhs: f64) -> BoundCheck {
    let mut check = BoundCheck::upper(id, &[], SCOPE, lhs, rhs);
    let tolerance = 2e-11 * lhs.abs().max(rhs.abs());
    check.passed = lhs.is_finite() && rhs.is_finite() && lhs <= rhs + tolerance;
    check
}
fn relative(id: &str, lhs: f64, rhs: f64, natural_scale: f64) -> BoundCheck {
    let scale = natural_scale.abs();
    let residual = if scale > 0. {
        (lhs - rhs).abs() / scale
    } else if lhs == rhs {
        0.
    } else {
        f64::INFINITY
    };
    upper(id, residual, 2e-9)
}
fn positive_identity(id: &str, lhs: f64, rhs: f64) -> BoundCheck {
    let residual = if lhs > 0. && rhs > 0. {
        (lhs.ln() - rhs.ln()).abs()
    } else {
        f64::INFINITY
    };
    upper(id, residual, 2e-9)
}
fn hypothesis(id: &str, value: bool) -> BoundCheck {
    upper(id, if value { 0. } else { 1. }, 0.)
}
fn norm2(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
}
fn difference2(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| (a - b).powi(2)).sum()
}
fn block(label: &str) -> Result<&'static str> {
    let start = SOURCE
        .find(&format!(":label: {label}\n"))
        .or_else(|| SOURCE.find(&format!("({label})=\n")))
        .ok_or_else(|| error(&format!("unknown scalar source label {label}")))?;
    let rest = &SOURCE[start..];
    Ok(&rest[..rest.find("\n:label: ").unwrap_or(rest.len())])
}
fn display(label: &str, marker: &str) -> Result<String> {
    let expressions = block(label)?
        .split("$$")
        .enumerate()
        .filter_map(|(i, expression)| (i % 2 == 1).then_some(expression.trim()))
        .collect::<Vec<_>>();
    expressions
        .iter()
        .find(|expression| expression.starts_with(marker))
        .or_else(|| {
            expressions
                .iter()
                .find(|expression| expression.contains(marker))
        })
        .map(|expression| (*expression).into())
        .ok_or_else(|| error(&format!("missing exact scalar display {label}: {marker}")))
}

#[derive(Default)]
struct Builder {
    records: BTreeMap<(String, String), EstimateEvidence>,
}
impl Builder {
    fn add(
        &mut self,
        labels: &[&str],
        quote: String,
        inputs: Value,
        hypotheses: Vec<BoundCheck>,
        checks: Vec<BoundCheck>,
        scope: &str,
    ) -> Result<()> {
        if checks.is_empty() || !SOURCE.contains(&quote) {
            return Err(error(
                "scalar evidence needs exact source and nonempty comparisons",
            ));
        }
        let key = (labels.join("|"), quote.clone());
        let record = self.records.entry(key).or_insert_with(|| EstimateEvidence {
            chapter: 1,
            source_labels: labels.iter().map(|label| (*label).into()).collect(),
            source_formula: quote,
            inputs: json!({"cases": []}),
            hypothesis_checks: vec![],
            checks: vec![],
            scope: scope.into(),
        });
        record.inputs["cases"].as_array_mut().unwrap().push(inputs);
        record.hypothesis_checks.extend(hypotheses);
        record.checks.extend(checks);
        Ok(())
    }
    fn shown(
        &mut self,
        label: &str,
        marker: &str,
        inputs: Value,
        hypotheses: Vec<BoundCheck>,
        checks: Vec<BoundCheck>,
    ) -> Result<()> {
        self.add(
            &[label],
            display(label, marker)?,
            inputs,
            hypotheses,
            checks,
            SCOPE,
        )
    }
    fn inline(
        &mut self,
        label: &str,
        quote: &str,
        inputs: Value,
        checks: Vec<BoundCheck>,
    ) -> Result<()> {
        if !block(label)?.contains(quote) {
            return Err(error(&format!(
                "missing exact scalar inline {label}: {quote}"
            )));
        }
        self.add(&[label], quote.into(), inputs, vec![], checks, SCOPE)
    }
}

impl Builder {
    fn inline_prefix(
        &mut self,
        label: &str,
        prefix: &str,
        inputs: Value,
        checks: Vec<BoundCheck>,
    ) -> Result<()> {
        let quote = block(label)?
            .split('$')
            .enumerate()
            .filter_map(|(i, quote)| (i % 2 == 1).then_some(quote.trim()))
            .find(|quote| quote.starts_with(prefix))
            .ok_or_else(|| error(&format!("missing scalar inline prefix {label}: {prefix}")))?;
        self.inline(label, quote, inputs, checks)
    }
}

/// Append individually quoted scalar identities to the framework suite.
pub fn append_framework_scalar(suite: &mut EstimateSuite) -> Result<()> {
    if suite.chapter != 1 {
        return Err(error("framework scalar evidence belongs to chapter 1"));
    }
    let mut builder = Builder::default();
    hermite_identities(&mut builder)?;
    standardization_identities(&mut builder)?;
    expected_errors(&mut builder)?;
    remaining_scalar_display_definitions(&mut builder)?;
    terminal_standardization_contract(&mut builder)?;
    suite.evidence.extend(builder.records.into_values());
    suite.scope_notes.push("Scalar supplement independently reconstructs each Hermite and native standardization proof identity. Positive small quantities use logarithmic or relative comparisons; absolute tolerances cannot validate a zero surrogate. Expected-error declarations use exact enumerated joint raw-value laws.".into());
    Ok(())
}

fn q(s: f64, x: f64) -> f64 {
    let log = x.ln_1p();
    (3. * (x - 2. * log) * s + 2. * (3. * log - 2. * x)) * s + x
}
fn hermite_identities(builder: &mut Builder) -> Result<()> {
    for knee in [1.0001_f64, 1.5, 2., 5., 100., 1e6, 1e12] {
        let patch = hermite_patch(knee)?;
        let [a, b, c, d] = patch.coefficients;
        let x = 1. / knee;
        let value = |s: f64| ((a * s + b) * s + c) * s + d;
        let derivative = |s: f64| (3. * a * s + 2. * b) * s + c;
        let input = json!({"z_max":knee,"x":x,"normalized_coefficients":patch.coefficients,"independent_endpoint_system_solution":patch.independently_solved_coefficients});
        hermite_inline_chains(builder, knee, patch.coefficients, &input)?;
        rescale_inline_clauses(builder, knee, patch.coefficients, &input)?;
        for label in [
            "def-asymmetric-rescale-function",
            "lem-cubic-patch-coefficients",
            "lem-cubic-patch-derivative-bounds",
        ] {
            builder.inline(
                label,
                r"z_{\max} > 1",
                input.clone(),
                vec![hypothesis(
                    "actual_valid_Hermite_knee",
                    knee.is_finite() && knee > 1.,
                )],
            )?;
        }
        builder.inline(
            "lem-cubic-patch-uniqueness",
            r"z_{\max}>1",
            input.clone(),
            vec![hypothesis("actual_unique_patch_knee", knee > 1.)],
        )?;
        for (quote, checks) in [
            (
                r"P(z_{\max}-1) = \log(z_{\max}) + 1",
                vec![positive_identity(
                    "patch_left_value",
                    value(0.),
                    knee.ln() + 1.,
                )],
            ),
            (
                r"P'(z_{\max}-1) = 1 / z_{\max}",
                vec![positive_identity(
                    "patch_positive_left_derivative",
                    derivative(0.),
                    x,
                )],
            ),
            (
                r"P(z_{\max}) = \log(1 + z_{\max}) + 1",
                vec![
                    positive_identity("patch_right_value", value(1.), knee.ln_1p() + 1.),
                    positive_identity("small_patch_endpoint_value_increment", a + b + c, x.ln_1p()),
                ],
            ),
            (
                r"P'(z_{\max}) = 0",
                vec![relative(
                    "patch_right_zero_derivative",
                    derivative(1.),
                    0.,
                    x,
                )],
            ),
        ] {
            builder.inline(
                "def-asymmetric-rescale-function",
                quote,
                input.clone(),
                checks,
            )?;
        }
        for (index, quote) in [
            r"A = \frac{1}{z_{\max}} - 2\log\left(1 + \frac{1}{z_{\max}}\right)",
            r"B = 3\log\left(1 + \frac{1}{z_{\max}}\right) - \frac{2}{z_{\max}}",
            r"C = \frac{1}{z_{\max}}",
            r"D = \log(z_{\max}) + 1",
        ]
        .iter()
        .enumerate()
        {
            builder.inline(
                "lem-cubic-patch-coefficients",
                quote,
                input.clone(),
                vec![relative(
                    "independently_solved_cubic_coefficient",
                    patch.coefficients[index],
                    patch.independently_solved_coefficients[index],
                    patch.coefficients[index].abs(),
                )],
            )?;
        }
        for s in [0., 0.03125, 0.25, 0.5, 0.75, 1.] {
            let z = knee - 1. + s;
            let qp = derivative(s);
            let proof = q(s, x);
            let h = x * 1e-5;
            let independent_partial = (q(s, x + h) - q(s, x - h)) / (2. * h);
            let first_partial = (3. * s * s - 4. * s + 1.) + (6. * s - 6. * s * s) / (1. + x);
            let second_partial = (3. * s - 1.) * (s - 1.) + 6. * s * (1. - s) / (1. + x);
            let point = json!({"patch":input,"s":s,"z":z,"q_from_native_coefficients":qp,"q_source":proof,"finite_difference_x_step":h,"partial_finite_difference":independent_partial});
            let hypotheses = vec![hypothesis(
                "source_normalized_patch_domain",
                (0. ..=1.).contains(&s) && x > 0. && x < 1.,
            )];
            builder.shown(
                "lem-cubic-patch-derivative-bounds",
                r"q(s, x) =",
                point.clone(),
                hypotheses.clone(),
                vec![relative("native_coefficient_to_q_identity", qp, proof, x)],
            )?;
            builder.shown(
                "lem-cubic-patch-derivative-bounds",
                r"\frac{\partial q}{\partial x} = (3s^2",
                point.clone(),
                hypotheses.clone(),
                vec![
                    relative(
                        "unfactored_partial_to_factorized_partial",
                        first_partial,
                        second_partial,
                        1.,
                    ),
                    upper(
                        "independent_partial_difference",
                        (first_partial - independent_partial).abs(),
                        2e-7,
                    ),
                ],
            )?;
            builder.shown(
                "lem-cubic-patch-derivative",
                r"Q'(s) =",
                point.clone(),
                hypotheses.clone(),
                vec![relative(
                    "normalized_Q_derivative_polynomial",
                    qp,
                    3. * a * s * s + 2. * b * s + c,
                    x,
                )],
            )?;
            builder.shown(
                "lem-cubic-patch-derivative",
                r"P'(z(s)) =",
                point.clone(),
                hypotheses,
                vec![relative(
                    "source_expanded_derivative_matches_native_coefficients",
                    qp,
                    proof,
                    x,
                )],
            )?;
            builder.inline(
                "lem-cubic-patch-coefficients",
                r"P(z)=Q(s)=As^3+Bs^2+Cs+D",
                point.clone(),
                vec![relative(
                    "normalized_Q_value_polynomial",
                    value(s),
                    a * s.powi(3) + b * s * s + c * s + d,
                    value(s),
                )],
            )?;
            builder.inline(
                "lem-cubic-patch-coefficients",
                r"s=z-(z_{\max}-1)",
                point.clone(),
                vec![relative(
                    "normalized_coordinate_offset",
                    s,
                    z - (knee - 1.),
                    s.abs().max(1.),
                )],
            )?;
            builder.inline(
                "lem-cubic-patch-derivative",
                r"s = z - (z_{\max}-1)",
                point.clone(),
                vec![relative(
                    "normalized_derivative_coordinate",
                    s,
                    z - (knee - 1.),
                    s.abs().max(1.),
                )],
            )?;
            let sup1 = 6. * s * (1. - s) * 2_f64.ln() + (3. * s - 1.) * (s - 1.);
            let sup2 = (3. - 6. * 2_f64.ln()) * s * s + (6. * 2_f64.ln() - 4.) * s + 1.;
            for marker in [r"q_{sup}(s) = \lim", r"q_{sup}(s) = (3"] {
                builder.shown(
                    "lem-cubic-patch-derivative-bounds",
                    marker,
                    point.clone(),
                    vec![],
                    vec![
                        relative("bounding_quadratic_limit_identity", q(s, 1.), sup1, 1.),
                        relative("bounding_quadratic_expansion", sup1, sup2, 1.),
                    ],
                )?;
            }
        }
        if knee <= 5. {
            let z0 = knee - 1.;
            let z1 = knee;
            let physical = [
                a,
                b - 3. * a * z0,
                3. * a * z0 * z0 - 2. * b * z0 + c,
                -a * z0.powi(3) + b * z0 * z0 - c * z0 + d,
            ];
            let matrix = [
                [z0.powi(3), z0 * z0, z0, 1.],
                [3. * z0 * z0, 2. * z0, 1., 0.],
                [z1.powi(3), z1 * z1, z1, 1.],
                [3. * z1 * z1, 2. * z1, 1., 0.],
            ];
            let targets = [knee.ln() + 1., x, knee.ln_1p() + 1., 0.];
            let mut checks = vec![];
            for (row, target) in matrix.iter().zip(targets) {
                let evaluated = row
                    .iter()
                    .zip(physical)
                    .map(|(entry, coefficient)| entry * coefficient)
                    .sum::<f64>();
                let scale = row
                    .iter()
                    .zip(physical)
                    .map(|(entry, coefficient)| (entry * coefficient).abs())
                    .sum::<f64>();
                checks.push(relative(
                    "physical_endpoint_matrix_row",
                    evaluated,
                    target,
                    scale,
                ));
            }
            let determinant = determinant4(matrix);
            checks.push(positive_identity(
                "endpoint_matrix_determinant",
                determinant,
                (z1 - z0).powi(4),
            ));
            builder.add(&["proof-lem-cubic-patch-uniqueness","lem-cubic-patch-uniqueness"],display("proof-lem-cubic-patch-uniqueness",r"\begin{pmatrix}")?,json!({"z_max":knee,"physical_coefficients":physical,"physical_endpoint_matrix":matrix,"targets":targets}),vec![hypothesis("distinct_endpoint_interval",z1>z0)],checks,SCOPE)?;
            builder.shown(
                "proof-lem-cubic-patch-uniqueness",
                r"\det(M)",
                input.clone(),
                vec![],
                vec![positive_identity(
                    "computed_physical_confluent_Vandermonde_determinant",
                    determinant,
                    (z1 - z0).powi(4),
                )],
            )?;
        }
        let mut log_values = vec![];
        let mut positivity = vec![];
        for z in [
            -700.,
            -10.,
            0.,
            (knee - 1.) / 2.,
            knee - 1.,
            knee - 0.5,
            knee,
            knee + 1.,
        ] {
            let (g, log_g) = if z <= 0. {
                (z.exp(), z)
            } else if z < knee - 1. {
                let g = z.ln_1p() + 1.;
                (g, g.ln())
            } else if z <= knee {
                let g = value(z - (knee - 1.));
                (g, g.ln())
            } else {
                let g = knee.ln_1p() + 1.;
                (g, g.ln())
            };
            log_values.push(json!({"z":z,"g_A":g,"log_g_A":log_g}));
            positivity.push(hypothesis(
                "actual_rescale_value_positive_finite",
                g > 0. && g.is_finite() && log_g.is_finite(),
            ));
            if z <= 0. {
                positivity.push(relative(
                    "tiny_positive_negative_branch_log_identity",
                    g.ln(),
                    z,
                    z.abs().max(1.),
                ));
            }
        }
        builder.inline("lem-potential-boundedness",r"g_{A,\max} := \log(1 + z_{\max}) + 1",json!({"auxiliary_Hermite_knee":knee,"native_auxiliary_endpoint_value":value(1.),"all_four_branch_values":log_values,"declared_global_maximum":knee.ln_1p()+1.,"scope":"Piecewise Hermite rescale maximum; native logistic uses its own configured maximum"}),vec![positive_identity("actual_auxiliary_rescale_endpoint_maximum",value(1.),knee.ln_1p()+1.),upper("sampled_all_branch_values_below_the_endpoint_maximum",log_values.iter().map(|point|point["g_A"].as_f64().unwrap()).fold(0_f64,f64::max),knee.ln_1p()+1.)])?;
        builder.inline(
            "def-asymmetric-rescale-function",
            r"g_A: \mathbb{R} \to \mathbb{R}_{>0}",
            json!({"z_max":knee,"independently_evaluated_all_branch_values":log_values}),
            positivity,
        )?;
    }
    let log2 = 2_f64.ln();
    let leading = 3. - 6. * log2;
    let linear = 6. * log2 - 4.;
    let vertex = -linear / (2. * leading);
    let native_max = q(vertex, 1.);
    let first = 1. - (6. * log2 - 4.).powi(2) / (4. * (3. - 6. * log2));
    let second = 1. - 4. * (3. * log2 - 2.).powi(2) / (12. * (1. - 2. * log2));
    let final_value = 1. + (3. * log2 - 2.).powi(2) / (3. * (2. * log2 - 1.));
    builder.shown("lem-cubic-patch-derivative-bounds",r"\max_s q_{sup}(s)",json!({"quadratic_leading":leading,"quadratic_linear":linear,"vertex":vertex,"evaluated_vertex_value":native_max}),vec![hypothesis("downward_quadratic_interior_vertex",leading<0. && (0. ..=1.).contains(&vertex))],vec![positive_identity("vertex_maximum_first_fraction",native_max,first),positive_identity("vertex_maximum_second_fraction",first,second),positive_identity("vertex_maximum_final_fraction",second,final_value)])?;
    let exact_maximum = hermite_patch(2.)?.uniform_derivative_bound;
    builder.shown("lem-cubic-patch-derivative-bounds",r"L_P =",json!({"native_auxiliary_uniform_derivative_constant":exact_maximum,"closed_vertex_maximum":final_value,"displayed_decimal":1.0054}),vec![],vec![positive_identity("uniform_constant_closed_vertex_identity",exact_maximum,final_value),upper("four_decimal_LP_rounding",(exact_maximum-1.0054).abs(),0.00005)])?;
    builder.inline_prefix(
        "lem-cubic-patch-derivative-bounds",
        r"(3 - 6\log(2)) \approx",
        json!({"actual_bounding_quadratic_leading_coefficient":leading}),
        vec![upper(
            "leading_coefficient_two_decimal_rounding",
            (leading - (-1.16)).abs(),
            0.005,
        )],
    )?;
    builder.inline(
        "lem-cubic-patch-derivative-bounds",
        r"s_v = -b/(2a)",
        json!({"a":leading,"b":linear,"vertex":vertex}),
        vec![relative(
            "actual_quadratic_vertex_definition",
            vertex,
            -linear / (2. * leading),
            vertex,
        )],
    )?;
    builder.inline_prefix(
        "lem-cubic-patch-derivative-bounds",
        r"s_v = \frac{-(6\log(2)-4)}",
        json!({"actual_quadratic_vertex":vertex}),
        vec![
            relative(
                "vertex_equivalent_fraction",
                vertex,
                (4. - 6. * log2) / (6. - 12. * log2),
                vertex,
            ),
            upper(
                "vertex_three_decimal_rounding",
                (vertex - 0.069).abs(),
                0.0005,
            ),
        ],
    )?;
    builder.inline(
        "lem-cubic-patch-derivative-bounds",
        r"L_P \approx 1.0054",
        json!({"exact_LP":exact_maximum,"decimal":1.0054}),
        vec![upper(
            "LP_four_decimal_example",
            (exact_maximum - 1.0054).abs(),
            0.00005,
        )],
    )?;
    let first_decimal = 1. + (2.079_f64 - 2.).powi(2) / (3. * (1.386 - 1.));
    let second_decimal = 1. + 0.00624 / 1.158;
    builder.shown("lem-cubic-patch-derivative-bounds",r"L_P \approx 1 +",json!({"exact_LP":exact_maximum,"decimal_intermediate_values":[first_decimal,second_decimal,1.0054]}),vec![],vec![upper("rounded_intermediate_first",(exact_maximum-first_decimal).abs(),0.0001),upper("rounded_intermediate_second",(first_decimal-second_decimal).abs(),0.0001),upper("rounded_intermediate_last",(second_decimal-1.0054).abs(),0.0001)])?;
    Ok(())
}

fn determinant4(mut matrix: [[f64; 4]; 4]) -> f64 {
    let mut determinant = 1.;
    for column in 0..4 {
        let pivot = (column..4)
            .max_by(|i, j| {
                matrix[*i][column]
                    .abs()
                    .total_cmp(&matrix[*j][column].abs())
            })
            .unwrap();
        if pivot != column {
            matrix.swap(pivot, column);
            determinant = -determinant;
        }
        let diagonal = matrix[column][column];
        determinant *= diagonal;
        if diagonal == 0. {
            return 0.;
        }
        let pivot_row = matrix[column];
        for row in matrix.iter_mut().skip(column + 1) {
            let factor = row[column] / diagonal;
            for (entry, pivot_entry) in row.iter_mut().zip(pivot_row).skip(column) {
                *entry -= factor * pivot_entry;
            }
        }
    }
    determinant
}

fn observations(n: usize) -> Result<ObservationBatch<f64>> {
    Ok(ObservationBatch::positions(TensorBatch::vectors(
        n,
        1,
        (0..n).map(|i| i as f64).collect(),
    )?))
}
fn native(values: &[f64], mask: &[bool], floor: f64) -> Result<(Vec<f64>, f64, f64)> {
    let (z, stats) = Standardizer::Global { sigma_min: floor }.apply(
        values,
        mask,
        &observations(values.len())?,
    )?;
    Ok((z, stats.mean[0], stats.scale[0]))
}
fn vector_identity(id: &str, left: &[f64], right: &[f64], scale: f64) -> BoundCheck {
    relative(id, difference2(left, right).sqrt(), 0., scale)
}

fn standardization_identities(builder: &mut Builder) -> Result<()> {
    for n in [1_usize, 2, 4, 8, 32] {
        for floor in [0.01, 0.1, 1.] {
            for constant in [false, true] {
                let raw = (0..n)
                    .map(|i| {
                        if constant {
                            0.2
                        } else {
                            (i as f64 * 1.37).sin()
                        }
                    })
                    .collect::<Vec<_>>();
                let changed = raw
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v + 0.07 * (i as f64 + 0.2).cos()).clamp(-1., 1.))
                    .collect::<Vec<_>>();
                let masks = [
                    vec![true; n],
                    (0..n).map(|i| i == 0 || i % 2 == 1).collect::<Vec<_>>(),
                ];
                for mask1 in &masks {
                    for mask2 in &masks {
                        let (z1, mu1, s1) = native(&raw, mask1, floor)?;
                        let (zi, mui, si) = native(&changed, mask1, floor)?;
                        let (z2, mu2, s2) = native(&changed, mask2, floor)?;
                        let k1 = mask1.iter().filter(|v| **v).count();
                        let k2 = mask2.iter().filter(|v| **v).count();
                        let stable = mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count();
                        let direct = (0..n)
                            .map(|i| {
                                if mask1[i] {
                                    (raw[i] - changed[i]) / s1
                                } else {
                                    0.
                                }
                            })
                            .collect::<Vec<_>>();
                        let mean = (0..n)
                            .map(|i| if mask1[i] { (mui - mu1) / s1 } else { 0. })
                            .collect::<Vec<_>>();
                        let denom = zi.iter().map(|z| z * (si - s1) / s1).collect::<Vec<_>>();
                        let error = z1.iter().zip(&zi).map(|(a, b)| a - b).collect::<Vec<_>>();
                        let sum = (0..n)
                            .map(|i| direct[i] + mean[i] + denom[i])
                            .collect::<Vec<_>>();
                        let scale = (norm2(&z1).sqrt()
                            + norm2(&zi).sqrt()
                            + norm2(&direct).sqrt()
                            + norm2(&mean).sqrt()
                            + norm2(&denom).sqrt())
                        .max(1.);
                        let coefficient = standardization_coefficients(k1, k2, stable, 1., floor)?;
                        let input = json!({"N":n,"canonical_physical_representative":"ascending physical positions 0..N, never walker labels","floor":floor,"raw1":raw,"raw2":changed,"alive1":mask1,"alive2":mask2,"native_z1":z1,"native_intermediate":zi,"native_z2":z2,"native_means":[mu1,mui,mu2],"native_scales":[s1,si,s2],"direct":direct,"mean_shift":mean,"denominator_shift":denom,"coefficients":coefficient});
                        let hypotheses = vec![
                            hypothesis("both_native_alive_supports_nonempty", k1 > 0 && k2 > 0),
                            hypothesis(
                                "native_positive_regularizer",
                                floor > 0. && s1 >= floor && si >= floor && s2 >= floor,
                            ),
                            hypothesis(
                                "actual_uniform_raw_bound",
                                raw.iter()
                                    .chain(&changed)
                                    .all(|v| v.is_finite() && v.abs() <= 1.),
                            ),
                        ];
                        pipeline_inline_clauses(
                            builder, &input, &raw, &changed, mask1, mask2, floor, &z1, &zi, &z2,
                            mu1, mui, s1, si,
                        )?;
                        for (label, third) in [
                            ("lem-sub-value-error-decomposition", "fluc"),
                            ("lem-algebraic-value-error-decomposition", "denom"),
                        ] {
                            builder.inline(
                                label,
                                if third == "fluc" {
                                    r"\Delta\mathbf{z} = \mathbf{z}_1 - \mathbf{z}_2"
                                } else {
                                    r"\Delta\mathbf{z} = z_1 - z_2"
                                },
                                input.clone(),
                                vec![vector_identity(
                                    "native_value_difference_definition",
                                    &error,
                                    &sum,
                                    scale,
                                )],
                            )?;
                            builder.shown(
                                label,
                                r"\Delta\mathbf{z} = \Delta",
                                input.clone(),
                                hypotheses.clone(),
                                vec![vector_identity(
                                    "native_three_component_value_identity",
                                    &error,
                                    &sum,
                                    scale,
                                )],
                            )?;
                            builder.shown(
                                label,
                                r"\Delta_{\text{direct}} :=",
                                input.clone(),
                                hypotheses.clone(),
                                vec![relative(
                                    "native_direct_norm_formula",
                                    norm2(&direct),
                                    (0..n)
                                        .filter(|i| mask1[*i])
                                        .map(|i| (raw[i] - changed[i]).powi(2) / s1.powi(2))
                                        .sum(),
                                    scale.powi(2),
                                )],
                            )?;
                            builder.shown(
                                label,
                                r"\Delta_{\text{mean}} :=",
                                input.clone(),
                                hypotheses.clone(),
                                vec![relative(
                                    "native_mean_shift_norm_formula",
                                    norm2(&mean),
                                    k1 as f64 * (mui - mu1).powi(2) / s1.powi(2),
                                    scale.powi(2),
                                )],
                            )?;
                            builder.shown(
                                label,
                                &format!("\\Delta_{{\\text{{{third}}}}} :="),
                                input.clone(),
                                hypotheses.clone(),
                                vec![relative(
                                    "native_denominator_shift_norm_formula",
                                    norm2(&denom),
                                    norm2(&zi) * (si - s1).powi(2) / s1.powi(2),
                                    scale.powi(2),
                                )],
                            )?;
                            builder.shown(
                                label,
                                r"\|\Delta\mathbf{z}\|_2^2 \le",
                                input.clone(),
                                hypotheses.clone(),
                                vec![upper(
                                    "actual_three_component_Cauchy",
                                    norm2(&error),
                                    3. * (norm2(&direct) + norm2(&mean) + norm2(&denom)),
                                )],
                            )?;
                            let first = (0..n)
                                .map(|i| {
                                    if mask1[i] {
                                        (raw[i] - mu1) / s1 - (changed[i] - mui) / s1
                                            + (changed[i] - mui) / s1
                                            - (changed[i] - mui) / si
                                    } else {
                                        0.
                                    }
                                })
                                .collect::<Vec<_>>();
                            let second = (0..n)
                                .map(|i| {
                                    if mask1[i] {
                                        (raw[i] - changed[i]) / s1
                                            + (mui - mu1) / s1
                                            + (changed[i] - mui) / s1
                                            - (changed[i] - mui) / si
                                    } else {
                                        0.
                                    }
                                })
                                .collect::<Vec<_>>();
                            let third_a = (0..n)
                                .map(|i| {
                                    direct[i]
                                        + mean[i]
                                        + if mask1[i] {
                                            (changed[i] - mui) * (1. / s1 - 1. / si)
                                        } else {
                                            0.
                                        }
                                })
                                .collect::<Vec<_>>();
                            let third_b = (0..n)
                                .map(|i| {
                                    direct[i]
                                        + mean[i]
                                        + if mask1[i] {
                                            (changed[i] - mui) / si * (si - s1) / s1
                                        } else {
                                            0.
                                        }
                                })
                                .collect::<Vec<_>>();
                            builder.shown(
                                label,
                                r"\Delta\mathbf{z} = \frac{\mathbf{v}_1",
                                input.clone(),
                                hypotheses.clone(),
                                vec![vector_identity(
                                    "first_full_equivalent_decomposition",
                                    &error,
                                    &first,
                                    scale,
                                )],
                            )?;
                            builder.shown(
                                label,
                                r"= \left( \frac{\mathbf{v}_1",
                                input.clone(),
                                hypotheses.clone(),
                                vec![vector_identity(
                                    "second_full_equivalent_decomposition",
                                    &error,
                                    &second,
                                    scale,
                                )],
                            )?;
                            builder.shown(
                                label,
                                r"= \Delta_{\text{direct}} +",
                                input.clone(),
                                hypotheses.clone(),
                                vec![
                                    vector_identity(
                                        "third_full_equivalent_decomposition_left",
                                        &error,
                                        &third_a,
                                        scale,
                                    ),
                                    vector_identity(
                                        "third_full_equivalent_decomposition_right",
                                        &error,
                                        &third_b,
                                        scale,
                                    ),
                                ],
                            )?;
                        }
                        let structural_checks = (0..n)
                            .filter(|i| mask1[*i] && mask2[*i])
                            .map(|i| {
                                relative(
                                    "actual_structural_denominator_identity",
                                    (changed[i] - mui) / si - (changed[i] - mu2) / s2,
                                    (mu2 - mui) / si + (changed[i] - mu2) * (s2 - si) / (si * s2),
                                    scale,
                                )
                            })
                            .collect::<Vec<_>>();
                        if !structural_checks.is_empty() {
                            builder.shown(
                                "lem-sub-indirect-structural-error",
                                r"\frac{v_i-\mu_1}{s_1}",
                                input.clone(),
                                hypotheses.clone(),
                                structural_checks,
                            )?;
                        }
                        let ev = difference2(&z1, &zi);
                        let es = difference2(&zi, &z2);
                        for (marker, lhs, rhs) in [
                            (
                                r"E_{V}^2",
                                ev,
                                (0..n).map(|i| (z1[i] - zi[i]).powi(2)).sum(),
                            ),
                            (
                                r"E_{S}^2",
                                es,
                                (0..n).map(|i| (zi[i] - z2[i]).powi(2)).sum(),
                            ),
                        ] {
                            builder.shown(
                                "thm-deterministic-error-decomposition",
                                marker,
                                input.clone(),
                                hypotheses.clone(),
                                vec![relative(
                                    "native_error_component_definition",
                                    lhs,
                                    rhs,
                                    scale.powi(2),
                                )],
                            )?;
                        }
                        for marker in [
                            r"\|\mathbf{z}_1 - \mathbf{z}_2\|_2^2",
                            r"\| \mathbf{z}_1 - \mathbf{z}_2 \|_2^2",
                        ] {
                            builder.shown(
                                "thm-deterministic-error-decomposition",
                                marker,
                                input.clone(),
                                hypotheses.clone(),
                                vec![upper(
                                    "native_total_error_value_structural_bound",
                                    difference2(&z1, &z2),
                                    2. * ev + 2. * es,
                                )],
                            )?;
                        }
                        builder.inline(
                            "thm-deterministic-error-decomposition",
                            r"z_1 = z(S_1, v_1, M)",
                            input.clone(),
                            vec![vector_identity(
                                "native_first_z_definition",
                                &z1,
                                &raw.iter()
                                    .enumerate()
                                    .map(|(i, v)| if mask1[i] { (v - mu1) / s1 } else { 0. })
                                    .collect::<Vec<_>>(),
                                scale,
                            )],
                        )?;
                        builder.inline(
                            "thm-deterministic-error-decomposition",
                            r"z_2 = z(S_2, v_2, M)",
                            input.clone(),
                            vec![vector_identity(
                                "native_second_z_definition",
                                &z2,
                                &changed
                                    .iter()
                                    .enumerate()
                                    .map(|(i, v)| if mask2[i] { (v - mu2) / s2 } else { 0. })
                                    .collect::<Vec<_>>(),
                                scale,
                            )],
                        )?;
                        let kappa = 0.6 * floor * floor;
                        let epsilon = (0.4_f64).sqrt() * floor;
                        for label in [
                            "def-lipschitz-value-error-coefficients",
                            "def-lipschitz-structural-error-coefficients",
                        ] {
                            builder.shown(label,r"\sigma'_{\min",json!({"native_regularizer":floor,"kappa_var_min":kappa,"epsilon_std":epsilon,"native_scales":[s1,si,s2]}),hypotheses.clone(),vec![positive_identity("combined_floor_matches_native_regularizer",floor,(kappa+epsilon.powi(2)).sqrt()),upper("native_scale_floor",floor,s1.min(si).min(s2))])?;
                        }
                        for (marker, lhs, rhs) in [
                            (
                                r"C_{V,\text{direct}}",
                                coefficient.value_direct,
                                1. / floor.powi(2),
                            ),
                            (
                                r"C_{V,\mu}",
                                coefficient.value_mean,
                                k1 as f64 * coefficient.mean_value_lipschitz.powi(2)
                                    / floor.powi(2),
                            ),
                            (
                                r"C_{V,\sigma}",
                                coefficient.value_scale,
                                k1 as f64
                                    * (2. / floor).powi(2)
                                    * (coefficient.scale_value_lipschitz / floor).powi(2),
                            ),
                            (
                                r"C_{V,\text{total}}",
                                coefficient.value_total,
                                3. * (coefficient.value_direct
                                    + coefficient.value_mean
                                    + coefficient.value_scale),
                            ),
                        ] {
                            builder.shown(
                                "def-lipschitz-value-error-coefficients",
                                marker,
                                input.clone(),
                                hypotheses.clone(),
                                vec![positive_identity(
                                    "independent_scalar_value_coefficient_identity",
                                    lhs,
                                    rhs,
                                )],
                            )?;
                        }
                        builder.shown(
                            "def-lipschitz-structural-error-coefficients",
                            r"C_{S,\text{direct}}",
                            input.clone(),
                            hypotheses.clone(),
                            vec![positive_identity(
                                "independent_structural_direct_coefficient_identity",
                                coefficient.structural_direct,
                                (2. / floor).powi(2),
                            )],
                        )?;
                        let split =
                            2. * stable as f64 * coefficient.mean_structural_lipschitz.powi(2)
                                / floor.powi(2)
                                + 2. * k1 as f64
                                    * (2. / floor).powi(2)
                                    * coefficient.scale_structural_lipschitz.powi(2)
                                    / floor.powi(2);
                        builder.shown(
                            "def-lipschitz-structural-error-coefficients",
                            r"C_{S,\text{indirect}}",
                            input.clone(),
                            hypotheses.clone(),
                            vec![positive_identity(
                                "independent_structural_indirect_coefficient_identity",
                                coefficient.structural_indirect_split,
                                split,
                            )],
                        )?;
                        for (quote, count, actual) in [
                            (
                                r"k_1:=|\mathcal{A}_1|",
                                k1,
                                mask1.iter().filter(|a| **a).count(),
                            ),
                            (
                                r"k_2:=|\mathcal{A}_2|",
                                k2,
                                mask2.iter().filter(|a| **a).count(),
                            ),
                            (
                                r"k_{\text{stable}}:=|\mathcal{A}_1\cap\mathcal{A}_2|",
                                stable,
                                mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count(),
                            ),
                        ] {
                            builder.inline(
                                "def-lipschitz-structural-error-coefficients",
                                quote,
                                input.clone(),
                                vec![relative(
                                    "native_alive_count_definition",
                                    count as f64,
                                    actual as f64,
                                    n as f64,
                                )],
                            )?;
                        }
                        if n <= 8 && floor == 0.1 {
                            let difference =
                                z1.iter().zip(&z2).map(|(a, b)| a - b).collect::<Vec<_>>();
                            let combined = z1
                                .iter()
                                .zip(&zi)
                                .zip(&z2)
                                .map(|((a, b), c)| (a - b) + (b - c))
                                .collect::<Vec<_>>();
                            builder.inline("lem-sub-value-error-decomposition",r"\Delta\mathbf{z} = \mathbf{z}_1 - \mathbf{z}_2 = \frac{\mathbf{v}_1 - \mu_1}{\sigma'_1} - \frac{\mathbf{v}_2 - \mu_2}{\sigma'_2}",input.clone(),vec![vector_identity("source_full_native_value_difference",&error,&raw.iter().zip(&changed).enumerate().map(|(i,(a,b))|if mask1[i]{(a-mu1)/s1-(b-mui)/si}else{0.}).collect::<Vec<_>>(),scale)])?;
                            builder.inline("lem-algebraic-value-error-decomposition",r"\Delta\mathbf{z} = z_1 - z_2 = (v_1 - \mu_1) / \sigma'_1 - (v_2 - \mu_2) / \sigma'_2",input.clone(),vec![vector_identity("algebraic_source_full_native_value_difference",&error,&raw.iter().zip(&changed).enumerate().map(|(i,(a,b))|if mask1[i]{(a-mu1)/s1-(b-mui)/si}else{0.}).collect::<Vec<_>>(),scale)])?;
                            for label in [
                                "lem-sub-value-error-decomposition",
                                "lem-algebraic-value-error-decomposition",
                            ] {
                                builder.inline(
                                    label,
                                    r"(a+b+c)^2 \leq 3(a^2+b^2+c^2)",
                                    input.clone(),
                                    (0..n)
                                        .map(|i| {
                                            upper(
                                                "actual_coordinate_three_term_Cauchy",
                                                (direct[i] + mean[i] + denom[i]).powi(2),
                                                3. * (direct[i].powi(2)
                                                    + mean[i].powi(2)
                                                    + denom[i].powi(2)),
                                            )
                                        })
                                        .collect(),
                                )?;
                            }
                            builder.inline(
                                "thm-deterministic-error-decomposition",
                                r"z_{\text{inter}} := z(S_1, v_2, M)",
                                input.clone(),
                                vec![vector_identity(
                                    "native_intermediate_definition",
                                    &zi,
                                    &changed
                                        .iter()
                                        .enumerate()
                                        .map(|(i, v)| if mask1[i] { (v - mui) / si } else { 0. })
                                        .collect::<Vec<_>>(),
                                    scale,
                                )],
                            )?;
                            builder.inline(
                                "thm-deterministic-error-decomposition",
                                r"z_1 - z_2 = (z_1 - z_{\text{inter}}) + (z_{\text{inter}} - z_2)",
                                input.clone(),
                                vec![vector_identity(
                                    "native_total_error_algebraic_split",
                                    &difference,
                                    &combined,
                                    scale,
                                )],
                            )?;
                            builder.inline(
                                "thm-deterministic-error-decomposition",
                                r"\|A+B\|_2^2 \leq 2(\|A\|_2^2 + \|B\|_2^2)",
                                input.clone(),
                                vec![upper(
                                    "native_two_component_Hilbert_Cauchy",
                                    norm2(&combined),
                                    2. * ev + 2. * es,
                                )],
                            )?;
                            builder.inline(
                                "thm-deterministic-error-decomposition",
                                r"\Delta\mathbf{z} = z(S, v_1, M) - z(S, v_2, M)",
                                input.clone(),
                                vec![vector_identity(
                                    "fixed_state_error_definition",
                                    &error,
                                    &sum,
                                    scale,
                                )],
                            )?;
                        }
                        radial_standardization(builder, &raw, &changed, mask1, floor, &input)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn radial_standardization(
    builder: &mut Builder,
    raw: &[f64],
    changed: &[f64],
    mask: &[bool],
    floor: f64,
    shared: &Value,
) -> Result<()> {
    let values = raw
        .iter()
        .zip(mask)
        .filter_map(|(v, a)| a.then_some(*v))
        .collect::<Vec<_>>();
    let other = changed
        .iter()
        .zip(mask)
        .filter_map(|(v, a)| a.then_some(*v))
        .collect::<Vec<_>>();
    let k = values.len();
    let mean = values.iter().sum::<f64>() / k as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / k as f64;
    let centered = values.iter().map(|v| v - mean).collect::<Vec<_>>();
    let (z, native_mean, native_scale) = native(&values, &vec![true; k], floor)?;
    let s = (norm2(&centered) / k as f64 + floor * floor).sqrt();
    let output = centered.iter().map(|u| u / s).collect::<Vec<_>>();
    let expected = values
        .iter()
        .map(|v| (v - mean) / (variance + floor * floor).sqrt())
        .collect::<Vec<_>>();
    let input = json!({"shared_native_fixture":shared,"probability_weights":vec![1./k as f64;k],"raw_atoms":values,"other_atoms":other,"centered_Hilbert_vector":centered,"H_norm_squared":norm2(&centered)/k as f64,"m":floor,"s":s,"native_pushforward_atoms":z,"native_mean":native_mean,"native_scale":native_scale});
    let hypotheses = vec![hypothesis(
        "finite_second_moment_and_positive_m",
        values.iter().all(|v| v.is_finite()) && floor > 0.,
    )];
    let other_mean = other.iter().sum::<f64>() / k as f64;
    let other_centered = other.iter().map(|v| v - other_mean).collect::<Vec<_>>();
    let centered_mean = centered.iter().sum::<f64>() / k as f64;
    let centered_twice = centered
        .iter()
        .map(|u| u - centered_mean)
        .collect::<Vec<_>>();
    builder.inline(
        "lem-empirical-standardization-uniform",
        r"H=L^2(\pi)",
        input.clone(),
        vec![
            relative(
                "empirical_coupling_total_mass",
                (0..k).map(|_| 1. / k as f64).sum(),
                1.,
                1.,
            ),
            relative(
                "actual_empirical_Hilbert_norm_squared",
                norm2(&centered) / k as f64,
                variance,
                variance.max(1.),
            ),
        ],
    )?;
    builder.inline(
        "lem-empirical-standardization-uniform",
        r"P:H\to\{U:\mathbb EU=0\}",
        input.clone(),
        vec![
            relative(
                "actual_centering_zero_probability_mean",
                centered_mean,
                0.,
                1.,
            ),
            vector_identity(
                "actual_centering_projection_idempotence",
                &centered,
                &centered_twice,
                norm2(&centered).sqrt().max(1.),
            ),
        ],
    )?;
    builder.inline(
        "lem-empirical-standardization-uniform",
        r"\|PX-PY\|_H\le\|X-Y\|_H",
        input.clone(),
        vec![upper(
            "actual_centering_Hilbert_contraction",
            difference2(&centered, &other_centered) / k as f64,
            difference2(&values, &other) / k as f64,
        )],
    )?;
    if norm2(&centered) == 0. {
        builder.inline(
            "lem-empirical-standardization-uniform",
            r"U=0",
            input.clone(),
            vec![
                relative("actual_zero_Hilbert_input", norm2(&centered), 0., 1.),
                positive_identity("zero_input_radial_denominator", s, floor),
            ],
        )?;
    }
    builder.inline(
        "lem-empirical-standardization-uniform",
        r"m>0",
        input.clone(),
        vec![hypothesis("actual_positive_empirical_floor", floor > 0.)],
    )?;
    builder.shown(
        "lem-empirical-standardization-uniform",
        r"T_\mu(x)=",
        input.clone(),
        hypotheses.clone(),
        vec![
            vector_identity(
                "native_empirical_standardizer_pushforward",
                &z,
                &expected,
                (norm2(&expected).sqrt()).max(1.),
            ),
            relative("native_empirical_probability_mean", native_mean, mean, 1.),
        ],
    )?;
    builder.shown(
        "lem-empirical-standardization-uniform",
        r"R_m(U)=",
        input.clone(),
        hypotheses.clone(),
        vec![vector_identity(
            "native_regularized_radial_map_identity",
            &z,
            &output,
            norm2(&z).sqrt().max(1.),
        )],
    )?;
    let h = other
        .iter()
        .enumerate()
        .map(|(i, v)| v - mean + 0.1 * (i as f64).cos())
        .collect::<Vec<_>>();
    let dot = centered.iter().zip(&h).map(|(u, v)| u * v).sum::<f64>() / k as f64;
    let derivative = centered
        .iter()
        .zip(&h)
        .map(|(u, h)| h / s - u * dot / s.powi(3))
        .collect::<Vec<_>>();
    let step = 1e-6;
    let mapped = |offset: f64| {
        let u = centered
            .iter()
            .zip(&h)
            .map(|(u, h)| u + offset * h)
            .collect::<Vec<_>>();
        let denom = (norm2(&u) / k as f64 + floor * floor).sqrt();
        u.iter().map(|u| u / denom).collect::<Vec<_>>()
    };
    let plus = mapped(step);
    let minus = mapped(-step);
    let finite_difference = plus
        .iter()
        .zip(minus)
        .map(|(p, m)| (p - m) / (2. * step))
        .collect::<Vec<_>>();
    let mut derivative_checks = vec![
        relative(
            "native_Hilbert_radial_denominator_identity",
            s,
            (norm2(&centered) / k as f64 + floor * floor).sqrt(),
            s,
        ),
        upper(
            "radial_derivative_finite_difference_relative",
            difference2(&derivative, &finite_difference).sqrt() / norm2(&derivative).sqrt().max(1.),
            2e-7,
        ),
    ];
    if variance > 0. {
        let radial = centered
            .iter()
            .map(|u| u * floor.powi(2) / s.powi(3))
            .collect::<Vec<_>>();
        let radial_formula = centered
            .iter()
            .map(|u| u / s - u * (norm2(&centered) / k as f64) / s.powi(3))
            .collect::<Vec<_>>();
        derivative_checks.push(vector_identity(
            "radial_derivative_eigenvalue_identity",
            &radial,
            &radial_formula,
            norm2(&radial).sqrt().max(1.),
        ));
    }
    builder.shown("lem-empirical-standardization-uniform",r"DR_m(U)h=",json!({"source_inputs":input,"direction_h":h,"analytic_derivative":derivative,"central_difference":finite_difference,"step":step}),hypotheses,derivative_checks)?;
    Ok(())
}

fn expected_errors(builder: &mut Builder) -> Result<()> {
    for n in [1_usize, 2, 4, 8] {
        for floor in [0.01, 0.1, 1.] {
            let mask1 = vec![true; n];
            let mask2 = (0..n).map(|i| i == 0 || i % 2 == 1).collect::<Vec<_>>();
            let mut worlds = vec![];
            let mut expected_value = 0.;
            let mut expected_structure = 0.;
            let mut coordinate_value = vec![0.; n];
            let mut coordinate_structure = vec![0.; n];
            for (world, probability) in [(0_usize, 0.25), (1, 0.75)] {
                let raw = (0..n)
                    .map(|i| ((i + 1 + world) as f64 * 0.63).sin())
                    .collect::<Vec<_>>();
                let changed = raw
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        if mask2[i] {
                            (v + 0.05 * ((i + world) as f64).cos()).clamp(-1., 1.)
                        } else {
                            0.
                        }
                    })
                    .collect::<Vec<_>>();
                let (left, _, _) = native(&raw, &mask1, floor)?;
                let (intermediate, _, _) = native(&changed, &mask1, floor)?;
                let (right, _, _) = native(&changed, &mask2, floor)?;
                expected_value += probability * difference2(&left, &intermediate);
                expected_structure += probability * difference2(&intermediate, &right);
                for i in 0..n {
                    coordinate_value[i] += probability * (left[i] - intermediate[i]).powi(2);
                    coordinate_structure[i] += probability * (intermediate[i] - right[i]).powi(2);
                }
                worlds.push(json!({"world":world,"probability":probability,"v1":raw,"v2":changed,"native_z1":left,"native_intermediate":intermediate,"native_z2":right}));
            }
            let input = json!({"N":n,"m":floor,"alive1":mask1,"alive2":mask2,"physical_position_state1":(0..n).map(|i|i as f64).collect::<Vec<_>>(),"physical_position_state2":(0..n).map(|i|i as f64+0.1).collect::<Vec<_>>(),"joint_raw_value_law":"two explicitly enumerated paired raw measurement vectors, masses 1/4 and 3/4; physical state2 is shifted and every dead measurement is zero","worlds":worlds,"expected_value_error":expected_value,"expected_structural_error":expected_structure,"coordinate_expected_value_errors":coordinate_value,"coordinate_expected_structural_errors":coordinate_structure});
            let hypotheses = vec![
                hypothesis(
                    "exact_joint_probability_mass",
                    worlds
                        .iter()
                        .map(|world| world["probability"].as_f64().unwrap())
                        .sum::<f64>()
                        == 1.,
                ),
                hypothesis(
                    "raw_operator_dead_entries_are_exactly_zero",
                    worlds.iter().all(|world| {
                        world["v2"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .zip(&mask2)
                            .all(|(value, alive)| *alive || value.as_f64().unwrap() == 0.)
                    }),
                ),
                hypothesis(
                    "positive_floor_and_nonempty_supports",
                    floor > 0. && mask1.iter().any(|a| *a) && mask2.iter().any(|a| *a),
                ),
            ];
            builder.shown(
                "def-expected-squared-value-error",
                r"E_{V,ms}^2",
                input.clone(),
                hypotheses.clone(),
                vec![relative(
                    "enumerated_joint_value_expectation_equals_coordinate_expectations",
                    expected_value,
                    coordinate_value.iter().sum(),
                    expected_value.max(1.),
                )],
            )?;
            builder.shown(
                "def-expected-squared-structural-error",
                r"E_{S,ms}^2",
                input.clone(),
                hypotheses,
                vec![relative(
                    "enumerated_joint_structural_expectation_equals_coordinate_expectations",
                    expected_structure,
                    coordinate_structure.iter().sum(),
                    expected_structure.max(1.),
                )],
            )?;
            builder.inline(
                "def-expected-squared-value-error",
                r"\mathbf{v}_1 \sim V(\mathcal{S}_1)",
                input.clone(),
                vec![relative(
                    "specified_first_marginal_probability_mass",
                    worlds
                        .iter()
                        .map(|w| w["probability"].as_f64().unwrap())
                        .sum(),
                    1.,
                    1.,
                )],
            )?;
            for label in [
                "def-expected-squared-value-error",
                "def-expected-squared-structural-error",
            ] {
                builder.inline(
                    label,
                    r"\mathbf{v}_2 \sim V(\mathcal{S}_2)",
                    input.clone(),
                    vec![relative(
                        "specified_second_marginal_probability_mass",
                        worlds
                            .iter()
                            .map(|w| w["probability"].as_f64().unwrap())
                            .sum(),
                        1.,
                        1.,
                    )],
                )?;
            }
        }
    }
    Ok(())
}

fn hermite_inline_chains(
    builder: &mut Builder,
    knee: f64,
    coefficients: [f64; 4],
    shared: &Value,
) -> Result<()> {
    let [a, b, c, d] = coefficients;
    let x = 1. / knee;
    let value = |s: f64| ((a * s + b) * s + c) * s + d;
    let derivative = |s: f64| (3. * a * s + 2. * b) * s + c;
    if knee <= 5. {
        let z0 = knee - 1.;
        let z1 = knee;
        let y0 = value(0.);
        let y1 = value(1.);
        let dy0 = derivative(0.);
        let dy1 = derivative(1.);
        let change = x.ln_1p();
        let physical = [
            a,
            b - 3. * a * z0,
            3. * a * z0 * z0 - 2. * b * z0 + c,
            -a * z0.powi(3) + b * z0 * z0 - c * z0 + d,
        ];
        let p = |z: f64| ((physical[0] * z + physical[1]) * z + physical[2]) * z + physical[3];
        let dp = |z: f64| (3. * physical[0] * z + 2. * physical[1]) * z + physical[2];
        let input = json!({"source_inputs":shared,"z0":z0,"z1":z1,"endpoint_values":[y0,y1],"endpoint_derivatives":[dy0,dy1],"stable_endpoint_value_increment":change,"physical_cubic_coefficients":physical});
        for (quote, lhs, rhs, scale) in [
            ("z₀ = z_max - 1", z0, knee - 1., knee),
            ("z₁ = z_max", z1, knee, knee),
            ("y₀ = P(z₀) = log(z_max) + 1", p(z0), knee.ln() + 1., y0),
            ("y'₀ = P'(z₀) = 1/z_max", dp(z0), x, x),
            (
                "y₁ = P(z₁) = log(1 + z_max) + 1",
                p(z1),
                knee.ln_1p() + 1.,
                y1,
            ),
            ("y'₁ = P'(z₁) = 0", dp(z1), 0., x),
            ("a z₀³ + b z₀² + c z₀ + d = y₀", p(z0), y0, y0),
            ("3a z₀² + 2b z₀ + c + 0d = y'₀", dp(z0), dy0, x),
            ("a z₁³ + b z₁² + c z₁ + d = y₁", p(z1), y1, y1),
            ("3a z₁² + 2b z₁ + c + 0d = y'₁", dp(z1), dy1, x),
            ("z₁ - z₀ = z_max - (z_max - 1) = 1", z1 - z0, 1., 1.),
            (
                "det(M) = 1⁴ = 1",
                determinant4([
                    [z0.powi(3), z0 * z0, z0, 1.],
                    [3. * z0 * z0, 2. * z0, 1., 0.],
                    [z1.powi(3), z1 * z1, z1, 1.],
                    [3. * z1 * z1, 2. * z1, 1., 0.],
                ]),
                1.,
                1.,
            ),
        ] {
            builder.inline(
                "proof-lem-cubic-patch-uniqueness",
                quote,
                input.clone(),
                vec![relative(
                    "physical_endpoint_inline_identity",
                    lhs,
                    rhs,
                    scale,
                )],
            )?;
        }
        for (quote, chain) in [
            (
                "y₀ = P(z₀) = log(z_max) + 1",
                vec![y0, p(z0), knee.ln() + 1.],
            ),
            ("y'₀ = P'(z₀) = 1/z_max", vec![dy0, dp(z0), x]),
            (
                "y₁ = P(z₁) = log(1 + z_max) + 1",
                vec![y1, p(z1), knee.ln_1p() + 1.],
            ),
            ("y'₁ = P'(z₁) = 0", vec![dy1, dp(z1), 0.]),
        ] {
            builder.inline(
                "proof-lem-cubic-patch-uniqueness",
                quote,
                input.clone(),
                chain
                    .windows(2)
                    .map(|pair| relative("each_physical_endpoint_clause", pair[0], pair[1], x))
                    .collect(),
            )?;
        }
        for (quote, checks) in [
            (
                "P(z) = az³ + bz² + cz + d",
                vec![relative(
                    "physical_cubic_horner_to_expansion",
                    p(z0 + 0.25),
                    physical[0] * (z0 + 0.25).powi(3)
                        + physical[1] * (z0 + 0.25).powi(2)
                        + physical[2] * (z0 + 0.25)
                        + physical[3],
                    value(0.25),
                )],
            ),
            (
                "P'(z) = 3az² + 2bz + c",
                vec![relative(
                    "physical_derivative_to_normalized_derivative",
                    dp(z0 + 0.25),
                    derivative(0.25),
                    x,
                )],
            ),
            (
                "**x** = [a, b, c, d]ᵀ",
                vec![relative(
                    "actual_physical_coefficient_vector_dimension",
                    physical.len() as f64,
                    4.,
                    4.,
                )],
            ),
            (
                "M * **x** = **y**",
                vec![
                    relative("matrix_value_at_start", p(z0), y0, y0),
                    relative("matrix_slope_at_start", dp(z0), dy0, x),
                    relative("matrix_value_at_end", p(z1), y1, y1),
                    relative("matrix_slope_at_end", dp(z1), dy1, x),
                ],
            ),
        ] {
            builder.inline(
                "proof-lem-cubic-patch-uniqueness",
                quote,
                input.clone(),
                checks,
            )?;
        }
        for (quote, lhs, rhs, scale) in [
            (r"z_0 = z_{\max}-1", z0, knee - 1., knee),
            (r"z_1 = z_{\max}", z1, knee, knee),
            (
                r"y_0 = P(z_0) = \log(z_{\max}) + 1",
                value(0.),
                knee.ln() + 1.,
                y0,
            ),
            (r"y'_0 = P'(z_0) = 1/z_{\max}", derivative(0.), x, x),
            (
                r"y_1 = P(z_1) = \log(1 + z_{\max}) + 1",
                value(1.),
                knee.ln_1p() + 1.,
                y1,
            ),
            (r"y'_1 = P'(z_1) = 0", derivative(1.), 0., x),
            (r"Q(0) = y_0", value(0.), y0, y0),
            (r"Q'(1) = y'_1", derivative(1.), dy1, x),
            (
                r"A(0)^3 + B(0)^2 + C(0) + D = y_0 \implies D = y_0 = \log(z_{\max}) + 1",
                d,
                y0,
                knee.ln() + 1.,
            ),
            (
                r"Q'(0) = y'_0 \implies 3A(0)^2 + 2B(0) + C = y'_0 \implies C = y'_0 = \frac{1}{z_{\max}}",
                c,
                dy0,
                x,
            ),
            (
                r"A(1)^3 + B(1)^2 + C(1) + D = y_1 \implies A + B + C + D = y_1",
                a + b + c + d,
                y1,
                y1,
            ),
            (
                r"3A(1)^2 + 2B(1) + C = y'_1 \implies 3A + 2B + C = y'_1",
                3. * a + 2. * b + c,
                dy1,
                x,
            ),
            (
                r"A + B = y_1 - D - C = y_1 - y_0 - y'_0",
                a + b,
                y1 - d - c,
                x,
            ),
            (
                r"3A + 2B = y'_1 - C = y'_1 - y'_0",
                3. * a + 2. * b,
                dy1 - c,
                x,
            ),
            (r"A + B = \Delta y - y'_0", a + b, change - dy0, x),
            (r"3A + 2B = y'_1 - y'_0", 3. * a + 2. * b, dy1 - dy0, x),
            (
                r"2A + 2B = 2\Delta y - 2y'_0",
                2. * a + 2. * b,
                2. * change - 2. * dy0,
                x,
            ),
            (
                r"A = (y'_1 - y'_0) - (2\Delta y - 2y'_0) = y'_1 + y'_0 - 2\Delta y",
                a,
                (dy1 - dy0) - (2. * change - 2. * dy0),
                x,
            ),
            (
                r"A = 0 + \frac{1}{z_{\max}} - 2\log\left(1 + \frac{1}{z_{\max}}\right)",
                a,
                x - 2. * x.ln_1p(),
                x,
            ),
            (
                r"B = (\Delta y - y'_0) - A = (\Delta y - y'_0) - (y'_1 + y'_0 - 2\Delta y) = 3\Delta y - 2y'_0 - y'_1",
                b,
                (change - dy0) - a,
                x,
            ),
            (
                r"B = 3\log\left(1 + \frac{1}{z_{\max}}\right) - \frac{2}{z_{\max}} - 0",
                b,
                3. * x.ln_1p() - 2. * x,
                x,
            ),
        ] {
            builder.inline(
                "lem-cubic-patch-coefficients",
                quote,
                input.clone(),
                vec![relative(
                    "native_coefficient_proof_inline_chain",
                    lhs,
                    rhs,
                    scale,
                )],
            )?;
        }
        for (quote, chain) in [
            (
                r"Q'(0) = y'_0 \implies 3A(0)^2 + 2B(0) + C = y'_0 \implies C = y'_0 = \frac{1}{z_{\max}}",
                vec![
                    derivative(0.),
                    dy0,
                    3. * a * 0_f64.powi(2) + 2. * b * 0. + c,
                    dy0,
                    c,
                    dy0,
                    x,
                ],
            ),
            (
                r"A(0)^3 + B(0)^2 + C(0) + D = y_0 \implies D = y_0 = \log(z_{\max}) + 1",
                vec![
                    a * 0_f64.powi(3) + b * 0_f64.powi(2) + c * 0. + d,
                    y0,
                    d,
                    y0,
                    knee.ln() + 1.,
                ],
            ),
            (
                r"A + B = y_1 - D - C = y_1 - y_0 - y'_0",
                vec![a + b, y1 - d - c, y1 - y0 - dy0],
            ),
            (
                r"3A + 2B = y'_1 - C = y'_1 - y'_0",
                vec![3. * a + 2. * b, dy1 - c, dy1 - dy0],
            ),
            (
                r"A = (y'_1 - y'_0) - (2\Delta y - 2y'_0) = y'_1 + y'_0 - 2\Delta y",
                vec![
                    a,
                    (dy1 - dy0) - (2. * change - 2. * dy0),
                    dy1 + dy0 - 2. * change,
                ],
            ),
            (
                r"B = (\Delta y - y'_0) - A = (\Delta y - y'_0) - (y'_1 + y'_0 - 2\Delta y) = 3\Delta y - 2y'_0 - y'_1",
                vec![
                    b,
                    (change - dy0) - a,
                    (change - dy0) - (dy1 + dy0 - 2. * change),
                    3. * change - 2. * dy0 - dy1,
                ],
            ),
        ] {
            builder.inline(
                "lem-cubic-patch-coefficients",
                quote,
                input.clone(),
                chain
                    .windows(2)
                    .map(|pair| {
                        relative(
                            "each_equivalent_coefficient_chain_clause",
                            pair[0],
                            pair[1],
                            x,
                        )
                    })
                    .collect(),
            )?;
        }
        builder.inline(
            "lem-cubic-patch-coefficients",
            r"\Delta y = y_1 - y_0 = \log(1+z_{\max}) - \log(z_{\max}) = \log(1+1/z_{\max})",
            input.clone(),
            vec![
                positive_identity("native_endpoint_value_difference", y1 - y0, change),
                positive_identity("log_endpoint_difference", knee.ln_1p() - knee.ln(), change),
            ],
        )?;
        builder.inline(
            "lem-cubic-patch-coefficients",
            r"y'_1=0, y'_0=1/z_{\max}, \Delta y=\log(1+1/z_{\max}))",
            input.clone(),
            vec![
                relative("end_slope_zero", dy1, 0., x),
                positive_identity("start_slope_inverse_knee", dy0, x),
                positive_identity("endpoint_log_increment", change, x.ln_1p()),
            ],
        )?;
        for s in [0., 0.25, 1.] {
            let z = z0 + s;
            let point = json!({"source_inputs":input,"s":s,"z":z});
            builder.inline(
                "lem-cubic-patch-coefficients",
                r"s = z-z_0 \in [0, 1]",
                point.clone(),
                vec![
                    relative("source_coordinate_offset", s, z - z0, 1.),
                    hypothesis("actual_normalized_interval", (0. ..=1.).contains(&s)),
                ],
            )?;
            builder.inline(
                "lem-cubic-patch-coefficients",
                r"P'(z) = \frac{d}{dz}Q(z-z_0) = Q'(z-z_0)",
                point.clone(),
                vec![
                    relative(
                        "chain_rule_to_physical_cubic_derivative",
                        dp(z),
                        derivative(s),
                        x,
                    ),
                    relative(
                        "coordinate_chain_rule_unit_slope",
                        ((z + 1e-5 - z0) - (z - 1e-5 - z0)) / (2e-5),
                        1.,
                        1.,
                    ),
                ],
            )?;
            builder.inline(
                "lem-cubic-patch-derivative",
                r"Q(s) = As^3 + Bs^2 + Cs + D",
                point.clone(),
                vec![relative(
                    "Q_horner_to_expanded_identity",
                    value(s),
                    a * s.powi(3) + b * s * s + c * s + d,
                    value(s),
                )],
            )?;
            builder.inline(
                "lem-cubic-patch-derivative",
                r"P'(z) = Q'(s)",
                point.clone(),
                vec![relative(
                    "actual_physical_normalized_derivative_chain_rule",
                    dp(z),
                    derivative(s),
                    x,
                )],
            )?;
            if s == 0. || s == 1. {
                builder.inline(
                    "lem-cubic-patch-coefficients",
                    if s == 0. { "s=0" } else { "s=1" },
                    point,
                    vec![relative(
                        "actual_endpoint_coordinate",
                        s,
                        if s == 0. { 0. } else { 1. },
                        1.,
                    )],
                )?;
            }
        }
    }
    for s in [0., 0.25, 0.5, 1.] {
        let t = 1. - 3. * s + 6. * s / (1. + x);
        let partial = (1. - s) * t;
        let point = json!({"source_inputs":shared,"s":s,"x":x,"T":t,"partial_q_x":partial,"q":q(s,x),"native_patch_derivative":derivative(s)});
        for (quote, checks) in [
            (
                r"q(s, x) = P'(z(s))",
                vec![relative(
                    "source_q_native_derivative_identity",
                    q(s, x),
                    derivative(s),
                    x,
                )],
            ),
            (
                r"x = 1/z_{\max} \in (0, 1)",
                vec![
                    positive_identity("inverse_knee_definition", x, 1. / knee),
                    hypothesis("actual_inverse_knee_interior", x > 0. && x < 1.),
                ],
            ),
            (
                r"x = 1/z_{\max}",
                vec![positive_identity(
                    "source_inverse_knee_identity",
                    x,
                    1. / knee,
                )],
            ),
            (
                r"P'(z(s)) \ge 0",
                vec![upper(
                    "native_q_nonnegativity_relative_roundoff",
                    -derivative(s) / x,
                    2e-9,
                )],
            ),
            (
                r"1+x \in (1,2)",
                vec![hypothesis(
                    "actual_shifted_inverse_knee_domain",
                    1. + x > 1. && 1. + x < 2.,
                )],
            ),
            (
                r"T(s,x) = 1 - 3s + \frac{6s}{1+x}",
                vec![relative(
                    "bracket_T_equivalent_assembly",
                    t,
                    1. + s * (6. / (1. + x) - 3.),
                    t,
                )],
            ),
            (
                r"\frac{\partial T}{\partial s} = -3 + \frac{6}{1+x} > 0",
                vec![
                    relative(
                        "T_partial_finite_difference",
                        (1. - 3. * (s + 1e-5) + 6. * (s + 1e-5) / (1. + x) - t) / 1e-5,
                        -3. + 6. / (1. + x),
                        1.,
                    ),
                    hypothesis("T_slope_strictly_positive", -3. + 6. / (1. + x) > 0.),
                ],
            ),
            (
                r"\frac{\partial q}{\partial x} \ge 0",
                vec![upper("actual_factored_q_partial_nonnegative", -partial, 0.)],
            ),
            (
                r"x \in (0, 1)",
                vec![hypothesis("actual_source_x_domain", x > 0. && x < 1.)],
            ),
        ] {
            builder.inline(
                "lem-cubic-patch-derivative-bounds",
                quote,
                point.clone(),
                checks,
            )?;
        }
        if s == 0. || s == 1. {
            builder.inline(
                "lem-cubic-patch-derivative-bounds",
                if s == 0. { "s=0" } else { "s=1" },
                point.clone(),
                vec![relative(
                    "actual_derivative_proof_endpoint",
                    s,
                    if s == 0. { 0. } else { 1. },
                    1.,
                )],
            )?;
        }
        builder.inline(
            "lem-cubic-patch-derivative-bounds",
            r"T(0,x)=1",
            point,
            vec![positive_identity(
                "bracket_actual_endpoint_minimum",
                1. - 3. * 0. + 6. * 0. / (1. + x),
                1.,
            )],
        )?;
    }
    Ok(())
}

fn remaining_scalar_display_definitions(builder: &mut Builder) -> Result<()> {
    for copies in [1_usize, 2, 8, 32] {
        let values = [-0.8_f64, -0.2, 0.1, 0.7]
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, copies))
            .collect::<Vec<_>>();
        let k = values.len() as f64;
        for floor in [0.01_f64, 0.1, 1.] {
            let (_, observed_mean, observed_scale) =
                native(&values, &vec![true; values.len()], floor)?;
            let direct_mean = values.iter().sum::<f64>() / k;
            let second = values.iter().map(|v| v * v).sum::<f64>() / k;
            let centered_variance = values
                .iter()
                .map(|v| (v - direct_mean).powi(2))
                .sum::<f64>()
                / k;
            let input = json!({"raw":values,"alive":vec![true;values.len()],"sigma_min":floor,"native_mean":observed_mean,"native_scale":observed_scale,"direct_second_moment":second,"direct_centered_variance":centered_variance});
            let hyps = vec![
                hypothesis(
                    "complete_nonempty_real_vector",
                    !values.is_empty() && values.iter().all(|v| v.is_finite()),
                ),
                hypothesis(
                    "positive_regularized_native_scale",
                    floor > 0. && observed_scale > 0.,
                ),
            ];
            builder.shown(
                "lem-empirical-moments-lipschitz",
                r"\mu(\mathbf v) =",
                input.clone(),
                hyps.clone(),
                vec![
                    relative(
                        "native_empirical_mean_definition",
                        observed_mean,
                        direct_mean,
                        values.iter().map(|v| v.abs()).fold(0., f64::max),
                    ),
                    relative(
                        "empirical_second_moment_from_centered_native_statistics",
                        second,
                        observed_scale.powi(2) - floor.powi(2) + observed_mean.powi(2),
                        second,
                    ),
                ],
            )?;
            builder.shown(
                "def-statistical-properties-measurement",
                r"\sigma'_{\text{reg}}(V) :=",
                input,
                hyps,
                vec![positive_identity(
                    "native_regularized_scale_definition",
                    observed_scale,
                    (centered_variance + floor.powi(2)).sqrt(),
                )],
            )?;
        }
    }
    for n in [2_usize, 3, 4] {
        let values = (0..n)
            .map(|i| (2. * i as f64 / (n - 1) as f64 - 1.) * 0.9)
            .collect::<Vec<_>>();
        let ceiling = values.iter().map(|v| v.abs()).fold(0., f64::max);
        for left_mask in 1..(1_usize << n) {
            for right_mask in 1..(1_usize << n) {
                let left = (0..n)
                    .filter(|&i| left_mask & (1 << i) != 0)
                    .collect::<Vec<_>>();
                let right = (0..n)
                    .filter(|&i| right_mask & (1 << i) != 0)
                    .collect::<Vec<_>>();
                let common = left
                    .iter()
                    .copied()
                    .filter(|i| right.contains(i))
                    .collect::<Vec<_>>();
                let only_left = left
                    .iter()
                    .copied()
                    .filter(|i| !right.contains(i))
                    .collect::<Vec<_>>();
                let only_right = right
                    .iter()
                    .copied()
                    .filter(|i| !left.contains(i))
                    .collect::<Vec<_>>();
                let sum = |set: &[usize]| set.iter().map(|&i| values[i]).sum::<f64>();
                let initial = (sum(&only_left) + sum(&common)) - (sum(&only_right) + sum(&common));
                let canceled = sum(&only_left) - sum(&only_right);
                let k1 = left.len() as f64;
                let k2 = right.len() as f64;
                let first = ((k2 - k1) / (k1 * k2)).abs() * (k2 * ceiling);
                let second = (k2 - k1).abs() / (k1 * k2) * k2 * ceiling;
                let last = ceiling / k1 * (k1 - k2).abs();
                let mu1 = sum(&left) / k1;
                let mu2 = sum(&right) / k2;
                let input = json!({"values":values,"S_1":left,"S_2":right,"common":common,"only_left":only_left,"only_right":only_right,"M_f":ceiling,"means":[mu1,mu2]});
                let hyps = vec![
                    hypothesis("nonempty_comparison_sets", k1 > 0. && k2 > 0.),
                    hypothesis(
                        "actual_absolute_value_bound",
                        values.iter().all(|v| v.abs() <= ceiling),
                    ),
                ];
                builder.shown(
                    "lem-set-difference-bound",
                    r"\left( \sum_{j \in S_1 \setminus S_2}",
                    input.clone(),
                    hyps.clone(),
                    vec![
                        relative(
                            "common_sum_cancellation",
                            initial,
                            canceled,
                            2. * ceiling * n as f64,
                        ),
                        relative(
                            "decomposed_full_sum_difference",
                            initial,
                            sum(&left) - sum(&right),
                            2. * ceiling * n as f64,
                        ),
                    ],
                )?;
                builder.shown(
                    "lem-normalization-difference-bound",
                    r"\left| \frac{|S_2| - |S_1|}",
                    input.clone(),
                    hyps.clone(),
                    vec![
                        relative(
                            "cardinality_factor_first_cancellation",
                            first,
                            second,
                            ceiling * n as f64,
                        ),
                        relative(
                            "cardinality_factor_second_cancellation",
                            second,
                            last,
                            ceiling * n as f64,
                        ),
                    ],
                )?;
                builder.shown(
                    "lem-lipschitz-bound-for-the-variance-functional",
                    r"|\mu(\mathbf v_1)^2-\mu(\mathbf v_2)^2|",
                    input,
                    hyps,
                    vec![relative(
                        "difference_of_empirical_mean_squares",
                        (mu1.powi(2) - mu2.powi(2)).abs(),
                        (mu1 + mu2).abs() * (mu1 - mu2).abs(),
                        ceiling * ceiling,
                    )],
                )?;
            }
        }
    }
    use algorithmic_gas::fitness::{PositiveMap, PositiveMapping};
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.,
    };
    for z in [-700_f64, -100., -20., -1., 0., 1., 20., 100., 700.] {
        let native_value = map.map(z)?;
        builder.shown(
            "def-canonical-logistic-rescale-function-example",
            r"g_A(z) :=",
            json!({"score":z,"native_positive_map":map,"native_value":native_value}),
            vec![
                hypothesis("finite_canonical_logistic_score", z.is_finite()),
                hypothesis(
                    "strict_positive_representable_native_value",
                    native_value > 0.,
                ),
            ],
            vec![positive_identity(
                "native_canonical_logistic_definition",
                native_value,
                2. / (1. + (-z).exp()),
            )],
        )?;
    }
    Ok(())
}

fn scalar_clauses(
    builder: &mut Builder,
    label: &str,
    input: &Value,
    clauses: Vec<(&str, Vec<BoundCheck>)>,
) -> Result<()> {
    for (quote, checks) in clauses {
        builder.inline(label, quote, input.clone(), checks)?;
    }
    Ok(())
}

fn auxiliary_rescale(knee: f64, coefficients: [f64; 4], z: f64) -> f64 {
    let [a, b, c, d] = coefficients;
    if z <= 0. {
        z.exp()
    } else if z < knee - 1. {
        z.ln_1p() + 1.
    } else if z <= knee {
        let s = z - (knee - 1.);
        ((a * s + b) * s + c) * s + d
    } else {
        knee.ln_1p() + 1.
    }
}

fn rescale_inline_clauses(
    builder: &mut Builder,
    knee: f64,
    coefficients: [f64; 4],
    shared: &Value,
) -> Result<()> {
    let [a, b, c, d] = coefficients;
    let x = 1. / knee;
    let ell = x.ln_1p();
    let slope = |s: f64| (3. * a * s + 2. * b) * s + c;
    let vertex = (3. * ell - 2. * x) / (3. * (2. * ell - x));
    let maximum = x + (3. * ell - 2. * x).powi(2) / (3. * (2. * ell - x));
    let log2 = 2_f64.ln();
    let lp = 1. + (3. * log2 - 2.).powi(2) / (3. * (2. * log2 - 1.));
    let input = json!({"actual_auxiliary_patch":shared,"x":x,"ell":ell,"normalized_secant":ell,"vertex":vertex,"patch_derivative_maximum":maximum,"global_rescale_modulus":maximum.max(1.),"uniform_L_P":lp,"precision_scope":"The normalized endpoint increment is retained independently of subtraction of large endpoint baselines."});
    scalar_clauses(
        builder,
        "thm-rescale-function-lipschitz",
        &input,
        vec![
            (
                r"x=1/z_{\max}\in(0,1)",
                vec![
                    positive_identity("actual_inverse_Hermite_knee", x, 1. / knee),
                    hypothesis("inverse_knee_in_open_unit_interval", x > 0. && x < 1.),
                ],
            ),
            (
                r"\ell=\log(1+x)",
                vec![positive_identity(
                    "actual_normalized_log_increment",
                    ell,
                    x.ln_1p(),
                )],
            ),
            (
                r"\ell>x/2",
                vec![hypothesis(
                    "actual_log_secant_strict_half_slope",
                    ell > x / 2.,
                )],
            ),
            (
                r"\ell\ge x\log2",
                vec![upper("actual_secant_endpoint_chord", x * log2, ell)],
            ),
            (
                r"3\ell>x",
                vec![hypothesis(
                    "actual_compatible_Hermite_initial_slope",
                    3. * ell > x,
                )],
            ),
            (
                r"s_*=(3\ell-2x)/(3(2\ell-x))",
                vec![
                    positive_identity("actual_derivative_vertex_coordinate", vertex, -b / (3. * a)),
                    hypothesis("actual_vertex_inside_patch", vertex > 0. && vertex < 1.),
                ],
            ),
            (
                r"q(s_*)=x+(3\ell-2x)^2/[3(2\ell-x)]",
                vec![positive_identity(
                    "actual_patch_derivative_vertex_value",
                    slope(vertex),
                    maximum,
                )],
            ),
            (
                r"q(s_*)\le L_P",
                vec![upper(
                    "actual_fixed_knee_patch_maximum_uniform_bound",
                    slope(vertex),
                    lp,
                )],
            ),
        ],
    )?;
    for s in [0_f64, 0.25, 0.5, 0.75, 1.] {
        let z = knee - 1. + s;
        let point = json!({"patch":input,"s":s,"physical_patch_coordinate":z,"independent_derivative":slope(s)});
        builder.inline(
            "thm-rescale-function-lipschitz",
            r"q(s)=3(x-2\ell)s^2+2(3\ell-2x)s+x",
            point.clone(),
            vec![relative(
                "actual_patch_normalized_derivative_polynomial",
                slope(s),
                3. * (x - 2. * ell) * s * s + 2. * (3. * ell - 2. * x) * s + x,
                x,
            )],
        )?;
        builder.inline(
            "thm-rescale-function-lipschitz",
            r"z_{\max}-1 \le z \le z_{\max}",
            point,
            vec![
                hypothesis(
                    "actual_normalized_patch_coordinate_in_unit_interval",
                    (0. ..=1.).contains(&s),
                ),
                hypothesis(
                    "actual_physical_patch_coordinate_in_its_declared_interval",
                    z >= knee - 1. && z <= knee,
                ),
            ],
        )?;
    }
    let left_z = -1_f64;
    builder.inline("thm-rescale-function-lipschitz",r"z \le 0",json!({"patch":input,"actual_exponential_branch_input":left_z,"actual_exponential_branch_value":auxiliary_rescale(knee,coefficients,left_z)}),vec![hypothesis("actual_exponential_branch_condition",left_z<=0.),positive_identity("actual_exponential_branch_evaluation",auxiliary_rescale(knee,coefficients,left_z),left_z.exp())])?;
    let middle_z = (knee - 1.) / 2.;
    let middle = json!({"patch":input,"actual_log_branch_input":middle_z,"actual_log_branch_value":auxiliary_rescale(knee,coefficients,middle_z),"actual_log_branch_derivative":1./(1.+middle_z)});
    builder.inline(
        "thm-rescale-function-lipschitz",
        r"0 < z < z_{\max} - 1",
        middle.clone(),
        vec![hypothesis(
            "actual_log_branch_open_interval",
            middle_z > 0. && middle_z < knee - 1.,
        )],
    )?;
    builder.inline(
        "thm-rescale-function-lipschitz",
        r"g'_A(z) = 1/(1+z)",
        middle,
        vec![
            positive_identity(
                "actual_log_branch_derivative",
                1. / (1. + middle_z),
                (1. + middle_z).recip(),
            ),
            upper(
                "actual_log_branch_derivative_below_one",
                1. / (1. + middle_z),
                1.,
            ),
        ],
    )?;
    let right_z = knee + 1.;
    builder.inline("thm-rescale-function-lipschitz",r"z > z_{\max}",json!({"patch":input,"actual_constant_branch_input":right_z,"actual_constant_branch_value":auxiliary_rescale(knee,coefficients,right_z)}),vec![hypothesis("actual_constant_branch_region",right_z>knee),positive_identity("actual_constant_branch_value",auxiliary_rescale(knee,coefficients,right_z),knee.ln_1p()+1.)])?;
    builder.inline(
        "tab-framework-constants",
        r"\max\{1,\sup|P'|\}\le L_P",
        input.clone(),
        vec![
            positive_identity(
                "actual_four_branch_global_modulus",
                maximum.max(1.),
                1_f64.max(slope(vertex)),
            ),
            upper(
                "actual_piecewise_rescale_global_modulus_uniform_bound",
                maximum.max(1.),
                lp,
            ),
        ],
    )?;
    if knee <= 5. {
        let z0 = knee - 1.;
        let z1 = knee;
        let y0 = d;
        let y1 = a + b + c + d;
        let secant = (y1 - y0) / (z1 - z0);
        let prescribed_final_slope = 0_f64;
        let endpoint = json!({"patch":input,"actual_endpoint_coordinates":[z0,z1],"actual_endpoint_values":[y0,y1],"prescribed_Hermite_endpoint_slopes":[x,prescribed_final_slope],"actual_evaluated_endpoint_derivatives":[slope(0.),slope(1.)],"actual_secant":secant});
        scalar_clauses(
            builder,
            "lem-polynomial-patch-monotonicity",
            &endpoint,
            vec![
                (
                    r"z_0=z_{\max}-1",
                    vec![relative("actual_patch_left_endpoint", z0, knee - 1., knee)],
                ),
                (
                    r"z_1=z_{\max}",
                    vec![positive_identity("actual_patch_right_endpoint", z1, knee)],
                ),
                (
                    r"y_0=\log(z_{\max})+1",
                    vec![positive_identity(
                        "actual_patch_left_endpoint_value",
                        y0,
                        knee.ln() + 1.,
                    )],
                ),
                (
                    r"y_1=\log(1+z_{\max})+1",
                    vec![positive_identity(
                        "actual_patch_right_endpoint_value",
                        y1,
                        knee.ln_1p() + 1.,
                    )],
                ),
                (
                    r"m_0=P'(z_0)=1/z_{\max}",
                    vec![positive_identity(
                        "actual_patch_left_endpoint_slope",
                        slope(0.),
                        1. / knee,
                    )],
                ),
                (
                    r"m_1=P'(z_1)=0",
                    vec![relative("actual_patch_right_zero_slope", slope(1.), 0., x)],
                ),
                (
                    r"\Delta:=(y_1-y_0)/(z_1-z_0)=\log(1+1/z_{\max})>0",
                    vec![
                        positive_identity("actual_physical_endpoint_secant", secant, ell),
                        hypothesis("actual_secant_strictly_positive", secant > 0.),
                    ],
                ),
                (
                    r"m_0,m_1\ge 0",
                    vec![
                        upper("actual_initial_slope_nonnegative", 0., slope(0.)),
                        upper(
                            "actual_prescribed_final_Hermite_slope_nonnegative",
                            0.,
                            prescribed_final_slope,
                        ),
                        relative(
                            "computed_endpoint_derivative_matches_prescribed_nonnegative_slope",
                            slope(1.),
                            prescribed_final_slope,
                            x,
                        ),
                    ],
                ),
                (
                    r"m_0, m_1 \le 3\Delta",
                    vec![
                        upper(
                            "actual_initial_slope_monotone_Hermite_ceiling",
                            slope(0.),
                            3. * secant,
                        ),
                        upper(
                            "actual_final_slope_monotone_Hermite_ceiling",
                            slope(1.),
                            3. * secant,
                        ),
                    ],
                ),
                (
                    r"m_1=0",
                    vec![relative(
                        "actual_right_endpoint_zero_slope",
                        slope(1.),
                        0.,
                        x,
                    )],
                ),
                (
                    r"m_0=1/z_{\max}>0",
                    vec![
                        positive_identity("actual_positive_left_endpoint_slope", slope(0.), x),
                        hypothesis("actual_left_slope_strict_positive", slope(0.) > 0.),
                    ],
                ),
                (
                    r"\log(1+x)\ge x/(1+x)\ge x/2",
                    vec![
                        upper("actual_log_integral_endpoint_lower", x / (1. + x), ell),
                        upper("actual_log_chord_half_slope", x / 2., x / (1. + x)),
                    ],
                ),
                (
                    r"m_0=x<3\log(1+x)=3\Delta",
                    vec![
                        positive_identity("actual_initial_slope_equal_inverse_knee", slope(0.), x),
                        hypothesis("actual_initial_slope_strict_monotone_secant", x < 3. * ell),
                        positive_identity(
                            "actual_monotone_secant_final_identity",
                            3. * ell,
                            3. * secant,
                        ),
                    ],
                ),
                (
                    r"0\le m_0,m_1\le 3\Delta",
                    vec![
                        upper("actual_monotone_Hermite_slope_initial_lower", 0., slope(0.)),
                        relative("actual_monotone_Hermite_slope_final_zero", slope(1.), 0., x),
                        upper(
                            "actual_monotone_Hermite_slope_initial_upper",
                            slope(0.),
                            3. * secant,
                        ),
                        upper(
                            "actual_monotone_Hermite_slope_final_upper",
                            slope(1.),
                            3. * secant,
                        ),
                    ],
                ),
            ],
        )?;
        let panels = 2048_usize;
        let dx = x / panels as f64;
        let integral = (0..=panels)
            .map(|i| {
                let weight = if i == 0 || i == panels {
                    1.
                } else if i % 2 == 0 {
                    2.
                } else {
                    4.
                };
                weight / (1. + i as f64 * dx)
            })
            .sum::<f64>()
            * dx
            / 3.;
        builder.inline("lem-polynomial-patch-monotonicity",r"\log(1+x)=\int_0^x(1+t)^{-1}\,dt",json!({"patch_endpoint_input":endpoint,"quadrature":"2048-panel Simpson integration of the explicitly declared scalar integrand","actual_integral":integral}),vec![positive_identity("actual_log_integral_identity",ell,integral)])?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn pipeline_inline_clauses(
    builder: &mut Builder,
    shared: &Value,
    raw: &[f64],
    changed: &[f64],
    mask1: &[bool],
    mask2: &[bool],
    floor: f64,
    z1: &[f64],
    zi: &[f64],
    z2: &[f64],
    mu1: f64,
    mui: f64,
    s1: f64,
    si: f64,
) -> Result<()> {
    let n = raw.len();
    let first = raw
        .iter()
        .zip(mask1)
        .filter_map(|(&v, &a)| a.then_some(v))
        .collect::<Vec<_>>();
    let second = changed
        .iter()
        .zip(mask1)
        .filter_map(|(&v, &a)| a.then_some(v))
        .collect::<Vec<_>>();
    let fixed_second_support = raw
        .iter()
        .zip(mask2)
        .filter_map(|(&v, &a)| a.then_some(v))
        .collect::<Vec<_>>();
    let k1 = first.len();
    let k2 = fixed_second_support.len();
    let k = k1 as f64;
    let vmax = 1_f64;
    let mean1 = first.iter().sum::<f64>() / k;
    let mean2 = second.iter().sum::<f64>() / k;
    let m21 = norm2(&first) / k;
    let m22 = norm2(&second) / k;
    let variance1 = first.iter().map(|v| (v - mean1).powi(2)).sum::<f64>() / k;
    let variance2 = second.iter().map(|v| (v - mean2).powi(2)).sum::<f64>() / k;
    let delta = difference2(&first, &second).sqrt();
    let lmu = norm2(&vec![1. / k; k1]).sqrt();
    let lm2 = norm2(&vec![2. * vmax / k; k1]).sqrt();
    let lvar = lm2 + 2. * vmax * lmu;
    let kappa_floor = 0.75 * floor * floor;
    let epsilon_std = 0.5 * floor;
    let combined_floor = (kappa_floor + epsilon_std * epsilon_std).sqrt();
    let nc = mask1.iter().zip(mask2).filter(|(a, b)| a != b).count();
    let structural_mean = fixed_second_support.iter().sum::<f64>() / k2 as f64;
    let structural_second = norm2(&fixed_second_support) / k2 as f64;
    let lmu_s = 2. * vmax / k2 as f64;
    let lm2_s = 2. * vmax * vmax / k2 as f64;
    let cmin = 0.5_f64;
    let doubled_lmu = 2. * vmax / (2 * k2) as f64;
    let doubled_lm2 = 2. * vmax * vmax / (2 * k2) as f64;
    let exponent_mu = (doubled_lmu / lmu_s).ln() / 2_f64.ln();
    let exponent_m2 = (doubled_lm2 / lm2_s).ln() / 2_f64.ln();
    let input = json!({"actual_native_standardization_fixture":shared,"alive_vectors":{"first":first,"changed_same_support":second,"fixed_raw_second_support":fixed_second_support},"alive_counts":[k1,k2],"raw_bound":vmax,"fixed_raw_status_mismatch_count":nc,"positive_native_regularizer":floor,"auxiliary_combined_floor":{"kappa_var_min":kappa_floor,"epsilon_std":epsilon_std,"m":combined_floor},"computed":{"empirical_mean":mean1,"changed_mean":mean2,"first_second_moment":m21,"changed_second_moment":m22,"first_variance":variance1,"changed_variance":variance2,"value_difference_norm":delta,"L_mu_M":lmu,"L_m2_M":lm2,"L_var_M":lvar,"L_mu_S":lmu_s,"L_m2_S":lm2_s,"structural_growth_exponents":[exponent_mu,exponent_m2]},"structure_scope":"These conditional scalar estimates keep the complete raw vector fixed while varying the two alive supports in the retained paired physical representation."});
    for (label, kquote, vquote) in [
        (
            "lem-empirical-moments-lipschitz",
            r"k=|\mathcal A(\mathcal S)|",
            r"|v_i|\le V_{\max}",
        ),
        (
            "lem-empirical-aggregator-properties",
            r"k = |\mathcal{A}(\mathcal{S})|",
            r"|v_i| \le V_{\max}",
        ),
    ] {
        scalar_clauses(
            builder,
            label,
            &input,
            vec![
                (
                    kquote,
                    vec![
                        relative(
                            "actual_alive_empirical_measure_cardinality",
                            k,
                            mask1.iter().filter(|&&a| a).count() as f64,
                            k,
                        ),
                        hypothesis("actual_nonempty_empirical_support", k1 > 0),
                    ],
                ),
                (
                    vquote,
                    vec![upper(
                        "actual_complete_raw_value_bound",
                        raw.iter()
                            .chain(changed)
                            .map(|v| v.abs())
                            .fold(0., f64::max),
                        vmax,
                    )],
                ),
            ],
        )?;
    }
    scalar_clauses(
        builder,
        "lem-empirical-moments-lipschitz",
        &input,
        vec![
            (
                r"\mathbf v\in\mathbb R^k",
                vec![hypothesis(
                    "actual_finite_alive_value_vector",
                    first.len() == k1 && first.iter().all(|v| v.is_finite()),
                )],
            ),
            (
                r"L_{\mu,M}=1/\sqrt{k}",
                vec![
                    positive_identity(
                        "actual_empirical_mean_gradient_norm_modulus",
                        lmu,
                        1. / k.sqrt(),
                    ),
                    upper(
                        "actual_native_mean_pair_value_bound",
                        (mu1 - mui).abs(),
                        lmu * delta,
                    ),
                ],
            ),
            (
                r"L_{m_2,M}=2V_{\max}/\sqrt{k}",
                vec![
                    positive_identity(
                        "actual_empirical_second_moment_gradient_envelope",
                        lm2,
                        2. * vmax / k.sqrt(),
                    ),
                    upper(
                        "actual_second_moment_pair_value_bound",
                        (m21 - m22).abs(),
                        lm2 * delta,
                    ),
                ],
            ),
            (
                r"\nabla\mu = (1/k)\,\mathbf 1",
                vec![positive_identity(
                    "actual_constant_mean_gradient_norm",
                    norm2(&vec![1. / k; k1]).sqrt(),
                    lmu,
                )],
            ),
            (
                r"\nabla m_2 = (2/k)\,(v_1,\dots,v_k)",
                vec![relative(
                    "actual_second_moment_directional_gradient",
                    first
                        .iter()
                        .zip(&second)
                        .map(|(a, b)| 2. * a / k * (b - a))
                        .sum::<f64>(),
                    m22 - m21 - delta * delta / k,
                    m21 + m22,
                )],
            ),
        ],
    )?;
    scalar_clauses(
        builder,
        "lem-empirical-aggregator-properties",
        &input,
        vec![
            (
                r"k_1 = |\mathcal{A}(\mathcal{S}_1)|",
                vec![relative(
                    "actual_first_alive_count",
                    k1 as f64,
                    mask1.iter().filter(|&&a| a).count() as f64,
                    n as f64,
                )],
            ),
            (
                r"k_2 = |\mathcal{A}(\mathcal{S}_2)|",
                vec![relative(
                    "actual_second_alive_count",
                    k2 as f64,
                    mask2.iter().filter(|&&a| a).count() as f64,
                    n as f64,
                )],
            ),
            (
                r"L_{\mu,M}(\mathcal{S}) = k^{-1/2}",
                vec![
                    positive_identity("actual_empirical_mean_modulus_alias", lmu, k.powf(-0.5)),
                    upper(
                        "actual_mean_modulus_bounds_value_difference",
                        (mu1 - mui).abs(),
                        lmu * delta,
                    ),
                ],
            ),
            (
                r"L_{m_2,M}(\mathcal{S}) \le 2V_{\max} k^{-1/2}",
                vec![
                    upper(
                        "actual_second_gradient_norm_envelope",
                        norm2(&first.iter().map(|v| 2. * v / k).collect::<Vec<_>>()).sqrt(),
                        2. * vmax * k.powf(-0.5),
                    ),
                    positive_identity(
                        "declared_second_moment_modulus_is_valid_envelope",
                        lm2,
                        2. * vmax * k.powf(-0.5),
                    ),
                ],
            ),
            (
                r"L_{\mathrm{var},M}(\mathcal{S}) := L_{m_2,M}(\mathcal{S}) + 2V_{\max} L_{\mu,M}(\mathcal{S}) \le 4V_{\max} k^{-1/2}",
                vec![
                    positive_identity(
                        "actual_variance_modulus_from_two_moment_moduli",
                        lvar,
                        lm2 + 2. * vmax * lmu,
                    ),
                    upper(
                        "actual_empirical_variance_modulus_population_dependence",
                        lvar,
                        4. * vmax * k.powf(-0.5),
                    ),
                    upper(
                        "actual_variance_pair_with_native_inputs",
                        (variance1 - variance2).abs(),
                        lvar * delta,
                    ),
                ],
            ),
            (
                r"L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}}{|\mathcal{A}(\mathcal{S}_2)|}",
                vec![
                    positive_identity(
                        "actual_fixed_raw_mean_status_modulus",
                        lmu_s,
                        2. * vmax / k2 as f64,
                    ),
                    upper(
                        "actual_fixed_raw_mean_alive_support_difference",
                        (mean1 - structural_mean).abs(),
                        lmu_s * nc as f64,
                    ),
                ],
            ),
            (
                r"L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}^2}{|\mathcal{A}(\mathcal{S}_2)|}",
                vec![
                    positive_identity(
                        "actual_fixed_raw_second_moment_status_modulus",
                        lm2_s,
                        2. * vmax * vmax / k2 as f64,
                    ),
                    upper(
                        "actual_fixed_raw_second_moment_alive_support_difference",
                        (m21 - structural_second).abs(),
                        lm2_s * nc as f64,
                    ),
                ],
            ),
            (
                r"L_{\mu,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}}{k_2}",
                vec![
                    positive_identity(
                        "actual_fixed_raw_mean_status_modulus_proof_alias",
                        lmu_s,
                        2. * vmax / k2 as f64,
                    ),
                    upper(
                        "actual_fixed_raw_proof_mean_difference",
                        (mean1 - structural_mean).abs(),
                        lmu_s * nc as f64,
                    ),
                ],
            ),
            (
                r"L_{m_2,S}(\mathcal{S}_1, \mathcal{S}_2) \le \frac{2V_{\max}^2}{k_2}",
                vec![
                    positive_identity(
                        "actual_fixed_raw_second_status_modulus_proof_alias",
                        lm2_s,
                        2. * vmax * vmax / k2 as f64,
                    ),
                    upper(
                        "actual_fixed_raw_proof_second_difference",
                        (m21 - structural_second).abs(),
                        lm2_s * nc as f64,
                    ),
                ],
            ),
            (
                r"\kappa_{\text{var}} = 1",
                vec![relative(
                    "actual_empirical_deviation_variance_coefficient_one",
                    first.iter().map(|v| (v - mean1).powi(2)).sum::<f64>(),
                    k * variance1,
                    m21 * k,
                )],
            ),
            (
                r"\kappa_{\text{range}} = 1",
                vec![upper(
                    "actual_empirical_variance_range_coefficient_one",
                    variance1,
                    vmax * vmax,
                )],
            ),
            (
                r"p_{\mu,S} = -1",
                vec![relative(
                    "actual_population_scaling_exponent_of_mean_status_modulus",
                    exponent_mu,
                    -1.,
                    1.,
                )],
            ),
            (
                r"p_{m_2,S} = -1",
                vec![relative(
                    "actual_population_scaling_exponent_of_second_status_modulus",
                    exponent_m2,
                    -1.,
                    1.,
                )],
            ),
            (
                r"p_{\text{worst-case}} = -1",
                vec![relative(
                    "actual_maximum_of_measured_status_scaling_exponents",
                    exponent_mu.max(exponent_m2),
                    -1.,
                    1.,
                )],
            ),
            (
                r"p_{\text{worst-case}} = \max(-1, -1) = -1",
                vec![
                    relative("actual_first_scaling_exponent", exponent_mu, -1., 1.),
                    relative("actual_second_scaling_exponent", exponent_m2, -1., 1.),
                    relative(
                        "actual_worst_structural_scaling_exponent",
                        exponent_mu.max(exponent_m2),
                        -1.,
                        1.,
                    ),
                ],
            ),
            (
                r"|v_{1,i} + v_{2,i}| \le 2V_{\max}",
                vec![upper(
                    "actual_pairwise_sum_raw_value_envelope",
                    first
                        .iter()
                        .zip(&second)
                        .map(|(a, b)| (a + b).abs())
                        .fold(0., f64::max),
                    2. * vmax,
                )],
            ),
            (
                r"n_c = \|\mathbf{s}_1 - \mathbf{s}_2\|_2^2 = |\mathcal{A}_1 \Delta \mathcal{A}_2|",
                vec![
                    relative(
                        "actual_fixed_raw_status_bit_squared_distance",
                        mask1
                            .iter()
                            .zip(mask2)
                            .map(|(&a, &b)| ((a as u8 as f64) - (b as u8 as f64)).powi(2))
                            .sum::<f64>(),
                        nc as f64,
                        n as f64,
                    ),
                    relative(
                        "actual_alive_symmetric_difference_count",
                        (0..n).filter(|&i| mask1[i] != mask2[i]).count() as f64,
                        nc as f64,
                        n as f64,
                    ),
                ],
            ),
            (
                r"|\mathcal{A}_1 \Delta \mathcal{A}_2| = n_c",
                vec![relative(
                    "actual_support_symmetric_difference_alias",
                    mask1.iter().zip(mask2).filter(|(a, b)| a != b).count() as f64,
                    nc as f64,
                    n as f64,
                )],
            ),
            (
                r"|k_2 - k_1| \le n_c",
                vec![upper(
                    "actual_alive_count_change_within_symmetric_difference",
                    k1.abs_diff(k2) as f64,
                    nc as f64,
                )],
            ),
            (
                r"\sum_{i \in \mathcal{A}} (v_i - \mu)^2 \le \kappa_{\text{var}} \cdot k \cdot \text{Var}[M]",
                vec![upper(
                    "actual_empirical_variance_deviation_axiom",
                    first.iter().map(|v| (v - mean1).powi(2)).sum::<f64>(),
                    k * variance1,
                )],
            ),
            (
                r"v_i^2 \le V_{\max}^2",
                vec![upper(
                    "actual_raw_second_moment_pointwise_support",
                    first.iter().map(|v| v * v).fold(0., f64::max),
                    vmax * vmax,
                )],
            ),
            (
                r"\mu^2 \ge 0",
                vec![upper(
                    "actual_empirical_mean_squared_nonnegative",
                    0.,
                    mean1 * mean1,
                )],
            ),
            (
                r"\text{Var}[M] \le \kappa_{\text{range}} \cdot V_{\max}^2",
                vec![upper(
                    "actual_range_variance_axiom_with_coefficient_one",
                    variance1,
                    vmax * vmax,
                )],
            ),
            (
                r"\text{Var}[M] \le \frac{1}{k}\sum V_{\max}^2 - \mu^2 = V_{\max}^2 - \mu^2 \le V_{\max}^2",
                vec![
                    upper(
                        "actual_empirical_variance_by_support_and_mean",
                        variance1,
                        vmax * vmax - mean1 * mean1,
                    ),
                    positive_identity(
                        "empirical_average_of_constant_support_squared",
                        (0..k1).map(|_| vmax * vmax).sum::<f64>() / k,
                        vmax * vmax,
                    ),
                    relative(
                        "constant_support_sum_minus_mean_square",
                        (0..k1).map(|_| vmax * vmax).sum::<f64>() / k - mean1 * mean1,
                        vmax * vmax - mean1 * mean1,
                        vmax * vmax,
                    ),
                    upper(
                        "support_minus_mean_square_below_support",
                        vmax * vmax - mean1 * mean1,
                        vmax * vmax,
                    ),
                ],
            ),
            (
                r"k_2 \ge c_{\min} k_1",
                vec![hypothesis(
                    "actual_uniform_relative_alive_support_floor",
                    k2 as f64 >= cmin * k1 as f64,
                )],
            ),
            (
                r"L_{\mu,S}\le 2V_{\max}/k_2\le2V_{\max}/(c_{\min}k_1)\in O(k_1^{-1})",
                vec![
                    positive_identity(
                        "actual_mean_structural_modulus_first_clause",
                        lmu_s,
                        2. * vmax / k2 as f64,
                    ),
                    upper(
                        "actual_mean_structural_modulus_relative_support_bound",
                        2. * vmax / k2 as f64,
                        2. * vmax / (cmin * k),
                    ),
                    positive_identity(
                        "actual_explicit_uniform_mean_scaling_constant",
                        (2. * vmax / (cmin * k)) * k,
                        2. * vmax / cmin,
                    ),
                    relative("actual_mean_scaling_power", exponent_mu, -1., 1.),
                ],
            ),
            (
                r"L_{m_2,S}\le 2V_{\max}^2/k_2\le2V_{\max}^2/(c_{\min}k_1)\in O(k_1^{-1})",
                vec![
                    positive_identity(
                        "actual_second_structural_modulus_first_clause",
                        lm2_s,
                        2. * vmax * vmax / k2 as f64,
                    ),
                    upper(
                        "actual_second_structural_modulus_relative_support_bound",
                        2. * vmax * vmax / k2 as f64,
                        2. * vmax * vmax / (cmin * k),
                    ),
                    positive_identity(
                        "actual_explicit_uniform_second_scaling_constant",
                        (2. * vmax * vmax / (cmin * k)) * k,
                        2. * vmax * vmax / cmin,
                    ),
                    relative("actual_second_scaling_power", exponent_m2, -1., 1.),
                ],
            ),
        ],
    )?;
    let normalization_term = (1. / k - 1. / k2 as f64).abs() * first.iter().sum::<f64>().abs();
    let support_term = (0..n)
        .filter(|&i| mask1[i] && !mask2[i])
        .map(|i| raw[i])
        .sum::<f64>()
        - (0..n)
            .filter(|&i| mask2[i] && !mask1[i])
            .map(|i| raw[i])
            .sum::<f64>();
    builder.inline("lem-empirical-aggregator-properties",r"\left|\frac{1}{k_1} - \frac{1}{k_2}\right| |\sum_{\mathcal{A}_1} v_i| \le \frac{|k_2 - k_1|}{k_1 k_2} (k_1 V_{\max}) = \frac{|k_2 - k_1|}{k_2}V_{\max}",input.clone(),vec![upper("actual_structural_normalization_term_bound",normalization_term,k1.abs_diff(k2)as f64/(k*k2 as f64)*(k*vmax)),relative("actual_cardinality_factor_cancellation",k1.abs_diff(k2)as f64/(k*k2 as f64)*(k*vmax),k1.abs_diff(k2)as f64/k2 as f64*vmax,vmax*n as f64)])?;
    builder.inline("lem-empirical-aggregator-properties",r"\frac{1}{k_2}|\sum_{i \in \mathcal{A}_1 \setminus \mathcal{A}_2} v_i - \sum_{i \in \mathcal{A}_2 \setminus \mathcal{A}_1} v_i| \le \frac{V_{\max}}{k_2}|\mathcal{A}_1 \Delta \mathcal{A}_2|",input.clone(),vec![upper("actual_fixed_raw_support_set_difference_term",support_term.abs()/k2 as f64,vmax/k2 as f64*nc as f64)])?;
    scalar_clauses(
        builder,
        "lem-lipschitz-bound-for-the-variance-functional",
        &input,
        vec![
            (
                r"\mathrm{Var}(\mathbf v):=m_2(\mathbf v) - \mu(\mathbf v)^2",
                vec![relative(
                    "actual_empirical_variance_from_two_moments",
                    variance1,
                    m21 - mean1 * mean1,
                    m21,
                )],
            ),
            (
                r"|v_i|\le V_{\max}",
                vec![upper(
                    "actual_variance_proof_raw_support_bound",
                    first
                        .iter()
                        .chain(&second)
                        .map(|v| v.abs())
                        .fold(0., f64::max),
                    vmax,
                )],
            ),
            (
                r"L_{\mathrm{var}}:=L_{m_2,M}+2 V_{\max} L_{\mu,M}",
                vec![
                    positive_identity(
                        "actual_variance_lipschitz_coefficient_from_moments",
                        lvar,
                        lm2 + 2. * vmax * lmu,
                    ),
                    upper(
                        "actual_variance_difference_with_retained_raw_inputs",
                        (variance1 - variance2).abs(),
                        lvar * delta,
                    ),
                ],
            ),
            (
                r"|\mu(\mathbf v_1)+\mu(\mathbf v_2)|\le 2 V_{\max}",
                vec![upper(
                    "actual_sum_of_empirical_means_support",
                    (mean1 + mean2).abs(),
                    2. * vmax,
                )],
            ),
            (
                r"|\mu(\mathbf v_j)|\le V_{\max}",
                vec![
                    upper("actual_first_empirical_mean_support", mean1.abs(), vmax),
                    upper("actual_changed_empirical_mean_support", mean2.abs(), vmax),
                ],
            ),
            (
                r"|\mu(\mathbf v_1)-\mu(\mathbf v_2)|\le L_{\mu,M}\,\|\mathbf v_1-\mathbf v_2\|_2",
                vec![upper(
                    "actual_empirical_mean_cauchy_value_bound",
                    (mean1 - mean2).abs(),
                    lmu * delta,
                )],
            ),
        ],
    )?;
    {
        let label = "cor-chain-rule-sigma-reg-var";
        scalar_clauses(
            builder,
            label,
            &input,
            vec![
                (
                    r"L_{\mu,M}=1/\sqrt{k}",
                    vec![positive_identity(
                        "actual_chain_rule_empirical_mean_modulus",
                        lmu,
                        1. / k.sqrt(),
                    )],
                ),
                (
                    r"L_{m_2,M}=2V_{\max}/\sqrt{k}",
                    vec![
                        positive_identity(
                            "actual_chain_rule_empirical_second_modulus",
                            lm2,
                            2. * vmax / k.sqrt(),
                        ),
                        upper(
                            "actual_regularized_scale_difference_chain_rule",
                            (s1 - si).abs(),
                            lvar * delta / (2. * floor),
                        ),
                    ],
                ),
            ],
        )?;
    }
    scalar_clauses(
        builder,
        "def-standardization-operator-n-dimensional",
        &input,
        vec![
            (
                r"k = |\mathcal{A}(S_t)|",
                vec![relative(
                    "actual_standardization_alive_count",
                    mask1.iter().filter(|&&a| a).count() as f64,
                    k,
                    n as f64,
                )],
            ),
            (
                r"v \sim V(S_t)",
                vec![
                    hypothesis(
                        "specified_deterministic_raw_measurement_law_is_admissible",
                        raw.iter().all(|v| v.is_finite()),
                    ),
                    relative(
                        "retained_Dirac_raw_law_total_mass",
                        [1_f64].iter().sum(),
                        1.,
                        1.,
                    ),
                ],
            ),
            (
                r"\mu_v = M(S_t; v_A)",
                vec![
                    relative(
                        "actual_alive_empirical_measure_mass",
                        (0..k1).map(|_| 1. / k).sum(),
                        1.,
                        1.,
                    ),
                    relative(
                        "actual_native_mean_matches_empirical_measure_first_moment",
                        mu1,
                        mean1,
                        vmax,
                    ),
                ],
            ),
            (
                r"z = z(S_t, V, M)",
                vec![
                    hypothesis(
                        "actual_native_output_vector_width_and_finiteness",
                        z1.len() == n && z1.iter().all(|z| z.is_finite()),
                    ),
                    relative(
                        "native_full_masked_standardization_output",
                        difference2(
                            z1,
                            &(0..n)
                                .map(|i| {
                                    if mask1[i] {
                                        (raw[i] - mean1) / (variance1 + floor * floor).sqrt()
                                    } else {
                                        0.
                                    }
                                })
                                .collect::<Vec<_>>(),
                        )
                        .sqrt(),
                        0.,
                        norm2(z1).sqrt().max(1.),
                    ),
                ],
            ),
        ],
    )?;
    for i in 0..n {
        let row = json!({"actual_native_fixture":input,"physical_row_for_evaluation":i,"alive":mask1[i],"specified_raw_measurement_law":"Dirac measure at the retained complete raw vector, an admissible deterministic special case."});
        builder.inline(
            "def-standardization-operator-n-dimensional",
            r"z_{\text{out}}[i] := z_i",
            row.clone(),
            vec![relative(
                "actual_native_alive_assignment_and_dead_zero",
                z1[i],
                if mask1[i] { (raw[i] - mu1) / s1 } else { 0. },
                z1[i].abs().max(1.),
            )],
        )?;
        if mask1[i] {
            builder.inline(
                "def-standardization-operator-n-dimensional",
                r"z_i := (v_i - \mu_A) / \sigma'_A",
                row.clone(),
                vec![relative(
                    "actual_native_eligible_standardized_score",
                    z1[i],
                    (raw[i] - mu1) / s1,
                    z1[i].abs().max(1.),
                )],
            )?;
            builder.inline(
                "thm-z-score-norm-bound",
                r"z_i = (v_i - \mu_{\mathcal{A}}) / \sigma'_{\mathcal{A}}",
                row.clone(),
                vec![relative(
                    "actual_eligible_score_formula_alias",
                    z1[i],
                    (raw[i] - mu1) / s1,
                    z1[i].abs().max(1.),
                )],
            )?;
            builder.inline(
                "thm-z-score-norm-bound",
                r"|v_i - \mu_{\mathcal{A}}| \le |v_i| + |\mu_{\mathcal{A}}|",
                row,
                vec![upper(
                    "actual_raw_score_numerator_triangle_bound",
                    (raw[i] - mu1).abs(),
                    raw[i].abs() + mu1.abs(),
                )],
            )?;
        }
    }
    let mut definition_clauses = vec![
        (
            r"|v_i| \le V_{\max}",
            vec![upper(
                "actual_norm_proof_complete_raw_support",
                first.iter().map(|v| v.abs()).fold(0., f64::max),
                vmax,
            )],
        ),
        (
            r"|\mu_{\mathcal{A}}| \le V_{\max}",
            vec![upper(
                "actual_native_alive_mean_inside_raw_support",
                mu1.abs(),
                vmax,
            )],
        ),
        (
            r"\sigma'_{\mathcal{A}} = \sigma'_{\text{reg}}(\operatorname{Var}[\mu_{\mathbf{v}}])",
            vec![positive_identity(
                "actual_native_scale_from_empirical_measure_variance",
                s1,
                (variance1 + combined_floor * combined_floor).sqrt(),
            )],
        ),
        (
            r"\sigma'_{\min\,\text{bound}} := \sqrt{\kappa_{\text{var,min}} + \varepsilon_{\mathrm{std}}^2}",
            vec![positive_identity(
                "actual_positive_combined_scale_floor",
                floor,
                (kappa_floor + epsilon_std * epsilon_std).sqrt(),
            )],
        ),
        (
            r"\sigma'_{\mathcal{A}} \ge \sigma'_{\min\,\text{bound}}",
            vec![upper(
                "actual_native_scale_above_combined_floor",
                combined_floor,
                s1,
            )],
        ),
        (
            r"\sigma'_{\mathrm{reg}}(V)=\sqrt{V+m^2}\ge m",
            vec![
                positive_identity(
                    "actual_native_regularized_scale_identity",
                    s1,
                    (variance1 + floor * floor).sqrt(),
                ),
                upper("actual_regularized_scale_lower_floor", floor, s1),
            ],
        ),
    ];
    for (quote, checks) in definition_clauses.drain(..) {
        builder.inline("thm-z-score-norm-bound", quote, input.clone(), checks)?;
    }
    scalar_clauses(
        builder,
        "lem-lipschitz-constant-of-the-patched-standardization",
        &input,
        vec![
            (
                r"m=\sigma'_{\min}>0",
                vec![
                    positive_identity(
                        "actual_native_map_positive_denominator_floor",
                        floor,
                        combined_floor,
                    ),
                    hypothesis("actual_native_map_floor_positive", floor > 0.),
                ],
            ),
            (
                r"z(\mathbf a)=(\mathbf a-\mu(\mathbf a))/\sqrt{\operatorname{Var}(\mathbf a)+m^2}",
                vec![
                    relative(
                        "actual_complete_centered_standardization_map",
                        difference2(
                            z1,
                            &(0..n)
                                .map(|i| {
                                    if mask1[i] {
                                        (raw[i] - mu1) / (variance1 + floor * floor).sqrt()
                                    } else {
                                        0.
                                    }
                                })
                                .collect::<Vec<_>>(),
                        )
                        .sqrt(),
                        0.,
                        norm2(z1).sqrt().max(1.),
                    ),
                    upper(
                        "actual_complete_native_standardization_N_uniform_value_change",
                        difference2(z1, zi),
                        delta * delta / (floor * floor),
                    ),
                ],
            ),
        ],
    )?;
    scalar_clauses(
        builder,
        "lem-sigma-patch-derivative-bound",
        &input,
        vec![
            (
                r"\sigma'_{\min} = \sqrt{\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2}",
                vec![positive_identity(
                    "actual_auxiliary_combined_floor_identity",
                    floor,
                    (kappa_floor + epsilon_std * epsilon_std).sqrt(),
                )],
            ),
            (
                r"\sigma'_{\text{reg}}(V) = \sqrt{V + \sigma'^2_{\min}}",
                vec![positive_identity(
                    "actual_native_regularized_scale_identity_alias",
                    s1,
                    (variance1 + floor * floor).sqrt(),
                )],
            ),
        ],
    )?;
    scalar_clauses(
        builder,
        "lem-sigma-reg-derivative-bounds",
        &input,
        vec![
            (
                r"a=\sigma'_{\min}>0",
                vec![
                    positive_identity(
                        "actual_derivative_formula_positive_floor",
                        floor,
                        combined_floor,
                    ),
                    hypothesis("actual_derivative_domain_positive_floor", floor > 0.),
                ],
            ),
            (
                r"s(V)=\sqrt{V+a^2}",
                vec![positive_identity(
                    "actual_native_scale_derivative_base_function",
                    s1,
                    (variance1 + floor * floor).sqrt(),
                )],
            ),
            (
                r"(-1)!!=1",
                vec![positive_identity(
                    "first_derivative_empty_double_factorial_convention",
                    2. * (0..1).map(|j| (0.5 - j as f64).abs()).product::<f64>(),
                    1.,
                )],
            ),
            (
                r"V=0",
                vec![
                    positive_identity(
                        "actual_native_zero_variance_scale",
                        native(&vec![mu1; n], mask1, floor)?.2,
                        floor,
                    ),
                    positive_identity(
                        "actual_scale_derivative_supremum_at_zero_variance",
                        1. / (2. * floor),
                        1. / (2. * native(&vec![mu1; n], mask1, floor)?.2),
                    ),
                ],
            ),
        ],
    )?;
    for order in 1..=8_usize {
        let coefficient = (0..order).map(|j| 0.5 - j as f64).product::<f64>();
        let double_factorial = if order == 1 {
            1.
        } else {
            (1..=2 * order - 3)
                .step_by(2)
                .map(|j| j as f64)
                .product::<f64>()
        };
        builder.inline("lem-sigma-reg-derivative-bounds",r"\prod_{j=0}^{n-1}(1/2-j)",json!({"actual_scale_input":input,"derivative_order":order,"differentiated_power_coefficient":coefficient,"independent_odd_double_factorial":double_factorial}),vec![positive_identity("actual_repeated_power_derivative_coefficient",coefficient.abs(),double_factorial/2_f64.powi(order as i32))])?;
    }
    let direct = (0..n)
        .map(|i| {
            if mask1[i] {
                (raw[i] - changed[i]) / s1
            } else {
                0.
            }
        })
        .collect::<Vec<_>>();
    scalar_clauses(
        builder,
        "lem-direct-value-shift-bound",
        &input,
        vec![
            (
                r"\Delta_{\text{direct}} = (\mathbf{v}_1 - \mathbf{v}_2) / \sigma'_1",
                vec![
                    relative(
                        "actual_direct_value_shift_norm_identity",
                        norm2(&direct),
                        delta * delta / (s1 * s1),
                        delta * delta / (s1 * s1),
                    ),
                    upper(
                        "actual_N_normalized_direct_value_shift_floor_bound",
                        norm2(&direct) / n as f64,
                        delta * delta / (floor * floor * n as f64),
                    ),
                ],
            ),
            (
                r"\sigma'_{\min,\text{bound}} := \sqrt{\kappa_{\text{var,min}}+\varepsilon_{\text{std}}^2}",
                vec![positive_identity(
                    "actual_direct_error_combined_floor_identity",
                    floor,
                    (kappa_floor + epsilon_std * epsilon_std).sqrt(),
                )],
            ),
            (
                r"\sigma'_1\ge \sigma'_{\min,\text{bound}}",
                vec![upper(
                    "actual_direct_error_denominator_floor",
                    combined_floor,
                    s1,
                )],
            ),
            (
                r"1/(\sigma'_1)^2 \le 1/(\sigma'_{\min,\text{bound}})^2",
                vec![upper(
                    "actual_direct_error_reciprocal_scale_bound",
                    1. / (s1 * s1),
                    1. / (combined_floor * combined_floor),
                )],
            ),
        ],
    )?;
    scalar_clauses(
        builder,
        "cor-empirical-standardization-uniform-continuity",
        &input,
        vec![
            (
                r"m=\sigma_{\min}",
                vec![
                    positive_identity(
                        "actual_native_Global_regularizer_definition",
                        floor,
                        combined_floor,
                    ),
                    upper(
                        "actual_N_uniform_native_standardizer_pair_bound",
                        difference2(z1, zi) / n as f64,
                        delta * delta / (floor * floor * n as f64),
                    ),
                ],
            ),
            (
                r"m=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}",
                vec![positive_identity(
                    "actual_combined_regularizer_identifies_same_native_map",
                    floor,
                    (kappa_floor + epsilon_std * epsilon_std).sqrt(),
                )],
            ),
            (
                r"\|\mathbf z_j\|_2^2/N\le k_j/N",
                vec![
                    upper(
                        "actual_first_standardized_empirical_second_moment",
                        norm2(z1) / n as f64,
                        k1 as f64 / n as f64,
                    ),
                    upper(
                        "actual_second_standardized_empirical_second_moment",
                        norm2(z2) / n as f64,
                        k2 as f64 / n as f64,
                    ),
                ],
            ),
        ],
    )?;
    let logistic = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.1,
    };
    let positive_values = z1
        .iter()
        .chain(zi)
        .map(|&z| logistic.map(z))
        .collect::<Result<Vec<_>>>()?;
    builder.inline("axiom-rescale-function",r"g_A: \mathbb{R} \to \mathbb{R}_{>0}",json!({"actual_native_fixture":input,"actual_native_rescale_configuration":logistic,"actual_positive_rescaled_values":positive_values}),vec![hypothesis("actual_native_rescale_arguments_finite",z1.iter().chain(zi).all(|z|z.is_finite())),hypothesis("actual_native_rescale_outputs_finite_and_positive",positive_values.iter().all(|v|v.is_finite()&&*v>0.))])?;
    let native_map_ceiling = match logistic {
        PositiveMap::Logistic { amplitude, floor } => amplitude + floor,
        _ => return Err(error("expected configured native scalar logistic map")),
    };
    builder.inline("axiom-rescale-function",r"g_{A,\max} > 0",json!({"actual_native_rescale_configuration":logistic,"declared_global_rescale_ceiling":native_map_ceiling,"actual_native_rescaled_values":positive_values}),vec![hypothesis("actual_rescale_configuration_has_positive_finite_ceiling",native_map_ceiling.is_finite()&&native_map_ceiling>0.),upper("actual_native_positive_map_values_below_configured_ceiling",positive_values.iter().copied().fold(0.,f64::max),native_map_ceiling)])?;
    let patch = hermite_patch(2.)?;
    let x = 0.5_f64;
    let ell = x.ln_1p();
    let lg = 1_f64.max(x + (3. * ell - 2. * x).powi(2) / (3. * (2. * ell - x)));
    let rescaled1 = z1
        .iter()
        .map(|&z| auxiliary_rescale(2., patch.coefficients, z))
        .collect::<Vec<_>>();
    let rescaledi = zi
        .iter()
        .map(|&z| auxiliary_rescale(2., patch.coefficients, z))
        .collect::<Vec<_>>();
    let composition = json!({"actual_native_fixture":input,"specified_auxiliary_Hermite_knee":2.,"actual_rescaled_first_standardized_vector":rescaled1,"actual_rescaled_changed_standardized_vector":rescaledi,"actual_fixed_knee_rescale_modulus":lg,"raw_value_difference_norm":delta,"scope":"The table's rescale composition bound uses its stated piecewise Hermite map following the actual native Global standardizer on a fixed alive support."});
    builder.inline(
        "tab-framework-constants",
        r"\le L_{g_A}/m",
        composition.clone(),
        vec![upper(
            "actual_fixed_support_standardization_piecewise_rescale_Lipschitz_ratio",
            difference2(&rescaled1, &rescaledi).sqrt(),
            lg / floor * delta,
        )],
    )?;
    builder.inline(
        "tab-framework-constants",
        r"m=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}",
        composition,
        vec![positive_identity(
            "actual_table_combined_native_floor",
            floor,
            (kappa_floor + epsilon_std * epsilon_std).sqrt(),
        )],
    )?;
    builder.inline(
        "tab-framework-axiom-summary",
        r"m=\sqrt{\kappa_{\mathrm{var,min}}+\varepsilon_{\mathrm{std}}^2}>0",
        input.clone(),
        vec![
            positive_identity(
                "actual_axiom_summary_combined_floor_identity",
                floor,
                (kappa_floor + epsilon_std * epsilon_std).sqrt(),
            ),
            hypothesis("actual_axiom_summary_positive_floor", floor > 0.),
        ],
    )?;
    let coefficients = standardization_coefficients(
        k1,
        k2,
        mask1.iter().zip(mask2).filter(|(a, b)| **a && **b).count(),
        vmax,
        floor,
    )?;
    let label = "def-composite-continuity-coeffs-recorrected";
    for (quote, checks) in [
        (
            r"C_{V,\text{direct}} = 1/\sigma'^2_{\min,\text{bound}} = 1/(\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2)",
            vec![
                positive_identity(
                    "computed_direct_coefficient_reciprocal_floor",
                    coefficients.value_direct,
                    1. / (floor * floor),
                ),
                positive_identity(
                    "computed_direct_coefficient_combined_floor",
                    coefficients.value_direct,
                    1. / (kappa_floor + epsilon_std * epsilon_std),
                ),
            ],
        ),
        (
            r"C_{V,\mu}(\mathcal{S}) = k (L_{\mu,M})^2 / \sigma'^2_{\min,\text{bound}} = 1/(\kappa_{\text{var,min}} + \varepsilon_{\text{std}}^2)",
            vec![
                positive_identity(
                    "computed_mean_coefficient_alive_count_cancellation",
                    coefficients.value_mean,
                    k * lmu * lmu / (floor * floor),
                ),
                positive_identity(
                    "computed_mean_coefficient_population_uniform_combined_floor",
                    coefficients.value_mean,
                    1. / (kappa_floor + epsilon_std * epsilon_std),
                ),
            ],
        ),
        (
            r"C_{V,\sigma}(\mathcal{S}) = \dfrac{64 V_{\max}^4 L_{\sigma'_{\text{reg}}}^2}{\sigma'^4_{\min,\text{bound}}}",
            vec![positive_identity(
                "computed_scale_coefficient_explicit_canonical_formula",
                coefficients.value_scale,
                64. * vmax.powi(4) * (1. / (2. * floor)).powi(2) / floor.powi(4),
            )],
        ),
    ] {
        builder.add(&[label],quote.into(),json!({"native_scalar_fixture":input,"independently_computed_canonical_coefficients":coefficients}),vec![hypothesis("actual_coefficient_fixture_nonempty_supports_and_positive_floor",k1>0&&k2>0&&floor>0.)],checks,SCOPE)?;
    }
    Ok(())
}

fn terminal_standardization_contract(builder: &mut Builder) -> Result<()> {
    let raw = [0.2_f64, -0.4];
    let mask = [false; 2];
    let native_extinction = matches!(native(&raw, &mask, 0.1), Err(GasError::Extinction));
    let terminal_vector = [0_f64; 2];
    builder.add(&["def-standardization-operator-n-dimensional"],r"k=0".into(),json!({"raw_values":raw,"alive":mask,"actual_native_result":"GasError::Extinction","mathematical_absorbing_extension_standardized_vector":terminal_vector,"terminal_scope":"The source definition terminates and returns a zero representative. The native low-level standardizer reports Extinction; the zero representative is the mathematical terminal extension, not a successful native normalization."}),vec![hypothesis("actual_support_empty",mask.iter().filter(|&&a|a).count()==0)],vec![hypothesis("native_low_level_standardizer_reports_extinction",native_extinction),relative("mathematical_absorbing_zero_output_representative",norm2(&terminal_vector),0.,1.)],"Actual empty-support native extinction contract, together with the source's explicitly declared terminal zero-vector representative. No successful low-level normalization on an extinct swarm is asserted.")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::positive_identity;

    #[test]
    fn zero_cannot_validate_a_small_positive_identity() {
        assert!(!positive_identity("tiny", 0., 1e-300).passed);
        assert!(positive_identity("tiny", 1e-300, 1e-300).passed);
        assert!(!positive_identity("tiny", 1e-300, 2e-300).passed);
    }
}
