//! Explicit finite witnesses for the elementary proof inequalities.
use crate::{
    convergence_estimates::{EstimateEvidence, EstimateSuite},
    convergence_framework::BoundCheck,
    convergence_validation::inventory,
};
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};

fn upper(id: &str, label: &str, lhs: f64, rhs: f64) -> BoundCheck {
    BoundCheck::upper(
        id,
        &[label],
        "Exact declared finite proof fixture",
        lhs,
        rhs,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "Arguments retain the source, fixture, hypotheses, comparisons and scope as separate evidence fields"
)]
fn record(
    suite: &mut EstimateSuite,
    catalog: &[Value],
    label: &str,
    prefix: &str,
    inputs: Value,
    hypotheses: Vec<BoundCheck>,
    checks: Vec<BoundCheck>,
    scope: &str,
) -> Result<()> {
    let formulas = catalog
        .iter()
        .filter(|chapter| chapter["chapter"] == suite.chapter)
        .flat_map(|chapter| {
            chapter["quantitative_expressions"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter(|expression| {
            expression["source_label"] == label || expression["source_label"].is_null()
        })
        .filter_map(|expression| expression["formula"].as_str())
        .filter(|formula| formula.starts_with(prefix))
        .collect::<Vec<_>>();
    if formulas.is_empty() {
        return Err(GasError::Configuration(format!(
            "Unbound proof fixture {label}/{prefix}"
        )));
    }
    for formula in formulas {
        suite.evidence.push(EstimateEvidence {
            chapter: suite.chapter,
            source_labels: vec![label.to_owned()],
            source_formula: formula.to_owned(),
            inputs: inputs.clone(),
            hypothesis_checks: hypotheses.clone(),
            checks: checks.clone(),
            scope: scope.to_owned(),
        });
    }
    Ok(())
}

/// Check every clause, retaining normalized probability weights at every N.
pub fn append_framework(suite: &mut EstimateSuite) -> Result<()> {
    let catalog = inventory()?;
    let label = "lem-inequality-toolbox";
    for n in [1_usize, 2, 4, 16, 64, 256] {
        let x = (0..n)
            .map(|i| (i as f64 + 0.5) / n as f64)
            .collect::<Vec<_>>();
        let p = (0..n)
            .map(|i| 2. * (i + 1) as f64 / (n * (n + 1)) as f64)
            .collect::<Vec<_>>();
        let mean = x.iter().zip(&p).map(|(x, p)| x * p).sum::<f64>();
        let second = x.iter().zip(&p).map(|(x, p)| x * x * p).sum::<f64>();
        let hypotheses = vec![
            upper(
                "nonnegative_values",
                label,
                -x.iter().copied().fold(f64::INFINITY, f64::min),
                0.,
            ),
            upper(
                "nonnegative_weights",
                label,
                -p.iter().copied().fold(f64::INFINITY, f64::min),
                0.,
            ),
            upper(
                "unit_weights",
                label,
                (p.iter().sum::<f64>() - 1.).abs(),
                0.,
            ),
        ];
        for alpha in [0.01_f64, 0.1, 0.5, 0.9, 1.] {
            let power_mean = x
                .iter()
                .zip(&p)
                .map(|(x, p)| p * x.powf(alpha))
                .sum::<f64>();
            let sum = x.iter().sum::<f64>();
            let power_sum = x.iter().map(|x| x.powf(alpha)).sum::<f64>();
            let mut hyp = hypotheses.clone();
            hyp.push(upper("alpha_positive", label, -alpha, 0.));
            hyp.push(upper("alpha_at_most_one", label, alpha, 1.));
            record(
                suite,
                &catalog,
                label,
                r"\left(\sum_i p_i x_i\right)",
                json!({"values":x,"probabilities":p,"alpha":alpha,"weighted_mean":mean}),
                hyp.clone(),
                vec![upper("concave_Jensen", label, power_mean, mean.powf(alpha))],
                "Concave Jensen for finite nonnegative probability laws with unit mass.",
            )?;
            record(
                suite,
                &catalog,
                "lem-subadditivity-power",
                r"\Big( \sum",
                json!({"values":x,"alpha":alpha,"sum":sum}),
                hyp,
                vec![upper(
                    "power_subadditivity",
                    "lem-subadditivity-power",
                    sum.powf(alpha),
                    power_sum,
                )],
                "Nonnegative finite sums, 0<alpha<=1, including equality at alpha=1.",
            )?;
        }
        record(
            suite,
            &catalog,
            label,
            r"(\mathbb{E}[X])^2",
            json!({"values":x,"probabilities":p,"mean":mean,"second_moment":second}),
            hypotheses,
            vec![upper("Cauchy_second_moment", label, mean * mean, second)],
            "Cauchy--Schwarz under a normalized finite probability law.",
        )?;
    }
    for a in [0_f64, 1e-12, 0.03, 1., 1e6] {
        for b in [0_f64, 1e-12, 0.7, 3., 1e6] {
            record(
                suite,
                &catalog,
                label,
                r"\sqrt{a + b}",
                json!({"a":a,"b":b}),
                vec![
                    upper("a_nonnegative", label, -a, 0.),
                    upper("b_nonnegative", label, -b, 0.),
                ],
                vec![upper(
                    "square_root_subadditivity",
                    label,
                    (a + b).sqrt(),
                    a.sqrt() + b.sqrt(),
                )],
                "Nonnegative reals, including zero and unequal physical scales.",
            )?;
        }
    }
    let label = "lem-sub-unify-holder-terms";
    let coefficients = [0.2, 1., 2.5, 0., 4.];
    let exponents = [0.1, 0.25, 0.5, 0.9, 1.];
    let total = coefficients.iter().sum::<f64>();
    for v in [0_f64, 1e-12, 0.03, 0.999, 1., 1.001, 3., 100.] {
        let actual = coefficients
            .iter()
            .zip(exponents)
            .map(|(a, p)| a * v.powf(p))
            .sum::<f64>();
        let piecewise = total * if v <= 1. { 1. } else { v };
        let coarse = total * (1. + v);
        record(
            suite,
            &catalog,
            label,
            r"\sum_{k=1}^M",
            json!({"A":coefficients,"p":exponents,
            "V":v,"A_sum":total,"p_max":1.,"actual_sum":actual,"piecewise_upper":piecewise,"coarse_upper":coarse}),
            vec![
                upper("nonnegative_V", label, -v, 0.),
                upper(
                    "nonnegative_A",
                    label,
                    -coefficients.iter().copied().fold(f64::INFINITY, f64::min),
                    0.,
                ),
                upper("exponents_positive", label, -exponents[0], 0.),
            ],
            vec![
                upper("Holder_sum_piecewise", label, actual, piecewise),
                upper("Holder_piecewise_coarse", label, piecewise, coarse),
            ],
            "Both inequalities checked independently, on both sides of V=1.",
        )?;
        let branch_prefix = if v <= 1. {
            r"\sum_k A_k V^{p_k} \le \sum_k A_k"
        } else {
            r"\sum_k A_k V^{p_k} \le \left(\sum_k A_k\right)"
        };
        record(
            suite,
            &catalog,
            label,
            branch_prefix,
            json!({"A":coefficients,"p":exponents,"V":v,"A_sum":total,
                "p_max":1.,"actual_sum":actual,"branch_upper":piecewise}),
            vec![
                upper(
                    "branch_domain",
                    label,
                    if v <= 1. { v } else { -v },
                    if v <= 1. { 1. } else { -1. },
                ),
                upper("positive_exponent", label, -exponents[0], 0.),
                upper(
                    "all_exponents_at_most_maximum",
                    label,
                    exponents.iter().copied().fold(0_f64, f64::max),
                    1.,
                ),
            ],
            vec![upper("Holder_individual_case", label, actual, piecewise)],
            "Individual Holder proof case with its actual V<=1 or V>1 hypothesis, normalized coefficient sum and maximum exponent.",
        )?;
    }
    coupling_bounds(suite, &catalog)?;
    gaussian_bounds(suite, &catalog)?;
    inline_algebra(suite, &catalog)?;
    gaussian_normalization(suite, &catalog)?;
    holder_inline_contracts(suite, &catalog)?;
    remaining_power_clauses(suite, &catalog)?;
    Ok(())
}

fn coupling_bounds(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    for n in [2_usize, 4, 16, 64, 256] {
        for scale in [0.1_f64, 1., 100.] {
            let x = (0..n)
                .map(|i| scale * (2. * i as f64 / (n - 1) as f64 - 1.))
                .collect::<Vec<_>>();
            let p = vec![1. / n as f64; n];
            let q = (0..n)
                .map(|i| 2. * (i + 1) as f64 / (n * (n + 1)) as f64)
                .collect::<Vec<_>>();
            let common = p.iter().zip(&q).map(|(p, q)| p.min(*q)).collect::<Vec<_>>();
            let epsilon = 1. - common.iter().sum::<f64>();
            let (mut e2, mut e4, mut mismatch) = (0., 0., 0.);
            for i in 0..n {
                for j in 0..n {
                    let joint = (p[i] - common[i]) * (q[j] - common[j]) / epsilon;
                    let difference = x[i] - x[j];
                    e2 += joint * difference.powi(2);
                    e4 += joint * difference.powi(4);
                    if i != j {
                        mismatch += joint;
                    }
                }
            }
            let m4p = x.iter().zip(&p).map(|(x, p)| p * x.powi(4)).sum::<f64>();
            let m4q = x.iter().zip(&q).map(|(x, p)| p * x.powi(4)).sum::<f64>();
            let cauchy = e4.sqrt() * mismatch.sqrt();
            let moment_upper = (8. * (m4p + m4q)).sqrt() * epsilon.sqrt();
            let diameter = x[n - 1] - x[0];
            let inputs = json!({"support":x,"left_weights":p,"right_weights":q,
            "common_weights":common,"TV":epsilon,"mismatch_probability":mismatch,"coupling_second_moment":e2,
            "coupling_fourth_moment":e4,"left_fourth_moment":m4p,"right_fourth_moment":m4q,"diameter":diameter});
            let label = "subsec-w2-coupling-offset-removal";
            let hyp = vec![
                upper(
                    "left_unit_mass",
                    label,
                    (p.iter().sum::<f64>() - 1.).abs(),
                    0.,
                ),
                upper(
                    "right_unit_mass",
                    label,
                    (q.iter().sum::<f64>() - 1.).abs(),
                    0.,
                ),
                upper(
                    "common_mass_mismatch",
                    label,
                    (mismatch - epsilon).abs(),
                    0.,
                ),
            ];
            record(
                suite,
                catalog,
                label,
                r"\mathbb E d(Z,W)^2",
                inputs.clone(),
                hyp.clone(),
                vec![
                    upper("coupling_Cauchy", label, e2, cauchy),
                    upper("fourth_moment_triangle", label, e4, 8. * (m4p + m4q)),
                    upper("coupling_moment_envelope", label, cauchy, moment_upper),
                ],
                "Exact maximal common-mass coupling of finite scalar probability laws. This tests the abstract coupling estimate independently of a native transition.",
            )?;
            let label = "prop-w2-bound-no-offset";
            let mut left = p.clone();
            let mut right = q.clone();
            let (mut i, mut j) = (0, 0);
            let mut w2 = 0.;
            while i < n && j < n {
                let mass = left[i].min(right[j]);
                w2 += mass * (x[i] - x[j]).powi(2);
                left[i] -= mass;
                right[j] -= mass;
                if left[i] <= 1e-15 {
                    i += 1;
                }
                if right[j] <= 1e-15 {
                    j += 1;
                }
            }
            record(
                suite,
                catalog,
                label,
                r"W_2^2(\mu,\nu)\le",
                inputs,
                hyp,
                vec![
                    upper("optimal_transport_below_admissible", label, w2, e2),
                    upper(
                        "bounded_diameter_coupling",
                        label,
                        e2,
                        diameter.powi(2) * epsilon,
                    ),
                    upper("optimal_transport_moment_envelope", label, w2, moment_upper),
                ],
                "Exact optimal scalar transport tested against both the diameter and fourth-moment envelopes, with unit-mass normalization.",
            )?;
        }
    }
    Ok(())
}

fn pdf(z: f64) -> f64 {
    (-z * z / 2.).exp() / (2. * std::f64::consts::PI).sqrt()
}
fn cdf(z: f64) -> f64 {
    let a = z.abs();
    let n = 4096;
    let h = a / n as f64;
    let mut sum = pdf(0.) + pdf(a);
    for i in 1..n {
        sum += (if i % 2 == 0 { 2. } else { 4. }) * pdf(i as f64 * h);
    }
    0.5 + z.signum() * sum * h / 3.
}

fn gaussian_bounds(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let label = "subsec-w2-coupling-offset-removal";
    for shift in [0.02_f64, 0.1, 0.5, 1., 3., 10.] {
        let tv = 2. * cdf(shift / 2.) - 1.;
        let derivative_bound = 0.5 * shift * (2. / std::f64::consts::PI).sqrt();
        record(
            suite,
            catalog,
            label,
            r"\frac12\int|g(z-u)-g(z)|",
            json!({"translation_norm":shift,"Gaussian_TV":tv,
            "E_abs_Z1":(2./std::f64::consts::PI).sqrt()}),
            vec![upper("fixed_positive_variance", label, -1., 0.)],
            vec![
                upper("Gaussian_TV_derivative", label, tv, derivative_bound),
                upper("Gaussian_derivative_coarse", label, derivative_bound, shift),
            ],
            "Equal-covariance Gaussian translation, independently integrated through its scalar normal CDF; the formula depends on the whitened norm in any dimension.",
        )?;
    }
    for input in [0.1_f64, 0.5, 1., 2.] {
        let p = [0.5, 0.5];
        let w = 1. / (1. + (-input).exp());
        let q = [w, 1. - w];
        let left_means = [-1., 1.];
        let right_means = [-1. + 0.2 * input, 1. + 0.2 * input];
        let lo = -12.;
        let hi = 12.;
        let n = 240_000;
        let h = (hi - lo) / n as f64;
        let density = |z: f64, weights: [f64; 2], means: [f64; 2]| {
            weights[0] * pdf(z - means[0]) + weights[1] * pdf(z - means[1])
        };
        let mut integral = 0.;
        for k in 0..n {
            let z = lo + (k as f64 + 0.5) * h;
            integral += (density(z, p, left_means) - density(z, q, right_means)).abs() * h / 2.;
        }
        let lipschitz = 2. * (-0.5_f64).exp() / (2. * std::f64::consts::PI).sqrt();
        let quadrature_error = lipschitz * (hi - lo) * h / 8.;
        let tail = 4. * (-10_f64.powi(2) / 2.).exp();
        let tv_upper = integral + quadrature_error + tail;
        let weight_tv = 0.5 * ((p[0] - q[0]).abs() + (p[1] - q[1]).abs());
        let component_tv = 2. * cdf(0.1 * input) - 1.;
        let mixture_rhs = weight_tv + q.iter().sum::<f64>() * component_tv;
        let local_constant = 0.25 + 0.2 / (2. * std::f64::consts::PI).sqrt();
        record(
            suite,
            catalog,
            label,
            r"\|\Psi(x,\cdot)-\Psi(y,\cdot)\|",
            json!({"input_distance":input,
            "left_weights":p,"right_weights":q,"left_means":left_means,"right_means":right_means,
            "density_TV_midpoint":integral,"integral_error_bound":quadrature_error,"tail_upper":tail,
            "density_TV_upper":tv_upper,"weight_TV":weight_tv,"conditional_Gaussian_TV":component_tv,"C_N":local_constant,"covariance":1.}),
            vec![
                upper("positive_covariance", label, -1., 0.),
                upper("means_inside_tail_region", label, right_means[1], 2.),
            ],
            vec![
                upper("mixture_full_TV", label, tv_upper, mixture_rhs),
                upper(
                    "mixture_local_Lipschitz",
                    label,
                    mixture_rhs,
                    local_constant * input,
                ),
            ],
            "Explicit affine Gaussian mixture with logistic weights and fixed covariance. Full density TV has a derivative-controlled integral enclosure plus tail bound. The computed C_N is specific to this fixture and is not transferred to the canonical engine.",
        )?;
    }
    Ok(())
}

fn inline_algebra(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let tokens = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    let formulas = catalog
        .iter()
        .filter(|c| c["chapter"] == 1)
        .flat_map(|c| {
            c["quantitative_expressions"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter(|e| e["source_label"].is_null())
        .filter_map(|e| e["formula"].as_str())
        .collect::<Vec<_>>();
    for n in [1_usize, 2, 4, 16, 64, 256] {
        let a = (0..n)
            .map(|i| 0.3 - i as f64 / n as f64)
            .collect::<Vec<_>>();
        let b = (0..n)
            .map(|i| -0.1 + 2. * i as f64 / n as f64)
            .collect::<Vec<_>>();
        let c = (0..n)
            .map(|i| if i % 2 == 0 { 0.4 } else { -0.5 })
            .collect::<Vec<_>>();
        let squared = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let two = a.iter().zip(&b).map(|(a, b)| a + b).collect::<Vec<_>>();
        let three = two.iter().zip(&c).map(|(a, b)| a + b).collect::<Vec<_>>();
        let mean = a.iter().sum::<f64>() / n as f64;
        let absmean = a.iter().map(|x| x.abs()).sum::<f64>() / n as f64;
        let second = squared(&a) / n as f64;
        let inputs = json!({"N":n,"A":a,"B":b,"C":c,"normalized_mean":mean,
            "normalized_absolute_mean":absmean,"normalized_second_moment":second,
            "scope":"Elementary source-null proof clauses under an exact finite probability law; vector inequalities retain their stated total norm while probability expectations carry unit mass."});
        for formula in &formulas {
            let f = tokens(formula);
            let check = if f == "|\\mathbb{E}[X]|\\le\\mathbb{E}[|X|]" {
                Some(upper(
                    "absolute_expectation",
                    "lem-inequality-toolbox",
                    mean.abs(),
                    absmean,
                ))
            } else if f == "(a+b)^2\\le2a^2+2b^2" || f == "(a+b)^2\\leq2a^2+2b^2" {
                Some(upper(
                    "two_scalar_squares",
                    "lem-inequality-toolbox",
                    (mean + absmean).powi(2),
                    2. * mean * mean + 2. * absmean * absmean,
                ))
            } else if f == "(a+b+c)^2\\le3(a^2+b^2+c^2)" {
                Some(upper(
                    "three_scalar_squares",
                    "lem-inequality-toolbox",
                    (mean + absmean + second).powi(2),
                    3. * (mean * mean + absmean * absmean + second * second),
                ))
            } else if f == "\\|a+b\\|_2^2\\leq2(\\|a\\|_2^2+\\|b\\|_2^2)"
                || f == "\\|a+b\\|^2\\leq2(\\|a\\|^2+\\|b\\|^2)"
            {
                Some(upper(
                    "two_vector_squares",
                    "lem-inequality-toolbox",
                    squared(&two),
                    2. * (squared(&a) + squared(&b)),
                ))
            } else if f == "\\|A+B+C\\|_2^2\\le3(\\|A\\|_2^2+\\|B\\|_2^2+\\|C\\|_2^2)"
                || f == "\\|\\Delta\\mathbf{z}\\|_2^2\\leq3(\\|\\Delta_{\\text{direct}}\\|_2^2+\\cdots)"
            {
                Some(upper(
                    "three_vector_squares",
                    "lem-inequality-toolbox",
                    squared(&three),
                    3. * (squared(&a) + squared(&b) + squared(&c)),
                ))
            } else if f == "E[X]\\leq√E[X^2]" {
                Some(upper(
                    "finite_probability_Cauchy",
                    "lem-inequality-toolbox",
                    mean,
                    second.sqrt(),
                ))
            } else if f == "\\sqrt{a+b}\\le\\sqrt{a}+\\sqrt{b}" {
                Some(upper(
                    "square_root_subadditivity_inline",
                    "lem-inequality-toolbox",
                    (absmean + second).sqrt(),
                    absmean.sqrt() + second.sqrt(),
                ))
            } else if f == "(\\sqrt{a}+\\sqrt{b})^2=a+b+2\\sqrt{ab}\\gea+b" {
                None
            } else if f == "\\mathbb{E}[f(X)]\\lef(\\mathbb{E}[X])" {
                Some(upper(
                    "concave_sqrt_Jensen_inline",
                    "lem-inequality-toolbox",
                    a.iter().map(|x| x.abs().sqrt()).sum::<f64>() / n as f64,
                    absmean.sqrt(),
                ))
            } else if f == "f(a+b)\\lef(a)+f(b)" {
                Some(upper(
                    "concave_zero_origin_subadditivity",
                    "lem-subadditivity-power",
                    (absmean + second).powf(0.4),
                    absmean.powf(0.4) + second.powf(0.4),
                ))
            } else if f == "|d(a,b)-d(c,d)|\\led(a,c)+d(b,d)" {
                Some(upper(
                    "metric_reverse_triangle",
                    "lem-inequality-toolbox",
                    ((a[0] - b[0]).abs() - (c[0] - absmean).abs()).abs(),
                    (a[0] - c[0]).abs() + (b[0] - absmean).abs(),
                ))
            } else {
                continue;
            };
            let checks = if let Some(check) = check {
                vec![check]
            } else {
                vec![
                    upper(
                        "sqrt_square_expansion_identity",
                        "lem-inequality-toolbox",
                        ((absmean.sqrt() + second.sqrt()).powi(2)
                            - (absmean + second + 2. * (absmean * second).sqrt()))
                        .abs(),
                        2e-12,
                    ),
                    upper(
                        "sqrt_square_dominates_sum",
                        "lem-inequality-toolbox",
                        absmean + second,
                        (absmean.sqrt() + second.sqrt()).powi(2),
                    ),
                ]
            };
            suite.evidence.push(EstimateEvidence {
                chapter: 1,
                source_labels: vec!["lem-inequality-toolbox".into()],
                source_formula: (*formula).into(),
                inputs: inputs.clone(),
                hypothesis_checks: vec![],
                checks,
                scope: inputs["scope"].as_str().unwrap().into(),
            });
        }
        for alpha in [0.1_f64, 0.5, 1.] {
            let distances = a.iter().map(|x| x.abs()).collect::<Vec<_>>();
            let lhs = distances.iter().map(|x| x.powf(2. * alpha)).sum::<f64>() / n as f64;
            let rhs = (distances.iter().map(|x| x * x).sum::<f64>() / n as f64).powf(alpha);
            for formula in &formulas {
                if tokens(formula)
                    == "N^{-1}\\sum_id_i^{2\\alpha_B}\\le(N^{-1}\\sum_id_i^2)^{\\alpha_B}"
                {
                    suite.evidence.push(EstimateEvidence {
                        chapter: 1,
                        source_labels: vec!["lem-inequality-toolbox".into()],
                        source_formula: (*formula).into(),
                        inputs: json!({"N":n,"distances":distances,"alpha_B":alpha}),
                        hypothesis_checks: vec![upper(
                            "Holder_alpha_at_most_one",
                            "lem-inequality-toolbox",
                            alpha,
                            1.,
                        )],
                        checks: vec![upper(
                            "normalized_distance_Jensen",
                            "lem-inequality-toolbox",
                            lhs,
                            rhs,
                        )],
                        scope: "Actual unit-mass distance-power Jensen, independent of N.".into(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn gaussian_normalization(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let panels = 8192;
    let extent = 10_f64;
    let step = 2. * extent / panels as f64;
    let density = |x: f64| (-x * x / 2.).exp() / std::f64::consts::TAU.sqrt();
    let mut integral = density(-extent) + density(extent);
    for i in 1..panels {
        integral += if i % 2 == 0 { 2. } else { 4. } * density(-extent + i as f64 * step);
    }
    integral *= step / 3.;
    let tail = 2. * density(extent) / extent;
    for m in [1_i32, 2, 4, 16] {
        for sigma in [1e-4_f64, 0.1, 1., 10.] {
            let factor = (std::f64::consts::TAU * sigma * sigma).powf(m as f64 / 2.);
            let measured = factor * integral.powi(m);
            record(
                suite,
                catalog,
                "def-ambient-euclidean",
                r"\int \exp(-\|y\|_2^2/(2\sigma^2))",
                json!({"dimension":m,"sigma":sigma,
            "standard_gaussian_one_dimensional_integral":integral,"tail_upper":tail,"panels":panels,
            "unnormalized_product_integral":measured,"analytic_normalizer":factor}),
                vec![upper("positive_sigma", "def-ambient-euclidean", -sigma, 0.)],
                vec![upper(
                    "Gaussian_product_normalization_relative_error",
                    "def-ambient-euclidean",
                    ((measured - factor) / factor).abs(),
                    m as f64 * tail + 2e-10,
                )],
                "Tensor-product Gaussian integral with explicit one-dimensional Mills tail envelope, recorded relative error and physical scale; no finite box is substituted for the full Gaussian law.",
            )?;
        }
    }
    Ok(())
}

fn holder_inline_contracts(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let expressions = catalog
        .iter()
        .filter(|c| c["chapter"] == 1)
        .flat_map(|c| {
            c["quantitative_expressions"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .collect::<Vec<_>>();
    for v in [0_f64, 1e-12, 0.03, 0.9, 1., 1.1, 100.] {
        let coefficients = [0.3_f64, 1., 2.];
        let exponents = [0.1_f64, 0.5, 1.];
        let sum = coefficients.iter().sum::<f64>();
        let inputs = json!({"V":v,"A":coefficients,"p":exponents,"A_sum":sum,"p_max":1.});
        for e in &expressions {
            let formula = e["formula"].as_str().unwrap();
            let f = formula
                .chars()
                .filter(|x| !x.is_whitespace())
                .collect::<String>();
            let check = if f == "0\\leV\\le1" && v <= 1. {
                Some(upper(
                    "unit_interval_Holder_case",
                    "lem-sub-unify-holder-terms",
                    v,
                    1.,
                ))
            } else if f == "V\\ge1" && v >= 1. {
                Some(upper(
                    "large_Holder_case",
                    "lem-sub-unify-holder-terms",
                    1.,
                    v,
                ))
            } else if f == "V\\ge0" {
                Some(upper(
                    "nonnegative_Holder_error",
                    "lem-sub-unify-holder-terms",
                    -v,
                    0.,
                ))
            } else if f == "V^{p_k}\\le1" && v <= 1. {
                Some(upper(
                    "individual_small_power",
                    "lem-sub-unify-holder-terms",
                    exponents.iter().map(|p| v.powf(*p)).fold(0., f64::max),
                    1.,
                ))
            } else if f == "V^{p_k}\\leV^{p_{\\max}}" && v >= 1. {
                Some(upper(
                    "individual_large_power",
                    "lem-sub-unify-holder-terms",
                    exponents.iter().map(|p| v.powf(*p)).fold(0., f64::max),
                    v,
                ))
            } else if f == "A_\\Sigma:=\\sum_kA_k" {
                Some(upper(
                    "coefficient_sum_alias",
                    "lem-sub-unify-holder-terms",
                    (sum - coefficients.iter().sum::<f64>()).abs(),
                    0.,
                ))
            } else if f == "p_{\\max}:=\\max_kp_k" {
                Some(upper(
                    "maximum_exponent_alias",
                    "lem-sub-unify-holder-terms",
                    (exponents.iter().copied().fold(0., f64::max) - 1.).abs(),
                    0.,
                ))
            } else {
                None
            };
            if let Some(check) = check {
                suite.evidence.push(EstimateEvidence{chapter:1,source_labels:vec![e["source_label"].as_str().unwrap_or("lem-sub-unify-holder-terms").into()],source_formula:formula.into(),inputs:inputs.clone(),
                    hypothesis_checks:vec![],checks:vec![check],scope:"Individual finite Holder proof branch, with nonnegative coefficients and its actual maximum exponent. Unit and large-error branches retain their matching domains.".into()});
            }
        }
    }
    // The source explicitly rules out a global upper bound with a smaller
    // exponent. Retain an actual violating point and its logical polarity.
    let v = 1e18_f64;
    let q = 0.5;
    let c = 1000.;
    let k = 1000.;
    let lhs = v;
    let rhs = c * v.powf(q) + k;
    record(
        suite,
        catalog,
        "lem-sub-unify-holder-terms",
        r"\sum_k A_k V^{p_k} \le C\,V^{q}+K",
        json!({"V":v,"A":[1.],"p":[1.],"p_max":1.,"q":q,"C":c,"K":k,"lhs":lhs,"rhs":rhs,
            "logical_polarity":"negated uniform bound; explicit counterexample at smaller q"}),
        vec![upper(
            "smaller_exponent",
            "lem-sub-unify-holder-terms",
            q,
            1.,
        )],
        vec![upper(
            "smaller_exponent_global_bound_is_violated",
            "lem-sub-unify-holder-terms",
            rhs - lhs,
            -1.,
        )],
        "The paragraph states that no such uniform inequality can hold for q<p_max. This record checks an explicit violating point and does not claim the quoted inequality is true.",
    )?;
    Ok(())
}

fn remaining_power_clauses(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let expressions = catalog
        .iter()
        .filter(|c| c["chapter"] == 1)
        .flat_map(|c| {
            c["quantitative_expressions"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .collect::<Vec<_>>();
    for alpha in [0.1_f64, 0.5, 1.] {
        let x = [0_f64, 0.2, 0.8, 1.4];
        let p = [0.1_f64, 0.2, 0.3, 0.4];
        let mean = x.iter().zip(p).map(|(x, p)| x * p).sum::<f64>();
        let power_mean = x.iter().zip(p).map(|(x, p)| x.powf(alpha) * p).sum::<f64>();
        let a = [0.3_f64, 1., 2.];
        let powers = [0.1_f64, 0.5, 1.];
        for e in &expressions {
            let formula = e["formula"].as_str().unwrap();
            let f = formula
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>();
            let checks = if f == "\\alpha_B\\in(0,1]" || f == "\\alpha\\in(0,1]" {
                vec![
                    upper(
                        "fractional_exponent_upper",
                        "lem-subadditivity-power",
                        alpha,
                        1.,
                    ),
                    upper(
                        "fractional_exponent_strict_positive",
                        "lem-subadditivity-power",
                        -alpha,
                        -0.05,
                    ),
                ]
            } else if f == "\\sum_ip_i=1" {
                vec![upper(
                    "unit_mass_weights",
                    "lem-inequality-toolbox",
                    (p.iter().sum::<f64>() - 1.).abs(),
                    1e-14,
                )]
            } else if f == "f(x)=x^{\\alpha_B}" {
                vec![upper(
                    "actual_fractional_function_Jensen",
                    "lem-inequality-toolbox",
                    power_mean,
                    mean.powf(alpha),
                )]
            } else if f == "f(0)=0" {
                vec![upper(
                    "fractional_function_at_zero",
                    "lem-subadditivity-power",
                    0_f64.powf(alpha),
                    0.,
                )]
            } else if f == "m=2" {
                vec![upper(
                    "two_term_power_subadditivity",
                    "lem-subadditivity-power",
                    (x[1] + x[2]).powf(alpha),
                    x[1].powf(alpha) + x[2].powf(alpha),
                )]
            } else if f == "X=\\Delta_{\\text{pos,clone}}^2" {
                vec![
                    upper(
                        "declared_nonnegative_displacement_square",
                        "lem-inequality-toolbox",
                        -(0.2_f64.powi(2) + 0.4_f64.powi(2)),
                        0.,
                    ),
                    upper(
                        "independent_pair_displacement_reconstruction",
                        "lem-inequality-toolbox",
                        ((0.2_f64 - 0.).powi(2) + (1.4_f64 - 1.).powi(2)
                            - (0.2_f64.powi(2) + 0.4_f64.powi(2)))
                        .abs(),
                        1e-14,
                    ),
                ]
            } else if f == "\\mathbb{R}_{\\ge0}" {
                x.iter()
                    .map(|x| {
                        upper(
                            "nonnegative_power_domain",
                            "lem-subadditivity-power",
                            -*x,
                            0.,
                        )
                    })
                    .collect()
            } else if f == "A_\\Sigma:=\\sum_{k=1}^MA_k" {
                vec![upper(
                    "complete_Holder_coefficient_sum",
                    "lem-sub-unify-holder-terms",
                    (a.iter().sum::<f64>() - 3.3).abs(),
                    1e-14,
                )]
            } else if f == "\\{A_k\\}_{k=1}^M\\subset[0,\\infty)" {
                a.iter()
                    .map(|a| {
                        upper(
                            "nonnegative_Holder_coefficient_domain",
                            "lem-sub-unify-holder-terms",
                            -*a,
                            0.,
                        )
                    })
                    .collect()
            } else if f == "\\{p_k\\}_{k=1}^M\\subset(0,1]" {
                powers
                    .iter()
                    .flat_map(|p| {
                        [
                            upper(
                                "positive_Holder_power",
                                "lem-sub-unify-holder-terms",
                                -*p,
                                -0.05,
                            ),
                            upper(
                                "unit_interval_Holder_power",
                                "lem-sub-unify-holder-terms",
                                *p,
                                1.,
                            ),
                        ]
                    })
                    .collect()
            } else if f == "V\\in[0,1]" {
                vec![
                    upper(
                        "recorded_small_error_case",
                        "lem-sub-unify-holder-terms",
                        0.8,
                        1.,
                    ),
                    upper(
                        "recorded_small_error_lower",
                        "lem-sub-unify-holder-terms",
                        -0.8,
                        0.,
                    ),
                ]
            } else if f == "q<p_{\\max}" {
                vec![upper(
                    "strictly_smaller_growth_power",
                    "lem-sub-unify-holder-terms",
                    0.5,
                    1. - 0.1,
                )]
            } else {
                continue;
            };
            suite.evidence.push(EstimateEvidence {chapter:1,source_labels:vec![e["source_label"].as_str().unwrap_or("lem-subadditivity-power").into()],source_formula:formula.into(),inputs:json!({"alpha":alpha,"x":x,"probability_weights":p,"X_mean":mean,"fractional_X_mean":power_mean,"Holder_coefficients":a,"Holder_powers":powers,"A_sum":3.3,"two_term_indices":[1,2],"entering_positions":[0.,1.],"compared_proposal_positions":[0.2,1.4],"declared_squared_position_displacement":0.2_f64.powi(2)+0.4_f64.powi(2),"small_V":0.8,"q":0.5,"p_max":1.}),hypothesis_checks:vec![],checks,scope:"Exact individually quoted finite power/weight/domain clauses, with normalized expectation weights and the stated branch hypotheses. X is the auxiliary squared positional displacement; this clause does not replace the normalized primary swarm error.".into()});
        }
    }
    Ok(())
}
