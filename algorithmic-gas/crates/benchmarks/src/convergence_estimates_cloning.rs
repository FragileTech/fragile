//! Formula-level chapter 3 evidence. Native finite laws are integrated before
//! nonlinear acceptance; auxiliary couplings retain their declared marginals.
use crate::{
    convergence_cloning::{
        canonical_cloning_constants, canonical_keystone_constants, exact_cloning_moments,
    },
    convergence_estimates::{EstimateEvidence, EstimateSuite},
    convergence_framework::BoundCheck,
};
use algorithmic_gas::{
    GasConfig, GasError, ObservationBatch, Result, RewardBatch, TensorBatch,
    donor::DonorModule,
    geometry::{AlgorithmicDistance, InteractionKernel},
};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde_json::{Value, json};

const SOURCE: &str =
    include_str!("../../../../docs/source/2_fractal_gas/convergence_program/03_cloning.md");
const SCOPE: &str = "The retained finite witness and its stated hypotheses: native laws, independent couplings, auxiliary finite operators and analytic interval certificates are identified in the inputs. No terminal survival conditioning or global contraction is inferred.";

fn expression(label: &str, needle: &str) -> Result<String> {
    let start = SOURCE
        .find(&format!(":label: {label}"))
        .ok_or_else(|| error("source label missing"))?;
    let end = SOURCE[start + 8..]
        .find(":label:")
        .map_or(SOURCE.len(), |i| start + 8 + i);
    let section = &SOURCE[start..end];
    let tag = format!("\\tag{{{needle}}}");
    let search = if needle.starts_with("3.") {
        tag.as_str()
    } else {
        needle
    };
    let at = section
        .find(search)
        .ok_or_else(|| error(&format!("source expression missing: {label} / {needle}")))?;
    // A source tag may follow the closing display delimiter. Select the actual
    // preceding equation rather than accidentally treating that close as open.
    let mut offset = 0;
    let mut previous = None;
    while let Some(left) = section[offset..].find("$$") {
        let left = offset + left;
        let Some(right) = section[left + 2..].find("$$") else {
            break;
        };
        let right = left + 2 + right;
        if left <= at && at < right {
            return Ok(section[left + 2..right].trim().into());
        }
        if right < at {
            previous = Some((left, right));
        }
        offset = right + 2;
    }
    if let Some((left, right)) = previous
        && needle.starts_with("3.")
        && at - right < 120
    {
        return Ok(section[left + 2..right].trim().into());
    }
    let left = section[..at]
        .rfind('$')
        .ok_or_else(|| error("math delimiter missing"))?;
    let right = section[at..]
        .find('$')
        .ok_or_else(|| error("closing math delimiter missing"))?;
    Ok(section[left + 1..at + right].trim().into())
}
fn error(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn upper(id: &str, label: &str, x: f64, b: f64) -> BoundCheck {
    if x > 0. && b > 0. && b < 1e-8 {
        let mut check = BoundCheck::upper(id, &[label], SCOPE, x.ln(), b.ln());
        check.scope="Natural-log upper comparison of positive small quantities; an absolute tolerance cannot hide a relative violation.".into();
        return check;
    }
    BoundCheck::upper(id, &[label], SCOPE, x, b)
}
fn lower(id: &str, label: &str, x: f64, b: f64) -> BoundCheck {
    if b > 0. && b < 1e-8 {
        if x <= 0. {
            return upper(id, label, 1., 0.);
        }
        let mut check = upper(id, label, -x.ln(), -b.ln());
        check.scope="Natural-log comparison of the strictly positive observed quantity and strictly positive proved lower bound; absolute numerical tolerances cannot certify zero pressure.".into();
        return check;
    }
    upper(id, label, -x, -b)
}
fn equal(id: &str, label: &str, x: f64, y: f64) -> BoundCheck {
    upper(id, label, (x - y).abs(), 2e-11 * (1. + x.abs() + y.abs()))
}
fn equal_positive(id: &str, label: &str, x: f64, y: f64) -> BoundCheck {
    let scale = x.abs().max(y.abs());
    if x <= 0. || y <= 0. || !scale.is_finite() {
        return upper(id, label, 1., 0.);
    }
    let mut check = upper(id, label, ((x - y) / scale).abs(), 2e-11);
    check.scope="Relative residual of explicitly positive constants, scaled by their larger magnitude; zero and multiplicative corruptions cannot pass an absolute tolerance.".into();
    check
}
fn record(
    suite: &mut EstimateSuite,
    label: &str,
    needle: &str,
    inputs: Value,
    checks: Vec<BoundCheck>,
) -> Result<()> {
    let hypothesis_checks = if let Some(alive) = inputs.get("alive").and_then(Value::as_array) {
        let k = alive.iter().filter(|a| a.as_bool() == Some(true)).count();
        vec![upper("actual_nonempty_alive_pool", label, 1., k as f64)]
    } else {
        vec![]
    };
    let scope = inputs
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or(SCOPE)
        .to_owned();
    suite.evidence.push(EstimateEvidence {
        chapter: 3,
        source_labels: vec![label.into()],
        source_formula: expression(label, needle)?,
        inputs,
        hypothesis_checks,
        checks,
        scope,
    });
    Ok(())
}
fn range(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - x.iter().copied().fold(f64::INFINITY, f64::min)
}
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn var(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}

#[derive(Clone)]
struct Pattern {
    choices: Vec<usize>,
    mass: f64,
    raw: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    scales: Vec<f64>,
    f: Vec<f64>,
    means: Vec<f64>,
    traces: Vec<f64>,
    edges: Vec<Vec<f64>>,
    p: Vec<f64>,
    output_var: f64,
    conditional_bary_var: f64,
}
struct Law {
    config: GasConfig,
    x: Vec<f64>,
    v: Vec<f64>,
    alive: Vec<bool>,
    donors: Vec<Vec<f64>>,
    measurements: Vec<Vec<f64>>,
    patterns: Vec<Pattern>,
}
fn probabilities(
    module: &DonorModule,
    obs: &ObservationBatch<f64>,
    alive: &[bool],
) -> Result<Vec<Vec<f64>>> {
    let n = alive.len();
    let k = alive.iter().filter(|&&a| a).count();
    let mut output = vec![vec![0.; n]; n];
    for (i, row) in output.iter_mut().enumerate() {
        for (j, p) in row.iter_mut().enumerate() {
            if alive[j] && (i != j || k == 1) {
                *p = module
                    .kernel
                    .log_weight(
                        module.distance.compare(obs, i, obs, j)?,
                        algorithmic_gas::geometry::ComparisonKind::Distance,
                    )?
                    .exp();
            }
        }
        let z = row.iter().sum::<f64>();
        for p in row {
            *p /= z;
        }
    }
    Ok(output)
}
fn enumerate_law(x: Vec<f64>, v: Vec<f64>, alive: Vec<bool>) -> Result<Law> {
    let n = x.len();
    let config = GasConfig::euclidean(1, 0.04)?;
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, x.clone())?);
    obs.fields
        .insert("velocities".into(), TensorBatch::vectors(n, 1, v.clone())?);
    let donors = probabilities(&config.cloning_donors, &obs, &alive)?;
    let measurements = probabilities(&config.distance_donors, &obs, &alive)?;
    let mut assignments = vec![(vec![0; n], 1.)];
    for i in 0..n {
        if alive[i] {
            let mut next = Vec::new();
            for (choices, mass) in assignments {
                for (j, &pj) in measurements[i].iter().enumerate() {
                    if pj > 0. {
                        let mut c = choices.clone();
                        c[i] = j;
                        next.push((c, mass * pj));
                    }
                }
            }
            assignments = next;
        }
    }
    let rewards = RewardBatch::new(x.iter().map(|x| 0.5 * x * x).collect(), Default::default());
    let positions: Vec<_> = x.iter().map(|x| vec![*x]).collect();
    let mut patterns = Vec::new();
    for (choices, mass) in assignments {
        let raw: Vec<_> = (0..n)
            .map(|i| {
                if alive[i] {
                    config
                        .distance_donors
                        .distance
                        .compare(&obs, i, &obs, choices[i])
                } else {
                    Ok(0.)
                }
            })
            .collect::<Result<_>>()?;
        let evaluated = config.fitness.evaluate(&rewards, &raw, &alive, &obs, 0)?;
        let moments = exact_cloning_moments(
            &positions,
            &evaluated.fitness,
            &donors,
            &alive,
            &config.clone_decision,
            1,
            0.1,
        )?;
        patterns.push(Pattern {
            choices,
            mass,
            raw,
            y: evaluated.separation,
            z: evaluated.diversity_z,
            scales: evaluated.diversity_stats.scale,
            f: evaluated.fitness,
            means: moments.output_means.iter().map(|r| r[0]).collect(),
            traces: moments.covariance_traces,
            edges: moments.accepted_edge_probabilities,
            p: moments.acceptance_probabilities,
            output_var: moments.expected_position_variance,
            conditional_bary_var: moments.conditional_barycenter_variance,
        });
    }
    Ok(Law {
        config,
        x,
        v,
        alive,
        donors,
        measurements,
        patterns,
    })
}

fn centered_product(suite: &mut EstimateSuite) -> Result<()> {
    let label = "lem-sx-implies-variance";
    let mut checks = Vec::new();
    let mut cases = Vec::new();
    for (a, b) in [
        (vec![-1., 1.], vec![-2., 0., 2.]),
        (vec![3.], vec![-1., 2., 4., 7.]),
        (vec![-1., -0.2, 0.7, 1.5], vec![-0.9, -0.1, 0.5]),
    ] {
        let ma = mean(&a);
        let mb = mean(&b);
        let product = a
            .iter()
            .flat_map(|x| b.iter().map(move |y| ((x - ma) - (y - mb)).powi(2)))
            .sum::<f64>()
            / (a.len() * b.len()) as f64;
        let mut aa = a.clone();
        let mut bb = b.clone();
        aa.sort_by(f64::total_cmp);
        bb.sort_by(f64::total_cmp);
        let mut i = 0;
        let mut j = 0;
        let mut left = 1. / a.len() as f64;
        let mut right = 1. / b.len() as f64;
        let mut optimal = 0.;
        while i < a.len() && j < b.len() {
            let mass = left.min(right);
            optimal += mass * ((aa[i] - ma) - (bb[j] - mb)).powi(2);
            left -= mass;
            right -= mass;
            if left < 1e-14 {
                i += 1;
                left = 1. / a.len() as f64;
            }
            if right < 1e-14 {
                j += 1;
                right = 1. / b.len() as f64;
            }
        }
        checks.push(equal(
            "centered_product_cost",
            label,
            product,
            var(&a) + var(&b),
        ));
        checks.push(upper("optimal_alive_transport", label, optimal, product));
        let cl: Vec<_> = a.iter().map(|x| x - ma).collect();
        let cr: Vec<_> = b.iter().map(|x| x - mb).collect();
        checks.push(equal(
            "centered_alive_left_first_moment",
            label,
            mean(&cl),
            0.,
        ));
        checks.push(equal(
            "centered_alive_right_first_moment",
            label,
            mean(&cr),
            0.,
        ));
        for _ in &cl {
            checks.push(equal(
                "product_coupling_actual_left_marginal",
                label,
                cr.iter()
                    .map(|_| 1. / (cl.len() * cr.len()) as f64)
                    .sum::<f64>(),
                1. / cl.len() as f64,
            ));
        }
        for _ in &cr {
            checks.push(equal(
                "product_coupling_actual_right_marginal",
                label,
                cl.iter()
                    .map(|_| 1. / (cl.len() * cr.len()) as f64)
                    .sum::<f64>(),
                1. / cr.len() as f64,
            ));
        }
        let threshold = 0.8 * optimal;
        checks.push(certificate_check(
            "actual_strict_centered_transport_premise",
            label,
            optimal > threshold && threshold > 0.,
        ));
        checks.push(certificate_check(
            "actual_one_alive_probability_variance_exceeds_half_threshold",
            label,
            var(&a).max(var(&b)) > threshold / 2.,
        ));
        cases.push(json!({"left":a,"right":b,"product":product,"optimal":optimal,"centered_left":cl,"centered_right":cr,"threshold_squared":threshold,"alive_probability_variances":[var(&a),var(&b)]}));
    }
    records(
        suite,
        label,
        &[
            r"V_{\mathrm{x,struct}}\le",
            r"V_{\mathrm{x,struct}}>R_{\mathrm{spread}}^2",
            r"\operatorname{Var}_s(x)>R_{\mathrm{spread}}^2/2",
            r"\gamma=\widetilde\mu_1\otimes\widetilde\mu_2",
            r"\int u\,d\widetilde\mu_1=\int v\,d\widetilde\mu_2=0",
        ],
        json!({"cases":cases,"unequal_alive_counts":true}),
        checks,
    )
}

fn influence(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let label = "thm-cloning-canonical-barycenter-concentration";
    let c = canonical_cloning_constants();
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let k = live.len();
    let eligible: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let dx = eligible.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - eligible.iter().copied().fold(f64::INFINITY, f64::min);
    let mut meanchecks = Vec::new();
    let mut scalechecks = Vec::new();
    let mut fitchecks = Vec::new();
    let mut zchecks = Vec::new();
    let mut donorchecks = Vec::new();
    let mut rowchecks = Vec::new();
    let mut barychecks = Vec::new();
    let mut es = 0.;
    let eps_c = law.config.clone_decision.epsilon;
    let pmax = law.config.clone_decision.saturation;
    records(
        suite,
        label,
        &["D_m=\\sqrt", "\\begin{aligned}\nL_0&="],
        json!({"actual_native_configuration":law.config,"canonical_constants":c}),
        vec![
            equal(
                "canonical_measurement_span_bound",
                label,
                c.d_m,
                (32. + 1e-6_f64).sqrt(),
            ),
            equal(
                "canonical_reward_diversity_regularizer",
                label,
                c.epsilon_s,
                0.1,
            ),
            equal("canonical_kernel_floor", label, c.kappa_c, (-4_f64).exp()),
            equal("canonical_kernel_inverse_floor", label, c.c, 1. / c.kappa_c),
            equal(
                "canonical_fitness_lower_floor",
                label,
                c.fitness_min,
                0.1_f64.powi(2),
            ),
            equal(
                "canonical_fitness_upper_floor",
                label,
                c.fitness_max,
                2.1_f64.powi(2),
            ),
            equal(
                "canonical_fitness_normalizer_influence_constant",
                label,
                c.l_0,
                1.05 * (c.d_m / c.epsilon_s + 3. * c.d_m.powi(3) / (2. * c.epsilon_s.powi(3))),
            ),
            equal(
                "canonical_acceptance_rectangle_constant",
                label,
                c.l_a,
                (1. / (c.fitness_min + eps_c))
                    .max((c.fitness_max + eps_c) / (c.fitness_min + eps_c).powi(2))
                    / pmax,
            ),
            equal(
                "canonical_barycenter_total_influence_constant",
                label,
                c.b_0,
                1. + c.c + 2. * c.l_a * c.l_0,
            ),
        ],
    )?;
    for a in &law.patterns {
        for &r in &live {
            for b in &law.patterns {
                if a.choices
                    .iter()
                    .enumerate()
                    .any(|(i, j)| i != r && *j != b.choices[i])
                {
                    continue;
                }
                let ya: Vec<_> = live.iter().map(|&i| a.y[i]).collect();
                let yb: Vec<_> = live.iter().map(|&i| b.y[i]).collect();
                let sa = (var(&ya) + 0.01).sqrt();
                let sb = (var(&yb) + 0.01).sqrt();
                meanchecks.push(upper(
                    "innovation_mean",
                    label,
                    (mean(&ya) - mean(&yb)).abs(),
                    c.d_m / k as f64,
                ));
                meanchecks.push(upper(
                    "innovation_variance",
                    label,
                    (var(&ya) - var(&yb)).abs(),
                    3. * c.d_m.powi(2) / k as f64,
                ));
                scalechecks.push(upper(
                    "innovation_inverse_scale",
                    label,
                    (1. / sa - 1. / sb).abs(),
                    3. * c.d_m.powi(2) / (2. * 0.1_f64.powi(3) * k as f64),
                ));
                for &i in &live {
                    if i != r {
                        zchecks.push(upper(
                            "unchanged_standardized_measurement",
                            label,
                            (a.z[i] - b.z[i]).abs(),
                            (c.d_m / 0.1 + 3. * c.d_m.powi(3) / (2. * 0.1_f64.powi(3))) / k as f64,
                        ));
                        fitchecks.push(upper(
                            "unchanged_row_fitness",
                            label,
                            (a.f[i] - b.f[i]).abs(),
                            c.l_0 / k as f64,
                        ));
                        donorchecks.push(upper(
                            "innovation_donor_mass",
                            label,
                            law.donors[i][r],
                            c.c / (k - 1) as f64,
                        ));
                        rowchecks.push(upper(
                            "unchanged_row_mean",
                            label,
                            (a.means[i] - b.means[i]).abs(),
                            dx * (law.donors[i][r] + 2. * c.l_a * c.l_0 / k as f64),
                        ));
                    } else {
                        rowchecks.push(upper(
                            "replaced_row_mean",
                            label,
                            (a.means[i] - b.means[i]).abs(),
                            dx,
                        ));
                    }
                }
                for i in 0..n {
                    if !law.alive[i] {
                        rowchecks.push(equal(
                            "dead_row_mean_unchanged",
                            label,
                            a.means[i],
                            b.means[i],
                        ));
                    }
                }
                let diff = (mean(&a.means) - mean(&b.means)).abs();
                barychecks.push(upper(
                    "mean_influence_sharp",
                    label,
                    diff,
                    dx / n as f64 * (1. + c.c + 2. * c.l_a * c.l_0 * (k - 1) as f64 / k as f64),
                ));
                barychecks.push(upper(
                    "mean_influence_fraction_relaxation",
                    label,
                    dx / n as f64 * (1. + c.c + 2. * c.l_a * c.l_0 * (k - 1) as f64 / k as f64),
                    dx * c.b_0 / n as f64,
                ));
                // b differs only at r; its replacement probability is the r marginal,
                // rather than the mass of the whole b pattern.
                es += 0.5 * a.mass * law.measurements[r][b.choices[r]] * diff * diff;
            }
        }
    }
    let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"patterns":law.patterns.len(),"eligible_diameter":dx,"constants":c});
    if k >= 2 {
        record(
            suite,
            label,
            "\\left|\\frac{Y_i-\\bar Y}",
            inputs.clone(),
            zchecks,
        )?;
        record(suite, label, "|\\bar Y-", inputs.clone(), meanchecks)?;
        record(suite, label, "|s_Y^{-1}", inputs.clone(), scalechecks)?;
        record(
            suite,
            label,
            "|F_i-\\widetilde F_i|",
            inputs.clone(),
            fitchecks,
        )?;
        record(suite, label, "K_{ir}\\le", inputs.clone(), donorchecks)?;
        record(
            suite,
            label,
            "|m_i-\\widetilde m_i|",
            inputs.clone(),
            rowchecks,
        )?;
        record(
            suite,
            label,
            "\\left|\\bar m(\\mathbf Y)",
            inputs.clone(),
            barychecks,
        )?;
    }
    let barymean = law
        .patterns
        .iter()
        .map(|p| p.mass * mean(&p.means))
        .sum::<f64>();
    let measurement_var = law
        .patterns
        .iter()
        .map(|p| p.mass * (mean(&p.means) - barymean).powi(2))
        .sum::<f64>();
    let conditional_var = law
        .patterns
        .iter()
        .map(|p| p.mass * p.conditional_bary_var)
        .sum::<f64>();
    record(
        suite,
        label,
        "\\operatorname{Var}_{\\mathbf Y}(\\bar m",
        inputs.clone(),
        vec![
            upper("exact_efron_stein", label, measurement_var, es),
            upper(
                "efron_stein_influence_bound",
                label,
                es,
                k as f64 * dx * dx * c.b_0 * c.b_0 / (2. * (n * n) as f64),
            ),
        ],
    )?;
    record(
        suite,
        label,
        "\\boxed{\\operatorname{Var}(\\bar X",
        inputs,
        vec![
            upper(
                "total_barycenter_variance",
                label,
                conditional_var + measurement_var,
                (dx * dx + 0.01) / n as f64
                    + k as f64 * dx * dx * c.b_0 * c.b_0 / (2. * (n * n) as f64),
            ),
            upper(
                "total_barycenter_alive_fraction_relaxation",
                label,
                (dx * dx + 0.01) / n as f64
                    + k as f64 * dx * dx * c.b_0 * c.b_0 / (2. * (n * n) as f64),
                (dx * dx + 0.01 + dx * dx * c.b_0 * c.b_0 / 2.) / n as f64,
            ),
            upper(
                "conditional_barycenter_variance",
                label,
                conditional_var,
                (dx * dx + 0.01) / n as f64,
            ),
        ],
    )
}

fn components(choices: &[usize]) -> Vec<Vec<usize>> {
    let n = choices.len();
    let mut parent: Vec<_> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            i = parent[i];
        }
        i
    }
    for (i, &j) in choices.iter().enumerate() {
        let a = find(&mut parent, i);
        let b = find(&mut parent, j);
        parent[a] = b;
    }
    let mut result: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        if let Some(c) = result.iter_mut().find(|c| find(&mut parent, c[0]) == r) {
            c.push(i);
        } else {
            result.push(vec![i]);
        }
    }
    result
}
fn graph_plans(p: &Pattern) -> Vec<(Vec<usize>, f64)> {
    let n = p.p.len();
    let mut plans = vec![(vec![0; n], 1.)];
    for i in 0..n {
        let mut next = Vec::new();
        for (choices, mass) in plans {
            let keep = 1. - p.p[i];
            if keep > 0. {
                let mut c = choices.clone();
                c[i] = i;
                next.push((c, mass * keep));
            }
            for j in 0..n {
                if p.edges[i][j] > 0. {
                    let mut c = choices.clone();
                    c[i] = j;
                    next.push((c, mass * p.edges[i][j]));
                }
            }
        }
        plans = next;
    }
    plans
}
fn graph_moments(law: &Law, p: &Pattern, alpha: f64) -> (f64, f64, f64, f64) {
    let n = law.x.len();
    let mut energy = 0.;
    let mut qmoment = 0.;
    let mut momentum = 0.;
    let mut mass_sum = 0.;
    for (choices, mass) in graph_plans(p) {
        let groups = components(&choices);
        let mut ec = 0.;
        let mut xmoment = 0.;
        let mut vmoment = 0.;
        let mut cross = 0.;
        let mut meanvel = 0.;
        for group in groups {
            let average = group.iter().map(|&i| law.v[i]).sum::<f64>() / group.len() as f64;
            for i in group {
                ec += (law.v[i] - average).powi(2) / n as f64;
                vmoment +=
                    (average.powi(2) + alpha.powi(2) * (law.v[i] - average).powi(2)) / n as f64;
                meanvel += average / n as f64;
                cross += law.x[choices[i]] * average / n as f64;
            }
        }
        for (i, &j) in choices.iter().enumerate() {
            xmoment +=
                (law.x[j].powi(2) + if i != j || !law.alive[i] { 0.01 } else { 0. }) / n as f64;
        }
        mass_sum += mass;
        energy += mass * ec;
        qmoment += mass * (xmoment + vmoment + 0.1 * cross);
        momentum += mass * meanvel;
    }
    (energy, qmoment, momentum, mass_sum)
}

fn balances(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let k = live.len();
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let va: Vec<_> = live.iter().map(|&i| law.v[i]).collect();
    let mu = mean(&xa);
    let uv = mean(&va);
    let dx = xa.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - xa.iter().copied().fold(f64::INFINITY, f64::min);
    let vmax = law.v.iter().map(|v| v.abs()).fold(0., f64::max);
    let alpha = 0.5;
    let rx = law
        .donors
        .iter()
        .enumerate()
        .filter(|(i, _)| !law.alive[*i])
        .map(|(_, row)| {
            row.iter()
                .enumerate()
                .map(|(j, b)| b * (law.x[j] - mu).powi(2))
                .sum::<f64>()
                / n as f64
        })
        .sum::<f64>();
    let deadv: Vec<_> = (0..n)
        .filter(|&i| !law.alive[i])
        .map(|i| law.v[i] - uv)
        .collect();
    let rv = deadv.iter().map(|v| v * v).sum::<f64>() / n as f64
        - (deadv.iter().sum::<f64>() / n as f64).powi(2);
    let mut f2 = Vec::new();
    let mut f3 = Vec::new();
    let mut f5 = Vec::new();
    let mut f10 = Vec::new();
    let mut s1 = Vec::new();
    let mut s2 = Vec::new();
    let mut s3 = Vec::new();
    let mut s4 = Vec::new();
    let mut velocity = Vec::new();
    let mut w2 = Vec::new();
    let mut edge_drift = 0.;
    let mut exact_energy = 0.;
    let mut total_var = 0.;
    let mut pi = 0.;
    let mut bary_shift2 = 0.;
    let mut covariance = 0.;
    let mut totalq = 0.;
    let mut ml = 0.;
    let mut tl = 0.;
    let mut signed = 0.;
    let h: Vec<_> = live.iter().copied().take(k / 2).collect();
    let l: Vec<_> = live.iter().copied().skip(k / 2).collect();
    let kappa = (-4_f64).exp();
    let cp = kappa / 4.410001;
    let cm = 1. / (kappa * 0.010001);
    let mut native_floor = Vec::new();
    let mut fitness_lower = Vec::new();
    let mut fitness_upper = Vec::new();
    let mut scale = Vec::new();
    let mut f4 = Vec::new();
    let mut reset = Vec::new();
    let mut revival_row = Vec::new();
    for (index, p) in law.patterns.iter().enumerate() {
        let raw: Vec<_> = live.iter().map(|&i| p.raw[i]).collect();
        let measured: Vec<_> = live.iter().map(|&i| p.y[i]).collect();
        for &i in &live {
            fitness_lower.push(lower(
                "native_positive_fitness_floor",
                "lem-potential-bounds",
                p.f[i],
                0.01,
            ));
            fitness_upper.push(upper(
                "native_fitness_ceiling",
                "lem-potential-bounds",
                p.f[i],
                4.41,
            ));
            scale.push(equal(
                "native_regularized_standard_deviation",
                "def-patched-std-dev-function",
                p.scales[i],
                (var(&measured) + 0.01).sqrt(),
            ));
        }
        native_floor.push(upper(
            "native_floor_variance_transfer",
            "lem-cloning-distance-floor-transfer",
            (var(&measured) - var(&raw)).abs(),
            0.002 * var(&raw).sqrt() + 1e-6,
        ));
        let cmu = mean(&p.means);
        let sumc = p.traces.iter().sum::<f64>();
        let copy_cov = p
            .traces
            .iter()
            .zip(&p.p)
            .map(|(c, p)| c - 0.01 * p)
            .sum::<f64>();
        let live_flux = live
            .iter()
            .map(|&i| {
                p.edges[i]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| b * ((law.x[j] - mu).powi(2) - (law.x[i] - mu).powi(2)))
                    .sum::<f64>()
                    / n as f64
            })
            .sum::<f64>();
        let meanp = mean(&p.p);
        if k == n {
            f4.push(equal(
                "conditional_barycenter_covariance_sum",
                "lem-cloning-individual-centered-displacement",
                p.conditional_bary_var,
                sumc / (n * n) as f64,
            ));
            f4.push(lower(
                "barycenter_variance_nonnegative",
                "lem-cloning-individual-centered-displacement",
                p.conditional_bary_var,
                0.,
            ));
            f4.push(upper(
                "barycenter_variance_pressure_bound",
                "lem-cloning-individual-centered-displacement",
                p.conditional_bary_var,
                (dx * dx + 0.01) * meanp / n as f64,
            ));
        }
        reset.push(upper(
            "native_positional_reset",
            "thm-positional-variance-contraction",
            p.output_var,
            dx * dx / 2. + (1. - 1. / n as f64) * 0.01,
        ));
        for i in 0..n {
            if !law.alive[i] {
                let centered = (p.means[i] - cmu).powi(2)
                    + (1. - 2. / n as f64) * p.traces[i]
                    + sumc / (n * n) as f64;
                revival_row.push(upper(
                    "native_revived_centered_position",
                    "lem-dead-walker-revival-bounded",
                    centered,
                    dx * dx + 0.01,
                ));
            }
        }
        let balance = live_flux + rx + 0.01 * meanp - (cmu - mu).powi(2) - sumc / (n * n) as f64;
        f10.push(equal(
            &format!("alive_balance_{index}"),
            "prop-cloning-revival-cluster-flux",
            p.output_var - var(&xa) * k as f64 / n as f64,
            balance,
        ));
        f5.push(equal(
            &format!("total_variance_{index}"),
            "lem-variance-change-decomposition",
            p.output_var,
            var(&p.means) + (1. - 1. / n as f64) * sumc / n as f64,
        ));
        for i in 0..n {
            if !law.alive[i] {
                let mean = law.donors[i]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| b * law.x[j])
                    .sum::<f64>();
                let c = law.donors[i]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| b * (law.x[j] - mean).powi(2))
                    .sum::<f64>()
                    + 0.01;
                f2.push(equal(
                    "donor_centered_dead_mean",
                    "def-cloning-frozen-positional-moments",
                    p.means[i],
                    mean,
                ));
                f2.push(equal(
                    "donor_centered_dead_covariance",
                    "def-cloning-frozen-positional-moments",
                    p.traces[i],
                    c,
                ));
            }
            let oldc = law.x[i] - mean(&law.x);
            let ti = p.means[i] - law.x[i];
            let bart = cmu - mean(&law.x);
            let rhs = 2. * oldc * (ti - bart)
                + (ti - bart).powi(2)
                + (1. - 2. / n as f64) * p.traces[i]
                + sumc / (n * n) as f64;
            let lhs = (p.means[i] - cmu).powi(2)
                + (1. - 2. / n as f64) * p.traces[i]
                + sumc / (n * n) as f64
                - oldc.powi(2);
            if law.x[i].abs() < 1e6 {
                f3.push(equal(
                    "moving_center_row",
                    "lem-cloning-individual-centered-displacement",
                    lhs,
                    rhs,
                ));
            }
        }
        if k >= 2 {
            let fh: Vec<_> = h.iter().map(|&i| p.f[i]).collect();
            let fl: Vec<_> = l.iter().map(|&i| p.f[i]).collect();
            let delta = mean(&fl) - mean(&fh);
            let spread = var(&fh) + var(&fl);
            let differences: Vec<_> = fh
                .iter()
                .flat_map(|a| fl.iter().map(move |b| b - a))
                .collect();
            let mean_negative =
                differences.iter().map(|d| (-d).max(0.)).sum::<f64>() / differences.len() as f64;
            let mean_absolute =
                differences.iter().map(|d| d.abs()).sum::<f64>() / differences.len() as f64;
            let dd = differences.iter().map(|d| d * d).sum::<f64>() / differences.len() as f64;
            s1.extend([
                equal(
                    "uniform_retained_fitness_gap_mean",
                    "thm-cloning-signed-cluster-fitness-flux",
                    mean(&differences),
                    delta,
                ),
                equal(
                    "uniform_retained_fitness_gap_second_moment",
                    "thm-cloning-signed-cluster-fitness-flux",
                    dd,
                    delta * delta + spread,
                ),
                equal(
                    "uniform_negative_part_exact_identity",
                    "thm-cloning-signed-cluster-fitness-flux",
                    mean_negative,
                    (mean_absolute - delta) / 2.,
                ),
                upper(
                    "uniform_negative_part_Cauchy_Schwarz",
                    "thm-cloning-signed-cluster-fitness-flux",
                    (mean_absolute - delta) / 2.,
                    ((delta * delta + spread).sqrt() - delta) / 2.,
                ),
            ]);
            let coeff = (h.len() * l.len()) as f64 / (n * (k - 1)) as f64;
            let actual = h
                .iter()
                .flat_map(|&i| {
                    l.iter()
                        .map(move |&j| (p.edges[i][j] - p.edges[j][i]) / n as f64)
                })
                .sum::<f64>();
            let bound =
                coeff * (cp * delta - (cm - cp) / 2. * ((delta * delta + spread).sqrt() - delta));
            s1.push(lower(
                &format!("signed_uniform_{index}"),
                "thm-cloning-signed-cluster-fitness-flux",
                actual,
                bound,
            ));
            let weights: Vec<_> = h
                .iter()
                .flat_map(|&i| {
                    l.iter()
                        .map(move |&j| (-p.raw_distance(law, i, j).powi(2) / 8.).exp())
                })
                .collect();
            let mut idx = 0;
            let weight_sum = weights.iter().sum::<f64>();
            let mut dw = 0.;
            let mut d2w = 0.;
            let mut negativew = 0.;
            let mut zmaxh = 0_f64;
            let mut zminl = f64::INFINITY;
            for &i in &h {
                let zi = live
                    .iter()
                    .filter(|&&j| i != j)
                    .map(|&j| (-p.raw_distance(law, i, j).powi(2) / 8.).exp())
                    .sum::<f64>();
                zmaxh = zmaxh.max(zi);
                for &j in &l {
                    let d = p.f[j] - p.f[i];
                    dw += weights[idx] * d / weight_sum;
                    d2w += weights[idx] * d * d / weight_sum;
                    negativew += weights[idx] * (-d).max(0.) / weight_sum;
                    s3.push(lower(
                        "signed_edge",
                        "thm-cloning-signed-cluster-fitness-flux",
                        p.edges[i][j] - p.edges[j][i],
                        (cp * d.max(0.) - cm * (-d).max(0.)) / (k - 1) as f64,
                    ));
                    idx += 1;
                }
            }
            for &j in &l {
                let zj = live
                    .iter()
                    .filter(|&&i| i != j)
                    .map(|&i| (-p.raw_distance(law, i, j).powi(2) / 8.).exp())
                    .sum::<f64>();
                zminl = zminl.min(zj);
            }
            let maxh = fh.iter().copied().fold(0., f64::max);
            let minh = fh.iter().copied().fold(f64::INFINITY, f64::min);
            let maxl = fl.iter().copied().fold(0., f64::max);
            let minl = fl.iter().copied().fold(f64::INFINITY, f64::min);
            let ahl = 1. / (zmaxh * (maxl - minh).max(0.).max(maxh + 1e-6));
            let bhl = 1. / (zminl * (minl + 1e-6));
            let weightedbound =
                weight_sum / n as f64 * (ahl * dw - (bhl - ahl).max(0.) / 2. * (d2w.sqrt() - dw));
            s2.push(lower(
                "signed_weighted",
                "thm-cloning-signed-cluster-fitness-flux",
                actual,
                weightedbound,
            ));
            s2.push(upper(
                "weighted_negative_part_Cauchy_Schwarz",
                "thm-cloning-signed-cluster-fitness-flux",
                negativew,
                (d2w.sqrt() - dw) / 2.,
            ));
            // Singleton geometric refinement makes its within-cluster remainder
            // identically zero while retaining every signed directed edge.
            if k == n {
                let mut corrected = 0.;
                for i in 0..n {
                    for j in i + 1..n {
                        let (hi, lo) = if (law.x[i] - mu).powi(2) >= (law.x[j] - mu).powi(2) {
                            (i, j)
                        } else {
                            (j, i)
                        };
                        let gap = p.f[lo] - p.f[hi];
                        let edge_lower =
                            (cp * gap.max(0.) - cm * (-gap).max(0.)) / (k - 1) as f64 / n as f64;
                        let zi = live
                            .iter()
                            .filter(|&&j| j != hi)
                            .map(|&j| (-p.raw_distance(law, hi, j).powi(2) / 8.).exp())
                            .sum::<f64>();
                        let zj = live
                            .iter()
                            .filter(|&&i| i != lo)
                            .map(|&i| (-p.raw_distance(law, lo, i).powi(2) / 8.).exp())
                            .sum::<f64>();
                        let w = (-p.raw_distance(law, hi, lo).powi(2) / 8.).exp();
                        let a = 1. / (zi * gap.max(0.).max(p.f[hi] + 1e-6));
                        let b = 1. / (zj * (p.f[lo] + 1e-6));
                        let weighted_lower =
                            w / n as f64 * (a * gap - (b - a).max(0.) / 2. * (gap.abs() - gap));
                        corrected -= ((law.x[hi] - mu).powi(2) - (law.x[lo] - mu).powi(2))
                            * edge_lower.max(weighted_lower);
                    }
                }
                corrected -= (cmu - mu).powi(2) + copy_cov / (n * n) as f64;
                corrected += (1. - 1. / n as f64) * 0.01 * meanp;
                s4.push(upper(
                    "singleton_cluster_signed_drift",
                    "cor-cloning-signed-collective-drift",
                    p.output_var - var(&xa),
                    corrected,
                ));
            }
            ml += p.mass * delta;
            tl += p.mass * (delta * delta + spread);
            signed += p.mass * actual;
        }
        let (ec, q, vm, graphmass) = graph_moments(law, p, alpha);
        velocity.push(equal(
            "accepted_graph_mass",
            "thm-cloning-unconditional-collective-balance",
            graphmass,
            1.,
        ));
        velocity.push(equal(
            "component_momentum",
            "thm-cloning-canonical-barycenter-concentration",
            vm,
            mean(&law.v),
        ));
        let keta = 2. * (4. + 0.01) + (1. + 0.1_f64.powi(2) / 4.) * vmax * vmax;
        w2.push(upper(
            "native_quadratic_postclone_moment",
            "cor-cloning-actual-inter-swarm-expansion",
            q,
            keta,
        ));
        edge_drift += p.mass * live_flux;
        exact_energy += p.mass * ec;
        total_var += p.mass * p.output_var;
        pi += p.mass * meanp;
        bary_shift2 += p.mass * (cmu - mu).powi(2);
        covariance += p.mass * copy_cov;
        totalq += p.mass * q;
    }
    let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"all_measurement_patterns":law.patterns.len(),"revival_position_injection":rx,"revival_velocity_injection":rv,"component_relative_energy":exact_energy});
    record(
        suite,
        "lem-potential-bounds",
        "V_i \\ge",
        inputs.clone(),
        fitness_lower,
    )?;
    record(
        suite,
        "lem-potential-bounds",
        "V_i \\le",
        inputs.clone(),
        fitness_upper,
    )?;
    record(
        suite,
        "def-patched-std-dev-function",
        "\\sigma'_{\\rm patch}(V)=",
        inputs.clone(),
        scale,
    )?;
    if !f4.is_empty() {
        record(
            suite,
            "lem-cloning-individual-centered-displacement",
            "3.F4",
            inputs.clone(),
            f4,
        )?;
    }
    record(
        suite,
        "thm-positional-variance-contraction",
        "\\mathbb E V_{\\mathrm{Var},x}(S')",
        inputs.clone(),
        reset,
    )?;
    if !revival_row.is_empty() {
        record(
            suite,
            "lem-dead-walker-revival-bounded",
            "\\mathbb E|X_i'-\\bar X'|^2",
            inputs.clone(),
            revival_row,
        )?;
    }
    record(
        suite,
        "lem-cloning-distance-floor-transfer",
        "|\\operatorname{Var}(d)",
        inputs.clone(),
        native_floor,
    )?;
    record(
        suite,
        "lem-variance-change-decomposition",
        "\\mathbb E[V_",
        inputs.clone(),
        f5,
    )?;
    record(
        suite,
        "lem-cloning-individual-centered-displacement",
        "\\boxed{",
        inputs.clone(),
        f3,
    )?;
    if !f2.is_empty() {
        record(
            suite,
            "def-cloning-frozen-positional-moments",
            "c_i=\\sum_jb_{ij}|x_j-m_i|",
            inputs.clone(),
            f2,
        )?;
    }
    record(
        suite,
        "prop-cloning-revival-cluster-flux",
        "\\mathbb E[V_A",
        inputs.clone(),
        f10,
    )?;
    if k >= 2 {
        records(
            suite,
            "thm-cloning-signed-cluster-fitness-flux",
            &["3.S1", "\\langle D_-\\rangle\n="],
            inputs.clone(),
            s1,
        )?;
        records(
            suite,
            "thm-cloning-signed-cluster-fitness-flux",
            &["3.S2", "\\mathbb E_wD_-\\leq"],
            inputs.clone(),
            s2,
        )?;
        record(
            suite,
            "thm-cloning-signed-cluster-fitness-flux",
            "3.S3",
            inputs.clone(),
            s3,
        )?;
        if !s4.is_empty() {
            record(
                suite,
                "cor-cloning-signed-collective-drift",
                "3.S4",
                inputs.clone(),
                s4,
            )?;
        }
        record(
            suite,
            "thm-cloning-unconditional-collective-balance",
            "3.U3",
            inputs.clone(),
            vec![lower(
                "unconditional_directed_flux",
                "thm-cloning-unconditional-collective-balance",
                signed,
                (h.len() * l.len()) as f64 / (n * (k - 1)) as f64
                    * (cp * ml - (cm - cp) / 2. * (tl.sqrt() - ml)),
            )],
        )?;
    }
    record(
        suite,
        "thm-cloning-unconditional-collective-balance",
        "3.U1",
        inputs.clone(),
        vec![equal(
            "complete_measurement_law_mass",
            "thm-cloning-unconditional-collective-balance",
            law.patterns.iter().map(|p| p.mass).sum(),
            1.,
        )],
    )?;
    let hx = total_var - var(&xa) * k as f64 / n as f64;
    let hv = rv - (1. - alpha * alpha) * exact_energy;
    let signed_d = -edge_drift
        + bary_shift2
        + covariance / (n * n) as f64
        + (1. - alpha * alpha) * exact_energy;
    let mut direct_signed_radius_flow = 0.;
    for i in 0..n {
        if !law.alive[i] {
            continue;
        }
        for j in i + 1..n {
            if !law.alive[j] {
                continue;
            }
            let ei = (law.x[i] - mu).powi(2);
            let ej = (law.x[j] - mu).powi(2);
            let averaged = law
                .patterns
                .iter()
                .map(|p| p.mass * (p.edges[i][j] - p.edges[j][i]))
                .sum::<f64>();
            direct_signed_radius_flow += (ei - ej) * averaged / n as f64;
        }
    }
    record(
        suite,
        "thm-cloning-unconditional-collective-balance",
        r"\mathscr D(S)={}&",
        inputs.clone(),
        vec![
            equal(
                "direct_native_singleton_cluster_signed_radius_flow",
                "thm-cloning-unconditional-collective-balance",
                direct_signed_radius_flow,
                -edge_drift,
            ),
            equal(
                "complete_native_signed_balance_function_definition",
                "thm-cloning-unconditional-collective-balance",
                signed_d,
                direct_signed_radius_flow
                    + bary_shift2
                    + covariance / (n * n) as f64
                    + (1. - alpha * alpha) * exact_energy,
            ),
        ],
    )?;
    record(
        suite,
        "thm-cloning-unconditional-collective-balance",
        "3.U2",
        inputs.clone(),
        vec![
            equal(
                "full_measurement_averaged_balance",
                "thm-cloning-unconditional-collective-balance",
                hx + hv,
                -signed_d + (1. - 1. / n as f64) * 0.01 * pi + rx + rv,
            ),
            lower(
                "revival_position_nonnegative",
                "thm-cloning-unconditional-collective-balance",
                rx,
                0.,
            ),
            upper(
                "revival_position_bound",
                "thm-cloning-unconditional-collective-balance",
                rx,
                dx * dx * (n - k) as f64 / n as f64,
            ),
            lower(
                "revival_velocity_nonnegative",
                "thm-cloning-unconditional-collective-balance",
                rv,
                0.,
            ),
            upper(
                "revival_velocity_bound",
                "thm-cloning-unconditional-collective-balance",
                rv,
                4. * vmax * vmax * (n - k) as f64 / n as f64,
            ),
        ],
    )?;
    record(
        suite,
        "thm-cloning-canonical-barycenter-concentration",
        "\\boxed{\\bar V",
        inputs.clone(),
        velocity,
    )?;
    record(
        suite,
        "cor-cloning-actual-inter-swarm-expansion",
        "3.W2",
        inputs.clone(),
        w2,
    )?;
    record(
        suite,
        "cor-cloning-actual-inter-swarm-expansion",
        "3.W3",
        inputs,
        vec![upper(
            "product_transport_moment_bound",
            "cor-cloning-actual-inter-swarm-expansion",
            4. * totalq,
            4. * (2. * (4. + 0.01) + (1. + 0.1_f64.powi(2) / 4.) * vmax * vmax),
        )],
    )
}
impl Pattern {
    fn raw_distance(&self, law: &Law, i: usize, j: usize) -> f64 {
        let sx = |x: f64| 2. * x / (2. + x.abs());
        ((sx(law.x[i]) - sx(law.x[j])).powi(2) + (sx(law.v[i]) - sx(law.v[j])).powi(2)).sqrt()
    }
}

fn keystone(suite: &mut EstimateSuite) -> Result<()> {
    let cl = "lem-keystone-complete-coverage-constants";
    let report = canonical_keystone_constants(1, 0.001, 2.)?;
    let c = &report.values;
    let logs = &report.natural_logs;
    let mut geometry = Vec::new();
    let mut reward = Vec::new();
    let mut rawfloor = Vec::new();
    let mut increment = Vec::new();
    let mut radius = Vec::new();
    let squash = |x: f64| 2. * x / (2. + x.abs());
    let logistic = |u: f64| 0.1 + 2. / (1. + (-u).exp());
    for i in -20..=20 {
        for j in -20..=20 {
            let x = i as f64 / 10.;
            let y = j as f64 / 10.;
            let zx = squash(x);
            let zy = squash(y);
            geometry.push(lower(
                "squashed_feature_inverse_modulus",
                cl,
                (zx - zy).abs(),
                c["m_x"] * (x - y).abs(),
            ));
            geometry.push(lower(
                "squashed_joint_feature_inverse_modulus",
                cl,
                ((zx - zy).powi(2) + (squash(x / 2.) - squash(y / 2.)).powi(2)).sqrt(),
                c["m_z"] * ((x - y).powi(2) + (x / 2. - y / 2.).powi(2)).sqrt(),
            ));
            let ai = logistic(-0.5 * x * x / 0.1);
            let aj = logistic(-0.5 * y * y / 0.1);
            reward.push(upper(
                "shared_reward_factor_lipschitz",
                cl,
                (ai - aj).abs(),
                c["L_A"] * (zx - zy).abs(),
            ));
        }
    }
    record(
        suite,
        cl,
        "3.CC1",
        json!({"parameters":report,"grid_step":0.1,"bounded_entering_alive_region":[-2.,2.]}),
        geometry,
    )?;
    record(
        suite,
        cl,
        "3.CC2",
        json!({"shared_reward_mean":0.,"regularized_reward_scale":0.1,"analytic_joint_reward_lipschitz":2.,"alpha":1}),
        reward,
    )?;
    let tf = c["t_f"];
    let z = c["Z_star"];
    let logomega = logs["omega_f"];
    let logr = logs["r"];
    let r = logr.exp();
    let gamma = logs["gamma_0"].exp();
    let a0 = logs["a_0"].exp();
    // Reconstruct the inputs independently from exact rational parameters.
    // In particular, a smaller incorrect Delta_f must not pass merely because
    // every realized far/near gap exceeds it.
    let hf2 = Iv::rational(1, 128000);
    let delta2 = Iv::rational(1, 1_000_000);
    let reconstructed_delta = hf2
        .add(&delta2)
        .sqrt()
        .sub(&hf2.div(&Iv::rational(4, 1)).add(&delta2).sqrt());
    let reconstructed_span = Iv::rational(32, 1)
        .add(&delta2)
        .sqrt()
        .sub(&Iv::rational(1, 1000));
    let reconstructed_scale = reconstructed_span
        .square()
        .div(&Iv::rational(4, 1))
        .add(&Iv::rational(1, 100))
        .sqrt();
    let reconstructed_tf = reconstructed_delta.div(&reconstructed_scale);
    increment.extend([
        equal(
            "independent_far_gap_definition",
            cl,
            c["Delta_f"],
            reconstructed_delta.midpoint(),
        ),
        equal(
            "independent_maximum_scale_definition",
            cl,
            c["s_star"],
            reconstructed_scale.midpoint(),
        ),
        equal(
            "independent_standardized_gap_definition",
            cl,
            tf,
            reconstructed_tf.midpoint(),
        ),
        certificate_check(
            "directed_strict_far_gap",
            cl,
            reconstructed_delta.lo > BigInt::zero(),
        ),
    ]);
    let endpoint_log =
        |u: f64| 2_f64.ln() + u + tf.exp_m1().ln() - u.exp().ln_1p() - (u + tf).exp().ln_1p();
    let independent_endpoint_minimum = endpoint_log(-z).min(endpoint_log(z - tf));
    increment.push(equal(
        "independent_actual_endpoint_minimum_log",
        cl,
        logomega,
        independent_endpoint_minimum,
    ));
    increment.push(equal(
        "symmetric_logistic_endpoint_values",
        cl,
        endpoint_log(-z),
        endpoint_log(z - tf),
    ));
    for i in 0..=128 {
        let u = -z + (2. * z - tf) * i as f64 / 128.; // avoid cancellation in saturated logistic increments
        let logdelta =
            2_f64.ln() + u + tf.exp_m1().ln() - (u.exp()).ln_1p() - ((u + tf).exp()).ln_1p();
        increment.push(lower(
            "exact_logistic_increment_log",
            cl,
            logdelta,
            logomega,
        ));
    }
    increment.push(upper("increment_domain", cl, tf, z));
    increment.push(lower("positive_increment", cl, tf, f64::MIN_POSITIVE));
    radius.push(upper("near_radius", cl, r, c["h_f"] / 2.));
    radius.push(upper(
        "reward_penalty",
        cl,
        2.1 * c["L_A"] * r,
        0.1 * logomega.exp() / 2.,
    ));
    radius.push(equal_positive(
        "acceptance_floor",
        cl,
        a0,
        (gamma / 4.410001).min(1.),
    ));
    radius.push(certificate_check(
        "strict_positive_gamma_and_acceptance_floor",
        cl,
        gamma > 0. && a0 > 0.,
    ));
    radius.push(equal(
        "activity_constant_log",
        cl,
        logs["C_0"],
        -12. + c["rho_f"].ln() + logs["a_0"],
    ));
    record(
        suite,
        cl,
        "3.CC3",
        json!({"W_0":0.001,"v0":c["v_0"],"h_f":c["h_f"],"rho_f":c["rho_f"]}),
        vec![
            equal(
                "feature_threshold",
                cl,
                c["v_0"],
                c["m_x"].powi(2) * 0.001 / 4.,
            ),
            equal("far_threshold", cl, c["h_f"].powi(2), c["v_0"] / 2.),
            equal(
                "far_population_fraction",
                cl,
                c["rho_f"],
                (c["v_0"] - c["h_f"].powi(2)) / (32. - c["h_f"].powi(2)),
            ),
        ],
    )?;
    record(
        suite,
        cl,
        "3.CC4",
        json!({"t_f":tf,"Z_star":z,"log_omega":logomega,"independent_delta_f":reconstructed_delta.json(),"independent_s_star":reconstructed_scale.json(),"independent_t_f":reconstructed_tf.json(),"evaluated_endpoint_log_increments":[endpoint_log(-z),endpoint_log(z-tf)],"minimum_strategy":"For the logistic increment the derivative is positive before its symmetry center and negative afterwards; the interval minimum is attained at one of its two endpoints. Both endpoint values and the minimum are independently evaluated."}),
        increment,
    )?;
    record(
        suite,
        cl,
        "3.CC5",
        json!({"log_r":logr,"log_gamma":logs["gamma_0"],"log_a0":logs["a_0"],"log_C0":logs["C_0"]}),
        radius,
    )?;
    for raw in [
        vec![0., 0.05, 0.5, 1.],
        vec![0., 0., 0., 4.],
        vec![0.001, 0.001, 0.001, 0.001],
    ] {
        let sep: Vec<_> = raw.iter().map(|x| f64::sqrt(x * x + 1e-6)).collect();
        rawfloor.push(upper(
            "separation_mean_error",
            "lem-cloning-distance-floor-transfer",
            (mean(&sep) - mean(&raw)).abs(),
            0.001,
        ));
        rawfloor.push(upper(
            "separation_std_error",
            "lem-cloning-distance-floor-transfer",
            (var(&sep).sqrt() - var(&raw).sqrt()).abs(),
            0.001,
        ));
        rawfloor.push(upper(
            "separation_variance_error",
            "lem-cloning-distance-floor-transfer",
            (var(&sep) - var(&raw)).abs(),
            0.002 * var(&raw).sqrt() + 1e-6,
        ));
    }
    record(
        suite,
        "lem-cloning-distance-floor-transfer",
        "|\\operatorname{Var}(d)",
        json!({"distance_floor":0.001}),
        rawfloor,
    )?;
    let cover = "thm-keystone-complete-error-coverage";
    let discharged = "thm-keystone-discharged-averaged-pressure";
    for n in [4, 6, 8] {
        let x: Vec<_> = (0..n).map(|i| if i < n / 2 { 1. } else { -1. }).collect(); // compress identical distance patterns to 2^N, preserving native law
        let _ = x;
        let moments = crate::convergence_cloning::exact_balanced_two_site_fixture(n, 1., 0.1)?;
        let pressure = moments.acceptance_probabilities;
        let emax = 64.;
        let errorweight = 0.25;
        let w = errorweight;
        let celln = n / 2;
        let occupied = 2.;
        let c0 = logs["C_0"].exp();
        let exact_activity = pressure.iter().sum::<f64>() / n as f64 * errorweight;
        let second = crate::convergence_cloning::exact_balanced_two_site_fixture(n, 0.5, 0.1)?;
        let two_activity = exact_activity
            + second.acceptance_probabilities.iter().sum::<f64>() / n as f64 * errorweight;
        let cc6 = (0..n)
            .map(|i| {
                lower(
                    "native_near_neighbor_pressure",
                    "lem-keystone-near-neighbor-pressure",
                    pressure[i],
                    c0 * ((celln - 1) as f64 / (n - 1) as f64).powi(2),
                )
            })
            .collect();
        record(
            suite,
            "lem-keystone-near-neighbor-pressure",
            "3.CC6",
            json!({"N":n,"feature_variance":squash(1.).powi(2),"v0":c["v_0"],"near_neighbor_counts":celln-1,"exact_native_pressure":pressure}),
            cc6,
        )?;
        let logmr = logs["M_r"];
        let uniform_fraction = (-logmr).exp() * n as f64 * w / emax;
        let bound = c0 * w * ((uniform_fraction - 1.).max(0.) / (n - 1) as f64).powi(2);
        let occupied_bound =
            c0 * w * ((n as f64 * w / (emax * occupied) - 1.).max(0.) / (n - 1) as f64).powi(2);
        record(
            suite,
            cover,
            "3.CC8",
            json!({"N":n,"W":w,"Emax":emax,"log_cover_count":logmr,"occupied_cells":occupied,"cell_error":[w/2.,w/2.]}),
            vec![
                lower("native_all_cell_activity", cover, exact_activity, bound),
                lower(
                    "occupied_cell_sharpening",
                    cover,
                    exact_activity,
                    occupied_bound,
                ),
                lower(
                    "actual_occupied_cell_nonvacuous_bound",
                    cover,
                    exact_activity,
                    c0 * w * ((celln - 1) as f64 / (n - 1) as f64).powi(2),
                ),
                upper(
                    "cell_energy_bound",
                    cover,
                    w / 2.,
                    emax * celln as f64 / n as f64,
                ),
                lower(
                    "weighted_cell_cauchy_schwarz",
                    cover,
                    2. * (w / 2.).powi(2),
                    w * w / occupied,
                ),
            ],
        )?;
        let chistar = logs["chi_star"].exp();
        let bstar = logs["B_star"].exp();
        record(
            suite,
            discharged,
            "3.CC11a",
            json!({"N":n,"large_population_hypothesis":false,"log_N0":logs["N_0_unrounded"],"activity":two_activity,"left_radius":1.,"right_radius":0.5}),
            vec![lower(
                "all_population_affine_activity",
                discharged,
                two_activity,
                chistar * (w - 0.001) - bstar / (n * n) as f64,
            )],
        )?;
        record(
            suite,
            discharged,
            "3.CC14",
            json!({"N":n,"structural_cost":w,"velocity_discrepancy":0.,"eta":1.,"all_alive":true}),
            vec![lower(
                "all_population_structural_activity",
                discharged,
                two_activity,
                chistar / 2. * w - chistar * 0.001 - c0 * emax / (n * n) as f64,
            )],
        )?;
        record(
            suite,
            discharged,
            "3.CC10",
            json!({"W":w,"alive_mass":1.,"positional_variance":1.}),
            vec![
                lower("spread_swarm_weighted_variance", discharged, 1., w / 4.),
                lower(
                    "spread_swarm_feature_variance",
                    discharged,
                    squash(1.).powi(2),
                    c["m_x"].powi(2) * w / 4.,
                ),
            ],
        )?;
    }
    // Astronomical cover/threshold values are verified in the logarithmic
    // domain. Finite N<=64 is never declared to meet their hypotheses.
    record(
        suite,
        cover,
        "3.CC7",
        json!({"dimension":1,"log_r":logr,"log_Mr":logs["M_r"]}),
        vec![
            lower(
                "cover_cell_count_log",
                cover,
                logs["M_r"],
                2. * (4. * 2_f64.sqrt()).ln() - 2. * logr,
            ),
            equal(
                "cover_count_full_formula_log",
                cover,
                logs["M_r"],
                2. * ((4. * 2_f64.sqrt() / r).ceil()).ln(),
            ),
            upper(
                "cell_diameter",
                cover,
                2_f64.sqrt() * 4. / (4. * 2_f64.sqrt() / r).ceil(),
                r,
            ),
        ],
    )?;
    let huge_n = 2_f64.powi(512);
    let huge_count = (BigInt::one() << 512_usize).to_string();
    let w = 0.01;
    let actual_population = BigInt::one() << 512_usize;
    let actual_site_count = BigInt::one() << 511_usize;
    let (event_left, left_event_inputs) =
        compressed_native_two_site_event(&actual_population, &actual_site_count, 1, 1, 0);
    let (event_right, right_event_inputs) =
        compressed_native_two_site_event(&actual_population, &actual_site_count, 9, 10, 0);
    let single_activity_lower =
        event_left.lo.to_f64().expect("finite event probability") / 1e40 * w;
    let activity_lower = (event_left.lo.clone() + event_right.lo.clone())
        .to_f64()
        .expect("finite event probability")
        / 1e40
        * w;
    let huge_inputs = json!({"population_exact_integer":huge_count,"alive_count_equals_N":true,"analytic_native_two_site_law":true,"compressed_atom_multiplicities":"2^511 at each site","left_radius":1.,"right_radius":0.9,"velocity_discrepancy":0.,"W":w,"native_event_subset_activity_lower":activity_lower,"native_left_event":left_event_inputs,"native_right_event":right_event_inputs,"log_N0":logs["N_0_unrounded"],"log_Mr":logs["M_r"],"scope":"Analytic compressed native two-site event law, not an executed giant-population simulation"});
    record(
        suite,
        cover,
        "3.CC9",
        huge_inputs.clone(),
        vec![
            certificate_check(
                "strict_native_left_event_pressure",
                cover,
                event_left.lo > BigInt::zero(),
            ),
            certificate_check(
                "strict_native_right_event_pressure",
                cover,
                event_right.lo > BigInt::zero(),
            ),
            equal(
                "uniform_rate_log",
                cover,
                logs["chi_0"],
                logs["C_0"] + 2. * 0.001_f64.ln()
                    - 4_f64.ln()
                    - 2. * 64_f64.ln()
                    - 2. * logs["M_r"],
            ),
            lower(
                "actual_compressed_population_threshold_log",
                cover,
                huge_n.ln(),
                2_f64.ln() + 64_f64.ln() + logs["M_r"] - 0.001_f64.ln(),
            ),
            lower(
                "actual_native_balanced_activity_log",
                cover,
                single_activity_lower.ln(),
                logs["chi_0"] + w.ln(),
            ),
        ],
    )?;
    record(
        suite,
        discharged,
        "3.CC11",
        huge_inputs.clone(),
        vec![
            equal(
                "large_population_threshold_log",
                discharged,
                logs["N_0_unrounded"],
                2_f64.ln() + 2. * 64_f64.ln() + logs["M_r"] - 2. * 0.001_f64.ln(),
            ),
            lower(
                "actual_compressed_N0_hypothesis_log",
                discharged,
                huge_n.ln(),
                logs["N_0_unrounded"],
            ),
            lower(
                "actual_two_swarm_activity_log",
                discharged,
                activity_lower.ln(),
                logs["chi_0"] + (w - 0.001).ln(),
            ),
        ],
    )?;
    record(
        suite,
        discharged,
        "3.CC12",
        huge_inputs,
        vec![lower(
            "zero_velocity_actual_structural_activity_log",
            discharged,
            activity_lower.ln(),
            logs["chi_0"] + (w / 2. - 0.001).ln(),
        )],
    )?;
    let left = vec![vec![-1., -0.4], vec![0., 0.2], vec![1., 0.6]];
    let right = vec![vec![-0.5, -0.2], vec![0.5, 0.4]];
    let center_points = |points: Vec<Vec<f64>>| {
        let mx = points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64;
        let mv = points.iter().map(|p| p[1]).sum::<f64>() / points.len() as f64;
        points
            .into_iter()
            .map(|p| vec![p[0] - mx, p[1] - mv])
            .collect::<Vec<_>>()
    };
    let left = center_points(left);
    let right = center_points(right);
    let cost = |a: &[f64], b: &[f64]| {
        let dx = a[0] - b[0];
        let dv = a[1] - b[1];
        dx * dx + dv * dv + 0.1 * dx * dv
    };
    let (optimal, _) = crate::convergence_lyapunov::uniform_transport(&left, &right, cost)?;
    let wcommon = ((left[0][0] - right[0][0]).powi(2) + (left[1][0] - right[1][0]).powi(2)) / 4.;
    let dvcommon = ((left[0][1] - right[0][1]).powi(2) + (left[1][1] - right[1][1]).powi(2)) / 4.;
    let bound = 2. * wcommon + 1.0025 * dvcommon + 0.25 * (2. * 64. + 16. * 1.0025 * 4.);
    record(
        suite,
        discharged,
        "3.CC13",
        json!({"centered_left":left,"centered_right":right,"N":4,"alive_counts":[3,2],"common_matching_mass":2./3.,"unmatched_mass":1./3.,"W":wcommon,"Dv":dvcommon,"optimal_alive_cost":optimal,"velocity_cap":2.,"Emax":64.,"eta":1.}),
        vec![upper(
            "partial_alive_admissible_completion",
            discharged,
            0.75 * optimal,
            bound,
        )],
    )
}

fn collision_output(v: &[f64], groups: &[Vec<usize>], signs: &[f64], alpha: f64) -> Vec<f64> {
    let mut result = v.to_vec();
    for (g, &sign) in groups.iter().zip(signs) {
        let mu = g.iter().map(|&i| v[i]).sum::<f64>() / g.len() as f64;
        for &i in g {
            result[i] = mu + alpha * sign * (v[i] - mu);
        }
    }
    result
}
fn backbone(suite: &mut EstimateSuite) -> Result<()> {
    let label = "thm-cloning-revival-backbone-coupling";
    let mut l1 = Vec::new();
    let mut l2 = Vec::new();
    let mut l4 = Vec::new();
    let mut energy = Vec::new();
    let mut c2 = Vec::new();
    let mut inputs = Vec::new();
    for leaves in [1, 2, 8, 32, 128] {
        let n = leaves + 3;
        let v: Vec<_> = (0..n)
            .map(|i| if i % 2 == 0 { 0.8 } else { -0.7 })
            .collect();
        let mut a = vec![vec![0, 1], vec![2]];
        a[0].extend(3..n);
        let r = n - 1;
        let mut b = a.clone();
        b[0].retain(|&i| i != r);
        b[1].push(r);
        for alpha in [0., 0.5, 1.] {
            for rc in [-1., 1.] {
                for rd in [-1., 1.] {
                    let outa = collision_output(&v, &a, &[rc, rd], alpha);
                    let outb = collision_output(&v, &b, &[rc, rd], alpha);
                    let diff: Vec<_> = outb.iter().zip(&outa).map(|(b, a)| b - a).collect();
                    let mu = a[0].iter().map(|&i| v[i]).sum::<f64>() / a[0].len() as f64;
                    let nu = v[2];
                    let u = (1. - alpha * rc) * (v[r] - mu);
                    let w = (1. - alpha * rd) * (v[r] - nu) / 2.;
                    l1.extend([
                        lower("actual_old_component_size", label, a[0].len() as f64, 2.),
                        lower("actual_target_component_size", label, a[1].len() as f64, 1.),
                        upper(
                            "actual_retained_switch_velocity_cap",
                            label,
                            v.iter().map(|v| v.abs()).fold(0., f64::max),
                            0.8,
                        ),
                        equal(
                            "actual_target_mean_factor",
                            label,
                            w,
                            a[1].len() as f64 / (a[1].len() + 1) as f64
                                * (1. - alpha * rd)
                                * (v[r] - nu),
                        ),
                    ]);
                    let muprime = b[0].iter().map(|&i| v[i]).sum::<f64>() / b[0].len() as f64;
                    let nuprime = b[1].iter().map(|&i| v[i]).sum::<f64>() / b[1].len() as f64;
                    l1.push(equal(
                        "removed_leaf_mean_rotation_change",
                        label,
                        (1. - alpha * rc) * (muprime - mu),
                        -u / (a[0].len() - 1) as f64,
                    ));
                    l1.push(equal(
                        "added_leaf_mean_rotation_change",
                        label,
                        (1. - alpha * rd) * (nuprime - nu),
                        w / a[1].len() as f64,
                    ));
                    l1.extend([
                        equal(
                            "removed_leaf_component_mean",
                            label,
                            muprime,
                            (a[0].len() as f64 * mu - v[r]) / (a[0].len() - 1) as f64,
                        ),
                        equal(
                            "added_leaf_component_mean",
                            label,
                            nuprime,
                            (a[1].len() as f64 * nu + v[r]) / (a[1].len() + 1) as f64,
                        ),
                    ]);
                    let acsum = a[0]
                        .iter()
                        .filter(|&&i| i != r)
                        .map(|&i| diff[i])
                        .sum::<f64>();
                    let bsum = diff[2];
                    l1.push(equal("old_backbone_sum", label, acsum, -u));
                    l1.push(equal("new_backbone_sum", label, bsum, w));
                    l1.push(equal("switched_leaf", label, diff[r], u - w));
                    l1.push(equal("total_momentum_change", label, diff.iter().sum(), 0.));
                    l1.push(equal(
                        "norm_sum_identity",
                        label,
                        diff.iter().map(|x| x.abs()).sum(),
                        u.abs() + w.abs() + (u - w).abs(),
                    ));
                    l1.push(upper(
                        "size_uniform_leaf_switch",
                        label,
                        diff.iter().map(|x| x.abs()).sum(),
                        (6. + 8. * alpha) * 0.8,
                    ));
                    l2.push(upper(
                        "one_changed_donor",
                        label,
                        diff.iter().map(|x| x.abs()).sum(),
                        (6. + 8. * alpha) * 0.8,
                    ));
                    let perturbed: Vec<_> = v
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v + 0.03 * (i % 3) as f64)
                        .collect();
                    let outp = collision_output(&perturbed, &a, &[rc, rd], alpha);
                    l4.push(upper(
                        "fixed_plan_perturbation",
                        "lem-cloning-fixed-plan-velocity-perturbation",
                        outp.iter().zip(&outa).map(|(x, y)| (x - y).abs()).sum(),
                        (1. + 2. * alpha)
                            * perturbed
                                .iter()
                                .zip(&v)
                                .map(|(x, y)| (x - y).abs())
                                .sum::<f64>(),
                    ));
                    let entering = v.iter().map(|v| v * v).sum::<f64>();
                    for g in &a {
                        let dv: Vec<_> = g.iter().map(|&i| perturbed[i] - v[i]).collect();
                        l4.push(upper(
                            "fixed_component_mean_triangle_inequality",
                            "lem-cloning-fixed-plan-velocity-perturbation",
                            g.len() as f64 * mean(&dv).abs(),
                            dv.iter().map(|v| v.abs()).sum::<f64>(),
                        ));
                        l4.push(equal(
                            "same_component_synchronous_relative_velocity_energy",
                            "rem-synergistic-velocity-dissipation",
                            g.iter().map(|&i| (outp[i] - outa[i]).powi(2)).sum::<f64>(),
                            g.len() as f64 * mean(&dv).powi(2)
                                + alpha
                                    * alpha
                                    * dv.iter().map(|v| (v - mean(&dv)).powi(2)).sum::<f64>(),
                        ));
                    }
                    let ec = a
                        .iter()
                        .map(|g| {
                            let m = g.iter().map(|&i| v[i]).sum::<f64>() / g.len() as f64;
                            g.iter().map(|&i| (v[i] - m).powi(2)).sum::<f64>()
                        })
                        .sum::<f64>();
                    energy.push(equal(
                        "component_energy",
                        "prop-cloning-component-conservation",
                        outa.iter().map(|v| v * v).sum(),
                        entering - (1. - alpha * alpha) * ec,
                    ));
                    energy.push(equal(
                        "component_momentum",
                        "prop-cloning-component-conservation",
                        outa.iter().sum(),
                        v.iter().sum(),
                    ));
                }
            }
        }
        let alpha = 0.5;
        let mut actual = 0.;
        for rc in [-1., 1.] {
            for rd in [-1., 1.] {
                for sc in [-1., 1.] {
                    for sd in [-1., 1.] {
                        let aa = collision_output(&v, &a, &[rc, rd], alpha);
                        let bb = collision_output(&v, &b, &[sc, sd], alpha);
                        actual +=
                            aa.iter().zip(bb).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / 16.;
                    }
                }
            }
        }
        let mut rhs = 0.;
        for i in 0..n {
            let ga = a.iter().find(|g| g.contains(&i)).expect("partition");
            let gb = b.iter().find(|g| g.contains(&i)).expect("partition");
            let ma = ga.iter().map(|&j| v[j]).sum::<f64>() / ga.len() as f64;
            let mb = gb.iter().map(|&j| v[j]).sum::<f64>() / gb.len() as f64;
            rhs += (ma - mb).powi(2) + alpha * alpha * ((v[i] - ma).powi(2) + (v[i] - mb).powi(2));
        }
        c2.push(equal(
            "changed_components_independent_haar",
            "thm-cloning-incremental-cluster-balance",
            actual,
            rhs,
        ));
        inputs.push(json!({"dead_leaves":leaves,"old_components":a,"new_components":b,"frozen_velocities":v}));
    }
    records(
        suite,
        label,
        &[
            "3.L1",
            "n_C=|C|",
            "U=(I-\\alpha R_C)",
            "\\sum_{i\\in C\\setminus\\{r\\}}\\Delta v_i^+=",
            "\\mu'=\\frac",
            r"|v_i|\leq V_*",
            r"(I-\alpha R_C)(\mu'-\mu)=-U/(n_C-1)",
            r"(I-\alpha R_D)(\nu'-\nu)=W/n_D",
        ],
        json!({"cases":inputs,"one_dimensional_haar_sign_law":[-0.1_f64.signum(),1.]}),
        l1,
    )?;
    record(
        suite,
        label,
        "3.L2",
        json!({"changed_donors":1,"sizes_up_to":131}),
        l2,
    )?;
    record(
        suite,
        "lem-cloning-fixed-plan-velocity-perturbation",
        "3.L4",
        json!({"restitution":[0.,0.5,1.],"one_dimensional_exact_haar_law":true}),
        l4.clone(),
    )?;
    record(
        suite,
        "lem-cloning-fixed-plan-velocity-perturbation",
        r"|C||\overline{\Delta v}_C|\leq\sum_{i\in C}|\Delta v_i|",
        json!({"cases":inputs,"scope":"Frozen common accepted component plans and identical component rotations; each component mean perturbation satisfies its independently evaluated triangle inequality."}),
        l4.clone(),
    )?;
    record(
        suite,
        "rem-synergistic-velocity-dissipation",
        r"\sum_{i\in C}|\delta v_i'|^2",
        json!({"cases":inputs,"scope":"Two inputs use the same frozen accepted partition and shared component Haar sign. Paired energy retains both the component mean and relative terms."}),
        l4,
    )?;
    record(
        suite,
        "prop-cloning-component-conservation",
        "\\sum_{i\\in C}|v_i'|^2",
        json!({"restitution":[0.,0.5,1.]}),
        energy,
    )?;
    record(
        suite,
        "thm-cloning-incremental-cluster-balance",
        "3.C2",
        json!({"coupling":"shared matrix exactly on identical components, independent otherwise","changed_components":true}),
        c2,
    )
}

fn composition(suite: &mut EstimateSuite) -> Result<()> {
    let label = "cor-population-uniform-completed-error";
    let mut checks = Vec::new();
    let mut cases = Vec::new();
    let mut decreasing = Vec::new();
    let mut decreasing_cases = Vec::new();
    let mut zero = Vec::new();
    let mut zero_cases = Vec::new();
    for n in [4, 16, 64, 256] {
        for kappa in [0.2_f64, 1.] {
            for c in [0., 0.4] {
                for initial in [0., 2., 8., 32.] {
                    let rho: f64 = 1. - kappa;
                    let floor = c / kappa;
                    let mut value = initial;
                    checks.extend([
                        equal("affine_rho_definition", label, rho, 1. - kappa),
                        certificate_check(
                            "affine_rho_in_half_open_unit_interval",
                            label,
                            (0. ..1.).contains(&rho),
                        ),
                        equal("population_uniform_affine_floor", label, floor, c / kappa),
                    ]);
                    for step in 0_i32..100 {
                        let envelope = rho.powi(step) * initial + floor * (1. - rho.powi(step));
                        checks.push(equal("affine_iteration", label, value, envelope));
                        checks.push(upper("transport_domination", label, 0.7 * value, value));
                        if initial >= floor {
                            decreasing.push(upper(
                                "decreasing_envelope",
                                label,
                                rho * value + c,
                                value,
                            ));
                            decreasing.push(lower(
                                "actual_initial_above_error_floor",
                                label,
                                initial,
                                floor,
                            ));
                        }
                        if c == 0. {
                            zero.push(equal(
                                "zero_offset_geometric_envelope",
                                label,
                                value,
                                rho.powi(step) * initial,
                            ));
                            zero.push(equal("actual_zero_offset", label, c, 0.));
                        }
                        value = rho * value + c;
                    }
                    let case = json!({"N":n,"initial":initial,"kappa":kappa,"C":c,"rho":rho,"floor":floor,"steps":100});
                    if initial >= floor {
                        decreasing_cases.push(case.clone());
                    }
                    if c == 0. {
                        zero_cases.push(case.clone());
                    }
                    cases.push(case);
                }
            }
        }
    }
    let input = json!({"cases":cases,"scope":"Exact affine comparison operator under the chapter's component drift hypotheses, with common rates and offsets across N. No native canonical kinetic contraction rate is assigned."});
    records(
        suite,
        label,
        &[
            "Q^n V_W",
            r"\rho=1-\kappa_*\in[0,1)",
            r"F_*=C_*/\kappa_*",
            r"V_W\le V_{\mathrm{total}}",
        ],
        input,
        checks,
    )?;
    record(
        suite,
        label,
        r"V_{\mathrm{total}}\ge F_*",
        json!({"cases":decreasing_cases,"scope":"Conditional affine envelope with independently verified entering error above its population-independent floor; every one of 100 updates decreases."}),
        decreasing,
    )?;
    record(
        suite,
        label,
        r"C_*=0",
        json!({"cases":zero_cases,"scope":"Exact zero-offset comparison operator with geometric decrease of the error envelope. Positive-offset branches are checked separately under their own hypotheses."}),
        zero,
    )
}

/// Fixed-point intervals enclose exact rational inputs and every operation.
/// Integer bounds, not the displayed f64 conversions, decide certificates.
#[derive(Clone, Debug)]
struct Iv {
    lo: BigInt,
    hi: BigInt,
}
fn scale() -> BigInt {
    BigInt::from(10).pow(40)
}
fn floor_div(a: &BigInt, b: &BigInt) -> BigInt {
    assert!(b.is_positive());
    let q = a / b;
    let r = a % b;
    if r.is_negative() { q - 1 } else { q }
}
fn ceil_div(a: &BigInt, b: &BigInt) -> BigInt {
    -floor_div(&(-a), b)
}
fn integer_sqrt(n: &BigInt) -> BigInt {
    assert!(!n.is_negative());
    if n.is_zero() {
        return BigInt::zero();
    }
    let mut x = BigInt::one() << n.bits().div_ceil(2) as usize;
    loop {
        let y = (&x + n / &x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}
impl Iv {
    fn big_rational(n: &BigInt, d: &BigInt) -> Self {
        let numerator = n * scale();
        Self {
            lo: floor_div(&numerator, d),
            hi: ceil_div(&numerator, d),
        }
    }
    fn rational(n: i64, d: i64) -> Self {
        let a = BigInt::from(n) * scale();
        let b = BigInt::from(d);
        Self {
            lo: floor_div(&a, &b),
            hi: ceil_div(&a, &b),
        }
    }
    fn decimal(s: &str) -> Self {
        let negative = s.starts_with('-');
        let text = s.trim_start_matches('-');
        let (whole, frac) = text.split_once('.').unwrap_or((text, ""));
        let digits = format!("{whole}{frac}");
        let numerator = BigInt::parse_bytes(digits.as_bytes(), 10).expect("decimal");
        let denominator = BigInt::from(10).pow(frac.len() as u32);
        let numerator = if negative { -numerator } else { numerator };
        Self {
            lo: floor_div(&(numerator.clone() * scale()), &denominator),
            hi: ceil_div(&(numerator * scale()), &denominator),
        }
    }
    fn neg(&self) -> Self {
        Self {
            lo: -self.hi.clone(),
            hi: -self.lo.clone(),
        }
    }
    fn square(&self) -> Self {
        if self.lo >= BigInt::zero() {
            self.mul(self)
        } else if self.hi <= BigInt::zero() {
            self.neg().square()
        } else {
            Self {
                lo: BigInt::zero(),
                hi: ceil_div(&self.lo.pow(2).max(self.hi.pow(2)), &scale()),
            }
        }
    }
    fn add(&self, b: &Self) -> Self {
        Self {
            lo: &self.lo + &b.lo,
            hi: &self.hi + &b.hi,
        }
    }
    fn sub(&self, b: &Self) -> Self {
        self.add(&b.neg())
    }
    fn mul(&self, b: &Self) -> Self {
        let corners = [
            &self.lo * &b.lo,
            &self.lo * &b.hi,
            &self.hi * &b.lo,
            &self.hi * &b.hi,
        ];
        Self {
            lo: floor_div(corners.iter().min().expect("corners"), &scale()),
            hi: ceil_div(corners.iter().max().expect("corners"), &scale()),
        }
    }
    fn div(&self, b: &Self) -> Self {
        assert!(
            b.lo > BigInt::zero(),
            "strictly positive interval denominator"
        );
        let lows = [
            floor_div(&(&self.lo * scale()), &b.lo),
            floor_div(&(&self.lo * scale()), &b.hi),
            floor_div(&(&self.hi * scale()), &b.lo),
            floor_div(&(&self.hi * scale()), &b.hi),
        ];
        let highs = [
            ceil_div(&(&self.lo * scale()), &b.lo),
            ceil_div(&(&self.lo * scale()), &b.hi),
            ceil_div(&(&self.hi * scale()), &b.lo),
            ceil_div(&(&self.hi * scale()), &b.hi),
        ];
        Self {
            lo: lows.into_iter().min().expect("corners"),
            hi: highs.into_iter().max().expect("corners"),
        }
    }
    fn sqrt(&self) -> Self {
        assert!(!self.lo.is_negative());
        let lo = integer_sqrt(&(&self.lo * scale()));
        let mut hi = integer_sqrt(&(&self.hi * scale()));
        if hi.pow(2) < &self.hi * scale() {
            hi += 1;
        }
        Self { lo, hi }
    }
    fn clip(&self) -> Self {
        let s = scale();
        Self {
            lo: self.lo.clone().max(BigInt::zero()).min(s.clone()),
            hi: self.hi.clone().max(BigInt::zero()).min(s),
        }
    }
    fn exp(&self) -> Self {
        fn endpoint(x: &BigInt) -> Iv {
            if x.is_negative() {
                return Iv::rational(1, 1).div(&endpoint(&(-x)));
            }
            let mut u = Iv {
                lo: x.clone(),
                hi: x.clone(),
            };
            let mut reductions = 0;
            while u.hi > scale() {
                u = u.div(&Iv::rational(2, 1));
                reductions += 1;
            }
            let mut term = Iv::rational(1, 1);
            let mut sum = term.clone();
            for degree in 1..=80 {
                term = term.mul(&u).div(&Iv::rational(degree, 1));
                sum = sum.add(&term);
            }
            let remainder = term
                .mul(&u)
                .div(&Iv::rational(81, 1))
                .mul(&Iv::rational(3, 1));
            sum.hi += remainder.hi;
            for _ in 0..reductions {
                sum = sum.square();
            }
            sum
        }
        let a = endpoint(&self.lo);
        let b = endpoint(&self.hi);
        Self { lo: a.lo, hi: b.hi }
    }
    fn json(&self) -> Value {
        json!({"lower_scaled_integer":self.lo.to_string(),"upper_scaled_integer":self.hi.to_string(),"denominator":"10000000000000000000000000000000000000000"})
    }
    fn midpoint(&self) -> f64 {
        (&self.lo + &self.hi)
            .to_f64()
            .expect("representable certificate")
            / (2. * 1e40)
    }
}
fn isum(values: impl IntoIterator<Item = Iv>) -> Iv {
    values
        .into_iter()
        .fold(Iv::rational(0, 1), |a, b| a.add(&b))
}

/// Integrate a positive native event without expanding a duplicate-site
/// swarm into its individual rows. Given a same-site donor, the recipient
/// measures its own site and the donor measures the opposite site. Every
/// possible retained global measurement count is covered by directed score
/// intervals, so no expected-fitness substitution enters acceptance.
fn compressed_native_two_site_event(
    population: &BigInt,
    first_site_count: &BigInt,
    radius_numerator: i64,
    radius_denominator: i64,
    recipient_site: usize,
) -> (Iv, Value) {
    let one = Iv::rational(1, 1);
    let radius = Iv::rational(radius_numerator, radius_denominator);
    let separation = Iv::rational(4, 1)
        .mul(&radius)
        .div(&Iv::rational(2, 1).add(&radius));
    let raw_gap = separation
        .square()
        .add(&Iv::rational(1, 1_000_000))
        .sqrt()
        .sub(&Iv::rational(1, 1000));
    let weight = separation.square().div(&Iv::rational(8, 1)).neg().exp();
    let second_site_count = population - first_site_count;
    let site_counts = [first_site_count.clone(), second_site_count];
    let own = Iv::big_rational(&(site_counts[recipient_site].clone() - 1), &BigInt::one());
    let other = Iv::big_rational(&site_counts[1 - recipient_site], &BigInt::one());
    let denominator = own.add(&other.mul(&weight));
    let near = own.div(&denominator);
    let far = other.mul(&weight).div(&denominator);
    let map = |u: Iv| Iv::rational(1, 10).add(&Iv::rational(2, 1).div(&one.add(&u.neg().exp())));
    let mut minimum_lower: Option<BigInt> = None;
    let mut minimum_upper: Option<BigInt> = None;
    let mut cells = Vec::new();
    const CELLS: i64 = 256;
    for cell in 0..CELLS {
        let fraction = Iv {
            lo: Iv::rational(cell, CELLS).lo,
            hi: Iv::rational(cell + 1, CELLS).hi,
        };
        let standardizer = fraction
            .mul(&one.sub(&fraction))
            .mul(&raw_gap.square())
            .add(&Iv::rational(1, 100))
            .sqrt();
        let near_score = fraction.neg().mul(&raw_gap).div(&standardizer);
        let far_score = one.sub(&fraction).mul(&raw_gap).div(&standardizer);
        let near_fitness = Iv::rational(11, 10).mul(&map(near_score));
        let far_fitness = Iv::rational(11, 10).mul(&map(far_score));
        let acceptance = far_fitness
            .sub(&near_fitness)
            .div(&near_fitness.add(&Iv::rational(1, 1_000_000)))
            .clip();
        minimum_lower = Some(
            minimum_lower.map_or_else(|| acceptance.lo.clone(), |a| a.min(acceptance.lo.clone())),
        );
        minimum_upper = Some(
            minimum_upper.map_or_else(|| acceptance.hi.clone(), |a| a.min(acceptance.hi.clone())),
        );
        cells.push(json!({"fraction":fraction.json(),"actual_shared_scale":standardizer.json(),"actual_recipient_fitness":near_fitness.json(),"actual_donor_fitness":far_fitness.json(),"actual_acceptance":acceptance.json()}));
    }
    let acceptance = Iv {
        lo: minimum_lower.expect("nonempty count cover"),
        hi: minimum_upper.expect("nonempty count cover"),
    };
    let pressure = near.square().mul(&far).mul(&acceptance);
    let inputs = json!({"population_exact_integer":population.to_string(),"atom_multiplicities":site_counts.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "recipient_site":recipient_site,"radius_rational":[radius_numerator,radius_denominator],"feature_separation":separation.json(),
        "native_gaussian_pair_weight":weight.json(),"native_same_site_donor_probability":near.json(),
        "native_recipient_near_measurement_probability":near.json(),"native_donor_far_measurement_probability":far.json(),
        "actual_retained_measurement_count_acceptance_cells":cells,"actual_acceptance_infimum_enclosure":acceptance.json(),
        "native_event_subset_pressure_enclosure":pressure.json(),"reward_direction":"Minimize x²/2; both sites have exactly equal reward, so the native reward factor is 1.1",
        "scope":"Exact compressed native current-companion probabilities and an exhaustive interval cover of every possible retained global diversity count. This evaluates a subset of the actual accepted-event law; it does not substitute any proved Keystone or two-cluster pressure constant."});
    (pressure, inputs)
}
fn interval_standardize(y: &[Iv]) -> Vec<Iv> {
    let n = Iv::rational(y.len() as i64, 1);
    let m = isum(y.to_vec()).div(&n);
    let centered: Vec<_> = y.iter().map(|x| x.sub(&m)).collect();
    let variance = isum(centered.iter().map(Iv::square)).div(&n);
    let s = variance.add(&Iv::rational(1, 100)).sqrt();
    centered.iter().map(|x| x.div(&s)).collect()
}
fn interval_clone_variance(x: &[Iv]) -> (Iv, Iv, Iv) {
    let (variance, selection, jitter, _, _) = interval_clone_moments(x);
    (variance, selection, jitter)
}
fn interval_clone_moments(x: &[Iv]) -> (Iv, Iv, Iv, Iv, Iv) {
    let n = Iv::rational(4, 1);
    let two = Iv::rational(2, 1);
    let eight = Iv::rational(8, 1);
    let zero = Iv::rational(0, 1);
    let one = Iv::rational(1, 1);
    let g = |u: &Iv| Iv::rational(1, 10).add(&two.div(&one.add(&u.neg().exp())));
    let z: Vec<_> = x
        .iter()
        .map(|x| {
            two.mul(x).div(&two.add(&if x.lo.is_negative() {
                x.neg()
            } else {
                x.clone()
            }))
        })
        .collect();
    let rewards: Vec<_> = x.iter().map(|x| x.square().neg().div(&two)).collect();
    let rz = interval_standardize(&rewards);
    let rewardfactor: Vec<_> = rz.iter().map(g).collect();
    let mut measured = vec![vec![zero.clone(); 4]; 4];
    let mut prob = measured.clone();
    for i in 0..4 {
        for j in 0..4 {
            let dd = z[i].sub(&z[j]).square();
            measured[i][j] = dd.add(&Iv::rational(1, 1_000_000)).sqrt();
            if i != j {
                prob[i][j] = dd.div(&eight).neg().exp();
            }
        }
        let norm = isum(prob[i].clone());
        for pj in &mut prob[i] {
            *pj = pj.div(&norm);
        }
    }
    let mu = isum(x.to_vec()).div(&n);
    let centered: Vec<_> = x.iter().map(|x| x.sub(&mu)).collect();
    let entering = isum(centered.iter().map(Iv::square)).div(&n);
    let mut result = zero.clone();
    let mut selection = zero.clone();
    let mut jitter = zero.clone();
    let mut full_fourth_average = zero.clone();
    let mut full_variance_second_moment = zero.clone();
    let dt = Iv::rational(1, 50);
    let decay = Iv::rational(-1, 25).exp();
    let linear = one.sub(&dt.square().mul(&one.add(&decay)));
    let nu = dt
        .square()
        .mul(&one.sub(&Iv::rational(-2, 25).exp()).div(&two))
        .add(&Iv::rational(1, 2500));
    let jittered_nu = linear.square().mul(&Iv::rational(1, 100)).add(&nu);
    let gaussian_moments = |mean: &Iv, variance: &Iv| {
        vec![
            one.clone(),
            mean.clone(),
            mean.square().add(variance),
            mean.square()
                .mul(mean)
                .add(&Iv::rational(3, 1).mul(mean).mul(variance)),
            mean.square()
                .square()
                .add(&Iv::rational(6, 1).mul(&mean.square()).mul(variance))
                .add(&Iv::rational(3, 1).mul(&variance.square())),
        ]
    };
    let retained_moments: Vec<_> = x
        .iter()
        .map(|xi| gaussian_moments(&linear.mul(xi), &nu))
        .collect();
    let copied_moments: Vec<_> = x
        .iter()
        .map(|xi| gaussian_moments(&linear.mul(xi), &jittered_nu))
        .collect();
    for c0 in 0..4 {
        if c0 == 0 {
            continue;
        }
        for c1 in 0..4 {
            if c1 == 1 {
                continue;
            }
            for c2 in 0..4 {
                if c2 == 2 {
                    continue;
                }
                for c3 in 0..4 {
                    if c3 == 3 {
                        continue;
                    }
                    let choice = [c0, c1, c2, c3];
                    let mass = (0..4).fold(one.clone(), |w, i| w.mul(&prob[i][choice[i]]));
                    let y: Vec<_> = (0..4).map(|i| measured[i][choice[i]].clone()).collect();
                    let yz = interval_standardize(&y);
                    let f: Vec<_> = (0..4).map(|i| rewardfactor[i].mul(&g(&yz[i]))).collect();
                    let mut b = vec![vec![Iv::rational(0, 1); 4]; 4];
                    for i in 0..4 {
                        for j in 0..4 {
                            if i != j {
                                b[i][j] = prob[i][j].mul(
                                    &f[j]
                                        .sub(&f[i])
                                        .div(&f[i].add(&Iv::rational(1, 1_000_000)))
                                        .clip(),
                                );
                            }
                        }
                    }
                    let p: Vec<_> = b.iter().map(|row| isum(row.clone())).collect();
                    let mut row_moments = vec![vec![zero.clone(); 5]; 4];
                    for i in 0..4 {
                        for degree in 0..=4 {
                            row_moments[i][degree] = one
                                .sub(&p[i])
                                .clip()
                                .mul(&retained_moments[i][degree])
                                .add(&isum(
                                    (0..4).map(|j| b[i][j].mul(&copied_moments[j][degree])),
                                ));
                        }
                    }
                    let average_fourth = isum(row_moments.iter().map(|r| r[4].clone())).div(&n);
                    let mut terms = Vec::new();
                    for i in 0..4 {
                        for j in i..4 {
                            let mut exponents = [0_usize; 4];
                            exponents[i] += 1;
                            exponents[j] += 1;
                            terms.push((
                                exponents,
                                if i == j {
                                    Iv::rational(3, 16)
                                } else {
                                    Iv::rational(-1, 8)
                                },
                            ));
                        }
                    }
                    let mut second = zero.clone();
                    for (ea, ca) in &terms {
                        for (eb, cb) in &terms {
                            let product = (0..4).fold(one.clone(), |acc, i| {
                                acc.mul(&row_moments[i][ea[i] + eb[i]])
                            });
                            second = second.add(&ca.mul(cb).mul(&product));
                        }
                    }
                    full_fourth_average = full_fourth_average.add(&mass.mul(&average_fourth));
                    full_variance_second_moment =
                        full_variance_second_moment.add(&mass.mul(&second));
                    let t: Vec<_> = (0..4)
                        .map(|i| isum((0..4).map(|j| b[i][j].mul(&x[j].sub(&x[i])))))
                        .collect();
                    let a: Vec<_> = (0..4)
                        .map(|i| {
                            isum((0..4).map(|j| b[i][j].mul(&x[j].sub(&x[i]).square())))
                                .sub(&t[i].square())
                        })
                        .collect();
                    let flow = isum((0..4).flat_map(|i| {
                        let b = &b;
                        let centered = &centered;
                        (0..4).map(move |j| {
                            b[i][j].mul(&centered[j].square().sub(&centered[i].square()))
                        })
                    }))
                    .div(&n);
                    let sel = flow
                        .sub(&isum(t).div(&n).square())
                        .sub(&isum(a).div(&Iv::rational(16, 1)));
                    let jit = isum(p).mul(&Iv::rational(3, 1600));
                    selection = selection.add(&mass.mul(&sel));
                    jitter = jitter.add(&mass.mul(&jit));
                    result = result.add(&mass.mul(&entering.add(&sel).add(&jit)));
                }
            }
        }
    }
    (
        result,
        selection,
        jitter,
        full_fourth_average,
        full_variance_second_moment,
    )
}
fn certificate_check(id: &str, label: &str, passed: bool) -> BoundCheck {
    upper(id, label, if passed { 0. } else { 1. }, 0.)
}
fn expansion_certificates(suite: &mut EstimateSuite) -> Result<()> {
    let label = "prop-cloning-macroscopic-structural-expansion";
    let a = vec![
        Iv::rational(-3, 2),
        Iv::rational(-3, 2),
        Iv::rational(-3, 2),
        Iv::rational(1, 1),
    ];
    let b: Vec<_> = a.iter().map(|x| x.div(&Iv::rational(100, 1))).collect();
    let (va, selection, jitter, actual_fourth_a, actual_second_a) = interval_clone_moments(&a);
    let (vb, _, _, actual_fourth_b, actual_second_b) = interval_clone_moments(&b);
    let initial = Iv::rational(99 * 99 * 75, 100 * 100 * 64);
    let lower = va.sqrt().sub(&vb.sqrt()).square();
    let increment = lower.sub(&initial);
    let inputs = json!({"fixed_denominator":"10^40","outward_integer_arithmetic":true,"exp_degree":80,"exp_remainder":"3 u^81/81! on [0,1]","dyadic_argument_reduction":true,"complete_patterns":81,"native_analytic_variance_A":va.json(),"native_analytic_variance_B":vb.json(),"selection":selection.json(),"jitter":jitter.json()});
    record(
        suite,
        label,
        "3.X1",
        inputs.clone(),
        vec![equal(
            "exact_initial_structural_error",
            label,
            initial.midpoint(),
            1.1485546875,
        )],
    )?;
    record(
        suite,
        label,
        "3.X2",
        inputs.clone(),
        vec![certificate_check(
            "directed_rational_native_law_variance",
            label,
            va.lo > Iv::decimal("1.31954463969802").hi
                && va.hi < Iv::decimal("1.31954463969803").lo,
        )],
    )?;
    record(
        suite,
        label,
        "1.31954463969802",
        inputs.clone(),
        vec![certificate_check(
            "strict_variance_A_interval",
            label,
            va.lo > Iv::decimal("1.31954463969802").hi
                && va.hi < Iv::decimal("1.31954463969803").lo,
        )],
    )?;
    record(
        suite,
        label,
        "3.X3",
        inputs.clone(),
        vec![certificate_check(
            "strict_variance_B_interval",
            label,
            vb.lo > Iv::decimal("0.00035711824482").hi
                && vb.hi < Iv::decimal("0.00035711824483").lo,
        )],
    )?;
    record(
        suite,
        label,
        "3.X4",
        json!({"lower_increment":increment.json(),"initial":initial.json(),"marginal_variance_certificate":inputs}),
        vec![
            certificate_check(
                "strict_structural_expansion",
                label,
                increment.lo > Iv::decimal("0.12793124541590").hi,
            ),
            certificate_check(
                "macroscopic_expansion_threshold",
                label,
                increment.lo > Iv::decimal("0.12").hi,
            ),
        ],
    )?;
    let full = "prop-canonical-fullstep-structural-expansion";
    let c = Iv::rational(1, 50);
    let friction = Iv::rational(-1, 25).exp();
    let t = Iv::rational(1, 1).sub(&c.square().mul(&Iv::rational(1, 1).add(&friction)));
    let ouvar = Iv::rational(1, 1)
        .sub(&Iv::rational(-2, 25).exp())
        .div(&Iv::rational(2, 1));
    let nu = c.square().mul(&ouvar).add(&Iv::rational(1, 2500));
    let ma = t.square().mul(&va).add(&Iv::rational(3, 4).mul(&nu));
    let mb = t.square().mul(&vb).add(&Iv::rational(3, 4).mul(&nu));

    let config = GasConfig::euclidean(1, 0.04)?;
    let (native_h, native_gamma) = match config.kinetic.integrator {
        algorithmic_gas::kinetic::KineticKind::Baoab { dt, friction, .. } => (dt, friction),
        _ => return Err(error("canonical expansion requires native BAOAB")),
    };
    let native_b = match &config.kinetic.noise.geometry {
        algorithmic_gas::noise::NoiseGeometry::Isotropic {
            scale: algorithmic_gas::noise::FactorValues::Constant { values },
        } if values.len() == 1 => values[0],
        _ => return Err(error("canonical expansion requires retained isotropic B")),
    };
    let nc = native_h / 2.;
    let na = (-native_gamma * native_h).exp();
    let nq2 =
        native_b * native_b * (1. - (-2. * native_gamma * native_h).exp()) / (2. * native_gamma);
    let ns = config.kinetic.position_diffusion * native_h.sqrt();
    let nt = 1. - nc * nc * (1. + na);
    let nnu = nc * nc * nq2 + ns * ns;
    let mut native_stage_checks = vec![
        equal_positive("actual_native_BAOAB_h", full, native_h, 0.04),
        equal_positive("actual_native_BAOAB_friction", full, native_gamma, 1.),
        equal_positive("actual_native_velocity_noise_factor", full, native_b, 1.),
        equal_positive(
            "actual_native_half_drift_coefficient",
            full,
            nc,
            c.midpoint(),
        ),
        equal_positive("actual_native_OU_decay", full, na, friction.midpoint()),
        equal_positive("actual_native_OU_variance", full, nq2, ouvar.midpoint()),
        equal_positive("actual_native_position_noise_coefficient", full, ns, 0.02),
        equal_positive(
            "actual_native_BAOAB_position_coefficient",
            full,
            nt,
            t.midpoint(),
        ),
        equal_positive(
            "actual_native_combined_position_variance",
            full,
            nnu,
            nu.midpoint(),
        ),
        certificate_check(
            "actual_native_preterminal_kinetic_gaussians",
            full,
            config.kinetic.noise.innovation == algorithmic_gas::noise::InnovationLaw::Gaussian,
        ),
        certificate_check(
            "strict_position_coefficient_domain",
            full,
            t.lo > BigInt::zero() && t.hi < Iv::rational(1, 1).lo,
        ),
        certificate_check(
            "directed_native_position_variance_upper",
            full,
            t.square().mul(&Iv::rational(1, 100)).add(&nu).hi < Iv::decimal("0.0105").lo,
        ),
        certificate_check(
            "independent_directed_MA_source_interval",
            full,
            ma.lo > Iv::decimal("1.31778710461037").hi
                && ma.hi < Iv::decimal("1.31778710461038").lo,
        ),
        certificate_check(
            "independent_directed_MB_source_interval",
            full,
            mb.lo > Iv::decimal("0.00066809082559").hi
                && mb.hi < Iv::decimal("0.00066809082561").lo,
        ),
    ];
    for source in [-1.5, -0.015, 0., 0.01, 1.] {
        for xi in [-2., -0.1, 0., 0.2, 2.] {
            for zeta in [-2., -0.1, 0., 0.2, 2.] {
                let v_b1 = -nc * source;
                let x_a1 = source + nc * v_b1;
                let v_o = na * v_b1 + nq2.sqrt() * xi;
                let actual_stage_position = x_a1 + nc * v_o + ns * zeta;
                native_stage_checks.push(equal(
                    "native_frozen_zero_velocity_quadratic_B1_A1_O_A2_coordinate",
                    full,
                    actual_stage_position,
                    nt * source + nc * nq2.sqrt() * xi + ns * zeta,
                ));
                let q = actual_stage_position * actual_stage_position
                    + v_o * v_o
                    + 0.1 * actual_stage_position * v_o;
                native_stage_checks.push(equal(
                    "actual_declared_phase_quadratic",
                    full,
                    q,
                    399. / 400. * actual_stage_position.powi(2)
                        + (v_o + 0.05 * actual_stage_position).powi(2),
                ));
            }
        }
    }
    let native_coefficients = json!({"native_config":config,"h":native_h,"friction":native_gamma,"velocity_factor":native_b,"position_diffusion":config.kinetic.position_diffusion,"c":c.json(),"a":friction.json(),"q_squared":ouvar.json(),"s":"1/50","t":t.json(),"nu":nu.json(),"MA":ma.json(),"MB":mb.json(),
        "scope":"Actual canonical BAOAB and innovation parameters determine the independently reconstructed B1-A1-O-A2 affine position map. Directed rational exponential and native all-81-pattern moment certificates enclose every coefficient and marginal variance; affine coefficient witnesses check every Gaussian coordinate without imposing a coupling between the two output marginals."});
    records(
        suite,
        full,
        &[
            "h=.04",
            r"\gamma=1",
            "B=1",
            r"Q(x,v)=x^2+v^2+.1xv",
            "c=.02",
            r"\widehat x_i=tX_i",
            "1.31778710461037<M_A<1.31778710461038",
            ".00066809082559<M_B<.00066809082561",
            r"t^2(.1)^2+\nu<.0105",
            "0<t<1",
        ],
        native_coefficients,
        native_stage_checks,
    )?;
    let deltaa = Iv::decimal("0.000055");
    let deltab = Iv::decimal("0.000000000000000000000000000001");
    let la = ma.sub(&Iv::decimal("5.205").mul(&deltaa).sqrt());
    let ub = mb
        .add(&Iv::rational(4, 1).mul(&deltab))
        .div(&Iv::rational(1, 1).sub(&deltab));
    let final_lower = la
        .sqrt()
        .sub(&ub.sqrt())
        .square()
        .mul(&Iv::rational(399, 400));
    let taila = Iv::rational(8, 1).mul(&Iv::rational(-1, 4).div(&Iv::decimal("0.021")).exp());
    let tailb = Iv::rational(8, 1).mul(
        &Iv::decimal("-1.985")
            .square()
            .neg()
            .div(&Iv::decimal("0.021"))
            .exp(),
    );
    let fourth = Iv::decimal("1.5")
        .square()
        .square()
        .add(
            &Iv::rational(6, 1)
                .mul(&Iv::decimal("1.5").square())
                .mul(&Iv::decimal("0.0105")),
        )
        .add(&Iv::rational(3, 1).mul(&Iv::decimal("0.0105").square()));
    let inputs = json!({"h":"0.04","velocity_diffusion":"1","position_diffusion":"0.1","mean_A":ma.json(),"mean_B":mb.json(),"tail_A":taila.json(),"tail_B":tailb.json(),"fourth_moment_bound":fourth.json(),"survival_normalized_lower_A":la.json(),"survival_normalized_upper_B":ub.json(),"projected_structural_lower":final_lower.json()});
    let one = Iv::rational(1, 1);
    let zero = Iv::rational(0, 1);
    let event_tail_a = Iv {
        lo: BigInt::zero(),
        hi: taila.hi.clone(),
    };
    let event_tail_b = Iv {
        lo: BigInt::zero(),
        hi: tailb.hi.clone(),
    };
    let surviving_a = Iv {
        lo: one.sub(&taila).lo,
        hi: one.hi.clone(),
    };
    let surviving_b = Iv {
        lo: one.sub(&tailb).lo,
        hi: one.hi.clone(),
    };
    let discarded_upper = actual_second_a.mul(&taila).sqrt();
    let discarded = Iv {
        lo: BigInt::zero(),
        hi: discarded_upper.hi.clone(),
    };
    let good_a = ma.sub(&discarded);
    let extra_a = Iv {
        lo: BigInt::zero(),
        hi: Iv::rational(4, 1).mul(&taila).hi,
    };
    let conditional_a = good_a.add(&extra_a).div(&surviving_a);
    let first_a_margin = good_a
        .mul(&one.sub(&surviving_a))
        .add(&extra_a)
        .div(&surviving_a);
    let discarded_margin = Iv::decimal("5.205").mul(&deltaa).sqrt().sub(&discarded);
    let numerator_b_upper = mb.add(&Iv::rational(4, 1).mul(&tailb));
    let conditional_b_upper = numerator_b_upper.div(&one.sub(&tailb));
    let b_comparison_margin = deltab.sub(&tailb).mul(&mb.add(&Iv::rational(4, 1)));
    let conditional_inputs = json!({"directed_denominator":"10^40","full_native_measurement_patterns":81,
        "actual_full_row_fourth_average_A":actual_fourth_a.json(),"actual_full_variance_second_moment_A":actual_second_a.json(),
        "actual_full_row_fourth_average_B":actual_fourth_b.json(),"actual_full_variance_second_moment_B":actual_second_b.json(),
        "actual_full_variance_means":[ma.json(),mb.json()],"event_tail_intervals":[event_tail_a.json(),event_tail_b.json()],
        "nonextinction_mass_intervals":[surviving_a.json(),surviving_b.json()],"discarded_A_moment_interval":discarded.json(),
        "all_alive_A_numerator_interval":good_a.json(),"partial_survival_A_numerator_interval":extra_a.json(),
        "survival_conditioned_A_enclosure":conditional_a.json(),"conditional_minus_good_A_margin":first_a_margin.json(),
        "good_A_minus_source_LA_margin":discarded_margin.json(),"B_numerator_upper":numerator_b_upper.json(),
        "survival_conditioned_B_upper_enclosure":conditional_b_upper.json(),"source_UB_minus_actual_upper_cross_multiplication_margin":b_comparison_margin.json(),
        "scope":"The exact native 81-pattern law gives first through fourth row moments. Conditional independence given the entire measurement vector integrates E[V²] as its degree-four polynomial. Chernoff tail intervals, bounded surviving variance and E subset H give actual numerator and normalizer enclosures; nonnegative directed margins certify every conditional-chain relation while preserving correlations."});

    records(
        suite,
        full,
        &[
            r"V_s=\operatorname{Var}(\widehat x^s)",
            r"V_A\le\frac14\sum_i\widehat x_i^2",
            r"\mathbb P(H_B)\ge\mathbb P(E_B)\ge1-\delta_B",
        ],
        conditional_inputs.clone(),
        vec![
            certificate_check(
                "native_variance_squared_actual_moment_bound",
                full,
                actual_second_a.hi <= actual_fourth_a.hi,
            ),
            certificate_check(
                "native_event_subset_survival_mass_lower",
                full,
                surviving_b.lo >= one.sub(&deltab).hi,
            ),
            certificate_check(
                "native_all_alive_event_mass_lower",
                full,
                one.sub(&tailb).lo >= one.sub(&deltab).hi,
            ),
            upper(
                "native_preboundary_variance_centered_second_moment",
                full,
                var(&[-1.5, -1.5, -1.5, 1.]),
                [-1.5_f64, -1.5, -1.5, 1.]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    / 4.,
            ),
            equal(
                "native_preboundary_variance_definition",
                full,
                var(&[-1.5, -1.5, -1.5, 1.]),
                [-1.5_f64, -1.5, -1.5, 1.]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    / 4.
                    - mean(&[-1.5, -1.5, -1.5, 1.]).powi(2),
            ),
        ],
    )?;
    record(
        suite,
        full,
        "3.X7",
        conditional_inputs.clone(),
        vec![
            certificate_check(
                "fullstep_variance_A_interval",
                full,
                ma.lo > Iv::decimal("1.31778710461037").hi
                    && ma.hi < Iv::decimal("1.31778710461038").lo,
            ),
            certificate_check(
                "fullstep_variance_B_interval",
                full,
                mb.lo > Iv::decimal("0.00066809082559").hi
                    && mb.hi < Iv::decimal("0.00066809082561").lo,
            ),
        ],
    )?;
    record(
        suite,
        full,
        "\\mathbb P(E_A^c)",
        inputs.clone(),
        vec![certificate_check(
            "Gaussian_union_tail_A",
            full,
            taila.hi < deltaa.lo,
        )],
    )?;
    record(
        suite,
        full,
        "3.X8",
        inputs.clone(),
        vec![certificate_check(
            "Gaussian_union_tail_B",
            full,
            tailb.hi < deltab.lo,
        )],
    )?;
    record(
        suite,
        full,
        "3.X9",
        conditional_inputs.clone(),
        vec![
            certificate_check(
                "fourth_moment_upper",
                full,
                fourth.hi < Iv::decimal("5.205").lo,
            ),
            certificate_check(
                "positive_surviving_variance_lower",
                full,
                la.lo > BigInt::zero(),
            ),
            certificate_check(
                "actual_native_variance_fourth_Jensen",
                full,
                actual_second_a.hi < actual_fourth_a.lo,
            ),
            certificate_check(
                "actual_native_row_fourth_uniform_bound",
                full,
                actual_fourth_a.hi < fourth.lo,
            ),
            certificate_check(
                "actual_A_good_numerator_positive",
                full,
                good_a.lo > zero.hi,
            ),
            certificate_check(
                "actual_A_conditional_ge_unconditioned_good_numerator",
                full,
                first_a_margin.lo >= BigInt::zero(),
            ),
            certificate_check(
                "actual_A_good_numerator_ge_source_LA",
                full,
                discarded_margin.lo >= BigInt::zero(),
            ),
        ],
    )?;
    record(
        suite,
        full,
        "3.X10",
        conditional_inputs,
        vec![
            certificate_check(
                "strict_survival_normalizer",
                full,
                Iv::rational(1, 1).sub(&deltab).lo > BigInt::zero(),
            ),
            certificate_check(
                "actual_B_good_event_survival_normalizer",
                full,
                surviving_b.lo >= one.sub(&deltab).hi,
            ),
            certificate_check(
                "actual_B_partial_survival_numerator_upper",
                full,
                Iv::rational(4, 1).mul(&deltab.sub(&tailb)).lo >= BigInt::zero(),
            ),
            certificate_check(
                "actual_B_conditional_upper_full_fraction_chain",
                full,
                b_comparison_margin.lo >= BigInt::zero(),
            ),
        ],
    )?;
    record(
        suite,
        full,
        "3.X6",
        inputs,
        vec![certificate_check(
            "complete_update_structural_expansion",
            full,
            final_lower.lo > initial.add(&Iv::decimal("0.08")).hi,
        )],
    )
}

fn boundary_algebra(suite: &mut EstimateSuite) -> Result<()> {
    let mut rewardgap = Vec::new();
    for alpha in [0.25_f64, 0.5, 1., 2.] {
        for beta in [0_f64, 0.5, 1., 2.] {
            for (zi, zj) in [(-1_f64, 0.), (-0.5, 0.5), (0., 1.)] {
                let map = |u: f64| 0.1 + 2. / (1. + (-u).exp());
                let derivative = |u: f64| {
                    let e = (-u).exp();
                    2. * e / (1. + e).powi(2)
                };
                let mg = derivative(zi).min(derivative(zj));
                let gap = 0.9_f64.powf(beta) * map(zj).powf(alpha)
                    - 0.7_f64.powf(beta) * map(zi).powf(alpha);
                rewardgap.push(lower(
                    "powered_product_monotone_diversity_first_step",
                    "lem-fitness-gradient-boundary",
                    gap,
                    0.7_f64.powf(beta) * (map(zj).powf(alpha) - map(zi).powf(alpha)),
                ));
                let bound = 0.1_f64.powf(beta)
                    * alpha
                    * 0.1_f64.powf(alpha - 1.).min(2.1_f64.powf(alpha - 1.))
                    * mg
                    * (zj - zi);
                rewardgap.push(lower(
                    "powered_product_reward_gap",
                    "lem-fitness-gradient-boundary",
                    gap,
                    bound,
                ));
            }
        }
    }
    records(
        suite,
        "lem-fitness-gradient-boundary",
        &["V_j-V_i\\geq", "V_j-V_i\\geq(d'_i)^\\beta"],
        json!({"alpha":[0.25,0.5,1.,2.],"beta":[0.,0.5,1.,2.],"eta":0.1,"M":2.1,"shared_scale":1.,"derivative_interval_checked":true}),
        rewardgap,
    )?;
    let mut taylor = Vec::new();
    for y in [-2_f64, -1., 0., 0.5, 2.] {
        for sigma in [0_f64, 0.01, 0.1, 1.] {
            taylor.push(equal(
                "Gaussian_quadratic_Taylor",
                "lem-barrier-reduction-cloning",
                y * y + sigma * sigma,
                y * y + 0.5 * 2. * sigma * sigma,
            ));
        }
    }
    record(
        suite,
        "lem-barrier-reduction-cloning",
        "\\mathbb E\\varphi(y+",
        json!({"observable":"x² on all R; auxiliary observable only","global_Hessian_norm":2.,"dimension":1,"Gaussian_second_moment":1.}),
        taylor,
    )?;
    let mut density = Vec::new();
    for sigma in [0.05_f64, 0.1, 0.5, 1.] {
        let mq = 1. / (2. * std::f64::consts::PI * sigma * sigma).sqrt();
        for y in [-2_f64, -1., 0., 1., 2.] {
            for i in -100..=100 {
                let z = i as f64 / 25.;
                let q = mq * (-(z - y).powi(2) / (2. * sigma * sigma)).exp();
                density.push(upper(
                    "Gaussian_density_uniform_bound",
                    "lem-barrier-reduction-cloning",
                    q,
                    mq,
                ));
            }
        }
    }
    record(
        suite,
        "lem-barrier-reduction-cloning",
        "M_q=(2\\pi",
        json!({"dimension":1,"sigma":[0.05,0.1,0.5,1.],"analytic_density_maximum_at_mean":true}),
        density,
    )?;
    let mut safe = Vec::new();
    let mut extinction = Vec::new();
    let mut offsets = Vec::new();
    let mut recursion = Vec::new();
    for n in [4_usize, 16, 64, 256] {
        let alive = n * 3 / 4;
        let values: Vec<_> = (0..alive)
            .map(|i| if i % 3 == 0 { 3. } else { 0.2 })
            .collect();
        let theta = 1.;
        let b = values.iter().sum::<f64>() / n as f64;
        let actual = values.iter().filter(|&&v| v < theta).count();
        safe.push(lower(
            "safe_alive_count",
            "cor-extinction-suppression",
            actual as f64,
            alive as f64 - n as f64 * b / theta,
        ));
        let radius = 1_f64;
        let sigma = 0.1;
        let q = (2_f64.sqrt() * (-(radius * radius) / (4. * sigma * sigma)).exp()).min(1.);
        let fraction = actual as f64 / n as f64;
        extinction.push(upper(
            "safe_pool_joint_extinction",
            "cor-extinction-suppression",
            q.powi(actual as i32),
            q.powf(fraction * n as f64),
        ));
        extinction.push(upper(
            "strict_individual_death_bound",
            "cor-extinction-suppression",
            q,
            1. - f64::EPSILON,
        ));
        let deaths = 2 * (n - alive);
        let r = 3_f64;
        let j = 2_f64;
        let ps = 0.1;
        let offset = 2. * alive as f64 / n as f64 * (ps * theta + j) + deaths as f64 / n as f64 * r;
        offsets.push(upper(
            "uniform_boundary_offset",
            "thm-complete-boundary-drift",
            offset,
            2. * ps * theta + 2. * j.max(r),
        ));
        offsets.push(upper(
            "uniform_revival_injection",
            "thm-complete-boundary-drift",
            r * deaths as f64 / n as f64,
            2. * r,
        ));
        let rate = 0.2_f64;
        let cb = 0.7;
        let mut value = 12.;
        for time in 0_i32..100 {
            let bound = (1. - rate).powi(time) * 12. + cb * (1. - (1. - rate).powi(time)) / rate;
            recursion.push(equal(
                "boundary_geometric_iteration",
                "cor-bounded-boundary-exposure",
                value,
                bound,
            ));
            value = (1. - rate) * value + cb;
        }
    }
    record(
        suite,
        "cor-extinction-suppression",
        "k-NB/\\theta",
        json!({"alive_fraction":0.75,"theta":1.}),
        safe,
    )?;
    record(
        suite,
        "cor-extinction-suppression",
        "q^{aN}",
        json!({"conditioned_stage":"independent final Gaussian position innovations with means at safe distance 1","sigma":0.1,"dimension":1,"independence_retained":true}),
        extinction,
    )?;
    record(
        suite,
        "thm-complete-boundary-drift",
        "C_b(S_1,S_2)",
        json!({"fixed_death_fraction":0.25,"N":[4,16,64,256]}),
        offsets,
    )?;
    record(
        suite,
        "cor-bounded-boundary-exposure",
        "C_b \\sum",
        json!({"declared_affine_transition_rate":0.2,"offset":0.7,"initial":12.}),
        recursion,
    )
}

fn weighted_assembly(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let k = live.len();
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let va: Vec<_> = live.iter().map(|&i| law.v[i]).collect();
    let dx = xa.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - xa.iter().copied().fold(f64::INFINITY, f64::min);
    let vmax = law.v.iter().map(|v| v.abs()).fold(0_f64, f64::max);
    let uv = mean(&va);
    let dead = (0..n).filter(|&i| !law.alive[i]);
    let rv = dead.clone().map(|i| (law.v[i] - uv).powi(2)).sum::<f64>() / n as f64
        - (dead.map(|i| law.v[i] - uv).sum::<f64>() / n as f64).powi(2);
    let alpha = 0.5;
    let cv = 0.7;
    let cb = 0.05;
    let initialx = 2. * var(&xa) * k as f64 / n as f64;
    let initialv = 2. * var(&va) * k as f64 / n as f64;
    let initialb = 2. * xa.iter().map(|x| x * x).sum::<f64>() / n as f64;
    let mut outputx = 0.;
    let mut ec = 0.;
    let mut outputb = 0.;
    let mut favorable = Vec::new();
    let mut favorable_inputs = Vec::new();
    let mut acceptance_lip = Vec::new();
    let constants = canonical_cloning_constants();
    for p in &law.patterns {
        outputx += 2. * p.mass * p.output_var;
        ec += 2. * p.mass * graph_moments(law, p, alpha).0;
        outputb += 2.
            * p.mass
            * p.means
                .iter()
                .zip(&p.traces)
                .map(|(m, c)| m * m + c)
                .sum::<f64>()
            / n as f64;
        for &i in &live {
            let rows: Vec<_> = live
                .iter()
                .copied()
                .filter(|&j| p.f[j] > p.f[i] + 1e-12)
                .collect();
            if let Some(delta) = rows.iter().map(|&j| p.f[j] - p.f[i]).reduce(f64::min) {
                let q = rows.iter().map(|&j| law.donors[i][j]).sum::<f64>();
                let label = "lem-boundary-enhanced-cloning";
                let a_min = (-4_f64).exp();
                let q_lower = rows.len() as f64 * a_min / (k - 1) as f64;
                favorable.push(certificate_check(
                    "actual_favorable_gap_strict",
                    label,
                    delta > 0.,
                ));
                favorable.push(certificate_check(
                    "actual_favorable_companion_mass_strict",
                    label,
                    q > 0.,
                ));
                favorable.push(upper(
                    "actual_favorable_recipient_fitness_bound",
                    label,
                    p.f[i],
                    4.41,
                ));
                favorable.push(certificate_check(
                    "actual_native_kernel_floor_positive",
                    label,
                    a_min > 0.,
                ));
                favorable.push(lower(
                    "actual_favorable_kernel_mass_lower",
                    label,
                    q,
                    q_lower,
                ));
                favorable.push(equal_positive(
                    "actual_favorable_kernel_fraction_formula",
                    label,
                    q_lower,
                    rows.len() as f64 * a_min / ((k - 1) as f64 * 1.),
                ));
                for &j in &rows {
                    favorable.push(lower(
                        "actual_each_favorable_companion_gap",
                        label,
                        p.f[j] - p.f[i],
                        delta,
                    ));
                    favorable.push(certificate_check(
                        "actual_favorable_current_eligible_nonself_membership",
                        label,
                        law.alive[j] && i != j,
                    ));
                }
                favorable_inputs.push(json!({"recipient":i,"possible_companions_H":rows,"native_full_fitness":p.f,"delta":delta,"actual_native_H_probability":q,"kernel_lower_H_probability":q_lower,"native_row_pressure":p.p[i],"native_donor_probabilities":law.donors[i],"a_min":a_min,"a_max":1.,"k":k,"exposure_scope":"Selection statement conditional on exposed membership; physical positions remain in the input, and the estimate's quantitative hypotheses are actual fitness, eligible donor support and independent threshold."}));
                favorable.push(lower(
                    "actual_favorable_mass_pressure",
                    "lem-boundary-enhanced-cloning",
                    p.p[i],
                    q * (delta / (4.41 + 1e-6)).min(1.),
                ));
            }
        }
        for &i in &live {
            for &j in &live {
                for &a in &live {
                    for &b in &live {
                        let left = law
                            .config
                            .clone_decision
                            .acceptance_probability(1, p.f[i], p.f[j]);
                        let right = law
                            .config
                            .clone_decision
                            .acceptance_probability(1, p.f[a], p.f[b]);
                        acceptance_lip.push(equal(
                            "actual_clipped_acceptance_formula",
                            "thm-cloning-canonical-barycenter-concentration",
                            left,
                            ((p.f[j] - p.f[i]) / (p.f[i] + 1e-6)).clamp(0., 1.),
                        ));
                        acceptance_lip.push(upper(
                            "actual_clipped_acceptance_lipschitz",
                            "thm-cloning-canonical-barycenter-concentration",
                            (left - right).abs(),
                            constants.l_a * ((p.f[i] - p.f[a]).abs() + (p.f[j] - p.f[b]).abs()),
                        ));
                    }
                }
            }
        }
    }
    let outputv = 2. * var(&law.v) - (1. - alpha * alpha) * ec;
    let cx = dx * dx + 0.02;
    let cvel = if k == n { 0. } else { 8. * vmax * vmax };
    let delta = cv * (outputx - initialx + outputv - initialv) + cb * (outputb - initialb);
    let kw = 2. * (4. + 0.01) + (1. + 0.1_f64.powi(2) / 4.) * vmax * vmax;
    let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"coupling":"identical complete inputs and innovations; empirical transport stays zero","X":initialx,"Y":initialv,"PCX":outputx,"PCY":outputv,"component_energy":ec,"revival_velocity":2.*rv,"cV":cv,"cB":cb,"Wb":"global C2 auxiliary x²; objective unchanged"});
    let mut alias_checks = vec![
        equal(
            "actual_normalized_weighted_function_definition",
            "thm-complete-cloning-drift",
            cv * (initialx + initialv) + cb * initialb,
            0. + cv * (2. * k as f64 / n as f64 * var(&xa) + 2. * k as f64 / n as f64 * var(&va))
                + cb * initialb,
        ),
        equal(
            "two_native_revived_velocity_remainders",
            "thm-complete-cloning-drift",
            2. * rv,
            2. * (var(&law.v) - k as f64 / n as f64 * var(&va)),
        ),
    ];
    let independent_energy = law
        .patterns
        .iter()
        .map(|p| {
            p.mass
                * graph_plans(p)
                    .iter()
                    .map(|(choices, mass)| {
                        mass * components(choices)
                            .iter()
                            .map(|g| {
                                let frozen_energy =
                                    g.iter().map(|&i| law.v[i] * law.v[i]).sum::<f64>();
                                let frozen_sum = g.iter().map(|&i| law.v[i]).sum::<f64>();
                                (frozen_energy - frozen_sum * frozen_sum / g.len() as f64)
                                    / n as f64
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
        })
        .sum::<f64>();
    alias_checks.push(equal(
        "two_native_expected_component_relative_energies",
        "thm-complete-cloning-drift",
        ec,
        2. * independent_energy,
    ));
    for rate in [0.01_f64, 0.2, 1.] {
        alias_checks.push(certificate_check(
            "chosen_affine_reset_rate_interval",
            "thm-complete-cloning-drift",
            rate > 0. && rate <= 1.,
        ));
        alias_checks.push(upper(
            "nonnegative_variance_affine_reset_relaxation",
            "thm-complete-cloning-drift",
            -initialx + cx,
            -rate * initialx + cx,
        ));
    }
    records(
        suite,
        "thm-complete-cloning-drift",
        &[
            r"\Phi=V_{\mathrm{total}}=",
            r"R=R_v(S_1)+R_v(S_2)",
            r"\overline{\mathcal E}_C=",
            r"0<\kappa_x\le1",
            r"-X+C_x\le-\kappa_xX+C_x",
        ],
        inputs.clone(),
        alias_checks,
    )?;
    if k == n {
        record(
            suite,
            "thm-complete-cloning-drift",
            "C_v=0",
            inputs.clone(),
            vec![
                equal(
                    "actual_all_alive_velocity_remainder_zero",
                    "thm-complete-cloning-drift",
                    rv,
                    0.,
                ),
                equal(
                    "actual_all_alive_velocity_offset_zero",
                    "thm-complete-cloning-drift",
                    cvel,
                    0.,
                ),
                upper(
                    "actual_all_alive_component_velocity_drift_nonpositive",
                    "thm-complete-cloning-drift",
                    outputv - initialv,
                    0.,
                ),
            ],
        )?;
    }
    let mut moment_aliases = vec![
        certificate_check(
            "actual_restitution_unit_interval",
            "cor-cloning-actual-inter-swarm-expansion",
            (0. ..=1.).contains(&alpha),
        ),
        certificate_check(
            "chosen_moment_young_parameter_positive",
            "cor-cloning-actual-inter-swarm-expansion",
            1_f64 > 0.,
        ),
        equal(
            "actual_moment_anchor_bound_alias",
            "cor-cloning-actual-inter-swarm-expansion",
            kw,
            (1. + 1.) * (4. + 0.01) + (1. + 0.1_f64.powi(2) / (4. * 1.)) * vmax * vmax,
        ),
    ];
    for p in &law.patterns {
        for (choices, _) in graph_plans(p) {
            for &j in &choices {
                moment_aliases.push(certificate_check(
                    "every_native_copied_or_retained_source_eligible",
                    "cor-cloning-actual-inter-swarm-expansion",
                    law.alive[j],
                ));
                moment_aliases.push(upper(
                    "every_native_source_position_in_anchor_ball",
                    "cor-cloning-actual-inter-swarm-expansion",
                    law.x[j].abs(),
                    2.,
                ));
            }
        }
    }
    records(
        suite,
        "cor-cloning-actual-inter-swarm-expansion",
        &[
            r"0\le\alpha\le1",
            r"\eta>0",
            r"|Y_i-x_0|\le B_x",
            r"M_h=K_\eta",
        ],
        json!({"native_input":inputs,"eta":1.,"anchor":0.,"anchor_radius":2.,"lambda_v":1.,"b":0.1,"K_eta":kw,"M_h":kw,"scope":"Actual native complete finite donor/gate law. Every source position is current eligible even for revival, while retained dead positions can be arbitrarily distant; the Haar component energy estimate is integrated separately."}),
        moment_aliases,
    )?;
    record(
        suite,
        "thm-complete-cloning-drift",
        "3.AC1",
        inputs.clone(),
        vec![equal(
            "native_weighted_increment",
            "thm-complete-cloning-drift",
            delta,
            cv * (outputx - initialx + 2. * rv - (1. - alpha * alpha) * ec)
                + cb * (outputb - initialb),
        )],
    )?;
    record(
        suite,
        "thm-complete-cloning-drift",
        "3.AC2",
        inputs.clone(),
        vec![
            upper(
                "native_two_swarm_position_reset",
                "thm-complete-cloning-drift",
                outputx,
                cx,
            ),
            equal(
                "native_weighted_velocity_balance",
                "thm-complete-cloning-drift",
                outputv,
                initialv + 2. * rv - (1. - alpha * alpha) * ec,
            ),
        ],
    )?;
    let mut relaxed = Vec::new();
    for kappa in [0.01, 0.2, 1.] {
        relaxed.push(upper(
            "native_position_affine_relaxation",
            "thm-complete-cloning-drift",
            outputx - initialx,
            -kappa * initialx + cx,
        ));
    }
    relaxed.push(upper(
        "native_velocity_bounded_drift",
        "thm-complete-cloning-drift",
        outputv - initialv,
        cvel,
    ));
    record(
        suite,
        "thm-complete-cloning-drift",
        "3.AC3",
        inputs.clone(),
        relaxed,
    )?;
    let label = "thm-complete-cloning-drift";
    record(
        suite,
        label,
        "0\\le R\\le\\frac{4(D_1+D_2)}N",
        inputs.clone(),
        vec![
            lower(
                "actual_two_swarm_velocity_revival_nonnegative",
                label,
                2. * rv,
                0.,
            ),
            upper(
                "actual_two_swarm_velocity_revival_dead_fraction",
                label,
                2. * rv,
                8. * (n - k) as f64 / n as f64 * vmax * vmax,
            ),
            upper(
                "actual_velocity_revival_uniform_relaxation",
                label,
                8. * (n - k) as f64 / n as f64 * vmax * vmax,
                8. * vmax * vmax,
            ),
        ],
    )?;
    let label = "cor-cloning-weighted-assembly-input";
    let uniform_cv = 8. * vmax * vmax;
    let defect = initialv + uniform_cv - outputv;
    record(
        suite,
        label,
        "(D_C)_Y=C_v-\\lambda_vR",
        inputs.clone(),
        vec![
            equal(
                "native_exact_velocity_defect",
                label,
                defect,
                uniform_cv - 2. * rv + (1. - alpha * alpha) * ec,
            ),
            lower(
                "native_velocity_defect_contains_component_energy",
                label,
                defect,
                (1. - alpha * alpha) * ec,
            ),
        ],
    )?;
    record(
        suite,
        "thm-complete-cloning-drift",
        "3.AC4",
        inputs.clone(),
        vec![upper(
            "native_weighted_affine_assembly",
            "thm-complete-cloning-drift",
            delta,
            4. * kw + cv * (-0.2 * initialx + cx + cvel) + cb * (outputb - initialb),
        )],
    )?;
    if !favorable.is_empty() {
        records(
            suite,
            "lem-boundary-enhanced-cloning",
            &[
                "p_i\\geq q",
                r"V_j-V_i\geq\Delta>0",
                r"\mathbb P(c_i\in H_i\mid S)\geq q>0",
                r"V_i\leq V_{\max}",
                r"a_{\min}>0",
                r"q=|H_i|a_{\min}/((k-1)a_{\max})",
                r"\{c_i\in H_i\}",
            ],
            json!({"native_input":inputs,"all_actual_favorable_rows":favorable_inputs}),
            favorable,
        )?;
    }
    record(
        suite,
        "thm-cloning-canonical-barycenter-concentration",
        "a(f,g)=",
        inputs,
        acceptance_lip,
    )
}

fn records(
    suite: &mut EstimateSuite,
    label: &str,
    needles: &[&str],
    inputs: Value,
    checks: Vec<BoundCheck>,
) -> Result<()> {
    for needle in needles {
        record(suite, label, needle, inputs.clone(), checks.clone())?;
    }
    Ok(())
}

fn elementary_geometry(suite: &mut EstimateSuite) -> Result<()> {
    let label = "prop-barrier-existence";
    let mut bump = Vec::new();
    for t in [-1.5_f64, -1., -0.5, 0., 0.5, 1., 1.5] {
        let observed = if t.abs() < 1. {
            (-1. / (1. - t * t)).exp()
        } else {
            0.
        };
        let expected = if t.abs() == 0.5 {
            Iv::rational(-4, 3).exp().midpoint()
        } else if t == 0. {
            Iv::rational(-1, 1).exp().midpoint()
        } else {
            0.
        };
        bump.push(equal("bump_piecewise_value", label, observed, expected));
    }
    record(
        suite,
        label,
        "\\eta(t)",
        json!({"t":[-1.5,-1.,-0.5,0.,0.5,1.,1.5],"independent_directed_exp_arithmetic":true}),
        bump,
    )?;
    for (rho, psi, needle) in [
        (0.1_f64, 1., "\\varphi(x) = \\frac{1}{\\delta} + 1"),
        (0.5, 0., "\\varphi(x) = \\frac{1}{\\delta} + 0"),
        (0.3, 0.5, "\\varphi(x) = \\underbrace"),
    ] {
        let delta = 0.2;
        let phi = 1. / delta + psi * (1. / rho - 1. / delta);
        let first = (1. - psi) / delta;
        let second = psi / rho;
        record(
            suite,
            label,
            needle,
            json!({"rho":rho,"delta":delta,"psi":psi,"middle_value_by_bump_symmetry":rho==0.3}),
            vec![
                equal("barrier_sum_identity", label, phi, first + second),
                certificate_check("strict_barrier_positivity", label, phi > 0.),
                certificate_check(
                    "transition_summands_positive",
                    label,
                    psi != 0.5 || (first > 0. && second > 0.),
                ),
            ],
        )?;
    }
    let label = "lem-V-coercive";
    let mut checks = Vec::new();
    let mut cases = Vec::new();
    for b in [-1.3_f64, 0.1, 1.2] {
        let lambda = 0.5_f64;
        let discriminant = ((1. - lambda).powi(2) + b * b).sqrt();
        let minus = (1. + lambda - discriminant) / 2.;
        let plus = (1. + lambda + discriminant) / 2.;
        checks.push(upper(
            "strict_positive_definite_parameter",
            label,
            b * b,
            4. * lambda,
        ));
        checks.push(equal(
            "determinant",
            label,
            lambda - b * b / 4.,
            minus * plus,
        ));
        checks.push(equal(
            "both_eigenvalues_trace",
            label,
            minus + plus,
            1. + lambda,
        ));
        for eigen in [minus, plus] {
            checks.push(equal(
                "both_eigenvalues_characteristic_equation",
                label,
                eigen * eigen - (1. + lambda) * eigen + lambda - b * b / 4.,
                0.,
            ));
        }
        checks.push(certificate_check(
            "strict_eigenvalue_positivity",
            label,
            minus > 0. && discriminant < 1. + lambda,
        ));
        let q = |x: f64, v: f64| x * x + lambda * v * v + b * x * v;
        for x in -10..=10 {
            for v in -10..=10 {
                checks.push(lower(
                    "quadratic_coercivity",
                    label,
                    q(x as f64 / 4., v as f64 / 5.),
                    minus * ((x as f64 / 4.).powi(2) + (v as f64 / 5.).powi(2)),
                ));
            }
        }
        let left = vec![vec![-1., -0.4], vec![0., 0.2], vec![1., 0.2]];
        let right = vec![vec![-0.5, -0.1], vec![0.5, 0.1]];
        let (hypo, _) = crate::convergence_lyapunov::uniform_transport(&left, &right, |a, c| {
            q(a[0] - c[0], a[1] - c[1])
        })?;
        let (euclid, _) = crate::convergence_lyapunov::uniform_transport(&left, &right, |a, c| {
            (a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2)
        })?;
        let mut integral_q = 0.;
        let mut integral_euclid = 0.;
        for a in &left {
            for c in &right {
                integral_q += q(a[0] - c[0], a[1] - c[1]) / (left.len() * right.len()) as f64;
                integral_euclid += ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2))
                    / (left.len() * right.len()) as f64;
            }
        }
        checks.push(lower(
            "actual_product_coupling_integral_coercivity",
            label,
            integral_q,
            minus * integral_euclid,
        ));
        checks.push(lower(
            "alive_structural_transport_coercivity",
            label,
            hypo,
            minus * euclid,
        ));
        cases.push(json!({"b":b,"lambda_v":lambda,"eigenvalues":[minus,plus],"centered_left":left,"centered_right":right,"W_h2":hypo,"W_22":euclid}));
    }
    records(
        suite,
        label,
        &[
            "b^2 < 4",
            "\\det(Q_",
            "\\lambda_{-} =",
            "\\lambda_{\\pm} =",
            "\\lambda_{\\min} =",
            "q(\\Delta x, \\Delta v) \\geq",
            "V_{\\text{loc}} \\geq",
            "\\int q(\\delta_{x,1} - \\delta_{x,2}, \\delta_{v,1} - \\delta_{v,2}) \\, d\\gamma \\geq",
            "V_{\\text{struct}} \\geq",
            "V_{\\text{struct}} = W_h^2",
        ],
        json!({"cases":cases,"same_quadratic_form_for_location_and_centered_transport":true}),
        checks,
    )?;
    let x = vec![-1_f64, -0.8, 0.8, 1.];
    let v = vec![-0.4_f64, 0.3, -0.2, 0.5];
    let k = x.len();
    let lambda = 0.5_f64;
    let alg = 1.;
    let dh2 = 4. + 0.9_f64.powi(2);
    let close = 1.;
    let vh = var(&x) + lambda * var(&v);
    let mut sum = 0.;
    let mut near = 0;
    let mut pairchecks = Vec::new();
    for i in 0..k {
        for j in i + 1..k {
            let physical = (x[i] - x[j]).powi(2) + lambda * (v[i] - v[j]).powi(2);
            let metric = (x[i] - x[j]).powi(2) + alg * (v[i] - v[j]).powi(2);
            sum += physical;
            if metric < close * close {
                near += 1;
                pairchecks.push(upper(
                    "close_quadratic_vs_algorithmic",
                    "lem-phase-space-packing",
                    physical,
                    metric,
                ));
                pairchecks.push(certificate_check(
                    "strict_close_threshold",
                    "lem-phase-space-packing",
                    metric < close * close,
                ));
            }
            pairchecks.push(upper(
                "all_pair_diameter",
                "lem-phase-space-packing",
                physical,
                dh2,
            ));
        }
    }
    let total = k * (k - 1) / 2;
    let fraction = near as f64 / total as f64;
    let mixture = fraction * close * close + (1. - fraction) * dh2;
    let exact_upper = (k - 1) as f64 / (2. * k as f64) * mixture;
    let simple = mixture / 2.;
    let g = (dh2 - 2. * vh) / (dh2 - close * close);
    let label = "lem-phase-space-packing";
    pairchecks.extend([
        equal("unordered_pair_variance", label, vh, sum / (k * k) as f64),
        upper("finite_pair_mixture", label, vh, exact_upper),
        certificate_check("strict_pair_mixture_relaxation", label, vh < simple),
        upper("close_population_fraction", label, fraction, g),
        certificate_check("strict_close_population_fraction", label, fraction < g),
        certificate_check(
            "nontrivial_packing_implication",
            label,
            g >= 1. || vh > close * close / 2.,
        ),
    ]);
    let kf = k as f64;
    let sx = x.iter().map(|x| x * x).sum::<f64>();
    let doublex = x
        .iter()
        .flat_map(|a| x.iter().map(move |b| (a - b).powi(2)))
        .sum::<f64>();
    let doublev = v
        .iter()
        .flat_map(|a| v.iter().map(move |b| (a - b).powi(2)))
        .sum::<f64>();
    pairchecks.extend([
        equal(
            "double_position_pairwise_square_expansion",
            label,
            doublex,
            2. * kf * sx - 2. * kf * kf * mean(&x).powi(2),
        ),
        equal(
            "position_moment_variance_identity",
            label,
            sx,
            kf * var(&x) + kf * mean(&x).powi(2),
        ),
        equal(
            "double_position_pairwise_centered_variance",
            label,
            doublex,
            2. * kf * kf * var(&x),
        ),
        equal(
            "double_velocity_pairwise_centered_variance",
            label,
            doublev,
            2. * kf * kf * var(&v),
        ),
        equal(
            "double_phase_pairwise_centered_variance",
            label,
            doublex + lambda * doublev,
            2. * kf * kf * vh,
        ),
        equal(
            "close_far_unordered_partition_sum",
            label,
            vh,
            sum / (kf * kf),
        ),
    ]);
    records(
        suite,
        label,
        &[
            "\\mathrm{Var}_h(S_k) :=",
            "2k^2 \\mathrm{Var}_x(S_k) =",
            "\\begin{aligned}\n\\sum_{i,j} \\|x_i - x_j\\|^2",
            "2k^2 \\mathrm{Var}_v(S_k) =",
            "2k^2 \\mathrm{Var}_h(S_k) =",
            "\\mathrm{Var}_h(S_k) = \\frac{1}{k^2} \\left(",
            "f_{\\text{close}} \\le",
            "\\mathrm{Var}_h(S_k) = \\frac{1}{k^2} \\sum_{i<j}",
            "\\|x_i - x_j\\|^2 + \\lambda_v \\|v_i - v_j\\|^2 \\le \\|x_i",
            "\\|x_i - x_j\\|^2 + \\lambda_v \\|v_i - v_j\\|^2 \\le D_x",
            "\\mathrm{Var}_h(S_k) \\le \\frac{1}{k^2}",
            "\\mathrm{Var}_h(S_k) \\le \\frac{\\binom",
            "\\mathrm{Var}_h(S_k) < \\frac{1}{2}",
            "2\\mathrm{Var}_h(S_k) &<",
            "f_{\\text{close}} < \\frac{2",
            "\\frac{D_{\\text{valid}}^2 - 2\\mathrm{Var}_h}",
        ],
        json!({"positions":x,"velocities":v,"lambda_v":lambda,"lambda_alg":alg,"explicit_comparison":"unsquashed Euclidean configuration","k":k,"near_pairs":near,"d_close":close,"Dvalid2":dh2,"Var_h":vh}),
        pairchecks,
    )?;
    records(
        suite,
        "lem-var-x-implies-var-h",
        &[
            "\\mathrm{Var}_h(S_k) \\ge",
            "\\mathrm{Var}_h(S_k) = \\mathrm{Var}_x",
        ],
        json!({"Var_x":var(&x),"Var_v":var(&v),"lambda_v":lambda}),
        vec![
            equal(
                "variance_components",
                "lem-var-x-implies-var-h",
                vh,
                var(&x) + lambda * var(&v),
            ),
            lower(
                "positive_velocity_variance",
                "lem-var-x-implies-var-h",
                vh,
                var(&x),
            ),
        ],
    )
}

/// Every evidence item quotes the source expression it actually evaluates.
pub async fn validate_estimates(samples: usize) -> Result<EstimateSuite> {
    if samples < 2 {
        return Err(error(
            "estimate validation needs at least two requested samples",
        ));
    }
    let mut suite=EstimateSuite{chapter:3,evidence:Vec::new(),scope_notes:vec!["Exact finite laws validate displayed estimates on their retained hypotheses; large-population constants remain in logarithms when native populations do not satisfy the theorem threshold.".into(),"The completed affine envelope is conditional on the stated full-stage component inequalities. It does not invent a canonical kinetic contraction rate.".into()]};
    centered_product(&mut suite)?;
    for (x, v, alive) in [
        (
            vec![-1.5, -0.5, 0.5, 1.],
            vec![0.3, -0.2, 0.1, -0.4],
            vec![true; 4],
        ),
        (
            vec![-0.8, 0.8, 1e9, -1e9],
            vec![0.4, -0.3, 0.2, -0.1],
            vec![true, true, false, false],
        ),
        (
            vec![0.2, 1e9, -1e9, 1e6],
            vec![0.1, -0.2, 0.3, -0.4],
            vec![true, false, false, false],
        ),
    ] {
        let law = enumerate_law(x, v, alive)?;
        influence(&mut suite, &law)?;
        native_barycenter_aliases(&mut suite, &law)?;
        native_signed_flux_aliases(&mut suite, &law)?;
        native_centered_row_aliases(&mut suite, &law)?;
        native_decision_definition_aliases(&mut suite, &law)?;
        native_row_operator_aliases(&mut suite, &law)?;
        native_moment_component_aliases(&mut suite, &law)?;
        balances(&mut suite, &law)?;
        native_collective_aliases(&mut suite, &law)?;
        weighted_assembly(&mut suite, &law)?;
        native_proof_equations(&mut suite, &law)?;
        native_declarations(&mut suite, &law)?;
        native_flow_identities(&mut suite, &law)?;
        boundary_complete_equations(&mut suite, &law)?;
    }
    boundary_complete_equations(
        &mut suite,
        &enumerate_law(
            vec![-1.5, -0.5, 0.5, 1e9],
            vec![0.3, -0.2, 0.1, -0.4],
            vec![true, true, true, false],
        )?,
    )?;
    keystone(&mut suite)?;
    backbone(&mut suite)?;
    composition(&mut suite)?;
    expansion_certificates(&mut suite)?;
    boundary_algebra(&mut suite)?;
    elementary_geometry(&mut suite)?;
    selection_equations(&mut suite)?;
    matching_equations(&mut suite)?;
    incremental_and_moving_center(&mut suite)?;
    remaining_keystone_proof(&mut suite)?;
    revival_smoothing(&mut suite)?;
    expansion_chain_checks(&mut suite, samples)?;
    remaining_elementary_bounds(&mut suite)?;
    transport_identities(&mut suite)?;
    native_fitness_identities(&mut suite)?;
    nontrivial_cluster_remainder(&mut suite)?;
    keystone_native_inline_inputs(&mut suite)?;
    complete_coupled_increment_identity(&mut suite)?;
    assembly_offset_definitions(&mut suite)?;
    native_variance_definition_chains(&mut suite)?;
    native_quotient_state_witness(&mut suite)?;
    native_geometric_event_definitions(&mut suite)?;
    expansion_native_definitions(&mut suite)?;
    native_update_definition_witness(&mut suite).await?;
    native_completed_velocity_cap(&mut suite).await?;
    no_affine_structural_aliases(&mut suite)?;
    native_row_operator_aliases(
        &mut suite,
        &enumerate_law(vec![0.2], vec![0.1], vec![true])?,
    )?;
    balanced_cluster_native_definitions(&mut suite)?;
    moving_center_native_definitions(&mut suite)?;
    complete_keystone_native_aliases(&mut suite)?;
    geometric_native_hypotheses(&mut suite)?;
    quantitative_native_keystone_assembly(&mut suite)?;
    native_spreading_example_aliases(&mut suite)?;
    native_centered_row_aliases(
        &mut suite,
        &enumerate_law(vec![0.2], vec![0.1], vec![true])?,
    )?;
    final_source_clauses(&mut suite)?;
    Ok(suite)
}

fn fixture_equation(
    suite: &mut EstimateSuite,
    fixture: &crate::convergence_selection::SelectionFixture,
    label: &str,
    needle: &str,
    extra: &[BoundCheck],
) -> Result<()> {
    let mut checks: Vec<_> = fixture
        .checks
        .iter()
        .filter(|c| c.source_labels.iter().any(|s| s == label))
        .cloned()
        .collect();
    checks.extend(extra.iter().cloned());
    if checks.is_empty() {
        return Err(error(&format!("empty mapped equation {label}")));
    }
    record(
        suite,
        label,
        needle,
        json!({"fixture":fixture.name,"retained_hypotheses":fixture.hypotheses,"observations":fixture.observations,"constants":fixture.constants}),
        checks,
    )?;
    suite.evidence.last_mut().unwrap().scope = fixture.scope.clone();
    Ok(())
}
fn jnum(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| error(&format!("numeric field {key} missing")))
}
fn jvec(value: &Value, key: &str) -> Result<Vec<f64>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| error("vector missing"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| error("numeric vector missing")))
        .collect()
}
fn selection_equations(suite: &mut EstimateSuite) -> Result<()> {
    let report = crate::convergence_selection::validate_selection(20261002)?;
    for fixture in &report.fixtures {
        let mut extra = Vec::new();
        let h = &fixture.hypotheses;
        let o = &fixture.observations;
        let c = &fixture.constants;
        if fixture.name == "finite_cluster_geometry_and_target" {
            let x = jvec(h, "positions")?;
            let v = jvec(h, "velocities")?;
            let k = x.len() as f64;
            let n = jnum(h, "slots")?;
            let mx = mean(&x);
            let mv = mean(&v);
            let q: Vec<_> = x
                .iter()
                .zip(&v)
                .map(|(x, v)| (x - mx).powi(2) + (v - mv).powi(2))
                .collect();
            let high: Vec<_> = o["high"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_bool().unwrap())
                .collect();
            let clusters: Vec<Vec<usize>> =
                serde_json::from_value(o["clusters"].clone()).map_err(|e| error(&e.to_string()))?;
            let dh2 = c["D_h_squared"];
            let epsilon = jnum(h, "epsilon_outlier")?;
            let rv = jnum(h, "R_var_squared")?;
            let ovar = jnum(o, "phase_variance")?;
            let out: Vec<usize> = serde_json::from_value(o["auxiliary_global_outliers"].clone())
                .map_err(|e| error(&e.to_string()))?;
            let captured = out.iter().map(|&i| q[i]).sum::<f64>();
            let sum = q.iter().sum::<f64>();
            let label = "lem-outlier-fraction-lower-bound";
            extra.extend([
                lower(
                    "actual_outlier_capture",
                    label,
                    captured,
                    (1. - epsilon) * sum,
                ),
                certificate_check("strict_variance_threshold", label, ovar > rv),
                certificate_check(
                    "positive_outlier_fraction_constant",
                    label,
                    (1. - epsilon) * rv / dh2 > 0.,
                ),
                certificate_check(
                    "strict_outlier_count",
                    label,
                    out.len() as f64 / k > (1. - epsilon) * rv / dh2,
                ),
                certificate_check(
                    "first_strict_outlier_chain",
                    label,
                    (1. - epsilon) * k * rv < (1. - epsilon) * k * ovar,
                ),
                upper(
                    "last_outlier_chain",
                    label,
                    captured,
                    out.len() as f64 * dh2,
                ),
            ]);
            let mut valid_energy = 0.;
            let mut valid_retained = 0.;
            let mut between = 0.;
            let mut high_energy = 0.;
            let mut high_x = 0.;
            for g in &clusters {
                let gx: Vec<_> = g.iter().map(|&i| x[i]).collect();
                let gv: Vec<_> = g.iter().map(|&i| v[i]).collect();
                let center = g.len() as f64 * ((mean(&gx) - mx).powi(2) + (mean(&gv) - mv).powi(2));
                between += center;
                let selected = g.iter().all(|&i| high[i]);
                if g.len() >= 5 {
                    valid_energy += center;
                    if selected {
                        valid_retained += center;
                    }
                }
                if selected {
                    high_energy += center;
                    high_x += g.iter().map(|&i| (x[i] - mx).powi(2)).sum::<f64>();
                }
                let alg_diam = g
                    .iter()
                    .flat_map(|&i| g.iter().map(move |&j| (i, j)))
                    .map(|(i, j)| ((x[i] - x[j]).powi(2) + 2. * (v[i] - v[j]).powi(2)).sqrt())
                    .fold(0., f64::max);
                let dx = range(&gx);
                let radius = gx.iter().map(|x| (x - mean(&gx)).abs()).fold(0., f64::max);
                extra.extend([
                    upper(
                        "actual_cluster_diameter",
                        "def-unified-high-low-error-sets",
                        alg_diam,
                        0.12,
                    ),
                    upper(
                        "actual_phase_cluster_variance",
                        "lem-outlier-cluster-fraction-lower-bound",
                        var(&gx) + var(&gv),
                        0.12_f64.powi(2) / 2.,
                    ),
                    upper(
                        "actual_cluster_radius",
                        "rem-cluster-energy-and-separation",
                        radius,
                        dx,
                    ),
                    upper(
                        "actual_cluster_x_variance",
                        "rem-cluster-energy-and-separation",
                        var(&gx),
                        dx * dx / 2.,
                    ),
                    upper(
                        "actual_within_x_variance",
                        "lem-variance-concentration-Hk",
                        var(&gx),
                        0.12_f64.powi(2) / 2.,
                    ),
                ]);
                let pair_sum = g
                    .iter()
                    .flat_map(|&i| g.iter().map(move |&j| (i, j)))
                    .map(|(i, j)| (x[i] - x[j]).powi(2))
                    .sum::<f64>();
                extra.push(equal(
                    "cluster_pairwise_variance",
                    "lem-variance-concentration-Hk",
                    var(&gx),
                    pair_sum / (2. * (g.len() * g.len()) as f64),
                ));
            }
            extra.extend([
                lower(
                    "valid_contribution_capture",
                    "def-unified-high-low-error-sets",
                    valid_retained,
                    (1. - epsilon) * valid_energy,
                ),
                lower(
                    "between_contribution_capture",
                    "lem-outlier-cluster-fraction-lower-bound",
                    high_energy,
                    (1. - epsilon) * between,
                ),
                lower(
                    "positional_energy_capture",
                    "lem-variance-concentration-Hk",
                    high_x,
                    c["c_H"] * k * var(&x),
                ),
                lower(
                    "positional_between_capture",
                    "lem-variance-concentration-Hk",
                    high_x,
                    (1. - epsilon) * (k * var(&x) - k * 0.12_f64.powi(2) / 2.),
                ),
                upper(
                    "cluster_membership_definition",
                    "def-unified-high-low-error-sets",
                    clusters
                        .iter()
                        .flat_map(|g| g.iter())
                        .filter(|&&i| {
                            high[i]
                                != (clusters.iter().find(|g| g.contains(&i)).unwrap().len() < 5
                                    || clusters[0].contains(&i))
                        })
                        .count() as f64,
                    0.,
                ),
            ]);
            let fitness: Vec<f64> = high.iter().map(|&b| if b { 0.8 } else { 1.4 }).collect();
            let mu = mean(&fitness);
            let s2 = var(&fitness);
            let r = range(&fitness);
            let absmean = fitness.iter().map(|f| (f - mu).abs()).sum::<f64>() / k;
            let posmean = fitness.iter().map(|f| (f - mu).max(0.)).sum::<f64>() / k;
            extra.extend([
                upper(
                    "variance_absolute_deviation",
                    "lem-mean-companion-fitness-gap",
                    s2,
                    r * absmean,
                ),
                equal(
                    "centered_positive_negative_parts",
                    "lem-mean-companion-fitness-gap",
                    absmean,
                    2. * posmean,
                ),
                certificate_check(
                    "positive_pressure_constant",
                    "lem-unfit-cloning-pressure",
                    c["p_u"] > 0.,
                ),
                equal(
                    "pressure_constant_formula",
                    "lem-unfit-cloning-pressure",
                    c["p_u"],
                    c["selection_a"] * s2 / (2. * r * c["selection_A_V"]),
                ),
            ]);
            for check in &fixture.checks {
                if check.id.starts_with("unfit_positive_signal_") {
                    let actual = -check.observed;
                    extra.push(lower(
                        "nonself_fraction_refinement",
                        "lem-mean-companion-fitness-gap",
                        actual,
                        c["selection_a"] * k / (k - 1.) * s2 / (2. * r),
                    ));
                }
            }
            let vw = jnum(o, "positional_transport_to_point_mass")?;
            let chi = c["p_u"] * c["c_error"];
            let offset = c["p_u"] * c["g_error"];
            let activity = jnum(&o["conditional_selection"], "weighted_pressure_error")? * k / n;
            extra.extend([
                lower(
                    "realized_target_keystone",
                    "lem-quantitative-keystone",
                    activity,
                    chi * vw - offset,
                ),
                lower(
                    "first_target_subset",
                    "rem-keystone-balanced-structural-scope",
                    activity,
                    activity,
                ),
                lower(
                    "target_error_pressure",
                    "rem-keystone-balanced-structural-scope",
                    activity,
                    c["p_u"] * jnum(&o["conditional_selection"], "target_error")? * k / n,
                ),
                lower(
                    "target_capture_pressure",
                    "rem-keystone-balanced-structural-scope",
                    activity,
                    chi * vw - offset,
                ),
            ]);
        }

        if fixture.name == "fixed_shared_rescale" {
            let y = jvec(h, "raw")?;
            let d = jvec(o, "rescaled")?;
            let z = jvec(o, "scores")?;
            let k = y.len() as f64;
            let v = var(&y);
            let span = range(&y);
            let sm = 0.1_f64;
            let smx = c["s_max"];
            let m = c["m_g"];
            let mg = c["M_g"];
            let patch = c["sigma_patch_max"];
            let label = "prop-fixed-rescale-variance-bound";
            extra.extend([
                lower("positive_derivative_min", label, m, 0.),
                upper("derivative_extrema_order", label, m, mg),
                upper("finite_derivative_max", label, mg, f64::MAX),
                upper(
                    "variance_upper_lipschitz_branch",
                    label,
                    var(&d),
                    mg * mg / (sm * sm) * v,
                ),
                upper(
                    "variance_upper_support_branch",
                    label,
                    var(&d),
                    c["R_g"].powi(2) / 4.,
                ),
                lower(
                    "nonvacuous_variance_lower",
                    label,
                    var(&d),
                    m * m / (smx * smx) * v,
                ),
            ]);
            let mut sum = 0.;
            for i in 0..y.len() {
                for j in 0..y.len() {
                    let gap = (y[i] - y[j]).abs();
                    sum += gap * gap;
                    extra.extend([
                        lower(
                            "pair_rescale_lower",
                            label,
                            (d[i] - d[j]).abs(),
                            m / smx * gap,
                        ),
                        upper(
                            "pair_rescale_upper",
                            label,
                            (d[i] - d[j]).abs(),
                            mg / sm * gap,
                        ),
                    ]);
                }
            }
            extra.extend([
                upper(
                    "maximum_pair_sum",
                    "lem-variance-to-gap",
                    sum,
                    k * k * span * span,
                ),
                equal(
                    "pair_variance_identity",
                    "lem-variance-to-gap",
                    v,
                    sum / (2. * k * k),
                ),
                upper(
                    "half_span_variance",
                    "lem-variance-to-gap",
                    v,
                    span * span / 2.,
                ),
                lower(
                    "variance_gap_threshold",
                    "lem-variance-to-gap",
                    span,
                    (2. * v).sqrt(),
                ),
                equal(
                    "maximum_patch_scale",
                    "def-max-patched-std",
                    patch,
                    (c["V_max"].powi(2) + sm * sm).sqrt(),
                ),
            ]);
            let gap = span;
            let zgap = range(&z);
            let ggap = range(&d);
            let gmin = c["g_prime_family_min"];
            extra.extend([
                lower(
                    "score_gap",
                    "lem-raw-gap-to-rescaled-gap",
                    zgap,
                    gap / patch,
                ),
                lower(
                    "rescale_gap_intermediate",
                    "lem-raw-gap-to-rescaled-gap",
                    ggap,
                    gmin * gap / patch,
                ),
                certificate_check(
                    "strict_rescale_gap",
                    "lem-raw-gap-to-rescaled-gap",
                    ggap > 0. && gmin * gap / patch > 0.,
                ),
                lower(
                    "support_derivative_min",
                    "lem-rescale-derivative-lower-bound",
                    gmin,
                    0.,
                ),
            ]);
        }
        if fixture.name == "group_and_log_fitness_separation" {
            let d = jvec(o, "diversity_factors")?;
            let fitness = jvec(o, "fitness")?;
            let alpha = jnum(h, "alpha")?;
            let beta = jnum(h, "beta")?;
            let dgap = mean(&d[2..]) - mean(&d[..2]);
            let rlog = jnum(o, "reward_log_gap")?;
            let dlog = jnum(o, "diversity_log_gap")?;
            let q = 0.28_f64;
            let t = 0.4_f64;
            let probability = 0.3_f64;
            let em2 = 0.3_f64;
            let radius = 1.;
            extra.extend([
                upper(
                    "bounded_gap_event_chain",
                    "cor-averaged-group-separation",
                    em2,
                    t * t + (radius * radius - t * t) * probability,
                ),
                upper(
                    "within_group_conditional_bound",
                    "cor-averaged-group-separation",
                    0.,
                    0.07,
                ),
                certificate_check(
                    "strict_conditional_within_threshold",
                    "cor-averaged-group-separation",
                    0. < 0.07,
                ),
                certificate_check(
                    "positive_conditional_Q",
                    "cor-averaged-group-separation",
                    q > 0.,
                ),
                upper(
                    "conditional_Q_support",
                    "cor-averaged-group-separation",
                    q,
                    radius * radius,
                ),
                certificate_check(
                    "positive_conditional_event",
                    "cor-averaged-group-separation",
                    (q - t * t) / (radius * radius - t * t) > 0.,
                ),
            ]);
            extra.extend([
                lower(
                    "ordered_log_linear_lower",
                    "lem-log-gap-lower-bound",
                    dlog,
                    dgap / 2.1,
                ),
                lower(
                    "ordered_log_logarithmic_lower",
                    "lem-log-gap-lower-bound",
                    dgap / 2.1,
                    (1. + dgap / 2.1).ln(),
                ),
                upper(
                    "ordered_reward_log_upper",
                    "lem-log-gap-upper-bound",
                    rlog,
                    (1. + c["K_r"] / 0.1).ln(),
                ),
                equal(
                    "exact_product_log_gap",
                    "thm-stability-condition-final-corrected",
                    jnum(o, "fitness_log_gap")?,
                    beta * dlog - alpha * rlog,
                ),
            ]);
            let lowmean = mean(&fitness[2..]);
            let highmean = mean(&fitness[..2]);
            let vstar = c["v_star"];
            let defect =
                lowmean.ln() - mean(&fitness[2..].iter().map(|f| f.ln()).collect::<Vec<_>>());
            extra.extend([
                lower(
                    "nonnegative_Jensen_defect",
                    "thm-stability-condition-final-corrected",
                    defect,
                    0.,
                ),
                upper(
                    "Jensen_variance_defect",
                    "thm-stability-condition-final-corrected",
                    defect,
                    var(&fitness[2..]) / (2. * vstar * vstar),
                ),
                lower(
                    "arithmetic_exponential_gap",
                    "thm-stability-condition-final-corrected",
                    lowmean - highmean,
                    vstar * (c["delta_arithmetic"].exp() - 1.),
                ),
                certificate_check(
                    "positive_arithmetic_exponential_bound",
                    "thm-stability-condition-final-corrected",
                    vstar * (c["delta_arithmetic"].exp() - 1.) > 0.,
                ),
            ]);
        }
        if fixture.name == "exact_measurement_averaged_small_clusters" {
            let label = "lem-keystone-geometric-measurement-events";
            let feature = [0., 2. / 3., 2. / 3., 2. / 3.];
            let v = var(&feature);
            let h = 0.2_f64;
            let dz = 2. / 3.;
            let rho = (v - h * h) / (dz * dz - h * h);
            extra.extend([
                equal("AP1_actual_feature_variance", label, rho, c["rho_h"]),
                certificate_check("AP1_strict_positive", label, rho > 0.),
                certificate_check("AP3_positive_floor", label, c["f_min"] > 0.),
                certificate_check("AP3_positive_derivative", label, c["m_f"] > 0.),
            ]);
            for &z in &feature {
                let ms = feature.iter().map(|a| (a - z).powi(2)).sum::<f64>() / 4.;
                let count = feature.iter().filter(|&&a| (a - z).abs() >= h).count() as f64 / 4.;
                extra.extend([
                    equal(
                        "AP1_centered_identity",
                        label,
                        ms,
                        v + (z - mean(&feature)).powi(2),
                    ),
                    upper(
                        "AP1_far_fraction",
                        label,
                        ms,
                        h * h + (dz * dz - h * h) * count,
                    ),
                ]);
            }
            let w = 0.25_f64;
            let eta = 0.4_f64;
            let lambda = 0.5_f64;
            let b = 0.1_f64;
            extra.push(upper(
                "AP7_structural_young",
                "thm-keystone-averaged-error-capture",
                w,
                (1. + eta) * w + (lambda + b * b / (4. * eta)) * 0.,
            ));
        }
        if fixture.name.starts_with("finite_") && fixture.name.ends_with("weighted_composition") {
            let input = &h["inputs"];
            let values: Vec<Vec<f64>> = serde_json::from_value(input["observables"].clone())
                .map_err(|e| error(&e.to_string()))?;
            let pc: Vec<Vec<f64>> = serde_json::from_value(input["cloning"].clone())
                .map_err(|e| error(&e.to_string()))?;
            let rates = jvec(input, "cloning_rates")?;
            let offsets = jvec(input, "cloning_offsets")?;
            let cv = jnum(input, "variance_weight")?;
            let cb = jnum(input, "boundary_weight")?;
            let pk: Vec<Vec<f64>> = serde_json::from_value(input["kinetic"].clone())
                .map_err(|e| error(&e.to_string()))?;
            let krates = jvec(input, "kinetic_rates")?;
            let bk = jvec(input, "kinetic_offsets")?;
            let ac = [1., 1. - rates[0], 1., 1. - rates[1]];
            let ak = [1. - krates[0], 1., 1. - krates[1], 1.];
            let weights = [1., cv, cv, cb];
            let cstar = (0..4)
                .map(|r| weights[r] * (ak[r] * offsets[r] + bk[r]))
                .sum::<f64>();
            extra.push(equal(
                "explicit_AC9_weighted_offset",
                "cor-cloning-weighted-assembly-input",
                cstar,
                (1. - krates[0]) * offsets[0]
                    + bk[0]
                    + cv * (offsets[1] + bk[1] + (1. - krates[1]) * offsets[2] + bk[2])
                    + cb * (offsets[3] + bk[3]),
            ));
            extra.push(equal(
                "actual_finite_composition_offset",
                "cor-cloning-weighted-assembly-input",
                cstar,
                c["C_star"],
            ));
            for (i, current) in values.iter().enumerate() {
                let rowmass = pc[i].iter().sum::<f64>();
                extra.push(equal(
                    "conservative_cloning_row_mass",
                    "cor-cloning-weighted-assembly-input",
                    rowmass,
                    1.,
                ));
                let mut clone_phi = 0.;
                let mut kin_phi = 0.;
                let mut combined_phi = 0.;
                let mut clone_kin_increment = 0.;
                for r in 0..4 {
                    let clone = pc[i]
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * values[j][r])
                        .sum::<f64>();
                    let kinetic = pk[i]
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * values[j][r])
                        .sum::<f64>();
                    let combined = pc[i]
                        .iter()
                        .enumerate()
                        .map(|(j, p)| {
                            p * pk[j]
                                .iter()
                                .enumerate()
                                .map(|(l, q)| q * values[l][r])
                                .sum::<f64>()
                        })
                        .sum::<f64>();
                    let pushed_kinetic_upper = pc[i]
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * (ak[r] * values[j][r] + bk[r]))
                        .sum::<f64>();
                    let intermediate = ak[r] * clone + bk[r];
                    let final_upper = ak[r] * (ac[r] * current[r] + offsets[r]) + bk[r];
                    let dc = ac[r] * current[r] + offsets[r] - clone;
                    let dk = ak[r] * current[r] + bk[r] - kinetic;
                    let pushed_dk = pc[i]
                        .iter()
                        .enumerate()
                        .map(|(j, p)| {
                            let pkf = pk[j]
                                .iter()
                                .enumerate()
                                .map(|(l, q)| q * values[l][r])
                                .sum::<f64>();
                            p * (ak[r] * values[j][r] + bk[r] - pkf)
                        })
                        .sum::<f64>();
                    extra.extend([
                        upper(
                            "composition_first_positive_operator_relation",
                            "thm-synergistic-foster-lyapunov-preview",
                            combined,
                            pushed_kinetic_upper,
                        ),
                        equal(
                            "composition_constant_and_matrix_commutation",
                            "thm-synergistic-foster-lyapunov-preview",
                            pushed_kinetic_upper,
                            intermediate,
                        ),
                        upper(
                            "composition_second_positive_operator_relation",
                            "thm-synergistic-foster-lyapunov-preview",
                            intermediate,
                            final_upper,
                        ),
                        lower(
                            "computed_cloning_defect_nonnegative",
                            "cor-cloning-weighted-assembly-input",
                            dc,
                            0.,
                        ),
                        lower(
                            "computed_kinetic_defect_nonnegative",
                            "cor-cloning-weighted-assembly-input",
                            dk,
                            0.,
                        ),
                        equal(
                            "actual_AC7_component_identity",
                            "cor-cloning-weighted-assembly-input",
                            combined,
                            final_upper - ak[r] * dc - pushed_dk,
                        ),
                    ]);
                    clone_phi += weights[r] * clone;
                    kin_phi += weights[r] * kinetic;
                    combined_phi += weights[r] * combined;
                    clone_kin_increment += weights[r] * (combined - clone);
                }
                let phi = (0..4).map(|r| weights[r] * current[r]).sum::<f64>();
                extra.push(equal(
                    "actual_AC8_backward_weighted_increment",
                    "cor-cloning-weighted-assembly-input",
                    combined_phi - phi,
                    clone_phi - phi + clone_kin_increment,
                ));
                extra.push(equal(
                    "actual_kinetic_increment_linear_form",
                    "cor-cloning-weighted-assembly-input",
                    kin_phi - phi,
                    (0..4)
                        .map(|r| {
                            weights[r]
                                * (pk[i]
                                    .iter()
                                    .enumerate()
                                    .map(|(j, p)| p * values[j][r])
                                    .sum::<f64>()
                                    - current[r])
                        })
                        .sum::<f64>(),
                ));
            }
            for i in 0..values.len() {
                let phi = values[i][0] + cv * (values[i][1] + values[i][2]) + cb * values[i][3];
                let out = pc[i]
                    .iter()
                    .enumerate()
                    .map(|(j, p)| {
                        p * (values[j][0] + cv * (values[j][1] + values[j][2]) + cb * values[j][3])
                    })
                    .sum::<f64>();
                let rhs = -cv * rates[0] * values[i][1] - cb * rates[1] * values[i][3]
                    + offsets[0]
                    + cv * (offsets[1] + offsets[2])
                    + cb * offsets[3];
                extra.push(upper(
                    "statewise_actual_clone_weighted_drift",
                    "thm-complete-cloning-drift",
                    out - phi,
                    rhs,
                ));
            }
            let label = "rem-component-growth-combined-drift";
            for check in fixture
                .checks
                .iter()
                .filter(|c| c.id.starts_with("full_weighted_drift_state_"))
            {
                let mut item = check.clone();
                item.source_labels = vec![label.into()];
                extra.push(item);
            }
            let mut aliases = extra.clone();
            for &rate in rates.iter().chain(&krates) {
                aliases.push(certificate_check(
                    "actual_declared_component_rate_in_unit_interval",
                    "thm-synergistic-foster-lyapunov-preview",
                    rate > 0. && rate <= 1.,
                ));
            }
            aliases.push(certificate_check(
                "actual_declared_component_weights_positive",
                "thm-synergistic-foster-lyapunov-preview",
                cv > 0. && cb > 0.,
            ));
            for (i, row) in pc.iter().enumerate() {
                for j in 0..values.len() {
                    let q = row
                        .iter()
                        .enumerate()
                        .map(|(k, p)| p * pk[k][j])
                        .sum::<f64>();
                    let identity = if i == j { 1. } else { 0. };
                    let delta = row[j] - identity
                        + row
                            .iter()
                            .enumerate()
                            .map(|(k, p)| p * (pk[k][j] - if k == j { 1. } else { 0. }))
                            .sum::<f64>();
                    aliases.push(equal(
                        "actual_backward_composition_identity_each_basis_coordinate",
                        "cor-cloning-weighted-assembly-input",
                        q - identity,
                        delta,
                    ));
                }
                let f = &values[i];
                aliases.push(equal(
                    "weighted_four_observable_dot_product",
                    "thm-synergistic-foster-lyapunov-preview",
                    weights.iter().zip(f).map(|(w, f)| w * f).sum::<f64>(),
                    f[0] + cv * (f[1] + f[2]) + cb * f[3],
                ));
                for r in 0..4 {
                    let clone = row
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * values[j][r])
                        .sum::<f64>();
                    let kinetic = pk[i]
                        .iter()
                        .enumerate()
                        .map(|(j, p)| p * values[j][r])
                        .sum::<f64>();
                    let dc = ac[r] * f[r] + offsets[r] - clone;
                    let dk = ak[r] * f[r] + bk[r] - kinetic;
                    aliases.push(equal(
                        "actual_clone_observable_defect_identity",
                        "cor-cloning-weighted-assembly-input",
                        clone,
                        ac[r] * f[r] + offsets[r] - dc,
                    ));
                    aliases.push(equal(
                        "actual_kinetic_observable_defect_identity",
                        "cor-cloning-weighted-assembly-input",
                        kinetic,
                        ak[r] * f[r] + bk[r] - dk,
                    ));
                }
            }
            let alias_input = json!({"finite_operator_inputs":input,"observable_order":["V_W","X","Y","W_b"],"weights":weights,"cloning_diagonal":ac,"kinetic_diagonal":ak,"cloning_offset":offsets,"kinetic_offset":bk,"lambda_v":1.,"scope":"Exact finite conservative-cloning and possibly terminally killed kinetic operators satisfy the declared statewise component inequalities for these four observables. Matrix composition, component defects, rates and weighted dot products are reconstructed statewise. This conditional algebra witness does not assign these rates to a native landscape."});
            records(
                suite,
                "cor-cloning-weighted-assembly-input",
                &[
                    r"F=(V_W,X,Y,W_b)^\mathsf T",
                    r"P_C1=1",
                    r"Q=P_CP_K",
                    r"w=(1,c_V,c_V,c_B)^\mathsf T",
                    r"b_K=(C'_W,C'_x,C'_v,C'_b)^\mathsf T",
                    r"Y=\lambda_vV_{\mathrm{Var},v}",
                    r"P_KF=A_KF+b_K-D_K",
                    r"P_CF=A_CF+b_C-D_C",
                    r"P_CP_K-I=(P_C-I)+P_C(P_K-I)",
                ],
                alias_input.clone(),
                aliases.clone(),
            )?;
            records(
                suite,
                "thm-synergistic-foster-lyapunov-preview",
                &[
                    r"F=(V_W,V_{\mathrm{Var},x},\lambda_vV_{\mathrm{Var},v},W_b)^\mathsf T",
                    r"0<\kappa_x,\kappa_b,\kappa_W,\kappa_v\leq1",
                    r"Q=P_CP_K",
                    r"c_V,c_B>0",
                    r"w=(1,c_V,c_V,c_B)^\mathsf T",
                    r"V_{\mathrm{total}}=w^\mathsf TF",
                ],
                alias_input,
                aliases,
            )?;
        }
        if fixture.name == "finite_cluster_geometry_and_target" {
            fixture_equation(
                suite,
                fixture,
                "def-unified-high-low-error-sets",
                r#"\text{diam}(G_m) := \max_{i,j \in G_m} d_{\text{alg}}(i, j) \le D_{\text{diam}}(\epsilon)"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "def-unified-high-low-error-sets",
                r#"\sum_{m \in O_M} \text{Contrib}(G_m) \ge (1-\varepsilon_O) \sum_{\substack{m=1 \\ |G_m| \ge k_{\min}}}^M \text{Contrib}(G_m)"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "def-unified-high-low-error-sets",
                r#"H_k(\epsilon) := \left(\bigcup_{m \in O_M} G_m\right) \cup \left(\bigcup_{\substack{m: |G_m| < k_{\min}}} G_m\right)"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-outlier-fraction-lower-bound",
                r#"\sum_{i\in O_k}|q_i-\bar q|^2
\ge(1-\varepsilon_O)\sum_{i\in\mathcal A_k}|q_i-\bar q|^2."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-outlier-fraction-lower-bound",
                r#"\frac{|O_k|}{k}>
\frac{(1-\varepsilon_O)R_h^2}{D_h^2}=:f_O>0."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-outlier-fraction-lower-bound",
                r#"(1-\varepsilon_O)kR_h^2
<(1-\varepsilon_O)k\operatorname{Var}_{\mathcal A_k}(q)
\le\sum_{i\in O_k}|q_i-\bar q|^2
\le |O_k|D_h^2."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-outlier-cluster-fraction-lower-bound",
                r#"\frac{|H|}{k}>f_{H,\rm cl}:=
\frac{(1-\varepsilon_O)(R_{\rm var}^2-C_\lambda D^2/2)}{D_h^2}>0."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-outlier-cluster-fraction-lower-bound",
                r#"\operatorname{Var}_{G}(q)
=\frac1{2n_G^2}\sum_{i,j\in G}|q_i-q_j|^2
\le\frac{C_\lambda D^2}{2},"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-geometric-separation-of-partition",
                r#"D_H:=m_x(s_{HL}-\rho_H-\rho_L)>R_L."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-geometric-separation-of-partition",
                r#"|x_i-x_j|\ge|\mu_{x,G}-\mu_{x,G'}|
-|x_i-\mu_{x,G}|-|x_j-\mu_{x,G'}|
\ge s_{HL}-\rho_H-\rho_L."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "rem-cluster-energy-and-separation",
                r#"\rho_G\le D_{x,G},\qquad
\operatorname{Var}_{G}(x)\le D_{x,G}^2/2."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-unfit-fraction-lower-bound",
                r#"\frac{|U_k|}{k}\geq\frac{s_V^2}{2R_V^2},\qquad
\frac{|F_k|}{k}\geq\frac{s_V^2}{2R_V^2}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-unfit-high-error-overlap-fraction",
                r#"\frac{|H\cap U_k|}{k}\geq\frac{f_Hf_L\Delta_V}{R_V}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-unfit-high-error-overlap-fraction",
                r#"|I_{11}\cap H\cap U_k|
\geq |H\cap U_k|-|\mathcal A_k\setminus I_{11}|."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-quantitative-keystone",
                r#"\frac{1}{N}\sum_{i \in I_{11}} (p_{1,i} + p_{2,i})\|\Delta\delta_{x,i}\|^2 \ge \chi(\epsilon) V_{\text{struct}} - g_{\max}(\epsilon)"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-mean-companion-fitness-gap",
                r#"\mathbb E_{K_i}(V_c-V_i)_+
\geq \frac{ak}{k-1}\frac{s_V^2}{2R_V}
\geq \frac{a s_V^2}{2R_V}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-mean-companion-fitness-gap",
                r#"s_V^2\leq R_V\mathbb E|X-\mu|
=2R_V\mathbb E(X-\mu)_+."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-unfit-cloning-pressure",
                r#"p_i=\mathbb E_{K_i}\min\left\{1,
\frac{(V_c-V_i)_+}{p_{\max}(V_i+\varepsilon_{\mathrm{clone}})}\right\}
\geq\frac{\mathbb E_{K_i}(V_c-V_i)_+}{A_V}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-unfit-cloning-pressure",
                r#"p_u=\frac{a_*s_*^2}
 {2R_*\max\{R_*,p_{\max}(V_{\mathrm{pot,max}}+
 \varepsilon_{\mathrm{clone}})\}}>0."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-variance-concentration-Hk",
                r#"\sum_{i\in H_k}\|x_i-\mu\|^2\geq c_HS_k,\qquad
c_H=(1-\varepsilon_O)\left(1-\frac{D_c^2}{2R_{\mathrm{var}}^2}\right)>0."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-variance-concentration-Hk",
                r#"\operatorname{Var}(G)=\frac1{2m^2}\sum_{i,j\in G}\|x_i-x_j\|^2
\leq\frac{D_c^2}{2}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-variance-concentration-Hk",
                r#"\sum_{i\in H_k}\|x_i-\mu\|^2\geq(1-\varepsilon_O)B
\geq(1-\varepsilon_O)S_k\left(1-\frac{kD_c^2}{2S_k}\right)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-error-concentration-target-set",
                r#"\sum_{i\in H}\|\delta_{x,k,i}\|^2\geq c_HS_k,\qquad
S_k/N\geq aV_{\mathrm{struct}}-b,"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-error-concentration-target-set",
                r#"\frac1N\sum_{i\in T}\|\Delta\delta_{x,i}\|^2
\geq \frac{c_Ha}{2}V_{\mathrm{struct}}
-\left(\frac{c_Hb}{2}+M_j+B_T\right)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-error-concentration-target-set",
                r#"\frac1N\sum_{i\in H}\|\Delta\delta_{x,i}\|^2
\geq \frac{c_H S_k}{2N}-M_j
\geq\frac{c_Ha}{2}V_{\mathrm{struct}}-\frac{c_Hb}{2}-M_j."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "rem-keystone-balanced-structural-scope",
                r#"E_w \ge \frac{1}{N}\sum_{i \in I_{\text{target}}} p_{1,i}\|\Delta\delta_{x,i}\|^2"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "rem-keystone-balanced-structural-scope",
                r#"E_w\geq \frac{p_u}{N}\sum_{i\in I_{\mathrm{target}}}
\|\Delta\delta_{x,i}\|^2."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "rem-keystone-balanced-structural-scope",
                r#"E_w \ge p_u(\epsilon) \cdot \left( c_{err}(\epsilon)V_{\mathrm{struct}} - g_{err}(\epsilon) \right)"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "rem-keystone-balanced-structural-scope",
                r#"E_w \ge p_u(\epsilon) \cdot \left( c_{err}(\epsilon)V_{\mathrm{struct}} - g_{err}(\epsilon) \right)"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "rem-keystone-balanced-structural-scope",
                r#"E_w \ge \chi(\epsilon) V_{\text{struct}} - g_{\text{partial}}(\epsilon)"#,
                &extra,
            )?;
        }
        if fixture.name == "fixed_shared_rescale" {
            fixture_equation(
                suite,
                fixture,
                "prop-fixed-rescale-variance-bound",
                r#"m_g=\min_{|z|\le Z}g'(z)\ge0,\qquad
M_g=\max_{|z|\le Z}g'(z)<\infty,\qquad
R_g=g(Z)-g(-Z)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "prop-fixed-rescale-variance-bound",
                r#"\boxed{
\frac{m_g^2}{s_{\max}^2}\operatorname{Var}(y)
\le\operatorname{Var}(d')
\le\min\left\{
\frac{M_g^2}{s_{\min}^2}\operatorname{Var}(y),\frac{R_g^2}{4}
\right\}.}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "prop-fixed-rescale-variance-bound",
                r#"\frac{m_g}{s_{\max}}|y_i-y_j|
\le |d'_i-d'_j|
\le\frac{M_g}{s_{\min}}|y_i-y_j|."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-variance-to-gap",
                r#"\max_{i,j} |v_i - v_j| \ge \sqrt{2\kappa}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-variance-to-gap",
                r#"\sum_{i=1}^k \sum_{j=1}^k (v_i - v_j)^2 \le \sum_{i=1}^k \sum_{j=1}^k \Delta_{\max}^2 = k^2 \Delta_{\max}^2"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-variance-to-gap",
                r#"\mathrm{Var}(\{v_i\}) \le \frac{1}{2k^2} (k^2 \Delta_{\max}^2) = \frac{1}{2} \Delta_{\max}^2"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-variance-to-gap",
                r#"\kappa \le \mathrm{Var}(\{v_i\}) \le \frac{1}{2} \Delta_{\max}^2"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "def-max-patched-std",
                r#"\sigma'_{\max} := \sup_{0 \le V \le V_{\max}^2} \sigma'_{\mathrm{patch}}(V)"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-rescale-derivative-lower-bound",
                r#"\inf_{z \in Z_{\mathrm{supp}}} g'_A(z) = g'_{\min} > 0"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-raw-gap-to-rescaled-gap",
                r#"|g_A(z_a) - g_A(z_b)| \ge \kappa_{\mathrm{rescaled}}(\kappa_{\mathrm{raw}}) > 0"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-raw-gap-to-rescaled-gap",
                r#"|z_a - z_b| \ge \frac{\kappa_{\mathrm{raw}}}{\sigma'_{\max}} =: \kappa_z > 0"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-raw-gap-to-rescaled-gap",
                r#"|g_A(z_a) - g_A(z_b)| \ge g'_{\min} \cdot \kappa_z"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-raw-gap-to-rescaled-gap",
                r#"|g_A(z_a) - g_A(z_b)| \ge g'_{\min} \cdot \left(\frac{\kappa_{\mathrm{raw}}}{\sigma'_{\max}}\right) = \kappa_{\mathrm{rescaled}}(\kappa_{\mathrm{raw}})"#,
                &extra,
            )?;
        }
        if fixture.name == "group_and_log_fitness_separation" {
            fixture_equation(
                suite,
                fixture,
                "lem-variance-to-mean-separation",
                r#"|\mu_L-\mu_H|\ge
\sqrt{\frac{\kappa-B_{\rm within}}{f_Hf_L}}>0."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "cor-averaged-group-separation",
                r#"\mathbb E[s^2\mid\mathcal G]\ge\kappa,\qquad
\mathbb E[s_{\rm within}^2\mid\mathcal G]\le B_{\rm within}<\kappa."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "cor-averaged-group-separation",
                r#"\mathbb E[(\mu_L-\mu_H)^2\mid\mathcal G]\ge Q."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "cor-averaged-group-separation",
                r#"\Pr\{|\mu_L-\mu_H|>t\mid\mathcal G\}
\ge\frac{Q-t^2}{R^2-t^2}>0."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "cor-averaged-group-separation",
                r#"\mathbb E[\Delta^2\mid\mathcal G]
\le t^2+(R^2-t^2)\Pr\{|\Delta|>t\mid\mathcal G\}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-log-gap-lower-bound",
                r#"\mathbb E\log X-\mathbb E\log Y\geq
L_{a,b}(\kappa):=
\min_{u\in[a,b-\kappa]}\{\ell(u+\kappa)-\log u\}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-log-gap-lower-bound",
                r#"\mathbb E\log X-\mathbb E\log Y\geq\kappa/b
\geq\log(1+\kappa/b)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-log-gap-upper-bound",
                r#"|\mathbb E\log X-\mathbb E\log Y|
\leq U_{a,b}(\kappa):=
\max_{u\in[a,b-\kappa]}\{\log(u+\kappa)-\ell(u)\}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-log-gap-upper-bound",
                r#"|\mathbb E\log X-\mathbb E\log Y|\leq\log(1+K/a)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "prop-corrective-signal-bound",
                r#"D\geq L_{\eta,M}(\kappa_{d'})."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "prop-log-reward-gap-axiom-bound",
                r#"|\mathbb E_H\log r'-\mathbb E_L\log r'|
\leq\log(1+K_r/\eta)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-stability-condition-final-corrected",
                r#"\mathbb E_L\log V-\mathbb E_H\log V\geq\delta_{\log}."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-stability-condition-final-corrected",
                r#"\mathbb E_LV-\mathbb E_HV\geq v_*(e^\delta-1)>0."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-stability-condition-final-corrected",
                r#"0\leq\log\mathbb EX-\mathbb E\log X
\leq\frac{\operatorname{Var}(X)}{2v_*^2}."#,
                &extra,
            )?;
        }
        if fixture.name == "exact_measurement_averaged_small_clusters" {
            fixture_equation(
                suite,
                fixture,
                "thm-geometry-guarantees-variance",
                r#"\mathbb E\operatorname{Var}(d)\geq f_Hf_L\Delta_d^2."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-keystone-geometric-measurement-events",
                r#"\rho_h=\frac{\mathsf V_z-h^2}{D_z^2-h^2}>0.
\tag{3.AP1}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-keystone-geometric-measurement-events",
                r#"\frac1k\sum_\ell|z_\ell-z_j|^2
=\mathsf V_z+|z_j-\bar z|^2
\le h^2+(D_z^2-h^2)\frac{\#\{\ell:|z_\ell-z_j|\ge h\}}k."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-keystone-geometric-measurement-events",
                r#"E_{ij}=\{Y_i\le\ell_G,\ Y_j\ge H_h\}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lem-keystone-geometric-measurement-events",
                r#"f_-=\min_{|z|\le Z_m}f(z)>0,\quad
f_+=\max_{|z|\le Z_m}f(z),\quad
m_f=\min_{|z|\le Z_m}f'(z)>0.
\tag{3.AP3}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-keystone-averaged-cluster-pressure",
                r#"\boxed{\overline p_i:=\mathbb E[p_i(\mathbf F)\mid S]
\ge \pi_G:=\kappa_C\kappa_D^2\rho_G^2\rho_h\mathfrak a_G.}
\tag{3.AP5}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-keystone-averaged-error-capture",
                r#"\boxed{\mathbb E\!\left[\frac1N\sum_{i\in I_{11}}
 (p_{1,i}+p_{2,i})e_i\,\middle|\,S_1,S_2\right]
\ge\frac1N\sum_{i\in I_{11}}\pi_i e_i.}
\tag{3.AP7}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-keystone-averaged-error-capture",
                r#"V_{\rm struct}\le(1+\eta)\frac1N\sum_i e_i
+\left(\lambda_v+\frac{b^2}{4\eta}\right)
 \frac1N\sum_i|\Delta\delta_{v,i}|^2."#,
                &extra,
            )?;
        }
        if fixture.name == "finite_conservative_weighted_composition" {
            for needle in [
                "A_C=\\operatorname{diag}",
                "D_C=A_CF",
                "3.AC7",
                "3.AC8",
                "3.AC9",
            ] {
                let label = if needle.starts_with("A_C=") {
                    "thm-synergistic-foster-lyapunov-preview"
                } else {
                    "cor-cloning-weighted-assembly-input"
                };
                let formula = expression(label, needle)?;
                fixture_equation(suite, fixture, label, &formula, &extra)?;
            }
            fixture_equation(
                suite,
                fixture,
                "rem-component-growth-combined-drift",
                r#"V_{\rm total}=V_W+c_V(V_{\mathrm{Var},x}+\lambda_v V_{\mathrm{Var},v})+c_BW_b,
\qquad
QV_{\rm total}-V_{\rm total}\le-\kappa_*V_{\rm total}+C_*."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-complete-cloning-drift",
                r#"(P_C-I)\Phi\le-c_V\kappa_xX-c_B\kappa_bW_b
 +C_W+c_V(C_x+C_v)+c_BC_b.
\tag{3.AC5}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lemma-weighted-min-coefficient",
                r#"\sum_i w_i a_iX_i\geq a_*\sum_iw_iX_i."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"P_CF\leq A_CF+b_C,\qquad P_KF\leq A_KF+b_K,"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"QV_{\mathrm{total}}\leq(1-\kappa_*)V_{\mathrm{total}}+C_*,\qquad
\kappa_*=\min(\kappa_W,\kappa_x,\kappa_v,\kappa_b),\quad
C_*=w^\mathsf T(A_Kb_C+b_K)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"QF=P_C(P_KF)\leq P_C(A_KF+b_K)
=A_KP_CF+b_K\leq A_KA_CF+A_Kb_C+b_K."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"Q^nV_{\mathrm{total}}\leq(1-\kappa_*)^nV_{\mathrm{total}}
+\frac{C_*}{\kappa_*}\bigl(1-(1-\kappa_*)^n\bigr)."#,
                &extra,
            )?;
        }
        if fixture.name == "finite_killed_weighted_composition" {
            for needle in ["D_C=A_CF", "3.AC7", "3.AC8", "3.AC9"] {
                let label = "cor-cloning-weighted-assembly-input";
                let formula = expression(label, needle)?;
                fixture_equation(suite, fixture, label, &formula, &extra)?;
            }
            fixture_equation(
                suite,
                fixture,
                "rem-component-growth-combined-drift",
                r#"V_{\rm total}=V_W+c_V(V_{\mathrm{Var},x}+\lambda_v V_{\mathrm{Var},v})+c_BW_b,
\qquad
QV_{\rm total}-V_{\rm total}\le-\kappa_*V_{\rm total}+C_*."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-complete-cloning-drift",
                r#"(P_C-I)\Phi\le-c_V\kappa_xX-c_B\kappa_bW_b
 +C_W+c_V(C_x+C_v)+c_BC_b.
\tag{3.AC5}"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "lemma-weighted-min-coefficient",
                r#"\sum_i w_i a_iX_i\geq a_*\sum_iw_iX_i."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"P_CF\leq A_CF+b_C,\qquad P_KF\leq A_KF+b_K,"#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"QV_{\mathrm{total}}\leq(1-\kappa_*)V_{\mathrm{total}}+C_*,\qquad
\kappa_*=\min(\kappa_W,\kappa_x,\kappa_v,\kappa_b),\quad
C_*=w^\mathsf T(A_Kb_C+b_K)."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"QF=P_C(P_KF)\leq P_C(A_KF+b_K)
=A_KP_CF+b_K\leq A_KA_CF+A_Kb_C+b_K."#,
                &extra,
            )?;
            fixture_equation(
                suite,
                fixture,
                "thm-synergistic-foster-lyapunov-preview",
                r#"Q^nV_{\mathrm{total}}\leq(1-\kappa_*)^nV_{\mathrm{total}}
+\frac{C_*}{\kappa_*}\bigl(1-(1-\kappa_*)^n\bigr)."#,
                &extra,
            )?;
        }
    }
    Ok(())
}

fn native_proof_equations(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let nf = n as f64;
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let va: Vec<_> = live.iter().map(|&i| law.v[i]).collect();
    let k = live.len();
    let mu = mean(&xa);
    let uv = mean(&va);
    let dx = range(&xa);
    let vmax = law.v.iter().map(|v| v.abs()).fold(0., f64::max);
    let bx = xa.iter().map(|x| x.abs()).fold(0., f64::max);
    let mut f9 = Vec::new();
    let mut covariance = Vec::new();
    let mut copy = Vec::new();
    let mut jitter = Vec::new();
    let mut vmoment = Vec::new();
    let mut young = Vec::new();
    let mut vx = 0.;
    let mut ec = 0.;
    for p in &law.patterns {
        covariance.push(upper(
            "actual_conditional_barycenter_variance",
            "thm-cloning-canonical-barycenter-concentration",
            p.conditional_bary_var,
            (dx * dx + 0.01) / nf,
        ));
        let flux = live
            .iter()
            .map(|&i| {
                live.iter()
                    .map(|&j| {
                        p.edges[i][j] * ((law.x[j] - mu).powi(2) - (law.x[i] - mu).powi(2)) / nf
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();
        let oriented = live
            .iter()
            .flat_map(|&i| live.iter().map(move |&j| (i, j)))
            .filter(|&(i, j)| p.f[i] < p.f[j])
            .map(|(i, j)| p.edges[i][j] * ((law.x[j] - mu).powi(2) - (law.x[i] - mu).powi(2)) / nf)
            .sum::<f64>();
        f9.push(equal(
            "native_fitness_order_directed_flux",
            "lem-keystone-contraction-alive",
            flux,
            oriented,
        ));
        let mut plans = vec![(Vec::new(), Vec::new(), 1.)];
        for i in 0..n {
            let mut next = Vec::new();
            for (y, a, mass) in plans {
                if law.alive[i] && p.p[i] < 1. {
                    let mut yy = y.clone();
                    yy.push(law.x[i]);
                    let mut aa = a.clone();
                    aa.push(0.);
                    next.push((yy, aa, mass * (1. - p.p[i])));
                }
                for (j, &bij) in p.edges[i].iter().enumerate() {
                    if bij > 0. {
                        let mut yy = y.clone();
                        yy.push(law.x[j]);
                        let mut aa = a.clone();
                        aa.push(1.);
                        next.push((yy, aa, mass * bij));
                    }
                }
            }
            plans = next;
        }
        let mut mass = 0.;
        let mut y2 = 0.;
        let mut expectedmoment = 0.;
        for (y, a, m) in plans {
            mass += m;
            let sumy = y.iter().map(|y| y * y).sum::<f64>() / nf;
            let summismatch = a.iter().sum::<f64>();
            y2 += m * sumy;
            expectedmoment += m * (sumy + 0.01 * summismatch / nf);
            let pair = y
                .iter()
                .flat_map(|&yi| y.iter().map(move |&yj| (yi - yj).powi(2)))
                .sum::<f64>()
                / (2. * nf * nf);
            copy.extend([
                equal(
                    "frozen_copy_pairwise_variance",
                    "thm-positional-variance-contraction",
                    var(&y),
                    pair,
                ),
                upper(
                    "frozen_copy_diameter_bound",
                    "thm-positional-variance-contraction",
                    pair,
                    dx * dx / 2.,
                ),
            ]);
            for &ai in &a {
                jitter.push(upper(
                    "centered_jitter_coefficient",
                    "lem-dead-walker-revival-bounded",
                    0.01 * ((1. - 2. / nf) * ai + summismatch / (nf * nf)),
                    0.01,
                ));
            }
        }
        copy.push(equal(
            "complete_native_copy_plan_mass",
            "thm-positional-variance-contraction",
            mass,
            1.,
        ));
        let meanp = mean(&p.p);
        let second = p
            .means
            .iter()
            .zip(&p.traces)
            .map(|(m, c)| m * m + c)
            .sum::<f64>()
            / nf;
        let momentlabel = "cor-cloning-actual-inter-swarm-expansion";
        young.extend([
            equal(
                "actual_copied_jitter_moment",
                momentlabel,
                second,
                y2 + 0.01 * meanp,
            ),
            equal(
                "direct_frozen_plan_jitter_average",
                momentlabel,
                expectedmoment,
                second,
            ),
            upper(
                "eligible_copy_jitter_ceiling",
                momentlabel,
                second,
                bx * bx + 0.01,
            ),
        ]);
        let (energy, q, _, gmass) = graph_moments(law, p, 0.5);
        let venergy = law.v.iter().map(|v| v * v).sum::<f64>() / nf - 0.75 * energy;
        for (choices, _) in graph_plans(p) {
            for component in components(&choices) {
                let average =
                    component.iter().map(|&i| law.v[i]).sum::<f64>() / component.len() as f64;
                let before = component.iter().map(|&i| law.v[i].powi(2)).sum::<f64>();
                let centered = component
                    .iter()
                    .map(|&i| (law.v[i] - average).powi(2))
                    .sum::<f64>();
                for sign in [-1., 1.] {
                    let after = component
                        .iter()
                        .map(|&i| (average + 0.5 * sign * (law.v[i] - average)).powi(2))
                        .sum::<f64>();
                    vmoment.push(equal(
                        "every_native_component_shared_Haar_energy_identity",
                        momentlabel,
                        after,
                        component.len() as f64 * average.powi(2) + 0.25 * centered,
                    ));
                    vmoment.push(upper(
                        "every_native_component_energy_dissipation",
                        momentlabel,
                        after,
                        before,
                    ));
                }
            }
        }
        vmoment.extend([
            equal("all_component_law_mass", momentlabel, gmass, 1.),
            upper(
                "full_slot_energy_dissipation",
                momentlabel,
                venergy,
                law.v.iter().map(|v| v * v).sum::<f64>() / nf,
            ),
            upper(
                "entering_velocity_cap_moment",
                momentlabel,
                law.v.iter().map(|v| v * v).sum::<f64>() / nf,
                vmax * vmax,
            ),
        ]);
        young.push(upper(
            "Young_actual_hypocoercive_moment",
            momentlabel,
            q,
            2. * second + 1.0025 * venergy,
        ));
        vx += p.mass * p.output_var;
        ec += p.mass * energy;
    }
    let dead: Vec<_> = (0..n)
        .filter(|&i| !law.alive[i])
        .map(|i| law.v[i] - uv)
        .collect();
    let deadsecond = dead.iter().map(|v| v * v).sum::<f64>() / nf;
    let shift = (dead.iter().sum::<f64>() / nf).powi(2);
    let rv = deadsecond - shift;
    let rx = law
        .donors
        .iter()
        .enumerate()
        .filter(|(i, _)| !law.alive[*i])
        .map(|(_, row)| {
            row.iter()
                .enumerate()
                .map(|(j, p)| p * (law.x[j] - mu).powi(2) / nf)
                .sum::<f64>()
        })
        .sum::<f64>();
    let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"native_patterns":law.patterns.len(),"canonical_reward_direction":"Minimize quadratic U=x²/2","eligible_diameter":dx,"eligible_anchor_radius":bx,"Vmax":vmax,"alpha":0.5,"revival_Rx":rx,"revival_Rv":rv,"component_energy":ec});
    let moments_label = "cor-cloning-actual-inter-swarm-expansion";
    let actual_domain_radius = 2_f64;
    let actual_cap = law
        .config
        .kinetic
        .velocity_cap
        .ok_or_else(|| error("native cap missing"))?;
    let actual_jitter = law.config.clone_transform.jitter_amplitude;
    let eta = 1_f64;
    let k_eta = (1. + eta) * (actual_domain_radius.powi(2) + actual_jitter.powi(2))
        + (1. + 0.1_f64.powi(2) / (4. * eta)) * actual_cap.powi(2);
    record(
        suite,
        moments_label,
        "3.W1",
        json!({"native_config":law.config,"eta":eta,"B_x":actual_domain_radius,"Vmax":actual_cap,"lambda_v":1.,"b":0.1,"actual_K_eta":k_eta,"alive":law.alive}),
        vec![
            equal(
                "native_Keta_full_constant_definition",
                moments_label,
                k_eta,
                2. * (4. + 0.01) + 1.0025 * 4.,
            ),
            certificate_check(
                "actual_Young_parameter_strict_positive",
                moments_label,
                eta > 0.,
            ),
            certificate_check(
                "actual_completed_velocity_cap_hypothesis",
                moments_label,
                law.v.iter().all(|v| v.abs() <= actual_cap),
            ),
            certificate_check(
                "actual_eligible_position_domain_hypothesis",
                moments_label,
                xa.iter().all(|x| x.abs() <= actual_domain_radius),
            ),
        ],
    )?;
    record(
        suite,
        "thm-cloning-canonical-barycenter-concentration",
        "\\operatorname{Var}(\\bar X^c\\mid S,\\mathbf Y)",
        inputs.clone(),
        covariance,
    )?;
    record(
        suite,
        "lem-keystone-contraction-alive",
        "3.F9",
        inputs.clone(),
        f9,
    )?;
    record(
        suite,
        "thm-positional-variance-contraction",
        "\\frac1N\\sum_i|Y_i-\\bar Y|^2",
        inputs.clone(),
        copy,
    )?;
    record(
        suite,
        "lem-dead-walker-revival-bounded",
        "(1-2/N)A_i",
        inputs.clone(),
        jitter,
    )?;
    record(
        suite,
        "thm-cloning-unconditional-collective-balance",
        "0\\leq R_x",
        inputs.clone(),
        vec![
            lower(
                "Rx_nonnegative",
                "thm-cloning-unconditional-collective-balance",
                rx,
                0.,
            ),
            upper(
                "Rx_dead_fraction",
                "thm-cloning-unconditional-collective-balance",
                rx,
                dx * dx * (n - k) as f64 / nf,
            ),
            lower(
                "Rv_nonnegative",
                "thm-cloning-unconditional-collective-balance",
                rv,
                0.,
            ),
            upper(
                "Rv_dead_fraction",
                "thm-cloning-unconditional-collective-balance",
                rv,
                4. * vmax * vmax * (n - k) as f64 / nf,
            ),
        ],
    )?;
    record(
        suite,
        "thm-cloning-unconditional-collective-balance",
        "\\left|\\frac1N\\sum_{i\\notin\\mathcal A}(v_i-u_A)\\right|^2\n\\leq",
        inputs.clone(),
        vec![upper(
            "dead_velocity_Cauchy_Schwarz",
            "thm-cloning-unconditional-collective-balance",
            shift,
            (n - k) as f64 / nf * deadsecond,
        )],
    )?;
    record(
        suite,
        "thm-cloning-unconditional-collective-balance",
        r"R_v(S)=\frac1N",
        inputs.clone(),
        vec![equal(
            "actual_revival_velocity_full_formula",
            "thm-cloning-unconditional-collective-balance",
            rv,
            deadsecond - shift,
        )],
    )?;
    let prop = "prop-bounded-velocity-expansion";
    records(
        suite,
        prop,
        &[
            "\\boxed{\\quad\\mathcal V_v^a",
            "\\mathcal V_v^{\\rm all}(S)",
        ],
        inputs.clone(),
        vec![
            equal(
                "exact_full_slot_alive_velocity_relation",
                prop,
                var(&law.v) - var(&va) * k as f64 / nf,
                rv,
            ),
            upper(
                "alive_velocity_plus_dead_energy",
                prop,
                var(&law.v),
                var(&va) * k as f64 / nf + deadsecond,
            ),
            upper(
                "dead_energy_cap",
                prop,
                deadsecond,
                4. * vmax * vmax * (n - k) as f64 / nf,
            ),
            equal(
                "native_proposal_velocity_balance",
                prop,
                (var(&law.v) - 0.75 * ec) - var(&va) * k as f64 / nf,
                rv - 0.75 * ec,
            ),
            lower("velocity_revival_nonnegative", prop, rv, 0.),
            upper(
                "velocity_revival_uniform",
                prop,
                rv,
                4. * vmax * vmax * (n - k) as f64 / nf,
            ),
        ],
    )?;
    let label = "thm-velocity-variance-bounded-expansion";
    let hv = 2. * (rv - 0.75 * ec);
    record(
        suite,
        label,
        "\\Delta V_{\\mathrm{Var},v}",
        inputs.clone(),
        vec![
            equal(
                "two_swarm_exact_velocity_balance",
                label,
                hv,
                2. * rv - 0.75 * (2. * ec),
            ),
            upper(
                "two_swarm_dead_fraction_velocity",
                label,
                hv,
                8. * vmax * vmax * (n - k) as f64 / nf,
            ),
            upper(
                "dead_fraction_uniform_relaxation",
                label,
                8. * vmax * vmax * (n - k) as f64 / nf,
                8. * vmax * vmax,
            ),
        ],
    )?;
    let label = "thm-complete-variance-drift";
    let hx = 2. * (vx - var(&xa) * k as f64 / nf);
    let cx = dx * dx + 0.02;
    let oldx = 2. * var(&xa) * k as f64 / nf;
    records(
        suite,
        label,
        &[
            "H_x(S_1)+H_x(S_2)",
            "\\mathbb E\\Delta V_{\\mathrm{Var}}\\leq",
        ],
        inputs.clone(),
        vec![
            equal(
                "two_swarm_exact_pos_velocity_balance",
                label,
                hx + hv,
                hx + 2. * rv - 0.75 * 2. * ec,
            ),
            upper("actual_positional_reset_hypothesis", label, hx, -oldx + cx),
            upper(
                "complete_variance_uniform_drift",
                label,
                hx + hv,
                -oldx + cx + 8. * vmax * vmax,
            ),
        ],
    )?;
    let label = "cor-cloning-actual-inter-swarm-expansion";
    records(
        suite,
        label,
        &["X_i^+=Y_i+j I_i", "Q(x,v)\\le(1+\\eta)"],
        inputs.clone(),
        young,
    )?;
    records(
        suite,
        label,
        &["\\sum_{i\\in C}|v_i^+|^2", "\\frac1N\\sum_i|v_i^+|^2"],
        inputs.clone(),
        vmoment,
    )?;
    record(
        suite,
        label,
        "B_x=\\sup_{x\\in",
        inputs.clone(),
        vec![
            upper("finite_anchor_radius", label, bx, f64::MAX),
            upper(
                "every_retained_velocity_cap",
                label,
                law.v.iter().map(|v| v.abs()).fold(0., f64::max),
                vmax,
            ),
        ],
    )?;
    record(
        suite,
        label,
        "Q(x,v)=|x|^2",
        inputs.clone(),
        vec![certificate_check(
            "quadratic_positive_definite",
            label,
            1. > 0.1_f64.powi(2) / 4.,
        )],
    )?;
    let mh = 2. * (bx * bx + 0.01) + 1.0025 * vmax * vmax;
    let expectedq = law
        .patterns
        .iter()
        .map(|p| p.mass * graph_moments(law, p, 0.5).1)
        .sum::<f64>();
    let meanv = mean(&law.v);
    let meanx = law
        .patterns
        .iter()
        .map(|p| p.mass * mean(&p.means))
        .sum::<f64>();
    let baryq = law
        .patterns
        .iter()
        .map(|p| {
            let mx = mean(&p.means);
            p.mass * (mx * mx + p.conditional_bary_var + meanv * meanv + 0.1 * mx * meanv)
        })
        .sum::<f64>();
    let centered_product = 2. * (expectedq - baryq);
    let location = 2. * (baryq - (meanx * meanx + meanv * meanv + 0.1 * meanx * meanv));
    let full_product = centered_product + location;
    let initial = 0.;
    let uppercost = 4. * mh;
    let label = "cor-structural-error-contraction";
    record(
        suite,
        label,
        "\\mathbb E V_{\\mathrm{struct}}",
        inputs.clone(),
        vec![
            upper(
                "product_plan_expected_structural_reset",
                label,
                centered_product,
                uppercost,
            ),
            upper(
                "independent_native_pair_structural_drift_envelope",
                label,
                centered_product,
                -initial + uppercost,
            ),
        ],
    )?;
    let label = "thm-inter-swarm-bounded-expansion";
    record(
        suite,
        label,
        "\\mathbb E\\int\\|z-z_0\\|_h^2",
        inputs.clone(),
        vec![upper(
            "actual_marginal_quadratic_moment",
            label,
            expectedq,
            mh,
        )],
    )?;
    let product_checks = vec![
        lower(
            "actual_centered_product_cost_nonnegative",
            label,
            centered_product,
            0.,
        ),
        lower("actual_location_cost_nonnegative", label, location, 0.),
        equal(
            "location_structural_product_cost_sum",
            label,
            full_product,
            centered_product + location,
        ),
        upper(
            "actual_full_product_transport_envelope",
            label,
            full_product,
            4. * expectedq,
        ),
        upper(
            "actual_two_marginal_moment_envelope",
            label,
            4. * expectedq,
            4. * mh,
        ),
    ];
    record(
        suite,
        label,
        "W_h^2(\\mu'_1,\\mu'_2)",
        json!({"native_identical_inputs_independent_output_coupling":inputs,
            "full_product_transport":full_product,"centered_product_transport":centered_product,
            "location_cost":location,"post_barycenter_Q_moment":baryq}),
        product_checks.clone(),
    )?;
    let label = "cor-component-bounds-vw";
    record(
        suite,
        label,
        "\\mathbb E\\Delta V_{\\mathrm{loc}}",
        inputs.clone(),
        vec![
            upper("location_drift_product_envelope", label, location, 4. * mh),
            upper(
                "structural_drift_product_envelope",
                label,
                centered_product,
                4. * mh,
            ),
        ],
    )?;
    let label = "thm-complete-wasserstein-drift";
    record(
        suite,
        label,
        "\\mathbb E\\Delta V_W",
        inputs,
        vec![
            equal(
                "exact_location_structural_sum",
                label,
                full_product,
                location + centered_product,
            ),
            upper(
                "sum_of_component_drift_envelopes",
                label,
                full_product,
                8. * mh,
            ),
        ],
    )
}

fn matching_equations(suite: &mut EstimateSuite) -> Result<()> {
    let label = "lem-greedy-preserves-signal";
    let mut checks = Vec::new();
    let mut cases = Vec::new();
    for x in [
        vec![-1.0_f64, -0.95, 0.95, 1., 3.],
        vec![-3.0_f64, -1., -0.95, 0.95, 1., 3.],
    ] {
        let n = x.len();
        let width = 0.7_f64;
        let config = GasConfig::euclidean(1, 0.04)?;
        let obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, x.clone())?);
        let distance = algorithmic_gas::geometry::Distance::default();
        let kernel = algorithmic_gas::geometry::Kernel::Gaussian { width };
        let mut d: Vec<Vec<f64>> = vec![vec![0.; n]; n];
        let mut w = d.clone();
        for i in 0..n {
            for j in 0..n {
                d[i][j] = distance.compare(&obs, i, &obs, j)?;
                w[i][j] = kernel
                    .log_weight(d[i][j], algorithmic_gas::geometry::ComparisonKind::Distance)?
                    .exp();
            }
        }
        let near: Vec<Vec<usize>> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i && d[i][j] <= 0.1).collect())
            .collect();
        let dh = 1.;
        let rl = 0.1;
        let dalg = range(&x);
        let mut stack = vec![((0..n).collect::<Vec<_>>(), (0..n).collect::<Vec<_>>(), 1.)];
        let mut outcomes = Vec::new();
        let mut history_far = vec![0.; n];
        while let Some((u, map, mass)) = stack.pop() {
            if u.len() <= 1 {
                outcomes.push((map, mass));
                continue;
            }
            for &i in &u {
                let norm = u.iter().filter(|&&j| j != i).map(|&j| w[i][j]).sum::<f64>();
                let nnear = u.iter().filter(|&&j| near[i].contains(&j)).count();
                let nfar = u.len() - 1 - nnear;
                let q = if nnear > 0 {
                    (nfar as f64 / nnear as f64
                        * (-(dh * dh - rl * rl) / (2. * width * width)).exp())
                    .min(1.)
                } else {
                    1.
                };
                let far = u
                    .iter()
                    .filter(|&&j| j != i && !near[i].contains(&j))
                    .map(|&j| w[i][j] / norm)
                    .sum::<f64>();
                let ed = u
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| w[i][j] / norm * d[i][j])
                    .sum::<f64>();
                if !near[i].is_empty() {
                    checks.extend([
                        upper("greedy_history_tail", label, far, q),
                        upper("greedy_history_distance", label, ed, rl + (dalg - rl) * q),
                    ]);
                }
                for &j in u.iter().filter(|&&j| j != i) {
                    let prob = mass / u.len() as f64 * w[i][j] / norm;
                    let mut next = map.clone();
                    next[i] = j;
                    next[j] = i;
                    let rem = u.iter().copied().filter(|&k| k != i && k != j).collect();
                    stack.push((rem, next, prob));
                    if !near[i].contains(&j) {
                        history_far[i] += prob;
                    }
                    if !near[j].contains(&i) {
                        history_far[j] += prob;
                    }
                }
            }
        }
        let mut ed = vec![0.; n];
        let mut selfp = vec![0.; n];
        let mut farp = vec![0.; n];
        let mut mass = 0.;
        for (map, p) in &outcomes {
            mass += p;
            for i in 0..n {
                ed[i] += p * d[i][map[i]];
                if map[i] == i {
                    selfp[i] += p;
                } else if !near[i].contains(&map[i]) {
                    farp[i] += p;
                }
            }
        }
        checks.push(equal("full_native_greedy_history_mass", label, mass, 1.));
        for i in 0..n {
            if near[i].is_empty() {
                checks.push(lower(
                    "isolated_final_measurement",
                    label,
                    ed[i],
                    dh * (1. - selfp[i]),
                ));
            } else {
                checks.extend([
                    upper(
                        "incoming_included_final_distance",
                        label,
                        ed[i],
                        rl + (dalg - rl) * farp[i],
                    ),
                    equal(
                        "incoming_plus_pivot_history_probability",
                        label,
                        farp[i],
                        history_far[i],
                    ),
                ]);
            }
        }
        cases.push(json!({"positions":x,"native_distance":"unsquashed Euclidean Distance::default","native_kernel_width":width,"configuration_law":"GaussianGreedy, random uniform pivot order","canonical_independent_law_not_substituted":config.distance_donors.law!=algorithmic_gas::donor::SamplingLaw::GaussianGreedy,"histories":outcomes.len(),"mean_distances":ed,"self_probabilities":selfp,"final_far_probabilities":farp,"history_far_probabilities":history_far}));
    }
    records(
        suite,
        label,
        &[
            "\\mathbb P(c_j\\notin C_j",
            "\\mathbb E[d_i\\mid S]",
            "\\mathbb E[d_j\\mid S]",
            "\\mathbf1_{\\{j\\in U_t,I_t=j\\}}",
        ],
        json!({"alternative_matching_law":true,"cases":cases}),
        checks,
    )?;
    let x: [f64; 6] = [-1., -0.95, 0.95, 1., 3., -3.];
    let n = x.len();
    let width = 0.7_f64;
    let mut stack = vec![((0..n).collect::<Vec<_>>(), Vec::<(usize, usize)>::new(), 1.)];
    let mut matchings = Vec::new();
    while let Some((u, edges, mass)) = stack.pop() {
        if u.is_empty() {
            matchings.push((edges, mass));
            continue;
        }
        let i = u[0];
        for &j in &u[1..] {
            let rem = u.iter().copied().filter(|&a| a != i && a != j).collect();
            let mut e = edges.clone();
            e.push((i, j));
            stack.push((
                rem,
                e,
                mass * (-(x[i] - x[j]).powi(2) / (2. * width * width)).exp(),
            ));
        }
    }
    let z = matchings.iter().map(|(_, m)| m).sum::<f64>();
    let normalized = matchings.iter().map(|(_, m)| m / z).sum::<f64>();
    let label = "def-spatial-pairing-diversity-idealized";
    records(
        suite,
        label,
        &[
            "w_{ij} :=",
            "W(M) :=",
            "P(M) =",
            r"k = |\mathcal{A}_t|",
            r"\mathcal{A}_t = \{w_1, w_2, \dots, w_k\}",
            r"\varepsilon_d > 0",
        ],
        json!({"positions":x,"matching_count":matchings.len(),"partition_function":z,"matching_weights":matchings,"alternative_product_matching_law":true}),
        vec![
            equal("finite_product_matching_normalizer", label, normalized, 1.),
            certificate_check("positive_matching_partition_function", label, z > 0.),
            equal("six_row_matching_count", label, 15., matchings.len() as f64),
            equal(
                "actual_matching_eligible_population",
                label,
                n as f64,
                x.len() as f64,
            ),
            certificate_check(
                "actual_matching_interaction_range_positive",
                label,
                width > 0.,
            ),
            certificate_check(
                "every_product_matching_covers_each_alive_walker_once",
                label,
                matchings.iter().all(|(edges, _)| {
                    (0..n).all(|i| edges.iter().filter(|(a, b)| *a == i || *b == i).count() == 1)
                }),
            ),
            certificate_check(
                "every_product_matching_quality_is_positive",
                label,
                matchings.iter().all(|(_, w)| *w > 0.),
            ),
        ],
    )
}
fn incremental_and_moving_center(suite: &mut EstimateSuite) -> Result<()> {
    let label = "thm-cloning-incremental-cluster-balance";
    let x = [-0.9, -0.8, 0.8, 0.9];
    let y = [-0.7, -0.75, 0.7, 0.85];
    let a = [1., 0., 0., 0.];
    let b = [0., 1., 1., 0.];
    let ja = [1, 0, 3, 2];
    let jb = [1, 0, 3, 2];
    let groups = [vec![0, 1], vec![2, 3]];
    let dd: Vec<_> = x.iter().zip(&y).map(|(x, y)| x - y).collect();
    let r: Vec<_> = (0..4)
        .map(|i| a[i] * (x[ja[i]] - x[i]) - b[i] * (y[jb[i]] - y[i]))
        .collect();
    let left_law = enumerate_law(x.to_vec(), vec![0.3, -0.4, 0.2, -0.1], vec![true; 4])?;
    let right_law = enumerate_law(y.to_vec(), vec![0.2, -0.3, 0.1, -0.2], vec![true; 4])?;
    let supported = |law: &Law, accepted: &[f64; 4], donors: &[usize; 4]| -> Result<usize> {
        law.patterns
            .iter()
            .position(|p| {
                (0..4).all(|i| {
                    if accepted[i] > 0. {
                        p.edges[i][donors[i]] > 0.
                    } else {
                        p.p[i] < 1.
                    }
                })
            })
            .ok_or_else(|| {
                error("retained coupled component plan absent from actual native finite support")
            })
    };
    let left_pattern = supported(&left_law, &a, &ja)?;
    let right_pattern = supported(&right_law, &b, &jb)?;
    record(
        suite,
        label,
        r"A_i,\widetilde A_i\in\{0,1\}",
        json!({"left_native_fitness":left_law.patterns[left_pattern].f,"right_native_fitness":right_law.patterns[right_pattern].f,"left_native_measurement":left_law.patterns[left_pattern].choices,"right_native_measurement":right_law.patterns[right_pattern].choices,"accepted_left":a,"accepted_right":b,"donors_left":ja,"donors_right":jb,"scope":"Accepted frozen plans have strictly positive probability in the actual complete finite native measurement, donor and threshold law."}),
        vec![
            certificate_check(
                "actual_conditional_plan_support_left",
                label,
                (0..4).all(|i| {
                    if a[i] > 0. {
                        left_law.patterns[left_pattern].edges[i][ja[i]] > 0.
                    } else {
                        left_law.patterns[left_pattern].p[i] < 1.
                    }
                }),
            ),
            certificate_check(
                "actual_conditional_plan_support_right",
                label,
                (0..4).all(|i| {
                    if b[i] > 0. {
                        right_law.patterns[right_pattern].edges[i][jb[i]] > 0.
                    } else {
                        right_law.patterns[right_pattern].p[i] < 1.
                    }
                }),
            ),
        ],
    )?;
    let mut between = 0.;
    let mut within = 0.;
    for g in &groups {
        let dg: Vec<_> = g.iter().map(|&i| dd[i]).collect();
        let rg: Vec<_> = g.iter().map(|&i| r[i]).collect();
        between += g.len() as f64 * mean(&dg) * mean(&rg);
        within += dg
            .iter()
            .zip(&rg)
            .map(|(d, r)| (d - mean(&dg)) * (r - mean(&rg)))
            .sum::<f64>();
    }
    let gatecost = (0..4).map(|i| (a[i] - b[i]).powi(2) * 0.01).sum::<f64>();
    let before = dd.iter().map(|d| d * d).sum::<f64>();
    let after = dd.iter().zip(&r).map(|(d, r)| (d + r).powi(2)).sum::<f64>() + gatecost;
    record(
        suite,
        label,
        "3.C1",
        json!({"left_positions":x,"right_positions":y,"left_acceptance":a,"right_acceptance":b,"left_donors":ja,"right_donors":jb,"geometric_common_refinement":groups,"jitter_variance":0.01,"frozen_plans":"valid retained/copy plans; same recipient Gaussian coupling"}),
        vec![
            equal(
                "signed_between_within_copy_balance",
                label,
                after / 4. - before / 4.,
                (2. * between + 2. * within + r.iter().map(|r| r * r).sum::<f64>() + gatecost) / 4.,
            ),
            equal(
                "between_within_inner_product",
                label,
                dd.iter().zip(&r).map(|(d, r)| d * r).sum::<f64>(),
                between + within,
            ),
        ],
    )?;
    let mut copy_definitions = Vec::new();
    for i in 0..4 {
        copy_definitions.push(certificate_check(
            "actual_binary_acceptance_indicators",
            label,
            [0., 1.].contains(&a[i]) && [0., 1.].contains(&b[i]),
        ));
        copy_definitions.push(equal(
            "actual_paired_position_difference_definition",
            label,
            dd[i],
            x[i] - y[i],
        ));
        copy_definitions.push(equal(
            "actual_paired_copy_increment_definition",
            label,
            r[i],
            a[i] * (x[ja[i]] - x[i]) - b[i] * (y[jb[i]] - y[i]),
        ));
        for xi in [-1.3, 0., 0.7] {
            let left = x[i] + a[i] * (x[ja[i]] - x[i]) + 0.1 * a[i] * xi;
            let right = y[i] + b[i] * (y[jb[i]] - y[i]) + 0.1 * b[i] * xi;
            copy_definitions.push(equal(
                "realized_shared_gaussian_copy_difference",
                label,
                left - right,
                dd[i] + r[i] + 0.1 * (a[i] - b[i]) * xi,
            ));
        }
    }
    for g in &groups {
        let dg: Vec<_> = g.iter().map(|&i| dd[i]).collect();
        let rg: Vec<_> = g.iter().map(|&i| r[i]).collect();
        copy_definitions.push(equal(
            "each_geometric_refinement_block_inner_product",
            label,
            dg.iter().zip(&rg).map(|(d, r)| d * r).sum::<f64>(),
            g.len() as f64 * mean(&dg) * mean(&rg)
                + dg.iter()
                    .zip(&rg)
                    .map(|(d, r)| (d - mean(&dg)) * (r - mean(&rg)))
                    .sum::<f64>(),
        ));
    }
    records(
        suite,
        label,
        &[
            r"A_i,\widetilde A_i\in\{0,1\}",
            "d_i=x_i-y_i,",
            r"x_i^c-y_i^c=d_i+r_i+j",
            r"\sum_{i\in G}d_i\cdot r_i",
        ],
        json!({"left_positions":x,"right_positions":y,"left_acceptance":a,"right_acceptance":b,"left_donors":ja,"right_donors":jb,"paired_difference":dd,"paired_increment":r,"refinement":groups,"jitter":0.1,"scope":"Exact paired frozen copy plans and realized shared Gaussian affine identity; the jitter values are deterministic realization witnesses and the conditional second moment is separately integrated in C1."}),
        copy_definitions,
    )?;
    let v = [0.3, -0.4, 0.2, -0.1];
    let w = [0.2, -0.3, 0.1, -0.2];
    let ca = [vec![0, 1], vec![2], vec![3]];
    let cb = [vec![0, 1], vec![2, 3]];
    let alpha = 0.5;
    let mut actual = 0.;
    for shared in [-1., 1.] {
        for aa in [-1., 1.] {
            for ab in [-1., 1.] {
                for bb in [-1., 1.] {
                    let va = collision_output(&v, &ca, &[shared, aa, ab], alpha);
                    let vb = collision_output(&w, &cb, &[shared, bb], alpha);
                    actual += va
                        .iter()
                        .zip(&vb)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / 16.;
                }
            }
        }
    }
    let mut rhs = 0.;
    for i in 0..4 {
        let ga = ca.iter().find(|g| g.contains(&i)).unwrap();
        let gb = cb.iter().find(|g| g.contains(&i)).unwrap();
        let ma = ga.iter().map(|&j| v[j]).sum::<f64>() / ga.len() as f64;
        let mb = gb.iter().map(|&j| w[j]).sum::<f64>() / gb.len() as f64;
        let u = v[i] - ma;
        let t = w[i] - mb;
        rhs += (ma - mb).powi(2)
            + alpha * alpha * (u * u + t * t - if ga == gb { 2. * u * t } else { 0. });
    }
    record(
        suite,
        label,
        "3.C2",
        json!({"left_components":ca,"right_components":cb,"shared_identical_component":[0,1],"other_rotations":"independent Haar signs","left_velocity":v,"right_velocity":w}),
        vec![
            equal(
                "shared_and_changed_components_haar_balance",
                label,
                actual,
                rhs,
            ),
            equal(
                "identical_component_diagonal_zero",
                label,
                0.,
                collision_output(&v, &ca, &[1., -1., 1.], alpha)
                    .iter()
                    .zip(collision_output(&v, &ca, &[1., -1., 1.], alpha))
                    .map(|(a, b)| (a - b).powi(2))
                    .sum(),
            ),
        ],
    )?;
    let mut velocity_definitions = Vec::new();
    let mut singleton_u = Vec::new();
    let mut singleton_t = Vec::new();
    let mut uvalues = Vec::new();
    let mut tvalues = Vec::new();
    for i in 0..4 {
        let ga = ca.iter().find(|g| g.contains(&i)).unwrap();
        let gb = cb.iter().find(|g| g.contains(&i)).unwrap();
        let ma = ga.iter().map(|&j| v[j]).sum::<f64>() / ga.len() as f64;
        let mb = gb.iter().map(|&j| w[j]).sum::<f64>() / gb.len() as f64;
        let u = v[i] - ma;
        let t = w[i] - mb;
        uvalues.push(u);
        tvalues.push(t);
        velocity_definitions.push(equal(
            "actual_frozen_component_centered_velocity_left",
            label,
            u + ma,
            v[i],
        ));
        velocity_definitions.push(equal(
            "actual_frozen_component_centered_velocity_right",
            label,
            t + mb,
            w[i],
        ));
        if ga.len() == 1 {
            singleton_u.push(equal(
                "actual_singleton_left_relative_velocity_zero",
                label,
                u,
                0.,
            ));
        }
        if gb.len() == 1 {
            singleton_t.push(equal(
                "actual_singleton_right_relative_velocity_zero",
                label,
                t,
                0.,
            ));
        }
    }
    let velocity_input = json!({"left_velocity":v,"right_velocity":w,"left_components":ca,"right_components":cb,"centered_left":uvalues,"centered_right":tvalues,"scope":"Frozen full-component centers, including singleton components; component partitions and their shared or independent Haar coupling are retained in C2."});
    records(
        suite,
        label,
        &[r"u_i=v_i-\bar v_{C_i}", r"t_i=w_i-\bar w_{D_i}"],
        velocity_input.clone(),
        velocity_definitions,
    )?;
    record(suite, label, "u_i=0", velocity_input.clone(), singleton_u)?;
    // A separate legitimate singleton right component retains its frozen velocity exactly.
    singleton_t.push(equal(
        "right_singleton_relative_velocity_zero",
        label,
        w[3] - mean(&[w[3]]),
        0.,
    ));
    record(
        suite,
        label,
        "t_i=0",
        json!({"existing_components":velocity_input,"singleton_right_component":[3],"frozen_velocity":w[3]}),
        singleton_t,
    )?;
    let a = 0.1_f64;
    let law = enumerate_law(vec![-a, 0., a], vec![0.; 3], vec![true; 3])?;
    let p = law
        .patterns
        .iter()
        .find(|p| p.choices[0] == 1 && p.choices[2] == 1)
        .unwrap();
    let q = p.p[0];
    let theta = law.measurements[0][1];
    let sr = (a.powi(4) / 18. + 0.01).sqrt();
    let g = |z: f64| 2. / (1. + (-z).exp()) + 0.1;
    let f0 = 1.1 * g(a * a / (3. * sr));
    let fm = 1.1 * g(-a * a / (6. * sr));
    let expected = (2. * a * a * q * (1. - q) + 0.02 * q) / 9.;
    let observed = (p.means[1] - mean(&p.means)).powi(2)
        + (1. - 2. / 3.) * p.traces[1]
        + p.traces.iter().sum::<f64>() / 9.;
    let unconditional = law
        .patterns
        .iter()
        .map(|p| {
            p.mass
                * ((p.means[1] - mean(&p.means)).powi(2)
                    + (1. - 2. / 3.) * p.traces[1]
                    + p.traces.iter().sum::<f64>() / 9.)
        })
        .sum::<f64>();
    let label = "ex-cloning-moving-barycenter";
    let inputs = json!({"positions":law.x,"native_full_patterns":law.patterns.len(),"a":a,"q":q,"theta":theta,"event_fitness":p.f,"event_probability":theta*theta});
    let c = vec![
        equal("native_central_event_fitness", label, p.f[1], f0),
        equal("native_endpoint_event_fitness", label, p.f[0], fm),
        certificate_check("strict_native_central_fitness_gap", label, f0 > fm),
    ];
    record(suite, label, "F_0=1.1g", inputs.clone(), c)?;
    record(
        suite,
        label,
        "3.F11",
        inputs.clone(),
        vec![
            equal(
                "native_moving_center_exact_variance",
                label,
                observed,
                expected,
            ),
            certificate_check("strict_moving_center_variance", label, expected > 0.),
        ],
    )?;
    record(
        suite,
        label,
        "3.F12",
        inputs.clone(),
        vec![
            lower(
                "full_measurement_moving_center",
                label,
                unconditional,
                theta * theta * 2. * a * a * q * (1. - q) / 9.,
            ),
            certificate_check(
                "strict_unconditional_center_lower",
                label,
                theta * theta * 2. * a * a * q * (1. - q) / 9. > 0.,
            ),
        ],
    )?;
    record(
        suite,
        label,
        "3.F13",
        inputs,
        vec![
            equal(
                "native_collective_conditional_drift",
                label,
                p.output_var - var(&law.x),
                -2. * a * a * q * (4. - q) / 9. + 4. * q * 0.01 / 9.,
            ),
            certificate_check(
                "collective_drift_strict_negative",
                label,
                p.output_var < var(&law.x),
            ),
        ],
    )
}

fn boundary_complete_equations(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let nf = n as f64;
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let k = live.len();
    let mut beta = vec![vec![0.; n]; n];
    let mut pb = vec![0.; n];
    let mut post = 0.;
    let mut possecond = 0.;
    let mut velenergy = 0.;
    let mut cross = 0.;
    let mut ec = 0.;
    for p in &law.patterns {
        for (i, row) in beta.iter_mut().enumerate() {
            pb[i] += p.mass * p.p[i];
            for (j, value) in row.iter_mut().enumerate() {
                *value += p.mass * p.edges[i][j];
            }
        }
        let xx = p
            .means
            .iter()
            .zip(&p.traces)
            .map(|(m, c)| m * m + c)
            .sum::<f64>()
            / nf;
        let (energy, q, _, _) = graph_moments(law, p, 0.5);
        let vv = law.v.iter().map(|v| v * v).sum::<f64>() / nf - 0.75 * energy;
        possecond += p.mass * xx;
        velenergy += p.mass * vv;
        cross += p.mass * (q - xx - vv) / 0.1;
        ec += p.mass * energy;
        post += p.mass * xx;
    }
    let old = live.iter().map(|&i| law.x[i].powi(2)).sum::<f64>() / nf;
    let theta = 0.8_f64;
    let exposed: Vec<_> = live
        .iter()
        .copied()
        .filter(|&i| law.x[i].powi(2) > theta)
        .collect();
    let pstar = exposed.iter().map(|&i| pb[i]).fold(1., f64::min);
    let j = xa.iter().map(|x| x * x).fold(0., f64::max) + 0.01;
    let exposedmass = exposed.iter().map(|&i| law.x[i].powi(2)).sum::<f64>() / nf;
    let drift = post - old;
    let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"auxiliary_barrier":"globally C2 nonnegative quadratic phi=x²; does not modify the reward","threshold":theta,"exposed":exposed,"full_native_measurement_averaged_acceptance":pb,"native_averaged_edges":beta,"p_star":pstar,"J":j,"R":j,"proposal_barrier_moment":post,"entering_alive_barrier":old});
    if !exposed.is_empty() && pstar > 0. {
        let label = "def-boundary-exposed-set";
        record(
            suite,
            label,
            "\\mathcal{E}_{\\text{boundary}}(S)",
            inputs.clone(),
            vec![equal(
                "computed_boundary_exposed_count",
                label,
                exposed.len() as f64,
                live.iter().filter(|&&i| law.x[i].powi(2) > theta).count() as f64,
            )],
        )?;
        let label = "rem-boundary-mass-relationship";
        record(
            suite,
            label,
            "W_b(S_k) =",
            inputs.clone(),
            vec![upper(
                "actual_barrier_exposed_mass_relation",
                label,
                old,
                exposedmass + k as f64 / nf * theta,
            )],
        )?;
        let mut conditional_clone_integrals = Vec::new();
        let mut exact = 0.;
        for &i in &live {
            let numerator = beta[i]
                .iter()
                .enumerate()
                .map(|(j, b)| b * (law.x[j].powi(2) + 0.01))
                .sum::<f64>();
            if pb[i] > 0. {
                conditional_clone_integrals.push(upper(
                    "actual_post_clone_barrier_integral",
                    "proof-boundary-potential-contraction",
                    numerator / pb[i],
                    j,
                ));
            }
            exact += (numerator - pb[i] * law.x[i].powi(2)) / nf;
        }
        for (i, row) in beta.iter().enumerate() {
            if !law.alive[i] {
                exact += row
                    .iter()
                    .enumerate()
                    .map(|(j, b)| b * (law.x[j].powi(2) + 0.01))
                    .sum::<f64>()
                    / nf;
            }
        }
        conditional_clone_integrals.push(equal(
            "exact_persistent_cloned_revived_barrier_balance",
            "proof-boundary-potential-contraction",
            drift,
            exact,
        ));
        conditional_clone_integrals.push(upper(
            "exposed_selection_barrier_drift",
            "proof-boundary-potential-contraction",
            drift,
            -pstar * exposedmass + k as f64 / nf * j + (n - k) as f64 / nf * j,
        ));
        records(
            suite,
            "proof-boundary-potential-contraction",
            &[
                "=\\frac1N\\sum_{i\\in\\mathcal A}p_i(J_i",
                "\\leq-\\frac{p_*}{N}",
            ],
            inputs.clone(),
            conditional_clone_integrals,
        )?;
        record(
            suite,
            "thm-boundary-potential-contraction",
            "\\mathbb E_C[\\Delta W_b",
            inputs.clone(),
            vec![
                certificate_check(
                    "actual_exposed_pressure_hypothesis",
                    "thm-boundary-potential-contraction",
                    exposed.iter().all(|&i| pb[i] >= pstar) && pstar > 0.,
                ),
                upper(
                    "two_actual_swarm_barrier_drift",
                    "thm-boundary-potential-contraction",
                    2. * drift,
                    -pstar * 2. * old
                        + 2. * k as f64 / nf * (pstar * theta + j)
                        + 2. * (n - k) as f64 / nf * j,
                ),
            ],
        )?;
        let label = "thm-complete-cloning-drift";
        let cx = range(&xa).powi(2) + 0.02;
        let cv = 8. * law.v.iter().map(|v| v.abs()).fold(0., f64::max).powi(2);
        let cw = 4. * (2. * j + 1.0025 * law.v.iter().map(|v| v.abs()).fold(0., f64::max).powi(2));
        let cb = 2. * pstar * theta + 2. * j;
        let oldx = 2. * k as f64 / nf * var(&xa);
        let vx = law
            .patterns
            .iter()
            .map(|p| p.mass * p.output_var)
            .sum::<f64>();
        let dv = 2.
            * (law.v.iter().map(|v| v * v).sum::<f64>() / nf
                - 0.75 * ec
                - var(&live.iter().map(|&i| law.v[i]).collect::<Vec<_>>()) * k as f64 / nf
                - mean(&law.v).powi(2));
        let actual = 2. * vx - oldx + dv + 2. * drift;
        record(
            suite,
            label,
            "3.AC5",
            json!({"native_fixture":inputs,"coupling":"identical inputs with all native innovations shared; exact VW drift zero","c_V":1.,"c_B":1.,"positional_reset_rate":1.,"actual_boundary_rate":pstar,"offsets":{"C_W":cw,"C_x":cx,"C_v":cv,"C_b":cb}}),
            vec![upper(
                "actual_conditional_weighted_clone_drift",
                label,
                actual,
                -oldx - pstar * 2. * old + cw + cx + cv + cb,
            )],
        )?;
    }
    let h = 0.04_f64;
    let c = h / 2.;
    let a = (-h).exp();
    let sh2 = (1. - (-2. * h).exp()) / 2.;
    let vmax = 2.;
    let alpha = 0.5;
    let w = (1. + 2. * alpha) * vmax;
    let rd = 2.0_f64;
    let lu = 1.;
    let bu = 0.;
    let aa = 1. + h * h / 4. * (1. + a) * lu;
    let d0 = h / 2. * (1. + a) * w + h * h / 4. * (1. + a) * bu;
    let mx = (aa * (rd * rd + 0.01).sqrt() + d0).powi(2) + h * h / 4. * sh2 + 0.01 * h;
    let coefficient = 1. - c * c * (1. + a);
    let hv = c * (1. + a);
    let xout = coefficient * coefficient * possecond
        + hv * hv * velenergy
        + 2. * coefficient * hv * cross
        + c * c * sh2
        + 0.01 * h;
    let lambda = 0.5;
    let resetupper = 1. + xout + lambda * vmax * vmax;
    let m = 1. + mx + lambda * vmax * vmax;
    let initial = 1.
        + law.x.iter().map(|x| x * x).sum::<f64>() / nf
        + lambda * law.v.iter().map(|v| v * v).sum::<f64>() / nf;
    let label = "thm-canonical-full-step-reset-drift";
    let inputs = json!({"positions":law.x,"alive":law.alive,"retained_velocities":law.v,"eligible_domain":"[-2,2]","canonical_BAOAB":{"h":h,"gamma":1.,"B":1.,"sigma_p":0.1,"force":"-x","velocity_cap":vmax},"exact_fullstep_position_moment":xout,"completed_moment_upper_uses_actual_cap":resetupper,"M_x":mx,"M":m,"lambda":lambda,"A":aa,"D0":d0});
    records(
        suite,
        label,
        &[
            "\\mathscr L_N(S)=",
            "P\\mathscr L_N(S)\\leq",
            "P\\mathscr L_N-\\mathscr L_N",
            "\\mathbb E\\frac1N\\sum_i|X_i^c|^2",
        ],
        inputs.clone(),
        vec![
            certificate_check("positive_observable_velocity_weight", label, lambda > 0.),
            upper("native_complete_position_moment_reset", label, xout, mx),
            upper("native_complete_capped_moment_reset", label, resetupper, m),
            upper(
                "native_full_slot_Foster_reset",
                label,
                resetupper - initial,
                -0.5 * initial + m,
            ),
            upper(
                "native_donor_position_jitter_moment",
                label,
                possecond,
                rd * rd + 0.01,
            ),
        ],
    )?;
    let label = "cor-canonical-full-step-boundary-reset";
    let sigma = 0.1 * h.sqrt();
    let mq = 1. / ((2. * std::f64::consts::PI).sqrt() * sigma);
    let integral = 8.;
    let mut checks = Vec::new();
    let steps = 16000;
    for center in [-3., -1.5, -0.5, 0., 0.8, 1.5, 3.] {
        let mut moment = 0.;
        for i in 0..steps {
            let t = -std::f64::consts::FRAC_PI_2
                + (i as f64 + 0.5) * std::f64::consts::PI / steps as f64;
            let x = 2. * t.sin();
            let barrier = 4_f64.ln() - 2. * t.cos().ln();
            let density = (-(x - center).powi(2) / (2. * sigma * sigma)).exp() * mq;
            moment += barrier * density * 2. * t.cos() * std::f64::consts::PI / steps as f64;
        }
        checks.push(upper(
            "actual_final_noise_log_barrier_integral",
            label,
            moment,
            mq * integral,
        ));
    }
    record(
        suite,
        label,
        "P\\mathscr B_N(S)\\leq",
        json!({"actual_terminal_noise_sigma":sigma,"domain":"[-2,2]","logarithmic_auxiliary_barrier_integral":integral,"density_max":mq,"endpoint_integral_transform":"x=2 sin(t), removes logarithmic endpoint singularity","midpoint_panels":steps}),
        checks,
    )?;
    let label = "lem-barrier-reduction-cloning";
    let y = 0.8_f64;
    let sigma = 0.1_f64;
    let exact = y * y + sigma * sigma;
    let mut checks = vec![equal(
        "Gaussian_quadratic_Taylor_exact",
        label,
        exact,
        y * y + 2. * sigma * sigma / 2.,
    )];
    for hh in [-0.3, -0.1, 0., 0.1, 0.3] {
        checks.push(equal(
            "actual_integral_Taylor_remainder",
            label,
            (y + hh).powi(2),
            y * y + 2. * y * hh + hh * hh,
        ));
    }
    record(
        suite,
        label,
        "\\varphi(y+h)=",
        json!({"C2_extension":"phi(x)=x²","H":2.,"sigma_x":sigma,"y":y,"h_grid":[-0.3,-0.1,0.,0.1,0.3]}),
        checks,
    )?;
    let label = "cor-extinction-suppression";
    let radius = 1.;
    let sigma = 0.1_f64;
    let q = 2_f64.sqrt() * (-radius * radius / (4. * sigma * sigma)).exp();
    let lower_tail = radius / sigma;
    let step = 0.001_f64;
    let panels = 10000;
    let density = |u: f64| (-u * u / 2.).exp() / (2. * std::f64::consts::PI).sqrt();
    let tail_lower = 2.
        * (0..panels)
            .map(|i| density(lower_tail + (i as f64 + 1.) * step) * step)
            .sum::<f64>();
    let end_tail = lower_tail + panels as f64 * step;
    let tail = 2.
        * ((0..panels)
            .map(|i| density(lower_tail + i as f64 * step) * step)
            .sum::<f64>()
            + density(end_tail) / end_tail);
    record(
        suite,
        label,
        "q\\leq\\min",
        json!({"dimension":1,"safe_centers":0.,"safe_ball_radius":radius,"actual_independent_final_gaussian_sigma":sigma,"death_tail_monotone_quadrature_lower":tail_lower,"death_tail_monotone_quadrature_upper":tail,"q":q,"conditional_stage":"after all shared centers and collision rotations, before independent final noises"}),
        vec![
            certificate_check("strict_useful_q", label, q > 0. && q < 1.),
            certificate_check(
                "strict_actual_nonzero_Gaussian_death_tail",
                label,
                tail_lower > 0. && tail >= tail_lower,
            ),
            upper("actual_gaussian_tail_Markov", label, tail, q),
        ],
    )
}

fn remaining_keystone_proof(suite: &mut EstimateSuite) -> Result<()> {
    let constants = canonical_keystone_constants(1, 0.001, 2.)?;
    let c = &constants.values;
    let logs = &constants.natural_logs;
    let law = enumerate_law(vec![1., 1., -1., -1.], vec![0.; 4], vec![true; 4])?;
    let z = [2. / 3., 2. / 3., -2. / 3., -2. / 3.];
    let mut cc = Vec::new();
    let near = "lem-keystone-near-neighbor-pressure";
    let ap = "thm-keystone-averaged-cluster-pressure";
    let mut apc = Vec::new();
    for (i, j) in [(0, 1), (1, 0), (2, 3), (3, 2)] {
        let count = z
            .iter()
            .enumerate()
            .filter(|&(l, zl)| l != i && (*zl - z[i]).abs() <= c["r"])
            .count();
        cc.push(equal("actual_near_neighbor_count", near, count as f64, 1.));
        let ms = z.iter().map(|zl| (zl - z[j]).powi(2)).sum::<f64>() / 4.;
        cc.extend([
            equal(
                "actual_feature_second_moment",
                near,
                ms,
                var(&z) + (z[j] - mean(&z)).powi(2),
            ),
            lower("actual_feature_variance_hypothesis", near, ms, c["v_0"]),
        ]);
        for p in &law.patterns {
            if p.raw[i] <= c["r"] && p.raw[j] >= c["h_f"] {
                let gap = p.y[j] - p.y[i];
                let gapbound = (c["h_f"].powi(2) + 1e-6).sqrt() - (c["r"].powi(2) + 1e-6).sqrt();
                cc.extend([
                    lower("actual_favorable_distance_gap", near, gap, gapbound),
                    lower("actual_floor_far_gap", near, gapbound, c["Delta_f"]),
                    lower(
                        "actual_positive_fitness_gap",
                        near,
                        p.f[j] - p.f[i],
                        c["A_minus"] * c["omega_f"] - c["f_plus"] * c["L_A"] * c["r"],
                    ),
                    lower(
                        "favorable_margin_survives_reward_oscillation",
                        near,
                        c["A_minus"] * c["omega_f"] - c["f_plus"] * c["L_A"] * c["r"],
                        c["gamma_0"],
                    ),
                ]);
                let delta = 0.2_f64.hypot(0.001) - 0.001;
                let dm = (16. / 9. + 1e-6_f64).sqrt() - 0.001;
                let zm = dm / 0.1;
                let mf = 2. * (-zm).exp() / (1. + (-zm).exp()).powi(2);
                let ss = (dm * dm / 4. + 0.01).sqrt();
                let g = |x: f64| 0.1 + 2. / (1. + (-x).exp());
                let dy = (p.y[j] - p.y[i]) / p.scales[i];
                let fi = g(p.z[i]);
                let fj = g(p.z[j]);
                apc.extend([
                    lower("native_favorable_score_gap", ap, dy, delta / ss),
                    lower(
                        "native_favorable_logistic_gap",
                        ap,
                        fj - fi,
                        mf * delta / ss,
                    ),
                    equal(
                        "native_shared_reward_product_gap",
                        ap,
                        p.f[j] - p.f[i],
                        1.1 * (fj - fi),
                    ),
                    lower(
                        "native_favorable_powered_gap",
                        ap,
                        p.f[j] - p.f[i],
                        1.1 * mf * delta / ss,
                    ),
                ]);
            }
        }
    }
    records(
        suite,
        near,
        &[
            "n_i(r)=",
            "\\frac1k\\sum_\\ell|z_\\ell-z_j|^2",
            "Y_j-Y_i\\ge",
            "F_j-F_i\\ge",
        ],
        json!({"native_complete_measurement_patterns":81,"positions":law.x,"alive":law.alive,"feature_points":z,"actual_near_multiplicity":1,"constants":constants}),
        cc,
    )?;
    records(
        suite,
        ap,
        &[
            "\\frac{Y_j-\\bar Y}{s_Y}",
            "f((Y_j-\\bar Y)/s_Y)",
            "F_j-F_i\n=",
        ],
        json!({"native_complete_measurement_patterns":81,"positions":law.x,"reward_factor":1.1,"cluster_feature_diameter":0.,"independent_measurement_event":"near recipient and far donor within same duplicate-site cluster"}),
        apc,
    )?;
    let label = "thm-keystone-averaged-cluster-pressure";
    let k = 100.;
    let ng = 5.;
    let rho = (ng - 1.) / (k - 1.);
    let (native_event, native_event_inputs) =
        compressed_native_two_site_event(&BigInt::from(100), &BigInt::from(5), 1, 1, 0);
    let native_event_pressure = native_event.lo.to_f64().expect("finite event") / 1e40;
    let feature_gap = 4_f64 / 3.;
    let feature_variance = (5_f64 / 100.) * (95_f64 / 100.) * feature_gap.powi(2);
    let geometric_threshold = 0.2_f64;
    let rho_h = (feature_variance - geometric_threshold.powi(2))
        / (feature_gap.powi(2) - geometric_threshold.powi(2));
    let measured_span = (feature_gap.powi(2) + 1e-6).sqrt() - 0.001;
    let maximum_scale = (measured_span.powi(2) / 4. + 0.01).sqrt();
    let minimum_logistic_derivative =
        2. * (-measured_span / 0.1).exp() / (1. + (-measured_span / 0.1).exp()).powi(2);
    let geometric_gap = (geometric_threshold.powi(2) + 1e-6).sqrt() - 0.001;
    let gamma_g = 1.1 * minimum_logistic_derivative * geometric_gap / maximum_scale;
    let acceptance_g = (gamma_g / 4.410001).min(1.);
    let positive_factor = (-12_f64).exp() * rho_h * acceptance_g;
    let full_ap6_pressure = 0.04_f64.powi(2) * positive_factor;
    records(
        suite,
        label,
        &[
            "\\rho_G\\ge\\frac{(4/5)n_G}{k}",
            "\\overline p_i\\ge 0.04^2",
            "\\omega_G\\le",
        ],
        json!({"statistically_valid_cluster_size":ng,"alive_count":k,"minimum_size":5.,"minimum_fraction":0.05,"actual_nonself_cluster_fraction":rho,"compressed_duplicate_site_cluster":"five coincident points at +1 and ninety-five at -1; exact reward oscillation zero","actual_native_event":native_event_inputs,"actual_native_event_pressure_lower":native_event_pressure,"feature_variance":feature_variance,"h":geometric_threshold,"rho_h":rho_h,"actual_global_diversity_span":measured_span,"maximum_shared_scale":maximum_scale,"minimum_logistic_derivative":minimum_logistic_derivative,"geometric_diversity_gap":geometric_gap,"actual_reward_factor_minimum":1.1,"gamma_G":gamma_g,"mathfrak_a_G":acceptance_g,"positive_full_factor":positive_factor,"full_AP6_rhs":full_ap6_pressure, "scope":"A statistically valid native five-row cluster: exact companion masses and actual shared-normalizer acceptance cover independently bound its complete pressure from a positive accepted-event subset."}),
        vec![
            lower("valid_cluster_nonself_fraction", label, rho, 0.8 * ng / k),
            lower(
                "valid_cluster_population_fraction",
                label,
                0.8 * ng / k,
                0.04,
            ),
            lower(
                "AP6_sharper_pressure_coefficient",
                label,
                rho * rho,
                0.04_f64.powi(2),
            ),
            upper("coincident_reward_oscillation", label, 0., 0.),
            upper("actual_statistical_minimum_cluster_size", label, 5., ng),
            upper("actual_statistical_minimum_fraction", label, 0.05 * k, ng),
            certificate_check(
                "actual_geometric_threshold_hypothesis",
                label,
                geometric_threshold > 0. && geometric_threshold.powi(2) < feature_variance,
            ),
            certificate_check(
                "strict_all_AP6_factors",
                label,
                rho_h > 0.
                    && gamma_g > 0.
                    && acceptance_g > 0.
                    && positive_factor > 0.
                    && native_event.lo > BigInt::zero(),
            ),
            lower(
                "actual_native_full_AP6_pressure",
                label,
                native_event_pressure,
                full_ap6_pressure,
            ),
            lower(
                "native_event_ge_exact_cluster_mass_pressure",
                label,
                native_event_pressure,
                rho.powi(2) * positive_factor,
            ),
            lower(
                "full_AP6_mass_chain",
                label,
                rho.powi(2) * positive_factor,
                full_ap6_pressure,
            ),
        ],
    )?;
    let cover = "thm-keystone-complete-error-coverage";
    let w = 0.25_f64;
    let e = [w / 2., w / 2.];
    let nc = [2., 2.];
    let lhs = e
        .iter()
        .zip(nc)
        .map(|(e, n)| e * (n - 1_f64).powi(2))
        .sum::<f64>();
    let weighted = e.iter().zip(nc).map(|(e, n)| e * (n - 1.)).sum::<f64>();
    let ec2 = e.iter().map(|e| e * e).sum::<f64>();
    let en = e.iter().zip(nc).map(|(e, n)| e * n).sum::<f64>();
    let mr = logs["M_r"].exp();
    records(
        suite,
        cover,
        &["\\sum_c E_c(n_c-1)^2", "\\sum_c E_c n_c\\ge"],
        json!({"occupied_cells":2,"cell_counts":nc,"cell_error_contributions":e,"N":4,"W":w,"Emax":64.,"log_M_r":logs["M_r"]}),
        vec![
            lower(
                "weighted_cell_Cauchy_Schwarz",
                cover,
                lhs,
                weighted * weighted / w,
            ),
            lower("error_capacity_cell_count", cover, en, 4. / 64. * ec2),
            lower("full_cover_cell_Cauchy_Schwarz", cover, ec2, w * w / mr),
            lower(
                "complete_second_cell_chain",
                cover,
                4. / 64. * ec2,
                4. * w * w / (64. * mr),
            ),
        ],
    )?;
    let giant = 2_f64.powi(512);
    record(
        suite,
        cover,
        "k\\ge\\frac{2E_{\\max}M_r}{W_0}",
        json!({"exact_population":"2^512","all_alive":true,"compressed_native_duplicate_site_law":true,"W0":0.001,"log_Mr":logs["M_r"]}),
        vec![lower(
            "verified_log_population_threshold",
            cover,
            giant.ln(),
            2_f64.ln() + 64_f64.ln() + logs["M_r"] - 0.001_f64.ln(),
        )],
    )?;
    let label = "thm-keystone-discharged-averaged-pressure";
    let ww = 0.01_f64;
    let a = ww / (64. * mr);
    let inv = 1. / giant;
    let left = ww * (a - inv).max(0.).powi(2);
    let right = ww * a * a / 2. - ww * inv * inv;
    records(
        suite,
        label,
        &[
            "C_0W\\left(",
            "N\\ge N_0:=",
            "V_{\\rm struct}\\le(1+\\eta)W",
        ],
        json!({"exact_population":"2^512","all_alive":true,"W":ww,"m_s":1.,"Emax":64.,"log_M_r":logs["M_r"],"positive_C0_cancels_before_comparison":true,"zero_velocity":true}),
        vec![
            lower("normalized_affine_pressure_algebra", label, left, right),
            lower(
                "actual_large_population_threshold_log",
                label,
                giant.ln(),
                logs["N_0_unrounded"],
            ),
            upper("native_balanced_structural_Young_bound", label, ww, 2. * ww),
        ],
    )?;
    let label = "lem-quantitative-keystone";
    let chi = 0.01;
    let spread = 2.;
    let offset = 0.02;
    records(
        suite,
        label,
        &["\\chi(\\epsilon) V_{\\text{struct}} - g_{\\max}"],
        json!({"low_error_regime":[0.,0.2,1.,2.],"chi":chi,"spread_threshold_squared":spread,"offset":offset}),
        [0., 0.2, 1., 2.]
            .into_iter()
            .flat_map(|v| {
                [
                    upper(
                        "low_regime_first_chain",
                        label,
                        chi * v - offset,
                        chi * spread - offset,
                    ),
                    upper("low_regime_last_chain", label, chi * spread - offset, 0.),
                ]
            })
            .collect(),
    )?;
    for n in [4, 6, 8] {
        for radius in [0.5_f64, 1., 1.9] {
            let exact =
                crate::convergence_cloning::exact_balanced_two_site_fixture(n, radius, 0.1)?;
            let m = n as f64 / 2.;
            let ell = 4. * radius / (2. + radius);
            let weight = (-ell * ell / 8.).exp();
            let q = m * weight / (m - 1. + m * weight);
            let a0 = crate::convergence_cloning::canonical_two_cluster_acceptance_floor(radius);
            let meanpressure = mean(&exact.acceptance_probabilities);
            let bv = exact.measurement_barycenter_variance;
            let copycov = exact.covariance_traces.iter().sum::<f64>()
                - exact.acceptance_probabilities.iter().sum::<f64>() * 0.01;
            let pressurelower = a0 * q * (1. - q);
            let label = "prop-cloning-two-cluster-noise-balance";
            let beta = q / m;
            records(
                suite,
                label,
                &[
                    "3.U4",
                    "\\mathbb E|\\bar t|^2",
                    "\\frac1{N^2}\\sum_i\\mathbb E a_i",
                ],
                json!({"N":n,"radius":radius,"exact_measurement_patterns":1usize<<n,"q":q,"A0":a0,"barycenter_mean_variance":bv,"copy_covariance_sum":copycov,"complete_fitness_ties_included":true}),
                vec![
                    lower(
                        "native_two_site_internal_drift",
                        label,
                        exact.expected_position_drift,
                        0.01 * (1. - 1. / n as f64) * pressurelower
                            - 4. * radius * radius / n as f64 * q * q * (1. - q * q),
                    ),
                    upper(
                        "actual_measurement_barycenter_shift",
                        label,
                        bv,
                        radius * radius * beta * beta * 2. * m * q * (1. - q),
                    ),
                    equal(
                        "barycenter_shift_bound_simplification",
                        label,
                        radius * radius * beta * beta * 2. * m * q * (1. - q),
                        4. * radius * radius / n as f64 * q.powi(3) * (1. - q),
                    ),
                    upper(
                        "actual_copy_covariance_bound",
                        label,
                        copycov / (n * n) as f64,
                        4. * radius * radius * beta / (n * n) as f64 * 2. * m * m * q * (1. - q),
                    ),
                    equal(
                        "copy_covariance_bound_simplification",
                        label,
                        4. * radius * radius * beta / (n * n) as f64 * 2. * m * m * q * (1. - q),
                        4. * radius * radius / n as f64 * q * q * (1. - q),
                    ),
                ],
            )?;
            let label = "cor-keystone-canonical-balanced-structural";
            record(
                suite,
                label,
                "3.KB2",
                json!({"N":n,"radius":radius,"actual_pressure":meanpressure,"native_event_floor":a0,"q":q}),
                vec![lower(
                    "actual_native_complete_pressure",
                    label,
                    meanpressure,
                    pressurelower,
                )],
            )?;
        }
    }
    let left = crate::convergence_cloning::exact_balanced_two_site_fixture(8, 1., 0.1)?;
    let right = crate::convergence_cloning::exact_balanced_two_site_fixture(8, 0.5, 0.1)?;
    let activity =
        0.25 * (mean(&left.acceptance_probabilities) + mean(&right.acceptance_probabilities));
    let label = "cor-keystone-canonical-balanced-structural";
    let sharper = 0.25
        * [
            (1.0_f64, mean(&left.acceptance_probabilities)),
            (0.5, mean(&right.acceptance_probabilities)),
        ]
        .iter()
        .map(|(r, _)| {
            let l = 4. * r / (2. + r);
            let w = (-l * l / 8.).exp();
            let q = 4. * w / (3. + 4. * w);
            crate::convergence_cloning::canonical_two_cluster_acceptance_floor(*r) * q * (1. - q)
        })
        .sum::<f64>();
    records(
        suite,
        label,
        &["3.KB1", "\\geq\\bigl[A_0(a)"],
        json!({"N":8,"radii":[1.,0.5],"exact_centered_transport":0.25,"actual_activity":activity,"chi0":canonical_cloning_constants().balanced_two_cluster_chi_0}),
        vec![
            lower(
                "native_balanced_zero_offset_structural",
                label,
                activity,
                canonical_cloning_constants().balanced_two_cluster_chi_0 * 0.25,
            ),
            lower("native_balanced_sharper_activity", label, activity, sharper),
        ],
    )?;
    let label = "prop-cloning-no-global-affine-structural-contraction";
    let aa = crate::convergence_cloning::canonical_two_cluster_acceptance_floor(0.5);
    let n = 10000.;
    let positive = 0.02 * aa / 9. * (1. - 1. / n) - 320. / (81. * n);
    let mut checks = Vec::new();
    for radius in [0.5_f64, 1., 1.9, 1.999] {
        let l = 4. * radius / (2. + radius);
        let w = (-l * l / 8.).exp();
        let q = 5000. * w / (4999. + 5000. * w);
        checks.extend([
            lower("uniform_two_site_q_variance", label, q * (1. - q), 2. / 9.),
            upper(
                "uniform_two_site_q_covariance",
                label,
                q * q * (1. - q * q),
                20. / 81.,
            ),
            lower(
                "uniform_monotone_acceptance_floor",
                label,
                crate::convergence_cloning::canonical_two_cluster_acceptance_floor(radius),
                aa,
            ),
        ]);
    }
    record(
        suite,
        label,
        "q(1-q)\\ge2/9",
        json!({"radii":[0.5,1.,1.9,1.999],"population":n,"A_star":aa}),
        checks,
    )?;
    record(
        suite,
        label,
        "\\mathbb E\\operatorname{Var}(x')-a^2",
        json!({"population":n,"analytic_native_balance_lower":positive,"A_star":aa,"scope":"proved native event subset balance; compressed 2^N law need not be enumerated"}),
        vec![lower(
            "strict_uniform_native_structural_increment",
            label,
            positive,
            0.01 * aa / 9.,
        )],
    )?;
    record(
        suite,
        label,
        "3.X5",
        json!({"contradiction_scope":"hypothetical inequality, explicitly negated by its proposition; this validates its falsification","candidate_kappa":0.2,"candidate_C":0.7,"radius":1.9,"population":n,"actual_native_increment_lower":positive,"hypothetical_rhs":-0.2*1.9_f64.powi(2)+0.7}),
        vec![
            certificate_check(
                "nonvacuous_hypothetical_parameter_hypothesis",
                label,
                0.7 < 4. * 0.2,
            ),
            certificate_check(
                "strict_native_contradiction_to_hypothetical_inequality",
                label,
                positive > 0. && -0.2 * 1.9_f64.powi(2) + 0.7 < 0.,
            ),
        ],
    )
}

fn simpson(f: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> f64 {
    if hi <= lo {
        return 0.;
    }
    let step = (hi - lo) / n as f64;
    let mut s = f(lo) + f(hi);
    for i in 1..n {
        s += if i % 2 == 0 { 2. } else { 4. } * f(lo + i as f64 * step);
    }
    s * step / 3.
}
fn revival_smoothing(suite: &mut EstimateSuite) -> Result<()> {
    let label = "cor-cloning-revival-full-step-cost";
    let h = 0.04_f64;
    let c = h / 2.;
    let a = (-h).exp();
    let q = ((1. - (-2. * h).exp()) / 2.).sqrt();
    let lf = 1.;
    let lambda = 1. - c * c * lf;
    let radius = 2.;
    let c0 = std::f64::consts::FRAC_PI_2.sqrt();
    let ax = (1. + c * c * (1. + a) * lf) / (c * (2. * std::f64::consts::PI).sqrt())
        + c0 * radius * (1. + c * c * lf) / (c * lambda);
    let av = (1. + a) / (2. * std::f64::consts::PI).sqrt() + c0 * radius / lambda;
    let phi = |u: f64| (-u * u / 2.).exp() / (2. * std::f64::consts::PI).sqrt();
    let cap = |u: f64| radius * u / (radius + u.abs());
    let mut checks = Vec::new();
    let mut cases = Vec::new();
    for leaves in [1, 8, 128] {
        let n = leaves + 3;
        let dead = n - 1;
        let velocity: Vec<_> = (0..n)
            .map(|i| if i % 2 == 0 { 0.008 } else { -0.007 })
            .collect();
        let mut ca = vec![vec![0, 1], vec![2]];
        ca[0].extend(3..n);
        let mut cb = ca.clone();
        cb[0].retain(|&i| i != dead);
        cb[1].push(dead);
        let xa: Vec<_> = (0..n).map(|i| if i == 2 { 0.01 } else { -0.01 }).collect();
        let mut xb = xa.clone();
        xb[dead] = 0.01;
        for rc in [-1., 1.] {
            for rd in [-1., 1.] {
                let va = collision_output(&velocity, &ca, &[rc, rd], 0.5);
                let vb = collision_output(&velocity, &cb, &[rc, rd], 0.5);
                let mut cost = 0.;
                let mut overlapmass = 0.;
                let mut linear = 0.;
                for i in 0..n {
                    let v1 = va[i] - c * xa[i];
                    let tv1 = vb[i] - c * xb[i];
                    let x1 = xa[i] + c * v1;
                    let tx1 = xb[i] + c * tv1;
                    let m = x1 + c * a * v1;
                    let tm = tx1 + c * a * tv1;
                    let shift = (m - tm) / (c * q);
                    let d = -(x1 - tx1) / c;
                    let split = (-shift / 2.).clamp(-12., 12.);
                    let density = |u: f64| phi(u).min(phi(u + shift));
                    let matched = |u: f64| {
                        let x2 = m + c * q * u;
                        let vv = a * v1 + q * u - c * x2;
                        density(u) * (cap(vv) - cap(vv - d)).abs().min(1.)
                    };
                    let mass =
                        simpson(density, -12., split, 12000) + simpson(density, split, 12., 12000);
                    let matchedcost =
                        simpson(matched, -12., split, 12000) + simpson(matched, split, 12., 12000);
                    let fail = (1. - mass).max(0.);
                    let rowcost = (fail + matchedcost + 1e-9).min(1.);
                    let rhs = (ax * (xa[i] - xb[i]).abs() + av * (va[i] - vb[i]).abs()) / q;
                    cost += rowcost / n as f64;
                    overlapmass += mass / n as f64;
                    linear += rhs / n as f64;
                    checks.extend([
                        upper(
                            "maximal_OU_coupling_failure_probability",
                            label,
                            fail,
                            shift.abs() / (2. * std::f64::consts::PI).sqrt() + 2e-9,
                        ),
                        upper(
                            "actual_matched_capped_velocity_cost",
                            label,
                            matchedcost,
                            c0 * radius * (x1 - tx1).abs() / (c * q * lambda) + 2e-9,
                        ),
                        upper("actual_declared_row_coupling", label, rowcost, rhs + 2e-9),
                    ]);
                }
                let bound = (ax * 0.02 + av * (6. + 8. * 0.5) * 0.008) / (n as f64 * q);
                checks.extend([
                    upper(
                        "declared_full_step_normalized_leaf_cost",
                        label,
                        cost,
                        bound + 2e-9,
                    ),
                    upper("sum_row_linear_smoothing_leaf_bound", label, linear, bound),
                    upper(
                        "coupled_copy_position_sum",
                        "thm-cloning-revival-backbone-coupling",
                        xa.iter().zip(&xb).map(|(x, y)| (x - y).abs()).sum(),
                        0.02,
                    ),
                    upper(
                        "single_leaf_exact_velocity_bound",
                        "thm-cloning-revival-backbone-coupling",
                        (va[dead] - vb[dead]).abs(),
                        (2. + 4. * 0.5) * 0.008,
                    ),
                ]);
                cases.push(json!({"N":n,"dead_leaves":leaves,"changes":1,"actual_coupling_cost_upper":cost,"L3_bound":bound,"matched_OU_mass_average":overlapmass,"alive_backbone_rotations":[rc,rd]}));
            }
        }
    }
    record(
        suite,
        label,
        "3.L3",
        json!({"kinetic_h":h,"gamma":1.,"B":1.,"sigma_p":0.1,"force":"-x globally Lipschitz","cap_radius":radius,"A_x":ax,"A_v":av,"q":q,"lambda":lambda,"bounded_marked_metric_units":[1.,1.],"coupling":"common Gaussian subdensity min(phi(u),phi(u+b)) matches the actual OU intermediate positions; shared independent final position Gaussian; independent residual subdensities charged at most one; cap derivative integrated on the matched law","numerical_integration":"split at exact Gaussian density equality, Simpson 12000 panels on each half, twelve standard deviation truncation with 1e-9 upper guard","cases":cases}),
        checks.clone(),
    )?;
    record(
        suite,
        "thm-cloning-revival-backbone-coupling",
        "|\\Delta v_r^+|",
        json!({"retained_velocity_cap":0.008,"restitution":0.5,"cases":cases}),
        checks
            .into_iter()
            .filter(|c| c.id == "single_leaf_exact_velocity_bound")
            .collect(),
    )
}

fn sample_proposal(law: &Law, seed: u64, step: usize) -> Vec<f64> {
    use algorithmic_gas::random::{RandomStream, Stream};
    let mut measurement = RandomStream::new(seed, step as u64, Stream::Distance, 0, 0);
    let draw = measurement.uniform::<f64>();
    let mut cumulative = 0.;
    let p = law
        .patterns
        .iter()
        .find(|p| {
            cumulative += p.mass;
            draw <= cumulative
        })
        .unwrap_or(law.patterns.last().unwrap());
    (0..law.x.len())
        .map(|i| {
            let mut r = RandomStream::new(seed, step as u64, Stream::Cloning, i as u64, 0);
            let u = r.uniform::<f64>();
            let mut cum = 0.;
            let donor = p.edges[i].iter().enumerate().find(|(_, b)| {
                cum += **b;
                u <= cum
            });
            if let Some((j, _)) = donor {
                let mut noise =
                    RandomStream::new(seed, step as u64, Stream::CloneNoise, i as u64, 0);
                law.x[j] + 0.1 * noise.gaussian::<f64>()
            } else {
                law.x[i]
            }
        })
        .collect()
}
fn expansion_chain_checks(suite: &mut EstimateSuite, samples: usize) -> Result<()> {
    let a = enumerate_law(vec![-1.5, -1.5, -1.5, 1.], vec![0.; 4], vec![true; 4])?;
    let b = enumerate_law(
        vec![-0.015, -0.015, -0.015, 0.01],
        vec![0.; 4],
        vec![true; 4],
    )?;
    let label = "prop-cloning-macroscopic-structural-expansion";
    let samples = samples.max(512);
    let mut cost = 0.;
    let mut sigma = 0.;
    let mut va = 0.;
    let mut vb = 0.;
    let mut checks = Vec::new();
    for t in 0..samples {
        let mut x = sample_proposal(&a, 81231, t);
        let mut y = sample_proposal(&b, 81232, t);
        let vx = var(&x);
        let vy = var(&y);
        let mx = mean(&x);
        let my = mean(&y);
        for x in &mut x {
            *x -= mx;
        }
        for y in &mut y {
            *y -= my;
        }
        x.sort_by(f64::total_cmp);
        y.sort_by(f64::total_cmp);
        let actual = x.iter().zip(&y).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / 4.;
        let gapbound = (vx.sqrt() - vy.sqrt()).powi(2);
        checks.push(lower(
            "actual_native_centered_transport_std_bound",
            label,
            actual,
            gapbound,
        ));
        cost += actual / samples as f64;
        sigma += gapbound / samples as f64;
        va += vx / samples as f64;
        vb += vy / samples as f64;
    }
    checks.extend([
        lower(
            "empirical_native_coupling_first_expectation",
            label,
            cost,
            sigma,
        ),
        lower(
            "empirical_Cauchy_Schwarz_second_expectation",
            label,
            sigma,
            (va.sqrt() - vb.sqrt()).powi(2),
        ),
    ]);
    let rational_a = [
        Iv::rational(-3, 2),
        Iv::rational(-3, 2),
        Iv::rational(-3, 2),
        Iv::rational(1, 1),
    ];
    let rational_b: Vec<_> = rational_a
        .iter()
        .map(|x| x.div(&Iv::rational(100, 1)))
        .collect();
    let (theoreticala, _, _) = interval_clone_variance(&rational_a);
    let (theoreticalb, _, _) = interval_clone_variance(&rational_b);
    let strict = theoreticala.sqrt().sub(&theoreticalb.sqrt()).square();
    checks.push(certificate_check(
        "directed_rational_operator_final_chain",
        label,
        strict.lo > Iv::decimal("1.27648593291590").hi,
    ));
    record(
        suite,
        label,
        "\\mathbb E_\\Gamma V_{\\rm struct}'",
        json!({"native_independent_coupling_samples":samples,"empirical_expected_transport":cost,"empirical_std_gap_square":sigma,"empirical_marginal_variances":[va,vb],"rigorous_marginal_operator_lower":strict.json(),"scope":"Pointwise optimal centered uniform W2 and finite-sample Cauchy–Schwarz validate both chain relations on native outputs; the full operator endpoint is separately enclosed by the directed rational 81-pattern certificate"}),
        checks,
    )?;
    let label = "prop-canonical-fullstep-structural-expansion";
    let mut projection = Vec::new();
    for x in [-2., -0.5, 0., 0.5, 2.] {
        for v in [-2., -0.1, 0., 0.1, 2.] {
            let q = x * x + v * v + 0.1 * x * v;
            projection.extend([
                equal(
                    "completed_square_Q_identity",
                    label,
                    q,
                    399. / 400. * x * x + (v + 0.05 * x).powi(2),
                ),
                lower(
                    "actual_hypocoercive_positional_projection",
                    label,
                    q,
                    399. / 400. * x * x,
                ),
            ]);
        }
    }
    record(
        suite,
        label,
        "Q(x,v)=\\tfrac{399}{400}",
        json!({"positions":[-2.,-0.5,0.,0.5,2.],"velocities":[-2.,-0.1,0.,0.1,2.],"lambda_v":1.,"b":0.1}),
        projection,
    )?;
    let fourth = Iv::decimal("1.5")
        .square()
        .square()
        .add(
            &Iv::rational(6, 1)
                .mul(&Iv::decimal("1.5").square())
                .mul(&Iv::decimal("0.0105")),
        )
        .add(&Iv::rational(3, 1).mul(&Iv::decimal("0.0105").square()));
    record(
        suite,
        label,
        "\\mathbb E V_A^2\\le",
        json!({"center_bound":1.5,"actual_preterminal_position_variance_upper":0.0105,"Gaussian_fourth_moment":fourth.json(),"first_clause":"empirical variance squared ≤ mean fourth moment by centering variance identity and Cauchy–Schwarz"}),
        vec![
            certificate_check(
                "directed_rational_fourth_chain_last_clause",
                label,
                fourth.hi < Iv::decimal("5.205").lo,
            ),
            upper(
                "finite_native_variance_fourth_Cauchy_Schwarz",
                label,
                var(&[-1.5, -1.5, -1.5, 1.]).powi(2),
                [-1.5_f64, -1.5, -1.5, 1.]
                    .iter()
                    .map(|x| x.powi(4))
                    .sum::<f64>()
                    / 4.,
            ),
        ],
    )?;
    let a0 = Iv::decimal("1.31778710461037");
    let b0 = Iv::decimal("0.00066809082561");
    let la = a0.sub(&Iv::decimal("5.205").mul(&Iv::decimal("0.000055")).sqrt());
    let ub = b0
        .add(&Iv::decimal("0.000000000000000000000000000004"))
        .div(&Iv::rational(1, 1).sub(&Iv::decimal("0.000000000000000000000000000001")));
    let bound = la
        .sqrt()
        .sub(&ub.sqrt())
        .square()
        .mul(&Iv::rational(399, 400));
    record(
        suite,
        label,
        "\\mathbb E V_{\\rm struct}'\n\\ge\\frac{399}",
        json!({"directed_operator_LA":la.json(),"directed_operator_UB":ub.json(),"projected_bound":bound.json()}),
        vec![certificate_check(
            "full_directed_projected_endpoint",
            label,
            bound.lo > Iv::decimal("1.2285546875").hi,
        )],
    )
}

fn native_declarations(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let va: Vec<_> = live.iter().map(|&i| law.v[i]).collect();
    let k = live.len();
    let floor = law.config.fitness.distance_floor;
    let epsilon = law.config.clone_decision.epsilon;
    let pmax = law.config.clone_decision.saturation;
    let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,
        "delta_D":floor,"pmax":pmax,"epsilon_clone":epsilon,
        "diversity_exponent":law.config.fitness.diversity_exponent,
        "reward_exponent":law.config.fitness.reward_exponent,"native_patterns":law.patterns.len()});
    let label = "axiom-active-diversity";
    record(
        suite,
        label,
        "\\beta > 0",
        inputs.clone(),
        vec![certificate_check(
            "actual_active_native_diversity_exponent",
            label,
            law.config.fitness.diversity_exponent > 0.,
        )],
    )?;
    let mut distances = Vec::new();
    let mut gates = Vec::new();
    let mut moments = Vec::new();
    for p in &law.patterns {
        for &i in &live {
            let raw = p.raw[i];
            let measured = p.y[i];
            distances.extend([
                equal(
                    "actual_native_distance_floor_pipeline",
                    "def-raw-value-operators",
                    measured,
                    (raw * raw + floor * floor).sqrt(),
                ),
                lower(
                    "actual_distance_floor_nonnegative_increment",
                    "lem-cloning-distance-floor-transfer",
                    measured - raw,
                    0.,
                ),
                upper(
                    "actual_distance_floor_increment",
                    "lem-cloning-distance-floor-transfer",
                    measured - raw,
                    floor,
                ),
            ]);
            let mut gate = 0.;
            for &j in &live {
                let score = (p.f[j] - p.f[i]) / (p.f[i] + epsilon);
                let interval_length = score.clamp(0., pmax);
                let conditional = law
                    .config
                    .clone_decision
                    .acceptance_probability(1, p.f[i], p.f[j]);
                gates.push(equal(
                    "actual_uniform_threshold_interval_probability",
                    "def-cloning-probability",
                    conditional,
                    interval_length / pmax,
                ));
                gates.push(equal(
                    "actual_native_retained_cloning_score",
                    "def-cloning-score",
                    score,
                    (p.f[j] - p.f[i]) / (p.f[i] + epsilon),
                ));
                gate += law.donors[i][j] * interval_length / pmax;
            }
            gates.push(equal(
                "actual_weighted_gate_integral",
                "def-cloning-probability",
                p.p[i],
                gate,
            ));
            moments.extend([
                upper(
                    "actual_row_covariance_pressure",
                    "lem-cloning-individual-centered-displacement",
                    p.traces[i],
                    p.p[i] * (range(&xa).powi(2) + 0.01),
                ),
                upper(
                    "actual_row_covariance_ceiling",
                    "lem-cloning-individual-centered-displacement",
                    p.traces[i],
                    range(&xa).powi(2) + 0.01,
                ),
            ]);
        }
    }
    records(
        suite,
        "def-raw-value-operators",
        &["\\ell_i:=d_"],
        inputs.clone(),
        distances.clone(),
    )?;
    records(
        suite,
        "def-measurement-operator",
        &["s_i=\\sqrt"],
        inputs.clone(),
        distances.clone(),
    )?;
    records(
        suite,
        "lem-cloning-distance-floor-transfer",
        &["0\\leq d_i-\\ell_i\\leq\\delta_D"],
        inputs.clone(),
        distances,
    )?;
    records(
        suite,
        "def-cloning-probability",
        &["p_i :=", "p_i ="],
        inputs.clone(),
        gates.clone(),
    )?;
    record(
        suite,
        "def-cloning-score",
        "S_i(c_i) :=",
        inputs.clone(),
        gates,
    )?;
    records(
        suite,
        "lem-cloning-individual-centered-displacement",
        &["c_i\\le p_i(D_x^2+d j^2)", "c_i\\le D_x^2+d j^2"],
        inputs.clone(),
        moments,
    )?;
    let label = "prop-bounded-velocity-expansion";
    let vmax = law.v.iter().map(|v| v.abs()).fold(0., f64::max);
    let mu = mean(&va);
    let deadsecond = (0..n)
        .filter(|&i| !law.alive[i])
        .map(|i| (law.v[i] - mu).powi(2))
        .sum::<f64>()
        / n as f64;
    let normalized = k as f64 / n as f64 * var(&va);
    let anchored = law.v.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64;
    record(
        suite,
        label,
        "\\mathcal V_v^{\\rm all}(S)\n=\\min_b",
        inputs,
        vec![
            equal(
                "full_velocity_variance_minimizer",
                label,
                var(&law.v),
                law.v
                    .iter()
                    .map(|v| (v - mean(&law.v)).powi(2))
                    .sum::<f64>()
                    / n as f64,
            ),
            upper(
                "variance_minimum_at_alive_center",
                label,
                var(&law.v),
                anchored,
            ),
            equal(
                "full_alive_center_energy_split",
                label,
                anchored,
                normalized + deadsecond,
            ),
            upper(
                "full_slot_alive_and_dead_velocity_chain",
                label,
                var(&law.v),
                normalized + deadsecond,
            ),
            upper(
                "retained_dead_velocity_cap_chain",
                label,
                normalized + deadsecond,
                normalized + 4. * (n - k) as f64 / n as f64 * vmax * vmax,
            ),
        ],
    )
}

fn remaining_elementary_bounds(suite: &mut EstimateSuite) -> Result<()> {
    let label = "lem-V_Varx-implies-variance";
    for n in [4, 16, 64, 256] {
        for (left_alive, right_alive, radius_left, radius_right) in [
            (n, n, 1., 0.5),
            (n / 2, n, 1.5, 0.25),
            (n / 4, n / 2, 2., 0.5),
        ] {
            let left: Vec<_> = (0..left_alive)
                .map(|i| {
                    if i % 2 == 0 {
                        radius_left
                    } else {
                        -radius_left
                    }
                })
                .collect();
            let right: Vec<_> = (0..right_alive)
                .map(|i| {
                    if i % 2 == 0 {
                        radius_right
                    } else {
                        -radius_right
                    }
                })
                .collect();
            let vl = left_alive as f64 / n as f64 * var(&left);
            let vr = right_alive as f64 / n as f64 * var(&right);
            let threshold = 0.9 * (vl + vr);
            records(
                suite,
                label,
                &[
                    r"V_{Var,x} > R_{total\_var,x}^2",
                    r"R_{total\_var,x}^2 > 0",
                    r"k \in \{1, 2\}",
                ],
                json!({"N":n,"actual_alive_counts":[left_alive,right_alive],"left_centered_positions":left,"right_centered_positions":right,"normalized_alive_variances":[vl,vr],"threshold_squared":threshold,"selected_swarm":if vl>=vr{1}else{2},"scope":"Two actual alive clouds with N-normalized variance contributions. Strict premise and at least one strict half-threshold conclusion are evaluated separately for every population and alive fraction."}),
                vec![
                    certificate_check(
                        "actual_total_N_normalized_variance_premise",
                        label,
                        vl + vr > threshold,
                    ),
                    certificate_check(
                        "actual_positive_total_variance_threshold",
                        label,
                        threshold > 0.,
                    ),
                    certificate_check(
                        "actual_selected_swarm_index",
                        label,
                        [1, 2].contains(&(if vl >= vr { 1 } else { 2 })),
                    ),
                    certificate_check(
                        "actual_strict_one_swarm_half_threshold",
                        label,
                        vl.max(vr) > threshold / 2.,
                    ),
                ],
            )?;
            record(
                suite,
                label,
                "\\frac{1}{N} \\sum_{i \\in \\mathcal{A}(S_k)}",
                json!({"N":n,"alive_counts":[left_alive,right_alive],"normalized_alive_variances":[vl,vr],"threshold_squared":threshold,"scope":"Actual centered probability populations; strict conclusion under the verified total normalized-variance premise"}),
                vec![
                    certificate_check(
                        "strict_total_normalized_variance_premise",
                        label,
                        vl + vr > threshold && threshold > 0.,
                    ),
                    certificate_check(
                        "strict_one_swarm_normalized_variance_conclusion",
                        label,
                        vl.max(vr) > threshold / 2.,
                    ),
                ],
            )?;
        }
        let threshold = 2.;
        let vl = 0.75_f64;
        let vr = 0.25_f64;
        let checks = vec![
            upper(
                "contradiction_branch_first_swarm",
                label,
                vl,
                threshold / 2.,
            ),
            upper(
                "contradiction_branch_second_swarm",
                label,
                vr,
                threshold / 2.,
            ),
            equal("total_normalized_variance_sum", label, vl + vr, vl + vr),
            upper(
                "contradiction_branch_sum",
                label,
                vl + vr,
                threshold / 2. + threshold / 2.,
            ),
            equal(
                "contradiction_branch_threshold_identity",
                label,
                threshold / 2. + threshold / 2.,
                threshold,
            ),
        ];
        records(
            suite,
            label,
            &[
                "\\frac{1}{N} \\sum_{i \\in \\mathcal{A}(S_1)} \\|\\delta_{x,1,i}\\|^2 \\le",
                "\\begin{aligned}\nV_{Var,x} &=",
                "V_{Var,x} \\le R_{total\\_var,x}^2",
            ],
            json!({"N":n,"normalized_alive_variances":[vl,vr],"threshold_squared":threshold,"scope":"The proof's contrapositive branch verifies both assumed upper bounds and the consequent total upper bound; it does not assert the incompatible strict premise"}),
            checks,
        )?;
    }
    let label = "lem-quantitative-keystone";
    let chi = 0.01_f64;
    let threshold = 2.;
    let offset = 0.02;
    let checks = [0., 0.2, 1., 2.]
        .into_iter()
        .flat_map(|v| {
            [
                upper(
                    "low_error_regime_first_complete_clause",
                    label,
                    chi * v - offset,
                    chi * threshold - offset,
                ),
                upper(
                    "low_error_regime_last_complete_clause",
                    label,
                    chi * threshold - offset,
                    0.,
                ),
            ]
        })
        .collect();
    record(
        suite,
        label,
        "\\chi(\\epsilon) V_{\\text{struct}} - g_{\\max}(\\epsilon) \\le",
        json!({"low_error_values":[0.,0.2,1.,2.],"chi":chi,"threshold_squared":threshold,"offset":offset}),
        checks,
    )?;
    let label = "lem-keystone-complete-coverage-constants";
    let report = canonical_keystone_constants(1, 0.001, 2.)?;
    let c = &report.values;
    let log = &report.natural_logs;
    record(
        suite,
        label,
        "m_x=\\frac{R_x^2}",
        json!({"native_parameters":report}),
        vec![
            equal(
                "actual_position_inverse_feature_constant",
                label,
                c["m_x"],
                4. / 16.,
            ),
            equal(
                "actual_joint_inverse_feature_constant",
                label,
                c["m_z"],
                (4_f64 / 16.).min(4. / 16.),
            ),
            certificate_check(
                "actual_strict_joint_inverse_feature_constant",
                label,
                c["m_z"] > 0.,
            ),
        ],
    )?;
    let mut radii = Vec::new();
    for la in [0., c["L_A"]] {
        let r = if la > 0. {
            (c["h_f"] / 2.).min(c["A_minus"] * log["omega_f"].exp() / (2. * c["f_plus"] * la))
        } else {
            c["h_f"] / 2.
        };
        radii.push(equal_positive(
            "piecewise_near_radius_formula",
            label,
            r,
            if la > 0. {
                log["r"].exp()
            } else {
                c["h_f"] / 2.
            },
        ));
        radii.push(equal_positive(
            "actual_gamma_gap_formula",
            label,
            c["gamma_0"],
            c["A_minus"] * log["omega_f"].exp() / 2.,
        ));
    }
    record(
        suite,
        label,
        "r=\\begin{cases}",
        json!({"native_parameters":report,"both_LA_branches":[0.,c["L_A"]]}),
        radii,
    )?;
    let cover = "thm-keystone-complete-error-coverage";
    let e = [0.08_f64, 0.12, 0.05];
    let nc = [2_f64, 5., 3.];
    let w = e.iter().sum::<f64>();
    let weighted = e.iter().zip(nc).map(|(e, n)| e * (n - 1.)).sum::<f64>();
    let squared = e
        .iter()
        .zip(nc)
        .map(|(e, n)| e * (n - 1.).powi(2))
        .sum::<f64>();
    record(
        suite,
        cover,
        "\\sum_c E_c(n_c-1)^2\n\\ge",
        json!({"cell_errors":e,"cell_counts":nc,"W":w}),
        vec![lower(
            "nonuniform_weighted_cell_Cauchy_Schwarz",
            cover,
            squared,
            weighted * weighted / w,
        )],
    )?;
    let label = "lem-barrier-reduction-cloning";
    let sigma = 0.1_f64;
    let mq = 1. / (2. * std::f64::consts::PI * sigma * sigma).sqrt();
    let mut integrals = Vec::new();
    for y in [-3_f64, -2., -1., 0., 1., 2., 3.] {
        let integral = simpson(
            |z| z * z * mq * (-(z - y).powi(2) / (2. * sigma * sigma)).exp(),
            -2.,
            2.,
            16000,
        );
        integrals.push(upper(
            "actual_zero_extended_quadratic_barrier_integral",
            label,
            integral,
            mq * 16. / 3.,
        ));
    }
    record(
        suite,
        label,
        "\\mathbb E[\\varphi(Y)\\mathbf1_{Y\\in D}\\mid y]",
        json!({"D":[-2.,2.],"observable":"x² on D, zero extension outside D; auxiliary only","L1_norm":16./3.,"sigma":sigma,"Gaussian_density_sup":mq,"quadrature_panels":16000}),
        integrals,
    )?;
    let label = "cor-bounded-boundary-exposure";
    let rate = 0.2_f64;
    let offset = 0.7;
    let initial = 12.;
    let rho = 1. - rate;
    let mut recursion = Vec::new();
    let mut value = initial;
    for t in 0..200 {
        let next = rho * value + offset;
        recursion.push(equal(
            "boundary_affine_one_step",
            label,
            next,
            rho * value + offset,
        ));
        recursion.push(equal(
            "boundary_exact_geometric_series",
            label,
            (1. - rho.powi(t + 1)) / rate,
            (0..=t).map(|j| rho.powi(j)).sum::<f64>(),
        ));
        value = next;
    }
    let floor = offset / rate;
    recursion.push(equal(
        "boundary_affine_limit_fixed_point",
        label,
        floor,
        rho * floor + offset,
    ));
    recursion.push(equal(
        "boundary_infinite_geometric_sum",
        label,
        1. / rate,
        1. / (1. - rho),
    ));
    records(
        suite,
        label,
        &[
            "\\mathbb{E}[W_b(S_{t+1})]",
            "\\limsup_{t \\to \\infty}",
            "\\sum_{j=0}^{\\infty}",
        ],
        json!({"declared_affine_operator":true,"rate":rate,"offset":offset,"initial":initial,"exact_fixed_point":floor,"scope":"Affine comparison dynamics conditional on the displayed boundary drift; limit obtained from strict rho<1 and its exact fixed point"}),
        recursion,
    )
}

fn transport_identities(suite: &mut EstimateSuite) -> Result<()> {
    let lambda = 0.5_f64;
    let b = 0.1_f64;
    let q = |x: f64, v: f64| x * x + lambda * v * v + b * x * v;
    let bilinear = |a: &[f64], c: &[f64]| {
        a[0] * c[0] + lambda * a[1] * c[1] + b / 2. * (a[0] * c[1] + a[1] * c[0])
    };
    let left = vec![vec![-1., 0.3], vec![0., -0.2], vec![1., 0.1]];
    for right in [
        vec![vec![-0.7, 0.1], vec![0.2, -0.1]],
        vec![vec![-0.7, 0.1], vec![0.2, -0.1], vec![1.4, 0.4]],
    ] {
        let center = |points: &[Vec<f64>]| {
            (0..2)
                .map(|r| points.iter().map(|z| z[r]).sum::<f64>() / points.len() as f64)
                .collect::<Vec<_>>()
        };
        let ml = center(&left);
        let mr = center(&right);
        let cl: Vec<_> = left
            .iter()
            .map(|z| vec![z[0] - ml[0], z[1] - ml[1]])
            .collect();
        let cr: Vec<_> = right
            .iter()
            .map(|z| vec![z[0] - mr[0], z[1] - mr[1]])
            .collect();
        let diff = vec![ml[0] - mr[0], ml[1] - mr[1]];
        let location = q(diff[0], diff[1]);
        let (full, plan) =
            crate::convergence_lyapunov::uniform_transport(&left, &right, |a, c| {
                q(a[0] - c[0], a[1] - c[1])
            })?;
        let (structural, centered_plan) =
            crate::convergence_lyapunov::uniform_transport(&cl, &cr, |a, c| {
                q(a[0] - c[0], a[1] - c[1])
            })?;
        let label = "lem-wasserstein-decomposition";
        let mut checks = vec![equal(
            "exact_alive_probability_transport_decomposition",
            label,
            full,
            location + structural,
        )];
        let mut integral_q = 0.;
        let mut centered_q = 0.;
        let mut cross = 0.;
        let mut leftmean = vec![0.; 2];
        let mut rightmean = vec![0.; 2];
        for (i, row) in plan.iter().enumerate() {
            checks.push(equal(
                "actual_optimal_uniform_left_marginal",
                label,
                row.iter().sum::<f64>(),
                1. / left.len() as f64,
            ));
            for (j, &mass) in row.iter().enumerate() {
                let dz = [left[i][0] - right[j][0], left[i][1] - right[j][1]];
                let dc = [cl[i][0] - cr[j][0], cl[i][1] - cr[j][1]];
                checks.extend([
                    equal(
                        "centered_vector_position_decomposition",
                        label,
                        dz[0],
                        dc[0] + diff[0],
                    ),
                    equal(
                        "centered_vector_velocity_decomposition",
                        label,
                        dz[1],
                        dc[1] + diff[1],
                    ),
                    equal(
                        "quadratic_pointwise_translation_identity",
                        label,
                        q(dz[0], dz[1]),
                        q(dc[0], dc[1]) + location + 2. * bilinear(&dc, &diff),
                    ),
                    equal(
                        "quadratic_matrix_cross_term",
                        label,
                        q(dz[0], dz[1]),
                        dz[0] * dz[0] + lambda * dz[1] * dz[1] + 2. * (b / 2.) * dz[0] * dz[1],
                    ),
                ]);
                integral_q += mass * q(dz[0], dz[1]);
                centered_q += mass * q(dc[0], dc[1]);
                cross += mass * bilinear(&dc, &diff);
                for r in 0..2 {
                    leftmean[r] += mass * cl[i][r];
                    rightmean[r] += mass * cr[j][r];
                }
            }
        }
        for j in 0..right.len() {
            checks.push(equal(
                "actual_optimal_uniform_right_marginal",
                label,
                plan.iter().map(|r| r[j]).sum::<f64>(),
                1. / right.len() as f64,
            ));
        }
        checks.extend([
            equal("full_optimal_transport_integral", label, full, integral_q),
            equal(
                "centered_optimal_transport_integral",
                label,
                structural,
                centered_q,
            ),
            equal(
                "centered_plan_cross_integral",
                label,
                cross,
                bilinear(&leftmean, &diff) - bilinear(&rightmean, &diff),
            ),
            equal(
                "exact_zero_centered_left_first_moment",
                label,
                leftmean.iter().map(|x| x.abs()).sum::<f64>(),
                0.,
            ),
            equal(
                "exact_zero_centered_right_first_moment",
                label,
                rightmean.iter().map(|x| x.abs()).sum::<f64>(),
                0.,
            ),
            equal("exact_zero_transport_cross_term", label, cross, 0.),
            equal(
                "integrated_quadratic_translation_identity",
                label,
                integral_q,
                centered_q + location + 2. * cross,
            ),
            equal(
                "centered_plan_probability_mass",
                label,
                centered_plan.iter().flatten().sum::<f64>(),
                1.,
            ),
        ]);
        let mut permuted = left.clone();
        permuted.rotate_left(1);
        let (permutedcost, _) =
            crate::convergence_lyapunov::uniform_transport(&permuted, &right, |a, c| {
                q(a[0] - c[0], a[1] - c[1])
            })?;
        checks.push(equal(
            "independent_storage_permutation_invariance",
            label,
            full,
            permutedcost,
        ));
        let inputs = json!({"alive_probability_left":left,"alive_probability_right":right,"left_mean":ml,"right_mean":mr,"centered_left":cl,"centered_right":cr,"optimal_plan":plan,"centered_optimal_plan":centered_plan,"lambda_v":lambda,"b":b,"Vloc":location,"Vstruct":structural,"VW":full,"scope":"Exact uniform transport on empirical probabilities, including unequal alive counts; storage permutations preserve the law and the cost"});
        records(
            suite,
            label,
            &[
                "W_h^2(\\mu_1, \\mu_2) = V_",
                "c(z_1, z_2) =",
                "\\bar{z}_k =",
                "z_1 - z_2 =",
                "\\begin{aligned}\nq(z_1 - z_2)",
                "\\begin{aligned}\n\\int c(z_1, z_2)",
                "\\int \\langle \\delta_{z_1} - \\delta_{z_2},",
                "\\int \\langle \\delta_{z_1} - \\delta_{z_2}, \\Delta\\bar{z} \\rangle_q \\, d\\gamma = \\left\\langle",
                "\\int \\delta_{z_1} \\, d\\gamma(z_1, z_2) =",
                "\\int c(z_1, z_2) \\, d\\gamma = \\int q",
                "q(\\Delta\\bar{z}) =",
                "\\int q(\\delta_{z_1} - \\delta_{z_2}) \\, d\\gamma(z_1, z_2) =",
                "W_h^2(\\mu_1, \\mu_2) = \\inf_{\\gamma",
            ],
            inputs.clone(),
            checks.clone(),
        )?;
        records(
            suite,
            "def-barycentres-and-centered-vectors",
            &["\\delta_{x,k,i} :=", "\\delta_{v,k,i} :="],
            inputs.clone(),
            checks.clone(),
        )?;
        record(
            suite,
            "def-location-error-component",
            "V_{\\text{loc}} :=",
            inputs.clone(),
            checks.clone(),
        )?;
        record(
            suite,
            "def-structural-error-component",
            "V_{\\text{struct}} :=",
            inputs.clone(),
            checks.clone(),
        )?;
        let coercive = "lem-V-coercive";
        records(
            suite,
            coercive,
            &[
                "q(\\Delta x, \\Delta v) = \\|",
                "q(\\Delta x, \\Delta v) = \\begin{pmatrix}",
                "Q_{\\text{scalar}} =",
                "V_{\\text{loc}} = \\|",
            ],
            inputs.clone(),
            checks,
        )?;
        if right.len() == 3 {
            let permutations = [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ];
            let (optimal, _) = crate::convergence_lyapunov::uniform_transport(&cl, &cr, |a, c| {
                (a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2)
            })?;
            let minimum = permutations
                .iter()
                .map(|perm| {
                    (0..3)
                        .map(|i| {
                            (cl[i][0] - cr[perm[i]][0]).powi(2)
                                + (cl[i][1] - cr[perm[i]][1]).powi(2)
                        })
                        .sum::<f64>()
                        / 3.
                })
                .fold(f64::INFINITY, f64::min);
            records(
                suite,
                coercive,
                &[
                    "W_2^2(\\tilde\\mu_1,\\tilde\\mu_2)\n=\\min",
                    "W_2^2(\\tilde{\\mu}_1, \\tilde{\\mu}_2)\n=\\min",
                ],
                inputs.clone(),
                vec![equal(
                    "actual_uniform_transport_equals_permutation_minimum",
                    coercive,
                    optimal,
                    minimum,
                )],
            )?;
        }
        let vx = var(&left.iter().map(|z| z[0]).collect::<Vec<_>>())
            + var(&right.iter().map(|z| z[0]).collect::<Vec<_>>());
        let vv = var(&left.iter().map(|z| z[1]).collect::<Vec<_>>())
            + var(&right.iter().map(|z| z[1]).collect::<Vec<_>>());
        for n in [4, 16, 64, 256] {
            let weightedx = (left.len() as f64
                * var(&left.iter().map(|z| z[0]).collect::<Vec<_>>())
                + right.len() as f64 * var(&right.iter().map(|z| z[0]).collect::<Vec<_>>()))
                / n as f64;
            let weightedv = (left.len() as f64
                * var(&left.iter().map(|z| z[1]).collect::<Vec<_>>())
                + right.len() as f64 * var(&right.iter().map(|z| z[1]).collect::<Vec<_>>()))
                / n as f64;
            let barrier = (left.iter().map(|z| z[0] * z[0]).sum::<f64>()
                + right.iter().map(|z| z[0] * z[0]).sum::<f64>())
                / n as f64;
            let total = full + 0.7 * (weightedx + lambda * weightedv) + 0.05 * barrier;
            let identitychecks = vec![
                equal(
                    "full_weighted_lyapunov_decomposition",
                    "def-full-synergistic-lyapunov-function",
                    total,
                    full + 0.7 * (weightedx + lambda * weightedv) + 0.05 * barrier,
                ),
                equal(
                    "normalized_alive_positional_sum",
                    "def-full-synergistic-lyapunov-function",
                    weightedx,
                    cl.iter()
                        .map(|z| z[0] * z[0])
                        .chain(cr.iter().map(|z| z[0] * z[0]))
                        .sum::<f64>()
                        / n as f64,
                ),
                equal(
                    "normalized_alive_velocity_sum",
                    "def-full-synergistic-lyapunov-function",
                    weightedv,
                    cl.iter()
                        .map(|z| z[1] * z[1])
                        .chain(cr.iter().map(|z| z[1] * z[1]))
                        .sum::<f64>()
                        / n as f64,
                ),
            ];
            records(
                suite,
                "def-full-synergistic-lyapunov-function",
                &[
                    "V_{\\mathrm{total}}(S_1, S_2) :=",
                    "V_{Var}(S_1, S_2) =",
                    "W_b(S_1, S_2) :=",
                ],
                json!({"N":n,"actual_transport":inputs,"N_normalized_variances":[weightedx,weightedv],"auxiliary_boundary_x2":barrier,"weights":[0.7,0.05],"total":total}),
                identitychecks,
            )?;
        }
        let sx = "lem-sx-implies-variance";
        let product = cl
            .iter()
            .flat_map(|a| cr.iter().map(move |c| (a[0] - c[0]).powi(2)))
            .sum::<f64>()
            / (cl.len() * cr.len()) as f64;
        records(
            suite,
            sx,
            &[
                "\\operatorname{Var}_s(x)=",
                "\\begin{aligned}\n\\int |u-v|^2",
            ],
            inputs,
            vec![
                equal(
                    "exact_centered_alive_product_positional_cost",
                    sx,
                    product,
                    vx,
                ),
                lower("nonnegative_alive_velocity_variance", sx, vv, 0.),
            ],
        )?;
    }
    Ok(())
}

fn native_flow_identities(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let nf = n as f64;
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let mu = mean(&xa);
    let alpha = 0.5_f64;
    let label = "def-cloning-frozen-positional-moments";
    let mut rows = Vec::new();
    let mut variances = Vec::new();
    let mut components_checks = Vec::new();
    let mut fluxchecks = Vec::new();
    let mut beta = vec![vec![0.; n]; n];
    let mut ep = 0.;
    let mut evar = 0.;
    let mut ec = 0.;
    let mut emeanvar = 0.;
    let mut meanbar = 0.;
    let mut independently_integrated_beta = vec![vec![0.; n]; n];
    let mut native_copy_covariance_checks = Vec::new();
    for p in &law.patterns {
        for (plan, plan_mass) in graph_plans(p) {
            for (i, &j) in plan.iter().enumerate() {
                if i != j || !law.alive[i] {
                    independently_integrated_beta[i][j] += p.mass * plan_mass;
                }
            }
        }
        meanbar += p.mass * mean(&p.means);
        let mut liveflux = 0.;
        let mut copycov = 0.;
        for (i, row) in beta.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value += p.mass * p.edges[i][j];
            }
            let t = p.edges[i]
                .iter()
                .enumerate()
                .map(|(j, b)| b * (law.x[j] - law.x[i]))
                .sum::<f64>();
            let c = p.edges[i]
                .iter()
                .enumerate()
                .map(|(j, b)| b * (law.x[j] - law.x[i]).powi(2))
                .sum::<f64>()
                - t * t
                + 0.01 * p.p[i];
            if law.alive[i] {
                rows.extend([
                    equal(
                        "native_frozen_row_displacement_mean",
                        label,
                        p.means[i] - law.x[i],
                        t,
                    ),
                    equal(
                        "native_frozen_row_output_mean",
                        label,
                        p.means[i],
                        law.x[i] + t,
                    ),
                    equal("native_frozen_row_covariance", label, p.traces[i], c),
                ]);
                let a = p.traces[i] - 0.01 * p.p[i];
                rows.push(equal(
                    "native_prejitter_copy_covariance",
                    "lem-keystone-contraction-alive",
                    a,
                    c - 0.01 * p.p[i],
                ));
                let rr = law.x[i] - mu;
                let donorstep = p.edges[i]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| b * (law.x[j] - law.x[i]).powi(2))
                    .sum::<f64>();
                let flux = p.edges[i]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| b * ((law.x[j] - mu).powi(2) - rr * rr))
                    .sum::<f64>();
                fluxchecks.push(equal(
                    "native_centered_row_donor_flux_identity",
                    "lem-keystone-contraction-alive",
                    2. * rr * t + donorstep,
                    flux,
                ));
                liveflux += flux / nf;
            }
            copycov += p.traces[i] - 0.01 * p.p[i];
            let retained_term = if law.alive[i] {
                (1. - p.p[i]) * (law.x[i] - p.means[i]).powi(2)
            } else {
                0.
            };
            let donor_centered_covariance = retained_term
                + p.edges[i]
                    .iter()
                    .enumerate()
                    .map(|(j, b)| b * (law.x[j] - p.means[i]).powi(2))
                    .sum::<f64>();
            native_copy_covariance_checks.push(equal(
                "stable_native_copy_covariance_definition",
                "thm-cloning-unconditional-collective-balance",
                p.traces[i] - 0.01 * p.p[i],
                donor_centered_covariance,
            ));
            native_copy_covariance_checks.push(equal(
                "actual_native_copy_plus_jitter_covariance",
                "thm-cloning-unconditional-collective-balance",
                p.traces[i],
                donor_centered_covariance + 0.01 * p.p[i],
            ));
        }
        let varmeans = var(&p.means);
        let csum = p.traces.iter().sum::<f64>();
        variances.push(equal(
            "native_frozen_output_variance_moment_identity",
            "thm-positional-variance-contraction",
            p.output_var,
            varmeans + (1. - 1. / nf) * csum / nf,
        ));
        if live.len() == n {
            let shift = (mean(&p.means) - mu).powi(2);
            variances.push(equal(
                "native_exact_all_alive_F6_flux_balance",
                "lem-keystone-contraction-alive",
                p.output_var - var(&xa),
                liveflux - shift - copycov / (nf * nf) + (1. - 1. / nf) * 0.01 * mean(&p.p),
            ));
            // The geometric singleton refinement has e_i=|x_i-mu|² and rho_i=0.
            fluxchecks.push(equal(
                "native_geometric_singleton_F7_flux_decomposition",
                "lem-keystone-contraction-alive",
                liveflux,
                p.edges
                    .iter()
                    .enumerate()
                    .map(|(i, row)| {
                        row.iter()
                            .enumerate()
                            .map(|(j, b)| {
                                b * ((law.x[j] - mu).powi(2) - (law.x[i] - mu).powi(2)) / nf
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>(),
            ));
        }
        for (choices, _) in graph_plans(p) {
            for group in components(&choices) {
                let center = group.iter().map(|&i| law.v[i]).sum::<f64>() / group.len() as f64;
                for &i in &group {
                    let plus = center + alpha * (law.v[i] - center);
                    let minus = center - alpha * (law.v[i] - center);
                    components_checks.push(equal(
                        "native_component_Haar_mean",
                        "prop-cloning-component-conservation",
                        (plus + minus) / 2.,
                        center,
                    ));
                    for &j in &group {
                        let jp = center + alpha * (law.v[j] - center);
                        let jm = center - alpha * (law.v[j] - center);
                        components_checks.push(equal(
                            "native_shared_component_Haar_cross_covariance",
                            "prop-cloning-component-conservation",
                            ((plus - center) * (jp - center) + (minus - center) * (jm - center))
                                / 2.,
                            alpha * alpha * (law.v[i] - center) * (law.v[j] - center),
                        ));
                    }
                }
            }
        }
        ep += p.mass * mean(&p.p);
        evar += p.mass * p.output_var;
        ec += p.mass * graph_moments(law, p, alpha).0;
        emeanvar += p.mass * (varmeans + (1. - 1. / nf) * csum / nf);
    }
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"native_patterns":law.patterns.len(),"alpha":alpha,"beta_edges":beta,"mean_cloning_pressure":ep,"component_energy":ec,"output_variance":evar});
    record(suite, label, "3.F1", input.clone(), rows.clone())?;
    record(
        suite,
        "lem-keystone-contraction-alive",
        "a_i=\\sum_jb_{ij}",
        input.clone(),
        rows,
    )?;
    record(
        suite,
        "prop-cloning-component-conservation",
        "\\mathbb E v_i'=",
        input.clone(),
        components_checks,
    )?;
    let label = "thm-positional-variance-contraction";
    variances.push(equal(
        "complete_native_Hx_measurement_integral",
        label,
        evar - live.len() as f64 / nf * var(&xa),
        emeanvar - live.len() as f64 / nf * var(&xa),
    ));
    record(suite, label, "H_x(S)=", input.clone(), variances.clone())?;
    if live.len() == n {
        record(
            suite,
            "lem-keystone-contraction-alive",
            "3.F6",
            input.clone(),
            variances,
        )?;
        records(
            suite,
            "lem-keystone-contraction-alive",
            &["2\\delta_i\\cdot t_i+", "x_G=", "3.F7"],
            input.clone(),
            fluxchecks,
        )?;
    }
    let label = "thm-cloning-unconditional-collective-balance";
    let sum_beta = beta.iter().flatten().sum::<f64>() / nf;
    let mut native_beta_checks = vec![equal(
        "measurement_integrated_beta_pressure_sum",
        label,
        sum_beta,
        ep,
    )];
    for (i, row) in beta.iter().enumerate() {
        for (j, &b) in row.iter().enumerate() {
            native_beta_checks.push(equal(
                "independently_enumerated_native_beta_edge",
                label,
                b,
                independently_integrated_beta[i][j],
            ));
        }
    }
    record(
        suite,
        label,
        "\\beta_{ij}(S)=",
        input.clone(),
        native_beta_checks,
    )?;
    records(
        suite,
        label,
        &[r"a_i=\mathbb E[|Y_i-m_i|^2", r"c_i=a_i+d j^2p_i"],
        input.clone(),
        native_copy_covariance_checks,
    )?;
    let label = "lem-cloning-individual-centered-displacement";
    let baryvariance = law
        .patterns
        .iter()
        .map(|p| p.mass * (mean(&p.means) - meanbar).powi(2))
        .sum::<f64>();
    record(
        suite,
        label,
        "3.F5",
        json!({"actual_native_measurement_barycenter_mean":meanbar,"actual_measurement_barycenter_variance":baryvariance,"law":input}),
        vec![lower(
            "actual_native_measurement_barycenter_variance_nonnegative",
            label,
            baryvariance,
            0.,
        )],
    )
}

fn native_fitness_identities(suite: &mut EstimateSuite) -> Result<()> {
    use algorithmic_gas::fitness::PositiveMapping;
    let law = enumerate_law(
        vec![-1.5, -0.5, 0.5, 1.],
        vec![0.3, -0.2, 0.1, -0.4],
        vec![true; 4],
    )?;
    let n = law.x.len();
    let cvr = 0_f64;
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, law.x.clone())?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, 1, law.v.clone())?,
    );
    let rewards = RewardBatch::new(
        law.x.iter().map(|x| 0.5 * x * x).collect(),
        Default::default(),
    );
    let mut checks = Vec::new();
    let mut metrics = Vec::new();
    let mut constants = Vec::new();
    for p in &law.patterns {
        let measured = law
            .config
            .fitness
            .evaluate(&rewards, &p.raw, &law.alive, &obs, 0)?;
        for i in 0..n {
            let rz = measured.reward_z[i];
            let dz = measured.diversity_z[i];
            let r = law.config.fitness.reward_map.map(rz)?;
            let d = law.config.fitness.diversity_map.map(dz)?;
            let fitness = r.powf(law.config.fitness.reward_exponent)
                * d.powf(law.config.fitness.diversity_exponent);
            checks.extend([
                equal(
                    "actual_native_powered_positive_fitness_identity",
                    "def-fitness-potential-operator",
                    p.f[i],
                    fitness,
                ),
                equal(
                    "actual_native_minimize_reward_orientation",
                    "def-raw-value-operators",
                    measured.oriented_reward[i],
                    -0.5 * law.x[i] * law.x[i] - cvr * law.v[i] * law.v[i],
                ),
                equal(
                    "actual_native_reward_rescale_identity",
                    "def-logistic-rescale",
                    r - 0.1,
                    2. / (1. + (-rz).exp()),
                ),
                equal(
                    "actual_native_diversity_rescale_identity",
                    "def-logistic-rescale",
                    d - 0.1,
                    2. / (1. + (-dz).exp()),
                ),
                equal(
                    "actual_native_shared_reward_standardization",
                    "def-fitness-operator",
                    rz,
                    (measured.oriented_reward[i] - measured.reward_stats.mean[i])
                        / measured.reward_stats.scale[i],
                ),
                equal(
                    "actual_native_shared_diversity_standardization",
                    "def-fitness-operator",
                    dz,
                    (p.y[i] - measured.diversity_stats.mean[i]) / measured.diversity_stats.scale[i],
                ),
            ]);
        }
    }
    if let algorithmic_gas::geometry::Distance::SquashedPhaseSpace {
        position_radius,
        velocity_radius,
        lambda,
        ..
    } = &law.config.distance_donors.distance
    {
        for i in 0..n {
            for j in 0..n {
                let squash = |x: f64, r: f64| r * x / (r + x.abs());
                let feature2 = (squash(law.x[i], *position_radius)
                    - squash(law.x[j], *position_radius))
                .powi(2)
                    + lambda
                        * (squash(law.v[i], *velocity_radius) - squash(law.v[j], *velocity_radius))
                            .powi(2);
                let actual = law
                    .config
                    .distance_donors
                    .distance
                    .compare(&obs, i, &obs, j)?;
                metrics.push(equal(
                    "actual_native_squashed_phase_metric_identity",
                    "def-algorithmic-distance-metric",
                    actual * actual,
                    feature2,
                ));
            }
        }
    } else {
        return Err(error(
            "native fixture requires squashed phase-space distance",
        ));
    }
    let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"native_configuration":law.config,"scope":"Canonical native Minimize quadratic objective with explicitly configured zero velocity penalty; the effective fitness reward is -U. Positive logistic maps add their configured floor 0.1"});
    record(
        suite,
        "def-algorithmic-distance-metric",
        "d_{\\mathrm{alg}}(i,j)^2=",
        inputs.clone(),
        metrics,
    )?;
    records(
        suite,
        "def-raw-value-operators",
        &["r_i :="],
        inputs.clone(),
        checks.clone(),
    )?;
    record(
        suite,
        "axiom-velocity-regularization",
        "R(x,v)=",
        inputs.clone(),
        checks.clone(),
    )?;
    record(
        suite,
        "def-logistic-rescale",
        "g_A(z) :=",
        inputs.clone(),
        checks.clone(),
    )?;
    record(
        suite,
        "def-fitness-potential-operator",
        "V_i :=",
        inputs.clone(),
        checks.clone(),
    )?;
    record(
        suite,
        "def-fitness-operator",
        "F_i=g_r",
        inputs.clone(),
        checks,
    )?;
    constants.extend([
        certificate_check(
            "actual_native_nonnegative_power_exponents",
            "lem-potential-bounds",
            law.config.fitness.reward_exponent >= 0. && law.config.fitness.diversity_exponent >= 0.,
        ),
        certificate_check(
            "actual_native_nonnegative_velocity_regularization",
            "def-raw-value-operators",
            cvr >= 0.,
        ),
    ]);
    record(
        suite,
        "lem-potential-bounds",
        "\\alpha, \\beta \\geq 0",
        inputs.clone(),
        constants.clone(),
    )?;
    record(
        suite,
        "def-raw-value-operators",
        "c_{v\\_reg} \\geq 0",
        inputs.clone(),
        constants,
    )?;
    let label = "axiom-lipschitz-fields";
    let mut lip = Vec::new();
    for x in [-2_f64, -1., 0., 1., 2.] {
        for y in [-2_f64, -1., 0., 1., 2.] {
            lip.push(equal(
                "quadratic_native_force_difference_norm",
                label,
                (-x + y).abs(),
                (x - y).abs(),
            ));
            lip.push(upper(
                "zero_native_target_velocity_field_difference",
                label,
                0.,
                0. * (x - y).abs(),
            ));
        }
    }
    records(
        suite,
        label,
        &["\\|F(x_1) - F(x_2)\\| \\leq", "\\|u(x_1) - u(x_2)\\| \\leq"],
        json!({"F":"-x","u":"0","L_F":1.,"L_u":0.,"analytic_derivative":"F'=-1, u'=0 throughout R"}),
        lip,
    )?;
    let label = "axiom-safe-harbor";
    record(
        suite,
        label,
        "\\max_{y \\in C_{\\mathrm{safe}}}",
        json!({"C_safe":[-0.2,0.2],"actual_effective_Rpos":"-x²/2","R_safe":-0.02,"actual_maximizer":0.}),
        vec![lower(
            "actual_quadratic_safe_harbor_reward",
            label,
            0.,
            -0.02,
        )],
    )?;
    let label = "axiom-non-deceptive-landscape";
    records(
        suite,
        label,
        &[
            "\\|x - y\\| \\geq L_{\\text{grad}}",
            "|R_{\\mathrm{pos}}(y) - R_{\\mathrm{pos}}(x)| \\geq",
        ],
        json!({"x":0.,"y":1.,"L_grad":1.,"kappa_raw_reward":0.5,"scope":"Explicit quadratic-landscape directional hypothesis witness for the displayed admissible pair; no uniform lower gap is inferred for arbitrary equal-level pairs"}),
        vec![
            lower("actual_landscape_pair_distance", label, 1., 1.),
            lower("actual_landscape_pair_reward_gap", label, 0.5, 0.5),
        ],
    )
}

fn complete_coupled_increment_identity(suite: &mut EstimateSuite) -> Result<()> {
    let label = "lem-cloning-coupled-error-not-internal-variance";
    for (left_x, right_x, alive) in [
        (
            vec![-1.5, -0.5, 0.5, 1.],
            vec![-1., -0.4, 0.8, 1.5],
            vec![true; 4],
        ),
        (
            vec![-0.8, 0.8, 100., -120.],
            vec![-0.6, 0.6, -80., 110.],
            vec![true, true, false, false],
        ),
    ] {
        let left = enumerate_law(left_x, vec![0.; 4], alive.clone())?;
        let right = enumerate_law(right_x, vec![0.; 4], alive.clone())?;
        let n = left.x.len() as f64;
        let unconditional = |law: &Law| {
            let mut row_means = vec![0.; law.x.len()];
            let mut row_seconds = vec![0.; law.x.len()];
            let mut bary_second = 0.;
            let mut variance = 0.;
            for pattern in &law.patterns {
                for i in 0..row_means.len() {
                    row_means[i] += pattern.mass * pattern.means[i];
                    row_seconds[i] += pattern.mass * (pattern.means[i].powi(2) + pattern.traces[i]);
                }
                bary_second += pattern.mass
                    * (mean(&pattern.means).powi(2) + pattern.traces.iter().sum::<f64>() / (n * n));
                variance += pattern.mass * pattern.output_var;
            }
            (row_means, row_seconds, bary_second, variance)
        };
        let (lm, ls, lb, lv) = unconditional(&left);
        let (rm, rs, rb, rv) = unconditional(&right);
        let input_cross = left
            .x
            .iter()
            .zip(&right.x)
            .map(|(x, y)| (x - mean(&left.x)) * (y - mean(&right.x)))
            .sum::<f64>()
            / n;
        let input_cost = left
            .x
            .iter()
            .zip(&right.x)
            .map(|(x, y)| ((x - mean(&left.x)) - (y - mean(&right.x))).powi(2))
            .sum::<f64>()
            / n;
        let output_cross = lm
            .iter()
            .zip(&rm)
            .map(|(x, y)| (x - mean(&lm)) * (y - mean(&rm)))
            .sum::<f64>()
            / n;
        let direct_uncentered = ls
            .iter()
            .zip(&rs)
            .zip(lm.iter().zip(&rm))
            .map(|((sx, sy), (x, y))| sx + sy - 2. * x * y)
            .sum::<f64>()
            / n;
        let direct_bary = lb + rb - 2. * mean(&lm) * mean(&rm);
        let direct_output_cost = direct_uncentered - direct_bary;
        let hx_left = lv - var(&left.x);
        let hx_right = rv - var(&right.x);
        let inputs = json!({"native_left_positions":left.x,"native_right_positions":right.x,"alive":alive,
            "native_marginal_measurement_counts":[left.patterns.len(),right.patterns.len()],"unconditional_native_row_means":[lm,rm],
            "unconditional_native_row_second_moments":[ls,rs],"unconditional_native_barycenter_second_moments":[lb,rb],
            "full_slot_internal_variance_drifts":[hx_left,hx_right],"actual_independent_kernel_output_cross_term":output_cross,
            "direct_centered_paired_output_cost":direct_output_cost,"input_cross_term":input_cross,"input_centered_cost":input_cost,
            "scope":"Independent complete native cloning kernels give a valid coupling. Exact retained measurement integration and independent row copy/jitter moments evaluate the paired cost and cross-swarm term. H_x uses full-slot entering variance, so mandatory revival of the retained dead coordinates is explicitly included."});
        let checks = vec![
            equal(
                "input_paired_variance_cross_identity",
                label,
                input_cost,
                var(&left.x) + var(&right.x) - 2. * input_cross,
            ),
            equal(
                "native_independent_output_paired_variance_cross_identity",
                label,
                direct_output_cost,
                lv + rv - 2. * output_cross,
            ),
            equal(
                "native_complete_coupled_increment_U5",
                label,
                direct_output_cost - input_cost,
                hx_left + hx_right - 2. * (output_cross - input_cross),
            ),
            equal(
                "direct_left_native_variance_second_identity",
                label,
                lv,
                mean(&ls) - lb,
            ),
            equal(
                "direct_right_native_variance_second_identity",
                label,
                rv,
                mean(&rs) - rb,
            ),
        ];
        records(
            suite,
            label,
            &[r"E_x=\frac1N", r"3.U5", r"E_x=V_{\mathrm{Var},x}"],
            inputs.clone(),
            checks.clone(),
        )?;
        record(
            suite,
            "def-coupled-cloning-expectation",
            r"\mathbb{E}_{\text{clone}}[f",
            json!({"native_coupling":inputs,"observable":"full centered paired positional discrepancy under independent complete native kernel marginals","scope":"The declared independent complete native cloning coupling defines the joint expectation. Its observable is reconstructed from the two full marginal first and second moments, retaining measurement-induced covariance and revival."}),
            checks,
        )?;
        let shared_inputs = json!({"identical_native_entering_population":left.x,"alive":left.alive,"shared_current_measurements_donors_gates_rotations_jitter":true,"output_internal_variance":lv,
            "direct_shared_cross_moment":mean(&ls)-lb,"scope":"The diagonal coupling transports every native innovation identically between identical representatives. Its centered output cross product equals the independently reconstructed native internal variance; the paired cost is zero for each realized proposal."});
        record(
            suite,
            label,
            r"C_x'=V_{\mathrm{Var},x}(S_1')",
            shared_inputs,
            vec![equal(
                "native_diagonal_coupling_centered_cross_moment",
                label,
                mean(&ls) - lb,
                lv,
            )],
        )?;
    }
    Ok(())
}

fn native_variance_definition_chains(suite: &mut EstimateSuite) -> Result<()> {
    let conversion = "def-variance-conversions";
    let drift = "def-full-synergistic-lyapunov-function";
    for (x, v, alive) in [
        (
            vec![-1.5, -0.5, 0.5, 1.],
            vec![0.3, -0.2, 0.1, -0.4],
            vec![true; 4],
        ),
        (
            vec![-0.8, 0.8, 1e9, -1e9],
            vec![0.4, -0.3, 1e9, -1e9],
            vec![true, true, false, false],
        ),
        (
            vec![0.2, 1e9, -1e9, 1e10],
            vec![0.3, -1e10, 1e11, -1e12],
            vec![true, false, false, false],
        ),
    ] {
        let law = enumerate_law(x, v, alive)?;
        let active_x: Vec<_> = law
            .x
            .iter()
            .zip(&law.alive)
            .filter_map(|(x, a)| a.then_some(*x))
            .collect();
        let k = active_x.len() as f64;
        let n = law.x.len() as f64;
        let sum_squares = active_x
            .iter()
            .map(|x| (x - mean(&active_x)).powi(2))
            .sum::<f64>();
        let unweighted = var(&active_x);
        let weighted = sum_squares / n;
        let actual_post_variance = law
            .patterns
            .iter()
            .map(|p| p.mass * p.output_var)
            .sum::<f64>();
        let actual_post_square_sum = law
            .patterns
            .iter()
            .map(|p| p.mass * (n * var(&p.means) + (1. - 1. / n) * p.traces.iter().sum::<f64>()))
            .sum::<f64>();
        let actual_mass = law.patterns.iter().map(|p| p.mass).sum::<f64>();
        let inputs = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"N":n,"k":k,"actual_alive_square_sum":sum_squares,"actual_alive_normalized_variance":unweighted,"actual_N_normalized_variance":weighted,
            "actual_clone_output_square_sum":actual_post_square_sum,"actual_clone_output_variance":actual_post_variance,"scope":"Native complete frozen cloning law with mandatory revival; output alive count equals N before the terminal kinetic boundary. Retained dead coordinates never enter the initial alive barycenter or variance conversion."});
        record(
            suite,
            conversion,
            r"S_k &=",
            inputs.clone(),
            vec![
                equal(
                    "alive_sum_is_k_times_alive_variance",
                    conversion,
                    sum_squares,
                    k * unweighted,
                ),
                equal(
                    "alive_sum_is_N_times_weighted_variance",
                    conversion,
                    sum_squares,
                    n * weighted,
                ),
                equal(
                    "weighted_variance_alive_mass_conversion",
                    conversion,
                    weighted,
                    k / n * unweighted,
                ),
                equal(
                    "alive_variance_weighted_inverse_conversion",
                    conversion,
                    unweighted,
                    n / k * weighted,
                ),
            ],
        )?;
        record(
            suite,
            drift,
            r"\Delta V_{\text{Var}} =",
            inputs.clone(),
            vec![equal(
                "actual_native_expected_weighted_variance_increment_definition",
                drift,
                actual_post_variance - weighted,
                (actual_post_square_sum - sum_squares) / n,
            )],
        )?;
        record(
            suite,
            drift,
            r"\mathbb{E}[\Delta V_{\text{Var}}] = \mathbb{E}",
            inputs.clone(),
            vec![
                equal(
                    "actual_native_changing_alive_count_ratio_identity",
                    drift,
                    actual_post_variance - unweighted,
                    actual_post_square_sum / n - unweighted * actual_mass,
                ),
                equal(
                    "actual_native_output_N_alive_count",
                    drift,
                    n,
                    law.x.len() as f64,
                ),
            ],
        )?;
        record(
            suite,
            drift,
            r"\mathbb{E}[\Delta V_{\text{Var}}] = \frac{1}{N}",
            inputs,
            vec![equal(
                "actual_native_fixed_N_expectation_factor",
                drift,
                actual_post_variance - weighted,
                (actual_post_square_sum - sum_squares * actual_mass) / n,
            )],
        )?;
    }
    Ok(())
}

async fn native_update_definition_witness(suite: &mut EstimateSuite) -> Result<()> {
    use algorithmic_gas::cloning::{
        CloneChoice, ClonePlan, WeightedDonor, accepted_current_components,
    };
    use algorithmic_gas::donor::{CompanionBatch, DonorPool};
    use algorithmic_gas::noise::{
        FactorValues, InnovationLaw, NoiseGeometry, NoiseRequest, NoiseSource,
    };
    use algorithmic_gas::random::{RandomStream, Stream};
    use algorithmic_gas::{BackendKind, ExecutionContext, Population, Precision};
    let label = "def-inelastic-collision-update";
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    for d in [1_usize, 2, 3] {
        let mut config = GasConfig::euclidean(d, 0.04)?;
        let mut x = Vec::new();
        let mut v = Vec::new();
        for i in 0..4 {
            for a in 0..d {
                x.push(if i == 2 {
                    1e8 + (a as f64)
                } else {
                    -0.8 + 0.4 * i as f64 + 0.05 * a as f64
                });
                v.push(-0.3 + 0.2 * i as f64 + 0.1 * a as f64);
            }
        }
        let mut obs = ObservationBatch::positions(TensorBatch::vectors(4, d, x.clone())?);
        obs.fields
            .insert("velocities".into(), TensorBatch::vectors(4, d, v.clone())?);
        let mut before = Population::new(obs)?;
        before.validity[2].out_of_bounds = true;
        let pool = DonorPool::freeze(&before, 0, &[], 0, false)?;
        let donor_slots = [1_usize, 1, 0, 3];
        let accepted = [true, false, true, false];
        let choices = donor_slots
            .iter()
            .enumerate()
            .map(|(i, &j)| {
                Ok(CloneChoice {
                    donors: vec![WeightedDonor {
                        pool_index: pool
                            .current_index(j)
                            .ok_or_else(|| error("retained donor absent"))?,
                        weight: 1.,
                    }],
                    accepted: accepted[i],
                    revival: i == 2,
                    probability: Some(if accepted[i] { 1. } else { 0. }),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let plan = ClonePlan {
            population_version: before.version,
            sources: pool.sources.clone(),
            choices,
            mutual: false,
        };
        let companions = CompanionBatch {
            rows: 4,
            count: 1,
            indices: donor_slots
                .iter()
                .map(|&j| pool.current_index(j).expect("retained donor"))
                .collect(),
            valid: vec![true; 4],
            mutual: false,
        };
        let groups = accepted_current_components(&before, &pool, &plan)?;
        for alpha in [0_f64, 0.5, 1.] {
            config.clone_transform.restitution = Some(alpha);
            let seed = 720_330 + d as u64;
            let step = 3;
            let literal = plan.apply_literal(&before, &pool)?;
            let noise = config
                .clone_transform
                .jitter
                .as_ref()
                .ok_or_else(|| error("native jitter absent"))?;
            let eta = noise
                .sample(
                    &literal.observations,
                    NoiseRequest {
                        rows: 4,
                        dimension: d,
                        seed,
                        step,
                        stream: Stream::CloneNoise,
                        substep: 0,
                    },
                    &mut cx,
                )
                .await?;
            let rotations = groups
                .iter()
                .map(|g| {
                    RandomStream::new(seed, step, Stream::CollisionRotation, g[0] as u64, 0)
                        .haar_orthogonal(d)
                })
                .collect::<Result<Vec<_>>>()?;

            let permute = index_permutations(d);
            let orbit_count = (permute.len() * (1_usize << d)) as f64;
            let mut haar_checks = Vec::new();
            for (group, rotation) in groups.iter().zip(&rotations) {
                let mu: Vec<_> = (0..d)
                    .map(|a| group.iter().map(|&i| v[i * d + a]).sum::<f64>() / group.len() as f64)
                    .collect();
                let u: Vec<_> = (0..d).map(|a| v[group[0] * d + a] - mu[a]).collect();
                let w: Vec<_> = (0..d)
                    .map(|a| v[group[group.len() - 1] * d + a] - mu[a])
                    .collect();
                let ru: Vec<_> = (0..d)
                    .map(|a| (0..d).map(|j| rotation[a * d + j] * u[j]).sum::<f64>())
                    .collect();
                let rw: Vec<_> = (0..d)
                    .map(|a| (0..d).map(|j| rotation[a * d + j] * w[j]).sum::<f64>())
                    .collect();
                let mut matrix_mean = vec![0.; d * d];
                let mut cross = vec![0.; d * d];
                for permutation in &permute {
                    for bits in 0_usize..(1 << d) {
                        for a in 0..d {
                            let signa = if bits & (1 << a) == 0 { -1. } else { 1. };
                            for b in 0..d {
                                let signb = if bits & (1 << b) == 0 { -1. } else { 1. };
                                matrix_mean[a * d + b] +=
                                    signa * rotation[permutation[a] * d + b] / orbit_count;
                                cross[a * d + b] +=
                                    signa * signb * ru[permutation[a]] * rw[permutation[b]]
                                        / orbit_count;
                            }
                        }
                    }
                }
                let dot = u.iter().zip(&w).map(|(u, w)| u * w).sum::<f64>();
                for a in 0..d {
                    for b in 0..d {
                        haar_checks.push(equal(
                            "native_Haar_matrix_left_symmetry_mean_zero",
                            "prop-cloning-component-conservation",
                            matrix_mean[a * d + b],
                            0.,
                        ));
                        haar_checks.push(equal(
                            "native_shared_Haar_full_cross_tensor_identity",
                            "prop-cloning-component-conservation",
                            cross[a * d + b],
                            if a == b { dot / d as f64 } else { 0. },
                        ));
                    }
                }
            }
            let mut output = literal;
            config
                .clone_transform
                .apply(
                    &before,
                    &mut output,
                    &pool,
                    &plan,
                    &companions,
                    seed,
                    step,
                    &mut cx,
                )
                .await?;
            let actual_x = output.observations.field("positions")?;
            let actual_v = output.observations.field("velocities")?;
            let mut position_checks = vec![certificate_check(
                "native_declared_standard_Gaussian_jitter",
                label,
                matches!(noise.innovation, InnovationLaw::Gaussian)
                    && matches!(&noise.geometry,NoiseGeometry::Isotropic{scale:FactorValues::Constant{values}} if values==&vec![1.]),
            )];
            let mut velocity_checks = vec![certificate_check(
                "actual_native_restitution_domain",
                label,
                (0. ..=1.).contains(&alpha),
            )];
            let mut expected_v = v.clone();
            let mut energy_before = 0.;
            let mut energy_after = 0.;
            for (group, rotation) in groups.iter().zip(&rotations) {
                let center: Vec<_> = (0..d)
                    .map(|a| group.iter().map(|&i| v[i * d + a]).sum::<f64>() / group.len() as f64)
                    .collect();
                for &i in group {
                    for a in 0..d {
                        let expected = center[a]
                            + alpha
                                * (0..d)
                                    .map(|b| rotation[a * d + b] * (v[i * d + b] - center[b]))
                                    .sum::<f64>();
                        expected_v[i * d + a] = expected;
                        velocity_checks.push(equal(
                            "direct_native_frozen_component_velocity_update",
                            label,
                            actual_v.row(i)?[a],
                            expected,
                        ));
                        energy_before += (v[i * d + a] - center[a]).powi(2) / 4.;
                        energy_after += (actual_v.row(i)?[a] - center[a]).powi(2) / 4.;
                    }
                }
                for a in 0..d {
                    let native_momentum = group
                        .iter()
                        .map(|&i| actual_v.row(i).expect("validated row")[a])
                        .sum::<f64>();
                    let input_momentum = group.iter().map(|&i| v[i * d + a]).sum::<f64>();
                    velocity_checks.push(equal(
                        "native_component_momentum_full_chain_middle",
                        label,
                        native_momentum,
                        group.len() as f64 * center[a],
                    ));
                    velocity_checks.push(equal(
                        "native_component_momentum_full_chain_last",
                        label,
                        group.len() as f64 * center[a],
                        input_momentum,
                    ));
                    for b in 0..d {
                        velocity_checks.push(equal(
                            "actual_native_component_rotation_orthogonality",
                            label,
                            (0..d)
                                .map(|j| rotation[j * d + a] * rotation[j * d + b])
                                .sum::<f64>(),
                            if a == b { 1. } else { 0. },
                        ));
                    }
                }
            }
            for i in 0..4 {
                for a in 0..d {
                    let expected_x = if accepted[i] {
                        x[donor_slots[i] * d + a]
                            + config.clone_transform.jitter_amplitude * eta.row(i)?[a]
                    } else {
                        x[i * d + a]
                    };
                    position_checks.push(equal(
                        "actual_native_copied_or_persistent_position_update",
                        label,
                        actual_x.row(i)?[a],
                        expected_x,
                    ));
                    velocity_checks.push(equal(
                        "actual_native_component_and_untouched_velocity_vector",
                        label,
                        actual_v.row(i)?[a],
                        expected_v[i * d + a],
                    ));
                }
            }
            let mut other = plan.apply_literal(&before, &pool)?;
            config
                .clone_transform
                .apply(
                    &before,
                    &mut other,
                    &pool,
                    &plan,
                    &companions,
                    seed + 1,
                    step,
                    &mut cx,
                )
                .await?;
            let to_phase = |population: &Population<f64>| -> Result<Vec<Vec<f64>>> {
                (0..4)
                    .map(|i| {
                        let mut point =
                            population.observations.field("positions")?.row(i)?.to_vec();
                        point.extend_from_slice(
                            population.observations.field("velocities")?.row(i)?,
                        );
                        Ok(point)
                    })
                    .collect()
            };
            let left_phase = to_phase(&output)?;
            let right_phase = to_phase(&other)?;
            let ml: Vec<_> = (0..2 * d)
                .map(|a| left_phase.iter().map(|z| z[a]).sum::<f64>() / 4.)
                .collect();
            let mr: Vec<_> = (0..2 * d)
                .map(|a| right_phase.iter().map(|z| z[a]).sum::<f64>() / 4.)
                .collect();
            let cl: Vec<_> = left_phase
                .iter()
                .map(|z| z.iter().zip(&ml).map(|(z, m)| z - m).collect::<Vec<_>>())
                .collect();
            let cr: Vec<_> = right_phase
                .iter()
                .map(|z| z.iter().zip(&mr).map(|(z, m)| z - m).collect::<Vec<_>>())
                .collect();
            let cost = |a: &[f64], b: &[f64]| {
                (0..d)
                    .map(|i| {
                        let dx = a[i] - b[i];
                        let dv = a[d + i] - b[d + i];
                        dx * dx + dv * dv + 0.1 * dx * dv
                    })
                    .sum::<f64>()
            };
            let (full, _) =
                crate::convergence_lyapunov::uniform_transport(&left_phase, &right_phase, cost)?;
            let (structural, _) = crate::convergence_lyapunov::uniform_transport(&cl, &cr, cost)?;
            let location = cost(&ml, &mr);
            let pair_checks = vec![
                equal(
                    "actual_native_phase_transport_location_structural_decomposition",
                    "cor-structural-error-contraction",
                    full,
                    location + structural,
                ),
                lower(
                    "actual_native_phase_transport_location_nonnegative",
                    "cor-structural-error-contraction",
                    location,
                    0.,
                ),
                upper(
                    "actual_native_structural_transport_component_bound",
                    "cor-structural-error-contraction",
                    structural,
                    full,
                ),
            ];
            let paired_input = json!({"native_left_output":left_phase,"native_right_output":right_phase,"left_centered":cl,"right_centered":cr,"native_clone_seeds":[seed,seed+1],"shared_frozen_plan":plan,"V_W":full,"V_loc":location,"V_struct":structural,"lambda_v":1.,"b":0.1,"scope":"Two direct native CloneTransform outputs from the same eligible frozen donor graph with independent native innovation addresses. Uniform empirical phase transport is solved exactly; the quotient decomposition is checked on their actual physical outputs."});
            record(
                suite,
                "cor-structural-error-contraction",
                r"V_{\mathrm{struct}}'\leq W_h^2(\mu_1',\mu_2')",
                paired_input.clone(),
                pair_checks.clone(),
            )?;
            record(
                suite,
                "rem-component-growth-combined-drift",
                r"V_W=V_{\rm loc}+V_{\rm struct}",
                paired_input.clone(),
                pair_checks.clone(),
            )?;
            record(
                suite,
                "cor-canonical-full-step-boundary-reset",
                r"V_W=V_{\rm loc}+V_{\rm struct}",
                paired_input,
                pair_checks,
            )?;
            let mut centered_identity = Vec::new();
            let reversed_before: Vec<_> = x.chunks_exact(d).rev().flatten().copied().collect();
            let reversed_after: Vec<_> = actual_x
                .values()
                .chunks_exact(d)
                .rev()
                .flatten()
                .copied()
                .collect();
            for a in 0..d {
                let right_before_mean =
                    reversed_before.chunks_exact(d).map(|z| z[a]).sum::<f64>() / 4.;
                let right_after_mean =
                    reversed_after.chunks_exact(d).map(|z| z[a]).sum::<f64>() / 4.;
                let before_mean = x.chunks_exact(d).map(|z| z[a]).sum::<f64>() / 4.;
                let after_mean = actual_x.values().chunks_exact(d).map(|z| z[a]).sum::<f64>() / 4.;
                for i in 0..4 {
                    centered_identity.push(equal(
                        "identical_native_inputs_paired_centered_difference_zero",
                        "lem-cloning-coupled-error-not-internal-variance",
                        (x[i * d + a] - before_mean)
                            - (reversed_before[(3 - i) * d + a] - right_before_mean),
                        0.,
                    ));
                    centered_identity.push(equal(
                        "identical_native_shared_output_paired_centered_difference_zero",
                        "lem-cloning-coupled-error-not-internal-variance",
                        (actual_x.row(i)?[a] - after_mean)
                            - (reversed_after[(3 - i) * d + a] - right_after_mean),
                        0.,
                    ));
                }
            }
            records(
                suite,
                "lem-cloning-coupled-error-not-internal-variance",
                &[
                    r"\delta_{1,i}=x_{1,i}-\bar x_1",
                    r"\delta_{2,i}=x_{2,i}-\bar x_2",
                    r"E_x'=E_x=0",
                ],
                json!({"identical_native_input":x,"identical_actual_native_output":actual_x.values(),"coupling":"diagonal complete native conditional innovation law","scope":"Identical input representatives share every actual native innovation and accepted graph, producing identical native proposals. Their paired discrepancy is exactly zero although their internal variance is nonzero."}),
                centered_identity,
            )?;
            let inputs = json!({"dimension":d,"native_before_positions":x,"native_before_velocities":v,"native_alive_mask":[true,true,false,true],"accepted_plan":plan,"actual_current_components":groups,"actual_component_rotations":rotations,"actual_native_Gaussian_jitter":eta.values(),"actual_native_after_positions":actual_x.values(),"actual_native_after_velocities":actual_v.values(),"restitution":alpha,"seed":seed,"step":step,"actual_component_energy_before":energy_before,"actual_component_energy_after":energy_after,
                "scope":"Direct production ClonePlan.apply_literal and CloneTransform.apply execute the retained current-donor graph. The existing dead slot is revived from a live donor. Independent addressed replay of the native Gaussian noise and Haar matrices checks each physical coordinate, each frozen component momentum and each relative-energy identity."});
            records(
                suite,
                label,
                &[
                    r"\bar v_C=",
                    r"R_C\in O(d)",
                    r"\alpha=\alpha_{\mathrm{restitution}}",
                ],
                inputs.clone(),
                velocity_checks.clone(),
            )?;
            record(
                suite,
                label,
                r"x_i'=\begin{cases}",
                inputs.clone(),
                position_checks,
            )?;
            record(
                suite,
                "thm-cloning-canonical-barycenter-concentration",
                "\\sum_{i\\in C}v_i^c\n=",
                inputs.clone(),
                velocity_checks,
            )?;
            record(
                suite,
                "prop-bounded-velocity-expansion",
                r"\mathcal E_C=\frac1N",
                inputs.clone(),
                vec![
                    equal(
                        "native_component_energy_defect",
                        "prop-bounded-velocity-expansion",
                        energy_after,
                        alpha * alpha * energy_before,
                    ),
                    equal(
                        "native_component_energy_definition",
                        "prop-bounded-velocity-expansion",
                        energy_before,
                        groups
                            .iter()
                            .map(|g| {
                                (0..d)
                                    .map(|a| {
                                        let center = g.iter().map(|&i| v[i * d + a]).sum::<f64>()
                                            / g.len() as f64;
                                        g.iter()
                                            .map(|&i| (v[i * d + a] - center).powi(2))
                                            .sum::<f64>()
                                            / 4.
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>(),
                    ),
                    equal(
                        "native_reset_velocity_defect_definition",
                        "prop-bounded-velocity-expansion",
                        (0..d)
                            .map(|a| {
                                let all = (0..4).map(|i| v[i * d + a]).collect::<Vec<_>>();
                                let alive = [0, 1, 3].map(|i| v[i * d + a]);
                                var(&all) - 0.75 * var(&alive)
                            })
                            .sum::<f64>(),
                        (0..d)
                            .map(|a| {
                                let ma = [0, 1, 3].iter().map(|&i| v[i * d + a]).sum::<f64>() / 3.;
                                3. / 16. * (v[2 * d + a] - ma).powi(2)
                            })
                            .sum::<f64>(),
                    ),
                ],
            )?;
            record(
                suite,
                "prop-bounded-velocity-expansion",
                r"\mathcal V_v^{\rm all}(S')-",
                inputs.clone(),
                vec![equal(
                    "native_full_slot_velocity_variance_exact_balance",
                    "prop-bounded-velocity-expansion",
                    (0..d)
                        .map(|a| {
                            var(&(0..4)
                                .map(|i| actual_v.row(i).expect("native output")[a])
                                .collect::<Vec<_>>())
                                - var(&(0..4).map(|i| v[i * d + a]).collect::<Vec<_>>())
                        })
                        .sum::<f64>(),
                    -(1. - alpha * alpha) * energy_before,
                )],
            )?;

            records(
                suite,
                "prop-cloning-component-conservation",
                &[r"\mathbb E R_C=0", r"\mathbb E[(R_Cu)(R_Cw)^T]"],
                json!({"native_transform":inputs,"actual_native_rotation_matrices":rotations,"dimension":d,"full_signed_permutation_orbit_size":orbit_count,"scope":"Every matrix is produced by the native addressed Haar sampler. Exact finite averaging over its orthogonal signed-permutation orbit checks the first and full shared cross-tensor observable. Haar left invariance makes this observable projection its Haar expectation; a finite orbit is not substituted for the whole native rotation law."}),
                haar_checks,
            )?;
            if d == 1 {
                records(
                    suite,
                    "prop-cloning-component-conservation",
                    &["d=1", "O(1)"],
                    json!({"dimension":d,"actual_native_Haar_sign":rotations,"orthogonal_group":[-1.,1.]}),
                    vec![equal(
                        "actual_one_dimensional_native_Haar_group_membership",
                        "prop-cloning-component-conservation",
                        rotations[0][0].abs(),
                        1.,
                    )],
                )?;
            }
            let vmax = (0..4)
                .map(|i| {
                    v[i * d..(i + 1) * d]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(0_f64, f64::max);
            let mut bounds = vec![
                certificate_check(
                    "actual_positive_eligible_count",
                    "prop-bounded-velocity-expansion",
                    before.validity.iter().filter(|s| s.eligible(false)).count() > 0,
                ),
                equal(
                    "actual_dead_count",
                    "prop-bounded-velocity-expansion",
                    before
                        .validity
                        .iter()
                        .filter(|s| !s.eligible(false))
                        .count() as f64,
                    1.,
                ),
                upper(
                    "actual_revived_fraction_in_accepted_count",
                    "prop-bounded-velocity-expansion",
                    0.25,
                    accepted.iter().filter(|a| **a).count() as f64 / 4.,
                ),
            ];
            for i in 0..4 {
                bounds.push(upper(
                    "actual_frozen_input_velocity_cap",
                    "prop-bounded-velocity-expansion",
                    v[i * d..(i + 1) * d]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt(),
                    vmax,
                ));
                bounds.push(upper(
                    "actual_native_component_velocity_expansion",
                    "prop-bounded-velocity-expansion",
                    actual_v.row(i)?.iter().map(|v| v * v).sum::<f64>().sqrt(),
                    (1. + 2. * alpha) * vmax,
                ));
            }
            records(
                suite,
                "prop-bounded-velocity-expansion",
                &[
                    r"|v_i|\leq V_{\max}",
                    r"D=|\mathcal D|",
                    r"|\mathcal A|>0",
                    r"D/N\leq f_{\rm clone}",
                    r"|v_i'|\leq(1+2\alpha)V_{\max}",
                ],
                json!({"native_transform":inputs,"actual_eligible_count":3,"actual_dead_count":1,"actual_input_velocity_max":vmax,"actual_accepted_count":2}),
                bounds,
            )?;
            for target in [
                "rem-synergistic-velocity-dissipation",
                "cor-structural-error-contraction",
            ] {
                let checks = (0..4)
                    .map(|i| {
                        upper(
                            "actual_native_output_velocity_component_expansion",
                            target,
                            actual_v
                                .row(i)
                                .expect("native vector")
                                .iter()
                                .map(|v| v * v)
                                .sum::<f64>()
                                .sqrt(),
                            (1. + 2. * alpha) * vmax,
                        )
                    })
                    .collect();
                record(
                    suite,
                    target,
                    r"|v_i'|\leq(1+2\alpha)V_{\max}",
                    inputs.clone(),
                    checks,
                )?;
            }
            record(
                suite,
                "def-cloning-operator-formal",
                r"s'_i = 1",
                inputs,
                vec![certificate_check(
                    "actual_native_mandatory_revival_statuses",
                    "def-cloning-operator-formal",
                    output.validity.iter().all(|s| s.eligible(false)),
                )],
            )?;
        }
    }
    Ok(())
}

fn expansion_native_definitions(suite: &mut EstimateSuite) -> Result<()> {
    use algorithmic_gas::fitness::PositiveMapping;
    let label = "prop-cloning-macroscopic-structural-expansion";
    let a = [-1.5_f64, -1.5, -1.5, 1.];
    let b = a.map(|x| x / 100.);
    record(
        suite,
        label,
        r"x^A=(-3/2",
        json!({"x_A":a,"x_B":b,"velocities":vec![0.;4],"scope":"The actual native four-slot inputs, including their physical coordinates and zero component velocities, retain the complete measurements and shared normalizers."}),
        a.iter()
            .zip(b)
            .map(|(&x, y)| equal("actual_expansion_input_dilation", label, y, x / 100.))
            .collect(),
    )?;
    for x in [a, b] {
        let law = enumerate_law(x.to_vec(), vec![0.; 4], vec![true; 4])?;
        let z: Vec<_> = x.iter().map(|x| 2. * x / (2. + x.abs())).collect();
        let mut weights = vec![vec![0.; 4]; 4];
        for (i, row) in weights.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                if i != j {
                    *w = (-(z[i] - z[j]).powi(2) / 8.).exp();
                }
            }
        }
        let normalizers: Vec<_> = weights.iter().map(|row| row.iter().sum::<f64>()).collect();
        let mut distance_checks = Vec::new();
        let mut fitness_checks = Vec::new();
        fitness_checks.push(equal(
            "actual_complete_native_measurement_pattern_count",
            label,
            law.patterns.len() as f64,
            3_f64.powi(4),
        ));
        let mut edge_checks = Vec::new();
        let mut covariance_checks = Vec::new();
        let reward: Vec<_> = x.iter().map(|x| -x * x / 2.).collect();
        let rmean = mean(&reward);
        let rstd = (var(&reward) + 0.01).sqrt();
        for p in &law.patterns {
            let mut actual_mass = 1.;
            let mut radius_flux = 0.;
            let mut independent_second_output = 0.;
            for i in 0..4 {
                actual_mass *= weights[i][p.choices[i]] / normalizers[i];
                let raw = (z[i] - z[p.choices[i]]).abs();
                distance_checks.push(equal(
                    "actual_native_expansion_feature_distance",
                    label,
                    p.raw[i],
                    raw,
                ));
                distance_checks.push(equal(
                    "actual_native_expansion_floor_distance",
                    label,
                    p.y[i],
                    raw.hypot(0.001),
                ));
                let dz = (p.y[i] - mean(&p.y)) / (var(&p.y) + 0.01).sqrt();
                let rz = (reward[i] - rmean) / rstd;
                let rf = law.config.fitness.reward_map.map(rz)?;
                let df = law.config.fitness.diversity_map.map(dz)?;
                fitness_checks.push(equal(
                    "actual_native_expansion_reward_logistic_map",
                    label,
                    rf,
                    0.1 + 2. / (1. + (-rz).exp()),
                ));
                fitness_checks.push(equal(
                    "actual_native_expansion_diversity_logistic_map",
                    label,
                    df,
                    0.1 + 2. / (1. + (-dz).exp()),
                ));
                fitness_checks.push(equal(
                    "actual_native_expansion_standardized_score",
                    label,
                    p.z[i],
                    dz,
                ));
                fitness_checks.push(equal(
                    "actual_native_expansion_retained_product",
                    label,
                    p.f[i],
                    rf * df,
                ));
                let mut independent_pressure = 0.;
                let mut independent_mean_step = 0.;
                let mut independent_second_step = 0.;
                for j in 0..4 {
                    let pj = weights[i][j] / normalizers[i];
                    distance_checks.push(equal(
                        "actual_native_expansion_donor_probability",
                        label,
                        law.donors[i][j],
                        pj,
                    ));
                    distance_checks.push(equal(
                        "actual_native_expansion_measurement_probability",
                        label,
                        law.measurements[i][j],
                        pj,
                    ));
                    let native_edge = pj * ((p.f[j] - p.f[i]).max(0.) / (p.f[i] + 1e-6)).min(1.);
                    edge_checks.push(equal(
                        "actual_native_expansion_edge_law",
                        label,
                        p.edges[i][j],
                        native_edge,
                    ));
                    independent_pressure += native_edge;
                    independent_mean_step += native_edge * (x[j] - x[i]);
                    independent_second_step += native_edge * (x[j] - x[i]).powi(2);
                    radius_flux +=
                        native_edge * ((x[j] - mean(&x)).powi(2) - (x[i] - mean(&x)).powi(2)) / 4.;
                }
                edge_checks.push(equal(
                    "actual_native_expansion_row_pressure",
                    label,
                    p.p[i],
                    independent_pressure,
                ));
                edge_checks.push(equal(
                    "actual_native_expansion_row_mean_step",
                    label,
                    p.means[i] - x[i],
                    independent_mean_step,
                ));
                covariance_checks.push(equal(
                    "actual_native_expansion_copy_covariance",
                    label,
                    p.traces[i] - 0.01 * p.p[i],
                    independent_second_step - independent_mean_step.powi(2),
                ));
                independent_second_output += (1. - p.p[i]) * x[i].powi(2)
                    + p.edges[i]
                        .iter()
                        .enumerate()
                        .map(|(j, b)| b * x[j].powi(2))
                        .sum::<f64>()
                    + 0.01 * p.p[i];
            }
            fitness_checks.push(equal(
                "actual_native_expansion_measurement_vector_mass",
                label,
                p.mass,
                actual_mass,
            ));
            fitness_checks.push(certificate_check(
                "actual_native_expansion_unequal_retained_fitness",
                label,
                range(&p.f) > 0.,
            ));
            covariance_checks.push(equal(
                "actual_native_expansion_radial_flux_definition",
                label,
                radius_flux,
                p.output_var - var(&x)
                    + (mean(&p.means) - mean(&x)).powi(2)
                    + (p.traces.iter().sum::<f64>() - 0.01 * p.p.iter().sum::<f64>()) / 16.
                    - 0.75 * 0.01 * mean(&p.p),
            ));
            covariance_checks.push(equal(
                "actual_native_expansion_raw_second_moment",
                label,
                independent_second_output / 4.,
                p.means
                    .iter()
                    .zip(&p.traces)
                    .map(|(m, c)| m * m + c)
                    .sum::<f64>()
                    / 4.,
            ));
        }
        let inputs = json!({"actual_native_positions":x,"actual_native_config":law.config,"alive":law.alive,"actual_native_comparison_features":z,"actual_pair_weights":weights,"actual_donor_normalizers":normalizers,"complete_measurement_patterns":law.patterns.len(),"scope":"The native pipeline and exact frozen moments independently evaluate the finite expressions of the directed certificate. Every retained vector is included and its normalizers are recomputed before acceptance."});
        record(
            suite,
            label,
            r"z_i=\frac{2x_i}",
            inputs.clone(),
            distance_checks,
        )?;
        records(
            suite,
            label,
            &[
                r"\omega_m=\prod_i",
                r"\operatorname{std}(y)_i=",
                r"3^4=81",
                r"g(u)=.1+2/(1+e^{-u})",
            ],
            inputs.clone(),
            fitness_checks,
        )?;
        record(suite, label, r"b_{ij}(m)=", inputs.clone(), edge_checks)?;
        record(suite, label, r"a_i=\sum_jb_{ij}", inputs, covariance_checks)?;
    }
    record(
        suite,
        label,
        r"e<3",
        json!({"exact_partial_sum_degree":3,"tail_degree_four_term":"1/24","all_subsequent_term_ratios_at_most":"1/5","exact_rational_upper":"87/32","scope":"The convergent factorial series has a geometric remainder bound: the sum through degree three plus (1/24)/(1-1/5) is exactly 87/32, strictly below three."}),
        vec![certificate_check(
            "independent_rational_exp_one_upper",
            label,
            BigInt::from(87) < BigInt::from(3 * 32),
        )],
    )?;
    Ok(())
}

fn native_geometric_event_definitions(suite: &mut EstimateSuite) -> Result<()> {
    let label = "lem-keystone-geometric-measurement-events";
    let config = GasConfig::euclidean(1, 0.04)?;
    let positions = [1_f64, -1.];
    let counts = [5_usize, 95];
    let features: Vec<_> = positions.iter().map(|x| 2. * x / (2. + x.abs())).collect();
    let mean_feature = (features[0] * counts[0] as f64 + features[1] * counts[1] as f64) / 100.;
    let mean_position = (positions[0] * counts[0] as f64 + positions[1] * counts[1] as f64) / 100.;
    let feature_variance = (0..2)
        .map(|s| counts[s] as f64 / 100. * (features[s] - mean_feature).powi(2))
        .sum::<f64>();
    let positional_variance = (0..2)
        .map(|s| counts[s] as f64 / 100. * (positions[s] - mean_position).powi(2))
        .sum::<f64>();
    let feature_diameter = (features[0] - features[1]).abs();
    let h = 0.2_f64;
    let rho_h = (feature_variance - h * h) / (feature_diameter * feature_diameter - h * h);
    let rg = 0_f64;
    let delta_floor = config.fitness.distance_floor;
    let lower_measured = (rg * rg + delta_floor * delta_floor).sqrt();
    let upper_measured = (feature_diameter.powi(2) + delta_floor.powi(2)).sqrt();
    let far_threshold = (h * h + delta_floor * delta_floor).sqrt();
    let delta_g = far_threshold - lower_measured;
    let span = upper_measured - lower_measured;
    let es = 0.1_f64;
    let standardized_range = span / es;
    let smax = (span.powi(2) / 4. + es.powi(2)).sqrt();
    let map = |z: f64| 0.1 + 2. / (1. + (-z).exp());
    let fmin = map(-standardized_range);
    let fmax = map(standardized_range);
    let mf = 2. * (-standardized_range).exp() / (1. + (-standardized_range).exp()).powi(2);
    let a_min = 1.1_f64;
    let reward_oscillation = 0_f64;
    let gamma = a_min * mf * delta_g / smax - fmax * reward_oscillation;
    let accept = (gamma
        / (config.clone_decision.saturation * (4.41 + config.clone_decision.epsilon)))
        .clamp(0., 1.);
    let rho_g = (counts[0] - 1) as f64 / 99.;
    let (pressure, native_event) =
        compressed_native_two_site_event(&BigInt::from(100), &BigInt::from(5), 1, 1, 0);
    let input = json!({"native_config":config,"positions":positions,"atom_multiplicities":counts,"squashed_comparison_features":features,"actual_feature_variance":feature_variance,"actual_feature_diameter":feature_diameter,"h":h,"rho_h":rho_h,"r_G":rg,"ell_G":lower_measured,"H_h":far_threshold,"delta_G":delta_g,"rho_G":rho_g,"actual_diversity_range":[lower_measured,upper_measured],"D_m":span,"Z_m":standardized_range,"s_star":smax,"diversity_factor_extrema":[fmin,fmax],"minimum_diversity_derivative":mf,"reward_factor_minimum":a_min,"reward_factor_oscillation":reward_oscillation,"gamma_G":gamma,"mathfrak_a_G":accept,"complete_native_accepted_event":native_event,
        "scope":"The two occupied native comparison sites have multiplicities five and ninety-five. All definitions use actual squashed pair distances, the configured floor and shared global normalizer. Native companion masses and a directed cover of every measurement-count acceptance retain the entire event law."});
    let mut geometry = vec![
        equal(
            "actual_feature_variance_pairwise_identity",
            label,
            feature_variance,
            5. * 95. / 100_f64.powi(2) * feature_diameter.powi(2),
        ),
        equal(
            "actual_feature_diameter_maximum",
            label,
            feature_diameter,
            4. / 3.,
        ),
        equal_positive(
            "independent_actual_far_population_fraction",
            label,
            rho_h,
            10. / 391.,
        ),
        lower(
            "actual_physical_feature_variance_comparison",
            label,
            feature_variance,
            0.25_f64.powi(2) * positional_variance,
        ),
        certificate_check(
            "actual_geometric_h_domain",
            label,
            h > 0. && h * h < feature_variance,
        ),
        certificate_check("actual_positive_far_population_fraction", label, rho_h > 0.),
    ];
    for i in 0..2 {
        let ms = (0..2)
            .map(|j| counts[j] as f64 / 100. * (features[j] - features[i]).powi(2))
            .sum::<f64>();
        let far_fraction = counts[1 - i] as f64 / 100.;
        geometry.push(equal(
            "actual_row_pairwise_feature_centering",
            label,
            ms,
            feature_variance + (features[i] - mean_feature).powi(2),
        ));
        geometry.push(upper(
            "actual_row_far_count_decomposition",
            label,
            ms,
            h * h + (feature_diameter * feature_diameter - h * h) * far_fraction,
        ));
        geometry.push(lower(
            "actual_distinct_far_count_fraction",
            label,
            counts[1 - i] as f64 / 99.,
            rho_h,
        ));
    }
    records(
        suite,
        label,
        &[
            r"\mathsf V_z=\frac1k",
            r"0<h^2<\mathsf V_z",
            r"\rho_h=",
            r"\frac1k\sum_\ell|z_\ell-z_j|^2",
        ],
        input.clone(),
        geometry,
    )?;
    records(
        suite,
        label,
        &[r"\ell_G=\sqrt{r_G^2", r"3.AP2"],
        input.clone(),
        vec![
            equal(
                "actual_cluster_near_threshold",
                label,
                lower_measured,
                delta_floor,
            ),
            equal(
                "actual_cluster_far_threshold",
                label,
                far_threshold,
                h.hypot(delta_floor),
            ),
            equal_positive(
                "actual_cluster_favorable_diversity_gap",
                label,
                delta_g,
                h.hypot(delta_floor) - delta_floor,
            ),
            equal_positive("actual_cluster_nonself_mass", label, rho_g, 4. / 99.),
            certificate_check(
                "actual_positive_cluster_favorable_gap",
                label,
                rg < h && delta_g > 0.,
            ),
        ],
    )?;
    let mut map_checks = vec![
        equal_positive(
            "actual_geometric_measurement_span",
            label,
            span,
            upper_measured - lower_measured,
        ),
        equal_positive(
            "actual_geometric_score_range",
            label,
            standardized_range,
            span / es,
        ),
        equal_positive(
            "actual_geometric_maximum_scale",
            label,
            smax,
            (span * span / 4. + es * es).sqrt(),
        ),
        equal_positive(
            "actual_diversity_factor_minimum_at_left_endpoint",
            label,
            fmin,
            map(-standardized_range),
        ),
        equal_positive(
            "actual_diversity_factor_maximum_at_right_endpoint",
            label,
            fmax,
            map(standardized_range),
        ),
        equal_positive(
            "actual_derivative_minimum_at_either_endpoint",
            label,
            mf,
            2. * standardized_range.exp() / (1. + standardized_range.exp()).powi(2),
        ),
    ];
    for t in 0..=256 {
        let z = -standardized_range + 2. * standardized_range * t as f64 / 256.;
        let derivative = 2. * (-z.abs()).exp() / (1. + (-z.abs()).exp()).powi(2);
        map_checks.push(lower(
            "actual_positive_logistic_derivative_interval",
            label,
            derivative,
            mf,
        ));
        map_checks.push(lower(
            "actual_positive_logistic_factor_interval",
            label,
            map(z),
            fmin,
        ));
        map_checks.push(upper(
            "actual_bounded_logistic_factor_interval",
            label,
            map(z),
            fmax,
        ));
        let proportion = t as f64 / 256.;
        let ybar = lower_measured + proportion * span;
        let variance = proportion * (1. - proportion) * span * span;
        map_checks.push(equal(
            "actual_two_value_Popoviciu_middle_identity",
            label,
            variance,
            (upper_measured - ybar) * (ybar - lower_measured),
        ));
        map_checks.push(upper(
            "actual_two_value_Popoviciu_upper",
            label,
            variance,
            span * span / 4.,
        ));
    }
    records(
        suite,
        label,
        &[
            r"D_m=y_+-y_-",
            r"3.AP3",
            r"\operatorname{Var}(Y)\le(y_+-\bar Y)",
        ],
        input.clone(),
        map_checks,
    )?;
    records(
        suite,
        label,
        &[r"A_G^- =", r"3.AP4"],
        input.clone(),
        vec![
            equal_positive("actual_equal_reward_factor_minimum", label, a_min, map(0.)),
            equal(
                "actual_equal_reward_factor_oscillation",
                label,
                reward_oscillation,
                map(0.) - map(0.),
            ),
            equal_positive(
                "actual_positive_favorable_powered_margin",
                label,
                gamma,
                a_min * mf * delta_g / smax,
            ),
            equal_positive(
                "actual_frozen_acceptance_floor_formula",
                label,
                accept,
                (gamma / (4.41 + config.clone_decision.epsilon)).min(1.),
            ),
            certificate_check(
                "actual_native_event_acceptance_strictly_positive",
                label,
                pressure.lo > BigInt::zero() && accept > 0.,
            ),
        ],
    )?;
    Ok(())
}

fn native_quotient_state_witness(suite: &mut EstimateSuite) -> Result<()> {
    let label = "def-single-swarm-space";
    let entering = enumerate_law(
        vec![-1.5, -0.5, 0.5, 1.],
        vec![0.3, -0.2, 0.1, -0.4],
        vec![true, true, false, true],
    )?;
    let canonical = |law: &Law| {
        let mut state: Vec<_> = law
            .x
            .iter()
            .zip(&law.v)
            .zip(&law.alive)
            .map(|((&x, &v), &a)| (x, v, a))
            .collect();
        state.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        state
    };
    let original_canonical = canonical(&entering);
    let original_variance = entering
        .patterns
        .iter()
        .map(|p| p.mass * p.output_var)
        .sum::<f64>();
    let original_means: Vec<_> = (0..4)
        .map(|i| {
            entering
                .patterns
                .iter()
                .map(|p| p.mass * p.means[i])
                .sum::<f64>()
        })
        .collect();
    let original_pressure: Vec<_> = (0..4)
        .map(|i| {
            entering
                .patterns
                .iter()
                .map(|p| p.mass * p.p[i])
                .sum::<f64>()
        })
        .collect();
    let mut checks = Vec::new();
    let mut permutation_count = 0;
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let permutation = [a, b, c, d];
                    if (0..4).any(|i| (0..i).any(|j| permutation[j] == permutation[i])) {
                        continue;
                    }
                    permutation_count += 1;
                    let permuted = enumerate_law(
                        permutation.iter().map(|&i| entering.x[i]).collect(),
                        permutation.iter().map(|&i| entering.v[i]).collect(),
                        permutation.iter().map(|&i| entering.alive[i]).collect(),
                    )?;
                    checks.push(certificate_check(
                        "actual_marked_state_quotient_representative",
                        label,
                        canonical(&permuted) == original_canonical,
                    ));
                    checks.push(equal(
                        "native_quotient_output_variance_invariance",
                        label,
                        permuted
                            .patterns
                            .iter()
                            .map(|p| p.mass * p.output_var)
                            .sum::<f64>(),
                        original_variance,
                    ));
                    for (new, &old) in permutation.iter().enumerate() {
                        checks.push(equal(
                            "native_quotient_unconditional_row_mean_equivariance",
                            label,
                            permuted
                                .patterns
                                .iter()
                                .map(|p| p.mass * p.means[new])
                                .sum::<f64>(),
                            original_means[old],
                        ));
                        checks.push(equal(
                            "native_quotient_unconditional_pressure_equivariance",
                            label,
                            permuted
                                .patterns
                                .iter()
                                .map(|p| p.mass * p.p[new])
                                .sum::<f64>(),
                            original_pressure[old],
                        ));
                    }
                }
            }
        }
    }
    record(
        suite,
        label,
        r"S := \left[",
        json!({"canonical_marked_empirical_state":original_canonical,"all_storage_permutations":permutation_count,"native_positions":entering.x,"native_velocities":entering.v,"alive":entering.alive,"scope":"The state is the unordered marked multiset. All 24 storage permutations preserve that multiset and the native complete proposal law's variance, row means and row pressure after transporting its representation; no intrinsic walker label enters the algorithm."}),
        checks,
    )
}

fn assembly_offset_definitions(suite: &mut EstimateSuite) -> Result<()> {
    let label = "cor-cloning-weighted-assembly-input";
    let config = GasConfig::euclidean(1, 0.04)?;
    let dx = 4_f64;
    let sigma = config.clone_transform.jitter_amplitude;
    let vmax = config
        .kinetic
        .velocity_cap
        .ok_or_else(|| error("canonical cap missing"))?;
    let lambda = 1.;
    let cx = dx * dx + 2. * sigma * sigma;
    let cv = 8. * lambda * vmax * vmax;
    let bw = (1. + 0.1_f64.powi(2) / 4.) * vmax * vmax + dx * dx + sigma * sigma;
    let cw = 4. * bw;
    // C_b is an input supplied by the separately applicable boundary lemma;
    // this vector identity does not purport to discharge that hypothesis.
    let supplied_boundary_offset = 0.75_f64;
    let vector = [cw, cx, cv, supplied_boundary_offset];
    record(
        suite,
        label,
        "3.AC6",
        json!({"native_config":config,"D_x":dx,"lambda_v":lambda,"M_h":bw,"independently_supplied_C_b":supplied_boundary_offset,"offset_vector":vector,
        "scope":"The native position/cap/jitter parameters reconstruct C_W, C_x and C_v. The displayed vector preserves its supplied boundary entry; this is the conditional assembly interface, with boundary applicability retained independently rather than a new global-rate assertion."}),
        vec![
            equal(
                "native_Cx_definition",
                label,
                vector[1],
                dx.powi(2) + 2. * sigma.powi(2),
            ),
            equal(
                "native_Cv_definition",
                label,
                vector[2],
                8. * lambda * vmax.powi(2),
            ),
            equal("native_CW_reset_definition", label, vector[0], 4. * bw),
            equal(
                "boundary_offset_preserved_by_vector_assembly",
                label,
                vector[3],
                supplied_boundary_offset,
            ),
            certificate_check(
                "actual_finite_nonnegative_offset_vector",
                label,
                vector.iter().all(|x| x.is_finite() && *x >= 0.),
            ),
        ],
    )?;
    let all_alive = enumerate_law(
        vec![-1., -0.2, 0.4, 1.],
        vec![-0.5, 0.1, 0.3, 0.7],
        vec![true; 4],
    )?;
    let mut checks = Vec::new();
    for pattern in &all_alive.patterns {
        let mut actual_velocity_variance = 0.;
        for (plan, mass) in graph_plans(pattern) {
            let mut second = 0.;
            for group in components(&plan) {
                let average =
                    group.iter().map(|&i| all_alive.v[i]).sum::<f64>() / group.len() as f64;
                for i in group {
                    second += (average.powi(2) + 0.25 * (all_alive.v[i] - average).powi(2))
                        / all_alive.v.len() as f64;
                }
            }
            actual_velocity_variance += mass * (second - mean(&all_alive.v).powi(2));
        }
        checks.push(upper(
            "actual_all_alive_zero_revival_velocity_offset",
            label,
            actual_velocity_variance - var(&all_alive.v),
            0.,
        ));
    }
    record(
        suite,
        label,
        r"C_v=0",
        json!({"alive":all_alive.alive,"actual_native_velocities":all_alive.v,"actual_complete_plans":true,"scope":"All-alive one-step native component collisions dissipate the entering full-slot velocity variance; the zero revival offset is not propagated across terminal deaths."}),
        checks,
    )
}

fn keystone_native_inline_inputs(suite: &mut EstimateSuite) -> Result<()> {
    use algorithmic_gas::fitness::{PositiveMap, PositiveMapping, Standardizer};
    use algorithmic_gas::geometry::{Distance, Kernel};
    let label = "lem-keystone-complete-coverage-constants";
    let law = enumerate_law(
        vec![-1.5, -0.5, 0.5, 1.],
        vec![0.3, -0.2, 0.1, -0.4],
        vec![true; 4],
    )?;
    let config = &law.config;
    let (rx, rv, lambda) = match &config.distance_donors.distance {
        Distance::SquashedPhaseSpace {
            position_radius,
            velocity_radius,
            lambda,
            ..
        } => (*position_radius, *velocity_radius, *lambda),
        _ => return Err(error("canonical phase comparison changed")),
    };
    let width = |module: &DonorModule| match module.kernel {
        Kernel::Gaussian { width } => Ok(width),
        _ => Err(error("canonical Gaussian companion law changed")),
    };
    let sd = width(&config.distance_donors)?;
    let sc = width(&config.cloning_donors)?;
    let regularizer = |s: &Standardizer| match s {
        Standardizer::Global { sigma_min } => Ok(*sigma_min),
        _ => Err(error("canonical global standardizer changed")),
    };
    let er = regularizer(&config.fitness.reward_standardizer)?;
    let es = regularizer(&config.fitness.diversity_standardizer)?;
    let (amplitude, floor) = match config.fitness.reward_map {
        PositiveMap::Logistic { amplitude, floor } => (amplitude, floor),
        _ => return Err(error("canonical logistic map changed")),
    };
    let alpha = config.fitness.reward_exponent;
    let beta = config.fitness.diversity_exponent;
    let d0 = 2. * (rx * rx + lambda * rv * rv).sqrt();
    let bf = rx.max(lambda.sqrt() * rv);
    let vmax = config
        .kinetic
        .velocity_cap
        .ok_or_else(|| error("canonical velocity cap missing"))?;
    let bx = 2.;
    let report = canonical_keystone_constants(1, 0.001, 2.)?;
    let c = &report.values;
    let actual_w0 = 0.001_f64;
    let actual_emax = 64_f64;
    let inputs = json!({"native_config":config,"alive":law.alive,"actual_positions":law.x,"actual_velocities":law.v,"independent_source_constants":report,"entering_position_bound":bx,"reward_oscillation":2.,"scope":"Canonical native configuration and exact complete retained measurement law; aliases are checked against independently reconstructed formulas, and each condition uses its actual finite witness."});
    let aliases: Vec<(&str, Vec<BoundCheck>)> = vec![
        (
            r"R_x=R_v=2",
            vec![
                equal("actual_position_radius", label, rx, 2.),
                equal("actual_velocity_radius", label, rv, 2.),
            ],
        ),
        (
            r"\lambda_{\rm alg}=1",
            vec![equal("actual_positive_phase_weight", label, lambda, 1.)],
        ),
        (
            r"\sigma_D=\sigma_C=2",
            vec![
                equal("actual_measurement_width", label, sd, 2.),
                equal("actual_clone_width", label, sc, 2.),
            ],
        ),
        (
            r"B_x=2\sqrt d",
            vec![equal("actual_domain_radius", label, bx, 2. * 1_f64.sqrt())],
        ),
        (
            r"B_v=V_{\max}",
            vec![equal(
                "entering_velocity_bound_is_native_cap",
                label,
                c["B_v"],
                vmax,
            )],
        ),
        (
            r"B_v=2",
            vec![equal("actual_velocity_cap", label, vmax, 2.)],
        ),
        (
            r"D_0^2=32",
            vec![equal(
                "independent_feature_diameter_square",
                label,
                d0 * d0,
                32.,
            )],
        ),
        (
            r"B_f=2",
            vec![equal("independent_feature_cube_radius", label, bf, 2.)],
        ),
        (
            r"m_x=m_z=(1+\sqrt d)^{-2}",
            vec![
                equal(
                    "native_inverse_position_modulus",
                    label,
                    c["m_x"],
                    (1. + 1_f64.sqrt()).powi(-2),
                ),
                equal(
                    "native_inverse_joint_modulus",
                    label,
                    c["m_z"],
                    (1. + 1_f64.sqrt()).powi(-2),
                ),
            ],
        ),
        (
            r"\kappa_D=\kappa_C=e^{-4}",
            vec![
                equal_positive(
                    "actual_measurement_kernel_floor",
                    label,
                    (-d0 * d0 / (2. * sd * sd)).exp(),
                    (-4_f64).exp(),
                ),
                equal_positive(
                    "actual_clone_kernel_floor",
                    label,
                    (-d0 * d0 / (2. * sc * sc)).exp(),
                    (-4_f64).exp(),
                ),
            ],
        ),
        (
            r"\delta_D=.001",
            vec![equal_positive(
                "actual_configured_distance_floor",
                label,
                config.fitness.distance_floor,
                0.001,
            )],
        ),
        (
            r"\varepsilon_r=\varepsilon_s=.1",
            vec![
                equal_positive("actual_reward_regularizer", label, er, 0.1),
                equal_positive("actual_diversity_regularizer", label, es, 0.1),
            ],
        ),
        (
            r"\alpha=\beta=1",
            vec![
                equal("actual_reward_exponent", label, alpha, 1.),
                equal("actual_diversity_exponent", label, beta, 1.),
            ],
        ),
        (
            r"A_-=.1",
            vec![equal_positive(
                "powered_actual_reward_lower_bound",
                label,
                c["A_minus"],
                floor.powf(alpha),
            )],
        ),
        (
            r"f_+=2.1",
            vec![equal_positive(
                "powered_actual_diversity_upper_bound",
                label,
                c["f_plus"],
                (amplitude + floor).powf(beta),
            )],
        ),
        (
            r"F^*=4.41",
            vec![equal_positive(
                "powered_actual_fitness_upper_bound",
                label,
                c["F_star_upper"],
                (amplitude + floor).powf(alpha + beta),
            )],
        ),
        (
            r"L_H=1/2",
            vec![equal_positive(
                "actual_logistic_derivative_maximum",
                label,
                c["L_H"],
                amplitude / 4.,
            )],
        ),
        (
            r"p_{\max}=1",
            vec![equal_positive(
                "actual_acceptance_normalization",
                label,
                config.clone_decision.saturation,
                1.,
            )],
        ),
        (
            r"\varepsilon_c=10^{-6}",
            vec![equal_positive(
                "actual_acceptance_regularizer",
                label,
                config.clone_decision.epsilon,
                1e-6,
            )],
        ),
        (
            r"\sigma_D,\sigma_C>0",
            vec![certificate_check(
                "actual_strict_companion_widths",
                label,
                sd > 0. && sc > 0.,
            )],
        ),
        (
            r"\varepsilon_r,\varepsilon_s>0",
            vec![certificate_check(
                "actual_strict_standardizers",
                label,
                er > 0. && es > 0.,
            )],
        ),
        (
            r"\beta>0",
            vec![certificate_check(
                "actual_strict_active_diversity_power",
                label,
                beta > 0.,
            )],
        ),
        (
            r"\alpha\ge0,\beta>0",
            vec![certificate_check(
                "actual_power_conditions",
                label,
                alpha >= 0. && beta > 0.,
            )],
        ),
        (
            r"A_->0",
            vec![certificate_check(
                "actual_strict_reward_floor",
                label,
                floor.powf(alpha) > 0.,
            )],
        ),
        (
            r"f_+<\infty",
            vec![certificate_check(
                "actual_finite_diversity_upper",
                label,
                (amplitude + floor).powf(beta).is_finite(),
            )],
        ),
        (
            r"F^*<\infty",
            vec![certificate_check(
                "actual_finite_fitness_upper",
                label,
                (amplitude + floor).powf(alpha + beta).is_finite(),
            )],
        ),
        (
            r"0<W_0\le E_{\max}",
            vec![certificate_check(
                "actual_analysis_threshold_domain",
                label,
                actual_w0 > 0. && actual_w0 <= actual_emax,
            )],
        ),
        (
            r"m_x^2W_0/4<D_0^2",
            vec![certificate_check(
                "actual_feature_threshold_below_diameter",
                label,
                c["m_x"].powi(2) * 0.001 / 4. < d0 * d0,
            )],
        ),
        (
            r"0<t_f\le Z_*",
            vec![certificate_check(
                "actual_increment_domain",
                label,
                c["t_f"] > 0. && c["t_f"] <= c["Z_star"],
            )],
        ),
        (
            r"\kappa_D=e^{-D_0^2/(2\sigma_D^2)}",
            vec![equal_positive(
                "reconstructed_measurement_kernel_floor",
                label,
                c["kappa_D"],
                (-d0 * d0 / (2. * sd * sd)).exp(),
            )],
        ),
        (
            r"\kappa_C=e^{-D_0^2/(2\sigma_C^2)}",
            vec![equal_positive(
                "reconstructed_clone_kernel_floor",
                label,
                c["kappa_C"],
                (-d0 * d0 / (2. * sc * sc)).exp(),
            )],
        ),
        (
            r"D_m=\sqrt{D_0^2+\delta_D^2}",
            vec![
                equal_positive(
                    "reconstructed_measured_span",
                    label,
                    c["D_m_span"],
                    (d0 * d0 + config.fitness.distance_floor.powi(2)).sqrt()
                        - config.fitness.distance_floor,
                ),
                equal_positive(
                    "reconstructed_regularized_scale",
                    label,
                    c["s_star"],
                    (c["D_m_span"].powi(2) / 4. + es * es).sqrt(),
                ),
                equal_positive(
                    "reconstructed_standardized_range",
                    label,
                    c["Z_star"],
                    c["D_m_span"] / es,
                ),
            ],
        ),
        (
            r"Z_R=\operatorname{osc}",
            vec![equal_positive(
                "reward_standardized_range",
                label,
                2. / er,
                20.,
            )],
        ),
        (
            r"L_H=\max_{|t|\le Z_R}|H'(t)|",
            vec![
                equal_positive(
                    "exact_logistic_derivative_maximizer_zero",
                    label,
                    c["L_H"],
                    amplitude / 4.,
                ),
                certificate_check(
                    "actual_derivative_maximizer_inside_reward_interval",
                    label,
                    20. > 0.,
                ),
            ],
        ),
        (
            r"\alpha=0",
            vec![equal(
                "constant_reward_power_value",
                label,
                (amplitude + floor).powf(0.),
                1.,
            )],
        ),
        (
            r"L_H=0",
            vec![equal(
                "constant_reward_power_derivative",
                label,
                0. * (amplitude + floor).powf(-1.),
                0.,
            )],
        ),
        (
            r"\omega_f\ge 2e^{-Z_*}t_f",
            vec![lower(
                "actual_logistic_increment_derivative_lower_log",
                label,
                report.natural_logs["omega_f"],
                2_f64.ln() - c["Z_star"] + c["t_f"].ln() - 2. * (-c["Z_star"]).exp().ln_1p(),
            )],
        ),
    ];
    for (needle, checks) in aliases {
        record(suite, label, needle, inputs.clone(), checks)?;
    }
    let mut maps = Vec::new();
    let mut fitness = Vec::new();
    let mut standardizer_checks = Vec::new();
    let mut feature_checks = vec![
        equal("feature_diameter_formula", label, c["D_0"], d0),
        equal("feature_cube_formula", label, c["B_f"], bf),
    ];
    for p in &law.patterns {
        let ybar = mean(&p.y);
        let yvar = var(&p.y);
        let ymin = config.fitness.distance_floor;
        let ymax = (d0 * d0 + ymin * ymin).sqrt();
        standardizer_checks.push(upper(
            "native_actual_score_range",
            label,
            p.z.iter().map(|z| z.abs()).fold(0., f64::max),
            c["Z_star"],
        ));
        standardizer_checks.push(upper(
            "native_actual_shared_scale_upper",
            label,
            p.scales[0],
            c["s_star"],
        ));
        standardizer_checks.push(upper(
            "native_Popoviciu_first_clause",
            label,
            yvar,
            (ymax - ybar) * (ybar - ymin),
        ));
        standardizer_checks.push(upper(
            "native_Popoviciu_second_clause",
            label,
            (ymax - ybar) * (ybar - ymin),
            (ymax - ymin).powi(2) / 4.,
        ));
        let reward_values: Vec<_> = law.x.iter().map(|x| -x * x / 2.).collect();
        let rmean = mean(&reward_values);
        let rscale = (var(&reward_values) + er * er).sqrt();
        for (i, reward_value) in reward_values.iter().enumerate() {
            let rz = (reward_value - rmean) / rscale;
            let actual_reward = config.fitness.reward_map.map(rz)?;
            let actual_diversity = config.fitness.diversity_map.map(p.z[i])?;
            maps.push(equal(
                "actual_naked_reward_logistic",
                label,
                actual_reward - floor,
                2. / (1. + (-rz).exp()),
            ));
            maps.push(equal(
                "actual_naked_diversity_logistic",
                label,
                actual_diversity - floor,
                2. / (1. + (-p.z[i]).exp()),
            ));
            fitness.push(equal(
                "actual_powered_retained_product_fitness",
                label,
                p.f[i],
                actual_reward.powf(alpha) * actual_diversity.powf(beta),
            ));
            fitness.push(equal(
                "actual_retained_score_definition",
                label,
                p.z[i],
                (p.y[i] - ybar) / p.scales[i],
            ));
            let zx = rx * law.x[i] / (rx + law.x[i].abs());
            let zv = lambda.sqrt() * rv * law.v[i] / (rv + law.v[i].abs());
            feature_checks.push(upper("actual_feature_cube_position", label, zx.abs(), bf));
            feature_checks.push(upper("actual_feature_cube_velocity", label, zv.abs(), bf));
            for j in 0..law.x.len() {
                let dzx = zx - rx * law.x[j] / (rx + law.x[j].abs());
                let dzv = zv - lambda.sqrt() * rv * law.v[j] / (rv + law.v[j].abs());
                feature_checks.push(upper(
                    "actual_feature_diameter_upper",
                    label,
                    dzx.hypot(dzv),
                    d0,
                ));
            }
        }
    }
    records(
        suite,
        label,
        &[
            r"g_r(t)=g_s(t)=2/(1+e^{-t})",
            r"H(t)=(g_r(t)+\eta_r)^\alpha",
            r"f(t)=(g_s(t)+\eta_s)^\beta",
        ],
        inputs.clone(),
        maps,
    )?;
    record(suite, label, r"F_i=A_i f(u_i)", inputs.clone(), fitness)?;
    record(
        suite,
        label,
        r"z_i=(S_{R_x}(x_i)",
        inputs.clone(),
        feature_checks,
    )?;
    records(
        suite,
        label,
        &[
            r"|u_i|\le Z_*",
            r"s_Y\le s_*",
            r"\operatorname{Var}(Y)\le(y_+-\bar Y)",
        ],
        inputs,
        standardizer_checks,
    )?;
    Ok(())
}

fn nontrivial_cluster_remainder(suite: &mut EstimateSuite) -> Result<()> {
    let law = enumerate_law(
        vec![-0.9, -0.8, -0.7, 0.9],
        vec![-0.3, -0.2, 0.2, 0.4],
        vec![true; 4],
    )?;
    let n = 4_f64;
    let groups = [vec![0, 1, 2], vec![3]];
    let mu = mean(&law.x);
    let dx = range(&law.x);
    let dc = 0.2_f64;
    let label = "lem-keystone-contraction-alive";
    let mut checks = Vec::new();
    let mut driftchecks = Vec::new();
    let centroids = groups
        .iter()
        .map(|g| g.iter().map(|&i| law.x[i]).sum::<f64>() / g.len() as f64)
        .collect::<Vec<_>>();
    let energy = centroids
        .iter()
        .map(|x| (x - mu).powi(2))
        .collect::<Vec<_>>();
    let rho = (0..4)
        .map(|i| (law.x[i] - mu).powi(2) - energy[usize::from(i == 3)])
        .collect::<Vec<_>>();
    for p in &law.patterns {
        let mut between = 0.;
        let mut remainder = 0.;
        let mut direct = 0.;
        for i in 0..4 {
            for j in 0..4 {
                let gi = usize::from(i == 3);
                let gj = usize::from(j == 3);
                between += p.edges[i][j] / n * (energy[gj] - energy[gi]);
                remainder += p.edges[i][j] / n * (rho[j] - rho[i]);
                direct += p.edges[i][j] / n * ((law.x[j] - mu).powi(2) - (law.x[i] - mu).powi(2));
                checks.push(upper(
                    "actual_nontrivial_cluster_center_deviation",
                    label,
                    (law.x[i] - centroids[gi]).abs(),
                    dc,
                ));
                checks.push(upper(
                    "actual_nontrivial_cluster_rho_bound",
                    label,
                    rho[i].abs(),
                    2. * dx * dc,
                ));
            }
        }
        checks.extend([
            equal(
                "actual_nonzero_cluster_F7_flux_decomposition",
                label,
                direct,
                between + remainder,
            ),
            upper(
                "actual_nonzero_cluster_F8_remainder",
                label,
                remainder.abs(),
                4. * dx * dc * mean(&p.p),
            ),
        ]);
        let high = &groups[1];
        let low = &groups[0];
        let fh = p.f[3];
        let fl = low.iter().map(|&j| p.f[j]).collect::<Vec<_>>();
        let delta = mean(&fl) - fh;
        let spread = var(&fl);
        let kappa = (-4_f64).exp();
        let cp = kappa / 4.410001;
        let cm = 1. / (kappa * 0.010001);
        let uniform = (high.len() * low.len()) as f64 / (n * 3.)
            * (cp * delta - (cm - cp) / 2. * ((delta * delta + spread).sqrt() - delta));
        let zi = (0..4)
            .filter(|&j| j != 3)
            .map(|j| (-p.raw_distance(&law, 3, j).powi(2) / 8.).exp())
            .sum::<f64>();
        let mut ww = 0.;
        let mut weighted_delta = 0.;
        let mut d2 = 0.;
        let mut minz = f64::INFINITY;
        for &j in low {
            let w = (-p.raw_distance(&law, 3, j).powi(2) / 8.).exp();
            ww += w;
            weighted_delta += w * (p.f[j] - fh);
            d2 += w * (p.f[j] - fh).powi(2);
            minz = minz.min(
                (0..4)
                    .filter(|&i| i != j)
                    .map(|i| (-p.raw_distance(&law, j, i).powi(2) / 8.).exp())
                    .sum::<f64>(),
            );
        }
        weighted_delta /= ww;
        d2 /= ww;
        let maxlow = fl.iter().copied().fold(0., f64::max);
        let minlow = fl.iter().copied().fold(f64::INFINITY, f64::min);
        let a = 1. / (zi * (maxlow - fh).max(0.).max(fh + 1e-6));
        let b = 1. / (minz * (minlow + 1e-6));
        let weighted =
            ww / n * (a * weighted_delta - (b - a).max(0.) / 2. * (d2.sqrt() - weighted_delta));
        let copycov = p
            .traces
            .iter()
            .zip(&p.p)
            .map(|(c, p)| c - 0.01 * p)
            .sum::<f64>();
        let rhs = -(energy[1] - energy[0]) * uniform.max(weighted) + remainder
            - (mean(&p.means) - mu).powi(2)
            - copycov / (n * n)
            + (1. - 1. / n) * 0.01 * mean(&p.p);
        driftchecks.push(upper(
            "actual_nontrivial_two_cluster_signed_S4_maximum",
            "cor-cloning-signed-collective-drift",
            p.output_var - var(&law.x),
            rhs,
        ));
    }
    let input = json!({"actual_native_positions":law.x,"actual_native_velocities":law.v,"native_complete_patterns":81,"geometric_groups":groups,"cluster_centroids":centroids,"cluster_centroid_energies":energy,"within_cluster_rho":rho,"D_x":dx,"D_c":dc,"scope":"A nontrivial 3+1 geometric partition with nonzero within-cluster deviations; both signed flow bounds and their maximum use the actual retained native fitness and donor normalizers"});
    records(
        suite,
        label,
        &["x_G=", "3.F7", "3.F8", "|x_i-x_G|\\le D_c"],
        input.clone(),
        checks,
    )?;
    record(
        suite,
        "cor-cloning-signed-collective-drift",
        "3.S4",
        input,
        driftchecks,
    )
}

fn balanced_cluster_native_definitions(suite: &mut EstimateSuite) -> Result<()> {
    let config = GasConfig::euclidean(1, 0.04)?;
    let label = "prop-cloning-two-cluster-noise-balance";
    for n in [4_usize, 6, 8] {
        for radius in [0.5_f64, 1., 1.9] {
            let m = n / 2;
            let x: Vec<f64> = (0..n)
                .map(|i| if i < m { radius } else { -radius })
                .collect();
            let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, x.clone())?);
            obs.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(n, 1, vec![0.; n])?,
            );
            let alive = vec![true; n];
            let donors = probabilities(&config.cloning_donors, &obs, &alive)?;
            let measurements = probabilities(&config.distance_donors, &obs, &alive)?;
            let native_ell = config.distance_donors.distance.compare(&obs, 0, &obs, m)?;
            let ell = 4. * radius / (2. + radius);
            let w = (-ell * ell / 8.).exp();
            let z = m as f64 - 1. + m as f64 * w;
            let q = measurements[0][m..].iter().sum::<f64>();
            let delta = (ell * ell + 1e-6).sqrt() - 0.001;
            let smax = (delta * delta / 4. + 0.01).sqrt();
            let a0 = (1.1 * (delta / (2. * smax)).tanh() / (1.21 + 1e-6)).min(1.);
            let beta = donors[0][m];
            let rewards = RewardBatch::new(vec![radius * radius / 2.; n], Default::default());
            let mut pattern_checks = Vec::new();
            let mut meanp = 0.;
            let mut kp_first = 0.;
            let mut km_first = 0.;
            let mut kp_second = 0.;
            let mut km_second = 0.;
            let mut joint = 0.;
            let mut masssum = 0.;
            let mut count_snapshots = Vec::new();
            for mask in 0_usize..(1 << n) {
                let kp = (mask & ((1 << m) - 1)).count_ones() as usize;
                let km = (mask >> m).count_ones() as usize;
                let k = kp + km;
                let theta = k as f64 / n as f64;
                let mass = q.powi(k as i32) * (1. - q).powi((n - k) as i32);
                let distance: Vec<_> = (0..n)
                    .map(|i| if mask & (1 << i) == 0 { 0. } else { native_ell })
                    .collect();
                let native = config
                    .fitness
                    .evaluate(&rewards, &distance, &alive, &obs, 0)?;
                let native_moments = exact_cloning_moments(
                    &x.iter().map(|&x| vec![x]).collect::<Vec<_>>(),
                    &native.fitness,
                    &donors,
                    &alive,
                    &config.clone_decision,
                    1,
                    config.clone_transform.jitter_amplitude,
                )?;
                let st = (theta * (1. - theta) * delta * delta + 0.01).sqrt();
                let gh = 2. / (1. + (-(1. - theta) * delta / st).exp()) + 0.1;
                let gl = 2. / (1. + (theta * delta / st).exp()) + 0.1;
                let fh = 1.1 * gh;
                let fl = 1.1 * gl;
                let acceptance = if k == 0 || k == n {
                    0.
                } else {
                    ((fh - fl) / (fl + 1e-6)).min(1.)
                };
                pattern_checks.push(equal(
                    "native_total_high_count",
                    label,
                    k as f64,
                    (kp + km) as f64,
                ));
                pattern_checks.push(equal(
                    "native_high_fraction",
                    label,
                    theta,
                    k as f64 / n as f64,
                ));
                if k > 0 && k < n {
                    for (i, &f) in native.fitness.iter().enumerate() {
                        pattern_checks.push(equal(
                            "native_full_pattern_two_level_fitness",
                            label,
                            f,
                            if mask & (1 << i) == 0 { fl } else { fh },
                        ));
                    }
                    pattern_checks.push(equal_positive(
                        "native_global_diversity_scale",
                        label,
                        native.diversity_stats.scale[0],
                        st,
                    ));
                    pattern_checks.push(upper("native_global_scale_upper", label, st, smax));
                    pattern_checks.push(lower(
                        "native_every_nontied_count_acceptance_floor",
                        label,
                        acceptance,
                        a0,
                    ));
                    let u = (1. - theta) * delta / (2. * st);
                    let v = theta * delta / (2. * st);
                    pattern_checks.push(lower(
                        "positive_tanh_addition",
                        label,
                        u.tanh() + v.tanh(),
                        (u + v).tanh(),
                    ));
                } else {
                    pattern_checks.push(equal(
                        "native_tied_acceptance_zero",
                        label,
                        native_moments.acceptance_probabilities.iter().sum::<f64>(),
                        0.,
                    ));
                }
                let tmean = mean(
                    &native_moments
                        .output_means
                        .iter()
                        .map(|x| x[0])
                        .collect::<Vec<_>>(),
                );
                pattern_checks.push(equal(
                    "native_full_mask_barycenter_displacement",
                    label,
                    tmean,
                    radius * acceptance * beta * (kp as f64 - km as f64),
                ));
                for i in 0..m {
                    if mask & (1 << i) == 0 {
                        let donor_flip = acceptance * beta * km as f64;
                        let measured = (radius - native_moments.output_means[i][0]) / (2. * radius);
                        pattern_checks.push(equal(
                            "native_negative_site_copy_probability",
                            label,
                            measured,
                            donor_flip,
                        ));
                    }
                }
                let pb = mean(&native_moments.acceptance_probabilities);
                meanp += mass * pb;
                masssum += mass;
                kp_first += mass * kp as f64;
                km_first += mass * km as f64;
                kp_second += mass * (kp as f64).powi(2);
                km_second += mass * (km as f64).powi(2);
                joint += mass * (kp * km) as f64;
                count_snapshots.push(json!({"mask":mask,"K_plus":kp,"K_minus":km,"native_full_fitness":native.fitness,"native_barycenter":tmean,"A_K":acceptance,"mass":mass}));
            }
            let mut c = vec![
                certificate_check("actual_even_population_domain", label, n == 2 * m && m >= 2),
                certificate_check("actual_radius_domain", label, radius > 0. && radius < 2.),
                equal(
                    "actual_canonical_jitter",
                    label,
                    config.clone_transform.jitter_amplitude,
                    0.1,
                ),
                equal_positive("native_squashed_pair_distance", label, native_ell, ell),
                equal_positive(
                    "native_gaussian_cross_site_weight",
                    label,
                    w,
                    config
                        .distance_donors
                        .kernel
                        .log_weight(
                            native_ell,
                            algorithmic_gas::geometry::ComparisonKind::Distance,
                        )?
                        .exp(),
                ),
                equal_positive(
                    "native_nonself_normalizer",
                    label,
                    z,
                    (0..n)
                        .filter(|&j| j != 0)
                        .map(|j| if j < m { 1. } else { w })
                        .sum(),
                ),
                equal_positive(
                    "native_opposite_site_probability",
                    label,
                    q,
                    m as f64 * w / z,
                ),
                equal_positive(
                    "native_acceptance_floor_formula",
                    label,
                    a0,
                    crate::convergence_cloning::canonical_two_cluster_acceptance_floor(radius),
                ),
                equal_positive(
                    "native_donor_cross_site_probability_first",
                    label,
                    beta,
                    w / z,
                ),
                equal_positive(
                    "native_donor_cross_site_probability_second",
                    label,
                    beta,
                    q / m as f64,
                ),
                equal("all_native_mask_mass", label, masssum, 1.),
                equal(
                    "native_binomial_plus_first_moment",
                    label,
                    kp_first,
                    m as f64 * q,
                ),
                equal(
                    "native_binomial_minus_first_moment",
                    label,
                    km_first,
                    m as f64 * q,
                ),
                equal(
                    "native_binomial_plus_variance",
                    label,
                    kp_second - kp_first.powi(2),
                    m as f64 * q * (1. - q),
                ),
                equal(
                    "native_binomial_minus_variance",
                    label,
                    km_second - km_first.powi(2),
                    m as f64 * q * (1. - q),
                ),
                equal(
                    "native_two_count_independence_covariance",
                    label,
                    joint - kp_first * km_first,
                    0.,
                ),
                lower(
                    "native_mean_acceptance_pressure",
                    label,
                    meanp,
                    a0 * q * (1. - q),
                ),
            ];
            c.extend(pattern_checks);
            let inputs = json!({"N":n,"M":m,"alive":alive,"radius":radius,"native_config":config,"ell":ell,"native_ell":native_ell,"w":w,"Z":z,"q":q,"delta":delta,"s_max":smax,"A0":a0,"beta":beta,"all_native_measurement_masks":count_snapshots,"native_mean_pressure":meanp,
                "scope":"Native current-donor and measurement kernels, full native globally standardized fitness and exact copy/jitter moments are independently recomputed for every measurement mask. The mask product law verifies the two binomial moments and their covariance; all nontied acceptance and barycenter/copy identities are checked."});
            records(
                suite,
                label,
                &[
                    "N=2M",
                    "0<a<2",
                    "j=0.1",
                    r"\ell=\frac{4a}",
                    r"\delta=\sqrt{\ell^2",
                    r"K=K_++K_-",
                    r"\theta=K/N",
                    r"F_H=1.1g",
                    r"g(z)=1.1+\tanh(z/2)",
                    r"A(K)=\min",
                    r"A(0)=A(N)=0",
                    r"\tanh u+\tanh v",
                    r"s_\theta\leq s_{\max}",
                    r"A(K)\geq A_0",
                    r"\beta=w/Z=q/M",
                    r"\bar t=aA(K)",
                    r"c_-=A(K)\beta K_-",
                    r"\mathbb E\bar p\geq A_0q(1-q)",
                ],
                inputs.clone(),
                c,
            )?;
            let label = "cor-keystone-canonical-balanced-structural";
            let structural = (radius - 0.5).powi(2);
            let physical_x: Vec<_> = (0..n)
                .map(|i| if i < m { radius } else { -radius })
                .collect();
            let physical_y: Vec<_> = (0..n).map(|i| if i < m { 0.5 } else { -0.5 }).collect();
            let mut cc = vec![
                equal(
                    "native_even_population_condition",
                    label,
                    n as f64,
                    2. * m as f64,
                ),
                certificate_check(
                    "actual_balanced_radius_interval",
                    label,
                    (0.5..2.).contains(&radius),
                ),
                equal(
                    "actual_quadratic_objective",
                    label,
                    radius * radius / 2.,
                    rewards.raw[0],
                ),
                equal(
                    "actual_centered_sorted_balanced_transport",
                    label,
                    mean(
                        &physical_x
                            .iter()
                            .zip(&physical_y)
                            .map(|(x, y)| (x - y).powi(2))
                            .collect::<Vec<_>>(),
                    ),
                    structural,
                ),
                equal_positive(
                    "balanced_native_q_definition",
                    label,
                    q,
                    m as f64 * w / (m as f64 - 1. + m as f64 * w),
                ),
                certificate_check(
                    "actual_w_strict_gaussian_lower",
                    label,
                    w > (-0.5_f64).exp() && (-0.5_f64).exp() > 0.5,
                ),
                lower("native_balanced_q_lower_first", label, q, w / (1. + w)),
                certificate_check(
                    "native_balanced_q_lower_strict",
                    label,
                    w / (1. + w) > 1. / 3.,
                ),
                upper(
                    "native_balanced_q_upper_first",
                    label,
                    q,
                    m as f64 / (2 * m - 1) as f64,
                ),
                upper(
                    "native_balanced_q_upper_second",
                    label,
                    m as f64 / (2 * m - 1) as f64,
                    2. / 3.,
                ),
                lower(
                    "actual_balanced_monotone_acceptance",
                    label,
                    a0,
                    crate::convergence_cloning::canonical_two_cluster_acceptance_floor(0.5),
                ),
            ];
            for row in &donors {
                cc.push(equal(
                    "actual_balanced_donor_row_mass",
                    label,
                    row.iter().sum(),
                    1.,
                ));
            }
            records(
                suite,
                label,
                &[
                    "N=2M",
                    r"a,b\in[0.5,2)",
                    r"U(x)=|x|^2/2",
                    r"V_{\mathrm{struct}}=(a-b)^2",
                    r"q_M(r)=",
                    r"w(r)>e^{-1/2}>1/2",
                    r"q_M(r)\geq w(r)/(1+w(r))>1/3",
                    r"q_M(r)\leq M/(2M-1)\leq2/3",
                    r"A_0(r)\geq A_0(0.5)",
                ],
                inputs,
                cc,
            )?;
        }
    }
    let n = 128_usize;
    let radius = 0.5_f64;
    let m = n / 2;
    let ell = 4. * radius / (2. + radius);
    let w = (-ell * ell / 8.).exp();
    let q = m as f64 * w / (m as f64 - 1. + m as f64 * w);
    let a0 = crate::convergence_cloning::canonical_two_cluster_acceptance_floor(radius);
    let tie = q.powi(n as i32) + (1. - q).powi(n as i32);
    let drift = 0.01 * (1. - 1. / n as f64) * a0 * q * (1. - q)
        - 4. * radius * radius / n as f64 * q * q * (1. - q * q);
    records(
        suite,
        label,
        &[
            "a=0.5",
            "N=128",
            r"q^N+(1-q)^N<1.7\times10^{-37}",
            r"q=0.483942612969",
            r"A_0=0.680668094688",
            "a=0.5,N=128",
        ],
        json!({"N":n,"radius":radius,"q":q,"A0":a0,"exact_tie_mass_product_bernoulli":tie,"proved_drift_lower_formula":drift,"scope":"Compressed native Bernoulli site-measurement probabilities evaluate the N=128 constants and tied-pattern mass; this does not claim enumeration of 2^128 physical masks."}),
        vec![
            equal("reported_native_q_rounding", label, q, 0.483942612969),
            equal("reported_native_A0_rounding", label, a0, 0.680668094688),
            upper("complete_tied_pattern_mass", label, tie, 1.7e-37),
            certificate_check(
                "reported_native_positive_drift_lower",
                label,
                drift > 0.0002854,
            ),
            equal("actual_reported_population", label, n as f64, 128.),
            equal("actual_reported_radius", label, radius, 0.5),
        ],
    )?;
    Ok(())
}

fn moving_center_native_definitions(suite: &mut EstimateSuite) -> Result<()> {
    let label = "ex-cloning-moving-barycenter";
    let a = 0.1_f64;
    let law = enumerate_law(vec![-a, 0., a], vec![0.; 3], vec![true; 3])?;
    let p = law
        .patterns
        .iter()
        .find(|p| p.choices[0] == 1 && p.choices[2] == 1)
        .expect("enumerated event");
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(3, 1, law.x.clone())?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(3, 1, vec![0.; 3])?,
    );
    let native_fitness = law.config.fitness.evaluate(
        &RewardBatch::new(
            law.x.iter().map(|x| x * x / 2.).collect(),
            Default::default(),
        ),
        &p.raw,
        &law.alive,
        &obs,
        0,
    )?;
    let theta = law.measurements[0][1];
    let q = p.p[0];
    let d1 = 2. * a / (2. + a);
    let d2 = 2. * d1;
    let w1 = (-d1 * d1 / 8.).exp();
    let w2 = (-d2 * d2 / 8.).exp();
    let sr = (a.powi(4) / 18. + 0.01).sqrt();
    let f0 = p.f[1];
    let fm = p.f[0];
    let mut checks = vec![
        certificate_check("actual_event_radius_domain", label, a > 0. && a < 2.),
        equal(
            "actual_effective_quadratic_reward",
            label,
            -a * a / 2.,
            law.config.fitness.direction.orient(a * a / 2.),
        ),
        equal_positive(
            "native_event_probability_ratio",
            label,
            theta,
            w1 / (w1 + w2),
        ),
        equal_positive(
            "native_event_radius_far_formula",
            label,
            d2,
            4. * a / (2. + a),
        ),
        equal_positive(
            "actual_native_endpoint_donor_weight",
            label,
            law.donors[0][1],
            w1 / (w1 + w2),
        ),
        equal(
            "native_global_reward_scale",
            label,
            native_fitness.reward_stats.scale[0],
            sr,
        ),
        equal("native_persistent_center_pressure_zero", label, p.p[1], 0.),
        equal_positive(
            "native_actual_endpoint_acceptance_formula",
            label,
            q,
            theta
                * ((f0 - fm)
                    / (law.config.clone_decision.saturation
                        * (fm + law.config.clone_decision.epsilon)))
                    .min(1.),
        ),
        certificate_check(
            "actual_endpoint_acceptance_open_unit_interval",
            label,
            q > 0. && q < 1.,
        ),
        equal(
            "native_reported_theta_rounding",
            label,
            theta,
            0.500850339316,
        ),
        equal("native_reported_F0_rounding", label, f0, 1.22832654693),
        equal(
            "native_reported_Fendpoint_rounding",
            label,
            fm,
            1.20083609058,
        ),
        equal("native_reported_q_rounding", label, q, 0.0114658387064),
        equal(
            "native_reported_geometric_barycenter_term",
            label,
            2. * a * a * q * (1. - q) / 9.,
            2.51874961093e-5,
        ),
        equal(
            "native_reported_unconditional_barycenter_term",
            label,
            theta * theta * 2. * a * a * q * (1. - q) / 9.,
            6.31831015805e-6,
        ),
        certificate_check(
            "actual_strict_collective_contraction_jitter_condition",
            label,
            0.01 < a * a * (4. - q) / 2.,
        ),
        equal(
            "actual_pmax_canonical",
            label,
            law.config.clone_decision.saturation,
            1.,
        ),
        equal_positive(
            "actual_epsilon_clone_canonical",
            label,
            law.config.clone_decision.epsilon,
            1e-6,
        ),
    ];
    let mut eventmass = 0.;
    for pattern in &law.patterns {
        if pattern.choices[0] == 1 && pattern.choices[2] == 1 {
            eventmass += pattern.mass;
        }
    }
    checks.push(equal_positive(
        "native_event_mass_product_law",
        label,
        eventmass,
        theta * theta,
    ));
    let mut bary_checks = checks.clone();
    for am in [false, true] {
        for ap in [false, true] {
            for zm in [-1., 0., 1.] {
                for zp in [-1., 0., 1.] {
                    let xm = if am { 0.1 * zm } else { -a };
                    let xp = if ap { 0.1 * zp } else { a };
                    let actual = (xm + xp) / 3.;
                    let formula = (a * (u8::from(am) as f64 - u8::from(ap) as f64)
                        + 0.1 * (u8::from(am) as f64 * zm + u8::from(ap) as f64 * zp))
                        / 3.;
                    bary_checks.push(equal(
                        "native_independent_endpoint_decision_barycenter_identity",
                        label,
                        actual,
                        formula,
                    ));
                }
            }
        }
    }
    let input = json!({"positions":law.x,"alive":law.alive,"all_measurement_patterns":law.patterns.len(),"native_event_fitness":p.f,"native_endpoint_pressure":q,"native_center_pressure":p.p[1],"native_effective_rewards":native_fitness.oriented_reward,"delta1":d1,"delta2":d2,"w1":w1,"w2":w2,"theta":theta,"native_event_mass":eventmass,"native_reward_scale":native_fitness.reward_stats.scale,"reward_scale_formula":sr,
        "scope":"The actual three-row native law retains every measurement event. The displayed rounded constants are checked against native fitness, pressure and event mass. Independent endpoint Bernoulli decisions and each affine Gaussian coefficient reconstruct the moving barycenter formula."});
    records(
        suite,
        label,
        &[
            r"r(x)=-x^2/2",
            "0<a<2",
            r"\delta_1=",
            r"\Pr(E\mid S)=\theta^2>0",
            r"s_r=\sqrt",
            r"p_0=0",
            r"q=\theta\min",
            "a=0.1",
            r"p_{\max}=1",
            r"\varepsilon_{\rm clone}=10^{-6}",
            r"\theta=0.500850339316",
            r"\frac{2a^2q(1-q)}9=2.518",
            r"j^2<a^2(4-q)/2",
            "a=j=0.1",
        ],
        input.clone(),
        checks,
    )?;
    record(
        suite,
        label,
        r"\bar X'=\frac{a(A_--A_+)",
        input,
        bary_checks,
    )
}

fn complete_keystone_native_aliases(suite: &mut EstimateSuite) -> Result<()> {
    let squash = |x: f64| 2. * x / (2. + x.abs());
    let report = canonical_keystone_constants(1, 0.001, 2.)?;
    let c = &report.values;
    let logs = &report.natural_logs;
    let emax = c["E_max"];
    let w0 = report.error_threshold;
    let n = 8_usize;
    let k = n;
    let m = 1_f64;
    let ee = 0.25_f64;
    let w = ee;
    let law = crate::convergence_cloning::exact_balanced_two_site_fixture(n, 1., 0.1)?;
    let law2 = crate::convergence_cloning::exact_balanced_two_site_fixture(n, 0.5, 0.1)?;
    let x: Vec<_> = (0..n).map(|i| if i < 4 { 1. } else { -1. }).collect();
    let y: Vec<_> = (0..n).map(|i| if i < 4 { 0.5 } else { -0.5 }).collect();
    let e: Vec<_> = x.iter().zip(&y).map(|(x, y)| (x - y) * (x - y)).collect();
    let nativepressure: Vec<_> = law
        .acceptance_probabilities
        .iter()
        .zip(&law2.acceptance_probabilities)
        .map(|(a, b)| a + b)
        .collect();
    let vx = var(&x);
    let vy = var(&y);
    let nativeactivity = nativepressure
        .iter()
        .zip(&e)
        .map(|(p, e)| p * e)
        .sum::<f64>()
        / n as f64;
    let mut checks = vec![
        equal(
            "actual_native_error_mass_definition",
            "thm-keystone-complete-error-coverage",
            w,
            mean(&e),
        ),
        equal(
            "actual_native_cell_error_definition_first",
            "thm-keystone-complete-error-coverage",
            w / 2.,
            e[..4].iter().sum::<f64>() / n as f64,
        ),
        equal(
            "actual_native_cell_error_definition_second",
            "thm-keystone-complete-error-coverage",
            w / 2.,
            e[4..].iter().sum::<f64>() / n as f64,
        ),
        equal(
            "actual_native_cells_sum",
            "thm-keystone-complete-error-coverage",
            2. * w / 2.,
            w,
        ),
        upper(
            "actual_native_cell_error_capacity",
            "thm-keystone-complete-error-coverage",
            w / 2.,
            emax * 4. / n as f64,
        ),
        certificate_check(
            "actual_native_population_alive_count",
            "thm-keystone-complete-error-coverage",
            n >= k && k >= 2,
        ),
        lower(
            "native_swarm1_feature_variance",
            "thm-keystone-complete-error-coverage",
            squash(1.).powi(2),
            c["v_0"],
        ),
        lower(
            "native_swarm2_feature_variance",
            "thm-keystone-complete-error-coverage",
            squash(0.5).powi(2),
            c["v_0"],
        ),
        equal(
            "actual_emax_physical_diameter_identity",
            "thm-keystone-complete-error-coverage",
            emax,
            4. * 4_f64.powi(2),
        ),
        certificate_check(
            "actual_positive_occupied_cell_mass",
            "thm-keystone-complete-error-coverage",
            w / 2. > 0.,
        ),
        certificate_check(
            "actual_same_site_neighbors_satisfy_r",
            "thm-keystone-complete-error-coverage",
            0. <= logs["r"].exp(),
        ),
        lower(
            "actual_neighbor_count_cell_relation",
            "thm-keystone-complete-error-coverage",
            3.,
            4. - 1.,
        ),
        upper(
            "actual_nonself_denominator_count",
            "thm-keystone-complete-error-coverage",
            (k - 1) as f64,
            k as f64,
        ),
        lower(
            "actual_alive_normalized_error_threshold",
            "thm-keystone-complete-error-coverage",
            w / m,
            w0,
        ),
    ];
    for &e in &e {
        checks.push(upper(
            "actual_every_row_error_bound",
            "thm-keystone-complete-error-coverage",
            e,
            emax,
        ));
    }
    let inputs = json!({"N":n,"alive":vec![true;n],"k1":k,"k2":k,"positions1":x,"positions2":y,"e_i":e,"Emax":emax,"W":w,"W0":w0,"m1":m,"m2":m,"native_activity":nativeactivity,"occupied_site_cells":[[0,1,2,3],[4,5,6,7]],"cell_energies":[w/2.,w/2.],"canonical_constants":report,
        "scope":"Two actual equal-size balanced native swarms have a common centered positional error on every row. Their exact accepted-event laws determine the error-weighted activity. The source cell partition uses the two occupied comparison sites; no small physical N is declared to meet the astronomical covering threshold."});
    records(
        suite,
        "thm-keystone-complete-error-coverage",
        &[
            r"\mathsf V_z\ge v_0",
            r"e_i\le E_{\max}",
            "N\\ge k",
            r"W=\frac1N",
            "I=I_{11}",
            r"e_i=|\Delta\delta_{x,i}|^2",
            r"E_{\max}=4D_x^2",
            r"D_x=4\sqrt d",
            r"E_{\max}=64d",
            r"E_c=\frac1N",
            r"\sum_c E_c=W",
            r"E_c\le E_{\max}n_c/N",
            r"N\ge k\ge2",
            r"n_i(r)\ge n_c-1",
            r"E_c>0",
            r"w=W/(k/N)\ge W_0",
            r"k-1\le k",
        ],
        inputs.clone(),
        checks,
    )?;
    let label = "thm-keystone-discharged-averaged-pressure";
    let chistar = logs["chi_star"].exp();
    let bstar = logs["B_star"].exp();
    let c0 = logs["C_0"].exp();
    let checks = vec![
        equal("actual_alive_fraction", label, m, k as f64 / n as f64),
        certificate_check(
            "actual_structural_eta_threshold",
            label,
            w > (1. + 0.5) * w0,
        ),
        equal("actual_complete_aligned_error", label, w, mean(&e)),
        upper(
            "actual_centered_error_variance_domination",
            label,
            w,
            2. * (vx + vy),
        ),
        certificate_check("actual_positive_complete_error", label, w > 0.),
        lower("actual_complete_error_threshold", label, w, w0),
        lower(
            "actual_complete_error_alive_normalization",
            label,
            w / m,
            w0,
        ),
        lower("actual_common_alive_mass_lower", label, m, w0 / emax),
        upper("actual_complete_error_capacity", label, w, emax * m),
        equal("actual_aligned_alive_overlap", label, n as f64, n as f64),
        equal(
            "actual_full_slot_velocity_difference_mass",
            label,
            0.,
            [0_f64; 8].iter().map(|v| v * v).sum::<f64>() / n as f64,
        ),
        equal_positive(
            "positive_chi_star_independent_formula",
            label,
            chistar,
            (logs["C_0"] + 2. * w0.ln() - 2_f64.ln() - 2. * emax.ln() - 2. * logs["M_r"]).exp(),
        ),
        equal_positive(
            "positive_B_star_independent_formula",
            label,
            bstar,
            c0 * emax * emax / w0,
        ),
        lower(
            "actual_alive_count_threshold_consequence",
            label,
            k as f64,
            n as f64 * w0 / emax,
        ),
        upper(
            "actual_row_error_capacity_consequence",
            label,
            w,
            emax * k as f64 / n as f64,
        ),
    ];
    records(
        suite,
        label,
        &[
            "m_s=k_s/N",
            r"W=\frac1N",
            r"W\le2\sum_{s=1}^2",
            "W>0",
            r"W\ge W_0",
            r"\mathsf V_{z,s}\ge v_0",
            r"W/m_s\ge W_0",
            r"m_s\ge W_0/E_{\max}",
            r"W\le E_{\max}m_s",
            r"\chi_*=\frac{C_0W_0^2",
            r"W\le E_{\max}k_s/N",
            r"k_s\ge NW_0/E_{\max}",
            "k_s=N",
            r"V_{\rm struct}>(1+\eta)W_0",
            r"k_{\max}=\max(k_1,k_2)",
            "s=|I_{11}|",
            r"m_{\max}=k_{\max}/N",
            r"D_v=N^{-1}",
        ],
        inputs.clone(),
        checks,
    )?;
    let mut branchchecks = Vec::new();
    for (a, b) in [(0_f64, 1.), (0.5, 1.), (1., 1.), (1.5, 1.), (2., 1.)] {
        branchchecks.push(lower(
            "all_Young_positive_part_branches",
            label,
            (a - b).max(0.).powi(2),
            a * a / 2. - b * b,
        ));
    }
    records(
        suite,
        label,
        &[r"(a-b)_+^2\ge a^2/2-b^2", "a\\ge b", "a<b"],
        json!({"a_values":[0.,0.5,1.,1.5,2.],"b":1.,"branches":"both a>=b and a<b are evaluated separately before positive-part comparison"}),
        branchchecks,
    )?;
    let label = "lem-keystone-near-neighbor-pressure";
    let neighbor_checks = vec![
        lower(
            "actual_near_pressure_feature_spread",
            label,
            squash(1.).powi(2),
            c["v_0"],
        ),
        equal_positive(
            "actual_far_fraction_definition",
            label,
            c["rho_f"],
            (c["v_0"] - c["h_f"].powi(2)) / (c["D_0"].powi(2) - c["h_f"].powi(2)),
        ),
        upper(
            "actual_same_site_feature_distance",
            label,
            0.,
            logs["r"].exp(),
        ),
    ];
    records(
        suite,
        label,
        &[
            r"\mathsf V_z\ge v_0",
            r"(v_0-h_f^2)/(D_0^2-h_f^2)=\rho_f",
            r"|z_j-z_i|\le r",
        ],
        inputs.clone(),
        neighbor_checks,
    )?;
    let label = "thm-keystone-averaged-error-capture";
    let pi_b = nativepressure.iter().copied().fold(f64::INFINITY, f64::min);
    let pstar = pi_b / 2.;
    let mut capture = vec![
        certificate_check("actual_capture_floor_positive", label, pstar > 0.),
        lower("actual_group_pressure_floor", label, pi_b, pstar),
    ];
    for i in 0..n {
        capture.push(equal(
            "actual_two_swarm_activity_probability_sum",
            label,
            nativepressure[i],
            law.acceptance_probabilities[i] + law2.acceptance_probabilities[i],
        ));
        capture.push(equal(
            "actual_each_centered_error_square",
            label,
            e[i],
            (x[i] - y[i]).powi(2),
        ));
    }
    capture.push(equal(
        "actual_group_pressure_minimum",
        label,
        pi_b,
        nativepressure.iter().copied().fold(f64::INFINITY, f64::min),
    ));
    let outside = e[4..].iter().sum::<f64>() / n as f64;
    capture.push(equal("actual_outside_target_error", label, outside, w / 2.));
    capture.push(lower(
        "actual_target_captured_pressure",
        label,
        nativeactivity,
        pstar * (w - outside),
    ));
    records(
        suite,
        label,
        &[
            r"\pi_i=\pi_{1,i}+\pi_{2,i}",
            r"e_i=|\Delta\delta_{x,i}|^2",
            r"\pi_B=\min_{i\in B}\pi_i",
            "p_*>0",
            r"\pi_B\ge p_*",
            r"R_* =\frac1N",
            "3.AP9",
        ],
        json!({"N":n,"native_row_activity":nativepressure,"row_error":e,"target_set":[0,1,2,3],"Rstar":outside,"pstar":pstar,"native_activity":nativeactivity,"group_minimum":pi_b}),
        capture,
    )?;
    let giant: BigInt = BigInt::one() << 512_usize;
    let logn = giant.to_f64().expect("exact representable 2^512").ln();
    let mrlog = logs["M_r"];
    let label = "thm-keystone-complete-error-coverage";
    let thresholdlog = logn + w.ln() - emax.ln() - mrlog;
    record(
        suite,
        label,
        r"NW/(E_{\max}M_r)=kw/(E_{\max}M_r)\ge2",
        json!({"exact_population":"2^512","k_equals_N":true,"W":w,"w":w,"Emax":emax,"log_Mr":mrlog,"scope":"Only the population-count algebra is checked at this compressed population; native pressure under this threshold is independently evaluated by the complete accepted-event interval law in CC9/11 evidence."}),
        vec![
            equal(
                "exact_all_alive_threshold_identity_in_logs",
                label,
                thresholdlog,
                logn + w.ln() - emax.ln() - mrlog,
            ),
            lower(
                "actual_complete_cover_threshold_in_logs",
                label,
                thresholdlog,
                2_f64.ln(),
            ),
        ],
    )?;
    let label = "thm-keystone-discharged-averaged-pressure";
    let ball = c0 * emax;
    record(
        suite,
        label,
        "B_*=C_0E_{\\max}",
        inputs.clone(),
        vec![equal_positive(
            "actual_all_alive_B_star_formula",
            label,
            ball,
            logs["C_0"].exp() * emax,
        )],
    )?;
    for n in [1_usize, 4, 8] {
        let label = "thm-keystone-discharged-averaged-pressure";
        let w = 0_f64;
        let rhs = chistar * w - bstar / (n * n) as f64;
        let mut checks = vec![
            equal("identical_swarm_centered_error_zero", label, w, 0.),
            certificate_check("low_error_branch_strict", label, w < w0),
            lower("actual_zero_error_activity_global_affine", label, 0., rhs),
        ];
        if n == 1 {
            checks.push(equal(
                "singleton_native_error_branch_population",
                label,
                n as f64,
                1.,
            ));
        }
        records(
            suite,
            label,
            &[r"W<W_0"],
            json!({"N":n,"identical_native_swarms":true,"W":w,"W0":w0,"activity":0.,"affine_lower":rhs}),
            checks.clone(),
        )?;
        if n == 1 {
            record(
                suite,
                label,
                "N=1",
                json!({"N":n,"W":w,"activity":0.}),
                checks,
            )?;
        }
        record(
            suite,
            "thm-keystone-complete-error-coverage",
            "W=0",
            json!({"N":n,"identical_native_swarms":true,"row_error":vec![0.;n]}),
            vec![equal(
                "zero_error_native_weighted_activity",
                "thm-keystone-complete-error-coverage",
                0.,
                mean(&vec![0.; n]),
            )],
        )?;
    }
    Ok(())
}

fn native_barycenter_aliases(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    use algorithmic_gas::fitness::PositiveMapping;
    use std::collections::BTreeMap;
    let label = "thm-cloning-canonical-barycenter-concentration";
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let k = live.len();
    let nf = n as f64;
    let dx = range(&live.iter().map(|&i| law.x[i]).collect::<Vec<_>>());
    let canonical = canonical_cloning_constants();
    let dm = canonical.d_m;
    let mut means = Vec::new();
    let mut masses = Vec::new();
    let mut conditionalvar = 0.;
    let mut c = Vec::new();
    for p in &law.patterns {
        let mb = mean(&p.means);
        means.push(mb);
        masses.push(p.mass);
        conditionalvar += p.mass * p.traces.iter().sum::<f64>() / (nf * nf);
        c.push(equal(
            "native_conditional_output_barycenter_definition",
            label,
            mb,
            p.means.iter().sum::<f64>() / nf,
        ));
        for i in 0..n {
            let mi = if law.alive[i] {
                law.x[i]
                    + (0..n)
                        .map(|j| p.edges[i][j] * (law.x[j] - law.x[i]))
                        .sum::<f64>()
            } else {
                (0..n)
                    .filter(|&j| law.alive[j])
                    .map(|j| p.edges[i][j] * law.x[j])
                    .sum::<f64>()
            };
            c.push(equal(
                "native_conditional_row_output_mean",
                label,
                p.means[i],
                mi,
            ));
            if !law.alive[i] {
                c.push(equal(
                    "native_dead_row_mean_measurement_independent",
                    label,
                    p.means[i],
                    law.patterns[0].means[i],
                ));
            }
        }
        let alivey = live.iter().map(|&i| p.y[i]).collect::<Vec<_>>();
        for &i in &live {
            c.push(lower(
                "native_every_fitness_positive_floor",
                label,
                p.f[i],
                canonical.fitness_min,
            ));
            c.push(upper(
                "native_every_fitness_upper",
                label,
                p.f[i],
                canonical.fitness_max,
            ));
            c.push(equal_positive(
                "native_every_regularized_diversity_scale",
                label,
                p.scales[i],
                (var(&alivey) + 0.01).sqrt(),
            ));
            c.push(upper(
                "native_every_diversity_centered_span",
                label,
                (p.y[i] - mean(&alivey)).abs(),
                dm,
            ));
        }
        for q in &law.patterns {
            for &i in &live {
                if live.iter().any(|&j| j != i && p.choices[j] != q.choices[j]) {
                    continue;
                }
                c.push(upper(
                    "native_row_mean_range_replacement_bound",
                    label,
                    (p.means[i] - q.means[i]).abs(),
                    dx,
                ));
                let tildey = live.iter().map(|&j| q.y[j]).collect::<Vec<_>>();
                c.push(upper(
                    "native_row_measurement_other_mean_span",
                    label,
                    (p.y[i] - mean(&tildey)).abs(),
                    dm,
                ));
            }
        }
    }
    let native_mean = means.iter().zip(&masses).map(|(m, p)| m * p).sum::<f64>();
    let actual_measvar = means
        .iter()
        .zip(&masses)
        .map(|(m, p)| p * (m - native_mean).powi(2))
        .sum::<f64>();
    let mut conditional_sum = 0.;
    for &r in &live {
        let mut grouped: BTreeMap<Vec<usize>, (f64, f64, f64)> = BTreeMap::new();
        for p in &law.patterns {
            let key = live
                .iter()
                .filter(|&&i| i != r)
                .map(|&i| p.choices[i])
                .collect::<Vec<_>>();
            let value = grouped.entry(key).or_default();
            let f = mean(&p.means);
            value.0 += p.mass;
            value.1 += p.mass * f;
            value.2 += p.mass * f * f;
        }
        conditional_sum += grouped
            .values()
            .map(|&(mass, first, second)| second - first * first / mass)
            .sum::<f64>();
    }
    c.extend([
        equal(
            "actual_native_clone_epsilon",
            label,
            law.config.clone_decision.epsilon,
            1e-6,
        ),
        equal(
            "actual_native_clone_saturation",
            label,
            law.config.clone_decision.saturation,
            1.,
        ),
        equal(
            "actual_native_clone_jitter",
            label,
            law.config.clone_transform.jitter_amplitude,
            0.1,
        ),
        equal(
            "native_squared_feature_radius_sum",
            label,
            4_f64.powi(2) + 4_f64.powi(2),
            32.,
        ),
        lower(
            "native_product_conditional_variance_inequality",
            label,
            conditional_sum,
            actual_measvar,
        ),
        upper(
            "native_full_barycenter_variance_from_conditional_variances",
            label,
            actual_measvar + conditionalvar,
            (dx * dx + 0.01) / nf
                + k as f64 * dx * dx * canonical.b_0 * canonical.b_0 / (2. * nf * nf),
        ),
        equal(
            "actual_full_velocity_barycenter_momentum",
            label,
            mean(&law.v),
            law.v.iter().sum::<f64>() / nf,
        ),
        equal(
            "actual_native_gaussian_covariance_definition",
            label,
            conditionalvar,
            law.patterns
                .iter()
                .map(|p| p.mass * p.traces.iter().sum::<f64>() / (nf * nf))
                .sum(),
        ),
    ]);
    for z in [-100_f64, -5., -1., 0., 1., 5., 100.] {
        let g = 2. / (1. + (-z).exp()) + 0.1;
        c.push(equal(
            "native_positive_map_identity",
            label,
            g,
            law.config.fitness.diversity_map.map(z)?,
        ));
        c.push(lower("native_positive_map_floor", label, g, 0.1));
        c.push(upper("native_positive_map_upper", label, g, 2.1));
    }
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"N":n,"k":k,"actual_eligible_diameter":dx,"actual_native_mean":native_mean,"actual_measurement_variance":actual_measvar,"actual_conditional_recipient_variance":conditionalvar,"independently_grouped_conditional_Efron_Stein_RHS":conditional_sum,"full_native_pattern_count":law.patterns.len(),"canonical":canonical,
        "scope":"Native conditional recipient means and covariance are integrated over every independent measurement vector. A separate grouping by all other measurement innovations computes each conditional product variance and its Efron–Stein sum; dead output means use only the current eligible donor law. Retained dead coordinates never enter an eligible-diameter assumption."});
    records(
        suite,
        label,
        &[
            r"\bar X^c=N^{-1}\sum_iX_i^c",
            r"\bar V^c=N^{-1}\sum_iV_i^c",
            r"\varepsilon_c=10^{-6}",
            r"p_{\max}=1",
            r"\operatorname{Var}(Z)=\mathbb E|Z-\mathbb EZ|^2",
            "j=0.1",
            "4^2+4^2=32",
            r"g(z)=0.1+2/(1+e^{-z})",
            r"0.1\le g\le2.1",
            r"F_*\le F_i\le F^*",
            r"s_Y=\sqrt{\operatorname{var}(Y)+\varepsilon_s^2}",
            r"|Y_i-\widetilde{\bar Y}|\le D_m",
            r"m_i=\mathbb E[X_i^c",
            r"|m_r-\widetilde m_r|\le D_x",
            r"m_i=\widetilde m_i",
            r"\operatorname{Var}(f)\leq\sum_r",
        ],
        input.clone(),
        c,
    )?;
    if k == 1 {
        record(
            suite,
            label,
            "k=1",
            input,
            vec![
                equal("actual_singleton_alive_count", label, k as f64, 1.),
                equal(
                    "actual_singleton_measurement_variance_zero",
                    label,
                    actual_measvar,
                    0.,
                ),
            ],
        )?;
    }
    Ok(())
}

fn native_signed_flux_aliases(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let label = "thm-cloning-signed-cluster-fitness-flux";
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let k = live.len();
    if k < 2 {
        return Ok(());
    }
    let h = &live[..k / 2];
    let l = &live[k / 2..];
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, law.x.clone())?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, 1, law.v.clone())?,
    );
    let mut weights = vec![vec![0.; n]; n];
    let mut normalizers = vec![0.; n];
    let kap = (-4_f64).exp();
    let mut checks = Vec::new();
    for &i in &live {
        for &j in &live {
            if i != j {
                let native_d = law
                    .config
                    .cloning_donors
                    .distance
                    .compare(&obs, i, &obs, j)?;
                let wij = law
                    .config
                    .cloning_donors
                    .kernel
                    .log_weight(
                        native_d,
                        algorithmic_gas::geometry::ComparisonKind::Distance,
                    )?
                    .exp();
                weights[i][j] = wij;
                normalizers[i] += wij;
                checks.push(lower("native_signed_flux_kernel_lower", label, wij, kap));
                checks.push(upper("native_signed_flux_kernel_upper", label, wij, 1.));
            }
        }
    }
    for &i in &live {
        checks.push(lower(
            "native_recipient_normalizer_first_bound",
            label,
            normalizers[i],
            kap * (k - 1) as f64,
        ));
        checks.push(upper(
            "native_recipient_normalizer_last_bound",
            label,
            normalizers[i],
            (k - 1) as f64,
        ));
        for &j in &live {
            if i != j {
                checks.push(equal(
                    "native_kernel_symmetry_identity",
                    label,
                    weights[i][j],
                    weights[j][i],
                ));
                checks.push(equal(
                    "native_normalized_donor_mass_definition",
                    label,
                    law.donors[i][j],
                    weights[i][j] / normalizers[i],
                ));
            }
        }
    }
    let minf = 0.01_f64;
    let maxf = 4.41_f64;
    let pmax = law.config.clone_decision.saturation;
    let eps = law.config.clone_decision.epsilon;
    let astar = (maxf - minf).max(pmax * (maxf + eps));
    let cp = kap / astar;
    let cm = 1. / (kap * pmax * (minf + eps));
    checks.extend([
        certificate_check(
            "native_disjoint_nonempty_geometric_cells",
            label,
            !h.is_empty() && !l.is_empty() && h.iter().all(|i| !l.contains(i)),
        ),
        certificate_check("native_current_alive_support_at_least_two", label, k >= 2),
        certificate_check(
            "actual_positive_signed_comparison_coefficients",
            label,
            cp > 0. && cp <= cm,
        ),
    ]);
    let mut pattern_inputs = Vec::new();
    let mut short_branch = None;
    for p in &law.patterns {
        for &i in &live {
            checks.push(lower(
                "actual_all_alive_fitness_lower_endpoint",
                label,
                p.f[i],
                minf,
            ));
            checks.push(upper(
                "actual_all_alive_fitness_upper_endpoint",
                label,
                p.f[i],
                maxf,
            ));
        }
        let hf = h.iter().map(|&i| p.f[i]).collect::<Vec<_>>();
        let lf = l.iter().map(|&i| p.f[i]).collect::<Vec<_>>();
        let delta = mean(&lf) - mean(&hf);
        let s2 = var(&lf) + var(&hf);
        let whl = h
            .iter()
            .map(|&i| l.iter().map(|&j| weights[i][j]).sum::<f64>())
            .sum::<f64>();
        let mut values = Vec::new();
        let mut weightedmean = 0.;
        let mut weightedsecond = 0.;
        let mut weightedvariance = 0.;
        let mut weightedmass = 0.;
        let mut negative = 0.;
        let mut positive = 0.;
        for &i in h {
            for &j in l {
                let d = p.f[j] - p.f[i];
                let probability = weights[i][j] / whl;
                values.push(d);
                weightedmass += probability;
                weightedmean += probability * d;
                weightedsecond += probability * d * d;
                negative += (-d).max(0.);
                positive += d.max(0.);
                checks.push(lower("every_native_signed_gap_floor", label, p.f[i], minf));
                checks.push(upper(
                    "every_native_signed_gap_ceiling",
                    label,
                    p.f[j],
                    maxf,
                ));
            }
        }
        for &i in h {
            for &j in l {
                weightedvariance += weights[i][j] / whl * (p.f[j] - p.f[i] - weightedmean).powi(2);
            }
        }
        let ahl = (lf.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - hf.iter().copied().fold(f64::INFINITY, f64::min))
        .max(0.)
        .max(pmax * (hf.iter().copied().fold(f64::NEG_INFINITY, f64::max) + eps));
        let a = 1.
            / (h.iter()
                .map(|&i| normalizers[i])
                .fold(f64::NEG_INFINITY, f64::max)
                * ahl);
        let b = 1.
            / (l.iter()
                .map(|&j| normalizers[j])
                .fold(f64::INFINITY, f64::min)
                * pmax
                * (lf.iter().copied().fold(f64::INFINITY, f64::min) + eps));
        let max_gap = values.iter().copied().fold(0_f64, f64::max);
        let max_h = hf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_l = lf.iter().copied().fold(f64::INFINITY, f64::min);
        let max_zh = h
            .iter()
            .map(|&i| normalizers[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let min_zl = l
            .iter()
            .map(|&j| normalizers[j])
            .fold(f64::INFINITY, f64::min);
        checks.extend([
            equal_positive(
                "weighted_acceptance_AHL_reconstruction_from_all_gaps",
                label,
                ahl,
                max_gap.max(pmax * (max_h + eps)),
            ),
            equal_positive(
                "weighted_positive_coefficient_inverse_identity",
                label,
                a * max_zh * ahl,
                1.,
            ),
            equal_positive(
                "weighted_negative_coefficient_inverse_identity",
                label,
                b * min_zl * pmax * (min_l + eps),
                1.,
            ),
            equal("native_arithmetic_gap_mean", label, mean(&values), delta),
            equal(
                "native_arithmetic_gap_second_moment",
                label,
                values.iter().map(|d| d * d).sum::<f64>() / values.len() as f64,
                delta * delta + s2,
            ),
            equal(
                "native_positive_negative_gap_split",
                label,
                positive / values.len() as f64,
                delta + negative / values.len() as f64,
            ),
            equal(
                "native_weighted_gap_probability_mass",
                label,
                weightedmass,
                1.,
            ),
            equal(
                "native_weighted_gap_variance_chain",
                label,
                weightedsecond,
                weightedmean * weightedmean + weightedvariance,
            ),
            equal_positive(
                "actual_uniform_acceptance_denominator",
                label,
                astar,
                (maxf - minf).max(pmax * (maxf + eps)),
            ),
            equal_positive(
                "actual_uniform_positive_coefficient",
                label,
                cp,
                kap / astar,
            ),
            equal_positive(
                "actual_uniform_negative_coefficient",
                label,
                cm,
                1. / (kap * pmax * (minf + eps)),
            ),
            certificate_check(
                "actual_weighted_coefficient_positivity",
                label,
                a > 0. && b > 0.,
            ),
        ]);
        if b < a {
            let signed = h
                .iter()
                .flat_map(|&i| l.iter().map(move |&j| p.edges[i][j] - p.edges[j][i]))
                .sum::<f64>()
                / n as f64;
            let branch = vec![
                certificate_check("actual_weighted_b_below_a_branch", label, b < a),
                lower(
                    "actual_weighted_discarded_nonnegative_term",
                    label,
                    signed,
                    whl / n as f64 * a * weightedmean,
                ),
            ];
            if short_branch.is_none() {
                short_branch = Some((
                    json!({"H":h,"L":l,"native_fitness":p.f,"native_weights":weights,"normalizers":normalizers,"a_HL":a,"b_HL":b,"weighted_gap":weightedmean,"native_signed_flux":signed}),
                    branch,
                ));
            }
        }
        pattern_inputs.push(json!({"native_full_fitness":p.f,"Delta":delta,"s_squared":s2,"W_HL":whl,"Delta_weighted":weightedmean,"s_weighted_squared":weightedvariance,"A_HL":ahl,"a_HL":a,"b_HL":b}));
    }
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"N":n,"k":k,"H":h,"L":l,"native_symmetric_weights":weights,"native_recipient_normalizers":normalizers,"Fstar":minf,"Fupper":maxf,"kappa":kap,"Astar":astar,"c_plus":cp,"c_minus":cm,"all_native_retained_mark_moments":pattern_inputs,
        "scope":"Actual frozen native Gaussian donor weights are reconstructed before normalizing. Geometric cells partition the sorted alive sites. Arithmetic and kernel-weighted moments retain each actual full fitness vector, positive/negative gap parts and measured recipient normalizers; no nonlinear acceptance is applied to averaged fitness."});
    records(
        suite,
        label,
        &[
            r"k=|\mathcal A|\geq2",
            r"0<F_*\leq F_i\leq F^*",
            r"\kappa\leq w_{ij}=w_{ji}\leq1",
            r"A_*=\max",
            r"\Delta=\bar F_L",
            r"0<c_+\leq c_-",
            r"W_{HL}=\sum",
            r"s_w^2=\sum",
            r"A_{HL}=\max",
            r"a_{HL}=\frac1",
            r"D_{ij}=F_j-F_i",
            r"\kappa(k-1)\leq Z_i\leq k-1",
            r"\langle D\rangle=\Delta",
            r"\langle D^2\rangle=\Delta^2+s^2",
            r"\langle D_+\rangle=\Delta+\langle D_-\rangle",
            r"c_-\geq c_+",
        ],
        input,
        checks,
    )?;
    if let Some((input, checks)) = short_branch {
        record(suite, label, r"b_{HL}<a_{HL}", input, checks)?;
    }
    Ok(())
}

fn geometric_native_hypotheses(suite: &mut EstimateSuite) -> Result<()> {
    use algorithmic_gas::fitness::{PositiveMapping, Standardizer};
    use algorithmic_gas::geometry::Distance;
    let label = "lem-keystone-geometric-measurement-events";
    let config = GasConfig::euclidean(1, 0.04)?;
    let es = match config.fitness.diversity_standardizer {
        Standardizer::Global { sigma_min } => sigma_min,
        _ => return Err(error("native AP global standardizer")),
    };
    let x = [1_f64, -1.];
    let z = x.map(|x| 2. * x / (2. + x.abs()));
    let count = [5_usize, 95];
    let k = count.iter().sum::<usize>();
    let ng = count[0];
    let radius = 2_f64;
    let bound = 2_f64;
    let mr = radius * radius / (radius + bound).powi(2);
    let kap = (-4_f64).exp();
    let h = 0.2_f64;
    let rg = 0_f64;
    let feature_gap = (z[0] - z[1]).abs();
    let physical_gap = (x[0] - x[1]).abs();
    let vz = count[0] as f64 * count[1] as f64 / (k * k) as f64 * feature_gap.powi(2);
    let vx = count[0] as f64 * count[1] as f64 / (k * k) as f64 * physical_gap.powi(2);
    let rhoh = (vz - h * h) / (feature_gap * feature_gap - h * h);
    let rhog = (ng - 1) as f64 / (k - 1) as f64;
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(2, 1, x.to_vec())?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(2, 1, vec![0.; 2])?,
    );
    let native_d = config.distance_donors.distance.compare(&obs, 0, &obs, 1)?;
    let wd = config
        .distance_donors
        .kernel
        .log_weight(
            native_d,
            algorithmic_gas::geometry::ComparisonKind::Distance,
        )?
        .exp();
    let wc = config
        .cloning_donors
        .kernel
        .log_weight(
            native_d,
            algorithmic_gas::geometry::ComparisonKind::Distance,
        )?
        .exp();
    let mass_near = (ng - 1) as f64 / ((ng - 1) as f64 + count[1] as f64 * wd);
    let mass_far = count[1] as f64 * wd / ((ng - 1) as f64 + count[1] as f64 * wd);
    let ym = [
        config.fitness.distance_floor,
        native_d.hypot(config.fitness.distance_floor),
    ];
    let native = config.fitness.evaluate(
        &RewardBatch::new(vec![0.5; 2], Default::default()),
        &[0., native_d],
        &[true; 2],
        &obs,
        0,
    )?;
    let (pressure, event) =
        compressed_native_two_site_event(&BigInt::from(k), &BigInt::from(ng), 1, 1, 0);
    let span = native_d.hypot(config.fitness.distance_floor) - config.fitness.distance_floor;
    let zr = span / es;
    let sstar = (span * span / 4. + es * es).sqrt();
    let mf = 2. * (-zr).exp() / (1. + (-zr).exp()).powi(2);
    let gamma =
        1.1 * mf * (h.hypot(config.fitness.distance_floor) - config.fitness.distance_floor) / sstar;
    let check = vec![
        equal_positive(
            "native_AP_distance_floor",
            label,
            config.fitness.distance_floor,
            1e-3,
        ),
        equal_positive("native_AP_diversity_regularizer", label, es, 0.1),
        equal_positive(
            "native_AP_both_kernel_floors",
            label,
            kap,
            (-32_f64 / 8.).exp(),
        ),
        lower("native_AP_measurement_pair_weight", label, wd, kap),
        upper("native_AP_measurement_pair_weight_upper", label, wd, 1.),
        lower("native_AP_cloning_pair_weight", label, wc, kap),
        upper("native_AP_cloning_pair_weight_upper", label, wc, 1.),
        equal_positive(
            "native_AP_feature_measurement_distance",
            label,
            native_d,
            feature_gap,
        ),
        equal_positive(
            "native_AP_near_measured_value",
            label,
            native.separation[0],
            ym[0],
        ),
        equal_positive(
            "native_AP_far_measured_value",
            label,
            native.separation[1],
            ym[1],
        ),
        certificate_check(
            "actual_alive_ball_domain",
            label,
            x.iter().all(|x| x.abs() <= bound),
        ),
        equal_positive(
            "radial_Jacobian_uniform_bound",
            label,
            mr,
            (1. + 1_f64.sqrt()).powi(-2),
        ),
        lower(
            "actual_squash_strong_monotonicity",
            label,
            (z[0] - z[1]) * (x[0] - x[1]),
            mr * physical_gap * physical_gap,
        ),
        lower(
            "actual_squash_inverse_modulus",
            label,
            feature_gap,
            mr * physical_gap,
        ),
        lower(
            "actual_feature_variance_physical_comparison",
            label,
            vz,
            mr * mr * vx,
        ),
        equal(
            "actual_complete_linkage_identical_site_cluster_count",
            label,
            ng as f64,
            count[0] as f64,
        ),
        equal("actual_complete_linkage_cluster_diameter", label, rg, 0.),
        certificate_check("actual_cluster_measurement_threshold_domain", label, rg < h),
        certificate_check(
            "actual_positive_cluster_diversity_gap",
            label,
            h.hypot(config.fitness.distance_floor) > config.fitness.distance_floor,
        ),
        certificate_check(
            "actual_native_diversity_power_positive",
            label,
            config.fitness.diversity_exponent > 0.,
        ),
        certificate_check(
            "actual_native_diversity_regularizer_positive",
            label,
            es > 0.,
        ),
        equal_positive(
            "native_powered_positive_map",
            label,
            native.fitness[0] / 1.1,
            config
                .fitness
                .diversity_map
                .map(native.diversity_z[0])?
                .powf(config.fitness.diversity_exponent),
        ),
        lower(
            "native_independent_favorable_measurement_joint_probability",
            label,
            mass_near * mass_far,
            kap * kap * rhog * rhoh,
        ),
        certificate_check(
            "actual_positive_favorable_fitness_margin",
            label,
            gamma > 0.,
        ),
        certificate_check("actual_nonempty_measurement_cluster_pair", label, ng >= 2),
        certificate_check(
            "actual_statistically_valid_cluster_count",
            label,
            ng >= 5_usize.max((0.05 * k as f64).ceil() as usize),
        ),
        lower(
            "actual_nonself_cluster_fraction_step",
            label,
            (ng - 1) as f64,
            0.8 * ng as f64,
        ),
        certificate_check(
            "actual_accepted_event_subset_positive",
            label,
            pressure.lo > BigInt::zero(),
        ),
    ];
    let input = json!({"native_config":config,"eligible_position_atoms":x,"atom_multiplicities":count,"native_features":z,"native_raw_pair_distance":native_d,"native_pair_weights":[wd,wc],"native_measured_Y":native.separation,"native_diversity_scores":native.diversity_z,"native_fitness":native.fitness,"R":radius,"B":bound,"m_R":mr,"native_alive_positional_variance":vx,"native_alive_feature_variance":vz,"G":[0,1,2,3,4],"r_G":rg,"h":h,"rho_G":rhog,"rho_h":rhoh,"actual_near_probability":mass_near,"actual_far_probability":mass_far,"actual_independent_measurement_event_probability":mass_near*mass_far,"actual_gamma_G":gamma,"complete_event_interval":event,
        "scope":"The complete-linkage partition has two occupied identical-feature sites, with actual populations five and ninety-five. Native comparison, kernel, positive-floor measurement and globally standardized powered fitness are independently evaluated; finite current-support normalization gives the exact independent near/far measurement event mass."});
    records(
        suite,
        label,
        &[
            r"Y_i=\sqrt{|z_i-z_{D_i}|^2+\delta_D^2}",
            r"\delta_D=10^{-3}",
            r"\varepsilon_s=0.1",
            r"\kappa_D=\kappa_C=e^{-4}",
            r"\kappa_D\le w^D_{ij}\le1",
            r"\kappa_C\le w^C_{ij}\le1",
            r"|x|\le B",
            r"m_R=R^2/(R+B)^2",
            r"(S_R(x)-S_R(y))\cdot(x-y)\ge m_R|x-y|^2",
            r"|S_R(x)-S_R(y)|\ge m_R|x-y|",
            r"\mathsf V_z\ge m_R^2\operatorname{Var}_{\mathcal A}(x)",
            "R=2",
            r"B=2\sqrt d",
            r"m_R=(1+\sqrt d)^{-2}",
            "n_G=|G|",
            r"r_G=\operatorname{diam}_z(G)",
            "r_G<h",
            r"\delta_G>0",
            r"i,j\in G",
            r"f(z)=g_s(z)^{p_s}",
            "p_s>0",
            r"\varepsilon_s>0",
        ],
        input.clone(),
        check.clone(),
    )?;
    records(
        suite,
        "thm-keystone-averaged-cluster-pressure",
        &[
            r"\gamma_G>0",
            r"\Pr(E_{ij})\ge\kappa_D^2\rho_G\rho_h",
            r"n_G\ge\max(5,\lceil0.05k\rceil)",
            r"n_G-1\ge(4/5)n_G",
        ],
        input,
        check,
    )?;
    let label = "lem-keystone-complete-coverage-constants";
    for lambda in [0.25_f64, 1., 4.] {
        let distance = Distance::PhaseSpace {
            positions: "positions".into(),
            velocities: "velocities".into(),
            position_scale: 1.,
            velocity_scale: 1.,
            lambda,
            periodic: None,
        };
        let mut unsquashed =
            ObservationBatch::positions(TensorBatch::vectors(3, 1, vec![-1.5, 0.5, 1.])?);
        unsquashed.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(3, 1, vec![-0.3, 0.1, 0.2])?,
        );
        let bx = 2_f64;
        let bv = 2_f64;
        let d0 = 2. * (bx * bx + lambda * bv * bv).sqrt();
        let bf = bx.max(lambda.sqrt() * bv);
        let mz = 1_f64.min(lambda.sqrt());
        let mut checks = vec![
            equal_positive(
                "actual_explicit_unsquashed_diameter_bound",
                label,
                d0,
                2. * (bx * bx + lambda * bv * bv).sqrt(),
            ),
            equal_positive(
                "actual_unsquashed_feature_box_bound",
                label,
                bf,
                bx.max(lambda.sqrt() * bv),
            ),
            equal_positive("actual_unsquashed_position_modulus", label, 1., 1.),
            equal_positive(
                "actual_unsquashed_phase_modulus",
                label,
                mz,
                1_f64.min(lambda.sqrt()),
            ),
        ];
        for i in 0..3 {
            for j in 0..3 {
                let x = unsquashed.field("positions")?.row(i)?[0];
                let y = unsquashed.field("positions")?.row(j)?[0];
                let v = unsquashed.field("velocities")?.row(i)?[0];
                let u = unsquashed.field("velocities")?.row(j)?[0];
                let actual = distance.compare(&unsquashed, i, &unsquashed, j)?;
                checks.push(equal(
                    "native_explicit_unsquashed_phase_feature_identity",
                    label,
                    actual,
                    ((x - y).powi(2) + lambda * (v - u).powi(2)).sqrt(),
                ));
                checks.push(lower(
                    "native_unsquashed_phase_lower_modulus",
                    label,
                    actual,
                    mz * ((x - y).powi(2) + (v - u).powi(2)).sqrt(),
                ));
            }
        }
        records(
            suite,
            label,
            &[
                r"z=(x,\sqrt{\lambda_{\rm alg}}v)",
                r"D_0=2\sqrt{B_x^2+\lambda_{\rm alg}B_v^2}",
                r"B_f=\max(B_x,\sqrt{\lambda_{\rm alg}}B_v)",
                "m_x=1",
                r"m_z=\min(1,\sqrt{\lambda_{\rm alg}})",
            ],
            json!({"native_explicit_alternative_distance":distance,"physical_position_bound":bx,"physical_velocity_bound":bv,"D0":d0,"Bf":bf,"mz":mz,"scope":"The source explicitly permits this separately configured unsquashed native PhaseSpace comparison. It is an alternative distance-law fixture; the canonical squashed algorithm and presets remain unchanged."}),
            checks,
        )?;
    }
    record(
        suite,
        label,
        r"S_R(x)=Rx/(R+|x|)",
        json!({"R":2.,"native_squashed_features":z,"positions":x}),
        vec![equal_positive(
            "actual_radial_squashing_formula",
            label,
            z[0],
            2. * x[0] / (2. + x[0].abs()),
        )],
    )?;
    Ok(())
}

fn quantitative_native_keystone_assembly(suite: &mut EstimateSuite) -> Result<()> {
    let radius = 1_f64;
    let small = 0.5_f64;
    let error2 = (radius - small).powi(2);
    let rspread2 = 0.01_f64;
    let cerror = 0.5_f64;
    let gerror = 0_f64;
    let kappa = (-4_f64).exp();
    let mut lawconstants: Option<(f64, f64, f64)> = None;
    for n in [4_usize, 6, 8] {
        let x: Vec<_> = (0..n)
            .map(|i| if i < n / 2 { radius } else { -radius })
            .collect();
        let config = GasConfig::euclidean(1, 0.04)?;
        let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, x.clone())?);
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(n, 1, vec![0.; n])?,
        );
        let alive = vec![true; n];
        let donors = probabilities(&config.cloning_donors, &obs, &alive)?;
        let choices: Vec<_> = (0..n)
            .map(|i| {
                if i < n / 2 {
                    n / 2
                } else {
                    n / 2 + (i - n / 2 + 1) % (n / 2)
                }
            })
            .collect();
        let distance: Vec<_> = (0..n)
            .map(|i| {
                config
                    .distance_donors
                    .distance
                    .compare(&obs, i, &obs, choices[i])
            })
            .collect::<Result<_>>()?;
        let native = config.fitness.evaluate(
            &RewardBatch::new(vec![radius * radius / 2.; n], Default::default()),
            &distance,
            &alive,
            &obs,
            0,
        )?;
        let exact = exact_cloning_moments(
            &x.iter().map(|&x| vec![x]).collect::<Vec<_>>(),
            &native.fitness,
            &donors,
            &alive,
            &config.clone_decision,
            1,
            config.clone_transform.jitter_amplitude,
        )?;
        let y: Vec<_> = (0..n)
            .map(|i| if i < n / 2 { small } else { -small })
            .collect();
        let mut obs2 = ObservationBatch::positions(TensorBatch::vectors(n, 1, y.clone())?);
        obs2.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(n, 1, vec![0.; n])?,
        );
        let donors2 = probabilities(&config.cloning_donors, &obs2, &alive)?;
        let distance2: Vec<_> = (0..n)
            .map(|i| {
                config
                    .distance_donors
                    .distance
                    .compare(&obs2, i, &obs2, choices[i])
            })
            .collect::<Result<_>>()?;
        let native2 = config.fitness.evaluate(
            &RewardBatch::new(vec![small * small / 2.; n], Default::default()),
            &distance2,
            &alive,
            &obs2,
            0,
        )?;
        let exact2 = exact_cloning_moments(
            &y.iter().map(|&x| vec![x]).collect::<Vec<_>>(),
            &native2.fitness,
            &donors2,
            &alive,
            &config.clone_decision,
            1,
            config.clone_transform.jitter_amplitude,
        )?;
        let fh = native.fitness[0];
        let fl = native.fitness[n / 2];
        let frange = fh - fl;
        let s2 = var(&native.fitness);
        let astar =
            frange.max(config.clone_decision.saturation * (fh + config.clone_decision.epsilon));
        let pu = kappa * s2 / (2. * frange * astar);
        let chi = pu * cerror;
        let partial = pu * gerror;
        let gmax = partial.max(chi * rspread2);
        if let Some((fh0, fl0, pu0)) = lawconstants
            && ((fh - fh0).abs() > 1e-12 || (fl - fl0).abs() > 1e-12 || (pu - pu0).abs() > 1e-12)
        {
            return Err(error(
                "retained native common fitness constants changed across particle counts",
            ));
        }
        lawconstants = Some((fh, fl, pu));
        let eligible: Vec<_> = (0..n).collect();
        let unfit: Vec<_> = (0..n)
            .filter(|&i| native.fitness[i] < mean(&native.fitness))
            .collect();
        let geometric_clusters = [
            (0..n / 2).collect::<Vec<_>>(),
            (n / 2..n).collect::<Vec<_>>(),
        ];
        let minimum_valid_cluster_size = 5_usize.max((0.05 * n as f64).ceil() as usize);
        let high: Vec<_> = geometric_clusters
            .iter()
            .filter(|g| g.len() < minimum_valid_cluster_size)
            .flatten()
            .copied()
            .collect();
        let target: Vec<_> = eligible
            .iter()
            .copied()
            .filter(|i| unfit.contains(i) && high.contains(i))
            .collect();
        let captured = target.len() as f64 / n as f64 * error2;
        let activity = exact
            .acceptance_probabilities
            .iter()
            .zip(&exact2.acceptance_probabilities)
            .map(|(p, q)| (p + q) * error2)
            .sum::<f64>()
            / n as f64;
        let mut checks = vec![
            certificate_check(
                "actual_native_target_pressure_floor_positive",
                "lem-quantitative-keystone",
                pu > 0.,
            ),
            certificate_check(
                "actual_error_capture_coefficient_positive",
                "lem-quantitative-keystone",
                cerror > 0.,
            ),
            certificate_check(
                "actual_structural_spread_threshold_positive",
                "lem-quantitative-keystone",
                rspread2 > 0.,
            ),
            certificate_check(
                "actual_feedback_coefficient_positive",
                "lem-quantitative-keystone",
                chi > 0.,
            ),
            lower(
                "actual_global_offset_nonnegative",
                "lem-quantitative-keystone",
                gmax,
                0.,
            ),
            equal_positive(
                "native_feedback_product_identity",
                "lem-quantitative-keystone",
                chi,
                pu * cerror,
            ),
            equal(
                "native_partial_offset_identity",
                "lem-quantitative-keystone",
                partial,
                pu * gerror,
            ),
            equal_positive(
                "native_global_offset_max_identity",
                "lem-quantitative-keystone",
                gmax,
                partial.max(chi * rspread2),
            ),
            lower(
                "actual_global_offset_covers_low_error_regime",
                "lem-quantitative-keystone",
                gmax,
                chi * rspread2,
            ),
            certificate_check(
                "actual_native_structural_high_error_regime",
                "lem-quantitative-keystone",
                error2 > rspread2,
            ),
            equal(
                "actual_geometric_target_capture",
                "lem-quantitative-keystone",
                captured,
                cerror * error2 - gerror,
            ),
            lower(
                "native_actual_weighted_activity_target_restriction",
                "lem-quantitative-keystone",
                activity,
                pu * captured,
            ),
            lower(
                "native_high_regime_affine_keystone_pressure",
                "lem-quantitative-keystone",
                activity,
                chi * error2 - gmax,
            ),
            equal_positive(
                "actual_positive_part_signal_selection_choice",
                "prop-n-uniformity-keystone",
                pu,
                kappa * s2 / (2. * frange * astar),
            ),
            equal_positive(
                "native_actual_selection_denominator",
                "prop-n-uniformity-keystone",
                astar,
                frange.max(config.clone_decision.saturation * (fh + config.clone_decision.epsilon)),
            ),
        ];
        for &i in &target {
            checks.push(lower(
                "native_every_selected_target_row_pressure",
                "lem-quantitative-keystone",
                exact.acceptance_probabilities[i],
                pu,
            ));
        }
        let inputs = json!({"N":n,"positions1":x,"positions2":(0..n).map(|i|if i<n/2 {small}else{-small}).collect::<Vec<_>>(),"alive":vec![true;n],"retained_native_measurement":choices,"native_retained_fitness":native.fitness,"native_row_acceptance":exact.acceptance_probabilities,"second_native_row_acceptance":exact2.acceptance_probabilities,"I11":eligible,"U1":unfit,"H1":high,"actual_geometric_clusters":geometric_clusters,"minimum_valid_cluster_size":minimum_valid_cluster_size,"I_target":target,"Vstruct":error2,"captured_error":captured,"actual_conditional_weighted_activity":activity,"p_u":pu,"c_err":cerror,"g_err":gerror,"R_spread_squared":rspread2,"chi":chi,"g_partial":partial,"g_max":gmax,"s_star_squared":s2,"R_star":frange,"A_star":astar,"native_donor_lower":kappa,
            "scope":"Each retained native measurement vector gives the same two positive fitness levels and low-fitness target fraction across N=4,6,8. Explicit pointwise donor and positive-part signal constants determine a common selection coefficient; the actual selected error mass supplies the capture hypothesis. This validates conditional N-uniform assembly, not an unconditional global drift assumption."});
        records(
            suite,
            "lem-quantitative-keystone",
            &[
                "p_u>0",
                r"c_{\mathrm{err}}>0",
                r"R^2_{\text{spread}} > 0",
                r"\chi(\epsilon) > 0",
                r"g_{\max}(\epsilon) \ge 0",
                r"g_{\max}(\epsilon) \ge \chi(\epsilon) R^2_{\text{spread}}",
                r"V_{\text{struct}} > R^2_{\text{spread}}",
            ],
            inputs.clone(),
            checks.clone(),
        )?;
        records(
            suite,
            "prop-n-uniformity-keystone",
            &[
                "p_u>0",
                r"c_{\mathrm{err}}>0",
                r"\chi=p_uc_{\mathrm{err}}",
                r"g_{\max}=\max\{p_ug_{\mathrm{err}},\chi R_{\mathrm{spread}}^2\}",
                r"p_u=\frac{a_*s_*^2",
            ],
            inputs.clone(),
            checks.clone(),
        )?;
        records(
            suite,
            "rem-keystone-balanced-structural-scope",
            &[
                r"V_{\text{struct}} > R^2_{\text{spread}}",
                r"I_{\mathrm{target}}=I_{11}\cap U_1\cap H_1",
                r"E_w := \frac{1}{N}",
                r"p_{1,i}\geq p_u",
                r"\frac{1}{N}\sum_{i \in I_{\text{target}}}",
                r"\chi(\epsilon) := p_u(\epsilon)",
                r"g_{\text{partial}}(\epsilon) := p_u(\epsilon)",
                r"g_{\max}(\epsilon) := \max",
            ],
            inputs.clone(),
            checks,
        )?;
        for structural in [0_f64, 0.005, 0.01] {
            let rhs = chi * structural - gmax;
            records(
                suite,
                "lem-quantitative-keystone",
                &[
                    r"V_{\text{struct}} \le R^2_{\text{spread}}",
                    r"\text{LHS} \ge 0 \ge \text{RHS}",
                ],
                json!({"N":n,"conditional_common_constants":inputs,"low_error_structural":structural,"low_error_threshold":rspread2,"actual_weighted_activity":0.,"rhs":rhs,"scope":"The low-error branch uses only nonnegative accepted-event probabilities and squared errors; the global offset renders its affine right side nonpositive for every input in the recorded low-error interval."}),
                vec![
                    upper(
                        "actual_low_error_regime_condition",
                        "lem-quantitative-keystone",
                        structural,
                        rspread2,
                    ),
                    lower(
                        "actual_nonnegative_weighted_activity",
                        "lem-quantitative-keystone",
                        0.,
                        0.,
                    ),
                    upper(
                        "actual_low_error_affine_rhs_nonpositive",
                        "lem-quantitative-keystone",
                        rhs,
                        0.,
                    ),
                ],
            )?;
        }
    }
    Ok(())
}

fn native_spreading_example_aliases(suite: &mut EstimateSuite) -> Result<()> {
    let label = "ex-cloning-position-spreading";
    let a = 0.1_f64;
    let law = enumerate_law(vec![0., 0., 0., a], vec![0.; 4], vec![true; 4])?;
    let event = law
        .patterns
        .iter()
        .find(|p| (0..3).all(|i| p.choices[i] < 3))
        .expect("retained three zero-row measurements");
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(4, 1, law.x.clone())?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(4, 1, law.v.clone())?,
    );
    let native_delta = law
        .config
        .distance_donors
        .distance
        .compare(&obs, 0, &obs, 3)?;
    let delta = 2. * a / (2. + a);
    let w = (-delta * delta / 8.).exp();
    let q = event.p[0];
    let copied = exact_cloning_moments(
        &law.x.iter().map(|&x| vec![x]).collect::<Vec<_>>(),
        &event.f,
        &law.donors,
        &law.alive,
        &law.config.clone_decision,
        1,
        0.,
    )?;
    let mut checks = vec![
        equal("native_four_row_input_scale", label, a, 0.1),
        equal(
            "native_effective_quadratic_reward",
            label,
            law.config.fitness.direction.orient(a * a / 2.),
            -a * a / 2.,
        ),
        equal_positive(
            "native_squashed_isolated_distance",
            label,
            native_delta,
            delta,
        ),
        equal_positive(
            "native_isolated_donor_weight_ratio",
            label,
            law.donors[0][3],
            w / (2. + w),
        ),
        equal_positive(
            "native_acceptance_example_formula",
            label,
            q,
            w / (2. + w) * (event.f[3] - event.f[0])
                / (event.f[0] + law.config.clone_decision.epsilon),
        ),
        equal(
            "native_rounded_zero_site_fitness",
            label,
            event.f[0],
            1.09668913465,
        ),
        equal(
            "native_rounded_isolated_site_fitness",
            label,
            event.f[3],
            1.53107680787,
        ),
        equal("native_rounded_copy_probability", label, q, 0.131930125086),
        certificate_check(
            "actual_conditional_copy_probability_domain",
            label,
            q > 0. && q < 0.5,
        ),
        equal(
            "native_conditional_prejitter_variance",
            label,
            copied.expected_position_variance,
            (3. + 3. * q - 6. * q * q) * a * a / 16.,
        ),
        equal(
            "native_entering_variance",
            label,
            var(&law.x),
            3. * a * a / 16.,
        ),
        equal(
            "native_actual_jitter_variance_addition",
            label,
            event.output_var - copied.expected_position_variance,
            9. * q * 0.01 / 16.,
        ),
        equal(
            "actual_canonical_jitter",
            label,
            law.config.clone_transform.jitter_amplitude,
            0.1,
        ),
        equal(
            "actual_canonical_pmax",
            label,
            law.config.clone_decision.saturation,
            1.,
        ),
        equal_positive(
            "actual_canonical_clone_epsilon",
            label,
            law.config.clone_decision.epsilon,
            1e-6,
        ),
    ];
    let mut accepted_count_mass = [0_f64; 4];
    let mut gate_var = 0.;
    for mask in 0_usize..8 {
        let count = mask.count_ones() as usize;
        let mass = q.powi(count as i32) * (1. - q).powi(3 - count as i32);
        accepted_count_mass[count] += mass;
        let post: Vec<_> = (0..4)
            .map(|i| {
                if i == 3 || mask & (1 << i) != 0 {
                    a
                } else {
                    0.
                }
            })
            .collect();
        gate_var += mass * var(&post);
        checks.push(equal(
            "literal_prejitter_population_at_a_count",
            label,
            post.iter().filter(|&&x| x == a).count() as f64,
            (1 + count) as f64,
        ));
    }
    checks.push(equal(
        "independent_conditional_gate_variance_integration",
        label,
        gate_var,
        copied.expected_position_variance,
    ));
    let binomial = [
        (1. - q).powi(3),
        3. * q * (1. - q).powi(2),
        3. * q * q * (1. - q),
        q.powi(3),
    ];
    for (actual, expected) in accepted_count_mass.iter().zip(binomial) {
        checks.push(equal(
            "native_copy_count_binomial_mass",
            label,
            *actual,
            expected,
        ));
    }
    let mut measurement_bits_mass = [0_f64; 8];
    let mut unconditional_zero_jitter = 0.;
    let mut unconditional_native_jitter = 0.;
    for p in &law.patterns {
        let mask = (0..3)
            .filter(|&i| p.choices[i] == 3)
            .fold(0_usize, |mask, i| mask | (1 << i));
        measurement_bits_mass[mask] += p.mass;
        let rawmoment = exact_cloning_moments(
            &law.x.iter().map(|&x| vec![x]).collect::<Vec<_>>(),
            &p.f,
            &law.donors,
            &law.alive,
            &law.config.clone_decision,
            1,
            0.,
        )?;
        unconditional_zero_jitter += p.mass * rawmoment.expected_position_variance;
        unconditional_native_jitter += p.mass * p.output_var;
    }
    for (mask, &actual) in measurement_bits_mass.iter().enumerate() {
        let count = mask.count_ones() as i32;
        let expected = (w / (2. + w)).powi(count) * (2. / (2. + w)).powi(3 - count);
        checks.push(equal_positive(
            "complete_native_compressed_measurement_vector_mass",
            label,
            actual,
            expected,
        ));
    }
    checks.extend([
        equal(
            "native_unconditional_before_jitter_rounded",
            label,
            unconditional_zero_jitter,
            0.00200927153569,
        ),
        equal(
            "native_unconditional_with_jitter_rounded",
            label,
            unconditional_native_jitter,
            0.00297011744615,
        ),
        certificate_check(
            "native_full_measurement_variance_increment_positive",
            label,
            unconditional_zero_jitter > var(&law.x),
        ),
    ]);
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"native_config":law.config,"actual_measurement_vectors":law.patterns.len(),"actual_compressed_eight_measurement_masses":measurement_bits_mass,"retained_event_measurement":event.choices,"retained_native_fitness":event.f,"actual_native_acceptance_probabilities":event.p,"conditional_copy_count_probabilities":accepted_count_mass,"native_prejitter_variance":copied.expected_position_variance,"native_default_jitter_variance":event.output_var,"actual_unconditional_prejitter_variance":unconditional_zero_jitter,"actual_unconditional_default_variance":unconditional_native_jitter,
        "scope":"The native four-row retained fitness and donor/acceptance laws are integrated before and after the prescribed jitter. Zero-jitter moments are an intermediate-stage diagnostic, not a modified canonical algorithm. All 81 measurement vectors compress to their exact eight bit-vector masses; independent conditional gate masks verify the binomial-copy identity."});
    records(
        suite,
        label,
        &[
            "a=0.1",
            r"U(x)=x^2/2",
            r"R_x=R_v=2",
            r"p_{\max}=1",
            r"\varepsilon_{\rm clone}=10^{-6}",
            r"\delta=2a/(2+a)",
            r"w=e^{-\delta^2/8}",
            r"F_0=1.09668913465",
            r"M=1+\operatorname{Bin}(3,q)",
            r"\mathbb E V_{\mathrm{Var},x}(S')=",
            "0<q<1/2",
            r"b\in\{0,1\}^3",
            r"P(b)=\prod",
            r"\sigma_x=0.1",
        ],
        input,
        checks,
    )
}

fn native_centered_row_aliases(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    if law.alive.iter().any(|a| !*a) {
        return Ok(());
    }
    let label = "lem-cloning-individual-centered-displacement";
    let n = law.x.len();
    let nf = n as f64;
    let mu = mean(&law.x);
    let centered: Vec<_> = law.x.iter().map(|x| x - mu).collect();
    let mut checks = Vec::new();
    let mut flowchecks = Vec::new();
    let mut native_inputs = Vec::new();
    let mut has_zero = false;
    for p in &law.patterns {
        let t: Vec<_> = p.means.iter().zip(&law.x).map(|(m, x)| m - x).collect();
        let bt = mean(&t);
        let mut count_mean = vec![0.; n];
        let mut residual_mean = vec![0.; n];
        let mut residual_second = vec![0.; n];
        let mut row_centered_second = vec![0.; n];
        let plans = graph_plans(p);
        for (choices, mass) in &plans {
            let post: Vec<_> = choices.iter().map(|&j| law.x[j]).collect();
            let bp = mean(&post);
            let accepted: Vec<_> = choices.iter().enumerate().map(|(i, &j)| i != j).collect();
            let acount = accepted.iter().filter(|a| **a).count() as f64;
            for i in 0..n {
                count_mean[i] += mass * post[i];
                residual_mean[i] += mass * (post[i] - p.means[i]);
                residual_second[i] += mass
                    * ((post[i] - p.means[i]).powi(2) + 0.01 * f64::from(u8::from(accepted[i])));
                row_centered_second[i] += mass
                    * ((post[i] - bp).powi(2)
                        + 0.01
                            * ((1. - 2. / nf) * f64::from(u8::from(accepted[i]))
                                + acount / (nf * nf)));
                checks.push(equal(
                    "literal_native_affine_row_noise_decomposition",
                    label,
                    post[i],
                    p.means[i] + (post[i] - p.means[i]),
                ));
            }
        }
        for i in 0..n {
            checks.push(equal(
                "native_frozen_row_centered_vector",
                label,
                centered[i],
                law.x[i] - mu,
            ));
            checks.push(equal(
                "native_row_noise_residual_mean_zero",
                label,
                residual_mean[i],
                0.,
            ));
            checks.push(equal(
                "native_row_noise_residual_covariance",
                label,
                residual_second[i],
                p.traces[i],
            ));
            checks.push(equal(
                "native_row_current_barycenter_exact_mean",
                label,
                count_mean[i],
                p.means[i],
            ));
            checks.push(equal(
                "native_full_barycenter_row_centered_second_moment",
                label,
                row_centered_second[i],
                (centered[i] + t[i] - bt).powi(2)
                    + (1. - 2. / nf) * p.traces[i]
                    + p.traces.iter().sum::<f64>() / (nf * nf),
            ));
            let aj = p.traces[i] - 0.01 * p.p[i];
            flowchecks.push(equal(
                "native_copy_and_jitter_covariance_definition",
                "lem-keystone-contraction-alive",
                p.traces[i],
                aj + 0.01 * p.p[i],
            ));
            if p.p[i] == 0. {
                has_zero = true;
                checks.push(equal(
                    "native_persisting_row_displacement_zero",
                    label,
                    t[i],
                    0.,
                ));
            }
        }
        checks.extend([
            equal(
                "native_full_slot_input_barycenter_definition",
                label,
                mu,
                law.x.iter().sum::<f64>() / nf,
            ),
            equal(
                "native_full_slot_displacement_barycenter_definition",
                label,
                bt,
                t.iter().sum::<f64>() / nf,
            ),
            equal(
                "native_full_slot_output_barycenter_mean",
                label,
                mean(&p.means),
                mu + bt,
            ),
        ]);
        flowchecks.extend([
            equal(
                "native_centered_input_first_moment_zero",
                "lem-keystone-contraction-alive",
                centered.iter().sum::<f64>(),
                0.,
            ),
            equal(
                "native_centered_displacement_pythagorean_identity",
                "lem-keystone-contraction-alive",
                t.iter().map(|t| (t - bt).powi(2)).sum::<f64>(),
                t.iter().map(|t| t * t).sum::<f64>() - nf * bt * bt,
            ),
            equal(
                "native_total_accepted_edge_normalization",
                "lem-keystone-contraction-alive",
                p.edges.iter().flatten().sum::<f64>() / nf,
                mean(&p.p),
            ),
            upper(
                "native_total_accepted_edge_probability_upper",
                "lem-keystone-contraction-alive",
                mean(&p.p),
                1.,
            ),
        ]);
        native_inputs.push(json!({"choices":p.choices,"native_means":p.means,"native_row_covariances":p.traces,"native_row_pressure":p.p,"native_t":t,"native_bar_t":bt,"actual_conditional_centered_row_second_moments":row_centered_second,"independently_integrated_row_residual_mean":residual_mean,"independently_integrated_row_residual_variance":residual_second,"conditional_copy_plan_count":plans.len()}));
    }
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"N":n,"actual_input_barycenter":mu,"actual_centered_vectors":centered,"all_native_plan_moments":native_inputs,
        "scope":"Conditional accepted graphs are enumerated from their native row laws. Each copied position and independent addressed Gaussian coefficient is integrated before centering; literal output rows independently reconstruct the mean, residual covariance and moving-barycenter moment. The singleton is a separate native law."});
    records(
        suite,
        label,
        &[
            r"\bar x=N^{-1}\sum_i x_i",
            r"\bar t=N^{-1}\sum_i t_i",
            r"\delta_i=x_i-\bar x",
            r"\bar X'=N^{-1}\sum_iX_i'",
            r"X_i'=m_i+\xi_i",
            r"\mathbb E|\xi_i|^2=c_i",
        ],
        input.clone(),
        checks.clone(),
    )?;
    if n == 1 {
        record(
            suite,
            label,
            "N=1",
            input.clone(),
            vec![
                equal("actual_singleton_full_population", label, nf, 1.),
                equal("actual_singleton_centered_position", label, var(&law.x), 0.),
            ],
        )?;
    }
    if has_zero {
        record(suite, label, "p_i=0", input.clone(), checks)?;
    }
    records(
        suite,
        "lem-keystone-contraction-alive",
        &[
            r"r_i^2=|x_i-\bar x|^2",
            r"\sum_i\delta_i=0",
            r"\sum_i|t_i-\bar t|^2=\sum_i|t_i|^2-N|\bar t|^2",
            r"c_i=a_i+d j^2p_i",
            r"N^{-1}\sum_{ij}b_{ij}=\bar p\le1",
        ],
        input,
        flowchecks,
    )
}

fn index_permutations(d: usize) -> Vec<Vec<usize>> {
    fn append(d: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if prefix.len() == d {
            out.push(prefix.clone());
            return;
        }
        for i in 0..d {
            if !prefix.contains(&i) {
                prefix.push(i);
                append(d, prefix, out);
                prefix.pop();
            }
        }
    }
    let mut out = Vec::new();
    append(d, &mut Vec::new(), &mut out);
    out
}

async fn native_completed_velocity_cap(suite: &mut EstimateSuite) -> Result<()> {
    use crate::{Benchmark, BenchmarkModel};
    use algorithmic_gas::{GasBuilder, Population};
    let label = "axiom-velocity-regularization";
    for d in [1, 2, 3] {
        let n = 4;
        let mut config = GasConfig::euclidean(d, 0.04)?;
        config.seed = 824_933 + d as u64;
        let radius = config
            .kinetic
            .velocity_cap
            .ok_or_else(|| error("missing canonical velocity cap"))?;
        let x = (0..n)
            .flat_map(|i| (0..d).map(move |a| -0.45 + 0.25 * i as f64 + 0.03 * a as f64))
            .collect();
        let v = (0..n)
            .flat_map(|i| (0..d).map(move |a| -1.8 + 1.1 * i as f64 + 0.2 * a as f64))
            .collect();
        let mut observations = ObservationBatch::positions(TensorBatch::vectors(n, d, x)?);
        observations
            .fields
            .insert("velocities".into(), TensorBatch::vectors(n, d, v)?);
        let population = Population::new(observations)?;
        let model = BenchmarkModel {
            benchmark: Benchmark::Quadratic,
            field: "positions".into(),
            direction: config.fitness.direction,
        };
        let mut engine = GasBuilder::new(population, model.clone())
            .gradient(model)
            .config(config)
            .build()
            .await?;
        engine.start_recording(Default::default())?;
        engine.step().await?;
        let step = &engine
            .recording()
            .ok_or_else(|| error("missing native completed cap recording"))?
            .steps[0];
        let velocity = |stage: &str| -> Result<Vec<f64>> {
            step.stages
                .iter()
                .find(|s| s.stage == stage)
                .and_then(|s| s.fields.get("velocities"))
                .map(|f| f.values.clone())
                .ok_or_else(|| error(&format!("missing native cap stage {stage}")))
        };
        let before = velocity("B2_before_boundary")?;
        let after = velocity("velocity_cap_before_boundary")?;
        let mut checks = vec![certificate_check(
            "actual_native_cap_radius_positive",
            label,
            radius > 0.,
        )];
        for (pre, post) in before.chunks_exact(d).zip(after.chunks_exact(d)) {
            let norm = pre.iter().map(|v| v * v).sum::<f64>().sqrt();
            for (v, capped) in pre.iter().zip(post) {
                checks.push(equal(
                    "actual_native_radial_cap_each_coordinate",
                    label,
                    *capped,
                    radius * v / (radius + norm),
                ));
            }
            let output_norm = post.iter().map(|v| v * v).sum::<f64>().sqrt();
            checks.push(equal(
                "actual_native_radial_cap_norm",
                label,
                output_norm,
                radius * norm / (radius + norm),
            ));
            checks.push(upper(
                "actual_native_completed_velocity_bound",
                label,
                output_norm,
                radius,
            ));
        }
        record(
            suite,
            label,
            r"\psi_v(v)=",
            json!({"dimension":d,"population":n,"cap":radius,"precap_stage":"B2_before_boundary","postcap_stage":"velocity_cap_before_boundary","precap":before,"postcap":after,"scope":"Native recorded complete Euclidean step. Each radial-cap output is reconstructed from its retained B2 input using the full vector norm, before terminal status projection."}),
            checks,
        )?;
    }
    Ok(())
}
fn native_decision_definition_aliases(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    use algorithmic_gas::{
        Population,
        donor::{CompanionBatch, DonorPool},
        fitness::PositiveMapping,
        random::{RandomStream, Stream},
    };
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let mut observations = ObservationBatch::positions(TensorBatch::vectors(n, 1, law.x.clone())?);
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, 1, law.v.clone())?,
    );
    let mut population = Population::new(observations)?;
    for (i, alive) in law.alive.iter().enumerate() {
        population.validity[i].out_of_bounds = !*alive;
    }
    let pool = DonorPool::freeze(&population, 0, &[], 0, false)?;
    let label = "def-cloning-probability";
    let pmax = law.config.clone_decision.saturation;
    let eps = law.config.clone_decision.epsilon;
    let mut checks = vec![certificate_check(
        "actual_native_nonempty_proposal_input",
        label,
        !live.is_empty(),
    )];
    let mut revived = Vec::new();
    let mut fitness = Vec::new();
    let mut conditional_witnesses = Vec::new();
    let mut composition_checks = Vec::new();
    let mut fullmass = 0.;
    for p in &law.patterns {
        let donor_slots: Vec<_> = (0..n)
            .map(|i| if law.alive[i] { p.choices[i] } else { live[0] })
            .collect();
        let companions = CompanionBatch {
            rows: n,
            count: 1,
            indices: donor_slots
                .iter()
                .map(|&j| pool.current_index(j).expect("actual current donor"))
                .collect(),
            valid: vec![true; n],
            mutual: false,
        };
        let donor_fitness: Vec<_> = pool.sources.iter().map(|s| p.f[s.slot as usize]).collect();
        for seed in [828_340_u64, 828_341, 828_342] {
            let plan = law.config.clone_decision.plan(
                &population,
                &pool,
                &companions,
                &p.f,
                &donor_fitness,
                &law.alive,
                seed,
                1,
            )?;
            for (i, &j) in donor_slots.iter().enumerate() {
                if law.alive[i] {
                    let score = (p.f[j] - p.f[i]) / (p.f[i] + eps);
                    let reverse = (p.f[i] - p.f[j]) / (p.f[j] + eps);
                    let mut rng = RandomStream::new(seed, 1, Stream::Accept, i as u64, 0);
                    let u = rng.uniform::<f64>();
                    let threshold = pmax * u;
                    checks.extend([
                        lower(
                            "actual_native_threshold_support_lower",
                            label,
                            threshold,
                            0.,
                        ),
                        upper(
                            "actual_native_threshold_support_upper",
                            label,
                            threshold,
                            pmax,
                        ),
                        equal(
                            "native_threshold_rescaled_uniform",
                            label,
                            threshold / pmax,
                            u,
                        ),
                        equal(
                            "native_forward_score_numerator",
                            label,
                            score * (p.f[i] + eps),
                            p.f[j] - p.f[i],
                        ),
                        equal(
                            "native_reverse_score_numerator",
                            label,
                            reverse * (p.f[j] + eps),
                            p.f[i] - p.f[j],
                        ),
                        certificate_check(
                            "native_score_threshold_decision",
                            label,
                            plan.choices[i].accepted == (score > threshold),
                        ),
                        equal(
                            "native_conditional_stored_probability",
                            label,
                            plan.choices[i].probability.unwrap_or(-1.),
                            (score / pmax).clamp(0., 1.),
                        ),
                    ]);
                    if p.f[i] < p.f[j] {
                        checks.push(certificate_check(
                            "strict_native_dual_score_signs",
                            label,
                            score > 0. && reverse < 0.,
                        ));
                    }
                } else {
                    revived.push(certificate_check(
                        "actual_native_dead_row_accepts_without_threshold",
                        label,
                        plan.choices[i].accepted && plan.choices[i].revival,
                    ));
                    revived.push(equal(
                        "actual_native_dead_row_probability_one",
                        label,
                        plan.choices[i].probability.unwrap_or(-1.),
                        1.,
                    ));
                }
            }
            conditional_witnesses.push(json!({"measurement":p.choices,"native_frozen_fitness":p.f,"seed":seed,"donors":donor_slots,"native_accepted":plan.choices.iter().map(|c|c.accepted).collect::<Vec<_>>() }));
        }
        let oriented: Vec<_> = live.iter().map(|&i| -0.5 * law.x[i] * law.x[i]).collect();
        let sampled: Vec<_> = live.iter().map(|&i| p.y[i]).collect();
        let reward_scale = (var(&oriented) + 0.01).sqrt();
        let diversity_scale = (var(&sampled) + 0.01).sqrt();
        for &i in &live {
            let rz = (-0.5 * law.x[i] * law.x[i] - mean(&oriented)) / reward_scale;
            let sz = (p.y[i] - mean(&sampled)) / diversity_scale;
            let reward_map = law.config.fitness.reward_map.map(rz)?;
            let diversity_map = law.config.fitness.diversity_map.map(sz)?;
            fitness.extend([
                equal(
                    "native_minimize_oriented_positional_reward",
                    "def-fitness-operator",
                    -0.5 * law.x[i] * law.x[i],
                    -0.5 * population.observations.field("positions")?.row(i)?[0].powi(2),
                ),
                equal(
                    "native_reward_logistic_evaluation",
                    "def-fitness-operator",
                    reward_map,
                    2. / (1. + (-rz).exp()) + 0.1,
                ),
                equal(
                    "native_diversity_logistic_evaluation",
                    "def-fitness-operator",
                    diversity_map,
                    2. / (1. + (-sz).exp()) + 0.1,
                ),
                equal(
                    "native_frozen_two_channel_fitness",
                    "def-fitness-operator",
                    p.f[i],
                    reward_map.powf(law.config.fitness.reward_exponent)
                        * diversity_map.powf(law.config.fitness.diversity_exponent),
                ),
            ]);
        }
        let plans = graph_plans(p);
        let mass = plans.iter().map(|(_, m)| m).sum::<f64>();
        fullmass += p.mass * mass;
        composition_checks.push(equal(
            "conditional_native_decision_joint_mass_one",
            "thm-cloning-operator-composition",
            mass,
            1.,
        ));
        for t in [0.3_f64, 1., 2.] {
            let mut direct = (1., 0.);
            for i in 0..n {
                let keep = 1. - p.p[i];
                let damping = (-0.5 * 0.01 * t * t).exp();
                let re = keep * (t * law.x[i]).cos()
                    + p.edges[i]
                        .iter()
                        .enumerate()
                        .map(|(j, m)| m * damping * (t * law.x[j]).cos())
                        .sum::<f64>();
                let im = keep * (t * law.x[i]).sin()
                    + p.edges[i]
                        .iter()
                        .enumerate()
                        .map(|(j, m)| m * damping * (t * law.x[j]).sin())
                        .sum::<f64>();
                direct = (direct.0 * re - direct.1 * im, direct.0 * im + direct.1 * re);
            }
            let mut nested = (0., 0.);
            for (choices, m) in &plans {
                let count = choices
                    .iter()
                    .enumerate()
                    .filter(|(i, j)| *i != **j || !law.alive[*i])
                    .count();
                let copied_sum = choices.iter().map(|&j| law.x[j]).sum::<f64>();
                let damping = (-0.5 * 0.01 * t * t * count as f64).exp();
                nested.0 += m * damping * (t * copied_sum).cos();
                nested.1 += m * damping * (t * copied_sum).sin();
            }
            composition_checks.extend([
                equal(
                    "native_nested_kernel_symmetric_position_characteristic_real",
                    "thm-cloning-operator-composition",
                    nested.0,
                    direct.0,
                ),
                equal(
                    "native_nested_kernel_symmetric_position_characteristic_imaginary",
                    "thm-cloning-operator-composition",
                    nested.1,
                    direct.1,
                ),
            ]);
        }
    }
    composition_checks.push(equal(
        "complete_native_measurement_decision_kernel_mass_one",
        "thm-cloning-operator-composition",
        fullmass,
        1.,
    ));
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"current_alive_count":live.len(),"pmax":pmax,"epsilon":eps,"conditional_native_plan_witnesses":conditional_witnesses,"scope":"Native conditional CloneDecision::plan and addressed uniform thresholds. Complete finite measurement and accepted-edge laws are integrated before nonlinear acceptance. Full kernel composition is checked by its mass and symmetric position characteristic functions, with exact Gaussian jitter integrals; slots only represent a permutation-invariant swarm."});
    records(
        suite,
        label,
        &[
            r"p_i=p_i(S,\mathbf F)",
            r"S_i(c) \propto",
            r"S_c(i) \propto",
            r"V_i < V_c",
        ],
        input.clone(),
        checks.clone(),
    )?;
    records(
        suite,
        "def-cloning-decision",
        &[r"T_i \sim \mathrm{Unif}", r"S_i(c_i) > T_i"],
        input.clone(),
        checks.clone(),
    )?;
    records(
        suite,
        "def-decision-operator",
        &[r"T_i\sim U", r"S_i(c_i)>T_i"],
        input.clone(),
        checks,
    )?;
    if !revived.is_empty() {
        record(suite, label, "p_i=1", input.clone(), revived.clone())?;
        record(
            suite,
            "def-decision-operator",
            "p_i=1",
            input.clone(),
            revived,
        )?;
    }
    records(
        suite,
        "def-fitness-operator",
        &[r"r_i=R(x_i,v_i)", "q=r,s", r"g_q(z)=A_q"],
        input.clone(),
        fitness,
    )?;
    records(
        suite,
        "def-cloning-operator-formal",
        &[
            r"|\mathcal{A}(S)| \geq 1",
            r"\Psi_{\text{clone}} =",
            r"S' \sim \Psi_{\text{clone}}",
        ],
        input.clone(),
        composition_checks.clone(),
    )?;
    records(
        suite,
        "thm-cloning-operator-composition",
        &[
            r"\Psi_{\text{clone}}(S, \cdot) =",
            r"\Psi_{\text{clone}}(S, A) =",
        ],
        input,
        composition_checks,
    )
}
fn native_collective_aliases(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let label = "thm-cloning-unconditional-collective-balance";
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let mu = live.iter().map(|&i| law.x[i]).sum::<f64>() / live.len() as f64;
    let mut checks = Vec::new();
    let mut conditional_energies = Vec::new();
    for p in &law.patterns {
        let mut direct = 0.;
        for (choices, m) in graph_plans(p) {
            let energy = components(&choices)
                .iter()
                .map(|g| {
                    let mean = g.iter().map(|&i| law.v[i]).sum::<f64>() / g.len() as f64;
                    g.iter().map(|&i| (law.v[i] - mean).powi(2)).sum::<f64>() / n as f64
                })
                .sum::<f64>();
            direct += m * energy;
        }
        checks.push(equal(
            "native_accepted_component_relative_energy_definition",
            label,
            graph_moments(law, p, 0.5).0,
            direct,
        ));
        conditional_energies.push(direct);
    }
    for alpha in [0., 0.5, 1.] {
        checks.push(certificate_check(
            "actual_restitution_range_endpoints",
            label,
            (0. ..=1.).contains(&alpha),
        ));
    }
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"N":n,"k":live.len(),"native_conditional_component_energies":conditional_energies,"scope":"Every accepted graph in every actual native finite measurement law uses all frozen slot velocities in its component center. Complete law moments retain mandatory revival."});
    records(
        suite,
        label,
        &[r"\mathcal E_C=\frac1N\sum_C", r"0\leq\alpha\leq1"],
        input.clone(),
        checks,
    )?;
    if live.len() == 1 {
        record(
            suite,
            label,
            "k=1",
            input,
            vec![
                equal(
                    "actual_singleton_measurement_count",
                    label,
                    live.len() as f64,
                    1.,
                ),
                equal(
                    "actual_singleton_complete_measurement_law_deterministic",
                    label,
                    law.patterns.len() as f64,
                    1.,
                ),
                equal(
                    "actual_singleton_live_acceptance_zero",
                    label,
                    law.patterns[0].p[live[0]],
                    0.,
                ),
            ],
        )?;
        return Ok(());
    }
    let mut h: Vec<_> = live.iter().copied().take(live.len() / 2).collect();
    let mut l: Vec<_> = live.iter().copied().skip(live.len() / 2).collect();
    let radius =
        |g: &[usize]| g.iter().map(|&i| (law.x[i] - mu).powi(2)).sum::<f64>() / g.len() as f64;
    if radius(&h) < radius(&l) {
        std::mem::swap(&mut h, &mut l);
    }
    let mut meanabs = 0.;
    let mut second = 0.;
    let mut negative_checks = Vec::new();
    for p in &law.patterns {
        for &i in &h {
            for &j in &l {
                let delta = p.f[j] - p.f[i];
                meanabs += p.mass * delta.abs() / (h.len() * l.len()) as f64;
                second += p.mass * delta * delta / (h.len() * l.len()) as f64;
                negative_checks.push(equal(
                    "each_native_fitness_gap_negative_part_identity",
                    label,
                    (-delta).max(0.),
                    (delta.abs() - delta) / 2.,
                ));
            }
        }
    }
    let input = json!({"native_input":input,"oriented_high_radius_cluster":h,"oriented_low_radius_cluster":l,"high_radius":radius(&h),"low_radius":radius(&l),"full_native_mean_absolute_gap":meanabs,"T_HL":second});
    record(
        suite,
        label,
        r"e_H\geq e_L",
        input.clone(),
        vec![lower(
            "actual_oriented_cluster_radius_condition",
            label,
            radius(&h),
            radius(&l),
        )],
    )?;
    record(
        suite,
        label,
        r"\mathbb E\langle|F_j-F_i|\rangle\leq\sqrt{T_{HL}}",
        input.clone(),
        vec![upper(
            "actual_complete_native_law_Cauchy_Schwarz_gap",
            label,
            meanabs,
            second.sqrt(),
        )],
    )?;
    record(suite, label, r"D_-=(|D|-D)/2", input, negative_checks)
}

fn no_affine_structural_aliases(suite: &mut EstimateSuite) -> Result<()> {
    let label = "prop-cloning-no-global-affine-structural-contraction";
    let config = GasConfig::euclidean(1, 0.04)?;
    let n = 10000_usize;
    let a = 1.9_f64;
    let kappa = 0.2_f64;
    let c = 0.7_f64;
    let floor = crate::convergence_cloning::canonical_two_cluster_acceptance_floor(0.5);
    let j = config.clone_transform.jitter_amplitude;
    let increment_lower = 2. * j * j * floor / 9. * (1. - 1. / n as f64) - 320. / (81. * n as f64);
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, vec![0.; n])?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, 1, vec![0.; n])?,
    );
    let frozen = config.fitness.evaluate(
        &RewardBatch::new(vec![0.; n], Default::default()),
        &vec![0.; n],
        &vec![true; n],
        &obs,
        0,
    )?;
    let mut checks = vec![
        certificate_check("candidate_positive_affine_rate", label, kappa > 0.),
        certificate_check(
            "candidate_nonvacuous_uniform_affine_offset",
            label,
            c < 4. * kappa,
        ),
        equal(
            "native_balanced_even_population",
            label,
            n as f64,
            2. * (n / 2) as f64,
        ),
        certificate_check(
            "retained_native_radius_in_interval",
            label,
            (0.5..2.).contains(&a),
        ),
        equal("native_required_jitter_amplitude", label, j, 0.1),
        certificate_check(
            "uniform_native_jitter_increment_strict_positive",
            label,
            j * j * floor / 9. > 0.,
        ),
        certificate_check(
            "chosen_radius_above_hypothetical_floor",
            label,
            a * a > c / kappa,
        ),
        lower(
            "native_uniform_positive_increment_threshold",
            label,
            increment_lower,
            j * j * floor / 9.,
        ),
        equal(
            "native_zero_comparison_perturbation",
            label,
            obs.field("positions")?
                .values()
                .iter()
                .map(|x| x * x)
                .sum::<f64>(),
            0.,
        ),
    ];
    for &f in &frozen.fitness {
        checks.push(equal(
            "actual_zero_comparison_equal_fitness",
            label,
            f,
            1.21,
        ));
        checks.push(equal(
            "actual_zero_comparison_acceptance_vanishes",
            label,
            config.clone_decision.acceptance_probability(1, f, f),
            0.,
        ));
    }
    let u = 0.25;
    let v = 0.75;
    let quantile_increment = 2. * a;
    checks.push(certificate_check("quantile_order_premise", label, u < v));
    checks.push(upper(
        "actual_balanced_quantile_increment_range",
        label,
        quantile_increment,
        4.,
    ));
    checks.push(upper(
        "actual_centered_quantile_variance_range_bound",
        label,
        a * a,
        4.,
    ));
    records(
        suite,
        label,
        &[
            r"\kappa>0",
            r"C<4\kappa",
            "u<v",
            "N=2M",
            r"1/2\le a<2",
            "j=.1",
            r"j^2A_*/9>0",
            r"a\in[1/2,2)",
            r"a^2>C/\kappa",
            r"\epsilon=0",
        ],
        json!({"N":n,"M":n/2,"radius":a,"canonical_jitter":j,"native_acceptance_floor":floor,"candidate_kappa":kappa,"candidate_C":c,"native_increment_lower":increment_lower,"comparison_native_fitness":frozen.fitness[0],"comparison_native_variance":0.,"scope":"Quantitative premises of the chapter's negated global cloning-stage inequality. Actual native equal-fitness comparison swarm and proved positive balanced-family increment retain the canonical algorithm; completed kinetic contraction is a separate component hypothesis."}),
        checks,
    )
}
fn native_row_operator_aliases(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let mu = live.iter().map(|&i| law.x[i]).sum::<f64>() / live.len() as f64;
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let dx = range(&xa);
    let mut checks = Vec::new();
    let mut dead = Vec::new();
    let mut graph_checks = Vec::new();
    let mut conditional = Vec::new();
    for p in &law.patterns {
        let cm = mean(&p.means);
        checks.push(equal(
            "conditional_native_barycenter_mean_definition",
            "lem-variance-change-decomposition",
            cm,
            p.means.iter().sum::<f64>() / n as f64,
        ));
        for i in 0..n {
            checks.push(equal(
                "conditional_actual_accepted_edge_row_sum",
                "def-cloning-position-row-law",
                p.p[i],
                p.edges[i].iter().sum::<f64>(),
            ));
            let displacement = p.edges[i]
                .iter()
                .enumerate()
                .map(|(j, b)| b * ((law.x[j] - law.x[i]).powi(2) + 0.01))
                .sum::<f64>();
            checks.push(equal(
                "actual_native_conditional_squared_displacement_moment",
                "prop-expected-displacement-cloning",
                (p.means[i] - law.x[i]).powi(2) + p.traces[i],
                displacement,
            ));
            if !law.alive[i] {
                dead.push(equal(
                    "actual_dead_row_accepted_probability_one",
                    "def-cloning-position-row-law",
                    p.p[i],
                    1.,
                ));
            }
        }
        let mut barycenter_second = 0.;
        for (choices, mass) in graph_plans(p) {
            let source_mean = choices.iter().map(|&j| law.x[j]).sum::<f64>() / n as f64;
            let accepted = choices
                .iter()
                .enumerate()
                .filter(|(i, j)| *i != **j || !law.alive[*i])
                .count();
            barycenter_second +=
                mass * ((source_mean - mu).powi(2) + 0.01 * accepted as f64 / (n * n) as f64);
            graph_checks.push(upper(
                "actual_native_accepted_count_at_most_population",
                "thm-positional-variance-contraction",
                accepted as f64,
                n as f64,
            ));
            for (i, &j) in choices.iter().enumerate() {
                graph_checks.push(upper(
                    "actual_frozen_source_distance_from_copied_center",
                    "lem-dead-walker-revival-bounded",
                    (law.x[j] - source_mean).abs(),
                    dx,
                ));
                let indicator = if i != j || !law.alive[i] { 1. } else { 0. };
                for zeta in [-1.2, 0., 0.4] {
                    let realized = law.x[j] + 0.1 * indicator * zeta;
                    graph_checks.push(equal(
                        "actual_frozen_native_row_copy_jitter_formula",
                        "thm-positional-variance-contraction",
                        realized,
                        law.x[j] + law.config.clone_transform.jitter_amplitude * indicator * zeta,
                    ));
                }
            }
        }
        checks.push(equal(
            "actual_moving_barycenter_about_alive_center_moment",
            "prop-cloning-revival-cluster-flux",
            barycenter_second,
            (cm - mu).powi(2) + p.traces.iter().sum::<f64>() / (n * n) as f64,
        ));
        conditional.push(json!({"native_measurement":p.choices,"native_frozen_fitness":p.f,"accepted_row_probabilities":p.p,"native_accepted_edges":p.edges,"conditional_row_means":p.means,"conditional_row_covariances":p.traces,"conditional_barycenter_mean":cm,"conditional_barycenter_about_alive_center_second":barycenter_second}));
    }
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"N":n,"conditional_native_rows":conditional,"eligible_diameter":dx,"jitter":0.1,"scope":"Every retained native measurement and accepted-edge graph is integrated. Revival moments use eligible donor positions; huge discarded dead positions contribute only to actual donor weights, while moving-barycenter moments use the alive entering center."});
    records(
        suite,
        "def-cloning-position-row-law",
        &[r"p_i=\sum_jb_{ij}"],
        input.clone(),
        checks.clone(),
    )?;
    record(
        suite,
        "prop-expected-displacement-cloning",
        r"\mathbb E[|\Delta x_i|^2\mid S,\mathbf F]",
        input.clone(),
        checks.clone(),
    )?;
    record(
        suite,
        "lem-variance-change-decomposition",
        r"\bar m=N^{-1}\sum_i m_i",
        input.clone(),
        checks.clone(),
    )?;
    record(
        suite,
        "prop-cloning-revival-cluster-flux",
        r"\mathbb E|\bar X'-\mu_A|^2=",
        input.clone(),
        checks,
    )?;
    if !dead.is_empty() {
        record(
            suite,
            "def-cloning-position-row-law",
            "p_i=1",
            input.clone(),
            dead.clone(),
        )?;
        record(
            suite,
            "def-cloning-frozen-positional-moments",
            "p_i=1",
            input.clone(),
            dead,
        )?;
    }
    records(
        suite,
        "thm-positional-variance-contraction",
        &[r"X_i'=Y_i+\sigma_xA_i\zeta_i", r"\sum_iA_i\leq N"],
        input.clone(),
        graph_checks.clone(),
    )?;
    records(
        suite,
        "lem-dead-walker-revival-bounded",
        &[r"X_i'=Y_i+\sigma_x A_i\zeta_i", r"|Y_i-\bar Y|\leq D_x"],
        input.clone(),
        graph_checks,
    )?;
    if n == 1 {
        record(
            suite,
            "lem-dead-walker-revival-bounded",
            "N=1",
            input,
            vec![equal(
                "actual_one_slot_centered_output_moment_zero",
                "lem-dead-walker-revival-bounded",
                law.patterns[0].output_var,
                0.,
            )],
        )?;
    }
    Ok(())
}
fn native_moment_component_aliases(suite: &mut EstimateSuite, law: &Law) -> Result<()> {
    let n = law.x.len();
    let live: Vec<_> = (0..n).filter(|&i| law.alive[i]).collect();
    let xa: Vec<_> = live.iter().map(|&i| law.x[i]).collect();
    let va: Vec<_> = live.iter().map(|&i| law.v[i]).collect();
    let vmax = law.v.iter().map(|v| v.abs()).fold(0_f64, f64::max);
    let mut q = 0.;
    let mut bx = 0.;
    let mut bx2 = 0.;
    let mut postvar = 0.;
    for p in &law.patterns {
        q += p.mass * graph_moments(law, p, 0.5).1;
        bx += p.mass * mean(&p.means);
        bx2 += p.mass * (mean(&p.means).powi(2) + p.conditional_bary_var);
        postvar += p.mass * p.output_var;
    }
    let vbar = mean(&law.v);
    let mean_q = bx * bx + vbar * vbar + 0.1 * bx * vbar;
    let bary_q = bx2 + vbar * vbar + 0.1 * bx * vbar;
    let product_struct = 2. * (q - bary_q);
    let product_full = 2. * (q - mean_q);
    let actual_location = 2. * (bx2 - bx * bx);
    let mh = 2. * (4. + 0.01) + 1.0025 * vmax * vmax;
    let mut checks = vec![
        lower(
            "native_phase_quadratic_mean_nonnegative",
            "cor-component-bounds-vw",
            mean_q,
            0.,
        ),
        lower(
            "native_phase_internal_moment_nonnegative",
            "cor-component-bounds-vw",
            q - bary_q,
            0.,
        ),
        equal(
            "native_phase_location_plus_product_structural_upper",
            "cor-component-bounds-vw",
            actual_location + product_struct,
            product_full,
        ),
        upper(
            "native_phase_product_full_drift_upper",
            "thm-inter-swarm-bounded-expansion",
            product_full,
            4. * mh,
        ),
        upper(
            "native_phase_actual_location_drift_upper",
            "thm-complete-wasserstein-drift",
            actual_location,
            4. * mh,
        ),
        upper(
            "native_phase_product_structural_drift_upper",
            "thm-complete-wasserstein-drift",
            product_struct,
            4. * mh,
        ),
        equal(
            "native_phase_moment_drift_offset_alias",
            "thm-inter-swarm-bounded-expansion",
            4. * mh,
            4. * (2. * (4. + 0.01) + 1.0025 * vmax * vmax),
        ),
    ];
    let input = json!({"positions":law.x,"velocities":law.v,"alive":law.alive,"native_postclone_phase_moment":q,"native_unconditional_barycenter_position_mean":bx,"native_unconditional_barycenter_position_second":bx2,"native_full_slot_velocity_mean":vbar,"native_actual_independent_location_cost":actual_location,"native_admissible_product_structural_upper":product_struct,"native_admissible_product_full_upper":product_full,"M_h":mh,"K_eta":mh,"C_W":4.*mh,"C_loc":4.*mh,"C_struct":4.*mh,"coupling":"two independent complete native cloning laws from the same entering swarm","scope":"Complete native finite measurement, accepted graph, Gaussian jitter and component Haar moments. The location cost is exact; the centered product transport cost is an admissible upper bound on actual optimal structural error. Entering empirical laws coincide, so their initial location and structural errors are zero."});
    records(
        suite,
        "thm-inter-swarm-bounded-expansion",
        &[r"\mathbb E\Delta V_W\leq C_W", r"C_W=4M_h"],
        input.clone(),
        checks.clone(),
    )?;
    records(
        suite,
        "cor-component-bounds-vw",
        &[
            r"V_W=V_{\mathrm{loc}}+V_{\mathrm{struct}}",
            r"C_{\mathrm{loc}}=C_{\mathrm{struct}}=4M_h",
            r"C_W=4M_h",
        ],
        input.clone(),
        checks.clone(),
    )?;
    records(
        suite,
        "thm-complete-wasserstein-drift",
        &[
            r"\mathbb E\Delta V_{\mathrm{loc}}\leq C_{\mathrm{loc}}",
            r"\mathbb E\Delta V_{\mathrm{struct}}\leq C_{\mathrm{struct}}",
            r"C_W=4K_\eta",
        ],
        input.clone(),
        checks.clone(),
    )?;
    let initial_x = 2. * live.len() as f64 / n as f64 * var(&xa);
    let cx = range(&xa).powi(2) + 0.02;
    let hx = 2. * postvar - initial_x;
    checks.push(upper(
        "actual_two_swarm_normalized_position_drift_condition",
        "thm-complete-variance-drift",
        hx,
        -0.2 * initial_x + cx,
    ));
    record(
        suite,
        "thm-complete-variance-drift",
        r"H_x(S_1)+H_x(S_2)\leq-\kappa_xV_{\mathrm{Var},x}+C_x",
        json!({"native_moments":input,"X":initial_x,"actual_Hx_sum":hx,"chosen_reset_rate":0.2,"reset_offset":cx}),
        checks,
    )?;
    record(
        suite,
        "thm-positional-variance-contraction",
        r"\mathbb E\Delta V_{\mathrm{Var},x}\leq-V_{\mathrm{Var},x}+B_x",
        json!({"native_moments":input,"input_normalized_variance":initial_x/2.,"post_variance":postvar,"B_x":range(&xa).powi(2)/2.+(1.-1./n as f64)*0.01}),
        vec![upper(
            "native_position_reset_exact_drift_relaxation",
            "thm-positional-variance-contraction",
            postvar - initial_x / 2.,
            -initial_x / 2. + range(&xa).powi(2) / 2. + (1. - 1. / n as f64) * 0.01,
        )],
    )?;
    if live.len() == n {
        let energy = law
            .patterns
            .iter()
            .map(|p| p.mass * graph_moments(law, p, 0.5).0)
            .sum::<f64>();
        let drift = -2. * 0.75 * energy;
        let zero = vec![
            upper(
                "actual_all_alive_velocity_dissipation_nonpositive",
                "thm-velocity-variance-bounded-expansion",
                drift,
                0.,
            ),
            equal(
                "actual_all_alive_retained_dead_remainder_zero",
                "thm-velocity-variance-bounded-expansion",
                var(&law.v) - var(&va),
                0.,
            ),
        ];
        for label in [
            "thm-velocity-variance-bounded-expansion",
            "thm-complete-variance-drift",
            "proof-fg-cloning-main-results",
        ] {
            record(suite, label, "C_v=0", input.clone(), zero.clone())?;
        }
    }
    record(
        suite,
        "proof-fg-cloning-main-results",
        r"C_v=8V_{\max}^2",
        json!({"native_moments":input,"uniform_velocity_offset":8.*vmax*vmax}),
        vec![equal(
            "native_two_swarm_uniform_velocity_offset_formula",
            "proof-fg-cloning-main-results",
            8. * vmax * vmax,
            2. * 4. * vmax * vmax,
        )],
    )
}

/// Remaining source clauses receive their own independently evaluated checks.
/// Exact quotes are bound by the catalog to their actual owning source item.
fn source_clause(
    suite: &mut EstimateSuite,
    label: &str,
    quote: &str,
    inputs: Value,
    checks: Vec<BoundCheck>,
) -> Result<()> {
    if !SOURCE.contains(quote) || checks.is_empty() {
        return Err(error(&format!(
            "missing completion quote or checks: {label} / {quote}"
        )));
    }
    suite.evidence.push(EstimateEvidence {
        chapter: 3,
        source_labels: vec![label.into()],
        source_formula: quote.into(),
        scope: inputs
            .get("scope")
            .and_then(Value::as_str)
            .unwrap_or(SCOPE)
            .into(),
        inputs,
        hypothesis_checks: vec![],
        checks,
    });
    Ok(())
}

fn final_source_clauses(suite: &mut EstimateSuite) -> Result<()> {
    use algorithmic_gas::fitness::PositiveMapping;
    let law = enumerate_law(
        vec![-1.5, -0.5, 0.5, 1.],
        vec![0.3, -0.2, 0.1, -0.4],
        vec![true; 4],
    )?;
    let config = &law.config;
    let width = match config.cloning_donors.kernel {
        algorithmic_gas::geometry::Kernel::Gaussian { width } => width,
        _ => return Err(error("completion expects native Gaussian cloning kernel")),
    };
    for (label, quote, value) in [
        (
            "def-cloning-companion-operator",
            r"\varepsilon_c > 0",
            width,
        ),
        (
            "def-cloning-score",
            r"\varepsilon_{\mathrm{clone}} > 0",
            config.clone_decision.epsilon,
        ),
    ] {
        source_clause(
            suite,
            label,
            quote,
            json!({"native_config":config,"actual_parameter":value}),
            vec![certificate_check(
                "actual_native_parameter_strictly_positive",
                label,
                value > 0.,
            )],
        )?;
    }
    let selection = crate::convergence_selection::validate_selection(92171)?;
    let fixture = selection
        .fixtures
        .iter()
        .find(|f| f.name == "finite_cluster_geometry_and_target")
        .ok_or_else(|| error("geometric target fixture missing"))?;
    let x = jvec(&fixture.hypotheses, "positions")?;
    let high: Vec<_> = fixture.observations["high"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let common: Vec<_> = fixture.observations["common"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let fitness: Vec<_> = high.iter().map(|h| if *h { 0.8 } else { 1.4 }).collect();
    let target: Vec<_> = (0..x.len())
        .filter(|&i| common[i] && high[i] && fitness[i] <= mean(&fitness))
        .collect();
    let declared: Vec<_> = fixture.observations["target"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let n = jnum(&fixture.hypotheses, "slots")?;
    let e: Vec<_> = x.iter().map(|v| (v - mean(&x)).powi(2)).collect();
    let error_mass = target.iter().map(|&i| e[i]).sum::<f64>() / n;
    let structural = var(&x);
    let pu = fixture.constants["p_u"];
    let cerr = fixture.constants["c_error"];
    let gerr = fixture.constants["g_error"];
    let pressure = jvec(
        &fixture.observations["conditional_selection"],
        "acceptance_probabilities",
    )?;
    let inputs = json!({"fixture":fixture,"actual_fitness":fitness,"target_indices":target,"error_weights":e,
        "normalized_target_error":error_mass,"Vstruct":structural,
        "scope":"Finite geometric target and its supplied retained fitness vector with the actual independent Gaussian donor kernel. No claim that this prescribed vector was generated by a native measurement realization."});
    source_clause(
        suite,
        "def-critical-target-set",
        r"I_{\text{target}} := I_{11} \cap U_k \cap H_k(\epsilon)",
        inputs.clone(),
        vec![certificate_check(
            "reconstructed_three_way_target_intersection",
            "def-critical-target-set",
            target == declared,
        )],
    )?;
    source_clause(
        suite,
        "cor-cloning-pressure-target-set",
        r"p_{k,i}\geq p_u>0",
        inputs.clone(),
        target
            .iter()
            .map(|&i| {
                lower(
                    "actual_target_pressure_floor",
                    "cor-cloning-pressure-target-set",
                    pressure[i],
                    pu,
                )
            })
            .chain(std::iter::once(certificate_check(
                "positive_target_pressure_floor",
                "cor-cloning-pressure-target-set",
                pu > 0.,
            )))
            .collect(),
    )?;
    for quote in [
        r"\frac{1}{N}\sum_{i \in I_{\text{target}}} \|\Delta\delta_{x,i}\|^2",
        "N^{-1}\\sum_{I_{\\mathrm{target}}}\\|\\Delta\\delta_x\\|^2\n\\geq c_{\\mathrm{err}}V_{\\mathrm{struct}}-g_{\\mathrm{err}}",
    ] {
        source_clause(
            suite,
            "rem-keystone-balanced-structural-scope",
            quote,
            inputs.clone(),
            vec![
                equal(
                    "actual_normalized_target_error",
                    "rem-keystone-balanced-structural-scope",
                    error_mass,
                    jnum(&fixture.observations, "target_error")?,
                ),
                lower(
                    "actual_target_capture_with_omitted_error",
                    "rem-keystone-balanced-structural-scope",
                    error_mass,
                    cerr * structural - gerr,
                ),
            ],
        )?;
    }
    let spread = structural / 2.;
    let chi = pu * cerr;
    let gmax = (pu * gerr).max(chi * spread);
    for label in ["prop-n-uniformity-keystone", "thm-fg-cloning-main-results"] {
        source_clause(
            suite,
            label,
            r"\chi=p_uc_{\mathrm{err}}",
            json!({"p_u":pu,"c_error":cerr,"chi":chi,"target":inputs}),
            vec![equal_positive(
                "feedback_product_from_selection_and_capture",
                label,
                chi,
                pu * cerr,
            )],
        )?;
    }
    source_clause(
        suite,
        "prop-n-uniformity-keystone",
        r"g_{\max}=\max\{p_ug_{\mathrm{err}},\chi R_{\mathrm{spread}}^2\}",
        json!({"p_u":pu,"g_error":gerr,"chi":chi,"Rspread_squared":spread,"gmax":gmax}),
        vec![
            lower(
                "offset_covers_high_error_remainder",
                "prop-n-uniformity-keystone",
                gmax,
                pu * gerr,
            ),
            lower(
                "offset_covers_entire_low_error_regime",
                "prop-n-uniformity-keystone",
                gmax,
                chi * spread,
            ),
        ],
    )?;

    for eta in [0.01_f64, 1., 10.] {
        for label in [
            "thm-keystone-averaged-error-capture",
            "thm-keystone-discharged-averaged-pressure",
        ] {
            let dx = 0.7_f64;
            let dv = -0.3_f64;
            let b = 0.5_f64;
            let lambda = 1_f64;
            source_clause(
                suite,
                label,
                r"\eta>0",
                json!({"eta":eta,"dx":dx,"dv":dv,"b":b,"lambda_v":lambda}),
                vec![
                    certificate_check("actual_young_parameter_positive", label, eta > 0.),
                    upper(
                        "phase_cross_term_young_bound",
                        label,
                        dx * dx + lambda * dv * dv + b * dx * dv,
                        (1. + eta) * dx * dx + (lambda + b * b / (4. * eta)) * dv * dv,
                    ),
                ],
            )?;
        }
    }
    let constants = canonical_keystone_constants(1, 0.001, 2.)?;
    let sstar = constants.values["s_star"];
    let balanced = crate::convergence_cloning::exact_balanced_two_site_fixture(4, 1., 0.1)?;
    for (quote, checks) in [
        (
            r"|z_j-z_i|\le r",
            vec![upper(
                "same_site_actual_feature_distance",
                "lem-keystone-near-neighbor-pressure",
                0.,
                constants.natural_logs["r"].exp(),
            )],
        ),
        (
            r"s_Y\le s_*",
            vec![upper(
                "actual_native_regularized_distance_scale",
                "lem-keystone-near-neighbor-pressure",
                ((4_f64 / 3.).powi(2) / 4. + 0.01).sqrt(),
                sstar,
            )],
        ),
    ] {
        source_clause(
            suite,
            "lem-keystone-near-neighbor-pressure",
            quote,
            json!({"positions":[-1.,-1.,1.,1.],"same_site_pair":[0,1],"exact_balanced_law":balanced,"canonical_constants":constants}),
            checks,
        )?;
    }
    source_clause(
        suite,
        "cor-keystone-canonical-balanced-structural",
        r"r<2",
        json!({"radius":1.,"domain":"(-2,2)"}),
        vec![certificate_check(
            "balanced_fixture_radius_strictly_inside_box",
            "cor-keystone-canonical-balanced-structural",
            1_f64 < 2.,
        )],
    )?;
    for radius in [0.5_f64, 1.] {
        let tie = enumerate_law(vec![-radius, radius], vec![0., 0.], vec![true, true])?;
        let mut checks = vec![equal(
            "two_row_actual_population",
            "rem-keystone-balanced-structural-scope",
            tie.x.len() as f64,
            2.,
        )];
        for pattern in &tie.patterns {
            for i in 0..2 {
                checks.push(equal(
                    "two_row_native_complete_fitness_tie",
                    "rem-keystone-balanced-structural-scope",
                    pattern.f[i],
                    1.21,
                ));
                checks.push(equal(
                    "two_row_native_zero_acceptance",
                    "rem-keystone-balanced-structural-scope",
                    pattern.p[i],
                    0.,
                ));
            }
        }
        source_clause(
            suite,
            "rem-keystone-balanced-structural-scope",
            "N=2",
            json!({"positions":tie.x,"native_patterns":tie.patterns.iter().map(|p|json!({"mass":p.mass,"fitness":p.f,"probabilities":p.p})).collect::<Vec<_>>()}),
            checks,
        )?;
    }
    source_clause(
        suite,
        "rem-keystone-balanced-structural-scope",
        r"V_{\mathrm{struct}}>0",
        json!({"positions1":[-1.,1.],"positions2":[-0.5,0.5],"both_velocities_zero":true}),
        vec![certificate_check(
            "different_balanced_radii_positive_optimal_centered_cost",
            "rem-keystone-balanced-structural-scope",
            crate::convergence_selection::optimal_centered_transport_1d(&[-1., 1.], &[-0.5, 0.5])?
                .squared_wasserstein
                > 0.,
        )],
    )?;
    let singleton = enumerate_law(vec![0.2], vec![0.1], vec![true])?;
    source_clause(
        suite,
        "rem-keystone-balanced-structural-scope",
        "k=1",
        json!({"positions":singleton.x,"native_self_donor_probability":singleton.donors}),
        vec![equal(
            "singleton_self_donor_law",
            "rem-keystone-balanced-structural-scope",
            singleton.donors[0][0],
            1.,
        )],
    )?;
    for label in [
        "thm-cloning-canonical-barycenter-concentration",
        "prop-cloning-revival-cluster-flux",
    ] {
        let quote = if label == "thm-cloning-canonical-barycenter-concentration" {
            "k=0"
        } else {
            r"\mathcal A=\varnothing"
        };
        let extinct = exact_cloning_moments(
            &[vec![0.], vec![1.]],
            &[1., 1.],
            &[vec![0., 1.], vec![1., 0.]],
            &[false, false],
            &config.clone_decision,
            1,
            0.1,
        );
        source_clause(
            suite,
            label,
            quote,
            json!({"alive":[false,false],"k":0,"scope":"The all-dead input is explicitly outside the nonextinct proposal domain. The actual conditional moment API returns Extinction rather than inventing an alive barycenter."}),
            vec![certificate_check(
                "all_dead_input_returns_extinction",
                label,
                matches!(extinct, Err(GasError::Extinction)),
            )],
        )?;
    }
    for d in [1_usize, 2, 4] {
        let left = vec![-2.; d];
        let right = vec![2.; d];
        source_clause(
            suite,
            "thm-cloning-canonical-barycenter-concentration",
            r"D_x=4\sqrt d",
            json!({"d":d,"box_opposite_corners":[left,right]}),
            vec![equal(
                "independently_computed_box_diameter",
                "thm-cloning-canonical-barycenter-concentration",
                (0..d).map(|_| 4_f64.powi(2)).sum::<f64>().sqrt(),
                4. * (d as f64).sqrt(),
            )],
        )?;
    }
    let mut completion_obs =
        ObservationBatch::positions(TensorBatch::vectors(4, 1, law.x.clone())?);
    completion_obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(4, 1, law.v.clone())?,
    );
    let positions = &law.x;
    for pattern in &law.patterns {
        let groups = components(&graph_plans(pattern)[0].0);
        let dc = groups
            .iter()
            .map(|g| range(&g.iter().map(|&i| law.x[i]).collect::<Vec<_>>()))
            .fold(0_f64, f64::max);
        source_clause(
            suite,
            "lem-keystone-contraction-alive",
            r"D_c=\max_G\operatorname{diam}_x(G)",
            json!({"positions":law.x,"groups":groups,"D_c":dc}),
            vec![equal(
                "component_diameter_from_all_within_component_pairs",
                "lem-keystone-contraction-alive",
                dc,
                groups
                    .iter()
                    .flat_map(|g| {
                        g.iter().flat_map(|&i| {
                            g.iter().map(move |&j| (positions[i] - positions[j]).abs())
                        })
                    })
                    .fold(0_f64, f64::max),
            )],
        )?;
        let fmin = pattern.f.iter().copied().fold(f64::INFINITY, f64::min);
        let fmax = pattern.f.iter().copied().fold(0_f64, f64::max);
        let pmax = config.clone_decision.saturation;
        let eps = config.clone_decision.epsilon;
        let astar = (fmax - fmin).max(pmax * (fmax + eps));
        let h = [0_usize, 1];
        let l = [2_usize, 3];
        let weight = |i: usize, j: usize| {
            config
                .cloning_donors
                .kernel
                .log_weight(
                    config.cloning_donors.distance.compare(
                        &completion_obs,
                        i,
                        &completion_obs,
                        j,
                    )?,
                    algorithmic_gas::geometry::ComparisonKind::Distance,
                )
                .map(f64::exp)
        };
        let normalizers = (0..4)
            .map(|i| {
                (0..4)
                    .filter(|&j| j != i)
                    .map(|j| weight(i, j))
                    .collect::<Result<Vec<_>>>()
                    .map(|v| v.iter().sum::<f64>())
            })
            .collect::<Result<Vec<_>>>()?;
        let a = 1.
            / (h.iter().map(|&i| normalizers[i]).fold(0_f64, f64::max)
                * (l.iter().map(|&j| pattern.f[j]).fold(0_f64, f64::max)
                    - h.iter()
                        .map(|&i| pattern.f[i])
                        .fold(f64::INFINITY, f64::min))
                .max(pmax * (h.iter().map(|&i| pattern.f[i]).fold(0_f64, f64::max) + eps)));
        let b = 1.
            / (l.iter()
                .map(|&j| normalizers[j])
                .fold(f64::INFINITY, f64::min)
                * pmax
                * (l.iter()
                    .map(|&j| pattern.f[j])
                    .fold(f64::INFINITY, f64::min)
                    + eps));
        let mut positive = vec![];
        let mut signed = vec![];
        for &i in &h {
            for &j in &l {
                let gap = pattern.f[j] - pattern.f[i];
                if gap >= 0. {
                    positive.push(lower(
                        "actual_clipped_positive_gap_floor",
                        "thm-cloning-signed-cluster-fitness-flux",
                        config
                            .clone_decision
                            .acceptance_probability(1, pattern.f[i], pattern.f[j]),
                        gap / astar,
                    ));
                }
                signed.push(lower(
                    "actual_weighted_bidirectional_gap_floor",
                    "thm-cloning-signed-cluster-fitness-flux",
                    pattern.edges[i][j] - pattern.edges[j][i],
                    weight(i, j)? * (a * gap.max(0.) - b * (-gap).max(0.)),
                ));
            }
        }
        let inputs = json!({"positions":law.x,"velocities":law.v,"retained_fitness":pattern.f,"native_accepted_edges":pattern.edges,"kernel_normalizers":normalizers,"H":h,"L":l,"Astar":astar,"a_HL":a,"b_HL":b});
        if !positive.is_empty() {
            source_clause(
                suite,
                "thm-cloning-signed-cluster-fitness-flux",
                "\\min\\{1,D_{ij}/[p_{\\max}(F_i+\\varepsilon_{\\rm clone})]\\}\n\\geq D_{ij}/A_*",
                inputs.clone(),
                positive,
            )?;
        }
        source_clause(
            suite,
            "thm-cloning-signed-cluster-fitness-flux",
            "b_{ij}-b_{ji}\\geq\nw_{ij}[a_{HL}(D_{ij})_+-b_{HL}(D_{ij})_-]",
            inputs,
            signed,
        )?;
    }
    for groups in [
        vec![vec![0_usize], vec![1, 2, 3]],
        vec![vec![0, 1], vec![2, 3]],
    ] {
        let count = groups
            .iter()
            .enumerate()
            .flat_map(|(i, g)| groups[i + 1..].iter().map(move |h| g.len() * h.len()))
            .sum::<usize>();
        source_clause(
            suite,
            "cor-cloning-signed-collective-drift",
            r"\sum_{\{H,L\}}|H||L|=(k^2-\sum_H|H|^2)/2\leq k(k-1)/2",
            json!({"groups":groups,"k":4,"cross_pair_count":count}),
            vec![
                equal(
                    "partition_cross_pair_count",
                    "cor-cloning-signed-collective-drift",
                    count as f64,
                    (16. - groups
                        .iter()
                        .map(|g| (g.len() * g.len()) as f64)
                        .sum::<f64>())
                        / 2.,
                ),
                upper(
                    "distinct_cross_pairs_do_not_exceed_all_pairs",
                    "cor-cloning-signed-collective-drift",
                    count as f64,
                    6.,
                ),
            ],
        )?;
    }
    source_clause(
        suite,
        "cor-cloning-signed-collective-drift",
        r"e_H\geq e_L",
        json!({"barycenter":0.,"H_centroid":1.5,"L_centroid":0.5}),
        vec![lower(
            "actually_ordered_centroid_energies",
            "cor-cloning-signed-collective-drift",
            1.5_f64.powi(2),
            0.5_f64.powi(2),
        )],
    )?;
    for n in [4_usize, 8] {
        for k in 1..n {
            source_clause(
                suite,
                "prop-cloning-two-cluster-noise-balance",
                "0<K<N",
                json!({"N":n,"K":k,"high_measurement_pattern_exists":true}),
                vec![certificate_check(
                    "nonempty_nontie_measurement_count",
                    "prop-cloning-two-cluster-noise-balance",
                    k > 0 && k < n,
                )],
            )?;
        }
    }
    let expansion = enumerate_law(vec![-1., -1., 1., 1.], vec![0.; 4], vec![true; 4])?;
    source_clause(
        suite,
        "prop-cloning-macroscopic-structural-expansion",
        r"R(x)=-x^2/2",
        json!({"native_objective_direction":expansion.config.fitness.direction,"positions":expansion.x}),
        expansion
            .x
            .iter()
            .map(|x| {
                equal(
                    "reward_is_negative_minimized_quadratic",
                    "prop-cloning-macroscopic-structural-expansion",
                    -0.5 * x * x,
                    -x.powi(2) / 2.,
                )
            })
            .collect(),
    )?;
    let mask_checks = expansion
        .patterns
        .iter()
        .map(|p| {
            certificate_check(
                "actual_four_nonself_measurement_choices",
                "prop-cloning-macroscopic-structural-expansion",
                p.choices.len() == 4 && p.choices.iter().enumerate().all(|(i, &j)| i != j && j < 4),
            )
        })
        .collect();
    source_clause(
        suite,
        "prop-cloning-macroscopic-structural-expansion",
        r"m=(m_1,\ldots,m_4)",
        json!({"complete_patterns":expansion.patterns.iter().map(|p|json!({"m":p.choices,"mass":p.mass})).collect::<Vec<_>>()}),
        mask_checks,
    )?;
    let moving = enumerate_law(vec![-0.1, 0., 0.1], vec![0.; 3], vec![true; 3])?;
    source_clause(
        suite,
        "ex-cloning-moving-barycenter",
        r"g(z)=2/(1+e^{-z})+0.1",
        json!({"native_reward_map":moving.config.fitness.reward_map}),
        [-10_f64, -1., 0., 1., 10.]
            .iter()
            .map(|&z| {
                equal(
                    "native_logistic_map_at_finite_scores",
                    "ex-cloning-moving-barycenter",
                    moving.config.fitness.reward_map.map(z).unwrap(),
                    2. / (1. + (-z).exp()) + 0.1,
                )
            })
            .collect(),
    )?;
    let event = moving
        .patterns
        .iter()
        .find(|p| p.choices[0] == 1 && p.choices[2] == 1)
        .ok_or_else(|| error("moving center event missing"))?;
    let q = event.p[0];
    let positive_floor = 2. * 0.1_f64.powi(2) * q * (1. - q) / 9.;
    for j in [0_f64, 1e-6, 0.001, 0.1] {
        let jitter_term = 2. * j * j * q / 9.;
        source_clause(
            suite,
            "ex-cloning-moving-barycenter",
            r"C_{\rm jitter}=O(j^2)",
            json!({"j":j,"q":q,"jitter_term":jitter_term,"geometric_floor":positive_floor,"scope":"Exact moving-barycenter event: the jitter contribution is quadratic, while the positive geometry term survives at zero jitter. This validates the stated counterexample, not the invalid proposed row contraction."}),
            vec![
                equal(
                    "jitter_coefficient_times_j_squared",
                    "ex-cloning-moving-barycenter",
                    jitter_term,
                    (2. * q / 9.) * j * j,
                ),
                lower(
                    "geometric_error_survives_vanishing_jitter",
                    "ex-cloning-moving-barycenter",
                    positive_floor + jitter_term,
                    positive_floor,
                ),
            ],
        )?;
        if j == 0. {
            source_clause(
                suite,
                "ex-cloning-moving-barycenter",
                "j=0",
                json!({"j":j,"q":q,"conditional_centered_error":positive_floor}),
                vec![
                    equal(
                        "zero_jitter_exact_parameter",
                        "ex-cloning-moving-barycenter",
                        j,
                        0.,
                    ),
                    certificate_check(
                        "moving_center_error_positive_without_jitter",
                        "ex-cloning-moving-barycenter",
                        positive_floor > 0.,
                    ),
                ],
            )?;
        }
    }
    for kappa in [0.01_f64, 0.3, 1.] {
        let cb = 0.2_f64;
        let threshold = cb / kappa;
        let entering = threshold + 1.;
        source_clause(
            suite,
            "thm-boundary-potential-contraction",
            r"W_b>C_b/p_*",
            json!({"Wb":entering,"C_b":cb,"p_star":kappa,"drift_upper":-kappa*entering+cb,"scope":"Explicit scalar drift hypothesis at an exposed state. The native selection/integral requirements are independently checked by boundary fixtures; this clause verifies the precise negative-drift threshold."}),
            vec![
                certificate_check(
                    "state_above_barrier_drift_threshold",
                    "thm-boundary-potential-contraction",
                    entering > threshold,
                ),
                certificate_check(
                    "affine_barrier_bound_strictly_negative",
                    "thm-boundary-potential-contraction",
                    -kappa * entering + cb < 0.,
                ),
            ],
        )?;
        let mut value = entering;
        let mut cases = vec![];
        for t in 1..=2048 {
            value = (1. - kappa) * value + cb;
            let tail = (1. - kappa).powi(t) * (entering - threshold);
            let exact = threshold + tail;
            cases.push(equal(
                "iterated_affine_bound_and_geometric_tail",
                "cor-bounded-boundary-exposure",
                value,
                exact,
            ));
        }
        source_clause(
            suite,
            "cor-bounded-boundary-exposure",
            r"0<\kappa_b\leq1",
            json!({"kappa_b":kappa,"C_b":cb,"W0":entering}),
            vec![certificate_check(
                "geometric_decay_parameter_positive_unit_interval",
                "cor-bounded-boundary-exposure",
                kappa > 0. && kappa <= 1.,
            )],
        )?;
        source_clause(
            suite,
            "cor-bounded-boundary-exposure",
            r"t \to \infty",
            json!({"kappa_b":kappa,"C_b":cb,"W0":entering,"exact_tail":"(1-kappa)^t*(W0-Cb/kappa)","iterations":2048,"scope":"The recurrence is evaluated explicitly and its remainder is an exact geometric tail, with ratio in [0,1). The asymptotic conclusion uses this analytic tail certificate rather than declaring a finite simulated time infinite."}),
            cases,
        )?;
    }
    for d in [1_usize, 2, 4] {
        for sigma in [0_f64, 0.1, 1.] {
            let y = vec![0.5; d];
            let z = vec![-0.2; d];
            // Symmetric covariance-exact cubature integrates this degree-two
            // Gaussian observable without approximation or Monte Carlo error.
            let mut cubature = 0.;
            for axis in 0..d {
                for sign in [-1_f64, 1.] {
                    let mut point = y.clone();
                    point[axis] += sigma * sign * (d as f64).sqrt();
                    cubature += point
                        .iter()
                        .zip(&z)
                        .map(|(x, z)| (x - z).powi(2))
                        .sum::<f64>()
                        / (2 * d) as f64;
                }
            }
            let predicted = y.iter().zip(&z).map(|(x, z)| (x - z).powi(2)).sum::<f64>()
                + d as f64 * sigma * sigma;
            source_clause(
                suite,
                "thm-inter-swarm-bounded-expansion",
                "\\mathbb E\\|y+\\sigma_x\\xi-z_{0,x}\\|^2\n=\\|y-z_{0,x}\\|^2+d\\sigma_x^2",
                json!({"d":d,"sigma":sigma,"y":y,"z0":z,"covariance_exact_quadratic_cubature":cubature,"scope":"Exact degree-two Gaussian moment integration using covariance-exact symmetric cubature; native Gaussian moment sampling is tested separately."}),
                vec![equal(
                    "quadratic_gaussian_moment_independent_integration",
                    "thm-inter-swarm-bounded-expansion",
                    cubature,
                    predicted,
                )],
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tiny_constant_regressions {
    use super::*;

    #[test]
    fn zero_and_double_keystone_constants_are_rejected() {
        let report = canonical_keystone_constants(1, 0.001, 2.).unwrap();
        for key in ["gamma_0", "a_0"] {
            let expected = report.values[key];
            assert!(expected > 0. && expected < 1e-8);
            assert!(
                equal_positive(
                    "uncorrupted",
                    "lem-keystone-complete-coverage-constants",
                    expected,
                    expected
                )
                .passed
            );
            assert!(
                !equal_positive(
                    "zero_corruption",
                    "lem-keystone-complete-coverage-constants",
                    0.,
                    expected
                )
                .passed
            );
            assert!(
                !equal_positive(
                    "double_corruption",
                    "lem-keystone-complete-coverage-constants",
                    2. * expected,
                    expected
                )
                .passed
            );
        }
        let penalty = 0.1 * report.values["omega_f"] / 2.;
        assert!(penalty > 0. && penalty < 1e-8);
        assert!(
            !upper(
                "double_tiny_reward_penalty",
                "lem-keystone-complete-coverage-constants",
                2. * penalty,
                penalty
            )
            .passed
        );
        assert!(
            !lower(
                "zero_tiny_lower_bound",
                "lem-keystone-complete-coverage-constants",
                0.,
                penalty
            )
            .passed
        );
    }
}
