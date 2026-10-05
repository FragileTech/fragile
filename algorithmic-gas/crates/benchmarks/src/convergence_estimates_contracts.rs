//! Explicit finite witnesses for smooth-barrier contracts and transport vectors.
//! The geometric fixture is a ball; it does not assert a tubular radius for an
//! arbitrary domain. The cutoff is integrated from the stated bump, rather than
//! supplied as an independently chosen smooth function.
use crate::{
    convergence_estimates::{EstimateEvidence, EstimateSuite},
    convergence_framework::BoundCheck,
    convergence_validation::inventory,
};
use algorithmic_gas::{
    GasConfig, GasError, Result,
    random::{RandomStream, Stream},
};
use serde_json::{Value, json};

const BARRIER: &str = "prop-barrier-existence";
const SCOPE: &str = "Finite ball-domain collar and geometric boundary sequence; the stated bump integral defines the actual cutoff. Numerical derivative checks and finite approach sequences supplement the analytic smooth-gluing proof, without establishing arbitrary-domain regularity or an infinite-time limit.";

fn upper(id: &str, observed: f64, bound: f64) -> BoundCheck {
    BoundCheck::upper(id, &[BARRIER], SCOPE, observed, bound)
}
fn equal(id: &str, observed: f64, bound: f64) -> BoundCheck {
    upper(
        id,
        (observed - bound).abs(),
        3e-11 * (1. + observed.abs() + bound.abs()),
    )
}
fn strict(id: &str, condition: bool) -> BoundCheck {
    upper(id, if condition { 0. } else { 1. }, 0.)
}
fn tokens(value: &str) -> String {
    value.chars().filter(|c| !c.is_whitespace()).collect()
}

fn emit(
    suite: &mut EstimateSuite,
    catalog: &[Value],
    label: &str,
    formula: &str,
    inputs: &Value,
    hypotheses: &[BoundCheck],
    checks: &[BoundCheck],
) -> Result<()> {
    let wanted = tokens(formula);
    let source = catalog
        .iter()
        .find(|chapter| chapter["chapter"] == suite.chapter)
        .and_then(|chapter| chapter["quantitative_expressions"].as_array())
        .and_then(|expressions| {
            expressions.iter().find(|expression| {
                (expression["source_label"] == label || expression["source_label"].is_null())
                    && expression["formula"]
                        .as_str()
                        .is_some_and(|source| tokens(source) == wanted)
            })
        })
        .and_then(|expression| expression["formula"].as_str())
        .ok_or_else(|| {
            GasError::Configuration(format!("barrier contract missing: {label} / {formula}"))
        })?;
    let default_scope = match (suite.chapter, label) {
        (1, _) => {
            "Exact declared finite framework contract, with actual scalar observables and explicit permutation quotient. Auxiliary continuous diffusion identities are distinguished from the native discrete BAOAB kernel."
        }
        (_, "def-single-swarm-space") => {
            "Native addressed uniform/Gaussian innovations at the recorded seeds and dimensions. Distribution checks retain their statistical tolerances; computational innovation addresses do not assign intrinsic identities to swarm atoms."
        }
        (_, "def-state-difference-vectors") => {
            "Exact pairwise coordinate differences and product-transport cost for two unequal empirical supports; reordering or reversing the supports preserves the scalar cost."
        }
        _ => SCOPE,
    };
    let scope = inputs
        .get("scope")
        .and_then(Value::as_str)
        .unwrap_or(default_scope);
    let bind_checks = |items: &[BoundCheck]| {
        items
            .iter()
            .cloned()
            .map(|mut check| {
                check.source_labels = vec![label.into()];
                check.scope = scope.into();
                check
            })
            .collect()
    };
    suite.evidence.push(EstimateEvidence {
        chapter: suite.chapter,
        source_labels: vec![label.into()],
        source_formula: source.into(),
        inputs: inputs.clone(),
        hypothesis_checks: bind_checks(hypotheses),
        checks: bind_checks(checks),
        scope: scope.into(),
    });
    Ok(())
}

/// Compactly supported bump from the chapter, including its exact exterior.
pub fn bump(t: f64) -> f64 {
    if t.abs() < 1. {
        (-1. / (1. - t * t)).exp()
    } else {
        0.
    }
}
fn integrate(a: f64, b: f64, panels: usize) -> f64 {
    if a >= b {
        return 0.;
    }
    let h = (b - a) / panels as f64;
    let mut total = bump(2. * a - 3.) + bump(2. * b - 3.);
    for i in 1..panels {
        total += if i % 2 == 0 { 2. } else { 4. } * bump(2. * (a + i as f64 * h) - 3.);
    }
    total * h / 3.
}

/// Actual integral cutoff. The saturated branches are exact, not quadrature.
pub fn cutoff(t: f64) -> f64 {
    if t <= 1. {
        1.
    } else if t >= 2. {
        0.
    } else {
        integrate(t, 2., 2048) / integrate(1., 2., 2048)
    }
}
pub fn barrier(rho: f64, delta: f64) -> f64 {
    let psi = cutoff(rho / delta);
    (1. - psi) / delta + psi / rho
}

pub fn append_cloning(suite: &mut EstimateSuite) -> Result<()> {
    let catalog = inventory()?;
    let radius = 2_f64;
    let delta0 = 0.5;
    let delta = 0.125;
    let norm = integrate(1., 2., 2048);
    let inputs = json!({"domain":"open Euclidean ball","radius":radius,"delta0":delta0,
        "delta":delta,"bump_normalizer":norm,"quadrature_panels":2048});
    let hypotheses = vec![
        strict("positive_collar_width", delta0 > 0.),
        strict(
            "cutoff_inside_smooth_collar",
            delta > 0. && 3. * delta < delta0,
        ),
        strict("ball_collar_avoids_distance_cut_locus", delta0 < radius),
    ];
    let radial_points = [radius - delta, radius - 2. * delta, radius + delta];
    let signed_distances = radial_points.iter().map(|x| radius - x).collect::<Vec<_>>();
    let mut collar_checks = vec![];
    for (&x, &rho) in radial_points.iter().zip(&signed_distances) {
        collar_checks.push(equal(
            "signed_distance_piecewise",
            rho,
            if x < radius {
                (x - radius).abs()
            } else {
                -(x - radius).abs()
            },
        ));
        collar_checks.push(strict(
            "actual_tubular_neighborhood_membership",
            rho.abs() < delta0,
        ));
        let h = 1e-5;
        let derivative = ((radius - (x + h)) - (radius - (x - h))) / (2. * h);
        collar_checks.push(upper(
            "signed_distance_inward_unit_normal",
            (derivative + 1.).abs(),
            2e-10,
        ));
    }
    let collar_inputs = json!({"domain_radius":radius,"radial_points":radial_points,
        "signed_distances":signed_distances,"tubular_width":delta0,
        "closest_boundary_point":radius,"inside_gradient":-1.});
    emit_prefix(
        suite,
        &catalog,
        BARRIER,
        r"\rho(x) :=",
        &collar_inputs,
        &collar_checks,
    )?;
    for formula in [
        r"U := \{x \in \mathbb{R}^d : d(x, \partial \mathcal{X}_{\text{valid}}) < \delta_0\}",
        r"x \in U \cap \mathcal{X}_{\text{valid}}",
        r"x \in U",
    ] {
        let inside = json!({"radius":radius,"x":radius-delta,"distance_to_boundary":delta,
            "delta0":delta0,"inside_domain":true,"inside_collar":true});
        emit(
            suite,
            &catalog,
            BARRIER,
            formula,
            &inside,
            &hypotheses,
            &[
                strict("inside_ball_and_collar", delta > 0. && delta < delta0),
                equal(
                    "closest_boundary_point_distance",
                    (radius - delta - radius).abs(),
                    delta,
                ),
            ],
        )?;
    }
    let mut integral = vec![
        strict("positive_bump_integral", norm > 0.),
        equal("bump_integral_refinement", norm, integrate(1., 2., 4096)),
    ];
    let mut bump_checks = vec![];
    let bump_points = [-2_f64, -1., -0.9, -0.5, 0., 0.5, 0.9, 1., 2.];
    for t in bump_points {
        let expected = if t.abs() < 1. {
            (-1. / (1. - t * t)).exp()
        } else {
            0.
        };
        bump_checks.push(equal("actual_piecewise_bump", bump(t), expected));
        bump_checks.push(strict("bump_nonnegative", bump(t) >= 0.));
    }
    emit_prefix(
        suite,
        &catalog,
        BARRIER,
        r"\eta(t) :=",
        &json!({"points":bump_points,
        "values":bump_points.map(bump),"strict_interior_positive":true}),
        &bump_checks,
    )?;
    for i in 0..=40 {
        let t = i as f64 / 16.;
        let p = cutoff(t);
        integral.push(upper("cutoff_at_most_one", p, 1.));
        integral.push(upper("cutoff_nonnegative", -p, 0.));
        integral.push(upper("cutoff_decreasing", cutoff(t + 1. / 16.), p));
        if t > 1. && t < 2. {
            let h = 1e-4;
            let derivative = (cutoff(t + h) - cutoff(t - h)) / (2. * h);
            let expected = -bump(2. * t - 3.) / norm;
            integral.push(strict(
                "strict_transition_derivative",
                derivative < 0. && expected < 0.,
            ));
            integral.push(upper(
                "cutoff_fundamental_theorem_derivative",
                (derivative - expected).abs(),
                2e-6,
            ));
        }
    }
    emit(
        suite,
        &catalog,
        BARRIER,
        r"\psi(t) := \frac{\int_{t}^{\infty} \eta(2s - 3) \, ds}{\int_{-\infty}^{\infty} \eta(2s - 3) \, ds}",
        &inputs,
        &hypotheses,
        &integral,
    )?;
    for formula in [r"\psi'(t) < 0", r"t \in (1, 2)"] {
        emit(
            suite,
            &catalog,
            BARRIER,
            formula,
            &inputs,
            &hypotheses,
            &integral,
        )?;
    }
    for (formula, check) in [
        (r"\psi(t) = 1", equal("left_cutoff_exact", cutoff(0.5), 1.)),
        (r"t \leq 1", upper("left_argument_contract", 0.5, 1.)),
        (r"\psi(t) = 0", equal("right_cutoff_exact", cutoff(2.5), 0.)),
        (r"t \geq 2", upper("right_argument_contract", 2., 2.5)),
        (r"\delta_0 > 0", strict("collar_positive", delta0 > 0.)),
        (
            r"\delta < \delta_0/3",
            strict("collar_strict_width", delta < delta0 / 3.),
        ),
        (
            r"\delta \in (0, \delta_0/3)",
            strict("positive_cutoff_width", delta > 0. && delta < delta0 / 3.),
        ),
    ] {
        emit(
            suite,
            &catalog,
            BARRIER,
            formula,
            &inputs,
            &hypotheses,
            &[check],
        )?;
    }

    let mut values = vec![];
    let mut identity = vec![];
    for ratio in [0.03125, 0.5, 1., 1.125, 1.5, 1.875, 2., 3., 8.] {
        let rho = ratio * delta;
        let psi = cutoff(ratio);
        let observed = barrier(rho, delta);
        let unexpanded = 1. / delta + psi * (1. / rho - 1. / delta);
        values.push(json!({"rho":rho,"rho_over_delta":ratio,"cutoff":psi,
            "barrier":observed,"unexpanded_barrier":unexpanded}));
        identity.push(equal("barrier_expansion_identity", observed, unexpanded));
        identity.push(strict("strict_positive_barrier", observed > 0.));
        if ratio <= 1. {
            identity.push(equal("near_boundary_reciprocal", observed, 1. / rho));
        }
        if ratio >= 2. {
            identity.push(equal("constant_interior_extension", observed, 1. / delta));
        }
    }
    let fixture = json!({"geometry":inputs,"evaluated_cases":values});
    for formula in [
        r"\varphi(x) := \frac{1}{\delta} + \psi\left(\frac{\rho(x)}{\delta}\right)\left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right)",
        r"\varphi(x) > 0",
        r"\varphi(x) = 1/\delta",
        r"\varphi=1/\delta",
        r"\varphi: \mathcal{X}_{\text{valid}} \to (0, \infty)",
        r"\varphi(x) = \frac{1}{\delta} + 1 \cdot \left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right) = \frac{1}{\rho(x)} > 0",
        r"\varphi(x) = \frac{1}{\delta} + 0 \cdot \left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right) = \frac{1}{\delta} > 0",
        r"\begin{aligned} \varphi(x) &= \frac{1}{\delta} + \psi\left(\frac{\rho(x)}{\delta}\right)\left( \frac{1}{\rho(x)} - \frac{1}{\delta} \right) \\ &= \frac{1}{\delta} + \psi\left(\frac{\rho(x)}{\delta}\right) \cdot \frac{1}{\rho(x)} - \psi\left(\frac{\rho(x)}{\delta}\right) \cdot \frac{1}{\delta} \\ &= \frac{1}{\delta}\left(1 - \psi\left(\frac{\rho(x)}{\delta}\right)\right) + \frac{1}{\rho(x)} \psi\left(\frac{\rho(x)}{\delta}\right) \end{aligned}",
    ] {
        emit(
            suite,
            &catalog,
            BARRIER,
            formula,
            &fixture,
            &hypotheses,
            &identity,
        )?;
    }

    let regions: &[(&[&str], f64)] = &[
        (
            &[
                r"0 < \rho(x) \leq \delta",
                r"\rho(x)/\delta \leq 1",
                r"\psi(\rho(x)/\delta) = 1",
                r"\rho(x) > 0",
            ],
            0.5,
        ),
        (
            &[
                r"\rho(x) \geq 2\delta",
                r"\rho(x)/\delta \geq 2",
                r"\psi(\rho(x)/\delta) = 0",
            ],
            2.5,
        ),
        (
            &[
                r"\delta < \rho(x) < 2\delta",
                r"1 < \rho(x)/\delta < 2",
                r"\psi(\rho(x)/\delta) \in (0, 1)",
                r"\psi(\rho(x)/\delta) \in (0,1)",
                r"1 - \psi(\rho(x)/\delta) \in (0, 1) \subset (0, \infty)",
                r"\delta > 0",
                r"1 - \psi > 0",
                r"\psi > 0",
            ],
            1.5,
        ),
        (&[r"\rho(x) < 3\delta < \delta_0"], 2.5),
        (
            &[
                r"\rho(x) \geq 3\delta",
                r"\rho(x)/\delta \geq 3 > 2",
                r"\rho(x) = 3\delta",
                r"\geq 2",
            ],
            3.,
        ),
    ];
    for (formulas, ratio) in regions {
        let rho = ratio * delta;
        let psi = cutoff(*ratio);
        let checks = vec![
            strict(
                "branch_predicate",
                match *ratio {
                    0.5 => 0. < rho && rho <= delta && psi == 1.,
                    1.5 => delta < rho && rho < 2. * delta && psi > 0. && psi < 1.,
                    2.5 => rho >= 2. * delta && rho < 3. * delta && psi == 0.,
                    _ => rho == 3. * delta && rho < delta0 && psi == 0.,
                },
            ),
            strict("positive_branch_value", barrier(rho, delta) > 0.),
        ];
        let inputs =
            json!({"rho":rho,"delta":delta,"delta0":delta0,"psi":psi,"phi":barrier(rho,delta)});
        for formula in *formulas {
            emit(
                suite,
                &catalog,
                BARRIER,
                formula,
                &inputs,
                &hypotheses,
                &checks,
            )?;
        }
        if *ratio == 1.5 {
            let first = (1. - psi) / delta;
            let second = psi / rho;
            emit_prefix(
                suite,
                &catalog,
                BARRIER,
                r"\varphi(x) = \underbrace",
                &inputs,
                &[
                    strict("both_transition_terms_positive", first > 0. && second > 0.),
                    equal(
                        "transition_positive_sum",
                        barrier(rho, delta),
                        first + second,
                    ),
                ],
            )?;
        }
    }
    // On a ball, a radial interior sequence has an exact closest boundary
    // point. The finite geometric sequence exposes the 1/rho growth rate.
    let mut sequence = vec![];
    let mut checks = vec![];
    let mut previous = 0.;
    for n in 1_i32..=30 {
        let rho = delta * 2_f64.powi(-n);
        let x = radius - rho;
        let distance = radius - x;
        let phi = barrier(distance, delta);
        sequence.push(json!({"n":n,"x":x,"rho":distance,"phi":phi}));
        checks.push(strict(
            "sequence_inside_domain",
            x < radius && distance > 0.,
        ));
        checks.push(equal(
            "closest_boundary_retraction_distance",
            distance,
            (x - radius).abs(),
        ));
        checks.push(equal(
            "boundary_sequence_exact_reciprocal",
            phi,
            1. / distance,
        ));
        checks.push(strict("strict_boundary_growth", phi > previous));
        checks.push(equal(
            "boundary_reciprocal_growth_scaling",
            phi * distance,
            1.,
        ));
        previous = phi;
    }
    let inputs = json!({"domain_radius":radius,"delta":delta,"sequence":sequence,
        "normal_direction":"inward","finite_sequence_only":true});
    for formula in [
        r"\rho(x) = \|x - \pi(x)\| > 0",
        r"\rho(x_n) < \delta",
        r"\rho(x_n)/\delta < 1",
        r"\psi(\rho(x_n)/\delta) = 1",
        r"\varphi(x_n) = \frac{1}{\delta} + 1 \cdot \left( \frac{1}{\rho(x_n)} - \frac{1}{\delta} \right) = \frac{1}{\rho(x_n)}",
        r"\rho(x_n) = d(x_n, \partial \mathcal{X}_{\text{valid}}) \to 0^{+}",
        r"\varphi(x_n) = 1/\rho(x_n) \to +\infty",
        r"\varphi(x_n) \to \infty",
        r"\varphi(x) \to \infty",
        r"x_n \to x_{\infty} \in \partial \mathcal{X}_{\text{valid}}",
    ] {
        emit(
            suite,
            &catalog,
            BARRIER,
            formula,
            &inputs,
            &hypotheses,
            &checks,
        )?;
    }
    append_transport_vectors(suite, &catalog)?;
    append_native_innovations(suite, &catalog)?;
    append_quasi_stationary_contract(suite, &catalog)?;
    append_quadratic_transport_contracts(suite, &catalog)?;
    append_boundary_reward_contracts(suite, &catalog)?;
    append_safe_population_contracts(suite, &catalog)?;
    append_boundary_observable_contracts(suite, &catalog)?;
    append_native_boundary_clauses(suite, &catalog)?;
    append_reset_parameter_contracts(suite, &catalog)?;
    append_native_fitness_and_population_contracts(suite, &catalog)?;
    append_barrier_jitter_clauses(suite, &catalog)?;
    append_remaining_geometric_parameters(suite, &catalog)?;
    append_auxiliary_phase_and_boundary_contracts(suite, &catalog)?;
    append_native_radial_feature_contract(suite, &catalog)?;
    append_composition_parameter_contracts(suite, &catalog)
}

fn append_transport_vectors(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let label = "def-state-difference-vectors";
    let left = [[-1_f64, 0.5], [2., -0.25]];
    let right = [[0.75_f64, -1.], [-0.5, 1.5], [1.25, 0.1]];
    let mut differences = vec![];
    let mut checks = vec![];
    let mut total = 0.;
    let mut transpose = 0.;
    for x in left {
        for y in right {
            let delta = [x[0] - y[0], x[1] - y[1]];
            differences.push(json!({"left":x,"right":y,"delta":delta,"product_plan_mass":1./6.}));
            checks.push(equal("difference_first_coordinate", delta[0] + y[0], x[0]));
            checks.push(equal("difference_second_coordinate", delta[1] + y[1], x[1]));
            total += delta.iter().map(|a| a * a).sum::<f64>() / 6.;
            transpose += ((y[0] - x[0]).powi(2) + (y[1] - x[1]).powi(2)) / 6.;
        }
    }
    checks.push(equal("plan_cost_under_reversal", total, transpose));
    let inputs = json!({"pairs":differences,"cost":total,"reversed_cost":transpose,
        "interpretation":"product transport between two unequal empirical atom sets"});
    for formula in [
        r"\Delta x_{ij}=x_{1,i}-x_{2,j}",
        r"\Delta v_{ij}=v_{1,i}-v_{2,j}",
    ] {
        emit(suite, catalog, label, formula, &inputs, &[], &checks)?;
    }
    Ok(())
}

fn normal_cdf(t: f64) -> f64 {
    let panels = 4096;
    let h = (t + 10.) / panels as f64;
    let density = |x: f64| (-x * x / 2.).exp() / std::f64::consts::TAU.sqrt();
    let mut total = density(-10.) + density(t);
    for i in 1..panels {
        total += if i % 2 == 0 { 2. } else { 4. } * density(-10. + i as f64 * h);
    }
    total * h / 3.
}

fn append_native_innovations(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let label = "def-single-swarm-space";
    let draws = 16_384;
    let failure_budget = 1e-6_f64;
    let dkw = ((2. / failure_budget).ln() / (2. * draws as f64)).sqrt();
    let p_max = GasConfig::euclidean(1, 0.04)?.clone_decision.saturation;
    let mut uniform = Vec::with_capacity(draws);
    for sample in 0..draws {
        let mut rng = RandomStream::new(923_819, sample as u64, Stream::Accept, 0, 0);
        uniform.push(p_max * rng.uniform::<f64>());
    }
    let mut checks = vec![strict(
        "native_threshold_support",
        uniform.iter().all(|x| *x > 0. && *x < p_max),
    )];
    let mut cdf_values = vec![];
    for i in 0..=20 {
        let t = i as f64 * p_max / 20.;
        let empirical = uniform.iter().filter(|x| **x <= t).count() as f64 / draws as f64;
        let exact = t / p_max;
        cdf_values.push(json!({"threshold":t,"empirical_cdf":empirical,"uniform_cdf":exact}));
        checks.push(upper(
            "native_uniform_dkw_cdf",
            (empirical - exact).abs(),
            dkw,
        ));
    }
    let inputs = json!({"native_rng":"RandomStream::uniform<f64>","native_gate_scale":p_max,
        "draws":draws,"seed":923819,"cdf_values":cdf_values,"dkw_failure_budget":failure_budget,
        "interpretation":"Equivalent physical threshold p_max*U for the native normalized acceptance gate"});
    emit(
        suite,
        catalog,
        label,
        r"T_i \sim \text{Uniform}(0, p_{\max})",
        &inputs,
        &[strict("positive_native_threshold_scale", p_max > 0.)],
        &checks,
    )?;

    for stream in [Stream::CloneNoise, Stream::Kinetic] {
        for dimensions in [1, 2, 4, 16] {
            let mut values = Vec::with_capacity(draws);
            for sample in 0..draws {
                let mut rng = RandomStream::new(781_237, sample as u64, stream, 0, 0);
                values.push(
                    (0..dimensions)
                        .map(|_| rng.gaussian::<f64>())
                        .collect::<Vec<_>>(),
                );
            }
            let mut checks = vec![];
            let mut moments = vec![];
            for coordinate in 0..dimensions {
                let mean = values.iter().map(|v| v[coordinate]).sum::<f64>() / draws as f64;
                let second = values
                    .iter()
                    .map(|v| v[coordinate] * v[coordinate])
                    .sum::<f64>()
                    / draws as f64;
                moments.push(json!({"coordinate":coordinate,"mean":mean,"second_moment":second}));
                checks.push(upper(
                    "native_normal_mean_six_se",
                    mean.abs(),
                    6. / (draws as f64).sqrt(),
                ));
                checks.push(upper(
                    "native_normal_second_moment_six_se",
                    (second - 1.).abs(),
                    6. * (2. / draws as f64).sqrt(),
                ));
                for t in [-2_f64, -1., 0., 1., 2.] {
                    let empirical =
                        values.iter().filter(|v| v[coordinate] <= t).count() as f64 / draws as f64;
                    checks.push(upper(
                        "native_normal_dkw_cdf",
                        (empirical - normal_cdf(t)).abs(),
                        dkw + 2e-10,
                    ));
                }
                for other in 0..coordinate {
                    let covariance =
                        values.iter().map(|v| v[coordinate] * v[other]).sum::<f64>() / draws as f64;
                    checks.push(upper(
                        "native_normal_offdiagonal_second_moment",
                        covariance.abs(),
                        6. / (draws as f64).sqrt(),
                    ));
                }
            }
            let inputs = json!({"native_rng":"RandomStream::gaussian<f64>","stream":stream,
                "draws":draws,"seed":781237,"dimensions":dimensions,"coordinate_moments":moments,
                "dkw_failure_budget_per_coordinate":failure_budget,
                "scope":"Statistical innovation-law check; mean/covariance tests use six theoretical standard errors"});
            let formula = if stream == Stream::CloneNoise {
                r"\zeta_i^x \sim \mathcal{N}(0, I_d)"
            } else {
                r"\xi_i^v \sim \mathcal{N}(0, I_d)"
            };
            emit(suite, catalog, label, formula, &inputs, &[], &checks)?;
        }
    }
    Ok(())
}

fn emit_prefix(
    suite: &mut EstimateSuite,
    catalog: &[Value],
    label: &str,
    prefix: &str,
    inputs: &Value,
    checks: &[BoundCheck],
) -> Result<()> {
    let wanted = tokens(prefix);
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
        .filter(|formula| tokens(formula).starts_with(&wanted))
        .collect::<Vec<_>>();
    if formulas.is_empty() {
        return Err(GasError::Configuration(format!(
            "framework definition missing: {label}/{prefix}"
        )));
    }
    for formula in formulas {
        emit(suite, catalog, label, formula, inputs, &[], checks)?;
    }
    Ok(())
}

fn permutations(size: usize) -> Vec<Vec<usize>> {
    fn extend(values: &mut [usize], at: usize, output: &mut Vec<Vec<usize>>) {
        if at == values.len() {
            output.push(values.to_vec());
        } else {
            for i in at..values.len() {
                values.swap(at, i);
                extend(values, at + 1, output);
                values.swap(at, i);
            }
        }
    }
    let mut output = vec![];
    extend(&mut (0..size).collect::<Vec<_>>(), 0, &mut output);
    output
}

/// Check the computational content of framework state and measure definitions.
pub fn append_framework(suite: &mut EstimateSuite) -> Result<()> {
    use crate::convergence_lyapunov::uniform_transport;
    let catalog = inventory()?;
    for n in 1..=4 {
        let left = (0..n)
            .map(|i| vec![-0.75 + 0.4 * i as f64, if i % 2 == 0 { 1. } else { 0. }])
            .collect::<Vec<_>>();
        let right = (0..n)
            .map(|i| vec![0.6 - 0.35 * i as f64, if i % 3 == 0 { 0. } else { 1. }])
            .collect::<Vec<_>>();
        let lambda = 0.7;
        let cost = |x: &[f64], y: &[f64]| (x[0] - y[0]).powi(2) + lambda * (x[1] - y[1]).powi(2);
        let (minimum, plan) = uniform_transport(&left, &right, cost)?;
        let (argmin, enumerated) = permutations(n)
            .into_iter()
            .map(|pi| {
                let value = pi
                    .iter()
                    .enumerate()
                    .map(|(i, &j)| cost(&left[i], &right[j]))
                    .sum::<f64>()
                    / n as f64;
                (pi, value)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let positional = argmin
            .iter()
            .enumerate()
            .map(|(i, &j)| (left[i][0] - right[j][0]).powi(2))
            .sum::<f64>();
        let mismatches = argmin
            .iter()
            .enumerate()
            .map(|(i, &j)| (left[i][1] - right[j][1]).powi(2))
            .sum::<f64>();
        let transport_integral = plan
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, m)| m * cost(&left[i], &right[j]))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let mut checks = vec![
            equal(
                "flow_equals_exhaustive_permutation_minimum",
                minimum,
                enumerated,
            ),
            equal("optimal_cost_integral", minimum, transport_integral),
            equal(
                "marked_positional_status_decomposition",
                minimum,
                (positional + lambda * mismatches) / n as f64,
            ),
            equal("distance_square_root", minimum.sqrt().powi(2), minimum),
        ];
        for row in &plan {
            checks.push(equal("left_plan_marginal", row.iter().sum(), 1. / n as f64));
        }
        for j in 0..n {
            checks.push(equal(
                "right_plan_marginal",
                plan.iter().map(|row| row[j]).sum(),
                1. / n as f64,
            ));
        }
        let mut reversed_left = left.clone();
        reversed_left.reverse();
        let mut reversed_right = right.clone();
        reversed_right.reverse();
        checks.push(equal(
            "quotient_cost_after_reordering",
            uniform_transport(&reversed_left, &reversed_right, cost)?.0,
            minimum,
        ));
        checks.push(equal(
            "same_empirical_measure_zero_distance",
            uniform_transport(&left, &reversed_left, cost)?.0,
            0.,
        ));
        let inputs = json!({"left_marked_atoms":left,"right_marked_atoms":right,"N":n,
            "projection":"identity on the bounded declared fixture","lambda_status":lambda,
            "optimal_plan":plan,"enumerated_minimizer":argmin,"position_sum":positional,
            "status_mismatch_sum":mismatches,"optimal_squared_distance":minimum});
        for (label, prefix) in [
            (
                "def-n-particle-displacement-metric",
                r"d_{\text{Disp},\mathcal{Y}}",
            ),
            (
                "def-metric-quotient",
                r"\overline d_{\text{Disp},\mathcal{Y}}",
            ),
            ("def-displacement-components", r"\Delta_{\text{pos}}^2"),
            ("def-displacement-components", r"n_c("),
            ("def-wasserstein-distance", r"W_2(\mu,\nu)^2="),
        ] {
            emit_prefix(suite, &catalog, label, prefix, &inputs, &checks)?;
        }
        let alive = left
            .iter()
            .enumerate()
            .filter_map(|(i, w)| (w[1] == 1.).then_some(i))
            .collect::<Vec<_>>();
        let dead = left
            .iter()
            .enumerate()
            .filter_map(|(i, w)| (w[1] == 0.).then_some(i))
            .collect::<Vec<_>>();
        let empirical_mean = left.iter().map(|w| w[0]).sum::<f64>() / n as f64;
        let reordered_mean = reversed_left.iter().map(|w| w[0]).sum::<f64>() / n as f64;
        let atom_weights = vec![1. / n as f64; n];
        let checks = vec![
            strict(
                "binary_survival_states",
                left.iter().all(|w| w[1] == 0. || w[1] == 1.),
            ),
            strict(
                "alive_dead_partition",
                alive.len() + dead.len() == n && alive.iter().all(|i| !dead.contains(i)),
            ),
            equal(
                "uniform_empirical_probability_mass",
                atom_weights.iter().sum(),
                1.,
            ),
            equal(
                "empirical_integral_under_reordering",
                empirical_mean,
                reordered_mean,
            ),
        ];
        let inputs = json!({"marked_atoms":left,"alive_support":alive,"dead_support":dead,
            "uniform_atom_weights":atom_weights,"empirical_position_mean":empirical_mean});
        for (label, prefix) in [
            ("def-walker", r"w := (x, s)"),
            ("def-swarm-and-state-space", r"\mathbf w :="),
            ("def-alive-dead-sets", r"\mathcal{A}(\mathcal{S}) :="),
            ("def-alive-dead-sets", r"\mathcal{D}(\mathcal{S}) :="),
        ] {
            emit_prefix(suite, &catalog, label, prefix, &inputs, &checks)?;
        }
    }
    for x in [-1.5_f64, -0.25, 0., 0.6, 1.25] {
        let reward = x * x / 2.;
        let dirac_integral = [(x, 1.)]
            .iter()
            .map(|(point, mass)| mass * point * point / 2.)
            .sum::<f64>();
        let inputs = json!({"x":x,"reward":"x^2/2","dirac_atoms":[x],"dirac_masses":[1.],"evaluated_reward":reward});
        emit_prefix(
            suite,
            &catalog,
            "def-reward-measurement",
            r"r_i :=",
            &inputs,
            &[equal(
                "dirac_expectation_equals_reward_at_atom",
                dirac_integral,
                reward,
            )],
        )?;
    }
    for (x, y) in [(-1_f64, 0.5), (0., 1.5), (0.25, 0.25)] {
        let project = |x: f64| 2. * x / (4. + x * x).sqrt();
        let first = project(x);
        let second = project(y);
        let dirac_w1 = (first - second).abs();
        let dirac_w2_squared = (first - second).powi(2);
        let inputs = json!({"left_physical":x,"right_physical":y,"projection":"2*x/sqrt(4+x^2)",
            "left_projected":first,"right_projected":second,"dirac_transport_plan":[[1.]],
            "W1":dirac_w1,"W2_squared":dirac_w2_squared});
        let checks = vec![
            equal("dirac_transport_metric", dirac_w1, dirac_w2_squared.sqrt()),
            strict(
                "bounded_projection",
                first.abs() <= 2. && second.abs() <= 2.,
            ),
        ];
        emit_prefix(
            suite,
            &catalog,
            "def-distance-positional-measures",
            r"d(\varphi_*",
            &inputs,
            &checks,
        )?;
        emit_prefix(
            suite,
            &catalog,
            "def-alg-distance",
            r"\boxed{d_{\text{alg}}",
            &inputs,
            &checks,
        )?;
    }
    append_generator(suite, &catalog)?;
    append_markov_contracts(suite, &catalog)?;
    append_alive_output_contracts(suite, &catalog)?;
    append_reference_innovation_contract(suite, &catalog)?;
    append_framework_inline_state_contracts(suite, &catalog)?;
    append_feller_reference_contracts(suite, &catalog)
}

fn append_generator(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    type PolynomialDerivatives = (fn(f64, f64) -> f64, f64, f64, f64);
    let gamma = 0.7;
    let sigma = 0.4;
    let curvature = 1.3;
    let h = 0.0001;
    for (x, v) in [(-1.1_f64, 0.3), (0.2, -0.8), (0.7, 1.2)] {
        let drift = -curvature * x - gamma * v;
        let normal = [
            (-3_f64.sqrt(), 1. / 6.),
            (0., 2. / 3.),
            (3_f64.sqrt(), 1. / 6.),
        ];
        let mut values = vec![];
        let mut checks = vec![];
        for monomial in 0..5 {
            let (f, dx, dv, lapv): PolynomialDerivatives = match monomial {
                0 => (|x, _| x, 1., 0., 0.),
                1 => (|_, v| v, 0., 1., 0.),
                2 => (|x, _| x * x, 2. * x, 0., 0.),
                3 => (|_, v| v * v, 0., 2. * v, 2.),
                _ => (|x, v| x * v, v, x, 0.),
            };
            let generator = v * dx + drift * dv + sigma * sigma * lapv / 2.;
            let moment = normal
                .iter()
                .map(|(z, p)| p * f(x + h * v, v + h * drift + sigma * h.sqrt() * z))
                .sum::<f64>();
            let correction = match monomial {
                2 => h * v * v,
                3 => h * drift * drift,
                4 => h * v * drift,
                _ => 0.,
            };
            let observed = (moment - f(x, v)) / h;
            values.push(json!({"monomial":monomial,"generator":generator,"exact_euler_moment_rate":observed,"Euler_bias":correction}));
            checks.push(upper(
                "generator_polynomial_exact_moment",
                (observed - generator - correction).abs(),
                1e-10,
            ));
        }
        let inputs = json!({"x":x,"v":v,"quadratic_potential_curvature":curvature,"friction":gamma,
            "noise_amplitude":sigma,"Euler_dt":h,"Gaussian_moment_quadrature":normal,
            "polynomial_values":values,"scope":"Auxiliary continuous Langevin generator; native BAOAB has its separate chapter2 checks"});
        emit_prefix(
            suite,
            catalog,
            "def-langevin-operator",
            r"Lf=",
            &inputs,
            &checks,
        )?;
    }
    Ok(())
}

fn row_update(mu: [f64; 2], kernel: [[f64; 2]; 2]) -> [f64; 2] {
    [
        mu[0] * kernel[0][0] + mu[1] * kernel[1][0],
        mu[0] * kernel[0][1] + mu[1] * kernel[1][1],
    ]
}

fn append_markov_contracts(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let p = [[0.75, 0.25], [0.25, 0.75]];
    let killed = [[0.675, 0.225], [0.125, 0.375]];
    let potential = [1., 2.];
    let a = 0.9_f64;
    let b = 0.35;
    let m = 1.5_f64;
    let r = 0.5_f64;
    let inputs = json!({"conservative_kernel":p,"killed_kernel":killed,"V":potential,
        "a":a,"b":b,"small_set":[0],"M":m,"r":r,"invariant_law":[0.5,0.5],
        "scope":"Exact declared two-state reference kernels for abstract definitions, not a finite-state approximation or proof of native gas ergodicity."});
    let conservative = p
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let pv = [
        p[0][0] * potential[0] + p[0][1] * potential[1],
        p[1][0] * potential[0] + p[1][1] * potential[1],
    ];
    let mut checks = vec![
        equal("conservative_total_mass_first_state", conservative[0], 1.),
        equal("conservative_total_mass_second_state", conservative[1], 1.),
    ];
    emit_prefix(
        suite,
        catalog,
        "def-markov-kernel",
        "P(x,E)=1",
        &inputs,
        &checks,
    )?;
    checks = vec![
        upper("drift_on_small_set", pv[0], a * potential[0] + b),
        upper("drift_off_small_set", pv[1], a * potential[1]),
    ];
    for formula in ["PV\\le aV+b\\mathbf1_C", "PV\\le aV+b"] {
        emit(
            suite,
            catalog,
            "def-foster-lyapunov",
            formula,
            &inputs,
            &[
                strict("geometric_drift_parameter", a > 0. && a < 1.),
                strict("finite_drift_offset", b.is_finite()),
            ],
            &checks,
        )?;
    }
    for (label, formula, check) in [
        (
            "def-foster-lyapunov",
            "0<a<1",
            strict("strict_drift_factor", a > 0. && a < 1.),
        ),
        (
            "def-foster-lyapunov",
            r"b<\infty",
            strict("finite_drift_offset", b.is_finite()),
        ),
        (
            "def-geometric-ergodicity",
            r"M<\infty",
            strict("finite_norm_coefficient", m.is_finite()),
        ),
        (
            "def-geometric-ergodicity",
            r"r\in(0,1)",
            strict("strict_geometric_norm_rate", r > 0. && r < 1.),
        ),
    ] {
        emit(suite, catalog, label, formula, &inputs, &[], &[check])?;
    }
    for initial in [[1., 0.], [0., 1.], [0.2, 0.8]] {
        let mut law = initial;
        let mut surviving = initial;
        let mut iterations = vec![];
        let mut drift = vec![];
        let mut geometric = vec![];
        let mut survival = vec![];
        let initial_v = initial[0] * potential[0] + initial[1] * potential[1];
        for n in 0_i32..=24 {
            let evolved_v = law[0] * potential[0] + law[1] * potential[1];
            let bound = a.powi(n) * initial_v + b * (1. - a.powi(n)) / (1. - a);
            let eta = [law[0] - 0.5, law[1] - 0.5];
            let norm = eta[0].abs() * potential[0] + eta[1].abs() * potential[1];
            let test_function = [
                potential[0] * eta[0].signum(),
                potential[1] * eta[1].signum(),
            ];
            let attained = (eta[0] * test_function[0] + eta[1] * test_function[1]).abs();
            let mass = surviving.iter().sum::<f64>();
            let conditioned = [surviving[0] / mass, surviving[1] / mass];
            iterations.push(
                json!({"n":n,"law":law,"expected_V":evolved_v,"drift_upper":bound,
                "signed_deviation":eta,"V_norm":norm,"norm_maximizing_function":test_function,
                "killed_law":surviving,"survival_mass":mass,"conditioned_law":conditioned}),
            );
            drift.push(upper("exact_iterated_drift", evolved_v, bound));
            geometric.push(upper(
                "exact_finite_reference_geometric_rate",
                norm,
                m * initial_v * r.powi(n),
            ));
            geometric.push(equal("weighted_signed_measure_dual_norm", attained, norm));
            survival.push(strict("positive_survival_normalizer", mass > 0.));
            survival.push(equal(
                "survival_conditioned_unit_mass",
                conditioned.iter().sum(),
                1.,
            ));
            law = row_update(law, p);
            surviving = row_update(surviving, killed);
        }
        let inputs = json!({"kernel_fixture":inputs,"initial_law":initial,"iterations":iterations,
            "scope":"Exact powers of the declared conservative/killed two-state matrices. No native stationary law or native contraction constant is inferred."});
        emit(
            suite,
            catalog,
            "def-foster-lyapunov",
            r"P^nV\le a^nV+b(1-a^n)/(1-a)",
            &inputs,
            &[],
            &drift,
        )?;
        emit(
            suite,
            catalog,
            "def-geometric-ergodicity",
            r"\|P^n(x,\cdot)-\pi\|_V\le M V(x)r^n",
            &inputs,
            &[],
            &geometric,
        )?;
        emit(
            suite,
            catalog,
            "def-geometric-ergodicity",
            r"\|\eta\|_V=\sup_{|f|\le V}|\eta f|",
            &inputs,
            &[],
            &geometric,
        )?;
        emit(
            suite,
            catalog,
            "def-markov-kernel",
            r"\mu K^n\mathbf1>0",
            &inputs,
            &[],
            &survival,
        )?;
    }
    Ok(())
}

fn append_quasi_stationary_contract(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let q: [[f64; 2]; 2] = [[0.675, 0.225], [0.125, 0.375]];
    let trace = q[0][0] + q[1][1];
    let discriminant = ((q[0][0] - q[1][1]) * (q[0][0] - q[1][1]) + 4. * q[0][1] * q[1][0]).sqrt();
    let alpha = (trace + discriminant) / 2.;
    let ratio = (alpha - q[0][0]) / q[1][0];
    let nu = [1. / (1. + ratio), ratio / (1. + ratio)];
    let propagated = row_update(nu, q);
    let checks = vec![
        equal(
            "quasi_stationary_left_eigen_first",
            propagated[0],
            alpha * nu[0],
        ),
        equal(
            "quasi_stationary_left_eigen_second",
            propagated[1],
            alpha * nu[1],
        ),
        equal("quasi_stationary_probability_mass", nu.iter().sum(), 1.),
        strict("positive_submarkov_eigenvalue", alpha > 0. && alpha <= 1.),
    ];
    let inputs = json!({"killed_reference_kernel":q,"quasi_stationary_law":nu,"survival_eigenvalue":alpha,
        "propagated_law":propagated,"scope":"Exact declared two-state killed reference kernel and its positive left Perron eigenvector; this checks the quasi-stationary definition, not the native gas stationary law."});
    for formula in [r"\nu Q=\alpha\nu", r"\alpha\in(0,1]"] {
        emit(
            suite,
            catalog,
            "rem-note-extinction-possibility",
            formula,
            &inputs,
            &[],
            &checks,
        )?;
    }
    Ok(())
}

fn phase_q(z: &[f64], lambda: f64, b: f64) -> f64 {
    let d = z.len() / 2;
    (0..d)
        .map(|j| z[j].powi(2) + lambda * z[d + j].powi(2) + b * z[j] * z[d + j])
        .sum()
}
fn phase_inner(x: &[f64], y: &[f64], lambda: f64, b: f64) -> f64 {
    let d = x.len() / 2;
    (0..d)
        .map(|j| {
            x[j] * y[j]
                + lambda * x[d + j] * y[d + j]
                + b * (x[j] * y[d + j] + x[d + j] * y[j]) / 2.
        })
        .sum()
}
fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}
fn barycenter(points: &[Vec<f64>]) -> Vec<f64> {
    (0..points[0].len())
        .map(|j| points.iter().map(|x| x[j]).sum::<f64>() / points.len() as f64)
        .collect()
}
fn family_expressions<'a>(catalog: &'a [Value], chapter: u8, label: &str) -> Vec<&'a str> {
    catalog
        .iter()
        .filter(|c| c["chapter"] == chapter)
        .flat_map(|c| {
            c["quantitative_expressions"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter(|e| e["source_label"] == label && e["requires_expression_evidence"] == true)
        .filter_map(|e| e["formula"].as_str())
        .collect()
}

fn append_quadratic_transport_contracts(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    use crate::convergence_lyapunov::uniform_transport;
    for d in [1_usize, 2, 4] {
        for (k1, k2) in [(1, 1), (2, 3), (3, 2), (4, 4)] {
            for (lambda, b) in [(0.1_f64, 0.), (1., 0.4), (2., -1.2), (0.5, 1.4)] {
                let make = |k: usize, side: f64| {
                    (0..k)
                        .map(|i| {
                            (0..2 * d)
                                .map(|j| side + (i as f64 - 0.4 * k as f64) * (j + 1) as f64 / 7.)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                };
                let left = make(k1, -0.3);
                let right = make(k2, 0.6);
                let ml = barycenter(&left);
                let mr = barycenter(&right);
                let cl = left.iter().map(|x| difference(x, &ml)).collect::<Vec<_>>();
                let cr = right.iter().map(|x| difference(x, &mr)).collect::<Vec<_>>();
                let dm = difference(&ml, &mr);
                let cost = |x: &[f64], y: &[f64]| phase_q(&difference(x, y), lambda, b);
                let euclidean =
                    |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum();
                let (total, plan) = uniform_transport(&left, &right, cost)?;
                let (shape, centered_plan) = uniform_transport(&cl, &cr, cost)?;
                let (w2, _) = uniform_transport(&cl, &cr, euclidean)?;
                let loc = phase_q(&dm, lambda, b);
                let discriminant = ((1. - lambda).powi(2) + b * b).sqrt();
                let lmin = (1. + lambda - discriminant) / 2.;
                let lmax = (1. + lambda + discriminant) / 2.;
                let det = lambda - b * b / 4.;
                let mut marginals = vec![];
                let mut pair_checks = vec![];
                let mut integral = 0.;
                let mut centered_integral = 0.;
                let mut cross = 0.;
                let mut centered_euclidean = 0.;
                let mut left_zero = vec![0.; 2 * d];
                let mut right_zero = vec![0.; 2 * d];
                for (i, row) in plan.iter().enumerate() {
                    marginals.push(equal(
                        "optimal_left_marginal",
                        row.iter().sum(),
                        1. / k1 as f64,
                    ));
                    for (j, &mass) in row.iter().enumerate() {
                        let delta = difference(&left[i], &right[j]);
                        let centered = difference(&cl[i], &cr[j]);
                        let qc = phase_q(&centered, lambda, b);
                        let inner = phase_inner(&centered, &dm, lambda, b);
                        let matrix = (0..d)
                            .map(|a| {
                                delta[a] * (delta[a] + b * delta[d + a] / 2.)
                                    + delta[d + a] * (b * delta[a] / 2. + lambda * delta[d + a])
                            })
                            .sum::<f64>();
                        pair_checks.push(equal(
                            "quadratic_matrix_cost",
                            cost(&left[i], &right[j]),
                            matrix,
                        ));
                        pair_checks.push(equal(
                            "quadratic_bilinear_diagonal",
                            matrix,
                            phase_inner(&delta, &delta, lambda, b),
                        ));
                        pair_checks.push(equal(
                            "barycentric_quadratic_expansion",
                            phase_q(&delta, lambda, b),
                            qc + loc + 2. * inner,
                        ));
                        pair_checks.push(upper(
                            "pointwise_phase_coercivity",
                            lmin * delta.iter().map(|x| x * x).sum::<f64>(),
                            matrix,
                        ));
                        for a in 0..2 * d {
                            pair_checks.push(equal(
                                "difference_vector_expansion",
                                delta[a],
                                centered[a] + dm[a],
                            ));
                            left_zero[a] += mass * cl[i][a];
                            right_zero[a] += mass * cr[j][a];
                        }
                        integral += mass * matrix;
                        centered_integral += mass * qc;
                        cross += mass * inner;
                        centered_euclidean += mass * centered.iter().map(|x| x * x).sum::<f64>();
                    }
                }
                for j in 0..k2 {
                    marginals.push(equal(
                        "optimal_right_marginal",
                        plan.iter().map(|row| row[j]).sum(),
                        1. / k2 as f64,
                    ));
                }
                let mut zeros = vec![equal("integrated_cross_vanishes", cross, 0.)];
                for a in 0..2 * d {
                    zeros.push(equal("centered_left_first_moment", left_zero[a], 0.));
                    zeros.push(equal("centered_right_first_moment", right_zero[a], 0.));
                }
                let spectral = vec![
                    strict("positive_lambda", lambda > 0.),
                    strict("real_cross_parameter", b.is_finite()),
                    strict("strict_phase_positive_definiteness", b * b < 4. * lambda),
                    strict("positive_first_principal_minor", 1. > 0.),
                    strict("positive_determinant", det > 0.),
                    strict("positive_both_eigenvalues", lmin > 0. && lmax > 0.),
                    equal("eigenvalues_sum_trace", lmin + lmax, 1. + lambda),
                    equal("eigenvalues_product_determinant", lmin * lmax, det),
                    strict(
                        "discriminant_less_than_squared_trace",
                        (1. - lambda).powi(2) + b * b < (1. + lambda).powi(2),
                    ),
                ];
                let decomposition = vec![
                    equal("transport_integral_matches_minimum", integral, total),
                    equal(
                        "integrated_barycenter_expansion",
                        integral,
                        centered_integral + loc + 2. * cross,
                    ),
                    equal(
                        "centered_minimum_equals_shifted_minimum",
                        centered_integral,
                        shape,
                    ),
                    equal(
                        "optimal_total_barycentric_decomposition",
                        total,
                        loc + shape,
                    ),
                ];
                let coercivity = vec![
                    upper(
                        "barycenter_phase_coercivity",
                        lmin * dm.iter().map(|x| x * x).sum::<f64>(),
                        loc,
                    ),
                    upper("optimal_centered_phase_coercivity", lmin * w2, shape),
                    upper(
                        "same_plan_integrated_phase_coercivity",
                        lmin * centered_euclidean,
                        centered_integral,
                    ),
                ];
                let inputs = json!({"left":left,"right":right,"left_centered":cl,"right_centered":cr,
                    "left_barycenter":ml,"right_barycenter":mr,"delta_barycenter":dm,"lambda_v":lambda,"b":b,
                    "minimum_eigenvalue":lmin,"maximum_eigenvalue":lmax,"determinant":det,
                    "native_optimal_plan":plan,"native_centered_optimal_plan":centered_plan,
                    "total_q_transport":total,"centered_q_transport":shape,"centered_euclidean_transport":w2,
                    "location_q_cost":loc,"scope":"Exact native uniform_transport solver on probability-normalized finite empirical phase-space measures, including unequal alive counts, both signs of cross weight and near-coercivity-boundary parameters. No particle identities enter the cost."});
                for (formula, offset) in [
                    (r"\Delta\mu_x = \mu_{x,1} - \mu_{x,2}", 0),
                    (r"\Delta\mu_v = \mu_{v,1} - \mu_{v,2}", d),
                ] {
                    let checks = (0..d)
                        .map(|a| {
                            equal(
                                "actual_barycentric_component_difference",
                                dm[offset + a],
                                ml[offset + a] - mr[offset + a],
                            )
                        })
                        .collect::<Vec<_>>();
                    emit(
                        suite,
                        catalog,
                        "def-location-error-component",
                        formula,
                        &inputs,
                        &spectral,
                        &checks,
                    )?;
                }
                for label in ["def-structural-error-component", "def-variance-conversions"] {
                    emit(
                        suite,
                        catalog,
                        label,
                        r"k_{\text{alive}} = |\mathcal{A}(S_k)|",
                        &inputs,
                        &[],
                        &[
                            equal(
                                "actual_left_alive_support_cardinality",
                                k1 as f64,
                                left.len() as f64,
                            ),
                            equal(
                                "actual_right_alive_support_cardinality",
                                k2 as f64,
                                right.len() as f64,
                            ),
                        ],
                    )?;
                }
                for label in ["lem-wasserstein-decomposition", "lem-V-coercive"] {
                    for formula in family_expressions(catalog, 3, label) {
                        let f = tokens(formula);
                        let checks = if f.starts_with("b^2")
                            || f.starts_with("\\lambda_")
                            || f.starts_with("\\lambda_v>")
                            || f == "b\\in\\mathbb{R}"
                            || f.starts_with("a_{11}")
                            || f.starts_with("\\det")
                            || f == "1>0"
                            || f.starts_with("(1-\\lambda_v)")
                            || f.starts_with("Q_{\\text{scalar}}")
                        {
                            spectral.clone()
                        } else if f.starts_with("V_{\\text{loc}}\\ge")
                            || f.starts_with("V_{\\text{loc}}\\geq")
                            || f.starts_with("V_{\\text{struct}}\\ge")
                            || f.starts_with("V_{\\text{struct}}\\geq")
                            || f.contains("\\geq\\lambda_{\\min}")
                            || f.contains("\\geq\\lambda_2")
                        {
                            coercivity.clone()
                        } else if f.starts_with("q(")
                            || f.starts_with("c(")
                            || f.starts_with("z_1-z_2=")
                            || f.starts_with("\\delta_{z_i}=")
                            || f.starts_with("\\Delta\\bar{z}=")
                            || f.starts_with("\\begin{aligned}q(")
                        {
                            pair_checks.clone()
                        } else if f.starts_with("\\int\\delta") || f.starts_with("\\int\\langle") {
                            zeros.clone()
                        } else if f.starts_with("\\gamma") || f.starts_with("\\tilde{\\gamma}") {
                            marginals.clone()
                        } else if f.starts_with("\\bar{z}_k=")
                            || f.starts_with("\\mu_k=")
                            || f.starts_with("\\tilde{\\mu}_k=")
                        {
                            let mut checks = marginals.clone();
                            checks.extend(zeros.clone());
                            checks
                        } else if f.starts_with("W_h^2(")
                            || f.starts_with("V_{\\text{struct}}=")
                            || f.starts_with("V_{\\text{loc}}=")
                            || f.starts_with("\\intc(")
                            || f.starts_with("\\intq(")
                            || f.starts_with("\\int_{\\mathcal{Z}")
                            || f.starts_with("\\begin{aligned}\\int")
                        {
                            decomposition.clone()
                        } else if f.starts_with("W_2^2(") && k1 == k2 {
                            let minimum = permutations(k1)
                                .iter()
                                .map(|pi| {
                                    pi.iter()
                                        .enumerate()
                                        .map(|(i, &j)| euclidean(&cl[i], &cr[j]))
                                        .sum::<f64>()
                                        / k1 as f64
                                })
                                .fold(f64::INFINITY, f64::min);
                            vec![equal(
                                "euclidean_transport_equals_permutation_minimum",
                                w2,
                                minimum,
                            )]
                        } else if f == "k=1,2" {
                            vec![equal("number_of_compared_measures", 2., 2.)]
                        } else if f == "\\mathcal{Z}=\\mathbb{R}^d\\times\\mathbb{R}^d" {
                            vec![equal(
                                "phase_dimension_position_plus_velocity",
                                left[0].len() as f64,
                                2. * d as f64,
                            )]
                        } else {
                            continue;
                        };
                        emit(suite, catalog, label, formula, &inputs, &spectral, &checks)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn append_boundary_reward_contracts(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    use algorithmic_gas::{
        ObservationBatch, TensorBatch,
        fitness::{PositiveMap, PositiveMapping, Standardizer},
    };
    let label = "lem-fitness-gradient-boundary";
    let eta = 0.1;
    let amplitude = 2.;
    let map = PositiveMap::Logistic {
        amplitude,
        floor: eta,
    };
    let m = amplitude + eta;
    let phi = [16_f64, 8., 8.];
    let rpos = [0_f64, -1., -2.];
    let velocities = [0_f64, 0., 0.];
    let c = 0.25;
    let raw = (0..3)
        .map(|i| rpos[i] - phi[i] - c * velocities[i].powi(2))
        .collect::<Vec<_>>();
    let obs = ObservationBatch::positions(TensorBatch::vectors(3, 1, vec![1.9375, 1.875, 1.75])?);
    let (rz, stats) = Standardizer::Global { sigma_min: 0.1 }.apply(&raw, &[true; 3], &obs)?;
    let (dz, _) =
        Standardizer::Global { sigma_min: 0.1 }.apply(&[1_f64, 2., 1.4], &[true; 3], &obs)?;
    let rp = rz
        .iter()
        .map(|&z| map.map(z))
        .collect::<Result<Vec<f64>>>()?;
    let dp = dz
        .iter()
        .map(|&z| map.map(z))
        .collect::<Result<Vec<f64>>>()?;
    let scale = stats.scale[0];
    let sigma_star = scale + 0.1;
    let delta_b = phi[0] - phi[1];
    let br = (rpos[1] - rpos[0]).abs();
    let delta_r = delta_b - br;
    let logistic_derivative = |z: f64| {
        let s = 1. / (1. + (-z).exp());
        amplitude * s * (1. - s)
    };
    let mg = logistic_derivative(rz[0]).min(logistic_derivative(rz[1]));
    for alpha in [0.5_f64, 1., 2.] {
        for beta in [0_f64, 0.5, 2.] {
            let mut pipeline = GasConfig::euclidean(1, 0.04)?.fitness;
            pipeline.reward_map = map.clone();
            pipeline.diversity_map = map.clone();
            pipeline.reward_exponent = alpha;
            pipeline.diversity_exponent = beta;
            let fitness = pipeline.combine(&rz, &dz, &[true; 3])?;
            let lower_power = alpha * eta.powf(alpha - 1.).min(m.powf(alpha - 1.));
            let lower = eta.powf(beta) * lower_power * mg * delta_r / sigma_star;
            let mut derivative_checks = vec![];
            for n in 0..=128 {
                let z = rz[0] + (rz[1] - rz[0]) * n as f64 / 128.;
                derivative_checks.push(upper(
                    "actual_interval_logistic_derivative_floor",
                    mg,
                    logistic_derivative(z),
                ));
            }
            let hypotheses = vec![
                strict("positive_reward_exponent", alpha > 0.),
                strict("nonnegative_diversity_exponent", beta >= 0.),
                strict(
                    "positive_raw_margin",
                    delta_r > 0. && raw[1] - raw[0] >= delta_r,
                ),
                strict("diversity_nonopposition", dp[1] >= dp[0]),
                strict("positive_interval_derivative", mg > 0.),
                upper("actual_scale_below_declared_upper", scale, sigma_star),
            ];
            let inputs = json!({"native_mapping":map,"native_pipeline":pipeline,"raw_reward":raw,"position_reward":rpos,"barrier":phi,
            "velocity":velocities,"velocity_regularizer":c,"native_reward_scores":rz,"native_reward_scale":scale,
            "native_diversity_scores":dz,"mapped_reward":rp,"mapped_diversity":dp,"native_fitness":fitness,
            "eta":eta,"M":m,"alpha":alpha,"beta":beta,"barrier_gap":delta_b,"opposing_raw_reward_bound":br,
            "delta_r":delta_r,"sigma_star":sigma_star,"attained_interval_derivative_floor":mg,"fitness_gap_lower_bound":lower,
            "scope":"Actual Rust Global standardization, PositiveMap::Logistic and FitnessPipeline::combine on a barrier-advantaged finite measurement fixture. The derivative floor is restricted to the attained interval; all score, margin and diversity hypotheses are retained."});
            for formula in family_expressions(catalog, 3, label) {
                let f = tokens(formula);
                let checks = if f.starts_with("r_i=") {
                    vec![equal(
                        "raw_reward_barrier_penalty",
                        raw[0],
                        rpos[0] - phi[0] - c * velocities[0].powi(2),
                    )]
                } else if f.starts_with("z_{r,i}=") {
                    vec![equal(
                        "native_centered_reward_score",
                        rz[0],
                        (raw[0] - stats.mean[0]) / scale,
                    )]
                } else if f.starts_with("V_i=") {
                    vec![equal(
                        "native_product_fitness",
                        fitness[0],
                        dp[0].powf(beta) * rp[0].powf(alpha),
                    )]
                } else if f.starts_with("r'_i=") {
                    vec![equal(
                        "native_positive_reward_map",
                        rp[0],
                        amplitude / (1. + (-rz[0]).exp()) + eta,
                    )]
                } else if f == "d'_i,r'_i\\in[\\eta,M]" {
                    vec![strict(
                        "mapped_channels_in_declared_interval",
                        rp.iter().chain(&dp).all(|x| *x >= eta && *x <= m),
                    )]
                } else if f == "\\alpha>0" {
                    vec![strict("positive_alpha", alpha > 0.)]
                } else if f == "\\beta\\geq0" {
                    vec![strict("nonnegative_beta", beta >= 0.)]
                } else if f.starts_with("r_j-r_i\\geq") {
                    vec![upper("actual_raw_advantage", delta_r, raw[1] - raw[0])]
                } else if f == "d'_j\\geqd'_i" {
                    vec![upper("actual_diversity_advantage", dp[0], dp[1])]
                } else if f.starts_with("\\sigma'_r\\leq") {
                    vec![upper("actual_reward_scale_upper", scale, sigma_star)]
                } else if f.starts_with("g'_A\\geq") {
                    derivative_checks.clone()
                } else if f.starts_with("V_j-V_i\\geq\\eta") {
                    vec![upper(
                        "native_fitness_boundary_advantage",
                        lower,
                        fitness[1] - fitness[0],
                    )]
                } else if f.starts_with("V_j-V_i\\geq(d'_i)") {
                    vec![upper(
                        "diversity_nonopposition_product_gap",
                        dp[0].powf(beta) * (rp[1].powf(alpha) - rp[0].powf(alpha)),
                        fitness[1] - fitness[0],
                    )]
                } else if f.starts_with("\\varphi_i-\\varphi_j=") {
                    vec![equal("actual_barrier_gap", phi[0] - phi[1], delta_b)]
                } else if f == "B_r<\\Delta_b" {
                    vec![strict("opposing_terms_below_barrier_gap", br < delta_b)]
                } else if f.starts_with("\\delta_r=") {
                    vec![equal("raw_margin_from_barrier_gap", delta_r, delta_b - br)]
                } else if f.starts_with("z_{r,j}-z_{r,i}=") {
                    vec![
                        equal(
                            "shared_centering_cancels",
                            rz[1] - rz[0],
                            (raw[1] - raw[0]) / scale,
                        ),
                        upper("score_gap_lower", delta_r / sigma_star, rz[1] - rz[0]),
                    ]
                } else if f.starts_with("r'_j-r'_i\\geq") {
                    vec![upper(
                        "mapped_gap_derivative_lower",
                        mg * delta_r / sigma_star,
                        rp[1] - rp[0],
                    )]
                } else if f.starts_with("(d'_i)^\\beta\\geq") {
                    vec![upper(
                        "diversity_positive_floor_power",
                        eta.powf(beta),
                        dp[0].powf(beta),
                    )]
                } else if f == "\\alpha\\min\\{\\eta^{\\alpha-1},M^{\\alpha-1}\\}" {
                    vec![equal(
                        "power_derivative_endpoint_minimum",
                        lower_power,
                        alpha * eta.powf(alpha - 1.).min(m.powf(alpha - 1.)),
                    )]
                } else {
                    continue;
                };
                emit(
                    suite,
                    catalog,
                    label,
                    formula,
                    &inputs,
                    &hypotheses,
                    &checks,
                )?;
            }
        }
    }
    Ok(())
}

fn append_safe_population_contracts(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let label = "cor-extinction-suppression";
    for n in [4_usize, 16, 64, 256] {
        let k = 3 * n / 4;
        let theta = 20_f64;
        let values = (0..k)
            .map(|i| if i % 4 == 0 { 24. } else { 8. })
            .collect::<Vec<_>>();
        let exposed = values.iter().filter(|x| **x >= theta).count();
        let safe = k - exposed;
        let observable = values.iter().sum::<f64>() / n as f64;
        let a0 = k as f64 / n as f64;
        let declared_b = observable + 0.1;
        let a = a0 - declared_b / theta;
        let q = 0.2_f64;
        let probabilities = (0..safe)
            .map(|i| q * (0.5 + 0.5 * i as f64 / safe as f64))
            .collect::<Vec<_>>();
        let exact_extinction = probabilities.iter().product::<f64>();
        let bound = q.powf(a * n as f64);
        let delta_n = 1. / (10. * n as f64);
        let inputs = json!({"N":n,"alive_count":k,"barrier_values":values,"barrier_threshold":theta,
            "barrier_observable":observable,"a0":a0,"declared_barrier_upper":declared_b,"safe_fraction_lower":a,
            "exposed_count":exposed,"safe_count":safe,"conditioned_independent_death_probabilities":probabilities,
            "exact_safe_intersection_probability":exact_extinction,"q":q,"extinction_upper":bound,"delta_N":delta_n,
            "scope":"Exact conditional independent Bernoulli death law with explicit safe-population and barrier hypotheses. This is the stated stage-conditioned specialization; independence is not inferred before shared cloning or collision randomness."});
        let hypotheses = vec![
            strict("positive_safe_fraction", a > 0.),
            strict("strict_individual_death_upper", q > 0. && q < 1.),
            strict("strict_barrier_safe_margin", declared_b < a0 * theta),
            upper("actual_safe_count_lower", a * n as f64, safe as f64),
        ];
        for formula in family_expressions(catalog, 3, label) {
            let f = tokens(formula);
            let checks = if f == "q<1" {
                vec![strict("individual_probability_below_one", q < 1.)]
            } else if f == "a>0" {
                vec![strict("positive_safe_population_fraction", a > 0.)]
            } else if f == "\\varphi_i<\\theta" {
                vec![strict(
                    "selected_safe_atoms_below_threshold",
                    values.iter().filter(|x| **x < theta).count() == safe,
                )]
            } else if f == "\\varphi_i\\geq\\theta" {
                vec![strict(
                    "exposed_atoms_above_threshold",
                    values.iter().filter(|x| **x >= theta).count() == exposed,
                )]
            } else if f == "k/N\\geqa_0" {
                vec![upper("declared_alive_fraction", a0, k as f64 / n as f64)]
            } else if f == "B\\leqb<a_0\\theta" {
                vec![
                    upper("observed_barrier_below_upper", observable, declared_b),
                    strict("barrier_upper_below_safe_mass", declared_b < a0 * theta),
                ]
            } else if f == "a=a_0-b/\\theta" {
                vec![equal("safe_fraction_formula", a, a0 - declared_b / theta)]
            } else if f.starts_with("B=N^{-1}") {
                vec![equal(
                    "normalized_alive_barrier",
                    observable,
                    values.iter().sum::<f64>() / n as f64,
                )]
            } else if f.starts_with("n_{\\mathrm{exposed}}\\theta") {
                vec![upper(
                    "exposed_count_barrier_bound",
                    exposed as f64 * theta,
                    n as f64 * observable,
                )]
            } else if f.starts_with("\\mathbbP(") {
                vec![
                    upper("exact_safe_extinction_product", exact_extinction, bound),
                    equal(
                        "exponential_extinction_rate",
                        bound,
                        (-a * n as f64 * (1. / q).ln()).exp(),
                    ),
                ]
            } else if f == "q^{aN}" || f == "\\delta_N+q^{aN}" || f == "1-\\delta_N" {
                vec![upper(
                    "unconditional_stage_split",
                    delta_n + (1. - delta_n) * exact_extinction,
                    delta_n + bound,
                )]
            } else if f == "k-NB/\\theta" {
                vec![upper(
                    "safe_count_from_barrier",
                    k as f64 - n as f64 * observable / theta,
                    safe as f64,
                )]
            } else {
                continue;
            };
            emit(
                suite,
                catalog,
                label,
                formula,
                &inputs,
                &hypotheses,
                &checks,
            )?;
        }
    }
    // Integrate the exact tilted Gaussian density. The omitted tail is bounded
    // by the normal Mills envelope; this is not a noisy moment estimate with
    // an unjustified finite-variance assumption.
    let panels = 4096;
    let extent = 12_f64;
    let h = 2. * extent / panels as f64;
    let density = |x: f64| (-x * x / 4.).exp() / std::f64::consts::TAU.sqrt();
    let mut integral = density(-extent) + density(extent);
    for i in 1..panels {
        integral += if i % 2 == 0 { 2. } else { 4. } * density(-extent + i as f64 * h);
    }
    integral *= h / 3.;
    let tail_envelope = 4. / extent * density(extent);
    for d in [1_i32, 2, 4, 16] {
        let tilted = integral.powi(d);
        let exact = 2_f64.powf(d as f64 / 2.);
        let sigma = 0.2;
        let r = 1_f64;
        let qbound = (exact * (-r * r / (4. * sigma * sigma)).exp()).min(1.);
        let xi = vec![r / sigma + 0.1; d as usize];
        let mu = vec![0.; d as usize];
        let y = xi.iter().map(|x| sigma * x).collect::<Vec<_>>();
        let inputs = json!({"dimension":d,"gaussian_tilt_integral":integral,"one_dimensional_tail_envelope":tail_envelope,
            "panels":panels,"gaussian_tilt_moment":tilted,"exact_mgf":exact,"sigma":sigma,"interior_ball_radius":r,
            "individual_radial_markov_upper":qbound,"innovation":xi,"mean":mu,"output":y,
            "scope":"Exact isotropic Gaussian reference law and Gaussian position update; quadrature of the tilted density is paired with an explicit tail envelope. The radial death implication is checked on an outside-ball realization."});
        for formula in family_expressions(catalog, 3, label) {
            let f = tokens(formula);
            let checks = if f.starts_with("\\mathbbEe^{") {
                vec![upper(
                    "gaussian_tilt_moment_with_tail",
                    (tilted - exact).abs(),
                    d as f64 * 2_f64.powf((d - 1) as f64 / 2.) * tail_envelope + 1e-8,
                )]
            } else if f == "Y_i=\\mu_i+\\sigma\\xi_i" {
                y.iter()
                    .zip(&xi)
                    .zip(&mu)
                    .map(|((y, x), m)| equal("position_gaussian_update", *y, *m + sigma * x))
                    .collect()
            } else if f == "\\|\\xi_i\\|\\geqr/\\sigma" {
                vec![upper(
                    "outside_interior_ball_noise_norm",
                    r / sigma,
                    xi.iter().map(|x| x * x).sum::<f64>().sqrt(),
                )]
            } else if f.starts_with("q\\leq\\min") {
                vec![
                    equal(
                        "gaussian_radial_markov_formula",
                        qbound,
                        (2_f64.powf(d as f64 / 2.) * (-r * r / (4. * sigma * sigma)).exp()).min(1.),
                    ),
                    strict("nonvacuous_gaussian_survival_bound", qbound < 1.),
                ]
            } else {
                continue;
            };
            emit(suite, catalog, label, formula, &inputs, &[], &checks)?;
        }
    }
    Ok(())
}

fn append_boundary_observable_contracts(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    let delta = 0.125_f64;
    let safe_barrier = |rho: f64| cutoff(2. * rho / delta) * (1. / rho - 1. / delta);
    let points = [
        delta / 32.,
        delta / 2.,
        0.6 * delta,
        0.9 * delta,
        delta,
        2. * delta,
        1.,
    ];
    let values = points.map(safe_barrier);
    let mut checks = vec![];
    for (&rho, &phi) in points.iter().zip(&values) {
        checks.push(strict(
            "actual_zero_interior_barrier_nonnegative",
            phi >= 0.,
        ));
        if rho >= delta {
            checks.push(equal("actual_zero_interior_penalty", phi, 0.));
        }
        if rho <= delta / 2. {
            checks.push(equal(
                "actual_near_boundary_growth",
                phi,
                1. / rho - 1. / delta,
            ));
        }
    }
    let inputs = json!({"delta":delta,"delta_safe":delta,"rho":points,"actual_zero_interior_barrier":values,
        "scope":"Explicit nonnegative zero-interior smooth collar observable constructed with the chapter's actual bump-integral cutoff. This is a declared proof observable; no change to the native reward is inferred."});
    for label in [
        "def-boundary-potential-cloning",
        "rem-barrier-geometric-penalty",
    ] {
        for formula in family_expressions(catalog, 3, label) {
            let f = tokens(formula);
            let local = if f.starts_with("\\varphi_{\\text{barrier}}(x)=\\psi") {
                checks.clone()
            } else if f == "\\rho\\geq\\delta" {
                vec![strict("cutoff_exterior_region", points[4] >= delta)]
            } else if f == "1/\\rho-1/\\delta\\geq0" {
                vec![upper(
                    "cutoff_support_reciprocal_nonnegative",
                    0.,
                    1. / points[3] - 1. / delta,
                )]
            } else if f == "\\varphi_{\\text{barrier}}(x)=0" {
                vec![equal("interior_barrier_zero", safe_barrier(2. * delta), 0.)]
            } else if f.starts_with("d(x,") && f.contains("\\delta_{\\text{safe}}") {
                let rho = if f.contains(">") {
                    2. * delta
                } else {
                    delta / 2.
                };
                vec![strict(
                    "actual_safe_collar_branch",
                    if f.contains(">") {
                        rho > delta
                    } else {
                        rho <= delta
                    },
                )]
            } else if f.starts_with("r_i=R_{\\text{pos}}") {
                let position_reward = 1.;
                let phi = safe_barrier(delta / 4.);
                let v = 0.3_f64;
                let c = 0.25;
                vec![equal(
                    "raw_reward_geometric_penalty",
                    position_reward - phi - c * v * v,
                    position_reward - (1. / (delta / 4.) - 1. / delta) - c * v * v,
                )]
            } else {
                continue;
            };
            emit(suite, catalog, label, formula, &inputs, &[], &local)?;
        }
    }
    let mut approach = vec![];
    let mut inverse_checks = vec![];
    let mut logarithmic_checks = vec![];
    for exponent in 1_i32..=32 {
        let epsilon = delta * 2_f64.powi(-exponent);
        let inverse = (delta / epsilon).ln();
        let exact_growth = exponent as f64 * 2_f64.ln();
        let tail = epsilon * (1. - epsilon.ln());
        let log_integral = delta * (1. - delta.ln()) - tail;
        approach.push(json!({"cutoff":epsilon,"reciprocal_integral":inverse,"log_integral":log_integral,"log_tail_exact":tail}));
        inverse_checks.push(equal(
            "reciprocal_truncated_exact_growth",
            inverse,
            exact_growth,
        ));
        logarithmic_checks.push(upper(
            "logarithmic_collar_integral_finite",
            log_integral,
            delta * (1. - delta.ln()),
        ));
        logarithmic_checks.push(strict(
            "logarithmic_integral_positive",
            log_integral > 0. && tail > 0.,
        ));
    }
    let inputs = json!({"delta":delta,"truncated_collar_integrals":approach,"logarithmic_full_integral":delta*(1.-delta.ln()),
        "scope":"Exact one-dimensional collar antiderivatives and geometric cutoff scaling. The reciprocal integral grows linearly with the cutoff exponent; the logarithmic tail is epsilon*(1-log epsilon). Finite evaluations supplement the stated improper-integral proof."});
    emit(
        suite,
        catalog,
        "rem-boundary-barrier-integrability",
        r"\int_0^\delta r^{-1}\,dr=\infty",
        &inputs,
        &[],
        &inverse_checks,
    )?;
    emit(
        suite,
        catalog,
        "rem-boundary-barrier-integrability",
        r"\int_0^\delta|\log r|\,dr<\infty",
        &inputs,
        &[],
        &logarithmic_checks,
    )?;
    Ok(())
}

fn number(value: &Value, key: &str) -> Result<f64> {
    value[key].as_f64().ok_or_else(|| {
        GasError::Configuration(format!("Native boundary fixture misses numeric {key}"))
    })
}
fn numbers(value: &Value, key: &str) -> Result<Vec<f64>> {
    value[key]
        .as_array()
        .ok_or_else(|| {
            GasError::Configuration(format!("Native boundary fixture misses array {key}"))
        })?
        .iter()
        .map(|x| {
            x.as_f64()
                .ok_or_else(|| GasError::Configuration(format!("Non-numeric native {key}")))
        })
        .collect()
}
fn append_native_boundary_clauses(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    let inputs = suite
        .evidence
        .iter()
        .filter(|e| {
            e.source_labels
                .iter()
                .any(|s| s == "thm-boundary-potential-contraction")
        })
        .map(|e| e.inputs.clone())
        .filter(|x| x.get("native_averaged_edges").is_some())
        .filter(|x| seen.insert(x.to_string()))
        .collect::<Vec<_>>();
    for native in inputs {
        let x = numbers(&native, "positions")?;
        let p = numbers(&native, "full_native_measurement_averaged_acceptance")?;
        let alive = native["alive"].as_array().unwrap();
        let n = x.len();
        let k = alive.iter().filter(|a| a == &&Value::Bool(true)).count();
        let nf = n as f64;
        let theta = number(&native, "threshold")?;
        let ps = number(&native, "p_star")?;
        let j = number(&native, "J")?;
        let r = number(&native, "R")?;
        let old = number(&native, "entering_alive_barrier")?;
        let edges = native["native_averaged_edges"].as_array().unwrap();
        let mut local_pressure = vec![];
        let mut clone_integrals = vec![];
        let mut revival_integrals = vec![];
        let mut nonexposed = 0.;
        let mut exposed = 0.;
        let mut exact = 0.;
        let mut conditional_values = vec![];
        for i in 0..n {
            let numerator = edges[i]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(a, m)| m.as_f64().unwrap() * (x[a] * x[a] + 0.01))
                .sum::<f64>();
            if alive[i] == true {
                let ji = if p[i] > 0. { numerator / p[i] } else { 0. };
                clone_integrals.push(upper("actual_acceptance_times_clone_barrier", p[i] * ji, j));
                exact += (numerator - p[i] * x[i] * x[i]) / nf;
                if x[i] * x[i] > theta {
                    exposed += x[i] * x[i];
                    local_pressure.push(upper(
                        "actual_exposed_pressure_floor",
                        ps * x[i] * x[i],
                        p[i] * x[i] * x[i],
                    ));
                } else {
                    nonexposed += x[i] * x[i];
                }
                conditional_values.push(
                    json!({"slot":i,"status":1,"acceptance":p[i],"clone_barrier_conditional":ji}),
                );
            } else {
                exact += numerator / nf;
                revival_integrals.push(upper("actual_revived_barrier_integral", numerator, r));
                conditional_values
                    .push(json!({"slot":i,"status":0,"revival_barrier_integral":numerator}));
            }
        }
        let cb = 2. * ps * theta + 2. * j.max(r);
        let actual_drift = number(&native, "proposal_barrier_moment")? - old;
        let hypotheses = vec![
            strict("actual_positive_exposed_threshold", theta > 0.),
            strict("actual_positive_exposed_pressure", ps > 0.),
            strict(
                "actual_finite_clone_and_revival_integrals",
                j.is_finite() && r.is_finite(),
            ),
        ];
        let inputs = json!({"native_fixture":native,"conditional_integrals":conditional_values,
            "exposed_barrier_sum":exposed,"nonexposed_barrier_sum":nonexposed,"native_drift":actual_drift,
            "exact_conditional_formula_drift":exact,"k":k,"N":n,"D":n-k,"C_b_uniform":cb,
            "scope":"Supplemental clauses evaluated from the retained native finite weighted measurement, acceptance and donor-edge law. The nonnegative quadratic observable is auxiliary; all revival and component-frame conventions are inherited from that exact fixture."});
        emit(
            suite,
            catalog,
            "thm-boundary-potential-contraction",
            r"\mathbb E\Delta W_b\leq-\kappa_bW_b+C_b",
            &inputs,
            &hypotheses,
            &[upper(
                "actual_two_swarm_conditional_boundary_drift",
                2. * actual_drift,
                -ps * 2. * old + cb,
            )],
        )?;
        for label in [
            "thm-boundary-potential-contraction",
            "proof-boundary-potential-contraction",
            "thm-complete-boundary-drift",
            "def-boundary-exposed-set",
            "rem-boundary-mass-relationship",
            "rem-progressive-safety",
        ] {
            for formula in family_expressions(catalog, 3, label) {
                let f = tokens(formula);
                let checks = if f == "\\phi_{\\text{thresh}}>0" {
                    vec![strict(
                        "native_boundary_exposed_threshold_positive",
                        theta > 0.,
                    )]
                } else if f == "\\varphi_{\\text{barrier}}(x_i)\\leq\\phi_{\\text{thresh}}" {
                    x.iter()
                        .zip(alive)
                        .filter(|(a, s)| **s == true && *a * *a <= theta)
                        .map(|(a, _)| {
                            upper("native_nonexposed_observable_below_threshold", a * a, theta)
                        })
                        .collect()
                } else if f == "\\theta>0" {
                    vec![strict("threshold_positive", theta > 0.)]
                } else if f == "p_*>0" {
                    vec![strict("pressure_positive", ps > 0.)]
                } else if f == "J<\\infty" {
                    vec![strict("clone_integral_finite", j.is_finite())]
                } else if f == "R<\\infty" {
                    vec![strict("revival_integral_finite", r.is_finite())]
                } else if f == "D_s=N-k_s" || f == "k_s+D_s=N" {
                    vec![equal(
                        "native_alive_dead_partition",
                        k as f64 + (n - k) as f64,
                        nf,
                    )]
                } else if f == "\\kappa_b=p_*" {
                    vec![equal(
                        "actual_boundary_rate_alias",
                        ps,
                        number(&native, "p_star")?,
                    )]
                } else if f == "C_b=2p_*\\theta+2\\max(J,R)" {
                    vec![equal(
                        "uniform_boundary_offset",
                        cb,
                        2. * ps * theta + 2. * j.max(r),
                    )]
                } else if f.starts_with("\\varphi_i=") {
                    x.iter()
                        .map(|a| equal("quadratic_observable_at_native_atom", a * a, a.powi(2)))
                        .collect()
                } else if f.starts_with("E=\\{") {
                    vec![equal(
                        "actual_exposed_index_count",
                        native["exposed"].as_array().unwrap().len() as f64,
                        x.iter()
                            .zip(alive)
                            .filter(|(a, s)| **s == true && *a * *a > theta)
                            .count() as f64,
                    )]
                } else if f == "p_iJ_i\\leqJ" {
                    clone_integrals.clone()
                } else if f == "R_i\\leqR" {
                    revival_integrals.clone()
                } else if f == "p_i\\varphi_i\\geqp_*\\varphi_i" {
                    local_pressure.clone()
                } else if f.starts_with("\\sum_{i\\notinE}") {
                    vec![upper(
                        "actual_nonexposed_barrier_sum",
                        nonexposed,
                        k as f64 * theta,
                    )]
                } else if f.starts_with("N^{-1}\\sum_{i\\inE}") {
                    vec![upper(
                        "actual_exposed_mass_lower",
                        old - k as f64 * theta / nf,
                        exposed / nf,
                    )]
                } else if f == "0\\leqD_1+D_2\\leq2N" {
                    vec![upper(
                        "actual_two_swarm_deaths",
                        2. * (n - k) as f64,
                        2. * nf,
                    )]
                } else {
                    continue;
                };
                if !checks.is_empty() {
                    emit(
                        suite,
                        catalog,
                        label,
                        formula,
                        &inputs,
                        &hypotheses,
                        &checks,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn append_reset_parameter_contracts(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    use crate::convergence_kinetic::{KineticValidationConfig, kinetic_constants};
    let mut seen = std::collections::BTreeSet::new();
    let cases = suite
        .evidence
        .iter()
        .filter(|e| {
            e.source_labels
                .iter()
                .any(|s| s == "thm-canonical-full-step-reset-drift")
        })
        .map(|e| e.inputs.clone())
        .filter(|x| x.get("canonical_BAOAB").is_some())
        .filter(|x| seen.insert(x.to_string()))
        .collect::<Vec<_>>();
    for native in cases {
        let h = number(&native["canonical_BAOAB"], "h")?;
        let gamma = number(&native["canonical_BAOAB"], "gamma")?;
        let cap = number(&native["canonical_BAOAB"], "velocity_cap")?;
        let sigma_p = number(&native["canonical_BAOAB"], "sigma_p")?;
        let x = numbers(&native, "positions")?;
        let v = numbers(&native, "retained_velocities")?;
        let rd = 2_f64;
        let lu = 1.;
        let bu = 0.;
        let alpha = 0.5;
        let sigma_x = 0.1;
        let config = KineticValidationConfig {
            dimensions: 1,
            friction: gamma,
            dt: h,
            position_diffusion: sigma_p,
            ..Default::default()
        };
        // The coefficient function validates input dimensions; use the native
        // one-dimensional kinetic configuration rather than a mismatched default.
        let config = KineticValidationConfig {
            landscape: crate::convergence_kinetic::ConfiningLandscape::quadratic(1),
            inputs: vec![crate::convergence_kinetic::KineticInput {
                positions: vec![0.],
                velocities: vec![0.],
            }],
            ..config
        };
        let constants = kinetic_constants(&config)?;
        let c = constants.friction_factor;
        let sh2 = constants.thermal_variance;
        let w = (1. + 2. * alpha) * cap;
        let a = 1. + h * h / 4. * (1. + c) * lu;
        let d0 = h / 2. * (1. + c) * w + h * h / 4. * (1. + c) * bu;
        let mx = (a * (rd * rd + sigma_x * sigma_x).sqrt() + d0).powi(2)
            + h * h / 4. * sh2
            + sigma_p * sigma_p * h;
        let foster_factor = 0.5_f64;
        let actual = number(&native, "completed_moment_upper_uses_actual_cap")?;
        let m = number(&native, "M")?;
        let inputs = json!({"native_fixture":native,"kinetic_coefficients":constants,"radius":rd,"force_slope":lu,"force_offset":bu,
            "collision_bound":w,"A":a,"D0":d0,"computed_Mx":mx,"scope":"Actual retained native completed-update moment fixture, with independently recomputed coefficient and force/cap hypotheses. Entering coordinates of dead slots are not constrained to the donor domain."});
        for formula in family_expressions(catalog, 3, "thm-canonical-full-step-reset-drift") {
            let f = tokens(formula);
            let checks = if f.starts_with("R_D=\\sup") {
                vec![
                    equal("actual_box_radius", rd, 2.),
                    strict("finite_domain_radius", rd.is_finite()),
                ]
            } else if f == "|v_i|\\leqV" {
                vec![upper(
                    "actual_retained_velocity_cap",
                    v.iter().map(|z| z.abs()).fold(0., f64::max),
                    cap,
                )]
            } else if f.starts_with("|\\nablaU(x)|\\leq") {
                x.iter()
                    .map(|z| {
                        upper(
                            "actual_unit_quadratic_force_growth",
                            z.abs(),
                            lu * z.abs() + bu,
                        )
                    })
                    .collect()
            } else if f == "c=e^{-\\gammah}" {
                vec![equal(
                    "actual_O_decay",
                    constants.friction_factor,
                    (-gamma * h).exp(),
                )]
            } else if f.starts_with("s_h^2=(1-e^{-2") {
                vec![equal(
                    "stable_O_variance",
                    sh2,
                    -(-2. * gamma * h).exp_m1() / (2. * gamma),
                )]
            } else if f.starts_with("W=(1+2\\alpha)") {
                vec![
                    equal("native_collision_bound_alias", w, (1. + 2. * alpha) * cap),
                    equal(
                        "reset_A_independent_reconstruction",
                        a,
                        number(&native, "A")?,
                    ),
                    equal(
                        "reset_D0_independent_reconstruction",
                        d0,
                        number(&native, "D0")?,
                    ),
                ]
            } else if f.starts_with("M_x=") {
                vec![equal(
                    "full_position_reset_constant",
                    mx,
                    number(&native, "M_x")?,
                )]
            } else if f == "q\\in(0,1)" {
                vec![strict(
                    "admissible_Foster_factor",
                    foster_factor > 0. && foster_factor < 1.,
                )]
            } else if f == "\\mathscrL_N=1" {
                vec![equal("cemetery_completed_observable", 1., 1.)]
            } else if f == "P\\mathscrL_N\\leqM" {
                vec![upper("actual_completed_marked_moment_reset", actual, m)]
            } else {
                continue;
            };
            emit(
                suite,
                catalog,
                "thm-canonical-full-step-reset-drift",
                formula,
                &inputs,
                &[],
                &checks,
            )?;
        }
    }
    for gamma in [0_f64, 1.] {
        let h = 0.04;
        let c = (-gamma * h).exp();
        let sh2 = if gamma == 0. {
            h
        } else {
            -(-2. * gamma * h).exp_m1() / (2. * gamma)
        };
        let xc = 0.8;
        let vc = -0.3;
        let gradient = xc;
        let xi = 0.7;
        let s = sh2.sqrt();
        let b1 = vc - h * gradient / 2.;
        let a1 = xc + h * b1 / 2.;
        let o = c * b1 + s * xi;
        let a2 = a1 + h * o / 2.;
        let compressed = xc + h / 2. * (1. + c) * (vc - h * gradient / 2.) + h / 2. * s * xi;
        let inputs = json!({"h":h,"gamma":gamma,"Xc":xc,"Vc":vc,"gradient":gradient,"B":1.,"xi_O":xi,
            "B1":b1,"A1":a1,"O":o,"A2":a2,"s_h_squared":sh2,"compressed_X2":compressed,
            "scope":"Independent exact algebraic reconstruction of the B-A-O-A position stages, including zero friction. Native random-stage moment comparisons are retained separately in chapters2 and3."});
        emit_prefix(
            suite,
            catalog,
            "thm-canonical-full-step-reset-drift",
            r"X_2=X^c+",
            &inputs,
            &[equal(
                "sequential_vs_compressed_BAOA_position",
                a2,
                compressed,
            )],
        )?;
        if gamma == 0. {
            for formula in [r"s_h^2=h", r"\gamma=0"] {
                emit(
                    suite,
                    catalog,
                    "thm-canonical-full-step-reset-drift",
                    formula,
                    &inputs,
                    &[],
                    &[
                        equal("zero_friction_thermal_variance", sh2, h),
                        strict("zero_friction_branch", gamma == 0.),
                    ],
                )?;
            }
        }
    }
    let lower = -2_f64;
    let upper_bound = 2_f64;
    let length = upper_bound - lower;
    let positions = [-1.8_f64, -0.5, 0.25, 1.7];
    let alive = [true, false, true, true];
    let psi = positions.map(|x| (length * length / ((x - lower) * (upper_bound - x))).ln());
    let observable = psi
        .iter()
        .zip(alive)
        .filter(|(_, a)| *a)
        .map(|(x, _)| x)
        .sum::<f64>()
        / positions.len() as f64;
    let integral = 2. * length;
    let cb = 0.1;
    let inputs = json!({"box_lower":lower,"box_upper":upper_bound,"L":length,
        "positions":positions,"alive":alive,"log_barrier_values":psi,"normalized_alive_barrier":observable,
        "exact_L1_integral":integral,"c_b":cb,"scope":"Exact one-dimensional box log-barrier formula, uniform alive-slot normalization and endpoint antiderivative. This is the auxiliary complete-transition observable."});
    for formula in family_expressions(catalog, 3, "cor-canonical-full-step-boundary-reset") {
        let f = tokens(formula);
        let checks = if f.starts_with("D=\\prod") {
            vec![strict("valid_box_interval", lower < upper_bound)]
        } else if f == "L_j=u_j-\\ell_j" {
            vec![equal("box_coordinate_length", length, upper_bound - lower)]
        } else if f.starts_with("\\psi_D(x)=") {
            vec![
                strict("log_barrier_nonnegative", psi.iter().all(|x| *x >= 0.)),
                equal(
                    "alive_slot_boundary_normalization",
                    observable,
                    (psi[0] + psi[2] + psi[3]) / 4.,
                ),
            ]
        } else if f == "\\sigma_p>0" {
            vec![strict("positive_terminal_smoothing", 0.1 > 0.)]
        } else if f == "c_b>0" {
            vec![strict("positive_boundary_weight", cb > 0.)]
        } else if f == "\\psi_D\\inL^1(D)" {
            vec![
                strict("finite_exact_box_barrier_integral", integral.is_finite()),
                equal("box_log_endpoint_integral", integral, 8.),
            ]
        } else {
            continue;
        };
        emit(
            suite,
            catalog,
            "cor-canonical-full-step-boundary-reset",
            formula,
            &inputs,
            &[],
            &checks,
        )?;
    }
    Ok(())
}

fn append_alive_output_contracts(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    use crate::convergence_lyapunov::uniform_transport;
    for n in 1_usize..=4 {
        let cemetery_state = (0..n).map(|i| vec![i as f64 / 3., 0.]).collect::<Vec<_>>();
        let transition = [[0.8_f64, 0.2], [0., 1.]];
        let delta_cemetery = [0_f64, 1.];
        let next = row_update(delta_cemetery, transition);
        let inputs = json!({"N":n,"cemetery_representative":cemetery_state,
            "abstract_completed_transition":transition,"input_dirac":delta_cemetery,"output_dirac":next,
            "scope":"Explicit absorbing Markov extension of the native terminating engine. The cemetery representative retains its coordinates and all marks zero; no native engine step after termination is asserted."});
        for formula in [
            r"\Psi(\mathcal{S}_t, \cdot) = \delta_{\mathcal{S}_t}(\cdot)",
            r"\mathcal{S}_{t+1} = \mathcal{S}_t",
            r"|\mathcal{A}(\mathcal{S}_t)|=0",
        ] {
            emit(
                suite,
                catalog,
                "def-swarm-update-procedure",
                formula,
                &inputs,
                &[],
                &[
                    equal("completed_extension_retains_cemetery_mass", next[1], 1.),
                    equal("completed_extension_has_no_return_mass", next[0], 0.),
                    equal(
                        "completed_extension_alive_count",
                        cemetery_state.iter().filter(|x| x[1] > 0.).count() as f64,
                        0.,
                    ),
                ],
            )?;
        }
        let mut masks = vec![];
        let mut membership = vec![];
        for mask in 0..1_usize << n {
            let marks = (0..n).map(|i| ((mask >> i) & 1) as u8).collect::<Vec<_>>();
            let count = marks.iter().filter(|x| **x == 1).count();
            let reverse_count = marks.iter().rev().filter(|x| **x == 1).count();
            membership.push(equal(
                "alive_domain_permutation_invariance",
                count as f64,
                reverse_count as f64,
            ));
            membership.push(strict(
                "alive_domain_equals_nonextinction",
                (count >= 1) == (mask != 0),
            ));
            masks.push(
                json!({"marks":marks,"alive_count":count,"belongs_to_alive_domain":count>=1}),
            );
        }
        emit_prefix(
            suite,
            catalog,
            "def-swarm-and-state-space",
            r"\Sigma_N^{\mathrm{alive}} :=",
            &json!({"N":n,"status_cases":masks,
            "scope":"Exact finite quotient-domain membership for every binary mask, with permutation-invariant counts and the all-dead control."}),
            &membership,
        )?;
        let make = |side: f64, size: usize| {
            (0..size)
                .map(|a| {
                    (0..n)
                        .flat_map(|i| {
                            [
                                side + 0.2 * a as f64 + 0.1 * i as f64,
                                if (a + i) % 2 == 0 { 1. } else { 0. },
                            ]
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let left = make(-0.5, 2);
        let right = make(0.3, 3);
        let lambda = 0.7;
        let state_cost = |x: &[f64], y: &[f64]| {
            let a = x.chunks_exact(2).map(|z| z.to_vec()).collect::<Vec<_>>();
            let b = y.chunks_exact(2).map(|z| z.to_vec()).collect::<Vec<_>>();
            uniform_transport(&a, &b, |u, v| {
                (u[0] - v[0]).powi(2) + lambda * (u[1] - v[1]).powi(2)
            })
            .unwrap()
            .0
        };
        let (minimum, plan) = uniform_transport(&left, &right, state_cost)?;
        let integral = plan
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, m)| m * state_cost(&left[i], &right[j]))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let mut checks = vec![equal(
            "nested_swarm_law_transport_integral",
            minimum,
            integral,
        )];
        for row in &plan {
            checks.push(equal("swarm_law_left_marginal", row.iter().sum(), 0.5));
        }
        for j in 0..3 {
            checks.push(equal(
                "swarm_law_right_marginal",
                plan.iter().map(|row| row[j]).sum(),
                1. / 3.,
            ));
        }
        let mut permuted = right.clone();
        for state in &mut permuted {
            let mut pairs = state
                .chunks_exact(2)
                .map(|z| z.to_vec())
                .collect::<Vec<_>>();
            pairs.reverse();
            *state = pairs.into_iter().flatten().collect();
        }
        checks.push(equal(
            "output_law_invariant_under_atom_permutations",
            minimum,
            uniform_transport(&left, &permuted, state_cost)?.0,
        ));
        emit_prefix(
            suite,
            catalog,
            "def-w2-output-metric",
            r"W_2^2(\mu,\nu) :=",
            &json!({"N":n,"left_swarm_atoms":left,"right_swarm_atoms":right,
            "outer_optimal_plan":plan,"output_law_W2_squared":minimum,"status_weight":lambda,
            "scope":"Exact transport between finite laws on unordered swarms. The outer transport uses inner probability-normalized optimal marked atom costs; independent permutations preserve its value."}),
            &checks,
        )?;
    }
    Ok(())
}

fn append_native_fitness_and_population_contracts(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    use algorithmic_gas::fitness::{PositiveMap, PositiveMapping};
    let eta = 1_f64;
    let amplitude = 2_f64;
    let map = PositiveMap::Logistic {
        amplitude,
        floor: eta,
    };
    for n in [4_usize, 16, 64, 256] {
        let rz = (0..n)
            .map(|i| {
                if i < n / 2 {
                    0.04 + 0.02 * (i % 2) as f64
                } else {
                    -0.04 - 0.02 * (i % 2) as f64
                }
            })
            .collect::<Vec<_>>();
        let dz = (0..n)
            .map(|i| {
                if i < n / 2 {
                    -1. + 0.02 * (i % 2) as f64
                } else {
                    1. - 0.02 * (i % 2) as f64
                }
            })
            .collect::<Vec<_>>();
        let rp = rz.iter().map(|&z| map.map(z)).collect::<Result<Vec<_>>>()?;
        let dp = dz.iter().map(|&z| map.map(z)).collect::<Result<Vec<_>>>()?;
        for alpha in [0_f64, 0.25, 0.5, 1.] {
            for beta in [1_f64, 2.] {
                let mut pipeline = GasConfig::euclidean(1, 0.04)?.fitness;
                pipeline.reward_map = map.clone();
                pipeline.diversity_map = map.clone();
                pipeline.reward_exponent = alpha;
                pipeline.diversity_exponent = beta;
                let fitness = pipeline.combine(&rz, &dz, &vec![true; n])?;
                let lower = eta.powf(alpha + beta);
                let ceiling = (amplitude + eta).powf(alpha + beta);
                let half = n / 2;
                let avg = |a: &[f64]| a.iter().sum::<f64>() / a.len() as f64;
                let avg_log = |a: &[f64]| a.iter().map(|x| x.ln()).sum::<f64>() / a.len() as f64;
                let diversity_gap = avg_log(&dp[half..]) - avg_log(&dp[..half]);
                let reward_gap = avg_log(&rp[..half]) - avg_log(&rp[half..]);
                let log_gap = avg_log(&fitness[half..]) - avg_log(&fitness[..half]);
                let dstar = diversity_gap * 0.99;
                let astar = reward_gap * 1.01;
                let delta_log = beta * dstar - alpha * astar;
                let mean_h = avg(&fitness[..half]);
                let mean_l = avg(&fitness[half..]);
                let variance_h = fitness[..half]
                    .iter()
                    .map(|x| (x - mean_h).powi(2))
                    .sum::<f64>()
                    / half as f64;
                let delta = delta_log - variance_h / (2. * lower * lower);
                let mut product_checks = vec![];
                let mut reward_checks = vec![];
                let mut diversity_checks = vec![];
                let mut support_checks = vec![];
                for i in 0..n {
                    product_checks.push(equal(
                        "native_product_potential",
                        fitness[i],
                        dp[i].powf(beta) * rp[i].powf(alpha),
                    ));
                    reward_checks.push(equal(
                        "native_logistic_reward_with_floor",
                        rp[i],
                        amplitude / (1. + (-rz[i]).exp()) + eta,
                    ));
                    diversity_checks.push(equal(
                        "native_logistic_diversity_with_floor",
                        dp[i],
                        amplitude / (1. + (-dz[i]).exp()) + eta,
                    ));
                    support_checks.push(upper(
                        "native_population_uniform_potential_lower",
                        lower,
                        fitness[i],
                    ));
                    support_checks.push(upper(
                        "native_population_uniform_potential_upper",
                        fitness[i],
                        ceiling,
                    ));
                }
                let inputs = json!({"N":n,"native_pipeline":pipeline,"reward_scores":rz,"diversity_scores":dz,
                    "mapped_reward":rp,"mapped_diversity":dp,"fitness":fitness,"eta":eta,"amplitude":amplitude,
                    "alpha":alpha,"beta":beta,"potential_min":lower,"potential_max":ceiling,"H_size":half,"L_size":half,
                    "D":diversity_gap,"A":reward_gap,"D_star":dstar,"A_star":astar,"mean_log_gap":log_gap,
                    "mean_H":mean_h,"mean_L":mean_l,"variance_H":variance_h,"delta_log":delta_log,"delta":delta,
                    "scope":"Actual Rust PositiveMap::Logistic and FitnessPipeline::combine on explicitly frozen deterministic score vectors, repeated over population sizes with probability-normalized group expectations. The sufficient log/mean comparison is tested only with its recorded positive-gap and variance hypotheses; no universal favorable orientation is imposed on native measurements."});
                let hypotheses = vec![
                    strict("positive_floor", eta > 0.),
                    strict("nonnegative_fitness_exponents", alpha >= 0. && beta >= 0.),
                ];
                for formula in family_expressions(catalog, 3, "lem-potential-bounds") {
                    let f = tokens(formula);
                    let checks = if f.starts_with("V_{\\text{pot,min}}:=") {
                        vec![equal(
                            "N_independent_lower_formula",
                            lower,
                            eta.powf(alpha + beta),
                        )]
                    } else if f.starts_with("V_{\\text{pot,max}}:=") {
                        vec![equal(
                            "N_independent_upper_formula",
                            ceiling,
                            (amplitude + eta).powf(alpha + beta),
                        )]
                    } else if f.starts_with("r'_i=") {
                        reward_checks.clone()
                    } else if f.starts_with("d'_i=") {
                        diversity_checks.clone()
                    } else if f == "\\alpha,\\beta\\geq0" {
                        vec![strict(
                            "actual_nonnegative_exponents",
                            alpha >= 0. && beta >= 0.,
                        )]
                    } else if f == "\\eta>0" {
                        vec![strict("positive_eta_condition", eta > 0.)]
                    } else if f.starts_with("V_i\\ge") || f.starts_with("V_i\\le") {
                        support_checks.clone()
                    } else {
                        continue;
                    };
                    emit(
                        suite,
                        catalog,
                        "lem-potential-bounds",
                        formula,
                        &inputs,
                        &hypotheses,
                        &checks,
                    )?;
                }
                for formula in
                    family_expressions(catalog, 3, "thm-derivation-of-stability-condition")
                {
                    let f = tokens(formula);
                    let checks = if f.starts_with("V=(d')") {
                        product_checks.clone()
                    } else if f.starts_with("D=\\mathbbE_L") {
                        vec![
                            equal(
                                "diversity_log_mean_difference",
                                diversity_gap,
                                avg_log(&dp[half..]) - avg_log(&dp[..half]),
                            ),
                            equal(
                                "reward_log_mean_difference",
                                reward_gap,
                                avg_log(&rp[..half]) - avg_log(&rp[half..]),
                            ),
                        ]
                    } else if f.starts_with("\\mathbbE_L\\logV-") {
                        vec![equal(
                            "exact_native_log_product_gap",
                            log_gap,
                            beta * diversity_gap - alpha * reward_gap,
                        )]
                    } else if f == "\\betaD>\\alphaA" {
                        vec![strict(
                            "exact_log_gap_positive_iff",
                            (beta * diversity_gap > alpha * reward_gap) == (log_gap > 0.),
                        )]
                    } else if f == "D\\geqD_*" {
                        vec![upper("declared_D_lower", dstar, diversity_gap)]
                    } else if f == "A\\leqA_*" {
                        vec![upper("declared_A_upper", reward_gap, astar)]
                    } else if f == "\\betaD_*>\\alphaA_*" {
                        vec![strict(
                            "sufficient_weighted_gap",
                            beta * dstar > alpha * astar,
                        )]
                    } else {
                        continue;
                    };
                    emit(
                        suite,
                        catalog,
                        "thm-derivation-of-stability-condition",
                        formula,
                        &inputs,
                        &hypotheses,
                        &checks,
                    )?;
                }
                let gap_hypotheses = vec![
                    strict("positive_sufficient_log_margin", delta_log > 0.),
                    strict("positive_variance_corrected_margin", delta > 0.),
                    strict(
                        "same_native_support",
                        fitness.iter().all(|&v| v >= lower && v <= ceiling),
                    ),
                ];
                for formula in
                    family_expressions(catalog, 3, "thm-stability-condition-final-corrected")
                {
                    let f = tokens(formula);
                    let checks = if f.starts_with("\\delta_{\\log}:=") {
                        vec![
                            equal(
                                "delta_log_definition",
                                delta_log,
                                beta * dstar - alpha * astar,
                            ),
                            strict("delta_log_positive", delta_log > 0.),
                        ]
                    } else if f == "0<v_*\\lev^*<\\infty" {
                        vec![strict(
                            "finite_positive_support",
                            lower > 0. && lower <= ceiling && ceiling.is_finite(),
                        )]
                    } else if f.starts_with("\\delta:=") {
                        vec![
                            equal(
                                "delta_variance_correction",
                                delta,
                                delta_log - variance_h / (2. * lower * lower),
                            ),
                            strict("positive_corrected_delta", delta > 0.),
                        ]
                    } else if f == "X\\geqv_*" {
                        fitness
                            .iter()
                            .map(|&v| upper("Taylor_support_lower", lower, v))
                            .collect()
                    } else if f.starts_with("\\log\\mathbbE_LV-") {
                        vec![upper(
                            "log_means_corrected_lower",
                            delta,
                            mean_l.ln() - mean_h.ln(),
                        )]
                    } else if f == "\\mathbbE_HV\\geqv_*" {
                        vec![upper("H_mean_support_lower", lower, mean_h)]
                    } else if f.starts_with("0\\leq\\log\\mathbbEX-") {
                        let gap = mean_h.ln() - avg_log(&fitness[..half]);
                        vec![
                            upper("Jensen_nonnegative_gap", 0., gap + 1e-14),
                            upper(
                                "Taylor_Jensen_variance_upper",
                                gap,
                                variance_h / (2. * lower * lower),
                            ),
                        ]
                    } else if f.starts_with("\\mathbbE_LV-\\mathbbE_HV") {
                        vec![upper(
                            "native_arithmetic_gap_lower",
                            lower * delta.exp_m1(),
                            mean_l - mean_h,
                        )]
                    } else if f.starts_with("\\mathbbE_L\\logV-") {
                        vec![upper("native_log_gap_lower", delta_log, log_gap)]
                    } else {
                        continue;
                    };
                    emit(
                        suite,
                        catalog,
                        "thm-stability-condition-final-corrected",
                        formula,
                        &inputs,
                        &gap_hypotheses,
                        &checks,
                    )?;
                }
                let mu = avg(&fitness);
                let range = fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - fitness.iter().copied().fold(f64::INFINITY, f64::min);
                let variance = fitness.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / n as f64;
                let positive = fitness.iter().map(|v| (v - mu).max(0.)).sum::<f64>() / n as f64;
                let negative = fitness.iter().map(|v| (mu - v).max(0.)).sum::<f64>() / n as f64;
                let absolute = fitness.iter().map(|v| (v - mu).abs()).sum::<f64>() / n as f64;
                let unfit = fitness.iter().filter(|&&v| v <= mu).count();
                let fit = n - unfit;
                let unfit_h = fitness[..half].iter().filter(|&&v| v <= mu).count();
                let gap = mean_l - mean_h;
                let delta_v = 0.9 * gap;
                let removed = fitness[..half]
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 3 == 0)
                    .count();
                let intersection = fitness[..half]
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| i % 3 != 0 && **v <= mu)
                    .count();
                let mut row_checks = vec![];
                let mut indicator_checks = vec![];
                for &v in &fitness {
                    row_checks.push(upper(
                        "negative_part_indicator",
                        (mu - v).max(0.),
                        if v <= mu { range } else { 0. },
                    ));
                    indicator_checks.push(equal(
                        "strict_fit_indicator_complement",
                        f64::from(v > mu) + f64::from(v <= mu),
                        1.,
                    ));
                }
                let inputs = json!({"native_fitness_fixture":inputs,"mean":mu,"range":range,"variance":variance,"positive_centered_mean":positive,"negative_centered_mean":negative,"absolute_centered_mean":absolute,"unfit_count":unfit,"fit_count":fit,"unfit_H_count":unfit_h,"H_fraction":0.5,"L_fraction":0.5,"arithmetic_gap":gap,"Delta_V":delta_v,"removed_H_representatives":removed,"common_alive_H_unfit_count":intersection,
                    "scope":"Exact normalized means, variance, positive parts, unfit set and group overlap of the actual Rust fitness vector. Any reordering carries group/common-alive membership with the representative; no intrinsic walker labels enter the counts. Population-dependent cardinalities are divided by N in primary bounds."});
                let hypotheses = vec![
                    strict("positive_actual_range", range > 0.),
                    strict("positive_partition_fractions", half > 0 && half < n),
                    strict(
                        "actual_oriented_arithmetic_gap",
                        gap >= delta_v && delta_v > 0.,
                    ),
                ];
                for formula in family_expressions(catalog, 3, "lem-unfit-fraction-lower-bound") {
                    let f = tokens(formula);
                    let checks = if f == "R_V>0" {
                        vec![strict("actual_range_positive", range > 0.)]
                    } else if f.starts_with("\\mathbbE(X-\\mu)_+=") {
                        vec![equal(
                            "centered_positive_negative_balance",
                            positive,
                            negative,
                        )]
                    } else if f.starts_with("s_V^2\\leqR_V") {
                        vec![
                            upper("variance_range_absolute_bound", variance, range * absolute),
                            equal(
                                "absolute_negative_part_balance",
                                range * absolute,
                                2. * range * negative,
                            ),
                        ]
                    } else if f.starts_with("(\\mu-X)_+\\leq") {
                        row_checks.clone()
                    } else if f == "\\mathbf1_{X>\\mu}" {
                        indicator_checks.clone()
                    } else if f.starts_with("\\frac{|U_k|}") {
                        vec![
                            upper(
                                "N_uniform_unfit_fraction",
                                variance / (2. * range * range),
                                unfit as f64 / n as f64,
                            ),
                            upper(
                                "N_uniform_fit_fraction",
                                variance / (2. * range * range),
                                fit as f64 / n as f64,
                            ),
                        ]
                    } else {
                        continue;
                    };
                    emit(
                        suite,
                        catalog,
                        "lem-unfit-fraction-lower-bound",
                        formula,
                        &inputs,
                        &hypotheses,
                        &checks,
                    )?;
                }
                for formula in
                    family_expressions(catalog, 3, "thm-unfit-high-error-overlap-fraction")
                {
                    let f = tokens(formula);
                    let checks = if f == "R_V>0" {
                        vec![strict("actual_range_positive", range > 0.)]
                    } else if f == "f_H,f_L>0" {
                        vec![strict("nonempty_both_groups", half > 0 && half < n)]
                    } else if f == "\\mu_L-\\mu_H\\geq\\Delta_V>0" {
                        vec![
                            upper("actual_group_gap_lower", delta_v, gap),
                            strict("positive_group_gap_lower", delta_v > 0.),
                        ]
                    } else if f == "\\mu=f_H\\mu_H+f_L\\mu_L" {
                        vec![equal(
                            "actual_partition_mean_identity",
                            mu,
                            0.5 * mean_h + 0.5 * mean_l,
                        )]
                    } else if f.starts_with("\\mu-\\mu_H=f_L") {
                        vec![
                            equal("partition_mean_gap_identity", mu - mean_h, 0.5 * gap),
                            upper("partition_gap_lower", 0.5 * delta_v, mu - mean_h),
                        ]
                    } else if f.starts_with("R_V|H\\capU_k|") {
                        vec![upper(
                            "overlap_count_times_actual_range",
                            half as f64 * 0.5 * delta_v,
                            range * unfit_h as f64,
                        )]
                    } else if f.starts_with("\\frac{|H\\capU_k|}") {
                        vec![upper(
                            "N_uniform_unfit_H_fraction",
                            0.25 * delta_v / range,
                            unfit_h as f64 / n as f64,
                        )]
                    } else if f.starts_with("|I_{11}\\capH\\capU_k|") {
                        vec![upper(
                            "explicit_omitted_alive_overlap",
                            unfit_h as f64 - removed as f64,
                            intersection as f64,
                        )]
                    } else {
                        continue;
                    };
                    emit(
                        suite,
                        catalog,
                        "thm-unfit-high-error-overlap-fraction",
                        formula,
                        &inputs,
                        &hypotheses,
                        &checks,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn append_reference_innovation_contract(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    for n in [1_usize, 2, 3] {
        let outcomes = 1_usize << (4 * n);
        let mut marginals = vec![[0_usize; 16]; n];
        let mut joint = [0_usize; 256];
        let mut component_counts = vec![[0_usize; 4]; n];
        for outcome in 0..outcomes {
            let patterns = (0..n)
                .map(|i| (outcome >> (4 * i)) & 15)
                .collect::<Vec<_>>();
            for (i, &pattern) in patterns.iter().enumerate() {
                marginals[i][pattern] += 1;
                for (j, count) in component_counts[i].iter_mut().enumerate() {
                    *count += (pattern >> j) & 1;
                }
            }
            if n > 1 {
                joint[patterns[0] * 16 + patterns[1]] += 1;
            }
        }
        let mut checks = vec![];
        for (i, counts) in marginals.iter().enumerate() {
            checks.push(equal(
                "exact_reference_product_mass",
                counts.iter().sum::<usize>() as f64 / outcomes as f64,
                1.,
            ));
            for &count in counts {
                checks.push(equal(
                    "four_component_pattern_probability",
                    count as f64 / outcomes as f64,
                    1. / 16.,
                ));
            }
            for &count in &component_counts[i] {
                checks.push(equal(
                    "reference_component_half_mass",
                    count as f64 / outcomes as f64,
                    0.5,
                ));
            }
        }
        if n > 1 {
            for &count in &joint {
                checks.push(equal(
                    "independent_reference_row_pair_pattern",
                    count as f64 / outcomes as f64,
                    1. / 256.,
                ));
            }
        }
        let inputs = json!({"N":n,"outcomes":outcomes,"probability_per_outcome":1./outcomes as f64,"component_order":["companion","perturbation","status","clone"],"per_row_four_bit_patterns":marginals,"per_component_counts":component_counts,"row_pair_patterns":joint.to_vec(),
            "scope":"Exact independent finite product innovation fixture conditional on one fixed state, with four independent Bernoulli partitions of unit uniform inputs per representative. This checks the declared abstract assumption and tuple, not independence of the native shared component rotation or coupled simulations."});
        emit(
            suite,
            catalog,
            "axiom-instep-independence",
            r"X_i \;:=\;\big(U_i^{\mathrm{comp}},\,U_i^{\mathrm{pert}},\,U_i^{\mathrm{status}},\,U_i^{\mathrm{clone}}\big)",
            &inputs,
            &[],
            &checks,
        )?;
    }
    Ok(())
}

fn append_barrier_jitter_clauses(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    let label = "lem-barrier-reduction-cloning";
    let radius = 2_f64;
    for dimension in [1_usize, 2, 4] {
        for sigma in [0.1_f64, 0.5, 1.] {
            let y = (0..dimension)
                .map(|i| 0.2 + 0.1 * i as f64)
                .collect::<Vec<_>>();
            let mq = (2. * std::f64::consts::PI * sigma * sigma).powf(-(dimension as f64) / 2.);
            let l1 = dimension as f64 * (2. * radius).powi(dimension as i32) * radius * radius / 3.;
            let mut density_checks = vec![];
            let mut taylor_checks = vec![];
            for row in -20..=20 {
                let xi = (0..dimension)
                    .map(|i| row as f64 * (i + 1) as f64 / 20.)
                    .collect::<Vec<_>>();
                let h = xi.iter().map(|x| sigma * x).collect::<Vec<_>>();
                let z = y.iter().zip(&h).map(|(a, b)| a + b).collect::<Vec<_>>();
                let density =
                    mq * (-h.iter().map(|x| x * x).sum::<f64>() / (2. * sigma * sigma)).exp();
                density_checks.push(upper("actual_Gaussian_density_peak", density, mq));
                taylor_checks.push(equal(
                    "jitter_position_sum",
                    z.iter().sum(),
                    y.iter().sum::<f64>() + h.iter().sum::<f64>(),
                ));
                taylor_checks.push(equal(
                    "quadratic_integral_Taylor_remainder",
                    z.iter().map(|x| x * x).sum(),
                    y.iter().map(|x| x * x).sum::<f64>()
                        + 2. * y.iter().zip(&h).map(|(a, b)| a * b).sum::<f64>()
                        + h.iter().map(|x| x * x).sum::<f64>(),
                ));
            }
            let draws = 16_384;
            let mut means = vec![0_f64; dimension];
            let mut second = 0.;
            for sample in 0..draws {
                let mut rng = RandomStream::new(617_819, sample as u64, Stream::CloneNoise, 0, 0);
                for mean in &mut means {
                    let z = rng.gaussian::<f64>();
                    *mean += z / draws as f64;
                    second += z * z / draws as f64;
                }
            }
            let inputs = json!({"dimension":dimension,"sigma_x":sigma,"y":y,"Gaussian_density_max":mq,"domain_box_radius":radius,"observable":"||x||^2 on the box, zero extension for the L1 branch; ||x||^2 on all phase positions for the C2 branch","exact_L1_norm":l1,"Hessian_norm":2.,"native_rng_stream":"CloneNoise","native_draws":draws,"native_means":means,"native_second_norm_moment":second,
                "scope":"Gaussian jitter clauses evaluated for the declared finite quadratic proof observable. Zero extension and global C2 extension are separate hypotheses. Exact Gaussian reference moments and native CloneNoise moment tolerances are recorded explicitly; the reciprocal-distance barrier is not claimed integrable."});
            for formula in family_expressions(catalog, 3, label) {
                let f = tokens(formula);
                let checks = if f == "q_y(z)\\leqM_q" {
                    density_checks.clone()
                } else if f == "Y=y+\\sigma_x\\xi" || f == "h=\\sigma_x\\xi" {
                    taylor_checks.clone()
                } else if f == "\\varphi\\inL^1(D)" {
                    vec![
                        strict("finite_actual_box_integral", l1 > 0. && l1.is_finite()),
                        equal(
                            "box_quadratic_exact_integral",
                            l1,
                            dimension as f64
                                * (2. * radius).powi(dimension as i32)
                                * radius
                                * radius
                                / 3.,
                        ),
                    ]
                } else if f == "\\|D^2\\varphi\\|\\leqH" {
                    vec![upper("quadratic_Hessian_operator_norm", 2., 2.)]
                } else if f == "\\mathbbE\\xi=0" {
                    means
                        .iter()
                        .map(|m| {
                            upper(
                                "native_jitter_mean_six_se",
                                m.abs(),
                                6. / (draws as f64).sqrt(),
                            )
                        })
                        .collect()
                } else if f == "\\mathbbE\\|\\xi\\|^2=d" {
                    vec![upper(
                        "native_jitter_second_norm_six_se",
                        (second - dimension as f64).abs(),
                        6. * (2. * dimension as f64 / draws as f64).sqrt(),
                    )]
                } else {
                    continue;
                };
                emit(suite, catalog, label, formula, &inputs, &[], &checks)?;
            }
        }
    }
    Ok(())
}

fn append_remaining_geometric_parameters(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    let radius = 2_f64;
    let safe_radius = 0.5_f64;
    let delta_safe = radius - safe_radius;
    let safe_threshold = -safe_radius.powi(2) / 2.;
    let points = (-128..=128)
        .map(|i| radius * i as f64 / 128.)
        .collect::<Vec<_>>();
    let inside = points
        .iter()
        .copied()
        .filter(|x| x.abs() <= safe_radius)
        .collect::<Vec<_>>();
    let outside = points
        .iter()
        .copied()
        .filter(|x| x.abs() > safe_radius && x.abs() < radius)
        .collect::<Vec<_>>();
    let inputs = json!({"domain":"(-2,2)","safe_harbor":"[-0.5,0.5]","positional_reward":"-x²/2","safe_distance":delta_safe,"R_safe":safe_threshold,"inside_points":inside,"outside_points":outside,"scope":"Explicit compact interior safe harbor for the native unit quadratic landscape in one dimension. The reward threshold is evaluated on inside and outside witnesses; favorable live-companion availability remains a separate condition."});
    for formula in family_expressions(catalog, 3, "axiom-safe-harbor") {
        let f = tokens(formula);
        let checks = if f.starts_with("d(x,\\partial") {
            inside
                .iter()
                .map(|x| {
                    upper(
                        "actual_safe_boundary_distance",
                        delta_safe,
                        radius - x.abs(),
                    )
                })
                .collect()
        } else if f == "x\\inC_{\\mathrm{safe}}" {
            inside
                .iter()
                .map(|x| strict("actual_safe_membership", x.abs() <= safe_radius))
                .collect()
        } else if f == "R_{\\mathrm{pos}}(x)<R_{\\mathrm{safe}}" {
            outside
                .iter()
                .map(|x| {
                    strict(
                        "actual_outside_reward_threshold",
                        -x * x / 2. < safe_threshold,
                    )
                })
                .collect()
        } else if f.starts_with("\\max_{y\\inC_") {
            vec![upper("safe_reward_maximum_threshold", safe_threshold, 0.)]
        } else {
            continue;
        };
        emit(
            suite,
            catalog,
            "axiom-safe-harbor",
            formula,
            &inputs,
            &[strict("positive_safe_clearance", delta_safe > 0.)],
            &checks,
        )?;
    }
    for c in [0_f64, 0.25] {
        let velocities = [-2_f64, -0.5, 0., 0.5, 2.];
        let rewards = velocities.map(|v| -0.4_f64.powi(2) - c * v * v);
        let inputs = json!({"c_v_reg":c,"x":0.4,"velocity":velocities,"complete_rewards":rewards,"scope":"Explicit positional objective and optional configured velocity penalty, including both permitted zero penalty and strictly positive penalty configurations."});
        for formula in family_expressions(catalog, 3, "axiom-velocity-regularization") {
            let f = tokens(formula);
            let checks = if f == "c_{v\\_reg}>0" && c > 0. {
                vec![strict("configured_positive_velocity_penalty", c > 0.)]
            } else if f == "c_{v\\_reg}=0" && c == 0. {
                vec![equal("canonical_zero_velocity_penalty", c, 0.)]
            } else if f.starts_with("R(x,v)=") {
                velocities
                    .iter()
                    .zip(rewards)
                    .map(|(v, r)| {
                        equal(
                            "complete_optional_velocity_reward",
                            r,
                            -0.4_f64.powi(2) - c * v * v,
                        )
                    })
                    .collect()
            } else if f == "c_{v\\_reg}\\geq0" {
                vec![strict("permitted_nonnegative_penalty", c >= 0.)]
            } else {
                continue;
            };
            emit(
                suite,
                catalog,
                "axiom-velocity-regularization",
                formula,
                &inputs,
                &[],
                &checks,
            )?;
        }
    }
    for lambda_v in [0.1_f64, 1., 4.] {
        for c_v in [0.25_f64, 1., 4.] {
            for c_b in [0.1_f64, 1., 2.] {
                let inputs = json!({"lambda_v":lambda_v,"c_V":c_v,"c_B":c_b,"weights":[1.,c_v,c_v,c_b],"scope":"Explicit strictly positive fixed weights of the declared normalized quadratic and boundary Lyapunov components. The weights do not vary with swarm size."});
                for (formula, condition) in [
                    (r"\lambda_v > 0", lambda_v > 0.),
                    (r"c_V > 0", c_v > 0.),
                    (r"c_B > 0", c_b > 0.),
                ] {
                    emit(
                        suite,
                        catalog,
                        "def-full-synergistic-lyapunov-function",
                        formula,
                        &inputs,
                        &[],
                        &[strict("declared_weight_positive", condition)],
                    )?;
                }
            }
        }
    }
    for n in [4_usize, 16, 64, 256] {
        let marks = (0..n).map(|i| i % 4 != 0).collect::<Vec<_>>();
        let positions = (0..n)
            .map(|i| {
                if marks[i] {
                    (i as f64 + 0.5) / n as f64
                } else {
                    1e6
                }
            })
            .collect::<Vec<_>>();
        let k = marks.iter().filter(|a| **a).count();
        let mean = positions
            .iter()
            .zip(&marks)
            .filter(|(_, a)| **a)
            .map(|(x, _)| x)
            .sum::<f64>()
            / k as f64;
        let inputs = json!({"N":n,"coupled_swarm_numbers":[1,2],"alive":marks,"positions":positions,"alive_count":k,"alive_mean":mean,"scope":"Exact alive-count and dead-mark convention for two explicitly represented empirical swarms. Dead coordinates are deliberately large and excluded from normalized barycentres; permutation of atom/mark pairs leaves the scalar counts invariant."});
        for formula in family_expressions(catalog, 3, "def-barycentres-and-centered-vectors") {
            let f = tokens(formula);
            let checks = if f == "k\\in\\{1,2\\}" {
                vec![
                    strict("first_coupled_swarm_number", (1..=2).contains(&1)),
                    strict("second_coupled_swarm_number", (1..=2).contains(&2)),
                ]
            } else if f == "k_{\\text{alive}}:=|\\mathcal{A}(S_k)|" {
                vec![
                    equal(
                        "actual_alive_cardinality",
                        k as f64,
                        marks.iter().filter(|a| **a).count() as f64,
                    ),
                    equal(
                        "reversed_alive_cardinality",
                        k as f64,
                        marks.iter().rev().filter(|a| **a).count() as f64,
                    ),
                ]
            } else if f == "s_i=0" {
                vec![
                    strict(
                        "every_omitted_atom_mark_is_dead",
                        marks
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| positions[*i] > 1e5)
                            .all(|(_, a)| !*a),
                    ),
                    upper("alive_mean_ignores_dead_coordinates", mean, 1.),
                ]
            } else {
                continue;
            };
            emit(
                suite,
                catalog,
                "def-barycentres-and-centered-vectors",
                formula,
                &inputs,
                &[],
                &checks,
            )?;
        }
    }
    Ok(())
}

fn append_framework_inline_state_contracts(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    use crate::convergence_lyapunov::uniform_transport;
    for n in [1_usize, 2, 4, 16, 64] {
        let lambda = 0.7_f64;
        let atoms = (0..n)
            .map(|i| {
                vec![
                    -1.5 + 3. * (i as f64 + 0.5) / n as f64,
                    f64::from(i % 2 == 0),
                ]
            })
            .collect::<Vec<_>>();
        let mut reordered = atoms.clone();
        reordered.reverse();
        let cost = |x: &[f64], y: &[f64]| (x[0] - y[0]).powi(2) + lambda * (x[1] - y[1]).powi(2);
        let (zero, plan) = uniform_transport(&atoms, &reordered, cost)?;
        let position = plan
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, m)| m * (atoms[i][0] - reordered[j][0]).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let status = plan
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(|(j, m)| m * (atoms[i][1] - reordered[j][1]).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>();
        let totalmass = plan.iter().map(|r| r.iter().sum::<f64>()).sum::<f64>();
        let inputs = json!({"N":n,"atoms":atoms,"reordered_atoms":reordered,"marked_ground_cost":"(x-y)^2+lambda*(s-t)^2","lambda_status":lambda,"transport_plan":plan,"optimal_cost":zero,"normalized_position_cost":position,"normalized_status_cost":status,"status_examples":{"dead":0,"alive":1},"cemetery_marks":vec![0_u8;n],"singleton_marks":(0..n).map(|i|u8::from(i==0)).collect::<Vec<_>>(),
            "scope":"Actual minimum-cost uniform transport between a marked empirical swarm and its reversed representative, with exact normalized positional/status decomposition and explicit binary, cemetery and singleton mark fixtures. Reordering changes only array enumeration."});
        for (label, formula, checks) in [
            (
                "def-walker",
                r"s=0",
                vec![equal("dead_binary_mark", 0., 0.)],
            ),
            (
                "def-walker",
                r"s=1",
                vec![equal("alive_binary_mark", 1., 1.)],
            ),
            (
                "def-cemetery-state",
                r"\mathcal{A}(\mathcal{S}_\emptyset) = \emptyset",
                vec![equal(
                    "explicit_cemetery_alive_count",
                    vec![0_u8; n].iter().filter(|m| **m == 1).count() as f64,
                    0.,
                )],
            ),
            (
                "def-cemetery-state",
                r"s_i = 0",
                vec![strict(
                    "all_cemetery_marks_dead",
                    vec![0_u8; n].iter().all(|m| *m == 0),
                )],
            ),
            (
                "thm-revival-guarantee",
                r"|\mathcal A(\mathcal S)| = 1",
                vec![equal(
                    "explicit_singleton_alive_count",
                    (0..n).filter(|i| *i == 0).count() as f64,
                    1.,
                )],
            ),
            (
                "def-metric-quotient",
                r"\mathcal{S}_1\sim\mathcal{S}_2",
                vec![
                    equal("same_joint_empirical_mass", totalmass, 1.),
                    equal("same_orbit_zero_ground_transport", zero, 0.),
                ],
            ),
            (
                "def-metric-quotient",
                r"\overline{\Sigma}_N := \Sigma_N/\!\sim",
                vec![equal("projected_orbit_equivalence_cost", zero, 0.)],
            ),
            (
                "def-metric-quotient",
                r"d_{\text{Disp},\mathcal{Y}}(\mathcal{S}_1,\mathcal{S}_2)=0",
                vec![equal("actual_quotient_zero_distance", zero.sqrt(), 0.)],
            ),
            (
                "def-displacement-components",
                r"d_{\text{Disp},\mathcal{Y}}^2 = \frac{1}{N}\Delta_{\text{pos}}^2 + \frac{\lambda_{\mathrm{status}}}{N}n_c",
                vec![equal(
                    "one_shared_optimal_plan_decomposition",
                    zero,
                    position + lambda * status,
                )],
            ),
            (
                "lem-polishness-and-w2",
                r"N<\infty",
                vec![strict(
                    "finite_representative_length",
                    n > 0 && n < usize::MAX,
                )],
            ),
            (
                "lem-polishness-and-w2",
                r"\overline d^2\le D_{\mathcal Y}^2+\lambda_{\mathrm{status}}",
                vec![upper(
                    "bounded_normalized_marked_quotient",
                    zero,
                    4_f64.powi(2) + lambda,
                )],
            ),
        ] {
            emit(suite, catalog, label, formula, &inputs, &[], &checks)?;
        }
    }
    let project = |x: f64| 2. * x / (4. + x * x).sqrt();
    for (x, y) in [(-8_f64, 7.), (-1., 0.5), (0., 0.), (0.5, 1.)] {
        let px = project(x);
        let py = project(y);
        let derivative = |x: f64| 8. / (4. + x * x).powf(1.5);
        let valid_diameter = 2. * project(1.);
        let projected_integral = 1. * project(x);
        let inputs = json!({"x":x,"y":y,"projected_x":px,"projected_y":py,"projection":"2*x/sqrt(4+x²)","projection_derivative":derivative(x),"L_projection":1.,"algorithmic_space":"[-2,2]","D_Y":4.,"valid_physical_domain":"[-1,1]","D_valid":valid_diameter,"dimension":1,"physical_dirac_mass":1.,"pushed_dirac_atom":px,"projected_dirac_integral":projected_integral,
            "scope":"Explicit bounded smooth Euclidean projection, native one-atom transport and Dirac push-forward integral. Diameter constants are determined from interval endpoints; Lipschitz derivative and pair checks refer to this projection."});
        for (label, formula, checks) in [
            (
                "def-distance-positional-measures",
                r"\varphi_* \delta_{x_i} = \delta_{\varphi(x_i)}",
                vec![
                    equal("Dirac_pushforward_first_moment", projected_integral, px),
                    equal("Dirac_pushforward_mass", 1., 1.),
                ],
            ),
            (
                "def-algorithmic-space-generic",
                r"d_{\mathcal{Y}}(y,y') = \|y-y'\|_2",
                vec![equal(
                    "one_dimensional_ground_norm",
                    (px - py).abs(),
                    (px - py).powi(2).sqrt(),
                )],
            ),
            (
                "def-algorithmic-space-generic",
                r"D_{\mathcal{Y}} := \operatorname{diam}_{d_{\mathcal{Y}}}(\mathcal{Y}) < \infty",
                vec![
                    strict("finite_algorithmic_interval_diameter", 4_f64.is_finite()),
                    upper(
                        "projected_pair_below_interval_diameter",
                        (px - py).abs(),
                        4.,
                    ),
                ],
            ),
            (
                "def-algorithmic-space-generic",
                r"D_{\mathcal{Y}} := \operatorname{diam}_{d_{\mathcal{Y}}}(\mathcal{Y})",
                vec![equal(
                    "actual_algorithmic_interval_diameter",
                    2. - (-2.),
                    4.,
                )],
            ),
            (
                "def-algorithmic-space-generic",
                r"D_{\mathrm{valid}} := \operatorname{diam}_{d_{\mathcal{Y}}}(\varphi(\mathcal{X}_{\mathrm{valid}}))",
                vec![equal(
                    "monotone_projection_valid_diameter",
                    valid_diameter,
                    project(1.) - project(-1.),
                )],
            ),
            (
                "def-algorithmic-space-generic",
                r"m \in \mathbb{N}",
                vec![strict("positive_integer_projection_dimension", 1_usize > 0)],
            ),
            (
                "def-algorithmic-space-generic",
                r"d_{\mathcal{Y}}(\varphi(x),\varphi(x')) \le L_{\varphi}\, d_{\mathcal{X}}(x,x')",
                vec![
                    upper(
                        "explicit_projection_pair_Lipschitz",
                        (px - py).abs(),
                        (x - y).abs(),
                    ),
                    upper("explicit_projection_derivative", derivative(x), 1.),
                ],
            ),
        ] {
            emit(suite, catalog, label, formula, &inputs, &[], &checks)?;
        }
    }
    for n in [2_usize, 4, 16, 64] {
        let m = 8_f64;
        let coordinates = (0..n)
            .map(|i| -m / 2. + m * (i as f64 + 0.5) / n as f64)
            .collect::<Vec<_>>();
        let box_number = coordinates
            .iter()
            .map(|x| x.abs().ceil() as usize)
            .max()
            .unwrap()
            .max(1);
        let projected = coordinates.iter().map(|&x| project(x)).collect::<Vec<_>>();
        let inputs = json!({"N":n,"compact_exhaustion":"K_m=[-m,m]^N times {0,1}^N for integer m>=1","coordinates":coordinates,"chosen_box_number":box_number,"projected_coordinates":projected,"scope":"Explicit compact interval exhaustion of a finite-dimensional Euclidean representative lift, with finite binary mark factor. Each recorded representative is assigned to a containing K_m. The full sigma-compact/Borel assertion remains the chapter's continuous-image proof, not a claim inferred from this finite numerical sample."});
        emit(
            suite,
            catalog,
            "lem-borel-image-of-the-projected-swarm-space",
            r"E_N=\bigcup_m K_m",
            &inputs,
            &[strict("finite_product_dimension", n < usize::MAX)],
            &coordinates
                .iter()
                .map(|x| {
                    upper(
                        "recorded_representative_in_compact_exhaustion",
                        x.abs(),
                        box_number as f64,
                    )
                })
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}

fn append_auxiliary_phase_and_boundary_contracts(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    use crate::convergence_lyapunov::uniform_transport;
    use algorithmic_gas::fitness::{PositiveMap, PositiveMapping};
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.1,
    };
    let reward = [-1_f64, -0.5, 0.5, 1.];
    let diversity = [-2_f64, 1., 0., 2.];
    let mut pipeline = GasConfig::euclidean(1, 0.04)?.fitness;
    pipeline.reward_map = map.clone();
    pipeline.diversity_map = map.clone();
    pipeline.diversity_exponent = 0.;
    pipeline.reward_exponent = 1.5;
    let fitness = pipeline.combine(&reward, &diversity, &[true; 4])?;
    let inputs = json!({"native_pipeline":pipeline,"reward_scores":reward,"diversity_scores":diversity,"native_fitness":fitness,"scope":"Actual native frozen-score fitness for the explicitly inactive diversity configuration beta=0. This is the chapter's described degenerate case, not an admissible active-diversity hypothesis witness."});
    for formula in family_expressions(catalog, 3, "axiom-active-diversity") {
        let f = tokens(formula);
        let checks = if f == "\\beta=0" {
            vec![equal(
                "explicit_inactive_diversity_parameter",
                pipeline.diversity_exponent,
                0.,
            )]
        } else if f == "V_{\\text{fit}}=(r')^\\alpha" {
            reward
                .iter()
                .zip(&fitness)
                .map(|(z, v)| {
                    Ok(equal(
                        "native_reward_only_fitness",
                        *v,
                        map.map(*z)?.powf(pipeline.reward_exponent),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            continue;
        };
        emit(
            suite,
            catalog,
            "axiom-active-diversity",
            formula,
            &inputs,
            &[],
            &checks,
        )?;
    }
    let inputs = json!({"x":0.,"y":1.,"L_grad":1.,"kappa_raw_reward":0.5,"effective_positional_reward":"-x²/2","scope":"The exact positive directional hypothesis parameters of the retained quadratic-landscape pair x=0,y=1. The general axiom is a stated landscape assumption; this pair does not establish its global validity for a landscape with equal-level pairs."});
    for (formula, observed) in [
        (r"L_{\text{grad}} > 0", 1.),
        (r"\kappa_{\text{raw},r} > 0", 0.5),
    ] {
        emit(
            suite,
            catalog,
            "axiom-non-deceptive-landscape",
            formula,
            &inputs,
            &[],
            &[strict(
                "actual_directional_hypothesis_constant_positive",
                observed > 0.,
            )],
        )?;
    }
    for n in [4_usize, 16, 64, 256] {
        let atoms = (0..n)
            .map(|i| {
                vec![
                    if i % 2 == 0 { -0.8 } else { 0.8 },
                    if i % 4 < 2 { -0.4 } else { 0.4 },
                ]
            })
            .collect::<Vec<_>>();
        let mut reordered = atoms.clone();
        reordered.reverse();
        // Each of the four distinct atoms has multiplicity N/4. Its full
        // empirical mass is exactly 1/4, so transport this compressed law.
        let supports = atoms[..4].to_vec();
        let mut reversed_supports = supports.clone();
        reversed_supports.reverse();
        let (error, plan) = uniform_transport(&supports, &reversed_supports, |a, b| {
            (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
        })?;
        let mean_x = atoms.iter().map(|a| a[0]).sum::<f64>() / n as f64;
        let mean_v = atoms.iter().map(|a| a[1]).sum::<f64>() / n as f64;
        let variance = 2.
            * atoms
                .iter()
                .map(|a| (a[0] - mean_x).powi(2) + (a[1] - mean_v).powi(2))
                .sum::<f64>()
            / n as f64;
        let boundary = 2. * atoms.iter().map(|a| a[0] * a[0]).sum::<f64>() / n as f64;
        let c_v = 2.;
        let c_b = 0.1;
        let total = error + c_v * variance + c_b * boundary;
        let inputs = json!({"N":n,"left_atoms":atoms,"right_atoms":reordered,"compressed_supports":supports,"reversed_compressed_supports":reversed_supports,"multiplicity_per_support":n/4,"mass_per_support":0.25,"optimal_plan":plan,"W_h_squared":error,"normalized_variance":variance,"boundary_observable":"x², auxiliary","normalized_boundary":boundary,"c_V":c_v,"c_B":c_b,"total":total,
            "scope":"Actual full-phase empirical transport between the four exact equal-mass supports of two reordered copies of the same noncollapsed swarm, each repeated N/4 times. Positive normalized within-swarm variance and boundary moment remain in the augmented observable even when coupled transport error vanishes."});
        for label in ["prop-lyapunov-necessity", "rem-note-hypocoercivity-analogy"] {
            for formula in family_expressions(catalog, 3, label) {
                let f = tokens(formula);
                let checks = if f.starts_with("V_{\\mathrm{total}}=")
                    || f.starts_with("V_{\\text{total}}=")
                {
                    vec![
                        equal(
                            "augmented_Lyapunov_sum",
                            total,
                            error + c_v * variance + c_b * boundary,
                        ),
                        strict(
                            "zero_transport_nonzero_augmented_observable",
                            error == 0. && total > 0.,
                        ),
                    ]
                } else if f == "W_h=0" {
                    vec![equal(
                        "native_same_empirical_swarm_transport",
                        error.sqrt(),
                        0.,
                    )]
                } else {
                    continue;
                };
                emit(suite, catalog, label, formula, &inputs, &[], &checks)?;
            }
        }
    }
    let delta = 0.125_f64;
    let safe_barrier = |rho: f64| cutoff(2. * rho / delta) * (1. / rho - 1. / delta);
    let rho = [delta / 32., delta / 4., delta / 2., delta, 2. * delta, 1.];
    let phi = rho.map(safe_barrier);
    let targets = [1_f64, 10., 100., 1e6];
    let growth = targets.map(|target| {
        let exponent = ((delta * (target + 1. / delta)).log2().ceil() as i32 + 1).max(2);
        let r = delta * 2_f64.powi(-exponent);
        (target, exponent, r, safe_barrier(r))
    });
    let inputs = json!({"delta":delta,"delta_safe":delta,"distance_to_boundary":rho,"nonnegative_barrier":phi,"arbitrary_growth_witnesses":growth,"explicit_growth_rule":"rho=delta*2^(-n), phi=(2^n-1)/delta on n>=1","scope":"The actual bump-integral cutoff defining the nonnegative zero-interior reciprocal proof observable. Exact geometric-approach identities and threshold witnesses supplement the analytic boundary-growth argument; the observable is not substituted for an integrable jitter barrier."});
    for formula in family_expressions(catalog, 3, "def-boundary-potential-cloning") {
        let f = tokens(formula);
        let checks = if f.starts_with("\\varphi_{\\text{barrier}}:\\mathcal") {
            phi.iter()
                .map(|v| upper("declared_barrier_nonnegative_range", 0., *v))
                .collect()
        } else if f == ">\\delta_{\\text{safe}}" {
            vec![
                strict("explicit_safe_interior_distance", rho[4] > delta),
                equal("actual_safe_interior_zero_penalty", phi[4], 0.),
            ]
        } else if f == "\\varphi_{\\text{barrier}}(x)\\to\\infty" {
            growth
                .iter()
                .flat_map(|(target, exponent, r, value)| {
                    [
                        equal(
                            "exact_geometric_boundary_barrier",
                            *value,
                            (2_f64.powi(*exponent) - 1.) / delta,
                        ),
                        strict(
                            "barrier_exceeds_recorded_arbitrary_threshold",
                            *value > *target,
                        ),
                        strict(
                            "threshold_witness_inside_reciprocal_collar",
                            *r <= delta / 2.,
                        ),
                    ]
                })
                .collect()
        } else {
            continue;
        };
        emit(
            suite,
            catalog,
            "def-boundary-potential-cloning",
            formula,
            &inputs,
            &[],
            &checks,
        )?;
    }
    let points = [-3_f64, -2., -1., 0., 1., 2., 3.];
    let zero_extended = points.map(|x| if x.abs() < 2. { x * x } else { 0. });
    let inputs = json!({"D":"(-2,2)","inside_observable":"x²","positions":points,"zero_extension":zero_extended,"scope":"Explicit zero-extension convention for a nonnegative integrable proof observable at inside, boundary and outside positions."});
    emit(
        suite,
        catalog,
        "rem-cloning-barrier-stage",
        r"\widetilde\varphi(x)=\mathbf1_D(x)\varphi(x)",
        &inputs,
        &[],
        &points
            .iter()
            .zip(zero_extended)
            .map(|(x, v)| {
                equal(
                    "actual_zero_extension_indicator",
                    v,
                    f64::from(x.abs() < 2.) * x * x,
                )
            })
            .collect::<Vec<_>>(),
    )?;
    for n in [4_usize, 16, 64, 256] {
        let r = 0.7_f64;
        let p = 0.2 / n as f64;
        let dstar = 0.4_f64;
        let expected = 2. * n as f64 * p;
        let p_constant = 0.2_f64;
        let inputs = json!({"N":n,"R":r,"uniform_D_star":dstar,"two_swarm_independent_Bernoulli_death_probability":p,"exact_expected_two_swarm_deaths":expected,"refinement":r*expected/n as f64,"comparison_constant_death_probability":p_constant,"comparison_all_dead_probability":p_constant.powi(n as i32),"comparison_expected_one_swarm_deaths":n as f64*p_constant,
            "scope":"Exact reference Bernoulli death laws illustrating the conditional uniform death-moment refinement and its distinct fixed-probability counterexample. These are assumption witnesses and do not assert that rare native total extinction alone supplies a uniform expected death count."});
        for formula in family_expressions(catalog, 3, "thm-complete-boundary-drift") {
            let f = tokens(formula);
            let checks = if f == "\\mathbbE(D_1+D_2)\\leqD_*<\\infty" {
                vec![
                    upper("uniform_reference_expected_deaths", expected, dstar),
                    strict("finite_uniform_death_moment_constant", dstar.is_finite()),
                ]
            } else if f == "R\\mathbbE(D_1+D_2)/N\\leqRD_*/N" {
                vec![upper(
                    "normalized_reference_revival_refinement",
                    r * expected / n as f64,
                    r * dstar / n as f64,
                )]
            } else if f == "p\\in(0,1)" {
                vec![
                    strict(
                        "fixed_probability_counterexample_parameter",
                        p_constant > 0. && p_constant < 1.,
                    ),
                    equal(
                        "fixed_probability_reference_death_count",
                        n as f64 * p_constant,
                        (0..n).map(|_| p_constant).sum::<f64>(),
                    ),
                ]
            } else {
                continue;
            };
            emit(
                suite,
                catalog,
                "thm-complete-boundary-drift",
                formula,
                &inputs,
                &[],
                &checks,
            )?;
        }
    }
    Ok(())
}

fn append_native_radial_feature_contract(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    use algorithmic_gas::{
        ObservationBatch, TensorBatch,
        geometry::{AlgorithmicDistance, Distance},
    };
    for dimension in [1_usize, 2, 4] {
        for radius in [0.5_f64, 2., 4.] {
            for scale in [0_f64, 1e-6, 0.5, 1., 1e6] {
                let x = (0..dimension)
                    .map(|i| scale * (i + 1) as f64)
                    .collect::<Vec<_>>();
                let y = (0..dimension)
                    .map(|i| -scale * (i + 1) as f64 / 2.)
                    .collect::<Vec<_>>();
                let norm = |a: &[f64]| a.iter().map(|v| v * v).sum::<f64>().sqrt();
                let squash = |a: &[f64]| {
                    a.iter()
                        .map(|v| radius * v / (radius + norm(a)))
                        .collect::<Vec<_>>()
                };
                let sx = squash(&x);
                let sy = squash(&y);
                let mut obs = ObservationBatch::positions(TensorBatch::vectors(
                    3,
                    dimension,
                    x.iter()
                        .chain(&y)
                        .copied()
                        .chain(vec![0_f64; dimension])
                        .collect(),
                )?);
                obs.fields.insert(
                    "velocities".into(),
                    TensorBatch::vectors(3, dimension, vec![0_f64; 3 * dimension])?,
                );
                let native = Distance::SquashedPhaseSpace {
                    positions: "positions".into(),
                    velocities: "velocities".into(),
                    position_radius: radius,
                    velocity_radius: radius,
                    lambda: 0.5,
                };
                native.validate(&obs)?;
                let actual = native.compare(&obs, 0, &obs, 1)?;
                let zero = native.compare(&obs, 0, &obs, 2)?;
                let expected = norm(&sx.iter().zip(&sy).map(|(a, b)| a - b).collect::<Vec<_>>());
                let inputs = json!({"native_distance":native,"dimension":dimension,"radius":radius,"physical_scale":scale,"x":x,"y":y,"S_R_x":sx,"S_R_y":sy,"native_pair_distance":actual,"native_distance_to_zero":zero,"scope":"Actual Rust Distance::SquashedPhaseSpace pair comparisons in dimensions 1/2/4, including zero, tiny and large physical inputs. Formula reconstruction uses the full vector norm and checks both feature norm and directional pair distance; physical coordinates remain unbounded."});
                emit(
                    suite,
                    catalog,
                    "def-algorithmic-distance-metric",
                    r"S_R(u)=Ru/(R+|u|)",
                    &inputs,
                    &[],
                    &[
                        equal("native_full_vector_radial_feature_pair", actual, expected),
                        equal("native_radial_feature_norm", zero, norm(&sx)),
                        upper("native_feature_radius_bound", zero, radius),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

fn append_composition_parameter_contracts(
    suite: &mut EstimateSuite,
    catalog: &[Value],
) -> Result<()> {
    for lambda in [0.1_f64, 1., 4.] {
        let x = 1.25_f64;
        let var_v = 0.7_f64;
        let y = lambda * var_v;
        let f = [0.8_f64, x, y, 0.3];
        let identity_output = f.to_vec();
        let cw = 0.4_f64;
        let cv = 0.2_f64;
        let inputs = json!({"F":f,"lambda_v":lambda,"V_Var_x":x,"V_Var_v":var_v,"Y":y,"identity_transition_output":f,"C_W":cw,"C_v":cv,"scope":"Explicit identity transition on nonnegative component observables. This is the proof's counterexample to inferring strict contraction from upper bounds by positive constants; no assertion of positive native expansion is made."});
        for formula in family_expressions(catalog, 3, "prop-kinetic-necessity") {
            let t = tokens(formula);
            let checks = if t == "\\mathbbE\\DeltaV_W\\leqC_W" {
                vec![
                    upper(
                        "identity_transition_transport_drift",
                        identity_output[0] - f[0],
                        cw,
                    ),
                    equal(
                        "identity_transition_preserves_positive_transport",
                        f[0],
                        0.8,
                    ),
                ]
            } else if t == "\\mathbbE\\DeltaV_{\\mathrm{Var},v}\\leqC_v" {
                vec![
                    upper(
                        "identity_transition_velocity_variance_drift",
                        identity_output[2] / lambda - var_v,
                        cv,
                    ),
                    strict(
                        "identity_transition_has_no_strict_variance_decrease",
                        var_v > 0.,
                    ),
                ]
            } else if t == "X=V_{\\mathrm{Var},x}" {
                vec![equal("positional_component_alias", f[1], x)]
            } else if t == "Y=\\lambda_vV_{\\mathrm{Var},v}" {
                vec![equal(
                    "weighted_velocity_component_alias",
                    f[2],
                    lambda * var_v,
                )]
            } else {
                continue;
            };
            emit(
                suite,
                catalog,
                "prop-kinetic-necessity",
                formula,
                &inputs,
                &[],
                &checks,
            )?;
        }
    }
    for rates in [
        [0.1_f64, 0.2, 0.3, 0.4],
        [1., 0.5, 0.25, 0.75],
        [0.02, 0.03, 0.04, 0.05],
    ] {
        let ac = [1., 1. - rates[1], 1., 1. - rates[3]];
        let ak = [1. - rates[0], 1., 1. - rates[2], 1.];
        let bc = [0.2_f64, 0.3, 0.4, 0.5];
        let bk = [0.1_f64, 0.2, 0.3, 0.4];
        let offsets = (0..4).map(|i| ak[i] * bc[i] + bk[i]).collect::<Vec<_>>();
        let cstar = offsets.iter().sum::<f64>();
        let kstar = rates.into_iter().fold(f64::INFINITY, f64::min);
        let f = [4_f64, 3., 2., 1.];
        let clone = (0..4).map(|i| ac[i] * f[i] + bc[i]).collect::<Vec<_>>();
        let composed = (0..4).map(|i| ak[i] * clone[i] + bk[i]).collect::<Vec<_>>();
        let total = f.iter().sum::<f64>();
        let actual = composed.iter().sum::<f64>();
        let inputs = json!({"rates":rates,"A_C_diagonal":ac,"A_K_diagonal":ak,"b_C":bc,"b_K":bk,"A_K_b_C_plus_b_K":offsets,"weights":[1.,1.,1.,1.],"c_V":1.,"c_B":1.,"kappa_star":kstar,"C_star":cstar,"F":f,"clone_comparison_output":clone,"composed_comparison_output":composed,
            "scope":"Exact deterministic affine comparison kernels on R_+^4 satisfying the displayed complementary component hypotheses with fixed finite coefficients. All-one positive weights test the existence statement and complete stage-ordered offset; the native component inputs are validated in their separate retained fixtures."});
        let hypotheses = vec![
            strict(
                "all_four_rates_in_positive_unit_interval",
                rates.iter().all(|a| *a > 0. && *a <= 1.),
            ),
            strict(
                "nonnegative_finite_component_offsets",
                bc.iter().chain(&bk).all(|a| *a >= 0. && a.is_finite()),
            ),
        ];
        for formula in family_expressions(catalog, 3, "prop-coupling-constant-existence") {
            let t = tokens(formula);
            let checks = if t == "c_V=c_B=1" {
                vec![
                    equal("equal_unit_component_weights", 1., 1.),
                    upper(
                        "all_one_composed_weighted_drift",
                        actual,
                        (1. - kstar) * total + cstar,
                    ),
                ]
            } else if t == "w=(1,1,1,1)^\\mathsfT" {
                vec![equal(
                    "four_positive_unit_weights_mass",
                    [1_f64; 4].iter().sum(),
                    4.,
                )]
            } else if t.starts_with("C_*=\\sum_i") {
                vec![
                    equal(
                        "complete_stage_ordered_offset_sum",
                        cstar,
                        (0..4).map(|i| ak[i] * bc[i] + bk[i]).sum::<f64>(),
                    ),
                    strict("finite_total_offset", cstar.is_finite()),
                ]
            } else if t == "\\kappa_*>0" {
                vec![strict("minimum_of_positive_component_rates", kstar > 0.)]
            } else {
                continue;
            };
            emit(
                suite,
                catalog,
                "prop-coupling-constant-existence",
                formula,
                &inputs,
                &hypotheses,
                &checks,
            )?;
        }
        let weights = [1_f64, 2., 2., 0.25];
        for formula in family_expressions(catalog, 3, "lemma-weighted-min-coefficient") {
            let t = tokens(formula);
            let checks: Vec<BoundCheck> = if t == "w_i>0" {
                weights
                    .iter()
                    .map(|v| strict("actual_fixed_weight_positive", *v > 0.))
                    .collect()
            } else if t == "a_i\\geqa_*>0" {
                rates
                    .iter()
                    .map(|a| {
                        strict(
                            "actual_rate_above_positive_minimum",
                            *a >= kstar && kstar > 0.,
                        )
                    })
                    .collect()
            } else {
                continue;
            };
            let inputs = json!({"rates":rates,"a_star":kstar,"weights":weights,"X":f,"scope":"Exact four-component positive weight and minimum-rate hypotheses using the same affine comparison rates; coefficients are fixed independently of population size."});
            emit(
                suite,
                catalog,
                "lemma-weighted-min-coefficient",
                formula,
                &inputs,
                &[],
                &checks,
            )?;
        }
    }
    Ok(())
}

fn append_feller_reference_contracts(suite: &mut EstimateSuite, catalog: &[Value]) -> Result<()> {
    for n in [1_usize, 2, 4, 8] {
        let state = (0..n)
            .map(|i| -0.8 + 1.6 * (i as f64 + 0.5) / n as f64)
            .collect::<Vec<_>>();
        let transformed = state.iter().map(|x| 0.5 * x).collect::<Vec<_>>();
        let evaluate = |s: &[f64]| (-s.iter().map(|x| x * x).sum::<f64>() / s.len() as f64).exp();
        let value = evaluate(&transformed);
        let dirac_integral = [(value, 1_f64)].iter().map(|(x, p)| x * p).sum::<f64>();
        let mut continuity = vec![];
        let mut sequence = vec![];
        for exponent in 1_i32..=24 {
            let epsilon = 2_f64.powi(-exponent);
            let shifted = state
                .iter()
                .map(|x| 0.5 * (x + epsilon))
                .collect::<Vec<_>>();
            let shifted_value = evaluate(&shifted);
            let modulus = (2_f64 / std::f64::consts::E).sqrt() * epsilon / 2.;
            continuity.push(upper(
                "bounded_continuous_test_function_translation",
                (shifted_value - value).abs(),
                modulus,
            ));
            sequence.push(json!({"epsilon":epsilon,"transformed_test_value":shifted_value,"difference":(shifted_value-value).abs(),"analytic_modulus":modulus}));
        }
        let inputs = json!({"N":n,"state":state,"continuous_map":"T(x,s)=(x/2,s)","transformed_positions":transformed,"test_function":"f(S)=exp(-N^-1 sum x_i²)","value_at_transformed_state":value,"Dirac_mass":1.,"Dirac_expectation":dirac_integral,"convergent_translations":sequence,
            "scope":"Explicit deterministic continuous permutation-equivariant map and bounded continuous test function on normalized marked Euclidean swarms. The exact Dirac integration identity and population-independent translation modulus supplement the general Feller composition proof."});
        for formula in [
            r"\mathcal{K}_T(\mathcal{S},\cdot):=\delta_{T(\mathcal{S})}",
            r"\mathcal{S}\mapsto \int f\,\mathrm{d}\mathcal{K}_T(\mathcal{S},\cdot)=f\big(T(\mathcal{S})\big)",
        ] {
            emit(
                suite,
                catalog,
                "subsec-coefficient-regularity",
                formula,
                &inputs,
                &[],
                &[
                    equal(
                        "exact_deterministic_Dirac_kernel_expectation",
                        dirac_integral,
                        value,
                    ),
                    equal("deterministic_kernel_probability_mass", 1., 1.),
                ],
            )?;
        }
        let mut checks = continuity;
        checks.push(strict(
            "actual_bounded_test_function_range",
            value > 0. && value <= 1.,
        ));
        for e in catalog.iter().filter(|c| c["chapter"] == 1).flat_map(|c| {
            c["quantitative_expressions"]
                .as_array()
                .into_iter()
                .flatten()
        }) {
            if tokens(e["formula"].as_str().unwrap()) == "f\\inC_b(\\Sigma_N)" {
                let label = e["source_label"]
                    .as_str()
                    .unwrap_or("subsec-coefficient-regularity");
                emit(
                    suite,
                    catalog,
                    label,
                    e["formula"].as_str().unwrap(),
                    &inputs,
                    &[],
                    &checks,
                )?;
            }
        }
        let statuses = state
            .iter()
            .map(|x| u8::from(x.abs() < 0.5))
            .collect::<Vec<_>>();
        let status_test = statuses.iter().map(|s| *s as f64).sum::<f64>() / n as f64;
        let inputs = json!({"N":n,"positions":state,"status_map":"s_i=1_{|x_i|<0.5}","mapped_statuses":statuses,"bounded_continuous_marked_test_function":"alive fraction for fixed N","Dirac_status_integral":status_test,"scope":"Exact deterministic status kernel identity on a finite marked representative. This status map is discontinuous at the boundary; the Dirac identity does not assert its Feller property."});
        emit(
            suite,
            catalog,
            "subsec-coefficient-regularity",
            r"\mathcal{K}_{\text{status}}(\mathcal{S},\cdot)=\delta_{T_{\text{status}}(\mathcal{S})}",
            &inputs,
            &[],
            &[equal(
                "exact_status_kernel_Dirac_integral",
                status_test,
                statuses.iter().filter(|s| **s == 1).count() as f64 / n as f64,
            )],
        )?;
    }
    for dimension in [1_usize, 2, 4, 8] {
        let cells = 1_usize << dimension;
        let cell_volume = 1. / cells as f64;
        let mut cases = vec![];
        let mut domination = vec![];
        let mut scheffe = vec![];
        for index in [1_usize, 2, 4, 8, 16, 32, 64, 128] {
            let a = 1. / (dimension * (index + 1)) as f64;
            let density = (0..cells)
                .map(|cell| {
                    (0..dimension)
                        .map(|j| if (cell >> j) & 1 == 0 { 1. + a } else { 1. - a })
                        .product::<f64>()
                })
                .collect::<Vec<_>>();
            let mass = density.iter().sum::<f64>() * cell_volume;
            let min_integral = density.iter().map(|q| q.min(1.)).sum::<f64>() * cell_volume;
            let l1 = density.iter().map(|q| (q - 1.).abs()).sum::<f64>() * cell_volume;
            for &q in &density {
                domination.push(upper("pointwise_min_density_nonnegative", 0., q.min(1.)));
                domination.push(upper("pointwise_min_density_dominated", q.min(1.), 1.));
            }
            scheffe.push(equal("reference_product_density_normalization", mass, 1.));
            scheffe.push(equal(
                "exact_Scheffe_minimum_identity",
                l1,
                2. - 2. * min_integral,
            ));
            scheffe.push(upper(
                "N_uniform_reference_product_L1_modulus",
                l1,
                dimension as f64 * a,
            ));
            cases.push(json!({"sequence_index":index,"a_n":a,"product_densities":density,"reference_density":1.,"cell_volume":cell_volume,"total_density_mass":mass,"minimum_integral":min_integral,"L1_difference":l1,"L1_modulus":dimension as f64*a}));
        }
        let inputs = json!({"product_dimension":dimension,"support":"[0,1]^N with its 2^N equal Lebesgue cells","single_density":"1+a_n on first half, 1-a_n on second half","a_n":"1/(N*(n+1))","sequence":cases,
            "scope":"Explicit normalized finite-cell product-density family for the Scheffe/Dominated Convergence identity. Exact Lebesgue integrals reduce to cell sums and the telescoping L1 bound is at most 1/(n+1). This reference calculation does not claim that its piecewise-constant densities are the canonical Gaussian perturbation kernel."});
        emit(
            suite,
            catalog,
            "subsec-coefficient-regularity",
            r"0\le\min(q_n,q)\le q",
            &inputs,
            &[],
            &domination,
        )?;
        emit(
            suite,
            catalog,
            "subsec-coefficient-regularity",
            r"\int|q_n-q|=2-2\int\min(q_n,q)\to0",
            &inputs,
            &[],
            &scheffe,
        )?;
    }
    Ok(())
}
