//! Exact chapter 3 geometric, scalar and conditional-selection proof estimates.
use crate::{
    Benchmark,
    convergence_estimates::{EstimateEvidence, EstimateSuite},
    convergence_framework::BoundCheck,
    convergence_selection::{SelectionFixture, validate_selection},
};
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasConfig, GasError, ObservationBatch, Population, Precision,
    Provenance, Result, RewardBatch, TensorBatch,
    cloning::CloneDecision,
    donor::{CompanionBatch, DonorPool},
    fitness::{PositiveMap, PositiveMapping, Standardizer},
    geometry::{AlgorithmicDistance, ComparisonKind, Distance, InteractionKernel, Kernel},
    noise::{FactorValues, InnovationLaw, NoiseGeometry, NoiseRequest, NoiseSource},
    random::{RandomStream, Stream},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const SOURCE: &str =
    include_str!("../../../../docs/source/2_fractal_gas/convergence_program/03_cloning.md");
const SCOPE: &str = "Chapter 3 exact finite-input diagnostic. Coordinates, cluster memberships, realized measurement vectors and conditional companion laws are retained. Quantitative hypotheses are computed on these inputs; finite checks do not certify an unsampled family or a complete-swarm rate. Phase-space comparisons use a positive quadratic form and retain velocity contributions.";

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
fn block(label: &str) -> Result<&'static str> {
    let start = SOURCE
        .find(&format!(":label: {label}\n"))
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
            chapter: 3,
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

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / x.len() as f64
}
fn range(x: &[f64]) -> f64 {
    x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - x.iter().copied().fold(f64::INFINITY, f64::min)
}
fn jvec(value: &Value, key: &str) -> Result<Vec<f64>> {
    serde_json::from_value(value[key].clone()).map_err(|e| error(&e.to_string()))
}
fn jnum(value: &Value, key: &str) -> Result<f64> {
    value[key]
        .as_f64()
        .ok_or_else(|| error(&format!("missing number {key}")))
}
fn bind_inline(
    builder: &mut Builder,
    label: &str,
    input: &Value,
    items: Vec<(&str, Vec<BoundCheck>)>,
) -> Result<()> {
    for (quote, checks) in items {
        builder.inline(label, quote, input.clone(), checks)?;
    }
    Ok(())
}
fn condition(id: &str, value: bool) -> Vec<BoundCheck> {
    vec![hypothesis(id, value)]
}
fn equal(id: &str, lhs: f64, rhs: f64) -> Vec<BoundCheck> {
    vec![relative(id, lhs, rhs, lhs.abs().max(rhs.abs()))]
}
fn le(id: &str, lhs: f64, rhs: f64) -> Vec<BoundCheck> {
    vec![upper(id, lhs, rhs)]
}
fn ge(id: &str, lhs: f64, rhs: f64) -> Vec<BoundCheck> {
    le(id, rhs, lhs)
}
fn strict(id: &str, lhs: f64, rhs: f64) -> Vec<BoundCheck> {
    condition(id, lhs.is_finite() && rhs.is_finite() && lhs < rhs)
}

/// Append independent, exactly quoted geometric and scalar cloning estimates.
pub fn append_cloning_scalar(suite: &mut EstimateSuite) -> Result<()> {
    if suite.chapter != 3 {
        return Err(error("cloning scalar evidence belongs to chapter 3"));
    }
    let report = validate_selection(20261002)?;
    let geometry = report
        .fixtures
        .iter()
        .find(|f| f.name == "finite_cluster_geometry_and_target")
        .ok_or_else(|| error("missing native geometry fixture"))?;
    let mut builder = Builder::default();
    geometry_identities(&mut builder, geometry)?;
    realized_scalar_vectors(&mut builder)?;
    logarithmic_comparisons(&mut builder)?;
    conditional_selection(&mut builder)?;
    positional_variance_lower_bounds(&mut builder)?;
    averaged_group_bounds(&mut builder)?;
    greedy_history_bounds(&mut builder)?;
    scalar_signal_and_operator_bounds(&mut builder)?;
    futures_lite::future::block_on(native_cloning_outputs(&mut builder))?;
    suite.evidence.extend(builder.records.into_values());
    suite.scope_notes.push("The scalar cloning supplement evaluates exact source expressions individually. Finite cluster identities retain actual coordinates and memberships; target concentration uses a proved optimal full positive phase-space coupling with matching velocities. Native logistic standardization and clipped acceptance are evaluated before conditional expectations. Tiny positive identities use relative or logarithmic residuals.".into());
    Ok(())
}

fn geometry_identities(builder: &mut Builder, fixture: &SelectionFixture) -> Result<()> {
    let h = &fixture.hypotheses;
    let o = &fixture.observations;
    let x = jvec(h, "positions")?;
    let v = jvec(h, "velocities")?;
    let clusters: Vec<Vec<usize>> =
        serde_json::from_value(o["clusters"].clone()).map_err(|e| error(&e.to_string()))?;
    let high: Vec<bool> =
        serde_json::from_value(o["high"].clone()).map_err(|e| error(&e.to_string()))?;
    let common: Vec<bool> =
        serde_json::from_value(o["common"].clone()).map_err(|e| error(&e.to_string()))?;
    let outliers: Vec<usize> = serde_json::from_value(o["auxiliary_global_outliers"].clone())
        .map_err(|e| error(&e.to_string()))?;
    let k = x.len() as f64;
    let n = jnum(h, "slots")?;
    let lv = jnum(h, "lambda_v")?;
    let la = jnum(h, "lambda_alg")?;
    let eps = jnum(h, "epsilon_outlier")?;
    let rvar = jnum(h, "R_var_squared")?;
    let d = jnum(h, "maximum_cluster_diameter")?;
    let cd = 2_f64;
    let clustering_epsilon = d / cd;
    let min_size = 5_usize.max((0.05 * k).ceil() as usize);
    let mx = mean(&x);
    let mv = mean(&v);
    let dx = range(&x);
    let dv = range(&v);
    let dh2 = dx * dx + lv * dv * dv;
    let dvalid2 = dx * dx + la * dv * dv;
    let clambda = 1_f64.max(lv / la);
    let varx = variance(&x);
    let varv = variance(&v);
    let varh = varx + lv * varv;
    let energy = x
        .iter()
        .zip(&v)
        .map(|(x, v)| (x - mx).powi(2) + lv * (v - mv).powi(2))
        .collect::<Vec<_>>();
    let captured = outliers.iter().map(|&i| energy[i]).sum::<f64>();
    let sk = k * varx;
    let high_x = x
        .iter()
        .enumerate()
        .filter(|(i, _)| high[*i])
        .map(|(_, x)| (x - mx).powi(2))
        .sum::<f64>();
    let high_count = high.iter().filter(|&&b| b).count() as f64;
    let centers = clusters
        .iter()
        .map(|g| mean(&g.iter().map(|&i| x[i]).collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    let vcenters = clusters
        .iter()
        .map(|g| mean(&g.iter().map(|&i| v[i]).collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    let radii = clusters
        .iter()
        .zip(&centers)
        .map(|(g, c)| g.iter().map(|&i| (x[i] - c).abs()).fold(0., f64::max))
        .collect::<Vec<_>>();
    let contributions = clusters
        .iter()
        .enumerate()
        .map(|(g, m)| {
            m.len() as f64 * ((centers[g] - mx).powi(2) + lv * (vcenters[g] - mv).powi(2))
        })
        .collect::<Vec<_>>();
    let cluster_high = clusters
        .iter()
        .map(|g| g.iter().all(|&i| high[i]))
        .collect::<Vec<_>>();
    let valid_energy = clusters
        .iter()
        .enumerate()
        .filter(|(_, g)| g.len() >= min_size)
        .map(|(i, _)| contributions[i])
        .sum::<f64>();
    let selected_valid = clusters
        .iter()
        .enumerate()
        .filter(|(i, g)| g.len() >= min_size && cluster_high[*i])
        .map(|(i, _)| contributions[i])
        .sum::<f64>();
    let invalid_energy = clusters
        .iter()
        .enumerate()
        .filter(|(_, g)| g.len() < min_size)
        .map(|(i, _)| contributions[i])
        .sum::<f64>();
    let between = contributions.iter().sum::<f64>() / k;
    let bh = contributions
        .iter()
        .enumerate()
        .filter(|(i, _)| cluster_high[*i])
        .map(|(_, e)| e)
        .sum::<f64>()
        / k;
    let within = clusters
        .iter()
        .map(|g| {
            let gx = g.iter().map(|&i| x[i]).collect::<Vec<_>>();
            let gv = g.iter().map(|&i| v[i]).collect::<Vec<_>>();
            g.len() as f64 / k * (variance(&gx) + lv * variance(&gv))
        })
        .sum::<f64>();
    let within_x = clusters
        .iter()
        .map(|g| g.len() as f64 * variance(&g.iter().map(|&i| x[i]).collect::<Vec<_>>()))
        .sum::<f64>();
    let between_x = clusters
        .iter()
        .enumerate()
        .map(|(g, m)| m.len() as f64 * (centers[g] - mx).powi(2))
        .sum::<f64>();
    let fh = (1. - eps) * (rvar - clambda * d * d / 2.) / dh2;
    let ch = (1. - eps) * (1. - d * d / (2. * rvar));
    let input = json!({"fixture":fixture.name,"positions":x,"velocities":v,"clusters":clusters,"high":high,"common":common,"auxiliary_outliers":outliers,"alive_count":k,"slots":n,"lambda_v":lv,"lambda_alg":la,"epsilon_outlier":eps,"R_var_squared":rvar,"maximum_cluster_diameter":d,"cluster_epsilon":clustering_epsilon,"c_d":cd,"minimum_valid_size":min_size,"centers_x":centers,"centers_v":vcenters,"radii":radii,"contributions":contributions,"computed":{"mean_x":mx,"mean_v":mv,"D_x":dx,"D_v":dv,"D_h_squared":dh2,"D_valid_squared":dvalid2,"C_lambda":clambda,"variance_x":varx,"variance_v":varv,"variance_h":varh,"within":within,"between":between,"B_H":bh,"f_H_lower":fh,"c_H":ch}});
    let base = vec![
        hypothesis(
            "same_finite_phase_space_inputs",
            x.len() == v.len() && x.iter().chain(&v).all(|z| z.is_finite()) && k >= 2. && k <= n,
        ),
        hypothesis(
            "positive_metric_and_thresholds",
            lv > 0. && la > 0. && d > 0. && rvar > clambda * d * d / 2. && eps > 0. && eps < 1.,
        ),
    ];
    let label = "def-unified-high-low-error-sets";
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (r"k \ge 2", condition("at_least_two_alive", k >= 2.)),
            (
                r"D_{\text{diam}}(\epsilon) := c_d \cdot \epsilon",
                equal("configured_diameter", d, cd * clustering_epsilon),
            ),
            (
                r"c_d > 0",
                condition("positive_cluster_diameter_multiplier", cd > 0.),
            ),
            (r"c_d = 2", equal("typical_diameter_multiplier", cd, 2.)),
            (
                r"k_{\min} := \max(5, \lceil 0.05k \rceil)",
                equal(
                    "computed_minimum_size",
                    min_size as f64,
                    5_f64.max((0.05 * k).ceil()),
                ),
            ),
            (
                r"\varepsilon_O \in (0, 1)",
                condition("actual_capture_parameter", eps > 0. && eps < 1.),
            ),
        ],
    )?;
    // The illustrative 0.1 capture parameter is checked on its own configured
    // partition, rather than being attributed to the 0.25 fixture.
    builder.inline(label,r"\varepsilon_O = 0.1",json!({"configured_capture_parameter":0.1,"case_scope":"definition's illustrative parameter"}),equal("illustrative_capture_parameter",0.1,1./10.))?;
    let all_members = clusters.iter().flatten().copied().collect::<Vec<_>>();
    let mut sorted = all_members.clone();
    sorted.sort_unstable();
    let mut definition_checks = vec![
        hypothesis(
            "clusters_form_actual_disjoint_alive_partition",
            sorted == (0..x.len()).collect::<Vec<_>>(),
        ),
        upper(
            "retained_valid_center_energy",
            (1. - eps) * valid_energy,
            selected_valid,
        ),
    ];
    let mut valid_order = clusters
        .iter()
        .enumerate()
        .filter(|(_, g)| g.len() >= min_size)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    valid_order.sort_by(|&a, &b| contributions[b].total_cmp(&contributions[a]));
    let mut prefix = Vec::new();
    let mut cumulative = 0.;
    for &i in &valid_order {
        if cumulative < (1. - eps) * valid_energy {
            prefix.push(i);
            cumulative += contributions[i];
        }
    }
    definition_checks.push(hypothesis(
        "actual_smallest_descending_valid_prefix",
        clusters
            .iter()
            .enumerate()
            .all(|(i, g)| cluster_high[i] == (g.len() < min_size || prefix.contains(&i))),
    ));
    builder.shown(
        label,
        r"\sum_{m \in O_M}",
        input.clone(),
        base.clone(),
        definition_checks,
    )?;
    builder.shown(
        label,
        r"H_k(\epsilon) :=",
        input.clone(),
        base.clone(),
        condition(
            "high_is_selected_valid_plus_all_invalid",
            clusters.iter().enumerate().all(|(i, g)| {
                g.iter()
                    .all(|&j| high[j] == (g.len() < min_size || prefix.contains(&i)))
            }),
        ),
    )?;
    builder.shown(
        label,
        r"L_k(\epsilon) :=",
        input.clone(),
        base.clone(),
        condition(
            "low_is_alive_complement",
            (0..x.len()).filter(|&i| !high[i]).count() + high_count as usize == x.len(),
        ),
    )?;
    for (gi, g) in clusters.iter().enumerate() {
        let gx = g.iter().map(|&i| x[i]).collect::<Vec<_>>();
        let gv = g.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let pairx = g
            .iter()
            .flat_map(|&i| g.iter().map(move |&j| (i, j)))
            .map(|(i, j)| (x[i] - x[j]).powi(2))
            .sum::<f64>();
        let pairq = g
            .iter()
            .flat_map(|&i| g.iter().map(move |&j| (i, j)))
            .map(|(i, j)| (x[i] - x[j]).powi(2) + lv * (v[i] - v[j]).powi(2))
            .sum::<f64>();
        let pairs = g
            .iter()
            .flat_map(|&i| g.iter().map(move |&j| (i, j)))
            .collect::<Vec<_>>();
        let diameter = pairs
            .iter()
            .map(|&(i, j)| ((x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2)).sqrt())
            .fold(0., f64::max);
        let giinput = json!({"cloud":input,"cluster_index":gi,"cluster_members":g,"cluster_x":gx,"cluster_v":gv,"cluster_pair_x_squared_sum":pairx,"cluster_pair_q_squared_sum":pairq,"actual_algorithmic_diameter":diameter});
        builder.shown(
            label,
            r"\text{diam}(G_m)",
            giinput.clone(),
            base.clone(),
            vec![upper("all_pairs_actual_diameter", diameter, d)],
        )?;
        builder.shown(
            label,
            r"\text{Contrib}(G_m)",
            giinput.clone(),
            base.clone(),
            equal(
                "actual_cluster_contribution",
                contributions[gi],
                g.len() as f64 * ((mean(&gx) - mx).powi(2) + lv * (mean(&gv) - mv).powi(2)),
            ),
        )?;
        let quote = if g.len() < min_size {
            r"|G_m| < k_{\min}"
        } else {
            r"|G_m| \ge k_{\min}"
        };
        builder.inline(
            label,
            quote,
            giinput.clone(),
            condition(
                "actual_cluster_validity_branch",
                if g.len() < min_size {
                    g.len() < min_size
                } else {
                    g.len() >= min_size
                },
            ),
        )?;
        let l = "lem-outlier-cluster-fraction-lower-bound";
        builder.shown(
            l,
            r"\operatorname{Var}_{G}(q)",
            giinput.clone(),
            base.clone(),
            vec![
                relative(
                    "cluster_full_phase_pair_identity",
                    variance(&gx) + lv * variance(&gv),
                    pairq / (2. * (g.len() * g.len()) as f64),
                    pairq,
                ),
                upper(
                    "cluster_full_phase_variance_bound",
                    variance(&gx) + lv * variance(&gv),
                    clambda * d * d / 2.,
                ),
            ],
        )?;
        builder.inline(
            l,
            r"|\bar q_G-\bar q|^2\le D_h^2",
            giinput.clone(),
            le(
                "actual_cluster_center_diameter",
                (centers[gi] - mx).powi(2) + lv * (vcenters[gi] - mv).powi(2),
                dh2,
            ),
        )?;
        let l = "rem-cluster-energy-and-separation";
        builder.shown(
            l,
            r"\rho_G\le",
            giinput.clone(),
            base.clone(),
            vec![
                upper(
                    "cluster_radius_bounded_by_actual_physical_diameter",
                    radii[gi],
                    range(&gx),
                ),
                upper(
                    "vector_cluster_variance_diameter_bound",
                    variance(&gx),
                    range(&gx).powi(2) / 2.,
                ),
            ],
        )?;
        let l = "lem-variance-concentration-Hk";
        builder.shown(
            l,
            r"\operatorname{Var}(G)=",
            giinput.clone(),
            base.clone(),
            vec![
                relative(
                    "within_x_pair_identity",
                    variance(&gx),
                    pairx / (2. * (g.len() * g.len()) as f64),
                    pairx,
                ),
                upper("within_x_common_diameter_bound", variance(&gx), d * d / 2.),
            ],
        )?;
        for &i in g {
            builder.inline(
                "rem-cluster-energy-and-separation",
                r"x_i-\mu_{x,G}=|G|^{-1}\sum_{j\in G}(x_i-x_j)",
                json!({"cluster":giinput,"row":i}),
                equal(
                    "cluster_centering_pair_average",
                    x[i] - centers[gi],
                    g.iter().map(|&j| x[i] - x[j]).sum::<f64>() / g.len() as f64,
                ),
            )?;
        }
        for &(i, j) in &pairs {
            let alg = (x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2);
            let physical = (x[i] - x[j]).powi(2) + lv * (v[i] - v[j]).powi(2);
            let pairinput = json!({"cloud":input,"i":i,"j":j,"algorithmic_squared_distance":alg,"physical_q_squared_distance":physical});
            builder.inline(
                label,
                r"d_{\text{alg}}(i, j)^2 := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2",
                pairinput.clone(),
                equal(
                    "computed_algorithmic_pair_distance",
                    alg,
                    (x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2),
                ),
            )?;
            builder.inline(
                "lem-outlier-cluster-fraction-lower-bound",
                r"|q_i-q_j|^2\le C_\lambda d_{\rm alg}(i,j)^2",
                pairinput,
                le("actual_phase_metric_comparison", physical, clambda * alg),
            )?;
        }
    }
    packing_identities(builder, &input, &x, &v, lv, la, dx, dv, d)?;
    let label = "lem-outlier-fraction-lower-bound";
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"\lambda_v>0",
                condition("positive_hypocoercive_velocity_weight", lv > 0.),
            ),
            (
                r"D_h^2=D_x^2+\lambda_vD_v^2>0",
                vec![
                    relative(
                        "phase_diameter_definition",
                        dh2,
                        dx * dx + lv * dv * dv,
                        dh2,
                    ),
                    hypothesis("strict_positive_phase_diameter", dh2 > 0.),
                ],
            ),
            (
                r"\varepsilon_O\in(0,1)",
                condition(
                    "capture_fraction_inside_unit_interval",
                    eps > 0. && eps < 1.,
                ),
            ),
            (
                r"\operatorname{Var}_{\mathcal A_k}(q)>R_h^2>0",
                vec![
                    hypothesis("strict_phase_variance_floor", varh > rvar),
                    hypothesis("strict_positive_phase_floor", rvar > 0.),
                ],
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\sum_{i\in O_k}",
        input.clone(),
        base.clone(),
        ge(
            "actual_auxiliary_outlier_capture",
            captured,
            (1. - eps) * k * varh,
        ),
    )?;
    builder.shown(
        label,
        r"\frac{|O_k|}{k}",
        input.clone(),
        base.clone(),
        vec![
            hypothesis(
                "actual_outlier_fraction_strict_lower",
                outliers.len() as f64 / k > (1. - eps) * rvar / dh2,
            ),
            hypothesis(
                "strict_positive_outlier_fraction",
                (1. - eps) * rvar / dh2 > 0.,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"(1-\varepsilon_O)kR_h^2",
        input.clone(),
        base.clone(),
        vec![
            hypothesis(
                "first_outlier_chain_strict",
                (1. - eps) * k * rvar < (1. - eps) * k * varh,
            ),
            upper(
                "second_outlier_chain_capture",
                (1. - eps) * k * varh,
                captured,
            ),
            upper(
                "third_outlier_chain_diameter",
                captured,
                outliers.len() as f64 * dh2,
            ),
        ],
    )?;
    for i in 0..x.len() {
        let qi = [x[i], lv.sqrt() * v[i]];
        let qbar = [mx, lv.sqrt() * mv];
        let centered = [qi[0] - qbar[0], qi[1] - qbar[1]];
        let averages = [
            x.iter().map(|&z| x[i] - z).sum::<f64>() / k,
            lv.sqrt() * v.iter().map(|&z| v[i] - z).sum::<f64>() / k,
        ];
        let rowinput = json!({"cloud":input,"row":i,"q_i":qi,"q_mean":qbar,"centered_q":centered,"average_pair_difference":averages});
        for l in [label, "lem-outlier-cluster-fraction-lower-bound"] {
            builder.inline(
                l,
                r"q_i=(x_i,\sqrt{\lambda_v}v_i)",
                rowinput.clone(),
                vec![
                    relative("phase_coordinate_position", qi[0], x[i], dx),
                    relative("phase_coordinate_velocity", qi[1], lv.sqrt() * v[i], dv),
                ],
            )?;
        }
        builder.inline(
            label,
            r"\bar q=k^{-1}\sum_iq_i",
            rowinput.clone(),
            vec![
                relative(
                    "phase_mean_position",
                    qbar[0],
                    x.iter().sum::<f64>() / k,
                    dx,
                ),
                relative(
                    "phase_mean_velocity",
                    qbar[1],
                    v.iter().map(|v| lv.sqrt() * v).sum::<f64>() / k,
                    dv,
                ),
            ],
        )?;
        builder.inline(
            label,
            r"q_i-\bar q=k^{-1}\sum_j(q_i-q_j)",
            rowinput.clone(),
            vec![
                relative("pair_average_position", centered[0], averages[0], dx),
                relative("pair_average_velocity", centered[1], averages[1], dv),
            ],
        )?;
        builder.inline(
            label,
            r"|q_i-\bar q|\le D_h",
            rowinput,
            le(
                "centered_phase_norm_diameter",
                (centered[0] * centered[0] + centered[1] * centered[1]).sqrt(),
                dh2.sqrt(),
            ),
        )?;
    }
    let label = "lem-outlier-cluster-fraction-lower-bound";
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"\lambda_{\rm alg}>0",
                condition("positive_algorithmic_velocity_weight", la > 0.),
            ),
            (
                r"\lambda_v>0",
                condition("positive_variance_velocity_weight", lv > 0.),
            ),
            (
                r"C_\lambda=\max\{1,\lambda_v/\lambda_{\rm alg}\}",
                equal("metric_comparison_constant", clambda, 1_f64.max(lv / la)),
            ),
            (
                r"D_h^2=D_x^2+\lambda_vD_v^2>0",
                vec![
                    relative("full_phase_diameter", dh2, dx * dx + lv * dv * dv, dh2),
                    hypothesis("diameter_is_positive", dh2 > 0.),
                ],
            ),
            (
                r"\operatorname{Var}_{\mathcal A}(x)>R_{\rm var}^2",
                strict("position_variance_floor", rvar, varx),
            ),
            (
                "\\operatorname{Var}_{\\mathcal A}(q)\\ge\n\\operatorname{Var}_{\\mathcal A}(x)>R_{\\rm var}^2",
                vec![
                    upper(
                        "actual_full_phase_variance_dominates_position_variance",
                        varx,
                        varh,
                    ),
                    hypothesis(
                        "actual_position_variance_strictly_exceeds_positive_outlier_floor",
                        varx > rvar && rvar > 0.,
                    ),
                ],
            ),
            (
                r"R_{\rm var}^2>C_\lambda D^2/2",
                strict(
                    "floor_exceeds_within_cluster_ceiling",
                    clambda * d * d / 2.,
                    rvar,
                ),
            ),
            (
                r"B>R_{\rm var}^2-C_\lambda D^2/2",
                strict(
                    "between_variance_floor",
                    rvar - clambda * d * d / 2.,
                    between,
                ),
            ),
            (
                r"B=B_{\rm invalid}+B_{\rm valid}",
                equal(
                    "invalid_valid_between_decomposition",
                    between,
                    (invalid_energy + valid_energy) / k,
                ),
            ),
            (
                r"B_H\le D_h^2|H|/k",
                le(
                    "selected_center_energy_count_bound",
                    bh,
                    dh2 * high_count / k,
                ),
            ),
        ],
    )?;
    builder.inline(
        label,
        r"d_{\rm alg}(i,j)^2=|x_i-x_j|^2+\lambda_{\rm alg}|v_i-v_j|^2",
        input.clone(),
        equal(
            "actual_unsquashed_diameter_squared",
            dvalid2,
            dx * dx + la * dv * dv,
        ),
    )?;
    builder.shown(
        label,
        r"\operatorname{Var}_{\mathcal A}(q)",
        input.clone(),
        base.clone(),
        vec![
            relative(
                "full_phase_cluster_decomposition",
                varh,
                within + between,
                varh,
            ),
            relative(
                "between_center_variance_definition",
                between,
                contributions.iter().sum::<f64>() / k,
                between,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\frac{|H|}{k}",
        input.clone(),
        base.clone(),
        vec![
            hypothesis(
                "actual_high_cluster_fraction_strict_floor",
                high_count / k > fh,
            ),
            positive_identity(
                "positive_cluster_fraction_definition",
                fh,
                (1. - eps) * (rvar - clambda * d * d / 2.) / dh2,
            ),
        ],
    )?;
    // This proof chain is inline and has to retain both inequalities.
    builder.inline(
        label,
        "B_H\\ge B_{\\rm invalid}+(1-\\varepsilon_O)B_{\\rm valid}\n\\ge(1-\\varepsilon_O)B",
        input.clone(),
        vec![
            upper(
                "invalid_plus_retained_valid_center_energy",
                (invalid_energy + (1. - eps) * valid_energy) / k,
                bh,
            ),
            upper(
                "total_center_capture",
                (1. - eps) * between,
                (invalid_energy + (1. - eps) * valid_energy) / k,
            ),
        ],
    )?;
    let label = "cor-vvarx-to-high-error-fraction";
    let observable = 2. * k / n * varx;
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"k_s=|\mathcal A_s|",
                equal("actual_alive_count", k, x.len() as f64),
            ),
            (
                r"R_{\rm var}^2>C_\lambda D^2/2",
                strict("shared_within_cluster_floor", clambda * d * d / 2., rvar),
            ),
            (
                r"V_{\mathrm{Var},x}>2R_{\rm var}^2",
                strict(
                    "normalized_two_swarm_variance_threshold",
                    2. * rvar,
                    observable,
                ),
            ),
            (
                r"|H_s|/k_s>f_{H,\rm cl}>0",
                vec![
                    hypothesis("one_swarm_actual_high_fraction", high_count / k > fh),
                    hypothesis("positive_shared_count_constant", fh > 0.),
                ],
            ),
            (r"k_s/N\le1", le("alive_fraction_is_at_most_one", k / n, 1.)),
        ],
    )?;
    builder.shown(
        label,
        r"V_{\mathrm{Var},x}=",
        input.clone(),
        base.clone(),
        equal(
            "actual_fixed_N_variance_definition",
            observable,
            k / n * varx + k / n * varx,
        ),
    )?;
    geometric_separation(
        builder, &input, &x, &v, &clusters, &high, &centers, &radii, la, min_size,
    )?;
    let label = "lem-variance-concentration-Hk";
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"S_k/k>R_{\mathrm{var}}^2>D_c^2/2",
                vec![
                    hypothesis("strict_position_variance_threshold", sk / k > rvar),
                    hypothesis(
                        "threshold_above_cluster_variance_ceiling",
                        rvar > d * d / 2.,
                    ),
                ],
            ),
            (
                r"W=\sum_G|G|\operatorname{Var}(G)",
                equal(
                    "computed_within_cluster_energy",
                    within_x,
                    clusters
                        .iter()
                        .map(|g| {
                            g.len() as f64 * variance(&g.iter().map(|&i| x[i]).collect::<Vec<_>>())
                        })
                        .sum(),
                ),
            ),
            (
                r"B=\sum_G|G|\|\mu_G-\mu\|^2",
                equal(
                    "computed_between_cluster_energy",
                    between_x,
                    clusters
                        .iter()
                        .enumerate()
                        .map(|(i, g)| g.len() as f64 * (centers[i] - mx).powi(2))
                        .sum(),
                ),
            ),
            (
                r"S_k=W+B",
                equal("position_energy_cluster_identity", sk, within_x + between_x),
            ),
            (
                r"W\leq kD_c^2/2",
                le("within_cluster_energy_ceiling", within_x, k * d * d / 2.),
            ),
            (
                r"S_k/k>R_{\mathrm{var}}^2",
                strict("actual_position_variance_floor", rvar, sk / k),
            ),
            (
                r"S_k=k\operatorname{Var}_{\mathcal A_k}(x)",
                equal("alive_variance_energy_definition", sk, k * varx),
            ),
            (
                r"(1-\varepsilon_O)S_k",
                ge(
                    "global_positional_outlier_capture",
                    outliers.iter().map(|&i| (x[i] - mx).powi(2)).sum(),
                    (1. - eps) * sk,
                ),
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\sum_{i\in H_k}",
        input.clone(),
        base.clone(),
        vec![
            upper("actual_high_energy_fraction", ch * sk, high_x),
            positive_identity(
                "high_energy_constant",
                ch,
                (1. - eps) * (1. - d * d / (2. * rvar)),
            ),
        ],
    )?;
    // The second display with the same leading token is selected by its B term.
    builder.shown(
        label,
        r"\geq(1-\varepsilon_O)B",
        input.clone(),
        base.clone(),
        vec![
            upper(
                "selected_between_position_energy",
                (1. - eps) * between_x,
                high_x,
            ),
            upper(
                "within_remainder_subtracted",
                (1. - eps) * sk * (1. - k * d * d / (2. * sk)),
                (1. - eps) * between_x,
            ),
        ],
    )?;
    target_concentration(builder, &input, &x, &v, &high, n, ch, dvalid2.sqrt(), lv)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn packing_identities(
    builder: &mut Builder,
    cloud: &Value,
    x: &[f64],
    v: &[f64],
    lv: f64,
    la: f64,
    dx: f64,
    dv: f64,
    close: f64,
) -> Result<()> {
    let label = "lem-phase-space-packing";
    let k = x.len() as f64;
    let varx = variance(x);
    let varv = variance(v);
    let varh = varx + lv * varv;
    let diameter2 = dx * dx + la * dv * dv;
    let unique = (0..x.len())
        .flat_map(|i| (i + 1..x.len()).map(move |j| (i, j)))
        .collect::<Vec<_>>();
    let distance2 = |i: usize, j: usize| (x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2);
    let phase2 = |i: usize, j: usize| (x[i] - x[j]).powi(2) + lv * (v[i] - v[j]).powi(2);
    let close_pairs = unique
        .iter()
        .copied()
        .filter(|&(i, j)| distance2(i, j) < close * close)
        .collect::<Vec<_>>();
    let far_pairs = unique
        .iter()
        .copied()
        .filter(|&(i, j)| distance2(i, j) >= close * close)
        .collect::<Vec<_>>();
    let count = unique.len() as f64;
    let fc = close_pairs.len() as f64 / count;
    let ordered_x = 2.
        * unique
            .iter()
            .map(|&(i, j)| (x[i] - x[j]).powi(2))
            .sum::<f64>();
    let ordered_v = 2.
        * unique
            .iter()
            .map(|&(i, j)| (v[i] - v[j]).powi(2))
            .sum::<f64>();
    let near_sum = close_pairs.iter().map(|&(i, j)| phase2(i, j)).sum::<f64>();
    let far_sum = far_pairs.iter().map(|&(i, j)| phase2(i, j)).sum::<f64>();
    let loose = fc * close * close + (1. - fc) * diameter2;
    let g = (diameter2 - 2. * varh) / (diameter2 - close * close);
    let input = json!({"cloud":cloud,"proximity_threshold":close,"unique_pairs":unique,"close_pairs":close_pairs,"far_pairs":far_pairs,"computed":{"variance_x":varx,"variance_v":varv,"variance_h":varh,"ordered_x_pair_squared_sum":ordered_x,"ordered_v_pair_squared_sum":ordered_v,"near_phase_sum":near_sum,"far_phase_sum":far_sum,"close_fraction":fc,"g_variance":g,"D_valid_squared":diameter2}});
    let hyps = vec![
        hypothesis(
            "positive_ordered_metric_and_proximity",
            k >= 2. && lv > 0. && lv <= la && close > 0. && close < diameter2.sqrt(),
        ),
        hypothesis("positive_packing_regime", varh > close * close / 2.),
    ];
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (r"k \geq 2", condition("at_least_two_phase_points", k >= 2.)),
            (r"k \ge 2", condition("at_least_two_pair_points", k >= 2.)),
            (
                r"\{(x_i, v_i)\}_{i=1}^k",
                condition(
                    "finite_matching_phase_rows",
                    x.len() == v.len() && x.iter().chain(v).all(|v| v.is_finite()),
                ),
            ),
            (
                r"0<d_{\text{close}}<D_{\text{valid}}",
                vec![
                    hypothesis("strict_positive_close_threshold", close > 0.),
                    hypothesis(
                        "threshold_below_full_phase_diameter",
                        close < diameter2.sqrt(),
                    ),
                ],
            ),
            (
                r"\lambda_v \le \lambda_{\text{alg}}",
                le("actual_variance_vs_algorithmic_velocity_weights", lv, la),
            ),
            (
                r"D_{\text{valid}}^2 := D_x^2 + \lambda_{\text{alg}} D_v^2",
                equal(
                    "phase_diameter_definition",
                    diameter2,
                    dx * dx + la * dv * dv,
                ),
            ),
            (
                r"f_{\text{close}} = N_{\text{close}} / \binom{k}{2}",
                equal(
                    "actual_unique_close_pair_fraction",
                    fc,
                    close_pairs.len() as f64 / (k * (k - 1.) / 2.),
                ),
            ),
            (
                r"N_{\text{close}} = f_{\text{close}} \binom{k}{2}",
                equal(
                    "unique_close_pair_count",
                    close_pairs.len() as f64,
                    fc * k * (k - 1.) / 2.,
                ),
            ),
            (
                r"N_{\text{far}} = (1 - f_{\text{close}}) \binom{k}{2}",
                equal(
                    "unique_far_pair_count",
                    far_pairs.len() as f64,
                    (1. - fc) * k * (k - 1.) / 2.,
                ),
            ),
            (
                r"\mathrm{Var}_h(S_k) > R_{\text{pack}}^2 := d_{\text{close}}^2 / 2",
                vec![
                    hypothesis("strict_packing_variance_floor", varh > close * close / 2.),
                    positive_identity(
                        "packing_threshold_definition",
                        close * close / 2.,
                        close.powi(2) * 0.5,
                    ),
                ],
            ),
            (
                r"g(\mathrm{Var}_h(S_k)) < 1",
                strict("strict_nontrivial_packing_fraction", g, 1.),
            ),
            (
                r"g(\mathrm{Var}_h) < 1",
                strict("strict_nontrivial_packing_proof", g, 1.),
            ),
            (
                r"(k-1)/(2k) < 1/2",
                strict("strict_pair_count_prefactor", (k - 1.) / (2. * k), 0.5),
            ),
            (
                r"d_{\text{close}} < D_{\text{valid}}",
                strict(
                    "positive_rearrangement_denominator",
                    close,
                    diameter2.sqrt(),
                ),
            ),
            (
                r"f_{\text{close}} \le g(\mathrm{Var}_h(S_k))",
                le("computed_packing_bound", fc, g),
            ),
            (
                r"g(V) := (D_{\text{valid}}^2 - 2V) / (D_{\text{valid}}^2 - d_{\text{close}}^2)",
                equal(
                    "packing_affine_function_at_actual_variance",
                    g,
                    (diameter2 - 2. * varh) / (diameter2 - close * close),
                ),
            ),
            (
                r"\mathrm{Var}_h > d_{\text{close}}^2 / 2",
                strict("packing_threshold_hypothesis", close * close / 2., varh),
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\mathrm{Var}_h(S_k) :=",
        input.clone(),
        hyps.clone(),
        equal(
            "full_hypocoercive_variance_definition",
            varh,
            varx + lv * varv,
        ),
    )?;
    builder.shown(
        label,
        r"2k^2 \mathrm{Var}_x(S_k)",
        input.clone(),
        hyps.clone(),
        equal(
            "ordered_position_pair_identity",
            2. * k * k * varx,
            ordered_x,
        ),
    )?;
    let first = ordered_x;
    let expanded = (0..x.len())
        .flat_map(|i| (0..x.len()).map(move |j| (i, j)))
        .map(|(i, j)| x[i] * x[i] - 2. * x[i] * x[j] + x[j] * x[j])
        .sum::<f64>();
    let next = 2. * k * x.iter().map(|x| x * x).sum::<f64>() - 2. * (k * mean(x)).powi(2);
    let next2 = 2. * k * x.iter().map(|x| x * x).sum::<f64>() - 2. * k * k * mean(x).powi(2);
    let last = 2. * k * (k * varx + k * mean(x).powi(2)) - 2. * k * k * mean(x).powi(2);
    builder.shown(
        label,
        r"\begin{aligned}",
        input.clone(),
        hyps.clone(),
        vec![
            relative("position_pair_expansion_first", first, expanded, first),
            relative("position_pair_expansion_mean", expanded, next, first),
            relative("position_pair_expansion_mean_square", next, next2, first),
            relative(
                "position_pair_expansion_centered_energy",
                next2,
                last,
                first,
            ),
            relative(
                "position_pair_expansion_variance",
                last,
                2. * k * k * varx,
                first,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"2k^2 \mathrm{Var}_v(S_k)",
        input.clone(),
        hyps.clone(),
        equal(
            "ordered_velocity_pair_identity",
            2. * k * k * varv,
            ordered_v,
        ),
    )?;
    builder.shown(
        label,
        r"2k^2 \mathrm{Var}_h(S_k)",
        input.clone(),
        hyps.clone(),
        vec![
            relative(
                "weighted_pair_identity_variance",
                2. * k * k * varh,
                2. * k * k * (varx + lv * varv),
                ordered_x + lv * ordered_v,
            ),
            relative(
                "weighted_pair_identity_pair_sum",
                2. * k * k * varh,
                ordered_x + lv * ordered_v,
                ordered_x + lv * ordered_v,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\mathrm{Var}_h(S_k) = \frac{1}{k^2} \sum_{i<j}",
        input.clone(),
        hyps.clone(),
        equal(
            "unique_pair_phase_variance",
            varh,
            (near_sum + far_sum) / (k * k),
        ),
    )?;
    builder.shown(
        label,
        r"\mathrm{Var}_h(S_k) = \frac{1}{k^2} \left(",
        input.clone(),
        hyps.clone(),
        equal(
            "near_far_partition_variance",
            varh,
            (near_sum + far_sum) / (k * k),
        ),
    )?;
    builder.shown(
        label,
        r"\mathrm{Var}_h(S_k) \le \frac{1}{k^2}",
        input.clone(),
        hyps.clone(),
        le(
            "near_far_count_diameter_bound",
            varh,
            (close_pairs.len() as f64 * close * close + far_pairs.len() as f64 * diameter2)
                / (k * k),
        ),
    )?;
    builder.shown(
        label,
        r"\mathrm{Var}_h(S_k) \le \frac{\binom{k}{2}}",
        input.clone(),
        hyps.clone(),
        vec![
            upper(
                "fraction_count_variance_bound",
                varh,
                count / (k * k) * loose,
            ),
            relative(
                "fraction_count_prefactor_identity",
                count / (k * k) * loose,
                (k - 1.) / (2. * k) * (fc * (close * close - diameter2) + diameter2),
                loose,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\mathrm{Var}_h(S_k) <",
        input.clone(),
        hyps.clone(),
        strict("strict_simplified_phase_variance_bound", varh, loose / 2.),
    )?;
    builder.shown(
        label,
        r"2\mathrm{Var}_h(S_k) &<",
        input.clone(),
        hyps.clone(),
        vec![
            hypothesis(
                "twice_variance_rearrangement",
                2. * varh < fc * (close * close - diameter2) + diameter2,
            ),
            hypothesis(
                "subtracted_diameter_rearrangement",
                2. * varh - diameter2 < fc * (close * close - diameter2),
            ),
        ],
    )?;
    builder.shown(
        label,
        r"f_{\text{close}} <",
        input.clone(),
        hyps.clone(),
        vec![
            hypothesis(
                "strict_negative_denominator_reversal",
                fc < (2. * varh - diameter2) / (close * close - diameter2),
            ),
            relative(
                "equivalent_positive_denominator_fraction",
                (2. * varh - diameter2) / (close * close - diameter2),
                g,
                g,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"f_{\text{close}} \le g",
        input.clone(),
        hyps.clone(),
        vec![
            upper("packing_fraction_bound", fc, g),
            relative(
                "packing_bound_function_definition",
                g,
                (diameter2 - 2. * varh) / (diameter2 - close * close),
                g,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\frac{D_{\text{valid}}^2 - 2\mathrm{Var}_h}{D_{\text{valid}}^2 - d_{\text{close}}^2}",
        input.clone(),
        hyps.clone(),
        vec![
            hypothesis("packing_implication_first", g < 1.),
            hypothesis(
                "packing_implication_second",
                diameter2 - 2. * varh < diameter2 - close * close,
            ),
            hypothesis("packing_implication_third", varh > close * close / 2.),
        ],
    )?;
    for &(i, j) in &unique {
        let alg = distance2(i, j);
        let physical = phase2(i, j);
        let pair = json!({"packing":input,"i":i,"j":j,"algorithmic_distance":alg.sqrt(),"algorithmic_squared_distance":alg,"physical_q_squared_distance":physical});
        builder.inline(
            label,
            r"i<j",
            pair.clone(),
            condition("unordered_pairs_use_one_orientation", i < j),
        )?;
        for quote in [
            r"d_{\text{alg}}(i, j)^2 := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2",
            r"d_{\text{alg}}(i,j)^2 = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2",
        ] {
            builder.inline(
                label,
                quote,
                pair.clone(),
                equal(
                    "actual_pair_metric_definition",
                    alg,
                    (x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2),
                ),
            )?;
        }
        if alg < close * close {
            for quote in [
                r"d_{\text{alg}}(i, j) < d_{\text{close}}",
                r"d_{\text{alg}}(i,j) < d_{\text{close}}",
            ] {
                builder.inline(
                    label,
                    quote,
                    pair.clone(),
                    strict("actual_close_pair_branch", alg.sqrt(), close),
                )?;
            }
            builder.inline(label,r"d_{\text{alg}}(i,j)^2 = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2 < d_{\text{close}}^2",pair.clone(),vec![relative("close_pair_squared_distance_definition",alg,(x[i]-x[j]).powi(2)+la*(v[i]-v[j]).powi(2),alg),hypothesis("strict_close_pair_squared_threshold",alg<close*close)])?;
            builder.shown(
                label,
                r"\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2 \le \|x_i",
                pair,
                hyps.clone(),
                vec![
                    upper("close_pair_weight_comparison", physical, alg),
                    relative(
                        "close_pair_algorithmic_identity",
                        alg,
                        (x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2),
                        alg,
                    ),
                    hypothesis("close_pair_strict_threshold", alg < close * close),
                ],
            )?;
        } else {
            builder.inline(
                label,
                r"d_{\text{alg}}(i,j) \ge d_{\text{close}}",
                pair.clone(),
                ge("actual_far_pair_branch", alg.sqrt(), close),
            )?;
            builder.shown(
                label,
                r"\|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2 \le D_x",
                pair,
                hyps.clone(),
                vec![
                    upper(
                        "far_pair_physical_diameter",
                        physical,
                        dx * dx + lv * dv * dv,
                    ),
                    upper(
                        "far_pair_weighted_diameter_comparison",
                        dx * dx + lv * dv * dv,
                        diameter2,
                    ),
                    relative(
                        "far_pair_diameter_definition",
                        diameter2,
                        dx * dx + la * dv * dv,
                        diameter2,
                    ),
                ],
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn geometric_separation(
    builder: &mut Builder,
    input: &Value,
    x: &[f64],
    v: &[f64],
    clusters: &[Vec<usize>],
    high: &[bool],
    centers: &[f64],
    radii: &[f64],
    la: f64,
    min_size: usize,
) -> Result<()> {
    let label = "lem-geometric-separation-of-partition";
    let k = x.len() as f64;
    let cluster_high = clusters
        .iter()
        .map(|g| g.iter().all(|&i| high[i]))
        .collect::<Vec<_>>();
    let center_gap = clusters
        .iter()
        .enumerate()
        .filter(|(i, _)| cluster_high[*i])
        .flat_map(|(i, _)| {
            clusters
                .iter()
                .enumerate()
                .filter(|(j, _)| !cluster_high[*j])
                .map(move |(j, _)| (centers[i] - centers[j]).abs())
        })
        .fold(f64::INFINITY, f64::min);
    let rho_h = radii
        .iter()
        .enumerate()
        .filter(|(i, _)| cluster_high[*i])
        .map(|(_, r)| *r)
        .fold(0., f64::max);
    let rho_l = radii
        .iter()
        .enumerate()
        .filter(|(i, _)| !cluster_high[*i])
        .map(|(_, r)| *r)
        .fold(0., f64::max);
    let rl = clusters
        .iter()
        .enumerate()
        .filter(|(i, _)| !cluster_high[*i])
        .flat_map(|(_, g)| g.iter().flat_map(move |&i| g.iter().map(move |&j| (i, j))))
        .map(|(i, j)| ((x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2)).sqrt())
        .fold(0., f64::max);
    let comparison = 1_f64;
    let dh = comparison * (center_gap - rho_h - rho_l);
    let input = json!({"cloud":input,"s_HL":center_gap,"rho_H":rho_h,"rho_L":rho_l,"actual_R_L":rl,"m_x":comparison,"D_H":dh,"f_c":0.04});
    let hyps = vec![
        hypothesis(
            "actual_nonempty_high_low",
            high.iter().any(|&b| b) && high.iter().any(|&b| !b),
        ),
        hypothesis(
            "directly_verified_positive_margin",
            comparison > 0. && dh > rl && rl > 0.,
        ),
    ];
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"m_x>0",
                condition("positive_unsquashed_position_comparison", comparison > 0.),
            ),
            (
                r"m_x=1",
                equal("unsquashed_position_comparison", comparison, 1.),
            ),
            (
                r"R_L>0",
                condition("positive_low_cluster_diameter", rl > 0.),
            ),
            (
                r"f_c=0.04",
                equal("minimum_valid_companion_fraction", 0.04, 0.8 * 0.05),
            ),
        ],
    )?;
    builder.shown(
        label,
        r"s_{HL}=",
        input.clone(),
        hyps.clone(),
        vec![
            relative(
                "actual_minimum_cross_center_gap",
                center_gap,
                clusters
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| cluster_high[*i])
                    .flat_map(|(i, _)| {
                        clusters
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| !cluster_high[*j])
                            .map(move |(j, _)| (centers[i] - centers[j]).abs())
                    })
                    .fold(f64::INFINITY, f64::min),
                center_gap,
            ),
            relative(
                "actual_maximum_high_radius",
                rho_h,
                radii
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| cluster_high[*i])
                    .map(|(_, r)| *r)
                    .fold(0., f64::max),
                rho_h,
            ),
            relative(
                "actual_maximum_low_radius",
                rho_l,
                radii
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !cluster_high[*i])
                    .map(|(_, r)| *r)
                    .fold(0., f64::max),
                rho_l,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"D_H:=",
        input.clone(),
        hyps.clone(),
        vec![
            positive_identity(
                "geometric_margin_definition",
                dh,
                comparison * (center_gap - rho_h - rho_l),
            ),
            hypothesis("actual_margin_strictly_exceeds_low_diameter", dh > rl),
        ],
    )?;
    for (gi, g) in clusters.iter().enumerate() {
        let c = mean(&g.iter().map(|&i| x[i]).collect::<Vec<_>>());
        let row =
            json!({"geometry":input,"cluster":gi,"members":g,"center_x":c,"radius_x":radii[gi]});
        builder.inline(
            label,
            r"\mu_{x,G}=|G|^{-1}\sum_{i\in G}x_i",
            row.clone(),
            equal(
                "actual_cluster_center_definition",
                centers[gi],
                g.iter().map(|&i| x[i]).sum::<f64>() / g.len() as f64,
            ),
        )?;
        builder.inline(
            label,
            r"\rho_G=\max_{i\in G}|x_i-\mu_{x,G}|",
            row.clone(),
            equal(
                "actual_cluster_radius_definition",
                radii[gi],
                g.iter().map(|&i| (x[i] - c).abs()).fold(0., f64::max),
            ),
        )?;
        if !cluster_high[gi] {
            bind_inline(
                builder,
                label,
                &row,
                vec![
                    (
                        r"|G_j|\ge\max(5,\lceil0.05k\rceil)",
                        ge("low_clusters_are_valid", g.len() as f64, min_size as f64),
                    ),
                    (
                        r"|G_j|-1\ge0.04k",
                        ge(
                            "actual_eligible_companion_count",
                            (g.len() - 1) as f64,
                            0.04 * k,
                        ),
                    ),
                    (
                        r"n\ge5",
                        ge("valid_cluster_minimum_size", g.len() as f64, 5.),
                    ),
                    (
                        r"n-1\ge(4/5)n",
                        ge(
                            "self_excluded_fraction",
                            (g.len() - 1) as f64,
                            0.8 * g.len() as f64,
                        ),
                    ),
                    (
                        r"n\ge0.05k",
                        ge(
                            "valid_cluster_population_fraction",
                            g.len() as f64,
                            0.05 * k,
                        ),
                    ),
                    (
                        r"n-1\ge0.04k",
                        ge(
                            "self_excluded_population_fraction",
                            (g.len() - 1) as f64,
                            0.04 * k,
                        ),
                    ),
                ],
            )?;
        }
    }
    for i in 0..x.len() {
        for j in 0..x.len() {
            let alg = ((x[i] - x[j]).powi(2) + la * (v[i] - v[j]).powi(2)).sqrt();
            let pair = json!({"geometry":input,"i":i,"j":j,"algorithmic_distance":alg});
            builder.inline(
                label,
                r"d_{\rm alg}(i,j)\ge m_x|x_i-x_j|",
                pair.clone(),
                ge(
                    "actual_position_metric_comparison",
                    alg,
                    comparison * (x[i] - x[j]).abs(),
                ),
            )?;
            if high[i] && !high[j] {
                builder.inline(
                    label,
                    r"d_{\rm alg}(i,j)\ge D_H>R_L",
                    pair.clone(),
                    vec![
                        upper("actual_cross_pair_margin", dh, alg),
                        hypothesis("strict_cross_margin", dh > rl),
                    ],
                )?;
                let gi = clusters.iter().position(|g| g.contains(&i)).unwrap();
                let gj = clusters.iter().position(|g| g.contains(&j)).unwrap();
                let intermediate = (centers[gi] - centers[gj]).abs()
                    - (x[i] - centers[gi]).abs()
                    - (x[j] - centers[gj]).abs();
                builder.shown(
                    label,
                    r"|x_i-x_j|\ge",
                    pair,
                    hyps.clone(),
                    vec![
                        upper(
                            "cross_pair_triangle_first",
                            intermediate,
                            (x[i] - x[j]).abs(),
                        ),
                        upper(
                            "cross_pair_triangle_cluster_radii",
                            center_gap - rho_h - rho_l,
                            intermediate,
                        ),
                    ],
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn target_concentration(
    builder: &mut Builder,
    cloud: &Value,
    x: &[f64],
    v: &[f64],
    high: &[bool],
    n: f64,
    ch: f64,
    diameter: f64,
    lv: f64,
) -> Result<()> {
    let label = "lem-error-concentration-target-set";
    let mx = mean(x);
    let mv = mean(v);
    // All right positions coincide, while velocities equal their left values.
    // Every permutation has the same positional cost, and velocity cost is
    // nonnegative. The matching-velocity plan attains the lower bound, proving
    // optimality for the full positive quadratic form diag(1, lambda_v).
    let right_x = vec![mx; x.len()];
    let right_v = v.to_vec();
    let right_mx = mean(&right_x);
    let right_mv = mean(&right_v);
    let deltax = x
        .iter()
        .zip(&right_x)
        .map(|(l, r)| (l - mx) - (r - right_mx))
        .collect::<Vec<_>>();
    let deltav = v
        .iter()
        .zip(&right_v)
        .map(|(l, r)| (l - mv) - (r - right_mv))
        .collect::<Vec<_>>();
    let phase_cost = deltax
        .iter()
        .zip(&deltav)
        .map(|(dx, dv)| dx * dx + lv * dv * dv)
        .sum::<f64>()
        / n;
    let position_cost = deltax.iter().map(|dx| dx * dx).sum::<f64>() / n;
    let velocity_remainder = lv * deltav.iter().map(|dv| dv * dv).sum::<f64>() / n;
    let sk = x.iter().map(|x| (x - mx).powi(2)).sum::<f64>();
    let sj = right_x.iter().map(|x| (x - right_mx).powi(2)).sum::<f64>();
    let common = vec![true; x.len()];
    // Both physical comparison clouds have the same alive mass. The omitted
    // high-error row is fit, rather than being falsely marked noncommon in an
    // equal-alive phase transport. Fitness is a supplied retained realization.
    let mut retained_fitness = high
        .iter()
        .map(|&h| if h { 0.8 } else { 1.4 })
        .collect::<Vec<_>>();
    retained_fitness[high.iter().position(|&h| h).unwrap()] = 1.7;
    let unfit = retained_fitness
        .iter()
        .map(|&v| v <= mean(&retained_fitness))
        .collect::<Vec<_>>();
    let target = high
        .iter()
        .enumerate()
        .map(|(i, &h)| h && common[i] && unfit[i])
        .collect::<Vec<_>>();
    let a = 0.5_f64;
    let b = sj / n + velocity_remainder / 2.;
    let mj = right_x
        .iter()
        .enumerate()
        .filter(|(i, _)| high[*i])
        .map(|(_, x)| (x - right_mx).powi(2))
        .sum::<f64>()
        / n;
    let bt = deltax
        .iter()
        .enumerate()
        .filter(|(i, _)| high[*i] && !target[*i])
        .map(|(_, dx)| dx * dx)
        .sum::<f64>()
        / n;
    let full_h = deltax
        .iter()
        .enumerate()
        .filter(|(i, _)| high[*i])
        .map(|(_, dx)| dx * dx)
        .sum::<f64>()
        / n;
    let target_error = deltax
        .iter()
        .enumerate()
        .filter(|(i, _)| target[*i])
        .map(|(_, dx)| dx * dx)
        .sum::<f64>()
        / n;
    let high_energy = x
        .iter()
        .enumerate()
        .filter(|(i, _)| high[*i])
        .map(|(_, x)| (x - mx).powi(2))
        .sum::<f64>();
    let input = json!({"cloud":cloud,"right_positions":right_x,"right_velocities":right_v,"centered_comparison_x":deltax,"centered_comparison_v":deltav,"positive_phase_quadratic":{"matrix":[[1.,0.],[0.,lv]],"minimum_eigenvalue":1_f64.min(lv),"cross_coefficient":0.},"phase_optimality":"right centered positions are identically zero for every permutation; each velocity term is nonnegative; matching velocities attains zero velocity cost","I_11":common,"U_k":unfit,"retained_prescribed_fitness":retained_fitness,"retained_fitness_mean":mean(&retained_fitness),"H":high,"T":target,"slots":n,"D_valid_for_bounded_position_remainders":diameter,"a":a,"b":b,"c_H":ch,"M_j":mj,"B_T":bt,"computed":{"V_struct_full_phase":phase_cost,"V_x_struct":position_cost,"explicit_velocity_remainder":velocity_remainder,"S_k":sk,"S_j":sj,"full_high_error":full_h,"target_error":target_error}});
    let hyps = vec![
        hypothesis(
            "strict_positive_full_phase_form",
            lv > 0. && 1_f64.min(lv) > 0.,
        ),
        hypothesis(
            "matching_velocities_optimal_phase_transport",
            deltav.iter().all(|&v| v == 0.) && right_x.iter().all(|&x| x == mx),
        ),
        hypothesis(
            "same_comparison_rows_and_normalization",
            x.len() == v.len()
                && x.len() == high.len()
                && high.len() == common.len()
                && n >= x.len() as f64,
        ),
        hypothesis(
            "strict_positive_concentration_coefficients",
            ch > 0. && a > 0.,
        ),
    ];
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"T=I_{11}\cap U_k\cap H",
                condition(
                    "actual_target_intersection",
                    target
                        .iter()
                        .enumerate()
                        .all(|(i, &t)| t == (common[i] && unfit[i] && high[i])),
                ),
            ),
            (
                r"c_H,a>0",
                condition(
                    "positive_error_concentration_coefficients",
                    ch > 0. && a > 0.,
                ),
            ),
            (
                r"N^{-1}\sum_{i\in H}\|\delta_{x,j,i}\|^2\leq M_j",
                le("actual_other_swarm_high_position_energy", mj, mj),
            ),
            (
                r"N^{-1}\sum_{i\in H\setminus T}\|\Delta\delta_{x,i}\|^2\leq B_T",
                le("actual_omitted_comparison_error", bt, bt),
            ),
            (
                r"M_j\leq D_{\mathrm{valid}}^2",
                le(
                    "other_position_diameter_energy_bound",
                    mj,
                    diameter * diameter,
                ),
            ),
            (
                r"B_T\leq4D_{\mathrm{valid}}^2",
                le(
                    "omitted_position_error_diameter_bound",
                    bt,
                    4. * diameter * diameter,
                ),
            ),
            (
                r"a=1/2",
                equal("position_phase_relation_factor", a, 1. / 2.),
            ),
            (
                r"b=S_j/N\leq D_{\mathrm{valid}}^2",
                vec![
                    relative(
                        "matching_velocity_remainder_is_retained_zero",
                        b,
                        sj / n,
                        b.abs().max(sj / n),
                    ),
                    upper(
                        "other_position_offset_diameter_bound",
                        sj / n,
                        diameter * diameter,
                    ),
                ],
            ),
            (
                r"V_{x,\mathrm{struct}}\leq2(S_1+S_2)/N",
                le(
                    "actual_positional_transport_variance_bound",
                    position_cost,
                    2. * (sk + sj) / n,
                ),
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\sum_{i\in H}\|\delta_{x,k,i}\|^2",
        input.clone(),
        hyps.clone(),
        vec![
            upper("actual_high_position_capture", ch * sk, high_energy),
            upper(
                "positive_phase_error_including_velocity_remainder",
                a * phase_cost - b,
                sk / n,
            ),
            relative(
                "full_phase_cost_not_position_surrogate",
                phase_cost,
                position_cost + velocity_remainder,
                phase_cost,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\frac1N\sum_{i\in T}",
        input.clone(),
        hyps.clone(),
        ge(
            "full_phase_target_error_with_explicit_complement",
            target_error,
            ch * a / 2. * phase_cost - (ch * b / 2. + mj + bt),
        ),
    )?;
    builder.shown(
        label,
        r"\frac1N\sum_{i\in H}",
        input.clone(),
        hyps.clone(),
        vec![
            upper(
                "first_high_error_comparison_bound",
                ch * sk / (2. * n) - mj,
                full_h,
            ),
            upper(
                "second_high_error_full_phase_bound",
                ch * a / 2. * phase_cost - ch * b / 2. - mj,
                ch * sk / (2. * n) - mj,
            ),
            relative(
                "actual_target_complement_identity",
                target_error,
                full_h - bt,
                full_h,
            ),
        ],
    )?;
    for i in 0..x.len() {
        let u = x[i] - mx;
        let w = right_x[i] - right_mx;
        let lhs = (u - w).powi(2) - 0.5 * u * u + w * w;
        let rhs = 0.5 * (u - 2. * w).powi(2);
        builder.inline(
            label,
            "\\|u-v\\|^2-\\tfrac12\\|u\\|^2+\\|v\\|^2\n=\\tfrac12\\|u-2v\\|^2\\geq0",
            json!({"target_fixture":input,"row":i,"u":u,"other_centered_position":w}),
            vec![
                relative("half_error_identity", lhs, rhs, u * u + w * w),
                upper("half_error_identity_nonnegative", 0., rhs),
            ],
        )?;
    }
    Ok(())
}

fn logistic_derivative(z: f64) -> f64 {
    let e = (-z.abs()).exp();
    2. * e / (1. + e).powi(2)
}
fn mean_value_point(left: f64, right: f64, slope: f64) -> Result<f64> {
    let low = left.min(right);
    let high = left.max(right);
    let cuts = if low < 0. && high > 0. {
        vec![low, 0., high]
    } else {
        vec![low, high]
    };
    for span in cuts.windows(2) {
        let mut a = span[0];
        let mut b = span[1];
        let mut fa = logistic_derivative(a) - slope;
        let fb = logistic_derivative(b) - slope;
        if fa == 0. {
            return Ok(a);
        }
        if fb == 0. {
            return Ok(b);
        }
        if fa * fb > 0. {
            continue;
        }
        for _ in 0..90 {
            let mid = (a + b) / 2.;
            let fm = logistic_derivative(mid) - slope;
            if fm == 0. {
                return Ok(mid);
            }
            if fm * fa > 0. {
                a = mid;
                fa = fm;
            } else {
                b = mid;
            }
        }
        return Ok((a + b) / 2.);
    }
    Err(error(
        "no numerical mean-value point for native logistic pair",
    ))
}

fn realized_scalar_vectors(builder: &mut Builder) -> Result<()> {
    for base in [
        vec![-0.8, -0.2, 0.1, 0.7],
        vec![-1., -0.9, 0.8, 1.],
        vec![-0.01, -0.009, 0.008, 0.01],
    ] {
        for copies in [1_usize, 2, 8, 32] {
            let raw = base
                .iter()
                .flat_map(|&v| std::iter::repeat_n(v, copies))
                .collect::<Vec<_>>();
            let k = raw.len() as f64;
            let mu = mean(&raw);
            let s2 = variance(&raw);
            let pair_sum = raw
                .iter()
                .flat_map(|&a| raw.iter().map(move |&b| (a - b).powi(2)))
                .sum::<f64>();
            let gap = range(&raw);
            let kappa = s2 * 0.9;
            let sigma_min = 0.1_f64;
            let vmax = base.iter().map(|x| x.abs()).fold(0., f64::max);
            let sigma_max = (vmax * vmax + sigma_min * sigma_min).sqrt();
            let obs = ObservationBatch::positions(TensorBatch::vectors(
                raw.len(),
                1,
                vec![0.; raw.len()],
            )?);
            let (z, stats) =
                Standardizer::Global { sigma_min }.apply(&raw, &vec![true; raw.len()], &obs)?;
            let map = PositiveMap::Logistic {
                amplitude: 2.,
                floor: 1e-6,
            };
            let output = z.iter().map(|&z| map.map(z)).collect::<Result<Vec<_>>>()?;
            let z_support = 2. * vmax / sigma_min;
            let min_derivative = logistic_derivative(z_support);
            let input = json!({"raw":raw,"native_scores":z,"native_output":output,"native_statistics":stats,"copies_of_each_base_value":copies,"minimum_scale":sigma_min,"family_maximum_absolute_raw_value":vmax,"family_sigma_max":sigma_max,"operational_score_radius":z_support,"family_logistic_derivative_minimum":min_derivative,"native_positive_map":map,"computed":{"mean":mu,"variance":s2,"ordered_squared_pair_sum":pair_sum,"largest_gap":gap,"variance_lower_bound":kappa}});
            let hyps = vec![
                hypothesis(
                    "complete_finite_nonconstant_raw_vector",
                    k >= 2. && raw.iter().all(|v| v.is_finite()) && s2 > 0.,
                ),
                hypothesis(
                    "native_shared_normalizer_with_valid_family_bounds",
                    stats
                        .scale
                        .iter()
                        .all(|&s| s >= sigma_min && s <= sigma_max)
                        && raw.iter().all(|v| v.abs() <= vmax),
                ),
                hypothesis(
                    "representable_positive_derivative_floor",
                    min_derivative.is_finite() && min_derivative > 0.,
                ),
            ];
            let label = "lem-variance-to-gap";
            bind_inline(
                builder,
                label,
                &input,
                vec![
                    (
                        r"\{v_i\}_{i=1}^k",
                        condition(
                            "scalar_realized_vector_contract",
                            raw.len() >= 2 && raw.iter().all(|v| v.is_finite()),
                        ),
                    ),
                    (
                        r"k \ge 2",
                        condition("realized_vector_minimum_size", k >= 2.),
                    ),
                    (
                        r"\text{Var}(\{v_i\}) \geq \kappa > 0",
                        vec![
                            upper("actual_variance_lower_bound", kappa, s2),
                            hypothesis("strict_positive_realized_variance_floor", kappa > 0.),
                        ],
                    ),
                    (
                        r"\text{Var}(\{v_i\}) = \frac{1}{k}\sum_i v_i^2 - (\frac{1}{k}\sum_i v_i)^2",
                        equal(
                            "realized_variance_second_moment_identity",
                            s2,
                            raw.iter().map(|v| v * v).sum::<f64>() / k - mu * mu,
                        ),
                    ),
                    (
                        r"\Delta_{\text{max}} := \max_{i,j} |v_i - v_j|",
                        equal(
                            "largest_realized_pair_gap",
                            gap,
                            raw.iter()
                                .flat_map(|&a| raw.iter().map(move |&b| (a - b).abs()))
                                .fold(0., f64::max),
                        ),
                    ),
                    (
                        r"\mathrm{Var}(\{v_i\}) \geq \kappa",
                        ge("realized_variance_premise", s2, kappa),
                    ),
                    (
                        r"\kappa \le \frac{1}{2} \Delta_{\max}^2",
                        le("variance_floor_to_squared_max_gap", kappa, 0.5 * gap * gap),
                    ),
                    (
                        r"\Delta_{\max}^2 \ge 2\kappa",
                        ge("realized_squared_max_gap_lower", gap * gap, 2. * kappa),
                    ),
                ],
            )?;
            builder.shown(
                label,
                r"\max_{i,j}",
                input.clone(),
                hyps.clone(),
                ge("realized_maximum_gap_lower", gap, (2. * kappa).sqrt()),
            )?;
            builder.shown(
                label,
                r"\mathrm{Var}(\{v_i\}) =",
                input.clone(),
                hyps.clone(),
                equal(
                    "ordered_pair_realized_variance_identity",
                    s2,
                    pair_sum / (2. * k * k),
                ),
            )?;
            builder.shown(
                label,
                r"\sum_{i=1}^k \sum_{j=1}^k (v_i - v_j)^2",
                input.clone(),
                hyps.clone(),
                vec![
                    upper(
                        "ordered_pair_sum_max_gap_bound",
                        pair_sum,
                        k * k * gap * gap,
                    ),
                    relative(
                        "ordered_max_gap_constant_sum",
                        raw.iter()
                            .flat_map(|_| raw.iter().map(|_| gap * gap))
                            .sum::<f64>(),
                        k * k * gap * gap,
                        k * k * gap * gap,
                    ),
                ],
            )?;
            builder.shown(
                label,
                r"\mathrm{Var}(\{v_i\}) \le",
                input.clone(),
                hyps.clone(),
                vec![
                    upper("variance_by_max_gap", s2, k * k * gap * gap / (2. * k * k)),
                    relative(
                        "max_gap_variance_prefactor",
                        k * k * gap * gap / (2. * k * k),
                        0.5 * gap * gap,
                        gap * gap,
                    ),
                ],
            )?;
            builder.shown(
                label,
                r"\kappa \le",
                input.clone(),
                hyps.clone(),
                vec![
                    upper("first_variance_to_gap_chain", kappa, s2),
                    upper("second_variance_to_gap_chain", s2, 0.5 * gap * gap),
                ],
            )?;
            let label = "lem-raw-gap-to-rescaled-gap";
            bind_inline(
                builder,
                label,
                &input,
                vec![
                    (
                        r"|v_i|\le V_{\max}<\infty",
                        vec![
                            upper(
                                "actual_complete_raw_support",
                                raw.iter().map(|v| v.abs()).fold(0., f64::max),
                                vmax,
                            ),
                            hypothesis("finite_family_raw_ceiling", vmax.is_finite()),
                        ],
                    ),
                    (
                        r"0<\sigma'_{\min,\mathrm{patch}}\le\sigma'\le\sigma'_{\max}",
                        vec![
                            hypothesis("positive_native_regularizer", sigma_min > 0.),
                            upper("native_scale_lower", sigma_min, stats.scale[0]),
                            upper("native_scale_upper", stats.scale[0], sigma_max),
                        ],
                    ),
                ],
            )?;
            // Every distinct raw pair is followed through the unchanged native
            // Global normalizer and native logistic map, including replicated N.
            for i in (0..raw.len()).step_by(copies) {
                for j in (i + copies..raw.len()).step_by(copies) {
                    let rawgap = (raw[i] - raw[j]).abs();
                    if rawgap == 0. {
                        continue;
                    }
                    let raw_floor = 0.9 * rawgap;
                    let zgap = (z[i] - z[j]).abs();
                    let gap_out = (output[i] - output[j]).abs();
                    let kappa_z = raw_floor / sigma_max;
                    let rescaled = min_derivative * kappa_z;
                    let c = mean_value_point(z[i], z[j], gap_out / zgap)?;
                    let pair = json!({"vector":input,"a":i,"b":j,"raw_gap_floor":raw_floor,"z_gap_floor":kappa_z,"rescaled_gap_floor":rescaled,"mean_value_point":c,"derivative_at_mean_value_point":logistic_derivative(c)});
                    builder.inline(
                        "lem-variance-to-gap",
                        r"(v_i - v_j)^2 \le \Delta_{\max}^2",
                        pair.clone(),
                        le("each_realized_squared_pair_gap", rawgap * rawgap, gap * gap),
                    )?;
                    builder.inline(
                        label,
                        r"|v_a-v_b|\ge\kappa_{\mathrm{raw}}>0",
                        pair.clone(),
                        vec![
                            upper("raw_pair_floor", raw_floor, rawgap),
                            hypothesis("strict_positive_raw_pair_floor", raw_floor > 0.),
                        ],
                    )?;
                    builder.shown(
                        label,
                        r"\kappa_{\mathrm{rescaled}}(\kappa_{\mathrm{raw}}) :=",
                        pair.clone(),
                        hyps.clone(),
                        vec![positive_identity(
                            "family_rescaled_gap_floor_definition",
                            rescaled,
                            min_derivative / sigma_max * raw_floor,
                        )],
                    )?;
                    builder.shown(
                        label,
                        r"|z_a - z_b| =",
                        pair.clone(),
                        hyps.clone(),
                        vec![
                            relative(
                                "native_score_difference_common_mean",
                                zgap,
                                ((raw[i] - stats.mean[i]) / stats.scale[i]
                                    - (raw[j] - stats.mean[j]) / stats.scale[j])
                                    .abs(),
                                zgap,
                            ),
                            relative(
                                "native_score_difference_raw_shared_scale",
                                zgap,
                                rawgap / stats.scale[i],
                                zgap,
                            ),
                            hypothesis(
                                "actual_shared_statistics",
                                stats.mean[i] == stats.mean[j] && stats.scale[i] == stats.scale[j],
                            ),
                        ],
                    )?;
                    builder.shown(
                        label,
                        r"|z_a - z_b| \ge",
                        pair.clone(),
                        hyps.clone(),
                        vec![
                            upper("score_gap_uniform_floor", raw_floor / sigma_max, zgap),
                            positive_identity(
                                "score_gap_floor_definition",
                                kappa_z,
                                raw_floor / sigma_max,
                            ),
                        ],
                    )?;
                    builder.shown(
                        label,
                        r"|g_A(z_a) - g_A(z_b)| =",
                        pair.clone(),
                        hyps.clone(),
                        vec![
                            hypothesis(
                                "mean_value_point_inside_actual_segment",
                                c >= z[i].min(z[j]) && c <= z[i].max(z[j]),
                            ),
                            relative(
                                "native_logistic_mean_value_identity",
                                gap_out,
                                logistic_derivative(c) * zgap,
                                gap_out,
                            ),
                        ],
                    )?;
                    builder.shown(
                        label,
                        r"|g_A(z_a) - g_A(z_b)| \ge g'_{\min} \cdot \kappa_z",
                        pair.clone(),
                        hyps.clone(),
                        vec![
                            upper(
                                "derivative_lower_at_retained_mean_value_point",
                                min_derivative,
                                logistic_derivative(c),
                            ),
                            upper(
                                "native_positive_gap_uniform_lower",
                                min_derivative * kappa_z,
                                gap_out,
                            ),
                        ],
                    )?;
                    builder.shown(
                        label,
                        r"|g_A(z_a) - g_A(z_b)| \ge g'_{\min} \cdot \left(",
                        pair.clone(),
                        hyps.clone(),
                        vec![
                            upper(
                                "native_rescaled_gap_lower",
                                min_derivative * (raw_floor / sigma_max),
                                gap_out,
                            ),
                            positive_identity(
                                "conclusion_rescaled_floor_identity",
                                min_derivative * (raw_floor / sigma_max),
                                rescaled,
                            ),
                        ],
                    )?;
                    builder.shown(
                        label,
                        r"|g_A(z_a) - g_A(z_b)| \ge \kappa",
                        pair,
                        hyps.clone(),
                        vec![
                            upper("native_rescaled_pair_gap_lower", rescaled, gap_out),
                            hypothesis("strict_positive_native_pair_floor", rescaled > 0.),
                        ],
                    )?;
                }
            }
            rescale_variance_bounds(builder, &input, &raw, &z, &output, sigma_min, sigma_max)?;
            group_separation(builder, &input, &raw, kappa)?;
        }
    }
    Ok(())
}

fn group_separation(builder: &mut Builder, vector: &Value, raw: &[f64], kappa: f64) -> Result<()> {
    let label = "lem-variance-to-mean-separation";
    let k = raw.len() as f64;
    let groups = [raw[..raw.len() / 2].to_vec(), raw[raw.len() / 2..].to_vec()];
    let fh = groups[0].len() as f64 / k;
    let fl = groups[1].len() as f64 / k;
    let mh = mean(&groups[0]);
    let ml = mean(&groups[1]);
    let sh = variance(&groups[0]);
    let sl = variance(&groups[1]);
    let s2 = variance(raw);
    let within = fh * sh + fl * sl;
    let bound = (fh * range(&groups[0]).powi(2) + fl * range(&groups[1]).powi(2)) / 4.;
    let lower = ((kappa - bound) / (fh * fl)).sqrt();
    let input = json!({"realized_vector":vector,"H_indices":(0..raw.len()/2).collect::<Vec<_>>(),"L_indices":(raw.len()/2..raw.len()).collect::<Vec<_>>(),"a":raw.iter().copied().fold(f64::INFINITY,f64::min),"b":raw.iter().copied().fold(f64::NEG_INFINITY,f64::max),"fractions":[fh,fl],"means":[mh,ml],"within_variances":[sh,sl],"within_group_widths":[range(&groups[0]),range(&groups[1])],"within":within,"within_bound":bound,"kappa":kappa,"positive_mean_gap_floor":lower});
    let hyps = vec![
        hypothesis(
            "actual_nonempty_fixed_groups",
            groups.iter().all(|g| !g.is_empty()) && raw.len() >= 2,
        ),
        hypothesis(
            "valid_positive_separation_remainder",
            s2 >= kappa && kappa > bound,
        ),
        upper("actual_within_group_variance_ceiling", within, bound),
        hypothesis("actual_signed_orientation", ml >= mh),
    ];
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"v_1,\ldots,v_k\in[a,b]",
                condition(
                    "actual_realized_vector_support",
                    raw.iter().all(|&v| v >= raw[0] && v <= raw[raw.len() - 1]),
                ),
            ),
            (r"k\ge2", condition("at_least_two_group_values", k >= 2.)),
            (
                r"f_H=|H|/k",
                equal("high_group_fraction", fh, groups[0].len() as f64 / k),
            ),
            (
                r"f_L=|L|/k",
                equal("low_group_fraction", fl, groups[1].len() as f64 / k),
            ),
            (
                r"s^2\ge\kappa",
                ge("actual_total_variance_floor", s2, kappa),
            ),
            (
                r"s_{\rm within}^2\le B_{\rm within}",
                le("actual_within_variance_ceiling", within, bound),
            ),
            (
                r"\kappa>B_{\rm within}",
                strict("strict_group_separation_remainder", bound, kappa),
            ),
            (
                r"\mu_L\ge\mu_H",
                ge("actual_signed_group_orientation", ml, mh),
            ),
            (
                r"\bar v=f_H\mu_H+f_L\mu_L",
                equal(
                    "global_mean_group_decomposition",
                    mean(raw),
                    fh * mh + fl * ml,
                ),
            ),
            (r"f_H+f_L=1", equal("group_partition_mass", fh + fl, 1.)),
        ],
    )?;
    builder.shown(
        label,
        r"s^2=",
        input.clone(),
        hyps.clone(),
        vec![
            relative(
                "realized_centered_variance_definition",
                s2,
                raw.iter().map(|v| (v - mean(raw)).powi(2)).sum::<f64>() / k,
                s2,
            ),
            relative(
                "within_variance_definition",
                within,
                fh * sh + fl * sl,
                within,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"\boxed{s^2=",
        input.clone(),
        hyps.clone(),
        equal(
            "within_between_group_variance_identity",
            s2,
            within + fh * fl * (ml - mh).powi(2),
        ),
    )?;
    builder.shown(
        label,
        r"|\mu_L-\mu_H|\ge",
        input.clone(),
        hyps.clone(),
        vec![
            upper("actual_group_mean_separation_lower", lower, (ml - mh).abs()),
            hypothesis("positive_group_mean_floor", lower > 0.),
        ],
    )?;
    builder.shown(
        label,
        r"B_{\rm within}=",
        input.clone(),
        hyps.clone(),
        vec![
            relative(
                "weighted_group_width_variance_bound_definition",
                bound,
                (fh * range(&groups[0]).powi(2) + fl * range(&groups[1]).powi(2)) / 4.,
                bound,
            ),
            upper(
                "high_group_width_variance_ceiling",
                sh,
                range(&groups[0]).powi(2) / 4.,
            ),
            upper(
                "low_group_width_variance_ceiling",
                sl,
                range(&groups[1]).powi(2) / 4.,
            ),
        ],
    )?;
    builder.shown(
        label,
        r"s^2=f_Hs_H^2",
        input.clone(),
        hyps.clone(),
        equal(
            "expanded_group_variance_identity",
            s2,
            fh * sh + fl * sl + fh * (mh - mean(raw)).powi(2) + fl * (ml - mean(raw)).powi(2),
        ),
    )?;
    for &v in &groups[0] {
        builder.inline(
            label,
            r"v_i-\bar v=(v_i-\mu_H)+(\mu_H-\bar v)",
            json!({"groups":input,"v_i":v}),
            vec![relative(
                "high_group_centering_expansion",
                v - mean(raw),
                (v - mh) + (mh - mean(raw)),
                range(raw),
            )],
        )?;
    }
    Ok(())
}

fn logarithmic_comparisons(builder: &mut Builder) -> Result<()> {
    for (a, b) in [(0.1_f64, 2.1), (1e-8, 1e-6), (1., 1.0001)] {
        for fraction in [0_f64, 0.1, 0.4, 1.] {
            let kappa = fraction * (b - a);
            let c = (b.ln() - a.ln()) / (b - a);
            let chord = |u: f64| (b - u) / (b - a) * a.ln() + (u - a) / (b - a) * b.ln();
            let lower_u = (1. / c).clamp(a, (b - kappa).max(a));
            let upper_u = (1. / c - kappa).clamp(a, (b - kappa).max(a));
            let lower = chord(lower_u + kappa) - lower_u.ln();
            let upper_bound = (upper_u + kappa).ln() - chord(upper_u);
            let public = crate::convergence_selection::log_gap_bounds(a, b, kappa)?;
            let lower_p = ((lower_u + kappa - a) / (b - a)).clamp(0., 1.);
            let upper_p = ((upper_u - a) / (b - a)).clamp(0., 1.);
            let midpoint = (a + b - kappa) / 2.;
            let laws = [
                (
                    "lower_extremal",
                    vec![(1. - lower_p, a, lower_u), (lower_p, b, lower_u)],
                    false,
                ),
                (
                    "upper_extremal",
                    vec![
                        (1. - upper_p, upper_u + kappa, a),
                        (upper_p, upper_u + kappa, b),
                    ],
                    false,
                ),
                (
                    "ordered",
                    vec![(0.5, a + kappa, a), (0.5, midpoint + kappa, midpoint)],
                    true,
                ),
            ];
            for (name, law, ordered) in laws {
                // Generate admissible endpoint atoms even when subtraction of
                // b-a rounds the proposed endpoint by one floating-point unit.
                let law = law
                    .into_iter()
                    .map(|(p, x, y)| (p, x.clamp(a, b), y.clamp(a, b)))
                    .collect::<Vec<_>>();
                let ex = law.iter().map(|(p, x, _)| p * x).sum::<f64>();
                let ey = law.iter().map(|(p, _, y)| p * y).sum::<f64>();
                let elogx = law.iter().map(|(p, x, _)| p * x.ln()).sum::<f64>();
                let elogy = law.iter().map(|(p, _, y)| p * y.ln()).sum::<f64>();
                let gap = elogx - elogy;
                let coupling = law.iter().map(|(p, x, y)| p * (x - y).abs()).sum::<f64>();
                let input = json!({"a":a,"b":b,"kappa":kappa,"finite_joint_law":law,"law_name":name,"ordered_coupling":ordered,"chord_slope":c,"lower_minimizer":lower_u,"upper_maximizer":upper_u,"computed":{"E_X":ex,"E_Y":ey,"E_log_X":elogx,"E_log_Y":elogy,"E_absolute_difference":coupling,"L_ab":lower,"U_ab":upper_bound},"public_sharp_log_bounds":public});
                let hyps = vec![
                    hypothesis("strict_positive_log_support", a > 0. && a < b),
                    hypothesis(
                        "probability_law_within_declared_support",
                        law.iter()
                            .all(|&(p, x, y)| p >= 0. && x >= a && x <= b && y >= a && y <= b),
                    ),
                    relative(
                        "law_probability_mass",
                        law.iter().map(|t| t.0).sum(),
                        1.,
                        1.,
                    ),
                    upper("arithmetic_mean_gap_floor", kappa, ex - ey),
                    upper("absolute_mean_gap_ceiling", (ex - ey).abs(), kappa),
                    hypothesis(
                        "admissible_log_mean_gap_parameter",
                        kappa >= 0. && kappa <= b - a,
                    ),
                ];
                let l = "lem-log-gap-lower-bound";
                bind_inline(
                    builder,
                    l,
                    &input,
                    vec![
                        (
                            r"X,Y\in[a,b]",
                            condition(
                                "lower_law_support",
                                law.iter()
                                    .all(|&(_, x, y)| x >= a && x <= b && y >= a && y <= b),
                            ),
                        ),
                        (
                            r"0<a<b",
                            condition("lower_positive_support", a > 0. && a < b),
                        ),
                        (
                            r"\mathbb EX-\mathbb EY\geq\kappa",
                            ge("lower_actual_mean_gap", ex - ey, kappa),
                        ),
                        (
                            r"0\leq\kappa\leq b-a",
                            vec![
                                upper("nonnegative_kappa", 0., kappa),
                                upper("kappa_within_support_range", kappa, b - a),
                            ],
                        ),
                        (
                            r"\mathbb E\log Y\leq\log\mathbb EY",
                            le("actual_Jensen_lower_comparison", elogy, ey.ln()),
                        ),
                    ],
                )?;
                builder.shown(
                    l,
                    r"\ell(u)=",
                    input.clone(),
                    hyps.clone(),
                    vec![
                        relative(
                            "lower_chord_intercept_at_a",
                            chord(a),
                            a.ln(),
                            a.ln().abs().max(1.),
                        ),
                        relative(
                            "lower_chord_intercept_at_b",
                            chord(b),
                            b.ln(),
                            b.ln().abs().max(1.),
                        ),
                        positive_identity("positive_chord_slope", c, (b.ln() - a.ln()) / (b - a)),
                    ],
                )?;
                let mut optimality = vec![
                    upper("actual_logarithmic_lower_gap", lower, gap),
                    relative(
                        "public_lower_projection_value",
                        public.lower,
                        lower,
                        lower.abs().max(c * (b - a)).max(1e-30),
                    ),
                    relative("public_lower_minimizer", public.lower_minimizer, lower_u, b),
                ];
                for j in 0..=256 {
                    let u = a + (b - kappa - a) * j as f64 / 256.;
                    optimality.push(upper(
                        "sampled_chord_minimizer_certificate",
                        lower,
                        chord(u + kappa) - u.ln(),
                    ));
                }
                builder.shown(
                    l,
                    r"L_{a,b}(\kappa):=",
                    input.clone(),
                    hyps.clone(),
                    optimality,
                )?;
                let l = "lem-log-gap-upper-bound";
                bind_inline(
                    builder,
                    l,
                    &input,
                    vec![
                        (
                            r"X,Y\in[a,b]",
                            condition(
                                "upper_law_support",
                                law.iter()
                                    .all(|&(_, x, y)| x >= a && x <= b && y >= a && y <= b),
                            ),
                        ),
                        (
                            r"0<a<b",
                            condition("upper_positive_support", a > 0. && a < b),
                        ),
                        (
                            r"|\mathbb EX-\mathbb EY|\leq\kappa\leq b-a",
                            vec![
                                upper("upper_actual_absolute_mean_gap", (ex - ey).abs(), kappa),
                                upper("upper_kappa_support_range", kappa, b - a),
                            ],
                        ),
                        (
                            r"\mathbb E|X-Y|\leq K",
                            le("actual_coupled_absolute_difference", coupling, coupling),
                        ),
                        (
                            r"\mathbb E\log X-\mathbb E\log Y\leq\log\mathbb EX-\ell(\mathbb EY)",
                            le("actual_Jensen_chord_upper", gap, ex.ln() - chord(ey)),
                        ),
                        (
                            r"\mathbb EY=u",
                            equal(
                                "actual_upper_comparison_mean_argument",
                                ey,
                                law.iter().map(|(p, _, y)| p * y).sum(),
                            ),
                        ),
                    ],
                )?;
                let mut optimality = vec![
                    upper(
                        "actual_absolute_logarithmic_upper_gap",
                        gap.abs(),
                        upper_bound,
                    ),
                    relative(
                        "public_upper_projection_value",
                        public.upper,
                        upper_bound,
                        upper_bound.abs().max(c * (b - a)),
                    ),
                    relative("public_upper_maximizer", public.upper_maximizer, upper_u, b),
                ];
                for j in 0..=256 {
                    let u = a + (b - kappa - a) * j as f64 / 256.;
                    optimality.push(upper(
                        "sampled_chord_maximizer_certificate",
                        (u + kappa).ln() - chord(u),
                        upper_bound,
                    ));
                }
                builder.shown(
                    l,
                    r"\leq U_{a,b}(\kappa):=",
                    input.clone(),
                    hyps.clone(),
                    optimality,
                )?;
                builder.shown(
                    l,
                    r"|\mathbb E\log X-\mathbb E\log Y|\leq\log(1+K/a)",
                    input.clone(),
                    hyps.clone(),
                    le(
                        "actual_coupled_Jensen_log_refinement",
                        gap.abs(),
                        (coupling / a).ln_1p(),
                    ),
                )?;
                for &(p, x, y) in &law {
                    if p == 0. {
                        continue;
                    }
                    let pair = json!({"law":input,"atom_probability":p,"X":x,"Y":y});
                    builder.inline(
                        "lem-log-gap-lower-bound",
                        r"\log x\geq\ell(x)",
                        pair.clone(),
                        ge("pointwise_log_chord", x.ln(), chord(x)),
                    )?;
                    builder.inline(
                        "lem-log-gap-upper-bound",
                        r"|\log X-\log Y|\leq\log(1+|X-Y|/a)",
                        pair.clone(),
                        le(
                            "pointwise_absolute_log_difference",
                            (x.ln() - y.ln()).abs(),
                            ((x - y).abs() / a).ln_1p(),
                        ),
                    )?;
                    if ordered {
                        builder.inline(
                            "lem-log-gap-lower-bound",
                            r"X\geq Y",
                            pair.clone(),
                            ge("actual_ordered_joint_atom", x, y),
                        )?;
                        builder.inline(
                            "lem-log-gap-lower-bound",
                            r"\log X-\log Y=\int_Y^X t^{-1}\,dt\geq(X-Y)/b",
                            pair,
                            vec![
                                relative(
                                    "reciprocal_integral_exact_primitive",
                                    x.ln() - y.ln(),
                                    (x / y).ln(),
                                    (x.ln() - y.ln()).abs().max(1e-30),
                                ),
                                upper(
                                    "ordered_reciprocal_integral_lower",
                                    (x - y) / b,
                                    x.ln() - y.ln(),
                                ),
                            ],
                        )?;
                    }
                }
                if ordered {
                    builder.shown(
                        "lem-log-gap-lower-bound",
                        r"\mathbb E\log X-\mathbb E\log Y\geq\kappa/b",
                        input.clone(),
                        hyps.clone(),
                        vec![
                            upper("ordered_expected_log_lower", kappa / b, gap),
                            upper("ordered_log1p_lower", (kappa / b).ln_1p(), kappa / b),
                        ],
                    )?;
                    builder.inline(
                        "lem-log-gap-lower-bound",
                        r"\log(1+z)\leq z",
                        json!({"law":input,"z":kappa/b}),
                        le("nonnegative_log1p_bound", (kappa / b).ln_1p(), kappa / b),
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn conditional_selection(builder: &mut Builder) -> Result<()> {
    let sigma_min = 0.1_f64;
    let base = [-0.8_f64, -0.2, 0.1, 0.7];
    let fixed_map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.1,
    };
    let base_mu = mean(&base);
    let base_sigma = (variance(&base) + sigma_min * sigma_min).sqrt();
    let base_fitness = base
        .iter()
        .map(|v| fixed_map.map((v - base_mu) / base_sigma))
        .collect::<Result<Vec<_>>>()?;
    let sstar = 0.8 * variance(&base_fitness);
    let rstar = 2_f64;
    let vmax = 2.1_f64;
    let width = 2_f64;
    let astar = (-range(&base).powi(2) / (2. * width * width)).exp();
    for copies in [1_usize, 2, 8, 32] {
        let raw = base
            .iter()
            .flat_map(|&v| std::iter::repeat_n(v, copies))
            .collect::<Vec<_>>();
        let k = raw.len();
        let kf = k as f64;
        let obs = ObservationBatch::positions(TensorBatch::vectors(k, 1, raw.clone())?);
        let distance = Distance::default();
        let kernel = Kernel::Gaussian { width };
        let (z, stats) = Standardizer::Global { sigma_min }.apply(&raw, &vec![true; k], &obs)?;
        let fitness = z
            .iter()
            .map(|&z| fixed_map.map(z))
            .collect::<Result<Vec<_>>>()?;
        let mut weights = vec![vec![0.; k]; k];
        let mut law = weights.clone();
        for (i, row) in weights.iter_mut().enumerate() {
            for (j, weight) in row.iter_mut().enumerate() {
                if i != j {
                    *weight = kernel
                        .log_weight(
                            distance.compare(&obs, i, &obs, j)?,
                            ComparisonKind::Distance,
                        )?
                        .exp();
                }
            }
            let normalizer = row.iter().sum::<f64>();
            for (probability, &weight) in law[i].iter_mut().zip(row.iter()) {
                *probability = weight / normalizer;
            }
        }
        let a = (kf - 1.)
            * law
                .iter()
                .enumerate()
                .flat_map(|(i, row)| {
                    row.iter()
                        .enumerate()
                        .filter(move |(j, _)| *j != i)
                        .map(|(_, p)| *p)
                })
                .fold(f64::INFINITY, f64::min);
        let mu = mean(&fitness);
        let s2 = variance(&fitness);
        let rv = range(&fitness);
        let fh = 0.5_f64;
        let fl = 0.5_f64;
        let mh = mean(&fitness[..k / 2]);
        let ml = mean(&fitness[k / 2..]);
        let sh = variance(&fitness[..k / 2]);
        let sl = variance(&fitness[k / 2..]);
        let positive_mean = fitness.iter().map(|v| (v - mu).max(0.)).sum::<f64>() / kf;
        let absolute_mean = fitness.iter().map(|v| (v - mu).abs()).sum::<f64>() / kf;
        for pmax in [0.05_f64, 1., 2.] {
            let decision = CloneDecision {
                epsilon: 1e-6,
                saturation: pmax,
                every: 1,
                ..Default::default()
            };
            decision.validate()?;
            let eps = decision.epsilon;
            let av = rv.max(pmax * (vmax + eps));
            let pu = astar * sstar / (2. * rstar * rstar.max(pmax * (vmax + eps)));
            let pressure = (0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| {
                            law[i][j] * decision.acceptance_probability(0, fitness[i], fitness[j])
                        })
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let signals = (0..k)
                .map(|i| {
                    (0..k)
                        .map(|j| law[i][j] * (fitness[j] - fitness[i]).max(0.))
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let input = json!({"raw":raw,"native_statistics":stats,"native_scores":z,"native_positive_map":fixed_map,"retained_fitness":fitness,"native_distance":distance,"native_kernel":kernel,"actual_nonself_companion_law":law,"native_clone_decision":decision,"active_cloning_step":0,"copies_of_each_base_value":copies,"fitness_support_ceiling":vmax,"positive_companion_lower_coefficient":a,"family_bounds":{"a_star":astar,"s_star_squared":sstar,"R_star":rstar},"computed":{"fitness_mean":mu,"fitness_variance":s2,"fitness_range":rv,"A_V":av,"p_u":pu,"positive_part_signals":signals,"actual_clipped_probabilities":pressure}});
            let hyps = vec![
                hypothesis(
                    "retained_strict_positive_fitness_and_range",
                    fitness.iter().all(|&v| v > 0. && v <= vmax) && rv > 0. && rv <= rstar,
                ),
                hypothesis(
                    "same_nonself_normalized_native_kernel",
                    law.iter().enumerate().all(|(i, row)| {
                        row[i] == 0. && row.iter().all(|p| p.is_finite() && *p >= 0.)
                    }),
                ),
                hypothesis(
                    "positive_common_selection_family_constants",
                    a >= astar && astar > 0. && s2 >= sstar && sstar > 0.,
                ),
                hypothesis(
                    "independent_active_uniform_threshold_configuration",
                    decision.every == 1 && decision.saturation > 0. && eps >= 0.,
                ),
            ];
            let label = "lem-mean-companion-fitness-gap";
            let mut fitness_checks = vec![
                upper(
                    "positive_support_maximum",
                    fitness.iter().copied().fold(0., f64::max),
                    vmax,
                ),
                hypothesis(
                    "positive_support_minimum",
                    fitness.iter().copied().fold(f64::INFINITY, f64::min) > 0.,
                ),
            ];
            fitness_checks.extend(law.iter().enumerate().map(|(i, row)| {
                relative(
                    &format!("actual_companion_probability_mass_{i}"),
                    row.iter().sum(),
                    1.,
                    1.,
                )
            }));
            bind_inline(
                builder,
                label,
                &input,
                vec![
                    (r"0<V_i\leq V_{\mathrm{pot,max}}", fitness_checks),
                    (
                        r"R_V=\max V_i-\min V_i>0",
                        vec![positive_identity(
                            "actual_realized_fitness_range",
                            rv,
                            fitness.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                                - fitness.iter().copied().fold(f64::INFINITY, f64::min),
                        )],
                    ),
                    (
                        r"a>0",
                        condition("strict_positive_native_companion_floor", a > 0.),
                    ),
                    (
                        r"s_V^2\geq f_Hf_L\Delta^2",
                        ge(
                            "native_realized_group_mean_signal",
                            s2,
                            fh * fl * (ml - mh).powi(2),
                        ),
                    ),
                    (
                        r"s_V^2=f_Hs_H^2+f_Ls_L^2+f_Hf_L(\mu_H-\mu_L)^2",
                        equal(
                            "native_fitness_group_variance_decomposition",
                            s2,
                            fh * sh + fl * sl + fh * fl * (mh - ml).powi(2),
                        ),
                    ),
                ],
            )?;
            builder.shown(
                label,
                r"s_V^2\leq R_V",
                input.clone(),
                hyps.clone(),
                vec![
                    upper(
                        "variance_by_range_absolute_deviation",
                        s2,
                        rv * absolute_mean,
                    ),
                    relative(
                        "absolute_deviation_positive_part_identity",
                        rv * absolute_mean,
                        2. * rv * positive_mean,
                        rv * absolute_mean,
                    ),
                ],
            )?;
            let label_pressure = "lem-unfit-cloning-pressure";
            bind_inline(
                builder,
                label_pressure,
                &input,
                vec![
                    (
                        r"a\geq a_*>0",
                        vec![
                            upper("actual_native_kernel_common_floor", astar, a),
                            hypothesis("strict_positive_common_kernel_floor", astar > 0.),
                        ],
                    ),
                    (
                        r"s_V^2\geq s_*^2>0",
                        vec![
                            upper("actual_fitness_common_variance_floor", sstar, s2),
                            hypothesis("strict_positive_common_fitness_variance_floor", sstar > 0.),
                        ],
                    ),
                    (
                        r"R_V\leq R_*<\infty",
                        vec![
                            upper("actual_fitness_common_range_ceiling", rv, rstar),
                            hypothesis("finite_common_range_ceiling", rstar.is_finite()),
                        ],
                    ),
                ],
            )?;
            builder.shown(
                label_pressure,
                r"A_V:=",
                input.clone(),
                hyps.clone(),
                vec![positive_identity(
                    "range_clipping_denominator_definition",
                    av,
                    rv.max(pmax * (vmax + eps)),
                )],
            )?;
            builder.shown(
                label_pressure,
                r"p_u=",
                input.clone(),
                hyps.clone(),
                vec![positive_identity(
                    "common_N_uniform_pressure_definition",
                    pu,
                    astar * sstar / (2. * rstar * rstar.max(pmax * (vmax + eps))),
                )],
            )?;
            // Distinct base-value representatives exhaust all row-value cases;
            // replicated rows differ only by permutation of their self-excluded
            // companion slots. All native row sums are nevertheless retained.
            for i in (0..k).step_by(copies) {
                let own = fitness[i];
                let bi = pmax * (own + eps);
                let uniform_mean =
                    (0..k).filter(|&j| j != i).map(|j| fitness[j]).sum::<f64>() / (kf - 1.);
                let row = json!({"selection":input,"row":i,"own_fitness":own,"uniform_nonself_companion_mean":uniform_mean,"b_i":bi,"positive_signal":signals[i],"actual_pressure":pressure[i]});
                builder.shown(
                    label,
                    r"\mu_{\mathrm{comp},i}-V_i=",
                    row.clone(),
                    hyps.clone(),
                    equal(
                        "uniform_nonself_companion_mean_identity",
                        uniform_mean - own,
                        kf / (kf - 1.) * (mu - own),
                    ),
                )?;
                builder.inline(
                    label,
                    r"|X-\mu|\leq R_V",
                    row.clone(),
                    le(
                        "realized_fitness_centered_range_bound",
                        (own - mu).abs(),
                        rv,
                    ),
                )?;
                builder.shown(
                    label_pressure,
                    r"p_i=",
                    row.clone(),
                    hyps.clone(),
                    vec![
                        relative(
                            "actual_native_clipped_acceptance_expectation",
                            pressure[i],
                            (0..k)
                                .map(|j| law[i][j] * ((fitness[j] - own).max(0.) / bi).min(1.))
                                .sum::<f64>(),
                            pressure[i].max(1e-30),
                        ),
                        upper(
                            "actual_clipped_pressure_signal_lower",
                            signals[i] / av,
                            pressure[i],
                        ),
                    ],
                )?;
                if own <= mu {
                    builder.inline(
                        label,
                        r"V_i\leq\mu",
                        row.clone(),
                        le("realized_unfit_branch", own, mu),
                    )?;
                    builder.inline(
                        label_pressure,
                        r"V_i\leq\mu",
                        row.clone(),
                        le("realized_pressure_unfit_branch", own, mu),
                    )?;
                    builder.shown(
                        label,
                        r"\mathbb E_{K_i}(V_c-V_i)_+",
                        row.clone(),
                        hyps.clone(),
                        vec![
                            upper(
                                "unfit_positive_signal_finite_k_lower",
                                a * kf / (kf - 1.) * s2 / (2. * rv),
                                signals[i],
                            ),
                            upper(
                                "finite_k_factor_at_least_one",
                                a * s2 / (2. * rv),
                                a * kf / (kf - 1.) * s2 / (2. * rv),
                            ),
                        ],
                    )?;
                    builder.inline(
                        label_pressure,
                        r"p_i\geq a s_V^2/(2R_VA_V)",
                        row.clone(),
                        ge(
                            "actual_unfit_clipped_pressure_lower",
                            pressure[i],
                            a * s2 / (2. * rv * av),
                        ),
                    )?;
                    // This evaluates the N-uniform constant against each native
                    // pressure, beyond merely computing its defining identity.
                    builder.shown(
                        label_pressure,
                        r"p_u=",
                        row.clone(),
                        hyps.clone(),
                        vec![
                            positive_identity(
                                "positive_family_pressure_identity",
                                pu,
                                astar * sstar / (2. * rstar * rstar.max(pmax * (vmax + eps))),
                            ),
                            upper("actual_native_N_uniform_unfit_pressure", pu, pressure[i]),
                        ],
                    )?;
                    builder.inline(
                        label,
                        r"j=i",
                        row.clone(),
                        equal(
                            "excluded_self_positive_part_is_zero",
                            (own - mu).max(0.),
                            0.,
                        ),
                    )?;
                }
                let favorable_gap = rv / 4.;
                let favorable = (0..k)
                    .filter(|&j| fitness[j] - own >= favorable_gap)
                    .collect::<Vec<_>>();
                let favorable_mass = favorable.iter().map(|&j| law[i][j]).sum::<f64>();
                if favorable_mass > 0. {
                    builder.inline(
                        label_pressure,
                        r"q_*>0",
                        row.clone(),
                        condition(
                            "actual_positive_favorable_companion_mass",
                            favorable_mass > 0.,
                        ),
                    )?;
                    builder.inline(label_pressure,"p_i\\geq q_*\\min\\{1,\\Delta/[p_{\\max}(V_{\\mathrm{pot,max}}+\n\\varepsilon_{\\mathrm{clone}})]\\}",json!({"row":row,"favorable_indices":favorable,"Delta":favorable_gap,"q_star":favorable_mass}),ge("actual_favorable_set_pressure",pressure[i],favorable_mass*(favorable_gap/(pmax*(vmax+eps))).min(1.)))?;
                }
                for j in (0..k).step_by(copies) {
                    let donor = fitness[j];
                    let signal = (donor - own).max(0.);
                    let pair = json!({"row":row,"donor_row":j,"donor_fitness":donor,"z":signal,"native_clipped_probability":decision.acceptance_probability(0,own,donor)});
                    bind_inline(
                        builder,
                        label_pressure,
                        &pair,
                        vec![
                            (
                                r"0\leq z\leq R_V",
                                vec![
                                    upper("nonnegative_positive_part", 0., signal),
                                    upper("positive_part_below_range", signal, rv),
                                ],
                            ),
                            (
                                r"b_i=p_{\max}(V_i+\varepsilon_{\mathrm{clone}})",
                                equal("recipient_clipping_denominator", bi, pmax * (own + eps)),
                            ),
                            (
                                r"\min(1,z/b_i)\geq z/A_V",
                                ge(
                                    "pointwise_actual_native_clipping_bound",
                                    decision.acceptance_probability(0, own, donor),
                                    signal / av,
                                ),
                            ),
                            (
                                r"z=(V_c-V_i)_+",
                                equal(
                                    "actual_donor_positive_part_signal",
                                    signal,
                                    (donor - own).max(0.),
                                ),
                            ),
                        ],
                    )?;
                    if signal <= bi {
                        builder.inline(
                            label_pressure,
                            r"z\leq b_i",
                            pair.clone(),
                            le("actual_unclipped_branch", signal, bi),
                        )?;
                        builder.inline(
                            label_pressure,
                            r"A_V\geq b_i",
                            pair.clone(),
                            ge("unclipped_denominator_ceiling", av, bi),
                        )?;
                    } else {
                        builder.inline(
                            label_pressure,
                            r"z>b_i",
                            pair.clone(),
                            strict("actual_saturated_branch", bi, signal),
                        )?;
                        builder.inline(
                            label_pressure,
                            r"A_V\geq R_V\geq z",
                            pair.clone(),
                            vec![
                                upper("saturated_range_ceiling", rv, av),
                                upper("saturated_signal_range", signal, rv),
                            ],
                        )?;
                    }
                    if i != j {
                        builder.inline(
                            label,
                            r"K_i(j)\geq a/(k-1)",
                            pair.clone(),
                            ge(
                                "actual_native_nonself_kernel_floor",
                                law[i][j],
                                a / (kf - 1.),
                            ),
                        )?;
                    }
                    if own <= mu {
                        builder.inline(
                            label,
                            r"(V_j-V_i)_+\geq(V_j-\mu)_+",
                            pair.clone(),
                            ge(
                                "unfit_donor_positive_part_monotonicity",
                                signal,
                                (donor - mu).max(0.),
                            ),
                        )?;
                    }
                    if donor - own >= favorable_gap {
                        builder.inline(
                            label_pressure,
                            r"V_j-V_i\geq\Delta>0",
                            pair,
                            vec![
                                upper("actual_favorable_donor_gap", favorable_gap, donor - own),
                                hypothesis("strict_positive_favorable_gap", favorable_gap > 0.),
                            ],
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn positional_variance_lower_bounds(builder: &mut Builder) -> Result<()> {
    let label = "lem-var-x-implies-var-h";
    for k in [1_usize, 2, 4, 16] {
        for lv in [0.1_f64, 1., 8.] {
            for velocity_factor in [0_f64, 0.2] {
                let x = (0..k)
                    .map(|i| 2. * i as f64 / k as f64 - 1.)
                    .collect::<Vec<_>>();
                let v = x.iter().map(|&x| velocity_factor * x).collect::<Vec<_>>();
                let vx = variance(&x);
                let vv = variance(&v);
                let vh = vx + lv * vv;
                let threshold = vx / 2.;
                let input = json!({"positions":x,"velocities":v,"lambda_v":lv,"position_variance":vx,"velocity_variance":vv,"hypocoercive_variance":vh,"positional_variance_threshold_squared":threshold,"case_scope":if threshold>0. {"positive threshold implication"}else{"singleton base inequality only"}});
                let hyps = vec![
                    hypothesis(
                        "finite_nonempty_entering_phase_coordinates",
                        k > 0 && x.iter().chain(&v).all(|v| v.is_finite()),
                    ),
                    hypothesis("positive_hypocoercive_velocity_weight", lv > 0.),
                ];
                bind_inline(
                    builder,
                    label,
                    &input,
                    vec![
                        (
                            r"\lambda_v > 0",
                            condition("actual_positive_hypocoercive_weight", lv > 0.),
                        ),
                        (
                            r"\mathrm{Var}_v(S_k) \ge 0",
                            ge("actual_nonnegative_velocity_variance", vv, 0.),
                        ),
                    ],
                )?;
                builder.shown(
                    label,
                    r"\mathrm{Var}_h(S_k) \ge",
                    input.clone(),
                    hyps.clone(),
                    ge("position_variance_is_phase_variance_lower_bound", vh, vx),
                )?;
                builder.shown(
                    label,
                    r"\mathrm{Var}_h(S_k) :=",
                    input.clone(),
                    hyps.clone(),
                    equal("actual_hypocoercive_variance_definition", vh, vx + lv * vv),
                )?;
                builder.shown(
                    label,
                    r"\mathrm{Var}_h(S_k) =",
                    input.clone(),
                    hyps.clone(),
                    vec![
                        relative("actual_phase_variance_sum_identity", vh, vx + lv * vv, vh),
                        upper("nonnegative_velocity_term_added", vx, vh),
                    ],
                )?;
                if threshold > 0. {
                    bind_inline(
                        builder,
                        label,
                        &input,
                        vec![
                            (
                                r"\mathrm{Var}_x(S_k) > R^2_{\text{var}}",
                                strict("actual_strict_position_variance_threshold", threshold, vx),
                            ),
                            (
                                r"R^2_{\text{var}} > 0",
                                condition("actual_positive_position_threshold", threshold > 0.),
                            ),
                            (
                                r"\mathrm{Var}_h(S_k) > R^2_{\text{var}}",
                                strict(
                                    "phase_variance_strictly_above_position_threshold",
                                    threshold,
                                    vh,
                                ),
                            ),
                            (
                                r"\mathrm{Var}_h(S_k) \ge \mathrm{Var}_x(S_k) > R^2_{\text{var}}",
                                vec![
                                    upper("phase_vs_position_variance_chain", vx, vh),
                                    hypothesis("strict_position_threshold_chain", vx > threshold),
                                ],
                            ),
                        ],
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rescale_variance_bounds(
    builder: &mut Builder,
    vector: &Value,
    raw: &[f64],
    z: &[f64],
    output: &[f64],
    smin: f64,
    smax: f64,
) -> Result<()> {
    let label = "prop-fixed-rescale-variance-bound";
    let a = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let b = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let k = raw.len() as f64;
    let mu = mean(raw);
    let scale = (variance(raw) + smin * smin).sqrt();
    let radius = (b - a) / smin;
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 0.,
    };
    let offset = 1e-6_f64;
    let minimum = logistic_derivative(radius);
    let maximum = logistic_derivative(0.);
    let low = map.map(-radius)? + offset;
    let high = map.map(radius)? + offset;
    let width = high - low;
    let vraw = variance(raw);
    let vout = variance(output);
    let lower = minimum.powi(2) / smax.powi(2) * vraw;
    let upper_linear = maximum.powi(2) / smin.powi(2) * vraw;
    let ceiling = width.powi(2) / 4.;
    let output_mean = mean(output);
    let input = json!({"native_realized_vector":vector,"raw_support":[a,b],"native_output_support":[low,high],"native_shared_scale":scale,"s_min":smin,"s_max":smax,"score_radius_Z":radius,"constant_additive_offset_eta":offset,"m_g":minimum,"M_g":maximum,"R_g":width,"computed":{"Var_raw":vraw,"Var_output":vout,"lower_bound":lower,"linear_upper_bound":upper_linear,"range_upper_bound":ceiling}});
    let hyps = vec![
        hypothesis(
            "actual_common_scale_and_finite_raw_interval",
            raw.len() >= 2
                && a <= b
                && smin > 0.
                && scale >= smin
                && scale <= smax
                && smax.is_finite(),
        ),
        hypothesis(
            "native_output_same_fixed_logistic_and_offset",
            raw.len() == z.len() && z.len() == output.len(),
        ),
        hypothesis(
            "representable_positive_operational_logistic_derivative",
            minimum.is_finite() && minimum > 0.,
        ),
    ];
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"y=(y_1,\ldots,y_k)\in[a,b]^k",
                condition(
                    "actual_complete_realized_interval_support",
                    raw.iter().all(|&v| v >= a && v <= b),
                ),
            ),
            (r"k\ge2", condition("actual_rescale_vector_size", k >= 2.)),
            (
                r"0<s_{\min}\le s(y)\le s_{\max}<\infty",
                vec![
                    hypothesis("strict_positive_scale_floor", smin > 0.),
                    upper("actual_scale_lower", smin, scale),
                    upper("actual_scale_upper", scale, smax),
                    hypothesis("finite_scale_ceiling", smax.is_finite()),
                ],
            ),
            (
                r"Z=(b-a)/s_{\min}",
                equal(
                    "operational_score_radius_definition",
                    radius,
                    (b - a) / smin,
                ),
            ),
            (
                r"m_g=2e^{-Z}/(1+e^{-Z})^2>0",
                vec![positive_identity(
                    "positive_operational_derivative_minimum",
                    minimum,
                    2. * (-radius).exp() / (1. + (-radius).exp()).powi(2),
                )],
            ),
            (
                r"M_g\le1/2",
                le("canonical_derivative_maximum", maximum, 0.5),
            ),
            (
                r"\operatorname{Var}(u)=(2k^2)^{-1}\sum_{i,j}(u_i-u_j)^2",
                vec![
                    relative(
                        "raw_ordered_pair_variance_identity",
                        vraw,
                        raw.iter()
                            .flat_map(|&a| raw.iter().map(move |&b| (a - b).powi(2)))
                            .sum::<f64>()
                            / (2. * k * k),
                        vraw,
                    ),
                    relative(
                        "output_ordered_pair_variance_identity",
                        vout,
                        output
                            .iter()
                            .flat_map(|&a| output.iter().map(move |&b| (a - b).powi(2)))
                            .sum::<f64>()
                            / (2. * k * k),
                        vout,
                    ),
                ],
            ),
            (
                r"\operatorname{Var}(X)\le(d-\mathbb EX)(\mathbb EX-c)\le(d-c)^2/4",
                vec![
                    upper(
                        "empirical_interval_variance_by_mean",
                        vout,
                        (high - output_mean) * (output_mean - low),
                    ),
                    upper(
                        "interval_mean_product_by_width",
                        (high - output_mean) * (output_mean - low),
                        width * width / 4.,
                    ),
                ],
            ),
        ],
    )?;
    let mut extrema = vec![
        positive_identity(
            "derivative_minimum_endpoint",
            minimum,
            logistic_derivative(-radius),
        ),
        positive_identity(
            "derivative_maximum_center",
            maximum,
            logistic_derivative(0.),
        ),
        positive_identity(
            "native_output_range_definition",
            width,
            map.map(radius)? - map.map(-radius)?,
        ),
    ];
    for j in 0..=64 {
        let score = -radius + 2. * radius * j as f64 / 64.;
        extrema.extend([
            upper(
                "derivative_minimum_on_operational_probes",
                minimum,
                logistic_derivative(score),
            ),
            upper(
                "derivative_maximum_on_operational_probes",
                logistic_derivative(score),
                maximum,
            ),
        ]);
    }
    builder.shown(label, r"m_g=", input.clone(), hyps.clone(), extrema)?;
    builder.shown(
        label,
        r"\boxed{",
        input.clone(),
        hyps.clone(),
        vec![
            upper("native_rescale_variance_lower", lower, vout),
            upper("native_rescale_variance_linear_upper", vout, upper_linear),
            upper("native_rescale_variance_range_upper", vout, ceiling),
        ],
    )?;
    for i in 0..raw.len() {
        let point = json!({"rescale":input,"row":i,"raw_value":raw[i],"native_score":z[i],"native_output":output[i]});
        builder.inline(
            label,
            r"z_i=(y_i-\bar y)/s(y)",
            point.clone(),
            vec![relative(
                "native_standardized_score_definition",
                z[i],
                (raw[i] - mu) / scale,
                radius,
            )],
        )?;
        builder.inline(
            label,
            r"d'_i=g(z_i)+\eta",
            point.clone(),
            vec![positive_identity(
                "native_logistic_output_and_fixed_offset",
                output[i],
                map.map(z[i])? + offset,
            )],
        )?;
        builder.inline(
            label,
            r"g(z)=2/(1+e^{-z})",
            point.clone(),
            vec![positive_identity(
                "configured_logistic_without_offset",
                map.map(z[i])?,
                2. / (1. + (-z[i]).exp()),
            )],
        )?;
        builder.inline(
            label,
            r"|z_i|\le Z",
            point.clone(),
            le("attained_score_within_family_interval", z[i].abs(), radius),
        )?;
        builder.inline(
            label,
            r"(X-c)(d-X)\ge0",
            point.clone(),
            ge(
                "empirical_interval_nonnegative_endpoint_product",
                (output[i] - low) * (high - output[i]),
                0.,
            ),
        )?;
        builder.inline(
            label,
            r"g'(z)=2e^{-z}/(1+e^{-z})^2=(2\cosh^2(z/2))^{-1}",
            point,
            vec![
                positive_identity(
                    "native_logistic_derivative_exponential_identity",
                    logistic_derivative(z[i]),
                    2. * (-z[i]).exp() / (1. + (-z[i]).exp()).powi(2),
                ),
                positive_identity(
                    "native_logistic_derivative_hyperbolic_identity",
                    logistic_derivative(z[i]),
                    1. / (2. * (z[i] / 2.).cosh().powi(2)),
                ),
            ],
        )?;
    }
    // Unique values exhaust the pair terms in the replicated native vectors.
    let mut indices = Vec::new();
    for i in 0..raw.len() {
        if !indices.iter().any(|&j| raw[j] == raw[i]) {
            indices.push(i);
        }
    }
    for &i in &indices {
        for &j in &indices {
            let pair = json!({"rescale":input,"i":i,"j":j});
            let gap = (raw[i] - raw[j]).abs();
            let outgap = (output[i] - output[j]).abs();
            builder.inline(
                label,
                r"z_i-z_j=(y_i-y_j)/s(y)",
                pair.clone(),
                vec![relative(
                    "actual_shared_center_and_scale_difference",
                    z[i] - z[j],
                    (raw[i] - raw[j]) / scale,
                    radius,
                )],
            )?;
            builder.shown(
                label,
                r"\frac{m_g}{s_{\max}}",
                pair,
                hyps.clone(),
                vec![
                    upper("pairwise_rescale_gap_lower", minimum / smax * gap, outgap),
                    upper("pairwise_rescale_gap_upper", outgap, maximum / smin * gap),
                ],
            )?;
        }
    }
    let vmax = a.abs().max(b.abs());
    let patchmax = (vmax * vmax + smin * smin).sqrt();
    let endpoint_values = [-vmax, vmax];
    let endpoint_obs =
        ObservationBatch::positions(TensorBatch::vectors(2, 1, endpoint_values.to_vec())?);
    let (endpoint_scores, endpoint_stats) = Standardizer::Global { sigma_min: smin }.apply(
        &endpoint_values,
        &[true; 2],
        &endpoint_obs,
    )?;
    let extremum_input = json!({"realized_rescale_input":input,"attaining_raw_interval_endpoint_measure":endpoint_values,"native_endpoint_scores":endpoint_scores,"native_endpoint_statistics":endpoint_stats,"computed_endpoint_variance":variance(&endpoint_values),"scope":"The regularized-scale supremum on the stated raw-value interval, attained by equal endpoint mass; this is not an assertion that a particular physical reward image contains both endpoints."});
    let derivative_support = 2. * vmax / smin;
    let family_minimum = logistic_derivative(derivative_support);
    builder.shown(
        "def-max-patched-std",
        r"\sigma'_{\max} :=",
        extremum_input,
        hyps.clone(),
        vec![
            positive_identity(
                "native_monotone_patch_maximum",
                patchmax,
                endpoint_stats.scale[0],
            ),
            positive_identity(
                "endpoint_empirical_variance_attains_allowed_maximum",
                variance(&endpoint_values),
                vmax * vmax,
            ),
            upper(
                "actual_native_scale_below_family_patch_supremum",
                scale,
                patchmax,
            ),
        ],
    )?;
    builder.inline(
        "def-max-patched-std",
        r"V_{\max}<\infty",
        input.clone(),
        condition("finite_complete_raw_family_bound", vmax.is_finite()),
    )?;
    builder.inline(
        "def-max-patched-std",
        r"\operatorname{Var}(v)\le V_{\max}^2",
        input.clone(),
        le(
            "complete_raw_variance_by_absolute_support",
            vraw,
            vmax * vmax,
        ),
    )?;
    builder.shown(
        "lem-rescale-derivative-lower-bound",
        r"\inf_{z \in Z_{\mathrm{supp}}}",
        input.clone(),
        hyps.clone(),
        vec![
            positive_identity(
                "actual_family_derivative_infimum",
                family_minimum,
                2. * (-derivative_support).exp() / (1. + (-derivative_support).exp()).powi(2),
            ),
            upper(
                "attained_derivatives_above_uniform_family_infimum",
                family_minimum,
                z.iter()
                    .map(|&z| logistic_derivative(z))
                    .fold(f64::INFINITY, f64::min),
            ),
        ],
    )?;
    bind_inline(
        builder,
        "lem-rescale-derivative-lower-bound",
        &input,
        vec![
            (
                r"g'_{\min} > 0",
                condition(
                    "representable_positive_family_logistic_floor",
                    family_minimum > 0.,
                ),
            ),
            (
                r"Z_{\text{supp}} := \left[ -2V_{\max}/\sigma'_{\min,\text{patch}}, 2V_{\max}/\sigma'_{\min,\text{patch}} \right]",
                vec![
                    positive_identity(
                        "score_support_radius_definition",
                        derivative_support,
                        2. * vmax / smin,
                    ),
                    upper(
                        "all_attained_scores_in_symmetric_support",
                        z.iter().map(|v| v.abs()).fold(0., f64::max),
                        derivative_support,
                    ),
                ],
            ),
        ],
    )?;
    builder.inline(
        "lem-compact-support-z-scores",
        r"\sigma'_{\min,\text{patch}} > 0",
        input.clone(),
        vec![
            hypothesis("positive_common_score_regularizer", smin > 0.),
            upper(
                "actual_compact_score_support",
                z.iter().map(|v| v.abs()).fold(0., f64::max),
                derivative_support,
            ),
        ],
    )?;
    for &score in z {
        let point = json!({"rescale":input,"score":score,"native_canonical_g":map.map(score)?});
        bind_inline(
            builder,
            "lem-rescale-derivative-lower-bound",
            &point,
            vec![
                (
                    r"g_A(z) = 2 / (1 + e^{-z})",
                    vec![positive_identity(
                        "native_canonical_g_formula",
                        map.map(score)?,
                        2. / (1. + (-score).exp()),
                    )],
                ),
                (
                    r"g'_A(z) = 2e^{-z} / (1+e^{-z})^2",
                    vec![positive_identity(
                        "canonical_g_derivative_formula",
                        logistic_derivative(score),
                        2. * (-score).exp() / (1. + (-score).exp()).powi(2),
                    )],
                ),
                (
                    r"z \in \mathbb{R}",
                    condition("finite_operational_score", score.is_finite()),
                ),
            ],
        )?;
        builder.shown(
            "def-logistic-rescale",
            r"g_A(z) :=",
            point,
            hyps.clone(),
            vec![positive_identity(
                "chapter3_native_canonical_logistic_definition",
                map.map(score)?,
                2. / (1. + (-score).exp()),
            )],
        )?;
    }
    let label = "rem-fixed-rescale-attainable-variance";
    let kappa = 0.9 * vout;
    bind_inline(
        builder,
        label,
        &input,
        vec![(
            r"\kappa\le\operatorname{Var}(d')\le R^2/4",
            vec![
                upper("attainable_realized_variance_floor", kappa, vout),
                upper("attainable_realized_variance_ceiling", vout, ceiling),
            ],
        )],
    )?;
    // A second, fully tied draw is retained before taking the outer expectation.
    let tied = vec![mu; raw.len()];
    let obs = ObservationBatch::positions(TensorBatch::vectors(raw.len(), 1, vec![0.; raw.len()])?);
    let (tied_z, _) =
        Standardizer::Global { sigma_min: smin }.apply(&tied, &vec![true; raw.len()], &obs)?;
    let native_map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: offset,
    };
    let tied_output = tied_z
        .iter()
        .map(|&z| native_map.map(z))
        .collect::<Result<Vec<_>>>()?;
    let expected = 0.5 * vout + 0.5 * variance(&tied_output);
    let lawinput = json!({"rescale":input,"law_probabilities":[0.5,0.5],"native_outputs":[output,tied_output],"tied_raw_draw":tied,"expected_output_variance":expected,"kappa":0.9*expected});
    builder.inline(
        label,
        r"\kappa\le\mathbb E\operatorname{Var}(d')\le R^2/4",
        lawinput,
        vec![
            upper(
                "attainable_expected_variance_floor",
                0.9 * expected,
                expected,
            ),
            upper("attainable_expected_variance_ceiling", expected, ceiling),
        ],
    )?;
    builder.add(&[label],r"\kappa>R^2/4".into(),json!({"native_case":input,"candidate_kappa":2.*ceiling,"admissible_variance_lower_bound":false,"reason":"candidate exceeds the actual output interval variance ceiling"}),hyps.clone(),vec![hypothesis("incompatible_candidate_floor_exceeds_ceiling",2.*ceiling>ceiling),upper("actual_output_cannot_attain_candidate_floor",vout,ceiling)],"Necessary-condition diagnostic: the proposed floor is deliberately incompatible; this record verifies infeasibility, not an admissible variance lower bound.")?;
    for gain in [0.25_f64, 1., 4.] {
        let floor = gain * logistic_derivative(gain * radius);
        let lo = map.map(-gain * radius)?;
        let hi = map.map(gain * radius)?;
        let gaininput = json!({"base_rescale_case":input,"auxiliary_configured_gain_gamma":gain,"positive_derivative_floor":floor,"gain_output_interval":[lo,hi],"scope":"scalar property of an already configured g(gamma*z); no gain is introduced into the native algorithm"});
        builder.inline(
            label,
            r"\gamma>0",
            gaininput.clone(),
            condition("positive_configured_scalar_gain", gain > 0.),
        )?;
        builder.inline(
            label,
            r"\gamma\min_{|u|\le\gamma Z}g'(u)",
            gaininput,
            vec![
                positive_identity(
                    "configured_composed_derivative_minimum",
                    floor,
                    gain * logistic_derivative(-gain * radius),
                ),
                upper("composed_output_range_ceiling", hi - lo, 2.),
            ],
        )?;
    }
    Ok(())
}

fn averaged_group_bounds(builder: &mut Builder) -> Result<()> {
    for copies in [1_usize, 4, 16] {
        for oriented in [true, false] {
            let expand = |base: &[f64]| {
                base.iter()
                    .flat_map(|&v| std::iter::repeat_n(v, copies))
                    .collect::<Vec<_>>()
            };
            let separated = expand(&[0.4, 0.6, 1.4, 1.6]);
            let tied = expand(&[1., 1., 1., 1.]);
            let inverted = expand(&[1.4, 1.6, 0.4, 0.6]);
            let law = if oriented {
                vec![(0.4, separated), (0.6, tied)]
            } else {
                vec![(0.25, separated), (0.25, inverted), (0.5, tied)]
            };
            let k = law[0].1.len();
            let fh = 0.5_f64;
            let fl = 0.5_f64;
            let a = 0.4_f64;
            let b = 1.6_f64;
            let r = b - a;
            let mut total = 0.;
            let mut within = 0.;
            let mut mean_gap_squared = 0.;
            let mut mean_vector = vec![0.; k];
            let mut worlds = Vec::new();
            for (mass, raw) in &law {
                let mh = mean(&raw[..k / 2]);
                let ml = mean(&raw[k / 2..]);
                let sh = variance(&raw[..k / 2]);
                let sl = variance(&raw[k / 2..]);
                let v = variance(raw);
                let w = fh * sh + fl * sl;
                let gap = ml - mh;
                total += mass * v;
                within += mass * w;
                mean_gap_squared += mass * gap * gap;
                for (i, &value) in raw.iter().enumerate() {
                    mean_vector[i] += mass * value;
                }
                let obs = ObservationBatch::positions(TensorBatch::vectors(k, 1, vec![0.; k])?);
                let (z, stats) =
                    Standardizer::Global { sigma_min: 0.1 }.apply(raw, &vec![true; k], &obs)?;
                let positive = PositiveMap::Logistic {
                    amplitude: 2.,
                    floor: 0.1,
                };
                let fitness = z
                    .iter()
                    .map(|&z| positive.map(z))
                    .collect::<Result<Vec<_>>>()?;
                let decision = CloneDecision::default();
                let pressure_h = fitness[..k / 2]
                    .iter()
                    .map(|&own| {
                        fitness[k / 2..]
                            .iter()
                            .map(|&donor| decision.acceptance_probability(0, own, donor))
                            .sum::<f64>()
                            / (k / 2) as f64
                    })
                    .sum::<f64>()
                    / (k / 2) as f64;
                worlds.push(json!({"probability":mass,"raw":raw,"group_means":[mh,ml],"group_variances":[sh,sl],"variance":v,"within_variance":w,"signed_gap":gap,"native_statistics":stats,"native_logistic_fitness":fitness,"native_H_to_L_clipped_pressure":pressure_h,"native_decision":decision}));
            }
            let kappa = 0.9 * total;
            let bound = 1.1 * within;
            let q = (kappa - bound) / (fh * fl);
            let expected_pressure = worlds
                .iter()
                .map(|world| {
                    world["probability"].as_f64().unwrap()
                        * world["native_H_to_L_clipped_pressure"].as_f64().unwrap()
                })
                .sum::<f64>();
            let input = json!({"conditioned_entering_information":"fixed finite joint measurement law and fixed H/L partition","worlds":worlds,"H_indices":(0..k/2).collect::<Vec<_>>(),"L_indices":(k/2..k).collect::<Vec<_>>(),"a":a,"b":b,"R":r,"f_H":fh,"f_L":fl,"kappa":kappa,"B_within":bound,"Q":q,"conditionally_nonnegative_orientation":oriented,"computed":{"expected_variance":total,"expected_within_variance":within,"expected_squared_mean_gap":mean_gap_squared,"mean_measurement_vector":mean_vector,"native_clipped_pressure_averaged_after_world_evaluation":expected_pressure}});
            let hyps = vec![
                hypothesis(
                    "fixed_finite_probability_law_before_measurement",
                    law.iter().all(|(p, raw)| {
                        *p >= 0. && raw.len() == k && raw.iter().all(|&v| v >= a && v <= b)
                    }),
                ),
                relative(
                    "joint_law_probability_mass",
                    law.iter().map(|(p, _)| *p).sum(),
                    1.,
                    1.,
                ),
                hypothesis(
                    "positive_averaged_separation_remainder",
                    kappa > bound && fh > 0. && fl > 0.,
                ),
                upper("actual_expected_total_variance_floor", kappa, total),
                upper("actual_expected_within_variance_ceiling", within, bound),
            ];
            let label = "cor-averaged-group-separation";
            bind_inline(
                builder,
                label,
                &input,
                vec![
                    (
                        r"v\in[a,b]^k",
                        condition(
                            "same_realized_measurement_support_in_every_world",
                            law.iter()
                                .all(|(_, raw)| raw.iter().all(|&v| v >= a && v <= b)),
                        ),
                    ),
                    (
                        r"Q=(\kappa-B_{\rm within})/(f_Hf_L)",
                        vec![positive_identity(
                            "positive_averaged_squared_gap_floor",
                            q,
                            (kappa - bound) / (fh * fl),
                        )],
                    ),
                    (
                        r"R=b-a",
                        equal("common_measurement_interval_width", r, b - a),
                    ),
                    (
                        r"0<Q\le R^2",
                        vec![
                            hypothesis("strict_positive_averaged_Q", q > 0.),
                            upper("feasible_averaged_squared_gap_floor", q, r * r),
                        ],
                    ),
                    (
                        r"Q\le R^2",
                        le("feasible_average_floor_from_support", q, r * r),
                    ),
                    (
                        r"\mathbb E[\Delta^2\mid\mathcal G]\ge Q",
                        ge(
                            "actual_expected_squared_mean_gap_lower",
                            mean_gap_squared,
                            q,
                        ),
                    ),
                ],
            )?;
            builder.shown(
                label,
                r"\mathbb E[s^2\mid\mathcal G]",
                input.clone(),
                hyps.clone(),
                vec![
                    upper("averaged_total_variance_premise", kappa, total),
                    upper("averaged_within_variance_premise", within, bound),
                    hypothesis("strict_within_vs_total_floor", bound < kappa),
                ],
            )?;
            builder.shown(
                label,
                r"\mathbb E[(\mu_L-\mu_H)^2",
                input.clone(),
                hyps.clone(),
                ge("conditional_squared_group_gap_bound", mean_gap_squared, q),
            )?;
            for fraction in [0.2_f64, 0.6, 0.95] {
                let t = fraction * q.sqrt();
                let probability = worlds
                    .iter()
                    .filter(|w| w["signed_gap"].as_f64().unwrap().abs() > t)
                    .map(|w| w["probability"].as_f64().unwrap())
                    .sum::<f64>();
                let lower = (q - t * t) / (r * r - t * t);
                let case = json!({"averaged_law":input,"t":t,"actual_magnitude_event_probability":probability,"event_probability_lower_bound":lower});
                bind_inline(
                    builder,
                    label,
                    &case,
                    vec![
                        (
                            r"0<t<\sqrt Q",
                            vec![
                                hypothesis("strict_positive_gap_event_threshold", t > 0.),
                                hypothesis("threshold_below_guaranteed_rms_gap", t < q.sqrt()),
                            ],
                        ),
                        (
                            r"R^2-t^2>0",
                            condition("positive_event_bound_denominator", r * r - t * t > 0.),
                        ),
                    ],
                )?;
                builder.shown(
                    label,
                    r"\Pr\{|\mu_L-\mu_H|>t",
                    case.clone(),
                    hyps.clone(),
                    vec![
                        upper(
                            "actual_conditional_magnitude_event_lower",
                            lower,
                            probability,
                        ),
                        hypothesis("strict_positive_event_probability_floor", lower > 0.),
                    ],
                )?;
                builder.shown(
                    label,
                    r"\le t^2+(R^2-t^2)",
                    case.clone(),
                    hyps.clone(),
                    le(
                        "actual_squared_gap_by_threshold_event",
                        mean_gap_squared,
                        t * t + (r * r - t * t) * probability,
                    ),
                )?;
                if oriented {
                    builder.inline(
                        label,
                        r"\mu_L-\mu_H\ge0",
                        case.clone(),
                        condition(
                            "actual_conditional_nonnegative_orientation",
                            worlds
                                .iter()
                                .all(|w| w["signed_gap"].as_f64().unwrap() >= 0.),
                        ),
                    )?;
                    let signed = worlds
                        .iter()
                        .filter(|w| w["signed_gap"].as_f64().unwrap() > t)
                        .map(|w| w["probability"].as_f64().unwrap())
                        .sum::<f64>();
                    builder.inline(
                        label,
                        r"\{\Delta>t\}",
                        case,
                        vec![
                            relative(
                                "oriented_signed_and_magnitude_event_equality",
                                signed,
                                probability,
                                1.,
                            ),
                            upper(
                                "native_event_pressure_evaluated_before_average",
                                0.,
                                expected_pressure,
                            ),
                        ],
                    )?;
                }
            }
            for w in &worlds {
                let gap = w["signed_gap"].as_f64().unwrap();
                builder.inline(
                    label,
                    r"|\mu_L-\mu_H|\le R",
                    json!({"law":input,"world":w}),
                    le("actual_world_mean_gap_support_bound", gap.abs(), r),
                )?;
                builder.inline(
                    label,
                    r"\Delta=\mu_L-\mu_H",
                    json!({"law":input,"world":w}),
                    equal(
                        "actual_world_signed_mean_gap_definition",
                        gap,
                        w["group_means"][1].as_f64().unwrap()
                            - w["group_means"][0].as_f64().unwrap(),
                    ),
                )?;
            }
            let label = "thm-geometry-guarantees-variance";
            let marginal_gap = mean(&mean_vector[k / 2..]) - mean(&mean_vector[..k / 2]);
            if oriented {
                let guaranteed = 0.9 * marginal_gap;
                let centered_expectation = variance(&mean_vector);
                let residual = law
                    .iter()
                    .map(|(p, raw)| {
                        let deviation = raw
                            .iter()
                            .zip(&mean_vector)
                            .map(|(v, m)| v - m)
                            .collect::<Vec<_>>();
                        p * variance(&deviation)
                    })
                    .sum::<f64>();
                let proofinput = json!({"joint_law":input,"marginal_mean_gap_floor":guaranteed,"marginal_mean_variance":centered_expectation,"expected_centered_random_remainder_variance":residual});
                let centering_matrix = (0..k)
                    .map(|i| {
                        (0..k)
                            .map(|j| {
                                if i == j {
                                    1. - 1. / k as f64
                                } else {
                                    -1. / k as f64
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let matrix_input = json!({"actual_joint_measurement_law":proofinput,"explicit_empirical_centering_matrix":centering_matrix,"alive_count":k});
                bind_inline(
                    builder,
                    label,
                    &matrix_input,
                    vec![
                        (
                            r"f_H,f_L>0",
                            vec![
                                hypothesis("actual_high_population_nonempty", fh > 0.),
                                hypothesis("actual_low_population_nonempty", fl > 0.),
                            ],
                        ),
                        (
                            r"\Delta_d>0",
                            condition("actual_positive_marginal_mean_gap_floor", guaranteed > 0.),
                        ),
                        (
                            r"P=I-k^{-1}\mathbf1\mathbf1^\top",
                            (0..k)
                                .flat_map(|i| {
                                    let centering_matrix = &centering_matrix;
                                    (0..k).map(move |j| {
                                        relative(
                                            "explicit_centering_matrix_entry",
                                            centering_matrix[i][j],
                                            (if i == j { 1. } else { 0. }) - 1. / k as f64,
                                            1.,
                                        )
                                    })
                                })
                                .collect(),
                        ),
                        (
                            r"\mathbb E\operatorname{Var}(d)\geq\operatorname{Var}(\mathbb Ed)",
                            vec![
                                upper(
                                    "actual_correlated_measurement_jensen_variance",
                                    centered_expectation,
                                    total,
                                ),
                                relative(
                                    "expected_centered_norm_retains_joint_random_remainder",
                                    total,
                                    centered_expectation + residual,
                                    total,
                                ),
                            ],
                        ),
                    ],
                )?;
                for (probability, measurements) in &law {
                    let projected = centering_matrix
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(measurements)
                                .map(|(&a, &v)| a * v)
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>();
                    let atom = json!({"actual_measurement_law":matrix_input,"realization_probability":probability,"measurement_vector":measurements,"projected_vector":projected});
                    builder.inline(
                        label,
                        r"d=(d_1,\ldots,d_k)",
                        atom.clone(),
                        vec![
                            hypothesis(
                                "actual_measurement_vector_finite_and_correct_alive_width",
                                measurements.len() == k
                                    && measurements.iter().all(|x| x.is_finite()),
                            ),
                            hypothesis("actual_joint_law_atom_positive", *probability > 0.),
                        ],
                    )?;
                    builder.inline(
                        label,
                        r"\operatorname{Var}(d)=k^{-1}\|Pd\|^2",
                        atom,
                        vec![relative(
                            "actual_empirical_variance_equals_centering_matrix_norm",
                            variance(measurements),
                            projected.iter().map(|a| a * a).sum::<f64>() / k as f64,
                            variance(measurements),
                        )],
                    )?;
                }
                builder.shown(
                    label,
                    r"\mathbb E\operatorname{Var}(d)",
                    proofinput.clone(),
                    hyps.clone(),
                    vec![
                        upper(
                            "correlated_measurement_expected_variance_lower",
                            fh * fl * guaranteed * guaranteed,
                            total,
                        ),
                        upper("actual_marginal_mean_separation", guaranteed, marginal_gap),
                    ],
                )?;
                builder.shown(
                    label,
                    r"\mathbb E\|Pd\|^2",
                    proofinput,
                    hyps.clone(),
                    vec![relative(
                        "centered_random_vector_Pythagorean_identity",
                        k as f64 * total,
                        k as f64 * (centered_expectation + residual),
                        k as f64 * total,
                    )],
                )?;
            }
        }
    }
    feasible_group_examples(builder)?;
    Ok(())
}

fn feasible_group_examples(builder: &mut Builder) -> Result<()> {
    let label = "rem-group-separation-feasibility";
    let high = [2_f64 / 5., 3. / 5.];
    let low = [7_f64 / 5., 8. / 5.];
    let whole = high.iter().chain(&low).copied().collect::<Vec<_>>();
    let within = (variance(&high) + variance(&low)) / 2.;
    let gap = mean(&low) - mean(&high);
    let input = json!({"H_values":high,"L_values":low,"empirical_variance":variance(&whole),"within_variance":within,"mean_gap":gap});
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"H=\{2/5,3/5\}",
                vec![
                    relative("high_example_first_value", high[0], 2. / 5., 1.),
                    relative("high_example_second_value", high[1], 3. / 5., 1.),
                ],
            ),
            (
                r"L=\{7/5,8/5\}",
                vec![
                    relative("low_example_first_value", low[0], 7. / 5., 2.),
                    relative("low_example_second_value", low[1], 8. / 5., 2.),
                ],
            ),
            (
                r"s^2=13/50",
                equal("feasible_total_variance", variance(&whole), 13. / 50.),
            ),
            (
                r"s_{\rm within}^2=1/100",
                vec![positive_identity(
                    "feasible_within_group_variance",
                    within,
                    1. / 100.,
                )],
            ),
            (r"\mu_L-\mu_H=1", equal("feasible_signed_mean_gap", gap, 1.)),
            (
                r"\kappa=13/50>B_{\rm within}=1/100",
                vec![
                    relative(
                        "example_exact_variance_floor",
                        variance(&whole),
                        13. / 50.,
                        1.,
                    ),
                    relative("example_exact_within_ceiling", within, 1. / 100., 1.),
                    hypothesis(
                        "example_strict_positive_remainder",
                        13_f64 / 50. > 1. / 100.,
                    ),
                ],
            ),
        ],
    )?;
    let same = [0.5_f64, 1.5];
    let whole = same.iter().chain(&same).copied().collect::<Vec<_>>();
    let input = json!({"H_values":same,"L_values":same,"storage_groups_disjoint":true,"physical_value_multisets_equal":true});
    bind_inline(
        builder,
        label,
        &input,
        vec![
            (
                r"H=L=\{1/2,3/2\}",
                vec![
                    relative("equal_multiset_first_value", same[0], 1. / 2., 1.),
                    relative("equal_multiset_second_value", same[1], 3. / 2., 2.),
                    relative("equal_empirical_group_mean", mean(&same), mean(&same), 1.),
                ],
            ),
            (
                r"s^2=s_{\rm within}^2=1/4",
                vec![
                    positive_identity(
                        "symmetric_counterexample_total_variance",
                        variance(&whole),
                        1. / 4.,
                    ),
                    positive_identity(
                        "symmetric_counterexample_within_variance",
                        variance(&same),
                        1. / 4.,
                    ),
                    relative(
                        "symmetric_counterexample_zero_mean_gap",
                        mean(&same) - mean(&same),
                        0.,
                        1.,
                    ),
                ],
            ),
            (
                r"(b-a)^2/4",
                le(
                    "symmetric_example_global_interval_ceiling",
                    variance(&whole),
                    range(&whole).powi(2) / 4.,
                ),
            ),
        ],
    )?;
    Ok(())
}

fn greedy_history_bounds(builder: &mut Builder) -> Result<()> {
    let label = "lem-greedy-preserves-signal";
    for x in [vec![0_f64, 0.05, 3., 6.], vec![0_f64, 0.05, 3., 6., 9.]] {
        let n = x.len();
        let width = 2_f64;
        let rl = 0.05_f64;
        let dh = 2.95_f64;
        let obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, x.clone())?);
        let distance = Distance::default();
        let kernel = Kernel::Gaussian { width };
        let mut distances = vec![vec![0.; n]; n];
        let mut weights = distances.clone();
        for (i, row) in distances.iter_mut().enumerate() {
            for (j, d) in row.iter_mut().enumerate() {
                *d = distance.compare(&obs, i, &obs, j)?;
                weights[i][j] = kernel.log_weight(*d, ComparisonKind::Distance)?.exp();
            }
        }
        let near = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i && distances[i][j] <= rl)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let dalg = range(&x);
        let cloud = json!({"positions":x,"velocities":vec![0.;n],"native_distance":distance,"native_gaussian_kernel":kernel,"configured_sampling_law":"GaussianGreedy with uniform pivot among current remaining candidates","low_group":[0,1],"isolated_group":(2..n).collect::<Vec<_>>(),"near_candidate_sets":near,"R_L":rl,"D_H":dh,"D_alg":dalg,"distance_matrix":distances,"native_edge_weights":weights});
        builder.inline(
            "def-greedy-pairing-algorithm",
            r"\varepsilon_d > 0",
            cloud.clone(),
            condition(
                "actual_positive_gaussian_greedy_interaction_width",
                width > 0.,
            ),
        )?;
        builder.shown(
            "def-greedy-pairing-algorithm",
            r"w_{ij} :=",
            cloud.clone(),
            vec![hypothesis(
                "actual_positive_native_gaussian_width",
                width > 0.,
            )],
            distances
                .iter()
                .enumerate()
                .flat_map(|(i, row)| {
                    let weights = &weights;
                    row.iter()
                        .enumerate()
                        .filter(move |(j, _)| i != *j)
                        .map(move |(j, &d)| {
                            positive_identity(
                                "native_greedy_gaussian_edge_weight_definition",
                                weights[i][j],
                                (-d * d / (2. * width * width)).exp(),
                            )
                        })
                })
                .collect(),
        )?;
        let hyps = vec![
            hypothesis(
                "positive_separated_greedy_configuration",
                rl >= 0. && rl < dh && width > 0.,
            ),
            hypothesis(
                "actual_separated_near_far_configuration",
                (0..n).all(|i| {
                    (0..n).filter(|&j| j != i).all(|j| {
                        if near[i].contains(&j) {
                            distances[i][j] <= rl
                        } else {
                            distances[i][j] >= dh
                        }
                    })
                }),
            ),
            hypothesis(
                "finite_positive_native_edges",
                weights.iter().flatten().all(|w| w.is_finite() && *w > 0.),
            ),
        ];
        bind_inline(
            builder,
            "def-geometric-partition",
            &cloud,
            vec![
                (
                    r"0\leq R_L<D_H",
                    vec![
                        upper("nonnegative_near_radius", 0., rl),
                        hypothesis("strict_positive_geometric_gap", rl < dh),
                    ],
                ),
                (
                    r"C_j\subset\mathcal A_k\setminus\{j\}",
                    condition(
                        "near_sets_are_actual_nonself_alive_candidates",
                        near.iter()
                            .enumerate()
                            .all(|(i, g)| g.iter().all(|&j| j < n && j != i)),
                    ),
                ),
                (
                    r"D_{\mathrm{alg}}=(D_{\mathrm{valid}}^2+4\lambda_{\mathrm{alg}}V_{\max}^2)^{1/2}",
                    vec![
                        positive_identity(
                            "phase_domain_candidate_distance_bound_with_zero_velocity",
                            dalg,
                            (range(&x).powi(2) + 4. * 1_f64 * 0_f64.powi(2)).sqrt(),
                        ),
                        upper(
                            "actual_all_candidate_distance_ceiling",
                            distances.iter().flatten().copied().fold(0., f64::max),
                            dalg,
                        ),
                    ],
                ),
            ],
        )?;
        for (i, distance_row) in distances.iter().enumerate().skip(2) {
            for (j, &edge_distance) in distance_row.iter().enumerate() {
                if i != j {
                    builder.inline(
                        "def-geometric-partition",
                        r"d_{\mathrm{alg}}(i,u)\geq D_H",
                        json!({"cloud":cloud,"isolated_row":i,"candidate":j}),
                        ge("actual_isolated_nonself_edge_floor", edge_distance, dh),
                    )?;
                }
            }
        }
        let mut stack = vec![(
            (0..n).collect::<Vec<_>>(),
            (0..n).collect::<Vec<_>>(),
            1_f64,
        )];
        let mut outcomes = Vec::new();
        let mut steps = Vec::new();
        let mut history_far = vec![0.; n];
        let mut history_checks = Vec::new();
        while let Some((remaining, map, mass)) = stack.pop() {
            if remaining.len() <= 1 {
                outcomes.push((map, mass));
                continue;
            }
            for &pivot in &remaining {
                let eligible = remaining
                    .iter()
                    .copied()
                    .filter(|&j| j != pivot)
                    .collect::<Vec<_>>();
                let normalizer = eligible.iter().map(|&j| weights[pivot][j]).sum::<f64>();
                let nnear = remaining
                    .iter()
                    .filter(|&&j| near[pivot].contains(&j))
                    .count();
                let nfar = eligible.len() - nnear;
                let q = if nnear > 0 {
                    (nfar as f64 / nnear as f64
                        * (-(dh * dh - rl * rl) / (2. * width * width)).exp())
                    .min(1.)
                } else {
                    1.
                };
                let far_probability = eligible
                    .iter()
                    .filter(|&&j| !near[pivot].contains(&j))
                    .map(|&j| weights[pivot][j] / normalizer)
                    .sum::<f64>();
                let mean_distance = eligible
                    .iter()
                    .map(|&j| weights[pivot][j] / normalizer * distances[pivot][j])
                    .sum::<f64>();
                let pivot_mass = mass / remaining.len() as f64;
                let step = json!({"cloud":cloud,"U":remaining,"pivot":pivot,"history_probability_before_pivot":mass,"history_and_pivot_probability":pivot_mass,"nonself_candidates":eligible,"n_near":nnear,"n_far":nfar,"Z_j_U":normalizer,"q_U_j":q,"actual_far_probability":far_probability,"actual_conditional_mean_distance":mean_distance,"case_scope":if nnear>0 {"positive remaining near count"}else{"fallback: no nearby candidate remains"}});
                let conditional_probs = eligible
                    .iter()
                    .map(|&j| weights[pivot][j] / normalizer)
                    .collect::<Vec<_>>();
                let mut proposal_checks = vec![relative(
                    "actual_native_greedy_conditional_proposal_total_mass",
                    conditional_probs.iter().sum(),
                    1.,
                    1.,
                )];
                proposal_checks.extend(eligible.iter().zip(&conditional_probs).map(
                    |(&j, &probability)| {
                        positive_identity(
                            "native_greedy_conditional_companion_probability",
                            probability,
                            weights[pivot][j]
                                / eligible.iter().map(|&l| weights[pivot][l]).sum::<f64>(),
                        )
                    },
                ));
                builder.inline("def-greedy-pairing-algorithm",r"P(\text{choose } j) = w_{ij} / (\sum_{l \in U} w_{il})",json!({"actual_greedy_history":step,"source_U_after_removing_pivot":eligible,"actual_native_companion_probabilities":conditional_probs}),proposal_checks)?;
                bind_inline(
                    builder,
                    label,
                    &step,
                    vec![
                        (
                            r"n_{\mathrm{far}}=|U\setminus(C_j\cup\{j\})|",
                            equal(
                                "actual_remaining_far_count",
                                nfar as f64,
                                remaining
                                    .iter()
                                    .filter(|&&j| j != pivot && !near[pivot].contains(&j))
                                    .count() as f64,
                            ),
                        ),
                        (
                            r"Z_j(U)=\sum_{u\in U\setminus\{j\}}w_{ju}",
                            vec![positive_identity(
                                "actual_native_conditional_normalizer",
                                normalizer,
                                remaining
                                    .iter()
                                    .filter(|&&j| j != pivot)
                                    .map(|&j| weights[pivot][j])
                                    .sum::<f64>(),
                            )],
                        ),
                    ],
                )?;
                if !near[pivot].is_empty() {
                    let mut condition_hyps = hyps.clone();
                    condition_hyps.push(hypothesis("pivot_belongs_to_actual_low_group", pivot < 2));
                    if nnear > 0 {
                        builder.inline(
                            label,
                            r"n_{\mathrm{near}}=|C_j\cap U|>0",
                            step.clone(),
                            vec![
                                relative(
                                    "actual_remaining_near_count",
                                    nnear as f64,
                                    near[pivot].iter().filter(|i| remaining.contains(i)).count()
                                        as f64,
                                    n as f64,
                                ),
                                hypothesis("remaining_near_count_is_positive", nnear > 0),
                            ],
                        )?;
                        builder.shown(
                            label,
                            r"q(U,j)=",
                            step.clone(),
                            condition_hyps.clone(),
                            vec![
                                relative(
                                    "history_tail_constant_definition",
                                    q,
                                    (nfar as f64 / nnear as f64
                                        * (-(dh * dh - rl * rl) / (2. * width * width)).exp())
                                    .min(1.),
                                    1.,
                                ),
                                upper(
                                    "actual_far_weight_vs_history_tail_constant",
                                    far_probability,
                                    q,
                                ),
                            ],
                        )?;
                    } else {
                        builder.inline(
                            label,
                            r"q(U,j)=1",
                            step.clone(),
                            equal("no_remaining_near_candidate_fallback", q, 1.),
                        )?;
                    }
                    builder.shown(
                        label,
                        r"\mathbb P(c_j\notin C_j\mid U,j)",
                        step.clone(),
                        condition_hyps,
                        vec![
                            upper(
                                "actual_conditional_far_edge_probability",
                                far_probability,
                                q,
                            ),
                            upper(
                                "actual_conditional_distance_from_near_far_counts",
                                mean_distance,
                                rl + (dalg - rl) * q,
                            ),
                        ],
                    )?;
                }
                let mut conditional_terms = Vec::new();
                for &j in &remaining {
                    let pivot_term = if j == pivot {
                        eligible
                            .iter()
                            .filter(|&&u| !near[j].contains(&u))
                            .map(|&u| weights[pivot][u] / normalizer)
                            .sum::<f64>()
                    } else {
                        0.
                    };
                    let incoming_term = if j != pivot && !near[j].contains(&pivot) {
                        weights[pivot][j] / normalizer
                    } else {
                        0.
                    };
                    let direct = eligible
                        .iter()
                        .filter(|&&u| {
                            (pivot == j && !near[j].contains(&u))
                                || (u == j && !near[j].contains(&pivot))
                        })
                        .map(|&u| weights[pivot][u] / normalizer)
                        .sum::<f64>();
                    history_far[j] += pivot_mass * (pivot_term + incoming_term);
                    history_checks.push(if direct > 0. {
                        positive_identity(
                            "full_history_pivot_plus_incoming_conditional_identity",
                            direct,
                            pivot_term + incoming_term,
                        )
                    } else {
                        relative(
                            "zero_conditional_history_term",
                            direct,
                            pivot_term + incoming_term,
                            1.,
                        )
                    });
                    conditional_terms.push(json!({"row_j":j,"pivot_term":pivot_term,"incoming_term":incoming_term,"direct_conditional_incident_far_probability":direct}));
                }
                steps.push(json!({"step":step,"conditional_far_terms":conditional_terms}));
                for &donor in &eligible {
                    let atom = json!({"step":step,"donor":donor,"native_edge_weight":weights[pivot][donor],"measurement_distance":distances[pivot][donor]});
                    builder.inline(
                        label,
                        r"w_{ju}=\exp[-d_{\mathrm{alg}}(j,u)^2/(2\varepsilon_d^2)]",
                        atom.clone(),
                        vec![positive_identity(
                            "actual_native_gaussian_edge_weight",
                            weights[pivot][donor],
                            (-distances[pivot][donor].powi(2) / (2. * width * width)).exp(),
                        )],
                    )?;
                    builder.inline(
                        label,
                        r"d_j\leq D_{\mathrm{alg}}",
                        atom.clone(),
                        le(
                            "actual_selected_distance_below_domain_ceiling",
                            distances[pivot][donor],
                            dalg,
                        ),
                    )?;
                    if near[pivot].contains(&donor) {
                        builder.inline(
                            label,
                            r"d_j\leq R_L",
                            atom,
                            le("actual_near_selected_distance", distances[pivot][donor], rl),
                        )?;
                    }
                    let probability = pivot_mass * weights[pivot][donor] / normalizer;
                    let mut next = map.clone();
                    next[pivot] = donor;
                    next[donor] = pivot;
                    stack.push((
                        remaining
                            .iter()
                            .copied()
                            .filter(|&i| i != pivot && i != donor)
                            .collect(),
                        next,
                        probability,
                    ));
                }
            }
        }
        let mass = outcomes.iter().map(|(_, p)| *p).sum::<f64>();
        let mut mean_distances = vec![0.; n];
        let mut self_probabilities = vec![0.; n];
        let mut final_far = vec![0.; n];
        for (map, p) in &outcomes {
            for i in 0..n {
                mean_distances[i] += p * distances[i][map[i]];
                if map[i] == i {
                    self_probabilities[i] += p;
                } else if !near[i].contains(&map[i]) {
                    final_far[i] += p;
                }
            }
        }
        let final_input = json!({"cloud":cloud,"all_greedy_outcomes":outcomes,"all_conditional_steps":steps,"computed":{"total_probability_mass":mass,"mean_measurement_distances":mean_distances,"self_match_probabilities":self_probabilities,"final_nonself_far_probabilities":final_far,"history_far_probabilities":history_far}});
        let mut checks = vec![relative(
            "exhaustive_native_greedy_probability_mass",
            mass,
            1.,
            1.,
        )];
        checks.extend(history_checks);
        for i in 0..2 {
            checks.push(positive_identity(
                "final_far_probability_includes_pivot_and_incoming_histories",
                final_far[i],
                history_far[i],
            ));
        }
        builder.shown(
            label,
            r"\mathbf1_{\{j\in U_t,I_t=j\}}",
            final_input.clone(),
            hyps.clone(),
            checks,
        )?;
        for i in 0..n {
            let row = json!({"history_law":final_input,"row":i});
            if near[i].is_empty() {
                builder.shown(
                    label,
                    r"\mathbb E[d_i\mid S]",
                    row,
                    hyps.clone(),
                    ge(
                        "isolated_measurement_with_actual_odd_self_mass",
                        mean_distances[i],
                        dh * (1. - self_probabilities[i]),
                    ),
                )?;
            } else {
                builder.shown(
                    label,
                    r"\mathbb E[d_j\mid S]",
                    row,
                    hyps.clone(),
                    le(
                        "low_final_marginal_includes_incoming_histories",
                        mean_distances[i],
                        rl + (dalg - rl) * final_far[i],
                    ),
                )?;
            }
        }
    }
    Ok(())
}

fn scalar_signal_and_operator_bounds(builder: &mut Builder) -> Result<()> {
    for copies in [1_usize, 2, 8, 32] {
        let k = 4 * copies;
        let n = k + 2;
        let positions = (0..n)
            .map(|i| [-0.8, -0.2, 0.1, 0.7][i % 4])
            .collect::<Vec<f64>>();
        let velocities = (0..n)
            .map(|i| [0.4, -0.1, 0.2, -0.3][i % 4])
            .collect::<Vec<f64>>();
        let alive = (0..n).map(|i| i < k).collect::<Vec<_>>();
        let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, positions.clone())?);
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(n, 1, velocities.clone())?,
        );
        let config = GasConfig::euclidean(1, 0.04)?;
        let velocity_regularizer = 0.3_f64;
        let raw = positions
            .iter()
            .zip(&velocities)
            .map(|(&x, &v)| Ok(-Benchmark::Quadratic.value(&[x])? - velocity_regularizer * v * v))
            .collect::<Result<Vec<_>>>()?;
        let reward = RewardBatch::new(raw.clone(), Provenance::default());
        let companions = (0..n)
            .map(|i| if i < k { (i + 1) % k } else { i })
            .collect::<Vec<_>>();
        let distances = companions
            .iter()
            .enumerate()
            .map(|(i, &j)| config.distance_donors.distance.compare(&obs, i, &obs, j))
            .collect::<Result<Vec<f64>>>()?;
        let native = config
            .fitness
            .evaluate(&reward, &distances, &alive, &obs, 0)?;
        let base = PositiveMap::Logistic {
            amplitude: 2.,
            floor: 0.,
        };
        let eta = 0.1_f64;
        let maximum = 2. + eta;
        let smin = 0.1_f64;
        let rescaled_reward = native
            .reward_z
            .iter()
            .map(|&z| config.fitness.reward_map.map(z))
            .collect::<Result<Vec<_>>>()?;
        let rescaled_distance = native
            .diversity_z
            .iter()
            .map(|&z| config.fitness.diversity_map.map(z))
            .collect::<Result<Vec<_>>>()?;
        let measurement_reward = raw
            .iter()
            .zip(&alive)
            .map(|(&x, &a)| if a { x } else { 0. })
            .collect::<Vec<_>>();
        let measurement_distance = native
            .separation
            .iter()
            .zip(&alive)
            .map(|(&x, &a)| if a { x } else { 0. })
            .collect::<Vec<_>>();
        let input = json!({"copies":copies,"alive_count":k,"positions":positions,"velocities":velocities,"alive":alive,"native_preset":config,"companion_map":companions,"complete_raw_reward":raw,"raw_feature_distance":distances,"native_intermediates":native,"positive_reward":rescaled_reward,"positive_diversity":rescaled_distance,"masked_measurement_reward":measurement_reward,"masked_measurement_diversity":measurement_distance,"velocity_regularizer":velocity_regularizer});
        let hyps = vec![
            hypothesis(
                "nonempty_actual_alive_population",
                k > 0 && alive.iter().filter(|&&a| a).count() == k,
            ),
            hypothesis(
                "complete_native_input_finite",
                obs.fields
                    .values()
                    .all(|f| f.values().iter().all(|x| x.is_finite()))
                    && raw.iter().all(|x| x.is_finite()),
            ),
            hypothesis(
                "fixed_positive_measurement_and_score_floors",
                config.fitness.distance_floor > 0. && smin > 0. && eta > 0.,
            ),
        ];
        builder.shown(
            "def-raw-value-operators",
            r"r_i :=",
            input.clone(),
            hyps.clone(),
            (0..k)
                .map(|i| {
                    relative(
                        "actual_complete_native_reward_with_velocity_term",
                        raw[i],
                        -0.5 * positions[i].powi(2) - velocity_regularizer * velocities[i].powi(2),
                        raw[i].abs(),
                    )
                })
                .collect(),
        )?;
        builder.shown(
            "def-raw-value-operators",
            r"\ell_i:=",
            input.clone(),
            hyps.clone(),
            (0..k)
                .flat_map(|i| {
                    [
                        positive_identity(
                            "actual_native_feature_edge_and_separation",
                            native.separation[i],
                            (distances[i].powi(2) + config.fitness.distance_floor.powi(2)).sqrt(),
                        ),
                        hypothesis(
                            "positive_configured_distance_floor",
                            config.fitness.distance_floor > 0.,
                        ),
                    ]
                })
                .collect(),
        )?;
        let distance_floor = config.fitness.distance_floor;
        let floor_differences = (0..k)
            .map(|i| native.separation[i] - distances[i])
            .collect::<Vec<_>>();
        let difference_l2 =
            (floor_differences.iter().map(|v| v * v).sum::<f64>() / k as f64).sqrt();
        builder.inline("lem-cloning-distance-floor-transfer",r"\|d-\ell\|_{L^2}\leq\delta_D",json!({"native_measurements":input,"actual_measurement_differences":floor_differences,"actual_empirical_difference_L2":difference_l2}),le("actual_native_regularized_distance_empirical_L2_error",difference_l2,distance_floor))?;
        builder.inline("lem-cloning-distance-floor-transfer",r"\lambda_{\rm alg}>0",input.clone(),condition("actual_native_phase_distance_velocity_weight_positive",matches!(config.distance_donors.distance,Distance::SquashedPhaseSpace{lambda,..}if lambda>0.)))?;
        for (i, &feature_distance) in distances.iter().enumerate().take(k) {
            builder.inline("lem-cloning-distance-floor-transfer",r"\sqrt{u^2+\delta_D^2}\leq u+\delta_D",json!({"native_measurements":input,"row":i,"actual_feature_distance_u":feature_distance}),vec![hypothesis("actual_feature_distance_nonnegative",feature_distance>=0.),upper("actual_native_positive_distance_floor_pointwise_envelope",native.separation[i],feature_distance+distance_floor)])?;
        }
        for i in k..n {
            let dead = json!({"actual_native_input":input,"dead_slot":i,"mask_convention":"Raw stored rewards and separations are retained; the mathematical measurement representatives set ineligible components to zero. Native standardization and combination use the eligibility mask."});
            builder.add(&["def-raw-value-operators"], r"r_j = 0".into(), dead.clone(), vec![hypothesis("actual_ineligible_slot", !alive[i])], equal("actual_masked_reward_measurement", measurement_reward[i], 0.), "A masked raw-channel measurement convention, with the original native storage retained explicitly. This identity does not assert an in-place overwrite of the stored objective values.")?;
            builder.add(&["def-raw-value-operators"], r"d_j = 0".into(), dead, vec![hypothesis("actual_ineligible_slot", !alive[i])], equal("actual_masked_separation_measurement", measurement_distance[i], 0.), "A masked raw-channel measurement convention, with actual native separation storage retained. The eligibility mask determines the empirical measure.")?;
        }
        for (name, values, z, stats) in [
            (
                "reward",
                &native.oriented_reward,
                &native.reward_z,
                &native.reward_stats,
            ),
            (
                "diversity",
                &native.separation,
                &native.diversity_z,
                &native.diversity_stats,
            ),
        ] {
            let xs = &values[..k];
            let mu = mean(xs);
            let var = variance(xs);
            let scale = (var + smin * smin).sqrt();
            let weights = vec![1. / k as f64; k];
            let measure_mean = weights.iter().zip(xs).map(|(&w, &x)| w * x).sum::<f64>();
            let measure_second = weights
                .iter()
                .zip(xs)
                .map(|(&w, &x)| w * x * x)
                .sum::<f64>();
            let channel = json!({"actual_pipeline":input,"channel":name,"eligible_empirical_measure":{"atoms":xs,"weights":weights},"computed":{"mean":mu,"variance":var,"second_moment":measure_second,"regularized_scale":scale}});
            bind_inline(
                builder,
                "lem-patching-properties",
                &channel,
                vec![
                    (
                        r"\sigma'_{\min,\rm patch}=\sigma_{\min}",
                        vec![
                            positive_identity(
                                "native_patch_minimum_at_zero_variance",
                                smin,
                                Standardizer::Global { sigma_min: smin }
                                    .apply(
                                        &vec![mu; k],
                                        &vec![true; k],
                                        &ObservationBatch::positions(TensorBatch::vectors(
                                            k,
                                            1,
                                            vec![0.; k],
                                        )?),
                                    )?
                                    .1
                                    .scale[0],
                            ),
                            upper(
                                "actual_native_regularized_scale_above_minimum",
                                smin,
                                stats.scale[0],
                            ),
                        ],
                    ),
                    (
                        r"(\sigma'_{\rm patch})'(V)=1/(2\sqrt{V+\sigma_{\min}^2})\leq1/(2\sigma_{\min})",
                        vec![
                            positive_identity(
                                "actual_regularized_scale_derivative_identity",
                                1. / (2. * stats.scale[0]),
                                1. / (2. * (var + smin * smin).sqrt()),
                            ),
                            upper(
                                "actual_regularized_scale_global_derivative_bound",
                                1. / (2. * stats.scale[0]),
                                1. / (2. * smin),
                            ),
                        ],
                    ),
                    (
                        r"V+\sigma_{\min}^2>0",
                        condition(
                            "actual_variance_plus_regularizer_strictly_positive",
                            var + smin * smin > 0.,
                        ),
                    ),
                ],
            )?;
            let measure_checks = vec![
                relative(
                    "actual_empirical_measure_total_mass",
                    weights.iter().sum(),
                    1.,
                    1.,
                ),
                relative(
                    "native_mean_matches_measure_first_moment",
                    stats.mean[0],
                    measure_mean,
                    xs.iter().map(|x| x.abs()).fold(0., f64::max),
                ),
                relative(
                    "actual_empirical_variance_and_second_moment",
                    var,
                    measure_second - measure_mean * measure_mean,
                    measure_second,
                ),
            ];
            builder.inline(
                "def-swarm-aggregation-operator",
                r"\mu_{\mathbf{v}} = M(\mathcal{S}, \mathbf{v}_{\mathcal{A}})",
                channel.clone(),
                measure_checks.clone(),
            )?;
            builder.inline(
                "def-standardization-operator",
                r"\mu_v = M(S, v_A)",
                channel.clone(),
                measure_checks,
            )?;
            builder.inline(
                "def-standardization-operator",
                r"\mu_A = \mathbb{E}[\mu_v]",
                channel.clone(),
                (0..k)
                    .map(|i| {
                        relative(
                            "native_mean_at_each_eligible_row",
                            stats.mean[i],
                            measure_mean,
                            xs.iter().map(|x| x.abs()).fold(0., f64::max),
                        )
                    })
                    .collect(),
            )?;
            builder.inline(
                "def-standardization-operator",
                r"\sigma'_A = \sigma'_{\text{patch}}(\text{Var}[\mu_v])",
                channel.clone(),
                (0..k)
                    .map(|i| {
                        positive_identity(
                            "native_scale_from_empirical_variance",
                            stats.scale[i],
                            scale,
                        )
                    })
                    .collect(),
            )?;
            builder.inline(
                "def-standardization-operator",
                r"z_i = (v_i - \mu_A) / \sigma'_A",
                channel.clone(),
                (0..n)
                    .map(|i| {
                        relative(
                            "native_masked_standardization_formula",
                            z[i],
                            if alive[i] {
                                (values[i] - mu) / scale
                            } else {
                                0.
                            },
                            ((values[i] - mu) / scale).abs(),
                        )
                    })
                    .collect(),
            )?;
            let (independent_z, _) =
                Standardizer::Global { sigma_min: smin }.apply(values, &alive, &obs)?;
            let quote = if name == "reward" {
                r"\mathbf{z}_r = z(S, \mathbf{r}, R_{agg})"
            } else {
                r"\mathbf{z}_d = z(S, \mathbf{d}, M_D)"
            };
            builder.inline(
                "def-fitness-potential-operator",
                quote,
                channel,
                z.iter()
                    .zip(&independent_z)
                    .map(|(&a, &b)| {
                        relative(
                            "actual_native_pipeline_channel_standardizer",
                            a,
                            b,
                            a.abs().max(b.abs()),
                        )
                    })
                    .collect(),
            )?;
        }
        for i in 0..k {
            let row = json!({"actual_pipeline":input,"row":i});
            builder.inline(
                "def-fitness-potential-operator",
                r"r'_i := g_A(z_{r,i}) + \eta",
                row.clone(),
                vec![positive_identity(
                    "native_positive_reward_channel_with_configured_floor",
                    rescaled_reward[i],
                    base.map(native.reward_z[i])? + eta,
                )],
            )?;
            builder.inline(
                "def-fitness-potential-operator",
                r"d'_i := g_A(z_{d,i}) + \eta",
                row.clone(),
                vec![positive_identity(
                    "native_positive_diversity_channel_with_configured_floor",
                    rescaled_distance[i],
                    base.map(native.diversity_z[i])? + eta,
                )],
            )?;
            builder.shown(
                "def-fitness-potential-operator",
                r"V_i :=",
                row,
                hyps.clone(),
                vec![positive_identity(
                    "actual_native_complete_fitness_product",
                    native.fitness[i],
                    rescaled_distance[i].powf(config.fitness.diversity_exponent)
                        * rescaled_reward[i].powf(config.fitness.reward_exponent),
                )],
            )?;
        }
        let mu_fit = mean(&native.fitness[..k]);
        let unfit = (0..k)
            .filter(|&i| native.fitness[i] <= mu_fit)
            .collect::<Vec<_>>();
        let fit = (0..k).filter(|i| !unfit.contains(i)).collect::<Vec<_>>();
        let population = json!({"actual_pipeline":input,"mean_alive_fitness":mu_fit,"unfit_members":unfit,"fit_members":fit});
        bind_inline(
            builder,
            "def-unfit-set",
            &population,
            vec![
                (
                    r"\mu=k^{-1}\sum_iV_i",
                    equal(
                        "actual_alive_fitness_average",
                        mu_fit,
                        native.fitness[..k].iter().sum::<f64>() / k as f64,
                    ),
                ),
                (
                    r"U_k=\{i\in\mathcal A_k:V_i\leq\mu\}",
                    condition(
                        "actual_unfit_membership_and_all_ties",
                        (0..k).all(|i| unfit.contains(&i) == (native.fitness[i] <= mu_fit)),
                    ),
                ),
                (
                    r"F_k=\mathcal A_k\setminus U_k",
                    vec![
                        hypothesis(
                            "actual_fit_set_is_alive_complement",
                            (0..k).all(|i| fit.contains(&i) != unfit.contains(&i)),
                        ),
                        relative(
                            "actual_unfit_fit_partition_count",
                            (fit.len() + unfit.len()) as f64,
                            k as f64,
                            k as f64,
                        ),
                    ],
                ),
            ],
        )?;
        let mut diversity_order = (0..k).collect::<Vec<_>>();
        diversity_order.sort_by(|&a, &b| rescaled_distance[a].total_cmp(&rescaled_distance[b]));
        let dh = diversity_order[..k / 2]
            .iter()
            .map(|&i| rescaled_distance[i])
            .collect::<Vec<_>>();
        let dl = diversity_order[k / 2..]
            .iter()
            .map(|&i| rescaled_distance[i])
            .collect::<Vec<_>>();
        let gap = mean(&dl) - mean(&dh);
        let floor_gap = 0.9 * gap;
        let log_gap = mean(&dl.iter().map(|x| x.ln()).collect::<Vec<_>>())
            - mean(&dh.iter().map(|x| x.ln()).collect::<Vec<_>>());
        let lower_log =
            crate::convergence_selection::log_gap_bounds(eta, maximum, floor_gap)?.lower;
        let diversity = json!({"actual_pipeline":input,"H":diversity_order[..k/2],"L":diversity_order[k/2..],"native_diversity_H":dh,"native_diversity_L":dl,"eta":eta,"M":maximum,"kappa_d":floor_gap,"computed":{"arithmetic_gap":gap,"D":log_gap,"mean_only_log_lower":lower_log},"coupling":"Product coupling of these sorted finite groups; every L value is at least every H value."});
        let ordered = dl.iter().all(|&l| dh.iter().all(|&h| l >= h));
        let d_hyps = vec![
            hypothesis(
                "actual_corrective_diversity_orientation",
                gap >= floor_gap && floor_gap > 0.,
            ),
            hypothesis(
                "all_native_rescaled_diversity_in_common_support",
                dh.iter().chain(&dl).all(|&x| x >= eta && x <= maximum),
            ),
            hypothesis("actual_product_coupling_ordered", ordered),
        ];
        bind_inline(
            builder,
            "prop-corrective-signal-bound",
            &diversity,
            vec![
                (
                    r"d'\in[\eta,M]",
                    condition(
                        "actual_native_diversity_support",
                        dh.iter().chain(&dl).all(|&x| x >= eta && x <= maximum),
                    ),
                ),
                (
                    r"0<\eta<M",
                    condition(
                        "positive_strict_diversity_support",
                        0. < eta && eta < maximum,
                    ),
                ),
                (
                    r"\mu_{d',L}-\mu_{d',H}\geq\kappa_{d'}\geq0",
                    vec![
                        upper(
                            "actual_native_mean_gap_exceeds_requested_floor",
                            floor_gap,
                            gap,
                        ),
                        upper("actual_nonnegative_diversity_gap_floor", 0., floor_gap),
                    ],
                ),
                (
                    r"d'_L\geq d'_H",
                    condition("all_product_coupling_diversity_pairs_ordered", ordered),
                ),
                (
                    r"D\geq\kappa_{d'}/M\geq\log(1+\kappa_{d'}/M)",
                    vec![
                        upper(
                            "actual_native_log_diversity_ordered_lower",
                            floor_gap / maximum,
                            log_gap,
                        ),
                        upper(
                            "linear_corrective_floor_exceeds_log_floor",
                            (floor_gap / maximum).ln_1p(),
                            floor_gap / maximum,
                        ),
                    ],
                ),
            ],
        )?;
        builder.shown(
            "prop-corrective-signal-bound",
            r"D\geq",
            diversity,
            d_hyps,
            ge(
                "actual_native_diversity_mean_only_log_lower",
                log_gap,
                lower_log,
            ),
        )?;
        let reward_h = &rescaled_reward[..k / 2];
        let reward_l = &rescaled_reward[k / 2..k];
        let raw_h = &raw[..k / 2];
        let raw_l = &raw[k / 2..k];
        let actual_cross = raw_h
            .iter()
            .flat_map(|a| raw_l.iter().map(move |b| (a - b).abs()))
            .fold(0., f64::max);
        let diameter = (0..k)
            .flat_map(|i| {
                let positions = &positions;
                let velocities = &velocities;
                (0..k).map(move |j| {
                    ((positions[i] - positions[j]).powi(2)
                        + (velocities[i] - velocities[j]).powi(2))
                    .sqrt()
                })
            })
            .fold(0., f64::max);
        let bx = positions[..k].iter().map(|x| x.abs()).fold(0., f64::max);
        let bv = velocities[..k].iter().map(|x| x.abs()).fold(0., f64::max);
        let reward_lipschitz = (bx.powi(2) + (2. * velocity_regularizer * bv).powi(2)).sqrt();
        let br = reward_lipschitz * diameter;
        let lg = logistic_derivative(0.);
        let kr = lg * br / smin;
        let reward_log_gap = (mean(&reward_h.iter().map(|x| x.ln()).collect::<Vec<_>>())
            - mean(&reward_l.iter().map(|x| x.ln()).collect::<Vec<_>>()))
        .abs();
        let cross_mean = raw_h
            .iter()
            .flat_map(|a| raw_l.iter().map(move |b| a - b))
            .sum::<f64>()
            / (raw_h.len() * raw_l.len()) as f64;
        let reward_input = json!({"actual_pipeline":input,"H":(0..k/2).collect::<Vec<_>>(),"L":(k/2..k).collect::<Vec<_>>(),"complete_reward_gradient_envelope":{"position_bound":bx,"velocity_bound":bv,"velocity_regularizer":velocity_regularizer,"L_R":reward_lipschitz,"convex_phase_box":"The quadratic reward gradient is (−x,−2 c_v v) throughout this box."},"phase_diameter":diameter,"actual_maximum_cross_reward_gap":actual_cross,"B_r":br,"L_g":lg,"sigma_min":smin,"K_r":kr,"eta":eta,"M":maximum,"computed":{"absolute_log_reward_gap":reward_log_gap,"absolute_raw_mean_gap":(mean(raw_h)-mean(raw_l)).abs(),"product_pair_mean_gap":cross_mean}});
        bind_inline(
            builder,
            "prop-adversarial-signal-bound-naive",
            &reward_input,
            vec![
                (
                    r"r'\in[\eta,M]",
                    condition(
                        "actual_complete_native_reward_channel_support",
                        reward_h
                            .iter()
                            .chain(reward_l)
                            .all(|&x| x >= eta && x <= maximum),
                    ),
                ),
                (
                    r"0<\eta\le M",
                    condition(
                        "positive_reward_support_endpoints",
                        eta > 0. && eta <= maximum,
                    ),
                ),
                (
                    r"|\mathbb E_H\log r'-\mathbb E_L\log r'|\leq\log(M/\eta)",
                    le(
                        "actual_native_global_reward_log_bound",
                        reward_log_gap,
                        (maximum / eta).ln(),
                    ),
                ),
            ],
        )?;
        bind_inline(
            builder,
            "prop-raw-reward-mean-gap-bound",
            &reward_input,
            vec![
                (
                    r"|r_i-r_j|\leq B_r",
                    le(
                        "actual_complete_raw_reward_cross_pair_envelope",
                        actual_cross,
                        br,
                    ),
                ),
                (
                    r"|\mu_{r,H}-\mu_{r,L}|\leq B_r",
                    vec![
                        relative(
                            "raw_mean_difference_is_product_pair_average",
                            mean(raw_h) - mean(raw_l),
                            cross_mean,
                            actual_cross,
                        ),
                        upper(
                            "actual_raw_mean_gap_with_complete_velocity_term",
                            (mean(raw_h) - mean(raw_l)).abs(),
                            br,
                        ),
                    ],
                ),
                (
                    r"B_r=L_RD",
                    vec![
                        positive_identity(
                            "complete_phase_reward_lipschitz_diameter_constant",
                            br,
                            reward_lipschitz * diameter,
                        ),
                        upper(
                            "all_actual_complete_raw_cross_pairs_bounded",
                            actual_cross,
                            br,
                        ),
                    ],
                ),
            ],
        )?;
        // Repeated physical measurements have identical cross-pair bounds;
        // retain every distinct pair of values rather than duplicate fixtures.
        for i in (0..k / 2).take(4) {
            for j in (k / 2..k).take(4) {
                let phase_distance = ((positions[i] - positions[j]).powi(2)
                    + (velocities[i] - velocities[j]).powi(2))
                .sqrt();
                let pair = json!({"actual_reward_model":reward_input,"H_row":i,"L_row":j,"phase_pair_distance":phase_distance});
                builder.inline(
                    "prop-raw-reward-mean-gap-bound",
                    r"|r_i-r_j|\leq L_R\|z_i-z_j\|",
                    pair.clone(),
                    le(
                        "complete_native_quadratic_phase_reward_lipschitz",
                        (raw[i] - raw[j]).abs(),
                        reward_lipschitz * phase_distance,
                    ),
                )?;
                builder.inline(
                    "prop-log-reward-gap-axiom-bound",
                    r"|z_{r,i}-z_{r,j}|\leq B_r/\sigma_{\min}",
                    pair.clone(),
                    vec![
                        relative(
                            "actual_shared_native_reward_center_cancels",
                            native.reward_z[i] - native.reward_z[j],
                            (native.oriented_reward[i] - native.oriented_reward[j])
                                / native.reward_stats.scale[i],
                            (native.reward_z[i] - native.reward_z[j]).abs(),
                        ),
                        upper(
                            "actual_standardized_cross_reward_bound",
                            (native.reward_z[i] - native.reward_z[j]).abs(),
                            br / smin,
                        ),
                    ],
                )?;
                builder.inline(
                    "prop-log-reward-gap-axiom-bound",
                    r"|r'_i-r'_j|\leq K_r",
                    pair,
                    le(
                        "actual_positive_native_cross_reward_lipschitz",
                        (rescaled_reward[i] - rescaled_reward[j]).abs(),
                        kr,
                    ),
                )?;
            }
        }
        bind_inline(
            builder,
            "prop-log-reward-gap-axiom-bound",
            &reward_input,
            vec![
                (
                    r"\sigma_{\min}>0",
                    condition("native_reward_regularizer_positive", smin > 0.),
                ),
                (
                    r"r'\ge\eta>0",
                    vec![
                        hypothesis(
                            "actual_reward_channel_above_configured_positive_floor",
                            rescaled_reward[..k].iter().all(|&x| x >= eta),
                        ),
                        hypothesis("configured_reward_floor_positive", eta > 0.),
                    ],
                ),
                (
                    r"K_r=L_gB_r/\sigma_{\min}",
                    vec![
                        positive_identity(
                            "actual_fixed_reward_cross_rescale_constant",
                            kr,
                            lg * br / smin,
                        ),
                        upper(
                            "native_logistic_global_derivative_bound",
                            native.reward_z[..k]
                                .iter()
                                .map(|&z| logistic_derivative(z))
                                .fold(0., f64::max),
                            lg,
                        ),
                    ],
                ),
            ],
        )?;
        builder.shown(
            "prop-log-reward-gap-axiom-bound",
            r"|\mathbb E_H\log",
            reward_input,
            hyps,
            le(
                "actual_complete_native_reward_log_gap_from_product_coupling",
                reward_log_gap,
                (kr / eta).ln_1p(),
            ),
        )?;
    }
    let a = 0.1_f64;
    let b = 2.1_f64;
    let fluctuating = [a, b];
    let constant = [(a + b) / 2.; 2];
    let zero_mean_gap = (mean(&fluctuating) - mean(&constant)).abs();
    let unequal_log_gap = (mean(&fluctuating.map(f64::ln)) - mean(&constant.map(f64::ln))).abs();
    let upper_zero = crate::convergence_selection::log_gap_bounds(a, b, 0.)?.upper;
    let input = json!({"support":[a,b],"fluctuating_population":fluctuating,"constant_population":constant,"kappa":zero_mean_gap,"unequal_log_averages":unequal_log_gap,"mean_only_upper_at_zero":upper_zero});
    builder.inline(
        "rem-log-gap-bound-tight-at-vmin",
        r"\kappa=0",
        input,
        vec![
            relative("actual_equal_means", zero_mean_gap, 0., b - a),
            hypothesis(
                "equal_means_retain_positive_log_gap",
                unequal_log_gap > 0. && upper_zero > 0.,
            ),
            upper(
                "actual_equal_mean_unequal_log_gap_within_mean_only_bound",
                unequal_log_gap,
                upper_zero,
            ),
        ],
    )?;
    builder.inline("rem-log-gap-bound-tight-at-vmin",r"K=0",json!({"identical_coupled_values":fluctuating,"actual_mean_absolute_coupling_gap":0.,"actual_log_coupling_gap":0.}),vec![relative("actual_identical_product_point_pair_gap",fluctuating.iter().zip(&fluctuating).map(|(x,y)|(x-y).abs()).sum::<f64>()/2.,0.,b-a),relative("actual_zero_coupling_log_bound",0.,(0_f64/a).ln_1p(),1.)])?;
    builder.inline(
        "rem-log-gap-bound-tight-at-vmin",
        r"\log(b/a)",
        json!({"first_constant_population":[b],"second_constant_population":[a],"support":[a,b]}),
        vec![positive_identity(
            "global_log_support_bound_attained_at_opposite_endpoints",
            b.ln() - a.ln(),
            (b / a).ln(),
        )],
    )?;
    Ok(())
}

async fn native_cloning_outputs(builder: &mut Builder) -> Result<()> {
    let label = "def-key-operator-outputs";
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    for d in [1_usize, 2, 4] {
        let config = GasConfig::euclidean(d, 0.04)?;
        let x = (0..2)
            .flat_map(|i| {
                (0..d).map(move |a| {
                    if i == 0 {
                        0.1 + 0.03 * a as f64
                    } else {
                        0.8 - 0.02 * a as f64
                    }
                })
            })
            .collect::<Vec<_>>();
        let v = (0..2)
            .flat_map(|i| {
                (0..d).map(move |a| {
                    if i == 0 {
                        0.2 + 0.02 * a as f64
                    } else {
                        -0.3 + 0.01 * a as f64
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut obs = ObservationBatch::positions(TensorBatch::vectors(2, d, x.clone())?);
        obs.fields
            .insert("velocities".into(), TensorBatch::vectors(2, d, v.clone())?);
        let mut before = Population::new(obs)?;
        before.rewards.raw = (0..2)
            .map(|i| {
                Ok(-Benchmark::Quadratic.value(&x[i * d..(i + 1) * d])?
                    - 0.3 * v[i * d..(i + 1) * d].iter().map(|a| a * a).sum::<f64>())
            })
            .collect::<Result<Vec<_>>>()?;
        let pool = DonorPool::freeze(&before, 0, &[], 0, false)?;
        let companions = CompanionBatch {
            rows: 2,
            count: 1,
            indices: vec![1, 0],
            valid: vec![true; 2],
            mutual: false,
        };
        let distances = (0..2)
            .map(|i| {
                config.distance_donors.distance.compare(
                    &before.observations,
                    i,
                    &before.observations,
                    1 - i,
                )
            })
            .collect::<Result<Vec<f64>>>()?;
        let fitness = config.fitness.evaluate(
            &before.rewards,
            &distances,
            &[true; 2],
            &before.observations,
            before.version,
        )?;
        let probabilities = (0..2)
            .map(|i| {
                config.clone_decision.acceptance_probability(
                    0,
                    fitness.fitness[i],
                    fitness.fitness[1 - i],
                )
            })
            .collect::<Vec<_>>();
        let jitter = config
            .clone_transform
            .jitter
            .as_ref()
            .ok_or_else(|| error("native Gaussian cloning jitter missing"))?;
        let gaussian_identity = jitter.innovation == InnovationLaw::Gaussian
            && matches!(&jitter.geometry,NoiseGeometry::Isotropic{scale:FactorValues::Constant{values}} if values.as_slice()==[1.]);
        let samples = 4096_usize;
        let gaussian = jitter
            .sample(
                &before.observations,
                NoiseRequest {
                    rows: samples,
                    dimension: d,
                    seed: 20261002 + d as u64,
                    step: 0,
                    stream: Stream::CloneNoise,
                    substep: 0,
                },
                &mut cx,
            )
            .await?;
        let innovations = gaussian.values();
        let count = innovations.len() as f64;
        let sample_mean = mean(innovations);
        let sample_var = variance(innovations);
        let frequency = 0.6_f64;
        let cosine = innovations
            .iter()
            .map(|&z| (frequency * z).cos())
            .sum::<f64>()
            / count;
        let sine = innovations
            .iter()
            .map(|&z| (frequency * z).sin())
            .sum::<f64>()
            / count;
        let expected_cosine = (-frequency * frequency / 2.).exp();
        let cosine_variance =
            (1. + (-2. * frequency * frequency).exp()) / 2. - (-frequency * frequency).exp();
        let sine_variance = (1. - (-2. * frequency * frequency).exp()) / 2.;
        builder.inline(label,r"\zeta_i^x \sim \mathcal{N}(0, I_d)",json!({"native_noise_configuration":jitter,"dimension":d,"replicas":samples,"actual_native_innovations":gaussian,"computed":{"mean":sample_mean,"population_variance":sample_var,"cosine_characteristic_function":cosine,"sine_characteristic_function":sine},"statistical_comparison":"Independent addressed native samples; each moment/characteristic-function comparison uses six analytic standard errors."}),vec![hypothesis("actual_native_standard_gaussian_isotropic_configuration",gaussian_identity),upper("native_gaussian_mean_six_standard_errors",sample_mean.abs()*count.sqrt(),6.),upper("native_gaussian_variance_six_standard_errors",(sample_var-(1.-1./count)).abs()/(2./count).sqrt(),6.),upper("native_gaussian_cosine_characteristic_function_six_standard_errors",(cosine-expected_cosine).abs()/(cosine_variance/count).sqrt(),6.),upper("native_gaussian_sine_characteristic_function_six_standard_errors",sine.abs()/(sine_variance/count).sqrt(),6.)])?;
        for seed in 1..=12_u64 {
            let plan = config.clone_decision.plan(
                &before,
                &pool,
                &companions,
                &fitness.fitness,
                &fitness.fitness,
                &[true; 2],
                seed,
                0,
            )?;
            let mut after = plan.apply_literal(&before, &pool)?;
            let eta = jitter
                .sample(
                    &after.observations,
                    NoiseRequest {
                        rows: 2,
                        dimension: d,
                        seed,
                        step: 0,
                        stream: Stream::CloneNoise,
                        substep: 0,
                    },
                    &mut cx,
                )
                .await?;
            config
                .clone_transform
                .apply(
                    &before,
                    &mut after,
                    &pool,
                    &plan,
                    &companions,
                    seed,
                    0,
                    &mut cx,
                )
                .await?;
            let components =
                algorithmic_gas::cloning::accepted_current_components(&before, &pool, &plan)?;
            let mut predicted_velocity = v.clone();
            let alpha = config
                .clone_transform
                .restitution
                .ok_or_else(|| error("native component restitution missing"))?;
            let mut rotations = Vec::new();
            for members in &components {
                let rotation =
                    RandomStream::new(seed, 0, Stream::CollisionRotation, members[0] as u64, 0)
                        .haar_orthogonal(d)?;
                let center = (0..d)
                    .map(|a| {
                        members.iter().map(|&i| v[i * d + a]).sum::<f64>() / members.len() as f64
                    })
                    .collect::<Vec<_>>();
                for &i in members {
                    for a in 0..d {
                        predicted_velocity[i * d + a] = center[a]
                            + alpha
                                * (0..d)
                                    .map(|b| rotation[a * d + b] * (v[i * d + b] - center[b]))
                                    .sum::<f64>();
                    }
                }
                rotations.push(
                    json!({"members":members,"rotation":rotation,"frozen_component_center":center}),
                );
            }
            let input = json!({"native_preset":config,"dimension":d,"seed":seed,"step":0,"before":before,"native_measurement_companions":companions,"native_fitness":fitness,"native_conditional_acceptance_probabilities":probabilities,"actual_native_plan":plan,"actual_native_position_innovations":eta,"actual_accepted_components":components,"independently_replayed_Haar_component_rotations":rotations,"independently_predicted_velocity":predicted_velocity,"actual_native_output":after,"conditional_law_scope":"N=2, both alive, current-frame pool, one nonself companion in both channels. Every companion proposal has a unique support atom, so total conditional probabilities given S equal native acceptance probabilities, with no omitted measurement or donor averaging."});
            let hyps = vec![
                hypothesis(
                    "actual_unique_nonself_companion_support",
                    pool.sources.len() == 2
                        && companions.indices == [1, 0]
                        && companions.valid.iter().all(|&b| b),
                ),
                hypothesis(
                    "native_input_and_output_finite",
                    before
                        .observations
                        .fields
                        .values()
                        .chain(after.observations.fields.values())
                        .all(|f| f.values().iter().all(|x| x.is_finite())),
                ),
                hypothesis("actual_native_gaussian_cloning_law", gaussian_identity),
            ];
            let mut transition_checks = vec![
                hypothesis(
                    "native_literal_copy_uses_frozen_plan_sources",
                    plan.sources == pool.sources,
                ),
                hypothesis(
                    "native_output_population_size_preserved",
                    before.len() == after.len(),
                ),
            ];
            for (i, &probability) in probabilities.iter().enumerate() {
                let mut gate = RandomStream::new(seed, 0, Stream::Accept, i as u64, 0);
                let draw = gate.uniform::<f64>();
                transition_checks.push(hypothesis(
                    "actual_native_acceptance_gate_replays_conditional_branch",
                    plan.choices[i].accepted == (draw < probability),
                ));
            }
            builder.add(&[label],r"S' \sim \Psi_{\text{clone}}(S, \cdot)".into(),input.clone(),hyps.clone(),transition_checks,"Actual native frozen-current-population cloning proposal with unique nonself companion support at N=2. Addressed acceptance, Gaussian position jitter, and accepted-component Haar restitution are retained explicitly. This is the cloning proposal before terminal kinetic and boundary stages.")?;
            for i in 0..2 {
                let row = json!({"native_transition":input,"row_for_evaluation":i});
                let native_probability = plan.choices[i]
                    .probability
                    .ok_or_else(|| error("native acceptance probability missing"))?;
                let independently_clipped = ((fitness.fitness[1 - i] - fitness.fitness[i])
                    / (fitness.fitness[i] + config.clone_decision.epsilon)
                    / config.clone_decision.saturation)
                    .clamp(0., 1.);
                builder.shown(
                    label,
                    r"p_i =",
                    row.clone(),
                    hyps.clone(),
                    vec![
                        relative(
                            "native_total_conditional_cloning_probability_unique_support",
                            native_probability,
                            independently_clipped,
                            native_probability.abs().max(independently_clipped.abs()),
                        ),
                        upper(
                            "native_conditional_probability_nonnegative",
                            0.,
                            native_probability,
                        ),
                        upper(
                            "native_conditional_probability_at_most_one",
                            native_probability,
                            1.,
                        ),
                    ],
                )?;
                let own_x = before.observations.field("positions")?.row(i)?;
                let out_x = after.observations.field("positions")?.row(i)?;
                let own_v = before.observations.field("velocities")?.row(i)?;
                let out_v = after.observations.field("velocities")?.row(i)?;
                let delta_x = out_x
                    .iter()
                    .zip(own_x)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let delta_v = out_v
                    .iter()
                    .zip(own_v)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let donor_x = before.observations.field("positions")?.row(1 - i)?;
                let position_prediction = (0..d)
                    .map(|a| {
                        if plan.choices[i].accepted {
                            donor_x[a] - own_x[a]
                                + config.clone_transform.jitter_amplitude * eta.row(i).unwrap()[a]
                        } else {
                            0.
                        }
                    })
                    .collect::<Vec<_>>();
                let detail = json!({"native_row":row,"measured_position_displacement":delta_x,"independently_predicted_position_displacement":position_prediction,"measured_velocity_displacement":delta_v});
                let displacement_checks = (0..d)
                    .map(|a| {
                        relative(
                            "actual_native_position_output_minus_input",
                            delta_x[a],
                            position_prediction[a],
                            own_x[a]
                                .abs()
                                .max(out_x[a].abs())
                                .max(position_prediction[a].abs()),
                        )
                    })
                    .collect::<Vec<_>>();
                builder.shown(
                    label,
                    r"\Delta x_i :=",
                    detail.clone(),
                    hyps.clone(),
                    displacement_checks.clone(),
                )?;
                if plan.choices[i].accepted {
                    builder.inline(
                        label,
                        r"\Delta x_i = x_{c_i} - x_i + \sigma_x \zeta_i^x",
                        detail.clone(),
                        displacement_checks,
                    )?;
                }
                builder.shown(label,r"\Delta v_i :=",detail,hyps.clone(),(0..d).map(|a|relative("actual_native_velocity_displacement_from_independent_component_restitution",delta_v[a],predicted_velocity[i*d+a]-own_v[a],own_v[a].abs().max(out_v[a].abs()))).collect())?;
            }
            let ox = after.observations.field("positions")?.values();
            let ov = after.observations.field("velocities")?.values();
            let xcenter = (0..d).map(|a| (x[a] + x[d + a]) / 2.).collect::<Vec<_>>();
            let vcenter = (0..d).map(|a| (v[a] + v[d + a]) / 2.).collect::<Vec<_>>();
            let oxcenter = (0..d).map(|a| (ox[a] + ox[d + a]) / 2.).collect::<Vec<_>>();
            let ovcenter = (0..d).map(|a| (ov[a] + ov[d + a]) / 2.).collect::<Vec<_>>();
            let coupling_cost = |swap: bool| {
                (0..2)
                    .flat_map(|i| {
                        let xcenter = &xcenter;
                        let vcenter = &vcenter;
                        let oxcenter = &oxcenter;
                        let ovcenter = &ovcenter;
                        let x = &x;
                        let v = &v;
                        (0..d).map(move |a| {
                            let j = if swap { 1 - i } else { i };
                            ((x[i * d + a] - xcenter[a]) - (ox[j * d + a] - oxcenter[a])).powi(2)
                                + 0.7
                                    * ((v[i * d + a] - vcenter[a]) - (ov[j * d + a] - ovcenter[a]))
                                        .powi(2)
                        })
                    })
                    .sum::<f64>()
                    / 2.
            };
            let identity_cost = coupling_cost(false);
            let swapped_cost = coupling_cost(true);
            let swap = swapped_cost < identity_cost;
            let mut centered_checks = vec![
                hypothesis("retained_phase_quadratic_positive", 0.7_f64 > 0.),
                upper(
                    "exhaustive_two_atom_full_phase_optimal_assignment",
                    coupling_cost(swap),
                    identity_cost.min(swapped_cost),
                ),
            ];
            let mut centered = Vec::new();
            for i in 0..2 {
                let j = if swap { 1 - i } else { i };
                for a in 0..d {
                    let difference = (x[i * d + a] - xcenter[a]) - (ox[j * d + a] - oxcenter[a]);
                    let independently_centered =
                        (x[i * d + a] - ox[j * d + a]) - (xcenter[a] - oxcenter[a]);
                    centered.push(difference);
                    centered_checks.push(relative(
                        "centered_difference_under_optimal_full_phase_permutation",
                        difference,
                        independently_centered,
                        x[i * d + a].abs().max(ox[j * d + a].abs()),
                    ));
                }
            }
            builder.shown(label,r"\Delta\delta_{x,i} :=",json!({"native_transition":input,"centers":{"x_first":xcenter,"x_second":oxcenter,"v_first":vcenter,"v_second":ovcenter},"full_positive_phase_quadratic":{"position_coefficient":1.,"velocity_coefficient":0.7},"both_permutation_costs":[identity_cost,swapped_cost],"minimizing_permutation":if swap{vec![1,0]}else{vec![0,1]},"actual_centered_position_differences":centered,"index_scope":"A representative of the optimal permutation coupling; these row indices carry no intrinsic particle identity."}),hyps,centered_checks)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_positive_cloning_constants_cannot_be_replaced_by_zero() {
        assert!(!positive_identity("tiny_positive", 0., 1e-30).passed);
        assert!(!relative("tiny_gap", 0., 1e-30, 1e-30).passed);
        assert!(positive_identity("tiny_positive", 1e-30, 1e-30).passed);
    }
}
