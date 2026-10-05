//! Independent native experiments for the quantitative chapter 3 inequalities.
//! Predictions are computed before each proposal from its retained input and
//! actual measurement marks. Conditional residuals are averaged across fresh
//! engines, rather than treating interacting rows as independent experiments.
use crate::{
    Benchmark, BenchmarkModel,
    convergence_cloning::{analyze_cloning_step, canonical_cloning_constants},
    convergence_lyapunov::uniform_transport,
};
use algorithmic_gas::{
    GasBuilder, GasConfig, GasError, ObservationBatch, Population, Result, TensorBatch,
    tracking::StageSnapshot,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmpiricalComparison {
    pub id: String,
    pub source_labels: Vec<String>,
    pub relation: String,
    pub samples: usize,
    pub observed_mean: f64,
    pub prediction_mean: f64,
    /// Observed minus prediction; nonpositive for upper bounds.
    pub residual_mean: f64,
    pub residual_standard_error: f64,
    pub six_standard_error_interval: [f64; 2],
    pub maximum_absolute_residual: f64,
    pub not_rejected: bool,
    /// A one-sided diagnostic only, not a simultaneous confidence theorem.
    pub upper_bound_supported: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmpiricalCase {
    pub name: String,
    pub native_config: GasConfig,
    pub benchmark: Benchmark,
    pub walkers: usize,
    pub dimensions: usize,
    pub alive: usize,
    pub initial_positions: Vec<Vec<f64>>,
    pub initial_velocities: Vec<Vec<f64>>,
    pub entering_alive: Vec<bool>,
    pub constants: Value,
    pub comparisons: Vec<EmpiricalComparison>,
    pub replicas: Vec<Value>,
    pub scope: String,
}
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn center(x: &[Vec<f64>]) -> Vec<f64> {
    (0..x[0].len())
        .map(|a| x.iter().map(|p| p[a]).sum::<f64>() / x.len() as f64)
        .collect()
}
fn norm(x: &[f64]) -> f64 {
    x.iter().map(|a| a * a).sum()
}
fn distance(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b).powi(2)).sum()
}
fn variance(x: &[Vec<f64>]) -> f64 {
    let m = center(x);
    x.iter().map(|p| distance(p, &m)).sum::<f64>() / x.len() as f64
}
fn stage_points(s: &StageSnapshot, field: &str, d: usize) -> Result<Vec<Vec<f64>>> {
    let f = s
        .fields
        .get(field)
        .ok_or_else(|| GasError::Configuration(format!("missing proposal field {field}")))?;
    Ok(f.values.chunks(d).map(<[f64]>::to_vec).collect())
}
fn comparison(
    id: &str,
    labels: &[&str],
    relation: &str,
    values: &[(f64, f64)],
) -> EmpiricalComparison {
    let residuals: Vec<_> = values.iter().map(|(a, b)| a - b).collect();
    let average = mean(&residuals);
    let se = (residuals.iter().map(|r| (r - average).powi(2)).sum::<f64>()
        / ((values.len() - 1) * values.len()) as f64)
        .sqrt();
    let scale = 1.
        + values
            .iter()
            .map(|(a, b)| a.abs().max(b.abs()))
            .fold(0_f64, f64::max);
    let tolerance = 2e-10 * scale;
    let interval = [average - 6. * se, average + 6. * se];
    let not_rejected = if relation == "equal" {
        average.abs() <= 6. * se + tolerance
    } else {
        interval[0] <= tolerance
    };
    EmpiricalComparison {
        id: id.into(),
        source_labels: labels.iter().map(|s| (*s).into()).collect(),
        relation: relation.into(),
        samples: values.len(),
        observed_mean: mean(&values.iter().map(|v| v.0).collect::<Vec<_>>()),
        prediction_mean: mean(&values.iter().map(|v| v.1).collect::<Vec<_>>()),
        residual_mean: average,
        residual_standard_error: se,
        six_standard_error_interval: interval,
        maximum_absolute_residual: residuals.iter().map(|r| r.abs()).fold(0_f64, f64::max),
        not_rejected,
        upper_bound_supported: relation == "upper" && interval[1] <= tolerance,
    }
}
fn error(s: &str) -> GasError {
    GasError::Configuration(s.into())
}

/// The complete matrix uses N=4/16/64, d=1/2, four landscapes and four
/// proposal profiles. Each independent seed starts from the same frozen state.
pub async fn validate_empirical(samples: usize, compact: bool) -> Result<Value> {
    if !(64..=100_000).contains(&samples) {
        return Err(error(
            "chapter 3 empirical comparisons require 64..100000 independent seeds",
        ));
    }
    let populations: &[usize] = if compact { &[4] } else { &[4, 16, 64] };
    let dimensions: &[usize] = if compact { &[1] } else { &[1, 2] };
    let landscapes: &[Benchmark] = if compact {
        &[Benchmark::Quadratic]
    } else {
        &[
            Benchmark::Quadratic,
            Benchmark::Sphere,
            Benchmark::Rastrigin,
            Benchmark::Constant,
        ]
    };
    let profiles: &[&str] = if compact {
        &["canonical", "revival", "singleton"]
    } else {
        &["canonical", "selection", "revival", "singleton"]
    };
    let mut cases = vec![];
    for &n in populations {
        for &d in dimensions {
            for &benchmark in landscapes {
                for &profile in profiles {
                    let mut config = GasConfig::euclidean(d, 0.04)?;
                    if profile == "selection" {
                        config.fitness.reward_exponent = 2.;
                        config.fitness.diversity_exponent = 0.5;
                        config.clone_decision.saturation = 0.5;
                        config.clone_transform.restitution = Some(0.5);
                        config.clone_transform.jitter_amplitude = 0.2;
                    }
                    let k = match profile {
                        "revival" => n / 2,
                        "singleton" => 1,
                        _ => n,
                    };
                    let alive: Vec<_> = (0..n).map(|i| i < k).collect();
                    let x: Vec<Vec<f64>> = (0..n)
                        .map(|i| {
                            (0..d)
                                .map(|a| {
                                    if alive[i] {
                                        -1.5 + 3. * ((i + a * 3) % n) as f64 / n as f64
                                    } else {
                                        if i % 2 == 0 { 1e9 } else { -1e9 }
                                    }
                                })
                                .collect()
                        })
                        .collect();
                    let v: Vec<Vec<f64>> = (0..n)
                        .map(|i| {
                            (0..d)
                                .map(|a| 0.3 * ((i + a + 1) as f64).sin() / (d as f64).sqrt())
                                .collect()
                        })
                        .collect();
                    cases.push(
                        native_case(
                            samples,
                            benchmark,
                            profile,
                            config,
                            x,
                            v,
                            alive,
                            cases.len() as u64,
                        )
                        .await?,
                    );
                }
            }
        }
    }
    let mut pressure_cases = vec![];
    for &n in populations {
        for &d in dimensions {
            pressure_cases
                .push(balanced_pressure_case(samples, n, d, pressure_cases.len() as u64).await?);
        }
    }
    let comparisons = cases.iter().map(|c| c.comparisons.len()).sum::<usize>()
        + pressure_cases
            .iter()
            .map(|c| c["comparisons"].as_array().unwrap().len())
            .sum::<usize>();
    let failed = cases
        .iter()
        .flat_map(|c| &c.comparisons)
        .filter(|c| !c.not_rejected)
        .count()
        + pressure_cases
            .iter()
            .flat_map(|c| c["comparisons"].as_array().unwrap())
            .filter(|c| c["not_rejected"] != true)
            .count();
    Ok(
        json!({"schema_version":1,"samples_per_case":samples,"cases":cases,"balanced_keystone_cases":pressure_cases,"summary":{"cases":cases.len()+pressure_cases.len(),"native_proposals":samples*(2*cases.len()+2*pressure_cases.len()),"comparisons":comparisons,"comparisons_violated":failed},"scope":"Measured conditional residuals for chapter 3 proposal inequalities. Each replicate is a fresh native engine with one complete update recorded, and proposal quantities are read before kinetics. Population normalization, dead-coordinate reset, actual sampled fitness and common component rotations are retained. Six-SE diagnostics quantify compatibility; exact source-formula coverage is checked by gas-estimates. These proposal experiments do not claim to validate external stationary-law or survivor-conditioned QSD theorems."}),
    )
}

async fn proposal(
    config: &GasConfig,
    benchmark: Benchmark,
    x: &[Vec<f64>],
    v: &[Vec<f64>],
    alive: &[bool],
    seed: u64,
) -> Result<algorithmic_gas::RunArchive<f64>> {
    let n = x.len();
    let d = x[0].len();
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, d, x.concat())?);
    obs.fields
        .insert("velocities".into(), TensorBatch::vectors(n, d, v.concat())?);
    let mut population = Population::new(obs)?;
    for (mark, &a) in population.validity.iter_mut().zip(alive) {
        mark.terminated = !a;
    }
    let mut actual = config.clone();
    actual.seed = seed;
    let model = BenchmarkModel {
        benchmark,
        field: "positions".into(),
        direction: actual.fitness.direction,
    };
    let mut gas = GasBuilder::new(population, model.clone())
        .gradient(model)
        .config(actual)
        .build()
        .await?;
    gas.start_recording(Default::default())?;
    gas.step().await?;
    Ok(gas.recording().unwrap().clone())
}

#[allow(clippy::too_many_arguments)]
async fn native_case(
    samples: usize,
    benchmark: Benchmark,
    profile: &str,
    config: GasConfig,
    x: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    alive: Vec<bool>,
    case: u64,
) -> Result<EmpiricalCase> {
    let n = x.len();
    let d = x[0].len();
    let k = alive.iter().filter(|a| **a).count();
    let nf = n as f64;
    let xa: Vec<_> = x
        .iter()
        .zip(&alive)
        .filter_map(|(x, a)| a.then_some(x.clone()))
        .collect();
    let va: Vec<_> = v
        .iter()
        .zip(&alive)
        .filter_map(|(v, a)| a.then_some(v.clone()))
        .collect();
    let diam2 = xa
        .iter()
        .flat_map(|a| xa.iter().map(move |b| distance(a, b)))
        .fold(0_f64, f64::max);
    let radius2 = xa.iter().map(|x| norm(x)).fold(0_f64, f64::max);
    let sigma = config.clone_transform.jitter_amplitude;
    let alpha = config
        .clone_transform
        .restitution
        .ok_or_else(|| error("native component collision missing"))?;
    let vmax = config
        .kinetic
        .velocity_cap
        .ok_or_else(|| error("native completed cap missing"))?;
    let initialx = variance(&xa) * k as f64 / nf;
    let initialv = variance(&va) * k as f64 / nf;
    let initialb = xa.iter().map(|x| norm(x)).sum::<f64>() / nf;
    let reset = diam2 / 2. + (1. - 1. / nf) * d as f64 * sigma * sigma;
    let mx = radius2 + d as f64 * sigma * sigma;
    let conditional_bary_bound = (diam2 + d as f64 * sigma * sigma) / nf;
    let lambda = 1_f64;
    let b = 0.5_f64;
    let cv = 0.1_f64;
    let cb = 0.05_f64;
    let mh =
        (1. + b.abs() / 2.) * mx + (lambda + b.abs() / 2.) * ((1. + 2. * alpha) * vmax).powi(2);
    let cw = 4. * mh;
    let cvel = if k == n { 0. } else { 8. * vmax * vmax };
    let mut pairs = vec![vec![]; 11];
    let mut replicas = vec![];
    for replicate in 0..samples {
        let seed = 8_000_000 + case * 1_000_000 + replicate as u64;
        let archive = proposal(&config, benchmark, &x, &v, &alive, seed).await?;
        let diagnostic = analyze_cloning_step(&archive, 0)?;
        if diagnostic.checks.iter().any(|c| !c.passed) {
            return Err(error("native exact cloning diagnostic failed"));
        }
        let moments = diagnostic
            .moments
            .ok_or_else(|| error("conditional native proposal moments missing"))?;
        let step = &archive.steps[0];
        let output = step
            .stages
            .iter()
            .find(|s| s.stage == "post_transform")
            .ok_or_else(|| error("native post-transform stage missing"))?;
        let ox = stage_points(output, "positions", d)?;
        let ov = stage_points(output, "velocities", d)?;
        let measuredx = variance(&ox);
        let measuredv = variance(&ov);
        let measuredb = ox.iter().map(|x| norm(x)).sum::<f64>() / nf;
        let expectedb = moments
            .output_means
            .iter()
            .zip(&moments.covariance_traces)
            .map(|(m, c)| norm(m) + c)
            .sum::<f64>()
            / nf;
        let mean_center = center(&moments.output_means);
        let measuredcenter = distance(&center(&ox), &mean_center);
        let choice: &[algorithmic_gas::cloning::CloneChoice] = &step.report.clone_plan.choices;
        let mut components = vec![vec![]; n];
        let mut parent: Vec<_> = (0..n).collect();
        fn root(p: &[usize], mut i: usize) -> usize {
            while p[i] != i {
                i = p[i];
            }
            i
        }
        for (i, c) in choice.iter().enumerate() {
            if c.accepted {
                for donor in &c.donors {
                    let j = step.report.clone_plan.sources[donor.pool_index as usize].slot as usize;
                    let a = root(&parent, i);
                    let z = root(&parent, j);
                    parent[a] = z;
                }
            }
        }
        for i in 0..n {
            let r = root(&parent, i);
            components[r].push(i);
        }
        let energy = components
            .iter()
            .filter(|g| !g.is_empty())
            .map(|g| {
                let points: Vec<_> = g.iter().map(|&i| v[i].clone()).collect();
                variance(&points) * g.len() as f64 / nf
            })
            .sum::<f64>();
        let expected_velocity = variance(&v) - (1. - alpha * alpha) * energy;
        let pstar = moments
            .acceptance_probabilities
            .iter()
            .enumerate()
            .filter(|(i, _)| alive[*i] && norm(&x[*i]) > radius2 / 2.)
            .map(|(_, p)| *p)
            .fold(1_f64, f64::min);
        let theta = radius2 / 2.;
        let boundary_upper = (1. - pstar) * initialb
            + k as f64 / nf * (pstar * theta + mx)
            + (n - k) as f64 / nf * mx;
        let weighted_measured = cv * (measuredx - initialx + lambda * (measuredv - initialv))
            + cb * (measuredb - initialb);
        let weighted_predicted = cv
            * (moments.expected_position_variance - initialx
                + lambda * (expected_velocity - initialv))
            + cb * (expectedb - initialb);
        // Compare two independent proposals from the same entering law. Their
        // input transport is zero, yet their output transport need not vanish.
        let right = proposal(&config, benchmark, &x, &v, &alive, seed + 500_000_000).await?;
        let ro = right.steps[0]
            .stages
            .iter()
            .find(|s| s.stage == "post_transform")
            .unwrap();
        let rx = stage_points(ro, "positions", d)?;
        let rv = stage_points(ro, "velocities", d)?;
        let leftphase: Vec<_> = ox
            .iter()
            .zip(&ov)
            .map(|(x, v)| [x.clone(), v.clone()].concat())
            .collect();
        let rightphase: Vec<_> = rx
            .iter()
            .zip(&rv)
            .map(|(x, v)| [x.clone(), v.clone()].concat())
            .collect();
        let cost = |a: &[f64], z: &[f64]| {
            let dx: Vec<_> = a[..d].iter().zip(&z[..d]).map(|(a, b)| a - b).collect();
            let dv: Vec<_> = a[d..].iter().zip(&z[d..]).map(|(a, b)| a - b).collect();
            norm(&dx) + lambda * norm(&dv) + b * dx.iter().zip(&dv).map(|(a, b)| a * b).sum::<f64>()
        };
        let transport = uniform_transport(&leftphase, &rightphase, cost)?.0;
        pairs[0].push((measuredx, moments.expected_position_variance));
        pairs[1].push((measuredx, reset));
        pairs[2].push((measuredcenter, moments.conditional_barycenter_variance));
        pairs[3].push((measuredcenter, conditional_bary_bound));
        pairs[4].push((measuredv, expected_velocity));
        pairs[5].push((measuredv - initialv, cvel));
        pairs[6].push((measuredb, expectedb));
        pairs[7].push((measuredb, boundary_upper));
        pairs[8].push((weighted_measured, weighted_predicted));
        pairs[9].push((transport, cw));
        let right_moments = analyze_cloning_step(&right, 0)?
            .moments
            .ok_or_else(|| error("right conditional moments missing"))?;
        let right_pstar = right_moments
            .acceptance_probabilities
            .iter()
            .enumerate()
            .filter(|(i, _)| alive[*i] && norm(&x[*i]) > theta)
            .map(|(_, p)| *p)
            .fold(1_f64, f64::min);
        let right_boundary_upper = (1. - right_pstar) * initialb
            + k as f64 / nf * (right_pstar * theta + mx)
            + (n - k) as f64 / nf * mx;
        let right_barrier = rx.iter().map(|x| norm(x)).sum::<f64>() / nf;
        let complete_weighted_drift = transport
            + cv * (measuredx + variance(&rx) - 2. * initialx
                + lambda * (measuredv + variance(&rv) - 2. * initialv))
            + cb * (measuredb + right_barrier - 2. * initialb);
        let complete_weighted_bound = cw
            + cv * (-2. * initialx
                + 2. * reset
                + if k == n {
                    0.
                } else {
                    8. * lambda * vmax * vmax
                })
            + cb * (boundary_upper + right_boundary_upper - 2. * initialb);
        pairs[10].push((complete_weighted_drift, complete_weighted_bound));
        replicas.push(json!({"seed":seed,"entering_fitness":step.report.pre_clone_fitness.fitness,"accepted_probabilities":moments.acceptance_probabilities,"proposal_position_variance":measuredx,"predicted_position_variance":moments.expected_position_variance,"proposal_velocity_variance":measuredv,"predicted_velocity_variance":expected_velocity,"accepted_component_energy":energy,"barrier_value":measuredb,"predicted_barrier_value":expectedb,"conditional_exposed_pressure_floor":pstar,"boundary_drift_upper":boundary_upper-initialb,"weighted_drift":weighted_measured,"predicted_weighted_drift":weighted_predicted,"independent_pair_proposal_transport":transport,"complete_two_swarm_weighted_drift":complete_weighted_drift,"complete_two_swarm_weighted_bound":complete_weighted_bound}));
    }
    let specs: [(&str, &[&str], &str); 11] = [
        (
            "positional_conditional_moment",
            &[
                "thm-positional-variance-contraction",
                "lem-variance-change-decomposition",
            ],
            "equal",
        ),
        (
            "positional_reset_Bx",
            &["thm-positional-variance-contraction"],
            "upper",
        ),
        (
            "conditional_barycenter_covariance",
            &["thm-cloning-canonical-barycenter-concentration"],
            "equal",
        ),
        (
            "conditional_barycenter_bound",
            &["thm-cloning-canonical-barycenter-concentration"],
            "upper",
        ),
        (
            "component_velocity_dissipation",
            &[
                "prop-cloning-component-conservation",
                "thm-complete-variance-drift",
            ],
            "equal",
        ),
        (
            "alive_velocity_drift_Cv",
            &["thm-velocity-variance-bounded-expansion"],
            "upper",
        ),
        (
            "declared_quadratic_barrier_integral",
            &["thm-boundary-potential-contraction"],
            "equal",
        ),
        (
            "conditional_boundary_affine_drift",
            &["thm-boundary-potential-contraction"],
            "upper",
        ),
        (
            "weighted_signed_cloning_drift",
            &["thm-complete-cloning-drift"],
            "equal",
        ),
        (
            "inter_swarm_transport_CW",
            &["thm-inter-swarm-bounded-expansion"],
            "upper",
        ),
        (
            "complete_two_swarm_weighted_drift",
            &["thm-complete-cloning-drift"],
            "upper",
        ),
    ];
    let comparisons = specs
        .iter()
        .zip(&pairs)
        .map(|((id, labels, relation), values)| comparison(id, labels, relation, values))
        .collect();
    Ok(EmpiricalCase{name:format!("{benchmark:?}_{profile}_n{n}_d{d}"),native_config:config,benchmark,walkers:n,dimensions:d,alive:k,initial_positions:x,initial_velocities:v,entering_alive:alive,
        constants:json!({"D_x_squared":diam2,"input_position_radius_squared":radius2,"jitter_sigma":sigma,"restitution":alpha,"B_x":reset,"conditional_barycenter_bound":conditional_bary_bound,"input_position_variance":initialx,"input_velocity_variance":initialv,"C_v":cvel,"M_x":mx,"M_h":mh,"C_W":cw,"lambda_v":lambda,"b":b,"c_V":cv,"c_B":cb,"barrier":"phi(x)=|x|^2, a declared global C2 nonnegative observable; objective unchanged","boundary_threshold":radius2/2.,"positive_boundary_rate_scope":"p_star is computed conditionally on each actual retained measurement; zero-pressure realizations receive zero and are retained. No positive state-uniform rate is inferred."}),comparisons,replicas,
        scope:"One native cloning proposal with complete measurement, donor, gate, revival, Gaussian jitter and shared component rotation. All normalized moments use N; the discarded dead positions do not enter the bound. Two independent copies also test the transport expansion bound. Conditional predictions depend on retained measurement marks and accepted component graphs; each statistical residual has an independent engine seed.".into()})
}

async fn balanced_pressure_case(samples: usize, n: usize, d: usize, case: u64) -> Result<Value> {
    let config = GasConfig::euclidean(d, 0.04)?;
    let positions = |radius: f64| {
        (0..n)
            .map(|i| {
                let mut x = vec![0.; d];
                x[0] = if i < n / 2 { radius } else { -radius };
                x
            })
            .collect::<Vec<_>>()
    };
    let x = positions(1.);
    let y = positions(0.5);
    let v = vec![vec![0.; d]; n];
    let alive = vec![true; n];
    let w = 0.25_f64;
    let chi = canonical_cloning_constants().balanced_two_cluster_chi_0;
    let mut values = vec![];
    let mut replicas = vec![];
    for replicate in 0..samples {
        let seed = 900_000_000 + case * 1_000_000 + replicate as u64;
        let a = proposal(&config, Benchmark::Quadratic, &x, &v, &alive, seed).await?;
        let z = proposal(
            &config,
            Benchmark::Quadratic,
            &y,
            &v,
            &alive,
            seed + 500_000,
        )
        .await?;
        let pressure = (a.steps[0]
            .report
            .clone_plan
            .choices
            .iter()
            .map(|c| c.probability.unwrap())
            .sum::<f64>()
            + z.steps[0]
                .report
                .clone_plan
                .choices
                .iter()
                .map(|c| c.probability.unwrap())
                .sum::<f64>())
            / n as f64;
        let activity = pressure * w;
        // Upper comparison representation of Q >= chi*Vstruct.
        values.push((-activity, -chi * w));
        replicas.push(json!({"seed":seed,"paired_error":w,"measured_retained_activity":activity,"chi_Vstruct":chi*w}));
    }
    let checked = comparison(
        "balanced_keystone_feedback",
        &["cor-keystone-canonical-balanced-structural"],
        "upper",
        &values,
    );
    Ok(
        json!({"N":n,"d":d,"native_config":config,"positions1":x,"positions2":y,"chi":chi,"Vstruct":w,"comparisons":[checked],"replicas":replicas,"scope":"Two centered balanced native clouds at radii 1 and 0.5 in the first coordinate, zero velocities, canonical quadratic objective and parameters. Pairing represents the monotone optimal coupling of their unlabeled entering empirical laws. Independent complete measurement samples compare actual error-weighted pressure with the chapter's N-uniform zero-offset coefficient."}),
    )
}

#[cfg(test)]
mod tests {
    use super::comparison;
    #[test]
    fn a_wrong_constant_cannot_pass_a_zero_uncertainty_comparison() {
        let wrong = vec![(2., 1.); 64];
        assert!(!comparison("wrong", &[], "upper", &wrong).not_rejected);
        assert!(!comparison("wrong", &[], "equal", &wrong).not_rejected);
        let valid = vec![(0.5, 1.); 64];
        assert!(comparison("valid", &[], "upper", &valid).upper_bound_supported);
    }
}
