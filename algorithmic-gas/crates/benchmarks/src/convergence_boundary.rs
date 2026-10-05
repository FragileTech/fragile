//! Complete-update moment and integrable boundary estimates, including revival.
use crate::{Benchmark, BenchmarkModel, convergence_framework::BoundCheck};
use algorithmic_gas::{
    GasBuilder, GasConfig, GasError, ObservationBatch, Population, Result, TensorBatch,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResetInputs {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub dt: f64,
    pub friction: f64,
    pub jitter: f64,
    pub velocity_cap: f64,
    pub restitution: f64,
    pub force_lipschitz: f64,
    pub force_offset: f64,
    pub thermostat_scale: f64,
    pub position_diffusion: f64,
    pub velocity_weight: f64,
}

/// The full-step reset theorem uses eligible source positions. Entering dead
/// positions do not occur in these constants, because mandatory revival copies
/// a current live source before the kinetic update.
pub fn reset_constants(input: &ResetInputs) -> Result<BTreeMap<String, f64>> {
    let d = input.lower.len();
    if d == 0
        || d > 256
        || input.upper.len() != d
        || input
            .lower
            .iter()
            .zip(&input.upper)
            .any(|(l, u)| !l.is_finite() || !u.is_finite() || u <= l || !(u - l).is_finite())
        || [
            input.dt,
            input.velocity_cap,
            input.position_diffusion,
            input.velocity_weight,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
        || [
            input.friction,
            input.jitter,
            input.restitution,
            input.force_lipschitz,
            input.force_offset,
            input.thermostat_scale,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.)
        || input.restitution > 1.
    {
        return Err(GasError::Configuration(
            "invalid complete reset inputs".into(),
        ));
    }
    let dimension = d as f64;
    let r_d_squared = input
        .lower
        .iter()
        .zip(&input.upper)
        .map(|(l, u)| l.abs().max(u.abs()).powi(2))
        .sum::<f64>();
    let c = (-input.friction * input.dt).exp();
    let s_h_squared = if input.friction == 0. {
        input.dt
    } else {
        -(-2. * input.friction * input.dt).exp_m1() / (2. * input.friction)
    };
    let w = (1. + 2. * input.restitution) * input.velocity_cap;
    let a = 1. + input.dt.powi(2) * (1. + c) * input.force_lipschitz / 4.;
    let d_0 = input.dt * (1. + c) * w / 2. + input.dt.powi(2) * (1. + c) * input.force_offset / 4.;
    let thermostat_trace = dimension * input.thermostat_scale.powi(2);
    let kinetic_position_variance = input.dt.powi(2) * s_h_squared * input.thermostat_scale.powi(2)
        / 4.
        + input.position_diffusion.powi(2) * input.dt;
    let m_x = (a * (r_d_squared + dimension * input.jitter.powi(2)).sqrt() + d_0).powi(2)
        + dimension * kinetic_position_variance;
    let m = 1. + m_x + input.velocity_weight * input.velocity_cap.powi(2);
    let log_volume = input
        .lower
        .iter()
        .zip(&input.upper)
        .map(|(l, u)| (u - l).ln())
        .sum::<f64>();
    // Integral of log(L²/[t(L-t)]) on an interval of length L is 2L.
    let log_barrier_l1 = (2. * dimension).ln() + log_volume;
    let log_density_max = -dimension / 2.
        * ((2. * std::f64::consts::PI).ln() + 2. * input.position_diffusion.ln() + input.dt.ln());
    let log_m_b = log_density_max + log_barrier_l1;
    let m_b = log_m_b.exp();
    let mut values = BTreeMap::from([
        ("R_D_squared".into(), r_d_squared),
        ("c".into(), c),
        ("s_h_squared".into(), s_h_squared),
        ("W".into(), w),
        ("A".into(), a),
        ("D_0".into(), d_0),
        ("thermostat_trace".into(), thermostat_trace),
        (
            "kinetic_position_variance".into(),
            kinetic_position_variance,
        ),
        ("M_x".into(), m_x),
        ("M".into(), m),
        ("log_box_volume".into(), log_volume),
        ("log_barrier_L1".into(), log_barrier_l1),
        ("log_terminal_density_max".into(), log_density_max),
        ("log_M_b".into(), log_m_b),
        ("foster_q_reference".into(), 0.5),
        ("foster_kappa_reference".into(), 0.5),
        ("boundary_weight_reference".into(), 0.05),
    ]);
    // Never replace a strictly positive factor/rate by zero when its product
    // can still be represented. The logarithms remain available in every case.
    for (name, log_value) in [
        ("box_volume", log_volume),
        ("barrier_L1", log_barrier_l1),
        ("terminal_density_max", log_density_max),
        ("M_b", log_m_b),
    ] {
        let value = log_value.exp();
        if value.is_finite() && value > 0. {
            values.insert(name.into(), value);
        }
    }
    if m_b.is_finite() && (m + 0.05 * m_b).is_finite() {
        values.insert("augmented_offset_reference".into(), m + 0.05 * m_b);
    }
    if values.values().any(|x| !x.is_finite()) {
        return Err(GasError::Configuration(
            "reset constants or logarithms overflow f64".into(),
        ));
    }
    Ok(values)
}

/// Nonnegative logarithmic barrier, extended by zero outside the box and at
/// its boundary. This auxiliary observable does not change the objective.
pub fn logarithmic_barrier(x: &[f64], lower: &[f64], upper: &[f64]) -> Result<f64> {
    if x.len() != lower.len()
        || x.len() != upper.len()
        || x.is_empty()
        || x.iter().any(|x| !x.is_finite())
        || lower
            .iter()
            .zip(upper)
            .any(|(l, u)| !l.is_finite() || !u.is_finite() || u <= l || !(u - l).is_finite())
    {
        return Err(GasError::Configuration(
            "invalid logarithmic barrier input".into(),
        ));
    }
    if x.iter()
        .zip(lower)
        .zip(upper)
        .any(|((x, l), u)| x <= l || x >= u)
    {
        return Ok(0.);
    }
    Ok(x.iter()
        .zip(lower)
        .zip(upper)
        .map(|((x, l), u)| 2. * (u - l).ln() - (x - l).ln() - (u - x).ln())
        .sum())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryEstimate {
    pub id: String,
    pub source_labels: Vec<String>,
    pub scope: String,
    pub samples: usize,
    pub sample_mean: f64,
    pub standard_error: f64,
    pub prediction_or_bound: f64,
    pub comparison: String,
    pub status: String,
}

fn estimate(
    id: &str,
    labels: &[&str],
    values: &[f64],
    bound: f64,
    equality: bool,
) -> BoundaryEstimate {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.);
    let se = (variance / n).sqrt();
    let tolerance = 6. * se + 1e-9 * (1. + bound.abs());
    let passed = mean.is_finite()
        && if equality {
            (mean - bound).abs() <= tolerance
        } else {
            mean <= bound + tolerance
        };
    BoundaryEstimate {
        id: id.into(),
        source_labels: labels.iter().map(|s| (*s).into()).collect(),
        scope: "Independent one-step native swarms with the same frozen input; six standard errors. The moment observable includes all slots, even terminally dead ones. This is finite-sample evidence, not a simultaneous confidence theorem.".into(),
        samples: values.len(),
        sample_mean: mean,
        standard_error: se,
        prediction_or_bound: bound,
        comparison: if equality { "mean_identity" } else { "upper_bound" }.into(),
        status: if passed { "not_rejected" } else { "violated" }.into(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryEnsemble {
    pub name: String,
    pub walkers: usize,
    pub dimensions: usize,
    pub entering_alive: usize,
    pub entering_dead_coordinate_magnitude: f64,
    pub inputs: ResetInputs,
    pub constants: BTreeMap<String, f64>,
    pub estimates: Vec<BoundaryEstimate>,
    pub checks: Vec<BoundCheck>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryReport {
    pub ensembles: Vec<BoundaryEnsemble>,
    pub checks: Vec<BoundCheck>,
    pub safe_population_constants: Vec<serde_json::Value>,
    pub scope_notes: Vec<String>,
}

/// Validate the direct full-step Foster reset, rather than assigning a permanent
/// displacement cost to dying walkers. Inputs include unrestricted dead rows.
pub async fn validate_boundary(samples: usize, seed: u64) -> Result<BoundaryReport> {
    if !(64..=100_000).contains(&samples) {
        return Err(GasError::Configuration(
            "boundary samples must be 64..100000".into(),
        ));
    }
    let mut ensembles = vec![];
    for (case, (walkers, dimensions, alive, magnitude, dt, diffusion)) in [
        (4, 1, 2, 1e3, 0.04, 0.1),
        (4, 1, 2, 1e9, 0.04, 0.1),
        (16, 2, 8, 1e6, 0.04, 0.1),
        (4, 1, 2, 1e6, 1.5, 0.5),
        (4, 1, 1, 1e9, 0.04, 0.1),
    ]
    .into_iter()
    .enumerate()
    {
        let inputs = ResetInputs {
            lower: vec![-2.; dimensions],
            upper: vec![2.; dimensions],
            dt,
            friction: 1.,
            jitter: 0.1,
            velocity_cap: 2.,
            restitution: 0.5,
            force_lipschitz: 1.,
            force_offset: 0.,
            thermostat_scale: 1.,
            position_diffusion: diffusion,
            velocity_weight: 1.,
        };
        let constants = reset_constants(&inputs)?;
        let x: Vec<_> = (0..walkers)
            .flat_map(|row| {
                let sign = if row % 2 == 0 { -1. } else { 1. };
                std::iter::repeat_n(sign * if row < alive { 0.5 } else { magnitude }, dimensions)
            })
            .collect();
        let v = vec![0.15 / (dimensions as f64).sqrt(); walkers * dimensions];
        let mut moment = vec![];
        let mut position_residual = vec![];
        let mut barrier = vec![];
        let mut max_velocity: f64 = 0.;
        let mut wrong_revivals = 0;
        let mut completed = 0;
        for repetition in 0..samples {
            let mut config = GasConfig::euclidean(dimensions, dt)?;
            config.seed = seed.wrapping_add(case as u64 * 1_000_000 + repetition as u64);
            config.kinetic.position_diffusion = diffusion;
            let mut observations =
                ObservationBatch::positions(TensorBatch::vectors(walkers, dimensions, x.clone())?);
            observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(walkers, dimensions, v.clone())?,
            );
            let population = Population::new(observations)?;
            let model = BenchmarkModel {
                benchmark: Benchmark::Quadratic,
                field: "positions".into(),
                direction: config.fitness.direction,
            };
            let mut gas = GasBuilder::new(population, model.clone())
                .gradient(model)
                .config(config)
                .build()
                .await?;
            gas.start_recording(Default::default())?;
            gas.step().await?;
            let archive = gas
                .recording()
                .ok_or_else(|| GasError::Configuration("missing boundary archive".into()))?;
            let step = &archive.steps[0];
            let clone = step
                .stages
                .iter()
                .find(|s| s.stage == "post_transform")
                .ok_or_else(|| GasError::Configuration("missing completed revival stage".into()))?;
            let clone_x = &clone
                .fields
                .get("positions")
                .ok_or_else(|| GasError::Configuration("missing cloned positions".into()))?
                .values;
            let clone_v = &clone
                .fields
                .get("velocities")
                .ok_or_else(|| GasError::Configuration("missing collision velocities".into()))?
                .values;
            let end_x = step
                .final_population
                .observations
                .field("positions")?
                .values();
            let end_v = step
                .final_population
                .observations
                .field("velocities")?
                .values();
            let c = constants["c"];
            let b = dt * (1. + c) / 2.;
            let a = 1. - b * dt / 2.;
            let conditional_position_moment = clone_x
                .iter()
                .zip(clone_v)
                .map(|(x, v)| (a * x + b * v).powi(2))
                .sum::<f64>()
                / walkers as f64
                + dimensions as f64 * constants["kinetic_position_variance"];
            let x_moment = end_x.iter().map(|x| x * x).sum::<f64>() / walkers as f64;
            let v_moment = end_v.iter().map(|v| v * v).sum::<f64>() / walkers as f64;
            moment.push(1. + x_moment + v_moment);
            position_residual.push(x_moment - conditional_position_moment);
            let mut barrier_sum = 0.;
            for row in 0..walkers {
                max_velocity = max_velocity.max(
                    end_v[row * dimensions..(row + 1) * dimensions]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt(),
                );
                if step.final_population.validity[row].eligible(false) {
                    barrier_sum += logarithmic_barrier(
                        &end_x[row * dimensions..(row + 1) * dimensions],
                        &inputs.lower,
                        &inputs.upper,
                    )?;
                }
                if !step.report.pre_clone_eligible[row] {
                    let choice = &step.report.clone_plan.choices[row];
                    if !choice.accepted
                        || !choice.revival
                        || choice.probability != Some(1.)
                        || choice.donors.len() != 1
                    {
                        wrong_revivals += 1;
                        continue;
                    }
                    let source =
                        &step.report.clone_plan.sources[choice.donors[0].pool_index as usize];
                    wrong_revivals += usize::from(
                        source.frame != 0 || !step.report.pre_clone_eligible[source.slot as usize],
                    );
                }
            }
            barrier.push(barrier_sum / walkers as f64);
            completed += 1;
        }
        let checks = vec![
            BoundCheck::upper(
                "completed_revival_uses_live_source",
                &[
                    "thm-canonical-full-step-reset-drift",
                    "lem-dead-walker-clone-prob",
                    "lem-eg-scheduled-revival",
                ],
                "Every entering dead destination is overwritten from a current live source before kinetics, irrespective of its retained coordinates.",
                wrong_revivals as f64,
                0.,
            ),
            BoundCheck::upper(
                "completed_velocity_cap",
                &["thm-canonical-full-step-reset-drift"],
                "All slots after B2 and the smooth cap, including terminally dead slots.",
                max_velocity,
                inputs.velocity_cap,
            ),
            BoundCheck::upper(
                "complete_one_step_ensemble",
                &["thm-canonical-full-step-reset-drift"],
                "Every independent replicate remains represented, including killed outcomes.",
                samples as f64 - completed as f64,
                0.,
            ),
        ];
        let estimates = vec![
            estimate(
                "complete_marked_reset_M",
                &["thm-canonical-full-step-reset-drift"],
                &moment,
                constants["M"],
                false,
            ),
            estimate(
                "conditional_gaussian_position_moment",
                &[
                    "thm-canonical-full-step-reset-drift",
                    "lem-euclidean-perturb-moment",
                ],
                &position_residual,
                0.,
                true,
            ),
            estimate(
                "complete_boundary_reset_Mb",
                &[
                    "cor-canonical-full-step-boundary-reset",
                    "lem-barrier-reduction-cloning",
                ],
                &barrier,
                constants["M_b"],
                false,
            ),
        ];
        ensembles.push(BoundaryEnsemble {
            name: format!("reset_n{walkers}_d{dimensions}_alive{alive}_dead{magnitude}_h{dt}"),
            walkers,
            dimensions,
            entering_alive: alive,
            entering_dead_coordinate_magnitude: magnitude,
            inputs,
            constants,
            estimates,
            checks,
        });
    }
    let mut checks = vec![];
    let mut safe_population_constants = vec![];
    for n in [4, 16, 64] {
        let barriers: Vec<_> = (0..n).map(|i| if i % 4 == 0 { 6. } else { 2. }).collect();
        let theta = 4.;
        let barrier_mean = barriers.iter().sum::<f64>() / n as f64;
        let safe = barriers.iter().filter(|&&b| b < theta).count() as f64;
        checks.push(BoundCheck::upper(&format!("barrier_safe_count_n{n}"), &["cor-extinction-suppression", "def-boundary-exposed-set"], "Exact finite barrier-count inequality at an all-alive conditioned stage; the safe-sublevel survival probability still requires its separate Gaussian margin hypothesis.", n as f64 - n as f64 * barrier_mean / theta, safe));
    }
    // Exact conditional independent death calculations at the stage after all
    // shared means are frozen. This is distinct from unconditional independence.
    for d in [1, 2, 4] {
        for n in [4, 16, 64] {
            let r: f64 = 1.;
            let sigma: f64 = 0.25;
            let q = (2_f64.powf(d as f64 / 2.) * (-r.powi(2) / (4. * sigma.powi(2))).exp()).min(1.);
            let safe_count = n / 2;
            let independent_extinction_upper = q.powi(safe_count);
            let exponent = safe_count as f64 * (1. / q).ln();
            safe_population_constants.push(serde_json::json!({
                "walkers":n, "dimensions":d, "safe_count":safe_count,
                "radius":r, "position_noise_scale":sigma, "safe_fraction":0.5,
                "individual_death_bound_q":q, "exponent_per_walker":0.5*(1./q).ln(),
                "log_extinction_bound":-exponent,
                "extinction_bound":independent_extinction_upper,
                "source":"cor-extinction-suppression",
                "scope":"Supplied independent Gaussian rows at the frozen-mean stage with ball margin r; this calculation does not infer a safe fraction from an interacting trajectory."
            }));
            checks.push(BoundCheck::upper(&format!("safe_population_exponent_n{n}_d{d}"), &["cor-extinction-suppression"], "Supplied ball margin r=1, Gaussian position scale 1/4, a=1/2; row noises are independent at the frozen-mean stage. Product of row upper bounds equals the exponential expression.", (independent_extinction_upper - (-exponent).exp()).abs(), 1e-14));
        }
    }
    Ok(BoundaryReport {
        ensembles, checks, safe_population_constants,
        scope_notes: vec![
            "The full-step canonical reset is checked on actual weighted-donor swarms, including huge entering dead coordinates. No death-position displacement is carried into its constants.".into(),
            "M and M_b are N-independent analytic upper bounds; favorable fitness mass and a strict clone-only boundary drift are additional hypotheses, not inferred from the reset.".into(),
            "The exponential safe-population formula is evaluated only with supplied stage-wise independent death bounds. It does not assert conditional-on-survival normalization or QSD mixing.".into(),
        ],
    })
}
