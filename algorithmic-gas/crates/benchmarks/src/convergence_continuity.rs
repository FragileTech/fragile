//! Finite fixtures for the remaining chapter 1 continuity calculations.
//!
//! Native distance, noise, mapping, gate, literal-copy and addressed-RNG
//! primitives are exercised here. Finite transport and support enumeration
//! are exact diagnostics; stochastic acceptance bands do not certify unsampled
//! states, boundary regularity, PRNG independence or complete-swarm rates.
use crate::{
    convergence_coefficients::{ComputedConstant, standardization_coefficients},
    convergence_framework::BoundCheck,
    convergence_lyapunov::uniform_transport,
};
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasError, ObservationBatch, Population, Precision, Result,
    TensorBatch,
    cloning::{CloneChoice, CloneDecision, ClonePlan, WeightedDonor},
    donor::{CompanionRequest, CompanionSampler, DonorModule, DonorPool},
    fitness::{PositiveMap, PositiveMapping, Standardizer},
    geometry::{AlgorithmicDistance, ComparisonKind, Distance, InteractionKernel, Kernel},
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry, NoiseRequest, NoiseSource},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuityReport {
    pub computed: Vec<ComputedConstant>,
    pub checks: Vec<BoundCheck>,
    pub noise_experiments: Vec<NoiseMomentExperiment>,
    pub tail_experiments: Vec<TailExperiment>,
    pub generic_revival: GenericRevivalReport,
    pub population_scaling: PopulationScalingReport,
    pub unavailable: BTreeMap<String, String>,
    pub scope: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopulationMeasurementBaseline {
    pub expected_operator_squared_error: f64,
    pub independent_raw_variance_contribution: f64,
    pub exact_raw_summed_mean_square: f64,
    pub exact_raw_indexed_mean_square_per_atom: f64,
    pub source_raw_summed_bound: f64,
    pub value_coefficient_c_n: f64,
    pub exact_standardized_summed_mean_square: f64,
    pub exact_standardized_indexed_mean_square_per_atom: f64,
    pub source_standardized_summed_bound: f64,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopulationScalingCase {
    pub walkers: usize,
    pub companion_law: String,
    pub sigma_min: f64,
    pub raw_support_diameter: f64,
    pub opposite_cluster_probability: f64,
    pub empirical_lipschitz_squared: f64,
    pub universal_squared_error_bound: f64,
    pub exact_raw_normalized_empirical_error: f64,
    pub exact_standardized_normalized_empirical_error: f64,
    pub exact_raw_error_to_expected_law: f64,
    pub exact_standardized_error_to_expected_law: f64,
    pub raw_expected_law_decay_bound: f64,
    pub standardized_expected_law_decay_bound: f64,
    pub sampled_independent_output_pairs: usize,
    pub observed_raw_normalized_empirical_error: f64,
    pub observed_standardized_normalized_empirical_error: f64,
    pub observed_raw_error_to_expected_law: f64,
    pub observed_standardized_error_to_expected_law: f64,
    pub raw_sample_standard_error: f64,
    pub standardized_sample_standard_error: f64,
    pub raw_hoeffding_margin: f64,
    pub standardized_hoeffding_margin: f64,
    pub confidence_per_quantity: f64,
    pub status: String,
    pub measurement_baseline: PopulationMeasurementBaseline,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopulationScalingReport {
    pub cases: Vec<PopulationScalingCase>,
    pub checks: Vec<BoundCheck>,
    pub computed: Vec<ComputedConstant>,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenericRevivalCase {
    pub name: String,
    pub eta: f64,
    pub alpha: f64,
    pub beta: f64,
    pub epsilon_clone: f64,
    pub p_max: f64,
    pub log_fitness_floor: f64,
    pub fitness_floor: Option<f64>,
    pub log_score_lower_bound: f64,
    pub score_lower_bound: Option<f64>,
    pub log_revival_ratio: f64,
    pub revival_ratio: Option<f64>,
    pub log_uniform_acceptance: f64,
    pub exact_uniform_acceptance: Option<f64>,
    pub strict_condition_holds: bool,
    pub status: String,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenericRevivalReport {
    pub cases: Vec<GenericRevivalCase>,
    pub checks: Vec<BoundCheck>,
    pub scope: String,
}

fn representable_positive(log_value: f64) -> Option<f64> {
    let value = log_value.exp();
    (value.is_finite() && value > 0.).then_some(value)
}

/// Generic score threshold, conditional on the source's supplied fitness floor.
/// Native dead-walker overriding is deliberately not used for this calculation.
pub fn generic_revival_case(
    name: &str,
    eta: f64,
    alpha: f64,
    beta: f64,
    epsilon_clone: f64,
    p_max: f64,
) -> Result<GenericRevivalCase> {
    require(
        [eta, epsilon_clone, p_max]
            .iter()
            .all(|x| x.is_finite() && *x > 0.)
            && [alpha, beta].iter().all(|x| x.is_finite() && *x >= 0.)
            && (alpha + beta).is_finite()
            && alpha + beta > 0.,
        "generic revival requires positive finite eta, epsilon and p_max and nonnegative finite exponents with positive sum",
    )?;
    let log_floor = (alpha + beta) * eta.ln();
    let log_score = log_floor - epsilon_clone.ln();
    let log_ratio = log_score - p_max.ln();
    require(
        [log_floor, log_score, log_ratio]
            .iter()
            .all(|x| x.is_finite()),
        "generic revival logarithmic calculation overflowed",
    )?;
    let strict = log_ratio > 0.;
    let log_acceptance = log_ratio.min(0.);
    Ok(GenericRevivalCase {
        name: name.into(), eta, alpha, beta, epsilon_clone, p_max,
        log_fitness_floor: log_floor,
        fitness_floor: representable_positive(log_floor),
        log_score_lower_bound: log_score,
        score_lower_bound: representable_positive(log_score),
        log_revival_ratio: log_ratio,
        revival_ratio: representable_positive(log_ratio),
        log_uniform_acceptance: log_acceptance,
        exact_uniform_acceptance: representable_positive(log_acceptance),
        strict_condition_holds: strict,
        status: if strict { "applicable" } else { "not_applicable" }.into(),
        scope: "Auxiliary generic dead score equals its supplied lower bound eta^(alpha+beta)/epsilon_clone, with T~Uniform(0,p_max). The acceptance is exact for this least-score fixture; it is a lower bound for larger scores. Strict theorem applicability is reported separately. Native unconditional dead revival is a different rule.".into(),
    })
}

pub fn generic_revival_validation_cases() -> Result<GenericRevivalReport> {
    let cases = vec![
        generic_revival_case("sufficient", 0.5, 1., 1., 0.1, 1.)?,
        generic_revival_case("insufficient", 0.5, 1., 1., 1., 1.)?,
        generic_revival_case("borderline", 0.5, 1., 1., 0.25, 1.)?,
        generic_revival_case("underflowing_fitness_floor", 1e-200, 1., 1., 1e-200, 1.)?,
        generic_revival_case("overflowing_ratio", 1e100, 1., 1., 1e-200, 1.)?,
    ];
    let mut report = GenericRevivalReport {
        cases,
        checks: vec![],
        scope: "Computes kappa_revival and the exact ideal continuous-uniform gate conditional on the stated common positive fitness floor. Sufficient cases exercise the strict score guarantee; insufficient and equality cases are not applications of that axiom. Logarithms preserve values outside floating-point exponent range.".into(),
    };
    for case in &report.cases {
        if let (Some(ratio), Some(score)) = (case.revival_ratio, case.score_lower_bound) {
            report.checks.push(BoundCheck::upper(
                "generic_revival_ratio_identity",
                &["axiom-guaranteed-revival"],
                &case.scope,
                (ratio - score / case.p_max).abs() / ratio,
                1e-12,
            ));
            // Lebesgue length of the accepted interval divided by p_max.
            let interval_probability = score.min(case.p_max) / case.p_max;
            if let Some(acceptance) = case.exact_uniform_acceptance {
                report.checks.push(BoundCheck::upper(
                    "generic_revival_uniform_interval_acceptance",
                    &["def-cloning-score-function"],
                    &case.scope,
                    (acceptance - interval_probability).abs() / acceptance,
                    1e-12,
                ));
            }
        }
        if case.strict_condition_holds {
            report.checks.push(BoundCheck::upper(
                "generic_revival_strict_sufficient_gate",
                &["thm-revival-guarantee"],
                &case.scope,
                (case.exact_uniform_acceptance.unwrap_or(0.) - 1.).abs(),
                0.,
            ));
        }
    }
    let mut population = Population::new(observations(vec![-2., 0.25, 3., 7.], 1)?)?;
    for (i, validity) in population.validity.iter_mut().enumerate() {
        validity.invalid = i != 1;
    }
    let pool = DonorPool::freeze(&population, 0, &[], 0, false)?;
    report.checks.push(BoundCheck::upper(
        "generic_revival_singleton_donor_support", &["thm-revival-guarantee", "def-companion-selection-measure"],
        "Native eligible donor freezing with one survivor: every normalized law on this sole available atom has donor probability one; no native cloning override is used.",
        ((pool.sources.len() as f64 - 1.).abs()) + u8::from(pool.sources[0].slot != 1) as f64, 0.,
    ));
    report.checks.push(BoundCheck::upper(
        "generic_revival_singleton_self_geometry", &["thm-revival-guarantee"],
        "Native Euclidean comparison of the unique alive donor with itself is zero; the positive fitness floor remains an explicit supplied hypothesis.",
        Distance::default().compare(&pool.population.observations, 0, &pool.population.observations, 0)?, 0.,
    ));
    Ok(report)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseMoments {
    pub dimensions: usize,
    pub innovation_rank: usize,
    pub expected_squared_norm: f64,
    pub squared_norm_variance: f64,
    pub covariance_trace_squared: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoiseMomentExperiment {
    pub name: String,
    pub sampling_primitive: String,
    pub dimensions: usize,
    pub samples: usize,
    pub exact_second_moment: f64,
    pub observed_second_moment: f64,
    pub standard_error: f64,
    pub acceptance_band: f64,
    pub maximum_projection_lipschitz_residual: f64,
    pub maximum_projected_squared_displacement: f64,
    pub status: String,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerturbationCoefficients {
    pub walkers: usize,
    pub diameter: f64,
    pub moment_squared: f64,
    pub failure_probability: f64,
    pub coordinate_bounded_difference: f64,
    pub sum_bounded_differences_squared: f64,
    pub mean_bound: f64,
    pub fluctuation_bound: f64,
    pub average_fluctuation_bound: f64,
    /// The paired metric theorem allocates delta/2 to each output.
    pub paired_metric_offset: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TailExperiment {
    pub walkers: usize,
    pub repetitions: usize,
    pub bernoulli_probability: f64,
    pub diameter: f64,
    pub failure_probability: f64,
    pub fluctuation_bound: f64,
    pub exact_two_sided_tail_probability: f64,
    pub empirical_two_sided_tail_probability: f64,
    pub empirical_hoeffding_margin: f64,
    pub diagnostic_confidence: f64,
    pub status: String,
    pub scope: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FiniteTransport {
    pub wasserstein_squared: f64,
    pub total_variation: f64,
    pub common_part_coupling_cost: f64,
    pub diameter_squared: f64,
    pub fourth_moment_mu: f64,
    pub fourth_moment_nu: f64,
    pub maximum_marginal_residual: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimalSwarmDisplacement {
    pub metric_squared: f64,
    pub positional_sum: f64,
    pub status_mismatch_count: f64,
    pub transport_plan: Vec<Vec<f64>>,
    /// The identity-array plan is only an admissible comparison coupling.
    pub candidate_coupling_cost: f64,
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(GasError::Configuration(message.into()))
    }
}

/// Exact quadratic moments of L*xi for independent unit-variance innovations.
/// The standardized uniform law is the native box law, not a uniform ball.
pub fn fixed_factor_moments(
    dimensions: usize,
    rank: usize,
    factor: &[f64],
    innovation: InnovationLaw,
) -> Result<NoiseMoments> {
    require(
        (1..=256).contains(&dimensions)
            && (1..=dimensions).contains(&rank)
            && dimensions.checked_mul(rank) == Some(factor.len())
            && factor.iter().all(|x| x.is_finite()),
        "noise moment calculation needs a finite d by r factor, 1<=r<=d<=256",
    )?;
    let second = factor.iter().map(|x| x * x).sum::<f64>();
    let trace_squared = (0..dimensions)
        .flat_map(|i| (0..dimensions).map(move |j| (i, j)))
        .map(|(i, j)| {
            (0..rank)
                .map(|a| factor[i * rank + a] * factor[j * rank + a])
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>();
    let column_fourths = (0..rank)
        .map(|a| {
            (0..dimensions)
                .map(|i| factor[i * rank + a].powi(2))
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>();
    let variance = 2. * trace_squared
        - if innovation == InnovationLaw::StandardizedUniform {
            1.2 * column_fourths
        } else {
            0.
        };
    require(
        [second, trace_squared, variance]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.),
        "noise moment calculation overflowed",
    )?;
    Ok(NoiseMoments {
        dimensions,
        innovation_rank: rank,
        expected_squared_norm: second,
        squared_norm_variance: variance,
        covariance_trace_squared: trace_squared,
    })
}

/// Evaluates the chapter's constants after the geometric/mean hypotheses have
/// supplied D and M_pert². This does not infer those hypotheses from samples.
pub fn perturbation_coefficients(
    walkers: usize,
    diameter: f64,
    moment_squared: f64,
    failure_probability: f64,
) -> Result<PerturbationCoefficients> {
    require(
        walkers > 0
            && diameter.is_finite()
            && diameter > 0.
            && moment_squared.is_finite()
            && moment_squared >= 0.
            && failure_probability.is_finite()
            && failure_probability > 0.
            && failure_probability < 1.,
        "perturbation coefficients need N>0, D>0, M²>=0 and 0<delta<1",
    )?;
    let n = walkers as f64;
    let diameter_squared = diameter * diameter;
    let c = diameter_squared / n;
    let sum_c_squared = n * c * c;
    let mean_bound = n * moment_squared;
    // Logarithm subtraction avoids overflowing 2/delta at tiny delta.
    let fluctuation_bound =
        diameter_squared * (n / 2. * (2_f64.ln() - failure_probability.ln())).sqrt();
    let paired_fluctuation =
        diameter_squared * (n / 2. * (4_f64.ln() - failure_probability.ln())).sqrt();
    let paired_metric_offset = 6. / n * (mean_bound + paired_fluctuation);
    require(
        [
            c,
            sum_c_squared,
            mean_bound,
            fluctuation_bound,
            paired_metric_offset,
        ]
        .iter()
        .all(|x| x.is_finite())
            && c > 0.
            && sum_c_squared > 0.,
        "perturbation coefficients overflowed or positive concentration constants underflowed",
    )?;
    Ok(PerturbationCoefficients {
        walkers,
        diameter,
        moment_squared,
        failure_probability,
        coordinate_bounded_difference: c,
        sum_bounded_differences_squared: sum_c_squared,
        mean_bound,
        fluctuation_bound,
        average_fluctuation_bound: fluctuation_bound / n,
        paired_metric_offset,
    })
}

/// Exact monotone transport on a sorted finite subset of the real line. Costs
/// use the native distance provider. The common-part/residual-product coupling
/// is constructed independently of the optimal coupling.
pub fn finite_line_transport(points: &[f64], mu: &[f64], nu: &[f64]) -> Result<FiniteTransport> {
    require(
        !points.is_empty()
            && points.len() == mu.len()
            && points.len() == nu.len()
            && points.iter().all(|x| x.is_finite())
            && points.windows(2).all(|p| p[0] < p[1])
            && mu.iter().chain(nu).all(|x| x.is_finite() && *x >= 0.)
            && (mu.iter().sum::<f64>() - 1.).abs() <= 1e-12
            && (nu.iter().sum::<f64>() - 1.).abs() <= 1e-12,
        "finite transport needs sorted finite points and two normalized nonnegative mass vectors",
    )?;
    let n = points.len();
    let obs = ObservationBatch::positions(TensorBatch::vectors(n, 1, points.to_vec())?);
    let metric = Distance::default();
    let mut rows = vec![0.; n];
    let mut columns = vec![0.; n];
    let mut left = mu.to_vec();
    let mut right = nu.to_vec();
    let mut i = 0;
    let mut j = 0;
    let mut cost = 0.;
    while i < n && j < n {
        let mass = left[i].min(right[j]);
        cost += mass * metric.compare(&obs, i, &obs, j)?.powi(2);
        rows[i] += mass;
        columns[j] += mass;
        left[i] -= mass;
        right[j] -= mass;
        if left[i] <= 0. {
            i += 1;
        }
        if right[j] <= 0. {
            j += 1;
        }
    }
    let marginal_residual = rows
        .iter()
        .zip(mu)
        .chain(columns.iter().zip(nu))
        .map(|(a, b)| (a - b).abs())
        .fold(0_f64, f64::max);
    let common: Vec<f64> = mu.iter().zip(nu).map(|(a, b)| a.min(*b)).collect();
    let residual_mu: Vec<f64> = mu.iter().zip(&common).map(|(a, c)| a - c).collect();
    let residual_nu: Vec<f64> = nu.iter().zip(&common).map(|(a, c)| a - c).collect();
    let tv = 0.5 * mu.iter().zip(nu).map(|(a, b)| (a - b).abs()).sum::<f64>();
    let mut common_cost = 0.;
    if tv > 0. {
        for (i, p) in residual_mu.iter().enumerate() {
            for (j, q) in residual_nu.iter().enumerate() {
                common_cost += p * q / tv * metric.compare(&obs, i, &obs, j)?.powi(2);
            }
        }
    }
    let fourth_mu = points.iter().zip(mu).map(|(x, p)| p * x.powi(4)).sum();
    let fourth_nu = points.iter().zip(nu).map(|(x, p)| p * x.powi(4)).sum();
    let diameter_squared = metric.compare(&obs, 0, &obs, n - 1)?.powi(2);
    require(
        [
            cost,
            common_cost,
            tv,
            fourth_mu,
            fourth_nu,
            diameter_squared,
        ]
        .iter()
        .all(|x| x.is_finite()),
        "finite transport moments/cost overflowed",
    )?;
    Ok(FiniteTransport {
        wasserstein_squared: cost,
        total_variation: tv,
        common_part_coupling_cost: common_cost,
        diameter_squared,
        fourth_moment_mu: fourth_mu,
        fourth_moment_nu: fourth_nu,
        maximum_marginal_residual: marginal_residual,
    })
}

fn constant(
    report: &mut ContinuityReport,
    name: &str,
    value: f64,
    formula: &str,
    labels: &[&str],
    scope: &str,
) {
    report.computed.push(ComputedConstant {
        name: name.into(),
        value,
        formula: formula.into(),
        source_labels: labels.iter().map(|x| (*x).into()).collect(),
        scope: scope.into(),
    });
}

fn check(
    report: &mut ContinuityReport,
    id: &str,
    labels: &[&str],
    scope: &str,
    observed: f64,
    bound: f64,
) {
    report
        .checks
        .push(BoundCheck::upper(id, labels, scope, observed, bound));
}

fn observations(values: Vec<f64>, dimensions: usize) -> Result<ObservationBatch<f64>> {
    require(
        dimensions > 0 && values.len().is_multiple_of(dimensions),
        "invalid fixture shape",
    )?;
    Ok(ObservationBatch::positions(TensorBatch::vectors(
        values.len() / dimensions,
        dimensions,
        values,
    )?))
}

fn squashed_observations(values: Vec<f64>, dimensions: usize) -> Result<ObservationBatch<f64>> {
    require(
        dimensions > 0,
        "squashed fixture dimension must be positive",
    )?;
    let rows = values.len() / dimensions;
    let mut obs = observations(values, dimensions)?;
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(rows, dimensions, vec![0.; rows * dimensions])?,
    );
    Ok(obs)
}

fn squashed_metric() -> Distance {
    Distance::SquashedPhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_radius: 1.,
        velocity_radius: 1.,
        lambda: 0.,
    }
}

/// Optimal uniform transport of the native projected marked empirical swarms.
/// Both components are evaluated on the same optimizer. Intrinsic observation
/// coordinates and marks order the atoms before optimization, making the tie
/// convention independent of their storage ordering.
pub fn optimal_swarm_displacement(
    metric: &Distance,
    left: &ObservationBatch<f64>,
    right: &ObservationBatch<f64>,
    left_alive: &[bool],
    right_alive: &[bool],
    status_weight: f64,
) -> Result<OptimalSwarmDisplacement> {
    metric.validate(left)?;
    metric.validate(right)?;
    let n = left.field(metric.field())?.rows();
    require(
        (1..=64).contains(&n)
            && right.field(metric.field())?.rows() == n
            && left_alive.len() == n
            && right_alive.len() == n
            && status_weight.is_finite()
            && status_weight > 0.,
        "marked swarm transport needs equal nonempty <=64-point supports and a positive finite status weight",
    )?;
    let kind = <Distance as AlgorithmicDistance<f64>>::comparison_kind(metric);
    require(
        kind != ComparisonKind::Dissimilarity,
        "a dissimilarity is not a squared metric cost for empirical transport",
    )?;
    let mut positions = vec![vec![0.; n]; n];
    let mut marked = vec![vec![0.; n]; n];
    for i in 0..n {
        for j in 0..n {
            let value = metric.compare(left, i, right, j)?;
            positions[i][j] = if kind == ComparisonKind::SquaredDistance {
                value
            } else {
                value * value
            };
            marked[i][j] =
                positions[i][j] + status_weight * u8::from(left_alive[i] != right_alive[j]) as f64;
        }
    }
    let order = |obs: &ObservationBatch<f64>, alive: &[bool]| -> Result<Vec<usize>> {
        let mut atoms = vec![];
        for (i, &mark) in alive.iter().enumerate() {
            let mut point = vec![];
            for field in obs.fields.values() {
                point.extend_from_slice(field.row(i)?);
            }
            require(
                point.iter().all(|x| x.is_finite()),
                "canonical transport coordinates must be finite",
            )?;
            point.push(u8::from(mark) as f64);
            atoms.push((i, point));
        }
        atoms.sort_by(|(_, a), (_, b)| {
            a.iter()
                .zip(b)
                .map(|(a, b)| a.total_cmp(b))
                .find(|o| !o.is_eq())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(atoms.into_iter().map(|(i, _)| i).collect())
    };
    let left_order = order(left, left_alive)?;
    let right_order = order(right, right_alive)?;
    let left_addresses: Vec<Vec<f64>> = left_order.iter().map(|i| vec![*i as f64]).collect();
    let right_addresses: Vec<Vec<f64>> = right_order.iter().map(|i| vec![*i as f64]).collect();
    let (metric_squared, canonical_plan) =
        uniform_transport(&left_addresses, &right_addresses, |l, r| {
            marked[l[0] as usize][r[0] as usize]
        })?;
    let mut transport_plan = vec![vec![0.; n]; n];
    for (i, &left_index) in left_order.iter().enumerate() {
        for (j, &right_index) in right_order.iter().enumerate() {
            transport_plan[left_index][right_index] = canonical_plan[i][j];
        }
    }
    let mut positional_sum = 0.;
    let mut status_mismatch_count = 0.;
    for i in 0..n {
        for j in 0..n {
            let mass = n as f64 * transport_plan[i][j];
            positional_sum += mass * positions[i][j];
            status_mismatch_count += mass * u8::from(left_alive[i] != right_alive[j]) as f64;
        }
    }
    let candidate_coupling_cost = (0..n).map(|i| marked[i][i]).sum::<f64>() / n as f64;
    Ok(OptimalSwarmDisplacement {
        metric_squared,
        positional_sum,
        status_mismatch_count,
        transport_plan,
        candidate_coupling_cost,
    })
}

fn optimal_position_sum(
    metric: &Distance,
    left: &ObservationBatch<f64>,
    right: &ObservationBatch<f64>,
) -> Result<f64> {
    let n = left.field(metric.field())?.rows();
    Ok(
        optimal_swarm_displacement(metric, left, right, &vec![true; n], &vec![true; n], 1.)?
            .positional_sum,
    )
}

/// Complete deterministic fixture sweep plus independently repeated native
/// noise diagnostics. All supplied factors and statuses are held fixed.
pub fn continuity_validation_cases() -> Result<ContinuityReport> {
    let mut report = ContinuityReport {
        computed: vec![], checks: vec![], noise_experiments: vec![], tail_experiments: vec![],
        generic_revival: generic_revival_validation_cases()?,
        population_scaling: population_scaling_validation_cases()?,
        unavailable: BTreeMap::from([
            ("general_boundary_modulus".into(), "The finite status fixture uses an explicitly defined smoothed Bernoulli rule. It does not establish a modulus for arbitrary validity sets.".into()),
            ("full_transition_local_tv_constant_C_N".into(), "The chapter gives existence and a mixture formula, not a universal numerical C_N. Finite atomic couplings below test the stated TV-to-W2 inequalities only.".into()),
            ("interacting_swarm_concentration".into(), "Independent-noise fixtures freeze their inputs. Shared component rotations or feedback require their own conditional argument.".into()),
        ]),
        scope: vec![
            "Native Gaussian, standardized-box and reference-ball innovations are identified separately. No uniform-ball claim is applied to the standardized-box provider.".into(),
            "Six-standard-error moment bands are diagnostic acceptance bands. They are not simultaneous rigorous confidence certificates.".into(),
            "Exact binomial tails are finite independent-innovation fixtures; native repetitions compare with those laws using a stated Hoeffding margin under the ideal independent-stream model.".into(),
            "Finite fixtures validate source algebra and the selected native primitives, not every global analytic hypothesis or long-time complete-swarm convergence rate.".into(),
        ],
    };
    displacement_cases(&mut report)?;
    uniform_support_cases(&mut report)?;
    single_walker_distance_cases(&mut report)?;
    transport_cases(&mut report)?;
    perturbation_tail_cases(&mut report)?;
    status_cases(&mut report)?;
    cloning_output_cases(&mut report)?;
    futures_lite::future::block_on(noise_cases(&mut report))?;
    Ok(report)
}

fn single_walker_distance_cases(report: &mut ContinuityReport) -> Result<()> {
    for n in [3, 5, 8] {
        for d in [1, 2] {
            let a: Vec<f64> = (0..n * d).map(|i| 0.8 * (i as f64 * 1.7).sin()).collect();
            let b: Vec<f64> = a
                .iter()
                .enumerate()
                .map(|(i, x)| (x + 0.04 * (i as f64 * 0.9).cos()).clamp(-1., 1.))
                .collect();
            for squashed in [false, true] {
                let x = if squashed {
                    squashed_observations(a.clone(), d)?
                } else {
                    observations(a.clone(), d)?
                };
                let original_y = if squashed {
                    squashed_observations(b.clone(), d)?
                } else {
                    observations(b.clone(), d)?
                };
                let metric = if squashed {
                    Distance::SquashedPhaseSpace {
                        positions: "positions".into(),
                        velocities: "velocities".into(),
                        position_radius: 1.,
                        velocity_radius: 1.,
                        lambda: 1.,
                    }
                } else {
                    Distance::default()
                };
                let diameter = if squashed {
                    2. * 2_f64.sqrt()
                } else {
                    2. * (d as f64).sqrt()
                };
                for survivors in [1, n - 1, n] {
                    let original_marks: Vec<bool> = (0..n).map(|i| i < survivors).collect();
                    let optimal = optimal_swarm_displacement(
                        &metric,
                        &x,
                        &original_y,
                        &vec![true; n],
                        &original_marks,
                        1.,
                    )?;
                    let permutation = optimal
                        .transport_plan
                        .iter()
                        .map(|row| {
                            let indices: Vec<usize> = row
                                .iter()
                                .enumerate()
                                .filter_map(|(j, p)| (*p > 1e-10).then_some(j))
                                .collect();
                            require(
                                indices.len() == 1,
                                "uniform finite fixture must return an integral matching",
                            )?;
                            Ok(indices[0] as u32)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let y = original_y.gather(&permutation)?;
                    let marks: Vec<bool> = permutation
                        .iter()
                        .map(|j| original_marks[*j as usize])
                        .collect();
                    let nc = (n - survivors) as f64;
                    let denominator = (n - 1) as f64;
                    let displacements = (0..n)
                        .map(|i| metric.compare(&x, i, &y, i))
                        .collect::<Result<Vec<_>>>()?;
                    let delta_squared = displacements.iter().map(|v| v * v).sum::<f64>();
                    let scope = format!(
                        "Native {} comparison, N={n}, d={d}, k1={n}, k2={survivors}; uniform nonself donor law (sole survivor uses itself); second atoms reordered by optimal marked empirical transport; positional comparison fixes first donor law, structural comparison fixes second geometry.",
                        if squashed {
                            "squashed Sasaki lambda=1"
                        } else {
                            "Euclidean on [-1,1]^d"
                        }
                    );
                    let mut positional_sum = 0.;
                    let mut structural_sum = 0.;
                    let mut stable_sum = 0.;
                    let mut unstable_sum = 0.;
                    let mut total_sum = 0.;
                    for i in 0..n {
                        let initial_donors: Vec<usize> = (0..n).filter(|j| *j != i).collect();
                        let mean = |obs: &ObservationBatch<f64>, donors: &[usize]| -> Result<f64> {
                            Ok(donors
                                .iter()
                                .map(|j| metric.compare(obs, i, obs, *j))
                                .collect::<Result<Vec<_>>>()?
                                .iter()
                                .sum::<f64>()
                                / donors.len() as f64)
                        };
                        let first = mean(&x, &initial_donors)?;
                        let fixed_law_second = mean(&y, &initial_donors)?;
                        let positional = (first - fixed_law_second).abs();
                        let mut positional_labels = vec!["lem-single-walker-positional-error"];
                        if squashed {
                            positional_labels.push("lem-sasaki-single-walker-positional-error");
                        }
                        check(
                            report,
                            "single_walker_fixed_law_positional_error",
                            &positional_labels,
                            &scope,
                            positional,
                            displacements[i]
                                + initial_donors
                                    .iter()
                                    .map(|j| displacements[*j])
                                    .sum::<f64>()
                                    / denominator,
                        );
                        let second = if marks[i] {
                            let donors: Vec<usize> = if survivors == 1 {
                                vec![i]
                            } else {
                                (0..n).filter(|j| *j != i && marks[*j]).collect()
                            };
                            mean(&y, &donors)?
                        } else {
                            0.
                        };
                        let error_squared = (first - second).powi(2);
                        total_sum += error_squared;
                        if marks[i] {
                            let structural = (fixed_law_second - second).abs();
                            positional_sum += positional.powi(2);
                            structural_sum += structural.powi(2);
                            stable_sum += error_squared;
                            let mut labels = vec!["lem-single-walker-structural-error"];
                            if squashed {
                                labels.push("lem-sasaki-single-walker-structural-error");
                            }
                            check(
                                report,
                                "single_walker_changed_support_structural_error",
                                &labels,
                                &scope,
                                structural,
                                2. * diameter * nc / denominator,
                            );
                            check(
                                report,
                                "single_walker_stable_two_component_bound",
                                &["lem-sub-stable-walker-error-decomposition"],
                                &scope,
                                error_squared,
                                2. * (positional.powi(2) + structural.powi(2)),
                            );
                        } else {
                            unstable_sum += error_squared;
                            check(
                                report,
                                "single_walker_own_status_error",
                                &["lem-single-walker-own-status-error"],
                                &scope,
                                (first - second).abs(),
                                diameter,
                            );
                        }
                    }
                    check(
                        report,
                        "stable_distance_positional_sum",
                        &["lem-sub-stable-positional-error-bound"],
                        &scope,
                        positional_sum,
                        6. * delta_squared,
                    );
                    check(
                        report,
                        "stable_distance_structural_sum",
                        &["lem-sub-stable-structural-error-bound"],
                        &scope,
                        structural_sum,
                        4. * n as f64 * diameter.powi(2) * nc.powi(2) / denominator.powi(2),
                    );
                    check(
                        report,
                        "stable_distance_full_sum",
                        &["lem-total-squared-error-stable"],
                        &scope,
                        stable_sum,
                        12. * delta_squared
                            + 8. * n as f64 * diameter.powi(2) * nc.powi(2) / denominator.powi(2),
                    );
                    check(
                        report,
                        "unstable_distance_full_sum",
                        &["lem-total-squared-error-unstable"],
                        &scope,
                        unstable_sum,
                        diameter.powi(2) * nc,
                    );
                    check(
                        report,
                        "expected_distance_stable_unstable_identity",
                        &["thm-total-expected-distance-error-decomposition"],
                        &scope,
                        (total_sum - stable_sum - unstable_sum).abs(),
                        0.,
                    );
                    if squashed {
                        let cpos = 2. * (1. + survivors as f64 / denominator);
                        constant(
                            report,
                            "sasaki_stable_C_pos",
                            cpos,
                            "2(1+k_stable/max(1,k1-1))",
                            &["lem-sasaki-total-squared-error-stable"],
                            &scope,
                        );
                        check(
                            report,
                            "sasaki_fixed_law_stable_positional_sum",
                            &["lem-sasaki-total-squared-error-stable"],
                            &scope,
                            positional_sum,
                            cpos * delta_squared,
                        );
                        check(
                            report,
                            "sasaki_stable_full_sum_with_support_change",
                            &["lem-sasaki-total-squared-error-stable"],
                            &scope,
                            stable_sum,
                            2. * cpos * delta_squared
                                + 8. * survivors as f64 * diameter.powi(2) * nc.powi(2)
                                    / denominator.powi(2),
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

fn displacement_cases(report: &mut ContinuityReport) -> Result<()> {
    for n in [1, 2, 8, 32] {
        for d in [1, 2, 5] {
            let mut rng = RandomStream::new(1917, n as u64, Stream::Initialize, d as u64, 0);
            let a: Vec<f64> = (0..n * d).map(|_| 4. * rng.uniform::<f64>() - 2.).collect();
            let b: Vec<f64> = a
                .iter()
                .enumerate()
                .map(|(i, x)| x + 0.03 * (i as f64).sin())
                .collect();
            let pa: Vec<f64> = a.iter().map(|x| x + 0.2 * rng.gaussian::<f64>()).collect();
            let pb: Vec<f64> = b.iter().map(|x| x + 0.2 * rng.gaussian::<f64>()).collect();
            for squashed in [false, true] {
                let observe = |x: Vec<f64>| {
                    if squashed {
                        squashed_observations(x, d)
                    } else {
                        observations(x, d)
                    }
                };
                let x = observe(a.clone())?;
                let y = observe(b.clone())?;
                let xp = observe(pa.clone())?;
                let yp = observe(pb.clone())?;
                let metric = if squashed {
                    squashed_metric()
                } else {
                    Distance::default()
                };
                let delta = optimal_position_sum(&metric, &x, &y)?;
                let da = optimal_position_sum(&metric, &xp, &x)?;
                let db = optimal_position_sum(&metric, &yp, &y)?;
                let output = optimal_position_sum(&metric, &xp, &yp)?;
                let scope = format!(
                    "native {} distance; N={n}, d={d}; optimal uniform empirical transport at every segment; storage rows only enumerate atoms",
                    if squashed { "squashed" } else { "Euclidean" }
                );
                check(
                    report,
                    "perturbation_three_segment_displacement",
                    &[
                        "lem-sub-perturbation-positional-bound-reproof",
                        "def-displacement-components",
                    ],
                    &scope,
                    output,
                    3. * (delta + da + db),
                );
                let left_alive = vec![true; n];
                let right_alive: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
                for penalty in [0.1_f64, 1., 4.] {
                    let optimal = optimal_swarm_displacement(
                        &metric,
                        &x,
                        &y,
                        &left_alive,
                        &right_alive,
                        penalty,
                    )?;
                    let v = optimal.metric_squared;
                    let changed = optimal.status_mismatch_count;
                    let marked_scope = format!(
                        "{scope}; first swarm all alive, second has fixture mark mask, so mismatch count is independent of pairing; lambda_status={penalty}"
                    );
                    check(
                        report,
                        "optimal_marked_displacement_decomposition",
                        &[
                            "def-n-particle-displacement-metric",
                            "def-displacement-components",
                        ],
                        &marked_scope,
                        (v - (optimal.positional_sum + penalty * changed) / n as f64).abs(),
                        0.,
                    );
                    check(
                        report,
                        "displacement_positional_component",
                        &[
                            "def-n-particle-displacement-metric",
                            "def-displacement-components",
                        ],
                        &marked_scope,
                        optimal.positional_sum,
                        n as f64 * v,
                    );
                    check(
                        report,
                        "displacement_status_component",
                        &[
                            "def-n-particle-displacement-metric",
                            "def-displacement-components",
                        ],
                        &marked_scope,
                        changed,
                        n as f64 / penalty * v,
                    );
                    check(
                        report,
                        "displacement_quadratic_status_component",
                        &["lem-sub-bound-sum-total-cloning-probs"],
                        &marked_scope,
                        changed * changed,
                        (n as f64 / penalty * v).powi(2),
                    );
                }
                if !squashed {
                    let mean_difference_squared = (0..d)
                        .map(|j| {
                            ((0..n).map(|i| a[i * d + j] - b[i * d + j]).sum::<f64>() / n as f64)
                                .powi(2)
                        })
                        .sum::<f64>();
                    check(
                        report,
                        "empirical_mean_displacement",
                        &[
                            "lem-inequality-toolbox",
                            "def-n-particle-displacement-metric",
                        ],
                        &scope,
                        mean_difference_squared,
                        delta / n as f64,
                    );
                }
            }
        }
    }
    // This is a quotient example: lambda=0 deliberately discards velocity.
    let a = squashed_observations(vec![0., 1.], 1)?;
    let mut b = a.clone();
    b.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(2, 1, vec![1e3, -1e3])?,
    );
    let metric = squashed_metric();
    check(
        report,
        "quotient_discards_zero_weight_feature",
        &["def-metric-quotient"],
        "native squashed phase-space with lambda=0 and equal positions",
        metric.compare(&a, 0, &b, 0)?.powi(2) + metric.compare(&a, 1, &b, 1)?.powi(2),
        0.,
    );
    let a = observations(vec![0., 1., 2.], 1)?;
    let b = observations(vec![2., 0., 1.], 1)?;
    let transport =
        optimal_swarm_displacement(&Distance::default(), &a, &b, &[true; 3], &[true; 3], 1.)?;
    constant(
        report,
        "candidate_coupling_cost",
        transport.candidate_coupling_cost,
        "sum_i |x_i-y_i|²/N under the prescribed identity-array coupling",
        &["def-n-particle-displacement-metric"],
        "permutation fixture [0,1,2] versus [2,0,1]; admissible coupling upper bound 2, never the physical swarm distance",
    );
    constant(
        report,
        "permuted_empirical_swarm_displacement_squared",
        transport.metric_squared,
        "minimum marked empirical transport cost",
        &["def-n-particle-displacement-metric"],
        "permutation copy of the same empirical swarm has physical displacement zero",
    );
    check(
        report,
        "permutation_copy_has_zero_swarm_displacement",
        &["def-n-particle-displacement-metric", "def-metric-quotient"],
        "independent storage permutation of one physical empirical swarm",
        transport.metric_squared,
        0.,
    );
    check(
        report,
        "optimal_empirical_displacement_below_candidate_coupling",
        &[
            "def-wasserstein-distance",
            "def-n-particle-displacement-metric",
        ],
        "optimal marked empirical transport; prescribed array pairing is only an admissible upper bound",
        transport.metric_squared,
        transport.candidate_coupling_cost,
    );
    Ok(())
}

fn uniform_support_cases(report: &mut ContinuityReport) -> Result<()> {
    for n in 1..=5 {
        let families: Vec<Vec<f64>> = vec![
            vec![1.; n],
            (0..n).map(|i| if i % 2 == 0 { -1. } else { 1. }).collect(),
            (0..n).map(|i| (i as f64 + 0.3).sin()).collect(),
        ];
        for (family, values) in families.iter().enumerate() {
            let maximum = values.iter().map(|v| v.abs()).fold(0_f64, f64::max);
            // Native Uniform.log_weight defines the actual constant weights.
            let weights = values
                .iter()
                .map(|v| {
                    Kernel::Uniform
                        .log_weight(v.abs(), ComparisonKind::Distance)
                        .map(f64::exp)
                })
                .collect::<Result<Vec<_>>>()?;
            for mask1 in 1_u32..(1 << n) {
                let support1: Vec<usize> = (0..n).filter(|i| mask1 & (1 << i) != 0).collect();
                for mask2 in 1_u32..(1 << n) {
                    let support2: Vec<usize> = (0..n).filter(|i| mask2 & (1 << i) != 0).collect();
                    let k1 = support1.len() as f64;
                    let k2 = support2.len() as f64;
                    let sum1: f64 = support1.iter().map(|i| values[*i] * weights[*i]).sum();
                    let sum2: f64 = support2.iter().map(|i| values[*i] * weights[*i]).sum();
                    let symmetric_difference = (mask1 ^ mask2).count_ones() as f64;
                    let intersection = (mask1 & mask2).count_ones() as f64;
                    let mut total_variation = 0.;
                    for (i, weight) in weights.iter().enumerate() {
                        let p = if mask1 & (1 << i) != 0 {
                            *weight / k1
                        } else {
                            0.
                        };
                        let q = if mask2 & (1 << i) != 0 {
                            *weight / k2
                        } else {
                            0.
                        };
                        total_variation += 0.5 * (p - q).abs();
                    }
                    let scope = format!(
                        "native uniform kernel; nonempty arbitrary supports {mask1:b}/{mask2:b}; N={n}, values={family}; no Gaussian reweighting or singleton convention change"
                    );
                    check(
                        report,
                        "uniform_set_difference",
                        &["lem-set-difference-bound"],
                        &scope,
                        ((sum1 - sum2) / k1).abs(),
                        maximum / k1 * symmetric_difference,
                    );
                    check(
                        report,
                        "uniform_normalization_difference",
                        &["lem-normalization-difference-bound"],
                        &scope,
                        ((1. / k1 - 1. / k2) * sum2).abs(),
                        maximum / k1 * (k1 - k2).abs(),
                    );
                    check(
                        report,
                        "uniform_total_structural_error",
                        &["thm-total-error-status-bound"],
                        &scope,
                        (sum1 / k1 - sum2 / k2).abs(),
                        2. * maximum / k1 * symmetric_difference,
                    );
                    check(
                        report,
                        "uniform_tv_common_mass_identity",
                        &["prop-w2-bound-no-offset", "def-companion-selection-measure"],
                        &scope,
                        (total_variation - (1. - intersection / k1.max(k2))).abs(),
                        0.,
                    );
                    check(
                        report,
                        "uniform_tv_support_change_bound",
                        &["thm-total-error-status-bound"],
                        &scope,
                        total_variation,
                        symmetric_difference / k1,
                    );
                    check(
                        report,
                        "uniform_bounded_observable_tv_bound",
                        &["thm-total-error-status-bound"],
                        &scope,
                        (sum1 / k1 - sum2 / k2).abs(),
                        2. * maximum * total_variation,
                    );
                }
            }
        }
    }
    Ok(())
}

fn transport_cases(report: &mut ContinuityReport) -> Result<()> {
    let points = [-10., -1., 0., 0.2, 8.];
    let measures = [
        [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 0.],
        [0.2; 5],
        [0.1, 0.2, 0.4, 0.2, 0.1],
        [0.1, 0.3, 0.3, 0.2, 0.1],
        [0., 0., 1. - 1e-9, 1e-9, 0.],
        [0.5, 0., 0., 0., 0.5],
    ];
    for (i, mu) in measures.iter().enumerate() {
        for (j, nu) in measures.iter().enumerate() {
            let transport = finite_line_transport(&points, mu, nu)?;
            let scope = format!(
                "exact 1D atomic optimal and common-part couplings; pair={i}/{j}; native Euclidean costs; o=0"
            );
            check(
                report,
                "finite_transport_marginals",
                &["def-wasserstein-distance", "def-w2-output-metric"],
                &scope,
                transport.maximum_marginal_residual,
                0.,
            );
            check(
                report,
                "optimal_transport_below_constructed_coupling",
                &["def-wasserstein-distance", "prop-w2-bound-no-offset"],
                &scope,
                transport.wasserstein_squared,
                transport.common_part_coupling_cost,
            );
            check(
                report,
                "common_coupling_bounded_diameter",
                &["prop-w2-bound-no-offset"],
                &scope,
                transport.common_part_coupling_cost,
                transport.diameter_squared * transport.total_variation,
            );
            check(
                report,
                "common_coupling_fourth_moment",
                &["prop-w2-bound-no-offset"],
                &scope,
                transport.common_part_coupling_cost,
                (8. * (transport.fourth_moment_mu + transport.fourth_moment_nu)
                    * transport.total_variation)
                    .sqrt(),
            );
            constant(
                report,
                &format!("finite_w2_squared_{i}_{j}"),
                transport.wasserstein_squared,
                "monotone quantile transport sum p_ij |x_i-x_j|²",
                &["def-wasserstein-distance", "def-w2-output-metric"],
                &scope,
            );
            constant(
                report,
                &format!("finite_tv_{i}_{j}"),
                transport.total_variation,
                "0.5 sum |mu_i-nu_i| = 1-common_mass",
                &["prop-w2-bound-no-offset"],
                &scope,
            );
        }
    }
    Ok(())
}

fn binomial_probabilities(n: usize, p: f64) -> Vec<f64> {
    let mut probabilities = Vec::with_capacity(n + 1);
    probabilities.push((1. - p).powi(n as i32));
    for k in 0..n {
        probabilities.push(probabilities[k] * (n - k) as f64 / (k + 1) as f64 * p / (1. - p));
    }
    probabilities
}

/// Scalar empirical-law transport, normalized to probability mass one.
/// Monotone quantile coupling is optimal for squared distance on the line.
/// Native scalar distance is evaluated on every segment of the coupling.
pub fn scalar_empirical_transport_squared(left: &[f64], right: &[f64]) -> Result<f64> {
    require(
        !left.is_empty() && !right.is_empty(),
        "nonempty empirical laws required",
    )?;
    weighted_scalar_transport_squared(
        left,
        &vec![1. / left.len() as f64; left.len()],
        right,
        &vec![1. / right.len() as f64; right.len()],
    )
}

fn weighted_scalar_transport_squared(
    left: &[f64],
    mu: &[f64],
    right: &[f64],
    nu: &[f64],
) -> Result<f64> {
    let sorted = |points: &[f64], weights: &[f64]| -> Result<Vec<(f64, f64)>> {
        require(
            !points.is_empty()
                && points.len() <= 256
                && points.len() == weights.len()
                && points.iter().all(|p| p.is_finite())
                && weights.iter().all(|p| p.is_finite() && *p >= 0.),
            "invalid scalar transport law",
        )?;
        let total = weights.iter().sum::<f64>();
        require(
            (total - 1.).abs() <= 1e-10,
            "scalar law must have unit mass",
        )?;
        let mut atoms: Vec<(f64, f64)> = points
            .iter()
            .copied()
            .zip(weights.iter().map(|p| p / total))
            .filter(|(_, p)| *p > 0.)
            .collect();
        atoms.sort_by(|(a, _), (b, _)| a.total_cmp(b));
        Ok(atoms)
    };
    let l = sorted(left, mu)?;
    let r = sorted(right, nu)?;
    let lo = observations(l.iter().map(|(x, _)| *x).collect(), 1)?;
    let ro = observations(r.iter().map(|(x, _)| *x).collect(), 1)?;
    let metric = Distance::default();
    let mut i = 0;
    let mut j = 0;
    let mut t = 0.;
    let mut end_l = l[0].1;
    let mut end_r = r[0].1;
    let mut cost = 0.;
    while i < l.len() && j < r.len() {
        let end = end_l.min(end_r);
        cost += (end - t).max(0.) * metric.compare(&lo, i, &ro, j)?.powi(2);
        t = end;
        let advance_l = end_l <= end_r;
        let advance_r = end_r <= end_l;
        if advance_l {
            i += 1;
            if i < l.len() {
                end_l = if i + 1 == l.len() { 1. } else { end_l + l[i].1 };
            }
        }
        if advance_r {
            j += 1;
            if j < r.len() {
                end_r = if j + 1 == r.len() { 1. } else { end_r + r[j].1 };
            }
        }
    }
    require(cost.is_finite(), "scalar transport cost overflowed")?;
    Ok(cost)
}

fn population_check(
    report: &mut PopulationScalingReport,
    id: &str,
    labels: &[&str],
    scope: &str,
    observed: f64,
    bound: f64,
) {
    report
        .checks
        .push(BoundCheck::upper(id, labels, scope, observed, bound));
}

fn population_constant(
    report: &mut PopulationScalingReport,
    name: &str,
    value: f64,
    formula: &str,
    labels: &[&str],
    scope: &str,
) {
    report.computed.push(ComputedConstant {
        name: name.into(),
        value,
        formula: formula.into(),
        source_labels: labels.iter().map(|s| (*s).into()).collect(),
        scope: scope.into(),
    });
}

fn empirical_standardization_uniform_cases(report: &mut PopulationScalingReport) -> Result<()> {
    let auxiliary_patch = crate::convergence_framework::hermite_patch(2.)?;
    let [a, b, c, d] = auxiliary_patch.coefficients;
    let rescale = |z: f64| {
        if z <= 0. {
            z.exp()
        } else if z < 1. {
            z.ln_1p() + 1.
        } else if z <= 2. {
            let s = z - 1.;
            ((a * s + b) * s + c) * s + d
        } else {
            2_f64.ln_1p() + 1.
        }
    };
    for n in [4, 8, 16, 32, 64] {
        let obs = observations(vec![0.; n], 1)?;
        for floor in [0.01_f64, 0.1, 1.] {
            let native = Standardizer::Global { sigma_min: floor };
            for profile in 0..4 {
                let raw: Vec<f64> = (0..n)
                    .map(|i| match profile {
                        0 => 7.,
                        1 => 1e-100 * (i as f64).sin(),
                        2 => (i as f64 * 1.7).sin(),
                        _ => 1e100 * (i as f64 * 1.7).sin(),
                    })
                    .collect();
                let changed: Vec<f64> = raw
                    .iter()
                    .enumerate()
                    .map(|(i, x)| match profile {
                        0 => -3.,
                        1 => -x,
                        2 => x + 0.2 * (i as f64 * 0.4).cos(),
                        _ => -0.7 * x,
                    })
                    .collect();
                let scope = format!(
                    "Native Global standardizer, N={n}, m={floor}, profile={profile}; arbitrary finite scalar values, including collapsed and extreme clouds; no V_max hypothesis. Supports nonempty. Law transport is scalar empirical transport, not joint physical-state transport."
                );
                for k1 in [1, n / 2, n] {
                    let alive1: Vec<bool> = (0..n).map(|i| i < k1).collect();
                    let (z1, stats) = native.apply(&raw, &alive1, &obs)?;
                    let (same_support_z2, _) = native.apply(&changed, &alive1, &obs)?;
                    let input_pair = raw
                        .iter()
                        .zip(&changed)
                        .zip(&alive1)
                        .filter(|(_, a)| **a)
                        .map(|((a, b), _)| (a - b).powi(2))
                        .sum::<f64>()
                        / n as f64;
                    let output_pair = z1
                        .iter()
                        .zip(&same_support_z2)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / n as f64;
                    population_check(
                        report,
                        "empirical_fixed_support_sharp_lipschitz",
                        &[
                            "lem-empirical-standardization-uniform",
                            "cor-empirical-standardization-uniform-continuity",
                        ],
                        &scope,
                        output_pair,
                        input_pair / floor.powi(2),
                    );
                    let rescaled_fixed_pair = z1
                        .iter()
                        .zip(&same_support_z2)
                        .zip(&alive1)
                        .filter(|(_, alive)| **alive)
                        .map(|((left, right), _)| (rescale(*left) - rescale(*right)).powi(2))
                        .sum::<f64>()
                        / n as f64;
                    population_check(
                        report,
                        "empirical_complete_patched_rescale_fixed_support",
                        &[
                            "lem-lipschitz-constant-of-the-patched-standardization",
                            "cor-closed-form-lipschitz-composite",
                        ],
                        &format!(
                            "{scope} Auxiliary piecewise rescale z_max=2, exact L_gA={}; applies to complete native scores on alive rows only, preserving native logistic configuration.",
                            auxiliary_patch.exact_rescale_lipschitz
                        ),
                        rescaled_fixed_pair,
                        auxiliary_patch.exact_rescale_lipschitz.powi(2) * input_pair
                            / floor.powi(2),
                    );
                    let scale = stats.scale[0];
                    population_check(
                        report,
                        "empirical_radial_jacobian_eigenvalue",
                        &["lem-empirical-standardization-uniform"],
                        &scope,
                        (floor / scale).powi(2) / scale,
                        1. / floor,
                    );
                    population_check(
                        report,
                        "empirical_tangential_jacobian_eigenvalue",
                        &["lem-empirical-standardization-uniform"],
                        &scope,
                        1. / scale,
                        1. / floor,
                    );
                    let norm1 = z1.iter().map(|x| x * x).sum::<f64>() / n as f64;
                    population_check(
                        report,
                        "empirical_support_fraction_norm_bound",
                        &["lem-empirical-standardization-uniform"],
                        &scope,
                        norm1,
                        k1 as f64 / n as f64,
                    );
                    for k2 in [1, n / 2, n] {
                        let alive2: Vec<bool> = (0..n).map(|i| i < k2).collect();
                        let (z2, _) = native.apply(&changed, &alive2, &obs)?;
                        let indexed = z1
                            .iter()
                            .zip(&z2)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            / n as f64;
                        let support_bound =
                            ((k1 as f64 / n as f64).sqrt() + (k2 as f64 / n as f64).sqrt()).powi(2);
                        population_check(
                            report,
                            "empirical_arbitrary_support_uniform_four",
                            &["lem-empirical-standardization-uniform"],
                            &scope,
                            indexed,
                            support_bound,
                        );
                        population_check(
                            report,
                            "empirical_arbitrary_support_bound_below_four",
                            &["lem-empirical-standardization-uniform"],
                            &scope,
                            support_bound,
                            4.,
                        );
                        let raw1: Vec<f64> = raw
                            .iter()
                            .zip(&alive1)
                            .filter_map(|(v, a)| a.then_some(*v))
                            .collect();
                        let raw2: Vec<f64> = changed
                            .iter()
                            .zip(&alive2)
                            .filter_map(|(v, a)| a.then_some(*v))
                            .collect();
                        let out1: Vec<f64> = z1
                            .iter()
                            .zip(&alive1)
                            .filter_map(|(v, a)| a.then_some(*v))
                            .collect();
                        let out2: Vec<f64> = z2
                            .iter()
                            .zip(&alive2)
                            .filter_map(|(v, a)| a.then_some(*v))
                            .collect();
                        let raw_w2 = scalar_empirical_transport_squared(&raw1, &raw2)?;
                        let output_w2 = scalar_empirical_transport_squared(&out1, &out2)?;
                        population_check(
                            report,
                            "empirical_unequal_alive_law_sharp_lipschitz",
                            &["cor-empirical-standardization-uniform-continuity"],
                            &scope,
                            output_w2,
                            raw_w2 / floor.powi(2),
                        );
                        population_check(
                            report,
                            "empirical_unequal_alive_law_uniform_four",
                            &["cor-empirical-standardization-uniform-continuity"],
                            &scope,
                            output_w2,
                            4.,
                        );
                        let rescaled1: Vec<f64> = out1.iter().map(|z| rescale(*z)).collect();
                        let rescaled2: Vec<f64> = out2.iter().map(|z| rescale(*z)).collect();
                        let rescaled_law =
                            scalar_empirical_transport_squared(&rescaled1, &rescaled2)?;
                        population_check(
                            report,
                            "empirical_complete_patched_rescale_law_transport",
                            &[
                                "lem-lipschitz-constant-of-the-patched-standardization",
                                "cor-closed-form-lipschitz-composite",
                            ],
                            &format!(
                                "{scope} Auxiliary piecewise rescale z_max=2 after native Global standardization, actual unequal alive empirical probabilities."
                            ),
                            rescaled_law,
                            auxiliary_patch.exact_rescale_lipschitz.powi(2) * raw_w2
                                / floor.powi(2),
                        );
                        // Independent storage permutations transport both values and marks.
                        let left_order: Vec<usize> = (0..n).rev().collect();
                        let right_order: Vec<usize> = (0..n).map(|i| (i + 3) % n).collect();
                        let reorder =
                            |values: &[f64], marks: &[bool], order: &[usize]| -> Result<Vec<f64>> {
                                let v: Vec<f64> = order.iter().map(|i| values[*i]).collect();
                                let m: Vec<bool> = order.iter().map(|i| marks[*i]).collect();
                                let (z, _) = native.apply(&v, &m, &obs)?;
                                Ok(z.into_iter()
                                    .zip(m)
                                    .filter_map(|(v, a)| a.then_some(v))
                                    .collect())
                            };
                        let reordered_cost = scalar_empirical_transport_squared(
                            &reorder(&raw, &alive1, &left_order)?,
                            &reorder(&changed, &alive2, &right_order)?,
                        )?;
                        population_check(
                            report,
                            "empirical_standardized_law_permutation_invariance",
                            &["cor-empirical-standardization-uniform-continuity"],
                            &scope,
                            (reordered_cost - output_w2).abs(),
                            0.,
                        );
                    }
                }
                population_constant(
                    report,
                    &format!("empirical_N{n}_m{floor}_profile{profile}_lipschitz_squared"),
                    floor.powi(-2),
                    "m^-2, independent of N and raw value magnitude",
                    &[
                        "lem-empirical-standardization-uniform",
                        "cor-empirical-standardization-uniform-continuity",
                    ],
                    &scope,
                );
                population_constant(
                    report,
                    &format!("empirical_N{n}_m{floor}_profile{profile}_universal_bound"),
                    4.,
                    "4, independent of N, support fractions and raw value magnitude",
                    &[
                        "lem-empirical-standardization-uniform",
                        "cor-empirical-standardization-uniform-continuity",
                    ],
                    &scope,
                );
            }
        }
    }
    for (left, right) in [
        (vec![0., 1.], vec![-0.2, 0.4, 1.3]),
        (vec![0., 0., 1.], vec![0., 1.]),
    ] {
        let (_, plan) = uniform_transport(
            &left.iter().map(|x| vec![*x]).collect::<Vec<_>>(),
            &right.iter().map(|x| vec![*x]).collect::<Vec<_>>(),
            |a, b| (a[0] - b[0]).powi(2),
        )?;
        let left_values = &left;
        let right_values = &right;
        let generic_cost = plan
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(j, p)| p * (left_values[i] - right_values[j]).powi(2))
            })
            .sum::<f64>();
        let quantile_cost = scalar_empirical_transport_squared(&left, &right)?;
        population_check(
            report,
            "scalar_unequal_empirical_quantile_vs_general_transport",
            &["cor-empirical-standardization-uniform-continuity"],
            "Explicit 2-vs-3 and 3-vs-2 scalar empirical laws; monotone native-distance cost agrees with existing general finite transport solver.",
            (quantile_cost - generic_cost).abs(),
            0.,
        );
    }
    Ok(())
}

/// Exact empirical-law costs and native independent-companion repetitions.
/// Scalar raw and standardized laws are the error target; summed indexed
/// measurement errors are retained only inside the separate noise baseline.
pub fn population_scaling_validation_cases() -> Result<PopulationScalingReport> {
    futures_lite::future::block_on(population_scaling_cases())
}

async fn population_scaling_cases() -> Result<PopulationScalingReport> {
    let mut report=PopulationScalingReport {
        cases:vec![],checks:vec![],computed:vec![],
        scope:"Probability-normalized scalar empirical W2² after native Global standardization. Exact Binomial count-pair laws and actual independent native donor draws on frozen balanced binary physical clouds. This is a scalar marginal law target; no joint physical-state matching, fitness contraction or completed-update convergence is inferred.".into(),
    };
    empirical_standardization_uniform_cases(&mut report)?;
    let repetitions = 2048;
    let failure_probability: f64 = 0.001;
    let base_margin = ((2. / failure_probability).ln() / (2. * repetitions as f64)).sqrt();
    let floor: f64 = 0.1;
    let native_standardizer = Standardizer::Global { sigma_min: floor };
    let canonical = algorithmic_gas::GasConfig::euclidean(1, 0.04)?;
    let donor_laws = [
        (
            "uniform",
            DonorModule {
                kernel: Kernel::Uniform,
                ..Default::default()
            },
        ),
        ("canonical_gaussian_width_2", canonical.distance_donors),
    ];
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    for (law, native_donors) in donor_laws {
        for n in [4, 8, 16, 32, 64] {
            let obs =
                squashed_observations((0..n).map(|i| u8::from(i >= n / 2) as f64).collect(), 1)?;
            let population = Population::new(obs)?;
            let pool = DonorPool::freeze(&population, 0, &[], 0, false)?;
            let alive = vec![true; n];
            let metric = &native_donors.distance;
            let diameter =
                metric.compare(&population.observations, 0, &population.observations, n - 1)?;
            let cross_weight = native_donors
                .kernel
                .log_weight(diameter, ComparisonKind::Distance)?
                .exp();
            let self_cluster_weight = native_donors
                .kernel
                .log_weight(0.0_f64, ComparisonKind::Distance)?
                .exp();
            let q = (n / 2) as f64 * cross_weight
                / (((n / 2 - 1) as f64) * self_cluster_weight + (n / 2) as f64 * cross_weight);
            let scope = format!(
                "{} N=k={n}, m={floor}, native donor law={law}, raw support[0,{diameter}], q={q}. Gaussian donor configuration is copied from GasConfig::euclidean(1,.04): width2, actual squashed phase-space comparison. Raw distance is measured before separation-floor regularization. Mixture reference law uses exact conditional native kernel weights. All finite stage law estimates assume ideal independent addressed random choices.",
                report.scope
            );
            let mut raw_variance_sum = 0.;
            for i in 0..n {
                let mut mean = 0.;
                let mut second = 0.;
                let mut mass = 0.;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let value =
                        metric.compare(&population.observations, i, &population.observations, j)?;
                    let weight = native_donors
                        .kernel
                        .log_weight(value, ComparisonKind::Distance)?
                        .exp();
                    mass += weight;
                    mean += weight * value;
                    second += weight * value * value;
                }
                mean /= mass;
                second /= mass;
                raw_variance_sum += second - mean * mean;
                population_check(
                    &mut report,
                    "population_native_raw_mean_identity",
                    &["def-distance-to-companion-measurement"],
                    &scope,
                    (mean - diameter * q).abs(),
                    0.,
                );
                population_check(
                    &mut report,
                    "population_native_raw_variance_identity",
                    &["axiom-bounded-measurement-variance"],
                    &scope,
                    ((second - mean * mean) - diameter.powi(2) * q * (1. - q)).abs(),
                    0.,
                );
            }
            let probabilities = binomial_probabilities(n, q);
            population_check(
                &mut report,
                "population_binomial_count_law_normalization",
                &["def-distance-to-companion-measurement"],
                &scope,
                (probabilities.iter().sum::<f64>() - 1.).abs(),
                0.,
            );
            let mixture_variance = diameter.powi(2) * q * (1. - q);
            let mixture_scale = (mixture_variance + floor.powi(2)).sqrt();
            let mixture_raw = [0., diameter];
            let mixture_z = [
                -diameter * q / mixture_scale,
                diameter * (1. - q) / mixture_scale,
            ];
            let mixture_weights = [1. - q, q];
            let mut raw_counts = vec![];
            let mut z_counts = vec![];
            let mut expected_norm = 0.;
            let mut exact_raw_to_mixture = 0.;
            let mut exact_z_to_mixture = 0.;
            for (count, probability) in probabilities.iter().enumerate() {
                let raw: Vec<f64> = (0..n)
                    .map(|i| if i < count { diameter } else { 0. })
                    .collect();
                let (z, _) = native_standardizer.apply(&raw, &alive, &population.observations)?;
                let norm = z.iter().map(|v| v * v).sum::<f64>();
                let variance =
                    diameter.powi(2) * (count as f64 / n as f64) * (1. - count as f64 / n as f64);
                population_check(
                    &mut report,
                    "population_native_standardization_count_norm",
                    &["lem-empirical-standardization-uniform"],
                    &scope,
                    (norm / n as f64 - variance / (variance + floor.powi(2))).abs(),
                    0.,
                );
                expected_norm += probability * norm;
                let raw_to_mixture = weighted_scalar_transport_squared(
                    &raw,
                    &vec![1. / n as f64; n],
                    &mixture_raw,
                    &mixture_weights,
                )?;
                population_check(
                    &mut report,
                    "population_binary_empirical_raw_cost_identity",
                    &["cor-native-empirical-measurement-population-decay"],
                    &scope,
                    (raw_to_mixture - diameter.powi(2) * (count as f64 / n as f64 - q).abs()).abs(),
                    0.,
                );
                let z_to_mixture = weighted_scalar_transport_squared(
                    &z,
                    &vec![1. / n as f64; n],
                    &mixture_z,
                    &mixture_weights,
                )?;
                population_check(
                    &mut report,
                    "population_each_count_uniform_standardized_law_bound",
                    &["cor-empirical-standardization-uniform-continuity"],
                    &scope,
                    z_to_mixture,
                    raw_to_mixture / floor.powi(2),
                );
                exact_raw_to_mixture += probability * raw_to_mixture;
                exact_z_to_mixture += probability * z_to_mixture;
                raw_counts.push(raw);
                z_counts.push(z);
            }
            let mut exact_raw_pair = 0.;
            let mut exact_z_pair = 0.;
            for (i, p) in probabilities.iter().enumerate() {
                for (j, r) in probabilities.iter().enumerate() {
                    let raw_cost =
                        scalar_empirical_transport_squared(&raw_counts[i], &raw_counts[j])?;
                    let z_cost = scalar_empirical_transport_squared(&z_counts[i], &z_counts[j])?;
                    population_check(
                        &mut report,
                        "population_each_pair_sharp_empirical_law_bound",
                        &["cor-empirical-standardization-uniform-continuity"],
                        &scope,
                        z_cost,
                        raw_cost / floor.powi(2),
                    );
                    exact_raw_pair += p * r * raw_cost;
                    exact_z_pair += p * r * z_cost;
                }
            }
            let raw_decay_bound = diameter.powi(2) / (2. * (n as f64).sqrt());
            let z_decay_bound = raw_decay_bound / floor.powi(2);
            population_check(
                &mut report,
                "population_exact_raw_expected_law_decay",
                &["cor-native-empirical-measurement-population-decay"],
                &scope,
                exact_raw_to_mixture,
                raw_decay_bound,
            );
            population_check(
                &mut report,
                "population_exact_standardized_expected_law_decay",
                &["cor-native-empirical-measurement-population-decay"],
                &scope,
                exact_z_to_mixture,
                z_decay_bound,
            );
            population_check(
                &mut report,
                "population_exact_pair_uniform_four",
                &["cor-empirical-standardization-uniform-continuity"],
                &scope,
                exact_z_pair,
                4.,
            );
            population_check(
                &mut report,
                "population_exact_pair_sharp_lipschitz",
                &["cor-empirical-standardization-uniform-continuity"],
                &scope,
                exact_z_pair,
                exact_raw_pair / floor.powi(2),
            );
            let mut sums = [0.; 4];
            let mut squares = [0.; 4];
            for repetition in 0..repetitions {
                let mut pair = vec![];
                for side in 0..2 {
                    let companions = native_donors
                        .sample(
                            CompanionRequest {
                                population: &population,
                                pool: &pool,
                                eligible: &alive,
                                seed: 91377 + n as u64,
                                step: (2 * repetition + side) as u64,
                                stream: Stream::Distance,
                            },
                            &mut cx,
                        )
                        .await?;
                    let raw = (0..n)
                        .map(|i| {
                            let donor = companions.row(i).next().ok_or_else(|| {
                                GasError::Numerical("missing native scaling donor".into())
                            })?;
                            metric.compare(
                                &population.observations,
                                i,
                                &pool.population.observations,
                                donor as usize,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (z, _) =
                        native_standardizer.apply(&raw, &alive, &population.observations)?;
                    pair.push((raw, z));
                }
                let errors = [
                    scalar_empirical_transport_squared(&pair[0].0, &pair[1].0)?,
                    scalar_empirical_transport_squared(&pair[0].1, &pair[1].1)?,
                    weighted_scalar_transport_squared(
                        &pair[0].0,
                        &vec![1. / n as f64; n],
                        &mixture_raw,
                        &mixture_weights,
                    )?,
                    weighted_scalar_transport_squared(
                        &pair[0].1,
                        &vec![1. / n as f64; n],
                        &mixture_z,
                        &mixture_weights,
                    )?,
                ];
                for (i, error) in errors.iter().enumerate() {
                    sums[i] += error;
                    squares[i] += error * error;
                }
            }
            let means = sums.map(|s| s / repetitions as f64);
            let se = |i: usize| {
                ((squares[i] - sums[i].powi(2) / repetitions as f64).max(0.)
                    / ((repetitions - 1) * repetitions) as f64)
                    .sqrt()
            };
            let margins = [
                diameter.powi(2) * base_margin,
                4. * base_margin,
                diameter.powi(2) * base_margin,
                4. * base_margin,
            ];
            let exact = [
                exact_raw_pair,
                exact_z_pair,
                exact_raw_to_mixture,
                exact_z_to_mixture,
            ];
            let passed = means
                .iter()
                .zip(exact)
                .zip(margins)
                .all(|((observed, expected), band)| (observed - expected).abs() <= band);
            let c_n = standardization_coefficients(n, n, n, diameter, floor)?.value_total;
            let prefix = format!("population_{law}_N{n}");
            for (name, value, formula, labels) in [
                (
                    "empirical_lipschitz_squared",
                    floor.powi(-2),
                    "m^-2",
                    vec!["cor-empirical-standardization-uniform-continuity"],
                ),
                (
                    "universal_squared_error_bound",
                    4.,
                    "4",
                    vec!["lem-empirical-standardization-uniform"],
                ),
                (
                    "raw_population_decay_prefactor",
                    diameter.powi(2) / 2.,
                    "D²/2, independent of N",
                    vec!["cor-native-empirical-measurement-population-decay"],
                ),
                (
                    "standardized_population_decay_prefactor",
                    diameter.powi(2) / (2. * floor.powi(2)),
                    "D²/(2m²), independent of N",
                    vec!["cor-native-empirical-measurement-population-decay"],
                ),
                (
                    "exact_raw_normalized_empirical_error",
                    exact_raw_pair,
                    "E W2²(raw empirical law 1,raw empirical law 2)",
                    vec!["cor-empirical-standardization-uniform-continuity"],
                ),
                (
                    "exact_standardized_normalized_empirical_error",
                    exact_z_pair,
                    "E W2²(native standardized empirical law 1,law 2)",
                    vec!["cor-empirical-standardization-uniform-continuity"],
                ),
                (
                    "exact_raw_error_to_expected_law",
                    exact_raw_to_mixture,
                    "Binomial(N,q) expectation of D²|M/N-q|",
                    vec!["cor-native-empirical-measurement-population-decay"],
                ),
                (
                    "exact_standardized_error_to_expected_law",
                    exact_z_to_mixture,
                    "Binomial count expectation of native standardized scalar law W2² to standardized mixture law",
                    vec!["cor-native-empirical-measurement-population-decay"],
                ),
            ] {
                population_constant(
                    &mut report,
                    &format!("{prefix}_{name}"),
                    value,
                    formula,
                    &labels,
                    &scope,
                );
            }
            report.cases.push(PopulationScalingCase{walkers:n,companion_law:law.into(),sigma_min:floor,
            raw_support_diameter:diameter,opposite_cluster_probability:q,
            empirical_lipschitz_squared:floor.powi(-2),universal_squared_error_bound:4.,
            exact_raw_normalized_empirical_error:exact_raw_pair,exact_standardized_normalized_empirical_error:exact_z_pair,
            exact_raw_error_to_expected_law:exact_raw_to_mixture,exact_standardized_error_to_expected_law:exact_z_to_mixture,
            raw_expected_law_decay_bound:raw_decay_bound,standardized_expected_law_decay_bound:z_decay_bound,
            sampled_independent_output_pairs:repetitions,
            observed_raw_normalized_empirical_error:means[0],observed_standardized_normalized_empirical_error:means[1],
            observed_raw_error_to_expected_law:means[2],observed_standardized_error_to_expected_law:means[3],
            raw_sample_standard_error:se(0),standardized_sample_standard_error:se(1),
            raw_hoeffding_margin:margins[0],standardized_hoeffding_margin:margins[1],
            confidence_per_quantity:1.-failure_probability,status:if passed{"not_rejected"}else{"violated"}.into(),
            measurement_baseline:PopulationMeasurementBaseline{expected_operator_squared_error:0.,
                independent_raw_variance_contribution:2.*raw_variance_sum,exact_raw_summed_mean_square:2.*raw_variance_sum,
                exact_raw_indexed_mean_square_per_atom:2.*raw_variance_sum/n as f64,
                source_raw_summed_bound:6.*n as f64*diameter.powi(2),value_coefficient_c_n:c_n,
                exact_standardized_summed_mean_square:2.*expected_norm,
                exact_standardized_indexed_mean_square_per_atom:2.*expected_norm/n as f64,
                source_standardized_summed_bound:c_n*6.*n as f64*diameter.powi(2),
                scope:"Indexed independent measurement noise baseline on a fixed cloud; distinct from normalized scalar empirical-law error and from joint physical-state transport. The source moment envelope is applied on this bounded frozen fixture only.".into()},scope});
        }
    }
    Ok(report)
}

fn perturbation_tail_cases(report: &mut ContinuityReport) -> Result<()> {
    let repetitions = 4096;
    let alpha: f64 = 0.001;
    let empirical_margin = ((2. / alpha).ln() / (2. * repetitions as f64)).sqrt();
    for n in [4, 8, 32, 64] {
        for p in [0.1_f64, 0.5, 0.9] {
            let probabilities = binomial_probabilities(n, p);
            let counts: Vec<usize> = (0..repetitions)
                .map(|trial| {
                    (0..n)
                        .filter(|i| {
                            RandomStream::new(
                                4783,
                                trial as u64,
                                Stream::Kinetic,
                                *i as u64,
                                n as u64,
                            )
                            .uniform::<f64>()
                                < p
                        })
                        .count()
                })
                .collect();
            for diameter in [1_f64, 2.] {
                let d2 = diameter * diameter;
                let scope = format!(
                    "finite atomic perturbation x'=D*Bernoulli(p) at x=0; N={n}, p={p}, D={diameter}; independent native addressed uniforms; concentration fixture, not a valid-noise certificate for the complete gas"
                );
                let base = observations(vec![0.; n], 1)?;
                let mut flipped = vec![0.; n];
                flipped[n - 1] = diameter;
                let flipped = observations(flipped, 1)?;
                let functional_change = (0..n)
                    .map(|i| {
                        Distance::default()
                            .compare(&base, i, &flipped, i)
                            .map(|d| d * d)
                    })
                    .sum::<Result<f64>>()?
                    / n as f64;
                check(
                    report,
                    "binomial_tail_law_normalization",
                    &["thm-mcdiarmids-inequality"],
                    &scope,
                    (probabilities.iter().sum::<f64>() - 1.).abs(),
                    0.,
                );
                for delta in [0.05_f64, 0.2] {
                    let coefficients = perturbation_coefficients(n, diameter, p * d2, delta)?;
                    let prefix = format!("perturbation_N{n}_p{p}_D{diameter}_delta{delta}");
                    for (name, value, formula) in [
                        ("M_pert_squared", p * d2, "E[D² Bernoulli(p)] = p D²"),
                        ("c_i", coefficients.coordinate_bounded_difference, "D²/N"),
                        (
                            "sum_c_i_squared",
                            coefficients.sum_bounded_differences_squared,
                            "D⁴/N",
                        ),
                        ("B_M", coefficients.mean_bound, "N M_pert²"),
                        (
                            "B_S",
                            coefficients.fluctuation_bound,
                            "D² sqrt(N/2 log(2/delta))",
                        ),
                        (
                            "B_S_average",
                            coefficients.average_fluctuation_bound,
                            "B_S/N",
                        ),
                        (
                            "paired_metric_offset",
                            coefficients.paired_metric_offset,
                            "6/N (B_M + B_S(N,delta/2))",
                        ),
                    ] {
                        constant(
                            report,
                            &format!("{prefix}_{name}"),
                            value,
                            formula,
                            &[
                                "axiom-bounded-second-moment-perturbation",
                                "def-perturbation-fluctuation-bounds-reproof",
                                "thm-perturbation-operator-continuity-reproof",
                            ],
                            &scope,
                        );
                    }
                    // A coordinate flip attains the stated bounded difference.
                    check(
                        report,
                        "bounded_difference_coordinate_flip",
                        &[
                            "thm-mcdiarmids-inequality",
                            "axiom-bounded-algorithmic-diameter",
                        ],
                        &scope,
                        functional_change,
                        coefficients.coordinate_bounded_difference,
                    );
                    let reconstructed_delta = 2.
                        * (-2. * (coefficients.fluctuation_bound / n as f64).powi(2)
                            / coefficients.sum_bounded_differences_squared)
                            .exp();
                    check(
                        report,
                        "mcdiarmid_inverted_probability",
                        &[
                            "thm-mcdiarmids-inequality",
                            "def-perturbation-fluctuation-bounds-reproof",
                        ],
                        &scope,
                        (reconstructed_delta - delta).abs(),
                        0.,
                    );
                    let exact_tail = probabilities
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| {
                            ((*k as f64 - n as f64 * p) * d2).abs()
                                >= coefficients.fluctuation_bound
                        })
                        .map(|(_, mass)| *mass)
                        .sum::<f64>();
                    let empirical_tail = counts
                        .iter()
                        .filter(|k| {
                            ((**k as f64 - n as f64 * p) * d2).abs()
                                >= coefficients.fluctuation_bound
                        })
                        .count() as f64
                        / repetitions as f64;
                    check(
                        report,
                        "exact_binomial_mcdiarmid_tail",
                        &[
                            "thm-mcdiarmids-inequality",
                            "lem-sub-probabilistic-bound-perturbation-displacement-reproof",
                        ],
                        &scope,
                        exact_tail,
                        delta,
                    );
                    let empirical_scope = format!(
                        "{scope}; {repetitions} independent repetitions, diagnostic confidence={} for this row",
                        1. - alpha
                    );
                    check(
                        report,
                        "native_binomial_tail_frequency",
                        &["thm-mcdiarmids-inequality"],
                        &empirical_scope,
                        (empirical_tail - exact_tail).abs(),
                        empirical_margin,
                    );
                    report.tail_experiments.push(TailExperiment {
                        walkers: n,
                        repetitions,
                        bernoulli_probability: p,
                        diameter,
                        failure_probability: delta,
                        fluctuation_bound: coefficients.fluctuation_bound,
                        exact_two_sided_tail_probability: exact_tail,
                        empirical_two_sided_tail_probability: empirical_tail,
                        empirical_hoeffding_margin: empirical_margin,
                        diagnostic_confidence: 1. - alpha,
                        status: if (empirical_tail - exact_tail).abs() <= empirical_margin
                            && exact_tail <= delta + 1e-12
                        {
                            "not_rejected"
                        } else {
                            "violated"
                        }
                        .into(),
                        scope: empirical_scope,
                    });
                }
                // Test the full inequality at several thresholds, not only the
                // algebraic inversion point used to define B_S.
                for fraction in [0.1_f64, 0.2, 0.3, 0.5] {
                    let total_threshold = fraction * n as f64 * d2;
                    let probability = probabilities
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| ((*k as f64 - n as f64 * p) * d2).abs() >= total_threshold)
                        .map(|(_, mass)| *mass)
                        .sum::<f64>();
                    check(
                        report,
                        "exact_binomial_mcdiarmid_threshold",
                        &["thm-mcdiarmids-inequality"],
                        &scope,
                        probability,
                        2. * (-2. * total_threshold.powi(2) / (n as f64 * d2 * d2)).exp(),
                    );
                }
            }
        }
    }
    Ok(())
}

fn status_cases(report: &mut ContinuityReport) -> Result<()> {
    let map = PositiveMap::Logistic {
        amplitude: 0.98,
        floor: 0.01,
    };
    let lipschitz: f64 = 0.98 / 4.;
    constant(
        report,
        "probabilistic_status_L_partial",
        lipschitz,
        ".98/4, per-atom positional probability modulus",
        &[
            "cor-normalized-status-single-position",
            "def-final-status-change-coeffs",
        ],
        "native logistic-of-position probabilistic status fixture; the derivative gives the per-atom global modulus, independent of N",
    );
    constant(
        report,
        "bernoulli_variance_maximum",
        0.25,
        "max_{0<=p<=1} p(1-p)=1/4",
        &["thm-post-perturbation-status-update-continuity"],
        "independent-output status variance calculation",
    );
    // This declared fixture assigns atom i survival probability g(x_i).
    // Its per-atom derivative is bounded by .245 independently of N.
    // Coincident input atoms give exact scalar expectations below; independent
    // heterogeneous input probabilities are checked separately afterward.
    for n in [1, 4, 16] {
        for x in [-10_f64, -2., 0., 2., 10.] {
            for y in [-10_f64, -2., 0., 2., 10.] {
                let p = map.map(x)?;
                let q = map.map(y)?;
                let mismatch = p * (1. - q) + (1. - p) * q;
                let decomposition = p * (1. - p) + q * (1. - q) + (p - q).powi(2);
                let obs1 = observations(vec![x; n], 1)?;
                let obs2 = observations(vec![y; n], 1)?;
                let input_squared = (0..n)
                    .map(|i| {
                        Distance::default()
                            .compare(&obs1, i, &obs2, i)
                            .map(|v| v * v)
                    })
                    .sum::<Result<f64>>()?
                    / n as f64;
                let scope = format!(
                    "independent probabilistic-status fixture p_i=g(x_i), native logistic A=.98 eta=.01; N={n}, x={x}, y={y}; L_partial=.245, alpha_B=1; input empirical transport is optimal because all positions coincide within each swarm; independent Bernoulli mismatches use a supplied coupling and bound the minimum marked-swarm cost; not a geometric-boundary certificate"
                );
                check(
                    report,
                    "independent_status_variance_identity",
                    &["thm-post-perturbation-status-update-continuity"],
                    &scope,
                    (mismatch - decomposition).abs(),
                    0.,
                );
                check(
                    report,
                    "independent_status_intrinsic_variance",
                    &["thm-post-perturbation-status-update-continuity"],
                    &scope,
                    p * (1. - p) + q * (1. - q),
                    0.5,
                );
                check(
                    report,
                    "smoothed_status_probability_modulus",
                    &["axiom-boundary-regularity"],
                    &scope,
                    (p - q).abs(),
                    lipschitz * input_squared.sqrt(),
                );
                check(
                    report,
                    "post_perturbation_status_continuity",
                    &["thm-post-perturbation-status-update-continuity"],
                    &scope,
                    n as f64 * mismatch,
                    n as f64 / 2. + n as f64 * lipschitz.powi(2) * input_squared,
                );
                check(
                    report,
                    "final_status_change_coefficients",
                    &[
                        "def-final-status-change-coeffs",
                        "lem-final-status-change-bound",
                    ],
                    &scope,
                    n as f64 * mismatch,
                    n as f64 / 2. + lipschitz.powi(2) * n as f64 * input_squared,
                );
                if n == 16 && ((x == 0. && y == 0.) || (x == -2. && y == 2.)) {
                    let repetitions = 4096;
                    let mut changes = 0;
                    for trial in 0..repetitions {
                        for i in 0..n {
                            let first =
                                RandomStream::new(617, trial as u64, Stream::Domain, i as u64, 0)
                                    .uniform::<f64>()
                                    < p;
                            let second =
                                RandomStream::new(617, trial as u64, Stream::Domain, i as u64, 1)
                                    .uniform::<f64>()
                                    < q;
                            changes += usize::from(first != second);
                        }
                    }
                    let margin = (2000_f64.ln() / (2. * (repetitions * n) as f64)).sqrt();
                    let frequency = changes as f64 / (repetitions * n) as f64;
                    check(
                        report,
                        "native_supplied_status_coupling_frequency",
                        &[
                            "axiom-instep-independence",
                            "thm-post-perturbation-status-update-continuity",
                        ],
                        &format!(
                            "{scope}; {} independent Bernoulli pairs; per-row Hoeffding confidence=.999",
                            repetitions * n
                        ),
                        (frequency - mismatch).abs(),
                        margin,
                    );
                }
            }
        }
        constant(
            report,
            &format!("probabilistic_status_N{n}_K_status_var"),
            n as f64 / 2.,
            "N/2",
            &["def-final-status-change-coeffs"],
            "native smoothed Bernoulli fixture; N/2 is the independent Bernoulli mismatch envelope under the supplied coupling, an upper bound on optimal marked transport",
        );
        constant(
            report,
            &format!("probabilistic_status_N{n}_C_status_H"),
            lipschitz.powi(2),
            "L_partial² N^(1-alpha_B), alpha_B=1",
            &["def-final-status-change-coeffs"],
            "explicit native logistic-of-position probability law; per-atom derivative bound established independently of N",
        );
    }
    for n in [1, 4, 16, 64] {
        let left = (0..n).map(|i| (1.37 * i as f64).sin()).collect::<Vec<_>>();
        let right = left
            .iter()
            .enumerate()
            .map(|(i, x)| x + 0.07 * (i as f64 + 0.3).cos())
            .collect::<Vec<_>>();
        let mut mismatch = 0.;
        let mut positional = 0.;
        for (x, y) in left.iter().zip(&right) {
            let p = map.map(*x)?;
            let q = map.map(*y)?;
            mismatch += p * (1. - q) + (1. - p) * q;
            positional += (x - y).powi(2);
        }
        check(
            report,
            "heterogeneous_normalized_status_population_uniform",
            &[
                "cor-normalized-status-single-position",
                "lem-final-status-change-bound",
            ],
            &format!(
                "Native logistic per-atom probability fixture, N={n}; exact independent Bernoulli expectations under the supplied matching. Normalized displacement is an admissible transport cost, not a walker identity."
            ),
            mismatch / n as f64,
            0.5 + lipschitz.powi(2) * positional / n as f64,
        );
        constant(
            report,
            &format!("probabilistic_status_N{n}_normalized_C_status_H"),
            lipschitz.powi(2),
            "L_partial²",
            &[
                "def-final-status-change-coeffs",
                "cor-normalized-status-single-position",
            ],
            "Primary normalized status coefficient, independent of N.",
        );
    }
    Ok(())
}

type LiteralCloneAtoms = Vec<(f64, Population<f64>)>;

fn literal_clone_atoms(
    positions: &[f64],
    fitness: &[f64],
    decision: &CloneDecision,
) -> Result<(LiteralCloneAtoms, f64)> {
    let n = positions.len();
    require(
        n == 3 && fitness.len() == n && fitness.iter().all(|x| x.is_finite() && *x > 0.),
        "literal clone fixture requires three finite positive fitnesses",
    )?;
    let population = Population::new(observations(positions.to_vec(), 1)?)?;
    let pool = DonorPool::freeze(&population, 0, &[], 0, false)?;
    let mut probabilities = vec![vec![0.; n]; n];
    let mut clone_sum = 0.;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                probabilities[i][j] =
                    decision.acceptance_probability(0, fitness[i], fitness[j]) / (n - 1) as f64;
                clone_sum += probabilities[i][j];
            }
        }
        probabilities[i][i] = 1. - probabilities[i].iter().sum::<f64>();
    }
    let mut atoms = vec![];
    for encoded in 0..n.pow(n as u32) {
        let mut code = encoded;
        let mut mass = 1.;
        let mut choices = vec![];
        for (i, row) in probabilities.iter().enumerate() {
            let j = code % n;
            code /= n;
            mass *= row[j];
            choices.push(CloneChoice {
                donors: vec![WeightedDonor {
                    pool_index: j as u32,
                    weight: 1.,
                }],
                accepted: i != j,
                revival: false,
                probability: Some(decision.acceptance_probability(0, fitness[i], fitness[j])),
            });
        }
        if mass == 0. {
            continue;
        }
        let plan = ClonePlan {
            population_version: population.version,
            sources: pool.sources.clone(),
            choices,
            mutual: false,
        };
        atoms.push((mass, plan.apply_literal(&population, &pool)?));
    }
    Ok((atoms, clone_sum))
}

fn cloning_output_cases(report: &mut ContinuityReport) -> Result<()> {
    let decision = CloneDecision {
        epsilon: 0.1,
        saturation: 1.,
        ..Default::default()
    };
    let fixtures = [
        ([-1., 0., 1.], [0.2, 0.4, 1.]),
        ([-0.9, 0.01, 0.8], [0.3, 0.5, 0.8]),
        ([-1., 0., 1.], [1., 1., 1.]),
    ];
    let diameter_squared = 4.;
    let n = 3.;
    for (a, (positions1, fitness1)) in fixtures.iter().enumerate() {
        let (atoms1, probability1) = literal_clone_atoms(positions1, fitness1, &decision)?;
        for (b, (positions2, fitness2)) in fixtures.iter().enumerate() {
            let (atoms2, probability2) = literal_clone_atoms(positions2, fitness2, &decision)?;
            let input1 = observations(positions1.to_vec(), 1)?;
            let input2 = observations(positions2.to_vec(), 1)?;
            let delta_in = optimal_position_sum(&Distance::default(), &input1, &input2)?;
            let mut expected_output = 0.;
            let mut expected_move1 = 0.;
            let mut expected_move2 = 0.;
            for (p, output) in &atoms1 {
                expected_move1 += p
                    * (0..3)
                        .map(|i| {
                            Distance::default()
                                .compare(&input1, i, &output.observations, i)
                                .map(|d| d * d)
                        })
                        .sum::<Result<f64>>()?;
                for (q, other) in &atoms2 {
                    expected_output += p
                        * q
                        * optimal_position_sum(
                            &Distance::default(),
                            &output.observations,
                            &other.observations,
                        )?;
                }
            }
            for (p, output) in &atoms2 {
                expected_move2 += p
                    * (0..3)
                        .map(|i| {
                            Distance::default()
                                .compare(&input2, i, &output.observations, i)
                                .map(|d| d * d)
                        })
                        .sum::<Result<f64>>()?;
            }
            let scope = format!(
                "exact atoms of native ClonePlan.apply_literal and CloneDecision, uniform nonself donors, all alive, fixtures={a}/{b}; input and output displacement use optimal empirical transport; literal-copy movement terms use supplied coupling upper bounds; pre-jitter literal-copy stage; bounded positions [-1,1], D=2; no measurement-noise or clone-jitter certificate"
            );
            check(
                report,
                "literal_clone_atom_mass",
                &["def-cloning-measure"],
                &scope,
                (atoms1.iter().map(|(p, _)| *p).sum::<f64>() - 1.).abs(),
                0.,
            );
            check(
                report,
                "literal_clone_expected_candidate_move",
                &["proof-cloning-transition-operator-continuity-recorrected"],
                &scope,
                expected_move1 + expected_move2,
                diameter_squared * (probability1 + probability2),
            );
            check(
                report,
                "literal_clone_three_segment_output",
                &[
                    "thm-cloning-transition-operator-continuity-recorrected",
                    "proof-cloning-transition-operator-continuity-recorrected",
                ],
                &scope,
                expected_output,
                3. * (delta_in + expected_move1 + expected_move2),
            );
            check(
                report,
                "literal_clone_expected_probability_offset",
                &["proof-cloning-transition-operator-continuity-recorrected"],
                &scope,
                expected_output / n,
                3. * delta_in / n + 3. * diameter_squared / n * (probability1 + probability2),
            );
            check(
                report,
                "literal_clone_coarse_coefficients",
                &[
                    "def-cloning-operator-continuity-coeffs-recorrected",
                    "lem-sub-bound-sum-total-cloning-probs",
                ],
                &scope,
                expected_output / n,
                3. * delta_in / n + 6. * diameter_squared,
            );
            constant(
                report,
                &format!("literal_clone_pair_{a}_{b}_independent_output_second_moment"),
                expected_output / n,
                "sum over native literal-copy output atoms of mu(s)nu(t) times optimal marked empirical transport cost",
                &["thm-cloning-transition-operator-continuity-recorrected"],
                &scope,
            );
        }
    }
    for (name, value, formula) in [
        ("C_clone_L_coarse", 3., "3, choosing C_P=0"),
        ("C_clone_H_coarse", 0., "0, choosing H_P=0"),
        ("K_clone_coarse", 6. * diameter_squared, "3 D² (2N)/N = 6D²"),
    ] {
        constant(
            report,
            name,
            value,
            formula,
            &[
                "proof-cloning-transition-operator-continuity-recorrected",
                "lem-sub-bound-sum-total-cloning-probs",
            ],
            "coarse bounded-output specialization using sum clone probabilities<=2N; conditional A1..A4 propagation remains a separate calculation",
        );
    }
    Ok(())
}

fn native_noise_fixture(
    name: &str,
    dimensions: usize,
    rank: usize,
    factor: Vec<f64>,
    geometry: NoiseGeometry,
    innovation: InnovationLaw,
) -> (String, usize, Noise, NoiseMoments) {
    let moments = fixed_factor_moments(dimensions, rank, &factor, innovation)
        .expect("internal native fixture has a valid finite factor");
    (
        format!("{name}_{innovation:?}"),
        dimensions,
        Noise {
            innovation,
            geometry,
        },
        moments,
    )
}

fn noise_moment_case(
    report: &mut ContinuityReport,
    name: &str,
    primitive: &str,
    dimensions: usize,
    values: Vec<f64>,
    squared_norm_moments: (f64, f64),
    physical_support_radius: Option<f64>,
) -> Result<()> {
    let (expected_second, squared_norm_variance) = squared_norm_moments;
    let samples = values.len() / dimensions;
    let noisy = observations(values.clone(), dimensions)?;
    let zero = observations(vec![0.; values.len()], dimensions)?;
    let squared: Vec<f64> = (0..samples)
        .map(|i| {
            Distance::default()
                .compare(&noisy, i, &zero, i)
                .map(|d| d * d)
        })
        .collect::<Result<_>>()?;
    let observed_second = squared.iter().sum::<f64>() / samples as f64;
    let standard_error = (squared_norm_variance / samples as f64).sqrt();
    let acceptance_band = 6. * standard_error;
    let mut maximum_lipschitz_residual = f64::NEG_INFINITY;
    let mut maximum_projected_displacement = 0_f64;
    let metric = squashed_metric();
    for initial in [0_f64, 0.5, 10.] {
        let entering = squashed_observations(vec![initial; values.len()], dimensions)?;
        let leaving =
            squashed_observations(values.iter().map(|x| x + initial).collect(), dimensions)?;
        for (i, physical_squared) in squared.iter().enumerate() {
            let projected_squared = metric.compare(&entering, i, &leaving, i)?.powi(2);
            maximum_lipschitz_residual =
                maximum_lipschitz_residual.max(projected_squared - physical_squared);
            maximum_projected_displacement = maximum_projected_displacement.max(projected_squared);
        }
    }
    let scope = format!(
        "{name}: {samples} independently addressed samples, d={dimensions}; fixed noise factor; native Euclidean and radius-1 squashed position distances; fixed inputs 0,.5,10; six exact-standard-error diagnostic band"
    );
    let moment_labels: &[&str] = if physical_support_radius.is_some() {
        &[
            "def-reference-measures",
            "lem-validation-of-the-uniform-ball-measure",
            "axiom-bounded-second-moment-perturbation",
        ]
    } else {
        &[
            "axiom-bounded-second-moment-perturbation",
            "def-perturbation-measure",
        ]
    };
    check(
        report,
        "native_noise_second_moment",
        moment_labels,
        &scope,
        (observed_second - expected_second).abs(),
        acceptance_band,
    );
    check(
        report,
        "native_projection_noise_displacement",
        &["axiom-bounded-second-moment-perturbation"],
        &scope,
        maximum_lipschitz_residual,
        0.,
    );
    check(
        report,
        "native_projection_noise_diameter",
        &["axiom-bounded-algorithmic-diameter"],
        &scope,
        maximum_projected_displacement,
        4.,
    );
    if let Some(radius) = physical_support_radius {
        check(
            report,
            "reference_ball_physical_support",
            &[
                "def-reference-measures",
                "lem-validation-of-the-uniform-ball-measure",
            ],
            &scope,
            squared.iter().copied().fold(0_f64, f64::max),
            radius * radius,
        );
        // The lemma states the coarser radius-squared bound. The exact radial
        // moment is a separate Euclidean specialization, not its replacement.
        check(
            report,
            "reference_ball_exact_moment_below_lemma_bound",
            &["lem-validation-of-the-uniform-ball-measure"],
            &scope,
            expected_second,
            radius * radius,
        );
    }
    constant(
        report,
        &format!("{name}_physical_second_moment"),
        expected_second,
        if physical_support_radius.is_some() {
            "d/(d+2) radius² (Euclidean reference-ball specialization)"
        } else {
            "trace(LL^T)=||L||_F² for independent unit-variance innovations"
        },
        moment_labels,
        &scope,
    );
    constant(
        report,
        &format!("{name}_projected_M_pert_squared"),
        expected_second.min(4.),
        "min(E||noise||²,4), using 1-Lipschitz radius-1 squash and diameter 2",
        &[
            "axiom-bounded-second-moment-perturbation",
            "axiom-bounded-algorithmic-diameter",
        ],
        &format!(
            "{scope}; bound uses the analytic Lipschitz/range properties of this selected radial map, not a sampled supremum"
        ),
    );
    report.noise_experiments.push(NoiseMomentExperiment {
        name: name.into(),
        sampling_primitive: primitive.into(),
        dimensions,
        samples,
        exact_second_moment: expected_second,
        observed_second_moment: observed_second,
        standard_error,
        acceptance_band,
        maximum_projection_lipschitz_residual: maximum_lipschitz_residual,
        maximum_projected_squared_displacement: maximum_projected_displacement,
        status: if (observed_second - expected_second).abs() <= acceptance_band
            && maximum_lipschitz_residual <= 1e-10
            && maximum_projected_displacement <= 4. + 1e-10
        {
            "not_rejected"
        } else {
            "violated"
        }
        .into(),
        scope,
    });
    Ok(())
}

async fn noise_cases(report: &mut ContinuityReport) -> Result<()> {
    let samples = 8192;
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    constant(
        report,
        "native_position_squash_L_phi",
        1.,
        "radial derivative (1+r)^(-2), tangential derivative (1+r)^(-1), both <=1",
        &[
            "axiom-bounded-second-moment-perturbation",
            "lem-validation-of-the-uniform-ball-measure",
        ],
        "the selected native radius-1 radial position projection; fixed lambda=0 discards velocity",
    );
    constant(
        report,
        "native_position_squash_D_Y",
        2.,
        "diameter of the image ball of radius 1",
        &["axiom-bounded-algorithmic-diameter"],
        "the selected native radius-1 radial position projection; no compact physical-state assumption",
    );
    let mut fixtures = vec![];
    for innovation in [InnovationLaw::Gaussian, InnovationLaw::StandardizedUniform] {
        for dimensions in [1, 2, 5] {
            for scale in [0.02_f64, 0.2, 1.] {
                let mut matrix = vec![0.; dimensions * dimensions];
                for i in 0..dimensions {
                    matrix[i * dimensions + i] = scale;
                }
                fixtures.push(native_noise_fixture(
                    &format!("isotropic_d{dimensions}_sigma{scale}"),
                    dimensions,
                    dimensions,
                    matrix,
                    NoiseGeometry::Isotropic {
                        scale: FactorValues::Constant {
                            values: vec![scale],
                        },
                    },
                    innovation,
                ));
            }
        }
        let diagonal = [0.02, 0.2, 0.8];
        let mut matrix = vec![0.; 9];
        for i in 0..3 {
            matrix[i * 3 + i] = diagonal[i];
        }
        fixtures.push(native_noise_fixture(
            "diagonal",
            3,
            3,
            matrix,
            NoiseGeometry::Diagonal {
                factor: FactorValues::Constant {
                    values: diagonal.to_vec(),
                },
            },
            innovation,
        ));
        let full = vec![0.2, 0.1, -0.3, 0.4];
        fixtures.push(native_noise_fixture(
            "full",
            2,
            2,
            full.clone(),
            NoiseGeometry::Full {
                factor: FactorValues::Constant { values: full },
            },
            innovation,
        ));
        let low_rank = vec![0.1, 0.2, 0.3, -0.1, 0.5, 0.];
        fixtures.push(native_noise_fixture(
            "low_rank",
            3,
            2,
            low_rank.clone(),
            NoiseGeometry::LowRank {
                rank: 2,
                factor: FactorValues::Constant { values: low_rank },
            },
            innovation,
        ));
    }
    for (index, (name, dimensions, noise, moments)) in fixtures.into_iter().enumerate() {
        let input = observations(vec![0.; samples * dimensions], dimensions)?;
        let request = NoiseRequest {
            rows: samples,
            dimension: dimensions,
            seed: 2039,
            step: index as u64,
            stream: Stream::Kinetic,
            substep: 0,
        };
        let output = noise.sample(&input, request, &mut cx).await?;
        noise_moment_case(
            report,
            &name,
            "native NoiseSource::sample",
            dimensions,
            output.values().to_vec(),
            (moments.expected_squared_norm, moments.squared_norm_variance),
            None,
        )?;
        constant(
            report,
            &format!("{name}_squared_norm_variance"),
            moments.squared_norm_variance,
            if noise.innovation == InnovationLaw::Gaussian {
                "2 tr((LL^T)^2)"
            } else {
                "2 tr((LL^T)^2) - (6/5) sum_a ||L_column_a||^4; standardized box law"
            },
            &["axiom-bounded-second-moment-perturbation"],
            "derived variance used to set the native second-moment diagnostic band; not a new convergence theorem",
        );
    }
    // The native provider's uniform law is a box. Construct the separately
    // specified reference-ball law with native Gaussian directions and an
    // independent native uniform radius; no provider is renamed here.
    for dimensions in [1, 2, 5] {
        for radius in [0.02_f64, 0.2, 1.] {
            let mut values = Vec::with_capacity(samples * dimensions);
            for i in 0..samples {
                let mut rng = RandomStream::new(
                    2237,
                    dimensions as u64,
                    Stream::Kinetic,
                    i as u64,
                    radius.to_bits(),
                );
                let direction: Vec<f64> = (0..dimensions).map(|_| rng.gaussian::<f64>()).collect();
                let length = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
                require(
                    length.is_finite() && length > 0.,
                    "reference-ball direction is numerically degenerate",
                )?;
                let radial_scale =
                    radius * rng.uniform::<f64>().powf(1. / dimensions as f64) / length;
                values.extend(direction.iter().map(|x| radial_scale * x));
            }
            let d = dimensions as f64;
            let expected_second = radius * radius * d / (d + 2.);
            let variance = radius.powi(4) * 4. * d / ((d + 4.) * (d + 2.).powi(2));
            noise_moment_case(
                report,
                &format!("reference_ball_d{dimensions}_radius{radius}"),
                "native RandomStream Gaussian direction and independent U^(1/d) radius; diagnostic reference law",
                dimensions,
                values,
                (expected_second, variance),
                Some(radius),
            )?;
        }
    }
    Ok(())
}
