//! Fixed-swarm geometry and Lyapunov observables from convergence chapters 1--3.
//! Optimal transport is computed for the alive empirical measures, including
//! unequal alive counts. Swarms are permutation invariant; row indices only
//! identify storage locations for transport plans and recorded donor choices.
use crate::convergence_framework::BoundCheck;
use crate::convergence_kinetic::{
    ConfiningLandscape, KineticCheckStatus, KineticInput, KineticValidationConfig,
    SourceMappedConstant, kinetic_constants,
};
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasError, ObservationBatch, Population, Precision, Result,
    TensorBatch,
    donor::{CompanionRequest, CompanionSampler, DonorModule, DonorPool},
    geometry::{AlgorithmicDistance, Distance, InteractionKernel, Kernel},
    random::Stream,
};
use serde::{Deserialize, Serialize};

const CH1: &str = "docs/source/2_fractal_gas/convergence_program/01_fragile_gas_framework.md";
const CH2: &str = "docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md";
const CH3: &str = "docs/source/2_fractal_gas/convergence_program/03_cloning.md";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LyapunovValidationConfig {
    pub lambda_v: f64,
    /// Cross coefficient of q(x,v)=|x|²+lambda_v|v|²+b<x,v>.
    pub cross_coefficient: f64,
    pub variance_weight: f64,
    pub boundary_weight: f64,
    pub status_weight: f64,
    pub position_projection_radius: f64,
    pub velocity_projection_radius: f64,
    pub companion_width: f64,
    pub companion_replicates: usize,
    pub seed: u64,
    /// Optional admissible coupling of the two alive empirical supports,
    /// enumerated in their storage order. None uses the uniform product plan.
    pub candidate_transport_plan: Option<Vec<Vec<f64>>>,
}
impl Default for LyapunovValidationConfig {
    fn default() -> Self {
        Self {
            lambda_v: 1.,
            cross_coefficient: 0.5,
            variance_weight: 0.1,
            boundary_weight: 0.05,
            status_weight: 1.,
            position_projection_radius: 2.,
            velocity_projection_radius: 2.,
            companion_width: 2.,
            companion_replicates: 512,
            seed: 570_031,
            candidate_transport_plan: None,
        }
    }
}
impl LyapunovValidationConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            [
                self.lambda_v,
                self.variance_weight,
                self.boundary_weight,
                self.status_weight,
                self.position_projection_radius,
                self.velocity_projection_radius,
                self.companion_width,
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.)
                && self.cross_coefficient.is_finite()
                && self.cross_coefficient.powi(2) < 4. * self.lambda_v
                && (32..=1_000_000).contains(&self.companion_replicates),
            "Lyapunov constants require finite positive weights/radii/width, b²<4lambda_v and 32..=1000000 conditional categorical replicates",
        )
    }
    fn distance(&self) -> Distance {
        Distance::SquashedPhaseSpace {
            positions: "positions".into(),
            velocities: "velocities".into(),
            position_radius: self.position_projection_radius,
            velocity_radius: self.velocity_projection_radius,
            lambda: self.lambda_v,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LyapunovSwarm {
    pub positions: Vec<Vec<f64>>,
    pub velocities: Vec<Vec<f64>>,
    pub alive: Vec<bool>,
    /// Values of an explicitly supplied nonnegative barrier at each row.
    /// Dead values are ignored. None leaves W_b and V_total unavailable.
    pub barrier_values: Option<Vec<f64>>,
}
impl LyapunovSwarm {
    fn validate(&self) -> Result<(usize, usize)> {
        let n = self.positions.len();
        let d = self.positions.first().map_or(0, Vec::len);
        require(
            (1..=64).contains(&n)
                && (1..=256).contains(&d)
                && self.velocities.len() == n
                && self.alive.len() == n
                && self.alive.iter().any(|a| *a)
                && self.positions.iter().chain(&self.velocities).all(|r| {
                    r.len() == d && r.iter().all(|v| v.is_finite()) && norm_sq(r).is_finite()
                })
                && self.barrier_values.as_ref().is_none_or(|values| {
                    values.len() == n
                        && values
                            .iter()
                            .zip(&self.alive)
                            .all(|(v, a)| !a || (v.is_finite() && *v >= 0.))
                }),
            "paired Lyapunov inputs need 1..=64 slots, 1..=256 finite matching coordinates, nonempty alive measures and optional finite nonnegative alive barrier values",
        )?;
        Ok((n, d))
    }
    fn observations(&self) -> Result<ObservationBatch<f64>> {
        let (n, d) = self.validate()?;
        let mut obs =
            ObservationBatch::positions(TensorBatch::vectors(n, d, self.positions.concat())?);
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(n, d, self.velocities.concat())?,
        );
        Ok(obs)
    }
    fn alive_points(&self) -> Vec<Vec<f64>> {
        self.positions
            .iter()
            .zip(&self.velocities)
            .zip(&self.alive)
            .filter(|(_, a)| **a)
            .map(|((x, v), _)| x.iter().chain(v).copied().collect())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VarianceConversions {
    pub slots: usize,
    pub alive: usize,
    pub sum_squares_position: f64,
    pub sum_squares_velocity: f64,
    pub physical_position_variance: f64,
    pub physical_velocity_variance: f64,
    pub n_normalized_position_variance: f64,
    pub n_normalized_velocity_variance: f64,
    pub pairwise_sasaki_variance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompanionRowLaw {
    pub query_slot: usize,
    pub donor_slots: Vec<usize>,
    pub probabilities: Vec<f64>,
    pub measured_probabilities: Vec<f64>,
    pub standard_errors: Vec<f64>,
    pub log_normalizer: f64,
    pub candidate_count: usize,
    pub log_normalizer_lower_bound: f64,
    pub log_normalizer_upper_bound: f64,
    pub minimum_probability_lower_bound: Option<f64>,
    pub frequency_status: Vec<KineticCheckStatus>,
    pub status: KineticCheckStatus,
    pub source_labels: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LyapunovValidationReport {
    pub slots: usize,
    pub dimensions: usize,
    pub config: LyapunovValidationConfig,
    pub source_mapped_constants: Vec<SourceMappedConstant>,
    pub coercivity_min_eigenvalue: f64,
    pub coercivity_max_eigenvalue: f64,
    pub position_barycenters: [Vec<f64>; 2],
    pub velocity_barycenters: [Vec<f64>; 2],
    pub location_error: f64,
    pub structural_error: f64,
    pub physical_wasserstein_squared: f64,
    pub centered_euclidean_wasserstein_squared: f64,
    /// Cost of the supplied admissible plan, or of the default product plan;
    /// this is a comparison upper bound, never the empirical swarm metric.
    pub candidate_coupling_cost: f64,
    pub candidate_transport_plan: Vec<Vec<f64>>,
    pub transport_plan: Vec<Vec<f64>>,
    pub centered_transport_plan: Vec<Vec<f64>>,
    pub variance_conversions: [VarianceConversions; 2],
    pub intra_swarm_variance: f64,
    pub boundary_potential: Option<f64>,
    pub augmented_lyapunov: Option<f64>,
    pub augmented_lyapunov_without_boundary: f64,
    pub projected_positional_displacement_sum: f64,
    pub physical_phase_displacement_sum: f64,
    pub status_difference_count: usize,
    pub dispersion_squared: f64,
    pub dispersion_transport_plan: Vec<Vec<f64>>,
    pub companion_laws: Vec<CompanionRowLaw>,
    /// Independent Gaussian position-law probability-distance probes. These
    /// hold their own checks, and are not duplicated in `checks`.
    pub boundary_probability_diagnostics: Vec<BoundaryProbabilitySeparation>,
    pub checks: Vec<BoundCheck>,
    pub scope_notes: Vec<String>,
}

/// Exact/numerical fixed-swarm observables, followed by conditional native
/// categorical sampling at the unchanged left input (no trajectory feedback).
pub async fn validate_lyapunov(
    config: &LyapunovValidationConfig,
    left: &LyapunovSwarm,
    right: &LyapunovSwarm,
) -> Result<LyapunovValidationReport> {
    config.validate()?;
    let (n, d) = left.validate()?;
    require(
        right.validate()? == (n, d),
        "paired swarm slot/dimension schemas differ",
    )?;
    let left_points = left.alive_points();
    let right_points = right.alive_points();
    let left_mean = mean_point(&left_points);
    let right_mean = mean_point(&right_points);
    let left_centered = center_points(&left_points, &left_mean);
    let right_centered = center_points(&right_points, &right_mean);
    let q = |a: &[f64], b: &[f64]| {
        hypocoercive_cost(a, b, d, config.lambda_v, config.cross_coefficient)
    };
    let (wasserstein, plan) = canonical_uniform_transport(&left_points, &right_points, q)?;
    let (structure, centered_plan) =
        canonical_uniform_transport(&left_centered, &right_centered, q)?;
    let (euclidean_centered, _) =
        canonical_uniform_transport(&left_centered, &right_centered, euclidean_cost)?;
    let location = q(&left_mean, &right_mean);
    let lambda_plus = (1.
        + config.lambda_v
        + ((1. - config.lambda_v).powi(2) + config.cross_coefficient.powi(2)).sqrt())
        / 2.;
    // Equivalent to the displayed eigenvalue formula, stable near degeneracy.
    let lambda_minus = (config.lambda_v - config.cross_coefficient.powi(2) / 4.) / lambda_plus;
    require(
        lambda_minus.is_finite() && lambda_minus > 0. && lambda_plus.is_finite(),
        "coercivity eigenvalues overflow/underflow",
    )?;
    let left_conversions = variance_conversions(left, &left_points, &left_centered, config);
    let right_conversions = variance_conversions(right, &right_points, &right_centered, config);
    let variance = left_conversions.n_normalized_position_variance
        + right_conversions.n_normalized_position_variance
        + config.lambda_v
            * (left_conversions.n_normalized_velocity_variance
                + right_conversions.n_normalized_velocity_variance);
    let barrier = left
        .barrier_values
        .as_ref()
        .zip(right.barrier_values.as_ref())
        .map(|(l, r)| {
            (l.iter()
                .zip(&left.alive)
                .filter_map(|(v, a)| a.then_some(v))
                .sum::<f64>()
                + r.iter()
                    .zip(&right.alive)
                    .filter_map(|(v, a)| a.then_some(v))
                    .sum::<f64>())
                / n as f64
        });
    let lyapunov_without_barrier = wasserstein + config.variance_weight * variance;
    let augmented = barrier.map(|b| lyapunov_without_barrier + config.boundary_weight * b);
    let candidate_plan = candidate_transport_plan(config, left_points.len(), right_points.len())?;
    let candidate_cost = candidate_plan
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, mass)| mass * q(&left_points[i], &right_points[j]))
                .sum::<f64>()
        })
        .sum::<f64>();
    let a = left.observations()?;
    let b = right.observations()?;
    let distance = config.distance();
    let physical = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 1.,
        velocity_scale: 1.,
        lambda: config.lambda_v,
        periodic: None,
    };
    let mut projected_squared = vec![vec![0.; n]; n];
    let mut physical_squared = vec![vec![0.; n]; n];
    let mut marked_squared = vec![vec![0.; n]; n];
    let mut projection_excess = 0_f64;
    let mut checks = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let projected = distance.compare(&a, i, &b, j)?;
            let phys = physical.compare(&a, i, &b, j)?;
            projected_squared[i][j] = projected * projected;
            physical_squared[i][j] = phys * phys;
            marked_squared[i][j] = projected * projected
                + config.status_weight * usize::from(left.alive[i] != right.alive[j]) as f64;
            projection_excess = projection_excess.max(projected - phys);
        }
    }
    // Row indices address the native distance matrix only; optimizing over
    // all uniform marked couplings removes any storage-order dependence.
    let left_order = canonical_marked_order(left);
    let right_order = canonical_marked_order(right);
    let left_addresses = left_order
        .iter()
        .map(|i| vec![*i as f64])
        .collect::<Vec<_>>();
    let right_addresses = right_order
        .iter()
        .map(|i| vec![*i as f64])
        .collect::<Vec<_>>();
    let (dispersion, canonical_dispersion_plan) =
        uniform_transport(&left_addresses, &right_addresses, |l, r| {
            marked_squared[l[0] as usize][r[0] as usize]
        })?;
    let dispersion_plan = reindex_plan(&canonical_dispersion_plan, &left_order, &right_order);
    let projected_sum = n as f64 * plan_cost(&dispersion_plan, &projected_squared);
    let physical_sum = n as f64 * plan_cost(&dispersion_plan, &physical_squared);
    let changed_mass = n as f64
        * dispersion_plan
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter_map(|(j, mass)| (left.alive[i] != right.alive[j]).then_some(mass))
                    .sum::<f64>()
            })
            .sum::<f64>();
    // Equal-size uniform empirical transport has a permutation optimizer;
    // the integer-flow solver returns one, so N times mismatch mass is an integer.
    require(
        (changed_mass - changed_mass.round()).abs() <= 1e-10,
        "uniform marked transport mismatch mass is not an integer number of atoms",
    )?;
    let changed = changed_mass.round() as usize;
    checks.push(upper("native_projection_nonexpansion", &["lem-squashing-properties-generic","lem-projection-lipschitz"],
        "Actual native Distance providers, physical and squashed Sasaki metrics with identical velocity weight; all slots including dead retained coordinates",projection_excess,0.));
    checks.push(identity("wasserstein_barycenter_decomposition", &["lem-wasserstein-decomposition","def-location-error-component","def-structural-error-component","def-barycentres-and-centered-vectors"],
        "Uniform alive empirical measures, possibly unequal alive counts; exact finite optimal transport with the physical SPD quadratic cost",wasserstein-location-structure));
    checks.push(upper(
        "location_coercivity",
        &["lem-V-coercive"],
        "Physical barycenter difference, b²<4lambda_v",
        lambda_minus * euclidean_cost(&left_mean, &right_mean),
        location,
    ));
    checks.push(upper(
        "structural_coercivity_optimal_transport",
        &["lem-V-coercive"],
        "Optimal centered Euclidean transport between permutation-invariant alive empirical measures",
        lambda_minus * euclidean_centered,
        structure,
    ));
    checks.push(upper(
        "structural_continuity_optimal_transport",
        &["lem-V-coercive"],
        "Physical quadratic cost q<=lambda_plus*Euclidean squared distance",
        structure,
        lambda_plus * euclidean_centered,
    ));
    checks.push(upper("optimal_cost_below_candidate_coupling", &["lem-V-coercive"],"Supplied admissible alive transport plan, or the uniform product plan when none is supplied; a comparison upper bound rather than the swarm metric",wasserstein,candidate_cost));
    checks.push(identity("dispersion_position_status_decomposition", &["def-n-particle-displacement-metric"],"Uniform all-slot marked empirical transport with native squashed phase-space cost and mark-mismatch weight; both components use the minimizing plan",dispersion-(projected_sum+config.status_weight*changed as f64)/n as f64));
    checks.push(upper(
        "dispersion_positional_conversion",
        &["def-n-particle-displacement-metric"],
        "Delta_pos²<=N*d_Disp²",
        projected_sum,
        n as f64 * dispersion,
    ));
    checks.push(upper(
        "dispersion_status_conversion",
        &["def-n-particle-displacement-metric"],
        "n_changed<=N*d_Disp²/lambda_status",
        changed as f64,
        n as f64 * dispersion / config.status_weight,
    ));
    for (index, c) in [&left_conversions, &right_conversions]
        .into_iter()
        .enumerate()
    {
        checks.push(identity(
            &format!("variance_normalization[{index}]"),
            &[
                "def-variance-conversions",
                "def-full-synergistic-lyapunov-function",
            ],
            "Alive barycenter; sums normalized by fixed slots N or alive count k as declared",
            c.n_normalized_position_variance
                - c.alive as f64 / n as f64 * c.physical_position_variance,
        ));
        checks.push(identity(&format!("variance_pairwise_identity[{index}]"), &["lem-phase-space-packing"],
            "Physical diagonal Sasaki variance, without the cross term: sum_{i<j} distance²/k² equals centered variance per alive row",
            c.pairwise_sasaki_variance-c.physical_position_variance-config.lambda_v*c.physical_velocity_variance));
    }
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    let (laws, mut law_checks) = companion_laws(config, left, &mut cx).await?;
    checks.append(&mut law_checks);
    require(
        [
            wasserstein,
            location,
            structure,
            variance,
            candidate_cost,
            dispersion,
            lyapunov_without_barrier,
        ]
        .iter()
        .all(|x| x.is_finite())
            && augmented.is_none_or(|x| x.is_finite()),
        "Lyapunov observables overflow",
    )?;
    Ok(LyapunovValidationReport {
        slots:n,dimensions:d,config:config.clone(),
        source_mapped_constants:source_constants(config,lambda_minus,lambda_plus),
        coercivity_min_eigenvalue:lambda_minus,coercivity_max_eigenvalue:lambda_plus,
        position_barycenters:[left_mean[..d].to_vec(),right_mean[..d].to_vec()],
        velocity_barycenters:[left_mean[d..].to_vec(),right_mean[d..].to_vec()],
        location_error:location,structural_error:structure,physical_wasserstein_squared:wasserstein,
        centered_euclidean_wasserstein_squared:euclidean_centered,candidate_coupling_cost:candidate_cost,candidate_transport_plan:candidate_plan,
        transport_plan:plan,centered_transport_plan:centered_plan,
        variance_conversions:[left_conversions,right_conversions],intra_swarm_variance:variance,
        boundary_potential:barrier,augmented_lyapunov:augmented,augmented_lyapunov_without_boundary:lyapunov_without_barrier,
        projected_positional_displacement_sum:projected_sum,physical_phase_displacement_sum:physical_sum,
        status_difference_count:changed,dispersion_squared:dispersion,dispersion_transport_plan:dispersion_plan,companion_laws:laws,boundary_probability_diagnostics:vec![],checks,
        scope_notes:vec![
            "c_V and c_B are supplied positive coefficients for computing the defined observable; they are not inferred to satisfy a complete-update drift theorem.".into(),
            "Physical swarms are permutation-invariant empirical measures; array indices serve storage and tracking only. candidate_coupling_cost is the cost of an admissible comparison plan and supplies an upper bound on the optimal transport cost, never a separate swarm metric. Reordering the supports reindexes a supplied plan; the default product plan is invariant under independent reorderings.".into(),
            "dispersion_squared is optimal transport between uniform all-slot marked empirical measures, with native squashed phase-space cost plus status-mismatch weight. Support atoms are ordered by their intrinsic position, velocity and mark before solving, and the chosen canonical minimizing plan is mapped back to storage order. Positional/status components belong to that plan; physical_phase_displacement_sum evaluates unsquashed physical distances on it. This intrinsic tie convention preserves the components under independent row reorderings even when multiple marked minimizers exist.".into(),
            "Supplied barrier values compute W_b at these finite states. Smoothness, divergence, integrability and a boundary drift certificate remain assumptions about that barrier and transition; no substitute is fabricated when values are absent.".into(),
            "Only categorical frequencies are statistical: each frozen-swarm donor draw uses an independent native random address. No interacting trajectory rows or complete steps are treated as independent samples.".into(),
        ],
    })
}

fn variance_conversions(
    s: &LyapunovSwarm,
    points: &[Vec<f64>],
    centered: &[Vec<f64>],
    c: &LyapunovValidationConfig,
) -> VarianceConversions {
    let (n, k, d) = (s.positions.len(), points.len(), s.positions[0].len());
    let sx = centered.iter().map(|z| norm_sq(&z[..d])).sum::<f64>();
    let sv = centered.iter().map(|z| norm_sq(&z[d..])).sum::<f64>();
    let pairwise = (0..k)
        .flat_map(|i| (i + 1..k).map(move |j| (i, j)))
        .map(|(i, j)| {
            euclidean_cost(&points[i][..d], &points[j][..d])
                + c.lambda_v * euclidean_cost(&points[i][d..], &points[j][d..])
        })
        .sum::<f64>()
        / (k * k) as f64;
    VarianceConversions {
        slots: n,
        alive: k,
        sum_squares_position: sx,
        sum_squares_velocity: sv,
        physical_position_variance: sx / k as f64,
        physical_velocity_variance: sv / k as f64,
        n_normalized_position_variance: sx / n as f64,
        n_normalized_velocity_variance: sv / n as f64,
        pairwise_sasaki_variance: pairwise,
    }
}

fn candidate_transport_plan(
    c: &LyapunovValidationConfig,
    left: usize,
    right: usize,
) -> Result<Vec<Vec<f64>>> {
    let plan = c
        .candidate_transport_plan
        .clone()
        .unwrap_or_else(|| vec![vec![1. / (left * right) as f64; right]; left]);
    require(
        plan.len() == left
            && plan
                .iter()
                .all(|row| row.len() == right && row.iter().all(|p| p.is_finite() && *p >= 0.)),
        "candidate transport plan must have one row/column per alive support atom and finite nonnegative masses",
    )?;
    require(
        plan.iter()
            .all(|row| (row.iter().sum::<f64>() - 1. / left as f64).abs() <= 1e-12)
            && (0..right).all(|j| {
                (plan.iter().map(|row| row[j]).sum::<f64>() - 1. / right as f64).abs() <= 1e-12
            }),
        "candidate transport plan must have the uniform alive empirical marginals",
    )?;
    Ok(plan)
}

fn plan_cost(plan: &[Vec<f64>], costs: &[Vec<f64>]) -> f64 {
    plan.iter()
        .zip(costs)
        .map(|(row, costs)| {
            row.iter()
                .zip(costs)
                .map(|(mass, cost)| mass * cost)
                .sum::<f64>()
        })
        .sum()
}

fn compare_points(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    a.iter()
        .zip(b)
        .map(|(a, b)| a.total_cmp(b))
        .find(|order| !order.is_eq())
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn canonical_marked_order(s: &LyapunovSwarm) -> Vec<usize> {
    let mut order = (0..s.positions.len()).collect::<Vec<_>>();
    order.sort_by(|&i, &j| {
        compare_points(&s.positions[i], &s.positions[j])
            .then_with(|| compare_points(&s.velocities[i], &s.velocities[j]))
            .then_with(|| s.alive[i].cmp(&s.alive[j]))
    });
    order
}

fn reindex_plan(plan: &[Vec<f64>], left: &[usize], right: &[usize]) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.; right.len()]; left.len()];
    for (i, &original_left) in left.iter().enumerate() {
        for (j, &original_right) in right.iter().enumerate() {
            result[original_left][original_right] = plan[i][j];
        }
    }
    result
}

fn canonical_uniform_transport(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    cost: impl Fn(&[f64], &[f64]) -> f64,
) -> Result<(f64, Vec<Vec<f64>>)> {
    let order = |points: &[Vec<f64>]| {
        let mut order = (0..points.len()).collect::<Vec<_>>();
        order.sort_by(|&i, &j| compare_points(&points[i], &points[j]));
        order
    };
    let left_order = order(left);
    let right_order = order(right);
    let l = left_order
        .iter()
        .map(|&i| left[i].clone())
        .collect::<Vec<_>>();
    let r = right_order
        .iter()
        .map(|&j| right[j].clone())
        .collect::<Vec<_>>();
    let (value, plan) = uniform_transport(&l, &r, cost)?;
    Ok((value, reindex_plan(&plan, &left_order, &right_order)))
}

async fn companion_laws(
    c: &LyapunovValidationConfig,
    s: &LyapunovSwarm,
    cx: &mut ExecutionContext,
) -> Result<(Vec<CompanionRowLaw>, Vec<BoundCheck>)> {
    let mut population = Population::new(s.observations()?)?;
    for (status, alive) in population.validity.iter_mut().zip(&s.alive) {
        status.terminated = !*alive;
    }
    let pool = DonorPool::freeze(&population, 0, &[], 0, false)?;
    let module = DonorModule {
        distance: c.distance(),
        kernel: Kernel::Gaussian {
            width: c.companion_width,
        },
        ..Default::default()
    };
    module.validate(&population.observations)?;
    let n = population.len();
    let k = pool.sources.len();
    let diameter_sq = 4.
        * (c.position_projection_radius.powi(2)
            + c.lambda_v * c.velocity_projection_radius.powi(2));
    let log_q_min = -diameter_sq / (2. * c.companion_width.powi(2));
    require(
        log_q_min.is_finite(),
        "companion global log-weight lower bound overflow",
    )?;
    let mut counts = vec![vec![0_usize; k]; n];
    let mut done = 0;
    while done < c.companion_replicates {
        let draws = (c.companion_replicates - done).min(1024);
        let sampler = DonorModule {
            count: draws,
            ..module.clone()
        };
        let result = sampler
            .sample(
                CompanionRequest {
                    population: &population,
                    pool: &pool,
                    eligible: &s.alive,
                    seed: c.seed,
                    step: done as u64,
                    stream: Stream::Distance,
                },
                cx,
            )
            .await?;
        for (i, alive) in s.alive.iter().enumerate() {
            if *alive {
                for donor in result.row(i) {
                    counts[i][donor as usize] += 1;
                }
            }
        }
        done += draws;
    }
    let mut laws = Vec::new();
    let mut checks = Vec::new();
    for (i, row_counts) in counts.iter().enumerate() {
        if !s.alive[i] {
            continue;
        }
        let eligible = (0..k)
            .filter(|&j| k == 1 || pool.sources[j].slot as usize != i)
            .collect::<Vec<_>>();
        let logs = eligible
            .iter()
            .map(|&j| {
                module.kernel.log_weight(
                    module.distance.compare(
                        &population.observations,
                        i,
                        &pool.population.observations,
                        j,
                    )?,
                    <Distance as AlgorithmicDistance<f64>>::comparison_kind(&module.distance),
                )
            })
            .collect::<Result<Vec<f64>>>()?;
        let max_log = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let log_z = max_log + logs.iter().map(|x| (x - max_log).exp()).sum::<f64>().ln();
        let probabilities = logs.iter().map(|x| (x - log_z).exp()).collect::<Vec<_>>();
        let measured = eligible
            .iter()
            .map(|&j| row_counts[j] as f64 / c.companion_replicates as f64)
            .collect::<Vec<_>>();
        let se = probabilities
            .iter()
            .map(|p| (p * (1. - p) / c.companion_replicates as f64).sqrt())
            .collect::<Vec<_>>();
        let candidates = eligible.len();
        checks.push(identity(&format!("companion_normalization[{i}]"), &["def-eg-frozen-measurements","def-cloning-companion-operator"],
            "Exact native squashed distances and native Gaussian log weights, current alive pool, self excluded except singleton fallback",probabilities.iter().sum::<f64>()-1.));
        checks.push(upper(&format!("companion_log_normalizer_lower[{i}]"), &["def-eg-frozen-measurements"],
            "Global bounded-feature distance gives Z>=candidate_count*exp(-D_alg²/(2epsilon²)) in log scale",(candidates as f64).ln()+log_q_min,log_z));
        checks.push(upper(
            &format!("companion_log_normalizer_upper[{i}]"),
            &["def-eg-frozen-measurements"],
            "Gaussian weights<=1",
            log_z,
            (candidates as f64).ln(),
        ));
        let frequency_status = measured
            .iter()
            .zip(&probabilities)
            .zip(&se)
            .map(|((m, p), se)| {
                if (m - p).abs() <= 6. * se + 6. / c.companion_replicates as f64 {
                    KineticCheckStatus::NotRejected
                } else {
                    KineticCheckStatus::Violated
                }
            })
            .collect::<Vec<_>>();
        let status = if frequency_status
            .iter()
            .all(|s| *s == KineticCheckStatus::NotRejected)
        {
            KineticCheckStatus::NotRejected
        } else {
            KineticCheckStatus::Violated
        };
        laws.push(CompanionRowLaw {
            query_slot: i,
            donor_slots: eligible
                .iter()
                .map(|&j| pool.sources[j].slot as usize)
                .collect(),
            probabilities,
            measured_probabilities: measured,
            standard_errors: se,
            log_normalizer: log_z,
            candidate_count: candidates,
            log_normalizer_lower_bound: (candidates as f64).ln() + log_q_min,
            log_normalizer_upper_bound: (candidates as f64).ln(),
            minimum_probability_lower_bound: (log_q_min - (candidates as f64).ln())
                .exp()
                .gt(&0.)
                .then(|| (log_q_min - (candidates as f64).ln()).exp()),
            frequency_status,
            status,
            source_labels: vec![
                "def-eg-frozen-measurements".into(),
                "def-cloning-companion-operator".into(),
            ],
        });
    }
    Ok((laws, checks))
}

/// Uniform discrete optimal transport by an integer min-cost flow: each left
/// point supplies k_right units, each right demands k_left units, and each unit
/// represents mass 1/(k_left*k_right). Fractional empirical weights are exact.
pub fn uniform_transport(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    cost: impl Fn(&[f64], &[f64]) -> f64,
) -> Result<(f64, Vec<Vec<f64>>)> {
    let (l, r) = (left.len(), right.len());
    require(
        l > 0 && r > 0 && l <= 64 && r <= 64,
        "uniform transport supports nonempty <=64-point measures",
    )?;
    let dimension = left[0].len();
    require(
        dimension > 0
            && left
                .iter()
                .chain(right)
                .all(|p| p.len() == dimension && p.iter().all(|x| x.is_finite())),
        "transport point dimensions/numerics",
    )?;
    let source = l + r;
    let sink = source + 1;
    let vertices = sink + 1;
    let mut graph = vec![Vec::<FlowEdge>::new(); vertices];
    for i in 0..l {
        add_edge(&mut graph, source, i, r, 0.);
    }
    for j in 0..r {
        add_edge(&mut graph, l + j, sink, l, 0.);
    }
    let mut references = vec![vec![0; r]; l];
    for i in 0..l {
        for j in 0..r {
            let value = cost(&left[i], &right[j]);
            require(
                value.is_finite() && value >= -1e-12,
                "transport cost must be finite and nonnegative",
            )?;
            references[i][j] = graph[i].len();
            add_edge(&mut graph, i, l + j, l * r, value.max(0.));
        }
    }
    let mut remaining = l * r;
    let mut objective = 0.;
    while remaining > 0 {
        let mut distances = vec![f64::INFINITY; vertices];
        distances[source] = 0.;
        let mut parents = vec![None; vertices];
        for _ in 0..vertices - 1 {
            let mut changed = false;
            for u in 0..vertices {
                if !distances[u].is_finite() {
                    continue;
                }
                for (edge_index, e) in graph[u].iter().enumerate() {
                    if e.capacity > 0 {
                        let value = distances[u] + e.cost;
                        if value < distances[e.target] - 1e-14 * (1. + value.abs()) {
                            distances[e.target] = value;
                            parents[e.target] = Some((u, edge_index));
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        require(
            parents[sink].is_some(),
            "transport flow has no residual augmenting path",
        )?;
        let mut amount = remaining;
        let mut vertex = sink;
        let mut path_length = 0;
        while vertex != source {
            let (u, e) = parents[vertex]
                .ok_or_else(|| GasError::Numerical("broken transport augmenting path".into()))?;
            amount = amount.min(graph[u][e].capacity);
            vertex = u;
            path_length += 1;
            require(path_length <= vertices, "cyclic transport augmenting path")?;
        }
        vertex = sink;
        while vertex != source {
            let (u, e) = parents[vertex].expect("checked transport path");
            let target = graph[u][e].target;
            let reverse = graph[u][e].reverse;
            graph[u][e].capacity -= amount;
            graph[target][reverse].capacity += amount;
            vertex = u;
        }
        objective += amount as f64 * distances[sink];
        remaining -= amount;
    }
    let plan = (0..l)
        .map(|i| {
            (0..r)
                .map(|j| {
                    let e = &graph[i][references[i][j]];
                    graph[e.target][e.reverse].capacity as f64 / (l * r) as f64
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok((objective / (l * r) as f64, plan))
}
#[derive(Clone)]
struct FlowEdge {
    target: usize,
    reverse: usize,
    capacity: usize,
    cost: f64,
}
fn add_edge(graph: &mut [Vec<FlowEdge>], u: usize, v: usize, capacity: usize, cost: f64) {
    let (u_index, v_index) = (graph[u].len(), graph[v].len());
    graph[u].push(FlowEdge {
        target: v,
        reverse: v_index,
        capacity,
        cost,
    });
    graph[v].push(FlowEdge {
        target: u,
        reverse: u_index,
        capacity: 0,
        cost: -cost,
    });
}
fn euclidean_cost(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}
fn hypocoercive_cost(a: &[f64], b: &[f64], d: usize, lambda: f64, cross: f64) -> f64 {
    (0..d)
        .map(|j| {
            let x = a[j] - b[j];
            let v = a[d + j] - b[d + j];
            x * x + lambda * v * v + cross * x * v
        })
        .sum()
}
fn norm_sq(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}
fn mean_point(points: &[Vec<f64>]) -> Vec<f64> {
    (0..points[0].len())
        .map(|j| points.iter().map(|x| x[j]).sum::<f64>() / points.len() as f64)
        .collect()
}
fn center_points(points: &[Vec<f64>], center: &[f64]) -> Vec<Vec<f64>> {
    points
        .iter()
        .map(|p| p.iter().zip(center).map(|(x, c)| x - c).collect())
        .collect()
}
fn upper(id: &str, labels: &[&str], scope: &str, observed: f64, bound: f64) -> BoundCheck {
    BoundCheck::upper(id, labels, scope, observed, bound)
}
fn identity(id: &str, labels: &[&str], scope: &str, residual: f64) -> BoundCheck {
    upper(id, labels, scope, residual.abs(), 0.)
}
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(GasError::Configuration(message.into()))
    }
}

fn source_constants(c: &LyapunovValidationConfig, min: f64, max: f64) -> Vec<SourceMappedConstant> {
    [
        ("lambda_minus",min,"(lambda_v-b²/4)/lambda_plus",CH3,"lem-V-coercive","Physical quadratic form q, b²<4lambda_v; lower-bounds optimal Euclidean transport between alive empirical measures"),
        ("lambda_plus",max,"(1+lambda_v+sqrt((1-lambda_v)²+b²))/2",CH3,"lem-V-coercive","Physical quadratic form q, b²<4lambda_v"),
        ("c_V",c.variance_weight,"Supplied positive Lyapunov variance weight",CH3,"def-full-synergistic-lyapunov-function","Defines the observable; no compositional contraction condition is inferred"),
        ("c_B",c.boundary_weight,"Supplied positive Lyapunov barrier weight",CH3,"def-full-synergistic-lyapunov-function","Requires explicitly supplied barrier values; no boundary drift is inferred"),
        ("lambda_status",c.status_weight,"Supplied status mismatch weight",CH1,"def-n-particle-displacement-metric","Uniform all-slot marked empirical transport, optimizing projected phase-space plus mark-mismatch cost over couplings"),
        ("log_q_min",-2.*(c.position_projection_radius.powi(2)+c.lambda_v*c.velocity_projection_radius.powi(2))/c.companion_width.powi(2),"-D_alg²/(2*epsilon²)",CH2,"def-eg-frozen-measurements","Gaussian kernel on bounded squashed features; log scale preserves a positive mathematical floor when exp underflows"),
    ].into_iter().map(|(symbol,value,formula,chapter,label,hypotheses)|SourceMappedConstant {
        symbol:symbol.into(),value:Some(value),formula:formula.into(),source:format!("{chapter}#{label}"),hypotheses:hypotheses.into(),
    }).collect()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LyapunovValidationCase {
    pub name: String,
    pub config: LyapunovValidationConfig,
    pub left: LyapunovSwarm,
    pub right: LyapunovSwarm,
}
pub fn default_lyapunov_cases(replicates: usize, seed: u64) -> Vec<LyapunovValidationCase> {
    let base = LyapunovValidationConfig {
        companion_replicates: replicates,
        seed,
        ..Default::default()
    };
    let make = |n: usize| {
        let positions = (0..n)
            .map(|i| vec![0.8 * (i as f64 * 0.7).sin(), 0.7 * (i as f64 * 0.9).cos()])
            .collect::<Vec<_>>();
        let barrier_values = Some(
            logarithmic_box_barrier_values(&positions, &vec![true; n], 2.)
                .expect("interior default fixture"),
        );
        LyapunovSwarm {
            positions,
            velocities: (0..n)
                .map(|i| vec![0.3 * (i as f64 * 0.3).cos(), 0.2 * (i as f64 * 0.5).sin()])
                .collect(),
            alive: vec![true; n],
            barrier_values,
        }
    };
    let broad = make(8);
    let mut permuted = broad.clone();
    permuted.positions.reverse();
    permuted.velocities.reverse();
    permuted
        .barrier_values
        .as_mut()
        .expect("default barrier")
        .reverse();
    let mut shifted = broad.clone();
    for row in &mut shifted.positions {
        row[0] += 0.2;
    }
    shifted.barrier_values = Some(
        logarithmic_box_barrier_values(&shifted.positions, &shifted.alive, 2.)
            .expect("translated interior fixture"),
    );
    let mut left = make(8);
    left.alive[0] = false;
    left.alive[1] = false;
    let mut right = shifted.clone();
    right.alive[4] = false;
    vec![
        LyapunovValidationCase {
            name: "permuted_identical_alive_measures".into(),
            config: base.clone(),
            left: broad.clone(),
            right: permuted,
        },
        LyapunovValidationCase {
            name: "translated_same_shape".into(),
            config: LyapunovValidationConfig {
                seed: seed.wrapping_add(1),
                ..base.clone()
            },
            left: broad,
            right: shifted,
        },
        LyapunovValidationCase {
            name: "unequal_alive_counts".into(),
            config: LyapunovValidationConfig {
                seed: seed.wrapping_add(2),
                ..base
            },
            left,
            right,
        },
    ]
}

/// Explicit auxiliary barrier 1-sum_j log(1-(x_j/r)²) on the open box.
/// It is positive, smooth in the interior and diverges at the boundary.
/// The box's corners do not satisfy chapter 3's smooth-domain EG-0 premise;
/// these finite values are therefore not a certificate of that premise or of
/// a boundary drift/integrability theorem for the completed gas.
pub fn logarithmic_box_barrier_values(
    positions: &[Vec<f64>],
    alive: &[bool],
    radius: f64,
) -> Result<Vec<f64>> {
    require(
        radius.is_finite() && radius > 0. && positions.len() == alive.len(),
        "logarithmic barrier radius/shape",
    )?;
    positions
        .iter()
        .zip(alive)
        .map(|(row, active)| {
            if !*active {
                return Ok(0.);
            }
            require(
                !row.is_empty() && row.iter().all(|x| x.is_finite() && x.abs() < radius),
                "alive logarithmic barrier input must be strictly inside its box",
            )?;
            let value = 1.
                - row
                    .iter()
                    .map(|x| (-(x / radius).powi(2)).ln_1p())
                    .sum::<f64>();
            require(
                value.is_finite() && value > 0.,
                "logarithmic barrier evaluation overflow",
            )?;
            Ok(value)
        })
        .collect()
}

pub async fn lyapunov_validation_cases(
    samples: usize,
    seed: u64,
) -> Result<Vec<LyapunovValidationReport>> {
    let mut reports = Vec::new();
    for case in default_lyapunov_cases(samples, seed) {
        let mut report = validate_lyapunov(&case.config, &case.left, &case.right).await?;
        let slot = case
            .left
            .alive
            .iter()
            .position(|a| *a)
            .expect("validated alive fixture");
        let left = KineticInput {
            positions: case.left.positions[slot].clone(),
            velocities: case.left.velocities[slot].clone(),
        };
        let mut right = left.clone();
        right.positions[0] += 1e-4;
        right.velocities[0] += 1e-4;
        let kernel = KineticValidationConfig {
            name: format!("{}_local_probability_probe", case.name),
            dimensions: report.dimensions,
            landscape: ConfiningLandscape::quadratic(report.dimensions),
            metric_weight: case.config.lambda_v,
            position_projection_radius: case.config.position_projection_radius,
            velocity_cap: case.config.velocity_projection_radius,
            box_half_width: None,
            inputs: vec![left.clone(), right.clone()],
            samples: 32,
            seed: case.config.seed,
            ..Default::default()
        };
        report
            .boundary_probability_diagnostics
            .push(boundary_probability_separation(&kernel, &left, &right)?);
        reports.push(report);
    }
    Ok(reports)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryProbabilitySeparation {
    /// Defines the analytic kinetic positional kernel used for this probe;
    /// no native sampling is claimed by this deterministic calculation.
    pub kinetic_configuration: KineticValidationConfig,
    pub physical_distance: f64,
    pub projected_distance: f64,
    pub gaussian_position_mean_separation: f64,
    pub gaussian_total_variation: f64,
    pub physical_probability_lipschitz_bound: f64,
    pub compact_inverse_projection_lipschitz: f64,
    pub compact_projected_probability_lipschitz_bound: f64,
    pub compact_position_radius: f64,
    pub compact_velocity_radius: f64,
    pub checks: Vec<BoundCheck>,
    pub scope: String,
}

/// Deterministic probability-distance calculation for the same Gaussian
/// positional law already measured by the native kinetic experiment. No
/// terminal survival conditioning or arbitrary domain perimeter is introduced.
pub fn boundary_probability_separation(
    kernel: &KineticValidationConfig,
    left: &KineticInput,
    right: &KineticInput,
) -> Result<BoundaryProbabilitySeparation> {
    let constants = kinetic_constants(kernel)?;
    let d = kernel.dimensions;
    require(
        left.positions.len() == d
            && right.positions.len() == d
            && left.velocities.len() == d
            && right.velocities.len() == d,
        "boundary probability probe input dimensions",
    )?;
    let l_death = constants.death_probability_lipschitz.ok_or_else(|| {
        GasError::Configuration("boundary probability Lipschitz requires s_h>0".into())
    })?;
    let observations = |input: &KineticInput| -> Result<ObservationBatch<f64>> {
        let mut obs =
            ObservationBatch::positions(TensorBatch::vectors(1, d, input.positions.clone())?);
        obs.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(1, d, input.velocities.clone())?,
        );
        Ok(obs)
    };
    let a = observations(left)?;
    let b = observations(right)?;
    let physical = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 1.,
        velocity_scale: 1.,
        lambda: kernel.metric_weight,
        periodic: None,
    };
    let projected = Distance::SquashedPhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_radius: kernel.position_projection_radius,
        velocity_radius: kernel.velocity_cap,
        lambda: kernel.metric_weight,
    };
    let physical_distance = physical.compare(&a, 0, &b, 0)?;
    let projected_distance = projected.compare(&a, 0, &b, 0)?;
    let mean = |input: &KineticInput| -> Result<Vec<f64>> {
        let gradient = kernel.landscape.analytic_gradient(&input.positions)?;
        Ok((0..d)
            .map(|j| {
                input.positions[j]
                    + constants.position_flow_coefficient
                        * (input.velocities[j] - kernel.dt * gradient[j] / 2.)
            })
            .collect())
    };
    let mean_separation = euclidean_cost(&mean(left)?, &mean(right)?).sqrt();
    let total_variation =
        2. * normal_cdf(mean_separation / (2. * constants.position_variance.sqrt())) - 1.;
    let position_radius = norm_sq(&left.positions)
        .sqrt()
        .max(norm_sq(&right.positions).sqrt());
    let velocity_radius = norm_sq(&left.velocities)
        .sqrt()
        .max(norm_sq(&right.velocities).sqrt());
    let inverse = (1. + position_radius / kernel.position_projection_radius)
        .powi(2)
        .max((1. + velocity_radius / kernel.velocity_cap).powi(2));
    let physical_bound = l_death * physical_distance;
    let projected_bound = l_death * inverse * projected_distance;
    let mut checks = vec![upper(
        "gaussian_terminal_event_lipschitz_physical",
        &["lem-euclidean-boundary-holder"],
        "Total variation dominates every fixed Borel terminal death-event difference; Gaussian CDF approximation has absolute error <=1.5e-7",
        total_variation,
        physical_bound + 1.5e-7,
    )];
    checks.push(upper("gaussian_terminal_event_lipschitz_local_projected", &["lem-euclidean-boundary-holder","lem-euclidean-reward-regularity"],
        "Inverse projection constant on the declared compact physical balls containing both inputs; no global conversion from bounded features is inferred",total_variation,projected_bound+1.5e-7));
    require(
        [
            physical_distance,
            projected_distance,
            mean_separation,
            total_variation,
            inverse,
            physical_bound,
            projected_bound,
        ]
        .iter()
        .all(|x| x.is_finite()),
        "boundary probability separation overflow",
    )?;
    Ok(BoundaryProbabilitySeparation {
        kinetic_configuration:kernel.clone(),
        physical_distance,projected_distance,gaussian_position_mean_separation:mean_separation,
        gaussian_total_variation:total_variation.max(0.),physical_probability_lipschitz_bound:physical_bound,
        compact_inverse_projection_lipschitz:inverse,compact_projected_probability_lipschitz_bound:projected_bound,
        compact_position_radius:position_radius,compact_velocity_radius:velocity_radius,checks,
        scope:"Configured analytic force, Gaussian BAOAB positional law and s_h>0. Bounds concern differences of unconditional terminal event probabilities for any fixed Borel domain. The projected bound is local to the stated compact physical position/velocity balls; squashing alone does not supply a global physical inverse constant.".into(),
    })
}

fn normal_cdf(z: f64) -> f64 {
    let x = z.abs();
    let t = 1. / (1. + 0.2316419 * x);
    let polynomial = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = (-0.5 * x * x).exp() / (2. * std::f64::consts::PI).sqrt() * polynomial;
    if z >= 0. { 1. - tail } else { tail }
}
