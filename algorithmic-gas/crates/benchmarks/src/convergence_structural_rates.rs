//! Population-normalized regional error budgets and bounded nonquadratic forces.
//! A regional matrix is a certificate input, never an empirical transition fit.
use algorithmic_gas::{GasError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Conditional within-region contraction; both actual kick queries must stay there.
pub fn local_force_rate(
    h: f64,
    gamma: f64,
    omega: f64,
    beta: f64,
    harmonic_delta: f64,
    residual_lipschitz: f64,
) -> Result<Value> {
    local_force_rate_metric(
        h,
        gamma,
        omega,
        1.,
        beta,
        harmonic_delta,
        residual_lipschitz,
    )
}

/// The same local bound with a general declared positive sector metric.
#[allow(clippy::too_many_arguments)]
pub fn local_force_rate_metric(
    h: f64,
    gamma: f64,
    omega: f64,
    alpha: f64,
    beta: f64,
    harmonic_delta: f64,
    residual_lipschitz: f64,
) -> Result<Value> {
    if ![h, gamma, omega, alpha, beta, harmonic_delta]
        .iter()
        .all(|x| x.is_finite())
        || h <= 0.
        || gamma <= 0.
        || omega <= 0.
        || alpha <= beta * beta
        || harmonic_delta <= 0.
        || harmonic_delta >= 1.
    {
        return Err(invalid(
            "Invalid positive local metric or harmonic certificate",
        ));
    }
    if !residual_lipschitz.is_finite() || residual_lipschitz < 0. {
        return Err(invalid("Invalid local residual modulus"));
    }
    let c = h / 2.;
    let a = (-gamma * h).exp();
    let input_x = 1. / (omega * (alpha - beta * beta)).sqrt();
    let input_v = 1. / (1. - beta * beta / alpha).sqrt();
    let eps1 = residual_lipschitz * input_x;
    let x2 = (1. - omega * c * c * (1. + a)).abs() * input_x
        + c * (1. + a) * input_v
        + c * c * (1. + a) * eps1;
    let eps2 = residual_lipschitz * x2;
    let dx = c * c * (1. + a) * eps1;
    let dv = c * (a - omega * c * c * (1. + a)).abs() * eps1 + c * eps2;
    let lambda_max = ((alpha + 1.) + ((alpha - 1.).powi(2) + 4. * beta * beta).sqrt()) / 2.;
    let residual = (lambda_max * (omega * dx * dx + dv * dv)).sqrt();
    let rho = up(((1. - harmonic_delta).sqrt() + residual).powi(2));
    if !rho.is_finite() {
        return Err(invalid("Local force coefficient overflow"));
    }
    Ok(
        json!({"rho_upper":rho,"contracts":rho<1.,"rate_per_physical_time":if rho<1. {Some(-rho.ln()/h)} else {None},
        "alpha":alpha,"beta":beta,"residual_lipschitz":residual_lipschitz,"query2_separation_factor":x2,"N_independent":true,"arithmetic":"Binary64 formula evaluation; independent interval certificate required for numerical upper-bound certification.",
        "scope":"Conditional on both actual kick query pairs lying in the certified region; escapes require separate regional discrepancy bounds."}),
    )
}

fn invalid(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
fn nonnegative(values: &[f64]) -> bool {
    values.iter().all(|x| x.is_finite() && *x >= 0.)
}
fn up(value: f64) -> f64 {
    if value == 0. { 0. } else { value.next_up() }
}

/// Bounds on *discrepancy transport*, including its dependence on conditioning.
/// Matrix entry [j][i] bounds error transferred from source i to target j.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegionalErrorBudget {
    pub dimensions: usize,
    pub dt: f64,
    pub block_steps: usize,
    pub regions: Vec<String>,
    pub transfer: Vec<Vec<f64>>,
    pub defects: Vec<f64>,
    pub weights: Vec<f64>,
    /// References to proved regional inequalities. Empty references gate claims.
    pub analytic_references: Vec<String>,
}
impl RegionalErrorBudget {
    /// Choose positive regional weights and certify the resulting column bound.
    /// Iteration is only a search: `compile` verifies the final inequality.
    /// In particular, a fitted transition matrix acquires no proof credit.
    pub fn optimize_weights(&self, iterations: usize) -> Result<Value> {
        let baseline = self.compile()?;
        if iterations > 100_000 {
            return Err(invalid(
                "Regional weight search exceeds its iteration budget",
            ));
        }
        let mut optimized = self.clone();
        let n = self.regions.len();
        let mut best = baseline.clone();
        let mut best_weights = self.weights.clone();
        let mut weights = vec![1.; n];
        for _ in 0..iterations {
            let next: Vec<_> = (0..n)
                .map(|i| {
                    weights[i]
                        + (0..n)
                            .map(|j| weights[j] * self.transfer[j][i])
                            .sum::<f64>()
                })
                .collect();
            let scale = next.iter().copied().fold(0., f64::max);
            if !scale.is_finite() || scale <= 0. {
                return Err(invalid("Regional weight search overflow"));
            }
            weights = next.iter().map(|w| w / scale).collect();
            if weights.iter().any(|w| *w <= 0. || !w.is_finite()) {
                return Err(invalid("Regional weight search lost strict positivity"));
            }
            optimized.weights.clone_from(&weights);
            let candidate = optimized.compile()?;
            if candidate["rho_upper"].as_f64().unwrap() < best["rho_upper"].as_f64().unwrap() {
                best = candidate;
                best_weights.clone_from(&weights);
            }
        }
        Ok(
            json!({"baseline":baseline,"optimized":best,"weights":best_weights,
            "iterations":iterations,"scope":"Positive weights are deterministic analysis choices. Final directed column sums certify the supplied matrix; convergence or optimality of the weight search is not assumed."}),
        )
    }

    pub fn compile(&self) -> Result<Value> {
        let n = self.regions.len();
        if n == 0
            || self.dimensions == 0
            || self.block_steps == 0
            || !self.dt.is_finite()
            || self.dt <= 0.
            || self.transfer.len() != n
            || self
                .transfer
                .iter()
                .any(|r| r.len() != n || !nonnegative(r))
            || self.defects.len() != n
            || !nonnegative(&self.defects)
            || self.weights.len() != n
            || self.weights.iter().any(|w| !w.is_finite() || *w <= 0.)
        {
            return Err(invalid(
                "Invalid regional discrepancy matrix, weights or dimensions",
            ));
        }
        let columns: Vec<_> = (0..n)
            .map(|i| {
                let sum = (0..n).fold(0., |sum, j| {
                    up(sum + up(self.weights[j] * self.transfer[j][i]))
                });
                up(sum / self.weights[i])
            })
            .collect();
        let rho = columns.iter().copied().fold(0., f64::max);
        let defect = (0..n).fold(0., |sum, i| up(sum + up(self.weights[i] * self.defects[i])));
        if !rho.is_finite() || !defect.is_finite() {
            return Err(invalid("Regional budget overflow"));
        }
        let references_present = self.analytic_references.len() == n
            && self
                .analytic_references
                .iter()
                .all(|s| !s.trim().is_empty());
        let contracts = rho < 1.;
        let smallest_weight = self.weights.iter().copied().fold(f64::INFINITY, f64::min);
        let largest_weight = self.weights.iter().copied().fold(0., f64::max);
        let unweighted_prefactor = up(largest_weight / smallest_weight);
        let unweighted_floor = if contracts {
            Some(up(up(defect / (1. - rho)) / smallest_weight))
        } else {
            None
        };
        if !unweighted_prefactor.is_finite() || unweighted_floor.is_some_and(|v| !v.is_finite()) {
            return Err(invalid("Regional metric conversion overflow"));
        }
        let time = self.dt * self.block_steps as f64;
        if !time.is_finite() {
            return Err(invalid("Block time overflow"));
        }
        Ok(
            json!({"dimensions":self.dimensions,"regions":self.regions,"column_coefficients_upper":columns,
            "rho_upper":rho,"weighted_defect_upper":defect,"error_floor_upper":if contracts {Some(up(defect/(1.-rho)))} else {None},
            "unweighted_prefactor_upper":unweighted_prefactor,"unweighted_error_floor_upper":unweighted_floor,
            "rate_per_physical_time":if contracts && rho>0. {Some(-rho.ln()/time)} else {None},
            "contracts":contracts,"analytic_inputs_referenced":references_present,
            "certificate_status":if references_present {"conditional_on_referenced_regional_inequalities"} else {"diagnostic_only_missing_regional_proofs"},
            "N_independent":null,"population_uniformity":"Conditional on N-uniform analytic bounds for every supplied transfer, defect and weight.","normalization":"Each regional error is an integral against a probability coupling, or a population average; no sum over particle count.",
            "bound":"E_n <= rho^n E_0 + b sum_{k=0}^{n-1} rho^k. A probability transition matrix alone does not certify discrepancy transfer."}),
        )
    }
}

/// Unbounded state-space truncation from a proved moment; the kernel is unchanged.
pub fn polynomial_tail(
    moment_order: f64,
    moment_bound: f64,
    cutoff: f64,
    cost_order: f64,
) -> Result<Value> {
    if ![moment_order, moment_bound, cutoff, cost_order]
        .iter()
        .all(|x| x.is_finite())
        || moment_order <= cost_order
        || cost_order < 0.
        || moment_bound < 0.
        || cutoff <= 0.
    {
        return Err(invalid(
            "Tail bound requires p>cost_order>=0, M_p>=0 and R>0",
        ));
    }
    let mass = up(moment_bound / cutoff.powf(moment_order));
    let cost = up(moment_bound / cutoff.powf(moment_order - cost_order));
    if !mass.is_finite() || !cost.is_finite() {
        return Err(invalid("Tail bound overflow"));
    }
    Ok(
        json!({"moment_order":moment_order,"moment_bound":moment_bound,"cutoff":cutoff,"cost_order":cost_order,
        "tail_mass_upper":mass.min(1.),"tail_cost_upper":cost,"N_independent":null,"population_uniformity":"Conditional on the supplied moment bound being uniform in N.",
        "scope":"Conditional on the proved population-normalized moment, includes the entire unbounded tail."}),
    )
}

/// Exact BAOAB perturbation of an independently certified harmonic sector LMI.
/// Force is F(x)=-omega*x+e(x). d1/d2 bound differences of e at both kick queries.
pub fn nonquadratic_force_budget(
    h: f64,
    gamma: f64,
    omega: f64,
    beta: f64,
    harmonic_delta: f64,
    first_kick_defect: f64,
    second_kick_defect: f64,
) -> Result<Value> {
    if ![
        h,
        gamma,
        omega,
        beta,
        harmonic_delta,
        first_kick_defect,
        second_kick_defect,
    ]
    .iter()
    .all(|x| x.is_finite())
        || h <= 0.
        || gamma <= 0.
        || omega <= 0.
        || beta.abs() >= 1.
        || harmonic_delta <= 0.
        || harmonic_delta >= 1.
        || !nonnegative(&[first_kick_defect, second_kick_defect])
    {
        return Err(invalid(
            "Invalid harmonic certificate or nonquadratic force defects",
        ));
    }
    let c = h / 2.;
    let a = (-gamma * h).exp();
    let dx = c * c * (1. + a) * first_kick_defect;
    let dv = c * (a - omega * c * c * (1. + a)).abs() * first_kick_defect + c * second_kick_defect;
    let residual = up((1. + beta.abs()) * up(omega * dx * dx + dv * dv));
    let theta = if residual == 0. {
        0.
    } else {
        harmonic_delta / (2. * (1. - harmonic_delta))
    };
    let rho = up((1. + theta) * (1. - harmonic_delta));
    let defect = if theta == 0. {
        0.
    } else {
        up((1. + 1. / theta) * residual)
    };
    if !rho.is_finite() || !defect.is_finite() || rho >= 1. {
        return Err(invalid("Perturbation budget overflow or lost gap"));
    }
    Ok(
        json!({"h":h,"gamma":gamma,"omega":omega,"beta":beta,"harmonic_delta_lower":harmonic_delta,
        "first_kick_defect":first_kick_defect,"second_kick_defect":second_kick_defect,
        "residual_squared_upper":residual,"young_theta":theta,"rho_upper":rho,"additive_defect_upper":defect,
        "rate_per_physical_time":-rho.ln()/h,"error_floor_upper":up(defect/(1.-rho)),"N_independent":true,"arithmetic":"Binary64 formula evaluation; independent interval certificate required for numerical upper-bound certification.",
        "metric":"omega |dx|^2 + 2 beta sqrt(omega) <dx,dv> + |dv|^2",
        "scope":"Shared-noise fixed-cap isotropic BAOAB; requires the harmonic sector LMI and both residual-force difference bounds. Cloning, viscosity and differing environments require their own regional budgets."}),
    )
}

/// Standard Rastrigin: F=-2x-20pi sin(2pi x), globally nonconvex.
/// Radius refers to a bound on the distance between two kick queries, not a cutoff.
pub fn rastrigin_force_defect(dimensions: usize, query_separation: Option<f64>) -> Result<Value> {
    if dimensions == 0 || query_separation.is_some_and(|r| !r.is_finite() || r < 0.) {
        return Err(invalid(
            "Invalid Rastrigin dimension or kick-query separation",
        ));
    }
    let amplitude = 20. * std::f64::consts::PI * (dimensions as f64).sqrt();
    let lipschitz = 40. * std::f64::consts::PI.powi(2);
    let defect = query_separation.map_or(2. * amplitude, |r| (lipschitz * r).min(2. * amplitude));
    if !amplitude.is_finite() || !defect.is_finite() {
        return Err(invalid("Rastrigin profile overflow"));
    }
    Ok(
        json!({"dimensions":dimensions,"harmonic_curvature":2.,"residual_amplitude":up(amplitude),
        "residual_lipschitz":up(lipschitz),"residual_difference_upper":up(defect),
        "query_separation_bound":query_separation,"reward_quadratic_coefficient":1.,"reward_offset":20.*dimensions as f64,
        "radial_coercivity":"For 0<k<2: b_Rd(k) <= (20pi)^2 d / [4(2-k)].",
        "scope":"Analytic global profiles, not sampled force extrema. No convexity assumption."}),
    )
}
