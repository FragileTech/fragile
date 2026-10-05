//! Native complete-update error trajectories, fixed contraction controls and
//! independent-noise floors. Empirical swarm measures have no intrinsic labels.
use crate::{
    Benchmark, BenchmarkModel, convergence_framework::BoundCheck,
    convergence_kinetic::KineticCheckStatus, convergence_lyapunov::uniform_transport,
};
use algorithmic_gas::{
    AlgorithmicGas, GasBuilder, GasConfig, GasError, ObservationBatch, Population, Result,
    TensorBatch,
    boundary::BoundaryPolicy,
    noise::{FactorValues, NoiseGeometry},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};

type Matrix = [[f64; 2]; 2];
const SOURCE: &str = "def-eg-baoab-canonical";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecayProfile {
    LinearQuadraticNoiseless,
    LinearSphereNoiseless,
    LinearQuadraticIndependentNoise,
    CanonicalQuadratic,
    CanonicalRastrigin,
    RastriginDiffusionCalibration,
    RastriginCloningCalibration,
}
impl DecayProfile {
    fn control(self) -> bool {
        matches!(
            self,
            Self::LinearQuadraticNoiseless
                | Self::LinearSphereNoiseless
                | Self::LinearQuadraticIndependentNoise
        )
    }
    fn curvature(self) -> f64 {
        if self == Self::LinearSphereNoiseless {
            2.
        } else {
            1.
        }
    }
    fn landscape(self) -> Benchmark {
        match self {
            Self::LinearSphereNoiseless => Benchmark::Sphere,
            Self::CanonicalRastrigin
            | Self::RastriginDiffusionCalibration
            | Self::RastriginCloningCalibration => Benchmark::Rastrigin,
            _ => Benchmark::Quadratic,
        }
    }
    fn random_address(self) -> u64 {
        match self {
            Self::LinearQuadraticNoiseless => 0,
            Self::LinearSphereNoiseless => 1,
            Self::LinearQuadraticIndependentNoise => 2,
            Self::CanonicalQuadratic => 3,
            Self::CanonicalRastrigin
            | Self::RastriginDiffusionCalibration
            | Self::RastriginCloningCalibration => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairRandomness {
    #[default]
    Independent,
    Shared,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecayValidationConfig {
    pub walkers: Vec<usize>,
    pub dimensions: Vec<usize>,
    /// Independent pair-replicate addresses; the two native engines use
    /// disjoint seed addresses within every pair.
    pub seeds: Vec<u64>,
    pub steps: usize,
    pub dt: f64,
    /// The fit always starts at step zero. No data-dependent window selection.
    pub fit_end_step: usize,
    pub profiles: Vec<DecayProfile>,
    /// Canonical cases begin with one shared outside atom, ensuring completed
    /// revival is represented. This is false for every linear control.
    pub canonical_initial_dead_atom: bool,
    /// Only the explicitly named diffusion-calibration profile changes this
    /// parameter. At the default h=.04, well spacing one and per-step standard
    /// deviation one quarter of that spacing give sigma_pos=1.25.
    pub calibration_position_diffusion: f64,
    /// Only the explicitly named cloning-calibration profile changes this
    /// accepted-copy/revival jitter amplitude; kinetic diffusion is retained.
    pub calibration_jitter_amplitude: f64,
    pub pair_randomness: PairRandomness,
}
impl Default for DecayValidationConfig {
    fn default() -> Self {
        Self {
            walkers: vec![4, 16, 64],
            dimensions: vec![1, 2],
            seeds: vec![7, 173, 1729, 7919, 104729, 31337, 65537, 99991],
            steps: 128,
            dt: 0.04,
            fit_end_step: 64,
            profiles: vec![
                DecayProfile::LinearQuadraticNoiseless,
                DecayProfile::LinearSphereNoiseless,
                DecayProfile::LinearQuadraticIndependentNoise,
                DecayProfile::CanonicalQuadratic,
                DecayProfile::CanonicalRastrigin,
            ],
            canonical_initial_dead_atom: true,
            calibration_position_diffusion: 1.25,
            calibration_jitter_amplitude: 0.1,
            pair_randomness: PairRandomness::Independent,
        }
    }
}
impl DecayValidationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.walkers.is_empty()
            || self.dimensions.is_empty()
            || self.seeds.len() < 2
            || self.profiles.is_empty()
            || self.walkers.iter().any(|n| !(2..=64).contains(n))
            || self.dimensions.iter().any(|d| !(1..=8).contains(d))
            || !(2..=512).contains(&self.steps)
            || !(2..=self.steps).contains(&self.fit_end_step)
            || !self.dt.is_finite()
            || !(0.001..=0.5).contains(&self.dt)
            || self.seeds.len() > 64
            || !self.calibration_position_diffusion.is_finite()
            || self.calibration_position_diffusion < 0.
            || !self.calibration_jitter_amplitude.is_finite()
            || self.calibration_jitter_amplitude < 0.
        {
            return Err(configuration(
                "decay axes require N=2..64, d=1..8, 2..64 distinct seed pairs, 2..512 steps, a fixed fit end in 2..steps, dt=.001..=.5 and finite nonnegative calibration diffusion/jitter",
            ));
        }
        let mut seeds = self.seeds.clone();
        seeds.sort_unstable();
        seeds.dedup();
        if seeds.len() != self.seeds.len() {
            return Err(configuration(
                "decay ensemble seed addresses must be distinct",
            ));
        }
        let cases = self
            .walkers
            .len()
            .checked_mul(self.dimensions.len())
            .and_then(|n| n.checked_mul(self.profiles.len()))
            .and_then(|n| n.checked_mul(self.seeds.len()));
        if cases.is_none_or(|n| n > 2048) {
            return Err(configuration("decay matrix exceeds 2048 independent pairs"));
        }
        Ok(())
    }
}
fn configuration(message: &str) -> GasError {
    GasError::Configuration(message.into())
}
/// A prescribed Gaussian exploration length per coordinate. This computes a
/// parameter from geometry and h, without fitting a measured contraction rate.
pub fn diffusion_from_well_spacing(
    dt: f64,
    nominal_well_spacing: f64,
    step_standard_deviation_fraction: f64,
) -> Result<f64> {
    if [dt, nominal_well_spacing, step_standard_deviation_fraction]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
    {
        return Err(configuration(
            "diffusion design needs positive finite timestep, spacing and exploration fraction",
        ));
    }
    let diffusion = nominal_well_spacing * step_standard_deviation_fraction / dt.sqrt();
    if !diffusion.is_finite() || diffusion <= 0. {
        return Err(configuration(
            "diffusion design is outside floating-point representability",
        ));
    }
    Ok(diffusion)
}

/// Re-index a numerical swarm representative without changing its empirical
/// state or native RNG step. Every first-class per-atom field is gathered.
pub fn canonical_decay_population(population: &Population<f64>) -> Result<Population<f64>> {
    population.validate()?;
    let keys = (0..population.len())
        .map(|i| {
            population
                .observations
                .fields
                .values()
                .map(|field| field.row(i).map(<[f64]>::to_vec))
                .collect::<Result<Vec<_>>>()
                .map(|fields| fields.concat())
        })
        .collect::<Result<Vec<_>>>()?;
    if keys.iter().flatten().any(|x| !x.is_finite()) {
        return Err(configuration(
            "canonical decay representative needs finite intrinsic observations",
        ));
    }
    let eligible = population.eligible(false);
    let mut indices = (0..population.len()).collect::<Vec<_>>();
    indices.sort_by(|&i, &j| {
        keys[i]
            .iter()
            .zip(&keys[j])
            .find_map(|(a, b)| {
                let order = a.total_cmp(b);
                (!order.is_eq()).then_some(order)
            })
            .unwrap_or_else(|| eligible[i].cmp(&eligible[j]))
    });
    let gather = indices.iter().map(|&i| i as u32).collect::<Vec<_>>();
    let mut reordered = population.clone();
    reordered.observations = population.observations.gather(&gather)?;
    reordered.rewards = population.rewards.gather(&gather)?;
    reordered.validity = indices.iter().map(|&i| population.validity[i]).collect();
    reordered.generations = indices.iter().map(|&i| population.generations[i]).collect();
    reordered.states = population
        .states
        .as_ref()
        .map(|states| states.gather(&gather))
        .transpose()?;
    reordered.validate()?;
    Ok(reordered)
}

/// Law-valid representative refresh for the memory-free numerical presets used
/// here. Checkpoint restoration retains step/version/provenance/counters and
/// removes only the obsolete storage-addressed last display report.
pub fn canonicalize_decay_engine(engine: &mut AlgorithmicGas<f64>) -> Result<()> {
    let mut checkpoint = engine.checkpoint();
    if checkpoint.config.n_elite != 0
        || checkpoint.config.distance_donors.history_window != 0
        || checkpoint.config.cloning_donors.history_window != 0
        || checkpoint.config.geometry.is_some()
        || !checkpoint.history.is_empty()
        || checkpoint.graph.is_some()
        || checkpoint.recording.is_some()
    {
        return Err(configuration(
            "shared decay coupling requires memory-free presets without elites, donor history, geometry or recording",
        ));
    }
    checkpoint.population = canonical_decay_population(&checkpoint.population)?;
    checkpoint.last_report = None;
    engine.restore(checkpoint)
}
fn transpose(a: Matrix) -> Matrix {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}
fn multiply(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j]))
}
fn add(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}
fn vector(a: Matrix, x: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * x[0] + a[0][1] * x[1],
        a[1][0] * x[0] + a[1][1] * x[1],
    ]
}
fn quadratic(p: Matrix, x: [f64; 2]) -> f64 {
    p[0][0] * (x[0] + p[0][1] / p[0][0] * x[1]).powi(2)
        + (p[1][1] - p[0][1] * p[0][1] / p[0][0]) * x[1] * x[1]
}
fn trace_product(a: Matrix, b: Matrix) -> f64 {
    a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[1][0] * b[0][1] + a[1][1] * b[1][1]
}
fn eigenvalues(a: Matrix) -> [f64; 2] {
    let spread = ((a[0][0] - a[1][1]).powi(2) + 4. * a[0][1] * a[0][1]).sqrt();
    [
        (a[0][0] + a[1][1] - spread) / 2.,
        (a[0][0] + a[1][1] + spread) / 2.,
    ]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearDecayConstants {
    pub dt: f64,
    pub friction: f64,
    pub curvature: f64,
    pub thermostat_scale: f64,
    pub position_diffusion: f64,
    pub map: Matrix,
    pub innovation_covariance: Matrix,
    /// Normalized so the x² coefficient is one.
    pub lyapunov_matrix: Matrix,
    pub lyapunov_identity_rhs: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub contraction_factor: f64,
    /// Rate of squared error, not its square root.
    pub squared_error_decay_rate: f64,
    pub lyapunov_identity_residual: f64,
    pub scope: String,
}

/// Exact linear BAOAB control. P is computed before observing trajectories;
/// Aᵀ P A=P-rI implies q(Az)<=rho q(z) with rho=1-r/lambda_max(P).
pub fn linear_decay_constants(
    dt: f64,
    friction: f64,
    curvature: f64,
    thermostat_scale: f64,
    position_diffusion: f64,
) -> Result<LinearDecayConstants> {
    if [dt, friction, curvature]
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.)
        || [thermostat_scale, position_diffusion]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
    {
        return Err(configuration(
            "linear decay requires positive finite dt/friction/curvature and nonnegative finite noise scales",
        ));
    }
    let c = (-friction * dt).exp();
    let b = dt * (1. + c) / 2.;
    let a = 1. - b * dt * curvature / 2.;
    let map = [
        [a, b],
        [-dt * curvature * (c + a) / 2., c - dt * curvature * b / 2.],
    ];
    let [[a, b], [c, d]] = map;
    let mut equations = [
        [a * a - 1., 2. * a * c, c * c, -1.],
        [a * b, a * d + b * c - 1., c * d, 0.],
        [b * b, 2. * b * d, d * d - 1., -1.],
    ];
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&l, &r| equations[l][col].abs().total_cmp(&equations[r][col].abs()))
            .unwrap();
        equations.swap(col, pivot);
        if !equations[col][col].is_finite() || equations[col][col].abs() < 1e-16 {
            return Err(configuration(
                "linear decay Lyapunov system is singular or numerically unresolved",
            ));
        }
        let scale = equations[col][col];
        for value in &mut equations[col][col..] {
            *value /= scale;
        }
        let row = equations[col];
        for (i, equation) in equations.iter_mut().enumerate() {
            if i != col {
                let factor = equation[col];
                for j in col..4 {
                    equation[j] -= factor * row[j];
                }
            }
        }
    }
    let raw = [
        [equations[0][3], equations[1][3]],
        [equations[1][3], equations[2][3]],
    ];
    let rhs = 1. / raw[0][0];
    let p = std::array::from_fn(|i| std::array::from_fn(|j| raw[i][j] / raw[0][0]));
    let [lambda_min, lambda_max] = eigenvalues(p);
    let rho = 1. - rhs / lambda_max;
    if p.iter().flatten().any(|x| !x.is_finite()) || lambda_min <= 0. || !(0. ..1.).contains(&rho) {
        return Err(configuration(
            "linear control has no positive resolved contracting Lyapunov matrix",
        ));
    }
    let evolved = multiply(multiply(transpose(map), p), map);
    let residual = (0..2)
        .flat_map(|i| {
            (0..2).map(move |j| (evolved[i][j] - p[i][j] + if i == j { rhs } else { 0. }).abs())
        })
        .fold(0., f64::max);
    let q_squared = thermostat_scale.powi(2) * (-(-2. * friction * dt).exp_m1()) / (2. * friction);
    let noise_x = dt / 2.;
    let noise_v = 1. - dt * dt * curvature / 4.;
    let covariance = [
        [
            noise_x * noise_x * q_squared + position_diffusion.powi(2) * dt,
            noise_x * noise_v * q_squared,
        ],
        [noise_x * noise_v * q_squared, noise_v * noise_v * q_squared],
    ];
    if covariance.iter().flatten().any(|x| !x.is_finite()) {
        return Err(configuration("linear innovation covariance overflow"));
    }
    Ok(LinearDecayConstants {dt,friction,curvature,thermostat_scale,position_diffusion,map,
        innovation_covariance:covariance,lyapunov_matrix:p,lyapunov_identity_rhs:rhs,
        lambda_min,lambda_max,contraction_factor:rho,squared_error_decay_rate:-rho.ln()/dt,
        lyapunov_identity_residual:residual,
        scope:"Complete native BAOAB extension with quadratic curvature k, unbounded domain, constant fitness (reward/diversity exponents zero), no accepted clones, no clone jitter/collision and no velocity cap. Innovation covariance describes one engine; noises are independent across rows in each marginal, and the experiment records its independent or shared pair coupling. P is solved from AᵀPA=P-rI and fixed before simulations; rho and its squared-error rate are independent of N. These constants do not certify the canonical selected/capped/killed swarm.".into()})
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecayObservables {
    pub full_marked_error: f64,
    pub marked_physical_component: f64,
    pub marked_status_component: f64,
    pub full_physical_wasserstein_squared: f64,
    pub barycenter_error: f64,
    pub structural_error: f64,
    pub alive_wasserstein_squared: Option<f64>,
    pub alive_barycenter_error: Option<f64>,
    pub alive_structural_error: Option<f64>,
    pub alive_counts: [usize; 2],
    pub squared_position_to_minimizer: f64,
}

fn points(population: &Population<f64>) -> Result<(Vec<Vec<f64>>, Vec<bool>)> {
    let x = population.observations.field("positions")?;
    let v = population.observations.field("velocities")?;
    if x.width() != v.width() || x.width() == 0 || population.len() > 64 || population.is_empty() {
        return Err(configuration(
            "decay observables require matching nonempty <=64-row physical phase space",
        ));
    }
    let alive = population.eligible(false);
    let mut atoms = (0..population.len())
        .map(|i| {
            Ok((
                x.row(i)?
                    .iter()
                    .chain(v.row(i)?)
                    .copied()
                    .collect::<Vec<_>>(),
                alive[i],
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if atoms
        .iter()
        .any(|(point, _)| point.iter().any(|x| !x.is_finite()))
    {
        return Err(configuration("nonfinite decay atom"));
    }
    atoms.sort_by(|(a, ma), (b, mb)| {
        a.iter()
            .zip(b)
            .find_map(|(x, y)| {
                let order = x.total_cmp(y);
                (!order.is_eq()).then_some(order)
            })
            .unwrap_or_else(|| ma.cmp(mb))
    });
    Ok(atoms.into_iter().unzip())
}
fn phase_cost(left: &[f64], right: &[f64], p: Matrix) -> f64 {
    let dimension = left.len() / 2;
    (0..dimension)
        .map(|i| {
            quadratic(
                p,
                [
                    left[i] - right[i],
                    left[dimension + i] - right[dimension + i],
                ],
            )
        })
        .sum()
}
fn mean(points: &[Vec<f64>]) -> Vec<f64> {
    (0..points[0].len())
        .map(|j| points.iter().map(|p| p[j]).sum::<f64>() / points.len() as f64)
        .collect()
}

// Hungarian assignment for equal-mass equal-size empirical measures. Intrinsic
// atom ordering removes storage-order dependence before resolving cost ties.
fn assignment(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    left_marks: &[bool],
    right_marks: &[bool],
    p: Matrix,
    status_weight: f64,
) -> Result<(f64, f64)> {
    let n = left.len();
    let cost = left
        .iter()
        .enumerate()
        .map(|(i, a)| {
            right
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    phase_cost(a, b, p)
                        + status_weight * usize::from(left_marks[i] != right_marks[j]) as f64
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if cost.iter().flatten().any(|c| !c.is_finite() || *c < 0.) {
        return Err(configuration("invalid decay assignment cost"));
    }
    let mut u = vec![0.; n + 1];
    let mut v = vec![0.; n + 1];
    let mut owner = vec![0; n + 1];
    let mut way = vec![0; n + 1];
    for i in 1..=n {
        owner[0] = i;
        let mut j0 = 0;
        let mut minima = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = owner[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let residual = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if residual < minima[j] {
                        minima[j] = residual;
                        way[j] = j0;
                    }
                    if minima[j] < delta {
                        delta = minima[j];
                        j1 = j;
                    }
                }
            }
            if !delta.is_finite() {
                return Err(configuration("unresolved decay optimal assignment"));
            }
            for j in 0..=n {
                if used[j] {
                    u[owner[j]] += delta;
                    v[j] -= delta;
                } else {
                    minima[j] -= delta;
                }
            }
            j0 = j1;
            if owner[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            owner[j0] = owner[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut physical = 0.;
    let mut marked = 0.;
    for j in 1..=n {
        let i = owner[j] - 1;
        physical += phase_cost(&left[i], &right[j - 1], p);
        marked += status_weight * usize::from(left_marks[i] != right_marks[j - 1]) as f64;
    }
    Ok((physical / n as f64, marked / n as f64))
}

/// N-normalized optimal marked transport on all retained atoms, with a
/// separate physical empirical W2 decomposition and alive-only observables.
pub fn decay_observables(
    left: &Population<f64>,
    right: &Population<f64>,
    p: Matrix,
    status_weight: f64,
) -> Result<DecayObservables> {
    let eigen = eigenvalues(p);
    if !status_weight.is_finite()
        || status_weight <= 0.
        || p.iter().flatten().any(|x| !x.is_finite())
        || (p[0][1] - p[1][0]).abs() > 1e-12
        || eigen[0] <= 0.
        || left.len() != right.len()
    {
        return Err(configuration(
            "decay transport needs symmetric positive definite P, positive status weight and equal N",
        ));
    }
    let (a, ma) = points(left)?;
    let (b, mb) = points(right)?;
    if a[0].len() != b[0].len() {
        return Err(configuration("paired decay dimensions differ"));
    }
    let (physical, _) = assignment(&a, &b, &ma, &mb, p, 0.)?;
    let (marked_physical, marked_status) = if ma.iter().all(|m| *m) && mb.iter().all(|m| *m) {
        (physical, 0.)
    } else {
        assignment(&a, &b, &ma, &mb, p, status_weight)?
    };
    let bary = phase_cost(&mean(&a), &mean(&b), p);
    let alive_a = a
        .iter()
        .zip(&ma)
        .filter(|(_, m)| **m)
        .map(|(p, _)| p.clone())
        .collect::<Vec<_>>();
    let alive_b = b
        .iter()
        .zip(&mb)
        .filter(|(_, m)| **m)
        .map(|(p, _)| p.clone())
        .collect::<Vec<_>>();
    let alive_counts = [alive_a.len(), alive_b.len()];
    let (alive_w, alive_bary) = if alive_a.is_empty() || alive_b.is_empty() {
        (None, None)
    } else {
        let value = if alive_a.len() == a.len() && alive_b.len() == b.len() {
            physical
        } else if alive_a.len() == alive_b.len() {
            assignment(
                &alive_a,
                &alive_b,
                &vec![true; alive_a.len()],
                &vec![true; alive_b.len()],
                p,
                0.,
            )?
            .0
        } else {
            uniform_transport(&alive_a, &alive_b, |x, y| phase_cost(x, y, p))?.0
        };
        (
            Some(value),
            Some(phase_cost(&mean(&alive_a), &mean(&alive_b), p)),
        )
    };
    let dimension = a[0].len() / 2;
    let minimizer = a
        .iter()
        .chain(&b)
        .map(|point| point[..dimension].iter().map(|x| x * x).sum::<f64>())
        .sum::<f64>()
        / (2 * a.len()) as f64;
    Ok(DecayObservables {
        full_marked_error: marked_physical + marked_status,
        marked_physical_component: marked_physical,
        marked_status_component: marked_status,
        full_physical_wasserstein_squared: physical,
        barycenter_error: bary,
        structural_error: (physical - bary).max(0.),
        alive_wasserstein_squared: alive_w,
        alive_barycenter_error: alive_bary,
        alive_structural_error: alive_w.zip(alive_bary).map(|(w, l)| (w - l).max(0.)),
        alive_counts,
        squared_position_to_minimizer: minimizer,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecayTimePoint {
    pub step: usize,
    pub time: f64,
    pub observables: DecayObservables,
    pub revived: [usize; 2],
    pub cloned: [usize; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecaySeedTrajectory {
    pub pair_seed: u64,
    pub native_engine_seeds: [u64; 2],
    pub points: Vec<DecayTimePoint>,
    pub termination: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnsembleValue {
    pub mean: f64,
    pub standard_error: f64,
    pub independent_pairs: usize,
}
fn ensemble(values: &[f64]) -> Option<EnsembleValue> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = if values.len() > 1 {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.)
    } else {
        0.
    };
    Some(EnsembleValue {
        mean,
        standard_error: (variance / n).sqrt(),
        independent_pairs: values.len(),
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecayEnsemblePoint {
    pub step: usize,
    pub time: f64,
    pub completed_pairs: usize,
    pub full_marked_error: Option<EnsembleValue>,
    pub barycenter_error: Option<EnsembleValue>,
    pub structural_error: Option<EnsembleValue>,
    pub alive_error: Option<EnsembleValue>,
    pub squared_position_to_minimizer: Option<EnsembleValue>,
    pub exact_barycenter_expectation: Option<f64>,
    pub exact_barycenter_noise_floor: Option<f64>,
    pub n_independent_transport_expectation_upper: Option<f64>,
    pub n_independent_geometric_noise_envelope: Option<f64>,
    pub noiseless_geometric_envelope: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecayStatisticalCheck {
    pub id: String,
    pub source_labels: Vec<String>,
    pub scope: String,
    pub observed: f64,
    pub prediction: f64,
    pub standard_error: f64,
    pub z_score: Option<f64>,
    pub status: KineticCheckStatus,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DecayRateEstimate {
    pub observable: String,
    #[serde(default)]
    pub requested_seed_pairs: usize,
    #[serde(default)]
    pub complete_window_seed_pairs: usize,
    #[serde(default)]
    pub missing_window_seed_pairs: usize,
    #[serde(default)]
    pub empty_alive_window_seed_pairs: usize,
    #[serde(default)]
    pub extinct_seed_pairs: usize,
    pub fixed_start_step: usize,
    pub fixed_end_step: usize,
    pub estimated_squared_error_rate: Option<f64>,
    pub seed_jackknife_standard_error: Option<f64>,
    pub jackknife_estimates_available: usize,
    pub applicable: bool,
    pub explanation: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DecayEndpointComparison {
    #[serde(default)]
    pub observable: String,
    #[serde(default)]
    pub requested_seed_pairs: usize,
    pub complete_seed_pairs: usize,
    #[serde(default)]
    pub missing_endpoint_seed_pairs: usize,
    #[serde(default)]
    pub empty_alive_endpoint_seed_pairs: usize,
    #[serde(default)]
    pub extinct_seed_pairs: usize,
    pub mean_initial_error: Option<f64>,
    pub mean_final_error: Option<f64>,
    pub terminal_to_initial_ratio: Option<f64>,
    pub empirical_mean_decreased: Option<bool>,
    pub paired_mean_change: Option<EnsembleValue>,
    pub scope: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffusionCalibration {
    pub nominal_well_spacing: f64,
    pub reference_step_standard_deviation_fraction: f64,
    pub reference_position_diffusion: f64,
    pub selected_position_diffusion: f64,
    pub actual_step_gaussian_standard_deviation: f64,
    pub scope: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CloningCalibration {
    pub baseline_jitter_amplitude: f64,
    pub selected_jitter_amplitude: f64,
    pub gaussian_jitter_variance_per_accepted_coordinate: Option<f64>,
    pub log_gaussian_jitter_variance_per_accepted_coordinate: Option<f64>,
    pub kinetic_position_diffusion: f64,
    pub scope: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecayCaseReport {
    pub profile: DecayProfile,
    pub pair_randomness: PairRandomness,
    pub walkers: usize,
    pub dimensions: usize,
    pub native_config: GasConfig,
    pub linear_control_theorem_applicable: bool,
    pub applicability_flags: Vec<String>,
    pub reference_linear_constants: LinearDecayConstants,
    pub seed_trajectories: Vec<DecaySeedTrajectory>,
    pub ensemble_trajectory: Vec<DecayEnsemblePoint>,
    pub empirical_rate: DecayRateEstimate,
    pub endpoint_decline: DecayEndpointComparison,
    pub diffusion_calibration: Option<DiffusionCalibration>,
    pub cloning_calibration: Option<CloningCalibration>,
    pub terminal_window_mean_error: Option<EnsembleValue>,
    pub total_revivals: usize,
    pub checks: Vec<BoundCheck>,
    pub statistical_checks: Vec<DecayStatisticalCheck>,
    pub scope: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecayValidationReport {
    pub config: DecayValidationConfig,
    pub cases: Vec<DecayCaseReport>,
    pub scope_notes: Vec<String>,
}

fn configured(
    profile: DecayProfile,
    dimensions: usize,
    dt: f64,
    calibration_position_diffusion: f64,
    calibration_jitter_amplitude: f64,
) -> Result<GasConfig> {
    let mut config = GasConfig::euclidean(dimensions, dt)?;
    if profile == DecayProfile::RastriginDiffusionCalibration {
        config.kinetic.position_diffusion = calibration_position_diffusion;
    }
    if profile == DecayProfile::RastriginCloningCalibration {
        config.clone_transform.jitter_amplitude = calibration_jitter_amplitude;
    }
    if profile.control() {
        config.boundary = BoundaryPolicy::Unbounded;
        config.fitness.reward_exponent = 0.;
        config.fitness.diversity_exponent = 0.;
        config.clone_transform.jitter = None;
        config.clone_transform.jitter_amplitude = 0.;
        config.clone_transform.restitution = None;
        config.kinetic.velocity_cap = None;
        let noisy = profile == DecayProfile::LinearQuadraticIndependentNoise;
        config.kinetic.position_diffusion = if noisy { 0.01 } else { 0. };
        config.kinetic.noise.geometry = NoiseGeometry::Isotropic {
            scale: FactorValues::Constant {
                values: vec![if noisy { 0.2 } else { 0. }],
            },
        };
    }
    Ok(config)
}
fn initial_population(
    walkers: usize,
    dimensions: usize,
    seed: u64,
    shift: f64,
    dead: bool,
) -> Result<Population<f64>> {
    let mut rng = RandomStream::new(seed, 0, Stream::Initialize, 0, 0);
    let mut rows = (0..walkers)
        .map(|_| {
            (0..dimensions)
                .map(|_| 0.3 * (2. * rng.uniform::<f64>() - 1.) / (dimensions as f64).sqrt())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        a.iter()
            .zip(b)
            .find_map(|(x, y)| {
                let order = x.total_cmp(y);
                (!order.is_eq()).then_some(order)
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for row in &mut rows {
        for x in row {
            *x += shift / (dimensions as f64).sqrt();
        }
    }
    if dead {
        rows[0] = vec![3.; dimensions];
    }
    let mut observation =
        ObservationBatch::positions(TensorBatch::vectors(walkers, dimensions, rows.concat())?);
    observation.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(walkers, dimensions, vec![0.; walkers * dimensions])?,
    );
    Population::new(observation)
}
fn hash_seed(seed: u64, address: u64) -> u64 {
    let mut value = seed.wrapping_add(address.wrapping_mul(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
fn slope_rate(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 3
        || points
            .iter()
            .any(|(t, x)| !t.is_finite() || !x.is_finite() || *x <= 0.)
    {
        return None;
    }
    let n = points.len() as f64;
    let tx = points.iter().map(|(t, _)| *t).sum::<f64>() / n;
    let ly = points.iter().map(|(_, v)| v.ln()).sum::<f64>() / n;
    let denominator = points.iter().map(|(t, _)| (t - tx).powi(2)).sum::<f64>();
    (denominator > 0.).then(|| {
        -points
            .iter()
            .map(|(t, v)| (t - tx) * (v.ln() - ly))
            .sum::<f64>()
            / denominator
    })
}
fn rate_estimate(
    trajectories: &[DecaySeedTrajectory],
    ensembles: &[DecayEnsemblePoint],
    control: bool,
    end: usize,
) -> DecayRateEstimate {
    let values = |omit: Option<usize>| -> Option<Vec<(f64, f64)>> {
        (0..=end)
            .map(|step| {
                let available = trajectories
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| Some(*i) != omit)
                    .map(|(_, t)| {
                        t.points.get(step).and_then(|p| {
                            p.observables.alive_wasserstein_squared.map(|alive| {
                                if control {
                                    p.observables.barycenter_error
                                } else {
                                    alive
                                }
                            })
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                let mean = available.iter().sum::<f64>() / available.len() as f64;
                let floor = if control {
                    ensembles[step].exact_barycenter_noise_floor?
                } else {
                    0.
                };
                Some((ensembles[step].time, mean - floor))
            })
            .collect()
    };
    let missing_window_seed_pairs = trajectories
        .iter()
        .filter(|trajectory| trajectory.points.get(end).is_none())
        .count();
    let empty_alive_window_seed_pairs = trajectories
        .iter()
        .filter(|trajectory| {
            trajectory.points.get(end).is_some()
                && trajectory.points[..=end]
                    .iter()
                    .any(|point| point.observables.alive_wasserstein_squared.is_none())
        })
        .count();
    let estimate = values(None).and_then(|p| slope_rate(&p));
    let jackknife = (0..trajectories.len())
        .filter_map(|i| values(Some(i)).and_then(|p| slope_rate(&p)))
        .collect::<Vec<_>>();
    let error = if jackknife.len() == trajectories.len() {
        let n = jackknife.len() as f64;
        let mean = jackknife.iter().sum::<f64>() / n;
        Some(((n - 1.) / n * jackknife.iter().map(|v| (v - mean).powi(2)).sum::<f64>()).sqrt())
    } else {
        None
    };
    DecayRateEstimate {observable:if control {"barycenter squared error minus exact finite-time pair-noise floor"}else{"normalized alive empirical physical q Wasserstein squared error"}.into(),
        requested_seed_pairs:trajectories.len(),
        complete_window_seed_pairs:trajectories.len()-missing_window_seed_pairs-empty_alive_window_seed_pairs,
        missing_window_seed_pairs,empty_alive_window_seed_pairs,
        extinct_seed_pairs:trajectories.iter().filter(|trajectory| trajectory.termination.is_some()).count(),
        fixed_start_step:0,fixed_end_step:end,estimated_squared_error_rate:estimate,seed_jackknife_standard_error:error,
        jackknife_estimates_available:jackknife.len(),applicable:estimate.is_some(),
        explanation:if estimate.is_none() {"The complete fixed window has a missing trajectory, an empty alive measure, or a nonpositive error after the prescribed floor subtraction; no subset or alternative window was selected."}
        else if control {"Least-squares slope of log ensemble transient error on the complete fixed window. The exact finite-time barycenter noise contribution is subtracted; uncertainty resamples independent seed pairs. The fixed analytic envelope rate is reported separately."}
        else {"Descriptive fixed-window slope of the normalized alive empirical physical q Wasserstein squared error V_W. Every seed pair must have two nonempty alive measures throughout the fixed window. Dead storage atoms contribute no mass. Stochastic noise, selection, revival and capping remain in the dynamics. The linear-control contraction theorem is inapplicable and no stochastic floor is subtracted."}.into()}
}

/// Every point follows a completed native engine step, including revival,
/// collision, kinetics and terminal status. Shared streams are assigned to
/// freshly canonicalized intrinsic representatives before every paired step;
/// no fixed physical labels or permanent row-address coupling is introduced.
pub async fn validate_decay(config: &DecayValidationConfig) -> Result<DecayValidationReport> {
    config.validate()?;
    let mut cases = Vec::new();
    for &profile in &config.profiles {
        for &d in &config.dimensions {
            for &n in &config.walkers {
                let native = configured(
                    profile,
                    d,
                    config.dt,
                    config.calibration_position_diffusion,
                    config.calibration_jitter_amplitude,
                )?;
                let noisy = profile == DecayProfile::LinearQuadraticIndependentNoise;
                let constants = linear_decay_constants(
                    config.dt,
                    1.,
                    profile.curvature(),
                    if noisy { 0.2 } else { 0. },
                    if noisy { 0.01 } else { 0. },
                )?;
                let p = constants.lyapunov_matrix;
                let control = profile.control();
                let shared = config.pair_randomness == PairRandomness::Shared;
                let difference_noise_factor = if shared { 0. } else { 2. };
                let zero_difference_noise = !noisy || shared;
                let mut trajectories = Vec::new();
                let mut total_revivals = 0;
                let mut maximum_ratio: f64 = 0.;
                let mut accepted_control_clones = 0;
                let address =
                    1_000_000 + profile.random_address() * 10_000 + d as u64 * 100 + n as u64;
                let mut engine_addresses = config
                    .seeds
                    .iter()
                    .flat_map(|&seed| {
                        let first = hash_seed(seed, address * 2);
                        [
                            first,
                            if shared {
                                first
                            } else {
                                hash_seed(seed, address * 2 + 1)
                            },
                        ]
                    })
                    .collect::<Vec<_>>();
                engine_addresses.sort_unstable();
                engine_addresses.dedup();
                if engine_addresses.len() != (if shared { 1 } else { 2 }) * config.seeds.len() {
                    return Err(configuration(
                        "decay seed addresses collide within an independent-pair ensemble",
                    ));
                }
                for &seed in &config.seeds {
                    let engine_seeds = [
                        hash_seed(seed, address * 2),
                        if shared {
                            hash_seed(seed, address * 2)
                        } else {
                            hash_seed(seed, address * 2 + 1)
                        },
                    ];
                    let mut left_config = native.clone();
                    left_config.seed = engine_seeds[0];
                    let mut right_config = native.clone();
                    right_config.seed = engine_seeds[1];
                    let model = BenchmarkModel {
                        benchmark: profile.landscape(),
                        field: "positions".into(),
                        direction: native.fitness.direction,
                    };
                    let dead = !control && config.canonical_initial_dead_atom;
                    let mut left =
                        GasBuilder::new(initial_population(n, d, seed, -0.5, dead)?, model.clone())
                            .gradient(model.clone())
                            .config(left_config)
                            .build()
                            .await?;
                    let mut right =
                        GasBuilder::new(initial_population(n, d, seed, 0.5, dead)?, model.clone())
                            .gradient(model)
                            .config(right_config)
                            .build()
                            .await?;
                    if shared {
                        canonicalize_decay_engine(&mut left)?;
                        canonicalize_decay_engine(&mut right)?;
                    }
                    let first = decay_observables(left.population(), right.population(), p, 1.)?;
                    let initial_error = first.full_marked_error;
                    let mut points = vec![DecayTimePoint {
                        step: 0,
                        time: 0.,
                        observables: first,
                        revived: [0; 2],
                        cloned: [0; 2],
                    }];
                    let mut termination = None;
                    for step in 1..=config.steps {
                        if shared {
                            canonicalize_decay_engine(&mut left)?;
                            canonicalize_decay_engine(&mut right)?;
                        }
                        let l = left.step().await;
                        let r = right.step().await;
                        let (l, r) = match (l, r) {
                            (Ok(l), Ok(r)) => (l, r),
                            (Err(GasError::Extinction), _) | (_, Err(GasError::Extinction)) => {
                                termination = Some(format!(
                                    "extinction prevented a paired complete update at step {step}; later ensemble points explicitly count available completed pairs"
                                ));
                                break;
                            }
                            (Err(e), _) | (_, Err(e)) => return Err(e),
                        };
                        total_revivals += l.revivals + r.revivals;
                        if control {
                            accepted_control_clones += l.clones + r.clones;
                        }
                        let observation =
                            decay_observables(left.population(), right.population(), p, 1.)?;
                        if control && zero_difference_noise {
                            maximum_ratio = maximum_ratio.max(
                                observation.full_marked_error
                                    / (initial_error
                                        * constants.contraction_factor.powi(step as i32)),
                            );
                        }
                        points.push(DecayTimePoint {
                            step,
                            time: step as f64 * config.dt,
                            observables: observation,
                            revived: [l.revivals, r.revivals],
                            cloned: [l.clones, r.clones],
                        });
                    }
                    trajectories.push(DecaySeedTrajectory {
                        pair_seed: seed,
                        native_engine_seeds: engine_seeds,
                        points,
                        termination,
                    });
                }
                let mut ensemble_trajectory = Vec::new();
                let mut statistical_checks = Vec::new();
                let mut covariance = [[0.; 2]; 2];
                let mut displacement = [1., 0.];
                for step in 0..=config.steps {
                    let points = trajectories
                        .iter()
                        .filter_map(|t| t.points.get(step))
                        .collect::<Vec<_>>();
                    let value = |extract: fn(&DecayObservables) -> f64| {
                        ensemble(
                            &points
                                .iter()
                                .map(|p| extract(&p.observables))
                                .collect::<Vec<_>>(),
                        )
                    };
                    let full = value(|o| o.full_marked_error);
                    let bary = value(|o| o.barycenter_error);
                    let floor = difference_noise_factor * d as f64 * trace_product(p, covariance)
                        / n as f64;
                    let transient = quadratic(p, displacement);
                    let transport_upper = transient + floor * n as f64;
                    let geometric = constants.contraction_factor.powi(step as i32);
                    let increment = difference_noise_factor
                        * d as f64
                        * trace_product(p, constants.innovation_covariance);
                    let geometric_envelope = geometric
                        + increment * (1. - geometric) / (1. - constants.contraction_factor);
                    if control && noisy && step > 0 {
                        let actual = bary.as_ref().unwrap();
                        let mean_covariance = covariance
                            .map(|row| row.map(|x| difference_noise_factor * x / n as f64));
                        let pc = multiply(p, mean_covariance);
                        let pm = vector(p, displacement);
                        let cm = vector(mean_covariance, pm);
                        let variance = 2. * d as f64 * trace_product(pc, pc)
                            + 4. * (pm[0] * cm[0] + pm[1] * cm[1]);
                        let standard_error = (variance / config.seeds.len() as f64).sqrt();
                        let residual = actual.mean - (transient + floor);
                        let status = if residual.abs()
                            <= 6. * standard_error + 1e-10 * (1. + transient + floor)
                        {
                            KineticCheckStatus::NotRejected
                        } else {
                            KineticCheckStatus::Violated
                        };
                        statistical_checks.push(DecayStatisticalCheck {id:format!("{}_barycenter_exact_mean_step_{step}",if shared {"shared"}else{"independent"}),source_labels:vec![SOURCE.into()],scope:if shared {"Exact linear-control difference prediction under common Gaussian innovations on intrinsic canonical representatives. Uniform translation preserves the common atom order; difference covariance and noise floor are zero. The diagnostic allows floating-point tolerance only."}else{"Exact Gaussian linear-control barycenter prediction; covariance is 2 Sigma_t/N for independently driven paired engines. Analytic quadratic-form variance gives seed-ensemble uncertainty, with a predeclared six-standard-error diagnostic."}.into(),observed:actual.mean,prediction:transient+floor,standard_error,z_score:(standard_error>0.).then_some(residual/standard_error),status});
                        let actual_full = full.as_ref().unwrap();
                        let residual = actual_full.mean - transport_upper;
                        let status = if residual
                            <= 6. * actual_full.standard_error + 1e-10 * (1. + transport_upper)
                        {
                            KineticCheckStatus::NotRejected
                        } else {
                            KineticCheckStatus::Violated
                        };
                        statistical_checks.push(DecayStatisticalCheck {id:format!("{}_transport_n_uniform_upper_step_{step}",if shared {"shared"}else{"independent"}),source_labels:vec![SOURCE.into()],scope:if shared {"Common innovations cancel in the exact quadratic control, after intrinsic canonical representative refresh at every step. Optimal empirical transport is bounded by the translated-cloud coupling q(A^t delta), with no difference-noise floor."}else{"An admissible coupling matches intrinsically paired initial atoms but drives the two engines independently. Optimal full empirical transport is at most this coupling cost; its expected bound is q(A^t delta)+2d tr(P Sigma_t), independent of N. Uncertainty uses independent complete swarm pairs, not interacting rows."}.into(),observed:actual_full.mean,prediction:transport_upper,standard_error:actual_full.standard_error,z_score:(actual_full.standard_error>0.).then_some(residual/actual_full.standard_error),status});
                    }
                    ensemble_trajectory.push(DecayEnsemblePoint {
                        step,
                        time: step as f64 * config.dt,
                        completed_pairs: points.len(),
                        full_marked_error: full,
                        barycenter_error: bary,
                        structural_error: value(|o| o.structural_error),
                        alive_error: ensemble(
                            &points
                                .iter()
                                .filter_map(|p| p.observables.alive_wasserstein_squared)
                                .collect::<Vec<_>>(),
                        ),
                        squared_position_to_minimizer: value(|o| o.squared_position_to_minimizer),
                        exact_barycenter_expectation: control.then_some(transient + floor),
                        exact_barycenter_noise_floor: control.then_some(floor),
                        n_independent_transport_expectation_upper: control
                            .then_some(transport_upper),
                        n_independent_geometric_noise_envelope: control
                            .then_some(geometric_envelope),
                        noiseless_geometric_envelope: (control && zero_difference_noise)
                            .then_some(geometric),
                    });
                    covariance = add(
                        multiply(
                            multiply(constants.map, covariance),
                            transpose(constants.map),
                        ),
                        constants.innovation_covariance,
                    );
                    displacement = vector(constants.map, displacement);
                }
                let mut checks = vec![BoundCheck::upper(
                    "reference_discrete_lyapunov_identity",
                    &[SOURCE],
                    &constants.scope,
                    constants.lyapunov_identity_residual,
                    1e-11,
                )];
                if control {
                    checks.push(BoundCheck::upper(
                        "linear_control_zero_accepted_clones",
                        &[SOURCE],
                        &constants.scope,
                        accepted_control_clones as f64,
                        0.,
                    ));
                }
                if control && zero_difference_noise {
                    checks.push(BoundCheck::upper(
                        "complete_noiseless_n_uniform_geometric_transport_envelope",
                        &[SOURCE],
                        &constants.scope,
                        maximum_ratio,
                        1.,
                    ));
                }
                let diffusion_calibration = if profile
                    == DecayProfile::RastriginDiffusionCalibration
                {
                    Some(DiffusionCalibration {
                        nominal_well_spacing: 1.,
                        reference_step_standard_deviation_fraction: 0.25,
                        reference_position_diffusion: diffusion_from_well_spacing(config.dt, 1., 0.25)?,
                        selected_position_diffusion: config.calibration_position_diffusion,
                        actual_step_gaussian_standard_deviation: config.calibration_position_diffusion * config.dt.sqrt(),
                        scope: "Only the native position-diffusion parameter changes. The Rastrigin cosine period supplies the nominal spacing ell=1; the geometry-based reference is sigma_pos=ell/(4 sqrt(h)), giving per-coordinate Gaussian standard deviation ell/4 per completed step. Baseline and calibration retain identical initial atoms, seed addresses, fit window, selection, cap, jitter and revival rules. The selected sweep value is explicit, and no fitted contraction rate or canonical theorem applicability is assigned.".into(),
                    })
                } else {
                    None
                };
                let cloning_calibration = if profile == DecayProfile::RastriginCloningCalibration {
                    Some(CloningCalibration {
                        baseline_jitter_amplitude: 0.1,
                        selected_jitter_amplitude: config.calibration_jitter_amplitude,
                        gaussian_jitter_variance_per_accepted_coordinate: {
                            let variance = config.calibration_jitter_amplitude.powi(2);
                            (config.calibration_jitter_amplitude == 0. || (variance.is_finite() && variance > 0.)).then_some(variance)
                        },
                        log_gaussian_jitter_variance_per_accepted_coordinate: (config.calibration_jitter_amplitude > 0.).then(|| 2. * config.calibration_jitter_amplitude.ln()),
                        kinetic_position_diffusion: native.kinetic.position_diffusion,
                        scope: "Only the existing native accepted-copy/revival Gaussian jitter amplitude changes. Independent jitter is applied to accepted destinations, including mandatory revivals, and has per-coordinate variance sigma_clone squared before collisions and kinetics. Kinetic position diffusion, selection, cap, collision, gate and revival rules retain their canonical values. Baseline and calibration use the same initial geometry, intrinsic profile RNG address and fixed fit window. A clone-jitter sweep does not assign a canonical contraction theorem or remove stochastic error floors.".into(),
                    })
                } else {
                    None
                };
                let mut applicability_flags = if control {
                    vec![
                        "exact quadratic force".into(),
                        "unbounded boundary, all slots alive".into(),
                        "constant fitness, no accepted clone edges".into(),
                        "no jitter, no collision and no cap".into(),
                        "P and rho fixed before observing errors, independent of N".into(),
                    ]
                } else {
                    vec!["selection, jitter, collision, cap and absorbing boundary retained".into(),"completed revival included in every recorded update".into(),"linear control contraction theorem inapplicable".into(),"no stochastic floor is subtracted from canonical or calibrated empirical error".into(),"missing completed trajectories remain counted, no survival claim inferred".into()]
                };
                applicability_flags.push(if shared {"shared addressed Gaussian, clone-jitter, gate, rotation and edge-Gumbel draws after intrinsic canonical representative refresh; each marginal uses its native conditional law"}else{"independent innovations between paired engines; seed pairs are independent ensemble units"}.into());
                if profile == DecayProfile::RastriginDiffusionCalibration {
                    applicability_flags.push("explicit position-diffusion calibration; other canonical parameters retained".into());
                }
                if profile == DecayProfile::RastriginCloningCalibration {
                    applicability_flags.push("explicit accepted-copy/revival jitter calibration; kinetic diffusion and other canonical parameters retained".into());
                }
                cases.push(DecayCaseReport {
                    profile,
                    pair_randomness: config.pair_randomness,
                    walkers: n,
                    dimensions: d,
                    native_config: native,
                    linear_control_theorem_applicable: control,
                    applicability_flags,
                    reference_linear_constants: constants,
                    seed_trajectories: trajectories,
                    ensemble_trajectory,
                    empirical_rate: DecayRateEstimate::default(),
                    endpoint_decline: DecayEndpointComparison::default(),
                    diffusion_calibration,
                    cloning_calibration,
                    terminal_window_mean_error: None,
                    total_revivals,
                    checks,
                    statistical_checks,
                    scope: String::new(),
                });
            }
        }
    }
    let mut report = DecayValidationReport {config:config.clone(),cases,scope_notes:vec![
        "The reference control derives an exact N-independent squared-error envelope from AᵀPA=P-rI. Its rate is fixed before simulation and is not a fitted Keystone rate or a theorem about the canonical selected/capped/killed configuration.".into(),
        if config.pair_randomness == PairRandomness::Shared {"Before every paired update, each swarm is gathered into an intrinsic canonical representative through checkpoint/restore, preserving native step/version/provenance/counters. Common addressed Gaussian, gate, jitter, rotation and edge-Gumbel variables define a law-valid coupling of the two conditional kernels; donor probabilities retain their native values. This coupling is not asserted maximal. Identical empirical states coalesce pathwise and no permanent physical labels are introduced."}else{"The two engines use independent seed addresses; Gaussian noises are independent across rows. Every measured distance minimizes empirical transport and is invariant under independent storage permutations."}.into(),
        "Independent-noise linear-control barycenter error has an exact finite-time floor 2d tr(P Sigma_t)/N. Shared Gaussian innovations cancel in the translated quadratic control and give zero difference-noise floor. Full optimal-transport error has its corresponding N-independent expected upper envelope. Terminal-window raw error is an empirical quantity, not a certified stationary floor.".into(),
        "Log-linear rate fits use the complete predeclared window. Noise-floor subtraction is permitted only for the exact linear barycenter model; if the fixed window becomes nonpositive or incomplete the fit is unavailable. Canonical fits retain their stochastic floor and may be negative.".into(),
    ]};
    refresh_decay_summary(&mut report)?;
    Ok(report)
}

/// Recompute decay conclusions from retained native trajectories without running
/// or mutating an engine. Full marked/storage observations, native configuration,
/// seeds, analytic constants and validation checks are preserved.
pub fn refresh_decay_summary(report: &mut DecayValidationReport) -> Result<()> {
    report.config.validate()?;
    let config = &report.config;
    for case in &mut report.cases {
        let trajectories = &case.seed_trajectories;
        let ensemble_trajectory = &case.ensemble_trajectory;
        if trajectories.len() != config.seeds.len()
            || ensemble_trajectory.len() != config.steps + 1
            || trajectories.iter().any(|trajectory| {
                trajectory.points.is_empty()
                    || trajectory
                        .points
                        .iter()
                        .enumerate()
                        .any(|(index, point)| point.step != index)
            })
        {
            return Err(configuration(
                "retained decay report must include each requested seed pair, step-zero observations and consecutively indexed complete-update points",
            ));
        }
        let control = case.profile.control();
        let estimate = rate_estimate(
            trajectories,
            ensemble_trajectory,
            control,
            config.fit_end_step,
        );
        let endpoints = trajectories
            .iter()
            .filter_map(|trajectory| {
                trajectory.points.get(config.steps).and_then(|last| {
                    trajectory.points[0]
                        .observables
                        .alive_wasserstein_squared
                        .zip(last.observables.alive_wasserstein_squared)
                })
            })
            .collect::<Vec<_>>();
        let initial = ensemble(
            &endpoints
                .iter()
                .map(|(first, _)| *first)
                .collect::<Vec<_>>(),
        );
        let final_error = ensemble(&endpoints.iter().map(|(_, last)| *last).collect::<Vec<_>>());
        let endpoint_decline = DecayEndpointComparison {
            observable: "normalized alive empirical physical q Wasserstein squared error V_W".into(),
            requested_seed_pairs: trajectories.len(),
            complete_seed_pairs: endpoints.len(),
            missing_endpoint_seed_pairs: trajectories.iter().filter(|trajectory|trajectory.points.get(config.steps).is_none()).count(),
            empty_alive_endpoint_seed_pairs: trajectories.iter().filter(|trajectory|trajectory.points.get(config.steps).is_some_and(|last|trajectory.points[0].observables.alive_wasserstein_squared.is_none() || last.observables.alive_wasserstein_squared.is_none())).count(),
            extinct_seed_pairs: trajectories.iter().filter(|trajectory|trajectory.termination.is_some()).count(),
            mean_initial_error: initial.as_ref().map(|v|v.mean),
            mean_final_error: final_error.as_ref().map(|v|v.mean),
            terminal_to_initial_ratio: initial.as_ref().zip(final_error.as_ref()).and_then(|(first,last)|(first.mean>0.).then_some(last.mean/first.mean)),
            empirical_mean_decreased: initial.as_ref().zip(final_error.as_ref()).map(|(first,last)|last.mean<first.mean),
            paired_mean_change: ensemble(&endpoints.iter().map(|(first,last)|last-first).collect::<Vec<_>>()),
            scope: "Paired endpoint comparison uses normalized alive empirical physical q Wasserstein squared error V_W and the same complete seed pairs with nonempty alive measures at both endpoints. Dead storage atoms contribute no mass; unequal alive supports retain their own probability normalization. Missing endpoints, empty alive endpoint measures and extinction are counted explicitly. This completed-pair description does not assert an unconditional or survival-conditioned theorem. A negative mean change is an empirical observation, independent of fitted-rate availability.".into(),
        };
        let terminal_start = config.steps * 3 / 4;
        let terminal_values = trajectories
            .iter()
            .filter_map(|trajectory| {
                (terminal_start..=config.steps)
                    .map(|step| {
                        trajectory
                            .points
                            .get(step)
                            .and_then(|p| p.observables.alive_wasserstein_squared)
                    })
                    .collect::<Option<Vec<_>>>()
                    .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            })
            .collect::<Vec<_>>();

        case.empirical_rate = estimate;
        case.endpoint_decline = endpoint_decline;
        case.terminal_window_mean_error = ensemble(&terminal_values);
        case.scope = "Full error is optimal N-normalized marked transport with fixed SPD physical q and unit status penalty. The primary canonical rate, endpoint comparison and terminal-window mean use normalized alive empirical physical q Wasserstein squared error V_W, including unequal alive counts and excluding dead storage atoms. Only complete paired windows with nonempty alive measures contribute; the full marked error remains a separate storage/status diagnostic. Barycenter and structural components use physical q. Distance to the known minimizer is a separate optimization observable. Canonical descriptive trajectories have no asserted linear-control rate. Extinction truncates paired complete trajectories explicitly.".into();
    }
    Ok(())
}
