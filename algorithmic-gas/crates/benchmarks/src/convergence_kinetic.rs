//! Conditional experiments for the kinetic statements in convergence chapters 1--2.
//!
//! Each row repeats the *same fixed input*, with independently addressed native
//! Gaussian innovations. No cloning, population feedback or survival conditioning
//! is applied. Consequently row standard errors are appropriate here; they would
//! not be appropriate for a recorded interacting swarm. All output coordinates,
//! including terminally dead rows, are used in the unmarked-kernel estimates.
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasError, ObservationBatch, Population, Precision, Result,
    TensorBatch,
    boundary::{BoundaryPolicy, BoxDomain},
    domain::{GradientProvider, NumericalDomain, OperatorFuture},
    kinetic::{KineticBoundarySchedule, KineticContext, KineticKind, KineticOperator},
    noise::{FactorValues, Noise, NoiseGeometry},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};

const CHAPTER_1: &str = "docs/source/2_fractal_gas/convergence_program/01_fragile_gas_framework.md";
const CHAPTER_2: &str = "docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md";

/// U(x) = sum_j [k_j (x_j-center_j)^2/2 + a(1-cos(w(x_j-center_j)))].
/// Positive k, nonnegative a and finite w provide global quadratic confinement
/// even when a*w² > min(k) and the Hessian has negative regions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfiningLandscape {
    pub name: String,
    pub curvature: Vec<f64>,
    pub center: Vec<f64>,
    pub ripple_amplitude: f64,
    pub ripple_frequency: f64,
}

impl ConfiningLandscape {
    pub fn quadratic(dimensions: usize) -> Self {
        Self {
            name: "unit_quadratic".into(),
            curvature: vec![1.; dimensions],
            center: vec![0.; dimensions],
            ripple_amplitude: 0.,
            ripple_frequency: 1.,
        }
    }

    pub fn validate(&self, dimensions: usize) -> Result<()> {
        require(
            dimensions > 0
                && self.curvature.len() == dimensions
                && self.center.len() == dimensions
                && self.curvature.iter().all(|x| x.is_finite() && *x > 0.)
                && self.center.iter().all(|x| x.is_finite())
                && self.ripple_amplitude.is_finite()
                && self.ripple_amplitude >= 0.
                && self.ripple_frequency.is_finite()
                && self.ripple_frequency >= 0.,
            "landscape needs positive finite d-vector curvature, finite center and nonnegative finite ripple parameters",
        )?;
        require(
            self.ripple_amplitude * self.ripple_frequency.powi(2) < f64::MAX,
            "landscape Hessian bound overflow",
        )
    }

    pub fn potential(&self, x: &[f64]) -> Result<f64> {
        self.validate(x.len())?;
        let value = x
            .iter()
            .zip(&self.center)
            .zip(&self.curvature)
            .map(|((&x, &center), &curvature)| {
                let z = x - center;
                0.5 * curvature * z * z
                    + self.ripple_amplitude * (1. - (self.ripple_frequency * z).cos())
            })
            .sum::<f64>();
        require(value.is_finite(), "nonfinite landscape potential")?;
        Ok(value)
    }

    pub fn analytic_gradient(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.validate(x.len())?;
        let gradient = x
            .iter()
            .zip(&self.center)
            .zip(&self.curvature)
            .map(|((&x, &center), &curvature)| {
                let z = x - center;
                curvature * z
                    + self.ripple_amplitude
                        * self.ripple_frequency
                        * (self.ripple_frequency * z).sin()
            })
            .collect::<Vec<_>>();
        require(
            gradient.iter().all(|x| x.is_finite()),
            "nonfinite analytic landscape gradient",
        )?;
        Ok(gradient)
    }

    pub fn assumptions(&self) -> Result<LandscapeAssumptions> {
        let d = self.curvature.len();
        self.validate(d)?;
        let min_curvature = self.curvature.iter().copied().fold(f64::INFINITY, f64::min);
        let max_curvature = self.curvature.iter().copied().fold(0., f64::max);
        let ripple_hessian = self.ripple_amplitude * self.ripple_frequency.powi(2);
        let force_at_zero = norm_sq(&self.analytic_gradient(&vec![0.; d])?).sqrt();
        let result = LandscapeAssumptions {
            force_lipschitz: max_curvature + ripple_hessian,
            force_at_zero,
            hessian_lower_bound: min_curvature - ripple_hessian,
            quadratic_potential_lower_coefficient: min_curvature / 2.,
            centered_dissipativity_coefficient: min_curvature / 2.,
            centered_dissipativity_offset: d as f64
                * (self.ripple_amplitude * self.ripple_frequency).powi(2)
                / (2. * min_curvature),
            continuously_differentiable_force: true,
            hypothesis_explanation: "Analytic global bounds: L_F=max(k)+a*w²; B_F=|grad U(0)|; U>=min(k)|x-center|²/2; (x-center).grad U >= min(k)|x-center|²/2-d(a*w)²/(2min(k)). The latter two follow from 1-cos>=0 and Young's inequality and do not assert a Foster drift for the complete swarm.".into(),
        };
        require(
            [
                result.force_lipschitz,
                result.force_at_zero,
                result.hessian_lower_bound,
                result.quadratic_potential_lower_coefficient,
                result.centered_dissipativity_coefficient,
                result.centered_dissipativity_offset,
            ]
            .iter()
            .all(|v| v.is_finite()),
            "nonfinite analytic landscape bounds",
        )?;
        Ok(result)
    }
}

impl GradientProvider<f64> for ConfiningLandscape {
    fn id(&self) -> String {
        format!("convergence/analytic-quadratic-cosine/{}/v1", self.name)
    }

    fn gradient<'a>(
        &'a self,
        population: &'a Population<f64>,
        _cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            let x = population.observations.field("positions")?;
            let mut values = Vec::with_capacity(x.values().len());
            for i in 0..population.len() {
                if population.validity[i].eligible(false) {
                    values.extend(self.analytic_gradient(x.row(i)?)?);
                } else {
                    values.extend(std::iter::repeat_n(0., x.width()));
                }
            }
            TensorBatch::vectors(population.len(), x.width(), values)
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LandscapeAssumptions {
    pub force_lipschitz: f64,
    pub force_at_zero: f64,
    pub hessian_lower_bound: f64,
    pub quadratic_potential_lower_coefficient: f64,
    pub centered_dissipativity_coefficient: f64,
    pub centered_dissipativity_offset: f64,
    pub continuously_differentiable_force: bool,
    pub hypothesis_explanation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KineticInput {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KineticValidationConfig {
    pub name: String,
    pub landscape: ConfiningLandscape,
    pub dimensions: usize,
    pub dt: f64,
    pub friction: f64,
    pub velocity_diffusion: f64,
    pub position_diffusion: f64,
    pub velocity_cap: f64,
    pub metric_weight: f64,
    pub samples: usize,
    pub seed: u64,
    pub inputs: Vec<KineticInput>,
    /// Terminal absorbing box [-r,r]^d; None inspects the unmarked kernel.
    pub box_half_width: Option<f64>,
    /// Displacement for a synchronous coupling of two kinetic inputs.
    pub coupling_position_shift: f64,
    pub coupling_velocity_shift: f64,
    /// Comparison-feature radius; this does not truncate physical positions.
    pub position_projection_radius: f64,
    /// Optional reward -U(x)-lambda_vel*|v|², for compact reward constants.
    pub reward_velocity_penalty: f64,
    /// Declared richness reference: uniform position in B(0,r), velocity zero.
    pub reference_ball_radius: f64,
    pub nondeception_segment_length: f64,
}

impl Default for KineticValidationConfig {
    fn default() -> Self {
        Self {
            name: "quadratic_canonical".into(),
            landscape: ConfiningLandscape::quadratic(2),
            dimensions: 2,
            dt: 0.04,
            friction: 1.,
            velocity_diffusion: 1.,
            position_diffusion: 0.1,
            velocity_cap: 2.,
            metric_weight: 1.,
            samples: 4096,
            seed: 510_002,
            inputs: vec![
                KineticInput {
                    positions: vec![0., 0.],
                    velocities: vec![0., 0.],
                },
                KineticInput {
                    positions: vec![0.8, -0.6],
                    velocities: vec![0.4, -0.3],
                },
                KineticInput {
                    positions: vec![1.99, 0.2],
                    velocities: vec![0.3, -0.1],
                },
            ],
            box_half_width: Some(2.),
            coupling_position_shift: 0.02,
            coupling_velocity_shift: 0.01,
            position_projection_radius: 2.,
            reward_velocity_penalty: 0.,
            reference_ball_radius: 1.,
            nondeception_segment_length: 0.5,
        }
    }
}

impl KineticValidationConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            self.dimensions > 0 && self.dimensions <= 256,
            "kinetic validation dimension must be 1..=256",
        )?;
        require(
            self.samples >= 32
                && self.samples <= 1_000_000
                && self
                    .samples
                    .checked_mul(self.dimensions)
                    .is_some_and(|n| n <= 1_000_000),
            "kinetic validation needs 32..=1000000 samples with samples*dimensions<=1000000",
        )?;
        require(
            !self.inputs.is_empty(),
            "kinetic validation requires frozen inputs",
        )?;
        require(
            self.dt.is_finite()
                && self.dt > 0.
                && self.friction.is_finite()
                && self.friction >= 0.
                && self.velocity_diffusion.is_finite()
                && self.velocity_diffusion >= 0.
                && self.position_diffusion.is_finite()
                && self.position_diffusion >= 0.
                && self.velocity_cap.is_finite()
                && self.velocity_cap > 0.
                && self.metric_weight.is_finite()
                && self.metric_weight > 0.
                && self.box_half_width.is_none_or(|r| r.is_finite() && r > 0.)
                && self.coupling_position_shift.is_finite()
                && self.coupling_velocity_shift.is_finite()
                && self.position_projection_radius.is_finite()
                && self.position_projection_radius > 0.
                && self.reward_velocity_penalty.is_finite()
                && self.reward_velocity_penalty >= 0.
                && self.reference_ball_radius.is_finite()
                && self.reference_ball_radius > 0.
                && self.nondeception_segment_length.is_finite()
                && self.nondeception_segment_length > 0.,
            "invalid kinetic timestep/friction/diffusion/cap/metric/boundary/coupling parameter",
        )?;
        require(
            self.box_half_width.is_none_or(|r| {
                self.reference_ball_radius <= r && self.nondeception_segment_length <= 2. * r
            }),
            "richness reference ball and test segment must lie inside the terminal box",
        )?;
        require(
            self.coupling_position_shift != 0. || self.coupling_velocity_shift != 0.,
            "synchronous coupling needs a nonzero input displacement",
        )?;
        let coupling_distance_squared = self.dimensions as f64
            * (self.coupling_position_shift.powi(2)
                + self.metric_weight * self.coupling_velocity_shift.powi(2));
        require(
            coupling_distance_squared.is_finite() && coupling_distance_squared > 0.,
            "synchronous coupling distance overflow or underflow",
        )?;
        self.landscape.validate(self.dimensions)?;
        for input in &self.inputs {
            require(
                input.positions.len() == self.dimensions
                    && input.velocities.len() == self.dimensions
                    && input
                        .positions
                        .iter()
                        .chain(&input.velocities)
                        .all(|x| x.is_finite())
                    && norm_sq(&input.positions).is_finite()
                    && norm_sq(&input.velocities).is_finite(),
                "kinetic frozen input must contain finite d-vectors with finite squared norms",
            )?;
            self.landscape.potential(&input.positions)?;
            self.landscape.analytic_gradient(&input.positions)?;
            let shifted_positions = input
                .positions
                .iter()
                .map(|x| x + self.coupling_position_shift)
                .collect::<Vec<_>>();
            let shifted_velocities = input
                .velocities
                .iter()
                .map(|x| x + self.coupling_velocity_shift)
                .collect::<Vec<_>>();
            require(
                norm_sq(&shifted_positions).is_finite() && norm_sq(&shifted_velocities).is_finite(),
                "synchronous coupling shifted input overflow",
            )?;
            self.landscape.analytic_gradient(&shifted_positions)?;
        }
        Ok(())
    }

    fn operator(&self) -> KineticOperator {
        KineticOperator {
            integrator: KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: self.dt,
                friction: self.friction,
            },
            noise: Noise {
                geometry: NoiseGeometry::Isotropic {
                    scale: FactorValues::Constant {
                        values: vec![self.velocity_diffusion],
                    },
                },
                ..Noise::default()
            },
            position_diffusion: self.position_diffusion,
            velocity_cap: Some(self.velocity_cap),
            boundary_schedule: KineticBoundarySchedule::EndOfStep,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KineticConstants {
    pub friction_factor: f64,
    /// OU velocity covariance q², before B2 and the radial cap.
    pub thermal_variance: f64,
    pub position_flow_coefficient: f64,
    pub position_variance: f64,
    pub position_flow_lipschitz: f64,
    pub death_probability_lipschitz: Option<f64>,
    pub moment_c_x: f64,
    pub moment_c_v: f64,
    pub moment_c_0: f64,
    pub phase_space_invertibility_ratio: f64,
    pub phase_space_nondegeneracy_hypotheses_hold: bool,
    pub positional_condition_number: Option<f64>,
    pub ou_decay_per_unit_time: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceMappedConstant {
    pub symbol: String,
    pub value: Option<f64>,
    pub formula: String,
    pub source: String,
    pub hypotheses: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonteCarloEstimate {
    pub mean: f64,
    pub standard_error: f64,
    pub samples: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KineticCheckStatus {
    NotRejected,
    Violated,
    NotApplicable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KineticCheck {
    pub item: String,
    pub source: String,
    pub estimate: Option<MonteCarloEstimate>,
    pub theoretical_value_or_bound: Option<f64>,
    pub status: KineticCheckStatus,
    pub criterion: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnergyDriftObservation {
    pub lyapunov_input: f64,
    pub lyapunov_output: MonteCarloEstimate,
    pub energy_increment: MonteCarloEstimate,
    pub observed_lyapunov_ratio: f64,
    pub source: String,
    pub interpretation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionalKineticExperiment {
    pub input: KineticInput,
    pub conditional_position_mean: Vec<f64>,
    pub position_drift: Vec<MonteCarloEstimate>,
    /// Covariance about the known analytic mean, in row-major order.
    pub position_covariance: Vec<MonteCarloEstimate>,
    /// O-stage velocity minus c*v_1, reconstructed from an uncapped B2 endpoint.
    pub thermostat_innovation_mean: Vec<MonteCarloEstimate>,
    pub thermostat_innovation_covariance: Vec<MonteCarloEstimate>,
    pub maximum_b2_to_final_cap_error: f64,
    pub physical_squared_increment: MonteCarloEstimate,
    pub physical_increment_upper_bound: f64,
    pub velocity_variance: Vec<MonteCarloEstimate>,
    pub full_covariance_min_eigenvalue: Option<f64>,
    pub full_covariance_max_eigenvalue: Option<f64>,
    pub measured_death_probability: MonteCarloEstimate,
    pub synchronous_death_probability_difference: MonteCarloEstimate,
    pub predicted_death_probability: Option<f64>,
    pub maximum_velocity_norm: f64,
    pub synchronous_position_lipschitz_ratio: f64,
    pub energy_drift: EnergyDriftObservation,
    pub checks: Vec<KineticCheck>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KineticValidationReport {
    pub config: KineticValidationConfig,
    pub landscape_assumptions: LandscapeAssumptions,
    pub constants: KineticConstants,
    pub source_mapped_constants: Vec<SourceMappedConstant>,
    pub geometry_and_reward_constants: GeometryRewardConstants,
    pub reference_landscape_validation: ReferenceLandscapeValidation,
    pub sampling_unit: String,
    pub experiments: Vec<ConditionalKineticExperiment>,
    pub scope_notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometryRewardConstants {
    pub algorithmic_diameter_bound: f64,
    pub compact_physical_position_radius: Option<f64>,
    pub inverse_position_projection_lipschitz: Option<f64>,
    pub inverse_velocity_projection_lipschitz: f64,
    pub reward_position_lipschitz_physical: Option<f64>,
    pub reward_lipschitz_squashed_sasaki: Option<f64>,
    pub reward_absolute_bound: Option<f64>,
    pub scope: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceLandscapeValidation {
    pub reference_law: String,
    pub reference_ball_radius: f64,
    pub predicted_quadratic_reward_variance: Option<f64>,
    pub measured_reward_variance: MonteCarloEstimate,
    pub nondeception_minimum_length: f64,
    pub nondeception_quadratic_gradient_bound: Option<f64>,
    pub measured_segment_gradient_averages: Vec<f64>,
    pub checks: Vec<KineticCheck>,
    pub scope: String,
}

pub fn geometry_reward_constants(
    config: &KineticValidationConfig,
) -> Result<GeometryRewardConstants> {
    config.validate()?;
    let d = config.dimensions;
    let radius = config.box_half_width.map(|r| r * (d as f64).sqrt());
    let inverse_x = radius.map(|r| (1. + r / config.position_projection_radius).powi(2));
    // Completed physical velocities are bounded by V_alg and the comparison
    // velocity projection uses that same radius, hence K_v <= (1+1)^2.
    let inverse_v = 4.;
    let physical_l = config.box_half_width.map(|r| {
        config
            .landscape
            .curvature
            .iter()
            .zip(&config.landscape.center)
            .map(|(k, c)| {
                (k * (r + c.abs())
                    + config.landscape.ripple_amplitude * config.landscape.ripple_frequency)
                    .powi(2)
            })
            .sum::<f64>()
            .sqrt()
    });
    let projected_l = physical_l.zip(inverse_x).map(|(l, k)| {
        l * k
            + 2. * config.reward_velocity_penalty * config.velocity_cap * inverse_v
                / config.metric_weight.sqrt()
    });
    let reward_bound = config.box_half_width.map(|r| {
        config
            .landscape
            .curvature
            .iter()
            .zip(&config.landscape.center)
            .map(|(k, c)| 0.5 * k * (r + c.abs()).powi(2) + 2. * config.landscape.ripple_amplitude)
            .sum::<f64>()
            + config.reward_velocity_penalty * config.velocity_cap.powi(2)
    });
    let result=GeometryRewardConstants {
        algorithmic_diameter_bound:2.*(config.position_projection_radius.powi(2)+config.metric_weight*config.velocity_cap.powi(2)).sqrt(),
        compact_physical_position_radius:radius,inverse_position_projection_lipschitz:inverse_x,
        inverse_velocity_projection_lipschitz:inverse_v,
        reward_position_lipschitz_physical:physical_l,
        reward_lipschitz_squashed_sasaki:projected_l,reward_absolute_bound:reward_bound,
        scope:"Comparison features use psi_Rx for positions and psi_V_alg for velocities with Sasaki weight lambda_v. Compact reward/inverse constants apply to completed alive rows x in [-r_box,r_box]^d and |v|<=V_alg. They are absent on an unbounded physical domain. Physical Lipschitz constants are multiplied by the inverse-projection constants before entering a squashed-metric theorem.".into(),
    };
    require(
        [
            Some(result.algorithmic_diameter_bound),
            radius,
            inverse_x,
            physical_l,
            projected_l,
            reward_bound,
        ]
        .iter()
        .all(|v| v.is_none_or(|v| v.is_finite())),
        "geometry/reward constants overflow",
    )?;
    Ok(result)
}

/// Exact variance of U for uniform Z in the physical ball B(0,r), for a
/// diagonal SPD quadratic centered at `landscape.center`. Velocities are zero.
pub fn quadratic_uniform_ball_variance(landscape: &ConfiningLandscape, radius: f64) -> Result<f64> {
    let d = landscape.curvature.len();
    landscape.validate(d)?;
    require(
        landscape.ripple_amplitude == 0. && radius.is_finite() && radius > 0.,
        "uniform-ball closed-form variance requires an SPD quadratic and r>0",
    )?;
    let trace = landscape.curvature.iter().sum::<f64>();
    let trace_squared = landscape.curvature.iter().map(|k| k * k).sum::<f64>();
    let weighted_center_squared = landscape
        .curvature
        .iter()
        .zip(&landscape.center)
        .map(|(k, c)| (k * c).powi(2))
        .sum::<f64>();
    let d2 = d as f64 + 2.;
    let d4 = d as f64 + 4.;
    let variance = radius.powi(2) * weighted_center_squared / d2
        + radius.powi(4) / (2. * d2 * d4) * (trace_squared - trace * trace / d2);
    require(
        variance.is_finite() && variance > 0.,
        "uniform-ball quadratic variance overflow or underflow",
    )?;
    Ok(variance)
}

/// For any physical segment of length at least L, integral average |grad U|²
/// is >= min(k)²*L²/12. This is global for SPD quadratics, so it also holds
/// for every segment with endpoints in the convex valid box.
pub fn quadratic_nondeception_bound(landscape: &ConfiningLandscape, length: f64) -> Result<f64> {
    landscape.validate(landscape.curvature.len())?;
    require(
        landscape.ripple_amplitude == 0. && length.is_finite() && length > 0.,
        "nondeception closed-form bound requires an SPD quadratic and L>0",
    )?;
    let m = landscape
        .curvature
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let bound = m * m * length * length / 12.;
    require(
        bound.is_finite() && bound > 0.,
        "nondeception bound overflow or underflow",
    )?;
    Ok(bound)
}

fn reference_landscape_validation(
    config: &KineticValidationConfig,
) -> Result<ReferenceLandscapeValidation> {
    let (d, n, r) = (
        config.dimensions,
        config.samples,
        config.reference_ball_radius,
    );
    let quadratic = config.landscape.ripple_amplitude == 0.;
    let prediction = if quadratic {
        Some(quadratic_uniform_ball_variance(&config.landscape, r)?)
    } else {
        None
    };
    let exact_mean = if quadratic {
        Some(
            0.5 * (config.landscape.curvature.iter().sum::<f64>() * r * r / (d as f64 + 2.)
                + config
                    .landscape
                    .curvature
                    .iter()
                    .zip(&config.landscape.center)
                    .map(|(k, c)| k * c * c)
                    .sum::<f64>()),
        )
    } else {
        None
    };
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let mut rng = RandomStream::new(
            config.seed,
            0,
            Stream::MeanFieldReference,
            i as u64,
            510_002,
        );
        let mut z = (0..d).map(|_| rng.gaussian::<f64>()).collect::<Vec<_>>();
        let radial = r * rng.uniform::<f64>().powf(1. / d as f64);
        let normalization = norm_sq(&z).sqrt();
        require(
            normalization > 0. && normalization.is_finite(),
            "uniform-ball direction sampling failed",
        )?;
        for x in &mut z {
            *x *= radial / normalization;
        }
        values.push(config.landscape.potential(&z)?);
    }
    let mean = exact_mean.unwrap_or_else(|| values.iter().sum::<f64>() / n as f64);
    let mut measured = estimate(values.iter().map(|v| (v - mean).powi(2)));
    if !quadratic {
        let correction = n as f64 / (n - 1) as f64;
        measured.mean *= correction;
        measured.standard_error *= correction;
    }
    let mut checks = Vec::new();
    if let Some(value) = prediction {
        checks.push(equality_check(
            "quadratic_uniform_ball_reward_variance".into(),
            "lem-euclidean-richness",
            measured.clone(),
            value,
            1e-12,
        ));
    }
    let bound = if quadratic {
        Some(quadratic_nondeception_bound(
            &config.landscape,
            config.nondeception_segment_length,
        )?)
    } else {
        None
    };
    let mut segment_averages = Vec::with_capacity(d);
    for axis in 0..d {
        // Numerical integral on the axis segment [-L/2,L/2] inside the box.
        // Simpson is exact for squared gradients of a quadratic potential.
        let mut sum = 0.;
        let intervals = 64;
        for j in 0..=intervals {
            let mut point = vec![0.; d];
            point[axis] = config.nondeception_segment_length * (j as f64 / intervals as f64 - 0.5);
            let value = norm_sq(&config.landscape.analytic_gradient(&point)?);
            sum += value
                * if j == 0 || j == intervals {
                    1.
                } else if j % 2 == 0 {
                    2.
                } else {
                    4.
                };
        }
        let average = sum / (3. * intervals as f64);
        segment_averages.push(average);
        if let Some(bound) = bound {
            checks.push(KineticCheck {
            item:format!("quadratic_nondeception_segment[{axis}]"),source:source("axiom-non-deceptive"),
            estimate:None,theoretical_value_or_bound:Some(bound),status:status(average+1e-12>=bound),
            criterion:format!("Squared-gradient segment average {average} >= min(k)²*L_grad²/12 for an SPD quadratic; global bound derived from midpoint/direction decomposition, numerical Simpson check on the declared segment"),
        });
        }
    }
    require(
        measured.mean.is_finite()
            && measured.standard_error.is_finite()
            && segment_averages.iter().all(|x| x.is_finite()),
        "reference landscape statistic overflow",
    )?;
    Ok(ReferenceLandscapeValidation {
        reference_law:format!("Uniform physical position in B(0,{r}) in R^{d}, with velocity fixed to zero"),
        reference_ball_radius:r,predicted_quadratic_reward_variance:prediction,measured_reward_variance:measured,
        nondeception_minimum_length:config.nondeception_segment_length,
        nondeception_quadratic_gradient_bound:bound,measured_segment_gradient_averages:segment_averages,checks,
        scope:"The richness value is the variance for this explicitly specified reference law and region, not a uniform lower bound over all translated local regions or population laws. The SPD quadratic nondeception bound is global for physical segments of length>=L_grad. Cosine-ripple landscapes retain empirical variance/segment measurements but no quadratic certificate.".into(),
    })
}

pub fn kinetic_constants(config: &KineticValidationConfig) -> Result<KineticConstants> {
    config.validate()?;
    let assumptions = config.landscape.assumptions()?;
    let h = config.dt;
    let gamma = config.friction;
    require(
        (gamma * h).is_finite() && (2. * gamma).is_finite(),
        "friction scale overflow",
    )?;
    let c = (-gamma * h).exp();
    let q_sq = config.velocity_diffusion.powi(2)
        * if gamma == 0. {
            h
        } else {
            -(-2. * gamma * h).exp_m1() / (2. * gamma)
        };
    let b = h * (1. + c) / 2.;
    let s_sq = h * h * q_sq / 4. + h * config.position_diffusion.powi(2);
    let l_flow = 1. + b * h * assumptions.force_lipschitz / 2. + b / config.metric_weight.sqrt();
    let constants = KineticConstants {
        friction_factor: c,
        thermal_variance: q_sq,
        position_flow_coefficient: b,
        position_variance: s_sq,
        position_flow_lipschitz: l_flow,
        death_probability_lipschitz: (s_sq > 0.)
            .then(|| l_flow / (2. * std::f64::consts::PI * s_sq).sqrt()),
        moment_c_x: 3. * b * b * h * h * assumptions.force_lipschitz.powi(2) / 4.,
        moment_c_v: 3. * b * b + 2. * config.metric_weight,
        moment_c_0: 3. * b * b * h * h * assumptions.force_at_zero.powi(2) / 4.
            + config.dimensions as f64 * s_sq
            + 2. * config.metric_weight * config.velocity_cap.powi(2),
        phase_space_invertibility_ratio: h * h * assumptions.force_lipschitz / 4.,
        phase_space_nondegeneracy_hypotheses_hold: q_sq > 0.
            && config.position_diffusion > 0.
            && h * h * assumptions.force_lipschitz / 4. < 1.,
        positional_condition_number: (s_sq > 0.).then_some(1.),
        ou_decay_per_unit_time: gamma,
    };
    require(
        [
            constants.friction_factor,
            constants.thermal_variance,
            constants.position_flow_coefficient,
            constants.position_variance,
            constants.position_flow_lipschitz,
            constants.moment_c_x,
            constants.moment_c_v,
            constants.moment_c_0,
            constants.phase_space_invertibility_ratio,
        ]
        .iter()
        .all(|x| x.is_finite())
            && constants
                .death_probability_lipschitz
                .is_none_or(|x| x.is_finite()),
        "kinetic constants overflow",
    )?;
    Ok(constants)
}

/// Includes admissible cases and the documented h=2 quadratic degeneracy.
/// A failed sufficient condition is reported as inapplicable, rather than
/// classified as an invalid transition or a disproof of that conditional lemma.
pub fn default_validation_cases(samples: usize, seed: u64) -> Vec<KineticValidationConfig> {
    let base = KineticValidationConfig {
        samples,
        seed,
        ..Default::default()
    };
    let mut anisotropic = base.clone();
    anisotropic.name = "anisotropic_shifted_quadratic".into();
    anisotropic.landscape.name = anisotropic.name.clone();
    anisotropic.landscape.curvature = vec![0.5, 4.];
    anisotropic.landscape.center = vec![0.3, -0.2];
    anisotropic.seed = seed.wrapping_add(1);
    let mut ripple = base.clone();
    ripple.name = "confined_nonconvex_ripples".into();
    ripple.landscape.name = ripple.name.clone();
    ripple.landscape.ripple_amplitude = 0.4;
    ripple.landscape.ripple_frequency = 3.;
    ripple.seed = seed.wrapping_add(2);
    let mut undamped = base.clone();
    undamped.name = "zero_friction_limit".into();
    undamped.friction = 0.;
    undamped.seed = seed.wrapping_add(3);
    let mut near = base.clone();
    near.name = "quadratic_near_nondegeneracy_threshold".into();
    near.dt = 1.9;
    near.seed = seed.wrapping_add(4);
    let mut obstruction = base.clone();
    obstruction.name = "quadratic_h2_velocity_covariance_obstruction".into();
    obstruction.dt = 2.;
    obstruction.seed = seed.wrapping_add(5);
    vec![base, anisotropic, ripple, undamped, near, obstruction]
}

/// Execute the production kinetic operator on conditional independent replicas.
pub async fn validate_kinetic(config: &KineticValidationConfig) -> Result<KineticValidationReport> {
    let constants = kinetic_constants(config)?;
    let assumptions = config.landscape.assumptions()?;
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    let operator = config.operator();
    let boundary = match config.box_half_width {
        Some(radius) => BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-radius; config.dimensions],
                upper: vec![radius; config.dimensions],
            },
        },
        None => BoundaryPolicy::Unbounded,
    };
    let mut experiments = Vec::with_capacity(config.inputs.len());
    for (index, input) in config.inputs.iter().enumerate() {
        let mut population = repeated_population(input, config.samples)?;
        operator.validate(&population, true)?;
        operator
            .advance(
                &mut population,
                KineticContext {
                    gradient: Some(&config.landscape),
                    domain: &NumericalDomain,
                    boundary: &boundary,
                    include_truncated: false,
                    seed: config.seed,
                    step: index as u64,
                    graph: None,
                    operators: None,
                    frozen_fitness: None,
                },
                &mut cx,
            )
            .await?;
        // Remove only the final maps to observe the unchanged native BAOAB
        // endpoint. Terminal marking cannot affect any preceding substep.
        let mut raw_operator = operator.clone();
        raw_operator.position_diffusion = 0.;
        raw_operator.velocity_cap = None;
        let mut raw_b2 = repeated_population(input, config.samples)?;
        raw_operator
            .advance(
                &mut raw_b2,
                KineticContext {
                    gradient: Some(&config.landscape),
                    domain: &NumericalDomain,
                    boundary: &BoundaryPolicy::Unbounded,
                    include_truncated: false,
                    seed: config.seed,
                    step: index as u64,
                    graph: None,
                    operators: None,
                    frozen_fitness: None,
                },
                &mut cx,
            )
            .await?;
        let shifted = KineticInput {
            positions: input
                .positions
                .iter()
                .map(|x| x + config.coupling_position_shift)
                .collect(),
            velocities: input
                .velocities
                .iter()
                .map(|v| v + config.coupling_velocity_shift)
                .collect(),
        };
        let mut coupled = repeated_population(&shifted, config.samples)?;
        operator
            .advance(
                &mut coupled,
                KineticContext {
                    gradient: Some(&config.landscape),
                    domain: &NumericalDomain,
                    boundary: &boundary,
                    include_truncated: false,
                    seed: config.seed,
                    step: index as u64,
                    graph: None,
                    operators: None,
                    frozen_fitness: None,
                },
                &mut cx,
            )
            .await?;
        experiments.push(analyze(
            config,
            &constants,
            input,
            &population,
            &coupled,
            &raw_b2,
        )?);
    }
    let geometry_and_reward_constants = geometry_reward_constants(config)?;
    let reference_landscape_validation = reference_landscape_validation(config)?;
    let mut source_mapped_constants = source_map(&constants, &assumptions);
    source_mapped_constants.extend(geometry_source_map(
        config,
        &geometry_and_reward_constants,
        &reference_landscape_validation,
    ));
    Ok(KineticValidationReport {
        config: config.clone(),
        landscape_assumptions: assumptions,
        source_mapped_constants,
        geometry_and_reward_constants,
        reference_landscape_validation,
        constants,
        sampling_unit: "Independent native kinetic innovations across replicas of each fixed post-collision input; no swarm-dependent force, cloning or conditioning on terminal survival".into(),
        experiments,
        scope_notes: vec![
            "The first three chapters give kinetic growth, continuity and covariance estimates. The thermostat damping gamma is not a complete-swarm convergence rate; kinetic contraction is studied in later chapters.".into(),
            "Six standard errors are diagnostic acceptance bands, not rigorous simultaneous confidence certificates or proofs of inequalities at unsampled inputs.".into(),
            "Confinement is established analytically by the potential envelope on all of R^d; sampling never substitutes a compact-support hypothesis.".into(),
            "Full phase-space covariance eigenvalues are computed for d<=8; the theorem supplies positivity under its hypotheses, but does not state a numerical global condition-number constant.".into(),
        ],
    })
}

fn repeated_population(input: &KineticInput, n: usize) -> Result<Population<f64>> {
    let d = input.positions.len();
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(n, d, input.positions.repeat(n))?);
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, d, input.velocities.repeat(n))?,
    );
    Population::new(observations)
}

fn analyze(
    config: &KineticValidationConfig,
    constants: &KineticConstants,
    input: &KineticInput,
    population: &Population<f64>,
    coupled: &Population<f64>,
    raw_b2: &Population<f64>,
) -> Result<ConditionalKineticExperiment> {
    let (d, n) = (config.dimensions, config.samples);
    let x = population.observations.field("positions")?;
    let v = population.observations.field("velocities")?;
    let coupled_x = coupled.observations.field("positions")?;
    let b2_x = raw_b2.observations.field("positions")?;
    let b2_v = raw_b2.observations.field("velocities")?;
    require(
        x.values()
            .iter()
            .chain(v.values())
            .chain(coupled_x.values())
            .all(|x| x.is_finite()),
        "nonfinite kinetic experiment output",
    )?;
    let gradient = config.landscape.analytic_gradient(&input.positions)?;
    let mut thermostat_innovations = Vec::with_capacity(n * d);
    let mut maximum_cap_error = 0_f64;
    for i in 0..n {
        let second_gradient = config.landscape.analytic_gradient(b2_x.row(i)?)?;
        let raw_radius = norm_sq(b2_v.row(i)?).sqrt();
        for j in 0..d {
            let v1 = input.velocities[j] - config.dt * gradient[j] / 2.;
            let v2 = b2_v.values()[i * d + j] + config.dt * second_gradient[j] / 2.;
            thermostat_innovations.push(v2 - constants.friction_factor * v1);
            let capped =
                config.velocity_cap * b2_v.values()[i * d + j] / (config.velocity_cap + raw_radius);
            maximum_cap_error = maximum_cap_error.max((capped - v.values()[i * d + j]).abs());
        }
    }
    let thermostat_innovation_mean = (0..d)
        .map(|j| estimate((0..n).map(|i| thermostat_innovations[i * d + j])))
        .collect::<Vec<_>>();
    let thermostat_innovation_covariance = (0..d * d)
        .map(|ij| {
            estimate((0..n).map(|i| {
                thermostat_innovations[i * d + ij / d] * thermostat_innovations[i * d + ij % d]
            }))
        })
        .collect::<Vec<_>>();
    let mean = (0..d)
        .map(|j| {
            input.positions[j]
                + constants.position_flow_coefficient
                    * (input.velocities[j] - config.dt * gradient[j] / 2.)
        })
        .collect::<Vec<_>>();
    let position_drift = (0..d)
        .map(|j| estimate((0..n).map(|i| x.values()[i * d + j] - input.positions[j])))
        .collect::<Vec<_>>();
    let position_covariance = (0..d * d)
        .map(|ij| {
            let (a, b) = (ij / d, ij % d);
            estimate(
                (0..n)
                    .map(|i| (x.values()[i * d + a] - mean[a]) * (x.values()[i * d + b] - mean[b])),
            )
        })
        .collect::<Vec<_>>();
    let physical_squared_increment = estimate((0..n).map(|i| {
        (0..d)
            .map(|j| {
                (x.values()[i * d + j] - input.positions[j]).powi(2)
                    + config.metric_weight * (v.values()[i * d + j] - input.velocities[j]).powi(2)
            })
            .sum()
    }));
    let bound = constants.moment_c_x * norm_sq(&input.positions)
        + constants.moment_c_v * norm_sq(&input.velocities)
        + constants.moment_c_0;
    require(bound.is_finite(), "input moment bound overflow")?;
    let velocity_means = (0..d)
        .map(|j| (0..n).map(|i| v.values()[i * d + j]).sum::<f64>() / n as f64)
        .collect::<Vec<_>>();
    let velocity_variance = (0..d)
        .map(|j| {
            let mut e =
                estimate((0..n).map(|i| (v.values()[i * d + j] - velocity_means[j]).powi(2)));
            let correction = n as f64 / (n - 1) as f64;
            e.mean *= correction;
            e.standard_error *= correction;
            e
        })
        .collect::<Vec<_>>();
    let mut maximum_velocity_norm = 0_f64;
    let mut max_coupled_sq = 0_f64;
    for i in 0..n {
        maximum_velocity_norm = maximum_velocity_norm.max(norm_sq(v.row(i)?).sqrt());
        max_coupled_sq = max_coupled_sq.max(
            (0..d)
                .map(|j| (coupled_x.values()[i * d + j] - x.values()[i * d + j]).powi(2))
                .sum(),
        );
    }
    let input_distance = (d as f64
        * (config.coupling_position_shift.powi(2)
            + config.metric_weight * config.coupling_velocity_shift.powi(2)))
    .sqrt();
    let lipschitz_ratio = max_coupled_sq.sqrt() / input_distance;
    let death = estimate(
        population
            .validity
            .iter()
            .map(|s| if s.eligible(false) { 0. } else { 1. }),
    );
    let death_difference = estimate(population.validity.iter().zip(&coupled.validity).map(
        |(left, right)| {
            let left_dead = if left.eligible(false) { 0. } else { 1. };
            let right_dead = if right.eligible(false) { 0. } else { 1. };
            right_dead - left_dead
        },
    ));
    let predicted_death = config.box_half_width.map(|r| {
        if constants.position_variance == 0. {
            if mean.iter().all(|m| *m >= -r && *m <= r) {
                0.
            } else {
                1.
            }
        } else {
            let sd = constants.position_variance.sqrt();
            1. - mean
                .iter()
                .map(|m| normal_cdf((r - m) / sd) - normal_cdf((-r - m) / sd))
                .product::<f64>()
        }
    });
    let lyapunov_before = lyapunov(&input.positions, &input.velocities, config);
    let lyapunov_output = estimate((0..n).map(|i| {
        lyapunov(
            x.row(i).expect("checked rows"),
            v.row(i).expect("checked rows"),
            config,
        )
    }));
    let initial_energy =
        config.landscape.potential(&input.positions)? + 0.5 * norm_sq(&input.velocities);
    let mut energy_changes = Vec::with_capacity(n);
    for i in 0..n {
        energy_changes.push(
            config.landscape.potential(x.row(i)?)? + 0.5 * norm_sq(v.row(i)?) - initial_energy,
        );
    }
    let mut checks = Vec::new();
    for (j, estimate) in thermostat_innovation_mean.iter().enumerate() {
        checks.push(equality_check(
            format!("thermostat_innovation_mean[{j}]"),
            "def-eg-baoab-canonical",
            estimate.clone(),
            0.,
            1e-12,
        ));
    }
    for (ij, estimate) in thermostat_innovation_covariance.iter().enumerate() {
        checks.push(equality_check(
            format!("thermostat_innovation_covariance[{},{}]", ij / d, ij % d),
            "def-eg-baoab-canonical",
            estimate.clone(),
            if ij / d == ij % d {
                constants.thermal_variance
            } else {
                0.
            },
            1e-12,
        ));
    }
    checks.push(KineticCheck {
        item:"native_b2_to_final_cap_identity".into(),source:source("def-eg-baoab-canonical"),estimate:None,
        theoretical_value_or_bound:Some(0.),status:status(maximum_cap_error<=1e-12),
        criterion:format!("Same-innovation native B2 endpoint with final diffusion/cap omitted reconstructs the production final cap; maximum absolute residual={maximum_cap_error}"),
    });
    for j in 0..d {
        checks.push(equality_check(
            format!("position_drift[{j}]"),
            "lem-euclidean-geometric-consistency",
            position_drift[j].clone(),
            mean[j] - input.positions[j],
            1e-12,
        ));
    }
    for (ij, estimate) in position_covariance.iter().enumerate() {
        checks.push(equality_check(
            format!("position_covariance[{},{}]", ij / d, ij % d),
            "lem-euclidean-geometric-consistency",
            estimate.clone(),
            if ij / d == ij % d {
                constants.position_variance
            } else {
                0.
            },
            1e-12,
        ));
    }
    checks.push(KineticCheck {
        item: "physical_squared_increment_bound".into(), source: source("lem-euclidean-perturb-moment"),
        estimate: Some(physical_squared_increment.clone()), theoretical_value_or_bound: Some(bound),
        status: status(physical_squared_increment.mean-6.*physical_squared_increment.standard_error<=bound+1e-12),
        criterion: "Measured conditional mean minus six standard errors <= C_x|x|²+C_v|v|²+C_0, under globally Lipschitz force growth and positive cap/metric weight".into(),
    });
    checks.push(KineticCheck {
        item: "synchronous_position_lipschitz".into(), source: source("lem-sasaki-kinetic-lipschitz"),
        estimate: None, theoretical_value_or_bound: Some(constants.position_flow_lipschitz),
        status: status(lipschitz_ratio <= constants.position_flow_lipschitz+1e-10),
        criterion: format!("Maximum native shared-innovation position ratio {lipschitz_ratio} <= L_flow (physical Sasaki metric)"),
    });
    checks.push(KineticCheck {
        item: "radial_velocity_cap".into(),
        source: source("def-eg-baoab-canonical"),
        estimate: None,
        theoretical_value_or_bound: Some(config.velocity_cap),
        status: status(maximum_velocity_norm < config.velocity_cap),
        criterion: format!("Maximum measured velocity norm {maximum_velocity_norm} < V_alg"),
    });
    if let Some(prediction) = predicted_death {
        // With zero empirical counts, binomial SE is zero. The finite-n additive
        // term keeps an extreme event from being rejected merely for not occurring.
        checks.push(equality_check(
            "terminal_death_probability".into(),
            "lem-euclidean-boundary-holder",
            death.clone(),
            prediction,
            6. / n as f64 + d as f64 * 1.5e-7,
        ));
    }
    if config.box_half_width.is_some()
        && let Some(l_death) = constants.death_probability_lipschitz
    {
        let death_bound = l_death * input_distance;
        checks.push(KineticCheck {
                item: "terminal_death_probability_lipschitz".into(),
                source: source("lem-euclidean-boundary-holder"),
                estimate: Some(death_difference.clone()),
                theoretical_value_or_bound: Some(death_bound),
                status: status(death_difference.mean.abs()
                    - 6. * death_difference.standard_error <= death_bound + 1e-12),
                criterion: "Absolute difference of paired native terminal-death probabilities minus six standard errors <= L_death times the physical input distance; s_h>0 and terminal Borel box".into(),
            });
    }
    let (min_eigen, max_eigen) = if d <= 8 {
        covariance_eigen_extremes(x.values(), v.values(), n, d)
    } else {
        (None, None)
    };
    let energy_increment = estimate(energy_changes);
    require(
        position_drift
            .iter()
            .chain(&position_covariance)
            .chain(&thermostat_innovation_mean)
            .chain(&thermostat_innovation_covariance)
            .chain(&velocity_variance)
            .chain([
                &physical_squared_increment,
                &lyapunov_output,
                &energy_increment,
                &death,
                &death_difference,
            ])
            .all(|e| e.mean.is_finite() && e.standard_error.is_finite())
            && min_eigen.is_none_or(|e| e.is_finite())
            && max_eigen.is_none_or(|e| e.is_finite())
            && lipschitz_ratio.is_finite()
            && lyapunov_before.is_finite(),
        "kinetic moment/statistical accumulation overflow",
    )?;
    checks.push(KineticCheck {
        item: "full_phase_space_covariance_positivity".into(), source: source("lem-euclidean-geometric-consistency"),
        estimate: None, theoretical_value_or_bound: None,
        status: if constants.phase_space_nondegeneracy_hypotheses_hold {
            min_eigen.map_or(KineticCheckStatus::NotApplicable, |minimum| status(minimum>0.))
        } else {KineticCheckStatus::NotApplicable},
        criterion: format!("Sample covariance positivity under C¹ force, q>0, sigma_x>0 and h²L_F/4<1; sufficient hypotheses hold={}; minimum measured eigenvalue={min_eigen:?}. No global numerical covariance lower bound is asserted.",constants.phase_space_nondegeneracy_hypotheses_hold),
    });
    if config.dt == 2.
        && config.landscape.ripple_amplitude == 0.
        && config.landscape.curvature.iter().all(|k| *k == 1.)
    {
        for (j, variance) in velocity_variance.iter().enumerate() {
            checks.push(equality_check(
                format!("h2_velocity_covariance_obstruction[{j}]"),
                "lem-euclidean-geometric-consistency",
                variance.clone(),
                0.,
                1e-24,
            ));
        }
    }
    Ok(ConditionalKineticExperiment {
        input: input.clone(), conditional_position_mean: mean, position_drift,
        position_covariance, thermostat_innovation_mean, thermostat_innovation_covariance,
        maximum_b2_to_final_cap_error:maximum_cap_error, physical_squared_increment,
        physical_increment_upper_bound: bound, velocity_variance,
        full_covariance_min_eigenvalue: min_eigen, full_covariance_max_eigenvalue: max_eigen,
        measured_death_probability: death, synchronous_death_probability_difference: death_difference,
        predicted_death_probability: predicted_death,
        maximum_velocity_norm, synchronous_position_lipschitz_ratio: lipschitz_ratio,
        energy_drift: EnergyDriftObservation {
            lyapunov_input: lyapunov_before,
            observed_lyapunov_ratio: lyapunov_output.mean/lyapunov_before,
            lyapunov_output,
            energy_increment,
            source: format!("{CHAPTER_1}#def-foster-lyapunov"),
            interpretation: "V=1+|x-center|²+lambda_v|v|², with empirical kinetic-only PV and energy change. A finite input grid cannot certify global PV<=aV+b1_C or a convergence rate; no a,b,C certificate is inferred.".into(),
        }, checks,
    })
}

fn source_map(c: &KineticConstants, a: &LandscapeAssumptions) -> Vec<SourceMappedConstant> {
    let growth = "Gaussian BAOAB with final independent position diffusion and smooth cap; h>0, gamma>=0, lambda_v>0, V_alg>0; force defined globally with |F(x)|<=B_F+L_F|x|";
    [
        ("L_F", Some(a.force_lipschitz), "max(k)+ripple_amplitude*ripple_frequency²", "lem-euclidean-perturb-moment", "For the declared global analytic quadratic-plus-cosine potential; it is a global force Lipschitz bound, not a sampled maximum"),
        ("B_F", Some(a.force_at_zero), "|analytic_gradient_U(0)|", "lem-euclidean-perturb-moment", "Force growth |F(x)|<=B_F+L_F|x| follows from global force Lipschitz continuity"),
        ("c", Some(c.friction_factor), "exp(-gamma*h)", "def-eg-baoab-canonical", growth),
        ("q²", Some(c.thermal_variance), "sigma_v²*(1-exp(-2*gamma*h))/(2*gamma), with sigma_v²*h at gamma=0", "def-eg-baoab-canonical", growth),
        ("b", Some(c.position_flow_coefficient), "h*(1+c)/2", "lem-sasaki-kinetic-lipschitz", growth),
        ("s_h²", Some(c.position_variance), "h²*q²/4+h*sigma_x²", "lem-sasaki-kinetic-lipschitz", growth),
        ("L_flow", Some(c.position_flow_lipschitz), "1+b*h*L_F/2+b/sqrt(lambda_v)", "lem-sasaki-kinetic-lipschitz", growth),
        ("L_death", c.death_probability_lipschitz, "L_flow/(sqrt(2*pi)*s_h)", "lem-euclidean-boundary-holder", "Previous hypotheses and s_h>0; any Borel terminal domain"),
        ("C_x", Some(c.moment_c_x), "3*b²*h²*L_F²/4", "lem-euclidean-perturb-moment", growth),
        ("C_v", Some(c.moment_c_v), "3*b²+2*lambda_v", "lem-euclidean-perturb-moment", growth),
        ("C_0", Some(c.moment_c_0), "3*b²*h²*B_F²/4+d*s_h²+2*lambda_v*V_alg²", "lem-euclidean-perturb-moment", growth),
        ("h²L_F/4", Some(c.phase_space_invertibility_ratio), "h²*L_F/4 < 1", "lem-euclidean-geometric-consistency", "C¹ force, q>0, sigma_x>0 and ratio<1 imply positive definite full covariance locally; no explicit numerical global lower eigenvalue constant"),
        ("kappa_position", c.positional_condition_number, "1", "lem-euclidean-geometric-consistency", "Positive s_h²; conditional positional covariance only, not full phase space"),
        ("gamma_OU", Some(c.ou_decay_per_unit_time), "gamma in c=exp(-gamma*h)", "def-eg-baoab-canonical", "Thermostat substep damping only; this is not a full-swarm convergence rate"),
    ].into_iter().map(|(symbol,value,formula,label,hypotheses)| SourceMappedConstant {
        symbol: symbol.into(), value, formula: formula.into(), source: source(label), hypotheses: hypotheses.into(),
    }).collect()
}

fn geometry_source_map(
    config: &KineticValidationConfig,
    g: &GeometryRewardConstants,
    r: &ReferenceLandscapeValidation,
) -> Vec<SourceMappedConstant> {
    let compact = "Completed alive physical inputs in the declared box and |v|<=V_alg; inverse projection acts on the image of this compact set only";
    [
        ("D_algorithmic",Some(g.algorithmic_diameter_bound),"2*sqrt(R_x²+lambda_v*V_alg²)","lem-squashing-properties-generic","Squashed comparison coordinates in the two open balls; this is a supremal diameter bound, not physical confinement"),
        ("K_inverse_x",g.inverse_position_projection_lipschitz,"(1+B_x/R_x)², B_x=r_box*sqrt(d)","lem-euclidean-reward-regularity",compact),
        ("K_inverse_v",Some(g.inverse_velocity_projection_lipschitz),"(1+B_v/V_alg)²=4 when B_v=V_alg","lem-euclidean-reward-regularity",compact),
        ("L_pos_physical",g.reward_position_lipschitz_physical,"sqrt(sum_j [k_j*(r_box+|center_j|)+a*w]²)","lem-euclidean-reward-regularity",compact),
        ("L_R_Sasaki",g.reward_lipschitz_squashed_sasaki,"L_pos_physical*K_inverse_x+2*lambda_vel*V_alg*K_inverse_v/sqrt(lambda_v)","lem-euclidean-reward-regularity",compact),
        ("R_absolute_max",g.reward_absolute_bound,"sum_j [k_j*(r_box+|center_j|)²/2+2*a]+lambda_vel*V_alg²","lem-euclidean-reward-regularity",compact),
        ("r_reference",Some(config.reference_ball_radius),"Declared uniform-ball reference radius","lem-euclidean-richness","Ball centered at zero inside the declared terminal box, velocity fixed to zero"),
        ("kappa_richness_reference",r.predicted_quadratic_reward_variance,"r²*|K*center|²/(d+2)+r⁴*(tr(K²)-tr(K)²/(d+2))/(2*(d+2)*(d+4))","lem-euclidean-richness","Derived exact variance for the declared uniform-ball reference and SPD diagonal quadratic only; no uniform-in-region population lower bound is inferred"),
        ("L_grad",Some(config.nondeception_segment_length),"Declared minimum physical segment length","axiom-non-deceptive","Segments with endpoints inside the convex physical valid box; gradient is that of the physical positional reward"),
        ("kappa_grad",r.nondeception_quadratic_gradient_bound,"min(k)²*L_grad²/12","axiom-non-deceptive","SPD quadratic only; squared-gradient average equals |K(midpoint-center)|²+length²*|K(direction)|²/12"),
    ].into_iter().map(|(symbol,value,formula,label,hypotheses)|SourceMappedConstant {
        symbol:symbol.into(),value,formula:formula.into(),source:source(label),hypotheses:hypotheses.into(),
    }).collect()
}

fn source(label: &str) -> String {
    format!("{CHAPTER_2}#{label}")
}
fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(GasError::Configuration(message.into()))
    }
}
fn norm_sq(x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum()
}
fn lyapunov(x: &[f64], v: &[f64], c: &KineticValidationConfig) -> f64 {
    1. + x
        .iter()
        .zip(&c.landscape.center)
        .map(|(x, center)| (x - center).powi(2))
        .sum::<f64>()
        + c.metric_weight * norm_sq(v)
}
fn estimate(values: impl IntoIterator<Item = f64>) -> MonteCarloEstimate {
    let (mut n, mut mean, mut m2) = (0_usize, 0_f64, 0_f64);
    for value in values {
        n += 1;
        let delta = value - mean;
        mean += delta / n as f64;
        m2 += delta * (value - mean);
    }
    MonteCarloEstimate {
        mean,
        standard_error: (m2.max(0.) / ((n - 1) * n) as f64).sqrt(),
        samples: n,
    }
}
fn status(ok: bool) -> KineticCheckStatus {
    if ok {
        KineticCheckStatus::NotRejected
    } else {
        KineticCheckStatus::Violated
    }
}
fn equality_check(
    item: String,
    label: &str,
    e: MonteCarloEstimate,
    prediction: f64,
    tolerance: f64,
) -> KineticCheck {
    let ok = (e.mean - prediction).abs() <= 6. * e.standard_error + tolerance;
    KineticCheck {
        item,
        source: source(label),
        estimate: Some(e),
        theoretical_value_or_bound: Some(prediction),
        status: status(ok),
        criterion: format!(
            "Absolute conditional-mean residual <= six Monte Carlo standard errors plus numerical/extreme-event tolerance {tolerance}"
        ),
    }
}

/// Abramowitz--Stegun 26.2.17; absolute CDF error is below 7.5e-8.
fn normal_cdf(z: f64) -> f64 {
    let x = z.abs();
    let t = 1. / (1. + 0.2316419 * x);
    let polynomial = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = (-0.5 * x * x).exp() / (2. * std::f64::consts::PI).sqrt() * polynomial;
    if z >= 0. { 1. - tail } else { tail }
}

fn covariance_eigen_extremes(
    x: &[f64],
    v: &[f64],
    n: usize,
    d: usize,
) -> (Option<f64>, Option<f64>) {
    let m = 2 * d;
    let coord = |row: usize, col: usize| {
        if col < d {
            x[row * d + col]
        } else {
            v[row * d + col - d]
        }
    };
    let mean = (0..m)
        .map(|j| (0..n).map(|i| coord(i, j)).sum::<f64>() / n as f64)
        .collect::<Vec<_>>();
    let mut matrix = vec![0.; m * m];
    for a in 0..m {
        for b in a..m {
            let value = (0..n)
                .map(|i| (coord(i, a) - mean[a]) * (coord(i, b) - mean[b]))
                .sum::<f64>()
                / (n - 1) as f64;
            matrix[a * m + b] = value;
            matrix[b * m + a] = value;
        }
    }
    // Symmetric Jacobi rotations. These diagnostic sample eigenvalues carry no
    // claim of an analytic covariance lower bound or an eigenvalue confidence band.
    for _ in 0..100 * m * m {
        let (mut a, mut b, mut largest) = (0, 1, 0.);
        for i in 0..m {
            for j in i + 1..m {
                if matrix[i * m + j].abs() > largest {
                    a = i;
                    b = j;
                    largest = matrix[i * m + j].abs();
                }
            }
        }
        let diagonal_scale = (0..m).map(|i| matrix[i * m + i].abs()).fold(0., f64::max);
        if largest <= 1e-14 * diagonal_scale.max(1e-300) {
            break;
        }
        let angle = 0.5 * (2. * matrix[a * m + b]).atan2(matrix[b * m + b] - matrix[a * m + a]);
        let (s, c) = angle.sin_cos();
        let aa = matrix[a * m + a];
        let bb = matrix[b * m + b];
        let ab = matrix[a * m + b];
        matrix[a * m + a] = c * c * aa - 2. * s * c * ab + s * s * bb;
        matrix[b * m + b] = s * s * aa + 2. * s * c * ab + c * c * bb;
        matrix[a * m + b] = 0.;
        matrix[b * m + a] = 0.;
        for j in 0..m {
            if j != a && j != b {
                let aj = matrix[a * m + j];
                let bj = matrix[b * m + j];
                matrix[a * m + j] = c * aj - s * bj;
                matrix[j * m + a] = matrix[a * m + j];
                matrix[b * m + j] = s * aj + c * bj;
                matrix[j * m + b] = matrix[b * m + j];
            }
        }
    }
    (
        Some(
            (0..m)
                .map(|i| matrix[i * m + i])
                .fold(f64::INFINITY, f64::min),
        ),
        Some(
            (0..m)
                .map(|i| matrix[i * m + i])
                .fold(f64::NEG_INFINITY, f64::max),
        ),
    )
}
