//! The Einstein-Hilbert gas: a free gas whose reward is each walker's share of
//! the Einstein-Hilbert action of the emergent geometry,
//! r_i = scale * R_i * sqrt(det g_i). The reward enters no force; it acts
//! through fitness and cloning only. Walkers are coupled by a viscous force
//! over the tessellation graph, whose curl rotates the velocities (Boris
//! BAOAB), and thermalized by an Ornstein-Uhlenbeck step at temperature T.
//! Companions for both the diversity distance and cloning are uniform random
//! mutual pairings.
use super::{
    CurvatureKind, CurvatureSpec, GeometryPipelineConfig, GeometryReward, GeometryStageConfig,
    GeometryTiming, Projection, RewardAllocationKind, WeightMode, WeightSpec,
};
use crate::{
    GasConfig, Precision,
    cloning::{CloneDecision, CloneTransform, CollisionRotation},
    donor::{DonorModule, OddPolicy, SamplingLaw},
    fitness::{FitnessPipeline, ObjectiveDirection, PositiveMap, Standardizer},
    geometry::Kernel,
    kinetic::{
        CurlRotationConfig, GraphViscosityConfig, KineticBoundarySchedule, KineticKind,
        KineticOperator, QftExecutionConfig,
    },
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry},
};
use serde::{Deserialize, Serialize};

pub const RICCI_SCALAR: &str = "ricci_scalar";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EinsteinHilbertGas {
    /// Scale of the reward r_i = eh_scale * R_i * sqrt(det g_i).
    pub eh_scale: f64,
    /// Treat the last position coordinate as Euclidean time and tessellate
    /// the remaining ones (when there are at least three coordinates).
    pub drop_time_axis: bool,
    pub timing: GeometryTiming,
    pub refresh_every: u64,
    pub clone_every: u64,
    /// Friction, temperature and time step of the BAOAB integrator.
    pub gamma: f64,
    pub temperature: f64,
    pub dt: f64,
    /// Viscous coupling strength, its edge weighting and kernel length scale.
    pub nu: f64,
    pub viscous_weights: WeightMode,
    pub length_scale: f64,
    /// Strength of the Boris rotation by the curl of the viscous force.
    pub beta_curl: f64,
    /// Cloning: saturation p_max, regularizer, position jitter and restitution.
    pub p_max: f64,
    pub epsilon_clone: f64,
    pub sigma_x: f64,
    pub restitution: f64,
    /// Fitness V = (d')^beta (r')^alpha with r' = amplitude * sigmoid(z) + eta.
    pub alpha: f64,
    pub beta: f64,
    pub eta: f64,
    pub amplitude: f64,
}
impl Default for EinsteinHilbertGas {
    fn default() -> Self {
        Self {
            eh_scale: 1.,
            drop_time_axis: true,
            timing: GeometryTiming::AfterCloning,
            refresh_every: 1,
            clone_every: 20,
            gamma: 1.,
            temperature: 0.33,
            dt: 0.002,
            nu: 3.,
            viscous_weights: WeightMode::RiemannianKernelVolume,
            length_scale: 1.,
            beta_curl: 1.,
            p_max: 1.,
            epsilon_clone: 0.,
            sigma_x: 0.,
            restitution: 1.,
            alpha: 1.,
            beta: 1.,
            eta: 0.,
            amplitude: 2.,
        }
    }
}
impl EinsteinHilbertGas {
    pub fn reward(&self) -> GeometryReward {
        GeometryReward {
            curvature: RICCI_SCALAR.into(),
            allocation: RewardAllocationKind::EinsteinHilbertDensity {
                scale: self.eh_scale,
            },
        }
    }
    pub fn geometry(&self) -> GeometryStageConfig {
        let mut modes = vec![
            WeightMode::InverseRiemannianDistance,
            WeightMode::Kernel,
            WeightMode::RiemannianKernelVolume,
        ];
        if !modes.contains(&self.viscous_weights) {
            modes.push(self.viscous_weights);
        }
        GeometryStageConfig {
            pipeline: GeometryPipelineConfig {
                projection: if self.drop_time_axis {
                    Projection::DropLast { min_ambient: 3 }
                } else {
                    Projection::Full
                },
                weights: modes
                    .into_iter()
                    .map(|mode| WeightSpec {
                        length_scale: self.length_scale,
                        ..WeightSpec::new(mode)
                    })
                    .collect(),
                curvature: vec![CurvatureSpec {
                    name: RICCI_SCALAR.into(),
                    estimator: CurvatureKind::ConformalLaplacian {
                        weights: WeightMode::InverseRiemannianDistance.name().into(),
                        det_floor: 1e-12,
                    },
                }],
                ..GeometryPipelineConfig::default()
            },
            timing: self.timing,
            refresh_every: self.refresh_every,
            refresh_on_clone: true,
            write_diffusion: true,
            record_graph: false,
        }
    }
    /// Engine configuration over the fields `positions` and `velocities`.
    /// Build the gas with `reward()` and a `ZeroPotential` gradient.
    pub fn config(&self, precision: Precision, seed: u64) -> GasConfig {
        let pairing = DonorModule {
            kernel: Kernel::Uniform,
            law: SamplingLaw::FisherYates,
            odd: OddPolicy::SelfCompanion,
            ..DonorModule::default()
        };
        // Sample standard deviation; a constant channel standardizes to zero.
        let standardizer = Standardizer::LegacySample { epsilon: 1e-30 };
        let map = PositiveMap::Logistic {
            amplitude: self.amplitude,
            floor: self.eta,
        };
        GasConfig {
            precision,
            seed,
            distance_donors: pairing.clone(),
            cloning_donors: pairing,
            fitness: FitnessPipeline {
                direction: ObjectiveDirection::Maximize,
                reward_standardizer: standardizer.clone(),
                diversity_standardizer: standardizer,
                reward_map: map.clone(),
                diversity_map: map,
                reward_exponent: self.alpha,
                diversity_exponent: self.beta,
                distance_floor: 1e-30,
            },
            clone_decision: CloneDecision {
                epsilon: self.epsilon_clone,
                saturation: self.p_max,
                revival_from_companion: false,
                every: self.clone_every,
            },
            clone_transform: CloneTransform {
                position_field: Some("positions".into()),
                jitter: (self.sigma_x > 0.).then(Noise::default),
                jitter_amplitude: self.sigma_x,
                velocity_field: Some("velocities".into()),
                restitution: Some(self.restitution),
                collision_rotation: CollisionRotation::Identity,
            },
            kinetic: KineticOperator {
                integrator: KineticKind::Baoab {
                    positions: "positions".into(),
                    velocities: "velocities".into(),
                    dt: self.dt,
                    friction: self.gamma,
                },
                // dv = B dW with B = sqrt(2 gamma T): the O step then has the
                // stationary velocity variance T.
                noise: Noise {
                    innovation: InnovationLaw::Gaussian,
                    geometry: NoiseGeometry::Isotropic {
                        scale: FactorValues::Constant {
                            values: vec![(2. * self.gamma * self.temperature).sqrt()],
                        },
                    },
                },
                position_diffusion: 0.,
                velocity_cap: None,
                boundary_schedule: KineticBoundarySchedule::Substeps,
            },
            qft: QftExecutionConfig {
                graph_viscosity: Some(GraphViscosityConfig {
                    coefficient: self.nu,
                    weights: self.viscous_weights.name().into(),
                }),
                curl: (self.beta_curl > 0.).then_some(CurlRotationConfig {
                    beta_curl: self.beta_curl,
                }),
                ..QftExecutionConfig::default()
            },
            geometry: Some(self.geometry()),
            ..GasConfig::default()
        }
    }
}
