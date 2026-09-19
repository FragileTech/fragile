//! The Einstein-Hilbert gas: a free gas whose reward is each walker's share of
//! the Einstein-Hilbert action of the emergent geometry,
//! r_i = R_i * sqrt(det g_i). The reward enters no force; it acts through
//! fitness and cloning only. Walkers are coupled by a viscous force over the
//! tessellation graph, whose curl rotates the velocities (Boris BAOAB), and
//! thermalized by an Ornstein-Uhlenbeck step. Companions for both the diversity
//! distance and cloning are uniform random mutual pairings.
use crate::{
    GasConfig, Result,
    cloning::{CloneDecision, CloneTransform, CollisionRotation},
    donor::{DonorModule, OddPolicy, SamplingLaw},
    error::require,
    fitness::{FitnessPipeline, ObjectiveDirection, PositiveMap, Standardizer},
    geometry::Kernel,
    kinetic::{
        CurlRotationConfig, GraphViscosityConfig, KineticKind, KineticOperator, QftExecutionConfig,
    },
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry},
    tessellation::{
        CurvatureKind, CurvatureSpec, GeometryPipelineConfig, GeometryStageConfig, Projection,
        WeightMode, WeightSpec, presets::RICCI_SCALAR,
    },
};

/// Temperature of the reference instance (`RunConfig::einstein_hilbert`).
pub const REFERENCE_TEMPERATURE: f64 = 0.33;

impl GasConfig {
    /// Einstein-Hilbert gas over the fields `positions` and `velocities`, at
    /// the given temperature and BAOAB time step. Build it with
    /// `GeometryReward::default()` as the reward and `ZeroPotential` as the
    /// gradient. The last of three or more position coordinates is Euclidean
    /// time and is left out of the tessellation. Every component is an ordinary
    /// field of the returned configuration: cloning period and elasticity in
    /// `clone_decision` / `clone_transform`, coupling and rotation strength in
    /// `qft`, the geometry pipeline and its schedule in `geometry`.
    pub fn einstein_hilbert(temperature: f64, dt: f64) -> Result<Self> {
        require(
            temperature.is_finite() && temperature > 0. && dt.is_finite() && dt > 0.,
            "invalid Einstein-Hilbert gas temperature/timestep",
        )?;
        let friction = 1.;
        let pairing = DonorModule {
            kernel: Kernel::Uniform,
            law: SamplingLaw::FisherYates,
            odd: OddPolicy::SelfCompanion,
            ..DonorModule::default()
        };
        // Sample standard deviation: a constant channel standardizes to zero,
        // as at a coincident start. No positivity floor beyond the logistic map.
        let standardizer = Standardizer::LegacySample { epsilon: 1e-30 };
        let map = PositiveMap::Logistic {
            amplitude: 2.,
            floor: 0.,
        };
        let viscous = WeightMode::RiemannianKernelVolume;
        Ok(Self {
            distance_donors: pairing.clone(),
            cloning_donors: pairing,
            fitness: FitnessPipeline {
                direction: ObjectiveDirection::Maximize,
                reward_standardizer: standardizer.clone(),
                diversity_standardizer: standardizer,
                reward_map: map.clone(),
                diversity_map: map,
                distance_floor: 1e-30,
                ..FitnessPipeline::default()
            },
            clone_decision: CloneDecision {
                epsilon: 0.,
                every: 20,
                ..CloneDecision::default()
            },
            // Elastic collisions that keep the direction of relative velocities.
            clone_transform: CloneTransform {
                velocity_field: Some("velocities".into()),
                restitution: Some(1.),
                collision_rotation: CollisionRotation::Identity,
                ..CloneTransform::default()
            },
            kinetic: KineticOperator {
                integrator: KineticKind::Baoab {
                    positions: "positions".into(),
                    velocities: "velocities".into(),
                    dt,
                    friction,
                },
                // dv = B dW with B = sqrt(2 gamma T): stationary velocity variance T.
                noise: Noise {
                    innovation: InnovationLaw::Gaussian,
                    geometry: NoiseGeometry::Isotropic {
                        scale: FactorValues::Constant {
                            values: vec![(2. * friction * temperature).sqrt()],
                        },
                    },
                },
                ..KineticOperator::default()
            },
            qft: QftExecutionConfig {
                graph_viscosity: Some(GraphViscosityConfig {
                    coefficient: 3.,
                    weights: viscous.name().into(),
                }),
                curl: Some(CurlRotationConfig { beta_curl: 1. }),
                ..QftExecutionConfig::default()
            },
            geometry: Some(GeometryStageConfig {
                pipeline: GeometryPipelineConfig {
                    projection: Projection::DropLast { min_ambient: 3 },
                    weights: vec![
                        WeightSpec::new(WeightMode::InverseRiemannianDistance),
                        WeightSpec::new(viscous),
                    ],
                    curvature: vec![CurvatureSpec {
                        name: RICCI_SCALAR.into(),
                        estimator: CurvatureKind::ConformalLaplacian {
                            weights: WeightMode::InverseRiemannianDistance.name().into(),
                            det_floor: 1e-12,
                        },
                    }],
                    ..GeometryPipelineConfig::default()
                },
                ..GeometryStageConfig::default()
            }),
            ..Self::default()
        })
    }
}
