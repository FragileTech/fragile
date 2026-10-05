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

/// Bounds for the actual quarter-kick matrix P = I + (h nu / 4)(W - diag(W 1)).
/// These are diagnostics of one supplied graph, not a uniform-in-population
/// certificate for a changing tessellation or a stationarity assertion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphKickBounds {
    pub maximum_row_sum: f64,
    pub maximum_column_sum: f64,
    /// Maximum column sum of P. Jensen gives sum |P v|^p <= kappa sum |v|^p
    /// for p >= 1. One Boris B stage has factor kappa^2 in this estimate.
    pub quarter_kick_moment_factor: f64,
}

/// Check nonnegative finite weights and the convexity condition for a frozen
/// graph. Includes isolated rows and the row-sum floor used by the native
/// weighting code. Does not change or renormalize the executed weights.
pub fn graph_kick_bounds<T: crate::Real>(
    graph: &crate::tessellation::NeighborGraph,
    weights: &[T],
    coefficient: f64,
    dt: f64,
) -> Result<GraphKickBounds> {
    graph.validate()?;
    require(
        graph.nodes() > 0
            && weights.len() == graph.edges()
            && coefficient.is_finite()
            && coefficient >= 0.
            && dt.is_finite()
            && dt > 0.,
        "invalid graph kick bound inputs",
    )?;
    let a = (dt * 0.25) * coefficient;
    require(a.is_finite(), "graph kick coefficient overflow")?;
    let mut rows = vec![0.; graph.nodes()];
    let mut columns = vec![0.; graph.nodes()];
    for (i, row) in rows.iter_mut().enumerate() {
        for e in graph.range(i) {
            let w = weights[e].to_f64();
            require(w.is_finite() && w >= 0., "invalid graph kick weight")?;
            *row += w;
            columns[graph.neighbors()[e] as usize] += w;
        }
    }
    require(
        rows.iter().chain(&columns).all(|x| x.is_finite()),
        "graph kick weight sum overflow",
    )?;
    let maximum_row_sum = rows.iter().copied().fold(0., f64::max);
    require(
        a * maximum_row_sum <= 1.,
        "graph quarter-kick is not a convex average: dt * coefficient * max_row_sum > 4",
    )?;
    Ok(GraphKickBounds {
        maximum_row_sum,
        maximum_column_sum: columns.iter().copied().fold(0., f64::max),
        quarter_kick_moment_factor: rows
            .iter()
            .zip(&columns)
            .map(|(&row, &column)| 1. - a * row + a * column)
            .fold(0., f64::max),
    })
}

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
