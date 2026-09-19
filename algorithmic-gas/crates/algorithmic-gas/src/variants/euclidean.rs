//! The Euclidean Gas used by the fixed-timestep population theorems.
use crate::{
    GasConfig, Precision, Result,
    boundary::{BoundaryPolicy, BoxDomain},
    cloning::CloneTransform,
    donor::DonorModule,
    error::require,
    fitness::{PositiveMap, Standardizer},
    geometry::{Distance, Kernel},
    kinetic::{KineticBoundarySchedule, KineticKind},
    noise::{FactorValues, Noise, NoiseGeometry},
};

impl GasConfig {
    /// Finite-step Euclidean Gas: frozen component collisions, Gaussian BAOAB,
    /// position diffusion, radial velocity cap, and terminal absorption.
    /// The objective and its force gradient are supplied by the caller.
    pub fn euclidean(dimensions: usize, dt: f64) -> Result<Self> {
        require(
            dimensions > 0 && dimensions <= 256 && dt.is_finite() && dt > 0.,
            "invalid Euclidean Gas dimension/timestep",
        )?;
        let distance = Distance::SquashedPhaseSpace {
            positions: "positions".into(),
            velocities: "velocities".into(),
            position_radius: 2.,
            velocity_radius: 2.,
            lambda: 1.,
        };
        let donor = DonorModule {
            distance,
            kernel: Kernel::Gaussian { width: 2. },
            ..Default::default()
        };
        let mut config = Self {
            precision: Precision::F64,
            distance_donors: donor.clone(),
            cloning_donors: donor,
            boundary: BoundaryPolicy::AbsorbingBox {
                field: "positions".into(),
                domain: BoxDomain {
                    lower: vec![-2.; dimensions],
                    upper: vec![2.; dimensions],
                },
            },
            clone_transform: CloneTransform {
                position_field: Some("positions".into()),
                jitter: Some(Noise::default()),
                jitter_amplitude: 0.1,
                velocity_field: Some("velocities".into()),
                restitution: Some(0.5),
                ..Default::default()
            },
            ..Default::default()
        };
        config.clone_decision.revival_from_companion = true;
        config.fitness.reward_standardizer = Standardizer::Global { sigma_min: 0.1 };
        config.fitness.diversity_standardizer = Standardizer::Global { sigma_min: 0.1 };
        config.fitness.reward_map = PositiveMap::Logistic {
            amplitude: 2.,
            floor: 0.1,
        };
        config.fitness.diversity_map = config.fitness.reward_map.clone();
        config.kinetic.integrator = KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt,
            friction: 1.,
        };
        config.kinetic.noise.geometry = NoiseGeometry::Isotropic {
            scale: FactorValues::Constant { values: vec![1.] },
        };
        config.kinetic.position_diffusion = 0.1;
        config.kinetic.velocity_cap = Some(2.);
        config.kinetic.boundary_schedule = KineticBoundarySchedule::EndOfStep;
        Ok(config)
    }
}
