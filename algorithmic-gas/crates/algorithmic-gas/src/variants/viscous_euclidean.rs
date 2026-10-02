//! The Viscous Euclidean Gas: the Euclidean Gas with one component changed.
//! Both B kicks of BAOAB add the dense Gaussian-kernel viscous force
//! F_i = nu * sum_j K_rho(x_i, x_j) (v_j - v_i) / normalizer, evaluated once per
//! kick on the whole population. Coupling, bandwidth and normalization are free
//! parameters of the variant; no theorem of the Euclidean Gas carries over for
//! nu > 0 without rechecking its hypotheses.
use crate::{
    GasConfig, ObservationBatch, Real, Result,
    error::require,
    kinetic::ViscousForceConfig,
    tessellation::{GeometryPipelineConfig, TessellationGeometry},
};

/// Observe all three spatial coordinates of every retained slot at one
/// declared recording stage, including slots marked dead in the gas.
///
/// This is the passive all-slot instrument of the book's
/// `def-variant-recorded-color-geometry`. The caller supplies the complete
/// geometry pipeline, any prior cell volumes required by a history-dependent
/// estimator, and the edge budget. The function reads observations only: it
/// cannot change gas fields, eligibility, reward, kinetics or random addresses.
/// Unlike `GeometryStageConfig::refresh`, it does not filter by alive marks or
/// write geometry back into the dynamical population. The returned geometry
/// belongs to the supplied stage; it must not be paired with another stage's
/// color force or velocity without an explicit alignment convention.
pub fn observe_recorded_geometry<T: Real>(
    observations: &ObservationBatch<T>,
    pipeline: &GeometryPipelineConfig,
    previous_cell_volume: Option<&[T]>,
    max_edges: usize,
) -> Result<TessellationGeometry<T>> {
    let positions = observations.field(&pipeline.positions)?;
    require(
        positions.item_shape() == [3],
        "recorded color-geometry gas requires three spatial coordinates",
    )?;
    require(
        pipeline.projection.axes(3)? == [0, 1, 2],
        "recorded geometry must retain all three spatial coordinates",
    )?;
    pipeline.validate::<T>(3)?;
    pipeline.evaluate(
        observations,
        &vec![true; positions.rows()],
        previous_cell_volume,
        max_edges,
    )
}

/// Coupling of the reference instance: nu = 0.3 per unit time, bandwidth
/// rho = 1 (half the companion width and half the box half-width of the
/// Euclidean Gas), eligible-count normalization. That normalization is the
/// book's total pairwise force with coupling nu / N and conserves the summed
/// momentum of the eligible walkers; row normalization need not.
pub fn reference_viscosity() -> ViscousForceConfig {
    ViscousForceConfig {
        coefficient: 0.3,
        bandwidth: 1.,
        row_normalized: false,
    }
}

impl GasConfig {
    /// `GasConfig::euclidean(dimensions, dt)` with `qft.viscosity` set to the
    /// given coupling; every other field is that of the Euclidean Gas. A zero
    /// coefficient leaves the trajectory law of the Euclidean Gas unchanged.
    pub fn viscous_euclidean(
        dimensions: usize,
        dt: f64,
        viscosity: ViscousForceConfig,
    ) -> Result<Self> {
        let mut config = Self::euclidean(dimensions, dt)?;
        config.qft.viscosity = Some(viscosity);
        // No innovation shifts are configured, so the row count is not consulted.
        config.qft.validate(1, dimensions)?;
        Ok(config)
    }
}
