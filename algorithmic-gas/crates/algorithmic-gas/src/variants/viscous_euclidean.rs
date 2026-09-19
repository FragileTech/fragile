//! The Viscous Euclidean Gas: the Euclidean Gas with one component changed.
//! Both B kicks of BAOAB add the dense Gaussian-kernel viscous force
//! F_i = nu * sum_j K_rho(x_i, x_j) (v_j - v_i) / normalizer, evaluated once per
//! kick on the whole population. Coupling, bandwidth and normalization are free
//! parameters of the variant; no theorem of the Euclidean Gas carries over for
//! nu > 0 without rechecking its hypotheses.
use crate::{GasConfig, Result, kinetic::ViscousForceConfig};

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
