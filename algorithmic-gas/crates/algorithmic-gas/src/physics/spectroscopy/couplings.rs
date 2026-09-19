//! Algorithmic scales of a gas configuration and the coupling proxies derived
//! from them. The Standard Model map is an inversion from reference inputs to
//! gas parameters; it is labelled as such and never presented as a measurement.
use super::{config::StandardModelInputs, measurement::Measurement, report::CouplingReport};
use crate::{GasConfig, GasError, Result};

/// Scales read from the configuration alone.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AlgorithmicScales {
    pub dimension: usize,
    pub dt: Option<f64>,
    pub friction: Option<f64>,
    pub temperature: Option<f64>,
    pub viscosity: Option<f64>,
    pub epsilon_d: Option<f64>,
    pub epsilon_c: Option<f64>,
    pub epsilon_clone: f64,
}
impl AlgorithmicScales {
    pub fn from_config(gas: &GasConfig, dimension: usize) -> Self {
        Self {
            dimension,
            epsilon_clone: gas.clone_decision.epsilon,
            ..Self::default()
        }
    }
}

/// Scales and couplings of one measurement: `gas`, `capabilities.dimension`,
/// `config` (`phase.mass`, `electroweak`) and `calibration`. The couplings
/// `g₁² = ħ_eff N₁ / ε_d²` and `g₃² = (ν²/ħ_eff²) d(d²−1)/12 · E_μ K_visc²`
/// read the measured pair statistics
/// `Calibration::{pair_weight_n1, viscous_kernel_second_moment}` and have
/// `value: None` where those are absent. Every row of `inversion` is an input
/// or a target of the dictionary applied to `inputs`; none compares a gas
/// number with a reference value. Missing ingredients give `value: None`,
/// never an error.
pub fn report(measurement: &Measurement, inputs: &StandardModelInputs) -> Result<CouplingReport> {
    inputs.validate()?;
    let _ = measurement;
    Err(GasError::Capability(
        "pending: spectroscopy::couplings".into(),
    ))
}
