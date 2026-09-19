//! Single-exponential fits over all windows, model-averaged with
//! `w ∝ exp(−AIC/2)`, `AIC = χ² + 2k + 2 N_cut` (Jay–Neil).
use crate::{
    GasError, Result,
    physics::spectroscopy::{config::AnalysisConfig, estimators::Estimated, report::FitOutcome},
};

/// Windows `[t_min, t_max]` from `analysis.window_scan`, correlated χ² through
/// a `Whitener` with `analysis.svd_cut` (or the stated diagonal approximation),
/// `value = Σ w m`, `statistical² = Σ w σ²`, `systematic² = Σ w (m − value)²`.
/// A correlator that changes sign or oscillates inside the usable range sets
/// `model_rejected`; a rate below `min_rate_snr` sets `no_signal`. Neither
/// case reports a number, and no positivity filter biases the average.
/// `N_cut` counts the points of the contiguous usable range left out of a
/// window. A whitener of reduced rank deflates χ²; the notes say so.
pub fn window_scan(data: &Estimated, analysis: &AnalysisConfig) -> Result<FitOutcome> {
    let _ = (data, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::fits::window_scan".into(),
    ))
}
