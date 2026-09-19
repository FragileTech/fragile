//! Measurement → correlator estimates with resamples. Three estimators share
//! one output type: the algorithm-time correlation of frame averages, the
//! source-frozen propagator, and the Euclidean-time slab correlation.
use super::{
    config::{AnalysisConfig, TimeUnit},
    measurement::Measurement,
    report::{CorrelatorEstimate, EstimatorKind},
};
use crate::{
    GasError, Result,
    physics::numerics::{BlockMoments, Samples},
};

/// A correlator and the resamples its errors came from; fits read both.
#[derive(Clone, Debug, PartialEq)]
pub struct Estimated {
    pub channel: String,
    pub kind: EstimatorKind,
    /// `covariance` is always `Some(sample_covariance(&samples))` here;
    /// `analyze` drops it from the report unless `report_covariance`.
    pub estimate: CorrelatorEstimate,
    /// Dimension `estimate.lags.len()`. Fits take their times from
    /// `estimate.lags[k] as f64 * estimate.time_step`.
    pub samples: Samples,
}

/// Cross-correlation moments of a channel basis over common blocks:
/// `moments[a * channels.len() + b]` holds `⟨O_a(t) O_b(t+τ)⟩`. `gevp` calls
/// `resample(&moments[k], subtraction, &analysis.resampling, tau_int)` with
/// the same `tau_int` for every entry, so that all entries share their blocks
/// and bootstrap draws; an automatic block per entry would be estimated from
/// meaningless off-diagonal ratios.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixCorrelator {
    pub channels: Vec<String>,
    /// Lag of every entry of the moments, in frames.
    pub lags: Vec<usize>,
    pub time_unit: TimeUnit,
    pub time_step: f64,
    /// Measured frames summed over replicas.
    pub frames: u64,
    /// Largest integrated autocorrelation time of the diagonal members.
    pub tau_int: Option<f64>,
    pub moments: Vec<BlockMoments>,
    pub replicas: usize,
}

/// Blocked moments of one channel under one estimator, combined over replicas
/// as `analysis.combine` says (`PooledBlocks` concatenates the replicas'
/// blocks; `RunsAsSamples` makes every replica one block and needs at least 8).
/// `GasError::Capability` when the channel or the estimator was not measured.
pub fn moments(
    measurements: &[Measurement],
    channel: &str,
    kind: EstimatorKind,
    analysis: &AnalysisConfig,
) -> Result<BlockMoments> {
    let _ = (measurements, channel, kind, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::estimators".into(),
    ))
}

/// Point estimate (the pooled moments), resamples, errors, covariance (always
/// filled), rank and the connected-estimator bias of one channel. The
/// disconnected part is `Subtraction::None` when `!analysis.connected`,
/// `analysis.frame_subtraction` for the frame mean and
/// `analysis.propagator_subtraction` for the source-frozen propagator, always
/// formed from sums pooled over replicas; the Euclidean-time estimator resamples
/// `SlabMoments` through `numerics::resample_blocks` with per-bin means and
/// reports `TimeUnit::Coordinate` with the bin width as `time_step`. Rates
/// fitted downstream are labelled by estimator (`report::RATE_QUANTITY*`).
/// `tau` overrides the autocorrelation time of an automatic block, which is
/// otherwise that of the retained operator series. Too few blocks for the
/// requested resampling is `GasError::Numerical`.
pub fn estimate(
    measurements: &[Measurement],
    channel: &str,
    kind: EstimatorKind,
    analysis: &AnalysisConfig,
    tau: Option<f64>,
) -> Result<Estimated> {
    let _ = (measurements, channel, kind, analysis, tau);
    Err(GasError::Capability(
        "pending: spectroscopy::estimators".into(),
    ))
}

/// Several channels resampled over the same blocks (the largest automatic
/// block of the members), so that `Samples::concat` yields their joint table.
pub fn joint(
    measurements: &[Measurement],
    channels: &[(String, EstimatorKind)],
    analysis: &AnalysisConfig,
) -> Result<Vec<Estimated>> {
    let _ = (measurements, channels, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::estimators".into(),
    ))
}

/// Frame-mean cross moments of a channel basis for the GEVP. Products of
/// different lags stay inside their origin block. Every member must have the
/// same `components`, else `GasError::Configuration`.
pub fn matrix(
    measurements: &[Measurement],
    channels: &[String],
    analysis: &AnalysisConfig,
) -> Result<MatrixCorrelator> {
    let _ = (measurements, channels, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::estimators".into(),
    ))
}
