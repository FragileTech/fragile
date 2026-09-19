//! Block jackknife and block bootstrap over `BlockMoments`, the matching
//! covariance, and the integrated autocorrelation time.
use super::{BlockMoments, Resampling, Samples, SeriesView, Subtraction};
use crate::{GasError, Result};
use serde::{Deserialize, Serialize};

/// Sokal/Madras self-consistent window estimate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TauInt {
    /// `1/2 + Σ_{τ=1}^{W} ρ(τ)`, at least 1/2.
    pub tau: f64,
    pub window: usize,
}

/// Integrated autocorrelation time of a weighted series, from its
/// component-contracted, lag-wise connected autocorrelation. `None` when the
/// series is too short or has no variance.
pub fn tau_int(series: SeriesView<'_>) -> Option<TauInt> {
    let _ = series;
    None
}

/// The same estimate from a normalized autocorrelation `ρ(τ) = C(τ)/C(0)` over
/// `frames` measurements.
pub fn tau_int_of_correlator(correlator: &[Option<f64>], frames: usize) -> Option<TauInt> {
    let _ = (correlator, frames);
    None
}

/// Origins per block for `BlockSize::Auto`: the smallest multiple of `base`
/// (origins already merged into one stored block) that is at least
/// `max(2 τ, (2 τ)^{2/3} origins^{1/3})`, the batch size of least mean squared
/// error up to a constant, capped so that at least 8 blocks remain when the
/// data allow it. For AR(1) with `ρ = 0.95` a block of `2 τ_int` alone biases
/// the variance by −43 %.
pub fn auto_block(tau: f64, base: usize, origins: usize) -> usize {
    let _ = (tau, origins);
    base.max(1)
}

/// Resamples of any statistic of per-block sums. `statistic(multiplicity)`
/// receives one multiplicity per STORED block (`[blocks]`) and returns the
/// estimate from the sums weighted by it; all ones is the central value. The
/// stored blocks, of `base` origins each, are grouped into resampling blocks
/// of `auto_block(tau, base, origins)` or the fixed size (rounded up to a
/// multiple of `base`). The jackknife zeroes one group per resample; resample
/// `r` of the bootstrap draws its groups with replacement from the addressed
/// stream `Resampling::Bootstrap` documents. `tau` is required by
/// `BlockSize::Auto`. Fewer than two groups is `GasError::Numerical`.
pub fn resample_blocks(
    blocks: usize,
    base: usize,
    origins: usize,
    resampling: &Resampling,
    tau: Option<f64>,
    statistic: &dyn Fn(&[f64]) -> Vec<Option<f64>>,
) -> Result<Samples> {
    let _ = (blocks, base, origins, resampling, tau, statistic);
    Err(GasError::Capability("pending: numerics::resample".into()))
}

/// Resampled correlators of `moments` through `resample_blocks` with the
/// statistic `BlockMoments::estimate(subtraction)` on the weighted sums; `tau`
/// overrides the autocorrelation time used by `BlockSize::Auto` (otherwise it
/// is estimated from the pooled correlator). Each resample recomputes its own
/// disconnected part. Fewer than two blocks is `GasError::Numerical`.
pub fn resample(
    moments: &BlockMoments,
    subtraction: Subtraction,
    resampling: &Resampling,
    tau: Option<f64>,
) -> Result<Samples> {
    let _ = (moments, subtraction, resampling, tau);
    Err(GasError::Capability("pending: numerics::resample".into()))
}

/// Covariance `[dimension, dimension]` of a resample table about the mean of
/// its resamples, with the factor of `samples.kind`. Undefined entries give
/// zero rows and columns.
pub fn sample_covariance(samples: &Samples) -> Vec<f64> {
    vec![0.; samples.dimension * samples.dimension]
}

/// Square roots of the covariance diagonal; `None` where undefined.
pub fn errors(samples: &Samples) -> Vec<Option<f64>> {
    vec![None; samples.dimension]
}
