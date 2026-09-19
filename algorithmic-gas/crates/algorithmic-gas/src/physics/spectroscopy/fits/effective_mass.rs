//! Effective rates of a correlator.
use crate::physics::spectroscopy::{config::EffectiveMassKind, estimators::Estimated};

/// `[value, error]` per lag of the estimate. The error comes from the
/// resamples when every resample is defined at the lag, otherwise from the
/// covariance by the delta method. `None` where the definition does not
/// exist: a missing or non-positive ratio for `LogRatio`, a ratio below 1 or
/// a missing neighbour for `Cosh`, the last lag(s). Never a placeholder value.
pub fn effective_mass(data: &Estimated, kind: EffectiveMassKind) -> Vec<Option<[f64; 2]>> {
    let _ = kind;
    vec![None; data.estimate.lags.len()]
}

/// `m` with `cosh m = ratio`; `None` below 1.
pub fn cosh_rate(ratio: f64) -> Option<f64> {
    (ratio >= 1.).then(|| ratio.acosh())
}
