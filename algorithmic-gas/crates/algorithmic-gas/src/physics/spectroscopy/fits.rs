//! Rates from correlators. The fitted quantity is the decay rate of the
//! algorithm-time autocorrelation; every fit may decline to report one
//! (rejected model, no signal, dominated prior) and then reports no number.
pub mod effective_mass;
pub mod gevp;
pub mod multi_exponential;
pub mod stability;
pub mod window_scan;
use super::{
    config::{AnalysisConfig, ChannelGroup, FitMethod},
    estimators::Estimated,
    report::{FitOutcome, GroupFit},
};
use crate::Result;

/// The configured fit methods on one channel, in reporting order.
pub fn fit_channel(data: &Estimated, analysis: &AnalysisConfig) -> Result<Vec<FitOutcome>> {
    let mut out = vec![];
    if matches!(analysis.fit, FitMethod::WindowScan | FitMethod::Both) {
        out.push(window_scan::window_scan(data, analysis)?);
    }
    if matches!(analysis.fit, FitMethod::MultiExponential | FitMethod::Both) {
        out.push(multi_exponential::fit(
            std::slice::from_ref(data),
            &analysis.multi_exponential,
            analysis,
        )?);
    }
    // Last, so that `ChannelReport::mass` stays the first fit with a rate.
    if let Some(scan) = &analysis.stability {
        out.push(stability::stability_scan(data, scan, analysis)?);
    }
    Ok(out)
}

/// Joint multi-exponential fit of channels resampled over the same blocks.
pub fn fit_group(
    group: &ChannelGroup,
    members: &[Estimated],
    analysis: &AnalysisConfig,
) -> Result<GroupFit> {
    multi_exponential::fit_group(group, members, analysis)
}
