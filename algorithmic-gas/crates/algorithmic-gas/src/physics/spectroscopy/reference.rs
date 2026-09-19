//! Comparison of assigned channels with a reference table. The assignment is
//! an input hypothesis; nothing here feeds back into a fit.
use super::{
    config::AnalysisConfig,
    report::{ChannelReport, Comparison},
};
use crate::{GasError, Result};

/// The channel an assignment key selects: the channel with that id, else the
/// first channel of that specification id that reports a rate.
pub fn assigned<'a>(channels: &'a [ChannelReport], key: &str) -> Option<&'a ChannelReport> {
    channels
        .iter()
        .find(|c| c.id == key)
        .or_else(|| {
            channels
                .iter()
                .find(|c| c.spec.id() == key && c.mass.is_some())
        })
        .or_else(|| channels.iter().find(|c| c.spec.id() == key))
}

/// Reference rows, one prediction table per anchor (`scale = reference /
/// measured` of the anchor), ratio table with tensions and the spread over
/// anchors. `None` when no assigned channel reports a rate. Ratios do not
/// depend on the time unit; rates of different estimators are never divided.
/// The notes state that the tensions carry no look-elsewhere correction and
/// ignore the correlation between channels.
pub fn compare(
    channels: &[ChannelReport],
    analysis: &AnalysisConfig,
) -> Result<Option<Comparison>> {
    let _ = (channels, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::reference".into(),
    ))
}
