//! `C_c(t) = Σ_n a_{c,n}² exp(−E_n t)`, `E_n = Σ_{m ≤ n} dE_m`, with gaps
//! shared by every channel of a group. Autocorrelations have equal source and
//! sink amplitudes. Priors are channel agnostic and scale free; every result
//! carries the prior-dominance diagnostic. Positive amplitudes cannot
//! represent the negative spectral weights of a non-reversible chain: such
//! data end as a rejected model, never as a fit.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::{AnalysisConfig, ChannelGroup, MultiExponentialConfig},
        estimators::Estimated,
        report::{FitOutcome, GroupFit},
    },
};

/// Fit of one or more channels that share gaps; `members` must have been
/// resampled over the same blocks. Parameters are `ln dE_n` and `ln a²`, so
/// the log-normal priors are Gaussian. A posterior/prior width ratio of
/// `ln dE_0` above `config.dominance_ratio` reports no rate. Every excited
/// level carries the `MassEstimate::prior_dominance` of its own gap.
/// `FitDiagnostics::chi2` is the augmented χ² and `q` its probability; the
/// prior part is stated in a note. `t_max: None` ends at the last lag that
/// passes `window_scan.min_point_snr`.
pub fn fit(
    members: &[Estimated],
    config: &MultiExponentialConfig,
    analysis: &AnalysisConfig,
) -> Result<FitOutcome> {
    config.validate()?;
    let _ = (members, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::fits::multi_exponential".into(),
    ))
}

/// `fit` on the members of a configured group, reported with all levels.
pub fn fit_group(
    group: &ChannelGroup,
    members: &[Estimated],
    analysis: &AnalysisConfig,
) -> Result<GroupFit> {
    group.validate()?;
    let _ = (members, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::fits::multi_exponential".into(),
    ))
}
