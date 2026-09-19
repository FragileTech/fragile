//! Fit stability against start time, number of exponentials and SVD cut.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::{AnalysisConfig, StabilityScan},
        estimators::Estimated,
        report::FitOutcome,
    },
};

/// An outcome with `method = FitMethodKind::Stability`, `mass = None` and one
/// `WindowFit` row per scanned variation in `windows`, told apart by `t_min`,
/// `nexp` and `svd_cut`; `notes` state whether the rate is stable within errors.
pub fn stability_scan(
    data: &Estimated,
    scan: &StabilityScan,
    analysis: &AnalysisConfig,
) -> Result<FitOutcome> {
    scan.validate()?;
    let _ = (data, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::fits::stability".into(),
    ))
}
