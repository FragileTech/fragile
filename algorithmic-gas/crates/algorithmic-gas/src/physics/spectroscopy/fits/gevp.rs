//! Generalized eigenvalue problem `C(t) v = λ(t, t0) C(t0) v` on a channel
//! basis. Resamples are drawn over origin blocks of the cross-lag products and
//! the problem is solved again in every resample; the symmetrization
//! `(C + Cᵀ)/2` assumes reversibility, so the norm of the discarded
//! antisymmetric part is reported.
use crate::{
    GasError, Result,
    physics::spectroscopy::{
        config::{AnalysisConfig, GevpBasis},
        estimators::MatrixCorrelator,
        report::GevpReport,
    },
};

/// Eigenvalues `λ_n(t, t0)` with errors, effective rates and one window-scan
/// `FitOutcome` per state (with its own diagnostics), keeping the directions
/// of `C(t0)` above `basis.cut`. Every entry of the matrix is resampled with
/// `matrix.tau_int`, so all entries share blocks and draws.
/// `antisymmetric_norm` is reported per lag.
pub fn gevp(
    matrix: &MatrixCorrelator,
    basis: &GevpBasis,
    analysis: &AnalysisConfig,
) -> Result<GevpReport> {
    basis.validate()?;
    let _ = (matrix, analysis);
    Err(GasError::Capability(
        "pending: spectroscopy::fits::gevp".into(),
    ))
}
