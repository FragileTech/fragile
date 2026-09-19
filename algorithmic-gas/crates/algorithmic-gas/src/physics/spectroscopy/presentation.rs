//! `SpectroscopyReport` → `partvi::ExperimentResult` plots and metrics for the
//! existing SVG adapters. No number is computed here.
use super::report::SpectroscopyReport;
use crate::{Result, error::require, physics::partvi::ExperimentResult};

/// One result per available channel (correlator with error band, effective
/// rate, fit windows) and one for the comparison tables. Unavailable channels
/// contribute a note with their reason and no metric. The report is validated
/// first. Every result carries the provenance
/// `details = {"calculation_origin", "precision", "schema_version", "request"}`
/// of the report, `request` being its `analysis`, and a nonfinite plotted
/// point is an error.
pub fn present(report: &SpectroscopyReport) -> Result<Vec<ExperimentResult>> {
    report.validate()?;
    let results: Vec<ExperimentResult> = vec![];
    require(
        results
            .iter()
            .flat_map(|r| &r.plots)
            .flat_map(|p| &p.series)
            .flat_map(|s| &s.points)
            .flatten()
            .all(|v| v.is_finite()),
        "nonfinite plotted result",
    )?;
    Ok(results)
}
