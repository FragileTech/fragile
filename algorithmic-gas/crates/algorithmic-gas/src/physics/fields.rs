//! Part VI geometry, finite-step field, boundary, and closure experiments.
//! Numerical measurements and analytic references are evaluated separately.
use super::partvi::{ExperimentRequest, ExperimentResult, Metric, Plot, Series};
use crate::{GasError, Result, tracking::RunArchive};
use serde_json::{Value, json};
mod boundary;
mod cosmology;
mod gravity;
pub use gravity::{archive_fitness_jet, archive_fitness_jet_with_history};
mod models;
mod numerics;
mod observed;
use numerics::*;

pub fn analyze(
    req: &ExperimentRequest,
    archive: Option<&RunArchive<f64>>,
) -> Result<ExperimentResult> {
    if !(37..=66).contains(&req.experiment) {
        return Err(GasError::Configuration(
            "fields experiment must be 37..66".into(),
        ));
    }
    for (_, value) in req.parameters.as_object().into_iter().flatten() {
        if value.is_number() && !value.as_f64().is_some_and(f64::is_finite) {
            return Err(GasError::Configuration(
                "finite experiment parameters required".into(),
            ));
        }
    }
    match req.experiment {
        37..=44 => gravity::run(req, archive),
        45..=51 => models::run(req, archive),
        52..=60 => boundary::run(req, archive),
        _ => cosmology::run(req, archive),
    }
}
fn result(r: &ExperimentRequest, title: &str, model: &str) -> ExperimentResult {
    ExperimentResult {
        experiment: r.experiment,
        title: title.into(),
        model: model.into(),
        metrics: vec![],
        plots: vec![],
        notes: vec![],
        details: json!({}),
    }
}
fn metric(r: &mut ExperimentResult, label: &str, value: f64, unit: &str) {
    r.metrics.push(Metric {
        label: label.into(),
        value: value.is_finite().then_some(value),
        unit: unit.into(),
    });
}
fn plot(r: &mut ExperimentResult, title: &str, x: &str, y: &str, series: Vec<Series>) {
    r.plots.push(Plot {
        title: title.into(),
        x_label: x.into(),
        y_label: y.into(),
        series,
    });
}
fn line(name: &str, points: Vec<[f64; 2]>) -> Series {
    Series {
        name: name.into(),
        points,
        kind: "line".into(),
    }
}
fn p(r: &ExperimentRequest, key: &str, default: f64, lo: f64, hi: f64) -> f64 {
    r.number(key, default).clamp(lo, hi)
}
fn n(r: &ExperimentRequest, key: &str, default: usize, lo: usize, hi: usize) -> usize {
    r.usize(key, default).clamp(lo, hi)
}
fn seed(r: &ExperimentRequest) -> u64 {
    r.usize("seed", 7) as u64
}
