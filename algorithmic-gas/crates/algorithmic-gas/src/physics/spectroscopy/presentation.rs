//! `SpectroscopyReport` → `partvi::ExperimentResult` plots and metrics for the
//! existing SVG adapters. No estimate is computed here: the only arithmetic is
//! the lag axis `lag · time_step` and the band edges `value ± error`. An
//! undefined number is a gap in a curve or a metric without a value, never a
//! zero.
use super::{
    config::{FrameNormalization, TimeUnit},
    contract::{ExchangeParity, SpatialParity},
    report::{
        ChannelReport, Comparison, CouplingReport, EstimatorKind, FitMethodKind, FitOutcome,
        FlowDiagnostic, GevpReport, GroupFit, MassEstimate, PriorDominance, Quantity,
        RATE_QUANTITY, RATE_QUANTITY_EUCLIDEAN, RATE_QUANTITY_SOURCE_FROZEN, SpectroscopyReport,
    },
};
use crate::{
    Result,
    error::require,
    physics::partvi::{ExperimentResult, Metric, Series},
};
use serde_json::json;

/// `ExperimentResult::experiment` of every result: outside the registered
/// Part VI range.
pub const EXPERIMENT: u32 = 0;

/// Line series over the maximal runs of defined points, all under one name: a
/// line is never drawn across an undefined point.
fn runs(name: &str, points: impl IntoIterator<Item = Option<[f64; 2]>>) -> Vec<Series> {
    let mut series = vec![];
    let mut run = vec![];
    for point in points.into_iter().chain([None]) {
        match point {
            Some(p) => run.push(p),
            None if run.is_empty() => {}
            None => series.push(Series::line(name, std::mem::take(&mut run))),
        }
    }
    series
}

/// A curve with its error band: the values, then `value + error` and
/// `value − error` where both are defined.
fn band(name: &str, x: &[f64], values: &[Option<f64>], errors: &[Option<f64>]) -> Vec<Series> {
    let edge = |sign: f64| {
        x.iter().zip(values).zip(errors).map(move |((x, v), e)| {
            let (v, e) = (*v).zip(*e)?;
            Some([*x, v + sign * e])
        })
    };
    let central = x.iter().zip(values).map(|(x, v)| v.map(|v| [*x, v]));
    let mut series = runs(name, central);
    series.extend(runs(&format!("{name} + error"), edge(1.)));
    series.extend(runs(&format!("{name} - error"), edge(-1.)));
    series
}
fn split(pairs: &[Option<[f64; 2]>]) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    pairs
        .iter()
        .map(|p| (p.map(|[v, _]| v), p.map(|[_, e]| e)))
        .unzip()
}
fn metric(result: &mut ExperimentResult, label: impl Into<String>, value: Option<f64>, unit: &str) {
    result.metrics.push(Metric {
        label: label.into(),
        value: value.filter(|v| v.is_finite()),
        unit: unit.into(),
    });
}
fn lag_label(unit: TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Frames => "lag (frames)",
        TimeUnit::StepDt => "lag (time)",
        TimeUnit::Coordinate => "separation (length)",
    }
}
fn rate_unit(unit: TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Frames => "1/frame",
        TimeUnit::StepDt => "1/time",
        TimeUnit::Coordinate => "1/length",
    }
}

/// How far the data moved the prior of a rate, and whether it moved it at all.
fn prior(result: &mut ExperimentResult, label: &str, dominance: Option<PriorDominance>) {
    let Some(dominance) = dominance else {
        return;
    };
    metric(
        result,
        format!("{label} posterior/prior width"),
        Some(dominance.width_ratio),
        "1",
    );
    metric(
        result,
        format!("{label} prior shift"),
        Some(dominance.shift_sigma),
        "prior sigma",
    );
    if dominance.dominated {
        result.note(format!("{label} returned its prior"));
    }
}
fn rate(result: &mut ExperimentResult, label: &str, mass: &MassEstimate) {
    let unit = rate_unit(mass.time_unit);
    metric(result, label, Some(mass.value), unit);
    metric(result, format!("{label} error"), Some(mass.error), unit);
    metric(
        result,
        format!("{label} statistical"),
        Some(mass.statistical),
        unit,
    );
    metric(
        result,
        format!("{label} systematic"),
        Some(mass.systematic),
        unit,
    );
    prior(result, label, mass.prior_dominance);
}

/// Rates, diagnostics, window table and notes of one fit.
fn fit(result: &mut ExperimentResult, label: &str, outcome: &FitOutcome) {
    let diagnostics = &outcome.diagnostics;
    if let Some(mass) = &outcome.mass {
        rate(result, &format!("{label} rate"), mass);
    } else if outcome.method != FitMethodKind::Stability {
        metric(result, format!("{label} rate"), None, "");
        prior(result, &format!("{label} gap"), diagnostics.prior_dominance);
    }
    for (n, level) in outcome.excited.iter().enumerate() {
        rate(result, &format!("{label} excited rate {}", n + 1), level);
    }
    metric(result, format!("{label} chi2"), diagnostics.chi2, "1");
    metric(
        result,
        format!("{label} dof"),
        diagnostics.dof.map(|d| d as f64),
        "1",
    );
    metric(result, format!("{label} Q"), diagnostics.q, "1");
    for (edge, at) in ["t_min", "t_max"].iter().zip([0, 1]) {
        metric(
            result,
            format!("{label} window {edge}"),
            diagnostics.window.map(|w| w[at] as f64),
            "frames",
        );
    }
    metric(
        result,
        format!("{label} windows"),
        Some(diagnostics.n_windows as f64),
        "1",
    );
    metric(
        result,
        format!("{label} covariance rank"),
        diagnostics.covariance_rank.map(|r| r as f64),
        "1",
    );
    metric(
        result,
        format!("{label} svd cut"),
        Some(diagnostics.svd_cut),
        "1",
    );
    if !diagnostics.correlated {
        result.note(format!(
            "{label}: a diagonal chi2 replaced the correlated one"
        ));
    }
    for reason in diagnostics
        .model_rejected
        .iter()
        .chain(&diagnostics.no_signal)
    {
        result.note(format!("{label}: no rate: {reason}"));
    }
    for note in &outcome.notes {
        result.note(format!("{label}: {note}"));
    }
    if outcome.windows.is_empty() {
        return;
    }
    let index: Vec<f64> = (0..outcome.windows.len()).map(|i| i as f64).collect();
    let values: Vec<Option<f64>> = outcome.windows.iter().map(|w| Some(w.value)).collect();
    let errors: Vec<Option<f64>> = outcome.windows.iter().map(|w| Some(w.error)).collect();
    result.plot(
        format!("{label}: rate per window"),
        "window (table order)",
        "rate",
        band("rate", &index, &values, &errors),
    );
    let weights = outcome
        .windows
        .iter()
        .zip(&index)
        .map(|(w, i)| Some([*i, w.weight]));
    result.plot(
        format!("{label}: window weights"),
        "window (table order)",
        "weight",
        runs("weight", weights),
    );
}
fn method(kind: FitMethodKind) -> &'static str {
    match kind {
        FitMethodKind::WindowScan => "window scan",
        FitMethodKind::MultiExponential => "multi-exponential",
        FitMethodKind::Gevp => "GEVP",
        FitMethodKind::Stability => "stability scan",
    }
}
fn exchange(parity: Option<ExchangeParity>) -> &'static str {
    match parity {
        Some(ExchangeParity::Even) => "even",
        Some(ExchangeParity::Odd) => "odd",
        Some(ExchangeParity::Mixed) => "mixed",
        None => "not defined",
    }
}
fn spatial(parity: Option<SpatialParity>) -> &'static str {
    match parity {
        Some(SpatialParity::Even) => "even",
        Some(SpatialParity::Odd) => "odd",
        None => "not verified",
    }
}

/// Denominator of the frame averages a rate was fitted on. The two arms are
/// different observables, so `09_qft_calibration` asks every reported rate to
/// state the one it used; a measurement that did not record it says so.
fn normalization(kind: Option<FrameNormalization>) -> &'static str {
    match kind {
        Some(FrameNormalization::ValidCount) => "the sum of valid element weights (04)",
        Some(FrameNormalization::FixedN) => "the population size N (08)",
        None => "not stated by the measurement",
    }
}

/// A channel is titled by its id and described by its algebraic definition
/// and parities; a particle name appears only in the comparison.
fn channel(report: &ChannelReport) -> ExperimentResult {
    let quantity = match report.estimator {
        Some(EstimatorKind::SourceFrozen) => RATE_QUANTITY_SOURCE_FROZEN,
        Some(EstimatorKind::EuclideanTime) => RATE_QUANTITY_EUCLIDEAN,
        Some(EstimatorKind::FrameMean) | None => RATE_QUANTITY,
    };
    let mut result = ExperimentResult::new(EXPERIMENT, report.id.clone(), quantity);
    if !report.definition.is_empty() {
        result.note(format!("Definition: {}", report.definition));
    }
    result.note(format!(
        "Exchange parity: {}; spatial parity: {}.",
        exchange(report.exchange),
        spatial(report.spatial_parity)
    ));
    if matches!(
        report.estimator,
        Some(EstimatorKind::FrameMean | EstimatorKind::EuclideanTime)
    ) {
        result.note(format!(
            "Frame normalisation: {}. The valid-count and fixed-N averages are different \
             observables and only the fixed-N one has a transfer-matrix reading.",
            normalization(report.normalization)
        ));
    }
    if let Some(mass) = &report.mass {
        rate(&mut result, "rate", mass);
    } else {
        metric(&mut result, "rate", None, "");
    }
    metric(
        &mut result,
        "valid elements",
        Some(report.coverage.valid as f64),
        "1",
    );
    metric(
        &mut result,
        "masked elements",
        Some(report.coverage.masked() as f64),
        "1",
    );
    if let Some(correlator) = &report.correlator {
        let x: Vec<f64> = correlator
            .lags
            .iter()
            .map(|lag| *lag as f64 * correlator.time_step)
            .collect();
        let axis = lag_label(correlator.time_unit);
        result.plot(
            "Correlator",
            axis,
            "C",
            band("C", &x, &correlator.value, &correlator.error),
        );
        result.note(format!(
            "Errors are resampling errors over {}.",
            correlator.samples_meta.sampling_unit
        ));
        let meta = &correlator.samples_meta;
        metric(&mut result, "tau_int", meta.tau_int, "frames");
        for (label, count) in [
            ("block length", meta.effective_block),
            ("blocks", meta.blocks),
            ("covariance rank", meta.covariance_rank),
            ("lags", correlator.lags.len()),
            ("replicas", meta.replicas),
        ] {
            metric(&mut result, label, Some(count as f64), "1");
        }
        if correlator.connected {
            metric(
                &mut result,
                "connected bias",
                correlator.connected_bias,
                "C",
            );
        }
        if let Some(effective) = &report.effective_mass {
            let (values, errors) = split(effective);
            result.plot(
                "Effective rate",
                axis,
                "rate",
                band("rate", &x, &values, &errors),
            );
        }
    }
    for outcome in &report.fits {
        fit(&mut result, method(outcome.method), outcome);
    }
    for note in &report.notes {
        result.note(note.clone());
    }
    result
}
fn levels(result: &mut ExperimentResult, levels: &[MassEstimate]) {
    for (n, level) in levels.iter().enumerate() {
        rate(result, &format!("level {n}"), level);
    }
}
fn group(report: &GroupFit) -> ExperimentResult {
    let mut result = ExperimentResult::new(
        EXPERIMENT,
        format!("Group {}", report.id),
        format!(
            "joint fit with shared gaps of {}",
            report.channels.join(", ")
        ),
    );
    levels(&mut result, &report.levels);
    metric(&mut result, "chi2", report.diagnostics.chi2, "1");
    metric(
        &mut result,
        "dof",
        report.diagnostics.dof.map(|d| d as f64),
        "1",
    );
    metric(&mut result, "Q", report.diagnostics.q, "1");
    for note in &report.notes {
        result.note(note.clone());
    }
    result
}
fn gevp(report: &GevpReport) -> ExperimentResult {
    let mut result = ExperimentResult::new(
        EXPERIMENT,
        format!("GEVP {}", report.id),
        format!(
            "generalized eigenvalue problem of {}",
            report.channels.join(", ")
        ),
    );
    metric(&mut result, "rank", Some(report.rank as f64), "1");
    metric(&mut result, "t0", Some(report.t0 as f64), "frames");
    let x: Vec<f64> = report.lags.iter().map(|lag| *lag as f64).collect();
    let curves = |states: &[Vec<Option<[f64; 2]>>]| -> Vec<Series> {
        states
            .iter()
            .enumerate()
            .flat_map(|(n, state)| {
                let (values, errors) = split(state);
                band(&format!("state {n}"), &x, &values, &errors)
            })
            .collect()
    };
    result.plot(
        "Generalized eigenvalues",
        "lag (frames)",
        "eigenvalue",
        curves(&report.eigenvalues),
    );
    result.plot(
        "Effective rates",
        "lag (frames)",
        "rate",
        curves(&report.effective_mass),
    );
    let asymmetry = x
        .iter()
        .zip(&report.antisymmetric_norm)
        .map(|(x, a)| a.map(|a| [*x, a]));
    result.plot(
        "Discarded antisymmetric part",
        "lag (frames)",
        "relative norm",
        runs("antisymmetric norm", asymmetry),
    );
    for (n, outcome) in report.levels.iter().enumerate() {
        fit(&mut result, &format!("state {n}"), outcome);
    }
    for note in &report.notes {
        result.note(note.clone());
    }
    result
}

/// Tables of a comparison. Rows keep the order of the report, which is the
/// order of the reference table.
fn comparison(report: &Comparison) -> ExperimentResult {
    let mut result = ExperimentResult::new(
        EXPERIMENT,
        format!("Reference comparison ({})", report.label),
        "channel-to-reference assignments are inputs of the analysis; tensions carry no \
         look-elsewhere correction",
    );
    for row in &report.reference {
        let label = format!("{} rate [{}]", row.name, row.channel);
        metric(&mut result, label, row.measured.map(|m| m[0]), "lattice");
        metric(
            &mut result,
            format!("{} rate error", row.name),
            row.measured.map(|m| m[1]),
            "lattice",
        );
        metric(
            &mut result,
            format!("{} reference", row.name),
            Some(row.reference),
            &row.unit,
        );
    }
    let unit = report.reference.first().map_or("", |r| r.unit.as_str());
    let index = |n: usize| -> Vec<f64> { (0..n).map(|i| i as f64).collect() };
    let mut measured = vec![];
    let mut expected = vec![];
    for row in &report.ratios {
        let label = format!("{}/{}", row.numerator, row.denominator);
        metric(
            &mut result,
            format!("{label} measured"),
            row.measured.map(|m| m[0]),
            "1",
        );
        metric(
            &mut result,
            format!("{label} error"),
            row.measured.map(|m| m[1]),
            "1",
        );
        metric(
            &mut result,
            format!("{label} reference"),
            Some(row.reference),
            "1",
        );
        metric(
            &mut result,
            format!("{label} tension"),
            row.tension_sigma,
            "sigma",
        );
        measured.push(row.measured);
        expected.push(Some(row.reference));
    }
    if !report.ratios.is_empty() {
        let x = index(report.ratios.len());
        let (values, errors) = split(&measured);
        let mut series = band("measured", &x, &values, &errors);
        series.extend(runs(
            "reference",
            x.iter().zip(&expected).map(|(x, r)| r.map(|r| [*x, r])),
        ));
        result.plot(
            "Ratios of rates",
            "ratio row (table order)",
            "ratio",
            series,
        );
    }
    for anchor in &report.anchors {
        let name = &anchor.anchor;
        let per_lattice = format!("{unit} per lattice unit");
        metric(
            &mut result,
            format!("scale at {name}"),
            anchor.scale.map(|s| s[0]),
            &per_lattice,
        );
        metric(
            &mut result,
            format!("scale error at {name}"),
            anchor.scale.map(|s| s[1]),
            &per_lattice,
        );
        for p in &anchor.predictions {
            let label = format!("{} at {name}", p.name);
            metric(
                &mut result,
                format!("{label} predicted"),
                p.predicted.map(|v| v[0]),
                unit,
            );
            metric(
                &mut result,
                format!("{label} error"),
                p.predicted.map(|v| v[1]),
                unit,
            );
            metric(
                &mut result,
                format!("{label} reference"),
                Some(p.reference),
                unit,
            );
            metric(
                &mut result,
                format!("{label} tension"),
                p.tension_sigma,
                "sigma",
            );
        }
        if anchor.predictions.is_empty() {
            continue;
        }
        let x = index(anchor.predictions.len());
        let predicted: Vec<Option<[f64; 2]>> =
            anchor.predictions.iter().map(|p| p.predicted).collect();
        let (values, errors) = split(&predicted);
        let mut series = band("rescaled rate", &x, &values, &errors);
        let references = x
            .iter()
            .zip(&anchor.predictions)
            .map(|(x, p)| Some([*x, p.reference]));
        series.extend(runs("reference", references));
        result.plot(
            format!("Rates rescaled by the anchor {name} (an input)"),
            "prediction row (table order)",
            unit,
            series,
        );
    }
    for row in &report.anchor_spread {
        metric(
            &mut result,
            format!("{} anchor spread", row.name),
            row.spread,
            "1",
        );
    }
    for note in &report.notes {
        result.note(note.clone());
    }
    result
}

/// One result per kind of number, so that a configured scale, a parameter
/// proxy and an inversion target never share a table.
fn couplings(report: &CouplingReport) -> Vec<ExperimentResult> {
    let table = |title: &str, model: &str, rows: &[Quantity], explain_all: bool| {
        let mut result = ExperimentResult::new(EXPERIMENT, title, model);
        for row in rows {
            metric(&mut result, row.name.clone(), row.value, &row.unit);
            if let Some(error) = row.error {
                metric(
                    &mut result,
                    format!("{} error", row.name),
                    Some(error),
                    &row.unit,
                );
            }
            if explain_all || row.value.is_none() {
                result.note(format!("{}: {}", row.name, row.definition));
            }
        }
        for note in &report.notes {
            result.note(note.clone());
        }
        result
    };
    vec![
        table(
            "Algorithmic scales",
            "configured scales and warm-up calibration; not measurements of a coupling",
            &report.scales,
            false,
        ),
        table(
            "Coupling proxies",
            "book formulas evaluated on chosen parameters; never compared with Standard Model \
             couplings",
            &report.couplings,
            false,
        ),
        table(
            "Calibration inversion",
            "Standard Model inputs mapped to target gas parameters; inputs of a calibration, \
             not results of this run",
            &report.inversion,
            true,
        ),
    ]
}
fn flow(report: &FlowDiagnostic) -> ExperimentResult {
    let mut result = ExperimentResult::new(
        EXPERIMENT,
        "Graph smoothing",
        "mean neighbour colour mismatch after each smoothing step; it defines no length scale",
    );
    metric(&mut result, "frames", Some(report.frames as f64), "1");
    let points = report
        .steps
        .iter()
        .zip(&report.roughness)
        .map(|(step, r)| r.map(|r| [*step as f64, r]));
    result.plot(
        "Roughness",
        "smoothing steps",
        "1 - Re q",
        runs("roughness", points),
    );
    result
}

/// One result per available channel (correlator with error band, effective
/// rate, fit windows) and one for the comparison tables. Unavailable channels
/// contribute a note with their reason and no metric. The report is validated
/// first. Every result carries the provenance
/// `details = {"calculation_origin", "precision", "schema_version", "request"}`
/// of the report, `request` being its `analysis`, and a nonfinite plotted
/// point is an error.
///
/// The first result is the overview: the notes of the report and the reason
/// of every unavailable channel, group and basis. Groups, GEVP bases, the
/// three coupling tables and the smoothing diagnostic follow the channels.
pub fn present(report: &SpectroscopyReport) -> Result<Vec<ExperimentResult>> {
    report.validate()?;
    let mut overview = ExperimentResult::new(
        EXPERIMENT,
        "Spectroscopy report",
        "decay rates of correlators of a recorded run",
    );
    metric(&mut overview, "replicas", Some(report.replicas as f64), "1");
    metric(&mut overview, "frames", Some(report.frames as f64), "1");
    for note in &report.notes {
        overview.note(note.clone());
    }
    let mut results = vec![];
    for item in &report.channels {
        if let Some(reason) = item.availability.reason() {
            overview.note(format!("{}: unavailable: {reason}", item.id));
        } else {
            results.push(channel(item));
        }
    }
    for item in &report.groups {
        if let Some(reason) = item.availability.reason() {
            overview.note(format!("group {}: unavailable: {reason}", item.id));
        } else {
            results.push(group(item));
        }
    }
    for item in &report.gevp {
        if let Some(reason) = item.availability.reason() {
            overview.note(format!("GEVP {}: unavailable: {reason}", item.id));
        } else {
            results.push(gevp(item));
        }
    }
    results.insert(0, overview);
    results.extend(report.comparison.as_ref().map(comparison));
    results.extend(report.couplings.iter().flat_map(couplings));
    results.extend(report.flow.as_ref().map(flow));
    let details = json!({"calculation_origin":report.calculation_origin,"precision":report.precision,"schema_version":report.schema_version,"request":report.analysis});
    for result in &mut results {
        result.details = details.clone();
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn an_undefined_point_splits_a_curve_instead_of_becoming_a_zero() {
        let x = [0., 1., 2., 3., 4.];
        let values = [Some(5.), Some(4.), None, Some(2.), Some(1.)];
        let errors = [Some(0.5), None, None, Some(0.25), Some(0.25)];
        let series = band("C", &x, &values, &errors);
        let shape: Vec<(&str, Vec<[f64; 2]>)> = series
            .iter()
            .map(|s| (s.name.as_str(), s.points.clone()))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("C", vec![[0., 5.], [1., 4.]]),
                ("C", vec![[3., 2.], [4., 1.]]),
                ("C + error", vec![[0., 5.5]]),
                ("C + error", vec![[3., 2.25], [4., 1.25]]),
                ("C - error", vec![[0., 4.5]]),
                ("C - error", vec![[3., 1.75], [4., 0.75]]),
            ]
        );
        assert!(series.iter().all(|s| s.kind == "line"));
        assert!(runs("empty", [None, None]).is_empty());
    }
}
