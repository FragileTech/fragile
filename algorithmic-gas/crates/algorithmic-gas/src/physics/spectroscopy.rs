//! Spectroscopy of recorded gas runs: gauge-theory operators on the companion
//! topology, their time correlators, decay rates and a comparison with
//! reference data.
//!
//! The work is split in two. A measurement streams recorded steps through an
//! `Accumulator` and keeps operator series, propagator moments and coverage; it
//! depends on `MeasurementConfig` only. An analysis turns measurements into a
//! `SpectroscopyReport` and can be repeated with another `AnalysisConfig`
//! without running the gas again.
//!
//! Every variant of the gas can be measured. What a variant does not record
//! makes the dependent channels `Unavailable` with a reason instead of failing
//! the run. The fitted quantity is the decay rate of the algorithm-time
//! autocorrelation; it is a mass only under a positive self-adjoint transfer
//! representation, and an oscillating correlator rejects the exponential model
//! instead of producing a number.
pub mod accumulator;
pub mod config;
pub mod contract;
pub mod couplings;
pub mod estimators;
pub mod fields;
pub mod fits;
pub mod flow;
pub mod frame;
pub mod measurement;
pub mod operators;
pub mod presentation;
pub mod reference;
pub mod report;
pub mod scales;
pub mod topology;
use crate::{GasError, Precision, Result, RunArchive, error::require};
pub use accumulator::{Accumulator, AccumulatorState, Extensions};
pub use config::{AnalysisConfig, ChannelSpec, MeasurementConfig, SpectroscopyConfig};
use config::{Combine, EstimatorChoice};
use contract::EXCHANGE_ODD_REASON;
pub use contract::{
    Availability, Capabilities, Element, ElementKind, ExchangeParity, FieldSource, Frame,
    FrameState, LocalOperator, OperatorContext, Record, Requirements, SPECTROSCOPY_VERSION,
    Signature, Topology,
};
use estimators::Estimated;
pub use measurement::{ChannelSeries, Measurement};
use report::{CALCULATION_ORIGIN, EstimatorKind, FlowDiagnostic, GevpReport, GroupFit};
pub use report::{ChannelReport, SpectroscopyReport};

/// Printed with every report: what the fitted numbers are.
pub const INTERPRETATION_NOTE: &str = "Fitted values are decay rates of the algorithm-time \
    autocorrelation. They are masses only under the positive self-adjoint transfer \
    representation (cor-effective-twistor-positive-transfer); the gas is not reversible, so \
    complex or oscillating modes are expected and reject the exponential model.";

/// Measure one validated archive from its first step.
pub fn measure_archive(
    config: &MeasurementConfig,
    archive: &RunArchive<f64>,
) -> Result<Measurement> {
    measure_archive_with(config, archive, Extensions::default())
}

/// `measure_archive` with injected operators, field source, capabilities or
/// calibration. An archive of an `f32` run is `GasError::Capability`: there is
/// no precision fallback.
pub fn measure_archive_with(
    config: &MeasurementConfig,
    archive: &RunArchive<f64>,
    extensions: Extensions,
) -> Result<Measurement> {
    config.validate()?;
    if archive.gas_config.precision != Precision::F64 {
        return Err(GasError::Capability(
            "spectroscopy requires an f64 recorded run".into(),
        ));
    }
    let anchor = archive
        .anchors
        .first()
        .ok_or_else(|| GasError::Checkpoint("missing archive epoch anchor".into()))?;
    let dimension = anchor
        .population
        .observations
        .field(frame::position_field(&archive.gas_config))?
        .width();
    let mut accumulator = Accumulator::with_extensions(
        config.clone(),
        &archive.gas_config,
        &archive.config,
        dimension,
        extensions,
    )?;
    accumulator.ingest_archive(archive)?;
    accumulator.into_measurement()
}

/// Outcome of an analysis step that may decline: `Capability` and `Numerical`
/// failures make one item unavailable instead of failing the analysis.
enum Soft<T> {
    Value { value: T },
    Declined { reason: String },
}
fn soft<T>(result: Result<T>) -> Result<Soft<T>> {
    match result {
        Ok(value) => Ok(Soft::Value { value }),
        Err(GasError::Capability(reason) | GasError::Numerical(reason)) => {
            Ok(Soft::Declined { reason })
        }
        Err(e) => Err(e),
    }
}

/// The estimator an analysis uses for a channel: `estimator: None` when it
/// has none, `note` then holding the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EstimatorSelection {
    pub estimator: Option<EstimatorKind>,
    pub note: Option<String>,
}
impl EstimatorSelection {
    fn of(estimator: EstimatorKind) -> Self {
        Self {
            estimator: Some(estimator),
            note: None,
        }
    }
    fn none(reason: impl Into<String>) -> Self {
        Self {
            estimator: None,
            note: Some(reason.into()),
        }
    }
}

/// The frame mean of an exchange-odd pair operator runs over unmirrored
/// elements only (`ChannelSeries::values`), so it does not exist when no frame
/// of any replica has one; its source-frozen propagator remains. An
/// exchange-odd triplet operator has a frame mean of zero expectation under
/// exchangeable companion roles, so `Auto` prefers its propagator when one was
/// measured. Live views call this with `EstimatorChoice::Auto`.
pub fn select_estimator(
    measurements: &[Measurement],
    series: &ChannelSeries,
    choice: EstimatorChoice,
) -> EstimatorSelection {
    let members: Vec<&ChannelSeries> = measurements
        .iter()
        .filter_map(|m| m.channel(&series.id))
        .collect();
    let odd = series.exchange == ExchangeParity::Odd;
    let measured = members.iter().any(|c| !c.weight.is_empty());
    let cancels = odd && measured && members.iter().all(|c| c.weight.iter().all(|w| *w == 0.));
    let all = |present: fn(&ChannelSeries) -> bool| {
        !members.is_empty() && members.iter().all(|c| present(c))
    };
    let propagator = all(|c| c.propagator.is_some());
    let euclidean = all(|c| c.euclidean.is_some());
    let frozen = |reason: &str| EstimatorSelection {
        estimator: Some(EstimatorKind::SourceFrozen),
        note: Some(format!(
            "{reason}: the source-frozen propagator is reported instead of the frame mean"
        )),
    };
    match choice {
        EstimatorChoice::Auto if cancels && propagator => frozen(EXCHANGE_ODD_REASON),
        EstimatorChoice::Auto if odd && series.kind == ElementKind::Triplet && propagator => {
            frozen("relabelling-odd triplet operator with a frame mean of zero expectation")
        }
        EstimatorChoice::Auto | EstimatorChoice::FrameMean if cancels => {
            EstimatorSelection::none(EXCHANGE_ODD_REASON)
        }
        EstimatorChoice::Auto | EstimatorChoice::FrameMean => {
            EstimatorSelection::of(EstimatorKind::FrameMean)
        }
        EstimatorChoice::SourceFrozen if propagator => {
            EstimatorSelection::of(EstimatorKind::SourceFrozen)
        }
        EstimatorChoice::SourceFrozen => {
            EstimatorSelection::none("no source-frozen propagator was measured for this channel")
        }
        EstimatorChoice::EuclideanTime if euclidean => {
            EstimatorSelection::of(EstimatorKind::EuclideanTime)
        }
        EstimatorChoice::EuclideanTime => {
            EstimatorSelection::none("no Euclidean-time correlator was measured for this channel")
        }
    }
}
fn channel_report(
    measurements: &[Measurement],
    series: &ChannelSeries,
    analysis: &AnalysisConfig,
) -> Result<(ChannelReport, Option<Estimated>)> {
    let mut report = ChannelReport {
        id: series.id.clone(),
        spec: series.spec.clone(),
        kind: Some(series.kind),
        scale: series.scale,
        definition: series.definition.clone(),
        book_label: series.book_label.clone(),
        exchange: Some(series.exchange),
        spatial_parity: series.spatial_parity,
        ..ChannelReport::default()
    };
    if !series.note.is_empty() {
        report.notes.push(series.note.clone());
    }
    // `B/09:334-337`: every reported rate states the denominator of the frame
    // averages behind it, the `Signature` override when the operator fixes one
    // and `MeasurementConfig::normalization` otherwise.
    report.normalization = measurements.first().map(|measurement| {
        let context = OperatorContext {
            gas: &measurement.gas,
            measurement: &measurement.config,
            capabilities: &measurement.capabilities,
        };
        series
            .spec
            .signature(series.kind, &context)
            .ok()
            .and_then(|signature| signature.normalization)
            .unwrap_or(measurement.config.normalization)
    });
    for member in measurements.iter().filter_map(|m| m.channel(&series.id)) {
        report.coverage.merge(&member.coverage);
        report.availability = report.availability.and(member.availability.clone());
    }
    if !report.availability.is_available() {
        return Ok((report, None));
    }
    if !series.correlatable {
        report
            .notes
            .push("diagnostic envelope: it never enters a correlator".into());
        return Ok((report, None));
    }
    let selection = select_estimator(measurements, series, analysis.estimator);
    let Some(kind) = selection.estimator else {
        report.availability = Availability::unavailable(selection.note.unwrap_or_default());
        return Ok((report, None));
    };
    report.notes.extend(selection.note);
    report.estimator = Some(kind);
    let data = match soft(estimators::estimate(
        measurements,
        &series.id,
        kind,
        analysis,
        None,
    ))? {
        Soft::Value { value } => value,
        Soft::Declined { reason } => {
            report.availability = Availability::unavailable(reason);
            return Ok((report, None));
        }
    };
    // What the resampling behind this estimate does not cover travels with the
    // channel: the delete-one geometry removes origins and not observations,
    // so beyond the effective block the error is a lower bound; an automatic
    // block may stop below the batch size its autocorrelation time asks; and a
    // table of fewer replicas than estimated lags leaves the lag covariance
    // rank deficient. A fit that scans a shorter window restates the last of
    // the three for its own point count, in its own notes.
    report.notes.extend(data.notes());
    report.effective_mass = Some(fits::effective_mass::effective_mass(
        &data,
        analysis.effective_mass,
    ));
    match soft(fits::fit_channel(&data, analysis))? {
        Soft::Value { value } => {
            report.mass = value.iter().find_map(|f| f.mass.clone());
            report.fits = value;
        }
        Soft::Declined { reason } => report.notes.push(format!("no fit: {reason}")),
    }
    let mut estimate = data.estimate.clone();
    if !analysis.report_covariance {
        estimate.covariance = None;
    }
    report.correlator = Some(estimate);
    Ok((report, Some(data)))
}

/// Channel ids selected by keys that name a channel or a specification, in
/// report order, restricted to channels that produced an estimate.
fn resolve(keys: &[String], analyzed: &[(ChannelReport, Option<Estimated>)]) -> Vec<String> {
    analyzed
        .iter()
        .filter(|(report, data)| {
            data.is_some()
                && keys
                    .iter()
                    .any(|k| *k == report.id || *k == report.spec.id())
        })
        .map(|(report, _)| report.id.clone())
        .collect()
}
fn group_fit(
    measurements: &[Measurement],
    group: &config::ChannelGroup,
    analyzed: &[(ChannelReport, Option<Estimated>)],
    analysis: &AnalysisConfig,
) -> Result<GroupFit> {
    let members: Vec<(String, EstimatorKind)> = resolve(&group.channels, analyzed)
        .into_iter()
        .filter_map(|id| {
            let kind = analyzed.iter().find(|(r, _)| r.id == id)?.0.estimator?;
            Some((id, kind))
        })
        .collect();
    let unavailable = |reason: String| GroupFit {
        id: group.id.clone(),
        channels: members.iter().map(|(id, _)| id.clone()).collect(),
        availability: Availability::unavailable(reason),
        ..GroupFit::default()
    };
    if members.len() < 2 {
        return Ok(unavailable(
            "fewer than two channels of the group have a correlator".into(),
        ));
    }
    let fit = estimators::joint(measurements, &members, analysis)
        .and_then(|joint| fits::fit_group(group, &joint, analysis));
    Ok(match soft(fit)? {
        Soft::Value { value } => value,
        Soft::Declined { reason } => unavailable(reason),
    })
}
fn gevp_report(
    measurements: &[Measurement],
    basis: &config::GevpBasis,
    analyzed: &[(ChannelReport, Option<Estimated>)],
    analysis: &AnalysisConfig,
) -> Result<GevpReport> {
    // The matrix correlator is a frame-mean object: exchange-odd members that
    // fell back to the propagator cannot enter it.
    let channels: Vec<String> = resolve(&basis.channels, analyzed)
        .into_iter()
        .filter(|id| {
            analyzed
                .iter()
                .any(|(r, _)| r.id == *id && r.estimator == Some(EstimatorKind::FrameMean))
        })
        .collect();
    let unavailable = |reason: String| GevpReport {
        id: basis.id.clone(),
        channels: channels.clone(),
        availability: Availability::unavailable(reason),
        t0: basis.t0,
        ..GevpReport::default()
    };
    if channels.len() < 2 {
        return Ok(unavailable(
            "fewer than two channels of the basis have a frame-mean correlator".into(),
        ));
    }
    let report = estimators::matrix(measurements, &channels, analysis)
        .and_then(|matrix| fits::gevp::gevp(&matrix, basis, analysis));
    Ok(match soft(report)? {
        Soft::Value { value } => value,
        Soft::Declined { reason } => unavailable(reason),
    })
}

/// Frame-weighted mean of the replicas' smoothing curves.
fn merge_flow(measurements: &[Measurement]) -> Option<FlowDiagnostic> {
    let mut parts = measurements.iter().filter_map(|m| m.flow.as_ref());
    let mut total = parts.next()?.clone();
    let mut weights: Vec<f64> = total
        .roughness
        .iter()
        .map(|r| r.map_or(0., |_| total.frames as f64))
        .collect();
    for value in &mut total.roughness {
        *value = value.map(|v| v * total.frames as f64);
    }
    for part in parts.filter(|p| p.steps == total.steps) {
        total.frames += part.frames;
        for ((sum, weight), value) in total
            .roughness
            .iter_mut()
            .zip(&mut weights)
            .zip(&part.roughness)
        {
            if let Some(v) = value {
                *sum = Some(sum.unwrap_or(0.) + v * part.frames as f64);
                *weight += part.frames as f64;
            }
        }
    }
    for (value, weight) in total.roughness.iter_mut().zip(&weights) {
        *value = value.filter(|_| *weight > 0.).map(|v| v / weight);
    }
    Some(total)
}

/// Analyse measurements of independent replicas of one configuration:
/// estimators → resampling → fits → groups → GEVP → comparison → report.
/// Pure and deterministic. A channel that cannot be analysed is reported
/// `Unavailable` with its reason; only malformed input is an error, and so is
/// a nonfinite number in the result (`SpectroscopyReport::validate`). Errors
/// are resampling errors whose unit every `SamplesMeta::sampling_unit` states;
/// with fewer than 4 replicas a note says that they are not replica standard
/// errors.
pub fn analyze(
    measurements: &[Measurement],
    analysis: &AnalysisConfig,
) -> Result<SpectroscopyReport> {
    analysis.validate()?;
    let first = measurements
        .first()
        .ok_or_else(|| GasError::Configuration("analysis requires a measurement".into()))?;
    for measurement in measurements {
        measurement.validate()?;
    }
    require(
        measurements.iter().all(|m| {
            m.schema_version == SPECTROSCOPY_VERSION
                && m.fingerprint == first.fingerprint
                && m.capabilities == first.capabilities
                && m.walkers == first.walkers
                && m.channels.len() == first.channels.len()
                && m.channels
                    .iter()
                    .zip(&first.channels)
                    .all(|(a, b)| a.id == b.id)
        }),
        "measurements of different configurations, population sizes or schema versions cannot \
         be combined",
    )?;
    require(
        analysis
            .max_lag()
            .is_none_or(|lag| lag <= first.config.max_lag),
        "analysis windows exceed the measured lag range",
    )?;
    require(
        analysis.combine != Combine::RunsAsSamples || measurements.len() >= 8,
        "runs-as-samples requires at least 8 replicas",
    )?;
    let mut notes = vec![INTERPRETATION_NOTE.to_string()];
    if measurements.len() < 4 {
        notes.push(
            "errors are block-resampling errors inside fewer than 4 independently seeded runs; \
             they are not replica standard errors"
                .into(),
        );
    }
    for measurement in measurements {
        for note in &measurement.notes {
            if !notes.contains(note) {
                notes.push(note.clone());
            }
        }
    }
    if measurements
        .iter()
        .any(|m| m.calibration != first.calibration)
    {
        notes.push(
            "warm-up calibration differs between replicas; the first replica's is reported".into(),
        );
    }
    let mut analyzed = vec![];
    for series in &first.channels {
        if analysis.selects(&series.spec.id(), &series.id) {
            analyzed.push(channel_report(measurements, series, analysis)?);
        }
    }
    if !analyzed.iter().any(|(_, data)| data.is_some()) {
        notes.push("no selected channel produced a correlator".into());
    }
    let groups = analysis
        .groups
        .iter()
        .map(|group| group_fit(measurements, group, &analyzed, analysis))
        .collect::<Result<Vec<_>>>()?;
    let gevp = analysis
        .gevp
        .iter()
        .map(|basis| gevp_report(measurements, basis, &analyzed, analysis))
        .collect::<Result<Vec<_>>>()?;
    let channels: Vec<ChannelReport> = analyzed.into_iter().map(|(report, _)| report).collect();
    let comparison = if analysis.assignments.is_empty() {
        None
    } else {
        match soft(reference::compare(&channels, analysis))? {
            Soft::Value { value } => value,
            Soft::Declined { reason } => {
                notes.push(format!("no reference comparison: {reason}"));
                None
            }
        }
    };
    let couplings = match soft(couplings::report(first, &analysis.standard_model))? {
        Soft::Value { value } => Some(value),
        Soft::Declined { reason } => {
            notes.push(format!("no coupling report: {reason}"));
            None
        }
    };
    let report = SpectroscopyReport {
        schema_version: SPECTROSCOPY_VERSION,
        calculation_origin: CALCULATION_ORIGIN.into(),
        precision: Precision::F64,
        measurement_fingerprint: first.fingerprint.clone(),
        analysis: analysis.clone(),
        capabilities: first.capabilities.clone(),
        calibration: first.calibration.clone(),
        replicas: measurements.len(),
        frames: measurements.iter().map(|m| m.frames() as u64).sum(),
        channels,
        groups,
        gevp,
        comparison,
        couplings,
        flow: merge_flow(measurements),
        notes,
    };
    report.validate()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GasConfig, RecordingConfig};
    use config::{MesonMode, MesonQuantum, TensorMode};
    use contract::channel_id;
    fn series(spec: ChannelSpec, exchange: ExchangeParity, frames: usize) -> ChannelSeries {
        ChannelSeries {
            id: channel_id(&spec.id(), ElementKind::DistancePair),
            spec,
            kind: ElementKind::DistancePair,
            scale: None,
            definition: String::new(),
            book_label: String::new(),
            note: String::new(),
            exchange,
            spatial_parity: None,
            correlatable: true,
            components: 1,
            availability: Availability::Available,
            coverage: report::Coverage {
                frames: frames as u64,
                valid: 8 * frames as u64,
                ..report::Coverage::default()
            },
            involutive_frames: frames as u64,
            values: vec![0.5; frames],
            // Every pair of a mutual pairing is mirrored: an odd operator has
            // no unmirrored element and hence no frame mean.
            weight: vec![f64::from(u8::from(exchange != ExchangeParity::Odd)); frames],
            propagator: None,
            euclidean: None,
        }
    }
    fn measurement(frames: usize) -> Measurement {
        let gas = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
        let config = MeasurementConfig::default();
        let meson = |quantum| ChannelSpec::Meson {
            quantum,
            mode: MesonMode::Standard,
        };
        let mut envelope = series(
            ChannelSpec::Tensor {
                mode: TensorMode::Envelope,
            },
            ExchangeParity::Even,
            frames,
        );
        envelope.correlatable = false;
        let mut missing = series(ChannelSpec::FitnessPhase, ExchangeParity::Even, frames);
        missing.availability = Availability::unavailable("not recorded");
        let capabilities = Capabilities::of(&gas, &RecordingConfig::default(), 3).refine(&config);
        Measurement {
            schema_version: SPECTROSCOPY_VERSION,
            fingerprint: config.fingerprint(&gas, &capabilities, &[]).unwrap(),
            capabilities,
            config,
            gas,
            walkers: 32,
            calibration: None,
            steps: (1..=frames as u64).collect(),
            segment: vec![0; frames],
            segments: 1,
            ingested: frames as u64,
            channels: vec![
                series(meson(MesonQuantum::Scalar), ExchangeParity::Even, frames),
                series(
                    meson(MesonQuantum::Pseudoscalar),
                    ExchangeParity::Odd,
                    frames,
                ),
                envelope,
                missing,
            ],
            flow: None,
            notes: vec![],
        }
    }
    #[test]
    fn exchange_odd_frame_means_on_mutual_pairings_are_unavailable_not_fitted() {
        let report = analyze(&[measurement(64)], &AnalysisConfig::default()).unwrap();
        assert_eq!(report.channels.len(), 4);
        let odd = report
            .channel("meson/pseudoscalar/standard/distance")
            .unwrap();
        assert_eq!(odd.availability.reason(), Some(EXCHANGE_ODD_REASON));
        assert!(odd.correlator.is_none() && odd.mass.is_none() && odd.estimator.is_none());
        let explicit = AnalysisConfig {
            estimator: EstimatorChoice::SourceFrozen,
            ..AnalysisConfig::default()
        };
        let report = analyze(&[measurement(64)], &explicit).unwrap();
        let odd = report
            .channel("meson/pseudoscalar/standard/distance")
            .unwrap();
        assert!(
            odd.availability
                .reason()
                .unwrap()
                .contains("no source-frozen propagator")
        );
    }
    #[test]
    fn envelopes_and_unrecorded_channels_are_reported_without_a_correlator() {
        let report = analyze(&[measurement(64)], &AnalysisConfig::default()).unwrap();
        let envelope = report.channel("tensor/envelope/distance").unwrap();
        assert!(envelope.availability.is_available() && envelope.correlator.is_none());
        assert!(envelope.notes[0].contains("never enters a correlator"));
        let missing = report.channel("fitness_phase/distance").unwrap();
        assert_eq!(missing.availability.reason(), Some("not recorded"));
        assert_eq!(report.notes[0], INTERPRETATION_NOTE);
        assert!(report.notes[1].contains("not replica standard errors"));
        let four: Vec<Measurement> = (0..4).map(|_| measurement(64)).collect();
        let pooled = analyze(&four, &AnalysisConfig::default()).unwrap();
        assert!(pooled.notes.iter().all(|n| !n.contains("replica standard")));
        assert_eq!((report.replicas, report.frames), (1, 64));
        assert_eq!(
            (report.calculation_origin.as_str(), report.precision),
            (CALCULATION_ORIGIN, Precision::F64)
        );
        let text = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<SpectroscopyReport>(&text).unwrap(),
            report
        );
    }
    #[test]
    fn analysis_selects_channels_and_rejects_mismatched_or_missing_measurements() {
        let selected = AnalysisConfig {
            channels: vec!["meson/scalar/standard".into()],
            ..AnalysisConfig::default()
        };
        let report = analyze(&[measurement(64)], &selected).unwrap();
        assert_eq!(report.channels.len(), 1);
        assert!(analyze(&[], &AnalysisConfig::default()).is_err());
        let mut other = measurement(64);
        other.fingerprint.push('x');
        assert!(analyze(&[measurement(64), other], &AnalysisConfig::default()).is_err());
        let mut larger = measurement(64);
        larger.walkers = 64;
        assert!(analyze(&[measurement(64), larger], &AnalysisConfig::default()).is_err());
        let runs = AnalysisConfig {
            combine: Combine::RunsAsSamples,
            ..AnalysisConfig::default()
        };
        assert!(analyze(&[measurement(64)], &runs).is_err());
    }
    #[test]
    fn one_unmirrored_frame_keeps_the_exchange_odd_frame_mean() {
        let mut m = measurement(8);
        let cancelled = select_estimator(
            std::slice::from_ref(&m),
            &m.channels[1],
            EstimatorChoice::Auto,
        );
        assert_eq!(cancelled, EstimatorSelection::none(EXCHANGE_ODD_REASON));
        m.channels[1].weight[3] = 1.;
        let kept = select_estimator(
            std::slice::from_ref(&m),
            &m.channels[1],
            EstimatorChoice::Auto,
        );
        assert_eq!(kept, EstimatorSelection::of(EstimatorKind::FrameMean));
    }
    #[test]
    fn injected_channels_are_selected_by_their_custom_specification() {
        let mut m = measurement(8);
        let mut custom = series(
            ChannelSpec::Custom { id: "probe".into() },
            ExchangeParity::Even,
            8,
        );
        custom.note = "injected".into();
        m.channels.push(custom);
        let selected = AnalysisConfig {
            channels: vec!["custom/probe".into()],
            ..AnalysisConfig::default()
        };
        let report = analyze(&[m], &selected).unwrap();
        assert_eq!(report.channels.len(), 1);
        assert_eq!(report.channels[0].id, "custom/probe/distance");
        assert_eq!(report.channels[0].notes[0], "injected");
    }
    #[test]
    fn smoothing_curves_merge_as_frame_weighted_means() {
        let mut a = measurement(8);
        let mut b = measurement(8);
        a.flow = Some(FlowDiagnostic {
            frames: 1,
            steps: vec![0, 1],
            roughness: vec![Some(1.), None],
        });
        b.flow = Some(FlowDiagnostic {
            frames: 3,
            steps: vec![0, 1],
            roughness: vec![Some(0.), Some(0.5)],
        });
        let merged = merge_flow(&[a, b]).unwrap();
        assert_eq!(merged.frames, 4);
        assert_eq!(merged.roughness, vec![Some(0.25), Some(0.5)]);
    }
    #[test]
    fn a_relabelling_odd_triplet_prefers_its_propagator_and_tampered_series_are_rejected() {
        let mut m = measurement(8);
        let mut baryon = series(
            ChannelSpec::Baryon {
                mode: config::BaryonMode::Complex,
                flux_alpha: 1.,
            },
            ExchangeParity::Even,
            8,
        );
        baryon.exchange = ExchangeParity::Odd;
        baryon.kind = ElementKind::Triplet;
        m.channels.push(baryon);
        let select = |m: &Measurement| {
            select_estimator(
                std::slice::from_ref(m),
                &m.channels[4],
                EstimatorChoice::Auto,
            )
        };
        assert_eq!(select(&m).estimator, Some(EstimatorKind::FrameMean));
        m.channels[4].propagator = Some(crate::physics::numerics::BlockMoments {
            lags: 1,
            components: 1,
            origins_per_block: 1,
            max_blocks: 4,
            blocks: 0,
            open: 0,
            ab: vec![],
            a: vec![],
            b: vec![],
            n: vec![],
        });
        let frozen = select(&m);
        assert_eq!(frozen.estimator, Some(EstimatorKind::SourceFrozen));
        assert!(frozen.note.unwrap().contains("relabelling-odd"));
        m.validate().unwrap();
        m.channels[0].weight.pop();
        assert!(matches!(m.validate(), Err(GasError::Configuration(_))));
        let mut versioned = measurement(8);
        versioned.schema_version += 1;
        assert!(matches!(
            analyze(&[versioned], &AnalysisConfig::default()),
            Err(GasError::Checkpoint(_))
        ));
    }
}
