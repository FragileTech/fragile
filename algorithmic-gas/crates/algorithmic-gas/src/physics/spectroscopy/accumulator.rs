//! Streaming measurement. The caller records short chunks with the engine's
//! `start_recording`/`stop_recording`, ingests each archive and drops it; the
//! archive path calls the same methods. One 64-step archive and eight 8-step
//! chunks give bit-identical measurements, and so does a checkpoint and
//! restore in the middle of a stream.
use super::{
    config::MeasurementConfig,
    contract::{Capabilities, ElementKind, FieldSource, LocalOperator, SPECTROSCOPY_VERSION},
    measurement::Measurement,
    operators::Injected,
    report::Calibration,
};
use crate::{GasConfig, GasError, RecordingConfig, Result, RunArchive, tracking::RecordedStep};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// User-supplied components, set like the slots of `GasBuilder`. `Default`
/// injects nothing.
#[derive(Clone, Default)]
pub struct Extensions {
    operators: Vec<Injected>,
    field: Option<Arc<dyn FieldSource>>,
    capabilities: Option<Capabilities>,
    calibration: Option<Calibration>,
}
impl Extensions {
    /// Measured on `kind` after the configured channels, in injection order.
    /// `operator.id()` is `custom/<name>` with a `<name>` that
    /// `ChannelSpec::Custom` validates, unique per kind; its series carries
    /// `ChannelSpec::Custom { id: <name> }`. Requirements are checked like
    /// those of a built-in operator; propagators, multiscale copies and
    /// analysis keys select it by `custom/<name>` or its channel id.
    pub fn operator(mut self, kind: ElementKind, operator: impl LocalOperator + 'static) -> Self {
        self.operators.push(Injected {
            kind,
            operator: Arc::new(operator),
        });
        self
    }
    /// Replaces the configured `ColorSource`. An injected source provides
    /// `Record::Color` whatever the gas records.
    pub fn field(mut self, field: impl FieldSource + 'static) -> Self {
        self.field = Some(Arc::new(field));
        self
    }
    /// Replaces `Capabilities::of(gas, recording, dimension).refine(&config)`
    /// for a run whose custom `GasOperators` record more or less than the
    /// configuration tells. `Accumulator::with_extensions` calls
    /// `Capabilities::validate` on it before use.
    pub fn capabilities(mut self, capabilities: Capabilities) -> Self {
        self.capabilities = Some(capabilities);
        self
    }
    /// Scales calibrated elsewhere, normally by replica 0 of the same
    /// configuration. The warm-up frames are still skipped and the
    /// fingerprint is unchanged, so pooled replicas measure one observable.
    /// `Accumulator::with_extensions` calls `Calibration::validate` on it.
    pub fn calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = Some(calibration);
        self
    }
    /// What the measurement fingerprint hashes beyond the configuration: the
    /// sorted channel ids of the injected operators, then `field:<id>`.
    pub fn identity(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .operators
            .iter()
            .map(|o| super::contract::channel_id(&o.operator.id(), o.kind))
            .collect();
        ids.sort();
        ids.extend(self.field.iter().map(|f| format!("field:{}", f.id())));
        ids
    }
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
            && self.field.is_none()
            && self.capabilities.is_none()
            && self.calibration.is_none()
    }
}

/// Checkpoint of an `Accumulator`. Only the accumulator constructs it: besides
/// the measurement it holds the warm-up calibration samples, the retained
/// previous frame, the `max_lag + 1` ring of source frames and the continuity
/// guard. Injected components are not serialised. The bytes come from a
/// browser: `restore_with` validates every part before any operator reads it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccumulatorState {
    pub schema_version: u32,
    pub measurement: Measurement,
}
pub struct Accumulator {
    measurement: Measurement,
    extensions: Extensions,
}
impl Accumulator {
    /// `with_extensions` without extensions.
    pub fn new(
        config: MeasurementConfig,
        gas: &GasConfig,
        recording: &RecordingConfig,
        dimension: usize,
    ) -> Result<Self> {
        Self::with_extensions(config, gas, recording, dimension, Extensions::default())
    }
    /// Validates `config` and the capabilities and calibration of
    /// `extensions`, resolves the capabilities, the channels of
    /// `operators::channels` and the fingerprint `config.fingerprint(gas,
    /// &capabilities, &extensions.identity())`. `GasError::Capability` when no
    /// channel is available, when the field source lacks a requirement or
    /// when the retained series cannot fit `budget.max_bytes`;
    /// `GasError::Configuration` when the channels, injected ones included,
    /// are not `1..=256`.
    pub fn with_extensions(
        config: MeasurementConfig,
        gas: &GasConfig,
        recording: &RecordingConfig,
        dimension: usize,
        extensions: Extensions,
    ) -> Result<Self> {
        let _ = (config, gas, recording, dimension, extensions);
        Err(GasError::Capability(
            "pending: spectroscopy::accumulator".into(),
        ))
    }
    /// Ingest every step of a validated archive, in order. The archive's gas
    /// configuration must equal the measured one up to the seed; an archive
    /// of an `f32` run is `GasError::Capability`.
    pub fn ingest_archive(&mut self, archive: &RunArchive<f64>) -> Result<()> {
        for step in &archive.steps {
            self.ingest_step(step)?;
        }
        Ok(())
    }
    /// One recorded step. A step that does not continue the previous one
    /// (gap, epoch change, generation mismatch) clears the ring and the
    /// retained previous frame and opens a new segment; lags never span it.
    /// The frame of every recorded step is retained as the `previous` of the
    /// next one, also for strided-over and warm-up steps. Bytes are admitted
    /// through `memory::enforce` against `budget.max_bytes`
    /// (`GasError::Capability`); reaching `budget.max_frames` is
    /// `GasError::Configuration`, like the recording horizon of an archive.
    /// A failed step leaves the accumulator as it was.
    pub fn ingest_step(&mut self, step: &RecordedStep<f64>) -> Result<()> {
        let _ = (step, &self.extensions);
        Err(GasError::Capability(
            "pending: spectroscopy::accumulator".into(),
        ))
    }
    /// The measurement so far. Propagator origins younger than `max_lag`
    /// frames are still in the ring and not yet in its moments.
    pub fn measurement(&self) -> &Measurement {
        &self.measurement
    }
    /// A copy with the pending propagator origins flushed: what
    /// `into_measurement` would return now.
    pub fn finalized(&self) -> Result<Measurement> {
        Ok(self.measurement.clone())
    }
    pub fn into_measurement(self) -> Result<Measurement> {
        self.finalized()
    }
    pub fn state(&self) -> AccumulatorState {
        AccumulatorState {
            schema_version: SPECTROSCOPY_VERSION,
            measurement: self.measurement.clone(),
        }
    }
    /// `restore_with` without extensions.
    pub fn restore(state: AccumulatorState) -> Result<Self> {
        Self::restore_with(state, Extensions::default())
    }
    /// Takes gas, configuration and capabilities from `state.measurement` and
    /// the injected components from `extensions`, whose `identity()` must
    /// reproduce the stored fingerprint (`GasError::Checkpoint` otherwise, as
    /// for an unsupported `schema_version`). The capabilities and calibration
    /// of `extensions` are ignored: the state holds both. Before use it calls
    /// `Measurement::validate`, `Frame::validate` on every retained frame and
    /// `Topology::validate(frame.n)` on every retained topology, so a
    /// malformed state is an error and never a panic.
    pub fn restore_with(state: AccumulatorState, extensions: Extensions) -> Result<Self> {
        let _ = (state, extensions);
        Err(GasError::Capability(
            "pending: spectroscopy::accumulator".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::spectroscopy::contract::{
        Element, ExchangeParity, FrameState, OperatorContext, Requirements, Signature,
    };
    struct Speed;
    impl LocalOperator for Speed {
        fn id(&self) -> String {
            "custom/speed".into()
        }
        fn signature(&self, _: ElementKind, _: &OperatorContext<'_>) -> Result<Signature> {
            Ok(Signature::new(
                Requirements::default(),
                1,
                ExchangeParity::Even,
            ))
        }
        fn evaluate(
            &self,
            element: &Element,
            state: &FrameState,
            _: &OperatorContext<'_>,
            out: &mut [f64],
        ) -> bool {
            out[0] = state.x[element.walkers[0] as usize * state.d];
            true
        }
    }
    #[test]
    fn injected_components_have_a_sorted_identity() {
        assert!(Extensions::default().is_empty());
        assert!(Extensions::default().identity().is_empty());
        let extensions = Extensions::default()
            .operator(ElementKind::Triplet, Speed)
            .operator(ElementKind::Site, Speed);
        assert!(!extensions.is_empty());
        assert_eq!(
            extensions.identity(),
            vec![
                "custom/speed/site".to_string(),
                "custom/speed/triplet".to_string()
            ]
        );
    }
}
