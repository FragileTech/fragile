//! Streaming measurement. The caller records short chunks with the engine's
//! `start_recording`/`stop_recording`, ingests each archive and drops it; the
//! archive path calls the same methods. One 64-step archive and eight 8-step
//! chunks give bit-identical measurements, and so does a checkpoint and
//! restore in the middle of a stream.
use super::{
    config::{
        FrameNormalization, IdentityPolicy, LengthScale, MeasurementConfig, Range, ScaleMode,
        ScaleSelection, TimeAxis,
    },
    contract::{
        Auxiliary, Availability, Capabilities, Element, ElementKind, ExchangeParity, FieldSource,
        Frame, FrameState, LocalOperator, OperatorContext, Record, Requirements,
        SPECTROSCOPY_VERSION, Signature, Topology, periodic_box,
    },
    fields::{color, score},
    flow, frame,
    measurement::{ChannelSeries, Measurement, SlabMoments},
    operators::{self, Channel, Injected},
    report::{Calibration, Coverage, FlowDiagnostic},
    scales, topology,
};
use crate::{
    GasConfig, GasError, RecordingConfig, Result, RunArchive,
    error::require,
    memory::{checked_add, checked_mul, enforce},
    physics::{numerics::BlockMoments, qft::math::C},
    tessellation::Parallelism,
    tracking::RecordedStep,
};
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

/// Samples pooled over the warm-up frames and frozen into `Calibration` when
/// the first measured frame arrives. Every frame contributes at most a fixed
/// share, so the samples of a run do not depend on how it was chunked.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Warmup {
    frames: usize,
    /// Companion distances or graph edge lengths, as `LengthScale` asks.
    length: Vec<f64>,
    /// Phase-velocity components of the rows a colour record covered.
    velocity: Vec<f64>,
    /// Pair distances of the recorded graph, for the scale ladder.
    scale: Vec<f64>,
    euclidean: Option<[f64; 2]>,
    /// Running `[sum, count]` of the two calibration pair statistics.
    pair_weight: [f64; 2],
    kernel: [f64; 2],
}
impl Warmup {
    fn buffer_bytes(&self) -> Result<usize> {
        let cells = [&self.length, &self.velocity, &self.scale]
            .iter()
            .try_fold(0, |sum, v| checked_add(sum, v.len()))?;
        checked_add(checked_mul(cells, 8)?, 128)
    }
}

/// The step a segment continues from: its number, its epoch and the
/// fingerprint the next frame of the same worldline carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Continuity {
    step: u64,
    epoch: u64,
    fingerprint: u64,
}

/// Elements frozen at one source time, shared by the channels of one element
/// kind and multiscale copy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceGroup {
    elements: Vec<Element>,
}

/// One propagator channel at one source time: the frozen operator values and
/// the lag sums filled while the frame sits in the ring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceChannel {
    /// Index into the measured channels of the plan.
    channel: usize,
    /// Index into `Measurement::channels`.
    series: usize,
    group: usize,
    /// `[elements, components]` and `[elements]`.
    values: Vec<f64>,
    valid: Vec<bool>,
    /// `[elements]` score factors of `Auxiliary::frozen`, frozen at this
    /// source time and reapplied at every sink; empty for every other channel.
    #[serde(default)]
    factor: Vec<f64>,
    /// `[lags]`, `[lags, components]`, `[lags, components]`, `[lags]`.
    ab: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    n: Vec<f64>,
}

/// A measured frame whose propagator lags are not all filled yet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceFrame {
    /// Index of the frame among the measured ones: a lag is a difference.
    index: u64,
    groups: Vec<SourceGroup>,
    channels: Vec<SourceChannel>,
}
impl SourceFrame {
    fn buffer_bytes(&self) -> Result<usize> {
        let mut bytes = 64;
        for group in &self.groups {
            bytes = checked_add(bytes, checked_mul(group.elements.len(), 48)?)?;
        }
        for channel in &self.channels {
            let cells = [
                &channel.values,
                &channel.factor,
                &channel.ab,
                &channel.a,
                &channel.b,
            ]
            .iter()
            .try_fold(channel.n.len(), |sum, v| checked_add(sum, v.len()))?;
            bytes = checked_add(bytes, checked_mul(cells, 8)?)?;
            bytes = checked_add(bytes, channel.valid.len())?;
        }
        Ok(bytes)
    }
}

/// Everything a stream carries from one recorded step to the next.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Stream {
    warmup: Warmup,
    /// The calibration is frozen; before that `kappa` is 0 and no frame is
    /// measured.
    frozen: bool,
    kappa: f64,
    scales: Vec<f64>,
    /// Frame of the last recorded step, the `previous` of the next one.
    previous: Option<Frame>,
    /// Source frames of the last `max_lag + 1` measured frames, oldest first.
    ring: Vec<SourceFrame>,
    last: Option<Continuity>,
    /// Measured frames so far, and whether the last one left a segment the
    /// next frame may continue.
    measured: u64,
    open: bool,
    /// Blocks of the slab moments: the bound, the blocks held, the frames per
    /// closed block and the frames pushed into the open one. Every slab
    /// channel is pushed on every measured frame, so one counter serves all.
    slab_cap: usize,
    slab_held: usize,
    slab_per: usize,
    slab_open: usize,
    /// Sums of the smoothing curve over the sampled frames.
    flow_sum: Vec<f64>,
    flow_count: Vec<u64>,
    flow_frames: u64,
}

/// One measured channel: where its operator, its elements and its state come
/// from, and what it retains.
#[derive(Clone, Debug)]
struct Measured {
    /// Index into the channels of `operators::channels`.
    channel: usize,
    /// Index into `Measurement::channels`.
    series: usize,
    group: usize,
    /// Index into the frame states; 0 is the unsmeared one.
    variant: usize,
    /// Index into the calibrated scales of a multiscale copy.
    scale: Option<usize>,
    components: usize,
    normalization: FrameNormalization,
    exchange: ExchangeParity,
    /// The extra column `evaluate` writes and how the frame collapses it.
    auxiliary: Option<Auxiliary>,
    degenerate: bool,
    requires: Requirements,
    propagator: bool,
    euclidean: bool,
}

/// The element kind and multiscale copy that a group of channels shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupSpec {
    kind: ElementKind,
    scale: Option<usize>,
}

/// What one channel measured on one frame, before anything is committed.
#[derive(Clone, Debug, Default)]
struct Sample {
    values: Vec<f64>,
    valid: Vec<bool>,
    /// `[elements]` auxiliary column of `Signature::auxiliary`, collapsed out
    /// of `values`; empty for a channel without one.
    aux: Vec<f64>,
    mean: Vec<f64>,
    weight: f64,
    coverage: Coverage,
    involutive: bool,
    slab: Vec<f64>,
    slab_weight: Vec<bool>,
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
    stream: Stream,
}
impl AccumulatorState {
    /// CBOR, with the measurement validated on both sides; decode is capped
    /// at 256 MiB and rejects trailing data. A codec failure is
    /// `GasError::Checkpoint`, the cap and the trailing bytes are
    /// `GasError::Configuration`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.measurement.validate()?;
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(self, &mut bytes)
            .map_err(|e| GasError::Checkpoint(e.to_string()))?;
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        require(
            bytes.len() <= 256 * 1024 * 1024,
            "accumulator state decode exceeds 256 MiB",
        )?;
        let mut remaining = bytes;
        let state: Self = ciborium::de::from_reader_with_recursion_limit(&mut remaining, 64)
            .map_err(|e| GasError::Checkpoint(e.to_string()))?;
        require(remaining.is_empty(), "trailing accumulator state data")?;
        state.measurement.validate()?;
        Ok(state)
    }
}
pub struct Accumulator {
    measurement: Measurement,
    extensions: Extensions,
    gas: GasConfig,
    config: MeasurementConfig,
    capabilities: Capabilities,
    channels: Vec<Channel>,
    plan: Vec<Measured>,
    groups: Vec<GroupSpec>,
    stream: Stream,
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
        config.validate()?;
        let mut capabilities = match &extensions.capabilities {
            Some(capabilities) => {
                capabilities.validate()?;
                capabilities.clone()
            }
            None => Capabilities::of(gas, recording, dimension).refine(&config),
        };
        if extensions.field.is_some() {
            // An injected source provides the matter field whatever the gas records.
            capabilities.missing.remove(&Record::Color);
        }
        match &extensions.calibration {
            Some(calibration) => calibration.validate()?,
            None => require(
                config.warmup > 0 || matches!(config.phase.length, LengthScale::Fixed { .. }),
                "a warm-up of zero frames needs a fixed colour phase length or an injected \
                 calibration",
            )?,
        }
        // A configured source that the run does not record leaves the colour
        // channels unavailable one by one; an injected one is an explicit
        // request and its missing record fails the measurement.
        if let Some(source) = extensions.field.as_deref()
            && let Some(reason) = capabilities.check(&source.requires()).reason()
        {
            return Err(GasError::Capability(format!(
                "colour source {}: {reason}",
                source.id()
            )));
        }
        if let TimeAxis::Euclidean { periodic: true, .. } = config.time
            && let Some(reason) = capabilities
                .check(&Requirements::new([Record::PeriodicBox]))
                .reason()
        {
            return Err(GasError::Configuration(format!(
                "a periodic Euclidean-time axis needs a periodic box: {reason}"
            )));
        }
        let context = OperatorContext {
            gas,
            measurement: &config,
            capabilities: &capabilities,
        };
        let channels = operators::channels(&context, &extensions.operators)?;
        require(
            (1..=256).contains(&channels.len()),
            "a measurement needs 1..=256 channels, injected operators included",
        )?;
        if let Some(reason) = every_channel_unavailable(&channels) {
            return Err(GasError::Capability(reason));
        }
        let fingerprint = config.fingerprint(gas, &capabilities, &extensions.identity())?;
        let (plan, groups, series) = plan(&config, &capabilities, &channels);
        let mut stream = Stream {
            slab_cap: config.propagators.max_blocks,
            slab_per: 1,
            ..Stream::default()
        };
        if let Some(flow) = &config.flow {
            stream.flow_sum = vec![0.; flow.steps + 1];
            stream.flow_count = vec![0; flow.steps + 1];
        }
        let mut accumulator = Self {
            measurement: Measurement {
                schema_version: SPECTROSCOPY_VERSION,
                fingerprint,
                config: config.clone(),
                gas: gas.clone(),
                capabilities: capabilities.clone(),
                walkers: 0,
                calibration: extensions.calibration.clone(),
                steps: vec![],
                segment: vec![],
                segments: 0,
                ingested: 0,
                channels: series,
                flow: None,
                notes: vec![],
            },
            extensions,
            gas: gas.clone(),
            config,
            capabilities,
            channels,
            plan,
            groups,
            stream,
        };
        accumulator.admit()?;
        if accumulator.measurement.calibration.is_some() {
            accumulator.freeze()?;
        }
        Ok(accumulator)
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
        let frame = frame::extract(&self.gas, step, &self.config, &self.capabilities)?;
        require(
            self.measurement.walkers == 0 || self.measurement.walkers == frame.n,
            "population size differs from the measured one",
        )?;
        let index = self.measurement.ingested;
        let warmup = self.config.warmup as u64;
        let warm = index < warmup;
        let measured = !warm && (index - warmup).is_multiple_of(self.config.stride as u64);
        // Before the ring of the previous segment is flushed, so that a
        // refused step leaves the retained records untouched.
        if measured {
            require(
                self.measurement.steps.len() < self.config.budget.max_frames,
                "measurement horizon reached; analyze the measurement and start a new one",
            )?;
        }
        let continues = self.stream.last.is_some_and(|last| {
            last.step.checked_add(1) == Some(frame.step)
                && frame.epoch <= last.epoch
                && last.fingerprint
                    == fingerprint(frame.n, frame.d, frame.generation.iter().copied())
        });
        if !continues {
            self.reopen()?;
        }
        if !warm && !self.stream.frozen {
            self.freeze()?;
        }
        let source = self
            .extensions
            .field
            .as_deref()
            .map_or(&self.config.color as &dyn FieldSource, |field| field);
        let mut state = frame::base_state(&frame, &self.capabilities);
        let color = source.fill(
            self.stream.previous.as_ref(),
            &frame,
            self.stream.kappa,
            &mut state,
        );
        if warm {
            self.collect(&frame, &state)?;
        } else if measured {
            let score = score::fill(&frame, &self.gas, &mut state);
            self.measure(&frame, state, &color, &score)?;
        }
        self.measurement.walkers = frame.n;
        self.measurement.ingested += 1;
        self.stream.last = Some(Continuity {
            step: frame.step,
            epoch: frame.epoch,
            fingerprint: fingerprint(
                frame.n,
                frame.d,
                frame
                    .generation
                    .iter()
                    .zip(&frame.cloned)
                    .map(|(generation, &cloned)| generation + u64::from(cloned)),
            ),
        });
        self.stream.previous = Some(frame);
        Ok(())
    }
    /// The measurement so far. Propagator origins younger than `max_lag`
    /// frames are still in the ring and not yet in its moments.
    pub fn measurement(&self) -> &Measurement {
        &self.measurement
    }
    /// A copy with the pending propagator origins flushed: what
    /// `into_measurement` would return now.
    pub fn finalized(&self) -> Result<Measurement> {
        let mut measurement = self.measurement.clone();
        for entry in &self.stream.ring {
            push_origins(&mut measurement, entry)?;
        }
        measurement.validate()?;
        Ok(measurement)
    }
    pub fn into_measurement(self) -> Result<Measurement> {
        self.finalized()
    }
    pub fn state(&self) -> AccumulatorState {
        AccumulatorState {
            schema_version: SPECTROSCOPY_VERSION,
            measurement: self.measurement.clone(),
            stream: self.stream.clone(),
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
    /// `Measurement::validate`, `Frame::validate` on every retained frame,
    /// `Topology::validate(frame.n)` on every retained topology and
    /// `check_stream` on the ring and the counters, so a malformed state is an
    /// error and never a panic.
    pub fn restore_with(state: AccumulatorState, extensions: Extensions) -> Result<Self> {
        if state.schema_version != SPECTROSCOPY_VERSION {
            return Err(GasError::Checkpoint(
                "unsupported accumulator state version".into(),
            ));
        }
        let AccumulatorState {
            measurement,
            stream,
            ..
        } = state;
        measurement.validate()?;
        let (gas, config, capabilities) = (
            measurement.gas.clone(),
            measurement.config.clone(),
            measurement.capabilities.clone(),
        );
        if config.fingerprint(&gas, &capabilities, &extensions.identity())?
            != measurement.fingerprint
        {
            return Err(GasError::Checkpoint(
                "restored measurement was taken with other injected components".into(),
            ));
        }
        if let Some(previous) = &stream.previous {
            previous.validate()?;
        }
        let walkers = measurement.walkers;
        for entry in &stream.ring {
            for group in &entry.groups {
                Topology {
                    step: 0,
                    kind: group.elements.first().map_or(ElementKind::Site, |e| e.kind),
                    elements: group.elements.clone(),
                    involutive: false,
                    coverage: Coverage::default(),
                }
                .validate(walkers)?;
            }
        }
        let context = OperatorContext {
            gas: &gas,
            measurement: &config,
            capabilities: &capabilities,
        };
        let channels = operators::channels(&context, &extensions.operators)?;
        let (plan, groups, series) = plan(&config, &capabilities, &channels);
        require(
            series.len() == measurement.channels.len()
                && series
                    .iter()
                    .zip(&measurement.channels)
                    .all(|(a, b)| a.id == b.id),
            "restored measurement channels differ from the configured ones",
        )?;
        let mut accumulator = Self {
            measurement,
            extensions,
            gas,
            config,
            capabilities,
            channels,
            plan,
            groups,
            stream,
        };
        accumulator.resume();
        accumulator.check_stream()?;
        Ok(accumulator)
    }
    /// What a restored stream must satisfy before a frame reaches it: the
    /// ring holds ascending source frames of the measured ones with a lag
    /// array per configured lag, and the block and flow counters stay inside
    /// their bounds. Checked after `resume`, against the plan the frozen
    /// calibration leaves.
    fn check_stream(&self) -> Result<()> {
        let (lags, stream) = (self.config.max_lag + 1, &self.stream);
        require(
            stream.frozen == self.measurement.calibration.is_some()
                && stream.measured == self.measurement.frames() as u64
                && stream.ring.len() <= self.config.max_lag
                && stream.ring.windows(2).all(|w| w[0].index < w[1].index)
                && stream.ring.last().is_none_or(|e| e.index < stream.measured),
            "restored stream continuity",
        )?;
        require(
            stream.slab_per >= 1
                && stream.slab_open <= stream.slab_per
                && stream.slab_held <= stream.slab_cap
                && stream.slab_cap <= self.config.propagators.max_blocks
                && stream.flow_sum.len() == stream.flow_count.len(),
            "restored stream counters",
        )?;
        for entry in &stream.ring {
            require(
                entry.channels.iter().all(|channel| {
                    channel.group < entry.groups.len()
                        && self.plan.get(channel.channel).is_some_and(|measured| {
                            let components = measured.components;
                            channel.series == measured.series
                                && channel.valid.len() == entry.groups[channel.group].elements.len()
                                && channel.values.len() == channel.valid.len() * components
                                // A frozen score factor is one number per
                                // element; every other channel carries none.
                                // A ring written before the factor existed
                                // decodes without it, and its sinks would be
                                // masked one by one instead of saying so.
                                && channel.factor.len()
                                    == match measured.auxiliary {
                                        Some(auxiliary) if auxiliary.frozen() => {
                                            channel.valid.len()
                                        }
                                        _ => 0,
                                    }
                                && channel.ab.len() == lags
                                && channel.n.len() == lags
                                && channel.a.len() == lags * components
                                && channel.b.len() == channel.a.len()
                        })
                }),
                "restored propagator ring shape",
            )?;
        }
        Ok(())
    }
    /// Reapply to the rebuilt plan what the frozen calibration decided: the
    /// scale of every multiscale copy and, when the warm-up gave no phase
    /// length, the channels that read the colour.
    fn resume(&mut self) {
        if !self.stream.frozen {
            return;
        }
        let (length, scales) = (
            self.measurement
                .calibration
                .as_ref()
                .map_or(0., |c| c.length),
            self.stream.scales.clone(),
        );
        self.apply_scales(&scales);
        if length <= 0. {
            let reason = self
                .measurement
                .calibration
                .as_ref()
                .map_or(String::new(), |c| c.length_source.clone());
            self.decolor(&reason);
        }
    }
    /// Retained bytes against `budget.max_bytes`, with the worst case of a
    /// full series and of the propagator and slab moments. The slab blocks
    /// are halved down to four before the budget refuses the measurement.
    fn admit(&mut self) -> Result<()> {
        let (budget, lags) = (self.config.budget, self.config.max_lag + 1);
        let bins = match self.config.time {
            TimeAxis::Euclidean { bins, .. } => bins,
            TimeAxis::MonteCarlo => 0,
        };
        let (mut fixed, mut block) = (0usize, 0usize);
        for measured in &self.plan {
            let cells = checked_mul(budget.max_frames, checked_add(measured.components, 1)?)?;
            fixed = checked_add(fixed, checked_mul(cells, 8)?)?;
            if measured.propagator {
                let entries = checked_mul(
                    checked_mul(self.config.propagators.max_blocks, lags)?,
                    checked_add(checked_mul(measured.components, 2)?, 2)?,
                )?;
                fixed = checked_add(fixed, checked_mul(entries, 8)?)?;
            }
            if measured.euclidean {
                let entries = checked_add(
                    checked_mul(checked_mul(bins, bins)?, 2)?,
                    checked_mul(bins, checked_add(measured.components, 1)?)?,
                )?;
                block = checked_add(block, checked_mul(entries, 8)?)?;
            }
        }
        let total = |blocks: usize| checked_add(fixed, checked_mul(block, blocks)?);
        while block > 0
            && self.stream.slab_cap > 4
            && total(self.stream.slab_cap)? > budget.max_bytes
        {
            self.stream.slab_cap /= 2;
        }
        enforce(total(self.stream.slab_cap)?, budget.max_bytes)
    }
    /// Close the blocks of the segment that ended, flush its pending origins
    /// and drop the ring and the retained frame: no lag spans a break.
    fn reopen(&mut self) -> Result<()> {
        for entry in std::mem::take(&mut self.stream.ring) {
            push_origins(&mut self.measurement, &entry)?;
        }
        for series in &mut self.measurement.channels {
            if let Some(moments) = &mut series.propagator {
                moments.close_block();
            }
        }
        if self.stream.slab_held > 0 {
            self.stream.slab_open = self.stream.slab_per;
        }
        self.stream.previous = None;
        self.stream.open = false;
        Ok(())
    }
    /// Pool the samples one warm-up frame contributes.
    fn collect(&mut self, frame: &Frame, state: &FrameState) -> Result<()> {
        let cap = (scales::CALIBRATION_SAMPLES / self.config.warmup.max(1)).max(1);
        let (n, d) = (frame.n, frame.d);
        self.stream.warmup.frames += 1;
        if let Some(velocity) = &state.phase_velocity {
            let held = self.stream.warmup.velocity.len();
            for i in (0..n).filter(|&i| state.force_valid[i]) {
                if self.stream.warmup.velocity.len() >= held + cap {
                    break;
                }
                self.stream
                    .warmup
                    .velocity
                    .extend_from_slice(&velocity[i * d..(i + 1) * d]);
            }
        }
        let held = self.stream.warmup.length.len();
        match self.config.phase.length {
            LengthScale::Fixed { .. } => {}
            LengthScale::WarmupCompanionMedian => {
                if let Some(companions) = &frame.distance {
                    let domain = periodic_box(&self.gas.boundary);
                    for i in 0..n {
                        let Some(k) = companions.first(i, &frame.eligible) else {
                            continue;
                        };
                        if self.stream.warmup.length.len() >= held + cap {
                            break;
                        }
                        let j = companions.slot[i * companions.count + k] as usize;
                        let squared: f64 = (0..d)
                            .map(|a| {
                                let delta = frame.x[j * d + a] - frame.x[i * d + a];
                                match domain.filter(|domain| a < domain.lower.len()) {
                                    Some(domain) => domain.minimum_image::<f64>(delta, a),
                                    None => delta,
                                }
                            })
                            .map(|delta| delta * delta)
                            .sum();
                        self.stream.warmup.length.push(squared.sqrt());
                    }
                }
            }
            LengthScale::WarmupEdgeMean { geodesic } => {
                if let Some(graph) = &frame.graph {
                    let lengths = if geodesic {
                        &graph.geodesic_length
                    } else {
                        &graph.euclidean_length
                    };
                    let mut slot = 0;
                    for i in 0..graph.graph.nodes() {
                        for &j in graph.graph.row(i) {
                            if frame.eligible[i]
                                && frame.eligible[j as usize]
                                && self.stream.warmup.length.len() < held + cap
                                && let Some(&length) = lengths.get(slot)
                            {
                                self.stream.warmup.length.push(length);
                            }
                            slot += 1;
                        }
                    }
                }
            }
        }
        if let (Some(scales), Some(graph)) = (&self.config.scales, &frame.graph) {
            let distances = scales::distances(
                graph,
                scales.length,
                f64::INFINITY,
                self.config.budget.max_bytes,
                Parallelism::Auto,
            )?;
            self.stream
                .warmup
                .scale
                .extend(scales::pair_samples(&distances, n, cap));
        }
        if let Some(times) = &state.euclidean_time {
            for (&y, _) in times
                .iter()
                .zip(&frame.eligible)
                .filter(|&(y, &eligible)| eligible && y.is_finite())
            {
                let range = self.stream.warmup.euclidean.get_or_insert([y, y]);
                range[0] = range[0].min(y);
                range[1] = range[1].max(y);
            }
        }
        self.pair_statistics(frame, state)
    }
    /// The two pair statistics of `Calibration`, over the first distance
    /// companions and over the ordered eligible pairs of a warm-up frame.
    fn pair_statistics(&mut self, frame: &Frame, state: &FrameState) -> Result<()> {
        let (n, d) = (frame.n, frame.d);
        if let (Some(epsilon), Some(companions)) = (self.epsilon_d(), &frame.distance) {
            let distance = match score::AmplitudeDistance::new(
                &self.gas,
                &self.gas.distance_donors.distance,
                &self.config.electroweak,
            ) {
                Ok(distance) => Some(distance),
                Err(GasError::Capability(_)) => None,
                Err(e) => return Err(e),
            };
            if let Some(distance) = distance {
                for i in 0..n {
                    let Some(k) = companions.first(i, &frame.eligible) else {
                        continue;
                    };
                    let j = companions.slot[i * companions.count + k] as usize;
                    if let Some(squared) = distance.squared(state, i, j) {
                        self.stream.warmup.pair_weight[0] += (-squared / (epsilon * epsilon)).exp();
                        self.stream.warmup.pair_weight[1] += 1.;
                    }
                }
            }
        }
        if let Some(viscosity) = self
            .gas
            .qft
            .viscosity
            .as_ref()
            .filter(|viscosity| viscosity.coefficient > 0.)
        {
            let rho = viscosity.bandwidth;
            for i in (0..n).filter(|&i| frame.eligible[i]) {
                for j in (0..n).filter(|&j| j != i && frame.eligible[j]) {
                    let squared: f64 = (0..d)
                        .map(|a| frame.x[i * d + a] - frame.x[j * d + a])
                        .map(|delta| delta * delta)
                        .sum();
                    let kernel = (-squared / (2. * rho * rho)).exp();
                    self.stream.warmup.kernel[0] += kernel * kernel;
                    self.stream.warmup.kernel[1] += 1.;
                }
            }
        }
        Ok(())
    }
    /// Interaction range of the distance companions, as `ElectroweakScales`
    /// resolves it against the companion kernel.
    fn epsilon_d(&self) -> Option<f64> {
        match self.config.electroweak.epsilon_d {
            Range::Fixed { value } => Some(value),
            Range::FromKernel => self.capabilities.distance_kernel_width,
        }
    }
    fn epsilon_c(&self) -> Option<f64> {
        match self.config.electroweak.epsilon_c {
            Range::Fixed { value } => Some(value),
            Range::FromKernel => self.capabilities.cloning_kernel_width,
        }
    }
    /// Freeze the warm-up samples into `Measurement::calibration`. An
    /// injected calibration replaces the samples; a phase length the warm-up
    /// could not give leaves `length` at 0 and makes every channel that reads
    /// the colour unavailable with that reason.
    fn freeze(&mut self) -> Result<()> {
        self.stream.frozen = true;
        if let Some(calibration) = self.measurement.calibration.clone() {
            self.stream.scales = calibration.scales.clone();
            self.apply_scales(&calibration.scales);
            if calibration.length > 0. {
                self.stream.kappa = color::kappa(&self.config.phase, calibration.length)?;
            } else {
                self.decolor(&calibration.length_source);
            }
            return Ok(());
        }
        let warmup = std::mem::take(&mut self.stream.warmup);
        let (length, source) = match color::phase_length(self.config.phase.length, &warmup.length) {
            Ok(length) => (
                length,
                format!(
                    "{}: {} samples over {} warm-up frames",
                    length_source(self.config.phase.length),
                    warmup.length.len(),
                    warmup.frames
                ),
            ),
            Err(GasError::Capability(reason)) => (0., format!("unavailable: {reason}")),
            Err(e) => return Err(e),
        };
        self.stream.kappa = if length > 0. {
            color::kappa(&self.config.phase, length)?
        } else {
            0.
        };
        self.stream.scales = match &self.config.scales {
            Some(scales) => scales::calibrate(&warmup.scale, &scales.scales)?,
            None => vec![],
        };
        let euclidean_range = match self.config.time {
            TimeAxis::Euclidean {
                periodic, range, ..
            } => self.euclidean_range(periodic, range, warmup.euclidean)?,
            TimeAxis::MonteCarlo => None,
        };
        let mean = |[sum, count]: [f64; 2]| (count > 0.).then(|| sum / count);
        let calibration = Calibration {
            warmup_frames: warmup.frames,
            length,
            length_source: source,
            kappa: self.stream.kappa,
            phase_wrapping: color::phase_wrapping(self.stream.kappa, &warmup.velocity),
            mass: self.config.phase.mass,
            h_eff: self.config.phase.h_eff,
            electroweak_h_eff: self.config.electroweak.h_eff,
            h_s: self
                .config
                .electroweak
                .h_s
                .unwrap_or(self.config.electroweak.h_eff),
            epsilon_d: self.epsilon_d(),
            epsilon_c: self.epsilon_c(),
            epsilon_clone: self.gas.clone_decision.epsilon,
            dt: self.capabilities.time_step,
            euclidean_range,
            scales: self.stream.scales.clone(),
            pair_weight_n1: mean(warmup.pair_weight),
            viscous_kernel_second_moment: mean(warmup.kernel),
        };
        calibration.validate()?;
        let scales = calibration.scales.clone();
        let (positive, reason) = (length > 0., calibration.length_source.clone());
        self.measurement.calibration = Some(calibration);
        self.apply_scales(&scales);
        if !positive {
            self.decolor(&reason);
        }
        Ok(())
    }
    /// Bin range of the Euclidean-time axis: the configured one, the box
    /// period of a periodic axis, or what the warm-up covered.
    fn euclidean_range(
        &self,
        periodic: bool,
        range: Option<[f64; 2]>,
        warmup: Option<[f64; 2]>,
    ) -> Result<Option<[f64; 2]>> {
        let period = self.capabilities.euclidean_axis.and_then(|axis| {
            periodic_box(&self.gas.boundary)
                .filter(|domain| axis < domain.lower.len())
                .map(|domain| [domain.lower[axis], domain.upper[axis]])
        });
        if !periodic {
            return Ok(range.or(warmup).filter(|[lo, hi]| lo < hi));
        }
        let Some(period) = period else {
            return Err(GasError::Configuration(
                "a periodic Euclidean-time axis needs a periodic box on that coordinate".into(),
            ));
        };
        let Some([lo, hi]) = range else {
            return Ok(Some(period));
        };
        require(
            hi - lo == period[1] - period[0],
            "a periodic Euclidean-time range must span the box period",
        )?;
        Ok(Some([lo, hi]))
    }
    /// Write the calibrated scale of every multiscale copy into its series
    /// and drop the copies the ladder does not reach.
    fn apply_scales(&mut self, scales: &[f64]) {
        for measured in &self.plan {
            if let Some(index) = measured.scale
                && let Some(series) = self.measurement.channels.get_mut(measured.series)
            {
                match scales.get(index) {
                    Some(&scale) => series.scale = Some(scale),
                    None => {
                        series.availability =
                            Availability::unavailable("the calibrated scale ladder is shorter");
                    }
                }
            }
        }
        self.plan
            .retain(|measured| measured.scale.is_none_or(|index| index < scales.len()));
    }
    /// Make every channel that reads the colour unavailable with `reason`.
    fn decolor(&mut self, reason: &str) {
        for measured in &self.plan {
            if measured.requires.records.contains(&Record::Color)
                && let Some(series) = self.measurement.channels.get_mut(measured.series)
            {
                series.availability = Availability::unavailable(reason);
            }
        }
        self.plan
            .retain(|measured| !measured.requires.records.contains(&Record::Color));
    }
    /// Evaluate one measured frame and commit it: retained series, coverage,
    /// propagator ring, Euclidean slabs and the smoothing diagnostic.
    fn measure(
        &mut self,
        frame: &Frame,
        base: FrameState,
        color: &Availability,
        score: &Availability,
    ) -> Result<()> {
        let context = OperatorContext {
            gas: &self.gas,
            measurement: &self.config,
            capabilities: &self.capabilities,
        };
        let smear = matches!(&self.config.scales, Some(scales) if scales.mode == ScaleMode::Smear);
        let largest = self.stream.scales.last().copied().unwrap_or(0.);
        let distances = match (&frame.graph, &self.config.scales) {
            (Some(graph), Some(scales)) if largest > 0. => Some(scales::distances(
                graph,
                scales.length,
                if smear {
                    scales::SMEAR_CUTOFF * largest
                } else {
                    largest
                },
                self.config.budget.max_bytes,
                Parallelism::Auto,
            )?),
            _ => None,
        };
        let mut states = vec![base];
        if smear {
            for &scale in &self.stream.scales {
                let mut variant = states[0].clone();
                match &distances {
                    Some(distances) => {
                        let (color, valid) = scales::smear(&states[0], distances, scale);
                        variant.color = color;
                        variant.color_valid = valid;
                    }
                    None => {
                        variant.color.fill(C::ZERO);
                        variant.color_valid.fill(false);
                    }
                }
                states.push(variant);
            }
        }
        let mut topologies: Vec<(ElementKind, Topology)> = vec![];
        for group in &self.groups {
            if !topologies.iter().any(|(kind, _)| *kind == group.kind) {
                topologies.push((group.kind, topology::build(frame, group.kind, &self.config)));
            }
        }
        let mut elements: Vec<(Vec<Element>, Coverage, bool)> = vec![];
        for group in &self.groups {
            let Some((_, source)) = topologies.iter().find(|(kind, _)| *kind == group.kind) else {
                elements.push((vec![], Coverage::default(), false));
                continue;
            };
            let mut coverage = source.coverage;
            let kept: Vec<Element> = match (group.scale, &distances) {
                (Some(index), Some(distances)) if !smear => {
                    let scale = self.stream.scales.get(index).copied().unwrap_or(0.);
                    let keep = scales::gate(source, distances, frame.n, scale, &states[0].cloned);
                    source
                        .elements
                        .iter()
                        .zip(&keep)
                        .filter(|&(_, &keep)| keep)
                        .map(|(element, _)| *element)
                        .collect()
                }
                (Some(_), None) => vec![],
                _ => source.elements.clone(),
            };
            let removed = (source.elements.len() - kept.len()) as u64;
            coverage.masked_scale += removed;
            coverage.valid -= removed;
            let involutive = if group.scale.is_some() && !smear {
                topology::is_involutive(&kept)
            } else {
                source.involutive
            };
            elements.push((kept, coverage, involutive));
        }
        let bins = match self.config.time {
            TimeAxis::Euclidean { bins, .. } => bins,
            TimeAxis::MonteCarlo => 0,
        };
        let slabs = self
            .measurement
            .calibration
            .as_ref()
            .and_then(|calibration| calibration.euclidean_range)
            .filter(|_| bins >= 2);
        let periodic = matches!(self.config.time, TimeAxis::Euclidean { periodic: true, .. });
        let mut samples: Vec<Sample> = Vec::with_capacity(self.plan.len());
        for measured in &self.plan {
            let (group, _, involutive) = &elements[measured.group];
            let mut sample = Sample {
                coverage: elements[measured.group].1,
                involutive: *involutive,
                ..Sample::default()
            };
            let state = states.get(measured.variant).unwrap_or(&states[0]);
            let operator = self.channels[measured.channel].operator(&self.extensions.operators);
            evaluate(
                operator,
                group,
                state,
                &context,
                measured,
                &mut sample.values,
                &mut sample.valid,
            );
            if let Some(auxiliary) = measured.auxiliary {
                collapse(
                    auxiliary,
                    measured.components,
                    group,
                    None,
                    &mut sample.values,
                    &mut sample.valid,
                    &mut sample.aux,
                );
            }
            let colored = measured.requires.records.contains(&Record::Color);
            for (index, element) in group.iter().enumerate() {
                let degenerate = element.kind == ElementKind::Triplet
                    && element.walkers[1] == element.walkers[2]
                    && !measured.degenerate;
                if degenerate {
                    sample.valid[index] = false;
                    sample.coverage.valid -= 1;
                    sample.coverage.masked_self += 1;
                } else if !sample.valid[index] {
                    sample.coverage.valid -= 1;
                    if colored {
                        sample.coverage.masked_color += 1;
                    } else {
                        sample.coverage.masked_ineligible += 1;
                    }
                }
            }
            let records = self.records(measured, frame, color, score, state);
            frame_mean(measured, group, &mut sample, frame.n, records);
            if measured.euclidean
                && let Some(range) = slabs
            {
                slab_sums(
                    measured,
                    group,
                    &mut sample,
                    frame.n,
                    state,
                    range,
                    bins,
                    periodic,
                    records,
                );
            }
            sample.involutive &= sample.coverage.valid > 0;
            samples.push(sample);
        }
        // Without a frozen range no frame carries a slab, so no channel opens
        // slab moments at all rather than holding an all-zero record.
        let retained = slabs.map_or(0, |_| bins);
        self.commit(frame, &states, &samples, &elements, retained)
    }
    /// Whether the records this channel reads exist on the frame at all: a
    /// frame whose readout could not be evaluated is missing data, never a
    /// value, under either normalisation.
    fn records(
        &self,
        measured: &Measured,
        frame: &Frame,
        color: &Availability,
        score: &Availability,
        state: &FrameState,
    ) -> bool {
        measured.requires.records.iter().all(|record| match record {
            Record::Color => color.is_available(),
            Record::Velocities => frame.v.is_some(),
            Record::Fitness => frame.fitness.is_some(),
            Record::DistanceCompanions => frame.distance.is_some(),
            Record::CloningCompanions | Record::ClonePlan => {
                frame.cloning.is_some() && score.is_available()
            }
            Record::Graph => frame.graph.is_some(),
            Record::EuclideanTime => state.euclidean_time.is_some(),
            Record::PeriodicBox => true,
        }) && measured.scale.is_none_or(|_| frame.graph.is_some())
    }
    /// Append one measured frame to the retained records.
    fn commit(
        &mut self,
        frame: &Frame,
        states: &[FrameState],
        samples: &[Sample],
        elements: &[(Vec<Element>, Coverage, bool)],
        bins: usize,
    ) -> Result<()> {
        let lags = self.config.max_lag + 1;
        let mut ring = SourceFrame {
            index: self.stream.measured,
            groups: elements
                .iter()
                .map(|(group, ..)| SourceGroup {
                    elements: group.clone(),
                })
                .collect(),
            channels: vec![],
        };
        for (index, (measured, sample)) in self.plan.iter().zip(samples).enumerate() {
            if !measured.propagator {
                continue;
            }
            let components = measured.components;
            let mut channel = SourceChannel {
                channel: index,
                series: measured.series,
                group: measured.group,
                values: sample.values.clone(),
                valid: sample.valid.clone(),
                factor: match measured.auxiliary {
                    Some(auxiliary) if auxiliary.frozen() => sample.aux.clone(),
                    _ => vec![],
                },
                ab: vec![0.; lags],
                a: vec![0.; lags * components],
                b: vec![0.; lags * components],
                n: vec![0.; lags],
            };
            for (element, row) in elements[measured.group]
                .0
                .iter()
                .zip(sample.values.chunks_exact(components))
                .zip(&sample.valid)
                .filter(|&(_, &valid)| valid)
                .map(|((element, row), _)| (element, row))
            {
                let weight = element.weight;
                channel.ab[0] += weight * row.iter().map(|x| x * x).sum::<f64>();
                channel.n[0] += weight;
                for (k, &value) in row.iter().enumerate() {
                    channel.a[k] += weight * value;
                    channel.b[k] += weight * value;
                }
            }
            ring.channels.push(channel);
        }
        let mut bytes = checked_add(self.measurement.buffer_bytes()?, ring.buffer_bytes()?)?;
        bytes = checked_add(bytes, self.stream.warmup.buffer_bytes()?)?;
        for entry in &self.stream.ring {
            bytes = checked_add(bytes, entry.buffer_bytes()?)?;
        }
        enforce(bytes, self.config.budget.max_bytes)?;
        if !self.stream.open {
            self.measurement.segments += 1;
            self.stream.open = true;
        }
        self.measurement.steps.push(frame.step);
        self.measurement.segment.push(self.measurement.segments - 1);
        for (measured, sample) in self.plan.iter().zip(samples) {
            let Some(series) = self.measurement.channels.get_mut(measured.series) else {
                continue;
            };
            series.values.extend_from_slice(&sample.mean);
            series.weight.push(sample.weight);
            let mut coverage = sample.coverage;
            if coverage.valid > 0 {
                coverage.frames = 1;
            } else {
                coverage.empty_frames = 1;
            }
            series.coverage.merge(&coverage);
            series.involutive_frames += u64::from(sample.involutive);
        }
        self.push_slabs(samples, bins);
        self.push_ring(ring, states)?;
        self.sample_flow(frame, &states[0])?;
        self.stream.measured += 1;
        Ok(())
    }
    /// One slab row per channel, opening, merging and closing blocks exactly
    /// as `BlockMoments` does for a propagator origin. `bins` is 0 when the
    /// axis carries no frozen range and nothing is retained.
    fn push_slabs(&mut self, samples: &[Sample], bins: usize) {
        if bins == 0 || !self.plan.iter().any(|measured| measured.euclidean) {
            return;
        }
        if self.stream.slab_open >= self.stream.slab_per
            && self.stream.slab_held == self.stream.slab_cap
        {
            let held = self.stream.slab_held;
            let open = self.stream.slab_open.min(self.stream.slab_per);
            for series in &mut self.measurement.channels {
                if let Some(slabs) = &mut series.euclidean {
                    merge_slab_pairs(slabs);
                }
            }
            self.stream.slab_held = held.div_ceil(2);
            self.stream.slab_open = ((held - 1) % 2) * self.stream.slab_per + open;
            self.stream.slab_per *= 2;
        }
        let opening = self.stream.slab_held == 0 || self.stream.slab_open >= self.stream.slab_per;
        if opening {
            self.stream.slab_held += 1;
            self.stream.slab_open = 0;
        }
        let (held, per) = (self.stream.slab_held, self.stream.slab_per);
        for (measured, sample) in self.plan.iter().zip(samples) {
            if !measured.euclidean {
                continue;
            }
            let Some(series) = self.measurement.channels.get_mut(measured.series) else {
                continue;
            };
            let components = measured.components;
            let slabs = series.euclidean.get_or_insert_with(|| SlabMoments {
                bins,
                components,
                origins_per_block: per,
                blocks: 0,
                pair_ab: vec![],
                pair_n: vec![],
                profile: vec![],
                profile_n: vec![],
            });
            slabs.origins_per_block = per;
            if slabs.blocks < held {
                slabs.blocks = held;
                slabs.pair_ab.resize(held * bins * bins, 0.);
                slabs.pair_n.resize(held * bins * bins, 0.);
                slabs.profile.resize(held * bins * components, 0.);
                slabs.profile_n.resize(held * bins, 0.);
            }
            if sample.slab_weight.len() != bins {
                continue;
            }
            let block = held - 1;
            for b in 0..bins {
                if !sample.slab_weight[b] {
                    continue;
                }
                let row = &sample.slab[b * components..(b + 1) * components];
                slabs.profile_n[block * bins + b] += 1.;
                for (k, &value) in row.iter().enumerate() {
                    slabs.profile[(block * bins + b) * components + k] += value;
                }
                for c in b..bins {
                    if !sample.slab_weight[c] {
                        continue;
                    }
                    let other = &sample.slab[c * components..(c + 1) * components];
                    let entry = (block * bins + b) * bins + c;
                    slabs.pair_ab[entry] += row.iter().zip(other).map(|(x, y)| x * y).sum::<f64>();
                    slabs.pair_n[entry] += 1.;
                }
            }
        }
        self.stream.slab_open += 1;
    }
    /// Push the source frame, fill the lags of every older ring entry against
    /// this sink state and flush the entry that just completed `max_lag`.
    fn push_ring(&mut self, ring: SourceFrame, states: &[FrameState]) -> Result<()> {
        if ring.channels.is_empty() && self.stream.ring.is_empty() {
            return Ok(());
        }
        let newest = ring.index;
        self.stream.ring.push(ring);
        let (max_lag, stride) = (self.config.max_lag, self.config.propagators.lag_stride);
        let incarnation = self.config.identity == IdentityPolicy::Incarnation;
        let (mut values, mut valid, mut aux) = (vec![], vec![], vec![]);
        let entries = self.stream.ring.len().saturating_sub(1);
        for entry in &mut self.stream.ring[..entries] {
            let lag = (newest - entry.index) as usize;
            if lag > max_lag || !lag.is_multiple_of(stride) {
                continue;
            }
            let SourceFrame {
                groups, channels, ..
            } = entry;
            for channel in channels.iter_mut() {
                let Some(measured) = self.plan.get(channel.channel) else {
                    continue;
                };
                let Some(group) = groups.get(channel.group) else {
                    continue;
                };
                let components = measured.components;
                let state = states.get(measured.variant).unwrap_or(&states[0]);
                let operator = self.channels[measured.channel].operator(&self.extensions.operators);
                evaluate(
                    operator,
                    &group.elements,
                    state,
                    &OperatorContext {
                        gas: &self.gas,
                        measurement: &self.config,
                        capabilities: &self.capabilities,
                    },
                    measured,
                    &mut values,
                    &mut valid,
                );
                // The score factor of the element is the one its source frame
                // determined; only the momentum weight is read here.
                if let Some(auxiliary) = measured.auxiliary {
                    collapse(
                        auxiliary,
                        components,
                        &group.elements,
                        Some(&channel.factor),
                        &mut values,
                        &mut valid,
                        &mut aux,
                    );
                }
                let colored = measured.requires.records.contains(&Record::Color);
                let mut coverage = Coverage::default();
                for (index, element) in group.elements.iter().enumerate() {
                    if !channel.valid[index] {
                        continue;
                    }
                    if incarnation && !topology::same_incarnation(element, &state.generation) {
                        coverage.masked_identity += 1;
                        continue;
                    }
                    if !valid[index] {
                        if colored {
                            coverage.masked_color += 1;
                        } else {
                            coverage.masked_ineligible += 1;
                        }
                        continue;
                    }
                    let weight = element.weight;
                    let source = &channel.values[index * components..(index + 1) * components];
                    let sink = &values[index * components..(index + 1) * components];
                    channel.ab[lag] +=
                        weight * source.iter().zip(sink).map(|(x, y)| x * y).sum::<f64>();
                    channel.n[lag] += weight;
                    for k in 0..components {
                        channel.a[lag * components + k] += weight * source[k];
                        channel.b[lag * components + k] += weight * sink[k];
                    }
                }
                if let Some(series) = self.measurement.channels.get_mut(measured.series) {
                    series.coverage.merge(&coverage);
                }
            }
        }
        if self.stream.ring.len() > max_lag {
            let entry = self.stream.ring.remove(0);
            push_origins(&mut self.measurement, &entry)?;
        }
        Ok(())
    }
    /// Add the colour roughness curve of every `flow.every`-th measured frame.
    fn sample_flow(&mut self, frame: &Frame, state: &FrameState) -> Result<()> {
        let Some(config) = &self.config.flow else {
            return Ok(());
        };
        if !self.stream.measured.is_multiple_of(config.every as u64) {
            return Ok(());
        }
        let Some(graph) = &frame.graph else {
            return Ok(());
        };
        let curve = flow::smooth(state, &graph.graph, config)?;
        self.stream.flow_frames += 1;
        for ((sum, count), value) in self
            .stream
            .flow_sum
            .iter_mut()
            .zip(&mut self.stream.flow_count)
            .zip(&curve)
        {
            if let Some(value) = value {
                *sum += value;
                *count += 1;
            }
        }
        self.measurement.flow = Some(FlowDiagnostic {
            frames: self.stream.flow_frames,
            steps: (0..self.stream.flow_sum.len()).collect(),
            roughness: self
                .stream
                .flow_sum
                .iter()
                .zip(&self.stream.flow_count)
                .map(|(sum, &count)| (count > 0).then(|| sum / count as f64))
                .collect(),
        });
        Ok(())
    }
}

/// The reason of the first channel when no channel is available.
fn every_channel_unavailable(channels: &[Channel]) -> Option<String> {
    channels
        .iter()
        .all(|channel| !channel.availability.is_available())
        .then(|| {
            channels
                .first()
                .and_then(|channel| channel.availability.reason())
                .unwrap_or("no channel is available")
                .to_string()
        })
}

/// One measured channel per available channel, then one multiscale copy per
/// configured scale of the channels a `ScaleConfig` selects. The series of an
/// unavailable channel stays empty; its scale is written when the ladder is
/// calibrated.
fn plan(
    config: &MeasurementConfig,
    capabilities: &Capabilities,
    channels: &[Channel],
) -> (Vec<Measured>, Vec<GroupSpec>, Vec<ChannelSeries>) {
    let graph = capabilities.check(&Requirements::new([Record::Graph]));
    let copies = match &config.scales {
        Some(scales) => match &scales.scales {
            ScaleSelection::Fixed { values } => values.len(),
            ScaleSelection::Quantiles { count, .. } => *count,
        },
        None => 0,
    };
    let smear = matches!(&config.scales, Some(scales) if scales.mode == ScaleMode::Smear);
    let (mut plan, mut groups, mut series) = (vec![], vec![], vec![]);
    for (index, channel) in channels.iter().enumerate() {
        let selected = config.scales.as_ref().is_some_and(|scales| {
            scales
                .channels
                .iter()
                .any(|entry| entry == "*" || *entry == channel.spec.id() || *entry == channel.id)
        });
        for copy in 0..=if selected { copies } else { 0 } {
            let scale = (copy > 0).then(|| copy - 1);
            let availability = match scale {
                Some(_) => graph.clone().and(channel.availability.clone()),
                None => channel.availability.clone(),
            };
            let spec = GroupSpec {
                kind: channel.kind,
                scale,
            };
            let group = groups
                .iter()
                .position(|held| *held == spec)
                .unwrap_or_else(|| {
                    groups.push(spec);
                    groups.len() - 1
                });
            let components = channel.signature.as_ref().map_or(1, |s| s.components);
            if availability.is_available()
                && let Some(signature) = &channel.signature
            {
                plan.push(Measured {
                    channel: index,
                    series: series.len(),
                    group,
                    variant: match (scale, smear) {
                        (Some(index), true) => index + 1,
                        _ => 0,
                    },
                    scale,
                    components,
                    normalization: signature.normalization.unwrap_or(config.normalization),
                    exchange: signature.exchange,
                    auxiliary: signature.auxiliary,
                    degenerate: signature.degenerate,
                    requires: signature.requires.clone(),
                    propagator: signature.correlatable
                        && signature.propagatable
                        && config.propagators.enabled(&channel.spec.id(), &channel.id),
                    euclidean: signature.correlatable
                        && matches!(config.time, TimeAxis::Euclidean { .. })
                        && capabilities.has(Record::EuclideanTime),
                });
            }
            series.push(new_series(
                match scale {
                    Some(index) => format!("{}@{index}", channel.id),
                    None => channel.id.clone(),
                },
                channel,
                components,
                availability,
            ));
        }
    }
    (plan, groups, series)
}

/// Name of a length arm inside `Calibration::length_source`.
fn length_source(length: LengthScale) -> &'static str {
    match length {
        LengthScale::Fixed { .. } => "fixed",
        LengthScale::WarmupCompanionMedian => "warmup_companion_median",
        LengthScale::WarmupEdgeMean { geodesic: true } => "warmup_edge_mean_geodesic",
        LengthScale::WarmupEdgeMean { geodesic: false } => "warmup_edge_mean_euclidean",
    }
}

/// FNV-1a over the population shape and one generation per walker: the
/// continuity guard. The frame that continues a worldline carries the
/// generations of its predecessor raised by its accepted clones.
fn fingerprint(n: usize, d: usize, generations: impl Iterator<Item = u64>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in [n as u64, d as u64].into_iter().chain(generations) {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// The series of one channel before any frame was measured.
fn new_series(
    id: String,
    channel: &Channel,
    components: usize,
    availability: Availability,
) -> ChannelSeries {
    let signature: Option<&Signature> = channel.signature.as_ref();
    let descriptor = signature.map(|signature| &signature.descriptor);
    ChannelSeries {
        id,
        spec: channel.spec.clone(),
        kind: channel.kind,
        scale: None,
        definition: descriptor.map(|d| d.definition.clone()).unwrap_or_default(),
        book_label: descriptor.map(|d| d.book_label.clone()).unwrap_or_default(),
        note: descriptor.map(|d| d.note.clone()).unwrap_or_default(),
        exchange: signature.map_or(ExchangeParity::Even, |signature| signature.exchange),
        spatial_parity: descriptor.and_then(|d| d.spatial_parity),
        correlatable: signature.is_none_or(|signature| signature.correlatable),
        components,
        availability,
        coverage: Coverage::default(),
        involutive_frames: 0,
        values: vec![],
        weight: vec![],
        propagator: None,
        euclidean: None,
    }
}

/// `evaluate_all` on rows of `Signature::width`, with a nonfinite component
/// counted as a mask, so that no retained sum can be nonfinite. The auxiliary
/// column is exempt: a nonfinite score factor is missing data that `collapse`
/// masks at a source time and the frozen factor overrides at a sink.
fn evaluate(
    operator: &dyn LocalOperator,
    elements: &[Element],
    state: &FrameState,
    context: &OperatorContext<'_>,
    measured: &Measured,
    values: &mut Vec<f64>,
    valid: &mut Vec<bool>,
) {
    let components = measured.components;
    let width = components + usize::from(measured.auxiliary.is_some());
    values.clear();
    values.resize(elements.len() * width, 0.);
    valid.clear();
    valid.resize(elements.len(), false);
    operator.evaluate_all(elements, state, context, values, valid);
    for (row, ok) in values.chunks_exact(width).zip(valid.iter_mut()) {
        *ok = *ok && row[..components].iter().all(|x| x.is_finite());
    }
}

/// Collapse the auxiliary column of a raw evaluation into the `components` of
/// the measured series, in place. `frozen` holds the score factors of the
/// source time at a sink and is `None` at the source frame, where the factors
/// of the evaluated frame are taken and left in `aux`. A momentum mode
/// subtracts the weighted frame mean of the observable before projecting it;
/// a score orientation the frame cannot determine masks the element.
fn collapse(
    auxiliary: Auxiliary,
    components: usize,
    elements: &[Element],
    frozen: Option<&[f64]>,
    values: &mut Vec<f64>,
    valid: &mut [bool],
    aux: &mut Vec<f64>,
) {
    let width = components + 1;
    aux.clear();
    aux.extend(values.chunks_exact(width).map(|row| row[components]));
    let bare = match auxiliary {
        Auxiliary::MomentumMode => {
            let (mut sum, mut denominator) = (0., 0.);
            for (index, element) in elements.iter().enumerate() {
                if valid[index] {
                    sum += element.weight * values[index * width];
                    denominator += element.weight;
                }
            }
            if denominator > 0. {
                sum / denominator
            } else {
                0.
            }
        }
        _ => 0.,
    };
    for index in 0..elements.len() {
        let factor = match frozen {
            Some(frozen) if auxiliary.frozen() => frozen.get(index).copied().unwrap_or(f64::NAN),
            _ => aux[index],
        };
        if !factor.is_finite() || (auxiliary == Auxiliary::ScoreOrientation && factor == 0.) {
            valid[index] = false;
        }
        for k in 0..components {
            // `width > components`, so the destination never runs ahead of the
            // source and the rows collapse in place.
            values[index * components + k] = (values[index * width + k] - bare) * factor;
        }
    }
    values.truncate(elements.len() * components);
}

/// `A_t(O)` and its weight. An exchange-odd operator leaves out the elements
/// whose mirror is valid too: they cancel exactly, so they enter neither the
/// sum nor the denominator. An operator with a frame weight of its own
/// (`Auxiliary::ScoreDispersion`) has no average where that weight vanishes.
fn frame_mean(
    measured: &Measured,
    elements: &[Element],
    sample: &mut Sample,
    walkers: usize,
    records: bool,
) {
    let components = measured.components;
    let mirrors = (measured.exchange == ExchangeParity::Odd).then(|| topology::mirrors(elements));
    let mut sum = vec![0.; components];
    let (mut denominator, mut dispersion) = (0., 0.);
    for (index, element) in elements.iter().enumerate() {
        let mirrored = mirrors
            .as_ref()
            .and_then(|mirrors| mirrors[index])
            .is_some_and(|mirror| sample.valid[mirror]);
        if !sample.valid[index] || mirrored {
            continue;
        }
        denominator += element.weight;
        dispersion += element.weight * sample.aux.get(index).copied().unwrap_or(0.);
        for (k, value) in sum.iter_mut().enumerate() {
            *value += element.weight * sample.values[index * components + k];
        }
    }
    let (mean, weight) = match measured.normalization {
        _ if !records => (vec![0.; components], 0.),
        // Every valid element of an odd operator cancelled against its
        // mirror: the sum is exactly zero by construction, which is no
        // measurement of the operator under either normalisation.
        _ if measured.exchange == ExchangeParity::Odd && denominator <= 0. => {
            (vec![0.; components], 0.)
        }
        // Every pair of the frame ties: the score dispersion the values carry
        // sums to zero, so the average has no denominator of its own and the
        // frame states that instead of a series of exact zeros.
        _ if measured.auxiliary == Some(Auxiliary::ScoreDispersion) && dispersion <= 0. => {
            (vec![0.; components], 0.)
        }
        FrameNormalization::ValidCount if denominator > 0. => {
            (sum.iter().map(|x| x / denominator).collect(), 1.)
        }
        FrameNormalization::FixedN if walkers > 0 => {
            (sum.iter().map(|x| x / walkers as f64).collect(), 1.)
        }
        _ => (vec![0.; components], 0.),
    };
    sample.mean = mean;
    sample.weight = weight;
}

/// Slab observables of one frame, binned by the Euclidean-time coordinate of
/// the anchor walker. A value outside the range belongs to no slab; a
/// periodic axis wraps it.
#[allow(clippy::too_many_arguments)] // one slab row needs its whole axis contract
fn slab_sums(
    measured: &Measured,
    elements: &[Element],
    sample: &mut Sample,
    walkers: usize,
    state: &FrameState,
    range: [f64; 2],
    bins: usize,
    periodic: bool,
    records: bool,
) {
    let Some(times) = &state.euclidean_time else {
        return;
    };
    if !records {
        return;
    }
    let components = measured.components;
    let span = range[1] - range[0];
    let width = span / bins as f64;
    let mut numerator = vec![0.; bins * components];
    let mut denominator = vec![0.; bins];
    let mut dispersion = vec![0.; bins];
    for (index, element) in elements.iter().enumerate() {
        if !sample.valid[index] {
            continue;
        }
        let Some(&y) = times.get(element.walkers[0] as usize) else {
            continue;
        };
        let Some(bin) = slab_of(y, range[0], span, width, bins, periodic) else {
            continue;
        };
        denominator[bin] += element.weight;
        dispersion[bin] += element.weight * sample.aux.get(index).copied().unwrap_or(0.);
        for k in 0..components {
            numerator[bin * components + k] +=
                element.weight * sample.values[index * components + k];
        }
    }
    sample.slab = vec![0.; bins * components];
    sample.slab_weight = vec![false; bins];
    for b in 0..bins {
        // A slab whose pairs all tie has no score dispersion and no average,
        // exactly as the frame does.
        if measured.auxiliary == Some(Auxiliary::ScoreDispersion) && dispersion[b] <= 0. {
            continue;
        }
        match measured.normalization {
            FrameNormalization::ValidCount if denominator[b] > 0. => {
                for k in 0..components {
                    sample.slab[b * components + k] =
                        numerator[b * components + k] / denominator[b];
                }
                sample.slab_weight[b] = true;
            }
            FrameNormalization::ValidCount => {}
            FrameNormalization::FixedN if walkers > 0 => {
                for k in 0..components {
                    sample.slab[b * components + k] =
                        numerator[b * components + k] / walkers as f64;
                }
                sample.slab_weight[b] = true;
            }
            FrameNormalization::FixedN => {}
        }
    }
}

/// Bin of a Euclidean-time coordinate, `None` when it lies outside an open
/// axis or is not finite. The floor is taken once and checked, so a value one
/// ulp below the top never indexes past the last bin.
fn slab_of(y: f64, lo: f64, span: f64, width: f64, bins: usize, periodic: bool) -> Option<usize> {
    if !y.is_finite() || width <= 0. {
        return None;
    }
    let u = if periodic {
        (y - lo).rem_euclid(span) / width
    } else {
        (y - lo) / width
    };
    if !u.is_finite() || u < 0. {
        return None;
    }
    let bin = u.floor() as usize;
    match (periodic, bin >= bins) {
        (true, true) => Some(0),
        (false, true) => None,
        _ => Some(bin),
    }
}

/// Merge the slab blocks `(0,1), (2,3), …`; an odd trailing block stays alone.
fn merge_slab_pairs(slabs: &mut SlabMoments) {
    let (bins, components) = (slabs.bins, slabs.components);
    let blocks = slabs.blocks.div_ceil(2);
    let (pairs, rows) = (bins * bins, bins * components);
    let mut merged = SlabMoments {
        blocks,
        origins_per_block: slabs.origins_per_block.saturating_mul(2),
        pair_ab: vec![0.; blocks * pairs],
        pair_n: vec![0.; blocks * pairs],
        profile: vec![0.; blocks * rows],
        profile_n: vec![0.; blocks * bins],
        ..*slabs
    };
    for block in 0..slabs.blocks {
        let to = block / 2;
        for i in 0..pairs {
            merged.pair_ab[to * pairs + i] += slabs.pair_ab[block * pairs + i];
            merged.pair_n[to * pairs + i] += slabs.pair_n[block * pairs + i];
        }
        for i in 0..rows {
            merged.profile[to * rows + i] += slabs.profile[block * rows + i];
        }
        for i in 0..bins {
            merged.profile_n[to * bins + i] += slabs.profile_n[block * bins + i];
        }
    }
    *slabs = merged;
}

/// Push the lag sums of one ring entry into the moments of its channels,
/// creating them on the first origin.
fn push_origins(measurement: &mut Measurement, entry: &SourceFrame) -> Result<()> {
    let (lags, max_blocks) = (
        measurement.config.max_lag + 1,
        measurement.config.propagators.max_blocks,
    );
    for channel in &entry.channels {
        let Some(series) = measurement.channels.get_mut(channel.series) else {
            continue;
        };
        let components = series.components;
        let moments = match &mut series.propagator {
            Some(moments) => moments,
            none => none.insert(BlockMoments::new(lags, components, 1, max_blocks)?),
        };
        moments.push_origin(&channel.ab, &channel.a, &channel.b, &channel.n)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GasConfig,
        physics::spectroscopy::{
            config::{ElectroweakScales, PairDistance},
            contract::{Companions, OperatorContext, Requirements},
        },
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
    /// Three walkers on a line with the companions of the couplings fixture.
    fn fixture() -> (Frame, FrameState) {
        let companions = Companions {
            count: 1,
            slot: vec![1, 0, 1],
            generation: vec![0; 3],
            valid: vec![true; 3],
            historical: vec![false; 3],
            mutual: false,
        };
        let frame = Frame {
            step: 1,
            n: 3,
            d: 1,
            x: vec![0., 1., 3.],
            v: Some(vec![0.5, -0.25, 1.]),
            eligible: vec![true; 3],
            generation: vec![0; 3],
            distance: Some(companions),
            cloned: vec![false; 3],
            revived: vec![false; 3],
            ..Frame::default()
        };
        let state = FrameState {
            n: 3,
            d: 1,
            x: frame.x.clone(),
            v: frame.v.clone(),
            eligible: vec![true; 3],
            cloned: vec![false; 3],
            generation: vec![0; 3],
            ..FrameState::default()
        };
        (frame, state)
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
    #[test]
    fn the_calibration_pair_statistics_reproduce_the_couplings_fixture() {
        let (frame, state) = fixture();
        let gas = GasConfig::default();
        let distance = score::AmplitudeDistance::new(
            &gas,
            &gas.distance_donors.distance,
            &ElectroweakScales {
                distance: PairDistance::Raw,
                lambda: Some(0.),
                ..ElectroweakScales::default()
            },
        )
        .unwrap();
        let companions = frame.distance.as_ref().unwrap();
        let (mut sum, mut count) = (0., 0.);
        for i in 0..frame.n {
            let k = companions.first(i, &frame.eligible).unwrap();
            let j = companions.slot[i * companions.count + k] as usize;
            let squared = distance.squared(&state, i, j).unwrap();
            sum += (-squared / 4.).exp();
            count += 1.;
        }
        assert!((sum / count - 0.641827002438084).abs() < 1e-15);
        let (rho, mut sum, mut count) = (1.5, 0., 0.);
        for i in 0..frame.n {
            for j in (0..frame.n).filter(|&j| j != i) {
                let delta = frame.x[i] - frame.x[j];
                let kernel = (-delta * delta / (2. * rho * rho)).exp();
                sum += kernel * kernel;
                count += 1.;
            }
        }
        assert!((sum / count - 0.276169780908252).abs() < 1e-15);
    }
    #[test]
    fn a_slab_bin_never_indexes_past_the_axis() {
        let (lo, span, bins) = (-1., 4., 4);
        let width = span / bins as f64;
        assert_eq!(slab_of(-1., lo, span, width, bins, false), Some(0));
        assert_eq!(
            slab_of(2.9999999999999996, lo, span, width, bins, false),
            Some(3)
        );
        assert_eq!(slab_of(3., lo, span, width, bins, false), None);
        assert_eq!(slab_of(-1.5, lo, span, width, bins, false), None);
        assert_eq!(slab_of(f64::NAN, lo, span, width, bins, false), None);
        assert_eq!(slab_of(3., lo, span, width, bins, true), Some(0));
        assert_eq!(
            slab_of(-1.0000000000000002, lo, span, width, bins, true),
            Some(0)
        );
        assert_eq!(slab_of(-2.5, lo, span, width, bins, true), Some(2));
    }
    #[test]
    fn the_continuity_fingerprint_follows_accepted_clones_only() {
        let after = |generations: [u64; 3], cloned: [bool; 3]| {
            fingerprint(
                3,
                2,
                generations
                    .iter()
                    .zip(&cloned)
                    .map(|(generation, &cloned)| generation + u64::from(cloned)),
            )
        };
        let plain = |generations: [u64; 3]| fingerprint(3, 2, generations.into_iter());
        assert_eq!(after([0, 1, 2], [false, true, false]), plain([0, 2, 2]));
        assert_ne!(after([0, 1, 2], [false, true, false]), plain([0, 1, 2]));
        assert_ne!(plain([0, 1, 2]), fingerprint(3, 3, [0, 1, 2].into_iter()));
    }
}
