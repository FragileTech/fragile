//! Resumable spectroscopy runs shared by the CLI and WASM: replicas of one
//! gas configuration streamed through `spectroscopy::Accumulator`, with
//! evidence that can be re-analysed without running again.
use crate::RunConfig;
use algorithmic_gas::{
    AlgorithmicGas, BackendKind, Checkpoint, GasError, Precision, RecordingConfig, Result,
    RunArchive,
    physics::{
        numerics::Subtraction,
        spectroscopy::{
            Accumulator, AccumulatorState, AnalysisConfig, Availability, Capabilities,
            ChannelSeries, ChannelSpec, Measurement, OperatorContext, SPECTROSCOPY_VERSION,
            SpectroscopyConfig, SpectroscopyReport,
            config::{EffectiveMassKind, EstimatorChoice, TimeUnit},
            estimators,
            fits::effective_mass::cosh_rate,
            frame::position_field,
            operators::{self, CatalogEntry},
            report::{Calibration, Coverage, EstimatorKind},
            select_estimator,
        },
    },
    variants::Variant,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// Session-level failures, CBOR and serde errors included, are configuration
// errors with sentence-case messages, as in the lecture session.
fn err(s: impl Into<String>) -> GasError {
    GasError::Configuration(s.into())
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpectroscopyRequest {
    /// Name of the variant `run` was derived from; a label for the variant
    /// selector. `run` is authoritative.
    pub variant: Option<String>,
    pub run: RunConfig,
    /// Engine steps per replica, warm-up included.
    pub steps: usize,
    /// Independently seeded runs. The default 4 is the fewest whose pooled
    /// errors a report states without the note that they are not replica
    /// standard errors.
    pub replicas: usize,
    /// Replica `r` runs with `seed + r · 104729`; `run.gas.seed` is ignored.
    pub seed: u64,
    /// Recorded steps per ingested chunk, `1..=32`, and at most 4 under dense
    /// viscosity, which records `2 N (N − 1)` influences per step.
    pub chunk: usize,
    pub spectroscopy: SpectroscopyConfig,
}
impl Default for SpectroscopyRequest {
    fn default() -> Self {
        let mut run =
            RunConfig::einstein_hilbert().expect("Einstein-Hilbert preset constants are valid");
        run.gas.precision = Precision::F64;
        Self {
            variant: Some("einstein_hilbert".into()),
            run: RunConfig {
                walkers: 200,
                ..run
            },
            steps: 2000,
            replicas: 4,
            seed: 7,
            chunk: 16,
            spectroscopy: SpectroscopyConfig::default(),
        }
    }
}
impl SpectroscopyRequest {
    /// Completes a request that names a variant and states nothing about the
    /// run beyond its size: `{"variant":"euclidean","run":{"walkers":128,
    /// "dimensions":3}}` becomes that variant's reference instance rebuilt in
    /// three dimensions, box domain included, which is all the browser sends.
    /// A run differing from `RunConfig::default()` in anything else is
    /// authoritative and kept as written, so resolving twice changes nothing.
    /// A size written as `RunConfig::default()` carries it cannot be told from
    /// an unstated one and keeps the reference instance's, so
    /// `{"variant":"einstein_hilbert","run":{"dimensions":2}}` stays in the
    /// three dimensions of its reference instance.
    ///
    /// `GasConfig::einstein_hilbert` inherits the engine default
    /// `Precision::F32` while spectroscopy is f64 only, so a run rebuilt here
    /// is set to f64 explicitly. A run the caller wrote in f32 is left alone
    /// and `validate` refuses it, as it refuses a run on an accelerator: the
    /// alternative is the silent fallback the engine does not do.
    pub fn resolve(mut self) -> Result<Self> {
        let Some(name) = self.variant.clone() else {
            return Ok(self);
        };
        let variant = Variant::from_name(&name)?;
        let default = RunConfig::default();
        let stated = RunConfig {
            walkers: default.walkers,
            dimensions: default.dimensions,
            ..self.run.clone()
        };
        if stated != default {
            return Ok(self);
        }
        let mut run = RunConfig::variant(&name)?;
        let reference = variant.reference().ok_or_else(|| {
            GasError::Capability(format!("the {} has no reference instance", variant.title()))
        })?;
        if self.run.walkers != default.walkers {
            run.walkers = self.run.walkers;
        }
        if self.run.dimensions != default.dimensions {
            run.dimensions = self.run.dimensions;
            run.gas = variant.config(self.run.dimensions, reference.dt)?;
        }
        run.gas.precision = Precision::F64;
        self.run = run;
        Ok(self)
    }
    pub fn validate(&self) -> Result<()> {
        self.run.validate()?;
        self.spectroscopy.validate()?;
        if self.run.gas.precision != Precision::F64 || self.run.gas.backend != BackendKind::Cpu {
            return Err(GasError::Capability(
                "spectroscopy analyses an f64 CPU run: set run.gas.precision to f64 and \
                 run.gas.backend to cpu"
                    .into(),
            ));
        }
        if !(1..=32).contains(&self.replicas)
            || !(1..=10_000_000).contains(&self.steps)
            || !(1..=32).contains(&self.chunk)
            || self.seed.checked_add(32 * 104_729).is_none()
        {
            return Err(err(
                "Spectroscopy budgets: replicas 1..32, steps 1..10000000, chunk 1..32",
            ));
        }
        if self.capabilities().dense_viscosity && self.chunk > 4 {
            return Err(err("Dense viscosity requires chunk 1..4"));
        }
        Ok(())
    }
    pub fn replica_seed(&self, replica: usize) -> u64 {
        self.seed
            .wrapping_add((replica as u64).wrapping_mul(104_729))
    }
    /// Static capabilities of the request, before any step.
    pub fn capabilities(&self) -> Capabilities {
        let recording = algorithmic_gas::RecordingConfig {
            graph: self.run.gas.geometry.is_some(),
            ..Default::default()
        };
        Capabilities::of(&self.run.gas, &recording, self.run.dimensions)
            .refine(&self.spectroscopy.measurement)
    }
    /// Recording of one ingested chunk. Only `graph` reaches `Capabilities`,
    /// so it repeats the flag of `capabilities`; the byte budget is the
    /// session budget of the lecture runner.
    fn recording(&self) -> RecordingConfig {
        RecordingConfig {
            max_steps: self.chunk,
            max_bytes: 256 * 1024 * 1024,
            graph: self.run.gas.geometry.is_some(),
        }
    }
}

/// Everything needed to repeat the analysis of a finished or running session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectroscopyEvidence {
    pub schema_version: u32,
    pub request: SpectroscopyRequest,
    /// Resolved configuration of every replica.
    pub configs: Vec<RunConfig>,
    pub measurements: Vec<Measurement>,
}
impl SpectroscopyEvidence {
    /// CBOR; decode is capped at 256 MiB and rejects trailing data.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![];
        ciborium::into_writer(self, &mut bytes).map_err(|e| err(e.to_string()))?;
        Ok(bytes)
    }
    /// Structure only; `analyze_evidence` validates the content.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(err("Spectroscopy evidence too large"));
        }
        let mut input = bytes;
        let evidence: Self = ciborium::de::from_reader_with_recursion_limit(&mut input, 64)
            .map_err(|e| err(e.to_string()))?;
        if !input.is_empty()
            || evidence.schema_version != SPECTROSCOPY_VERSION
            || evidence.configs.len() != evidence.measurements.len()
        {
            return Err(err("Invalid spectroscopy evidence"));
        }
        Ok(evidence)
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReplicaStatus {
    pub seed: u64,
    pub step: u64,
    /// Measured frames and contiguous segments so far.
    pub frames: u64,
    pub segments: u32,
    /// Why the replica stopped early, e.g. `extinct_population`.
    pub terminal: Option<String>,
}

/// Walkers of the first replica, for the swarm view.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WalkerCloud {
    pub dimension: usize,
    /// `[walkers, dimension]`; nonfinite coordinates are replaced by 0 and
    /// marked ineligible.
    pub positions: Vec<f64>,
    pub eligible: Vec<bool>,
}

/// Live, unresampled view of one channel while the session runs. The
/// estimator is the one `spectroscopy::select_estimator(.., Auto)` chooses, so
/// an exchange-odd channel on a mutual pairing shows its source-frozen
/// propagator and never the roundoff of a cancelled frame mean.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LiveChannel {
    pub id: String,
    pub availability: Availability,
    pub coverage: Coverage,
    /// `None` when `select_estimator` found none (`note` holds its reason) and
    /// for channels that are unavailable or not correlatable.
    pub estimator: Option<EstimatorKind>,
    /// Pooled connected correlator of `estimator` per lag; `None` where
    /// undefined, empty without an estimator and beyond the live-curve budget.
    pub correlator: Vec<Option<f64>>,
    /// Log-ratio effective rate per lag.
    pub effective_mass: Vec<Option<f64>>,
    /// The note or the refusal of `select_estimator`.
    pub note: Option<String>,
}

/// What `advance` and `snapshot` return, as JSON.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionSnapshot {
    pub schema_version: u32,
    /// Steps taken by the slowest live replica, and the request's total.
    pub step: u64,
    pub steps: u64,
    pub done: bool,
    pub chunk: usize,
    pub replicas: Vec<ReplicaStatus>,
    pub capabilities: Capabilities,
    pub calibration: Option<Calibration>,
    pub walkers: WalkerCloud,
    /// Every requested channel with the estimator it would be read with; live
    /// curves for the first few that yield one.
    pub channels: Vec<LiveChannel>,
    pub notes: Vec<String>,
}
/// Selecting an estimator costs one pass over the measured weights and every
/// channel carries it, but a live curve costs one pass per lag and a snapshot
/// is drawn inside one browser frame, so only the first few channels that
/// yield a curve carry one.
const LIVE_CURVES: usize = 4;
/// One independently seeded run and the accumulator measuring it.
struct Replica {
    seed: u64,
    config: RunConfig,
    gas: AlgorithmicGas<f64>,
    accumulator: Accumulator,
    terminal: Option<String>,
}
impl Replica {
    /// The measurement with its pending propagator origins flushed. A state
    /// that cannot be finalized is carried unflushed instead, and
    /// `analyze_evidence` then refuses it through `Measurement::validate`.
    fn measurement(&self) -> Measurement {
        self.accumulator
            .finalized()
            .unwrap_or_else(|_| self.accumulator.measurement().clone())
    }
    fn status(&self) -> ReplicaStatus {
        let live = self.accumulator.measurement();
        ReplicaStatus {
            seed: self.seed,
            step: self.gas.step_number(),
            frames: live.frames() as u64,
            segments: live.segments,
            terminal: self.terminal.clone(),
        }
    }
}
/// Everything a checkpointed session restores from. The live view is derived
/// from the accumulators and is never stored.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedSession {
    schema_version: u32,
    request: SpectroscopyRequest,
    states: Vec<Checkpoint<f64>>,
    accumulators: Vec<AccumulatorState>,
    terminal: Vec<Option<String>>,
}
pub struct SpectroscopySession {
    request: SpectroscopyRequest,
    recording: RecordingConfig,
    replicas: Vec<Replica>,
}
impl SpectroscopySession {
    /// Resolves and validates the request, builds `replicas` f64 gases with
    /// seeds `seed + r · 104729` and one `Accumulator` each.
    /// `GasError::Capability` when no requested channel can be measured on
    /// this configuration.
    pub async fn create(request: SpectroscopyRequest) -> Result<Self> {
        let request = request.resolve()?;
        request.validate()?;
        let recording = request.recording();
        let mut replicas = Vec::with_capacity(request.replicas);
        for replica in 0..request.replicas {
            let mut config = request.run.clone();
            config.gas.seed = request.replica_seed(replica);
            let gas = config.build::<f64>().await?;
            let dimension = gas
                .population()
                .observations
                .field(position_field(&config.gas))?
                .width();
            let accumulator = Accumulator::new(
                request.spectroscopy.measurement.clone(),
                &config.gas,
                &recording,
                dimension,
            )?;
            replicas.push(Replica {
                seed: config.gas.seed,
                config,
                gas,
                accumulator,
                terminal: None,
            });
        }
        Ok(Self {
            request,
            recording,
            replicas,
        })
    }
    /// Advance every live replica by up to `count` (`1..=64`) engine steps in
    /// recorded chunks of `request.chunk`, ingesting and dropping each chunk,
    /// so that the retained bytes are one chunk and the measurement instead of
    /// the whole run. The chunks continue one stream: the accumulator sees the
    /// same steps in the same order as one archive of the whole run. An
    /// extinct population ends its replica with a terminal reason, as in
    /// `LectureSession`. Returns `snapshot()`.
    pub async fn advance(&mut self, count: usize) -> Result<Value> {
        if !(1..=64).contains(&count) {
            return Err(err("Advance batch must be 1..64"));
        }
        let (steps, chunk) = (self.request.steps as u64, self.request.chunk);
        for replica in &mut self.replicas {
            let mut taken = 0;
            while taken < count && replica.terminal.is_none() && replica.gas.step_number() < steps {
                let remaining = (steps - replica.gas.step_number()) as usize;
                let budget = chunk.min(count - taken).min(remaining);
                replica.gas.start_recording(RecordingConfig {
                    max_steps: budget,
                    ..self.recording
                })?;
                for _ in 0..budget {
                    match replica.gas.step().await {
                        Ok(_) => taken += 1,
                        Err(GasError::Extinction)
                            if !replica
                                .gas
                                .population()
                                .eligible(replica.gas.config().include_truncated)
                                .iter()
                                .any(|a| *a) =>
                        {
                            replica.terminal = Some("extinct_population".into());
                            break;
                        }
                        Err(e) => {
                            replica.gas.stop_recording();
                            return Err(e);
                        }
                    }
                }
                let archive = replica
                    .gas
                    .stop_recording()
                    .ok_or_else(|| err("Chunk recording ended before it was archived"))?;
                replica.accumulator.ingest_archive(&archive)?;
            }
        }
        self.snapshot()
    }
    /// `SessionSnapshot` as JSON: the live, unresampled view. Its correlators
    /// are the pooled moments of the measured frames, so they carry no errors
    /// and no fit; `analyze` is the reported result.
    pub fn snapshot(&self) -> Result<Value> {
        let first = self
            .replicas
            .first()
            .ok_or_else(|| err("Session has no replica"))?;
        let live: Vec<Measurement> = self
            .replicas
            .iter()
            .map(|r| r.accumulator.measurement().clone())
            .collect();
        let analysis = &self.request.spectroscopy.analysis;
        let mut curves = 0;
        let mut channels = Vec::with_capacity(first.accumulator.measurement().channels.len());
        for (index, series) in first.accumulator.measurement().channels.iter().enumerate() {
            let channel = live_channel(&live, index, series, analysis, curves < LIVE_CURVES);
            curves += usize::from(!channel.correlator.is_empty());
            channels.push(channel);
        }
        let mut notes = vec![];
        for measurement in &live {
            for note in &measurement.notes {
                if !notes.contains(note) {
                    notes.push(note.clone());
                }
            }
        }
        let snapshot = SessionSnapshot {
            schema_version: SPECTROSCOPY_VERSION,
            step: self.step(),
            steps: self.request.steps as u64,
            done: self.done(),
            chunk: self.request.chunk,
            replicas: self.replicas.iter().map(Replica::status).collect(),
            capabilities: first.accumulator.measurement().capabilities.clone(),
            calibration: first.accumulator.measurement().calibration.clone(),
            walkers: walker_cloud(&first.gas, &first.config)?,
            channels,
            notes,
        };
        serde_json::to_value(snapshot).map_err(|e| err(e.to_string()))
    }
    /// Engine steps of the slowest replica that is still running, or of the
    /// furthest one once every replica has stopped.
    fn step(&self) -> u64 {
        let live = self
            .replicas
            .iter()
            .filter(|r| r.terminal.is_none())
            .map(|r| r.gas.step_number())
            .min();
        live.unwrap_or_else(|| {
            self.replicas
                .iter()
                .map(|r| r.gas.step_number())
                .max()
                .unwrap_or(0)
        })
    }
    pub fn request(&self) -> &SpectroscopyRequest {
        &self.request
    }
    pub fn analyze(&self, analysis: &AnalysisConfig) -> Result<SpectroscopyReport> {
        analyze_evidence(&self.evidence(), analysis)
    }
    pub fn evidence(&self) -> SpectroscopyEvidence {
        SpectroscopyEvidence {
            schema_version: SPECTROSCOPY_VERSION,
            request: self.request.clone(),
            configs: self.replicas.iter().map(|r| r.config.clone()).collect(),
            measurements: self.replicas.iter().map(Replica::measurement).collect(),
        }
    }
    /// CBOR of the request, every replica's engine checkpoint, accumulator
    /// state and terminal reason.
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        let saved = SavedSession {
            schema_version: SPECTROSCOPY_VERSION,
            request: self.request.clone(),
            states: self.replicas.iter().map(|r| r.gas.checkpoint()).collect(),
            accumulators: self
                .replicas
                .iter()
                .map(|r| r.accumulator.state())
                .collect(),
            terminal: self.replicas.iter().map(|r| r.terminal.clone()).collect(),
        };
        let mut bytes = vec![];
        ciborium::into_writer(&saved, &mut bytes).map_err(|e| err(e.to_string()))?;
        Ok(bytes)
    }
    /// Decode is capped at 256 MiB and rejects trailing data; the restored
    /// session continues bit-identically. Every replica's engine
    /// configuration and measured gas must be the request's run with that
    /// replica's seed, so a re-seeded or foreign state is refused rather than
    /// measured on.
    pub async fn restore(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(err("Spectroscopy checkpoint too large"));
        }
        let mut input = bytes;
        let saved: SavedSession = ciborium::de::from_reader_with_recursion_limit(&mut input, 64)
            .map_err(|e| err(e.to_string()))?;
        if !input.is_empty()
            || saved.schema_version != SPECTROSCOPY_VERSION
            || saved.states.is_empty()
            || saved.accumulators.len() != saved.states.len()
            || saved.terminal.len() != saved.states.len()
        {
            return Err(err("Invalid spectroscopy checkpoint"));
        }
        let request = saved.request.resolve()?;
        request.validate()?;
        if saved.states.len() != request.replicas {
            return Err(err("Invalid spectroscopy checkpoint"));
        }
        let recording = request.recording();
        let mut replicas = Vec::with_capacity(request.replicas);
        for (replica, ((state, accumulator), terminal)) in saved
            .states
            .into_iter()
            .zip(saved.accumulators)
            .zip(saved.terminal)
            .enumerate()
        {
            let mut config = request.run.clone();
            config.gas.seed = request.replica_seed(replica);
            if config.gas != state.config || config.gas != accumulator.measurement.gas {
                return Err(err("Checkpoint configuration mismatch"));
            }
            let mut gas = config.build::<f64>().await?;
            gas.restore(state)?;
            replicas.push(Replica {
                seed: config.gas.seed,
                config,
                gas,
                accumulator: Accumulator::restore(accumulator)?,
                terminal,
            });
        }
        Ok(Self {
            request,
            recording,
            replicas,
        })
    }
    pub fn done(&self) -> bool {
        let steps = self.request.steps as u64;
        self.replicas
            .iter()
            .all(|r| r.terminal.is_some() || r.gas.step_number() >= steps)
    }
}

/// Walkers of one replica for the swarm view. A row with a nonfinite
/// coordinate is reported at the origin and marked ineligible, so the page
/// never plots a placeholder it cannot recognise.
fn walker_cloud(gas: &AlgorithmicGas<f64>, config: &RunConfig) -> Result<WalkerCloud> {
    let population = gas.population();
    let field = population.observations.field(position_field(&config.gas))?;
    let dimension = field.width();
    let alive = population.eligible(config.gas.include_truncated);
    let mut positions = Vec::with_capacity(field.values().len());
    let mut eligible = Vec::with_capacity(alive.len());
    for (row, valid) in field.values().chunks_exact(dimension).zip(&alive) {
        let finite = row.iter().all(|x| x.is_finite());
        eligible.push(finite && *valid);
        positions.extend(row.iter().map(|x| if finite { *x } else { 0. }));
    }
    Ok(WalkerCloud {
        dimension,
        positions,
        eligible,
    })
}
/// The live view of channel `index` of every replica: merged coverage, the
/// estimator `Auto` would take on every available correlatable channel, and,
/// for `curves`, the pooled connected correlator and its effective rate. Lags
/// are frame lags, so a propagator measured every `lag_stride` frames leaves
/// the lags in between `None` instead of shifting the curve.
fn live_channel(
    measurements: &[Measurement],
    index: usize,
    series: &ChannelSeries,
    analysis: &AnalysisConfig,
    curves: bool,
) -> LiveChannel {
    let mut coverage = Coverage::default();
    for measurement in measurements {
        if let Some(member) = measurement
            .channels
            .get(index)
            .filter(|c| c.id == series.id)
        {
            coverage.merge(&member.coverage);
        }
    }
    let mut live = LiveChannel {
        id: series.id.clone(),
        availability: series.availability.clone(),
        coverage,
        ..LiveChannel::default()
    };
    if !series.correlatable || !series.availability.is_available() {
        return live;
    }
    let selection = select_estimator(measurements, series, EstimatorChoice::Auto);
    live.note = selection.note;
    let Some(kind) = selection.estimator else {
        return live;
    };
    live.estimator = Some(kind);
    if !curves {
        return live;
    }
    let subtraction = match (analysis.connected, kind) {
        (false, _) => Subtraction::None,
        (true, EstimatorKind::FrameMean) => analysis.frame_subtraction,
        (true, _) => analysis.propagator_subtraction,
    };
    let Ok(moments) = estimators::moments(measurements, &series.id, kind, analysis) else {
        return live;
    };
    live.correlator = moments.estimate(subtraction);
    if let Some(measurement) = measurements.first()
        && let Some(step) = live_step(measurement, analysis.time_unit)
    {
        live.effective_mass = live_rates(&live.correlator, analysis.effective_mass, step);
    }
    live
}
/// Lag spacing of the live curves in the analysis time unit, as the estimator
/// takes it: one per frame in `Frames`, `stride · dt` in `StepDt`, and none
/// on a coordinate axis, which belongs to the Euclidean-time estimator alone.
fn live_step(measurement: &Measurement, unit: TimeUnit) -> Option<f64> {
    match unit {
        TimeUnit::Frames => Some(1.),
        TimeUnit::StepDt => {
            let step = measurement.config.stride as f64 * measurement.calibration.as_ref()?.dt?;
            (step.is_finite() && step > 0.).then_some(step)
        }
        TimeUnit::Coordinate => None,
    }
}
/// Effective rate at every lag of the live correlator, without an error: the
/// rate at a lag reads the next defined lag and its actual spacing, so an
/// undefined lag widens the interval instead of being skipped. A non-positive
/// correlator is not a decaying one and has no rate.
fn live_rates(correlator: &[Option<f64>], kind: EffectiveMassKind, step: f64) -> Vec<Option<f64>> {
    let next = |from: usize| (from..correlator.len()).find(|&k| correlator[k].is_some());
    let previous = |before: usize| (0..before).rev().find(|&k| correlator[k].is_some());
    (0..correlator.len())
        .map(|k| {
            let near = correlator[k]?;
            let far = next(k + 1)?;
            let width = (far - k) as f64 * step;
            if near <= 0. || width <= 0. {
                return None;
            }
            match kind {
                EffectiveMassKind::LogRatio => {
                    let next = correlator[far]?;
                    (next > 0.).then(|| (near / next).ln() / width)
                }
                EffectiveMassKind::Cosh => {
                    let back = previous(k)?;
                    ((k - back) == (far - k))
                        .then(|| cosh_rate((correlator[back]? + correlator[far]?) / (2. * near)))
                        .flatten()
                        .map(|rate| rate / width)
                }
            }
        })
        .collect()
}

/// Re-analyse stored evidence. Equals `SpectroscopySession::analyze` of the
/// session that produced it. The payload is not trusted: the request is
/// resolved and validated again, every replica configuration must be the
/// resolved run with the replica seed, and every measurement must be one of
/// that gas and measurement configuration.
pub fn analyze_evidence(
    evidence: &SpectroscopyEvidence,
    analysis: &AnalysisConfig,
) -> Result<SpectroscopyReport> {
    let request = evidence.request.clone().resolve()?;
    request.validate()?;
    if evidence.schema_version != SPECTROSCOPY_VERSION
        || evidence.configs.len() != evidence.measurements.len()
        || evidence.configs.len() > request.replicas
    {
        return Err(err("Invalid spectroscopy evidence"));
    }
    for (replica, (config, measurement)) in evidence
        .configs
        .iter()
        .zip(&evidence.measurements)
        .enumerate()
    {
        let mut expected = request.run.clone();
        expected.gas.seed = request.replica_seed(replica);
        if *config != expected
            || measurement.gas != config.gas
            || measurement.config != request.spectroscopy.measurement
        {
            return Err(err("Evidence configuration mismatch"));
        }
    }
    algorithmic_gas::physics::spectroscopy::analyze(&evidence.measurements, analysis)
}

/// Measure and analyse complete archives of replicas of one configuration.
pub fn analyze_archive(
    config: &SpectroscopyConfig,
    archives: &[RunArchive<f64>],
) -> Result<SpectroscopyReport> {
    config.validate()?;
    let measurements = archives
        .iter()
        .map(|a| algorithmic_gas::physics::spectroscopy::measure_archive(&config.measurement, a))
        .collect::<Result<Vec<_>>>()?;
    algorithmic_gas::physics::spectroscopy::analyze(&measurements, &config.analysis)
}

/// `{schema_version, request, variants, catalog, reference}`: the default
/// request, one default request per implemented variant of the registry, the
/// channel catalog (`operators::catalog` over the same specifications as
/// `capabilities`) with requirements and availability for the default
/// request, and the reference table.
pub fn defaults() -> Result<Value> {
    let request = SpectroscopyRequest::default();
    let mut variants = vec![];
    for info in algorithmic_gas::variants::catalog() {
        let mut row = serde_json::to_value(&info).map_err(|e| err(e.to_string()))?;
        row["request"] = if info.implemented {
            serde_json::to_value(variant_request(info.name)?).map_err(|e| err(e.to_string()))?
        } else {
            Value::Null
        };
        variants.push(row);
    }
    Ok(
        json!({"schema_version":SPECTROSCOPY_VERSION,"request":request,"variants":variants,"catalog":catalog_entries(&request)?,"reference":request.spectroscopy.analysis.reference}),
    )
}
/// The default request on the reference instance of one implemented variant.
/// A dense viscous kernel records `2 N (N − 1)` influences per step, so its
/// chunk is the shortest the budget of `validate` admits.
fn variant_request(name: &str) -> Result<SpectroscopyRequest> {
    let mut request = SpectroscopyRequest {
        variant: Some(name.to_string()),
        run: RunConfig::default(),
        ..SpectroscopyRequest::default()
    }
    .resolve()?;
    if request.capabilities().dense_viscosity {
        request.chunk = 4;
    }
    request.validate()?;
    Ok(request)
}
/// Every selectable channel under the request's own context: the built-in
/// specifications followed by the requested ones they lack, so a channel the
/// caller added by hand is listed with its availability too.
fn catalog_entries(request: &SpectroscopyRequest) -> Result<Vec<CatalogEntry>> {
    let capabilities = request.capabilities();
    let context = OperatorContext {
        gas: &request.run.gas,
        measurement: &request.spectroscopy.measurement,
        capabilities: &capabilities,
    };
    let mut specs = ChannelSpec::all();
    for spec in &request.spectroscopy.measurement.channels {
        if !specs.iter().any(|known| known.id() == spec.id()) {
            specs.push(spec.clone());
        }
    }
    operators::catalog(&context, &specs, &request.spectroscopy.analysis.assignments)
}

/// `{capabilities, channels: [{id, spec, availability, requested}], chunk}` of
/// a request, before any step: what the page greys out. `channels` reduces
/// `operators::catalog(&context_of_request, &specs, &analysis.assignments)`
/// with `specs = ChannelSpec::all()` followed by the requested specifications
/// it lacks, so every selectable channel is listed, not only the requested ones.
pub fn capabilities(request: &SpectroscopyRequest) -> Result<Value> {
    let request = request.clone().resolve()?;
    request.validate()?;
    let channels: Vec<Value> = catalog_entries(&request)?
        .into_iter()
        .map(|entry| {
            let requested = request
                .spectroscopy
                .measurement
                .channels
                .iter()
                .any(|spec| spec.id() == entry.spec.id());
            json!({"id":entry.id,"spec":entry.spec,"availability":entry.availability,"requested":requested})
        })
        .collect();
    Ok(json!({"capabilities":request.capabilities(),"channels":channels,"chunk":request.chunk}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_request_validates_and_round_trips_through_json() {
        let request = SpectroscopyRequest::default();
        request.validate().unwrap();
        assert_eq!(request.run.gas.precision, Precision::F64);
        assert_eq!(request.replicas, 4);
        assert_eq!(request.replica_seed(2), 7 + 2 * 104_729);
        let text = serde_json::to_string_pretty(&request).unwrap();
        let back: SpectroscopyRequest = serde_json::from_str(&text).unwrap();
        assert_eq!(back, request);
        assert!(serde_json::from_str::<SpectroscopyRequest>(r#"{"replica":2}"#).is_err());
        let caps = request.capabilities();
        assert!(caps.mutual_distance && caps.euclidean_axis == Some(2));
    }
    #[test]
    fn dense_viscosity_requires_a_short_chunk_and_bad_requests_are_explicit() {
        let mut request = SpectroscopyRequest::default();
        request.run.gas.qft.graph_viscosity = None;
        request.run.gas.qft.curl = None;
        request.run.gas.qft.viscosity = Some(algorithmic_gas::kinetic::ViscousForceConfig {
            coefficient: 0.15,
            bandwidth: 1.,
            row_normalized: false,
        });
        assert!(request.capabilities().dense_viscosity);
        assert!(matches!(
            request.validate(),
            Err(GasError::Configuration(_))
        ));
        request.chunk = 4;
        request.validate().unwrap();
        request.replicas = 33;
        assert!(request.validate().is_err());
        let mut single = SpectroscopyRequest::default();
        single.run.gas.precision = Precision::F32;
        assert!(matches!(single.validate(), Err(GasError::Capability(_))));
        let mut accelerated = SpectroscopyRequest::default();
        accelerated.run.gas.backend = BackendKind::Wgpu;
        assert!(matches!(
            accelerated.validate(),
            Err(GasError::Capability(_))
        ));
    }
    #[test]
    fn snapshot_and_evidence_have_stable_serialized_shapes() {
        let snapshot = SessionSnapshot {
            schema_version: SPECTROSCOPY_VERSION,
            step: 48,
            steps: 2000,
            chunk: 16,
            replicas: vec![ReplicaStatus {
                seed: 7,
                step: 48,
                frames: 32,
                segments: 1,
                terminal: None,
            }],
            capabilities: SpectroscopyRequest::default().capabilities(),
            walkers: WalkerCloud {
                dimension: 3,
                positions: vec![0.1, -0.2, 0.05],
                eligible: vec![true],
            },
            channels: vec![
                LiveChannel {
                    id: "meson/scalar/standard/distance".into(),
                    coverage: Coverage {
                        frames: 32,
                        valid: 6400,
                        ..Coverage::default()
                    },
                    estimator: Some(EstimatorKind::FrameMean),
                    correlator: vec![Some(0.02), Some(0.016), None],
                    effective_mass: vec![Some(0.22), None, None],
                    ..LiveChannel::default()
                },
                LiveChannel {
                    id: "meson/pseudoscalar/standard/distance".into(),
                    coverage: Coverage {
                        frames: 32,
                        valid: 6400,
                        ..Coverage::default()
                    },
                    estimator: Some(EstimatorKind::SourceFrozen),
                    correlator: vec![Some(0.11), Some(0.07), None],
                    effective_mass: vec![Some(0.45), None, None],
                    note: Some(
                        "exchange-odd operator cancels on a mutual pairing: the frame mean is \
                         unavailable and the source-frozen propagator is reported instead"
                            .into(),
                    ),
                    ..LiveChannel::default()
                },
                LiveChannel {
                    id: "glueball/re_plaquette/p0_cos1/triplet".into(),
                    availability: Availability::unavailable(
                        "momentum projection needs a periodic box",
                    ),
                    ..LiveChannel::default()
                },
            ],
            ..SessionSnapshot::default()
        };
        let text = serde_json::to_string_pretty(&snapshot).unwrap();
        assert_eq!(
            serde_json::from_str::<SessionSnapshot>(&text).unwrap(),
            snapshot
        );
        let evidence = SpectroscopyEvidence {
            schema_version: SPECTROSCOPY_VERSION,
            request: SpectroscopyRequest::default(),
            configs: vec![],
            measurements: vec![],
        };
        let bytes = evidence.to_bytes().unwrap();
        assert_eq!(SpectroscopyEvidence::from_bytes(&bytes).unwrap(), evidence);
        let mut trailing = bytes;
        trailing.push(0);
        assert!(SpectroscopyEvidence::from_bytes(&trailing).is_err());
    }
    #[test]
    fn a_strided_live_correlator_takes_its_rate_over_the_lags_that_hold_a_pair() {
        let correlator = vec![
            Some(1.),
            None,
            Some((-2f64).exp()),
            None,
            Some((-4f64).exp()),
        ];
        let rates = live_rates(&correlator, EffectiveMassKind::LogRatio, 0.5);
        assert_eq!(rates.len(), correlator.len());
        assert!(rates[1].is_none() && rates[3].is_none() && rates[4].is_none());
        assert!((rates[0].unwrap() - 2.).abs() < 1e-12);
        assert!((rates[2].unwrap() - 2.).abs() < 1e-12);
        let cosh = live_rates(&correlator, EffectiveMassKind::Cosh, 0.5);
        assert!(cosh[0].is_none() && cosh[2].unwrap() > 0.);
        let negative = vec![Some(-1.), Some(1.), Some(1.)];
        let rates = live_rates(&negative, EffectiveMassKind::LogRatio, 1.);
        assert_eq!(rates, vec![None, Some(0.), None]);
    }
}
