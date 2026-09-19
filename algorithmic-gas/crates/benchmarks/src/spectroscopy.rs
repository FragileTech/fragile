//! Resumable spectroscopy runs shared by the CLI and WASM: replicas of one
//! gas configuration streamed through `spectroscopy::Accumulator`, with
//! evidence that can be re-analysed without running again.
use crate::RunConfig;
use algorithmic_gas::{
    GasError, Precision, Result, RunArchive,
    physics::spectroscopy::{
        AnalysisConfig, Availability, Capabilities, Measurement, SPECTROSCOPY_VERSION,
        SpectroscopyConfig, SpectroscopyReport,
        report::{Calibration, Coverage, EstimatorKind},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    pub fn validate(&self) -> Result<()> {
        self.run.validate()?;
        self.spectroscopy.validate()?;
        if self.run.gas.precision != Precision::F64 {
            return Err(GasError::Capability(
                "spectroscopy requires an f64 run: set run.gas.precision to f64".into(),
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
    /// undefined, empty without an estimator.
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
    /// Every requested channel; live curves for the first few available ones.
    pub channels: Vec<LiveChannel>,
    pub notes: Vec<String>,
}
pub struct SpectroscopySession {
    request: SpectroscopyRequest,
}
impl SpectroscopySession {
    /// Validates the request, builds `replicas` f64 gases with seeds
    /// `seed + r · 104729` and one `Accumulator` each. `GasError::Capability`
    /// when no requested channel can be measured on this configuration.
    pub async fn create(request: SpectroscopyRequest) -> Result<Self> {
        request.validate()?;
        let _ = Self { request };
        Err(GasError::Capability(
            "pending: benchmarks::spectroscopy".into(),
        ))
    }
    /// Advance every live replica by up to `count` (`1..=64`) engine steps in
    /// recorded chunks of `request.chunk`, ingesting and dropping each chunk.
    /// An extinct population ends its replica with a terminal reason, as in
    /// `LectureSession`. Returns `snapshot()`.
    pub async fn advance(&mut self, count: usize) -> Result<Value> {
        if !(1..=64).contains(&count) {
            return Err(err("Advance batch must be 1..64"));
        }
        Err(GasError::Capability(
            "pending: benchmarks::spectroscopy".into(),
        ))
    }
    /// `SessionSnapshot` as JSON.
    pub fn snapshot(&self) -> Result<Value> {
        Err(GasError::Capability(
            "pending: benchmarks::spectroscopy".into(),
        ))
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
            configs: vec![],
            measurements: vec![],
        }
    }
    /// CBOR of the request, every replica's engine checkpoint, accumulator
    /// state and terminal reason.
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        Err(GasError::Capability(
            "pending: benchmarks::spectroscopy".into(),
        ))
    }
    /// Decode is capped at 256 MiB and rejects trailing data; the restored
    /// session continues bit-identically.
    pub async fn restore(bytes: &[u8]) -> Result<Self> {
        let _ = bytes;
        Err(GasError::Capability(
            "pending: benchmarks::spectroscopy".into(),
        ))
    }
    pub fn done(&self) -> bool {
        false
    }
}

/// Re-analyse stored evidence. Equals `SpectroscopySession::analyze` of the
/// session that produced it. The payload is not trusted: the request is
/// validated again, every replica configuration must be the request's run with
/// the replica seed, and every measurement must be one of that gas and
/// measurement configuration.
pub fn analyze_evidence(
    evidence: &SpectroscopyEvidence,
    analysis: &AnalysisConfig,
) -> Result<SpectroscopyReport> {
    evidence.request.validate()?;
    if evidence.schema_version != SPECTROSCOPY_VERSION
        || evidence.configs.len() != evidence.measurements.len()
        || evidence.configs.len() > evidence.request.replicas
    {
        return Err(err("Invalid spectroscopy evidence"));
    }
    for (replica, (config, measurement)) in evidence
        .configs
        .iter()
        .zip(&evidence.measurements)
        .enumerate()
    {
        let mut expected = evidence.request.run.clone();
        expected.gas.seed = evidence.request.replica_seed(replica);
        if *config != expected
            || measurement.gas != config.gas
            || measurement.config != evidence.request.spectroscopy.measurement
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
    Err(GasError::Capability(
        "pending: benchmarks::spectroscopy".into(),
    ))
}

/// `{capabilities, channels: [{id, spec, availability, requested}], chunk}` of
/// a request, before any step: what the page greys out. `channels` reduces
/// `operators::catalog(&context_of_request, &specs, &analysis.assignments)`
/// with `specs = ChannelSpec::all()` followed by the requested specifications
/// it lacks, so every selectable channel is listed, not only the requested ones.
pub fn capabilities(request: &SpectroscopyRequest) -> Result<Value> {
    request.validate()?;
    Err(GasError::Capability(
        "pending: benchmarks::spectroscopy".into(),
    ))
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
}
