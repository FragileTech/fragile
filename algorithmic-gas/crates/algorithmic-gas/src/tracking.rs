//! Durable observational archive. Recording never draws random numbers and commits atomically.
use crate::{GasConfig, GasError, Population, Real, Result, StepReport, Validity, error::require};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecordingConfig {
    pub max_steps: usize,
    pub max_bytes: usize,
}
impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            max_steps: 256,
            max_bytes: 128 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldSnapshot {
    pub item_shape: Vec<usize>,
    pub values: Vec<f64>,
}
/// Actual operator boundary snapshot, including validity after boundary handling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageSnapshot {
    pub stage: String,
    pub version: u64,
    pub generations: Vec<u64>,
    pub validity: Vec<Validity>,
    pub fields: BTreeMap<String, FieldSnapshot>,
}
impl StageSnapshot {
    fn matches_population<T: Real>(&self, p: &Population<T>) -> bool {
        self.version == p.version
            && self.generations == p.generations
            && self.validity == p.validity
            && self.fields.len() == p.observations.fields.len()
            && self.fields.iter().all(|(name, f)| {
                p.observations.fields.get(name).is_some_and(|v| {
                    f.item_shape == v.item_shape()
                        && f.values.len() == v.values().len()
                        && f.values
                            .iter()
                            .zip(v.values())
                            .all(|(&a, b)| a == b.to_f64() || (a.is_nan() && b.to_f64().is_nan()))
                })
            })
    }
    pub(crate) fn capture<T: Real>(stage: &str, p: &Population<T>) -> Self {
        Self {
            stage: stage.into(),
            version: p.version,
            generations: p.generations.clone(),
            validity: p.validity.clone(),
            fields: p
                .observations
                .fields
                .iter()
                .map(|(name, field)| {
                    (
                        name.clone(),
                        FieldSnapshot {
                            item_shape: field.item_shape().to_vec(),
                            values: field.values().iter().map(|x| x.to_f64()).collect(),
                        },
                    )
                })
                .collect(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldEvaluation {
    /// Empty is the legacy all-present encoding; otherwise one flag per row.
    #[serde(default)]
    pub available: Vec<bool>,
    pub stage: String,
    pub field: String,
    pub version: u64,
    pub rows: usize,
    pub item_shape: Vec<usize>,
    pub values: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InfluenceRecord {
    pub stage: String,
    pub field: String,
    pub recipient: u32,
    pub source: crate::donor::SourceRef,
    pub weight: f64,
}
/// Captured noise sample. `sample` is the actual B ξ before integrator time scaling.
/// `raw_innovation` and compact `factor` are available for built-in Noise providers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoiseSnapshot {
    pub stage: String,
    pub step: u64,
    pub stream: crate::random::Stream,
    pub substep: u64,
    pub rows: usize,
    pub dimension: usize,
    pub sample: Vec<f64>,
    pub raw_innovation: Option<Vec<f64>>,
    pub factor: Option<Vec<f64>>,
    pub geometry: Option<crate::noise::NoiseGeometry>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct ArchiveAnchor<T: Real> {
    pub epoch: u64,
    pub step: u64,
    /// Initial, recording_started, external_replace, or checkpoint_v2_migration.
    pub reason: String,
    pub population: Population<T>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct RecordedStep<T: Real> {
    pub epoch: u64,
    pub before: Population<T>,
    pub final_population: Population<T>,
    pub stages: Vec<StageSnapshot>,
    pub noise: Vec<NoiseSnapshot>,
    pub field_evaluations: Vec<FieldEvaluation>,
    pub influences: Vec<InfluenceRecord>,
    pub report: StepReport<T>,
    /// Actual pool-aligned inputs used for the clone decision (including historical rescoring).
    pub donor_fitness: Vec<T>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real", deny_unknown_fields)]
pub struct RunArchive<T: Real> {
    pub schema_version: u32,
    pub config: RecordingConfig,
    pub gas_config: GasConfig,
    #[serde(default)]
    pub providers: BTreeMap<String, String>,
    pub epoch: u64,
    pub anchors: Vec<ArchiveAnchor<T>>,
    pub steps: Vec<RecordedStep<T>>,
}
impl<T: Real> RunArchive<T> {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(self, &mut bytes)
            .map_err(|e| GasError::Checkpoint(e.to_string()))?;
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        require(
            bytes.len() <= 256 * 1024 * 1024,
            "archive decode exceeds 256 MiB",
        )?;
        let mut remaining = bytes;
        let archive: Self = ciborium::de::from_reader_with_recursion_limit(&mut remaining, 64)
            .map_err(|e| GasError::Checkpoint(e.to_string()))?;
        require(remaining.is_empty(), "trailing archive data")?;
        archive.validate()?;
        Ok(archive)
    }

    pub(crate) fn new(
        config: RecordingConfig,
        gas_config: GasConfig,
        step: u64,
        population: &Population<T>,
        reason: &str,
    ) -> Result<Self> {
        require(
            config.max_steps > 0 && config.max_bytes > 0,
            "recording budgets must be positive",
        )?;
        let archive = Self {
            schema_version: 1,
            config,
            gas_config,
            providers: BTreeMap::new(),
            epoch: 0,
            anchors: vec![ArchiveAnchor {
                epoch: 0,
                step,
                reason: reason.into(),
                population: population.clone(),
            }],
            steps: vec![],
        };
        archive.check_capacity()?;
        Ok(archive)
    }
    /// Conservative retained host-buffer size, independent of JSON encoding/nonfinite values.
    pub fn buffer_bytes(&self) -> Result<usize> {
        use crate::memory::{checked_add, checked_mul};
        let mut bytes = 4096;
        for a in &self.anchors {
            bytes = checked_add(bytes, checked_add(a.population.buffer_bytes()?, 512)?)?;
        }
        for s in &self.steps {
            bytes = checked_add(
                bytes,
                checked_add(s.before.buffer_bytes()?, s.final_population.buffer_bytes()?)?,
            )?;
            bytes = checked_add(bytes, checked_mul(s.before.len(), 1024)?)?;
            bytes = checked_add(
                bytes,
                checked_mul(
                    s.report.distance_sources.len() + s.report.clone_plan.sources.len(),
                    128,
                )?,
            )?;
            bytes = checked_add(
                bytes,
                checked_mul(
                    s.report.distance_companions.indices.len()
                        + s.report.cloning_companions.indices.len(),
                    32,
                )?,
            )?;
            bytes = checked_add(bytes, checked_mul(s.influences.len(), 256)?)?;
            for f in &s.field_evaluations {
                bytes = checked_add(bytes, checked_add(256, checked_mul(f.values.len(), 8)?)?)?;
            }
            for noise in &s.noise {
                bytes = checked_add(
                    bytes,
                    checked_mul(
                        noise.sample.len()
                            + noise.raw_innovation.as_ref().map_or(0, Vec::len)
                            + noise.factor.as_ref().map_or(0, Vec::len),
                        8,
                    )?,
                )?;
            }
            for stage in &s.stages {
                bytes = checked_add(bytes, checked_mul(stage.generations.len(), 128)?)?;
                for (name, f) in &stage.fields {
                    bytes = checked_add(
                        bytes,
                        checked_add(name.len() + 128, checked_mul(f.values.len(), 8)?)?,
                    )?;
                }
            }
        }
        Ok(bytes)
    }
    pub(crate) fn check_capacity(&self) -> Result<()> {
        require(
            self.config.max_steps > 0 && self.config.max_bytes > 0,
            "recording budgets must be positive",
        )?;
        require(
            self.steps.len() <= self.config.max_steps,
            "recording horizon reached; export archive and start a new recording",
        )?;
        crate::memory::enforce(self.buffer_bytes()?, self.config.max_bytes)
    }
    pub(crate) fn append(&mut self, step: RecordedStep<T>) -> Result<()> {
        self.steps.push(step);
        if let Err(e) = self.check_capacity() {
            self.steps.pop();
            return Err(e);
        }
        Ok(())
    }
    pub(crate) fn anchor(&mut self, step: u64, population: &Population<T>) -> Result<()> {
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| GasError::Numerical("archive epoch overflow".into()))?;
        self.anchors.push(ArchiveAnchor {
            epoch,
            step,
            reason: "external_replace".into(),
            population: population.clone(),
        });
        if let Err(e) = self.check_capacity() {
            self.anchors.pop();
            return Err(e);
        }
        self.epoch = epoch;
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        require(
            self.schema_version == 1 && !self.anchors.is_empty(),
            "invalid archive schema/anchors",
        )?;
        self.check_capacity()?;
        let mut previous = None;
        for a in &self.anchors {
            a.population.validate()?;
            require(
                a.epoch <= self.epoch && previous.is_none_or(|p| p < a.epoch),
                "invalid archive anchor order",
            )?;
            previous = Some(a.epoch);
        }
        require(
            self.anchors.last().is_some_and(|a| a.epoch == self.epoch),
            "archive current epoch lacks anchor",
        )?;
        let mut last = None;
        for s in &self.steps {
            s.before.validate()?;
            s.final_population.validate()?;
            require(
                s.before.len() == s.final_population.len(),
                "archive endpoint row counts differ",
            )?;
            require(
                s.before
                    .generations
                    .iter()
                    .zip(&s.final_population.generations)
                    .zip(&s.report.clone_plan.choices)
                    .all(|((&before, &after), choice)| {
                        before.checked_add(u64::from(choice.accepted)) == Some(after)
                    }),
                "archive recipient incarnation mismatch",
            )?;
            let anchor = self
                .anchors
                .iter()
                .find(|a| a.epoch == s.epoch)
                .ok_or_else(|| GasError::Checkpoint("missing archive epoch anchor".into()))?;
            require(
                s.report.step > anchor.step
                    && match last {
                        None => s.report.step == anchor.step + 1,
                        Some((epoch, step)) => {
                            (s.epoch > epoch && s.report.step == anchor.step + 1)
                                || (s.epoch == epoch && s.report.step == step + 1)
                        }
                    },
                "nonconsecutive archive steps",
            )?;
            s.report.validate(
                &self.gas_config,
                &s.final_population,
                s.report.step,
                s.report.reward_evaluations,
            )?;
            require(
                s.before.version == s.report.source_version
                    && s.donor_fitness.len() == s.report.clone_plan.sources.len(),
                "archive source/fitness mismatch",
            )?;
            require(
                s.stages.first().is_some_and(|v| v.stage == "pre_clone")
                    && s.stages.last().is_some_and(|v| v.stage == "post_kinetic"),
                "archive stage coverage incomplete",
            )?;
            require(
                s.stages
                    .first()
                    .is_some_and(|v| v.matches_population(&s.before))
                    && s.stages
                        .last()
                        .is_some_and(|v| v.matches_population(&s.final_population)),
                "archive endpoint stage differs from population",
            )?;
            for noise in &s.noise {
                if let Some(geometry) = &noise.geometry {
                    use crate::noise::NoiseGeometry;
                    let (rank, width) = match geometry {
                        NoiseGeometry::Isotropic { .. } => (noise.dimension, 1),
                        NoiseGeometry::Diagonal { .. } => (noise.dimension, noise.dimension),
                        NoiseGeometry::Full { .. } => (
                            noise.dimension,
                            crate::memory::checked_mul(noise.dimension, noise.dimension)?,
                        ),
                        NoiseGeometry::LowRank { rank, .. } => {
                            (*rank, crate::memory::checked_mul(noise.dimension, *rank)?)
                        }
                    };
                    require(
                        noise
                            .raw_innovation
                            .as_ref()
                            .is_some_and(|v| noise.rows.checked_mul(rank) == Some(v.len()))
                            && noise
                                .factor
                                .as_ref()
                                .is_some_and(|v| noise.rows.checked_mul(width) == Some(v.len())),
                        "archive raw noise/factor shape mismatch",
                    )?;
                }
                require(
                    noise.step == s.report.step
                        && noise.rows == s.before.len()
                        && noise.rows.checked_mul(noise.dimension) == Some(noise.sample.len())
                        && noise.sample.iter().all(|x| x.is_finite()),
                    "archive noise shape/value mismatch",
                )?;
                require(
                    noise
                        .raw_innovation
                        .as_ref()
                        .is_none_or(|v| v.iter().all(|x| x.is_finite()))
                        && noise
                            .factor
                            .as_ref()
                            .is_none_or(|v| v.iter().all(|x| x.is_finite())),
                    "archive noise innovations/factors nonfinite",
                )?;
            }
            for influence in &s.influences {
                require(
                    (influence.recipient as usize) < s.before.len() && influence.weight.is_finite(),
                    "archive influence shape/value mismatch",
                )?;
            }
            for f in &s.field_evaluations {
                require(
                    f.available.is_empty() || f.available.len() == f.rows,
                    "archive field coverage shape mismatch",
                )?;
                require(
                    f.rows == s.before.len()
                        && f.item_shape
                            .iter()
                            .try_fold(f.rows, |n, &d| n.checked_mul(d))
                            == Some(f.values.len()),
                    "archive field evaluation shape mismatch",
                )?;
            }
            for stage in &s.stages {
                require(
                    stage.generations.len() == s.before.len()
                        && stage.validity.len() == s.before.len(),
                    "archive stage row shape",
                )?;
                for field in stage.fields.values() {
                    let width = field
                        .item_shape
                        .iter()
                        .try_fold(1usize, |a, b| a.checked_mul(*b));
                    require(
                        width.and_then(|w| w.checked_mul(s.before.len()))
                            == Some(field.values.len()),
                        "archive stage field shape",
                    )?;
                }
            }
            last = Some((s.epoch, s.report.step));
        }
        Ok(())
    }
    pub fn terminal(&self) -> (u64, &Population<T>) {
        if let Some(step) = self.steps.last().filter(|s| s.epoch == self.epoch) {
            (step.report.step, &step.final_population)
        } else {
            let anchor = self
                .anchors
                .last()
                .expect("validated archive has an anchor");
            (anchor.step, &anchor.population)
        }
    }
    pub fn graph(&self) -> crate::fractal_set::FractalSet {
        crate::fractal_set::FractalSet::from_archive(self)
    }
}

/// Self-describing scalar codec. Each component has an explicit index; missing
/// components and duplicate components are rejected instead of filled with zero.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScalarComponent {
    pub field: String,
    pub index: usize,
    pub value: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScalarStageEncoding {
    pub codec: String,
    pub rows: usize,
    pub shapes: BTreeMap<String, Vec<usize>>,
    pub components: Vec<ScalarComponent>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reconstruction {
    pub codec: String,
    pub fields: BTreeMap<String, FieldSnapshot>,
    pub scalar_count: usize,
    pub max_absolute_residual: f64,
}
impl ScalarStageEncoding {
    pub fn encode(stage: &StageSnapshot) -> Self {
        Self {
            codec: "indexed-real-components/v1".into(),
            rows: stage.generations.len(),
            shapes: stage
                .fields
                .iter()
                .map(|(name, f)| (name.clone(), f.item_shape.clone()))
                .collect(),
            components: stage
                .fields
                .iter()
                .flat_map(|(name, f)| {
                    f.values
                        .iter()
                        .enumerate()
                        .map(|(index, &value)| ScalarComponent {
                            field: name.clone(),
                            index,
                            value,
                        })
                })
                .collect(),
        }
    }
    pub fn decode(&self) -> Result<BTreeMap<String, FieldSnapshot>> {
        require(
            self.codec == "indexed-real-components/v1",
            "unsupported scalar coordinate codec",
        )?;
        let mut fields = BTreeMap::new();
        let mut coverage = BTreeMap::new();
        for (name, shape) in &self.shapes {
            let len = shape
                .iter()
                .try_fold(self.rows, |n, &d| n.checked_mul(d))
                .ok_or_else(|| GasError::Shape("scalar codec shape overflow".into()))?;
            require(
                len <= self.components.len(),
                "scalar codec missing component coverage",
            )?;
            fields.insert(
                name.clone(),
                FieldSnapshot {
                    item_shape: shape.clone(),
                    values: vec![0.; len],
                },
            );
            coverage.insert(name.clone(), vec![false; len]);
        }
        for c in &self.components {
            let field = fields
                .get_mut(&c.field)
                .ok_or_else(|| GasError::Shape("scalar codec unknown field".into()))?;
            let seen = coverage
                .get_mut(&c.field)
                .expect("field coverage created above");
            require(
                c.index < seen.len() && !seen[c.index],
                "scalar codec duplicate or out-of-range component",
            )?;
            seen[c.index] = true;
            field.values[c.index] = c.value;
        }
        require(
            coverage.values().all(|v| v.iter().all(|&x| x)),
            "scalar codec missing component coverage",
        )?;
        Ok(fields)
    }
}
impl<T: Real> RunArchive<T> {
    pub fn reconstruct(&self, epoch: u64, step: u64, stage: &str) -> Result<Reconstruction> {
        let snapshot = self
            .steps
            .iter()
            .find(|s| s.epoch == epoch && s.report.step == step)
            .and_then(|s| s.stages.iter().find(|s| s.stage == stage))
            .ok_or_else(|| {
                GasError::Shape("requested stage is outside recorded coverage".into())
            })?;
        let mut encoding = ScalarStageEncoding::encode(snapshot);
        // Reordering verifies that reconstruction uses component addresses.
        encoding.components.reverse();
        let fields = encoding.decode()?;
        let mut residual = 0f64;
        for (name, expected) in &snapshot.fields {
            for (&actual, &reference) in fields[name].values.iter().zip(&expected.values) {
                if actual == reference || (actual.is_nan() && reference.is_nan()) {
                    continue;
                }
                residual = residual.max((actual - reference).abs());
            }
        }
        Ok(Reconstruction {
            codec: encoding.codec,
            fields,
            scalar_count: encoding.components.len(),
            max_absolute_residual: residual,
        })
    }
}
