//! Vocabulary shared by every spectroscopy module: what a run records, what an
//! operator needs, the per-step data an operator reads, and the two roles
//! (`LocalOperator`, `FieldSource`). Layouts are flat and row-major.
use super::{
    config::{ColorSource, FrameNormalization, MeasurementConfig, TimeAxis},
    report::Coverage,
};
use crate::{
    GasConfig, RecordingConfig, Result,
    boundary::{BoundaryPolicy, BoxDomain},
    donor::SamplingLaw,
    error::require,
    geometry::Kernel,
    kinetic::KineticKind,
    memory::checked_mul,
    physics::qft::math::C,
    tessellation::{GraphSnapshot, Projection},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The single supported schema of measurements, reports and evidence.
pub const SPECTROSCOPY_VERSION: u32 = 1;

/// A kind of recorded information an analysis may depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Record {
    Velocities,
    Color,
    Fitness,
    DistanceCompanions,
    CloningCompanions,
    ClonePlan,
    Graph,
    EuclideanTime,
    PeriodicBox,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Requirements {
    pub records: BTreeSet<Record>,
    /// Exact position dimension, when the definition exists in one dimension only.
    pub dimension: Option<usize>,
}
impl Requirements {
    pub fn new(records: impl IntoIterator<Item = Record>) -> Self {
        Self {
            records: records.into_iter().collect(),
            dimension: None,
        }
    }
    pub fn with(mut self, record: Record) -> Self {
        self.records.insert(record);
        self
    }
    pub fn in_dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }
    pub fn union(mut self, other: &Self) -> Self {
        self.records.extend(other.records.iter().copied());
        self.dimension = self.dimension.or(other.dimension);
        self
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Availability {
    #[default]
    Available,
    Unavailable {
        reason: String,
    },
}
impl Availability {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Available => None,
            Self::Unavailable { reason } => Some(reason),
        }
    }
    /// The first failure wins.
    pub fn and(self, other: Self) -> Self {
        if self.is_available() { other } else { self }
    }
}

/// Reason given to an exchange-odd frame mean on a mutual pairing.
pub const EXCHANGE_ODD_REASON: &str = "exchange-odd operator cancels on a mutual pairing";

/// What a gas configuration records, decided before any step. An empty
/// `missing` map means that every record is present. `of` knows the built-in
/// kinetic integrators and companion laws only; a run driven by custom
/// `GasOperators` states its own through `Extensions::capabilities`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Capabilities {
    /// Width of the position vectors.
    pub dimension: usize,
    pub missing: BTreeMap<Record, String>,
    /// The companion law of each role pairs walkers mutually: `c(c(i)) = i`.
    pub mutual_distance: bool,
    pub mutual_cloning: bool,
    /// Coordinate read as Euclidean time, when one is declared.
    pub euclidean_axis: Option<usize>,
    /// Width of a Gaussian companion kernel of each role; the interaction
    /// ranges `ε_d`, `ε_c`. `None` for any other kernel.
    pub distance_kernel_width: Option<f64>,
    pub cloning_kernel_width: Option<f64>,
    /// Dense viscosity records `2 N (N − 1)` influences per step.
    pub dense_viscosity: bool,
    /// Integrator time step, when the kinetic operator has one.
    pub time_step: Option<f64>,
}
impl Capabilities {
    /// Records of the variant alone, for positions of width `dimension`.
    pub fn of(config: &GasConfig, recording: &RecordingConfig, dimension: usize) -> Self {
        let mut missing = BTreeMap::new();
        let velocities = matches!(config.kinetic.integrator, KineticKind::Baoab { .. });
        if !velocities {
            missing.insert(
                Record::Velocities,
                "kinetic integrator records no velocities".to_string(),
            );
        }
        let dense = config
            .qft
            .viscosity
            .as_ref()
            .is_some_and(|v| v.coefficient > 0.);
        let graph = config
            .qft
            .graph_viscosity
            .as_ref()
            .is_some_and(|v| v.coefficient > 0.);
        if !velocities {
            missing.insert(
                Record::Color,
                "kinetic integrator records no viscous force".to_string(),
            );
        } else if !dense && !graph {
            missing.insert(
                Record::Color,
                "viscous force is identically zero: configure qft.viscosity or \
                 qft.graph_viscosity, or select an explicit RecordedField colour source"
                    .to_string(),
            );
        }
        if config.geometry.is_none() {
            missing.insert(Record::Graph, "no geometry stage".to_string());
        } else if !recording.graph {
            missing.insert(Record::Graph, "graph not recorded".to_string());
        }
        let euclidean_axis = config
            .geometry
            .as_ref()
            .and_then(|g| match g.pipeline.projection {
                Projection::DropLast { min_ambient }
                    if dimension >= min_ambient && dimension >= 2 =>
                {
                    Some(dimension - 1)
                }
                _ => None,
            });
        if euclidean_axis.is_none() {
            missing.insert(
                Record::EuclideanTime,
                "no Euclidean-time axis declared".to_string(),
            );
        }
        if periodic_box(&config.boundary).is_none() {
            missing.insert(
                Record::PeriodicBox,
                "momentum projection needs a periodic box".to_string(),
            );
        }
        let width = |kernel: &Kernel| match kernel {
            Kernel::Gaussian { width } => Some(*width),
            _ => None,
        };
        let mutual = |law| matches!(law, SamplingLaw::FisherYates | SamplingLaw::GaussianGreedy);
        Self {
            dimension,
            missing,
            mutual_distance: mutual(config.distance_donors.law),
            mutual_cloning: mutual(config.cloning_donors.law),
            euclidean_axis,
            distance_kernel_width: width(&config.distance_donors.kernel),
            cloning_kernel_width: width(&config.cloning_donors.kernel),
            dense_viscosity: dense,
            time_step: match config.kinetic.integrator {
                KineticKind::Baoab { dt, .. } | KineticKind::Brownian { dt, .. } => Some(dt),
                _ => None,
            },
        }
    }
    /// Apply the choices of a measurement: an explicit colour source replaces
    /// the viscous-force requirement (its presence is then checked per step),
    /// and an explicit Euclidean-time axis declares one.
    pub fn refine(mut self, measurement: &MeasurementConfig) -> Self {
        if matches!(measurement.color, ColorSource::RecordedField { .. }) {
            self.missing.remove(&Record::Color);
        }
        if let TimeAxis::Euclidean {
            axis: Some(axis), ..
        } = measurement.time
        {
            if axis < self.dimension && self.dimension >= 2 {
                self.euclidean_axis = Some(axis);
                self.missing.remove(&Record::EuclideanTime);
            } else {
                self.euclidean_axis = None;
                self.missing.insert(
                    Record::EuclideanTime,
                    format!(
                        "Euclidean-time axis {axis} is outside the {} position coordinates",
                        self.dimension
                    ),
                );
            }
        }
        self
    }
    /// Three position dimensions with every record present, unit Gaussian
    /// kernel widths and a unit time step: the context in which the catalog
    /// describes an operator.
    pub fn nominal() -> Self {
        Self {
            dimension: 3,
            euclidean_axis: Some(2),
            distance_kernel_width: Some(1.),
            cloning_kernel_width: Some(1.),
            time_step: Some(1.),
            ..Self::default()
        }
    }
    /// Ranges of a deserialized or injected value. `Measurement::validate`,
    /// `SpectroscopyReport::validate` and the accumulator, for the override
    /// of `Extensions::capabilities`, call it before use.
    pub fn validate(&self) -> Result<()> {
        let positive = |x: Option<f64>| x.is_none_or(|x| x.is_finite() && x > 0.);
        require(
            self.dimension >= 1
                && self.euclidean_axis.is_none_or(|a| a < self.dimension)
                && positive(self.distance_kernel_width)
                && positive(self.cloning_kernel_width)
                && positive(self.time_step),
            "capabilities need dimension >= 1, a Euclidean axis below it and positive kernel \
             widths and time step",
        )
    }
    pub fn has(&self, record: Record) -> bool {
        !self.missing.contains_key(&record)
    }
    pub fn check(&self, requirements: &Requirements) -> Availability {
        if let Some(d) = requirements.dimension
            && d != self.dimension
        {
            return Availability::unavailable(format!(
                "defined in {d} position dimensions; the run has {}",
                self.dimension
            ));
        }
        requirements
            .records
            .iter()
            .find_map(|r| self.missing.get(r))
            .map_or(Availability::Available, Availability::unavailable)
    }
}

/// The periodic box of a boundary policy, searching composed policies.
pub fn periodic_box(boundary: &BoundaryPolicy) -> Option<&BoxDomain> {
    match boundary {
        BoundaryPolicy::PeriodicBox { domain, .. } => Some(domain),
        BoundaryPolicy::Composed { policies } => policies.iter().find_map(periodic_box),
        _ => None,
    }
}

/// One recorded B-stage force with the velocity that entered it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StageKick {
    /// `[n, d]`.
    pub force: Vec<f64>,
    /// `[n, d]`, same population version as `force`.
    pub velocity: Vec<f64>,
    /// `[n]`: both records cover the row and every entry is finite.
    pub available: Vec<bool>,
    /// `[n]`: generation of each row in the population version the force was
    /// evaluated on (the `StageSnapshot` of the record's `version`). For the
    /// B stages that is the post-clone population of the step, so the B2
    /// record of step `t − 1` already carries the incarnations of frame `t`.
    pub generation: Vec<u64>,
}
impl StageKick {
    /// Shapes for `n` rows of width `d`, and finite entries where `available`.
    pub fn validate(&self, n: usize, d: usize) -> Result<()> {
        let cells = checked_mul(n, d)?;
        require(
            self.force.len() == cells
                && self.velocity.len() == cells
                && self.available.len() == n
                && self.generation.len() == n,
            "stage kick shape",
        )?;
        require(
            d == 0
                || self
                    .force
                    .chunks_exact(d)
                    .zip(self.velocity.chunks_exact(d))
                    .zip(&self.available)
                    .all(|((f, v), &a)| !a || f.iter().chain(v).all(|x| x.is_finite())),
            "nonfinite available stage kick",
        )
    }
}

/// Viscous-force records of one step. A stage is `None` when the step holds no
/// such record or its force and velocity versions differ.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Kick {
    pub b1: Option<StageKick>,
    pub b2: Option<StageKick>,
}
impl Kick {
    pub fn validate(&self, n: usize, d: usize) -> Result<()> {
        for stage in [&self.b1, &self.b2].into_iter().flatten() {
            stage.validate(n, d)?;
        }
        Ok(())
    }
}

/// Why a companion entry builds no element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompanionMask {
    /// The engine sampled no companion for the entry.
    Unsampled,
    Historical,
    /// The walker or its companion is ineligible (dead, revived or nonfinite).
    Ineligible,
    SelfCompanion,
}

/// Companions of one role, `[n, count]`, resolved from pool indices to slots.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Companions {
    pub count: usize,
    /// Slot of the source walker; meaningful where `valid`.
    pub slot: Vec<u32>,
    /// Generation of the source walker at sampling time.
    pub generation: Vec<u64>,
    /// The engine sampled a companion for this entry.
    pub valid: Vec<bool>,
    /// The source is a walker of an earlier frame (`source.frame != step − 1`).
    pub historical: Vec<bool>,
    /// The engine reported a mutual pairing for this step.
    pub mutual: bool,
}
impl Companions {
    /// Shapes for `n` walkers and every sampled slot below `n`. `mask` and
    /// `first` index without checks: a record that did not come from
    /// `frame::extract` is validated first.
    pub fn validate(&self, n: usize) -> Result<()> {
        let entries = checked_mul(n, self.count)?;
        require(
            self.count >= 1
                && self.slot.len() == entries
                && self.generation.len() == entries
                && self.valid.len() == entries
                && self.historical.len() == entries
                && self
                    .slot
                    .iter()
                    .zip(&self.valid)
                    .all(|(&j, &valid)| !valid || (j as usize) < n),
            "companion record shape",
        )
    }
    /// Why entry `k` of row `i` builds no element, `None` when it does. The
    /// one masking rule of `topology::build` and of the companion maps of
    /// `frame::base_state`; reasons are tested in the listed order. `self` is
    /// valid for `eligible.len()` walkers and `i`, `k` are in range.
    pub fn mask(&self, i: usize, k: usize, eligible: &[bool]) -> Option<CompanionMask> {
        let entry = i * self.count + k;
        let j = self.slot[entry] as usize;
        if !self.valid[entry] {
            Some(CompanionMask::Unsampled)
        } else if self.historical[entry] {
            Some(CompanionMask::Historical)
        } else if !eligible[i] || !eligible.get(j).is_some_and(|e| *e) {
            Some(CompanionMask::Ineligible)
        } else if j == i {
            Some(CompanionMask::SelfCompanion)
        } else {
            None
        }
    }
    /// First entry of row `i` that builds an element.
    pub fn first(&self, i: usize, eligible: &[bool]) -> Option<usize> {
        (0..self.count).find(|&k| self.mask(i, k, eligible).is_none())
    }
}

/// One measurement time: the pre-clone population of step `step` together
/// with the decisions the engine took on it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Frame {
    pub step: u64,
    pub epoch: u64,
    pub n: usize,
    pub d: usize,
    /// `[n, d]`.
    pub x: Vec<f64>,
    pub v: Option<Vec<f64>>,
    /// Eligible before cloning and finite in every coordinate read; false on
    /// revived rows.
    pub eligible: Vec<bool>,
    pub generation: Vec<u64>,
    pub fitness: Option<Vec<f64>>,
    /// Raw pre-clone reward.
    pub reward: Option<Vec<f64>>,
    pub distance: Option<Companions>,
    pub cloning: Option<Companions>,
    /// `[n]`: the pool-aligned decision fitness of the first cloning
    /// companion; 0 where the row has none (gate on `cloning.valid`).
    pub companion_fitness: Option<Vec<f64>>,
    /// The clone decision of this step was accepted for the walker.
    pub cloned: Vec<bool>,
    /// The accepted decision was a revival.
    pub revived: Vec<bool>,
    /// Viscous force and matched input velocity of this step's B stages.
    pub kick: Option<Kick>,
    /// The fields named by `ColorSource::RecordedField` as recorded in this
    /// step, when selected: `force` holds the amplitude field, `velocity` the
    /// phase field, `generation` the rows of the record's population version.
    /// `StepAlignment::Preceding` reads it from the previous frame.
    pub recorded_color: Option<StageKick>,
    pub graph: Option<GraphSnapshot<f64>>,
}
impl Frame {
    /// Shapes of every per-walker record, finite positions on eligible rows,
    /// then the companions, force records and graph. A frame restored from an
    /// `AccumulatorState` is validated before any operator reads it;
    /// `Accumulator::restore_with` does so for every frame it holds.
    pub fn validate(&self) -> Result<()> {
        let (n, d) = (self.n, self.d);
        let cells = checked_mul(n, d)?;
        let rows = |v: &Option<Vec<f64>>, len: usize| v.as_ref().is_none_or(|v| v.len() == len);
        require(
            n >= 1
                && d >= 1
                && self.x.len() == cells
                && rows(&self.v, cells)
                && self.eligible.len() == n
                && self.generation.len() == n
                && rows(&self.fitness, n)
                && rows(&self.reward, n)
                && rows(&self.companion_fitness, n)
                && self.cloned.len() == n
                && self.revived.len() == n,
            "frame shape",
        )?;
        require(
            self.x
                .chunks_exact(d)
                .zip(&self.eligible)
                .all(|(x, &e)| !e || x.iter().all(|a| a.is_finite())),
            "nonfinite eligible frame position",
        )?;
        for companions in [&self.distance, &self.cloning].into_iter().flatten() {
            companions.validate(n)?;
        }
        if let Some(kick) = &self.kick {
            kick.validate(n, d)?;
        }
        if let Some(record) = &self.recorded_color {
            record.validate(n, d)?;
        }
        if let Some(graph) = &self.graph {
            graph.validate(n)?;
        }
        Ok(())
    }
}

/// Role of a walker in the cloning interaction of one frame, read from the
/// ungated score sign and the accepted decisions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WalkerRole {
    /// Its own clone decision was accepted.
    Cloner,
    /// Not cloning, and the companion of at least one cloner.
    StrongResister,
    /// Not cloning, not targeted, with a fitter companion.
    WeakResister,
    /// Not cloning, not targeted, no fitter companion.
    #[default]
    Persister,
}
impl WalkerRole {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cloner => "cloner",
            Self::StrongResister => "strong_resister",
            Self::WeakResister => "weak_resister",
            Self::Persister => "persister",
        }
    }
}

/// Per-walker derived fields at one time; what an operator evaluation reads.
/// Optional fields are absent when the run does not record them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameState {
    pub step: u64,
    pub n: usize,
    pub d: usize,
    pub x: Vec<f64>,
    pub v: Option<Vec<f64>>,
    /// `[n, d]` colour vectors, zero where invalid.
    pub color: Vec<C>,
    pub color_valid: Vec<bool>,
    /// `[n, d]` force behind `color` under the same alignment, zero where
    /// `force_valid` is false: the record does not cover the row. A zero
    /// force on a covered row is valid and has an invalid colour.
    pub force: Option<Vec<f64>>,
    pub force_valid: Vec<bool>,
    /// `[n, d]` velocity inside the colour phase. It differs from `v`, the
    /// pre-clone velocity of the frame, under every alignment that reads the
    /// input velocity of a force record.
    pub phase_velocity: Option<Vec<f64>>,
    pub fitness: Option<Vec<f64>>,
    /// Unclipped, ungated score of each walker against its cloning companion,
    /// zero where `score_valid` is false (never NaN).
    pub score: Option<Vec<f64>>,
    /// `[n]`: the row is eligible and the engine sampled its cloning
    /// companion (`frame.cloning.valid`). A historical companion counts: its
    /// rescored `companion_fitness` is the decision input, so the row has a
    /// score although `cloning_companion` is `NO_COMPANION`. Empty without
    /// `score`. Every reader of `score` masks on it.
    pub score_valid: Vec<bool>,
    /// `[n, d]` mean of `(S_j − S_i) û_ij` over the first distance and the
    /// first cloning companion of each walker with both scores valid and
    /// `|r_ij| > 0`; zero where it has none. `û_ij = r_ij / |r_ij|` with
    /// `r_ij = x_j − x_i`, the minimum image on a periodic box: the
    /// displacement the vector channels use. Their projections and the
    /// `ScoreGradient` displacement read it.
    pub score_gradient: Option<Vec<f64>>,
    pub role: Option<Vec<WalkerRole>>,
    pub cloned: Vec<bool>,
    pub generation: Vec<u64>,
    pub eligible: Vec<bool>,
    pub euclidean_time: Option<Vec<f64>>,
    /// `[n]`: slot of the first valid current-frame companion of each role,
    /// `NO_COMPANION` where it is absent, historical, the walker itself or
    /// ineligible. Two-hop operators read `k(k(i))` here. These are the maps
    /// of the frame the state belongs to: at the sink of a propagator the
    /// element `(i, k(i))` is frozen at the source while a second hop follows
    /// the sink frame's map.
    pub distance_companion: Option<Vec<u32>>,
    pub cloning_companion: Option<Vec<u32>>,
}

/// Entry of `FrameState::{distance_companion, cloning_companion}` without a companion.
pub const NO_COMPANION: u32 = u32::MAX;

/// Which walkers an operator acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementKind {
    /// One walker.
    Site,
    /// A walker and its distance companion.
    DistancePair,
    /// A walker and its cloning companion.
    CloningPair,
    /// A walker, its distance companion and its cloning companion.
    Triplet,
}
impl ElementKind {
    pub fn arity(self) -> u8 {
        match self {
            Self::Site => 1,
            Self::DistancePair | Self::CloningPair => 2,
            Self::Triplet => 3,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Site => "site",
            Self::DistancePair => "distance",
            Self::CloningPair => "cloning",
            Self::Triplet => "triplet",
        }
    }
    /// Companion records the element is built from.
    pub fn requires(self) -> Requirements {
        match self {
            Self::Site => Requirements::default(),
            Self::DistancePair => Requirements::new([Record::DistanceCompanions]),
            Self::CloningPair => Requirements::new([Record::CloningCompanions]),
            Self::Triplet => {
                Requirements::new([Record::DistanceCompanions, Record::CloningCompanions])
            }
        }
    }
}

/// Key of one measured channel: a channel specification bound to an element kind.
pub fn channel_id(spec_id: &str, kind: ElementKind) -> String {
    format!("{spec_id}/{}", kind.name())
}

/// One site, pair or triplet of a source-time topology, as sampled: `i` chose
/// the others. An element carries no direction. The score-directed meson and
/// score-ordered baryon modes orient inside `evaluate` from `state.score` of
/// the walkers and mask the element on a tie; every mode that reads a score
/// masks a walker without `state.score_valid`. At a sink time they therefore
/// read the sink-time direction. `Su2 { directed }` reads no score: it takes
/// the absolute value of every hop phase, computed from `state.fitness`. A
/// triplet whose two companions coincide is an element; see
/// `Signature::degenerate`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Element {
    /// Slots `(i, j, k)`; entries beyond the arity of `kind` repeat `i`.
    pub walkers: [u32; 3],
    pub kind: ElementKind,
    /// `1 / K_valid(i)` under `CompanionChoice::All`, otherwise 1.
    pub weight: f64,
    /// Generations of the walkers at the source time, for `IdentityPolicy::Incarnation`.
    pub generation: [u64; 3],
}

/// Source-time topology, frozen: what a propagator carries forward.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Topology {
    pub step: u64,
    pub kind: ElementKind,
    pub elements: Vec<Element>,
    /// Pairs only: every element `(i, j)` has its mirror `(j, i)` with the
    /// same weight. Unequal weights break the exact cancellation of an
    /// exchange-odd frame mean.
    pub involutive: bool,
    /// Candidates kept and masked while building the elements.
    pub coverage: Coverage,
}
impl Topology {
    /// Every element is of `kind`, names walkers below `n` and has a finite
    /// positive weight. A topology restored from an `AccumulatorState` is
    /// validated before any operator reads it.
    pub fn validate(&self, n: usize) -> Result<()> {
        require(
            self.elements.iter().all(|e| {
                e.kind == self.kind
                    && e.walkers.iter().all(|&w| (w as usize) < n)
                    && e.weight.is_finite()
                    && e.weight > 0.
            }),
            "topology element kind, walker index or weight",
        )
    }
}

/// Behaviour of an operator under exchange of the pair `(i, j) ↔ (j, i)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExchangeParity {
    Even,
    Odd,
    Mixed,
}

/// Behaviour under the spatial parity `c → −c*`, `x → −x`, when verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialParity {
    Even,
    Odd,
}

/// Documentation of an operator: its algebra, never a particle name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Descriptor {
    /// Algebraic definition in the markup the report page renders.
    pub definition: String,
    /// Label of the definition in Volume II; empty for an arm it does not define.
    pub book_label: String,
    pub spatial_parity: Option<SpatialParity>,
    /// Caveats a reader needs (non-book arm, non-equivariant embedding, …).
    pub note: String,
}

/// Static description of an operator on one element kind.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Signature {
    /// Records read by `evaluate`. The companion records of the element kind
    /// are added by the caller.
    pub requires: Requirements,
    /// 1, or the number of kept components of a vector- or tensor-valued
    /// operator. Correlators contract them: `C(τ) = Σ_k C_kk(τ)`.
    pub components: usize,
    /// An `Odd` operator has no frame mean over mirrored pairs.
    pub exchange: ExchangeParity,
    /// False for a diagnostic envelope, which never enters a correlator.
    pub correlatable: bool,
    /// Frame normalisation the definition fixes, overriding
    /// `MeasurementConfig::normalization`: a fixed `1/N` in which walkers
    /// without a value contribute 0.
    pub normalization: Option<FrameNormalization>,
    /// The operator is defined on a triplet whose two companions coincide.
    /// Such triplets are masked, and counted as `masked_self`, for every
    /// other operator.
    pub degenerate: bool,
    pub descriptor: Descriptor,
}
impl Signature {
    /// Correlatable, configured normalisation, no degenerate triplets, empty descriptor.
    pub fn new(requires: Requirements, components: usize, exchange: ExchangeParity) -> Self {
        Self {
            requires,
            components,
            exchange,
            correlatable: true,
            normalization: None,
            degenerate: false,
            descriptor: Descriptor::default(),
        }
    }
}

/// What an operator may read besides the frame state. Interaction ranges and
/// the time step come from `capabilities`, never from `gas`.
#[derive(Clone, Copy, Debug)]
pub struct OperatorContext<'a> {
    pub gas: &'a GasConfig,
    pub measurement: &'a MeasurementConfig,
    pub capabilities: &'a Capabilities,
}

/// A local operator on sites, pairs or triplets. `ChannelSpec` implements it
/// for every built-in family; any other implementation is injected through
/// `Extensions::operator` together with the element kind it is measured on.
pub trait LocalOperator: Send + Sync {
    /// Specification id. `channel_id(id, kind)` keys the measured channel. An
    /// injected operator answers `custom/<name>` with a `<name>` that
    /// `ChannelSpec::Custom` validates; the id must change with its semantics.
    fn id(&self) -> String;
    /// `GasError::Capability` carries the reason a well-formed operator does
    /// not exist on `kind` in this context (an interaction range requested
    /// from a kernel that has none, an identically vanishing combination of
    /// options); the channel is then unavailable. Any other error aborts.
    fn signature(&self, kind: ElementKind, context: &OperatorContext<'_>) -> Result<Signature>;
    /// Write the `components` values of the operator on `element`, read from
    /// `state`, into `out`. `false` masks the element (invalid colour, missing
    /// field, a tie of a directed mode); `out` is then unspecified. The same
    /// method serves the source time and every sink time of a propagator.
    fn evaluate(
        &self,
        element: &Element,
        state: &FrameState,
        context: &OperatorContext<'_>,
        out: &mut [f64],
    ) -> bool;
    /// `evaluate` on every element: `values` is `[elements, components]`,
    /// `valid` is `[elements]`. The accumulator calls only this method, so an
    /// implementation may hoist per-frame work out of the element loop. An
    /// override agrees with `evaluate` element by element.
    fn evaluate_all(
        &self,
        elements: &[Element],
        state: &FrameState,
        context: &OperatorContext<'_>,
        values: &mut [f64],
        valid: &mut [bool],
    ) {
        let components = values.len() / elements.len().max(1);
        for (element, (out, ok)) in elements.iter().zip(
            values
                .chunks_exact_mut(components.max(1))
                .zip(valid.iter_mut()),
        ) {
            *ok = self.evaluate(element, state, context, out);
        }
    }
}

/// Provider of the per-walker matter field written into `FrameState::color`.
/// `ColorSource` implements it; any other implementation is injected through
/// `Extensions::field`.
pub trait FieldSource: Send + Sync {
    /// Stable key `<name>/v<N>`; it enters the measurement fingerprint of an
    /// injected source and changes with its semantics.
    fn id(&self) -> String;
    fn requires(&self) -> Requirements;
    /// Fill `color`, `color_valid`, `force`, `force_valid` and
    /// `phase_velocity` of `state` for `frame`. `previous` is the frame of the
    /// recorded step `frame.step − 1` of the same segment, if any; it is
    /// retained for every recorded step, also under `stride > 1` and during
    /// warm-up, and is `None` after a gap. A source that reads a record of
    /// `previous` masks row `i` iff the generation of that record differs from
    /// `frame.generation[i]`; a source that reads a record of `frame` masks
    /// `frame.cloned[i]`. `kappa` is the calibrated phase factor
    /// `m ℓ₀ / ħ_eff`. `Unavailable` when the frame holds no usable record:
    /// the state is left all-invalid and the frame has weight 0 for every
    /// channel that requires `Record::Color`.
    fn fill(
        &self,
        previous: Option<&Frame>,
        frame: &Frame,
        kappa: f64,
        state: &mut FrameState,
    ) -> Availability;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GasError, physics::spectroscopy::config::ColorAlignment};
    fn recording(graph: bool) -> RecordingConfig {
        RecordingConfig {
            graph,
            ..RecordingConfig::default()
        }
    }
    #[test]
    fn einstein_hilbert_gas_records_colour_on_mutual_pairings_with_a_euclidean_axis() {
        let gas = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
        let caps = Capabilities::of(&gas, &recording(true), 3);
        assert!(caps.has(Record::Color) && caps.has(Record::Velocities) && caps.has(Record::Graph));
        assert!(caps.mutual_distance && caps.mutual_cloning && !caps.dense_viscosity);
        assert_eq!(caps.euclidean_axis, Some(2));
        assert_eq!(caps.distance_kernel_width, None);
        assert_eq!(
            caps.missing.keys().collect::<Vec<_>>(),
            vec![&Record::PeriodicBox]
        );
        let unrecorded = Capabilities::of(&gas, &recording(false), 3);
        assert_eq!(unrecorded.missing[&Record::Graph], "graph not recorded");
        let planar = Capabilities::of(&gas, &recording(true), 2);
        assert_eq!(planar.euclidean_axis, None);
        let baryon = Requirements::new([Record::Color]).in_dimension(3);
        assert!(caps.check(&baryon).is_available());
        assert!(
            planar
                .check(&baryon)
                .reason()
                .unwrap()
                .contains("3 position dimensions")
        );
    }
    #[test]
    fn euclidean_gas_has_no_colour_and_says_why_until_a_recorded_field_is_selected() {
        let gas = GasConfig::euclidean(3, 0.05).unwrap();
        let caps = Capabilities::of(&gas, &recording(false), 3);
        let reason = caps
            .check(&Requirements::new([Record::Color]))
            .reason()
            .unwrap()
            .to_string();
        assert!(reason.starts_with("viscous force is identically zero"));
        assert_eq!(caps.missing[&Record::Graph], "no geometry stage");
        assert_eq!(caps.distance_kernel_width, Some(2.));
        assert!(!caps.mutual_distance);
        assert!(
            caps.check(&Requirements::new([Record::Fitness, Record::ClonePlan]))
                .is_available()
        );
        let explicit = MeasurementConfig {
            color: ColorSource::RecordedField {
                stage: "B1".into(),
                amplitude: "total_force".into(),
                phase: "force_input_velocity".into(),
                alignment: crate::physics::spectroscopy::config::StepAlignment::Preceding,
                threshold: 1e-12,
            },
            time: TimeAxis::Euclidean {
                axis: Some(0),
                bins: 16,
                range: None,
                periodic: false,
            },
            ..MeasurementConfig::default()
        };
        let refined = caps.clone().refine(&explicit);
        assert!(refined.has(Record::Color) && refined.has(Record::EuclideanTime));
        assert_eq!(refined.euclidean_axis, Some(0));
        let default = caps.refine(&MeasurementConfig::default());
        assert!(!default.has(Record::Color));
        assert_eq!(
            MeasurementConfig::default().color,
            ColorSource::ViscousForce {
                alignment: ColorAlignment::PrecedingKick,
                threshold: 1e-12
            }
        );
    }
    #[test]
    fn companion_entries_are_masked_in_a_fixed_order_and_the_first_usable_one_is_found() {
        // Walker 0: unsampled, then historical, then walker 2. Walker 1: itself,
        // then the ineligible walker 3.
        let companions = Companions {
            count: 3,
            slot: vec![1, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0],
            generation: vec![0; 12],
            valid: vec![
                false, true, true, true, true, false, true, true, true, true, true, true,
            ],
            historical: vec![
                false, true, false, false, false, false, false, false, false, false, false, false,
            ],
            mutual: false,
        };
        let eligible = [true, true, true, false];
        let masks: Vec<_> = (0..3).map(|k| companions.mask(0, k, &eligible)).collect();
        assert_eq!(
            masks,
            vec![
                Some(CompanionMask::Unsampled),
                Some(CompanionMask::Historical),
                None
            ]
        );
        assert_eq!(companions.first(0, &eligible), Some(2));
        assert_eq!(
            companions.mask(1, 0, &eligible),
            Some(CompanionMask::SelfCompanion)
        );
        assert_eq!(
            companions.mask(1, 1, &eligible),
            Some(CompanionMask::Ineligible)
        );
        assert_eq!(companions.first(1, &eligible), None);
        assert_eq!(companions.first(2, &eligible), Some(0));
        assert_eq!(
            companions.mask(3, 0, &eligible),
            Some(CompanionMask::Ineligible)
        );
    }
    #[test]
    fn malformed_companion_record_is_a_result_not_a_panic() {
        let companions = Companions {
            count: 1,
            slot: vec![1, 0],
            generation: vec![0; 2],
            valid: vec![true; 2],
            historical: vec![false; 2],
            mutual: true,
        };
        companions.validate(2).unwrap();
        let invalid = |c: &Companions, n| matches!(c.validate(n), Err(GasError::Configuration(_)));
        assert!(invalid(&companions, 3));
        let mut outside = companions.clone();
        outside.slot[0] = 2;
        assert!(invalid(&outside, 2));
        let mut unsampled = outside.clone();
        unsampled.valid[0] = false;
        unsampled.validate(2).unwrap();
        let mut short = companions.clone();
        short.historical.pop();
        assert!(invalid(&short, 2));
        assert!(invalid(&Companions::default(), 0));
        let kick = StageKick {
            force: vec![0., f64::NAN],
            velocity: vec![0.; 2],
            available: vec![true, false],
            generation: vec![0; 2],
        };
        kick.validate(2, 1).unwrap();
        assert!(kick.validate(1, 2).is_err() && kick.validate(2, 2).is_err());
        let frame = Frame {
            n: 2,
            d: 1,
            x: vec![0., f64::INFINITY],
            eligible: vec![true, false],
            generation: vec![0; 2],
            cloning: Some(companions.clone()),
            cloned: vec![false; 2],
            revived: vec![false; 2],
            kick: Some(Kick {
                b1: None,
                b2: Some(kick.clone()),
            }),
            ..Frame::default()
        };
        frame.validate().unwrap();
        let tampered = [
            Frame {
                cloning: Some(outside),
                ..frame.clone()
            },
            Frame {
                eligible: vec![true; 2],
                ..frame.clone()
            },
            Frame {
                fitness: Some(vec![1.]),
                ..frame.clone()
            },
            Frame {
                recorded_color: Some(StageKick::default()),
                ..frame.clone()
            },
            Frame::default(),
        ];
        assert!(tampered.iter().all(|f| f.validate().is_err()));
        let element = |walkers, weight| Element {
            walkers,
            kind: ElementKind::CloningPair,
            weight,
            generation: [0; 3],
        };
        let topology = |elements| Topology {
            step: 1,
            kind: ElementKind::CloningPair,
            elements,
            involutive: false,
            coverage: Coverage::default(),
        };
        topology(vec![element([0, 1, 0], 1.)]).validate(2).unwrap();
        assert!(topology(vec![element([0, 2, 0], 1.)]).validate(2).is_err());
        assert!(topology(vec![element([0, 1, 0], 0.)]).validate(2).is_err());
        let mut nominal = Capabilities::nominal();
        nominal.validate().unwrap();
        nominal.euclidean_axis = Some(3);
        assert!(nominal.validate().is_err());
        assert!(Capabilities::default().validate().is_err());
        let zero_step = Capabilities {
            time_step: Some(0.),
            ..Capabilities::nominal()
        };
        assert!(zero_step.validate().is_err());
    }
    #[test]
    fn a_jump_kinetic_records_neither_velocities_nor_colour() {
        let caps = Capabilities::of(&GasConfig::default(), &recording(false), 2);
        assert_eq!(
            caps.missing[&Record::Velocities],
            "kinetic integrator records no velocities"
        );
        assert!(!caps.has(Record::Color));
    }
    #[test]
    fn capabilities_and_availability_round_trip_through_json() {
        let gas = GasConfig::euclidean(2, 0.05).unwrap();
        let caps = Capabilities::of(&gas, &recording(false), 2);
        let text = serde_json::to_string(&caps).unwrap();
        assert_eq!(serde_json::from_str::<Capabilities>(&text).unwrap(), caps);
        let unavailable = Availability::unavailable(EXCHANGE_ODD_REASON);
        assert_eq!(
            serde_json::to_string(&unavailable).unwrap(),
            r#"{"status":"unavailable","reason":"exchange-odd operator cancels on a mutual pairing"}"#
        );
        assert_eq!(
            serde_json::to_string(&Availability::Available).unwrap(),
            r#"{"status":"available"}"#
        );
        assert_eq!(
            channel_id("meson/scalar/standard", ElementKind::DistancePair),
            "meson/scalar/standard/distance"
        );
    }
}
