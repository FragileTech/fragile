//! Spectroscopy configuration. A measurement is streamed and expensive and
//! depends on `MeasurementConfig` alone; an analysis is pure, cheap and
//! repeatable on the same measurement with a different `AnalysisConfig`.
//!
//! Every documented definition of an observable is an arm; the default is the
//! one of the Standard Model chapter of Volume II.
use super::contract::{Capabilities, ElementKind, SPECTROSCOPY_VERSION, WalkerRole};
use crate::{
    GasConfig, GasError, Result,
    error::require,
    physics::numerics::{Resampling, Subtraction},
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A unit-only enum with its wire name, which is also the identifier segment.
macro_rules! named_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident => $text:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(
            Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
        )]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $($(#[$vmeta])* $variant),+
        }
        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $text),+
                }
            }
        }
    };
}
fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.
}
/// A fit window from `t_min` to an explicit `t_max` holds `points` lags.
fn holds(t_min: usize, t_max: Option<usize>, points: usize) -> bool {
    t_max.is_none_or(|t| t >= t_min + points.saturating_sub(1))
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpectroscopyConfig {
    pub measurement: MeasurementConfig,
    pub analysis: AnalysisConfig,
}
impl SpectroscopyConfig {
    pub fn validate(&self) -> Result<()> {
        self.measurement.validate()?;
        self.analysis.validate()?;
        require(
            self.analysis
                .max_lag()
                .is_none_or(|l| l <= self.measurement.max_lag),
            "analysis windows exceed the measured lag range",
        )
    }
}
named_enum! {
    /// B stage of the BAOAB step whose force record is read.
    KickStage {
        #[default]
        B1 => "b1",
        B2 => "b2",
    }
}

/// Which force record and which velocity build the colour of frame `t`, the
/// pre-clone population of step `t`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColorAlignment {
    /// B2 viscous force of step `t − 1` with its own input velocity: one
    /// velocity inside the force and the phase, on the population whose
    /// companions and positions frame `t` holds. Rows whose B2 generation
    /// differs from the frame's are masked.
    #[default]
    PrecedingKick,
    /// A B stage of step `t` itself with its own input velocity, evaluated
    /// after cloning; walkers cloned at `t` are masked. Uncloned partners of
    /// an inelastic collision also changed velocity and are not masked.
    MatchedKick {
        #[serde(default)]
        stage: KickStage,
    },
    /// B2 viscous force of step `t − 1` with the pre-clone velocity of step
    /// `t`: the force of the preceding step, read step by step. Force and
    /// phase hold different velocities. Masks as `PrecedingKick`.
    PrecedingForce,
    /// B1 viscous force of step `t` with the pre-clone velocity of step `t`:
    /// what the reference implementation computes, whose per-step force array
    /// is stored one row behind its per-frame velocity array. Walkers cloned
    /// at `t` are masked.
    ReferenceOffset,
}
named_enum! {
    /// Which step's record of a `ColorSource::RecordedField` builds the colour
    /// of frame `t`.
    StepAlignment {
        /// The record of step `t − 1` (`previous.recorded_color`). Row `i` is
        /// masked iff the record's generation differs from the generation of
        /// frame `t`; the frame has no colour when there is no previous frame.
        #[default]
        Preceding => "preceding",
        /// The record of step `t` itself, evaluated after cloning for a
        /// post-clone stage; walkers cloned at `t` are masked.
        Matched => "matched",
    }
}
fn default_threshold() -> f64 {
    1e-12
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColorSource {
    /// `c_a = F_a e^{i κ v_a} / max(|F|, threshold)` from the recorded viscous
    /// force; invalid where `|F| ≤ threshold`, an absolute bound in force units.
    ViscousForce {
        #[serde(default)]
        alignment: ColorAlignment,
        #[serde(default = "default_threshold")]
        threshold: f64,
    },
    /// Explicit opt-in: any two recorded fields of one stage as amplitude and
    /// phase. The only colour route of a variant whose kinetic integrator is
    /// not BAOAB.
    RecordedField {
        stage: String,
        amplitude: String,
        phase: String,
        #[serde(default)]
        alignment: StepAlignment,
        #[serde(default = "default_threshold")]
        threshold: f64,
    },
}
impl Default for ColorSource {
    fn default() -> Self {
        Self::ViscousForce {
            alignment: ColorAlignment::PrecedingKick,
            threshold: default_threshold(),
        }
    }
}
impl ColorSource {
    pub fn threshold(&self) -> f64 {
        match self {
            Self::ViscousForce { threshold, .. } | Self::RecordedField { threshold, .. } => {
                *threshold
            }
        }
    }
    pub fn validate(&self) -> Result<()> {
        let threshold = self.threshold();
        require(
            threshold.is_finite() && threshold >= 0.,
            "colour threshold must be finite and nonnegative",
        )?;
        match self {
            Self::ViscousForce { .. } => Ok(()),
            Self::RecordedField {
                stage,
                amplitude,
                phase,
                ..
            } => require(
                !stage.is_empty() && !amplitude.is_empty() && !phase.is_empty(),
                "recorded colour source needs a stage and two field names",
            ),
        }
    }
}

/// Length `ℓ₀` of the colour phase `m v ℓ₀ / ħ_eff`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LengthScale {
    Fixed {
        value: f64,
    },
    /// Median Euclidean distance to the distance companion over the warm-up frames.
    #[default]
    WarmupCompanionMedian,
    /// Mean edge length of the recorded graph over the warm-up frames.
    WarmupEdgeMean {
        #[serde(default)]
        geodesic: bool,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhaseScale {
    pub h_eff: f64,
    pub mass: f64,
    pub length: LengthScale,
}
impl Default for PhaseScale {
    fn default() -> Self {
        Self {
            h_eff: 1.,
            mass: 1.,
            length: LengthScale::default(),
        }
    }
}
impl PhaseScale {
    pub fn validate(&self) -> Result<()> {
        require(
            positive(self.h_eff)
                && positive(self.mass)
                && !matches!(self.length, LengthScale::Fixed { value } if !positive(value)),
            "colour phase scales must be positive",
        )
    }
}

/// An interaction range. It is never the clone regulariser.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Range {
    /// Width of the Gaussian companion kernel of the gas configuration; any
    /// other kernel makes the dependent channels unavailable.
    #[default]
    FromKernel,
    Fixed {
        value: f64,
    },
}
named_enum! {
    /// Distance `D` inside a companion amplitude `exp(−D²/4ε²)`.
    PairDistance {
        /// `D² = |Δx|² + λ|Δv|²` on the recorded coordinates.
        #[default]
        Raw => "raw",
        /// The distance of the role's donor module (scaled, squashed or
        /// periodic as configured): the one its kernel width refers to.
        Configured => "configured",
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElectroweakScales {
    /// Action scale `ħ_eff` of the U(1) fitness phase.
    pub h_eff: f64,
    /// Action scale `h_S` of the SU(2) cloning phase; `None` uses `h_eff`.
    pub h_s: Option<f64>,
    /// Range `ε_d` of the distance-companion amplitude `exp(−D²/4ε_d²)`.
    pub epsilon_d: Range,
    /// Range `ε_c` of the cloning-companion amplitude.
    pub epsilon_c: Range,
    pub distance: PairDistance,
    /// Velocity weight `λ` of the raw distance; `None` reads the distance of
    /// each role's donor module (0 when it has none), a value applies to both.
    pub lambda: Option<f64>,
}
impl Default for ElectroweakScales {
    fn default() -> Self {
        Self {
            h_eff: 1.,
            h_s: None,
            epsilon_d: Range::FromKernel,
            epsilon_c: Range::FromKernel,
            distance: PairDistance::Raw,
            lambda: None,
        }
    }
}
impl ElectroweakScales {
    pub fn validate(&self) -> Result<()> {
        let range = |r: Range| !matches!(r, Range::Fixed { value } if !positive(value));
        require(
            positive(self.h_eff)
                && self.h_s.is_none_or(positive)
                && range(self.epsilon_d)
                && range(self.epsilon_c)
                && self.lambda.is_none_or(|l| l.is_finite() && l >= 0.),
            "electroweak action scales and ranges must be positive",
        )
    }
}
named_enum! {
    /// Companion roles on which pair operators are measured.
    PairSelection {
        Distance => "distance",
        Cloning => "cloning",
        #[default]
        Both => "both",
    }
}
impl PairSelection {
    pub fn kinds(self) -> &'static [ElementKind] {
        match self {
            Self::Distance => &[ElementKind::DistancePair],
            Self::Cloning => &[ElementKind::CloningPair],
            Self::Both => &[ElementKind::DistancePair, ElementKind::CloningPair],
        }
    }
}
named_enum! {
    /// Use of `K > 1` distance companions.
    CompanionChoice {
        /// The first companion that builds an element (`Companions::first`),
        /// as for `K = 1`.
        #[default]
        First => "first",
        /// One element per companion with weight `1 / K_valid(i)`.
        All => "all",
    }
}
named_enum! {
    /// Companions sampled from an earlier frame.
    HistoricalPolicy {
        /// Masked and counted.
        #[default]
        Mask => "mask",
    }
}
named_enum! {
    /// Identity of a walker between the source and the sink of a propagator.
    IdentityPolicy {
        /// The slot worldline, through cloning.
        #[default]
        Slot => "slot",
        /// A sink is masked when the generation of any walker of the element changed.
        Incarnation => "incarnation",
    }
}
named_enum! {
    /// Denominator of a frame average.
    FrameNormalization {
        /// Sum of valid element weights.
        #[default]
        ValidCount => "valid_count",
        /// The population size `N`: the frame observable with a transfer reading.
        FixedN => "fixed_n",
    }
}
fn default_bins() -> usize {
    32
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TimeAxis {
    /// Algorithm time only.
    #[default]
    MonteCarlo,
    /// Also correlate slabs of one position coordinate inside each frame.
    /// The algorithm-time series are measured regardless.
    Euclidean {
        /// `None` uses the axis the geometry projection drops.
        #[serde(default)]
        axis: Option<usize>,
        #[serde(default = "default_bins")]
        bins: usize,
        /// `None` takes the range covered during warm-up.
        #[serde(default)]
        range: Option<[f64; 2]>,
        #[serde(default)]
        periodic: bool,
    },
}
impl TimeAxis {
    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::MonteCarlo => Ok(()),
            Self::Euclidean { bins, range, .. } => require(
                (2..=4096).contains(&bins)
                    && range.is_none_or(|[a, b]| a.is_finite() && b.is_finite() && a < b),
                "Euclidean-time axis needs 2..=4096 bins and an ordered finite range",
            ),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PropagatorConfig {
    /// Specification ids, channel ids, or `*`.
    pub enabled_for: Vec<String>,
    /// Evaluate sinks at every `lag_stride`-th lag.
    pub lag_stride: usize,
    /// Bound on stored blocks; pairs are merged beyond it.
    pub max_blocks: usize,
}
impl Default for PropagatorConfig {
    fn default() -> Self {
        Self {
            // Exchange-odd channels of the standard set: on a mutual pairing
            // the propagator is their only estimator.
            enabled_for: vec![
                "meson/pseudoscalar/standard".into(),
                "vector/vector/full/raw".into(),
                "baryon/complex".into(),
                "tensor/components".into(),
            ],
            lag_stride: 1,
            max_blocks: 256,
        }
    }
}
impl PropagatorConfig {
    /// `max_lag` is `MeasurementConfig::max_lag`.
    pub fn validate(&self, max_lag: usize) -> Result<()> {
        require(
            self.lag_stride >= 1
                && self.lag_stride <= max_lag
                && (4..=65_536).contains(&self.max_blocks)
                && self.max_blocks.is_multiple_of(2),
            "propagators need lag_stride 1..=max_lag and an even max_blocks 4..=65536",
        )
    }
    pub fn enabled(&self, spec_id: &str, channel_id: &str) -> bool {
        self.enabled_for
            .iter()
            .any(|e| e == "*" || e == spec_id || e == channel_id)
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScaleSelection {
    /// Geometric ladder between two quantiles of the warm-up pair distances.
    Quantiles {
        count: usize,
        low: f64,
        high: f64,
    },
    Fixed {
        values: Vec<f64>,
    },
}
impl ScaleSelection {
    pub fn validate(&self) -> Result<()> {
        require(
            match self {
                Self::Quantiles { count, low, high } => {
                    (1..=64).contains(count) && 0. < *low && low < high && *high < 1.
                }
                Self::Fixed { values } => {
                    (1..=64).contains(&values.len())
                        && values.iter().all(|&v| positive(v))
                        && values.windows(2).all(|w| w[0] < w[1])
                }
            },
            "scales must be 1..=64 positive ascending values or ordered quantiles",
        )
    }
}
named_enum! {
    ScaleMode {
        /// Keep an element when its geodesic diameter is within the scale.
        #[default]
        Gate => "gate",
        /// Average colours over a Gaussian geodesic kernel of the scale.
        Smear => "smear",
    }
}
named_enum! {
    EdgeLength {
        #[default]
        Geodesic => "geodesic",
        Euclidean => "euclidean",
    }
}

/// Multiscale copies of selected channels. Needs `Record::Graph`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ScaleConfig {
    pub scales: ScaleSelection,
    pub mode: ScaleMode,
    pub length: EdgeLength,
    /// Specification ids, channel ids, or `*`.
    pub channels: Vec<String>,
}
impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            scales: ScaleSelection::Quantiles {
                count: 4,
                low: 0.05,
                high: 0.95,
            },
            mode: ScaleMode::Gate,
            length: EdgeLength::Geodesic,
            channels: vec!["*".into()],
        }
    }
}
impl ScaleConfig {
    pub fn validate(&self) -> Result<()> {
        self.scales.validate()
    }
}

/// Graph smoothing diagnostic: convex neighbour averaging of the colour
/// field. It defines no length scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlowConfig {
    pub steps: usize,
    /// Mixing weight of the neighbour average per step, in `(0, 1]`.
    pub step_size: f64,
    /// Run on every `every`-th measured frame.
    pub every: usize,
}
impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            steps: 20,
            step_size: 0.1,
            every: 8,
        }
    }
}
impl FlowConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            (1..=10_000).contains(&self.steps)
                && self.step_size > 0.
                && self.step_size <= 1.
                && self.every >= 1,
            "graph smoothing needs steps 1..=10000, step_size in (0, 1] and every >= 1",
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeasurementBudget {
    pub max_frames: usize,
    pub max_bytes: usize,
}
impl Default for MeasurementBudget {
    fn default() -> Self {
        Self {
            max_frames: 20_000,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}
impl MeasurementBudget {
    /// `max_lag` is `MeasurementConfig::max_lag`.
    pub fn validate(&self, max_lag: usize) -> Result<()> {
        require(
            self.max_frames > max_lag && self.max_bytes > 0,
            "measurement budget needs max_frames > max_lag and max_bytes > 0",
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeasurementConfig {
    /// Frames consumed only for calibration (`ℓ₀`, scales, Euclidean range).
    pub warmup: usize,
    pub max_lag: usize,
    /// Measure every `stride`-th recorded step.
    pub stride: usize,
    pub color: ColorSource,
    pub phase: PhaseScale,
    pub electroweak: ElectroweakScales,
    pub channels: Vec<ChannelSpec>,
    pub pairs: PairSelection,
    pub companions: CompanionChoice,
    pub historical: HistoricalPolicy,
    pub identity: IdentityPolicy,
    pub normalization: FrameNormalization,
    pub time: TimeAxis,
    pub propagators: PropagatorConfig,
    pub scales: Option<ScaleConfig>,
    pub flow: Option<FlowConfig>,
    pub budget: MeasurementBudget,
}
impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            warmup: 16,
            max_lag: 80,
            stride: 1,
            color: ColorSource::default(),
            phase: PhaseScale::default(),
            electroweak: ElectroweakScales::default(),
            channels: ChannelSpec::standard_set(),
            pairs: PairSelection::default(),
            companions: CompanionChoice::default(),
            historical: HistoricalPolicy::default(),
            identity: IdentityPolicy::default(),
            normalization: FrameNormalization::default(),
            time: TimeAxis::default(),
            propagators: PropagatorConfig::default(),
            scales: None,
            flow: None,
            budget: MeasurementBudget::default(),
        }
    }
}
impl MeasurementConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            (1..=4096).contains(&self.max_lag) && self.stride >= 1 && self.warmup <= 1_000_000,
            "measurement needs max_lag 1..=4096, stride >= 1 and warmup <= 1000000",
        )?;
        self.phase.validate()?;
        self.color.validate()?;
        self.electroweak.validate()?;
        // An accumulator needs 1..=256 channels, injected operators included.
        require(
            self.channels.len() <= 256,
            "measurement takes at most 256 channels",
        )?;
        let mut ids = BTreeSet::new();
        for channel in &self.channels {
            channel.validate()?;
            require(
                ids.insert(channel.id()),
                format!("duplicate channel `{}`", channel.id()),
            )?;
        }
        self.time.validate()?;
        self.propagators.validate(self.max_lag)?;
        if let Some(scales) = &self.scales {
            scales.validate()?;
        }
        if let Some(flow) = &self.flow {
            flow.validate()?;
        }
        self.budget.validate(self.max_lag)
    }
    /// Stable identity of a measurement: this configuration, the gas it was
    /// taken on, the resolved capabilities (position dimension, recorded
    /// graph, kernel widths, any override of `Extensions::capabilities`) and
    /// the identities of injected components
    /// (`accumulator::Extensions::identity`; empty without extensions).
    /// Measurements combine only when it matches.
    pub fn fingerprint(
        &self,
        gas: &GasConfig,
        capabilities: &Capabilities,
        injected: &[String],
    ) -> Result<String> {
        let mut gas = gas.clone();
        // Replicas differ by seed only.
        gas.seed = 0;
        let identity = (SPECTROSCOPY_VERSION, self, &gas, capabilities, injected);
        let text =
            serde_json::to_string(&identity).map_err(|e| GasError::Configuration(e.to_string()))?;
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in text.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        Ok(format!("{hash:016x}"))
    }
}
named_enum! {
    MesonQuantum {
        #[default]
        Scalar => "scalar",
        Pseudoscalar => "pseudoscalar",
    }
}
named_enum! {
    MesonMode {
        /// `Re q_ij` / `Im q_ij` with `q_ij = c_i† c_j`.
        #[default]
        Standard => "standard",
        /// Pair oriented from the lower to the higher score.
        ScoreDirected => "score_directed",
        /// Pair value weighted by the score difference.
        ScoreWeighted => "score_weighted",
        /// `Σ_a c*_a (γ5)_aa c_a` with alternating signs.
        Gamma5Diagonal => "gamma5_diagonal",
        /// `|q_ij|²`, phase invariant.
        Abs2 => "abs2",
    }
}
named_enum! {
    VectorQuantum {
        /// `Re q_ij · r_ij`.
        #[default]
        Vector => "vector",
        /// `Im q_ij · r_ij`.
        Axial => "axial",
    }
}
named_enum! {
    /// Part of the pair displacement kept, along or across the score gradient
    /// of the anchor (`FrameState::score_gradient`); applies in every mode.
    VectorProjection {
        #[default]
        Full => "full",
        Longitudinal => "longitudinal",
        Transverse => "transverse",
    }
}
named_enum! {
    Displacement {
        /// `r_ij = x_j − x_i`.
        #[default]
        Raw => "raw",
        /// `r_ij / |r_ij|`.
        Unit => "unit",
        /// The score gradient of the anchor in place of the displacement.
        ScoreGradient => "score_gradient",
        /// No displacement: component `μ` is the colour bilinear
        /// `h^μ = i (c̄_i^μ c_j^ν − c̄_i^ν c_j^μ)`, `ν = μ + 1 mod d`, of the
        /// colour-space gamma matrices; vector is `Re h`, axial `Im h`. Needs
        /// `d ≥ 3` and the full projection.
        ColorGamma => "color_gamma",
    }
}
named_enum! {
    /// Part of `b_ijk = det[c_i, c_j, c_k]`.
    BaryonMode {
        Real => "real",
        Imag => "imag",
        /// `(Re b, Im b)` as two components: the contracted correlator is the
        /// complex determinant correlator `Re ⟨b̄_s b_t⟩`.
        #[default]
        Complex => "complex",
        Abs2 => "abs2",
        /// `|b|`: not a Volume II mode; kept for comparison with the reference code.
        Abs => "abs",
        ScoreOrdered => "score_ordered",
        FluxWeighted => "flux_weighted",
    }
}
named_enum! {
    /// Function of the plaquette `Π_ijk = q_ij q_jk q_ki`, or the force norm.
    GlueballObservable {
        #[default]
        RePlaquette => "re_plaquette",
        OneMinusRe => "one_minus_re",
        OneMinusCos => "one_minus_cos",
        Sin2 => "sin2",
        /// `Σ_i |F_visc,i|²`.
        ForceNorm => "force_norm",
    }
}
named_enum! {
    MomentumPhase {
        #[default]
        Cos => "cos",
        Sin => "sin",
    }
}
/// Fourier weight `cos|sin(2π mode x_axis / L)` on a periodic box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Momentum {
    pub axis: usize,
    pub mode: u32,
    pub phase: MomentumPhase,
}
impl Momentum {
    pub fn validate(&self) -> Result<()> {
        require(
            self.mode <= 64 && !(self.phase == MomentumPhase::Sin && self.mode == 0),
            "momentum mode must be at most 64, and the sine mode 0 vanishes identically",
        )
    }
}
named_enum! {
    TensorMode {
        /// Components of the `σ_μν` bilinear, contracted in the correlator.
        #[default]
        Components => "components",
        /// RMS over components: a diagnostic that never enters a correlator.
        Envelope => "envelope",
    }
}
named_enum! {
    /// `ψ̄_i Γ ψ_j` in the Dirac representation.
    DiracGamma {
        #[default]
        Scalar => "scalar",
        Pseudoscalar => "pseudoscalar",
        Vector => "vector",
        Axial => "axial",
        Tensor => "tensor",
        TensorTime => "tensor_time",
    }
}
named_enum! {
    U1Mode {
        /// `e^{i q θ_ij}`, `θ_ij = −(Φ_j − Φ_i)/ħ_eff`.
        #[default]
        Phase => "phase",
        /// The phase times `exp(−D²/4ε_d²)`.
        Dressed => "dressed",
    }
}
named_enum! {
    Su2Mode {
        #[default]
        Phase => "phase",
        Component => "component",
        Doublet => "doublet",
        DoubletDiff => "doublet_diff",
    }
}
named_enum! {
    ChiralityObservable {
        /// Chirality `χ_i` of the walker roles: `+1` on cloners and strong
        /// resisters, `−1` on weak resisters and persisters.
        #[default]
        Chi => "chi",
        /// Indicator of a left-handed walker; its fixed-`N` frame mean is `|L_t| / N`.
        LeftFraction => "left_fraction",
        /// `e^{i (F_c − F_i) / ħ_eff}` on cloners with a right-handed cloning companion.
        LeftRightCoupling => "left_right_coupling",
    }
}
named_enum! {
    /// Chiral projector right-multiplying `γ⁰Γ` of a Dirac bilinear.
    ChiralProjector {
        #[default]
        None => "none",
        Left => "left",
        Right => "right",
    }
}
named_enum! {
    /// Restriction of a pair `(i, j)` to the chirality classes of its walkers,
    /// read from `FrameState::role` (left = cloner or strong resister).
    RoleClass {
        #[default]
        Any => "any",
        LeftLeft => "left_left",
        RightRight => "right_right",
        LeftRight => "left_right",
        RightLeft => "right_left",
    }
}
named_enum! {
    /// Electroweak phase multiplying a Dirac bilinear.
    PhaseLink {
        #[default]
        None => "none",
        /// `e^{i θ_ij}` of the U(1) fitness phase.
        U1 => "u1",
        /// The SU(2) cloning phase.
        Su2 => "su2",
    }
}
named_enum! {
    /// Observables of the spinor contraction `τ` and the Pauli-vector bilinear
    /// `W` of two effective edge twistors.
    TwistorObservable {
        #[default]
        Scalar => "scalar",
        Pseudoscalar => "pseudoscalar",
        Abs2 => "abs2",
        /// `Re W^a`, three components contracted by the dot product.
        Vector => "vector",
        /// `Im W^a`, three components.
        Axial => "axial",
        /// The five spin-2 components `Q`, kept separate.
        Tensor => "tensor",
        /// Their mean as one component: the documented scalar, which is not
        /// rotation invariant.
        TensorMean => "tensor_mean",
    }
}
fn default_charge() -> u8 {
    1
}
fn one() -> f64 {
    1.
}

/// One measured operator family member. `id()` is its stable key; a measured
/// channel binds it to an element kind (`contract::channel_id`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChannelSpec {
    Meson {
        #[serde(default)]
        quantum: MesonQuantum,
        #[serde(default)]
        mode: MesonMode,
    },
    Vector {
        #[serde(default)]
        quantum: VectorQuantum,
        #[serde(default)]
        projection: VectorProjection,
        #[serde(default)]
        displacement: Displacement,
    },
    Baryon {
        #[serde(default)]
        mode: BaryonMode,
        /// Exponent of the flux weight; read by `FluxWeighted` only, and part
        /// of its id when it is not 1.
        #[serde(default = "one")]
        flux_alpha: f64,
    },
    Glueball {
        #[serde(default)]
        observable: GlueballObservable,
        #[serde(default)]
        momentum: Option<Momentum>,
    },
    Tensor {
        #[serde(default)]
        mode: TensorMode,
    },
    /// `ψ̄_i Γ P ψ_j`, optionally on one role class and times an electroweak
    /// phase: the chiral currents and Yukawa scalars.
    Dirac {
        #[serde(default)]
        gamma: DiracGamma,
        #[serde(default)]
        projector: ChiralProjector,
        #[serde(default)]
        pairs: RoleClass,
        #[serde(default)]
        link: PhaseLink,
    },
    U1 {
        #[serde(default)]
        mode: U1Mode,
        #[serde(default = "default_charge")]
        charge: u8,
    },
    Su2 {
        #[serde(default)]
        mode: Su2Mode,
        #[serde(default)]
        directed: bool,
    },
    ElectroweakMixed,
    FitnessPhase,
    CloneIndicator,
    ParityVelocity {
        #[serde(default)]
        role: WalkerRole,
    },
    Chirality {
        #[serde(default)]
        observable: ChiralityObservable,
    },
    Twistor {
        #[serde(default)]
        observable: TwistorObservable,
        /// Velocity scale `α > 0` of the edge four-vector; part of the id when
        /// it is not 1.
        #[serde(default = "one")]
        velocity_scale: f64,
    },
    /// An operator injected through `Extensions::operator`. Listed in
    /// `MeasurementConfig::channels` it measures nothing; series of injected
    /// operators carry it.
    Custom {
        id: String,
    },
}
impl Default for ChannelSpec {
    fn default() -> Self {
        Self::Meson {
            quantum: MesonQuantum::Scalar,
            mode: MesonMode::Standard,
        }
    }
}
impl ChannelSpec {
    pub fn id(&self) -> String {
        match self {
            Self::Meson { quantum, mode } => format!("meson/{}/{}", quantum.name(), mode.name()),
            Self::Vector {
                quantum,
                projection,
                displacement,
            } => format!(
                "vector/{}/{}/{}",
                quantum.name(),
                projection.name(),
                displacement.name()
            ),
            Self::Baryon {
                mode: BaryonMode::FluxWeighted,
                flux_alpha,
            } if *flux_alpha != 1. => format!("baryon/flux_weighted/a{flux_alpha}"),
            Self::Baryon { mode, .. } => format!("baryon/{}", mode.name()),
            Self::Glueball {
                observable,
                momentum: None,
            } => format!("glueball/{}", observable.name()),
            Self::Glueball {
                observable,
                momentum: Some(m),
            } => format!(
                "glueball/{}/p{}_{}{}",
                observable.name(),
                m.axis,
                m.phase.name(),
                m.mode
            ),
            Self::Tensor { mode } => format!("tensor/{}", mode.name()),
            Self::Dirac {
                gamma,
                projector,
                pairs,
                link,
            } => {
                // Non-default options are appended, so plain bilinears keep their ids.
                let mut id = format!("dirac/{}", gamma.name());
                if *projector != ChiralProjector::None {
                    id = format!("{id}/{}", projector.name());
                }
                if *pairs != RoleClass::Any {
                    id = format!("{id}/{}", pairs.name());
                }
                if *link != PhaseLink::None {
                    id = format!("{id}/{}", link.name());
                }
                id
            }
            Self::U1 { mode, charge } => format!("u1/{}/q{charge}", mode.name()),
            Self::Su2 { mode, directed } => format!(
                "su2/{}{}",
                mode.name(),
                if *directed { "/directed" } else { "" }
            ),
            Self::ElectroweakMixed => "electroweak_mixed".into(),
            Self::FitnessPhase => "fitness_phase".into(),
            Self::CloneIndicator => "clone_indicator".into(),
            Self::ParityVelocity { role } => format!("parity_velocity/{}", role.name()),
            Self::Chirality { observable } => format!("chirality/{}", observable.name()),
            Self::Twistor {
                observable,
                velocity_scale,
            } if *velocity_scale != 1. => {
                format!("twistor/{}/a{velocity_scale}", observable.name())
            }
            Self::Twistor { observable, .. } => format!("twistor/{}", observable.name()),
            Self::Custom { id } => format!("custom/{id}"),
        }
    }
    /// Family segment of `id()`.
    pub fn family(&self) -> &'static str {
        match self {
            Self::Meson { .. } => "meson",
            Self::Vector { .. } => "vector",
            Self::Baryon { .. } => "baryon",
            Self::Glueball { .. } => "glueball",
            Self::Tensor { .. } => "tensor",
            Self::Dirac { .. } => "dirac",
            Self::U1 { .. } => "u1",
            Self::Su2 { .. } => "su2",
            Self::ElectroweakMixed => "electroweak_mixed",
            Self::FitnessPhase => "fitness_phase",
            Self::CloneIndicator => "clone_indicator",
            Self::ParityVelocity { .. } => "parity_velocity",
            Self::Chirality { .. } => "chirality",
            Self::Twistor { .. } => "twistor",
            Self::Custom { .. } => "custom",
        }
    }
    /// Element kinds the specification is measured on. The chirality and the
    /// left fraction are quantities of single walkers, over all `N` of them;
    /// the left-right coupling reads a walker and its cloning companion. An
    /// injected `Custom` operator states its own kind at injection.
    pub fn kinds(&self, pairs: PairSelection) -> Vec<ElementKind> {
        match self {
            Self::Meson { .. } | Self::Vector { .. } | Self::Tensor { .. } | Self::Dirac { .. } => {
                pairs.kinds().to_vec()
            }
            Self::Glueball {
                observable: GlueballObservable::ForceNorm,
                ..
            }
            | Self::FitnessPhase
            | Self::CloneIndicator
            | Self::ParityVelocity { .. }
            | Self::Chirality {
                observable: ChiralityObservable::Chi | ChiralityObservable::LeftFraction,
            } => vec![ElementKind::Site],
            Self::Chirality { .. } | Self::Su2 { .. } => vec![ElementKind::CloningPair],
            Self::Custom { .. } => vec![],
            Self::Baryon { .. }
            | Self::Glueball { .. }
            | Self::ElectroweakMixed
            | Self::Twistor { .. } => vec![ElementKind::Triplet],
            Self::U1 { .. } => vec![ElementKind::DistancePair],
        }
    }
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Vector {
                projection,
                displacement: Displacement::ColorGamma,
                ..
            } => require(
                *projection == VectorProjection::Full,
                "colour-space gamma vector takes the full projection only",
            ),
            Self::U1 { charge, .. } => {
                require((1..=8).contains(charge), "U(1) charge must be in 1..=8")
            }
            Self::Baryon { flux_alpha, .. } => require(
                flux_alpha.is_finite() && *flux_alpha >= 0.,
                "baryon flux exponent must be finite and nonnegative",
            ),
            Self::Glueball {
                momentum: Some(m), ..
            } => m.validate(),
            Self::Twistor { velocity_scale, .. } => require(
                positive(*velocity_scale),
                "twistor velocity scale must be finite and positive",
            ),
            Self::Custom { id } => require(
                !id.is_empty() && !id.contains(['/', '@']),
                "custom channel id must be nonempty without `/` or `@`",
            ),
            _ => Ok(()),
        }
    }
    /// The catalog member measuring the same series as this one, where a
    /// second id names one observable. `Re q_ij` is unmoved by the
    /// conjugation a score orientation applies, so off a tie the
    /// score-directed scalar meson is the standard one; a displacement that
    /// *is* the score gradient has no transverse part, so its longitudinal
    /// projection is the whole of it. Two such members of one basis give the
    /// joint covariance an exact null direction.
    pub fn twin(&self) -> Option<Self> {
        match *self {
            Self::Meson {
                quantum: MesonQuantum::Scalar,
                mode: MesonMode::ScoreDirected,
            } => Some(Self::Meson {
                quantum: MesonQuantum::Scalar,
                mode: MesonMode::Standard,
            }),
            Self::Vector {
                quantum,
                projection: VectorProjection::Longitudinal,
                displacement: Displacement::ScoreGradient,
            } => Some(Self::Vector {
                quantum,
                projection: VectorProjection::Full,
                displacement: Displacement::ScoreGradient,
            }),
            _ => None,
        }
    }
    /// Default channels: the strong sector, the electroweak phases and the
    /// symmetry-breaking scalars as the Standard Model chapter defines them.
    pub fn standard_set() -> Vec<Self> {
        vec![
            Self::Meson {
                quantum: MesonQuantum::Scalar,
                mode: MesonMode::Standard,
            },
            Self::Meson {
                quantum: MesonQuantum::Pseudoscalar,
                mode: MesonMode::Standard,
            },
            Self::Vector {
                quantum: VectorQuantum::Vector,
                projection: VectorProjection::Full,
                displacement: Displacement::Raw,
            },
            Self::Vector {
                quantum: VectorQuantum::Axial,
                projection: VectorProjection::Full,
                displacement: Displacement::Raw,
            },
            Self::Baryon {
                mode: BaryonMode::Complex,
                flux_alpha: 1.,
            },
            Self::Baryon {
                mode: BaryonMode::Abs2,
                flux_alpha: 1.,
            },
            Self::Glueball {
                observable: GlueballObservable::RePlaquette,
                momentum: None,
            },
            Self::Glueball {
                observable: GlueballObservable::ForceNorm,
                momentum: None,
            },
            Self::Tensor {
                mode: TensorMode::Components,
            },
            Self::U1 {
                mode: U1Mode::Phase,
                charge: 1,
            },
            Self::U1 {
                mode: U1Mode::Dressed,
                charge: 1,
            },
            Self::Su2 {
                mode: Su2Mode::Phase,
                directed: false,
            },
            Self::Su2 {
                mode: Su2Mode::Doublet,
                directed: false,
            },
            Self::FitnessPhase,
            Self::CloneIndicator,
            Self::Chirality {
                observable: ChiralityObservable::Chi,
            },
        ]
    }
    /// Every built-in arm with every unit option of its first level, default
    /// parameters, no momentum projection and no Dirac projector, role class
    /// or phase link: the channel catalog. `Custom` is never listed.
    pub fn all() -> Vec<Self> {
        let mut out = vec![];
        for &quantum in MesonQuantum::ALL {
            for &mode in MesonMode::ALL {
                out.push(Self::Meson { quantum, mode });
            }
        }
        for &quantum in VectorQuantum::ALL {
            for &projection in VectorProjection::ALL {
                for &displacement in Displacement::ALL {
                    if displacement == Displacement::ColorGamma
                        && projection != VectorProjection::Full
                    {
                        continue;
                    }
                    out.push(Self::Vector {
                        quantum,
                        projection,
                        displacement,
                    });
                }
            }
        }
        out.extend(BaryonMode::ALL.iter().map(|&mode| Self::Baryon {
            mode,
            flux_alpha: 1.,
        }));
        out.extend(
            GlueballObservable::ALL
                .iter()
                .map(|&observable| Self::Glueball {
                    observable,
                    momentum: None,
                }),
        );
        out.extend(TensorMode::ALL.iter().map(|&mode| Self::Tensor { mode }));
        out.extend(DiracGamma::ALL.iter().map(|&gamma| Self::Dirac {
            gamma,
            projector: ChiralProjector::None,
            pairs: RoleClass::Any,
            link: PhaseLink::None,
        }));
        for &mode in U1Mode::ALL {
            for charge in [1, 2] {
                out.push(Self::U1 { mode, charge });
            }
        }
        for &mode in Su2Mode::ALL {
            for directed in [false, true] {
                out.push(Self::Su2 { mode, directed });
            }
        }
        out.extend([
            Self::ElectroweakMixed,
            Self::FitnessPhase,
            Self::CloneIndicator,
        ]);
        out.extend(
            [
                WalkerRole::Cloner,
                WalkerRole::StrongResister,
                WalkerRole::WeakResister,
                WalkerRole::Persister,
            ]
            .map(|role| Self::ParityVelocity { role }),
        );
        out.extend(
            ChiralityObservable::ALL
                .iter()
                .map(|&observable| Self::Chirality { observable }),
        );
        out.extend(
            TwistorObservable::ALL
                .iter()
                .map(|&observable| Self::Twistor {
                    observable,
                    velocity_scale: 1.,
                }),
        );
        out
    }
}
named_enum! {
    /// How independent replicas enter the error estimate.
    Combine {
        /// Concatenate the blocks of all replicas.
        #[default]
        PooledBlocks => "pooled_blocks",
        /// One resampling unit per replica; needs at least 8 replicas.
        RunsAsSamples => "runs_as_samples",
    }
}
named_enum! {
    EstimatorChoice {
        /// The frame mean, or the source-frozen propagator where the frame
        /// mean is unavailable and a propagator was measured.
        #[default]
        Auto => "auto",
        FrameMean => "frame_mean",
        SourceFrozen => "source_frozen",
        EuclideanTime => "euclidean_time",
    }
}
named_enum! {
    EffectiveMassKind {
        /// `ln C(t)/C(t+1)`.
        #[default]
        LogRatio => "log_ratio",
        /// `arccosh((C(t−1) + C(t+1)) / 2C(t))`; needs a ratio of at least 1.
        Cosh => "cosh",
    }
}
named_enum! {
    FitMethod {
        #[default]
        WindowScan => "window_scan",
        MultiExponential => "multi_exponential",
        Both => "both",
    }
}
named_enum! {
    TimeUnit {
        /// Lags count measured frames.
        #[default]
        Frames => "frames",
        /// Lags are multiplied by `stride · dt` of the integrator.
        StepDt => "step_dt",
        /// Lags count bins of a position coordinate, `time_step` being the bin
        /// width. Report only: set by the Euclidean-time estimator and
        /// rejected by `AnalysisConfig::validate`.
        Coordinate => "coordinate",
    }
}

/// Log-linear single-exponential fits over all windows, averaged with weights
/// `exp(−AIC/2)`, `AIC = χ² + 2k + 2 N_cut`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WindowScanConfig {
    pub t_min: usize,
    /// `None` extends to the last lag that passes `min_point_snr`.
    pub t_max: Option<usize>,
    pub min_points: usize,
    /// Points with `|C|/σ` below this value end the usable range.
    pub min_point_snr: f64,
    /// A rate with `value/error` below this value is reported as no signal.
    pub min_rate_snr: f64,
    /// Correlated χ²; otherwise the diagonal approximation, stated in the notes.
    pub correlated: bool,
    /// Usable lags the scan fits. The window count is quadratic in it and each
    /// window costs an eigendecomposition of its own size, so the cost grows as
    /// the fifth power of it; a usable range beyond this is a `Configuration`
    /// error naming the cap, not a scan that runs for hours.
    pub max_usable: usize,
}
impl Default for WindowScanConfig {
    fn default() -> Self {
        Self {
            t_min: 1,
            t_max: None,
            min_points: 4,
            min_point_snr: 2.,
            min_rate_snr: 2.,
            correlated: true,
            max_usable: 128,
        }
    }
}
impl WindowScanConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            self.min_points >= 3
                && holds(self.t_min, self.t_max, self.min_points)
                && self.min_point_snr.is_finite()
                && self.min_point_snr >= 0.
                && self.min_rate_snr.is_finite()
                && self.min_rate_snr >= 0.
                && self.max_usable >= self.min_points,
            "window scan needs at least 3 points per window, nonnegative thresholds and a usable \
             cap of at least min_points",
        )
    }
}

/// `C(t) = Σ_n a_n² exp(−E_n t)` with `E_n = Σ_{m≤n} dE_m`. Priors are
/// channel-agnostic and scale free: a Gaussian on `ln dE_n` that
/// is the same for every channel, and a wide Gaussian on `ln a_n²` about
/// `ln |C(t_min)|`, which is reported because it reads the data.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MultiExponentialConfig {
    pub nexp: usize,
    pub t_min: usize,
    pub t_max: Option<usize>,
    /// Mean and width of the prior on `ln dE_n` (inverse frames).
    pub log_gap_mean: f64,
    pub log_gap_sigma: f64,
    pub log_amplitude_sigma: f64,
    /// Posterior/prior width ratio of `ln dE_0` above which the fit is
    /// prior dominated and reports no rate.
    pub dominance_ratio: f64,
}
impl Default for MultiExponentialConfig {
    fn default() -> Self {
        Self {
            nexp: 1,
            t_min: 1,
            t_max: None,
            log_gap_mean: (0.1_f64).ln(),
            log_gap_sigma: 3.,
            log_amplitude_sigma: 5.,
            dominance_ratio: 0.7,
        }
    }
}
impl MultiExponentialConfig {
    pub fn validate(&self) -> Result<()> {
        require(
            (1..=4).contains(&self.nexp)
                && holds(self.t_min, self.t_max, 2 * self.nexp + 1)
                && self.log_gap_mean.is_finite()
                && positive(self.log_gap_sigma)
                && positive(self.log_amplitude_sigma)
                && self.dominance_ratio > 0.
                && self.dominance_ratio <= 1.,
            "multi-exponential fit needs 1..=4 states, a window and positive prior widths",
        )
    }
}

/// Channels fitted jointly with shared gaps, e.g. operator variants of one state.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChannelGroup {
    pub id: String,
    /// Channel ids, or specification ids (every measured role).
    pub channels: Vec<String>,
}
impl ChannelGroup {
    pub fn validate(&self) -> Result<()> {
        require(
            !self.id.is_empty() && (2..=16).contains(&self.channels.len()),
            "a channel group needs an id and 2..=16 channels",
        )
    }
}
named_enum! {
    /// How a state of the pencil `C(t) v = λ(t, t0) C(t0) v` is told from the
    /// others at every lag.
    GevpProjection {
        /// One eigenvector per state, solved at the reference lag and held
        /// across lags and resamples: `λ_n(t) = vₙᵀ C(t) vₙ / vₙᵀ C(t0) vₙ`,
        /// so a state keeps its identity.
        #[default]
        FixedVector => "fixed_vector",
        /// The eigenvalues of the pencil at each lag, descending, a state
        /// being whatever sits at its place in that order.
        MaxEigenvalue => "max_eigenvalue",
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GevpBasis {
    pub id: String,
    pub channels: Vec<String>,
    pub t0: usize,
    /// Relative eigenvalue cut on `C(t0)`.
    pub cut: f64,
    pub projection: GevpProjection,
    /// Lag `FixedVector` solves its eigenvectors at, beyond `t0`; the first
    /// measured lag beyond `t0` with a defined matrix when unset.
    pub t_ref: Option<usize>,
}
impl Default for GevpBasis {
    fn default() -> Self {
        Self {
            id: String::new(),
            channels: vec![],
            t0: 1,
            cut: 1e-3,
            projection: GevpProjection::FixedVector,
            t_ref: None,
        }
    }
}
impl GevpBasis {
    pub fn validate(&self) -> Result<()> {
        require(
            !self.id.is_empty()
                && (2..=16).contains(&self.channels.len())
                && self.cut.is_finite()
                && (0. ..1.).contains(&self.cut)
                && self.t_ref.is_none_or(|t| t > self.t0),
            "a GEVP basis needs an id, 2..=16 channels, a cut in [0, 1) and a reference lag \
             beyond t0",
        )
    }
}

/// Repeat the fit over start times, exponential counts and SVD cuts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StabilityScan {
    pub t_min: [usize; 2],
    pub nexp_max: usize,
    pub svd_cuts: Vec<f64>,
}
impl Default for StabilityScan {
    fn default() -> Self {
        Self {
            t_min: [1, 8],
            nexp_max: 2,
            svd_cuts: vec![1e-8, 1e-6, 1e-4],
        }
    }
}
impl StabilityScan {
    pub fn validate(&self) -> Result<()> {
        require(
            self.t_min[0] <= self.t_min[1]
                && self.t_min[1] <= 4096
                && (1..=4).contains(&self.nexp_max)
                && self.svd_cuts.len() <= 16
                && self.svd_cuts.iter().all(|c| (0. ..1.).contains(c)),
            "stability scan needs an ordered start range up to 4096, 1..=4 states and at most 16 \
             cuts in [0, 1)",
        )
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceEntry {
    pub name: String,
    pub value: f64,
    pub error: f64,
    pub source: String,
}

/// Reference values a run may be compared with under an explicit assignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceTable {
    pub unit: String,
    pub entries: Vec<ReferenceEntry>,
}
impl Default for ReferenceTable {
    fn default() -> Self {
        let pdg = "PDG 2024";
        let entry = |name: &str, value: f64, error: f64, source: &str| ReferenceEntry {
            name: name.into(),
            value,
            error,
            source: source.into(),
        };
        Self {
            unit: "MeV".into(),
            entries: vec![
                entry("pion", 139.570_39, 0.000_18, pdg),
                entry("f0_500", 500., 100., "PDG 2024 pole estimate 400-550"),
                entry("rho", 775.26, 0.23, pdg),
                entry("a1", 1230., 40., pdg),
                entry("nucleon", 938.272_088, 0.000_000_3, pdg),
                entry("glueball_0pp", 1710., 80., "quenched lattice QCD"),
                entry("electron", 0.510_998_95, 0.000_000_000_15, pdg),
                entry("muon", 105.658_375_5, 0.000_002_3, pdg),
                entry("tau", 1776.93, 0.09, pdg),
                entry("w_boson", 80_369.2, 13.3, pdg),
                entry("z_boson", 91_188.0, 2.0, pdg),
                entry("higgs", 125_200., 110., pdg),
            ],
        }
    }
}
impl ReferenceTable {
    pub fn validate(&self) -> Result<()> {
        let mut names = BTreeSet::new();
        require(
            self.entries.len() <= 256
                && self.entries.iter().all(|e| {
                    positive(e.value)
                        && e.error.is_finite()
                        && e.error >= 0.
                        && names.insert(e.name.as_str())
                }),
            "reference table takes at most 256 entries with unique names, positive values and \
             nonnegative errors",
        )
    }
    pub fn get(&self, name: &str) -> Option<&ReferenceEntry> {
        self.entries.iter().find(|e| e.name == name)
    }
}
named_enum! {
    /// Normalisation in which the hypercharge coupling is quoted. The
    /// inversion itself always uses the hypercharge coupling.
    HyperchargeNormalization {
        #[default]
        Hypercharge => "hypercharge",
        /// Multiplied by `sqrt(5/3)` after the inversion.
        Unified => "unified",
    }
}

/// Reference couplings the calibration dictionary inverts into gas
/// parameters. They are inputs of an inversion, never measurements.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StandardModelInputs {
    /// `[value, error]` of `1/α_em`, `sin²θ_W` and `α_s` at the common `scale`.
    /// The Thomson-limit `1/α_em = 137.035999084` reproduces the reference
    /// script, which mixes scales.
    pub alpha_em_inverse: [f64; 2],
    pub sin2_theta_w: [f64; 2],
    pub alpha_s: [f64; 2],
    pub scale: String,
    pub hypercharge: HyperchargeNormalization,
    pub source: String,
}
impl Default for StandardModelInputs {
    fn default() -> Self {
        Self {
            alpha_em_inverse: [127.951, 0.009],
            sin2_theta_w: [0.231_21, 0.000_04],
            alpha_s: [0.1179, 0.0009],
            scale: "M_Z".into(),
            hypercharge: HyperchargeNormalization::Hypercharge,
            source: "PDG 2022 electroweak review".into(),
        }
    }
}
impl StandardModelInputs {
    pub fn validate(&self) -> Result<()> {
        require(
            [self.alpha_em_inverse, self.sin2_theta_w, self.alpha_s]
                .iter()
                .all(|[value, error]| positive(*value) && error.is_finite() && *error >= 0.)
                && self.sin2_theta_w[0] < 1.,
            "Standard Model inputs must be positive with nonnegative errors and sin2_theta_w < 1",
        )
    }
}
/// The part of a member id beyond the specification it names: empty for the
/// specification itself, `/<element kind>` for one of its channels.
fn role<'a>(member: &'a str, spec_id: &str) -> Option<&'a str> {
    member
        .strip_prefix(spec_id)
        .filter(|rest| rest.is_empty() || rest.starts_with('/'))
}
/// Whether two members, read against the two specifications of one twin
/// relation, reach a common channel. A member that names a specification
/// stands for every measured role of it, so it meets any role of the twin;
/// two members that each name a role meet on that role alone.
fn overlap(a: &str, b: &str, [one, other]: &[String; 2]) -> bool {
    match (role(a, one), role(b, other)) {
        (Some(x), Some(y)) => x.is_empty() || y.is_empty() || x == y,
        _ => false,
    }
}
/// The two members of a group or basis that measure the same series, under
/// the `[id, twin id]` relation of `ChannelSpec::twin`.
fn twinned<'a>(members: &'a [String], twins: &[[String; 2]]) -> Option<[&'a str; 2]> {
    members
        .iter()
        .flat_map(|a| members.iter().map(move |b| [a.as_str(), b.as_str()]))
        .find(|[a, b]| twins.iter().any(|twin| overlap(a, b, twin)))
}
/// Two members of one group or basis that measure the same series leave the
/// joint covariance an exact null direction, which no eigenvalue cut tells
/// from a resolved one.
fn distinct(id: &str, members: &[String], twins: &[[String; 2]]) -> Result<()> {
    match twinned(members, twins) {
        Some([one, other]) => Err(GasError::Configuration(format!(
            "`{id}` holds `{one}` and `{other}`, which measure the same series"
        ))),
        None => Ok(()),
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalysisConfig {
    /// Subtract the disconnected part, recomputed inside every resample.
    pub connected: bool,
    /// Disconnected part of a connected frame-mean correlator: the
    /// lag-dependent leg means, or one global mean. They differ at order
    /// `1/T`. `Subtraction::None` is rejected (clear `connected` instead).
    pub frame_subtraction: Subtraction,
    /// The same for a source-frozen propagator, where the single source mean
    /// leaves the floor `(ā_ℓ − Ō)(b̄_ℓ − Ō)` when sink masks depend on the
    /// lag; a fit on it reports that floor.
    pub propagator_subtraction: Subtraction,
    pub resampling: Resampling,
    /// Eigenvalues of the lag correlation matrix are floored at
    /// `svd_cut · λ_max`. A correlated window scan is biased low below about
    /// `1e-2` at typical block counts.
    pub svd_cut: f64,
    pub combine: Combine,
    pub estimator: EstimatorChoice,
    pub effective_mass: EffectiveMassKind,
    pub fit: FitMethod,
    pub window_scan: WindowScanConfig,
    pub multi_exponential: MultiExponentialConfig,
    pub groups: Vec<ChannelGroup>,
    pub gevp: Vec<GevpBasis>,
    pub stability: Option<StabilityScan>,
    pub reference: ReferenceTable,
    pub standard_model: StandardModelInputs,
    /// Channel or specification id → reference name. A hypothesis, never a result.
    pub assignments: BTreeMap<String, String>,
    /// Reference names that set the scale, one prediction table per anchor.
    pub anchors: Vec<String>,
    pub time_unit: TimeUnit,
    /// Channel or specification ids to analyse; empty analyses everything measured.
    pub channels: Vec<String>,
    /// Include the full lag covariance in every correlator of the report.
    pub report_covariance: bool,
}
impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            connected: true,
            frame_subtraction: Subtraction::GlobalMean,
            propagator_subtraction: Subtraction::LagMeans,
            resampling: Resampling::default(),
            svd_cut: 1e-2,
            combine: Combine::default(),
            estimator: EstimatorChoice::default(),
            effective_mass: EffectiveMassKind::default(),
            fit: FitMethod::default(),
            window_scan: WindowScanConfig::default(),
            multi_exponential: MultiExponentialConfig::default(),
            groups: vec![],
            gevp: vec![],
            stability: None,
            reference: ReferenceTable::default(),
            standard_model: StandardModelInputs::default(),
            assignments: [
                ("meson/pseudoscalar/standard", "pion"),
                ("meson/scalar/standard", "f0_500"),
                ("vector/vector/full/raw", "rho"),
                ("vector/axial/full/raw", "a1"),
                ("baryon/complex", "nucleon"),
                ("glueball/re_plaquette", "glueball_0pp"),
            ]
            .into_iter()
            .map(|(channel, name)| (channel.to_string(), name.to_string()))
            .collect(),
            anchors: vec!["nucleon".into()],
            time_unit: TimeUnit::default(),
            channels: vec![],
            report_covariance: false,
        }
    }
}
impl AnalysisConfig {
    pub fn validate(&self) -> Result<()> {
        self.resampling.validate()?;
        require(
            self.time_unit != TimeUnit::Coordinate,
            "coordinate time unit is report only",
        )?;
        require(
            self.frame_subtraction != Subtraction::None
                && self.propagator_subtraction != Subtraction::None,
            "a connected correlator needs a subtraction; clear connected instead",
        )?;
        require(
            self.svd_cut.is_finite() && (0. ..1.).contains(&self.svd_cut),
            "SVD cut must lie in [0, 1)",
        )?;
        self.window_scan.validate()?;
        self.multi_exponential.validate()?;
        require(
            self.groups.len() <= 64 && self.gevp.len() <= 64,
            "analysis takes at most 64 channel groups and 64 GEVP bases",
        )?;
        let mut ids = BTreeSet::new();
        let twins: Vec<[String; 2]> = ChannelSpec::all()
            .iter()
            .filter_map(|spec| spec.twin().map(|twin| [spec.id(), twin.id()]))
            .collect();
        for group in &self.groups {
            group.validate()?;
            require(
                ids.insert(&group.id),
                format!("duplicate group or GEVP basis `{}`", group.id),
            )?;
            distinct(&group.id, &group.channels, &twins)?;
        }
        for basis in &self.gevp {
            basis.validate()?;
            require(
                ids.insert(&basis.id),
                format!("duplicate group or GEVP basis `{}`", basis.id),
            )?;
            distinct(&basis.id, &basis.channels, &twins)?;
        }
        if let Some(scan) = &self.stability {
            scan.validate()?;
        }
        self.reference.validate()?;
        require(
            self.assignments
                .values()
                .chain(&self.anchors)
                .all(|name| self.reference.get(name).is_some()),
            "assignments and anchors must name reference entries",
        )?;
        self.standard_model.validate()
    }
    /// Largest lag any configured window names explicitly.
    pub fn max_lag(&self) -> Option<usize> {
        [
            self.window_scan.t_max,
            self.multi_exponential.t_max,
            self.gevp
                .iter()
                .flat_map(|b| [Some(b.t0), b.t_ref])
                .max()
                .flatten(),
        ]
        .into_iter()
        .flatten()
        .max()
    }
    /// Whether a channel is selected by `channels`.
    pub fn selects(&self, spec_id: &str, channel_id: &str) -> bool {
        self.channels.is_empty()
            || self
                .channels
                .iter()
                .any(|c| c == spec_id || c == channel_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wire<T: Serialize>(value: &T) -> String {
        serde_json::to_value(value)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }
    #[test]
    fn default_config_validates_and_round_trips_through_json() {
        let config = SpectroscopyConfig::default();
        config.validate().unwrap();
        let text = serde_json::to_string(&config).unwrap();
        let back: SpectroscopyConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, config);
        assert_eq!(serde_json::to_string(&back).unwrap(), text);
    }
    #[test]
    fn an_empty_object_is_the_default_config_and_unknown_fields_are_rejected() {
        let empty: SpectroscopyConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, SpectroscopyConfig::default());
        assert!(serde_json::from_str::<SpectroscopyConfig>(r#"{"measurment":{}}"#).is_err());
        assert!(
            serde_json::from_str::<MeasurementConfig>(
                r#"{"color":{"kind":"viscous_force","x":1}}"#
            )
            .is_err()
        );
        let partial: MeasurementConfig =
            serde_json::from_str(r#"{"channels":[{"kind":"u1"},{"kind":"fitness_phase"}]}"#)
                .unwrap();
        assert_eq!(partial.channels[0].id(), "u1/phase/q1");
        assert_eq!(partial.channels[1], ChannelSpec::FitnessPhase);
    }
    #[test]
    fn standard_set_and_catalog_ids_are_unique_and_every_arm_round_trips() {
        for set in [ChannelSpec::standard_set(), ChannelSpec::all()] {
            let ids: BTreeSet<String> = set.iter().map(ChannelSpec::id).collect();
            assert_eq!(ids.len(), set.len());
            for spec in &set {
                spec.validate().unwrap();
                assert!(spec.id().starts_with(spec.family()));
                let text = serde_json::to_string(spec).unwrap();
                assert_eq!(&serde_json::from_str::<ChannelSpec>(&text).unwrap(), spec);
            }
        }
        let all: BTreeSet<String> = ChannelSpec::all().iter().map(ChannelSpec::id).collect();
        assert!(
            ChannelSpec::standard_set()
                .iter()
                .all(|s| all.contains(&s.id()))
        );
        let momentum = ChannelSpec::Glueball {
            observable: GlueballObservable::RePlaquette,
            momentum: Some(Momentum {
                axis: 2,
                mode: 1,
                phase: MomentumPhase::Sin,
            }),
        };
        assert_eq!(momentum.id(), "glueball/re_plaquette/p2_sin1");
    }
    #[test]
    fn identifier_segments_equal_the_serde_wire_names() {
        macro_rules! check {
            ($($name:ident),+) => {
                $(for &v in $name::ALL {
                    assert_eq!(wire(&v), v.name());
                })+
            };
        }
        check!(
            StepAlignment,
            ChiralProjector,
            RoleClass,
            PhaseLink,
            KickStage,
            PairSelection,
            CompanionChoice,
            HistoricalPolicy,
            IdentityPolicy,
            FrameNormalization,
            ScaleMode,
            EdgeLength,
            MesonQuantum,
            MesonMode,
            VectorQuantum,
            VectorProjection,
            Displacement,
            BaryonMode,
            GlueballObservable,
            MomentumPhase,
            TensorMode,
            DiracGamma,
            U1Mode,
            Su2Mode,
            ChiralityObservable,
            TwistorObservable,
            Combine,
            EstimatorChoice,
            EffectiveMassKind,
            FitMethod,
            TimeUnit,
            GevpProjection,
            PairDistance,
            HyperchargeNormalization
        );
        for role in [
            WalkerRole::Cloner,
            WalkerRole::StrongResister,
            WalkerRole::WeakResister,
            WalkerRole::Persister,
        ] {
            assert_eq!(wire(&role), role.name());
        }
    }
    #[test]
    fn validation_rejects_vanishing_modes_duplicates_and_unknown_references() {
        let sine_zero = ChannelSpec::Glueball {
            observable: GlueballObservable::RePlaquette,
            momentum: Some(Momentum {
                axis: 0,
                mode: 0,
                phase: MomentumPhase::Sin,
            }),
        };
        assert!(sine_zero.validate().is_err());
        let mut measurement = MeasurementConfig::default();
        measurement.channels.push(ChannelSpec::FitnessPhase);
        assert!(measurement.validate().is_err());
        let mut analysis = AnalysisConfig::default();
        analysis.anchors.push("kaon".into());
        assert!(analysis.validate().is_err());
        let mut config = SpectroscopyConfig::default();
        config.analysis.window_scan.t_max = Some(config.measurement.max_lag + 1);
        assert!(config.validate().is_err());
    }
    #[test]
    fn fingerprint_ignores_the_seed_and_tracks_the_measurement() {
        let measurement = MeasurementConfig::default();
        let gas = GasConfig::default();
        let reseeded = GasConfig {
            seed: 99,
            ..GasConfig::default()
        };
        let caps = Capabilities::of(&gas, &crate::RecordingConfig::default(), 3);
        let a = measurement.fingerprint(&gas, &caps, &[]).unwrap();
        assert_eq!(a, measurement.fingerprint(&reseeded, &caps, &[]).unwrap());
        let shorter = MeasurementConfig {
            max_lag: 40,
            ..MeasurementConfig::default()
        };
        assert_ne!(a, shorter.fingerprint(&gas, &caps, &[]).unwrap());
        let planar = Capabilities::of(&gas, &crate::RecordingConfig::default(), 2);
        assert_ne!(a, measurement.fingerprint(&gas, &planar, &[]).unwrap());
        let injected = ["custom/probe/site".to_string()];
        assert_ne!(a, measurement.fingerprint(&gas, &caps, &injected).unwrap());
    }
    #[test]
    fn option_arms_extend_identifiers_without_changing_the_plain_ones() {
        let plain: ChannelSpec =
            serde_json::from_str(r#"{"kind":"dirac","gamma":"vector"}"#).unwrap();
        assert_eq!(plain.id(), "dirac/vector");
        let yukawa = ChannelSpec::Dirac {
            gamma: DiracGamma::Scalar,
            projector: ChiralProjector::Left,
            pairs: RoleClass::LeftRight,
            link: PhaseLink::None,
        };
        assert_eq!(yukawa.id(), "dirac/scalar/left/left_right");
        let current = ChannelSpec::Dirac {
            gamma: DiracGamma::Vector,
            projector: ChiralProjector::Left,
            pairs: RoleClass::Any,
            link: PhaseLink::Su2,
        };
        assert_eq!(current.id(), "dirac/vector/left/su2");
        let twistor: ChannelSpec =
            serde_json::from_str(r#"{"kind":"twistor","observable":"tensor"}"#).unwrap();
        assert_eq!(
            twistor,
            ChannelSpec::Twistor {
                observable: TwistorObservable::Tensor,
                velocity_scale: 1.
            }
        );
        let scaled = |velocity_scale| ChannelSpec::Twistor {
            observable: TwistorObservable::Tensor,
            velocity_scale,
        };
        assert_eq!(scaled(0.5).id(), "twistor/tensor/a0.5");
        assert!(scaled(0.).validate().is_err());
        let recorded: ColorSource = serde_json::from_str(
            r#"{"kind":"recorded_field","stage":"kick","amplitude":"f","phase":"v"}"#,
        )
        .unwrap();
        assert!(matches!(
            recorded,
            ColorSource::RecordedField {
                alignment: StepAlignment::Preceding,
                ..
            }
        ));
    }
    #[test]
    fn custom_channels_are_named_outside_the_catalog_and_report_only_units_are_rejected() {
        let custom = ChannelSpec::Custom { id: "probe".into() };
        custom.validate().unwrap();
        assert_eq!(
            (custom.id().as_str(), custom.family()),
            ("custom/probe", "custom")
        );
        for bad in ["", "a/b", "a@1"] {
            assert!(ChannelSpec::Custom { id: bad.into() }.validate().is_err());
        }
        assert!(
            ChannelSpec::all()
                .iter()
                .all(|s| !matches!(s, ChannelSpec::Custom { .. }))
        );
        let injected_only = MeasurementConfig {
            channels: vec![],
            ..MeasurementConfig::default()
        };
        injected_only.validate().unwrap();
        let coordinate = AnalysisConfig {
            time_unit: TimeUnit::Coordinate,
            ..AnalysisConfig::default()
        };
        assert!(coordinate.validate().is_err());
        let raw = AnalysisConfig {
            propagator_subtraction: Subtraction::None,
            ..AnalysisConfig::default()
        };
        assert!(raw.validate().is_err());
        assert!(
            PropagatorConfig::default().enabled("tensor/components", "tensor/components/distance")
        );
    }
    #[test]
    fn documented_alternatives_are_arms_with_literal_wire_names() {
        let alignment: ColorAlignment =
            serde_json::from_str(r#"{"kind":"preceding_force"}"#).unwrap();
        assert_eq!(alignment, ColorAlignment::PrecedingForce);
        let legacy: ColorSource = serde_json::from_str(
            r#"{"kind":"viscous_force","alignment":{"kind":"reference_offset"}}"#,
        )
        .unwrap();
        assert_eq!(
            legacy,
            ColorSource::ViscousForce {
                alignment: ColorAlignment::ReferenceOffset,
                threshold: 1e-12
            }
        );
        let gamma = |projection| ChannelSpec::Vector {
            quantum: VectorQuantum::Vector,
            projection,
            displacement: Displacement::ColorGamma,
        };
        assert_eq!(
            gamma(VectorProjection::Full).id(),
            "vector/vector/full/color_gamma"
        );
        gamma(VectorProjection::Full).validate().unwrap();
        assert!(matches!(
            gamma(VectorProjection::Transverse).validate(),
            Err(GasError::Configuration(_))
        ));
        let complex: ChannelSpec = serde_json::from_str(r#"{"kind":"baryon"}"#).unwrap();
        assert_eq!(complex.id(), "baryon/complex");
        assert!(ChannelSpec::standard_set().contains(&complex));
        assert!(PropagatorConfig::default().enabled("baryon/complex", "baryon/complex/triplet"));
        assert_eq!(
            AnalysisConfig::default().assignments["baryon/complex"],
            "nucleon"
        );
        let flux = ChannelSpec::Baryon {
            mode: BaryonMode::FluxWeighted,
            flux_alpha: 2.,
        };
        assert_eq!(flux.id(), "baryon/flux_weighted/a2");
        let mean: ChannelSpec =
            serde_json::from_str(r#"{"kind":"twistor","observable":"tensor_mean"}"#).unwrap();
        assert_eq!(mean.id(), "twistor/tensor_mean");
        let scales: ElectroweakScales =
            serde_json::from_str(r#"{"distance":"configured"}"#).unwrap();
        assert_eq!(scales.distance, PairDistance::Configured);
    }
    #[test]
    fn child_configurations_validate_alone_and_scan_budgets_are_literal_limits() {
        FlowConfig::default().validate().unwrap();
        let stalled = FlowConfig {
            step_size: 0.,
            ..FlowConfig::default()
        };
        assert!(stalled.validate().is_err());
        PropagatorConfig::default().validate(80).unwrap();
        assert!(PropagatorConfig::default().validate(0).is_err());
        assert!(MeasurementBudget::default().validate(20_000).is_err());
        assert!(GevpBasis::default().validate().is_err());
        let quantiles = ScaleSelection::Quantiles {
            count: 4,
            low: 0.9,
            high: 0.1,
        };
        assert!(quantiles.validate().is_err());
        StabilityScan::default().validate().unwrap();
        let wide = StabilityScan {
            svd_cuts: vec![1e-6; 17],
            ..StabilityScan::default()
        };
        let late = StabilityScan {
            t_min: [1, 4097],
            ..StabilityScan::default()
        };
        for scan in [wide, late] {
            let message = scan.validate().unwrap_err().to_string();
            assert!(message.contains("4096") && message.contains("16"));
        }
        let group = |id: &str| ChannelGroup {
            id: id.into(),
            channels: vec!["meson/scalar/standard".into(), "meson/scalar/abs2".into()],
        };
        group("scalars").validate().unwrap();
        let twice = AnalysisConfig {
            groups: vec![group("scalars"), group("scalars")],
            ..AnalysisConfig::default()
        };
        assert!(matches!(twice.validate(), Err(GasError::Configuration(_))));
        let message = twice.validate().unwrap_err().to_string();
        assert!(message.contains("`scalars`"));
        let inputs = StandardModelInputs {
            alpha_s: [0., 0.],
            ..StandardModelInputs::default()
        };
        assert!(inputs.validate().unwrap_err().to_string().is_ascii());
    }
    #[test]
    fn analysis_defaults_follow_the_book_centring_and_reject_bad_reference_inputs() {
        let analysis = AnalysisConfig::default();
        assert_eq!(analysis.frame_subtraction, Subtraction::GlobalMean);
        assert_eq!(analysis.propagator_subtraction, Subtraction::LagMeans);
        assert_eq!(analysis.svd_cut, 1e-2);
        assert_eq!(analysis.standard_model.alpha_em_inverse, [127.951, 0.009]);
        let raw = AnalysisConfig {
            frame_subtraction: Subtraction::None,
            ..AnalysisConfig::default()
        };
        assert!(raw.validate().is_err());
        let mut inputs = AnalysisConfig::default();
        inputs.standard_model.sin2_theta_w = [1.2, 0.];
        assert!(matches!(inputs.validate(), Err(GasError::Configuration(_))));
        assert!(serde_json::from_str::<StandardModelInputs>(r#"{"alpha":1}"#).is_err());
    }
}
