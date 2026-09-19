//! Result types. They serialize to plain JSON for the browser: an undefined
//! number is `None`, never NaN and never a placeholder zero, and every number
//! present is finite, which `SpectroscopyReport::validate` enforces.
//!
//! The fitted quantity is the decay rate of the algorithm-time
//! autocorrelation. It is a mass only under a positive self-adjoint transfer
//! representation; `notes` say so.
use super::{
    config::{AnalysisConfig, ChannelSpec, Combine, TimeUnit},
    contract::{
        Availability, Capabilities, ElementKind, ExchangeParity, SPECTROSCOPY_VERSION,
        SpatialParity,
    },
};
use crate::{GasError, Precision, Result, error::require, physics::numerics::ResampleKind};
use serde::{Deserialize, Serialize};

/// Label of every comparison with reference data.
pub const HYPOTHESIS_LABEL: &str = "hypothesis mapping";
/// What a fitted rate is called in reports, per estimator. Only the frame
/// correlator has a transfer-matrix reading.
pub const RATE_QUANTITY: &str = "decay rate of the algorithm-time autocorrelation";
pub const RATE_QUANTITY_SOURCE_FROZEN: &str =
    "decay rate of the source-frozen pair correlator (no transfer-matrix reading)";
pub const RATE_QUANTITY_EUCLIDEAN: &str = "decay rate of the Euclidean-coordinate slab correlator";
/// `SpectroscopyReport::calculation_origin`: every number comes from recorded steps.
pub const CALCULATION_ORIGIN: &str = "executed_algorithm_archive";

/// Every number held by a report value is finite.
trait Finite {
    fn finite(&self) -> bool;
}
impl Finite for f64 {
    fn finite(&self) -> bool {
        self.is_finite()
    }
}
impl<T: Finite> Finite for Option<T> {
    fn finite(&self) -> bool {
        self.as_ref().is_none_or(Finite::finite)
    }
}
impl<T: Finite> Finite for Vec<T> {
    fn finite(&self) -> bool {
        self.iter().all(Finite::finite)
    }
}
impl<T: Finite> Finite for [T; 2] {
    fn finite(&self) -> bool {
        self.iter().all(Finite::finite)
    }
}

/// Element counts of one channel, accumulated over frames. For a propagator
/// the identity and colour masks also count sink-time failures.
///
/// `topology::build` counts `masked_historical`, `masked_ineligible` and
/// `masked_self`. The accumulator attributes every `evaluate == false`: at the
/// source time to `masked_color` when `operator.requires()` contains
/// `Record::Color`, else to `masked_ineligible`; at a sink time to
/// `masked_identity` when the generation changed under
/// `IdentityPolicy::Incarnation`, else by the same rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Coverage {
    /// Frames that contributed at least one valid element.
    pub frames: u64,
    /// Frames with no valid element (weight 0 in the series).
    pub empty_frames: u64,
    pub valid: u64,
    pub masked_historical: u64,
    pub masked_ineligible: u64,
    pub masked_color: u64,
    pub masked_identity: u64,
    pub masked_self: u64,
    /// Removed by a multiscale gate.
    pub masked_scale: u64,
}
impl Coverage {
    pub fn merge(&mut self, other: &Self) {
        self.frames += other.frames;
        self.empty_frames += other.empty_frames;
        self.valid += other.valid;
        self.masked_historical += other.masked_historical;
        self.masked_ineligible += other.masked_ineligible;
        self.masked_color += other.masked_color;
        self.masked_identity += other.masked_identity;
        self.masked_self += other.masked_self;
        self.masked_scale += other.masked_scale;
    }
    pub fn masked(&self) -> u64 {
        self.masked_historical
            + self.masked_ineligible
            + self.masked_color
            + self.masked_identity
            + self.masked_self
            + self.masked_scale
    }
}

/// Scales fixed causally during the warm-up frames and frozen afterwards.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Calibration {
    pub warmup_frames: usize,
    /// Length `ℓ₀` of the colour phase and how it was obtained.
    pub length: f64,
    pub length_source: String,
    /// Phase factor `κ = m ℓ₀ / ħ_eff` in `c_a ∝ F_a e^{i κ v_a}`.
    pub kappa: f64,
    /// Fraction of warm-up phase-velocity components with `|κ v| > π`.
    pub phase_wrapping: Option<f64>,
    /// Mass `m` of the colour phase, from `PhaseScale::mass`.
    pub mass: f64,
    /// Action scale of the colour phase, `PhaseScale::h_eff`.
    pub h_eff: f64,
    /// Action scales of the U(1) and SU(2) phases, from `ElectroweakScales`.
    pub electroweak_h_eff: f64,
    pub h_s: f64,
    /// Interaction ranges and the clone regulariser, where defined.
    pub epsilon_d: Option<f64>,
    pub epsilon_c: Option<f64>,
    pub epsilon_clone: f64,
    /// Integrator time step, when the kinetic operator has one.
    pub dt: Option<f64>,
    /// Bin range of the Euclidean-time axis.
    pub euclidean_range: Option<[f64; 2]>,
    /// Geodesic scales of the multiscale gate, ascending.
    pub scales: Vec<f64>,
    /// `N₁ = E_μ exp(−D²/ε_d²)`, the squared companion kernel weight and not
    /// the amplitude `exp(−D²/4ε_d²)`, entering `g₁² = ħ_eff N₁ / ε_d²`. The
    /// law `μ` is uniform over the valid first distance-companion pairs
    /// (`Companions::first`) of the warm-up frames, `D` is the distance of
    /// `ElectroweakScales::{distance, lambda}` and `ε_d` the resolved
    /// `epsilon_d` of this calibration. `None` without that range.
    pub pair_weight_n1: Option<f64>,
    /// `E_μ K²` entering `g₃² = (ν²/ħ_eff²) d(d²−1)/12 · E_μ K²`, with the
    /// unnormalised kernel `K_ij = exp(−|x_i − x_j|²/2ρ²) ≤ 1` on the raw
    /// position difference the dense viscous force uses, `ρ` its bandwidth,
    /// and `μ` uniform over the ordered eligible pairs `i ≠ j` of the warm-up
    /// frames. The engine divides `K` by the eligible count or by the row
    /// sum, so this moment bounds the engine's from above. `None` without
    /// dense viscosity, graph viscosity included.
    pub viscous_kernel_second_moment: Option<f64>,
}
impl Finite for Calibration {
    fn finite(&self) -> bool {
        [
            self.length,
            self.kappa,
            self.mass,
            self.h_eff,
            self.electroweak_h_eff,
            self.h_s,
            self.epsilon_clone,
        ]
        .iter()
        .all(|x| x.is_finite())
            && [
                self.phase_wrapping,
                self.epsilon_d,
                self.epsilon_c,
                self.dt,
                self.pair_weight_n1,
                self.viscous_kernel_second_moment,
            ]
            .iter()
            .all(Finite::finite)
            && self.euclidean_range.finite()
            && self.scales.finite()
    }
}
impl Calibration {
    /// `Measurement::validate` and `SpectroscopyReport::validate` call it, and
    /// the accumulator for the value of `Extensions::calibration`.
    pub fn validate(&self) -> Result<()> {
        require(self.finite(), "nonfinite calibration")
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimatorKind {
    /// Correlation of the frame averages `A_t(O)` over algorithm time.
    #[default]
    FrameMean,
    /// `Σ_I w_I O_I(t) O_I(t+τ)` over the topology frozen at the source time.
    SourceFrozen,
    /// Correlation between Euclidean-time slabs of single frames.
    EuclideanTime,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplesMeta {
    pub resampling: ResampleKind,
    /// Time origins per resampling block and the number of blocks.
    pub effective_block: usize,
    pub blocks: usize,
    pub tau_int: Option<f64>,
    /// Numerical rank of the covariance against its dimension `lags`.
    pub covariance_rank: usize,
    pub replicas: usize,
    /// What one resampled unit is, `SamplesMeta::sampling_unit`. Errors over
    /// time blocks are not standard errors over independently seeded runs.
    pub sampling_unit: String,
}
impl SamplesMeta {
    /// The statement stored in `sampling_unit`.
    pub fn sampling_unit(combine: Combine, effective_block: usize, replicas: usize) -> String {
        match combine {
            Combine::PooledBlocks => format!(
                "time blocks of {effective_block} origins pooled over {replicas} independently \
                 seeded runs"
            ),
            Combine::RunsAsSamples => "independently seeded runs".into(),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelatorEstimate {
    /// Lags in frames (bins under `TimeUnit::Coordinate`); multiply by
    /// `time_step` for the unit of `time_unit`. Not contiguous under a
    /// propagator `lag_stride > 1`.
    pub lags: Vec<usize>,
    pub time_unit: TimeUnit,
    pub time_step: f64,
    pub value: Vec<Option<f64>>,
    pub error: Vec<Option<f64>>,
    /// `[lags, lags]`, zero rows where undefined. `estimators::estimate`
    /// always fills it; `analyze` drops it from the report unless
    /// `AnalysisConfig::report_covariance`.
    pub covariance: Option<Vec<f64>>,
    pub samples_meta: SamplesMeta,
    pub connected: bool,
    /// Leading bias `−2 τ_int σ² / T` of the connected estimator.
    pub connected_bias: Option<f64>,
}
impl Finite for CorrelatorEstimate {
    fn finite(&self) -> bool {
        self.time_step.is_finite()
            && self.value.finite()
            && self.error.finite()
            && self.covariance.finite()
            && self.samples_meta.tau_int.finite()
            && self.connected_bias.finite()
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitMethodKind {
    #[default]
    WindowScan,
    MultiExponential,
    Gevp,
    /// A stability scan: a table of variations, never a rate.
    Stability,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MassEstimate {
    /// `RATE_QUANTITY`, or the constant of the estimator that produced it.
    pub quantity: String,
    pub value: f64,
    /// `sqrt(statistical² + systematic²)`.
    pub error: f64,
    pub statistical: f64,
    /// Spread over fit windows or models.
    pub systematic: f64,
    pub method: FitMethodKind,
    pub time_unit: TimeUnit,
    /// Prior dominance of this level's gap parameter; `None` for a
    /// prior-free method. A dominated ground state reports no `mass` at all,
    /// so `dominated == true` appears only in `excited` and `levels`.
    #[serde(default)]
    pub prior_dominance: Option<PriorDominance>,
}
impl Finite for MassEstimate {
    fn finite(&self) -> bool {
        [self.value, self.error, self.statistical, self.systematic]
            .iter()
            .all(|x| x.is_finite())
            && self.prior_dominance.finite()
    }
}

/// How far the data moved a Gaussian prior.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriorDominance {
    /// Posterior width over prior width of the parameter.
    pub width_ratio: f64,
    /// Posterior mean shift in units of the prior width.
    pub shift_sigma: f64,
    /// The fit returned its prior; no mass is reported.
    pub dominated: bool,
}
impl Finite for PriorDominance {
    fn finite(&self) -> bool {
        self.width_ratio.is_finite() && self.shift_sigma.is_finite()
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitDiagnostics {
    pub chi2: Option<f64>,
    pub dof: Option<usize>,
    pub q: Option<f64>,
    /// `[t_min, t_max]` in frames of the selected or best window.
    pub window: Option<[usize; 2]>,
    pub n_windows: usize,
    /// False when a diagonal χ² replaced the correlated one; `notes` say why.
    pub correlated: bool,
    pub svd_cut: f64,
    pub covariance_rank: Option<usize>,
    /// Of the ground-state gap.
    pub prior_dominance: Option<PriorDominance>,
    /// Oscillating or sign-changing correlator: the exponential model is rejected.
    pub model_rejected: Option<String>,
    /// Signal-to-noise below threshold: no rate is reported.
    pub no_signal: Option<String>,
}
impl Finite for FitDiagnostics {
    fn finite(&self) -> bool {
        self.chi2.finite()
            && self.q.finite()
            && self.svd_cut.is_finite()
            && self.prior_dominance.finite()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowFit {
    pub t_min: usize,
    pub t_max: usize,
    pub value: f64,
    pub error: f64,
    pub chi2: f64,
    pub dof: usize,
    /// Normalized model weight `∝ exp(−AIC/2)`, `AIC = χ² + 2k + 2 N_cut`.
    pub weight: f64,
    /// Exponentials of the row's model and its SVD cut, which tell the rows
    /// of a stability scan apart. 1 and `None` in a window scan.
    #[serde(default = "one_exponential")]
    pub nexp: usize,
    #[serde(default)]
    pub svd_cut: Option<f64>,
}
fn one_exponential() -> usize {
    1
}
impl Finite for WindowFit {
    fn finite(&self) -> bool {
        [self.value, self.error, self.chi2, self.weight]
            .iter()
            .all(|x| x.is_finite())
            && self.svd_cut.finite()
    }
}

/// Result of one fit method on one channel. `mass` is `None` whenever
/// `diagnostics` rejects the model, finds no signal or a dominated prior.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitOutcome {
    pub method: FitMethodKind,
    pub mass: Option<MassEstimate>,
    /// Excited levels of a multi-exponential fit, ascending.
    pub excited: Vec<MassEstimate>,
    pub diagnostics: FitDiagnostics,
    pub windows: Vec<WindowFit>,
    pub notes: Vec<String>,
}
impl Finite for FitOutcome {
    fn finite(&self) -> bool {
        self.mass.finite()
            && self.excited.finite()
            && self.diagnostics.finite()
            && self.windows.finite()
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChannelReport {
    pub id: String,
    pub spec: ChannelSpec,
    pub kind: Option<ElementKind>,
    /// Geodesic scale of a multiscale copy of the channel.
    pub scale: Option<f64>,
    pub definition: String,
    pub book_label: String,
    pub exchange: Option<ExchangeParity>,
    pub spatial_parity: Option<SpatialParity>,
    pub availability: Availability,
    pub coverage: Coverage,
    pub estimator: Option<EstimatorKind>,
    pub correlator: Option<CorrelatorEstimate>,
    /// `[value, error]` per lag of `correlator`.
    pub effective_mass: Option<Vec<Option<[f64; 2]>>>,
    /// The first fit of `fits` that reports a rate.
    pub mass: Option<MassEstimate>,
    pub fits: Vec<FitOutcome>,
    pub notes: Vec<String>,
}
impl Finite for ChannelReport {
    fn finite(&self) -> bool {
        self.scale.finite()
            && self.correlator.finite()
            && self.effective_mass.finite()
            && self.mass.finite()
            && self.fits.finite()
    }
}

/// Joint fit of several channels with shared energy gaps.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GroupFit {
    pub id: String,
    pub channels: Vec<String>,
    pub availability: Availability,
    /// Cumulative levels `E_n = Σ_{m ≤ n} dE_m`, ascending.
    pub levels: Vec<MassEstimate>,
    pub diagnostics: FitDiagnostics,
    pub notes: Vec<String>,
}
impl Finite for GroupFit {
    fn finite(&self) -> bool {
        self.levels.finite() && self.diagnostics.finite()
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GevpReport {
    pub id: String,
    pub channels: Vec<String>,
    pub availability: Availability,
    pub t0: usize,
    pub lags: Vec<usize>,
    /// `[state][lag]` generalized eigenvalues `[value, error]`.
    pub eigenvalues: Vec<Vec<Option<[f64; 2]>>>,
    /// `[state][lag]` effective rates `[value, error]`.
    pub effective_mass: Vec<Vec<Option<[f64; 2]>>>,
    /// One fit per state, ascending, with its own diagnostics (rejected
    /// model, no signal) and window table.
    pub levels: Vec<FitOutcome>,
    /// Basis directions kept after the cut on `C(t0)`.
    pub rank: usize,
    /// `‖C − Cᵀ‖_F / ‖C + Cᵀ‖_F` per entry of `lags`: what symmetrization
    /// discarded. It grows with the lag on a non-reversible chain.
    pub antisymmetric_norm: Vec<Option<f64>>,
    pub notes: Vec<String>,
}
impl Finite for GevpReport {
    fn finite(&self) -> bool {
        self.eigenvalues.finite()
            && self.effective_mass.finite()
            && self.levels.finite()
            && self.antisymmetric_norm.finite()
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceRow {
    pub name: String,
    pub channel: String,
    pub reference: f64,
    pub reference_error: f64,
    pub unit: String,
    /// Measured rate in lattice units, `[value, error]`.
    pub measured: Option<[f64; 2]>,
    /// Estimator behind `measured`; `reference::compare` notes every ratio
    /// that mixes estimators.
    pub estimator: Option<EstimatorKind>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Prediction {
    pub name: String,
    /// `[value, error]` in the reference unit after anchor rescaling.
    pub predicted: Option<[f64; 2]>,
    pub reference: f64,
    pub tension_sigma: Option<f64>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnchorPrediction {
    pub anchor: String,
    /// Reference units per lattice unit, `[value, error]`.
    pub scale: Option<[f64; 2]>,
    pub predictions: Vec<Prediction>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RatioRow {
    pub numerator: String,
    pub denominator: String,
    pub measured: Option<[f64; 2]>,
    pub reference: f64,
    pub tension_sigma: Option<f64>,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnchorSpread {
    pub name: String,
    /// Relative spread of the prediction over anchors.
    pub spread: Option<f64>,
}

/// Comparison of assigned channels with a reference table. Always a
/// hypothesis mapping: the assignment is an input, not a result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Comparison {
    /// `HYPOTHESIS_LABEL`.
    pub label: String,
    pub reference: Vec<ReferenceRow>,
    pub anchors: Vec<AnchorPrediction>,
    pub ratios: Vec<RatioRow>,
    pub anchor_spread: Vec<AnchorSpread>,
    pub notes: Vec<String>,
}
impl Finite for Comparison {
    fn finite(&self) -> bool {
        self.reference.iter().all(|r| {
            r.reference.is_finite() && r.reference_error.is_finite() && r.measured.finite()
        }) && self.anchors.iter().all(|a| {
            a.scale.finite()
                && a.predictions.iter().all(|p| {
                    p.predicted.finite() && p.reference.is_finite() && p.tension_sigma.finite()
                })
        }) && self
            .ratios
            .iter()
            .all(|r| r.measured.finite() && r.reference.is_finite() && r.tension_sigma.finite())
            && self.anchor_spread.iter().all(|s| s.spread.finite())
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Quantity {
    pub name: String,
    pub symbol: String,
    pub value: Option<f64>,
    pub error: Option<f64>,
    pub unit: String,
    pub definition: String,
    pub book_label: String,
}

/// Algorithmic scales of the gas configuration and the couplings the book
/// derives from them. `inversion` maps Standard Model inputs to gas
/// parameters; its entries are inputs of a calibration, not measurements.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CouplingReport {
    pub scales: Vec<Quantity>,
    pub couplings: Vec<Quantity>,
    pub inversion: Vec<Quantity>,
    pub notes: Vec<String>,
}
impl Finite for CouplingReport {
    fn finite(&self) -> bool {
        [&self.scales, &self.couplings, &self.inversion]
            .iter()
            .all(|rows| rows.iter().all(|q| q.value.finite() && q.error.finite()))
    }
}

/// Graph smoothing diagnostic: mean neighbour colour mismatch
/// `1 − Re q_ij` over graph edges after each smoothing step, averaged over
/// the sampled frames. It defines no length scale.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FlowDiagnostic {
    pub frames: u64,
    pub steps: Vec<usize>,
    pub roughness: Vec<Option<f64>>,
}

/// Every field is required on the wire, so an object without
/// `schema_version` is no report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectroscopyReport {
    pub schema_version: u32,
    /// `CALCULATION_ORIGIN`.
    pub calculation_origin: String,
    /// Always `f64`: an `f32` run is refused, never analysed at lower precision.
    pub precision: Precision,
    pub measurement_fingerprint: String,
    pub analysis: AnalysisConfig,
    pub capabilities: Capabilities,
    pub calibration: Option<Calibration>,
    pub replicas: usize,
    /// Measured frames summed over replicas.
    pub frames: u64,
    pub channels: Vec<ChannelReport>,
    pub groups: Vec<GroupFit>,
    pub gevp: Vec<GevpReport>,
    pub comparison: Option<Comparison>,
    pub couplings: Option<CouplingReport>,
    pub flow: Option<FlowDiagnostic>,
    pub notes: Vec<String>,
}
impl SpectroscopyReport {
    /// The version by equality (`GasError::Checkpoint`), the provenance, the
    /// echoed analysis, capabilities and calibration, and a finite value in
    /// every number present. `analyze` closes with it and
    /// `presentation::present` opens with it.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SPECTROSCOPY_VERSION {
            return Err(GasError::Checkpoint("unsupported report version".into()));
        }
        require(
            self.calculation_origin == CALCULATION_ORIGIN && self.precision == Precision::F64,
            "report provenance must be an executed f64 archive",
        )?;
        self.analysis.validate()?;
        self.capabilities.validate()?;
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        require(
            self.channels.finite()
                && self.groups.finite()
                && self.gevp.finite()
                && self.comparison.finite()
                && self.couplings.finite()
                && self.flow.as_ref().is_none_or(|f| f.roughness.finite()),
            "nonfinite report value",
        )
    }
    pub fn channel(&self, id: &str) -> Option<&ChannelReport> {
        self.channels.iter().find(|c| c.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GasConfig, RecordingConfig};
    fn rate(value: f64, method: FitMethodKind) -> MassEstimate {
        MassEstimate {
            quantity: RATE_QUANTITY.into(),
            value,
            error: 0.013,
            statistical: 0.012,
            systematic: 0.005,
            method,
            time_unit: TimeUnit::Frames,
            prior_dominance: None,
        }
    }
    /// A populated report of the shape the browser receives.
    fn example() -> SpectroscopyReport {
        let spec = ChannelSpec::default();
        let mass = rate(0.21, FitMethodKind::WindowScan);
        let available = ChannelReport {
            id: "meson/scalar/standard/distance".into(),
            spec: spec.clone(),
            kind: Some(ElementKind::DistancePair),
            definition: r"\operatorname{Re}\, c_i^\dagger c_j".into(),
            book_label: "def-sm-direct-observable-law".into(),
            exchange: Some(ExchangeParity::Even),
            spatial_parity: Some(SpatialParity::Even),
            coverage: Coverage {
                frames: 7936,
                valid: 1_587_200,
                masked_self: 12,
                ..Coverage::default()
            },
            estimator: Some(EstimatorKind::FrameMean),
            correlator: Some(CorrelatorEstimate {
                lags: vec![0, 1, 2, 3],
                time_unit: TimeUnit::Frames,
                time_step: 1.,
                value: vec![Some(0.02), Some(0.0162), Some(0.0131), None],
                error: vec![Some(0.001), Some(0.0009), Some(0.0009), None],
                covariance: None,
                samples_meta: SamplesMeta {
                    resampling: ResampleKind::Jackknife,
                    effective_block: 24,
                    blocks: 328,
                    tau_int: Some(9.7),
                    covariance_rank: 3,
                    replicas: 4,
                    sampling_unit: SamplesMeta::sampling_unit(Combine::PooledBlocks, 24, 4),
                },
                connected: true,
                connected_bias: Some(-3.9e-6),
            }),
            effective_mass: Some(vec![Some([0.2107, 0.02]), Some([0.2124, 0.03]), None, None]),
            mass: Some(mass.clone()),
            fits: vec![FitOutcome {
                method: FitMethodKind::WindowScan,
                mass: Some(mass),
                diagnostics: FitDiagnostics {
                    chi2: Some(3.1),
                    dof: Some(4),
                    q: Some(0.54),
                    window: Some([2, 7]),
                    n_windows: 21,
                    correlated: true,
                    svd_cut: 1e-6,
                    covariance_rank: Some(6),
                    ..FitDiagnostics::default()
                },
                windows: vec![WindowFit {
                    t_min: 2,
                    t_max: 7,
                    value: 0.2104,
                    error: 0.012,
                    chi2: 3.1,
                    dof: 4,
                    weight: 0.31,
                    nexp: 1,
                    svd_cut: None,
                }],
                ..FitOutcome::default()
            }],
            ..ChannelReport::default()
        };
        let unavailable = ChannelReport {
            id: "meson/pseudoscalar/standard/distance".into(),
            spec: ChannelSpec::Meson {
                quantum: crate::physics::spectroscopy::config::MesonQuantum::Pseudoscalar,
                mode: crate::physics::spectroscopy::config::MesonMode::Standard,
            },
            kind: Some(ElementKind::DistancePair),
            exchange: Some(ExchangeParity::Odd),
            availability: Availability::unavailable(
                crate::physics::spectroscopy::contract::EXCHANGE_ODD_REASON,
            ),
            ..ChannelReport::default()
        };
        let recording = RecordingConfig {
            graph: true,
            ..RecordingConfig::default()
        };
        SpectroscopyReport {
            schema_version: SPECTROSCOPY_VERSION,
            calculation_origin: CALCULATION_ORIGIN.into(),
            precision: Precision::F64,
            measurement_fingerprint: "9f3c5a1be07d2264".into(),
            analysis: AnalysisConfig::default(),
            capabilities: Capabilities::of(
                &GasConfig::einstein_hilbert(0.33, 0.002).unwrap(),
                &recording,
                3,
            ),
            replicas: 4,
            frames: 7936,
            calibration: Some(Calibration {
                warmup_frames: 16,
                length: 0.41,
                length_source: "warmup_companion_median".into(),
                kappa: 0.41,
                mass: 1.,
                phase_wrapping: Some(0.),
                h_eff: 1.,
                electroweak_h_eff: 1.,
                h_s: 1.,
                epsilon_clone: 0.01,
                dt: Some(0.002),
                ..Calibration::default()
            }),
            channels: vec![available, unavailable],
            groups: vec![],
            gevp: vec![],
            comparison: Some(Comparison {
                label: HYPOTHESIS_LABEL.into(),
                reference: vec![ReferenceRow {
                    name: "f0_500".into(),
                    channel: "meson/scalar/standard/distance".into(),
                    reference: 500.,
                    reference_error: 100.,
                    unit: "MeV".into(),
                    measured: Some([0.21, 0.013]),
                    estimator: Some(EstimatorKind::FrameMean),
                }],
                ..Comparison::default()
            }),
            couplings: None,
            flow: None,
            notes: vec![crate::physics::spectroscopy::INTERPRETATION_NOTE.into()],
        }
    }
    #[test]
    fn a_populated_report_is_plain_json_and_round_trips() {
        let report = example();
        report.validate().unwrap();
        let text = serde_json::to_string_pretty(&report).unwrap();
        assert!(!text.contains("NaN") && !text.contains("inf"));
        assert_eq!(
            serde_json::from_str::<SpectroscopyReport>(&text).unwrap(),
            report
        );
        assert!(text.contains(r#""status": "unavailable""#));
        assert_eq!(
            report.channels[0].mass.as_ref().unwrap().quantity,
            RATE_QUANTITY
        );
    }
    #[test]
    fn an_unversioned_foreign_or_nonfinite_report_is_rejected_explicitly() {
        assert!(serde_json::from_str::<SpectroscopyReport>("{}").is_err());
        let mut versioned = example();
        versioned.schema_version += 1;
        assert!(matches!(versioned.validate(), Err(GasError::Checkpoint(_))));
        let mut foreign = example();
        foreign.calculation_origin = "synthetic".into();
        assert!(matches!(
            foreign.validate(),
            Err(GasError::Configuration(_))
        ));
        let mut nonfinite = example();
        nonfinite.channels[0].fits[0].windows[0].chi2 = f64::NAN;
        assert!(matches!(
            nonfinite.validate(),
            Err(GasError::Configuration(_))
        ));
        let mut undefined = example();
        undefined.channels[0].correlator.as_mut().unwrap().value[3] = Some(f64::INFINITY);
        assert!(undefined.validate().is_err());
        let mut planar = example();
        planar.capabilities.dimension = 0;
        assert!(planar.validate().is_err());
    }
    #[test]
    fn sampling_units_name_blocks_or_runs_and_coverage_merges_by_addition() {
        assert_eq!(
            SamplesMeta::sampling_unit(Combine::PooledBlocks, 24, 4),
            "time blocks of 24 origins pooled over 4 independently seeded runs"
        );
        assert_eq!(
            SamplesMeta::sampling_unit(Combine::RunsAsSamples, 24, 8),
            "independently seeded runs"
        );
        let mut a = Coverage {
            frames: 2,
            valid: 10,
            masked_self: 1,
            ..Coverage::default()
        };
        a.merge(&Coverage {
            frames: 1,
            valid: 4,
            masked_color: 3,
            ..Coverage::default()
        });
        assert_eq!((a.frames, a.valid, a.masked()), (3, 14, 4));
    }
}
