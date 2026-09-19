//! Serialisable product of a streamed measurement: retained operator series,
//! propagator moments and coverage. Correlators are formed at analysis time,
//! so the resampling block stays an analysis parameter. State an accumulator
//! needs beyond these records lives in `AccumulatorState`.
use super::{
    config::{ChannelSpec, MeasurementConfig},
    contract::{
        Availability, Capabilities, ElementKind, ExchangeParity, SPECTROSCOPY_VERSION,
        SpatialParity,
    },
    report::{Calibration, Coverage, FlowDiagnostic},
};
use crate::{
    GasConfig, GasError, Result,
    error::require,
    memory::checked_mul,
    physics::numerics::{BlockMoments, SeriesView},
};
use serde::{Deserialize, Serialize};

/// Slab sums of single frames along the Euclidean-time coordinate. With
/// `S_b(t)` the normalised operator sum over the elements of slab `b`, the
/// connected estimate is
/// `C(Δ) = (1/n_Δ) Σ_b [⟨S_b S_{b+Δ}⟩ − ⟨S_b⟩⟨S_{b+Δ}⟩]` over the `n_Δ` bin
/// pairs with data, where `⟨S_b S_b'⟩ = pair_ab/pair_n` and
/// `⟨S_b⟩ = profile/profile_n` come from the blocks kept by a resample. The
/// per-bin means matter: on a confined coordinate `⟨S_b⟩` varies with `b`, and
/// a subtraction by separation alone leaves `Cov_b(⟨S_b⟩, ⟨S_{b+Δ}⟩)`, which
/// decays on the cloud width and is no fluctuation. A periodic axis wraps `Δ`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlabMoments {
    pub bins: usize,
    pub components: usize,
    /// Frames per closed block and blocks held, merged in pairs like `BlockMoments`.
    pub origins_per_block: usize,
    pub blocks: usize,
    /// `[blocks, bins, bins]`, entries `b ≤ b'` only: `Σ_t S_b(t)·S_b'(t)`
    /// over frames where both slabs hold a valid element.
    pub pair_ab: Vec<f64>,
    /// `[blocks, bins, bins]`: the number of such frames.
    pub pair_n: Vec<f64>,
    /// `[blocks, bins, components]`: `Σ_t S_b(t)`.
    pub profile: Vec<f64>,
    /// `[blocks, bins]`: frames in which slab `b` held a valid element.
    pub profile_n: Vec<f64>,
}
impl SlabMoments {
    pub fn validate(&self) -> Result<()> {
        let rows = checked_mul(self.blocks, self.bins)?;
        require(
            self.bins >= 2
                && self.components > 0
                && self.origins_per_block > 0
                && self.pair_ab.len() == checked_mul(rows, self.bins)?
                && self.pair_n.len() == self.pair_ab.len()
                && self.profile.len() == checked_mul(rows, self.components)?
                && self.profile_n.len() == rows,
            "slab moments shape",
        )?;
        require(
            [&self.pair_ab, &self.pair_n, &self.profile, &self.profile_n]
                .iter()
                .all(|v| v.iter().all(|x| x.is_finite())),
            "nonfinite slab moments",
        )
    }
}

/// Everything measured for one channel (one operator on one element kind,
/// optionally at one geodesic scale).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelSeries {
    /// `contract::channel_id`, with `@<scale index>` appended for a multiscale copy.
    pub id: String,
    /// `ChannelSpec::Custom` for an injected operator.
    pub spec: ChannelSpec,
    pub kind: ElementKind,
    pub scale: Option<f64>,
    /// `definition`, `book_label`, `spatial_parity` and `note` of the
    /// operator's `Descriptor`.
    pub definition: String,
    pub book_label: String,
    pub note: String,
    pub exchange: ExchangeParity,
    pub spatial_parity: Option<SpatialParity>,
    pub correlatable: bool,
    pub components: usize,
    pub availability: Availability,
    pub coverage: Coverage,
    /// How many of the `coverage.frames` frames with a valid element had an
    /// involutive source topology. A diagnostic; whether an exchange-odd
    /// frame mean exists is decided by `weight`.
    pub involutive_frames: u64,
    /// Frame averages `A_t(O)`, `[frames, components]`, aligned with
    /// `Measurement::steps`. Entries of zero-weight frames are 0 and ignored.
    ///
    /// An `ExchangeParity::Odd` operator is averaged over the valid elements
    /// whose mirror `(j, i)` is not a valid element of the same topology:
    /// mirrored pairs cancel exactly, so they are left out of the sum and of
    /// the denominator instead of contributing roundoff.
    pub values: Vec<f64>,
    /// `[frames]`. `ValidCount`: 1 where `Σ w_I m_I > 0`, else 0. `FixedN`
    /// (configured, or `LocalOperator::normalization`): 1 with value
    /// `(1/N) Σ w_I m_I O_I`, possibly exactly 0, whenever the records of the
    /// frame exist, and 0 only when a record itself is missing (no fitness,
    /// no preceding kick). An `Odd` operator has weight 0 on every frame
    /// without an unmirrored valid element, under either normalisation.
    pub weight: Vec<f64>,
    /// Source-frozen propagator moments over all valid elements, when enabled
    /// for the channel. `lags` holds every `lag_stride`-th lag.
    pub propagator: Option<BlockMoments>,
    /// Euclidean-time slab moments, when a Euclidean axis is measured.
    pub euclidean: Option<SlabMoments>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    pub schema_version: u32,
    /// `MeasurementConfig::fingerprint`; measurements combine only when equal.
    pub fingerprint: String,
    pub config: MeasurementConfig,
    pub gas: GasConfig,
    pub capabilities: Capabilities,
    /// Population size of the first ingested step; 0 before it. Measurements
    /// of different sizes do not combine.
    pub walkers: usize,
    /// `None` until the warm-up frames are consumed.
    pub calibration: Option<Calibration>,
    /// Engine step of every measured frame, ascending.
    pub steps: Vec<u64>,
    /// `[frames]`: contiguous runs of frames; lags never span two segments.
    pub segment: Vec<u32>,
    pub segments: u32,
    /// Recorded steps seen, including warm-up and strided-over steps.
    pub ingested: u64,
    pub channels: Vec<ChannelSeries>,
    pub flow: Option<FlowDiagnostic>,
    pub notes: Vec<String>,
}
impl Measurement {
    pub fn frames(&self) -> usize {
        self.steps.len()
    }
    pub fn channel(&self, id: &str) -> Option<&ChannelSeries> {
        self.channels.iter().find(|c| c.id == id)
    }
    /// The retained frame-average series of a channel.
    pub fn series<'a>(&'a self, channel: &'a ChannelSeries) -> SeriesView<'a> {
        SeriesView {
            values: &channel.values,
            weight: &channel.weight,
            segment: &self.segment,
            components: channel.components,
        }
    }
    /// Schema version, configuration, capabilities and calibration, shapes,
    /// alignment of every series with `steps`, finite weighted values and
    /// every `BlockMoments::validate`.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SPECTROSCOPY_VERSION {
            return Err(GasError::Checkpoint(
                "unsupported measurement version".into(),
            ));
        }
        self.config.validate()?;
        self.capabilities.validate()?;
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        let frames = self.steps.len();
        require(
            self.segment.len() == frames
                && self.steps.windows(2).all(|w| w[0] < w[1])
                && self.segment.windows(2).all(|w| w[0] <= w[1])
                && self.segment.last().is_none_or(|s| *s < self.segments)
                && self.ingested >= frames as u64,
            "measurement frame alignment",
        )?;
        for channel in &self.channels {
            // An unavailable channel retains no series.
            require(
                channel.components > 0 && [0, frames].contains(&channel.weight.len()),
                "channel series alignment",
            )?;
            if channel.weight.len() == frames {
                self.series(channel).validate()?;
            } else {
                require(channel.values.is_empty(), "channel series alignment")?;
            }
            if let Some(moments) = &channel.propagator {
                moments.validate()?;
                require(
                    moments.components == channel.components,
                    "propagator components",
                )?;
            }
            if let Some(slabs) = &channel.euclidean {
                slabs.validate()?;
                require(slabs.components == channel.components, "slab components")?;
            }
        }
        Ok(())
    }
    /// Conservative retained size, through `crate::memory` checked arithmetic.
    pub fn buffer_bytes(&self) -> Result<usize> {
        Err(GasError::Capability(
            "pending: spectroscopy::measurement".into(),
        ))
    }
    /// CBOR, validated on both sides; decode is capped at 256 MiB and rejects
    /// trailing data. Codec failures are `GasError::Checkpoint`.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Err(GasError::Capability(
            "pending: spectroscopy::measurement".into(),
        ))
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let _ = bytes;
        Err(GasError::Capability(
            "pending: spectroscopy::measurement".into(),
        ))
    }
}
