//! Blocked lagged moments by direct lag sums. The pooled moments are the
//! central estimate; blocks exist so that resampling can recompute every
//! estimate, including its disconnected part, from the same sums.
use super::{SeriesView, Subtraction};
use crate::{
    GasError, Result,
    error::require,
    memory::{checked_add, checked_mul},
};
use serde::{Deserialize, Serialize};

/// Sums over time origins `t` of one block, per lag `τ`, of a pair of series
/// `a`, `b` with `components` contracted components:
/// `ab = Σ w a_t·b_{t+τ}`, `a = Σ w a_t`, `b = Σ w b_{t+τ}`, `n = Σ w`, with
/// `w` the pair weight. The correlator is `Σab/Σn` minus the disconnected part
/// a `Subtraction` names, the dot running over components. The pair
/// `(t, t+τ)` belongs to the block of its origin `t`; a zero-weight frame is
/// still an origin, and a trailing incomplete block stays a block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockMoments {
    /// Number of lags, `max_lag + 1`.
    pub lags: usize,
    pub components: usize,
    /// Time origins per closed block. Doubles on every `merge_pairs`.
    pub origins_per_block: usize,
    /// Pair-merge when this many blocks are held. Even, at least 4.
    pub max_blocks: usize,
    /// Blocks held, the last of which may be open.
    pub blocks: usize,
    /// Origins pushed into the last block; `origins_per_block` when it is closed.
    pub open: usize,
    /// `[blocks, lags]`.
    pub ab: Vec<f64>,
    /// `[blocks, lags, components]`.
    pub a: Vec<f64>,
    /// `[blocks, lags, components]`.
    pub b: Vec<f64>,
    /// `[blocks, lags]`.
    pub n: Vec<f64>,
}
impl BlockMoments {
    pub fn new(
        lags: usize,
        components: usize,
        origins_per_block: usize,
        max_blocks: usize,
    ) -> Result<Self> {
        let _ = (lags, components, origins_per_block, max_blocks);
        Err(GasError::Capability("pending: numerics::moments".into()))
    }
    pub fn validate(&self) -> Result<()> {
        let entries = checked_mul(self.blocks, self.lags)?;
        require(
            self.lags > 0
                && self.components > 0
                && self.origins_per_block > 0
                && self.max_blocks >= 4
                && self.max_blocks.is_multiple_of(2)
                && self.blocks <= self.max_blocks
                && self.open <= self.origins_per_block
                && self.ab.len() == entries
                && self.n.len() == entries
                && self.a.len() == checked_mul(entries, self.components)?
                && self.b.len() == self.a.len(),
            "block moments shape",
        )?;
        require(
            self.ab
                .iter()
                .chain(&self.a)
                .chain(&self.b)
                .all(|x| x.is_finite())
                && self.n.iter().all(|n| n.is_finite() && *n >= 0.),
            "nonfinite block moments",
        )
    }
    /// Add one time origin: `ab`, `n` are `[lags]`, `a`, `b` are
    /// `[lags, components]`. Lags without a partner carry `n = 0`. Opens a new
    /// block when the last one is full, pair-merging first at `max_blocks`.
    pub fn push_origin(&mut self, ab: &[f64], a: &[f64], b: &[f64], n: &[f64]) -> Result<()> {
        let _ = (ab, a, b, n);
        Err(GasError::Capability("pending: numerics::moments".into()))
    }
    /// Close the open block early (a gap in the series). The next origin opens
    /// a new block.
    pub fn close_block(&mut self) {}
    /// Merge blocks `(0,1), (2,3), …` and double `origins_per_block`; an odd
    /// trailing block stays alone and open. Equals `coarsen(2)` on closed blocks.
    pub fn merge_pairs(&mut self) {}
    /// Moments over blocks of `k` consecutive blocks.
    pub fn coarsen(&self, k: usize) -> Result<Self> {
        let _ = k;
        Err(GasError::Capability("pending: numerics::moments".into()))
    }
    /// Blocks of independent runs, in order. All parts must agree on `lags`,
    /// `components` and `origins_per_block`.
    pub fn concat(parts: &[&Self]) -> Result<Self> {
        let _ = parts;
        Err(GasError::Capability("pending: numerics::moments".into()))
    }
    /// Sums over all blocks except `skip`.
    pub fn pooled(&self, skip: Option<usize>) -> LagMoments {
        let _ = skip;
        LagMoments::default()
    }
    /// Correlator per lag from the pooled sums; `None` where `Σn = 0`.
    pub fn estimate(&self, subtraction: Subtraction) -> Vec<Option<f64>> {
        let _ = subtraction;
        vec![None; self.lags]
    }
    /// Conservative retained size, in checked arithmetic.
    pub fn buffer_bytes(&self) -> Result<usize> {
        let entries = [&self.ab, &self.a, &self.b, &self.n]
            .iter()
            .try_fold(0, |sum, v| checked_add(sum, v.len()))?;
        checked_add(checked_mul(entries, 8)?, 96)
    }
}

/// Sums of one block, or pooled over blocks: `ab`, `n` are `[lags]`, `a`, `b`
/// are `[lags, components]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LagMoments {
    pub ab: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub n: Vec<f64>,
}

/// Autocorrelation moments of a series, contracted over components, with one
/// block per `block` consecutive origins inside each segment. The pair weight
/// is `w_t w_{t+τ}`; pairs that span two segments are dropped.
pub fn series_moments(
    series: SeriesView<'_>,
    max_lag: usize,
    block: usize,
) -> Result<BlockMoments> {
    cross_moments(series, series, max_lag, block)
}

/// Cross moments `⟨a_t · b_{t+τ}⟩` of two series over the same frames.
pub fn cross_moments(
    a: SeriesView<'_>,
    b: SeriesView<'_>,
    max_lag: usize,
    block: usize,
) -> Result<BlockMoments> {
    let _ = (a, b, max_lag, block);
    Err(GasError::Capability("pending: numerics::moments".into()))
}
