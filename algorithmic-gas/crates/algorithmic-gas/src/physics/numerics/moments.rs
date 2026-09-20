//! Blocked lagged moments by direct lag sums. The pooled moments are the
//! central estimate; blocks exist so that resampling can recompute every
//! estimate, including its disconnected part, from the same sums.
use super::{SeriesView, Subtraction};
use crate::{
    GasError, Result,
    error::require,
    memory::{DEFAULT_MEMORY_BYTES, checked_add, checked_mul, enforce},
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
        require(
            (1..=4097).contains(&lags) && (1..=4096).contains(&components),
            "block moments need 1..=4097 lags and 1..=4096 components",
        )?;
        require(
            origins_per_block >= 1 && max_blocks >= 4 && max_blocks.is_multiple_of(2),
            "block moments need a block of at least one origin and an even max_blocks >= 4",
        )?;
        Ok(Self {
            lags,
            components,
            origins_per_block,
            max_blocks,
            blocks: 0,
            open: 0,
            ab: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            n: Vec::new(),
        })
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
        let (l, c) = (self.lags, self.components);
        require(
            ab.len() == l
                && n.len() == l
                && a.len() == l * c
                && b.len() == a.len()
                && ab.iter().chain(a).chain(b).all(|x| x.is_finite())
                && n.iter().all(|n| n.is_finite() && *n >= 0.),
            "origin sums shape, negative weight or nonfinite entry",
        )?;
        if self.open == self.origins_per_block && self.blocks == self.max_blocks {
            self.merge_pairs();
        }
        if self.blocks == 0 || self.open == self.origins_per_block {
            self.blocks += 1;
            self.open = 0;
            self.ab.resize(self.blocks * l, 0.);
            self.n.resize(self.blocks * l, 0.);
            self.a.resize(self.blocks * l * c, 0.);
            self.b.resize(self.blocks * l * c, 0.);
        }
        let k = self.blocks - 1;
        for (sum, x) in self.ab[k * l..].iter_mut().zip(ab) {
            *sum += x;
        }
        for (sum, x) in self.n[k * l..].iter_mut().zip(n) {
            *sum += x;
        }
        for (sum, x) in self.a[k * l * c..].iter_mut().zip(a) {
            *sum += x;
        }
        for (sum, x) in self.b[k * l * c..].iter_mut().zip(b) {
            *sum += x;
        }
        self.open += 1;
        Ok(())
    }
    /// Close the open block early (a gap in the series). The next origin opens
    /// a new block.
    pub fn close_block(&mut self) {
        if self.blocks > 0 {
            self.open = self.origins_per_block;
        }
    }
    /// Merge blocks `(0,1), (2,3), …` and double `origins_per_block`; an odd
    /// trailing block stays alone and open. Equals `coarsen(2)` on closed blocks.
    pub fn merge_pairs(&mut self) {
        *self = self.grouped(2, self.origins_per_block.saturating_mul(2));
    }
    /// Moments over blocks of `k` consecutive blocks.
    pub fn coarsen(&self, k: usize) -> Result<Self> {
        require(k >= 1, "coarsening needs a group of at least one block")?;
        Ok(self.grouped(k, checked_mul(self.origins_per_block, k)?))
    }
    /// Sums of `k` consecutive blocks; the last group may hold fewer and keeps
    /// the origin count of an open last block.
    fn grouped(&self, k: usize, origins_per_block: usize) -> Self {
        let (l, c) = (self.lags, self.components);
        let blocks = self.blocks.div_ceil(k);
        let mut out = Self {
            origins_per_block,
            blocks,
            open: match self.blocks {
                0 => 0,
                held => ((held - 1) % k)
                    .saturating_mul(self.origins_per_block)
                    .saturating_add(self.open),
            },
            ab: vec![0.; blocks * l],
            a: vec![0.; blocks * l * c],
            b: vec![0.; blocks * l * c],
            n: vec![0.; blocks * l],
            ..*self
        };
        for block in 0..self.blocks {
            let (from, to) = (block * l, block / k * l);
            for lag in 0..l {
                out.ab[to + lag] += self.ab[from + lag];
                out.n[to + lag] += self.n[from + lag];
            }
            for i in 0..l * c {
                out.a[to * c + i] += self.a[from * c + i];
                out.b[to * c + i] += self.b[from * c + i];
            }
        }
        out
    }
    /// Running sums: block `k` holds the sum of blocks `0..=k`, or of blocks
    /// `k..` when `reverse`. A delete-one sum is then one addition of
    /// nonnegative weights, with no cancellation.
    pub(super) fn running(&self, reverse: bool) -> Self {
        let (l, c) = (self.lags, self.components);
        let mut out = self.clone();
        for step in 1..self.blocks {
            let (from, to) = if reverse {
                (self.blocks - step, self.blocks - step - 1)
            } else {
                (step - 1, step)
            };
            for lag in 0..l {
                out.ab[to * l + lag] += out.ab[from * l + lag];
                out.n[to * l + lag] += out.n[from * l + lag];
            }
            for i in 0..l * c {
                out.a[to * l * c + i] += out.a[from * l * c + i];
                out.b[to * l * c + i] += out.b[from * l * c + i];
            }
        }
        out
    }
    /// Blocks of independent runs, in order. All parts must agree on `lags`,
    /// `components` and `origins_per_block`.
    pub fn concat(parts: &[&Self]) -> Result<Self> {
        let first = parts
            .first()
            .ok_or_else(|| GasError::Configuration("no block moments to join".into()))?;
        require(
            parts.iter().all(|p| {
                p.lags == first.lags
                    && p.components == first.components
                    && p.origins_per_block == first.origins_per_block
            }),
            "joined block moments must share lags, components and origins_per_block",
        )?;
        let mut out = Self {
            blocks: 0,
            open: 0,
            ab: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
            n: Vec::new(),
            ..**first
        };
        for part in parts.iter().filter(|p| p.blocks > 0) {
            part.validate()?;
            out.blocks = checked_add(out.blocks, part.blocks)?;
            out.open = part.open;
            out.max_blocks = out.max_blocks.max(part.max_blocks);
            out.ab.extend_from_slice(&part.ab);
            out.a.extend_from_slice(&part.a);
            out.b.extend_from_slice(&part.b);
            out.n.extend_from_slice(&part.n);
        }
        out.max_blocks = out.max_blocks.max(checked_add(out.blocks, out.blocks % 2)?);
        Ok(out)
    }
    /// Sums over all blocks except `skip`.
    pub fn pooled(&self, skip: Option<usize>) -> LagMoments {
        let mut sums = LagMoments::zeros(self.lags, self.components);
        for block in (0..self.blocks).filter(|&k| Some(k) != skip) {
            sums.add(self, block, 1.);
        }
        sums
    }
    /// Sums over blocks with block `k` counted `multiplicity[k]` times;
    /// `multiplicity` is `[blocks]`.
    pub fn weighted(&self, multiplicity: &[f64]) -> LagMoments {
        let mut sums = LagMoments::zeros(self.lags, self.components);
        for (block, &m) in multiplicity.iter().enumerate().take(self.blocks) {
            if m != 0. {
                sums.add(self, block, m);
            }
        }
        sums
    }
    /// Time origins held, counting a block closed early as full.
    pub fn origins(&self) -> usize {
        match self.blocks {
            0 => 0,
            held => (held - 1)
                .saturating_mul(self.origins_per_block)
                .saturating_add(self.open),
        }
    }
    /// Correlator per lag from the pooled sums; `None` where `Σn = 0`.
    pub fn estimate(&self, subtraction: Subtraction) -> Vec<Option<f64>> {
        self.pooled(None).estimate(subtraction)
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
impl LagMoments {
    pub(super) fn zeros(lags: usize, components: usize) -> Self {
        Self {
            ab: vec![0.; lags],
            a: vec![0.; lags * components],
            b: vec![0.; lags * components],
            n: vec![0.; lags],
        }
    }
    /// Add `scale` times block `block` of `moments`, which has this shape.
    pub(super) fn add(&mut self, moments: &BlockMoments, block: usize, scale: f64) {
        let (l, c) = (moments.lags, moments.components);
        for lag in 0..l {
            self.ab[lag] += scale * moments.ab[block * l + lag];
            self.n[lag] += scale * moments.n[block * l + lag];
        }
        for i in 0..l * c {
            self.a[i] += scale * moments.a[block * l * c + i];
            self.b[i] += scale * moments.b[block * l * c + i];
        }
    }
    /// Correlator per lag; `None` where `n = 0`, and at every lag of
    /// `Subtraction::GlobalMean` when `n = 0` at lag 0. The global means are
    /// `Σa/Σn` of the first leg and `Σb/Σn` of the second at lag 0; they
    /// coincide for an autocorrelation.
    pub fn estimate(&self, subtraction: Subtraction) -> Vec<Option<f64>> {
        let lags = self.n.len();
        let c = self.a.len() / lags.max(1);
        let n0 = self.n.first().copied().unwrap_or(0.);
        (0..lags)
            .map(|lag| {
                let n = self.n[lag];
                if n <= 0. {
                    return None;
                }
                let (a, b) = (
                    &self.a[lag * c..(lag + 1) * c],
                    &self.b[lag * c..(lag + 1) * c],
                );
                let disconnected = match subtraction {
                    Subtraction::None => 0.,
                    Subtraction::LagMeans => a.iter().zip(b).map(|(a, b)| (a / n) * (b / n)).sum(),
                    Subtraction::GlobalMean if n0 > 0. => (0..c)
                        .map(|k| {
                            let (ma, mb) = (self.a[k] / n0, self.b[k] / n0);
                            ma * (b[k] / n) + mb * (a[k] / n) - ma * mb
                        })
                        .sum(),
                    Subtraction::GlobalMean => return None,
                };
                Some(self.ab[lag] / n - disconnected).filter(|x: &f64| x.is_finite())
            })
            .collect()
    }
}

/// Pairs `(t, t+τ)` of `origins` consecutive time origins, over the lags
/// `τ = 0..lags`, whose two frames fall in different blocks of `block`
/// origins. A pair belongs to the block of its origin, so a delete-one
/// jackknife that drops one block keeps counting these pairs through their
/// origin while their sink sits inside the deleted block: the deletion removes
/// origins, not observations, and beyond `τ = block` it removes none of the
/// products a lag holds. Zero weights and segment breaks only drop pairs, so
/// the count bounds a masked series from above. A lag of `origins` or more
/// reaches no sink at all, so the sum runs over the shorter of the two and
/// costs `O(min(lags, origins))`.
pub fn straddling_pairs(lags: usize, block: usize, origins: usize) -> u64 {
    let block = block.max(1);
    (1..lags.min(origins))
        .map(|lag| {
            // A pair straddles when its origin sits in the last min(τ, block)
            // positions of its block; the last lag of the series has no sink.
            let last = lag.min(block);
            let reach = origins.saturating_sub(lag);
            let whole = (reach / block * last) as u64;
            whole + (reach % block).saturating_sub(block - last) as u64
        })
        .sum()
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

/// Cross moments `⟨a_t · b_{t+τ}⟩` of two series over the same frames and
/// segments, with pair weight `w^a_t w^b_{t+τ}`.
pub fn cross_moments(
    a: SeriesView<'_>,
    b: SeriesView<'_>,
    max_lag: usize,
    block: usize,
) -> Result<BlockMoments> {
    a.validate()?;
    b.validate()?;
    require(
        a.components == b.components && a.segment == b.segment && max_lag <= 4096 && block >= 1,
        "lagged moments need series of equal shape and segments, max_lag <= 4096 and block >= 1",
    )?;
    let (frames, c, l) = (a.frames(), a.components, max_lag + 1);
    let mut blocks = 0usize;
    let mut held = block;
    for t in 0..frames {
        if held == block || a.segment[t] != a.segment[t - 1] {
            blocks += 1;
            held = 0;
        }
        held += 1;
    }
    enforce(
        checked_mul(
            checked_mul(blocks, l)?,
            checked_mul(checked_add(c, 1)?, 16)?,
        )?,
        DEFAULT_MEMORY_BYTES,
    )?;
    let mut moments = BlockMoments::new(l, c, block, (blocks + blocks % 2).max(4))?;
    let (mut ab, mut n) = (vec![0.; l], vec![0.; l]);
    let (mut sa, mut sb) = (vec![0.; l * c], vec![0.; l * c]);
    for t in 0..frames {
        if t > 0 && a.segment[t] != a.segment[t - 1] {
            moments.close_block();
        }
        ab.fill(0.);
        n.fill(0.);
        sa.fill(0.);
        sb.fill(0.);
        for lag in 0..l.min(frames - t) {
            let s = t + lag;
            let w = a.weight[t] * b.weight[s];
            if a.segment[s] != a.segment[t] {
                break;
            }
            if w == 0. {
                continue;
            }
            let (x, y) = (&a.values[t * c..(t + 1) * c], &b.values[s * c..(s + 1) * c]);
            ab[lag] = w * x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>();
            n[lag] = w;
            for k in 0..c {
                sa[lag * c + k] = w * x[k];
                sb[lag * c + k] = w * y[k];
            }
        }
        moments.push_origin(&ab, &sa, &sb, &n)?;
    }
    moments.validate()?;
    Ok(moments)
}
