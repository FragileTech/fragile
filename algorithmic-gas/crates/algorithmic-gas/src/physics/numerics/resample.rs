//! Block jackknife and block bootstrap over `BlockMoments`, the matching
//! covariance, and the integrated autocorrelation time.
use super::{
    BlockMoments, BlockSize, LagMoments, ResampleKind, Resampling, Samples, SeriesView, Subtraction,
};
use crate::{
    GasError, Result,
    error::require,
    memory::{DEFAULT_MEMORY_BYTES, checked_mul, enforce},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};

/// Sokal/Madras self-consistent window estimate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TauInt {
    /// `1/2 + Σ_{τ=1}^{W} ρ(τ)`, at least 1/2.
    pub tau: f64,
    pub window: usize,
}

/// Integrated autocorrelation time of a weighted series, from its
/// component-contracted, lag-wise connected autocorrelation. Lags are summed
/// on demand until the window closes, at most `frames / 2` of them; a window
/// that never closes leaves a lower bound with `window` at the last lag
/// reached. `None` when the series is too short or has no variance.
pub fn tau_int(series: SeriesView<'_>) -> Option<TauInt> {
    series.validate().ok()?;
    let frames = series.frames();
    // Index of the contiguous run of equal segment ids a frame belongs to.
    let mut run = vec![0usize; frames];
    for t in 1..frames {
        run[t] = run[t - 1] + usize::from(series.segment[t] != series.segment[t - 1]);
    }
    let variance = lagged(series, &run, 0).filter(|c| *c > 0.)?;
    (series.weight.iter().filter(|w| **w > 0.).count() >= 4).then(|| {
        window(
            |lag| lagged(series, &run, lag).map(|c| c / variance),
            frames / 2,
        )
    })
}

/// Lag-wise connected autocorrelation of `series` at one lag; `run` is
/// `[frames]` and a pair needs both frames in the same run.
fn lagged(series: SeriesView<'_>, run: &[usize], lag: usize) -> Option<f64> {
    let c = series.components;
    let mut sums = LagMoments::zeros(1, c);
    for t in 0..series.frames().saturating_sub(lag) {
        let s = t + lag;
        let w = series.weight[t] * series.weight[s];
        if w == 0. || run[t] != run[s] {
            continue;
        }
        let (x, y) = (
            &series.values[t * c..(t + 1) * c],
            &series.values[s * c..(s + 1) * c],
        );
        sums.ab[0] += w * x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>();
        sums.n[0] += w;
        for k in 0..c {
            sums.a[k] += w * x[k];
            sums.b[k] += w * y[k];
        }
    }
    sums.estimate(Subtraction::LagMeans)[0]
}

/// Self-consistent window: the smallest `W ≥ 1` with `W ≥ 5 max(τ(W), 1/2)`,
/// `τ(W) = 1/2 + Σ_{τ=1}^{W} ρ(τ)`, searched up to `limit` and up to the
/// first undefined lag.
fn window(rho: impl Fn(usize) -> Option<f64>, limit: usize) -> TauInt {
    let mut estimate = TauInt {
        tau: 0.5,
        window: 0,
    };
    let mut tau = 0.5;
    for lag in 1..=limit {
        let Some(r) = rho(lag) else { break };
        tau += r;
        estimate = TauInt {
            tau: tau.max(0.5),
            window: lag,
        };
        if lag as f64 >= 5. * estimate.tau {
            break;
        }
    }
    estimate
}

/// The same estimate from a normalized autocorrelation `ρ(τ) = C(τ)/C(0)` over
/// `frames` measurements.
pub fn tau_int_of_correlator(correlator: &[Option<f64>], frames: usize) -> Option<TauInt> {
    let variance = correlator.first().copied().flatten().filter(|c| *c > 0.)?;
    (frames >= 4).then(|| {
        window(
            |lag| correlator[lag].map(|c| c / variance),
            (correlator.len() - 1).min(frames / 2),
        )
    })
}

/// Origins per block for `BlockSize::Auto`: the smallest multiple of `base`
/// (origins already merged into one stored block) that is at least
/// `max(2 τ, (2 τ)^{2/3} origins^{1/3})`, the batch size of least mean squared
/// error up to a constant, capped so that at least 8 blocks remain when the
/// data allow it. For AR(1) with `ρ = 0.95` a block of `2 τ_int` alone biases
/// the variance by −43 %. The bound is decided by `b ≥ 2 τ` and
/// `b³ ≥ (2 τ)² origins`, so the block does not depend on the rounding of a
/// platform's `powf`.
pub fn auto_block(tau: f64, base: usize, origins: usize) -> usize {
    let base = base.max(1);
    let twice = twice_tau(tau);
    let most = (origins / base.saturating_mul(8)).max(1);
    let target = twice.max(twice.powf(2. / 3.) * (origins as f64).powf(1. / 3.));
    let mut multiple = ((target / base as f64).ceil() as usize).clamp(1, most);
    while multiple > 1 && reaches(base * (multiple - 1), twice, origins) {
        multiple -= 1;
    }
    while multiple < most && !reaches(base * multiple, twice, origins) {
        multiple += 1;
    }
    base * multiple
}

/// True when a block of `block` time origins is short of the batch size of
/// least mean squared error for an autocorrelation time `tau` over `origins`
/// origins. `auto_block` reaches it whenever the cap that keeps eight blocks
/// leaves room; when it does not, the returned block is the largest the data
/// support and nothing else says so, and a resampled variance over blocks
/// below the rule is biased low.
pub fn block_below_target(tau: f64, block: usize, origins: usize) -> bool {
    !reaches(block, twice_tau(tau), origins)
}

/// Twice the autocorrelation time an automatic block targets, at least one.
fn twice_tau(tau: f64) -> f64 {
    2. * if tau.is_finite() { tau.max(0.5) } else { 0.5 }
}

/// Whether `block` meets both halves of the batch-size rule, `b ≥ 2 τ` and
/// `b³ ≥ (2 τ)² origins`.
fn reaches(block: usize, twice: f64, origins: usize) -> bool {
    let block = block as f64;
    block >= twice && block * block * block >= twice * twice * origins as f64
}

/// Origins per resampling block: a multiple of `base`.
fn block_origins(
    base: usize,
    origins: usize,
    resampling: &Resampling,
    tau: Option<f64>,
) -> Result<usize> {
    require(base >= 1, "stored blocks must hold at least one origin")?;
    match resampling.block() {
        BlockSize::Fixed { frames } => checked_mul(base, frames.div_ceil(base)),
        BlockSize::Auto => {
            let tau = tau.filter(|t| t.is_finite() && *t >= 0.5).ok_or_else(|| {
                GasError::Configuration(
                    "automatic block needs an autocorrelation time of at least 1/2".into(),
                )
            })?;
            Ok(auto_block(tau, base, origins))
        }
    }
}

/// Resamples of any statistic of per-block sums. `statistic(multiplicity)`
/// receives one multiplicity per STORED block (`[blocks]`) and returns the
/// estimate from the sums weighted by it; all ones is the central value. The
/// stored blocks, of `base` origins each, are grouped into resampling blocks
/// of `auto_block(tau, base, origins)` or the fixed size (rounded up to a
/// multiple of `base`). The jackknife zeroes one group per resample; resample
/// `r` of the bootstrap draws its groups with replacement from the addressed
/// stream `Resampling::Bootstrap` documents. `tau` is required by
/// `BlockSize::Auto`. Fewer than two groups is `GasError::Numerical`.
pub fn resample_blocks(
    blocks: usize,
    base: usize,
    origins: usize,
    resampling: &Resampling,
    tau: Option<f64>,
    statistic: &dyn Fn(&[f64]) -> Vec<Option<f64>>,
) -> Result<Samples> {
    resampling.validate()?;
    let block = block_origins(base, origins, resampling, tau)?;
    let group = block / base;
    let groups = blocks.div_ceil(group);
    if groups < 2 {
        return Err(GasError::Numerical(
            "resampling needs at least two blocks".into(),
        ));
    }
    let (kind, count) = match resampling {
        Resampling::Bootstrap { samples, .. } => (ResampleKind::Bootstrap, *samples),
        Resampling::BlockJackknife { .. } | Resampling::Uncorrelated => {
            (ResampleKind::Jackknife, groups)
        }
    };
    let mut multiplicity = vec![1.; blocks];
    let center = statistic(&multiplicity);
    let dimension = center.len();
    let mut defined: Vec<bool> = center
        .iter()
        .map(|x| x.is_some_and(f64::is_finite))
        .collect();
    let entries = checked_mul(count, dimension)?;
    enforce(checked_mul(entries, 8)?, DEFAULT_MEMORY_BYTES)?;
    let mut values = Vec::with_capacity(entries);
    let mut draws = vec![0.; groups];
    for r in 0..count {
        match resampling {
            Resampling::Bootstrap { seed, .. } => {
                let mut stream = RandomStream::new(*seed, 0, Stream::Initialize, r as u64, 811);
                draws.fill(0.);
                for _ in 0..groups {
                    draws[stream.index(groups)] += 1.;
                }
                for (stored, m) in multiplicity.iter_mut().enumerate() {
                    *m = draws[stored / group];
                }
            }
            Resampling::BlockJackknife { .. } | Resampling::Uncorrelated => {
                for (stored, m) in multiplicity.iter_mut().enumerate() {
                    *m = if stored / group == r { 0. } else { 1. };
                }
            }
        }
        let sample = statistic(&multiplicity);
        if sample.len() != dimension {
            return Err(GasError::Shape(
                "resampled statistic changed its length".into(),
            ));
        }
        for (i, x) in sample.iter().enumerate() {
            defined[i] &= x.is_some_and(f64::is_finite);
            values.push(x.unwrap_or(0.));
        }
    }
    for row in values.chunks_exact_mut(dimension.max(1)) {
        for (x, keep) in row.iter_mut().zip(&defined) {
            if !keep {
                *x = 0.;
            }
        }
    }
    Ok(Samples {
        kind,
        dimension,
        count,
        central: center
            .iter()
            .zip(&defined)
            .map(|(x, keep)| x.filter(|_| *keep).unwrap_or(0.))
            .collect(),
        values,
        defined,
        effective_block: block,
        blocks: groups,
        tau_int: tau.filter(|_| resampling.block() == BlockSize::Auto),
    })
}

/// Resampled correlators of `moments` through `resample_blocks` with the
/// statistic `BlockMoments::estimate(subtraction)` on the weighted sums; `tau`
/// overrides the autocorrelation time used by `BlockSize::Auto` (otherwise it
/// is estimated from the pooled lag-wise connected correlator, and is 1/2
/// when that has no variance). Each resample recomputes its own disconnected
/// part. Fewer than two blocks is `GasError::Numerical`.
///
/// A resample deletes or draws whole blocks of time origins, so a pair whose
/// origin survives keeps its sink even when that sink lies in a deleted block:
/// beyond `τ = effective_block` the deletion removes no product of a lag at
/// all and the resampled error is a lower bound. `moments::straddling_pairs`
/// counts the pairs this leaves in. Excising the observations instead
/// overshoots, so the geometry stays and is stated rather than changed.
pub fn resample(
    moments: &BlockMoments,
    subtraction: Subtraction,
    resampling: &Resampling,
    tau: Option<f64>,
) -> Result<Samples> {
    moments.validate()?;
    resampling.validate()?;
    let origins = moments.origins();
    let tau = tau.or_else(|| {
        (resampling.block() == BlockSize::Auto).then(|| {
            tau_int_of_correlator(&moments.estimate(Subtraction::LagMeans), origins)
                .map_or(0.5, |t| t.tau)
        })
    });
    let base = moments.origins_per_block;
    let coarse = moments.coarsen(block_origins(base, origins, resampling, tau)? / base)?;
    // Delete-one sums come from running sums: subtracting a block from the
    // total would leave rounding residue in a denominator that is exactly zero.
    let running = match resampling.kind() {
        ResampleKind::Jackknife => {
            enforce(
                checked_mul(coarse.buffer_bytes()?, 2)?,
                DEFAULT_MEMORY_BYTES,
            )?;
            Some((coarse.running(false), coarse.running(true)))
        }
        ResampleKind::Bootstrap => None,
    };
    let statistic = |multiplicity: &[f64]| {
        let mut holes = multiplicity.iter().enumerate().filter(|(_, m)| **m != 1.);
        let sums = match (&running, holes.next(), holes.next()) {
            (Some((forward, backward)), Some((k, m)), None) if *m == 0. => {
                let mut sums = LagMoments::zeros(coarse.lags, coarse.components);
                if k > 0 {
                    sums.add(forward, k - 1, 1.);
                }
                if k + 1 < coarse.blocks {
                    sums.add(backward, k + 1, 1.);
                }
                sums
            }
            _ => coarse.weighted(multiplicity),
        };
        sums.estimate(subtraction)
    };
    resample_blocks(
        coarse.blocks,
        coarse.origins_per_block,
        origins,
        resampling,
        tau,
        &statistic,
    )
}

/// Covariance `[dimension, dimension]` of a resample table about the mean of
/// its resamples, with the factor of `samples.kind`. Undefined entries give
/// zero rows and columns.
pub fn sample_covariance(samples: &Samples) -> Vec<f64> {
    let (d, n) = (samples.dimension, samples.count);
    let mut covariance = vec![0.; d * d];
    if n < 2 {
        return covariance;
    }
    let mean = means(samples);
    for row in samples.values.chunks_exact(d.max(1)) {
        for i in (0..d).filter(|&i| samples.defined[i]) {
            for j in (i..d).filter(|&j| samples.defined[j]) {
                covariance[i * d + j] += (row[i] - mean[i]) * (row[j] - mean[j]);
            }
        }
    }
    let factor = factor(samples.kind, n);
    for i in 0..d {
        for j in i..d {
            covariance[i * d + j] *= factor;
            covariance[j * d + i] = covariance[i * d + j];
        }
    }
    covariance
}

/// Square roots of the covariance diagonal; `None` where undefined.
pub fn errors(samples: &Samples) -> Vec<Option<f64>> {
    let (d, n) = (samples.dimension, samples.count);
    let mean = means(samples);
    (0..d)
        .map(|i| {
            (samples.defined[i] && n >= 2).then(|| {
                let spread: f64 = samples
                    .values
                    .chunks_exact(d)
                    .map(|row| (row[i] - mean[i]).powi(2))
                    .sum();
                (factor(samples.kind, n) * spread).sqrt()
            })
        })
        .collect()
}
fn means(samples: &Samples) -> Vec<f64> {
    let mut mean = vec![0.; samples.dimension];
    for row in samples.values.chunks_exact(samples.dimension.max(1)) {
        for (m, x) in mean.iter_mut().zip(row) {
            *m += x / samples.count as f64;
        }
    }
    mean
}
fn factor(kind: ResampleKind, n: usize) -> f64 {
    match kind {
        ResampleKind::Jackknife => (n - 1) as f64 / n as f64,
        ResampleKind::Bootstrap => 1. / (n - 1) as f64,
    }
}
