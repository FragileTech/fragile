//! Effective rates of a correlator.
use crate::physics::{
    numerics::{Samples, errors, sample_covariance},
    spectroscopy::{config::EffectiveMassKind, estimators::Estimated},
};

/// `[value, error]` per lag of the estimate. The error comes from the
/// resamples when every resample is defined at the lag, otherwise from the
/// covariance by the delta method. `None` where the definition does not
/// exist: a missing or non-positive ratio for `LogRatio`, a ratio below 1 or
/// a missing neighbour for `Cosh`, the last lag(s). Never a placeholder value.
pub fn effective_mass(data: &Estimated, kind: EffectiveMassKind) -> Vec<Option<[f64; 2]>> {
    let estimate = &data.estimate;
    let n = estimate.lags.len();
    // Shape is settled once: below, an entry of `defined` is reachable for
    // every index the supports use.
    let resamples = Some(&data.samples).filter(|s| s.dimension == n && s.validate().is_ok());
    let owned;
    let covariance: &[f64] = match &estimate.covariance {
        Some(c) if c.len() == n * n => c,
        _ => {
            owned = resamples.map_or_else(Vec::new, sample_covariance);
            &owned
        }
    };
    (0..n)
        .map(|k| {
            let support = support(kind, k, n)?;
            let step = spacing(&estimate.lags, &support)? as f64 * estimate.time_step;
            let central: Vec<f64> = support
                .iter()
                .map(|&i| estimate.value.get(i).copied().flatten())
                .collect::<Option<_>>()?;
            let (value, gradient) = rate(kind, &central, step)?;
            let error = resamples
                .and_then(|s| resample_rates(s, &support, kind, step))
                .and_then(|rates| spread(value, rates, &data.samples))
                .or_else(|| delta(covariance, n, &support, &gradient))?;
            (value.is_finite() && error.is_finite()).then_some([value, error])
        })
        .collect()
}

/// `m` with `cosh m = ratio`; `None` below 1.
pub fn cosh_rate(ratio: f64) -> Option<f64> {
    (ratio >= 1.).then(|| ratio.acosh())
}

/// Entries of the estimate that a rate at entry `k` reads, in the order of
/// its gradient: `[k, k+1]` for the log ratio, `[k−1, k, k+1]` for the
/// arccosh. `None` at an entry whose neighbours do not exist.
fn support(kind: EffectiveMassKind, k: usize, lags: usize) -> Option<Vec<usize>> {
    match kind {
        EffectiveMassKind::LogRatio => (k + 1 < lags).then(|| vec![k, k + 1]),
        EffectiveMassKind::Cosh => (k >= 1 && k + 1 < lags).then(|| vec![k - 1, k, k + 1]),
    }
}

/// Common lag spacing of `support` in frames. Lags need not be contiguous, and
/// the arccosh has no meaning on entries that are not equally spaced.
fn spacing(lags: &[usize], support: &[usize]) -> Option<usize> {
    let step = lags[support[1]].checked_sub(lags[support[0]])?;
    (step > 0
        && support
            .windows(2)
            .all(|w| lags[w[1]].checked_sub(lags[w[0]]) == Some(step)))
    .then_some(step)
}

/// The rate at the entry and the gradient of its definition with respect to
/// the correlator at `support`, over a lag spacing `step` in the time unit of
/// the estimate. A ratio at or below 1 has an infinite arccosh error and a
/// non-positive correlator is not a decaying one, so both give `None`.
fn rate(kind: EffectiveMassKind, c: &[f64], step: f64) -> Option<(f64, Vec<f64>)> {
    if step <= 0. {
        return None;
    }
    match kind {
        EffectiveMassKind::LogRatio => {
            let (near, far) = (c[0], c[1]);
            if near <= 0. || far <= 0. {
                return None;
            }
            let gradient = vec![1. / (step * near), -1. / (step * far)];
            Some(((near / far).ln() / step, gradient))
        }
        EffectiveMassKind::Cosh => {
            let middle = c[1];
            if middle <= 0. {
                return None;
            }
            let ratio = 0.5 * (c[0] + c[2]) / middle;
            if ratio <= 1. {
                return None;
            }
            let value = cosh_rate(ratio)? / step;
            let scale = 1. / (step * (ratio * ratio - 1.).sqrt());
            let gradient = vec![
                scale / (2. * middle),
                -scale * ratio / middle,
                scale / (2. * middle),
            ];
            Some((value, gradient))
        }
    }
}

/// The rate of every resample at the entry; `None` as soon as one resample
/// leaves the definition undefined, which sends the error to the delta method.
fn resample_rates(
    samples: &Samples,
    support: &[usize],
    kind: EffectiveMassKind,
    step: f64,
) -> Option<Vec<f64>> {
    if !support.iter().all(|&i| samples.defined[i]) {
        return None;
    }
    (0..samples.count)
        .map(|r| {
            let row = samples.sample(r);
            let c: Vec<f64> = support.iter().map(|&i| row[i]).collect();
            rate(kind, &c, step).map(|(value, _)| value)
        })
        .collect()
}

/// Spread of the rate over the resamples, through the same estimator that
/// gives the correlator its errors, so that both carry the factor of `kind`.
fn spread(value: f64, rates: Vec<f64>, samples: &Samples) -> Option<f64> {
    let table = Samples {
        kind: samples.kind,
        dimension: 1,
        count: rates.len(),
        central: vec![value],
        values: rates,
        defined: vec![true],
        effective_block: samples.effective_block,
        blocks: samples.blocks,
        tau_int: samples.tau_int,
    };
    errors(&table)[0]
}

/// Delta method on the lag covariance: `sqrt(gᵀ Σ g)` over `support`. A lag
/// the covariance gives no variance carries no error either, and a rate is
/// missing rather than exact.
fn delta(covariance: &[f64], n: usize, support: &[usize], gradient: &[f64]) -> Option<f64> {
    (covariance.len() == n * n && support.iter().all(|&i| covariance[i * n + i] > 0.)).then(|| {
        let variance: f64 = support
            .iter()
            .zip(gradient)
            .flat_map(|(&i, g)| {
                support
                    .iter()
                    .zip(gradient)
                    .map(move |(&j, h)| g * h * covariance[i * n + j])
            })
            .sum();
        variance.max(0.).sqrt()
    })
}
