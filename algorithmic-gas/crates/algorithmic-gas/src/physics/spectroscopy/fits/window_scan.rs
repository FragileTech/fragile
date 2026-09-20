//! Single-exponential fits over all windows, model-averaged with
//! `w ∝ exp(−AIC/2)`, `AIC = χ² + 2k + 2 N_cut` (Jay–Neil). The scan costs one
//! eigendecomposition per window, so it grows as the fifth power of the number
//! of usable lags; `WindowScanConfig::max_usable` keeps that bounded.
use crate::{
    Result,
    error::require,
    physics::{
        numerics::{Whitener, chi2_q, linear_fit, sample_covariance, submatrix},
        spectroscopy::{
            config::AnalysisConfig,
            estimators::{Estimated, resample_support_note},
            report::{
                EstimatorKind, FitDiagnostics, FitMethodKind, FitOutcome, MassEstimate,
                RATE_QUANTITY, RATE_QUANTITY_EUCLIDEAN, RATE_QUANTITY_SOURCE_FROZEN, WindowFit,
            },
        },
    },
};

/// A correlator this many errors below zero inside the examined range is not a
/// sum of decaying exponentials, whatever a fit would return.
const REJECT_Z: f64 = 3.;
/// Below this probability the best window is reported as a poor description.
const Q_MIN: f64 = 0.01;
const LOG_NOTE: &str = "Rates come from a log-linear fit; the logarithm of a point at \
                        signal-to-noise s carries a bias of about -1/(2 s^2).";

/// Windows `[t_min, t_max]` from `analysis.window_scan`, correlated χ² through
/// a `Whitener` with `analysis.svd_cut` (or the stated diagonal approximation),
/// `value = Σ w m`, `statistical² = Σ w σ²`, `systematic² = Σ w (m − value)²`.
/// A correlator that changes sign or oscillates inside the usable range sets
/// `model_rejected`; a rate below `min_rate_snr` sets `no_signal`. Neither
/// case reports a number, and no positivity filter biases the average.
/// `N_cut` counts the points of the contiguous usable range left out of a
/// window. A whitener of reduced rank deflates χ²; the notes say so.
pub fn window_scan(data: &Estimated, analysis: &AnalysisConfig) -> Result<FitOutcome> {
    let scan = &analysis.window_scan;
    scan.validate()?;
    data.samples.validate()?;
    let estimate = &data.estimate;
    let lags = estimate.lags.len();
    require(
        (1..=4096).contains(&lags)
            && estimate.value.len() == lags
            && data.samples.dimension == lags
            && estimate.lags.windows(2).all(|w| w[0] < w[1])
            && estimate.time_step.is_finite()
            && estimate.time_step > 0.,
        "window scan needs 1..=4096 ascending lags with a positive time step, a correlator value \
         and a resample table of that size",
    )?;
    let owned;
    let covariance: &[f64] = match &estimate.covariance {
        Some(c) if c.len() == lags * lags => c,
        _ => {
            owned = sample_covariance(&data.samples);
            &owned
        }
    };
    let error: Vec<f64> = (0..lags)
        .map(|i| covariance[i * lags + i].max(0.).sqrt())
        .collect();
    let time: Vec<f64> = estimate
        .lags
        .iter()
        .map(|&l| l as f64 * estimate.time_step)
        .collect();
    let mut outcome = FitOutcome {
        method: FitMethodKind::WindowScan,
        diagnostics: FitDiagnostics {
            correlated: scan.correlated,
            svd_cut: analysis.svd_cut,
            ..FitDiagnostics::default()
        },
        ..FitOutcome::default()
    };

    // The usable range is contiguous: the first lag that carries no signal
    // ends it, and a lag that is significantly negative ends it as well
    // because the logarithm of it does not exist.
    let start = estimate.lags.partition_point(|&l| l < scan.t_min);
    let cap = scan
        .t_max
        .map_or(lags, |t| estimate.lags.partition_point(|&l| l <= t));
    let mut signal = vec![];
    while start + signal.len() < cap {
        let k = start + signal.len();
        let Some(c) = estimate.value[k]
            .filter(|c| *c > 0. && error[k] > 0. && *c / error[k] >= scan.min_point_snr)
        else {
            break;
        };
        signal.push(c);
    }
    let usable = signal.len();
    let end = start + usable;
    let negative = (start..cap)
        .find(|&k| estimate.value[k].is_some_and(|c| error[k] > 0. && c < -REJECT_Z * error[k]));
    if let Some(k) = negative.filter(|&k| k <= (2 * end).max(start.saturating_add(scan.min_points)))
    {
        outcome.diagnostics.model_rejected =
            Some(format!("sign change at lag {}", estimate.lags[k]));
        return Ok(outcome);
    }
    if usable < scan.min_points {
        outcome.diagnostics.no_signal = Some("too few usable lags".into());
        return Ok(outcome);
    }
    require(
        usable <= scan.max_usable,
        format!(
            "window scan fits at most {} usable lags; lower window_scan.t_max or raise \
             window_scan.max_usable",
            scan.max_usable
        ),
    )?;

    // In log space the model is a straight line and the covariance of ln C is
    // the covariance of C divided by the two correlator values.
    let log_covariance: Vec<f64> = (0..usable * usable)
        .map(|c| {
            let (i, j) = (c / usable, c % usable);
            covariance[(start + i) * lags + start + j] / (signal[i] * signal[j])
        })
        .collect();
    let rank = Whitener::new(&log_covariance, usable, analysis.svd_cut)?.rank();
    // The widest window of the scan fits every usable lag, so the resample
    // table has to support that many points; `estimate` asked the same
    // question of the whole measured range.
    outcome
        .notes
        .extend(resample_support_note(data.samples.blocks, usable));
    let mut windows = vec![];
    for a in 0..usable {
        for b in a + scan.min_points - 1..usable {
            let offsets: Vec<usize> = (a..=b).collect();
            let points = offsets.len();
            let block = submatrix(&log_covariance, usable, &offsets);
            let y: Vec<f64> = offsets.iter().map(|&i| signal[i].ln()).collect();
            let design: Vec<f64> = offsets
                .iter()
                .flat_map(|&i| [1., -time[start + i]])
                .collect();
            let (fit, variance) = if scan.correlated {
                let whitener = Whitener::new(&block, points, analysis.svd_cut)?;
                let fit = linear_fit(&design, &y, &whitener, 2)?;
                let variance = fit.covariance[3];
                (fit, variance)
            } else {
                let errors: Vec<f64> = (0..points).map(|i| block[i * points + i].sqrt()).collect();
                let weight: Vec<f64> = errors.iter().map(|e| 1. / e).collect();
                let fit = linear_fit(&design, &y, &Whitener::diagonal(&errors)?, 2)?;
                let variance = sandwich(&design, &fit.covariance, &weight, &block);
                (fit, variance)
            };
            windows.push(WindowFit {
                t_min: estimate.lags[start + a],
                t_max: estimate.lags[start + b],
                value: fit.beta[1],
                error: variance.max(0.).sqrt(),
                chi2: fit.chi2,
                dof: fit.dof,
                weight: 0.,
                // Two parameters per window, and every usable lag outside it is
                // a cut point.
                aic: fit.chi2 + 4. + 2. * (usable - points) as f64,
                nexp: 1,
                svd_cut: None,
            });
        }
    }

    // Model averaging: the weights are shifted by the best AIC before they are
    // exponentiated, and no window is dropped for the rate it happens to give.
    let best_criterion = windows
        .iter()
        .map(|window| window.aic)
        .fold(f64::INFINITY, f64::min);
    let total: f64 = windows
        .iter_mut()
        .map(|window| {
            window.weight = (-0.5 * (window.aic - best_criterion)).exp();
            window.weight
        })
        .sum();
    for window in &mut windows {
        window.weight /= total;
    }
    let value: f64 = windows.iter().map(|w| w.weight * w.value).sum();
    let statistical: f64 = windows
        .iter()
        .map(|w| w.weight * w.error * w.error)
        .sum::<f64>()
        .max(0.)
        .sqrt();
    let systematic: f64 = windows
        .iter()
        .map(|w| w.weight * (w.value - value) * (w.value - value))
        .sum::<f64>()
        .max(0.)
        .sqrt();
    let spread = statistical.hypot(systematic);
    require(
        value.is_finite() && spread.is_finite(),
        "window scan model average overflow",
    )?;
    let best = (1..windows.len()).fold(0, |b, i| {
        if windows[i].weight.total_cmp(&windows[b].weight).is_gt() {
            i
        } else {
            b
        }
    });
    outcome.diagnostics.chi2 = Some(windows[best].chi2);
    outcome.diagnostics.dof = Some(windows[best].dof);
    outcome.diagnostics.q = chi2_q(windows[best].chi2, windows[best].dof);
    outcome.diagnostics.window = Some([windows[best].t_min, windows[best].t_max]);
    outcome.diagnostics.n_windows = windows.len();
    outcome.diagnostics.covariance_rank = Some(rank);
    if spread > 0. && value / spread >= scan.min_rate_snr {
        outcome.mass = Some(MassEstimate {
            quantity: quantity(data.kind).into(),
            value,
            error: spread,
            statistical,
            systematic,
            method: FitMethodKind::WindowScan,
            time_unit: estimate.time_unit,
            prior_dominance: None,
        });
    } else {
        outcome.diagnostics.no_signal = Some("rate S/N below threshold".into());
    }
    if !scan.correlated {
        outcome.notes.push(
            "Each window is fitted with a diagonal chi^2; its slope error still carries the \
             full lag covariance."
                .into(),
        );
    }
    // Only a correlated chi^2 reads the floored covariance, so only it is
    // deflated by a reduced rank; the diagonal arm reports the rank alone.
    if rank < usable && scan.correlated {
        outcome.notes.push(format!(
            "The SVD floor is active on {} of {usable} directions of the lag covariance: chi^2 \
             is deflated and its probability is not calibrated.",
            usable - rank
        ));
    }
    if let Some(q) = outcome.diagnostics.q.filter(|q| *q < Q_MIN) {
        outcome.notes.push(format!(
            "The best window has q = {q:.4}: a single exponential describes it poorly."
        ));
    }
    outcome.notes.push(LOG_NOTE.into());
    outcome.windows = windows;
    Ok(outcome)
}

/// Variance of the slope of a diagonally weighted fit under the full log
/// covariance: `β = G y` with `G = (Xᵀ W² X)⁻¹ Xᵀ W²`, so `Var = G Σ Gᵀ`.
/// `weight` is the diagonal of `W`, the design has the two columns of a line.
fn sandwich(design: &[f64], parameters: &[f64], weight: &[f64], block: &[f64]) -> f64 {
    let n = weight.len();
    let row: Vec<f64> = (0..n)
        .map(|j| {
            weight[j]
                * weight[j]
                * (parameters[2] * design[j * 2] + parameters[3] * design[j * 2 + 1])
        })
        .collect();
    (0..n * n).map(|c| row[c / n] * block[c] * row[c % n]).sum()
}

/// What a rate of this estimator is: only the frame-mean correlator carries a
/// transfer-matrix reading.
fn quantity(kind: EstimatorKind) -> &'static str {
    match kind {
        EstimatorKind::FrameMean => RATE_QUANTITY,
        EstimatorKind::SourceFrozen => RATE_QUANTITY_SOURCE_FROZEN,
        EstimatorKind::EuclideanTime => RATE_QUANTITY_EUCLIDEAN,
    }
}
