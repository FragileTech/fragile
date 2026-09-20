//! Bayesian fit of `C_c(t) = Σ_n a_{c,n}² exp(−E_n t)`, `E_n = Σ_{m ≤ n} dE_m`,
//! with gaps shared by every channel of a group. Autocorrelations have equal
//! source and sink amplitudes. Priors are channel agnostic and scale free;
//! every result carries the prior-dominance diagnostic. Positive amplitudes
//! cannot represent the negative spectral weights of a non-reversible chain:
//! such data end as a rejected model, never as a fit.
use crate::{
    GasError, Result,
    error::require,
    physics::{
        numerics::{
            LevenbergConfig, Model, NonlinearFit, Prior, Samples, Whitener, chi2_q, minimize,
            sample_covariance,
        },
        spectroscopy::{
            config::{AnalysisConfig, ChannelGroup, MultiExponentialConfig},
            contract::Availability,
            estimators::{Estimated, resample_support_note},
            report::{
                EstimatorKind, FitDiagnostics, FitMethodKind, FitOutcome, GroupFit, MassEstimate,
                PriorDominance, RATE_QUANTITY, RATE_QUANTITY_EUCLIDEAN,
                RATE_QUANTITY_SOURCE_FROZEN,
            },
        },
    },
};
/// Standard deviations below zero at which a lag contradicts a sum of
/// positive exponentials.
const REJECT_Z: f64 = 3.;
/// χ² probability below which the fitted model is rejected.
const MIN_Q: f64 = 0.01;
/// Posterior/prior width ratio from which the prior is worth a note.
const PRIOR_INFLUENCE: f64 = 0.3;
/// `C_c(t) = Σ_n exp(p_{c,n}) exp(−E_n t)` with `E_n = Σ_{m ≤ n} exp(p_m)`,
/// the lag `t` in frames. The parameters are `ln dE_0 … ln dE_{nexp−1}` and
/// then `ln a²` per channel and level: equal source and sink amplitudes of an
/// autocorrelation leave one positive amplitude each.
struct Decays {
    nexp: usize,
    channels: usize,
    /// Frame lag of every fitted point and the channel it belongs to.
    times: Vec<f64>,
    channel: Vec<usize>,
}
impl Decays {
    /// Cumulative levels `E_n` of a parameter point.
    fn energies(&self, p: &[f64]) -> Vec<f64> {
        p[..self.nexp]
            .iter()
            .scan(0., |e, x| {
                *e += x.exp();
                Some(*e)
            })
            .collect()
    }
}
impl Model for Decays {
    fn parameters(&self) -> usize {
        self.nexp * (1 + self.channels)
    }
    fn outputs(&self) -> usize {
        self.times.len()
    }
    fn evaluate(&self, p: &[f64], out: &mut [f64]) {
        let e = self.energies(p);
        for ((&t, &c), out) in self.times.iter().zip(&self.channel).zip(out) {
            let first = self.nexp * (1 + c);
            *out = (0..self.nexp)
                .map(|n| (p[first + n] - e[n] * t).exp())
                .sum();
        }
    }
    fn jacobian(&self, p: &[f64], out: &mut [f64]) {
        let (e, k) = (self.energies(p), self.parameters());
        out.fill(0.);
        for ((&t, &c), row) in self
            .times
            .iter()
            .zip(&self.channel)
            .zip(out.chunks_exact_mut(k))
        {
            let first = self.nexp * (1 + c);
            for n in 0..self.nexp {
                let term = (p[first + n] - e[n] * t).exp();
                row[first + n] = term;
                for m in 0..=n {
                    row[m] -= t * p[m].exp() * term;
                }
            }
        }
    }
}
/// One joint fit of the shared gaps, before a report type is chosen. `levels`
/// holds every cumulative level the minimizer produced, whether or not the
/// guards let it be reported.
pub(super) struct GapFit {
    pub(super) levels: Vec<MassEstimate>,
    pub(super) diagnostics: FitDiagnostics,
    pub(super) notes: Vec<String>,
    pub(super) converged: bool,
    /// Fitted points and free parameters, which an AIC weight needs.
    pub(super) points: usize,
    pub(super) parameters: usize,
}
impl GapFit {
    /// A fit that was not attempted; its diagnostics carry the reason.
    fn declined(diagnostics: FitDiagnostics, notes: Vec<String>, points: usize) -> Self {
        Self {
            levels: vec![],
            diagnostics,
            notes,
            converged: false,
            points,
            parameters: 0,
        }
    }
    /// Only a converged, accepted, data-dominated fit states a rate.
    pub(super) fn reported(&self) -> bool {
        self.converged
            && !self.levels.is_empty()
            && self.diagnostics.model_rejected.is_none()
            && self.diagnostics.no_signal.is_none()
            && !self
                .diagnostics
                .prior_dominance
                .is_some_and(|d| d.dominated)
    }
}
/// Value and error of a lag entry that carries an estimate.
fn point(data: &Estimated, k: usize) -> Option<(f64, f64)> {
    let (value, error) = (data.estimate.value[k]?, data.estimate.error[k]?);
    (data.samples.defined[k] && error > 0.).then_some((value, error))
}
/// Entries of `estimate.lags` the fit uses, with their values: from `t_min` to
/// `t_max`, or without one to the end of the contiguous run that stays
/// positive and above `min_snr`.
fn window(data: &Estimated, t_min: usize, t_max: Option<usize>, min_snr: f64) -> Vec<(usize, f64)> {
    let lags = &data.estimate.lags;
    let from = (0..lags.len()).filter(|&k| lags[k] >= t_min);
    match t_max {
        Some(cap) => from
            .take_while(|&k| lags[k] <= cap)
            .filter_map(|k| point(data, k).map(|(value, _)| (k, value)))
            .collect(),
        None => from
            .map_while(|k| {
                point(data, k)
                    .filter(|&(value, error)| value > 0. && value / error >= min_snr)
                    .map(|(value, _)| (k, value))
            })
            .collect(),
    }
}
/// First lag of the scanned range that is significantly negative.
fn sign_change(data: &Estimated, t_min: usize, cap: usize) -> Option<usize> {
    let lags = &data.estimate.lags;
    (0..lags.len())
        .filter(|&k| (t_min..=cap).contains(&lags[k]))
        .find_map(|k| {
            let (value, error) = point(data, k)?;
            (value < -REJECT_Z * error).then_some(lags[k])
        })
}
/// How a rate of this estimator must be called in a report.
fn quantity(kind: EstimatorKind) -> &'static str {
    match kind {
        EstimatorKind::FrameMean => RATE_QUANTITY,
        EstimatorKind::SourceFrozen => RATE_QUANTITY_SOURCE_FROZEN,
        EstimatorKind::EuclideanTime => RATE_QUANTITY_EUCLIDEAN,
    }
}
/// Posterior width and mean shift of the gap `ln dE_n` in units of its prior
/// width; `width_ratio²` is the share of the information that came from the
/// prior.
fn dominance(fit: &NonlinearFit, n: usize, config: &MultiExponentialConfig) -> PriorDominance {
    let k = fit.p.len();
    let width_ratio = fit.covariance[n * k + n].max(0.).sqrt() / config.log_gap_sigma;
    PriorDominance {
        width_ratio,
        shift_sigma: (fit.p[n] - config.log_gap_mean) / config.log_gap_sigma,
        dominated: width_ratio > config.dominance_ratio,
    }
}
/// Gap priors, then the wide amplitude prior of each channel about the
/// logarithm of its first fitted value.
fn priors(
    config: &MultiExponentialConfig,
    windows: &[Vec<(usize, f64)>],
) -> impl Iterator<Item = Prior> {
    let gap = Prior::Gaussian {
        mean: config.log_gap_mean,
        sigma: config.log_gap_sigma,
    };
    let nexp = config.nexp;
    let sigma = config.log_amplitude_sigma;
    std::iter::repeat_n(gap, nexp).chain(windows.iter().flat_map(move |w| {
        let amplitude = Prior::Gaussian {
            mean: w[0].1.abs().ln(),
            sigma,
        };
        std::iter::repeat_n(amplitude, nexp)
    }))
}
/// Deterministic start: the log-ratio rate of the first two fitted lags of the
/// leading channel, doubled gaps above it, and amplitudes that reproduce each
/// channel's first fitted value.
fn start(
    members: &[Estimated],
    windows: &[Vec<(usize, f64)>],
    config: &MultiExponentialConfig,
) -> Vec<f64> {
    let lag = |c: usize, i: usize| members[c].estimate.lags[windows[c][i].0] as f64;
    let lead = &windows[0];
    let theta = match lead.get(1) {
        Some(&(_, y)) if lead[0].1 > y && y > 0. => {
            ((lead[0].1 / y).ln() / (lag(0, 1) - lag(0, 0))).ln()
        }
        _ => config.log_gap_mean,
    };
    let (nexp, ground) = (config.nexp, theta.exp());
    let split = if nexp > 1 { (nexp as f64).ln() } else { 0. };
    (0..nexp)
        .map(|n| if n == 0 { theta } else { theta + 2_f64.ln() })
        .chain((0..windows.len()).flat_map(|c| {
            let scale = windows[c][0].1.abs().ln() + ground * lag(c, 0) - split;
            (0..nexp).map(move |n| scale - n as f64 * 2_f64.ln())
        }))
        .collect()
}
/// The joint fit both report shapes are built from. Every member is fitted
/// over its own window with one shared set of gaps, against the joint
/// covariance of the resamples the members share.
pub(super) fn gaps(
    members: &[Estimated],
    config: &MultiExponentialConfig,
    analysis: &AnalysisConfig,
) -> Result<GapFit> {
    let first = members
        .first()
        .ok_or_else(|| GasError::Capability("multi-exponential fit needs a channel".into()))?;
    let step = first.estimate.time_step;
    require(
        members.len() <= 16
            && step.is_finite()
            && step > 0.
            && members.iter().all(|m| {
                let lags = m.estimate.lags.len();
                (1..=4096).contains(&lags)
                    && m.estimate.value.len() == lags
                    && m.estimate.error.len() == lags
                    && m.samples.dimension == lags
                    && m.estimate.lags.is_sorted_by(|a, b| a < b)
                    && m.estimate.time_unit == first.estimate.time_unit
                    && m.estimate.time_step == step
            }),
        "a multi-exponential fit needs at most 16 channels of 1..=4096 ascending lags, each with \
         a value, an error and a resample entry per lag, one common time unit and a positive \
         time step",
    )?;
    let scan = &analysis.window_scan;
    let mut notes = vec![];
    let mut diagnostics = FitDiagnostics {
        correlated: true,
        svd_cut: analysis.svd_cut,
        ..FitDiagnostics::default()
    };
    let windows: Vec<Vec<(usize, f64)>> = members
        .iter()
        .map(|m| window(m, config.t_min, config.t_max, scan.min_point_snr))
        .collect();
    let points: usize = windows.iter().map(Vec::len).sum();
    diagnostics.window = members
        .iter()
        .zip(&windows)
        .filter_map(|(m, w)| w.last().map(|&(k, _)| m.estimate.lags[k]))
        .max()
        .map(|t| [config.t_min, t]);
    for (m, w) in members.iter().zip(&windows) {
        let cap = config
            .t_max
            .or_else(|| m.estimate.lags.last().copied())
            .unwrap_or(config.t_min);
        let end = w
            .last()
            .map_or(config.t_min, |&(k, _)| m.estimate.lags[k].saturating_add(1));
        let limit = end
            .saturating_mul(2)
            .max(config.t_min.saturating_add(scan.min_points));
        if let Some(lag) = sign_change(m, config.t_min, cap).filter(|&lag| lag <= limit) {
            diagnostics.model_rejected = Some(format!("sign change at lag {lag}"));
            return Ok(GapFit::declined(diagnostics, notes, points));
        }
    }
    if points < 2 * config.nexp + 1
        || windows
            .iter()
            .any(|w| w.first().is_none_or(|&(_, value)| value == 0.))
    {
        diagnostics.no_signal = Some("too few usable lags".into());
        return Ok(GapFit::declined(diagnostics, notes, points));
    }
    let model = Decays {
        nexp: config.nexp,
        channels: members.len(),
        times: members
            .iter()
            .zip(&windows)
            .flat_map(|(m, w)| w.iter().map(|&(k, _)| m.estimate.lags[k] as f64))
            .collect(),
        channel: (0..members.len())
            .flat_map(|c| std::iter::repeat_n(c, windows[c].len()))
            .collect(),
    };
    let data: Vec<f64> = windows.iter().flatten().map(|&(_, value)| value).collect();
    let selected = members
        .iter()
        .zip(&windows)
        .map(|(m, w)| {
            m.samples
                .select(&w.iter().map(|&(k, _)| k).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<Samples>>>()?;
    let joint = Samples::concat(&selected.iter().collect::<Vec<_>>())?;
    // The fitted window is narrower than the measured range, so the question
    // `estimate` asked of the whole correlator is asked again of the points
    // this fit actually reads.
    notes.extend(resample_support_note(joint.blocks, points));
    let whitener = Whitener::new(&sample_covariance(&joint), points, analysis.svd_cut)?;
    diagnostics.covariance_rank = Some(whitener.rank());
    if whitener.rank() < points {
        notes.push(format!(
            "the SVD floor is active on {} of {points} directions of the joint covariance: chi2 \
             is deflated, its probability is not calibrated, and the degrees of freedom count \
             the rank rather than the fitted points",
            points - whitener.rank()
        ));
    }
    let priors: Vec<Prior> = priors(config, &windows).collect();
    let fit = minimize(
        &model,
        &data,
        &whitener,
        &priors,
        &start(members, &windows, config),
        &LevenbergConfig::default(),
    )?;
    let (k, quantity) = (fit.p.len(), quantity(first.kind));
    let gap: Vec<f64> = fit.p[..config.nexp].iter().map(|x| x.exp()).collect();
    let levels: Vec<MassEstimate> = (0..config.nexp)
        .map(|n| {
            let variance: f64 = (0..=n)
                .flat_map(|i| (0..=n).map(move |j| (i, j)))
                .map(|(i, j)| gap[i] * fit.covariance[i * k + j] * gap[j])
                .sum();
            let error = variance.max(0.).sqrt() / step;
            MassEstimate {
                quantity: quantity.into(),
                value: gap[..=n].iter().sum::<f64>() / step,
                error,
                statistical: error,
                systematic: 0.,
                method: FitMethodKind::MultiExponential,
                time_unit: first.estimate.time_unit,
                prior_dominance: Some(dominance(&fit, n, config)),
            }
        })
        .collect();
    let ground = dominance(&fit, 0, config);
    // A floored direction of the joint covariance carries no independent
    // constraint: a member listed twice, or two members that are linear
    // combinations of one another, add points but no information, and the
    // probability has to be read at the rank so that duplicating a channel
    // cannot improve it.
    let dof = fit.dof.min(whitener.rank());
    diagnostics.n_windows = 1;
    diagnostics.chi2 = Some(fit.chi2);
    diagnostics.dof = Some(dof);
    diagnostics.q = chi2_q(fit.chi2, dof);
    diagnostics.prior_dominance = Some(ground);
    if !fit.converged {
        notes.push("the minimizer did not converge".into());
    }
    if levels[0].error <= 0. || levels[0].value / levels[0].error < scan.min_rate_snr {
        diagnostics.no_signal = Some("rate S/N below threshold".into());
    }
    if let Some(q) = diagnostics.q.filter(|q| *q < MIN_Q) {
        diagnostics.model_rejected = Some(format!("chi2 probability {q:.2e} below {MIN_Q}"));
    }
    notes.push(format!(
        "augmented chi2 carries a prior part of {:.6}",
        fit.chi2_prior
    ));
    notes.push(format!(
        "amplitude priors are Gaussians of width {} about the logarithm of each channel's first \
         fitted value: they read the data",
        config.log_amplitude_sigma
    ));
    if ground.dominated {
        notes.push("ground gap prior-dominated: levels are not measurements".into());
    } else if ground.width_ratio > PRIOR_INFLUENCE {
        notes.push(format!(
            "prior-influenced: {:.0}% of the information on the ground gap comes from the prior",
            100. * ground.width_ratio * ground.width_ratio
        ));
    }
    for (n, level) in levels.iter().enumerate().skip(1) {
        if level.prior_dominance.is_some_and(|d| d.dominated) {
            notes.push(format!("level {n}: gap prior-dominated"));
        }
    }
    Ok(GapFit {
        levels,
        diagnostics,
        notes,
        converged: fit.converged,
        points,
        parameters: k,
    })
}
/// Fit of one or more channels that share gaps; `members` must have been
/// resampled over the same blocks. Parameters are `ln dE_n` and `ln a²`, so
/// the log-normal priors are Gaussian. A posterior/prior width ratio of
/// `ln dE_0` above `config.dominance_ratio` reports no rate. Every excited
/// level carries the `MassEstimate::prior_dominance` of its own gap.
/// `FitDiagnostics::chi2` is the augmented χ² and `q` its probability; the
/// prior part is stated in a note. `t_max: None` ends at the last lag that
/// passes `window_scan.min_point_snr`.
pub fn fit(
    members: &[Estimated],
    config: &MultiExponentialConfig,
    analysis: &AnalysisConfig,
) -> Result<FitOutcome> {
    config.validate()?;
    analysis.window_scan.validate()?;
    let fit = gaps(members, config, analysis)?;
    let (mass, excited) = match (fit.reported(), fit.levels.split_first()) {
        (true, Some((ground, excited))) => (Some(ground.clone()), excited.to_vec()),
        _ => (None, vec![]),
    };
    Ok(FitOutcome {
        method: FitMethodKind::MultiExponential,
        mass,
        excited,
        diagnostics: fit.diagnostics,
        windows: vec![],
        notes: fit.notes,
    })
}
/// `fit` on the members of a configured group, reported with all levels.
pub fn fit_group(
    group: &ChannelGroup,
    members: &[Estimated],
    analysis: &AnalysisConfig,
) -> Result<GroupFit> {
    group.validate()?;
    analysis.multi_exponential.validate()?;
    analysis.window_scan.validate()?;
    require(
        members.len() >= 2,
        "a channel group is fitted with at least two measured members",
    )?;
    let mut fit = gaps(members, &analysis.multi_exponential, analysis)?;
    let listed = fit.converged && fit.diagnostics.model_rejected.is_none();
    // A group has no `mass` to withhold, so the reason a listed level is not a
    // measurement has to be said.
    if let Some(reason) = fit.diagnostics.no_signal.clone() {
        fit.notes
            .push(format!("levels are not measurements: {reason}"));
    }
    Ok(GroupFit {
        id: group.id.clone(),
        channels: members.iter().map(|m| m.channel.clone()).collect(),
        availability: Availability::Available,
        levels: if listed { fit.levels } else { vec![] },
        diagnostics: fit.diagnostics,
        notes: fit.notes,
    })
}
