//! Fit stability against start time, number of exponentials and SVD cut. The
//! scan tabulates the variations and their model weights; it never states a
//! rate of its own.
use super::multi_exponential::gaps;
use crate::{
    Result,
    error::require,
    physics::spectroscopy::{
        config::{AnalysisConfig, MultiExponentialConfig, StabilityScan},
        estimators::Estimated,
        report::{FitDiagnostics, FitMethodKind, FitOutcome, WindowFit},
    },
};
/// Variations one scan may run. Each is a nonlinear fit with an
/// eigendecomposition of its own window, so the product of the scanned ranges
/// is bounded, not the ranges themselves.
const MAX_VARIATIONS: usize = 4096;
/// One scanned variation, before its AIC weight is known.
struct Variation {
    /// Position of the SVD cut in the scan, which the weights are normalized
    /// within: different cuts are different likelihoods.
    cut_index: usize,
    cut: f64,
    t_min: usize,
    t_max: usize,
    nexp: usize,
    points: usize,
    parameters: usize,
    value: f64,
    error: f64,
    /// Augmented χ² and its degrees of freedom.
    chi2: f64,
    dof: usize,
    converged: bool,
    q: Option<f64>,
    width_ratio: f64,
    /// Converged, not prior dominated, above the rate S/N and not rejected.
    accepted: bool,
}
/// An outcome with `method = FitMethodKind::Stability`, `mass = None` and one
/// `WindowFit` row per scanned variation in `windows`, told apart by `t_min`,
/// `nexp` and `svd_cut`; `notes` state whether the rate is stable within errors.
pub fn stability_scan(
    data: &Estimated,
    scan: &StabilityScan,
    analysis: &AnalysisConfig,
) -> Result<FitOutcome> {
    scan.validate()?;
    analysis.multi_exponential.validate()?;
    analysis.window_scan.validate()?;
    let members = std::slice::from_ref(data);
    // A start time beyond the measured lags fits nothing, so the data bound
    // the scan however wide the configured range is.
    let last = data.estimate.lags.last().copied().unwrap_or(0);
    let cuts: Vec<f64> = if scan.svd_cuts.is_empty() {
        vec![analysis.svd_cut]
    } else {
        scan.svd_cuts.clone()
    };
    let starts = (scan.t_min[0]..=scan.t_min[1].min(last)).count();
    require(
        starts * scan.nexp_max * cuts.len() <= MAX_VARIATIONS,
        format!(
            "a stability scan runs at most {MAX_VARIATIONS} variations; narrow t_min, nexp_max \
             or svd_cuts"
        ),
    )?;
    let reference = gaps(members, &analysis.multi_exponential, analysis)?;
    let mut variations: Vec<Variation> = vec![];
    for (cut_index, &cut) in cuts.iter().enumerate() {
        let varied = AnalysisConfig {
            svd_cut: cut,
            ..analysis.clone()
        };
        for nexp in 1..=scan.nexp_max {
            for t_min in scan.t_min[0]..=scan.t_min[1].min(last) {
                let config = MultiExponentialConfig {
                    nexp,
                    t_min,
                    ..analysis.multi_exponential
                };
                if config.validate().is_err() {
                    continue;
                }
                let Ok(row) = gaps(members, &config, &varied) else {
                    continue;
                };
                let (Some(level), Some(chi2), Some(dof), Some(prior), Some(window)) = (
                    row.levels.first(),
                    row.diagnostics.chi2,
                    row.diagnostics.dof,
                    row.diagnostics.prior_dominance,
                    row.diagnostics.window,
                ) else {
                    continue;
                };
                variations.push(Variation {
                    cut_index,
                    cut,
                    t_min,
                    t_max: window[1],
                    nexp,
                    points: row.points,
                    parameters: row.parameters,
                    value: level.value,
                    error: level.error,
                    chi2,
                    dof,
                    converged: row.converged,
                    q: row.diagnostics.q,
                    width_ratio: prior.width_ratio,
                    accepted: row.reported(),
                });
            }
        }
    }
    // The widest window is the one that starts earliest, and it is the same
    // for every number of exponentials.
    let reference_points = variations.iter().map(|v| v.points).max().unwrap_or(0);
    let aic: Vec<f64> = variations
        .iter()
        .map(|v| {
            let cut = reference_points.saturating_sub(v.points);
            v.chi2 + 2. * (v.parameters + cut) as f64
        })
        .collect();
    let mut weights = vec![0.; variations.len()];
    for index in 0..cuts.len() {
        let group: Vec<usize> = (0..variations.len())
            .filter(|&i| variations[i].cut_index == index)
            .collect();
        let best = group.iter().map(|&i| aic[i]).fold(f64::INFINITY, f64::min);
        let likelihood = |i: usize| (-0.5 * (aic[i] - best)).exp();
        let total: f64 = group.iter().map(|&i| likelihood(i)).sum();
        for &i in &group {
            weights[i] = if total > 0. {
                likelihood(i) / total
            } else {
                0.
            };
        }
    }
    let windows: Vec<WindowFit> = variations
        .iter()
        .zip(weights.iter().zip(&aic))
        .map(|(v, (&weight, &aic))| WindowFit {
            t_min: v.t_min,
            t_max: v.t_max,
            value: v.value,
            error: v.error,
            chi2: v.chi2,
            dof: v.dof,
            weight,
            aic,
            nexp: v.nexp,
            svd_cut: Some(v.cut),
        })
        .collect();
    let mut notes = reference.notes;
    for (index, v) in variations.iter().enumerate() {
        notes.push(format!(
            "row {index}: converged={} q={} width_ratio={:.4}",
            v.converged,
            v.q.map_or_else(|| "none".to_string(), |q| format!("{q:.3e}")),
            v.width_ratio
        ));
    }
    let accepted: Vec<&Variation> = variations.iter().filter(|v| v.accepted).collect();
    match reference.levels.first() {
        Some(rate) => {
            let agrees = |v: &Variation| (v.value - rate.value).abs() <= v.error.max(rate.error);
            let disagreeing: Vec<String> = accepted
                .iter()
                .filter(|v| !agrees(v))
                .map(|v| format!("t_min={} nexp={} svd_cut={:e}", v.t_min, v.nexp, v.cut))
                .collect();
            notes.push(format!(
                "stable: {} of {} accepted variations agree with the reference; disagreeing: {}",
                accepted.len() - disagreeing.len(),
                accepted.len(),
                if disagreeing.is_empty() {
                    "none".to_string()
                } else {
                    disagreeing.join(", ")
                }
            ));
        }
        None => notes
            .push("the configured fit states no rate, so the variations have no reference".into()),
    }
    Ok(FitOutcome {
        method: FitMethodKind::Stability,
        mass: None,
        excited: vec![],
        diagnostics: FitDiagnostics {
            n_windows: windows.len(),
            ..reference.diagnostics
        },
        windows,
        notes,
    })
}
