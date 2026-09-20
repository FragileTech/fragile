//! Generalized eigenvalue problem `C(t) v = λ(t, t0) C(t0) v` on a channel
//! basis. Resamples are drawn over origin blocks of the cross-lag products and
//! the problem is solved again in every resample; the symmetrization
//! `(C + Cᵀ)/2` assumes reversibility, so the norm of the discarded
//! antisymmetric part is reported. A state is followed across lags by the
//! eigenvector it was solved with at the reference lag, or, under
//! `GevpProjection::MaxEigenvalue`, by its place in the ordering of each lag.
use super::window_scan::window_scan;
use crate::{
    GasError, Result,
    error::require,
    memory::{DEFAULT_MEMORY_BYTES, checked_mul, enforce},
    physics::{
        numerics::{
            BlockSize, Samples, Subtraction, Whitener, errors, generalized_symmetric, resample,
            sample_covariance, submatrix, tau_int_of_correlator,
        },
        spectroscopy::{
            config::{AnalysisConfig, GevpBasis, GevpProjection},
            contract::Availability,
            estimators::{Estimated, MatrixCorrelator},
            report::{
                CorrelatorEstimate, EstimatorKind, FitMethodKind, FitOutcome, GevpReport,
                SamplesMeta,
            },
        },
    },
};

/// Eigenvalues are sorted at every lag on their own, which is a labelling, not
/// a spectral identification. `max_n λ_n(t)` is the maximum of a noisy
/// spectrum, so it sits above the exact largest eigenvalue by an amount that
/// grows as the gaps close: a one-signed bias driven by the noise, not a
/// second convention for the same number.
const ORDERING_NOTE: &str = "states are labelled by the descending eigenvalue at each lag; \
                             eigenvectors are not tracked from lag to lag, so the largest \
                             eigenvalue carries a one-signed noise bias growing with the lag";
const RANK_NOTE: &str =
    "rank of C(t0) unstable under resampling: raise the cut or drop an operator";
const FAILED_NOTE: &str = "the generalized eigenproblem did not solve in every resample";

/// Eigenvalues `λ_n(t, t0)` with errors, effective rates and one window-scan
/// `FitOutcome` per state (with its own diagnostics), keeping the directions
/// of `C(t0)` above `basis.cut`. Every entry of the matrix is resampled with
/// one autocorrelation time, so all entries share blocks and draws.
/// `antisymmetric_norm` is reported per lag.
pub fn gevp(
    matrix: &MatrixCorrelator,
    basis: &GevpBasis,
    analysis: &AnalysisConfig,
) -> Result<GevpReport> {
    basis.validate()?;
    let n = matrix.channels.len();
    let measured = matrix.lags.len();
    require(
        (2..=16).contains(&n)
            && (1..=4097).contains(&measured)
            && matrix.lags.windows(2).all(|w| w[0] < w[1])
            && matrix.moments.len() == n * n
            && matrix.moments.iter().all(|m| m.lags == measured)
            && matrix.time_step.is_finite()
            && matrix.time_step > 0.,
        "a GEVP needs 2..=16 channels, 1..=4097 strictly increasing lags, one moment table of \
         them per ordered pair and a positive time step",
    )?;
    let Some(start) = matrix.lags.iter().position(|&t| t == basis.t0) else {
        return Ok(declined(
            matrix,
            basis,
            "the reference lag t0 is not measured",
        ));
    };
    let lags: Vec<usize> = matrix.lags[start..].to_vec();
    let count = lags.len();
    let subtraction = if analysis.connected {
        Subtraction::LagMeans
    } else {
        Subtraction::None
    };
    // One autocorrelation time for the whole matrix, so that an automatic
    // block is the same for every entry and the bootstrap draws one index
    // stream; an entry left to itself would read the meaningless `ρ` of an
    // off-diagonal pair.
    let tau = matrix.tau_int.or_else(|| {
        (analysis.resampling.block() == BlockSize::Auto).then(|| diagonal_tau(matrix, n))
    });
    let entries: Vec<Samples> = matrix
        .moments
        .iter()
        .map(|m| resample(m, subtraction, &analysis.resampling, tau))
        .collect::<Result<_>>()?;
    let shared = &entries[0];
    require(
        entries.iter().all(|s| {
            s.dimension == measured && s.count == shared.count && s.blocks == shared.blocks
        }),
        "the entries of a GEVP matrix must share their lags, blocks and draws",
    )?;
    let draws = shared.count;
    // `FixedVector` solves its eigenvectors once, at the requested lag or at
    // the first one beyond t0 whose matrix is defined. Whether an entry is
    // defined comes from the pair weights and not from a draw, so every
    // resample holds vectors from the same lag and its states are the states
    // of the central estimate.
    let requested = match basis.t_ref {
        Some(t) => match lags.iter().position(|&l| l == t) {
            Some(k) => k,
            None => {
                return Ok(declined(
                    matrix,
                    basis,
                    "the reference lag t_ref is not measured",
                ));
            }
        },
        None => 1,
    };
    let reference = (basis.projection == GevpProjection::FixedVector)
        .then(|| (requested..count).find(|&k| symmetrized(&entries, None, n, start + k).is_some()))
        .flatten();
    let Some((states, central)) = spectrum(&entries, None, n, start, count, basis, reference)?
    else {
        return Ok(declined(
            matrix,
            basis,
            "the correlator matrix at t0 has an undefined entry",
        ));
    };
    if states == 0 {
        return Ok(declined(
            matrix,
            basis,
            "no direction of C(t0) survives the cut",
        ));
    }
    // `[state, draw, lag]`, which is `[draw, lag]` per state: one resample
    // table each, as `Samples` lays it out.
    enforce(
        checked_mul(checked_mul(checked_mul(states, draws)?, count)?, 8)?,
        DEFAULT_MEMORY_BYTES,
    )?;
    let mut values = vec![0.; states * draws * count];
    let mut defined: Vec<bool> = (0..states * count)
        .map(|c| central[(c % count) * states + c / count].is_some())
        .collect();
    let (mut unstable, mut failed) = (false, false);
    for r in 0..draws {
        let table = match spectrum(&entries, Some(r), n, start, count, basis, reference) {
            Ok(Some((rank, table))) => {
                unstable |= rank != states;
                Some((rank, table))
            }
            Ok(None) => None,
            Err(GasError::Numerical(_)) => {
                failed = true;
                None
            }
            Err(error) => return Err(error),
        };
        for s in 0..states {
            for k in 0..count {
                let value = table
                    .as_ref()
                    .filter(|(rank, _)| s < *rank)
                    .and_then(|(rank, table)| table[k * rank + s]);
                match value {
                    Some(x) => values[(s * draws + r) * count + k] = x,
                    None => defined[s * count + k] = false,
                }
            }
        }
    }
    for (s, k) in (0..states).flat_map(|s| (0..count).map(move |k| (s, k))) {
        for r in (0..draws).filter(|_| !defined[s * count + k]) {
            values[(s * draws + r) * count + k] = 0.;
        }
    }
    // The eigenvalue at t0 is exactly one with no variance and must not enter
    // a fit, so every level is scanned from beyond t0.
    let mut scanned = analysis.clone();
    scanned.window_scan.t_min = analysis.window_scan.t_min.max(basis.t0 + 1);
    let mut eigenvalues = Vec::with_capacity(states);
    let mut effective_mass = Vec::with_capacity(states);
    let mut levels = Vec::with_capacity(states);
    for s in 0..states {
        // The resample table keeps a nonpositive eigenvalue as estimated, so
        // that a fit can reject the exponential model on it; the reported
        // tables leave it out, where no rate is defined.
        let table = Samples {
            kind: shared.kind,
            dimension: count,
            count: draws,
            central: (0..count)
                .map(|k| {
                    central[k * states + s]
                        .filter(|_| defined[s * count + k])
                        .unwrap_or(0.)
                })
                .collect(),
            values: values[s * draws * count..(s + 1) * draws * count].to_vec(),
            defined: defined[s * count..(s + 1) * count].to_vec(),
            effective_block: shared.effective_block,
            blocks: shared.blocks,
            tau_int: shared.tau_int,
        };
        table.validate()?;
        let error = errors(&table);
        eigenvalues.push(
            (0..count)
                .map(|k| {
                    let value = table.central[k];
                    Some([value, error[k].filter(|_| value > 0.)?])
                })
                .collect(),
        );
        let effective = rates(&table, &lags, matrix.time_step);
        let rate_error = errors(&effective);
        effective_mass.push(
            (0..count)
                .map(|k| Some([effective.central[k], rate_error[k]?]))
                .collect(),
        );
        levels.push(level(&table, &lags, matrix, basis, &scanned, s)?);
    }
    let antisymmetric_norm: Vec<Option<f64>> = (0..count)
        .map(|k| antisymmetric_norm(&entries, n, start + k))
        .collect();
    let mut notes = vec![match (basis.projection, reference) {
        (GevpProjection::MaxEigenvalue, _) => ORDERING_NOTE.into(),
        (GevpProjection::FixedVector, Some(k)) => format!(
            "states are labelled by the eigenvectors of the pencil at lag {}, held across lags \
             and resamples",
            lags[k]
        ),
        (GevpProjection::FixedVector, None) => "no lag beyond t0 has a defined correlator \
                                                matrix: the states are the directions of C(t0) \
                                                itself"
            .into(),
    }];
    if let Some((lag, value)) = lags
        .iter()
        .zip(&antisymmetric_norm)
        .filter_map(|(t, v)| v.map(|v| (*t, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
    {
        notes.push(format!(
            "largest antisymmetric norm {value:.3e} at lag {lag}; symmetrization assumes a \
             reversible transfer"
        ));
    }
    if unstable {
        notes.push(RANK_NOTE.into());
    }
    if failed {
        notes.push(FAILED_NOTE.into());
    }
    Ok(GevpReport {
        id: basis.id.clone(),
        channels: matrix.channels.clone(),
        availability: Availability::Available,
        t0: basis.t0,
        lags,
        eigenvalues,
        effective_mass,
        levels,
        rank: states,
        antisymmetric_norm,
        notes,
    })
}

/// Largest integrated autocorrelation time of the diagonal members, as
/// `MatrixCorrelator::tau_int` defines it, falling back to the shortest time a
/// block may assume when no member has one.
fn diagonal_tau(matrix: &MatrixCorrelator, n: usize) -> f64 {
    (0..n)
        .filter_map(|a| {
            let moments = &matrix.moments[a * n + a];
            tau_int_of_correlator(&moments.estimate(Subtraction::LagMeans), moments.origins())
        })
        .fold(0.5, |best: f64, t| best.max(t.tau))
}

/// An empty report of a basis that could not be solved.
fn declined(matrix: &MatrixCorrelator, basis: &GevpBasis, reason: &str) -> GevpReport {
    GevpReport {
        id: basis.id.clone(),
        channels: matrix.channels.clone(),
        availability: Availability::unavailable(reason),
        t0: basis.t0,
        ..GevpReport::default()
    }
}

/// Entry `(a, b) = index` of the correlator matrix at one measured lag, from
/// the central estimate (`draw` is `None`) or from one resample.
fn entry(entries: &[Samples], draw: Option<usize>, index: usize, lag: usize) -> Option<f64> {
    let samples = &entries[index];
    samples.defined[lag].then(|| match draw {
        Some(r) => samples.sample(r)[lag],
        None => samples.central[lag],
    })
}

/// `(C + Cᵀ)/2` at one measured lag, `None` when an entry is undefined.
fn symmetrized(entries: &[Samples], draw: Option<usize>, n: usize, lag: usize) -> Option<Vec<f64>> {
    let mut out = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let x = entry(entries, draw, i * n + j, lag)?;
            let y = entry(entries, draw, j * n + i, lag)?;
            out[i * n + j] = 0.5 * (x + y);
            out[j * n + i] = out[i * n + j];
        }
    }
    Some(out)
}

/// `‖C − Cᵀ‖_F / ‖C + Cᵀ‖_F` of the central estimate at one measured lag:
/// what the symmetrization discarded. `None` for an undefined entry or a
/// vanishing symmetric part.
fn antisymmetric_norm(entries: &[Samples], n: usize, lag: usize) -> Option<f64> {
    let (mut odd, mut even) = (0., 0.);
    for i in 0..n {
        for j in 0..n {
            let x = entry(entries, None, i * n + j, lag)?;
            let y = entry(entries, None, j * n + i, lag)?;
            odd += (x - y) * (x - y);
            even += (x + y) * (x + y);
        }
    }
    (even > 0.)
        .then(|| (odd / even).sqrt())
        .filter(|x: &f64| x.is_finite())
}

/// Eigenvalues of `S(t) v = λ S(t0) v` over the `count` lags from `start`,
/// `[lags, rank]` row major with `None` at a lag holding an undefined entry.
/// Under `GevpProjection::FixedVector` row `s` is one state, read through the
/// eigenvector solved at the `reference` offset and held; otherwise each lag
/// is solved on its own and row `s` is its `s`-th largest eigenvalue. `None`
/// when the metric `S(t0)` itself is undefined.
fn spectrum(
    entries: &[Samples],
    draw: Option<usize>,
    n: usize,
    start: usize,
    count: usize,
    basis: &GevpBasis,
    reference: Option<usize>,
) -> Result<Option<(usize, Vec<Option<f64>>)>> {
    let Some(metric) = symmetrized(entries, draw, n, start) else {
        return Ok(None);
    };
    // Lag `t0` is the metric against itself, so the same solve both fixes the
    // rank and gives the eigenvalues there. With nothing to solve the held
    // vectors at, they are the directions of the metric.
    let unit = generalized_symmetric(&metric, &metric, n, basis.cut)?;
    let rank = unit.rank;
    let held = (basis.projection == GevpProjection::FixedVector)
        .then(
            || match reference.and_then(|k| symmetrized(entries, draw, n, start + k)) {
                Some(a) => generalized_symmetric(&a, &metric, n, basis.cut),
                None => Ok(unit.clone()),
            },
        )
        .transpose()?;
    let mut table = vec![None; count * rank];
    for (s, value) in unit.values.iter().enumerate() {
        table[s] = Some(*value);
    }
    for k in 1..count {
        let Some(a) = symmetrized(entries, draw, n, start + k) else {
            continue;
        };
        let values = match &held {
            Some(solved) => solved.project(&a, n),
            None => generalized_symmetric(&a, &metric, n, basis.cut)?.values,
        };
        for (s, value) in values.iter().take(rank).enumerate() {
            table[k * rank + s] = Some(*value);
        }
    }
    Ok(Some((rank, table)))
}

/// Effective rates `ln(λ(t_k) / λ(t_{k+1})) / (t_{k+1} − t_k)` of one state,
/// resample by resample, undefined at the last lag and wherever the ratio of
/// two positive eigenvalues does not exist.
fn rates(table: &Samples, lags: &[usize], time_step: f64) -> Samples {
    let count = table.dimension;
    let rate = |x: f64, y: f64, k: usize| {
        let step = (lags[k + 1] - lags[k]) as f64 * time_step;
        (x > 0. && y > 0. && step > 0.).then(|| (x / y).ln() / step)
    };
    let mut defined: Vec<bool> = (0..count)
        .map(|k| {
            k + 1 < count
                && table.defined[k]
                && table.defined[k + 1]
                && rate(table.central[k], table.central[k + 1], k).is_some()
        })
        .collect();
    let mut values = vec![0.; table.count * count];
    for r in 0..table.count {
        let sample = table.sample(r);
        for k in 0..count {
            if !defined[k] {
                continue;
            }
            match rate(sample[k], sample[k + 1], k) {
                Some(x) => values[r * count + k] = x,
                None => defined[k] = false,
            }
        }
    }
    for row in values.chunks_exact_mut(count) {
        for (x, keep) in row.iter_mut().zip(&defined) {
            if !keep {
                *x = 0.;
            }
        }
    }
    Samples {
        kind: table.kind,
        dimension: count,
        count: table.count,
        central: (0..count)
            .map(|k| {
                defined[k]
                    .then(|| rate(table.central[k], table.central[k + 1], k))
                    .flatten()
                    .unwrap_or(0.)
            })
            .collect(),
        values,
        defined,
        effective_block: table.effective_block,
        blocks: table.blocks,
        tau_int: table.tau_int,
    }
}

/// A level with no rate, carrying the reason it was not fitted.
fn unfitted(reason: &str) -> FitOutcome {
    FitOutcome {
        method: FitMethodKind::Gevp,
        notes: vec![format!("no fit of this level: {reason}")],
        ..FitOutcome::default()
    }
}

/// Window scan of one principal correlator over the lags beyond `t0`, where
/// the eigenvalue is exactly one with no variance. `analysis` already carries
/// the raised `window_scan.t_min`; a fit that declines its input leaves the
/// level without a rate and says why.
fn level(
    table: &Samples,
    lags: &[usize],
    matrix: &MatrixCorrelator,
    basis: &GevpBasis,
    analysis: &AnalysisConfig,
    state: usize,
) -> Result<FitOutcome> {
    let selected: Vec<usize> = (0..lags.len()).filter(|&k| lags[k] > basis.t0).collect();
    if selected.is_empty() {
        return Ok(unfitted("no lag beyond t0 was measured"));
    }
    // Starting beyond t0 can leave no window of the requested length inside
    // the configured last lag, which is a declined level, not a bad request.
    if analysis.window_scan.validate().is_err() {
        return Ok(unfitted(
            "no window beyond t0 fits inside the requested lag range",
        ));
    }
    let samples = table.select(&selected)?;
    let covariance = sample_covariance(&samples);
    let kept: Vec<usize> = (0..samples.dimension)
        .filter(|&k| samples.defined[k])
        .collect();
    let error = errors(&samples);
    let data = Estimated {
        channel: format!("{}/{state}", basis.id),
        kind: EstimatorKind::FrameMean,
        estimate: CorrelatorEstimate {
            lags: selected.iter().map(|&k| lags[k]).collect(),
            time_unit: matrix.time_unit,
            time_step: matrix.time_step,
            value: (0..samples.dimension)
                .map(|k| samples.defined[k].then(|| samples.central[k]))
                .collect(),
            error,
            covariance: Some(covariance.clone()),
            samples_meta: SamplesMeta {
                resampling: samples.kind,
                effective_block: samples.effective_block,
                blocks: samples.blocks,
                tau_int: samples.tau_int,
                covariance_rank: Whitener::new(
                    &submatrix(&covariance, samples.dimension, &kept),
                    kept.len(),
                    analysis.svd_cut,
                )
                .map_or(0, |w| w.rank()),
                replicas: matrix.replicas,
                sampling_unit: SamplesMeta::sampling_unit(
                    analysis.combine,
                    samples.effective_block,
                    matrix.replicas,
                ),
            },
            connected: analysis.connected,
            connected_bias: None,
        },
        samples,
    };
    let mut outcome = match window_scan(&data, analysis) {
        Ok(outcome) => outcome,
        Err(GasError::Capability(reason) | GasError::Numerical(reason)) => unfitted(&reason),
        Err(error) => return Err(error),
    };
    outcome.method = FitMethodKind::Gevp;
    for level in outcome.mass.iter_mut().chain(&mut outcome.excited) {
        level.method = FitMethodKind::Gevp;
    }
    Ok(outcome)
}
