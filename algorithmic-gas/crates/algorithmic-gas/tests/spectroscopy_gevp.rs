use algorithmic_gas::{
    GasError,
    physics::{
        numerics::{
            BlockMoments, BlockSize, Resampling, SeriesView, Subtraction, cross_moments, resample,
            series_moments, tau_int_of_correlator,
        },
        qft::math::Rng,
        spectroscopy::{
            AnalysisConfig, Availability,
            config::{ChannelGroup, GevpBasis, GevpProjection, TimeUnit},
            estimators::MatrixCorrelator,
            fits::gevp::gevp,
            report::FitMethodKind,
        },
    },
};

/// Three operators overlapping two decaying states: `C(t) = Z e^{−E t} Zᵀ`.
const Z: [f64; 6] = [1., 0.5, 0.4, 1., 0.7, -0.3];
const RATES: [f64; 2] = [0.25, 0.7];
/// `Σ (Aᵀ)^τ` of the VAR(1) process `A = [[0.8, 0.3], [0, 0.5]]` with unit
/// noise, whose correlator is genuinely non-symmetric.
const VAR1: [[f64; 4]; 3] = [
    [
        3.55555555555556,
        0.333333333333333,
        0.333333333333333,
        1.33333333333333,
    ],
    [
        2.94444444444445,
        0.166666666666667,
        0.666666666666667,
        0.666666666666667,
    ],
    [
        2.40555555555556,
        0.0833333333333333,
        0.733333333333333,
        0.333333333333333,
    ],
];

fn close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1.),
        "{actual} vs {expected}"
    );
}
fn ar1(seed: u64, frames: usize, phi: f64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut x = vec![rng.normal() / (1. - phi * phi).sqrt()];
    for t in 1..frames {
        x.push(phi * x[t - 1] + rng.normal());
    }
    x
}
fn view<'a>(values: &'a [f64], weight: &'a [f64], segment: &'a [u32]) -> SeriesView<'a> {
    SeriesView {
        values,
        weight,
        segment,
        components: 1,
    }
}
/// Blocked moments whose block `k` holds `scale[k] · values`, as eight
/// single-origin blocks a delete-one jackknife can pull apart. The pooled raw
/// estimate is `mean(scale) · values`.
fn blocked(values: &[f64], scale: &[f64; 8]) -> BlockMoments {
    let lags = values.len();
    let mut moments = BlockMoments::new(lags, 1, 1, 64).unwrap();
    let (zero, one) = (vec![0.; lags], vec![1.; lags]);
    for s in scale {
        let ab: Vec<f64> = values.iter().map(|v| s * v).collect();
        moments.push_origin(&ab, &zero, &zero, &one).unwrap();
    }
    moments
}
/// Blocked moments whose pooled raw estimate is exactly `values` per lag, as
/// eight identical single-origin blocks: every resample reproduces the point
/// estimate and the errors vanish.
fn exact(values: &[f64]) -> BlockMoments {
    blocked(values, &[1.; 8])
}
/// The same moments with no pair weight at one lag: the correlator there has
/// a zero denominator and no estimate.
fn blind(values: &[f64], lag: usize) -> BlockMoments {
    let lags = values.len();
    let mut moments = BlockMoments::new(lags, 1, 1, 64).unwrap();
    let (zero, mut one) = (vec![0.; lags], vec![1.; lags]);
    let mut ab = values.to_vec();
    (one[lag], ab[lag]) = (0., 0.);
    for _ in 0..8 {
        moments.push_origin(&ab, &zero, &zero, &one).unwrap();
    }
    moments
}
/// A matrix correlator of `n` channels from `[n * n, lags]` exact entries.
fn correlator(entries: &[f64], n: usize, lags: usize) -> MatrixCorrelator {
    MatrixCorrelator {
        channels: (0..n).map(|a| format!("channel/{a}")).collect(),
        lags: (0..lags).collect(),
        time_unit: TimeUnit::Frames,
        time_step: 1.,
        frames: 8,
        tau_int: None,
        moments: entries.chunks_exact(lags).map(exact).collect(),
        replicas: 1,
    }
}
/// `C(t) = Z e^{−E t} Zᵀ` of the two-state system, `[9, lags]`.
fn mixing(lags: usize, mix: &[f64; 9]) -> Vec<f64> {
    let mut out = vec![0.; 9 * lags];
    for t in 0..lags {
        for i in 0..3 {
            for j in 0..3 {
                let c: f64 = (0..2)
                    .map(|n| Z[i * 2 + n] * Z[j * 2 + n] * (-RATES[n] * t as f64).exp())
                    .sum();
                // The basis change `O → M O` maps `C` to `M C Mᵀ`.
                for (a, b) in (0..3).flat_map(|a| (0..3).map(move |b| (a, b))) {
                    out[(a * 3 + b) * lags + t] += mix[a * 3 + i] * c * mix[b * 3 + j];
                }
            }
        }
    }
    out
}
/// `C(t) = Z e^{−E t} Zᵀ` of the same two-state system at arbitrary measured
/// times, `[9, times.len()]`.
fn sampled(times: &[usize]) -> Vec<f64> {
    let count = times.len();
    let mut out = vec![0.; 9 * count];
    for (k, &t) in times.iter().enumerate() {
        for (i, j) in (0..3).flat_map(|i| (0..3).map(move |j| (i, j))) {
            out[(i * 3 + j) * count + k] = (0..2)
                .map(|n| Z[i * 2 + n] * Z[j * 2 + n] * (-RATES[n] * t as f64).exp())
                .sum();
        }
    }
    out
}
/// A matrix correlator over an arbitrary lag list and time step.
fn measured(entries: &[f64], n: usize, lags: &[usize], time_step: f64) -> MatrixCorrelator {
    MatrixCorrelator {
        channels: (0..n).map(|a| format!("channel/{a}")).collect(),
        lags: lags.to_vec(),
        time_unit: TimeUnit::StepDt,
        time_step,
        frames: 8,
        tau_int: None,
        moments: entries.chunks_exact(lags.len()).map(exact).collect(),
        replicas: 1,
    }
}
fn basis(t0: usize, channels: usize) -> GevpBasis {
    GevpBasis {
        id: "mixing".into(),
        channels: (0..channels).map(|a| format!("channel/{a}")).collect(),
        t0,
        cut: 1e-3,
        ..GevpBasis::default()
    }
}
/// The same basis reading each lag's own ordering instead of holding the
/// eigenvectors of the reference lag.
fn untracked(t0: usize, channels: usize) -> GevpBasis {
    GevpBasis {
        projection: GevpProjection::MaxEigenvalue,
        ..basis(t0, channels)
    }
}
fn analysis(connected: bool, block: usize) -> AnalysisConfig {
    AnalysisConfig {
        connected,
        resampling: Resampling::BlockJackknife {
            block: BlockSize::Fixed { frames: block },
        },
        ..AnalysisConfig::default()
    }
}
fn value(entry: Option<[f64; 2]>) -> f64 {
    entry.unwrap()[0]
}
fn error(entry: Option<[f64; 2]>) -> f64 {
    entry.unwrap()[1]
}

#[test]
fn a_two_state_mixing_system_has_exactly_exponential_generalized_eigenvalues() {
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let matrix = correlator(&mixing(7, &identity), 3, 7);
    // The constructed moments hold the closed form exactly.
    let expected = [
        0.902947109019257,
        0.559812965124267,
        0.470672752581272,
        0.559812965124267,
        0.621193429082834,
        0.0690886281225705,
        0.470672752581272,
        0.0690886281225705,
        0.426305061046215,
    ];
    for (e, expected) in expected.iter().enumerate() {
        close(
            matrix.moments[e].estimate(Subtraction::None)[1].unwrap(),
            *expected,
            1e-14,
        );
    }
    let report = gevp(&matrix, &basis(1, 3), &analysis(false, 1)).unwrap();
    assert_eq!(report.availability, Availability::Available);
    assert_eq!(report.rank, 2);
    assert_eq!(report.t0, 1);
    assert_eq!(report.lags, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(report.eigenvalues.len(), 2);
    assert_eq!(report.levels.len(), 2);
    for (n, rate) in RATES.iter().enumerate() {
        for (k, lag) in report.lags.iter().enumerate() {
            close(
                value(report.eigenvalues[n][k]),
                (-rate * (lag - 1) as f64).exp(),
                1e-10,
            );
            assert!(error(report.eigenvalues[n][k]) < 1e-10);
            if k + 1 < report.lags.len() {
                close(value(report.effective_mass[n][k]), *rate, 1e-10);
            }
        }
        assert_eq!(report.effective_mass[n][report.lags.len() - 1], None);
    }
    // A symmetric correlator discards nothing.
    assert_eq!(report.antisymmetric_norm, vec![Some(0.); 6]);
    // With no noise the eigenvectors of the pencil are the same at every lag,
    // so holding those of the reference lag and reading each lag's own
    // ordering are the same numbers: the two arms differ on noise alone.
    let ordered = gevp(&matrix, &untracked(1, 3), &analysis(false, 1)).unwrap();
    for n in 0..2 {
        for k in 0..report.lags.len() {
            close(
                value(ordered.eigenvalues[n][k]),
                value(report.eigenvalues[n][k]),
                1e-10,
            );
        }
    }
    assert!(
        report.notes[0].contains("eigenvectors of the pencil at lag 2"),
        "{:?}",
        report.notes
    );
    assert!(
        ordered.notes[0].contains("one-signed noise bias"),
        "{:?}",
        ordered.notes
    );
}

#[test]
fn the_effective_rate_divides_by_the_measured_lag_gap_and_the_time_step() {
    // Unequal gaps of 2, 3 and 4 frames: a rate that divides by the wrong gap,
    // or by a constant one, cannot be right at all three.
    let times = [0, 1, 3, 6, 10];
    let entries = sampled(&times);
    let report = gevp(
        &measured(&entries, 3, &times, 0.5),
        &basis(1, 3),
        &analysis(false, 1),
    )
    .unwrap();
    assert_eq!(report.rank, 2);
    assert_eq!(report.lags, vec![1, 3, 6, 10]);
    for (n, rate) in RATES.iter().enumerate() {
        for (k, lag) in report.lags.iter().enumerate() {
            close(
                value(report.eigenvalues[n][k]),
                (-rate * (lag - 1) as f64).exp(),
                1e-10,
            );
            if k + 1 < report.lags.len() {
                // ln(λ_k/λ_{k+1}) = E (t_{k+1} − t_k) and the gap lasts
                // (t_{k+1} − t_k) · 0.5 time units, so the rate is 2E.
                close(value(report.effective_mass[n][k]), 2. * rate, 1e-10);
            }
        }
        assert_eq!(report.effective_mass[n][3], None);
    }
    // Relabelling the time unit divides every rate by the same factor and
    // leaves the eigenvalues alone (`thm-qft-ratio-rescale`).
    let slower = gevp(
        &measured(&entries, 3, &times, 1.),
        &basis(1, 3),
        &analysis(false, 1),
    )
    .unwrap();
    for n in 0..2 {
        for k in 0..3 {
            close(
                2. * value(slower.effective_mass[n][k]),
                value(report.effective_mass[n][k]),
                1e-12,
            );
        }
        assert_eq!(slower.eigenvalues[n], report.eigenvalues[n]);
    }
}

#[test]
fn an_undefined_entry_and_a_nonpositive_eigenvalue_remove_a_level_and_its_adjacent_rates() {
    // Two uncoupled operators against the unit metric at t0 = 0, so the
    // eigenvalues are the diagonal entries themselves.
    let large = [1., 0.8, 0.64, 0.512, 0.4096, 0.32768];
    let small = [1., 0.5, 0.25, -0.1, 0.0625, 0.03125];
    let off = [0.; 6];
    let matrix = MatrixCorrelator {
        channels: vec!["channel/0".into(), "channel/1".into()],
        lags: (0..6).collect(),
        time_unit: TimeUnit::Frames,
        time_step: 1.,
        frames: 8,
        tau_int: None,
        // One off-diagonal entry has no pair weight at lag 1.
        moments: vec![exact(&large), blind(&off, 1), exact(&off), exact(&small)],
        replicas: 1,
    };
    let report = gevp(&matrix, &basis(0, 2), &analysis(false, 1)).unwrap();
    assert_eq!(report.rank, 2);
    assert_eq!(report.lags, vec![0, 1, 2, 3, 4, 5]);
    // Only the undefined lag is missing from the upper level; the lower one
    // loses the lag where its eigenvalue turns negative as well.
    for (k, expected) in large.iter().enumerate() {
        match k {
            1 => assert_eq!(report.eigenvalues[0][k], None),
            _ => {
                close(value(report.eigenvalues[0][k]), *expected, 1e-12);
                assert!(error(report.eigenvalues[0][k]) < 1e-12);
            }
        }
    }
    for (k, expected) in small.iter().enumerate() {
        match k {
            1 | 3 => assert_eq!(report.eigenvalues[1][k], None),
            _ => close(value(report.eigenvalues[1][k]), *expected, 1e-12),
        }
    }
    // A rate needs two positive eigenvalues at consecutive measured lags: the
    // undefined lag 1 takes the rates at lag 0 and lag 1 with it, and the
    // negative eigenvalue at lag 3 takes those at lag 2 and lag 3.
    for k in 0..6 {
        // ln(1/0.8) wherever the upper level has both ends.
        match k {
            2..=4 => close(value(report.effective_mass[0][k]), 0.223143551314210, 1e-12),
            _ => assert_eq!(report.effective_mass[0][k], None),
        }
        // ln(0.0625/0.03125) = ln 2 on the only surviving pair of the lower one.
        match k {
            4 => close(
                value(report.effective_mass[1][k]),
                std::f64::consts::LN_2,
                1e-12,
            ),
            _ => assert_eq!(report.effective_mass[1][k], None),
        }
    }
    // The discarded antisymmetric part is undefined exactly where an entry is.
    assert_eq!(report.antisymmetric_norm[1], None);
    for k in [0, 2, 3, 4, 5] {
        assert_eq!(report.antisymmetric_norm[k], Some(0.));
    }
}

#[test]
fn a_resample_that_changes_the_rank_or_loses_a_positive_metric_is_reported_in_the_notes() {
    // `C(t) = [[0.8^t, ρ 0.9^t], [ρ 0.9^t, 0.8^t]]`, with ρ carried by the
    // off-diagonal blocks alone, so that only the rank of the metric moves
    // while the two directions keep well separated eigenvalues.
    let diagonal: Vec<f64> = (0..6).map(|t| 0.8_f64.powi(t)).collect();
    let coupling: Vec<f64> = (0..6).map(|t| 0.9_f64.powi(t)).collect();
    let straddle = |scale: [f64; 8]| MatrixCorrelator {
        channels: vec!["channel/0".into(), "channel/1".into()],
        lags: (0..6).collect(),
        time_unit: TimeUnit::Frames,
        time_step: 1.,
        frames: 8,
        tau_int: None,
        moments: vec![
            exact(&diagonal),
            blocked(&coupling, &scale),
            blocked(&coupling, &scale),
            exact(&diagonal),
        ],
        replicas: 1,
    };
    // Central ρ = 0.9985 leaves 1 − ρ below the cut, so C(t0) has one
    // direction; deleting the first block lowers ρ to 0.9885 and a second
    // direction appears that the central rank does not have. Its eigenvalue
    // is −7.8 at lag 1 and falls to −22 at lag 5, so keeping the wrong end of
    // that resample would be unmissable.
    let more = gevp(
        &straddle([
            1.0685, 0.9885, 0.9885, 0.9885, 0.9885, 0.9885, 0.9885, 0.9885,
        ]),
        &basis(0, 2),
        &analysis(false, 1),
    )
    .unwrap();
    assert_eq!(more.rank, 1);
    assert_eq!(more.eigenvalues.len(), 1);
    assert!(
        more.notes
            .iter()
            .any(|n| n.contains("rank of C(t0) unstable")),
        "{:?}",
        more.notes
    );
    assert!(
        !more.notes.iter().any(|n| n.contains("did not solve")),
        "{:?}",
        more.notes
    );
    // `(0.8^t + ρ 0.9^t)/(1 + ρ)` at the central ρ, jackknifed over the eight
    // blocks; the one resample of rank two contributes its larger value.
    let eigenvalue = [
        1.,
        0.84996247185389,
        0.724936202151614,
        0.620418563922942,
        0.53275749311984,
        0.458986372279209,
    ];
    let spread = [
        0.,
        0.000251454793955663,
        0.000427473149724356,
        0.000545656902883704,
        0.000619836067100588,
        0.000660848343994637,
    ];
    let rate = [
        0.162563081232577,
        0.159108543799108,
        0.155689300589383,
        0.152328017497978,
        0.149045816266812,
    ];
    let rate_spread = [
        0.00029587981087047,
        0.00029393954346113,
        0.000290010932334779,
        0.000284200064823988,
        0.000276659932864031,
    ];
    for k in 0..6 {
        close(value(more.eigenvalues[0][k]), eigenvalue[k], 1e-12);
        close(error(more.eigenvalues[0][k]), spread[k], 1e-12);
        if k < 5 {
            close(value(more.effective_mass[0][k]), rate[k], 1e-12);
            close(error(more.effective_mass[0][k]), rate_spread[k], 1e-12);
        }
    }
    assert_eq!(more.effective_mass[0][5], None);
    // Central ρ = 0.9921875 keeps two directions; deleting the first block
    // raises ρ above one and the metric is no longer positive.
    let broken = gevp(
        &straddle([0.5, 1.0625, 1.0625, 1.0625, 1.0625, 1.0625, 1.0625, 1.0625]),
        &basis(0, 2),
        &analysis(false, 1),
    )
    .unwrap();
    assert_eq!(broken.availability, Availability::Available);
    assert_eq!(broken.rank, 2);
    assert!(
        broken
            .notes
            .iter()
            .any(|n| n.contains("did not solve in every resample")),
        "{:?}",
        broken.notes
    );
    assert!(
        !broken.notes.iter().any(|n| n.contains("rank of C(t0)")),
        "{:?}",
        broken.notes
    );
    // A resample that does not solve has no eigenvalue to offer, so no lag of
    // any state carries an error and no level is fitted; the basis stays
    // available and says why.
    assert!(broken.eigenvalues.iter().flatten().all(Option::is_none));
    assert!(broken.effective_mass.iter().flatten().all(Option::is_none));
    for level in &broken.levels {
        assert_eq!(level.method, FitMethodKind::Gevp);
        assert!(level.mass.is_none());
    }
    assert_eq!(broken.antisymmetric_norm, vec![Some(0.); 6]);
}

#[test]
fn a_lag_list_that_does_not_increase_is_rejected_before_a_rate_is_formed() {
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let entries = mixing(7, &identity);
    // `rates` subtracts consecutive lags in `usize`; a descending pair has no
    // gap to divide by.
    let mut swapped = correlator(&entries, 3, 7);
    swapped.lags = vec![0, 1, 3, 2, 4, 5, 6];
    assert!(matches!(
        gevp(&swapped, &basis(1, 3), &analysis(false, 1)),
        Err(GasError::Configuration(_))
    ));
    let mut repeated = correlator(&entries, 3, 7);
    repeated.lags = vec![0, 1, 2, 2, 4, 5, 6];
    assert!(matches!(
        gevp(&repeated, &basis(1, 3), &analysis(false, 1)),
        Err(GasError::Configuration(_))
    ));
    // A moment table that does not cover the lag list is rejected as well.
    let mut short = correlator(&entries, 3, 7);
    short.moments[4] = exact(&entries[..2]);
    assert!(matches!(
        gevp(&short, &basis(1, 3), &analysis(false, 1)),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn a_basis_change_of_the_operators_leaves_the_eigenvalues_invariant() {
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let mix = [1., 0.5, 0., -0.25, 1., 0.75, 0.3, 0., 2.];
    let plain = gevp(
        &correlator(&mixing(7, &identity), 3, 7),
        &basis(1, 3),
        &analysis(false, 1),
    )
    .unwrap();
    let mixed = gevp(
        &correlator(&mixing(7, &mix), 3, 7),
        &basis(1, 3),
        &analysis(false, 1),
    )
    .unwrap();
    assert_eq!(mixed.rank, plain.rank);
    for n in 0..plain.rank {
        for k in 0..plain.lags.len() {
            close(
                value(mixed.eigenvalues[n][k]),
                value(plain.eigenvalues[n][k]),
                1e-10,
            );
        }
    }
}

#[test]
fn duplicated_operators_leave_one_direction_above_the_cut() {
    let single: Vec<f64> = (0..4)
        .flat_map(|_| (0..5).map(|t| 0.8 * (-0.3 * t as f64).exp()))
        .collect();
    let report = gevp(
        &correlator(&single, 2, 5),
        &basis(1, 2),
        &analysis(false, 1),
    )
    .unwrap();
    assert_eq!(report.rank, 1);
    assert_eq!(report.eigenvalues.len(), 1);
    for (k, lag) in report.lags.iter().enumerate() {
        close(
            value(report.eigenvalues[0][k]),
            (-0.3 * (lag - 1) as f64).exp(),
            1e-10,
        );
    }
}

#[test]
fn the_reported_antisymmetric_norm_is_what_symmetrization_discards_per_lag() {
    let entries: Vec<f64> = (0..4)
        .flat_map(|e| VAR1.iter().map(move |c| c[e]))
        .collect();
    let report = gevp(
        &correlator(&entries, 2, 3),
        &basis(0, 2),
        &analysis(false, 1),
    )
    .unwrap();
    assert_eq!(report.lags, vec![0, 1, 2]);
    assert_eq!(report.antisymmetric_norm[0], Some(0.));
    close(
        report.antisymmetric_norm[1].unwrap(),
        0.114941497604098,
        1e-12,
    );
    close(
        report.antisymmetric_norm[2].unwrap(),
        0.184123700800824,
        1e-12,
    );
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.contains("1.841e-1") && n.contains("lag 2")),
        "{:?}",
        report.notes
    );
}

#[test]
fn block_resampling_of_the_cross_lag_products_keeps_the_autocorrelation_a_frame_shuffle_destroys() {
    let x = ar1(11, 2000, 0.9);
    let (weight, segment) = (vec![1.; 2000], vec![0u32; 2000]);
    let moments = series_moments(view(&x, &weight, &segment), 1, 50).unwrap();
    let central = moments.estimate(Subtraction::LagMeans);
    let ratio = central[1].unwrap() / central[0].unwrap();
    close(ratio, 0.881556411342408, 1e-9);
    // Origin blocks keep every lag product at its original separation.
    let samples = resample(
        &moments,
        Subtraction::LagMeans,
        &Resampling::Bootstrap {
            block: BlockSize::Fixed { frames: 50 },
            samples: 64,
            seed: 5,
        },
        None,
    )
    .unwrap();
    assert_eq!(samples.count, 64);
    assert_eq!(samples.blocks, 40);
    let drawn: Vec<f64> = (0..samples.count)
        .map(|r| samples.sample(r)[1] / samples.sample(r)[0])
        .collect();
    let mean: f64 = drawn.iter().sum::<f64>() / drawn.len() as f64;
    close(mean, ratio, 0.05);
    assert!(drawn.iter().fold(1., |s: f64, x| s.min(*x)) > 0.75);
    // Drawing single time points instead leaves an uncorrelated series.
    let mut rng = Rng::new(5);
    let shuffled: Vec<f64> = (0..2000)
        .map(|_| x[((rng.uniform() * 2000.) as usize).min(1999)])
        .collect();
    let broken = series_moments(view(&shuffled, &weight, &segment), 1, 2000)
        .unwrap()
        .estimate(Subtraction::LagMeans);
    assert!(
        (broken[1].unwrap() / broken[0].unwrap()).abs() < 0.1,
        "{:?}",
        broken
    );
}

#[test]
fn two_mixed_autoregressive_operators_give_the_resampled_eigenvalues_of_the_reference_run() {
    let (x1, x2) = (ar1(21, 3000, 0.9), ar1(22, 3000, 0.6));
    let a: Vec<f64> = x1.iter().zip(&x2).map(|(x, y)| x + 0.5 * y).collect();
    let b: Vec<f64> = x1.iter().zip(&x2).map(|(x, y)| 0.3 * x + y).collect();
    close(a[0], -0.561797650904643, 1e-12);
    close(b[2], -0.919661156810104, 1e-12);
    let (weight, segment) = (vec![1.; 3000], vec![0u32; 3000]);
    let series = [view(&a, &weight, &segment), view(&b, &weight, &segment)];
    let moments: Vec<BlockMoments> = (0..4)
        .map(|e| cross_moments(series[e / 2], series[e % 2], 5, 60).unwrap())
        .collect();
    // The lag-0 cross-correlator matrix is exactly symmetric.
    let lag0: Vec<f64> = moments
        .iter()
        .map(|m| m.estimate(Subtraction::LagMeans)[0].unwrap())
        .collect();
    assert_eq!(lag0[1].to_bits(), lag0[2].to_bits());
    close(lag0[0], 5.84154585642799, 1e-12);
    close(lag0[3], 2.11412486471125, 1e-12);
    let matrix = MatrixCorrelator {
        channels: vec!["channel/0".into(), "channel/1".into()],
        lags: (0..6).collect(),
        time_unit: TimeUnit::Frames,
        time_step: 1.,
        frames: 3000,
        tau_int: None,
        moments,
        replicas: 1,
    };
    let report = gevp(&matrix, &untracked(1, 2), &analysis(true, 60)).unwrap();
    assert_eq!(report.rank, 2);
    assert_eq!(report.lags, vec![1, 2, 3, 4, 5]);
    assert_eq!(report.levels.len(), 2);
    let expected: [[f64; 5]; 2] = [
        [
            1.,
            0.908559449483505,
            0.829184548736916,
            0.757526523264195,
            0.691568811258051,
        ],
        [
            1.,
            0.639852883159889,
            0.375617843606528,
            0.207320678176258,
            0.117020066923589,
        ],
    ];
    let spread: [[f64; 5]; 2] = [
        [
            0.,
            0.0104015959032272,
            0.0182132826383913,
            0.0236770755716304,
            0.0279351925714259,
        ],
        [
            0.,
            0.0244358668924399,
            0.0347467005208947,
            0.0377950304467993,
            0.0374563586083795,
        ],
    ];
    let rates: [[f64; 4]; 2] = [
        [
            0.0958949563209326,
            0.0914175762132378,
            0.0903841954672669,
            0.0910958946921279,
        ],
        [
            0.446516999115265,
            0.532666026721546,
            0.594305488672226,
            0.571921332214447,
        ],
    ];
    let rate_spread: [[f64; 4]; 2] = [
        [
            0.0114669231499437,
            0.0110300852734079,
            0.0103473291752698,
            0.0113343336308073,
        ],
        [
            0.0382049300550485,
            0.0626346138302223,
            0.106579679091461,
            0.178875617963554,
        ],
    ];
    for n in 0..2 {
        for k in 0..5 {
            close(value(report.eigenvalues[n][k]), expected[n][k], 1e-13);
            close(error(report.eigenvalues[n][k]), spread[n][k], 1e-13);
            if k < 4 {
                close(value(report.effective_mass[n][k]), rates[n][k], 1e-13);
                close(error(report.effective_mass[n][k]), rate_spread[n][k], 1e-13);
            }
        }
        assert_eq!(report.effective_mass[n][4], None);
    }
    let antisymmetric = [
        0.0013421416062722,
        0.000560610961370486,
        0.000208237167103629,
        0.00788483391170636,
        0.0113410524924094,
    ];
    for (k, expected) in antisymmetric.iter().enumerate() {
        close(report.antisymmetric_norm[k].unwrap(), *expected, 1e-9);
    }
    // Each level is the correlated log-linear fit of its principal correlator
    // over the single window 2..5 that lies beyond t0, relabelled as a GEVP
    // fit. The SVD floor at 1e-2 keeps two of the four directions of the
    // first state's log covariance.
    let fitted = [
        [0.0909137072043883, 0.0100702194429821, 0.00869544703006697],
        [0.523415160486443, 0.0615841722315521, 0.703303641433862],
    ];
    for (n, expected) in fitted.iter().enumerate() {
        let level = &report.levels[n];
        assert_eq!(level.method, FitMethodKind::Gevp);
        assert_eq!(level.diagnostics.window, Some([2, 5]));
        assert_eq!(level.diagnostics.dof, Some(2));
        let mass = level.mass.as_ref().unwrap();
        assert_eq!(mass.method, FitMethodKind::Gevp);
        close(mass.value, expected[0], 1e-12);
        close(mass.error, expected[1], 1e-12);
        close(level.diagnostics.chi2.unwrap(), expected[2], 1e-12);
    }
    assert_eq!(report.levels[0].diagnostics.covariance_rank, Some(2));
    assert_eq!(report.levels[1].diagnostics.covariance_rank, Some(4));
    // Holding the eigenvectors of the reference lag across the lags and the
    // resamples is a second estimator of the same spectrum. These operators
    // mix, so the ordering of a lag is not by itself the state the reference
    // lag identified, and the two part beyond that lag.
    let tracked = gevp(&matrix, &basis(1, 2), &analysis(true, 60)).unwrap();
    let held: [[f64; 5]; 2] = [
        [
            1.,
            0.908559449483505,
            0.828627687844395,
            0.755799083514158,
            0.68926866715761,
        ],
        [
            1.,
            0.639852883159889,
            0.376174704499048,
            0.209048117926295,
            0.119320211024029,
        ],
    ];
    let held_fit: [[f64; 3]; 2] = [
        [0.0920661456429825, 0.0101375405972694, 0.000101451263420474],
        [0.523014251742016, 0.0619088265986632, 0.607336997522242],
    ];
    for (n, expected) in held.iter().enumerate() {
        for (k, expected) in expected.iter().enumerate() {
            close(value(tracked.eigenvalues[n][k]), *expected, 1e-13);
        }
        let mass = tracked.levels[n].mass.as_ref().unwrap();
        close(mass.value, held_fit[n][0], 1e-12);
        close(mass.error, held_fit[n][1], 1e-12);
        close(
            tracked.levels[n].diagnostics.chi2.unwrap(),
            held_fit[n][2],
            1e-12,
        );
    }
    // At the reference lag the same eigenvector solves both, and the
    // antisymmetric part is a property of the matrix, not of the labelling.
    close(
        value(tracked.eigenvalues[0][1]),
        value(report.eigenvalues[0][1]),
        1e-13,
    );
    assert!(
        (value(tracked.eigenvalues[0][2]) - value(report.eigenvalues[0][2])).abs() > 5e-4,
        "{:?} vs {:?}",
        tracked.eigenvalues[0][2],
        report.eigenvalues[0][2]
    );
    assert_eq!(tracked.antisymmetric_norm, report.antisymmetric_norm);
    assert!(
        tracked.notes[0].contains("eigenvectors of the pencil at lag 2"),
        "{:?}",
        tracked.notes
    );
}

#[test]
fn an_automatic_block_is_the_largest_diagonal_autocorrelation_time_of_the_whole_matrix() {
    let (x1, x2) = (ar1(21, 3000, 0.9), ar1(22, 3000, 0.6));
    let a: Vec<f64> = x1.iter().zip(&x2).map(|(x, y)| x + 0.5 * y).collect();
    let b: Vec<f64> = x1.iter().zip(&x2).map(|(x, y)| 0.3 * x + y).collect();
    let (weight, segment) = (vec![1.; 3000], vec![0u32; 3000]);
    let series = [view(&a, &weight, &segment), view(&b, &weight, &segment)];
    let moments: Vec<BlockMoments> = (0..4)
        .map(|e| cross_moments(series[e / 2], series[e % 2], 5, 10).unwrap())
        .collect();
    let auto = AnalysisConfig {
        connected: true,
        resampling: Resampling::BlockJackknife {
            block: BlockSize::Auto,
        },
        ..AnalysisConfig::default()
    };
    // Left to themselves the members disagree: over stored blocks of ten
    // origins the second channel asks for 50 and the rest for 60, and
    // resamples over different blocks cannot be combined into one matrix.
    let alone: Vec<usize> = moments
        .iter()
        .map(|m| {
            resample(m, Subtraction::LagMeans, &auto.resampling, None)
                .unwrap()
                .effective_block
        })
        .collect();
    assert_eq!(alone, vec![60, 60, 60, 50]);
    let shared = (0..2)
        .filter_map(|e| {
            let m = &moments[e * 2 + e];
            tau_int_of_correlator(&m.estimate(Subtraction::LagMeans), m.origins())
        })
        .fold(0.5, |best: f64, t| best.max(t.tau));
    let lowest = (0..2)
        .filter_map(|e| {
            let m = &moments[e * 2 + e];
            tau_int_of_correlator(&m.estimate(Subtraction::LagMeans), m.origins())
        })
        .fold(f64::INFINITY, |best: f64, t| best.min(t.tau));
    assert!(lowest < shared, "{lowest} vs {shared}");
    let matrix = MatrixCorrelator {
        channels: vec!["channel/0".into(), "channel/1".into()],
        lags: (0..6).collect(),
        time_unit: TimeUnit::Frames,
        time_step: 1.,
        frames: 3000,
        tau_int: None,
        moments,
        replicas: 1,
    };
    let stated = MatrixCorrelator {
        tau_int: Some(shared),
        ..matrix.clone()
    };
    let derived = gevp(&matrix, &basis(1, 2), &auto).unwrap();
    assert_eq!(derived.availability, Availability::Available);
    assert_eq!(derived.rank, 2);
    assert_eq!(derived, gevp(&stated, &basis(1, 2), &auto).unwrap());
    // It is the largest of the diagonal times, and a stated one is used as it
    // stands: the smaller member gives a different block and a different
    // report, so neither the choice nor `matrix.tau_int` is decorative.
    let lowered = MatrixCorrelator {
        tau_int: Some(lowest),
        ..matrix.clone()
    };
    assert_ne!(derived, gevp(&lowered, &basis(1, 2), &auto).unwrap());
}

#[test]
fn a_reference_lag_past_the_last_fitted_window_leaves_the_levels_unfitted() {
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let matrix = correlator(&mixing(7, &identity), 3, 7);
    let mut config = analysis(false, 1);
    config.window_scan.t_max = Some(5);
    config.window_scan.validate().unwrap();
    // The scan is raised to lag 3, one beyond t0, where the eigenvalue is no
    // longer exactly one with no variance; four points from there do not fit
    // below lag 5, so the level declines and the basis does not. The request
    // itself is valid from lag 2, so only the raise can decline it.
    let report = gevp(&matrix, &basis(2, 3), &config).unwrap();
    assert_eq!(report.availability, Availability::Available);
    assert_eq!(report.levels.len(), report.rank);
    for level in &report.levels {
        assert_eq!(level.method, FitMethodKind::Gevp);
        assert!(level.mass.is_none());
        assert!(
            level
                .notes
                .iter()
                .any(|n| n.contains("no window beyond t0")),
            "{:?}",
            level.notes
        );
    }
}

#[test]
fn a_basis_without_a_measured_or_estimable_reference_lag_is_unavailable() {
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let entries = mixing(7, &identity);
    let matrix = correlator(&entries, 3, 7);
    let missing = gevp(&matrix, &basis(9, 3), &analysis(false, 1)).unwrap();
    assert_eq!(
        missing.availability,
        Availability::unavailable("the reference lag t0 is not measured")
    );
    assert!(missing.eigenvalues.is_empty() && missing.rank == 0);
    assert_eq!(missing.channels.len(), 3);
    // One entry of C(t0) without a pair weight leaves the metric undefined.
    let mut unweighted = correlator(&entries, 3, 7);
    unweighted.moments[1] = blind(&entries[7..14], 1);
    let undefined = gevp(&unweighted, &basis(1, 3), &analysis(false, 1)).unwrap();
    assert_eq!(
        undefined.availability,
        Availability::unavailable("the correlator matrix at t0 has an undefined entry")
    );
    assert!(undefined.eigenvalues.is_empty() && undefined.antisymmetric_norm.is_empty());
    let mut single = basis(1, 3);
    single.channels.truncate(1);
    assert!(matches!(
        gevp(&matrix, &single, &analysis(false, 1)),
        Err(GasError::Configuration(_))
    ));
}

/// Two operators overlapping two states through a non-orthogonal overlap:
/// `C(t) = Z e^{−E t} Zᵀ`, the system the tracking oracle measures on.
const BIAS_RATES: [f64; 2] = [0.20, 0.85];
const BIAS_Z: [f64; 4] = [1., 0.6, 0.45, 1.];
/// One replica of that system as `[4, lags]` entries, with symmetric Gaussian
/// matrix noise of width `sigma` added to every lag independently.
fn noisy(rng: &mut Rng, lags: usize, sigma: f64) -> Vec<f64> {
    let mut out = vec![0.; 4 * lags];
    for t in 0..lags {
        let n: Vec<f64> = (0..4).map(|_| sigma * rng.normal()).collect();
        for (i, j) in (0..2).flat_map(|i| (0..2).map(move |j| (i, j))) {
            let c: f64 = (0..2)
                .map(|s| BIAS_Z[i * 2 + s] * BIAS_Z[j * 2 + s] * (-BIAS_RATES[s] * t as f64).exp())
                .sum();
            out[(i * 2 + j) * lags + t] = c + 0.5 * (n[i * 2 + j] + n[j * 2 + i]);
        }
    }
    out
}
/// The same matrix over two identical blocks instead of eight: the point
/// estimate and the resamples are unchanged and the jackknife is the
/// cheapest one a replica study can carry.
fn replicated(entries: &[f64], n: usize, lags: usize) -> MatrixCorrelator {
    let pair = |values: &[f64]| {
        let mut moments = BlockMoments::new(lags, 1, 1, 64).unwrap();
        let (zero, one) = (vec![0.; lags], vec![1.; lags]);
        for _ in 0..2 {
            moments.push_origin(values, &zero, &zero, &one).unwrap();
        }
        moments
    };
    MatrixCorrelator {
        moments: entries.chunks_exact(lags).map(pair).collect(),
        ..correlator(entries, n, lags)
    }
}
/// Unweighted least-squares decay rate of a positive curve against its lags.
fn log_slope(lags: &[usize], values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let t: Vec<f64> = lags.iter().map(|&t| t as f64).collect();
    let y: Vec<f64> = values.iter().map(|v| v.ln()).collect();
    let (tbar, ybar) = (t.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let covariance: f64 = t.iter().zip(&y).map(|(t, y)| (t - tbar) * (y - ybar)).sum();
    let spread: f64 = t.iter().map(|t| (t - tbar) * (t - tbar)).sum();
    -covariance / spread
}

#[test]
fn fixed_vectors_remove_the_upward_bias_of_the_largest_eigenvalue() {
    let (lags, t0, replicas) = (21, 3, 600);
    let count = lags - t0;
    // Row 0 holds the eigenvectors of the reference lag, row 1 reads the
    // ordering of each lag; `[2, count]`, summed over the replicas that
    // carried an eigenvalue there.
    let arms = [basis(t0, 2), untracked(t0, 2)];
    let (mut sum, mut seen) = (vec![0.; 2 * count], vec![0.; 2 * count]);
    let mut rng = Rng::new(20260920);
    for _ in 0..replicas {
        let matrix = replicated(&noisy(&mut rng, lags, 0.002), 2, lags);
        for (a, arm) in arms.iter().enumerate() {
            let report = gevp(&matrix, arm, &analysis(false, 1)).unwrap();
            for k in 0..count {
                if let Some([value, _]) = report.eigenvalues[0][k] {
                    sum[a * count + k] += value;
                    seen[a * count + k] += 1.;
                }
            }
        }
    }
    let mean: Vec<f64> = (0..2 * count).map(|c| sum[c] / seen[c]).collect();
    // Two operators span the two states, so the pencil is solved exactly and
    // the principal correlator is `e^{−E₀ (t − t0)}` at every lag.
    let exact = |k: usize| (-BIAS_RATES[0] * k as f64).exp();
    // Taking the largest of a noisy spectrum lifts every lag, the more the
    // closer the gap; holding one eigenvector does not. Lags 8 and 17 are the
    // oracle's points, where the exact values are 0.367879441171442 and
    // 0.0608100626252178.
    for k in [5, 11, 14] {
        assert!(
            mean[count + k] > exact(k),
            "lag {}: {} vs {}",
            t0 + k,
            mean[count + k],
            exact(k)
        );
        // Relative to the eigenvalue, not to unity: at lag 17 the exact value
        // is 0.06, and the labelled arm sits 15 % above it.
        assert!(
            (mean[k] - exact(k)).abs() <= 0.02 * exact(k),
            "lag {}: {} vs {}",
            t0 + k,
            mean[k],
            exact(k)
        );
    }
    assert!(mean[count + 11] > 1.04 * exact(11), "{}", mean[count + 11]);
    assert!(mean[count + 14] > 1.1 * exact(14), "{}", mean[count + 14]);
    // The ordering arm is never undefined, because the largest of the two
    // eigenvalues of a noisy pencil stays positive where the state it labels
    // has decayed into the noise; that is the same mechanism as the bias.
    assert_eq!(seen[count..], vec![replicas as f64; count]);
    assert!(
        seen[count - 1] > 0.98 * replicas as f64,
        "{}",
        seen[count - 1]
    );
    let measured: Vec<usize> = (t0..lags).collect();
    let tracked = log_slope(&measured[1..], &mean[1..count]);
    let labelled = log_slope(&measured[1..], &mean[count + 1..]);
    assert!(
        (tracked - BIAS_RATES[0]).abs() <= 0.01 * BIAS_RATES[0],
        "{tracked}"
    );
    assert!(labelled < 0.19, "{labelled}");
}

#[test]
fn a_basis_or_a_group_of_two_channels_that_measure_the_same_series_is_refused() {
    // A displacement that is the score gradient has no transverse part, so
    // its longitudinal projection is the whole of it; `Re q_ij` is unmoved by
    // the conjugation a score orientation applies, so off a tie the
    // score-directed scalar meson is the standard one. Either pair inside one
    // basis is an exact null direction of the joint covariance.
    let pairs = [
        [
            "vector/vector/full/score_gradient",
            "vector/vector/longitudinal/score_gradient",
        ],
        ["meson/scalar/standard", "meson/scalar/score_directed"],
    ];
    for [one, other] in pairs {
        let refused = AnalysisConfig {
            gevp: vec![GevpBasis {
                id: "twins".into(),
                channels: vec![one.into(), other.into()],
                ..GevpBasis::default()
            }],
            ..AnalysisConfig::default()
        };
        let message = refused.validate().unwrap_err().to_string();
        assert!(
            message.contains(one) && message.contains(other) && message.contains("same series"),
            "{message}"
        );
        // A group is held to the same rule, and a member may name one channel
        // of a specification, which carries the element kind of its role.
        let grouped = AnalysisConfig {
            groups: vec![ChannelGroup {
                id: "twins".into(),
                channels: vec![format!("{one}/cloning"), format!("{other}/cloning")],
            }],
            ..AnalysisConfig::default()
        };
        assert!(matches!(
            grouped.validate(),
            Err(GasError::Configuration(_))
        ));
        // The same two specifications on different roles are two series.
        let roles = AnalysisConfig {
            groups: vec![ChannelGroup {
                id: "roles".into(),
                channels: vec![format!("{one}/cloning"), format!("{other}/distance")],
            }],
            ..AnalysisConfig::default()
        };
        roles.validate().unwrap();
        // A member that names a specification alone stands for every measured
        // role of it, so it carries the role the twin's channel names and the
        // basis resolves to that one series twice.
        let mixed = AnalysisConfig {
            gevp: vec![GevpBasis {
                id: "mixed".into(),
                channels: vec![one.into(), format!("{other}/distance")],
                ..GevpBasis::default()
            }],
            ..AnalysisConfig::default()
        };
        assert!(matches!(mixed.validate(), Err(GasError::Configuration(_))));
    }
    // The transverse part of the same displacement is a different observable.
    let apart = AnalysisConfig {
        gevp: vec![GevpBasis {
            id: "apart".into(),
            channels: vec![
                "vector/vector/full/score_gradient".into(),
                "vector/vector/transverse/score_gradient".into(),
            ],
            ..GevpBasis::default()
        }],
        ..AnalysisConfig::default()
    };
    apart.validate().unwrap();
}

#[test]
fn a_reference_lag_must_lie_beyond_t0_and_the_projection_round_trips_through_json() {
    let mut early = basis(3, 2);
    early.t_ref = Some(3);
    let message = early.validate().unwrap_err().to_string();
    assert!(message.contains("reference lag"), "{message}");
    early.t_ref = Some(4);
    early.validate().unwrap();
    let identity = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let matrix = correlator(&mixing(7, &identity), 3, 7);
    // A reference lag the matrix does not carry declines the basis; it does
    // not fail the analysis of the rest of the report.
    let mut absent = basis(1, 3);
    absent.t_ref = Some(9);
    let declined = gevp(&matrix, &absent, &analysis(false, 1)).unwrap();
    assert_eq!(
        declined.availability,
        Availability::unavailable("the reference lag t_ref is not measured")
    );
    // On an exactly exponential system every lag gives the same eigenvectors,
    // so moving the reference lag moves nothing but the note.
    let mut later = basis(1, 3);
    later.t_ref = Some(4);
    let moved = gevp(&matrix, &later, &analysis(false, 1)).unwrap();
    let plain = gevp(&matrix, &basis(1, 3), &analysis(false, 1)).unwrap();
    assert!(moved.notes[0].contains("at lag 4"), "{:?}", moved.notes);
    for n in 0..2 {
        for k in 0..plain.lags.len() {
            close(
                value(moved.eigenvalues[n][k]),
                value(plain.eigenvalues[n][k]),
                1e-10,
            );
        }
    }
    let text = serde_json::to_string(&absent).unwrap();
    assert!(text.contains(r#""projection":"fixed_vector""#), "{text}");
    assert!(text.contains(r#""t_ref":9"#), "{text}");
    assert_eq!(serde_json::from_str::<GevpBasis>(&text).unwrap(), absent);
    let parsed: GevpBasis =
        serde_json::from_str(r#"{"id":"b","channels":["x","y"],"projection":"max_eigenvalue"}"#)
            .unwrap();
    assert_eq!(parsed.projection, GevpProjection::MaxEigenvalue);
    assert_eq!(parsed.t_ref, None);
    assert!(serde_json::from_str::<GevpBasis>(r#"{"projection":"tracked"}"#).is_err());
    assert!(serde_json::from_str::<GevpBasis>(r#"{"reference":3}"#).is_err());
}
