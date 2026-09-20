//! Effective rates and the AIC window scan against closed forms, the
//! reference tables of the statistics dossier and the Python parity fixture.
use algorithmic_gas::{
    GasConfig, GasError, RecordingConfig,
    physics::{
        numerics::{ResampleKind, Samples},
        qft::math::Rng,
        spectroscopy::{
            config::{
                AnalysisConfig, ChannelSpec, EffectiveMassKind, MeasurementConfig, TimeUnit,
                WindowScanConfig,
            },
            contract::{
                Availability, Capabilities, ElementKind, ExchangeParity, SPECTROSCOPY_VERSION,
            },
            estimators::{Estimated, estimate},
            fits::{
                effective_mass::{cosh_rate, effective_mass},
                window_scan::window_scan,
            },
            measurement::{ChannelSeries, Measurement},
            report::{
                CorrelatorEstimate, Coverage, EstimatorKind, FitMethodKind, RATE_QUANTITY,
                RATE_QUANTITY_EUCLIDEAN, RATE_QUANTITY_SOURCE_FROZEN, SamplesMeta, WindowFit,
            },
        },
    },
};

/// Two exponentials `0.7 exp(-0.3 t) + 0.3 exp(-1.2 t)` plus a fixed
/// deterministic wiggle, lags 0..=10. The asymptotic rate is 0.3.
const REFERENCE: [f64; 11] = [
    1.00236416165329,
    0.614618730953712,
    0.408508611620576,
    0.288963900937168,
    0.216686255172136,
    0.159497873237547,
    0.112276643234918,
    0.0843721839504245,
    0.0671727676189093,
    0.0474347866893261,
    0.031456805983983,
];
const REFERENCE_SIGMA: [f64; 11] = [
    0.01,
    0.0078188290418685,
    0.00678256778391962,
    0.00619848879759992,
    0.00579822758711437,
    0.00547756056710919,
    0.00519576544114444,
    0.00493669869434739,
    0.0046937416765498,
    0.00446397769665109,
    0.0042459391734781,
];
/// The same two exponentials without the wiggle, lags 0..=7.
const SMOOTH: [f64; 8] = [
    1.,
    0.608931018050863,
    0.411383531252642,
    0.292795878552607,
    0.213304872453248,
    0.156934737756901,
    0.115933197497624,
    0.085786959974341,
];
const SMOOTH_SIGMA: [f64; 8] = [
    0.01,
    0.0078188290418685,
    0.00678256778391962,
    0.00619848879759992,
    0.00579822758711437,
    0.00547756056710919,
    0.00519576544114444,
    0.00493669869434739,
];
/// Log-ratio rate and its delta-method error at lags 0..=6 of `SMOOTH`.
const LOG_RATIO: [[f64; 2]; 7] = [
    [0.496050288540309, 0.00770895974823933],
    [0.392179045009546, 0.00989850025296203],
    [0.340050239425726, 0.0127099259118933],
    [0.316753237596306, 0.0163198679150891],
    [0.306892432559167, 0.0209551251999612],
    [0.302815894270568, 0.0269069133666239],
    [0.301147128288017, 0.0345491606473603],
];
/// Arccosh rate and its delta-method error at lags 1..=6 of `SMOOTH`.
const COSH: [[f64; 2]; 6] = [
    [0.556532123450988, 0.0121099282871613],
    [0.434676321636509, 0.0190056603224112],
    [0.363412645934583, 0.0285799397244724],
    [0.327762653939878, 0.0403270156610098],
    [0.311674397840296, 0.0542533504390054],
    [0.304815505450337, 0.0711228699538017],
];

/// `sigma_i sigma_j rho^|i-j|`.
fn covariance(sigma: &[f64], rho: f64) -> Vec<f64> {
    let n = sigma.len();
    (0..n * n)
        .map(|c| sigma[c / n] * sigma[c % n] * rho.powi((c / n).abs_diff(c % n) as i32))
        .collect()
}
/// Lower Cholesky factor of a symmetric positive definite `[n, n]` matrix.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let inner: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            l[i * n + j] = if i == j {
                (a[i * n + i] - inner).sqrt()
            } else {
                (a[i * n + j] - inner) / l[j * n + j]
            };
        }
    }
    l
}
/// Jackknife resamples whose covariance is exactly `covariance`: every column
/// of its Cholesky factor taken with both signs, so the resample mean is the
/// central value and the factor `(n-1)/n` reproduces the input.
fn resampled(central: &[f64], covariance: &[f64]) -> Samples {
    let n = central.len();
    let l = cholesky(covariance, n);
    let count = 2 * n;
    let scale = (count as f64 / (2. * (count as f64 - 1.))).sqrt();
    let mut values = vec![0.; count * n];
    for (r, row) in values.chunks_exact_mut(n).enumerate() {
        let sign = if r.is_multiple_of(2) { scale } else { -scale };
        for (i, v) in row.iter_mut().enumerate() {
            *v = central[i] + sign * l[i * n + r / 2];
        }
    }
    Samples {
        kind: ResampleKind::Jackknife,
        dimension: n,
        count,
        central: central.to_vec(),
        values,
        defined: vec![true; n],
        effective_block: 16,
        blocks: count,
        tau_int: Some(2.),
    }
}
/// An estimate with contiguous lags, a supplied covariance and no resamples,
/// so that every error is the delta method on that covariance.
fn estimated(value: &[f64], covariance: Vec<f64>) -> Estimated {
    let n = value.len();
    strided(&(0..n).collect::<Vec<usize>>(), value, covariance, 1.)
}
/// The same with explicit lags and a time step.
fn strided(lags: &[usize], value: &[f64], covariance: Vec<f64>, time_step: f64) -> Estimated {
    let n = value.len();
    let error: Vec<Option<f64>> = (0..n).map(|i| Some(covariance[i * n + i].sqrt())).collect();
    Estimated {
        channel: "test/channel".into(),
        kind: EstimatorKind::FrameMean,
        estimate: CorrelatorEstimate {
            lags: lags.to_vec(),
            time_unit: TimeUnit::Frames,
            time_step,
            value: value.iter().map(|&c| Some(c)).collect(),
            error,
            covariance: Some(covariance),
            samples_meta: SamplesMeta {
                resampling: ResampleKind::Jackknife,
                effective_block: 16,
                blocks: 32,
                tau_int: Some(2.),
                covariance_rank: n,
                replicas: 1,
                sampling_unit: "time blocks of 16 origins".into(),
            },
            connected: true,
            connected_bias: None,
        },
        samples: Samples {
            kind: ResampleKind::Jackknife,
            dimension: n,
            count: 0,
            central: value.to_vec(),
            values: vec![],
            defined: vec![true; n],
            effective_block: 16,
            blocks: 32,
            tau_int: Some(2.),
        },
    }
}
fn analysis(correlated: bool) -> AnalysisConfig {
    AnalysisConfig {
        svd_cut: 1e-6,
        window_scan: WindowScanConfig {
            correlated,
            ..WindowScanConfig::default()
        },
        ..AnalysisConfig::default()
    }
}
fn row(windows: &[WindowFit], t_min: usize, t_max: usize) -> WindowFit {
    *windows
        .iter()
        .find(|w| w.t_min == t_min && w.t_max == t_max)
        .unwrap()
}
/// `chi2 + 2 k + 2 N_cut` recovered from a row of a scan over `usable` lags.
fn criterion(window: &WindowFit, usable: usize) -> f64 {
    window.chi2 + 4. + 2. * (usable - window.dof - 2) as f64
}
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.)
}

#[test]
fn the_correlated_window_scan_reproduces_the_reference_table_and_its_model_average() {
    let data = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(fit.method, FitMethodKind::WindowScan);
    assert_eq!(fit.diagnostics.n_windows, 28);
    assert_eq!(fit.windows.len(), 28);
    assert!(fit.diagnostics.correlated);
    assert_eq!(fit.diagnostics.window, Some([3, 10]));
    assert_eq!(fit.diagnostics.covariance_rank, Some(10));
    assert_eq!(fit.diagnostics.svd_cut, 1e-6);
    assert_eq!(fit.diagnostics.model_rejected, None);
    assert_eq!(fit.diagnostics.no_signal, None);
    let mass = fit.mass.clone().unwrap();
    assert_eq!(mass.quantity, RATE_QUANTITY);
    assert_eq!(mass.time_unit, TimeUnit::Frames);
    assert!(mass.prior_dominance.is_none());
    assert!(close(mass.value, 0.305508641581079, 1e-9));
    assert!(close(mass.statistical, 0.0143346873421187, 1e-9));
    assert!(close(mass.systematic, 0.00793410421634506, 1e-9));
    assert!(close(mass.error, 0.0163839333162742, 1e-9));
    // The data are two exponentials whose slow rate is 0.3.
    assert!((mass.value - 0.3).abs() <= 2. * mass.error);
    // Every window of the dossier table, spot-checked at the extremes of the
    // weight range; `criterion` recovers the AIC the weights came from.
    for (limits, want) in [
        (
            [1, 4],
            [
                0.363118514850536,
                0.00834222234450293,
                21.9167504358576,
                37.9167504358576,
                7.3427705470541e-07,
            ],
        ),
        (
            [2, 10],
            [
                0.315642253584867,
                0.00829908362301219,
                8.40626702490373,
                14.4062670249037,
                0.0935613680366756,
            ],
        ),
        (
            [3, 10],
            [
                0.302008358083002,
                0.0109499719693221,
                4.76343572647449,
                12.7634357264745,
                0.212732011768374,
            ],
        ),
        (
            [7, 10],
            [
                0.296487630938332,
                0.0393685175831246,
                2.79477133276629,
                18.7947713327663,
                0.0104266539742302,
            ],
        ),
    ] {
        let window = row(&fit.windows, limits[0], limits[1]);
        assert_eq!(window.nexp, 1);
        assert_eq!(window.svd_cut, None);
        assert_eq!(window.dof, limits[1] - limits[0] - 1);
        assert!(close(window.value, want[0], 1e-9));
        assert!(close(window.error, want[1], 1e-9));
        assert!(close(window.chi2, want[2], 1e-9));
        assert!(close(criterion(&window, 10), want[3], 1e-9));
        // The reported criterion is the unshifted AIC, so the two-parameter
        // term is visible in the output and not only in the weights, where a
        // constant offset would cancel.
        assert_eq!(window.aic.to_bits(), criterion(&window, 10).to_bits());
        assert!(close(window.weight, want[4], 1e-8));
    }
    let total: f64 = fit.windows.iter().map(|w| w.weight).sum();
    assert!((total - 1.).abs() < 1e-14);
    // The reported chi^2, dof and probability belong to the window the
    // diagnostics name, not to the first or the last row of the scan.
    let best = row(&fit.windows, 3, 10);
    assert_eq!(fit.diagnostics.chi2, Some(best.chi2));
    assert_eq!(fit.diagnostics.dof, Some(best.dof));
    assert!(close(fit.diagnostics.q.unwrap(), 0.574492470124602, 1e-9));
    assert!(fit.notes.iter().all(|n| !n.contains("describes it poorly")));
    // Only the frame-mean correlator carries a transfer-matrix reading; the
    // other estimators name the rate they really measure.
    for (kind, want) in [
        (EstimatorKind::SourceFrozen, RATE_QUANTITY_SOURCE_FROZEN),
        (EstimatorKind::EuclideanTime, RATE_QUANTITY_EUCLIDEAN),
    ] {
        let mut other = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
        other.kind = kind;
        let mass = window_scan(&other, &analysis(true)).unwrap().mass.unwrap();
        assert_eq!(mass.quantity, want);
        assert!(close(mass.value, 0.305508641581079, 1e-9));
    }
}

#[test]
fn the_diagonal_window_scan_keeps_the_full_covariance_in_its_slope_error() {
    let data = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
    let fit = window_scan(&data, &analysis(false)).unwrap();
    assert!(!fit.diagnostics.correlated);
    assert_eq!(fit.diagnostics.window, Some([2, 10]));
    assert_eq!(fit.diagnostics.n_windows, 28);
    assert!(
        fit.notes
            .iter()
            .any(|n| n.contains("diagonal chi^2") && n.contains("full lag covariance"))
    );
    let mass = fit.mass.clone().unwrap();
    assert!(close(mass.value, 0.307813533022845, 1e-9));
    assert!(close(mass.statistical, 0.0129333408373911, 1e-9));
    assert!(close(mass.systematic, 0.00547357752790727, 1e-9));
    assert!(close(mass.error, 0.0140439081515845, 1e-9));
    for (limits, want) in [
        (
            [1, 4],
            [
                0.359194018217451,
                0.00847579423594053,
                10.1347751916869,
                26.1347751916869,
                6.8684067723392e-05,
            ],
        ),
        (
            [2, 10],
            [
                0.311740657323631,
                0.00889572393967615,
                3.76388115940763,
                9.76388115940763,
                0.246462289651813,
            ],
        ),
        (
            [3, 10],
            [
                0.304818374976664,
                0.0116497329704557,
                2.23360001763569,
                10.2336000176357,
                0.194873293317696,
            ],
        ),
        (
            [7, 10],
            [
                0.308840865570659,
                0.040166838866683,
                1.24844525808772,
                17.2484452580877,
                0.00584114564262085,
            ],
        ),
    ] {
        let window = row(&fit.windows, limits[0], limits[1]);
        assert!(close(window.value, want[0], 1e-9));
        assert!(close(window.error, want[1], 1e-9));
        assert!(close(window.chi2, want[2], 1e-9));
        assert!(close(criterion(&window, 10), want[3], 1e-9));
        assert_eq!(window.aic.to_bits(), criterion(&window, 10).to_bits());
        assert!(close(window.weight, want[4], 1e-8));
    }
}

#[test]
fn a_pure_exponential_gives_one_rate_in_every_window_and_analytic_cut_weights() {
    let value: Vec<f64> = (0..11).map(|t| 0.9 * (-0.4 * t as f64).exp()).collect();
    let sigma: Vec<f64> = value.iter().map(|c| 0.02 * c).collect();
    let data = estimated(&value, covariance(&sigma, 0.3));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    for window in &fit.windows {
        assert!((window.value - 0.4).abs() < 1e-12);
        assert!(window.chi2 < 1e-20);
    }
    let mass = fit.mass.clone().unwrap();
    assert!((mass.value - 0.4).abs() < 1e-12);
    assert!(mass.systematic < 1e-14);
    assert!(close(mass.statistical, 0.00341586457858027, 1e-9));
    // With every chi2 zero the weights are pure `exp(-N_cut)`; the full usable
    // window carries `1 / sum_{j=0}^{6} (j + 1) exp(-j)`.
    let analytic = 1.
        / (0..7)
            .map(|j| (j + 1) as f64 * (-(j as f64)).exp())
            .sum::<f64>();
    assert!(close(analytic, 0.401562859003376, 1e-12));
    assert!(close(row(&fit.windows, 1, 10).weight, analytic, 1e-12));
}

/// Lags the measurement skipped are still the times the fit reads, so every
/// quantity that names a lag must name the lag and not its place in the table.
#[test]
fn the_scan_fits_the_real_lag_of_every_point_and_labels_its_windows_with_it() {
    let lags = [0, 2, 4, 7, 11, 16, 22, 29];
    let value: Vec<f64> = lags
        .iter()
        .map(|&l| 0.8 * (-0.2 * l as f64).exp())
        .collect();
    let sigma: Vec<f64> = value.iter().map(|c| 0.03 * c).collect();
    let data = strided(&lags, &value, covariance(&sigma, 0.), 1.);
    let fit = window_scan(&data, &analysis(true)).unwrap();
    // Seven usable lags (1..=7) and windows of at least four points.
    assert_eq!(fit.diagnostics.n_windows, 10);
    assert!(
        fit.windows
            .iter()
            .all(|w| lags.contains(&w.t_min) && lags.contains(&w.t_max))
    );
    // A single exponential read at the right times fits every window exactly;
    // read at the point index instead it fits none of them.
    for window in &fit.windows {
        assert!((window.value - 0.2).abs() < 1e-12);
        assert!(window.chi2 < 1e-20);
    }
    assert_eq!(fit.diagnostics.window, Some([2, 29]));
    let full = row(&fit.windows, 2, 29);
    assert_eq!(full.dof, 5);
    let analytic = 1.
        / (0..4)
            .map(|j| (j + 1) as f64 * (-(j as f64)).exp())
            .sum::<f64>();
    assert!(close(analytic, 0.427183751655940, 1e-12));
    assert!(close(full.weight, analytic, 1e-12));
    let mass = fit.mass.unwrap();
    assert!((mass.value - 0.2).abs() < 1e-12);
    assert!(mass.systematic < 1e-14);
    assert!(close(mass.statistical, 0.00179022122243937, 1e-9));
}

#[test]
fn an_explicit_t_max_is_the_last_lag_the_scan_fits_and_the_range_it_cuts_from() {
    let config = AnalysisConfig {
        svd_cut: 1e-6,
        window_scan: WindowScanConfig {
            t_max: Some(6),
            ..WindowScanConfig::default()
        },
        ..AnalysisConfig::default()
    };
    let data = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
    let fit = window_scan(&data, &config).unwrap();
    // Lags 1..=6 are usable, so the widths are 4, 5 and 6; an exclusive bound
    // would leave five lags and only three windows.
    assert_eq!(fit.diagnostics.n_windows, 6);
    assert!(fit.windows.iter().all(|w| w.t_max <= 6 && w.t_min >= 1));
    assert_eq!(fit.diagnostics.window, Some([3, 6]));
    // The widest window now cuts nothing, which is what its weight reads.
    let full = row(&fit.windows, 1, 6);
    assert_eq!(full.dof, 4);
    assert!(close(full.value, 0.353990305409422, 1e-9));
    assert!(close(full.error, 0.0071251511000376, 1e-9));
    assert!(close(full.chi2, 27.0520390623939, 1e-9));
    assert!(close(criterion(&full, 6), 31.0520390623939, 1e-9));
    assert!(close(full.weight, 1.26491863172586e-05, 1e-8));
    let mass = fit.mass.unwrap();
    assert!(close(mass.value, 0.313714157124050, 1e-9));
    assert!(close(mass.statistical, 0.0121413510024576, 1e-9));
    assert!(close(mass.systematic, 0.00761398872332731, 1e-9));
}

#[test]
fn a_sign_changing_or_oscillating_correlator_is_rejected_instead_of_fitted() {
    let oscillating: Vec<f64> = (0..13)
        .map(|t| 0.9_f64.powi(t) * (0.6 * t as f64).cos())
        .collect();
    let sigma = vec![0.005; 13];
    let data = estimated(&oscillating, covariance(&sigma, 0.4));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(
        fit.diagnostics.model_rejected.as_deref(),
        Some("sign change at lag 3")
    );
    assert!(fit.mass.is_none());
    assert!(fit.windows.is_empty());
    assert_eq!(fit.diagnostics.no_signal, None);
    assert_eq!(fit.diagnostics.chi2, None);
    // A dip past the usable range but inside twice it is still the model
    // failing: the range ends at lag 10 and the dip at lag 16 rejects.
    let mut mid: Vec<f64> = (0..21).map(|t| (-0.3 * t as f64).exp()).collect();
    mid[16] = -0.2;
    let data = estimated(&mid, covariance(&[0.02; 21], 0.4));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(
        fit.diagnostics.model_rejected.as_deref(),
        Some("sign change at lag 16")
    );
    assert!(fit.mass.is_none());
    assert!(fit.windows.is_empty());
    // A dip that is far outside twice the usable range is noise, not a sign
    // change of the model: the scan still fits the range it has.
    let mut late: Vec<f64> = (0..31).map(|t| (-0.25 * t as f64).exp()).collect();
    late[30] = -0.5;
    let data = estimated(&late, covariance(&vec![0.02; 31], 0.4));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(fit.diagnostics.model_rejected, None);
    assert!(close(fit.mass.unwrap().value, 0.25, 1e-9));
}

#[test]
fn noise_with_no_decay_reports_no_signal_and_never_a_rate_of_zero() {
    let noise = [1., 0.004, -0.006, 0.013, -0.002, 0.009, -0.011, 0.005];
    let data = estimated(&noise, covariance(&[0.01; 8], 0.2));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(
        fit.diagnostics.no_signal.as_deref(),
        Some("too few usable lags")
    );
    assert!(fit.mass.is_none());
    assert_eq!(fit.diagnostics.model_rejected, None);
    // A flat correlator has usable points but no rate the errors resolve.
    let flat = [1., 1.02, 0.98, 1.01, 0.99, 1., 1.01, 0.97];
    let data = estimated(&flat, covariance(&[0.15; 8], 0.2));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(
        fit.diagnostics.no_signal.as_deref(),
        Some("rate S/N below threshold")
    );
    assert!(fit.mass.is_none());
    assert!(!fit.windows.is_empty());
    assert!(fit.diagnostics.chi2.is_some());
    // No positivity or minimum-rate filter drops a window: the three that
    // slope upward keep their weight and cancel the ones that slope down,
    // which is what leaves the rate unresolved instead of positive noise.
    let rising: Vec<&WindowFit> = fit.windows.iter().filter(|w| w.value < 0.).collect();
    assert_eq!(rising.len(), 3);
    assert!(rising.iter().all(|w| w.weight > 0.02));
    let total: f64 = fit.windows.iter().map(|w| w.weight).sum();
    assert!((total - 1.).abs() < 1e-14);
}

#[test]
fn a_rank_deficient_lag_covariance_deflates_only_the_correlated_chi2() {
    // Neighbouring lags correlated at 0.99 leave two directions above an SVD
    // floor of 1e-2; the other eight are raised and stop carrying weight.
    let floored = |correlated| AnalysisConfig {
        svd_cut: 1e-2,
        window_scan: WindowScanConfig {
            correlated,
            ..WindowScanConfig::default()
        },
        ..AnalysisConfig::default()
    };
    let data = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.99));
    let fit = window_scan(&data, &floored(true)).unwrap();
    assert_eq!(fit.diagnostics.covariance_rank, Some(2));
    assert_eq!(fit.diagnostics.window, Some([4, 7]));
    assert!(
        fit.notes
            .iter()
            .any(|n| n.contains("SVD floor is active on 8 of 10") && n.contains("not calibrated"))
    );
    let mass = fit.mass.unwrap();
    assert!(close(mass.value, 0.29607319275292976, 1e-9));
    assert!(close(mass.statistical, 0.011641343510544671, 1e-9));
    assert!(close(mass.systematic, 0.026113574645197896, 1e-9));
    // The diagonal arm never touches the floored covariance, so its chi2 is
    // not deflated and its rates do not depend on the correlation at all.
    let fit = window_scan(&data, &floored(false)).unwrap();
    assert_eq!(fit.diagnostics.covariance_rank, Some(2));
    assert!(fit.notes.iter().all(|n| !n.contains("SVD floor")));
    assert!(close(fit.mass.unwrap().value, 0.307813533022845, 1e-9));
    // Thirty-two blocks support ten fitted points, so neither fit above says
    // anything about the resample count. Six blocks do not: five delete-one
    // replicas cannot span ten lags, so the floor the scan then applies is
    // fabricating the missing directions rather than regulating measured ones.
    assert!(
        fit.notes
            .iter()
            .all(|n| !n.contains("independent replicas against")),
        "{:?}",
        fit.notes
    );
    let mut thin = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
    thin.samples.blocks = 6;
    thin.estimate.samples_meta.blocks = 6;
    let fit = window_scan(&thin, &analysis(true)).unwrap();
    let stated = "6 resampling blocks leave 5 independent replicas against 10 fitted points: the \
                  lag covariance is rank deficient by construction";
    assert!(fit.notes.iter().any(|n| n == stated), "{:?}", fit.notes);
    // The scan states it of the widest window it fits: ten lags are usable
    // here and ten is the count the note names, which is what the estimate
    // says of its own range as well.
    assert_eq!(fit.diagnostics.window, Some([3, 10]));
    assert!(
        thin.notes()
            .iter()
            .any(|n| n.contains("5 independent replicas"))
    );
}

#[test]
fn a_best_window_a_single_exponential_describes_poorly_is_named_with_its_probability() {
    // The same two exponentials measured three times as precisely: the values
    // do not move, the chi^2 of every window grows by about a factor of ten.
    let sharp: Vec<f64> = REFERENCE_SIGMA.iter().map(|s| 0.3 * s).collect();
    let data = estimated(&REFERENCE, covariance(&sharp, 0.6));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(fit.diagnostics.window, Some([4, 7]));
    assert_eq!(fit.diagnostics.dof, Some(2));
    assert!(close(fit.diagnostics.chi2.unwrap(), 13.2468078010101, 1e-9));
    assert!(close(fit.diagnostics.q.unwrap(), 0.00132889979551124, 1e-9));
    assert!(
        fit.notes
            .iter()
            .any(|n| n.contains("q = 0.0013") && n.contains("describes it poorly")),
        "{:?}",
        fit.notes
    );
    // The rate is still reported, and the disagreement between the windows now
    // dominates its error.
    let mass = fit.mass.unwrap();
    assert!(close(mass.value, 0.307899664855841, 1e-9));
    assert!(close(mass.statistical, 0.00619758895606091, 1e-9));
    assert!(close(mass.systematic, 0.0154470495513922, 1e-9));
}

#[test]
fn the_no_signal_gate_reads_the_total_error_and_not_the_statistical_part_alone() {
    // `0.3 exp(-0.02 t) + 0.7 exp(-2 t)`: the early windows and the late ones
    // disagree enough that the model spread, not the resampling, decides.
    let value: Vec<f64> = (0..13)
        .map(|t| 0.3 * (-0.02 * t as f64).exp() + 0.7 * (-2. * t as f64).exp())
        .collect();
    let sigma: Vec<f64> = value.iter().map(|c| 0.09 * c).collect();
    let data = estimated(&value, covariance(&sigma, 0.));
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert_eq!(fit.diagnostics.n_windows, 45);
    assert_eq!(
        fit.diagnostics.no_signal.as_deref(),
        Some("rate S/N below threshold")
    );
    assert!(fit.mass.is_none());
    // Recomputed from the published rows: the statistical error on its own
    // would have cleared `min_rate_snr = 2`, the reported error does not.
    let total: f64 = fit.windows.iter().map(|w| w.weight).sum();
    assert!((total - 1.).abs() < 1e-14);
    let average: f64 = fit.windows.iter().map(|w| w.weight * w.value).sum();
    let statistical: f64 = fit
        .windows
        .iter()
        .map(|w| w.weight * w.error * w.error)
        .sum::<f64>()
        .sqrt();
    let systematic: f64 = fit
        .windows
        .iter()
        .map(|w| w.weight * (w.value - average) * (w.value - average))
        .sum::<f64>()
        .sqrt();
    assert!(close(average, 0.0228116809378781, 1e-9));
    assert!(close(statistical, 0.0110330845696713, 1e-9));
    assert!(close(systematic, 0.00417432368155424, 1e-9));
    assert!(average / statistical > 2.);
    assert!(average / statistical.hypot(systematic) < 2.);
}

#[test]
fn relabelling_the_time_unit_divides_every_rate_and_leaves_chi2_and_weights_unchanged() {
    let lags: Vec<usize> = (0..11).collect();
    let frames = window_scan(
        &strided(&lags, &REFERENCE, covariance(&REFERENCE_SIGMA, 0.6), 1.),
        &analysis(true),
    )
    .unwrap();
    let quarters = window_scan(
        &strided(&lags, &REFERENCE, covariance(&REFERENCE_SIGMA, 0.6), 0.25),
        &analysis(true),
    )
    .unwrap();
    let (a, b) = (frames.mass.unwrap(), quarters.mass.unwrap());
    assert!(close(b.value, 4. * a.value, 1e-12));
    assert!(close(b.error, 4. * a.error, 1e-12));
    assert!(close(b.statistical, 4. * a.statistical, 1e-12));
    assert_eq!(frames.diagnostics.q, quarters.diagnostics.q);
    assert_eq!(frames.diagnostics.window, quarters.diagnostics.window);
    for (x, y) in frames.windows.iter().zip(&quarters.windows) {
        assert!(close(y.value, 4. * x.value, 1e-12));
        assert!(close(y.error, 4. * x.error, 1e-12));
        assert!(close(y.chi2, x.chi2, 1e-12));
        assert!(close(y.weight, x.weight, 1e-12));
    }
}

#[test]
fn the_scan_falls_back_to_the_resample_covariance_of_the_estimate() {
    let lag_covariance = covariance(&REFERENCE_SIGMA, 0.6);
    let mut data = estimated(&REFERENCE, lag_covariance.clone());
    data.samples = resampled(&REFERENCE, &lag_covariance);
    data.estimate.covariance = None;
    let fit = window_scan(&data, &analysis(true)).unwrap();
    assert!(close(fit.mass.unwrap().value, 0.305508641581079, 1e-7));
}

#[test]
fn the_scan_rejects_a_mismatched_estimate_and_an_oversized_usable_range() {
    let mut data = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
    data.estimate.value.pop();
    assert!(matches!(
        window_scan(&data, &analysis(true)),
        Err(GasError::Configuration(_))
    ));
    let long: Vec<f64> = (0..200).map(|t| (-0.01 * t as f64).exp()).collect();
    let sigma: Vec<f64> = long.iter().map(|c| 0.05 * c).collect();
    let data = estimated(&long, covariance(&sigma, 0.));
    let error = window_scan(&data, &analysis(true)).unwrap_err();
    assert!(matches!(error, GasError::Configuration(_)));
    assert!(error.to_string().contains("128 usable lags"));
    // The cap is the configured one, not a module constant: a lowered cap
    // rejects a range the default accepts, and names its own value.
    let capped = AnalysisConfig {
        window_scan: WindowScanConfig {
            max_usable: 5,
            ..WindowScanConfig::default()
        },
        ..AnalysisConfig::default()
    };
    let reference = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
    let error = window_scan(&reference, &capped).unwrap_err();
    assert!(matches!(error, GasError::Configuration(_)));
    assert!(error.to_string().contains("5 usable lags"));
    window_scan(&reference, &analysis(true)).unwrap();
    // A cap below the points one window needs could never fit anything.
    assert!(matches!(
        WindowScanConfig {
            max_usable: 3,
            min_points: 4,
            ..WindowScanConfig::default()
        }
        .validate(),
        Err(GasError::Configuration(_))
    ));
    let bad = AnalysisConfig {
        window_scan: WindowScanConfig {
            min_points: 2,
            ..WindowScanConfig::default()
        },
        ..AnalysisConfig::default()
    };
    let data = estimated(&REFERENCE, covariance(&REFERENCE_SIGMA, 0.6));
    assert!(matches!(
        window_scan(&data, &bad),
        Err(GasError::Configuration(_))
    ));
    // The usable range is found by bisection on the lags, and every time is a
    // lag times the step, so neither may be degenerate.
    for time_step in [0., -1., f64::NAN] {
        let data = strided(
            &(0..11).collect::<Vec<usize>>(),
            &REFERENCE,
            covariance(&REFERENCE_SIGMA, 0.6),
            time_step,
        );
        assert!(matches!(
            window_scan(&data, &analysis(true)),
            Err(GasError::Configuration(_))
        ));
    }
    let data = strided(
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9],
        &REFERENCE,
        covariance(&REFERENCE_SIGMA, 0.6),
        1.,
    );
    assert!(matches!(
        window_scan(&data, &analysis(true)),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn a_malformed_resample_table_leaves_the_rates_on_the_covariance_and_never_panics() {
    let lag_covariance = covariance(&SMOOTH_SIGMA, 0.8);
    let mut data = estimated(&SMOOTH, lag_covariance.clone());
    let delta = effective_mass(&data, EffectiveMassKind::LogRatio);
    for damage in 0..3 {
        data.samples = resampled(&SMOOTH, &lag_covariance);
        match damage {
            0 => data.samples.defined.truncate(7),
            1 => data.samples.values.truncate(3),
            _ => data.samples.dimension = 7,
        }
        assert_eq!(effective_mass(&data, EffectiveMassKind::LogRatio), delta);
    }
}

#[test]
fn the_window_scan_matches_the_python_aic_table_of_the_parity_fixture() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qft/non_involutive.json"
    ))
    .unwrap();
    let fixture: serde_json::Value = serde_json::from_str(&text).unwrap();
    let scan = &fixture["window_scan"];
    let numbers = |key: &str| -> Vec<f64> {
        scan[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let table = |key: &str| -> Vec<Vec<Option<f64>>> {
        scan[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64()).collect())
            .collect()
    };
    let (value, error) = (numbers("correlator"), numbers("error"));
    let valid = scan["point_valid"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v.as_u64() == Some(1))
        .count();
    assert_eq!(valid, 11);
    let n = value.len();
    let mut lag_covariance = vec![0.; n * n];
    for (i, e) in error.iter().enumerate() {
        lag_covariance[i * n + i] = e * e;
    }
    // Python validates a point by `error / C <= max_log_error = 0.5`, which is
    // the scan's `min_point_snr = 2`, and counts `N_cut` from lag 0.
    let config = AnalysisConfig {
        window_scan: WindowScanConfig {
            t_min: 0,
            min_points: 3,
            correlated: false,
            ..WindowScanConfig::default()
        },
        ..AnalysisConfig::default()
    };
    let fit = window_scan(&estimated(&value, lag_covariance), &config).unwrap();
    let (mass, variance) = (table("window_mass"), table("window_mass_variance"));
    let (chi2, aic) = (table("window_chi2"), table("window_aic"));
    let widths: Vec<usize> = numbers("window_widths")
        .iter()
        .map(|w| *w as usize)
        .collect();
    let mut checked = 0;
    for (index, width) in widths.iter().enumerate() {
        for (start, expected) in mass[index].iter().enumerate() {
            let Some(expected) = expected else { continue };
            let window = row(&fit.windows, start, start + width - 1);
            assert!(close(window.value, *expected, 1e-9));
            assert!(close(
                window.error * window.error,
                variance[index][start].unwrap(),
                1e-9
            ));
            assert!((window.chi2 - chi2[index][start].unwrap()).abs() < 1e-9);
            assert!(close(
                criterion(&window, valid),
                aic[index][start].unwrap(),
                1e-9
            ));
            checked += 1;
        }
    }
    assert_eq!(checked, scan["n_valid_windows"].as_u64().unwrap() as usize);
    // The Python `mass > 0` filter is inert here, so no window is dropped and
    // the scan reports a rate of the same size without any positivity cut.
    assert!(fit.windows.iter().all(|w| w.value > 0.));
    let rate = fit.mass.unwrap();
    assert!((rate.value - fixture["window_scan"]["mass"].as_f64().unwrap()).abs() < 0.02);
}

#[test]
fn effective_rates_follow_the_delta_method_on_the_lag_covariance() {
    let data = estimated(&SMOOTH, covariance(&SMOOTH_SIGMA, 0.8));
    let log_ratio = effective_mass(&data, EffectiveMassKind::LogRatio);
    assert_eq!(log_ratio.len(), 8);
    assert_eq!(log_ratio[7], None);
    for (got, want) in log_ratio.iter().zip(&LOG_RATIO) {
        let got = got.unwrap();
        assert!(close(got[0], want[0], 1e-12));
        assert!(close(got[1], want[1], 1e-12));
    }
    let cosh = effective_mass(&data, EffectiveMassKind::Cosh);
    assert_eq!(cosh[0], None);
    assert_eq!(cosh[7], None);
    for (got, want) in cosh[1..7].iter().zip(&COSH) {
        let got = got.unwrap();
        assert!(close(got[0], want[0], 1e-12));
        assert!(close(got[1], want[1], 1e-12));
    }
}

#[test]
fn effective_rates_divide_by_the_time_step_as_well_as_the_lag_spacing() {
    let lags: Vec<usize> = (0..8).collect();
    let lag_covariance = covariance(&SMOOTH_SIGMA, 0.8);
    let quarters = strided(&lags, &SMOOTH, lag_covariance.clone(), 0.25);
    // A lag of a quarter of a frame is four times the rate per frame, in the
    // value and in its error alike.
    for (got, want) in effective_mass(&quarters, EffectiveMassKind::LogRatio)
        .iter()
        .zip(&LOG_RATIO)
    {
        let got = got.unwrap();
        assert!(close(got[0], 4. * want[0], 1e-12));
        assert!(close(got[1], 4. * want[1], 1e-12));
    }
    for (got, want) in effective_mass(&quarters, EffectiveMassKind::Cosh)[1..7]
        .iter()
        .zip(&COSH)
    {
        let got = got.unwrap();
        assert!(close(got[0], 4. * want[0], 1e-12));
        assert!(close(got[1], 4. * want[1], 1e-12));
    }
    // The resampled arm carries the same step as the delta-method arm.
    let mut frames = estimated(&SMOOTH, lag_covariance.clone());
    frames.samples = resampled(&SMOOTH, &lag_covariance);
    let mut quarters = quarters;
    quarters.samples = resampled(&SMOOTH, &lag_covariance);
    let scaled = effective_mass(&quarters, EffectiveMassKind::LogRatio);
    let plain = effective_mass(&frames, EffectiveMassKind::LogRatio);
    for (a, b) in scaled[..7].iter().zip(&plain[..7]) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!(close(a[0], 4. * b[0], 1e-14));
        assert!(close(a[1], 4. * b[1], 1e-14));
    }
}

#[test]
fn effective_rate_errors_come_from_the_resamples_when_every_resample_is_defined() {
    let lag_covariance = covariance(&SMOOTH_SIGMA, 0.8);
    let mut data = estimated(&SMOOTH, lag_covariance.clone());
    let delta = effective_mass(&data, EffectiveMassKind::LogRatio);
    data.samples = resampled(&SMOOTH, &lag_covariance);
    let resample = effective_mass(&data, EffectiveMassKind::LogRatio);
    assert_eq!(resample[7], None);
    for (a, b) in resample[..7].iter().zip(&delta[..7]) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a[0] - b[0]).abs() < 1e-15);
        // The rate is not linear in the correlator, so the two errors differ
        // by the curvature over one standard deviation, about a percent here.
        assert!(a[1] > 0. && (a[1] - b[1]).abs() < 0.02 * b[1]);
    }
    // One resample that leaves the definition undefined sends the lag back to
    // the delta method rather than dropping it or inventing a spread.
    data.samples.values[3] = -1.;
    let mixed = effective_mass(&data, EffectiveMassKind::LogRatio);
    assert_eq!(mixed[3], delta[3]);
    assert_eq!(mixed[0], resample[0]);
}

#[test]
fn undefined_effective_rates_are_reported_as_missing_and_never_as_zero() {
    let mut value = SMOOTH;
    value[3] = -0.05;
    let data = estimated(&value, covariance(&SMOOTH_SIGMA, 0.8));
    let log_ratio = effective_mass(&data, EffectiveMassKind::LogRatio);
    assert_eq!(log_ratio[2], None);
    assert_eq!(log_ratio[3], None);
    assert!(log_ratio[4].is_some());
    let cosh = effective_mass(&data, EffectiveMassKind::Cosh);
    assert_eq!(cosh[3], None);
    // A ratio at or below 1 has an infinite arccosh error.
    let flat = [0.5, 0.5, 0.5, 0.5];
    let data = estimated(&flat, covariance(&[0.01; 4], 0.5));
    assert!(
        effective_mass(&data, EffectiveMassKind::Cosh)
            .iter()
            .all(Option::is_none)
    );
    assert!(effective_mass(&data, EffectiveMassKind::LogRatio)[1].unwrap()[0].abs() < 1e-15);
    // Lags that are not equally spaced leave the arccosh undefined while the
    // log ratio divides by the spacing it really has.
    let lags = [0, 1, 3, 4];
    let value = [1., 0.8, 0.5, 0.4];
    let data = strided(&lags, &value, covariance(&[0.01; 4], 0.5), 1.);
    assert_eq!(effective_mass(&data, EffectiveMassKind::Cosh)[1], None);
    let log_ratio = effective_mass(&data, EffectiveMassKind::LogRatio);
    assert!(close(
        log_ratio[1].unwrap()[0],
        (0.8_f64 / 0.5).ln() / 2.,
        1e-14
    ));
    assert!(close(log_ratio[2].unwrap()[0], (0.5_f64 / 0.4).ln(), 1e-14));
    // Lags the covariance gives no variance carry no error, and a rate
    // without an error is missing rather than exact.
    let data = estimated(&SMOOTH, vec![0.; SMOOTH.len() * SMOOTH.len()]);
    assert!(
        effective_mass(&data, EffectiveMassKind::LogRatio)
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn the_arccosh_rate_exists_only_at_or_above_a_ratio_of_one() {
    assert_eq!(cosh_rate(0.999999), None);
    assert_eq!(cosh_rate(1.), Some(0.));
    assert!(close(cosh_rate(0.4_f64.cosh()).unwrap(), 0.4, 1e-14));
}

/// Study: how often the one-sigma interval of the model-averaged rate holds
/// the decay rate the series was built with. It runs the whole chain, from the
/// frame series to the scan, on 300 seeded realisations per length, in under a
/// second in a debug build.
#[test]
fn the_window_scan_rate_covers_the_true_decay_of_a_seeded_ar1_ensemble() {
    // An AR(1) series with phi = exp(-0.30) has the autocovariance
    // `phi^tau / (1 - phi^2)`: one exponential of rate 0.30 per frame, with an
    // integrated autocorrelation time of (1 + phi) / (1 - phi) = 6.7 frames.
    let phi = (-0.30_f64).exp();
    let config = AnalysisConfig::default();
    for (frames, survivors, mean, coverage, spread) in [
        (120usize, 65usize, 0.261347, 0.4923, 1.3705),
        (600, 289, 0.292812, 0.6747, 0.9559),
    ] {
        let mut kept = 0.;
        let mut covered = 0;
        let mut quoted = 0.;
        let mut sum = 0.;
        let mut square = 0.;
        for replica in 0..300 {
            let series = ar1(1000 + replica, frames, phi);
            let data = estimate(
                &[scalar(6, &series)],
                "probe",
                EstimatorKind::FrameMean,
                &config,
                None,
            )
            .unwrap();
            let fit = window_scan(&data, &config).unwrap();
            if let Some(rate) = fit.mass {
                kept += 1.;
                sum += rate.value;
                square += rate.value * rate.value;
                quoted += rate.error;
                covered += usize::from((rate.value - 0.30).abs() <= rate.error);
            }
        }
        let measured = sum / kept;
        let scatter = (square / kept - measured * measured).sqrt();
        eprintln!(
            "frames {frames}: kept {kept} of 300, mean {measured:.6}, scatter {scatter:.6}, \
             quoted {:.6}, coverage {:.4}",
            quoted / kept,
            covered as f64 / kept
        );
        let held = covered as f64 / kept;
        assert_eq!(kept as usize, survivors);
        assert!((measured - mean).abs() < 1e-4, "{measured}");
        assert!((held - coverage).abs() < 1e-3, "{held}");
        assert!((scatter * kept / quoted - spread).abs() < 1e-3, "{scatter}");
        // Python kept 110 of 300 at 120 frames and its survivors averaged
        // 0.2412, 20 % below the truth, because it dropped every window whose
        // rate came out non-positive. No window is dropped here, so a series
        // too short to resolve a rate is declined instead of answered with one
        // biased low.
        assert!(measured > 0.2412, "{measured}");
        if frames < 600 {
            // 65 of 300 survive at 120 frames, and their scatter is 1.37 times
            // the quoted error: with an autocorrelation time of 6.7 frames a
            // series this short cannot support a rate, and the estimator says
            // so instead of answering.
            assert!(kept < 0.5 * 300.);
            continue;
        }
        // At 600 frames the ensemble is answered, almost unbiased, the quoted
        // error matches the scatter, and the one-sigma interval holds the truth
        // at the nominal rate. A coverage near 0.67 from 289 fits has a
        // binomial sd of 0.028, so [0.55, 0.80] is more than four sd wide on
        // the near side; the mean has a standard error of 0.0033.
        assert!(kept > 0.9 * 300.);
        assert!((measured - 0.30).abs() < 0.02, "{measured}");
        assert!((0.55..=0.80).contains(&held), "{held}");
    }
}

/// An AR(1) series `x_t = phi x_{t-1} + e_t` started in its stationary law.
fn ar1(seed: u64, frames: usize, phi: f64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut x = vec![rng.normal() / (1. - phi * phi).sqrt()];
    for t in 1..frames {
        x.push(phi * x[t - 1] + rng.normal());
    }
    x
}
/// One replica of one scalar channel over one segment, the shape
/// `estimators::estimate` reads a frame series from.
fn scalar(max_lag: usize, values: &[f64]) -> Measurement {
    let config = MeasurementConfig {
        max_lag,
        ..MeasurementConfig::default()
    };
    let gas = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    let capabilities = Capabilities::of(&gas, &RecordingConfig::default(), 3).refine(&config);
    let frames = values.len();
    Measurement {
        schema_version: SPECTROSCOPY_VERSION,
        fingerprint: config.fingerprint(&gas, &capabilities, &[]).unwrap(),
        capabilities,
        config,
        gas,
        walkers: 3,
        calibration: None,
        steps: (1..=frames as u64).collect(),
        segments: 1,
        segment: vec![0; frames],
        ingested: frames as u64,
        channels: vec![ChannelSeries {
            id: "probe".into(),
            spec: ChannelSpec::Custom { id: "probe".into() },
            kind: ElementKind::DistancePair,
            scale: None,
            definition: String::new(),
            book_label: String::new(),
            note: String::new(),
            exchange: ExchangeParity::Even,
            spatial_parity: None,
            correlatable: true,
            components: 1,
            availability: Availability::Available,
            coverage: Coverage::default(),
            involutive_frames: 0,
            values: values.to_vec(),
            weight: vec![1.; frames],
            propagator: None,
            euclidean: None,
        }],
        flow: None,
        notes: vec![],
    }
}
