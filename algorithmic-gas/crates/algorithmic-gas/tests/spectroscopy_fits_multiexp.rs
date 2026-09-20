use algorithmic_gas::{
    GasConfig, GasError, RecordingConfig,
    physics::{
        numerics::{ResampleKind, Samples, errors, sample_covariance},
        qft::math::Rng,
        spectroscopy::{
            AnalysisConfig,
            config::{
                ChannelGroup, ChannelSpec, Combine, MeasurementConfig, MultiExponentialConfig,
                StabilityScan, TimeUnit, WindowScanConfig,
            },
            contract::{
                Availability, Capabilities, ElementKind, ExchangeParity, SPECTROSCOPY_VERSION,
            },
            estimators::{Estimated, estimate},
            fits::{multi_exponential, stability::stability_scan},
            measurement::{ChannelSeries, Measurement},
            report::{
                CorrelatorEstimate, Coverage, EstimatorKind, FitMethodKind, RATE_QUANTITY,
                SamplesMeta,
            },
        },
    },
};
// Reference correlator 0.7 exp(-0.3 t) + 0.3 exp(-1.2 t) at t = 1..=12 plus correlated noise,
// with the lag covariance sigma_i sigma_j 0.6^|i-j|. Every fitted number quoted below comes from
// scipy.optimize.least_squares(method = "lm") on the same augmented residuals, cross-checked with
// lsqfit; they are independent of this crate.
const Y: [f64; 12] = [
    0.607756001510337,
    0.40909227610832,
    0.291479674032826,
    0.211699984223383,
    0.152148316299675,
    0.11198794564233,
    0.082550602767283,
    0.0617476130191936,
    0.0455554951887863,
    0.0338499637317231,
    0.0262995778816251,
    0.0196382570436904,
];
const SIGMA: [f64; 12] = [
    0.00297500009990642,
    0.00245484844850228,
    0.00213403549995025,
    0.00189887489556627,
    0.00170637138359428,
    0.00153964708367527,
    0.00139153310897461,
    0.0012585236546817,
    0.00113854313903858,
    0.00103011691519575,
    0.000932059076395571,
    0.00084335116640899,
];
/// Two operator variants of the same two states, measured over common blocks.
const Y_A: [f64; 8] = [
    0.605999127018749,
    0.410643989549884,
    0.291386809118012,
    0.211050911440536,
    0.154993435686855,
    0.113931164154812,
    0.0838450346523933,
    0.0618521482447156,
];
const Y_B: [f64; 8] = [
    0.41865115145118,
    0.191445883593272,
    0.106629086581926,
    0.0666074800399191,
    0.0469096373603237,
    0.033215006493661,
    0.023659777908054,
    0.0173402695256227,
];
const SIGMA_B: [f64; 8] = [
    0.00204823592264596,
    0.00114219162211419,
    0.000771888022669685,
    0.000602192336827955,
    0.000509481136966814,
    0.00044797281671122,
    0.000400551018109124,
    0.000360670836754226,
];
/// `sigma_i sigma_j rho^|i-j|`.
fn covariance(sigma: &[f64], rho: f64) -> Vec<f64> {
    let n = sigma.len();
    (0..n * n)
        .map(|c| sigma[c / n] * sigma[c % n] * rho.powi((c / n).abs_diff(c % n) as i32))
        .collect()
}
/// Lower Cholesky factor of a symmetric positive-definite `[n, n]` matrix.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.; n * n];
    for i in 0..n {
        for j in 0..=i {
            let overlap: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            l[i * n + j] = if i == j {
                (a[i * n + i] - overlap).sqrt()
            } else {
                (a[i * n + j] - overlap) / l[j * n + j]
            };
        }
    }
    l
}
/// A jackknife table of `2 d` resamples placed symmetrically along the columns
/// of the Cholesky factor, so that its mean is `central` and its resample
/// covariance is exactly `covariance`.
fn table(central: &[f64], covariance: &[f64]) -> Samples {
    let d = central.len();
    let l = cholesky(covariance, d);
    let spread = (d as f64 / (2 * d - 1) as f64).sqrt();
    let values: Vec<f64> = (0..2 * d)
        .flat_map(|r| (0..d).map(move |i| (r, i)))
        .map(|(r, i)| {
            let sign = if r.is_multiple_of(2) { 1. } else { -1. };
            central[i] + sign * spread * l[i * d + r / 2]
        })
        .collect();
    Samples {
        kind: ResampleKind::Jackknife,
        dimension: d,
        count: 2 * d,
        central: central.to_vec(),
        values,
        defined: vec![true; d],
        effective_block: 24,
        blocks: 2 * d,
        tau_int: Some(3.5),
    }
}
/// A frame-mean correlator over the lags `1..=values.len()` in frames.
fn estimated(channel: &str, values: &[f64], samples: Samples) -> Estimated {
    let n = values.len();
    Estimated {
        channel: channel.into(),
        kind: EstimatorKind::FrameMean,
        estimate: CorrelatorEstimate {
            lags: (1..=n).collect(),
            time_unit: TimeUnit::Frames,
            time_step: 1.,
            value: values.iter().map(|&v| Some(v)).collect(),
            error: errors(&samples),
            covariance: Some(sample_covariance(&samples)),
            samples_meta: SamplesMeta {
                resampling: samples.kind,
                effective_block: samples.effective_block,
                blocks: samples.blocks,
                tau_int: samples.tau_int,
                covariance_rank: n,
                replicas: 4,
                sampling_unit: SamplesMeta::sampling_unit(Combine::PooledBlocks, 24, 4),
            },
            connected: true,
            connected_bias: None,
        },
        samples,
    }
}
fn single_channel() -> Estimated {
    estimated(
        "meson/scalar/standard/distance",
        &Y,
        table(&Y, &covariance(&SIGMA, 0.6)),
    )
}
/// The two variants of the group, resampled over one common table so that
/// their joint covariance has the stated 0.5 correlation between channels.
fn group_channels() -> (Estimated, Estimated) {
    let sigma: Vec<f64> = SIGMA[..8].iter().chain(&SIGMA_B).copied().collect();
    let joint: Vec<f64> = (0..256)
        .map(|c| (c / 16, c % 16))
        .map(|(i, j)| {
            let block = if i / 8 == j / 8 { 1. } else { 0.5 };
            block * sigma[i] * sigma[j] * 0.6_f64.powi((i % 8).abs_diff(j % 8) as i32)
        })
        .collect();
    let central: Vec<f64> = Y_A.iter().chain(&Y_B).copied().collect();
    let shared = table(&central, &joint);
    let entries: Vec<usize> = (0..16).collect();
    (
        estimated(
            "meson/scalar/standard/distance",
            &Y_A,
            shared.select(&entries[..8]).unwrap(),
        ),
        estimated(
            "meson/scalar/score_directed/distance",
            &Y_B,
            shared.select(&entries[8..]).unwrap(),
        ),
    )
}
fn config(nexp: usize, t_min: usize, t_max: usize) -> MultiExponentialConfig {
    MultiExponentialConfig {
        nexp,
        t_min,
        t_max: Some(t_max),
        ..MultiExponentialConfig::default()
    }
}
/// The reference numbers were produced at the SVD cut 1e-6, which the AR(1)
/// covariances below leave inactive.
fn analysis(multi_exponential: MultiExponentialConfig) -> AnalysisConfig {
    AnalysisConfig {
        multi_exponential,
        svd_cut: 1e-6,
        ..AnalysisConfig::default()
    }
}
/// A long, cheap correlator with uncorrelated points: the scanned ranges are
/// bounded by the measured lags, so only a long channel can exceed the budget.
fn long_channel(lags: usize) -> Estimated {
    let values: Vec<f64> = (1..=lags).map(|t| 0.7 * (-0.3 * t as f64).exp()).collect();
    let sigma = vec![0.001; lags];
    estimated(
        "meson/scalar/standard/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
/// A flat correlator whose points carry a signal-to-noise ratio of about one.
fn white_noise() -> Estimated {
    let values: Vec<f64> = [1., 0.7, 1.3, 0.9, 1.1, 0.6, 1.4, 1., 0.8, 1.2, 0.95, 1.05]
        .iter()
        .map(|x| 0.01 * x)
        .collect();
    let sigma = [0.01; 12];
    estimated(
        "glueball/force_norm/site",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
/// The same correlator with the lag at index `k` taken out of every array: a
/// lag the fit refuses to read must come to exactly this fit.
fn without(data: &Estimated, k: usize) -> Estimated {
    let keep: Vec<usize> = (0..data.estimate.lags.len()).filter(|&i| i != k).collect();
    let samples = data.samples.select(&keep).unwrap();
    let mut out = data.clone();
    out.estimate.lags = keep.iter().map(|&i| data.estimate.lags[i]).collect();
    out.estimate.value = keep.iter().map(|&i| data.estimate.value[i]).collect();
    out.estimate.error = errors(&samples);
    out.estimate.covariance = Some(sample_covariance(&samples));
    out.samples = samples;
    out
}
/// The same correlator with the lag at index `k` marked undefined and its
/// error left stale, as a resample with a zero denominator leaves it.
fn undefined_at(data: &Estimated, k: usize) -> Estimated {
    let mut out = data.clone();
    out.samples.defined[k] = false;
    out.estimate.covariance = Some(sample_covariance(&out.samples));
    out
}
/// A correlator whose lag 4 carries no signal while the lags behind it do.
fn dropout_channel() -> Estimated {
    let mut values: Vec<f64> = (1..=8).map(|t| 0.7 * (-0.3 * t as f64).exp()).collect();
    values[3] = 0.001;
    let sigma = vec![0.001; 8];
    estimated(
        "meson/scalar/standard/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
/// A decaying correlator whose entry at lag 3 sits `z` standard deviations
/// below zero; every error is 0.01.
fn dip_at_lag_3(z: f64) -> Estimated {
    let mut values: Vec<f64> = (1..=12).map(|t| 0.5 * (-0.2 * t as f64).exp()).collect();
    values[2] = -0.01 * z;
    let sigma = [0.01; 12];
    estimated(
        "meson/scalar/standard/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
/// One clean exponential. Fitted with two states its second amplitude has no
/// data that fixes it, so the minimizer drifts along that direction and
/// exhausts its iterations while the ground gap and the probability stay
/// healthy: non-convergence is then the only thing withholding the rate.
fn one_state_over_two() -> Estimated {
    let values: Vec<f64> = (1..=12).map(|t| 0.7 * (-0.3 * t as f64).exp()).collect();
    let sigma = vec![1e-4; 12];
    estimated(
        "meson/scalar/standard/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
/// `0.6 exp(-0.3 t) + 0.3 exp(-t) + 0.1 exp(-2.5 t)`, measured precisely
/// enough that a three-state fit reports a ground level and two excited ones.
fn three_states() -> Estimated {
    let values: Vec<f64> = (1..=12)
        .map(|t| {
            let t = t as f64;
            0.6 * (-0.3 * t).exp() + 0.3 * (-t).exp() + 0.1 * (-2.5 * t).exp()
        })
        .collect();
    let sigma = vec![1e-3; 12];
    estimated(
        "meson/scalar/standard/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
/// `scale · 0.7 exp(-0.3 t)` whose first lag is a negative value its own
/// error admits. The amplitude prior has to read `ln |C(t_min)|` from it, so
/// the fit stays covariant under `C → scale C`.
fn negative_first_point(scale: f64) -> Estimated {
    let mut values: Vec<f64> = (1..=12).map(|t| 0.7 * (-0.3 * t as f64).exp()).collect();
    values[0] = -0.02;
    let values: Vec<f64> = values.iter().map(|v| v * scale).collect();
    let mut sigma = vec![0.002 * scale; 12];
    sigma[0] = 0.3 * scale;
    estimated(
        "meson/scalar/standard/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
/// A channel whose usable window ends at lag 5, with one strongly negative
/// entry at `lag` out in the noise beyond it.
fn late_negative(lag: usize) -> Estimated {
    let mut values: Vec<f64> = (1..=40)
        .map(|t| {
            if t <= 5 {
                0.7 * (-0.3 * t as f64).exp()
            } else {
                0.0005
            }
        })
        .collect();
    values[lag - 1] = -0.01;
    let sigma = vec![0.001; 40];
    estimated(
        "meson/scalar/standard/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    )
}
#[test]
fn one_state_reference_fit_reproduces_the_posterior_rate_and_its_prior_diagnostic() {
    let data = single_channel();
    let config = config(1, 5, 12);
    let outcome =
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config)).unwrap();
    assert_eq!(outcome.method, FitMethodKind::MultiExponential);
    assert!(outcome.excited.is_empty());
    let mass = outcome.mass.unwrap();
    assert_eq!(mass.quantity, RATE_QUANTITY);
    assert_eq!(mass.time_unit, TimeUnit::Frames);
    assert_eq!(mass.systematic, 0.);
    assert_eq!(mass.statistical, mass.error);
    assert!((mass.value - 0.297934984122792).abs() < 1e-9);
    assert!((mass.error - 0.00470831575453567).abs() < 1e-8 * 0.00470831575453567);
    let diagnostics = outcome.diagnostics;
    assert!((diagnostics.chi2.unwrap() - 3.77727268426804).abs() < 1e-8);
    assert_eq!(diagnostics.dof, Some(8));
    assert!((diagnostics.q.unwrap() - 0.876638780592568).abs() < 1e-9);
    assert_eq!(diagnostics.window, Some([5, 12]));
    assert_eq!(diagnostics.covariance_rank, Some(8));
    assert_eq!(diagnostics.n_windows, 1);
    assert!(diagnostics.correlated && diagnostics.svd_cut == 1e-6);
    assert!(diagnostics.model_rejected.is_none() && diagnostics.no_signal.is_none());
    let prior = diagnostics.prior_dominance.unwrap();
    assert!(!prior.dominated);
    assert!((prior.width_ratio - 0.00526772171272908).abs() < 1e-9);
    assert!((prior.shift_sigma - 0.363901700876431).abs() < 1e-9);
    assert_eq!(mass.prior_dominance, Some(prior));
    assert!(
        outcome
            .notes
            .iter()
            .any(|n| n.contains("prior part of 0.220935"))
    );
    assert!(
        outcome
            .notes
            .iter()
            .any(|n| n.contains("they read the data"))
    );
}
#[test]
fn an_open_upper_window_ends_at_the_last_lag_that_passes_the_point_threshold() {
    let data = single_channel();
    let open = MultiExponentialConfig {
        t_max: None,
        ..config(1, 5, 12)
    };
    let closed = multi_exponential::fit(
        std::slice::from_ref(&data),
        &config(1, 5, 12),
        &analysis(config(1, 5, 12)),
    )
    .unwrap();
    let scanned =
        multi_exponential::fit(std::slice::from_ref(&data), &open, &analysis(open)).unwrap();
    assert_eq!(scanned.diagnostics.window, Some([5, 12]));
    assert_eq!(scanned.mass, closed.mass);
    // Only the lags 11 and 12 remain above the start time, one fewer than the model needs.
    let short = MultiExponentialConfig {
        t_min: 11,
        t_max: None,
        ..config(1, 5, 12)
    };
    let empty =
        multi_exponential::fit(std::slice::from_ref(&data), &short, &analysis(short)).unwrap();
    assert!(empty.mass.is_none() && empty.excited.is_empty());
    assert_eq!(
        empty.diagnostics.no_signal.as_deref(),
        Some("too few usable lags")
    );
    assert!(empty.diagnostics.chi2.is_none());
    // A start time at the end of the integer range must not overflow the bound
    // of the sign-change search; it simply fits nothing.
    let far = MultiExponentialConfig {
        t_min: usize::MAX,
        ..short
    };
    let none = multi_exponential::fit(std::slice::from_ref(&data), &far, &analysis(far)).unwrap();
    assert_eq!(
        none.diagnostics.no_signal.as_deref(),
        Some("too few usable lags")
    );
}
#[test]
fn two_state_fit_recovers_both_levels_each_with_the_dominance_of_its_own_gap() {
    let data = single_channel();
    let config = config(2, 1, 12);
    let outcome =
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config)).unwrap();
    let mass = outcome.mass.unwrap();
    assert!((mass.value - 0.300974328832973).abs() < 1e-9);
    assert!((mass.error - 0.00516358135199234).abs() < 1e-8 * 0.00516358135199234);
    assert_eq!(outcome.excited.len(), 1);
    let excited = &outcome.excited[0];
    assert!((excited.value - 1.10759987793386).abs() < 1e-8);
    assert!((excited.error - 0.156002858822353).abs() < 1e-7 * 0.156002858822353);
    assert!((outcome.diagnostics.chi2.unwrap() - 9.91992978038108).abs() < 1e-8);
    assert_eq!(outcome.diagnostics.dof, Some(12));
    assert!((outcome.diagnostics.q.unwrap() - 0.622985133442628).abs() < 1e-9);
    let ground = mass.prior_dominance.unwrap();
    assert!((ground.width_ratio - 0.00571873950403468).abs() < 1e-9);
    assert!((ground.shift_sigma - 0.367284929618172).abs() < 1e-9);
    // The excited level carries the dominance of ln dE_1, not of the ground gap.
    let gap = excited.prior_dominance.unwrap();
    assert!((gap.width_ratio - 0.062781406410689).abs() < 1e-8);
    assert!((gap.shift_sigma - 0.695896457002141).abs() < 1e-8);
    assert!(!gap.dominated);
    // Both levels are the same measurement seen at two truncations of the model,
    // and `E_n = Σ_{m≤n} dE_m` with a positive gap puts every one of them above
    // the level below it: no reported spectrum can be inverted.
    assert!(outcome.excited.iter().all(|e| e.value > mass.value));
    assert!(outcome.excited.windows(2).all(|w| w[0].value < w[1].value));
}
#[test]
fn every_level_of_a_three_state_fit_lies_above_the_one_below_it() {
    let data = three_states();
    let config = config(3, 1, 12);
    let outcome =
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config)).unwrap();
    let mass = outcome.mass.unwrap();
    assert_eq!(outcome.excited.len(), 2);
    // The ground level is the rate the data were built with.
    assert!((mass.value - 0.3).abs() < 0.005, "{}", mass.value);
    // Levels are cumulative sums of positive gaps, so the whole ladder is
    // strictly increasing; a per-level rate would not have to be.
    let levels: Vec<f64> = std::iter::once(mass.value)
        .chain(outcome.excited.iter().map(|e| e.value))
        .collect();
    assert!(levels.windows(2).all(|w| w[1] > w[0]), "{levels:?}");
    assert!(outcome.excited.iter().all(|e| e.value > mass.value));
    // Every level carries the dominance of its own gap, so a ladder that is
    // ordered is also auditable level by level: here the third gap is the one
    // the data do not fix, and it says so without touching the two below it.
    let dominance: Vec<bool> = std::iter::once(&mass)
        .chain(&outcome.excited)
        .map(|level| level.prior_dominance.unwrap().dominated)
        .collect();
    assert_eq!(dominance, [false, false, true]);
}
#[test]
fn a_fit_that_did_not_converge_reports_no_rate() {
    let data = one_state_over_two();
    let config = config(2, 1, 12);
    let outcome =
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config)).unwrap();
    // Every other guard is open: the model is not rejected, the rate carries
    // signal and the prior does not dominate. The minimizer stopping short is
    // the only reason there is no number, and it is stated.
    assert!(outcome.diagnostics.model_rejected.is_none());
    assert!(outcome.diagnostics.no_signal.is_none());
    assert!(!outcome.diagnostics.prior_dominance.unwrap().dominated);
    assert!(outcome.diagnostics.q.is_some_and(|q| q > 0.99));
    assert!(outcome.diagnostics.chi2.is_some() && outcome.diagnostics.dof == Some(12));
    assert!(
        outcome
            .notes
            .iter()
            .any(|n| n == "the minimizer did not converge"),
        "{:?}",
        outcome.notes
    );
    assert!(outcome.mass.is_none() && outcome.excited.is_empty());
    // The stability scan reads the same flag: a row it could not minimize is
    // tabulated and said to be unconverged, and it is not an accepted variation.
    let scan = StabilityScan {
        t_min: [1, 1],
        nexp_max: 2,
        svd_cuts: vec![1e-6],
    };
    let table = stability_scan(&data, &scan, &analysis(config)).unwrap();
    assert!(
        table
            .notes
            .iter()
            .any(|n| n.starts_with("row 1:") && n.contains("converged=false")),
        "{:?}",
        table.notes
    );
    assert!(
        table
            .notes
            .iter()
            .any(|n| n.contains("stable: 1 of 1 accepted variations"))
    );
}
#[test]
fn a_negative_first_point_leaves_the_amplitude_prior_on_the_scale_of_the_data() {
    let config = config(1, 1, 12);
    let fit = |d: &Estimated| {
        multi_exponential::fit(std::slice::from_ref(d), &config, &analysis(config)).unwrap()
    };
    let natural = fit(&negative_first_point(1.));
    // A negative `C(t_min)` is a measurement, not a reason to fall back to a
    // fixed prior: the window keeps the lag and the fit runs on it.
    assert_eq!(natural.diagnostics.window, Some([1, 12]));
    assert!(natural.diagnostics.model_rejected.is_none());
    assert!(natural.diagnostics.no_signal.is_none());
    let mass = natural.mass.unwrap();
    assert!((mass.value - 0.3).abs() < 0.005, "{}", mass.value);
    // The amplitude prior is the only part of the fit that could carry an
    // absolute scale. Because its centre is `ln |C(t_min)|`, nine decades of
    // rescaling leave every fitted number where it was; a fixed `0.5(5)` prior
    // sits `ln(0.5 / 1.4e-11) = 24` wide priors away at the smaller scale.
    let tiny = fit(&negative_first_point(1e-9));
    let scaled = tiny.mass.unwrap();
    assert!((scaled.value - mass.value).abs() < 1e-12 * mass.value);
    assert!((scaled.error - mass.error).abs() < 1e-12 * mass.error);
    let (a, b) = (
        natural.diagnostics.chi2.unwrap(),
        tiny.diagnostics.chi2.unwrap(),
    );
    assert!((b - a).abs() < 1e-12 * a);
    let (a, b) = (
        natural.diagnostics.prior_dominance.unwrap(),
        tiny.diagnostics.prior_dominance.unwrap(),
    );
    assert!((b.width_ratio - a.width_ratio).abs() < 1e-12 * a.width_ratio);
    assert!((b.shift_sigma - a.shift_sigma).abs() < 1e-12 * a.shift_sigma.abs());
    assert!(tiny.notes.iter().any(|n| n.contains("they read the data")));
}
#[test]
fn a_duplicated_member_adds_no_information_to_a_group() {
    let (a, _) = group_channels();
    let config = config(2, 1, 8);
    let fit = |members: &[Estimated]| {
        multi_exponential::fit(members, &config, &analysis(config)).unwrap()
    };
    let alone = fit(std::slice::from_ref(&a));
    let twice = fit(&[a.clone(), a]);
    // The copy is perfectly correlated with the original, so the joint
    // covariance keeps the rank of one member and the SVD floor says so.
    assert_eq!(alone.diagnostics.covariance_rank, Some(8));
    assert_eq!(twice.diagnostics.covariance_rank, Some(8));
    assert!(
        twice
            .notes
            .iter()
            .any(|n| n.contains("the SVD floor is active")),
        "{:?}",
        twice.notes
    );
    // The rate does not move: the second copy carries no information. The
    // measured shift is 0.0012 of one standard error.
    let (one, two) = (alone.mass.unwrap(), twice.mass.unwrap());
    assert!((two.value - one.value).abs() < 0.01 * one.error);
    assert!((two.error - one.error).abs() < 1e-3 * one.error);
    // And it must not look like twice the data: the probability of the fit is
    // read at the rank of the covariance, not at the doubled point count.
    // Python's aggregate summed the two fits as if they were independent and
    // moved Q from 0.2126 to 0.1518 with logGBF doubling.
    assert_eq!(twice.diagnostics.dof, alone.diagnostics.dof);
    assert!(twice.diagnostics.q.unwrap() <= alone.diagnostics.q.unwrap());
}
#[test]
fn two_channels_sharing_their_gaps_recover_one_rate_and_sharpen_it() {
    let (a, b) = group_channels();
    let config = config(2, 1, 8);
    let group = ChannelGroup {
        id: "scalar".into(),
        channels: vec![a.channel.clone(), b.channel.clone()],
    };
    let members = [a.clone(), b];
    let joint = multi_exponential::fit_group(&group, &members, &analysis(config)).unwrap();
    assert_eq!(joint.id, "scalar");
    assert_eq!(joint.channels.len(), 2);
    assert!(joint.availability.is_available());
    assert_eq!(joint.levels.len(), 2);
    assert!((joint.levels[0].value - 0.306704448797).abs() < 1e-8);
    assert!((joint.levels[0].error - 0.003548545102).abs() < 1e-7 * 0.003548545102);
    assert!((joint.levels[1].value - 1.193014615806).abs() < 1e-7);
    assert!((joint.levels[1].error - 0.014524724752).abs() < 1e-6 * 0.014524724752);
    assert!((joint.diagnostics.chi2.unwrap() - 19.3954172936977).abs() < 1e-7);
    assert_eq!(joint.diagnostics.dof, Some(16));
    assert!((joint.diagnostics.q.unwrap() - 0.248692193024625).abs() < 1e-9);
    let ground = joint.levels[0].prior_dominance.unwrap();
    assert!((ground.width_ratio - 0.00385663908479393).abs() < 1e-9);
    assert!((ground.shift_sigma - 0.373571463850319).abs() < 1e-9);
    let gap = joint.levels[1].prior_dominance.unwrap();
    assert!((gap.width_ratio - 0.00512099244343857).abs() < 1e-9);
    assert!((gap.shift_sigma - 0.727298927159455).abs() < 1e-9);
    // The relative error of the ground level is the posterior width of ln dE_0.
    let alone = multi_exponential::fit(&members[..1], &config, &analysis(config))
        .unwrap()
        .mass
        .unwrap();
    assert!((alone.error / alone.value - 0.0235718011878448).abs() < 1e-8);
    assert!((joint.levels[0].error / joint.levels[0].value - 0.0115699172543818).abs() < 1e-8);
    assert!(joint.levels[0].error < alone.error);
    // The members of a group are a set. The deterministic start reads
    // `windows[0]`, and a start point is not a prior: listing them the other
    // way round has to give the same fit, not a fit seeded by whichever member
    // happens to come first.
    let swapped = multi_exponential::fit_group(
        &ChannelGroup {
            id: "scalar".into(),
            channels: vec![members[1].channel.clone(), members[0].channel.clone()],
        },
        &[members[1].clone(), members[0].clone()],
        &analysis(config),
    )
    .unwrap();
    assert_eq!(swapped.levels.len(), joint.levels.len());
    for (level, same) in joint.levels.iter().zip(&swapped.levels) {
        assert!((level.value - same.value).abs() < 1e-12 * level.value);
        assert!((level.error - same.error).abs() < 1e-12 * level.error);
    }
    assert_eq!(swapped.diagnostics.chi2, joint.diagnostics.chi2);
    assert_eq!(swapped.diagnostics.dof, joint.diagnostics.dof);
    assert_eq!(swapped.diagnostics.q, joint.diagnostics.q);
}
#[test]
fn white_noise_returns_its_prior_and_reports_no_rate() {
    let data = white_noise();
    let config = config(1, 1, 12);
    let outcome =
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config)).unwrap();
    let prior = outcome.diagnostics.prior_dominance.unwrap();
    assert!(prior.dominated && prior.width_ratio > 0.7);
    assert!(outcome.mass.is_none() && outcome.excited.is_empty());
    assert!(
        outcome
            .notes
            .iter()
            .any(|n| n.contains("ground gap prior-dominated"))
    );
    // A dominated fit still reports what it did, so the diagnostic is auditable.
    assert!(outcome.diagnostics.chi2.is_some() && outcome.diagnostics.dof == Some(12));
}
#[test]
fn a_sign_changing_correlator_rejects_the_exponential_model_without_a_rate() {
    let values: Vec<f64> = (0..12)
        .map(|t| 0.5 * (-0.2 * t as f64).exp() * if t % 2 == 0 { 1. } else { -1. })
        .collect();
    let sigma = [0.01; 12];
    let data = estimated(
        "vector/vector/full/raw/distance",
        &values,
        table(&values, &covariance(&sigma, 0.)),
    );
    let config = config(1, 1, 12);
    let outcome =
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config)).unwrap();
    assert_eq!(
        outcome.diagnostics.model_rejected.as_deref(),
        Some("sign change at lag 2")
    );
    assert!(outcome.mass.is_none() && outcome.excited.is_empty());
    assert!(outcome.diagnostics.chi2.is_none() && outcome.diagnostics.prior_dominance.is_none());
}
#[test]
fn relabelling_the_time_unit_divides_the_rate_and_leaves_the_fit_untouched() {
    let frames = single_channel();
    let mut rescaled = frames.clone();
    rescaled.estimate.time_unit = TimeUnit::StepDt;
    rescaled.estimate.time_step = 0.01;
    let config = config(2, 1, 12);
    let plain =
        multi_exponential::fit(std::slice::from_ref(&frames), &config, &analysis(config)).unwrap();
    let other = multi_exponential::fit(std::slice::from_ref(&rescaled), &config, &analysis(config))
        .unwrap();
    let (a, b) = (plain.mass.unwrap(), other.mass.unwrap());
    assert_eq!(b.time_unit, TimeUnit::StepDt);
    assert!((b.value - 100. * a.value).abs() < 1e-9 * b.value);
    assert!((b.error - 100. * a.error).abs() < 1e-9 * b.error);
    assert_eq!(plain.diagnostics.chi2, other.diagnostics.chi2);
    assert_eq!(plain.diagnostics.q, other.diagnostics.q);
    assert_eq!(
        plain.diagnostics.prior_dominance,
        other.diagnostics.prior_dominance
    );
    let excited = other.excited[0].value;
    assert!((excited - 100. * plain.excited[0].value).abs() < 1e-12 * excited);
}
#[test]
fn the_stability_scan_tabulates_every_start_time_and_weights_it_by_its_information_criterion() {
    let data = single_channel();
    let reference = config(1, 5, 12);
    let scan = StabilityScan {
        t_min: [1, 6],
        nexp_max: 2,
        svd_cuts: vec![1e-6],
    };
    let outcome = stability_scan(&data, &scan, &analysis(reference)).unwrap();
    assert_eq!(outcome.method, FitMethodKind::Stability);
    assert!(outcome.mass.is_none() && outcome.excited.is_empty());
    assert_eq!(outcome.windows.len(), 12);
    assert_eq!(outcome.diagnostics.n_windows, 12);
    // The loop is deterministic: every start time of one exponential, then of two.
    // Rates, errors and augmented chi2 of all twelve rows come from the scipy
    // reference table; the two-state rows above t_min = 1 are the ones a start
    // point or a stopping rule of its own would move.
    let rates = [
        0.339284969937688,
        0.318333329732775,
        0.310689315696988,
        0.306763460587287,
        0.297934984122792,
        0.294410618384104,
        0.300974328832973,
        0.255690892827192,
        0.210652992778699,
        0.277730231394455,
        0.202123444559918,
        0.194985921082771,
    ];
    let widths = [
        0.00202959397044966,
        0.00242663986804239,
        0.0029944583781607,
        0.00374982635451561,
        0.00470831575453567,
        0.00600939419633027,
        0.00516358135199234,
        0.0529181366796303,
        0.138698252942103,
        0.0277220139531375,
        0.195904947889078,
        0.203150007796837,
    ];
    // Each entry is the reference chi2_data plus its chi2_prior.
    let augmented = [
        209.267067452529 + 0.170083581931515,
        31.5954160866318 + 0.164928145837604,
        15.016726599662 + 0.177274204168323,
        12.2635593417802 + 0.199421573916614,
        3.55633731547114 + 0.220935368796893,
        2.71352058498332 + 0.254098952203157,
        9.27951392746472 + 0.640415852899023,
        4.01850172835813 + 0.145254101721506,
        3.53635822909567 + 0.128283572040262,
        2.0474708798651 + 0.374167252955787,
        1.56054688035967 + 0.156929740491439,
        1.54885104324133 + 0.195091499265178,
    ];
    for ((row, want), width) in outcome.windows.iter().zip(rates).zip(widths) {
        assert!((row.value - want).abs() < 1e-7 * want);
        assert!((row.error - width).abs() < 1e-7 * width);
    }
    for (row, want) in outcome.windows.iter().zip(augmented) {
        assert!((row.chi2 - want).abs() < 1e-7 * want);
    }
    for (index, row) in outcome.windows.iter().enumerate() {
        assert_eq!(row.t_min, 1 + index % 6);
        assert_eq!(row.nexp, 1 + index / 6);
        assert_eq!(row.svd_cut, Some(1e-6));
        assert_eq!(row.t_max, 12);
        assert_eq!(row.dof, 13 - row.t_min);
    }
    assert!((outcome.windows[4].chi2 - 3.77727268426804).abs() < 1e-8);
    assert!((outcome.windows[6].chi2 - 9.91992978038108).abs() < 1e-8);
    let total: f64 = outcome.windows.iter().map(|row| row.weight).sum();
    assert!((total - 1.).abs() < 1e-12);
    // AIC = augmented chi2 + 2k + 2 N_cut against the widest window of 12 points.
    let aic =
        |chi2: f64, parameters: usize, points: usize| chi2 + 2. * (parameters + 12 - points) as f64;
    let ratio = (-0.5 * (aic(3.77727268426804, 2, 8) - aic(9.91992978038108, 4, 12))).exp();
    let got = outcome.windows[4].weight / outcome.windows[6].weight;
    assert!((got - ratio).abs() < 1e-7 * ratio);
    assert!(
        outcome
            .notes
            .iter()
            .any(|n| n.starts_with("row 0: converged=true"))
    );
    assert!(
        outcome
            .notes
            .iter()
            .any(|n| n.contains("5 of 7 accepted variations agree with the reference"))
    );
    assert!(outcome.notes.iter().any(|n| n.contains("t_min=3 nexp=1")));
}
#[test]
fn a_scan_whose_start_times_exceed_the_measured_lags_tabulates_nothing() {
    let data = single_channel();
    let reference = config(1, 5, 12);
    let scan = StabilityScan {
        t_min: [40, 64],
        nexp_max: 1,
        svd_cuts: vec![],
    };
    let outcome = stability_scan(&data, &scan, &analysis(reference)).unwrap();
    assert!(outcome.windows.is_empty() && outcome.diagnostics.n_windows == 0);
    assert!(
        outcome
            .notes
            .iter()
            .any(|n| n.contains("0 of 0 accepted variations"))
    );
}
#[test]
fn malformed_fits_and_mismatched_members_are_rejected_explicitly() {
    let data = single_channel();
    let config = config(1, 5, 12);
    let analysis = analysis(config);
    assert!(matches!(
        multi_exponential::fit(&[], &config, &analysis),
        Err(GasError::Capability(_))
    ));
    let broken = MultiExponentialConfig { nexp: 0, ..config };
    assert!(matches!(
        multi_exponential::fit(std::slice::from_ref(&data), &broken, &analysis),
        Err(GasError::Configuration(_))
    ));
    let mut other = data.clone();
    other.estimate.time_step = 0.5;
    assert!(matches!(
        multi_exponential::fit(&[data.clone(), other.clone()], &config, &analysis),
        Err(GasError::Configuration(_))
    ));
    let mut stopped = data.clone();
    stopped.estimate.time_step = 0.;
    assert!(matches!(
        multi_exponential::fit(std::slice::from_ref(&stopped), &config, &analysis),
        Err(GasError::Configuration(_))
    ));
    let group = ChannelGroup {
        id: "scalar".into(),
        channels: vec![data.channel.clone(), other.channel.clone()],
    };
    assert!(matches!(
        multi_exponential::fit_group(&group, std::slice::from_ref(&data), &analysis),
        Err(GasError::Configuration(_))
    ));
    let unnamed = ChannelGroup {
        id: String::new(),
        ..group
    };
    assert!(matches!(
        multi_exponential::fit_group(&unnamed, &[data.clone(), other], &analysis),
        Err(GasError::Configuration(_))
    ));
    let scan = StabilityScan {
        nexp_max: 9,
        ..StabilityScan::default()
    };
    assert!(matches!(
        stability_scan(&data, &scan, &analysis),
        Err(GasError::Configuration(_))
    ));
    // The scan reads the configured fit, so an unusable one is rejected rather
    // than tabulated with a model the fit itself would refuse.
    let invalid = AnalysisConfig {
        multi_exponential: MultiExponentialConfig { nexp: 5, ..config },
        ..analysis.clone()
    };
    assert!(matches!(
        stability_scan(&data, &StabilityScan::default(), &invalid),
        Err(GasError::Configuration(_))
    ));
}
#[test]
fn a_correlator_whose_arrays_disagree_with_its_lags_is_rejected_instead_of_indexed() {
    let config = config(1, 5, 12);
    let analysis = analysis(config);
    let fit =
        |data: &Estimated| multi_exponential::fit(std::slice::from_ref(data), &config, &analysis);
    let mut truncated = single_channel();
    truncated.estimate.value.pop();
    assert!(matches!(fit(&truncated), Err(GasError::Configuration(_))));
    let mut shortened = single_channel();
    shortened.samples = shortened.samples.select(&[0, 1, 2, 3]).unwrap();
    assert!(matches!(fit(&shortened), Err(GasError::Configuration(_))));
    let mut shuffled = single_channel();
    shuffled.estimate.lags.swap(0, 1);
    assert!(matches!(fit(&shuffled), Err(GasError::Configuration(_))));
    // An infinite step would divide every rate to zero instead of failing.
    let mut unbounded = single_channel();
    unbounded.estimate.time_step = f64::INFINITY;
    assert!(matches!(fit(&unbounded), Err(GasError::Configuration(_))));
}
#[test]
fn a_stability_scan_beyond_the_variation_budget_is_rejected_before_any_fit() {
    let data = long_channel(100);
    let analysis = analysis(config(1, 5, 12));
    let scan = StabilityScan {
        t_min: [1, 4096],
        nexp_max: 4,
        svd_cuts: vec![1e-8; 16],
    };
    let Err(GasError::Configuration(message)) = stability_scan(&data, &scan, &analysis) else {
        panic!("a 6400-variation scan must be refused");
    };
    assert!(message.contains("4096"));
    // The same data with one cut and one model stay inside the budget.
    let narrow = StabilityScan {
        t_min: [1, 2],
        nexp_max: 1,
        svd_cuts: vec![1e-6],
    };
    let outcome = stability_scan(&data, &narrow, &analysis).unwrap();
    assert_eq!(outcome.windows.len(), 2);
}
#[test]
fn moving_or_widening_the_gap_prior_leaves_a_data_dominated_rate_in_place() {
    let data = single_channel();
    let base = config(1, 5, 12);
    let fitted = |c: MultiExponentialConfig| {
        multi_exponential::fit(std::slice::from_ref(&data), &c, &analysis(c))
            .unwrap()
            .mass
            .unwrap()
    };
    let reference = fitted(base);
    // Audit D2: a scale-free prior may not set the answer. Moving its centre by
    // one prior width, or doubling that width, must move a data-dominated rate
    // by far less than half of its own posterior width.
    for varied in [
        MultiExponentialConfig {
            log_gap_mean: base.log_gap_mean + base.log_gap_sigma,
            ..base
        },
        MultiExponentialConfig {
            log_gap_mean: base.log_gap_mean - base.log_gap_sigma,
            ..base
        },
        MultiExponentialConfig {
            log_gap_sigma: 2. * base.log_gap_sigma,
            ..base
        },
    ] {
        let moved = fitted(varied);
        assert!((moved.value - reference.value).abs() < 0.5 * reference.error);
    }
}
#[test]
fn a_lag_without_a_usable_error_is_dropped_exactly_as_if_it_had_not_been_measured() {
    let data = single_channel();
    let config = config(1, 5, 12);
    let fit = |d: &Estimated| {
        multi_exponential::fit(std::slice::from_ref(d), &config, &analysis(config)).unwrap()
    };
    // Lag 7 taken out of the correlator: seven fitted points instead of eight.
    let dropped = fit(&without(&data, 6));
    assert_eq!(dropped.diagnostics.dof, Some(7));
    assert_eq!(dropped.diagnostics.window, Some([5, 12]));
    assert!(dropped.mass.as_ref().unwrap().value != fit(&data).mass.unwrap().value);
    // Undefined resamples with a stale error beside them, a missing error and
    // a zero error must each reproduce that fit, note for note.
    let stale = undefined_at(&data, 6);
    assert!(stale.estimate.error[6].is_some_and(|e| e > 0.));
    assert_eq!(fit(&stale), dropped);
    let mut missing = data.clone();
    missing.estimate.error[6] = None;
    assert_eq!(fit(&missing), dropped);
    let mut zero = data.clone();
    zero.estimate.error[6] = Some(0.);
    assert_eq!(fit(&zero), dropped);
}
#[test]
fn an_open_window_stops_at_the_first_unusable_lag_instead_of_skipping_over_it() {
    let data = dropout_channel();
    let open = MultiExponentialConfig {
        t_max: None,
        ..config(1, 1, 8)
    };
    let scanned =
        multi_exponential::fit(std::slice::from_ref(&data), &open, &analysis(open)).unwrap();
    assert_eq!(scanned.diagnostics.window, Some([1, 3]));
    assert_eq!(scanned.diagnostics.dof, Some(3));
    assert!(scanned.mass.is_some() && scanned.diagnostics.model_rejected.is_none());
    // The usable range is the contiguous one: it is the window [1, 3], not the
    // seven lags that pass the threshold one by one.
    let stopped = multi_exponential::fit(
        std::slice::from_ref(&data),
        &config(1, 1, 3),
        &analysis(config(1, 1, 3)),
    )
    .unwrap();
    assert_eq!(scanned, stopped);
    let whole = multi_exponential::fit(
        std::slice::from_ref(&data),
        &config(1, 1, 8),
        &analysis(config(1, 1, 8)),
    )
    .unwrap();
    assert_eq!(whole.diagnostics.dof, Some(8));
    assert!(whole.diagnostics.model_rejected.is_some() && whole.mass.is_none());
}
#[test]
fn the_model_is_rejected_only_by_a_clearly_negative_lag_within_reach_of_the_window() {
    let config = config(1, 1, 12);
    let fit = |d: &Estimated| {
        multi_exponential::fit(std::slice::from_ref(d), &config, &analysis(config)).unwrap()
    };
    // Two standard deviations below zero are noise: the fit runs, and whatever
    // its probability then says, no lag contradicted a sum of exponentials.
    let noise = fit(&dip_at_lag_3(2.));
    assert!(noise.diagnostics.chi2.is_some());
    assert!(
        !noise
            .diagnostics
            .model_rejected
            .is_some_and(|reason| reason.contains("sign change"))
    );
    // Four of them are a sign change, and it is decided before any fit.
    let negative = fit(&dip_at_lag_3(4.));
    assert_eq!(
        negative.diagnostics.model_rejected.as_deref(),
        Some("sign change at lag 3")
    );
    assert!(negative.diagnostics.chi2.is_none());
    // Out beyond twice the end of the fitted window the correlator is noise,
    // and a negative entry there says nothing about the states fitted below it.
    let open = MultiExponentialConfig::default();
    let fit = |d: &Estimated| {
        multi_exponential::fit(std::slice::from_ref(d), &open, &analysis(open)).unwrap()
    };
    let far = fit(&late_negative(20));
    assert_eq!(far.diagnostics.window, Some([1, 5]));
    assert!(far.diagnostics.model_rejected.is_none() && far.mass.is_some());
    assert_eq!(
        fit(&late_negative(10))
            .diagnostics
            .model_rejected
            .as_deref(),
        Some("sign change at lag 10")
    );
}
#[test]
fn a_rate_is_judged_by_the_rate_threshold_and_a_lag_by_the_point_threshold() {
    let data = single_channel();
    let config = config(1, 5, 12);
    let scanned = |window_scan| {
        let analysis = AnalysisConfig {
            window_scan,
            ..analysis(config)
        };
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis).unwrap()
    };
    let refused = scanned(WindowScanConfig {
        min_rate_snr: 100.,
        ..WindowScanConfig::default()
    });
    assert_eq!(
        refused.diagnostics.no_signal.as_deref(),
        Some("rate S/N below threshold")
    );
    assert!(refused.mass.is_none() && refused.excited.is_empty());
    // The refused fit still states what it did.
    assert!((refused.diagnostics.chi2.unwrap() - 3.77727268426804).abs() < 1e-8);
    // A point threshold no lag could pass leaves a closed window alone: the
    // configured range is the window, whatever the scan would have kept.
    let kept = scanned(WindowScanConfig {
        min_point_snr: 1e9,
        ..WindowScanConfig::default()
    });
    assert!(kept.diagnostics.no_signal.is_none());
    assert_eq!(
        kept.mass,
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config))
            .unwrap()
            .mass
    );
}
#[test]
fn an_active_svd_floor_deflates_chi2_and_is_reported_as_such() {
    let data = single_channel();
    let config = config(1, 5, 12);
    let floored = AnalysisConfig {
        svd_cut: 0.5,
        ..analysis(config)
    };
    let cut = multi_exponential::fit(std::slice::from_ref(&data), &config, &floored).unwrap();
    assert_eq!(cut.diagnostics.covariance_rank, Some(2));
    assert_eq!(cut.diagnostics.svd_cut, 0.5);
    // Flooring the eigenvalues only enlarges variances, so chi2 falls.
    assert!(cut.diagnostics.chi2.unwrap() < 3.77727268426804);
    assert!(
        cut.notes
            .iter()
            .any(|n| n.contains("the SVD floor is active"))
    );
    // A deflated chi2 read against the eight fitted points would be a
    // probability of 0.98 for a fit the data constrain in two directions. Six
    // of the eight directions were raised to the floor and carry no
    // constraint, so the degrees of freedom are the rank and the probability
    // is 0.38; the note says exactly that.
    assert_eq!(cut.diagnostics.dof, Some(2));
    assert!((cut.diagnostics.q.unwrap() - 0.3770418596731393).abs() < 1e-9);
    assert!(
        cut.notes
            .iter()
            .any(|n| n
                .contains("the degrees of freedom count the rank rather than the fitted points")),
        "{:?}",
        cut.notes
    );
    let plain =
        multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis(config)).unwrap();
    assert_eq!(plain.diagnostics.covariance_rank, Some(8));
    // With no floor active the rank is the point count, so nothing is deflated.
    assert_eq!(plain.diagnostics.dof, Some(8));
    assert!(!plain.notes.iter().any(|n| n.contains("SVD floor")));
    // Twenty-four blocks support the eight fitted points, so nothing is said
    // about the resample count. Six do not: five delete-one replicas cannot
    // span eight lags, and a floor that is then active is fabricating the
    // constraint rather than regulating a measured one.
    assert!(
        !plain
            .notes
            .iter()
            .any(|n| n.contains("independent replicas against"))
    );
    let mut thin = data.clone();
    thin.samples.blocks = 6;
    let fitted = multi_exponential::fit(std::slice::from_ref(&thin), &config, &floored).unwrap();
    assert!(
        fitted.notes.iter().any(|n| n
            == "6 resampling blocks leave 5 independent replicas against 8 fitted points: the \
                lag covariance is rank deficient by construction"),
        "{:?}",
        fitted.notes
    );
}
#[test]
fn a_group_fits_each_member_over_its_own_lags_and_reports_the_widest_window() {
    let (a, b) = group_channels();
    let group = ChannelGroup {
        id: "scalar".into(),
        channels: vec![a.channel.clone(), b.channel.clone()],
    };
    let config = config(2, 1, 8);
    let fitted =
        |members: &[Estimated]| multi_exponential::fit_group(&group, members, &analysis(config));
    // The second member loses its last lag, the first keeps eight.
    let masked = fitted(&[a.clone(), undefined_at(&b, 7)]).unwrap();
    assert_eq!(masked.diagnostics.window, Some([1, 8]));
    assert_eq!(masked.diagnostics.dof, Some(15));
    assert_eq!(masked, fitted(&[a.clone(), without(&b, 7)]).unwrap());
    let whole = fitted(&[a, b]).unwrap();
    assert_eq!(whole.diagnostics.dof, Some(16));
    assert!(masked.levels[0].value != whole.levels[0].value);
}
#[test]
fn two_svd_cuts_are_two_likelihoods_and_each_is_weighted_within_itself() {
    let data = single_channel();
    let scan = StabilityScan {
        t_min: [1, 6],
        nexp_max: 2,
        svd_cuts: vec![1e-6, 1e-2],
    };
    let outcome = stability_scan(&data, &scan, &analysis(config(1, 5, 12))).unwrap();
    assert_eq!(outcome.windows.len(), 24);
    let (first, second) = outcome.windows.split_at(12);
    assert!(first.iter().all(|row| row.svd_cut == Some(1e-6)));
    assert!(second.iter().all(|row| row.svd_cut == Some(1e-2)));
    // Neither floor is active on these covariances, so the two halves are the
    // same twelve fits and carry the same weights.
    for (row, same) in first.iter().zip(second) {
        assert_eq!(
            (row.value, row.chi2, row.weight),
            (same.value, same.chi2, same.weight)
        );
    }
    for half in [first, second] {
        let total: f64 = half.iter().map(|row| row.weight).sum();
        assert!((total - 1.).abs() < 1e-12);
    }
}
#[test]
fn repeated_fits_of_the_same_data_are_bitwise_identical() {
    let data = single_channel();
    let config = config(2, 1, 12);
    let analysis = analysis(config);
    let once = multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis).unwrap();
    let twice = multi_exponential::fit(std::slice::from_ref(&data), &config, &analysis).unwrap();
    assert_eq!(once, twice);
    assert_eq!(
        once.mass.unwrap().value.to_bits(),
        twice.mass.unwrap().value.to_bits()
    );
}
/// Study: how often the one-sigma interval of the fitted rate holds the decay
/// rate the series was built with. It runs the whole chain, from the frame
/// series to the fit, on 300 seeded realisations per length, in under a second
/// in a debug build.
#[test]
fn the_multi_exponential_rate_covers_the_true_decay_of_a_seeded_ar1_ensemble() {
    // An AR(1) series with phi = exp(-0.30) has the autocovariance
    // `phi^tau / (1 - phi^2)`: one exponential of rate 0.30 per frame, with an
    // integrated autocorrelation time of (1 + phi) / (1 - phi) = 6.7 frames.
    let phi = (-0.30_f64).exp();
    let analysis = AnalysisConfig::default();
    for (frames, survivors, mean, coverage) in [
        (120usize, 164usize, 0.318649, 0.6159),
        (600, 299, 0.313998, 0.6288),
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
                &analysis,
                None,
            )
            .unwrap();
            let fit =
                multi_exponential::fit(&[data], &analysis.multi_exponential, &analysis).unwrap();
            if let Some(rate) = fit.mass {
                kept += 1.;
                sum += rate.value;
                square += rate.value * rate.value;
                quoted += rate.error;
                covered += usize::from((rate.value - 0.30).abs() <= rate.error);
            }
        }
        let measured = sum / kept;
        eprintln!(
            "frames {frames}: kept {kept} of 300, mean {measured:.6}, scatter {:.6}, quoted \
             {:.6}, coverage {:.4}",
            (square / kept - measured * measured).sqrt(),
            quoted / kept,
            covered as f64 / kept
        );
        // The seeded ensemble gives the numbers in the loop above. A coverage
        // near 0.62 from ~300 fits has a binomial sd of 0.028, so the band
        // [0.55, 0.80] is more than two sd wide on the near side; the mean has
        // a standard error of 0.009 at 120 frames and 0.0036 at 600.
        let held = covered as f64 / kept;
        assert_eq!(kept as usize, survivors);
        assert!((measured - mean).abs() < 1e-4, "{measured}");
        assert!((measured - 0.30).abs() < 0.02, "{measured}");
        assert!((held - coverage).abs() < 1e-3, "{held}");
        assert!((0.55..=0.80).contains(&held), "{held}");
        // Python kept 110 of 300 at 120 frames and its survivors averaged
        // 0.2412, 20 % below the truth, because it dropped every window whose
        // rate came out non-positive and clamped what was left. Nothing is
        // dropped for the rate it gives here, and the bias is upward and
        // smaller: the prior-dominance gate, not a sign filter, decides.
        assert!(measured > 0.2412, "{measured}");
        assert!(kept > 0.5 * 300.);
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
