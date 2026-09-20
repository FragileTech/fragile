use algorithmic_gas::{
    GasConfig, GasError, RecordingConfig,
    physics::{
        numerics::{
            BlockMoments, BlockSize, ResampleKind, Resampling, Samples, Subtraction, errors,
        },
        qft::math::Rng,
        spectroscopy::{
            config::{
                AnalysisConfig, ChannelSpec, Combine, MeasurementConfig, PropagatorConfig,
                TimeAxis, TimeUnit,
            },
            contract::{
                Availability, Capabilities, EXCHANGE_ODD_REASON, ElementKind, ExchangeParity,
                SPECTROSCOPY_VERSION,
            },
            estimators::{
                DELETION_GEOMETRY_NOTE, Estimated, estimate, joint, matrix, moments,
                resample_support_note,
            },
            measurement::{ChannelSeries, Measurement, SlabMoments},
            report::{
                Calibration, Coverage, EstimatorKind, RATE_QUANTITY, RATE_QUANTITY_EUCLIDEAN,
                RATE_QUANTITY_SOURCE_FROZEN,
            },
        },
    },
};

/// The eight-frame record of the dossier, an oracle of every frame-average sum.
const D: [f64; 8] = [1., 2., 4., 3., 0., 5., 2., 1.];

fn close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol * e.abs().max(1.),
            "entry {i}: {a} vs {e}"
        );
    }
}
fn some(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().map(|x| x.unwrap()).collect()
}
fn jackknife(frames: usize) -> Resampling {
    Resampling::BlockJackknife {
        block: BlockSize::Fixed { frames },
    }
}
fn analysis(resampling: Resampling, subtraction: Subtraction) -> AnalysisConfig {
    AnalysisConfig {
        resampling,
        frame_subtraction: subtraction,
        propagator_subtraction: subtraction,
        ..AnalysisConfig::default()
    }
}
/// A retained frame series of `frames` frames, unit weight unless masked.
fn series(id: &str, components: usize, values: &[f64]) -> ChannelSeries {
    let frames = values.len() / components;
    ChannelSeries {
        id: id.into(),
        spec: ChannelSpec::Custom { id: "probe".into() },
        kind: ElementKind::DistancePair,
        scale: None,
        definition: String::new(),
        book_label: String::new(),
        note: String::new(),
        exchange: ExchangeParity::Even,
        spatial_parity: None,
        correlatable: true,
        components,
        availability: Availability::Available,
        coverage: Coverage::default(),
        involutive_frames: 0,
        values: values.to_vec(),
        weight: vec![1.; frames],
        propagator: None,
        euclidean: None,
    }
}
fn measurement(
    config: MeasurementConfig,
    channels: Vec<ChannelSeries>,
    segment: &[u32],
) -> Measurement {
    let gas = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    let capabilities = Capabilities::of(&gas, &RecordingConfig::default(), 3).refine(&config);
    let frames = segment.len();
    Measurement {
        schema_version: SPECTROSCOPY_VERSION,
        fingerprint: config.fingerprint(&gas, &capabilities, &[]).unwrap(),
        capabilities,
        config,
        gas,
        walkers: 3,
        calibration: None,
        steps: (1..=frames as u64).collect(),
        segments: segment.last().map_or(0, |s| s + 1),
        segment: segment.to_vec(),
        ingested: frames as u64,
        channels,
        flow: None,
        notes: vec![],
    }
}
/// One replica of one scalar channel over one segment.
fn scalar(max_lag: usize, values: &[f64]) -> Measurement {
    let config = MeasurementConfig {
        max_lag,
        ..MeasurementConfig::default()
    };
    let segment = vec![0; values.len()];
    measurement(config, vec![series("probe", 1, values)], &segment)
}
fn central(data: &Estimated) -> Vec<Option<f64>> {
    data.estimate.value.clone()
}
/// Per-origin propagator sums of the six-frame record of the dossier, under
/// `IdentityPolicy::Slot`: `n | ab | a | b` over lags 0..3.
const PROPAGATOR: [[[f64; 4]; 4]; 6] = [
    [
        [3., 1., 3., 3.],
        [49., 4., 48., 12.],
        [11., 2., 11., 11.],
        [11., 2., 11., 2.],
    ],
    [
        [2., 2., 2., 0.],
        [8., 12., 0., 0.],
        [4., 4., 4., 0.],
        [4., 6., 0., 0.],
    ],
    [
        [3., 3., 0., 0.],
        [44., 12., 0., 0.],
        [10., 10., 0., 0.],
        [10., 2., 0., 0.],
    ],
    [
        [3., 0., 0., 0.],
        [4., 0., 0., 0.],
        [2., 0., 0., 0.],
        [2., 0., 0., 0.],
    ],
    [
        [3., 3., 0., 0.],
        [49., 35., 0., 0.],
        [11., 11., 0., 0.],
        [11., 9., 0., 0.],
    ],
    [
        [3., 0., 0., 0.],
        [33., 0., 0., 0.],
        [9., 0., 0., 0.],
        [9., 0., 0., 0.],
    ],
];
fn propagator() -> BlockMoments {
    strided_propagator(1)
}
/// The same record stored as the accumulator stores it under a lag stride: one
/// entry per frame lag, empty where no sink was evaluated.
fn strided_propagator(stride: usize) -> BlockMoments {
    let lags = 3 * stride + 1;
    let mut moments = BlockMoments::new(lags, 1, 1, 8).unwrap();
    let spread = |row: &[f64; 4]| {
        let mut out = vec![0.; lags];
        for (lag, x) in row.iter().enumerate() {
            out[lag * stride] = *x;
        }
        out
    };
    for origin in &PROPAGATOR {
        let [n, ab, a, b] = origin;
        moments
            .push_origin(&spread(ab), &spread(a), &spread(b), &spread(n))
            .unwrap();
    }
    moments
}
/// Slab sums of one frame: every ordered bin pair `b <= b'` is present once.
fn slabs(frames: &[[f64; 4]]) -> SlabMoments {
    let held: Vec<[Option<f64>; 4]> = frames.iter().map(|frame| frame.map(Some)).collect();
    masked_slabs(&held)
}
/// The same, with `None` for a bin that held no element in that frame: it
/// enters neither its own mean nor any pair of that frame.
fn masked_slabs(frames: &[[Option<f64>; 4]]) -> SlabMoments {
    let bins = 4;
    let mut out = SlabMoments {
        bins,
        components: 1,
        origins_per_block: 1,
        blocks: frames.len(),
        pair_ab: vec![0.; frames.len() * bins * bins],
        pair_n: vec![0.; frames.len() * bins * bins],
        profile: vec![0.; frames.len() * bins],
        profile_n: vec![0.; frames.len() * bins],
    };
    for (block, s) in frames.iter().enumerate() {
        for (b, held) in s.iter().enumerate() {
            let Some(x) = held else { continue };
            out.profile[block * bins + b] = *x;
            out.profile_n[block * bins + b] = 1.;
            for (other, partner) in s.iter().enumerate().skip(b) {
                if let Some(y) = partner {
                    out.pair_ab[(block * bins + b) * bins + other] = x * y;
                    out.pair_n[(block * bins + b) * bins + other] = 1.;
                }
            }
        }
    }
    out
}
fn euclidean_measurement(periodic: bool, frames: &[[f64; 4]]) -> Measurement {
    euclidean_record(periodic, 3, slabs(frames))
}
fn euclidean_record(periodic: bool, max_lag: usize, slabs: SlabMoments) -> Measurement {
    let config = MeasurementConfig {
        max_lag,
        time: TimeAxis::Euclidean {
            axis: None,
            bins: 4,
            range: Some([0., 4.]),
            periodic,
        },
        ..MeasurementConfig::default()
    };
    let frames = slabs.blocks;
    let mut channel = series("probe", 1, &vec![0.; frames]);
    channel.euclidean = Some(slabs);
    measurement(config, vec![channel], &vec![0; frames])
}
fn autoregressive(seed: u64, frames: usize, rho: f64, variance: f64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut x = vec![rng.normal() * (variance / (1. - rho * rho)).sqrt()];
    for t in 1..frames {
        x.push(rho * x[t - 1] + variance.sqrt() * rng.normal());
    }
    x
}

#[test]
fn frame_average_correlator_of_a_masked_segmented_series_matches_its_lag_sums() {
    let config = MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let mut channel = series("probe", 1, &D);
    channel.weight[3] = 0.;
    let measured = measurement(config, vec![channel], &[0, 0, 0, 0, 0, 1, 1, 1]);
    let analysis = analysis(jackknife(2), Subtraction::LagMeans);
    let moments = moments(
        std::slice::from_ref(&measured),
        "probe",
        EstimatorKind::FrameMean,
        &analysis,
    )
    .unwrap();
    let sums = moments.pooled(None);
    assert_eq!(sums.n, vec![7., 4., 3., 1.]);
    assert_eq!(sums.ab, vec![51., 22., 9., 0.]);
    assert_eq!(sums.a, vec![15., 10., 10., 2.]);
    assert_eq!(sums.b, vec![15., 9., 5., 0.]);
    close(
        &some(&moments.estimate(Subtraction::None)),
        &[7.285_714_285_714_29, 5.5, 3., 0.],
        1e-12,
    );
    close(
        &some(&moments.estimate(Subtraction::LagMeans)),
        &[2.693_877_551_020_41, -0.125, -2.555_555_555_555_56, 0.],
        1e-12,
    );
    close(
        &some(&moments.estimate(Subtraction::GlobalMean)),
        &[
            2.693_877_551_020_41,
            -0.086_734_693_877_551_7,
            -3.122_448_979_591_84,
            0.306_122_448_979_592,
        ],
        1e-12,
    );
}

#[test]
fn frame_average_jackknife_errors_and_covariance_of_the_eight_frame_record() {
    let measured = [scalar(3, &D)];
    for (subtraction, value, error, covariance) in [
        (
            Subtraction::LagMeans,
            [2.4375, -1.040_816_326_530_61, -1.25, 1.4],
            [
                1.261_979_632_400_06,
                0.978_979_060_041_633,
                0.1875,
                1.723_127_002_477_38,
            ],
            -1.222_222_222_222_22,
        ),
        (
            Subtraction::GlobalMean,
            [2.4375, -1.008_928_571_428_57, -1.1875, 1.4125],
            [
                1.261_979_632_400_06,
                0.962_042_941_022_057,
                0.196_908_984_427_192,
                1.455_598_626_377_17,
            ],
            -1.210_648_148_148_15,
        ),
    ] {
        let data = estimate(
            &measured,
            "probe",
            EstimatorKind::FrameMean,
            &analysis(jackknife(2), subtraction),
            None,
        )
        .unwrap();
        close(&some(&central(&data)), &value, 1e-12);
        close(&some(&data.estimate.error), &error, 1e-12);
        // Every delete-one sample recentres itself. A jackknife that froze the
        // full-sample mean, as the Python reference does, gives 1.101 at lag 3
        // under either centring.
        assert!(data.estimate.error[3].unwrap() > 1.3);
        let meta = &data.estimate.samples_meta;
        assert_eq!(
            (meta.resampling, meta.effective_block, meta.blocks),
            (ResampleKind::Jackknife, 2, 4)
        );
        assert_eq!((meta.replicas, meta.tau_int), (1, None));
        assert!(meta.sampling_unit.contains("time blocks of 2 origins"));
        // Four delete-one samples span at most three directions.
        assert!((1..=3).contains(&meta.covariance_rank));
        let table = data.estimate.covariance.as_ref().unwrap();
        assert_eq!(table.len(), 16);
        assert!((table[1] - covariance).abs() < 1e-12);
        assert!((table[1] - table[4]).abs() < 1e-15);
        // A fixed block names no autocorrelation time, so the bias the note
        // would quote has no value to quote with it.
        assert!(data.estimate.connected && data.estimate.connected_bias.is_none());
        assert_eq!(data.samples.count, 4);
        assert_eq!(data.estimate.lags, vec![0, 1, 2, 3]);
        assert_eq!(
            (data.estimate.time_unit, data.estimate.time_step),
            (TimeUnit::Frames, 1.)
        );
        assert_eq!(
            (data.channel.as_str(), data.kind),
            ("probe", EstimatorKind::FrameMean)
        );
    }
}

#[test]
fn a_zero_weight_frame_stays_an_origin_of_its_resampling_block() {
    let config = MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let mut channel = series("probe", 1, &D);
    channel.weight[3] = 0.;
    let measured = [measurement(config, vec![channel], &[0; 8])];
    let data = estimate(
        &measured,
        "probe",
        EstimatorKind::FrameMean,
        &analysis(jackknife(2), Subtraction::LagMeans),
        None,
    )
    .unwrap();
    close(
        &some(&central(&data)),
        &[2.693_877_551_020_41, -1.2, -2.125, 2.666_666_666_666_67],
        1e-12,
    );
    // The blocks are the frame windows {0,1},{2,3},{4,5},{6,7}; skipping the
    // masked frame as an origin would move every boundary and every error.
    close(
        &some(&data.estimate.error),
        &[
            1.592_576_746_798_86,
            1.463_024_201_718_76,
            1.084_184_361_244_28,
            2.850_438_562_747_85,
        ],
        1e-12,
    );
    assert_eq!(data.estimate.samples_meta.blocks, 4);
}

#[test]
fn fixed_and_valid_count_frame_averages_are_different_observables() {
    let config = MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let segment = [0, 0, 0, 0, 1, 1];
    let valid_count: [f64; 6] = [11. / 3., 2., 10. / 3., 2. / 3., 11. / 3., 3.];
    let fixed: [f64; 6] = [11. / 3., 4. / 3., 10. / 3., 2. / 3., 11. / 3., 3.];
    // Two of three elements are valid in frame 1, and `f = A · Σ w m / N`.
    assert!((fixed[1] - valid_count[1] * 2. / 3.).abs() < 1e-15);
    let measured = measurement(
        config,
        vec![
            series("valid_count", 1, &valid_count),
            series("fixed", 1, &fixed),
        ],
        &segment,
    );
    let analysis = analysis(jackknife(2), Subtraction::GlobalMean);
    let pooled = |channel: &str| {
        moments(
            std::slice::from_ref(&measured),
            channel,
            EstimatorKind::FrameMean,
            &analysis,
        )
        .unwrap()
    };
    let counted = pooled("valid_count");
    // No lag pairs the two segments.
    assert_eq!(counted.pooled(None).n, vec![6., 4., 2., 1.]);
    close(
        &some(&counted.estimate(Subtraction::LagMeans)),
        &[
            1.163_580_246_913_58,
            -0.319_444_444_444_444,
            1.111_111_111_111_11,
            0.,
        ],
        1e-12,
    );
    close(
        &some(&counted.estimate(Subtraction::GlobalMean)),
        &[
            1.163_580_246_913_58,
            -0.529_320_987_654_321,
            1.030_864_197_530_86,
            -1.941_358_024_691_36,
        ],
        1e-12,
    );
    close(
        &some(&pooled("fixed").estimate(Subtraction::GlobalMean)),
        &[
            1.385_802_469_135_80,
            -0.816_358_024_691_358,
            1.623_456_790_123_46,
            -2.052_469_135_802_47,
        ],
        1e-12,
    );
}

#[test]
fn source_frozen_propagator_centres_on_one_source_mean_or_on_each_leg() {
    let config = MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let mut channel = series("probe", 1, &[0.; 6]);
    channel.propagator = Some(propagator());
    let measured = [measurement(config, vec![channel], &[0, 0, 0, 0, 1, 1])];
    let analysis = analysis(jackknife(1), Subtraction::LagMeans);
    let moments = moments(&measured, "probe", EstimatorKind::SourceFrozen, &analysis).unwrap();
    let sums = moments.pooled(None);
    assert_eq!(sums.n, vec![17., 9., 5., 3.]);
    assert_eq!(sums.ab, vec![187., 63., 48., 12.]);
    assert_eq!(sums.a, vec![47., 27., 15., 11.]);
    assert_eq!(sums.b, vec![47., 19., 11., 2.]);
    let raw = some(&moments.estimate(Subtraction::None));
    close(&raw, &[11., 7., 9.6, 4.], 1e-12);
    let legs = some(&moments.estimate(Subtraction::LagMeans));
    close(
        &legs,
        &[
            3.356_401_384_083_04,
            0.666_666_666_666_667,
            3.,
            1.555_555_555_555_56,
        ],
        1e-12,
    );
    let source = some(&moments.estimate(Subtraction::GlobalMean));
    close(
        &source,
        &[
            3.356_401_384_083_04,
            0.512_879_661_668_589,
            2.867_128_027_681_66,
            -0.336_793_540_945_79,
        ],
        1e-12,
    );
    // The single source mean leaves the selection floor of the masked sinks,
    // which does not decay with the lag.
    let floor: Vec<f64> = source.iter().zip(&legs).map(|(s, l)| s - l).collect();
    close(
        &floor,
        &[0., -0.153_787_005, -0.132_871_972, -1.892_349_096],
        1e-8,
    );
    let data = estimate(
        &measured,
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
        None,
    )
    .unwrap();
    // Every pair of lag 3 sits in origin 0, so delete-one leaves the lag
    // undefined instead of an error bar of one block.
    assert_eq!(data.samples.defined, vec![true, true, true, false]);
    close(&some(&central(&data)[..3]), &legs[..3], 1e-12);
    assert!(central(&data)[3].is_none() && data.estimate.error[3].is_none());
    // The rank is that of the covariance over the defined lags; carrying the
    // undefined one into the whitener would leave a zero diagonal and rank 0.
    assert!((1..=3).contains(&data.estimate.samples_meta.covariance_rank));
    assert_eq!(data.kind, EstimatorKind::SourceFrozen);
    assert_ne!(RATE_QUANTITY_SOURCE_FROZEN, RATE_QUANTITY);
}

#[test]
fn propagator_lags_step_by_the_measured_lag_stride_in_the_requested_unit() {
    let config = MeasurementConfig {
        max_lag: 6,
        stride: 2,
        propagators: PropagatorConfig {
            lag_stride: 2,
            ..PropagatorConfig::default()
        },
        ..MeasurementConfig::default()
    };
    let mut channel = series("probe", 1, &[0.; 6]);
    channel.propagator = Some(strided_propagator(2));
    let mut measured = measurement(config, vec![channel], &[0; 6]);
    let analysis = AnalysisConfig {
        resampling: jackknife(1),
        time_unit: TimeUnit::StepDt,
        ..AnalysisConfig::default()
    };
    let unmeasured = estimate(
        std::slice::from_ref(&measured),
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
        None,
    );
    assert!(matches!(unmeasured, Err(GasError::Capability(_))));
    measured.calibration = Some(Calibration {
        dt: Some(0.002),
        ..Calibration::default()
    });
    let data = estimate(
        std::slice::from_ref(&measured),
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
        None,
    )
    .unwrap();
    // The unevaluated lags 1, 3 and 5 hold no pair, so they are left out of
    // the correlator instead of reaching a fit as undefined points.
    assert_eq!(data.estimate.lags, vec![0, 2, 4, 6]);
    assert_eq!((data.samples.dimension, data.estimate.value.len()), (4, 4));
    assert_eq!(data.estimate.time_unit, TimeUnit::StepDt);
    assert!((data.estimate.time_step - 0.004).abs() < 1e-15);
    // The kept entries are the ones the unstrided record estimates.
    let unstrided = estimate(
        &[measurement(
            MeasurementConfig {
                max_lag: 3,
                ..MeasurementConfig::default()
            },
            vec![{
                let mut channel = series("probe", 1, &[0.; 6]);
                channel.propagator = Some(propagator());
                channel
            }],
            &[0; 6],
        )],
        "probe",
        EstimatorKind::SourceFrozen,
        &AnalysisConfig {
            resampling: jackknife(1),
            ..AnalysisConfig::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(data.estimate.value, unstrided.estimate.value);
    assert_eq!(data.estimate.error, unstrided.estimate.error);
    // The axis steps by the propagator's own lag stride; the frame stride only
    // scales the unit. A record whose two strides differ separates them.
    let mut channel = series("probe", 1, &[0.; 6]);
    channel.propagator = Some(strided_propagator(2));
    let mut wide = measurement(
        MeasurementConfig {
            max_lag: 6,
            stride: 3,
            propagators: PropagatorConfig {
                lag_stride: 2,
                ..PropagatorConfig::default()
            },
            ..MeasurementConfig::default()
        },
        vec![channel],
        &[0; 6],
    );
    wide.calibration = Some(Calibration {
        dt: Some(0.002),
        ..Calibration::default()
    });
    let data = estimate(
        &[wide],
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
        None,
    )
    .unwrap();
    assert_eq!(data.estimate.lags, vec![0, 2, 4, 6]);
    assert!((data.estimate.time_step - 0.006).abs() < 1e-15);
}

#[test]
fn a_propagator_of_another_lag_range_is_declined() {
    // Four stored lags are the record of a measurement of three lags, not of
    // this one: placing them on its axis would invent the lags 4..6.
    let config = MeasurementConfig {
        max_lag: 6,
        ..MeasurementConfig::default()
    };
    let mut channel = series("probe", 1, &[0.; 6]);
    channel.propagator = Some(propagator());
    let measured = [measurement(config, vec![channel], &[0; 6])];
    let analysis = analysis(jackknife(1), Subtraction::LagMeans);
    match estimate(
        &measured,
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
        None,
    ) {
        Err(GasError::Capability(reason)) => assert!(reason.contains("propagator lags")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_malformed_propagator_table_is_an_error_and_never_an_index_panic() {
    let config = MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let mut broken = propagator();
    // A measurement that was never validated: one more block than its sums hold.
    broken.blocks += 1;
    let mut channel = series("probe", 1, &[0.; 6]);
    channel.propagator = Some(broken);
    let measured = [measurement(config, vec![channel], &[0; 6])];
    let analysis = analysis(jackknife(1), Subtraction::LagMeans);
    assert!(matches!(
        estimate(
            &measured,
            "probe",
            EstimatorKind::SourceFrozen,
            &analysis,
            None
        ),
        Err(GasError::Configuration(_))
    ));
    assert!(matches!(
        moments(&measured, "probe", EstimatorKind::SourceFrozen, &analysis),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn a_vector_channel_contracts_its_components_and_never_their_mean() {
    let config = MeasurementConfig {
        max_lag: 2,
        ..MeasurementConfig::default()
    };
    let vector = [1., -1., 2., 0.5, 0., 1.5, 3., -2., 1., 1., 2., 0.];
    let first: Vec<f64> = vector.chunks_exact(2).map(|v| v[0]).collect();
    let second: Vec<f64> = vector.chunks_exact(2).map(|v| v[1]).collect();
    let average: Vec<f64> = vector
        .chunks_exact(2)
        .map(|v| 0.5 * (v[0] + v[1]))
        .collect();
    let measured = measurement(
        config,
        vec![
            series("vector", 2, &vector),
            series("first", 1, &first),
            series("second", 1, &second),
            series("average", 1, &average),
        ],
        &[0; 6],
    );
    let analysis = analysis(jackknife(2), Subtraction::LagMeans);
    let value = |channel: &str| {
        some(
            &moments(
                std::slice::from_ref(&measured),
                channel,
                EstimatorKind::FrameMean,
                &analysis,
            )
            .unwrap()
            .estimate(Subtraction::LagMeans),
        )
    };
    let (contracted, first, second) = (value("vector"), value("first"), value("second"));
    close(&first, &[0.916_666_666_666_667, -0.84, 0.75], 1e-12);
    close(&second, &[1.416_666_666_666_67, -0.95, -0.21875], 1e-12);
    close(&contracted, &[2.333_333_333_333_33, -1.79, 0.53125], 1e-12);
    let sum: Vec<f64> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
    close(&contracted, &sum, 1e-12);
    close(
        &value("average"),
        &[0.166_666_666_666_667, -0.0675, -0.039_062_5],
        1e-12,
    );
}

#[test]
fn autoregressive_correlators_and_their_bias_agree_with_the_exact_autocovariance() {
    let frames = 20_000;
    let rho = 0.9;
    let single = autoregressive(7, frames, rho, 1.);
    let slow = autoregressive(11, frames, (-0.2_f64).exp(), 0.329_679_953_964_361);
    let fast = autoregressive(13, frames, (-0.8_f64).exp(), 1.596_206_964_010_69);
    let sum: Vec<f64> = slow.iter().zip(&fast).map(|(a, b)| a + b).collect();
    let config = MeasurementConfig {
        max_lag: 8,
        ..MeasurementConfig::default()
    };
    let measured = [measurement(
        config,
        vec![series("single", 1, &single), series("sum", 1, &sum)],
        &vec![0; frames],
    )];
    let analysis = AnalysisConfig::default();
    let data = estimate(
        &measured,
        "single",
        EstimatorKind::FrameMean,
        &analysis,
        None,
    )
    .unwrap();
    let variance = 1. / (1. - rho * rho);
    for lag in 0..=8 {
        let (value, error) = (
            data.estimate.value[lag].unwrap(),
            data.estimate.error[lag].unwrap(),
        );
        let exact = variance * rho.powi(lag as i32);
        // A jackknife error of about 0.15 covers every lag: the largest pull
        // of this seeded realization is 1.02, and 6 sigma is p < 1e-8 a lag.
        assert!(
            (value - exact).abs() < 6. * error,
            "lag {lag}: {value} vs {exact} +- {error}"
        );
    }
    // The one-lag effective rate reproduces -ln(rho) to two percent, which a
    // rate wrong by a per cent and a half would already miss.
    let rate = (data.estimate.value[0].unwrap() / data.estimate.value[1].unwrap()).ln();
    assert!((rate - 0.105_360_515_657_826).abs() < 0.002, "rate {rate}");
    // Sokal's window estimate of (1 + rho) / (2 (1 - rho)) = 9.5, whose own
    // relative error at this length is about ten percent.
    let tau = data.estimate.samples_meta.tau_int.unwrap();
    assert!((tau - 9.5).abs() < 4.75, "tau_int {tau}");
    let bias = data.estimate.connected_bias.unwrap();
    let expected = -2. * tau * data.estimate.value[0].unwrap() / frames as f64;
    assert!((bias - expected).abs() < 1e-15 && bias < 0.);
    let two = estimate(&measured, "sum", EstimatorKind::FrameMean, &analysis, None).unwrap();
    for lag in 0..=8 {
        let (value, error) = (
            two.estimate.value[lag].unwrap(),
            two.estimate.error[lag].unwrap(),
        );
        // Two rates sum to C(0) = 3; the largest pull here is 2.9, and the
        // lags of one realization move together, so the envelope is 6 sigma.
        let exact = (-0.2 * lag as f64).exp() + 2. * (-0.8 * lag as f64).exp();
        assert!(
            (value - exact).abs() < 6. * error,
            "lag {lag}: {value} vs {exact} +- {error}"
        );
    }
    // Two rates leave the first effective rate above both of them, at
    // ln(3 / (exp(-0.2) + 2 exp(-0.8))), not at the slow rate 0.2.
    let rate = (two.estimate.value[0].unwrap() / two.estimate.value[1].unwrap()).ln();
    assert!((rate - 0.557_807_360_028_419).abs() < 0.03, "rate {rate}");
}

#[test]
fn euclidean_slabs_average_bin_pairs_and_centre_every_bin_on_its_own_mean() {
    let frames = [[1., 3., 2., 0.], [2., 1., 4., 1.]];
    for (periodic, raw, connected) in [
        (
            false,
            [4.5, 3.166_666_666_666_67, 2.75, 1.],
            [0.625, -0.333_333_333_333_333, 0., 0.25],
        ),
        (
            true,
            [4.5, 2.625, 2.75, 2.625],
            [0.625, -0.1875, 0., -0.1875],
        ),
    ] {
        let measured = [euclidean_measurement(periodic, &frames)];
        let mut analysis = analysis(jackknife(1), Subtraction::LagMeans);
        analysis.connected = false;
        let bare = estimate(
            &measured,
            "probe",
            EstimatorKind::EuclideanTime,
            &analysis,
            None,
        )
        .unwrap();
        close(&some(&central(&bare)), &raw, 1e-12);
        assert!(!bare.estimate.connected && bare.estimate.connected_bias.is_none());
        analysis.connected = true;
        let data = estimate(
            &measured,
            "probe",
            EstimatorKind::EuclideanTime,
            &analysis,
            None,
        )
        .unwrap();
        close(&some(&central(&data)), &connected, 1e-12);
        assert_eq!(data.estimate.lags, vec![0, 1, 2, 3]);
        assert_eq!(data.estimate.time_unit, TimeUnit::Coordinate);
        assert!((data.estimate.time_step - 1.).abs() < 1e-15);
        // Both slabs of a pair vary together inside one frame, so a single
        // frame has no fluctuation left once its own profile is removed.
        assert_eq!(some(&data.estimate.error), vec![0., 0., 0., 0.]);
        assert!(errors(&bare.samples).iter().any(|e| e.unwrap() > 0.));
        // A product of pooled slab means would leave the static profile in.
        assert!((central(&data)[0].unwrap() - 1.4375).abs() > 0.5);
    }
    let measured = [euclidean_measurement(false, &frames)];
    let analysis = analysis(jackknife(1), Subtraction::LagMeans);
    assert!(matches!(
        moments(&measured, "probe", EstimatorKind::EuclideanTime, &analysis),
        Err(GasError::Capability(_))
    ));
    assert_ne!(RATE_QUANTITY_EUCLIDEAN, RATE_QUANTITY);
}

#[test]
fn replicas_pool_their_blocks_or_become_one_resampled_sample_each() {
    let pooled = analysis(jackknife(2), Subtraction::LagMeans);
    let one = estimate(
        &[scalar(3, &D)],
        "probe",
        EstimatorKind::FrameMean,
        &pooled,
        None,
    )
    .unwrap();
    let two: Vec<Measurement> = (0..2).map(|_| scalar(3, &D)).collect();
    let both = estimate(&two, "probe", EstimatorKind::FrameMean, &pooled, None).unwrap();
    close(&some(&central(&both)), &some(&central(&one)), 1e-12);
    let meta = &both.estimate.samples_meta;
    assert_eq!(
        (meta.blocks, meta.effective_block, meta.replicas),
        (8, 2, 2)
    );
    assert!(
        meta.sampling_unit
            .contains("pooled over 2 independently seeded runs")
    );
    let runs = AnalysisConfig {
        combine: Combine::RunsAsSamples,
        ..pooled.clone()
    };
    let seven: Vec<Measurement> = (0..7).map(|_| scalar(3, &D)).collect();
    match estimate(&seven, "probe", EstimatorKind::FrameMean, &runs, None) {
        Err(GasError::Capability(reason)) => assert!(reason.contains("at least 8")),
        other => panic!("{other:?}"),
    }
    let eight: Vec<Measurement> = (0..8).map(|_| scalar(3, &D)).collect();
    let data = estimate(&eight, "probe", EstimatorKind::FrameMean, &runs, None).unwrap();
    close(&some(&central(&data)), &some(&central(&one)), 1e-12);
    let meta = &data.estimate.samples_meta;
    assert_eq!(
        (meta.blocks, meta.effective_block, meta.replicas),
        (8, 8, 8)
    );
    assert_eq!(meta.sampling_unit, "independently seeded runs");
    // Identical runs differ from their pooled estimate by nothing at all.
    assert_eq!(some(&data.estimate.error), vec![0., 0., 0., 0.]);
}

#[test]
fn runs_as_samples_deletes_one_whole_run_whatever_an_automatic_block_says() {
    // A ramp inside every run has tau_int = 2.61, and the automatic block of
    // 24 such runs of 8 frames reaches 36 origins, two runs wide. The unit of
    // this combination is the run, so the block must not group two of them.
    let ramp: Vec<f64> = (0..8).map(f64::from).collect();
    let runs: Vec<Measurement> = (0..24).map(|_| scalar(3, &ramp)).collect();
    let analysis = AnalysisConfig {
        combine: Combine::RunsAsSamples,
        ..AnalysisConfig::default()
    };
    let data = estimate(&runs, "probe", EstimatorKind::FrameMean, &analysis, None).unwrap();
    let meta = &data.estimate.samples_meta;
    assert_eq!((meta.blocks, meta.replicas), (24, 24));
    // One unit is the eight frames of one run, not a block rounded up to them.
    assert_eq!(meta.effective_block, 8);
    assert_eq!(data.samples.count, 24);
    assert_eq!(meta.sampling_unit, "independently seeded runs");
    // A run, not an automatic time block, is the unit here: no autocorrelation
    // time selected it, so the centering note has no tau to quote.
    assert!(meta.tau_int.is_none() && data.estimate.connected_bias.is_none());
}

#[test]
fn channels_estimated_jointly_share_their_blocks_and_concatenate() {
    let config = MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let other: Vec<f64> = D.iter().map(|x| 3. - x).collect();
    let measured = [measurement(
        config,
        vec![series("a", 1, &D), series("b", 1, &other)],
        &[0; 8],
    )];
    let channels = [
        ("a".to_string(), EstimatorKind::FrameMean),
        ("b".to_string(), EstimatorKind::FrameMean),
    ];
    let members = joint(
        &measured,
        &channels,
        &analysis(jackknife(2), Subtraction::LagMeans),
    )
    .unwrap();
    assert_eq!(members.len(), 2);
    assert_eq!(
        members[0].samples.effective_block,
        members[1].samples.effective_block
    );
    assert_eq!(members[0].samples.blocks, members[1].samples.blocks);
    let table = Samples::concat(&[&members[0].samples, &members[1].samples]).unwrap();
    assert_eq!(table.dimension, 8);
    assert!(
        joint(
            &measured,
            &[],
            &analysis(jackknife(2), Subtraction::LagMeans)
        )
        .is_err()
    );
}

#[test]
fn a_correlator_matrix_holds_every_ordered_pair_of_its_basis() {
    let config = MeasurementConfig {
        max_lag: 2,
        ..MeasurementConfig::default()
    };
    let (a, b) = ([1., 2., 4., 3.], [2., 0., 1., 5.]);
    let measured = [measurement(
        config,
        vec![
            series("a", 1, &a),
            series("b", 1, &b),
            series("wide", 2, &[0.; 8]),
        ],
        &[0; 4],
    )];
    let analysis = analysis(jackknife(1), Subtraction::LagMeans);
    let basis = ["a".to_string(), "b".to_string()];
    let table = matrix(&measured, &basis, &analysis).unwrap();
    assert_eq!(table.channels, basis);
    assert_eq!(table.lags, vec![0, 1, 2]);
    assert_eq!((table.frames, table.replicas), (4, 1));
    assert_eq!(table.moments.len(), 4);
    let sums = table.moments[1].pooled(None);
    assert_eq!(sums.ab, vec![21., 22., 11.]);
    assert_eq!(sums.n, vec![4., 3., 2.]);
    assert_eq!(sums.a, vec![10., 7., 3.]);
    assert_eq!(sums.b, vec![8., 6., 6.]);
    // The reverse entry pairs the same frames the other way round.
    assert_eq!(table.moments[2].pooled(None).ab, vec![21., 7., 8.]);
    assert_eq!(table.moments[0].pooled(None).ab, vec![30., 22., 10.]);
    let diagonal = moments(&measured, "a", EstimatorKind::FrameMean, &analysis).unwrap();
    assert_eq!(table.moments[0].pooled(None).ab, diagonal.pooled(None).ab);
    // At lag 0 the sink is the source, so the matrix is symmetric to the bit
    // under both the raw and the connected estimate.
    for subtraction in [Subtraction::None, Subtraction::LagMeans] {
        let (upper, lower) = (
            table.moments[1].estimate(subtraction),
            table.moments[2].estimate(subtraction),
        );
        assert_eq!(upper[0].unwrap().to_bits(), lower[0].unwrap().to_bits());
        assert_ne!(upper[1], lower[1]);
    }
    // A basis that mixes component counts is a capability failure, not a
    // configuration one: `analyze` declines that basis and keeps the report.
    let mixed = ["a".to_string(), "wide".to_string()];
    let error = matrix(&measured, &mixed, &analysis).unwrap_err();
    assert!(matches!(error, GasError::Capability(_)));
    assert!(error.to_string().contains("channels of equal components"));
    assert!(matrix(&measured, &basis[..1], &analysis).is_err());
}

#[test]
fn a_channel_without_a_measured_estimator_declines_instead_of_estimating_zeros() {
    let config = MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let mut odd = series("odd", 1, &[0.; 8]);
    odd.exchange = ExchangeParity::Odd;
    odd.weight = vec![0.; 8];
    let mut empty = series("empty", 1, &[0.; 8]);
    empty.weight = vec![0.; 8];
    let measured = [measurement(
        config,
        vec![odd, empty, series("plain", 1, &D)],
        &[0; 8],
    )];
    let analysis = analysis(jackknife(2), Subtraction::LagMeans);
    let decline = |channel: &str, kind| match estimate(&measured, channel, kind, &analysis, None) {
        Err(GasError::Capability(reason)) => reason,
        other => panic!("{other:?}"),
    };
    // An exchange-odd operator whose elements all mirror has no frame mean at
    // all; a correlator of zeros would invent one.
    assert_eq!(
        decline("odd", EstimatorKind::FrameMean),
        EXCHANGE_ODD_REASON
    );
    assert!(decline("empty", EstimatorKind::FrameMean).contains("no measured frame"));
    assert!(decline("absent", EstimatorKind::FrameMean).contains("was not measured"));
    assert!(decline("plain", EstimatorKind::SourceFrozen).contains("no source-frozen propagator"));
    assert!(decline("plain", EstimatorKind::EuclideanTime).contains("no Euclidean-time axis"));
    let one = [scalar(3, &D[..2])];
    // Two frames make one block; a resampling needs two.
    assert!(matches!(
        estimate(&one, "probe", EstimatorKind::FrameMean, &analysis, None),
        Err(GasError::Numerical(_))
    ));
}

#[test]
fn a_frame_mean_and_a_propagator_each_centre_with_their_own_configured_subtraction() {
    let frames = [scalar(3, &D)];
    let pairs = || {
        let mut channel = series("probe", 1, &[0.; 6]);
        channel.propagator = Some(propagator());
        [measurement(
            MeasurementConfig {
                max_lag: 3,
                ..MeasurementConfig::default()
            },
            vec![channel],
            &[0, 0, 0, 0, 1, 1],
        )]
    };
    // The two knobs never stand in for each other: each configuration and its
    // mirror give the other estimator's numbers only to that estimator.
    for (frame_subtraction, propagator_subtraction, mean, pair) in [
        (
            Subtraction::LagMeans,
            Subtraction::GlobalMean,
            [2.4375, -1.040_816_326_530_612, -1.25, 1.4],
            [
                3.356_401_384_083_045,
                0.512_879_661_668_589,
                2.867_128_027_681_661,
            ],
        ),
        (
            Subtraction::GlobalMean,
            Subtraction::LagMeans,
            [2.4375, -1.008_928_571_428_571, -1.1875, 1.4125],
            [3.356_401_384_083_045, 0.666_666_666_666_667, 3.],
        ),
    ] {
        let analysis = AnalysisConfig {
            frame_subtraction,
            propagator_subtraction,
            resampling: jackknife(1),
            ..AnalysisConfig::default()
        };
        let average =
            estimate(&frames, "probe", EstimatorKind::FrameMean, &analysis, None).unwrap();
        close(&some(&central(&average)), &mean, 1e-12);
        let frozen = estimate(
            &pairs(),
            "probe",
            EstimatorKind::SourceFrozen,
            &analysis,
            None,
        )
        .unwrap();
        close(&some(&central(&frozen)[..3]), &pair, 1e-12);
        assert!(average.estimate.connected && frozen.estimate.connected);
    }
    // Without the connected flag both read the raw moment, whatever the two
    // subtractions say.
    let analysis = AnalysisConfig {
        connected: false,
        resampling: jackknife(1),
        ..AnalysisConfig::default()
    };
    let average = estimate(&frames, "probe", EstimatorKind::FrameMean, &analysis, None).unwrap();
    close(
        &some(&central(&average)),
        &[7.5, 4.857_142_857_142_857, 5., 5.8],
        1e-12,
    );
    assert!(!average.estimate.connected && average.estimate.connected_bias.is_none());
    let frozen = estimate(
        &pairs(),
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
        None,
    )
    .unwrap();
    close(&some(&central(&frozen)[..3]), &[11., 7., 9.6], 1e-12);
}

#[test]
fn only_the_frame_mean_carries_a_closed_form_centring_bias() {
    let analysis = AnalysisConfig::default();
    let average = estimate(
        &[scalar(3, &D)],
        "probe",
        EstimatorKind::FrameMean,
        &analysis,
        None,
    )
    .unwrap();
    assert!(average.estimate.samples_meta.tau_int.is_some());
    assert!(average.estimate.connected_bias.is_some());
    // The other two estimators are centred as well, and an automatic block
    // gives them an autocorrelation time to quote, but no closed form of the
    // centring bias is proved for either.
    let mut channel = series("probe", 1, &[0.; 6]);
    channel.propagator = Some(propagator());
    let pairs = [measurement(
        MeasurementConfig {
            max_lag: 3,
            ..MeasurementConfig::default()
        },
        vec![channel],
        &[0, 0, 0, 0, 1, 1],
    )];
    let frozen = estimate(
        &pairs,
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
        None,
    )
    .unwrap();
    assert!(frozen.estimate.samples_meta.tau_int.is_some());
    assert!(frozen.estimate.connected && frozen.estimate.connected_bias.is_none());
    let slabs = [euclidean_measurement(
        false,
        &[[1., 3., 2., 0.], [2., 1., 4., 1.]],
    )];
    let spatial = estimate(
        &slabs,
        "probe",
        EstimatorKind::EuclideanTime,
        &analysis,
        None,
    )
    .unwrap();
    assert!(spatial.estimate.samples_meta.tau_int.is_some());
    assert!(spatial.estimate.connected && spatial.estimate.connected_bias.is_none());
}

#[test]
fn the_centring_bias_divides_by_the_retained_frame_weight_of_every_replica() {
    let masked = || {
        let mut channel = series("probe", 1, &D);
        channel.weight[3] = 0.;
        measurement(
            MeasurementConfig {
                max_lag: 3,
                ..MeasurementConfig::default()
            },
            vec![channel],
            &[0; 8],
        )
    };
    let analysis = AnalysisConfig::default();
    // `T` is the measured frames that carry an average, seven of the eight
    // recorded frames of a run, summed over the replicas.
    for (replicas, weight) in [(1usize, 7.), (2, 14.)] {
        let measured: Vec<Measurement> = (0..replicas).map(|_| masked()).collect();
        let data = estimate(
            &measured,
            "probe",
            EstimatorKind::FrameMean,
            &analysis,
            None,
        )
        .unwrap();
        let tau = data.estimate.samples_meta.tau_int.unwrap();
        let bias = data.estimate.connected_bias.unwrap();
        let expected = -2. * tau * data.estimate.value[0].unwrap() / weight;
        assert!((bias - expected).abs() < 1e-15, "{bias} vs {expected}");
    }
}

#[test]
fn runs_as_samples_measures_its_unit_in_origins_of_a_run_and_not_in_stored_blocks() {
    // Eight frames blocked in threes are stored as 3, 3 and 2 origins; merged
    // into one unit they are the eight origins of the run, never the nine
    // origins three stored blocks would span.
    let runs = AnalysisConfig {
        combine: Combine::RunsAsSamples,
        resampling: jackknife(3),
        ..AnalysisConfig::default()
    };
    let eight: Vec<Measurement> = (0..8).map(|_| scalar(3, &D)).collect();
    let data = estimate(&eight, "probe", EstimatorKind::FrameMean, &runs, None).unwrap();
    let meta = &data.estimate.samples_meta;
    assert_eq!(
        (meta.blocks, meta.effective_block, data.samples.count),
        (8, 8, 8)
    );
    let pooled = AnalysisConfig {
        combine: Combine::PooledBlocks,
        ..runs
    };
    let one = estimate(
        &[scalar(3, &D)],
        "probe",
        EstimatorKind::FrameMean,
        &pooled,
        None,
    )
    .unwrap();
    close(&some(&central(&data)), &some(&central(&one)), 1e-12);
    assert_eq!(some(&data.estimate.error), vec![0., 0., 0., 0.]);
}

#[test]
fn channels_estimated_jointly_all_take_the_block_of_their_slowest_member() {
    let frames = 2_000;
    let slow = autoregressive(7, frames, 0.9, 1.);
    let fast = autoregressive(11, frames, 0., 1.);
    let measured = [measurement(
        MeasurementConfig {
            max_lag: 4,
            ..MeasurementConfig::default()
        },
        vec![series("slow", 1, &slow), series("fast", 1, &fast)],
        &vec![0; frames],
    )];
    let analysis = AnalysisConfig::default();
    let alone =
        |id: &str| estimate(&measured, id, EstimatorKind::FrameMean, &analysis, None).unwrap();
    let (patient, quick) = (alone("slow"), alone("fast"));
    // An uncorrelated channel carries a much shorter block of its own, and two
    // tables blocked differently do not join.
    assert!(quick.samples.effective_block < patient.samples.effective_block);
    assert!(Samples::concat(&[&patient.samples, &quick.samples]).is_err());
    let members = joint(
        &measured,
        &[
            ("slow".to_string(), EstimatorKind::FrameMean),
            ("fast".to_string(), EstimatorKind::FrameMean),
        ],
        &analysis,
    )
    .unwrap();
    assert_eq!(
        members[0].samples.effective_block,
        patient.samples.effective_block
    );
    assert_eq!(
        members[1].samples.effective_block,
        members[0].samples.effective_block
    );
    assert_eq!(
        members[0].estimate.samples_meta.tau_int,
        patient.estimate.samples_meta.tau_int
    );
    assert_eq!(
        members[1].estimate.samples_meta.tau_int,
        members[0].estimate.samples_meta.tau_int
    );
    let table = Samples::concat(&[&members[0].samples, &members[1].samples]).unwrap();
    assert_eq!(table.dimension, 10);
    // Only the blocks moved: the correlator of the slow channel is its own.
    assert_eq!(members[0].estimate.value, patient.estimate.value);
}

#[test]
fn a_replica_boundary_separates_two_runs_exactly_as_a_segment_break_does() {
    let ramp: Vec<f64> = (0..8).map(f64::from).collect();
    let twice: Vec<f64> = ramp.iter().chain(&ramp).copied().collect();
    let mut segment = vec![0u32; 8];
    segment.extend(std::iter::repeat_n(1u32, 8));
    let one = [measurement(
        MeasurementConfig {
            max_lag: 3,
            ..MeasurementConfig::default()
        },
        vec![series("probe", 1, &twice)],
        &segment,
    )];
    let two: Vec<Measurement> = (0..2).map(|_| scalar(3, &ramp)).collect();
    // An automatic block reports the autocorrelation time that chose it, so a
    // lagged product pairing the last frame of a run with the first of the
    // next would be visible in it.
    let analysis = AnalysisConfig::default();
    let single = estimate(&one, "probe", EstimatorKind::FrameMean, &analysis, None).unwrap();
    let replicas = estimate(&two, "probe", EstimatorKind::FrameMean, &analysis, None).unwrap();
    assert_eq!(
        single.estimate.samples_meta.tau_int,
        replicas.estimate.samples_meta.tau_int
    );
    assert_eq!(single.estimate.value, replicas.estimate.value);
    assert_eq!(single.estimate.error, replicas.estimate.error);
    assert_eq!(
        single.samples.effective_block,
        replicas.samples.effective_block
    );
}

#[test]
fn replicas_pool_their_propagator_blocks_only_when_the_stored_blocks_nest() {
    let replica = |origins_per_block: usize| {
        let mut moments = BlockMoments::new(4, 1, origins_per_block, 8).unwrap();
        for origin in &PROPAGATOR {
            let [n, ab, a, b] = origin;
            moments.push_origin(ab, a, b, n).unwrap();
        }
        let mut channel = series("probe", 1, &[0.; 6]);
        channel.propagator = Some(moments);
        measurement(
            MeasurementConfig {
                max_lag: 3,
                ..MeasurementConfig::default()
            },
            vec![channel],
            &[0, 0, 0, 0, 1, 1],
        )
    };
    let analysis = analysis(jackknife(1), Subtraction::LagMeans);
    // Blocks of one and of two origins nest, the coarser one winning, and the
    // pooled sums are the two records added.
    let table = moments(
        &[replica(1), replica(2)],
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
    )
    .unwrap();
    assert_eq!(table.origins_per_block, 2);
    assert_eq!(table.pooled(None).n, vec![34., 18., 10., 6.]);
    assert_eq!(table.pooled(None).ab, vec![374., 126., 96., 24.]);
    // A ratio of doubled sums is the single record's correlator.
    close(
        &some(&table.estimate(Subtraction::LagMeans)),
        &[
            3.356_401_384_083_045,
            0.666_666_666_666_667,
            3.,
            1.555_555_555_555_556,
        ],
        1e-12,
    );
    // Blocks of three and of four origins do not nest, and no coarsening of
    // either places both records on one axis of blocks.
    match moments(
        &[replica(3), replica(4)],
        "probe",
        EstimatorKind::SourceFrozen,
        &analysis,
    ) {
        Err(GasError::Capability(reason)) => assert!(reason.contains("do not nest")),
        other => panic!("{other:?}"),
    }
}

#[test]
fn euclidean_replicas_pool_their_slab_blocks_or_become_one_resampled_unit_each() {
    let frames = [[1., 3., 2., 0.], [2., 1., 4., 1.]];
    let pooled = analysis(jackknife(1), Subtraction::LagMeans);
    let one = [euclidean_measurement(false, &frames)];
    let single = estimate(&one, "probe", EstimatorKind::EuclideanTime, &pooled, None).unwrap();
    let runs = AnalysisConfig {
        combine: Combine::RunsAsSamples,
        ..pooled.clone()
    };
    let seven: Vec<Measurement> = (0..7)
        .map(|_| euclidean_measurement(false, &frames))
        .collect();
    match estimate(&seven, "probe", EstimatorKind::EuclideanTime, &runs, None) {
        Err(GasError::Capability(reason)) => assert!(reason.contains("at least 8")),
        other => panic!("{other:?}"),
    }
    let eight: Vec<Measurement> = (0..8)
        .map(|_| euclidean_measurement(false, &frames))
        .collect();
    let data = estimate(&eight, "probe", EstimatorKind::EuclideanTime, &runs, None).unwrap();
    close(&some(&central(&data)), &some(&central(&single)), 1e-12);
    let meta = &data.estimate.samples_meta;
    assert_eq!(
        (meta.blocks, meta.effective_block, meta.replicas),
        (8, 2, 8)
    );
    assert_eq!(meta.sampling_unit, "independently seeded runs");
    assert_eq!(some(&data.estimate.error), vec![0., 0., 0., 0.]);
    // Pooled, the same eight runs are sixteen blocks of one frame each and
    // the estimate is still the single run's.
    let together = estimate(&eight, "probe", EstimatorKind::EuclideanTime, &pooled, None).unwrap();
    close(&some(&central(&together)), &some(&central(&single)), 1e-12);
    assert_eq!(together.estimate.samples_meta.blocks, 16);
}

#[test]
fn euclidean_lags_stop_at_the_last_bin_and_every_bin_centres_on_the_frames_that_hold_it() {
    // A measurement may ask for more lags than the axis has bins; a periodic
    // axis would otherwise wrap lag 4 back onto lag 0 and report it twice.
    let wide = euclidean_record(true, 5, slabs(&[[1., 3., 2., 0.], [2., 1., 4., 1.]]));
    let analysis = analysis(jackknife(1), Subtraction::LagMeans);
    let data = estimate(
        &[wide],
        "probe",
        EstimatorKind::EuclideanTime,
        &analysis,
        None,
    )
    .unwrap();
    assert_eq!(data.estimate.lags, vec![0, 1, 2, 3]);
    close(
        &some(&central(&data)),
        &[0.625, -0.1875, 0., -0.1875],
        1e-12,
    );
    // A bin that holds no element in a frame is absent from that frame's
    // pairs, and its mean is the mean over the frames that do hold it.
    let masked = || {
        euclidean_record(
            false,
            3,
            masked_slabs(&[
                [Some(1.), Some(3.), Some(2.), None],
                [Some(2.), Some(1.), Some(4.), Some(1.)],
                [Some(3.), Some(0.), Some(1.), Some(2.)],
            ]),
        )
    };
    let mut bare = analysis.clone();
    bare.connected = false;
    let raw = estimate(
        &[masked()],
        "probe",
        EstimatorKind::EuclideanTime,
        &bare,
        None,
    )
    .unwrap();
    close(
        &some(&central(&raw)),
        &[4.375, 2.666_666_666_666_667, 2.416_666_666_666_667, 4.],
        1e-12,
    );
    let data = estimate(
        &[masked()],
        "probe",
        EstimatorKind::EuclideanTime,
        &analysis,
        None,
    )
    .unwrap();
    close(
        &some(&central(&data)),
        &[
            1.006_944_444_444_444,
            -0.425_925_925_925_926,
            -0.916_666_666_666_667,
            1.,
        ],
        1e-12,
    );
    // Centring a bin on the frames of a pair instead of on its own would give
    // -1.009, -1.417 and -0.5 at the three later lags.
    assert!(central(&data)[3].unwrap() > 0.);
}

#[test]
fn a_channel_missing_from_a_replica_and_a_coordinate_lag_unit_are_declined() {
    let config = || MeasurementConfig {
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let full = || {
        measurement(
            config(),
            vec![series("probe", 1, &D), series("other", 1, &D)],
            &[0; 8],
        )
    };
    let partial = measurement(config(), vec![series("probe", 1, &D)], &[0; 8]);
    let analysis = analysis(jackknife(2), Subtraction::LagMeans);
    // Half the replicas are not a shorter measurement of the same channel.
    match estimate(
        &[full(), partial],
        "other",
        EstimatorKind::FrameMean,
        &analysis,
        None,
    ) {
        Err(GasError::Capability(reason)) => assert!(reason.contains("was not measured")),
        other => panic!("{other:?}"),
    }
    // The coordinate unit belongs to the Euclidean estimator alone; every
    // entry point rejects a request for it before it reaches a lag axis.
    let coordinate = AnalysisConfig {
        time_unit: TimeUnit::Coordinate,
        ..analysis.clone()
    };
    match estimate(
        &[full()],
        "probe",
        EstimatorKind::FrameMean,
        &coordinate,
        None,
    ) {
        Err(GasError::Configuration(reason)) => assert!(reason.contains("coordinate time unit")),
        other => panic!("{other:?}"),
    }
    // A slab table that was never validated is an error, never an index panic.
    let mut broken = slabs(&[[1., 3., 2., 0.], [2., 1., 4., 1.]]);
    broken.blocks += 1;
    assert!(matches!(
        estimate(
            &[euclidean_record(false, 3, broken)],
            "probe",
            EstimatorKind::EuclideanTime,
            &analysis,
            None
        ),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn an_estimate_states_what_its_resampling_geometry_does_not_cover() {
    let (frames, max_lag) = (64, 20);
    let x = autoregressive(7, frames, 0.9, 1.);
    let config = MeasurementConfig {
        max_lag,
        ..MeasurementConfig::default()
    };
    let measured = [measurement(
        config,
        vec![series("probe", 1, &x)],
        &vec![0; frames],
    )];
    let fixed = analysis(jackknife(8), Subtraction::LagMeans);
    let data = estimate(&measured, "probe", EstimatorKind::FrameMean, &fixed, None).unwrap();
    assert_eq!(
        (
            data.estimate.samples_meta.effective_block,
            data.estimate.samples_meta.blocks
        ),
        (8, 8)
    );
    let notes = data.notes();
    // Every pair of a lag at or beyond the block straddles it, so the deletion
    // of a block removes no product those lags hold: at least 734 of the pairs
    // up to lag 20 survive it, the count over the 57 origins the eight blocks
    // are certain to hold.
    assert_eq!(notes.len(), 2, "{notes:?}");
    assert!(notes[0].starts_with(DELETION_GEOMETRY_NOTE), "{notes:?}");
    assert!(
        notes[0].contains("at least 734 of the pairs up to lag 20")
            && notes[0].contains("block of 8 origins"),
        "{notes:?}"
    );
    // A fixed block carries no autocorrelation time, so only the deletion
    // geometry and the replica count are stated.
    assert_eq!(notes[1], resample_support_note(8, 21).unwrap(), "{notes:?}");
    assert!(notes[1].contains("8 resampling blocks") && notes[1].contains("21 fitted points"));
    // Sixty-four origins are too few for the batch-size rule at this
    // autocorrelation time, and an automatic block says so.
    let auto = analysis(Resampling::default(), Subtraction::LagMeans);
    let data = estimate(&measured, "probe", EstimatorKind::FrameMean, &auto, None).unwrap();
    let notes = data.notes();
    assert_eq!(notes.len(), 3, "{notes:?}");
    assert!(
        notes[1].contains("below the batch size") && notes[1].contains("57 origins"),
        "{notes:?}"
    );
    // A long series resampled over many short lags needs no statement at all.
    let long = autoregressive(11, 20_000, 0.9, 1.);
    let measured = [measurement(
        MeasurementConfig {
            max_lag: 8,
            ..MeasurementConfig::default()
        },
        vec![series("probe", 1, &long)],
        &vec![0; 20_000],
    )];
    let data = estimate(&measured, "probe", EstimatorKind::FrameMean, &auto, None).unwrap();
    assert_eq!(data.notes(), Vec::<String>::new());
    assert!(data.estimate.samples_meta.effective_block > 8);
}

#[test]
fn a_resample_table_smaller_than_the_fitted_window_names_both_numbers() {
    // A delete-one table of K blocks spans K - 1 independent directions; a
    // window of more points than that leaves the lag covariance singular.
    assert_eq!(resample_support_note(22, 21), None);
    assert_eq!(resample_support_note(2, 1), None);
    assert_eq!(resample_support_note(0, 0), None);
    let note = resample_support_note(21, 21).unwrap();
    assert!(
        note.contains("21 resampling blocks")
            && note.contains("20 independent replicas")
            && note.contains("21 fitted points")
            && note.contains("rank deficient"),
        "{note}"
    );
}
