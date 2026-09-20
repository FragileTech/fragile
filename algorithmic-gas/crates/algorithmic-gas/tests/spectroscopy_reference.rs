use algorithmic_gas::{
    GasConfig, GasError, Precision, RecordingConfig,
    geometry::{Distance, Kernel},
    kinetic::{KineticKind, ViscousForceConfig},
    noise::{FactorValues, NoiseGeometry},
    physics::{
        numerics::ResampleKind,
        partvi::ExperimentResult,
        spectroscopy::{
            AnalysisConfig, Availability, Capabilities, ChannelReport, ChannelSpec, Measurement,
            MeasurementConfig, SPECTROSCOPY_VERSION, SpectroscopyReport, analyze,
            config::{
                FrameNormalization, HyperchargeNormalization, Range, ReferenceEntry,
                ReferenceTable, StandardModelInputs, TimeUnit,
            },
            contract::EXCHANGE_ODD_REASON,
            couplings::{
                self, AlgorithmicScales, CalibrationTargets, TargetCouplings, casimir, g1_squared,
                g2_casimir_squared, g2_clock_squared, gd_squared, kernel_action_scale,
                kernel_factor, presupposed_kernel_second_moment,
            },
            presentation::{EXPERIMENT, present},
            reference::{assigned, compare, prediction, ratio, spread, tension},
            report::{
                CALCULATION_ORIGIN, Calibration, Comparison, CorrelatorEstimate, CouplingReport,
                Coverage, EstimatorKind, FitDiagnostics, FitMethodKind, FitOutcome, FlowDiagnostic,
                GevpReport, GroupFit, HYPOTHESIS_LABEL, MassEstimate, PriorDominance, Quantity,
                RATE_QUANTITY, RATE_QUANTITY_EUCLIDEAN, RATE_QUANTITY_SOURCE_FROZEN, RatioRow,
                SamplesMeta, WindowFit,
            },
        },
    },
    tessellation::MetricKind,
    variants::Variant,
};

const PION: &str = "meson/pseudoscalar/standard";
const SIGMA: &str = "meson/scalar/standard";
const RHO: &str = "vector/vector/full/raw";
const NUCLEON: &str = "baryon/complex";

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1e-300)
}
fn pair_close(a: Option<[f64; 2]>, b: [f64; 2]) -> bool {
    a.is_some_and(|a| close(a[0], b[0]) && close(a[1], b[1]))
}
fn rate(value: f64, error: f64, time_unit: TimeUnit) -> MassEstimate {
    MassEstimate {
        quantity: RATE_QUANTITY.into(),
        value,
        error,
        statistical: error,
        systematic: 0.,
        method: FitMethodKind::WindowScan,
        time_unit,
        prior_dominance: None,
    }
}
fn channel(key: &str, measured: Option<[f64; 2]>) -> ChannelReport {
    let spec = ChannelSpec::all()
        .into_iter()
        .find(|s| s.id() == key)
        .unwrap();
    ChannelReport {
        id: format!("{key}/distance"),
        spec,
        estimator: Some(EstimatorKind::FrameMean),
        mass: measured.map(|[value, error]| rate(value, error, TimeUnit::Frames)),
        ..ChannelReport::default()
    }
}
fn assignments(analysis: &mut AnalysisConfig, names: &[&str]) {
    analysis
        .assignments
        .retain(|_, name| names.contains(&name.as_str()));
}

/// Synthetic rates near the reference ratios: small tensions of both signs.
fn near_channels() -> Vec<ChannelReport> {
    vec![
        channel(PION, Some([0.10, 0.01])),
        channel(SIGMA, Some([0.35, 0.06])),
        channel(RHO, Some([0.56, 0.04])),
        channel(NUCLEON, Some([0.70, 0.05])),
    ]
}

/// Large tensions, and a scalar rate 2.5 sigma from zero.
fn far_channels() -> Vec<ChannelReport> {
    vec![
        channel(PION, Some([0.30, 0.02])),
        channel(SIGMA, Some([0.20, 0.08])),
        channel(RHO, Some([0.45, 0.03])),
        channel(NUCLEON, Some([0.50, 0.04])),
    ]
}
fn four_names(anchors: &[&str]) -> AnalysisConfig {
    let mut analysis = AnalysisConfig {
        anchors: anchors.iter().map(|a| a.to_string()).collect(),
        ..AnalysisConfig::default()
    };
    assignments(&mut analysis, &["pion", "f0_500", "rho", "nucleon"]);
    analysis
}
fn label(row: &RatioRow) -> String {
    format!("{}/{}", row.numerator, row.denominator)
}

#[test]
fn ratio_rows_follow_the_table_order_and_reproduce_the_hand_computed_tensions() {
    let comparison = compare(&near_channels(), &four_names(&["nucleon"]))
        .unwrap()
        .unwrap();
    let expected = [
        (
            "f0_500/pion",
            [3.5, 0.69462219947249],
            3.58242174432557,
            -0.082593310981923,
        ),
        (
            "rho/pion",
            [5.6, 0.68818602136341],
            5.55461656301168,
            0.0659462782784833,
        ),
        (
            "nucleon/pion",
            [7., 0.860232526704263],
            6.72257266028991,
            0.322502731626395,
        ),
        (
            "rho/f0_500",
            [1.6, 0.297142857142857],
            1.55052,
            0.115207369467613,
        ),
        (
            "nucleon/f0_500",
            [2., 0.371428571428571],
            1.876544176,
            0.233804469983878,
        ),
        (
            "nucleon/rho",
            [1.25, 0.126269068069026],
            1.21026763666383,
            0.314662991564436,
        ),
    ];
    assert_eq!(comparison.ratios.len(), expected.len());
    for (row, (name, value, target, sigma)) in comparison.ratios.iter().zip(expected) {
        assert_eq!(label(row), name);
        assert!(pair_close(row.measured, value), "{name}");
        assert!(close(row.reference, target), "{name}");
        // Tensions are differences of nearly equal numbers: 1e-9 absolute.
        assert!((row.tension_sigma.unwrap() - sigma).abs() < 1e-9, "{name}");
    }
    assert_eq!(comparison.label, HYPOTHESIS_LABEL);
    assert_eq!(comparison.label, "hypothesis mapping");
    let names: Vec<&str> = comparison
        .reference
        .iter()
        .map(|r| r.name.as_str())
        .collect();
    assert_eq!(names, ["pion", "f0_500", "rho", "nucleon"]);
    assert_eq!(comparison.reference[0].channel, format!("{PION}/distance"));
    assert_eq!(comparison.reference[0].measured, Some([0.10, 0.01]));
    assert_eq!(
        comparison.reference[0].estimator,
        Some(EstimatorKind::FrameMean)
    );
    assert_eq!(comparison.reference[0].unit, "MeV");
}

#[test]
fn a_tension_is_the_signed_difference_over_both_errors_in_quadrature() {
    // R = 2/1 with relative errors 0.1 and 0.1: sigma_R = 2 sqrt(0.02).
    // Reference 30(3)/10(0) = 3 with error 0.3.
    // Tension = (2 - 3)/sqrt(0.08 + 0.09) = -1/sqrt(0.17).
    let measured = ratio([2., 0.2], [1., 0.1], 0.);
    let expected = ratio([30., 3.], [10., 0.], 0.);
    assert!(close(measured[0], 2.) && close(measured[1], 0.282842712474619));
    assert!(close(expected[0], 3.) && close(expected[1], 0.3));
    assert!(close(
        tension(measured, expected).unwrap(),
        -2.42535625036333
    ));
    assert!(close(
        tension(measured, expected).unwrap(),
        -1. / 0.17f64.sqrt()
    ));
    let table = AnalysisConfig {
        reference: ReferenceTable {
            unit: "GeV".into(),
            entries: vec![
                ReferenceEntry {
                    name: "light".into(),
                    value: 10.,
                    error: 0.,
                    source: "test".into(),
                },
                ReferenceEntry {
                    name: "heavy".into(),
                    value: 30.,
                    error: 3.,
                    source: "test".into(),
                },
            ],
        },
        assignments: [(SIGMA, "light"), (RHO, "heavy")]
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .into(),
        anchors: vec!["light".into()],
        ..AnalysisConfig::default()
    };
    let channels = [
        channel(SIGMA, Some([1., 0.1])),
        channel(RHO, Some([2., 0.2])),
    ];
    let comparison = compare(&channels, &table).unwrap().unwrap();
    let row = &comparison.ratios[0];
    assert_eq!(
        (row.numerator.as_str(), row.denominator.as_str()),
        ("heavy", "light")
    );
    assert!(close(row.tension_sigma.unwrap(), -2.42535625036333));
    // An anchor without a reference error: the prediction tension equals the
    // ratio tension, P = 10 * 2 = 20(2.83) against 30(3).
    let anchor = &comparison.anchors[0];
    assert!(pair_close(anchor.scale, [10., 1.]));
    assert_eq!(anchor.predictions.len(), 1);
    assert!(pair_close(
        anchor.predictions[0].predicted,
        [20., 2.82842712474619]
    ));
    assert!(close(
        anchor.predictions[0].tension_sigma.unwrap(),
        row.tension_sigma.unwrap()
    ));
}

#[test]
fn anchor_rescaling_omits_the_anchor_and_reproduces_scales_predictions_and_spread() {
    let comparison = compare(&near_channels(), &four_names(&["nucleon", "pion", "rho"]))
        .unwrap()
        .unwrap();
    let expected = [
        (
            "nucleon",
            [1340.38869714286, 95.7420497959184],
            vec![
                (
                    "pion",
                    [134.038869714286, 16.4720850815576],
                    -0.335811784482413,
                ),
                ("f0_500", [469.136044, 87.1252653142857], -0.232706572399128),
                ("rho", [750.6176704, 75.8238349740413], -0.324993008330335),
            ],
        ),
        (
            "pion",
            [1395.7039, 139.570390011607],
            vec![
                (
                    "f0_500",
                    [488.496365, 96.9486912850802],
                    -0.0825933109819632,
                ),
                ("rho", [781.594184, 96.0503913995287], 0.0659462782784247),
                ("nucleon", [976.99273, 120.062989249411], 0.322502731625016),
            ],
        ),
        (
            "rho",
            [1384.39285714286, 98.8860570176131],
            vec![
                (
                    "pion",
                    [138.439285714286, 17.0129033682003],
                    -0.0664850825970822,
                ),
                ("f0_500", [484.5375, 89.9856505330127], -0.114940058083519),
                ("nucleon", [969.075, 97.8917798938747], 0.314662906664824),
            ],
        ),
    ];
    assert_eq!(comparison.anchors.len(), 3);
    for (anchor, (name, scale, predictions)) in comparison.anchors.iter().zip(expected) {
        assert_eq!(anchor.anchor, name);
        assert!(pair_close(anchor.scale, scale), "{name}");
        assert!(anchor.predictions.iter().all(|p| p.name != name));
        assert_eq!(anchor.predictions.len(), predictions.len());
        for (p, (target, value, sigma)) in anchor.predictions.iter().zip(predictions) {
            assert_eq!(p.name, target);
            assert!(pair_close(p.predicted, value), "{target} at {name}");
            assert!((p.tension_sigma.unwrap() - sigma).abs() < 1e-9);
        }
    }
    let spreads = [
        ("pion", 0.0161496102066557),
        ("f0_500", 0.0173723854838043),
        ("rho", 0.0202168606847974),
        ("nucleon", 0.00406857884643106),
    ];
    for (row, (name, value)) in comparison.anchor_spread.iter().zip(spreads) {
        assert_eq!(row.name, name);
        assert!((row.spread.unwrap() - value).abs() < 1e-12, "{name}");
    }
    // The spread is the coefficient of variation of the lattice scales.
    let scales = [1395.7039, 1384.39285714286, 1340.38869714286];
    assert!((spread(&scales).unwrap() - 0.0173723854838043).abs() < 1e-12);
    assert!(
        comparison
            .notes
            .iter()
            .any(|n| n.starts_with("Anchor spread is the population standard deviation"))
    );
    // The prediction tension and the ratio tension agree up to the tiny
    // reference error of the anchor.
    let ratio_tension = comparison.ratios[1].tension_sigma.unwrap();
    let at_pion = comparison.anchors[1].predictions[1].tension_sigma.unwrap();
    assert!((ratio_tension - at_pion).abs() < 1e-9);
    let single = compare(&near_channels(), &four_names(&["nucleon"]))
        .unwrap()
        .unwrap();
    assert_eq!(single.anchor_spread.len(), 4);
    assert!(single.anchor_spread.iter().all(|s| s.spread.is_none()));
    assert!(single.notes.iter().all(|n| !n.starts_with("Anchor spread")));
}

#[test]
fn the_notes_label_the_mapping_and_state_the_look_elsewhere_and_correlation_caveats() {
    let comparison = compare(&near_channels(), &AnalysisConfig::default())
        .unwrap()
        .unwrap();
    let always = [
        "Hypothesis mapping: every channel-to-reference assignment is an input of this analysis, \
         fixed in the analysis configuration. No reference value enters a prior, a fit window or \
         a channel selection.",
        "The compared quantity is the decay rate of the algorithm-time autocorrelation. It is a \
         mass only under the positive transfer representation \
         (cor-effective-twistor-positive-transfer); the gas is a non-reversible Markov chain and \
         that assumption is not tested here.",
        "Ratios of rates do not depend on the time unit assigned to a lag \
         (thm-qft-ratio-rescale). They do depend on the integrator step, the recording stride, \
         the estimator, the frame normalisation and the smearing scale, so only channels that \
         share the time unit, the estimator, the scale and the frame normalisation are compared.",
        "Anchor rescaling uses one reference value as an input; the anchor's own row is not a \
         prediction and is omitted. Of the 6 ratio rows only 3 are algebraically independent, \
         and rows that share a channel are statistically correlated.",
        "Tensions are (measured - reference)/sigma with sigma^2 = sigma_measured^2 + \
         sigma_reference^2 from first-order error propagation. They carry no look-elsewhere \
         correction for the number of rows or for the choice among operator variants, fit \
         windows and assignments. A tension below 1 is expected in 68% of rows when the \
         hypothesis is true and is also produced by a large error bar; it is not evidence for \
         the assignment.",
        "Channel rates are estimated on the same frames but are treated as uncorrelated because \
         the report carries no cross-channel covariance. A positive correlation makes the quoted \
         ratio errors too large and the tensions too small in magnitude; a negative correlation \
         does the opposite.",
    ];
    assert_eq!(&comparison.notes[..6], &always);
    // The six default assignments give 15 rows; the 9 without a rate keep
    // their reference ratio and nothing else.
    assert_eq!(comparison.ratios.len(), 15);
    let empty: Vec<_> = comparison
        .ratios
        .iter()
        .filter(|r| r.measured.is_none())
        .collect();
    assert_eq!(empty.len(), 9);
    assert!(
        empty
            .iter()
            .all(|r| r.tension_sigma.is_none() && r.reference > 0.)
    );
    assert!(empty.iter().all(|r| {
        [&r.numerator, &r.denominator]
            .iter()
            .any(|n| *n == "a1" || *n == "glueball_0pp")
    }));
    for broad in ["f0_500", "a1", "glueball_0pp"] {
        assert!(
            comparison
                .notes
                .iter()
                .any(|n| n.starts_with(&format!("Reference '{broad}'"))
                    && n.contains("a range, not a Gaussian standard deviation"))
        );
    }
    assert!(
        comparison
            .notes
            .iter()
            .all(|n| !n.starts_with("Reference 'pion'"))
    );
    let text = serde_json::to_string(&comparison).unwrap();
    for forbidden in ["\u{2713}", "Matches", "matches", "best anchor", "Best"] {
        assert!(!text.contains(forbidden), "{forbidden}");
    }
    assert_eq!(
        serde_json::from_str::<Comparison>(&text).unwrap(),
        comparison
    );
}

#[test]
fn a_rate_below_three_sigma_is_a_numerator_but_neither_a_denominator_nor_an_anchor() {
    let comparison = compare(&far_channels(), &four_names(&["nucleon", "f0_500"]))
        .unwrap()
        .unwrap();
    let expected = [
        (
            "f0_500/pion",
            Some(([0.666666666666667, 0.270345001346588], -3.80750637891163)),
        ),
        (
            "rho/pion",
            Some(([1.5, 0.14142135623731], -28.6685223767356)),
        ),
        (
            "nucleon/pion",
            Some(([1.66666666666667, 0.173561103909037], -29.130409253245)),
        ),
        ("rho/f0_500", None),
        ("nucleon/f0_500", None),
        (
            "nucleon/rho",
            Some(([1.11111111111111, 0.115707402606025], -0.856955094588002)),
        ),
    ];
    for (row, (name, value)) in comparison.ratios.iter().zip(expected) {
        assert_eq!(label(row), name);
        match value {
            Some((pair, sigma)) => {
                assert!(pair_close(row.measured, pair), "{name}");
                let tension = row.tension_sigma.unwrap();
                assert!((tension - sigma).abs() < 1e-10 * sigma.abs(), "{name}");
            }
            None => assert!(
                row.measured.is_none() && row.tension_sigma.is_none(),
                "{name}"
            ),
        }
    }
    let gate: Vec<_> = comparison
        .notes
        .iter()
        .filter(|n| n.contains("less than 3 sigma from zero"))
        .collect();
    assert_eq!(gate.len(), 1);
    assert!(gate[0].starts_with("'f0_500': rate 0.2 +- 0.08"));
    let nucleon = &comparison.anchors[0];
    assert!(pair_close(nucleon.scale, [1876.544176, 150.12353408]));
    let predictions = [
        ("pion", [562.9632528, 58.6251141697141], 7.22203903215135),
        (
            "f0_500",
            [375.3088352, 153.096565944686],
            -0.681886210976367,
        ),
        ("rho", [844.4448792, 87.9376712545711], 0.786746357674096),
    ];
    for (p, (name, value, sigma)) in nucleon.predictions.iter().zip(predictions) {
        assert_eq!(p.name, name);
        assert!(pair_close(p.predicted, value), "{name}");
        assert!((p.tension_sigma.unwrap() - sigma).abs() < 1e-9, "{name}");
    }
    let scalar = &comparison.anchors[1];
    assert_eq!(scalar.anchor, "f0_500");
    assert!(scalar.scale.is_none() && scalar.predictions.is_empty());
    assert!(
        comparison
            .notes
            .iter()
            .any(|n| n.starts_with("Anchor 'f0_500' has no assigned channel with a usable rate"))
    );
    assert!(comparison.notes[3].contains("Of the 4 ratio rows only 3 are"));
}

#[test]
fn ratios_tensions_and_predictions_are_invariant_under_a_change_of_the_time_unit() {
    let frames = compare(&near_channels(), &four_names(&["nucleon", "pion", "rho"]))
        .unwrap()
        .unwrap();
    // One frame is dt = 0.01 time units: every rate and error is divided by dt.
    let dt = 0.01;
    let relabelled: Vec<ChannelReport> = near_channels()
        .into_iter()
        .map(|mut c| {
            let mass = c.mass.as_mut().unwrap();
            mass.value /= dt;
            mass.error /= dt;
            mass.statistical /= dt;
            mass.time_unit = TimeUnit::StepDt;
            c
        })
        .collect();
    let time = compare(&relabelled, &four_names(&["nucleon", "pion", "rho"]))
        .unwrap()
        .unwrap();
    let relative = |a: f64, b: f64| (a - b).abs() <= 4e-15 * b.abs();
    for (a, b) in frames.ratios.iter().zip(&time.ratios) {
        let (x, y) = (a.measured.unwrap(), b.measured.unwrap());
        assert!(relative(x[0], y[0]) && relative(x[1], y[1]));
        assert!((a.tension_sigma.unwrap() - b.tension_sigma.unwrap()).abs() < 1e-12);
    }
    for (a, b) in frames.anchors.iter().zip(&time.anchors) {
        // The lattice scale carries the unit; the rescaled rates do not.
        assert!(relative(a.scale.unwrap()[0] * dt, b.scale.unwrap()[0]));
        for (p, q) in a.predictions.iter().zip(&b.predictions) {
            let (x, y) = (p.predicted.unwrap(), q.predicted.unwrap());
            assert!(relative(x[0], y[0]) && relative(x[1], y[1]));
            assert!((p.tension_sigma.unwrap() - q.tension_sigma.unwrap()).abs() < 1e-12);
        }
    }
    for (a, b) in frames.anchor_spread.iter().zip(&time.anchor_spread) {
        assert_eq!(a.spread.is_some(), b.spread.is_some());
        if let Some((x, y)) = a.spread.zip(b.spread) {
            assert!((x - y).abs() < 1e-13);
        }
    }
    let scaled = ratio([0.56 / dt, 0.04 / dt], [0.10 / dt, 0.01 / dt], 0.);
    assert!(relative(scaled[0], 5.6) && relative(scaled[1], 0.68818602136341));
}

#[test]
fn a_cross_channel_correlation_changes_the_ratio_error_and_the_tension_as_derived() {
    let reference = ratio([775.26, 0.23], [139.57039, 0.00018], 0.);
    assert!(close(reference[1], 0.00164792957279818));
    for (c, error, sigma) in [
        (-0.5, 0.835224520712844, 0.0543367053133752),
        (0., 0.68818602136341, 0.0659462782784833),
        (0.5, 0.499599839871872, 0.090839080560763),
        (0.9, 0.265329983228432, 0.171041964918635),
    ] {
        let measured = ratio([0.56, 0.04], [0.10, 0.01], c);
        assert!(close(measured[0], 5.6) && close(measured[1], error), "{c}");
        assert!((tension(measured, reference).unwrap() - sigma).abs() < 1e-10);
    }
    // The signed linear tension is not antisymmetric under inversion of the
    // ratio, so the orientation is fixed by the table and never by the data.
    let inverted = tension(
        ratio([0.10, 0.01], [0.56, 0.04], 0.),
        ratio([139.57039, 0.00018], [775.26, 0.23], 0.),
    );
    assert!((inverted.unwrap() + 0.0664850794183002).abs() < 1e-10);
    // Without reference errors the magnitude is |R - R_ref|/sigma_R.
    let bare = tension(ratio([0.56, 0.04], [0.10, 0.01], 0.), [reference[0], 0.]);
    assert!((bare.unwrap() - 0.0659464673496368).abs() < 1e-10);
    let rescaled = prediction([0.56, 0.04], [0.70, 0.05], [938.272088, 0.0000003], 0.);
    assert!(close(rescaled[0], 750.6176704) && close(rescaled[1], 75.8238349740413));
}

#[test]
fn a_correlation_of_one_cancels_equal_relative_errors_without_a_negative_variance() {
    let [r, error] = ratio([2., 0.2], [4., 0.4], 1.);
    assert_eq!((r, error), (0.5, 0.));
    assert_eq!(tension([1., 0.], [1., 0.]), None);
    assert_eq!(spread(&[3.]), None);
    assert_eq!(spread(&[2., 2.]), Some(0.));
}

#[test]
fn nothing_measured_decides_a_row_an_order_or_an_assignment() {
    let analysis = four_names(&["nucleon"]);
    let near = compare(&near_channels(), &analysis).unwrap().unwrap();
    let far = compare(&far_channels(), &analysis).unwrap().unwrap();
    let order = |c: &Comparison| -> Vec<String> { c.ratios.iter().map(label).collect() };
    assert_eq!(order(&near), order(&far));
    // The measured columns do not read the reference values.
    let mut shifted = analysis.clone();
    for (i, entry) in shifted.reference.entries.iter_mut().enumerate() {
        entry.value *= 2. + i as f64;
        entry.error *= 0.5;
    }
    let moved = compare(&near_channels(), &shifted).unwrap().unwrap();
    for (a, b) in near.ratios.iter().zip(&moved.ratios) {
        assert_eq!(a.measured, b.measured);
        assert_ne!(a.reference, b.reference);
    }
    assert_eq!(
        near.reference
            .iter()
            .map(|r| r.measured)
            .collect::<Vec<_>>(),
        moved
            .reference
            .iter()
            .map(|r| r.measured)
            .collect::<Vec<_>>()
    );
    // Two keys for one name: the first key in key order with a rate is kept
    // even though the other one sits on the reference ratio to the pion.
    let mut double = analysis.clone();
    double
        .assignments
        .insert("meson/scalar/score_directed".into(), "f0_500".into());
    let mut channels = near_channels();
    channels[1].mass = Some(rate(0.10 * 500. / 139.57039, 0.01, TimeUnit::Frames));
    channels.push(channel("meson/scalar/score_directed", Some([0.35, 0.06])));
    let kept = compare(&channels, &double).unwrap().unwrap();
    assert_eq!(
        kept.reference[1].channel,
        "meson/scalar/score_directed/distance"
    );
    assert_eq!(kept.reference[1].measured, Some([0.35, 0.06]));
    assert_eq!(kept.ratios, near.ratios);
    assert!(kept.notes.iter().any(|n| n
        == "'meson/scalar/standard' is also assigned to 'f0_500' and is not used: the first \
            assignment with a rate is kept, never the best agreeing one."));
    // Without a rate on the first key the second one is taken.
    channels[4].mass = None;
    let second = compare(&channels, &double).unwrap().unwrap();
    assert_eq!(second.reference[1].channel, format!("{SIGMA}/distance"));
    assert!(second.ratios[0].tension_sigma.unwrap().abs() < 1e-3);
}

#[test]
fn channels_without_a_common_unit_estimator_or_scale_are_never_divided() {
    let analysis = four_names(&["nucleon"]);
    let mut channels = near_channels();
    channels[0].mass.as_mut().unwrap().time_unit = TimeUnit::StepDt;
    channels[2].estimator = Some(EstimatorKind::SourceFrozen);
    let comparison = compare(&channels, &analysis).unwrap().unwrap();
    let measured: Vec<bool> = comparison
        .ratios
        .iter()
        .map(|r| r.measured.is_some())
        .collect();
    // Only nucleon/f0_500 shares unit, estimator and scale.
    assert_eq!(measured, [false, false, false, false, true, false]);
    for pair in [
        "'pion' and 'f0_500'",
        "'pion' and 'rho'",
        "'f0_500' and 'rho'",
        "'rho' and 'nucleon'",
    ] {
        assert!(comparison.notes.iter().any(|n| n
            == &format!(
                "{pair} differ in time unit, estimator, scale or frame normalisation and are \
                 not compared."
            )));
    }
    assert_eq!(
        comparison.reference[2].estimator,
        Some(EstimatorKind::SourceFrozen)
    );
    assert!(comparison.notes[3].contains("Of the 1 ratio rows only 1 are"));
    let mut scaled = near_channels();
    scaled[3].scale = Some(0.4);
    let comparison = compare(&scaled, &analysis).unwrap().unwrap();
    assert!(
        comparison.anchors[0]
            .predictions
            .iter()
            .all(|p| p.predicted.is_none())
    );
    assert!(comparison.anchors[0].scale.is_some());
}

#[test]
fn missing_rates_are_explicit_gaps_and_invalid_assignments_are_configuration_errors() {
    let mut odd = channel(PION, None);
    odd.availability = Availability::unavailable(EXCHANGE_ODD_REASON);
    odd.estimator = None;
    let mut channels = near_channels();
    channels[0] = odd;
    let comparison = compare(&channels, &four_names(&["pion"])).unwrap().unwrap();
    assert_eq!(comparison.reference[0].measured, None);
    assert_eq!(comparison.reference[0].estimator, None);
    assert!(comparison.notes.iter().any(|n| n
        == &format!(
            "'{PION}/distance' assigned to 'pion' is exchange-odd and cancels on the mutual \
             pairing of this run; it reports no rate."
        )));
    assert!(comparison.anchors[0].scale.is_none());
    assert!(comparison.ratios[..3].iter().all(|r| r.measured.is_none()));
    let unrated: Vec<ChannelReport> = [PION, SIGMA].iter().map(|k| channel(k, None)).collect();
    assert_eq!(compare(&unrated, &AnalysisConfig::default()).unwrap(), None);
    assert_eq!(compare(&[], &AnalysisConfig::default()).unwrap(), None);
    let mut unknown = AnalysisConfig::default();
    unknown
        .assignments
        .insert(SIGMA.into(), "sigma_meson".into());
    assert!(matches!(
        compare(&near_channels(), &unknown),
        Err(GasError::Configuration(_))
    ));
    let mut negative = near_channels();
    negative[1].mass.as_mut().unwrap().value = -0.35;
    let comparison = compare(&negative, &four_names(&["nucleon"]))
        .unwrap()
        .unwrap();
    assert!(comparison.ratios[0].measured.is_none());
    assert!(comparison.ratios[3].measured.is_none());
    assert!(
        comparison
            .notes
            .iter()
            .any(|n| n.starts_with("'f0_500': rate -0.35"))
    );
    // An infinite rate is no anchor: it would imply a lattice scale of zero.
    let mut infinite = near_channels();
    infinite[3].mass.as_mut().unwrap().value = f64::INFINITY;
    let comparison = compare(&infinite, &four_names(&["nucleon"]))
        .unwrap()
        .unwrap();
    assert!(comparison.anchors[0].scale.is_none());
    assert!(comparison.ratios[5].measured.is_none());
    // The reference row of that rate is a gap and the comparison stays a
    // valid report value.
    assert_eq!(comparison.reference[3].measured, None);
    assert!(
        comparison
            .notes
            .iter()
            .any(|n| n.starts_with("'nucleon': rate inf +- 0.05 is not finite and positive"))
    );
    let text = serde_json::to_string(&comparison).unwrap();
    assert_eq!(
        serde_json::from_str::<Comparison>(&text).unwrap(),
        comparison
    );
    let mut electroweak = four_names(&["nucleon"]);
    electroweak
        .assignments
        .insert("u1/phase/q1".into(), "electron".into());
    let comparison = compare(&near_channels(), &electroweak).unwrap().unwrap();
    assert!(
        comparison
            .notes
            .iter()
            .any(|n| n.starts_with("Electroweak assignments follow the legacy dashboard"))
    );
    assert_eq!(comparison.reference[4].channel, "u1/phase/q1");
}

fn measurement(gas: GasConfig, dimension: usize, calibration: Option<Calibration>) -> Measurement {
    let config = MeasurementConfig::default();
    let capabilities =
        Capabilities::of(&gas, &RecordingConfig::default(), dimension).refine(&config);
    Measurement {
        schema_version: SPECTROSCOPY_VERSION,
        fingerprint: config.fingerprint(&gas, &capabilities, &[]).unwrap(),
        capabilities,
        config,
        gas,
        walkers: 16,
        calibration,
        steps: vec![],
        segment: vec![],
        segments: 0,
        ingested: 0,
        channels: vec![],
        flow: None,
        notes: vec![],
    }
}
fn viscous(dt: f64, coefficient: f64, bandwidth: f64) -> GasConfig {
    let viscosity = ViscousForceConfig {
        coefficient,
        bandwidth,
        row_normalized: false,
    };
    GasConfig::viscous_euclidean(3, dt, viscosity).unwrap()
}
fn row<'a>(rows: &'a [Quantity], name: &str) -> &'a Quantity {
    rows.iter().find(|q| q.name == name).unwrap()
}
fn value(rows: &[Quantity], name: &str) -> f64 {
    row(rows, name).value.unwrap()
}
fn thomson() -> StandardModelInputs {
    StandardModelInputs {
        alpha_em_inverse: [137.035999084, 0.],
        ..StandardModelInputs::default()
    }
}

#[test]
fn group_factors_and_target_couplings_reproduce_the_reference_values() {
    for (n, c2, ratio, factor) in [
        (2, 0.75, 1., 0.5),
        (3, 1.33333333333333, 0.5625, 2.),
        (4, 1.875, 0.4, 5.),
        (5, 2.4, 0.3125, 10.),
    ] {
        assert!((casimir(n) - c2).abs() < 1e-14);
        assert!(close(casimir(2) / casimir(n), ratio));
        assert!(close(kernel_factor(n), factor));
    }
    assert_eq!((casimir(1), kernel_factor(1)), (0., 0.));
    let low = TargetCouplings::of(&thomson()).unwrap();
    assert!(close(low.e_em[0], 0.302822120871753));
    assert!(close(low.g1[0], 0.345369302878055));
    assert!(close(low.g2[0], 0.629773366250774));
    assert!(close(low.g3[0], 1.21719969414757));
    assert_eq!(low.e_em[1], 0.);
    let at_mz = TargetCouplings::of(&StandardModelInputs::default()).unwrap();
    for ([value, error], target, sigma) in [
        (at_mz.e_em, 0.313388524592966, 1.10217845946366e-05),
        (at_mz.g1, 0.357420309840882, 1.56355928034258e-05),
        (at_mz.g2, 0.651748113741204, 6.08587906018469e-05),
        (at_mz.g3, 1.21719969414757, 0.00464580035934188),
    ] {
        assert!(close(value, target));
        assert!((error - sigma).abs() < 1e-12 * sigma);
    }
    for t in [low, at_mz] {
        let (e2, a, b) = (t.e_em[0].powi(2), t.g1[0].powi(2), t.g2[0].powi(2));
        assert!((a / (a + b) - 0.23121).abs() < 1e-15);
        assert!((1. / a + 1. / b - 1. / e2).abs() < 1e-14 / e2);
    }
    let invalid = StandardModelInputs {
        sin2_theta_w: [1.2, 0.],
        ..StandardModelInputs::default()
    };
    assert!(matches!(
        TargetCouplings::of(&invalid),
        Err(GasError::Configuration(_))
    ));
    assert!(matches!(
        couplings::report(&measurement(viscous(0.01, 1.5, 0.8), 3, None), &invalid),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn the_dictionary_inverts_standard_model_inputs_and_the_proxies_return_them() {
    let low = TargetCouplings::of(&thomson()).unwrap();
    let at_mz = TargetCouplings::of(&StandardModelInputs::default()).unwrap();
    // (couplings, d, m, h, N1, K2) -> (eps_c, eps_d, nu, eps_F, tau, rho).
    let cases = [
        (
            low,
            (3, 1., 1., 1., 1.),
            [
                1.68419343944988,
                2.89545130869111,
                0.860690157789941,
                10.9049783178775,
                1.41825377074301,
                0.890634035773204,
            ],
        ),
        (
            at_mz,
            (3, 1., 1., 1., 1.),
            [
                1.62740811889943,
                2.79782645940066,
                0.860690157789941,
                10.1820170617756,
                1.32422859272988,
                0.921711021703893,
            ],
        ),
        (
            low,
            (3, 91.1876, 1., 1., 1.),
            [
                1.68419343944988,
                2.89545130869111,
                0.860690157789941,
                994.398800859288,
                129.327157545005,
                0.00976705205283617,
            ],
        ),
        (
            low,
            (2, 1., 1., 1., 1.),
            [
                2.24559125259984,
                2.89545130869111,
                1.72138031557988,
                10.9049783178775,
                2.52134003687646,
                0.890634035773204,
            ],
        ),
        (
            low,
            (4, 1., 1., 1., 1.),
            [
                1.4202366103932,
                2.89545130869111,
                0.544348251661186,
                10.9049783178775,
                1.00853601475058,
                0.890634035773204,
            ],
        ),
        (
            low,
            (3, 0.75, 0.4, 0.37, 0.052),
            [
                1.0651774577949,
                1.11390235909085,
                1.5097504100566,
                8.17873373840814,
                1.06369032805726,
                0.751048563922989,
            ],
        ),
    ];
    for (targets, (d, m, h, n1, k2), expected) in cases {
        let solved = CalibrationTargets::of(&targets, d, m, h, n1, k2);
        let got = [
            solved.epsilon_c.unwrap(),
            solved.epsilon_d,
            solved.viscosity.unwrap(),
            solved.fitness_scale,
            solved.time_step.unwrap(),
            solved.viscous_range.unwrap(),
        ];
        for (a, b) in got.iter().zip(expected) {
            assert!(close(*a, b), "{a} {b}");
        }
        let [eps_c, eps_d, nu, eps_f, tau, rho] = got;
        let squares = [targets.g1, targets.g2, targets.g3, targets.e_em].map(|g| g[0] * g[0]);
        let forward = [
            g1_squared(h, n1, eps_d),
            g2_casimir_squared(h, eps_c, d).unwrap(),
            gd_squared(nu, h, d, k2).unwrap(),
            m / eps_f,
        ];
        for (a, b) in forward.iter().zip(squares) {
            assert!((a - b).abs() < 2e-15 * b);
        }
        assert!((g2_clock_squared(m, tau, rho, eps_c) - squares[1]).abs() < 2e-15 * squares[1]);
        assert!((kernel_action_scale(m, eps_c, tau) - h).abs() < 2e-15 * h);
        let c = casimir(2) / casimir(d);
        assert!((tau - m * c / squares[1]).abs() < 2e-15 * tau);
        assert!((eps_c / rho - m * c.sqrt() / squares[1]).abs() < 4e-15 * eps_c / rho);
    }
    let line = CalibrationTargets::of(&low, 1, 1., 1., 1., 1.);
    assert_eq!(
        (
            line.epsilon_c,
            line.viscosity,
            line.time_step,
            line.viscous_range
        ),
        (None, None, None, None)
    );
    assert!(close(line.epsilon_d, 2.89545130869111));
    // tau -> 4 tau, eps_c -> 2 eps_c, rho -> 2 rho keeps the kernel action
    // scale and rho/eps_c and moves the two weak proxies in opposite ways.
    let (eps_c, tau, rho) = (1.68419343944988, 1.41825377074301, 0.890634035773204);
    assert_eq!(
        g2_casimir_squared(1., 2. * eps_c, 3).unwrap() / g2_casimir_squared(1., eps_c, 3).unwrap(),
        0.25
    );
    assert_eq!(
        g2_clock_squared(1., 4. * tau, 2. * rho, 2. * eps_c)
            / g2_clock_squared(1., tau, rho, eps_c),
        4.
    );
    assert_eq!(
        kernel_action_scale(1., 2. * eps_c, 4. * tau) / kernel_action_scale(1., eps_c, tau),
        1.
    );
}

#[test]
fn the_report_evaluates_the_proxies_of_a_configuration_and_leaves_unmeasured_ones_empty() {
    let bare = couplings::report(
        &measurement(viscous(0.01, 1.5, 0.8), 3, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    for (name, expected) in [
        ("temperature", 0.5),
        ("kernel_action_scale", 200.),
        ("energy_clone", 0.5),
        ("energy_viscous", 1.25),
        ("energy_friction", 1.),
        ("separation", 2.5),
        ("viscosity", 1.5),
        ("viscous_range", 0.8),
        ("epsilon_d", 2.),
        ("epsilon_c", 2.),
        ("velocity_weight", 1.),
        ("noise_scale", 1.),
        ("position_diffusion", 0.1),
        ("clone_period", 1.),
    ] {
        assert!(close(value(&bare.scales, name), expected), "{name}");
    }
    for (name, expected) in [
        ("g1_upper", 0.5),
        ("g2_casimir", 0.530330085889911),
        ("g2_clock", 0.04),
        ("g2_clock_over_casimir", 0.00568888888888889),
        ("g3_upper", 2.12132034355964),
        ("sin2_theta_proxy_upper", 0.470588235294118),
        ("alpha_1_upper", 0.0198943678864869),
        ("alpha_2_casimir", 0.0223811638722978),
        ("alpha_2_clock", 0.000127323954473516),
        ("alpha_3_upper", 0.358098621956765),
    ] {
        assert!(close(value(&bare.couplings, name), expected), "{name}");
    }
    // No pair statistic without a calibration: a gap with its reason, never 0.
    for name in [
        "g1",
        "g3",
        "alpha_1",
        "alpha_3",
        "e_fitness",
        "alpha_fitness",
    ] {
        let q = row(&bare.couplings, name);
        assert_eq!(q.value, None, "{name}");
        assert!(q.definition.contains("undefined here: no "), "{name}");
    }
    assert_eq!(row(&bare.scales, "fitness_force_scale").value, None);
    assert_eq!(row(&bare.scales, "pair_statistic_n1").value, None);
    assert!(
        bare.notes
            .iter()
            .any(|n| n.starts_with("No warm-up calibration is available"))
    );
    let calibration = Calibration {
        mass: 1.,
        h_eff: 1.,
        electroweak_h_eff: 1.,
        h_s: 1.,
        epsilon_d: Some(2.),
        epsilon_c: Some(2.),
        dt: Some(0.01),
        pair_weight_n1: Some(0.641827002438084),
        viscous_kernel_second_moment: Some(0.276169780908252),
        ..Calibration::default()
    };
    let measured = couplings::report(
        &measurement(viscous(0.01, 1.5, 0.8), 3, Some(calibration)),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert!(close(value(&measured.couplings, "g1"), 0.400570531379333));
    assert!(close(
        value(&measured.couplings, "alpha_1"),
        0.0127687425059844
    ));
    assert!(close(value(&measured.couplings, "g3"), 1.11479326069327));
    assert!(close(
        value(&measured.couplings, "alpha_3"),
        0.0988960179693466
    ));
    assert!(value(&measured.couplings, "g1") <= value(&measured.couplings, "g1_upper"));
    assert!(value(&measured.couplings, "g3") <= value(&measured.couplings, "g3_upper"));
    let n1 = row(&measured.scales, "pair_statistic_n1");
    assert_eq!(n1.value, Some(0.641827002438084));
    assert!(n1.definition.contains("exp(-D^2/epsilon_d^2)"));
    assert!(n1.definition.contains("first distance-companion pairs"));
    let k2 = row(&measured.scales, "kernel_second_moment");
    assert!(k2.definition.contains("unnormalised kernel"));
    assert!(k2.definition.contains("ordered eligible pairs"));
    for report in [&bare, &measured] {
        let all = [&report.scales, &report.couplings];
        assert!(
            all.iter()
                .all(|rows| rows.iter().all(|q| q.error.is_none()))
        );
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("Their ratio clock/Casimir here is 0.00568888888888"))
        );
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("would give m epsilon_c^2/(2 tau) = 200"))
        );
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("divided by the eligible population"))
        );
    }
}

#[test]
fn the_calibration_is_an_inversion_of_inputs_and_never_a_result_of_the_run() {
    let bare = measurement(viscous(0.01, 1.5, 0.8), 3, None);
    let report = couplings::report(&bare, &StandardModelInputs::default()).unwrap();
    let names: Vec<&str> = report.inversion.iter().map(|q| q.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "alpha_em",
            "sin2_theta_w",
            "alpha_s",
            "e_em",
            "g1",
            "g2",
            "g3",
            "dimension",
            "mass",
            "action_scale",
            "pair_statistic_n1",
            "kernel_second_moment",
            "target_epsilon_c",
            "target_epsilon_d",
            "target_viscosity",
            "target_fitness_scale",
            "target_time_step",
            "target_viscous_range",
        ]
    );
    for q in &report.inversion {
        let role = if q.name.starts_with("target_") {
            "target: "
        } else {
            "input: "
        };
        assert!(q.definition.starts_with(role), "{}", q.name);
    }
    assert!(close(
        value(&report.inversion, "alpha_em"),
        0.00781549186798071
    ));
    let alpha_error = row(&report.inversion, "alpha_em").error.unwrap();
    assert!((alpha_error - 5.49737218246254e-07).abs() < 1e-18);
    assert!(
        row(&report.inversion, "alpha_em")
            .definition
            .contains("PDG 2022")
    );
    for (name, expected) in [
        ("target_epsilon_c", 1.62740811889943),
        ("target_epsilon_d", 2.79782645940066),
        ("target_viscosity", 0.860690157789941),
        ("target_fitness_scale", 10.1820170617756),
        ("target_time_step", 1.32422859272988),
        ("target_viscous_range", 0.921711021703893),
    ] {
        assert!(close(value(&report.inversion, name), expected), "{name}");
    }
    assert!(
        row(&report.inversion, "target_epsilon_d")
            .definition
            .ends_with("an upper bound")
    );
    assert!(
        row(&report.inversion, "target_viscosity")
            .definition
            .ends_with("a lower bound")
    );
    assert!(
        row(&report.inversion, "pair_statistic_n1")
            .definition
            .contains("placeholder 1")
    );
    assert!(report.notes[2].starts_with("Inversion: Standard Model inputs (M_Z; g1 = gY)"));
    assert!(
        report.notes[2].contains("Targets are inputs of a calibration, not results of this run")
    );
    assert!(report.notes[3].contains("N1 at the placeholder value 1"));
    // The targets do not read the gas: another variant at the same dimension
    // and analysis parameters has the same inversion.
    let other = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    let elsewhere = couplings::report(
        &measurement(other, 3, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert_eq!(elsewhere.inversion, report.inversion);
    assert_ne!(elsewhere.scales, report.scales);
    // The scales and proxies do not read the Standard Model inputs.
    let moved = couplings::report(&bare, &thomson()).unwrap();
    assert_eq!(moved.scales, report.scales);
    assert_eq!(moved.couplings, report.couplings);
    assert_ne!(moved.inversion, report.inversion);
    // Measured pair statistics, another mass and action scale.
    let mut custom = bare.clone();
    custom.calibration = Some(Calibration {
        mass: 0.75,
        h_eff: 0.4,
        electroweak_h_eff: 0.4,
        h_s: 0.4,
        epsilon_d: Some(2.),
        epsilon_c: Some(2.),
        dt: Some(0.01),
        pair_weight_n1: Some(0.37),
        viscous_kernel_second_moment: Some(0.052),
        ..Calibration::default()
    });
    let held = couplings::report(&custom, &thomson()).unwrap();
    for (name, expected) in [
        ("target_epsilon_c", 1.0651774577949),
        ("target_epsilon_d", 1.11390235909085),
        ("target_viscosity", 1.5097504100566),
        ("target_fitness_scale", 8.17873373840814),
        ("target_time_step", 1.06369032805726),
        ("target_viscous_range", 0.751048563922989),
    ] {
        assert!(close(value(&held.inversion, name), expected), "{name}");
    }
    assert!(
        row(&held.inversion, "pair_statistic_n1")
            .definition
            .contains("held fixed")
    );
    assert!(
        !row(&held.inversion, "target_epsilon_d")
            .definition
            .contains("bound")
    );
    assert!(held.notes[3].contains("N1 at 0.37, its warm-up value"));
    let unified = StandardModelInputs {
        hypercharge: HyperchargeNormalization::Unified,
        ..StandardModelInputs::default()
    };
    let quoted = couplings::report(&bare, &unified).unwrap();
    assert!(close(
        value(&quoted.inversion, "g1_unified"),
        0.46142763587001
    ));
    assert!(close(value(&quoted.inversion, "g1"), 0.357420309840882));
    assert_eq!(
        value(&quoted.inversion, "target_epsilon_d"),
        value(&report.inversion, "target_epsilon_d")
    );
    // Configuring the gas at the targets is a choice: the two weak proxies
    // and the kernel action scale then return the inputs.
    let low = TargetCouplings::of(&thomson()).unwrap();
    let solved = CalibrationTargets::of(&low, 3, 1., 1., 1., 1.);
    let mut tuned = viscous(
        solved.time_step.unwrap(),
        solved.viscosity.unwrap(),
        solved.viscous_range.unwrap(),
    );
    tuned.distance_donors.kernel = Kernel::Gaussian {
        width: solved.epsilon_d,
    };
    tuned.cloning_donors.kernel = Kernel::Gaussian {
        width: solved.epsilon_c.unwrap(),
    };
    let tuned = couplings::report(&measurement(tuned, 3, None), &thomson()).unwrap();
    for (proxy, input) in [
        ("g1_upper", "g1"),
        ("g2_casimir", "g2"),
        ("g2_clock", "g2"),
        ("g3_upper", "g3"),
    ] {
        let (a, b) = (
            value(&tuned.couplings, proxy),
            value(&tuned.inversion, input),
        );
        assert!((a - b).abs() < 4e-15 * b, "{proxy}");
    }
    assert!((value(&tuned.scales, "kernel_action_scale") - 1.).abs() < 4e-15);
    let text = serde_json::to_string(&tuned).unwrap();
    for forbidden in [
        "\u{2713}",
        "tension_sigma",
        "Exact",
        "MeV",
        "GeV",
        "validated",
    ] {
        assert!(!text.contains(forbidden), "{forbidden}");
    }
    assert_eq!(
        serde_json::from_str::<CouplingReport>(&text).unwrap(),
        tuned
    );
}

/// The audit's calibration pair: `nu = 0.948271` with `l_visc = 0.00976705`
/// was advertised as `alpha_s = 0.1179` while the kernel at that width has
/// `<K_visc^2> = 1.3596989447251e-05`, so the realised coupling is
/// `1.945932764315832e-06`, sixty thousand times smaller. Nothing in the
/// reference pipeline stated or checked the moment `nu` presupposes.
#[test]
fn a_configured_viscosity_states_the_kernel_moment_it_presupposes_and_flags_a_disagreement() {
    let inputs = StandardModelInputs::default();
    let g3 = TargetCouplings::of(&inputs).unwrap().g3[0];
    let presupposed = presupposed_kernel_second_moment(0.948271, 1., 3, g3).unwrap();
    assert!(close(presupposed, 0.823813178557852));
    // It is the proxy solved for the moment: at it, nu returns the input.
    assert!(close(
        gd_squared(0.948271, 1., 3, presupposed).unwrap(),
        g3 * g3
    ));
    assert_eq!(presupposed_kernel_second_moment(0., 1., 3, g3), None);
    assert_eq!(presupposed_kernel_second_moment(0.948271, 1., 1, g3), None);

    let measured = 1.3596989447251e-05;
    let calibrated = |k2: f64| Calibration {
        mass: 1.,
        h_eff: 1.,
        electroweak_h_eff: 1.,
        h_s: 1.,
        dt: Some(0.01),
        viscous_kernel_second_moment: Some(k2),
        ..Calibration::default()
    };
    let gas = viscous(0.01, 0.948271, 0.00976705);
    let advertised = couplings::report(
        &measurement(gas.clone(), 3, Some(calibrated(measured))),
        &inputs,
    )
    .unwrap();
    assert!(close(
        value(&advertised.couplings, "g3"),
        0.004945029050169188
    ));
    assert!(close(
        value(&advertised.couplings, "alpha_3"),
        1.945932764315832e-06
    ));
    let check = |report: &CouplingReport| -> String {
        report
            .notes
            .iter()
            .find(|n| n.starts_with("Calibration check:"))
            .unwrap()
            .clone()
    };
    let note = check(&advertised);
    assert!(note.contains(&format!("nu = {}", 0.948271)));
    assert!(note.contains(&format!("presupposes <K_visc^2> = {presupposed}")));
    assert!(note.contains(&format!(
        "rho = {} measured <K_visc^2> = {measured}",
        0.00976705
    )));
    assert!(note.contains(&format!("a factor {}", presupposed / measured)));
    assert!(note.contains(&format!("realised alpha_3 is {}", 1.945932764315832e-06)));
    let flag = advertised
        .notes
        .iter()
        .find(|n| n.starts_with("The viscous calibration of this run is inconsistent"))
        .unwrap();
    assert!(flag.contains(&format!("is not the input alpha_s = {}", 0.1179)));
    assert!(flag.contains("meaningful only together with the <K_visc^2> of the kernel that runs"));

    // At the moment it presupposes, the pair is consistent: the realised
    // proxy is the input and no flag is raised. A factor of two is the
    // tolerance and is still accepted.
    for k2 in [presupposed, presupposed / 2., 2. * presupposed] {
        let consistent =
            couplings::report(&measurement(gas.clone(), 3, Some(calibrated(k2))), &inputs).unwrap();
        assert!(
            !consistent
                .notes
                .iter()
                .any(|n| n.starts_with("The viscous calibration"))
        );
        assert!(check(&consistent).contains("Calibration check:"));
    }
    assert!(close(
        value(
            &couplings::report(
                &measurement(gas.clone(), 3, Some(calibrated(presupposed))),
                &inputs
            )
            .unwrap()
            .couplings,
            "alpha_3"
        ),
        0.1179
    ));
    // Without a warm-up moment nothing is checked, and the check says so
    // instead of assuming the presupposition holds.
    let bare = couplings::report(&measurement(gas, 3, None), &inputs).unwrap();
    let note = check(&bare);
    assert!(note.contains(&format!("realises the input alpha_s = {} only if", 0.1179)));
    assert!(note.contains("no warm-up measurement of <K_visc^2>"));
    assert_eq!(row(&bare.couplings, "alpha_3").value, None);
}

/// What the recorded geodesic lengths are made of. The reference
/// implementation recorded a metric built from neighbour displacements alone,
/// with a zero fitness field, and called its distances a fitness manifold.
#[test]
fn the_recorded_metric_is_named_and_no_parameter_row_claims_a_fitness_manifold() {
    let geodesic = Calibration {
        length: 0.25,
        length_source: "warmup_edge_mean_geodesic".into(),
        mass: 1.,
        h_eff: 1.,
        electroweak_h_eff: 1.,
        h_s: 1.,
        ..Calibration::default()
    };
    let gas = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    assert!(matches!(
        gas.geometry.as_ref().unwrap().pipeline.metric,
        MetricKind::NeighborCovariance { .. }
    ));
    let report = couplings::report(
        &measurement(gas.clone(), 3, Some(geodesic.clone())),
        &StandardModelInputs::default(),
    )
    .unwrap();
    let named = |report: &CouplingReport| -> String {
        report
            .notes
            .iter()
            .find(|n| n.starts_with("The recorded tessellation metric is"))
            .unwrap()
            .clone()
    };
    let note = named(&report);
    assert!(note.contains("the inverse neighbour covariance"));
    assert!(note.contains("it reads no fitness"));
    assert!(note.contains("not distances of the fitness manifold"));
    assert!(note.contains(&format!("l0 = {} is a mean of its edge lengths", 0.25)));
    let length = row(&report.scales, "phase_length");
    assert_eq!(length.value, Some(0.25));
    assert!(length.definition.contains("warmup_edge_mean_geodesic"));
    assert!(length.definition.contains("kappa = m l0/h_eff"));
    // No row of either parameter table claims what the metric does not carry.
    for rows in [&report.scales, &report.couplings] {
        assert!(rows.iter().all(|q| {
            !q.definition.contains("fitness manifold") && !q.name.contains("fitness_manifold")
        }));
    }
    // The companion-median arm measures no geodesic, so the sentence about
    // the edge lengths is dropped while the metric is still named.
    let median = Calibration {
        length_source: "warmup_companion_median".into(),
        ..geodesic.clone()
    };
    let report = couplings::report(
        &measurement(gas.clone(), 3, Some(median)),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert!(!named(&report).contains("edge lengths"));
    assert!(
        row(&report.scales, "phase_length")
            .definition
            .contains("warmup_companion_median")
    );
    // The Hessian arm is the one the book's definition asks for, and is named
    // as such; a gas with no geometry stage has no such note, and without a
    // calibration the length is an explicit gap.
    let mut hessian = gas;
    hessian.geometry.as_mut().unwrap().pipeline.metric = MetricKind::HessianFd {
        scalar_field: "fitness".into(),
        full: true,
        epsilon_sigma: 1e-3,
        min_eig: Some(1e-6),
        max_eig: None,
        policy: Default::default(),
    };
    let report = couplings::report(
        &measurement(hessian, 3, Some(geodesic)),
        &StandardModelInputs::default(),
    )
    .unwrap();
    let note = named(&report);
    assert!(note.contains("a finite-difference Hessian of `fitness`"));
    assert!(note.contains("fitness-manifold arm of def-adaptive-diffusion-tensor-latent"));
    let flat = couplings::report(
        &measurement(viscous(0.01, 1.5, 0.8), 3, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert!(
        !flat
            .notes
            .iter()
            .any(|n| n.starts_with("The recorded tessellation metric"))
    );
    let length = row(&flat.scales, "phase_length");
    assert_eq!(length.value, None);
    assert!(
        length
            .definition
            .contains("undefined here: no warm-up calibration")
    );
}

#[test]
fn every_implemented_variant_yields_its_scales_and_explains_what_it_lacks() {
    for variant in Variant::all().iter().filter(|v| v.implemented()) {
        let gas = variant.default_config().unwrap();
        let d = variant.reference().unwrap().dimensions;
        let report =
            couplings::report(&measurement(gas, d, None), &StandardModelInputs::default()).unwrap();
        let rows = [&report.scales, &report.couplings, &report.inversion];
        assert!(rows.iter().all(|r| {
            r.iter()
                .all(|q| q.value.is_none_or(f64::is_finite) && !q.definition.is_empty())
        }));
    }
    let hilbert = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    let scales = AlgorithmicScales::from_config(&hilbert, 3);
    assert_eq!(
        scales,
        AlgorithmicScales {
            dimension: 3,
            dt: Some(0.002),
            friction: Some(1.),
            temperature: scales.temperature,
            viscosity: Some(3.),
            epsilon_d: None,
            epsilon_c: None,
            epsilon_clone: 0.,
        }
    );
    assert!((scales.temperature.unwrap() - 0.33).abs() < 1e-15);
    let report = couplings::report(
        &measurement(hilbert, 3, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert_eq!(value(&report.scales, "clone_period"), 20.);
    assert_eq!(row(&report.scales, "viscous_range").value, None);
    for name in ["g1_upper", "g2_casimir", "g2_clock", "g3", "g3_upper"] {
        assert_eq!(row(&report.couplings, name).value, None, "{name}");
    }
    for role in ["distance", "cloning"] {
        assert!(report.notes.iter().any(|n| n.starts_with(&format!(
            "The {role} companion kernel is uniform, not Gaussian"
        ))));
    }
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.starts_with("Graph viscosity uses tessellation"))
    );
    let euclidean = GasConfig::euclidean(2, 0.04).unwrap();
    assert_eq!(
        AlgorithmicScales::from_config(&euclidean, 2),
        AlgorithmicScales {
            dimension: 2,
            dt: Some(0.04),
            friction: Some(1.),
            temperature: Some(0.5),
            viscosity: None,
            epsilon_d: Some(2.),
            epsilon_c: Some(2.),
            epsilon_clone: euclidean.clone_decision.epsilon,
        }
    );
    let report = couplings::report(
        &measurement(euclidean, 2, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.contains("squashed phase-space distance"))
    );
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.contains("at d = 2 the proxy g_d defines"))
    );
    assert_eq!(row(&report.couplings, "gd_upper").value, None);
    assert!(
        row(&report.couplings, "gd_upper")
            .definition
            .contains("no dense viscosity")
    );
    let reference = Variant::ViscousEuclidean.default_config().unwrap();
    let scales = AlgorithmicScales::from_config(&reference, 3);
    assert_eq!(scales.viscosity, Some(0.3));
}

#[test]
fn ranges_follow_the_distance_scaling_and_the_thermostat_target_is_the_o_step_fixed_point() {
    let mut gas = GasConfig::euclidean(3, 0.01).unwrap();
    gas.distance_donors.kernel = Kernel::Gaussian { width: 3. };
    gas.distance_donors.distance = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 0.5,
        velocity_scale: 2.,
        lambda: 0.25,
        periodic: None,
    };
    assert_eq!(AlgorithmicScales::from_config(&gas, 3).epsilon_d, Some(1.5));
    let report = couplings::report(
        &measurement(gas.clone(), 3, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert_eq!(value(&report.scales, "velocity_weight"), 0.015625);
    assert!(
        row(&report.scales, "velocity_weight")
            .definition
            .contains("the cloning role differs")
    );
    let euclidean = |scales: Vec<f64>, field: &str| Distance::Euclidean {
        field: field.into(),
        scales,
        squared: false,
        periodic: None,
    };
    gas.distance_donors.distance = euclidean(vec![2., 2., 2.], "positions");
    assert_eq!(AlgorithmicScales::from_config(&gas, 3).epsilon_d, Some(6.));
    gas.distance_donors.distance = euclidean(vec![], "positions");
    assert_eq!(AlgorithmicScales::from_config(&gas, 3).epsilon_d, Some(3.));
    // A missing scale is 1: two explicit scales of 2 leave the third unequal.
    gas.distance_donors.distance = euclidean(vec![2., 2.], "positions");
    assert_eq!(AlgorithmicScales::from_config(&gas, 3).epsilon_d, None);
    gas.distance_donors.distance = euclidean(vec![], "velocities");
    assert_eq!(AlgorithmicScales::from_config(&gas, 3).epsilon_d, None);
    let report = couplings::report(
        &measurement(gas.clone(), 3, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert!(report.notes.iter().any(|n| n.starts_with(
        "The distance companion distance is anisotropic or not spatial: epsilon_d is undefined"
    )));
    gas.distance_donors.kernel = Kernel::Exponential { temperature: 1. };
    assert_eq!(AlgorithmicScales::from_config(&gas, 3).epsilon_d, None);
    // A fixed measurement range wins over the kernel and is stated.
    let mut fixed = measurement(GasConfig::euclidean(3, 0.01).unwrap(), 3, None);
    fixed.config.electroweak.epsilon_c = Range::Fixed { value: 0.5 };
    fixed.fingerprint = fixed
        .config
        .fingerprint(&fixed.gas, &fixed.capabilities, &[])
        .unwrap();
    let report = couplings::report(&fixed, &StandardModelInputs::default()).unwrap();
    assert_eq!(value(&report.scales, "epsilon_c"), 0.5);
    assert!(
        report.notes.iter().any(
            |n| n == "The measurement used epsilon_c = 0.5 while the companion kernel gives 2."
        )
    );
    // v <- c v + s sigma xi with c = exp(-gamma h), s^2 = (1 - c^2)/(2 gamma).
    let (gamma, h, sigma) = (1.3f64, 0.01f64, 0.7f64);
    let c = (-gamma * h).exp();
    let s = (-(-2. * gamma * h).exp_m1() / (2. * gamma)).sqrt();
    assert!(close(c, 0.987084135020288) && close(s, 0.0993535071411687));
    let stationary = sigma * sigma * s * s / (1. - c * c);
    let mut thermostat = GasConfig::euclidean(3, h).unwrap();
    thermostat.kinetic.integrator = KineticKind::Baoab {
        positions: "positions".into(),
        velocities: "velocities".into(),
        dt: h,
        friction: gamma,
    };
    thermostat.kinetic.noise.geometry = NoiseGeometry::Isotropic {
        scale: FactorValues::Constant {
            values: vec![sigma],
        },
    };
    let temperature = AlgorithmicScales::from_config(&thermostat, 3)
        .temperature
        .unwrap();
    assert!((temperature - stationary).abs() < 1e-14);
    assert!(close(temperature, 0.188461538461538));
    thermostat.kinetic.noise.geometry = NoiseGeometry::Diagonal {
        factor: FactorValues::Constant {
            values: vec![sigma; 3],
        },
    };
    assert_eq!(
        AlgorithmicScales::from_config(&thermostat, 3).temperature,
        None
    );
}

#[test]
fn a_line_has_no_casimir_or_strong_proxy_and_the_dictionary_no_solution() {
    let mut gas = GasConfig::viscous_euclidean(
        1,
        0.01,
        ViscousForceConfig {
            coefficient: 1.5,
            bandwidth: 0.8,
            row_normalized: true,
        },
    )
    .unwrap();
    gas.precision = Precision::F64;
    let report =
        couplings::report(&measurement(gas, 1, None), &StandardModelInputs::default()).unwrap();
    for name in [
        "g2_casimir",
        "gd_upper",
        "alpha_d_upper",
        "sin2_theta_proxy_upper",
    ] {
        let q = row(&report.couplings, name);
        assert_eq!(q.value, None, "{name}");
    }
    assert!(
        row(&report.couplings, "gd_upper")
            .definition
            .contains("no d >= 2")
    );
    for name in [
        "target_epsilon_c",
        "target_viscosity",
        "target_time_step",
        "target_viscous_range",
    ] {
        let q = row(&report.inversion, name);
        assert_eq!(q.value, None, "{name}");
        assert!(q.definition.contains("undefined here: no d >= 2"), "{name}");
    }
    assert!(row(&report.inversion, "target_epsilon_d").value.is_some());
    assert!(
        row(&report.inversion, "target_fitness_scale")
            .value
            .is_some()
    );
    assert!(close(value(&report.couplings, "g2_clock"), 0.04));
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.contains("divided by the row mass"))
    );
}

fn correlator(value: Vec<Option<f64>>, error: Vec<Option<f64>>) -> CorrelatorEstimate {
    CorrelatorEstimate {
        lags: (0..value.len()).collect(),
        time_unit: TimeUnit::StepDt,
        time_step: 0.5,
        value,
        error,
        covariance: None,
        samples_meta: SamplesMeta {
            resampling: ResampleKind::Jackknife,
            effective_block: 8,
            blocks: 16,
            tau_int: None,
            covariance_rank: 4,
            replicas: 1,
            sampling_unit: "time blocks".into(),
        },
        connected: true,
        connected_bias: None,
    }
}
fn report() -> SpectroscopyReport {
    let bare = measurement(viscous(0.01, 1.5, 0.8), 3, None);
    let analysis = AnalysisConfig::default();
    let mut channels = near_channels();
    channels[1].correlator = Some(correlator(
        vec![Some(1.), Some(0.5), None, Some(0.125)],
        vec![Some(0.1), None, None, Some(0.025)],
    ));
    channels[1].effective_mass = Some(vec![Some([0.7, 0.1]), None, None, None]);
    let mut odd = channel("vector/axial/full/raw", None);
    odd.availability = Availability::unavailable(EXCHANGE_ODD_REASON);
    channels.push(odd);
    SpectroscopyReport {
        schema_version: SPECTROSCOPY_VERSION,
        calculation_origin: CALCULATION_ORIGIN.into(),
        precision: Precision::F64,
        measurement_fingerprint: bare.fingerprint.clone(),
        comparison: compare(&channels, &analysis).unwrap(),
        couplings: Some(couplings::report(&bare, &analysis.standard_model).unwrap()),
        analysis,
        capabilities: bare.capabilities.clone(),
        calibration: None,
        replicas: 1,
        frames: 128,
        channels,
        groups: vec![],
        gevp: vec![],
        flow: None,
        notes: vec!["report note".into()],
    }
}

#[test]
fn a_presentation_draws_gaps_for_undefined_points_and_carries_the_provenance() {
    let report = report();
    let results = present(&report).unwrap();
    let titles: Vec<&str> = results.iter().map(|r| r.title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "Spectroscopy report",
            "meson/pseudoscalar/standard/distance",
            "meson/scalar/standard/distance",
            "vector/vector/full/raw/distance",
            "baryon/complex/distance",
            "Reference comparison (hypothesis mapping)",
            "Algorithmic scales",
            "Coupling proxies",
            "Calibration inversion",
        ]
    );
    for result in &results {
        assert_eq!((result.experiment, EXPERIMENT), (0, 0));
        let details = result.details.as_object().unwrap();
        assert_eq!(
            details.keys().map(String::as_str).collect::<Vec<_>>(),
            [
                "calculation_origin",
                "precision",
                "request",
                "schema_version"
            ]
        );
        assert_eq!(details["calculation_origin"], "executed_algorithm_archive");
        assert_eq!(details["precision"], "f64");
        assert_eq!(details["schema_version"], SPECTROSCOPY_VERSION);
        assert_eq!(
            details["request"],
            serde_json::to_value(&report.analysis).unwrap()
        );
    }
    assert_eq!(
        results[0].notes,
        [
            "report note".into(),
            format!("vector/axial/full/raw/distance: unavailable: {EXCHANGE_ODD_REASON}"),
        ]
    );
    assert!(
        results
            .iter()
            .all(|r| r.title != "vector/axial/full/raw/distance")
    );
    let scalar = &results[2];
    assert_eq!(scalar.model, RATE_QUANTITY);
    assert_eq!(scalar.metrics[0].value, Some(0.35));
    assert_eq!(scalar.metrics[0].unit, "1/frame");
    let plot = &scalar.plots[0];
    assert_eq!(
        (plot.title.as_str(), plot.x_label.as_str()),
        ("Correlator", "lag (time)")
    );
    let shape: Vec<(&str, &[[f64; 2]])> = plot
        .series
        .iter()
        .map(|s| (s.name.as_str(), s.points.as_slice()))
        .collect();
    assert_eq!(
        shape,
        [
            ("C", &[[0., 1.], [0.5, 0.5]][..]),
            ("C", &[[1.5, 0.125]][..]),
            ("C + error", &[[0., 1.1]][..]),
            ("C + error", &[[1.5, 0.15]][..]),
            ("C - error", &[[0., 0.9]][..]),
            ("C - error", &[[1.5, 0.1]][..]),
        ]
    );
    assert_eq!(
        scalar.notes,
        [
            "Exchange parity: not defined; spatial parity: not verified.",
            "Frame normalisation: not stated by the measurement. The valid-count and fixed-N \
             averages are different observables and only the fixed-N one has a transfer-matrix \
             reading.",
            "Errors are resampling errors over time blocks."
        ]
    );
    // The resampling behind every error bar is stated in numbers.
    let of = |label: &str| scalar.metrics.iter().find(|m| m.label == label).unwrap();
    assert_eq!(of("block length").value, Some(8.));
    assert_eq!(of("blocks").value, Some(16.));
    assert_eq!(of("covariance rank").value, Some(4.));
    assert_eq!(of("lags").value, Some(4.));
    assert_eq!(of("tau_int").value, None);
    assert_eq!(of("connected bias").value, None);
    // A channel is named by its id; particle names live in the comparison only.
    for result in &results[1..5] {
        let text = serde_json::to_string(&(&result.title, &result.model, &result.metrics)).unwrap();
        for name in report.analysis.assignments.values() {
            assert!(!text.contains(name.as_str()), "{name} in {}", result.title);
        }
    }
    let effective = &scalar.plots[1];
    assert_eq!(effective.series.len(), 3);
    assert!(effective.series.iter().all(|s| s.points.len() == 1));
    // A missing rate and a missing proxy are metrics without a value.
    let comparison = &results[5];
    let of = |label: &str| {
        comparison
            .metrics
            .iter()
            .find(|m| m.label == label)
            .unwrap()
            .value
    };
    assert_eq!(of("a1/pion measured"), None);
    assert_eq!(of("a1/pion tension"), None);
    assert!(of("a1/pion reference").is_some());
    assert!(of("rho/pion tension").is_some());
    assert_eq!(comparison.notes, report.comparison.as_ref().unwrap().notes);
    let ratios = &comparison.plots[0];
    let measured: usize = ratios
        .series
        .iter()
        .filter(|s| s.name == "measured")
        .map(|s| s.points.len())
        .sum();
    assert_eq!(measured, 6);
    assert!(
        results
            .iter()
            .flat_map(|r| &r.plots)
            .flat_map(|p| &p.series)
            .all(|s| { s.kind == "line" && !s.points.is_empty() })
    );
    let proxies = &results[7];
    let g1 = proxies.metrics.iter().find(|m| m.label == "g1").unwrap();
    assert_eq!(g1.value, None);
    assert!(proxies.notes.iter().any(|n| n.starts_with("g1: ")));
    let inversion = &results[8];
    assert!(inversion.model.contains("not results of this run"));
    assert!(
        inversion
            .notes
            .iter()
            .any(|n| n.starts_with("target_epsilon_c: target: "))
    );
    assert!(
        inversion
            .notes
            .iter()
            .any(|n| n.starts_with("alpha_em: input: "))
    );
}

#[test]
fn an_invalid_report_is_rejected_before_anything_is_drawn() {
    let mut nonfinite = report();
    nonfinite.channels[1].correlator.as_mut().unwrap().value[0] = Some(f64::INFINITY);
    assert!(matches!(
        present(&nonfinite),
        Err(GasError::Configuration(_))
    ));
    let mut versioned = report();
    versioned.schema_version += 1;
    assert!(matches!(present(&versioned), Err(GasError::Checkpoint(_))));
    let mut origin = report();
    origin.calculation_origin = "synthetic".into();
    assert!(matches!(present(&origin), Err(GasError::Configuration(_))));
}

#[test]
fn an_analysis_attaches_the_coupling_report_and_no_comparison_without_a_rate() {
    let bare = measurement(viscous(0.01, 1.5, 0.8), 3, None);
    let report = analyze(&[bare], &AnalysisConfig::default()).unwrap();
    assert_eq!(report.comparison, None);
    let couplings = report.couplings.as_ref().unwrap();
    assert!(close(
        value(&couplings.couplings, "g2_casimir"),
        0.530330085889911
    ));
    assert!(
        report
            .notes
            .iter()
            .all(|n| !n.starts_with("no coupling report"))
    );
    let results = present(&report).unwrap();
    assert_eq!(results.len(), 4);
}

/// `09_qft_calibration` asks every reported rate to state the frame
/// normalization it used, because the valid-count average of chapter 04 and
/// the fixed-`1/N` average of chapter 08 are two different observables. The
/// reference implementation divided by a frame-varying denominator that is
/// neither, and printed nothing.
#[test]
fn every_frame_mean_rate_states_its_normalisation_and_two_of_them_are_never_divided() {
    let mut channels = near_channels();
    channels[0].normalization = Some(FrameNormalization::ValidCount);
    channels[1].normalization = Some(FrameNormalization::FixedN);
    channels[2].estimator = Some(EstimatorKind::SourceFrozen);
    channels[3].estimator = Some(EstimatorKind::EuclideanTime);
    let mut report = report();
    report.channels = channels.clone();
    let results = present(&report).unwrap();
    let notes = |title: &str| -> Vec<String> {
        results
            .iter()
            .find(|r| r.title == title)
            .unwrap()
            .notes
            .clone()
    };
    let stated = |notes: &[String]| -> Option<String> {
        notes
            .iter()
            .find(|n| n.starts_with("Frame normalisation:"))
            .cloned()
    };
    let first = stated(&notes(&format!("{PION}/distance"))).unwrap();
    assert!(first.starts_with("Frame normalisation: the sum of valid element weights (04)."));
    assert!(first.contains("only the fixed-N one has a transfer-matrix reading"));
    assert!(
        stated(&notes(&format!("{SIGMA}/distance")))
            .unwrap()
            .starts_with("Frame normalisation: the population size N (08).")
    );
    // A source-frozen propagator has no frame denominator; a channel whose
    // measurement did not record one says so rather than implying a default.
    assert_eq!(stated(&notes(&format!("{RHO}/distance"))), None);
    assert!(
        stated(&notes(&format!("{NUCLEON}/distance")))
            .unwrap()
            .contains("not stated by the measurement")
    );

    // Two normalizations are two observables: their ratio is not formed, and
    // the reference rows carry the normalization behind each rate.
    let mut analysis = AnalysisConfig::default();
    assignments(&mut analysis, &["pion", "f0_500"]);
    let comparison = compare(&channels[..2], &analysis).unwrap().unwrap();
    assert_eq!(comparison.ratios.len(), 1);
    assert_eq!(comparison.ratios[0].measured, None);
    assert!(comparison.notes.iter().any(|n| n
        == "'pion' and 'f0_500' differ in time unit, estimator, scale or frame normalisation \
            and are not compared."));
    assert_eq!(
        comparison.reference[0].normalization,
        Some(FrameNormalization::ValidCount)
    );
    assert_eq!(
        comparison.reference[1].normalization,
        Some(FrameNormalization::FixedN)
    );
    assert!(comparison.notes[2].contains("the scale and the frame normalisation are compared"));
    // The same two channels under one normalization are compared again.
    let mut shared = channels[..2].to_vec();
    shared[1].normalization = Some(FrameNormalization::ValidCount);
    let comparison = compare(&shared, &analysis).unwrap().unwrap();
    assert!(comparison.ratios[0].measured.is_some());
}

#[test]
fn two_names_on_one_channel_are_not_compared_and_named_broad_states_stay_flagged() {
    // The specification id and the channel id resolve to the same channel.
    let mut analysis = four_names(&["f0_500"]);
    analysis
        .assignments
        .insert(format!("{SIGMA}/distance"), "a1".into());
    let comparison = compare(&near_channels(), &analysis).unwrap().unwrap();
    let names: Vec<&str> = comparison
        .reference
        .iter()
        .map(|r| r.name.as_str())
        .collect();
    assert_eq!(names, ["pion", "f0_500", "rho", "a1", "nucleon"]);
    assert_eq!(
        comparison.reference[1].channel,
        comparison.reference[3].channel
    );
    let shared = comparison
        .ratios
        .iter()
        .find(|r| label(r) == "a1/f0_500")
        .unwrap();
    assert_eq!((shared.measured, shared.tension_sigma), (None, None));
    assert!(comparison.notes.iter().any(|n| n
        == "'f0_500' and 'a1' are assigned the same channel; their ratio is 1 by construction \
            and is not compared."));
    // Rescaled by that anchor the other name would return the anchor's own
    // reference: a circular row, left empty like the anchor's.
    let at_scalar = &comparison.anchors[0];
    let a1 = at_scalar
        .predictions
        .iter()
        .find(|p| p.name == "a1")
        .unwrap();
    assert_eq!((a1.predicted, a1.tension_sigma), (None, None));
    assert!(at_scalar.predictions.iter().all(|p| p.name != "f0_500"));
    assert!(comparison.notes[3].contains("Of the 9 ratio rows only 4 are"));
    // A broad state stays flagged when an edited table gives it a small error.
    let mut narrow = four_names(&["nucleon"]);
    for entry in &mut narrow.reference.entries {
        entry.error = 0.;
    }
    let comparison = compare(&near_channels(), &narrow).unwrap().unwrap();
    let flagged: Vec<&String> = comparison
        .notes
        .iter()
        .filter(|n| n.starts_with("Reference '"))
        .collect();
    assert_eq!(flagged.len(), 1);
    assert!(flagged[0].starts_with("Reference 'f0_500'"));
    // Exact references: the tension is the measured deviation alone.
    let row = &comparison.ratios[1];
    let expected = (5.6 - 775.26 / 139.57039) / 0.68818602136341;
    assert!((row.tension_sigma.unwrap() - expected).abs() < 1e-10);
    assert_eq!(spread(&[-1., -3.]), Some(0.5));
}

#[test]
fn the_clone_regulariser_is_never_a_range_and_an_infinite_formula_is_a_stated_gap() {
    // A uniform kernel has no range, whatever the regulariser of the score.
    let mut hilbert = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    hilbert.clone_decision.epsilon = 1e-8;
    let scales = AlgorithmicScales::from_config(&hilbert, 3);
    assert_eq!((scales.epsilon_d, scales.epsilon_c), (None, None));
    assert_eq!(scales.epsilon_clone, 1e-8);
    // Changing the regulariser moves its own row and nothing else.
    let inputs = StandardModelInputs::default();
    let gas = viscous(0.01, 1.5, 0.8);
    let before = couplings::report(&measurement(gas.clone(), 3, None), &inputs).unwrap();
    let mut regularised = gas.clone();
    regularised.clone_decision.epsilon += 0.25;
    let after = couplings::report(&measurement(regularised, 3, None), &inputs).unwrap();
    assert_eq!(after.couplings, before.couplings);
    assert_eq!(after.inversion, before.inversion);
    let moved: Vec<&str> = before
        .scales
        .iter()
        .zip(&after.scales)
        .filter(|(a, b)| a != b)
        .map(|(a, _)| a.name.as_str())
        .collect();
    assert_eq!(moved, ["epsilon_clone"]);
    // A phase-space distance on other coordinates has no spatial range.
    let mut elsewhere = gas.clone();
    elsewhere.distance_donors.kernel = Kernel::Gaussian { width: 3. };
    elsewhere.distance_donors.distance = Distance::PhaseSpace {
        positions: "velocities".into(),
        velocities: "velocities".into(),
        position_scale: 1.,
        velocity_scale: 1.,
        lambda: 0.,
        periodic: None,
    };
    assert_eq!(
        AlgorithmicScales::from_config(&elsewhere, 3).epsilon_d,
        None
    );
    // A vanishing kernel moment: the strong proxy is 0 and the viscosity
    // target has no finite value, which the row says.
    let calibration = Calibration {
        mass: 1.,
        h_eff: 1.,
        electroweak_h_eff: 1.,
        h_s: 1.,
        epsilon_d: Some(2.),
        epsilon_c: Some(2.),
        dt: Some(0.01),
        pair_weight_n1: Some(0.5),
        viscous_kernel_second_moment: Some(0.),
        ..Calibration::default()
    };
    let report = couplings::report(&measurement(gas, 3, Some(calibration)), &inputs).unwrap();
    assert_eq!(row(&report.couplings, "g3").value, Some(0.));
    let target = row(&report.inversion, "target_viscosity");
    assert_eq!(target.value, None);
    assert!(target.definition.starts_with("target: "));
    assert!(
        target
            .definition
            .ends_with("undefined here: the formula is not finite on these scales")
    );
    assert!(serde_json::to_string(&report).is_ok());
}

#[test]
fn groups_bases_and_the_smoothing_diagnostic_are_presented_with_their_prior_diagnostics() {
    let mut report = report();
    let dominated = PriorDominance {
        width_ratio: 0.98,
        shift_sigma: 0.05,
        dominated: true,
    };
    let mut level = rate(0.4, 0.05, TimeUnit::Frames);
    level.prior_dominance = Some(PriorDominance {
        width_ratio: 0.2,
        shift_sigma: 1.5,
        dominated: false,
    });
    report.groups = vec![GroupFit {
        id: "scalars".into(),
        channels: vec![format!("{SIGMA}/distance"), format!("{RHO}/distance")],
        levels: vec![level.clone()],
        ..GroupFit::default()
    }];
    let white = FitOutcome {
        method: FitMethodKind::Gevp,
        diagnostics: FitDiagnostics {
            prior_dominance: Some(dominated),
            ..FitDiagnostics::default()
        },
        ..FitOutcome::default()
    };
    let fitted = FitOutcome {
        method: FitMethodKind::Gevp,
        mass: Some(level),
        diagnostics: FitDiagnostics {
            chi2: Some(3.5),
            dof: Some(4),
            window: Some([2, 9]),
            n_windows: 12,
            correlated: true,
            ..FitDiagnostics::default()
        },
        ..FitOutcome::default()
    };
    report.gevp = vec![GevpReport {
        id: "basis".into(),
        channels: vec![format!("{SIGMA}/distance"), format!("{RHO}/distance")],
        t0: 1,
        lags: vec![2, 3, 4],
        eigenvalues: vec![vec![Some([0.5, 0.05]), None, Some([0.125, 0.02])]],
        effective_mass: vec![vec![Some([0.7, 0.1]), None, None]],
        levels: vec![fitted, white],
        rank: 2,
        antisymmetric_norm: vec![Some(0.01), None, Some(0.04)],
        ..GevpReport::default()
    }];
    report.flow = Some(FlowDiagnostic {
        frames: 8,
        steps: vec![0, 1, 2],
        roughness: vec![Some(0.5), None, Some(0.25)],
    });
    let results = present(&report).unwrap();
    let find = |title: &str| results.iter().find(|r| r.title == title).unwrap();
    let group = find("Group scalars");
    let of = |result: &ExperimentResult, label: &str| {
        result
            .metrics
            .iter()
            .find(|m| m.label == label)
            .unwrap()
            .value
    };
    assert_eq!(of(group, "level 0"), Some(0.4));
    assert_eq!(of(group, "level 0 posterior/prior width"), Some(0.2));
    assert_eq!(of(group, "level 0 prior shift"), Some(1.5));
    assert!(group.notes.is_empty());
    let basis = find("GEVP basis");
    assert_eq!(of(basis, "t0"), Some(1.));
    assert_eq!(of(basis, "state 0 rate"), Some(0.4));
    assert_eq!(of(basis, "state 0 window t_min"), Some(2.));
    assert_eq!(of(basis, "state 0 window t_max"), Some(9.));
    assert_eq!(of(basis, "state 0 windows"), Some(12.));
    // White noise returns its prior: no rate, the diagnostic and a note.
    assert_eq!(of(basis, "state 1 rate"), None);
    assert_eq!(of(basis, "state 1 gap posterior/prior width"), Some(0.98));
    assert_eq!(
        basis.notes,
        [
            "state 1 gap returned its prior",
            "state 1: a diagonal chi2 replaced the correlated one"
        ]
    );
    let eigenvalues = &basis.plots[0];
    assert_eq!(eigenvalues.x_label, "lag (frames)");
    let central: Vec<&[[f64; 2]]> = eigenvalues
        .series
        .iter()
        .filter(|s| s.name == "state 0")
        .map(|s| s.points.as_slice())
        .collect();
    assert_eq!(central, [&[[2., 0.5]][..], &[[4., 0.125]][..]]);
    let flow = find("Graph smoothing");
    assert!(flow.model.contains("defines no length scale"));
    assert_eq!(flow.plots[0].series.len(), 2);
    assert_eq!(flow.plots[0].series[1].points, [[2., 0.25]]);
}

/// `light` 10(0), `heavy` 30(3) and `narrow` 100(1) on the scalar, vector and
/// baryon channels, anchored at `light`.
fn three_entries() -> AnalysisConfig {
    let entry = |name: &str, value: f64, error: f64| ReferenceEntry {
        name: name.into(),
        value,
        error,
        source: "test".into(),
    };
    AnalysisConfig {
        reference: ReferenceTable {
            unit: "GeV".into(),
            entries: vec![
                entry("light", 10., 0.),
                entry("heavy", 30., 3.),
                entry("narrow", 100., 1.),
            ],
        },
        assignments: [(SIGMA, "light"), (RHO, "heavy"), (NUCLEON, "narrow")]
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .into(),
        anchors: vec!["light".into()],
        ..AnalysisConfig::default()
    }
}
fn three_channels(light: [f64; 2], heavy: [f64; 2]) -> [ChannelReport; 3] {
    [
        channel(SIGMA, Some(light)),
        channel(RHO, Some(heavy)),
        channel(NUCLEON, Some([5., 0.5])),
    ]
}
fn measured(comparison: &Comparison) -> Vec<bool> {
    comparison
        .ratios
        .iter()
        .map(|r| r.measured.is_some())
        .collect()
}

#[test]
fn a_rate_at_exactly_three_sigma_is_a_denominator_and_an_anchor_and_a_range_is_flagged_by_its_error()
 {
    // 0.75/0.25 = 3 exactly. R = 2/0.75 with relative errors 0.1 and 1/3:
    // sigma_R = R sqrt(0.01 + 1/9); the reference ratio is 3(0.3).
    let analysis = three_entries();
    let comparison = compare(&three_channels([0.75, 0.25], [2., 0.2]), &analysis)
        .unwrap()
        .unwrap();
    assert_eq!(measured(&comparison), [true, true, true]);
    let row = &comparison.ratios[0];
    assert_eq!(label(row), "heavy/light");
    assert!(pair_close(
        row.measured,
        [2.6666666666666665, 0.9280272452364933]
    ));
    assert!((row.tension_sigma.unwrap() + 0.34177078284690643).abs() < 1e-12);
    let anchor = &comparison.anchors[0];
    assert!(pair_close(
        anchor.scale,
        [13.333333333333334, 4.444444444444445]
    ));
    assert!(pair_close(
        anchor.predictions[0].predicted,
        [26.666666666666664, 9.280272452364933]
    ));
    assert!(comparison.notes.iter().all(|n| !n.contains("3 sigma")));
    // The fraction rule alone: 10% is a range, 1% and 0 are not.
    let flagged: Vec<&String> = comparison
        .notes
        .iter()
        .filter(|n| n.starts_with("Reference '"))
        .collect();
    assert_eq!(flagged.len(), 1);
    assert!(flagged[0].starts_with("Reference 'heavy' (test) is not a narrow experimental state"));
    // A hair below three sigma the rate is a numerator only.
    let below = compare(&three_channels([0.75, 0.2500001], [2., 0.2]), &analysis)
        .unwrap()
        .unwrap();
    assert_eq!(measured(&below), [false, false, true]);
    assert!(below.anchors[0].scale.is_none());
    assert!(below.notes.iter().any(|n| n.contains("3 sigma")));
}

#[test]
fn a_rate_without_a_positive_finite_error_enters_no_ratio_on_either_side() {
    let analysis = three_entries();
    // A vanishing error is no infinite significance: neither a denominator
    // nor an anchor. The row still shows the number that was reported.
    let exact = compare(&three_channels([0.75, 0.], [2., 0.2]), &analysis)
        .unwrap()
        .unwrap();
    assert_eq!(measured(&exact), [false, false, true]);
    assert!(exact.ratios[..2].iter().all(|r| r.tension_sigma.is_none()));
    assert!(exact.anchors[0].scale.is_none() && exact.anchors[0].predictions.is_empty());
    assert_eq!(exact.reference[0].measured, Some([0.75, 0.]));
    assert!(
        exact
            .notes
            .iter()
            .any(|n| n.starts_with("'light': rate 0.75 +- 0 is not finite and positive"))
    );
    // narrow/heavy = 5/2 with relative errors 0.1 and 0.1.
    assert!(pair_close(
        exact.ratios[2].measured,
        [2.5, 0.35355339059327373]
    ));
    assert!(exact.notes[3].contains("Of the 1 ratio rows only 1 are"));
    // The same gate on the numerator side.
    let numerator = compare(&three_channels([0.75, 0.25], [2., 0.]), &analysis)
        .unwrap()
        .unwrap();
    assert_eq!(measured(&numerator), [false, true, false]);
    assert_eq!(numerator.anchors[0].predictions[0].predicted, None);
    assert!(numerator.anchors[0].predictions[1].predicted.is_some());
    // A NaN error is a gap in the reference row as well and the comparison
    // stays a valid report value.
    let undefined = compare(&three_channels([0.75, 0.25], [2., f64::NAN]), &analysis)
        .unwrap()
        .unwrap();
    assert_eq!(measured(&undefined), [false, true, false]);
    assert_eq!(undefined.reference[1].measured, None);
    assert_eq!(undefined.reference[0].measured, Some([0.75, 0.25]));
    assert!(serde_json::to_string(&undefined).is_ok());
}

#[test]
fn ratio_and_prediction_errors_are_positive_and_carry_the_reference_error_of_the_anchor() {
    // A negative ratio keeps a positive error.
    let [r, error] = ratio([-2., 0.2], [1., 0.1], 0.);
    assert!(close(r, -2.) && close(error, 0.282842712474619));
    // P = 10 * 2 = 20; sigma_P^2 = (10 * 0.2828)^2 + (2 * 1)^2 = 8 + 4.
    let [p, error] = prediction([2., 0.2], [1., 0.1], [10., 1.], 0.);
    assert!(close(p, 20.) && close(error, 3.4641016151377544));
    // Fully correlated equal relative errors leave the anchor's error alone.
    let [p, error] = prediction([2., 0.2], [1., 0.1], [10., 1.], 1.);
    assert_eq!((p, error), (20., 2.));
    // The anchor's reference error enters the prediction, not the scale's
    // rate error twice: scale = 10(1)/1(0.1) has relative error sqrt(0.02).
    let scale = ratio([10., 1.], [1., 0.1], 0.);
    assert!(close(scale[0], 10.) && close(scale[1], std::f64::consts::SQRT_2));
}

#[test]
fn an_assignment_key_selects_a_channel_id_first_and_then_the_first_rated_channel_of_a_specification()
 {
    let mut rated = channel(SIGMA, Some([0.35, 0.06]));
    rated.id = format!("{SIGMA}/clone");
    let channels = [channel(SIGMA, None), rated, channel(RHO, None)];
    assert_eq!(
        assigned(&channels, SIGMA).unwrap().id,
        format!("{SIGMA}/clone")
    );
    let exact = assigned(&channels, &format!("{SIGMA}/distance")).unwrap();
    assert_eq!(exact.id, format!("{SIGMA}/distance"));
    assert!(exact.mass.is_none());
    // Without any rate the first channel of the specification is kept.
    assert_eq!(
        assigned(&channels, RHO).unwrap().id,
        format!("{RHO}/distance")
    );
    assert!(assigned(&channels, "meson/unknown").is_none());
    assert!(assigned(&[], SIGMA).is_none());
    // The comparison reads the rated copy: same ratios as the plain table.
    let mut copies = near_channels();
    copies[1].mass = None;
    let mut clone = channel(SIGMA, Some([0.35, 0.06]));
    clone.id = format!("{SIGMA}/clone");
    copies.push(clone);
    let analysis = four_names(&["nucleon"]);
    let comparison = compare(&copies, &analysis).unwrap().unwrap();
    assert_eq!(comparison.reference[1].channel, format!("{SIGMA}/clone"));
    assert_eq!(
        comparison.ratios,
        compare(&near_channels(), &analysis)
            .unwrap()
            .unwrap()
            .ratios
    );
}

#[test]
fn a_repeated_anchor_gives_one_table_and_does_not_weigh_twice_in_the_spread() {
    let comparison = compare(
        &near_channels(),
        &four_names(&["nucleon", "pion", "nucleon"]),
    )
    .unwrap()
    .unwrap();
    let names: Vec<&str> = comparison
        .anchors
        .iter()
        .map(|a| a.anchor.as_str())
        .collect();
    assert_eq!(names, ["nucleon", "pion"]);
    // Two anchors with lattice scales s1, s2: spread = |s1 - s2|/(s1 + s2).
    let (s1, s2) = (1395.7039f64, 1340.38869714286f64);
    let expected = (s1 - s2) / (s1 + s2);
    assert!((expected - 0.0202168606847974).abs() < 1e-12);
    let spreads: Vec<Option<f64>> = comparison.anchor_spread.iter().map(|s| s.spread).collect();
    // The pion and the nucleon are each predicted by one anchor only.
    assert_eq!((spreads[0], spreads[3]), (None, None));
    for spread in [spreads[1], spreads[2]] {
        assert!((spread.unwrap() - expected).abs() < 1e-12);
    }
}

fn with_friction(mut gas: GasConfig, gamma: f64) -> GasConfig {
    if let KineticKind::Baoab { friction, .. } = &mut gas.kinetic.integrator {
        *friction = gamma;
    }
    gas
}

#[test]
fn every_scale_reaches_its_own_row_when_the_action_scales_and_the_calibrated_ranges_differ() {
    // Every ingredient differs from its neighbours and from 1: h_eff = 0.4,
    // colour 0.5, h_S = 0.3, m = 2, gamma = 1.3, tau = 0.02 against the
    // configured 0.01, ranges 1 and 4 against the kernel's 2, nu = 1.5,
    // rho = 0.8, N1 = 0.5, <K^2> = 0.25, d = 3 with C2(2)/C2(3) = 0.5625.
    let gas = with_friction(viscous(0.01, 1.5, 0.8), 1.3);
    let calibration = Calibration {
        mass: 2.,
        h_eff: 0.5,
        electroweak_h_eff: 0.4,
        h_s: 0.3,
        epsilon_d: Some(1.),
        epsilon_c: Some(4.),
        dt: Some(0.02),
        pair_weight_n1: Some(0.5),
        viscous_kernel_second_moment: Some(0.25),
        ..Calibration::default()
    };
    let report =
        couplings::report(&measurement(gas.clone(), 3, Some(calibration)), &thomson()).unwrap();
    for (name, expected) in [
        ("action_scale", 0.4),
        ("color_action_scale", 0.5),
        ("clone_action_scale", 0.3),
        ("mass", 2.),
        ("time_step", 0.02),
        ("friction", 1.3),
        ("temperature", 0.3846153846153846),
        ("epsilon_d", 1.),
        ("epsilon_c", 4.),
        ("energy_clone", 0.1),
        ("energy_viscous", 0.5),
        ("energy_friction", 0.52),
        ("separation", 5.),
        ("kernel_action_scale", 800.),
        ("pair_statistic_n1", 0.5),
        ("kernel_second_moment", 0.25),
    ] {
        assert!(close(value(&report.scales, name), expected), "{name}");
    }
    for (name, expected) in [
        // sqrt(0.4 * 0.5)/1 and sqrt(0.4)/1.
        ("g1", 0.4472135954999579),
        ("g1_upper", 0.6324555320336759),
        // sqrt(2 * 0.4 * 0.5625/16).
        ("g2_casimir", 0.16770509831248423),
        // sqrt(2 * 0.02 * (0.8/4)^2).
        ("g2_clock", 0.04),
        ("g2_clock_over_casimir", 0.0568888888888889),
        // (1.5/0.4) sqrt(2 * 0.25) and (1.5/0.4) sqrt(2).
        ("g3", 2.6516504294495533),
        ("g3_upper", 5.303300858899107),
        ("alpha_1", 0.015915494309189534),
        ("alpha_3", 0.5595290968074446),
        ("sin2_theta_proxy_upper", 0.9343065693430657),
    ] {
        assert!(close(value(&report.couplings, name), expected), "{name}");
    }
    for (name, expected) in [
        ("action_scale", 0.4),
        ("mass", 2.),
        ("pair_statistic_n1", 0.5),
        ("kernel_second_moment", 0.25),
        ("target_epsilon_c", 1.065177457794901),
        ("target_epsilon_d", 1.2948851903548098),
        ("target_viscosity", 0.6885521262319527),
        ("target_fitness_scale", 21.80995663575504),
        ("target_time_step", 2.8365075414860197),
        ("target_viscous_range", 0.2816432114711207),
    ] {
        assert!(close(value(&report.inversion, name), expected), "{name}");
    }
    for note in [
        "The measurement has three action scales: h_eff = 0.4 of the U(1) phase, 0.5 of the \
         colour phase and h_S = 0.3 of the cloning phase. The book's proxies have one; the first \
         is used for every proxy and for the inversion.",
        "The measurement used epsilon_d = 1 while the companion kernel gives 2.",
        "The measurement used epsilon_c = 4 while the companion kernel gives 2.",
    ] {
        assert!(report.notes.iter().any(|n| n == note), "{note}");
    }
    // Without a calibration the same numbers come from the measurement
    // configuration, and an unset cloning scale is the U(1) one.
    let mut bare = measurement(gas, 3, None);
    bare.config.phase.mass = 2.;
    bare.config.phase.h_eff = 0.5;
    bare.config.electroweak.h_eff = 0.4;
    bare.fingerprint = bare
        .config
        .fingerprint(&bare.gas, &bare.capabilities, &[])
        .unwrap();
    let configured = couplings::report(&bare, &thomson()).unwrap();
    for (name, expected) in [
        ("action_scale", 0.4),
        ("color_action_scale", 0.5),
        ("clone_action_scale", 0.4),
        ("mass", 2.),
        ("time_step", 0.01),
        ("kernel_action_scale", 400.),
        ("energy_friction", 0.52),
    ] {
        assert!(close(value(&configured.scales, name), expected), "{name}");
    }
    for (name, expected) in [
        ("g1_upper", 0.31622776601683794),
        ("g2_casimir", 0.33541019662496846),
        ("g2_clock", 0.05656854249492381),
    ] {
        assert!(
            close(value(&configured.couplings, name), expected),
            "{name}"
        );
    }
    assert!(close(
        value(&configured.inversion, "target_fitness_scale"),
        21.80995663575504
    ));
    bare.config.electroweak.h_s = Some(0.3);
    bare.fingerprint = bare
        .config
        .fingerprint(&bare.gas, &bare.capabilities, &[])
        .unwrap();
    let configured = couplings::report(&bare, &thomson()).unwrap();
    assert_eq!(value(&configured.scales, "clone_action_scale"), 0.3);
    assert_eq!(value(&configured.scales, "action_scale"), 0.4);
}

#[test]
fn the_comparison_tables_carry_the_numbers_of_the_report_row_by_row() {
    let results = present(&report()).unwrap();
    let comparison = results
        .iter()
        .find(|r| r.title.starts_with("Reference comparison"))
        .unwrap();
    let of = |label: &str| {
        let found: Vec<_> = comparison
            .metrics
            .iter()
            .filter(|m| m.label == label)
            .collect();
        assert_eq!(found.len(), 1, "{label}");
        (found[0].value, found[0].unit.as_str())
    };
    for (label, expected, unit) in [
        (
            "pion rate [meson/pseudoscalar/standard/distance]",
            0.10,
            "lattice",
        ),
        ("pion rate error", 0.01, "lattice"),
        ("pion reference", 139.57039, "MeV"),
        ("rho/pion measured", 5.6, "1"),
        ("rho/pion error", 0.68818602136341, "1"),
        ("rho/pion reference", 5.55461656301168, "1"),
        ("scale at nucleon", 1340.38869714286, "MeV per lattice unit"),
        (
            "scale error at nucleon",
            95.7420497959184,
            "MeV per lattice unit",
        ),
        ("rho at nucleon predicted", 750.6176704, "MeV"),
        ("rho at nucleon error", 75.8238349740413, "MeV"),
        ("rho at nucleon reference", 775.26, "MeV"),
    ] {
        let (value, found) = of(label);
        assert!(close(value.unwrap(), expected), "{label}");
        assert_eq!(found, unit, "{label}");
    }
    for (label, expected) in [
        ("rho/pion tension", 0.0659462782784833),
        ("rho at nucleon tension", -0.324993008330335),
    ] {
        let (value, unit) = of(label);
        assert!((value.unwrap() - expected).abs() < 1e-9, "{label}");
        assert_eq!(unit, "sigma");
    }
    assert_eq!(of("a1 at nucleon predicted").0, None);
    assert_eq!(of("a1 at nucleon reference").0, Some(1230.));
    assert_eq!(of("rho anchor spread").0, None);
    // Row 2 (a1/pion) and the other rows without a rate are gaps.
    let ratios = &comparison.plots[0];
    let runs = |name: &str| -> Vec<Vec<[f64; 2]>> {
        ratios
            .series
            .iter()
            .filter(|s| s.name == name)
            .map(|s| s.points.clone())
            .collect()
    };
    let expected = [
        vec![[0., 3.5], [1., 5.6]],
        vec![[3., 7.]],
        vec![[5., 1.6]],
        vec![[7., 2.]],
        vec![[10., 1.25]],
    ];
    let central = runs("measured");
    assert_eq!(central.len(), expected.len());
    for (run, target) in central.iter().zip(&expected) {
        assert_eq!(run.len(), target.len());
        for (point, at) in run.iter().zip(target) {
            assert_eq!(point[0], at[0]);
            assert!(close(point[1], at[1]), "{point:?}");
        }
    }
    let upper = runs("measured + error");
    let lower = runs("measured - error");
    assert!(close(upper[0][1][1], 5.6 + 0.68818602136341));
    assert!(close(lower[0][1][1], 5.6 - 0.68818602136341));
    let reference = runs("reference");
    assert_eq!(reference.len(), 1);
    assert_eq!(reference[0].len(), 15);
    assert!(close(reference[0][1][1], 5.55461656301168));
    let rescaled = &comparison.plots[1];
    assert_eq!(
        rescaled.title,
        "Rates rescaled by the anchor nucleon (an input)"
    );
    assert_eq!(rescaled.y_label, "MeV");
    let central: Vec<&[[f64; 2]]> = rescaled
        .series
        .iter()
        .filter(|s| s.name == "rescaled rate")
        .map(|s| s.points.as_slice())
        .collect();
    assert_eq!(central.len(), 1);
    assert_eq!(central[0].len(), 3);
    assert!(close(central[0][2][1], 750.6176704));
    let references = rescaled
        .series
        .iter()
        .find(|s| s.name == "reference")
        .unwrap();
    assert_eq!(
        references.points,
        [
            [0., 139.57039],
            [1., 500.],
            [2., 775.26],
            [3., 1230.],
            [4., 1710.]
        ]
    );
}

#[test]
fn every_fit_shows_its_own_rate_errors_window_table_and_reasons_and_each_estimator_its_quantity() {
    let mut report = report();
    let estimate = MassEstimate {
        error: 0.05,
        statistical: 0.04,
        systematic: 0.03,
        ..rate(0.35, 0.05, TimeUnit::Frames)
    };
    let window = |t_min: usize, value: f64, error: f64, weight: f64| WindowFit {
        t_min,
        t_max: 9,
        value,
        error,
        chi2: 3.5,
        dof: 4,
        weight,
        // chi^2 + 2k + 2 N_cut for a window ending at the last lag: every lag
        // below `t_min` is a cut point.
        aic: 7.5 + 2. * (t_min - 1) as f64,
        nexp: 1,
        svd_cut: None,
    };
    let scan = FitOutcome {
        method: FitMethodKind::WindowScan,
        mass: Some(estimate.clone()),
        excited: vec![rate(0.9, 0.2, TimeUnit::Frames)],
        diagnostics: FitDiagnostics {
            chi2: Some(3.5),
            dof: Some(4),
            q: Some(0.48),
            window: Some([2, 9]),
            n_windows: 2,
            correlated: true,
            svd_cut: 1e-3,
            covariance_rank: Some(5),
            ..FitDiagnostics::default()
        },
        windows: vec![window(2, 0.375, 0.125, 0.75), window(3, 0.5, 0.25, 0.25)],
        ..FitOutcome::default()
    };
    let silent = FitOutcome {
        method: FitMethodKind::MultiExponential,
        diagnostics: FitDiagnostics {
            correlated: true,
            prior_dominance: Some(PriorDominance {
                width_ratio: 0.98,
                shift_sigma: 0.05,
                dominated: true,
            }),
            model_rejected: Some("sign change".into()),
            no_signal: Some("signal below noise".into()),
            ..FitDiagnostics::default()
        },
        ..FitOutcome::default()
    };
    let stability = FitOutcome {
        method: FitMethodKind::Stability,
        diagnostics: FitDiagnostics {
            n_windows: 1,
            correlated: true,
            ..FitDiagnostics::default()
        },
        windows: vec![window(2, 0.375, 0.125, 1.)],
        notes: vec!["scan note".into()],
        ..FitOutcome::default()
    };
    {
        let scalar = &mut report.channels[1];
        scalar.mass = Some(estimate);
        scalar.fits = vec![scan, silent, stability];
        scalar.coverage = Coverage {
            frames: 128,
            valid: 100,
            masked_self: 3,
            masked_color: 4,
            ..Coverage::default()
        };
        scalar.correlator.as_mut().unwrap().connected = false;
        let vector = &mut report.channels[2];
        vector.estimator = Some(EstimatorKind::EuclideanTime);
        vector.mass.as_mut().unwrap().time_unit = TimeUnit::Coordinate;
        let mut slabs = correlator(vec![Some(1.), Some(0.5)], vec![Some(0.1), Some(0.05)]);
        slabs.time_unit = TimeUnit::Coordinate;
        slabs.time_step = 0.25;
        slabs.samples_meta.tau_int = Some(2.5);
        slabs.samples_meta.replicas = 3;
        slabs.connected_bias = Some(0.001);
        vector.correlator = Some(slabs);
        let baryon = &mut report.channels[3];
        baryon.estimator = Some(EstimatorKind::SourceFrozen);
        baryon.mass.as_mut().unwrap().time_unit = TimeUnit::StepDt;
    }
    report.groups = vec![GroupFit {
        id: "g".into(),
        availability: Availability::unavailable("no common window"),
        ..GroupFit::default()
    }];
    report.gevp = vec![GevpReport {
        id: "b".into(),
        availability: Availability::unavailable("singular C(t0)"),
        ..GevpReport::default()
    }];
    let results = present(&report).unwrap();
    assert!(
        results
            .iter()
            .all(|r| r.title != "Group g" && r.title != "GEVP b")
    );
    for note in [
        "group g: unavailable: no common window",
        "GEVP b: unavailable: singular C(t0)",
    ] {
        assert!(results[0].notes.iter().any(|n| n == note), "{note}");
    }
    let overview: Vec<(&str, Option<f64>)> = results[0]
        .metrics
        .iter()
        .map(|m| (m.label.as_str(), m.value))
        .collect();
    assert_eq!(overview, [("replicas", Some(1.)), ("frames", Some(128.))]);
    let of = |result: &ExperimentResult, label: &str| {
        let found: Vec<_> = result.metrics.iter().filter(|m| m.label == label).collect();
        assert_eq!(found.len(), 1, "{label}");
        found[0].value
    };
    let scalar = &results[2];
    for (label, expected) in [
        ("rate", 0.35),
        ("rate error", 0.05),
        ("rate statistical", 0.04),
        ("rate systematic", 0.03),
        ("valid elements", 100.),
        ("masked elements", 7.),
        ("window scan rate", 0.35),
        ("window scan rate error", 0.05),
        ("window scan rate statistical", 0.04),
        ("window scan rate systematic", 0.03),
        ("window scan excited rate 1", 0.9),
        ("window scan excited rate 1 error", 0.2),
        ("window scan chi2", 3.5),
        ("window scan dof", 4.),
        ("window scan Q", 0.48),
        ("window scan window t_min", 2.),
        ("window scan window t_max", 9.),
        ("window scan windows", 2.),
        ("window scan covariance rank", 5.),
        ("window scan svd cut", 1e-3),
        ("multi-exponential gap posterior/prior width", 0.98),
        ("multi-exponential gap prior shift", 0.05),
        ("stability scan windows", 1.),
    ] {
        assert_eq!(of(scalar, label), Some(expected), "{label}");
    }
    for label in [
        "multi-exponential rate",
        "multi-exponential chi2",
        "multi-exponential window t_min",
        "stability scan covariance rank",
    ] {
        assert_eq!(of(scalar, label), None, "{label}");
    }
    // A stability scan is never a rate, and an unsubtracted correlator has no
    // connected bias.
    assert!(
        scalar
            .metrics
            .iter()
            .all(|m| !m.label.starts_with("stability scan rate") && m.label != "connected bias")
    );
    for note in [
        "multi-exponential gap returned its prior",
        "multi-exponential: no rate: sign change",
        "multi-exponential: no rate: signal below noise",
        "stability scan: scan note",
    ] {
        assert!(scalar.notes.iter().any(|n| n == note), "{note}");
    }
    assert!(scalar.notes.iter().all(|n| !n.contains("diagonal chi2")));
    let titles: Vec<&str> = scalar.plots.iter().map(|p| p.title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "Correlator",
            "Effective rate",
            "window scan: rate per window",
            "window scan: window weights",
            "stability scan: rate per window",
            "stability scan: window weights",
        ]
    );
    let shape = |plot: usize| -> Vec<(&str, &[[f64; 2]])> {
        scalar.plots[plot]
            .series
            .iter()
            .map(|s| (s.name.as_str(), s.points.as_slice()))
            .collect()
    };
    assert_eq!(
        shape(2),
        [
            ("rate", &[[0., 0.375], [1., 0.5]][..]),
            ("rate + error", &[[0., 0.5], [1., 0.75]][..]),
            ("rate - error", &[[0., 0.25], [1., 0.25]][..]),
        ]
    );
    assert_eq!(shape(3), [("weight", &[[0., 0.75], [1., 0.25]][..])]);
    // A Euclidean rate is per length on a separation axis of lag * bin width.
    let vector = &results[3];
    assert_eq!(vector.model, RATE_QUANTITY_EUCLIDEAN);
    assert_eq!(vector.metrics[0].unit, "1/length");
    assert_eq!(vector.plots[0].x_label, "separation (length)");
    assert_eq!(vector.plots[0].series[0].points, [[0., 1.], [0.25, 0.5]]);
    assert_eq!(of(vector, "tau_int"), Some(2.5));
    assert_eq!(of(vector, "replicas"), Some(3.));
    assert_eq!(of(vector, "connected bias"), Some(0.001));
    let baryon = &results[4];
    assert_eq!(baryon.model, RATE_QUANTITY_SOURCE_FROZEN);
    assert_eq!(baryon.metrics[0].unit, "1/time");
    assert_ne!(RATE_QUANTITY_SOURCE_FROZEN, RATE_QUANTITY);
    assert_ne!(RATE_QUANTITY_EUCLIDEAN, RATE_QUANTITY);
}

#[test]
fn independent_ratios_are_counted_per_group_and_a_range_starts_above_two_percent() {
    // Two estimators split the four names into two pairs that are never
    // divided by each other: 2 measured rows, both independent.
    let mut channels = near_channels();
    for channel in &mut channels[2..] {
        channel.estimator = Some(EstimatorKind::SourceFrozen);
    }
    let comparison = compare(&channels, &four_names(&["nucleon"]))
        .unwrap()
        .unwrap();
    assert_eq!(
        measured(&comparison),
        [true, false, false, false, false, true]
    );
    assert!(comparison.notes[3].contains("Of the 2 ratio rows only 2 are"));
    let rescaled: Vec<(&str, bool)> = comparison.anchors[0]
        .predictions
        .iter()
        .map(|p| (p.name.as_str(), p.predicted.is_some()))
        .collect();
    assert_eq!(
        rescaled,
        [("pion", false), ("f0_500", false), ("rho", true)]
    );
    // 3% is a range; exactly 2% and 1% are not.
    let flagged = |error: f64| -> Vec<String> {
        let mut analysis = three_entries();
        analysis.reference.entries[2].error = error;
        let comparison = compare(&three_channels([0.75, 0.25], [2., 0.2]), &analysis)
            .unwrap()
            .unwrap();
        comparison
            .notes
            .iter()
            .filter(|n| n.starts_with("Reference '"))
            .map(|n| n.split('\'').nth(1).unwrap().to_string())
            .collect()
    };
    assert_eq!(flagged(3.), ["heavy", "narrow"]);
    assert_eq!(flagged(2.), ["heavy"]);
    assert_eq!(flagged(1.), ["heavy"]);
}

#[test]
fn an_undefined_error_stays_undefined_through_a_ratio_a_prediction_and_a_tension() {
    let [r, error] = ratio([2., f64::NAN], [1., 0.1], 0.);
    assert_eq!(r, 2.);
    assert!(error.is_nan());
    let [r, error] = ratio([2., 0.2], [1., f64::NAN], 0.5);
    assert_eq!(r, 2.);
    assert!(error.is_nan());
    let [p, error] = prediction([2., f64::NAN], [1., 0.1], [10., 1.], 0.);
    assert_eq!(p, 20.);
    assert!(error.is_nan());
    assert_eq!(tension([2., f64::NAN], [3., 0.3]), None);
    assert_eq!(tension([f64::NAN, 0.2], [3., 0.3]), None);
    assert_eq!(spread(&[1., f64::NAN]), None);
    assert_eq!(spread(&[0., 0.]), None);
    assert_eq!(spread(&[]), None);
}

#[test]
fn a_vanishing_friction_or_viscous_coefficient_is_a_gap_and_never_an_infinite_or_zero_scale() {
    let frictionless = with_friction(viscous(0.01, 1.5, 0.8), 0.);
    let scales = AlgorithmicScales::from_config(&frictionless, 3);
    assert_eq!((scales.friction, scales.temperature), (Some(0.), None));
    let mut inviscid = viscous(0.01, 1.5, 0.8);
    inviscid.qft.viscosity.as_mut().unwrap().coefficient = 0.;
    assert_eq!(AlgorithmicScales::from_config(&inviscid, 3).viscosity, None);
    let report = couplings::report(
        &measurement(inviscid, 3, None),
        &StandardModelInputs::default(),
    )
    .unwrap();
    assert_eq!(row(&report.scales, "viscosity").value, None);
    assert_eq!(value(&report.scales, "viscous_range"), 0.8);
    for name in ["g3", "g3_upper", "alpha_3", "alpha_3_upper"] {
        assert_eq!(row(&report.couplings, name).value, None, "{name}");
    }
    assert!(
        row(&report.couplings, "g3_upper")
            .definition
            .contains("undefined here: no dense viscosity")
    );
    assert!(
        report
            .notes
            .iter()
            .all(|n| !n.contains("dense viscous force is"))
    );
}

#[test]
fn every_curve_carries_its_own_values_and_an_overflowing_band_edge_is_an_error() {
    let mut report = report();
    report.channels[0].correlator = Some(CorrelatorEstimate {
        time_unit: TimeUnit::Frames,
        time_step: 1.,
        ..correlator(vec![Some(1.), Some(0.25)], vec![Some(0.5), Some(0.125)])
    });
    report.gevp = vec![GevpReport {
        id: "basis".into(),
        channels: vec![format!("{SIGMA}/distance"), format!("{RHO}/distance")],
        t0: 1,
        lags: vec![2, 3, 4],
        eigenvalues: vec![
            vec![Some([0.5, 0.0625]), None, Some([0.125, 0.03125])],
            vec![Some([0.25, 0.125]), None, None],
        ],
        effective_mass: vec![vec![Some([0.75, 0.25]), None, None], vec![None; 3]],
        rank: 2,
        antisymmetric_norm: vec![Some(0.01), None, Some(0.04)],
        ..GevpReport::default()
    }];
    let results = present(&report).unwrap();
    let shape = |result: &ExperimentResult, plot: usize| -> Vec<(String, Vec<[f64; 2]>)> {
        result.plots[plot]
            .series
            .iter()
            .map(|s| (s.name.clone(), s.points.clone()))
            .collect()
    };
    let curve = |name: &str, points: &[[f64; 2]]| (name.to_string(), points.to_vec());
    // A lag in frames is its own abscissa.
    let pion = &results[1];
    assert_eq!(pion.plots[0].x_label, "lag (frames)");
    assert_eq!(
        shape(pion, 0),
        [
            curve("C", &[[0., 1.], [1., 0.25]]),
            curve("C + error", &[[0., 1.5], [1., 0.375]]),
            curve("C - error", &[[0., 0.5], [1., 0.125]]),
        ]
    );
    // The effective rate is drawn from its own values and errors.
    let scalar = &results[2];
    assert_eq!(scalar.plots[1].title, "Effective rate");
    assert_eq!(scalar.plots[1].x_label, "lag (time)");
    let effective = shape(scalar, 1);
    let expected = [("rate", 0.7), ("rate + error", 0.8), ("rate - error", 0.6)];
    assert_eq!(effective.len(), 3);
    for ((name, points), (target, y)) in effective.iter().zip(expected) {
        assert_eq!(name, target);
        assert_eq!(points.len(), 1);
        assert!(points[0][0] == 0. && close(points[0][1], y), "{target}");
    }
    let basis = results.iter().find(|r| r.title == "GEVP basis").unwrap();
    let titles: Vec<&str> = basis.plots.iter().map(|p| p.title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "Generalized eigenvalues",
            "Effective rates",
            "Discarded antisymmetric part"
        ]
    );
    assert_eq!(
        shape(basis, 0),
        [
            curve("state 0", &[[2., 0.5]]),
            curve("state 0", &[[4., 0.125]]),
            curve("state 0 + error", &[[2., 0.5625]]),
            curve("state 0 + error", &[[4., 0.15625]]),
            curve("state 0 - error", &[[2., 0.4375]]),
            curve("state 0 - error", &[[4., 0.09375]]),
            curve("state 1", &[[2., 0.25]]),
            curve("state 1 + error", &[[2., 0.375]]),
            curve("state 1 - error", &[[2., 0.125]]),
        ]
    );
    assert_eq!(
        shape(basis, 1),
        [
            curve("state 0", &[[2., 0.75]]),
            curve("state 0 + error", &[[2., 1.]]),
            curve("state 0 - error", &[[2., 0.5]]),
        ]
    );
    assert_eq!(
        shape(basis, 2),
        [
            curve("antisymmetric norm", &[[2., 0.01]]),
            curve("antisymmetric norm", &[[4., 0.04]]),
        ]
    );
    // An input of the inversion shows its error; a scale and a proxy have none,
    // and only a row without a value is explained outside the inversion.
    let find = |title: &str| results.iter().find(|r| r.title == title).unwrap();
    let inversion = find("Calibration inversion");
    let error = inversion
        .metrics
        .iter()
        .find(|m| m.label == "alpha_em error")
        .unwrap();
    assert!((error.value.unwrap() - 5.49737218246254e-07).abs() < 1e-18);
    assert_eq!(error.unit, "1");
    let scales = find("Algorithmic scales");
    assert!(scales.metrics.iter().all(|m| !m.label.ends_with(" error")));
    let range = scales
        .metrics
        .iter()
        .find(|m| m.label == "epsilon_d")
        .unwrap();
    assert_eq!((range.value, range.unit.as_str()), (Some(2.), "length"));
    let explained: Vec<&str> = scales
        .notes
        .iter()
        .take(4)
        .map(|n| n.split(": ").next().unwrap())
        .collect();
    assert_eq!(
        explained,
        [
            "phase_length",
            "fitness_force_scale",
            "pair_statistic_n1",
            "kernel_second_moment"
        ]
    );
    assert_eq!(
        scales.notes[4..],
        report.couplings.as_ref().unwrap().notes[..]
    );
    // Two finite numbers whose sum is not: the band edge is rejected after the
    // report itself passed its validation.
    let mut overflow = report.clone();
    let estimate = overflow.channels[0].correlator.as_mut().unwrap();
    estimate.value[0] = Some(1.5e308);
    estimate.error[0] = Some(1.5e308);
    assert!(overflow.validate().is_ok());
    assert!(matches!(
        present(&overflow),
        Err(GasError::Configuration(message)) if message == "nonfinite plotted result"
    ));
}
