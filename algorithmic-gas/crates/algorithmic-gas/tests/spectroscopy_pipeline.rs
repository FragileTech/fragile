//! End to end: `measure_archive` then `analyze` on real recorded runs of the
//! Einstein-Hilbert, Viscous Euclidean and Euclidean gases with the standard
//! channel set. What the variant records decides which channels carry a
//! correlator, and the mutuality of its companion law decides which estimator
//! an exchange-odd operator gets; nothing here fails a run.
use algorithmic_gas::{
    AlgorithmicGas, ExecutionContext, GasBuilder, GasConfig, InputBatch, ObservationBatch,
    Population, Precision, Provenance, Real, RecordingConfig, RewardBatch, RunArchive, TensorBatch,
    domain::{GradientProvider, OperatorFuture, RewardSource},
    physics::spectroscopy::{
        AnalysisConfig, Capabilities, ChannelSpec, ElementKind, ExchangeParity, MeasurementConfig,
        Record, Requirements, SpectroscopyReport, analyze,
        config::{ChannelGroup, GevpBasis},
        contract::EXCHANGE_ODD_REASON,
        estimators::DELETION_GEOMETRY_NOTE,
        measure_archive,
        report::{
            EstimatorKind, RATE_QUANTITY, RATE_QUANTITY_EUCLIDEAN, RATE_QUANTITY_SOURCE_FROZEN,
        },
    },
    tessellation::{GeometryReward, GeometrySchedule, ZeroPotential},
    variants::viscous_euclidean::reference_viscosity,
};
use futures_lite::future::block_on;
use std::collections::BTreeSet;

/// Deterministic cloud in `[-spread/2, spread/2]^d` with velocities in
/// `[-speed/2, speed/2]^d`.
fn population<T: Real>(n: usize, d: usize, spread: f64, speed: f64) -> Population<T> {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let unit = |k: u64| (mix(k) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let x = (0..n * d)
        .map(|k| T::from_f64(spread * unit(k as u64)))
        .collect();
    let v = (0..n * d)
        .map(|k| T::from_f64(speed * unit(1_000_003 + k as u64)))
        .collect();
    let mut observations = ObservationBatch::positions(TensorBatch::vectors(n, d, x).unwrap());
    observations
        .fields
        .insert("velocities".into(), TensorBatch::vectors(n, d, v).unwrap());
    Population::new(observations).unwrap()
}
/// `U = |x|² / 2` as reward and potential; the Euclidean Gas minimizes it.
struct Quadratic;
impl RewardSource<f64> for Quadratic {
    fn id(&self) -> String {
        "test-quadratic/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<f64>> {
        Box::pin(async move {
            let x = p.observations.field("positions")?;
            Ok(RewardBatch::new(
                x.values()
                    .chunks_exact(x.width())
                    .map(|r| 0.5 * r.iter().map(|a| a * a).sum::<f64>())
                    .collect(),
                Provenance {
                    population_version: p.version,
                    stage: stage.into(),
                    ..Default::default()
                },
            ))
        })
    }
}
impl GradientProvider<f64> for Quadratic {
    fn id(&self) -> String {
        "test-quadratic-gradient/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            let x = p.observations.field("positions")?;
            let (d, eligible) = (x.width(), p.eligible(false));
            let mut gradient = x.values().to_vec();
            for (row, alive) in gradient.chunks_exact_mut(d).zip(eligible) {
                if !alive {
                    row.fill(0.);
                }
            }
            TensorBatch::vectors(p.len(), d, gradient)
        })
    }
}
fn recording(steps: usize, graph: bool) -> RecordingConfig {
    RecordingConfig {
        max_steps: steps,
        graph,
        ..RecordingConfig::default()
    }
}
fn record(gas: &mut AlgorithmicGas<f64>, steps: usize, graph: bool) -> RunArchive<f64> {
    gas.start_recording(recording(steps, graph)).unwrap();
    for _ in 0..steps {
        block_on(gas.step()).unwrap();
    }
    gas.stop_recording().unwrap()
}
/// Einstein-Hilbert gas in three dimensions. The preset inherits `F32` from
/// `GasConfig::default`, and spectroscopy measures `f64` runs only.
fn einstein_hilbert(walkers: usize, seed: u64) -> AlgorithmicGas<f64> {
    let mut config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    config.precision = Precision::F64;
    config.seed = seed;
    config.clone_decision.every = 2;
    config.geometry.as_mut().unwrap().schedule = GeometrySchedule::EveryStage;
    block_on(
        GasBuilder::new(
            population::<f64>(walkers, 3, 1., 1.),
            GeometryReward::default(),
        )
        .gradient(ZeroPotential::new("velocities"))
        .config(config)
        .build(),
    )
    .unwrap()
}
/// The Euclidean Gas, with the reference dense viscosity when `viscous`.
fn euclidean(walkers: usize, seed: u64, viscous: bool) -> AlgorithmicGas<f64> {
    let mut config = if viscous {
        GasConfig::viscous_euclidean(3, 0.05, reference_viscosity()).unwrap()
    } else {
        GasConfig::euclidean(3, 0.05).unwrap()
    };
    config.seed = seed;
    block_on(
        GasBuilder::new(population::<f64>(walkers, 3, 2., 1.), Quadratic)
            .gradient(Quadratic)
            .config(config)
            .build(),
    )
    .unwrap()
}
fn measurement() -> MeasurementConfig {
    MeasurementConfig {
        warmup: 4,
        max_lag: 8,
        ..MeasurementConfig::default()
    }
}
/// The capability model of a recorded archive, as `measure_archive` builds it.
fn capabilities(archive: &RunArchive<f64>) -> Capabilities {
    Capabilities::of(&archive.gas_config, &archive.config, 3).refine(&measurement())
}
fn measure(archive: &RunArchive<f64>) -> SpectroscopyReport {
    let measured = measure_archive(&measurement(), archive).unwrap();
    analyze(&[measured], &AnalysisConfig::default()).unwrap()
}
/// Channels of the standard set whose operator reads the colour field. Every
/// one of them is unavailable on a gas that records no viscous force, and the
/// remaining seven are not.
const COLOUR_CHANNELS: [&str; 14] = [
    "meson/scalar/standard/distance",
    "meson/scalar/standard/cloning",
    "meson/pseudoscalar/standard/distance",
    "meson/pseudoscalar/standard/cloning",
    "vector/vector/full/raw/distance",
    "vector/vector/full/raw/cloning",
    "vector/axial/full/raw/distance",
    "vector/axial/full/raw/cloning",
    "baryon/complex/triplet",
    "baryon/abs2/triplet",
    "glueball/re_plaquette/triplet",
    "glueball/force_norm/site",
    "tensor/components/distance",
    "tensor/components/cloning",
];
const FITNESS_CHANNELS: [&str; 7] = [
    "u1/phase/q1/distance",
    "u1/dressed/q1/distance",
    "su2/phase/cloning",
    "su2/doublet/cloning",
    "fitness_phase/site",
    "clone_indicator/site",
    "chirality/chi/site",
];
/// `select_estimator` read off the capability model: an exchange-odd pair
/// operator has no frame mean on a mutual pairing and is reported through its
/// source-frozen propagator; an exchange-odd triplet operator has a frame mean
/// of zero expectation whatever the pairing, so it is frozen as well.
fn predicted(exchange: ExchangeParity, kind: ElementKind, caps: &Capabilities) -> EstimatorKind {
    let mutual = match kind {
        ElementKind::DistancePair => caps.mutual_distance,
        ElementKind::CloningPair => caps.mutual_cloning,
        ElementKind::Triplet => true,
        _ => false,
    };
    if exchange == ExchangeParity::Odd && mutual {
        EstimatorKind::SourceFrozen
    } else {
        EstimatorKind::FrameMean
    }
}

#[test]
fn an_einstein_hilbert_run_freezes_every_exchange_odd_channel_of_its_mutual_pairings() {
    let archive = record(&mut einstein_hilbert(24, 7), 48, true);
    let caps = capabilities(&archive);
    // Both companion laws of the preset are mutual, and neither kernel is a
    // Gaussian of a declared width.
    assert!(caps.mutual_distance && caps.mutual_cloning);
    assert_eq!(caps.distance_kernel_width, None);
    assert_eq!(caps.cloning_kernel_width, None);
    assert!(
        caps.check(&Requirements::new([Record::Color]))
            .is_available()
    );
    let report = measure(&archive);
    assert_eq!(report.capabilities, caps);
    // Every specification of the standard set is present, the pair ones on
    // both companion roles: sixteen specifications, twenty-one channels.
    let specs: BTreeSet<String> = report.channels.iter().map(|c| c.spec.id()).collect();
    assert_eq!(specs.len(), ChannelSpec::standard_set().len());
    assert_eq!(
        report.channels.len(),
        COLOUR_CHANNELS.len() + FITNESS_CHANNELS.len()
    );

    // The two channels that ask for a companion range are the only ones the
    // model declines, and they decline with the range reason, not a colour one.
    let declined: Vec<&str> = report
        .channels
        .iter()
        .filter(|c| !c.availability.is_available())
        .map(|c| c.id.as_str())
        .collect();
    assert_eq!(declined, ["u1/dressed/q1/distance", "su2/doublet/cloning"]);
    for (id, epsilon) in [
        ("u1/dressed/q1/distance", "epsilon_d"),
        ("su2/doublet/cloning", "epsilon_c"),
    ] {
        let channel = report.channel(id).unwrap();
        let reason = channel.availability.reason().unwrap();
        assert!(reason.contains(epsilon) && reason.contains("Gaussian companion kernel"));
        assert!(channel.estimator.is_none() && channel.correlator.is_none());
        assert!(channel.mass.is_none() && channel.fits.is_empty());
    }
    // Every available channel takes the estimator the capability model
    // predicts, and each frozen one says in its own notes why.
    let mut frozen = 0;
    for channel in report
        .channels
        .iter()
        .filter(|c| c.availability.is_available())
    {
        let kind = channel.kind.unwrap();
        let exchange = channel.exchange.unwrap();
        assert_eq!(
            channel.estimator,
            Some(predicted(exchange, kind, &caps)),
            "{}",
            channel.id
        );
        if channel.estimator == Some(EstimatorKind::SourceFrozen) {
            frozen += 1;
            assert_eq!(exchange, ExchangeParity::Odd);
            let reason = if kind == ElementKind::Triplet {
                "relabelling-odd triplet operator with a frame mean of zero expectation"
            } else {
                EXCHANGE_ODD_REASON
            };
            assert!(
                channel.notes.iter().any(|n| {
                    n == &format!(
                        "{reason}: the source-frozen propagator is reported instead of the \
                         frame mean"
                    )
                }),
                "{}: {:?}",
                channel.id,
                channel.notes
            );
        }
    }
    // Pseudoscalar and vector mesons, the antisymmetric tensor on both
    // pairings, and the complex baryon: seven frozen channels, not zero.
    assert_eq!(frozen, 7);
}

#[test]
fn a_viscous_euclidean_run_keeps_the_frame_mean_of_its_odd_pair_channels_and_freezes_the_triplet() {
    let archive = record(&mut euclidean(24, 7, true), 48, false);
    let caps = capabilities(&archive);
    // The Euclidean companion law is a Gaussian kernel and is not mutual, so an
    // exchange-odd pair operator keeps a frame mean here.
    assert!(!caps.mutual_distance && !caps.mutual_cloning);
    assert_eq!(caps.distance_kernel_width, Some(2.));
    assert!(caps.dense_viscosity);
    assert!(
        caps.check(&Requirements::new([Record::Color]))
            .is_available()
    );
    let report = measure(&archive);
    assert_eq!(report.capabilities, caps);
    assert!(
        report
            .channels
            .iter()
            .all(|c| c.availability.is_available())
    );
    let mut frozen = vec![];
    for channel in &report.channels {
        let (kind, exchange) = (channel.kind.unwrap(), channel.exchange.unwrap());
        assert_eq!(
            channel.estimator,
            Some(predicted(exchange, kind, &caps)),
            "{}",
            channel.id
        );
        if channel.estimator == Some(EstimatorKind::SourceFrozen) {
            frozen.push(channel.id.as_str());
        }
    }
    // Only the relabelling-odd triplet operator; every odd pair operator kept
    // its frame mean because the pairing is not mutual.
    assert_eq!(frozen, ["baryon/complex/triplet"]);
    assert_eq!(
        report
            .channel("tensor/components/distance")
            .unwrap()
            .estimator,
        Some(EstimatorKind::FrameMean)
    );
}

#[test]
fn a_gevp_basis_and_a_channel_group_of_a_real_run_reach_the_report_through_the_matrix_correlator() {
    let archive = record(&mut euclidean(24, 7, true), 48, false);
    let channels = vec![
        "meson/scalar/standard/distance".to_string(),
        "meson/scalar/standard/cloning".to_string(),
    ];
    let analysis = AnalysisConfig {
        groups: vec![ChannelGroup {
            id: "scalar_group".into(),
            channels: channels.clone(),
        }],
        // The metric is `C(t0)`: at `t0 = 1` this short run leaves it with a
        // nonpositive diagonal and the basis declines, which is the honest
        // answer; `t0 = 0` takes the variance, which is positive.
        gevp: vec![GevpBasis {
            id: "scalar_basis".into(),
            channels: channels.clone(),
            t0: 0,
            ..GevpBasis::default()
        }],
        ..AnalysisConfig::default()
    };
    let measured = measure_archive(&measurement(), &archive).unwrap();
    let report = analyze(&[measured], &analysis).unwrap();
    report.validate().unwrap();
    assert_eq!(report.groups.len(), 1);
    assert_eq!(report.groups[0].id, "scalar_group");
    assert_eq!(report.groups[0].channels, channels);
    assert_eq!(report.gevp.len(), 1);
    let gevp = &report.gevp[0];
    assert_eq!(
        (gevp.id.as_str(), &gevp.channels),
        ("scalar_basis", &channels)
    );
    assert!(gevp.availability.is_available(), "{:?}", gevp.availability);
    // The matrix correlator hands the solver a strictly increasing lag axis
    // that opens at `t0` and runs to the measured range, one eigenvalue row
    // per lag and a basis of full rank.
    assert!(gevp.lags.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(gevp.lags.first(), Some(&gevp.t0));
    assert_eq!(gevp.lags.len(), measurement().max_lag + 1 - gevp.t0);
    assert_eq!(gevp.rank, channels.len());
    // One eigenvalue and one effective-rate series per state, each over the
    // whole lag axis, and one fit per state.
    assert_eq!(gevp.eigenvalues.len(), gevp.rank);
    assert_eq!(gevp.effective_mass.len(), gevp.rank);
    assert!(
        gevp.eigenvalues
            .iter()
            .chain(&gevp.effective_mass)
            .all(|state| state.len() == gevp.lags.len())
    );
    assert_eq!(gevp.antisymmetric_norm.len(), gevp.lags.len());
    assert_eq!(gevp.levels.len(), gevp.rank);
    let text = serde_json::to_string(&report).unwrap();
    assert_eq!(
        serde_json::from_str::<SpectroscopyReport>(&text).unwrap(),
        report
    );

    // A basis that mixes a scalar with a three-component vector channel is
    // declined with its reason; the channels and the group of the same report
    // are analysed all the same.
    let mut mixed = analysis.clone();
    mixed.gevp[0].channels = vec![
        "meson/scalar/standard/distance".into(),
        "vector/vector/full/raw/distance".into(),
    ];
    let measured = measure_archive(&measurement(), &archive).unwrap();
    let declined = analyze(&[measured], &mixed).unwrap();
    declined.validate().unwrap();
    let basis = &declined.gevp[0];
    assert!(
        basis
            .availability
            .reason()
            .is_some_and(|r| r.contains("channels of equal components")),
        "{:?}",
        basis.availability
    );
    assert!(basis.levels.is_empty() && basis.rank == 0);
    assert_eq!(declined.channels.len(), report.channels.len());
    assert_eq!(declined.groups[0].levels, report.groups[0].levels);
}

#[test]
fn a_euclidean_run_leaves_the_colour_channels_unavailable_and_analyses_the_fitness_ones() {
    let archive = record(&mut euclidean(24, 7, false), 48, false);
    let caps = capabilities(&archive);
    assert!(!caps.dense_viscosity);
    let colour = caps.check(&Requirements::new([Record::Color]));
    let reason = colour.reason().unwrap();
    assert!(reason.starts_with("viscous force is identically zero"));
    let report = measure(&archive);
    for id in COLOUR_CHANNELS {
        let channel = report.channel(id).unwrap();
        // The reason is the capability model's own sentence, word for word.
        assert_eq!(channel.availability.reason(), Some(reason));
        assert!(channel.estimator.is_none() && channel.correlator.is_none());
        assert!(channel.mass.is_none() && channel.effective_mass.is_none());
    }
    for id in FITNESS_CHANNELS {
        let channel = report.channel(id).unwrap();
        assert!(channel.availability.is_available(), "{id}");
        assert_eq!(channel.estimator, Some(EstimatorKind::FrameMean), "{id}");
        assert!(channel.correlator.is_some(), "{id}");
    }
    assert_eq!(
        COLOUR_CHANNELS.len() + FITNESS_CHANNELS.len(),
        report.channels.len()
    );
    // A declined channel is a reported channel, not a failed analysis.
    report.validate().unwrap();
}

#[test]
fn every_report_round_trips_through_json_carries_no_nonfinite_number_and_states_what_a_rate_is() {
    let mut total_rates = 0;
    for (mut gas, graph) in [
        (einstein_hilbert(24, 7), true),
        (euclidean(24, 7, true), false),
        (euclidean(24, 7, false), false),
    ] {
        let report = measure(&record(&mut gas, 48, graph));
        // `validate` rejects a nonfinite number anywhere in the report, and
        // serde writes one as a JSON `null` that no `f64` field accepts back.
        report.validate().unwrap();
        let text = serde_json::to_string(&report).unwrap();
        assert!(!text.contains("NaN") && !text.contains("Infinity"));
        assert_eq!(
            serde_json::from_str::<SpectroscopyReport>(&text).unwrap(),
            report
        );
        // The honest wording travels with every report: the fitted number is a
        // decay rate, and a mass only under the transfer representation.
        assert_eq!(
            report.notes[0],
            "Fitted values are decay rates of the algorithm-time autocorrelation. They are \
             masses only under the positive self-adjoint transfer representation \
             (cor-effective-twistor-positive-transfer); the gas is not reversible, so complex \
             or oscillating modes are expected and reject the exponential model."
        );
        // No fitted number is ever called a mass, and each estimator names its
        // own quantity. The Einstein-Hilbert run does report rates, so the
        // claim is not vacuous for want of a fit.
        let mut rates = 0;
        for channel in &report.channels {
            for fit in &channel.fits {
                for mass in fit.mass.iter().chain(&fit.excited) {
                    rates += 1;
                    assert!(
                        [
                            RATE_QUANTITY,
                            RATE_QUANTITY_SOURCE_FROZEN,
                            RATE_QUANTITY_EUCLIDEAN
                        ]
                        .contains(&mass.quantity.as_str()),
                        "{}",
                        mass.quantity
                    );
                }
            }
        }
        total_rates += rates;
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("not replica standard errors"))
        );
        // What the resampling behind a channel does not cover is published
        // with the channel, not left to a reader who never sees the resample
        // table: a delete-one geometry removes origins and not observations,
        // so every channel whose lags reach beyond its effective block states
        // that its error there is a lower bound. These runs are 48 frames
        // against an automatic block, so every analysed channel says it.
        let mut stated = 0;
        for channel in &report.channels {
            if channel.correlator.is_none() {
                continue;
            }
            assert!(
                channel
                    .notes
                    .iter()
                    .any(|n| n.starts_with(DELETION_GEOMETRY_NOTE)),
                "{}: {:?}",
                channel.id,
                channel.notes
            );
            stated += 1;
        }
        assert!(stated > 0);
    }
    assert!(total_rates > 0);
}

#[test]
fn the_same_seed_gives_the_same_report_and_another_seed_does_not() {
    let first = measure(&record(&mut einstein_hilbert(24, 7), 48, true));
    let again = measure(&record(&mut einstein_hilbert(24, 7), 48, true));
    assert_eq!(first, again);
    let other = measure(&record(&mut einstein_hilbert(24, 11), 48, true));
    assert_eq!(first.capabilities, other.capabilities);
    assert_ne!(first, other);
    // The fingerprint covers the configuration, not the draw, so two seeds of
    // one variant stay comparable.
    assert_eq!(first.measurement_fingerprint, other.measurement_fingerprint);
}
