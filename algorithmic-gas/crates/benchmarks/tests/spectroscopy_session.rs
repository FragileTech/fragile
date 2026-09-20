//! The spectroscopy session measures a real run in chunks and re-analyses its
//! evidence without running the gas again.
use algorithmic_gas::{
    GasError, Precision, RecordingConfig,
    noise::{FactorValues, NoiseGeometry},
    physics::spectroscopy::{
        AnalysisConfig, MeasurementConfig, SPECTROSCOPY_VERSION, SpectroscopyConfig,
        config::EffectiveMassKind, measure_archive,
    },
};
use algorithmic_gas_benchmarks::{
    RunConfig,
    spectroscopy::{
        SpectroscopyEvidence, SpectroscopyRequest, SpectroscopySession, analyze_archive,
        analyze_evidence, capabilities, defaults,
    },
};
use futures_lite::future::block_on;
use serde_json::{Value, json};

/// A request short enough to run inside a test: a few frames past the warm-up
/// and the shortest lag range the window scan can still fit.
fn short(variant: &str, walkers: usize, dimensions: usize, steps: usize) -> SpectroscopyRequest {
    SpectroscopyRequest {
        variant: Some(variant.into()),
        run: RunConfig {
            walkers,
            dimensions,
            ..RunConfig::default()
        },
        steps,
        replicas: 2,
        seed: 7,
        chunk: 4,
        spectroscopy: SpectroscopyConfig {
            measurement: MeasurementConfig {
                warmup: 2,
                max_lag: 6,
                ..MeasurementConfig::default()
            },
            analysis: AnalysisConfig::default(),
        },
    }
    .resolve()
    .unwrap()
}
fn finished(request: SpectroscopyRequest) -> SpectroscopySession {
    block_on(async {
        let mut session = SpectroscopySession::create(request).await.unwrap();
        while !session.done() {
            session.advance(8).await.unwrap();
        }
        session
    })
}

#[test]
fn every_implemented_variant_measures_a_short_session() {
    for (variant, dimensions) in [
        ("euclidean", 2),
        ("viscous_euclidean", 3),
        ("einstein_hilbert", 3),
    ] {
        let request = short(variant, 32, dimensions, 20);
        assert_eq!(request.run.gas.precision, Precision::F64, "{variant}");
        let session = finished(request.clone());
        let snapshot: Value = session.snapshot().unwrap();
        assert_eq!(snapshot["step"], json!(20), "{variant}");
        assert_eq!(snapshot["done"], json!(true), "{variant}");
        assert_eq!(snapshot["walkers"]["dimension"], json!(dimensions));
        assert_eq!(
            snapshot["walkers"]["eligible"].as_array().unwrap().len(),
            32,
            "{variant}"
        );
        let channels = snapshot["channels"].as_array().unwrap();
        assert!(channels.len() >= 10, "{variant}");
        let curves: Vec<&Value> = channels
            .iter()
            .filter(|c| !c["correlator"].as_array().unwrap().is_empty())
            .collect();
        assert!(!curves.is_empty() && curves.len() <= 4, "{variant}");
        for curve in curves.iter() {
            assert!(curve["estimator"].is_string(), "{variant}");
            assert_eq!(
                curve["correlator"].as_array().unwrap().len(),
                curve["effective_mass"].as_array().unwrap().len(),
                "{variant}"
            );
        }
        // Every channel that can be correlated states the estimator it would
        // be read with, and only the first few of them carry a curve.
        let stated = channels
            .iter()
            .filter(|c| c["estimator"].is_string())
            .count();
        assert!(stated > curves.len(), "{variant}");
        let report = session
            .analyze(&request.spectroscopy.analysis)
            .unwrap_or_else(|e| panic!("{variant}: {e}"));
        report.validate().unwrap();
        assert_eq!(report.replicas, 2, "{variant}");
        assert!(report.frames > 0, "{variant}");
        assert!(
            report
                .channels
                .iter()
                .any(|c| c.availability.is_available()),
            "{variant}"
        );
    }
}

#[test]
fn chunked_advance_measures_what_one_archive_of_the_whole_run_measures() {
    block_on(async {
        let mut request = short("euclidean", 32, 2, 20);
        request.replicas = 1;
        let session = finished(request.clone());
        let chunked = session.evidence().measurements;
        let mut config = request.run.clone();
        config.gas.seed = request.replica_seed(0);
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig {
            max_steps: request.steps,
            max_bytes: 256 * 1024 * 1024,
            graph: config.gas.geometry.is_some(),
        })
        .unwrap();
        for _ in 0..request.steps {
            gas.step().await.unwrap();
        }
        let archive = gas.stop_recording().unwrap();
        let once = measure_archive(&request.spectroscopy.measurement, &archive).unwrap();
        assert_eq!(chunked.len(), 1);
        assert_eq!(chunked[0].segments, 1);
        assert_eq!(chunked[0], once);
        let direct = analyze_archive(&request.spectroscopy, &[archive]).unwrap();
        assert_eq!(
            json!(direct.channels),
            json!(
                session
                    .analyze(&request.spectroscopy.analysis)
                    .unwrap()
                    .channels
            )
        );
    });
}

#[test]
fn one_chunk_is_recorded_at_a_time_so_a_longer_run_needs_no_more_engine_memory() {
    // An engine memory budget that one four-step chunk fits in and one archive
    // of the whole run does not: the chunked session runs four times as long
    // under the same budget because it drops every chunk it has ingested.
    let mut request = short("euclidean", 32, 2, 40);
    request.replicas = 1;
    request.run.gas.max_memory_bytes = 2_000_000;
    let whole = block_on(async {
        let mut config = request.run.clone();
        config.gas.seed = request.replica_seed(0);
        let mut gas = config.build::<f64>().await?;
        gas.start_recording(RecordingConfig {
            max_steps: request.steps,
            max_bytes: 256 * 1024 * 1024,
            graph: config.gas.geometry.is_some(),
        })?;
        for _ in 0..request.steps {
            gas.step().await?;
        }
        Ok::<_, GasError>(())
    });
    assert!(matches!(whole, Err(GasError::Capability(_))), "{whole:?}");
    let mut longer = request.clone();
    longer.steps *= 4;
    assert_eq!(finished(request).evidence().measurements[0].frames(), 38);
    assert_eq!(finished(longer).evidence().measurements[0].frames(), 158);
}

#[test]
fn an_advance_outside_the_batch_range_is_refused_before_any_step_is_taken() {
    block_on(async {
        let request = short("euclidean", 32, 2, 20);
        let mut session = SpectroscopySession::create(request).await.unwrap();
        let before = session.snapshot().unwrap();
        for count in [0, 65, usize::MAX] {
            assert!(
                matches!(
                    session.advance(count).await,
                    Err(GasError::Configuration(_))
                ),
                "{count}"
            );
        }
        assert_eq!(session.snapshot().unwrap(), before);
        assert_eq!(before["step"], json!(0));
        session.advance(64).await.unwrap();
        assert_eq!(session.snapshot().unwrap()["step"], json!(20));
    });
}

#[test]
fn replicas_are_seeded_apart_and_the_same_request_measures_the_same_run_again() {
    let request = short("euclidean", 32, 2, 16);
    let once = finished(request.clone()).evidence();
    assert_eq!(finished(request.clone()).evidence(), once);
    assert_eq!(once.configs[0].gas.seed, request.seed);
    assert_eq!(once.configs[1].gas.seed, request.seed + 104_729);
    assert_ne!(once.measurements[0], once.measurements[1]);
    let mut reseeded = request;
    reseeded.seed += 1;
    assert_ne!(
        finished(reseeded).evidence().measurements,
        once.measurements
    );
}

#[test]
fn a_restored_checkpoint_continues_the_stream_it_interrupted() {
    block_on(async {
        let request = short("euclidean", 32, 2, 16);
        let mut uninterrupted = SpectroscopySession::create(request.clone()).await.unwrap();
        uninterrupted.advance(8).await.unwrap();
        let bytes = uninterrupted.checkpoint().unwrap();
        uninterrupted.advance(8).await.unwrap();
        let mut restored = SpectroscopySession::restore(&bytes).await.unwrap();
        assert_eq!(restored.request(), &request);
        assert!(!restored.done());
        restored.advance(8).await.unwrap();
        assert!(uninterrupted.done() && restored.done());
        assert_eq!(
            restored.evidence().measurements,
            uninterrupted.evidence().measurements
        );
        assert_eq!(
            restored.snapshot().unwrap(),
            uninterrupted.snapshot().unwrap()
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(SpectroscopySession::restore(&trailing).await.is_err());
    });
}

#[test]
fn two_analyses_of_one_session_report_their_own_configuration_without_running_again() {
    let request = short("euclidean", 32, 2, 20);
    let session = finished(request.clone());
    let before = session.snapshot().unwrap();
    let first = session.analyze(&request.spectroscopy.analysis).unwrap();
    let second = session
        .analyze(&AnalysisConfig {
            effective_mass: EffectiveMassKind::Cosh,
            report_covariance: true,
            ..request.spectroscopy.analysis.clone()
        })
        .unwrap();
    assert_eq!(session.snapshot().unwrap(), before);
    assert_ne!(json!(first.analysis), json!(second.analysis));
    assert_eq!(
        first.measurement_fingerprint,
        second.measurement_fingerprint
    );
    assert_eq!(first.frames, second.frames);
    assert_eq!(first.channels.len(), second.channels.len());
}

#[test]
fn exported_evidence_re_analyses_to_the_report_the_session_gave() {
    let request = short("euclidean", 32, 2, 20);
    let session = finished(request.clone());
    let evidence = session.evidence();
    let bytes = evidence.to_bytes().unwrap();
    let imported = SpectroscopyEvidence::from_bytes(&bytes).unwrap();
    assert_eq!(imported, evidence);
    let analysis = &request.spectroscopy.analysis;
    assert_eq!(
        json!(analyze_evidence(&imported, analysis).unwrap()),
        json!(session.analyze(analysis).unwrap())
    );
    let mut tampered = imported;
    tampered.request.seed += 1;
    assert!(matches!(
        analyze_evidence(&tampered, analysis),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn an_extinct_population_ends_its_replica_as_a_terminal_status() {
    block_on(async {
        let mut request = short("euclidean", 16, 2, 12);
        request.replicas = 1;
        // One uncapped kick of this size puts every walker outside the
        // absorbing box, so the second step finds no eligible walker left.
        request.run.gas.kinetic.velocity_cap = None;
        request.run.gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
            scale: FactorValues::Constant { values: vec![1e6] },
        };
        let mut session = SpectroscopySession::create(request).await.unwrap();
        for _ in 0..4 {
            session.advance(8).await.unwrap();
        }
        let snapshot = session.snapshot().unwrap();
        assert_eq!(
            snapshot["replicas"][0]["terminal"],
            json!("extinct_population")
        );
        assert!(session.done());
        assert_eq!(snapshot["done"], json!(true));
    });
}

#[test]
fn a_variant_request_rebuilds_the_dimension_dependent_run_and_resolves_only_once() {
    let default = SpectroscopyRequest::default();
    assert_eq!(default.clone().resolve().unwrap(), default);
    let stated = SpectroscopyRequest {
        variant: Some("euclidean".into()),
        run: RunConfig {
            walkers: 128,
            dimensions: 4,
            ..RunConfig::default()
        },
        ..SpectroscopyRequest::default()
    }
    .resolve()
    .unwrap();
    assert_eq!(stated.run.walkers, 128);
    assert_eq!(stated.run.dimensions, 4);
    assert_eq!(stated.run.gas.precision, Precision::F64);
    assert_eq!(stated.capabilities().dimension, 4);
    assert_eq!(stated.clone().resolve().unwrap(), stated);
    let bare = SpectroscopyRequest {
        variant: Some("viscous_euclidean".into()),
        run: RunConfig::default(),
        ..SpectroscopyRequest::default()
    }
    .resolve()
    .unwrap();
    assert_eq!((bare.run.walkers, bare.run.dimensions), (200, 3));
    assert!(bare.capabilities().dense_viscosity);
    assert!(matches!(
        SpectroscopyRequest {
            variant: Some("geometric".into()),
            run: RunConfig::default(),
            ..SpectroscopyRequest::default()
        }
        .resolve(),
        Err(GasError::Capability(_))
    ));
    assert!(
        SpectroscopyRequest {
            variant: Some("no_such_variant".into()),
            ..SpectroscopyRequest::default()
        }
        .resolve()
        .is_err()
    );
}

#[test]
fn the_defaults_payload_carries_every_variant_the_catalog_and_the_reference_table() {
    let payload = defaults().unwrap();
    // The payload announces the crate's own schema version, the one every
    // stored container is checked against, and not a literal that drifts away
    // from it: `SPECTROSCOPY_VERSION` rose to 2 with the audit corrections.
    assert_eq!(payload["schema_version"], json!(SPECTROSCOPY_VERSION));
    let variants = payload["variants"].as_array().unwrap();
    assert_eq!(variants.len(), 6);
    for row in variants {
        assert_eq!(
            row["request"].is_null(),
            !row["implemented"].as_bool().unwrap(),
            "{}",
            row["name"]
        );
        assert!(
            row["book_label"]
                .as_str()
                .unwrap()
                .starts_with("def-variant-")
        );
    }
    assert_eq!(
        variants[1]["request"]["chunk"],
        json!(4),
        "a dense viscous kernel needs the short chunk"
    );
    assert!(payload["catalog"].as_array().unwrap().len() > 30);
    assert!(payload["reference"]["entries"].as_array().unwrap().len() > 3);
    assert_eq!(payload["request"], json!(SpectroscopyRequest::default()));
    let view = capabilities(&SpectroscopyRequest::default()).unwrap();
    let channels = view["channels"].as_array().unwrap();
    assert_eq!(channels.len(), payload["catalog"].as_array().unwrap().len());
    assert!(channels.iter().any(|c| c["requested"] == json!(true)));
    assert!(channels.iter().any(|c| c["requested"] == json!(false)));
    assert_eq!(view["chunk"], json!(16));
    assert_eq!(view["capabilities"]["dimension"], json!(3));
}
