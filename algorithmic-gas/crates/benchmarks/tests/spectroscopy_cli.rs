//! The native CLI and the browser read the same spectroscopy payloads.
use algorithmic_gas::{
    RecordingConfig,
    physics::spectroscopy::{AnalysisConfig, MeasurementConfig, SpectroscopyConfig},
};
use algorithmic_gas_benchmarks::{
    RunConfig,
    spectroscopy::{SpectroscopyRequest, SpectroscopySession, analyze_archive, defaults},
};
use futures_lite::future::block_on;
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
fn cli(args: &[&str], input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_gas-spectroscopy"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
fn success(out: std::process::Output) -> Value {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}
fn scratch(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name)
}
/// One replica of a handful of frames: the CLI runs it to the end.
fn request() -> SpectroscopyRequest {
    SpectroscopyRequest {
        variant: Some("euclidean".into()),
        run: RunConfig {
            walkers: 16,
            dimensions: 2,
            ..RunConfig::default()
        },
        steps: 14,
        replicas: 1,
        seed: 7,
        chunk: 4,
        spectroscopy: SpectroscopyConfig {
            measurement: MeasurementConfig {
                warmup: 2,
                max_lag: 5,
                ..MeasurementConfig::default()
            },
            analysis: AnalysisConfig::default(),
        },
    }
    .resolve()
    .unwrap()
}

#[test]
fn the_defaults_and_variants_commands_print_the_browser_payload() {
    let payload = success(cli(&["defaults"], ""));
    assert_eq!(payload, defaults().unwrap());
    let variants = success(cli(&["variants"], ""));
    assert_eq!(variants, payload["variants"]);
    assert_eq!(variants.as_array().unwrap().len(), 6);
    let path = scratch("spectroscopy_defaults.json");
    assert!(
        cli(&["defaults", "--output", path.to_str().unwrap()], "")
            .status
            .success()
    );
    assert_eq!(
        serde_json::from_str::<Value>(&fs::read_to_string(&path).unwrap()).unwrap(),
        payload
    );
}

#[test]
fn a_command_line_run_reports_what_the_library_analyses_and_leaves_reusable_evidence() {
    let request = request();
    let evidence = scratch("spectroscopy_evidence.cbor");
    let printed = success(cli(
        &["run", "-", "--evidence", evidence.to_str().unwrap()],
        &serde_json::to_string(&request).unwrap(),
    ));
    let expected = block_on(async {
        let mut session = SpectroscopySession::create(request.clone()).await.unwrap();
        while !session.done() {
            session.advance(64).await.unwrap();
        }
        json!(session.analyze(&request.spectroscopy.analysis).unwrap())
    });
    assert_eq!(printed, expected);
    let reanalysed = success(cli(&["analyze", evidence.to_str().unwrap()], ""));
    assert_eq!(reanalysed, printed);
    let analysis = scratch("spectroscopy_analysis.json");
    fs::write(
        &analysis,
        serde_json::to_string(&AnalysisConfig {
            report_covariance: true,
            ..AnalysisConfig::default()
        })
        .unwrap(),
    )
    .unwrap();
    let other = success(cli(
        &[
            "analyze",
            evidence.to_str().unwrap(),
            analysis.to_str().unwrap(),
        ],
        "",
    ));
    assert_eq!(other["analysis"]["report_covariance"], json!(true));
    assert_ne!(other["analysis"], printed["analysis"]);
}

#[test]
fn a_recorded_archive_is_measured_and_analysed_without_the_engine() {
    let request = request();
    let archive = block_on(async {
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
        gas.stop_recording().unwrap()
    });
    let request_path = scratch("spectroscopy_request.json");
    let archive_path = scratch("spectroscopy_archive.cbor");
    fs::write(&request_path, serde_json::to_string(&request).unwrap()).unwrap();
    fs::write(&archive_path, archive.to_bytes().unwrap()).unwrap();
    let printed = success(cli(
        &[
            "archive",
            request_path.to_str().unwrap(),
            archive_path.to_str().unwrap(),
        ],
        "",
    ));
    assert_eq!(
        printed,
        json!(analyze_archive(&request.spectroscopy, &[archive]).unwrap())
    );
    assert_eq!(
        printed["calculation_origin"],
        json!("executed_algorithm_archive")
    );
}

#[test]
fn an_unknown_command_or_a_missing_argument_is_refused() {
    for args in [
        vec!["spectrum"],
        vec!["run"],
        vec!["analyze"],
        vec!["archive"],
        vec!["defaults", "--output"],
    ] {
        assert!(!cli(&args, "").status.success(), "{args:?}");
    }
    for args in [vec![], vec!["spectrum"]] {
        let refusal = cli(&args, "");
        assert!(
            String::from_utf8_lossy(&refusal.stderr).contains("Usage: gas-spectroscopy defaults"),
            "{args:?}"
        );
        assert!(refusal.stdout.is_empty(), "{args:?}");
    }
    assert!(
        !cli(&["run", "-"], "{\"replica\":2}").status.success(),
        "an unknown request field is refused"
    );
}
