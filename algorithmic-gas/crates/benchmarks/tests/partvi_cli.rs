//! CLI-level reproducibility, archive replay and failure-path coverage.
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "fragile-partvi-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn cli(args: &[&str], input: &Value) -> Output {
    let mut process = Command::new(env!("CARGO_BIN_EXE_algorithmic-gas-qft"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    process
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(input).unwrap().as_bytes())
        .unwrap();
    process.wait_with_output().unwrap()
}
fn success(out: Output) -> Value {
    assert!(
        out.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}
fn validate_result(document: &Value, expected: usize) {
    assert_eq!(document["schema"], "fragile-partvi-results-v1");
    let results = document["results"].as_array().unwrap();
    assert_eq!(results.len(), expected);
    for result in results {
        assert!(result["title"].as_str().is_some_and(|s| !s.is_empty()));
        assert!(result["model"].as_str().is_some_and(|s| !s.is_empty()));
        assert!(result["details"].is_object());
        for metric in result["metrics"].as_array().unwrap() {
            if !metric["value"].is_null() {
                assert!(metric["value"].as_f64().unwrap().is_finite());
            }
        }
        for plot in result["plots"].as_array().unwrap() {
            for series in plot["series"].as_array().unwrap() {
                for point in series["points"].as_array().unwrap() {
                    assert_eq!(point.as_array().unwrap().len(), 2);
                    assert!(
                        point
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|x| x.as_f64().is_some_and(f64::is_finite))
                    );
                }
            }
        }
    }
}
#[test]
fn all_sixty_six_reference_requests_are_reproducible_through_the_cli() {
    let request: Value =
        serde_json::from_str(include_str!("../../../examples/partvi/all-reference.json")).unwrap();
    let first = success(cli(&["sweep", "-"], &request));
    let second = success(cli(&["sweep", "-"], &request));
    validate_result(&first, 66);
    assert_eq!(
        first, second,
        "Seeded reference results changed between independent CLI processes"
    );
    for (i, result) in first["results"].as_array().unwrap().iter().enumerate() {
        assert_eq!(result["experiment"], json!(i + 1));
        assert!(result["details"]["request"].is_object());
    }
}
#[test]
fn saved_native_archive_replays_the_same_request_and_current_schema() {
    let files = Workspace::new();
    let archive = files.0.join("archive.json");
    let path = archive.to_str().unwrap();
    let request = json!({"experiment":8,"parameters":{"seed":37}});
    let run = success(cli(
        &["run", "-", "--steps", "6", "--save-archive", path],
        &request,
    ));
    validate_result(&run, 1);
    let recorded: Value = serde_json::from_slice(&fs::read(&archive).unwrap()).unwrap();
    assert_eq!(recorded["schema_version"], 2);
    assert_eq!(recorded["steps"].as_array().unwrap().len(), 6);
    assert!(recorded["gas_config"]["qft"]["viscosity"].is_object());
    let replay = success(cli(&["analyze", "-", "--archive", path], &request));
    validate_result(&replay, 1);
    assert_eq!(run["results"], replay["results"]);
    assert_eq!(replay["results"][0]["details"]["archive_steps"], 6);
    assert_eq!(
        replay["results"][0]["details"]["source_coverage"]
            .as_array()
            .unwrap()
            .len(),
        32
    );
}
#[test]
fn ensemble_archive_export_is_rejected_before_creating_a_file() {
    let files = Workspace::new();
    let archive = files.0.join("ensemble.json");
    for id in [19, 22, 45] {
        let out = cli(
            &["run", "-", "--save-archive", archive.to_str().unwrap()],
            &json!({"experiment":id,"parameters":{"replicas":2,"horizon":1,"warmup":1}}),
        );
        assert!(!out.status.success());
        assert!(
            String::from_utf8_lossy(&out.stderr)
                .contains("--save-archive requires a single trajectory")
        );
        assert!(!archive.exists());
    }
}
#[test]
fn malformed_sweeps_fail_with_an_explanation() {
    let invalid = [
        json!({"experiment":4294967297u64,"parameter":"n","values":[32]}),
        json!({"experiment":7,"parameters":[],"parameter":"n","values":[32]}),
        json!({"experiment":7,"parameter":"n","values":[{"nested":true}]}),
        json!({"experiment":7,"parameter":"n"}),
    ];
    for request in invalid {
        let output = cli(&["sweep", "-"], &request);
        assert!(
            !output.status.success(),
            "accepted malformed sweep {request}"
        );
        assert!(!output.stderr.is_empty());
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn archive_requests_cannot_report_an_unrelated_finite_fixture() {
    let files = Workspace::new();
    let archive = files.0.join("archive.json");
    let path = archive.to_str().unwrap();
    let native = success(cli(
        &["run", "-", "--steps", "4", "--save-archive", path],
        &json!({"experiment":8,"parameters":{"seed":37}}),
    ));
    assert_eq!(
        native["results"][0]["details"]["calculation_origin"],
        "executed_algorithm_archive"
    );
    assert!(native["results"][0]["details"]["executed_gas_config"].is_object());
    for id in [1, 7, 11, 19, 22, 24, 28, 45, 58, 66] {
        let output = cli(
            &["analyze", "-", "--archive", path],
            &json!({"experiment":id}),
        );
        assert!(
            !output.status.success(),
            "VI-{id} silently accepted an unrelated archive"
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("no single-archive calculation"));
    }
    let finite = success(cli(&["analyze", "-"], &json!({"experiment":28})));
    assert_eq!(
        finite["results"][0]["details"]["calculation_origin"],
        "constructed_finite_model"
    );
    let replicas = success(cli(
        &["run", "-"],
        &json!({"experiment":19,"parameters":{"replicas":4,"horizon":1}}),
    ));
    assert_eq!(
        replicas["results"][0]["details"]["calculation_origin"],
        "independent_algorithm_continuations"
    );
}
