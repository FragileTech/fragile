//! Native CLI and browser consume the same Rust registry and run evidence.
use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};
fn cli(args: &[&str], v: &Value) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_algorithmic-gas-qft"))
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
        .write_all(serde_json::to_string(v).unwrap().as_bytes())
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
#[test]
fn native_registry_covers_all128() {
    let out = success(cli(&["catalog"], &json!(null)));
    assert_eq!(out.as_array().unwrap().len(), 128);
}
#[test]
fn recorded_native_measurements_recompute_from_evidence() {
    for id in ["I-07", "V-02", "VI-08", "VI-19"] {
        let r = json!({"id":id,"seed":37,"steps":32,"parameters":{}});
        let first = success(cli(&["run", "-"], &r));
        let second = success(cli(&["analyze", "-"], &first["evidence"]));
        assert_eq!(
            first["snapshot"]["result"]["plots"], second["plots"],
            "{id}"
        );
        assert!(first["evidence"]["archives"].is_array());
    }
}
#[test]
fn unexecuted_or_mismatched_evidence_cannot_supply_a_plot() {
    for input in [
        json!({"results":[{"experiment":1}]}),
        json!({"experiment":28}),
        json!({"request":{"id":"VI-01","seed":7,"steps":32},"configs":[],"archives":[]}),
    ] {
        assert!(!cli(&["analyze", "-"], &input).status.success());
    }
    let input = json!({"id":"VI-01","seed":7,"steps":32,"parameters":{"fabricated_curvature":2}});
    assert!(!cli(&["run", "-"], &input).status.success());
}
