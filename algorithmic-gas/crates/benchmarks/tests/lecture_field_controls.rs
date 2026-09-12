//! Native coverage generated from the same catalog that drives WASM controls.
use algorithmic_gas::{
    GasError, RecordingConfig,
    lecture::{LectureRequest, catalog},
    physics::{fields, partvi::ExperimentRequest},
};
use algorithmic_gas_benchmarks::lecture::{LectureSession, gas_config};
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn steps(id: &str, parameters: &Value) -> usize {
    if matches!(
        id,
        "VI-04" | "VI-12" | "VI-13" | "VI-28" | "VI-32" | "VI-35" | "VI-36"
    ) {
        32.max(
            parameters
                .get("lag")
                .or_else(|| parameters.get("max_lag"))
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize
                * 6
                + 8,
        )
    } else {
        4
    }
}
async fn execute(id: &str, parameters: Value, seed: u64, default_run: bool) -> Result<(), String> {
    let count = if default_run {
        32
    } else {
        steps(id, &parameters)
    };
    let request = LectureRequest {
        id: id.into(),
        seed,
        parameters: parameters.clone(),
        steps: count,
    };
    let mut session = LectureSession::create(request)
        .await
        .map_err(|e| format!("create: {e}"))?;
    let mut output = None;
    while !session.done() {
        output = Some(
            session
                .advance(32)
                .await
                .map_err(|e| format!("advance: {e}"))?,
        );
    }
    let result = output
        .as_ref()
        .and_then(|o| o.get("result"))
        .ok_or("missing result")?;
    if result.is_null() {
        return Err("null result after completed engine run".into());
    }
    let plots = result["plots"]
        .as_array()
        .ok_or("missing scientific plots")?;
    let points = plots
        .iter()
        .flat_map(|p| p["series"].as_array().into_iter().flatten())
        .flat_map(|s| s["points"].as_array().into_iter().flatten())
        .collect::<Vec<_>>();
    if points.is_empty() {
        return Err("no measured plot points".into());
    }
    if points.iter().any(|p| {
        p.as_array()
            .is_none_or(|a| a.iter().any(|v| v.as_f64().is_none_or(|x| !x.is_finite())))
    }) {
        return Err("nonfinite plot coordinate".into());
    }
    let evidence = session.evidence();
    if evidence.archives.iter().any(|a| a.steps.is_empty()) {
        return Err("experiment has an empty engine archive".into());
    }
    for a in &evidence.archives {
        a.validate().map_err(|e| format!("archive: {e}"))?;
    }
    if matches!(id, "VI-19" | "VI-22" | "VI-45") {
        let bytes: Vec<u8> = serde_json::from_value(
            result["details"]["replay_evidence"]["frozen_checkpoint_cbor"].clone(),
        )
        .map_err(|e| format!("frozen checkpoint evidence: {e}"))?;
        let checkpoint = algorithmic_gas::Checkpoint::<f64>::from_bytes(&bytes)
            .map_err(|e| format!("decode frozen checkpoint: {e}"))?;
        let last = evidence.archives[0]
            .steps
            .last()
            .ok_or("missing displayed final step")?;
        if checkpoint.step != last.report.step
            || result["details"]["frozen_step"] != json!(last.report.step)
            || serde_json::to_value(&checkpoint.population).unwrap()
                != serde_json::to_value(&last.final_population).unwrap()
        {
            return Err(
                "continuations do not start from the displayed final population and step".into(),
            );
        }
    }
    for c in &evidence.configs {
        if let Some(v) = parameters.get("walkers").and_then(Value::as_u64)
            && c.walkers != v as usize
        {
            return Err("walker control did not configure actual population".into());
        }
        if let Some(v) = parameters.get("engine_memory").and_then(Value::as_u64)
            && (c.gas.distance_donors.history_window != v as usize
                || c.gas.cloning_donors.history_window != v as usize)
        {
            return Err("memory control did not configure both actual donor histories".into());
        }
        if let Some(v) = parameters.get("engine_dt").and_then(Value::as_f64)
            && !matches!(c.gas.kinetic.integrator,algorithmic_gas::kinetic::KineticKind::Baoab{dt,..}if dt==v)
        {
            return Err("time-step control did not configure actual BAOAB update".into());
        }
        if let Some(v) = parameters.get("engine_viscosity").and_then(Value::as_f64)
            && c.gas
                .qft
                .viscosity
                .as_ref()
                .is_none_or(|q| q.coefficient != v)
        {
            return Err("viscosity control did not configure actual force".into());
        }
        if let Some(v) = parameters.get("engine_innovation")
            && serde_json::to_value(c.gas.kinetic.noise.innovation).unwrap() != *v
        {
            return Err("innovation control did not configure actual noise carrier".into());
        }
    }
    Ok(())
}
#[test]
fn every_v_and_vi_default_runs_for_three_independent_seeds() {
    futures_lite::future::block_on(async {
        let mut failures = vec![];
        let mut count = 0;
        for spec in catalog()
            .iter()
            .filter(|s| s["part"] == "V" || s["part"] == "VI")
        {
            let id = spec["id"].as_str().unwrap();
            for seed in [0, 7, 516] {
                count += 1;
                if let Err(e) = execute(id, json!({}), seed, true).await {
                    let failure = format!("{id} seed{seed}: {e}");
                    eprintln!("{failure}");
                    failures.push(failure);
                }
            }
            eprintln!("DEFAULT {id} completed");
        }
        assert_eq!(count, 258);
        assert!(
            failures.is_empty(),
            "{} default failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}
#[test]
fn every_v_and_vi_control_endpoint_executes_its_declared_rust_route() {
    futures_lite::future::block_on(async {
        let mut failures = vec![];
        let mut count = 0;
        for spec in catalog()
            .iter()
            .filter(|s| s["part"] == "V" || s["part"] == "VI")
        {
            let id = spec["id"].as_str().unwrap();
            let mut seen = BTreeSet::new();
            for control in spec["controls"].as_array().unwrap() {
                let key = control["key"].as_str().unwrap();
                let values = if control["type"] == "select" {
                    control["options"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v["value"].clone())
                        .collect::<Vec<_>>()
                } else {
                    vec![control["min"].clone(), control["max"].clone()]
                };
                for value in values {
                    let mut parameters = json!({});
                    parameters[key] = value.clone();
                    if !seen.insert(parameters.to_string()) {
                        continue;
                    }
                    count += 1;
                    if let Err(e) = execute(id, parameters.clone(), 7, false).await {
                        let failure = format!("{id} {parameters}: {e}");
                        eprintln!("{failure}");
                        failures.push(failure);
                    }
                }
            }
            eprintln!("ENDPOINT {id} completed");
        }
        eprintln!("CONTROL_ENDPOINT_CASES {count}");
        assert!(count > 1000);
        assert!(
            failures.is_empty(),
            "{} endpoint failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}
#[test]
fn missing_conditional_provider_is_an_explicit_capability_error() {
    futures_lite::future::block_on(async {
        let mut config = gas_config("VI-37", &json!({"walkers":8}), 7).unwrap();
        config.physics_metric = None;
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..4 {
            gas.step().await.unwrap();
        }
        for id in [37, 38, 39, 40, 62] {
            let error = fields::analyze(
                &ExperimentRequest {
                    experiment: id,
                    parameters: json!({}),
                },
                gas.recording(),
            )
            .unwrap_err();
            assert!(matches!(error, GasError::Capability(_)), "{id}: {error}");
            assert!(error.to_string().contains("provider"), "{id}: {error}");
        }
    });
}

#[test]
fn generated_stress_windows_include_resolved_lag_guards() {
    for id in ["VI-04", "VI-12", "VI-13", "VI-28", "VI-32", "VI-36"] {
        for request in algorithmic_gas::lecture::stress_cases(id).unwrap() {
            let lag = request
                .parameters
                .get("lag")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .max(
                    request
                        .parameters
                        .get("max_lag")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                ) as usize;
            assert!(request.steps >= 6 * lag + 8, "{request:?}");
            if id == "VI-28" {
                assert!(request.steps > 2 * request.parameters["modes"].as_u64().unwrap() as usize);
            }
            assert!(request.resolve().is_ok());
        }
    }
}

#[test]
fn wave_operator_plot_keeps_original_slots_when_walkers_are_killed() {
    futures_lite::future::block_on(async {
        let parameters = json!({"walkers":32,"epsilon":0.05,"engine_dt":0.1});
        let mut config = gas_config("V-15", &parameters, 7).unwrap();
        config.gas.boundary = algorithmic_gas::boundary::BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: algorithmic_gas::boundary::BoxDomain {
                lower: vec![-0.65; 2],
                upper: vec![0.65; 2],
            },
        };
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..12 {
            gas.step().await.unwrap();
        }
        let mut archive = gas.recording().unwrap().clone();
        let record = archive
            .steps
            .iter()
            .enumerate()
            .rfind(|(i, s)| *i >= 2 && s.before.validity.iter().any(|v| !v.eligible(false)))
            .map(|(i, _)| i)
            .unwrap_or_else(|| {
                panic!(
                    "test must exercise ineligible slots: {:?}",
                    archive
                        .steps
                        .iter()
                        .map(|s| (
                            s.before
                                .validity
                                .iter()
                                .filter(|v| v.eligible(false))
                                .count(),
                            s.final_population
                                .validity
                                .iter()
                                .filter(|v| v.eligible(false))
                                .count()
                        ))
                        .collect::<Vec<_>>()
                )
            });
        archive.steps.truncate(record + 1);
        let expected = archive
            .steps
            .last()
            .unwrap()
            .before
            .validity
            .iter()
            .enumerate()
            .filter(|(_, v)| v.eligible(false))
            .map(|(i, _)| i as f64)
            .collect::<Vec<_>>();
        let result =
            algorithmic_gas_benchmarks::lecture_fractal::analyze("V-15", &parameters, &archive)
                .unwrap();
        let spatial = result
            .plots
            .iter()
            .flat_map(|p| &p.series)
            .find(|s| s.name == "Empirical spatial kernel generator")
            .unwrap();
        assert_eq!(
            spatial.points.iter().map(|p| p[0]).collect::<Vec<_>>(),
            expected
        );
    });
}

#[test]
fn displayed_checkpoint_continuations_cover_defaults_and_control_endpoints() {
    futures_lite::future::block_on(async {
        let mut failures = vec![];
        let mut count = 0;
        for spec in catalog()
            .iter()
            .filter(|s| matches!(s["id"].as_str(), Some("VI-19" | "VI-22" | "VI-45")))
        {
            let id = spec["id"].as_str().unwrap();
            for seed in [0, 7, 516] {
                count += 1;
                if let Err(e) = execute(id, json!({}), seed, true).await {
                    failures.push(format!("{id} default seed {seed}: {e}"));
                }
            }
            let mut seen = BTreeSet::new();
            for control in spec["controls"].as_array().unwrap() {
                let key = control["key"].as_str().unwrap();
                let values = if control["type"] == "select" {
                    control["options"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v["value"].clone())
                        .collect::<Vec<_>>()
                } else {
                    vec![control["min"].clone(), control["max"].clone()]
                };
                for value in values {
                    let mut parameters = json!({});
                    parameters[key] = value;
                    if !seen.insert(parameters.to_string()) {
                        continue;
                    }
                    count += 1;
                    if let Err(e) = execute(id, parameters.clone(), 7, false).await {
                        failures.push(format!("{id} {parameters}: {e}"));
                    }
                }
            }
            eprintln!("DISPLAYED_CHECKPOINT {id} defaults and endpoints completed");
        }
        eprintln!("DISPLAYED_CHECKPOINT_CASES {count}");
        assert!(count > 50);
        assert!(
            failures.is_empty(),
            "{} failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
    });
}
