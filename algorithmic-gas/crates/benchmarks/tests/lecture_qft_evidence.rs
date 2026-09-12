use algorithmic_gas::physics::partvi::ExperimentRequest;
use algorithmic_gas_benchmarks::{lecture, qft_experiments};
use serde_json::json;
#[test]
fn checkpoint_evidence_replays_all_actual_continuation_experiments() {
    futures_lite::future::block_on(async {
        for experiment in [19, 22, 45] {
            let config =
                lecture::gas_config(&format!("VI-{experiment:02}"), &json!({"walkers":8}), 516)
                    .unwrap();
            let request = ExperimentRequest {
                experiment,
                parameters: json!({"replicas":2,"horizon":1,"warmup":1}),
            };
            let result = qft_experiments::run(&config, &request).await.unwrap();
            let evidence = &result.details["replay_evidence"];
            let rows = evidence["runs"].as_array().unwrap();
            assert_eq!(rows.len(), if experiment == 19 { 8 } else { 4 });
            assert!(rows.iter().all(|r| r["recorded_steps"] == 1));
            assert!(rows.iter().any(|r| r["archive_cbor"].is_array()));
            assert!(evidence["frozen_checkpoint_cbor"].is_array());
            let replay = qft_experiments::replay_evidence(evidence).await.unwrap();
            assert_eq!(
                serde_json::to_value(result).unwrap(),
                serde_json::to_value(replay).unwrap()
            );
        }
    })
}
#[test]
fn corrupted_continuation_manifest_is_rejected() {
    futures_lite::future::block_on(async {
        let config = lecture::gas_config("VI-19", &json!({"walkers":8}), 7).unwrap();
        let request = ExperimentRequest {
            experiment: 19,
            parameters: json!({"replicas":2,"horizon":1,"warmup":1}),
        };
        let result = qft_experiments::run(&config, &request).await.unwrap();
        let mut evidence = result.details["replay_evidence"].clone();
        evidence["runs"][0]["archive_fingerprint_fnv1a64"] = json!("0000000000000000");
        assert!(qft_experiments::replay_evidence(&evidence).await.is_err());
        let mut evidence = result.details["replay_evidence"].clone();
        evidence["runs"][0]["terminal_readout"] = json!(999.);
        assert!(qft_experiments::replay_evidence(&evidence).await.is_err());
    })
}

#[test]
fn browser_json_roundtrip_preserves_continuation_evidence() {
    fn javascript_numbers(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Number(n) if n.is_f64() => {
                let f = n.as_f64().unwrap();
                if f.fract() == 0. && f.abs() < 9_007_199_254_740_992. {
                    *value = json!(f as i64);
                }
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(javascript_numbers),
            serde_json::Value::Object(o) => o.values_mut().for_each(javascript_numbers),
            _ => {}
        }
    }
    futures_lite::future::block_on(async {
        for id in ["VI-19", "VI-22", "VI-45"] {
            let mut session =
                lecture::LectureSession::create(algorithmic_gas::lecture::LectureRequest {
                    id: id.into(),
                    seed: 7,
                    steps: 8,
                    parameters: json!({"walkers":8,"replicas":4,"horizon":1}),
                })
                .await
                .unwrap();
            session.advance(8).await.unwrap();
            let expected = session.snapshot().unwrap();
            assert_eq!(
                expected["result"]["details"]["frozen_step"], 8,
                "The measured checkpoint must be the displayed completed run"
            );
            let bytes: Vec<u8> = serde_json::from_value(
                expected["result"]["details"]["replay_evidence"]["frozen_checkpoint_cbor"].clone(),
            )
            .unwrap();
            let checkpoint = algorithmic_gas::Checkpoint::<f64>::from_bytes(&bytes).unwrap();
            assert!(
                checkpoint
                    .recording
                    .as_ref()
                    .is_none_or(|a| a.steps.len() <= 4)
            );
            assert!(
                bytes.len() < 4 * 1024 * 1024,
                "Conditioning retains state and donor memory without copying the full audit archive"
            );

            assert_eq!(
                checkpoint.population,
                session.evidence().archives[0]
                    .steps
                    .last()
                    .unwrap()
                    .final_population
            );

            let mut value = serde_json::to_value(session.evidence()).unwrap();
            javascript_numbers(&mut value);
            let evidence = serde_json::from_value(value).unwrap();
            let replay = lecture::analyze_evidence(&evidence).await.unwrap();
            assert_eq!(
                serde_json::to_value(replay.plots).unwrap(),
                expected["result"]["plots"]
            );
        }
    });
}

#[test]
fn continuation_evidence_rejects_a_different_conditioning_state() {
    futures_lite::future::block_on(async {
        let mut run = lecture::LectureSession::create(algorithmic_gas::lecture::LectureRequest {
            id: "VI-19".into(),
            seed: 7,
            steps: 8,
            parameters: json!({"walkers":8,"replicas":4,"horizon":1}),
        })
        .await
        .unwrap();
        run.advance(8).await.unwrap();
        let mut evidence = run.evidence();
        let mut other = lecture::LectureSession::create(algorithmic_gas::lecture::LectureRequest {
            id: "VI-19".into(),
            seed: 7,
            steps: 4,
            parameters: json!({"walkers":8,"replicas":4,"horizon":1}),
        })
        .await
        .unwrap();
        other.advance(4).await.unwrap();
        evidence.continuation = other.evidence().continuation;
        let error = lecture::analyze_evidence(&evidence).await.unwrap_err();
        assert!(error.to_string().contains("displayed archive state"));
    });
}
