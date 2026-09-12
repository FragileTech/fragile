use algorithmic_gas::lecture::{LectureRequest, catalog};
use algorithmic_gas_benchmarks::lecture::LectureSession;
use serde_json::json;

#[test]
fn all_registered_defaults_execute_and_measure_real_gas() {
    futures_lite::future::block_on(async {
        assert_eq!(catalog().len(), 128);
        let selection = std::env::var("LECTURE_TEST_PART").unwrap_or_default();
        for spec in catalog()
            .iter()
            .filter(|s| selection.is_empty() || s["part"] == selection)
        {
            let id = spec["id"].as_str().unwrap();
            eprintln!("Executing {id}");
            let mut session = LectureSession::create(LectureRequest {
                id: id.into(),
                seed: 7,
                parameters: json!({}),
                steps: 32,
            })
            .await
            .unwrap_or_else(|e| panic!("{id}: {e}"));
            while !session.done() {
                session
                    .advance(8)
                    .await
                    .unwrap_or_else(|e| panic!("{id}: {e}"));
            }
            let result = session.snapshot().unwrap();
            if let Ok(dir) = std::env::var("LECTURE_RESULTS_DIR") {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(
                    std::path::Path::new(&dir).join(format!("{id}.json")),
                    serde_json::to_vec(&result).unwrap(),
                )
                .unwrap();
            }
            assert!(result["result"].is_object(), "{id}: no measured result");
            assert!(
                result["result"]["plots"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|p| p["series"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|s| s["points"].as_array().is_some_and(|p| !p.is_empty()))),
                "{id}: no measured plot"
            );
            let evidence = session.evidence();
            assert!(evidence.archives.iter().all(|a| !a.steps.is_empty()));
            for a in evidence.archives {
                a.validate().unwrap();
            }
        }
    });
}

#[test]
fn session_checkpoint_preserves_actual_future_and_measurements() {
    futures_lite::future::block_on(async {
        let mut run = LectureSession::create(LectureRequest {
            id: "V-02".into(),
            seed: 516,
            parameters: json!({}),
            steps: 16,
        })
        .await
        .unwrap();
        run.advance(8).await.unwrap();
        let checkpoint = run.checkpoint().unwrap();
        let mut restored = LectureSession::restore(&checkpoint).await.unwrap();
        assert_eq!(
            run.advance(8).await.unwrap(),
            restored.advance(8).await.unwrap()
        );
        assert_eq!(run.evidence().archives, restored.evidence().archives);
    });
}

#[test]
fn requests_reject_unconsumed_controls() {
    let r = LectureRequest {
        id: "V-01".into(),
        seed: 7,
        parameters: json!({"fabricated_curvature":1.}),
        steps: 32,
    };
    assert!(r.resolve().is_err());
}

#[test]
fn evidence_rejects_a_different_algorithm_request() {
    futures_lite::future::block_on(async {
        let mut run = LectureSession::create(LectureRequest {
            id: "V-02".into(),
            seed: 7,
            parameters: json!({}),
            steps: 8,
        })
        .await
        .unwrap();
        run.advance(8).await.unwrap();
        let mut evidence = run.evidence();
        evidence.request.parameters["engine_viscosity"] = json!(1.);
        assert!(algorithmic_gas_benchmarks::lecture::analyze(&evidence).is_err());
    });
}
#[test]
fn requests_reject_fractional_discrete_counts() {
    let r = LectureRequest {
        id: "VI-28".into(),
        seed: 7,
        parameters: json!({"modes":3.5}),
        steps: 32,
    };
    assert!(r.resolve().is_err());
}
