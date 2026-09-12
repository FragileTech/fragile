use algorithmic_gas::lecture::LectureRequest;
use algorithmic_gas_benchmarks::lecture::LectureSession;
use serde_json::json;

#[test]
fn actual_default_curvature_refines_independently_of_the_packed_prediction() {
    futures_lite::future::block_on(async {
        let mut session = LectureSession::create(LectureRequest {
            id: "VI-39".into(),
            seed: 7,
            steps: 96,
            parameters: json!({}),
        })
        .await
        .unwrap();
        while !session.done() {
            session.advance(8).await.unwrap();
        }
        let snapshot = session.snapshot().unwrap();
        let details = &snapshot["result"]["details"];
        let refinement = &details["finite_difference_refinement"];
        assert_eq!(details["run_steps"], json!([96]));
        assert_eq!(details["archive_field"]["step"], 96);
        assert_eq!(refinement["status"], "converged");
        let sequence = refinement["sequence"].as_array().unwrap();
        assert!(sequence.len() <= 13);
        let packed = details["curvature"]["scalar"].as_f64().unwrap();
        assert_eq!(sequence[0]["status"], "unsupported_stencil");
        let first = sequence
            .iter()
            .find_map(|row| row["scalar"].as_f64())
            .unwrap();
        let last = sequence.last().unwrap();
        let numerical = details["numerical_metric_curvature"]["scalar"]
            .as_f64()
            .unwrap();
        eprintln!(
            "packed={packed:.12}, numerical={numerical:.12}, selected_step={}, status={}",
            refinement["selected_step"], refinement["status"]
        );
        // This actual archived field defeats a single coarse difference stencil.
        assert!((first - packed).abs() / packed.abs() > 1e-4);
        assert!((numerical - packed).abs() / packed.abs() < 1e-5);
        assert_eq!(last["scalar"], json!(numerical));
        assert!(refinement["selected_step"].as_f64().unwrap() < 0.0002);
        for pair in sequence.windows(2) {
            assert_eq!(
                pair[1]["step"].as_f64().unwrap(),
                pair[0]["step"].as_f64().unwrap() * 0.5
            );
        }
        for row in sequence.iter().rev().take(2) {
            assert!(row["scaled_change"].as_f64().unwrap() <= 1.);
        }
        assert_eq!(
            details["archive_field"]["source"]["frame"], 94,
            "refinement must preserve the actual immutable donor context"
        );
    });
}
