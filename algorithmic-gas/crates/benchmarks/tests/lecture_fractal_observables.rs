use algorithmic_gas::RecordingConfig;
use algorithmic_gas_benchmarks::{lecture, lecture_fractal};
use serde_json::json;
#[test]
fn every_fractal_measurement_uses_one_actual_planar_archive() {
    futures_lite::future::block_on(async {
        let parameters = json!({"walkers":8,"resolution":2});
        let mut config = lecture::gas_config("V-20", &parameters, 7).unwrap();
        config.physics_metric.as_mut().unwrap().epsilon = 1.;
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..8 {
            gas.step().await.unwrap();
        }
        let archive = gas.recording().unwrap();
        let before = archive.to_bytes().unwrap();
        for i in 1..=20 {
            let id = format!("V-{i:02}");
            let result = lecture_fractal::analyze(&id, &parameters, archive)
                .unwrap_or_else(|e| panic!("{id}: {e}"));
            assert_eq!(
                result.details["calculation_origin"],
                "executed_algorithm_archive"
            );
            assert!(!result.plots.is_empty(), "{id}");
            let scene = &result.details["scene"];
            let nodes = scene["nodes"].as_array().unwrap();
            assert!(!nodes.is_empty(), "{id}: missing actual scene nodes");
            for n in nodes {
                let p = n["position"].as_array().unwrap();
                assert_eq!(p.len(), 3);
                assert!(p.iter().all(|v| v.as_f64().unwrap().is_finite()));
            }
            for f in scene["faces"].as_array().unwrap() {
                let vertices = f["vertices"].as_array().unwrap();
                assert!(vertices.len() >= 3, "{id}: invalid scene face");
                for p in vertices {
                    assert_eq!(p.as_array().unwrap().len(), 3);
                }
            }
            if matches!(i, 7 | 9 | 10 | 11 | 12) {
                assert!(
                    !scene["faces"].as_array().unwrap().is_empty(),
                    "{id}: missing metric cell scene"
                );
            }
            assert!(
                result
                    .plots
                    .iter()
                    .flat_map(|p| &p.series)
                    .flat_map(|s| &s.points)
                    .flatten()
                    .all(|x| x.is_finite()),
                "{id}"
            );
            if i == 5 || i == 13 {
                for row in result.details["conditional_derivative_audit"]
                    .as_array()
                    .unwrap()
                {
                    let norm = row["recorded_hessian"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_f64().unwrap().abs())
                        .fold(1., f64::max);
                    assert!(
                        row["absolute_residual"].as_f64().unwrap() < norm * 1e-3,
                        "{id}: {row}"
                    );
                }
            }
            if i == 11 {
                assert_eq!(
                    result.details["slab_slots"]["tracked_slots"]
                        .as_array()
                        .unwrap()
                        .len(),
                    8
                );
            }
        }
        assert_eq!(before, archive.to_bytes().unwrap());
    });
}
#[test]
fn planar_curvature_and_cell_measurements_recompute_after_replay() {
    futures_lite::future::block_on(async {
        let parameters = json!({"walkers":8,"resolution":1});
        let config = lecture::gas_config("V-20", &parameters, 516).unwrap();
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..4 {
            gas.step().await.unwrap();
        }
        let archive = gas.recording().unwrap();
        let roundtrip =
            algorithmic_gas::RunArchive::<f64>::from_bytes(&archive.to_bytes().unwrap()).unwrap();
        for id in ["V-05", "V-10", "V-11", "V-12", "V-13", "V-20"] {
            let first = lecture_fractal::analyze(id, &parameters, archive).unwrap();
            let second = lecture_fractal::analyze(id, &parameters, &roundtrip).unwrap();
            assert_eq!(
                serde_json::to_value(first).unwrap(),
                serde_json::to_value(second).unwrap(),
                "{id}"
            );
        }
    });
}

#[test]
fn changing_eligibility_preserves_slot_labels_in_cell_interfaces() {
    futures_lite::future::block_on(async {
        let parameters = json!({"walkers":32});
        let mut config = lecture::gas_config("V-10", &parameters, 7).unwrap();
        config.gas.boundary = algorithmic_gas::boundary::BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: algorithmic_gas::boundary::BoxDomain {
                lower: vec![-0.65; 2],
                upper: vec![0.65; 2],
            },
        };
        config.physics_metric.as_mut().unwrap().epsilon = 1.;
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..6 {
            gas.step().await.unwrap();
        }
        let archive = gas.recording().unwrap();
        let result = lecture_fractal::analyze("V-10", &parameters, archive).unwrap();
        let records = result.details["interface_records"].as_array().unwrap();
        let mut changed = false;
        for row in records {
            for (edges, slots) in [
                ("before_edges", "before_slots"),
                ("post_clone_edges", "post_clone_slots"),
                ("final_edges", "final_slots"),
            ] {
                let selected = row[slots].as_array().unwrap();
                changed |= selected.len() != 32;
                for edge in row[edges].as_array().unwrap() {
                    for endpoint in edge.as_array().unwrap() {
                        assert!(selected.contains(endpoint), "{row}");
                    }
                }
            }
        }
        assert!(changed, "test did not exercise eligibility changes");
    });
}

#[test]
fn full_horizon_graph_measurements_keep_every_window_edge_and_source() {
    use algorithmic_gas::fractal_set::EventRef;
    use std::collections::{BTreeMap, BTreeSet};
    fn event_id(e: EventRef) -> String {
        format!(
            "{}:{}:{}:{}:{}",
            e.epoch, e.step, e.version, e.slot, e.generation
        )
    }
    futures_lite::future::block_on(async {
        let parameters = json!({"walkers":32});
        let config = lecture::gas_config("V-01", &parameters, 7).unwrap();
        let mut gas = config.build::<f64>().await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..96 {
            gas.step().await.unwrap();
        }
        let archive = gas.recording().unwrap();
        let graph = archive.graph();
        let result = lecture_fractal::analyze("V-01", &parameters, archive).unwrap();
        assert!(result.details.get("graph").is_none());
        let summary = &result.details["graph_summary"];
        assert_eq!(summary["nodes"], graph.nodes.len());
        assert_eq!(summary["edges"], graph.edges.len());
        assert_eq!(summary["triangles"], graph.triangles.len());
        assert_eq!(summary["recorded_steps"], 96);
        assert_eq!(summary["unresolved_sources"], 0);
        let scene = &result.details["scene"];
        let display = &scene["display"];
        assert_eq!(display["window_start_step"], 89);
        assert_eq!(display["window_end_step"], 96);
        let nodes = scene["nodes"].as_array().unwrap();
        let edges = scene["edges"].as_array().unwrap();
        let ids = nodes
            .iter()
            .map(|n| n["id"].as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>();
        let expected = graph
            .edges
            .iter()
            .filter(|e| e.target.epoch == 0 && (89..=96).contains(&e.target.step))
            .collect::<Vec<_>>();
        let mut expected_edges = BTreeMap::<(String, String, String), usize>::new();
        for edge in &expected {
            let kind = serde_json::to_value(edge.kind).unwrap();
            *expected_edges
                .entry((
                    event_id(edge.source),
                    event_id(edge.target),
                    kind.as_str().unwrap().into(),
                ))
                .or_default() += 1;
        }
        let mut actual_edges = BTreeMap::new();
        for edge in edges {
            let source = edge["source"].as_str().unwrap().to_owned();
            let target = edge["target"].as_str().unwrap().to_owned();
            assert!(ids.contains(&source) && ids.contains(&target));
            *actual_edges
                .entry((source, target, edge["layer"].as_str().unwrap().to_owned()))
                .or_default() += 1;
        }
        assert_eq!(actual_edges, expected_edges);
        assert!(expected.iter().any(|e| e.source.step < 89));
        assert_eq!(display["unpositioned_events"], 0);
        assert_eq!(display["displayed_edges"], expected.len());
        assert_eq!(display["omitted_edges"], graph.edges.len() - expected.len());
        assert_eq!(display["displayed_nodes"], nodes.len());
        assert_eq!(display["omitted_nodes"], graph.nodes.len() - nodes.len());
        // Presentation is bounded independently of the full96 measurement horizon.
        assert!(edges.len() < graph.edges.len() / 4);
        assert_eq!(result.plots[0].series[0].points.len(), 96);
        eprintln!(
            "full96 graph: {} nodes, {} edges; scene: {} nodes, {} edges; result bytes: {}",
            graph.nodes.len(),
            graph.edges.len(),
            nodes.len(),
            edges.len(),
            serde_json::to_vec(&result).unwrap().len()
        );
    });
}
