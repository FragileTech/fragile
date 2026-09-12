use algorithmic_gas::{fractal_set::EdgeKind, *};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel};
use futures_lite::future::block_on;
fn population() -> Population<f64> {
    Population::new(ObservationBatch::positions(
        TensorBatch::vectors(4, 2, vec![1., 2., 3., 4., -1., 0.2, 0.1, -0.1]).unwrap(),
    ))
    .unwrap()
}
async fn gas() -> AlgorithmicGas<f64> {
    GasBuilder::new(
        population(),
        BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        },
    )
    .build()
    .await
    .unwrap()
}
#[test]
fn archive_is_observational_consecutive_and_checkpoint_durable() {
    block_on(async {
        let mut recorded = gas().await;
        let mut plain = gas().await;
        recorded
            .start_recording(RecordingConfig::default())
            .unwrap();
        for _ in 0..16 {
            assert_eq!(recorded.step().await.unwrap(), plain.step().await.unwrap());
            assert_eq!(recorded.population(), plain.population());
        }
        let a = recorded.recording().unwrap();
        a.validate().unwrap();
        assert_eq!(a.steps.len(), 16);
        assert_eq!(a.anchors[0].step, 0);
        for (i, s) in a.steps.iter().enumerate() {
            assert_eq!(s.report.step, i as u64 + 1);
            for stage in [
                "pre_clone",
                "literal_clone",
                "post_transform",
                "post_clone",
                "post_kinetic",
            ] {
                assert!(s.stages.iter().any(|v| v.stage == stage));
            }
        }
        let graph = a.graph();
        assert!(graph.unresolved_sources.is_empty());
        assert_eq!(
            graph
                .edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Cst)
                .count(),
            64
        );
        assert!(
            graph
                .edges
                .iter()
                .filter(|e| e.kind == EdgeKind::Cst)
                .all(|e| e.target.step > e.source.step && e.target.slot == e.source.slot)
        );
        let initial = graph
            .nodes
            .iter()
            .find(|n| n.step == 0 && n.slot == 0)
            .copied()
            .unwrap();
        assert_eq!(graph.causal_future(initial).len(), 16);
        let checkpoint =
            Checkpoint::from_bytes(&recorded.checkpoint().to_bytes().unwrap()).unwrap();
        let mut restored = gas().await;
        restored.restore(checkpoint).unwrap();
        assert_eq!(restored.recording(), recorded.recording());
        restored.step().await.unwrap();
        recorded.step().await.unwrap();
        assert_eq!(restored.recording(), recorded.recording());
    });
}
#[test]
fn recording_limit_cancellation_and_replacement_are_atomic() {
    block_on(async {
        let mut g = gas().await;
        g.start_recording(RecordingConfig {
            max_steps: 1,
            ..Default::default()
        })
        .unwrap();
        g.step().await.unwrap();
        let saved = g.checkpoint().to_bytes().unwrap();
        assert!(g.step().await.is_err());
        assert_eq!(saved, g.checkpoint().to_bytes().unwrap());
        g.stop_recording();
        g.start_recording(RecordingConfig::default()).unwrap();
        let before = g.checkpoint().to_bytes().unwrap();
        g.cancellation_token().cancel();
        assert!(g.step().await.is_err());
        assert_eq!(before, g.checkpoint().to_bytes().unwrap());
        g.cancellation_token().reset();
        g.replace_population(population()).await.unwrap();
        g.step().await.unwrap();
        let a = g.recording().unwrap();
        a.validate().unwrap();
        assert_eq!(a.epoch, 1);
        assert_eq!(a.anchors.len(), 2);
        assert!(
            a.graph()
                .edges
                .iter()
                .all(|e| e.source.epoch == e.target.epoch)
        );
    });
}
#[test]
fn v2_migration_records_only_restored_coverage() {
    block_on(async {
        let mut g = gas().await;
        g.step().await.unwrap();
        let mut bytes = g.checkpoint().to_bytes().unwrap();
        let key = b"schema_version";
        let position = bytes.windows(key.len()).position(|w| w == key).unwrap() + key.len();
        assert_eq!(bytes[position], 3);
        bytes[position] = 2;
        let migrated: Checkpoint<f64> = Checkpoint::from_bytes(&bytes).unwrap();
        assert_eq!(migrated.schema_version, 3);
        let a = migrated.recording.unwrap();
        assert!(a.steps.is_empty());
        assert_eq!(a.anchors[0].step, 1);
        assert_eq!(a.anchors[0].reason, "checkpoint_v2_migration");
    });
}

#[test]
fn historical_sources_resolve_by_full_identity_and_coverage() {
    block_on(async {
        let mut c = GasConfig {
            precision: Precision::F64,
            ..Default::default()
        };
        c.distance_donors.history_window = 2;
        c.cloning_donors.history_window = 2;
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        };
        let mut g = GasBuilder::new(population(), model)
            .config(c)
            .build()
            .await
            .unwrap();
        g.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..6 {
            g.step().await.unwrap();
        }
        let a = g.recording().unwrap();
        a.validate().unwrap();
        let graph = a.graph();
        assert!(graph.unresolved_sources.is_empty());
        assert!(graph.edges.iter().any(|e| matches!(
            e.kind,
            EdgeKind::HistoricalDistance | EdgeKind::HistoricalCloning
        )));
        assert!(graph.boundary_squared_zero());
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        assert!(!g.recording().unwrap().graph().unresolved_sources.is_empty());
    });
}
#[test]
fn revival_has_actual_ancestry_and_information_source_without_cst() {
    block_on(async {
        let mut p = population();
        p.validity[0].terminated = true;
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        };
        let mut g = GasBuilder::new(p, model).build().await.unwrap();
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        let a = g.recording().unwrap();
        let choice = &a.steps[0].report.clone_plan.choices[0];
        assert!(choice.revival && choice.accepted);
        assert_eq!(choice.probability, Some(1.));
        let graph = a.graph();
        let ancestry = graph
            .edges
            .iter()
            .find(|e| e.kind == EdgeKind::Ancestry && e.target.slot == 0)
            .unwrap();
        let information = graph
            .edges
            .iter()
            .find(|e| e.kind == EdgeKind::IaRevival && e.source.slot == 0)
            .unwrap();
        assert_eq!(ancestry.source, information.target);
        assert_eq!(ancestry.target, information.source);
        assert!(
            !graph
                .edges
                .iter()
                .any(|e| e.kind == EdgeKind::Cst && e.source.slot == 0)
        );
        assert!(!graph.triangles.iter().any(|t| t.vertices[0].slot == 0));
    });
}
#[test]
fn scalar_reconstruction_rejects_missing_and_duplicate_components() {
    block_on(async {
        use algorithmic_gas::tracking::ScalarStageEncoding;
        let mut g = gas().await;
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        let a = g.recording().unwrap();
        let r = a.reconstruct(0, 1, "post_kinetic").unwrap();
        assert_eq!(r.max_absolute_residual, 0.);
        assert_eq!(r.scalar_count, 8);
        assert!(a.reconstruct(0, 0, "post_kinetic").is_err());
        let stage = a.steps[0].stages.first().unwrap();
        let mut codec = ScalarStageEncoding::encode(stage);
        codec.components.pop();
        assert!(codec.decode().is_err());
        let mut codec = ScalarStageEncoding::encode(stage);
        codec.components.push(codec.components[0].clone());
        assert!(codec.decode().is_err());
    });
}
#[test]
fn archive_capacity_failure_after_execution_rolls_back_and_invalid_archives_fail() {
    block_on(async {
        let mut g = gas().await;
        g.start_recording(RecordingConfig::default()).unwrap();
        let anchor_bytes = g.recording().unwrap().buffer_bytes().unwrap();
        g.start_recording(RecordingConfig {
            max_bytes: anchor_bytes + 1000,
            ..Default::default()
        })
        .unwrap();
        let before = g.checkpoint().to_bytes().unwrap();
        assert!(g.step().await.is_err());
        assert_eq!(before, g.checkpoint().to_bytes().unwrap());
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        let mut archive = g.recording().unwrap().clone();
        archive.steps[0].stages[0]
            .fields
            .get_mut("positions")
            .unwrap()
            .values[0] += 1.;
        assert!(archive.validate().is_err());
        let mut checkpoint = g.checkpoint();
        checkpoint
            .population
            .observations
            .field_mut("positions")
            .unwrap()
            .replace_row(0, &[9., 9.])
            .unwrap();
        assert!(checkpoint.validate().is_err());
        let mut archive = g.recording().unwrap().clone();
        archive.steps[0].noise[0].sample.pop();
        assert!(archive.validate().is_err());
    });
}

#[test]
fn baoab_records_actual_force_noise_and_operator_stages() {
    block_on(async {
        let mut p = population();
        p.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(4, 2, vec![0.; 8]).unwrap(),
        );
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        };
        let mut c = GasConfig {
            precision: Precision::F64,
            ..Default::default()
        };
        c.kinetic.integrator = algorithmic_gas::kinetic::KineticKind::Baoab {
            positions: "positions".into(),
            velocities: "velocities".into(),
            dt: 0.02,
            friction: 1.,
        };
        let mut g = GasBuilder::new(p, model.clone())
            .gradient(model)
            .config(c)
            .build()
            .await
            .unwrap();
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        let a = g.recording().unwrap();
        a.validate().unwrap();
        let s = &a.steps[0];
        for name in ["B1", "A1", "O", "A2", "B2"] {
            assert!(s.stages.iter().any(|v| v.stage == name));
        }
        let forces = s
            .field_evaluations
            .iter()
            .filter(|f| f.field == "potential_gradient")
            .collect::<Vec<_>>();
        assert_eq!(forces.len(), 2);
        assert!(forces[1].version > forces[0].version);
        let raw = s
            .noise
            .iter()
            .find(|n| {
                n.stage == "raw_noise" && n.stream == algorithmic_gas::random::Stream::Kinetic
            })
            .unwrap();
        assert_eq!(raw.raw_innovation.as_ref().unwrap(), &raw.sample);
        assert_eq!(raw.factor.as_ref().unwrap(), &vec![1.; 4]);
        let graph = a.graph();
        assert!(graph.boundary_squared_zero());
        let mut supports = std::collections::BTreeSet::new();
        for t in &graph.triangles {
            assert!(supports.insert(t.vertices));
            assert!(!t.channels.is_empty());
        }
        for edge in graph.edges.iter().filter(|e| e.kind == EdgeKind::Cst) {
            let [x, y] = edge.attributes.position_displacement.unwrap();
            let [u, v] = edge.attributes.spin2_displacement.unwrap();
            assert!((u * u - v * v - x).abs() < 1e-12);
            assert!((2. * u * v - y).abs() < 1e-12);
        }
        assert_eq!(RunArchive::from_bytes(&a.to_bytes().unwrap()).unwrap(), *a);
    });
}

#[test]
fn light_cone_comparison_uses_physical_time_and_separate_cst_order() {
    block_on(async {
        let mut g = gas().await;
        g.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..4 {
            g.step().await.unwrap();
        }
        let a = g.recording().unwrap();
        let first = a.compare_orders(1., 0.04, 20).unwrap();
        let rescaled = a.compare_orders(2., 0.02, 20).unwrap();
        assert_eq!(first.cst_pairs, rescaled.cst_pairs);
        assert_eq!(first.lorentz_pairs, rescaled.lorentz_pairs);
        let large = a.compare_orders(1e6, 0.04, 20).unwrap();
        assert_eq!(first.cst_pairs, large.cst_pairs);
        assert!(large.lorentz_pairs.len() > first.lorentz_pairs.len());
        assert!(
            first
                .cst_pairs
                .iter()
                .all(|[i, j]| first.nodes[*i].event.slot == first.nodes[*j].event.slot)
        );
        assert_eq!(first.both + first.cst_only, first.cst_pairs.len());
        assert_eq!(first.both + first.lorentz_only, first.lorentz_pairs.len());
        assert!(a.compare_orders(1., 0.04, 513).is_err());
    });
}

#[test]
fn restitution_records_field_influence_even_for_noncloning_partner() {
    block_on(async {
        let mut p = population();
        p.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(4, 2, vec![1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
        );
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        };
        let mut c = GasConfig {
            precision: Precision::F64,
            ..Default::default()
        };
        c.cloning_donors.law = algorithmic_gas::donor::SamplingLaw::FisherYates;
        c.cloning_donors.kernel = algorithmic_gas::geometry::Kernel::Uniform;
        c.clone_decision.saturation = 0.001;
        c.clone_transform.velocity_field = Some("velocities".into());
        c.clone_transform.restitution = Some(0.25);
        let mut g = GasBuilder::new(p, model).config(c).build().await.unwrap();
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        let a = g.recording().unwrap();
        a.validate().unwrap();
        let s = &a.steps[0];
        assert!(!s.influences.is_empty());
        assert!(
            s.influences
                .iter()
                .any(|w| !s.report.clone_plan.choices[w.recipient as usize].accepted)
        );
        let transformed = s
            .stages
            .iter()
            .find(|v| v.stage == "post_transform")
            .unwrap();
        for recipient in 0..4 {
            let weights = s
                .influences
                .iter()
                .filter(|w| w.recipient as usize == recipient)
                .collect::<Vec<_>>();
            if weights.is_empty() {
                continue;
            }
            assert_eq!(weights.iter().map(|w| w.weight).sum::<f64>(), 1.);
            for component in 0..2 {
                let expected = weights
                    .iter()
                    .map(|w| {
                        w.weight
                            * s.before
                                .observations
                                .field("velocities")
                                .unwrap()
                                .row(w.source.slot as usize)
                                .unwrap()[component]
                    })
                    .sum::<f64>();
                assert!(
                    (expected - transformed.fields["velocities"].values[recipient * 2 + component])
                        .abs()
                        < 1e-12
                );
            }
        }
        assert_eq!(
            a.graph()
                .edges
                .iter()
                .filter(|e| e.kind == EdgeKind::IaTransform)
                .count(),
            s.influences.len()
        );
    });
}

#[test]
fn archive_rejects_missing_operator_stages_and_fabricated_recorded_sources() {
    block_on(async {
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        };
        let mut config = GasConfig {
            precision: Precision::F64,
            ..Default::default()
        };
        config.distance_donors.history_window = 2;
        let mut g = GasBuilder::new(population(), model)
            .config(config)
            .build()
            .await
            .unwrap();
        g.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..4 {
            g.step().await.unwrap();
        }
        let original = g.recording().unwrap();
        original.validate().unwrap();
        let mut archive = original.clone();
        archive.steps[0]
            .stages
            .retain(|stage| stage.stage != "post_clone");
        assert!(archive.validate().is_err());
        let mut archive = original.clone();
        let duplicate = archive.steps[0].stages[0].clone();
        archive.steps[0].stages.insert(1, duplicate);
        assert!(archive.validate().is_err());
        let mut archive = original.clone();
        archive.steps[0].stages[1].generations[0] += 1;
        assert!(archive.validate().is_err());
        let mut archive = original.clone();
        let source = archive.steps[3]
            .report
            .distance_sources
            .iter_mut()
            .find(|source| source.frame == 1)
            .unwrap();
        source.version = 0;
        assert!(archive.validate().is_err());
        let mut archive = original.clone();
        archive.steps[0].donor_fitness[0] = f64::NAN;
        assert!(archive.validate().is_err());
    });
}

#[test]
fn interaction_displacements_decode_the_actual_current_or_historical_source() {
    block_on(async {
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        };
        let mut config = GasConfig {
            precision: Precision::F64,
            ..Default::default()
        };
        config.distance_donors.history_window = 2;
        let mut g = GasBuilder::new(population(), model)
            .config(config)
            .build()
            .await
            .unwrap();
        g.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..6 {
            g.step().await.unwrap();
        }
        let archive = g.recording().unwrap();
        let graph = archive.graph();
        let mut historical = 0;
        for edge in graph.edges.iter().filter(|edge| {
            matches!(
                edge.kind,
                EdgeKind::IgDistance
                    | EdgeKind::IgCloning
                    | EdgeKind::HistoricalDistance
                    | EdgeKind::HistoricalCloning
            )
        }) {
            let receiver = &archive.steps[edge.source.step as usize].before;
            let donor = &archive.steps[edge.target.step as usize].before;
            let receiver_x = receiver
                .observations
                .field("positions")
                .unwrap()
                .row(edge.source.slot as usize)
                .unwrap();
            let donor_x = donor
                .observations
                .field("positions")
                .unwrap()
                .row(edge.target.slot as usize)
                .unwrap();
            let expected = [donor_x[0] - receiver_x[0], donor_x[1] - receiver_x[1]];
            assert_eq!(edge.attributes.position_displacement, Some(expected));
            let [u, v] = edge.attributes.spin2_displacement.unwrap();
            assert!((u * u - v * v - expected[0]).abs() < 1e-12);
            assert!((2. * u * v - expected[1]).abs() < 1e-12);
            historical += usize::from(edge.source.step != edge.target.step);
        }
        assert!(historical > 0);
        assert!(
            graph
                .edges
                .iter()
                .filter(|edge| matches!(edge.kind, EdgeKind::IaDistance | EdgeKind::IaCloning))
                .all(|edge| edge.attributes.position_displacement.is_none())
        );
    });
}

#[test]
fn archive_rejects_conflicting_anchor_and_adjacent_boundary_coordinates() {
    block_on(async {
        let mut g = gas().await;
        g.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..2 {
            g.step().await.unwrap();
        }
        let original = g.recording().unwrap();
        let mut archive = original.clone();
        archive.anchors[0]
            .population
            .observations
            .field_mut("positions")
            .unwrap()
            .replace_row(0, &[9., 9.])
            .unwrap();
        assert!(
            archive
                .validate()
                .unwrap_err()
                .to_string()
                .contains("archive boundary")
        );

        let mut archive = original.clone();
        archive.steps[0]
            .final_population
            .observations
            .field_mut("positions")
            .unwrap()
            .replace_row(0, &[9., 9.])
            .unwrap();
        let last_stage = archive.steps[0].stages.last_mut().unwrap();
        last_stage.fields.get_mut("positions").unwrap().values[..2].copy_from_slice(&[9., 9.]);
        // The step still agrees with its own post_kinetic stage; the next
        // pre_clone record contradicts the same event's coordinates.
        assert!(
            archive
                .validate()
                .unwrap_err()
                .to_string()
                .contains("archive boundary")
        );

        let mut archive = original.clone();
        let field = archive.anchors[0]
            .population
            .observations
            .fields
            .remove("positions")
            .unwrap();
        archive.anchors[0]
            .population
            .observations
            .fields
            .insert("renamed".into(), field);
        assert!(
            archive
                .validate()
                .unwrap_err()
                .to_string()
                .contains("archive boundary")
        );

        let mut archive = original.clone();
        archive.anchors[0].population.generations[0] += 1;
        assert!(
            archive
                .validate()
                .unwrap_err()
                .to_string()
                .contains("archive boundary")
        );
    });
}

#[test]
fn archive_boundary_accepts_explicit_extraction_version_and_recorded_nan() {
    block_on(async {
        use algorithmic_gas::{
            domain::{NamedObservationExtractor, NamedRewardExtractor},
            extraction::ExtractionPipeline,
        };
        use std::collections::BTreeMap;
        let mut g = gas().await;
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        let pipeline = ExtractionPipeline {
            observations: Box::new(NamedObservationExtractor {
                fields: vec![("features".into(), "positions".into())],
            }),
            rewards: Box::new(NamedRewardExtractor {
                field: "objective".into(),
            }),
            derived: None,
        };
        let input = InputBatch {
            rows: 4,
            version: 12,
            numerical: BTreeMap::from([
                (
                    "features".into(),
                    TensorBatch::vectors(4, 2, vec![2., 2., 2., 2., 2., 2., 2., 2.]).unwrap(),
                ),
                (
                    "objective".into(),
                    TensorBatch::scalars(vec![8.; 4]).unwrap(),
                ),
            ]),
            bytes: BTreeMap::new(),
        };
        g.step_with_extraction(&input, &pipeline).await.unwrap();
        g.recording().unwrap().validate().unwrap();
        let archive = g.recording().unwrap();
        assert!(archive.steps[1].before.version > archive.steps[0].final_population.version);

        let mut p = population();
        p.observations
            .field_mut("positions")
            .unwrap()
            .replace_row(0, &[f64::NAN, 0.])
            .unwrap();
        p.validity[0].terminated = true;
        let model = BenchmarkModel {
            benchmark: Benchmark::Sphere,
            field: "positions".into(),
            direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
        };
        let mut g = GasBuilder::new(p, model).build().await.unwrap();
        g.start_recording(RecordingConfig::default()).unwrap();
        g.step().await.unwrap();
        g.recording().unwrap().validate().unwrap();
    });
}
