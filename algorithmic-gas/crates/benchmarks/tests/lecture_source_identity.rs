use algorithmic_gas::{RecordingConfig, fractal_set::EdgeKind};
use algorithmic_gas_benchmarks::lecture::gas_config;
use serde_json::json;

#[test]
fn graph_resolves_material_force_sources_at_their_recorded_stage_versions() {
    futures_lite::future::block_on(async {
        for seed in [0, 7, 516] {
            let config = gas_config(
                "V-01",
                &json!({"walkers":8,"engine_memory":2,"engine_viscosity":0.15}),
                seed,
            )
            .unwrap();
            let mut gas = config.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig {
                max_steps: 8,
                max_bytes: 32 * 1024 * 1024,
            })
            .unwrap();
            for _ in 0..8 {
                gas.step().await.unwrap();
            }
            let archive = gas.recording().unwrap();
            let graph = archive.graph();
            assert!(
                graph.unresolved_sources.is_empty(),
                "{:?}",
                graph.unresolved_sources
            );
            let mut intermediates = 0;
            for step in &archive.steps {
                for influence in &step.influences {
                    let source = influence.source;
                    assert!(
                        graph
                            .edges
                            .iter()
                            .any(|edge| edge.kind == EdgeKind::IaTransform
                                && edge.target.epoch == step.epoch
                                && edge.target.step == source.frame
                                && edge.target.slot == source.slot
                                && edge.target.version == source.version
                                && edge.target.generation == source.generation)
                    );
                    if source.version != step.before.version
                        && source.version != step.final_population.version
                    {
                        intermediates += 1;
                    }
                }
            }
            assert!(intermediates > 0);
            assert!(graph.boundary_squared_zero());
        }
    });
}
