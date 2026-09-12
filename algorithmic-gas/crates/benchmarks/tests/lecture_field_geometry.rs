use algorithmic_gas::{
    Precision, RecordingConfig,
    physics::{fields, partvi::ExperimentRequest},
};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig, physics_metric::PhysicsMetricConfig};
use serde_json::json;
#[test]
fn executed_conditional_geometry_has_no_supplied_spacetime() {
    futures_lite::future::block_on(async {
        for dimension in [2, 3] {
            let mut config = RunConfig {
                benchmark: Benchmark::Quadratic,
                walkers: 8,
                dimensions: dimension,
                initial_lower: -0.5,
                initial_upper: 0.5,
                physics_metric: Some(PhysicsMetricConfig {
                    curvature: false,
                    epsilon: 1.,
                    ..Default::default()
                }),
                ..Default::default()
            };
            config.gas.precision = Precision::F64;
            config.gas.boundary = algorithmic_gas::boundary::BoundaryPolicy::Unbounded;
            config.gas.kinetic.integrator = algorithmic_gas::kinetic::KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.04,
                friction: 0.7,
            };
            config.gas.seed = 7;
            config.gas.distance_donors.history_window = 2;
            config.gas.cloning_donors.history_window = 2;
            let mut gas = config.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig::default()).unwrap();
            for _ in 0..5 {
                gas.step().await.unwrap();
            }
            let archive = gas.recording().unwrap();
            let before = archive.to_bytes().unwrap();
            for id in [37, 38, 40, 62] {
                let request = ExperimentRequest {
                    experiment: id,
                    parameters: json!({"transport_steps":2}),
                };
                let result = fields::analyze(&request, Some(archive))
                    .unwrap_or_else(|e| panic!("dimension {dimension} experiment{id}: {e}"));
                assert_eq!(result.details["source"], "recorded_engine_fields");
                assert_eq!(result.details["dimension"], dimension);
                assert!(
                    result
                        .plots
                        .iter()
                        .flat_map(|p| &p.series)
                        .flat_map(|s| &s.points)
                        .flatten()
                        .all(|x| x.is_finite())
                );
                if id == 40 {
                    let residual = result
                        .metrics
                        .iter()
                        .find(|m| m.label == "Metric compatibility residual")
                        .unwrap()
                        .value
                        .unwrap();
                    assert!(residual < 1e-9, "{result:?}");
                }
            }
            assert_eq!(before, archive.to_bytes().unwrap());
        }
    });
}
