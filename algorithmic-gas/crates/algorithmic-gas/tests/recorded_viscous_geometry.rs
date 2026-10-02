//! The all-slot geometry instrument observes a viscous gas without feeding
//! geometric fields or random draws back into its numerical transition.
use algorithmic_gas::{
    AlgorithmicGas, ExecutionContext, GasBuilder, GasConfig, InputBatch, ObservationBatch,
    Population, Provenance, RewardBatch, TensorBatch,
    domain::{OperatorFuture, RewardSource},
    tessellation::{GeometryPipelineConfig, MetricKind, Projection, ZeroPotential, sites::NO_SITE},
    variants::viscous_euclidean::{observe_recorded_geometry, reference_viscosity},
};
use futures_lite::future::block_on;

fn population() -> Population<f64> {
    let x = vec![
        0., 0., 0., 0.4, 0., 0., 0., 0.5, 0., 0., 0., 0.6, 0.2, 0.3, 0.4,
    ];
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(5, 3, x).unwrap());
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(5, 3, vec![0.; 15]).unwrap(),
    );
    Population::new(observations).unwrap()
}

#[test]
fn the_recorded_spatial_graph_includes_dead_slots_and_preserves_observations() {
    let mut p = population();
    for status in &mut p.validity[1..] {
        status.terminated = true;
    }
    assert_eq!(p.eligible(false).iter().filter(|&&alive| alive).count(), 1);
    let before = p.clone();
    let pipeline = GeometryPipelineConfig {
        metric: MetricKind::Identity,
        ..Default::default()
    };
    let geometry = observe_recorded_geometry(&p.observations, &pipeline, None, 1000).unwrap();
    assert_eq!(geometry.dimension, 3);
    assert_eq!(geometry.axes, [0, 1, 2]);
    assert_eq!(geometry.graph().nodes(), 5);
    assert!(
        geometry
            .tessellation
            .sites
            .site_of_walker
            .iter()
            .all(|&site| site != NO_SITE)
    );
    assert!(geometry.graph().edges() > 0);
    assert_eq!(p, before);
}

#[test]
fn dropping_a_spatial_coordinate_is_rejected() {
    let p = population();
    let pipeline = GeometryPipelineConfig {
        projection: Projection::DropLast { min_ambient: 3 },
        ..Default::default()
    };
    assert!(observe_recorded_geometry(&p.observations, &pipeline, None, 1000).is_err());
}

#[test]
fn the_observer_preserves_edge_budget_errors_and_the_input() {
    let p = population();
    let before = p.clone();
    let result = observe_recorded_geometry(
        &p.observations,
        &GeometryPipelineConfig::default(),
        None,
        0,
    );
    assert!(result.unwrap_err().to_string().contains("above the budget"));
    assert_eq!(p, before);
}

struct ZeroReward;
impl RewardSource<f64> for ZeroReward {
    fn id(&self) -> String {
        "recorded-geometry-zero/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<f64>> {
        Box::pin(async move {
            Ok(RewardBatch::new(
                vec![0.; p.len()],
                Provenance {
                    population_version: p.version,
                    stage: stage.into(),
                    ..Default::default()
                },
            ))
        })
    }
}

fn gas() -> AlgorithmicGas<f64> {
    let mut config = GasConfig::viscous_euclidean(3, 0.04, reference_viscosity()).unwrap();
    config.seed = 20261002;
    block_on(
        GasBuilder::new(population(), ZeroReward)
            .gradient(ZeroPotential::new("velocities"))
            .config(config)
            .build(),
    )
    .unwrap()
}

#[test]
fn passive_geometry_preserves_seeded_steps_and_checkpoint_bytes() {
    let mut observed = gas();
    let mut reference = gas();
    let pipeline = GeometryPipelineConfig::default();
    for _ in 0..8 {
        let before = observed.checkpoint().to_bytes().unwrap();
        let geometry =
            observe_recorded_geometry(&observed.population().observations, &pipeline, None, 1000)
                .unwrap();
        assert_eq!(geometry.dimension, 3);
        assert_eq!(observed.checkpoint().to_bytes().unwrap(), before);
        block_on(observed.step()).unwrap();
        block_on(reference.step()).unwrap();
        assert_eq!(observed.population(), reference.population());
        assert_eq!(
            observed.checkpoint().to_bytes().unwrap(),
            reference.checkpoint().to_bytes().unwrap()
        );
    }
}
