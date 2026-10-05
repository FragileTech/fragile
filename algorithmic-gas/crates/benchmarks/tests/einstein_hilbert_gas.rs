//! The Einstein-Hilbert gas through the shared run configuration and archive.
use algorithmic_gas::{GasConfig, Precision, RecordingConfig, RunArchive};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig};
use futures_lite::future::block_on;

fn small() -> RunConfig {
    let mut config = RunConfig::einstein_hilbert().unwrap();
    config.walkers = 48;
    config.gas.precision = Precision::F64;
    config.gas.clone_decision.every = 4;
    config
}

#[test]
fn run_configuration_builds_records_and_replays_the_gas() {
    let config = small();
    // The configuration is plain data: it survives a JSON round trip.
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(serde_json::from_str::<RunConfig>(&json).unwrap(), config);

    let run = |config: &RunConfig| {
        let mut gas = block_on(config.build::<f64>()).unwrap();
        gas.start_recording(RecordingConfig {
            graph: true,
            ..RecordingConfig::default()
        })
        .unwrap();
        let mut clones = 0;
        for _ in 0..12 {
            clones += block_on(gas.step()).unwrap().clones;
        }
        (gas.stop_recording().unwrap(), clones)
    };
    let (archive, clones) = run(&config);
    assert!(clones > 0);
    for step in &archive.steps {
        let velocities = |name: &str| {
            &step.stages.iter().find(|s| s.stage == name).unwrap().fields["velocities"].values
        };
        assert_eq!(velocities("pre_clone"), velocities("post_transform"));
        let graph = step.graph.as_ref().unwrap();
        let bounds = algorithmic_gas::variants::einstein_hilbert::graph_kick_bounds(
            &graph.graph,
            &graph.weights["riemannian_kernel_volume"],
            3.,
            0.002,
        )
        .unwrap();
        let energy = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let max_energy = |v: &[f64]| {
            v.chunks_exact(config.dimensions)
                .map(energy)
                .fold(0., f64::max)
        };
        for (before, after) in [("post_clone", "B1"), ("A2", "B2")] {
            let (input, output) = (velocities(before), velocities(after));
            assert!(max_energy(output) <= max_energy(input) + 1e-12);
            assert!(
                energy(output) <= bounds.quarter_kick_moment_factor.powi(2) * energy(input) + 1e-11
            );
        }
    }
    // Every walker starts at the origin at rest and the gas spreads them.
    let first = archive.steps[0]
        .before
        .observations
        .field("positions")
        .unwrap();
    assert!(first.values().iter().all(|&x| x == 0.));
    let last = archive.steps.last().unwrap();
    let x = last
        .final_population
        .observations
        .field("positions")
        .unwrap();
    assert!(x.values().iter().any(|&v| v != 0.) && x.values().iter().all(|v| v.is_finite()));
    assert!(last.graph.as_ref().unwrap().graph.edges() > 0);
    assert!(archive.providers["reward"].starts_with("geometry-reward/v1"));
    assert!(archive.providers["gradient"].starts_with("zero-potential/v1"));
    assert_eq!(
        RunArchive::<f64>::from_bytes(&archive.to_bytes().unwrap()).unwrap(),
        archive
    );
    // Same seed, same run.
    assert_eq!(run(&config).0, archive);
}

#[test]
fn geometry_reward_composes_with_a_potential_and_is_validated() {
    // A confining potential supplies the force; the reward stays geometric.
    let mut confined = small();
    confined.potential = Some(Benchmark::Sphere);
    confined.initial_lower = -1.;
    confined.initial_upper = 1.;
    let mut gas = block_on(confined.build::<f64>()).unwrap();
    for _ in 0..3 {
        block_on(gas.step()).unwrap();
    }
    assert!(gas.graph().is_some());

    let mut missing_stage = small();
    missing_stage.gas.geometry = None;
    assert!(missing_stage.validate().is_err());
    assert!(GasConfig::einstein_hilbert(0., 0.002).is_err());
}

#[test]
fn finite_coincident_inputs_can_exceed_the_geometry_budget() {
    let mut config = small();
    config.walkers = 11;
    config.gas.max_batch_elements = 99;
    // 11 * 3 * 3 fits this batch budget, but the lifted coincident clique
    // has 110 directed edges. Failure is an execution error, not killing.
    let error = block_on(config.build::<f64>())
        .err()
        .expect("geometry budget must fail");
    assert!(error.to_string().contains("neighbor edges"), "{error}");
}
