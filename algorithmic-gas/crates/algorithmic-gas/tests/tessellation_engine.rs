//! The geometry stage inside the engine: the Einstein-Hilbert gas end to end.
use algorithmic_gas::{
    AlgorithmicGas, GasBuilder, GasConfig, ObservationBatch, Population, Real, TensorBatch,
    noise::Noise,
    tessellation::{
        GeometryReward, GeometrySchedule, ZeroPotential,
        stage::{VOLUME_FIELD, curvature_field},
    },
};
use futures_lite::future::block_on;

fn population<T: Real>(n: usize, spread: f64) -> Population<T> {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let x: Vec<T> = (0..n * 3)
        .map(|k| T::from_f64(spread * ((mix(k as u64) >> 11) as f64 / (1u64 << 53) as f64 - 0.5)))
        .collect();
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, 3, x).unwrap());
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, 3, vec![T::ZERO; n * 3]).unwrap(),
    );
    Population::new(obs).unwrap()
}
/// Einstein-Hilbert gas with a cloning period and geometry schedule.
fn eh_config<T: Real>(seed: u64, clone_every: u64, schedule: GeometrySchedule) -> GasConfig {
    let mut config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    config.precision = T::PRECISION;
    config.seed = seed;
    config.clone_decision.every = clone_every;
    config.geometry.as_mut().unwrap().schedule = schedule;
    config
}
fn build<T: Real>(config: GasConfig, p: Population<T>) -> AlgorithmicGas<T> {
    block_on(
        GasBuilder::new(p, GeometryReward::default())
            .gradient(ZeroPotential::new("velocities"))
            .config(config)
            .build(),
    )
    .unwrap()
}
fn bits<T: Real>(p: &Population<T>) -> Vec<(String, Vec<u64>)> {
    p.observations
        .fields
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.values().iter().map(|x| x.to_f64().to_bits()).collect(),
            )
        })
        .collect()
}

#[test]
fn coincident_start_runs_the_canonical_gas() {
    // Every walker at the origin and at rest, single precision, odd count.
    let n = 41;
    let mut gas = build::<f32>(
        eh_config::<f32>(7, 5, GeometrySchedule::EveryStage),
        population(n, 0.),
    );
    // Coincident walkers: complete graph, flat, g = 1e5 I on the two tessellated axes.
    assert_eq!(gas.graph().unwrap().graph.edges(), n * (n - 1));
    let field = |gas: &AlgorithmicGas<f32>, name: &str| {
        gas.population()
            .observations
            .field(name)
            .unwrap()
            .values()
            .to_vec()
    };
    assert!(
        field(&gas, &curvature_field("ricci_scalar"))
            .iter()
            .all(|&r| r == 0.)
    );
    assert!(
        field(&gas, VOLUME_FIELD)
            .iter()
            .all(|&v| (v - 1e5).abs() < 1.)
    );
    let mut clones = 0;
    for step in 1..=30u64 {
        let report = block_on(gas.step()).unwrap();
        if step % 5 != 0 {
            assert_eq!(report.clones, 0, "step {step} cloned outside its period");
        }
        clones += report.clones;
        let graph = &gas.graph().unwrap().graph;
        graph.validate().unwrap();
        assert!(graph.edges() > 0);
        // The committed graph belongs to the committed positions, which the
        // first kinetic update already separates: a planar triangulation.
        {
            assert!(
                graph.edges() <= 2 * (3 * n - 6),
                "step {step}: {}",
                graph.edges()
            );
        }
        // The committed reward is the Einstein-Hilbert density of the fields.
        let r = field(&gas, &curvature_field("ricci_scalar"));
        let v = field(&gas, VOLUME_FIELD);
        for i in 0..n {
            assert_eq!(report.final_rewards.raw[i], r[i] * v[i]);
        }
    }
    assert!(clones > 0, "no walker cloned in 30 steps");
    assert!(field(&gas, "positions").iter().all(|x| x.is_finite()));
    let diffusion = gas
        .population()
        .observations
        .field("geometry.diffusion")
        .unwrap();
    assert_eq!(diffusion.item_shape(), &[3, 3]);
    // The time axis is outside the tessellation: unit diffusion there.
    assert!(
        diffusion
            .values()
            .chunks_exact(9)
            .all(|m| m[8] == 1. && m[2] == 0. && m[6] == 0.)
    );
}

#[test]
fn checkpoints_resume_bit_for_bit() {
    for schedule in [
        GeometrySchedule::EveryStage,
        GeometrySchedule::PostClone {
            every: 1,
            on_clone: true,
        },
        GeometrySchedule::PostClone {
            every: 3,
            on_clone: true,
        },
    ] {
        // Inelastic collisions and position jitter exercise every clone stream.
        let mut config = eh_config::<f64>(11, 2, schedule);
        config.clone_transform.position_field = Some("positions".into());
        config.clone_transform.jitter = Some(Noise::default());
        config.clone_transform.jitter_amplitude = 0.01;
        config.clone_transform.restitution = Some(0.5);
        let mut reference = build::<f64>(config.clone(), population(30, 1.));
        let mut checkpoint = None;
        for step in 1..=10 {
            block_on(reference.step()).unwrap();
            if step == 5 {
                let bytes = reference.checkpoint().to_bytes().unwrap();
                checkpoint = Some(algorithmic_gas::Checkpoint::<f64>::from_bytes(&bytes).unwrap());
            }
        }
        let mut resumed = build::<f64>(config, population(30, 1.));
        resumed.restore(checkpoint.unwrap()).unwrap();
        for _ in 6..=10 {
            block_on(resumed.step()).unwrap();
        }
        assert_eq!(
            bits(reference.population()),
            bits(resumed.population()),
            "{schedule:?}"
        );
        assert_eq!(reference.graph(), resumed.graph());
    }
}

#[test]
fn post_clone_schedule_carries_geometry_with_the_clones() {
    // No refresh after the scheduled one at step 1: fields change only by cloning.
    let schedule = GeometrySchedule::PostClone {
        every: 1_000_000,
        on_clone: false,
    };
    let mut gas = build::<f64>(eh_config::<f64>(3, 1, schedule), population(40, 1.));
    let name = curvature_field("ricci_scalar");
    let mut inherited = 0;
    block_on(gas.step()).unwrap();
    for _ in 0..4 {
        let before = gas
            .population()
            .observations
            .field(&name)
            .unwrap()
            .values()
            .to_vec();
        let report = block_on(gas.step()).unwrap();
        let after = gas.population().observations.field(&name).unwrap().values();
        for (i, choice) in report.clone_plan.choices.iter().enumerate() {
            if choice.accepted {
                let donor = report.clone_plan.sources[choice.donors[0].pool_index as usize].slot;
                assert_eq!(after[i], before[donor as usize]);
                inherited += 1;
            } else {
                assert_eq!(after[i], before[i]);
            }
        }
    }
    assert!(inherited > 0);
}

#[test]
fn recorded_steps_carry_the_graph_of_their_forces() {
    use algorithmic_gas::{RecordingConfig, RunArchive};
    let mut gas = build::<f64>(
        eh_config::<f64>(5, 2, GeometrySchedule::EveryStage),
        population(25, 1.),
    );
    gas.start_recording(RecordingConfig {
        graph: true,
        ..RecordingConfig::default()
    })
    .unwrap();
    for _ in 0..4 {
        block_on(gas.step()).unwrap();
    }
    let archive = gas.stop_recording().unwrap();
    let decoded = RunArchive::<f64>::from_bytes(&archive.to_bytes().unwrap()).unwrap();
    assert_eq!(decoded, archive);
    for step in &archive.steps {
        let graph = step.graph.as_ref().unwrap();
        graph.validate(25).unwrap();
        assert!(graph.weights.contains_key("riemannian_kernel_volume"));
        // Per-walker geometry and the recorded graph forces are in the archive.
        assert!(
            step.final_population
                .observations
                .fields
                .contains_key(VOLUME_FIELD)
        );
        for name in ["viscous_force", "curl_field", "boris_rotation_angle"] {
            assert!(
                step.field_evaluations.iter().any(|f| f.field == name),
                "{name} missing from the recorded field evaluations"
            );
        }
    }
    // Without the option the archive stores no graphs.
    let mut gas = build::<f64>(
        eh_config::<f64>(5, 2, GeometrySchedule::EveryStage),
        population(25, 1.),
    );
    gas.start_recording(RecordingConfig::default()).unwrap();
    block_on(gas.step()).unwrap();
    assert!(gas.stop_recording().unwrap().steps[0].graph.is_none());
}

#[test]
fn geometry_budget_and_missing_stage_are_reported() {
    // 60 coincident walkers need 1770 clique edges.
    let mut config = eh_config::<f64>(7, 20, GeometrySchedule::EveryStage);
    config.max_batch_elements = 2000;
    let error = block_on(
        GasBuilder::new(population::<f64>(60, 0.), GeometryReward::default())
            .gradient(ZeroPotential::new("velocities"))
            .config(config)
            .build(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("neighbor edges"), "{error}");
    // Graph viscosity without a geometry stage is a configuration error.
    let mut config = eh_config::<f64>(7, 20, GeometrySchedule::EveryStage);
    config.geometry = None;
    assert!(
        block_on(
            GasBuilder::new(population::<f64>(10, 1.), GeometryReward::default())
                .gradient(ZeroPotential::new("velocities"))
                .config(config)
                .build()
        )
        .is_err()
    );
}
