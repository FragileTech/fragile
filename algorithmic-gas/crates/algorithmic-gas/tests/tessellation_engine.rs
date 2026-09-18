//! The geometry stage inside the engine: the Einstein-Hilbert gas end to end.
use algorithmic_gas::{
    AlgorithmicGas, GasBuilder, ObservationBatch, Population, Precision, Real, TensorBatch,
    tessellation::{
        EinsteinHilbertGas, GeometryTiming, ZeroPotential,
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
fn build<T: Real>(preset: &EinsteinHilbertGas, p: Population<T>, seed: u64) -> AlgorithmicGas<T> {
    block_on(
        GasBuilder::new(p, preset.reward())
            .gradient(ZeroPotential::new("velocities"))
            .config(preset.config(T::PRECISION, seed))
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
    let preset = EinsteinHilbertGas {
        clone_every: 5,
        ..EinsteinHilbertGas::default()
    };
    assert_eq!(preset.config(Precision::F32, 7).precision, Precision::F32);
    let mut gas = build::<f32>(&preset, population(n, 0.), 7);
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
        // The committed graph is the post-cloning, pre-kinetic one: step 1 still
        // sees coincident walkers, later steps a planar triangulation.
        if report.clones == 0 && step > 1 {
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
    for (refresh_every, timing) in [
        (1, GeometryTiming::AfterCloning),
        (3, GeometryTiming::AfterCloning),
        (1, GeometryTiming::Both),
    ] {
        let preset = EinsteinHilbertGas {
            clone_every: 2,
            refresh_every,
            timing,
            sigma_x: 0.01,
            restitution: 0.5,
            ..EinsteinHilbertGas::default()
        };
        let mut reference = build::<f64>(&preset, population(30, 1.), 11);
        let mut checkpoint = None;
        for step in 1..=10 {
            block_on(reference.step()).unwrap();
            if step == 5 {
                let bytes = reference.checkpoint().to_bytes().unwrap();
                checkpoint = Some(algorithmic_gas::Checkpoint::<f64>::from_bytes(&bytes).unwrap());
            }
        }
        let mut resumed = build::<f64>(&preset, population(30, 1.), 11);
        resumed.restore(checkpoint.unwrap()).unwrap();
        for _ in 6..=10 {
            block_on(resumed.step()).unwrap();
        }
        assert_eq!(
            bits(reference.population()),
            bits(resumed.population()),
            "refresh_every={refresh_every} timing={timing:?}"
        );
        assert_eq!(reference.graph(), resumed.graph());
    }
}

#[test]
fn clones_inherit_the_geometry_of_their_donor() {
    // No refresh after the initial one: fields change only by cloning.
    let mut preset = EinsteinHilbertGas {
        clone_every: 1,
        refresh_every: 1_000_000,
        ..EinsteinHilbertGas::default()
    };
    let mut config = preset.config(Precision::F64, 3);
    config.geometry.as_mut().unwrap().refresh_on_clone = false;
    preset.refresh_every = 1_000_000;
    let mut gas = block_on(
        GasBuilder::new(population::<f64>(40, 1.), preset.reward())
            .gradient(ZeroPotential::new("velocities"))
            .config(config)
            .build(),
    )
    .unwrap();
    let name = curvature_field("ricci_scalar");
    let mut inherited = 0;
    // Step 1 is a scheduled refresh: (step - 1) is a multiple of every period.
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
fn geometry_budget_and_missing_stage_are_reported() {
    let preset = EinsteinHilbertGas::default();
    // 60 coincident walkers need 1770 clique edges.
    let mut config = preset.config(Precision::F64, 7);
    config.max_batch_elements = 2000;
    let error = block_on(
        GasBuilder::new(population::<f64>(60, 0.), preset.reward())
            .gradient(ZeroPotential::new("velocities"))
            .config(config)
            .build(),
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("neighbor edges"), "{error}");
    // Graph viscosity without a geometry stage is a configuration error.
    let mut config = preset.config(Precision::F64, 7);
    config.geometry = None;
    assert!(
        block_on(
            GasBuilder::new(population::<f64>(10, 1.), preset.reward())
                .gradient(ZeroPotential::new("velocities"))
                .config(config)
                .build()
        )
        .is_err()
    );
}
