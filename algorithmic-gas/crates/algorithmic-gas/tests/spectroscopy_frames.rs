//! Frames extracted from recorded steps of real gases against the raw
//! records, closed-form kicks and tampered steps.
use algorithmic_gas::{
    AlgorithmicGas, ExecutionContext, GasBuilder, GasConfig, GasError, InputBatch,
    ObservationBatch, Population, Precision, Provenance, Real, RecordingConfig, RewardBatch,
    RunArchive, TensorBatch,
    domain::{GradientProvider, OperatorFuture, RewardSource},
    donor::{CompanionBatch, SourceRef},
    kinetic::{KineticKind, KineticOperator, QftExecutionConfig, ViscousForceConfig},
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry},
    physics::spectroscopy::{
        config::{ColorSource, MeasurementConfig, StepAlignment},
        contract::{
            Capabilities, CompanionMask, Companions, Frame, NO_COMPANION, Record, StageKick,
        },
        frame::{base_state, extract, position_field},
    },
    tessellation::{GeometryReward, GeometrySchedule, ZeroPotential},
    tracking::{FieldEvaluation, RecordedStep, StageSnapshot},
};
use futures_lite::future::block_on;

/// Deterministic cloud in `[-spread/2, spread/2]^d` with velocities in `[-speed/2, speed/2]^d`.
fn population<T: Real>(n: usize, d: usize, spread: f64, speed: f64) -> Population<T> {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let unit = |k: u64| (mix(k) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
    let x = (0..n * d)
        .map(|k| T::from_f64(spread * unit(k as u64)))
        .collect();
    let v = (0..n * d)
        .map(|k| T::from_f64(speed * unit(1_000_003 + k as u64)))
        .collect();
    let mut observations = ObservationBatch::positions(TensorBatch::vectors(n, d, x).unwrap());
    observations
        .fields
        .insert("velocities".into(), TensorBatch::vectors(n, d, v).unwrap());
    Population::new(observations).unwrap()
}
fn record<T: Real>(gas: &mut AlgorithmicGas<T>, steps: usize, graph: bool) -> RunArchive<T> {
    gas.start_recording(RecordingConfig {
        max_steps: steps,
        graph,
        ..RecordingConfig::default()
    })
    .unwrap();
    for _ in 0..steps {
        block_on(gas.step()).unwrap();
    }
    let archive = gas.stop_recording().unwrap();
    archive.validate().unwrap();
    archive
}
/// Einstein-Hilbert gas in three dimensions that clones on every second step
/// and refreshes its graph on every stage.
fn einstein_hilbert<T: Real>(walkers: usize, seed: u64) -> AlgorithmicGas<T> {
    let mut config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    config.precision = T::PRECISION;
    config.seed = seed;
    config.clone_decision.every = 2;
    config.geometry.as_mut().unwrap().schedule = GeometrySchedule::EveryStage;
    block_on(
        GasBuilder::new(
            population::<T>(walkers, 3, 1., 1.),
            GeometryReward::default(),
        )
        .gradient(ZeroPotential::new("velocities"))
        .config(config)
        .build(),
    )
    .unwrap()
}
/// `U = |x|² / 2` as reward and potential; the Euclidean Gas minimizes it.
struct Quadratic;
impl RewardSource<f64> for Quadratic {
    fn id(&self) -> String {
        "test-quadratic/v1".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<f64>> {
        Box::pin(async move {
            let x = p.observations.field("positions")?;
            Ok(RewardBatch::new(
                x.values()
                    .chunks_exact(x.width())
                    .map(|r| 0.5 * r.iter().map(|a| a * a).sum::<f64>())
                    .collect(),
                Provenance {
                    population_version: p.version,
                    stage: stage.into(),
                    ..Default::default()
                },
            ))
        })
    }
}
impl GradientProvider<f64> for Quadratic {
    fn id(&self) -> String {
        "test-quadratic-gradient/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            let x = p.observations.field("positions")?;
            let (d, eligible) = (x.width(), p.eligible(false));
            let mut gradient = x.values().to_vec();
            for (row, alive) in gradient.chunks_exact_mut(d).zip(eligible) {
                if !alive {
                    row.fill(0.);
                }
            }
            TensorBatch::vectors(p.len(), d, gradient)
        })
    }
}
fn euclidean(mut config: GasConfig, seed: u64, spread: f64) -> AlgorithmicGas<f64> {
    config.seed = seed;
    block_on(
        GasBuilder::new(population::<f64>(12, 3, spread, 1.), Quadratic)
            .gradient(Quadratic)
            .config(config)
            .build(),
    )
    .unwrap()
}
fn unit_viscosity() -> ViscousForceConfig {
    ViscousForceConfig {
        coefficient: 1.,
        bandwidth: 1.,
        row_normalized: false,
    }
}
struct Zero;
impl RewardSource<f64> for Zero {
    fn id(&self) -> String {
        "zero/v1".into()
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
impl GradientProvider<f64> for Zero {
    fn id(&self) -> String {
        "zero-gradient/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move { TensorBatch::vectors(p.len(), 3, vec![0.; p.len() * 3]) })
    }
}
/// Walkers in three dimensions under a zero reward and potential.
fn free_walkers(x: Vec<f64>, v: Vec<f64>, config: GasConfig) -> AlgorithmicGas<f64> {
    let n = x.len() / 3;
    let mut observations = ObservationBatch::positions(TensorBatch::vectors(n, 3, x).unwrap());
    observations
        .fields
        .insert("velocities".into(), TensorBatch::vectors(n, 3, v).unwrap());
    block_on(
        GasBuilder::new(Population::new(observations).unwrap(), Zero)
            .config(config)
            .gradient(Zero)
            .build(),
    )
    .unwrap()
}
/// Two walkers at the origin with opposite unit velocities.
fn two_walkers(config: GasConfig) -> AlgorithmicGas<f64> {
    free_walkers(vec![0.; 6], vec![1., 0., 0., -1., 0., 0.], config)
}
/// Noiseless frictionless BAOAB with a dense viscous force and no other force.
fn viscous_baoab(viscosity: ViscousForceConfig) -> GasConfig {
    GasConfig {
        precision: Precision::F64,
        qft: QftExecutionConfig {
            viscosity: Some(viscosity),
            ..Default::default()
        },
        kinetic: KineticOperator {
            integrator: KineticKind::Baoab {
                positions: "positions".into(),
                velocities: "velocities".into(),
                dt: 0.1,
                friction: 0.,
            },
            noise: Noise {
                innovation: InnovationLaw::Gaussian,
                geometry: NoiseGeometry::Isotropic {
                    scale: FactorValues::Constant { values: vec![0.] },
                },
            },
            ..Default::default()
        },
        ..Default::default()
    }
}
fn capabilities(archive: &RunArchive<f64>, measurement: &MeasurementConfig) -> Capabilities {
    let dimension = archive.steps[0]
        .before
        .observations
        .field(position_field(&archive.gas_config))
        .unwrap()
        .width();
    Capabilities::of(&archive.gas_config, &archive.config, dimension).refine(measurement)
}
fn frames(archive: &RunArchive<f64>, measurement: &MeasurementConfig) -> Vec<Frame> {
    let capabilities = capabilities(archive, measurement);
    archive
        .steps
        .iter()
        .map(|step| {
            let frame = extract(&archive.gas_config, step, measurement, &capabilities).unwrap();
            frame.validate().unwrap();
            frame
        })
        .collect()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn evaluation<'a>(step: &'a RecordedStep<f64>, stage: &str, field: &str) -> &'a FieldEvaluation {
    step.field_evaluations
        .iter()
        .find(|e| e.stage == stage && e.field == field)
        .unwrap()
}
fn evaluation_mut<'a>(
    step: &'a mut RecordedStep<f64>,
    stage: &str,
    field: &str,
) -> &'a mut FieldEvaluation {
    step.field_evaluations
        .iter_mut()
        .find(|e| e.stage == stage && e.field == field)
        .unwrap()
}
fn snapshot<'a>(step: &'a RecordedStep<f64>, stage: &str) -> &'a StageSnapshot {
    step.stages.iter().find(|s| s.stage == stage).unwrap()
}
/// The historical-donor run of four steps.
fn historical_archive() -> RunArchive<f64> {
    let mut config = GasConfig::viscous_euclidean(3, 0.05, unit_viscosity()).unwrap();
    config.distance_donors.history_window = 2;
    config.cloning_donors.history_window = 2;
    // Companion revival and the component collision are defined on current-frame donors only.
    config.clone_decision.revival_from_companion = false;
    config.clone_transform.restitution = None;
    record(&mut euclidean(config, 9, 1.), 4, false)
}
/// The same pool in the opposite order, with the sampled indices following their sources.
fn reverse_pool(batch: &mut CompanionBatch, sources: &mut [SourceRef]) {
    let last = sources.len() as u32 - 1;
    sources.reverse();
    for (index, _) in batch.indices.iter_mut().zip(&batch.valid).filter(|e| *e.1) {
        *index = last - *index;
    }
}

/// Companions recomputed from the pool of the role.
fn assert_companions(
    companions: &Companions,
    batch: &CompanionBatch,
    sources: &[SourceRef],
    step: u64,
) {
    assert_eq!(companions.count, batch.count);
    assert_eq!(companions.valid, batch.valid);
    assert_eq!(companions.mutual, batch.mutual);
    for (e, &index) in batch.indices.iter().enumerate() {
        let entry = (
            companions.slot[e],
            companions.generation[e],
            companions.historical[e],
        );
        if batch.valid[e] {
            let source = sources[index as usize];
            assert_eq!(
                entry,
                (source.slot, source.generation, source.frame != step - 1)
            );
        } else {
            assert_eq!(entry, (0, 0, false));
        }
    }
}
/// Every per-walker field of `frame` against the raw records of `step`.
fn assert_frame(frame: &Frame, step: &RecordedStep<f64>, gas: &GasConfig) {
    let report = &step.report;
    let n = step.before.len();
    let x = step.before.observations.field("positions").unwrap();
    assert_eq!(
        (frame.step, frame.epoch, frame.n, frame.d),
        (report.step, step.epoch, n, x.width())
    );
    assert_eq!(step.stages[0].stage, "pre_clone");
    assert_eq!(bits(&frame.x), bits(x.values()));
    assert_eq!(
        bits(&frame.x),
        bits(&step.stages[0].fields["positions"].values)
    );
    let v = step.before.observations.field("velocities").unwrap();
    assert_eq!(bits(frame.v.as_ref().unwrap()), bits(v.values()));
    assert!(x.values().iter().chain(v.values()).all(|a| a.is_finite()));
    assert_eq!(frame.eligible, step.before.eligible(gas.include_truncated));
    assert_eq!(frame.eligible, report.pre_clone_eligible);
    assert_eq!(frame.generation, step.before.generations);
    let fitness = frame.fitness.as_ref().unwrap();
    assert_eq!(bits(fitness), bits(&report.pre_clone_fitness.fitness));
    assert_eq!(
        bits(frame.reward.as_ref().unwrap()),
        bits(&report.pre_clone_rewards.raw)
    );
    let distance = frame.distance.as_ref().unwrap();
    let cloning = frame.cloning.as_ref().unwrap();
    assert_companions(
        distance,
        &report.distance_companions,
        &report.distance_sources,
        report.step,
    );
    assert_companions(
        cloning,
        &report.cloning_companions,
        &report.clone_plan.sources,
        report.step,
    );
    assert_eq!(cloning.count, 1);
    let companion_fitness = frame.companion_fitness.as_ref().unwrap();
    for i in 0..n {
        let choice = &report.clone_plan.choices[i];
        assert_eq!(
            (frame.cloned[i], frame.revived[i]),
            (choice.accepted, choice.revival)
        );
        assert!(!frame.revived[i] || (frame.cloned[i] && !frame.eligible[i]));
        assert_eq!(fitness[i] > 0., frame.eligible[i]);
        assert!(frame.eligible[i] || fitness[i] == 0.);
        for k in 0..distance.count {
            let e = i * distance.count + k;
            assert!(!distance.valid[e] || frame.eligible[i]);
            assert!(
                !distance.valid[e]
                    || distance.historical[e]
                    || frame.eligible[distance.slot[e] as usize]
            );
        }
        if !cloning.valid[i] {
            assert_eq!(companion_fitness[i], 0.);
            continue;
        }
        let index = report.cloning_companions.indices[i] as usize;
        assert_eq!(
            companion_fitness[i].to_bits(),
            step.donor_fitness[index].to_bits()
        );
        if !cloning.historical[i] {
            let j = cloning.slot[i] as usize;
            assert!(frame.eligible[j]);
            assert_eq!(companion_fitness[i].to_bits(), fitness[j].to_bits());
            assert_eq!(cloning.generation[i], frame.generation[j]);
        }
        if frame.eligible[i] {
            let probability = gas.clone_decision.acceptance_probability(
                report.step,
                fitness[i],
                companion_fitness[i],
            );
            assert_eq!(choice.probability, Some(probability));
        }
    }
}
/// A stage kick against the two recorded fields and the input snapshot of the stage.
fn assert_kick(kick: &StageKick, step: &RecordedStep<f64>, stage: &str, gas: &GasConfig) {
    let force = evaluation(step, stage, "viscous_force");
    let velocity = evaluation(step, stage, "force_input_velocity");
    let input = snapshot(step, &format!("{stage}_input"));
    assert_eq!(
        (force.version, velocity.version),
        (input.version, input.version)
    );
    assert_eq!(bits(&kick.force), bits(&force.values));
    assert_eq!(bits(&kick.velocity), bits(&velocity.values));
    assert_eq!(
        bits(&kick.velocity),
        bits(&input.fields["velocities"].values)
    );
    assert_eq!(kick.generation, input.generations);
    assert_eq!(kick.generation, step.final_population.generations);
    let eligible: Vec<bool> = input
        .validity
        .iter()
        .map(|v| v.eligible(gas.include_truncated))
        .collect();
    assert_eq!(kick.available, eligible);
}
/// Dense viscous force in the operation order of the kinetic operator.
fn dense_force(x: &[f64], kick: &StageKick, viscosity: &ViscousForceConfig, d: usize) -> Vec<f64> {
    let n = kick.available.len();
    let alive = kick.available.iter().filter(|&&a| a).count() as f64;
    let denominator = 2. * viscosity.bandwidth * viscosity.bandwidth;
    let mut force = vec![0.; n * d];
    for i in (0..n).filter(|&i| kick.available[i]) {
        for j in (0..n).filter(|&j| j != i && kick.available[j]) {
            let mut distance = 0.;
            for a in 0..d {
                let delta = x[i * d + a] - x[j * d + a];
                distance += delta * delta;
            }
            let weight = viscosity.coefficient * (-distance / denominator).exp() / alive;
            for a in 0..d {
                force[i * d + a] += weight * (kick.velocity[j * d + a] - kick.velocity[i * d + a]);
            }
        }
    }
    force
}

#[test]
fn einstein_hilbert_frames_reproduce_every_record_of_their_step() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 6, true);
    let config = &archive.gas_config;
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    assert_eq!(capabilities.dimension, 3);
    assert_eq!(
        capabilities.missing.keys().collect::<Vec<_>>(),
        vec![&Record::PeriodicBox]
    );
    let frames = frames(&archive, &measurement);
    assert_eq!(frames.len(), 6);
    let viscosity = config.qft.graph_viscosity.as_ref().unwrap();
    let mut clones = 0;
    for (t, (frame, step)) in frames.iter().zip(&archive.steps).enumerate() {
        assert_eq!((frame.step, frame.n, frame.d), (t as u64 + 1, 12, 3));
        assert_frame(frame, step, config);
        assert!(frame.eligible.iter().all(|&e| e));
        let kick = frame.kick.as_ref().unwrap();
        assert!(frame.recorded_color.is_none());
        let graph = frame.graph.as_ref().unwrap();
        assert_eq!(Some(graph), step.graph.as_ref());
        let weights = &graph.weights[&viscosity.weights];
        for (stage, name) in [(&kick.b1, "B1"), (&kick.b2, "B2")] {
            let stage = stage.as_ref().unwrap();
            assert_kick(stage, step, name, config);
            // Graph viscous force over the archived post-clone graph, edge by edge.
            let mut force = vec![0.; 36];
            for i in 0..12 {
                for e in graph.graph.range(i) {
                    let j = graph.graph.neighbors()[e] as usize;
                    let w = viscosity.coefficient * weights[e];
                    for a in 0..3 {
                        force[i * 3 + a] +=
                            w * (stage.velocity[j * 3 + a] - stage.velocity[i * 3 + a]);
                    }
                }
            }
            assert_eq!(bits(&stage.force), bits(&force));
            assert!(stage.force.iter().any(|&f| f != 0.));
        }
        // Uniform matchings of an even population are fixed-point-free involutions.
        for companions in [&frame.distance, &frame.cloning] {
            let companions = companions.as_ref().unwrap();
            assert!(companions.mutual && companions.historical.iter().all(|&h| !h));
            for i in 0..12 {
                let j = companions.slot[i] as usize;
                assert!(companions.valid[i] && j != i && companions.slot[j] as usize == i);
            }
        }
        let cloned = frame.cloned.iter().filter(|&&c| c).count();
        assert!(frame.step.is_multiple_of(2) || cloned == 0);
        assert!(frame.revived.iter().all(|&r| !r));
        clones += cloned;
        if let Some(next) = frames.get(t + 1) {
            let advanced: Vec<u64> = frame
                .generation
                .iter()
                .zip(&frame.cloned)
                .map(|(&g, &c)| g + u64::from(c))
                .collect();
            assert_eq!(next.generation, advanced);
            assert_eq!(kick.b2.as_ref().unwrap().generation, next.generation);
        }
        let state = base_state(frame, &capabilities);
        assert_eq!((state.step, state.n, state.d), (frame.step, 12, 3));
        assert_eq!(bits(&state.x), bits(&frame.x));
        assert_eq!(state.v, frame.v);
        assert_eq!(state.fitness, frame.fitness);
        assert_eq!(state.generation, frame.generation);
        assert_eq!(state.cloned, frame.cloned);
        assert_eq!(state.eligible, frame.eligible);
        let time: Vec<f64> = frame.x.chunks_exact(3).map(|x| x[2]).collect();
        assert_eq!(state.euclidean_time, Some(time));
        assert_eq!(
            state.distance_companion.as_ref(),
            Some(&frame.distance.as_ref().unwrap().slot)
        );
        assert_eq!(
            state.cloning_companion.as_ref(),
            Some(&frame.cloning.as_ref().unwrap().slot)
        );
        assert_eq!(state.color.len(), 36);
        assert!(state.color.iter().all(|c| c.abs2() == 0.));
        assert!(
            state
                .color_valid
                .iter()
                .chain(&state.force_valid)
                .all(|&m| !m)
        );
        assert_eq!((state.color_valid.len(), state.force_valid.len()), (12, 12));
        assert!(state.force.is_none() && state.phase_velocity.is_none());
        assert!(state.score.is_none() && state.score_valid.is_empty());
        assert!(state.score_gradient.is_none() && state.role.is_none());
    }
    assert!(clones > 0);
}

#[test]
fn odd_einstein_hilbert_population_pairs_one_walker_with_itself_per_role() {
    let mut gas = einstein_hilbert::<f64>(11, 5);
    let archive = record(&mut gas, 2, false);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    assert!(capabilities.missing[&Record::Graph].contains("not recorded"));
    for (frame, step) in frames(&archive, &measurement).iter().zip(&archive.steps) {
        assert_frame(frame, step, &archive.gas_config);
        assert!(frame.graph.is_none() && step.graph.is_none());
        let state = base_state(frame, &capabilities);
        for (companions, map) in [
            (&frame.distance, &state.distance_companion),
            (&frame.cloning, &state.cloning_companion),
        ] {
            let companions = companions.as_ref().unwrap();
            let own: Vec<usize> = (0..11)
                .filter(|&i| companions.slot[i] as usize == i)
                .collect();
            assert_eq!(own.len(), 1);
            assert_eq!(
                companions.mask(own[0], 0, &frame.eligible),
                Some(CompanionMask::SelfCompanion)
            );
            for (i, &j) in map.as_ref().unwrap().iter().enumerate() {
                assert_eq!(
                    j,
                    if i == own[0] {
                        NO_COMPANION
                    } else {
                        companions.slot[i]
                    }
                );
            }
        }
    }
}

#[test]
fn two_walker_viscous_run_yields_the_closed_form_kicks() {
    let config = viscous_baoab(ViscousForceConfig {
        coefficient: 2.,
        bandwidth: 1.,
        row_normalized: false,
    });
    let archive = record(&mut two_walkers(config), 2, false);
    let frames = frames(&archive, &MeasurementConfig::default());
    // (pre-clone v, B1 force, B1 input v, B2 force, B2 input v) of walker 0 along x, from an
    // independent BAOAB integration; walker 1 is the mirror image.
    let expected = [
        [1., -2., 1., -1.771074925690515, 0.9],
        [
            0.8114462537154743,
            -1.5968134594455339,
            0.8114462537154743,
            -1.3873434213287468,
            0.7316055807431976,
        ],
    ];
    for ((frame, step), row) in frames.iter().zip(&archive.steps).zip(expected) {
        assert_frame(frame, step, &archive.gas_config);
        let kick = frame.kick.as_ref().unwrap();
        let (b1, b2) = (kick.b1.as_ref().unwrap(), kick.b2.as_ref().unwrap());
        let mirrored = |value: f64| vec![value, 0., 0., -value, 0., 0.];
        assert_eq!(frame.v, Some(mirrored(row[0])));
        assert_eq!(b1.force, mirrored(row[1]));
        assert_eq!(b1.velocity, mirrored(row[2]));
        assert_eq!(b2.force, mirrored(row[3]));
        assert_eq!(b2.velocity, mirrored(row[4]));
        assert_eq!(b1.available, [true, true]);
        assert_eq!(frame.cloned, [false, false]);
        assert_eq!(frame.distance.as_ref().unwrap().slot, [1, 0]);
        assert_eq!(frame.cloning.as_ref().unwrap().slot, [1, 0]);
        let fitness = frame.fitness.as_ref().unwrap();
        assert!((fitness[0] - 1.000002000001).abs() < 1e-12);
        assert_eq!(frame.companion_fitness.as_ref(), Some(fitness));
    }
    // The B2 input velocity of step 1 plus its half kick is the pre-clone velocity of step 2.
    let previous = frames[0].kick.as_ref().unwrap().b2.as_ref().unwrap();
    let carried = previous.velocity[0] + 0.05 * previous.force[0];
    assert_eq!(frames[1].v.as_ref().unwrap()[0], carried);
}

#[test]
fn three_walker_dense_kernel_yields_the_tabulated_first_kick() {
    let mut config = viscous_baoab(ViscousForceConfig {
        coefficient: 1.5,
        bandwidth: 0.8,
        row_normalized: false,
    });
    // Unequal separations give unequal fitness: keep the clone gate closed on step 1.
    config.clone_decision.every = 1000;
    let x = vec![0., 0., 0., 0.5, -0.3, 0.2, -0.4, 0.9, 0.1];
    let v = vec![0.9, -0.4, 2.1, -1.3, 0.8, 0.25, 0.2, 5., -2.6];
    let archive = record(&mut free_walkers(x, v.clone(), config), 1, false);
    let frame = &frames(&archive, &MeasurementConfig::default())[0];
    assert_frame(frame, &archive.steps[0], &archive.gas_config);
    assert_eq!(frame.cloned, [false; 3]);
    let b1 = frame.kick.as_ref().unwrap().b1.as_ref().unwrap();
    assert_eq!(b1.velocity, v);
    assert_eq!(frame.v.as_ref(), Some(&v));
    // `F_i = (ν / 3) Σ_{j ≠ i} exp(−|x_i − x_j|² / 2ρ²) (v_j − v_i)` from an independent
    // double-precision evaluation.
    let expected = [
        -0.9802157043825539,
        1.7014987471632073,
        -1.780253123383634,
        0.9457604596641412,
        -0.08661450004143556,
        0.4436128761244382,
        0.03445524471841274,
        -1.6148842471217717,
        1.3366402472591958,
    ];
    for (force, expected) in b1.force.iter().zip(expected) {
        assert!((force - expected).abs() < 1e-14, "{force} {expected}");
    }
    // The symmetric kernel exchanges momentum: the forces sum to zero.
    for a in 0..3 {
        let total: f64 = b1.force.iter().skip(a).step_by(3).sum();
        assert!(total.abs() < 1e-15, "{total}");
    }
}

#[test]
fn viscous_euclidean_frames_resolve_pool_indices_past_dead_slots() {
    let config = GasConfig::viscous_euclidean(3, 0.05, unit_viscosity()).unwrap();
    let archive = record(&mut euclidean(config, 9, 5.), 4, false);
    let config = &archive.gas_config;
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    assert!(capabilities.has(Record::Color) && capabilities.dense_viscosity);
    assert!(!capabilities.mutual_distance && !capabilities.mutual_cloning);
    let frames = frames(&archive, &measurement);
    let (mut dead, mut shifted, mut revived) = (0, 0, 0);
    for (t, (frame, step)) in frames.iter().zip(&archive.steps).enumerate() {
        assert_frame(frame, step, config);
        assert!(frame.graph.is_none());
        // A revival is an accepted decision: it advances the incarnation like a clone.
        if let Some(next) = frames.get(t + 1) {
            for i in 0..12 {
                let advanced = frame.generation[i] + u64::from(frame.cloned[i]);
                assert_eq!(next.generation[i], advanced);
            }
        }
        let (distance, cloning) = (
            frame.distance.as_ref().unwrap(),
            frame.cloning.as_ref().unwrap(),
        );
        assert!(!distance.mutual && !cloning.mutual);
        let state = base_state(frame, &capabilities);
        assert!(state.euclidean_time.is_none());
        for i in 0..12 {
            if frame.eligible[i] {
                let index = step.report.cloning_companions.indices[i];
                shifted += usize::from(cloning.valid[i] && index != cloning.slot[i]);
                continue;
            }
            dead += 1;
            revived += usize::from(frame.revived[i]);
            // A dead row samples no distance companion and revives from its cloning companion.
            assert!(!distance.valid[i] && cloning.valid[i]);
            assert_eq!(
                distance.mask(i, 0, &frame.eligible),
                Some(CompanionMask::Unsampled)
            );
            assert_eq!(
                cloning.mask(i, 0, &frame.eligible),
                Some(CompanionMask::Ineligible)
            );
            assert_eq!(state.distance_companion.as_ref().unwrap()[i], NO_COMPANION);
            assert_eq!(state.cloning_companion.as_ref().unwrap()[i], NO_COMPANION);
        }
        let kick = frame.kick.as_ref().unwrap();
        for (stage, name) in [(&kick.b1, "B1"), (&kick.b2, "B2")] {
            let stage = stage.as_ref().unwrap();
            assert_kick(stage, step, name, config);
            let x = &snapshot(step, &format!("{name}_input")).fields["positions"].values;
            let force = dense_force(x, stage, config.qft.viscosity.as_ref().unwrap(), 3);
            assert_eq!(bits(&stage.force), bits(&force));
        }
    }
    assert!(dead > 0 && revived == dead && shifted > 0);
}

#[test]
fn euclidean_gas_without_viscosity_declares_colour_missing_and_keeps_no_kick() {
    let archive = record(
        &mut euclidean(GasConfig::euclidean(3, 0.05).unwrap(), 9, 1.),
        2,
        false,
    );
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    assert!(capabilities.missing[&Record::Color].contains("identically zero"));
    assert!(capabilities.missing[&Record::Graph].contains("geometry"));
    assert!(capabilities.has(Record::Velocities) && capabilities.has(Record::Fitness));
    for (frame, step) in frames(&archive, &measurement).iter().zip(&archive.steps) {
        assert_frame(frame, step, &archive.gas_config);
        assert!(frame.kick.is_none() && frame.recorded_color.is_none() && frame.graph.is_none());
    }
    // The records exist; a caller that declares the colour present reads a zero force.
    let mut declared = capabilities.clone();
    declared.missing.remove(&Record::Color);
    let step = &archive.steps[0];
    let frame = extract(&archive.gas_config, step, &measurement, &declared).unwrap();
    let kick = frame.kick.unwrap();
    for (stage, name) in [(&kick.b1, "B1"), (&kick.b2, "B2")] {
        let stage = stage.as_ref().unwrap();
        assert_kick(stage, step, name, &archive.gas_config);
        assert!(stage.force.iter().all(|&f| f == 0.));
        assert!(stage.available.iter().all(|&a| a));
    }
}

#[test]
fn capabilities_gate_every_optional_record_of_a_frame() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 1, true);
    let measurement = MeasurementConfig::default();
    let full = capabilities(&archive, &measurement);
    let step = &archive.steps[0];
    let without = |record: Record| {
        let mut capabilities = full.clone();
        capabilities.missing.insert(record, "declared".into());
        extract(&archive.gas_config, step, &measurement, &capabilities).unwrap()
    };
    let frame = without(Record::Velocities);
    assert!(frame.v.is_none() && frame.kick.is_none() && frame.fitness.is_some());
    let frame = without(Record::Color);
    assert!(frame.v.is_some() && frame.kick.is_none() && frame.recorded_color.is_none());
    let frame = without(Record::Fitness);
    assert!(frame.fitness.is_none() && frame.companion_fitness.is_none());
    assert!(frame.reward.is_some() && frame.cloning.is_some());
    let frame = without(Record::DistanceCompanions);
    assert!(frame.distance.is_none() && frame.cloning.is_some());
    assert!(base_state(&frame, &full).distance_companion.is_none());
    let frame = without(Record::CloningCompanions);
    assert!(frame.cloning.is_none() && frame.companion_fitness.is_none());
    assert!(frame.distance.is_some() && frame.fitness.is_some());
    let frame = without(Record::Graph);
    assert!(frame.graph.is_none() && frame.kick.is_some());
    // The decisions feed the colour and scale masks: no declaration drops them.
    let reference = without(Record::PeriodicBox);
    assert_eq!(reference, frames(&archive, &measurement)[0]);
    assert_eq!(without(Record::ClonePlan), reference);
    let mut timeless = full.clone();
    timeless
        .missing
        .insert(Record::EuclideanTime, "declared".into());
    assert_eq!(timeless.euclidean_axis, Some(2));
    assert!(base_state(&reference, &full).euclidean_time.is_some());
    assert!(base_state(&reference, &timeless).euclidean_time.is_none());
    // A recorded-field source without the colour record reads nothing either.
    let recorded = MeasurementConfig {
        color: ColorSource::RecordedField {
            stage: "B2".into(),
            amplitude: "total_force".into(),
            phase: "force_input_velocity".into(),
            alignment: StepAlignment::default(),
            threshold: 1e-12,
        },
        ..MeasurementConfig::default()
    };
    let mut colorless = full.clone();
    colorless.missing.insert(Record::Color, "declared".into());
    let frame = extract(&archive.gas_config, step, &recorded, &colorless).unwrap();
    assert!(frame.kick.is_none() && frame.recorded_color.is_none());
    let frame = extract(&archive.gas_config, step, &recorded, &full).unwrap();
    assert!(frame.kick.is_none() && frame.recorded_color.is_some());
    // It is the colour route of a run without velocities: it does not need them.
    let mut still = full.clone();
    still.missing.insert(Record::Velocities, "declared".into());
    let slow = extract(&archive.gas_config, step, &recorded, &still).unwrap();
    assert!(slow.v.is_none() && slow.kick.is_none());
    assert_eq!(slow.recorded_color, frame.recorded_color);
    let mut planar = full.clone();
    planar.dimension = 2;
    planar.euclidean_axis = None;
    assert!(matches!(
        extract(&archive.gas_config, step, &measurement, &planar),
        Err(GasError::Shape(_))
    ));
    let mut invalid = full.clone();
    invalid.time_step = Some(0.);
    assert!(matches!(
        extract(&archive.gas_config, step, &measurement, &invalid),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn coordinates_are_read_from_the_fields_the_integrator_names() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 1, false);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let reference = &frames(&archive, &measurement)[0];
    let mut step = archive.steps[0].clone();
    let fields = &mut step.stages[0].fields;
    for (from, to) in [("positions", "q"), ("velocities", "p")] {
        let field = fields.remove(from).unwrap();
        fields.insert(to.into(), field);
    }
    let mut renamed = archive.gas_config.clone();
    let KineticKind::Baoab {
        positions,
        velocities,
        ..
    } = &mut renamed.kinetic.integrator
    else {
        panic!("Einstein-Hilbert kinetics are BAOAB");
    };
    (*positions, *velocities) = ("q".into(), "p".into());
    assert_eq!(position_field(&renamed), "q");
    let frame = extract(&renamed, &step, &measurement, &capabilities).unwrap();
    assert_eq!(&frame, reference);
    let error = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap_err();
    assert!(matches!(error, GasError::MissingField(name) if name == "positions"));
}

#[test]
fn repeated_and_chunked_recordings_yield_identical_frames() {
    let run = || {
        let config = GasConfig::viscous_euclidean(3, 0.05, unit_viscosity()).unwrap();
        euclidean(config, 9, 5.)
    };
    let measurement = MeasurementConfig::default();
    let whole = frames(&record(&mut run(), 4, false), &measurement);
    assert_eq!(whole, frames(&record(&mut run(), 4, false), &measurement));
    let mut gas = run();
    let mut chunked = Vec::new();
    for chunk in 0..2 {
        let archive = record(&mut gas, 2, false);
        assert_eq!(archive.anchors[0].step, 2 * chunk);
        chunked.extend(frames(&archive, &measurement));
    }
    // `step` continues across chunks while `epoch` restarts, so the frames are the same.
    assert_eq!(chunked, whole);
    assert_eq!(
        whole.iter().map(|f| f.step).collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
}

#[test]
fn three_distance_companions_per_walker_keep_the_row_major_entry_order() {
    let mut config = GasConfig::viscous_euclidean(3, 0.05, unit_viscosity()).unwrap();
    config.distance_donors.count = 3;
    let archive = record(&mut euclidean(config, 9, 1.), 3, false);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    for (frame, step) in frames(&archive, &measurement).iter().zip(&archive.steps) {
        assert_frame(frame, step, &archive.gas_config);
        let distance = frame.distance.as_ref().unwrap();
        assert_eq!((distance.count, distance.slot.len()), (3, 36));
        assert_eq!(frame.cloning.as_ref().unwrap().slot.len(), 12);
        let state = base_state(frame, &capabilities);
        for (i, &j) in state
            .distance_companion
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
        {
            let first = (0..3)
                .map(|k| i * 3 + k)
                .find(|&e| distance.valid[e] && distance.slot[e] as usize != i)
                .map_or(NO_COMPANION, |e| distance.slot[e]);
            assert_eq!(j, first);
        }
    }
}

#[test]
fn historical_donors_are_flagged_and_carry_their_rescored_fitness() {
    let archive = historical_archive();
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let frames = frames(&archive, &measurement);
    let (mut historical, mut rescored) = (0, 0);
    for (frame, step) in frames.iter().zip(&archive.steps) {
        assert_frame(frame, step, &archive.gas_config);
        let state = base_state(frame, &capabilities);
        let report = &step.report;
        for (companions, batch, sources, map) in [
            (
                frame.distance.as_ref().unwrap(),
                &report.distance_companions,
                &report.distance_sources,
                state.distance_companion.as_ref().unwrap(),
            ),
            (
                frame.cloning.as_ref().unwrap(),
                &report.cloning_companions,
                &report.clone_plan.sources,
                state.cloning_companion.as_ref().unwrap(),
            ),
        ] {
            // The pool holds the current frame and up to two earlier ones.
            let depth = sources.iter().map(|s| frame.step - s.frame).max().unwrap();
            assert_eq!(depth, frame.step.min(3));
            for i in (0..12).filter(|&i| companions.valid[i] && companions.historical[i]) {
                historical += 1;
                assert_eq!(
                    companions.mask(i, 0, &frame.eligible),
                    Some(CompanionMask::Historical)
                );
                assert_eq!(map[i], NO_COMPANION);
                // The source is a row of the pre-clone population of step `source.frame + 1`.
                let source = sources[batch.indices[i] as usize];
                assert!(source.frame + 1 < frame.step);
                let origin = &frames[source.frame as usize];
                assert_eq!(origin.step, source.frame + 1);
                assert!(origin.eligible[source.slot as usize]);
                assert_eq!(
                    companions.generation[i],
                    origin.generation[source.slot as usize]
                );
            }
        }
        let cloning = frame.cloning.as_ref().unwrap();
        let companion_fitness = frame.companion_fitness.as_ref().unwrap();
        for i in (0..12).filter(|&i| cloning.valid[i] && cloning.historical[i]) {
            let source = report.clone_plan.sources[report.cloning_companions.indices[i] as usize];
            let origin = &frames[source.frame as usize];
            let recorded = origin.fitness.as_ref().unwrap()[source.slot as usize];
            // The decision input is the fitness rescored in this step, not the recorded one.
            assert!(companion_fitness[i] > 0.);
            rescored += usize::from(companion_fitness[i] != recorded);
        }
    }
    assert!(historical > 0 && rescored > 0);
}

#[test]
fn recorded_field_source_reads_the_named_stage_with_the_generations_of_its_version() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 2, false);
    let source = |stage: &str, amplitude: &str, phase: &str| MeasurementConfig {
        color: ColorSource::RecordedField {
            stage: stage.into(),
            amplitude: amplitude.into(),
            phase: phase.into(),
            alignment: StepAlignment::default(),
            threshold: 1e-12,
        },
        ..MeasurementConfig::default()
    };
    let measurement = source("B2", "total_force", "force_input_velocity");
    let step = &archive.steps[1];
    let frame = &frames(&archive, &measurement)[1];
    assert!(frame.kick.is_none());
    let recorded = frame.recorded_color.as_ref().unwrap();
    assert_eq!(
        bits(&recorded.force),
        bits(&evaluation(step, "B2", "total_force").values)
    );
    assert_eq!(
        bits(&recorded.velocity),
        bits(&evaluation(step, "B2", "force_input_velocity").values)
    );
    assert_eq!(recorded.generation, step.final_population.generations);
    assert!(recorded.available.iter().all(|&a| a));
    // Collision records belong to the pre-clone version and cover the collision members only.
    assert!(frame.cloned.iter().any(|&c| c));
    let measurement = source(
        "component_collision",
        "collision_output_velocity",
        "collision_input_velocity",
    );
    let frame = &frames(&archive, &measurement)[1];
    let recorded = frame.recorded_color.as_ref().unwrap();
    assert_eq!(recorded.generation, frame.generation);
    assert_ne!(recorded.generation, step.final_population.generations);
    let members = &evaluation(step, "component_collision", "collision_component_id").available;
    assert_eq!(&recorded.available, members);
    assert!((0..12).all(|i| !frame.cloned[i] || members[i]) && members.iter().any(|&m| !m));
    // An absent field, and a field that is no `[d]` vector, yield no record.
    for (stage, amplitude) in [
        ("B2", "absent"),
        ("B9", "total_force"),
        ("B2", "curl_field"),
    ] {
        let measurement = source(stage, amplitude, "force_input_velocity");
        let frames = frames(&archive, &measurement);
        assert!(
            frames
                .iter()
                .all(|f| f.recorded_color.is_none() && f.kick.is_none())
        );
    }
    // The Euclidean Gas without viscosity has a colour through this source only.
    let euclidean = GasConfig::euclidean(3, 0.05).unwrap();
    let recording = RecordingConfig::default();
    let plain = Capabilities::of(&euclidean, &recording, 3);
    assert!(plain.missing[&Record::Color].contains("identically zero"));
    assert!(plain.refine(&measurement).has(Record::Color));
}

#[test]
fn jump_kinetics_yield_frames_without_velocities_or_kicks() {
    let config = GasConfig {
        precision: Precision::F64,
        ..Default::default()
    };
    assert!(matches!(
        config.kinetic.integrator,
        KineticKind::DirectJump { .. }
    ));
    let archive = record(&mut two_walkers(config), 2, false);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    assert!(capabilities.missing[&Record::Velocities].contains("no velocities"));
    assert!(capabilities.missing[&Record::Color].contains("no viscous force"));
    for (frame, step) in frames(&archive, &measurement).iter().zip(&archive.steps) {
        assert!(frame.v.is_none() && frame.kick.is_none());
        assert_eq!(
            bits(&frame.x),
            bits(
                step.before
                    .observations
                    .field("positions")
                    .unwrap()
                    .values()
            )
        );
        assert!(frame.distance.is_some() && frame.companion_fitness.is_some());
        assert!(base_state(frame, &capabilities).v.is_none());
    }
}

#[test]
fn single_precision_run_is_refused_explicitly() {
    let mut gas = einstein_hilbert::<f32>(12, 5);
    let bytes = record(&mut gas, 2, false).to_bytes().unwrap();
    let archive = RunArchive::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(archive.gas_config.precision, Precision::F32);
    // Every recorded force of the run is the image of a single-precision number.
    let force = &evaluation(&archive.steps[0], "B2", "viscous_force").values;
    assert!(force.iter().all(|&f| f64::from(f as f32) == f) && force.iter().any(|&f| f != 0.));
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let error = extract(
        &archive.gas_config,
        &archive.steps[0],
        &measurement,
        &capabilities,
    )
    .unwrap_err();
    assert!(matches!(error, GasError::Capability(_)), "{error}");
    assert!(error.to_string().contains("f64"), "{error}");
}

#[test]
fn tampered_stage_names_and_records_are_absent_or_explicit_errors() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 2, true);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let frame =
        |step: &RecordedStep<f64>| extract(&archive.gas_config, step, &measurement, &capabilities);
    let kick = |step: &RecordedStep<f64>| frame(step).unwrap().kick.unwrap();
    let original = &archive.steps[1];
    assert!(kick(original).b1.is_some() && kick(original).b2.is_some());

    let mut step = original.clone();
    step.stages[0].stage = "before_clone".into();
    let error = frame(&step).unwrap_err();
    assert!(matches!(error, GasError::Capability(_)), "{error}");
    assert!(error.to_string().contains("pre_clone"), "{error}");
    let mut step = original.clone();
    step.stages[0].fields.remove("positions");
    assert!(matches!(frame(&step), Err(GasError::MissingField(name)) if name == "positions"));
    let mut step = original.clone();
    step.stages[0].fields.remove("velocities");
    assert!(matches!(frame(&step), Err(GasError::MissingField(name)) if name == "velocities"));
    let mut step = original.clone();
    step.stages[0]
        .fields
        .get_mut("positions")
        .unwrap()
        .values
        .pop();
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    let mut step = original.clone();
    step.stages[0]
        .fields
        .get_mut("velocities")
        .unwrap()
        .item_shape = vec![2];
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));

    // A renamed stage or a missing record leaves that stage without a kick and the other intact.
    let mut step = original.clone();
    for e in step
        .field_evaluations
        .iter_mut()
        .filter(|e| e.stage == "B1")
    {
        e.stage = "B1_renamed".into();
    }
    assert!(kick(&step).b1.is_none());
    assert_eq!(kick(&step).b2, kick(original).b2);
    let mut step = original.clone();
    step.field_evaluations
        .retain(|e| !(e.stage == "B2" && e.field == "force_input_velocity"));
    assert_eq!(kick(&step).b1, kick(original).b1);
    assert!(kick(&step).b2.is_none());
    let mut step = original.clone();
    evaluation_mut(&mut step, "B1", "viscous_force").version += 1;
    assert!(kick(&step).b1.is_none());
    let mut step = original.clone();
    for field in ["viscous_force", "force_input_velocity"] {
        evaluation_mut(&mut step, "B1", field).version = 999;
    }
    assert!(kick(&step).b1.is_none());
    let mut step = original.clone();
    evaluation_mut(&mut step, "B1", "viscous_force").item_shape = vec![2];
    assert!(kick(&step).b1.is_none());
    let mut step = original.clone();
    evaluation_mut(&mut step, "B1", "viscous_force").rows = 11;
    assert!(kick(&step).b1.is_none());
    let mut step = original.clone();
    evaluation_mut(&mut step, "B1", "viscous_force")
        .values
        .pop();
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    let mut step = original.clone();
    evaluation_mut(&mut step, "B2", "force_input_velocity")
        .available
        .pop();
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));

    // The generations of a kick are those of the first snapshot of the record's version.
    let version = evaluation(original, "B1", "viscous_force").version;
    let holder = original
        .stages
        .iter()
        .position(|s| s.version == version)
        .unwrap();
    assert_ne!(original.stages[holder].stage, "pre_clone");
    let mut step = original.clone();
    step.stages[holder].generations[3] += 7;
    let tampered = kick(&step).b1.unwrap().generation;
    assert_eq!(tampered[3], original.final_population.generations[3] + 7);
    let mut step = original.clone();
    step.stages[holder].generations.pop();
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    let mut step = original.clone();
    step.stages[0].generations.pop();
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));

    // A nonfinite or uncovered row is unavailable, never an error.
    let mut step = original.clone();
    evaluation_mut(&mut step, "B1", "viscous_force").values[3 * 4 + 1] = f64::NAN;
    evaluation_mut(&mut step, "B1", "force_input_velocity").available[7] = false;
    let available = kick(&step).b1.unwrap().available;
    assert_eq!(
        available,
        (0..12).map(|i| i != 4 && i != 7).collect::<Vec<_>>()
    );
    let mut step = original.clone();
    step.stages[0].fields.get_mut("positions").unwrap().values[3 * 5] = f64::INFINITY;
    let tampered = frame(&step).unwrap();
    tampered.validate().unwrap();
    assert_eq!(
        tampered.eligible,
        (0..12).map(|i| i != 5).collect::<Vec<_>>()
    );

    // Pool indices and sources outside the recorded data.
    let mut step = original.clone();
    step.report.cloning_companions.indices[0] = 12;
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    let mut step = original.clone();
    step.report.distance_sources[0].slot = 12;
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    let mut step = original.clone();
    step.report.distance_companions.rows = 11;
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    // The score of a row reads one cloning companion: a wider well-formed batch is refused.
    let mut step = original.clone();
    let batch = &mut step.report.cloning_companions;
    batch.count = 2;
    batch.indices = batch.indices.iter().flat_map(|&j| [j, j]).collect();
    batch.valid = batch.valid.iter().flat_map(|&v| [v, v]).collect();
    let error = frame(&step).unwrap_err();
    assert!(matches!(error, GasError::Shape(_)), "{error}");
    assert!(error.to_string().contains("count"), "{error}");
    let mut step = original.clone();
    step.donor_fitness.pop();
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    let mut step = original.clone();
    step.report.clone_plan.choices.pop();
    assert!(matches!(frame(&step), Err(GasError::Shape(_))));
    let mut step = original.clone();
    step.graph.as_mut().unwrap().euclidean_length.pop();
    assert!(frame(&step).is_err());
}

#[test]
fn hand_built_frame_yields_the_tabulated_base_state() {
    let (t, f) = (true, false);
    let frame = Frame {
        step: 5,
        n: 4,
        d: 2,
        x: vec![0., 10., 1., 11., 2., 12., 3., 13.],
        eligible: vec![t, t, t, f],
        generation: vec![4, 0, 9, 1],
        cloned: vec![f, t, f, t],
        revived: vec![f, f, f, t],
        // Row 0: itself, the ineligible walker 3, then walker 2. Row 1: a historical source,
        // an unsampled entry, then walker 0. Row 2: no entry builds an element. Row 3 is
        // ineligible itself.
        distance: Some(Companions {
            count: 3,
            slot: vec![0, 3, 2, 2, 3, 0, 2, 2, 3, 0, 1, 2],
            generation: vec![0; 12],
            valid: vec![t, t, t, t, f, t, t, t, t, t, t, t],
            historical: vec![f, f, f, t, f, f, f, f, f, f, f, f],
            mutual: false,
        }),
        cloning: Some(Companions {
            count: 1,
            slot: vec![1, 0, 2, 0],
            generation: vec![0; 4],
            valid: vec![t; 4],
            historical: vec![f; 4],
            mutual: false,
        }),
        ..Frame::default()
    };
    frame.validate().unwrap();
    let planar = |axis| Capabilities {
        dimension: 2,
        euclidean_axis: Some(axis),
        ..Capabilities::nominal()
    };
    let state = base_state(&frame, &planar(0));
    assert_eq!(
        state.distance_companion,
        Some(vec![2, 0, NO_COMPANION, NO_COMPANION])
    );
    assert_eq!(
        state.cloning_companion,
        Some(vec![1, 0, NO_COMPANION, NO_COMPANION])
    );
    assert_eq!(state.euclidean_time, Some(vec![0., 1., 2., 3.]));
    assert_eq!(
        base_state(&frame, &planar(1)).euclidean_time,
        Some(vec![10., 11., 12., 13.])
    );
    assert_eq!((state.step, state.n, state.d), (5, 4, 2));
    assert_eq!(state.generation, [4, 0, 9, 1]);
    assert_eq!(state.cloned, [f, t, f, t]);
    assert_eq!(state.eligible, [t, t, t, f]);
    assert!(state.v.is_none() && state.fitness.is_none());
    assert_eq!((state.color.len(), state.color_valid.len()), (8, 4));
    // An axis of the capabilities beyond the width of the frame reads no time.
    let spatial = Capabilities::nominal();
    assert_eq!((spatial.dimension, spatial.euclidean_axis), (3, Some(2)));
    assert!(base_state(&frame, &spatial).euclidean_time.is_none());
}

#[test]
fn external_replacement_opens_the_epoch_its_frames_carry() {
    let config = viscous_baoab(ViscousForceConfig {
        coefficient: 2.,
        bandwidth: 1.,
        row_normalized: false,
    });
    let mut gas = two_walkers(config);
    gas.start_recording(RecordingConfig {
        max_steps: 2,
        ..RecordingConfig::default()
    })
    .unwrap();
    block_on(gas.step()).unwrap();
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(2, 3, vec![0.; 6]).unwrap());
    let v = vec![2., 0., 0., -2., 0., 0.];
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(2, 3, v.clone()).unwrap(),
    );
    block_on(gas.replace_population(Population::new(observations).unwrap())).unwrap();
    block_on(gas.step()).unwrap();
    let archive = gas.stop_recording().unwrap();
    archive.validate().unwrap();
    let frames = frames(&archive, &MeasurementConfig::default());
    let labels: Vec<_> = frames.iter().map(|f| (f.step, f.epoch)).collect();
    assert_eq!(labels, [(1, 0), (2, 1)]);
    let frame = &frames[1];
    assert_frame(frame, &archive.steps[1], &archive.gas_config);
    assert_eq!((&frame.x, frame.v.as_ref()), (&vec![0.; 6], Some(&v)));
    // `F_0 = (ν / 2) (v_1 − v_0) = −4` on coincident walkers.
    let b1 = frame.kick.as_ref().unwrap().b1.as_ref().unwrap();
    assert_eq!(b1.force, [-4., 0., 0., 4., 0., 0.]);
    // The epoch is a label: the same step under another epoch is the same frame otherwise.
    let mut step = archive.steps[1].clone();
    step.epoch = 7;
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let relabelled = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap();
    assert_eq!(
        relabelled,
        Frame {
            epoch: 7,
            ..frame.clone()
        }
    );
}

#[test]
fn pool_order_is_immaterial_to_the_resolved_companions() {
    let archive = historical_archive();
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let original = &archive.steps[3];
    let reference = extract(&archive.gas_config, original, &measurement, &capabilities).unwrap();
    let cloning = reference.cloning.as_ref().unwrap();
    assert!(cloning.historical.iter().any(|&h| h) && cloning.historical.iter().any(|&h| !h));
    // The cloning pool alone: its indices no longer resolve through the distance pool.
    let mut step = original.clone();
    let report = &mut step.report;
    reverse_pool(
        &mut report.cloning_companions,
        &mut report.clone_plan.sources,
    );
    step.donor_fitness.reverse();
    assert_ne!(step.report.clone_plan.sources, step.report.distance_sources);
    assert_ne!(
        step.report.cloning_companions,
        original.report.cloning_companions
    );
    let frame = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap();
    assert_eq!(frame, reference);
    let report = &mut step.report;
    reverse_pool(
        &mut report.distance_companions,
        &mut report.distance_sources,
    );
    assert_ne!(
        step.report.distance_companions,
        original.report.distance_companions
    );
    let frame = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap();
    assert_eq!(frame, reference);
}

#[test]
fn unsampled_cloning_entry_reads_neither_its_pool_index_nor_a_fitness() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 2, false);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let reference = &frames(&archive, &measurement)[1];
    let mut expected = reference.clone();
    let cloning = expected.cloning.as_mut().unwrap();
    let fitness = expected.companion_fitness.as_mut().unwrap();
    assert!(cloning.valid[0] && cloning.slot[0] != 0 && fitness[0] > 0.);
    (cloning.valid[0], cloning.slot[0], cloning.generation[0]) = (false, 0, 0);
    fitness[0] = 0.;
    // An unsampled entry holds no pool index: whatever number it carries is never read.
    for index in [
        archive.steps[1].report.cloning_companions.indices[0],
        u32::MAX,
    ] {
        let mut step = archive.steps[1].clone();
        let batch = &mut step.report.cloning_companions;
        (batch.valid[0], batch.indices[0]) = (false, index);
        let frame = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap();
        frame.validate().unwrap();
        assert_eq!(frame, expected);
        let cloning = frame.cloning.as_ref().unwrap();
        assert_eq!(
            cloning.mask(0, 0, &frame.eligible),
            Some(CompanionMask::Unsampled)
        );
        let state = base_state(&frame, &capabilities);
        assert_eq!(state.cloning_companion.as_ref().unwrap()[0], NO_COMPANION);
        assert_eq!(
            state.cloning_companion.as_ref().unwrap()[1..],
            reference.cloning.as_ref().unwrap().slot[1..]
        );
    }
}

#[test]
fn nonfinite_or_uncovered_rows_of_either_record_are_masked_and_kept_verbatim() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 2, false);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let original = &archive.steps[1];
    // Force and velocity exchange the roles they have in the tamper test above.
    let mut step = original.clone();
    evaluation_mut(&mut step, "B1", "force_input_velocity").values[3 * 2] = f64::NAN;
    evaluation_mut(&mut step, "B1", "viscous_force").values[3 * 10 + 2] = f64::NEG_INFINITY;
    evaluation_mut(&mut step, "B1", "viscous_force").available[9] = false;
    let frame = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap();
    frame.validate().unwrap();
    let kick = frame.kick.as_ref().unwrap();
    let b1 = kick.b1.as_ref().unwrap();
    let masked = |rows: &[usize]| (0..12).map(|i| !rows.contains(&i)).collect::<Vec<_>>();
    assert_eq!(b1.available, masked(&[2, 9, 10]));
    // The records are copied, never repaired: a masked row keeps what was recorded.
    assert_eq!(
        bits(&b1.force),
        bits(&evaluation(&step, "B1", "viscous_force").values)
    );
    assert_eq!(
        bits(&b1.velocity),
        bits(&evaluation(&step, "B1", "force_input_velocity").values)
    );
    assert!(b1.velocity[6].is_nan() && b1.force[32] == f64::NEG_INFINITY);
    assert_eq!(kick.b2.as_ref().unwrap().available, [true; 12]);
    assert_eq!(frame.eligible, [true; 12]);

    // A nonfinite pre-clone velocity makes the walker ineligible while velocities are read.
    let mut step = original.clone();
    step.stages[0].fields.get_mut("velocities").unwrap().values[3 * 6 + 1] = f64::INFINITY;
    let frame = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap();
    frame.validate().unwrap();
    assert_eq!(frame.eligible, masked(&[6]));
    assert_eq!(frame.v.as_ref().unwrap()[19], f64::INFINITY);
    let mut blind = capabilities.clone();
    blind.missing.insert(Record::Velocities, "declared".into());
    let frame = extract(&archive.gas_config, &step, &measurement, &blind).unwrap();
    assert!(frame.v.is_none());
    assert_eq!(frame.eligible, [true; 12]);
}

#[test]
fn short_or_misshapen_records_are_shape_errors_not_panics() {
    let mut gas = einstein_hilbert::<f64>(12, 5);
    let archive = record(&mut gas, 2, false);
    let measurement = MeasurementConfig::default();
    let capabilities = capabilities(&archive, &measurement);
    let original = &archive.steps[1];
    let refused = |tamper: &dyn Fn(&mut RecordedStep<f64>)| {
        let mut step = original.clone();
        tamper(&mut step);
        let result = extract(&archive.gas_config, &step, &measurement, &capabilities);
        assert!(matches!(result, Err(GasError::Shape(_))), "{result:?}");
    };
    refused(&|step| step.report.pre_clone_eligible.truncate(11));
    refused(&|step| step.report.pre_clone_fitness.fitness.truncate(11));
    refused(&|step| step.report.pre_clone_rewards.raw.truncate(11));
    refused(&|step| {
        let velocities = step.stages[0].fields.get_mut("velocities").unwrap();
        velocities.values.pop();
    });
    for item_shape in [vec![], vec![3, 1]] {
        refused(&|step| {
            let positions = step.stages[0].fields.get_mut("positions").unwrap();
            positions.item_shape = item_shape.clone();
        });
    }
    // A well-formed batch over eleven rows is not a record of this population.
    refused(&|step| {
        let batch = &mut step.report.distance_companions;
        batch.rows = 11;
        batch.indices.pop();
        batch.valid.pop();
    });
    refused(&|step| {
        let batch = &mut step.report.cloning_companions;
        batch.rows = 11;
        batch.indices.pop();
        batch.valid.pop();
    });
    refused(&|step| {
        evaluation_mut(step, "B1", "force_input_velocity")
            .values
            .truncate(35)
    });
    refused(&|step| {
        evaluation_mut(step, "B2", "viscous_force")
            .available
            .truncate(11)
    });

    // The velocity record of a stage has the same shape rule as its force record.
    for tamper in [
        (|e| e.item_shape = vec![2]) as fn(&mut FieldEvaluation),
        |e| e.rows = 11,
    ] {
        let mut step = original.clone();
        tamper(evaluation_mut(&mut step, "B2", "force_input_velocity"));
        let frame = extract(&archive.gas_config, &step, &measurement, &capabilities).unwrap();
        let kick = frame.kick.unwrap();
        assert!(kick.b1.is_some() && kick.b2.is_none());
    }

    // The colour source is validated before any record is read.
    for color in [
        ColorSource::ViscousForce {
            alignment: Default::default(),
            threshold: f64::NAN,
        },
        ColorSource::RecordedField {
            stage: String::new(),
            amplitude: "total_force".into(),
            phase: "force_input_velocity".into(),
            alignment: StepAlignment::default(),
            threshold: 1e-12,
        },
    ] {
        let measurement = MeasurementConfig {
            color,
            ..MeasurementConfig::default()
        };
        let result = extract(&archive.gas_config, original, &measurement, &capabilities);
        assert!(
            matches!(result, Err(GasError::Configuration(_))),
            "{result:?}"
        );
    }
}
