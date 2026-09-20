//! Streaming measurement of recorded runs: chunk invariance, checkpoint and
//! restore, warm-up calibration, coverage, the source-frozen propagator on a
//! hand-checkable record, and the budgets.
use algorithmic_gas::{
    AlgorithmicGas, ExecutionContext, GasBuilder, GasConfig, GasError, InputBatch,
    ObservationBatch, Population, Precision, Provenance, Real, RecordingConfig, RewardBatch,
    RunArchive, TensorBatch,
    boundary::{BoundaryPolicy, BoxDomain},
    domain::{GradientProvider, OperatorFuture, RewardSource},
    donor::{CompanionBatch, SourceRef},
    geometry::Distance,
    kinetic::ViscousForceConfig,
    physics::{
        numerics::{BlockMoments, Subtraction, series_moments},
        spectroscopy::{
            Accumulator, AccumulatorState, Capabilities, ChannelSeries, ChannelSpec, Element,
            ElementKind, ExchangeParity, Extensions, FieldSource, Frame, FrameState, LocalOperator,
            Measurement, MeasurementConfig, OperatorContext, Record, Requirements,
            SPECTROSCOPY_VERSION, Signature, analyze,
            config::AnalysisConfig,
            config::{
                ColorSource, CompanionChoice, EdgeLength, FlowConfig, FrameNormalization,
                GlueballObservable, IdentityPolicy, LengthScale, MesonMode, MesonQuantum, Momentum,
                MomentumPhase, PairSelection, PhaseScale, PropagatorConfig, ScaleConfig, ScaleMode,
                ScaleSelection, TimeAxis, U1Mode,
            },
            contract::Availability,
            frame::position_field,
            measure_archive, measure_archive_with,
        },
    },
    tessellation::{GeometryReward, GeometrySchedule, ZeroPotential},
    tracking::RecordedStep,
};
use futures_lite::future::block_on;

/// Deterministic cloud in `[-spread/2, spread/2]^d` with velocities in
/// `[-speed/2, speed/2]^d`.
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
fn record(gas: &mut AlgorithmicGas<f64>, steps: usize, graph: bool) -> RunArchive<f64> {
    gas.start_recording(RecordingConfig {
        max_steps: steps,
        graph,
        ..RecordingConfig::default()
    })
    .unwrap();
    for _ in 0..steps {
        block_on(gas.step()).unwrap();
    }
    gas.stop_recording().unwrap()
}
/// Einstein-Hilbert gas in three dimensions. The preset inherits `F32` from
/// `GasConfig::default`, and spectroscopy measures `f64` runs only.
fn einstein_hilbert(walkers: usize, seed: u64) -> AlgorithmicGas<f64> {
    let mut config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    config.precision = Precision::F64;
    config.seed = seed;
    config.clone_decision.every = 2;
    config.geometry.as_mut().unwrap().schedule = GeometrySchedule::EveryStage;
    block_on(
        GasBuilder::new(
            population::<f64>(walkers, 3, 1., 1.),
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
fn euclidean(
    walkers: usize,
    seed: u64,
    viscosity: Option<ViscousForceConfig>,
) -> AlgorithmicGas<f64> {
    let mut config = GasConfig::euclidean(3, 0.05).unwrap();
    config.seed = seed;
    config.qft.viscosity = viscosity;
    block_on(
        GasBuilder::new(population::<f64>(walkers, 3, 2., 1.), Quadratic)
            .gradient(Quadratic)
            .config(config)
            .build(),
    )
    .unwrap()
}
/// One reward for every walker, so every cloning score is exactly 0 and every
/// pair of every frame ties: the vanishing score dispersion of Q27.
struct Flat;
impl RewardSource<f64> for Flat {
    fn id(&self) -> String {
        "test-flat/v1".into()
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
                vec![1.; p.len()],
                Provenance {
                    population_version: p.version,
                    stage: stage.into(),
                    ..Default::default()
                },
            ))
        })
    }
}
impl GradientProvider<f64> for Flat {
    fn id(&self) -> String {
        "test-flat-gradient/v1".into()
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<f64>> {
        Box::pin(async move {
            let x = p.observations.field("positions")?;
            TensorBatch::vectors(p.len(), x.width(), vec![0.; x.values().len()])
        })
    }
}
fn flat(walkers: usize, seed: u64, viscosity: Option<ViscousForceConfig>) -> AlgorithmicGas<f64> {
    let mut config = GasConfig::euclidean(3, 0.05).unwrap();
    config.seed = seed;
    config.qft.viscosity = viscosity;
    // One reward and no diversity term: every fitness of every frame is the
    // same number, so every cloning score is 0.
    config.fitness.diversity_exponent = 0.;
    block_on(
        GasBuilder::new(population::<f64>(walkers, 3, 2., 1.), Flat)
            .gradient(Flat)
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
fn dimension(archive: &RunArchive<f64>) -> usize {
    archive.anchors[0]
        .population
        .observations
        .field(position_field(&archive.gas_config))
        .unwrap()
        .width()
}
fn accumulator(config: &MeasurementConfig, archive: &RunArchive<f64>) -> Accumulator {
    Accumulator::new(
        config.clone(),
        &archive.gas_config,
        &archive.config,
        dimension(archive),
    )
    .unwrap()
}
fn accumulator_with(
    config: &MeasurementConfig,
    archive: &RunArchive<f64>,
    extensions: Extensions,
) -> Accumulator {
    Accumulator::with_extensions(
        config.clone(),
        &archive.gas_config,
        &archive.config,
        dimension(archive),
        extensions,
    )
    .unwrap()
}
/// A measurement configuration without a warm-up: the phase length is fixed,
/// so nothing has to be calibrated from frames.
fn immediate(max_lag: usize) -> MeasurementConfig {
    MeasurementConfig {
        warmup: 0,
        max_lag,
        phase: PhaseScale {
            length: LengthScale::Fixed { value: 1. },
            ..PhaseScale::default()
        },
        ..MeasurementConfig::default()
    }
}

/// One walker value per frame, frozen into the positions: a noiseless AR(1)
/// `x_t = rho x_{t-1}` whose decay rate is `-ln rho`.
struct Geometric {
    rho: f64,
    base: u64,
}
impl LocalOperator for Geometric {
    fn id(&self) -> String {
        "custom/geometric".into()
    }
    fn signature(
        &self,
        _: ElementKind,
        _: &OperatorContext<'_>,
    ) -> algorithmic_gas::Result<Signature> {
        Ok(Signature::new(
            Requirements::default(),
            1,
            ExchangeParity::Even,
        ))
    }
    fn evaluate(
        &self,
        _: &Element,
        state: &FrameState,
        _: &OperatorContext<'_>,
        out: &mut [f64],
    ) -> bool {
        out[0] = self.rho.powi((state.step - self.base) as i32);
        true
    }
}

/// The six-frame record of the estimators dossier: three walkers, one
/// distance companion each, a step gap between frames 3 and 4, an invalid
/// matter field on walker 2 at frame 1 and two accepted clones.
const STEPS: [u64; 6] = [10, 11, 12, 13, 15, 16];
const VALUES: [[f64; 3]; 6] = [
    [1., 2., 3.],
    [2., 1., 0.],
    [1., 3., 2.],
    [0., 2., 1.],
    [3., 1., 2.],
    [1., 1., 4.],
];
const FIELD: [[bool; 3]; 6] = [
    [true, true, true],
    [true, true, false],
    [true, true, true],
    [true, true, true],
    [true, true, true],
    [true, true, true],
];
const COMPANION: [[u32; 3]; 6] = [
    [1, 2, 0],
    [1, 0, 2],
    [2, 2, 0],
    [1, 0, 1],
    [2, 0, 1],
    [1, 2, 0],
];
const GENERATION: [[u64; 3]; 6] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
];
const CLONED: [[bool; 3]; 6] = [
    [false, false, false],
    [false, true, false],
    [false, false, false],
    [false, false, false],
    [true, false, false],
    [false, false, false],
];
/// Marks the walkers of `FIELD` valid; an injected source provides the matter
/// field whatever the gas records.
struct Tabulated;
impl FieldSource for Tabulated {
    fn id(&self) -> String {
        "tabulated/v1".into()
    }
    fn requires(&self) -> Requirements {
        Requirements::default()
    }
    fn fill(
        &self,
        _: Option<&Frame>,
        frame: &Frame,
        _: f64,
        state: &mut FrameState,
    ) -> Availability {
        let Some(index) = STEPS.iter().position(|&step| step == frame.step) else {
            return Availability::unavailable("step outside the table");
        };
        state.color_valid = FIELD[index].to_vec();
        state.force_valid = FIELD[index].to_vec();
        Availability::Available
    }
}
/// `O(i, j) = u_i u_j` on the tabulated values, masked where the field is
/// invalid: the exchange-even scalar stand-in of the dossier.
struct Pair;
impl LocalOperator for Pair {
    fn id(&self) -> String {
        "custom/pair".into()
    }
    fn signature(
        &self,
        _: ElementKind,
        _: &OperatorContext<'_>,
    ) -> algorithmic_gas::Result<Signature> {
        Ok(Signature::new(
            Requirements::new([Record::Color]),
            1,
            ExchangeParity::Even,
        ))
    }
    fn evaluate(
        &self,
        element: &Element,
        state: &FrameState,
        _: &OperatorContext<'_>,
        out: &mut [f64],
    ) -> bool {
        let [i, j, _] = element.walkers.map(|w| w as usize);
        if !state.color_valid[i] || !state.color_valid[j] {
            return false;
        }
        out[0] = state.x[i * state.d] * state.x[j * state.d];
        true
    }
}
/// Rewrite one recorded step into a frame of a tabulated record: its step
/// number, the first position coordinate of every walker, the generations and
/// the distance companions. `companions` is `[walkers, count]` of slots, whose
/// sources name the preceding step. The frame stays the frame of an executed
/// run.
fn rewrite(
    step: &mut RecordedStep<f64>,
    field: &str,
    number: u64,
    values: &[f64],
    generations: &[u64],
    companions: &[u32],
) {
    let n = values.len();
    step.epoch = 0;
    step.report.step = number;
    for stage in &mut step.stages {
        stage.generations = generations.to_vec();
        if let Some(snapshot) = stage.fields.get_mut(field) {
            let d = snapshot.item_shape[0];
            for (walker, row) in snapshot.values.chunks_exact_mut(d).enumerate() {
                row[0] = values[walker];
            }
        }
    }
    for evaluation in &mut step.field_evaluations {
        if evaluation.field == field {
            let d = evaluation.item_shape[0];
            for (walker, row) in evaluation.values.chunks_exact_mut(d).enumerate() {
                row[0] = values[walker];
            }
        }
    }
    step.report.pre_clone_eligible = vec![true; n];
    step.report.distance_sources = (0..n as u32)
        .map(|slot| SourceRef {
            frame: number - 1,
            slot,
            generation: generations[slot as usize],
            version: 1,
        })
        .collect();
    step.report.distance_companions = CompanionBatch {
        rows: n,
        count: companions.len() / n,
        indices: companions.to_vec(),
        valid: vec![true; companions.len()],
        mutual: false,
    };
}
/// Rewrite a real six-step archive into the record of the dossier: step
/// numbers, walker values, distance companions, generations and clones.
fn tabulate(archive: &mut RunArchive<f64>) {
    let field = position_field(&archive.gas_config).to_string();
    for (index, step) in archive.steps.iter_mut().enumerate() {
        rewrite(
            step,
            &field,
            STEPS[index],
            &VALUES[index],
            &GENERATION[index],
            &COMPANION[index],
        );
        for (choice, &cloned) in step
            .report
            .clone_plan
            .choices
            .iter_mut()
            .zip(&CLONED[index])
        {
            choice.accepted = cloned;
            choice.revival = false;
        }
    }
}
/// `O(i, j) = (u_i − u_j)³` on the tabulated values: an exchange-odd pair
/// operator that does not telescope over a permutation companion map.
struct OddPair;
impl LocalOperator for OddPair {
    fn id(&self) -> String {
        "custom/odd".into()
    }
    fn signature(
        &self,
        _: ElementKind,
        _: &OperatorContext<'_>,
    ) -> algorithmic_gas::Result<Signature> {
        Ok(Signature::new(
            Requirements::new([Record::Color]),
            1,
            ExchangeParity::Odd,
        ))
    }
    fn evaluate(
        &self,
        element: &Element,
        state: &FrameState,
        _: &OperatorContext<'_>,
        out: &mut [f64],
    ) -> bool {
        let [i, j, _] = element.walkers.map(|w| w as usize);
        if !state.color_valid[i] || !state.color_valid[j] {
            return false;
        }
        let gap = state.x[i * state.d] - state.x[j * state.d];
        out[0] = gap * gap * gap;
        true
    }
}
/// Refuses the record on the third frame and marks every walker invalid on
/// the fifth: the two ways a frame carries no frame average.
struct Patchy;
impl FieldSource for Patchy {
    fn id(&self) -> String {
        "patchy/v1".into()
    }
    fn requires(&self) -> Requirements {
        Requirements::default()
    }
    fn fill(
        &self,
        _: Option<&Frame>,
        frame: &Frame,
        _: f64,
        state: &mut FrameState,
    ) -> Availability {
        if frame.step == STEPS[2] {
            return Availability::unavailable("the record of this step is missing");
        }
        state.color_valid = vec![frame.step != STEPS[4]; frame.n];
        state.force_valid = state.color_valid.clone();
        Availability::Available
    }
}
/// Marks every walker valid, whatever the gas records.
struct Everything;
impl FieldSource for Everything {
    fn id(&self) -> String {
        "everything/v1".into()
    }
    fn requires(&self) -> Requirements {
        Requirements::default()
    }
    fn fill(
        &self,
        _: Option<&Frame>,
        frame: &Frame,
        _: f64,
        state: &mut FrameState,
    ) -> Availability {
        state.color_valid = vec![true; frame.n];
        state.force_valid = state.color_valid.clone();
        Availability::Available
    }
}
/// Marks walker 1 invalid on every frame: a field whose validity breaks the
/// symmetry of a pair.
struct Lopsided;
impl FieldSource for Lopsided {
    fn id(&self) -> String {
        "lopsided/v1".into()
    }
    fn requires(&self) -> Requirements {
        Requirements::default()
    }
    fn fill(
        &self,
        _: Option<&Frame>,
        frame: &Frame,
        _: f64,
        state: &mut FrameState,
    ) -> Availability {
        state.color_valid = (0..frame.n).map(|i| i != 1).collect();
        state.force_valid = state.color_valid.clone();
        Availability::Available
    }
}
/// `O(i, j) = (u_i − u_j)³` masked by the field of the anchor alone: an
/// exchange-odd operator that can be valid where its mirror is not.
struct AnchorOdd;
impl LocalOperator for AnchorOdd {
    fn id(&self) -> String {
        "custom/anchor".into()
    }
    fn signature(
        &self,
        _: ElementKind,
        _: &OperatorContext<'_>,
    ) -> algorithmic_gas::Result<Signature> {
        Ok(Signature::new(
            Requirements::new([Record::Color]),
            1,
            ExchangeParity::Odd,
        ))
    }
    fn evaluate(
        &self,
        element: &Element,
        state: &FrameState,
        _: &OperatorContext<'_>,
        out: &mut [f64],
    ) -> bool {
        let [i, j, _] = element.walkers.map(|w| w as usize);
        if !state.color_valid[i] {
            return false;
        }
        let gap = state.x[i * state.d] - state.x[j * state.d];
        out[0] = gap * gap * gap;
        true
    }
}
/// Claims a valid readout everywhere and returns a nonfinite value on two of
/// the three walkers.
struct Poisoned;
impl LocalOperator for Poisoned {
    fn id(&self) -> String {
        "custom/poisoned".into()
    }
    fn signature(
        &self,
        _: ElementKind,
        _: &OperatorContext<'_>,
    ) -> algorithmic_gas::Result<Signature> {
        Ok(Signature::new(
            Requirements::default(),
            1,
            ExchangeParity::Even,
        ))
    }
    fn evaluate(
        &self,
        element: &Element,
        _: &FrameState,
        _: &OperatorContext<'_>,
        out: &mut [f64],
    ) -> bool {
        out[0] = match element.walkers[0] {
            1 => f64::NAN,
            2 => f64::INFINITY,
            _ => 1.,
        };
        true
    }
}
/// Two distance companions per walker, the second of walker 1 its own slot:
/// under `CompanionChoice::All` walker 1 carries one element of weight 1 and
/// each of the other anchors two elements of weight 1/2.
const WEIGHTED_STEPS: [u64; 2] = [20, 21];
const WEIGHTED: [[f64; 3]; 2] = [[1., 2., 3.], [2., 3., 1.]];
const WEIGHTED_COMPANION: [u32; 6] = [1, 2, 0, 1, 0, 1];
/// The two-frame record of unequal companion counts, measured on distance
/// pairs with the injected operator of the dossier.
fn weighted() -> Measurement {
    let mut gas = euclidean(3, 7, None);
    let mut archive = record(&mut gas, 2, false);
    let field = position_field(&archive.gas_config).to_string();
    for (index, step) in archive.steps.iter_mut().enumerate() {
        rewrite(
            step,
            &field,
            WEIGHTED_STEPS[index],
            &WEIGHTED[index],
            &[0; 3],
            &WEIGHTED_COMPANION,
        );
        for choice in &mut step.report.clone_plan.choices {
            choice.accepted = false;
            choice.revival = false;
        }
    }
    let config = MeasurementConfig {
        companions: CompanionChoice::All,
        max_lag: 1,
        ..dossier_config(IdentityPolicy::Slot)
    };
    let extensions = Extensions::default()
        .operator(ElementKind::DistancePair, Pair)
        .field(Everything);
    let mut accumulator = accumulator_with(&config, &archive, extensions);
    accumulator.ingest_archive(&archive).unwrap();
    accumulator.into_measurement().unwrap()
}
/// No built-in channel, distance pairs and a propagator on whatever is
/// injected: the configuration the dossier record is measured with.
fn dossier_config(identity: IdentityPolicy) -> MeasurementConfig {
    MeasurementConfig {
        channels: vec![],
        pairs: PairSelection::Distance,
        identity,
        propagators: PropagatorConfig {
            enabled_for: vec!["*".into()],
            ..PropagatorConfig::default()
        },
        ..immediate(3)
    }
}
/// The dossier record on a Euclidean-time axis: the first coordinate, two
/// bins over `[0, 4]` and no propagator, so only the slabs are retained.
fn slab_config(max_blocks: usize) -> MeasurementConfig {
    MeasurementConfig {
        time: TimeAxis::Euclidean {
            axis: Some(0),
            bins: 2,
            range: Some([0., 4.]),
            periodic: false,
        },
        propagators: PropagatorConfig {
            enabled_for: vec![],
            max_blocks,
            ..PropagatorConfig::default()
        },
        ..dossier_config(IdentityPolicy::Slot)
    }
}
fn slab_extensions() -> Extensions {
    Extensions::default()
        .operator(ElementKind::DistancePair, Pair)
        .field(Tabulated)
}
/// The dossier record measured with the injected components of `extensions`.
fn dossier_with(config: MeasurementConfig, extensions: Extensions) -> Measurement {
    let mut gas = euclidean(3, 7, None);
    let mut archive = record(&mut gas, 6, false);
    tabulate(&mut archive);
    let mut accumulator = accumulator_with(&config, &archive, extensions);
    accumulator.ingest_archive(&archive).unwrap();
    accumulator.into_measurement().unwrap()
}
/// The dossier record measured with one injected pair operator.
fn dossier(identity: IdentityPolicy) -> Measurement {
    dossier_with(
        dossier_config(identity),
        Extensions::default()
            .operator(ElementKind::DistancePair, Pair)
            .field(Tabulated),
    )
}

#[test]
fn one_archive_and_eight_chunks_of_it_measure_the_same_bits() {
    let config = MeasurementConfig {
        warmup: 4,
        max_lag: 6,
        ..MeasurementConfig::default()
    };
    let whole = measure_archive(&config, &record(&mut einstein_hilbert(8, 7), 64, true)).unwrap();
    let mut gas = einstein_hilbert(8, 7);
    let first = record(&mut gas, 8, true);
    let mut accumulator = accumulator(&config, &first);
    accumulator.ingest_archive(&first).unwrap();
    for _ in 1..8 {
        let chunk = record(&mut gas, 8, true);
        accumulator.ingest_archive(&chunk).unwrap();
    }
    let chunked = accumulator.into_measurement().unwrap();
    assert_eq!(whole.frames(), 60);
    assert_eq!(whole.segments, 1);
    assert_eq!(whole.ingested, 64);
    assert_eq!(whole.to_bytes().unwrap(), chunked.to_bytes().unwrap());
}

#[test]
fn a_checkpoint_in_the_middle_of_a_stream_continues_bit_for_bit() {
    let config = MeasurementConfig {
        warmup: 4,
        max_lag: 6,
        ..MeasurementConfig::default()
    };
    let archive = record(&mut einstein_hilbert(8, 7), 32, true);
    let uninterrupted = measure_archive(&config, &archive).unwrap();
    let mut accumulator = accumulator(&config, &archive);
    for step in &archive.steps[..17] {
        accumulator.ingest_step(step).unwrap();
    }
    let bytes = accumulator.state().to_bytes().unwrap();
    let state = AccumulatorState::from_bytes(&bytes).unwrap();
    assert_eq!(state.schema_version, SPECTROSCOPY_VERSION);
    assert_eq!(state.measurement.frames(), 13);
    let mut restored = Accumulator::restore(state).unwrap();
    for step in &archive.steps[17..] {
        restored.ingest_step(step).unwrap();
    }
    let continued = restored.into_measurement().unwrap();
    assert_eq!(
        uninterrupted.to_bytes().unwrap(),
        continued.to_bytes().unwrap()
    );
    assert_eq!(
        AccumulatorState::from_bytes(&[&bytes[..], &[0u8]].concat())
            .unwrap_err()
            .to_string(),
        GasError::Configuration("trailing accumulator state data".into()).to_string()
    );
}

#[test]
fn a_restored_stream_rejects_other_injected_components() {
    let archive = record(&mut einstein_hilbert(6, 7), 4, true);
    let config = immediate(3);
    let mut accumulator = accumulator(&config, &archive);
    accumulator.ingest_archive(&archive).unwrap();
    let state = accumulator.state();
    assert!(Accumulator::restore(state.clone()).is_ok());
    let injected =
        Extensions::default().operator(ElementKind::Site, Geometric { rho: 0.9, base: 1 });
    assert!(matches!(
        Accumulator::restore_with(state.clone(), injected),
        Err(GasError::Checkpoint(_))
    ));
    let mut version = state.clone();
    version.schema_version += 1;
    assert!(matches!(
        Accumulator::restore(version),
        Err(GasError::Checkpoint(_))
    ));
    // A frozen stream without its calibration measures other frames than the
    // ring it carries: refused before any operator reads it.
    let mut uncalibrated = state;
    uncalibrated.measurement.calibration = None;
    assert!(matches!(
        Accumulator::restore(uncalibrated),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn merging_propagator_blocks_equals_coarsening_them_in_pairs() {
    let config = MeasurementConfig {
        propagators: PropagatorConfig {
            enabled_for: vec!["*".into()],
            ..PropagatorConfig::default()
        },
        ..immediate(4)
    };
    let archive = record(&mut einstein_hilbert(6, 7), 24, true);
    let measurement = measure_archive(&config, &archive).unwrap();
    let moments = measurement
        .channels
        .iter()
        .find_map(|channel| channel.propagator.as_ref())
        .unwrap();
    assert_eq!(moments.blocks, 24);
    let mut merged = moments.clone();
    merged.merge_pairs();
    let coarse = moments.coarsen(2).unwrap();
    assert_eq!(merged.blocks, coarse.blocks);
    assert_eq!(merged.origins_per_block, coarse.origins_per_block);
    assert_eq!(merged.ab, coarse.ab);
    assert_eq!(merged.a, coarse.a);
    assert_eq!(merged.b, coarse.b);
    assert_eq!(merged.n, coarse.n);
    let capped = MeasurementConfig {
        propagators: PropagatorConfig {
            max_blocks: 4,
            ..config.propagators.clone()
        },
        ..config
    };
    let bounded = measure_archive(&capped, &archive).unwrap();
    let small = bounded
        .channels
        .iter()
        .find_map(|channel| channel.propagator.as_ref())
        .unwrap();
    assert!(small.blocks <= 4 && small.origins_per_block == 8);
    let (a, b) = (
        moments.estimate(Subtraction::LagMeans),
        small.estimate(Subtraction::LagMeans),
    );
    for (a, b) in a.iter().zip(&b) {
        match (a, b) {
            (Some(a), Some(b)) => assert!((a - b).abs() <= 1e-9 * (1. + a.abs())),
            (a, b) => assert_eq!(a, b),
        }
    }
}

#[test]
fn an_injected_noiseless_autoregressive_operator_recovers_its_decay_rate() {
    let (rho, frames) = (0.9, 32usize);
    let archive = record(&mut einstein_hilbert(4, 7), frames, true);
    let base = archive.steps[0].report.step;
    let config = MeasurementConfig {
        channels: vec![],
        propagators: PropagatorConfig {
            enabled_for: vec![],
            ..PropagatorConfig::default()
        },
        ..immediate(4)
    };
    let extensions = Extensions::default().operator(ElementKind::Site, Geometric { rho, base });
    let measurement = measure_archive_with(&config, &archive, extensions).unwrap();
    let channel = measurement.channel("custom/geometric/site").unwrap();
    assert!(channel.availability.is_available() && channel.weight == vec![1.; frames]);
    let moments = series_moments(measurement.series(channel), 4, 1).unwrap();
    let raw = moments.estimate(Subtraction::None);
    // Identity 5 of the estimator dossier for x_t = rho^t over T frames.
    let closed = |lag: usize| {
        let count = (frames - lag) as f64;
        rho.powi(lag as i32) * (1. - rho.powi(2 * (frames - lag) as i32))
            / ((1. - rho * rho) * count)
    };
    for (lag, value) in raw.iter().enumerate() {
        assert!((value.unwrap() - closed(lag)).abs() < 1e-12 * closed(0));
    }
    // The finite-record correction is of order rho^{2T} = 1.2e-3 at T = 32.
    let ratio = raw[1].unwrap() * (frames - 1) as f64 / (raw[0].unwrap() * frames as f64);
    assert!((-ratio.ln() - 0.105360515657826).abs() < 2e-3);
}

#[test]
fn the_source_frozen_propagator_reproduces_the_hand_computed_six_frame_record() {
    let measurement = dossier(IdentityPolicy::Slot);
    assert_eq!(measurement.frames(), 6);
    assert_eq!(measurement.segments, 2);
    assert_eq!(measurement.segment, vec![0, 0, 0, 0, 1, 1]);
    assert_eq!(measurement.steps, STEPS.to_vec());
    let channel = measurement.channel("custom/pair/distance").unwrap();
    assert_eq!(channel.coverage.valid, 17);
    assert_eq!(channel.coverage.masked_self, 1);
    let moments = channel.propagator.as_ref().unwrap();
    assert_eq!(moments.blocks, 6);
    let pooled = moments.pooled(None);
    assert_eq!(pooled.n, vec![17., 9., 5., 3.]);
    assert_eq!(pooled.ab, vec![187., 63., 48., 12.]);
    assert_eq!(pooled.a, vec![47., 27., 15., 11.]);
    assert_eq!(pooled.b, vec![47., 19., 11., 2.]);
    let raw = moments.estimate(Subtraction::None);
    assert_eq!(raw, vec![Some(11.), Some(7.), Some(9.6), Some(4.)]);
    let connected = moments.estimate(Subtraction::LagMeans);
    let expected = [970. / 289., 2. / 3., 3., 14. / 9.];
    for (value, expected) in connected.iter().zip(&expected) {
        assert!((value.unwrap() - expected).abs() < 1e-12);
    }
    let global = moments.estimate(Subtraction::GlobalMean);
    let book = [
        3.35640138408304,
        0.512879661668589,
        2.86712802768166,
        -0.33679354094579,
    ];
    for (value, expected) in global.iter().zip(&book) {
        assert!((value.unwrap() - expected).abs() < 1e-12);
    }
}

#[test]
fn the_incarnation_identity_masks_every_sink_whose_generation_changed() {
    let measurement = dossier(IdentityPolicy::Incarnation);
    let channel = measurement.channel("custom/pair/distance").unwrap();
    let moments = channel.propagator.as_ref().unwrap();
    let pooled = moments.pooled(None);
    assert_eq!(pooled.n, vec![17., 5., 1., 1.]);
    assert_eq!(pooled.ab, vec![187., 24., 6., 0.]);
    assert_eq!(pooled.a, vec![47., 14., 3., 3.]);
    assert_eq!(pooled.b, vec![47., 8., 2., 0.]);
    let slot = dossier(IdentityPolicy::Slot);
    let reference = slot.channel("custom/pair/distance").unwrap();
    let wide = reference.propagator.as_ref().unwrap().pooled(None);
    assert!(pooled.n.iter().zip(&wide.n).all(|(a, b)| a <= b));
    assert!(channel.coverage.masked_identity > 0);
    assert_eq!(reference.coverage.masked_identity, 0);
}

#[test]
fn every_frame_of_the_six_frame_record_carries_a_valid_count_average() {
    let measurement = dossier(IdentityPolicy::Slot);
    let channel = measurement.channel("custom/pair/distance").unwrap();
    let expected = [11. / 3., 2., 10. / 3., 2. / 3., 11. / 3., 3.];
    for (value, expected) in channel.values.iter().zip(&expected) {
        assert!((value - expected).abs() < 1e-12);
    }
    assert_eq!(channel.weight, vec![1.; 6]);
    assert_eq!(channel.coverage.empty_frames, 0);
    assert_eq!(channel.coverage.frames, 6);
    let moments = series_moments(measurement.series(channel), 3, 1).unwrap();
    let raw = moments.estimate(Subtraction::None);
    let dossier = [463. / 54., 245. / 36., 61. / 9., 22. / 9.];
    for (value, expected) in raw.iter().zip(&dossier) {
        assert!((value.unwrap() - expected).abs() < 1e-12);
    }
}

#[test]
fn an_exchange_odd_operator_has_no_frame_mean_where_every_valid_pair_is_mirrored() {
    let injected = || {
        Extensions::default()
            .operator(ElementKind::DistancePair, OddPair)
            .field(Tabulated)
    };
    let measurement = dossier_with(dossier_config(IdentityPolicy::Slot), injected());
    let channel = measurement.channel("custom/odd/distance").unwrap();
    assert_eq!(channel.exchange, ExchangeParity::Odd);
    // Frame 1 holds only the pairs (0,1) and (1,0), which are each other's
    // mirror: their exact cancellation is no value of the operator.
    assert_eq!(channel.involutive_frames, 1);
    assert_eq!(channel.weight, vec![1., 0., 1., 1., 1., 1.]);
    assert_eq!(channel.values, vec![2., 0., 1., -1., -2., 0.]);
    // Coverage and the propagator keep the mirrored elements.
    assert_eq!(channel.coverage.valid, 17);
    assert_eq!(channel.coverage.empty_frames, 0);
    assert_eq!(channel.propagator.as_ref().unwrap().pooled(None).n[0], 17.);
    let fixed = dossier_with(
        MeasurementConfig {
            normalization: FrameNormalization::FixedN,
            ..dossier_config(IdentityPolicy::Slot)
        },
        injected(),
    );
    let channel = fixed.channel("custom/odd/distance").unwrap();
    assert_eq!(channel.weight, vec![1., 0., 1., 1., 1., 1.]);
    let expected = [2., 0., 1. / 3., -1. / 3., -2., 0.];
    for (value, expected) in channel.values.iter().zip(&expected) {
        assert!((value - expected).abs() < 1e-15);
    }
}

#[test]
fn a_frame_the_record_cannot_evaluate_is_missing_data_under_both_normalisations() {
    let injected = || {
        Extensions::default()
            .operator(ElementKind::DistancePair, Pair)
            .field(Patchy)
    };
    let measurement = dossier_with(dossier_config(IdentityPolicy::Slot), injected());
    let channel = measurement.channel("custom/pair/distance").unwrap();
    // Frame 2 carries no record at all, frame 4 an empty readout: under the
    // valid-count average both leave the frame out.
    assert_eq!(channel.weight, vec![1., 1., 0., 1., 0., 1.]);
    assert_eq!(channel.coverage.empty_frames, 2);
    assert_eq!(channel.coverage.frames, 4);
    let fixed = dossier_with(
        MeasurementConfig {
            normalization: FrameNormalization::FixedN,
            ..dossier_config(IdentityPolicy::Slot)
        },
        injected(),
    );
    let channel = fixed.channel("custom/pair/distance").unwrap();
    // The fixed 1/N average of an empty readout is the value 0; the frame
    // whose record is missing stays missing.
    assert_eq!(channel.weight, vec![1., 1., 0., 1., 1., 1.]);
    assert_eq!(channel.values[4], 0.);
    assert_eq!(channel.coverage.empty_frames, 2);
}

#[test]
fn the_fixed_normalisation_divides_the_same_sums_by_the_slot_count() {
    let fixed = dossier_with(
        MeasurementConfig {
            normalization: FrameNormalization::FixedN,
            ..dossier_config(IdentityPolicy::Slot)
        },
        Extensions::default()
            .operator(ElementKind::DistancePair, Pair)
            .field(Tabulated),
    );
    let channel = fixed.channel("custom/pair/distance").unwrap();
    assert_eq!(channel.weight, vec![1.; 6]);
    let expected = [11. / 3., 4. / 3., 10. / 3., 2. / 3., 11. / 3., 3.];
    for (value, expected) in channel.values.iter().zip(&expected) {
        assert!((value - expected).abs() < 1e-12);
    }
    // Identity between the two normalisations, frame by frame: the fixed
    // average is the valid-count average times the valid share of the slots.
    let counted = dossier(IdentityPolicy::Slot);
    let valid = counted.channel("custom/pair/distance").unwrap();
    let share = [1., 2. / 3., 1., 1., 1., 1.];
    for ((fixed, counted), share) in channel.values.iter().zip(&valid.values).zip(&share) {
        assert!((fixed - counted * share).abs() < 1e-12);
    }
}

#[test]
fn a_measurement_survives_a_byte_round_trip_and_refuses_a_tampered_one() {
    let measurement = dossier(IdentityPolicy::Slot);
    let bytes = measurement.to_bytes().unwrap();
    assert_eq!(Measurement::from_bytes(&bytes).unwrap(), measurement);
    assert!(matches!(
        Measurement::from_bytes(&[&bytes[..], &[0u8]].concat()),
        Err(GasError::Configuration(_))
    ));
    let mut version = measurement.clone();
    version.schema_version += 1;
    assert!(matches!(version.to_bytes(), Err(GasError::Checkpoint(_))));
    let mut shape = measurement;
    shape.steps.push(99);
    assert!(matches!(shape.to_bytes(), Err(GasError::Configuration(_))));
}

#[test]
fn a_euclidean_gas_reports_why_the_colour_channels_are_unavailable_and_measures_the_rest() {
    let mut gas = euclidean(8, 7, None);
    let archive = record(&mut gas, 12, false);
    let config = MeasurementConfig {
        warmup: 4,
        max_lag: 4,
        ..MeasurementConfig::default()
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    let colored = measurement
        .channel("meson/scalar/standard/distance")
        .unwrap();
    assert_eq!(
        colored.availability.reason(),
        Some(
            "viscous force is identically zero: configure qft.viscosity or qft.graph_viscosity, \
             or select an explicit RecordedField colour source"
        )
    );
    assert!(colored.values.is_empty() && colored.weight.is_empty());
    let fitness = measurement.channel("fitness_phase/site").unwrap();
    assert!(fitness.availability.is_available());
    assert_eq!(fitness.weight.len(), measurement.frames());
    assert_eq!(measurement.frames(), 8);
    let calibration = measurement.calibration.as_ref().unwrap();
    assert_eq!(calibration.warmup_frames, 4);
    assert!(calibration.length > 0. && calibration.kappa > 0.);
    assert!(
        calibration
            .length_source
            .starts_with("warmup_companion_median")
    );
    assert_eq!(calibration.epsilon_d, Some(2.));
    assert_eq!(calibration.viscous_kernel_second_moment, None);
    measurement.validate().unwrap();
}

#[test]
fn a_viscous_euclidean_run_measures_colour_channels_and_both_pair_statistics() {
    let mut gas = euclidean(6, 7, Some(unit_viscosity()));
    let archive = record(&mut gas, 14, false);
    let config = MeasurementConfig {
        warmup: 4,
        max_lag: 4,
        time: TimeAxis::Euclidean {
            axis: Some(2),
            bins: 4,
            range: None,
            periodic: false,
        },
        ..MeasurementConfig::default()
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    measurement.validate().unwrap();
    let meson = measurement
        .channel("meson/scalar/standard/distance")
        .unwrap();
    assert!(meson.availability.is_available());
    assert_eq!(meson.weight.len(), measurement.frames());
    assert!(meson.coverage.valid > 0);
    let slabs = meson.euclidean.as_ref().unwrap();
    slabs.validate().unwrap();
    assert_eq!(slabs.bins, 4);
    assert_eq!(slabs.blocks, measurement.frames());
    let calibration = measurement.calibration.as_ref().unwrap();
    let moment = calibration.viscous_kernel_second_moment.unwrap();
    assert!(moment > 0. && moment <= 1.);
    assert!(calibration.pair_weight_n1.unwrap() > 0.);
    assert!(calibration.euclidean_range.unwrap()[0] < calibration.euclidean_range.unwrap()[1]);
    assert!(calibration.phase_wrapping.unwrap() >= 0.);
    let pseudoscalar = measurement
        .channel("meson/pseudoscalar/standard/distance")
        .unwrap();
    assert_eq!(pseudoscalar.exchange, ExchangeParity::Odd);
    assert!(pseudoscalar.propagator.is_some());
}

#[test]
fn the_warm_up_is_skipped_and_the_calibration_is_frozen_before_the_first_frame() {
    let archive = record(&mut einstein_hilbert(6, 7), 10, true);
    let config = MeasurementConfig {
        warmup: 6,
        max_lag: 2,
        stride: 2,
        ..MeasurementConfig::default()
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    let steps: Vec<u64> = archive.steps[6..]
        .iter()
        .step_by(2)
        .map(|step| step.report.step)
        .collect();
    assert_eq!(measurement.steps, steps);
    assert_eq!(measurement.ingested, 10);
    let calibration = measurement.calibration.as_ref().unwrap();
    assert_eq!(calibration.warmup_frames, 6);
    // The same six warm-up frames calibrate whatever follows them.
    let shorter = measure_archive(
        &config,
        &RunArchive {
            steps: archive.steps[..8].to_vec(),
            ..archive.clone()
        },
    )
    .unwrap();
    assert_eq!(
        shorter.calibration.as_ref().unwrap().length,
        calibration.length
    );
}

#[test]
fn a_gap_between_two_archives_opens_a_segment_and_no_lag_spans_it() {
    let config = MeasurementConfig {
        propagators: PropagatorConfig {
            enabled_for: vec!["*".into()],
            ..PropagatorConfig::default()
        },
        ..immediate(3)
    };
    let mut gas = einstein_hilbert(6, 7);
    let first = record(&mut gas, 6, true);
    block_on(gas.step()).unwrap();
    let second = record(&mut gas, 6, true);
    let mut accumulator = accumulator(&config, &first);
    accumulator.ingest_archive(&first).unwrap();
    accumulator.ingest_archive(&second).unwrap();
    let measurement = accumulator.into_measurement().unwrap();
    assert_eq!(measurement.segments, 2);
    assert_eq!(
        measurement.segment,
        vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    );
    let contiguous =
        measure_archive(&config, &record(&mut einstein_hilbert(6, 7), 12, true)).unwrap();
    assert_eq!(contiguous.segments, 1);
    let lagged = |measurement: &Measurement| {
        measurement
            .channels
            .iter()
            .filter_map(|channel| channel.propagator.as_ref())
            .map(|moments| moments.pooled(None).n[3])
            .sum::<f64>()
    };
    assert!(lagged(&measurement) < lagged(&contiguous));
}

#[test]
fn a_budget_that_cannot_hold_the_run_is_an_error_and_not_a_panic() {
    let archive = record(&mut einstein_hilbert(6, 7), 6, true);
    let tiny = MeasurementConfig {
        budget: algorithmic_gas::physics::spectroscopy::config::MeasurementBudget {
            max_frames: 20_000,
            max_bytes: 4096,
        },
        ..immediate(3)
    };
    assert!(matches!(
        measure_archive(&tiny, &archive),
        Err(GasError::Capability(_))
    ));
    let short = MeasurementConfig {
        budget: algorithmic_gas::physics::spectroscopy::config::MeasurementBudget {
            max_frames: 4,
            max_bytes: 64 * 1024 * 1024,
        },
        ..immediate(3)
    };
    let mut accumulator = accumulator(&short, &archive);
    let outcome = accumulator.ingest_archive(&archive);
    assert!(matches!(outcome, Err(GasError::Configuration(_))));
    let measurement = accumulator.into_measurement().unwrap();
    assert_eq!(measurement.frames(), 4);
    assert_eq!(measurement.ingested, 4);
}

#[test]
fn a_measurement_without_an_available_channel_and_a_zero_warm_up_are_refused() {
    let archive = record(&mut euclidean(4, 7, None), 3, false);
    let colored = MeasurementConfig {
        channels: vec![ChannelSpec::Meson {
            quantum: algorithmic_gas::physics::spectroscopy::config::MesonQuantum::Scalar,
            mode: algorithmic_gas::physics::spectroscopy::config::MesonMode::Standard,
        }],
        ..immediate(2)
    };
    assert!(matches!(
        measure_archive(&colored, &archive),
        Err(GasError::Capability(_))
    ));
    let uncalibrated = MeasurementConfig {
        warmup: 0,
        ..MeasurementConfig::default()
    };
    assert!(matches!(
        measure_archive(&uncalibrated, &archive),
        Err(GasError::Configuration(_))
    ));
    let empty = MeasurementConfig {
        channels: vec![],
        ..immediate(2)
    };
    assert!(matches!(
        measure_archive(&empty, &archive),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn an_injected_calibration_replaces_the_warm_up_without_changing_the_fingerprint() {
    let archive = record(&mut euclidean(6, 7, Some(unit_viscosity())), 10, false);
    let config = MeasurementConfig {
        warmup: 4,
        max_lag: 3,
        ..MeasurementConfig::default()
    };
    let reference = measure_archive(&config, &archive).unwrap();
    let calibration = reference.calibration.clone().unwrap();
    let pooled = measure_archive_with(
        &config,
        &archive,
        Extensions::default().calibration(calibration.clone()),
    )
    .unwrap();
    assert_eq!(pooled.fingerprint, reference.fingerprint);
    assert_eq!(
        pooled.calibration.as_ref().unwrap().length,
        calibration.length
    );
    assert_eq!(pooled.steps, reference.steps);
    assert_eq!(pooled.to_bytes().unwrap(), reference.to_bytes().unwrap());
}

#[test]
fn an_explicit_colour_source_and_capabilities_reach_a_gas_that_records_no_viscous_force() {
    let mut gas = euclidean(6, 7, None);
    let archive = record(&mut gas, 8, false);
    let config = MeasurementConfig {
        warmup: 2,
        max_lag: 2,
        color: ColorSource::RecordedField {
            stage: "B1".into(),
            amplitude: "total_force".into(),
            phase: "force_input_velocity".into(),
            alignment: algorithmic_gas::physics::spectroscopy::config::StepAlignment::Matched,
            threshold: 1e-12,
        },
        ..MeasurementConfig::default()
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    let meson = measurement
        .channel("meson/scalar/standard/distance")
        .unwrap();
    assert!(meson.availability.is_available());
    assert_eq!(meson.weight.len(), measurement.frames());
    let capabilities = Capabilities::of(&archive.gas_config, &archive.config, 3).refine(&config);
    assert!(capabilities.has(Record::Color));
    let injected = measure_archive_with(
        &config,
        &archive,
        Extensions::default().capabilities(capabilities),
    )
    .unwrap();
    // The injected capabilities are the ones the accumulator resolves itself.
    assert_eq!(injected.fingerprint, measurement.fingerprint);
    assert_eq!(injected.frames(), measurement.frames());
}

#[test]
fn a_multiscale_gate_copies_a_channel_per_scale_and_removes_the_wide_elements() {
    let archive = record(&mut einstein_hilbert(8, 7), 12, true);
    let config = MeasurementConfig {
        warmup: 2,
        max_lag: 3,
        channels: vec![ChannelSpec::U1 {
            mode: U1Mode::Phase,
            charge: 1,
        }],
        scales: Some(ScaleConfig {
            scales: ScaleSelection::Fixed {
                values: vec![1e-9, 1e9],
            },
            mode: ScaleMode::Gate,
            length: EdgeLength::Geodesic,
            channels: vec!["*".into()],
        }),
        flow: Some(FlowConfig {
            steps: 3,
            step_size: 0.5,
            every: 2,
        }),
        ..MeasurementConfig::default()
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    measurement.validate().unwrap();
    let base = measurement.channel("u1/phase/q1/distance").unwrap();
    let narrow = measurement.channel("u1/phase/q1/distance@0").unwrap();
    let wide = measurement.channel("u1/phase/q1/distance@1").unwrap();
    assert_eq!(
        (base.scale, narrow.scale, wide.scale),
        (None, Some(1e-9), Some(1e9))
    );
    // Below the geodesic edge floor every pair is outside the scale; above
    // every graph diameter only the pairs holding a cloned walker are, since
    // the recorded graph places a clone on its donor.
    assert_eq!(narrow.coverage.valid, 0);
    assert!(narrow.coverage.masked_scale >= base.coverage.valid);
    assert_eq!(narrow.weight, vec![0.; measurement.frames()]);
    assert!(wide.coverage.valid > 0 && wide.coverage.valid <= base.coverage.valid);
    assert!(wide.coverage.masked_scale < narrow.coverage.masked_scale);
    let flow = measurement.flow.as_ref().unwrap();
    assert_eq!(flow.frames, 5);
    assert_eq!(flow.steps, vec![0, 1, 2, 3]);
    assert_eq!(flow.roughness.len(), 4);
}

#[test]
fn a_warm_up_that_gives_no_phase_length_says_so_on_every_colour_channel() {
    let archive = record(&mut euclidean(6, 7, Some(unit_viscosity())), 8, false);
    let config = MeasurementConfig {
        warmup: 2,
        max_lag: 2,
        // The Euclidean Gas has no geometry stage, so no edge length exists.
        phase: PhaseScale {
            length: LengthScale::WarmupEdgeMean { geodesic: false },
            ..PhaseScale::default()
        },
        ..MeasurementConfig::default()
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    measurement.validate().unwrap();
    let calibration = measurement.calibration.as_ref().unwrap();
    assert_eq!(calibration.length, 0.);
    assert_eq!(calibration.kappa, 0.);
    assert_eq!(
        calibration.length_source,
        "unavailable: positive warm-up phase length unavailable"
    );
    let meson = measurement
        .channel("meson/scalar/standard/distance")
        .unwrap();
    assert_eq!(
        meson.availability.reason(),
        Some("unavailable: positive warm-up phase length unavailable")
    );
    assert!(meson.values.is_empty() && meson.weight.is_empty());
    let fitness = measurement.channel("fitness_phase/site").unwrap();
    assert!(fitness.availability.is_available());
    assert_eq!(fitness.weight.len(), measurement.frames());
}

#[test]
fn every_propagator_block_holds_one_source_frame_in_measurement_order() {
    let rows = |moments: &BlockMoments| {
        (
            moments.n.clone(),
            moments.ab.clone(),
            moments.a.clone(),
            moments.b.clone(),
        )
    };
    let slot = dossier(IdentityPolicy::Slot);
    let channel = slot.channel("custom/pair/distance").unwrap();
    let moments = channel.propagator.as_ref().unwrap();
    assert_eq!((moments.blocks, moments.origins_per_block), (6, 1));
    // The per-origin rows of the dossier record, t0..t5 over the lags 0..3.
    // Pooling hides their order and their boundaries, so a ring flushed
    // newest-first or an origin pooled with its neighbour is invisible in
    // `pooled`; a block resampling reads exactly these rows.
    assert_eq!(
        rows(moments),
        (
            [
                [3., 1., 3., 3.],
                [2., 2., 2., 0.],
                [3., 3., 0., 0.],
                [3., 0., 0., 0.],
                [3., 3., 0., 0.],
                [3., 0., 0., 0.],
            ]
            .concat(),
            [
                [49., 4., 48., 12.],
                [8., 12., 0., 0.],
                [44., 12., 0., 0.],
                [4., 0., 0., 0.],
                [49., 35., 0., 0.],
                [33., 0., 0., 0.],
            ]
            .concat(),
            [
                [11., 2., 11., 11.],
                [4., 4., 4., 0.],
                [10., 10., 0., 0.],
                [2., 0., 0., 0.],
                [11., 11., 0., 0.],
                [9., 0., 0., 0.],
            ]
            .concat(),
            [
                [11., 2., 11., 2.],
                [4., 6., 0., 0.],
                [10., 2., 0., 0.],
                [2., 0., 0., 0.],
                [11., 9., 0., 0.],
                [9., 0., 0., 0.],
            ]
            .concat(),
        )
    );
    let incarnation = dossier(IdentityPolicy::Incarnation);
    let channel = incarnation.channel("custom/pair/distance").unwrap();
    let moments = channel.propagator.as_ref().unwrap();
    assert_eq!(
        rows(moments),
        (
            [
                [3., 1., 1., 1.],
                [2., 0., 0., 0.],
                [3., 3., 0., 0.],
                [3., 0., 0., 0.],
                [3., 1., 0., 0.],
                [3., 0., 0., 0.],
            ]
            .concat(),
            [
                [49., 4., 6., 0.],
                [8., 0., 0., 0.],
                [44., 12., 0., 0.],
                [4., 0., 0., 0.],
                [49., 8., 0., 0.],
                [33., 0., 0., 0.],
            ]
            .concat(),
            [
                [11., 2., 3., 3.],
                [4., 0., 0., 0.],
                [10., 10., 0., 0.],
                [2., 0., 0., 0.],
                [11., 2., 0., 0.],
                [9., 0., 0., 0.],
            ]
            .concat(),
            [
                [11., 2., 2., 0.],
                [4., 0., 0., 0.],
                [10., 2., 0., 0.],
                [2., 0., 0., 0.],
                [11., 4., 0., 0.],
                [9., 0., 0., 0.],
            ]
            .concat(),
        )
    );
    // Four sinks of origin 0, four of origin 1 and two of origin 4 lose a
    // walker that was cloned in between; none of them is also colour-masked,
    // so the identity mask owns all ten.
    assert_eq!(channel.coverage.masked_identity, 10);
}

#[test]
fn an_element_weight_divides_the_frame_mean_and_every_propagator_sum() {
    let measurement = weighted();
    let channel = measurement.channel("custom/pair/distance").unwrap();
    // Five elements of weights 1/2, 1/2, 1, 1/2, 1/2 over two frames: the
    // self-companion of walker 1 leaves it one element of full weight.
    assert_eq!(channel.coverage.valid, 10);
    assert_eq!(channel.coverage.masked_self, 2);
    let expected = [3., 12.5 / 3.];
    for (value, expected) in channel.values.iter().zip(&expected) {
        assert!((value - expected).abs() < 1e-15);
    }
    // The unweighted frame mean of the first frame is 16/5, the unweighted
    // lag-0 sums are n = 5 and ab = 62.
    let moments = channel.propagator.as_ref().unwrap();
    assert_eq!(moments.blocks, 2);
    assert_eq!(moments.n, vec![3., 3., 3., 0.]);
    assert_eq!(moments.ab, vec![33., 33., 62.5, 0.]);
    assert_eq!(moments.a, vec![9., 9., 12.5, 0.]);
    assert_eq!(moments.b, vec![9., 12.5, 12.5, 0.]);
    assert_eq!(
        moments.estimate(Subtraction::None),
        vec![Some(95.5 / 6.), Some(11.)]
    );
}

#[test]
fn the_euclidean_slabs_bin_the_anchor_and_pair_every_slab_with_itself() {
    let measurement = dossier_with(slab_config(256), slab_extensions());
    let channel = measurement.channel("custom/pair/distance").unwrap();
    let slabs = channel.euclidean.as_ref().unwrap();
    slabs.validate().unwrap();
    assert_eq!(
        (slabs.bins, slabs.blocks, slabs.origins_per_block),
        (2, 6, 1)
    );
    // The anchor coordinate of the six frames falls in the bins [0,2) and
    // [2,4); walker 2 of the last frame sits at 4 and belongs to no slab, so
    // the upper slab of that frame holds nothing.
    assert_eq!(
        slabs.profile,
        [[2., 4.5], [2., 2.], [2., 4.], [1., 0.], [3., 4.], [2.5, 0.]].concat()
    );
    assert_eq!(
        slabs.profile_n,
        [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 0.]].concat()
    );
    // `[blocks, bins, bins]` with the entries `b ≤ b'`: the diagonal is the
    // slab paired with itself, which the lag-0 estimate is made of.
    assert_eq!(
        slabs.pair_ab,
        [
            [4., 9., 0., 20.25],
            [4., 4., 0., 4.],
            [4., 8., 0., 16.],
            [1., 0., 0., 0.],
            [9., 12., 0., 16.],
            [6.25, 0., 0., 0.],
        ]
        .concat()
    );
    assert_eq!(
        slabs.pair_n,
        [
            [1., 1., 0., 1.],
            [1., 1., 0., 1.],
            [1., 1., 0., 1.],
            [1., 1., 0., 1.],
            [1., 1., 0., 1.],
            [1., 0., 0., 0.],
        ]
        .concat()
    );
}

#[test]
fn slab_blocks_merge_in_pairs_when_the_block_cap_is_reached() {
    let measurement = dossier_with(slab_config(4), slab_extensions());
    let channel = measurement.channel("custom/pair/distance").unwrap();
    let slabs = channel.euclidean.as_ref().unwrap();
    slabs.validate().unwrap();
    // Four one-frame blocks merge into two two-frame blocks when the fifth
    // frame arrives, and the segment break between the fourth and the fifth
    // frame keeps the merged pairs inside their segment.
    assert_eq!((slabs.blocks, slabs.origins_per_block), (3, 2));
    assert_eq!(slabs.profile, [[4., 6.5], [3., 4.], [5.5, 4.]].concat());
    assert_eq!(slabs.profile_n, [[2., 2.], [2., 2.], [2., 1.]].concat());
    assert_eq!(
        slabs.pair_ab,
        [
            [8., 13., 0., 24.25],
            [5., 8., 0., 16.],
            [15.25, 12., 0., 16.]
        ]
        .concat()
    );
    assert_eq!(
        slabs.pair_n,
        [[2., 2., 0., 2.], [2., 2., 0., 2.], [2., 1., 0., 1.]].concat()
    );
    // Merging moves no weight: every entry sums to what the six one-frame
    // blocks of the same record hold.
    let open = dossier_with(slab_config(256), slab_extensions());
    let channel = open.channel("custom/pair/distance").unwrap();
    let unmerged = channel.euclidean.as_ref().unwrap();
    let totals = |values: &[f64], width: usize| {
        let mut sums = vec![0.; width];
        for (entry, value) in values.iter().enumerate() {
            sums[entry % width] += value;
        }
        sums
    };
    assert_eq!(totals(&slabs.pair_ab, 4), totals(&unmerged.pair_ab, 4));
    assert_eq!(totals(&slabs.pair_n, 4), totals(&unmerged.pair_n, 4));
    assert_eq!(totals(&slabs.profile, 2), totals(&unmerged.profile, 2));
    assert_eq!(totals(&slabs.profile_n, 2), totals(&unmerged.profile_n, 2));
}

#[test]
fn a_nonfinite_operator_value_is_masked_and_never_reaches_a_retained_sum() {
    let measurement = dossier_with(
        dossier_config(IdentityPolicy::Slot),
        Extensions::default().operator(ElementKind::Site, Poisoned),
    );
    measurement.validate().unwrap();
    let channel = measurement.channel("custom/poisoned/site").unwrap();
    // The operator claims a valid readout on all three sites of every frame;
    // the nonfinite value of two of them is a mask, not the number 0.
    assert_eq!(channel.coverage.valid, 6);
    assert_eq!(channel.coverage.masked_ineligible, 12);
    assert_eq!(channel.values, vec![1.; 6]);
    assert_eq!(channel.weight, vec![1.; 6]);
    let pooled = channel.propagator.as_ref().unwrap().pooled(None);
    assert_eq!(pooled.n, vec![6., 4., 2., 1.]);
    assert_eq!(pooled.ab, vec![6., 4., 2., 1.]);
}

#[test]
fn a_lag_stride_leaves_the_skipped_lags_without_a_sample() {
    let config = MeasurementConfig {
        propagators: PropagatorConfig {
            enabled_for: vec!["*".into()],
            lag_stride: 2,
            ..PropagatorConfig::default()
        },
        ..dossier_config(IdentityPolicy::Slot)
    };
    let measurement = dossier_with(
        config,
        Extensions::default()
            .operator(ElementKind::DistancePair, Pair)
            .field(Tabulated),
    );
    let channel = measurement.channel("custom/pair/distance").unwrap();
    let moments = channel.propagator.as_ref().unwrap();
    // The evaluated lags carry the columns of the unstrided record; the
    // skipped ones are never evaluated and hold no sample at all.
    let pooled = moments.pooled(None);
    assert_eq!(pooled.n, vec![17., 0., 5., 0.]);
    assert_eq!(pooled.ab, vec![187., 0., 48., 0.]);
    assert_eq!(pooled.a, vec![47., 0., 15., 0.]);
    assert_eq!(pooled.b, vec![47., 0., 11., 0.]);
    assert_eq!(
        moments.estimate(Subtraction::None),
        vec![Some(11.), None, Some(9.6), None]
    );
}

#[test]
fn an_exchange_odd_element_whose_mirror_is_invalid_keeps_its_frame_mean() {
    let measurement = dossier_with(
        dossier_config(IdentityPolicy::Slot),
        Extensions::default()
            .operator(ElementKind::DistancePair, AnchorOdd)
            .field(Lopsided),
    );
    let channel = measurement.channel("custom/anchor/distance").unwrap();
    assert_eq!(channel.exchange, ExchangeParity::Odd);
    // Frames 1 and 3 hold the pair (0,1) whose mirror (1,0) is invalid: it
    // cancels against nothing and stays in the average. Frame 2 holds the two
    // valid mirrors (0,2) and (2,0), which do cancel.
    assert_eq!(channel.weight, vec![1., 1., 0., 1., 1., 1.]);
    assert_eq!(channel.values, vec![3.5, 1., 0., -4.5, 1., 13.5]);
}

#[test]
fn a_restored_state_whose_stream_outruns_its_measurement_is_refused() {
    let injected = || {
        Extensions::default()
            .operator(ElementKind::DistancePair, Pair)
            .field(Tabulated)
    };
    let mut gas = euclidean(3, 7, None);
    let mut archive = record(&mut gas, 6, false);
    tabulate(&mut archive);
    let config = dossier_config(IdentityPolicy::Slot);
    let mut accumulator = accumulator_with(&config, &archive, injected());
    accumulator.ingest_archive(&archive).unwrap();
    let state = accumulator.state();
    assert!(Accumulator::restore_with(state.clone(), injected()).is_ok());
    // A measurement one frame shorter than the stream that carries it: the
    // ring of a checkpoint from a browser is checked against the frames it
    // claims before any operator reads it.
    let mut short = state;
    short.measurement.steps.pop();
    short.measurement.segment.pop();
    for channel in &mut short.measurement.channels {
        channel.weight.pop();
        let kept = channel.values.len().saturating_sub(channel.components);
        channel.values.truncate(kept);
    }
    short.measurement.validate().unwrap();
    assert!(matches!(
        Accumulator::restore_with(short, injected()),
        Err(GasError::Configuration(_))
    ));
}

/// A viscous Euclidean gas inside a periodic box: the run a momentum
/// projection needs, with a matter field the plaquette can read.
fn periodic_euclidean(walkers: usize, seed: u64) -> AlgorithmicGas<f64> {
    let mut config = GasConfig::euclidean(3, 0.05).unwrap();
    config.seed = seed;
    config.qft.viscosity = Some(unit_viscosity());
    let field = position_field(&config).to_string();
    let domain = BoxDomain {
        lower: vec![-2.5; 3],
        upper: vec![2.5; 3],
    };
    config.boundary = BoundaryPolicy::PeriodicBox {
        field: field.clone(),
        domain: domain.clone(),
    };
    // A periodic box needs the companion distance of the same coordinates to
    // take the same minimum image.
    for module in [&mut config.distance_donors, &mut config.cloning_donors] {
        module.distance = Distance::Euclidean {
            field: field.clone(),
            scales: vec![],
            squared: false,
            periodic: Some(domain.clone()),
        };
    }
    block_on(
        GasBuilder::new(population::<f64>(walkers, 3, 2., 1.), Quadratic)
            .gradient(Quadratic)
            .config(config)
            .build(),
    )
    .unwrap()
}
/// Connected autocorrelation of a retained one-component frame series, over
/// the frames of one segment, with one global mean.
fn frame_correlator(series: &ChannelSeries, lags: usize) -> Vec<f64> {
    // The colour of the first frame has no preceding kick behind it; every
    // later frame of the segment carries a value.
    let first = series.weight.iter().position(|w| *w > 0.).unwrap();
    assert!(
        series.weight[first..].iter().all(|w| *w > 0.),
        "masked frame"
    );
    let values = &series.values[first * series.components..];
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (0..=lags)
        .map(|lag| {
            let kept = values.len() - lag;
            (0..kept)
                .map(|t| (values[t] - mean) * (values[t + lag] - mean))
                .sum::<f64>()
                / kept as f64
        })
        .collect()
}
fn glueball(observable: GlueballObservable, momentum: Option<Momentum>) -> ChannelSpec {
    ChannelSpec::Glueball {
        observable,
        momentum,
    }
}
#[test]
fn momentum_projection_is_invariant_under_an_additive_shift_of_the_observable() {
    let archive = record(&mut periodic_euclidean(12, 20260920), 240, false);
    let mode = Some(Momentum {
        axis: 0,
        mode: 1,
        phase: MomentumPhase::Cos,
    });
    let config = MeasurementConfig {
        channels: vec![
            glueball(GlueballObservable::RePlaquette, mode),
            glueball(GlueballObservable::OneMinusRe, mode),
            glueball(GlueballObservable::RePlaquette, None),
            glueball(GlueballObservable::OneMinusRe, None),
        ],
        propagators: PropagatorConfig {
            enabled_for: vec!["*".into()],
            ..PropagatorConfig::default()
        },
        ..immediate(8)
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    measurement.validate().unwrap();
    assert!(measurement.frames() >= 200);
    let series = |observable, momentum| {
        let id = format!("{}/triplet", glueball(observable, momentum).id());
        let channel = measurement.channel(&id).unwrap_or_else(|| panic!("{id}"));
        assert!(
            channel.availability.is_available(),
            "{id}: {:?}",
            channel.availability
        );
        channel
    };
    let re = series(GlueballObservable::RePlaquette, mode);
    let one_minus = series(GlueballObservable::OneMinusRe, mode);
    // `1 − Re Π` is an affine image of `Re Π`, so the connected momentum mode
    // of the two arms is the same series up to its sign, frame by frame.
    // `|Re Π| ≤ 1`, so the two subtractions differ by the rounding of a
    // number of order one and the equality is stated in absolute terms.
    for ((a, b), weight) in re.values.iter().zip(&one_minus.values).zip(&re.weight) {
        assert!(*weight == 0. || (a + b).abs() < 1e-14, "{a} vs {b}");
    }
    // The two series are formed by two different subtractions, so the equality
    // holds to the cancellation error of `1 − Re Π` and not to the last bit.
    let (here, there) = (frame_correlator(re, 6), frame_correlator(one_minus, 6));
    for (a, b) in here.iter().zip(&there) {
        assert!((a - b).abs() < 1e-14, "{a} vs {b}");
    }
    assert!(here[0] > 0.);
    // Q3 at channel level: the subtraction of the source-frozen propagator is
    // covariant under the same shift, so the two unprojected arms agree at
    // every lag without any projection at all.
    let estimate = |channel: &ChannelSeries| {
        channel
            .propagator
            .as_ref()
            .unwrap()
            .estimate(Subtraction::LagMeans)
    };
    let bare = estimate(series(GlueballObservable::RePlaquette, None));
    let shifted = estimate(series(GlueballObservable::OneMinusRe, None));
    assert_eq!(bare.len(), config.max_lag + 1);
    for (a, b) in bare.iter().zip(&shifted) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a - b).abs() < 1e-14, "{a} vs {b}");
    }
    assert!(bare[0].unwrap() > 0.);
}

#[test]
fn a_frozen_score_directed_pair_keeps_its_source_orientation_at_the_sink() {
    let archive = record(&mut euclidean(8, 11, Some(unit_viscosity())), 40, false);
    let config = MeasurementConfig {
        channels: vec![
            ChannelSpec::Meson {
                quantum: MesonQuantum::Pseudoscalar,
                mode: MesonMode::Standard,
            },
            ChannelSpec::Meson {
                quantum: MesonQuantum::Pseudoscalar,
                mode: MesonMode::ScoreDirected,
            },
        ],
        pairs: PairSelection::Distance,
        propagators: PropagatorConfig {
            enabled_for: vec!["*".into()],
            ..PropagatorConfig::default()
        },
        ..immediate(6)
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    measurement.validate().unwrap();
    let channel = |id: &str| measurement.channel(id).unwrap();
    let standard = channel("meson/pseudoscalar/standard/distance");
    let directed = channel("meson/pseudoscalar/score_directed/distance");
    assert_eq!(standard.coverage.valid, directed.coverage.valid);
    // The orientation is frozen with the element, so it enters the lag product
    // squared and the directed propagator would be the standard one lag by lag
    // — measured equal to 1e-12 on this record while the arm still published
    // one. A second id for one observable gives a joint covariance an exact
    // null direction, so the directed arm carries no propagator of its own and
    // only its frame mean is a channel.
    assert!(standard.propagator.is_some() && directed.propagator.is_none());
    // The frame sums still carry the orientation, so the two channels are not
    // the same series: only the product of two frozen ends is.
    let (a, b) = (&standard.values, &directed.values);
    assert_eq!(a.len(), b.len());
    assert!(
        a.iter()
            .zip(b)
            .any(|(x, y)| (x - y).abs() > 1e-6 * x.abs().max(1e-6))
    );
}

#[test]
fn a_frame_whose_scores_all_tie_has_no_score_weighted_average() {
    let archive = record(&mut flat(6, 13, Some(unit_viscosity())), 10, false);
    let weighted = ChannelSpec::Meson {
        quantum: MesonQuantum::Scalar,
        mode: MesonMode::ScoreWeighted,
    };
    let standard = ChannelSpec::Meson {
        quantum: MesonQuantum::Scalar,
        mode: MesonMode::Standard,
    };
    let config = MeasurementConfig {
        channels: vec![weighted.clone(), standard.clone()],
        pairs: PairSelection::Distance,
        ..immediate(3)
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    measurement.validate().unwrap();
    let frames = measurement.frames();
    let channel = |spec: &ChannelSpec| {
        measurement
            .channel(&format!("{}/distance", spec.id()))
            .unwrap()
    };
    // Every score of every frame is 0, so every pair ties: the weighted arm
    // has no denominator and no frame carries a value, while the arm that
    // reads no score still measures the same frames.
    let tied = channel(&weighted);
    assert!(tied.coverage.valid > 0 && frames > 0);
    assert_eq!(tied.weight, vec![0.; frames]);
    // The colour of the first frame has no preceding kick behind it; the arm
    // that reads no score measures every later frame.
    let plain = channel(&standard);
    assert_eq!(plain.weight[0], 0.);
    assert_eq!(plain.weight[1..], vec![1.; frames - 1]);
    assert!(plain.values.iter().any(|v| v.abs() > 1e-9));
}

/// The one-dimensional Euclidean gas: `rewrite` then controls every
/// coordinate of every walker, so a companion distance is a difference of two
/// tabulated numbers.
fn linear(walkers: usize, seed: u64) -> AlgorithmicGas<f64> {
    let mut config = GasConfig::euclidean(1, 0.05).unwrap();
    config.seed = seed;
    block_on(
        GasBuilder::new(population::<f64>(walkers, 1, 2., 1.), Quadratic)
            .gradient(Quadratic)
            .config(config)
            .build(),
    )
    .unwrap()
}
/// Five walkers on a line. Source 0 and source 1 both name slot 0; source 1
/// is a donor of an earlier frame. Walker 0 takes the first, which is itself,
/// walker 1 the second, and the rest name walkers 3, 4 and 2.
const WARMUP_STEPS: [u64; 2] = [30, 31];
const WARMUP_LINE: [f64; 5] = [0., 1., 3., 7., 15.];
const WARMUP_COMPANION: [u32; 5] = [0, 1, 3, 4, 2];

#[test]
fn the_warm_up_phase_length_ignores_self_companions_and_historical_donors() {
    let mut gas = linear(5, 23);
    let mut archive = record(&mut gas, 2, false);
    let field = position_field(&archive.gas_config).to_string();
    for (index, step) in archive.steps.iter_mut().enumerate() {
        rewrite(
            step,
            &field,
            WARMUP_STEPS[index],
            &WARMUP_LINE,
            &[0; 5],
            &WARMUP_COMPANION,
        );
        // Source 1 names walker 0 in the frame before the last one.
        step.report.distance_sources[1].slot = 0;
        step.report.distance_sources[1].frame = WARMUP_STEPS[index] - 2;
        for choice in &mut step.report.clone_plan.choices {
            choice.accepted = false;
            choice.revival = false;
        }
    }
    let config = MeasurementConfig {
        warmup: 1,
        channels: vec![ChannelSpec::FitnessPhase],
        pairs: PairSelection::Distance,
        phase: PhaseScale {
            length: LengthScale::WarmupCompanionMedian,
            ..PhaseScale::default()
        },
        ..MeasurementConfig::default()
    };
    let measurement = measure_archive(&config, &archive).unwrap();
    let calibration = measurement.calibration.as_ref().unwrap();
    // Walker 0 is its own companion and walker 1 has a historical donor:
    // neither builds a sample, so the median is taken over |7 − 3| = 4,
    // |15 − 7| = 8 and |3 − 15| = 12. Admitting the zero of the self-companion
    // and the |0 − 1| of the historical donor would give the median of
    // [0, 1, 4, 8, 12], which is 4.
    assert_eq!(calibration.warmup_frames, 1);
    assert!((calibration.length - 8.).abs() < 1e-15, "{calibration:?}");
    assert!(
        calibration
            .length_source
            .contains("3 samples over 1 warm-up frames"),
        "{}",
        calibration.length_source
    );
}

/// Two distance companions per walker, both entries of walker 0 its own
/// neighbour: the duplicate slot of a `CompanionChoice::All` row.
const DUPLICATE_COMPANION: [u32; 6] = [1, 1, 2, 0, 0, 1];

#[test]
fn a_row_whose_two_entries_hold_the_same_slot_builds_two_elements_of_half_weight() {
    let mut gas = euclidean(3, 7, None);
    let mut archive = record(&mut gas, 2, false);
    let field = position_field(&archive.gas_config).to_string();
    for (index, step) in archive.steps.iter_mut().enumerate() {
        rewrite(
            step,
            &field,
            WEIGHTED_STEPS[index],
            &WEIGHTED[index],
            &[0; 3],
            &DUPLICATE_COMPANION,
        );
        for choice in &mut step.report.clone_plan.choices {
            choice.accepted = false;
            choice.revival = false;
        }
    }
    let config = MeasurementConfig {
        companions: CompanionChoice::All,
        max_lag: 1,
        ..dossier_config(IdentityPolicy::Slot)
    };
    let extensions = Extensions::default()
        .operator(ElementKind::DistancePair, Pair)
        .field(Everything);
    let mut accumulator = accumulator_with(&config, &archive, extensions);
    accumulator.ingest_archive(&archive).unwrap();
    let measurement = accumulator.into_measurement().unwrap();
    let channel = measurement.channel("custom/pair/distance").unwrap();
    // The duplicate slot is a second element, not one entry counted twice:
    // walker 0 carries two elements (0, 1) of weight 1/2, so every anchor
    // still contributes the total weight 1 and the frame holds six elements.
    assert_eq!(channel.coverage.valid, 12);
    assert_eq!(channel.coverage.masked(), 0);
    let pooled = channel.propagator.as_ref().unwrap().pooled(None);
    assert_eq!(pooled.n[0], 6.);
    // u = (1, 2, 3): 2·½·(1·2) + ½·(2·3) + ½·(2·1) + ½·(3·1) + ½·(3·2) = 10.5.
    assert!(
        (channel.values[0] - 3.5).abs() < 1e-15,
        "{:?}",
        channel.values
    );
    assert_eq!(channel.weight, vec![1.; 2]);
}

#[test]
fn every_channel_report_states_the_frame_normalisation_it_used() {
    let archive = record(&mut euclidean(6, 7, Some(unit_viscosity())), 16, false);
    for normalization in [FrameNormalization::ValidCount, FrameNormalization::FixedN] {
        let config = MeasurementConfig {
            warmup: 2,
            max_lag: 3,
            normalization,
            channels: vec![
                ChannelSpec::Meson {
                    quantum: MesonQuantum::Scalar,
                    mode: MesonMode::Standard,
                },
                ChannelSpec::CloneIndicator,
            ],
            ..MeasurementConfig::default()
        };
        let measurement = measure_archive(&config, &archive).unwrap();
        let report = analyze(&[measurement], &AnalysisConfig::default()).unwrap();
        assert!(!report.channels.is_empty());
        // `B/09:334-337`: the valid-count average and the fixed 1/N average are
        // two different observables, so every row of the report names its own.
        for channel in &report.channels {
            assert!(channel.normalization.is_some(), "{}", channel.id);
        }
        let row = |id: &str| {
            report
                .channels
                .iter()
                .find(|channel| channel.id == id)
                .unwrap_or_else(|| panic!("{id}"))
        };
        assert_eq!(
            row("meson/scalar/standard/distance").normalization,
            Some(normalization)
        );
        // The clone indicator fixes the fixed denominator in its own signature
        // and keeps it whatever the measurement configured.
        assert_eq!(
            row("clone_indicator/site").normalization,
            Some(FrameNormalization::FixedN)
        );
    }
}
