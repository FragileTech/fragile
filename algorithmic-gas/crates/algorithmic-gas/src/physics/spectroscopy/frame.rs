//! `RecordedStep` → `Frame`: the pre-clone population of a recorded step with
//! the decisions and force records of that step. Field names come from the
//! recorded gas configuration, stages from the recorded labels.
use super::{
    config::{ColorSource, MeasurementConfig},
    contract::{
        Capabilities, Companions, Frame, FrameState, Kick, NO_COMPANION, Record, StageKick,
    },
};
use crate::{
    GasConfig, GasError, Precision, Result,
    donor::{CompanionBatch, SourceRef},
    geometry::Distance,
    kinetic::KineticKind,
    memory::checked_mul,
    physics::qft::math::C,
    tracking::{FieldEvaluation, RecordedStep},
};

/// Observation field that holds the positions of a gas configuration. An
/// environment kinetic names none, so the coordinates are those the distance
/// of the distance donors is measured on, the same lookup
/// `ElectroweakScales::lambda` documents.
pub fn position_field(gas: &GasConfig) -> &str {
    match &gas.kinetic.integrator {
        KineticKind::Baoab { positions, .. } => positions,
        KineticKind::DirectJump { field, .. } | KineticKind::Brownian { field, .. } => field,
        KineticKind::Environment => match &gas.distance_donors.distance {
            Distance::Euclidean { field, .. } | Distance::Cosine { field, .. } => field,
            Distance::PhaseSpace { positions, .. }
            | Distance::SquashedPhaseSpace { positions, .. } => positions,
        },
    }
}
/// Only the BAOAB integrator carries velocities.
fn velocity_field(gas: &GasConfig) -> Option<&str> {
    match &gas.kinetic.integrator {
        KineticKind::Baoab { velocities, .. } => Some(velocities),
        _ => None,
    }
}
fn shape(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(GasError::Shape(message.into()))
    }
}

/// Pool indices of one role resolved through its source pool. A source of
/// frame `step − 1` is a walker of this pre-clone population; any other is
/// historical and its slot names a row of that earlier frame.
fn companions(
    batch: &CompanionBatch,
    sources: &[SourceRef],
    n: usize,
    step: u64,
) -> Result<Companions> {
    batch.validate_indices(sources.len())?;
    shape(batch.rows == n, "companion rows differ from the population")?;
    let entries = batch.indices.len();
    let mut out = Companions {
        count: batch.count,
        slot: vec![0; entries],
        generation: vec![0; entries],
        valid: batch.valid.clone(),
        historical: vec![false; entries],
        mutual: batch.mutual,
    };
    for (e, &index) in batch.indices.iter().enumerate() {
        if batch.valid[e] {
            let source = sources[index as usize];
            shape((source.slot as usize) < n, "companion source slot")?;
            out.slot[e] = source.slot;
            out.generation[e] = source.generation;
            out.historical[e] = source.frame.checked_add(1) != Some(step);
        }
    }
    Ok(out)
}

/// Two `[n, d]` records of one stage evaluated on one population version,
/// with the generations of the stage snapshot of that version. `None` when a
/// record or that snapshot is absent, the versions differ, or a record is not
/// `n` rows of `d`-vectors.
fn stage_kick(
    step: &RecordedStep<f64>,
    stage: &str,
    force: &str,
    velocity: &str,
    n: usize,
    d: usize,
) -> Result<Option<StageKick>> {
    let find = |field: &str| {
        step.field_evaluations
            .iter()
            .find(|e| e.stage == stage && e.field == field)
    };
    let vectors = |e: &FieldEvaluation| e.item_shape == [d] && e.rows == n;
    let (Some(f), Some(v)) = (find(force), find(velocity)) else {
        return Ok(None);
    };
    let Some(snapshot) = step.stages.iter().find(|s| s.version == f.version) else {
        return Ok(None);
    };
    if f.version != v.version || !vectors(f) || !vectors(v) {
        return Ok(None);
    }
    let cells = checked_mul(n, d)?;
    shape(
        f.values.len() == cells
            && v.values.len() == cells
            && f.available.len() == n
            && v.available.len() == n
            && snapshot.generations.len() == n,
        "stage record shape",
    )?;
    let available = f
        .values
        .chunks_exact(d)
        .zip(v.values.chunks_exact(d))
        .zip(f.available.iter().zip(&v.available))
        .map(|((f, v), (&a, &b))| a && b && f.iter().chain(v).all(|x| x.is_finite()))
        .collect();
    Ok(Some(StageKick {
        force: f.values.clone(),
        velocity: v.values.clone(),
        available,
        generation: snapshot.generations.clone(),
    }))
}

/// The `pre_clone` stage of `step` with the decisions of its `StepReport`:
/// - `eligible` combines the pre-clone engine eligibility with finite
///   coordinates, so that revived rows are ineligible;
/// - companions are resolved from pool indices through `distance_sources` /
///   `clone_plan.sources`; `historical` marks `source.frame != step − 1`;
/// - `companion_fitness[i] = donor_fitness[pool index of i's first cloning
///   companion]`, the actual decision input;
/// - `kick` holds the `B1`/`B2` `viscous_force` records with their
///   `force_input_velocity`; a stage whose two records disagree in version or
///   shape is `None`. `StageKick::generation` is read from the `StageSnapshot`
///   whose `version` equals the record's (the post-clone population for a B
///   stage); a record without such a snapshot is `None`;
/// - `recorded_color` is filled for `ColorSource::RecordedField` from the
///   named stage and fields of this step, with the same generation rule;
/// - `graph` is kept only when `capabilities` has `Record::Graph`. It is the
///   post-clone graph that drove the forces of the step.
///
/// Records the capabilities declare missing stay `None`, as does a B stage
/// the step never reached; `kick` needs `Record::Color` of the viscous-force
/// source, `companion_fitness` the cloning companions and the fitness. The
/// decisions `cloned` and `revived` and the raw reward are kept whatever the
/// capabilities declare: the colour and scale masks read them, and every
/// operator of the decisions is gated on `Record::ClonePlan` itself. A
/// missing `pre_clone` stage and a configuration of another precision than
/// `F64` are `GasError::Capability`, a missing position or velocity field
/// `GasError::MissingField`, a malformed record `GasError::Shape`.
pub fn extract(
    gas: &GasConfig,
    step: &RecordedStep<f64>,
    measurement: &MeasurementConfig,
    capabilities: &Capabilities,
) -> Result<Frame> {
    capabilities.validate()?;
    measurement.color.validate()?;
    if gas.precision != Precision::F64 {
        return Err(GasError::Capability(
            "spectroscopy requires an f64 recorded run".into(),
        ));
    }
    let stage = step
        .stages
        .iter()
        .find(|s| s.stage == "pre_clone")
        .ok_or_else(|| GasError::Capability("pre_clone stage is unavailable".into()))?;
    let name = position_field(gas);
    let x = stage
        .fields
        .get(name)
        .ok_or_else(|| GasError::MissingField(name.into()))?;
    shape(x.item_shape.len() == 1, "positions are not vectors")?;
    let report = &step.report;
    let plan = &report.clone_plan;
    let (n, d) = (stage.generations.len(), x.item_shape[0]);
    let cells = checked_mul(n, d)?;
    shape(
        n >= 1
            && d >= 1
            && d == capabilities.dimension
            && x.values.len() == cells
            && report.pre_clone_eligible.len() == n
            && report.pre_clone_fitness.fitness.len() == n
            && report.pre_clone_rewards.raw.len() == n
            && plan.choices.len() == n
            && step.donor_fitness.len() == plan.sources.len(),
        "pre_clone record shape",
    )?;
    let v = match velocity_field(gas) {
        Some(name) if capabilities.has(Record::Velocities) => {
            let v = stage
                .fields
                .get(name)
                .ok_or_else(|| GasError::MissingField(name.into()))?;
            shape(
                v.item_shape == x.item_shape && v.values.len() == cells,
                "velocities differ in shape from positions",
            )?;
            Some(v.values.clone())
        }
        _ => None,
    };
    let finite =
        |values: &[f64], i: usize| values[i * d..(i + 1) * d].iter().all(|a| a.is_finite());
    let eligible = (0..n)
        .map(|i| {
            report.pre_clone_eligible[i]
                && finite(&x.values, i)
                && v.as_ref().is_none_or(|v| finite(v, i))
        })
        .collect();
    let distance = capabilities
        .has(Record::DistanceCompanions)
        .then(|| {
            companions(
                &report.distance_companions,
                &report.distance_sources,
                n,
                report.step,
            )
        })
        .transpose()?;
    let cloning = capabilities
        .has(Record::CloningCompanions)
        .then(|| companions(&report.cloning_companions, &plan.sources, n, report.step))
        .transpose()?;
    // The score of row `i` reads entry `i`: one cloning companion per walker.
    shape(
        cloning.as_ref().is_none_or(|c| c.count == 1),
        "cloning companion count must be one",
    )?;
    let fitness = capabilities.has(Record::Fitness);
    let companion_fitness = (fitness && cloning.is_some()).then(|| {
        (0..n)
            .map(|i| {
                report
                    .cloning_companions
                    .row(i)
                    .next()
                    .map_or(0., |j| step.donor_fitness[j as usize])
            })
            .collect()
    });
    let color = capabilities.has(Record::Color);
    let (kick, recorded_color) = match &measurement.color {
        ColorSource::ViscousForce { .. } if color && v.is_some() => {
            let kick =
                |stage| stage_kick(step, stage, "viscous_force", "force_input_velocity", n, d);
            (
                Some(Kick {
                    b1: kick("B1")?,
                    b2: kick("B2")?,
                }),
                None,
            )
        }
        ColorSource::RecordedField {
            stage,
            amplitude,
            phase,
            ..
        } if color => (None, stage_kick(step, stage, amplitude, phase, n, d)?),
        _ => (None, None),
    };
    let graph = match &step.graph {
        Some(graph) if capabilities.has(Record::Graph) => {
            graph.validate(n)?;
            Some(graph.clone())
        }
        _ => None,
    };
    Ok(Frame {
        step: report.step,
        epoch: step.epoch,
        n,
        d,
        x: x.values.clone(),
        v,
        eligible,
        generation: stage.generations.clone(),
        fitness: fitness.then(|| report.pre_clone_fitness.fitness.clone()),
        reward: Some(report.pre_clone_rewards.raw.clone()),
        distance,
        cloning,
        companion_fitness,
        cloned: plan.choices.iter().map(|c| c.accepted).collect(),
        revived: plan.choices.iter().map(|c| c.revival).collect(),
        kick,
        recorded_color,
        graph,
    })
}

/// State of a frame before any field source ran: coordinates, fitness,
/// identity, `euclidean_time` from `capabilities.euclidean_axis` unless
/// `Record::EuclideanTime` is declared missing, the
/// `distance_companion` / `cloning_companion` maps (`Companions::first`, else
/// `NO_COMPANION`), an all-invalid colour and force and no score. `frame` is
/// valid (`Frame::validate`): the companion maps index it without checks.
pub fn base_state(frame: &Frame, capabilities: &Capabilities) -> FrameState {
    let (n, d) = (frame.n, frame.d);
    let first = |companions: &Option<Companions>| {
        companions.as_ref().map(|c| {
            (0..n)
                .map(|i| {
                    c.first(i, &frame.eligible)
                        .map_or(NO_COMPANION, |k| c.slot[i * c.count + k])
                })
                .collect()
        })
    };
    FrameState {
        step: frame.step,
        n,
        d,
        x: frame.x.clone(),
        v: frame.v.clone(),
        color: vec![C::ZERO; n * d],
        color_valid: vec![false; n],
        force: None,
        force_valid: vec![false; n],
        phase_velocity: None,
        fitness: frame.fitness.clone(),
        score: None,
        score_valid: vec![],
        score_gradient: None,
        role: None,
        cloned: frame.cloned.clone(),
        generation: frame.generation.clone(),
        eligible: frame.eligible.clone(),
        euclidean_time: capabilities
            .euclidean_axis
            .filter(|&axis| axis < d && capabilities.has(Record::EuclideanTime))
            .map(|axis| frame.x.chunks_exact(d).map(|x| x[axis]).collect()),
        distance_companion: first(&frame.distance),
        cloning_companion: first(&frame.cloning),
    }
}
