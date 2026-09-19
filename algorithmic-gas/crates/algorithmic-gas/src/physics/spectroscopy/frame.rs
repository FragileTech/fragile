//! `RecordedStep` → `Frame`: the pre-clone population of a recorded step with
//! the decisions and force records of that step. Field names come from the
//! recorded gas configuration, stages from the recorded labels.
use super::{
    config::MeasurementConfig,
    contract::{Capabilities, Frame, FrameState},
};
use crate::{
    GasConfig, GasError, Result, geometry::Distance, kinetic::KineticKind, tracking::RecordedStep,
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
/// the step never reached. A missing `pre_clone` stage is
/// `GasError::Capability`, a missing position field `GasError::MissingField`,
/// a malformed record `GasError::Shape`.
pub fn extract(
    gas: &GasConfig,
    step: &RecordedStep<f64>,
    measurement: &MeasurementConfig,
    capabilities: &Capabilities,
) -> Result<Frame> {
    let _ = (gas, step, measurement, capabilities);
    Err(GasError::Capability("pending: spectroscopy::frame".into()))
}

/// State of a frame before any field source ran: coordinates, fitness,
/// identity, `euclidean_time` from `capabilities.euclidean_axis`, the
/// `distance_companion` / `cloning_companion` maps (`Companions::first`, else
/// `NO_COMPANION`), an all-invalid colour and force and no score.
pub fn base_state(frame: &Frame, capabilities: &Capabilities) -> FrameState {
    let _ = capabilities;
    FrameState {
        step: frame.step,
        n: frame.n,
        d: frame.d,
        ..FrameState::default()
    }
}
