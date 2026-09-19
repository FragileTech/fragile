//! Per-walker fields derived from a frame: colour, score and roles, spinors.
//! `ColorSource` is the built-in field source; any other is injected through
//! `Extensions::field`.
pub mod color;
pub mod dirac;
pub mod score;
pub mod spinor;
use super::{
    config::ColorSource,
    contract::{Availability, FieldSource, Frame, FrameState, Record, Requirements},
};

impl FieldSource for ColorSource {
    fn id(&self) -> String {
        format!(
            "color/v1:{}",
            serde_json::to_string(self).unwrap_or_default()
        )
    }
    fn requires(&self) -> Requirements {
        match self {
            Self::ViscousForce { .. } => Requirements::new([Record::Velocities, Record::Color]),
            // Presence of the named stage and fields is checked step by step.
            Self::RecordedField { .. } => Requirements::default(),
        }
    }
    fn fill(
        &self,
        previous: Option<&Frame>,
        frame: &Frame,
        kappa: f64,
        state: &mut FrameState,
    ) -> Availability {
        match self {
            Self::ViscousForce {
                alignment,
                threshold,
            } => color::viscous_force(*alignment, *threshold, previous, frame, kappa, state),
            Self::RecordedField {
                alignment,
                threshold,
                ..
            } => color::recorded_field(*alignment, *threshold, previous, frame, kappa, state),
        }
    }
}
