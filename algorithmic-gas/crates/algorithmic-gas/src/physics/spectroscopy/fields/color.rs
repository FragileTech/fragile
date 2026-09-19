//! Colour state `c_a = F_a e^{i κ v_a} / max(|F|, δ)` of the viscous force under
//! the alignment arms, and the colour determinant. Only `κ` is observable:
//! mass, length and action scale enter through it alone. The map acts
//! component by component in the recorded basis and is normalised over all `d`
//! components; it is not covariant under a rotation of that basis.
use crate::{
    GasError, Result,
    error::require,
    physics::{
        qft::math::{C, mean},
        spectroscopy::{
            config::{ColorAlignment, KickStage, LengthScale, PhaseScale, StepAlignment},
            contract::{Availability, Frame, FrameState, StageKick},
        },
    },
};
use std::f64::consts::PI;

const NO_PRECEDING_STEP: &str = "colour needs the preceding recorded step of the segment";

/// Colour of one walker into `out` (`[d]`); false, with `out` zeroed, when
/// `|F| ≤ threshold`, `|F| = 0` or an input is not finite. Agrees bit for bit
/// with the colour of the recorded-colour lecture experiment under
/// `MatchedKick { B1 }`.
/// `|F|` is the real norm of `force`, which bounds `threshold` in force units.
pub fn color_vector(
    force: &[f64],
    velocity: &[f64],
    kappa: f64,
    threshold: f64,
    out: &mut [C],
) -> bool {
    debug_assert!(force.len() == out.len() && velocity.len() == out.len());
    let norm = force.iter().map(|x| x * x).sum::<f64>().sqrt();
    // A finite norm implies finite components; an overflowing one would
    // otherwise scale every component to zero and still pass the threshold.
    // A zero norm has no direction even under a negative threshold.
    let valid = norm.is_finite()
        && norm > threshold
        && norm > 0.
        && velocity.iter().all(|v| (kappa * v).is_finite());
    if !valid {
        out.fill(C::ZERO);
        return false;
    }
    let scale = norm.max(threshold);
    for ((c, f), v) in out.iter_mut().zip(force).zip(velocity) {
        *c = C::phase(kappa * v) * (f / scale);
    }
    true
}

/// `det[a, b, c]` of three colour vectors in three dimensions, columns in
/// that order, by cofactors of the first column.
pub fn det3(a: &[C], b: &[C], c: &[C]) -> C {
    debug_assert!(a.len() == 3 && b.len() == 3 && c.len() == 3);
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

/// Phase factor `κ = m ℓ₀ / ħ_eff` of a phase scale whose length resolved to
/// `length`, formed once in that order so that equal inputs give equal bits.
pub fn kappa(phase: &PhaseScale, length: f64) -> Result<f64> {
    phase.validate()?;
    require(
        length.is_finite() && length > 0.,
        "colour phase length must be positive",
    )?;
    let kappa = phase.mass * length / phase.h_eff;
    require(kappa.is_finite(), "colour phase factor overflow")?;
    Ok(kappa)
}

/// Length `ℓ₀` of a phase scale from the pooled warm-up `samples` its arm
/// names; `Fixed` reads none. `WarmupCompanionMedian` takes the median of the
/// finite companion distances, the mean of the two middle order statistics
/// for an even count. `WarmupEdgeMean` takes the mean of the finite positive
/// edge lengths, so the zero-length edges of a clone step do not enter.
/// `GasError::Capability` when the warm-up samples give no positive length.
pub fn phase_length(length: LengthScale, samples: &[f64]) -> Result<f64> {
    let value = match length {
        LengthScale::Fixed { value } => {
            require(
                value.is_finite() && value > 0.,
                "colour phase length must be positive",
            )?;
            Some(value)
        }
        LengthScale::WarmupCompanionMedian => {
            let mut sorted: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
            sorted.sort_by(f64::total_cmp);
            let half = sorted.len() / 2;
            match sorted.len() {
                0 => None,
                count if count % 2 == 1 => Some(sorted[half]),
                _ => Some(0.5 * (sorted[half - 1] + sorted[half])),
            }
        }
        LengthScale::WarmupEdgeMean { .. } => {
            let kept: Vec<f64> = samples
                .iter()
                .copied()
                .filter(|x| x.is_finite() && *x > 0.)
                .collect();
            (!kept.is_empty()).then(|| mean(&kept))
        }
    };
    value
        .filter(|x| x.is_finite() && *x > 0.)
        .ok_or_else(|| GasError::Capability("positive warm-up phase length unavailable".into()))
}

/// Fraction of the finite phase-velocity components with `|κ v| > π`, where
/// the map from velocity to colour aliases; `None` without a finite component.
pub fn phase_wrapping(kappa: f64, velocities: &[f64]) -> Option<f64> {
    let (wrapped, count) = velocities
        .iter()
        .filter(|v| v.is_finite())
        .fold((0usize, 0usize), |(wrapped, count), v| {
            (wrapped + usize::from((kappa * v).abs() > PI), count + 1)
        });
    (count > 0).then(|| wrapped as f64 / count as f64)
}

/// Colour of every walker from the viscous-force records `alignment` names,
/// with `force`, `force_valid` and `phase_velocity` of the same records.
/// `PrecedingKick` and `PrecedingForce` read the B2 record of `previous` and
/// mask row `i` iff its `generation[i] != frame.generation[i]`: that record is
/// post-clone, so walkers cloned at `t − 1` stay valid. `MatchedKick` and
/// `ReferenceOffset` read a stage of `frame` and mask `frame.cloned[i]`.
/// Ineligible rows and rows the record does not cover are invalid.
/// `Unavailable` without the record an arm reads (no `previous`, no B stage).
pub(super) fn viscous_force(
    alignment: ColorAlignment,
    threshold: f64,
    previous: Option<&Frame>,
    frame: &Frame,
    kappa: f64,
    state: &mut FrameState,
) -> Availability {
    if !reset(frame, state) {
        return Availability::unavailable("colour frame shape");
    }
    let (record, preceding) = match alignment {
        ColorAlignment::PrecedingKick | ColorAlignment::PrecedingForce => {
            let Some(before) = contiguous(previous, frame) else {
                return Availability::unavailable(NO_PRECEDING_STEP);
            };
            (stage(before, KickStage::B2), true)
        }
        ColorAlignment::MatchedKick { stage: selected } => (stage(frame, selected), false),
        ColorAlignment::ReferenceOffset => (stage(frame, KickStage::B1), false),
    };
    let Some(record) = record else {
        return Availability::unavailable("step holds no viscous-force record of the colour stage");
    };
    let velocity = match alignment {
        ColorAlignment::PrecedingKick | ColorAlignment::MatchedKick { .. } => &record.velocity,
        ColorAlignment::PrecedingForce | ColorAlignment::ReferenceOffset => {
            let Some(velocity) = &frame.v else {
                return Availability::unavailable("frame holds no pre-clone velocity");
            };
            velocity
        }
    };
    colors(record, velocity, preceding, threshold, frame, kappa, state)
}

/// Colour from `recorded_color`: its `force` as amplitude, its `velocity` as
/// phase. `StepAlignment::Preceding` reads `previous.recorded_color` with the
/// generation mask, `Matched` reads `frame.recorded_color` and masks
/// `frame.cloned[i]`.
pub(super) fn recorded_field(
    alignment: StepAlignment,
    threshold: f64,
    previous: Option<&Frame>,
    frame: &Frame,
    kappa: f64,
    state: &mut FrameState,
) -> Availability {
    if !reset(frame, state) {
        return Availability::unavailable("colour frame shape");
    }
    let (record, preceding) = match alignment {
        StepAlignment::Preceding => {
            let Some(before) = contiguous(previous, frame) else {
                return Availability::unavailable(NO_PRECEDING_STEP);
            };
            (before.recorded_color.as_ref(), true)
        }
        StepAlignment::Matched => (frame.recorded_color.as_ref(), false),
    };
    let Some(record) = record else {
        return Availability::unavailable("step holds no record of the colour fields");
    };
    colors(
        record,
        &record.velocity,
        preceding,
        threshold,
        frame,
        kappa,
        state,
    )
}
fn stage(frame: &Frame, stage: KickStage) -> Option<&StageKick> {
    frame.kick.as_ref().and_then(|kick| match stage {
        KickStage::B1 => kick.b1.as_ref(),
        KickStage::B2 => kick.b2.as_ref(),
    })
}

/// `previous` when it is the frame of step `frame.step − 1` on the same rows.
fn contiguous<'a>(previous: Option<&'a Frame>, frame: &Frame) -> Option<&'a Frame> {
    previous
        .filter(|p| p.step.checked_add(1) == Some(frame.step) && p.n == frame.n && p.d == frame.d)
}

/// All-invalid colour and force of the frame's shape; false, with empty
/// fields, when the per-walker records of `frame` do not have that shape.
fn reset(frame: &Frame, state: &mut FrameState) -> bool {
    let (n, d) = (frame.n, frame.d);
    let cells = n.checked_mul(d).filter(|_| {
        n >= 1
            && d >= 1
            && frame.eligible.len() == n
            && frame.generation.len() == n
            && frame.cloned.len() == n
    });
    let (n, cells) = cells.map_or((0, 0), |cells| (n, cells));
    state.color.clear();
    state.color.resize(cells, C::ZERO);
    state.color_valid.clear();
    state.color_valid.resize(n, false);
    state.force = None;
    state.force_valid.clear();
    state.force_valid.resize(n, false);
    state.phase_velocity = None;
    cells > 0
}

/// Rows of `record` that `frame` holds: covered, eligible, finite and the
/// same incarnation. A record of the preceding step is compared by
/// generation, a record of the frame's own step by `frame.cloned`. The phase
/// velocity of a held row is written when finite, zero otherwise.
fn colors(
    record: &StageKick,
    velocity: &[f64],
    preceding: bool,
    threshold: f64,
    frame: &Frame,
    kappa: f64,
    state: &mut FrameState,
) -> Availability {
    let (n, d) = (frame.n, frame.d);
    let cells = state.color.len();
    if record.force.len() != cells
        || velocity.len() != cells
        || record.available.len() != n
        || record.generation.len() != n
    {
        return Availability::unavailable("colour record shape");
    }
    if !kappa.is_finite() {
        return Availability::unavailable("colour phase factor is not finite");
    }
    let (mut force, mut phase) = (vec![0.; cells], vec![0.; cells]);
    for (i, (f, v)) in record
        .force
        .chunks_exact(d)
        .zip(velocity.chunks_exact(d))
        .enumerate()
    {
        let replaced = if preceding {
            record.generation[i] != frame.generation[i]
        } else {
            frame.cloned[i]
        };
        if !record.available[i]
            || !frame.eligible[i]
            || replaced
            || !f.iter().all(|x| x.is_finite())
        {
            continue;
        }
        let row = i * d..(i + 1) * d;
        state.force_valid[i] = true;
        force[row.clone()].copy_from_slice(f);
        if v.iter().all(|x| x.is_finite()) {
            phase[row.clone()].copy_from_slice(v);
        }
        state.color_valid[i] = color_vector(f, v, kappa, threshold, &mut state.color[row]);
    }
    state.force = Some(force);
    state.phase_velocity = Some(phase);
    Availability::Available
}
