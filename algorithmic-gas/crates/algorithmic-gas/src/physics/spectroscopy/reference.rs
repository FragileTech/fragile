//! Comparison of assigned channels with a reference table. The assignment is
//! an input hypothesis; nothing here feeds back into a fit, and no row, order
//! or choice depends on how well a measured number agrees with a reference.
use super::{
    config::{AnalysisConfig, ReferenceEntry},
    contract::EXCHANGE_ODD_REASON,
    report::{
        AnchorPrediction, AnchorSpread, ChannelReport, Comparison, HYPOTHESIS_LABEL, Prediction,
        RatioRow, ReferenceRow,
    },
};
use crate::{Result, physics::qft::math::mean};

/// A rate is a denominator or an anchor only when `value / error` reaches this:
/// below it the first-order error of a ratio is not valid.
const MIN_SIGNIFICANCE: f64 = 3.;
/// Correlation assumed between the rates of two channels. A `ChannelReport`
/// carries no cross-channel covariance, so `compare` passes 0 and says so.
const CORRELATION: f64 = 0.;
/// A reference error above this fraction of its value marks a range.
const BROAD: f64 = 0.02;
/// Reference names that are broad or not experimental states whatever error
/// the table gives them.
const BROAD_NAMES: [&str; 3] = ["f0_500", "a1", "glueball_0pp"];
/// Reference names of the electroweak dashboard mapping.
const ELECTROWEAK: [&str; 6] = ["electron", "muon", "tau", "w_boson", "z_boson", "higgs"];

/// Notes every comparison carries, in this order, before the conditional ones.
const HYPOTHESIS_NOTE: &str = "Hypothesis mapping: every channel-to-reference assignment is \
    an input of this analysis, fixed in the analysis configuration. No reference value enters a \
    prior, a fit window or a channel selection.";
const RATE_NOTE: &str = "The compared quantity is the decay rate of the algorithm-time \
    autocorrelation. It is a mass only under the positive transfer representation \
    (cor-effective-twistor-positive-transfer); the gas is a non-reversible Markov chain and that \
    assumption is not tested here.";
const UNIT_NOTE: &str = "Ratios of rates do not depend on the time unit assigned to a lag \
    (thm-qft-ratio-rescale). They do depend on the integrator step, the recording stride, the \
    estimator, the frame normalisation and the smearing scale, so only channels that share the \
    time unit, the estimator, the scale and the frame normalisation are compared.";
const LOOK_ELSEWHERE_NOTE: &str = "Tensions are (measured - reference)/sigma with sigma^2 = \
    sigma_measured^2 + sigma_reference^2 from first-order error propagation. They carry no \
    look-elsewhere correction for the number of rows or for the choice among operator variants, \
    fit windows and assignments. A tension below 1 is expected in 68% of rows when the \
    hypothesis is true and is also produced by a large error bar; it is not evidence for the \
    assignment.";
const CORRELATION_NOTE: &str = "Channel rates are estimated on the same frames but are \
    treated as uncorrelated because the report carries no cross-channel covariance. A positive \
    correlation makes the quoted ratio errors too large and the tensions too small in magnitude; \
    a negative correlation does the opposite.";
const SPREAD_NOTE: &str = "Anchor spread is the population standard deviation over anchors of \
    the rescaled prediction divided by its mean, the channel's own anchor excluded. It is the \
    relative dispersion of the lattice scales reference/rate and vanishes when every anchor \
    implies the same scale.";
const ELECTROWEAK_NOTE: &str = "Electroweak assignments follow the legacy dashboard mapping of \
    09_qft_calibration; the book records that these proxy channels do not reproduce electroweak \
    mass ratios and treats them as phase-coherence diagnostics.";

/// The channel an assignment key selects: the channel with that id, else the
/// first channel of that specification id that reports a rate.
pub fn assigned<'a>(channels: &'a [ChannelReport], key: &str) -> Option<&'a ChannelReport> {
    channels
        .iter()
        .find(|c| c.id == key)
        .or_else(|| {
            channels
                .iter()
                .find(|c| c.spec.id() == key && c.mass.is_some())
        })
        .or_else(|| channels.iter().find(|c| c.spec.id() == key))
}

/// `[value, error]` of `numerator / denominator` for `[value, error]` inputs
/// with nonzero values, to first order:
/// `σ² = R² [(s_a/a)² + (s_b/b)² − 2 c (s_a/a)(s_b/b)]` with `c` the
/// correlation of the two estimates. Multiplying both values and errors by one
/// factor, a change of the time unit, leaves the result unchanged. A variance
/// that rounds below zero is an error of 0; an undefined error stays NaN.
pub fn ratio(numerator: [f64; 2], denominator: [f64; 2], correlation: f64) -> [f64; 2] {
    let r = numerator[0] / denominator[0];
    let (a, b) = (numerator[1] / numerator[0], denominator[1] / denominator[0]);
    let variance = a * a + b * b - 2. * correlation * a * b;
    // `f64::max` drops a NaN operand: it would turn an undefined error into 0.
    let relative = if variance < 0. { 0. } else { variance.sqrt() };
    [r, r.abs() * relative]
}

/// Signed `(measured − reference) / sqrt(σ_measured² + σ_reference²)` of two
/// `[value, error]` pairs; `None` when both errors vanish.
pub fn tension(measured: [f64; 2], reference: [f64; 2]) -> Option<f64> {
    let sigma = measured[1].hypot(reference[1]);
    let t = (measured[0] - reference[0]) / sigma;
    (sigma > 0. && t.is_finite()).then_some(t)
}

/// `[value, error]` of a rate rescaled by an anchor: `P = M_A r / r_A` with
/// `σ_P² = M_A² σ_R² + R² S_A²`, `R = r / r_A` from `ratio` and `[M_A, S_A]`
/// the anchor's reference.
pub fn prediction(
    rate: [f64; 2],
    anchor: [f64; 2],
    anchor_reference: [f64; 2],
    correlation: f64,
) -> [f64; 2] {
    let [r, error] = ratio(rate, anchor, correlation);
    let [m, s] = anchor_reference;
    [m * r, (m * error).hypot(r * s)]
}

/// Population standard deviation over mean; `None` below two values or for a
/// vanishing mean.
pub fn spread(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64;
    let relative = variance.sqrt() / m.abs();
    (m != 0. && relative.is_finite()).then_some(relative)
}

/// One reference entry with the assignment kept for it.
struct Assigned<'a> {
    entry: &'a ReferenceEntry,
    key: &'a str,
    channel: Option<&'a ChannelReport>,
}
impl Assigned<'_> {
    fn name(&self) -> &str {
        &self.entry.name
    }
    fn rate(&self) -> Option<[f64; 2]> {
        let mass = self.channel?.mass.as_ref()?;
        Some([mass.value, mass.error])
    }
    /// Two names on one channel share one estimate: their ratio is 1 and the
    /// rescaled rate is the anchor's reference, whatever was measured.
    fn shares_channel(&self, other: &Self) -> bool {
        matches!((self.channel, other.channel), (Some(a), Some(b)) if a.id == b.id)
    }
    fn reference(&self) -> [f64; 2] {
        [self.entry.value, self.entry.error]
    }
    /// Same time unit, estimator, scale and frame normalisation: a Euclidean
    /// rate is per length, a multiscale copy another observable, only the
    /// frame mean has a transfer-matrix reading, and the valid-count and
    /// fixed-`1/N` frame averages are two different observables
    /// (`09_qft_calibration`). Two channels that state no normalisation are
    /// compared as before.
    fn comparable(&self, other: &Self) -> bool {
        match (self.channel, other.channel) {
            (Some(a), Some(b)) => {
                a.estimator == b.estimator
                    && a.scale == b.scale
                    && a.normalization == b.normalization
                    && a.mass.as_ref().map(|m| m.time_unit) == b.mass.as_ref().map(|m| m.time_unit)
            }
            _ => false,
        }
    }
}
fn once(notes: &mut Vec<String>, note: String) {
    if !notes.contains(&note) {
        notes.push(note);
    }
}
fn finite(pair: [f64; 2]) -> Option<[f64; 2]> {
    pair.iter().all(|x| x.is_finite()).then_some(pair)
}

/// A rate that may enter a ratio: finite and positive with a finite positive
/// error.
fn positive(row: &Assigned<'_>, notes: &mut Vec<String>) -> Option<[f64; 2]> {
    let rate = row.rate()?;
    if rate.iter().all(|x| x.is_finite() && *x > 0.) {
        return Some(rate);
    }
    once(
        notes,
        format!(
            "'{}': rate {} +- {} is not finite and positive with a finite positive error; it \
             enters no ratio.",
            row.name(),
            rate[0],
            rate[1]
        ),
    );
    None
}

/// A rate that may be a denominator or an anchor.
fn significant(row: &Assigned<'_>, notes: &mut Vec<String>) -> Option<[f64; 2]> {
    let rate = positive(row, notes)?;
    if rate[0] / rate[1] >= MIN_SIGNIFICANCE {
        return Some(rate);
    }
    once(
        notes,
        format!(
            "'{}': rate {} +- {} is less than 3 sigma from zero; it is not used as a denominator \
             or anchor because first-order ratio errors are not valid there.",
            row.name(),
            rate[0],
            rate[1]
        ),
    );
    None
}

/// The two rates of `rows[a] / rows[b]` when both exist, are comparable and
/// pass their gates. Names in a note follow the table order.
fn gated(rows: &[Assigned<'_>], a: usize, b: usize, notes: &mut Vec<String>) -> Option<[f64; 4]> {
    let (numerator, denominator) = (&rows[a], &rows[b]);
    numerator.rate()?;
    denominator.rate()?;
    if numerator.shares_channel(denominator) {
        let (first, second) = (&rows[a.min(b)], &rows[a.max(b)]);
        once(
            notes,
            format!(
                "'{}' and '{}' are assigned the same channel; their ratio is 1 by construction \
                 and is not compared.",
                first.name(),
                second.name()
            ),
        );
        return None;
    }
    if !numerator.comparable(denominator) {
        let (first, second) = (&rows[a.min(b)], &rows[a.max(b)]);
        once(
            notes,
            format!(
                "'{}' and '{}' differ in time unit, estimator, scale or frame normalisation and \
                 are not compared.",
                first.name(),
                second.name()
            ),
        );
        return None;
    }
    let top = positive(numerator, notes);
    let bottom = significant(denominator, notes);
    let ([r, s], [q, e]) = (top?, bottom?);
    Some([r, s, q, e])
}

/// Number of algebraically independent ratios among the measured rows: names
/// that occur minus connected groups of names.
fn independent(names: usize, measured: &[[usize; 2]]) -> usize {
    let mut group: Vec<usize> = (0..names).collect();
    for [a, b] in measured {
        let (from, to) = (group[*a], group[*b]);
        for g in &mut group {
            if *g == from {
                *g = to;
            }
        }
    }
    let mut seen = vec![false; names];
    let mut groups = vec![false; names];
    for [a, b] in measured {
        seen[*a] = true;
        seen[*b] = true;
        groups[group[*a]] = true;
    }
    seen.iter().filter(|s| **s).count() - groups.iter().filter(|g| **g).count()
}

/// Reference rows, one prediction table per anchor (`scale = reference /
/// measured` of the anchor), ratio table with tensions and the spread over
/// anchors. `None` when no assigned channel reports a rate. Ratios do not
/// depend on the time unit; rates of different estimators are never divided.
/// The notes state that the tensions carry no look-elsewhere correction and
/// ignore the correlation between channels.
///
/// Rows follow the order of the reference table and a ratio is always a later
/// entry over an earlier one. Of several keys assigned to one reference name
/// the first in key order with a rate is kept. An anchor is an input: its own
/// row is no prediction and is left out of its table. Two names assigned the
/// same channel are not compared: their ratio is 1 by construction.
pub fn compare(
    channels: &[ChannelReport],
    analysis: &AnalysisConfig,
) -> Result<Option<Comparison>> {
    analysis.validate()?;
    let table = &analysis.reference;
    let mut notes = vec![];
    let mut rows = vec![];
    for entry in &table.entries {
        let keys: Vec<&String> = analysis
            .assignments
            .iter()
            .filter(|(_, name)| **name == entry.name)
            .map(|(key, _)| key)
            .collect();
        let rated = |key: &str| assigned(channels, key).is_some_and(|c| c.mass.is_some());
        let first = keys.iter().copied().find(|k| rated(k.as_str()));
        let Some(kept) = first.or_else(|| keys.first().copied()) else {
            continue;
        };
        for key in keys.iter().filter(|k| **k != kept) {
            once(
                &mut notes,
                format!(
                    "'{key}' is also assigned to '{}' and is not used: the first assignment \
                     with a rate is kept, never the best agreeing one.",
                    entry.name
                ),
            );
        }
        rows.push(Assigned {
            entry,
            key: kept.as_str(),
            channel: assigned(channels, kept),
        });
    }
    if rows.iter().all(|r| r.rate().is_none()) {
        return Ok(None);
    }
    for row in &rows {
        if let Some(channel) = row.channel
            && channel.availability.reason() == Some(EXCHANGE_ODD_REASON)
        {
            notes.push(format!(
                "'{}' assigned to '{}' is exchange-odd and cancels on the mutual pairing of \
                 this run; it reports no rate.",
                channel.id,
                row.name()
            ));
        }
        if BROAD_NAMES.contains(&row.name()) || row.entry.error > BROAD * row.entry.value {
            notes.push(format!(
                "Reference '{}' ({}) is not a narrow experimental state: its table error is a \
                 range, not a Gaussian standard deviation.",
                row.name(),
                row.entry.source
            ));
        }
    }
    if rows.iter().any(|r| ELECTROWEAK.contains(&r.name())) {
        notes.push(ELECTROWEAK_NOTE.into());
    }
    let reference = rows
        .iter()
        .map(|row| ReferenceRow {
            name: row.name().into(),
            channel: row.channel.map_or(row.key, |c| &c.id).into(),
            reference: row.entry.value,
            reference_error: row.entry.error,
            unit: table.unit.clone(),
            measured: row.rate().and_then(finite),
            estimator: row.rate().and(row.channel).and_then(|c| c.estimator),
            normalization: row.rate().and(row.channel).and_then(|c| c.normalization),
        })
        .collect();
    let mut ratios = vec![];
    let mut measured = vec![];
    for earlier in 0..rows.len() {
        for later in earlier + 1..rows.len() {
            let expected = ratio(rows[later].reference(), rows[earlier].reference(), 0.);
            let value = gated(&rows, later, earlier, &mut notes)
                .and_then(|[r, s, q, e]| finite(ratio([r, s], [q, e], CORRELATION)));
            if value.is_some() {
                measured.push([later, earlier]);
            }
            ratios.push(RatioRow {
                numerator: rows[later].name().into(),
                denominator: rows[earlier].name().into(),
                measured: value,
                reference: expected[0],
                tension_sigma: value.and_then(|v| tension(v, expected)),
            });
        }
    }
    let mut anchors: Vec<AnchorPrediction> = vec![];
    for name in &analysis.anchors {
        if anchors.iter().any(|a| a.anchor == *name) {
            continue;
        }
        let at = rows.iter().position(|r| r.name() == name);
        let usable = at.and_then(|a| significant(&rows[a], &mut notes).map(|rate| (a, rate)));
        let Some((a, rate)) = usable else {
            notes.push(format!(
                "Anchor '{name}' has no assigned channel with a usable rate; no prediction \
                 table is produced for it."
            ));
            anchors.push(AnchorPrediction {
                anchor: name.clone(),
                ..AnchorPrediction::default()
            });
            continue;
        };
        let target = rows[a].reference();
        let predictions = (0..rows.len())
            .filter(|i| *i != a)
            .map(|i| {
                let predicted = gated(&rows, i, a, &mut notes).and_then(|[r, s, q, e]| {
                    finite(prediction([r, s], [q, e], target, CORRELATION))
                });
                let [value, error] = rows[i].reference();
                Prediction {
                    name: rows[i].name().into(),
                    predicted,
                    reference: value,
                    tension_sigma: predicted.and_then(|p| tension(p, [value, error])),
                }
            })
            .collect();
        anchors.push(AnchorPrediction {
            anchor: name.clone(),
            scale: finite(ratio(target, rate, 0.)),
            predictions,
        });
    }
    let spread_of = |row: &Assigned<'_>| {
        let predicted: Vec<f64> = anchors
            .iter()
            .flat_map(|a| &a.predictions)
            .filter(|p| p.name == row.name())
            .filter_map(|p| p.predicted.map(|[value, _]| value))
            .collect();
        AnchorSpread {
            name: row.name().into(),
            spread: spread(&predicted),
        }
    };
    let anchor_spread: Vec<AnchorSpread> = if anchors.is_empty() {
        vec![]
    } else {
        rows.iter().map(spread_of).collect()
    };
    if anchor_spread.iter().any(|s| s.spread.is_some()) {
        notes.push(SPREAD_NOTE.into());
    }
    let anchors_note = format!(
        "Anchor rescaling uses one reference value as an input; the anchor's own row is not a \
         prediction and is omitted. Of the {} ratio rows only {} are algebraically independent, \
         and rows that share a channel are statistically correlated.",
        measured.len(),
        independent(rows.len(), &measured)
    );
    let always = [
        HYPOTHESIS_NOTE.into(),
        RATE_NOTE.into(),
        UNIT_NOTE.into(),
        anchors_note,
        LOOK_ELSEWHERE_NOTE.into(),
        CORRELATION_NOTE.into(),
    ];
    Ok(Some(Comparison {
        label: HYPOTHESIS_LABEL.into(),
        reference,
        anchors,
        ratios,
        anchor_spread,
        notes: always.into_iter().chain(notes).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_ratios_count_names_minus_connected_groups() {
        assert_eq!(independent(4, &[]), 0);
        assert_eq!(
            independent(4, &[[1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]),
            3
        );
        assert_eq!(independent(5, &[[1, 0], [3, 2]]), 2);
        assert_eq!(independent(5, &[[1, 0], [3, 2], [4, 2], [4, 3]]), 3);
    }
}
