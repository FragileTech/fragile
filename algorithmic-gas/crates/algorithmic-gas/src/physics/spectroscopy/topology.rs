//! Companions → sites, pairs and triplets of one source time, with masks and
//! coverage counters.
use super::{
    config::MeasurementConfig,
    contract::{Element, ElementKind, Frame, Topology},
    report::Coverage,
};

/// Elements of `kind` on `frame`:
/// - `Site`: every eligible walker;
/// - `DistancePair` / `CloningPair`: `(i, c(i))`; with `K > 1` distance
///   companions `CompanionChoice::First` takes the first valid one and `All`
///   one element per companion with weight `1 / K_valid(i)`;
/// - `Triplet`: `(i, j_k, c)`, the k-th distance companion with the single
///   cloning companion. A triplet with `j_k = c` is kept; the accumulator
///   masks it for operators without `Signature::degenerate`.
///
/// Entries are masked by `Companions::mask` and counted in `coverage`:
/// historical sources, ineligible or revived rows on either end,
/// self-companions in every family. A frame without the companion record
/// gives an empty topology. Elements are stored as sampled and carry no
/// direction; directed operator modes orient inside `evaluate`.
pub fn build(frame: &Frame, kind: ElementKind, measurement: &MeasurementConfig) -> Topology {
    let _ = measurement;
    Topology {
        step: frame.step,
        kind,
        elements: vec![],
        involutive: false,
        coverage: Coverage::default(),
    }
}

/// Every pair `(i, j)` has its mirror `(j, i)` with the same weight. False for
/// an empty list and for elements that are not pairs.
pub fn is_involutive(elements: &[Element]) -> bool {
    let _ = elements;
    false
}
