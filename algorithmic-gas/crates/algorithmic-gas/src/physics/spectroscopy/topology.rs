//! Companions → sites, pairs and triplets of one source time, with masks and
//! coverage counters.
//!
//! The topology is colour-blind: it applies the structural masks of the
//! companion record only. Masks that depend on the evaluated state are decided
//! by the operators at every source and sink time.
use super::{
    config::{CompanionChoice, MeasurementConfig},
    contract::{CompanionMask, Companions, Element, ElementKind, Frame, Topology},
    report::Coverage,
};

/// Elements of `kind` on `frame`:
/// - `Site`: every eligible walker;
/// - `DistancePair` / `CloningPair`: `(i, c(i))`; with `K > 1` companions
///   `CompanionChoice::First` takes the first entry that builds an element and
///   `All` one element per entry with weight `1 / K_valid(i)`, `K_valid(i)`
///   the number of elements of anchor `i`;
/// - `Triplet`: `(i, j_k, c)`, the k-th distance companion with the first
///   cloning companion that builds an element. A triplet with `j_k = c` is
///   kept and counts in `K_valid(i)`; the accumulator masks it for operators
///   without `Signature::degenerate`.
///
/// Entries are masked by `Companions::mask` and every candidate is counted
/// once in `coverage`: `valid` when it is kept, else `masked_historical`,
/// `masked_self`, or `masked_ineligible` for an ineligible walker on either
/// end and for an entry the engine did not sample. Candidates are the `n`
/// walkers under `First` and for sites, the `n · K` entries under `All`. A row
/// without an element under `First` carries the reason of its first sampled
/// entry. A triplet masked on both companions is counted once, under the first
/// of `masked_historical`, `masked_ineligible`, `masked_self` that either
/// companion gives. `frames` and `empty_frames` stay 0: whether a frame has a
/// valid element is known after evaluation, where a masked candidate also
/// leaves `valid`.
///
/// A frame without the companion record gives an empty topology. Elements are
/// stored as sampled, carry the source-time generations of their walkers and
/// no direction; directed operator modes orient inside `evaluate`. `frame` is
/// valid (`Frame::validate`).
pub fn build(frame: &Frame, kind: ElementKind, measurement: &MeasurementConfig) -> Topology {
    let (eligible, choice) = (&frame.eligible[..], measurement.companions);
    let mut elements = vec![];
    let mut coverage = Coverage::default();
    for i in 0..frame.n {
        let own = i as u32;
        let row = match (kind, &frame.distance, &frame.cloning) {
            (ElementKind::Site, ..) if eligible[i] => vec![Ok([own; 2])],
            (ElementKind::Site, ..) => vec![Err(CompanionMask::Ineligible)],
            (ElementKind::DistancePair, Some(companions), _)
            | (ElementKind::CloningPair, _, Some(companions)) => {
                candidates(companions, i, eligible, choice)
                    .into_iter()
                    .map(|j| j.map(|j| [j, own]))
                    .collect()
            }
            (ElementKind::Triplet, Some(distance), Some(cloning)) => {
                let k = first(cloning, i, eligible);
                candidates(distance, i, eligible, choice)
                    .into_iter()
                    .map(|j| both(j, k))
                    .collect()
            }
            _ => vec![],
        };
        let kept = row.iter().filter(|candidate| candidate.is_ok()).count();
        for candidate in row {
            match candidate {
                Ok([j, k]) => elements.push(Element {
                    walkers: [own, j, k],
                    kind,
                    weight: 1. / kept as f64,
                    generation: [own, j, k].map(|w| frame.generation[w as usize]),
                }),
                Err(CompanionMask::Historical) => coverage.masked_historical += 1,
                Err(CompanionMask::Unsampled | CompanionMask::Ineligible) => {
                    coverage.masked_ineligible += 1
                }
                Err(CompanionMask::SelfCompanion) => coverage.masked_self += 1,
            }
        }
    }
    coverage.valid = elements.len() as u64;
    Topology {
        step: frame.step,
        kind,
        involutive: is_involutive(&elements),
        elements,
        coverage,
    }
}

/// Every pair `(i, j)` has its mirror `(j, i)` with the same weight. False for
/// an empty list and for elements that are not pairs.
pub fn is_involutive(elements: &[Element]) -> bool {
    !elements.is_empty() && mirrors(elements).iter().all(Option::is_some)
}

/// `[elements]`: index of the mirror `(j, i)` of every pair `(i, j)`, of the
/// same kind and with bitwise the same weight; `None` for an unmirrored pair,
/// a pair `(i, i)` and an element that is not a pair. The matching is one to
/// one: of `m` copies of `(i, j)` and `m' < m` copies of `(j, i)` the first
/// `m'` copies are mirrored. An exchange-odd operator cancels exactly between
/// an element and its mirror.
pub fn mirrors(elements: &[Element]) -> Vec<Option<usize>> {
    let pair = |e: usize| {
        let Element {
            walkers: [i, j, _],
            kind,
            weight,
            ..
        } = elements[e];
        (kind, i.min(j), i.max(j), weight.to_bits())
    };
    let reversed = |e: usize| elements[e].walkers[0] > elements[e].walkers[1];
    let mut order: Vec<usize> = (0..elements.len())
        .filter(|&e| {
            let element = &elements[e];
            element.kind.arity() == 2 && element.walkers[0] != element.walkers[1]
        })
        .collect();
    order.sort_unstable_by_key(|&e| (pair(e), reversed(e), e));
    let mut mirror = vec![None; elements.len()];
    for group in order.chunk_by(|&a, &b| pair(a) == pair(b)) {
        let (forward, backward) = group.split_at(group.partition_point(|&e| !reversed(e)));
        for (&a, &b) in forward.iter().zip(backward) {
            mirror[a] = Some(b);
            mirror[b] = Some(a);
        }
    }
    mirror
}

/// Every walker of `element` has in `generation` (`[n]`, of a sink frame) the
/// generation it had at the source time: the sink test of
/// `IdentityPolicy::Incarnation`. `IdentityPolicy::Slot` applies none.
pub fn same_incarnation(element: &Element, generation: &[u64]) -> bool {
    element
        .walkers
        .iter()
        .zip(&element.generation)
        .all(|(&w, source)| generation.get(w as usize) == Some(source))
}

/// Slot of entry `k` of row `i`, or why it builds no element.
fn entry(
    companions: &Companions,
    i: usize,
    k: usize,
    eligible: &[bool],
) -> Result<u32, CompanionMask> {
    match companions.mask(i, k, eligible) {
        Some(mask) => Err(mask),
        None => Ok(companions.slot[i * companions.count + k]),
    }
}

/// Slot of `Companions::first`; a row without one carries the reason of its
/// first sampled entry.
fn first(companions: &Companions, i: usize, eligible: &[bool]) -> Result<u32, CompanionMask> {
    match companions.first(i, eligible) {
        Some(k) => entry(companions, i, k, eligible),
        None => (0..companions.count)
            .map(|k| entry(companions, i, k, eligible))
            .find(|candidate| *candidate != Err(CompanionMask::Unsampled))
            .unwrap_or(Err(CompanionMask::Unsampled)),
    }
}

/// Candidates of row `i`: every entry under `All`, one under `First`.
fn candidates(
    companions: &Companions,
    i: usize,
    eligible: &[bool],
    choice: CompanionChoice,
) -> Vec<Result<u32, CompanionMask>> {
    match choice {
        CompanionChoice::First => vec![first(companions, i, eligible)],
        CompanionChoice::All => (0..companions.count)
            .map(|k| entry(companions, i, k, eligible))
            .collect(),
    }
}

/// Companions `[j, k]` of a triplet; when both are masked, the reason whose
/// counter comes first: historical, ineligible (with unsampled), self.
fn both(
    j: Result<u32, CompanionMask>,
    k: Result<u32, CompanionMask>,
) -> Result<[u32; 2], CompanionMask> {
    let rank = |mask: CompanionMask| match mask {
        CompanionMask::Historical => 0,
        CompanionMask::Unsampled | CompanionMask::Ineligible => 1,
        CompanionMask::SelfCompanion => 2,
    };
    match (j, k) {
        (Ok(j), Ok(k)) => Ok([j, k]),
        (Err(a), Err(b)) => Err(if rank(b) < rank(a) { b } else { a }),
        (Err(mask), Ok(_)) | (Ok(_), Err(mask)) => Err(mask),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_triplet_masked_on_both_companions_carries_the_reason_of_the_earlier_counter() {
        use CompanionMask::{Historical, Ineligible, SelfCompanion, Unsampled};
        assert_eq!(both(Ok(1), Ok(2)), Ok([1, 2]));
        assert_eq!(both(Err(SelfCompanion), Ok(2)), Err(SelfCompanion));
        assert_eq!(both(Ok(1), Err(Historical)), Err(Historical));
        assert_eq!(both(Err(SelfCompanion), Err(Historical)), Err(Historical));
        assert_eq!(both(Err(Historical), Err(Unsampled)), Err(Historical));
        assert_eq!(both(Err(Unsampled), Err(Historical)), Err(Historical));
        assert_eq!(both(Err(Ineligible), Err(Historical)), Err(Historical));
        assert_eq!(both(Err(SelfCompanion), Err(Unsampled)), Err(Unsampled));
        assert_eq!(both(Err(Ineligible), Err(SelfCompanion)), Err(Ineligible));
    }
    #[test]
    fn a_row_without_an_element_reports_its_first_sampled_entry() {
        let companions = Companions {
            count: 3,
            slot: vec![0, 0, 1, 0, 0, 0],
            generation: vec![0; 6],
            valid: vec![false, true, true, false, false, false],
            historical: vec![false, false, true, false, false, false],
            mutual: false,
        };
        let eligible = [true, true];
        assert_eq!(
            first(&companions, 0, &eligible),
            Err(CompanionMask::SelfCompanion)
        );
        assert_eq!(
            first(&companions, 1, &eligible),
            Err(CompanionMask::Unsampled)
        );
    }
}
