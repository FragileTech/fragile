use algorithmic_gas::physics::spectroscopy::{
    config::{CompanionChoice, MeasurementConfig},
    contract::{Companions, Element, ElementKind, Frame, Topology},
    report::Coverage,
    topology::{build, is_involutive, mirrors, same_incarnation},
};
use std::collections::BTreeSet;

/// Unnormalised colours `(re, im)` of six walkers in three dimensions.
const RAW: [[[f64; 2]; 3]; 6] = [
    [[1., 0.], [0., 2.], [-1., 0.]],
    [[2., 1.], [1., 0.], [1., -1.]],
    [[0.5, 0.], [-1., 2.], [3., 0.]],
    [[-1., -1.], [2., 0.], [0., 0.5]],
    [[3., 0.], [1., 1.], [0., -2.]],
    [[1., 0.], [1., 0.], [1., 0.]],
];

/// Imaginary part of the overlap of the normalised colours `i` and `j`: odd
/// under their exchange, bit for bit.
fn overlap_imaginary(i: u32, j: u32) -> f64 {
    let (a, b) = (&RAW[i as usize], &RAW[j as usize]);
    let norm = |c: &[[f64; 2]; 3]| {
        c.iter()
            .map(|z| z[0] * z[0] + z[1] * z[1])
            .sum::<f64>()
            .sqrt()
    };
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(a, b)| a[0] * b[1] - a[1] * b[0])
        .sum();
    sum / (norm(a) * norm(b))
}

/// Eligible walkers at rest on a line, generation `10 + slot`.
fn frame(n: usize) -> Frame {
    Frame {
        step: 3,
        n,
        d: 1,
        x: (0..n).map(|i| i as f64).collect(),
        eligible: vec![true; n],
        generation: (0..n as u64).map(|i| 10 + i).collect(),
        cloned: vec![false; n],
        revived: vec![false; n],
        ..Frame::default()
    }
}

/// Sampled current-frame companions `[n, count]`.
fn companions(count: usize, slot: &[u32]) -> Companions {
    Companions {
        count,
        slot: slot.to_vec(),
        generation: vec![0; slot.len()],
        valid: vec![true; slot.len()],
        historical: vec![false; slot.len()],
        mutual: false,
    }
}

/// Weighted sum over the elements of the imaginary overlap of their pair.
fn exchange_odd_sum(topology: &Topology) -> f64 {
    topology.elements.iter().fold(0., |sum, e| {
        sum + e.weight * overlap_imaginary(e.walkers[0], e.walkers[1])
    })
}

/// The topology of a valid frame, with the invariants every topology keeps.
fn built(frame: &Frame, kind: ElementKind, choice: CompanionChoice) -> Topology {
    frame.validate().unwrap();
    let measurement = MeasurementConfig {
        companions: choice,
        ..MeasurementConfig::default()
    };
    let topology = build(frame, kind, &measurement);
    assert_eq!(build(frame, kind, &measurement), topology);
    topology.validate(frame.n).unwrap();
    let coverage = &topology.coverage;
    assert_eq!((topology.step, topology.kind), (frame.step, kind));
    assert_eq!(coverage.valid, topology.elements.len() as u64);
    assert_eq!(
        coverage.masked(),
        coverage.masked_historical + coverage.masked_ineligible + coverage.masked_self
    );
    assert_eq!((coverage.frames, coverage.empty_frames), (0, 0));
    assert_eq!(topology.involutive, is_involutive(&topology.elements));
    topology
}
fn walkers(topology: &Topology) -> Vec<[u32; 3]> {
    topology.elements.iter().map(|e| e.walkers).collect()
}
fn weights(topology: &Topology) -> Vec<f64> {
    topology.elements.iter().map(|e| e.weight).collect()
}
fn pair(i: u32, j: u32, kind: ElementKind, weight: f64) -> Element {
    Element {
        walkers: [i, j, i],
        kind,
        weight,
        generation: [0; 3],
    }
}
#[test]
fn mutual_pairing_of_an_odd_population_masks_the_self_companion_and_stays_involutive() {
    // Distance companions recorded at step 1 of an eleven-walker matching.
    let slot = [7, 8, 6, 5, 9, 3, 2, 0, 1, 4, 10];
    let mut frame = frame(11);
    frame.distance = Some(Companions {
        mutual: true,
        ..companions(1, &slot)
    });
    let topology = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    let expected: Vec<[u32; 3]> = (0..10).map(|i| [i, slot[i as usize], i]).collect();
    assert_eq!(walkers(&topology), expected);
    assert_eq!(weights(&topology), vec![1.; 10]);
    assert_eq!(
        topology.coverage,
        Coverage {
            valid: 10,
            masked_self: 1,
            ..Coverage::default()
        }
    );
    assert!(topology.involutive);
    let mirror: Vec<Option<usize>> = (0..10).map(|i| Some(slot[i] as usize)).collect();
    assert_eq!(mirrors(&topology.elements), mirror);
    // An exchange-odd quantity cancels over the interleaved mirrors.
    let odd = |e: &Element| (0.7 * e.walkers[0] as f64).sin() - (0.7 * e.walkers[1] as f64).sin();
    let sum = topology.elements.iter().fold(0., |sum, e| sum + odd(e));
    assert!(topology.elements.iter().any(|e| odd(e).abs() > 0.5));
    assert!(sum.abs() < 11e-14);
}
#[test]
fn involutivity_is_read_from_the_built_elements_and_not_from_the_engine_flag() {
    let mut frame = frame(6);
    frame.cloning = Some(Companions {
        mutual: true,
        ..companions(1, &[1, 2, 3, 4, 5, 0])
    });
    let cycle = built(&frame, ElementKind::CloningPair, CompanionChoice::First);
    assert_eq!(cycle.coverage.valid, 6);
    assert!(!cycle.involutive);
    assert_eq!(mirrors(&cycle.elements), vec![None; 6]);
    frame.cloning = Some(companions(1, &[1, 0, 3, 2, 5, 4]));
    let involution = built(&frame, ElementKind::CloningPair, CompanionChoice::First);
    assert!(involution.involutive);
}
#[test]
fn exchange_odd_sum_vanishes_exactly_on_an_involution_and_not_on_a_cycle() {
    let mut frame = frame(6);
    frame.distance = Some(companions(1, &[1, 0, 3, 2, 5, 4]));
    let involution = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    assert_eq!(exchange_odd_sum(&involution), 0.);
    frame.distance = Some(companions(1, &[1, 2, 3, 4, 5, 0]));
    let cycle = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    assert!((exchange_odd_sum(&cycle) / 6. - 0.206741558050249).abs() < 1e-14);
}
#[test]
fn unmirrored_element_of_a_partly_mutual_map_is_the_one_an_exchange_odd_mean_keeps() {
    let mut frame = frame(6);
    frame.distance = Some(companions(1, &[4, 1, 3, 2, 0, 2]));
    let topology = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    assert_eq!(
        walkers(&topology),
        [[0, 4, 0], [2, 3, 2], [3, 2, 3], [4, 0, 4], [5, 2, 5]]
    );
    assert_eq!(topology.coverage.masked_self, 1);
    assert!(!topology.involutive);
    let mirror = mirrors(&topology.elements);
    assert_eq!(mirror, [Some(3), Some(2), Some(1), Some(0), None]);
    for (e, m) in mirror.iter().enumerate() {
        let [i, j, _] = topology.elements[e].walkers;
        if let Some(m) = *m {
            let [k, l, _] = topology.elements[m].walkers;
            assert_eq!((i, j), (l, k));
            assert_eq!(overlap_imaginary(i, j) + overlap_imaginary(k, l), 0.);
        } else {
            assert!((overlap_imaginary(i, j) - 0.305887645160749).abs() < 1e-14);
        }
    }
}
#[test]
fn mirrored_pairs_with_unequal_weights_are_not_an_involution() {
    let mut frame = frame(6);
    let mut distance = companions(2, &[0, 0, 2, 0, 1, 3, 2, 0, 0, 0, 0, 0]);
    distance.valid = vec![
        false, false, true, false, true, true, true, false, false, false, false, false,
    ];
    frame.distance = Some(distance);
    let topology = built(&frame, ElementKind::DistancePair, CompanionChoice::All);
    assert_eq!(
        walkers(&topology),
        [[1, 2, 1], [2, 1, 2], [2, 3, 2], [3, 2, 3]]
    );
    assert_eq!(weights(&topology), [1., 0.5, 0.5, 1.]);
    assert_eq!(
        topology.coverage,
        Coverage {
            valid: 4,
            masked_ineligible: 8,
            ..Coverage::default()
        }
    );
    assert!(!topology.involutive);
    assert_eq!(mirrors(&topology.elements), vec![None; 4]);
    // The residual of the cancellation, (Im q_12 + Im q_32) / 2.
    assert!((exchange_odd_sum(&topology) - 0.369675690453686).abs() < 1e-14);
}
#[test]
fn two_mutual_rounds_with_equal_weights_are_an_involution() {
    let mut frame = frame(4);
    frame.distance = Some(companions(2, &[1, 2, 0, 3, 3, 0, 2, 1]));
    let topology = built(&frame, ElementKind::DistancePair, CompanionChoice::All);
    assert_eq!(weights(&topology), vec![0.5; 8]);
    assert!(topology.involutive);
    let mirror = mirrors(&topology.elements);
    for (e, m) in mirror.iter().enumerate() {
        let m = m.unwrap();
        assert_eq!(mirror[m], Some(e));
        let ([i, j, _], [k, l, _]) = (topology.elements[e].walkers, topology.elements[m].walkers);
        assert_eq!((i, j), (l, k));
    }
    assert!(exchange_odd_sum(&topology).abs() < 4e-14);
    let first = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    assert_eq!(
        walkers(&first),
        [[0, 1, 0], [1, 0, 1], [2, 3, 2], [3, 2, 3]]
    );
    assert!(first.involutive);
}
#[test]
fn three_companions_give_the_first_buildable_entry_or_every_entry_weighted_by_the_survivors() {
    let mut frame = frame(5);
    frame.eligible[4] = false;
    let mut distance = companions(3, &[0, 2, 3, 4, 2, 0, 3, 3, 1, 0, 0, 0, 0, 0, 1]);
    distance.historical[4] = true;
    distance.historical[14] = true;
    for e in [9, 10, 11, 12] {
        distance.valid[e] = false;
    }
    frame.distance = Some(distance);
    let first = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    assert_eq!(walkers(&first), [[0, 2, 0], [1, 0, 1], [2, 3, 2]]);
    assert_eq!(weights(&first), [1.; 3]);
    assert_eq!(
        first.coverage,
        Coverage {
            valid: 3,
            masked_ineligible: 2,
            ..Coverage::default()
        }
    );
    let all = built(&frame, ElementKind::DistancePair, CompanionChoice::All);
    assert_eq!(
        walkers(&all),
        [
            [0, 2, 0],
            [0, 3, 0],
            [1, 0, 1],
            [2, 3, 2],
            [2, 3, 2],
            [2, 1, 2]
        ]
    );
    assert_eq!(weights(&all), [0.5, 0.5, 1., 1. / 3., 1. / 3., 1. / 3.]);
    assert_eq!(
        all.coverage,
        Coverage {
            valid: 6,
            masked_historical: 2,
            masked_ineligible: 6,
            masked_self: 1,
            ..Coverage::default()
        }
    );
    assert_eq!(all.coverage.valid + all.coverage.masked(), 15);
    for anchor in 0..3 {
        let total: f64 = all
            .elements
            .iter()
            .filter(|e| e.walkers[0] == anchor)
            .map(|e| e.weight)
            .sum();
        assert!((total - 1.).abs() < 1e-15);
    }
}
#[test]
fn every_pair_mask_is_counted_once_in_the_order_of_the_companion_rule() {
    let mut frame = frame(6);
    frame.eligible[2] = false;
    frame.cloned[2] = true;
    frame.revived[2] = true;
    let mut cloning = companions(1, &[0, 3, 0, 2, 4, 0]);
    cloning.valid[0] = false;
    cloning.historical[1] = true;
    frame.cloning = Some(cloning);
    let topology = built(&frame, ElementKind::CloningPair, CompanionChoice::First);
    assert_eq!(walkers(&topology), [[5, 0, 5]]);
    assert_eq!(
        topology.coverage,
        Coverage {
            valid: 1,
            masked_historical: 1,
            masked_ineligible: 3,
            masked_self: 1,
            ..Coverage::default()
        }
    );
    assert_eq!(topology.coverage.valid + topology.coverage.masked(), 6);
    assert!(!topology.involutive);
    // A historical source of a revived row is counted as historical.
    frame.cloning.as_mut().unwrap().historical[2] = true;
    let topology = built(&frame, ElementKind::CloningPair, CompanionChoice::All);
    assert_eq!(
        (
            topology.coverage.masked_historical,
            topology.coverage.masked_ineligible
        ),
        (2, 2)
    );
}
#[test]
fn triplets_join_the_distance_and_the_cloning_companion_and_keep_coinciding_companions() {
    let mut frame = frame(8);
    frame.distance = Some(companions(1, &[1, 0, 0, 0, 1, 5, 3, 4]));
    frame.cloning = Some(companions(1, &[2, 2, 1, 0, 2, 2, 5, 3]));
    let topology = built(&frame, ElementKind::Triplet, CompanionChoice::First);
    assert_eq!(
        walkers(&topology),
        [
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1],
            [3, 0, 0],
            [4, 1, 2],
            [6, 3, 5],
            [7, 4, 3]
        ]
    );
    assert_eq!(weights(&topology), [1.; 7]);
    assert_eq!(
        topology.coverage,
        Coverage {
            valid: 7,
            masked_self: 1,
            ..Coverage::default()
        }
    );
    assert_eq!(topology.elements[5].generation, [16, 13, 15]);
    assert!(!topology.involutive);
    assert_eq!(mirrors(&topology.elements), vec![None; 7]);
}
#[test]
fn two_mutual_pairings_of_an_odd_population_never_repeat_a_triplet() {
    let mut frame = frame(11);
    frame.distance = Some(companions(1, &[7, 8, 6, 5, 9, 3, 2, 0, 1, 4, 10]));
    frame.cloning = Some(companions(1, &[1, 0, 3, 2, 5, 4, 8, 7, 6, 10, 9]));
    let topology = built(&frame, ElementKind::Triplet, CompanionChoice::First);
    assert_eq!(
        topology.coverage,
        Coverage {
            valid: 9,
            masked_self: 2,
            ..Coverage::default()
        }
    );
    assert!(
        topology
            .elements
            .iter()
            .all(|e| e.walkers[0] != 7 && e.walkers[0] != 10)
    );
    let triples: BTreeSet<[u32; 3]> = topology
        .elements
        .iter()
        .map(|e| {
            let mut sorted = e.walkers;
            sorted.sort_unstable();
            sorted
        })
        .collect();
    assert_eq!(triples.len(), 9);
    assert!(!topology.involutive);
}
#[test]
fn triplet_masks_combine_both_roles_and_weights_count_the_surviving_triplets() {
    let mut frame = frame(5);
    frame.eligible[4] = false;
    frame.revived[4] = true;
    let mut distance = companions(2, &[1, 2, 1, 3, 0, 1, 3, 3, 0, 0]);
    distance.valid[8] = false;
    distance.valid[9] = false;
    let mut cloning = companions(1, &[3, 3, 0, 0, 0]);
    cloning.historical[2] = true;
    cloning.historical[3] = true;
    frame.distance = Some(distance);
    frame.cloning = Some(cloning);
    let all = built(&frame, ElementKind::Triplet, CompanionChoice::All);
    assert_eq!(walkers(&all), [[0, 1, 3], [0, 2, 3], [1, 3, 3]]);
    assert_eq!(weights(&all), [0.5, 0.5, 1.]);
    assert_eq!(
        all.coverage,
        Coverage {
            valid: 3,
            masked_historical: 4,
            masked_ineligible: 2,
            masked_self: 1,
            ..Coverage::default()
        }
    );
    assert_eq!(all.coverage.valid + all.coverage.masked(), 10);
    let first = built(&frame, ElementKind::Triplet, CompanionChoice::First);
    assert_eq!(walkers(&first), [[0, 1, 3], [1, 3, 3]]);
    assert_eq!(weights(&first), [1.; 2]);
    assert_eq!(
        first.coverage,
        Coverage {
            valid: 2,
            masked_historical: 2,
            masked_ineligible: 1,
            ..Coverage::default()
        }
    );
}
#[test]
fn historical_source_of_a_dead_anchor_is_historical_for_its_pair_and_for_its_triplet() {
    let mut frame = frame(4);
    frame.eligible[3] = false;
    frame.revived[3] = true;
    let mut distance = companions(1, &[1, 0, 1, 0]);
    distance.valid[3] = false;
    let mut cloning = companions(1, &[2, 2, 0, 1]);
    cloning.historical[3] = true;
    frame.distance = Some(distance);
    frame.cloning = Some(cloning);
    let expected = Coverage {
        valid: 3,
        masked_historical: 1,
        ..Coverage::default()
    };
    let pairs = built(&frame, ElementKind::CloningPair, CompanionChoice::First);
    assert_eq!(walkers(&pairs), [[0, 2, 0], [1, 2, 1], [2, 0, 2]]);
    assert_eq!(pairs.coverage, expected);
    for &choice in CompanionChoice::ALL {
        let triplets = built(&frame, ElementKind::Triplet, choice);
        assert_eq!(walkers(&triplets), [[0, 1, 2], [1, 0, 2], [2, 1, 0]]);
        assert_eq!(triplets.coverage, expected);
    }
    let distance = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    assert_eq!(distance.coverage.masked_ineligible, 1);
}
#[test]
fn triplets_take_the_first_buildable_cloning_companion_of_a_row_of_two() {
    let mut frame = frame(6);
    frame.eligible[5] = false;
    frame.distance = Some(companions(2, &[1, 2, 0, 3, 3, 0, 2, 2, 0, 1, 0, 1]));
    // Rows: self then 3; dead then 2; historical then 1; self then dead;
    // unsampled then historical; a dead anchor.
    let mut cloning = companions(2, &[0, 3, 5, 2, 0, 1, 3, 5, 0, 0, 0, 1]);
    cloning.historical[4] = true;
    cloning.valid[8] = false;
    cloning.historical[9] = true;
    frame.cloning = Some(cloning.clone());
    let all = built(&frame, ElementKind::Triplet, CompanionChoice::All);
    assert_eq!(
        walkers(&all),
        [
            [0, 1, 3],
            [0, 2, 3],
            [1, 0, 2],
            [1, 3, 2],
            [2, 3, 1],
            [2, 0, 1]
        ]
    );
    assert_eq!(weights(&all), [0.5; 6]);
    // One candidate per DISTANCE entry: n · K = 12.
    assert_eq!(
        all.coverage,
        Coverage {
            valid: 6,
            masked_historical: 2,
            masked_ineligible: 2,
            masked_self: 2,
            ..Coverage::default()
        }
    );
    let first = built(&frame, ElementKind::Triplet, CompanionChoice::First);
    assert_eq!(walkers(&first), [[0, 1, 3], [1, 0, 2], [2, 3, 1]]);
    let once = Coverage {
        valid: 3,
        masked_historical: 1,
        masked_ineligible: 1,
        masked_self: 1,
        ..Coverage::default()
    };
    assert_eq!(first.coverage, once);
    // The third walker is the companion `Companions::first` names.
    for e in all.elements.iter().chain(&first.elements) {
        let i = e.walkers[0] as usize;
        let k = cloning.first(i, &frame.eligible).unwrap();
        assert_eq!(e.walkers[2], cloning.slot[2 * i + k]);
    }
    let pairs = built(&frame, ElementKind::CloningPair, CompanionChoice::First);
    assert_eq!(walkers(&pairs), [[0, 3, 0], [1, 2, 1], [2, 1, 2]]);
    assert_eq!(pairs.coverage, once);
    let pairs = built(&frame, ElementKind::CloningPair, CompanionChoice::All);
    assert_eq!(walkers(&pairs), [[0, 3, 0], [1, 2, 1], [2, 1, 2]]);
    assert_eq!(weights(&pairs), [1.; 3]);
    assert_eq!(
        pairs.coverage,
        Coverage {
            valid: 3,
            masked_historical: 2,
            masked_ineligible: 5,
            masked_self: 2,
            ..Coverage::default()
        }
    );
}
#[test]
fn out_of_range_companion_is_refused_by_validation_and_masked_without_a_panic() {
    let mut frame = frame(3);
    frame.distance = Some(companions(1, &[1, 99, 0]));
    frame.cloning = Some(companions(1, &[2, 0, 1]));
    assert!(frame.validate().is_err());
    for kind in [ElementKind::DistancePair, ElementKind::Triplet] {
        let topology = build(&frame, kind, &MeasurementConfig::default());
        assert_eq!(topology.elements.len(), 2);
        assert!(topology.elements.iter().all(|e| e.walkers[0] != 1));
        assert_eq!(topology.coverage.masked_ineligible, 1);
        topology.validate(frame.n).unwrap();
    }
}
#[test]
fn sites_are_the_eligible_walkers_and_need_no_companion_record() {
    let mut frame = frame(3);
    frame.eligible[1] = false;
    for &choice in CompanionChoice::ALL {
        let topology = built(&frame, ElementKind::Site, choice);
        assert_eq!(walkers(&topology), [[0, 0, 0], [2, 2, 2]]);
        assert_eq!(weights(&topology), [1.; 2]);
        assert_eq!(topology.elements[1].generation, [12; 3]);
        assert_eq!(
            topology.coverage,
            Coverage {
                valid: 2,
                masked_ineligible: 1,
                ..Coverage::default()
            }
        );
        assert!(!topology.involutive);
    }
}
#[test]
fn missing_companion_record_gives_an_empty_topology_without_counts() {
    let mut frame = frame(4);
    for kind in [
        ElementKind::DistancePair,
        ElementKind::CloningPair,
        ElementKind::Triplet,
    ] {
        let topology = built(&frame, kind, CompanionChoice::All);
        assert!(topology.elements.is_empty() && !topology.involutive);
        assert_eq!(topology.coverage, Coverage::default());
    }
    frame.distance = Some(companions(1, &[1, 0, 3, 2]));
    let cloning = built(&frame, ElementKind::CloningPair, CompanionChoice::First);
    let triplet = built(&frame, ElementKind::Triplet, CompanionChoice::First);
    assert!(cloning.elements.is_empty() && triplet.elements.is_empty());
    assert_eq!(triplet.coverage, Coverage::default());
}
#[test]
fn elements_carry_source_generations_and_a_changed_walker_breaks_the_incarnation() {
    let mut frame = frame(4);
    frame.distance = Some(companions(1, &[1, 0, 3, 2]));
    let topology = built(&frame, ElementKind::DistancePair, CompanionChoice::First);
    assert_eq!(topology.elements[0].generation, [10, 11, 10]);
    assert_eq!(topology.elements[3].generation, [13, 12, 13]);
    let mut sink = frame.generation.clone();
    assert!(topology.elements.iter().all(|e| same_incarnation(e, &sink)));
    sink[1] += 1;
    let kept: Vec<bool> = topology
        .elements
        .iter()
        .map(|e| same_incarnation(e, &sink))
        .collect();
    assert_eq!(kept, [false, false, true, true]);
    assert!(!same_incarnation(&topology.elements[3], &sink[..3]));
}
#[test]
fn a_changed_generation_of_any_walker_of_a_triplet_breaks_the_incarnation() {
    let mut frame = frame(5);
    frame.distance = Some(companions(1, &[1, 0, 3, 2, 0]));
    frame.cloning = Some(companions(1, &[2, 3, 0, 1, 1]));
    let topology = built(&frame, ElementKind::Triplet, CompanionChoice::First);
    assert_eq!(
        walkers(&topology),
        [[0, 1, 2], [1, 0, 3], [2, 3, 0], [3, 2, 1], [4, 0, 1]]
    );
    assert_eq!(topology.elements[3].generation, [13, 12, 11]);
    // Walker 2 is the cloning companion of the first triplet, the anchor of
    // the third and the distance companion of the fourth.
    let mut sink = frame.generation.clone();
    sink[2] += 1;
    let kept: Vec<bool> = topology
        .elements
        .iter()
        .map(|e| same_incarnation(e, &sink))
        .collect();
    assert_eq!(kept, [false, true, false, false, true]);
    // A generation that went back is another incarnation too.
    sink[2] -= 2;
    assert!(!same_incarnation(&topology.elements[0], &sink));
    // The sink frame has no slot for the cloning companion alone.
    assert!(!same_incarnation(
        &topology.elements[0],
        &frame.generation[..2]
    ));
}
#[test]
fn mirrors_pair_the_copies_in_order_whatever_direction_comes_first() {
    let kind = ElementKind::DistancePair;
    let reversed_first = [
        pair(1, 0, kind, 1.),
        pair(1, 0, kind, 1.),
        pair(0, 1, kind, 1.),
    ];
    assert_eq!(mirrors(&reversed_first), [Some(2), None, Some(0)]);
    assert!(is_involutive(&reversed_first[1..]));
    let twice = [
        pair(3, 1, kind, 0.5),
        pair(1, 3, kind, 0.5),
        pair(2, 0, kind, 1.),
        pair(1, 3, kind, 0.5),
        pair(3, 1, kind, 0.5),
        pair(0, 2, kind, 1.),
    ];
    assert_eq!(
        mirrors(&twice),
        [Some(1), Some(0), Some(5), Some(4), Some(3), Some(2)]
    );
    assert!(is_involutive(&twice));
    // Many copies against two mirrors: the first two copies are the mirrored ones.
    let mut many: Vec<Element> = (0..64)
        .flat_map(|e| [pair(0, 1, kind, 1.), pair(2 + e, 1, kind, 1.)])
        .collect();
    many.extend([pair(1, 0, kind, 1.); 2]);
    let mirror = mirrors(&many);
    assert_eq!((mirror[0], mirror[2]), (Some(128), Some(129)));
    assert_eq!((mirror[128], mirror[129]), (Some(0), Some(2)));
    assert_eq!(mirror.iter().flatten().count(), 4);
}
#[test]
fn mirror_weights_are_compared_bit_for_bit() {
    let kind = ElementKind::CloningPair;
    let (sum, third): (f64, f64) = (0.1 + 0.2, 1. / 3.);
    assert!(sum != 0.3 && sum as f32 == 0.3_f32);
    let close = [pair(0, 1, kind, 0.3), pair(1, 0, kind, sum)];
    assert_eq!(mirrors(&close), [None, None]);
    assert!(!is_involutive(&close));
    let next = f64::from_bits(third.to_bits() + 1);
    assert_eq!(
        mirrors(&[pair(0, 1, kind, third), pair(1, 0, kind, next)]),
        [None, None]
    );
    let equal = [pair(0, 1, kind, third), pair(1, 0, kind, 1. / 3.)];
    assert_eq!(mirrors(&equal), [Some(1), Some(0)]);
}
#[test]
fn mirrors_match_one_to_one_within_a_kind_and_a_weight() {
    let (distance, cloning) = (ElementKind::DistancePair, ElementKind::CloningPair);
    assert!(!is_involutive(&[]));
    assert!(!is_involutive(&[pair(1, 1, distance, 1.)]));
    let repeated = [
        pair(0, 1, distance, 1.),
        pair(0, 1, distance, 1.),
        pair(1, 0, distance, 1.),
    ];
    assert_eq!(mirrors(&repeated), [Some(2), None, Some(0)]);
    assert!(!is_involutive(&repeated));
    assert!(is_involutive(&repeated[1..]));
    let kinds = [pair(0, 1, distance, 1.), pair(1, 0, cloning, 1.)];
    assert_eq!(mirrors(&kinds), [None, None]);
    let site = Element {
        walkers: [0; 3],
        kind: ElementKind::Site,
        weight: 1.,
        generation: [0; 3],
    };
    let triplet = Element {
        walkers: [0, 1, 2],
        kind: ElementKind::Triplet,
        weight: 1.,
        generation: [0; 3],
    };
    assert_eq!(mirrors(&[site, triplet]), [None, None]);
    assert!(!is_involutive(&[site]) && !is_involutive(&[triplet]));
}
