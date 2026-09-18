use algorithmic_gas::{
    cloning::{
        CloneChoice, CloneDecision, ClonePlan, CloneTransform, WeightedDonor,
        accepted_current_components, apply_component_rotations,
    },
    donor::{CompanionBatch, DonorPool},
    noise::{Noise, NoiseRequest, NoiseSource},
    random::{RandomStream, Stream},
    *,
};
use futures_lite::future::block_on;

fn population(d: usize) -> Population<f64> {
    let mut observations = ObservationBatch::positions(
        TensorBatch::vectors(5, d, (0..5 * d).map(|j| j as f64 * 0.17).collect()).unwrap(),
    );
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(
            5,
            d,
            (0..5 * d).map(|j| (j as f64 + 0.3).sin() * 3.).collect(),
        )
        .unwrap(),
    );
    Population::new(observations).unwrap()
}
fn plan(p: &Population<f64>, pool: &DonorPool<f64>, edges: &[(usize, usize)]) -> ClonePlan {
    let mut choices = vec![
        CloneChoice {
            donors: vec![],
            accepted: false,
            revival: false,
            probability: Some(0.)
        };
        p.len()
    ];
    for &(i, j) in edges {
        let index = pool
            .sources
            .iter()
            .position(|s| s.frame == pool.current_frame && s.slot as usize == j)
            .unwrap();
        choices[i] = CloneChoice {
            donors: vec![WeightedDonor {
                pool_index: index as u32,
                weight: 1.,
            }],
            accepted: true,
            revival: !p.validity[i].eligible(false),
            probability: Some(1.),
        };
    }
    ClonePlan {
        population_version: p.version,
        sources: pool.sources.clone(),
        choices,
        mutual: false,
    }
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 2e-11 * (1. + a.abs() + b.abs()),
        "{a} != {b}"
    );
}

#[test]
fn overlapping_edges_use_frozen_velocities_including_revived_dead_slots() {
    block_on(async {
        for d in [1, 2, 3] {
            for alpha in [0., 0.5, 1.] {
                let mut before = population(d);
                before.validity[0].out_of_bounds = true;
                let pool = DonorPool::freeze(&before, 0, &[], 0, false).unwrap();
                let plan = plan(&before, &pool, &[(0, 1), (1, 2), (3, 4)]);
                let components = accepted_current_components(&before, &pool, &plan).unwrap();
                assert_eq!(components, vec![vec![0, 1, 2], vec![3, 4]]);
                let mut after = plan.apply_literal(&before, &pool).unwrap();
                let literal = after.clone();
                let rotations = components
                    .iter()
                    .map(|g| {
                        RandomStream::new(71, 1, Stream::CollisionRotation, g[0] as u64, 0)
                            .haar_orthogonal(d)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let reports = apply_component_rotations(
                    &before,
                    &mut after,
                    &components,
                    &rotations,
                    "velocities",
                    alpha,
                )
                .unwrap();
                for r in reports {
                    for (&a, &b) in r.momentum_before.iter().zip(&r.momentum_after) {
                        close(a, b);
                    }
                    close(
                        r.relative_energy_after,
                        alpha * alpha * r.relative_energy_before,
                    );
                    for &i in &r.members {
                        for a in 0..d {
                            let original = before.observations.field("velocities").unwrap();
                            let expected = r.center_of_mass[a]
                                + alpha
                                    * (0..d)
                                        .map(|b| {
                                            r.rotation[a * d + b]
                                                * (original.row(i).unwrap()[b]
                                                    - r.center_of_mass[b])
                                        })
                                        .sum::<f64>();
                            close(
                                after
                                    .observations
                                    .field("velocities")
                                    .unwrap()
                                    .row(i)
                                    .unwrap()[a],
                                expected,
                            );
                        }
                    }
                }
                let transform = CloneTransform {
                    position_field: Some("positions".into()),
                    jitter: Some(Noise::default()),
                    jitter_amplitude: 0.2,
                    velocity_field: Some("velocities".into()),
                    restitution: Some(alpha),
                    ..Default::default()
                };
                let mut actual = literal.clone();
                let companions = CompanionBatch {
                    rows: 5,
                    count: 1,
                    indices: vec![0; 5],
                    valid: vec![false; 5],
                    mutual: false,
                };
                let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
                    .await
                    .unwrap();
                let eta = Noise::default()
                    .sample(
                        &literal.observations,
                        NoiseRequest {
                            rows: 5,
                            dimension: d,
                            seed: 71,
                            step: 1,
                            stream: Stream::CloneNoise,
                            substep: 0,
                        },
                        &mut cx,
                    )
                    .await
                    .unwrap();
                transform
                    .apply(
                        &before,
                        &mut actual,
                        &pool,
                        &plan,
                        &companions,
                        71,
                        1,
                        &mut cx,
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    actual.observations.field("velocities").unwrap(),
                    after.observations.field("velocities").unwrap()
                );
                for i in 0..5 {
                    for a in 0..d {
                        close(
                            actual
                                .observations
                                .field("positions")
                                .unwrap()
                                .row(i)
                                .unwrap()[a],
                            literal
                                .observations
                                .field("positions")
                                .unwrap()
                                .row(i)
                                .unwrap()[a]
                                + if plan.choices[i].accepted {
                                    0.2 * eta.row(i).unwrap()[a]
                                } else {
                                    0.
                                },
                        );
                    }
                }
                assert!(actual.validity[0].eligible(false));
                assert_ne!(
                    actual
                        .observations
                        .field("positions")
                        .unwrap()
                        .row(0)
                        .unwrap(),
                    literal
                        .observations
                        .field("positions")
                        .unwrap()
                        .row(0)
                        .unwrap()
                );
            }
        }
    });
}

#[test]
fn deterministic_map_commutes_with_permutation_when_rotations_are_transported() {
    for d in [1, 2, 3] {
        for alpha in [0., 0.5, 1.] {
            let before = population(d);
            let permutation = [3, 0, 4, 2, 1]; // new row -> old row
            let inverse = (0..5)
                .map(|old| permutation.iter().position(|&i| i == old).unwrap())
                .collect::<Vec<_>>();
            let components = vec![vec![0, 1, 2], vec![3, 4]];
            let rotations = vec![
                RandomStream::new(11, 0, Stream::CollisionRotation, 0, 0)
                    .haar_orthogonal(d)
                    .unwrap(),
                RandomStream::new(11, 0, Stream::CollisionRotation, 3, 0)
                    .haar_orthogonal(d)
                    .unwrap(),
            ];
            let mut result = before.clone();
            apply_component_rotations(
                &before,
                &mut result,
                &components,
                &rotations,
                "velocities",
                alpha,
            )
            .unwrap();
            let mut permuted = before.clone();
            for field in permuted.observations.fields.values_mut() {
                *field = field.gather(&permutation.map(|i| i as u32)).unwrap();
            }
            let mut permuted_result = permuted.clone();
            let transported = components
                .iter()
                .map(|g| g.iter().map(|&i| inverse[i]).collect())
                .collect::<Vec<Vec<usize>>>();
            apply_component_rotations(
                &permuted,
                &mut permuted_result,
                &transported,
                &rotations,
                "velocities",
                alpha,
            )
            .unwrap();
            for (i, &old) in permutation.iter().enumerate() {
                for a in 0..d {
                    close(
                        permuted_result
                            .observations
                            .field("velocities")
                            .unwrap()
                            .row(i)
                            .unwrap()[a],
                        result
                            .observations
                            .field("velocities")
                            .unwrap()
                            .row(old)
                            .unwrap()[a],
                    );
                }
            }
        }
    }
}

#[test]
fn historical_accepted_edges_reject_and_unaccepted_candidates_do_not_collide() {
    let before = population(2);
    let pool = DonorPool::freeze(&before, 1, &[(0, before.clone())], 1, false).unwrap();
    let mut plan = plan(&before, &pool, &[(0, 1)]);
    let historical = pool.sources.iter().position(|s| s.frame == 0).unwrap() as u32;
    plan.choices[0].donors[0].pool_index = historical;
    assert!(
        accepted_current_components(&before, &pool, &plan)
            .unwrap_err()
            .to_string()
            .contains("historical")
    );
    plan.choices[0].accepted = false;
    assert!(
        accepted_current_components(&before, &pool, &plan)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn revival_uses_the_sampled_current_companion() {
    let mut before = population(2);
    before.validity[0].out_of_bounds = true;
    let pool = DonorPool::freeze(&before, 0, &[], 0, false).unwrap();
    let selected = pool.sources.iter().position(|s| s.slot == 3).unwrap() as u32;
    let companions = CompanionBatch {
        rows: 5,
        count: 1,
        indices: vec![selected; 5],
        valid: vec![true; 5],
        mutual: false,
    };
    let decision = CloneDecision {
        revival_from_companion: true,
        ..Default::default()
    };
    let p = decision
        .plan(
            &before,
            &pool,
            &companions,
            &[1.; 5],
            &[1.; 4],
            &before.eligible(false),
            9,
            1,
        )
        .unwrap();
    assert!(p.choices[0].revival && p.choices[0].accepted);
    assert_eq!(p.choices[0].probability, Some(1.));
    assert_eq!(p.choices[0].donors[0].pool_index, selected);
    let mut config = GasConfig {
        precision: Precision::F64,
        ..Default::default()
    };
    config.clone_decision = decision;
    config.cloning_donors.history_window = 1;
    assert!(config.validate(&before, false).is_err());
    config.cloning_donors.history_window = 0;
    config.cloning_donors.law = algorithmic_gas::donor::SamplingLaw::FisherYates;
    assert!(config.validate(&before, false).is_err());
}

#[test]
fn haar_orthogonal_has_both_parities_and_isotropic_second_moments() {
    for d in [1, 2, 3] {
        let mut means = vec![0.; d * d];
        let mut squares = means.clone();
        let mut positive = 0;
        for seed in 0..1024 {
            let q = RandomStream::new(seed, 3, Stream::CollisionRotation, 0, 0)
                .haar_orthogonal(d)
                .unwrap();
            for a in 0..d {
                for b in 0..d {
                    close(
                        (0..d).map(|k| q[k * d + a] * q[k * d + b]).sum(),
                        if a == b { 1. } else { 0. },
                    );
                }
            }
            let det = match d {
                1 => q[0],
                2 => q[0] * q[3] - q[1] * q[2],
                _ => {
                    q[0] * (q[4] * q[8] - q[5] * q[7]) - q[1] * (q[3] * q[8] - q[5] * q[6])
                        + q[2] * (q[3] * q[7] - q[4] * q[6])
                }
            };
            positive += usize::from(det > 0.);
            for k in 0..d * d {
                means[k] += q[k] / 1024.;
                squares[k] += q[k] * q[k] / 1024.;
            }
        }
        assert!((430..595).contains(&positive));
        for k in 0..d * d {
            assert!(means[k].abs() < 0.12);
            assert!((squares[k] - 1. / d as f64).abs() < 0.08);
        }
    }
}

struct ConstantReward;
impl algorithmic_gas::domain::RewardSource<f64> for ConstantReward {
    fn id(&self) -> String {
        "collision-test-constant".into()
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<f64>,
        _: Option<&'a InputBatch<f64>>,
        stage: &'a str,
        _: &'a mut ExecutionContext,
    ) -> algorithmic_gas::domain::OperatorFuture<'a, RewardBatch<f64>> {
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
#[test]
fn actual_engine_records_revived_component_and_empty_graph_and_validates_archive() {
    block_on(async {
        for dead in [false, true] {
            let mut p = population(3);
            p.validity[0].terminated = dead;
            let mut config = GasConfig {
                precision: Precision::F64,
                ..Default::default()
            };
            config.clone_decision.revival_from_companion = true;
            config.clone_transform = CloneTransform {
                position_field: Some("positions".into()),
                jitter: Some(Noise::default()),
                jitter_amplitude: 0.1,
                velocity_field: Some("velocities".into()),
                restitution: Some(0.5),
                ..Default::default()
            };
            config.fitness.diversity_exponent = 0.;
            let mut gas = GasBuilder::new(p, ConstantReward)
                .config(config)
                .build()
                .await
                .unwrap();
            gas.start_recording(Default::default()).unwrap();
            gas.step().await.unwrap();
            let archive = gas.recording().unwrap();
            let step = &archive.steps[0];
            let fields = &step.field_evaluations;
            let ids = fields
                .iter()
                .find(|f| f.stage == "component_collision" && f.field == "collision_component_id")
                .unwrap();
            assert_eq!(
                ids.available.iter().filter(|&&v| v).count(),
                if dead { 2 } else { 0 }
            );
            if dead {
                assert!(ids.available[0]);
                assert!(step.report.clone_plan.choices[0].revival);
                assert_eq!(
                    step.report.clone_plan.choices[0].donors[0].pool_index,
                    step.report.cloning_companions.row(0).next().unwrap()
                );
                let r = fields
                    .iter()
                    .find(|f| f.stage == "component_collision" && f.field == "collision_rotation")
                    .unwrap();
                let peer = (1..5).find(|&i| ids.available[i]).unwrap();
                assert_eq!(&r.values[..9], &r.values[peer * 9..(peer + 1) * 9]);
                let a = fields
                    .iter()
                    .find(|f| {
                        f.stage == "component_collision" && f.field == "collision_momentum_before"
                    })
                    .unwrap();
                let b = fields
                    .iter()
                    .find(|f| {
                        f.stage == "component_collision" && f.field == "collision_momentum_after"
                    })
                    .unwrap();
                for i in 0..3 {
                    close(a.values[i], b.values[i]);
                }
            }
            let json = serde_json::to_string(archive).unwrap();
            algorithmic_gas::tracking::RunArchive::<f64>::from_json(&json).unwrap();
            Checkpoint::<f64>::from_bytes(&gas.checkpoint().to_bytes().unwrap()).unwrap();
        }
    });
}

#[test]
fn enumerate_all_five_slot_strict_fitness_forests_against_independent_bfs() {
    // A rejected gate is None; an accepted gate points to a strictly higher rank.
    // This exhausts 5*4*3*2*1=120 admissible edge configurations.
    for code in 0..120 {
        let mut digits = code;
        let mut edges = vec![];
        for i in 0..5 {
            let choice = digits % (5 - i);
            digits /= 5 - i;
            if choice > 0 {
                edges.push((i, i + choice));
            }
        }
        let before = population(3);
        let pool = DonorPool::freeze(&before, 0, &[], 0, false).unwrap();
        let plan = plan(&before, &pool, &edges);
        let mut seen = [false; 5];
        let mut expected = vec![];
        for root in 0..5 {
            if seen[root] {
                continue;
            }
            let mut component = vec![root];
            seen[root] = true;
            let mut cursor = 0;
            while cursor < component.len() {
                let i = component[cursor];
                cursor += 1;
                for &(u, v) in &edges {
                    let neighbor = if u == i {
                        Some(v)
                    } else if v == i {
                        Some(u)
                    } else {
                        None
                    };
                    if let Some(j) = neighbor
                        && !seen[j]
                    {
                        seen[j] = true;
                        component.push(j);
                    }
                }
            }
            component.sort_unstable();
            if component.len() > 1 {
                expected.push(component);
            }
        }
        let actual = accepted_current_components(&before, &pool, &plan).unwrap();
        assert_eq!(actual, expected, "edge configuration {edges:?}");
        for alpha in [0., 0.3, 1.] {
            let mut after = plan.apply_literal(&before, &pool).unwrap();
            let rotations = actual
                .iter()
                .map(|_| vec![0., 1., 0., -1., 0., 0., 0., 0., -1.])
                .collect::<Vec<_>>();
            for report in apply_component_rotations(
                &before,
                &mut after,
                &actual,
                &rotations,
                "velocities",
                alpha,
            )
            .unwrap()
            {
                for (a, b) in report.momentum_before.iter().zip(&report.momentum_after) {
                    close(*a, *b);
                }
                close(
                    report.relative_energy_after,
                    alpha * alpha * report.relative_energy_before,
                );
            }
            for &(i, j) in &edges {
                assert_eq!(
                    after
                        .observations
                        .field("positions")
                        .unwrap()
                        .row(i)
                        .unwrap(),
                    before
                        .observations
                        .field("positions")
                        .unwrap()
                        .row(j)
                        .unwrap()
                );
            }
        }
    }
}
