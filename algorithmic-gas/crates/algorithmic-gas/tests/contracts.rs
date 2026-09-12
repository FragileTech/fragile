use algorithmic_gas::{
    boundary::{BoundaryPolicy, BoxDomain},
    cloning::{CloneChoice, ClonePlan, WeightedDonor},
    compute::{Binary, Expression},
    donor::{
        CompanionReducer, CompanionRequest, CompanionSampler, DonorModule, DonorPool, OddPolicy,
        SamplingLaw,
    },
    fitness::{FitnessPipeline, PositiveMap, PositiveMapping, Standardizer},
    geometry::{AlgorithmicDistance, Distance, InteractionKernel, Kernel},
    noise::{FactorValues, InnovationLaw, Noise, NoiseGeometry, NoiseRequest, NoiseSource},
    random::{RandomStream, Stream},
    *,
};
use futures_lite::future::block_on;

fn population<T: Real>(values: Vec<T>, d: usize) -> Population<T> {
    Population::new(ObservationBatch::positions(
        TensorBatch::vectors(values.len() / d, d, values).unwrap(),
    ))
    .unwrap()
}
fn cpu<T: Real>() -> ExecutionContext {
    block_on(ExecutionContext::new(BackendKind::Cpu, T::PRECISION)).unwrap()
}

#[test]
fn batch_shapes_and_integer_gathers() {
    assert!(TensorBatch::<f32>::new(2, vec![3, 2], vec![0.; 11]).is_err());
    let t = TensorBatch::new(2, vec![1, 1, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    assert_eq!(
        t.gather(&[1, 0, 1]).unwrap().values(),
        &[4., 5., 6., 1., 2., 3., 4., 5., 6.]
    );
    assert_eq!(t.item_shape(), &[1, 1, 3]);
    assert!(t.gather(&[2]).is_err());
}
#[test]
fn periodic_bounds_must_remain_ordered_in_run_precision() {
    let domain = BoxDomain {
        lower: vec![1.],
        upper: vec![1. + 1e-9],
    };
    assert!(domain.validate_precision::<f32>(1).is_err());
    assert!(domain.validate_precision::<f64>(1).is_ok());
    let policy = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain,
    };
    assert!(policy.apply(&mut population(vec![2_f32], 1)).is_err());
    let domain = BoxDomain {
        lower: vec![-1.],
        upper: vec![2.],
    };
    let policy = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain,
    };
    let mut p = population(vec![f32::MAX], 1);
    // Overflow must fail or return a finite, validly wrapped coordinate.
    if policy.apply(&mut p).is_ok() {
        let x = p.observations.field("positions").unwrap().values()[0];
        assert!(x.is_finite() && (-1. ..2.).contains(&x));
    }
}
#[test]
fn f64_is_not_an_f32_roundtrip() {
    block_on(async {
        let mut cx = cpu::<f64>();
        let mut e = Expression::default();
        let x = e.input(0);
        let one = e.scalar(1.);
        e.binary(Binary::Add, x, one);
        let output = cx
            .evaluate(
                &e,
                &[TensorBatch::vectors(1, 1, vec![16_777_217_f64]).unwrap()],
            )
            .await
            .unwrap();
        assert_eq!(output.values(), &[16_777_218_f64]);
        assert!(
            cx.evaluate(&e, &[TensorBatch::vectors(1, 1, vec![1_f32]).unwrap()])
                .await
                .is_err()
        );
    });
}
#[test]
fn malformed_graph_is_a_result_not_a_panic() {
    block_on(async {
        let mut cx = cpu::<f32>();
        let e = Expression {
            nodes: vec![algorithmic_gas::compute::Node::Unary(
                algorithmic_gas::compute::Unary::Square,
                2,
            )],
        };
        assert!(cx.evaluate::<f32>(&e, &[]).await.is_err());
    });
}
#[test]
fn unsupported_precision_is_explicit() {
    assert!(block_on(ExecutionContext::new(BackendKind::Wgpu, Precision::F64)).is_err());
}
#[test]
fn addressed_rng_has_independent_streams_and_order() {
    let draw = |i, stream| RandomStream::new(7, 3, stream, i, 0).next_u64();
    let a = (0..10)
        .map(|i| draw(i, Stream::Distance))
        .collect::<Vec<_>>();
    let mut b = (0..10)
        .rev()
        .map(|i| draw(i, Stream::Distance))
        .collect::<Vec<_>>();
    b.reverse();
    assert_eq!(a, b);
    assert_ne!(draw(1, Stream::Distance), draw(1, Stream::Cloning));
    let mut rng = RandomStream::new(0, 0, Stream::Initialize, 0, 0);
    for _ in 0..10_000 {
        let x = rng.uniform::<f32>();
        assert!(x > 0. && x < 1.);
    }
}
#[test]
fn cosine_zero_policy_and_explicit_kernel() {
    let p = population(vec![0_f64, 0., 1., 0., -1., 0.], 2);
    let d = Distance::Cosine {
        field: "positions".into(),
        zero_tolerance: 0.,
    };
    assert_eq!(
        d.compare(&p.observations, 0, &p.observations, 0).unwrap(),
        0.
    );
    assert_eq!(
        d.compare(&p.observations, 0, &p.observations, 1).unwrap(),
        1.
    );
    assert_eq!(
        d.compare(&p.observations, 1, &p.observations, 2).unwrap(),
        2.
    );
    assert!(
        Kernel::Gaussian { width: 1. }
            .log_weight(
                1_f64,
                <Distance as AlgorithmicDistance<f64>>::comparison_kind(&d)
            )
            .is_err()
    );
}
#[test]
fn periodic_distance_uses_minimum_image() {
    let p = population(vec![-0.9_f64, 0.9], 1);
    let domain = BoxDomain {
        lower: vec![-1.],
        upper: vec![1.],
    };
    let distance = Distance::Euclidean {
        field: "positions".into(),
        scales: vec![],
        squared: false,
        periodic: Some(domain),
    };
    assert!(
        (distance
            .compare(&p.observations, 0, &p.observations, 1)
            .unwrap()
            - 0.2)
            .abs()
            < 1e-12
    );
}
#[test]
fn distances_and_squared_kernel_convention() {
    block_on(async {
        let p = population(vec![0_f64, 0., 3., 4.], 2);
        let d = Distance::default();
        let mut cx = cpu::<f64>();
        assert_eq!(
            d.pairs(&p.observations, &p.observations, &[(0, 1)], &mut cx)
                .await
                .unwrap(),
            vec![5.]
        );
        let squared = Distance::Euclidean {
            field: "positions".into(),
            scales: vec![],
            squared: true,
            periodic: None,
        };
        assert_eq!(
            squared
                .compare(&p.observations, 0, &p.observations, 1)
                .unwrap(),
            25.
        );
        let kernel = Kernel::Gaussian { width: 2. };
        assert_eq!(
            kernel
                .log_weight(5_f64, algorithmic_gas::geometry::ComparisonKind::Distance)
                .unwrap(),
            kernel
                .log_weight(
                    25_f64,
                    algorithmic_gas::geometry::ComparisonKind::SquaredDistance
                )
                .unwrap()
        );
    });
}
#[test]
fn phase_space_requires_velocity_and_scaling() {
    let mut p = population(vec![0_f32, 3.], 1);
    let d = Distance::PhaseSpace {
        positions: "positions".into(),
        velocities: "velocities".into(),
        position_scale: 1.,
        velocity_scale: 1.,
        lambda: 4.,
        periodic: None,
    };
    assert!(d.validate(&p.observations).is_err());
    p.observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(2, 1, vec![0., 2.]).unwrap(),
    );
    assert_eq!(
        d.compare(&p.observations, 0, &p.observations, 1).unwrap(),
        5.
    );
}
#[test]
fn boundary_signals_remain_distinct() {
    let mut p = population(vec![-2_f64, 0., 2., f64::NAN], 1);
    p.validity[1].truncated = true;
    BoundaryPolicy::AbsorbingBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![-1.],
            upper: vec![1.],
        },
    }
    .apply(&mut p)
    .unwrap();
    assert!(p.validity[0].out_of_bounds);
    assert!(!p.validity[0].terminated);
    assert!(p.validity[3].invalid);
    assert!(!p.validity[1].eligible(false));
    assert!(p.validity[1].eligible(true));
}
#[test]
fn periodic_repair_reports_fields() {
    let mut p = population(vec![-3.5_f64, 1.], 1);
    let changed = BoundaryPolicy::PeriodicBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![-1.],
            upper: vec![1.],
        },
    }
    .apply(&mut p)
    .unwrap();
    assert_eq!(changed, vec!["positions"]);
    assert_eq!(
        p.observations.field("positions").unwrap().values(),
        &[0.5, -1.]
    );
}
#[test]
fn fitness_alive_only_regularization_and_degeneracy() {
    let p = population(vec![0_f64, 1., 2.], 1);
    let s = Standardizer::Global { sigma_min: 0.1 };
    let (z, stats) = s
        .apply(&[5., 5., 1e99], &[true, true, false], &p.observations)
        .unwrap();
    assert_eq!(z, vec![0., 0., 0.]);
    assert_eq!(stats.mean, vec![5.; 3]);
    assert_eq!(stats.scale, vec![0.1; 3]);
    assert!(s.apply(&[0.; 3], &[false; 3], &p.observations).is_err());
}
#[test]
fn stable_positive_maps_and_oriented_rewards() {
    let map = PositiveMap::Logistic {
        amplitude: 2.,
        floor: 1e-6,
    };
    assert!((map.map(0_f64).unwrap() - 1.000001).abs() < 1e-12);
    assert!(map.map(-1000_f32).unwrap() > 0.);
    let legacy = PositiveMap::LegacyAsymmetric { floor: 0. };
    assert_eq!(legacy.map(0_f64).unwrap(), 1.);
    assert!((legacy.map(1_f64).unwrap() - (1. + 2_f64.ln())).abs() < 1e-12);
    let p = population(vec![0_f64, 1.], 1);
    let rewards = RewardBatch::new(vec![1., 2.], Provenance::default());
    let f = FitnessPipeline::default()
        .evaluate(&rewards, &[1., 1.], &[true, true], &p.observations, 0)
        .unwrap();
    assert!(f.fitness[0] > f.fitness[1]);
    assert_eq!(rewards.raw, vec![1., 2.]);
}
#[test]
fn local_uniform_statistics_equal_global() {
    let p = population(vec![0_f64, 1., 2.], 1);
    let v = [1., 4., 7.];
    let alive = [true; 3];
    let global = Standardizer::Global { sigma_min: 0.2 }
        .apply(&v, &alive, &p.observations)
        .unwrap()
        .0;
    let local = Standardizer::Local {
        include_self: true,
        sigma_min: 0.2,
        distance: Distance::default(),
        kernel: Kernel::Uniform,
    }
    .apply(&v, &alive, &p.observations)
    .unwrap()
    .0;
    for (a, b) in global.iter().zip(local) {
        assert!((a - b).abs() < 1e-12);
    }
}
#[test]
fn fisher_yates_is_mutual_uniform_and_not_a_permutation_law() {
    block_on(async {
        let p = population(vec![0_f64, 1., 2., 3.], 1);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let module = DonorModule {
            law: SamplingLaw::FisherYates,
            kernel: Kernel::Uniform,
            ..Default::default()
        };
        let mut cx = cpu::<f64>();
        let mut counts = [0; 4];
        for seed in 0..6000 {
            let c = module
                .sample(
                    CompanionRequest {
                        population: &p,
                        pool: &pool,
                        eligible: &[true; 4],
                        seed,
                        step: 1,
                        stream: Stream::Distance,
                    },
                    &mut cx,
                )
                .await
                .unwrap();
            for i in 0..4 {
                let j = c.indices[i] as usize;
                assert_ne!(i, j);
                assert_eq!(c.indices[j], i as u32);
            }
            counts[c.indices[0] as usize] += 1;
        }
        assert_eq!(counts[0], 0);
        for count in &counts[1..] {
            assert!((*count as f64 / 6000. - 1. / 3.).abs() < 0.025);
        }
    });
}
#[test]
fn odd_matching_and_singleton_policies() {
    block_on(async {
        let p = population(vec![0_f32, 1., 2.], 1);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let mut module = DonorModule {
            law: SamplingLaw::FisherYates,
            kernel: Kernel::Uniform,
            odd: OddPolicy::Unmatched,
            ..Default::default()
        };
        let mut cx = cpu::<f32>();
        let c = module
            .sample(
                CompanionRequest {
                    population: &p,
                    pool: &pool,
                    eligible: &[true; 3],
                    seed: 1,
                    step: 1,
                    stream: Stream::Distance,
                },
                &mut cx,
            )
            .await
            .unwrap();
        assert_eq!(c.valid.iter().filter(|&&v| v).count(), 2);
        module.odd = OddPolicy::Reject;
        assert!(
            module
                .sample(
                    CompanionRequest {
                        population: &p,
                        pool: &pool,
                        eligible: &[true; 3],
                        seed: 1,
                        step: 1,
                        stream: Stream::Distance
                    },
                    &mut cx
                )
                .await
                .is_err()
        );
        let singleton = population(vec![0_f32], 1);
        let pool = DonorPool::freeze(&singleton, 0, &[], 0, false).unwrap();
        let c = module
            .sample(
                CompanionRequest {
                    population: &singleton,
                    pool: &pool,
                    eligible: &[true],
                    seed: 1,
                    step: 1,
                    stream: Stream::Distance,
                },
                &mut cx,
            )
            .await
            .unwrap();
        assert_eq!(c.indices, vec![0]);
        assert!(c.valid[0]);
    });
}
#[test]
fn multi_companion_reduction_and_no_replacement() {
    block_on(async {
        let p = population(vec![0_f64, 1., 3.], 1);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let module = DonorModule {
            kernel: Kernel::Uniform,
            count: 2,
            replacement: false,
            ..Default::default()
        };
        let mut cx = cpu::<f64>();
        let c = module
            .sample(
                CompanionRequest {
                    population: &p,
                    pool: &pool,
                    eligible: &[true; 3],
                    seed: 7,
                    step: 1,
                    stream: Stream::Distance,
                },
                &mut cx,
            )
            .await
            .unwrap();
        for i in 0..3 {
            let row = c.row(i).collect::<Vec<_>>();
            assert_ne!(row[0], row[1]);
        }
        let d = CompanionReducer::Mean
            .measure(&Distance::default(), &p, &pool, &c, &mut cx)
            .await
            .unwrap();
        assert_eq!(d, vec![2., 1.5, 2.5]);
    });
}
#[test]
fn gaussian_independent_distribution_and_tile_invariance() {
    block_on(async {
        let recipients = population(vec![0_f64; 4000], 1);
        let donors = population(vec![1_f64, 2.], 1);
        let pool = DonorPool::freeze(&donors, 0, &[], 0, false).unwrap();
        let mut module = DonorModule {
            allow_self: true,
            tile_edges: 73,
            ..Default::default()
        };
        let alive = vec![true; 4000];
        let mut cx = cpu::<f64>();
        let a = module
            .sample(
                CompanionRequest {
                    population: &recipients,
                    pool: &pool,
                    eligible: &alive,
                    seed: 9,
                    step: 1,
                    stream: Stream::Distance,
                },
                &mut cx,
            )
            .await
            .unwrap();
        module.tile_edges = 2048;
        let b = module
            .sample(
                CompanionRequest {
                    population: &recipients,
                    pool: &pool,
                    eligible: &alive,
                    seed: 9,
                    step: 1,
                    stream: Stream::Distance,
                },
                &mut cx,
            )
            .await
            .unwrap();
        assert_eq!(a, b);
        let probability = a.indices.iter().filter(|&&j| j == 0).count() as f64 / 4000.;
        let expected = 1. / (1. + (-1.5_f64).exp());
        assert!((probability - expected).abs() < 0.025);
    });
}
#[test]
fn simultaneous_cloning_copies_observations_rewards_and_state() {
    let mut p = population(vec![10_f64, 20., 30., 40.], 1);
    p.rewards.raw = vec![1., 2., 3., 4.];
    p.states = Some(StateStore {
        snapshots: vec![vec![1], vec![2], vec![3], vec![4]],
        codec: "mock/v1".into(),
    });
    let pool = DonorPool::freeze(&p, 5, &[], 0, false).unwrap();
    let plan = ClonePlan {
        population_version: p.version,
        sources: pool.sources.clone(),
        choices: [1, 2, 0, 2]
            .iter()
            .map(|&j| CloneChoice {
                donors: vec![WeightedDonor {
                    pool_index: j,
                    weight: 1.,
                }],
                accepted: true,
                revival: false,
                probability: None,
            })
            .collect(),
        mutual: false,
    };
    let result = plan.apply_literal(&p, &pool).unwrap();
    assert_eq!(
        result.observations.field("positions").unwrap().values(),
        &[20., 30., 10., 30.]
    );
    assert_eq!(result.rewards.raw, vec![2., 3., 1., 3.]);
    assert_eq!(
        result.states.unwrap().snapshots,
        vec![vec![2], vec![3], vec![1], vec![3]]
    );
    assert_eq!(
        p.observations.field("positions").unwrap().values(),
        &[10., 20., 30., 40.]
    );
}
#[test]
fn historical_pool_has_no_ancestry_filter() {
    let mut old = population(vec![1_f32, 2.], 1);
    old.generations = vec![89, 90];
    let mut current = population(vec![3_f32, 4.], 1);
    current.generations = vec![0, 0];
    let pool = DonorPool::freeze(&current, 2, &[(1, old)], 1, false).unwrap();
    assert_eq!(pool.sources.len(), 4);
    assert_eq!(pool.sources[2].generation, 89);
    assert_eq!(pool.sources[2].frame, 1);
}
fn covariance(noise: Noise) -> [[f64; 2]; 2] {
    block_on(async {
        let n = 16000;
        let p = population(vec![0_f64; n * 2], 2);
        let mut cx = cpu::<f64>();
        let values = noise
            .sample(
                &p.observations,
                NoiseRequest {
                    rows: n,
                    dimension: 2,
                    seed: 3,
                    step: 4,
                    stream: Stream::Kinetic,
                    substep: 0,
                },
                &mut cx,
            )
            .await
            .unwrap();
        let mut sum = [0.; 2];
        let mut sq = [[0.; 2]; 2];
        for r in values.values().chunks(2) {
            for i in 0..2 {
                sum[i] += r[i];
                for j in 0..2 {
                    sq[i][j] += r[i] * r[j];
                }
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                sq[i][j] = sq[i][j] / n as f64 - sum[i] * sum[j] / (n * n) as f64;
            }
        }
        sq
    })
}
#[test]
fn isotropic_noise_covariance() {
    let c = covariance(Noise::default());
    assert!((c[0][0] - 1.).abs() < 0.035);
    assert!((c[1][1] - 1.).abs() < 0.035);
    assert!(c[0][1].abs() < 0.035);
}
#[test]
fn anisotropic_full_factor_covariance() {
    let c = covariance(Noise {
        innovation: InnovationLaw::Gaussian,
        geometry: NoiseGeometry::Full {
            factor: FactorValues::Constant {
                values: vec![0.2, 0., 0.1, 0.3],
            },
        },
    });
    assert!((c[0][0] - 0.04).abs() < 0.002);
    assert!((c[0][1] - 0.02).abs() < 0.002);
    assert!((c[1][1] - 0.1).abs() < 0.004);
}
#[test]
fn uniform_low_rank_noise_covariance() {
    let c = covariance(Noise {
        innovation: InnovationLaw::StandardizedUniform,
        geometry: NoiseGeometry::LowRank {
            rank: 1,
            factor: FactorValues::Constant {
                values: vec![1., 2.],
            },
        },
    });
    assert!((c[0][0] - 1.).abs() < 0.04);
    assert!((c[0][1] - 2.).abs() < 0.08);
    assert!((c[1][1] - 4.).abs() < 0.16);
}
