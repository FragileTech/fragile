use algorithmic_gas::{
    donor::{
        CompanionBatch, CompanionReducer, CompanionRequest, CompanionSampler, DonorModule,
        DonorPool, InsufficientPolicy, OddPolicy, PivotOrder, SamplingLaw,
    },
    fitness::{FitnessPipeline, PositiveMap, Standardizer},
    geometry::{AlgorithmicDistance, Distance, Kernel},
    random::Stream,
    *,
};
use futures_lite::future::block_on;

fn population<T: Real>(values: &[f64]) -> Population<T> {
    Population::new(ObservationBatch::positions(
        TensorBatch::vectors(
            values.len(),
            1,
            values.iter().map(|&x| T::from_f64(x)).collect(),
        )
        .unwrap(),
    ))
    .unwrap()
}

fn squared_reduction<T: Real>() {
    block_on(async {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, T::PRECISION)
            .await
            .unwrap();
        let p = population::<T>(&[0., 3., 4.]);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let batch = CompanionBatch {
            rows: 3,
            count: 2,
            indices: vec![1, 2, 0, 2, 0, 1],
            valid: vec![true; 6],
            mutual: false,
        };
        for reducer in [
            CompanionReducer::Mean,
            CompanionReducer::WeightedMean {
                weights: vec![1., 3.],
            },
            CompanionReducer::Minimum,
            CompanionReducer::Maximum,
        ] {
            let distance = Distance::default();
            let squared = Distance::Euclidean {
                field: "positions".into(),
                scales: vec![],
                squared: true,
                periodic: None,
            };
            let a = reducer
                .measure(&distance, &p, &pool, &batch, &mut cx)
                .await
                .unwrap();
            let b = reducer
                .measure(&squared, &p, &pool, &batch, &mut cx)
                .await
                .unwrap();
            assert_eq!(a, b);
            let fitness = FitnessPipeline::default();
            assert_eq!(
                fitness
                    .evaluate(&p.rewards, &a, &[true; 3], &p.observations, 0)
                    .unwrap(),
                fitness
                    .evaluate(&p.rewards, &b, &[true; 3], &p.observations, 0)
                    .unwrap()
            );
        }
    });
}
#[test]
fn squared_distances_preserve_reducer_and_fitness_semantics() {
    squared_reduction::<f32>();
    squared_reduction::<f64>();
}

#[test]
fn local_statistics_exclude_self_and_record_singleton_fallback() {
    let p = population::<f64>(&[0., 2., 9.]);
    let local: Standardizer = serde_json::from_str(r#"{"kind":"local","sigma_min":1.0,"distance":{"kind":"euclidean","field":"positions","scales":[],"squared":false,"periodic":null},"kernel":{"kind":"uniform"}}"#).unwrap();
    let (z, s) = local
        .apply(&[0., 2., f64::NAN], &[true, true, false], &p.observations)
        .unwrap();
    assert_eq!(z, vec![-2., 2., 0.]);
    assert_eq!(&s.mean[..2], &[2., 0.]);
    assert_eq!(s.global_fallback, vec![false; 3]);
    let (z, s) = local
        .apply(
            &[7., f64::NAN, f64::NAN],
            &[true, false, false],
            &p.observations,
        )
        .unwrap();
    assert_eq!(z, vec![0.; 3]);
    assert_eq!(s.global_fallback, vec![true, false, false]);
}

fn extreme_cosine<T: Real>(small: f64, large: f64) {
    block_on(async {
        let p = population::<T>(&[small, -small, large, -large, 0.]);
        let d = Distance::Cosine {
            field: "positions".into(),
            zero_tolerance: 0.,
        };
        let edges = [(0, 1), (2, 3), (0, 2), (0, 4), (4, 4)];
        let expected = [2., 2., 0., 1., 0.];
        let mut cx = ExecutionContext::new(BackendKind::Cpu, T::PRECISION)
            .await
            .unwrap();
        let batch = d
            .pairs(&p.observations, &p.observations, &edges, &mut cx)
            .await
            .unwrap();
        for ((i, j), (actual, expected)) in edges.into_iter().zip(batch.into_iter().zip(expected)) {
            assert_eq!(actual, T::from_f64(expected));
            assert_eq!(
                d.compare(&p.observations, i as usize, &p.observations, j as usize)
                    .unwrap(),
                actual
            );
        }
    });
}
#[test]
fn cosine_preserves_small_and_large_nonzero_vectors() {
    extreme_cosine::<f32>(1e-30, 1e30);
    extreme_cosine::<f64>(1e-200, 1e200);
}

#[test]
fn tensor_deserialization_validates_shapes_without_panicking() {
    for input in [
        r#"{"rows":2,"item_shape":[1],"values":[1.0]}"#,
        r#"{"rows":1,"item_shape":[18446744073709551615,2],"values":[1.0]}"#,
        r#"{"rows":0,"item_shape":[1],"values":[]}"#,
        r#"{"rows":1,"item_shape":[0],"values":[]}"#,
    ] {
        assert!(serde_json::from_str::<TensorBatch<f64>>(input).is_err());
    }
    let tensor: TensorBatch<f64> =
        serde_json::from_str(r#"{"rows":1,"item_shape":[1],"values":[1.0]}"#).unwrap();
    assert!(tensor.row(1).is_err());
    assert!(tensor.gather(&[u32::MAX]).is_err());
    let malformed = CompanionBatch {
        rows: 1,
        count: usize::MAX,
        indices: vec![],
        valid: vec![],
        mutual: false,
    };
    assert_eq!(malformed.row(usize::MAX).count(), 0);
    let rewards = RewardBatch {
        raw: vec![1.],
        valid: vec![],
        provenance: Default::default(),
    };
    assert!(rewards.gather(&[0]).is_err());
}

#[test]
fn insufficient_candidates_are_padded_or_explicitly_rejected() {
    block_on(async {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        for kernel in [Kernel::Uniform, Kernel::Gaussian { width: 1. }] {
            for values in [&[0., 1.][..], &[0.][..]] {
                let p = population::<f64>(values);
                let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
                let alive = vec![true; p.len()];
                let request = || CompanionRequest {
                    population: &p,
                    pool: &pool,
                    eligible: &alive,
                    seed: 7,
                    step: 1,
                    stream: Stream::Distance,
                };
                let mut module = DonorModule {
                    count: 3,
                    replacement: false,
                    kernel: kernel.clone(),
                    ..Default::default()
                };
                let output = module.sample(request(), &mut cx).await.unwrap();
                assert_eq!(output.count, 3);
                for i in 0..p.len() {
                    assert_eq!(output.row(i).count(), 1);
                }
                module.insufficient = InsufficientPolicy::Reject;
                assert!(module.sample(request(), &mut cx).await.is_err());
            }
        }
    });
}

#[test]
fn uniform_sampling_handles_large_populations_without_pair_materialization() {
    block_on(async {
        let n = 20_000;
        let p = population::<f32>(&vec![0.; n]);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let alive = vec![true; n];
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F32)
            .await
            .unwrap();
        for law in [SamplingLaw::Independent, SamplingLaw::FisherYates] {
            let module = DonorModule {
                law,
                kernel: Kernel::Uniform,
                ..Default::default()
            };
            let batch = module
                .sample(
                    CompanionRequest {
                        population: &p,
                        pool: &pool,
                        eligible: &alive,
                        seed: 7,
                        step: 1,
                        stream: Stream::Distance,
                    },
                    &mut cx,
                )
                .await
                .unwrap();
            for i in 0..n {
                let j = batch.row(i).next().unwrap() as usize;
                assert_ne!(i, j);
                if law == SamplingLaw::FisherYates {
                    assert_eq!(batch.row(j).next(), Some(i as u32));
                }
            }
        }
        assert_eq!(cx.stats.evaluations, 0);
    });
}

#[test]
fn uniform_without_replacement_has_uniform_subsets() {
    block_on(async {
        let p = population::<f64>(&[0., 1., 2., 3.]);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        let module = DonorModule {
            kernel: Kernel::Uniform,
            count: 2,
            replacement: false,
            ..Default::default()
        };
        let mut missing = [0usize; 4];
        for seed in 0..6000 {
            let b = module
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
            let row = b.row(0).collect::<Vec<_>>();
            assert_eq!(row.len(), 2);
            assert_ne!(row[0], row[1]);
            assert!(!row.contains(&0));
            for (j, count) in missing.iter_mut().enumerate().skip(1) {
                if !row.contains(&(j as u32)) {
                    *count += 1;
                }
            }
        }
        // Each omitted donor has p=1/3; a 200-count band exceeds five sigma.
        for &count in &missing[1..] {
            assert!((1800..2200).contains(&count), "{missing:?}");
        }
    });
}

#[test]
fn memory_limits_cover_graph_totals_and_checked_population_reservations() {
    block_on(async {
        let p = population::<f64>(&[0., 1.]);
        let c = GasConfig {
            precision: Precision::F64,
            max_memory_bytes: 1,
            ..Default::default()
        };
        assert!(c.validate(&p, false).is_err());
        let mut c = GasConfig {
            precision: Precision::F64,
            ..Default::default()
        };
        c.distance_donors.history_window = usize::MAX;
        assert!(c.validate(&p, false).is_err());
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        cx.max_batch_elements = 1024;
        cx.max_memory_bytes = 256;
        let mut expression = algorithmic_gas::compute::Expression::default();
        let input = expression.input(0);
        for _ in 0..20 {
            expression.unary(algorithmic_gas::compute::Unary::Square, input);
        }
        assert!(
            cx.evaluate(
                &expression,
                &[TensorBatch::vectors(2, 1, vec![1., 2.]).unwrap()]
            )
            .await
            .is_err()
        );
        assert_eq!(cx.stats.evaluations, 0);
    });
}

/// Sequential weighted sampling without replacement is an ordered law: entry
/// `a` is draw `a`, so the first entry is categorical in the kernel weights.
/// Positional consumer weights (`WeightedMean`) depend on that order.
#[test]
fn gaussian_without_replacement_rows_are_in_draw_order() {
    block_on(async {
        let p = population::<f64>(&[0., 1., 2.]);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        let mut module = DonorModule {
            kernel: Kernel::Gaussian { width: 1. },
            count: 2,
            replacement: false,
            ..Default::default()
        };
        let mut nearest_first = 0usize;
        for seed in 0..4000 {
            let request = || CompanionRequest {
                population: &p,
                pool: &pool,
                eligible: &[true; 3],
                seed,
                step: 1,
                stream: Stream::Distance,
            };
            module.tile_edges = 4096;
            let batch = module.sample(request(), &mut cx).await.unwrap();
            let row = batch.row(0).collect::<Vec<_>>();
            assert_eq!(row.len(), 2);
            assert!(row.contains(&1) && row.contains(&2));
            nearest_first += usize::from(row[0] == 1);
            // The order is a property of the draws, not of tile traversal.
            module.tile_edges = 1;
            assert_eq!(batch, module.sample(request(), &mut cx).await.unwrap());
        }
        // w1 = exp(-1/2), w2 = exp(-2): P(first = 1) = 0.8176, sigma = 24.4.
        assert!((3150..3390).contains(&nearest_first), "{nearest_first}");
    });
}

/// Greedy matching pivots on the first remaining slot in ascending order by
/// default; a random pivot permutation is a different, explicit law.
#[test]
fn greedy_pivot_order_is_explicit() {
    block_on(async {
        let p = population::<f64>(&[0., 1., 1.2]);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        let mut module = DonorModule {
            law: SamplingLaw::GaussianGreedy,
            kernel: Kernel::Gaussian { width: 1. },
            ..Default::default()
        };
        assert_eq!(module.pivot, PivotOrder::Ascending);
        let mut chose_one = 0usize;
        let mut leftover = [0usize; 2];
        for (law, pivot) in [PivotOrder::Ascending, PivotOrder::Random]
            .into_iter()
            .enumerate()
        {
            module.pivot = pivot;
            for seed in 0..3000 {
                let batch = module
                    .sample(
                        CompanionRequest {
                            population: &p,
                            pool: &pool,
                            eligible: &[true; 3],
                            seed,
                            step: 1,
                            stream: Stream::Distance,
                        },
                        &mut cx,
                    )
                    .await
                    .unwrap();
                assert!(batch.mutual);
                let partner = batch.row(0).next().unwrap();
                leftover[law] += usize::from(partner == 0);
                if pivot == PivotOrder::Ascending {
                    chose_one += usize::from(partner == 1);
                    assert_eq!(batch.row(partner as usize).next(), Some(0));
                }
            }
        }
        // Slot 0 is the first ascending pivot: it is always matched and its
        // partner is categorical in w01 = exp(-0.5), w02 = exp(-0.72).
        assert_eq!(leftover[0], 0);
        assert!((1500..1830).contains(&chose_one), "{chose_one}");
        // Under a uniform pivot permutation it is left over when slot 1 picks
        // slot 2 or slot 2 picks slot 1, w12 = exp(-0.02):
        // (0.6178 + 0.6682) / 3 = 0.4287, sigma = 27.1.
        assert!((1150..1422).contains(&leftover[1]), "{leftover:?}");
        module.law = SamplingLaw::Independent;
        assert!(module.validate(&p.observations).is_err());
    });
}

/// An unmatched walker has no donor: reporting zero separation would fabricate
/// a diversity measurement.
#[test]
fn unmatched_walker_has_no_diversity_measurement() {
    block_on(async {
        let p = population::<f64>(&[0., 1., 2.]);
        let pool = DonorPool::freeze(&p, 0, &[], 0, false).unwrap();
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        let mut module = DonorModule {
            law: SamplingLaw::FisherYates,
            kernel: Kernel::Uniform,
            odd: OddPolicy::Unmatched,
            ..Default::default()
        };
        let request = || CompanionRequest {
            population: &p,
            pool: &pool,
            eligible: &[true; 3],
            seed: 3,
            step: 1,
            stream: Stream::Distance,
        };
        let unmatched = module.sample(request(), &mut cx).await.unwrap();
        assert!(matches!(
            CompanionReducer::Mean
                .measure(&Distance::default(), &p, &pool, &unmatched, &mut cx)
                .await,
            Err(GasError::Topology(_))
        ));
        module.odd = OddPolicy::SelfCompanion;
        let matched = module.sample(request(), &mut cx).await.unwrap();
        let separation = CompanionReducer::Mean
            .measure(&Distance::default(), &p, &pool, &matched, &mut cx)
            .await
            .unwrap();
        assert_eq!(separation.iter().filter(|&&d| d == 0.).count(), 1);
    });
}

/// A disabled channel contributes one and is not evaluated: its positive map
/// cannot fail a run that does not consume it.
#[test]
fn disabled_fitness_channel_is_not_evaluated() {
    let mut pipeline = FitnessPipeline {
        diversity_map: PositiveMap::LegacyAsymmetric { floor: 0. },
        ..Default::default()
    };
    // exp(-200) underflows to zero in f32, which the unfloored map rejects.
    let (reward_z, diversity_z) = ([0.5_f32], [-200_f32]);
    assert!(pipeline.combine(&reward_z, &diversity_z, &[true]).is_err());
    pipeline.diversity_exponent = 0.;
    let reward_only = pipeline.combine(&reward_z, &diversity_z, &[true]).unwrap();
    let mut both = pipeline.clone();
    both.diversity_exponent = 1.;
    both.diversity_map = PositiveMap::default();
    let reference = both.combine(&reward_z, &[0_f32], &[true]).unwrap();
    // Logistic(0) + floor is the second factor of the reference.
    assert!((reward_only[0] * (1. + 1e-6) - reference[0]).abs() < 1e-6);
    pipeline.reward_exponent = 0.;
    assert_eq!(
        pipeline.combine(&reward_z, &diversity_z, &[true]).unwrap(),
        vec![1.]
    );
}
