use algorithmic_gas::{
    fitness::FitnessPipeline,
    partv_geometry::{self, MetricPolicy},
    physics::{
        fitness::{ConditionalFitnessCache, Objective, companion_distance},
        geometry::{
            FitnessJet, MetricJet, fitness_curvature, hessian_curvature, metric_curvature,
            metric_spectrum,
        },
        jet::JetSpace,
    },
};
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() <= t * (1. + a.abs() + b.abs()),
        "{a} != {b}, tolerance {t}"
    );
}
fn cubic(d: usize, x: &[f64]) -> FitnessJet<f64> {
    let s = JetSpace::new(d, 4).unwrap();
    let q: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| s.variable(v, i).unwrap())
        .collect();
    let mut f = Objective::Sphere.evaluate(&q).unwrap().scale(0.5);
    for i in 1..d {
        f = f.add(&q[0].mul(&q[i].pow(2.)).scale(0.5));
    }
    FitnessJet::from_jet(&f).unwrap()
}
#[test]
fn packed_three_dimensional_derivatives_are_exact() {
    let s = JetSpace::new(3, 3).unwrap();
    assert_eq!(s.indices.len(), 20);
    assert_eq!(JetSpace::new(3, 4).unwrap().indices.len(), 35);
    let x = s.variable(2., 0).unwrap();
    let y = s.variable(3., 1).unwrap();
    let z = s.variable(-1., 2).unwrap();
    let f = x.pow(2.).mul(&y).add(&z.pow(3.));
    close(f.derivative(&[0, 0, 1]).unwrap(), 2., 0.);
    close(f.derivative(&[2, 2, 2]).unwrap(), 6., 0.);
    let j = FitnessJet::from_jet(&f).unwrap();
    assert_eq!(j.hessian.len(), 6);
    assert_eq!(j.third.len(), 10);
    assert!(j.fourth.is_none());
}
#[test]
fn scalar_ricci_match_independent_christoffel_reference() {
    for d in [1, 2, 3, 4, 8] {
        let jet = cubic(d, &vec![0.03; d]);
        let h = jet.dense_hessian().unwrap();
        let fast = hessian_curvature(&jet, 0.2, true).unwrap();
        // For this cubic, dg is analytic and ddg=0. No use of the fast tensor formula.
        let mut first = vec![0.; d * d * d];
        for i in 1..d {
            first[i * d + i] = 1.;
            first[(i * d) * d + i] = 1.;
            first[(i * d + i) * d] = 1.;
        }
        let mut metric = h;
        for i in 0..d {
            metric[i * d + i] += 0.2;
        }
        let reference = metric_curvature(
            &MetricJet {
                dimension: d,
                metric,
                first,
                second: vec![0.; d.pow(4)],
            },
            true,
        )
        .unwrap();
        close(fast.scalar, reference.scalar, 2e-12);
        for (&a, &b) in fast.ricci.iter().zip(&reference.ricci) {
            close(a, b, 2e-12);
        }
        for (&a, &b) in fast
            .riemann
            .as_ref()
            .unwrap()
            .iter()
            .zip(reference.riemann.as_ref().unwrap())
        {
            close(a, b, 2e-12);
        }
        if d == 2 {
            for a in fast.einstein {
                close(a, 0., 2e-12);
            }
        }
    }
    let at_origin = hessian_curvature(&cubic(3, &[0., 0., 0.]), 1., false).unwrap();
    close(at_origin.scalar, 0.5 / 8., 1e-14);
}
#[test]
fn unit_sphere_has_positive_scalar_curvature() {
    let theta: f64 = 0.71;
    let sin = theta.sin();
    let cos = theta.cos();
    let mut first = vec![0.; 8];
    first[3] = 2. * sin * cos;
    let mut second = vec![0.; 16];
    second[3] = 2. * (cos * cos - sin * sin);
    let r = metric_curvature(
        &MetricJet {
            dimension: 2,
            metric: vec![1., 0., 0., sin * sin],
            first,
            second,
        },
        true,
    )
    .unwrap();
    close(r.scalar, 2., 1e-12);
    close(r.sectional[0], 1., 1e-12);
}
#[test]
fn anisotropic_flat_metric_and_precision() {
    for d in [1, 2, 3, 4, 8] {
        let mut j = cubic(d, &vec![0.; d]);
        j.third.fill(0.);
        let mut k = 0;
        for i in 0..d {
            for l in i..d {
                j.hessian[k] = if i == l { (i + 1) as f64 } else { 0. };
                k += 1;
            }
        }
        let r = hessian_curvature(&j, 1e-3, false).unwrap();
        assert_eq!(r.scalar, 0.);
        assert!(r.riemann.is_none());
    }
    let s = JetSpace::new(3, 3).unwrap();
    let x: Vec<_> = (0..3).map(|i| s.variable(0.2f32, i).unwrap()).collect();
    let j = FitnessJet::from_jet(&Objective::Rastrigin.evaluate(&x).unwrap()).unwrap();
    let r = fitness_curvature(&j, 0.1, MetricPolicy::Clipped, 1e-5, false).unwrap();
    assert!(r.scalar.is_finite());
}
#[test]
fn cached_statistics_match_existing_two_dimensional_pipeline() {
    let points: Vec<[f64; 2]> = vec![[0.1, 0.4], [0.3, 0.7], [0.9, -0.3], [-0.5, 0.6]];
    let companions = vec![1, 2, 3, 0];
    let pipeline = FitnessPipeline::default();
    let alive = vec![true, true, true, true];
    let rewards: Vec<_> = points.iter().map(|x| x[0] * x[0] + x[1] * x[1]).collect();
    let distances: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let p = points[companions[i]];
            ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2) + pipeline.distance_floor.powi(2)).sqrt()
        })
        .collect();
    let cache = ConditionalFitnessCache::new(&rewards, &distances, &alive, &pipeline).unwrap();
    let s = JetSpace::new(2, 4).unwrap();
    for target in 0..points.len() {
        let query = [0.23, -0.41];
        let x: Vec<_> = query
            .iter()
            .enumerate()
            .map(|(i, &v)| s.variable(v, i).unwrap())
            .collect();
        let r = Objective::Sphere.evaluate(&x).unwrap();
        let distance =
            companion_distance(&x, &points[companions[target]], 0., pipeline.distance_floor)
                .unwrap();
        let j = cache.evaluate(target, &r, &distance).unwrap();
        let reference = partv_geometry::conditional_fitness_pipeline(
            &partv_geometry::FitnessInput {
                points: points.clone(),
                velocities: vec![],
                alive: alive.clone(),
                companions: companions.clone(),
                companion_valid: vec![],
                target,
                query,
                objective: partv_geometry::Objective::Sphere,
                sigma_min: 1e-3,
                distance_floor: pipeline.distance_floor,
                velocity_weight: 0.,
                reward_exponent: 1.,
                distance_exponent: 1.,
                map_amplitude: 2.,
                map_floor: 1e-6,
                maximize: false,
                metric_epsilon: 0.1,
                metric_policy: MetricPolicy::Clipped,
            },
            &pipeline,
        )
        .unwrap();
        close(j.value(), reference.value, 2e-12);
        for a in 0..2 {
            close(j.derivative(&[a]).unwrap(), reference.gradient[a], 2e-11);
            for b in 0..2 {
                close(
                    j.derivative(&[a, b]).unwrap(),
                    reference.hessian[a][b],
                    2e-10,
                );
            }
        }
    }
}
#[test]
fn clipped_curvature_matches_finite_difference_metric() {
    let s = JetSpace::new(3, 4).unwrap();
    let potential = |point: &[f64]| {
        let x: Vec<_> = point
            .iter()
            .enumerate()
            .map(|(i, &v)| s.variable(v, i).unwrap())
            .collect();
        let f = x[0]
            .pow(2.)
            .sub(&x[1].pow(2.))
            .add(&x[2].pow(2.))
            .add(&x[0].mul(&x[1].pow(2.)).scale(0.7))
            .add(&x[0].pow(2.).mul(&x[2].pow(2.)).scale(0.3));
        FitnessJet::from_jet(&f).unwrap()
    };
    let p = [0.13, -0.17, 0.23];
    let j = potential(&p);
    let actual = fitness_curvature(&j, 0.4, MetricPolicy::Clipped, 1e-9, true).unwrap();
    assert_eq!(actual.branch, "mixed_clipped");
    let metric = |x: &[f64]| {
        metric_spectrum(
            &potential(x).dense_hessian().unwrap(),
            3,
            0.4,
            MetricPolicy::Clipped,
        )
        .unwrap()
        .metric
    };
    let base = metric(&p);
    let h = 1e-4;
    let mut first = vec![0.; 27];
    let mut second = vec![0.; 81];
    for k in 0..3 {
        let mut a = p;
        let mut b = p;
        a[k] += h;
        b[k] -= h;
        let ga = metric(&a);
        let gb = metric(&b);
        for ij in 0..9 {
            first[k * 9 + ij] = (ga[ij] - gb[ij]) / (2. * h);
            second[(k * 3 + k) * 9 + ij] = (ga[ij] - 2. * base[ij] + gb[ij]) / (h * h);
        }
        for l in 0..k {
            let mut pp = p;
            let mut pm = p;
            let mut mp = p;
            let mut mm = p;
            pp[k] += h;
            pp[l] += h;
            pm[k] += h;
            pm[l] -= h;
            mp[k] -= h;
            mp[l] += h;
            mm[k] -= h;
            mm[l] -= h;
            let (pp, pm, mp, mm) = (metric(&pp), metric(&pm), metric(&mp), metric(&mm));
            for ij in 0..9 {
                let v = (pp[ij] - pm[ij] - mp[ij] + mm[ij]) / (4. * h * h);
                second[(k * 3 + l) * 9 + ij] = v;
                second[(l * 3 + k) * 9 + ij] = v;
            }
        }
    }
    let reference = metric_curvature(
        &MetricJet {
            dimension: 3,
            metric: base,
            first,
            second,
        },
        true,
    )
    .unwrap();
    close(actual.scalar, reference.scalar, 2e-6);
    for (&a, &b) in actual.ricci.iter().zip(&reference.ricci) {
        close(a, b, 2e-6);
    }
    let mut incomplete = j.clone();
    incomplete.fourth = None;
    assert!(fitness_curvature(&incomplete, 0.4, MetricPolicy::Clipped, 1e-9, false).is_err());
    let mut threshold = cubic(3, &[0.; 3]);
    threshold.hessian.fill(0.);
    assert!(fitness_curvature(&threshold, 0.1, MetricPolicy::Clipped, 1e-9, false).is_err());
    threshold.hessian = vec![-1., 0., 0., -1., 0., -1.];
    assert_eq!(
        fitness_curvature(&threshold, 0.1, MetricPolicy::Clipped, 1e-9, false)
            .unwrap()
            .scalar,
        0.
    );
}

#[test]
fn finite_kernel_thermodynamics_and_poisson_response_have_exact_references() {
    use algorithmic_gas::physics::thermodynamics::{ExtendedValue, FiniteKernel};
    for (p, q) in [(0.3, 0.2), (0.4, 0.4), (0.7, 0.)] {
        let mut transition = vec![0.; 9];
        for i in 0..3 {
            transition[i * 3 + i] = 1. - p - q;
            transition[i * 3 + (i + 1) % 3] = p;
            transition[i * 3 + (i + 2) % 3] = q;
        }
        let report = FiniteKernel {
            states: 3,
            transition,
        }
        .thermodynamics()
        .unwrap();
        for pi in report.stationary {
            close(pi, 1. / 3., 1e-13);
        }
        if q == 0. {
            assert_eq!(report.irreversibility_per_step, ExtendedValue::Infinite);
        } else if let ExtendedValue::Finite(rate) = report.irreversibility_per_step {
            close(rate, (p - q) * (p / q).ln(), 1e-13);
        } else {
            panic!("finite support");
        }
    }
    let (a, b) = (0.3, 0.6);
    let kernel = FiniteKernel {
        states: 2,
        transition: vec![1. - a, a, b, 1. - b],
    };
    let r = kernel.response(&[-1., 1., 0., 0.], &[0., 1.]).unwrap();
    close(r.observable_response, b / (a + b).powi(2), 1e-13);
    close(r.poisson_response, r.observable_response, 1e-13);
    close(r.stationary_fisher, b / (a * (a + b).powi(2)), 1e-13);
    close(r.transition_fisher, b / ((a + b) * a * (1. - a)), 1e-13);
    assert!(
        FiniteKernel {
            states: 2,
            transition: vec![1., 0., 0., 1.]
        }
        .stationary()
        .is_err()
    );
}

#[test]
fn arbitrary_cubic_tensors_match_independent_metric_curvature() {
    use algorithmic_gas::random::{RandomStream, Stream};
    for d in [2usize, 3, 4] {
        for seed in 0..5 {
            let space = JetSpace::new(d, 4).unwrap();
            let mut rng = RandomStream::new(seed, 0, Stream::Initialize, 0, 0);
            let x: Vec<_> = (0..d)
                .map(|axis| space.variable(rng.uniform::<f64>() * 0.1, axis).unwrap())
                .collect();
            let mut f = Objective::Sphere.evaluate(&x).unwrap();
            for a in 0..d {
                for b in a..d {
                    for c in b..d {
                        f = f.add(
                            &x[a]
                                .mul(&x[b])
                                .mul(&x[c])
                                .scale(rng.gaussian::<f64>() * 0.1),
                        );
                    }
                }
            }
            let jet = FitnessJet::from_jet(&f).unwrap();
            let mut metric = jet.dense_hessian().unwrap();
            for i in 0..d {
                metric[i * d + i] += 0.3;
            }
            let mut first = vec![];
            for a in 0..d {
                for b in 0..d {
                    for c in 0..d {
                        first.push(f.derivative(&[a, b, c]).unwrap());
                    }
                }
            }
            let fast = hessian_curvature(&jet, 0.3, true).unwrap();
            let reference = metric_curvature(
                &MetricJet {
                    dimension: d,
                    metric,
                    first,
                    second: vec![0.; d.pow(4)],
                },
                true,
            )
            .unwrap();
            close(fast.scalar, reference.scalar, 2e-12);
            for (&a, &b) in fast
                .riemann
                .unwrap()
                .iter()
                .zip(reference.riemann.unwrap().iter())
            {
                close(a, b, 2e-12);
            }
        }
    }
}

#[test]
fn conditional_thermostat_moments_match_independent_simulated_increments() {
    use algorithmic_gas::{
        noise::InnovationLaw,
        physics::balances::thermostat_moments,
        random::{RandomStream, Stream},
    };
    let v = [0.7, -0.3, 1.1];
    let b = [0.8, -0.1, 0.2, 0.7, 0.3, -0.4];
    for law in [InnovationLaw::Gaussian, InnovationLaw::StandardizedUniform] {
        for gamma in [0., 0.7] {
            let prediction = thermostat_moments(&v, &b, 3, 2, &[true], 0.13, gamma, law).unwrap();
            let samples = 131072;
            let mut mean = 0.;
            let mut m2 = 0.;
            for sample in 0..samples {
                let mut rng = RandomStream::new(97431, sample as u64, Stream::Initialize, 0, 0);
                let mut xi = [0.; 2];
                for x in &mut xi {
                    *x = match law {
                        InnovationLaw::Gaussian => rng.gaussian::<f64>(),
                        InnovationLaw::StandardizedUniform => {
                            (2. * rng.uniform::<f64>() - 1.) * 3f64.sqrt()
                        }
                    };
                }
                let mut delta = 0.;
                for i in 0..3 {
                    let eta = (0..2).map(|j| b[i * 2 + j] * xi[j]).sum::<f64>();
                    let next =
                        prediction.decay * v[i] + prediction.innovation_scale_squared.sqrt() * eta;
                    delta += 0.5 * (next * next - v[i] * v[i]);
                }
                let diff = delta - mean;
                mean += diff / (sample + 1) as f64;
                m2 += diff * (delta - mean);
            }
            let variance = m2 / (samples - 1) as f64;
            assert!(
                (mean - prediction.energy_mean_delta).abs()
                    < 6. * (prediction.energy_variance / samples as f64).sqrt()
            );
            close(variance, prediction.energy_variance, 0.005);
        }
    }
}

#[test]
fn independent_validation_rejects_reused_samples_and_false_closure() {
    use algorithmic_gas::physics::{
        closure::{ClosureSample, test_linear_closure},
        evolution::{ReplicaSample, metric_evolution},
    };
    let train: Vec<_> = (0..30)
        .map(|i| ReplicaSample {
            key: i,
            increment: vec![i as f64],
        })
        .collect();
    let valid: Vec<_> = (30..60)
        .map(|i| ReplicaSample {
            key: i,
            increment: vec![(i - 30) as f64],
        })
        .collect();
    let r = metric_evolution(&train, &valid).unwrap();
    assert_eq!(r.validation_residual, vec![0.]);
    assert!(metric_evolution(&train, &train).is_err());
    let sample = |key: u64| {
        let x = key as f64 / 10.;
        ClosureSample {
            key,
            predictors: vec![1., x],
            observed: 2. + 3. * x,
        }
    };
    let train: Vec<_> = (0..30).map(sample).collect();
    let valid: Vec<_> = (30..60).map(sample).collect();
    let exact = test_linear_closure(&train, &valid, 0.).unwrap();
    assert!(exact.validation_rmse < 1e-12);
    let false_train: Vec<_> = train
        .iter()
        .cloned()
        .map(|mut s| {
            s.observed = 2.;
            s
        })
        .collect();
    assert!(
        test_linear_closure(&false_train, &valid, 0.)
            .unwrap()
            .validation_rmse
            > 10.
    );
    assert!(test_linear_closure(&train, &train, 0.).is_err());
}

#[test]
fn eigensolver_handles_large_finite_entries_and_rejects_nonintegrable_metric_jet() {
    use algorithmic_gas::physics::geometry::symmetric_eigen;
    let (mut eigen, _) = symmetric_eigen(&[1e200, 5e199, 5e199, 1e200], 2).unwrap();
    eigen.sort_by(f64::total_cmp);
    close(eigen[0] / 1e200, 0.5, 1e-13);
    close(eigen[1] / 1e200, 1.5, 1e-13);
    let mut jet = MetricJet {
        dimension: 2,
        metric: vec![1., 0., 0., 1.],
        first: vec![0.; 8],
        second: vec![0.; 16],
    };
    jet.second[4] = 1.;
    assert!(metric_curvature(&jet, false).is_err());
}

#[test]
fn batched_backend_curvature_matches_scalar_reference_and_respects_budget() {
    use algorithmic_gas::{BackendKind, ExecutionContext, Precision};
    futures_lite::future::block_on(async {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        for d in [1, 2, 3, 4, 8] {
            let jets: Vec<_> = (0..7)
                .map(|i| cubic(d, &vec![0.01 * i as f64; d]))
                .collect();
            let actual = cx
                .smooth_curvature_batch(&jets, 0.2, MetricPolicy::Strict, 1e-9)
                .await
                .unwrap();
            for (jet, actual) in jets.iter().zip(actual) {
                let reference = hessian_curvature(jet, 0.2, false).unwrap();
                close(actual.scalar, reference.scalar, 3e-12);
                for (&a, &b) in actual.ricci.iter().zip(&reference.ricci) {
                    close(a, b, 3e-12);
                }
                for (&a, &b) in actual.sectional.iter().zip(&reference.sectional) {
                    close(a, b, 3e-12);
                }
            }
        }
        cx.max_batch_elements = 1;
        assert!(
            cx.smooth_curvature_batch(&[cubic(3, &[0.; 3])], 0.2, MetricPolicy::Strict, 1e-9)
                .await
                .is_err()
        );
    });
}

#[test]
fn local_three_dimensional_jets_match_production_normalization_and_finite_differences() {
    use algorithmic_gas::{
        ObservationBatch, TensorBatch,
        fitness::{PositiveMapping, Standardizer},
        geometry::{Distance, Kernel},
        physics::fitness::{local_log_weights, pipeline_from_measurement_jets},
    };
    let points = vec![
        vec![0.2, 0.1, -0.3],
        vec![0.7, -0.2, 0.4],
        vec![-0.5, 0.6, 0.1],
        vec![0.3, -0.7, 0.8],
    ];
    let alive = vec![true; 4];
    let target = 0;
    let query = [0.23, -0.14, 0.34];
    let mut pipeline = FitnessPipeline::default();
    pipeline.reward_standardizer = Standardizer::Local {
        sigma_min: 0.03,
        distance: Distance::default(),
        kernel: Kernel::Gaussian { width: 0.8 },
        include_self: false,
    };
    pipeline.diversity_standardizer = pipeline.reward_standardizer.clone();
    let rewards: Vec<f64> = points
        .iter()
        .map(|p| p.iter().map(|x| x * x).sum())
        .collect();
    let distances = vec![0.5, 0.3, 0.7, 0.9];
    let compute = |query: &[f64], order| {
        let s = JetSpace::new(3, order).unwrap();
        let x: Vec<_> = query
            .iter()
            .enumerate()
            .map(|(a, &v)| s.variable(v, a).unwrap())
            .collect();
        let mut r: Vec<_> = rewards.iter().map(|&v| s.constant(v)).collect();
        r[target] = Objective::Sphere.evaluate(&x).unwrap();
        let mut dist: Vec<_> = distances.iter().map(|&v| s.constant(v)).collect();
        dist[target] = companion_distance(&x, &points[1], 0., pipeline.distance_floor).unwrap();
        let logs = local_log_weights(
            &x,
            &points,
            &[],
            &alive,
            target,
            &pipeline.reward_standardizer,
        )
        .unwrap()
        .unwrap();
        pipeline_from_measurement_jets(
            &r,
            &dist,
            &alive,
            target,
            &pipeline,
            Some(&logs),
            Some(&logs),
        )
        .unwrap()
    };
    let j = compute(&query, 4);
    let reference = |query: &[f64]| {
        let mut coordinates = points.clone();
        coordinates[target] = query.to_vec();
        let obs = ObservationBatch::positions(
            TensorBatch::vectors(4, 3, coordinates.iter().flatten().copied().collect()).unwrap(),
        );
        let mut r = rewards.clone();
        r[target] = query.iter().map(|x| x * x).sum();
        for v in &mut r {
            *v = -*v;
        }
        let mut ds = distances.clone();
        ds[target] = (query
            .iter()
            .zip(&points[1])
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            + pipeline.distance_floor.powi(2))
        .sqrt();
        let zr = pipeline
            .reward_standardizer
            .apply(&r, &alive, &obs)
            .unwrap()
            .0;
        let zd = pipeline
            .diversity_standardizer
            .apply(&ds, &alive, &obs)
            .unwrap()
            .0;
        pipeline.reward_map.map(zr[target]).unwrap()
            * pipeline.diversity_map.map(zd[target]).unwrap()
    };
    close(j.value(), reference(&query), 1e-12);
    for a in 0..3 {
        let h = 1e-4;
        let mut plus = query;
        let mut minus = query;
        plus[a] += h;
        minus[a] -= h;
        close(
            j.derivative(&[a]).unwrap(),
            (reference(&plus) - reference(&minus)) / (2. * h),
            2e-6,
        );
        let pj = compute(&plus, 3);
        let mj = compute(&minus, 3);
        for b in 0..3 {
            for c in 0..3 {
                close(
                    j.derivative(&[a, b, c]).unwrap(),
                    (pj.derivative(&[b, c]).unwrap() - mj.derivative(&[b, c]).unwrap()) / (2. * h),
                    2e-5,
                );
            }
        }
    }
}

#[test]
fn coupled_population_derivative_matches_executed_fitness_with_fixed_companions() {
    use algorithmic_gas::{
        ObservationBatch, Provenance, RewardBatch, TensorBatch,
        physics::fitness::pipeline_from_measurement_jets,
    };
    let points = [
        [0.2_f64, 0.1, -0.3],
        [0.7, -0.2, 0.4],
        [-0.5, 0.6, 0.1],
        [0.3, -0.7, 0.8],
    ];
    // Two other rows use the perturbed walker as their actual companion.
    let companions = [1, 0, 0, 2];
    let alive = [true; 4];
    let pipeline = FitnessPipeline::default();
    let space = JetSpace::new(3, 3).unwrap();
    let coordinates: Vec<Vec<_>> = points
        .iter()
        .enumerate()
        .map(|(row, p)| {
            p.iter()
                .enumerate()
                .map(|(axis, &v)| {
                    if row == 0 {
                        space.variable(v, axis).unwrap()
                    } else {
                        space.constant(v)
                    }
                })
                .collect()
        })
        .collect();
    let rewards: Vec<_> = coordinates
        .iter()
        .map(|p| Objective::Sphere.evaluate(p).unwrap())
        .collect();
    let separation: Vec<_> = (0..4)
        .map(|row| {
            (0..3)
                .fold(
                    space.constant(pipeline.distance_floor.powi(2)),
                    |s, axis| {
                        s.add(
                            &coordinates[row][axis]
                                .sub(&coordinates[companions[row]][axis])
                                .pow(2.),
                        )
                    },
                )
                .pow(0.5)
        })
        .collect();
    let coupled =
        pipeline_from_measurement_jets(&rewards, &separation, &alive, 0, &pipeline, None, None)
            .unwrap();
    let execute = |query: [f64; 3]| {
        let mut p = points;
        p[0] = query;
        let obs = ObservationBatch::positions(
            TensorBatch::vectors(4, 3, p.iter().flatten().copied().collect()).unwrap(),
        );
        let r = RewardBatch::new(
            p.iter().map(|v| v.iter().map(|x| x * x).sum()).collect(),
            Provenance::default(),
        );
        let distances: Vec<_> = (0..4)
            .map(|row| {
                (0..3)
                    .map(|axis| (p[row][axis] - p[companions[row]][axis]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect();
        pipeline
            .evaluate(&r, &distances, &alive, &obs, 0)
            .unwrap()
            .fitness[0]
    };
    let center = points[0];
    close(coupled.value(), execute(center), 1e-12);
    let h = 1e-4;
    for a in 0..3 {
        let mut plus = center;
        plus[a] += h;
        let mut minus = center;
        minus[a] -= h;
        close(
            coupled.derivative(&[a]).unwrap(),
            (execute(plus) - execute(minus)) / (2. * h),
            2e-6,
        );
        for b in 0..3 {
            let mut pp = plus;
            pp[b] += h;
            let mut pm = plus;
            pm[b] -= h;
            let mut mp = minus;
            mp[b] += h;
            let mut mm = minus;
            mm[b] -= h;
            close(
                coupled.derivative(&[a, b]).unwrap(),
                (execute(pp) - execute(pm) - execute(mp) + execute(mm)) / (4. * h * h),
                2e-5,
            );
        }
    }
    let cache = ConditionalFitnessCache::new(
        &rewards.iter().map(|v| v.value()).collect::<Vec<_>>(),
        &separation.iter().map(|v| v.value()).collect::<Vec<_>>(),
        &alive,
        &pipeline,
    )
    .unwrap();
    let frozen = cache
        .evaluate(
            0,
            &rewards[0],
            &companion_distance(&coordinates[0], &points[1], 0., pipeline.distance_floor).unwrap(),
        )
        .unwrap();
    close(frozen.value(), coupled.value(), 1e-12);
    assert!(
        (0..3).any(
            |a| (frozen.derivative(&[a]).unwrap() - coupled.derivative(&[a]).unwrap()).abs() > 1e-3
        ),
        "query and coupled-population derivatives must preserve their different dependency graphs"
    );
}

#[test]
fn rotated_negative_direction_keeps_the_clipped_metric_flat() {
    fn check<T: algorithmic_gas::Real>(tolerance: f64) {
        let space = JetSpace::new(3, 4).unwrap();
        let x: Vec<_> = (0..3)
            .map(|i| space.variable(T::ZERO, i).unwrap())
            .collect();
        for i in 0..12 {
            let theta = 0.15 + 0.1 * i as f64;
            let phi = 0.23 * i as f64;
            let n = [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ];
            let p = [-phi.sin(), phi.cos(), 0.];
            let t = [
                n[1] * p[2] - n[2] * p[1],
                n[2] * p[0] - n[0] * p[2],
                n[0] * p[1] - n[1] * p[0],
            ];
            let linear = |axis: [f64; 3]| {
                (0..3).fold(space.constant(T::ZERO), |s, j| {
                    s.add(&x[j].scale(T::from_f64(axis[j])))
                })
            };
            let normal = linear(n);
            // Only the negative eigenvalue varies near the query. Clipping it
            // gives the same constant positive metric throughout that region.
            let potential = linear(p)
                .pow(2.)
                .sub(&normal.pow(2.).scale(T::from_f64(1.5)))
                .sub(&linear(t).pow(2.).scale(T::from_f64(0.5)))
                .add(&normal.pow(3.).scale(T::from_f64(0.5)))
                .add(&normal.pow(4.).scale(T::from_f64(1. / 12.)));
            let jet = FitnessJet::from_jet(&potential).unwrap();
            let result = fitness_curvature(
                &jet,
                T::from_f64(0.2),
                MetricPolicy::Clipped,
                T::from_f64(1e-6),
                true,
            )
            .unwrap();
            assert_eq!(result.branch, "mixed_clipped");
            close(result.scalar.to_f64(), 0., tolerance);
            for value in result.ricci.iter().chain(result.riemann.as_ref().unwrap()) {
                close(value.to_f64(), 0., tolerance);
            }
        }
    }
    check::<f64>(1e-10);
    check::<f32>(1e-4);
}
