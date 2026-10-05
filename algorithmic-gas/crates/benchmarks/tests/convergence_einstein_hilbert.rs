use algorithmic_gas::{GasConfig, Precision, RecordingConfig};
use algorithmic_gas_benchmarks::{
    RunConfig,
    convergence_einstein_hilbert::{analyze_step, covariance, matching_moments, mean},
};
use futures_lite::future::block_on;
fn permutations(v: &mut [usize], k: usize, visit: &mut impl FnMut(&[usize])) {
    if k == v.len() {
        visit(v);
        return;
    }
    for j in k..v.len() {
        v.swap(k, j);
        permutations(v, k + 1, visit);
        v.swap(k, j);
    }
}
#[test]
fn exact_matching_second_moments_match_exhaustive_even_and_odd_laws() {
    let decision = GasConfig::einstein_hilbert(0.33, 0.002)
        .unwrap()
        .clone_decision;
    for n in 4..=7 {
        for step in [19, 20] {
            let d = 2;
            let nf = n as f64;
            let x: Vec<_> = (0..n * d)
                .map(|i| (i as f64 * 0.7).sin() + i as f64 * 0.1)
                .collect();
            let v: Vec<_> = (0..n * d).map(|i| (i as f64 * 1.1).cos()).collect();
            let f: Vec<_> = (0..n).map(|i| [0.1, 0.13, 1.2, 1.2][i % 4]).collect();
            let prediction = matching_moments(&x, &v, &f, d, &decision, step).unwrap();
            let mx = mean(&x, d);
            let mut sum = [0.; 3];
            let mut count = 0.;
            permutations(&mut (0..n).collect::<Vec<_>>(), 0, &mut |order| {
                let mut partner: Vec<_> = (0..n).collect();
                for pair in order.chunks_exact(2) {
                    partner[pair[0]] = pair[1];
                    partner[pair[1]] = pair[0];
                }
                let mut m = x.clone();
                let mut trace = 0.;
                for i in 0..n {
                    let j = partner[i];
                    let p = decision.acceptance_probability(step, f[i], f[j]);
                    for a in 0..d {
                        let dx = x[j * d + a] - x[i * d + a];
                        m[i * d + a] += p * dx;
                        trace += p * (1. - p) * dx * dx;
                    }
                }
                sum[0] += covariance(&m, &m, d) + (1. - 1. / nf) * trace / nf;
                sum[1] += mean(&m, d)
                    .iter()
                    .zip(&mx)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    + trace / (nf * nf);
                sum[2] += covariance(&m, &v, d);
                count += 1.;
            });
            for (actual, expected) in sum.iter().map(|x| x / count).zip([
                prediction.expected_variance,
                prediction.center_displacement_second_moment,
                prediction.expected_cross_covariance,
            ]) {
                assert!(
                    (actual - expected).abs() < 1e-11,
                    "N={n} step={step}: {actual} != {expected}"
                );
            }
            assert!(prediction.center_variance >= -1e-13);
            assert!(prediction.center_variance <= prediction.center_variance_bound + 1e-13);
            assert!(prediction.center_variance_bound <= 3. * prediction.entering_variance / nf);
        }
    }
}
#[test]
fn full_native_even_and_odd_trajectories_satisfy_centered_stage_identities() {
    for n in [8, 17] {
        let mut config = RunConfig::einstein_hilbert().unwrap();
        config.walkers = n;
        config.gas.precision = Precision::F64;
        let mut gas = block_on(config.build::<f64>()).unwrap();
        for _ in 0..40 {
            gas.start_recording(RecordingConfig {
                graph: true,
                ..RecordingConfig::default()
            })
            .unwrap();
            block_on(gas.step()).unwrap();
            let archive = gas.stop_recording().unwrap();
            let e = analyze_step(
                &archive.steps[0],
                0.002,
                3.,
                0.33,
                &config.gas.clone_decision,
            )
            .unwrap();
            assert!(e.exact_checks_passed, "N={n} step={}: {e:?}", e.step);
            assert!(e.kinetic_position_variance >= 0.);
        }
    }
}

#[test]
fn spread_fluctuation_bound_covers_exhaustive_matching_and_gate_laws() {
    let decision = GasConfig::einstein_hilbert(0.33, 0.002)
        .unwrap()
        .clone_decision;
    for n in 4..=7 {
        for step in [19, 20] {
            let x: Vec<_> = (0..2 * n)
                .map(|i| ((i * i + 3) as f64).sin() * (i + 1) as f64)
                .collect();
            let f: Vec<_> = (0..n).map(|i| [0.1, 0.13, 1.2, 1.2][i % 4]).collect();
            let m = matching_moments(&x, &vec![0.; 2 * n], &f, 2, &decision, step).unwrap();
            let (mut first, mut second, mut count) = (0., 0., 0.);
            permutations(&mut (0..n).collect::<Vec<_>>(), 0, &mut |order| {
                let pairs: Vec<_> = order
                    .chunks_exact(2)
                    .map(|pair| {
                        let (i, j) = if f[pair[0]] < f[pair[1]] {
                            (pair[0], pair[1])
                        } else {
                            (pair[1], pair[0])
                        };
                        (i, j, decision.acceptance_probability(step, f[i], f[j]))
                    })
                    .collect();
                for bits in 0..(1 << pairs.len()) {
                    let mut z = x.clone();
                    let mut probability = 1.;
                    for (k, &(i, j, p)) in pairs.iter().enumerate() {
                        if bits & (1 << k) != 0 {
                            probability *= p;
                            for a in 0..2 {
                                z[2 * i + a] = x[2 * j + a];
                            }
                        } else {
                            probability *= 1. - p;
                        }
                    }
                    let w = covariance(&z, &z, 2);
                    first += probability * w;
                    second += probability * w * w;
                }
                count += 1.;
            });
            let variance = (second / count - (first / count).powi(2)).max(0.);
            assert!((first / count - m.expected_variance).abs() < 1e-9);
            assert!(
                variance <= m.spread_variance_bound + 1e-8,
                "N={n} step={step}"
            );
            if step == 19 {
                assert_eq!(m.spread_variance_bound, 0.);
            }
        }
    }
}

#[test]
fn native_balanced_duplicate_clusters_have_the_computed_one_over_n_pressure() {
    use algorithmic_gas::{ObservationBatch, TensorBatch, fitness::PositiveMapping};
    let config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    for n in [8, 16, 32, 64] {
        for radius in [1_f64, 16.] {
            let m = n / 2;
            let k = m / 2;
            let d = 3;
            let mut x = vec![0.; n * d];
            for i in 0..n {
                x[i * d] = if i < m { -radius } else { radius };
            }
            let obs = ObservationBatch::positions(TensorBatch::vectors(n, d, x.clone()).unwrap());
            let geometry = config
                .geometry
                .as_ref()
                .unwrap()
                .pipeline
                .evaluate(&obs, &vec![true; n], None, n * n)
                .unwrap();
            assert_eq!(geometry.graph().edges(), n * (n - 1));
            assert!(
                geometry.curvature["ricci_scalar"]
                    .scalar
                    .iter()
                    .all(|r| r.abs() < 1e-12)
            );
            // A realizable diversity matching has k cross-cluster pairs; its
            // remaining m-k rows in each cluster are paired internally.
            let distance: Vec<_> = (0..n)
                .map(|i| if i % m < k { 2. * radius } else { 1e-30 })
                .collect();
            let (z, _) = config
                .fitness
                .diversity_standardizer
                .apply(&distance, &vec![true; n], &obs)
                .unwrap();
            let fitness: Vec<_> = z
                .into_iter()
                .map(|z| config.fitness.diversity_map.map(z).unwrap())
                .collect();
            let moments = matching_moments(
                &x,
                &vec![0.; n * d],
                &fitness,
                d,
                &config.clone_decision,
                20,
            )
            .unwrap();
            let nf = n as f64;
            let loss = (nf - 4.) / (4. * (nf - 1.) * (nf - 3.));
            assert!((moments.expected_variance / (radius * radius) - (1. - loss)).abs() < 1e-12);
            assert!(
                moments
                    .mean_center_displacement
                    .iter()
                    .all(|x| x.abs() < 1e-12)
            );
            assert_eq!(moments.signed_radial_increment, 0.);
        }
    }
}

#[test]
fn reference_first_step_matches_the_explicit_gaussian_linear_map() {
    for n in [8, 32, 128] {
        let mut config = RunConfig::einstein_hilbert().unwrap();
        config.walkers = n;
        config.gas.precision = Precision::F64;
        let mut gas = block_on(config.build::<f64>()).unwrap();
        gas.start_recording(RecordingConfig {
            graph: true,
            ..RecordingConfig::default()
        })
        .unwrap();
        block_on(gas.step()).unwrap();
        let archive = gas.stop_recording().unwrap();
        let step = &archive.steps[0];
        let field = |stage: &str, name: &str| {
            &step
                .stages
                .iter()
                .find(|s| s.stage == stage)
                .unwrap()
                .fields[name]
                .values
        };
        let w = field("O", "velocities");
        let x = field("B2", "positions");
        let v = field("B2", "velocities");
        let mw = mean(w, 3);
        let r = 1. - 0.0015 * n as f64 / (n - 1) as f64;
        for i in 0..n * 3 {
            assert!((x[i] - 0.001 * w[i]).abs() < 1e-13);
            assert!((v[i] - r * r * w[i] - (1. - r * r) * mw[i % 3]).abs() < 1e-12);
        }
    }
}
