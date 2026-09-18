use algorithmic_gas::{GasBuilder, GasConfig, ObservationBatch, Population, TensorBatch};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel};

const JITTER: f64 = 0.1;

fn donor_weights(x: &[f64; 3]) -> [[f64; 3]; 3] {
    let feature = x.map(|v| 2. * v / (2. + v.abs()));
    std::array::from_fn(|i| {
        let raw: [f64; 3] = std::array::from_fn(|j| {
            if i == j {
                0.
            } else {
                (-(feature[i] - feature[j]).powi(2) / 8.).exp()
            }
        });
        let z = raw.iter().sum::<f64>();
        raw.map(|w| w / z)
    })
}

fn measured_fitness(x: &[f64; 3], companions: &[usize; 3]) -> [f64; 3] {
    let standardize = |values: [f64; 3]| {
        let mean = values.iter().sum::<f64>() / 3.;
        let scale = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3. + 0.01).sqrt();
        values.map(|v| (v - mean) / scale)
    };
    let feature = x.map(|v| 2. * v / (2. + v.abs()));
    let distance =
        std::array::from_fn(|i| ((feature[i] - feature[companions[i]]).powi(2) + 1e-6).sqrt());
    let reward_z = standardize(x.map(|v| -0.5 * v * v));
    let diversity_z = standardize(distance);
    let logistic = |z: f64| 0.1 + 2. / (1. + (-z).exp());
    std::array::from_fn(|i| logistic(reward_z[i]) * logistic(diversity_z[i]))
}

fn choices(code: usize) -> [usize; 3] {
    std::array::from_fn(|i| {
        let bit = (code >> i) & 1;
        if bit >= i { bit + 1 } else { bit }
    })
}

#[derive(Debug)]
struct Prediction {
    drift: f64,
    variance: f64,
    activity: f64,
    conditional_variance: f64,
    measurement_variance: f64,
}

// Independent integration of the finite transition, not an alternative engine.
// Sum every measurement, donor and gate configuration; integrate Gaussian jitter
// through its second/fourth moments. Shared rotations do not change positions.
fn integrate(a: f64) -> Prediction {
    let x = [-a, 0., a];
    let weights = donor_weights(&x);
    let entering_variance = 2. * a * a / 3.;
    let mut mass = 0.;
    let mut drift = 0.;
    let mut second = 0.;
    let mut activity = 0.;
    let mut conditional_variance = 0.;
    let mut measurement_second = 0.;
    for measurement_code in 0..8 {
        let measurement = choices(measurement_code);
        let measurement_mass = (0..3).map(|i| weights[i][measurement[i]]).product::<f64>();
        let fitness = measured_fitness(&x, &measurement);
        let accepted: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                weights[i][j] * ((fitness[j] - fitness[i]) / (fitness[i] + 1e-6)).clamp(0., 1.)
            })
        });
        let t: [f64; 3] =
            std::array::from_fn(|i| (0..3).map(|j| accepted[i][j] * (x[j] - x[i])).sum());
        let copy_variance: [f64; 3] = std::array::from_fn(|i| {
            (0..3)
                .map(|j| accepted[i][j] * (x[j] - x[i]).powi(2))
                .sum::<f64>()
                - t[i].powi(2)
        });
        let mean_activity = accepted.iter().flatten().sum::<f64>() / 3.;
        let signed_flux = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| accepted[i][j] * (x[j].powi(2) - x[i].powi(2)))
            .sum::<f64>()
            / 3.;
        let row_prediction = signed_flux
            - (t.iter().sum::<f64>() / 3.).powi(2)
            - copy_variance.iter().sum::<f64>() / 9.
            + 2. / 3. * JITTER.powi(2) * mean_activity;
        let mut conditional_mass = 0.;
        let mut conditional_drift = 0.;
        let mut conditional_second = 0.;
        for donor_code in 0..8 {
            let donor = choices(donor_code);
            let donor_mass = (0..3).map(|i| weights[i][donor[i]]).product::<f64>();
            let p: [f64; 3] = std::array::from_fn(|i| {
                ((fitness[donor[i]] - fitness[i]) / (fitness[i] + 1e-6)).clamp(0., 1.)
            });
            for gate in 0..8 {
                let copied: [bool; 3] = std::array::from_fn(|i| gate & (1 << i) != 0);
                let branch_mass = donor_mass
                    * (0..3)
                        .map(|i| if copied[i] { p[i] } else { 1. - p[i] })
                        .product::<f64>();
                let y: [f64; 3] =
                    std::array::from_fn(|i| if copied[i] { x[donor[i]] } else { x[i] });
                let mean = y.iter().sum::<f64>() / 3.;
                let centered = y.map(|v| v - mean);
                let count = copied.iter().filter(|&&c| c).count() as f64;
                let mean_drift = centered.iter().map(|v| v * v).sum::<f64>() / 3.
                    + 2. * JITTER.powi(2) * count / 9.
                    - entering_variance;
                // For Q = Y^T P Y / N: Var Q = [2 tr(P Sigma P Sigma)
                // + 4 (P y)^T Sigma (P y)] / N^2, P=I-11^T/N.
                let trace_square = JITTER.powi(4) * (count / 3. + count * count / 9.);
                let linear_variance = JITTER.powi(2)
                    * (0..3)
                        .filter(|&i| copied[i])
                        .map(|i| centered[i].powi(2))
                        .sum::<f64>();
                let jitter_variance = (2. * trace_square + 4. * linear_variance) / 9.;
                conditional_mass += branch_mass;
                conditional_drift += branch_mass * mean_drift;
                conditional_second += branch_mass * (mean_drift.powi(2) + jitter_variance);
            }
        }
        assert!((conditional_mass - 1.).abs() < 2e-14);
        assert!((conditional_drift - row_prediction).abs() < 2e-14);
        assert!(conditional_second + 1e-14 >= conditional_drift.powi(2));
        mass += measurement_mass;
        drift += measurement_mass * conditional_drift;
        second += measurement_mass * conditional_second;
        measurement_second += measurement_mass * conditional_drift.powi(2);
        conditional_variance += measurement_mass * (conditional_second - conditional_drift.powi(2));
        activity += measurement_mass * mean_activity;
    }
    assert!((mass - 1.).abs() < 2e-14);
    let variance = second - drift.powi(2);
    let measurement_variance = measurement_second - drift.powi(2);
    assert!((variance - conditional_variance - measurement_variance).abs() < 2e-14);
    Prediction {
        drift,
        variance,
        activity,
        conditional_variance,
        measurement_variance,
    }
}

#[test]
fn exact_measurement_average_retains_signed_flux_and_total_variance() {
    for a in [0.1, 0.5, 1.9] {
        let p = integrate(a);
        assert!(p.variance > 0. && p.conditional_variance > 0. && p.measurement_variance > 0.);
        assert!(p.activity > 0.);
        assert_eq!(p.drift > 0., a == 0.1);
        eprintln!("complete measurement integration: a={a}, {p:?}");
    }
}

#[test]
fn full_engine_matches_unconditional_measurement_predictions_in_disjoint_batches() {
    futures_lite::future::block_on(async {
        for (case, a) in [0.1, 0.5, 1.9].into_iter().enumerate() {
            let prediction = integrate(a);
            for batch in 0..2 {
                let first_seed = 300_000 + 20_000 * case as u64 + 5_000 * batch;
                let samples = 2048;
                let mut drifts = Vec::with_capacity(samples);
                let mut activity = 0.;
                for seed in first_seed..first_seed + samples as u64 {
                    let mut config = GasConfig::euclidean(1, 0.04).unwrap();
                    config.seed = seed;
                    let x = [-a, 0., a];
                    let mut population = Population::new(ObservationBatch::positions(
                        TensorBatch::vectors(3, 1, x.to_vec()).unwrap(),
                    ))
                    .unwrap();
                    population.observations.fields.insert(
                        "velocities".into(),
                        TensorBatch::vectors(3, 1, vec![0.; 3]).unwrap(),
                    );
                    let model = BenchmarkModel {
                        benchmark: Benchmark::Quadratic,
                        field: "positions".into(),
                        direction: config.fitness.direction,
                    };
                    let mut gas = GasBuilder::new(population, model.clone())
                        .gradient(model)
                        .config(config)
                        .build()
                        .await
                        .unwrap();
                    gas.start_recording(Default::default()).unwrap();
                    gas.step().await.unwrap();
                    let record = &gas.recording().unwrap().steps[0];
                    let measurement = std::array::from_fn(|i| {
                        record.report.distance_companions.indices[i] as usize
                    });
                    for (actual, expected) in record
                        .report
                        .pre_clone_fitness
                        .fitness
                        .iter()
                        .zip(measured_fitness(&x, &measurement))
                    {
                        assert!((actual - expected).abs() < 2e-14);
                    }
                    let clone = record
                        .stages
                        .iter()
                        .find(|s| s.stage == "post_clone")
                        .unwrap();
                    let output = &clone.fields["positions"].values;
                    let mean = output.iter().sum::<f64>() / 3.;
                    drifts.push(
                        output.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3.
                            - 2. * a * a / 3.,
                    );
                    activity += record
                        .report
                        .clone_plan
                        .choices
                        .iter()
                        .filter(|c| c.accepted)
                        .count() as f64
                        / 3.;
                    assert!(record.stages.iter().any(|s| s.stage == "B2"));
                }
                let observed = drifts.iter().sum::<f64>() / samples as f64;
                let observed_variance = drifts.iter().map(|v| (v - observed).powi(2)).sum::<f64>()
                    / (samples - 1) as f64;
                let exact_se = (prediction.variance / samples as f64).sqrt();
                assert!((observed - prediction.drift).abs() < 6. * exact_se);
                let mean_activity = activity / samples as f64;
                assert!(
                    (mean_activity - prediction.activity).abs()
                        < 6. * (0.25 / samples as f64).sqrt()
                );
                eprintln!(
                    "full measurement Rust cohort: a={a}, seeds={first_seed}..{}, n={samples}, predicted drift={}, measured={observed}, exact SE={exact_se}, predicted variance={}, measured variance={observed_variance}, predicted activity={}, measured activity={mean_activity}",
                    first_seed + samples as u64,
                    prediction.drift,
                    prediction.variance,
                    prediction.activity
                );
            }
        }
    });
}

// Exact integration over the two unchanged geometric clusters. The binomial
// counts aggregate independent measurement marks, not collision components.
fn balanced_prediction(n: usize, a: f64) -> (f64, f64, f64) {
    let m = n / 2;
    let radius = 2. * a / (2. + a);
    let w = (-radius * radius / 2.).exp();
    let z = (m - 1) as f64 + m as f64 * w;
    let q = m as f64 * w / z;
    let delta = (4. * radius * radius + 1e-6).sqrt() - 0.001;
    let mut probabilities = vec![(1. - q).powi(m as i32)];
    for k in 0..m {
        probabilities.push(probabilities[k] * (m - k) as f64 / (k + 1) as f64 * q / (1. - q));
    }
    assert!((probabilities.iter().sum::<f64>() - 1.).abs() < 2e-14);
    let mut drift = 0.;
    let mut activity = 0.;
    for kp in 0..=m {
        for km in 0..=m {
            let fraction = (kp + km) as f64 / n as f64;
            let scale = (fraction * (1. - fraction) * delta * delta + 0.01).sqrt();
            let g = |v: f64| 0.1 + 2. / (1. + (-v).exp());
            let low = 1.1 * g(-fraction * delta / scale);
            let high = 1.1 * g((1. - fraction) * delta / scale);
            let acceptance = ((high - low) / (low + 1e-6)).clamp(0., 1.);
            let u_plus = acceptance * w * kp as f64 / z;
            let u_minus = acceptance * w * km as f64 / z;
            let barycenter = a * acceptance * w / z * (kp as f64 - km as f64);
            let copy_variance = 4.
                * a
                * a
                * ((m - kp) as f64 * u_minus * (1. - u_minus)
                    + (m - km) as f64 * u_plus * (1. - u_plus));
            let p = acceptance
                * ((m - kp) as f64 * (kp as f64 + w * km as f64)
                    + (m - km) as f64 * (km as f64 + w * kp as f64))
                / (n as f64 * z);
            let change = -barycenter.powi(2) - copy_variance / (n * n) as f64
                + (1. - 1. / n as f64) * JITTER.powi(2) * p;
            drift += probabilities[kp] * probabilities[km] * change;
            activity += probabilities[kp] * probabilities[km] * p;
        }
    }
    let minimum_acceptance =
        (1.1 * (delta / (2. * (delta * delta / 4. + 0.01).sqrt())).tanh() / 1.210001).min(1.);
    let lower = JITTER.powi(2) * (1. - 1. / n as f64) * minimum_acceptance * q * (1. - q)
        - 4. * a * a * q * q * (1. - q * q) / n as f64;
    assert!(drift >= lower - 2e-14);
    (drift, activity, lower)
}

#[test]
fn full_engine_matches_two_cluster_average_with_unequal_retained_fitness() {
    futures_lite::future::block_on(async {
        for (case, n) in [16, 32, 64, 128, 256].into_iter().enumerate() {
            let a = 0.5;
            let (prediction, predicted_activity, lower) = balanced_prediction(n, a);
            if n >= 128 {
                assert!(lower > 0.);
            }
            for batch in 0..2 {
                let first_seed = 400_000 + 1000 * case as u64 + 200 * batch;
                let samples = 128;
                let mut values = Vec::with_capacity(samples);
                let mut activity = 0.;
                let mut equal_fitness = 0;
                for seed in first_seed..first_seed + samples as u64 {
                    let mut config = GasConfig::euclidean(1, 0.04).unwrap();
                    config.seed = seed;
                    let positions = (0..n).map(|i| if i < n / 2 { a } else { -a }).collect();
                    let mut population = Population::new(ObservationBatch::positions(
                        TensorBatch::vectors(n, 1, positions).unwrap(),
                    ))
                    .unwrap();
                    population.observations.fields.insert(
                        "velocities".into(),
                        TensorBatch::vectors(n, 1, vec![0.; n]).unwrap(),
                    );
                    let model = BenchmarkModel {
                        benchmark: Benchmark::Quadratic,
                        field: "positions".into(),
                        direction: config.fitness.direction,
                    };
                    let mut gas = GasBuilder::new(population, model.clone())
                        .gradient(model)
                        .config(config)
                        .build()
                        .await
                        .unwrap();
                    gas.start_recording(Default::default()).unwrap();
                    gas.step().await.unwrap();
                    let record = &gas.recording().unwrap().steps[0];
                    let f = &record.report.pre_clone_fitness.fitness;
                    equal_fitness += usize::from(f.iter().all(|v| (v - f[0]).abs() < 1e-12));
                    let clone = record
                        .stages
                        .iter()
                        .find(|s| s.stage == "post_clone")
                        .unwrap();
                    let x = &clone.fields["positions"].values;
                    let mean = x.iter().sum::<f64>() / n as f64;
                    values
                        .push(x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64 - a * a);
                    activity += record
                        .report
                        .clone_plan
                        .choices
                        .iter()
                        .filter(|c| c.accepted)
                        .count() as f64
                        / n as f64;
                    assert!(clone.fields["velocities"].values.iter().all(|&v| v == 0.));
                    assert!(record.stages.iter().any(|s| s.stage == "B2"));
                }
                let mean = values.iter().sum::<f64>() / samples as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (samples - 1) as f64;
                let se = (variance / samples as f64).sqrt();
                assert!((mean - prediction).abs() < 6. * se);
                assert_eq!(equal_fitness, 0);
                eprintln!(
                    "two-cluster Rust cohort: N={n}, a={a}, seeds={first_seed}..{}, n={samples}, predicted drift={prediction}, lower bound={lower}, measured drift={mean}, empirical SE={se}, predicted activity={predicted_activity}, measured activity={}, equal-fitness runs={equal_fitness}",
                    first_seed + samples as u64,
                    activity / samples as f64
                );
            }
        }
    });
}

#[test]
fn integrated_balanced_families_satisfy_structural_keystone_without_an_offset() {
    let a_min: f64 = 0.5;
    let feature_gap = 4. * a_min / (2. + a_min);
    let delta = (feature_gap.powi(2) + 1e-6).sqrt() - 0.001;
    let a_zero =
        (1.1 * (delta / (2. * (delta.powi(2) / 4. + 0.01).sqrt())).tanh() / 1.210001).min(1.);
    let chi = 4. * a_zero / 9.;
    for n in [4, 10, 16, 32, 64, 128, 256] {
        for radius in [0.5, 0.75, 1., 1.9] {
            let (_, p_a, _) = balanced_prediction(n, a_min);
            let (_, p_b, _) = balanced_prediction(n, radius);
            // Same-sign matching is optimal for these centered one-dimensional
            // empirical laws, and every input velocity is zero. Thus this is
            // actual structural discrepancy, not either cloud's variance.
            let structural_error = (radius - a_min).powi(2);
            let pressure = (p_a + p_b) * structural_error;
            assert!(pressure + 2e-14 >= chi * structural_error);
            assert!(p_a >= 2. * a_zero / 9. && p_b >= 2. * a_zero / 9.);
        }
    }
    eprintln!(
        "balanced structural Keystone integration: chi={chi}, zero offset, N=4,10,16,32,64,128,256, radii=0.5,0.75,1,1.9"
    );
}
