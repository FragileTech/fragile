use algorithmic_gas::{GasBuilder, GasConfig, ObservationBatch, Population, TensorBatch};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel};

const X: [f64; 3] = [-0.5, 0., 0.5];
const JITTER: f64 = 0.1;

fn weight(x: f64, y: f64) -> f64 {
    let distance = (2. * x / (2. + x.abs()) - 2. * y / (2. + y.abs())).abs();
    (-0.5 * (distance / 2.).powi(2)).exp()
}

// Independent implementation of the canonical measured fitness. The sampled
// distance is retained; no expectation is substituted inside acceptance.
fn fitness(companions: &[u32]) -> [f64; 3] {
    let reward = X.map(|x| -0.5 * x * x);
    let distance = std::array::from_fn::<_, 3, _>(|i| {
        let x = X[i];
        let y = X[companions[i] as usize];
        let r = 2. * x / (2. + x.abs()) - 2. * y / (2. + y.abs());
        (r.powi(2) + 1e-6).sqrt()
    });
    let standardize = |values: [f64; 3]| {
        let mean = values.iter().sum::<f64>() / 3.;
        let scale = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3. + 0.01).sqrt();
        values.map(|v| (v - mean) / scale)
    };
    let reward_z = standardize(reward);
    let distance_z = standardize(distance);
    let logistic = |z: f64| 0.1 + 2. / (1. + (-z).exp());
    std::array::from_fn(|i| logistic(reward_z[i]) * logistic(distance_z[i]))
}

fn mean(v: &[f64; 3]) -> f64 {
    v.iter().sum::<f64>() / 3.
}

fn variance(v: &[f64; 3]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 3.
}

// Enumerate the complete row donor/gate law and integrate the Gaussian jitters
// analytically. These are quadrature states, not substitute gas trajectories.
fn enumerate(f: [f64; 3], alive: [bool; 3]) -> (f64, f64) {
    let mut mass = 0.;
    let mut center_second = 0.;
    let mut variance_drift = 0.;
    for donor_code in 0..27 {
        let donors = [donor_code % 3, donor_code / 3 % 3, donor_code / 9];
        if (0..3).any(|i| donors[i] == i || !alive[donors[i]]) {
            continue;
        }
        let mut donor_mass = 1.;
        let mut probability = [0.; 3];
        for i in 0..3 {
            let denominator = (0..3)
                .filter(|&j| j != i && alive[j])
                .map(|j| weight(X[i], X[j]))
                .sum::<f64>();
            donor_mass *= weight(X[i], X[donors[i]]) / denominator;
            probability[i] = if alive[i] {
                ((f[donors[i]] - f[i]) / (f[i] + 1e-6)).clamp(0., 1.)
            } else {
                1.
            };
        }
        for gate in 0..8 {
            let accepted: [bool; 3] = std::array::from_fn(|i| gate & (1 << i) != 0);
            let mut branch_mass = donor_mass;
            let output: [f64; 3] = std::array::from_fn(|i| {
                branch_mass *= if accepted[i] {
                    probability[i]
                } else {
                    1. - probability[i]
                };
                if accepted[i] { X[donors[i]] } else { X[i] }
            });
            let displacement: [f64; 3] = std::array::from_fn(|i| output[i] - X[i]);
            let count = accepted.iter().filter(|&&a| a).count() as f64;
            let jitter_variance = JITTER.powi(2) * count * 2. / 9.;
            let direct_drift = variance(&output) + jitter_variance - variance(&X);
            let mx = mean(&X);
            let md = mean(&displacement);
            // Original geometric-cluster ANOVA, with clusters {0,1} and {2}.
            let mut cluster_cross = 0.;
            for cluster in [&[0, 1][..], &[2][..]] {
                let size = cluster.len() as f64;
                let cx = cluster.iter().map(|&i| X[i]).sum::<f64>() / size;
                let cd = cluster.iter().map(|&i| displacement[i]).sum::<f64>() / size;
                cluster_cross += size * (cx - mx) * (cd - md);
                cluster_cross += cluster
                    .iter()
                    .map(|&i| (X[i] - cx) * (displacement[i] - cd))
                    .sum::<f64>();
            }
            let flux_drift = 2. * cluster_cross / 3.
                + displacement.iter().map(|d| d * d).sum::<f64>() / 3.
                - md * md
                + jitter_variance;
            assert!((direct_drift - flux_drift).abs() < 2e-16);
            // Includes barycenter noise even for a recipient that persists.
            let center_jitter = JITTER.powi(2) * (f64::from(accepted[1]) / 3. + count / 9.);
            center_second += branch_mass * ((output[1] - mean(&output)).powi(2) + center_jitter);
            variance_drift += branch_mass * direct_drift;
            mass += branch_mass;
        }
    }
    assert!((mass - 1.).abs() < 2e-15);
    (center_second, variance_drift)
}

#[test]
fn exact_donor_gate_enumeration_preserves_signed_cluster_and_barycenter_terms() {
    let f = fitness(&[1, 0, 1]);
    assert_eq!(f[0], f[2]);
    assert!(f[1] > f[0]);
    let acceptance = ((f[1] - f[0]) / (f[0] + 1e-6)).clamp(0., 1.);
    let q = weight(X[0], X[1]) / (weight(X[0], X[1]) + weight(X[0], X[2])) * acceptance;
    let prediction = (2. * X[0].powi(2) * q * (1. - q) + 2. * q * JITTER.powi(2)) / 9.;
    let (center, drift) = enumerate(f, [true; 3]);
    assert!((center - prediction).abs() < 2e-16);
    let drift_prediction = -2. * X[0].powi(2) * q * (4. - q) / 9. + 4. * q * JITTER.powi(2) / 9.;
    assert!((drift - drift_prediction).abs() < 2e-16);
    assert!(center > 0.);
    assert!(drift < 0.);
    // Retained dead coordinates participate in the full-slot identity and the
    // revived row takes its frozen donor position plus the same specified jitter.
    let (revived_center, revived_drift) = enumerate([0., 2., 1.], [false, true, true]);
    assert!(revived_center > 0. && revived_drift.is_finite());
    eprintln!(
        "exact conditional q={q}, center second moment={center}, aggregate variance drift={drift}; revival center={revived_center}, revival drift={revived_drift}"
    );
}

#[test]
fn full_rust_runs_match_conditional_persisting_center_prediction_in_disjoint_batches() {
    futures_lite::future::block_on(async {
        let event_fitness = fitness(&[1, 0, 1]);
        let (prediction, drift_prediction) = enumerate(event_fitness, [true; 3]);
        for first_seed in [120_000, 130_000] {
            let mut values = Vec::new();
            let mut drift_values = Vec::new();
            for seed in first_seed..first_seed + 4096 {
                let mut config = GasConfig::euclidean(1, 0.04).unwrap();
                config.seed = seed;
                let observations =
                    ObservationBatch::positions(TensorBatch::vectors(3, 1, X.to_vec()).unwrap());
                let mut population = Population::new(observations).unwrap();
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
                let companions = &record.report.distance_companions.indices;
                let independent_fitness = fitness(companions);
                for (expected, actual) in independent_fitness
                    .iter()
                    .zip(&record.report.pre_clone_fitness.fitness)
                {
                    assert!((expected - actual).abs() < 2e-14);
                }
                if companions[0] != 1 || companions[2] != 1 {
                    continue;
                }
                assert!(!record.report.clone_plan.choices[1].accepted);
                let stage = record
                    .stages
                    .iter()
                    .find(|s| s.stage == "post_transform")
                    .unwrap();
                let x = &stage.fields["positions"].values;
                assert_eq!(x[1], 0.);
                let center = x.iter().sum::<f64>() / 3.;
                values.push((x[1] - center).powi(2));
                drift_values
                    .push(x.iter().map(|x| (x - center).powi(2)).sum::<f64>() / 3. - variance(&X));
                // The recorded full step includes the prescribed kinetic and
                // terminal boundary stages; only the relevant cloning stage is measured.
                assert!(record.stages.iter().any(|s| s.stage == "B2"));
            }
            let n = values.len();
            assert!(n > 900);
            let estimate = values.iter().sum::<f64>() / n as f64;
            let variance =
                values.iter().map(|v| (v - estimate).powi(2)).sum::<f64>() / (n - 1) as f64;
            let se = (variance / n as f64).sqrt();
            assert!(
                (estimate - prediction).abs() < 5. * se,
                "seeds={first_seed}..{}, conditional n={n}, prediction={prediction}, measured={estimate}, SE={se}",
                first_seed + 4096
            );
            eprintln!(
                "seeds={first_seed}..{}, 4096 full Rust runs, conditional n={n}, prediction={prediction}, measured={estimate}, independent-run SE={se}",
                first_seed + 4096
            );
            let drift_estimate = drift_values.iter().sum::<f64>() / n as f64;
            let drift_variance = drift_values
                .iter()
                .map(|v| (v - drift_estimate).powi(2))
                .sum::<f64>()
                / (n - 1) as f64;
            let drift_se = (drift_variance / n as f64).sqrt();
            assert!((drift_estimate - drift_prediction).abs() < 5. * drift_se);
            assert!(drift_estimate + 5. * drift_se < 0.);
            eprintln!(
                "same conditional batch: aggregate variance drift prediction={drift_prediction}, measured={drift_estimate}, independent-run SE={drift_se}"
            );
        }
    });
}

#[test]
fn dead_leaf_switch_cancels_component_size_in_the_production_collision_operator() {
    use algorithmic_gas::cloning::apply_component_rotations;

    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let velocity_bound = 1.7;
    let mut cases = 0;
    for dimensions in [1, 2, 3] {
        // Supplied orthogonal matrices test the exact physical map. They are
        // deterministic operator inputs, not a claim about a Haar sample law.
        let rotations = match dimensions {
            1 => vec![vec![-1.], vec![1.]],
            2 => vec![vec![0., -1., 1., 0.], vec![1., 0., 0., -1.]],
            3 => vec![
                vec![0., 1., 0., 0., 0., 1., 1., 0., 0.],
                vec![-1., 0., 0., 0., 1., 0., 0., 0., -1.],
            ],
            _ => unreachable!(),
        };
        for (c, d) in [
            (2, 1),
            (2, 15),
            (15, 2),
            (32, 32),
            (2, 1022),
            (512, 512),
            (1023, 1),
        ] {
            let n = c + d;
            // Roots 0 and 1 form separate alive backbones. Every other slot is
            // a dead leaf whose retained velocity enters its revival collision.
            let mut observations = ObservationBatch::positions(
                TensorBatch::vectors(n, dimensions, vec![0.; n * dimensions]).unwrap(),
            );
            let mut velocities = Vec::with_capacity(n * dimensions);
            for i in 0..n {
                let row: Vec<_> = (0..dimensions)
                    .map(|j| ((i + 1) as f64 * 0.731 + j as f64 * 1.317).sin())
                    .collect();
                let scale = velocity_bound / norm(&row).max(1.);
                velocities.extend(row.iter().map(|v| scale * v));
            }
            observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(n, dimensions, velocities.clone()).unwrap(),
            );
            let mut before = Population::new(observations).unwrap();
            for validity in &mut before.validity[2..] {
                validity.terminated = true;
            }
            let mut left_group = vec![0];
            left_group.extend(2..c + 1);
            let mut right_group = vec![1];
            right_group.extend(c + 1..n);
            let old_groups = vec![left_group, right_group];
            let moved = 2;
            let mut new_groups = old_groups.clone();
            new_groups[0].retain(|&i| i != moved);
            new_groups[1].push(moved);
            assert_eq!(old_groups[0].len(), c);
            assert_eq!(old_groups[1].len(), d);

            let component_mean = |members: &[usize]| -> Vec<f64> {
                (0..dimensions)
                    .map(|j| {
                        members
                            .iter()
                            .map(|&i| velocities[i * dimensions + j])
                            .sum::<f64>()
                            / members.len() as f64
                    })
                    .collect()
            };
            let mu = component_mean(&old_groups[0]);
            let nu = component_mean(&old_groups[1]);
            let leaf = &velocities[moved * dimensions..(moved + 1) * dimensions];
            for restitution in [0., 0.5, 1.] {
                let mut old_output = before.clone();
                let mut new_output = before.clone();
                for after in [&mut old_output, &mut new_output] {
                    for validity in &mut after.validity[2..] {
                        validity.terminated = false;
                    }
                }
                // The same matrix follows each alive backbone when a leaf is
                // reassigned, even though both full component vertex sets change.
                apply_component_rotations(
                    &before,
                    &mut old_output,
                    &old_groups,
                    &rotations,
                    "velocities",
                    restitution,
                )
                .unwrap();
                apply_component_rotations(
                    &before,
                    &mut new_output,
                    &new_groups,
                    &rotations,
                    "velocities",
                    restitution,
                )
                .unwrap();
                let transform = |rotation: &[f64], difference: &[f64]| -> Vec<f64> {
                    (0..dimensions)
                        .map(|i| {
                            difference[i]
                                - restitution
                                    * (0..dimensions)
                                        .map(|j| rotation[i * dimensions + j] * difference[j])
                                        .sum::<f64>()
                        })
                        .collect()
                };
                let u = transform(
                    &rotations[0],
                    &leaf.iter().zip(&mu).map(|(w, m)| w - m).collect::<Vec<_>>(),
                );
                let w: Vec<_> = transform(
                    &rotations[1],
                    &leaf.iter().zip(&nu).map(|(w, m)| w - m).collect::<Vec<_>>(),
                )
                .into_iter()
                .map(|x| d as f64 / (d + 1) as f64 * x)
                .collect();
                let moved_difference: Vec<_> = u.iter().zip(&w).map(|(u, w)| u - w).collect();
                let exact_l1 = norm(&u) + norm(&w) + norm(&moved_difference);
                let old = old_output
                    .observations
                    .field("velocities")
                    .unwrap()
                    .values();
                let new = new_output
                    .observations
                    .field("velocities")
                    .unwrap()
                    .values();
                let mut actual_l1 = 0.;
                let mut momentum_difference = vec![0.; dimensions];
                for i in 0..n {
                    let difference: Vec<_> = (0..dimensions)
                        .map(|j| new[i * dimensions + j] - old[i * dimensions + j])
                        .collect();
                    let expected = if i == moved {
                        moved_difference.clone()
                    } else if old_groups[0].contains(&i) {
                        u.iter().map(|v| -v / (c - 1) as f64).collect()
                    } else {
                        w.iter().map(|v| v / d as f64).collect()
                    };
                    for j in 0..dimensions {
                        assert!((difference[j] - expected[j]).abs() < 3e-14);
                        momentum_difference[j] += difference[j];
                    }
                    actual_l1 += norm(&difference);
                }
                assert!(norm(&momentum_difference) < 2e-11);
                assert!((actual_l1 - exact_l1).abs() < 2e-11);
                assert!(actual_l1 <= (6. + 8. * restitution) * velocity_bound + 2e-11);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 63);
    eprintln!(
        "63 exact production collision comparisons: dimensions 1/2/3, restitution 0/0.5/1, N through 1024, including 1023-to-1 component load; full-slot momentum and N-independent dead-leaf switch bound passed"
    );
}

#[test]
fn one_alive_donor_giant_component_preserves_barycenter_concentration() {
    futures_lite::future::block_on(async {
        let donor_position = 0.25;
        let samples = 128;
        for (cohort, n) in [16, 32, 64, 128, 256].into_iter().enumerate() {
            let first_seed = 140_000 + 1000 * cohort as u64;
            let mut barycenters = Vec::with_capacity(samples);
            let mut largest_velocity_error: f64 = 0.;
            let velocities: Vec<_> = (0..n).map(|i| 0.9 * (0.7 * i as f64 + 0.2).cos()).collect();
            let input_velocity_mean = velocities.iter().sum::<f64>() / n as f64;
            assert!((input_velocity_mean - velocities[0]).abs() > 0.1);
            for seed in first_seed..first_seed + samples as u64 {
                let positions: Vec<_> = (0..n)
                    .map(|i| {
                        if i == 0 {
                            donor_position
                        } else if i % 2 == 0 {
                            3. + i as f64 / n as f64
                        } else {
                            -3. - i as f64 / n as f64
                        }
                    })
                    .collect();
                let mut observations =
                    ObservationBatch::positions(TensorBatch::vectors(n, 1, positions).unwrap());
                observations.fields.insert(
                    "velocities".into(),
                    TensorBatch::vectors(n, 1, velocities.clone()).unwrap(),
                );
                let mut population = Population::new(observations).unwrap();
                for validity in &mut population.validity[1..] {
                    validity.terminated = true;
                }
                let mut config = GasConfig::euclidean(1, 0.04).unwrap();
                config.seed = seed;
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
                assert_eq!(record.report.revivals, n - 1);
                // The report counts accepted living recipients separately.
                assert_eq!(record.report.clones, 0);
                assert_eq!(
                    record
                        .report
                        .pre_clone_eligible
                        .iter()
                        .filter(|&&a| a)
                        .count(),
                    1
                );
                assert!(!record.report.clone_plan.choices[0].accepted);
                for choice in &record.report.clone_plan.choices[1..] {
                    assert!(choice.accepted && choice.revival);
                    assert_eq!(choice.donors.len(), 1);
                    assert_eq!(
                        record.report.clone_plan.sources[choice.donors[0].pool_index as usize].slot,
                        0
                    );
                }
                let component_sizes = record
                    .field_evaluations
                    .iter()
                    .find(|f| {
                        f.stage == "component_collision" && f.field == "collision_component_size"
                    })
                    .unwrap();
                assert!(component_sizes.available.iter().all(|&a| a));
                assert!(component_sizes.values.iter().all(|&size| size == n as f64));
                let clone = record
                    .stages
                    .iter()
                    .find(|s| s.stage == "post_clone")
                    .unwrap();
                assert!(clone.validity.iter().all(|v| v.eligible(false)));
                let positions = &clone.fields["positions"].values;
                assert_eq!(positions[0], donor_position);
                barycenters.push(positions.iter().sum::<f64>() / n as f64);
                let output_velocity_mean =
                    clone.fields["velocities"].values.iter().sum::<f64>() / n as f64;
                largest_velocity_error =
                    largest_velocity_error.max((output_velocity_mean - input_velocity_mean).abs());
                assert!((output_velocity_mean - input_velocity_mean).abs() < 5e-14);
                assert!(record.stages.iter().any(|s| s.stage == "B2"));
            }
            let prediction = JITTER.powi(2) * (n - 1) as f64 / (n * n) as f64;
            let observed_mean = barycenters.iter().sum::<f64>() / samples as f64;
            let observed_variance = barycenters
                .iter()
                .map(|x| (x - observed_mean).powi(2))
                .sum::<f64>()
                / (samples - 1) as f64;
            // The post-cloning barycenter is exactly Gaussian: this standard
            // error follows from its fourth moment, not an estimated closure.
            let variance_se = prediction * (2. / (samples - 1) as f64).sqrt();
            let mean_se = (prediction / samples as f64).sqrt();
            assert!((observed_mean - donor_position).abs() < 6. * mean_se);
            assert!((observed_variance - prediction).abs() < 6. * variance_se);
            eprintln!(
                "one alive donor: N={n}, seeds={first_seed}..{}, post-clone position barycenter mean={observed_mean}, variance prediction={prediction}, measured={observed_variance}, exact Gaussian variance SE={variance_se}, maximum full-slot velocity mean error={largest_velocity_error}",
                first_seed + samples as u64
            );
        }
    });
}
