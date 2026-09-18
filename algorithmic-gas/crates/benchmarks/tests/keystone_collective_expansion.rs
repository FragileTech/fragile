use algorithmic_gas::{GasBuilder, GasConfig, ObservationBatch, Population, TensorBatch};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel};

fn binomial(n: usize, p: f64) -> Vec<f64> {
    // Polynomial multiplication also handles p=1 (the singleton donor group).
    let mut law = vec![1.];
    for _ in 0..n {
        let mut next = vec![0.; law.len() + 1];
        for (k, mass) in law.into_iter().enumerate() {
            next[k] += mass * (1. - p);
            next[k + 1] += mass * p;
        }
        law = next;
    }
    law
}

#[derive(Clone, Copy, Debug, Default)]
struct Prediction {
    variance: f64,
    signed_flux: f64,
    activity: f64,
    unequal_fitness: f64,
}

// Finite-population integration of the actual marked cloning kernel. Two
// geometric locations and their retained near/far measurement classes suffice;
// this aggregates exchangeable marks, without changing the sampling operator.
fn prediction(n: usize, scale: f64) -> Prediction {
    let counts = [3 * n / 4, n / 4];
    let x = [-1.5 * scale, scale];
    let feature = x.map(|v| 2. * v / (2. + v.abs()));
    let delta = ((feature[0] - feature[1]).powi(2) + 1e-6).sqrt() - 0.001;
    let w = (-(feature[0] - feature[1]).powi(2) / 8.).exp();
    let z = std::array::from_fn::<_, 2, _>(|g| (counts[g] - 1) as f64 + counts[1 - g] as f64 * w);
    let q = std::array::from_fn::<_, 2, _>(|g| counts[1 - g] as f64 * w / z[g]);
    let law = [binomial(counts[0], q[0]), binomial(counts[1], q[1])];
    let reward = x.map(|v| -0.5 * v * v);
    let reward_mean = (3. * reward[0] + reward[1]) / 4.;
    let reward_scale = (0.75 * (reward[0] - reward_mean).powi(2)
        + 0.25 * (reward[1] - reward_mean).powi(2)
        + 0.01)
        .sqrt();
    let g = |v: f64| 0.1 + 2. / (1. + (-v).exp());
    let a = reward.map(|v| g((v - reward_mean) / reward_scale));
    let entering_mean = (3. * x[0] + x[1]) / 4.;
    let entering_variance = 3. / 16. * (x[0] - x[1]).powi(2);
    let mut out = Prediction::default();
    let mut mass = 0.;
    for (k0, &p0) in law[0].iter().enumerate() {
        for (k1, &p1) in law[1].iter().enumerate() {
            let probability = p0 * p1;
            if probability == 0. {
                continue;
            }
            mass += probability;
            let classes = [counts[0] - k0, k0, counts[1] - k1, k1];
            let far_fraction = (k0 + k1) as f64 / n as f64;
            let s = (delta * delta * far_fraction * (1. - far_fraction) + 0.01).sqrt();
            let fitness = std::array::from_fn::<_, 4, _>(|r| {
                a[r / 2] * g((r as f64 % 2. - far_fraction) * delta / s)
            });
            let mut t_mean = 0.;
            let mut copy_variance = 0.;
            let mut signed_flux = 0.;
            let mut activity = 0.;
            let mut min_fitness = f64::INFINITY;
            let mut max_fitness = f64::NEG_INFINITY;
            for r in 0..4 {
                if classes[r] == 0 {
                    continue;
                }
                min_fitness = min_fitness.min(fitness[r]);
                max_fitness = max_fitness.max(fitness[r]);
                let mut first = 0.;
                let mut second = 0.;
                let mut row_flux = 0.;
                let mut row_activity = 0.;
                for donor in 0..4 {
                    let eligible = classes[donor] - usize::from(r == donor);
                    let weight = if r / 2 == donor / 2 { 1. } else { w };
                    let gate = ((fitness[donor] - fitness[r]) / (fitness[r] + 1e-6)).clamp(0., 1.);
                    let accepted = eligible as f64 * weight / z[r / 2] * gate;
                    let displacement = x[donor / 2] - x[r / 2];
                    first += accepted * displacement;
                    second += accepted * displacement * displacement;
                    row_flux += accepted
                        * ((x[donor / 2] - entering_mean).powi(2)
                            - (x[r / 2] - entering_mean).powi(2));
                    row_activity += accepted;
                }
                let fraction = classes[r] as f64 / n as f64;
                t_mean += fraction * first;
                copy_variance += fraction * (second - first * first) / n as f64;
                signed_flux += fraction * row_flux;
                activity += fraction * row_activity;
            }
            let variance = entering_variance + signed_flux - t_mean * t_mean - copy_variance
                + (1. - 1. / n as f64) * 0.01 * activity;
            out.variance += probability * variance;
            out.signed_flux += probability * signed_flux;
            out.activity += probability * activity;
            out.unequal_fitness += probability * f64::from(max_fitness > min_fitness);
        }
    }
    assert!((mass - 1.).abs() < 1e-13);
    out
}

#[test]
fn finite_mark_integration_proves_coupling_independent_structural_expansion() {
    let left = prediction(4, 1.);
    let right = prediction(4, 0.01);
    assert!((left.variance - 1.3195446396980208).abs() < 2e-14);
    assert!((right.variance - 0.0003571182448213061).abs() < 2e-16);
    let entering = 75. / 64. * 0.99_f64.powi(2);
    let lower = (left.variance.sqrt() - right.variance.sqrt()).powi(2);
    assert!(lower - entering > 0.1279);
    assert!((left.unequal_fitness - 1.).abs() < 1e-14);
    assert!((right.unequal_fitness - 1.).abs() < 1e-14);
    eprintln!(
        "exact structural lower bound: entering={entering}, post_clone_lower={lower}, increment_lower={}, left={left:?}, right={right:?}",
        lower - entering
    );
}

#[test]
fn rust_validates_collective_expansion_on_disjoint_population_batches() {
    futures_lite::future::block_on(async {
        for (case, n) in [4, 16, 32, 64, 128, 256].into_iter().enumerate() {
            for (which, scale) in [1., 0.01].into_iter().enumerate() {
                let expected = prediction(n, scale);
                for batch in 0..2_u64 {
                    let first_seed =
                        800_000 + 20_000 * case as u64 + 5_000 * which as u64 + 1_000 * batch;
                    let mut variances = Vec::new();
                    let mut full_step_variances = Vec::new();
                    let mut preboundary_variances = Vec::new();
                    let mut activity = 0.;
                    let mut unequal = 0;
                    for seed in first_seed..first_seed + 128 {
                        let mut config = GasConfig::euclidean(1, 0.04).unwrap();
                        config.seed = seed;
                        let x: Vec<f64> = (0..n)
                            .map(|i| if i < 3 * n / 4 { -1.5 * scale } else { scale })
                            .collect();
                        let mut population = Population::new(ObservationBatch::positions(
                            TensorBatch::vectors(n, 1, x).unwrap(),
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
                        let clone = record
                            .stages
                            .iter()
                            .find(|s| s.stage == "post_clone")
                            .unwrap();
                        let output = &clone.fields["positions"].values;
                        let mean = output.iter().sum::<f64>() / n as f64;
                        variances.push(
                            output.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64,
                        );
                        assert!(clone.fields["velocities"].values.iter().all(|v| *v == 0.));
                        let fitness = &record.report.pre_clone_fitness.fitness;
                        unequal += usize::from(fitness.iter().any(|f| *f != fitness[0]));
                        activity += record
                            .report
                            .clone_plan
                            .choices
                            .iter()
                            .filter(|c| c.accepted)
                            .count() as f64
                            / n as f64;
                        assert!(record.stages.iter().any(|s| s.stage == "B2"));
                        let kinetic = record
                            .stages
                            .iter()
                            .find(|s| s.stage == "post_kinetic")
                            .unwrap();
                        let preboundary = &kinetic.fields["positions"].values;
                        let pre_mean = preboundary.iter().sum::<f64>() / n as f64;
                        preboundary_variances.push(
                            preboundary
                                .iter()
                                .map(|v| (v - pre_mean).powi(2))
                                .sum::<f64>()
                                / n as f64,
                        );
                        let final_population = gas.population();
                        let positions = final_population.observations.fields["positions"].values();
                        let eligible = final_population.eligible(false);
                        let alive: Vec<f64> = positions
                            .iter()
                            .zip(eligible)
                            .filter_map(|(&x, alive)| alive.then_some(x))
                            .collect();
                        if !alive.is_empty() {
                            let center = alive.iter().sum::<f64>() / alive.len() as f64;
                            full_step_variances.push(
                                alive.iter().map(|v| (v - center).powi(2)).sum::<f64>()
                                    / alive.len() as f64,
                            );
                        }
                    }
                    let mean = variances.iter().sum::<f64>() / 128.;
                    let se =
                        (variances.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 127. / 128.)
                            .sqrt();
                    assert!((mean - expected.variance).abs() <= 6. * se + 1e-13);
                    assert_eq!(unequal, 128);
                    // Independent full BAOAB Gaussian moment integration, before
                    // terminal eligibility changes the empirical normalization.
                    let t = 1. - 0.02_f64.powi(2) * (1. + (-0.04_f64).exp());
                    let nu =
                        0.02_f64.powi(2) * (1. - (-0.08_f64).exp()) / 2. + 0.1_f64.powi(2) * 0.04;
                    let full_prediction = t * t * expected.variance + (1. - 1. / n as f64) * nu;
                    let pre_mean = preboundary_variances.iter().sum::<f64>() / 128.;
                    let pre_se = (preboundary_variances
                        .iter()
                        .map(|v| (v - pre_mean).powi(2))
                        .sum::<f64>()
                        / 127.
                        / 128.)
                        .sqrt();
                    assert!((pre_mean - full_prediction).abs() <= 6. * pre_se + 1e-13);
                    eprintln!(
                        "full-kinetic cohort: N={n}, scale={scale}, seeds={first_seed}..{}, predicted_preboundary_variance={full_prediction}, measured_preboundary_variance={pre_mean}, SE={pre_se}",
                        first_seed + 128
                    );
                    eprintln!(
                        "collective-expansion cohort: N={n}, scale={scale}, seeds={first_seed}..{}, runs=128, predicted_variance={}, measured_variance={mean}, SE={se}, predicted_activity={}, measured_activity={}, full_step_alive_variance={}",
                        first_seed + 128,
                        expected.variance,
                        expected.activity,
                        activity / 128.,
                        full_step_variances.iter().sum::<f64>() / full_step_variances.len() as f64
                    );
                }
            }
        }
    });
}
