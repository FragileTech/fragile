use algorithmic_gas::{GasBuilder, GasConfig, ObservationBatch, Population, TensorBatch};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel};

// Independent integration of the eight distinct measurement-distance patterns.
// Integrates donor choices, gates and Gaussian jitter by their first two moments.
fn exact_position_variance(jitter: f64) -> f64 {
    let x: [f64; 4] = [0., 0., 0., 0.1];
    let distance = 2. * x[3] / (2. + x[3]);
    let weight = (-0.5 * (distance / 2.).powi(2)).exp();
    let rewards = x.map(|x| 0.5 * x * x);
    let reward_mean = rewards.iter().sum::<f64>() / 4.;
    let reward_sd = (rewards
        .iter()
        .map(|r| (r - reward_mean).powi(2))
        .sum::<f64>()
        / 4.
        + 0.01)
        .sqrt();
    let logistic = |z: f64| 0.1 + 2. / (1. + (-z).exp());
    let mut result = 0.;
    for mask in 0..8 {
        let mut distances = [(distance * distance + 1e-6).sqrt(); 4];
        let mut mass = 1.;
        for (i, value) in distances.iter_mut().enumerate().take(3) {
            if mask & (1 << i) == 0 {
                *value = 0.001;
                mass *= 2. / (2. + weight);
            } else {
                mass *= weight / (2. + weight);
            }
        }
        let dm = distances.iter().sum::<f64>() / 4.;
        let ds = (distances.iter().map(|v| (v - dm).powi(2)).sum::<f64>() / 4. + 0.01).sqrt();
        let fitness: Vec<_> = (0..4)
            .map(|i| {
                logistic((reward_mean - rewards[i]) / reward_sd)
                    * logistic((distances[i] - dm) / ds)
            })
            .collect();
        let mut means = x;
        let mut seconds = x.map(|v| v * v);
        for i in 0..4 {
            let denominator = (0..4)
                .filter(|&j| j != i)
                .map(|j| if x[i] == x[j] { 1. } else { weight })
                .sum::<f64>();
            for j in 0..4 {
                if i == j {
                    continue;
                }
                let w = if x[i] == x[j] { 1. } else { weight };
                let p = w / denominator
                    * ((fitness[j] - fitness[i]) / (fitness[i] + 1e-6)).clamp(0., 1.);
                means[i] += p * (x[j] - x[i]);
                seconds[i] += p * (x[j] * x[j] + jitter * jitter - x[i] * x[i]);
            }
        }
        let sum_second = seconds.iter().sum::<f64>();
        let expected = sum_second / 4.
            - (sum_second - means.iter().map(|m| m * m).sum::<f64>()
                + means.iter().sum::<f64>().powi(2))
                / 16.;
        result += mass * expected;
    }
    result
}
#[test]
fn exact_frozen_cloning_variance_matches_real_engine_and_exhibits_positive_drift() {
    let prediction = exact_position_variance(0.1);
    assert!((exact_position_variance(0.) - 0.002009271535689).abs() < 1e-13);
    assert!((prediction - 0.00297011744615).abs() < 1e-13);
    assert!(prediction > 0.001875);
    futures_lite::future::block_on(async {
        let mut values = Vec::new();
        for seed in 0..4096 {
            let mut config = GasConfig::euclidean(1, 0.04).unwrap();
            config.seed = seed + 50000;
            let observations = ObservationBatch::positions(
                TensorBatch::vectors(4, 1, vec![0., 0., 0., 0.1]).unwrap(),
            );
            let mut population = Population::new(observations).unwrap();
            population.observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(4, 1, vec![0.; 4]).unwrap(),
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
            let stage = gas.recording().unwrap().steps[0]
                .stages
                .iter()
                .find(|s| s.stage == "post_transform")
                .unwrap();
            let x = &stage.fields["positions"].values;
            let m = x.iter().sum::<f64>() / 4.;
            values.push(x.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 4.);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let se = (variance / values.len() as f64).sqrt();
        assert!(
            (mean - prediction).abs() < 5. * se,
            "prediction {prediction}, actual {mean}, SE {se}"
        );
        eprintln!(
            "Exact clone variance={prediction}; actual Rust mean={mean}; independent-run SE={se}"
        );
    });
}
