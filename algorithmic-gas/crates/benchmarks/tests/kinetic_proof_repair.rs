use algorithmic_gas::{
    BackendKind, ExecutionContext, GasConfig, ObservationBatch, Population, Precision, TensorBatch,
    boundary::BoundaryPolicy, domain::NumericalDomain, fitness::ObjectiveDirection,
    kinetic::KineticContext,
};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel, RunConfig};

fn population(n: usize, d: usize, shifted: bool, c: f64) -> Population<f64> {
    let mut x = Vec::with_capacity(n * d);
    let mut v = Vec::with_capacity(n * d);
    for i in 0..n {
        for j in 0..d {
            let base = ((i * d + j) as f64 * 0.173).sin();
            let shift = if shifted { 0.1 } else { 0. };
            x.push(base + shift);
            // Differences lie in the undamped direction of uncapped BAOAB.
            v.push(0.2 * base + c * shift);
        }
    }
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, d, x).unwrap());
    obs.fields
        .insert("velocities".into(), TensorBatch::vectors(n, d, v).unwrap());
    Population::new(obs).unwrap()
}

#[test]
fn exact_baoab_metric_and_cap_gap_have_no_additive_offset() {
    for h in [0.001_f64, 0.04, 0.5, 1., 1.9] {
        let c = h / 2.;
        let k = 1. - c * c;
        let a = (-h).exp();
        let matrix = [
            [1. - c * c * (1. + a), c * (1. + a)],
            [-c * (1. + a) * k, a - c * c * (1. + a)],
        ];
        let w = [-c, 1.];
        for i in 0..2 {
            for j in 0..2 {
                let lhs = k * matrix[0][i] * matrix[0][j] + matrix[1][i] * matrix[1][j];
                let rhs = if i == j { [k, 1.][i] } else { 0. } - k * (1. - a * a) * w[i] * w[j];
                assert!((lhs - rhs).abs() < 2e-15);
            }
        }
        let q = ((1. - a * a) / 2.).sqrt();
        let eta = (2. / std::f64::consts::PI).sqrt()
            * (-2_f64).exp()
            * (1. - (2. / (2. + k * q)).powi(2));
        let trace = 1. - a * a + eta * (a * a + c * c * (1. - a * a));
        let determinant = eta * c * c * (1. - a * a);
        let delta = determinant / trace;
        assert!(delta > 0. && delta < 1.);
        if h == 0.04 {
            assert!(delta > 6.03e-6 && delta < 6.04e-6);
        }
    }
}

#[test]
fn actual_rust_kinetic_stage_obeys_the_exact_synchronous_coupling() {
    futures_lite::future::block_on(async {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        let gradient = BenchmarkModel {
            benchmark: Benchmark::Quadratic,
            field: "positions".into(),
            direction: ObjectiveDirection::Minimize,
        };
        for d in [1, 2, 3] {
            for h in [0.04_f64, 0.5, 1.5] {
                let n = 128;
                let c = h / 2.;
                let k = 1. - c * c;
                let a = (-h).exp();
                let q = ((1. - a * a) / 2.).sqrt();
                let eta = (2. / std::f64::consts::PI).sqrt()
                    * (-2_f64).exp()
                    * (1. - (2. / (2. + k * q)).powi(2));
                let delta = eta * c * c * (1. - a * a)
                    / (1. - a * a + eta * (a * a + c * c * (1. - a * a)));
                let input_cost = d as f64 * (k * 0.1_f64.powi(2) + (c * 0.1).powi(2));
                let config = GasConfig::euclidean(d, h).unwrap();
                let mut total = 0.;
                for seed in 80_000..80_128 {
                    let mut left = population(n, d, false, c);
                    let mut right = population(n, d, true, c);
                    for p in [&mut left, &mut right] {
                        config
                            .kinetic
                            .advance(
                                p,
                                KineticContext {
                                    gradient: Some(&gradient),
                                    domain: &NumericalDomain,
                                    // Inspect the kinetic stage before terminal marking.
                                    boundary: &BoundaryPolicy::Unbounded,
                                    include_truncated: false,
                                    seed,
                                    step: 0,
                                    operators: None,
                                    graph: None,
                                    frozen_fitness: None,
                                },
                                &mut cx,
                            )
                            .await
                            .unwrap();
                    }
                    let lx = left.observations.field("positions").unwrap().values();
                    let rx = right.observations.field("positions").unwrap().values();
                    let lv = left.observations.field("velocities").unwrap().values();
                    let rv = right.observations.field("velocities").unwrap().values();
                    for i in 0..n * d {
                        // A(1,c)=(1,-c), exactly; position noise cancels.
                        assert!((rx[i] - lx[i] - 0.1).abs() < 3e-14);
                        total += k * (rx[i] - lx[i]).powi(2) + (rv[i] - lv[i]).powi(2);
                    }
                }
                let mean = total / (128 * n) as f64;
                assert!(
                    mean <= (1. - delta) * input_cost,
                    "d={d}, h={h}: measured={mean}, bound={}",
                    (1. - delta) * input_cost
                );
            }
        }
    });
}

#[test]
fn actual_full_steps_retain_the_derived_ou_independent_memory_observable() {
    use algorithmic_gas::{
        RecordingConfig,
        noise::{FactorValues, NoiseGeometry},
    };
    futures_lite::future::block_on(async {
        let h = 0.04;
        let c = h / 2.;
        let k = 1. - c * c;
        let mut previous: Option<Vec<f64>> = None;
        for scale in [0.5, 1., 2.] {
            let mut config = RunConfig {
                benchmark: Benchmark::Quadratic,
                dimensions: 2,
                walkers: 16,
                gas: GasConfig::euclidean(2, h).unwrap(),
                ..Default::default()
            };
            config.gas.seed = 95_000;
            config.gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
                scale: FactorValues::Constant {
                    values: vec![scale],
                },
            };
            let mut gas = config.build::<f64>().await.unwrap();
            let mut initial = gas.population().clone();
            for row in &mut initial.validity[..4] {
                row.terminated = true;
            }
            gas.replace_population(initial).await.unwrap();
            gas.start_recording(RecordingConfig::default()).unwrap();
            gas.step().await.unwrap();
            let archive = gas.stop_recording().unwrap();
            let step = &archive.steps[0];
            assert_eq!(step.report.revivals, 4);
            let clone = step
                .stages
                .iter()
                .find(|s| s.stage == "post_clone")
                .unwrap();
            let b2 = step.stages.iter().find(|s| s.stage == "B2").unwrap();
            let final_state = gas.population();
            let mut memory = vec![];
            for i in 0..16 {
                let velocity = final_state
                    .observations
                    .field("velocities")
                    .unwrap()
                    .row(i)
                    .unwrap();
                let radius = velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
                for (j, v) in velocity.iter().enumerate() {
                    let index = 2 * i + j;
                    let position = final_state
                        .observations
                        .field("positions")
                        .unwrap()
                        .values()[index];
                    let inverse_cap = 2. * v / (2. - radius);
                    let actual = inverse_cap - k / c * position;
                    let position_noise = position - b2.fields["positions"].values[index];
                    let expected = -k / c * clone.fields["positions"].values[index]
                        - clone.fields["velocities"].values[index]
                        - k / c * position_noise;
                    assert!((actual - expected).abs() < 2e-12);
                    memory.push(actual);
                }
            }
            if let Some(prior) = &previous {
                for (a, b) in memory.iter().zip(prior) {
                    assert!((a - b).abs() < 2e-12);
                }
            }
            previous = Some(memory);
        }
    });
}
