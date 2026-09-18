use algorithmic_gas::{
    BackendKind, ExecutionContext, GasConfig, ObservationBatch, Population, Precision, TensorBatch,
    domain::NumericalDomain,
    fitness::ObjectiveDirection,
    kinetic::KineticContext,
    noise::{FactorValues, InnovationShift, NoiseGeometry},
    random::{RandomStream, Stream},
};
use algorithmic_gas_benchmarks::{Benchmark, BenchmarkModel};

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn force(benchmark: Benchmark, x: f64) -> f64 {
    match benchmark {
        Benchmark::Quadratic => -x,
        Benchmark::Rastrigin => {
            -2. * x - 10. * std::f64::consts::TAU * (std::f64::consts::TAU * x).sin()
        }
        _ => unreachable!(),
    }
}

fn inputs(n: usize, d: usize, shifted: bool) -> Population<f64> {
    let mut x = Vec::with_capacity(n * d);
    let mut v = Vec::with_capacity(n * d);
    for i in 0..n * d {
        let phase = i as f64 * 0.173;
        x.push(
            0.4 * phase.sin()
                + if shifted {
                    0.005 + 0.002 * phase.cos()
                } else {
                    0.
                },
        );
        v.push(0.2 * phase.cos() + if shifted { 0.007 * phase.sin() } else { 0. });
    }
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, d, x).unwrap());
    obs.fields
        .insert("velocities".into(), TensorBatch::vectors(n, d, v).unwrap());
    Population::new(obs).unwrap()
}

/// Both returned Gaussian vectors have standard normal marginals. Only their
/// joint law is changed: the reference stream implements translated maximal
/// coupling, while the production stream remains the left innovation.
fn translated_maximal(
    base: &[f64],
    shift: &[f64],
    reference: &mut RandomStream,
) -> (Vec<f64>, bool) {
    let half_square = shift.iter().map(|b| b * b).sum::<f64>() / 2.;
    let dot = base.iter().zip(shift).map(|(x, b)| x * b).sum::<f64>();
    if reference.uniform::<f64>().ln() <= (-dot - half_square).min(0.) {
        return (base.iter().zip(shift).map(|(x, b)| x + b).collect(), true);
    }
    loop {
        let candidate: Vec<f64> = (0..base.len())
            .map(|_| reference.gaussian::<f64>())
            .collect();
        let dot = candidate.iter().zip(shift).map(|(x, b)| x * b).sum::<f64>();
        if reference.uniform::<f64>().ln() > (dot - half_square).min(0.) {
            return (candidate, false);
        }
    }
}

#[test]
fn actual_nonlinear_baoab_obeys_translated_gaussian_marked_coupling() {
    futures_lite::future::block_on(async {
        let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64)
            .await
            .unwrap();
        let h: f64 = 0.04;
        let c = h / 2.;
        let a = (-h).exp();
        let radius = 2.;
        let n = 32;
        let runs = 64;
        let mut matched_total = 0;
        let mut unmatched_total = 0;
        let mut dead_matched_total = 0;
        let mut marginal_sum = 0.;
        let mut marginal_square = 0.;
        let mut marginal_count = 0;
        for (model_index, benchmark) in [Benchmark::Quadratic, Benchmark::Rastrigin]
            .into_iter()
            .enumerate()
        {
            let gradient = BenchmarkModel {
                benchmark,
                field: "positions".into(),
                direction: ObjectiveDirection::Minimize,
            };
            let lipschitz = match benchmark {
                Benchmark::Quadratic => 1.,
                Benchmark::Rastrigin => 2. + 10. * std::f64::consts::TAU.powi(2),
                _ => unreachable!(),
            };
            let lambda = 1. - c * c * lipschitz;
            assert!(lambda > 0.);
            for d in [1, 2, 3] {
                let left_input = inputs(n, d, false);
                let right_input = inputs(n, d, true);
                let lx = left_input.observations.field("positions").unwrap().values();
                let rx = right_input
                    .observations
                    .field("positions")
                    .unwrap()
                    .values();
                let lv = left_input
                    .observations
                    .field("velocities")
                    .unwrap()
                    .values();
                let rv = right_input
                    .observations
                    .field("velocities")
                    .unwrap()
                    .values();
                let mut delta_x1 = vec![vec![0.; d]; n];
                let mut delta_m = vec![vec![0.; d]; n];
                for row in 0..n {
                    for j in 0..d {
                        let i = row * d + j;
                        let vl = lv[i] + c * force(benchmark, lx[i]);
                        let vr = rv[i] + c * force(benchmark, rx[i]);
                        delta_x1[row][j] = lx[i] + c * vl - rx[i] - c * vr;
                        delta_m[row][j] = delta_x1[row][j] + c * a * (vl - vr);
                    }
                }
                for (scale_index, scale) in [16., 64., 256.].into_iter().enumerate() {
                    let q = scale * ((1. - a * a) / 2.).sqrt();
                    let mut config = GasConfig::euclidean(d, h).unwrap();
                    config.kinetic.noise.geometry = NoiseGeometry::Isotropic {
                        scale: FactorValues::Constant {
                            values: vec![scale],
                        },
                    };
                    let bound = delta_m
                        .iter()
                        .zip(&delta_x1)
                        .map(|(dm, dx)| {
                            (norm(dm) / (c * q * (2. * std::f64::consts::PI).sqrt())
                                + (std::f64::consts::PI / 2.).sqrt() * radius * norm(dx)
                                    / (c * q * lambda))
                                .min(1.)
                        })
                        .sum::<f64>()
                        / n as f64;
                    let seed_start =
                        120_000 + (model_index * 10_000 + d * 1_000 + scale_index * 100) as u64;
                    let mut run_means = Vec::with_capacity(runs);
                    for seed in seed_start..seed_start + runs as u64 {
                        let mut shifts = Vec::with_capacity(n * d);
                        let mut matched = Vec::with_capacity(n);
                        for (row, dm) in delta_m.iter().enumerate() {
                            let mut actual =
                                RandomStream::new(seed, 0, Stream::Kinetic, row as u64, 2);
                            let base: Vec<f64> = (0..d).map(|_| actual.gaussian::<f64>()).collect();
                            let shift: Vec<f64> = dm.iter().map(|x| x / (c * q)).collect();
                            let mut reference = RandomStream::new(
                                seed,
                                0,
                                Stream::MeanFieldReference,
                                row as u64,
                                29,
                            );
                            let (coupled, is_match) =
                                translated_maximal(&base, &shift, &mut reference);
                            matched.push(is_match);
                            for j in 0..d {
                                marginal_sum += coupled[j];
                                marginal_square += coupled[j] * coupled[j];
                                marginal_count += 1;
                                shifts.push(InnovationShift {
                                    step: 0,
                                    stream: Stream::Kinetic,
                                    substep: 2,
                                    walker: row,
                                    coordinate: j,
                                    shift: coupled[j] - base[j],
                                });
                            }
                        }
                        let mut left = left_input.clone();
                        let mut right = right_input.clone();
                        for (population, innovations) in [(&mut left, vec![]), (&mut right, shifts)]
                        {
                            cx.set_innovation_shifts(innovations).unwrap();
                            config
                                .kinetic
                                .advance(
                                    population,
                                    KineticContext {
                                        gradient: Some(&gradient),
                                        domain: &NumericalDomain,
                                        boundary: &config.boundary,
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
                        let mut cost = 0.;
                        for row in 0..n {
                            let x = left
                                .observations
                                .field("positions")
                                .unwrap()
                                .row(row)
                                .unwrap();
                            let y = right
                                .observations
                                .field("positions")
                                .unwrap()
                                .row(row)
                                .unwrap();
                            let v = left
                                .observations
                                .field("velocities")
                                .unwrap()
                                .row(row)
                                .unwrap();
                            let w = right
                                .observations
                                .field("velocities")
                                .unwrap()
                                .row(row)
                                .unwrap();
                            let dx: Vec<f64> = x.iter().zip(y).map(|(x, y)| x - y).collect();
                            let dv: Vec<f64> = v.iter().zip(w).map(|(v, w)| v - w).collect();
                            let status_differs = left.validity[row] != right.validity[row];
                            cost += (norm(&dx) + norm(&dv) + f64::from(status_differs)).min(1.);
                            if matched[row] {
                                matched_total += 1;
                                dead_matched_total +=
                                    usize::from(!left.validity[row].eligible(false));
                                assert!(norm(&dx) < 2e-12, "matched positions differ: {dx:?}");
                                assert!(
                                    !status_differs,
                                    "terminal classification differs at matched positions"
                                );
                                let vn = norm(v);
                                let wn = norm(w);
                                for j in 0..d {
                                    let pre_cap_difference = radius * v[j] / (radius - vn)
                                        - radius * w[j] / (radius - wn);
                                    let expected = -delta_x1[row][j] / c;
                                    assert!(
                                        (pre_cap_difference - expected).abs() < 2e-8,
                                        "nonlinear final force failed to cancel: measured={pre_cap_difference}, expected={expected}"
                                    );
                                }
                            } else {
                                unmatched_total += 1;
                            }
                        }
                        run_means.push(cost / n as f64);
                    }
                    let mean = run_means.iter().sum::<f64>() / runs as f64;
                    let se = (run_means.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / ((runs - 1) * runs) as f64)
                        .sqrt();
                    eprintln!(
                        "{benchmark:?}, d={d}, B={scale}: mean={mean:.8}, independent-run SE={se:.8}, bound={bound:.8}"
                    );
                    assert!(
                        mean <= bound + 6. * se,
                        "{benchmark:?}, d={d}, B={scale}: {mean} exceeds {bound}, SE={se}"
                    );
                }
            }
        }
        // Independent statistical checks detect an accidentally shifted Gaussian
        // marginal; individual innovations are never asserted equal in law by
        // their deterministic seed alone.
        let gaussian_mean = marginal_sum / marginal_count as f64;
        let gaussian_second = marginal_square / marginal_count as f64;
        assert!(gaussian_mean.abs() < 0.04);
        assert!((gaussian_second - 1.).abs() < 0.06);
        assert!(matched_total > 30_000 && unmatched_total > 20);
        assert!(
            dead_matched_total > 1_000,
            "terminal killing was not exercised"
        );
        eprintln!(
            "matched={matched_total}, unmatched={unmatched_total}, matched dead={dead_matched_total}, right Gaussian mean={gaussian_mean:.6}, second moment={gaussian_second:.6}"
        );
    });
}
