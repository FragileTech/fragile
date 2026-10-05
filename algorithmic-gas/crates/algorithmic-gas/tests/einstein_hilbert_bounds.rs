use algorithmic_gas::{
    GasConfig, ObservationBatch, TensorBatch,
    tessellation::{
        NeighborGraph, Parallelism,
        forces::{GraphField, boris_rotate, curl, viscous_force},
    },
    variants::einstein_hilbert::graph_kick_bounds,
};

#[test]
fn native_path_weights_retain_neighbor_noise_as_population_grows() {
    let config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    let pipeline = &config.geometry.as_ref().unwrap().pipeline;
    for n in [4, 17, 65] {
        let observations = ObservationBatch::positions(
            TensorBatch::vectors(n, 1, (0..n).map(|i| i as f64 / (n - 1) as f64).collect())
                .unwrap(),
        );
        let geometry = pipeline
            .evaluate(&observations, &vec![true; n], None, 10000)
            .unwrap();
        let weights = &geometry.weights["riemannian_kernel_volume"];
        for i in 1..n - 1 {
            assert_eq!(geometry.graph().degree(i), 2);
            let row = &weights[geometry.graph().range(i)];
            assert!(row.iter().all(|w| (w - 0.5).abs() < 1e-12));
            // Exact conditional variance coefficient, not a Monte Carlo fit.
            let variance = 9. * 0.33 * row.iter().map(|w| w * w).sum::<f64>();
            assert!((variance - 9. * 0.33 / 2.).abs() < 1e-12);
        }
    }
}

#[test]
fn sample_normalization_controls_native_fitness_tails_without_raw_moment_bounds() {
    let config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    for n in [4, 17, 128, 512] {
        for amplitude in [0_f64, 1., 1e120] {
            let rewards = algorithmic_gas::RewardBatch::new(
                (0..n)
                    .map(|i| if i == 0 { -amplitude } else { (i as f64).sin() })
                    .collect(),
                Default::default(),
            );
            let distance: Vec<_> = (0..n)
                .map(|i| if i == 1 { amplitude } else { 0. })
                .collect();
            let observations =
                ObservationBatch::positions(TensorBatch::vectors(n, 3, vec![0.; 3 * n]).unwrap());
            let fitness = config
                .fitness
                .evaluate(&rewards, &distance, &vec![true; n], &observations, 0)
                .unwrap();
            for z in [&fitness.reward_z, &fitness.diversity_z] {
                assert!(z.iter().sum::<f64>().abs() / (n as f64) < 1e-12);
                assert!(z.iter().map(|a| a * a).sum::<f64>() / (n as f64) <= 1. + 1e-12);
            }
            for l in [0.5_f64, 1., 2., 4., 8.] {
                let threshold = 4. / (1. + l.exp()).powi(2);
                let count = fitness.fitness.iter().filter(|&&f| f < threshold).count();
                assert!((count as f64) / (n as f64) <= 2. / (1. + l * l));
            }
        }
    }
}

#[test]
fn native_curvature_action_balance_and_dilation_match_the_geometric_proof() {
    let config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    let pipeline = &config.geometry.as_ref().unwrap().pipeline;
    for n in [8, 32] {
        let x: Vec<_> = (0..3 * n)
            .map(|i| (i as f64 * 1.718).sin() + i as f64 * 0.017)
            .collect();
        let evaluate = |scale: f64| {
            let observations = ObservationBatch::positions(
                TensorBatch::vectors(n, 3, x.iter().map(|a| scale * a).collect()).unwrap(),
            );
            pipeline
                .evaluate(&observations, &vec![true; n], None, 10000)
                .unwrap()
        };
        let base = evaluate(1.);
        for scale in [0.25_f64, 1., 4.] {
            let geometry = evaluate(scale);
            let graph = geometry.graph();
            assert_eq!(graph, base.graph());
            let b = &geometry.volume;
            let u: Vec<_> = b.iter().map(|a| a.ln() / 2.).collect();
            let r = &geometry.curvature["ricci_scalar"].scalar;
            let w = &geometry.weights["inverse_riemannian_distance"];
            let mut mass: Vec<_> = (0..n)
                .map(|i| {
                    graph
                        .range(i)
                        .map(|e| 1. / (geometry.lengths.geodesic_sq[e].max(1e-8).sqrt() + 1e-8))
                        .sum::<f64>()
                        .max(1e-12)
                })
                .collect();
            let sum: f64 = mass.iter().sum();
            for m in &mut mass {
                *m /= sum;
            }
            let (mut curvature, mut action, mut dirichlet, mut action_edges, mut r2) =
                (0., 0., 0., 0., 0.);
            for i in 0..n {
                curvature += mass[i] * r[i];
                action += mass[i] * b[i] * r[i];
                r2 += mass[i] * r[i] * r[i];
                for e in graph.range(i) {
                    let j = graph.neighbors()[e] as usize;
                    dirichlet += mass[i] * w[e] * (u[i] - u[j]).powi(2);
                    action_edges += mass[i] * w[e] * (b[i] - b[j]) * (u[i] - u[j]);
                }
                assert!((r[i] - base.curvature["ricci_scalar"].scalar[i]).abs() < 1e-8);
                assert!(
                    (b[i] * scale * scale - base.volume[i]).abs() < 1e-8 * (1. + base.volume[i])
                );
            }
            assert!(curvature.abs() < 1e-10);
            assert!((action - action_edges).abs() < 1e-10 * (1. + action.abs()));
            let min = b.iter().copied().fold(f64::INFINITY, f64::min);
            let max = b.iter().copied().fold(0_f64, f64::max);
            assert!(action >= 2. * min * dirichlet - 1e-10);
            assert!(action <= 2. * max * dirichlet + 1e-10);
            assert!(r2 <= 4. * dirichlet + 1e-10);
        }
    }
}

#[test]
fn native_volume_kernel_has_reversible_masses_even_when_the_floor_binds() {
    let config = GasConfig::einstein_hilbert(0.33, 0.002).unwrap();
    for length_scale in [1_f64, 0.1] {
        let observations = ObservationBatch::positions(
            TensorBatch::vectors(5, 1, vec![0_f64, 1., 2., 4., 8.]).unwrap(),
        );
        let mut pipeline = config.geometry.as_ref().unwrap().pipeline.clone();
        // Exercise the existing configurable kernel with a short length scale
        // as well as the preset. The volume floor alone does not force a
        // binding normalization floor when the preset kernel is broad.
        for spec in &mut pipeline.weights {
            spec.length_scale = length_scale;
        }
        let geometry = pipeline
            .evaluate(&observations, &[true; 5], None, 100)
            .unwrap();
        let graph = geometry.graph();
        let weights = &geometry.weights["riemannian_kernel_volume"];
        let mass: Vec<f64> = (0..5)
            .map(|i| {
                let sum: f64 = graph
                    .range(i)
                    .map(|e| {
                        let j = graph.neighbors()[e] as usize;
                        (-geometry.lengths.geodesic_sq[e] / (2. * length_scale.powi(2))).exp()
                            * geometry.volume[j]
                    })
                    .sum();
                geometry.volume[i] * sum.max(1e-12)
            })
            .collect();
        for i in 0..5 {
            for e in graph.range(i) {
                let j = graph.neighbors()[e] as usize;
                let left = mass[i] * weights[e];
                let right = mass[j] * weights[graph.reverse()[e] as usize];
                assert!((left - right).abs() <= 1e-12 * left.max(right));
            }
        }
        if length_scale < 1. {
            assert!(
                graph_kick_bounds(graph, weights, 3., 0.002)
                    .unwrap()
                    .maximum_row_sum
                    < 1.
            );
        }
    }
}

#[test]
fn enumerate_even_and_odd_matching_expectations_with_the_native_gate() {
    fn permutations(values: &mut [usize], k: usize, visit: &mut impl FnMut(&[usize])) {
        if k == values.len() {
            visit(values);
            return;
        }
        for j in k..values.len() {
            values.swap(k, j);
            permutations(values, k + 1, visit);
            values.swap(k, j);
        }
    }
    let decision = GasConfig::einstein_hilbert(0.33, 0.002)
        .unwrap()
        .clone_decision;
    for n in 2..=7 {
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.7 - 1.).collect();
        let v: Vec<f64> = (0..n).map(|i| (i as f64 + 0.3).sin()).collect();
        let fitness: Vec<f64> = (0..n).map(|i| 0.1 + i as f64 * 0.2).collect();
        let phi = |i: usize, j: usize| (x[j] + 2. * v[i]).tanh();
        let base = (0..n).map(|i| phi(i, i)).sum::<f64>() / n as f64;
        let delta = |i: usize, j: usize| {
            decision.acceptance_probability(20, fitness[i], fitness[j]) * (phi(i, j) - phi(i, i))
        };
        let product_increment = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| delta(i, j))
            .sum::<f64>()
            / (n * n) as f64;
        let mut means = Vec::new();
        let mut variances = Vec::new();
        permutations(&mut (0..n).collect::<Vec<_>>(), 0, &mut |order| {
            let mut partners: Vec<_> = (0..n).collect();
            for pair in order.chunks_exact(2) {
                partners[pair[0]] = pair[1];
                partners[pair[1]] = pair[0];
            }
            means.push(base + (0..n).map(|i| delta(i, partners[i])).sum::<f64>() / n as f64);
            variances.push(
                (0..n)
                    .map(|i| {
                        let j = partners[i];
                        let p = decision.acceptance_probability(20, fitness[i], fitness[j]);
                        p * (1. - p) * (phi(i, j) - phi(i, i)).powi(2)
                    })
                    .sum::<f64>()
                    / (n * n) as f64,
            );
        });
        let mean = means.iter().sum::<f64>() / means.len() as f64;
        let correction = if n % 2 == 0 {
            n as f64 / (n - 1) as f64
        } else {
            1.
        };
        assert!((mean - base - correction * product_increment).abs() < 1e-12);
        let variance = means
            .iter()
            .zip(variances)
            .map(|(m, v)| v + (m - mean).powi(2))
            .sum::<f64>()
            / means.len() as f64;
        assert!(variance <= 65. / n as f64);
        assert_eq!(decision.acceptance_probability(19, 0.1, 0.9), 0.);
    }
}

#[test]
fn row_averaging_bounds_the_actual_boris_kick_but_can_increase_mean_energy() {
    // A star with row-stochastic asymmetric weights. This tests the bound's
    // hypotheses; it does not assert these are a particular EH geometry's weights.
    let graph = NeighborGraph::from_undirected(5, &[[0, 1], [0, 2], [0, 3], [0, 4]]).unwrap();
    let weights = vec![0.25, 0.25, 0.25, 0.25, 1., 1., 1., 1.];
    let bounds = graph_kick_bounds(&graph, &weights, 3., 4. / 3.).unwrap();
    assert_eq!(bounds.maximum_row_sum, 1.);
    assert_eq!(bounds.maximum_column_sum, 4.);
    assert_eq!(bounds.quarter_kick_moment_factor, 4.);
    let positions = [0., 0., 1., 0., 0., 1., -1., 0., 0., -1.];
    let eligible = [true; 5];
    let field = GraphField {
        graph: &graph,
        weights: &weights,
        positions: &positions,
        dimension: 2,
        eligible: &eligible,
        wrap: &[],
    };
    let v = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.];
    let force = viscous_force(&field, &v, 3., Parallelism::Serial);
    let first: Vec<_> = v.iter().zip(&force).map(|(v, f)| v + f / 3.).collect();
    let energy = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    assert_eq!(energy(&first), 4. * energy(&v));
    let omega = curl(&field, &force, Parallelism::Serial).unwrap();
    let (rotated, _) =
        boris_rotate(&first, &omega, 2, 1. / 3., &eligible, Parallelism::Serial).unwrap();
    let force2 = viscous_force(&field, &rotated, 3., Parallelism::Serial);
    let result: Vec<_> = rotated
        .iter()
        .zip(force2)
        .map(|(v, f)| v + f / 3.)
        .collect();
    for row in result.chunks_exact(2) {
        assert!(energy(row) <= 1. + 1e-12);
    }
    assert!(energy(&result) <= bounds.quarter_kick_moment_factor.powi(2) * energy(&v) + 1e-12);
}

#[test]
fn bounds_include_empty_rows_and_reject_invalid_or_nonconvex_kicks() {
    let empty = NeighborGraph::empty(2);
    assert_eq!(
        graph_kick_bounds(&empty, &[] as &[f64], 3., 10.)
            .unwrap()
            .quarter_kick_moment_factor,
        1.
    );
    let pair = NeighborGraph::from_undirected(2, &[[0, 1]]).unwrap();
    let bounds = graph_kick_bounds(&pair, &[0.1, 0.3], 3., 0.002).unwrap();
    assert!((bounds.quarter_kick_moment_factor - 1.0003).abs() < 1e-14);
    assert!(graph_kick_bounds(&pair, &[1., 1.], 3., 2.).is_err());
    for weights in [[-1., 1.], [f64::NAN, 1.], [f64::INFINITY, 1.]] {
        assert!(graph_kick_bounds(&pair, &weights, 3., 0.002).is_err());
    }
    assert!(graph_kick_bounds(&pair, &[1.], 3., 0.002).is_err());
    assert!(graph_kick_bounds(&pair, &[1., 1.], f64::MAX, f64::MAX).is_err());
}
