//! Shortest-path scales, gates, colour smearing and the graph smoothing
//! diagnostic against hand-computed vectors, a Floyd-Warshall oracle and
//! closed forms.
use algorithmic_gas::{
    GasError,
    physics::{
        numerics::shortest_paths,
        qft::math::{C, dot},
        spectroscopy::{
            config::{EdgeLength, FlowConfig, ScaleSelection},
            contract::{Element, ElementKind, FrameState, Topology},
            flow,
            report::Coverage,
            scales,
        },
    },
    tessellation::{GraphSnapshot, NeighborGraph, Parallelism},
};
use std::collections::BTreeMap;

const INF: f64 = f64::INFINITY;
const BYTES: usize = 1 << 24;
const G6_PAIRS: [[u32; 2]; 6] = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4]];
const G6_EUCLIDEAN: [f64; 12] = [1., 1.5, 1., 1., 2., 1.5, 1., 1.25, 2., 1.25, 0.5, 0.5];
const G6_GEODESIC: [f64; 12] = [1.5, 3., 1.5, 1., 2.5, 3., 1., 1.75, 2.5, 1.75, 0.75, 0.75];
/// Row-major table from its rows.
const fn flat(rows: [[f64; 6]; 6]) -> [f64; 36] {
    let mut table = [0.; 36];
    let mut k = 0;
    while k < 36 {
        table[k] = rows[k / 6][k % 6];
        k += 1;
    }
    table
}
const D_EUCLIDEAN: [f64; 36] = flat([
    [0., 1., 1.5, 2.75, 3.25, INF],
    [1., 0., 1., 2., 2.5, INF],
    [1.5, 1., 0., 1.25, 1.75, INF],
    [2.75, 2., 1.25, 0., 0.5, INF],
    [3.25, 2.5, 1.75, 0.5, 0., INF],
    [INF, INF, INF, INF, INF, 0.],
]);
const D_GEODESIC: [f64; 36] = flat([
    [0., 1.5, 2.5, 4., 4.75, INF],
    [1.5, 0., 1., 2.5, 3.25, INF],
    [2.5, 1., 0., 1.75, 2.5, INF],
    [4., 2.5, 1.75, 0., 0.75, INF],
    [4.75, 3.25, 2.5, 0.75, 0., INF],
    [INF, INF, INF, INF, INF, 0.],
]);

/// Five connected nodes with dyadic lengths, so every distance is exact, and
/// an isolated sixth node.
fn g6() -> GraphSnapshot<f64> {
    GraphSnapshot {
        graph: NeighborGraph::from_undirected(6, &G6_PAIRS).unwrap(),
        weights: BTreeMap::new(),
        euclidean_length: G6_EUCLIDEAN.to_vec(),
        geodesic_length: G6_GEODESIC.to_vec(),
        wrap: Vec::new(),
        stale_steps: 0,
    }
}
/// Deterministic hash to the unit interval.
fn unit(k: u64) -> f64 {
    let mut z = k.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}
/// Random geometric graph in the unit square with its irrational Euclidean
/// slot lengths.
fn geometric(seed: u64, n: usize, radius: f64) -> (NeighborGraph, Vec<f64>) {
    let x: Vec<f64> = (0..2 * n as u64).map(|k| unit(seed * 4096 + k)).collect();
    let apart = |a: usize, b: usize| (x[2 * a] - x[2 * b]).hypot(x[2 * a + 1] - x[2 * b + 1]);
    let mut pairs = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            if apart(a, b) < radius {
                pairs.push([a as u32, b as u32]);
            }
        }
    }
    let graph = NeighborGraph::from_undirected(n, &pairs).unwrap();
    let length = graph
        .coo()
        .iter()
        .map(|&[a, b]| apart(a as usize, b as usize))
        .collect();
    (graph, length)
}
/// Independent oracle: min-plus closure of the dense length matrix.
fn floyd_warshall(graph: &NeighborGraph, length: &[f64]) -> Vec<f64> {
    let n = graph.nodes();
    let mut d = vec![INF; n * n];
    for i in 0..n {
        d[i * n + i] = 0.;
    }
    for (&[a, b], &l) in graph.coo().iter().zip(length) {
        let slot = a as usize * n + b as usize;
        d[slot] = d[slot].min(l);
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                d[i * n + j] = d[i * n + j].min(d[i * n + k] + d[k * n + j]);
            }
        }
    }
    d
}
fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}
fn table(graph: &NeighborGraph, length: &[f64], radius: f64) -> Vec<f64> {
    shortest_paths::all_pairs(graph, length, radius, BYTES, Parallelism::Serial).unwrap()
}
fn element(kind: ElementKind, walkers: [u32; 3]) -> Element {
    Element {
        walkers,
        kind,
        weight: 1.,
        generation: [0; 3],
    }
}
fn topology(kind: ElementKind, walkers: &[[u32; 3]]) -> Topology {
    Topology {
        step: 0,
        kind,
        elements: walkers.iter().map(|&w| element(kind, w)).collect(),
        involutive: false,
        coverage: Coverage::default(),
    }
}
fn state(d: usize, color: Vec<C>, color_valid: Vec<bool>) -> FrameState {
    let n = color_valid.len();
    FrameState {
        n,
        d,
        color,
        color_valid,
        cloned: vec![false; n],
        ..FrameState::default()
    }
}
fn g6_colors() -> Vec<C> {
    let h = 0.5f64.sqrt();
    vec![
        C::new(1., 0.),
        C::ZERO,
        C::new(h, 0.),
        C::new(0., h),
        C::ZERO,
        C::ONE,
        C::new(h, 0.),
        C::new(h, 0.),
        C::new(0., 1.),
        C::ZERO,
        C::new(h, 0.),
        C::new(-h, 0.),
    ]
}
fn g6_state() -> FrameState {
    state(2, g6_colors(), vec![true, true, true, true, false, true])
}
/// Random unit colours `[n, d]`.
fn unit_colors(seed: u64, n: usize, d: usize) -> Vec<C> {
    let mut color: Vec<C> = (0..(n * d) as u64)
        .map(|k| {
            C::new(
                unit(seed * 8192 + 2 * k) - 0.5,
                unit(seed * 8192 + 2 * k + 1) - 0.5,
            )
        })
        .collect();
    for row in color.chunks_exact_mut(d) {
        let norm = row.iter().map(|c| c.abs2()).sum::<f64>().sqrt();
        row.iter_mut().for_each(|c| *c = *c / norm);
    }
    color
}
/// The same `U(2)` matrix applied to every two-component colour.
fn rotated(color: &[C]) -> Vec<C> {
    let (s, c) = 0.7f64.sin_cos();
    let global = C::phase(0.4);
    let u = [
        global * C::phase(1.3) * c,
        global * C::phase(-0.6) * s,
        -(global * C::phase(0.6) * s),
        global * C::phase(-1.3) * c,
    ];
    color
        .chunks_exact(2)
        .flat_map(|r| [u[0] * r[0] + u[1] * r[1], u[2] * r[0] + u[3] * r[1]])
        .collect()
}
fn curve(state: &FrameState, graph: &NeighborGraph, steps: usize, step_size: f64) -> Vec<f64> {
    let config = FlowConfig {
        steps,
        step_size,
        every: 1,
    };
    flow::smooth(state, graph, &config)
        .unwrap()
        .into_iter()
        .map(|r| r.unwrap())
        .collect()
}
fn close(a: &[f64], b: &[f64], t: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < t, "{x} != {y}, tolerance {t}");
    }
}

#[test]
fn hand_computed_distance_tables_are_reproduced_exactly_for_both_edge_lengths() {
    let g = g6();
    assert_eq!(g.graph.offsets(), [0, 2, 5, 8, 11, 12, 12]);
    assert_eq!(g.graph.neighbors(), [1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 4, 3]);
    let euclidean =
        scales::distances(&g, EdgeLength::Euclidean, INF, BYTES, Parallelism::Serial).unwrap();
    let geodesic =
        scales::distances(&g, EdgeLength::Geodesic, INF, BYTES, Parallelism::Serial).unwrap();
    assert_eq!(bits(&euclidean), bits(&D_EUCLIDEAN));
    assert_eq!(bits(&geodesic), bits(&D_GEODESIC));
    for (e, g) in euclidean
        .iter()
        .zip(&geodesic)
        .filter(|(e, _)| e.is_finite())
    {
        assert!(*e <= *g && *g <= 2. * e);
    }
}
#[test]
fn truncation_keeps_a_distance_equal_to_the_radius_and_drops_everything_beyond() {
    let g = g6();
    let cut = |full: &[f64], radius: f64| -> Vec<f64> {
        full.iter()
            .map(|&d| if d <= radius { d } else { INF })
            .collect()
    };
    let euclidean =
        scales::distances(&g, EdgeLength::Euclidean, 2., BYTES, Parallelism::Serial).unwrap();
    assert_eq!(bits(&euclidean), bits(&cut(&D_EUCLIDEAN, 2.)));
    assert_eq!(euclidean[9], 2.);
    assert_eq!([euclidean[3], euclidean[4], euclidean[10]], [INF; 3]);
    let geodesic =
        scales::distances(&g, EdgeLength::Geodesic, 2.5, BYTES, Parallelism::Serial).unwrap();
    assert_eq!(bits(&geodesic), bits(&cut(&D_GEODESIC, 2.5)));
    assert_eq!(geodesic.iter().filter(|d| **d == 2.5).count(), 6);
}
#[test]
fn dijkstra_agrees_with_floyd_warshall_and_is_a_bit_symmetric_metric_on_random_graphs() {
    let mut asymmetric = 0;
    for seed in 0..12 {
        let n = 12 + seed as usize;
        let (graph, length) = geometric(seed, n, 0.45);
        let d = table(&graph, &length, INF);
        let oracle = floyd_warshall(&graph, &length);
        for (a, b) in d.iter().zip(&oracle) {
            assert!(a == b || (a - b).abs() < 1e-12, "{a} vs {b}");
        }
        let rows: Vec<f64> = (0..n)
            .flat_map(|i| shortest_paths::from_source(&graph, &length, i, INF).unwrap())
            .collect();
        for i in 0..n {
            assert_eq!(d[i * n + i], 0.);
            for j in 0..n {
                assert_eq!(d[i * n + j].to_bits(), d[j * n + i].to_bits());
                assert_eq!(d[i * n + j], rows[i * n + j].min(rows[j * n + i]));
                asymmetric += usize::from(rows[i * n + j] != rows[j * n + i]);
                for k in 0..n {
                    assert!(d[i * n + k] <= d[i * n + j] + d[j * n + k] + 1e-12);
                }
            }
        }
    }
    // The symmetrisation is not a no-op: opposite summation orders round apart.
    assert!(asymmetric > 0);
}
#[test]
fn truncated_distances_equal_the_full_ones_within_the_radius_exactly() {
    for seed in 20..28 {
        let (graph, length) = geometric(seed, 20, 0.4);
        let n = graph.nodes();
        let full = table(&graph, &length, INF);
        for radius in [0., 0.3, 0.7, full[1].min(2.)] {
            let expected: Vec<f64> = full
                .iter()
                .map(|&d| if d <= radius { d } else { INF })
                .collect();
            assert_eq!(bits(&table(&graph, &length, radius)), bits(&expected));
            let row = shortest_paths::from_source(&graph, &length, 3, radius).unwrap();
            let untruncated = shortest_paths::from_source(&graph, &length, 3, INF).unwrap();
            for j in 0..n {
                let kept = if untruncated[j] <= radius {
                    untruncated[j]
                } else {
                    INF
                };
                assert_eq!(row[j].to_bits(), kept.to_bits());
            }
        }
    }
}
#[test]
fn path_cycle_and_star_graphs_have_their_closed_form_distances() {
    let path: Vec<[u32; 2]> = (0..8).map(|i| [i, i + 1]).collect();
    let graph = NeighborGraph::from_undirected(9, &path).unwrap();
    let d = table(&graph, &vec![0.25; graph.edges()], INF);
    for i in 0..9 {
        for j in 0..9 {
            assert_eq!(d[i * 9 + j], 0.25 * i.abs_diff(j) as f64);
        }
    }
    let cycle: Vec<[u32; 2]> = (0..7).map(|i| [i, (i + 1) % 7]).collect();
    let graph = NeighborGraph::from_undirected(7, &cycle).unwrap();
    let d = table(&graph, &vec![1.; graph.edges()], INF);
    for i in 0..7usize {
        for j in 0..7usize {
            assert_eq!(d[i * 7 + j], i.abs_diff(j).min(7 - i.abs_diff(j)) as f64);
        }
    }
    let star: Vec<[u32; 2]> = (1..6).map(|leaf| [0, leaf]).collect();
    let graph = NeighborGraph::from_undirected(6, &star).unwrap();
    let length: Vec<f64> = graph
        .coo()
        .iter()
        .map(|&[a, b]| 0.25 * a.max(b) as f64)
        .collect();
    let d = table(&graph, &length, INF);
    for a in 1..6 {
        for b in (1..6).filter(|&b| b != a) {
            assert_eq!(d[a * 6 + b], 0.25 * (a + b) as f64);
        }
    }
}
#[test]
fn grid_graph_distance_between_opposite_corners_exceeds_the_straight_line() {
    let mut pairs = Vec::new();
    for r in 0..3u32 {
        for c in 0..3u32 {
            if c < 2 {
                pairs.push([3 * r + c, 3 * r + c + 1]);
            }
            if r < 2 {
                pairs.push([3 * r + c, 3 * r + c + 3]);
            }
        }
    }
    let graph = NeighborGraph::from_undirected(9, &pairs).unwrap();
    let d = table(&graph, &vec![0.5; graph.edges()], INF);
    assert_eq!(d[8], 2.);
    assert!(d[8] > 2f64.sqrt());
}
#[test]
fn zero_length_edges_merge_their_endpoints_and_other_components_stay_unreachable() {
    let graph = NeighborGraph::from_undirected(7, &[[0, 1], [1, 2], [2, 3], [4, 5]]).unwrap();
    let length: Vec<f64> = graph
        .coo()
        .iter()
        .map(|&[a, b]| if a.min(b) == 1 { 0. } else { 0.75 })
        .collect();
    let d = table(&graph, &length, INF);
    assert_eq!(d[7 + 2], 0.);
    for k in 0..7 {
        assert_eq!(d[7 + k].to_bits(), d[2 * 7 + k].to_bits());
    }
    assert_eq!([d[3], d[4], d[6], d[4 * 7 + 6]], [1.5, INF, INF, INF]);
    assert_eq!(d[4 * 7 + 5], 0.75);
    assert_eq!(table(&graph, &length, 0.)[7 + 2], 0.);
}
#[test]
fn serial_and_parallel_distance_tables_are_bit_identical() {
    let (graph, length) = geometric(40, 60, 0.3);
    for radius in [INF, 0.6] {
        let serial =
            shortest_paths::all_pairs(&graph, &length, radius, BYTES, Parallelism::Serial).unwrap();
        let parallel =
            shortest_paths::all_pairs(&graph, &length, radius, BYTES, Parallelism::Always).unwrap();
        assert_eq!(bits(&serial), bits(&parallel));
        let auto =
            shortest_paths::all_pairs(&graph, &length, radius, BYTES, Parallelism::Auto).unwrap();
        assert_eq!(bits(&serial), bits(&auto));
    }
    let empty = NeighborGraph::empty(0);
    assert!(table(&empty, &[], INF).is_empty());
    assert_eq!(table(&NeighborGraph::empty(1), &[], 0.), [0.]);
}
#[test]
fn invalid_lengths_radii_sources_and_budgets_are_rejected_explicitly() {
    let g = g6();
    for bad in [-1., f64::NAN, INF] {
        let mut length = G6_GEODESIC.to_vec();
        length[4] = bad;
        assert!(matches!(
            shortest_paths::all_pairs(&g.graph, &length, INF, BYTES, Parallelism::Serial),
            Err(GasError::Numerical(_))
        ));
        assert!(matches!(
            shortest_paths::from_source(&g.graph, &length, 0, INF),
            Err(GasError::Numerical(_))
        ));
    }
    assert!(matches!(
        shortest_paths::from_source(&g.graph, &G6_GEODESIC[..11], 0, INF),
        Err(GasError::Configuration(_))
    ));
    for radius in [-1., f64::NAN] {
        assert!(matches!(
            shortest_paths::from_source(&g.graph, &G6_GEODESIC, 0, radius),
            Err(GasError::Configuration(_))
        ));
    }
    assert!(matches!(
        shortest_paths::from_source(&g.graph, &G6_GEODESIC, 6, INF),
        Err(GasError::Configuration(_))
    ));
    let malformed: NeighborGraph =
        serde_json::from_str(r#"{"offsets":[0,1,2],"neighbors":[7,0],"reverse":[1,0]}"#).unwrap();
    assert!(matches!(
        shortest_paths::from_source(&malformed, &[1., 1.], 0, INF),
        Err(GasError::Configuration(_))
    ));
    assert!(matches!(
        shortest_paths::all_pairs(&malformed, &[1., 1.], INF, BYTES, Parallelism::Serial),
        Err(GasError::Configuration(_))
    ));
    let field = state(1, vec![C::ONE; 2], vec![true; 2]);
    assert!(matches!(
        flow::smooth(&field, &malformed, &FlowConfig::default()),
        Err(GasError::Configuration(_))
    ));
    assert!(matches!(
        shortest_paths::all_pairs(&g.graph, &G6_GEODESIC, INF, 287, Parallelism::Serial),
        Err(GasError::Capability(_))
    ));
    assert!(
        shortest_paths::all_pairs(&g.graph, &G6_GEODESIC, INF, 288, Parallelism::Serial).is_ok()
    );
}
#[test]
fn rescaled_edge_lengths_bound_the_distance_by_the_extreme_edge_ratios() {
    let (graph, euclidean) = geometric(50, 30, 0.4);
    let ratio: Vec<f64> = graph
        .coo()
        .iter()
        .map(|&[a, b]| 0.5 + 2.5 * unit(900_000 + 64 * a.min(b) as u64 + a.max(b) as u64))
        .collect();
    let metric: Vec<f64> = euclidean.iter().zip(&ratio).map(|(l, r)| l * r).collect();
    let (low, high) = ratio
        .iter()
        .fold((INF, 0f64), |(l, h), &r| (l.min(r), h.max(r)));
    let (flat, curved) = (table(&graph, &euclidean, INF), table(&graph, &metric, INF));
    for (e, g) in flat.iter().zip(&curved) {
        assert_eq!(e.is_finite(), g.is_finite());
        if e.is_finite() {
            assert!(low * e <= g * (1. + 1e-12) && *g <= high * e * (1. + 1e-12));
        }
    }
}

#[test]
fn a_hop_costs_the_length_of_the_csr_slot_it_leaves_through() {
    let graph = NeighborGraph::from_undirected(3, &[[0, 1], [1, 2]]).unwrap();
    assert_eq!(graph.neighbors(), [1, 0, 2, 1]);
    // Slots 0→1, 1→0, 1→2, 2→1 with a different length in each direction.
    let length = [1., 4., 0.5, 2.];
    let from = |source, radius| shortest_paths::from_source(&graph, &length, source, radius);
    assert_eq!(from(0, INF).unwrap(), [0., 1., 1.5]);
    assert_eq!(from(1, INF).unwrap(), [4., 0., 0.5]);
    assert_eq!(from(2, INF).unwrap(), [6., 2., 0.]);
    assert_eq!(from(2, 2.).unwrap(), [INF, 2., 0.]);
    // The table keeps the shorter direction, also when only that one is
    // within the radius.
    let shorter = [0., 1., 1.5, 1., 0., 0.5, 1.5, 0.5, 0.];
    assert_eq!(table(&graph, &length, INF), shorter);
    assert_eq!(table(&graph, &length, 1.5), shorter);
    assert_eq!(
        table(&graph, &length, 1.),
        [0., 1., INF, 1., 0., 0.5, INF, 0.5, 0.]
    );
}
#[test]
fn pair_samples_walk_the_upper_triangle_once_at_a_constant_stride() {
    let finite = [1.5, 2.5, 4., 4.75, 1., 2.5, 3.25, 1.75, 2.5, 0.75];
    assert_eq!(scales::pair_samples(&D_GEODESIC, 6, 100), finite);
    assert_eq!(scales::pair_samples(&D_GEODESIC, 6, 10), finite);
    // The stride spans the whole triangle: the last rows are sampled too.
    assert_eq!(
        scales::pair_samples(&D_GEODESIC, 6, 9),
        [1.5, 4., 1., 3.25, 2.5]
    );
    assert_eq!(
        scales::pair_samples(&D_GEODESIC, 6, 4),
        [1.5, 4.75, 3.25, 0.75]
    );
    assert_eq!(scales::pair_samples(&D_GEODESIC, 6, 3), [1.5, 1., 2.5]);
    assert_eq!(scales::pair_samples(&D_GEODESIC, 6, 1), [1.5]);
    assert!(scales::pair_samples(&D_GEODESIC, 6, 0).is_empty());
    assert!(scales::pair_samples(&[], 0, 8).is_empty());
    let n = 40;
    let (graph, length) = geometric(80, n, 0.4);
    let d = table(&graph, &length, INF);
    let all = scales::pair_samples(&d, n, usize::MAX);
    for cap in [1, 7, all.len() / 2 + 1, all.len() - 1] {
        let sample = scales::pair_samples(&d, n, cap);
        let stride = all.len().div_ceil(cap);
        assert!(sample.len() <= cap && sample.len() == all.len().div_ceil(stride));
        assert!(all.len() - 1 - (sample.len() - 1) * stride < stride);
        for (k, s) in sample.iter().enumerate() {
            assert_eq!(s.to_bits(), all[k * stride].to_bits());
        }
    }
    let mut coincident = D_GEODESIC;
    coincident[1] = 0.;
    assert_eq!(scales::pair_samples(&coincident, 6, 100), finite[1..]);
}
#[test]
fn quantile_ladder_matches_the_closed_form_and_the_hand_computed_graph_vector() {
    let quantiles = |count| ScaleSelection::Quantiles {
        count,
        low: 0.05,
        high: 0.95,
    };
    let integers: Vec<f64> = (1..=101).map(f64::from).collect();
    let ladder = scales::calibrate(&integers, &quantiles(3)).unwrap();
    assert_eq!([ladder[0], ladder[2]], [6., 96.]);
    assert!((ladder[1] - 24.).abs() < 1e-13);
    let samples = scales::pair_samples(&D_GEODESIC, 6, scales::CALIBRATION_SAMPLES);
    let ladder = scales::calibrate(&samples, &quantiles(4)).unwrap();
    assert_eq!([ladder[0], ladder[3]], [0.8625, 4.4125]);
    close(&ladder[1..3], &[1.48616709682881, 2.56080306051776], 1e-14);
    for rung in ladder.windows(2) {
        assert!((rung[1] / rung[0] - 1.72309228617833).abs() < 1e-14);
    }
    assert_eq!(
        scales::calibrate(&samples, &quantiles(2)).unwrap(),
        [0.8625, 4.4125]
    );
    let single = scales::calibrate(&samples, &quantiles(1)).unwrap();
    assert_eq!(single.len(), 1);
    assert!((single[0] - 1.95084116472869).abs() < 1e-14);
}
#[test]
fn calibration_ignores_unreachable_and_coincident_pairs_and_sample_order_and_scales_linearly() {
    let selection = ScaleSelection::Quantiles {
        count: 5,
        low: 0.1,
        high: 0.8,
    };
    let samples: Vec<f64> = (0..200).map(|k| 0.05 + 3. * unit(7000 + k)).collect();
    let ladder = scales::calibrate(&samples, &selection).unwrap();
    assert!(ladder.len() == 5 && ladder.windows(2).all(|w| w[0] < w[1]));
    let mut padded: Vec<f64> = samples.iter().rev().copied().collect();
    padded.extend([INF, 0., f64::NAN, -1., INF]);
    assert_eq!(
        bits(&scales::calibrate(&padded, &selection).unwrap()),
        bits(&ladder)
    );
    let doubled: Vec<f64> = samples.iter().map(|s| 4. * s).collect();
    let scaled = scales::calibrate(&doubled, &selection).unwrap();
    for (a, b) in scaled.iter().zip(&ladder) {
        assert!((a - 4. * b).abs() < 1e-12 * a);
    }
}
#[test]
fn degenerate_samples_give_one_scale_and_empty_samples_fail_explicitly() {
    let selection = ScaleSelection::Quantiles {
        count: 4,
        low: 0.05,
        high: 0.95,
    };
    assert_eq!(scales::calibrate(&[0.75; 9], &selection).unwrap(), [0.75]);
    for empty in [&[][..], &[INF, 0., f64::NAN][..]] {
        assert!(matches!(
            scales::calibrate(empty, &selection),
            Err(GasError::Numerical(_))
        ));
    }
    let fixed = ScaleSelection::Fixed {
        values: vec![0.5, 1.25],
    };
    assert_eq!(scales::calibrate(&[], &fixed).unwrap(), [0.5, 1.25]);
    let unordered = ScaleSelection::Quantiles {
        count: 4,
        low: 0.9,
        high: 0.1,
    };
    assert!(matches!(
        scales::calibrate(&[1., 2.], &unordered),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn every_selection_is_validated_before_it_is_used_and_the_sample_budget_is_the_stated_one() {
    assert_eq!(scales::CALIBRATION_SAMPLES, 500_000);
    assert_eq!(scales::SMEAR_CUTOFF, 8.);
    let fixed = [
        vec![],
        vec![1., 0.5],
        vec![0.5, 0.5],
        vec![0., 1.],
        vec![-1., 1.],
        vec![f64::NAN],
        vec![1., INF],
        vec![0.25; 65],
    ];
    for values in fixed {
        assert!(matches!(
            scales::calibrate(&[1., 2.], &ScaleSelection::Fixed { values }),
            Err(GasError::Configuration(_))
        ));
    }
    for (count, low, high) in [
        (0, 0.05, 0.95),
        (65, 0.05, 0.95),
        (4, 0., 0.95),
        (4, 0.05, 1.),
    ] {
        assert!(matches!(
            scales::calibrate(&[1., 2.], &ScaleSelection::Quantiles { count, low, high }),
            Err(GasError::Configuration(_))
        ));
    }
}
#[test]
fn pair_and_triplet_gates_reproduce_the_hand_computed_table() {
    let distance = [1u32, 0, 3, 2, 3, 5];
    let cloning = [2u32, 3, 1, 4, 5, 0];
    let walkers = |other: &[u32; 6]| -> Vec<[u32; 3]> {
        (0..6u32).map(|i| [i, other[i as usize], i]).collect()
    };
    let pair_j = topology(ElementKind::DistancePair, &walkers(&distance));
    let pair_k = topology(ElementKind::CloningPair, &walkers(&cloning));
    let triplets: Vec<[u32; 3]> = (0..6u32)
        .map(|i| [i, distance[i as usize], cloning[i as usize]])
        .collect();
    let triplet = topology(ElementKind::Triplet, &triplets);
    let reach = |t: &Topology| -> Vec<f64> {
        t.elements
            .iter()
            .map(|e| scales::diameter(e, &D_GEODESIC, 6))
            .collect()
    };
    assert_eq!(reach(&pair_j), [1.5, 1.5, 1.75, 1.75, 0.75, 0.]);
    assert_eq!(reach(&pair_k), [2.5, 2.5, 1., 0.75, INF, INF]);
    assert_eq!(reach(&triplet), [2.5, 4., 2.5, 2.5, INF, INF]);
    let gate = |t: &Topology, scale: f64| -> Vec<u8> {
        scales::gate(t, &D_GEODESIC, 6, scale, &[false; 6])
            .into_iter()
            .map(u8::from)
            .collect()
    };
    // Walker 5 is its own distance companion at distance 0: the gate keeps
    // it, the base validity of the topology is what removes it.
    let rows = [
        (1., [[0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0], [0; 6]]),
        (1.75, [[1; 6], [0, 0, 1, 1, 0, 0], [0; 6]]),
        (2.5, [[1; 6], [1, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 0]]),
        (4., [[1; 6], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]]),
    ];
    for (scale, [j, k, t]) in rows {
        assert_eq!(gate(&pair_j, scale), j);
        assert_eq!(gate(&pair_k, scale), k);
        assert_eq!(gate(&triplet, scale), t);
    }
    assert_eq!(gate(&pair_k, 1e300), [1, 1, 1, 1, 0, 0]);
    assert_eq!(gate(&pair_k, INF), [1, 1, 1, 1, 0, 0]);
}
#[test]
fn sites_always_pass_and_cloned_walkers_or_undefined_distances_never_do() {
    let sites: Vec<[u32; 3]> = (0..6).map(|i| [i; 3]).collect();
    let site = topology(ElementKind::Site, &sites);
    let cloned = [false, true, false, false, false, false];
    assert_eq!(
        scales::gate(&site, &D_GEODESIC, 6, 1e-9, &cloned),
        [true; 6]
    );
    let pairs = topology(
        ElementKind::DistancePair,
        &[[0, 1, 0], [1, 0, 1], [2, 3, 2], [0, 2, 0]],
    );
    assert_eq!(
        scales::gate(&pairs, &D_GEODESIC, 6, 10., &cloned),
        [false, false, true, true]
    );
    let triplet = topology(ElementKind::Triplet, &[[0, 2, 1], [0, 2, 3]]);
    assert_eq!(
        scales::gate(&triplet, &D_GEODESIC, 6, 10., &cloned),
        [false, true]
    );
    let mut undefined = D_GEODESIC;
    undefined[2 * 6 + 3] = f64::NAN;
    assert!(scales::diameter(&triplet.elements[1], &undefined, 6).is_nan());
    assert_eq!(
        scales::gate(&triplet, &undefined, 6, 10., &[false; 6]),
        [true, false]
    );
}
#[test]
fn triplet_diameter_is_the_largest_side_whichever_slots_span_it() {
    // Walkers 0, 1, 4 have the sides 1.5, 3.25 and 4.75: over the six orders
    // each of the slot pairs (i, j), (i, k) and (j, k) spans each side twice.
    let orders = [
        [0, 1, 4],
        [0, 4, 1],
        [1, 0, 4],
        [1, 4, 0],
        [4, 0, 1],
        [4, 1, 0],
    ];
    let triplet = topology(ElementKind::Triplet, &orders);
    for e in &triplet.elements {
        assert_eq!(scales::diameter(e, &D_GEODESIC, 6), 4.75);
    }
    let clean = [false; 6];
    assert_eq!(
        scales::gate(&triplet, &D_GEODESIC, 6, 4.75, &clean),
        [true; 6]
    );
    assert_eq!(
        scales::gate(&triplet, &D_GEODESIC, 6, 4.75f64.next_down(), &clean),
        [false; 6]
    );
    // An undefined side is never hidden by a finite one read after it.
    for [a, b] in [[0, 1], [1, 4], [0, 4]] {
        let mut undefined = D_GEODESIC;
        undefined[a * 6 + b] = f64::NAN;
        undefined[b * 6 + a] = f64::NAN;
        for e in &triplet.elements {
            assert!(scales::diameter(e, &undefined, 6).is_nan());
        }
        assert_eq!(
            scales::gate(&triplet, &undefined, 6, INF, &clean),
            [false; 6]
        );
        let pair = topology(
            ElementKind::CloningPair,
            &[
                [a as u32, b as u32, a as u32],
                [b as u32, a as u32, b as u32],
            ],
        );
        assert_eq!(scales::gate(&pair, &undefined, 6, INF, &clean), [false; 2]);
    }
}
#[test]
fn gates_grow_with_the_scale_and_keep_a_mutual_pairing_exchange_odd_sum_cancelled() {
    let n = 24;
    let (graph, length) = geometric(60, n, 0.35);
    let d = table(&graph, &length, INF);
    let ladder = scales::calibrate(
        &scales::pair_samples(&d, n, scales::CALIBRATION_SAMPLES),
        &ScaleSelection::Quantiles {
            count: 6,
            low: 0.05,
            high: 0.95,
        },
    )
    .unwrap();
    let mutual: Vec<[u32; 3]> = (0..n as u32).map(|i| [i, i ^ 1, i]).collect();
    let pairs = topology(ElementKind::DistancePair, &mutual);
    let color = unit_colors(3, n, 3);
    let mut previous = vec![false; n];
    let mut kept = Vec::new();
    for &scale in &ladder {
        let gate = scales::gate(&pairs, &d, n, scale, &vec![false; n]);
        let (mut odd, mut oriented) = (0., 0.);
        for (i, e) in pairs.elements.iter().enumerate() {
            let j = e.walkers[1] as usize;
            assert_eq!(gate[i], gate[j]);
            assert!(gate[i] || !previous[i]);
            if gate[i] {
                let q = dot(&color[3 * i..3 * i + 3], &color[3 * j..3 * j + 3]);
                odd += q.im;
                oriented += q.re * (unit(j as u64) - unit(i as u64));
            }
        }
        assert!(odd.abs() < 1e-14 * n as f64 && oriented.abs() < 1e-14 * n as f64);
        kept.push(gate.iter().filter(|g| **g).count());
        previous = gate;
    }
    assert!(kept[0] < kept[ladder.len() - 1]);
}

#[test]
fn a_pair_whose_two_path_sums_round_apart_is_gated_together_at_its_own_distance() {
    let mut apart = 0;
    for seed in 0..12 {
        let n = 12 + seed as usize;
        let (graph, length) = geometric(seed, n, 0.45);
        let rows: Vec<f64> = (0..n)
            .flat_map(|i| shortest_paths::from_source(&graph, &length, i, INF).unwrap())
            .collect();
        for i in 0..n {
            for j in (0..n).filter(|&j| rows[i * n + j] < rows[j * n + i]) {
                apart += 1;
                // The scale and the radius admit one summation order only.
                let scale = rows[i * n + j];
                let d = table(&graph, &length, scale);
                let mutual = topology(
                    ElementKind::DistancePair,
                    &[
                        [i as u32, j as u32, i as u32],
                        [j as u32, i as u32, j as u32],
                    ],
                );
                assert_eq!(
                    scales::gate(&mutual, &d, n, scale, &vec![false; n]),
                    [true; 2]
                );
                assert_eq!(
                    scales::gate(&mutual, &d, n, scale.next_down(), &vec![false; n]),
                    [false; 2]
                );
            }
        }
    }
    assert!(apart > 0);
}

#[test]
fn smeared_colours_match_the_hand_computed_field() {
    let (color, valid) = scales::smear(&g6_state(), &D_GEODESIC, 1.75);
    assert_eq!(valid, [true, true, true, true, false, true]);
    let re = [
        0.92353265033762,
        0.247014907132522,
        0.783726203021746,
        0.523055757705109,
        0.66762264627755,
        0.686336285557922,
        0.611942646428934,
        0.776423046390416,
        0.,
        0.,
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ];
    let im = [
        0.,
        0.293378730337907,
        0.,
        0.334941656156189,
        0.,
        0.28848311095091,
        0.,
        0.150643454937078,
        0.,
        0.,
        0.,
        0.,
    ];
    for (k, c) in color.iter().enumerate() {
        assert!((c.re - re[k]).abs() < 1e-13 && (c.im - im[k]).abs() < 1e-13);
    }
    for row in color
        .chunks_exact(2)
        .zip(&valid)
        .filter(|r| *r.1)
        .map(|r| r.0)
    {
        assert!((dot(row, row).re - 1.).abs() < 1e-14);
    }
}
#[test]
fn vanishing_scale_is_the_identity_and_zero_distance_is_the_renormalized_mean() {
    let state = g6_state();
    let (color, valid) = scales::smear(&state, &D_GEODESIC, 1e-3);
    assert_eq!(valid, state.color_valid);
    for (k, (a, b)) in color.iter().zip(&state.color).enumerate() {
        let expected = if state.color_valid[k / 2] {
            *b
        } else {
            C::ZERO
        };
        assert!((*a - expected).abs() < 1e-15);
    }
    let collapsed: Vec<f64> = D_GEODESIC
        .iter()
        .map(|&d| if d.is_finite() { 0. } else { d })
        .collect();
    let (color, _) = scales::smear(&state, &collapsed, 1.75);
    assert!((color[0] - C::new(0.794104487760818, 0.)).abs() < 1e-13);
    assert!((color[1] - C::new(0.561516668266344, 0.232587819494474)).abs() < 1e-13);
}
#[test]
fn gaussian_kernel_weight_and_its_cutoff_are_the_stated_ones() {
    let color = vec![C::ONE, C::ZERO, C::ZERO, C::ONE];
    let two = |reach: f64| {
        let (smeared, _) = scales::smear(
            &state(2, color.clone(), vec![true; 2]),
            &[0., reach, reach, 0.],
            1.75,
        );
        smeared[1].re / smeared[0].re
    };
    assert!((two(1.75) - 0.606530659712633).abs() < 1e-14);
    assert!((two(3.5) - 0.135335283236613).abs() < 1e-14);
    assert!((two(14.) - 1.26641655490942e-14).abs() < 1e-27);
    assert_eq!(two(14.000001), 0.);
}
#[test]
fn invalid_and_cloned_walkers_neither_give_nor_receive_a_smeared_colour() {
    let reference = scales::smear(&g6_state(), &D_GEODESIC, 1.75);
    let mut flipped = g6_state();
    flipped.color[8] = C::new(-0.6, 0.);
    flipped.color[9] = C::new(0., 0.8);
    assert_eq!(scales::smear(&flipped, &D_GEODESIC, 1.75), reference);
    let mut cloned = g6_state();
    cloned.color_valid[4] = true;
    cloned.cloned[4] = true;
    assert_eq!(scales::smear(&cloned, &D_GEODESIC, 1.75), reference);
    let mut donor = g6_state();
    donor.cloned[1] = true;
    let (color, valid) = scales::smear(&donor, &D_GEODESIC, 1.75);
    assert!(!valid[1] && color[2] == C::ZERO && color[3] == C::ZERO);
    assert!(valid[0] && color[0] != reference.0[0]);
}
#[test]
fn a_cancelling_kernel_sum_is_an_invalid_smeared_colour_not_a_zero_one() {
    let color = vec![C::ONE, C::ZERO, C::new(-1., 0.), C::ZERO, C::ZERO, C::ONE];
    let coincident = [0., 0., INF, 0., 0., INF, INF, INF, 0.];
    let (smeared, valid) = scales::smear(&state(2, color.clone(), vec![true; 3]), &coincident, 1.);
    assert_eq!(valid, [false, false, true]);
    assert_eq!(smeared[..4], [C::ZERO; 4]);
    assert_eq!(smeared[4..], color[4..]);
}
#[test]
fn a_nonfinite_colour_of_an_invalid_or_cloned_walker_never_leaks() {
    let g = g6();
    let smeared = scales::smear(&g6_state(), &D_GEODESIC, 1.75);
    let smoothed = curve(&g6_state(), &g.graph, 4, 0.25);
    let mut invalid = g6_state();
    invalid.color[8..10].fill(C::new(f64::NAN, f64::INFINITY));
    let mut cloned = invalid.clone();
    cloned.color_valid[4] = true;
    cloned.cloned[4] = true;
    for poisoned in [invalid, cloned] {
        // A leaked NaN compares unequal to everything.
        assert_eq!(scales::smear(&poisoned, &D_GEODESIC, 1.75), smeared);
        assert_eq!(bits(&curve(&poisoned, &g.graph, 4, 0.25)), bits(&smoothed));
    }
}
#[test]
fn smearing_commutes_with_a_global_rotation_and_fixes_a_constant_field() {
    let base = g6_state();
    let (smeared, _) = scales::smear(&base, &D_GEODESIC, 1.75);
    let turned = state(2, rotated(&base.color), base.color_valid.clone());
    let (color, valid) = scales::smear(&turned, &D_GEODESIC, 1.75);
    assert_eq!(valid, base.color_valid);
    for (a, b) in color.iter().zip(rotated(&smeared)) {
        assert!((*a - b).abs() < 1e-13);
    }
    let constant: Vec<C> = (0..6)
        .flat_map(|_| [C::new(0.6, 0.), C::new(0., 0.8)])
        .collect();
    let (color, valid) = scales::smear(&state(2, constant.clone(), vec![true; 6]), &D_GEODESIC, 2.);
    assert_eq!(valid, [true; 6]);
    for (a, b) in color.iter().zip(&constant) {
        assert!((*a - *b).abs() < 1e-15);
    }
}
#[test]
fn smearing_from_a_table_truncated_at_the_kernel_cutoff_is_exact() {
    let n = 30;
    let (graph, length) = geometric(70, n, 0.3);
    let smeared = state(3, unit_colors(5, n, 3), vec![true; n]);
    let scale = 0.04;
    let full = table(&graph, &length, INF);
    let truncated = table(&graph, &length, scales::SMEAR_CUTOFF * scale);
    assert!(
        truncated.iter().filter(|d| d.is_infinite()).count()
            > full.iter().filter(|d| d.is_infinite()).count()
    );
    assert_eq!(
        scales::smear(&smeared, &truncated, scale),
        scales::smear(&smeared, &full, scale)
    );
}

#[test]
fn two_node_roughness_follows_the_closed_form_relaxation() {
    let graph = NeighborGraph::from_undirected(2, &[[0, 1]]).unwrap();
    let closed = |overlap: f64, step_size: f64, steps: usize| -> Vec<f64> {
        (0..=steps)
            .map(|s| {
                let r = (1. - 2. * step_size).powi(2 * s as i32) * (1. - overlap) / (1. + overlap);
                2. * r / (1. + r)
            })
            .collect()
    };
    let orthogonal = state(2, vec![C::ONE, C::ZERO, C::ZERO, C::ONE], vec![true; 2]);
    let measured = curve(&orthogonal, &graph, 5, 0.1);
    close(&measured, &closed(0., 0.1, 5), 1e-14);
    let listed = [
        1.,
        0.780487804878049,
        0.581157775255392,
        0.415394756858172,
        0.287337146314569,
        0.193925746340391,
    ];
    close(&measured, &listed, 1e-14);
    let (sin, cos) = 2f64.sin_cos();
    let complex = state(
        2,
        vec![C::ONE, C::ZERO, C::new(cos, 0.), C::new(0., sin)],
        vec![true; 2],
    );
    let measured = curve(&complex, &graph, 3, 0.3);
    close(&measured, &closed(cos, 0.3, 3), 1e-14);
    let listed = [
        1.41614683654714,
        0.559163981062511,
        0.116926230258933,
        0.0196743866228237,
    ];
    close(&measured, &listed, 1e-14);
    let aligned = curve(&orthogonal, &graph, 2, 0.5);
    assert!(aligned[0] == 1. && aligned[1].abs() < 1e-15 && aligned[2].abs() < 1e-15);
}
#[test]
fn antipodal_colours_at_half_mixing_keep_their_colour_instead_of_dividing_by_zero() {
    let graph = NeighborGraph::from_undirected(2, &[[0, 1]]).unwrap();
    let antipodal = state(
        2,
        vec![C::ONE, C::ZERO, C::new(-1., 0.), C::ZERO],
        vec![true; 2],
    );
    assert_eq!(curve(&antipodal, &graph, 3, 0.5), [2.; 4]);
}
#[test]
fn graph_roughness_curves_match_the_hand_computed_vectors() {
    let g = g6();
    let masked = curve(&g6_state(), &g.graph, 4, 0.25);
    assert!((masked[0] - (4.5 - 2f64.sqrt()) / 5.).abs() < 1e-15);
    let listed = [
        0.617157287525381,
        0.359389290312452,
        0.185955036458343,
        0.090084804113954,
        0.042512616392993,
    ];
    close(&masked, &listed, 1e-13);
    let complete = curve(&state(2, g6_colors(), vec![true; 6]), &g.graph, 4, 0.25);
    assert!((complete[0] - (5.5 - 2f64.sqrt()) / 6.).abs() < 1e-15);
    let listed = [
        0.680964406271151,
        0.40239323474269,
        0.214849918661738,
        0.110884100556484,
        0.0582572935634911,
    ];
    close(&complete, &listed, 1e-13);
    let mut cloned = state(2, g6_colors(), vec![true; 6]);
    cloned.cloned[4] = true;
    assert_eq!(bits(&curve(&cloned, &g.graph, 4, 0.25)), bits(&masked));
}
#[test]
fn roughness_can_increase_under_full_mixing() {
    let graph = NeighborGraph::from_undirected(3, &[[0, 1], [1, 2]]).unwrap();
    let field = state(
        2,
        vec![C::ONE, C::ZERO, C::ZERO, C::new(-1., 0.), C::ZERO, C::ONE],
        vec![true; 3],
    );
    let measured = curve(&field, &graph, 1, 1.);
    close(&measured, &[1.5, 1.70710678118655], 1e-14);
}
#[test]
fn roughness_is_invariant_under_a_global_rotation_but_not_under_walker_rephasing() {
    let g = g6();
    let base = g6_state();
    let reference = curve(&base, &g.graph, 4, 0.25);
    let turned = state(2, rotated(&base.color), base.color_valid.clone());
    close(&curve(&turned, &g.graph, 4, 0.25), &reference, 1e-13);
    let phase = [0.3, -1.1, 2., 0.7, 0., 1.9];
    let rephased: Vec<C> = base
        .color
        .iter()
        .enumerate()
        .map(|(k, &c)| C::phase(phase[k / 2]) * c)
        .collect();
    let measured = curve(
        &state(2, rephased, base.color_valid.clone()),
        &g.graph,
        4,
        0.25,
    );
    let listed = [
        0.857588021706214,
        0.538248032148225,
        0.289075819685355,
        0.140909183521517,
        0.0659512547740298,
    ];
    close(&measured, &listed, 1e-13);
    assert!((measured[0] - reference[0]).abs() > 0.2);
}
#[test]
fn roughness_does_not_depend_on_the_node_labelling() {
    let g = g6();
    let reference = curve(&g6_state(), &g.graph, 4, 0.25);
    let mirrored: Vec<[u32; 2]> = G6_PAIRS.iter().map(|&[a, b]| [5 - a, 5 - b]).collect();
    let graph = NeighborGraph::from_undirected(6, &mirrored).unwrap();
    let base = g6_state();
    let color: Vec<C> = (0..6)
        .rev()
        .flat_map(|i| [base.color[2 * i], base.color[2 * i + 1]])
        .collect();
    let valid: Vec<bool> = base.color_valid.iter().rev().copied().collect();
    close(
        &curve(&state(2, color, valid), &graph, 4, 0.25),
        &reference,
        1e-13,
    );
}
#[test]
fn constant_field_has_no_roughness_up_to_rounding() {
    let g = g6();
    let constant: Vec<C> = (0..6)
        .flat_map(|_| [C::new(0.6, 0.), C::new(0., 0.8)])
        .collect();
    for r in curve(&state(2, constant, vec![true; 6]), &g.graph, 6, 0.3) {
        assert!(r.abs() <= 1e-15);
    }
}
#[test]
fn frames_without_a_valid_edge_report_no_roughness_instead_of_zero() {
    let g = g6();
    let config = FlowConfig {
        steps: 3,
        step_size: 0.1,
        every: 1,
    };
    let invalid = state(2, vec![C::ZERO; 12], vec![false; 6]);
    assert_eq!(
        flow::smooth(&invalid, &g.graph, &config).unwrap(),
        [None; 4]
    );
    let mut cloned = state(2, g6_colors(), vec![true; 6]);
    cloned.cloned = vec![false, true, true, true, false, false];
    assert_eq!(flow::smooth(&cloned, &g.graph, &config).unwrap(), [None; 4]);
    let edgeless = NeighborGraph::empty(6);
    assert_eq!(
        flow::smooth(&g6_state(), &edgeless, &config).unwrap(),
        [None; 4]
    );
}
#[test]
fn smoothing_rejects_an_invalid_configuration_or_a_mismatched_state_explicitly() {
    let g = g6();
    let unbounded = FlowConfig {
        steps: 3,
        step_size: 1.5,
        every: 1,
    };
    assert!(matches!(
        flow::smooth(&g6_state(), &g.graph, &unbounded),
        Err(GasError::Configuration(_))
    ));
    // Also on a frame without a valid edge, which returns before any sweep.
    let invalid = state(2, vec![C::ZERO; 12], vec![false; 6]);
    let idle = FlowConfig {
        steps: 0,
        step_size: 0.25,
        every: 1,
    };
    for config in [unbounded, idle] {
        assert!(matches!(
            flow::smooth(&invalid, &g.graph, &config),
            Err(GasError::Configuration(_))
        ));
    }
    let short = state(2, g6_colors()[..10].to_vec(), vec![true; 5]);
    assert!(matches!(
        flow::smooth(&short, &g.graph, &FlowConfig::default()),
        Err(GasError::Configuration(_))
    ));
    let mut unflagged = g6_state();
    unflagged.cloned.clear();
    assert!(matches!(
        flow::smooth(&unflagged, &g.graph, &FlowConfig::default()),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn a_tessellation_carried_over_from_an_earlier_step_is_declined() {
    let fresh = g6();
    assert_eq!(fresh.stale_steps, 0);
    let stale = GraphSnapshot {
        stale_steps: 1,
        ..g6()
    };
    assert!(stale.same_geometry(&fresh));
    for length in [EdgeLength::Euclidean, EdgeLength::Geodesic] {
        // The same geometry, so the table would be the same; what differs is
        // that the walkers of this frame are no longer the ones it describes.
        scales::distances(&fresh, length, INF, BYTES, Parallelism::Serial).unwrap();
        let declined = scales::distances(&stale, length, INF, BYTES, Parallelism::Serial)
            .err()
            .unwrap();
        assert!(
            matches!(declined, algorithmic_gas::GasError::Capability(ref reason)
                if reason.contains("refreshed every step")),
            "{declined}"
        );
    }
}
