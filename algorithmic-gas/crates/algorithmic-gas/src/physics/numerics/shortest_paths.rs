//! Radius-truncated shortest paths on a CSR neighbor graph: binary-heap
//! Dijkstra from every source. A distance is a sum of the given slot lengths
//! along graph edges; it is not a continuum geodesic distance.
use crate::{
    GasError, Result,
    error::require,
    memory::{checked_mul, enforce},
    tessellation::{NeighborGraph, Parallelism, par},
};
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

/// Tentative distance of a node; the node index breaks ties so that the pop
/// order is a function of the graph alone.
#[derive(Clone, Copy, Debug)]
struct Reach {
    distance: f64,
    node: u32,
}
impl PartialEq for Reach {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Reach {}
impl Ord for Reach {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.node.cmp(&other.node))
    }
}
impl PartialOrd for Reach {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
fn check(graph: &NeighborGraph, length: &[f64], radius: f64) -> Result<()> {
    graph.validate()?;
    require(
        length.len() == graph.edges() && radius >= 0.,
        "shortest paths need one length per edge slot and a nonnegative radius",
    )?;
    if length.iter().all(|l| l.is_finite() && *l >= 0.) {
        Ok(())
    } else {
        Err(GasError::Numerical(
            "nonfinite or negative edge length".into(),
        ))
    }
}
/// Lazy-deletion Dijkstra into `reach` (`[nodes]`) on a checked graph. A node
/// is relaxed only to a distance within `radius`, and every prefix of a
/// shortest path is no longer than the path, so the truncated result equals
/// the untruncated one wherever that is within `radius`.
fn dijkstra(graph: &NeighborGraph, length: &[f64], source: usize, radius: f64, reach: &mut [f64]) {
    debug_assert_eq!(reach.len(), graph.nodes());
    let neighbors = graph.neighbors();
    reach.fill(f64::INFINITY);
    reach[source] = 0.;
    let mut heap = BinaryHeap::from([Reverse(Reach {
        distance: 0.,
        node: source as u32,
    })]);
    while let Some(Reverse(Reach { distance, node })) = heap.pop() {
        if distance > reach[node as usize] {
            continue;
        }
        for e in graph.range(node as usize) {
            let next = distance + length[e];
            if next <= radius && next < reach[neighbors[e] as usize] {
                reach[neighbors[e] as usize] = next;
                heap.push(Reverse(Reach {
                    distance: next,
                    node: neighbors[e],
                }));
            }
        }
    }
}

/// Dijkstra distances from `source` along edges of CSR-slot lengths `length`;
/// `f64::INFINITY` beyond `radius` (inclusive) or when unreachable. `[nodes]`.
/// An infinite radius does not truncate.
pub fn from_source(
    graph: &NeighborGraph,
    length: &[f64],
    source: usize,
    radius: f64,
) -> Result<Vec<f64>> {
    check(graph, length, radius)?;
    require(source < graph.nodes(), "shortest path source out of range")?;
    let mut reach = vec![0.; graph.nodes()];
    dijkstra(graph, length, source, radius, &mut reach);
    Ok(reach)
}

/// All sources through `tessellation::par::fill_rows`; `[nodes, nodes]`, row =
/// source. The dense table is admitted against `max_bytes` with
/// `memory::enforce(nodes · nodes · 8, max_bytes)` before it is allocated.
/// Lengths must be finite and nonnegative, one per CSR slot. The table is
/// bit-symmetric: a path summed from either end differs by rounding, which
/// would let a truncation or a gate keep one direction of a pair only, so
/// each entry is the smaller of the two directions.
pub fn all_pairs(
    graph: &NeighborGraph,
    length: &[f64],
    radius: f64,
    max_bytes: usize,
    par: Parallelism,
) -> Result<Vec<f64>> {
    check(graph, length, radius)?;
    let n = graph.nodes();
    enforce(checked_mul(checked_mul(n, n)?, 8)?, max_bytes)?;
    // A source is a whole Dijkstra run, a coarse task as in `par::map_tasks`.
    let par = match par {
        Parallelism::Serial => Parallelism::Serial,
        _ => Parallelism::Always,
    };
    let mut table = vec![0.; n * n];
    par::fill_rows(&mut table, n, par, |source, reach| {
        dijkstra(graph, length, source, radius, reach);
    });
    for i in 0..n {
        for j in i + 1..n {
            let d = table[i * n + j].min(table[j * n + i]);
            table[i * n + j] = d;
            table[j * n + i] = d;
        }
    }
    Ok(table)
}
