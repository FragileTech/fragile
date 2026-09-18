//! Shared read-only view every geometry component works from: projected
//! positions in the run dtype, eligibility, the neighbor graph and the
//! observation batch for components that read auxiliary fields.
use super::{
    graph::NeighborGraph,
    linalg::bilinear,
    metric::MetricField,
    par::{Parallelism, map_indexed},
};
use crate::{ObservationBatch, Real, Result, TensorBatch, error::require};

pub struct GeometryFrame<'a, T: Real> {
    pub dimension: usize,
    /// `[walkers, dimension]` projected positions.
    pub x: Vec<T>,
    pub eligible: &'a [bool],
    pub graph: &'a NeighborGraph,
    pub observations: &'a ObservationBatch<T>,
    /// Box lengths of a periodic domain; displacements use the minimum image.
    pub period: Option<Vec<T>>,
    /// Source walker of every edge slot.
    pub sources: Vec<u32>,
}
impl<'a, T: Real> GeometryFrame<'a, T> {
    pub fn new(
        positions: &TensorBatch<T>,
        axes: &[usize],
        eligible: &'a [bool],
        graph: &'a NeighborGraph,
        observations: &'a ObservationBatch<T>,
        period: Option<Vec<T>>,
    ) -> Result<Self> {
        require(
            graph.nodes() == positions.rows() && eligible.len() == positions.rows(),
            "geometry frame walker count",
        )?;
        let width = positions.width();
        let x = positions
            .values()
            .chunks_exact(width)
            .flat_map(|row| axes.iter().map(|&a| row[a]))
            .collect();
        Ok(Self {
            dimension: axes.len(),
            x,
            eligible,
            graph,
            observations,
            period,
            sources: graph.sources(),
        })
    }
    pub fn walkers(&self) -> usize {
        self.eligible.len()
    }
    pub fn position(&self, i: usize) -> &[T] {
        &self.x[i * self.dimension..(i + 1) * self.dimension]
    }
    /// x_j - x_i, by the minimum image in a periodic domain.
    pub fn delta(&self, i: usize, j: usize) -> Vec<T> {
        let (a, b) = (self.position(i), self.position(j));
        let mut out: Vec<T> = b.iter().zip(a).map(|(&q, &p)| q - p).collect();
        if let Some(period) = &self.period {
            let half = T::from_f64(0.5);
            for (v, &l) in out.iter_mut().zip(period) {
                *v = *v - l * (*v / l + half).floor();
            }
        }
        out
    }
}

/// Euclidean length and squared metric length of every edge slot. The edge
/// metric is the mean of its endpoint metrics, so both directions agree.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeLengths<T: Real> {
    pub euclidean: Vec<T>,
    pub geodesic_sq: Vec<T>,
}
impl<T: Real> EdgeLengths<T> {
    pub fn compute(
        frame: &GeometryFrame<'_, T>,
        metric: &MetricField<T>,
        par: Parallelism,
    ) -> Self {
        let d = frame.dimension;
        let half = T::from_f64(0.5);
        let pairs = map_indexed(frame.graph.edges(), par, |e| {
            let (i, j) = (
                frame.sources[e] as usize,
                frame.graph.neighbors()[e] as usize,
            );
            let dx = frame.delta(i, j);
            let euclidean = dx.iter().fold(T::ZERO, |s, &v| s + v * v).sqrt();
            let g: Vec<T> = metric
                .tensor(i)
                .iter()
                .zip(metric.tensor(j))
                .map(|(&a, &b)| half * (a + b))
                .collect();
            debug_assert_eq!(g.len(), d * d);
            (euclidean, bilinear(&g, &dx, &dx).max(T::ZERO))
        });
        let (euclidean, geodesic_sq) = pairs.into_iter().unzip();
        Self {
            euclidean,
            geodesic_sq,
        }
    }
    /// Geodesic edge length with the recorded floor of 1e-12 on its square.
    pub fn geodesic(&self) -> Vec<T> {
        self.geodesic_sq
            .iter()
            .map(|&v| v.max(T::from_f64(1e-12)).sqrt())
            .collect()
    }
}
