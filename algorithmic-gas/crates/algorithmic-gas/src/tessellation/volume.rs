//! The integration measure attached to every walker.
use super::{metric::MetricField, voronoi::VoronoiCells};
use crate::{GasError, Real, Result, error::require};
use serde::{Deserialize, Serialize};

pub trait VolumeElement<T: Real> {
    fn needs_cells(&self) -> bool {
        false
    }
    fn volume(&self, metric: &MetricField<T>, cells: Option<&VoronoiCells<T>>) -> Result<Vec<T>>;
}

fn default_det_floor() -> f64 {
    1e-12
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VolumeKind {
    /// sqrt(max(det g, floor)): the Riemannian density of a unit coordinate cell.
    SqrtDetMetric {
        #[serde(default = "default_det_floor")]
        det_floor: f64,
    },
    /// Euclidean Voronoi cell volume; unbounded or empty cells take the mean
    /// of the bounded cells.
    VoronoiCell,
    /// Voronoi cell volume times sqrt(det g).
    RiemannianCell {
        #[serde(default = "default_det_floor")]
        det_floor: f64,
    },
    Unit,
}
impl Default for VolumeKind {
    fn default() -> Self {
        Self::SqrtDetMetric {
            det_floor: default_det_floor(),
        }
    }
}
impl VolumeKind {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::SqrtDetMetric { det_floor } | Self::RiemannianCell { det_floor } => require(
                det_floor.is_finite() && *det_floor > 0.,
                "determinant floor must be positive",
            ),
            Self::VoronoiCell | Self::Unit => Ok(()),
        }
    }
}
fn density<T: Real>(metric: &MetricField<T>, floor: f64) -> Vec<T> {
    metric
        .determinant
        .iter()
        .map(|&v| v.max(T::from_f64(floor)).sqrt())
        .collect()
}
/// Cell volumes with unbounded and empty cells replaced by the bounded mean
/// (one when no cell is bounded).
pub fn filled_cell_volumes<T: Real>(cells: &VoronoiCells<T>) -> Vec<T> {
    let mut sum = T::ZERO;
    let mut count = 0usize;
    for (i, &v) in cells.volume.iter().enumerate() {
        if cells.bounded[i] && v > T::ZERO {
            sum = sum + v;
            count += 1;
        }
    }
    let mean = if count > 0 {
        sum / T::from_f64(count as f64)
    } else {
        T::ONE
    };
    cells
        .volume
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if cells.bounded[i] && v > T::ZERO {
                v
            } else {
                mean
            }
        })
        .collect()
}
impl<T: Real> VolumeElement<T> for VolumeKind {
    fn needs_cells(&self) -> bool {
        matches!(self, Self::VoronoiCell | Self::RiemannianCell { .. })
    }
    fn volume(&self, metric: &MetricField<T>, cells: Option<&VoronoiCells<T>>) -> Result<Vec<T>> {
        let need = || {
            cells.ok_or_else(|| {
                GasError::Configuration("this volume element requires Voronoi cells".into())
            })
        };
        Ok(match self {
            Self::SqrtDetMetric { det_floor } => density(metric, *det_floor),
            Self::Unit => vec![T::ONE; metric.walkers()],
            Self::VoronoiCell => filled_cell_volumes(need()?),
            Self::RiemannianCell { det_floor } => filled_cell_volumes(need()?)
                .into_iter()
                .zip(density(metric, *det_floor))
                .map(|(v, s)| v * s)
                .collect(),
        })
    }
}
