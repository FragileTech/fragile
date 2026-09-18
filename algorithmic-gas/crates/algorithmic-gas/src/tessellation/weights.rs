//! Edge weightings of the neighbor graph. Every scheme yields one weight per
//! directed edge slot; normalized schemes sum to one over each source walker.
use super::{
    frame::{EdgeLengths, GeometryFrame},
    par::{Parallelism, map_indexed},
    voronoi::VoronoiCells,
};
use crate::{GasError, Real, Result, error::require};
use serde::{Deserialize, Serialize};

pub struct EdgeContext<'a, T: Real> {
    pub frame: &'a GeometryFrame<'a, T>,
    pub lengths: &'a EdgeLengths<T>,
    /// The pipeline's volume element, sqrt(det g) by default.
    pub volume: &'a [T],
    pub cells: Option<&'a VoronoiCells<T>>,
}

pub trait EdgeWeighting<T: Real> {
    fn needs_cells(&self) -> bool {
        false
    }
    fn weights(&self, cx: &EdgeContext<'_, T>, par: Parallelism) -> Result<Vec<T>>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightMode {
    Uniform,
    /// 1 / (|dx| + 1e-8)
    InverseDistance,
    /// 1 / (V_j + 1e-12) with unit coordinate cells.
    InverseVolume,
    /// 1 / (volume_j + 1e-12) with the pipeline's volume element.
    InverseRiemannianVolume,
    /// 1 / (sqrt(max(d_g^2, 1e-8)) + 1e-8)
    InverseRiemannianDistance,
    /// exp(-|dx|^2 / 2 l^2)
    Kernel,
    /// exp(-d_g^2 / 2 l^2)
    RiemannianKernel,
    /// exp(-d_g^2 / 2 l^2) volume_j: a lattice sum with the Riemannian measure.
    RiemannianKernelVolume,
    /// Voronoi facet area A_ij.
    FacetArea,
    /// A_ij / |dx_ij|: the finite-volume (cotangent) Laplacian weight.
    FacetAreaOverDistance,
    /// A_ij / sqrt(V_i V_j)
    FacetAreaOverVolume,
}
impl WeightMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::InverseDistance => "inverse_distance",
            Self::InverseVolume => "inverse_volume",
            Self::InverseRiemannianVolume => "inverse_riemannian_volume",
            Self::InverseRiemannianDistance => "inverse_riemannian_distance",
            Self::Kernel => "kernel",
            Self::RiemannianKernel => "riemannian_kernel",
            Self::RiemannianKernelVolume => "riemannian_kernel_volume",
            Self::FacetArea => "facet_area",
            Self::FacetAreaOverDistance => "facet_area_over_distance",
            Self::FacetAreaOverVolume => "facet_area_over_volume",
        }
    }
}

fn yes() -> bool {
    true
}
fn one() -> f64 {
    1.
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightSpec {
    pub mode: WeightMode,
    /// Key under which the weights are stored; defaults to the mode name.
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "yes")]
    pub normalize: bool,
    #[serde(default = "one")]
    pub length_scale: f64,
}
impl WeightSpec {
    pub fn new(mode: WeightMode) -> Self {
        Self {
            mode,
            name: None,
            normalize: true,
            length_scale: 1.,
        }
    }
    pub fn key(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.mode.name().into())
    }
    pub fn validate(&self) -> Result<()> {
        require(
            self.length_scale.is_finite() && self.length_scale > 0.,
            "edge weight length scale must be positive",
        )?;
        require(
            self.name.as_ref().is_none_or(|n| !n.is_empty()),
            "edge weight name must not be empty",
        )
    }
}
/// Divide every edge weight by the sum over its source walker (floored at 1e-12).
pub fn normalize_rows<T: Real>(offsets: &[u32], raw: &mut [T]) {
    for w in offsets.windows(2) {
        let row = &mut raw[w[0] as usize..w[1] as usize];
        let sum = row
            .iter()
            .fold(T::ZERO, |s, &v| s + v)
            .max(T::from_f64(1e-12));
        for v in row {
            *v = *v / sum;
        }
    }
}
impl<T: Real> EdgeWeighting<T> for WeightSpec {
    fn needs_cells(&self) -> bool {
        matches!(
            self.mode,
            WeightMode::FacetArea
                | WeightMode::FacetAreaOverDistance
                | WeightMode::FacetAreaOverVolume
        )
    }
    fn weights(&self, cx: &EdgeContext<'_, T>, par: Parallelism) -> Result<Vec<T>> {
        let graph = cx.frame.graph;
        let cells = || {
            cx.cells.ok_or_else(|| {
                GasError::Configuration(format!(
                    "edge weighting {} requires Voronoi cells",
                    self.mode.name()
                ))
            })
        };
        let facets = match self.mode {
            WeightMode::FacetArea
            | WeightMode::FacetAreaOverDistance
            | WeightMode::FacetAreaOverVolume => Some(cells()?),
            _ => None,
        };
        let cell_volume = match self.mode {
            WeightMode::FacetAreaOverVolume => Some(super::volume::filled_cell_volumes(cells()?)),
            _ => None,
        };
        let two_l2 = T::from_f64(2. * self.length_scale * self.length_scale);
        let tiny = T::from_f64(1e-12);
        let small = T::from_f64(1e-8);
        let mut raw = map_indexed(graph.edges(), par, |e| {
            let (i, j) = (cx.frame.sources[e] as usize, graph.neighbors()[e] as usize);
            let euclid = cx.lengths.euclidean[e];
            let geo_sq = cx.lengths.geodesic_sq[e];
            match self.mode {
                WeightMode::Uniform => T::ONE,
                WeightMode::InverseDistance => T::ONE / (euclid + small),
                WeightMode::InverseVolume => T::ONE / (T::ONE + tiny),
                WeightMode::InverseRiemannianVolume => T::ONE / (cx.volume[j] + tiny),
                WeightMode::InverseRiemannianDistance => {
                    T::ONE / (geo_sq.max(small).sqrt() + small)
                }
                WeightMode::Kernel => (-(euclid * euclid) / two_l2).exp(),
                WeightMode::RiemannianKernel => (-geo_sq / two_l2).exp(),
                WeightMode::RiemannianKernelVolume => (-geo_sq / two_l2).exp() * cx.volume[j],
                WeightMode::FacetArea => facets.map_or(T::ZERO, |c| c.facet_area[e]),
                WeightMode::FacetAreaOverDistance => {
                    facets.map_or(T::ZERO, |c| c.facet_area[e]) / (euclid + small)
                }
                WeightMode::FacetAreaOverVolume => {
                    let v = cell_volume.as_deref().map_or(T::ONE, |v| v[i] * v[j]);
                    facets.map_or(T::ZERO, |c| c.facet_area[e]) / v.max(tiny).sqrt()
                }
            }
        });
        if self.normalize {
            normalize_rows(graph.offsets(), &mut raw);
        }
        Ok(raw)
    }
}
