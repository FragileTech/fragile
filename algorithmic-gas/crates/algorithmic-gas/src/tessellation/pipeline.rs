//! Composition of the geometry stage: sites, tessellation, optional Voronoi
//! cells, metric, volume element, edge weights and curvature estimators. Every
//! role is a config enum that can be exchanged independently; `evaluate_with`
//! accepts a custom tessellator.
use super::{
    curvature::{
        CurvatureContext, CurvatureEstimator, CurvatureKind, CurvatureOutput, CurvatureSpec,
    },
    degenerate::{DegeneracyPolicy, Tessellation, tessellate},
    domain::TessellationDomain,
    frame::{EdgeLengths, GeometryFrame},
    mesh::{Tessellator, TessellatorKind},
    metric::{MetricEstimator, MetricField, MetricKind},
    par::Parallelism,
    sites::{Projection, SiteSet},
    volume::{VolumeElement, VolumeKind},
    voronoi::{self, VoronoiCells, VoronoiCellsConfig},
    weights::{EdgeContext, EdgeWeighting, WeightMode, WeightSpec},
};
use crate::{GasError, ObservationBatch, Real, Result, error::require};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeometryPipelineConfig {
    /// Observation field holding one position vector per walker.
    pub positions: String,
    pub projection: Projection,
    pub domain: TessellationDomain,
    pub tessellator: TessellatorKind,
    pub degeneracy: DegeneracyPolicy,
    pub metric: MetricKind,
    pub volume: VolumeKind,
    pub weights: Vec<WeightSpec>,
    pub curvature: Vec<CurvatureSpec>,
    /// Voronoi cells are built when a component needs them or this is set.
    pub cells: Option<VoronoiCellsConfig>,
    pub parallelism: Parallelism,
}
impl Default for GeometryPipelineConfig {
    fn default() -> Self {
        Self {
            positions: "positions".into(),
            projection: Projection::Full,
            domain: TessellationDomain::Open,
            tessellator: TessellatorKind::Auto,
            degeneracy: DegeneracyPolicy::default(),
            metric: MetricKind::default(),
            volume: VolumeKind::default(),
            weights: vec![WeightSpec::new(WeightMode::InverseRiemannianDistance)],
            curvature: vec![CurvatureSpec {
                name: "ricci_scalar".into(),
                estimator: CurvatureKind::ConformalLaplacian {
                    weights: WeightMode::InverseRiemannianDistance.name().into(),
                    det_floor: 1e-12,
                },
            }],
            cells: None,
            parallelism: Parallelism::Auto,
        }
    }
}

/// Everything the geometry stage derived from one population.
#[derive(Clone, Debug, PartialEq)]
pub struct TessellationGeometry<T: Real> {
    /// Tessellated and ambient dimensions and the projected coordinate axes.
    pub dimension: usize,
    pub ambient: usize,
    pub axes: Vec<usize>,
    /// (ambient axis, box length) of every periodic coordinate.
    pub wrap: Vec<(usize, f64)>,
    pub tessellation: Tessellation,
    pub metric: MetricField<T>,
    pub lengths: EdgeLengths<T>,
    pub volume: Vec<T>,
    pub weights: BTreeMap<String, Vec<T>>,
    pub curvature: BTreeMap<String, CurvatureOutput<T>>,
    pub cells: Option<VoronoiCells<T>>,
}
impl<T: Real> TessellationGeometry<T> {
    pub fn graph(&self) -> &super::graph::NeighborGraph {
        &self.tessellation.graph
    }
    /// Diffusion factors g^{-1/2} embedded in the ambient space: identity on
    /// the coordinates outside the tessellated subspace.
    pub fn ambient_diffusion(&self) -> Vec<T> {
        let (d, a) = (self.dimension, self.ambient);
        let n = self.volume.len();
        let mut out = vec![T::ZERO; n * a * a];
        for i in 0..n {
            for k in 0..a {
                out[i * a * a + k * a + k] = T::ONE;
            }
            for (p, &ap) in self.axes.iter().enumerate() {
                for (q, &aq) in self.axes.iter().enumerate() {
                    out[i * a * a + ap * a + aq] = self.metric.diffusion[i * d * d + p * d + q];
                }
            }
        }
        out
    }
}

impl GeometryPipelineConfig {
    fn needs_cells<T: Real>(&self) -> bool {
        self.cells.is_some()
            || MetricEstimator::<T>::needs_cells(&self.metric)
            || VolumeElement::<T>::needs_cells(&self.volume)
            || self
                .weights
                .iter()
                .any(|w| EdgeWeighting::<T>::needs_cells(w))
            || self
                .curvature
                .iter()
                .any(|c| CurvatureEstimator::<T>::requires(&c.estimator).cells)
    }
    /// True when an estimator reads the cell volumes of the previous evaluation.
    pub fn needs_previous_cell_volume<T: Real>(&self) -> bool {
        self.curvature
            .iter()
            .any(|c| CurvatureEstimator::<T>::requires(&c.estimator).previous_cell_volume)
    }
    pub fn validate<T: Real>(&self, ambient: usize) -> Result<()> {
        require(!self.positions.is_empty(), "geometry position field name")?;
        let axes = self.projection.axes(ambient)?;
        require(
            self.tessellator.supports(axes.len()),
            format!(
                "tessellator {:?} cannot tessellate {} projected coordinates",
                self.tessellator,
                axes.len()
            ),
        )?;
        self.domain.validate(axes.len())?;
        self.metric.validate()?;
        self.volume.validate()?;
        let mut keys = BTreeSet::new();
        for w in &self.weights {
            w.validate()?;
            require(
                keys.insert(w.key()),
                format!("duplicate edge weight name {:?}", w.key()),
            )?;
        }
        let mut names = BTreeSet::new();
        for c in &self.curvature {
            c.estimator.validate()?;
            require(
                !c.name.is_empty() && names.insert(c.name.clone()),
                format!("curvature names must be unique and non-empty: {:?}", c.name),
            )?;
            if let Some(w) = CurvatureEstimator::<T>::requires(&c.estimator).weights {
                require(
                    keys.contains(&w),
                    format!(
                        "curvature {:?} needs edge weights {w:?}, which are not configured",
                        c.name
                    ),
                )?;
            }
        }
        Ok(())
    }
    pub fn evaluate<T: Real>(
        &self,
        observations: &ObservationBatch<T>,
        eligible: &[bool],
        previous_cell_volume: Option<&[T]>,
        max_edges: usize,
    ) -> Result<TessellationGeometry<T>> {
        self.evaluate_with(
            &self.tessellator,
            observations,
            eligible,
            previous_cell_volume,
            max_edges,
        )
    }
    pub fn evaluate_with<T: Real>(
        &self,
        tessellator: &dyn Tessellator,
        observations: &ObservationBatch<T>,
        eligible: &[bool],
        previous_cell_volume: Option<&[T]>,
        max_edges: usize,
    ) -> Result<TessellationGeometry<T>> {
        let par = self.parallelism;
        let positions = observations.field(&self.positions)?;
        let ambient = positions.width();
        let axes = self.projection.axes(ambient)?;
        let wrap = match &self.domain {
            TessellationDomain::Periodic { bounds } => Some(bounds),
            _ => None,
        };
        let sites = SiteSet::build(positions, &axes, eligible, wrap)?;
        let tessellation = tessellate(
            sites,
            tessellator,
            &self.degeneracy,
            &self.domain,
            max_edges,
        )?;
        let period = self
            .domain
            .period()
            .map(|p| p.into_iter().map(T::from_f64).collect());
        let frame = GeometryFrame::new(
            positions,
            &axes,
            eligible,
            &tessellation.graph,
            observations,
            period,
        )?;
        let cells = if self.needs_cells::<T>() {
            Some(voronoi::cells::<T>(
                &tessellation,
                &self.domain,
                &self.cells.clone().unwrap_or_default(),
                &frame.sources,
                par,
            )?)
        } else {
            None
        };
        let metric = self.metric.estimate(&frame, cells.as_ref(), par)?;
        let lengths = EdgeLengths::compute(&frame, &metric, par);
        let volume = self.volume.volume(&metric, cells.as_ref())?;
        let mut weights = BTreeMap::new();
        for spec in &self.weights {
            let cx = EdgeContext {
                frame: &frame,
                lengths: &lengths,
                volume: &volume,
                cells: cells.as_ref(),
            };
            weights.insert(spec.key(), spec.weights(&cx, par)?);
        }
        let mut curvature = BTreeMap::new();
        for spec in &self.curvature {
            let cx = CurvatureContext {
                frame: &frame,
                tessellation: &tessellation,
                metric: &metric,
                lengths: &lengths,
                volume: &volume,
                weights: &weights,
                cells: cells.as_ref(),
                previous_cell_volume,
            };
            let out = spec.estimator.curvature(&cx, par)?;
            if out.scalar.iter().any(|v| !v.is_finite()) {
                return Err(GasError::Numerical(format!(
                    "curvature estimator {:?} produced a non-finite value",
                    spec.name
                )));
            }
            curvature.insert(spec.name.clone(), out);
        }
        let dimension = frame.dimension;
        drop(frame);
        let wrap = self
            .domain
            .period()
            .map(|lengths| axes.iter().copied().zip(lengths).collect())
            .unwrap_or_default();
        Ok(TessellationGeometry {
            dimension,
            ambient,
            axes,
            wrap,
            tessellation,
            metric,
            lengths,
            volume,
            weights,
            curvature,
            cells,
        })
    }
}
