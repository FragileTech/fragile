//! Engine glue of the geometry stage. The stage writes its per-walker results
//! into observation fields, which makes them part of the population: they are
//! checkpointed with it, copied from donor to clone by literal cloning, and
//! visible to every reward source, noise geometry and recorder without any
//! further interface. The neighbor graph, which is not per-walker data, is kept
//! as a `GraphSnapshot` for the graph forces of the kinetic operator.
use super::{
    graph::NeighborGraph,
    pipeline::{GeometryPipelineConfig, TessellationGeometry},
};
use crate::{GasError, Population, Real, Result, TensorBatch, error::require};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const VOLUME_FIELD: &str = "geometry.volume_element";
pub const DIFFUSION_FIELD: &str = "geometry.diffusion";
pub const CELL_VOLUME_FIELD: &str = "geometry.cell_volume";
pub fn curvature_field(name: &str) -> String {
    format!("geometry.curvature.{name}")
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryTiming {
    /// Tessellate the post-cloning population only. The pre-cloning reward of
    /// a step then reads the geometry of the previous step's post-cloning state.
    #[default]
    AfterCloning,
    /// Also tessellate at the start of the step, so the pre-cloning reward
    /// sees the geometry of the current positions.
    Both,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeometryStageConfig {
    pub pipeline: GeometryPipelineConfig,
    pub timing: GeometryTiming,
    /// Post-cloning refresh period in steps.
    pub refresh_every: u64,
    /// Refresh on any step that cloned or revived a walker, whatever the period.
    pub refresh_on_clone: bool,
    /// Write g^{-1/2}, embedded in the ambient space, as `geometry.diffusion`.
    pub write_diffusion: bool,
    /// Store the neighbor graph and its edge arrays in every recorded step.
    /// Per-walker results are observation fields and are always recorded.
    pub record_graph: bool,
}
impl Default for GeometryStageConfig {
    fn default() -> Self {
        Self {
            pipeline: GeometryPipelineConfig::default(),
            timing: GeometryTiming::AfterCloning,
            refresh_every: 1,
            refresh_on_clone: true,
            write_diffusion: true,
            record_graph: false,
        }
    }
}

/// The neighbor graph and its per-edge arrays: what graph forces consume and
/// what a recorder stores. Edge arrays follow the CSR slot order of `graph`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct GraphSnapshot<T: Real> {
    pub graph: NeighborGraph,
    pub weights: BTreeMap<String, Vec<T>>,
    pub euclidean_length: Vec<T>,
    pub geodesic_length: Vec<T>,
    /// (ambient axis, box length) of every periodic coordinate.
    #[serde(default)]
    pub wrap: Vec<(usize, f64)>,
}
impl<T: Real> GraphSnapshot<T> {
    pub fn of(geometry: &TessellationGeometry<T>) -> Self {
        Self {
            graph: geometry.graph().clone(),
            weights: geometry.weights.clone(),
            euclidean_length: geometry.lengths.euclidean.clone(),
            geodesic_length: geometry.lengths.geodesic(),
            wrap: geometry.wrap.clone(),
        }
    }
    pub fn validate(&self, walkers: usize) -> Result<()> {
        self.graph.validate()?;
        let e = self.graph.edges();
        require(
            self.graph.nodes() == walkers
                && self.euclidean_length.len() == e
                && self.geodesic_length.len() == e
                && self.weights.values().all(|w| w.len() == e),
            "graph snapshot shape",
        )
    }
    pub fn buffer_bytes(&self) -> usize {
        let e = self.graph.edges();
        (self.graph.nodes() + 1 + 2 * e) * 4 + (2 + self.weights.len()) * e * size_of::<T>()
    }
}

impl GeometryStageConfig {
    pub fn validate<T: Real>(&self, p: &Population<T>) -> Result<()> {
        require(self.refresh_every >= 1, "geometry refresh period")?;
        let x = p.observations.field(&self.pipeline.positions)?;
        require(
            x.item_shape().len() == 1,
            "geometry positions must be one vector per walker",
        )?;
        self.pipeline.validate::<T>(x.width())
    }
    fn outputs<T: Real>(&self, ambient: usize) -> Vec<(String, Vec<usize>, T)> {
        let mut out: Vec<(String, Vec<usize>, T)> = self
            .pipeline
            .curvature
            .iter()
            .map(|c| (curvature_field(&c.name), vec![], T::ZERO))
            .collect();
        out.push((VOLUME_FIELD.into(), vec![], T::ONE));
        if self.write_diffusion {
            out.push((DIFFUSION_FIELD.into(), vec![ambient, ambient], T::ZERO));
        }
        if self.pipeline.needs_previous_cell_volume::<T>() {
            out.push((CELL_VOLUME_FIELD.into(), vec![], T::ZERO));
        }
        out
    }
    /// Insert the stage's observation fields where they are missing, so the
    /// population schema is fixed before the first evaluation.
    pub fn prepare<T: Real>(&self, p: &mut Population<T>) -> Result<()> {
        let n = p.len();
        let ambient = p.observations.field(&self.pipeline.positions)?.width();
        for (name, shape, fill) in self.outputs::<T>(ambient) {
            if p.observations.fields.contains_key(&name) {
                continue;
            }
            let width: usize = shape.iter().product();
            let mut values = vec![fill; n * width];
            if name == DIFFUSION_FIELD {
                for i in 0..n {
                    for k in 0..ambient {
                        values[i * width + k * ambient + k] = T::ONE;
                    }
                }
            }
            p.observations
                .fields
                .insert(name, TensorBatch::new(n, shape, values)?);
        }
        Ok(())
    }
    /// True when the post-cloning stage of step `step` must tessellate.
    pub fn due(&self, step: u64, cloned: bool, cached: bool) -> bool {
        !cached
            || (step - 1).is_multiple_of(self.refresh_every)
            || (cloned && self.refresh_on_clone)
    }
    /// Evaluate the pipeline on `p` and write its per-walker results back.
    pub fn refresh<T: Real>(
        &self,
        p: &mut Population<T>,
        include_truncated: bool,
        max_edges: usize,
        max_memory_bytes: usize,
    ) -> Result<TessellationGeometry<T>> {
        self.prepare(p)?;
        let n = p.len();
        let eligible = p.eligible(include_truncated);
        // Sites, mesh, graph and per-edge arrays: a conservative linear bound;
        // lifted cliques are bounded separately by `max_edges`.
        crate::memory::enforce(
            crate::memory::checked_mul(n, 4096)?,
            max_memory_bytes.max(1),
        )?;
        let previous = if self.pipeline.needs_previous_cell_volume::<T>() {
            Some(p.observations.field(CELL_VOLUME_FIELD)?.values().to_vec())
        } else {
            None
        };
        let geometry =
            self.pipeline
                .evaluate(&p.observations, &eligible, previous.as_deref(), max_edges)?;
        let mut write = |name: &str, shape: Vec<usize>, values: Vec<T>| -> Result<()> {
            if values.iter().any(|v| !v.is_finite()) {
                return Err(GasError::Numerical(format!(
                    "geometry stage produced a non-finite {name}"
                )));
            }
            p.observations
                .fields
                .insert(name.into(), TensorBatch::new(n, shape, values)?);
            Ok(())
        };
        for (name, out) in &geometry.curvature {
            write(&curvature_field(name), vec![], out.scalar.clone())?;
        }
        write(VOLUME_FIELD, vec![], geometry.volume.clone())?;
        if self.write_diffusion {
            let a = geometry.ambient;
            write(DIFFUSION_FIELD, vec![a, a], geometry.ambient_diffusion())?;
        }
        if previous.is_some() {
            let cells = geometry.cells.as_ref().ok_or_else(|| {
                GasError::Configuration("expansion rates need Voronoi cells".into())
            })?;
            write(CELL_VOLUME_FIELD, vec![], cells.volume.clone())?;
        }
        Ok(geometry)
    }
}
