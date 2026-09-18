//! Population-geometry rewards. The allocation rule turns curvature and volume
//! into one action contribution per walker; a single broadcast total would
//! give cloning nothing to discriminate on after standardization.
use super::stage::{VOLUME_FIELD, curvature_field};
use crate::{
    ExecutionContext, InputBatch, Population, Provenance, Real, Result, RewardBatch, TensorBatch,
    domain::{GradientProvider, OperatorFuture, RewardSource},
    error::require,
};
use serde::{Deserialize, Serialize};

pub trait RewardAllocation<T: Real> {
    fn allocate(&self, curvature: &[T], volume: &[T], eligible: &[bool]) -> Result<Vec<T>>;
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RewardAllocationKind {
    /// r_i = scale R_i vol_i: the walker's share of the Einstein-Hilbert
    /// action. Maximizing it minimizes U = -r.
    EinsteinHilbertDensity { scale: f64 },
    /// r_i = scale R_i.
    CurvatureOnly { scale: f64 },
}
impl Default for RewardAllocationKind {
    fn default() -> Self {
        Self::EinsteinHilbertDensity { scale: 1. }
    }
}
impl<T: Real> RewardAllocation<T> for RewardAllocationKind {
    fn allocate(&self, curvature: &[T], volume: &[T], eligible: &[bool]) -> Result<Vec<T>> {
        require(
            curvature.len() == volume.len() && curvature.len() == eligible.len(),
            "reward allocation shapes",
        )?;
        let (scale, weighted) = match self {
            Self::EinsteinHilbertDensity { scale } => (*scale, true),
            Self::CurvatureOnly { scale } => (*scale, false),
        };
        require(scale.is_finite(), "reward scale must be finite")?;
        let scale = T::from_f64(scale);
        Ok((0..curvature.len())
            .map(|i| {
                if !eligible[i] {
                    T::ZERO
                } else if weighted {
                    scale * curvature[i] * volume[i]
                } else {
                    scale * curvature[i]
                }
            })
            .collect())
    }
}

/// Reward read from the geometry stage's observation fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryReward {
    /// Name of the curvature estimator in the geometry pipeline.
    pub curvature: String,
    #[serde(default)]
    pub allocation: RewardAllocationKind,
}
impl Default for GeometryReward {
    fn default() -> Self {
        Self {
            curvature: super::presets::RICCI_SCALAR.into(),
            allocation: RewardAllocationKind::default(),
        }
    }
}
impl<T: Real> RewardSource<T> for GeometryReward {
    fn id(&self) -> String {
        format!(
            "geometry-reward/v1:{}",
            serde_json::to_string(self).unwrap_or_default()
        )
    }
    fn evaluate<'a>(
        &'a self,
        population: &'a Population<T>,
        _input: Option<&'a InputBatch<T>>,
        stage: &'a str,
        _cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<T>> {
        Box::pin(async move {
            let curvature = population
                .observations
                .field(&curvature_field(&self.curvature))?;
            let volume = population.observations.field(VOLUME_FIELD)?;
            // Walkers that are already invalid keep a finite reward: their
            // validity flags, not this source, decide their fate.
            let raw = self.allocation.allocate(
                curvature.values(),
                volume.values(),
                &vec![true; population.len()],
            )?;
            Ok(RewardBatch::new(
                raw,
                Provenance {
                    population_version: population.version,
                    stage: stage.into(),
                    ..Default::default()
                },
            ))
        })
    }
}

/// The free gas: no potential force. BAOAB needs a gradient provider; this one
/// returns zero so that only graph forces, friction and noise act.
#[derive(Clone, Debug, Default)]
pub struct ZeroPotential {
    pub field: String,
}
impl ZeroPotential {
    pub fn new(velocity_field: impl Into<String>) -> Self {
        Self {
            field: velocity_field.into(),
        }
    }
}
impl<T: Real> GradientProvider<T> for ZeroPotential {
    fn id(&self) -> String {
        format!("zero-potential/v1:{}", self.field)
    }
    fn gradient<'a>(
        &'a self,
        population: &'a Population<T>,
        _cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>> {
        Box::pin(async move {
            let width = population.observations.field(&self.field)?.width();
            TensorBatch::vectors(
                population.len(),
                width,
                vec![T::ZERO; population.len() * width],
            )
        })
    }
}
