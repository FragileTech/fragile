//! The curvature estimator role: one scalar per walker, optionally a tensor
//! and an integrated action, from whatever earlier stages it declares it needs.
use super::{
    degenerate::Tessellation,
    frame::{EdgeLengths, GeometryFrame},
    metric::MetricField,
    par::Parallelism,
    proxies,
    regge::{self, ReggeLengths},
    ricci,
    voronoi::VoronoiCells,
};
use crate::{GasError, Real, Result, error::require};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub struct CurvatureContext<'a, T: Real> {
    pub frame: &'a GeometryFrame<'a, T>,
    pub tessellation: &'a Tessellation,
    pub metric: &'a MetricField<T>,
    pub lengths: &'a EdgeLengths<T>,
    pub volume: &'a [T],
    pub weights: &'a BTreeMap<String, Vec<T>>,
    pub cells: Option<&'a VoronoiCells<T>>,
    /// Cell volumes of the previous evaluation, for expansion rates.
    pub previous_cell_volume: Option<&'a [T]>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct CurvatureOutput<T: Real> {
    pub scalar: Vec<T>,
    /// `[walkers, d, d]` Ricci tensor when the estimator provides one.
    pub tensor: Option<Vec<T>>,
    /// False where the estimator has no value (the scalar is then zero).
    pub valid: Vec<bool>,
    /// Integrated curvature when the estimator defines it.
    pub total: Option<T>,
    /// Per-walker measure the scalar integrates against to give `total`
    /// (the barycentric dual volume of the Regge estimator).
    pub measure: Option<Vec<T>>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Requirements {
    pub cells: bool,
    pub previous_cell_volume: bool,
    pub weights: Option<String>,
}
pub trait CurvatureEstimator<T: Real> {
    fn requires(&self) -> Requirements;
    fn curvature(
        &self,
        cx: &CurvatureContext<'_, T>,
        par: Parallelism,
    ) -> Result<CurvatureOutput<T>>;
}

fn default_reg() -> f64 {
    1e-6
}
fn default_det_floor() -> f64 {
    1e-12
}
fn yes() -> bool {
    true
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CurvatureKind {
    /// R = -2 (d-1) Lap_w u with u = log det g / 2d and the named edge weights.
    ConformalLaplacian {
        weights: String,
        #[serde(default = "default_det_floor")]
        det_floor: f64,
    },
    /// Local quadratic fit of u: R = -2 (d-1) e^{-2u} (Lap u + (d-2)/2 |grad u|^2).
    ConformalQuadraticFit {
        #[serde(default)]
        weights: Option<String>,
        #[serde(default = "default_reg")]
        reg: f64,
        #[serde(default = "yes")]
        conformal_factor: bool,
        #[serde(default)]
        tensor: bool,
        #[serde(default = "default_det_floor")]
        det_floor: f64,
    },
    /// Deficit angles of the Delaunay complex.
    ReggeDeficit {
        #[serde(default)]
        lengths: ReggeLengths,
    },
    /// 1 - V / <V> over the Voronoi cells.
    VolumeDistortion,
    /// 1 - r_in / r_circ of the Voronoi cells.
    ShapeDistortion,
    /// -(V - V_prev) / (dt V) of the Voronoi cells.
    RaychaudhuriExpansion { dt: f64 },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurvatureSpec {
    pub name: String,
    pub estimator: CurvatureKind,
}
impl CurvatureKind {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::ConformalLaplacian { weights, det_floor } => require(
                !weights.is_empty() && det_floor.is_finite() && *det_floor > 0.,
                "conformal Laplacian needs named weights and a positive determinant floor",
            ),
            Self::ConformalQuadraticFit { reg, det_floor, .. } => require(
                reg.is_finite() && *reg > 0. && det_floor.is_finite() && *det_floor > 0.,
                "quadratic curvature fit needs positive regularization and determinant floor",
            ),
            Self::RaychaudhuriExpansion { dt } => require(
                dt.is_finite() && *dt > 0.,
                "expansion rate needs a positive time step",
            ),
            Self::ReggeDeficit { .. } | Self::VolumeDistortion | Self::ShapeDistortion => Ok(()),
        }
    }
}
impl<T: Real> CurvatureEstimator<T> for CurvatureKind {
    fn requires(&self) -> Requirements {
        match self {
            Self::ConformalLaplacian { weights, .. } => Requirements {
                weights: Some(weights.clone()),
                ..Requirements::default()
            },
            Self::ConformalQuadraticFit { weights, .. } => Requirements {
                weights: weights.clone(),
                ..Requirements::default()
            },
            Self::ReggeDeficit { .. } => Requirements::default(),
            Self::VolumeDistortion | Self::ShapeDistortion => Requirements {
                cells: true,
                ..Requirements::default()
            },
            Self::RaychaudhuriExpansion { .. } => Requirements {
                cells: true,
                previous_cell_volume: true,
                ..Requirements::default()
            },
        }
    }
    fn curvature(
        &self,
        cx: &CurvatureContext<'_, T>,
        par: Parallelism,
    ) -> Result<CurvatureOutput<T>> {
        let n = cx.frame.walkers();
        let named = |name: &String| {
            cx.weights.get(name).map(Vec::as_slice).ok_or_else(|| {
                GasError::Configuration(format!("curvature estimator needs edge weights {name:?}"))
            })
        };
        let cells = || {
            cx.cells.ok_or_else(|| {
                GasError::Configuration("curvature estimator needs Voronoi cells".into())
            })
        };
        let from_proxy = |p: proxies::ProxyField<T>| CurvatureOutput {
            scalar: p.scalar,
            tensor: None,
            valid: p.valid,
            total: Some(p.mean),
            measure: None,
        };
        Ok(match self {
            Self::ConformalLaplacian { weights, det_floor } => {
                let u = ricci::conformal_potential(
                    &cx.metric.determinant,
                    cx.frame.dimension,
                    *det_floor,
                );
                CurvatureOutput {
                    scalar: ricci::conformal_laplacian(cx.frame, &u, named(weights)?, par),
                    tensor: None,
                    valid: cx.frame.eligible.to_vec(),
                    total: None,
                    measure: None,
                }
            }
            Self::ConformalQuadraticFit {
                weights,
                reg,
                conformal_factor,
                tensor,
                det_floor,
            } => {
                let u = ricci::conformal_potential(
                    &cx.metric.determinant,
                    cx.frame.dimension,
                    *det_floor,
                );
                let w = weights.as_ref().map(named).transpose()?;
                let (scalar, tensor) = ricci::conformal_quadratic(
                    cx.frame,
                    &u,
                    w,
                    *reg,
                    *conformal_factor,
                    *tensor,
                    par,
                )?;
                CurvatureOutput {
                    scalar,
                    tensor,
                    valid: cx.frame.eligible.to_vec(),
                    total: None,
                    measure: None,
                }
            }
            Self::ReggeDeficit { lengths } => {
                let slot_length = match lengths {
                    ReggeLengths::Geodesic => cx.lengths.geodesic(),
                    ReggeLengths::Euclidean => cx.lengths.euclidean.clone(),
                };
                let r = regge::curvature(cx.tessellation, &slot_length, par);
                CurvatureOutput {
                    scalar: r.scalar,
                    tensor: None,
                    valid: r.valid,
                    total: Some(r.action),
                    measure: Some(r.dual_volume),
                }
            }
            Self::VolumeDistortion => from_proxy(proxies::volume_distortion(cells()?)),
            Self::ShapeDistortion => from_proxy(proxies::shape_distortion(cells()?, par)),
            Self::RaychaudhuriExpansion { dt } => from_proxy(proxies::raychaudhuri(
                cells()?,
                cx.previous_cell_volume,
                *dt,
            )),
        })
        .inspect(|out| debug_assert_eq!(out.scalar.len(), n))
    }
}
