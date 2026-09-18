//! Per-walker metric tensors on the tessellated coordinates.
//!
//! The emergent metric is the inverse covariance of the displacements to the
//! tessellation neighbors: walkers crowd where the metric is large. Other
//! estimators plug into the same role.
use super::{
    frame::GeometryFrame,
    linalg::{SpdSpectrum, clamp_spectrum, pinv_clamped},
    par::{Parallelism, try_map_indexed},
    voronoi::VoronoiCells,
};
use crate::{GasError, Real, Result, error::require};
use serde::{Deserialize, Serialize};

/// Metric, determinant and diffusion factor g^{-1/2} of every walker.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricField<T: Real> {
    pub dimension: usize,
    /// `[walkers, dimension, dimension]`, exactly symmetric.
    pub metric: Vec<T>,
    pub determinant: Vec<T>,
    /// `[walkers, dimension, dimension]`.
    pub diffusion: Vec<T>,
}
impl<T: Real> MetricField<T> {
    pub fn walkers(&self) -> usize {
        self.determinant.len()
    }
    pub fn tensor(&self, i: usize) -> &[T] {
        let w = self.dimension * self.dimension;
        &self.metric[i * w..(i + 1) * w]
    }
    fn assemble(dimension: usize, spectra: Vec<SpdSpectrum<T>>, diffusion_floor: T) -> Self {
        let mut out = Self {
            dimension,
            metric: Vec::with_capacity(spectra.len() * dimension * dimension),
            determinant: Vec::with_capacity(spectra.len()),
            diffusion: Vec::with_capacity(spectra.len() * dimension * dimension),
        };
        for s in spectra {
            out.metric.extend(s.matrix());
            out.determinant.push(s.determinant());
            out.diffusion.extend(s.power(-0.5, diffusion_floor));
        }
        out
    }
}

pub trait MetricEstimator<T: Real> {
    /// True when the estimator reads Voronoi cells.
    fn needs_cells(&self) -> bool {
        false
    }
    fn estimate(
        &self,
        frame: &GeometryFrame<'_, T>,
        cells: Option<&VoronoiCells<T>>,
        par: Parallelism,
    ) -> Result<MetricField<T>>;
}

fn default_ridge() -> f64 {
    1e-5
}
fn default_min_eig() -> Option<f64> {
    Some(1e-6)
}
fn default_epsilon_sigma() -> f64 {
    1e-3
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MetricKind {
    /// g_i = pinv(mean_j dx dx^T + ridge I), eigenvalues clamped.
    NeighborCovariance {
        #[serde(default = "default_ridge")]
        ridge: f64,
        #[serde(default = "default_min_eig")]
        min_eig: Option<f64>,
        #[serde(default)]
        max_eig: Option<f64>,
    },
    /// Flat space: curvature estimators must return zero.
    Identity,
    /// A symmetric `[dimension, dimension]` observation field, eigenvalues clamped.
    ObservationField {
        field: String,
        #[serde(default = "default_min_eig")]
        min_eig: Option<f64>,
        #[serde(default)]
        max_eig: Option<f64>,
    },
    /// g = H + epsilon_sigma I from a neighbor finite-difference Hessian of a
    /// scalar observation field.
    HessianFd {
        scalar_field: String,
        #[serde(default)]
        full: bool,
        #[serde(default = "default_epsilon_sigma")]
        epsilon_sigma: f64,
        #[serde(default = "default_min_eig")]
        min_eig: Option<f64>,
        #[serde(default)]
        max_eig: Option<f64>,
    },
    /// Inverse covariance of the Voronoi cell vertices about the walker.
    VoronoiCovariance {
        #[serde(default = "default_ridge")]
        ridge: f64,
        #[serde(default = "default_min_eig")]
        min_eig: Option<f64>,
        #[serde(default)]
        max_eig: Option<f64>,
    },
}
impl Default for MetricKind {
    fn default() -> Self {
        Self::NeighborCovariance {
            ridge: default_ridge(),
            min_eig: default_min_eig(),
            max_eig: None,
        }
    }
}
fn clamp_bounds(min_eig: Option<f64>, max_eig: Option<f64>) -> Result<()> {
    require(
        min_eig.is_none_or(|v| v.is_finite() && v > 0.)
            && max_eig.is_none_or(|v| v.is_finite() && v > 0.)
            && min_eig.zip(max_eig).is_none_or(|(a, b)| a <= b),
        "metric eigenvalue clamp must be positive and ordered",
    )
}
impl MetricKind {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::NeighborCovariance {
                ridge,
                min_eig,
                max_eig,
            }
            | Self::VoronoiCovariance {
                ridge,
                min_eig,
                max_eig,
            } => {
                require(
                    ridge.is_finite() && *ridge > 0.,
                    "covariance ridge must be positive",
                )?;
                clamp_bounds(*min_eig, *max_eig)
            }
            Self::Identity => Ok(()),
            Self::ObservationField {
                field,
                min_eig,
                max_eig,
            } => {
                require(!field.is_empty(), "metric observation field name")?;
                clamp_bounds(*min_eig, *max_eig)
            }
            Self::HessianFd {
                scalar_field,
                epsilon_sigma,
                min_eig,
                max_eig,
                ..
            } => {
                require(!scalar_field.is_empty(), "Hessian scalar field name")?;
                require(
                    epsilon_sigma.is_finite() && *epsilon_sigma >= 0.,
                    "Hessian metric shift must be nonnegative",
                )?;
                clamp_bounds(*min_eig, *max_eig)
            }
        }
    }
}
fn identity_spectrum<T: Real>(d: usize) -> SpdSpectrum<T> {
    let mut q = vec![T::ZERO; d * d];
    for k in 0..d {
        q[k * d + k] = T::ONE;
    }
    SpdSpectrum {
        eigenvalues: vec![T::ONE; d],
        eigenvectors: q,
    }
}
/// Inverse of the ridge-regularized mean outer product of `deltas`.
fn inverse_covariance<T: Real>(
    d: usize,
    deltas: impl Iterator<Item = Vec<T>>,
    ridge: T,
    min_eig: Option<T>,
    max_eig: Option<T>,
) -> Result<SpdSpectrum<T>> {
    let mut c = vec![T::ZERO; d * d];
    let mut count = 0usize;
    for dx in deltas {
        for a in 0..d {
            for b in a..d {
                c[a * d + b] = c[a * d + b] + dx[a] * dx[b];
            }
        }
        count += 1;
    }
    let scale = T::from_f64(count.max(1) as f64);
    for a in 0..d {
        for b in a..d {
            let v = c[a * d + b] / scale + if a == b { ridge } else { T::ZERO };
            c[a * d + b] = v;
            c[b * d + a] = v;
        }
    }
    pinv_clamped(&c, d, T::EPSILON * T::from_f64(d as f64), min_eig, max_eig)
}
impl<T: Real> MetricEstimator<T> for MetricKind {
    fn needs_cells(&self) -> bool {
        matches!(self, Self::VoronoiCovariance { .. })
    }
    fn estimate(
        &self,
        frame: &GeometryFrame<'_, T>,
        cells: Option<&VoronoiCells<T>>,
        par: Parallelism,
    ) -> Result<MetricField<T>> {
        let d = frame.dimension;
        let n = frame.walkers();
        let opt = |v: &Option<f64>| v.map(T::from_f64);
        let floor = |min_eig: &Option<f64>| T::from_f64(min_eig.unwrap_or(1e-6).max(1e-6));
        let (spectra, diffusion_floor) = match self {
            Self::NeighborCovariance {
                ridge,
                min_eig,
                max_eig,
            } => (
                try_map_indexed(n, par, |i| {
                    if !frame.eligible[i] {
                        return Ok(identity_spectrum(d));
                    }
                    inverse_covariance(
                        d,
                        frame
                            .graph
                            .row(i)
                            .iter()
                            .map(|&j| frame.delta(i, j as usize)),
                        T::from_f64(*ridge),
                        opt(min_eig),
                        opt(max_eig),
                    )
                })?,
                floor(min_eig),
            ),
            Self::Identity => (
                (0..n).map(|_| identity_spectrum(d)).collect(),
                T::from_f64(1e-6),
            ),
            Self::ObservationField {
                field,
                min_eig,
                max_eig,
            } => {
                let g = frame.observations.field(field)?;
                require(
                    g.rows() == n && g.width() == d * d,
                    "metric observation field must hold one dimension x dimension matrix per walker",
                )?;
                (
                    try_map_indexed(n, par, |i| {
                        if !frame.eligible[i] {
                            return Ok(identity_spectrum(d));
                        }
                        let row = g.row(i)?;
                        let mut sym = vec![T::ZERO; d * d];
                        for a in 0..d {
                            for b in 0..d {
                                sym[a * d + b] =
                                    (row[a * d + b] + row[b * d + a]) / T::from_f64(2.);
                            }
                        }
                        clamp_spectrum(&sym, d, opt(min_eig), opt(max_eig))
                    })?,
                    floor(min_eig),
                )
            }
            Self::HessianFd {
                scalar_field,
                full,
                epsilon_sigma,
                min_eig,
                max_eig,
            } => {
                let values = frame.observations.field(scalar_field)?;
                require(
                    values.rows() == n && values.width() == 1,
                    "Hessian scalar field must hold one value per walker",
                )?;
                let hessians = super::hessian::estimate(frame, values.values(), *full, par)?;
                (
                    try_map_indexed(n, par, |i| {
                        if !frame.eligible[i] {
                            return Ok(identity_spectrum(d));
                        }
                        let mut g = hessians[i * d * d..(i + 1) * d * d].to_vec();
                        for a in 0..d {
                            g[a * d + a] = g[a * d + a] + T::from_f64(*epsilon_sigma);
                        }
                        clamp_spectrum(&g, d, opt(min_eig), opt(max_eig))
                    })?,
                    floor(min_eig),
                )
            }
            Self::VoronoiCovariance {
                ridge,
                min_eig,
                max_eig,
            } => {
                let cells = cells.ok_or_else(|| {
                    GasError::Configuration(
                        "the Voronoi covariance metric requires Voronoi cells".into(),
                    )
                })?;
                (
                    try_map_indexed(n, par, |i| {
                        let vertices = cells.cell_vertices(i);
                        if !frame.eligible[i] || !cells.bounded[i] || vertices.is_empty() {
                            return Ok(identity_spectrum(d));
                        }
                        let x = frame.position(i);
                        inverse_covariance(
                            d,
                            vertices
                                .chunks_exact(d)
                                .map(|v| v.iter().zip(x).map(|(&a, &b)| a - b).collect()),
                            T::from_f64(*ridge),
                            opt(min_eig),
                            opt(max_eig),
                        )
                    })?,
                    floor(min_eig),
                )
            }
        };
        Ok(MetricField::assemble(d, spectra, diffusion_floor))
    }
}
