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
use crate::{GasError, Real, Result, error::require, partv_geometry::MetricPolicy};
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
    /// `[walkers, dimension]`: the eigenvalues a clamp or a sign repair moved,
    /// so a positive definite estimate is never confused with a repaired one.
    pub clipped: Vec<bool>,
}
impl<T: Real> MetricField<T> {
    pub fn walkers(&self) -> usize {
        self.determinant.len()
    }
    pub fn tensor(&self, i: usize) -> &[T] {
        let w = self.dimension * self.dimension;
        &self.metric[i * w..(i + 1) * w]
    }
    /// True when the metric of walker `i` had to be repaired to be positive.
    pub fn repaired(&self, i: usize) -> bool {
        self.clipped[i * self.dimension..(i + 1) * self.dimension]
            .iter()
            .any(|&moved| moved)
    }
    fn assemble(dimension: usize, spectra: Vec<SpdSpectrum<T>>, diffusion_floor: T) -> Self {
        let mut out = Self {
            dimension,
            metric: Vec::with_capacity(spectra.len() * dimension * dimension),
            determinant: Vec::with_capacity(spectra.len()),
            diffusion: Vec::with_capacity(spectra.len() * dimension * dimension),
            clipped: Vec::with_capacity(spectra.len() * dimension),
        };
        for s in spectra {
            out.metric.extend(s.matrix());
            out.determinant.push(s.determinant());
            out.diffusion.extend(s.power(-0.5, diffusion_floor));
            out.clipped.extend(s.clipped);
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
fn is_relative(scale: &RidgeScale) -> bool {
    *scale == RidgeScale::RelativeToTrace
}
fn is_clipped(policy: &MetricPolicy) -> bool {
    *policy == MetricPolicy::Clipped
}

/// Units of the ridge and of the eigenvalue bounds of a covariance metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RidgeScale {
    /// The numbers are coordinates: a ridge in length^2 and bounds in
    /// length^-2. The convention of the reference estimators, and the one a
    /// recorded archive was measured with; the metric it produces does not
    /// scale with the cloud.
    Absolute,
    /// The numbers multiply the scale `tau = tr(C)/d` of the displacement
    /// covariance: the ridge is `ridge * tau` and the bounds are `min_eig /
    /// tau` and `max_eig / tau`. Under `x -> lambda x` the covariance scales as
    /// `lambda^2` and the regularized metric as `lambda^-2`, so the scale
    /// ladder, the gate and the smearing width built on it are covariant.
    /// Displacements with no scale, a coincident group or a walker with no
    /// neighbor, keep `tau = 1` and therefore the absolute convention.
    #[default]
    RelativeToTrace,
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
        #[serde(default, skip_serializing_if = "is_relative")]
        scale: RidgeScale,
        #[serde(default, skip_serializing_if = "is_clipped")]
        policy: MetricPolicy,
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
        #[serde(default, skip_serializing_if = "is_clipped")]
        policy: MetricPolicy,
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
        #[serde(default, skip_serializing_if = "is_clipped")]
        policy: MetricPolicy,
    },
    /// Inverse covariance of the Voronoi cell vertices about the walker.
    VoronoiCovariance {
        #[serde(default = "default_ridge")]
        ridge: f64,
        #[serde(default = "default_min_eig")]
        min_eig: Option<f64>,
        #[serde(default)]
        max_eig: Option<f64>,
        #[serde(default, skip_serializing_if = "is_relative")]
        scale: RidgeScale,
        #[serde(default, skip_serializing_if = "is_clipped")]
        policy: MetricPolicy,
    },
}
impl Default for MetricKind {
    fn default() -> Self {
        Self::NeighborCovariance {
            ridge: default_ridge(),
            min_eig: default_min_eig(),
            max_eig: None,
            scale: RidgeScale::default(),
            policy: MetricPolicy::default(),
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
                ..
            }
            | Self::VoronoiCovariance {
                ridge,
                min_eig,
                max_eig,
                ..
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
                ..
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
        clipped: vec![false; d],
    }
}
/// Inverse of the ridge-regularized mean outer product of `deltas`. `scale`
/// fixes the units of `ridge` and of the eigenvalue bounds; see `RidgeScale`.
fn inverse_covariance<T: Real>(
    d: usize,
    deltas: impl Iterator<Item = Vec<T>>,
    ridge: T,
    min_eig: Option<T>,
    max_eig: Option<T>,
    scale: RidgeScale,
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
    let edges = T::from_f64(count.max(1) as f64);
    for a in 0..d {
        for b in a..d {
            let v = c[a * d + b] / edges;
            c[a * d + b] = v;
            c[b * d + a] = v;
        }
    }
    let trace = (0..d).fold(T::ZERO, |s, k| s + c[k * d + k]) / T::from_f64(d as f64);
    let tau = match scale {
        RidgeScale::RelativeToTrace if trace > T::ZERO => trace,
        _ => T::ONE,
    };
    for k in 0..d {
        c[k * d + k] = c[k * d + k] + ridge * tau;
    }
    let bound = |v: Option<T>| v.map(|b| b / tau);
    pinv_clamped(
        &c,
        d,
        T::EPSILON * T::from_f64(d as f64),
        bound(min_eig),
        bound(max_eig),
    )
}
/// `Strict` refuses a spectrum a clamp or a sign repair had to move; `Clipped`
/// keeps it and carries the flags, the policy of `metric_from_hessian`.
fn honour<T: Real>(policy: MetricPolicy, spectrum: SpdSpectrum<T>) -> Result<SpdSpectrum<T>> {
    if policy == MetricPolicy::Strict && spectrum.repaired() {
        return Err(GasError::Numerical(
            "strict metric policy: the estimated metric is not positive definite".into(),
        ));
    }
    Ok(spectrum)
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
                scale,
                policy,
            } => (
                try_map_indexed(n, par, |i| {
                    if !frame.eligible[i] {
                        return Ok(identity_spectrum(d));
                    }
                    honour(
                        *policy,
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
                            *scale,
                        )?,
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
                policy,
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
                        honour(
                            *policy,
                            clamp_spectrum(&sym, d, opt(min_eig), opt(max_eig))?,
                        )
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
                policy,
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
                        honour(*policy, clamp_spectrum(&g, d, opt(min_eig), opt(max_eig))?)
                    })?,
                    floor(min_eig),
                )
            }
            Self::VoronoiCovariance {
                ridge,
                min_eig,
                max_eig,
                scale,
                policy,
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
                        honour(
                            *policy,
                            inverse_covariance(
                                d,
                                vertices
                                    .chunks_exact(d)
                                    .map(|v| v.iter().zip(x).map(|(&a, &b)| a - b).collect()),
                                T::from_f64(*ridge),
                                opt(min_eig),
                                opt(max_eig),
                                *scale,
                            )?,
                        )
                    })?,
                    floor(min_eig),
                )
            }
        };
        Ok(MetricField::assemble(d, spectra, diffusion_floor))
    }
}
