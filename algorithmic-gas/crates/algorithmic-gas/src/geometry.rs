use crate::{
    ComputeBackend, ExecutionContext, GasError, ObservationBatch, Real, Result, TensorBatch,
    boundary::BoxDomain,
    compute::{Binary, Expression, Unary},
    error::require,
};
use serde::{Deserialize, Serialize};

// Scale before squaring: finite very small/large coordinates must not be
// mistaken for zero or overflow. Compare the norm with the model tolerance
// without forming scale * norm (which can itself overflow).
fn cosine_unit<T: Real>(row: &[T], tolerance: T) -> Result<Option<Vec<T>>> {
    if row.iter().any(|x| !x.is_finite()) {
        return Err(GasError::Numerical("nonfinite cosine input".into()));
    }
    let scale = row.iter().fold(T::ZERO, |m, x| m.max(x.abs()));
    if scale == T::ZERO {
        return Ok(None);
    }
    let mut unit: Vec<T> = row.iter().map(|&x| x / scale).collect();
    let norm = unit.iter().fold(T::ZERO, |s, &x| s + x * x).sqrt();
    if scale <= tolerance / norm {
        return Ok(None);
    }
    for x in &mut unit {
        *x = *x / norm;
    }
    Ok(Some(unit))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonKind {
    Distance,
    SquaredDistance,
    Dissimilarity,
}
pub trait AlgorithmicDistance<T: Real> {
    fn comparison_kind(&self) -> ComparisonKind;
    fn compare(
        &self,
        a: &ObservationBatch<T>,
        i: usize,
        b: &ObservationBatch<T>,
        j: usize,
    ) -> Result<T>;
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Distance {
    Euclidean {
        field: String,
        scales: Vec<f64>,
        squared: bool,
        periodic: Option<BoxDomain>,
    },
    PhaseSpace {
        positions: String,
        velocities: String,
        position_scale: f64,
        velocity_scale: f64,
        lambda: f64,
        periodic: Option<BoxDomain>,
    },
    Cosine {
        field: String,
        zero_tolerance: f64,
    },
}
impl Default for Distance {
    fn default() -> Self {
        Self::Euclidean {
            field: "positions".into(),
            scales: vec![],
            squared: false,
            periodic: None,
        }
    }
}
impl Distance {
    pub fn validate<T: Real>(&self, obs: &ObservationBatch<T>) -> Result<()> {
        match self {
            Self::Euclidean {
                field,
                scales,
                periodic,
                ..
            } => {
                let t = obs.field(field)?;
                require(
                    scales.is_empty() || scales.len() == t.width(),
                    "distance scales must be empty or one positive scale per coordinate",
                )?;
                require(
                    scales.iter().all(|s| s.is_finite() && *s > 0.),
                    "distance scales must be positive",
                )?;
                if let Some(p) = periodic {
                    require(
                        t.item_shape().len() == 1,
                        "periodic distance needs a coordinate vector",
                    )?;
                    p.validate_precision::<T>(t.width())?;
                }
            }
            Self::PhaseSpace {
                positions,
                velocities,
                position_scale,
                velocity_scale,
                lambda,
                periodic,
            } => {
                let x = obs.field(positions)?;
                let v = obs.field(velocities)?;
                require(
                    x.item_shape().len() == 1 && v.item_shape() == x.item_shape(),
                    "phase-space distance requires matching position and velocity vectors",
                )?;
                require(
                    position_scale.is_finite()
                        && *position_scale > 0.
                        && velocity_scale.is_finite()
                        && *velocity_scale > 0.
                        && lambda.is_finite()
                        && *lambda >= 0.,
                    "invalid phase-space scales",
                )?;
                if let Some(p) = periodic {
                    p.validate_precision::<T>(x.width())?;
                }
            }
            Self::Cosine {
                field,
                zero_tolerance,
            } => {
                obs.field(field)?;
                require(
                    zero_tolerance.is_finite() && *zero_tolerance >= 0.,
                    "invalid cosine numerical zero tolerance",
                )?;
            }
        }
        Ok(())
    }
    pub fn field(&self) -> &str {
        match self {
            Self::Euclidean { field, .. } | Self::Cosine { field, .. } => field,
            Self::PhaseSpace { positions, .. } => positions,
        }
    }
    pub fn periodic(&self) -> Option<&BoxDomain> {
        match self {
            Self::Euclidean { periodic, .. } | Self::PhaseSpace { periodic, .. } => {
                periodic.as_ref()
            }
            _ => None,
        }
    }
    /// Batched pair evaluation. No N×N allocation: the caller supplies a bounded
    /// edge tile. Periodic deltas and schema packing happen on the host; sums,
    /// products, and square roots use the selected Burn adapter.
    pub async fn pairs<T: Real>(
        &self,
        a: &ObservationBatch<T>,
        b: &ObservationBatch<T>,
        pairs: &[(u32, u32)],
        cx: &mut ExecutionContext,
    ) -> Result<Vec<T>> {
        self.validate(a)?;
        self.validate(b)?;
        if pairs.is_empty() {
            return Ok(vec![]);
        }
        if let Self::Cosine {
            field,
            zero_tolerance,
        } = self
        {
            let x = a.field(field)?;
            let y = b.field(field)?;
            require(x.item_shape() == y.item_shape(), "cosine schemas differ")?;
            let mut av = Vec::new();
            let mut bv = Vec::new();
            let mut zeros = Vec::with_capacity(pairs.len());
            for &(i, j) in pairs {
                let a = cosine_unit(x.row(i as usize)?, T::from_f64(*zero_tolerance))?;
                let b = cosine_unit(y.row(j as usize)?, T::from_f64(*zero_tolerance))?;
                zeros.push((a.is_none(), b.is_none()));
                av.extend(a.unwrap_or_else(|| vec![T::ZERO; x.width()]));
                bv.extend(b.unwrap_or_else(|| vec![T::ZERO; y.width()]));
            }
            let aa = TensorBatch::vectors(pairs.len(), x.width(), av)?;
            let bb = TensorBatch::vectors(pairs.len(), x.width(), bv)?;
            let mut e = Expression::default();
            let an = e.input(0);
            let bn = e.input(1);
            let ab = e.binary(Binary::Multiply, an, bn);
            e.unary(Unary::SumRows, ab);
            let result = cx.evaluate(&e, &[aa, bb]).await?;
            if result.values().iter().any(|v| !v.is_finite()) {
                return Err(GasError::Numerical("cosine intermediate overflow".into()));
            }
            return Ok(result
                .values()
                .iter()
                .zip(zeros)
                .map(|(&dot, (az, bz))| {
                    if az && bz {
                        T::ZERO
                    } else if az || bz {
                        T::ONE
                    } else {
                        T::ONE - dot.min(T::ONE).max(-T::ONE)
                    }
                })
                .collect());
        }
        let mut values = Vec::new();
        let mut width = 0;
        for &(i, j) in pairs {
            let delta = self.deltas(a, i as usize, b, j as usize)?;
            width = delta.len();
            values.extend(delta);
        }
        let input = TensorBatch::vectors(pairs.len(), width, values)?;
        let mut e = Expression::default();
        let x = e.input(0);
        let sq = e.unary(Unary::Square, x);
        let sum = e.unary(Unary::SumRows, sq);
        if self.comparison_kind_for() != ComparisonKind::SquaredDistance {
            e.unary(Unary::Sqrt, sum);
        }
        let result = cx.evaluate(&e, &[input]).await?;
        if result.values().iter().any(|x| !x.is_finite()) {
            return Err(GasError::Numerical("distance overflow".into()));
        }
        Ok(result.values().to_vec())
    }
    fn comparison_kind_for(&self) -> ComparisonKind {
        match self {
            Self::Euclidean { squared: true, .. } => ComparisonKind::SquaredDistance,
            Self::Cosine { .. } => ComparisonKind::Dissimilarity,
            _ => ComparisonKind::Distance,
        }
    }
    fn deltas<T: Real>(
        &self,
        a: &ObservationBatch<T>,
        i: usize,
        b: &ObservationBatch<T>,
        j: usize,
    ) -> Result<Vec<T>> {
        let mut delta = Vec::new();
        match self {
            Self::Euclidean {
                field,
                scales,
                periodic,
                ..
            } => {
                let av = a.field(field)?;
                let bv = b.field(field)?;
                require(
                    av.item_shape() == bv.item_shape(),
                    "distance schemas differ",
                )?;
                for (k, (&x, &y)) in av.row(i)?.iter().zip(bv.row(j)?).enumerate() {
                    let d = periodic
                        .as_ref()
                        .map_or(x - y, |p| p.minimum_image(x - y, k));
                    delta.push(d / T::from_f64(scales.get(k).copied().unwrap_or(1.)));
                }
            }
            Self::PhaseSpace {
                positions,
                velocities,
                position_scale,
                velocity_scale,
                lambda,
                periodic,
            } => {
                require(
                    a.field(positions)?.item_shape() == b.field(positions)?.item_shape()
                        && a.field(velocities)?.item_shape() == b.field(velocities)?.item_shape(),
                    "phase-space schemas differ",
                )?;
                for (k, (&x, &y)) in a
                    .field(positions)?
                    .row(i)?
                    .iter()
                    .zip(b.field(positions)?.row(j)?)
                    .enumerate()
                {
                    delta.push(
                        periodic
                            .as_ref()
                            .map_or(x - y, |p| p.minimum_image(x - y, k))
                            / T::from_f64(*position_scale),
                    );
                }
                for (&x, &y) in a
                    .field(velocities)?
                    .row(i)?
                    .iter()
                    .zip(b.field(velocities)?.row(j)?)
                {
                    delta
                        .push((x - y) / T::from_f64(*velocity_scale) * T::from_f64(*lambda).sqrt());
                }
            }
            Self::Cosine { .. } => unreachable!(),
        }
        Ok(delta)
    }
}
impl<T: Real> AlgorithmicDistance<T> for Distance {
    fn comparison_kind(&self) -> ComparisonKind {
        self.comparison_kind_for()
    }
    fn compare(
        &self,
        a: &ObservationBatch<T>,
        i: usize,
        b: &ObservationBatch<T>,
        j: usize,
    ) -> Result<T> {
        self.validate(a)?;
        self.validate(b)?;
        if let Self::Cosine {
            field,
            zero_tolerance,
        } = self
        {
            let av = a.field(field)?;
            let bv = b.field(field)?;
            require(av.item_shape() == bv.item_shape(), "cosine schemas differ")?;
            let z = T::from_f64(*zero_tolerance);
            return Ok(
                match (cosine_unit(av.row(i)?, z)?, cosine_unit(bv.row(j)?, z)?) {
                    (None, None) => T::ZERO,
                    (Some(a), Some(b)) => {
                        let dot = a.iter().zip(b).fold(T::ZERO, |s, (&x, y)| s + x * y);
                        T::ONE - dot.max(-T::ONE).min(T::ONE)
                    }
                    _ => T::ONE,
                },
            );
        }
        let sum = self
            .deltas(a, i, b, j)?
            .iter()
            .fold(T::ZERO, |s, &x| s + x * x);
        if !sum.is_finite() {
            return Err(GasError::Numerical("distance overflow".into()));
        }
        Ok(
            if self.comparison_kind_for() == ComparisonKind::SquaredDistance {
                sum
            } else {
                sum.sqrt()
            },
        )
    }
}
pub trait InteractionKernel<T: Real> {
    fn log_weight(&self, value: T, kind: ComparisonKind) -> Result<T>;
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Kernel {
    Uniform,
    Gaussian { width: f64 },
    Exponential { temperature: f64 },
}
impl Kernel {
    pub fn validate(&self, kind: ComparisonKind) -> Result<()> {
        match self {
            Self::Uniform => Ok(()),
            Self::Gaussian { width } => {
                require(
                    width.is_finite() && *width > 0.,
                    "Gaussian width must be positive",
                )?;
                require(
                    kind != ComparisonKind::Dissimilarity,
                    "use an explicit exponential kernel for cosine dissimilarity",
                )
            }
            Self::Exponential { temperature } => require(
                temperature.is_finite() && *temperature > 0.,
                "temperature must be positive",
            ),
        }
    }
}
impl<T: Real> InteractionKernel<T> for Kernel {
    fn log_weight(&self, value: T, kind: ComparisonKind) -> Result<T> {
        self.validate(kind)?;
        if !value.is_finite() || value < T::ZERO {
            return Err(GasError::Numerical("invalid comparison value".into()));
        }
        Ok(match self {
            Self::Uniform => T::ZERO,
            Self::Gaussian { width } => {
                let sq = if kind == ComparisonKind::SquaredDistance {
                    value
                } else {
                    value * value
                };
                -sq / (T::from_f64(2.) * T::from_f64(*width) * T::from_f64(*width))
            }
            Self::Exponential { temperature } => -value / T::from_f64(*temperature),
        })
    }
}
