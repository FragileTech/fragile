use crate::{
    ComputeBackend, ExecutionContext, GasError, ObservationBatch, Real, Result, TensorBatch,
    compute::{Binary, Expression, Unary},
    error::require,
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InnovationLaw {
    Gaussian,
    StandardizedUniform,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FactorValues {
    Constant { values: Vec<f64> },
    PerWalker { values: Vec<f64> },
    ObservationField { field: String },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NoiseGeometry {
    Isotropic { scale: FactorValues },
    Diagonal { factor: FactorValues },
    Full { factor: FactorValues },
    LowRank { rank: usize, factor: FactorValues },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Noise {
    pub innovation: InnovationLaw,
    pub geometry: NoiseGeometry,
}
impl Default for Noise {
    fn default() -> Self {
        Self {
            innovation: InnovationLaw::Gaussian,
            geometry: NoiseGeometry::Isotropic {
                scale: FactorValues::Constant { values: vec![1.] },
            },
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct NoiseRequest {
    pub rows: usize,
    pub dimension: usize,
    pub seed: u64,
    pub step: u64,
    pub stream: Stream,
    pub substep: u64,
}
#[allow(async_fn_in_trait)]
pub trait NoiseSource<T: Real> {
    async fn sample(
        &self,
        observations: &ObservationBatch<T>,
        request: NoiseRequest,
        cx: &mut ExecutionContext,
    ) -> Result<TensorBatch<T>>;
}
impl FactorValues {
    fn validate<T: Real>(&self, obs: &ObservationBatch<T>, n: usize, width: usize) -> Result<()> {
        let expected = n
            .checked_mul(width)
            .ok_or_else(|| GasError::Shape("factor shape overflow".into()))?;
        match self {
            Self::Constant { values } => require(
                values.len() == width && values.iter().all(|&x| T::from_f64(x).is_finite()),
                "constant factor shape or nonfinite factor",
            ),
            Self::PerWalker { values } => require(
                values.len() == expected && values.iter().all(|&x| T::from_f64(x).is_finite()),
                "per-walker factor shape or nonfinite factor",
            ),
            Self::ObservationField { field } => {
                let t = obs.field(field)?;
                require(
                    t.rows() == n && t.width() == width && t.values().iter().all(|x| x.is_finite()),
                    "observation-dependent factor shape or nonfinite factor",
                )
            }
        }
    }
    fn resolve<T: Real>(
        &self,
        obs: &ObservationBatch<T>,
        n: usize,
        width: usize,
    ) -> Result<Vec<T>> {
        let values = match self {
            Self::Constant { values } => {
                require(values.len() == width, "constant noise factor shape")?;
                (0..n)
                    .flat_map(|_| values.iter().map(|&x| T::from_f64(x)))
                    .collect()
            }
            Self::PerWalker { values } => {
                require(values.len() == n * width, "per-walker noise factor shape")?;
                values.iter().map(|&x| T::from_f64(x)).collect()
            }
            Self::ObservationField { field } => {
                let t = obs.field(field)?;
                require(
                    t.rows() == n && t.width() == width,
                    "observation-dependent factor shape",
                )?;
                t.values().to_vec()
            }
        };
        if values.iter().any(|v| !v.is_finite()) {
            return Err(GasError::Numerical("nonfinite noise factor".into()));
        }
        Ok(values)
    }
}
impl Noise {
    pub fn validate<T: Real>(&self, obs: &ObservationBatch<T>, n: usize, d: usize) -> Result<()> {
        require(n > 0 && d > 0, "empty noise batch")?;
        let (values, width) = match &self.geometry {
            NoiseGeometry::Isotropic { scale } => (scale, 1),
            NoiseGeometry::Diagonal { factor } => (factor, d),
            NoiseGeometry::Full { factor } => (
                factor,
                d.checked_mul(d)
                    .ok_or_else(|| GasError::Shape("noise matrix overflow".into()))?,
            ),
            NoiseGeometry::LowRank { rank, factor } => {
                require(*rank > 0 && *rank <= d, "low rank must be 1..=dimension")?;
                (factor, d * rank)
            }
        };
        values.validate(obs, n, width)?;
        Ok(())
    }
    pub fn required_elements(&self, n: usize, d: usize) -> Result<usize> {
        let rank = match self.geometry {
            NoiseGeometry::Full { .. } => d,
            NoiseGeometry::LowRank { rank, .. } => rank,
            _ => 1,
        };
        n.checked_mul(d)
            .and_then(|x| x.checked_mul(rank))
            .ok_or_else(|| GasError::Shape("noise batch size overflow".into()))
    }
}
impl<T: Real> NoiseSource<T> for Noise {
    async fn sample(
        &self,
        obs: &ObservationBatch<T>,
        r: NoiseRequest,
        cx: &mut ExecutionContext,
    ) -> Result<TensorBatch<T>> {
        crate::memory::enforce(
            crate::memory::checked_mul(
                self.required_elements(r.rows, r.dimension)?,
                std::mem::size_of::<T>() * 6,
            )?,
            cx.max_memory_bytes,
        )?;
        if self.required_elements(r.rows, r.dimension)? > cx.max_batch_elements {
            return Err(GasError::Capability(
                "noise factor batch exceeds configured memory budget".into(),
            ));
        }
        self.validate(obs, r.rows, r.dimension)?;
        let n = r.rows;
        let d = r.dimension;
        let rank = match &self.geometry {
            NoiseGeometry::LowRank { rank, .. } => *rank,
            _ => d,
        };
        let mut xi = vec![T::ZERO; n * rank];
        for i in 0..n {
            let mut rng = RandomStream::new(r.seed, r.step, r.stream, i as u64, r.substep);
            for a in 0..rank {
                xi[i * rank + a] = match self.innovation {
                    InnovationLaw::Gaussian => rng.gaussian(),
                    InnovationLaw::StandardizedUniform => {
                        (T::from_f64(12.).sqrt()) * (rng.uniform::<T>() - T::from_f64(0.5))
                    }
                };
            }
        }
        // Temporal scaling is deliberately absent. The integrator owns it.
        match &self.geometry {
            NoiseGeometry::Isotropic { scale } | NoiseGeometry::Diagonal { factor: scale } => {
                let width = if matches!(self.geometry, NoiseGeometry::Isotropic { .. }) {
                    1
                } else {
                    d
                };
                let factors = scale.resolve(obs, n, width)?;
                let mut e = Expression::default();
                let x = e.input(0);
                let l = e.input(1);
                e.binary(Binary::Multiply, x, l);
                cx.evaluate(
                    &e,
                    &[
                        TensorBatch::vectors(n, d, xi)?,
                        TensorBatch::vectors(n, width, factors)?,
                    ],
                )
                .await
            }
            NoiseGeometry::Full { factor } | NoiseGeometry::LowRank { factor, .. } => {
                let factors = factor.resolve(obs, n, d * rank)?;
                let expanded = (0..n)
                    .flat_map(|i| (0..d).flat_map(move |_| i * rank..(i + 1) * rank))
                    .map(|a| xi[a])
                    .collect();
                let mut e = Expression::default();
                let x = e.input(0);
                let l = e.input(1);
                let product = e.binary(Binary::Multiply, x, l);
                e.unary(Unary::SumRows, product);
                let output = cx
                    .evaluate(
                        &e,
                        &[
                            TensorBatch::vectors(n * d, rank, expanded)?,
                            TensorBatch::vectors(n * d, rank, factors)?,
                        ],
                    )
                    .await?;
                TensorBatch::vectors(n, d, output.values().to_vec())
            }
        }
    }
}
