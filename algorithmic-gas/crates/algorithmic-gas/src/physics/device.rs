//! Private Burn batch contractions. This is a child of compute, not a public
//! tensor API. Small symmetric eigensolves run on the host; whitening, Gram,
//! Ricci, scalar, and sectional contractions run on the selected backend.
use super::{BackendKind, Device, ExecutionContext, guarded};
use crate::{
    GasError, Precision, Real, Result,
    error::require,
    partv_geometry::MetricPolicy,
    physics::geometry::{CurvatureBatch, FitnessJet, expand, finish, spectrum, symmetric_eigen},
};
use burn::tensor::{DType, Tensor, TensorData, backend::Backend};

impl ExecutionContext {
    /// Batched smooth Hessian curvature with O(N d^3) storage, O(N d^4) work.
    /// Mixed-sign clipping requires the general metric path and is explicitly
    /// rejected here. No CPU fallback is performed for device contractions.
    pub async fn smooth_curvature_batch<T: Real>(
        &mut self,
        jets: &[FitnessJet<T>],
        epsilon: T,
        policy: MetricPolicy,
        threshold: T,
    ) -> Result<Vec<CurvatureBatch<T>>> {
        require(
            !jets.is_empty() && jets.len() <= 16384,
            "curvature batch count",
        )?;
        require(
            T::PRECISION == self.precision,
            "curvature dtype differs from execution policy",
        )?;
        require(
            epsilon.is_finite()
                && epsilon > T::ZERO
                && threshold.is_finite()
                && threshold >= T::ZERO,
            "metric regularization/threshold",
        )?;
        let d = jets[0].dimension;
        require((1..=16).contains(&d), "curvature batch dimension")?;
        let n = jets.len();
        let peak = crate::memory::checked_mul(n, d.pow(3))?;
        require(
            peak <= self.max_batch_elements,
            "curvature batch exceeds allocation policy; tile the queries",
        )?;
        crate::memory::enforce(
            crate::memory::checked_mul(peak, std::mem::size_of::<T>() * 24)?,
            self.max_memory_bytes,
        )?;
        let mut specs = Vec::with_capacity(n);
        let mut branches = Vec::with_capacity(n);
        let mut thirds = Vec::with_capacity(peak);
        let mut frames = Vec::with_capacity(n * d * d);
        let mut duals = Vec::with_capacity(n * d * d);
        for jet in jets {
            jet.validate()?;
            require(jet.dimension == d, "curvature batch dimensions differ")?;
            let (lambda, q) = symmetric_eigen(&jet.dense_hessian()?, d)?;
            let constant = if policy == MetricPolicy::Clipped {
                if lambda.iter().any(|x| x.abs() <= threshold) {
                    return Err(GasError::Capability(
                        "batch contains a clipping threshold; classical curvature unavailable"
                            .into(),
                    ));
                }
                let negative = lambda.iter().all(|&x| x < T::ZERO);
                if !negative && !lambda.iter().all(|&x| x > T::ZERO) {
                    return Err(GasError::Capability("mixed-sign clipping requires metric_curvature with fourth derivatives; not a smooth batch".into()));
                }
                negative
            } else {
                false
            };
            let spec = spectrum(
                lambda
                    .iter()
                    .map(|&v| {
                        if policy == MetricPolicy::Strict {
                            v + epsilon
                        } else {
                            v.max(T::ZERO) + epsilon
                        }
                    })
                    .collect(),
                q,
                d,
            )?;
            for i in 0..d {
                for j in 0..d {
                    frames.push(spec.eigenvectors[i * d + j] / spec.eigenvalues[j].sqrt());
                    duals.push(spec.eigenvectors[i * d + j] * spec.eigenvalues[j].sqrt());
                }
            }
            thirds.extend(if constant {
                vec![T::ZERO; d.pow(3)]
            } else {
                expand(d, 3, &jet.third)?
            });
            specs.push(spec);
            branches.push(if constant {
                "constant_clipped"
            } else {
                "smooth_hessian"
            });
        }
        let data = guarded(async {
            match &self.device {
                #[cfg(feature = "cpu")]
                Device::Cpu(device) => {
                    contract::<burn::backend::Flex, T>(device, n, d, &thirds, &frames, &duals).await
                }
                #[cfg(feature = "wgpu")]
                Device::Wgpu(device) => {
                    contract::<burn::backend::Wgpu<f32, i32>, T>(
                        device, n, d, &thirds, &frames, &duals,
                    )
                    .await
                }
                #[cfg(feature = "cuda")]
                Device::Cuda(device) => {
                    contract::<burn::backend::Cuda<f32, i32>, T>(
                        device, n, d, &thirds, &frames, &duals,
                    )
                    .await
                }
            }
        })
        .await?;
        self.stats.evaluations += 1;
        self.stats.synchronizations += 1;
        self.stats.peak_batch_elements = self.stats.peak_batch_elements.max(peak);
        if self.kind != BackendKind::Cpu {
            self.stats.uploaded_bytes +=
                ((thirds.len() + frames.len() + duals.len()) * std::mem::size_of::<T>()) as u64;
            self.stats.downloaded_bytes += (data.len() * std::mem::size_of::<T>()) as u64;
        }
        let width = d * d + 1 + d * (d - 1) / 2;
        specs
            .into_iter()
            .enumerate()
            .map(|(row, spec)| {
                let values = &data[row * width..(row + 1) * width];
                finish(
                    spec,
                    values[..d * d].to_vec(),
                    values[d * d],
                    values[d * d + 1..].to_vec(),
                    None,
                    branches[row],
                )
            })
            .collect()
    }
}
fn upload<B: Backend, T: Real, const D: usize>(
    values: &[T],
    shape: [usize; D],
    device: &B::Device,
    dtype: DType,
) -> Tensor<B, D> {
    let data = match T::PRECISION {
        Precision::F32 => TensorData::new(
            values.iter().map(|x| x.to_f64() as f32).collect::<Vec<_>>(),
            shape,
        ),
        Precision::F64 => {
            TensorData::new(values.iter().map(|x| x.to_f64()).collect::<Vec<_>>(), shape)
        }
    };
    Tensor::from_data(data, (device, dtype))
}
async fn contract<B: Backend, T: Real>(
    device: &B::Device,
    n: usize,
    d: usize,
    thirds: &[T],
    frames: &[T],
    duals: &[T],
) -> Result<Vec<T>> {
    let dtype = if T::PRECISION == Precision::F64 {
        DType::F64
    } else {
        DType::F32
    };
    let e = upload::<B, T, 3>(frames, [n, d, d], device, dtype);
    let f = upload::<B, T, 3>(duals, [n, d, d], device, dtype);
    let mut c = upload::<B, T, 4>(thirds, [n, d, d, d], device, dtype);
    for axis in 1..=3 {
        c = e
            .clone()
            .transpose()
            .matmul(c.swap_dims(1, axis).reshape([n, d, d * d]))
            .reshape([n, d, d, d])
            .swap_dims(1, axis);
    }
    let flat = c.clone().reshape([n, d, d * d]);
    let gram = flat.clone().matmul(flat.transpose());
    let mut trace = c.clone().slice([0..n, 0..1, 0..1, 0..d]).reshape([n, 1, d]);
    for a in 1..d {
        trace = trace
            + c.clone()
                .slice([0..n, a..a + 1, a..a + 1, 0..d])
                .reshape([n, 1, d]);
    }
    let contracted = c
        .clone()
        .reshape([n, d * d, d])
        .matmul(trace.transpose())
        .reshape([n, d, d]);
    let ric_hat = (gram - contracted) * 0.25;
    let mut scalar = ric_hat.clone().slice([0..n, 0..1, 0..1]).reshape([n, 1]);
    for i in 1..d {
        scalar = scalar
            + ric_hat
                .clone()
                .slice([0..n, i..i + 1, i..i + 1])
                .reshape([n, 1]);
    }
    let ricci = f
        .clone()
        .matmul(ric_hat)
        .matmul(f.transpose())
        .reshape([n, d * d]);
    let mut parts = vec![ricci, scalar];
    for i in 0..d {
        for j in i + 1..d {
            let ij = c
                .clone()
                .slice([0..n, i..i + 1, j..j + 1, 0..d])
                .reshape([n, d]);
            let ii = c
                .clone()
                .slice([0..n, i..i + 1, i..i + 1, 0..d])
                .reshape([n, d]);
            let jj = c
                .clone()
                .slice([0..n, j..j + 1, j..j + 1, 0..d])
                .reshape([n, d]);
            parts.push((ij.clone() * ij - ii * jj).sum_dim(1) * 0.25);
        }
    }
    let data = Tensor::cat(parts, 1)
        .into_data_async()
        .await
        .map_err(|e| GasError::Execution(format!("{e:?}")))?;
    match T::PRECISION {
        Precision::F32 => Ok(data
            .as_slice::<f32>()
            .map_err(|e| GasError::Execution(format!("{e:?}")))?
            .iter()
            .map(|&v| T::from_f64(v as f64))
            .collect()),
        Precision::F64 => Ok(data
            .as_slice::<f64>()
            .map_err(|e| GasError::Execution(format!("{e:?}")))?
            .iter()
            .map(|&v| T::from_f64(v))
            .collect()),
    }
}
