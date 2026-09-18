//! Host-side objective execution: scalar f64 evaluation, central-difference gradients,
//! graph constants, and replay-exact reward noise.
use crate::{Benchmark, classics::ObjectiveGraph};
use algorithmic_gas::{
    Population, Real, Result, RewardBatch, TensorBatch,
    random::{RandomStream, Stream},
};

/// Graph inputs `1..`: the objective's constant matrices in the run precision.
pub fn constants<T: Real>(graph: &ObjectiveGraph) -> Result<Vec<TensorBatch<T>>> {
    graph
        .constants
        .iter()
        .map(|c| {
            TensorBatch::vectors(
                c.rows,
                c.columns,
                c.values.iter().map(|&v| T::from_f64(v)).collect(),
            )
        })
        .collect()
}

fn row_f64<T: Real>(row: &[T]) -> Vec<f64> {
    row.iter().map(|x| x.to_f64()).collect()
}

/// One objective evaluation per row.
pub fn values<T: Real>(benchmark: Benchmark, x: &TensorBatch<T>) -> Result<Vec<T>> {
    let width = x.width();
    let evaluator = benchmark.evaluator(width)?;
    Ok(x.values()
        .chunks(width)
        .map(|row| T::from_f64(evaluator.value(&row_f64(row))))
        .collect())
}

/// Central differences with `h = 1e-5·max(1, |x_k|)`; ineligible rows stay zero.
/// Returns the gradient and the number of objective evaluations spent.
pub fn central_gradient<T: Real>(
    benchmark: Benchmark,
    x: &TensorBatch<T>,
    eligible: impl Fn(usize) -> bool,
) -> Result<(TensorBatch<T>, u64)> {
    let width = x.width();
    let evaluator = benchmark.evaluator(width)?;
    let mut gradient = vec![T::ZERO; x.values().len()];
    let mut evaluations = 0;
    for (i, row) in x.values().chunks(width).enumerate() {
        if !eligible(i) {
            continue;
        }
        let mut point = row_f64(row);
        for k in 0..width {
            let value = point[k];
            let h = 1e-5 * value.abs().max(1.);
            point[k] = value + h;
            let upper = evaluator.value(&point);
            point[k] = value - h;
            let lower = evaluator.value(&point);
            point[k] = value;
            gradient[i * width + k] = T::from_f64((upper - lower) / (2. * h));
        }
        evaluations += 2 * width as u64;
    }
    Ok((
        TensorBatch::vectors(x.rows(), width, gradient)?,
        evaluations,
    ))
}

/// Additive Gaussian reward noise addressed by `(seed, population version, stage, walker)`,
/// so checkpoints and replays reproduce every draw.
#[derive(Clone, Debug)]
pub struct RewardNoise {
    pub std: f64,
    pub seed: u64,
}

impl RewardNoise {
    pub fn apply<T: Real>(&self, rewards: &mut RewardBatch<T>, p: &Population<T>, stage: &str) {
        let stage = match stage {
            "initial" => 0,
            "pre_clone" => 1,
            "post_clone" => 2,
            "post_kinetic" => 3,
            "external_replace" => 4,
            _ => 5,
        };
        let std = T::from_f64(self.std);
        for (i, reward) in rewards.raw.iter_mut().enumerate() {
            let mut rng =
                RandomStream::new(self.seed, p.version, Stream::RewardNoise, i as u64, stage);
            *reward = *reward + std * rng.gaussian::<T>();
        }
    }
}
