//! Analytic objectives shared by the native runner and browser bindings.
pub mod bbob;
mod benchmark;
pub mod catalog;
pub mod classics;
mod host;
pub mod lecture;
pub mod lecture_early;
pub mod lecture_fractal;
pub mod lecture_meanfield;
pub mod lecture_qft_protocols;
mod lecture_taylor;
pub mod mixture;
mod mt64;
pub mod physics_metric;
use algorithmic_gas::{
    AlgorithmicGas, ComputeBackend, ExecutionContext, GasBuilder, GasConfig, GasError, InputBatch,
    ObservationBatch, Population, Provenance, Real, Result, RewardBatch, TensorBatch,
    boundary::{BoundaryPolicy, BoxDomain},
    compute::{Binary, Expression, Node, Unary},
    domain::{GradientProvider, OperatorFuture, RewardSource},
    fitness::ObjectiveDirection,
    kinetic::KineticKind,
    random::{RandomStream, Stream},
};
pub use benchmark::{Benchmark, Evaluator, GradientExecution, ObjectiveExecution};
use serde::{Deserialize, Serialize};

impl Benchmark {
    /// Constant-free tensor graph of the five lecture objectives; see [`Benchmark::graph`].
    pub fn expression(self, d: usize, gradient: bool) -> Result<Expression> {
        self.validate(d)?;
        let mut e = Expression::default();
        let x = e.input(0);
        let x2 = e.unary(Unary::Square, x);
        let c2 = e.scalar(2.);
        let c10 = e.scalar(10.);
        let c16 = e.scalar(16.);
        let c5 = e.scalar(5.);
        let half = e.scalar(0.5);
        let term = match self {
            Self::Sphere => {
                if gradient {
                    e.binary(Binary::Multiply, c2, x)
                } else {
                    x2
                }
            }
            Self::Quadratic => {
                if gradient {
                    x
                } else {
                    e.binary(Binary::Multiply, half, x2)
                }
            }
            Self::Rastrigin => {
                let tau = e.scalar(std::f64::consts::TAU);
                let angle = e.binary(Binary::Multiply, tau, x);
                if gradient {
                    let sin = e.unary(Unary::Sin, angle);
                    let coeff = e.binary(Binary::Multiply, c10, tau);
                    let periodic = e.binary(Binary::Multiply, coeff, sin);
                    let linear = e.binary(Binary::Multiply, c2, x);
                    e.binary(Binary::Add, linear, periodic)
                } else {
                    let cos = e.unary(Unary::Cos, angle);
                    let periodic = e.binary(Binary::Multiply, c10, cos);
                    let difference = e.binary(Binary::Subtract, x2, periodic);
                    e.binary(Binary::Add, difference, c10)
                }
            }
            Self::StyblinskiTang => {
                if gradient {
                    let cube = e.binary(Binary::Multiply, x2, x);
                    let twice = e.binary(Binary::Multiply, c2, cube);
                    let linear = e.binary(Binary::Multiply, c16, x);
                    let difference = e.binary(Binary::Subtract, twice, linear);
                    let bias = e.scalar(2.5);
                    e.binary(Binary::Add, difference, bias)
                } else {
                    let fourth = e.unary(Unary::Square, x2);
                    let quadratic = e.binary(Binary::Multiply, c16, x2);
                    let difference = e.binary(Binary::Subtract, fourth, quadratic);
                    let linear = e.binary(Binary::Multiply, c5, x);
                    let sum = e.binary(Binary::Add, difference, linear);
                    e.binary(Binary::Multiply, half, sum)
                }
            }
            Self::Rosenbrock => {
                let left = e.push(Node::Columns {
                    source: x,
                    start: 0,
                    end: d - 1,
                });
                let right = e.push(Node::Columns {
                    source: x,
                    start: 1,
                    end: d,
                });
                let left2 = e.unary(Unary::Square, left);
                let delta = e.binary(Binary::Subtract, left2, right);
                let one = e.scalar(1.);
                let bias = e.binary(Binary::Subtract, left, one);
                if gradient {
                    let c400 = e.scalar(400.);
                    let c200 = e.scalar(-200.);
                    let product = e.binary(Binary::Multiply, left, delta);
                    let term = e.binary(Binary::Multiply, c400, product);
                    let linear = e.binary(Binary::Multiply, c2, bias);
                    let first = e.binary(Binary::Add, term, linear);
                    let second = e.binary(Binary::Multiply, c200, delta);
                    let column = e.push(Node::Columns {
                        source: x,
                        start: 0,
                        end: 1,
                    });
                    let zero = e.scalar(0.);
                    let zeros = e.binary(Binary::Multiply, column, zero);
                    let first = e.push(Node::ConcatColumns(vec![first, zeros]));
                    let second = e.push(Node::ConcatColumns(vec![zeros, second]));
                    e.binary(Binary::Add, first, second)
                } else {
                    let sq = e.unary(Unary::Square, delta);
                    let c100 = e.scalar(100.);
                    let term = e.binary(Binary::Multiply, c100, sq);
                    let bias2 = e.unary(Unary::Square, bias);
                    e.binary(Binary::Add, term, bias2)
                }
            }
            _ => {
                return Err(GasError::Capability(
                    "this objective needs Benchmark::graph (constants) or the host evaluator"
                        .into(),
                ));
            }
        };
        if gradient {
            // Ensure a terminal identity even when the result is input x.
            let one = e.scalar(1.);
            e.binary(Binary::Multiply, term, one);
        } else {
            e.unary(Unary::SumRows, term);
        }
        Ok(e)
    }
}
#[derive(Clone, Debug)]
pub struct BenchmarkModel {
    pub benchmark: Benchmark,
    pub field: String,
    pub direction: ObjectiveDirection,
}
impl<T: Real> RewardSource<T> for BenchmarkModel {
    fn id(&self) -> String {
        match self.benchmark.execution() {
            ObjectiveExecution::Graph => {
                format!("benchmark/{:?}/{}/v1", self.benchmark, self.field)
            }
            ObjectiveExecution::Host => {
                format!("benchmark-host/{:?}/{}/v1", self.benchmark, self.field)
            }
        }
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<T>,
        input: Option<&'a InputBatch<T>>,
        stage: &'a str,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<T>> {
        Box::pin(async move {
            let mut x = p.observations.field(&self.field)?.clone();
            if x.item_shape().len() != 1 {
                return Err(GasError::Shape(
                    "benchmark input must be coordinate vectors".into(),
                ));
            }
            for v in x.values_mut() {
                if !v.is_finite() {
                    *v = T::ZERO;
                }
            }
            let values = if matches!(self.benchmark, Benchmark::Bbob { .. }) {
                cx.stats.host_reward_evaluations += x.rows() as u64;
                host::values(self.benchmark, &x)?
            } else {
                let graph = self.benchmark.graph(x.width(), false)?;
                let mut inputs = vec![x];
                inputs.extend(host::constants::<T>(&graph)?);
                cx.evaluate(&graph.expression, &inputs)
                    .await?
                    .values()
                    .to_vec()
            };
            let mut rewards = RewardBatch::new(
                values,
                Provenance {
                    input_version: input.map_or(0, |i| i.version),
                    population_version: p.version,
                    stage: stage.into(),
                },
            );
            for (i, state) in p.validity.iter().enumerate() {
                rewards.valid[i] &= !state.invalid;
            }
            Ok(rewards)
        })
    }
}
impl<T: Real> GradientProvider<T> for BenchmarkModel {
    fn id(&self) -> String {
        match self.benchmark.gradient_execution() {
            GradientExecution::HostCentralDifference => format!(
                "central-difference-gradient/{:?}/{:?}/v1",
                self.benchmark, self.direction
            ),
            _ => format!(
                "analytic-gradient/{:?}/{:?}/v1",
                self.benchmark, self.direction
            ),
        }
    }
    fn gradient<'a>(
        &'a self,
        p: &'a Population<T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>> {
        Box::pin(async move {
            let mut x = p.observations.field(&self.field)?.clone();
            for v in x.values_mut() {
                if !v.is_finite() {
                    *v = T::ZERO;
                }
            }
            let mut gradient = if matches!(self.benchmark, Benchmark::Bbob { .. }) {
                let (gradient, evaluations) =
                    host::central_gradient(self.benchmark, &x, |i| p.validity[i].eligible(true))?;
                cx.stats.host_gradient_evaluations += evaluations;
                gradient
            } else {
                let graph = self.benchmark.graph(x.width(), true)?;
                let mut inputs = vec![x];
                inputs.extend(host::constants::<T>(&graph)?);
                cx.evaluate(&graph.expression, &inputs).await?
            };
            let width = gradient.width();
            let sign = if self.direction == ObjectiveDirection::Minimize {
                T::ONE
            } else {
                -T::ONE
            };
            for (i, row) in gradient.values_mut().chunks_mut(width).enumerate() {
                for x in row {
                    *x = if p.validity[i].eligible(true) {
                        *x * sign
                    } else {
                        T::ZERO
                    };
                }
            }
            Ok(gradient)
        })
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RunConfig {
    pub benchmark: Benchmark,
    /// Optional independent force potential (reward objective remains benchmark).
    pub potential: Option<Benchmark>,
    /// General-dimensional metric and optional curvature at the actual O query.
    pub physics_metric: Option<physics_metric::PhysicsMetricConfig>,
    /// Reward the walkers with their share of a population-geometry action
    /// instead of the benchmark objective. Requires `gas.geometry`. The force
    /// comes from `potential`, or vanishes (a free gas) when there is none.
    pub geometry_reward: Option<algorithmic_gas::tessellation::GeometryReward>,
    /// Translate the reward optimum; empty means zero in every coordinate.
    pub reward_shift: Vec<f64>,
    pub walkers: usize,
    pub dimensions: usize,
    pub initial_lower: f64,
    pub initial_upper: f64,
    pub gas: GasConfig,
}
impl Default for RunConfig {
    fn default() -> Self {
        let gas = GasConfig {
            boundary: BoundaryPolicy::AbsorbingBox {
                field: "positions".into(),
                domain: BoxDomain {
                    lower: vec![-5.12; 2],
                    upper: vec![5.12; 2],
                },
            },
            ..GasConfig::default()
        };
        Self {
            benchmark: Benchmark::Rastrigin,
            potential: None,
            physics_metric: None,
            geometry_reward: None,
            reward_shift: vec![],
            walkers: 256,
            dimensions: 2,
            initial_lower: -1.,
            initial_upper: 1.,
            gas,
        }
    }
}
impl RunConfig {
    pub fn validate(&self) -> Result<()> {
        self.benchmark.validate(self.dimensions)?;
        if let Some(potential) = self.potential {
            potential.validate(self.dimensions)?;
        }
        if self.physics_metric.is_some() && !self.benchmark.supports_physics_metric() {
            return Err(GasError::Capability(format!(
                "the conditional physics metric has no closed-form jet for `{}`",
                self.benchmark.id()
            )));
        }
        if !self.reward_shift.is_empty()
            && (self.reward_shift.len() != self.dimensions
                || self.reward_shift.iter().any(|x| !x.is_finite()))
        {
            return Err(GasError::Configuration(
                "reward shift must be empty or a finite d-vector".into(),
            ));
        }
        if self.walkers == 0
            || self.walkers > 1_000_000
            || self
                .walkers
                .checked_mul(self.dimensions)
                .is_none_or(|n| n > self.gas.max_batch_elements)
        {
            return Err(GasError::Configuration(
                "population exceeds configured memory budget".into(),
            ));
        }
        if !self.initial_lower.is_finite()
            || !self.initial_upper.is_finite()
            || self.initial_lower > self.initial_upper
        {
            return Err(GasError::Configuration(
                "initial bounds must be finite and ordered".into(),
            ));
        }
        if self.geometry_reward.is_some()
            && (self.gas.geometry.is_none() || self.physics_metric.is_some())
        {
            return Err(GasError::Configuration(
                "a geometry reward needs the gas geometry stage and excludes the physics metric"
                    .into(),
            ));
        }
        Ok(())
    }
    pub fn initial_population<T: Real>(&self) -> Result<Population<T>> {
        self.validate()?;
        if self
            .reward_shift
            .iter()
            .any(|&x| !T::from_f64(x).is_finite())
        {
            return Err(GasError::Configuration(
                "reward shift is not representable in the run precision".into(),
            ));
        }
        let lower = T::from_f64(self.initial_lower);
        let upper = T::from_f64(self.initial_upper);
        let width = upper - lower;
        // Equal bounds start every walker at one point.
        if !lower.is_finite() || !upper.is_finite() || !width.is_finite() || width < T::ZERO {
            return Err(GasError::Configuration(
                "initial bounds are not representable in the run precision".into(),
            ));
        }
        let mut values = Vec::with_capacity(self.walkers * self.dimensions);
        for i in 0..self.walkers {
            let mut rng = RandomStream::new(self.gas.seed, 0, Stream::Initialize, i as u64, 0);
            for _ in 0..self.dimensions {
                values.push(lower + width * rng.uniform::<T>());
            }
        }
        let mut obs = ObservationBatch::positions(TensorBatch::vectors(
            self.walkers,
            self.dimensions,
            values,
        )?);
        if let KineticKind::Baoab { velocities, .. } = &self.gas.kinetic.integrator {
            obs.fields.insert(
                velocities.clone(),
                TensorBatch::vectors(
                    self.walkers,
                    self.dimensions,
                    vec![T::ZERO; self.walkers * self.dimensions],
                )?,
            );
        }
        Population::new(obs)
    }
    /// The Einstein-Hilbert gas of `GasConfig::einstein_hilbert`: 500 walkers
    /// in three dimensions, all starting at the origin at rest, rewarded with
    /// their share of the Einstein-Hilbert action and subject to no potential.
    pub fn einstein_hilbert() -> Result<Self> {
        Ok(Self {
            geometry_reward: Some(algorithmic_gas::tessellation::GeometryReward::default()),
            walkers: 500,
            dimensions: 3,
            initial_lower: 0.,
            initial_upper: 0.,
            gas: GasConfig::einstein_hilbert(0.33, 0.002)?,
            ..Self::default()
        })
    }
    pub async fn build<T: Real>(&self) -> Result<AlgorithmicGas<T>> {
        if let Some(reward) = &self.geometry_reward {
            let builder = GasBuilder::new(self.initial_population()?, reward.clone())
                .config(self.gas.clone());
            let builder = match (self.potential, &self.gas.kinetic.integrator) {
                (Some(benchmark), _) => builder.gradient(BenchmarkModel {
                    benchmark,
                    field: "positions".into(),
                    direction: ObjectiveDirection::Minimize,
                }),
                (None, KineticKind::Baoab { velocities, .. }) => builder.gradient(
                    algorithmic_gas::tessellation::ZeroPotential::new(velocities.clone()),
                ),
                (None, _) => builder,
            };
            return builder.build().await;
        }
        let model = BenchmarkModel {
            benchmark: self.benchmark,
            field: "positions".into(),
            direction: self.gas.fitness.direction,
        };
        let gradient = BenchmarkModel {
            benchmark: self.potential.unwrap_or(self.benchmark),
            direction: if self.potential.is_some() {
                ObjectiveDirection::Minimize
            } else {
                model.direction
            },
            ..model.clone()
        };
        let builder = GasBuilder::new(
            self.initial_population()?,
            ShiftedReward {
                noise: match self.benchmark {
                    Benchmark::StochasticGaussian { std } => Some(host::RewardNoise {
                        std,
                        seed: self.gas.seed,
                    }),
                    _ => None,
                },
                model,
                shift: self.reward_shift.clone(),
            },
        )
        .config(self.gas.clone())
        .gradient(gradient);
        let builder = if let Some(config) = &self.physics_metric {
            builder.operators(physics_metric::PhysicsMetricOperators {
                config: config.clone(),
                benchmark: self.benchmark,
                reward_shift: self.reward_shift.clone(),
            })
        } else {
            builder
        };
        builder.build().await
    }
}

/// Stateless translated reward; the potential provider is configured separately.
struct ShiftedReward {
    model: BenchmarkModel,
    shift: Vec<f64>,
    /// Observation noise of a stochastic objective, drawn after the deterministic value.
    noise: Option<host::RewardNoise>,
}
impl<T: Real> RewardSource<T> for ShiftedReward {
    fn id(&self) -> String {
        let id = if self.shift.is_empty() {
            <BenchmarkModel as RewardSource<T>>::id(&self.model)
        } else {
            format!("shifted/{:?}/{:?}/v1", self.model.benchmark, self.shift)
        };
        match &self.noise {
            Some(noise) => format!("noisy/{:?}/{}/{id}", noise.std, noise.seed),
            None => id,
        }
    }
    fn evaluate<'a>(
        &'a self,
        p: &'a Population<T>,
        input: Option<&'a InputBatch<T>>,
        stage: &'a str,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<T>> {
        Box::pin(async move {
            if self.shift.is_empty() {
                let mut rewards = self.model.evaluate(p, input, stage, cx).await?;
                if let Some(noise) = &self.noise {
                    noise.apply(&mut rewards, p, stage);
                }
                return Ok(rewards);
            }
            let mut translated = p.clone();
            let field = translated.observations.field_mut("positions")?;
            let width = field.width();
            for row in field.values_mut().chunks_mut(width) {
                for (value, shift) in row.iter_mut().zip(&self.shift) {
                    let translated_value = *value - T::from_f64(*shift);
                    if value.is_finite() && !translated_value.is_finite() {
                        return Err(GasError::Numerical(
                            "translated reward coordinate overflowed the run precision".into(),
                        ));
                    }
                    *value = translated_value;
                }
            }
            let mut rewards = self.model.evaluate(&translated, input, stage, cx).await?;
            if let Some(noise) = &self.noise {
                noise.apply(&mut rewards, p, stage);
            }
            Ok(rewards)
        })
    }
}

pub mod qft_experiments;
