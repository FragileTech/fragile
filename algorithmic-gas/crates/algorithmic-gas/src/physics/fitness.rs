//! Exact conditional derivatives with frozen donor coordinates and eligibility.
//! Global statistics are differentiated, not stopped. Prefix/suffix Welford
//! summaries cost O(N) once, then every target query costs O(1) in population size.
use super::jet::{Jet, JetSpace};
use crate::{
    GasError, Real, Result,
    error::require,
    fitness::{FitnessPipeline, PositiveMap, Standardizer},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Objective {
    Sphere,
    Quadratic { matrix: Vec<f64> },
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
}
impl Objective {
    pub fn evaluate<T: Real>(&self, x: &[Jet<T>]) -> Result<Jet<T>> {
        require(!x.is_empty(), "objective dimension")?;
        let c = |v| x[0].constant(v);
        let mut out = c(0.);
        match self {
            Self::Sphere => {
                for a in x {
                    out = out.add(&a.mul(a));
                }
            }
            Self::Quadratic { matrix } => {
                require(
                    matrix.len() == x.len() * x.len() && matrix.iter().all(|v| v.is_finite()),
                    "quadratic matrix shape/finiteness",
                )?;
                for i in 0..x.len() {
                    for j in 0..x.len() {
                        out = out.add(
                            &x[i]
                                .mul(&x[j])
                                .scale(T::from_f64(0.5 * matrix[i * x.len() + j])),
                        );
                    }
                }
            }
            Self::Rastrigin => {
                for a in x {
                    out = out.add(
                        &a.mul(a)
                            .sub(
                                &a.scale(T::from_f64(std::f64::consts::TAU))
                                    .cos()
                                    .scale(T::from_f64(10.)),
                            )
                            .add(&c(10.)),
                    );
                }
            }
            Self::Rosenbrock => {
                for p in x.windows(2) {
                    out = out
                        .add(&p[1].sub(&p[0].mul(&p[0])).pow(2.).scale(T::from_f64(100.)))
                        .add(&p[0].sub(&c(1.)).pow(2.));
                }
            }
            Self::StyblinskiTang => {
                for a in x {
                    out = out.add(
                        &a.pow(4.)
                            .sub(&a.pow(2.).scale(T::from_f64(16.)))
                            .add(&a.scale(T::from_f64(5.)))
                            .scale(T::from_f64(0.5)),
                    );
                }
            }
        }
        out.validate()?;
        Ok(out)
    }
    pub fn value<T: Real>(&self, x: &[T]) -> Result<T> {
        let space = JetSpace::new(x.len(), 0)?;
        self.evaluate(&x.iter().map(|&v| space.constant(v)).collect::<Vec<_>>())
            .map(|j| j.value())
    }
}
#[derive(Clone, Copy, Debug)]
struct Moments<T: Real> {
    n: usize,
    mean: T,
    m2: T,
}
impl<T: Real> Moments<T> {
    fn empty() -> Self {
        Self {
            n: 0,
            mean: T::ZERO,
            m2: T::ZERO,
        }
    }
    fn merge(self, other: Self) -> Self {
        if self.n == 0 {
            return other;
        }
        if other.n == 0 {
            return self;
        }
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        Self {
            n,
            mean: self.mean + delta * T::from_f64(other.n as f64 / n as f64),
            m2: self.m2
                + other.m2
                + delta * delta * T::from_f64(self.n as f64 * other.n as f64 / n as f64),
        }
    }
}
fn exclusions<T: Real>(values: &[T], alive: &[bool]) -> Vec<Moments<T>> {
    let n = values.len();
    let mut prefix = vec![Moments::empty(); n + 1];
    let mut suffix = prefix.clone();
    let row = |i| {
        if alive[i] {
            Moments {
                n: 1,
                mean: values[i],
                m2: T::ZERO,
            }
        } else {
            Moments::empty()
        }
    };
    for i in 0..n {
        prefix[i + 1] = prefix[i].merge(row(i));
    }
    for i in (0..n).rev() {
        suffix[i] = row(i).merge(suffix[i + 1]);
    }
    (0..n).map(|i| prefix[i].merge(suffix[i + 1])).collect()
}
#[derive(Clone, Debug)]
pub struct ConditionalFitnessCache<T: Real> {
    reward: Vec<Moments<T>>,
    distance: Vec<Moments<T>>,
    alive: Vec<bool>,
    pipeline: FitnessPipeline,
}
impl<T: Real> ConditionalFitnessCache<T> {
    /// Rewards here are raw objective values; direction is applied exactly once.
    /// Distances include the pipeline's distance floor, as in FitnessPipeline::evaluate.
    pub fn new(
        rewards: &[T],
        distances: &[T],
        alive: &[bool],
        pipeline: &FitnessPipeline,
    ) -> Result<Self> {
        pipeline.validate()?;
        require(
            !rewards.is_empty() && rewards.len() == distances.len() && rewards.len() == alive.len(),
            "conditional cache shape",
        )?;
        require(
            alive.iter().any(|v| *v),
            "conditional cache has no eligible row",
        )?;
        require(
            rewards
                .iter()
                .zip(distances)
                .zip(alive)
                .all(|((&r, &d), &a)| !a || (r.is_finite() && d.is_finite() && d >= T::ZERO)),
            "finite eligible measurements",
        )?;
        for (s, m) in [
            (&pipeline.reward_standardizer, &pipeline.reward_map),
            (&pipeline.diversity_standardizer, &pipeline.diversity_map),
        ] {
            if !matches!(
                (s, m),
                (Standardizer::Global { .. }, PositiveMap::Logistic { .. })
            ) {
                return Err(GasError::Capability("cached conditional jets require global/logistic channels; no frozen-statistics fallback".into()));
            }
        }
        let rewards: Vec<_> = rewards
            .iter()
            .map(|&r| pipeline.direction.orient(r))
            .collect();
        Ok(Self {
            reward: exclusions(&rewards, alive),
            distance: exclusions(distances, alive),
            alive: alive.to_vec(),
            pipeline: pipeline.clone(),
        })
    }
    pub fn evaluate(&self, target: usize, reward: &Jet<T>, distance: &Jet<T>) -> Result<Jet<T>> {
        require(
            target < self.alive.len() && self.alive[target],
            "conditional target must be eligible",
        )?;
        let channel = |value: &Jet<T>,
                       m: Moments<T>,
                       standardizer: &Standardizer,
                       map: &PositiveMap,
                       exponent: f64| {
            let (sigma, amplitude, floor) = match (standardizer, map) {
                (
                    Standardizer::Global { sigma_min },
                    PositiveMap::Logistic { amplitude, floor },
                ) => (*sigma_min, *amplitude, *floor),
                _ => unreachable!(),
            };
            let delta = value.sub(&value.space.constant(m.mean));
            let n = (m.n + 1) as f64;
            let mean = value
                .space
                .constant(m.mean)
                .add(&delta.scale(T::from_f64(1. / n)));
            let variance = value
                .space
                .constant(m.m2 / n_as::<T>(n))
                .add(&delta.mul(&delta).scale(T::from_f64(m.n as f64 / (n * n))));
            value
                .sub(&mean)
                .mul(&variance.add(&value.constant(sigma * sigma)).pow(-0.5))
                .logistic(amplitude, floor)
                .pow(exponent)
        };
        let r = reward.scale(self.pipeline.direction.orient(T::ONE));
        let out = channel(
            &r,
            self.reward[target],
            &self.pipeline.reward_standardizer,
            &self.pipeline.reward_map,
            self.pipeline.reward_exponent,
        )
        .mul(&channel(
            distance,
            self.distance[target],
            &self.pipeline.diversity_standardizer,
            &self.pipeline.diversity_map,
            self.pipeline.diversity_exponent,
        ));
        out.validate()?;
        Ok(out)
    }
}
fn n_as<T: Real>(n: f64) -> T {
    T::from_f64(n)
}

/// Frozen source position and velocity. Only the destination query is variable.
pub fn companion_distance<T: Real>(
    query: &[Jet<T>],
    source: &[T],
    velocity_squared: T,
    floor: T,
) -> Result<Jet<T>> {
    require(
        !query.is_empty()
            && query.len() == source.len()
            && source.iter().all(|x| x.is_finite())
            && velocity_squared.is_finite()
            && velocity_squared >= T::ZERO
            && floor.is_finite()
            && floor > T::ZERO,
        "smooth companion distance inputs",
    )?;
    let mut squared = query[0].space.constant(velocity_squared + floor * floor);
    for (x, &y) in query.iter().zip(source) {
        squared = squared.add(&x.sub(&x.space.constant(y)).pow(2.));
    }
    Ok(squared.pow(0.5))
}

/// Exact O(N) fallback when multiple measurements depend on a query, or when
/// local normalization is configured. Log weights are query jets for the target
/// neighborhood; both their derivatives and measurement derivatives are retained.
pub fn pipeline_from_measurement_jets<T: Real>(
    rewards: &[Jet<T>],
    distances: &[Jet<T>],
    alive: &[bool],
    target: usize,
    pipeline: &FitnessPipeline,
    reward_log_weights: Option<&[Jet<T>]>,
    distance_log_weights: Option<&[Jet<T>]>,
) -> Result<Jet<T>> {
    pipeline.validate()?;
    let n = alive.len();
    require(
        n > 0
            && n <= 16384
            && target < n
            && alive[target]
            && rewards.len() == n
            && distances.len() == n,
        "full conditional jet shape/capacity",
    )?;
    let channel = |values: &[Jet<T>],
                   standardizer: &Standardizer,
                   map: &PositiveMap,
                   logs: Option<&[Jet<T>]>,
                   exponent: f64|
     -> Result<Jet<T>> {
        let value = &values[target];
        let constant = |v| value.constant(v);
        let sigma = match standardizer {
            Standardizer::Global { sigma_min } | Standardizer::Local { sigma_min, .. } => {
                *sigma_min
            }
            _ => {
                return Err(GasError::Capability(
                    "sample standard deviation is not a globally smooth jet channel".into(),
                ));
            }
        };
        let (amplitude, floor) = match map {
            PositiveMap::Logistic { amplitude, floor } => (*amplitude, *floor),
            _ => {
                return Err(GasError::Capability(
                    "the piecewise map is unavailable for smooth high-order jets".into(),
                ));
            }
        };
        let mut indices: Vec<_> = (0..n).filter(|&i| alive[i]).collect();
        let weights = if let Standardizer::Local { include_self, .. } = standardizer {
            let logs = logs.ok_or_else(|| {
                GasError::Capability(
                    "local conditional channel requires differentiated neighborhood log weights"
                        .into(),
                )
            })?;
            require(logs.len() == n, "local log-weight shape")?;
            if !include_self {
                indices.retain(|&i| i != target);
            }
            if indices.is_empty() {
                indices = (0..n).filter(|&i| alive[i]).collect();
                vec![constant(1.); indices.len()]
            } else {
                let maximum = indices
                    .iter()
                    .map(|&i| logs[i].value())
                    .fold(T::from_f64(f64::NEG_INFINITY), |a, b| a.max(b));
                require(maximum.is_finite(), "nonfinite local support")?;
                indices
                    .iter()
                    .map(|&i| logs[i].sub(&value.space.constant(maximum)).exp())
                    .collect()
            }
        } else {
            vec![constant(1.); indices.len()]
        };
        let anchor = values[indices[0]].value();
        let mut total = constant(0.);
        let mut centered = constant(0.);
        for (&i, w) in indices.iter().zip(&weights) {
            values[i].validate()?;
            total = total.add(w);
            centered = centered.add(&w.mul(&values[i].sub(&value.space.constant(anchor))));
        }
        require(total.value() > T::ZERO, "empty local support")?;
        let inverse = total.pow(-1.);
        let mean = centered.mul(&inverse).add(&value.space.constant(anchor));
        let mut variance = constant(0.);
        for (&i, w) in indices.iter().zip(&weights) {
            variance = variance.add(&w.mul(&values[i].sub(&mean).pow(2.)));
        }
        Ok(value
            .sub(&mean)
            .mul(
                &variance
                    .mul(&inverse)
                    .add(&constant(sigma * sigma))
                    .pow(-0.5),
            )
            .logistic(amplitude, floor)
            .pow(exponent))
    };
    let oriented: Vec<_> = rewards
        .iter()
        .map(|v| v.scale(pipeline.direction.orient(T::ONE)))
        .collect();
    let out = channel(
        &oriented,
        &pipeline.reward_standardizer,
        &pipeline.reward_map,
        reward_log_weights,
        pipeline.reward_exponent,
    )?
    .mul(&channel(
        distances,
        &pipeline.diversity_standardizer,
        &pipeline.diversity_map,
        distance_log_weights,
        pipeline.diversity_exponent,
    )?);
    out.validate()?;
    Ok(out)
}

/// Neighborhood log weights for a query replacing one position row. Supports
/// the smooth Euclidean/phase-space kernels of the production standardizer.
/// Donor selection remains frozen; this is only the normalization neighborhood.
pub fn local_log_weights<T: Real>(
    query: &[Jet<T>],
    points: &[Vec<T>],
    velocities: &[Vec<T>],
    alive: &[bool],
    target: usize,
    standardizer: &Standardizer,
) -> Result<Option<Vec<Jet<T>>>> {
    use crate::geometry::{Distance, Kernel};
    let Standardizer::Local {
        distance, kernel, ..
    } = standardizer
    else {
        return Ok(None);
    };
    let d = query.len();
    require(
        d > 0
            && target < points.len()
            && alive.len() == points.len()
            && points.iter().all(|x| x.len() == d),
        "local coordinates",
    )?;
    let constant = |v| query[0].constant(v);
    let mut logs = Vec::with_capacity(points.len());
    for (row, point) in points.iter().enumerate() {
        if !alive[row] {
            logs.push(constant(0.));
            continue;
        }
        if matches!(kernel, Kernel::Uniform) {
            logs.push(constant(0.));
            continue;
        }
        let mut squared = constant(0.);
        match distance {
            Distance::Euclidean {
                field,
                scales,
                periodic: None,
                ..
            } if field == "positions" => {
                require(
                    scales.is_empty()
                        || (scales.len() == d && scales.iter().all(|v| v.is_finite() && *v > 0.)),
                    "local Euclidean scales",
                )?;
                if row != target {
                    for a in 0..d {
                        squared = squared.add(
                            &query[a]
                                .sub(&query[0].space.constant(point[a]))
                                .scale(T::from_f64(1. / scales.get(a).copied().unwrap_or(1.)))
                                .pow(2.),
                        );
                    }
                }
            }
            Distance::PhaseSpace {
                positions,
                position_scale,
                velocity_scale,
                lambda,
                periodic: None,
                ..
            } if positions == "positions" => {
                require(
                    velocities.len() == points.len()
                        && velocities.iter().all(|v| v.len() == d)
                        && position_scale.is_finite()
                        && *position_scale > 0.
                        && velocity_scale.is_finite()
                        && *velocity_scale > 0.
                        && lambda.is_finite()
                        && *lambda >= 0.,
                    "local phase-space inputs",
                )?;
                if row != target {
                    for a in 0..d {
                        squared = squared.add(
                            &query[a]
                                .sub(&query[0].space.constant(point[a]))
                                .scale(T::from_f64(1. / position_scale))
                                .pow(2.),
                        );
                    }
                    let velocity = (0..d).fold(T::ZERO, |s, a| {
                        let v = velocities[target][a] - velocities[row][a];
                        s + v * v * T::from_f64(lambda / (velocity_scale * velocity_scale))
                    });
                    squared = squared.add(&query[0].space.constant(velocity));
                }
            }
            _ => {
                return Err(GasError::Capability(
                    "local jets require Euclidean/phase-space coordinates without periodic seams"
                        .into(),
                ));
            }
        }
        let log = match kernel {
            Kernel::Gaussian { width } => {
                require(
                    width.is_finite() && *width > 0.,
                    "positive local Gaussian width",
                )?;
                squared.scale(T::from_f64(-0.5 / (width * width)))
            }
            Kernel::Exponential { temperature } => {
                require(
                    temperature.is_finite() && *temperature > 0.,
                    "positive local kernel temperature",
                )?;
                if row == target {
                    constant(0.)
                } else if matches!(distance, Distance::Euclidean { squared: true, .. }) {
                    squared.scale(T::from_f64(-1. / temperature))
                } else {
                    require(
                        squared.value() > T::ZERO,
                        "exponential distance kernel is nonsmooth at coincident neighbors",
                    )?;
                    squared.pow(0.5).scale(T::from_f64(-1. / temperature))
                }
            }
            Kernel::Uniform => unreachable!(),
        };
        logs.push(log);
    }
    Ok(Some(logs))
}
