use crate::{
    GasError, ObservationBatch, Real, Result, RewardBatch,
    error::require,
    geometry::{AlgorithmicDistance, Distance, InteractionKernel, Kernel},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}
impl ObjectiveDirection {
    pub fn orient<T: Real>(self, x: T) -> T {
        if self == Self::Maximize { x } else { -x }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Standardizer {
    Global {
        sigma_min: f64,
    },
    Local {
        sigma_min: f64,
        distance: Distance,
        kernel: Kernel,
        #[serde(default)]
        include_self: bool,
    },
    /// Legacy sample standard deviation, with zero output for singleton or
    /// constant populations. The positive map remains a separate operation.
    LegacySample {
        epsilon: f64,
    },
}
impl Default for Standardizer {
    fn default() -> Self {
        Self::Global { sigma_min: 1e-3 }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct Statistics<T: Real> {
    pub mean: Vec<T>,
    pub scale: Vec<T>,
    /// Local empty/zero-support neighborhoods use global statistics.
    pub global_fallback: Vec<bool>,
}
impl Standardizer {
    pub fn validate(&self) -> Result<()> {
        let x = match self {
            Self::Global { sigma_min } | Self::Local { sigma_min, .. } => *sigma_min,
            Self::LegacySample { epsilon } => *epsilon,
        };
        require(
            x.is_finite() && x > 0.,
            "standardization regularizer must be positive",
        )
    }
    pub fn apply<T: Real>(
        &self,
        values: &[T],
        alive: &[bool],
        obs: &ObservationBatch<T>,
    ) -> Result<(Vec<T>, Statistics<T>)> {
        self.validate()?;
        require(values.len() == alive.len(), "standardizer shape")?;
        let indices: Vec<usize> = alive
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| v.then_some(i))
            .collect();
        if indices.is_empty() {
            return Err(GasError::Extinction);
        }
        if indices.iter().any(|&i| !values[i].is_finite()) {
            return Err(GasError::Numerical(
                "nonfinite alive fitness measurement".into(),
            ));
        }
        let n = values.len();
        let mut stats = Statistics {
            mean: vec![T::ZERO; n],
            scale: vec![T::ONE; n],
            global_fallback: vec![false; n],
        };
        match self {
            Self::Global { sigma_min } | Self::LegacySample { epsilon: sigma_min } => {
                // Welford avoids subtracting nearly equal squared magnitudes.
                let mut mean = T::ZERO;
                let mut m2 = T::ZERO;
                for (count, &i) in indices.iter().enumerate() {
                    let delta = values[i] - mean;
                    mean = mean + delta / T::from_f64((count + 1) as f64);
                    m2 = m2 + delta * (values[i] - mean);
                }
                let legacy = matches!(self, Self::LegacySample { .. });
                let denom = if legacy {
                    indices.len().saturating_sub(1).max(1)
                } else {
                    indices.len()
                };
                let var = (m2 / T::from_f64(denom as f64)).max(T::ZERO);
                let eps = T::from_f64(*sigma_min);
                let scale = if legacy {
                    var.sqrt() + eps
                } else {
                    (var + eps * eps).sqrt()
                };
                stats.mean.fill(mean);
                stats.scale.fill(scale);
            }
            Self::Local {
                sigma_min,
                distance,
                kernel,
                include_self,
            } => {
                distance.validate(obs)?;
                let kind = <Distance as AlgorithmicDistance<T>>::comparison_kind(distance);
                kernel.validate(kind)?;
                let (_, global) = Self::Global {
                    sigma_min: *sigma_min,
                }
                .apply(values, alive, obs)?;
                for &i in &indices {
                    let neighbors: Vec<_> = indices
                        .iter()
                        .copied()
                        .filter(|&j| *include_self || j != i)
                        .collect();
                    let mut logs = Vec::with_capacity(indices.len());
                    let mut max = T::from_f64(f64::NEG_INFINITY);
                    for &j in &neighbors {
                        let w = kernel.log_weight(distance.compare(obs, i, obs, j)?, kind)?;
                        max = max.max(w);
                        logs.push(w);
                    }
                    if neighbors.is_empty() {
                        stats.mean[i] = global.mean[i];
                        stats.scale[i] = global.scale[i];
                        stats.global_fallback[i] = true;
                        continue;
                    }
                    // Positive Gaussian support becoming -infinity is a
                    // numerical failure, not an empty mathematical support.
                    require(
                        max.is_finite(),
                        "local kernel log-weights underflowed; rescale geometry or bandwidth",
                    )?;
                    let mut total = T::ZERO;
                    let mut mean = T::ZERO;
                    for (&j, &w) in neighbors.iter().zip(&logs) {
                        let w = (w - max).exp();
                        total = total + w;
                        mean = mean + w * values[j];
                    }
                    mean = mean / total;
                    let mut var = T::ZERO;
                    for (&j, &w) in neighbors.iter().zip(&logs) {
                        let d = values[j] - mean;
                        var = var + (w - max).exp() * d * d;
                    }
                    let eps = T::from_f64(*sigma_min);
                    stats.mean[i] = mean;
                    stats.scale[i] = (var / total + eps * eps).sqrt();
                }
            }
        }
        let z = values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if alive[i] {
                    (v - stats.mean[i]) / stats.scale[i]
                } else {
                    T::ZERO
                }
            })
            .collect::<Vec<_>>();
        if z.iter().any(|z| !z.is_finite()) {
            return Err(GasError::Numerical("standardization overflow".into()));
        }
        Ok((z, stats))
    }
}
pub trait PositiveMapping<T: Real> {
    fn map(&self, z: T) -> Result<T>;
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PositiveMap {
    Logistic { amplitude: f64, floor: f64 },
    LegacyAsymmetric { floor: f64 },
}
impl Default for PositiveMap {
    fn default() -> Self {
        Self::Logistic {
            amplitude: 2.,
            floor: 1e-6,
        }
    }
}
impl PositiveMap {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Logistic { amplitude, floor } => require(
                amplitude.is_finite() && *amplitude > 0. && floor.is_finite() && *floor >= 0.,
                "invalid logistic amplitude or additive positivity floor",
            ),
            Self::LegacyAsymmetric { floor } => require(
                floor.is_finite() && *floor >= 0.,
                "invalid positivity floor",
            ),
        }
    }
}
impl<T: Real> PositiveMapping<T> for PositiveMap {
    fn map(&self, z: T) -> Result<T> {
        self.validate()?;
        let value = match self {
            Self::Logistic { amplitude, floor } => {
                let s = if z >= T::ZERO {
                    T::ONE / (T::ONE + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (T::ONE + e)
                };
                T::from_f64(*amplitude) * s + T::from_f64(*floor)
            }
            Self::LegacyAsymmetric { floor } => {
                if z <= T::ZERO {
                    z.exp() + T::from_f64(*floor)
                } else {
                    T::ONE + z.ln_1p() + T::from_f64(*floor)
                }
            }
        };
        if !value.is_finite() || value <= T::ZERO {
            return Err(GasError::Numerical(
                "positive map overflow/underflow; configure a representable floor".into(),
            ));
        }
        Ok(value)
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitnessPipeline {
    pub direction: ObjectiveDirection,
    pub reward_standardizer: Standardizer,
    pub diversity_standardizer: Standardizer,
    pub reward_map: PositiveMap,
    pub diversity_map: PositiveMap,
    pub reward_exponent: f64,
    pub diversity_exponent: f64,
    pub distance_floor: f64,
}
impl Default for FitnessPipeline {
    fn default() -> Self {
        Self {
            direction: ObjectiveDirection::Minimize,
            reward_standardizer: Standardizer::default(),
            diversity_standardizer: Standardizer::default(),
            reward_map: PositiveMap::default(),
            diversity_map: PositiveMap::default(),
            reward_exponent: 1.,
            diversity_exponent: 1.,
            distance_floor: 1e-3,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct FitnessBatch<T: Real> {
    pub oriented_reward: Vec<T>,
    pub separation: Vec<T>,
    pub reward_z: Vec<T>,
    pub diversity_z: Vec<T>,
    pub fitness: Vec<T>,
    pub reward_stats: Statistics<T>,
    pub diversity_stats: Statistics<T>,
    pub population_version: u64,
    pub stage: String,
}
impl<T: Real> FitnessBatch<T> {
    pub fn validate(&self, alive: &[bool], version: u64) -> Result<()> {
        require(
            self.reward_stats.global_fallback.len() == alive.len()
                && self.diversity_stats.global_fallback.len() == alive.len(),
            "statistics fallback shape",
        )?;
        require(
            self.population_version == version,
            "stale fitness population version",
        )?;
        for values in [
            &self.oriented_reward,
            &self.separation,
            &self.reward_z,
            &self.diversity_z,
            &self.fitness,
            &self.reward_stats.mean,
            &self.reward_stats.scale,
            &self.diversity_stats.mean,
            &self.diversity_stats.scale,
        ] {
            require(values.len() == alive.len(), "fitness batch shape")?;
            require(
                values.iter().zip(alive).all(|(v, &a)| !a || v.is_finite()),
                "nonfinite alive fitness value",
            )?;
        }
        require(
            alive.iter().enumerate().all(|(i, &a)| {
                !a || (self.fitness[i] > T::ZERO
                    && self.reward_stats.scale[i] > T::ZERO
                    && self.diversity_stats.scale[i] > T::ZERO
                    && self.separation[i] >= T::ZERO)
            }),
            "invalid positive fitness, scale or separation",
        )
    }
}
impl FitnessPipeline {
    pub fn validate(&self) -> Result<()> {
        self.reward_standardizer.validate()?;
        self.diversity_standardizer.validate()?;
        self.reward_map.validate()?;
        self.diversity_map.validate()?;
        require(
            [self.reward_exponent, self.diversity_exponent]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.),
            "fitness exponents must be finite and nonnegative",
        )?;
        require(
            self.distance_floor.is_finite() && self.distance_floor > 0.,
            "distance measurement regularizer must be positive",
        )
    }
    pub fn evaluate<T: Real>(
        &self,
        rewards: &RewardBatch<T>,
        distance: &[T],
        alive: &[bool],
        obs: &ObservationBatch<T>,
        version: u64,
    ) -> Result<FitnessBatch<T>> {
        self.validate()?;
        rewards.validate(alive.len())?;
        require(distance.len() == alive.len(), "diversity scalar shape")?;
        let oriented_reward: Vec<T> = rewards
            .raw
            .iter()
            .map(|&x| self.direction.orient(x))
            .collect();
        let floor = T::from_f64(self.distance_floor);
        let separation = distance
            .iter()
            .map(|&x| (x * x + floor * floor).sqrt())
            .collect::<Vec<_>>();
        let (reward_z, reward_stats) =
            self.reward_standardizer
                .apply(&oriented_reward, alive, obs)?;
        let (diversity_z, diversity_stats) =
            self.diversity_standardizer.apply(&separation, alive, obs)?;
        let fitness = self.combine(&reward_z, &diversity_z, alive)?;
        Ok(FitnessBatch {
            oriented_reward,
            separation,
            reward_z,
            diversity_z,
            fitness,
            reward_stats,
            diversity_stats,
            population_version: version,
            stage: "pre_clone".into(),
        })
    }
    pub fn combine<T: Real>(
        &self,
        reward_z: &[T],
        diversity_z: &[T],
        alive: &[bool],
    ) -> Result<Vec<T>> {
        require(
            reward_z.len() == alive.len() && diversity_z.len() == alive.len(),
            "fitness channel shape",
        )?;
        let mut output = vec![T::ZERO; alive.len()];
        for i in 0..alive.len() {
            if alive[i] {
                let r = self.reward_map.map(reward_z[i])?;
                let d = self.diversity_map.map(diversity_z[i])?;
                let f = r.powf(T::from_f64(self.reward_exponent))
                    * d.powf(T::from_f64(self.diversity_exponent));
                if !f.is_finite() || f <= T::ZERO {
                    return Err(GasError::Numerical(
                        "fitness channel combination overflow".into(),
                    ));
                }
                output[i] = f;
            }
        }
        Ok(output)
    }
}
