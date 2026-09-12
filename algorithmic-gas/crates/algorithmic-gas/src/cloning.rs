use crate::{
    ExecutionContext, GasError, Population, Real, Result,
    donor::{CompanionBatch, DonorPool, SourceRef},
    error::require,
    noise::{Noise, NoiseRequest, NoiseSource},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WeightedDonor {
    pub pool_index: u32,
    pub weight: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CloneChoice {
    pub donors: Vec<WeightedDonor>,
    pub accepted: bool,
    pub revival: bool,
    /// Actual decision probability; custom providers may leave this unavailable.
    #[serde(default)]
    pub probability: Option<f64>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClonePlan {
    pub population_version: u64,
    pub sources: Vec<SourceRef>,
    pub choices: Vec<CloneChoice>,
    pub mutual: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CloneDecision {
    pub epsilon: f64,
    pub saturation: f64,
}
impl Default for CloneDecision {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            saturation: 1.,
        }
    }
}
impl CloneDecision {
    pub fn validate(&self) -> Result<()> {
        require(
            self.epsilon.is_finite()
                && self.epsilon > 0.
                && self.saturation.is_finite()
                && self.saturation > 0.,
            "invalid clone acceptance regularizers",
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn plan<T: Real>(
        &self,
        p: &Population<T>,
        pool: &DonorPool<T>,
        companions: &CompanionBatch,
        fitness: &[T],
        donor_fitness: &[T],
        alive: &[bool],
        seed: u64,
        step: u64,
    ) -> Result<ClonePlan> {
        self.validate()?;
        companions.validate(pool)?;
        require(
            companions.count == 1,
            "multi-donor recombination is an extension point, not implemented; cloning count must be one",
        )?;
        require(
            companions.rows == p.len()
                && fitness.len() == p.len()
                && donor_fitness.len() == pool.sources.len()
                && alive.len() == p.len(),
            "clone plan shapes",
        )?;
        let revival: Vec<u32> = pool
            .sources
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s.frame == pool.current_frame).then_some(i as u32))
            .collect();
        if revival.is_empty() {
            return Err(GasError::Extinction);
        }
        let mut choices = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            if !alive[i] {
                let mut rng = RandomStream::new(seed, step, Stream::Revival, i as u64, 0);
                choices.push(CloneChoice {
                    donors: vec![WeightedDonor {
                        pool_index: revival[rng.index(revival.len())],
                        weight: 1.,
                    }],
                    accepted: true,
                    revival: true,
                    probability: Some(1.),
                });
                continue;
            }
            let donor = companions.row(i).next();
            let mut accepted = false;
            let mut acceptance_probability = T::ZERO;
            if let Some(j) = donor {
                let f = fitness[i];
                let d = donor_fitness[j as usize];
                if !f.is_finite() || f <= T::ZERO || !d.is_finite() || d <= T::ZERO {
                    return Err(GasError::Numerical(
                        "clone fitness must be positive and finite".into(),
                    ));
                }
                let probability =
                    ((d - f) / (f + T::from_f64(self.epsilon)) / T::from_f64(self.saturation))
                        .max(T::ZERO)
                        .min(T::ONE);
                acceptance_probability = probability;
                let mut rng = RandomStream::new(seed, step, Stream::Accept, i as u64, 0);
                accepted = rng.uniform::<T>() < probability;
            }
            choices.push(CloneChoice {
                donors: donor
                    .into_iter()
                    .map(|pool_index| WeightedDonor {
                        pool_index,
                        weight: 1.,
                    })
                    .collect(),
                accepted,
                revival: false,
                probability: Some(acceptance_probability.to_f64()),
            });
        }
        Ok(ClonePlan {
            population_version: p.version,
            sources: pool.sources.clone(),
            choices,
            mutual: companions.mutual,
        })
    }
}

impl ClonePlan {
    pub fn validate<T: Real>(&self, p: &Population<T>, pool: &DonorPool<T>) -> Result<()> {
        require(
            self.population_version == p.version && self.sources == pool.sources,
            "clone plan snapshot is stale",
        )?;
        require(self.choices.len() == p.len(), "clone recipient count")?;
        for choice in &self.choices {
            if choice.accepted {
                require(
                    choice.donors.len() == 1 && choice.donors[0].weight == 1.,
                    "literal cloning requires exactly one unit-weight donor",
                )?;
                require(
                    (choice.donors[0].pool_index as usize) < pool.sources.len(),
                    "clone donor index",
                )?;
            }
        }
        Ok(())
    }
    /// Separate destination storage is mandatory, including swaps, chains, and
    /// repeated donors. Opaque state and numerical fields use the same source.
    pub fn apply_literal<T: Real>(
        &self,
        p: &Population<T>,
        pool: &DonorPool<T>,
    ) -> Result<Population<T>> {
        self.validate(p, pool)?;
        let mut output = p.clone();
        for (i, choice) in self.choices.iter().enumerate() {
            if !choice.accepted {
                continue;
            }
            let j = choice.donors[0].pool_index as usize;
            for (name, field) in &mut output.observations.fields {
                field.replace_row(i, pool.population.observations.field(name)?.row(j)?)?;
            }
            output.rewards.raw[i] = pool.population.rewards.raw[j];
            output.rewards.valid[i] = pool.population.rewards.valid[j];
            output.validity[i] = pool.population.validity[j];
            if let Some(states) = &mut output.states {
                let source = pool
                    .population
                    .states
                    .as_ref()
                    .ok_or_else(|| GasError::Domain("missing donor state".into()))?;
                states.snapshots[i] = source.snapshots[j].clone();
            }
            output.generations[i] = p.generations[i]
                .checked_add(1)
                .ok_or_else(|| GasError::Numerical("generation overflow".into()))?;
        }
        output.version = p
            .version
            .checked_add(1)
            .ok_or_else(|| GasError::Numerical("population version overflow".into()))?;
        Ok(output)
    }
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CloneTransform {
    pub position_field: Option<String>,
    pub jitter: Option<Noise>,
    pub jitter_amplitude: f64,
    pub velocity_field: Option<String>,
    pub restitution: Option<f64>,
}
impl CloneTransform {
    pub fn validate<T: Real>(&self, p: &Population<T>, mutual: bool, count: usize) -> Result<()> {
        require(
            self.jitter_amplitude.is_finite() && self.jitter_amplitude >= 0.,
            "invalid clone jitter amplitude",
        )?;
        if let Some(noise) = &self.jitter {
            let field = self
                .position_field
                .as_ref()
                .ok_or_else(|| GasError::MissingField("clone position field".into()))?;
            let x = p.observations.field(field)?;
            require(
                x.item_shape().len() == 1,
                "jitter requires coordinate vectors",
            )?;
            noise.validate(&p.observations, p.len(), x.width())?;
        }
        if let Some(a) = self.restitution {
            require(
                a.is_finite() && (0.0..=1.0).contains(&a),
                "restitution must be in [0,1]",
            )?;
            require(
                mutual && count == 1,
                "velocity restitution requires disjoint current mutual pairs",
            )?;
            let name = self
                .velocity_field
                .as_ref()
                .ok_or_else(|| GasError::MissingField("restitution velocity field".into()))?;
            require(
                p.observations.field(name)?.item_shape().len() == 1,
                "restitution requires velocity vectors",
            )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub async fn apply<T: Real>(
        &self,
        before: &Population<T>,
        after: &mut Population<T>,
        pool: &DonorPool<T>,
        plan: &ClonePlan,
        companions: &CompanionBatch,
        seed: u64,
        step: u64,
        cx: &mut ExecutionContext,
    ) -> Result<Vec<String>> {
        self.validate(before, companions.mutual, companions.count)?;
        let mut changed = vec![];
        if let Some(noise) = &self.jitter {
            let field = self.position_field.as_ref().unwrap();
            let d = after.observations.field(field)?.width();
            let eta = noise
                .sample(
                    &after.observations,
                    NoiseRequest {
                        rows: after.len(),
                        dimension: d,
                        seed,
                        step,
                        stream: Stream::CloneNoise,
                        substep: 0,
                    },
                    cx,
                )
                .await?;
            let amplitude = T::from_f64(self.jitter_amplitude);
            let target = after.observations.field_mut(field)?;
            for (i, c) in plan.choices.iter().enumerate() {
                if c.accepted && !c.revival {
                    let row = target
                        .row(i)?
                        .iter()
                        .zip(eta.row(i)?)
                        .map(|(&x, &e)| x + amplitude * e)
                        .collect::<Vec<_>>();
                    target.replace_row(i, &row)?;
                }
            }
            changed.push(field.clone());
        }
        if let Some(a) = self.restitution {
            let field = self.velocity_field.as_ref().unwrap();
            let old = before.observations.field(field)?;
            let target = after.observations.field_mut(field)?;
            for i in 0..before.len() {
                if plan.choices[i].revival {
                    continue;
                }
                let Some(donor) = companions.row(i).next() else {
                    continue;
                };
                let source = pool.sources[donor as usize];
                require(
                    source.frame == pool.current_frame,
                    "restitution cannot use historical pairs",
                )?;
                let j = source.slot as usize;
                if i >= j {
                    continue;
                }
                let reverse = companions
                    .row(j)
                    .next()
                    .ok_or_else(|| GasError::Topology("missing reciprocal pair".into()))?;
                require(
                    pool.sources[reverse as usize].slot as usize == i && !plan.choices[j].revival,
                    "overlapping/nonreciprocal restitution pair",
                )?;
                if !plan.choices[i].accepted && !plan.choices[j].accepted {
                    continue;
                }
                let own = ((T::ONE + T::from_f64(a)) * T::from_f64(0.5)).to_f64();
                let partner = ((T::ONE - T::from_f64(a)) * T::from_f64(0.5)).to_f64();
                let source_i = pool.sources[reverse as usize];
                for (recipient, donor, weight) in [
                    (i, source_i, own),
                    (i, source, partner),
                    (j, source, own),
                    (j, source_i, partner),
                ] {
                    cx.record_influence(
                        "velocity_restitution",
                        field,
                        recipient as u32,
                        donor,
                        weight,
                    );
                }
                let mut vi = vec![];
                let mut vj = vec![];
                for (&x, &y) in old.row(i)?.iter().zip(old.row(j)?) {
                    let mean = (x + y) * T::from_f64(0.5);
                    vi.push(mean + T::from_f64(a) * (x - mean));
                    vj.push(mean + T::from_f64(a) * (y - mean));
                }
                target.replace_row(i, &vi)?;
                target.replace_row(j, &vj)?;
            }
            changed.push(field.clone());
        }
        Ok(changed)
    }
}
