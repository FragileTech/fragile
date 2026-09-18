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
    /// Revive through the same sampled current-donor kernel as living rows.
    #[serde(default)]
    pub revival_from_companion: bool,
    /// Living walkers clone only on steps divisible by this period; the
    /// acceptance probability is zero on every other step. Revival is ungated.
    #[serde(default = "every_step")]
    pub every: u64,
}
fn every_step() -> u64 {
    1
}
impl Default for CloneDecision {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            saturation: 1.,
            revival_from_companion: false,
            every: 1,
        }
    }
}
impl CloneDecision {
    pub fn validate(&self) -> Result<()> {
        require(
            // Fitness is validated strictly positive, so epsilon = 0 is safe.
            self.epsilon.is_finite()
                && self.epsilon >= 0.
                && self.saturation.is_finite()
                && self.saturation > 0.
                && self.every >= 1,
            "invalid clone acceptance regularizers or period",
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
                let donor = if self.revival_from_companion {
                    let donor = companions.row(i).next().ok_or_else(|| {
                        GasError::Capability(
                            "distance-weighted revival requires an eligible sampled current donor"
                                .into(),
                        )
                    })?;
                    require(
                        pool.sources[donor as usize].frame == pool.current_frame,
                        "distance-weighted revival requires a current donor",
                    )?;
                    donor
                } else {
                    revival[rng.index(revival.len())]
                };
                choices.push(CloneChoice {
                    donors: vec![WeightedDonor {
                        pool_index: donor,
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
                let probability = if step.is_multiple_of(self.every) {
                    ((d - f) / (f + T::from_f64(self.epsilon)) / T::from_f64(self.saturation))
                        .max(T::ZERO)
                        .min(T::ONE)
                } else {
                    T::ZERO
                };
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
    #[serde(default)]
    pub collision_rotation: CollisionRotation,
}
/// Orientation applied to the relative velocities of a collision component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollisionRotation {
    /// One Haar-random orthogonal matrix per component.
    #[default]
    Haar,
    /// Relative velocities keep their direction and are scaled by the restitution.
    Identity,
}
impl CloneTransform {
    pub fn validate<T: Real>(&self, p: &Population<T>) -> Result<()> {
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
            let name = self
                .velocity_field
                .as_ref()
                .ok_or_else(|| GasError::MissingField("restitution velocity field".into()))?;
            require(
                p.observations.field(name)?.item_shape().len() == 1
                    && (1..=256).contains(&p.observations.field(name)?.width()),
                "component restitution requires velocity vectors of dimension 1..=256",
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
        _companions: &CompanionBatch,
        seed: u64,
        step: u64,
        cx: &mut ExecutionContext,
    ) -> Result<Vec<String>> {
        self.validate(before)?;
        let components = if self.restitution.is_some() {
            accepted_current_components(before, pool, plan)?
        } else {
            Vec::new()
        };
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
                if c.accepted {
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
        if let Some(alpha) = self.restitution {
            let field = self.velocity_field.as_ref().unwrap();
            let d = before.observations.field(field)?.width();
            let n = before.len();
            let allocation = crate::memory::checked_mul(
                crate::memory::checked_mul(n, d * d + 6 * d + 5)?,
                std::mem::size_of::<f64>(),
            )?;
            crate::memory::enforce(allocation, cx.max_memory_bytes)?;
            let rotations = components
                .iter()
                .map(|members| match self.collision_rotation {
                    CollisionRotation::Haar => RandomStream::new(
                        seed,
                        step,
                        Stream::CollisionRotation,
                        members[0] as u64,
                        0,
                    )
                    .haar_orthogonal(d),
                    CollisionRotation::Identity => Ok((0..d * d)
                        .map(|k| if k % (d + 1) == 0 { 1. } else { 0. })
                        .collect()),
                })
                .collect::<Result<Vec<_>>>()?;
            let reports =
                apply_component_rotations(before, after, &components, &rotations, field, alpha)?;
            record_components(before, after, field, &reports, cx)?;
            changed.push(field.clone());
        }
        Ok(changed)
    }
}

/// One connected component of the undirected accepted recipient/current-donor
/// graph. All velocities are frozen before any donor copies, including revived
/// dead slots. A single orthogonal matrix acts on every relative velocity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollisionComponent {
    pub key: usize,
    pub members: Vec<usize>,
    pub rotation: Vec<f64>,
    pub center_of_mass: Vec<f64>,
    pub momentum_before: Vec<f64>,
    pub momentum_after: Vec<f64>,
    pub relative_energy_before: f64,
    pub relative_energy_after: f64,
}

/// Resolve connected components from accepted edges, never from unaccepted
/// candidates. Historical edges have no current physical donor velocity and are
/// explicitly unsupported for this collision map.
pub fn accepted_current_components<T: Real>(
    before: &Population<T>,
    pool: &DonorPool<T>,
    plan: &ClonePlan,
) -> Result<Vec<Vec<usize>>> {
    plan.validate(before, pool)?;
    let n = before.len();
    let mut parents: Vec<_> = (0..n).collect();
    let mut touched = vec![false; n];
    fn root(parents: &mut [usize], mut i: usize) -> usize {
        while parents[i] != i {
            parents[i] = parents[parents[i]];
            i = parents[i];
        }
        i
    }
    for (i, choice) in plan.choices.iter().enumerate() {
        if !choice.accepted {
            continue;
        }
        let source = pool.sources[choice.donors[0].pool_index as usize];
        if source.frame != pool.current_frame {
            return Err(GasError::Capability("component collision requires accepted current-frame donors; historical collision is undefined".into()));
        }
        let j = source.slot as usize;
        require(
            j < n && source.version == before.version && source.generation == before.generations[j],
            "component donor incarnation differs from frozen pre-clone population",
        )?;
        touched[i] = true;
        touched[j] = true;
        let a = root(&mut parents, i);
        let b = root(&mut parents, j);
        parents[a.max(b)] = a.min(b);
    }
    let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for (i, touched) in touched.iter().enumerate() {
        if *touched {
            groups.entry(root(&mut parents, i)).or_default().push(i);
        }
    }
    Ok(groups.into_values().collect())
}

/// Deterministic collision map for supplied component rotations. This separates
/// the physical map from the addressed Haar draws and permits transporting the
/// same random matrices when testing permutation equivariance.
pub fn apply_component_rotations<T: Real>(
    before: &Population<T>,
    after: &mut Population<T>,
    components: &[Vec<usize>],
    rotations: &[Vec<f64>],
    velocity_field: &str,
    restitution: f64,
) -> Result<Vec<CollisionComponent>> {
    require(
        restitution.is_finite() && (0. ..=1.).contains(&restitution),
        "component restitution outside [0,1]",
    )?;
    require(
        components.len() == rotations.len() && before.len() == after.len(),
        "component collision shapes",
    )?;
    let old = before.observations.field(velocity_field)?;
    let d = old.width();
    require(
        old.item_shape().len() == 1
            && d > 0
            && after.observations.field(velocity_field)?.item_shape() == old.item_shape(),
        "component velocity shape",
    )?;
    let mut covered = vec![false; before.len()];
    let mut reports = Vec::with_capacity(components.len());
    for (members, rotation) in components.iter().zip(rotations) {
        require(
            !members.is_empty()
                && rotation.len() == d * d
                && rotation.iter().all(|x| x.is_finite()),
            "component rotation shape/finiteness",
        )?;
        for a in 0..d {
            for b in 0..d {
                let product = (0..d)
                    .map(|j| rotation[j * d + a] * rotation[j * d + b])
                    .sum::<f64>();
                require(
                    (product - if a == b { 1. } else { 0. }).abs() < 1e-9,
                    "component rotation is not orthogonal",
                )?;
            }
        }
        let mut momentum = vec![0.; d];
        for &i in members {
            require(
                i < before.len() && !covered[i],
                "collision components overlap or reference absent slot",
            )?;
            covered[i] = true;
            for (a, value) in old.row(i)?.iter().enumerate() {
                require(
                    value.is_finite(),
                    "component collision requires finite frozen velocities, including revived slots",
                )?;
                momentum[a] += value.to_f64();
            }
        }
        let center: Vec<_> = momentum.iter().map(|x| x / members.len() as f64).collect();
        require(
            center.iter().all(|x| x.is_finite()),
            "component center of mass overflow",
        )?;
        let mut before_energy = 0.;
        let mut after_energy = 0.;
        let mut after_momentum = vec![0.; d];
        for &i in members {
            let relative: Vec<_> = old
                .row(i)?
                .iter()
                .zip(&center)
                .map(|(v, m)| v.to_f64() - m)
                .collect();
            before_energy += 0.5 * relative.iter().map(|x| x * x).sum::<f64>();
            let output: Vec<T> = (0..d)
                .map(|a| {
                    T::from_f64(
                        center[a]
                            + restitution
                                * (0..d)
                                    .map(|b| rotation[a * d + b] * relative[b])
                                    .sum::<f64>(),
                    )
                })
                .collect();
            require(
                output.iter().all(|x| x.is_finite()),
                "component collision output overflow",
            )?;
            for a in 0..d {
                after_momentum[a] += output[a].to_f64();
                after_energy += 0.5 * (output[a].to_f64() - center[a]).powi(2);
            }
            after
                .observations
                .field_mut(velocity_field)?
                .replace_row(i, &output)?;
        }
        require(
            before_energy.is_finite() && after_energy.is_finite(),
            "component relative energy overflow",
        )?;
        reports.push(CollisionComponent {
            key: *members.iter().min().unwrap(),
            members: members.clone(),
            rotation: rotation.clone(),
            center_of_mass: center,
            momentum_before: momentum,
            momentum_after: after_momentum,
            relative_energy_before: before_energy,
            relative_energy_after: after_energy,
        });
    }
    Ok(reports)
}

fn record_components<T: Real>(
    before: &Population<T>,
    after: &Population<T>,
    velocity_field: &str,
    components: &[CollisionComponent],
    cx: &mut ExecutionContext,
) -> Result<()> {
    if cx.recorded_fields.is_none() {
        return Ok(());
    }
    let n = before.len();
    let old = before.observations.field(velocity_field)?;
    let output = after.observations.field(velocity_field)?;
    let d = old.width();
    let mut available = vec![false; n];
    let mut ids = vec![0.; n];
    let mut sizes = vec![0.; n];
    let mut rotations = vec![0.; n * d * d];
    let mut centers = vec![0.; n * d];
    let mut momentum_before = vec![0.; n * d];
    let mut momentum_after = vec![0.; n * d];
    let mut energy_before = vec![0.; n];
    let mut energy_after = vec![0.; n];
    let mut input = vec![0.; n * d];
    let mut actual_output = vec![0.; n * d];
    for component in components {
        for &i in &component.members {
            available[i] = true;
            ids[i] = component.key as f64;
            sizes[i] = component.members.len() as f64;
            rotations[i * d * d..(i + 1) * d * d].copy_from_slice(&component.rotation);
            centers[i * d..(i + 1) * d].copy_from_slice(&component.center_of_mass);
            momentum_before[i * d..(i + 1) * d].copy_from_slice(&component.momentum_before);
            momentum_after[i * d..(i + 1) * d].copy_from_slice(&component.momentum_after);
            energy_before[i] = component.relative_energy_before;
            energy_after[i] = component.relative_energy_after;
            for a in 0..d {
                input[i * d + a] = old.values()[i * d + a].to_f64();
                actual_output[i * d + a] = output.values()[i * d + a].to_f64();
            }
        }
    }
    for (name, shape, values) in [
        ("collision_component_id", vec![1], ids),
        ("collision_component_size", vec![1], sizes),
        ("collision_rotation", vec![d, d], rotations),
        ("collision_center_of_mass", vec![d], centers),
        ("collision_input_velocity", vec![d], input),
        ("collision_output_velocity", vec![d], actual_output),
        ("collision_momentum_before", vec![d], momentum_before),
        ("collision_momentum_after", vec![d], momentum_after),
        ("collision_relative_energy_before", vec![1], energy_before),
        ("collision_relative_energy_after", vec![1], energy_after),
    ] {
        cx.record_field_with_coverage(
            "component_collision",
            name,
            before.version,
            &crate::TensorBatch::<f64>::new(n, shape, values)?,
            &available,
        );
    }
    Ok(())
}
