//! Atomic-law reference for the fixed-step rooted-component limit.
//!
//! Each input row carries mass 1/M. Measurement marks retain the full sampled
//! companion law before nonlinear fitness. Repeated atom types represent
//! different population labels, so atom self-mass is included in the integral.
//! Incoming Poisson processes count rare incoming LABELS in the N→∞ graph;
//! they are not event times and do not change the finite-step clone probability.
//! Capacity exhaustion returns an error; components are never truncated or
//! resampled conditional on their size.
use crate::{
    GasConfig, GasError, Population, Result,
    donor::{CompanionReducer, SamplingLaw},
    error::require,
    fitness::Standardizer,
    geometry::{AlgorithmicDistance, Distance, InteractionKernel, Kernel},
    noise::{FactorValues, InnovationLaw, NoiseGeometry},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MeanFieldLimits {
    pub max_atoms: usize,
    pub max_nodes: usize,
    pub max_proposals: usize,
}
impl Default for MeanFieldLimits {
    fn default() -> Self {
        Self {
            max_atoms: 512,
            max_nodes: 4096,
            max_proposals: 1_000_000,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasurementType {
    pub input_slot: usize,
    pub measurement_companion: Option<usize>,
    pub probability: f64,
    pub alive: bool,
    pub oriented_reward: f64,
    pub separation: f64,
    pub fitness: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightedNormalizer {
    pub mean: f64,
    pub variance: f64,
    pub scale: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootedVertex {
    pub type_index: usize,
    /// Component vertex index. Incoming children have this fixed to their parent.
    pub outgoing: Option<usize>,
    pub outgoing_forced: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RootedCollisionSample {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub root_type: usize,
    pub root_donor_type: Option<usize>,
    pub component: Vec<RootedVertex>,
    pub center_of_mass: Vec<f64>,
    pub rotation: Vec<f64>,
    pub free_outgoing_draws: usize,
    pub forced_incoming_children: usize,
    pub incoming_proposals: usize,
}
#[derive(Clone, Debug)]
pub struct AtomicMeanField {
    pub types: Vec<MeasurementType>,
    pub reward_normalizer: WeightedNormalizer,
    pub diversity_normalizer: WeightedNormalizer,
    pub alive_mass: f64,
    /// beta(u,t) <= C, using w<=1 and exact atomic clone normalizers.
    pub incoming_intensity_bound: f64,
    input: Population<f64>,
    slot_types: Vec<Vec<usize>>,
    measurement_probabilities: Vec<Vec<f64>>,
    clone_probabilities: Vec<Vec<f64>>,
    clone_weights: Vec<Vec<f64>>,
    clone_normalizers: Vec<f64>,
    epsilon: f64,
    saturation: f64,
    alpha: f64,
    jitter_scale: f64,
    positions_field: String,
    velocities_field: String,
    limits: MeanFieldLimits,
}
fn weighted_normalizer(
    types: &[MeasurementType],
    value: impl Fn(&MeasurementType) -> f64,
    alive_mass: f64,
    floor: f64,
) -> Result<WeightedNormalizer> {
    let mean = types
        .iter()
        .filter(|t| t.alive)
        .map(|t| t.probability * value(t))
        .sum::<f64>()
        / alive_mass;
    let variance = types
        .iter()
        .filter(|t| t.alive)
        .map(|t| t.probability * (value(t) - mean).powi(2))
        .sum::<f64>()
        / alive_mass;
    let scale = (variance + floor * floor).sqrt();
    require(
        mean.is_finite() && variance.is_finite() && scale.is_finite() && scale > 0.,
        "atomic fitness moments overflow",
    )?;
    Ok(WeightedNormalizer {
        mean,
        variance,
        scale,
    })
}
fn choose(weights: &[f64], rng: &mut RandomStream) -> usize {
    let draw = rng.uniform::<f64>();
    let mut cumulative = 0.;
    for (j, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if draw < cumulative {
            return j;
        }
    }
    weights
        .iter()
        .rposition(|&w| w > 0.)
        .expect("validated probability row")
}
impl AtomicMeanField {
    /// Rewards must already be evaluated for the entering law. No external
    /// reward or field providers are invoked or inferred by this constructor.
    pub fn new(
        input: &Population<f64>,
        config: &GasConfig,
        limits: MeanFieldLimits,
    ) -> Result<Self> {
        input.validate()?;
        require(
            input.len() <= limits.max_atoms
                && limits.max_atoms > 0
                && limits.max_nodes > 0
                && limits.max_proposals > 0,
            "atomic mean-field solver capacity configuration",
        )?;
        config.fitness.validate()?;
        config.clone_decision.validate()?;
        config.clone_transform.validate(input)?;
        require(
            config.clone_decision.revival_from_companion,
            "atomic component reference requires current weighted revival",
        )?;
        require(
            config.clone_decision.every == 1,
            "atomic component reference requires cloning on every step",
        )?;
        require(
            matches!(config.reducer, CompanionReducer::Mean),
            "atomic component reference requires one-companion mean measurement",
        )?;
        for module in [&config.distance_donors, &config.cloning_donors] {
            module.validate(&input.observations)?;
            require(
                module.law == SamplingLaw::Independent
                    && module.history_window == 0
                    && module.count == 1
                    && matches!(module.distance, Distance::SquashedPhaseSpace { .. })
                    && matches!(module.kernel, Kernel::Gaussian { .. }),
                "atomic component reference requires independent current squashed Gaussian donors",
            )?;
        }
        let (reward_floor, diversity_floor) = match (
            &config.fitness.reward_standardizer,
            &config.fitness.diversity_standardizer,
        ) {
            (Standardizer::Global { sigma_min: r }, Standardizer::Global { sigma_min: d }) => {
                (*r, *d)
            }
            _ => {
                return Err(GasError::Capability(
                    "atomic component reference requires global fitness normalizers".into(),
                ));
            }
        };
        let alpha = config.clone_transform.restitution.ok_or_else(|| {
            GasError::Capability("atomic component reference requires component restitution".into())
        })?;
        let positions_field = config
            .clone_transform
            .position_field
            .clone()
            .ok_or_else(|| GasError::MissingField("component position field".into()))?;
        let velocities_field = config.clone_transform.velocity_field.clone().unwrap();
        let positions = input.observations.field(&positions_field)?;
        let velocities = input.observations.field(&velocities_field)?;
        require(
            positions.item_shape() == velocities.item_shape()
                && positions
                    .values()
                    .iter()
                    .chain(velocities.values())
                    .all(|x| x.is_finite()),
            "atomic law requires finite matched phase coordinates, including retained dead rows",
        )?;
        let jitter_scale = match &config.clone_transform.jitter {
            None => 0.,
            Some(noise) => match (&noise.innovation, &noise.geometry) {
                (
                    InnovationLaw::Gaussian,
                    NoiseGeometry::Isotropic {
                        scale: FactorValues::Constant { values },
                    },
                ) if values.len() == 1 => values[0] * config.clone_transform.jitter_amplitude,
                _ => return Err(GasError::Capability(
                    "atomic component reference supports constant isotropic Gaussian clone jitter"
                        .into(),
                )),
            },
        };
        let n = input.len();
        let alive = input.eligible(config.include_truncated);
        require(
            alive
                .iter()
                .enumerate()
                .all(|(i, &a)| !a || (input.rewards.valid[i] && input.rewards.raw[i].is_finite())),
            "atomic alive reward is unavailable",
        )?;
        let alive_mass = alive.iter().filter(|&&a| a).count() as f64 / n as f64;
        if alive_mass == 0. {
            return Err(GasError::Extinction);
        }
        let mut measurement_probabilities = vec![vec![0.; n]; n];
        let mut clone_probabilities = measurement_probabilities.clone();
        let mut clone_weights = measurement_probabilities.clone();
        let mut separations = measurement_probabilities.clone();
        let mut clone_normalizers = vec![0.; n];
        for i in 0..n {
            for (j, &eligible) in alive.iter().enumerate() {
                if eligible {
                    for (module, weights) in [
                        (&config.distance_donors, &mut measurement_probabilities),
                        (&config.cloning_donors, &mut clone_weights),
                    ] {
                        let value = module.distance.compare(
                            &input.observations,
                            i,
                            &input.observations,
                            j,
                        )?;
                        let kind = <Distance as AlgorithmicDistance<f64>>::comparison_kind(
                            &module.distance,
                        );
                        weights[i][j] = module.kernel.log_weight(value, kind)?.exp();
                    }
                    let distance = config.distance_donors.distance.compare(
                        &input.observations,
                        i,
                        &input.observations,
                        j,
                    )?;
                    separations[i][j] =
                        (distance * distance + config.fitness.distance_floor.powi(2)).sqrt();
                }
            }
            let measurement_sum = measurement_probabilities[i].iter().sum::<f64>();
            let clone_sum = clone_weights[i].iter().sum::<f64>();
            require(
                measurement_sum.is_finite()
                    && measurement_sum > 0.
                    && clone_sum.is_finite()
                    && clone_sum > 0.,
                "atomic Gaussian donor normalizer underflow",
            )?;
            for j in 0..n {
                measurement_probabilities[i][j] /= measurement_sum;
                clone_probabilities[i][j] = clone_weights[i][j] / clone_sum;
            }
            clone_normalizers[i] = clone_sum / n as f64;
        }
        let mut types = Vec::new();
        let mut slot_types = vec![Vec::new(); n];
        for i in 0..n {
            for (j, &eligible) in alive.iter().enumerate() {
                if alive[i] && eligible {
                    slot_types[i].push(types.len());
                    types.push(MeasurementType {
                        input_slot: i,
                        measurement_companion: Some(j),
                        probability: measurement_probabilities[i][j] / n as f64,
                        alive: true,
                        oriented_reward: config.fitness.direction.orient(input.rewards.raw[i]),
                        separation: separations[i][j],
                        fitness: 0.,
                    });
                }
            }
            if !alive[i] {
                slot_types[i].push(types.len());
                types.push(MeasurementType {
                    input_slot: i,
                    measurement_companion: None,
                    probability: 1. / n as f64,
                    alive: false,
                    oriented_reward: 0.,
                    separation: 0.,
                    fitness: 0.,
                });
            }
        }
        let reward_normalizer =
            weighted_normalizer(&types, |t| t.oriented_reward, alive_mass, reward_floor)?;
        let diversity_normalizer =
            weighted_normalizer(&types, |t| t.separation, alive_mass, diversity_floor)?;
        let fitness = config.fitness.combine(
            &types
                .iter()
                .map(|t| (t.oriented_reward - reward_normalizer.mean) / reward_normalizer.scale)
                .collect::<Vec<_>>(),
            &types
                .iter()
                .map(|t| (t.separation - diversity_normalizer.mean) / diversity_normalizer.scale)
                .collect::<Vec<_>>(),
            &types.iter().map(|t| t.alive).collect::<Vec<_>>(),
        )?;
        for (t, f) in types.iter_mut().zip(fitness) {
            t.fitness = f;
        }
        let incoming_intensity_bound = clone_normalizers.iter().map(|z| 1. / z).fold(0., f64::max);
        require(
            incoming_intensity_bound.is_finite(),
            "atomic incoming intensity bound overflow",
        )?;
        Ok(Self {
            types,
            reward_normalizer,
            diversity_normalizer,
            alive_mass,
            incoming_intensity_bound,
            input: input.clone(),
            slot_types,
            measurement_probabilities,
            clone_probabilities,
            clone_weights,
            clone_normalizers,
            epsilon: config.clone_decision.epsilon,
            saturation: config.clone_decision.saturation,
            alpha,
            jitter_scale,
            positions_field,
            velocities_field,
            limits,
        })
    }
    fn sample_slot_type(&self, slot: usize, rng: &mut RandomStream) -> usize {
        if self.slot_types[slot].len() == 1 {
            return self.slot_types[slot][0];
        }
        let j = choose(&self.measurement_probabilities[slot], rng);
        self.slot_types[slot]
            .iter()
            .copied()
            .find(|&t| self.types[t].measurement_companion == Some(j))
            .unwrap()
    }
    fn sample_type(&self, rng: &mut RandomStream) -> usize {
        self.sample_slot_type(rng.index(self.input.len()), rng)
    }
    fn acceptance(&self, source: usize, target: usize) -> f64 {
        let s = &self.types[source];
        let t = &self.types[target];
        if !t.alive {
            0.
        } else if !s.alive {
            1.
        } else {
            ((t.fitness - s.fitness) / (s.fitness + self.epsilon) / self.saturation).clamp(0., 1.)
        }
    }
    /// Density of an outgoing accepted edge relative to the full marked law eta.
    pub fn edge_density(&self, source: usize, target: usize) -> Result<f64> {
        require(
            source < self.types.len() && target < self.types.len(),
            "atomic type index",
        )?;
        let s = self.types[source].input_slot;
        let t = self.types[target].input_slot;
        Ok(self.clone_weights[s][t] / self.clone_normalizers[s] * self.acceptance(source, target))
    }
    fn append(&self, nodes: &mut Vec<RootedVertex>, node: RootedVertex) -> Result<usize> {
        if nodes.len() >= self.limits.max_nodes {
            return Err(GasError::Execution("rooted mean-field component exceeds max_nodes; sample was not truncated or retried".into()));
        }
        let i = nodes.len();
        nodes.push(node);
        Ok(i)
    }
    pub fn sample_root(&self, seed: u64, draw_index: u64) -> Result<RootedCollisionSample> {
        let mut rng = RandomStream::new(seed, draw_index, Stream::MeanFieldReference, 0, 0);
        let root_type = self.sample_type(&mut rng);
        let mut nodes = vec![RootedVertex {
            type_index: root_type,
            outgoing: None,
            outgoing_forced: false,
        }];
        let mut free_outgoing_draws = 0;
        let mut forced_incoming_children = 0;
        let mut incoming_proposals = 0;
        let mut cursor = 0;
        while cursor < nodes.len() {
            let t = nodes[cursor].type_index;
            if !nodes[cursor].outgoing_forced {
                free_outgoing_draws += 1;
                let donor_slot = choose(
                    &self.clone_probabilities[self.types[t].input_slot],
                    &mut rng,
                );
                let donor = self.sample_slot_type(donor_slot, &mut rng);
                if rng.uniform::<f64>() < self.acceptance(t, donor) {
                    let j = self.append(
                        &mut nodes,
                        RootedVertex {
                            type_index: donor,
                            outgoing: None,
                            outgoing_forced: false,
                        },
                    )?;
                    nodes[cursor].outgoing = Some(j);
                }
            }
            if self.types[t].alive {
                // Unit-rate exponential spacings generate Poisson(C) proposal labels.
                let mut arrival = -rng.uniform::<f64>().ln();
                while arrival < self.incoming_intensity_bound {
                    incoming_proposals += 1;
                    if incoming_proposals > self.limits.max_proposals {
                        return Err(GasError::Execution("rooted mean-field exploration exceeds max_proposals; sample was not truncated or retried".into()));
                    }
                    let child = self.sample_type(&mut rng);
                    let probability = self.edge_density(child, t)? / self.incoming_intensity_bound;
                    if rng.uniform::<f64>() < probability {
                        self.append(
                            &mut nodes,
                            RootedVertex {
                                type_index: child,
                                outgoing: Some(cursor),
                                outgoing_forced: true,
                            },
                        )?;
                        forced_incoming_children += 1;
                    }
                    arrival -= rng.uniform::<f64>().ln();
                }
            }
            cursor += 1;
        }
        let old = self.input.observations.field(&self.velocities_field)?;
        let d = old.width();
        let mut center = vec![0.; d];
        for node in &nodes {
            for (a, &v) in old
                .row(self.types[node.type_index].input_slot)?
                .iter()
                .enumerate()
            {
                center[a] += v / nodes.len() as f64;
            }
        }
        let rotation = RandomStream::new(seed, draw_index, Stream::MeanFieldReference, 0, 1)
            .haar_orthogonal(d)?;
        let root_velocity = old.row(self.types[root_type].input_slot)?;
        let velocities = (0..d)
            .map(|a| {
                center[a]
                    + self.alpha
                        * (0..d)
                            .map(|b| rotation[a * d + b] * (root_velocity[b] - center[b]))
                            .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let root_donor_type = nodes[0].outgoing.map(|i| nodes[i].type_index);
        let position_type = root_donor_type.unwrap_or(root_type);
        let mut positions = self
            .input
            .observations
            .field(&self.positions_field)?
            .row(self.types[position_type].input_slot)?
            .to_vec();
        if root_donor_type.is_some() {
            let mut jitter = RandomStream::new(seed, draw_index, Stream::MeanFieldReference, 0, 2);
            for x in &mut positions {
                *x += self.jitter_scale * jitter.gaussian::<f64>();
            }
        }
        require(
            positions.iter().chain(&velocities).all(|x| x.is_finite()),
            "rooted component output overflow",
        )?;
        Ok(RootedCollisionSample {
            positions,
            velocities,
            root_type,
            root_donor_type,
            component: nodes,
            center_of_mass: center,
            rotation,
            free_outgoing_draws,
            forced_incoming_children,
            incoming_proposals,
        })
    }
}
