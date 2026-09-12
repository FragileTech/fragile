use crate::{
    ExecutionContext, GasError, ObservationBatch, Population, Provenance, Real, Result,
    RewardBatch, StateStore, TensorBatch,
    error::require,
    geometry::{AlgorithmicDistance, ComparisonKind, Distance, InteractionKernel, Kernel},
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub frame: u64,
    pub slot: u32,
    pub generation: u64,
    pub version: u64,
}

#[derive(Clone, Debug)]
pub struct DonorPool<T: Real> {
    pub population: Population<T>,
    pub sources: Vec<SourceRef>,
    pub current_frame: u64,
}
impl<T: Real> DonorPool<T> {
    /// Freeze eligible rows, never ancestry-filter them. Historical frames must
    /// use the same schema and snapshot codec. Stored fitness is never reused.
    pub fn freeze(
        current: &Population<T>,
        frame: u64,
        history: &[(u64, Population<T>)],
        window: usize,
        include_truncated: bool,
    ) -> Result<Self> {
        let mut frames = vec![(frame, current)];
        frames.extend(history.iter().rev().take(window).map(|(f, p)| (*f, p)));
        let mut fields: BTreeMap<String, Vec<T>> = current
            .observations
            .fields
            .keys()
            .map(|k| (k.clone(), vec![]))
            .collect();
        let mut sources = vec![];
        let mut raw = vec![];
        let mut valid = vec![];
        let mut validity = vec![];
        let mut states = vec![];
        let mut generations = vec![];
        for (f, p) in frames {
            p.validate()?;
            require(
                p.observations.fields.len() == fields.len(),
                "historical observation schema differs",
            )?;
            require(
                p.states.as_ref().map(|s| &s.codec) == current.states.as_ref().map(|s| &s.codec),
                "historical state codec differs",
            )?;
            for (name, values) in &mut fields {
                let source = p.observations.field(name)?;
                let target = current.observations.field(name)?;
                require(
                    source.item_shape() == target.item_shape(),
                    "historical field shape differs",
                )?;
                for (i, s) in p.validity.iter().enumerate() {
                    if s.eligible(include_truncated) && p.rewards.valid[i] {
                        values.extend_from_slice(source.row(i)?);
                    }
                }
            }
            for (i, s) in p.validity.iter().enumerate() {
                if !s.eligible(include_truncated) || !p.rewards.valid[i] {
                    continue;
                }
                sources.push(SourceRef {
                    frame: f,
                    slot: i as u32,
                    generation: p.generations[i],
                    version: p.version,
                });
                raw.push(p.rewards.raw[i]);
                valid.push(p.rewards.valid[i]);
                validity.push(*s);
                generations.push(p.generations[i]);
                if let Some(store) = &p.states {
                    states.push(store.snapshots[i].clone());
                }
            }
        }
        if sources.is_empty() {
            return Err(GasError::Extinction);
        }
        let n = sources.len();
        let obs = ObservationBatch {
            fields: fields
                .into_iter()
                .map(|(k, v)| {
                    let shape = current.observations.field(&k)?.item_shape().to_vec();
                    Ok((k, TensorBatch::new(n, shape, v)?))
                })
                .collect::<Result<_>>()?,
            provenance: Provenance {
                population_version: current.version,
                stage: "donor_pool".into(),
                ..Default::default()
            },
        };
        Ok(Self {
            population: Population {
                observations: obs,
                rewards: RewardBatch {
                    raw,
                    valid,
                    provenance: Provenance::default(),
                },
                validity,
                states: current.states.as_ref().map(|s| StateStore {
                    snapshots: states,
                    codec: s.codec.clone(),
                }),
                generations,
                version: current.version,
            },
            sources,
            current_frame: frame,
        })
    }
    pub fn current_index(&self, slot: usize) -> Option<u32> {
        self.sources
            .iter()
            .position(|s| s.frame == self.current_frame && s.slot as usize == slot)
            .map(|i| i as u32)
    }
    /// Build once per sampling batch. Repeated linear current_index searches
    /// would turn an otherwise linear shuffle into a quadratic algorithm.
    fn current_lookup(&self, rows: usize) -> Result<Vec<Option<u32>>> {
        self.population.validate()?;
        require(
            self.population.len() == self.sources.len(),
            "donor pool shape",
        )?;
        let mut lookup = vec![None; rows];
        for (index, source) in self.sources.iter().enumerate() {
            if source.frame == self.current_frame {
                let entry = lookup
                    .get_mut(source.slot as usize)
                    .ok_or_else(|| GasError::Shape("current donor slot".into()))?;
                require(entry.is_none(), "duplicate current donor slot")?;
                *entry = Some(index as u32);
            }
        }
        Ok(lookup)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingLaw {
    Independent,
    FisherYates,
    GaussianGreedy,
    LegacyPermutation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OddPolicy {
    SelfCompanion,
    Unmatched,
    Reject,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsufficientPolicy {
    #[default]
    UseAvailable,
    Reject,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DonorModule {
    pub distance: Distance,
    pub kernel: Kernel,
    pub law: SamplingLaw,
    pub count: usize,
    pub allow_self: bool,
    pub replacement: bool,
    #[serde(default)]
    pub insufficient: InsufficientPolicy,
    pub odd: OddPolicy,
    pub history_window: usize,
    pub tile_edges: usize,
}
impl Default for DonorModule {
    fn default() -> Self {
        Self {
            distance: Distance::default(),
            kernel: Kernel::Gaussian { width: 1. },
            law: SamplingLaw::Independent,
            count: 1,
            allow_self: false,
            replacement: true,
            insufficient: InsufficientPolicy::UseAvailable,
            odd: OddPolicy::SelfCompanion,
            history_window: 0,
            tile_edges: 4096,
        }
    }
}

/// Dense `[N,K]` integer pool indices plus a mask. Masked entries have no donor;
/// index zero is not a sentinel. Mutual pairs are identified separately.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompanionBatch {
    pub rows: usize,
    pub count: usize,
    pub indices: Vec<u32>,
    pub valid: Vec<bool>,
    pub mutual: bool,
}
impl CompanionBatch {
    pub fn validate<T: Real>(&self, pool: &DonorPool<T>) -> Result<()> {
        self.validate_indices(pool.sources.len())
    }
    pub(crate) fn validate_indices(&self, source_count: usize) -> Result<()> {
        if self.rows == 0
            || self.count == 0
            || self.rows.checked_mul(self.count) != Some(self.indices.len())
            || self.valid.len() != self.indices.len()
        {
            return Err(GasError::Shape("companion [N,K] shape".into()));
        }
        if self
            .indices
            .iter()
            .zip(&self.valid)
            .any(|(&i, &v)| v && i as usize >= source_count)
        {
            return Err(GasError::Shape("companion pool index".into()));
        }
        Ok(())
    }
    pub fn row(&self, i: usize) -> impl Iterator<Item = u32> + '_ {
        let range = i
            .checked_mul(self.count)
            .and_then(|start| start.checked_add(self.count).map(|end| start..end));
        let indices = range
            .clone()
            .and_then(|r| self.indices.get(r))
            .unwrap_or(&[]);
        let valid = range.and_then(|r| self.valid.get(r)).unwrap_or(&[]);
        indices
            .iter()
            .zip(valid)
            .filter_map(|(&j, &v)| v.then_some(j))
    }
}
pub struct CompanionRequest<'a, T: Real> {
    pub population: &'a Population<T>,
    pub pool: &'a DonorPool<T>,
    pub eligible: &'a [bool],
    pub seed: u64,
    pub step: u64,
    pub stream: Stream,
}
#[allow(async_fn_in_trait)]
pub trait CompanionSampler<T: Real> {
    async fn sample(
        &self,
        request: CompanionRequest<'_, T>,
        cx: &mut ExecutionContext,
    ) -> Result<CompanionBatch>;
}

impl DonorModule {
    pub fn validate<T: Real>(&self, obs: &ObservationBatch<T>) -> Result<()> {
        require(
            self.count > 0 && self.count <= 1024,
            "companion count must be 1..=1024",
        )?;
        require(
            self.tile_edges > 0 && self.tile_edges <= 1_048_576,
            "invalid pair tile size",
        )?;
        self.distance.validate(obs)?;
        self.kernel
            .validate(<Distance as AlgorithmicDistance<T>>::comparison_kind(
                &self.distance,
            ))?;
        if self.law != SamplingLaw::Independent {
            require(
                self.history_window == 0,
                "mutual and legacy laws require the current population",
            )?;
        }
        if self.law == SamplingLaw::FisherYates || self.law == SamplingLaw::LegacyPermutation {
            require(
                matches!(self.kernel, Kernel::Uniform),
                "uniform matching/permutation requires the uniform kernel",
            )?;
        }
        if self.law != SamplingLaw::Independent && self.count > 1 && !self.replacement {
            return Err(GasError::Topology(
                "repeated matching rounds allow repeated companions; select replacement explicitly"
                    .into(),
            ));
        }
        Ok(())
    }
}
impl<T: Real> CompanionSampler<T> for DonorModule {
    async fn sample(
        &self,
        r: CompanionRequest<'_, T>,
        cx: &mut ExecutionContext,
    ) -> Result<CompanionBatch> {
        self.validate(&r.population.observations)?;
        let n = r.population.len();
        let k = self.count;
        let m = r.pool.sources.len();
        require(r.eligible.len() == n, "eligibility shape")?;
        require(
            n.checked_mul(k)
                .is_some_and(|size| size <= cx.max_batch_elements),
            "companion batch memory limit",
        )?;
        let current = r.pool.current_lookup(n)?;
        let current_index = |slot: usize| {
            current
                .get(slot)
                .copied()
                .flatten()
                .ok_or(GasError::Extinction)
        };
        let mut out = CompanionBatch {
            rows: n,
            count: k,
            indices: vec![0; n * k],
            valid: vec![false; n * k],
            mutual: matches!(
                self.law,
                SamplingLaw::FisherYates | SamplingLaw::GaussianGreedy
            ),
        };
        let candidates = |i: usize| {
            (0..m)
                .filter(|&j| {
                    self.allow_self
                        || r.pool.sources[j].frame != r.pool.current_frame
                        || r.pool.sources[j].slot as usize != i
                })
                .map(|j| j as u32)
                .collect::<Vec<_>>()
        };
        if self.law == SamplingLaw::Independent && matches!(self.kernel, Kernel::Uniform) {
            for (i, &own_index) in current.iter().enumerate() {
                if !r.eligible[i] {
                    continue;
                }
                let excluded = if self.allow_self {
                    None
                } else {
                    own_index.map(|j| j as usize)
                };
                let available = m - usize::from(excluded.is_some());
                let singleton = available == 0;
                let available = available.max(1);
                if singleton {
                    current_index(i)?;
                }
                require(
                    self.replacement
                        || self.insufficient == InsufficientPolicy::UseAvailable
                        || k <= available,
                    "not enough distinct eligible companions",
                )?;
                let draws = if self.replacement {
                    k
                } else {
                    k.min(available)
                };
                // Sparse partial Fisher-Yates: O(K log K) without replacement,
                // O(K) with replacement, without an O(M) list per walker.
                let mut swaps = BTreeMap::new();
                let mut rng = RandomStream::new(r.seed, r.step, r.stream, i as u64, 0);
                for a in 0..draws {
                    let remaining = if self.replacement {
                        available
                    } else {
                        available - a
                    };
                    let j = rng.index(remaining);
                    let rank = swaps.get(&j).copied().unwrap_or(j);
                    let donor = if singleton {
                        current_index(i)?
                    } else {
                        (rank + usize::from(excluded.is_some_and(|s| rank >= s))) as u32
                    };
                    out.indices[i * k + a] = donor;
                    out.valid[i * k + a] = true;
                    if !self.replacement {
                        let last = remaining - 1;
                        let replacement = swaps.remove(&last).unwrap_or(last);
                        if j != last {
                            swaps.insert(j, replacement);
                        }
                    }
                }
            }
        } else if self.law == SamplingLaw::Independent {
            // Streaming Gumbel-max for independent categorical draws; one
            // ranking per row gives weighted sampling without replacement.
            let mut best = vec![T::from_f64(f64::NEG_INFINITY); n * k];
            let mut expected = vec![0; n];
            let mut edges = Vec::with_capacity(self.tile_edges);
            for (i, expected_count) in expected.iter_mut().enumerate() {
                if !r.eligible[i] {
                    continue;
                }
                let mut c = candidates(i);
                if c.is_empty() {
                    c.push(current_index(i)?);
                }
                require(
                    self.replacement
                        || self.insufficient == InsufficientPolicy::UseAvailable
                        || k <= c.len(),
                    "not enough distinct eligible companions",
                )?;
                *expected_count = if self.replacement { k } else { k.min(c.len()) };
                for j in c {
                    edges.push((i as u32, j));
                    if edges.len() == self.tile_edges {
                        self.score_tile(&r, &edges, &mut best, &mut out, cx).await?;
                        edges.clear();
                    }
                }
            }
            if !edges.is_empty() {
                self.score_tile(&r, &edges, &mut best, &mut out, cx).await?;
            }
            for (i, &alive) in r.eligible.iter().enumerate() {
                if alive && out.row(i).count() != expected[i] {
                    return Err(GasError::Numerical(
                        "all kernel log-weights underflowed; increase width or rescale distance"
                            .into(),
                    ));
                }
            }
        } else {
            let alive: Vec<usize> = r
                .eligible
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| v.then_some(i))
                .collect();
            if alive.is_empty() {
                return Err(GasError::Extinction);
            }
            require(
                r.pool
                    .sources
                    .iter()
                    .all(|s| s.frame == r.pool.current_frame),
                "matching requires current sources",
            )?;
            for round in 0..k {
                let mut rng = RandomStream::new(r.seed, r.step, r.stream, 0, round as u64);
                let mut order = alive.clone();
                rng.shuffle(&mut order);
                if self.law == SamplingLaw::LegacyPermutation {
                    // Exact legacy distinction: permutation with no deaths;
                    // independent replacement draws if any slot is dead.
                    for &i in &alive {
                        let j = if alive.len() == n {
                            order[i]
                        } else {
                            alive[rng.index(alive.len())]
                        };
                        out.indices[i * k + round] = current_index(j)?;
                        out.valid[i * k + round] = true;
                    }
                    continue;
                }
                while order.len() > 1 {
                    let i = order.pop().unwrap();
                    let choice = if self.law == SamplingLaw::FisherYates {
                        order.len() - 1
                    } else {
                        let edges: Vec<_> = order
                            .iter()
                            .map(|&j| Ok((i as u32, current_index(j)?)))
                            .collect::<Result<_>>()?;
                        let values = self
                            .distance
                            .pairs(
                                &r.population.observations,
                                &r.pool.population.observations,
                                &edges,
                                cx,
                            )
                            .await?;
                        let mut best = T::from_f64(f64::NEG_INFINITY);
                        let mut choice = None;
                        for (a, v) in values.into_iter().enumerate() {
                            let w = self.kernel.log_weight(
                                v,
                                <Distance as AlgorithmicDistance<T>>::comparison_kind(
                                    &self.distance,
                                ),
                            )?;
                            let score = w - (-rng.uniform::<T>().ln()).ln();
                            if score > best {
                                best = score;
                                choice = Some(a);
                            }
                        }
                        choice.ok_or_else(|| {
                            GasError::Numerical("greedy kernel has no finite weight".into())
                        })?
                    };
                    let j = order.swap_remove(choice);
                    out.indices[i * k + round] = current_index(j)?;
                    out.indices[j * k + round] = current_index(i)?;
                    out.valid[i * k + round] = true;
                    out.valid[j * k + round] = true;
                }
                if let Some(i) = order.pop() {
                    match self.odd {
                        OddPolicy::Reject if alive.len() > 1 => {
                            return Err(GasError::Topology("odd eligible population".into()));
                        }
                        OddPolicy::Unmatched if alive.len() > 1 => {}
                        _ => {
                            out.indices[i * k + round] = current_index(i)?;
                            out.valid[i * k + round] = true;
                        }
                    }
                }
            }
        }
        out.validate(r.pool)?;
        Ok(out)
    }
}
impl DonorModule {
    async fn score_tile<T: Real>(
        &self,
        r: &CompanionRequest<'_, T>,
        edges: &[(u32, u32)],
        best: &mut [T],
        out: &mut CompanionBatch,
        cx: &mut ExecutionContext,
    ) -> Result<()> {
        let values = self
            .distance
            .pairs(
                &r.population.observations,
                &r.pool.population.observations,
                edges,
                cx,
            )
            .await?;
        let k = self.count;
        for (&(i, j), v) in edges.iter().zip(values) {
            let w = self.kernel.log_weight(
                v,
                <Distance as AlgorithmicDistance<T>>::comparison_kind(&self.distance),
            )?;
            let i = i as usize;
            for round in 0..if self.replacement { k } else { 1 } {
                let mut rng = RandomStream::new(
                    r.seed,
                    r.step,
                    r.stream,
                    i as u64,
                    (j as u64) * k as u64 + round as u64 + 1,
                );
                let score = w - (-rng.uniform::<T>().ln()).ln();
                let slot = if self.replacement {
                    i * k + round
                } else {
                    (i * k..(i + 1) * k)
                        .min_by(|&a, &b| best[a].partial_cmp(&best[b]).unwrap())
                        .unwrap()
                };
                if score > best[slot] {
                    best[slot] = score;
                    out.indices[slot] = j;
                    out.valid[slot] = true;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompanionReducer {
    Mean,
    WeightedMean { weights: Vec<f64> },
    Minimum,
    Maximum,
}
impl CompanionReducer {
    pub async fn measure<T: Real>(
        &self,
        distance: &Distance,
        population: &Population<T>,
        pool: &DonorPool<T>,
        companions: &CompanionBatch,
        cx: &mut ExecutionContext,
    ) -> Result<Vec<T>> {
        companions.validate(pool)?;
        require(
            companions.rows == population.len(),
            "companion recipient count",
        )?;
        if let Self::WeightedMean { weights } = self {
            require(
                weights.len() == companions.count
                    && weights.iter().all(|w| w.is_finite() && *w >= 0.),
                "one nonnegative weight per companion required",
            )?;
        }
        let mut edges = Vec::new();
        let mut slots = Vec::new();
        for i in 0..companions.rows {
            for a in 0..companions.count {
                let slot = i * companions.count + a;
                if companions.valid[slot] {
                    edges.push((i as u32, companions.indices[slot]));
                    slots.push((i, a));
                }
            }
        }
        let mut sums = vec![T::ZERO; population.len()];
        let mut denom = vec![T::ZERO; population.len()];
        for (edge_tile, slot_tile) in edges.chunks(4096).zip(slots.chunks(4096)) {
            let values = distance
                .pairs(
                    &population.observations,
                    &pool.population.observations,
                    edge_tile,
                    cx,
                )
                .await?;
            for ((i, a), value) in slot_tile.iter().copied().zip(values) {
                // All these reducers operate on distance, not squared
                // distance. Cosine retains its dimensionless dissimilarity.
                let value = if <Distance as AlgorithmicDistance<T>>::comparison_kind(distance)
                    == ComparisonKind::SquaredDistance
                {
                    value.sqrt()
                } else {
                    value
                };
                match self {
                    Self::Mean => {
                        sums[i] = sums[i] + value;
                        denom[i] = denom[i] + T::ONE;
                    }
                    Self::WeightedMean { weights } => {
                        let w = T::from_f64(weights[a]);
                        sums[i] = sums[i] + w * value;
                        denom[i] = denom[i] + w;
                    }
                    Self::Minimum => {
                        sums[i] = if denom[i] == T::ZERO {
                            value
                        } else {
                            sums[i].min(value)
                        };
                        denom[i] = T::ONE;
                    }
                    Self::Maximum => {
                        sums[i] = sums[i].max(value);
                        denom[i] = T::ONE;
                    }
                }
            }
        }
        for i in 0..sums.len() {
            if denom[i] > T::ZERO {
                sums[i] = sums[i] / denom[i];
            } else if population.validity[i].eligible(false) && companions.row(i).next().is_some() {
                return Err(GasError::Numerical("zero total companion weight".into()));
            }
        }
        Ok(sums)
    }
}
