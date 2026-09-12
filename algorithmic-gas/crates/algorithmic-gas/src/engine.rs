use crate::operators::{
    BuiltinOperators, CloneRequest, GasOperators, MeasurementRequest, TransformRequest,
};
use crate::{
    BackendKind, ExecutionContext, GasError, InputBatch, Population, Precision, Real, Result,
    RewardBatch,
    boundary::BoundaryPolicy,
    cloning::{CloneDecision, ClonePlan, CloneTransform},
    compute::ExecutionStats,
    domain::{DomainAdapter, GradientProvider, NumericalDomain, RewardSource},
    donor::{
        CompanionBatch, CompanionReducer, CompanionRequest, DonorModule, DonorPool, SamplingLaw,
        SourceRef,
    },
    error::require,
    fitness::{FitnessBatch, FitnessPipeline, Standardizer},
    kinetic::{KineticContext, KineticOperator},
    random::{RNG_VERSION, Stream},
};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GasConfig {
    pub backend: BackendKind,
    pub precision: Precision,
    pub seed: u64,
    pub boundary: BoundaryPolicy,
    pub distance_donors: DonorModule,
    pub cloning_donors: DonorModule,
    pub reducer: CompanionReducer,
    pub fitness: FitnessPipeline,
    pub clone_decision: CloneDecision,
    pub clone_transform: CloneTransform,
    pub kinetic: KineticOperator,
    pub include_truncated: bool,
    pub invalid_reward: InvalidRewardPolicy,
    pub max_batch_elements: usize,
    pub max_memory_bytes: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidRewardPolicy {
    #[default]
    Error,
    Exclude,
}
impl Default for GasConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::Cpu,
            precision: Precision::F32,
            seed: 7,
            boundary: BoundaryPolicy::default(),
            distance_donors: DonorModule::default(),
            cloning_donors: DonorModule::default(),
            reducer: CompanionReducer::Mean,
            fitness: FitnessPipeline::default(),
            clone_decision: CloneDecision::default(),
            clone_transform: CloneTransform::default(),
            kinetic: KineticOperator::default(),
            include_truncated: false,
            invalid_reward: InvalidRewardPolicy::Error,
            max_batch_elements: 16_777_216,
            max_memory_bytes: crate::memory::DEFAULT_MEMORY_BYTES,
        }
    }
}
impl GasConfig {
    pub fn validate<T: Real>(&self, p: &Population<T>, has_gradient: bool) -> Result<()> {
        validate_float_parameters::<T>(
            &serde_json::to_value(self).map_err(|e| GasError::Configuration(e.to_string()))?,
            "gas",
        )?;
        p.validate()?;
        require(self.max_memory_bytes > 0, "memory budget must be positive")?;
        self.working_set_bytes(p, &[], None)?;
        require(
            T::PRECISION == self.precision,
            "population dtype differs from configuration",
        )?;
        require(
            self.max_batch_elements > 0,
            "batch memory limit must be positive",
        )?;
        self.distance_donors.validate(&p.observations)?;
        self.cloning_donors.validate(&p.observations)?;
        require(
            self.cloning_donors.count == 1,
            "multi-donor cloning is not implemented; cloning count must be one",
        )?;
        self.fitness.validate()?;
        require(
            self.cloning_donors.history_window == 0
                || (!matches!(self.fitness.reward_standardizer, Standardizer::Local { .. })
                    && !matches!(
                        self.fitness.diversity_standardizer,
                        Standardizer::Local { .. }
                    )),
            "historical cloning currently requires global standardization",
        )?;
        for field in p.observations.fields.values() {
            require(
                field.values().len() <= self.max_batch_elements,
                "observation exceeds batch memory limit",
            )?;
        }
        for module in [&self.distance_donors, &self.cloning_donors] {
            require(
                p.len()
                    .checked_mul(module.count)
                    .is_some_and(|n| n <= self.max_batch_elements),
                "companion batch exceeds memory limit",
            )?;
        }
        self.clone_decision.validate()?;
        self.clone_transform.validate(
            p,
            matches!(
                self.cloning_donors.law,
                SamplingLaw::FisherYates | SamplingLaw::GaussianGreedy
            ),
            1,
        )?;
        self.kinetic.validate(p, has_gradient)?;
        for module in [&self.distance_donors, &self.cloning_donors] {
            let boundary = self.boundary.periodic_domain(module.distance.field());
            require(
                boundary == module.distance.periodic(),
                "periodic distance and boundary must share the same domain",
            )?;
        }
        Ok(())
    }
}
fn validate_float_parameters<T: Real>(value: &serde_json::Value, path: &str) -> Result<()> {
    match value {
        serde_json::Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            let cast = T::from_f64(x);
            require(
                cast.is_finite() && (x == 0. || cast != T::ZERO),
                format!("{path} is not representable in {:?}", T::PRECISION),
            )?;
        }
        serde_json::Value::Object(fields) => {
            for (key, v) in fields {
                validate_float_parameters::<T>(v, &format!("{path}.{key}"))?;
            }
        }
        serde_json::Value::Array(values) => {
            for (i, v) in values.iter().enumerate() {
                validate_float_parameters::<T>(v, &format!("{path}[{i}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    pub fn reset(&self) {
        self.0.store(false, Ordering::Relaxed);
    }
    fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(GasError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct StepReport<T: Real> {
    pub step: u64,
    pub source_version: u64,
    pub result_version: u64,
    pub pre_clone_rewards: RewardBatch<T>,
    pub pre_clone_eligible: Vec<bool>,
    pub pre_clone_fitness: FitnessBatch<T>,
    pub distance_companions: CompanionBatch,
    pub distance_sources: Vec<SourceRef>,
    pub cloning_companions: CompanionBatch,
    pub clone_plan: ClonePlan,
    pub final_rewards: RewardBatch<T>,
    pub eligible: usize,
    pub clones: usize,
    pub revivals: usize,
    pub reward_evaluations: u64,
    pub execution: ExecutionStats,
}
pub trait HistorySink<T: Real> {
    fn record(&mut self, population: &Population<T>, report: &StepReport<T>) -> Result<()>;
}

/// Checkpoints contain immutable numerical/domain state and the history needed
/// for donor windows. Custom providers are reconstructed by the application;
/// their stable identifiers must match before restore is accepted.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Real", deny_unknown_fields)]
pub struct Checkpoint<T: Real> {
    pub schema_version: u32,
    pub rng_version: u32,
    pub config: GasConfig,
    pub population: Population<T>,
    pub step: u64,
    pub reward_evaluations: u64,
    pub execution: ExecutionStats,
    pub history: Vec<(u64, Population<T>)>,
    pub last_report: Option<StepReport<T>>,
    #[serde(default)]
    pub recording: Option<crate::tracking::RunArchive<T>>,
    pub reward_provider: String,
    pub gradient_provider: Option<String>,
    pub domain_provider: String,
    pub operator_set: String,
}
impl<T: Real> Checkpoint<T> {
    /// CBOR preserves non-finite invalid observations, unlike JSON. Configuration
    /// and ordinary result exports remain human-readable JSON.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = vec![];
        ciborium::ser::into_writer(self, &mut bytes)
            .map_err(|e| GasError::Checkpoint(e.to_string()))?;
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(GasError::Checkpoint(
                "checkpoint exceeds 256 MiB decode limit".into(),
            ));
        }
        let mut remaining = bytes;
        let mut checkpoint: Self =
            ciborium::de::from_reader_with_recursion_limit(&mut remaining, 64)
                .map_err(|e| GasError::Checkpoint(e.to_string()))?;
        require(remaining.is_empty(), "trailing checkpoint data")?;
        if checkpoint.schema_version == 2 {
            checkpoint.schema_version = crate::checkpoint::CHECKPOINT_VERSION;
            checkpoint.recording = Some(crate::tracking::RunArchive::new(
                Default::default(),
                checkpoint.config.clone(),
                checkpoint.step,
                &checkpoint.population,
                "checkpoint_v2_migration",
            )?);
            if let Some(archive) = &mut checkpoint.recording {
                archive
                    .providers
                    .insert("reward".into(), checkpoint.reward_provider.clone());
                archive
                    .providers
                    .insert("domain".into(), checkpoint.domain_provider.clone());
                archive
                    .providers
                    .insert("operators".into(), checkpoint.operator_set.clone());
                if let Some(gradient) = &checkpoint.gradient_provider {
                    archive
                        .providers
                        .insert("gradient".into(), gradient.clone());
                }
            }
        }
        checkpoint.validate()?;
        Ok(checkpoint)
    }
}

pub struct GasBuilder<T: Real> {
    config: GasConfig,
    population: Population<T>,
    reward: Arc<dyn RewardSource<T>>,
    gradient: Option<Arc<dyn GradientProvider<T>>>,
    domain: Arc<dyn DomainAdapter<T>>,
    operators: Arc<dyn GasOperators<T>>,
}
impl<T: Real> GasBuilder<T> {
    pub fn new(population: Population<T>, reward: impl RewardSource<T> + 'static) -> Self {
        Self {
            config: GasConfig {
                precision: T::PRECISION,
                ..Default::default()
            },
            population,
            reward: Arc::new(reward),
            gradient: None,
            domain: Arc::new(NumericalDomain),
            operators: Arc::new(BuiltinOperators),
        }
    }
    pub fn operators(mut self, operators: impl GasOperators<T> + 'static) -> Self {
        self.operators = Arc::new(operators);
        self
    }
    pub fn config(mut self, config: GasConfig) -> Self {
        self.config = config;
        self
    }
    pub fn gradient(mut self, gradient: impl GradientProvider<T> + 'static) -> Self {
        self.gradient = Some(Arc::new(gradient));
        self
    }
    pub fn domain(mut self, domain: impl DomainAdapter<T> + 'static) -> Self {
        self.domain = Arc::new(domain);
        self
    }
    pub async fn build(self) -> Result<AlgorithmicGas<T>> {
        self.config
            .validate(&self.population, self.gradient.is_some())?;
        self.operators.validate(&self.config, &self.population)?;
        let mut cx = ExecutionContext::new(self.config.backend, self.config.precision).await?;
        cx.max_batch_elements = self.config.max_batch_elements;
        cx.max_memory_bytes = self.config.max_memory_bytes
            - self.config.working_set_bytes(&self.population, &[], None)?;
        let mut gas = AlgorithmicGas {
            config: self.config,
            population: self.population,
            reward: self.reward,
            gradient: self.gradient,
            domain: self.domain,
            operators: self.operators,
            cx,
            step: 0,
            reward_evaluations: 0,
            history: vec![],
            last_report: None,
            recording: None,
            cancellation: CancellationToken::default(),
        };
        gas.domain.refresh_observations(&mut gas.population)?;
        gas.operators.boundary(
            &gas.config.boundary,
            &mut gas.population,
            gas.domain.as_ref(),
        )?;
        gas.population.rewards = gas
            .reward
            .evaluate(&gas.population, None, "initial", &mut gas.cx)
            .await?;
        gas.reward_evaluations += gas.population.len() as u64;
        validate_rewards(&mut gas.population, &gas.config)?;
        gas.population.observations.provenance.population_version = gas.population.version;
        gas.population.observations.provenance.stage = "initial".into();
        if !gas
            .population
            .eligible(gas.config.include_truncated)
            .iter()
            .any(|&x| x)
        {
            return Err(GasError::Extinction);
        }
        Ok(gas)
    }
}
pub struct AlgorithmicGas<T: Real> {
    config: GasConfig,
    population: Population<T>,
    reward: Arc<dyn RewardSource<T>>,
    gradient: Option<Arc<dyn GradientProvider<T>>>,
    domain: Arc<dyn DomainAdapter<T>>,
    operators: Arc<dyn GasOperators<T>>,
    cx: ExecutionContext,
    step: u64,
    reward_evaluations: u64,
    history: Vec<(u64, Population<T>)>,
    last_report: Option<StepReport<T>>,
    recording: Option<crate::tracking::RunArchive<T>>,
    cancellation: CancellationToken,
}
fn validate_rewards<T: Real>(p: &mut Population<T>, config: &GasConfig) -> Result<()> {
    p.rewards.validate(p.len())?;
    require(
        p.rewards.provenance.population_version == p.version,
        "stale reward population version",
    )?;
    for i in 0..p.len() {
        let invalid = !p.rewards.valid[i] || !p.rewards.raw[i].is_finite();
        if invalid
            && p.validity[i].eligible(config.include_truncated)
            && config.invalid_reward == InvalidRewardPolicy::Error
        {
            return Err(GasError::Numerical(format!(
                "reward evaluation failed for eligible walker {i}"
            )));
        }
        p.validity[i].invalid |= invalid;
        p.rewards.valid[i] = !invalid;
    }
    p.validate()
}
impl<T: Real> AlgorithmicGas<T> {
    fn execution_allowance(
        &self,
        p: &Population<T>,
        input: Option<&InputBatch<T>>,
    ) -> Result<usize> {
        use crate::memory::{checked_add, checked_mul, enforce};
        let mut reserved = self.config.working_set_bytes(p, &self.history, input)?;
        if let Some(archive) = &self.recording {
            // Retained archive plus typed per-stage copies and a transactional candidate.
            reserved = checked_add(reserved, archive.buffer_bytes()?)?;
            reserved = checked_add(reserved, checked_mul(p.buffer_bytes()?, 24)?)?;
        }
        enforce(reserved, self.config.max_memory_bytes)?;
        Ok(self.config.max_memory_bytes - reserved)
    }
    /// Begin a new archive at the current state. Export the old archive before restarting.
    pub fn start_recording(&mut self, config: crate::tracking::RecordingConfig) -> Result<()> {
        let mut archive = crate::tracking::RunArchive::new(
            config,
            self.config.clone(),
            self.step,
            &self.population,
            if self.step == 0 {
                "initial"
            } else {
                "recording_started"
            },
        )?;
        archive.providers.insert("reward".into(), self.reward.id());
        archive.providers.insert("domain".into(), self.domain.id());
        archive
            .providers
            .insert("operators".into(), self.operators.id());
        if let Some(gradient) = &self.gradient {
            archive.providers.insert("gradient".into(), gradient.id());
        }
        crate::memory::enforce(
            crate::memory::checked_add(
                archive.buffer_bytes()?,
                self.config
                    .working_set_bytes(&self.population, &self.history, None)?,
            )?,
            self.config.max_memory_bytes,
        )?;
        self.recording = Some(archive);
        self.cx.recorded_stages = Some(Vec::new());
        self.cx.recorded_noise = Some(Vec::new());
        self.cx.recorded_fields = Some(Vec::new());
        self.cx.recorded_influences = Some(Vec::new());
        Ok(())
    }
    pub fn recording(&self) -> Option<&crate::tracking::RunArchive<T>> {
        self.recording.as_ref()
    }
    pub fn stop_recording(&mut self) -> Option<crate::tracking::RunArchive<T>> {
        self.cx.recorded_stages = None;
        self.cx.recorded_noise = None;
        self.cx.recorded_fields = None;
        self.cx.recorded_influences = None;
        self.recording.take()
    }
    /// Enable a last-step replay for small teaching populations. Disabled by default.
    pub fn set_trace(&mut self, enabled: bool) -> Result<()> {
        require(
            !enabled
                || self
                    .population
                    .observations
                    .fields
                    .values()
                    .map(|f| f.values().len())
                    .sum::<usize>()
                    <= 8192,
            "stage tracing is limited to 8192 observation scalars",
        )?;
        self.cx.stage_trace = enabled.then(Vec::new);
        Ok(())
    }
    pub fn stage_trace(&self) -> Option<&[serde_json::Value]> {
        self.cx.stage_trace.as_deref()
    }
    pub fn config(&self) -> &GasConfig {
        &self.config
    }
    pub fn population(&self) -> &Population<T> {
        &self.population
    }
    pub fn step_number(&self) -> u64 {
        self.step
    }
    pub fn reward_evaluations(&self) -> u64 {
        self.reward_evaluations
    }
    pub fn last_report(&self) -> Option<&StepReport<T>> {
        self.last_report.as_ref()
    }
    pub fn execution_stats(&self) -> &ExecutionStats {
        &self.cx.stats
    }
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }
    pub async fn step(&mut self) -> Result<StepReport<T>> {
        self.step_with_input(None).await
    }
    /// Inputs are immutable for the entire transaction. Reward sources can
    /// combine them with current/post-clone observations without reusing stale
    /// derived values. The evaluation counter counts submitted reward rows,
    /// including masked rows, not a provider's internal simulator calls.
    pub async fn step_with_input(
        &mut self,
        input: Option<&InputBatch<T>>,
    ) -> Result<StepReport<T>> {
        self.step_transaction(input, None).await
    }
    /// Admit a batch that derives both first-class observations and rewards.
    /// Later-stage reward refresh uses the configured RewardSource; it must
    /// implement the same objective at the changed observations/domain state.
    pub async fn step_with_extraction(
        &mut self,
        input: &InputBatch<T>,
        pipeline: &crate::extraction::ExtractionPipeline<T>,
    ) -> Result<StepReport<T>> {
        self.config
            .working_set_bytes(&self.population, &self.history, Some(input))?;
        let mut population = pipeline.extract(input, &self.population)?;
        self.config.validate(&population, self.gradient.is_some())?;
        let fields = population
            .observations
            .fields
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        self.domain.reconcile(&mut population, &fields)?;
        self.step_transaction(Some(input), Some(population)).await
    }
    async fn step_transaction(
        &mut self,
        input: Option<&InputBatch<T>>,
        extracted: Option<Population<T>>,
    ) -> Result<StepReport<T>> {
        let previous_trace = self.cx.stage_trace.clone();
        let previous_stages = self.cx.recorded_stages.clone();
        let previous_noise = self.cx.recorded_noise.clone();
        let previous_fields = self.cx.recorded_fields.clone();
        let previous_influences = self.cx.recorded_influences.clone();
        let previous_stats = self.cx.stats.clone();
        let previous_allowance = self.cx.max_memory_bytes;
        let result = self.execute_transaction(input, extracted).await;
        if result.is_err() {
            self.cx.stage_trace = previous_trace;
            self.cx.recorded_stages = previous_stages;
            self.cx.recorded_noise = previous_noise;
            self.cx.recorded_fields = previous_fields;
            self.cx.recorded_influences = previous_influences;
            self.cx.stats = previous_stats;
            self.cx.max_memory_bytes = previous_allowance;
        }
        result
    }
    async fn execute_transaction(
        &mut self,
        input: Option<&InputBatch<T>>,
        extracted: Option<Population<T>>,
    ) -> Result<StepReport<T>> {
        self.cancellation.check()?;
        if let Some(archive) = &self.recording {
            require(
                archive.steps.len() < archive.config.max_steps,
                "recording horizon reached; export archive and start a new recording",
            )?;
        }
        if let Some(stages) = &mut self.cx.recorded_stages {
            stages.clear();
        }
        if let Some(noise) = &mut self.cx.recorded_noise {
            noise.clear();
        }
        if let Some(fields) = &mut self.cx.recorded_fields {
            fields.clear();
        }
        if let Some(influences) = &mut self.cx.recorded_influences {
            influences.clear();
        }
        if let Some(trace) = &mut self.cx.stage_trace {
            trace.clear();
        }
        self.cx.max_memory_bytes = self.execution_allowance(&self.population, input)?;
        if let Some(input) = input {
            require(
                self.config.cloning_donors.history_window == 0,
                "external input with historical cloning needs a source-aligned input adapter; not implemented",
            )?;
            input.validate()?;
            require(
                input.rows == self.population.len(),
                "external input walker count",
            )?;
        }
        let extracted_version = extracted.as_ref().map(|p| p.version);
        let mut p = extracted.unwrap_or_else(|| self.population.clone());
        let next = self
            .step
            .checked_add(1)
            .ok_or_else(|| GasError::Numerical("step overflow".into()))?;
        let mut evals = self.reward_evaluations;
        self.domain.refresh_observations(&mut p)?;
        self.operators
            .boundary(&self.config.boundary, &mut p, self.domain.as_ref())?;
        self.cx.max_memory_bytes = self.execution_allowance(&p, input)?;
        p.observations.provenance.population_version = p.version;
        p.observations.provenance.stage = "pre_clone".into();
        if extracted_version != Some(p.version) {
            p.rewards = self
                .reward
                .evaluate(&p, input, "pre_clone", &mut self.cx)
                .await?;
        }
        evals += p.len() as u64;
        validate_rewards(&mut p, &self.config)?;
        let alive = p.eligible(self.config.include_truncated);
        if !alive.iter().any(|&x| x) {
            return Err(GasError::Extinction);
        }
        self.cx.trace_population("pre_clone", &p);
        let distance_pool = DonorPool::freeze(
            &p,
            self.step,
            &self.history,
            self.config.distance_donors.history_window,
            self.config.include_truncated,
        )?;
        let distance_companions = self
            .operators
            .companions(
                &self.config.distance_donors,
                CompanionRequest {
                    population: &p,
                    pool: &distance_pool,
                    eligible: &alive,
                    seed: self.config.seed,
                    step: next,
                    stream: Stream::Distance,
                },
                &mut self.cx,
            )
            .await?;
        distance_companions.validate(&distance_pool)?;
        require(
            distance_companions.rows == p.len()
                && distance_companions.count == self.config.distance_donors.count,
            "distance companion shape differs from configuration",
        )?;
        let separation = self
            .operators
            .measure(
                &self.config.reducer,
                MeasurementRequest {
                    distance: &self.config.distance_donors.distance,
                    population: &p,
                    pool: &distance_pool,
                    companions: &distance_companions,
                },
                &mut self.cx,
            )
            .await?;
        require(
            separation.len() == p.len()
                && separation
                    .iter()
                    .zip(&alive)
                    .all(|(v, &a)| !a || (v.is_finite() && *v >= T::ZERO)),
            "invalid diversity measurements",
        )?;
        let fitness = self.operators.fitness(
            &self.config.fitness,
            &p.rewards,
            &separation,
            &alive,
            &p.observations,
            p.version,
        )?;
        fitness.validate(&alive, p.version)?;
        self.cancellation.check()?;
        let mut clone_pool = DonorPool::freeze(
            &p,
            self.step,
            &self.history,
            self.config.cloning_donors.history_window,
            self.config.include_truncated,
        )?;
        let cloning_companions = self
            .operators
            .companions(
                &self.config.cloning_donors,
                CompanionRequest {
                    population: &p,
                    pool: &clone_pool,
                    eligible: &alive,
                    seed: self.config.seed,
                    step: next,
                    stream: Stream::Cloning,
                },
                &mut self.cx,
            )
            .await?;
        cloning_companions.validate(&clone_pool)?;
        require(
            cloning_companions.rows == p.len() && cloning_companions.count == 1,
            "cloning companion shape differs from configuration",
        )?;
        let mut donor_fitness = vec![T::ZERO; clone_pool.sources.len()];
        for (j, s) in clone_pool.sources.iter().enumerate() {
            if s.frame == self.step {
                donor_fitness[j] = fitness.fitness[s.slot as usize];
            }
        }
        if clone_pool.sources.iter().any(|s| s.frame != self.step) {
            // Historical scores are re-evaluated, not looked up in current slot
            // arrays. Their positive maps use current global channel statistics.
            require(
                !matches!(
                    self.config.fitness.reward_standardizer,
                    Standardizer::Local { .. }
                ) && !matches!(
                    self.config.fitness.diversity_standardizer,
                    Standardizer::Local { .. }
                ),
                "historical cloning currently requires global standardization",
            )?;
            clone_pool.population.rewards = self
                .reward
                .evaluate(
                    &clone_pool.population,
                    input,
                    "historical_rescore",
                    &mut self.cx,
                )
                .await?;
            evals += clone_pool.sources.len() as u64;
            validate_rewards(&mut clone_pool.population, &self.config)?;
            require(
                clone_pool
                    .population
                    .validity
                    .iter()
                    .all(|s| s.eligible(self.config.include_truncated)),
                "historical reward became invalid under current inputs",
            )?;
            let score_pool = DonorPool::freeze(
                &clone_pool.population,
                self.step,
                &[],
                0,
                self.config.include_truncated,
            )?;
            let mut module = self.config.distance_donors.clone();
            module.history_window = 0;
            // A separate stream prevents rescore work from changing primary draws.
            let all_alive = vec![true; score_pool.sources.len()];
            let companions = self
                .operators
                .companions(
                    &module,
                    CompanionRequest {
                        population: &clone_pool.population,
                        pool: &score_pool,
                        eligible: &all_alive,
                        seed: self.config.seed,
                        step: next,
                        stream: Stream::HistoricalDistance,
                    },
                    &mut self.cx,
                )
                .await?;
            companions.validate(&score_pool)?;
            require(
                companions.rows == all_alive.len() && companions.count == module.count,
                "historical companion shape",
            )?;
            let distance = self
                .operators
                .measure(
                    &self.config.reducer,
                    MeasurementRequest {
                        distance: &module.distance,
                        population: &clone_pool.population,
                        pool: &score_pool,
                        companions: &companions,
                    },
                    &mut self.cx,
                )
                .await?;
            require(
                distance.len() == all_alive.len()
                    && distance.iter().all(|v| v.is_finite() && *v >= T::ZERO),
                "historical distance shape or value",
            )?;
            let floor = T::from_f64(self.config.fitness.distance_floor);
            let rz = clone_pool
                .population
                .rewards
                .raw
                .iter()
                .map(|&x| {
                    (self.config.fitness.direction.orient(x) - fitness.reward_stats.mean[0])
                        / fitness.reward_stats.scale[0]
                })
                .collect::<Vec<_>>();
            let dz = distance
                .iter()
                .map(|&x| {
                    ((x * x + floor * floor).sqrt() - fitness.diversity_stats.mean[0])
                        / fitness.diversity_stats.scale[0]
                })
                .collect::<Vec<_>>();
            let scores =
                self.operators
                    .historical_fitness(&self.config.fitness, &rz, &dz, &all_alive)?;
            require(
                scores.len() == all_alive.len()
                    && scores.iter().all(|v| v.is_finite() && *v > T::ZERO),
                "historical fitness shape or value",
            )?;
            for (j, s) in clone_pool.sources.iter().enumerate() {
                if s.frame != self.step {
                    donor_fitness[j] = scores[j];
                }
            }
        }
        let plan = self.operators.clone_plan(
            &self.config.clone_decision,
            CloneRequest {
                population: &p,
                pool: &clone_pool,
                companions: &cloning_companions,
                fitness: &fitness.fitness,
                donor_fitness: &donor_fitness,
                alive: &alive,
                seed: self.config.seed,
                step: next,
            },
        )?;
        let mut destination = plan.apply_literal(&p, &clone_pool)?;
        self.cx.trace_population("literal_clone", &destination);
        let changed = self
            .operators
            .transform(
                &self.config.clone_transform,
                TransformRequest {
                    before: &p,
                    pool: &clone_pool,
                    plan: &plan,
                    companions: &cloning_companions,
                    seed: self.config.seed,
                    step: next,
                },
                &mut destination,
                &mut self.cx,
            )
            .await?;
        self.cx.trace_population("post_transform", &destination);
        self.domain.reconcile(&mut destination, &changed)?;
        self.domain.refresh_observations(&mut destination)?;
        self.cx.max_memory_bytes = self.execution_allowance(&destination, input)?;
        // Classify/repair newly transformed coordinates before a bounded
        // reward provider sees them and before applying invalid-reward policy.
        self.operators.boundary(
            &self.config.boundary,
            &mut destination,
            self.domain.as_ref(),
        )?;
        destination.rewards = self
            .reward
            .evaluate(&destination, input, "post_clone", &mut self.cx)
            .await?;
        evals += destination.len() as u64;
        validate_rewards(&mut destination, &self.config)?;
        self.operators.boundary(
            &self.config.boundary,
            &mut destination,
            self.domain.as_ref(),
        )?;
        self.cx.trace_population("post_clone", &destination);
        self.cancellation.check()?;
        if destination
            .validity
            .iter()
            .any(|s| s.eligible(self.config.include_truncated))
        {
            self.operators
                .kinetic(
                    &self.config.kinetic,
                    &mut destination,
                    KineticContext {
                        gradient: self.gradient.as_deref(),
                        domain: self.domain.as_ref(),
                        boundary: &self.config.boundary,
                        include_truncated: self.config.include_truncated,
                        seed: self.config.seed,
                        step: next,
                        operators: Some(self.operators.as_ref()),
                        frozen_fitness: Some(crate::operators::FrozenFitnessContext {
                            population: &p,
                            pool: &distance_pool,
                            companions: &distance_companions,
                            alive: &alive,
                            config: &self.config,
                            clone_plan: &plan,
                        }),
                    },
                    &mut self.cx,
                )
                .await?;
        }
        self.operators.boundary(
            &self.config.boundary,
            &mut destination,
            self.domain.as_ref(),
        )?;
        destination.rewards = self
            .reward
            .evaluate(&destination, input, "post_kinetic", &mut self.cx)
            .await?;
        evals += destination.len() as u64;
        validate_rewards(&mut destination, &self.config)?;
        destination.observations.provenance.population_version = destination.version;
        destination.observations.provenance.stage = "post_kinetic".into();
        self.config
            .working_set_bytes(&destination, &self.history, input)?;
        self.cx.trace_population("post_kinetic", &destination);
        let report = StepReport {
            step: next,
            source_version: p.version,
            result_version: destination.version,
            pre_clone_rewards: p.rewards.clone(),
            pre_clone_eligible: alive,
            pre_clone_fitness: fitness,
            distance_companions,
            distance_sources: distance_pool.sources,
            cloning_companions,
            clones: plan
                .choices
                .iter()
                .filter(|c| c.accepted && !c.revival)
                .count(),
            revivals: plan.choices.iter().filter(|c| c.revival).count(),
            clone_plan: plan,
            final_rewards: destination.rewards.clone(),
            eligible: destination
                .eligible(self.config.include_truncated)
                .iter()
                .filter(|&&x| x)
                .count(),
            reward_evaluations: evals,
            execution: self.cx.stats.clone(),
        };
        report.validate(&self.config, &destination, next, evals)?;
        self.cancellation.check()?;
        if let Some(archive) = &mut self.recording {
            let record = crate::tracking::RecordedStep {
                epoch: archive.epoch,
                before: p.clone(),
                final_population: destination.clone(),
                stages: self.cx.recorded_stages.clone().unwrap_or_default(),
                noise: self.cx.recorded_noise.clone().unwrap_or_default(),
                field_evaluations: self.cx.recorded_fields.clone().unwrap_or_default(),
                influences: self.cx.recorded_influences.clone().unwrap_or_default(),
                report: report.clone(),
                donor_fitness,
            };
            archive.append(record)?;
            let admission = (|| {
                crate::memory::enforce(
                    crate::memory::checked_add(
                        archive.buffer_bytes()?,
                        self.config
                            .working_set_bytes(&destination, &self.history, input)?,
                    )?,
                    self.config.max_memory_bytes,
                )
            })();
            if let Err(error) = admission {
                archive.steps.pop();
                return Err(error);
            }
        }
        let window = self
            .config
            .distance_donors
            .history_window
            .max(self.config.cloning_donors.history_window);
        if window > 0 {
            self.history.push((self.step, p));
            if self.history.len() > window {
                self.history.remove(0);
            }
        }
        self.population = destination;
        self.step = next;
        self.reward_evaluations = evals;
        self.last_report = Some(report.clone());
        Ok(report)
    }
    /// Explicit observation/input extraction barrier. Revalidates the complete
    /// population, reconciles opaque state, and refreshes reward before commit.
    pub async fn replace_population(&mut self, population: Population<T>) -> Result<()> {
        let previous_stats = self.cx.stats.clone();
        let previous_allowance = self.cx.max_memory_bytes;
        let result = self.replace_population_transaction(population).await;
        if result.is_err() {
            self.cx.stats = previous_stats;
            self.cx.max_memory_bytes = previous_allowance;
        }
        result
    }
    async fn replace_population_transaction(
        &mut self,
        mut population: Population<T>,
    ) -> Result<()> {
        population.version = self
            .population
            .version
            .checked_add(1)
            .ok_or_else(|| GasError::Numerical("population version overflow".into()))?;
        self.config.validate(&population, self.gradient.is_some())?;
        self.cx.max_memory_bytes = self.execution_allowance(&population, None)?;
        let fields = population
            .observations
            .fields
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        self.domain.reconcile(&mut population, &fields)?;
        self.domain.refresh_observations(&mut population)?;
        self.operators
            .boundary(&self.config.boundary, &mut population, self.domain.as_ref())?;
        population.rewards = self
            .reward
            .evaluate(&population, None, "external_replace", &mut self.cx)
            .await?;
        validate_rewards(&mut population, &self.config)?;
        population.observations.provenance.population_version = population.version;
        population.observations.provenance.stage = "external_replace".into();
        let reward_evaluations = self
            .reward_evaluations
            .checked_add(population.len() as u64)
            .ok_or_else(|| GasError::Numerical("reward counter overflow".into()))?;
        if let Some(archive) = &mut self.recording {
            let previous_epoch = archive.epoch;
            archive.anchor(self.step, &population)?;
            let admission = (|| {
                crate::memory::enforce(
                    crate::memory::checked_add(
                        archive.buffer_bytes()?,
                        self.config.working_set_bytes(&population, &[], None)?,
                    )?,
                    self.config.max_memory_bytes,
                )
            })();
            if let Err(error) = admission {
                archive.anchors.pop();
                archive.epoch = previous_epoch;
                return Err(error);
            }
        }
        self.reward_evaluations = reward_evaluations;
        self.population = population;
        self.history.clear();
        self.last_report = None;
        if let Some(trace) = &mut self.cx.stage_trace {
            trace.clear();
        }
        Ok(())
    }
    pub fn checkpoint(&self) -> Checkpoint<T> {
        Checkpoint {
            schema_version: crate::checkpoint::CHECKPOINT_VERSION,
            rng_version: RNG_VERSION,
            config: self.config.clone(),
            population: self.population.clone(),
            step: self.step,
            reward_evaluations: self.reward_evaluations,
            execution: self.cx.stats.clone(),
            history: self.history.clone(),
            last_report: self.last_report.clone(),
            recording: self.recording.clone(),
            reward_provider: self.reward.id(),
            gradient_provider: self.gradient.as_ref().map(|g| g.id()),
            domain_provider: self.domain.id(),
            operator_set: self.operators.id(),
        }
    }
    pub fn restore(&mut self, checkpoint: Checkpoint<T>) -> Result<()> {
        checkpoint.validate()?;
        require(
            checkpoint.schema_version == crate::checkpoint::CHECKPOINT_VERSION
                && checkpoint.rng_version == RNG_VERSION,
            "unsupported checkpoint/RNG version",
        )?;
        require(
            checkpoint.config == self.config,
            "checkpoint execution and operator configuration differs",
        )?;
        require(
            checkpoint.reward_provider == self.reward.id()
                && checkpoint.gradient_provider == self.gradient.as_ref().map(|g| g.id())
                && checkpoint.domain_provider == self.domain.id()
                && checkpoint.operator_set == self.operators.id(),
            "checkpoint provider identity differs",
        )?;
        self.config
            .validate(&checkpoint.population, self.gradient.is_some())?;
        self.operators
            .validate(&self.config, &checkpoint.population)?;
        let window = self
            .config
            .distance_donors
            .history_window
            .max(self.config.cloning_donors.history_window);
        require(
            checkpoint.history.len() <= window,
            "checkpoint donor history exceeds configured window",
        )?;
        let mut last = None;
        for (frame, p) in &checkpoint.history {
            p.validate()?;
            require(
                *frame < checkpoint.step && last.is_none_or(|previous| previous < *frame),
                "invalid checkpoint history ordering",
            )?;
            last = Some(*frame);
        }
        self.population = checkpoint.population;
        self.step = checkpoint.step;
        self.reward_evaluations = checkpoint.reward_evaluations;
        self.history = checkpoint.history;
        self.last_report = checkpoint.last_report;
        self.recording = checkpoint.recording;
        self.cx.recorded_stages = self.recording.as_ref().map(|_| Vec::new());
        self.cx.recorded_noise = self.recording.as_ref().map(|_| Vec::new());
        self.cx.recorded_fields = self.recording.as_ref().map(|_| Vec::new());
        self.cx.recorded_influences = self.recording.as_ref().map(|_| Vec::new());
        self.cx.stats = checkpoint.execution;
        self.cancellation.reset();
        if let Some(trace) = &mut self.cx.stage_trace {
            trace.clear();
        }
        Ok(())
    }
}
