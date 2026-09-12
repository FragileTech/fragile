//! Batch-level extension hooks. Override only the stages being studied; default
//! methods preserve the built-in implementation for every other stage. Hooks
//! operate on an uncommitted population and must keep external effects reversible.
use crate::{
    ExecutionContext, ObservationBatch, Population, Real, Result, RewardBatch, TensorBatch,
    boundary::BoundaryPolicy,
    cloning::{CloneDecision, ClonePlan, CloneTransform},
    domain::{DomainAdapter, OperatorFuture},
    donor::{
        CompanionBatch, CompanionReducer, CompanionRequest, CompanionSampler, DonorModule,
        DonorPool,
    },
    fitness::{FitnessBatch, FitnessPipeline},
    geometry::Distance,
    kinetic::{KineticContext, KineticOperator},
    noise::{Noise, NoiseRequest, NoiseSource},
};

pub struct MeasurementRequest<'a, T: Real> {
    pub distance: &'a Distance,
    pub population: &'a Population<T>,
    pub pool: &'a DonorPool<T>,
    pub companions: &'a CompanionBatch,
}
pub struct CloneRequest<'a, T: Real> {
    pub population: &'a Population<T>,
    pub pool: &'a DonorPool<T>,
    pub companions: &'a CompanionBatch,
    pub fitness: &'a [T],
    pub donor_fitness: &'a [T],
    pub alive: &'a [bool],
    pub seed: u64,
    pub step: u64,
}
pub struct TransformRequest<'a, T: Real> {
    pub before: &'a Population<T>,
    pub pool: &'a DonorPool<T>,
    pub plan: &'a ClonePlan,
    pub companions: &'a CompanionBatch,
    pub seed: u64,
    pub step: u64,
}

/// Immutable selection-stage data carried transactionally to O-stage providers.
/// Query observations are supplied separately at the actual noise evaluation.
#[derive(Clone, Copy)]
pub struct FrozenFitnessContext<'a, T: Real> {
    pub population: &'a Population<T>,
    pub pool: &'a DonorPool<T>,
    pub companions: &'a CompanionBatch,
    pub alive: &'a [bool],
    pub config: &'a crate::GasConfig,
    pub clone_plan: &'a ClonePlan,
}

/// Own the identifiers and parameters of any custom implementation. `id` must
/// change when its semantics/parameters change; checkpoints verify it. Numerical
/// loops belong inside concrete operators, not per-walker dynamic dispatch.
pub trait GasOperators<T: Real> {
    fn id(&self) -> String;
    fn validate(&self, _config: &crate::GasConfig, _population: &Population<T>) -> Result<()> {
        Ok(())
    }
    fn boundary(
        &self,
        policy: &BoundaryPolicy,
        p: &mut Population<T>,
        domain: &dyn DomainAdapter<T>,
    ) -> Result<()> {
        crate::kinetic::check_boundary(p, policy, domain)
    }
    fn companions<'a>(
        &'a self,
        module: &'a DonorModule,
        request: CompanionRequest<'a, T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, CompanionBatch> {
        Box::pin(async move { module.sample(request, cx).await })
    }
    fn measure<'a>(
        &'a self,
        reducer: &'a CompanionReducer,
        request: MeasurementRequest<'a, T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, Vec<T>> {
        Box::pin(async move {
            reducer
                .measure(
                    request.distance,
                    request.population,
                    request.pool,
                    request.companions,
                    cx,
                )
                .await
        })
    }
    fn fitness(
        &self,
        pipeline: &FitnessPipeline,
        rewards: &RewardBatch<T>,
        separation: &[T],
        alive: &[bool],
        obs: &ObservationBatch<T>,
        version: u64,
    ) -> Result<FitnessBatch<T>> {
        pipeline.evaluate(rewards, separation, alive, obs, version)
    }
    fn historical_fitness(
        &self,
        pipeline: &FitnessPipeline,
        reward_z: &[T],
        diversity_z: &[T],
        alive: &[bool],
    ) -> Result<Vec<T>> {
        pipeline.combine(reward_z, diversity_z, alive)
    }
    fn clone_plan(&self, decision: &CloneDecision, r: CloneRequest<'_, T>) -> Result<ClonePlan> {
        decision.plan(
            r.population,
            r.pool,
            r.companions,
            r.fitness,
            r.donor_fitness,
            r.alive,
            r.seed,
            r.step,
        )
    }
    fn transform<'a>(
        &'a self,
        transform: &'a CloneTransform,
        r: TransformRequest<'a, T>,
        destination: &'a mut Population<T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, Vec<String>> {
        Box::pin(async move {
            transform
                .apply(
                    r.before,
                    destination,
                    r.pool,
                    r.plan,
                    r.companions,
                    r.seed,
                    r.step,
                    cx,
                )
                .await
        })
    }
    fn noise<'a>(
        &'a self,
        noise: &'a Noise,
        observations: &'a ObservationBatch<T>,
        request: NoiseRequest,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>> {
        Box::pin(async move { noise.sample(observations, request, cx).await })
    }
    fn conditioned_noise<'a>(
        &'a self,
        noise: &'a Noise,
        observations: &'a ObservationBatch<T>,
        request: NoiseRequest,
        _frozen: Option<&'a FrozenFitnessContext<'a, T>>,
        _eligible: Option<&'a [bool]>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>> {
        self.noise(noise, observations, request, cx)
    }
    fn kinetic<'a>(
        &'a self,
        operator: &'a KineticOperator,
        population: &'a mut Population<T>,
        context: KineticContext<'a, T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, ()> {
        Box::pin(async move { operator.advance(population, context, cx).await })
    }
}
#[derive(Clone, Debug, Default)]
pub struct BuiltinOperators;
impl<T: Real> GasOperators<T> for BuiltinOperators {
    fn id(&self) -> String {
        "builtin-operators/v2".into()
    }
}

pub(crate) struct HookNoise<'a, T: Real> {
    pub operators: &'a dyn GasOperators<T>,
    pub config: &'a Noise,
    pub frozen: Option<FrozenFitnessContext<'a, T>>,
}
impl<T: Real> NoiseSource<T> for HookNoise<'_, T> {
    async fn sample(
        &self,
        obs: &ObservationBatch<T>,
        request: NoiseRequest,
        cx: &mut ExecutionContext,
    ) -> Result<TensorBatch<T>> {
        self.operators
            .conditioned_noise(self.config, obs, request, self.frozen.as_ref(), None, cx)
            .await
    }
    async fn sample_masked(
        &self,
        obs: &ObservationBatch<T>,
        request: NoiseRequest,
        eligible: &[bool],
        cx: &mut ExecutionContext,
    ) -> Result<TensorBatch<T>> {
        self.operators
            .conditioned_noise(
                self.config,
                obs,
                request,
                self.frozen.as_ref(),
                Some(eligible),
                cx,
            )
            .await
    }
}
