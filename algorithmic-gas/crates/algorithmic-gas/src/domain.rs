use crate::{
    ExecutionContext, ExtractionContext, GasError, InputBatch, ObservationBatch,
    ObservationExtractor, Population, Provenance, Real, Result, RewardBatch, RewardExtractor,
    TensorBatch,
};
use std::{future::Future, pin::Pin};

pub type OperatorFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + 'a>>;

/// A reward is a raw scalar objective, not fitness. Providers receive the
/// immutable current population and optional external inputs, so an action,
/// function evaluation, or velocity-dependent functional can be composed here.
pub trait RewardSource<T: Real> {
    fn id(&self) -> String;
    fn evaluate<'a>(
        &'a self,
        population: &'a Population<T>,
        input: Option<&'a InputBatch<T>>,
        stage: &'a str,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, RewardBatch<T>>;
}
pub trait GradientProvider<T: Real> {
    fn id(&self) -> String;
    /// Gradient of the declared potential U, evaluated at this exact substep.
    /// Not a derivative of sampled fitness. Ineligible rows must return zeros.
    fn gradient<'a>(
        &'a self,
        population: &'a Population<T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>>;
}
pub trait HessianProvider<T: Real> {
    fn hessian<'a>(
        &'a self,
        population: &'a Population<T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>>;
}
pub trait DriftProvider<T: Real> {
    fn drift<'a>(
        &'a self,
        population: &'a Population<T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>>;
}
pub trait DiffusionProvider<T: Real> {
    fn factor<'a>(
        &'a self,
        population: &'a Population<T>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, TensorBatch<T>>;
}

/// Domain operations modify only the supplied transactional population. An
/// adapter must not advance an external simulator irreversibly before commit.
/// Restore/advance/capture from a per-walker snapshot satisfies this contract.
pub trait DomainAdapter<T: Real> {
    fn id(&self) -> String;
    fn refresh_observations(&self, population: &mut Population<T>) -> Result<()>;
    fn reconcile(&self, population: &mut Population<T>, changed_fields: &[String]) -> Result<()>;
    fn transition(
        &self,
        population: &mut Population<T>,
        eligible: &[bool],
        seed: u64,
        step: u64,
    ) -> Result<()>;
}
#[derive(Clone, Debug, Default)]
pub struct NumericalDomain;
impl<T: Real> DomainAdapter<T> for NumericalDomain {
    fn id(&self) -> String {
        "numerical/v1".into()
    }
    fn refresh_observations(&self, p: &mut Population<T>) -> Result<()> {
        if p.states.is_some() {
            return Err(GasError::Domain(
                "opaque snapshots require their own domain adapter".into(),
            ));
        }
        Ok(())
    }
    fn reconcile(&self, p: &mut Population<T>, _: &[String]) -> Result<()> {
        self.refresh_observations(p)
    }
    fn transition(&self, _: &mut Population<T>, _: &[bool], _: u64, _: u64) -> Result<()> {
        Err(GasError::Capability(
            "numerical domain has no environment transition".into(),
        ))
    }
}
pub struct NamedObservationExtractor {
    pub fields: Vec<(String, String)>,
}
impl<T: Real> ObservationExtractor<T> for NamedObservationExtractor {
    fn extract(
        &self,
        input: &InputBatch<T>,
        context: &ExtractionContext<'_, T>,
    ) -> Result<ObservationBatch<T>> {
        input.validate()?;
        let fields = self
            .fields
            .iter()
            .map(|(source, target)| {
                Ok((
                    target.clone(),
                    input
                        .numerical
                        .get(source)
                        .cloned()
                        .ok_or_else(|| GasError::MissingField(source.clone()))?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(ObservationBatch {
            fields,
            provenance: Provenance {
                input_version: input.version,
                population_version: context.population.version,
                stage: context.stage.into(),
            },
        })
    }
}
pub struct NamedRewardExtractor {
    pub field: String,
}
impl<T: Real> RewardExtractor<T> for NamedRewardExtractor {
    fn extract(
        &self,
        input: &InputBatch<T>,
        context: &ExtractionContext<'_, T>,
    ) -> Result<RewardBatch<T>> {
        input.validate()?;
        let values = input
            .numerical
            .get(&self.field)
            .ok_or_else(|| GasError::MissingField(self.field.clone()))?;
        if values.width() != 1 {
            return Err(GasError::Shape(
                "reward extraction must produce one scalar per walker".into(),
            ));
        }
        Ok(RewardBatch::new(
            values.values().to_vec(),
            Provenance {
                input_version: input.version,
                population_version: context.population.version,
                stage: context.stage.into(),
            },
        ))
    }
}
