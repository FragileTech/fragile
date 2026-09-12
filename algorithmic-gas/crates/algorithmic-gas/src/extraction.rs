use crate::{
    DerivedFieldProvider, ExtractionContext, GasError, InputBatch, ObservationExtractor,
    Population, Real, Result, RewardExtractor,
};

/// Independent observation/reward extractors see the same derived input batch
/// and immutable pre-extraction algorithm state. Derivations are evaluated once
/// per call; no implicit cross-stage cache survives population changes.
pub struct ExtractionPipeline<T: Real> {
    pub observations: Box<dyn ObservationExtractor<T>>,
    pub rewards: Box<dyn RewardExtractor<T>>,
    pub derived: Option<Box<dyn DerivedFieldProvider<T>>>,
}
impl<T: Real> ExtractionPipeline<T> {
    pub fn extract(
        &self,
        input: &InputBatch<T>,
        population: &Population<T>,
    ) -> Result<Population<T>> {
        input.validate()?;
        if input.rows != population.len() {
            return Err(GasError::Shape("extraction walker count".into()));
        }
        let mut expanded = input.clone();
        let context = ExtractionContext {
            population,
            stage: "external_input",
        };
        if let Some(provider) = &self.derived {
            for (name, value) in provider.evaluate(input, &context)? {
                if expanded.numerical.contains_key(&name) {
                    return Err(GasError::Configuration(format!(
                        "derived field would overwrite input {name}"
                    )));
                }
                expanded.numerical.insert(name, value);
            }
        }
        expanded.validate()?;
        let mut output = population.clone();
        output.observations = self.observations.extract(&expanded, &context)?;
        output.rewards = self.rewards.extract(&expanded, &context)?;
        output.version = population
            .version
            .checked_add(1)
            .ok_or_else(|| GasError::Numerical("population version overflow".into()))?;
        output.observations.provenance.population_version = output.version;
        output.rewards.provenance.population_version = output.version;
        output.validate()?;
        Ok(output)
    }
}
