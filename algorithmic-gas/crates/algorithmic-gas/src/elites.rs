//! Historical best walkers. Selection does not consume randomness.
use crate::{GasConfig, Population, Real, Result, error::require};
use serde::{Deserialize, Serialize};

/// A present but empty bank distinguishes a new elite-enabled run from a legacy checkpoint.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(bound = "T: Real", deny_unknown_fields)]
pub struct EliteBank<T: Real> {
    pub population: Option<Population<T>>,
}

fn copy_row<T: Real>(
    source: &Population<T>,
    row: usize,
    target: &mut Population<T>,
    slot: usize,
) -> Result<()> {
    for (name, field) in &mut target.observations.fields {
        field.replace_row(slot, source.observations.field(name)?.row(row)?)?;
    }
    target.rewards.raw[slot] = source.rewards.raw[row];
    target.rewards.valid[slot] = source.rewards.valid[row];
    target.validity[slot] = source.validity[row];
    target.generations[slot] = source.generations[row];
    if let Some(states) = &mut target.states {
        let src = source
            .states
            .as_ref()
            .ok_or_else(|| crate::GasError::Domain("missing elite state".into()))?;
        require(states.codec == src.codec, "elite state codec differs")?;
        states.snapshots[slot] = src.snapshots[row].clone();
    }
    Ok(())
}

impl<T: Real> EliteBank<T> {
    pub(crate) fn validate(&self, config: &GasConfig, current: &Population<T>) -> Result<()> {
        if let Some(p) = &self.population {
            p.validate()?;
            require(
                config.n_elite > 0 && p.len() == config.n_elite,
                "elite bank size",
            )?;
            require(
                p.version <= current.version
                    && p.rewards.provenance.population_version == p.version
                    && p.observations.provenance.population_version == p.version,
                "elite bank version",
            )?;
            require(
                p.observations.fields.len() == current.observations.fields.len()
                    && p.observations.fields.iter().all(|(name, f)| {
                        current
                            .observations
                            .fields
                            .get(name)
                            .is_some_and(|other| f.item_shape() == other.item_shape())
                    }),
                "elite observation schema",
            )?;
            require(
                p.states.as_ref().map(|s| &s.codec) == current.states.as_ref().map(|s| &s.codec),
                "elite state schema",
            )?;
            require(
                (0..p.len()).all(|i| {
                    p.validity[i].eligible(config.include_truncated)
                        && p.rewards.valid[i]
                        && p.rewards.raw[i].is_finite()
                }),
                "invalid elite walker",
            )?;
        }
        Ok(())
    }
    pub(crate) fn inject(&self, p: &mut Population<T>) -> Result<usize> {
        let Some(bank) = &self.population else {
            return Ok(0);
        };
        require(bank.len() <= p.len(), "elite bank exceeds population")?;
        for i in 0..bank.len() {
            // Generations identify recipient incarnations, not physical donor state.
            let generation = p.generations[i]
                .checked_add(1)
                .ok_or_else(|| crate::GasError::Numerical("elite generation overflow".into()))?;
            copy_row(bank, i, p, i)?;
            p.generations[i] = generation;
        }
        p.version = p
            .version
            .checked_add(1)
            .ok_or_else(|| crate::GasError::Numerical("population version overflow".into()))?;
        p.observations.provenance.population_version = p.version;
        p.rewards.provenance.population_version = p.version;
        Ok(bank.len())
    }
    pub(crate) fn select(&self, config: &GasConfig, current: &Population<T>) -> Result<Self> {
        if config.n_elite == 0 {
            return Ok(Self { population: None });
        }
        let mut candidates = Vec::new();
        for source in self.population.iter().chain(std::iter::once(current)) {
            for i in 0..source.len() {
                if source.validity[i].eligible(config.include_truncated)
                    && source.rewards.valid[i]
                    && source.rewards.raw[i].is_finite()
                {
                    candidates.push((source, i));
                }
            }
        }
        candidates.sort_by(|(a, i), (b, j)| {
            config
                .fitness
                .direction
                .orient(b.rewards.raw[*j])
                .partial_cmp(&config.fitness.direction.orient(a.rewards.raw[*i]))
                .unwrap()
        });
        if candidates.is_empty() {
            return Ok(Self { population: None });
        }
        let indices = vec![0; config.n_elite];
        let mut bank = Population {
            observations: current.observations.gather(&indices)?,
            rewards: current.rewards.gather(&indices)?,
            validity: vec![current.validity[0]; config.n_elite],
            states: current
                .states
                .as_ref()
                .map(|s| s.gather(&indices))
                .transpose()?,
            generations: vec![0; config.n_elite],
            version: current.version,
        };
        let selected = candidates.len().min(config.n_elite);
        for i in 0..config.n_elite {
            let (source, row) = candidates[i % selected];
            copy_row(source, row, &mut bank, i)?;
        }
        Ok(Self {
            population: Some(bank),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObservationBatch, TensorBatch, batch::StateStore, fitness::ObjectiveDirection};

    fn population() -> Population<f64> {
        let mut p = Population::new(ObservationBatch::positions(
            TensorBatch::scalars(vec![10., 20., 30.]).unwrap(),
        ))
        .unwrap();
        p.observations.fields.insert(
            "velocities".into(),
            TensorBatch::scalars(vec![1., 2., 3.]).unwrap(),
        );
        p.rewards.raw = vec![2., 1., 1.];
        p.states = Some(StateStore {
            codec: "test".into(),
            snapshots: vec![vec![10], vec![20], vec![30]],
        });
        p
    }
    #[test]
    fn selection_ties_direction_and_complete_rows() {
        let p = population();
        let mut config = GasConfig {
            n_elite: 2,
            ..Default::default()
        };
        config.fitness.direction = ObjectiveDirection::Minimize;
        let empty = EliteBank { population: None };
        let bank = empty.select(&config, &p).unwrap();
        let selected = bank.population.as_ref().unwrap();
        assert_eq!(
            selected.observations.field("positions").unwrap().values(),
            &[20., 30.]
        );
        assert_eq!(
            selected.observations.field("velocities").unwrap().values(),
            &[2., 3.]
        );
        assert_eq!(
            selected.states.as_ref().unwrap().snapshots,
            vec![vec![20], vec![30]]
        );
        let mut changed = p.clone();
        changed.rewards.raw = vec![1.; 3];
        assert_eq!(bank.select(&config, &changed).unwrap(), bank);
        config.fitness.direction = ObjectiveDirection::Maximize;
        assert_eq!(
            empty
                .select(&config, &p)
                .unwrap()
                .population
                .unwrap()
                .rewards
                .raw,
            vec![2., 1.]
        );
        let mut restored = p.clone();
        restored.generations = vec![4, 9, 0];
        assert_eq!(bank.inject(&mut restored).unwrap(), 2);
        assert_eq!(restored.generations, vec![5, 10, 0]);
        assert_eq!(
            restored.states.unwrap().snapshots,
            vec![vec![20], vec![30], vec![30]]
        );
    }
    #[test]
    fn invalid_candidates_repeat_survivors_and_disable() {
        let mut p = population();
        let mut config = GasConfig {
            n_elite: 3,
            ..Default::default()
        };
        p.validity[0].terminated = true;
        p.rewards.valid[1] = false;
        let bank = EliteBank { population: None }.select(&config, &p).unwrap();
        assert_eq!(
            bank.population
                .as_ref()
                .unwrap()
                .observations
                .field("positions")
                .unwrap()
                .values(),
            &[30.; 3]
        );
        p.validity[2].truncated = true;
        assert!(
            EliteBank { population: None }
                .select(&config, &p)
                .unwrap()
                .population
                .is_none()
        );
        config.include_truncated = true;
        assert!(
            EliteBank { population: None }
                .select(&config, &p)
                .unwrap()
                .population
                .is_some()
        );
        config.n_elite = 0;
        assert!(bank.select(&config, &p).unwrap().population.is_none());
    }
}
