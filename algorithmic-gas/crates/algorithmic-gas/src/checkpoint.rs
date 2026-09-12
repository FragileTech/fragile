//! Validate the complete archive before restoring any live state.
use crate::{
    Checkpoint, GasConfig, Population, Real, Result, RewardBatch, StepReport,
    donor::{CompanionBatch, DonorModule, SourceRef},
    error::require,
    random::RNG_VERSION,
};
use std::collections::BTreeSet;

/// The single supported checkpoint schema.
pub const CHECKPOINT_VERSION: u32 = 4;

fn rewards<T: Real>(r: &RewardBatch<T>, n: usize, version: u64) -> Result<()> {
    r.validate(n)?;
    require(
        r.provenance.population_version == version,
        "checkpoint reward version",
    )?;
    require(
        r.raw
            .iter()
            .zip(&r.valid)
            .all(|(x, &valid)| !valid || x.is_finite()),
        "checkpoint valid reward is nonfinite",
    )
}

fn sources(
    records: &[SourceRef],
    module: &DonorModule,
    n: usize,
    frame: u64,
    version: u64,
    alive: &[bool],
) -> Result<()> {
    let limit = n.checked_mul(module.history_window.saturating_add(1));
    require(
        !records.is_empty() && limit.is_some_and(|max| records.len() <= max),
        "checkpoint source capacity",
    )?;
    let mut seen = BTreeSet::new();
    let mut current = vec![false; n];
    let earliest = frame.saturating_sub(module.history_window as u64);
    for s in records {
        require(
            (s.slot as usize) < n
                && s.frame <= frame
                && s.frame >= earliest
                && s.version <= version
                && seen.insert((s.frame, s.slot)),
            "invalid or duplicate checkpoint source",
        )?;
        if s.frame == frame {
            require(
                s.version == version && alive[s.slot as usize],
                "invalid current checkpoint source",
            )?;
            current[s.slot as usize] = true;
        } else {
            require(
                s.version < version,
                "historical source version is not older",
            )?;
        }
    }
    require(
        current == alive,
        "checkpoint current source set differs from eligibility",
    )
}

fn companions(
    batch: &CompanionBatch,
    sources: &[SourceRef],
    count: usize,
    alive: &[bool],
    frame: u64,
) -> Result<()> {
    batch.validate_indices(sources.len())?;
    require(
        batch.rows == alive.len() && batch.count == count,
        "checkpoint companion shape",
    )?;
    for (i, &eligible) in alive.iter().enumerate() {
        require(
            eligible || batch.row(i).next().is_none(),
            "dead recipient has companions",
        )?;
        if batch.mutual {
            for a in 0..count {
                let index = i * count + a;
                if !batch.valid[index] {
                    continue;
                }
                let s = sources[batch.indices[index] as usize];
                require(
                    s.frame == frame,
                    "mutual checkpoint companion is historical",
                )?;
                let reverse = s.slot as usize * count + a;
                require(
                    batch.valid[reverse],
                    "missing reciprocal checkpoint companion",
                )?;
                let r = sources[batch.indices[reverse] as usize];
                require(
                    r.frame == frame && r.slot as usize == i,
                    "nonreciprocal checkpoint companion",
                )?;
            }
        }
    }
    Ok(())
}

impl<T: Real> StepReport<T> {
    pub fn validate(
        &self,
        config: &GasConfig,
        population: &Population<T>,
        step: u64,
        evaluations: u64,
    ) -> Result<()> {
        let n = population.len();
        require(
            step > 0
                && self.step == step
                && self.result_version == population.version
                && self.source_version < self.result_version,
            "checkpoint report stage/version mismatch",
        )?;
        let alive = &self.pre_clone_eligible;
        require(
            alive.len() == n && alive.iter().any(|&a| a),
            "checkpoint pre-clone eligibility",
        )?;
        rewards(&self.pre_clone_rewards, n, self.source_version)?;
        rewards(&self.final_rewards, n, self.result_version)?;
        self.pre_clone_fitness
            .validate(alive, self.source_version)?;
        require(
            self.pre_clone_fitness.stage == "pre_clone",
            "checkpoint fitness stage",
        )?;
        require(
            alive
                .iter()
                .enumerate()
                .all(|(i, &a)| !a || self.pre_clone_rewards.valid[i]),
            "eligible checkpoint reward is invalid",
        )?;
        require(
            self.final_rewards.valid == population.rewards.valid
                && self.final_rewards.provenance == population.rewards.provenance
                && self
                    .final_rewards
                    .raw
                    .iter()
                    .zip(&population.rewards.raw)
                    .all(|(&a, &b)| a == b || (a.to_f64().is_nan() && b.to_f64().is_nan())),
            "checkpoint final rewards differ from population",
        )?;
        let frame = step - 1;
        sources(
            &self.distance_sources,
            &config.distance_donors,
            n,
            frame,
            self.source_version,
            alive,
        )?;
        sources(
            &self.clone_plan.sources,
            &config.cloning_donors,
            n,
            frame,
            self.source_version,
            alive,
        )?;
        companions(
            &self.distance_companions,
            &self.distance_sources,
            config.distance_donors.count,
            alive,
            frame,
        )?;
        companions(
            &self.cloning_companions,
            &self.clone_plan.sources,
            1,
            alive,
            frame,
        )?;
        require(
            self.clone_plan.population_version == self.source_version
                && self.clone_plan.choices.len() == n
                && self.clone_plan.mutual == self.cloning_companions.mutual,
            "checkpoint clone plan snapshot",
        )?;
        for source in self.distance_sources.iter().chain(&self.clone_plan.sources) {
            let slot = source.slot as usize;
            let increment = u64::from(self.clone_plan.choices[slot].accepted);
            let before = population.generations[slot]
                .checked_sub(increment)
                .ok_or_else(|| {
                    crate::GasError::Checkpoint("invalid clone generation increment".into())
                })?;
            require(
                if source.frame == frame {
                    source.generation == before
                } else {
                    source.generation <= before
                },
                "checkpoint source generation disagrees with recipient history",
            )?;
        }
        for (i, choice) in self.clone_plan.choices.iter().enumerate() {
            require(
                choice.probability.is_none_or(|p| {
                    p.is_finite()
                        && (0. ..=1.).contains(&p)
                        && (!choice.revival || p == 1.)
                        && (!choice.accepted || p > 0.)
                }),
                "invalid recorded clone probability",
            )?;
            require(
                choice.donors.len() <= 1
                    && (!choice.accepted || choice.donors.len() == 1)
                    && (!choice.revival || (choice.accepted && !alive[i])),
                "invalid checkpoint clone decision",
            )?;
            for donor in &choice.donors {
                require(
                    donor.weight == 1.
                        && (donor.pool_index as usize) < self.clone_plan.sources.len(),
                    "checkpoint clone donor",
                )?;
                let s = self.clone_plan.sources[donor.pool_index as usize];
                if choice.revival {
                    require(s.frame == frame, "revival donor is historical")?;
                } else {
                    require(
                        self.cloning_companions.row(i).next() == Some(donor.pool_index),
                        "clone decision differs from companion proposal",
                    )?;
                }
            }
        }
        require(
            self.clones
                == self
                    .clone_plan
                    .choices
                    .iter()
                    .filter(|c| c.accepted && !c.revival)
                    .count()
                && self.revivals == self.clone_plan.choices.iter().filter(|c| c.revival).count()
                && self.eligible
                    == population
                        .eligible(config.include_truncated)
                        .iter()
                        .filter(|&&a| a)
                        .count()
                && self.reward_evaluations == evaluations,
            "checkpoint report counters",
        )
    }
}

impl<T: Real> Checkpoint<T> {
    pub fn validate(&self) -> Result<()> {
        require(
            self.schema_version == CHECKPOINT_VERSION && self.rng_version == RNG_VERSION,
            "unsupported checkpoint/RNG version",
        )?;
        if let Some(archive) = &self.recording {
            archive.validate()?;
            let (step, terminal) = archive.terminal();
            require(
                step == self.step
                    && terminal.version == self.population.version
                    && terminal.generations == self.population.generations
                    && terminal.validity == self.population.validity,
                "archive terminal identity mismatch",
            )?;
            require(
                terminal.observations.fields.len() == self.population.observations.fields.len(),
                "archive terminal field count",
            )?;
            for (name, field) in &terminal.observations.fields {
                let actual = self.population.observations.field(name)?;
                require(
                    field.item_shape() == actual.item_shape()
                        && field.values().len() == actual.values().len()
                        && field
                            .values()
                            .iter()
                            .zip(actual.values())
                            .all(|(a, b)| a == b || (a.to_f64().is_nan() && b.to_f64().is_nan())),
                    "archive terminal coordinates mismatch",
                )?;
            }
            require(
                archive.gas_config == self.config,
                "archive configuration mismatch",
            )?;
            crate::memory::enforce(
                crate::memory::checked_add(
                    archive.buffer_bytes()?,
                    self.config
                        .working_set_bytes(&self.population, &self.history, None)?,
                )?,
                self.config.max_memory_bytes,
            )?;
        }
        self.config
            .validate(&self.population, self.gradient_provider.is_some())?;
        self.config
            .working_set_bytes(&self.population, &self.history, None)?;
        rewards(
            &self.population.rewards,
            self.population.len(),
            self.population.version,
        )?;
        require(
            self.population.observations.provenance.population_version == self.population.version,
            "checkpoint observation version",
        )?;
        let window = self
            .config
            .distance_donors
            .history_window
            .max(self.config.cloning_donors.history_window);
        require(self.history.len() <= window, "checkpoint history capacity")?;
        let mut previous = None;
        for (frame, p) in &self.history {
            self.config.validate(p, self.gradient_provider.is_some())?;
            rewards(&p.rewards, p.len(), p.version)?;
            require(
                p.len() == self.population.len()
                    && p.version < self.population.version
                    && *frame < self.step
                    && previous.is_none_or(|prev| prev < *frame)
                    && self.step - *frame <= window as u64,
                "checkpoint history ordering or population shape",
            )?;
            require(
                p.states.as_ref().map(|s| &s.codec)
                    == self.population.states.as_ref().map(|s| &s.codec),
                "checkpoint history codec",
            )?;
            for (name, tensor) in &self.population.observations.fields {
                require(
                    p.observations.field(name)?.item_shape() == tensor.item_shape(),
                    "checkpoint historical schema",
                )?;
            }
            require(
                p.observations.fields.len() == self.population.observations.fields.len(),
                "checkpoint historical fields",
            )?;
            previous = Some(*frame);
        }
        if let Some(report) = &self.last_report {
            report.validate(
                &self.config,
                &self.population,
                self.step,
                self.reward_evaluations,
            )?;
            require(
                report.execution.evaluations <= self.execution.evaluations
                    && report.execution.synchronizations <= self.execution.synchronizations
                    && report.execution.uploaded_bytes <= self.execution.uploaded_bytes
                    && report.execution.downloaded_bytes <= self.execution.downloaded_bytes
                    && report.execution.peak_batch_elements <= self.execution.peak_batch_elements,
                "checkpoint execution counters",
            )?;
        }
        Ok(())
    }
}
