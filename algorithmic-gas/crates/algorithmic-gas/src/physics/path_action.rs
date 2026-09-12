//! Likelihood of recorded primitive outcomes, conditional on each retained source state.
//! Counting measure is used for sampler outputs/gates. Gaussian outputs use Bxi
//! Lebesgue measure when nonsingular; singular/uniform maps retain latent coordinates.
//! Deterministic maps and boundary tests carry no additional random density factor.
use super::partvi::{ExperimentRequest, ExperimentResult, Series};
use crate::{
    GasError, ObservationBatch, Provenance, Result, RunArchive, TensorBatch,
    donor::{CompanionBatch, DonorModule, OddPolicy, SamplingLaw, SourceRef},
    geometry::{AlgorithmicDistance, InteractionKernel, Kernel},
    random::Stream,
    tracking::{RecordedStep, StageSnapshot},
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Likelihood {
    Available { log_density: f64 },
    Unavailable { reason: String },
}
impl Likelihood {
    fn value(&self) -> Option<f64> {
        match self {
            Self::Available { log_density } => Some(*log_density),
            _ => None,
        }
    }
    fn from_result(value: Result<f64>) -> Self {
        match value {
            Ok(x) if x.is_finite() => Self::Available { log_density: x },
            Ok(_) => Self::Unavailable {
                reason: "recorded outcome has zero probability or nonfinite density".into(),
            },
            Err(e) => Self::Unavailable {
                reason: e.to_string(),
            },
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Component {
    pub name: String,
    pub carrier: String,
    pub likelihood: Likelihood,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepLikelihood {
    pub epoch: u64,
    pub step: u64,
    pub components: Vec<Component>,
    pub log_density: Option<f64>,
    pub available_log_density: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathLikelihood {
    pub steps: Vec<StepLikelihood>,
    pub complete: bool,
    pub log_density: Option<f64>,
    pub available_log_density: f64,
}
fn unavailable(s: &str) -> GasError {
    GasError::Capability(s.into())
}
fn check(ok: bool, s: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(GasError::Configuration(s.into()))
    }
}

/// Joint probability of an actual *ordered* uniform companion batch.
/// Uniform matchings sum over the shuffles that induce the same matching.
pub fn uniform_assignment_log_probability(
    module: &DonorModule,
    batch: &CompanionBatch,
    sources: &[SourceRef],
    current_frame: u64,
    eligible: &[bool],
) -> Result<f64> {
    check(
        matches!(module.kernel, Kernel::Uniform),
        "uniform probability requires uniform kernel",
    )?;
    check(
        batch.rows == eligible.len() && batch.count == module.count,
        "assignment shape mismatch",
    )?;
    batch.validate_indices(sources.len())?;
    let active: Vec<_> = eligible
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| a.then_some(i))
        .collect();
    check(!active.is_empty(), "empty eligible assignment")?;
    let current = |i: usize| {
        sources
            .iter()
            .position(|s| s.frame == current_frame && s.slot as usize == i)
    };
    for (i, &e) in eligible.iter().enumerate() {
        if !e {
            check(batch.row(i).count() == 0, "ineligible row has companion")?;
        }
    }
    match module.law {
        SamplingLaw::Independent => {
            let mut logp = 0.;
            for &i in &active {
                let mut allowed: BTreeSet<usize> = sources
                    .iter()
                    .enumerate()
                    .filter_map(|(j, s)| {
                        (module.allow_self || s.frame != current_frame || s.slot as usize != i)
                            .then_some(j)
                    })
                    .collect();
                if allowed.is_empty() {
                    allowed.insert(
                        current(i).ok_or_else(|| unavailable("singleton current source absent"))?,
                    );
                }
                let expected = if module.replacement {
                    module.count
                } else {
                    module.count.min(allowed.len())
                };
                check(
                    batch.row(i).count() == expected,
                    "assignment draw count differs from sampler",
                )?;
                for donor in batch.row(i) {
                    check(
                        allowed.contains(&(donor as usize)),
                        "assignment outside candidate set",
                    )?;
                    logp -= (allowed.len() as f64).ln();
                    if !module.replacement {
                        allowed.remove(&(donor as usize));
                    }
                }
            }
            Ok(logp)
        }
        SamplingLaw::FisherYates | SamplingLaw::LegacyPermutation => {
            check(
                sources.iter().all(|s| s.frame == current_frame),
                "matching requires current pool",
            )?;
            let n = active.len();
            let mut logp = 0.;
            for round in 0..module.count {
                let mut map = BTreeMap::new();
                for &i in &active {
                    let at = i * batch.count + round;
                    if batch.valid[at] {
                        let s = sources[batch.indices[at] as usize];
                        check(
                            eligible[s.slot as usize],
                            "matching selected ineligible slot",
                        )?;
                        map.insert(i, s.slot as usize);
                    }
                }
                if module.law == SamplingLaw::LegacyPermutation {
                    check(map.len() == n, "permutation assignment misses row")?;
                    if n == eligible.len() {
                        check(
                            map.values().copied().collect::<BTreeSet<_>>().len() == n,
                            "full-live assignment is not a permutation",
                        )?;
                        logp -= (1..=n).map(|k| (k as f64).ln()).sum::<f64>();
                    } else {
                        logp -= n as f64 * (n as f64).ln();
                    }
                } else {
                    let mut fixed = 0;
                    for (&i, &j) in &map {
                        check(map.get(&j) == Some(&i), "matching is not reciprocal")?;
                        if i == j {
                            fixed += 1;
                        }
                    }
                    let odd = n % 2 == 1;
                    if n == 1 {
                        check(fixed == 1, "singleton matching requires self")?;
                    } else if odd && module.odd == OddPolicy::Unmatched {
                        check(map.len() == n - 1 && fixed == 0, "odd unmatched shape")?;
                    } else {
                        check(
                            map.len() == n && fixed == usize::from(odd),
                            "matching parity mismatch",
                        )?;
                        check(
                            !odd || module.odd != OddPolicy::Reject,
                            "rejected odd matching",
                        )?;
                    }
                    // (n-1)!! matchings for even n; n!! choices for odd n including the leftover.
                    let start = if odd { n } else { n.saturating_sub(1) };
                    logp -= (1..=start)
                        .rev()
                        .step_by(2)
                        .map(|k| (k as f64).ln())
                        .sum::<f64>();
                }
            }
            Ok(logp)
        }
        SamplingLaw::GaussianGreedy => Err(unavailable(
            "Gaussian greedy joint probability requires the unrecorded shuffle/selection path",
        )),
    }
}
fn observations(stage: &StageSnapshot) -> Result<ObservationBatch<f64>> {
    Ok(ObservationBatch {
        fields: stage
            .fields
            .iter()
            .map(|(name, f)| {
                Ok((
                    name.clone(),
                    TensorBatch::new(stage.validity.len(), f.item_shape.clone(), f.values.clone())?,
                ))
            })
            .collect::<Result<_>>()?,
        provenance: Provenance::default(),
    })
}
pub(crate) fn source_observations(
    a: &RunArchive<f64>,
    epoch: u64,
    source: SourceRef,
) -> Result<ObservationBatch<f64>> {
    for s in &a.steps {
        if s.epoch != epoch {
            continue;
        }
        if s.report.step.checked_sub(1) == Some(source.frame)
            && let Some(stage) = s.stages.iter().find(|x| {
                x.stage == "pre_clone"
                    && x.version == source.version
                    && x.generations.get(source.slot as usize) == Some(&source.generation)
            })
        {
            return observations(stage);
        }
        if s.report.step == source.frame
            && s.final_population.version == source.version
            && s.final_population.generations.get(source.slot as usize) == Some(&source.generation)
        {
            return Ok(s.final_population.observations.clone());
        }
        if s.report.step.checked_sub(1) == Some(source.frame)
            && s.before.version == source.version
            && s.before.generations.get(source.slot as usize) == Some(&source.generation)
        {
            return Ok(s.before.observations.clone());
        }
    }
    for anchor in &a.anchors {
        if anchor.epoch == epoch
            && anchor.step == source.frame
            && anchor.population.version == source.version
            && anchor.population.generations.get(source.slot as usize) == Some(&source.generation)
        {
            return Ok(anchor.population.observations.clone());
        }
    }
    Err(unavailable(
        "recorded donor source predates retained coordinate coverage",
    ))
}
fn weighted_probability(
    a: &RunArchive<f64>,
    s: &RecordedStep<f64>,
    module: &DonorModule,
    batch: &CompanionBatch,
    sources: &[SourceRef],
) -> Result<f64> {
    check(
        module.law == SamplingLaw::Independent,
        "weighted joint law requires independent sampler",
    )?;
    if !module.replacement && module.count > 1 {
        return Err(unavailable(
            "weighted top-k storage is not an ordered Plackett-Luce draw; its slot assignment probability is unavailable",
        ));
    }
    check(
        batch
            .rows
            .checked_mul(sources.len())
            .is_some_and(|n| n <= 4_194_304),
        "weighted likelihood edge budget exceeded",
    )?;
    let stage = s
        .stages
        .iter()
        .find(|x| x.stage == "pre_clone")
        .ok_or_else(|| unavailable("pre-clone coordinates absent"))?;
    let query = observations(stage)?;
    let current = s.report.step - 1;
    let mut cache = BTreeMap::new();
    for src in sources {
        let key = (src.frame, src.version);
        if let std::collections::btree_map::Entry::Vacant(e) = cache.entry(key) {
            e.insert(source_observations(a, s.epoch, *src)?);
        }
    }
    let mut logp = 0.;
    for (i, &active) in s.report.pre_clone_eligible.iter().enumerate() {
        if !active {
            continue;
        }
        let mut candidates = sources
            .iter()
            .enumerate()
            .filter(|(_, src)| module.allow_self || src.frame != current || src.slot as usize != i)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            candidates = sources
                .iter()
                .enumerate()
                .filter(|(_, src)| src.frame == current && src.slot as usize == i)
                .collect();
        }
        let mut weights = BTreeMap::new();
        for (j, src) in candidates {
            let obs = &cache[&(src.frame, src.version)];
            let value = module.distance.compare(&query, i, obs, src.slot as usize)?;
            let w = module.kernel.log_weight(
                value,
                <crate::geometry::Distance as AlgorithmicDistance<f64>>::comparison_kind(
                    &module.distance,
                ),
            )?;
            weights.insert(j as u32, w);
        }
        let max = weights.values().copied().fold(f64::NEG_INFINITY, f64::max);
        let logz = max + weights.values().map(|w| (w - max).exp()).sum::<f64>().ln();
        check(
            batch.row(i).count() == module.count,
            "weighted assignment count mismatch",
        )?;
        for j in batch.row(i) {
            logp += weights
                .get(&j)
                .ok_or_else(|| unavailable("weighted donor outside candidate set"))?
                - logz;
        }
    }
    Ok(logp)
}
fn assignment(
    a: &RunArchive<f64>,
    s: &RecordedStep<f64>,
    module: &DonorModule,
    batch: &CompanionBatch,
    sources: &[SourceRef],
) -> Result<f64> {
    if matches!(module.kernel, Kernel::Uniform) {
        uniform_assignment_log_probability(
            module,
            batch,
            sources,
            s.report.step - 1,
            &s.report.pre_clone_eligible,
        )
    } else {
        weighted_probability(a, s, module, batch, sources)
    }
}
fn gates(s: &RecordedStep<f64>) -> Result<f64> {
    let mut logp = 0.;
    for c in &s.report.clone_plan.choices {
        let p = c
            .probability
            .ok_or_else(|| unavailable("clone gate probability absent"))?;
        check((0. ..=1.).contains(&p), "invalid gate probability")?;
        logp += if c.accepted { p.ln() } else { (-p).ln_1p() };
    }
    Ok(logp)
}
fn revival(s: &RecordedStep<f64>) -> Result<f64> {
    let current = s.report.step - 1;
    let sources = &s.report.clone_plan.sources;
    let n = sources.iter().filter(|p| p.frame == current).count();
    let mut logp = 0.;
    for c in s.report.clone_plan.choices.iter().filter(|c| c.revival) {
        check(n > 0, "revival pool empty")?;
        check(
            c.donors.len() == 1 && sources[c.donors[0].pool_index as usize].frame == current,
            "revival donor is not current",
        )?;
        logp -= (n as f64).ln();
    }
    Ok(logp)
}
fn add(components: &mut Vec<Component>, name: impl Into<String>, x: Result<f64>) {
    let name = name.into();
    let carrier = if name.contains("latent innovation") {
        "standardized_innovation_lebesgue"
    } else if name.starts_with("noise ") {
        "Bxi_lebesgue"
    } else {
        "finite_outcome_counting"
    };
    components.push(Component {
        name,
        carrier: carrier.into(),
        likelihood: Likelihood::from_result(x),
    });
}
/// Density on the retained independent innovation coordinates. This carrier
/// remains valid for singular factor maps and standardized uniform innovations.
fn latent_innovation_log_density(
    noise: &crate::tracking::NoiseSnapshot,
    mask: &[bool],
) -> Result<f64> {
    use crate::noise::InnovationLaw;
    check(
        noise.rows > 0 && mask.len() == noise.rows,
        "latent innovation mask shape",
    )?;
    let raw = noise
        .raw_innovation
        .as_ref()
        .ok_or_else(|| unavailable("latent innovation coordinates absent"))?;
    check(raw.len() % noise.rows == 0, "latent innovation row shape")?;
    let rank = raw.len() / noise.rows;
    check(rank > 0, "empty latent innovation coordinate space")?;
    let mut out = 0.;
    for (i, &active) in mask.iter().enumerate() {
        if !active {
            continue;
        }
        let factor = noise.dense_factor(i)?;
        check(
            factor.len() == noise.dimension * rank,
            "latent factor shape",
        )?;
        for k in 0..rank {
            let mut x = raw[i * rank + k];
            for sh in &noise.applied_source_shifts {
                sh.validate(noise.rows, rank)?;
                if sh.walker == i && sh.coordinate == k {
                    x -= sh.shift;
                }
            }
            check(x.is_finite(), "nonfinite latent innovation")?;
            out += match noise.innovation_law {
                Some(InnovationLaw::Gaussian) => -0.5 * x * x - 0.5 * std::f64::consts::TAU.ln(),
                Some(InnovationLaw::StandardizedUniform) => {
                    check(
                        x.abs() <= 3_f64.sqrt(),
                        "uniform innovation outside its actual support",
                    )?;
                    -(2. * 3_f64.sqrt()).ln()
                }
                None => return Err(unavailable("latent innovation law absent")),
            };
        }
    }
    Ok(out)
}
/// Conditional primitive execution likelihood; initial-state probabilities and survival conditioning are not inferred.
pub fn likelihood(a: &RunArchive<f64>) -> Result<PathLikelihood> {
    a.validate()?;
    check(!a.steps.is_empty(), "path action requires recorded steps")?;
    let builtin = a.providers.get("operators").is_some_and(|x| {
        x == "builtin-operators/v2" || x.starts_with("conditional-fitness-nd/v1/")
    });
    let mut rows = vec![];
    for s in &a.steps {
        let mut c = vec![];
        if builtin {
            add(
                &mut c,
                "distance companion assignment",
                assignment(
                    a,
                    s,
                    &a.gas_config.distance_donors,
                    &s.report.distance_companions,
                    &s.report.distance_sources,
                ),
            );
            add(
                &mut c,
                "cloning companion assignment",
                assignment(
                    a,
                    s,
                    &a.gas_config.cloning_donors,
                    &s.report.cloning_companions,
                    &s.report.clone_plan.sources,
                ),
            );
            add(&mut c, "clone acceptance gates", gates(s));
            add(&mut c, "revival donor choices", revival(s));
        } else {
            add(
                &mut c,
                "selection and cloning",
                Err(unavailable(
                    "custom or unidentified operator joint law is not specified by the archive",
                )),
            );
        }
        if builtin {
            if a.gas_config.clone_transform.jitter.is_some()
                && !s.noise.iter().any(|n| n.stream == Stream::CloneNoise)
            {
                add(
                    &mut c,
                    "clone jitter density",
                    Err(unavailable(
                        "configured clone jitter sample is absent from the archive",
                    )),
                );
            }
            let (stage, substep) = match a.gas_config.kinetic.integrator {
                crate::kinetic::KineticKind::Baoab { .. } => ("A1", 2),
                crate::kinetic::KineticKind::Environment => ("", 0),
                _ => ("post_clone", 0),
            };
            if stage.is_empty() {
                add(
                    &mut c,
                    "environment transition",
                    Err(unavailable(
                        "external environment transition density is not recorded",
                    )),
                );
            } else if s.stages.iter().any(|x| {
                x.stage == stage
                    && x.validity
                        .iter()
                        .any(|v| v.eligible(a.gas_config.include_truncated))
            }) && !s
                .noise
                .iter()
                .any(|n| n.stream == Stream::Kinetic && n.substep == substep)
            {
                add(
                    &mut c,
                    "kinetic innovation density",
                    Err(unavailable(
                        "eligible kinetic update has no recorded innovation sample",
                    )),
                );
            }
        }
        for (index, noise) in s.noise.iter().enumerate() {
            let eligible = if noise.stream == Stream::CloneNoise {
                Some(vec![true; noise.rows])
            } else if noise.stream == Stream::Kinetic && noise.substep == 2 {
                s.stages.iter().find(|x| x.stage == "A1").map(|st| {
                    st.validity
                        .iter()
                        .map(|x| x.eligible(a.gas_config.include_truncated))
                        .collect()
                })
            } else if noise.stream == Stream::Kinetic {
                Some(vec![true; noise.rows])
            } else {
                None
            };
            let density = eligible
                .as_ref()
                .ok_or_else(|| unavailable("noise eligibility stage unavailable"))
                .and_then(|m| noise.conditional_gaussian_log_density(m));
            match density {
                Ok(total) => {
                    let m = eligible.unwrap();
                    let mut quadratic = 0.;
                    let raw = noise.raw_innovation.as_ref().unwrap();
                    for (i, &e) in m.iter().enumerate() {
                        if e {
                            for j in 0..noise.dimension {
                                let mut xi = raw[i * noise.dimension + j];
                                for shift in &noise.applied_source_shifts {
                                    if shift.walker == i && shift.coordinate == j {
                                        xi -= shift.shift;
                                    }
                                }
                                quadratic -= 0.5 * xi * xi;
                            }
                        }
                    }
                    let normalization = -0.5
                        * m.iter().filter(|&&e| e).count() as f64
                        * noise.dimension as f64
                        * std::f64::consts::TAU.ln();
                    add(
                        &mut c,
                        format!("noise {index}: Gaussian quadratic"),
                        Ok(quadratic),
                    );
                    add(
                        &mut c,
                        format!("noise {index}: Gaussian normalization"),
                        Ok(normalization),
                    );
                    add(
                        &mut c,
                        format!("noise {index}: factor log determinant"),
                        Ok(total - quadratic - normalization),
                    );
                }
                Err(e) => {
                    let latent = eligible
                        .as_ref()
                        .ok_or_else(|| unavailable("noise eligibility stage unavailable"))
                        .and_then(|mask| latent_innovation_log_density(noise, mask));
                    match latent {
                        Ok(p) => add(
                            &mut c,
                            format!(
                                "noise {index}: latent innovation density; deterministic factor map"
                            ),
                            Ok(p),
                        ),
                        Err(_) => add(
                            &mut c,
                            format!("noise {index}: conditional Bxi density"),
                            Err(e),
                        ),
                    }
                }
            }
        }
        if !builtin {
            add(
                &mut c,
                "additional provider random outcomes",
                Err(unavailable(
                    "custom provider may use random inputs outside the built-in trace",
                )),
            );
        }
        let sum = c.iter().filter_map(|c| c.likelihood.value()).sum();
        let complete = c.iter().all(|c| c.likelihood.value().is_some());
        rows.push(StepLikelihood {
            epoch: s.epoch,
            step: s.report.step,
            components: c,
            log_density: complete.then_some(sum),
            available_log_density: sum,
        });
    }
    let complete = rows.iter().all(|s| s.log_density.is_some());
    let sum = rows.iter().map(|s| s.available_log_density).sum();
    Ok(PathLikelihood {
        steps: rows,
        complete,
        log_density: complete.then_some(sum),
        available_log_density: sum,
    })
}

pub fn analyze(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    r.validate()?;
    check(
        matches!(r.experiment, 16 | 17 | 21),
        "archive path analysis supports 16, 17, 21",
    )?;
    let report = likelihood(a)?;
    let mut out = ExperimentResult::new(
        r.experiment,
        "Recorded path action and descriptor law",
        "Executed conditional primitive assignments, acceptance gates, revival choices, and Gaussian Bxi samples",
    );
    out.metric(
        "Complete likelihood coverage",
        if report.complete { 1. } else { 0. },
        "boolean",
    )
    .metric(
        "Conditional primitive action",
        report.log_density.map_or(f64::NAN, |x| -x),
        "nats",
    )
    .metric(
        "Available component action",
        -report.available_log_density,
        "nats",
    );
    let mut cumulative = 0.;
    let mut points = vec![];
    let mut by_component = BTreeMap::<String, f64>::new();
    for (i, s) in report.steps.iter().enumerate() {
        cumulative -= s.available_log_density;
        points.push([i as f64, cumulative]);
        for c in &s.components {
            if let Some(x) = c.likelihood.value() {
                *by_component.entry(c.name.clone()).or_default() -= x;
            }
        }
    }
    out.plot(
        "Recorded action components",
        "component index",
        "negative log density",
        vec![Series::line(
            "Available contributions",
            by_component
                .values()
                .enumerate()
                .map(|(i, &v)| [i as f64, v])
                .collect(),
        )],
    );
    out.plot(
        "Conditional action along the retained segment",
        "recorded step index",
        "negative log density",
        vec![Series::line("Available cumulative action", points)],
    );
    let descriptors = a
        .steps
        .iter()
        .map(|s| {
            let field = s.final_population.observations.fields.get("positions");
            let x = field
                .map(|f| f.values().iter().sum::<f64>() / f.values().len().max(1) as f64)
                .unwrap_or(0.)
                .tanh();
            [x, s.report.clones as f64 / s.before.len() as f64]
        })
        .collect::<Vec<_>>();
    let theta = r
        .number("angle", 0.7)
        .clamp(-std::f64::consts::PI, std::f64::consts::PI);
    let z = |t: f64| {
        descriptors.iter().map(|x| (t * x[0]).exp()).sum::<f64>() / descriptors.len() as f64
    };
    if r.experiment == 16 {
        let first = descriptors
            .iter()
            .map(|x| x[0] * (theta * x[0]).exp())
            .sum::<f64>()
            / descriptors.len() as f64;
        let h = 1e-5;
        let fd = (z(theta + h) - z(theta - h)) / (2. * h);
        out.metric("Empirical partition at zero", z(0.), "").metric(
            "Source derivative residual",
            (first - fd).abs(),
            "",
        );
        out.plot(
            "Bounded recorded descriptor generating function",
            "source",
            "empirical Z",
            vec![Series::line(
                "Uniform recorded-step empirical measure",
                (-32..=32)
                    .map(|i| {
                        let t = i as f64 / 16.;
                        [t, z(t)]
                    })
                    .collect(),
            )],
        );
        out.details["generating_function"] = json!({"source":theta,"derivative":first,"finite_difference":fd,"descriptor":"tanh(mean of all recorded terminal position components)","empirical_mass":1.,"weighting":"uniform retained steps; dependent trajectory observations; no trajectory likelihood reweighting"});
    }
    if r.experiment == 21 {
        let bins = r.usize("bins", 4).clamp(2, 16);
        let mut counts = vec![vec![0usize; bins]; bins];
        for d in &descriptors {
            let i = (((d[0] + 1.) * 0.5 * bins as f64) as usize).min(bins - 1);
            let j = ((d[1] * bins as f64) as usize).min(bins - 1);
            counts[i][j] += 1;
        }
        let mut decomposition = vec![];
        let mut error: f64 = 0.;
        let n = descriptors.len() as f64;
        for (i, row) in counts.iter().enumerate() {
            let m = row.iter().sum::<usize>();
            for (j, &c) in row.iter().enumerate() {
                if c > 0 {
                    let joint = c as f64 / n;
                    let marginal = m as f64 / n;
                    let conditional = c as f64 / m as f64;
                    let residual = (-joint.ln() + marginal.ln() + conditional.ln()).abs();
                    error = error.max(residual);
                    decomposition.push(json!({"geometry_bin":i,"fiber_bin":j,"count":c,"joint":joint,"marginal":marginal,"conditional":conditional,"joint_action":-joint.ln(),"marginal_action":-marginal.ln(),"conditional_action":-conditional.ln()}));
                }
            }
        }
        out.metric("Empirical disintegration residual", error, "")
            .metric("Histogram sample count", n, "recorded steps");
        out.plot(
            "Empirical geometry/fiber joint law",
            "flattened bin",
            "empirical probability",
            vec![Series::line(
                "Count / retained steps",
                counts
                    .iter()
                    .flatten()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v as f64 / n])
                    .collect(),
            )],
        );
        out.details["empirical_disintegration"] = json!({"counts":counts,"occupied_bins":decomposition,"geometry_descriptor":"tanh(mean terminal position components)","fiber_descriptor":"accepted clone fraction","sampling_law":"empirical distribution of retained steps; no independence assumed","empty_bins":"zero mass; conditional law absent for empty geometry bins"});
    }
    out.details["likelihood"] =
        serde_json::to_value(report).map_err(|e| GasError::Configuration(e.to_string()))?;
    out.details["component_order"] = json!(by_component.keys().collect::<Vec<_>>());
    out.details["bounded_descriptors"] = json!(descriptors);
    out.note("This is the conditional density of retained sampler outcomes and raw Bxi variables, including sampled unused clone-noise rows. Deterministic copying and boundary maps have no independent density factor. It is not a Lebesgue density of the final state.");
    out.note("Each likelihood component exports its reference carrier. A singular factor or standardized-uniform stage retains the latent innovation density and its deterministic factor map; it has no invented full-dimensional Bxi density.");
    out.note("The initial state and any externally replaced epoch anchors are conditioned inputs. Surviving recorded trajectories use the unconditioned transition factors; no unknown survival-normalization factor is inserted. Unavailable components leave the total unavailable while preserving available contributions.");
    Ok(out)
}
