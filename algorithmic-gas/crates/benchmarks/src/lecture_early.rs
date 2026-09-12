//! Engine-backed measurements for the first four lecture parts.
//! Every observation is taken from a recorded transition, including stage masks.
use crate::{Benchmark, RunConfig};
use algorithmic_gas::{
    AlgorithmicGas, GasError, Precision, Result, RunArchive, TensorBatch,
    boundary::{BoundaryPolicy, BoxDomain},
    donor::SamplingLaw,
    fitness::{PositiveMap, Standardizer},
    geometry::Kernel,
    kinetic::{KineticKind, ViscousForceConfig},
    noise::{FactorValues, Noise, NoiseGeometry},
    physics::partvi::{ExperimentResult, Series},
};
use serde_json::{Value, json};
fn err(s: &str) -> GasError {
    GasError::Configuration(s.into())
}
fn number(p: &Value, k: &str, d: f64) -> f64 {
    p.get(k).and_then(Value::as_f64).unwrap_or(d)
}
fn text<'a>(p: &'a Value, k: &str, d: &'a str) -> &'a str {
    p.get(k).and_then(Value::as_str).unwrap_or(d)
}
fn index(id: &str) -> Result<(usize, usize)> {
    let (p, n) = id.split_once('-').ok_or_else(|| err("lecture ID"))?;
    let p = match p {
        "I" => 1,
        "II" => 2,
        "III" => 3,
        "IV" => 4,
        _ => return Err(err("early lecture part")),
    };
    let n = n.parse::<usize>().map_err(|_| err("lecture number"))?;
    if n == 0 || n > [10, 8, 8, 16][p - 1] {
        return Err(err("lecture number outside part"));
    }
    Ok((p, n))
}
fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}
fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    mean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>())
}
fn coordinates(
    p: &algorithmic_gas::Population<f64>,
    name: &str,
    include: bool,
) -> Result<Vec<f64>> {
    let f = p.observations.field(name)?;
    Ok(f.values()
        .chunks(f.width())
        .zip(p.eligible(include))
        .filter(|(_, a)| *a)
        .map(|(r, _)| r[0])
        .collect())
}
fn dt(c: &RunConfig) -> f64 {
    match c.gas.kinetic.integrator {
        KineticKind::Baoab { dt, .. } | KineticKind::Brownian { dt, .. } => dt,
        _ => 1.,
    }
}
pub fn config(id: &str, p: &Value, seed: u64) -> Result<RunConfig> {
    let (part, n) = index(id)?;
    let mut c = RunConfig {
        benchmark: Benchmark::Quadratic,
        walkers: number(
            p,
            "walkers",
            number(
                p,
                "N",
                if matches!(id, "I-04" | "IV-07") {
                    16.
                } else {
                    64.
                },
            ),
        ) as usize,
        dimensions: 2,
        ..Default::default()
    };
    c.gas.seed = seed;
    c.gas.precision = Precision::F64;
    c.gas.boundary = BoundaryPolicy::Unbounded;
    c.gas.fitness.reward_exponent = number(p, "alpha", number(p, "selection", 1.));
    c.gas.fitness.diversity_exponent = number(p, "beta", 1.);
    let h = number(p, "dt", number(p, "h", 0.04));
    let gamma = number(p, "gamma", 1.);
    let temperature = number(p, "temperature", number(p, "theta", 0.5));
    c.gas.kinetic.integrator = KineticKind::Baoab {
        positions: "positions".into(),
        velocities: "velocities".into(),
        dt: h,
        friction: gamma,
    };
    c.gas.kinetic.noise.geometry = NoiseGeometry::Isotropic {
        scale: FactorValues::Constant {
            values: vec![(2. * gamma * temperature).sqrt()],
        },
    };
    let width = number(p, "width", number(p, "rho", 1.));
    c.gas.distance_donors.kernel = Kernel::Gaussian { width };
    c.gas.cloning_donors.kernel = Kernel::Gaussian {
        width: number(p, "cloneWidth", width),
    };
    c.gas.distance_donors.law = match text(p, "law", "independent") {
        "uniform" => {
            c.gas.distance_donors.kernel = Kernel::Uniform;
            SamplingLaw::Independent
        }
        "mutual" | "matching" => {
            c.gas.distance_donors.kernel = Kernel::Uniform;
            SamplingLaw::FisherYates
        }
        "greedy" | "shuffled_greedy" => SamplingLaw::GaussianGreedy,
        _ => SamplingLaw::Independent,
    };
    c.gas.distance_donors.count = number(p, "count", 1.) as usize;
    c.gas.clone_decision.saturation = number(p, "saturation", number(p, "cap", 1.));
    c.gas.fitness.distance_floor = number(p, "delta", 0.01);
    let floor = number(p, "floor", 0.01);
    c.gas.fitness.reward_map = PositiveMap::Logistic {
        amplitude: 2.,
        floor,
    };
    c.gas.fitness.diversity_map = c.gas.fitness.reward_map.clone();
    let standardizer = if text(p, "stats", "global") == "local" || (part == 4 && matches!(n, 5..=8))
    {
        Standardizer::Local {
            sigma_min: number(p, "sigma", 0.1),
            distance: c.gas.distance_donors.distance.clone(),
            kernel: Kernel::Gaussian {
                width: number(p, "rho", 0.7),
            },
            include_self: text(p, "self", "include") != "exclude",
        }
    } else {
        Standardizer::Global {
            sigma_min: number(p, "sigma", 0.01),
        }
    };
    c.gas.fitness.reward_standardizer = standardizer.clone();
    c.gas.fitness.diversity_standardizer = standardizer;
    let jitter = number(p, "jitter", 0.);
    if jitter > 0. {
        c.gas.clone_transform.position_field = Some("positions".into());
        c.gas.clone_transform.jitter = Some(Noise::default());
        c.gas.clone_transform.jitter_amplitude = jitter;
    }
    if text(p, "benchmark", "") == "rastrigin"
        || text(p, "landscape", "") == "multiwell"
        || id == "IV-09"
    {
        c.benchmark = Benchmark::Rastrigin;
    }
    if text(p, "landscape", "") == "shifted" {
        c.reward_shift = vec![2., 0.];
        c.potential = Some(Benchmark::Quadratic);
    }
    if matches!(
        id,
        "I-05" | "II-05" | "III-04" | "IV-01" | "IV-10" | "IV-13" | "IV-14"
    ) {
        c.gas.fitness.reward_exponent = 0.;
        c.gas.fitness.diversity_exponent = 0.;
    }
    if matches!(
        id,
        "I-04" | "I-08" | "II-06" | "II-08" | "III-01" | "III-06" | "IV-03"
    ) || text(p, "card", "") == "survival"
        || (id == "IV-04" && text(p, "boundary", "unbounded") == "absorbing")
    {
        let b = number(p, "box", number(p, "length", number(p, "boundary", 1.1)));
        let domain = BoxDomain {
            lower: vec![-b; 2],
            upper: vec![b; 2],
        };
        c.gas.boundary = if text(p, "boundary", "") == "periodic" {
            BoundaryPolicy::PeriodicBox {
                field: "positions".into(),
                domain,
            }
        } else {
            BoundaryPolicy::AbsorbingBox {
                field: "positions".into(),
                domain,
            }
        };
    }
    if id == "I-09" {
        c.gas.cloning_donors.law = SamplingLaw::FisherYates;
        c.gas.cloning_donors.kernel = Kernel::Uniform;
        c.gas.clone_transform.velocity_field = Some("velocities".into());
        c.gas.clone_transform.restitution = Some(number(p, "restitution", 0.5));
    }
    if id == "IV-12" {
        c.gas.qft.viscosity = Some(ViscousForceConfig {
            coefficient: number(p, "viscosity", 1.),
            bandwidth: width,
            row_normalized: true,
        });
    }
    if matches!(id, "I-10" | "IV-11") {
        c.physics_metric = Some(crate::physics_metric::PhysicsMetricConfig {
            epsilon: number(p, "shift", 0.5),
            temperature,
            curvature: false,
            ..Default::default()
        });
    }
    if id == "III-05" {
        c.gas.fitness.reward_exponent = number(p, "ratio", 1.);
        c.gas.fitness.diversity_exponent = 1.;
    }
    if id == "III-06" {
        c.gas.kinetic.integrator = KineticKind::Brownian {
            field: "positions".into(),
            amplitude: (2. * number(p, "diffusivity", 0.5)).sqrt(),
            dt: h,
        };
        c.gas.kinetic.noise = Noise::default();
    }

    if let BoundaryPolicy::PeriodicBox { domain, .. } = &c.gas.boundary {
        for module in [&mut c.gas.distance_donors, &mut c.gas.cloning_donors] {
            if let algorithmic_gas::geometry::Distance::Euclidean { periodic, .. } =
                &mut module.distance
            {
                *periodic = Some(domain.clone());
            }
        }
    }
    if text(p, "benchmark", "") == "sphere" {
        c.benchmark = Benchmark::Sphere;
    }
    c.validate()?;
    Ok(c)
}
pub fn configs(id: &str, p: &Value, seed: u64) -> Result<Vec<RunConfig>> {
    let base = config(id, p, seed)?;
    let mut out = vec![base.clone()];
    if id == "IV-14" {
        out.clear();
        for h in [0.2, 0.1, 0.05] {
            let mut c = base.clone();
            if let KineticKind::Baoab { dt, .. } = &mut c.gas.kinetic.integrator {
                *dt = h;
            }
            out.push(c);
        }
    } else if matches!(id, "II-02" | "III-03" | "III-07" | "IV-16" | "I-06") {
        let replicas = number(p, "replicas", if id == "I-06" { 4. } else { 8. }) as usize;
        out.clear();
        for i in 0..replicas {
            let mut c = base.clone();
            c.gas.seed = seed.wrapping_add((i as u64 + 1) * 104729);
            out.push(c);
        }
    } else if matches!(id, "II-03" | "II-04") {
        let mut c = base;
        c.gas.seed = seed;
        let shift = number(p, "translation", 1.);
        c.initial_lower += shift;
        c.initial_upper += shift;
        out.push(c);
    }
    Ok(out)
}
pub fn steps(id: &str, p: &Value, c: &RunConfig) -> usize {
    if matches!(id, "IV-13" | "IV-14") {
        (number(p, "T", 5.) / dt(c)).ceil() as usize
    } else {
        number(p, "steps", number(p, "updates", 64.)) as usize
    }
}
pub async fn initialize(id: &str, p: &Value, gas: &mut AlgorithmicGas<f64>) -> Result<()> {
    index(id)?;
    let mut population = gas.population().clone();
    let n = population.len();
    let d = population.observations.field("positions")?.width();
    let mut x = population
        .observations
        .field("positions")?
        .values()
        .to_vec();
    let mut v = population
        .observations
        .fields
        .get("velocities")
        .map(|v| v.values().to_vec())
        .unwrap_or_else(|| vec![0.; n * d]);
    if id == "II-02" {
        for i in 0..n {
            x[i * d] = (i as f64 / n as f64 - 0.5) * 2.;
            x[i * d + 1] = (i as f64 * 2.399963229728653).sin() * 0.25;
            v[i * d] = 0.;
            v[i * d + 1] = 0.;
        }
    }
    if matches!(id, "I-04" | "III-01") {
        let survivors = number(p, "survivors", 4.) as usize;
        for (i, a) in population.validity.iter_mut().enumerate() {
            a.terminated = i >= survivors;
        }
    }
    if id == "I-03" {
        x[0] = number(p, "outlier", 1.);
        population.validity[0].terminated = text(p, "exclude", "yes") == "no";
    }
    if id == "II-01" {
        let fraction = number(p, "fraction", 0.15);
        for i in 0..n {
            if (i as f64) < n as f64 * fraction {
                x[i * d] += number(p, "separation", 2.);
            }
        }
    }
    if matches!(id, "IV-01" | "IV-03" | "II-06") {
        let shift = number(
            p,
            "displacement",
            number(p, "center", number(p, "start", 0.8)),
        );
        for row in x.chunks_mut(d) {
            row[0] += shift;
        }
    }
    if id == "IV-06" && text(p, "geometry", "clustered") == "clustered" {
        for a in &mut x {
            *a *= 0.1;
        }
    }
    if id == "IV-12" {
        for i in 0..n {
            x[i * d] += if i < n / 2 {
                -number(p, "gap", 1.5)
            } else {
                number(p, "gap", 1.5)
            };
            v[i * d] = if i < n / 2 { 1. } else { -1. };
        }
    }
    if id == "I-09" {
        for i in 0..n {
            v[i * d] = if i % 2 == 0 {
                number(p, "vx", 1.5)
            } else {
                -number(p, "vx", 1.5)
            };
            v[i * d + 1] = number(p, "vy", 0.5);
        }
    }
    if id == "III-07" || (id == "III-03" && text(p, "initial", "") == "shared") {
        let rho = number(p, "correlation", 1.);
        let shared = x[0];
        for row in x.chunks_mut(d) {
            row[0] = (1. - rho).sqrt() * row[0] + rho.sqrt() * shared;
        }
    }
    population
        .observations
        .fields
        .insert("positions".into(), TensorBatch::new(n, vec![d], x)?);
    population
        .observations
        .fields
        .insert("velocities".into(), TensorBatch::new(n, vec![d], v)?);
    gas.replace_population(population).await
}
fn series(name: &str, values: impl IntoIterator<Item = [f64; 2]>) -> Series {
    Series::line(name, values.into_iter().collect())
}
fn histogram(values: &[f64], lo: f64, hi: f64, bins: usize) -> Vec<[f64; 2]> {
    let w = (hi - lo) / bins as f64;
    let mut counts = vec![0.; bins];
    for &x in values {
        if x >= lo && x <= hi {
            let j = (((x - lo) / w) as usize).min(bins - 1);
            counts[j] += 1.;
        }
    }
    counts
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            [
                lo + (i as f64 + 0.5) * w,
                if values.is_empty() {
                    0.
                } else {
                    n / (values.len() as f64 * w)
                },
            ]
        })
        .collect()
}
fn entropy(v: &[f64], lo: f64, hi: f64, bins: usize) -> f64 {
    let w = (hi - lo) / bins as f64;
    histogram(v, lo, hi, bins)
        .iter()
        .filter(|p| p[1] > 0.)
        .map(|p| -p[1] * w * (p[1] * w).ln())
        .sum()
}

fn conditional_jet(
    id: &str,
    p: &Value,
    a: &RunArchive<f64>,
    offset: f64,
    order: usize,
) -> Result<algorithmic_gas::physics::jet::Jet<f64>> {
    use algorithmic_gas::physics::{
        fitness::{companion_distance, local_log_weights, pipeline_from_measurement_jets},
        jet::JetSpace,
    };
    let last = a.steps.last().ok_or_else(|| err("missing transition"))?;
    let population = &last.before;
    let d = population.observations.field("positions")?.width();
    let points = population
        .observations
        .field("positions")?
        .values()
        .chunks(d)
        .map(|x| x.to_vec())
        .collect::<Vec<_>>();
    let alive = &last.report.pre_clone_eligible;
    let target = alive
        .iter()
        .position(|&v| v)
        .ok_or_else(|| err("conditional derivative requires eligible target"))?;
    let space = JetSpace::new(d, order)?;
    let query = points[target]
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if order == 0 {
                Ok(space.constant(x + if i == 0 { offset } else { 0. }))
            } else {
                space.variable(x + if i == 0 { offset } else { 0. }, i)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let objective = config(id, p, a.gas_config.seed)?
        .benchmark
        .physics_objective(d);
    let mut rewards = last
        .report
        .pre_clone_rewards
        .raw
        .iter()
        .map(|&x| space.constant(x))
        .collect::<Vec<_>>();
    let mut distances = last
        .report
        .pre_clone_fitness
        .separation
        .iter()
        .map(|&x| space.constant(x))
        .collect::<Vec<_>>();
    rewards[target] = objective.evaluate(&query)?;
    let companions = &last.report.distance_companions;
    let mut ds = vec![];
    for k in 0..companions.count {
        let i = target * companions.count + k;
        if companions.valid[i] {
            let source = &last.report.distance_sources[companions.indices[i] as usize];
            let source_point = &points[source.slot as usize];
            ds.push(companion_distance(
                &query,
                source_point,
                0.,
                a.gas_config.fitness.distance_floor,
            )?);
        }
    }
    if !ds.is_empty() {
        let mut sum = space.constant(0.);
        for distance in &ds {
            sum = sum.add(distance);
        }
        distances[target] = sum.scale(1. / ds.len() as f64);
    }
    let logs = |standardizer: &Standardizer| {
        local_log_weights(&query, &points, &[], alive, target, standardizer)
    };
    pipeline_from_measurement_jets(
        &rewards,
        &distances,
        alive,
        target,
        &a.gas_config.fitness,
        logs(&a.gas_config.fitness.reward_standardizer)?.as_deref(),
        logs(&a.gas_config.fitness.diversity_standardizer)?.as_deref(),
    )
}
fn derivative_experiment(
    id: &str,
    p: &Value,
    a: &RunArchive<f64>,
    r: &mut ExperimentResult,
) -> Result<()> {
    let order = (number(p, "order", 6.) as usize).min(12);
    let jet = conditional_jet(id, p, a, 0., order)?;
    let radius = number(p, "radius", 0.1);
    let mut measured = vec![];
    let mut predicted = vec![];
    let d = jet.space.dimension;
    for i in 0..41 {
        let delta = radius * (2. * i as f64 / 40. - 1.);
        let actual = conditional_jet(id, p, a, delta, 0)?.value();
        let approximation = jet
            .space
            .indices
            .iter()
            .zip(&jet.coefficients)
            .filter(|(index, _)| index[1..].iter().all(|&x| x == 0))
            .map(|(index, &coefficient)| coefficient * delta.powi(index[0] as i32))
            .sum::<f64>();
        measured.push([delta, actual]);
        predicted.push([delta, approximation]);
    }
    let epsilon = 1e-5;
    let fd = (conditional_jet(id, p, a, epsilon, 0)?.value()
        - conditional_jet(id, p, a, -epsilon, 0)?.value())
        / (2. * epsilon);
    let mut unit = vec![0; d];
    unit[0] = 1;
    let derivative = jet
        .space
        .indices
        .iter()
        .position(|x| *x == unit)
        .map(|i| jet.coefficients[i])
        .unwrap_or(0.);
    r.plot(
        "Recorded conditional fitness field and Taylor prediction",
        "target displacement",
        "fitness",
        vec![
            series("Recomputed from recorded measurements", measured),
            series("Taylor polynomial", predicted),
        ],
    );
    r.metric("Analytic first derivative", derivative, "")
        .metric("Finite difference first derivative", fd, "")
        .metric("Derivative residual", derivative - fd, "");
    r.details["jet_multi_indices"] = json!(jet.space.indices);
    r.details["jet_coefficients"] = json!(jet.coefficients);
    r.note("The selected donor realization and other rows' measured rewards and separations are held fixed; normalization and local weights are differentiated.");
    Ok(())
}

/// Analysis never draws random numbers or advances the physical state.
pub fn analyze(id: &str, p: &Value, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let (part, n) = index(id)?;
    a.validate()?;
    if a.steps.is_empty() {
        let (step, population) = a.terminal();
        if population
            .eligible(a.gas_config.include_truncated)
            .iter()
            .any(|&x| x)
        {
            return Err(err("experiment requires an executed transition"));
        }
        let mut result = ExperimentResult::new(n as u32, id, "Extinct Euclidean Gas population");
        result.metric("Eligible walkers", 0., "");
        result.plot(
            "Recorded eligibility at termination",
            "recorded step",
            "eligible walkers",
            vec![series("Eligible population", [[step as f64, 0.]])],
        );
        result.details = json!({"calculation_origin":"executed_rust_archive","lecture_id":id,"archive_steps":0,"terminal_outcome":"extinction","recorded_terminal_step":step,"seed":a.gas_config.seed});
        result.note("The recorded population has no eligible donor; the attempted update terminates before a transition is committed.");
        return Ok(result);
    }
    let last = a.steps.last().unwrap();
    let include = a.gas_config.include_truncated;
    let count = last.final_population.len();
    let mut r = ExperimentResult::new(n as u32, id, "Recorded Euclidean Gas transitions");
    let h = match a.gas_config.kinetic.integrator {
        KineticKind::Baoab { dt, .. } | KineticKind::Brownian { dt, .. } => dt,
        _ => 1.,
    };
    let x = coordinates(&last.final_population, "positions", include)?;
    let v = coordinates(&last.final_population, "velocities", include)?;
    let trace = |name: &str, field: &str, stat: fn(&[f64]) -> f64| -> Result<Series> {
        Ok(series(
            name,
            a.steps
                .iter()
                .map(|s| {
                    Ok([
                        s.report.step as f64 * h,
                        stat(&coordinates(&s.final_population, field, include)?),
                    ])
                })
                .collect::<Result<Vec<_>>>()?,
        ))
    };
    r.metric("Recorded updates", a.steps.len() as f64, "")
        .metric("Eligible walkers", x.len() as f64, "");
    r.details = json!({"calculation_origin":"executed_rust_archive","lecture_id":id,"seed":a.gas_config.seed,"archive_steps":a.steps.len(),"gas_config":a.gas_config,"sampling_unit":"recorded interacting population; no walker-iid uncertainty claimed","positions":last.final_population.observations.field("positions")?.values(),"dimensions":last.final_population.observations.field("positions")?.width(),"observables":[]});
    match (part, n) {
        (1, 1 | 5 | 8) | (2, 2) | (4, 12) => {
            let walker = (number(p, "walker", 0.) as usize).min(count - 1);
            let d = last
                .final_population
                .observations
                .field("positions")?
                .width();
            let mut ps = vec![];
            let mut vs = vec![];
            let mut stage_names = vec![];
            for (i, s) in last.stages.iter().enumerate() {
                stage_names.push(&s.stage);
                if let Some(f) = s.fields.get("positions") {
                    ps.push([i as f64, f.values[walker * d]]);
                }
                if let Some(f) = s.fields.get("velocities") {
                    vs.push([i as f64, f.values[walker * d]]);
                }
            }
            r.plot(
                "Executed operator stages",
                "stage index",
                "selected coordinate",
                vec![series("Position", ps), series("Velocity", vs)],
            );
            r.details["stage_names"] = json!(stage_names);
            r.details["mechanical_budgets"] =
                serde_json::to_value(last.mechanical_budgets("velocities", 1., include)?)
                    .map_err(|e| err(&e.to_string()))?;
        }
        (1, 2 | 7) | (2, 1) | (4, 8 | 15) => {
            let fitness = &last.report.pre_clone_fitness;
            let probabilities: Vec<_> = last
                .report
                .clone_plan
                .choices
                .iter()
                .map(|c| c.probability)
                .collect();
            let pred = series(
                "Conditional acceptance probability",
                probabilities
                    .iter()
                    .enumerate()
                    .filter_map(|(i, q)| q.map(|q| [i as f64, q])),
            );
            let obs = series(
                "Executed acceptance",
                last.report
                    .clone_plan
                    .choices
                    .iter()
                    .enumerate()
                    .map(|(i, c)| [i as f64, if c.accepted { 1. } else { 0. }]),
            );
            r.plot(
                "Conditional cloning law and executed draws",
                "recipient slot",
                "probability / event",
                vec![pred, obs],
            );
            r.plot(
                "Fitness entering the clone decision",
                "recipient slot",
                "fitness",
                vec![series(
                    "Fitness",
                    fitness
                        .fitness
                        .iter()
                        .enumerate()
                        .map(|(i, &f)| [i as f64, f]),
                )],
            );
            r.details["distance_sources"] = json!(last.report.distance_sources);
            r.details["distance_companions"] = json!(last.report.distance_companions);
            r.details["clone_plan"] = json!(last.report.clone_plan);
            let residual: f64 = a
                .steps
                .iter()
                .flat_map(|s| &s.report.clone_plan.choices)
                .filter_map(|c| {
                    c.probability
                        .map(|p| (if c.accepted { 1. } else { 0. }) - p)
                })
                .sum();
            r.metric("Acceptance martingale residual sum", residual, "");
        }
        (4, 5 | 7) => {
            derivative_experiment(id, p, a, &mut r)?;
        }
        (1, 3) | (3, 5) | (4, 6) => {
            let f = &last.report.pre_clone_fitness;
            r.plot(
                "Executed fitness pipeline",
                "slot",
                "value",
                vec![
                    series(
                        "Oriented reward",
                        f.oriented_reward
                            .iter()
                            .enumerate()
                            .map(|(i, &x)| [i as f64, x]),
                    ),
                    series(
                        "Separation",
                        f.separation.iter().enumerate().map(|(i, &x)| [i as f64, x]),
                    ),
                    series(
                        "Reward z",
                        f.reward_z.iter().enumerate().map(|(i, &x)| [i as f64, x]),
                    ),
                    series(
                        "Fitness",
                        f.fitness.iter().enumerate().map(|(i, &x)| [i as f64, x]),
                    ),
                ],
            );
            r.details["reward_statistics"] = json!(f.reward_stats);
            r.details["diversity_statistics"] = json!(f.diversity_stats);
            r.metric("Fitness variance on population", variance(&f.fitness), "");
            if !last.field_evaluations.is_empty() {
                r.details["executed_field_evaluations"] = json!(last.field_evaluations);
            }
        }
        (1, 4) | (2, 6 | 8) | (3, 1 | 6) | (4, 3) => {
            r.plot(
                "Population eligibility ledger",
                "time",
                "walkers",
                vec![
                    series(
                        "Before clone",
                        a.steps.iter().map(|s| {
                            [
                                s.report.step as f64 * h,
                                s.report.pre_clone_eligible.iter().filter(|&&x| x).count() as f64,
                            ]
                        }),
                    ),
                    series(
                        "Revived",
                        a.steps
                            .iter()
                            .map(|s| [s.report.step as f64 * h, s.report.revivals as f64]),
                    ),
                    series(
                        "After full step",
                        a.steps
                            .iter()
                            .map(|s| [s.report.step as f64 * h, s.report.eligible as f64]),
                    ),
                ],
            );
            r.plot(
                "Conditional surviving shape",
                "position coordinate",
                "density",
                vec![series(
                    "Eligible final population",
                    histogram(&x, -4., 4., 48),
                )],
            );
            r.metric("Surviving mass", x.len() as f64 / count as f64, "");
        }
        (1, 9) | (3, 2) => {
            let mut before = vec![];
            let mut after = vec![];
            for s in &a.steps {
                for b in s.mechanical_budgets("velocities", 1., include)? {
                    if b.to.to_lowercase().contains("clone") {
                        before.push([s.report.step as f64 * h, b.energy_before]);
                        after.push([s.report.step as f64 * h, b.energy_after]);
                    }
                }
            }
            r.plot(
                "Executed clone energy change",
                "time",
                "kinetic energy",
                vec![series("Before", before), series("After", after)],
            );
            r.plot(
                "Executed event counts",
                "time",
                "events",
                vec![
                    series(
                        "Clones",
                        a.steps
                            .iter()
                            .map(|s| [s.report.step as f64 * h, s.report.clones as f64]),
                    ),
                    series(
                        "Revivals",
                        a.steps
                            .iter()
                            .map(|s| [s.report.step as f64 * h, s.report.revivals as f64]),
                    ),
                ],
            );
            r.details["clone_plan"] = json!(last.report.clone_plan);
        }
        (1, 10) | (4, 11) => {
            let mut measured = vec![];
            let mut predicted = vec![];
            for s in &a.steps {
                for noise in &s.noise {
                    if noise.stream == algorithmic_gas::random::Stream::Kinetic
                        && noise.substep == 2
                        && noise.geometry.is_some()
                    {
                        let d = noise.dimension;
                        let eligible = s
                            .stages
                            .iter()
                            .find(|f| f.stage == "A1")
                            .map(|f| {
                                f.validity
                                    .iter()
                                    .map(|v| v.eligible(include))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        for j in 0..d {
                            let mut obs = vec![];
                            let mut expected = vec![];
                            for i in 0..noise.rows {
                                if eligible.get(i) == Some(&true) {
                                    obs.push(noise.sample[i * d + j].powi(2));
                                    let b = noise.dense_factor(i)?;
                                    expected
                                        .push(b[j * d..(j + 1) * d].iter().map(|x| x * x).sum());
                                }
                            }
                            measured.push([s.report.step as f64 + (j as f64) * 0.1, mean(&obs)]);
                            predicted
                                .push([s.report.step as f64 + (j as f64) * 0.1, mean(&expected)]);
                        }
                    }
                }
            }
            r.plot(
                "Executed anisotropic innovation second moments",
                "step + coordinate offset",
                "second moment",
                vec![
                    series("Measured", measured),
                    series("Conditional prediction", predicted),
                ],
            );
            r.details["field_evaluations"] = json!(last.field_evaluations);
        }
        (2, 5) | (4, 1 | 10 | 13 | 14) => {
            let mut obs = vec![];
            let mut pred = vec![];
            if let KineticKind::Baoab { dt, friction, .. } = a.gas_config.kinetic.integrator {
                for s in &a.steps {
                    let m = s.thermostat_moments("velocities", 1., dt, friction, include)?;
                    obs.push([s.report.step as f64 * h, m.measured_energy_change]);
                    pred.push([s.report.step as f64 * h, m.predicted_energy_change]);
                }
            }
            r.plot(
                "Thermostat conditional energy identity",
                "time",
                "energy increment",
                vec![
                    series("Executed increment", obs),
                    series("Conditional expectation", pred),
                ],
            );
            r.plot(
                "Velocity law relaxation",
                "time",
                "observable",
                vec![
                    trace("Mean velocity", "velocities", mean)?,
                    trace("Velocity variance", "velocities", variance)?,
                    trace("Mean cos(v)", "velocities", |v| {
                        mean(&v.iter().map(|x| x.cos()).collect::<Vec<_>>())
                    })?,
                ],
            );
        }
        (2, 3 | 4) => {
            let initial = coordinates(&a.steps[0].before, "positions", include)?;
            let mut s = initial.clone();
            let mut t = x.clone();
            s.sort_by(f64::total_cmp);
            t.sort_by(f64::total_cmp);
            if s.len() == t.len() && !s.is_empty() {
                let w = mean(
                    &s.iter()
                        .zip(&t)
                        .map(|(a, b)| (a - b).powi(2))
                        .collect::<Vec<_>>(),
                );
                r.metric("Exact one-coordinate W2 squared", w, "");
                r.metric(
                    "Independent coupling cost",
                    variance(&initial) + variance(&x) + (mean(&initial) - mean(&x)).powi(2),
                    "",
                );
            }
            r.plot(
                "Measured transport quantiles",
                "quantile index",
                "position",
                vec![
                    series("Initial", s.iter().enumerate().map(|(i, &v)| [i as f64, v])),
                    series("Final", t.iter().enumerate().map(|(i, &v)| [i as f64, v])),
                ],
            );
            r.note("Sorted matching is exact for this one-coordinate empirical marginal; it does not assert full phase-space optimality.");
        }
        (2, 7) | (3, 4) => {
            let squared = |x: &[f64]| mean(&x.iter().map(|x| x * x).collect::<Vec<_>>());
            let points = a
                .steps
                .iter()
                .map(|s| {
                    let before = coordinates(&s.before, "positions", include)?;
                    let after = coordinates(&s.final_population, "positions", include)?;
                    Ok([
                        s.report.step as f64 * h,
                        (squared(&after) - squared(&before)) / h,
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            r.plot(
                "Finite-step generator acting on x squared",
                "time",
                "generator value",
                vec![series("Measured increment / h", points)],
            );
            r.plot(
                "Position moments",
                "time",
                "moment",
                vec![
                    trace("Mean", "positions", mean)?,
                    trace("Second moment", "positions", squared)?,
                ],
            );
        }
        (3, 3 | 7 | 8) => {
            r.plot(
                "Collective fluctuations in recorded populations",
                "time",
                "statistic",
                vec![
                    trace("Population mean", "positions", mean)?,
                    trace("Within-population variance", "positions", variance)?,
                ],
            );
            let mut collision = vec![];
            for s in &a.steps {
                let batch = &s.report.distance_companions;
                let labels: Vec<_> = batch
                    .indices
                    .iter()
                    .zip(&batch.valid)
                    .filter(|(_, v)| **v)
                    .map(|(&i, _)| i)
                    .take(number(p, "tuple", batch.rows as f64) as usize)
                    .collect();
                let pairs = labels.len().saturating_mul(labels.len().saturating_sub(1)) / 2;
                let mut repeats = 0;
                for i in 0..labels.len() {
                    for j in 0..i {
                        if labels[i] == labels[j] {
                            repeats += 1;
                        }
                    }
                }
                if pairs > 0 {
                    collision.push([s.report.step as f64 * h, repeats as f64 / pairs as f64]);
                }
            }
            r.plot(
                "Repeated actual donor labels",
                "time",
                "pair collision fraction",
                vec![series("Executed companion draws", collision)],
            );
            r.note("Interacting walkers are not treated as independent replicate runs.");
        }
        (4, 2 | 4 | 9 | 16) => {
            let window = number(p, "window", 4.);
            r.plot(
                "Empirical density from executed walkers",
                "position coordinate",
                "density",
                vec![
                    series(
                        "Initial",
                        histogram(
                            &coordinates(&a.steps[0].before, "positions", include)?,
                            -window,
                            window,
                            48,
                        ),
                    ),
                    series("Final", histogram(&x, -window, window, 48)),
                ],
            );
            let points = a
                .steps
                .iter()
                .map(|s| {
                    Ok([
                        s.report.step as f64 * h,
                        entropy(
                            &coordinates(&s.final_population, "positions", include)?,
                            -window,
                            window,
                            48,
                        ),
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            r.plot(
                "Binned position entropy in the declared window",
                "time",
                "discrete entropy",
                vec![series("Measured", points)],
            );
            r.details["density_window"] = json!([-window, window]);
            r.details["density_bins"] = json!(48);
            r.metric(
                "Mass inside density window",
                x.iter().filter(|&&x| x >= -window && x <= window).count() as f64 / count as f64,
                "",
            );
        }
        (1, 6) => {
            r.plot(
                "Reward and mechanical motion",
                "time",
                "population statistic",
                vec![
                    trace("Position mean", "positions", mean)?,
                    trace("Velocity mean", "velocities", mean)?,
                    series(
                        "Mean reward",
                        a.steps
                            .iter()
                            .map(|s| [s.report.step as f64 * h, mean(&s.report.final_rewards.raw)]),
                    ),
                ],
            );
        }
        _ => return Err(err("unimplemented early lecture measurement")),
    }
    if matches!(id, "II-05" | "IV-01" | "IV-10" | "IV-13" | "IV-14") {
        let prediction = harmonic_prediction(a)?;
        r.plot(
            "Bounded velocity observable and exact finite-step expectation",
            "physical time",
            "mean cos(v)",
            vec![
                trace("Measured cos(v)", "velocities", |v| {
                    mean(&v.iter().map(|x| x.cos()).collect::<Vec<_>>())
                })?,
                series(
                    "Conditional expectation from recorded initial population",
                    prediction.iter().map(|p| [p[0], p[1]]),
                ),
            ],
        );
        r.details["harmonic_prediction"] = json!(prediction);
        r.note("The Gaussian convolution prediction propagates the recorded initial population through the exact configured BAOAB matrix; selection is disabled by constant fitness for this component experiment.");
    }

    if matches!(id, "I-02" | "I-07" | "III-08" | "IV-15")
        && a.gas_config.distance_donors.law == SamplingLaw::Independent
    {
        let rows = donor_rows(a)?;
        let selected = number(p, "recipient", 0.) as usize;
        if let Some(row) = rows.get(selected) {
            r.plot(
                "Conditional donor distribution on the recorded population",
                "source slot",
                "selection probability",
                vec![series(
                    "Exact normalized kernel",
                    row.iter().enumerate().map(|(j, &p)| [j as f64, p]),
                )],
            );
        }
        r.details["conditional_donor_rows"] = json!(rows);
        if id == "III-08" {
            let k = (number(p, "tuple", 8.) as usize).min(rows.len());
            let mut collision = 0.;
            for i in 0..k {
                for j in 0..i {
                    collision += rows[i]
                        .iter()
                        .zip(&rows[j])
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                }
            }
            if k > 1 {
                r.metric(
                    "Conditional donor-pair collision probability",
                    collision / (k * (k - 1) / 2) as f64,
                    "",
                );
            }
        }
    }
    if id == "IV-06" {
        let counts = local_effective_counts(a)?;
        r.plot(
            "Effective local normalization sample size",
            "query slot",
            "effective neighbors",
            vec![series(
                "Inverse normalized squared weight",
                counts.iter().enumerate().map(|(i, &x)| [i as f64, x]),
            )],
        );
        r.metric("Mean effective neighbors", mean(&counts), "");
        r.details["effective_neighbor_counts"] = json!(counts);
    }
    if id == "III-02" {
        r.plot(
            "Conditional expected and actual copy counts",
            "physical time",
            "copies per update",
            vec![
                series(
                    "Executed copies",
                    a.steps.iter().map(|s| {
                        [
                            s.report.step as f64 * h,
                            s.report
                                .clone_plan
                                .choices
                                .iter()
                                .filter(|c| c.accepted && !c.revival)
                                .count() as f64,
                        ]
                    }),
                ),
                series(
                    "Sum of conditional probabilities",
                    a.steps.iter().map(|s| {
                        [
                            s.report.step as f64 * h,
                            s.report
                                .clone_plan
                                .choices
                                .iter()
                                .filter(|c| !c.revival)
                                .filter_map(|c| c.probability)
                                .sum::<f64>(),
                        ]
                    }),
                ),
            ],
        );
    }
    r.details["observables"] = json!(r.plots.iter().map(|p| &p.title).collect::<Vec<_>>());
    r.metric("Final mean position", mean(&x), "")
        .metric("Final mean velocity", mean(&v), "");
    Ok(r)
}

fn donor_rows(a: &RunArchive<f64>) -> Result<Vec<Vec<f64>>> {
    use algorithmic_gas::geometry::{AlgorithmicDistance, InteractionKernel};
    let step = a
        .steps
        .last()
        .ok_or_else(|| err("missing recorded donor draw"))?;
    let module = &a.gas_config.distance_donors;
    if module.history_window != 0 || module.law != SamplingLaw::Independent {
        return Err(err("independent current donor rows required"));
    }
    let alive = &step.report.pre_clone_eligible;
    let population = &step.before;
    let mut rows = vec![];
    for i in 0..population.len() {
        let mut logs = vec![f64::NEG_INFINITY; population.len()];
        if alive[i] {
            for j in 0..population.len() {
                if alive[j] && (module.allow_self || i != j) {
                    let distance = module.distance.compare(
                        &population.observations,
                        i,
                        &population.observations,
                        j,
                    )?;
                    logs[j]=module.kernel.log_weight(distance, <algorithmic_gas::geometry::Distance as AlgorithmicDistance<f64>>::comparison_kind(&module.distance))?;
                }
            }
        }
        let peak = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut weights = logs
            .iter()
            .map(|&l| if l.is_finite() { (l - peak).exp() } else { 0. })
            .collect::<Vec<_>>();
        let mass = weights.iter().sum::<f64>();
        if mass > 0. {
            for w in &mut weights {
                *w /= mass;
            }
        }
        rows.push(weights);
    }
    Ok(rows)
}
fn local_effective_counts(a: &RunArchive<f64>) -> Result<Vec<f64>> {
    use algorithmic_gas::physics::{fitness::local_log_weights, jet::JetSpace};
    let step = a.steps.last().ok_or_else(|| err("missing population"))?;
    let field = step.before.observations.field("positions")?;
    let points = field
        .values()
        .chunks(field.width())
        .map(|p| p.to_vec())
        .collect::<Vec<_>>();
    let space = JetSpace::new(field.width(), 0)?;
    let mut counts = vec![];
    for (i, point) in points.iter().enumerate() {
        if !step.report.pre_clone_eligible[i] {
            counts.push(0.);
            continue;
        }
        let query = point.iter().map(|&x| space.constant(x)).collect::<Vec<_>>();
        let Some(logs) = local_log_weights(
            &query,
            &points,
            &[],
            &step.report.pre_clone_eligible,
            i,
            &a.gas_config.fitness.reward_standardizer,
        )?
        else {
            return Err(err("local standardization required"));
        };
        let peak = logs
            .iter()
            .map(|l| l.value())
            .fold(f64::NEG_INFINITY, f64::max);
        let weights = logs
            .iter()
            .map(|l| (l.value() - peak).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let squared = weights.iter().map(|w| w * w).sum::<f64>();
        counts.push(if squared > 0. {
            total * total / squared
        } else {
            0.
        });
    }
    Ok(counts)
}
fn harmonic_prediction(a: &RunArchive<f64>) -> Result<Vec<[f64; 3]>> {
    let (h, gamma) = match a.gas_config.kinetic.integrator {
        KineticKind::Baoab { dt, friction, .. } => (dt, friction),
        _ => return Err(err("harmonic prediction requires BAOAB")),
    };
    if a.gas_config.fitness.reward_exponent != 0.
        || a.gas_config.fitness.diversity_exponent != 0.
        || !matches!(a.gas_config.boundary, BoundaryPolicy::Unbounded)
    {
        return Err(err(
            "harmonic conditional prediction requires constant-fitness conservative configuration",
        ));
    }
    let scale = match &a.gas_config.kinetic.noise.geometry {
        NoiseGeometry::Isotropic {
            scale: FactorValues::Constant { values },
        } => values[0],
        _ => return Err(err("harmonic constant isotropic factor")),
    };
    let decay = (-gamma * h).exp();
    let variance = if gamma == 0. {
        h
    } else {
        -(-2. * gamma * h).exp_m1() / (2. * gamma)
    } * scale
        * scale;
    let update = |mut x: f64, mut v: f64, z: f64| {
        v -= h * 0.5 * x;
        x += h * 0.5 * v;
        v = decay * v + z;
        x += h * 0.5 * v;
        v -= h * 0.5 * x;
        [x, v]
    };
    let c0 = update(1., 0., 0.);
    let c1 = update(0., 1., 0.);
    let g = update(0., 0., variance.sqrt());
    let m = [[c0[0], c1[0]], [c0[1], c1[1]]];
    let initial = &a.steps[0].before;
    let xs = coordinates(initial, "positions", false)?;
    let vs = coordinates(initial, "velocities", false)?;
    let mut centers = xs
        .into_iter()
        .zip(vs)
        .map(|(x, v)| [x, v])
        .collect::<Vec<_>>();
    let mut covariance = [[0.; 2]; 2];
    let mut out = vec![];
    for step in &a.steps {
        for center in &mut centers {
            *center = update(center[0], center[1], 0.);
        }
        let mut next = [[0.; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                next[i][j] = g[i] * g[j];
                for k in 0..2 {
                    for l in 0..2 {
                        next[i][j] += m[i][k] * covariance[k][l] * m[j][l];
                    }
                }
            }
        }
        covariance = next;
        let expected = mean(
            &centers
                .iter()
                .map(|c| c[1].cos() * (-0.5 * covariance[1][1]).exp())
                .collect::<Vec<_>>(),
        );
        let second = mean(
            &centers
                .iter()
                .map(|c| c[1] * c[1] + covariance[1][1])
                .collect::<Vec<_>>(),
        );
        out.push([step.report.step as f64 * h, expected, second]);
    }
    Ok(out)
}
pub fn analyze_ensemble(
    id: &str,
    p: &Value,
    archives: &[RunArchive<f64>],
) -> Result<ExperimentResult> {
    let first = archives.first().ok_or_else(|| err("empty ensemble"))?;
    let mut result = analyze(id, p, first)?;
    if matches!(id, "II-03" | "II-04") && archives.len() == 2 {
        let left = coordinates(
            archives[0].terminal().1,
            "positions",
            first.gas_config.include_truncated,
        )?;
        let right = coordinates(
            archives[1].terminal().1,
            "positions",
            first.gas_config.include_truncated,
        )?;
        let mut ls = left.clone();
        let mut rs = right.clone();
        ls.sort_by(f64::total_cmp);
        rs.sort_by(f64::total_cmp);
        if ls.len() == rs.len() && !ls.is_empty() {
            let cost = mean(
                &ls.iter()
                    .zip(&rs)
                    .map(|(a, b)| (a - b).powi(2))
                    .collect::<Vec<_>>(),
            );
            result.metric("Between-run marginal W2 squared", cost, "");
            result.metric(
                "Between-run independent coupling cost",
                variance(&left) + variance(&right) + (mean(&left) - mean(&right)).powi(2),
                "",
            );
        }
        result.plot(
            "Two executed coordinate distributions",
            "sorted index",
            "position",
            vec![
                series(
                    "First run",
                    ls.iter().enumerate().map(|(i, &x)| [i as f64, x]),
                ),
                series(
                    "Second run",
                    rs.iter().enumerate().map(|(i, &x)| [i as f64, x]),
                ),
            ],
        );
    }
    if id == "IV-14" {
        let mut observations = vec![];
        let mut predictions = vec![];
        for a in archives {
            let h = match a.gas_config.kinetic.integrator {
                KineticKind::Baoab { dt, .. } => dt,
                _ => unreachable!(),
            };
            let v = coordinates(a.terminal().1, "velocities", false)?;
            observations.push([h, mean(&v.iter().map(|v| v.cos()).collect::<Vec<_>>())]);
            predictions.push([h, harmonic_prediction(a)?.last().unwrap()[1]]);
        }
        result.plot(
            "Matched-duration timestep refinement",
            "timestep h",
            "endpoint mean cos(v)",
            vec![
                series("Executed estimate", observations),
                series("Exact finite-step expectation", predictions),
            ],
        );
    }
    if archives.len() > 1 {
        let mut points = vec![];
        let mut means = vec![];
        for (i, a) in archives.iter().enumerate() {
            let (_, pop) = a.terminal();
            let xs = coordinates(pop, "positions", a.gas_config.include_truncated)?;
            let m = mean(&xs);
            points.push([i as f64, m]);
            means.push(m);
        }
        result.plot(
            "Independent run endpoint means",
            "run index",
            "mean position",
            vec![series("Executed runs", points)],
        );
        result.metric("Across-run endpoint variance", variance(&means), "");
        result.details["run_seeds"] = json!(
            archives
                .iter()
                .map(|a| a.gas_config.seed)
                .collect::<Vec<_>>()
        );
        result.details["run_steps"] =
            json!(archives.iter().map(|a| a.steps.len()).collect::<Vec<_>>());
        if id == "IV-14" {
            result.note("Refinement runs share seeds but are not Brownian-bridge coupled; differences retain sampling error.");
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use algorithmic_gas::RecordingConfig;
    async fn archive(id: &str, p: &Value, seed: u64, updates: usize) -> Result<RunArchive<f64>> {
        let c = config(id, p, seed)?;
        let mut gas = c.build::<f64>().await?;
        initialize(id, p, &mut gas).await?;
        gas.start_recording(RecordingConfig {
            max_steps: updates,
            max_bytes: 32 * 1024 * 1024,
        })?;
        for _ in 0..updates {
            gas.step().await?;
        }
        Ok(gas.recording().unwrap().clone())
    }
    #[test]
    fn every_early_demo_measures_executed_updates() {
        futures_lite::future::block_on(async {
            for (part, count) in [("I", 10), ("II", 8), ("III", 8), ("IV", 16)] {
                for n in 1..=count {
                    let id = format!("{part}-{n:02}");
                    for seed in [0, 7, 516] {
                        let p = json!({"walkers":8,"N":8});
                        let a = archive(&id, &p, seed, 4)
                            .await
                            .unwrap_or_else(|e| panic!("{id} seed {seed}: {e}"));
                        let result = analyze(&id, &p, &a)
                            .unwrap_or_else(|e| panic!("{id} seed {seed}: {e}"));
                        assert!(!result.plots.is_empty(), "{id}");
                        assert!(
                            result
                                .plots
                                .iter()
                                .flat_map(|p| &p.series)
                                .any(|s| !s.points.is_empty()),
                            "{id}: no observations"
                        );
                        assert_eq!(result.details["archive_steps"], 4);
                        assert!(
                            result
                                .plots
                                .iter()
                                .flat_map(|p| &p.series)
                                .flat_map(|s| &s.points)
                                .flatten()
                                .all(|x| x.is_finite()),
                            "{id}"
                        );
                    }
                }
            }
        });
    }
    #[test]
    fn derivatives_differentiate_normalization_on_executed_data() {
        futures_lite::future::block_on(async {
            for id in ["IV-05", "IV-07"] {
                for seed in [0, 7, 516] {
                    let p = json!({"walkers":8,"sigma":0.15,"rho":0.7});
                    let a = archive(id, &p, seed, 3).await.unwrap();
                    let jet = conditional_jet(id, &p, &a, 0., 4).unwrap();
                    let measured = a.steps.last().unwrap().report.pre_clone_fitness.fitness[0];
                    assert!(
                        (jet.value() - measured).abs() < 1e-10,
                        "{id} reconstructed {} actual {measured}",
                        jet.value()
                    );
                    let e = 1e-5;
                    let fd = (conditional_jet(id, &p, &a, e, 0).unwrap().value()
                        - conditional_jet(id, &p, &a, -e, 0).unwrap().value())
                        / (2. * e);
                    let j = jet
                        .space
                        .indices
                        .iter()
                        .position(|x| x == &vec![1, 0])
                        .unwrap();
                    assert!(
                        (fd - jet.coefficients[j]).abs() < 1e-5 * (1. + fd.abs()),
                        "{id}: {fd} vs {}",
                        jet.coefficients[j]
                    );
                }
            }
        });
    }
    #[test]
    fn recorded_analysis_does_not_change_future_dynamics() {
        futures_lite::future::block_on(async {
            let p = json!({"walkers":8});
            let c = config("I-01", &p, 7).unwrap();
            let mut gas = c.build::<f64>().await.unwrap();
            gas.start_recording(RecordingConfig {
                max_steps: 4,
                max_bytes: 32 * 1024 * 1024,
            })
            .unwrap();
            gas.step().await.unwrap();
            let checkpoint = gas.checkpoint();
            let result = analyze("I-01", &p, gas.recording().unwrap()).unwrap();
            assert!(!result.plots.is_empty());
            let mut replay = c.build::<f64>().await.unwrap();
            replay.restore(checkpoint).unwrap();
            gas.step().await.unwrap();
            replay.step().await.unwrap();
            assert_eq!(gas.population(), replay.population());
        });
    }
    #[test]
    fn matching_cost_is_bounded_by_independent_coupling() {
        futures_lite::future::block_on(async {
            for seed in [0, 7, 516] {
                let p = json!({"walkers":16});
                let a = archive("II-03", &p, seed, 8).await.unwrap();
                let r = analyze("II-03", &p, &a).unwrap();
                let metric = |name: &str| {
                    r.metrics
                        .iter()
                        .find(|m| m.label == name)
                        .unwrap()
                        .value
                        .unwrap()
                };
                assert!(
                    metric("Exact one-coordinate W2 squared")
                        <= metric("Independent coupling cost") + 1e-12
                );
            }
        });
    }
    #[test]
    fn timestep_protocol_uses_the_same_physical_horizon() {
        let p = json!({"T":5.});
        let cs = configs("IV-14", &p, 7).unwrap();
        for c in cs {
            assert!((steps("IV-14", &p, &c) as f64 * dt(&c) - 5.).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod control_tests {
    use super::*;
    #[test]
    fn every_declared_control_endpoint_runs_in_rust() {
        futures_lite::future::block_on(async {
            let catalog = algorithmic_gas::lecture::catalog();
            for spec in catalog {
                let id = spec["id"].as_str().unwrap();
                if index(id).is_err() {
                    continue;
                }
                let controls = spec["controls"].as_array().unwrap();
                let base: serde_json::Map<String, Value> = controls
                    .iter()
                    .map(|c| (c["key"].as_str().unwrap().into(), c["value"].clone()))
                    .collect();
                for control in controls {
                    let key = control["key"].as_str().unwrap();
                    let values = if let Some(options) = control["options"].as_array() {
                        options
                            .iter()
                            .map(|o| o["value"].clone())
                            .collect::<Vec<_>>()
                    } else {
                        vec![control["min"].clone(), control["max"].clone()]
                    };
                    for value in values {
                        let mut p = Value::Object(base.clone());
                        p[key] = value.clone();
                        let c =
                            config(id, &p, 7).unwrap_or_else(|e| panic!("{id} {key}={value}: {e}"));
                        let mut gas = c
                            .build::<f64>()
                            .await
                            .unwrap_or_else(|e| panic!("{id} {key}={value}: {e}"));
                        initialize(id, &p, &mut gas).await.unwrap();
                        gas.start_recording(algorithmic_gas::RecordingConfig {
                            max_steps: 1,
                            max_bytes: 128 * 1024 * 1024,
                        })
                        .unwrap();
                        if let Err(error) = gas.step().await {
                            assert!(
                                matches!(error, GasError::Extinction),
                                "{id} {key}={value}: {error}"
                            );
                        }
                        analyze(id, &p, gas.recording().unwrap())
                            .unwrap_or_else(|e| panic!("{id} {key}={value}: {e}"));
                    }
                }
            }
        });
    }
    #[test]
    fn harmonic_prediction_matches_independent_population_sampling() {
        futures_lite::future::block_on(async {
            let mut residuals = vec![];
            for seed in 0..24 {
                let p = json!({"walkers":32,"temperature":0.5,"h":0.08});
                let c = config("II-05", &p, seed).unwrap();
                let mut gas = c.build::<f64>().await.unwrap();
                initialize("II-05", &p, &mut gas).await.unwrap();
                gas.start_recording(algorithmic_gas::RecordingConfig {
                    max_steps: 16,
                    max_bytes: 32 * 1024 * 1024,
                })
                .unwrap();
                for _ in 0..16 {
                    gas.step().await.unwrap();
                }
                let a = gas.recording().unwrap();
                let pred = harmonic_prediction(a).unwrap().last().unwrap()[1];
                let v = coordinates(gas.population(), "velocities", false).unwrap();
                residuals.push(mean(&v.iter().map(|v| v.cos()).collect::<Vec<_>>()) - pred);
            }
            let se = (variance(&residuals) / (residuals.len() - 1) as f64).sqrt();
            assert!(
                mean(&residuals).abs() < 4. * se,
                "residual {} SE {se}",
                mean(&residuals)
            );
        });
    }
}
