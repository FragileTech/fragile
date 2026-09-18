//! Part III uses the canonical fixed-step gas and independent trajectory ensembles.
use crate::{Benchmark, RunConfig};
use algorithmic_gas::{
    GasConfig, GasError, Result, RunArchive,
    boundary::{BoundaryPolicy, BoxDomain},
    mean_field::step_diagnostics,
    mean_field_population::{AtomicMeanField, MeanFieldLimits},
    physics::partvi::{ExperimentResult, Series},
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
fn number(p: &Value, k: &str, d: f64) -> f64 {
    p.get(k).and_then(Value::as_f64).unwrap_or(d)
}
pub fn config(id: &str, p: &Value, seed: u64) -> Result<RunConfig> {
    let mut gas = GasConfig::euclidean(2, number(p, "h", number(p, "dt", 0.04)))?;
    gas.seed = seed;
    gas.clone_decision.saturation = number(p, "saturation", 1.);
    gas.clone_transform.restitution = Some(number(p, "restitution", 0.5));
    gas.clone_transform.jitter_amplitude = number(p, "jitter", 0.1);
    gas.kinetic.position_diffusion = number(p, "position_diffusion", 0.1);
    let b = number(p, "box", if id == "III-01" { 1.1 } else { 2. });
    gas.boundary = BoundaryPolicy::AbsorbingBox {
        field: "positions".into(),
        domain: BoxDomain {
            lower: vec![-b; 2],
            upper: vec![b; 2],
        },
    };
    let mut c = RunConfig {
        benchmark: Benchmark::Quadratic,
        walkers: number(p, "walkers", number(p, "N", 64.)) as usize,
        dimensions: 2,
        initial_lower: -1.,
        initial_upper: 1.,
        gas,
        ..Default::default()
    };
    if id == "III-05" {
        c.gas.fitness.reward_exponent = number(p, "ratio", 1.);
    }
    if id == "III-06" {
        c.gas.kinetic.position_diffusion = (2. * number(p, "diffusivity", 0.5)).sqrt();
    }
    c.validate()?;
    Ok(c)
}
pub fn configs(id: &str, p: &Value, seed: u64) -> Result<Vec<RunConfig>> {
    let base = config(id, p, seed)?;
    if !matches!(id, "III-03" | "III-07") {
        return Ok(vec![base]);
    }
    let max = base.walkers;
    let mut sizes = vec![(max / 4).max(4), (max / 2).max(4), max.max(4)];
    sizes.sort();
    sizes.dedup();
    let replicas = number(p, "replicas", 8.) as usize;
    if !(2..=128).contains(&replicas) {
        return Err(GasError::Configuration(
            "mean-field ensembles require 2..128 independent runs".into(),
        ));
    }
    let mut out = vec![];
    for n in sizes {
        for r in 0..replicas {
            let mut c = base.clone();
            c.walkers = n;
            c.gas.seed = seed.wrapping_add((r as u64 + 1) * 104729);
            out.push(c);
        }
    }
    Ok(out)
}
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}
fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (x.len().saturating_sub(1).max(1)) as f64
}
/// Independent rooted-component prediction conditioned only on the entering
/// marked population, before sampling measurement marks, donors and gates.
/// The Monte Carlo standard error belongs to the reference mean; it does not
/// include the fluctuation of the one observed finite population.
pub fn rooted_clone_reference(
    archive: &RunArchive<f64>,
    samples: usize,
    seed: u64,
) -> Result<Value> {
    if !(256..=1024).contains(&samples) {
        return Err(GasError::Configuration(
            "rooted reference requires 256..1024 independent draws".into(),
        ));
    }
    let Some(initial) = archive.steps.first() else {
        return Ok(json!({"status":"unavailable","reason":"no recorded clone stage"}));
    };
    let velocity_field = archive
        .gas_config
        .clone_transform
        .velocity_field
        .as_deref()
        .ok_or_else(|| GasError::MissingField("component velocity field".into()))?;
    let position_field = archive
        .gas_config
        .clone_transform
        .position_field
        .as_deref()
        .ok_or_else(|| GasError::MissingField("component position field".into()))?;
    // Initialization has zero velocity. Use the first informative entering law
    // when it exists, while evaluating only one archive stage.
    let step = archive
        .steps
        .iter()
        .find(|s| {
            s.before
                .observations
                .field(velocity_field)
                .is_ok_and(|v| v.values().iter().any(|v| v.abs() > 1e-14))
        })
        .unwrap_or(initial);
    let solver = AtomicMeanField::new(
        &step.before,
        &archive.gas_config,
        MeanFieldLimits::default(),
    )?;
    let actual = step
        .stages
        .iter()
        .find(|s| s.stage == "post_transform")
        .ok_or_else(|| {
            GasError::Checkpoint("rooted reference requires actual post_transform snapshot".into())
        })?;
    let positions = actual
        .fields
        .get(position_field)
        .ok_or_else(|| GasError::MissingField(position_field.into()))?;
    let velocities = actual
        .fields
        .get(velocity_field)
        .ok_or_else(|| GasError::MissingField(velocity_field.into()))?;
    let d = step.before.observations.field(velocity_field)?.width();
    let readout = |x: &[f64], v: &[f64]| {
        [
            x[0].sin(),
            x[0].cos(),
            v[0],
            v[0].powi(2),
            0.5 * v.iter().map(|v| v * v).sum::<f64>(),
        ]
    };
    let mut observed = [0.; 5];
    for (x, v) in positions.values.chunks(d).zip(velocities.values.chunks(d)) {
        for (a, value) in readout(x, v).into_iter().enumerate() {
            observed[a] += value / step.before.len() as f64;
        }
    }
    let mut values: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(samples));
    let mut root_sizes = BTreeMap::<usize, usize>::new();
    let mut proposals = 0;
    let mut maximum_proposals = 0;
    let mut maximum_nodes = 0;
    for draw in 0..samples {
        // Any capacity error propagates. There is no selected-small-component retry.
        let root = solver.sample_root(seed, draw as u64)?;
        for (a, value) in readout(&root.positions, &root.velocities)
            .into_iter()
            .enumerate()
        {
            values[a].push(value);
        }
        *root_sizes.entry(root.component.len()).or_default() += 1;
        proposals += root.incoming_proposals;
        maximum_proposals = maximum_proposals.max(root.incoming_proposals);
        maximum_nodes = maximum_nodes.max(root.component.len());
    }
    let names = [
        "sin_position_0",
        "cos_position_0",
        "momentum_0",
        "stress_00",
        "kinetic_energy",
    ];
    let estimates=names.into_iter().enumerate().map(|(a,name)| {
        let prediction=mean(&values[a]);
        json!({"observable":name,"reference_mean":prediction,"reference_mc_standard_error":(variance(&values[a])/samples as f64).sqrt(),
            "finite_population_value":observed[a],"finite_population_minus_reference":observed[a]-prediction})
    }).collect::<Vec<_>>();
    let diagnostic = step_diagnostics(step)?;
    let mut finite_sizes = BTreeMap::<usize, usize>::new();
    // A uniformly tagged walker sees a component of size k with mass k/N.
    for size in diagnostic.component_sizes {
        *finite_sizes.entry(size).or_default() += size;
    }
    let support = root_sizes
        .keys()
        .chain(finite_sizes.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let distribution=support.into_iter().map(|size| {
        let probability=*root_sizes.get(&size).unwrap_or(&0) as f64/samples as f64;
        json!({"component_size":size,"reference_probability":probability,"reference_mc_standard_error":(probability*(1.-probability)/samples as f64).sqrt(),
            "finite_tagged_walker_probability":*finite_sizes.get(&size).unwrap_or(&0) as f64/step.before.len() as f64})
    }).collect::<Vec<_>>();
    Ok(
        json!({"status":"available","calculation_origin":"independent_rust_rooted_component_reference","conditioned_on":"entering_full_marked_population_and_raw_rewards",
        "stage":"post_transform","step":step.report.step,"walkers":step.before.len(),"independent_root_draws":samples,"prediction_seed":seed.to_string(),
        "random_stream":"MeanFieldReference","measurement_type_count":solver.types.len(),"alive_input_mass":solver.alive_mass,
        "reward_normalizer":solver.reward_normalizer,"diversity_normalizer":solver.diversity_normalizer,
        "incoming_intensity_bound":solver.incoming_intensity_bound,"incoming_proposals":proposals,"maximum_proposals_per_root":maximum_proposals,
        "maximum_component_size":maximum_nodes,"estimates":estimates,"component_size_distribution":distribution,
        "uncertainty_scope":"Monte Carlo error of independent reference draws only; finite-population variation and finite-N bias are separate"}),
    )
}

pub fn analyze(id: &str, archives: &[RunArchive<f64>]) -> Result<ExperimentResult> {
    let first = archives
        .first()
        .ok_or_else(|| GasError::Configuration("empty mean-field ensemble".into()))?;
    let mut result = ExperimentResult::new(
        id[4..].parse().unwrap_or(0),
        id,
        "Canonical fixed-timestep Euclidean Gas",
    );
    let mut traces = vec![];
    for a in archives {
        a.validate()?;
        traces.push(
            a.steps
                .iter()
                .map(step_diagnostics)
                .collect::<Result<Vec<_>>>()?,
        );
    }
    let trace = &traces[0];
    let collision_observable =
        algorithmic_gas::physics::field_evolution::WeakFieldObservable::Stress {
            k: vec![0.; 2],
            a: 0,
            b: 0,
        };
    let collision_balances = first
        .steps
        .iter()
        .map(|step| {
            algorithmic_gas::physics::field_evolution::collision_field_balance(
                &first.gas_config,
                step,
                &collision_observable,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let line = |name: &str, f: fn(&algorithmic_gas::mean_field::MeanFieldStep) -> f64| {
        Series::line(name, trace.iter().map(|s| [s.step as f64, f(s)]).collect())
    };
    result.plot(
        "Accepted clones and conditional prediction",
        "update",
        "walkers",
        vec![
            line("Actual clones", |s| s.clones as f64),
            line("Sum of acceptance probabilities", |s| s.expected_clones),
        ],
    );
    result.plot(
        "Collision component dependence",
        "update",
        "fraction",
        vec![
            line("Largest component / N", |s| s.largest_component_fraction),
            line("Two distinct walkers share a component", |s| {
                s.shared_component_probability
            }),
        ],
    );
    result.plot(
        "Alive population and immediate revival",
        "update",
        "walkers",
        vec![
            line("Alive after update", |s| s.alive_after as f64),
            line("Revived this update", |s| s.revivals as f64),
        ],
    );
    result.plot(
        "Executed bounded population observables",
        "update",
        "full-slot mean",
        vec![
            line("Mean sin(x₀)", |s| s.mean_sin_position),
            line("Mean cos(x₀)", |s| s.mean_cos_position),
        ],
    );
    result.plot(
        "Component rotation: conditional stress increment",
        "update",
        "mean v₀² increment",
        vec![
            Series::line(
                "Actual component transform",
                collision_balances
                    .iter()
                    .map(|b| [b.step as f64, b.realized_increment[0]])
                    .collect(),
            ),
            Series::line(
                "Shared-Haar conditional mean",
                collision_balances
                    .iter()
                    .map(|b| [b.step as f64, b.conditional_increment[0]])
                    .collect(),
            ),
        ],
    );
    result.metric(
        "Maximum component momentum residual",
        collision_balances
            .iter()
            .map(|b| b.maximum_momentum_conservation_residual)
            .fold(0., f64::max),
        "",
    );
    result.metric(
        "Maximum component relative-energy residual",
        collision_balances
            .iter()
            .map(|b| b.maximum_relative_energy_residual)
            .fold(0., f64::max),
        "",
    );
    if trace.is_empty() {
        result.plot(
            "Extinct population",
            "update",
            "alive walkers",
            vec![Series::line("Recorded extinction", vec![[0., 0.]])],
        );
    }
    let mut by_size: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, a) in archives.iter().enumerate() {
        by_size.entry(a.terminal().1.len()).or_default().push(i);
    }
    let mut estimates = vec![];
    let mut variance_points = vec![];
    let mut correlation_points = vec![];
    for (n, indices) in by_size {
        let mut endpoints = vec![];
        let mut pair_products = vec![];
        for i in &indices {
            let pop = archives[*i].terminal().1;
            let o = algorithmic_gas::mean_field::population_observables(pop)?;
            endpoints.push(o[1]);
            pair_products.push(o[3]);
        }
        let r = endpoints.len();
        let m = mean(&endpoints);
        let v = variance(&endpoints);
        let independent_product = if r > 1 {
            (endpoints.iter().sum::<f64>().powi(2) - endpoints.iter().map(|x| x * x).sum::<f64>())
                / (r * (r - 1)) as f64
        } else {
            m * m
        };
        let pair_cov = mean(&pair_products) - independent_product;
        estimates.push(json!({"N":n,"runs":r,"mean_sin":m,"mean_standard_error":(v/r as f64).sqrt(),"variance_of_population_mean":v,"distinct_pair_covariance":(r>1&&n>1).then_some(pair_cov),"seeds":indices.iter().map(|i|archives[*i].gas_config.seed).collect::<Vec<_>>()}));
        if r > 1 {
            variance_points.push([n as f64, v]);
            correlation_points.push([n as f64, pair_cov]);
        }
    }
    if !variance_points.is_empty() {
        result.plot(
            "Population-size scaling at fixed update count",
            "walkers N",
            "across-run variance",
            vec![Series::line("Var(mean sin(x₀))", variance_points)],
        );
        result.plot(
            "Distinct-particle correlations",
            "walkers N",
            "covariance",
            vec![Series::line(
                "E[sin(xᵢ)sin(xⱼ)] − E[sin(xᵢ)]²",
                correlation_points,
            )],
        );
    }
    let resid: f64 = trace
        .iter()
        .map(|s| s.clones as f64 - s.expected_clones)
        .sum();
    let var: f64 = trace.iter().map(|s| s.clone_count_variance).sum();
    result.metric("Recorded updates", trace.len() as f64, "");
    result.metric(
        "Total revivals",
        trace.iter().map(|s| s.revivals).sum::<usize>() as f64,
        "",
    );
    result.metric(
        "Clone-count martingale / predicted scale",
        if var > 0. {
            resid / var.sqrt()
        } else {
            f64::NAN
        },
        "",
    );
    result.note("Each curve uses full-slot observables from Rust runs. Across-run uncertainty uses independent trajectories; walkers within a collision component share one rotation.");
    result.note("The clone-count prediction conditions on sampled companions and realized fitness. It tests the actual finite-step Bernoulli gates.");
    if id == "III-07" {
        result.note("Shared initial randomness tests an empirical-mixture limit. Exchangeability alone does not force concentration at a deterministic population law.");
    }
    result.details = json!({"calculation_origin":"executed_rust_archive","lecture_id":id,"archive_steps":first.steps.len(),"canonical_kernel":true,"population_estimates":estimates,"step_diagnostics":traces,"terminal_outcome":if trace.is_empty(){"extinction"}else{"completed"}});
    result.details["collision_field_balances"] = json!(collision_balances);
    result.details["run_seeds"] = json!(
        archives
            .iter()
            .map(|a| a.gas_config.seed)
            .collect::<std::collections::BTreeSet<_>>()
    );
    result.details["ensemble_configurations"] = json!(archives.iter().enumerate().map(|(i,a)| {
        let h = match &a.gas_config.kinetic.integrator {
            algorithmic_gas::kinetic::KineticKind::Baoab{dt,..} => Some(*dt),
            _ => None,
        };
        json!({"archive_index":i,"N":a.terminal().1.len(),"seed":a.gas_config.seed,"h":h,"recorded_updates":a.steps.len()})
    }).collect::<Vec<_>>());
    result.details["ensemble_sampling"] = json!(
        "Independent seeds within each N; corresponding seeds are shared across N for paired size comparisons"
    );

    let reference =
        rooted_clone_reference(first, 512, first.gas_config.seed ^ 0x4d46_5245_465f_4949)?;
    if reference["status"] == "available" {
        let estimates = reference["estimates"].as_array().unwrap();
        for (title, indices, axis) in [
            (
                "Rooted population prediction: position",
                vec![0, 1],
                "0: sin(x₀), 1: cos(x₀)",
            ),
            (
                "Rooted population prediction: collision moments",
                vec![2, 3, 4],
                "0: v₀, 1: v₀², 2: |v|²/2",
            ),
        ] {
            let series = |name: &str, field: &str, offset: f64| {
                Series::line(
                    name,
                    indices
                        .iter()
                        .enumerate()
                        .map(|(j, &i)| {
                            [
                                j as f64,
                                estimates[i][field].as_f64().unwrap()
                                    + offset
                                        * estimates[i]["reference_mc_standard_error"]
                                            .as_f64()
                                            .unwrap(),
                            ]
                        })
                        .collect(),
                )
            };
            result.plot(
                title,
                axis,
                "full-slot mean",
                vec![
                    series("Actual finite population", "finite_population_value", 0.),
                    series("Independent rooted mean-field mean", "reference_mean", 0.),
                    series("Reference mean + MC standard error", "reference_mean", 1.),
                    series("Reference mean − MC standard error", "reference_mean", -1.),
                ],
            );
        }
        let distribution = reference["component_size_distribution"].as_array().unwrap();
        let line = |name: &str, field: &str| {
            Series::line(
                name,
                distribution
                    .iter()
                    .map(|row| {
                        [
                            row["component_size"].as_f64().unwrap(),
                            row[field].as_f64().unwrap(),
                        ]
                    })
                    .collect(),
            )
        };
        result.plot(
            "Rooted component-size distribution",
            "component size",
            "probability for a uniformly tagged walker",
            vec![
                line(
                    "Actual finite population",
                    "finite_tagged_walker_probability",
                ),
                line("Independent rooted mean-field law", "reference_probability"),
            ],
        );
        result.metric("Independent rooted reference draws", 512., "");
        result.metric(
            "Rooted reference update",
            reference["step"].as_f64().unwrap(),
            "",
        );
        result.note("The rooted population prediction redraws measurement marks, accepted graph components and shared rotations from the entering atomic law. It uses no realized companion, fitness normalization or accepted graph to predict the output. Its standard-error curves describe only reference Monte Carlo error; one finite population also has sampling variation and finite-N bias.");
    }
    result.details["rooted_mean_field_reference"] = reference;

    let preferred = match id {
        "III-01" | "III-06" => "Alive population",
        "III-03" => "Population-size",
        "III-04" => "Component rotation",
        "III-07" => "Distinct-particle",
        "III-08" => "Collision component",
        _ => "Accepted clones",
    };
    if let Some(i) = result
        .plots
        .iter()
        .position(|p| p.title.starts_with(preferred))
    {
        result.plots.swap(0, i);
    }
    Ok(result)
}
