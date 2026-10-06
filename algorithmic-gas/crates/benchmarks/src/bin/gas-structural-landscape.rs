//! Actual nonconvex native trajectories, regional observations and force budgets.
use algorithmic_gas::{
    GasBuilder, GasConfig, ObservationBatch, Population, RecordingConfig, TensorBatch,
    boundary::BoundaryPolicy,
};
use algorithmic_gas_benchmarks::{
    Benchmark, BenchmarkModel,
    convergence_experiments::{ArchiveStore, ExperimentConfig},
    convergence_experiments_chapter05::dimension::{
        optimal_sector_cost, sector_constants, sector_curvature_profile,
    },
    convergence_structural_rates::{
        local_force_rate_metric, nonquadratic_force_budget, rastrigin_force_defect,
    },
};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct NativeParameters {
    timestep: f64,
    friction: f64,
    velocity_cap: f64,
    position_diffusion: f64,
    clone_jitter: f64,
    restitution: f64,
    reward_exponent: f64,
    diversity_exponent: f64,
    cloning_bandwidth: f64,
}
impl Default for NativeParameters {
    fn default() -> Self {
        Self {
            timestep: 0.04,
            friction: 1.,
            velocity_cap: 2.,
            position_diffusion: 0.1,
            clone_jitter: 0.1,
            restitution: 0.5,
            reward_exponent: 1.,
            diversity_exponent: 1.,
            cloning_bandwidth: 2.,
        }
    }
}
impl NativeParameters {
    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if ![
            self.timestep,
            self.friction,
            self.velocity_cap,
            self.cloning_bandwidth,
        ]
        .iter()
        .all(|x| x.is_finite() && *x > 0.)
            || ![
                self.position_diffusion,
                self.clone_jitter,
                self.reward_exponent,
                self.diversity_exponent,
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.)
            || !self.restitution.is_finite()
            || !(0. ..=1.).contains(&self.restitution)
        {
            return Err("Invalid native structural experiment parameters".into());
        }
        Ok(())
    }
}
fn population(
    n: usize,
    d: usize,
    center: f64,
    shift: f64,
) -> algorithmic_gas::Result<Population<f64>> {
    let x = (0..n)
        .flat_map(|i| {
            (0..d).map(move |j| center + shift + 0.002 * ((i * 7 + j * 3) % 11) as f64 / 11.)
        })
        .collect();
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, d, x)?);
    obs.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, d, vec![0.; n * d])?,
    );
    Population::new(obs)
}
fn coupling(
    a: &Population<f64>,
    b: &Population<f64>,
    omega: f64,
    beta: f64,
) -> algorithmic_gas::Result<f64> {
    let ax = a.observations.field("positions")?;
    let bx = b.observations.field("positions")?;
    let av = a.observations.field("velocities")?;
    let bv = b.observations.field("velocities")?;
    Ok(ax
        .values()
        .iter()
        .zip(bx.values())
        .zip(av.values().iter().zip(bv.values()))
        .map(|((x, y), (v, w))| {
            let dx = x - y;
            let dv = v - w;
            omega * dx * dx + 2. * beta * omega.sqrt() * dx * dv + dv * dv
        })
        .sum::<f64>()
        / ax.rows() as f64)
}
fn occupancy(p: &Population<f64>) -> algorithmic_gas::Result<Value> {
    let x = p.observations.field("positions")?;
    let d = x.width();
    let mut counts = [0usize; 4];
    let mut m4 = 0.;
    let mut m8 = 0.;
    for row in x.values().chunks(d) {
        let norm2 = row.iter().map(|x| x * x).sum::<f64>();
        let phase = row.iter().map(|x| (x - x.round()).abs()).fold(0., f64::max);
        let region = if norm2 > 16. * d as f64 {
            3
        } else if phase <= 0.15 {
            0
        } else if phase >= 0.35 {
            2
        } else {
            1
        };
        counts[region] += 1;
        m4 += norm2.powi(2);
        m8 += norm2.powi(4);
    }
    Ok(
        json!({"regions":["well_core","transition","slow_zone","tail"],"fractions":counts.map(|n|n as f64/x.rows() as f64),"moment4":m4/x.rows() as f64,"moment8":m8/x.rows() as f64,
        "scope":"Recorded empirical diagnostics; moments and occupancy are not global analytic envelopes."}),
    )
}
fn physical_points(p: &Population<f64>) -> algorithmic_gas::Result<Vec<Vec<f64>>> {
    let x = p.observations.field("positions")?;
    let v = p.observations.field("velocities")?;
    (0..x.rows())
        .map(|i| Ok(x.row(i)?.iter().chain(v.row(i)?).copied().collect()))
        .collect()
}
fn recorded_field<'a>(
    r: &'a algorithmic_gas::tracking::RecordedStep<f64>,
    stage: &str,
    field: &str,
) -> algorithmic_gas::Result<&'a [f64]> {
    r.stages
        .iter()
        .find(|s| s.stage == stage)
        .and_then(|s| s.fields.get(field))
        .map(|x| x.values.as_slice())
        .ok_or_else(|| algorithmic_gas::GasError::MissingField(format!("{stage}/{field}")))
}
fn prepared_budget(
    l: &algorithmic_gas::tracking::RecordedStep<f64>,
    r: &algorithmic_gas::tracking::RecordedStep<f64>,
    n: usize,
    d: usize,
    beta: f64,
) -> algorithmic_gas::Result<(f64, f64, f64)> {
    let x = recorded_field(l, "B1_input", "positions")?;
    let y = recorded_field(r, "B1_input", "positions")?;
    let v = recorded_field(l, "B1_input", "velocities")?;
    let w = recorded_field(r, "B1_input", "velocities")?;
    let cost = x
        .iter()
        .zip(y)
        .zip(v.iter().zip(w))
        .map(|((x, y), (v, w))| {
            let dx = x - y;
            let dv = v - w;
            2. * dx * dx + 2. * beta * 2f64.sqrt() * dx * dv + dv * dv
        })
        .sum::<f64>()
        / n as f64;
    let mut differences = [0.; 2];
    for (index, stage) in ["B1_input", "B2_input"].iter().enumerate() {
        let x = recorded_field(l, stage, "positions")?;
        let y = recorded_field(r, stage, "positions")?;
        differences[index] = x
            .chunks(d)
            .zip(y.chunks(d))
            .map(|(x, y)| {
                x.iter()
                    .zip(y)
                    .map(|(x, y)| {
                        let e = 20.
                            * std::f64::consts::PI
                            * ((std::f64::consts::TAU * x).sin()
                                - (std::f64::consts::TAU * y).sin());
                        e * e
                    })
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0., f64::max);
    }
    Ok((cost, differences[0], differences[1]))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=4).contains(&args.len()) || args.get(2).is_some_and(|x| x != "selected") {
        return Err("gas-structural-landscape CONFIG.json EMPTY_OUTPUT_DIR [selected [NATIVE_PARAMETERS.json]]".into());
    }
    let selected = args.len() >= 3;
    let p: NativeParameters = if let Some(path) = args.get(3) {
        serde_json::from_reader(fs::File::open(path)?)?
    } else {
        NativeParameters::default()
    };
    p.validate()?;
    let cfg: ExperimentConfig = serde_json::from_reader(fs::File::open(&args[0])?)?;
    cfg.validate()?;
    let mut store = ArchiveStore::new(&args[1])?;
    store.save_json("structural-source",&json!({"config":cfg,"native_parameters":p,"selected_cloning":selected,"module":include_str!("../convergence_structural_rates.rs"),"runner":include_str!("gas-structural-landscape.rs"),"chapter6a":include_str!("../../../../docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md"),"algorithm_source":include_str!("../../../algorithmic-gas/src/kinetic.rs"),"cloning_source":include_str!("../../../algorithmic-gas/src/cloning.rs"),"fitness_source":include_str!("../../../algorithmic-gas/src/fitness.rs"),"benchmark_source":include_str!("../lib.rs"),"cargo_lock":include_str!("../../../../Cargo.lock"),"scope":"Unbounded capped native Rastrigin BAOAB; selected mode measures complete native cloning and applies kinetic bound to actual prepared state. No full nonlinear mixing rate is inferred from diagnostics."}))?;
    let report = futures_lite::future::block_on(async {
        let sector = sector_constants(p.timestep, p.friction, 2.)?;
        let beta = sector["beta"].as_f64().unwrap();
        let delta = sector["delta_pathwise_lower"].as_f64().unwrap();
        let mut cases = vec![];
        let mut checks = 0usize;
        let mut failures = 0usize;
        for d in if cfg.compact {
            vec![1, 2]
        } else {
            vec![1, 2, 4]
        } {
            let profile = rastrigin_force_defect(d, None)?;
            let bound = profile["residual_difference_upper"].as_f64().unwrap();
            let budget =
                nonquadratic_force_budget(p.timestep, p.friction, 2., beta, delta, bound, bound)?;
            let rho = budget["rho_upper"].as_f64().unwrap();
            let defect = budget["additive_defect_upper"].as_f64().unwrap();
            for n in if cfg.compact { vec![4] } else { vec![4, 64] } {
                for (zone, center) in [("well", 0.), ("slow_zone", 0.5), ("tail", 4.1)] {
                    let mut rows = vec![];
                    let mut archives = vec![];
                    let mut maximum_residual = f64::NEG_INFINITY;
                    for rep in 0..cfg.samples {
                        let mut gc = GasConfig::euclidean(d, p.timestep)?;
                        gc.boundary = BoundaryPolicy::Unbounded;
                        gc.seed = cfg.seed.wrapping_add(
                            0x71000000 + d as u64 * 100000 + n as u64 * 1000 + rep as u64,
                        );
                        if !selected {
                            gc.fitness.reward_exponent = 0.;
                            gc.fitness.diversity_exponent = 0.;
                            gc.clone_transform.jitter_amplitude = 0.;
                        }
                        gc.kinetic.velocity_cap = Some(p.velocity_cap);
                        if let algorithmic_gas::kinetic::KineticKind::Baoab { friction, .. } =
                            &mut gc.kinetic.integrator
                        {
                            *friction = p.friction;
                        }
                        gc.cloning_donors.kernel = algorithmic_gas::geometry::Kernel::Gaussian {
                            width: p.cloning_bandwidth,
                        };
                        if selected {
                            gc.fitness.reward_exponent = p.reward_exponent;
                            gc.fitness.diversity_exponent = p.diversity_exponent;
                            gc.clone_transform.jitter_amplitude = p.clone_jitter;
                            gc.clone_transform.restitution = Some(p.restitution);
                        }
                        gc.kinetic.position_diffusion = p.position_diffusion;
                        let reward = BenchmarkModel {
                            benchmark: if selected {
                                Benchmark::Rastrigin
                            } else {
                                Benchmark::Constant
                            },
                            field: "positions".into(),
                            direction: gc.fitness.direction,
                        };
                        let force = BenchmarkModel {
                            benchmark: Benchmark::Rastrigin,
                            ..reward.clone()
                        };
                        let left = population(n, d, center, -0.1)?;
                        let right = population(n, d, center, 0.1)?;
                        let initial = coupling(&left, &right, 2., beta)?;
                        let mut left = GasBuilder::new(left, reward.clone())
                            .gradient(force.clone())
                            .config(gc.clone())
                            .build()
                            .await?;
                        let mut right = GasBuilder::new(right, reward)
                            .gradient(force)
                            .config(gc)
                            .build()
                            .await?;
                        for gas in [&mut left, &mut right] {
                            gas.start_recording(RecordingConfig {
                                max_steps: 128,
                                max_bytes: 128 * 1024 * 1024,
                                graph: false,
                            })?;
                        }
                        let mut previous = initial;
                        let mut envelope = initial;
                        rows.push(json!({"replicate":rep,"step":0,"coupling_error":initial,"optimal_error":initial,"endpoint_states":{"left":physical_points(left.population())?,"right":physical_points(right.population())?},"left":occupancy(left.population())?,"right":occupancy(right.population())?}));
                        for step in 1..=cfg.steps {
                            let lr = left.step().await?;
                            let rr = right.step().await?;
                            let error = coupling(left.population(), right.population(), 2., beta)?;
                            let lrec = left.recording().unwrap().steps.last().unwrap();
                            let rrec = right.recording().unwrap().steps.last().unwrap();
                            let (prepared, measured1, measured2) =
                                prepared_budget(lrec, rrec, n, d, beta)?;
                            let refined = nonquadratic_force_budget(
                                p.timestep, p.friction, 2., beta, delta, measured1, measured2,
                            )?;
                            let refined_defect = refined["additive_defect_upper"].as_f64().unwrap();
                            let refined_residual = error - rho * prepared - refined_defect;
                            checks += 1;
                            if refined_residual > 1e-9 {
                                failures += 1;
                            }
                            let residual = error - rho * prepared - defect;
                            maximum_residual = maximum_residual.max(residual);
                            checks += 1;
                            if residual > 1e-9 {
                                failures += 1;
                            }
                            envelope = if selected {
                                rho * prepared + defect
                            } else {
                                rho * envelope + defect
                            };
                            let optimal = optimal_sector_cost(
                                left.population(),
                                right.population(),
                                2.,
                                beta,
                            )?;
                            checks += 1;
                            if optimal > error + 1e-10 {
                                failures += 1;
                            }
                            rows.push(json!({"replicate":rep,"step":step,"coupling_error":error,"optimal_error":optimal,"endpoint_states":if step==cfg.steps {Some(json!({"left":physical_points(left.population())?,"right":physical_points(right.population())?}))} else {None},"analytic_envelope":envelope,"prepared_coupling_error":prepared,"preparation_error_change":prepared-previous,"accepted_clones":lr.clones+rr.clones,"revivals":lr.revivals+rr.revivals,"measured_kick_defects":[measured1,measured2],"refined_force_budget_residual":refined_residual,"left":occupancy(left.population())?,"right":occupancy(right.population())?}));
                            previous = error;
                            if step % cfg.archive_chunk_steps == 0 || step == cfg.steps {
                                for (side, gas) in [("left", &mut left), ("right", &mut right)] {
                                    let tag = format!(
                                        "rastrigin-d{d}-n{n}-{zone}-rep{rep}-{side}-{step}"
                                    );
                                    archives.push(
                                        store.save_archive(&tag, &gas.stop_recording().unwrap())?,
                                    );
                                    archives.push(store.save_checkpoint(
                                        &format!("{tag}-resume"),
                                        &gas.checkpoint(),
                                    )?);
                                    if step < cfg.steps {
                                        gas.start_recording(RecordingConfig {
                                            max_steps: 128,
                                            max_bytes: 128 * 1024 * 1024,
                                            graph: false,
                                        })?;
                                    }
                                }
                            }
                        }
                    }
                    let observations = store.save_json(
                        &format!("rastrigin-d{d}-n{n}-{zone}-observations"),
                        &json!({"rows":rows}),
                    )?;
                    cases.push(json!({"landscape":"standard_rastrigin","d":d,"N":n,"initial_zone":zone,"initial_pair_coordinate_separation":0.2,"selected_cloning":selected,"samples":cfg.samples,"steps":cfg.steps,"profile":profile,"budget":budget,"observations":observations,"archives":archives,"maximum_force_budget_residual":maximum_residual}));
                }
            }
        }
        let well_omega = 2. + 40. * std::f64::consts::PI.powi(2);
        let well_sector = sector_curvature_profile(p.timestep, p.friction, well_omega);
        let well_rates = match well_sector {
            Ok(s) => {
                let rows:Vec<_>=[0.0001,0.001,0.005,0.01,0.05,0.15].iter().map(|r| {
                let lip=20.*std::f64::consts::PI.powi(2)*(1.-(std::f64::consts::TAU*r).cos());
                let omega_mid=well_omega-lip;
                let mid=sector_curvature_profile(p.timestep,p.friction,omega_mid)?;
                let alpha_mid=mid["alpha"].as_f64().unwrap();let b_mid=mid["beta"].as_f64().unwrap();let delta_mid=mid["delta_pathwise_lower"].as_f64().unwrap();
                Ok(json!({"well_half_width":r,"harmonic_curvature":omega_mid,"sector":mid,"endpoint_reference_sector":s,"rate":local_force_rate_metric(p.timestep,p.friction,omega_mid,alpha_mid,b_mid,delta_mid,lip)?}))
            }).collect::<algorithmic_gas::Result<_>>()?;
                json!(rows)
            }
            Err(e) => json!({"status":"harmonic_reference_not_certified","reason":e.to_string()}),
        };
        Ok::<Value, algorithmic_gas::GasError>(
            json!({"cases":cases,"local_well_rates":well_rates,"harmonic_reference":sector,"native_parameters":p,"summary":{"cases":cases.len(),"native_steps":cases.len()*cfg.samples*cfg.steps*2,"checks":checks,"failures":failures},"structural_tail_examples":([1,2,4,8].iter().map(|d| algorithmic_gas_benchmarks::convergence_structural_tails::nonquadratic_gaussian_example(*d)).collect::<algorithmic_gas::Result<Vec<_>>>()?),"initial_pair_coordinate_separation":0.2,"scope":"Rate plus explicit nonquadratic defect floor; the global Rastrigin floor is conservative. Slow-zone/tail occupancy and moments are empirical. Local rates require residence at both actual kick queries; neither sampled transitions nor sampled moments close the full-law theorem."}),
        )
    })?;
    store.save_json("structural-report", &report)?;
    store.finish("complete")?;
    fs::write(
        PathBuf::from(&args[1]).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", report["summary"]);
    if report["summary"]["failures"].as_u64() != Some(0) {
        return Err("Native structural comparisons failed; complete evidence retained".into());
    }
    Ok(())
}
