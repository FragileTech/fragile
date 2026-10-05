//! Actual canonical frozen-input updates and independently integrated marked roots.
use algorithmic_gas::{
    GasConfig, GasError, RecordingConfig, TensorBatch,
    boundary::{BoundaryPolicy, BoxDomain},
    geometry::{AlgorithmicDistance, Distance, InteractionKernel, Kernel},
    mean_field::step_diagnostics,
    mean_field_population::{AtomicMeanField, MeanFieldLimits},
    physics::field_evolution::{WeakFieldObservable, collision_field_balance},
    random::{RandomStream, Stream},
};
use algorithmic_gas_benchmarks::{
    Benchmark, RunConfig,
    convergence_chapter08_completion::{
        Comparison, SOURCE, alive_probability, bounded_observables, component_bounds,
        covariance_experiment, finite_moment_budget, mean_variance, root_observables,
    },
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::json;
use std::{fs, path::PathBuf};
fn gradient(x: f64, rastrigin: bool) -> f64 {
    if rastrigin {
        2. * x + 20. * std::f64::consts::PI * (2. * std::f64::consts::PI * x).sin()
    } else {
        x
    }
}
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn moment(x: &[f64], d: usize, q: usize) -> f64 {
    x.chunks_exact(d)
        .map(|r| norm(r).powi(q as i32))
        .sum::<f64>()
        / (x.len() / d) as f64
}
fn field<'a>(
    step: &'a algorithmic_gas::tracking::RecordedStep<f64>,
    stage: &str,
    name: &str,
) -> Result<&'a [f64], Box<dyn std::error::Error>> {
    Ok(&step
        .stages
        .iter()
        .find(|s| s.stage == stage)
        .ok_or_else(|| format!("missing stage {stage}"))?
        .fields[name]
        .values)
}
async fn run(
    output: &str,
    replicas: usize,
    roots: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = ArchiveStore::new(output)?;
    store.save_json("execution-source",&json!({"module":include_str!("../convergence_chapter08_completion.rs"),"runner":include_str!("gas-chapter08-completion.rs"),"chapter":SOURCE,
        "atomic_mean_field":include_str!("../../../algorithmic-gas/src/mean_field_population.rs"),"native_kinetic":include_str!("../../../algorithmic-gas/src/kinetic.rs"),
        "native_config":include_str!("../../../algorithmic-gas/src/variants/euclidean.rs"),"executable_sha256":sha256_file(&std::env::current_exe()?)?,"replicas":replicas,"roots":roots,
        "scope":"Each seed runs one complete actual native update from the same deterministic S. Rooted references redraw measurement marks before fitness, finite components with exact weighted Poisson incoming offspring, common Haar rotation, original jitter, Gaussian BAOAB and terminal classification. Capacities cause error, never conditioned truncation."}))?;
    let mut cases = vec![];
    let mut updates = 0;
    let mut all_failed = 0;
    let mut checks_total = 0;
    for (profile, rastrigin, center, box_width, kernel_width) in [
        ("quadratic-unbounded", false, 0., None, 2.),
        ("rastrigin-well", true, 0., None, 4.),
        ("rastrigin-saddle", true, 0.5, None, 4.),
        ("rastrigin-tail", true, 4.1, None, 4.),
        ("quadratic-boundary-revival", false, 0., Some(0.6), 4.),
    ] {
        for d in [1, 2, 4] {
            for n in [8, 32, 128] {
                let id = format!("{profile}-d{d}-N{n}");
                let h = 0.04;
                let gamma = 1.;
                let cap = 2.;
                let restitution = 0.5;
                let sigma_x = if box_width.is_some() { 0.7 } else { 0.1 };
                let mut gas_cfg = GasConfig::euclidean(d, h)?;
                gas_cfg.cloning_donors.kernel = Kernel::Gaussian {
                    width: kernel_width,
                };
                gas_cfg.distance_donors.kernel = Kernel::Gaussian {
                    width: kernel_width,
                };
                gas_cfg.kinetic.position_diffusion = sigma_x;
                gas_cfg.boundary = if let Some(b) = box_width {
                    BoundaryPolicy::AbsorbingBox {
                        field: "positions".into(),
                        domain: BoxDomain {
                            lower: vec![-b; d],
                            upper: vec![b; d],
                        },
                    }
                } else {
                    BoundaryPolicy::Unbounded
                };
                let cfg = RunConfig {
                    benchmark: if rastrigin {
                        Benchmark::Rastrigin
                    } else {
                        Benchmark::Quadratic
                    },
                    walkers: n,
                    dimensions: d,
                    initial_lower: -0.4,
                    initial_upper: 0.4,
                    gas: gas_cfg,
                    ..Default::default()
                };
                let mut base = cfg.build::<f64>().await?;
                let mut frozen = base.population().clone();
                let xs = (0..n)
                    .flat_map(|i| {
                        (0..d).map(move |a| {
                            center
                                + 0.3 * (((i * 17 + a * 13) % n) as f64 / (n - 1) as f64 * 2. - 1.)
                        })
                    })
                    .collect::<Vec<_>>();
                let mut xs = xs;
                if let Some(b) = box_width {
                    for i in 0..n / 4 {
                        xs[i * d] = b + 0.1 + (i % 3) as f64 * 0.1;
                    }
                }
                let vs = (0..n)
                    .flat_map(|i| (0..d).map(move |a| 0.6 * ((i + a) % 5) as f64 / 5. - 0.3))
                    .collect::<Vec<_>>();
                frozen
                    .observations
                    .fields
                    .insert("positions".into(), TensorBatch::vectors(n, d, xs)?);
                frozen
                    .observations
                    .fields
                    .insert("velocities".into(), TensorBatch::vectors(n, d, vs)?);
                base.replace_population(frozen).await?;
                let frozen = base.population().clone();
                let initial = bounded_observables(&frozen)?;
                let k = frozen.eligible(false).iter().filter(|&&a| a).count();
                let graph_bounds = component_bounds(32., kernel_width, k as f64 / n as f64, 16)?;
                let checkpoint =
                    store.save_checkpoint(&format!("{id}-frozen"), &base.checkpoint())?;
                let mut checks = vec![];
                let mut raw = vec![];
                let mut native_observables = vec![];
                let mut native_clone_observables = vec![];
                let mut marks = vec![];
                let mut clone_gate_residual = 0.;
                let mut clone_gate_variance = 0.;
                let mut stress_residual = 0.;
                let mut stress_variance = 0.;
                let mut gaussian_cos_residual = 0.;
                let mut gaussian_cos_variance = 0.;
                let mut alive_residual = 0.;
                let mut alive_variance = 0.;
                let mut component_histogram = vec![0usize; n + 1];
                let mut exact_mean = 0.;
                let mut exact_second = 0.;
                let mut mean_variance_sum = 0.;
                let mut second_variance_sum = 0.;
                let alive = frozen.eligible(false);
                // The finite law excludes the recipient. Its diversity second moment
                // includes the full sampled mark distribution, not a conditional mean shortcut.
                for i in 0..n {
                    if !alive[i] {
                        continue;
                    }
                    let mut values = vec![];
                    let mut weights = vec![];
                    for (j, &a) in alive.iter().enumerate() {
                        if !a || i == j {
                            continue;
                        }
                        let dis = cfg.gas.distance_donors.distance.compare(
                            &frozen.observations,
                            i,
                            &frozen.observations,
                            j,
                        )?;
                        let kind = <Distance as AlgorithmicDistance<f64>>::comparison_kind(
                            &cfg.gas.distance_donors.distance,
                        );
                        weights.push(cfg.gas.distance_donors.kernel.log_weight(dis, kind)?.exp());
                        values.push((dis * dis + cfg.gas.fitness.distance_floor.powi(2)).sqrt());
                    }
                    let z = weights.iter().sum::<f64>();
                    let e = weights
                        .iter()
                        .zip(&values)
                        .map(|(w, s)| w / z * s)
                        .sum::<f64>();
                    let e2 = weights
                        .iter()
                        .zip(&values)
                        .map(|(w, s)| w / z * s * s)
                        .sum::<f64>();
                    let e4 = weights
                        .iter()
                        .zip(&values)
                        .map(|(w, s)| w / z * s.powi(4))
                        .sum::<f64>();
                    exact_mean += e / k as f64;
                    exact_second += e2 / k as f64;
                    mean_variance_sum += (e2 - e * e) / (k * k) as f64;
                    second_variance_sum += (e4 - e2 * e2) / (k * k) as f64;
                    let total = z + 1.;
                    let self_mass = 1. / total;
                    checks.push(Comparison::upper(&format!("self-exclusion-{i}"),"lem-mean-field-measurement-consistency",self_mass,1./(graph_bounds.kappa*k as f64),1e-12,json!({"alive_count":k,"removed_self_mass":self_mass,"nonself_normalizer":z,"actual_kernel_lower":graph_bounds.kappa})));
                }
                for rep in 0..replicas {
                    let mut rcfg = cfg.clone();
                    rcfg.gas.seed = 202610048000 + rep as u64 + checks_total as u64 * 10_000;
                    let mut gas = rcfg.build::<f64>().await?;
                    gas.replace_population(frozen.clone()).await?;
                    gas.start_recording(RecordingConfig {
                        max_steps: 1,
                        max_bytes: 64 * 1024 * 1024,
                        ..Default::default()
                    })?;
                    gas.step().await?;
                    let archive = gas.stop_recording().ok_or("missing archive")?;
                    let step = &archive.steps[0];
                    updates += 1;
                    let archive_path = store.save_archive(&format!("{id}-rep{rep}"), &archive)?;
                    let diag = step_diagnostics(step)?;
                    checks.push(Comparison::identity(&format!("forest-{rep}"),"lem-mean-field-component-bound",diag.accepted_edges as f64,(n-diag.component_sizes.len()) as f64,json!({"component_sizes":diag.component_sizes,"accepted_edges":diag.accepted_edges})));
                    for s in &diag.component_sizes {
                        component_histogram[*s] += s;
                    }
                    let balance = collision_field_balance(
                        &rcfg.gas,
                        step,
                        &WeakFieldObservable::Stress {
                            k: vec![0.; d],
                            a: 0,
                            b: 0,
                        },
                    )?;
                    checks.push(Comparison::identity(
                        &format!("component-momentum-{rep}"),
                        "thm-mean-field-component-identities",
                        balance.maximum_momentum_conservation_residual,
                        0.,
                        json!({"component_slots":balance.component_slots}),
                    ));
                    checks.push(Comparison::identity(&format!("component-energy-{rep}"),"thm-mean-field-component-identities",balance.maximum_relative_energy_residual,0.,json!({"restitution":restitution,"component_slots":balance.component_slots})));
                    if rep == 0
                        && let Some(members) = balance.component_slots.iter().find(|m| m.len() > 1)
                    {
                        let before_v = step.before.observations.field("velocities")?;
                        let velocities = members
                            .iter()
                            .map(|&i| before_v.row(i).map(|v| v.to_vec()))
                            .collect::<algorithmic_gas::Result<Vec<_>>>()?;
                        let experiment = covariance_experiment(
                            &velocities,
                            restitution,
                            rcfg.gas.seed ^ 0x48414152,
                            256,
                        )?;
                        for check in experiment["checks"].as_array().ok_or("covariance checks")? {
                            checks.push(serde_json::from_value(check.clone())?);
                        }
                        store
                            .save_json(&format!("{id}-actual-component-covariance"), &experiment)?;
                    }
                    stress_residual += balance.martingale_increment[0];
                    stress_variance += balance.martingale_covariance[0];
                    clone_gate_residual += diag.clones as f64 - diag.expected_clones;
                    clone_gate_variance += diag.clone_count_variance;
                    let fb = &step.report.pre_clone_fitness;
                    let measured = fb
                        .separation
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| alive[*i])
                        .map(|(_, s)| s / k as f64)
                        .sum::<f64>();
                    let measured_second = fb
                        .separation
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| alive[*i])
                        .map(|(_, s)| s * s / k as f64)
                        .sum::<f64>();
                    marks.push([measured, measured_second]);
                    let reward_mean = fb
                        .oriented_reward
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| alive[*i])
                        .map(|(_, r)| r / k as f64)
                        .sum::<f64>();
                    let reward_var = fb
                        .oriented_reward
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| alive[*i])
                        .map(|(_, r)| (r - reward_mean).powi(2) / k as f64)
                        .sum::<f64>();
                    let diversity_var = fb
                        .separation
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| alive[*i])
                        .map(|(_, v)| (v - measured).powi(2) / k as f64)
                        .sum::<f64>();
                    let live_slot = alive.iter().position(|a| *a).ok_or("alive slot")?;
                    for (name, lhs, rhs) in [
                        ("reward-mean", fb.reward_stats.mean[live_slot], reward_mean),
                        (
                            "reward-scale",
                            fb.reward_stats.scale[live_slot],
                            (reward_var + 0.01).sqrt(),
                        ),
                        (
                            "diversity-mean",
                            fb.diversity_stats.mean[live_slot],
                            measured,
                        ),
                        (
                            "diversity-scale",
                            fb.diversity_stats.scale[live_slot],
                            (diversity_var + 0.01).sqrt(),
                        ),
                    ] {
                        checks.push(Comparison::identity(&format!("{name}-{rep}"),"def-mean-field-fitness-potential",lhs,rhs,json!({"alive_count":k,"reward_mean":reward_mean,"reward_variance":reward_var,"separation_mean":measured,"separation_variance":diversity_var,"sigma_floor":0.1,"source_archive":archive_path})));
                    }
                    for (i, &a) in alive.iter().enumerate() {
                        if a {
                            let rz =
                                (fb.oriented_reward[i] - reward_mean) / (reward_var + 0.01).sqrt();
                            let sz = (fb.separation[i] - measured) / (diversity_var + 0.01).sqrt();
                            let fitness =
                                (2. / (1. + (-rz).exp()) + 0.1) * (2. / (1. + (-sz).exp()) + 0.1);
                            checks.push(Comparison::identity(&format!("frozen-fitness-{rep}-{i}"),"def-mean-field-fitness-potential",fb.fitness[i],fitness,json!({"reward_z":rz,"sampled_separation_z":sz,"amplitudes":[2.,2.],"floors":[0.1,0.1],"exponents":[1.,1.],"source_archive":archive_path})));
                        }
                    }

                    for (i, choice) in step.report.clone_plan.choices.iter().enumerate() {
                        if choice.accepted {
                            let donor = &choice.donors[0];
                            let source = &step.report.clone_plan.sources[donor.pool_index as usize];
                            let j = source.slot as usize;
                            if !choice.revival {
                                checks.push(Comparison::upper(&format!("increasing-fitness-{rep}-{i}"),"def-mean-field-accepted-graph",fb.fitness[i],fb.fitness[j],0.,json!({"recipient_fitness":fb.fitness[i],"donor_fitness":fb.fitness[j],"accepted":true,"strict_gap":fb.fitness[j]-fb.fitness[i]})));
                                if fb.fitness[j] <= fb.fitness[i] {
                                    return Err("accepted non-increasing frozen fitness".into());
                                }
                            }
                        }
                    }
                    let x0 = field(step, "post_clone", "positions")?;
                    let v0 = field(step, "post_clone", "velocities")?;
                    let v1 = field(step, "B1", "velocities")?;
                    let x1 = field(step, "A1", "positions")?;
                    let v2 = field(step, "O", "velocities")?;
                    let x2 = field(step, "A2", "positions")?;
                    let v3 = field(step, "B2", "velocities")?;
                    let x3 = field(step, "position_diffusion", "positions")?;
                    let ch = (-gamma * h).exp();
                    let sh = (-(-2. * gamma * h).exp_m1() / (2. * gamma)).sqrt();
                    let eta_o = &step
                        .field_evaluations
                        .iter()
                        .find(|f| f.stage == "O" && f.field == "executed_noise")
                        .ok_or("OU innovation missing")?
                        .values;
                    let eta_x = &step
                        .field_evaluations
                        .iter()
                        .find(|f| f.stage == "position_diffusion" && f.field == "executed_noise")
                        .ok_or("position innovation missing")?
                        .values;
                    let final_v = step
                        .final_population
                        .observations
                        .field("velocities")?
                        .values();
                    for a in 0..n * d {
                        for (name, lhs, rhs) in [
                            ("B1", v1[a], v0[a] - h / 2. * gradient(x0[a], rastrigin)),
                            ("A1", x1[a], x0[a] + h / 2. * v1[a]),
                            ("O", v2[a], ch * v1[a] + sh * eta_o[a]),
                            ("A2", x2[a], x1[a] + h / 2. * v2[a]),
                            ("B2", v3[a], v2[a] - h / 2. * gradient(x2[a], rastrigin)),
                            ("position", x3[a], x2[a] + sigma_x * h.sqrt() * eta_x[a]),
                            (
                                "cap",
                                final_v[a],
                                cap * v3[a] / (cap + norm(&v3[(a / d) * d..(a / d + 1) * d])),
                            ),
                        ] {
                            checks.push(Comparison::identity(&format!("{name}-{rep}-{a}"),"def-baoab-update-rule",lhs,rhs,json!({"row":a/d,"coordinate":a%d,"h":h,"friction":gamma,"source_archive":archive_path})));
                        }
                    }
                    let output = bounded_observables(&step.final_population)?;
                    native_observables.push(output);
                    let mut clone_tests = [0.; 7];
                    for row in x0.chunks_exact(d) {
                        for (o, v) in clone_tests.iter_mut().zip(root_observables(row, true)) {
                            *o += v / n as f64;
                        }
                    }
                    native_clone_observables.push(clone_tests);
                    for (i, row) in x2.chunks_exact(d).enumerate() {
                        let variance = sigma_x * sigma_x * h;
                        let expected = (-variance / 2.).exp() * row[0].cos();
                        let second = 0.5 * (1. + (-2. * variance).exp() * (2. * row[0]).cos());
                        gaussian_cos_residual += (x3[i * d].cos() - expected) / n as f64;
                        gaussian_cos_variance += (second - expected * expected) / (n * n) as f64;
                        let p = if let Some(b) = box_width {
                            alive_probability(row, b, sigma_x * h.sqrt())?
                        } else {
                            1.
                        };
                        let a = f64::from(step.final_population.eligible(false)[i]);
                        alive_residual += (a - p) / n as f64;
                        alive_variance += p * (1. - p) / (n * n) as f64;
                        checks.push(Comparison::identity(
                            &format!("terminal-classification-{rep}-{i}"),
                            "thm-mass-conservation",
                            a,
                            if let Some(b) = box_width {
                                f64::from(x3[i * d..(i + 1) * d].iter().all(|v| v.abs() <= b))
                            } else {
                                1.
                            },
                            json!({"actual_position":&x3[i*d..(i+1)*d],"box_half_width":box_width}),
                        ));
                    }
                    checks.push(Comparison::identity(&format!("mass-balance-{rep}"),"thm-mass-conservation",output[3]-k as f64/n as f64,(n-k) as f64/n as f64-(n-diag.alive_after) as f64/n as f64,json!({"entering_alive":k,"terminal_alive":diag.alive_after,"revivals":diag.revivals,"population":n})));
                    raw.push(json!({"replica":rep,"seed":rcfg.gas.seed,"native_archive":archive_path,"output_tests":output,"clone_tests":clone_tests,"measurement_mean":measured,"measurement_second":measured_second,"diagnostics":diag,"collision_balance":balance}));
                }
                for (name, label, residual, variance) in [
                    (
                        "clone-gates",
                        "def-mean-field-accepted-graph",
                        clone_gate_residual,
                        clone_gate_variance,
                    ),
                    (
                        "shared-haar-stress",
                        "thm-mean-field-component-identities",
                        stress_residual,
                        stress_variance,
                    ),
                    (
                        "final-gaussian-cos",
                        "def-baoab-update-rule",
                        gaussian_cos_residual,
                        gaussian_cos_variance,
                    ),
                    (
                        "terminal-alive",
                        "thm-mass-conservation",
                        alive_residual,
                        alive_variance,
                    ),
                ] {
                    // Chebyshev applies to independent complete preparations. Family
                    // allocation is fixed before observation; no normal-fit acceptance.
                    let allowance = (variance / 0.00001).sqrt()
                        + 128. * f64::EPSILON * replicas as f64
                        + if name == "terminal-alive" {
                            replicas as f64 * d as f64 * 1e-7
                        } else {
                            0.
                        };
                    checks.push(Comparison::upper(name,label,residual.abs(),0.,allowance,json!({"signed_residual":residual,"conditional_variance_sum":variance,"independent_seed_units":replicas,"failure_budget":0.00001,"normal_cdf_abs_error_allowance":1e-7})));
                }
                for (j, expected, variance) in [
                    (0, exact_mean, mean_variance_sum),
                    (1, exact_second, second_variance_sum),
                ] {
                    let observed = marks.iter().map(|r| r[j]).sum::<f64>() / replicas as f64;
                    checks.push(Comparison::upper(&format!("sampled-mark-moment-{j}"),"lem-mean-field-measurement-consistency",(observed-expected).abs(),0.,(variance/replicas as f64/0.00001).sqrt(),json!({"observed":observed,"exact_finite_excluded_donor_expectation":expected,"single_population_mean_variance":variance,"alive_count":k,"population":n,"replicas":replicas,"failure_budget":0.00001})));
                }
                let input = frozen.clone();
                let solver = AtomicMeanField::new(
                    &input,
                    &cfg.gas,
                    MeanFieldLimits {
                        max_atoms: 128,
                        max_nodes: 16384,
                        max_proposals: 2_000_000,
                    },
                )?;
                let chosen_types = (0..solver.types.len())
                    .step_by((solver.types.len() / 64).max(1))
                    .collect::<Vec<_>>();
                for &t in &chosen_types {
                    let outgoing = solver
                        .types
                        .iter()
                        .enumerate()
                        .map(|(u, v)| solver.edge_density(t, u).map(|b| b * v.probability))
                        .collect::<algorithmic_gas::Result<Vec<_>>>()?
                        .iter()
                        .sum::<f64>();
                    checks.push(Comparison::upper(&format!("atomic-outgoing-{t}"),"def-mean-field-rooted-collision",outgoing,1.,1e-12,json!({"root_type":solver.types[t],"integrated_outgoing_mass":outgoing,"normalization":"full marked law eta, exact atomic companion weights including repeated types"})));
                    if !solver.types[t].alive {
                        checks.push(Comparison::identity(
                            &format!("atomic-revival-{t}"),
                            "def-mean-field-rooted-collision",
                            outgoing,
                            1.,
                            json!({"dead_root_type":solver.types[t]}),
                        ));
                    }
                }
                let reference_seed = 202610048999 + updates as u64;
                let mut reference_rows = vec![];
                let mut reference_tests = vec![];
                let mut refclone_tests = vec![];
                let mut reference_moments = [0_f64; 2];
                for draw in 0..roots {
                    let root = solver.sample_root(reference_seed, draw as u64)?;
                    let x0 = root.positions.clone();
                    let v0 = root.velocities.clone();
                    let clone_tests = root_observables(&x0, true);
                    refclone_tests.push(clone_tests);
                    let mut ou = RandomStream::new(
                        reference_seed,
                        draw as u64,
                        Stream::MeanFieldReference,
                        0,
                        3,
                    );
                    let mut final_noise = RandomStream::new(
                        reference_seed,
                        draw as u64,
                        Stream::MeanFieldReference,
                        0,
                        4,
                    );
                    let xi_o = (0..d).map(|_| ou.gaussian::<f64>()).collect::<Vec<_>>();
                    let xi_x = (0..d)
                        .map(|_| final_noise.gaussian::<f64>())
                        .collect::<Vec<_>>();
                    let ch = (-gamma * h).exp();
                    let sh = (-(-2. * gamma * h).exp_m1() / (2. * gamma)).sqrt();
                    let v1 = (0..d)
                        .map(|a| v0[a] - h / 2. * gradient(x0[a], rastrigin))
                        .collect::<Vec<_>>();
                    let x1 = (0..d).map(|a| x0[a] + h / 2. * v1[a]).collect::<Vec<_>>();
                    let v2 = (0..d)
                        .map(|a| ch * v1[a] + sh * xi_o[a])
                        .collect::<Vec<_>>();
                    let x2 = (0..d).map(|a| x1[a] + h / 2. * v2[a]).collect::<Vec<_>>();
                    let v3 = (0..d)
                        .map(|a| v2[a] - h / 2. * gradient(x2[a], rastrigin))
                        .collect::<Vec<_>>();
                    let x3 = (0..d)
                        .map(|a| x2[a] + sigma_x * h.sqrt() * xi_x[a])
                        .collect::<Vec<_>>();
                    let v4 = v3
                        .iter()
                        .map(|v| cap * v / (cap + norm(&v3)))
                        .collect::<Vec<_>>();
                    let a4 = box_width.is_none_or(|b| x3.iter().all(|v| v.abs() <= b));
                    let tests = root_observables(&x3, a4);
                    reference_tests.push(tests);
                    for (j, q) in [4, 8].into_iter().enumerate() {
                        reference_moments[j] += norm(&x3).powi(q) / roots as f64;
                    }
                    reference_rows.push(json!({"draw":draw,"reference_seed":reference_seed,"root":root,"xi_O":xi_o,"xi_x":xi_x,"X1":x1,"V1":v1,"X2":x2,"V2":v2,"V3":v3,"X3":x3,"V4":v4,"A4":a4,"output_tests":tests,"clone_tests":clone_tests}));
                }
                let refpath=store.save_json(&format!("{id}-independent-rooted-reference"),&json!({"types":solver.types,"reward_normalizer":solver.reward_normalizer,"diversity_normalizer":solver.diversity_normalizer,"alive_mass":solver.alive_mass,"incoming_intensity_bound":solver.incoming_intensity_bound,"rows":reference_rows,"source_checkpoint":checkpoint,"truncation_probability":0.,"capacity_failure_policy":"error, no truncation and no retry"}))?;
                let (native_mean, native_var) = mean_variance(&native_observables);
                let (reference_mean, reference_var) = mean_variance(&reference_tests);
                let (clone_mean, clone_var) = mean_variance(&native_clone_observables);
                let (refclone_mean, refclone_var) = mean_variance(&refclone_tests);
                let mean_size = component_histogram
                    .iter()
                    .enumerate()
                    .map(|(s, count)| s as f64 * *count as f64)
                    .sum::<f64>()
                    / (n * replicas) as f64;
                checks.push(Comparison::upper("component-mean-log","lem-mean-field-component-bound",mean_size.ln(),graph_bounds.log_expected_component_upper,0.,json!({"tagged_root_mean_component_size":mean_size,"component_bound":graph_bounds,"scope":"Conservative inequality; empirical average does not certify conditional expectation universally."})));
                if box_width.is_none() {
                    for q in [4, 8] {
                        let lu = if rastrigin {
                            2. + 40. * std::f64::consts::PI.powi(2)
                        } else {
                            1.
                        };
                        let budget = finite_moment_budget(
                            d,
                            q,
                            graph_bounds.kappa,
                            h,
                            gamma,
                            lu,
                            0.,
                            cap,
                            restitution,
                            0.1,
                            1.,
                            sigma_x,
                        )?;
                        let before = moment(input.observations.field("positions")?.values(), d, q);
                        let second = finite_moment_budget(
                            d,
                            2 * q,
                            graph_bounds.kappa,
                            h,
                            gamma,
                            lu,
                            0.,
                            cap,
                            restitution,
                            0.1,
                            1.,
                            sigma_x,
                        )?;
                        let higher =
                            moment(input.observations.field("positions")?.values(), d, 2 * q);
                        let allowance = ((second.coefficient * higher + second.additive)
                            / roots as f64
                            / 0.00001)
                            .sqrt();
                        let observed = reference_moments[if q == 4 { 0 } else { 1 }];
                        checks.push(Comparison::upper(&format!("finite-moment-p{q}"),"lem-mean-field-finite-moments",observed,budget.coefficient*before+budget.additive,allowance,json!({"budget":budget,"input_N_moment":before,"comparison_scope":"Independent complete rooted outputs compared with global configuration-only analytic moment envelope; second moment envelope supplies Chebyshev allowance.","root_draws":roots,"higher_moment_budget":second,"input_higher_moment":higher,"failure_budget":0.00001})));
                    }
                }
                let failed = checks.iter().filter(|c| !c.passed).count();
                all_failed += failed;
                checks_total += checks.len();
                let detailpath = store.save_json(
                    &format!("{id}-native-operands"),
                    &json!({"configuration":cfg,"frozen_input":input,"raw":raw,"checks":checks}),
                )?;
                cases.push(json!({"id":id,"N":n,"d":d,"profile":profile,"box_half_width":box_width,"initial_tests":initial,"entering_alive":k,"native_replicas":replicas,"root_reference_draws":roots,"native_output_mean":native_mean,"native_output_variance":native_var,"reference_output_mean":reference_mean,"reference_output_variance":reference_var,"native_clone_mean":clone_mean,"native_clone_variance":clone_var,"reference_clone_mean":refclone_mean,"reference_clone_variance":refclone_var,"component_histogram_tagged_counts":component_histogram,"graph_bounds":graph_bounds,"actual_configuration":cfg,"checks":checks.len(),"failed":failed,"native_operands":detailpath,"root_reference":refpath,"input_checkpoint":checkpoint,"reference_truncation_probability":0.}));
                eprintln!(
                    "{id}: {replicas} native updates, {roots} independent roots, {failed} failed comparisons"
                );
            }
        }
    }
    let report = json!({"chapter":8,"cases":cases,"summary":{"cases":cases.len(),"new_complete_native_updates":updates,"independent_rooted_draws":cases.len()*roots,"comparisons":checks_total,"comparisons_failed":all_failed},"test_order":["sin(x_first)","cos(x_first)","|x|²/(1+|x|²)","alive fraction","alive submass sin","alive submass cos","alive submass tail"],"scope":"Stage and graph identities, independently seeded canonical preparations and independent rooted population integration. Analytic existence, continuity and asymptotic limiting assertions require proofs, not pass labels from finite runs."});
    store.save_json("completion-report", &report)?;
    store.finish("complete")?;
    fs::write(
        PathBuf::from(output).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let verification = store.verify(true)?;
    fs::write(
        PathBuf::from(output).join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!("{}", report["summary"]);
    if all_failed > 0 {
        return Err(
            GasError::Execution("Chapter 8 comparison failure; retained evidence".into()).into(),
        );
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err("gas-chapter08-completion EMPTY_OUTPUT [REPLICAS=64] [ROOTS=512]".into());
    }
    let reps = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(64);
    let roots = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(512);
    if reps < 2 || roots < 2 {
        return Err("independent integration needs >=2 seed units".into());
    }
    futures_lite::future::block_on(run(&args[0], reps, roots))
}
