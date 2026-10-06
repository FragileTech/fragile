//! Original-parameter native measurement replacement on retained complete Rust inputs.
//! Original cloning donors and native gate addresses are held fixed; only one
//! independently sampled measurement companion is replaced.
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasConfig, Population, Precision,
    cloning::{ClonePlan, accepted_current_components},
    donor::{CompanionBatch, CompanionRequest, CompanionSampler, DonorPool},
    fitness::FitnessBatch,
    geometry::{AlgorithmicDistance, Distance, InteractionKernel},
    random::{RandomStream, Stream},
};
use algorithmic_gas_benchmarks::{
    convergence_chapter09_completion::constants_from_native,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{fs, path::Path};
type Error = Box<dyn std::error::Error>;
type ComponentSupport = (usize, Vec<usize>, Vec<Vec<usize>>, Vec<Vec<usize>>);

fn categorical(p: &[f64], rng: &mut RandomStream) -> usize {
    let mut u = rng.uniform::<f64>() * p.iter().sum::<f64>();
    for (j, &v) in p.iter().enumerate() {
        u -= v;
        if u < 0. {
            return j;
        }
    }
    p.iter()
        .rposition(|&x| x > 0.)
        .expect("positive categorical mass")
}

#[allow(clippy::too_many_arguments)] // Mirrors the native CloneDecision::plan inputs.
fn plan(
    cfg: &GasConfig,
    p: &Population<f64>,
    pool: &DonorPool<f64>,
    companions: &CompanionBatch,
    f: &FitnessBatch<f64>,
    alive: &[bool],
    seed: u64,
    step: u64,
) -> Result<ClonePlan, Error> {
    let donor_fitness = pool
        .sources
        .iter()
        .map(|s| f.fitness[s.slot as usize])
        .collect::<Vec<_>>();
    Ok(cfg.clone_decision.plan(
        p,
        pool,
        companions,
        &f.fitness,
        &donor_fitness,
        alive,
        seed,
        step,
    )?)
}

fn probabilities(
    cfg: &GasConfig,
    f: &FitnessBatch<f64>,
    alive: &[bool],
    pool: &DonorPool<f64>,
    step: u64,
) -> Vec<Vec<f64>> {
    (0..alive.len())
        .map(|i| {
            pool.sources
                .iter()
                .map(|s| {
                    if alive[i] {
                        cfg.clone_decision.acceptance_probability(
                            step,
                            f.fitness[i],
                            f.fitness[s.slot as usize],
                        )
                    } else {
                        1.
                    }
                })
                .collect()
        })
        .collect()
}

fn donor_law(
    cfg: &GasConfig,
    p: &Population<f64>,
    pool: &DonorPool<f64>,
) -> Result<Vec<Vec<f64>>, Error> {
    let module = &cfg.cloning_donors;
    let kind = <Distance as AlgorithmicDistance<f64>>::comparison_kind(&module.distance);
    let mut laws = vec![];
    for i in 0..p.len() {
        let mut law = vec![];
        for (j, s) in pool.sources.iter().enumerate() {
            let w = if s.slot as usize == i {
                0.
            } else {
                let dist = module.distance.compare(
                    &p.observations,
                    i,
                    &pool.population.observations,
                    j,
                )?;
                module.kernel.log_weight(dist, kind)?.exp()
            };
            law.push(w);
        }
        let z = law.iter().sum::<f64>();
        if z <= 0. {
            return Err("positive nonself donor normalizer required".into());
        }
        for w in &mut law {
            *w /= z;
        }
        laws.push(law);
    }
    Ok(laws)
}

fn support(
    p: &Population<f64>,
    pool: &DonorPool<f64>,
    a: &ClonePlan,
    b: &ClonePlan,
    r: usize,
) -> Result<ComponentSupport, Error> {
    let ca = accepted_current_components(p, pool, a)?;
    let cb = accepted_current_components(p, pool, b)?;
    let mut seeds = vec![false; p.len()];
    seeds[r] = true;
    for i in 0..p.len() {
        if a.choices[i].accepted != b.choices[i].accepted {
            seeds[i] = true;
            for c in [&a.choices[i], &b.choices[i]] {
                if c.accepted {
                    seeds[pool.sources[c.donors[0].pool_index as usize].slot as usize] = true;
                }
            }
        }
    }
    let mut touched = seeds.clone();
    for component in ca.iter().chain(&cb) {
        if component.iter().any(|&i| seeds[i]) {
            for &i in component {
                touched[i] = true;
            }
        }
    }
    Ok((
        touched.iter().filter(|&&x| x).count(),
        touched
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| v.then_some(i))
            .collect(),
        ca,
        cb,
    ))
}

fn statistic(xs: &[f64], maximum: f64, failure: f64) -> Value {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let allowance = maximum * ((2. / failure).ln() / (2. * xs.len() as f64)).sqrt();
    json!({"observed_mean":mean,"independent_samples":xs.len(),"range_upper":maximum,
           "failure_budget":failure,"hoeffding_allowance":allowance,
           "upper_confidence":(mean+allowance).min(maximum),"observed_maximum":xs.iter().copied().fold(0.,f64::max)})
}

#[allow(clippy::too_many_arguments)] // Each evidence row preserves its scope and source operands.
fn compare(
    checks: &mut Vec<Value>,
    id: &str,
    source: &[usize],
    lhs: f64,
    rhs: f64,
    equality: bool,
    scope: &str,
    operands: Value,
) {
    let allowance = 2e-10 * lhs.abs().max(rhs.abs()).max(1.);
    let delta = if equality {
        (lhs - rhs).abs()
    } else {
        lhs - rhs
    };
    checks.push(json!({"id":id,"source_expressions":source.iter().map(|i|format!("chapter09-expression-{i:04}")).collect::<Vec<_>>(),
        "lhs":lhs,"rhs":rhs,"relation":if equality {"equality"} else {"upper_bound"},
        "allowance":allowance,"passed":delta<=allowance,"scope":scope,"operands":operands}));
}

fn permutation_guard(
    p: &Population<f64>,
    pool: &DonorPool<f64>,
    a: &ClonePlan,
    b: &ClonePlan,
    r: usize,
) -> Result<bool, Error> {
    // Transport realized source/gate addresses with physical rows. Same raw RNG
    // addresses at different permuted rows would represent a different coupling.
    let n = p.len();
    let mut pp = p.clone();
    for (name, field) in &p.observations.fields {
        let values = (0..n)
            .rev()
            .flat_map(|i| field.row(i).expect("valid row").iter().copied())
            .collect();
        pp.observations.fields.insert(
            name.clone(),
            algorithmic_gas::TensorBatch::new(n, field.item_shape().to_vec(), values)?,
        );
    }
    pp.generations.reverse();
    pp.validity.reverse();
    pp.rewards.raw.reverse();
    pp.rewards.valid.reverse();
    let qp = DonorPool::freeze(&pp, pool.current_frame, &[], 0, false)?;
    let reverse = |pl: &ClonePlan| {
        let mut z = pl.clone();
        z.choices.reverse();
        z.sources = qp.sources.clone();
        for (old, new) in pl.choices.iter().zip(z.choices.iter_mut().rev()) {
            let oldslot = pool.sources[old.donors[0].pool_index as usize].slot as usize;
            new.donors[0].pool_index = qp
                .current_index(n - 1 - oldslot)
                .expect("transported donor");
        }
        z
    };
    let aa = reverse(a);
    let bb = reverse(b);
    let (d, rows, _, _) = support(p, pool, a, b, r)?;
    let (dd, rr, _, _) = support(&pp, &qp, &aa, &bb, n - 1 - r)?;
    let mut transported = rows.iter().map(|i| n - 1 - i).collect::<Vec<_>>();
    transported.sort_unstable();
    Ok(d == dd && transported == rr)
}

async fn run(input: &str, output: &str) -> Result<(), Error> {
    let native = ArchiveStore::open(input)?;
    let mut store = ArchiveStore::new(output)?;
    let book = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md");
    store.save_json("execution-source",&json!({"runner":include_str!("gas-native-original-measurement-replacement.rs"),
        "coupling_primitives_parent_snapshot":include_str!("gas-native-measurement-replacement.rs"),
        "native_fitness":include_str!("../../../algorithmic-gas/src/fitness.rs"),
        "native_cloning":include_str!("../../../algorithmic-gas/src/cloning.rs"),
        "native_donor":include_str!("../../../algorithmic-gas/src/donor.rs"),
        "source":include_str!("../../../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"),
        "source_sha256":sha256_file(&book)?,"executable_sha256":sha256_file(&std::env::current_exe()?)?,
        "input_index_sha256":sha256_file(&Path::new(input).join("archive-index.json"))?,
        "scope":"Original archived parameters and original native clonedonors/native accept seed-step; independent new original-law measurement draw. Public-stage counterfactuals only, zero new complete engine updates."}))?;
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    let mut checks = vec![];
    let mut cases = vec![];
    let mut samples_total = 0;
    let mut changed_total = 0;
    let mut changed_events = 0;
    for entry in native
        .entries()
        .iter()
        .filter(|e| e.tag.ends_with("native-operands"))
    {
        let payload = native.load_json(&entry.path)?;
        let tag = entry.tag.trim_end_matches("-native-operands");
        let mut rows = vec![];
        let mut constants = None;
        let mut frozen = None;
        for (rep, raw) in payload["raw"]
            .as_array()
            .ok_or("missing native raw records")?
            .iter()
            .enumerate()
        {
            let original_path = raw["native_archive"]
                .as_str()
                .ok_or("missing complete native archive")?;
            let archive = native.load_archive(original_path)?;
            let rec = &archive.steps[0];
            let cfg = &archive.gas_config;
            let mut p = rec.before.clone();
            p.rewards = rec.report.pre_clone_rewards.clone();
            let alive = &rec.report.pre_clone_eligible;
            let frame = rec.report.clone_plan.sources[0].frame;
            if p.eligible(false) != *alive {
                return Err("original preclone eligibility mismatch".into());
            }
            let pool = DonorPool::freeze(&p, frame, &[], 0, cfg.include_truncated)?;
            if pool.sources != rec.report.clone_plan.sources {
                return Err("original pool-source reconstruction mismatch".into());
            }
            let step = rec.report.step;
            let n = p.len();
            let da = cfg
                .reducer
                .measure(
                    &cfg.distance_donors.distance,
                    &p,
                    &pool,
                    &rec.report.distance_companions,
                    &mut cx,
                )
                .await?;
            let fa = cfg
                .fitness
                .evaluate(&p.rewards, &da, alive, &p.observations, p.version)?;
            let a = plan(
                cfg,
                &p,
                &pool,
                &rec.report.cloning_companions,
                &fa,
                alive,
                cfg.seed,
                step,
            )?;
            let ferror = fa
                .fitness
                .iter()
                .zip(&rec.report.pre_clone_fitness.fitness)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            if a != rec.report.clone_plan || ferror > 1e-11 {
                return Err(format!("original native replay failed {tag} replica{rep}").into());
            }
            compare(
                &mut checks,
                &format!("{tag}/rep{rep}/original-replay"),
                &[209],
                f64::from(a == rec.report.clone_plan),
                1.,
                true,
                "Exact original native plan baseline prerequisite, not a moment inequality",
                json!({"archive":original_path,"fitness_error":ferror}),
            );
            let seed = 202610069000 + samples_total as u64 * 104729;
            let fresh = cfg
                .distance_donors
                .sample(
                    CompanionRequest {
                        population: &p,
                        pool: &pool,
                        eligible: alive,
                        seed,
                        step,
                        stream: Stream::Distance,
                    },
                    &mut cx,
                )
                .await?;
            let root_weights = alive.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
            let mut rng = RandomStream::new(seed, step, Stream::MeanFieldReference, 0, 913);
            let r = categorical(&root_weights, &mut rng);
            let mut donor_batch = rec.report.distance_companions.clone();
            donor_batch.indices[r] = fresh.indices[r];
            donor_batch.valid[r] = fresh.valid[r];
            let db = cfg
                .reducer
                .measure(
                    &cfg.distance_donors.distance,
                    &p,
                    &pool,
                    &donor_batch,
                    &mut cx,
                )
                .await?;
            let fb = cfg
                .fitness
                .evaluate(&p.rewards, &db, alive, &p.observations, p.version)?;
            let b = plan(
                cfg,
                &p,
                &pool,
                &rec.report.cloning_companions,
                &fb,
                alive,
                cfg.seed,
                step,
            )?;
            let c =
                constants_from_native(cfg, alive.iter().filter(|&&v| v).count() as f64 / n as f64)?;
            let pa = probabilities(cfg, &fa, alive, &pool, step);
            let pb = probabilities(cfg, &fb, alive, &pool, step);
            let w = donor_law(cfg, &p, &pool)?;
            let qi = (0..n)
                .map(|i| {
                    w[i].iter()
                        .enumerate()
                        .map(|(j, &v)| v * (pa[i][j] - pb[i][j]).abs())
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let difference = fa
                .fitness
                .iter()
                .zip(&fb.fitness)
                .map(|(a, b)| (a - b).abs())
                .collect::<Vec<_>>();
            let max_difference = (0..n)
                .filter(|&i| i != r && alive[i])
                .map(|i| difference[i])
                .fold(0., f64::max);
            let max_q = (0..n).filter(|&i| i != r).map(|i| qi[i]).fold(0., f64::max);
            compare(
                &mut checks,
                &format!("{tag}/rep{rep}/fitness-all"),
                &[218],
                max_difference,
                c.l_0 / n as f64,
                false,
                "Original parameter native fitness: maximum over every unchanged alive row, full arrays retained",
                json!({"L0":c.l_0,"N":n,"root":r}),
            );
            compare(
                &mut checks,
                &format!("{tag}/rep{rep}/q-all"),
                &[220],
                max_q,
                c.b / n as f64,
                false,
                "Original parameter exact common-donor/common-native-uniform mismatch law: maximum over every nonroot row",
                json!({"B":c.b,"N":n,"root":r}),
            );
            let (d, affected, ca, cb) = support(&p, &pool, &a, &b, r)?;
            let changed_rows = a
                .choices
                .iter()
                .zip(&b.choices)
                .enumerate()
                .filter_map(|(i, (a, b))| (a.accepted != b.accepted).then_some(i))
                .collect::<Vec<_>>();
            let changed = !changed_rows.is_empty();
            changed_events += usize::from(changed);
            changed_total += changed_rows.len();
            samples_total += 1;
            compare(
                &mut checks,
                &format!("{tag}/rep{rep}/permutation"),
                &[209],
                f64::from(permutation_guard(&p, &pool, &a, &b, r)?),
                1.,
                true,
                "Permutation of physical input and transport of original donor/gate addresses preserve conservative affected support",
                json!({"root":r}),
            );
            let mut ceff = 0_f64;
            for i in 0..n {
                for j in 0..w[i].len() {
                    ceff = ceff.max(n as f64 * w[i][j] * pa[i][j].max(pb[i][j]));
                }
            }
            let gates = (0..n)
                .map(|i| {
                    RandomStream::new(cfg.seed, step, Stream::Accept, i as u64, 0).uniform::<f64>()
                })
                .collect::<Vec<_>>();
            rows.push(json!({"replica":rep,"root":r,"original_archive":original_path,"original_archive_sha256":sha256_file(&Path::new(input).join(original_path))?,
                "original_native_seed":cfg.seed,"replacement_native_seed":seed,"step":step,"original_native_config":cfg,
                "old_measurement_companions":rec.report.distance_companions,"new_measurement_companions":donor_batch,
                "old_raw_distance":da,"new_raw_distance":db,"native_fitness_a":fa,"native_fitness_b":fb,
                "original_cloning_companions":rec.report.cloning_companions,"common_native_gate_uniforms":gates,
                "native_plan_a":a,"native_plan_b":b,"q_i":qi,"fitness_difference_by_row":difference,
                "changed_acceptance_rows":changed_rows,"affected_rows":affected,"D":d,"D_squared":d*d,
                "affected_fraction":d as f64/n as f64,"normalized_squared_influence":(d*d) as f64/(n*n) as f64,
                "components_a":ca,"components_b":cb,"C_effective_frozen":ceff,
                "effective_constant_scope":"Input/both-measurement-array conditional diagnostic, not substituted into source global moment bound"}));
            constants = Some(c);
            frozen = Some(p);
        }
        let c = constants.as_ref().ok_or("no native records")?;
        let n = frozen.as_ref().expect("swarm").len();
        let ds = rows
            .iter()
            .map(|r| r["D_squared"].as_u64().expect("D²") as f64)
            .collect::<Vec<_>>();
        let stats = statistic(&ds, (n * n) as f64, 1e-6);
        let fraction = rows
            .iter()
            .map(|r| r["affected_fraction"].as_f64().expect("fraction"))
            .sum::<f64>()
            / rows.len() as f64;
        let norm_square = ds.iter().sum::<f64>() / ds.len() as f64 / (n * n) as f64;
        let changed_rows = rows
            .iter()
            .map(|r| {
                r["changed_acceptance_rows"]
                    .as_array()
                    .expect("changed rows")
                    .len()
            })
            .sum::<usize>();
        let archived = store.save_json(
            &format!("{tag}-original-parameter-replacements"),
            &json!({"population":frozen,"constants":c,"rows":rows,"statistics":stats}),
        )?;
        compare(
            &mut checks,
            &format!("{tag}/unconditional-mean"),
            &[209],
            stats["upper_confidence"].as_f64().expect("CI").ln(),
            c.log_a_d,
            false,
            "Original unchanged algorithm parameters and fixed physical input; independent original native baseline/replacement seed units; log mean-square confidence upper versus log A_D",
            json!({"statistics":stats,"lossless_operands":archived,"log_A_D":c.log_a_d}),
        );
        compare(
            &mut checks,
            &format!("{tag}/all-root-pathwise-certificate"),
            &[209],
            ((n * n) as f64).ln(),
            c.log_a_d,
            false,
            "Finite-N exact D²≤N²≤A_D suffices for all roots; source small-N branch is retained and not presented as informative convergence",
            json!({"N":n,"B":c.b,"N_lt_2B":(n as f64)<2.*c.b}),
        );
        cases.push(json!({"tag":tag,"N":n,"constants":c,"statistics":stats,"lossless_operands":archived,
            "changed_acceptance_rows":changed_rows,"observed_mean_affected_fraction":fraction,"observed_mean_normalized_squared_influence":norm_square,
            "scope":"Original parameters, native primitive counterfactual; not a new full engine update or an observed QSD law"}));
    }
    let failed = checks
        .iter()
        .filter(|c| c["passed"] == false)
        .map(|c| c["id"].clone())
        .collect::<Vec<_>>();
    let report = json!({"source_sha256":sha256_file(&book)?,"runner_source_sha256":sha256_file(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/gas-native-original-measurement-replacement.rs"))?,
        "executable_sha256":sha256_file(&std::env::current_exe()?)?,"summary":{"original_native_records_exactly_replayed":samples_total,
        "original_parameter_native_measurement_replacements":samples_total,"informative_changed_counterfactuals":changed_events,
        "informative_changed_acceptance_rows":changed_total,"cases":cases.len(),"checks":checks.len(),"failures":failed.len(),"new_complete_engine_updates":0},
        "cases":cases,"checks":checks,"failed_check_ids":failed});
    store.save_json("completion-report", &report)?;
    fs::write(
        Path::new(output).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    store.finish(if failed.is_empty() {
        "complete"
    } else {
        "comparisons_failed"
    })?;
    let verified = store.verify(true)?;
    fs::write(
        Path::new(output).join("verification.json"),
        serde_json::to_vec_pretty(&verified)?,
    )?;
    println!("{}", report["summary"]);
    if !failed.is_empty() {
        return Err("original native replacement inequality failure; evidence retained".into());
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("gas-native-original-measurement-replacement NATIVE08_DIR EMPTY_OUTPUT".into());
    }
    futures_lite::future::block_on(run(&args[0], &args[1]))
}
