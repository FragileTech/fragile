//! Controlled native fitness/decision/component counterfactuals on retained Rust inputs.
//! The conditional branch uses the original independent row law conditioned on
//! agreement outside a fixed exceptional set; no component-size rejection occurs.
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasConfig, Population, Precision,
    cloning::{ClonePlan, accepted_current_components},
    donor::{CompanionBatch, CompanionRequest, CompanionSampler, DonorPool},
    fitness::{FitnessBatch, Standardizer},
    geometry::{AlgorithmicDistance, Distance, InteractionKernel, Kernel},
    random::{RandomStream, Stream},
};
use algorithmic_gas_benchmarks::{
    convergence_chapter09_completion::{constants_from_native, log_component_moment},
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

async fn run(input: &str, output: &str, conditional_draws: usize) -> Result<(), Error> {
    let native = ArchiveStore::open(input)?;
    let mut store = ArchiveStore::new(output)?;
    store.save_json("execution-source",&json!({"runner":include_str!("gas-native-measurement-replacement.rs"),
        "fitness":include_str!("../../../algorithmic-gas/src/fitness.rs"),
        "cloning":include_str!("../../../algorithmic-gas/src/cloning.rs"),
        "donor":include_str!("../../../algorithmic-gas/src/donor.rs"),
        "random":include_str!("../../../algorithmic-gas/src/random.rs"),
        "chapter09":include_str!("../../../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"),
        "source_sha256":sha256_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md").as_path())?,
        "executable_sha256":sha256_file(&std::env::current_exe()?)?,"conditional_draws":conditional_draws,
        "input_archive_index_sha256":sha256_file(&Path::new(input).join("archive-index.json"))?,
        "scope":"Native public fitness, decision and component evaluations; zero new complete engine updates. Conditional draws use exact native row probabilities conditioned on fixed exceptional outcomes and agreement outside E; no size-based rejection."}))?;
    let mut cx = ExecutionContext::new(BackendKind::Cpu, Precision::F64).await?;
    let mut checks = vec![];
    let mut cases = vec![];
    let mut replayed = 0;
    let mut replacements = 0;
    let mut conditional_samples = 0;
    let mut changed_unconditional = 0;
    for entry in native
        .entries()
        .iter()
        .filter(|e| e.tag.ends_with("native-operands"))
    {
        let operands = native.load_json(&entry.path)?;
        let tag = entry.tag.trim_end_matches("-native-operands");
        let raws = operands["raw"]
            .as_array()
            .ok_or("missing native raw rows")?;
        let mut replays = vec![];
        let mut regime_rows = [vec![], vec![]];
        let mut conditional_prototype = [None, None];
        let mut constants_regime = [None, None];
        let mut source_population = None;
        for (rep, raw) in raws.iter().enumerate() {
            let archive_path = raw["native_archive"]
                .as_str()
                .ok_or("missing native archive")?;
            let archive = native.load_archive(archive_path)?;
            let recorded = &archive.steps[0];
            let cfg = &archive.gas_config;
            let mut p = recorded.before.clone();
            p.rewards = recorded.report.pre_clone_rewards.clone();
            let alive = &recorded.report.pre_clone_eligible;
            if p.eligible(false) != *alive {
                return Err("recorded pre-clone eligibility cannot be reconstructed".into());
            }
            let frame = recorded.report.clone_plan.sources[0].frame;
            let pool = DonorPool::freeze(&p, frame, &[], 0, cfg.include_truncated)?;
            if pool.sources != recorded.report.clone_plan.sources {
                return Err("original frozen pool source mismatch".into());
            }
            let step = recorded.report.step;
            let measured = cfg
                .reducer
                .measure(
                    &cfg.distance_donors.distance,
                    &p,
                    &pool,
                    &recorded.report.distance_companions,
                    &mut cx,
                )
                .await?;
            let original_f =
                cfg.fitness
                    .evaluate(&p.rewards, &measured, alive, &p.observations, p.version)?;
            let original_plan = plan(
                cfg,
                &p,
                &pool,
                &recorded.report.cloning_companions,
                &original_f,
                alive,
                cfg.seed,
                step,
            )?;
            let ferror = original_f
                .fitness
                .iter()
                .zip(&recorded.report.pre_clone_fitness.fitness)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            let plan_equal = original_plan == recorded.report.clone_plan;
            compare(
                &mut checks,
                &format!("{tag}/replay-{rep}-fitness"),
                &[218],
                ferror,
                0.,
                true,
                "Original complete native recorded pre-clone fitness replay prerequisite",
                json!({"native_archive":archive_path}),
            );
            compare(
                &mut checks,
                &format!("{tag}/replay-{rep}-plan"),
                &[220],
                f64::from(plan_equal),
                1.,
                true,
                "Original native recorded cloning donors and exact gate-address replay prerequisite",
                json!({"native_archive":archive_path,"seed":cfg.seed,"step":step}),
            );
            if !plan_equal || ferror > 1e-11 {
                return Err(format!("exact original native replay failed {tag} {rep}: fitness {ferror}, plan {plan_equal}, report.step {step}, noise {:?}",recorded.noise.iter().map(|n|n.step).collect::<Vec<_>>()).into());
            }
            replayed += 1;
            replays.push(json!({"native_archive":archive_path,"native_archive_sha256":sha256_file(&Path::new(input).join(archive_path))?,
                "fitness_replay_max_error":ferror,"plan_exact":plan_equal,"report_step":step,"recorded_noise_steps":recorded.noise.iter().map(|n|n.step).collect::<Vec<_>>(),
                "original_distance_companions":recorded.report.distance_companions,"original_cloning_companions":recorded.report.cloning_companions,
                "actual_measurements":measured,"original_fitness_batch":original_f,"original_clone_plan":original_plan}));
            source_population = Some(p.clone());
            for (regime, sigma) in [10., 1.].into_iter().enumerate() {
                if regime == 1 && p.len() < 128 {
                    continue;
                }
                let mut weak = cfg.clone();
                weak.fitness.reward_exponent = 0.01;
                weak.fitness.diversity_exponent = 0.01;
                weak.fitness.diversity_standardizer = Standardizer::Global { sigma_min: sigma };
                weak.cloning_donors.kernel = Kernel::Gaussian { width: 8. };
                let m = alive.iter().filter(|&&a| a).count() as f64 / p.len() as f64;
                let c = constants_from_native(&weak, m)?;
                if (p.len() as f64) < 2. * c.b {
                    return Err(format!(
                        "conditional regime not feasible: {tag} sigma{sigma} N{} B{}",
                        p.len(),
                        c.b
                    )
                    .into());
                }
                constants_regime[regime] = Some(c.clone());
                let seed = 202610059000 + rep as u64 + replacements as u64 * 104729;
                let clone_comp = weak
                    .cloning_donors
                    .sample(
                        CompanionRequest {
                            population: &p,
                            pool: &pool,
                            eligible: &vec![true; p.len()],
                            seed,
                            step,
                            stream: Stream::Cloning,
                        },
                        &mut cx,
                    )
                    .await?;
                let replacement = cfg
                    .distance_donors
                    .sample(
                        CompanionRequest {
                            population: &p,
                            pool: &pool,
                            eligible: alive,
                            seed: seed ^ 0xa38574,
                            step,
                            stream: Stream::Distance,
                        },
                        &mut cx,
                    )
                    .await?;
                let mut rng = RandomStream::new(seed, step, Stream::MeanFieldReference, 0, 901);
                let roots = alive
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &a)| a.then_some(i))
                    .collect::<Vec<_>>();
                let r = roots[rng.index(roots.len())];
                let mut changed_comp = recorded.report.distance_companions.clone();
                changed_comp.indices[r] = replacement.indices[r];
                changed_comp.valid[r] = replacement.valid[r];
                let changed_measure = cfg
                    .reducer
                    .measure(
                        &cfg.distance_donors.distance,
                        &p,
                        &pool,
                        &changed_comp,
                        &mut cx,
                    )
                    .await?;
                let fa = weak.fitness.evaluate(
                    &p.rewards,
                    &measured,
                    alive,
                    &p.observations,
                    p.version,
                )?;
                let fb = weak.fitness.evaluate(
                    &p.rewards,
                    &changed_measure,
                    alive,
                    &p.observations,
                    p.version,
                )?;
                let a = plan(&weak, &p, &pool, &clone_comp, &fa, alive, seed, step)?;
                let b = plan(&weak, &p, &pool, &clone_comp, &fb, alive, seed, step)?;
                let gates = (0..p.len())
                    .map(|i| {
                        RandomStream::new(seed, step, Stream::Accept, i as u64, 0).uniform::<f64>()
                    })
                    .collect::<Vec<_>>();
                let pa = probabilities(&weak, &fa, alive, &pool, step);
                let pb = probabilities(&weak, &fb, alive, &pool, step);
                let wl = donor_law(&weak, &p, &pool)?;
                let qs = (0..p.len())
                    .map(|i| {
                        wl[i]
                            .iter()
                            .enumerate()
                            .map(|(j, w)| w * (pa[i][j] - pb[i][j]).abs())
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>();
                let fdiff = fa
                    .fitness
                    .iter()
                    .zip(&fb.fitness)
                    .map(|(a, b)| (a - b).abs())
                    .collect::<Vec<_>>();
                let max_fdiff = (0..p.len())
                    .filter(|&i| i != r && alive[i])
                    .map(|i| fdiff[i])
                    .fold(0., f64::max);
                let max_q = (0..p.len())
                    .filter(|&i| i != r)
                    .map(|i| qs[i])
                    .fold(0., f64::max);
                let mut c_effective = 0_f64;
                for row in 0..p.len() {
                    for j in 0..wl[row].len() {
                        c_effective = c_effective
                            .max(p.len() as f64 * wl[row][j] * pa[row][j].max(pb[row][j]));
                    }
                }
                compare(
                    &mut checks,
                    &format!("{tag}/sigma{sigma}/rep{rep}/fitness-all"),
                    &[218],
                    max_fdiff,
                    c.l_0 / p.len() as f64,
                    false,
                    "Maximum over every unchanged alive measurement row is compared to its identical RHS; full native recomputed arrays retained",
                    json!({"root":r,"L0":c.l_0,"N":p.len()}),
                );
                compare(
                    &mut checks,
                    &format!("{tag}/sigma{sigma}/rep{rep}/q-all"),
                    &[220],
                    max_q,
                    c.b / p.len() as f64,
                    false,
                    "Maximum over every nonroot integrated native common-donor/common-gate change probability; all q_i retained",
                    json!({"root":r,"B":c.b,"N":p.len()}),
                );
                let (dd, affected, ca, cb) = support(&p, &pool, &a, &b, r)?;
                let changes = a
                    .choices
                    .iter()
                    .zip(&b.choices)
                    .filter(|(a, b)| a.accepted != b.accepted)
                    .count();
                changed_unconditional += changes;
                replacements += 1;
                compare(
                    &mut checks,
                    &format!("{tag}/sigma{sigma}/rep{rep}/permutation"),
                    &[209],
                    f64::from(permutation_guard(&p, &pool, &a, &b, r)?),
                    1.,
                    true,
                    "Counterfactual support with transported physical rows, donors and gate addresses",
                    json!({"root":r}),
                );
                regime_rows[regime].push(json!({"replica":rep,"native_seed":seed,"step":step,"root":r,
                    "baseline_record":archive_path,"old_measurement_companions":recorded.report.distance_companions,
                    "changed_measurement_companions":changed_comp,"old_raw_distance":measured,"new_raw_distance":changed_measure,
                    "native_fitness_batch_a":fa,"native_fitness_batch_b":fb,"common_cloning_companions":clone_comp,
                    "common_gate_uniforms":gates,"native_plan_a":a,"native_plan_b":b,"changed_rows":changes,
                    "affected_rows":affected,"D":dd,"D_squared":dd*dd,"affected_fraction":dd as f64/p.len() as f64,
                    "normalized_squared_influence":(dd*dd) as f64/(p.len()*p.len()) as f64,
                    "components_a":ca,"components_b":cb,"q_i":qs,"fitness_difference_by_row":fdiff,
                    "C_effective_frozen_accepted_edges":c_effective,
                    "effective_constant_scope":"Input and both-measurement-array conditional diagnostic; not substituted for the source global constant"}));
                if conditional_prototype[regime].is_none()
                    && qs.iter().enumerate().any(|(i, &q)| i != r && q > 0.)
                {
                    conditional_prototype[regime] = Some((
                        weak,
                        p.clone(),
                        pool.clone(),
                        alive.to_vec(),
                        fa,
                        fb,
                        pa,
                        pb,
                        wl,
                        r,
                        measured.clone(),
                        changed_measure,
                        qs,
                        seed,
                        step,
                    ));
                }
            }
        }
        store.save_json(
            &format!("{tag}-original-replay"),
            &json!({"population":source_population,"records":replays,
            "original_native_operands_path":entry.path,"source_sha256":entry.sha256}),
        )?;
        for regime in 0..2 {
            if regime_rows[regime].is_empty() {
                continue;
            }
            let c = constants_regime[regime].as_ref().expect("constant record");
            let n = source_population.as_ref().expect("population").len();
            let samples = regime_rows[regime]
                .iter()
                .map(|v| v["D_squared"].as_u64().expect("D²") as f64)
                .collect::<Vec<_>>();
            let stats = statistic(&samples, (n * n) as f64, 1e-6);
            let archive = store.save_json(
                &format!("{tag}-sigma{}-unconditional", [10, 1][regime]),
                &json!({"parameters":c,"counterfactuals":regime_rows[regime],"statistics":stats}),
            )?;
            compare(
                &mut checks,
                &format!("{tag}/sigma{regime}/unconditional-mean"),
                &[209],
                stats["upper_confidence"].as_f64().expect("CI").ln(),
                c.log_a_d,
                false,
                "Native controlled primitives, uniformly selected alive root; log mean-square confidence upper bound versus log A_D; fixed S, independent recorded baseline seed units",
                json!({"statistics":stats,"lossless_operands":archive,"log_A_D":c.log_a_d}),
            );
            compare(
                &mut checks,
                &format!("{tag}/sigma{regime}/all-root-pathwise-certificate"),
                &[209],
                ((n * n) as f64).ln(),
                c.log_a_d,
                false,
                "Finite-N exact consequence D_r²≤N²≤A_D for every root and realization; does not claim a uniform-N theorem from data",
                json!({"N":n,"log_A_D":c.log_a_d}),
            );
            let Some((
                weak,
                p,
                pool,
                alive,
                fa,
                fb,
                pa,
                pb,
                wl,
                r,
                measure_a,
                measure_b,
                qs,
                seed,
                step,
            )) = conditional_prototype[regime].take()
            else {
                return Err(format!(
                    "no informative changed-event conditional stratum {tag} regime {regime}"
                )
                .into());
            };
            let i = (0..n)
                .filter(|&i| i != r)
                .max_by(|&i, &j| qs[i].total_cmp(&qs[j]))
                .expect("nonroot");
            let mut mismatch_rng =
                RandomStream::new(seed, step, Stream::MeanFieldReference, i as u64, 1001);
            let mismatch_weights = wl[i]
                .iter()
                .enumerate()
                .map(|(j, &w)| w * (pa[i][j] - pb[i][j]).abs())
                .collect::<Vec<_>>();
            let donor_i = categorical(&mismatch_weights, &mut mismatch_rng);
            let gate_i = pa[i][donor_i].min(pb[i][donor_i])
                + (pa[i][donor_i] - pb[i][donor_i]).abs() * mismatch_rng.uniform::<f64>();
            let mut r_rng =
                RandomStream::new(seed, step, Stream::MeanFieldReference, r as u64, 1002);
            let donor_r = categorical(&wl[r], &mut r_rng);
            let gate_r = r_rng.uniform::<f64>();
            if (gate_i < pa[i][donor_i]) == (gate_i < pb[i][donor_i]) {
                return Err("forced changed event failed".into());
            }
            let exceptional = [r, i];
            let q = exceptional.len();
            let mut fixed_seeds = vec![
                r,
                i,
                pool.sources[donor_r].slot as usize,
                pool.sources[donor_i].slot as usize,
            ];
            fixed_seeds.sort_unstable();
            fixed_seeds.dedup();
            let log_bound = (9. * (q * q) as f64).ln() + log_component_moment(2. * c.c, 2)?;
            compare(
                &mut checks,
                &format!("{tag}/regime{regime}/conditional-feasible"),
                &[226],
                2. * c.b,
                n as f64,
                false,
                "Explicit N≥2B analytic prerequisite evaluated on native parameters",
                json!({"B":c.b,"N":n}),
            );
            let max_remaining_q = (0..n)
                .filter(|i| !exceptional.contains(i))
                .map(|i| qs[i])
                .fold(0., f64::max);
            let mut max_common_edge = 0_f64;
            let mut frozen_c_effective = 0_f64;
            for row in 0..n {
                for j in 0..wl[row].len() {
                    frozen_c_effective =
                        frozen_c_effective.max(n as f64 * wl[row][j] * pa[row][j].max(pb[row][j]));
                    if !exceptional.contains(&row) {
                        max_common_edge = max_common_edge
                            .max(wl[row][j] * pa[row][j].min(pb[row][j]) / (1. - qs[row]));
                    }
                }
            }
            compare(
                &mut checks,
                &format!("{tag}/regime{regime}/conditional-q-all"),
                &[226],
                max_remaining_q,
                0.5,
                false,
                "Maximum over every remaining conditional row; full q_i saved, exact local agreement acceptance≥1/2",
                json!({"B_over_N":c.b/n as f64}),
            );
            compare(
                &mut checks,
                &format!("{tag}/regime{regime}/conditional-edge-all"),
                &[226],
                max_common_edge,
                2. * c.c / n as f64,
                false,
                "Maximum over every exact conditional native common accepted edge; complete probability matrices saved",
                json!({"C":c.c,"N":n}),
            );
            let metadata=store.save_json(&format!("{tag}-regime{regime}-conditional-fixed"),&json!({"native_config":weak,
                "population":p,"pool_sources":pool.sources,"current_frame":pool.current_frame,"measurements_a":measure_a,"measurements_b":measure_b,
                "fitness_batch_a":fa,"fitness_batch_b":fb,"native_probabilities_a":pa,"native_probabilities_b":pb,
                "native_weighted_donor_laws":wl,"q_i":qs,"constants":c,"root":r,"E":exceptional,"Q":q,"fixed_seed_slots":fixed_seeds,
                "fixed_exceptional_outcomes":[{"row":r,"donor":donor_r,"uniform":gate_r,"accepted_a":gate_r<pa[r][donor_r],"accepted_b":gate_r<pb[r][donor_r]},
                    {"row":i,"donor":donor_i,"uniform":gate_i,"accepted_a":gate_i<pa[i][donor_i],"accepted_b":gate_i<pb[i][donor_i]}],
                "conditional_sampling":"For i≠r force mismatch using categorical weight w_ij|pA_ij−pB_ij| then uniform between thresholds; fix these exceptional outcomes once. Each other row independently samples its native donor law and common uniform until the two acceptances agree. No graph- or size-dependent rejection.",
                "log_bound":log_bound,"draws":conditional_draws,
                "C_effective_frozen":frozen_c_effective,"C_effective_common_conditioned":n as f64*max_common_edge,
                "effective_constant_scope":"Frozen input, both measurement arrays and E; diagnostic only, no global moment theorem substituted"}))?;
            let mut conditional_values = vec![];
            let mut chunk = vec![];
            for draw in 0..conditional_draws {
                let mut donors = vec![0u32; n];
                let mut gates = vec![0.; n];
                let mut attempts = vec![0usize; n];
                donors[r] = donor_r as u32;
                donors[i] = donor_i as u32;
                gates[r] = gate_r;
                gates[i] = gate_i;
                for row in 0..n {
                    if exceptional.contains(&row) {
                        continue;
                    }
                    let mut rng = RandomStream::new(
                        seed ^ 0x79ade14,
                        step,
                        Stream::MeanFieldReference,
                        row as u64,
                        10_000 + draw as u64,
                    );
                    loop {
                        attempts[row] += 1;
                        let j = categorical(&wl[row], &mut rng);
                        let u = rng.uniform::<f64>();
                        if (u < pa[row][j]) == (u < pb[row][j]) {
                            donors[row] = j as u32;
                            gates[row] = u;
                            break;
                        }
                    }
                }
                let batch = CompanionBatch {
                    rows: n,
                    count: 1,
                    indices: donors.clone(),
                    valid: vec![true; n],
                    mutual: false,
                };
                let mut a = plan(&weak, &p, &pool, &batch, &fa, &alive, seed, step)?;
                let mut b = plan(&weak, &p, &pool, &batch, &fb, &alive, seed, step)?;
                for row in 0..n {
                    a.choices[row].accepted = gates[row] < pa[row][donors[row] as usize];
                    b.choices[row].accepted = gates[row] < pb[row][donors[row] as usize];
                }
                let (dd, affected, ca, cb) = support(&p, &pool, &a, &b, r)?;
                let mut common = a.clone();
                for &row in &exceptional {
                    common.choices[row].accepted = false;
                }
                let forest = accepted_current_components(&p, &pool, &common)?;
                let mut forest_support = vec![false; n];
                for &s in &fixed_seeds {
                    forest_support[s] = true;
                }
                for members in &forest {
                    if members.iter().any(|j| fixed_seeds.contains(j)) {
                        for &j in members {
                            forest_support[j] = true;
                        }
                    }
                }
                let contains = affected.iter().all(|&j| forest_support[j]);
                compare(
                    &mut checks,
                    &format!("{tag}/regime{regime}/conditional{draw}-seedunion"),
                    &[226],
                    f64::from(contains),
                    1.,
                    true,
                    "Native old/new component support is contained in the common forest components meeting fixed exceptional recipient/donor seeds",
                    json!({"fixed_metadata":metadata,"draw":draw}),
                );
                if !contains {
                    return Err("affected support escapes fixed common-forest seed union".into());
                }
                let common_edges = common.choices.iter().filter(|c| c.accepted).count();
                let forest_edges = forest.iter().map(|c| c.len() - 1).sum::<usize>();
                compare(
                    &mut checks,
                    &format!("{tag}/regime{regime}/conditional{draw}-forest"),
                    &[226],
                    common_edges as f64,
                    forest_edges as f64,
                    true,
                    "Actual native common accepted graph is an ordered forest after removal of E outgoing edges",
                    json!({"fixed_metadata":metadata,"draw":draw}),
                );
                if draw == 0 {
                    compare(
                        &mut checks,
                        &format!("{tag}/regime{regime}/conditional-permutation"),
                        &[226],
                        f64::from(permutation_guard(&p, &pool, &a, &b, r)?),
                        1.,
                        true,
                        "Physical permutation and transported conditional donor/gate outcomes preserve influence support",
                        json!({"fixed_metadata":metadata}),
                    );
                }
                conditional_values.push((dd * dd) as f64);
                conditional_samples += 1;
                chunk.push(json!({"draw":draw,"sample_stream":Stream::MeanFieldReference,"seed":seed^0x79ade14,"step":step,"substep":10_000+draw as u64,
                    "companion_batch":batch,"common_gate_uniforms":gates,"local_rejection_attempts":attempts,
                    "native_plan_a":a,"native_plan_b":b,"common_forest_plan":common,"common_forest_components":forest,
                    "components_a":ca,"components_b":cb,"affected_rows":affected,"D":dd,"D_squared":dd*dd,
                    "affected_fraction":dd as f64/n as f64,"normalized_squared_influence":(dd*dd) as f64/(n*n) as f64}));
                if chunk.len() == 64 || draw + 1 == conditional_draws {
                    store.save_json(
                        &format!("{tag}-regime{regime}-conditional-through{draw}"),
                        &json!({"fixed_metadata":metadata,"draws":chunk}),
                    )?;
                    chunk.clear();
                }
            }
            let stats = statistic(&conditional_values, (n * n) as f64, 1e-6);
            compare(
                &mut checks,
                &format!("{tag}/regime{regime}/conditional-mean"),
                &[226],
                stats["upper_confidence"].as_f64().expect("CI").ln(),
                log_bound,
                false,
                "Independent conditional native row-graph preparations for fixed both measurement arrays, E, exceptional outcomes; log mean-square confidence upper versus log(9Q²M2(2C))",
                json!({"statistics":stats,"lossless_operands":metadata,"log_bound":log_bound,"Q":q}),
            );
            compare(
                &mut checks,
                &format!("{tag}/regime{regime}/conditional-pathwise-certificate"),
                &[226],
                ((n * n) as f64).ln(),
                log_bound,
                false,
                "Finite-N exact D²≤N² certificate implies this conditional moment bound for all graph realizations in the declared stratum",
                json!({"N":n,"Q":q,"log_bound":log_bound}),
            );
            cases.push(json!({"tag":tag,"regime":regime,"sigma_s":([10.,1.][regime]),"N":n,"constants":c,
                "unconditional_statistics":statistic(&samples,(n*n) as f64,1e-6),"conditional_statistics":stats,
                "conditional_fixed_metadata":metadata,"unconditional_data":archive,"conditional_Q":q,
                "C_effective_frozen":frozen_c_effective,"C_effective_common_conditioned":n as f64*max_common_edge,
                "conditional_observed_normalized_D_squared":conditional_values.iter().sum::<f64>()/conditional_values.len() as f64/(n*n) as f64,
                "unconditional_observed_normalized_D_squared":samples.iter().sum::<f64>()/samples.len() as f64/(n*n) as f64,
                "global_constant_bound_character":"Loose analytic guarantee; not informative empirical convergence",
                "effective_constant_scope":"Frozen-input diagnostic; neither a global constant nor a proven convergence rate"}));
        }
    }
    let failed = checks
        .iter()
        .filter(|c| c["passed"] == false)
        .map(|c| c["id"].clone())
        .collect::<Vec<_>>();
    let report = json!({"scope":"Actual native public-stage counterfactuals, not new complete Gas updates or QSD observations",
        "source_sha256":sha256_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md").as_path())?,
        "runner_source_sha256":sha256_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/gas-native-measurement-replacement.rs").as_path())?,
        "executable_sha256":sha256_file(&std::env::current_exe()?)?,
        "summary":{"original_native_records_exactly_replayed":replayed,"native_measurement_replacements":replacements,
            "conditional_graph_draws":conditional_samples,"informative_unconditional_changed_rows":changed_unconditional,
            "conditional_strata":cases.len(),"checks":checks.len(),"failures":failed.len(),"new_complete_engine_updates":0},
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
    let verification = store.verify(true)?;
    fs::write(
        Path::new(output).join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!("{}", report["summary"]);
    if !failed.is_empty() {
        return Err("native replacement comparison failure; raw evidence retained".into());
    }
    Ok(())
}
fn main() -> Result<(), Error> {
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    if a.len() < 2 {
        return Err(
            "gas-native-measurement-replacement NATIVE08_DIR EMPTY_OUTPUT [CONDITIONAL_DRAWS=512]"
                .into(),
        );
    }
    let draws = a.get(2).map(|s| s.parse()).transpose()?.unwrap_or(512);
    if draws < 2 {
        return Err("at least two independent conditional preparations required".into());
    }
    futures_lite::future::block_on(run(&a[0], &a[1], draws))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_row_rejection_has_the_exact_agreement_law() {
        let weights: [f64; 2] = [0.3, 0.7];
        let a: [f64; 2] = [0.2, 0.8];
        let b: [f64; 2] = [0.4, 0.6];
        let q = weights
            .iter()
            .enumerate()
            .map(|(j, &w)| w * (a[j] - b[j]).abs())
            .sum::<f64>();
        let expected = weights
            .iter()
            .enumerate()
            .map(|(j, &w)| w * a[j].min(b[j]))
            .sum::<f64>()
            / (1. - q);
        let mut rng = RandomStream::new(89102, 1, Stream::MeanFieldReference, 0, 71);
        let samples = 20_000;
        let mut accepted = 0;
        for _ in 0..samples {
            loop {
                let j = categorical(&weights, &mut rng);
                let u = rng.uniform::<f64>();
                if (u < a[j]) == (u < b[j]) {
                    accepted += usize::from(u < a[j]);
                    break;
                }
            }
        }
        let observed = accepted as f64 / samples as f64;
        let tolerance = ((2e6_f64).ln() / (2. * samples as f64)).sqrt();
        assert!((observed - expected).abs() <= tolerance);
        assert!((q - 0.2).abs() < 1e-12);
    }

    #[test]
    fn conditional_native_component_support_transports_with_physical_rows() {
        futures_lite::future::block_on(async {
            let cfg = algorithmic_gas_benchmarks::RunConfig {
                walkers: 8,
                dimensions: 1,
                gas: GasConfig::euclidean(1, 0.04).expect("native config"),
                ..Default::default()
            };
            let gas = cfg.build::<f64>().await.expect("native swarm");
            let p = gas.population();
            let alive = p.eligible(false);
            let pool = DonorPool::freeze(p, 0, &[], 0, false).expect("native pool");
            let companions = CompanionBatch {
                rows: 8,
                count: 1,
                indices: (0..8).map(|i| ((i + 1) % 8) as u32).collect(),
                valid: vec![true; 8],
                mutual: false,
            };
            let da = (0..8).map(|i| 0.1 + 0.1 * i as f64).collect::<Vec<_>>();
            let mut db = da.clone();
            db[0] = 1.3;
            let fa = cfg
                .gas
                .fitness
                .evaluate(&p.rewards, &da, &alive, &p.observations, p.version)
                .expect("fitness a");
            let fb = cfg
                .gas
                .fitness
                .evaluate(&p.rewards, &db, &alive, &p.observations, p.version)
                .expect("fitness b");
            let mut a = plan(&cfg.gas, p, &pool, &companions, &fa, &alive, 17, 1).expect("plan a");
            let mut b = plan(&cfg.gas, p, &pool, &companions, &fb, &alive, 17, 1).expect("plan b");
            let i = (0..8)
                .find(|&i| {
                    (a.choices[i].probability.unwrap() - b.choices[i].probability.unwrap()).abs()
                        > 1e-8
                })
                .expect("informative native gate");
            let u = (a.choices[i].probability.unwrap() + b.choices[i].probability.unwrap()) / 2.;
            a.choices[i].accepted = u < a.choices[i].probability.unwrap();
            b.choices[i].accepted = u < b.choices[i].probability.unwrap();
            assert_ne!(a.choices[i].accepted, b.choices[i].accepted);
            assert!(permutation_guard(p, &pool, &a, &b, 0).expect("native permutation"));
            let (d, _, _, _) = support(p, &pool, &a, &b, 0).expect("native affected support");
            assert!((2..=8).contains(&d));
        });
    }
}
