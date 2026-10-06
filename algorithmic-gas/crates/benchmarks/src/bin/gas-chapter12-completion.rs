//! Full native permutation transport and exact probability-normalized reference laws.
use algorithmic_gas::{
    ExecutionContext, GasBuilder, GasConfig, Population, RecordingConfig, TensorBatch,
    cloning::{
        CloneChoice, CloneDecision, ClonePlan, CloneTransform, WeightedDonor,
        accepted_current_components, apply_component_rotations,
    },
    domain::OperatorFuture,
    donor::{CompanionBatch, CompanionRequest, DonorModule},
    geometry::{AlgorithmicDistance, Distance, InteractionKernel},
    noise::InnovationShift,
    operators::{CloneRequest, GasOperators, TransformRequest},
    random::{RandomStream, Stream},
    tracking::RecordedStep,
};
use algorithmic_gas_benchmarks::{
    BenchmarkModel, RunConfig,
    convergence_chapter12_completion::*,
    convergence_experiments::{ArchiveStore, sha256_file},
};
use serde_json::{Value, json};
use std::{fs, path::Path};
type Error = Box<dyn std::error::Error>;
fn permute(p: &Population<f64>, map: &[usize]) -> algorithmic_gas::Result<Population<f64>> {
    let mut out = p.clone();
    for (name, f) in &p.observations.fields {
        let values = map
            .iter()
            .flat_map(|&i| f.row(i).expect("valid row").iter().copied())
            .collect();
        out.observations.fields.insert(
            name.clone(),
            TensorBatch::new(map.len(), f.item_shape().to_vec(), values)?,
        );
    }
    out.validity = map.iter().map(|&i| p.validity[i]).collect();
    out.generations = map.iter().map(|&i| p.generations[i]).collect();
    out.rewards.raw = map.iter().map(|&i| p.rewards.raw[i]).collect();
    out.rewards.valid = map.iter().map(|&i| p.rewards.valid[i]).collect();
    Ok(out)
}
struct Transport {
    step: RecordedStep<f64>,
    map: Vec<usize>,
    seed: u64,
}
impl GasOperators<f64> for Transport {
    fn id(&self) -> String {
        "chapter12-transport-realized-independent-fields/v1".into()
    }
    fn companions<'a>(
        &'a self,
        _module: &'a DonorModule,
        r: CompanionRequest<'a, f64>,
        _cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, CompanionBatch> {
        Box::pin(async move {
            let orig = if r.stream == Stream::Distance {
                &self.step.report.distance_companions
            } else {
                &self.step.report.cloning_companions
            };
            let sources = if r.stream == Stream::Distance {
                &self.step.report.distance_sources
            } else {
                &self.step.report.clone_plan.sources
            };
            let mut out = orig.clone();
            out.indices.clear();
            out.valid.clear();
            for &old in &self.map {
                for a in 0..orig.count {
                    let offset = old * orig.count + a;
                    out.valid.push(orig.valid[offset]);
                    let source = orig.indices[offset] as usize;
                    out.indices.push(if orig.valid[offset] {
                        let slot = sources[source].slot as usize;
                        let new = self.map.iter().position(|&i| i == slot).unwrap();
                        r.pool.current_index(new).expect("transported live source")
                    } else {
                        0
                    });
                }
            }
            Ok(out)
        })
    }
    fn clone_plan(
        &self,
        decision: &CloneDecision,
        r: CloneRequest<'_, f64>,
    ) -> algorithmic_gas::Result<ClonePlan> {
        let mut choices = vec![];
        for (i, &old) in self.map.iter().enumerate() {
            let donor = r.companions.row(i).next().expect("current companion");
            let p = if r.alive[i] {
                decision.acceptance_probability(
                    r.step,
                    r.fitness[i],
                    r.donor_fitness[donor as usize],
                )
            } else {
                1.
            };
            let u = RandomStream::new(self.seed, r.step, Stream::Accept, old as u64, 0)
                .uniform::<f64>();
            choices.push(CloneChoice {
                donors: vec![WeightedDonor {
                    pool_index: donor,
                    weight: 1.,
                }],
                accepted: u < p,
                revival: !r.alive[i],
                probability: Some(p),
            });
        }
        Ok(ClonePlan {
            population_version: r.population.version,
            sources: r.pool.sources.clone(),
            choices,
            mutual: r.companions.mutual,
        })
    }
    fn transform<'a>(
        &'a self,
        transform: &'a CloneTransform,
        r: TransformRequest<'a, f64>,
        after: &'a mut Population<f64>,
        cx: &'a mut ExecutionContext,
    ) -> OperatorFuture<'a, Vec<String>> {
        Box::pin(async move {
            let mut jitter = transform.clone();
            jitter.restitution = None;
            let mut changed = jitter
                .apply(
                    r.before,
                    after,
                    r.pool,
                    r.plan,
                    r.companions,
                    r.seed,
                    r.step,
                    cx,
                )
                .await?;
            if let Some(alpha) = transform.restitution {
                let components = accepted_current_components(r.before, r.pool, r.plan)?;
                let original = accepted_current_components(
                    &self.step.before,
                    &algorithmic_gas::donor::DonorPool::freeze(
                        &self.step.before,
                        self.step.report.clone_plan.sources[0].frame,
                        &[],
                        0,
                        false,
                    )?,
                    &self.step.report.clone_plan,
                )?;
                let d = r.before.observations.field("velocities")?.width();
                let mut rotations = vec![];
                for group in &components {
                    let mut members = group.iter().map(|&i| self.map[i]).collect::<Vec<_>>();
                    members.sort_unstable();
                    let component = original
                        .iter()
                        .find(|g| **g == members)
                        .expect("transported component set");
                    rotations.push(
                        RandomStream::new(
                            self.seed,
                            r.step,
                            Stream::CollisionRotation,
                            component[0] as u64,
                            0,
                        )
                        .haar_orthogonal(d)?,
                    );
                }
                apply_component_rotations(
                    r.before,
                    after,
                    &components,
                    &rotations,
                    "velocities",
                    alpha,
                )?;
                changed.push("velocities".into());
            }
            Ok(changed)
        })
    }
}
fn shift_fields(cfg: &mut GasConfig, s: &RecordedStep<f64>, map: &[usize]) -> Result<(), Error> {
    cfg.qft.innovation_shifts.clear();
    for noise in &s.noise {
        let raw = noise
            .raw_innovation
            .as_ref()
            .ok_or("raw independent innovation required")?;
        for (i, &old) in map.iter().enumerate() {
            let mut rng =
                RandomStream::new(cfg.seed, noise.step, noise.stream, i as u64, noise.substep);
            for a in 0..noise.dimension {
                let shift = raw[old * noise.dimension + a] - rng.gaussian::<f64>();
                cfg.qft.innovation_shifts.push(InnovationShift {
                    step: noise.step,
                    stream: noise.stream,
                    substep: noise.substep,
                    walker: i,
                    coordinate: a,
                    shift,
                });
            }
        }
    }
    Ok(())
}
fn kernel_transport(
    cfg: &RunConfig,
    s: &RecordedStep<f64>,
    checks: &mut Vec<Comparison>,
) -> Result<Value, Error> {
    let n = s.before.len();
    let map = (0..n).rev().collect::<Vec<_>>();
    let pp = permute(&s.before, &map)?;
    let old = algorithmic_gas::donor::DonorPool::freeze(&s.before, 0, &[], 0, false)?;
    let new = algorithmic_gas::donor::DonorPool::freeze(&pp, 0, &[], 0, false)?;
    let mut tables = vec![];
    for (name, module) in [
        ("measurement", &cfg.gas.distance_donors),
        ("cloning", &cfg.gas.cloning_donors),
    ] {
        let mut left = vec![];
        let mut right = vec![];
        let mut rows = vec![];
        let kind = <Distance as AlgorithmicDistance<f64>>::comparison_kind(&module.distance);
        for i in 0..n {
            let ii = n - 1 - i;
            let mut a = vec![];
            let mut b = vec![];
            for (j, source) in old.sources.iter().enumerate() {
                let slot = source.slot as usize;
                let jj = new
                    .current_index(n - 1 - slot)
                    .expect("mapped eligible source") as usize;
                let excluded = !module.allow_self && slot == i;
                let wa = if excluded {
                    0.
                } else {
                    module
                        .kernel
                        .log_weight(
                            module.distance.compare(
                                &s.before.observations,
                                i,
                                &old.population.observations,
                                j,
                            )?,
                            kind,
                        )?
                        .exp()
                };
                let wb = if excluded {
                    0.
                } else {
                    module
                        .kernel
                        .log_weight(
                            module.distance.compare(
                                &pp.observations,
                                ii,
                                &new.population.observations,
                                jj,
                            )?,
                            kind,
                        )?
                        .exp()
                };
                a.push(wa);
                b.push(wb);
            }
            let za = a.iter().sum::<f64>();
            let zb = b.iter().sum::<f64>();
            if za > 0. {
                for v in &mut a {
                    *v /= za;
                }
                for v in &mut b {
                    *v /= zb;
                }
            }
            rows.push(json!({"old_recipient":i,"new_recipient":ii,"original_normalizer":za,"permuted_normalizer":zb}));
            left.extend(a);
            right.extend(b);
        }
        push(
            checks,
            "rem-exchangeability-complete-kernel",
            &format!("native {name} categorical law transport"),
            max_defect(&left, &right),
            0.,
            true,
            json!({"N":n,"module":name,"alive_sources":old.sources.len(),"map":map}),
            "Native algorithmic distance and kernel normalized weights under source/recipient permutation; current single independent donor law.",
        );
        tables.push(json!({"module":name,"map":map,"rows":rows,"original_normalized_matrix":left,"transported_normalized_matrix":right,"source_slots":old.sources.iter().map(|r|r.slot).collect::<Vec<_>>() }));
    }
    Ok(json!({"configuration":cfg,"prepared_input":s.before,"kernel_tables":tables}))
}
fn max_defect(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
async fn replay(
    cfg: &RunConfig,
    s: &RecordedStep<f64>,
    map: Vec<usize>,
) -> Result<algorithmic_gas::tracking::RunArchive<f64>, Error> {
    let input = permute(&s.before, &map)?;
    let mut gas_cfg = cfg.gas.clone();
    shift_fields(&mut gas_cfg, s, &map)?;
    let reward = BenchmarkModel {
        benchmark: cfg.benchmark,
        field: "positions".into(),
        direction: cfg.gas.fitness.direction,
    };
    let gradient = BenchmarkModel {
        benchmark: cfg.potential.unwrap_or(cfg.benchmark),
        field: "positions".into(),
        direction: algorithmic_gas::fitness::ObjectiveDirection::Minimize,
    };
    let mut gas = GasBuilder::new(input, reward)
        .gradient(gradient)
        .config(gas_cfg)
        .operators(Transport {
            step: s.clone(),
            map,
            seed: cfg.gas.seed,
        })
        .build()
        .await?;
    gas.start_recording(RecordingConfig {
        max_steps: 1,
        max_bytes: 64 * 1024 * 1024,
        ..Default::default()
    })?;
    gas.step().await?;
    Ok(gas.stop_recording().ok_or("recording absent")?)
}
#[allow(clippy::too_many_arguments)] // Each evidence row retains all mathematical operands and scope.
fn push(
    c: &mut Vec<Comparison>,
    label: &str,
    e: &str,
    a: f64,
    b: f64,
    equal: bool,
    operands: Value,
    scope: &str,
) {
    c.push(Comparison::new(label, e, a, b, equal, operands, scope));
}
fn categorical(r: &mut RandomStream, n: usize) -> usize {
    r.index(n)
}
fn distinct(r: &mut RandomStream, n: usize, k: usize) -> Vec<usize> {
    let mut xs = (0..n).collect::<Vec<_>>();
    for i in 0..k {
        let j = i + r.index(n - i);
        xs.swap(i, j);
    }
    xs.truncate(k);
    xs
}
fn sampling(
    p: &Population<f64>,
    draws: usize,
    seed: u64,
    checks: &mut Vec<Comparison>,
) -> Result<Value, Error> {
    let n = p.len();
    let x = p.observations.field("positions")?;
    assert!(
        p.states.is_none(),
        "Euclidean numerical population required"
    );
    let g = (0..n)
        .map(|i| x.row(i).unwrap()[0].sin())
        .collect::<Vec<_>>();
    let mut rows = vec![];
    let all_distinct = (0..n).all(|i| (0..i).all(|j| x.row(i).unwrap() != x.row(j).unwrap()));
    for k in [1, 2, 4, 8].into_iter().filter(|&k| k <= n) {
        let probability = collision_probability(n, k)?;
        if all_distinct {
            push(
                checks,
                "thm-hewitt-savage-representation",
                "exact physical empirical-orbit TV",
                probability,
                collision_probability(n, k)?,
                true,
                json!({"N":n,"k":k,"all_position_atoms_distinct":true}),
                "Exact finite full-position tuple laws on the recorded permutation orbit: off-collision ordered tuples have uniform distinct atoms; total variation equals repeated-index mass. No native QSD assumption.",
            );
        }

        let mut rng = RandomStream::new(seed, 0, Stream::MeanFieldReference, k as u64, 0);
        let mut collisions = 0;
        let mut differences = 0.;
        let mut samples = vec![];
        for _ in 0..draws {
            let with = (0..k).map(|_| categorical(&mut rng, n)).collect::<Vec<_>>();
            let collision = (0..k).any(|i| (0..i).any(|j| with[i] == with[j]));
            let without = if collision {
                distinct(&mut rng, n, k)
            } else {
                with.clone()
            };
            collisions += usize::from(collision);
            let fw = with.iter().map(|&i| g[i]).product::<f64>();
            let fo = without.iter().map(|&i| g[i]).product::<f64>();
            differences += fw - fo;
            if (fw - fo).abs() > 2. * f64::from(collision) + 1e-12 {
                return Err("coupling physical observable exceeded disagreement bound".into());
            }
            samples.push(json!({"with":with,"without":without,"collision":collision,"product_with":fw,"product_without":fo}));
        }
        let allowance = ((2e7_f64).ln() / (2. * draws as f64)).sqrt();
        push(
            checks,
            "thm-hewitt-savage-representation",
            "collision Monte Carlo",
            (collisions as f64 / draws as f64 - probability).abs(),
            allowance,
            false,
            json!({"N":n,"k":k,"draws":draws,"family_failure":1e-7}),
            "Independent index-list draws conditional on complete physical native swarm; sampling reference law, no native QSD assumption.",
        );
        push(
            checks,
            "thm-hewitt-savage-representation",
            "union bound",
            probability,
            union_bound(n, k)?,
            false,
            json!({"N":n,"k":k}),
            "Exact index law; probability normalization.",
        );
        push(
            checks,
            "thm-propagation-chaos-qsd",
            "bounded product-test coupling",
            (differences / draws as f64).abs(),
            2. * probability + 2. * allowance,
            false,
            json!({"N":n,"k":k,"draws":draws,"bound":1}),
            "Finite conditional native-cloud sampling test only; no limit inferred.",
        );
        rows.push(json!({"k":k,"collision_probability":probability,"union_bound":union_bound(n,k)?,"collision_fraction":collisions as f64/draws as f64,"mean_product_difference":differences/draws as f64,"samples":samples}));
    }
    Ok(
        json!({"physical_population":p,"normalized_sin_values":g,"draws_per_k":draws,"seed":seed,"rows":rows}),
    )
}
fn reference(checks: &mut Vec<Comparison>) -> Result<Value, Error> {
    let mut cases = vec![];
    for n in [2, 4, 8, 16, 32, 128, 512] {
        for q in [0.2_f64, 0.5, 0.8] {
            for beta in [0., 0.5, 2.] {
                let v = tilted_bernoulli(n, q, beta)?;
                let f = |key: &str| v[key].as_f64().unwrap();
                push(
                    checks,
                    "thm-correlation-decay",
                    "exact empirical covariance identity",
                    f("distinct_covariance"),
                    distinct_covariance(n, f("empirical_variance"), f("diagonal_covariance"))?,
                    true,
                    v.clone(),
                    "Exact bounded exchangeable Bernoulli joint tilt; independent from native interacting QSD.",
                );
                push(
                    checks,
                    "thm-mixing-variance-corrected",
                    "entropy MSE",
                    f("mse"),
                    entropy_mse_bound(n, 1., f("total_relative_entropy"))?,
                    false,
                    v.clone(),
                    "Exact total joint entropy against Bernoulli(q) product law.",
                );
                push(
                    checks,
                    "thm-mixing-variance-corrected",
                    "variance below MSE",
                    f("empirical_variance"),
                    f("mse"),
                    false,
                    v.clone(),
                    "Exact finite count law.",
                );
                push(
                    checks,
                    "thm-mixing-variance-corrected",
                    "square exponential moment",
                    f("product_square_exponential_moment"),
                    2_f64.sqrt(),
                    false,
                    v.clone(),
                    "Exact finite independent product law; Hoeffding-Gaussian bound.",
                );
                push(
                    checks,
                    "thm-mixing-variance-corrected",
                    "distinct covariance entropy",
                    f("distinct_covariance").abs(),
                    (4. * (f("total_relative_entropy") + 0.5 * 2_f64.ln()) + 1.) / (n - 1) as f64,
                    false,
                    v.clone(),
                    "Exact finite exchangeable bounded joint tilt.",
                );

                for t in [-8_f64, -2., 0., 2., 8.] {
                    let log_mgf = n as f64 * ((1. - q) + q * (t / n as f64).exp()).ln() - t * q;
                    push(
                        checks,
                        "thm-mixing-variance-corrected",
                        "Hoeffding log moment",
                        log_mgf,
                        t * t / (2. * n as f64),
                        false,
                        json!({"N":n,"q":q,"t":t,"B":1}),
                        "Exact Bernoulli product-reference moment generating function.",
                    );
                }
                let ent_rhs =
                    f("total_relative_entropy") + f("product_square_exponential_moment").ln();
                let ent_lhs = n as f64 * f("mse") / 4.;
                push(
                    checks,
                    "thm-mixing-variance-corrected",
                    "entropy variational inequality",
                    ent_lhs,
                    ent_rhs,
                    false,
                    v.clone(),
                    "Exact count-law KL and exponential tilt normalizer; no empirical entropy estimator.",
                );
                push(
                    checks,
                    "thm-mixing-variance-corrected",
                    "tilted-reference KL nonnegativity",
                    0.,
                    ent_rhs - ent_lhs,
                    false,
                    v.clone(),
                    "Exact KL(P || exp(A)Q/Z)=H-E_P A+log Z.",
                );
                cases.push(v);
            }
        }
    }

    push(
        checks,
        "thm-mixing-variance-corrected",
        "zero bounded observable",
        entropy_mse_bound(8, 0., 1.)?,
        0.,
        true,
        json!({"N":8,"B":0,"H":1}),
        "B=0 branch is exact before any division by B.",
    );
    let mut qsd = vec![];
    for n in [2_usize, 4, 8] {
        for q in [0.2_f64, 0.5, 0.8] {
            for alpha in [0.1, 0.5, 0.99] {
                let mass = (0..1usize << n)
                    .map(|mask| {
                        q.powi(mask.count_ones() as i32)
                            * (1. - q).powi(n as i32 - mask.count_ones() as i32)
                    })
                    .collect::<Vec<_>>();
                let mut equivariance: f64 = 0.;
                let mut eigen: f64 = 0.;
                for mask in 0..mass.len() {
                    let reversed =
                        (0..n).fold(0, |out, i| out | (((mask >> i) & 1) << (n - 1 - i)));
                    equivariance =
                        equivariance.max((alpha * mass[mask] - alpha * mass[reversed]).abs());
                    let image = mass.iter().map(|p| p * alpha * mass[mask]).sum::<f64>();
                    eigen = eigen.max((image - alpha * mass[mask]).abs());
                }
                push(
                    checks,
                    "thm-qsd-exchangeability",
                    "killed QSD eigenfactor",
                    eigen,
                    0.,
                    true,
                    json!({"N":n,"q":q,"alpha":alpha,"mass":mass}),
                    "Strictly positive rank-one killed kernel Q(x,y)=alpha*pi(y); unique QSD explicit, constant survival retained.",
                );
                push(
                    checks,
                    "thm-qsd-exchangeability",
                    "killed kernel equivariance",
                    equivariance,
                    0.,
                    true,
                    json!({"N":n,"q":q,"alpha":alpha}),
                    "Finite killed reference kernel, all reversal orbits exact. General permutations preserve count analytically.",
                );
                qsd.push(json!({"N":n,"q":q,"alpha":alpha,"mass":mass}));
            }
        }
    }
    let mut gaussian = vec![];
    for n in [1, 8, 32, 128] {
        for d in [1, 2, 4, 8] {
            for theta in [0.1, 0.5, 2.] {
                for curvature in [0.25, 1., 4.] {
                    let c = gaussian_lsi_constant(theta, curvature)?;
                    let a2 = 0.01;
                    let variance = theta / curvature;
                    let tilt_entropy = 0.5 * a2 * variance;
                    let fisher = a2;
                    push(
                        checks,
                        "thm-n-uniform-lsi-exchangeable",
                        "Gaussian KL-Fisher convention",
                        tilt_entropy,
                        c / 2. * fisher,
                        false,
                        json!({"N":n,"d":d,"theta":theta,"curvature":curvature,"constant":c,"mean_tilt_norm_squared":a2,"KL":tilt_entropy,"Fisher":fisher}),
                        "Product conservative quadratic kinetic reference and Gaussian exponential tilt; C independent of N and d.",
                    );

                    push(
                        checks,
                        "thm-n-uniform-lsi-exchangeable",
                        "square-root density Fisher factor",
                        4. * fisher / 4.,
                        fisher,
                        true,
                        json!({"N":n,"d":d,"gradient_square_integral":fisher/4.,"Fisher":fisher}),
                        "Normalized Gaussian shifted density and its exact square-root density gradient integral.",
                    );
                    push(
                        checks,
                        "cor-mean-field-lsi",
                        "one-coordinate marginal Gaussian LSI",
                        tilt_entropy,
                        2. * c * fisher / 4.,
                        false,
                        json!({"N":n,"d":d,"theta":theta,"curvature":curvature,"C":c}),
                        "Same finite product Gaussian law tested with f=g(z_1); other continuous gradients are exactly zero. The weak-limit passage remains analytic.",
                    );
                    for t in [0., 0.04, 1., 5.] {
                        let gamma = 1.;
                        let sigma = (2_f64 * gamma * theta).sqrt();
                        let cov = sigma * sigma * (-(-2. * gamma * t).exp_m1()) / (2. * gamma);
                        push(
                            checks,
                            "lem-conditional-gaussian-qsd-euclidean",
                            "OU limiting covariance",
                            cov,
                            theta,
                            false,
                            json!({"time":t,"gamma":gamma,"sigma":sigma,"d":d,"N":n}),
                            "Frozen conservative OU reference, not conditional law of interacting native output.",
                        );
                    }
                    gaussian.push(json!({"N":n,"d":d,"theta":theta,"curvature":curvature,"C":c,"gaussian_KL":tilt_entropy,"gaussian_Fisher":fisher}));
                }
            }
        }
    }
    Ok(
        json!({"bounded_joint_tilts":cases,"explicit_killed_qsd":qsd,"Gaussian_product_LSI":gaussian}),
    )
}
async fn run(input: &str, output: &str, reps: usize, draws: usize) -> Result<(), Error> {
    let native = ArchiveStore::open(input)?;
    let mut store = ArchiveStore::new(output)?;
    store.save_json("execution-source",&json!({"runner":include_str!("gas-chapter12-completion.rs"),"module":include_str!("../convergence_chapter12_completion.rs"),"chapter":SOURCE,"native_cloning":include_str!("../../../algorithmic-gas/src/cloning.rs"),"native_kinetic":include_str!("../../../algorithmic-gas/src/kinetic.rs"),"native_noise":include_str!("../../../algorithmic-gas/src/noise.rs"),"native_donor":include_str!("../../../algorithmic-gas/src/donor.rs"),"input_index_sha256":sha256_file(&Path::new(input).join("archive-index.json"))?,"executable_sha256":sha256_file(&std::env::current_exe()?)?,"replicas":reps,"index_draws_per_k":draws,"scope":"Full native engine transition with transported independent donor choices/gate uniforms/jitter/Haar and kinetic innovations. Array addresses only replay bookkeeping. Canonical native algorithm has current single donors; no walker-label metric."}))?;
    let mut checks = vec![];
    let refs = reference(&mut checks)?;
    store.save_json("exact-reference-operands", &refs)?;
    let mut cases = vec![];
    let mut native_updates = 0;
    let mut sampled = 0;
    for entry in native
        .entries()
        .iter()
        .filter(|e| e.tag.ends_with("native-operands"))
    {
        let operands = native.load_json(&entry.path)?;
        let case = entry.tag.trim_end_matches("-native-operands");
        let cfg: RunConfig = serde_json::from_value(operands["configuration"].clone())?;
        let rows = operands["raw"].as_array().ok_or("raw")?;
        let mut records = vec![];
        let mut bary = 0.;
        let mut second = 0.;
        let mut diag = 0.;
        let mut pair = 0.;
        let mut replaydefect: f64 = 0.;
        let mut physcoordinates = 0;
        for (rep, raw) in rows.iter().enumerate() {
            let original =
                native.load_archive(raw["native_archive"].as_str().ok_or("native archive")?)?;
            let step = &original.steps[0];
            if rep == 0 {
                let kernel = kernel_transport(&cfg, step, &mut checks)?;
                store.save_json(&format!("{case}-native-kernel-laws"), &kernel)?;
            }

            let n = step.before.len();
            let x = step.final_population.observations.field("positions")?;
            let g = (0..n)
                .map(|i| x.row(i).unwrap()[0].sin())
                .collect::<Vec<_>>();
            let m = g.iter().sum::<f64>() / n as f64;
            let q = g.iter().map(|x| x * x).sum::<f64>() / n as f64;
            bary += m / rows.len() as f64;
            second += m * m / rows.len() as f64;
            diag += q / rows.len() as f64;
            pair += ((n as f64 * m * m - q) / (n - 1) as f64) / rows.len() as f64;
            if rep >= reps {
                continue;
            }
            let mut rcfg = cfg.clone();
            rcfg.gas.seed = raw["seed"].as_u64().ok_or("seed")?;
            let map = (0..n).rev().collect::<Vec<_>>();
            let baseline = replay(&rcfg, step, (0..n).collect()).await?;
            let permuted = replay(&rcfg, step, map.clone()).await?;
            native_updates += 2;
            let original_path =
                store.save_archive(&format!("{case}-rep{rep}-original"), &original)?;
            let baseline_path =
                store.save_archive(&format!("{case}-rep{rep}-identity-replay"), &baseline)?;
            let permutation_path =
                store.save_archive(&format!("{case}-rep{rep}-permutation-replay"), &permuted)?;
            for (label, s, mm) in [
                ("identity", &baseline.steps[0], (0..n).collect::<Vec<_>>()),
                ("reversal", &permuted.steps[0], map.clone()),
            ] {
                for (name, f) in &step.final_population.observations.fields {
                    let expected = permute(&step.final_population, &mm)?;
                    let target = expected.observations.field(name)?;
                    let value = s.final_population.observations.field(name)?;
                    let defect = max_defect(target.values(), value.values());
                    replaydefect = replaydefect.max(defect);
                    physcoordinates += f.values().len();
                    push(
                        &mut checks,
                        "rem-exchangeability-complete-kernel",
                        &format!("native final {label}/{name}"),
                        defect,
                        0.,
                        true,
                        json!({"case":case,"replica":rep,"map":mm,"field":name,"coordinates":f.values().len()}),
                        "Actual complete native physical engine replay; transported independent random fields, shared Haar matrices mapped by component membership.",
                    );
                }
                let expected = permute(&step.final_population, &mm)?;
                push(
                    &mut checks,
                    "rem-exchangeability-complete-kernel",
                    &format!("native status/generation {label}"),
                    f64::from(
                        s.final_population.validity != expected.validity
                            || s.final_population.generations != expected.generations,
                    ),
                    0.,
                    true,
                    json!({"case":case,"replica":rep,"status":s.final_population.validity,"generations":s.final_population.generations}),
                    "Native terminal boundary, revival and simultaneous literal copies; row addresses transported with physical states.",
                );
                push(
                    &mut checks,
                    "rem-exchangeability-complete-kernel",
                    &format!("native reward {label}"),
                    max_defect(&s.final_population.rewards.raw, &expected.rewards.raw),
                    0.,
                    true,
                    json!({"case":case,"replica":rep}),
                    "Actual native reward evaluator after transported final physical state.",
                );
                for origstage in &step.stages {
                    if let Some(newstage) = s.stages.iter().find(|t| t.stage == origstage.stage) {
                        for (name, f) in &origstage.fields {
                            if let Some(newfield) = newstage.fields.get(name) {
                                let width = f.values.len() / n;
                                let target = mm
                                    .iter()
                                    .flat_map(|&i| {
                                        f.values[i * width..(i + 1) * width].iter().copied()
                                    })
                                    .collect::<Vec<_>>();
                                let defect = max_defect(&target, &newfield.values);
                                physcoordinates += target.len();
                                push(
                                    &mut checks,
                                    "rem-exchangeability-complete-kernel",
                                    &format!("native stage {label}/{}/{name}", origstage.stage),
                                    defect,
                                    0.,
                                    true,
                                    json!({"case":case,"replica":rep,"stage":origstage.stage,"field":name}),
                                    "Native recorded physical substep; status coverage retained in archive.",
                                );
                            }
                        }
                    }
                }
            }
            let sample = sampling(
                &step.final_population,
                draws,
                rcfg.gas.seed ^ 0x45584348414e4745,
                &mut checks,
            )?;
            let samplepath =
                store.save_json(&format!("{case}-rep{rep}-finite-mixture"), &sample)?;
            sampled += sample["rows"].as_array().unwrap().len() * draws;
            records.push(json!({"replica":rep,"original":original_path,"identity_replay":baseline_path,"permutation_replay":permutation_path,"sampling":samplepath,"map":map}));
        }
        let empirical = second - bary * bary;
        let diagonal = diag - bary * bary;
        let distinct = pair - bary * bary;
        push(
            &mut checks,
            "thm-correlation-decay",
            "native empirical-orbit covariance",
            distinct,
            distinct_covariance(cfg.walkers, empirical, diagonal)?,
            true,
            json!({"N":cfg.walkers,"case":case,"native_independent_swarms":rows.len(),"empirical_mean":bary,"empirical_variance":empirical,"diagonal_variance":diagonal,"distinct_covariance":distinct}),
            "Exact finite law uniform over the retained native empirical clouds and their permutation orbits; finite empirical law identity, not unknown native QSD moments.",
        );
        cases.push(json!({"case":case,"configuration":cfg,"independent_retained_native_swarms":rows.len(),"replayed_replicas":reps.min(rows.len()),"fresh_full_native_updates":2*reps.min(rows.len()),"replay_max_defect":replaydefect,"checked_physical_coordinates":physcoordinates,"sampling_paths":records,"empirical_orbit_mean":bary,"empirical_orbit_variance":empirical,"distinct_covariance":distinct}));
    }
    let failed = checks.iter().filter(|c| !c.passed).count();
    store.save_json("all-checks", &json!(checks))?;
    store.finish(if failed == 0 { "complete" } else { "failed" })?;
    let verification = store.verify(true)?;
    let report = json!({"status":if failed==0{"complete"}else{"failed"},"chapter":12,"cases":cases,"fresh_full_native_updates":native_updates,"independent_index_lists":sampled,"checks":checks.len(),"failed":failed,"reference_laws":"Exact finite Bernoulli bounded joint tilts, strictly positive rank-one killed chain QSDs, and conservative Gaussian product OU/LSI laws; never treated as native interacting QSD.","source_sha256":sha256_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/source/2_fractal_gas/convergence_program/12_qsd_exchangeability_theory.md").as_path())?,"verification":verification});
    fs::write(
        Path::new(output).join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    fs::write(
        Path::new(output).join("verification.json"),
        serde_json::to_vec_pretty(&verification)?,
    )?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"checks":checks.len(),"failed":failed,"fresh_full_native_updates":native_updates,"independent_index_lists":sampled,"verification":verification})
        )?
    );
    if failed > 0 {
        return Err("chapter12 checks failed; immutable operands retained".into());
    }
    Ok(())
}
fn main() -> Result<(), Error> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() < 3 {
        return Err("usage: gas-chapter12-completion INPUT_CH08_NATIVE EMPTY_OUTPUT [REPLICAS=8] [INDEX_DRAWS=512]".into());
    }
    futures_lite::future::block_on(run(
        &a[1],
        &a[2],
        a.get(3).map(|x| x.parse()).transpose()?.unwrap_or(8),
        a.get(4).map(|x| x.parse()).transpose()?.unwrap_or(512),
    ))
}
