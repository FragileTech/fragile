//! Direct algebraic measurements of executed algorithm records.
use super::{colors, complex_json, math::*};
use crate::{
    Result, RunArchive,
    physics::partvi::{ExperimentRequest, ExperimentResult, Series},
};
use serde_json::json;
pub(super) fn analyze(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    if r.experiment == 2 {
        return cloning(a);
    }
    if r.experiment == 14 {
        return writes(a);
    }
    if r.experiment == 10 {
        return doublets(r, a);
    }
    let read = colors(r, Some(a))?;
    let raw = read.details["raw_color"].as_array().unwrap();
    let masks = read.details["valid_mask"].as_array().unwrap();
    let cs: Vec<[C; 3]> = raw
        .iter()
        .zip(masks)
        .filter(|(_, m)| m.as_bool() == Some(true))
        .map(|(row, _)| {
            std::array::from_fn(|i| {
                C::new(
                    row[i]["real"].as_f64().unwrap(),
                    row[i]["imaginary"].as_f64().unwrap(),
                )
            })
        })
        .collect();
    let angle = r.number("rotation", 0.7);
    let (c, s) = (angle.cos(), angle.sin());
    let rotate = |z: &[C; 3]| {
        [
            z[0] * c + z[1] * s,
            (z[1] * c - z[0] * s) * C::phase(angle),
            z[2] * C::phase(-angle),
        ]
    };
    let mut out = ExperimentResult::new(
        9,
        "Recorded color contraction invariants",
        "Actual matched B1 force/velocity colors under common SU(3) rotation and independent ray phases",
    );
    let mut residual: f64 = 0.;
    let mut projectors: f64 = 0.;
    let mut gram: f64 = 0.;
    let mut values = vec![];
    let mut triples = vec![];
    for (i, t) in cs.windows(3).enumerate() {
        let [x, y, z] = [t[0], t[1], t[2]];
        let q = dot(&x, &y);
        let triangle = q * dot(&y, &z) * dot(&z, &x);
        let matrix: Vec<C> = (0..9).map(|k| t[k % 3][k / 3]).collect();
        let b = determinant(&matrix, 3);
        let rt = [rotate(&x), rotate(&y), rotate(&z)];
        let rb = determinant(&(0..9).map(|k| rt[k % 3][k / 3]).collect::<Vec<_>>(), 3);
        residual = residual
            .max((dot(&rt[0], &rt[1]) - q).abs())
            .max((rb - b).abs());
        let ps: Vec<Vec<C>> = t
            .iter()
            .map(|v| (0..9).map(|k| v[k / 3] * v[k % 3].conj()).collect())
            .collect();
        let tr = trace(&mul(&mul(&ps[0], &ps[1], 3), &ps[2], 3), 3);
        projectors = projectors.max((tr - triangle).abs());
        let g: Vec<C> = (0..9).map(|k| dot(&t[k / 3], &t[k % 3])).collect();
        gram = gram.max((determinant(&g, 3) - C::from(b.abs2())).abs());
        let phases = [
            C::phase(angle),
            C::phase(-0.2 * angle),
            C::phase(1.7 * angle),
        ];
        let ph: [Vec<C>; 3] =
            std::array::from_fn(|j| t[j].iter().map(|&v| v * phases[j]).collect());
        let phase_tri = dot(&ph[0], &ph[1]) * dot(&ph[1], &ph[2]) * dot(&ph[2], &ph[0]);
        residual = residual.max((phase_tri - triangle).abs());
        values.push([i as f64, triangle.im]);
        triples.push(json!({"valid_color_indices":[i,i+1,i+2],"pair":complex_json(q),"baryon":complex_json(b),"triangle":complex_json(triangle)}));
    }
    out.metric("Valid colors", cs.len() as f64, "walkers")
        .metric("Valid triples", triples.len() as f64, "triples")
        .metric("Common SU(3) and ray-phase residual", residual, "")
        .metric("Projector trace residual", projectors, "")
        .metric("Baryon Gram determinant residual", gram, "");
    out.plot(
        "Recorded triangle phase readout",
        "consecutive valid triple",
        "imaginary part",
        vec![Series::line("Actual color triangle", values)],
    );
    out.details = json!({"status":if triples.is_empty(){"unavailable"}else{"available"},"reason":if triples.is_empty(){"At least three valid colors are required"}else{""},"color_record":read.details,"triples":triples,"selection":"Consecutive triples in recorded valid-slot order","theory_labels":["thm-sm-direct-color-invariants","prop-sm-direct-triangle-projectors"],"validation":"Independent dense projector multiplication and Gram determinant; common non-diagonal SU(3) action on actual colors"});
    Ok(out)
}
fn cloning(a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let eps = a.gas_config.clone_decision.epsilon;
    let saturation = a.gas_config.clone_decision.saturation;
    let mut o = ExperimentResult::new(
        2,
        "Executed cloning scores, clipping and gates",
        "Recorded pool-aligned fitness and actual directed clone choices, including historical donor rescoring",
    );
    let mut anti: f64 = 0.;
    let mut probability_error: f64 = 0.;
    let mut gates = 0.;
    let mut expected = 0.;
    let mut variance = 0.;
    let mut revivals = 0usize;
    let mut rows = vec![];
    let mut actual = vec![];
    let mut predicted = vec![];
    for step in &a.steps {
        let mut observed = 0.;
        let mut forecast = 0.;
        for (i, c) in step.report.clone_plan.choices.iter().enumerate() {
            if c.revival {
                revivals += 1;
                continue;
            }
            let vi = step.report.pre_clone_fitness.fitness[i];
            let vj = c
                .donors
                .first()
                .map(|d| step.donor_fitness[d.pool_index as usize]);
            let p = if let Some(vj) = vj {
                let sij = (vj - vi) / (vi + eps);
                let sji = (vi - vj) / (vj + eps);
                anti = anti.max(((vi + eps) * sij + (vj + eps) * sji).abs());
                (sij / saturation).clamp(0., 1.)
            } else {
                0.
            };
            if let Some(stored) = c.probability {
                probability_error = probability_error.max((p - stored).abs());
            }
            observed += f64::from(c.accepted);
            forecast += p;
            variance += p * (1. - p);
            rows.push(json!({"epoch":step.epoch,"step":step.report.step,"slot":i,"source_version":step.report.source_version,"own_fitness":vi,"donor_fitness":vj,"donors":c.donors,"source_refs":c.donors.iter().map(|d|step.report.clone_plan.sources[d.pool_index as usize]).collect::<Vec<_>>(),"predicted_probability":p,"recorded_probability":c.probability,"accepted":c.accepted}));
        }
        gates += observed;
        expected += forecast;
        actual.push([step.report.step as f64, observed]);
        predicted.push([step.report.step as f64, forecast]);
    }
    o.metric("Weighted antisymmetry residual", anti, "")
        .metric(
            "Recorded acceptance probability residual",
            probability_error,
            "",
        )
        .metric("Observed ordinary clone gates", gates, "gates")
        .metric(
            "Conditional expected ordinary clone gates",
            expected,
            "gates",
        )
        .metric("Gate innovation", gates - expected, "gates")
        .metric("Predictable gate variance", variance, "gates squared")
        .metric("Revival gates", revivals as f64, "gates");
    o.plot(
        "Actual gates and conditional expectations",
        "recorded step",
        "ordinary gates",
        vec![
            Series::line("Executed", actual),
            Series::line("Conditional clipped score", predicted),
        ],
    );
    o.details = json!({"status":"available","stage":"pre_clone to literal_clone","epsilon":eps,"saturation":saturation,"rows":rows,"theory_labels":["thm-cloning-antisymmetry-lqft"],"derivation":"Given actual selected companions and pool-aligned fitness, each ordinary gate is a Bernoulli draw with the clipped directed score. Revival is a separate deterministic gate.","validation":"Stored probabilities and realized gates are read independently of reconstructed probabilities. Predictable variance describes conditional gate noise; no independent-time sampling assumption is used."});
    Ok(o)
}
fn writes(a: &RunArchive<f64>) -> Result<ExperimentResult> {
    use crate::GasError;
    let velocity = match &a.gas_config.kinetic.integrator {
        crate::kinetic::KineticKind::Baoab { velocities, .. } => velocities.as_str(),
        _ => {
            return Err(GasError::Capability(
                "Clone momentum accounting requires a recorded velocity field".into(),
            ));
        }
    };
    let mut o = ExperimentResult::new(
        14,
        "Actual cloning and transform momentum budgets",
        "Immutable donor copying, followed by each executed transform, reconciliation and boundary stage",
    );
    let mut residual: f64 = 0.;
    let mut measured = vec![];
    let mut expected = vec![];
    let mut budgets = vec![];
    for step in &a.steps {
        let pre = step.before.observations.field(velocity)?;
        let d = pre.width();
        let mut predicted = pre.values().to_vec();
        let mut source_counts = vec![0usize; step.report.clone_plan.sources.len()];
        for (i, c) in step.report.clone_plan.choices.iter().enumerate() {
            if !c.accepted {
                continue;
            }
            for j in 0..d {
                predicted[i * d + j] = 0.;
            }
            for donor in &c.donors {
                let source = step.report.clone_plan.sources[donor.pool_index as usize];
                let p = if source.version == step.before.version
                    && source.frame + 1 == step.report.step
                {
                    &step.before
                } else {
                    &a.steps
                        .iter()
                        .find(|s| {
                            s.epoch == step.epoch
                                && s.report.step == source.frame + 1
                                && s.before.version == source.version
                        })
                        .ok_or_else(|| {
                            GasError::Capability(
                                "A historical clone source lies outside the recorded archive"
                                    .into(),
                            )
                        })?
                        .before
                };
                let row = p.observations.field(velocity)?.row(source.slot as usize)?;
                for j in 0..d {
                    predicted[i * d + j] += donor.weight * row[j];
                }
                source_counts[donor.pool_index as usize] += 1;
            }
        }
        let literal = step
            .stages
            .iter()
            .find(|s| s.stage == "literal_clone")
            .ok_or_else(|| GasError::Capability("literal_clone stage unavailable".into()))?;
        let recorded = &literal
            .fields
            .get(velocity)
            .ok_or_else(|| GasError::MissingField(velocity.into()))?
            .values;
        residual = residual.max(
            predicted
                .iter()
                .zip(recorded)
                .map(|(x, y)| (x - y).abs())
                .fold(0., f64::max),
        );
        let oldsum: Vec<f64> = (0..d)
            .map(|j| pre.values().chunks_exact(d).map(|v| v[j]).sum())
            .collect();
        let total = |v: &[f64]| -> Vec<f64> {
            (0..d)
                .map(|j| v.chunks_exact(d).map(|v| v[j]).sum())
                .collect()
        };
        let pred = total(&predicted);
        let seen = total(recorded);
        measured.push([step.report.step as f64, seen[0] - oldsum[0]]);
        expected.push([step.report.step as f64, pred[0] - oldsum[0]]);
        let stages:Vec<_>=step.stages.iter().filter(|s|["pre_clone","literal_clone","post_transform","post_clone"].contains(&s.stage.as_str())).filter_map(|s|s.fields.get(velocity).map(|f|json!({"stage":s.stage,"version":s.version,"all_slot_momentum":total(&f.values),"eligible_count":s.validity.iter().filter(|v|v.eligible(a.gas_config.include_truncated)).count()}))).collect();
        budgets.push(json!({"epoch":step.epoch,"step":step.report.step,"source_copy_counts":source_counts,"stages":stages}));
    }
    o.metric("Literal clone write residual", residual, "velocity");
    o.plot(
        "Measured momentum change from clone writes",
        "recorded step",
        "momentum x (unit mass)",
        vec![
            Series::line("Recorded literal-clone increment", measured),
            Series::line("Immutable donor-copy prediction", expected),
        ],
    );
    o.details = json!({"status":"available","velocity_field":velocity,"mass":1.,"budgets":budgets,"clone_transform":a.gas_config.clone_transform,"normalization":"All slots at unit mass; eligibility counts retained separately at each stage","derivation":"Each accepted recipient is replaced by its immutable weighted donor row. Net copying momentum equals donor contributions minus overwritten recipients; later restitution, jitter, reconciliation and boundary effects remain separate measured stage increments.","theory_labels":["prop-sm-implemented-collision-increments","cor-sm-physics-paired-cloning"]});
    Ok(o)
}
fn doublets(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    use crate::{
        GasError,
        donor::SamplingLaw,
        geometry::{AlgorithmicDistance, ComparisonKind, InteractionKernel},
    };
    let module = &a.gas_config.cloning_donors;
    let ell = r.number("length", 1.).clamp(1e-6, 100.);
    let hs = r.number("phase_scale", 1.).clamp(1e-6, 100.);
    let eps = a.gas_config.clone_decision.epsilon;
    let mut rows = vec![];
    let mut values = vec![];
    let mut actual_residual = C::ZERO;
    let mut total_variance = 0.;
    let mut exact_rows = 0;
    let mut variance_identity: f64 = 0.;
    let mut historical_scalars = std::collections::HashMap::new();
    for step in &a.steps {
        let mut assigned = vec![None; step.before.len()];
        for (i, out) in assigned.iter_mut().enumerate() {
            if !step.report.pre_clone_eligible[i] {
                continue;
            }
            let fi = step.report.pre_clone_fitness.fitness[i];
            let selected = step.report.cloning_companions.row(i).next();
            let mut candidate = vec![];
            for (k, source) in step.report.clone_plan.sources.iter().enumerate() {
                if !module.allow_self
                    && source.frame + 1 == step.report.step
                    && source.slot as usize == i
                {
                    continue;
                }
                let p = if source.frame + 1 == step.report.step
                    && source.version == step.before.version
                {
                    &step.before
                } else {
                    &a.steps
                        .iter()
                        .find(|s| {
                            s.epoch == step.epoch
                                && s.report.step == source.frame + 1
                                && s.before.version == source.version
                        })
                        .ok_or_else(|| {
                            GasError::Capability(
                                "Companion source history is outside the archive".into(),
                            )
                        })?
                        .before
                };
                let distance = module.distance.compare(
                    &step.before.observations,
                    i,
                    &p.observations,
                    source.slot as usize,
                )?;
                let kind = <crate::geometry::Distance as AlgorithmicDistance<f64>>::comparison_kind(
                    &module.distance,
                );
                let d2 = if kind == ComparisonKind::SquaredDistance {
                    distance
                } else {
                    distance * distance
                };
                let fj = step.donor_fitness[k];
                let z =
                    C::phase((fj - fi) / ((fi.abs() + eps) * hs)) * (-d2 / (4. * ell * ell)).exp();
                let logw = module.kernel.log_weight(distance, kind)?;
                candidate.push((k, z, logw));
                if selected == Some(k as u32) {
                    *out = Some(z);
                }
            }
            if module.law == SamplingLaw::Independent && !candidate.is_empty() {
                let high = candidate
                    .iter()
                    .map(|x| x.2)
                    .fold(f64::NEG_INFINITY, f64::max);
                let normal = candidate.iter().map(|x| (x.2 - high).exp()).sum::<f64>();
                let probs: Vec<f64> = candidate
                    .iter()
                    .map(|x| (x.2 - high).exp() / normal)
                    .collect();
                let mean = candidate
                    .iter()
                    .zip(&probs)
                    .fold(C::ZERO, |m, (x, p)| m + x.1 * *p);
                let var = candidate
                    .iter()
                    .zip(&probs)
                    .map(|(x, p)| p * (x.1 - mean).abs2())
                    .sum::<f64>();
                let pair_var = candidate
                    .iter()
                    .enumerate()
                    .map(|(j, x)| {
                        candidate
                            .iter()
                            .enumerate()
                            .map(|(k, y)| 0.5 * probs[j] * probs[k] * (x.1 - y.1).abs2())
                            .sum::<f64>()
                    })
                    .sum::<f64>();
                variance_identity = variance_identity.max((var - pair_var).abs());
                if let Some(z) = out {
                    actual_residual = actual_residual + *z - mean;
                    total_variance += var;
                    exact_rows += 1;
                }
                rows.push(json!({"epoch":step.epoch,"step":step.report.step,"slot":i,"conditional_mean":complex_json(mean),"conditional_variance":var,"pairwise_variance":pair_var,"candidate_probabilities":probs,"candidate_pool_indices":candidate.iter().map(|x|x.0).collect::<Vec<_>>()}));
            }
        }
        for (i, z) in assigned.iter().enumerate() {
            if let Some(z) = z {
                historical_scalars.insert(
                    (
                        step.epoch,
                        step.report.step - 1,
                        step.before.version,
                        i as u32,
                    ),
                    *z,
                );
            }
        }
        for (i, z) in assigned.iter().enumerate() {
            let selected_source = step
                .report
                .cloning_companions
                .row(i)
                .next()
                .map(|j| step.report.clone_plan.sources[j as usize]);
            let second = selected_source.and_then(|s| {
                historical_scalars
                    .get(&(step.epoch, s.frame, s.version, s.slot))
                    .copied()
            });
            let norm = z.zip(second).map(|(x, y)| (x.abs2() + y.abs2()).sqrt());
            let spinor = z
                .zip(second)
                .zip(norm)
                .filter(|(_, n)| *n > 1e-12)
                .map(|((x, y), n)| [complex_json(x / n), complex_json(y / n)]);
            if let Some(z) = z {
                values.push([
                    step.report.step as f64 + i as f64 / assigned.len() as f64,
                    z.im,
                ]);
            }
            rows.push(json!({"epoch":step.epoch,"step":step.report.step,"slot":i,"selected_source":step.report.cloning_companions.row(i).next().map(|j|step.report.clone_plan.sources[j as usize]),"scalar":z.map(complex_json),"companion_own_scalar":second.map(complex_json),"normalized_doublet":spinor,"valid_doublet":spinor.is_some()}));
        }
    }
    let mut o = ExperimentResult::new(
        10,
        "Actual directed companion doublets",
        "Configured donor distance, pool-aligned rescored fitness, and the executed companion-of-companion map",
    );
    o.metric("Exact conditional scalar rows", exact_rows as f64, "rows")
        .metric("Conditional innovation real", actual_residual.re, "")
        .metric("Conditional innovation imaginary", actual_residual.im, "")
        .metric(
            "Conditional complex innovation variance",
            total_variance,
            "",
        )
        .metric(
            "Conditional pairwise variance residual",
            variance_identity,
            "",
        );
    o.plot(
        "Actual selected companion phase",
        "frame plus slot fraction",
        "imaginary scalar",
        vec![Series::line("Directed readout", values)],
    );
    o.details = json!({"status":"available","stage":"pre_clone","length":ell,"phase_scale":hs,"epsilon":eps,"sampler":module,"rows":rows,"conditional_law":"Exact normalized configured kernel for Independent count-one cloning. Mutual samplers retain their actual assigned doublets; their joint-law variance is not replaced by independent-row variance.","doublet_rule":"Second component is the selected companion's own directed scalar at its immutable source frame/version, resolved from the corresponding earlier record when covered; only missing source-record coverage remains masked.","theory_labels":["def-sm-direct-companion-doublet","thm-sm-native-doublet-fluctuations"]});
    Ok(o)
}
