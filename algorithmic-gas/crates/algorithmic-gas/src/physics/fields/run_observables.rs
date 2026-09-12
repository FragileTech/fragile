//! Field experiments whose only data source is an executed Euclidean Gas archive.
//! Constructed empirical observables are named explicitly; no continuum spacetime,
//! equilibrium law, or constitutive equation is inserted as measured evidence.
use super::*;
use crate::{
    Population,
    fractal_set::{EdgeKind, FractalSet},
    physics::geometry::{self, FitnessJet, MetricSpectrum},
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

pub(super) fn run(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    if a.steps.is_empty() {
        return Err(GasError::Capability(
            "field measurement requires completed Euclidean Gas updates".into(),
        ));
    }
    let mut out = match r.experiment {
        37 | 38 | 40 | 62 => metric_geometry(r, a)?,
        41..=43 | 63 => expansion(r, a)?,
        44 | 52 | 54..=56 | 59 | 60 | 64 => graph(r, a)?,
        46 | 47 | 49 | 50 | 66 => mechanics(r, a)?,
        53 | 57 | 58 | 61 | 65 => information(r, a)?,
        _ => {
            return Err(GasError::Configuration(
                "no archive field observable for this experiment".into(),
            ));
        }
    };
    out.details["source"] = json!("recorded_engine_fields");
    out.details["executed_steps"] = json!(a.steps.len());
    out.details["gas_config"] = json!(a.gas_config);
    let final_population = &a.steps.last().unwrap().final_population;
    let position = final_population.observations.field(names(a).0)?;
    let nodes=(0..final_population.len()).filter(|&i|final_population.validity[i].eligible(a.gas_config.include_truncated)).map(|i|{let x=position.row(i)?;Ok(json!({"id":format!("slot-{i}"),"owner":i,"position":[x[0],x.get(1).copied().unwrap_or(0.),x.get(2).copied().unwrap_or(0.)],"layer":"recorded walkers","label":format!("slot {i}"),"detail":format!("Actual final population version {}, slot {}, generation {}",final_population.version,i,final_population.generations[i])}))}).collect::<Result<Vec<_>>>()?;
    out.details["scene"] = json!({"title":out.title,"coordinateSystem":"spatial","nodes":nodes,"edges":[],"faces":[],"message":"Actual final walker positions; measurements use the recorded trajectory and stated observable definitions."});
    if let Some(path) = out
        .details
        .get("recorded_point_path")
        .and_then(Value::as_array)
    {
        let path = path.clone();
        out.details["scene"]["nodes"]=json!(path.iter().enumerate().map(|(i,x)|json!({"id":format!("path-{i}"),"position":[x[0],x[1],x.get(2).cloned().unwrap_or(json!(0.))],"layer":"recorded path points","label":format!("Recorded path point {i}")})).collect::<Vec<_>>());
        out.details["scene"]["edges"]=json!((1..path.len()).map(|i|json!({"source":format!("path-{}",i-1),"target":format!("path-{i}"),"layer":"conditional metric transport path"})).collect::<Vec<_>>());
    }

    out.details["sampling_unit"] = json!(
        "one dependent gas trajectory; empirical descriptors carry no independent-walker uncertainty claim"
    );
    out.details["prediction_scope"] = json!(
        "Displayed exact identities characterize the recorded finite-step algorithm or an explicitly defined observable of its states. Fitted laws are evaluated as hypotheses and are not exact kernel identities."
    );
    Ok(out)
}

fn names(a: &RunArchive<f64>) -> (&str, &str, f64) {
    match &a.gas_config.kinetic.integrator {
        crate::kinetic::KineticKind::Baoab {
            positions,
            velocities,
            dt,
            ..
        } => (positions, velocities, *dt),
        _ => ("positions", "velocities", 1.),
    }
}
fn rows(p: &Population<f64>, field: &str, include: bool) -> Result<Vec<Vec<f64>>> {
    let f = p.observations.field(field)?;
    (0..p.len())
        .filter(|&i| p.validity[i].eligible(include))
        .map(|i| {
            let row = f.row(i)?.to_vec();
            if row.iter().any(|x| !x.is_finite()) {
                return Err(GasError::Numerical(
                    "eligible field row contains nonfinite coordinates".into(),
                ));
            }
            Ok(row)
        })
        .collect()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn transpose(a: &[f64], d: usize) -> Vec<f64> {
    (0..d * d).map(|k| a[(k % d) * d + k / d]).collect()
}
fn mul(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    geometry::matmul(a, b, d)
}
fn identity(d: usize) -> Vec<f64> {
    geometry::identity(d)
}
fn frob(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}
fn mean_cov(x: &[Vec<f64>], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.; d];
    let mut cov = vec![0.; d * d];
    if x.is_empty() {
        return (mean, cov);
    }
    for row in x {
        for j in 0..d {
            mean[j] += row[j] / x.len() as f64;
        }
    }
    for row in x {
        for i in 0..d {
            for j in 0..d {
                cov[i * d + j] += (row[i] - mean[i]) * (row[j] - mean[j]) / x.len() as f64;
            }
        }
    }
    (mean, cov)
}
fn regularized_cov(x: &[Vec<f64>], d: usize, ridge: f64) -> Result<MetricSpectrum<f64>> {
    geometry::metric_spectrum(
        &mean_cov(x, d).1,
        d,
        ridge,
        crate::partv_geometry::MetricPolicy::Strict,
    )
}
fn metric_at(
    a: &RunArchive<f64>,
    record: usize,
    walker: usize,
    x: &[f64],
) -> Result<MetricSpectrum<f64>> {
    let (j, c) = archive_fitness_jet(a, record, walker, x, 2)?;
    let mut hessian = vec![0.; x.len() * x.len()];
    for i in 0..x.len() {
        for k in 0..x.len() {
            hessian[i * x.len() + k] = j.derivative(&[i, k])?;
        }
    }
    let policy = if c["policy"] == "strict" {
        crate::partv_geometry::MetricPolicy::Strict
    } else {
        crate::partv_geometry::MetricPolicy::Clipped
    };
    geometry::metric_spectrum(
        &hessian,
        x.len(),
        c["epsilon"]
            .as_f64()
            .ok_or_else(|| GasError::Capability("metric epsilon missing from provider".into()))?,
        policy,
    )
}
fn metric_geometry(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let record = r.usize("record", a.steps.len() - 1);
    let step = a
        .steps
        .get(record)
        .ok_or_else(|| GasError::Configuration("record outside archive".into()))?;
    let walker = r.usize("walker", 0);
    if walker >= step.before.len()
        || !step.before.validity[walker].eligible(a.gas_config.include_truncated)
    {
        return Err(GasError::Capability(
            "selected geometry walker is unavailable".into(),
        ));
    }
    let f = step.before.observations.field(names(a).0)?;
    let x = f.row(walker)?.to_vec();
    let d = x.len();
    let g = metric_at(a, record, walker, &x)?;
    let mut out = result(
        r,
        match r.experiment {
            37 => "Metric isometries along executed displacements",
            38 => "Recorded-loop frame transport",
            40 => "Conditional metric connection",
            _ => "Measured fitness curvature scale",
        },
        "Conditional fitness geometry reconstructed from the executed donor context",
    );
    out.details = json!({"record":record,"walker":walker,"point":x,"dimension":d,"conditional_context":archive_fitness_jet(a,record,walker,&x,2)?.1});
    if r.experiment == 62 {
        let (j, c) = archive_fitness_jet(a, record, walker, &x, 4)?;
        let policy = if c["policy"] == "strict" {
            crate::partv_geometry::MetricPolicy::Strict
        } else {
            crate::partv_geometry::MetricPolicy::Clipped
        };
        let curvature = geometry::fitness_curvature(
            &FitnessJet::from_jet(&j)?,
            c["epsilon"].as_f64().unwrap(),
            policy,
            c["clipping_threshold"].as_f64().unwrap(),
            false,
        )?;
        metric(
            &mut out,
            "Scalar curvature",
            curvature.scalar,
            "inverse metric length squared",
        );
        if curvature.scalar != 0. {
            metric(
                &mut out,
                "Absolute scalar curvature radius",
                (d as f64 * (d - 1) as f64 / curvature.scalar.abs()).sqrt(),
                "metric length",
            );
        }
        let isotropic: Vec<f64> = g
            .metric
            .iter()
            .map(|x| curvature.scalar / d as f64 * x)
            .collect();
        metric(
            &mut out,
            "Ricci isotropy residual",
            frob(
                &curvature
                    .ricci
                    .iter()
                    .zip(&isotropic)
                    .map(|(x, y)| x - y)
                    .collect::<Vec<_>>(),
            ),
            "Ricci norm",
        );
        plot(
            &mut out,
            "Measured Ricci versus constant-curvature hypothesis",
            "tensor component",
            "curvature",
            vec![
                line(
                    "Measured Ricci",
                    curvature
                        .ricci
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| [i as f64, x])
                        .collect(),
                ),
                line(
                    "R g / dimension",
                    isotropic
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| [i as f64, x])
                        .collect(),
                ),
            ],
        );
        out.details["curvature"] = json!(curvature);
        out.details["interpretation"] = json!(
            "The radius is a scalar-derived diagnostic. The Ricci residual tests local isotropy; no AdS radius or cosmological constant is assumed."
        );
    } else if r.experiment == 40 {
        let h = p(r, "difference_step", 1e-4, 1e-7, 0.01);
        let mut dg = vec![0.; d * d * d];
        let mut dg2 = vec![0.; d * d * d];
        for k in 0..d {
            for (scale, target) in [(h, &mut dg), (h / 2., &mut dg2)] {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[k] += scale;
                xm[k] -= scale;
                let gp = metric_at(a, record, walker, &xp)?;
                let gm = metric_at(a, record, walker, &xm)?;
                for ij in 0..d * d {
                    target[k * d * d + ij] = (gp.metric[ij] - gm.metric[ij]) / (2. * scale);
                }
            }
        }
        let mut gamma = vec![0.; d * d * d];
        for k in 0..d {
            for i in 0..d {
                for j in 0..d {
                    for l in 0..d {
                        gamma[(k * d + i) * d + j] += 0.5
                            * g.inverse[k * d + l]
                            * (dg2[i * d * d + l * d + j] + dg2[j * d * d + l * d + i]
                                - dg2[l * d * d + i * d + j]);
                    }
                }
            }
        }
        let mut compatibility = 0_f64;
        for k in 0..d {
            for i in 0..d {
                for j in 0..d {
                    let reconstructed = (0..d)
                        .map(|l| {
                            gamma[(l * d + k) * d + i] * g.metric[l * d + j]
                                + gamma[(l * d + k) * d + j] * g.metric[i * d + l]
                        })
                        .sum::<f64>();
                    compatibility =
                        compatibility.max((dg2[k * d * d + i * d + j] - reconstructed).abs());
                }
            }
        }
        metric(
            &mut out,
            "Metric compatibility residual",
            compatibility,
            "metric derivative",
        );
        metric(
            &mut out,
            "Derivative refinement discrepancy",
            maxdiff(&dg, &dg2),
            "metric derivative",
        );
        plot(
            &mut out,
            "Connection of recorded conditional metric",
            "connection component",
            "inverse coordinate",
            vec![line(
                "Christoffel coefficients",
                gamma
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            )],
        );
        out.details["connection"] = json!(gamma);
        out.details["difference_step"] = json!(h);
        out.notes.push("Centered finite differences retain the configured spectral clipping. Refinement discrepancy exposes threshold crossings; connection values are not certified derivatives at a clipping kink.".into());
    } else {
        let xf = step
            .final_population
            .observations
            .field(names(a).0)?
            .row(walker)?
            .to_vec();
        let mut path = vec![x.clone(), xf.clone()];
        if r.experiment == 38 {
            let neighbor = (0..step.before.len())
                .filter(|&i| {
                    i != walker && step.before.validity[i].eligible(a.gas_config.include_truncated)
                })
                .min_by(|&i, &j| {
                    let di = f
                        .row(i)
                        .unwrap()
                        .iter()
                        .zip(&x)
                        .map(|(u, v)| (u - v).powi(2))
                        .sum::<f64>();
                    let dj = f
                        .row(j)
                        .unwrap()
                        .iter()
                        .zip(&x)
                        .map(|(u, v)| (u - v).powi(2))
                        .sum::<f64>();
                    di.total_cmp(&dj)
                })
                .ok_or_else(|| {
                    GasError::Capability("loop needs two eligible recorded walkers".into())
                })?;
            path.extend([f.row(neighbor)?.to_vec(), x.clone()]);
        }
        let resolution = n(r, "transport_steps", 4, 1, 32);
        let coarse = transport_path(a, record, walker, &path, resolution)?;
        let transport = transport_path(a, record, walker, &path, 2 * resolution)?;
        let target = metric_at(a, record, walker, path.last().unwrap())?;
        let recovered = mul(
            &mul(&transpose(&transport, d), &target.metric, d),
            &transport,
            d,
        );
        let norm_residual = maxdiff(&recovered, &g.metric);
        metric(&mut out, "Metric norm residual", norm_residual, "metric");
        metric(
            &mut out,
            "Transport refinement discrepancy",
            maxdiff(&transport, &coarse),
            "transport coefficient",
        );
        if r.experiment == 38 {
            metric(
                &mut out,
                "Recorded loop holonomy departure",
                frob(
                    &transport
                        .iter()
                        .zip(identity(d))
                        .map(|(x, y)| x - y)
                        .collect::<Vec<_>>(),
                ),
                "transport coefficient",
            );
        }
        plot(
            &mut out,
            "Levi-Civita transport along recorded-point segments",
            "matrix component",
            "coefficient",
            vec![
                line(
                    "Refined conditional-metric transport",
                    transport
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| [i as f64, x])
                        .collect(),
                ),
                line(
                    "Coarse transport",
                    coarse
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| [i as f64, x])
                        .collect(),
                ),
            ],
        );
        out.details["recorded_point_path"] = json!(path);
        out.details["transport_matrix"] = json!(transport);
        out.details["transport_steps"] = json!(2 * resolution);
        out.details["transport_definition"] = json!(
            "Integrate du^k/ds = -Gamma^k_ij(x(s)) dx^i/ds u^j along straight segments between actual recorded points, using the selected frozen conditional fitness metric and implicit midpoint solves. Gamma is computed by centered finite differences of that metric. Coarse/refined transport and metric norm residuals diagnose numerical accuracy."
        );
        out.notes.push("Geometry is evaluated along segments joining recorded states in their actual coordinate chart. Clipping derivative kinks and large steps can prevent convergence; residuals remain visible. This is spatial metric holonomy, not an assumed Lorentzian or gauge connection.".into());
    }
    Ok(out)
}

fn transport_path(
    a: &RunArchive<f64>,
    record: usize,
    walker: usize,
    path: &[Vec<f64>],
    steps: usize,
) -> Result<Vec<f64>> {
    let d = path[0].len();
    let mut transport = identity(d);
    for segment in path.windows(2) {
        let dx: Vec<f64> = segment[1]
            .iter()
            .zip(&segment[0])
            .map(|(a, b)| (a - b) / steps as f64)
            .collect();
        for n in 0..steps {
            let point: Vec<f64> = segment[0]
                .iter()
                .zip(&dx)
                .map(|(x, v)| x + (n as f64 + 0.5) * v)
                .collect();
            let g = metric_at(a, record, walker, &point)?;
            let h = 1e-5;
            let mut dg = vec![0.; d * d * d];
            for k in 0..d {
                let mut xp = point.clone();
                let mut xm = point.clone();
                xp[k] += h;
                xm[k] -= h;
                let gp = metric_at(a, record, walker, &xp)?;
                let gm = metric_at(a, record, walker, &xm)?;
                for ij in 0..d * d {
                    dg[k * d * d + ij] = (gp.metric[ij] - gm.metric[ij]) / (2. * h);
                }
            }
            let mut contracted = vec![0.; d * d];
            for k in 0..d {
                for j in 0..d {
                    for i in 0..d {
                        for l in 0..d {
                            contracted[k * d + j] += 0.5
                                * g.inverse[k * d + l]
                                * (dg[i * d * d + l * d + j] + dg[j * d * d + l * d + i]
                                    - dg[l * d * d + i * d + j])
                                * dx[i];
                        }
                    }
                }
            }
            let mut lhs = vec![vec![0.; d]; d];
            let mut rhs = identity(d);
            for i in 0..d {
                for j in 0..d {
                    lhs[i][j] = f64::from(i == j) + contracted[i * d + j] / 2.;
                    rhs[i * d + j] -= contracted[i * d + j] / 2.;
                }
            }
            let mut local = vec![0.; d * d];
            for j in 0..d {
                let column = solve(lhs.clone(), (0..d).map(|i| rhs[i * d + j]).collect())?;
                for i in 0..d {
                    local[i * d + j] = column[i];
                }
            }
            transport = mul(&local, &transport, d);
        }
    }
    Ok(transport)
}

fn expansion(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let (position, _, dt) = names(a);
    let ridge = p(r, "covariance_ridge", 1e-6, 1e-12, 1.);
    let d = a.steps[0].before.observations.field(position)?.width();
    let mut out = result(
        r,
        match r.experiment {
            41 => "Finite-step cloud expansion balance",
            42 => "Recorded material and population volumes",
            43 => "Measured cloud focusing",
            _ => "Measured scale-factor evolution",
        },
        "Regularized empirical covariance volume measured at actual operator stages",
    );
    let mut direct = vec![];
    let mut ledger = vec![];
    let mut volumes = vec![];
    let mut expansion = vec![];
    let mut stage_rows = vec![];
    let mut focus = vec![];
    let mut last_theta = None;
    let mut worst = 0_f64;
    for (index, s) in a.steps.iter().enumerate() {
        let before = rows(&s.before, position, a.gas_config.include_truncated)?;
        let after = rows(
            &s.final_population,
            position,
            a.gas_config.include_truncated,
        )?;
        let old = regularized_cov(&before, d, ridge)?;
        let new = regularized_cov(&after, d, ridge)?;
        let delta = (new.log_determinant - old.log_determinant) / 2.;
        let theta = delta / dt;
        let mut prev = old.log_determinant / 2.;
        let mut sum = 0.;
        let mut contributions = vec![];
        for stage in &s.stages {
            let field = stage.fields.get(position).ok_or_else(|| {
                GasError::Capability("stage positions missing for volume budget".into())
            })?;
            let x: Vec<Vec<f64>> = stage
                .validity
                .iter()
                .enumerate()
                .filter(|(_, v)| v.eligible(a.gas_config.include_truncated))
                .map(|(i, _)| field.values[i * d..(i + 1) * d].to_vec())
                .collect();
            let next = regularized_cov(&x, d, ridge)?.log_determinant / 2.;
            let change = next - prev;
            sum += change;
            prev = next;
            contributions
                .push(json!({"stage":stage.stage,"log_volume_change":change,"eligible":x.len()}));
        }
        let terminal = new.log_determinant / 2. - prev;
        sum += terminal;
        worst = worst.max((sum - delta).abs());
        let time = (index + 1) as f64 * dt;
        direct.push([time, delta]);
        ledger.push([time, sum]);
        volumes.push([time, (new.log_determinant / (2. * d as f64)).exp()]);
        expansion.push([time, theta]);
        if let Some(last) = last_theta {
            focus.push([time, (theta - last) / dt + theta * theta / d as f64]);
        }
        last_theta = Some(theta);
        stage_rows.push(json!({"step":s.report.step,"stages":contributions,"terminal_change":terminal,"before_eligible":before.len(),"after_eligible":after.len(),"delta_log_volume":delta,"sum_stage_changes":sum}));
    }
    metric(
        &mut out,
        "Log-volume telescope residual",
        worst,
        "log volume",
    );
    match r.experiment {
        41 => plot(
            &mut out,
            "Direct expansion versus executed-stage budget",
            "physical time",
            "log volume increment",
            vec![
                line("Direct endpoint difference", direct),
                line("Sum of recorded stage changes", ledger),
            ],
        ),
        42 => {
            plot(
                &mut out,
                "Population covariance length scale",
                "physical time",
                "coordinate",
                vec![line("det(C + ridge I)^(1/(2d))", volumes)],
            );
            plot(
                &mut out,
                "Volume replacement and transport budget",
                "physical time",
                "log volume increment",
                vec![
                    line("Executed endpoint", direct),
                    line("Executed stage reconstruction", ledger),
                ],
            );
        }
        43 => {
            plot(
                &mut out,
                "Measured focusing residual",
                "physical time",
                "inverse time squared",
                vec![line("Delta theta / dt + theta squared / dimension", focus)],
            );
            plot(
                &mut out,
                "Measured expansion",
                "physical time",
                "inverse time",
                vec![line("Delta log volume / dt", expansion)],
            );
        }
        _ => {
            plot(
                &mut out,
                "Scale factor and expansion of the executed cloud",
                "physical time",
                "coordinate",
                vec![line("Empirical cloud scale", volumes)],
            );
            plot(
                &mut out,
                "Scale evolution",
                "physical time",
                "inverse time",
                vec![line("Expansion theta", expansion)],
            );
        }
    }
    out.details = json!({"covariance_ridge":ridge,"volume_definition":"V = sqrt(det(population covariance + ridge I)); scale = V^(1/d)","stage_budget":stage_rows,"interpretation":"Population volume includes cloning, killing and eligibility changes. A nonzero focusing residual is measured; no Lorentzian congruence or Riccati closure is imposed.","extinction":"The regularized empty-cloud covariance is ridge I; eligible counts distinguish this constructed floor from an occupied material volume."});
    Ok(out)
}

fn mechanics(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let (position, velocity, dt) = names(a);
    let d = a.steps[0].before.observations.field(position)?.width();
    let wave = p(r, "wave_number", 1., 0.05, 16.);
    let mut out = result(
        r,
        match r.experiment {
            46 => "Executed virial and dilation work",
            47 => "Recorded density-mode evolution",
            49 => "Recorded velocity-mode energy",
            50 => "Measured kinetic and spatial scales",
            _ => "Dimensionless measured gas scales",
        },
        "Finite population observables and stage identities of the executed gas",
    );
    let mut measured = vec![];
    let mut predicted = vec![];
    let mut second = vec![];
    let mut reports = vec![];
    let mut maxres = 0_f64;
    for (index, s) in a.steps.iter().enumerate() {
        let xfield = s.final_population.observations.field(position)?;
        let vfield = s.final_population.observations.field(velocity)?;
        let eligible: Vec<usize> = (0..s.final_population.len())
            .filter(|&i| s.final_population.validity[i].eligible(a.gas_config.include_truncated))
            .collect();
        let count = eligible.len();
        let time = (index + 1) as f64 * dt;
        if count == 0 {
            reports.push(json!({"step":s.report.step,"eligible":0,"status":"extinct"}));
            continue;
        }
        let x: Vec<Vec<f64>> = eligible
            .iter()
            .map(|&i| xfield.row(i).unwrap().to_vec())
            .collect();
        let v: Vec<Vec<f64>> = eligible
            .iter()
            .map(|&i| vfield.row(i).unwrap().to_vec())
            .collect();
        let (xm, xc) = mean_cov(&x, d);
        let (vm, vc) = mean_cov(&v, d);
        let radius = (0..d).map(|i| xc[i * d + i]).sum::<f64>().sqrt();
        let speed = (v.iter().map(|row| dot(row, row)).sum::<f64>() / count as f64).sqrt();
        if r.experiment == 46 {
            let mut total_direct = 0.;
            let mut total_reconstructed = 0.;
            let mut transitions = vec![];
            for pair in s.stages.windows(2) {
                let old = &pair[0];
                let new = &pair[1];
                let ox = &old.fields[position].values;
                let nx = &new.fields[position].values;
                let ov = &old.fields[velocity].values;
                let nv = &new.fields[velocity].values;
                let mut delta = 0.;
                let mut impulse = 0.;
                let mut transport = 0.;
                for row in 0..old.validity.len() {
                    for j in 0..d {
                        let k = row * d + j;
                        let xx = if old.validity[row].eligible(a.gas_config.include_truncated) {
                            ox[k]
                        } else {
                            0.
                        };
                        let vv = if old.validity[row].eligible(a.gas_config.include_truncated) {
                            ov[k]
                        } else {
                            0.
                        };
                        let yy = if new.validity[row].eligible(a.gas_config.include_truncated) {
                            nx[k]
                        } else {
                            0.
                        };
                        let ww = if new.validity[row].eligible(a.gas_config.include_truncated) {
                            nv[k]
                        } else {
                            0.
                        };
                        delta += yy * ww - xx * vv;
                        impulse += xx * (ww - vv);
                        transport += (yy - xx) * ww;
                    }
                }
                total_direct += delta;
                total_reconstructed += impulse + transport;
                transitions.push(json!({"from":old.stage,"to":new.stage,"virial_change":delta,"impulse":impulse,"transport":transport}));
            }
            maxres = maxres.max((total_direct - total_reconstructed).abs());
            measured.push([time, total_direct]);
            predicted.push([time, total_reconstructed]);
            reports.push(json!({"step":s.report.step,"stages":transitions}));
        } else if r.experiment == 47 {
            let re = x.iter().map(|row| (wave * row[0]).cos()).sum::<f64>() / count as f64;
            let im = x.iter().map(|row| (wave * row[0]).sin()).sum::<f64>() / count as f64;
            let source = s.before.observations.field(position)?;
            let mut predicted_delta = [0.; 2];
            let mut old = [0.; 2];
            let capacity = s.before.len() as f64;
            for i in 0..s.before.len() {
                if s.before.validity[i].eligible(a.gas_config.include_truncated) {
                    let phase = wave * source.row(i)?[0];
                    old[0] += phase.cos() / capacity;
                    old[1] += phase.sin() / capacity;
                }
                if s.final_population.validity[i].eligible(a.gas_config.include_truncated) {
                    let phase = wave * xfield.row(i)?[0];
                    predicted_delta[0] += phase.cos() / capacity;
                    predicted_delta[1] += phase.sin() / capacity;
                }
            }
            predicted_delta[0] -= old[0];
            predicted_delta[1] -= old[1];
            measured.push([time, re]);
            second.push([time, im]);
            predicted.push([time, (re * re + im * im).sqrt()]);
            reports.push(json!({"step":s.report.step,"capacity_normalized_mode_increment":predicted_delta,"eligible":count,"normalization":"plot uses eligible count; finite-step increments use fixed capacity and zero extension"}));
        } else if r.experiment == 49 {
            // The DFT is over the explicitly ordered eligible walker slots; Parseval
            // tests the transform without inventing spatial independence or masses.
            let mut energies = vec![0.; count];
            for (k, energy) in energies.iter_mut().enumerate() {
                for j in 0..d {
                    let mut re = 0.;
                    let mut im = 0.;
                    for (i, row) in v.iter().enumerate() {
                        let phase = std::f64::consts::TAU * (i * k) as f64 / count as f64;
                        re += row[j] * phase.cos();
                        im += row[j] * phase.sin();
                    }
                    *energy += (re * re + im * im) / (2. * count as f64);
                }
            }
            let direct = v.iter().map(|row| dot(row, row) / 2.).sum::<f64>();
            let reconstructed = energies.iter().sum::<f64>();
            maxres = maxres.max((direct - reconstructed).abs());
            measured.push([time, direct]);
            predicted.push([time, reconstructed]);
            reports.push(json!({"step":s.report.step,"ordered_eligible_slots":eligible,"mode_energies":energies}));
        } else if r.experiment == 50 {
            measured.push([time, radius]);
            predicted.push([time, speed]);
            second.push([time, (0..d).map(|j| vc[j * d + j]).sum::<f64>()]);
            reports.push(json!({"step":s.report.step,"mean_position":xm,"mean_velocity":vm,"position_covariance":xc,"velocity_covariance":vc,"crossing_time":if speed>0.{Some(radius/speed)}else{None}}));
        } else {
            measured.push([time, if radius > 0. { dt * speed / radius } else { 0. }]);
            predicted.push([time, count as f64 / s.before.len() as f64]);
            reports.push(json!({"step":s.report.step,"length_unit":radius,"speed_unit":speed,"time_unit":if speed>0.{Some(radius/speed)}else{None},"dimensionless_step":if radius>0.{Some(dt*speed/radius)}else{None},"scale_status":if radius>0.&&speed>0.{"available"}else{"degenerate empirical unit"}}));
        }
    }
    match r.experiment {
        46 => {
            metric(
                &mut out,
                "Virial product-rule residual",
                maxres,
                "coordinate times velocity",
            );
            plot(
                &mut out,
                "Executed virial balance",
                "physical time",
                "virial increment",
                vec![
                    line("Direct stage increment", measured),
                    line("Impulse plus transport", predicted),
                ],
            );
        }
        47 => {
            plot(
                &mut out,
                "Density Fourier mode of actual positions",
                "physical time",
                "normalized mode",
                vec![
                    line("Real part", measured),
                    line("Imaginary part", second),
                    line("Magnitude", predicted),
                ],
            );
        }
        49 => {
            metric(&mut out, "Parseval energy residual", maxres, "energy");
            plot(
                &mut out,
                "Kinetic energy from walkers and modes",
                "physical time",
                "energy",
                vec![
                    line("Sum of walker energies", measured),
                    line("Sum of slot DFT mode energies", predicted),
                ],
            );
        }
        50 => {
            plot(
                &mut out,
                "Measured spatial and kinetic scales",
                "physical time",
                "coordinate / velocity",
                vec![
                    line("Position RMS spread", measured),
                    line("Velocity RMS", predicted),
                ],
            );
            plot(
                &mut out,
                "Centered kinetic variance",
                "physical time",
                "velocity squared",
                vec![line("Trace velocity covariance", second)],
            );
        }
        _ => {
            plot(
                &mut out,
                "Run-derived dimensionless scales",
                "physical time",
                "dimensionless",
                vec![
                    line("Step / empirical crossing time", measured),
                    line("Eligible fraction", predicted),
                ],
            );
        }
    }
    out.details = json!({"wave_number":wave,"records":reports,"interpretation":match r.experiment{46=>"Virial product rule includes every recorded stage and eligibility by zero extension; no deformation pressure is assumed.",47=>"Mode phases and amplitudes come from actual positions; oscillation or decay is not assigned a mass.",49=>"Orthonormal Fourier transform over eligible slot order; this is a complete kinetic-energy decomposition, not a dispersion relation.",50=>"Empirical cloud and velocity scales; no ideal-gas or cosmological crossover law is assumed.",_=>"Units are constructed from measured RMS length and speed; no particle mass anchor or physical cosmological scale is inserted."}});
    Ok(out)
}

fn selected(kind: EdgeKind) -> bool {
    matches!(
        kind,
        EdgeKind::IgDistance
            | EdgeKind::IgCloning
            | EdgeKind::HistoricalDistance
            | EdgeKind::HistoricalCloning
    )
}
fn flow(capacity: &[Vec<f64>], source: usize, sink: usize) -> (f64, Vec<bool>) {
    let n = capacity.len();
    let mut residual = capacity.to_vec();
    let mut total = 0.;
    loop {
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut queue = VecDeque::from([source]);
        while let Some(u) = queue.pop_front() {
            for (v, &c) in residual[u].iter().enumerate() {
                if c > 1e-12 && parent[v] == usize::MAX {
                    parent[v] = u;
                    queue.push_back(v);
                }
            }
        }
        if parent[sink] == usize::MAX {
            return (total, parent.iter().map(|&p| p != usize::MAX).collect());
        }
        let mut amount = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v];
            amount = amount.min(residual[u][v]);
            v = u;
        }
        v = sink;
        while v != source {
            let u = parent[v];
            residual[u][v] -= amount;
            residual[v][u] += amount;
            v = u;
        }
        total += amount;
    }
}
fn graph(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let g = FractalSet::from_archive(a);
    let capacity = a.steps[0].before.len();
    if capacity < 2 {
        return Err(GasError::Capability(
            "graph field experiment requires at least two walker slots".into(),
        ));
    }
    let mut adjacency = vec![vec![0.; capacity]; capacity];
    let observed: Vec<_> = g.edges.iter().filter(|e| selected(e.kind)).collect();
    for e in &observed {
        adjacency[e.source.slot as usize][e.target.slot as usize] += 1.;
    }
    let fraction = p(r, "partition_fraction", 0.5, 0., 1.);
    let split = ((fraction * capacity as f64).round() as usize).clamp(1, capacity - 1);
    let mut out = result(
        r,
        match r.experiment {
            44 => "Recorded Fractal Set topology",
            52 => "Cuts of executed interaction edges",
            54 => "Recorded interaction perimeter scan",
            55 => "Empirical companion sampling support",
            56 => "Discrete cut variation",
            59 => "Recorded ancestry depth and boundaries",
            60 => "Minimum cut of selected interactions",
            _ => "Information carried by selected interactions",
        },
        "Directed recorded interaction graph with unit capacity per executed selected edge",
    );
    out.details = json!({"graph_nodes":g.nodes.len(),"graph_edges":g.edges.len(),"interaction_edges":observed.len(),"unresolved_sources":g.unresolved_sources,"projection":"Edges are projected to persistent slot labels for cut and information calculations; counts sum all retained epochs and steps, including historical selected edges.","capacity_convention":"one unit for each actual selected directed edge; no inferred dense Gaussian graph","partition_split_slot":split});
    let channels = [
        EdgeKind::IgDistance,
        EdgeKind::IgCloning,
        EdgeKind::HistoricalDistance,
        EdgeKind::HistoricalCloning,
    ];
    out.details["recorded_interaction_cut"] = json!({"split_slot":split,"channels":channels.iter().map(|&kind|json!({"channel":kind,"observed_edges":observed.iter().filter(|e|e.kind==kind).count(),"crossing_edges":observed.iter().filter(|e|e.kind==kind&&((e.source.slot as usize)<split)!=((e.target.slot as usize)<split)).count()})).collect::<Vec<_>>()});
    match r.experiment {
        44 => {
            let mut parent: Vec<usize> = (0..g.nodes.len()).collect();
            let indices: BTreeMap<_, _> =
                g.nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            fn root(parent: &mut [usize], mut i: usize) -> usize {
                while parent[i] != i {
                    parent[i] = parent[parent[i]];
                    i = parent[i];
                }
                i
            }
            let mut edges = 0;
            let mut missing = 0;
            for e in &g.edges {
                if let (Some(&i), Some(&j)) = (indices.get(&e.source), indices.get(&e.target)) {
                    let ri = root(&mut parent, i);
                    let rj = root(&mut parent, j);
                    parent[ri] = rj;
                    edges += 1;
                } else {
                    missing += 1;
                }
            }
            let components = (0..parent.len())
                .map(|i| root(&mut parent, i))
                .collect::<BTreeSet<_>>()
                .len();
            let cycle_rank = edges + components - g.nodes.len();
            metric(
                &mut out,
                "Recorded vertices",
                g.nodes.len() as f64,
                "events",
            );
            metric(
                &mut out,
                "Recorded edges with resolved endpoints",
                edges as f64,
                "edges",
            );
            metric(
                &mut out,
                "Undirected graph cycle rank",
                cycle_rank as f64,
                "cycles",
            );
            metric(
                &mut out,
                "Graph Euler identity residual",
                ((g.nodes.len() + cycle_rank) as f64 - (edges + components) as f64).abs(),
                "count",
            );
            let mut boundary_errors = vec![];
            for (i, t) in g.triangles.iter().enumerate() {
                let mut boundary: BTreeMap<_, i32> = BTreeMap::new();
                for (&edge, sign) in t.boundary_edges.iter().zip([1, 1, -1]) {
                    let e = &g.edges[edge];
                    *boundary.entry(e.source).or_default() -= sign;
                    *boundary.entry(e.target).or_default() += sign;
                }
                boundary_errors.push([
                    i as f64,
                    boundary.values().map(|x| x.abs()).sum::<i32>() as f64,
                ]);
            }
            plot(
                &mut out,
                "Boundary of recorded interaction triangles",
                "triangle index",
                "boundary residual",
                vec![line("Boundary of boundary", boundary_errors)],
            );
            out.details["topology"] = json!({"components":components,"cycle_rank":cycle_rank,"resolved_edges":edges,"missing_endpoint_edges":missing,"interaction_triangles":g.triangles.len(),"interpretation":"The graph identity V-E = components-cycle_rank is exact. Triangles use recorded CST+IA-IG boundaries; no triangulated spacetime or Gauss-Bonnet curvature is supplied."});
        }
        52 | 54 | 56 => {
            let cuts: Vec<f64> = (0..=capacity)
                .map(|k| {
                    observed
                        .iter()
                        .filter(|e| {
                            ((e.source.slot as usize) < k) != ((e.target.slot as usize) < k)
                        })
                        .count() as f64
                })
                .collect();
            let mut changes = vec![];
            let mut predicted = vec![];
            for k in 0..capacity {
                let delta = cuts[k + 1] - cuts[k];
                let pred = observed
                    .iter()
                    .filter(|e| e.source.slot != e.target.slot)
                    .map(|e| {
                        let s = e.source.slot as usize;
                        let t = e.target.slot as usize;
                        if s == k {
                            if t < k { -1. } else { 1. }
                        } else if t == k {
                            if s < k { -1. } else { 1. }
                        } else {
                            0.
                        }
                    })
                    .sum::<f64>();
                changes.push([k as f64, delta]);
                predicted.push([k as f64, pred]);
            }
            let residual = changes
                .iter()
                .zip(&predicted)
                .map(|(a, b)| (a[1] - b[1]).abs())
                .fold(0., f64::max);
            metric(
                &mut out,
                "Selected partition cut",
                cuts[split],
                "directed edges",
            );
            metric(
                &mut out,
                "Discrete cut first-variation residual",
                residual,
                "edges",
            );
            if r.experiment == 52 {
                let kinds = [
                    EdgeKind::IgDistance,
                    EdgeKind::IgCloning,
                    EdgeKind::HistoricalDistance,
                    EdgeKind::HistoricalCloning,
                ];
                let channels: Vec<_> = kinds
                    .iter()
                    .enumerate()
                    .map(|(i, &kind)| {
                        [
                            i as f64,
                            observed
                                .iter()
                                .filter(|e| {
                                    e.kind == kind
                                        && ((e.source.slot as usize) < split)
                                            != ((e.target.slot as usize) < split)
                                })
                                .count() as f64,
                        ]
                    })
                    .collect();
                plot(
                    &mut out,
                    "Actual edge cuts by channel",
                    "distance / cloning / historical distance / historical cloning",
                    "edges",
                    vec![line("Crossing selected edges", channels)],
                );
            }
            if r.experiment != 56 {
                plot(
                    &mut out,
                    "Cut perimeter of nested slot regions",
                    "region size",
                    "executed edge count",
                    vec![line(
                        "Actual selected interaction cut",
                        cuts.iter()
                            .enumerate()
                            .map(|(i, &v)| [i as f64, v])
                            .collect(),
                    )],
                );
            } else {
                plot(
                    &mut out,
                    "Exact first variation of recorded graph perimeter",
                    "added slot",
                    "cut increment",
                    vec![
                        line("Measured cut difference", changes),
                        line("Incident edge prediction", predicted),
                    ],
                );
            }
            out.details["cuts"] = json!(cuts);
            out.details["interpretation"] = json!(
                "This is a measured directed cardinality perimeter of nested slot regions, not spatial area or quantum entanglement entropy."
            );
        }
        55 => {
            let mut entropy_points = vec![];
            let mut support = vec![];
            let mut distributions = vec![];
            for (i, row) in adjacency.iter().enumerate() {
                let total = row.iter().sum::<f64>();
                if total == 0. {
                    continue;
                }
                let probs: Vec<f64> = row.iter().map(|x| x / total).collect();
                let h = entropy(&probs);
                let k = row.iter().filter(|&&x| x > 0.).count();
                entropy_points.push([i as f64, h]);
                support.push([i as f64, (k as f64).ln()]);
                distributions.push(json!({"source_slot":i,"observed_count":total,"target_frequencies":probs,"support":k}));
            }
            plot(
                &mut out,
                "Observed companion diversity",
                "source slot",
                "nats",
                vec![
                    line("Empirical target entropy", entropy_points),
                    line("Log observed support bound", support),
                ],
            );
            out.details["empirical_sampling"] = json!(distributions);
            out.details["interpretation"] = json!(
                "Target frequencies summarize dependent executed selections. Entropy <= log observed support is an exact discrete bound; frequencies are not asserted conditional sampler probabilities."
            );
        }
        59 => {
            let mut depths: BTreeMap<_, usize> = g.nodes.iter().map(|&v| (v, 0)).collect();
            let mut ancestry: Vec<_> = g
                .edges
                .iter()
                .filter(|e| matches!(e.kind, EdgeKind::Ancestry | EdgeKind::Persistence))
                .collect();
            ancestry.sort_by_key(|e| e.target);
            for e in ancestry {
                let depth = depths.get(&e.source).copied().unwrap_or(0) + 1;
                depths
                    .entry(e.target)
                    .and_modify(|x| *x = (*x).max(depth))
                    .or_insert(depth);
            }
            let mut counts = BTreeMap::new();
            for &depth in depths.values() {
                *counts.entry(depth).or_insert(0usize) += 1;
            }
            plot(
                &mut out,
                "Ancestral depth of recorded events",
                "ancestry edges from recorded start",
                "event count",
                vec![line(
                    "Recorded event depth histogram",
                    counts
                        .iter()
                        .map(|(&depth, &count)| [depth as f64, count as f64])
                        .collect(),
                )],
            );
            metric(
                &mut out,
                "Maximum recorded ancestral depth",
                depths.values().copied().max().unwrap_or(0) as f64,
                "edges",
            );
            out.details["depth_histogram"] = json!(counts);
            out.details["interpretation"] = json!(
                "Depth is measured from retained ancestry/persistence edges. Missing pre-recording ancestry truncates depth; no spacetime horizon is inferred."
            );
        }
        60 => {
            let source = r.usize("source_slot", 0);
            let sink = r.usize("sink_slot", capacity - 1);
            if source >= capacity || sink >= capacity || source == sink {
                return Err(GasError::Configuration(
                    "distinct source and sink slots inside capacity required".into(),
                ));
            }
            let (value, set) = flow(&adjacency, source, sink);
            let cut = (0..capacity)
                .flat_map(|i| (0..capacity).map(move |j| (i, j)))
                .filter(|&(i, j)| set[i] && !set[j])
                .map(|(i, j)| adjacency[i][j])
                .sum::<f64>();
            metric(
                &mut out,
                "Maximum directed flow",
                value,
                "selected edge count",
            );
            metric(&mut out, "Minimum directed cut", cut, "selected edge count");
            metric(
                &mut out,
                "Max-flow min-cut residual",
                (value - cut).abs(),
                "edges",
            );
            plot(
                &mut out,
                "Measured cut capacity by source slot",
                "source slot",
                "crossing edge count",
                vec![line(
                    "Outgoing capacity across minimum partition",
                    (0..capacity)
                        .map(|i| {
                            [
                                i as f64,
                                if set[i] {
                                    (0..capacity)
                                        .filter(|&j| !set[j])
                                        .map(|j| adjacency[i][j])
                                        .sum()
                                } else {
                                    0.
                                },
                            ]
                        })
                        .collect(),
                )],
            );
            out.details["minimum_cut"] = json!({"source_slot":source,"sink_slot":sink,"source_side":set,"flow":value,"cut":cut});
        }
        _ => {
            let total = observed.len() as f64;
            let mut joint = vec![vec![0.; 2]; 2];
            for e in observed {
                joint[usize::from((e.source.slot as usize) >= split)]
                    [usize::from((e.target.slot as usize) >= split)] += 1.;
            }
            if total == 0. {
                return Err(GasError::Capability(
                    "no selected interactions recorded for information measurement".into(),
                ));
            }
            for row in &mut joint {
                for v in row {
                    *v /= total;
                }
            }
            let px = vec![joint[0].iter().sum(), joint[1].iter().sum()];
            let py = vec![joint[0][0] + joint[1][0], joint[0][1] + joint[1][1]];
            let hxy = entropy(&joint.iter().flatten().copied().collect::<Vec<_>>());
            let hx = entropy(&px);
            let hy = entropy(&py);
            let mi = hx + hy - hxy;
            metric(
                &mut out,
                "Source-target region mutual information",
                mi,
                "nats",
            );
            metric(&mut out, "Conditional target entropy", hxy - hx, "nats");
            metric(
                &mut out,
                "Information chain-rule residual",
                (hy - (hxy - hx) - mi).abs(),
                "nats",
            );
            plot(
                &mut out,
                "Information decomposition of actual selected edges",
                "entropy component",
                "nats",
                vec![line(
                    "Source H / target H / joint H / mutual information",
                    vec![[0., hx], [1., hy], [2., hxy], [3., mi]],
                )],
            );
            out.details["region_joint_distribution"] = json!(joint);
            out.details["interpretation"] = json!(
                "Classical empirical mutual information of source and target region labels, weighted by actual selected-edge counts. It is not a quantum entropy or a Bekenstein bound."
            );
        }
    }
    Ok(out)
}

fn histogram(values: &[f64], bins: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut counts = vec![0.; bins];
    for &v in values {
        let k = bin(v, bins, lo, hi);
        counts[k] += 1.;
    }
    if !values.is_empty() {
        for x in &mut counts {
            *x /= values.len() as f64;
        }
    }
    counts
}
fn bin(value: f64, bins: usize, lo: f64, hi: f64) -> usize {
    if hi <= lo {
        return 0;
    }
    (((value - lo) / (hi - lo) * bins as f64).floor() as usize).min(bins - 1)
}
fn information(r: &ExperimentRequest, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    let (position, velocity, dt) = names(a);
    let bins = n(r, "bins", 6, 2, 32);
    let mut frames = Vec::new();
    let mut energies = Vec::new();
    let mut eligible_counts = Vec::new();
    for s in &a.steps {
        let x = rows(
            &s.final_population,
            position,
            a.gas_config.include_truncated,
        )?;
        let v = rows(
            &s.final_population,
            velocity,
            a.gas_config.include_truncated,
        )?;
        eligible_counts.push(x.len());
        frames.push(x.iter().map(|row| row[0]).collect::<Vec<_>>());
        energies.push(v.iter().map(|row| dot(row, row) / 2.).collect::<Vec<_>>());
    }
    let all: Vec<f64> = frames.iter().flatten().copied().collect();
    if all.is_empty() {
        return Err(GasError::Capability(
            "no eligible recorded field observations".into(),
        ));
    }
    let cut = (frames.len() * 3 / 5).clamp(1, frames.len());
    let training: Vec<f64> = frames[..cut].iter().flatten().copied().collect();
    if training.is_empty() {
        return Err(GasError::Capability(
            "no eligible observations in training prefix".into(),
        ));
    }
    let lo = training.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = training.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = result(
        r,
        match r.experiment {
            53 => "Empirical transition time asymmetry",
            57 => "Measured energy histogram and Gibbs hypothesis",
            58 => "Empirical source susceptibility",
            61 => "Recorded distribution stationarity",
            _ => "Held-out empirical field closure",
        },
        "Explicit empirical distributions of recorded gas observables",
    );
    out.details = json!({"bins":bins,"training_steps":cut,"bin_range":[lo,hi],"eligible_counts":eligible_counts,"bin_convention":"First-coordinate bins fixed using the first 60% of recorded frames; held-out tails are assigned to the outer bins."});
    match r.experiment {
        53 | 65 => {
            // Whole-frame descriptor avoids treating interacting walker trajectories as replicas.
            let means: Vec<Option<f64>> = frames
                .iter()
                .map(|x| {
                    if x.is_empty() {
                        None
                    } else {
                        Some(x.iter().sum::<f64>() / x.len() as f64)
                    }
                })
                .collect();
            let labels: Vec<Option<usize>> = means
                .iter()
                .map(|x| x.map(|v| bin(v, bins, lo, hi)))
                .collect();
            let alpha = p(r, "pseudocount", 0.5, 0.0001, 10.);
            let mut counts = vec![vec![0.; bins]; bins];
            for pair in labels[..cut].windows(2) {
                if let (Some(i), Some(j)) = (pair[0], pair[1]) {
                    counts[i][j] += 1.;
                }
            }
            let mut kernel = counts.clone();
            for row in &mut kernel {
                let sum = row.iter().sum::<f64>() + alpha * bins as f64;
                for v in row {
                    *v = (*v + alpha) / sum;
                }
            }
            if r.experiment == 53 {
                let total = counts.iter().flatten().sum::<f64>() + alpha * (bins * bins) as f64;
                let mut asymmetry = 0.;
                let mut cells = vec![];
                for (i, row) in counts.iter().enumerate() {
                    for (j, &value) in row.iter().enumerate() {
                        let p = (value + alpha) / total;
                        let q = (counts[j][i] + alpha) / total;
                        let term = p * (p / q).ln();
                        asymmetry += term;
                        cells.push([(i * bins + j) as f64, term]);
                    }
                }
                metric(
                    &mut out,
                    "Regularized empirical forward/reverse KL",
                    asymmetry,
                    "nats",
                );
                plot(
                    &mut out,
                    "Empirical time-asymmetry contributions",
                    "transition cell",
                    "nats",
                    vec![line("p(i,j) log[p(i,j)/p(j,i)]", cells)],
                );
                out.details["interpretation"] = json!(
                    "KL compares the empirical whole-frame transition table with its transpose using explicit symmetric pseudocounts. It is a coarse observable time-asymmetry diagnostic, not the likelihood ratio of the complete algorithm path."
                );
            } else {
                let mut baseline = vec![alpha; bins];
                for k in labels[..cut].iter().flatten() {
                    baseline[*k] += 1.;
                }
                let norm = baseline.iter().sum::<f64>();
                for x in &mut baseline {
                    *x /= norm;
                }
                let mut fitted = vec![];
                let mut constant = vec![];
                let mut two_step = vec![];
                let mut sum = 0.;
                let mut base = 0.;
                let mut count = 0;
                for i in cut..labels.len() {
                    if let (Some(from), Some(target)) = (labels[i - 1], labels[i]) {
                        let score = (0..bins)
                            .map(|j| (kernel[from][j] - f64::from(j == target)).powi(2))
                            .sum::<f64>();
                        let b = (0..bins)
                            .map(|j| (baseline[j] - f64::from(j == target)).powi(2))
                            .sum::<f64>();
                        sum += score;
                        base += b;
                        count += 1;
                        fitted.push([i as f64 * dt, score]);
                        constant.push([i as f64 * dt, b]);
                        if i >= 2
                            && let Some(start) = labels[i - 2]
                        {
                            let p2 = evolve(&kernel[start], &kernel);
                            two_step.push([
                                i as f64 * dt,
                                (0..bins)
                                    .map(|j| (p2[j] - f64::from(j == target)).powi(2))
                                    .sum(),
                            ]);
                        }
                    }
                }
                if count > 0 {
                    metric(
                        &mut out,
                        "Held-out transition Brier score",
                        sum / count as f64,
                        "squared probability",
                    );
                    metric(
                        &mut out,
                        "Held-out constant Brier score",
                        base / count as f64,
                        "squared probability",
                    );
                }
                plot(
                    &mut out,
                    "Held-out descriptor predictions",
                    "physical time",
                    "Brier score",
                    vec![
                        line("One-step empirical transition", fitted),
                        line("Training marginal baseline", constant),
                        line("Two-step iterated transition", two_step),
                    ],
                );
                out.details["held_out_transitions"] = json!(count);
                out.details["interpretation"] = json!(
                    "The descriptor is the whole-frame mean first coordinate. Iteration tests a fitted memory-free reduction; donor memory and unresolved field information are not assumed absent. No held-out transitions means insufficient chronological coverage."
                );
            }
            out.details["transition_counts"] = json!(counts);
            out.details["transition_kernel"] = json!(kernel);
            out.details["pseudocount"] = json!(alpha);
        }
        57 => {
            let training_energy: Vec<f64> = energies[..cut].iter().flatten().copied().collect();
            let energy: Vec<f64> = energies[cut..].iter().flatten().copied().collect();
            if energy.is_empty() {
                return Err(GasError::Capability(
                    "energy hypothesis requires an eligible held-out suffix".into(),
                ));
            }
            let maximum = training_energy.iter().copied().fold(0., f64::max);
            let distribution = histogram(&energy, bins, 0., maximum);
            let mean = training_energy.iter().sum::<f64>() / training_energy.len() as f64;
            // Explicit exponential-bin hypothesis, with its mean inferred from measured energy.
            let mut prediction: Vec<f64> = (0..bins)
                .map(|k| {
                    let l = maximum * k as f64 / bins as f64;
                    let u = maximum * (k + 1) as f64 / bins as f64;
                    if mean > 0. {
                        (-l / mean).exp() - if k + 1 == bins { 0. } else { (-u / mean).exp() }
                    } else {
                        f64::from(k == 0)
                    }
                })
                .collect();
            let norm = prediction.iter().sum::<f64>();
            if norm > 0. {
                for x in &mut prediction {
                    *x /= norm;
                }
            }
            metric(
                &mut out,
                "Energy exponential-hypothesis total variation",
                tv(&distribution, &prediction),
                "probability",
            );
            plot(
                &mut out,
                "Measured energy law versus fitted exponential hypothesis",
                "energy bin",
                "probability",
                vec![
                    line(
                        "Held-out kinetic-energy histogram",
                        distribution
                            .iter()
                            .enumerate()
                            .map(|(i, &x)| [i as f64, x])
                            .collect(),
                    ),
                    line(
                        "Normalized exponential-bin hypothesis",
                        prediction
                            .iter()
                            .enumerate()
                            .map(|(i, &x)| [i as f64, x])
                            .collect(),
                    ),
                ],
            );
            out.details["mean_energy"] = json!(mean);
            out.details["maximum_energy"] = json!(maximum);
            out.details["interpretation"] = json!(
                "The exponential energy law is fitted on the training prefix and evaluated on the chronological suffix; the last bin includes the entire upper tail. It is not a QSD or Gibbs law derived from the gas. Its held-out residual is displayed without an acceptance claim."
            );
        }
        58 => {
            let source = p(r, "source", 0.3, -4., 4.);
            let values: Vec<f64> = frames
                .iter()
                .filter(|x| !x.is_empty())
                .map(|x| x.iter().sum::<f64>() / x.len() as f64)
                .collect();
            let weighted = |theta: f64| {
                let offset = values
                    .iter()
                    .map(|x| theta * x)
                    .fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = values.iter().map(|x| (theta * x - offset).exp()).collect();
                let z = weights.iter().sum::<f64>();
                let m = values
                    .iter()
                    .zip(&weights)
                    .map(|(x, w)| x * w / z)
                    .sum::<f64>();
                let variance = values
                    .iter()
                    .zip(&weights)
                    .map(|(x, w)| (x - m).powi(2) * w / z)
                    .sum::<f64>();
                (m, variance)
            };
            let h = 1e-4;
            let (mean, variance) = weighted(source);
            let derivative = (weighted(source + h).0 - weighted(source - h).0) / (2. * h);
            metric(&mut out, "Empirical tilted mean", mean, "coordinate");
            metric(
                &mut out,
                "Tilted variance susceptibility",
                variance,
                "coordinate squared",
            );
            metric(
                &mut out,
                "Source derivative residual",
                (variance - derivative).abs(),
                "coordinate squared",
            );
            plot(
                &mut out,
                "Source tilt of retained whole-frame observations",
                "source",
                "mean coordinate",
                vec![line(
                    "Empirical tilted mean",
                    (0..41)
                        .map(|i| {
                            let t = source - 1. + i as f64 / 20.;
                            [t, weighted(t).0]
                        })
                        .collect(),
                )],
            );
            out.details["interpretation"] = json!(
                "A finite empirical generating function over recorded frame observables. The exact derivative equals tilted variance. This changes readout weights, not the gas transition law; dynamical response requires independent perturbed continuations."
            );
        }
        _ => {
            let reference = histogram(&training, bins, lo, hi);
            let mut distances = vec![];
            let mut h = vec![];
            let mut successive = vec![];
            let mut previous: Option<Vec<f64>> = None;
            for (i, frame) in frames.iter().enumerate() {
                if frame.is_empty() {
                    continue;
                }
                let hist = histogram(frame, bins, lo, hi);
                distances.push([i as f64 * dt, tv(&hist, &reference)]);
                h.push([i as f64 * dt, entropy(&hist)]);
                if let Some(ref old) = previous {
                    successive.push([i as f64 * dt, tv(&hist, old)]);
                }
                previous = Some(hist);
            }
            plot(
                &mut out,
                "Measured stationarity diagnostics",
                "physical time",
                "total variation",
                vec![
                    line("Distance to training pooled distribution", distances),
                    line("Successive-frame distance", successive),
                ],
            );
            plot(
                &mut out,
                "Recorded spatial histogram entropy",
                "physical time",
                "nats",
                vec![line("Frame histogram entropy", h)],
            );
            out.details["training_reference"] = json!(reference);
            out.details["interpretation"] = json!(
                "Finite-frame distribution drift is observed directly. A stationary or quasi-stationary gas law is not supplied; survival and eligibility counts remain explicit."
            );
        }
    }
    Ok(out)
}
