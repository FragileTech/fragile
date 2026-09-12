use super::*;
use crate::partv_geometry::MetricPolicy;
use crate::physics::{
    fitness::{ConditionalFitnessCache, Objective, companion_distance},
    geometry::{FitnessJet, MetricJet, fitness_curvature, metric_curvature},
    jet::JetSpace,
};

pub(super) fn run(
    r: &ExperimentRequest,
    archive: Option<&RunArchive<f64>>,
) -> Result<ExperimentResult> {
    match r.experiment {
        37 => transport(r),
        38 => holonomy(r),
        39 => curvature(r, archive),
        40 => connection(r),
        41 => raychaudhuri(r),
        42 => volumes(r),
        43 => focusing(r),
        _ => topology(r, archive),
    }
}
fn transport(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let h = p(r, "expansion", 0.4, -1., 1.);
    let length = p(r, "length", 1., 0.05, 3.);
    let steps = n(r, "resolution", 32, 4, 512);
    let mut out = result(
        r,
        "Transport in a changing geometry",
        "Intrinsic slice and ambient expanding-slab Levi-Civita transport, c=1",
    );
    let mut y = vec![0., 1.];
    let mut numerical = vec![[0., 0.]];
    let mut exact = vec![[0., 0.]];
    for i in 0..steps {
        y = rk4(&y, 0., length / steps as f64, |_, v| {
            vec![-h * v[1], -h * v[0]]
        });
        let x = (i + 1) as f64 * length / steps as f64;
        numerical.push([x, y[0]]);
        exact.push([x, -(h * x).sinh()]);
    }
    metric(&mut out, "Ambient time component", y[0], "");
    metric(
        &mut out,
        "Transport error",
        (y[0] + (h * length).sinh()).abs(),
        "",
    );
    metric(
        &mut out,
        "Metric norm residual",
        (-y[0] * y[0] + y[1] * y[1] - 1.).abs(),
        "",
    );
    plot(
        &mut out,
        "The time component generated along a spatial path",
        "path length",
        "V time",
        vec![
            line("RK4 ambient transport", numerical),
            line("Exact -sinh(Hx)", exact),
            line("Intrinsic transport", vec![[0., 0.], [length, 0.]]),
        ],
    );
    out.details = json!({"metric_signature":[-1,1],"initial_vector":[0,1],"connection_along_path":[[0,h],[h,0]],"final_vector":y,"spatial_slice_time":0});
    Ok(out)
}
fn holonomy(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let theta = p(r, "latitude", 0.8, 0.3, 1.2);
    let steps = n(r, "resolution", 64, 8, 512);
    let mut out = result(
        r,
        "Curvature per plaquette area",
        "Unit sphere intrinsic Levi-Civita connection; ordered coordinate rectangles",
    );
    let mut recovered = vec![];
    let mut errors = vec![];
    let mut raw = vec![];
    for j in 0..8 {
        let width = 0.4 * 0.65_f64.powi(j);
        let mut y = vec![1., 0.];
        let corners = [
            [theta, 0.],
            [theta + width, 0.],
            [theta + width, width],
            [theta, width],
            [theta, 0.],
        ];
        for edge in corners.windows(2) {
            let d = [edge[1][0] - edge[0][0], edge[1][1] - edge[0][1]];
            for i in 0..steps {
                let t = i as f64 / steps as f64;
                y = rk4(&y, t, 1. / steps as f64, |s, v| {
                    let q = edge[0][0] + d[0] * s;
                    vec![
                        q.sin() * q.cos() * v[1] * d[1],
                        -q.cos() / q.sin() * (v[0] * d[1] + v[1] * d[0]),
                    ]
                });
            }
        }
        let area = width * (theta.cos() - (theta + width).cos());
        let angle = (theta.sin() * y[1]).atan2(y[0]);
        let k = angle.abs() / area;
        recovered.push([area, k]);
        errors.push([area, (k - 1.).abs()]);
        raw.push(json!({"width":width,"area":area,"angle":angle,"vector":y}));
    }
    metric(
        &mut out,
        "Finest curvature estimate",
        recovered.last().unwrap()[1],
        "inverse area",
    );
    plot(
        &mut out,
        "Sphere curvature from loop transport",
        "oriented area magnitude",
        "curvature",
        vec![
            line("Integrated holonomy angle / area", recovered),
            line("Sphere K=1", vec![[0., 1.], [0.2, 1.]]),
        ],
    );
    plot(
        &mut out,
        "Transport integration error",
        "area",
        "absolute curvature error",
        vec![line("Numerical error", errors)],
    );
    out.details = json!({"loops":raw,"connection":"sphere chart theta, phi","shape_control":"equal coordinate side lengths; theta bounded away from poles","orientation":"theta+, phi+, theta-, phi-"});
    Ok(out)
}
fn fixture_jet(mode: &str, x: &[f64]) -> Result<FitnessJet<f64>> {
    let space = JetSpace::new(3, 4)?;
    let z = x
        .iter()
        .enumerate()
        .map(|(i, &v)| space.variable(v, i))
        .collect::<Result<Vec<_>>>()?;
    let field = if mode == "conditional" {
        let positions: [[f64; 3]; 4] = [
            [0.2, -0.1, 0.3],
            [1., 0.5, -0.2],
            [-0.8, 0.3, 0.5],
            [0.4, -1., 0.6],
        ];
        let rewards = positions
            .iter()
            .map(|x| Objective::Sphere.value(x))
            .collect::<Result<Vec<_>>>()?;
        let sources = [positions[1], positions[2], positions[3], positions[0]];
        let distances = positions
            .iter()
            .zip(sources)
            .map(|(a, b)| {
                (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() + 0.04).sqrt()
            })
            .collect::<Vec<_>>();
        let cache = ConditionalFitnessCache::new(
            &rewards,
            &distances,
            &[true; 4],
            &crate::fitness::FitnessPipeline {
                distance_floor: 0.2,
                ..Default::default()
            },
        )?;
        cache.evaluate(
            0,
            &Objective::Sphere.evaluate(&z)?,
            &companion_distance(&z, &sources[0], 0., 0.2)?,
        )?
    } else {
        let mut f = space.constant(0.);
        for (i, v) in z.iter().enumerate() {
            let sign = if mode == "mixed" && i == 0 { -1. } else { 1. };
            f = f.add(&v.pow(2.).scale(sign)).add(&v.pow(4.).scale(0.1));
        }
        if mode == "coupled" {
            f = f.add(&z[0].mul(&z[1]).mul(&z[2]).scale(0.4));
        }
        f
    };
    FitnessJet::from_jet(&field)
}
fn curvature(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    let mode = r.text("mode", "conditional");
    let mut threshold = 1e-8;
    let mut epsilon = p(r, "epsilon", 3., 0.1, 10.);
    let mut x = [p(r, "query", 0.2, -0.7, 0.7), -0.1, 0.3];
    let mut policy = if mode == "mixed" {
        MetricPolicy::Clipped
    } else {
        MetricPolicy::Strict
    };
    let target = r.usize("walker", 0);
    let record_index = archive
        .map(|a| r.usize("record", a.steps.len().saturating_sub(1)))
        .unwrap_or(0);
    if let Some(a) = archive {
        let record = a
            .steps
            .get(record_index)
            .ok_or_else(|| GasError::Capability("no completed archived step".into()))?;
        let point = record.before.observations.field("positions")?.row(target)?;
        if point.len() != 3 {
            return Err(GasError::Capability(
                "archived curvature workbench requires actual three-dimensional positions".into(),
            ));
        }
        x = [r.number("query", point[0]), point[1], point[2]];
        let context = archive_fitness_jet(a, record_index, target, &x, 4)?.1;
        epsilon = r.number("epsilon", context["epsilon"].as_f64().unwrap());
        threshold = context["clipping_threshold"].as_f64().unwrap();
        policy = if context["policy"] == "strict" {
            MetricPolicy::Strict
        } else {
            MetricPolicy::Clipped
        };
    }
    let make_jet = |z: &[f64]| -> Result<FitnessJet<f64>> {
        if let Some(a) = archive {
            if a.steps
                .get(record_index)
                .ok_or_else(|| GasError::Capability("no completed archived step".into()))?
                .before
                .observations
                .field("positions")?
                .width()
                != 3
            {
                return Err(GasError::Capability(
                    "archived curvature workbench requires actual three-dimensional positions"
                        .into(),
                ));
            }
            FitnessJet::from_jet(&archive_fitness_jet(a, record_index, target, z, 4)?.0)
        } else {
            fixture_jet(mode, z)
        }
    };
    let j = make_jet(&x)?;
    let raw = crate::physics::geometry::symmetric_eigen(&j.dense_hessian()?, 3)?.0;
    let unavailable = |reason: String| {
        let mut out = result(
            r,
            "Three-dimensional fitness curvature",
            "Declared conditional field; curvature requires the displayed metric domain",
        );
        metric(
            &mut out,
            "Minimum shifted Hessian eigenvalue",
            raw.iter().copied().fold(f64::INFINITY, f64::min) + epsilon,
            "",
        );
        metric(
            &mut out,
            "Packed scalar curvature",
            f64::NAN,
            "inverse fitness",
        );
        plot(
            &mut out,
            "Raw fitness Hessian spectrum",
            "eigenvalue index",
            "eigenvalue",
            vec![
                line(
                    "Unregularized Hessian",
                    raw.iter()
                        .enumerate()
                        .map(|(i, &x)| [i as f64, x])
                        .collect(),
                ),
                line(
                    "Shifted Hessian",
                    raw.iter()
                        .enumerate()
                        .map(|(i, &x)| [i as f64, x + epsilon])
                        .collect(),
                ),
            ],
        );
        out.notes.push(reason.clone());
        out.details = json!({"status":"curvature_unavailable","reason":reason,"raw_eigenvalues":raw,"fitness_jet":j,"epsilon":epsilon,"policy":policy,"clipping_threshold":threshold,"query":x});
        out
    };
    let c = match fitness_curvature(&j, epsilon, policy, threshold, true) {
        Ok(c) => c,
        Err(e) => return Ok(unavailable(e.to_string())),
    };
    let initial_step = p(r, "difference_step", 0.002, 0.0002, 0.02);
    let center = c.spectrum.metric.clone();
    let positive_count = raw.iter().filter(|&&v| v > 0.).count();
    let evaluate = |z: &[f64]| -> Result<Vec<f64>> {
        let jet = make_jet(z)?;
        if policy == MetricPolicy::Clipped {
            let eigenvalues =
                crate::physics::geometry::symmetric_eigen(&jet.dense_hessian()?, 3)?.0;
            if eigenvalues.iter().filter(|&&v| v > 0.).count() != positive_count {
                return Err(GasError::Capability(
                    "finite-difference stencil crosses a spectral clipping surface".into(),
                ));
            }
        }
        Ok(fitness_curvature(&jet, epsilon, policy, threshold, false)?
            .spectrum
            .metric)
    };
    let estimate = |step: f64| -> Result<crate::physics::geometry::CurvatureBatch<f64>> {
        let mut first = vec![0.; 27];
        let mut second = vec![0.; 81];
        for a in 0..3 {
            let mut xp = x;
            let mut xm = x;
            xp[a] += step;
            xm[a] -= step;
            let plus = evaluate(&xp)?;
            let minus = evaluate(&xm)?;
            for k in 0..9 {
                first[a * 9 + k] = (plus[k] - minus[k]) / (2. * step);
                second[(a * 3 + a) * 9 + k] = (plus[k] - 2. * center[k] + minus[k]) / step.powi(2);
            }
            for b in 0..a {
                let mut pp = x;
                let mut pm = x;
                let mut mp = x;
                let mut mm = x;
                pp[a] += step;
                pp[b] += step;
                pm[a] += step;
                pm[b] -= step;
                mp[a] -= step;
                mp[b] += step;
                mm[a] -= step;
                mm[b] -= step;
                let pp = evaluate(&pp)?;
                let pm = evaluate(&pm)?;
                let mp = evaluate(&mp)?;
                let mm = evaluate(&mm)?;
                for k in 0..9 {
                    let v = (pp[k] - pm[k] - mp[k] + mm[k]) / (4. * step * step);
                    second[(a * 3 + b) * 9 + k] = v;
                    second[(b * 3 + a) * 9 + k] = v;
                }
            }
        }
        metric_curvature(
            &MetricJet {
                dimension: 3,
                metric: center.clone(),
                first,
                second,
            },
            true,
        )
    };
    // Select resolution using only successive numerical scalar/Ricci estimates.
    // The packed curvature is never used in the stopping criterion.
    let relative_tolerance = 1e-5;
    let absolute_tolerance = 1e-6;
    let mut sequence = Vec::new();
    let mut previous: Option<crate::physics::geometry::CurvatureBatch<f64>> = None;
    let mut selected_step = initial_step;
    let mut consecutive_stable = 0;
    let mut converged = false;
    let mut refinement_curve = Vec::new();
    for level in 0..=12 {
        let step = initial_step * 0.5_f64.powi(level);
        let estimate = match estimate(step) {
            Ok(value) => value,
            Err(error) => {
                sequence.push(
                    json!({"step":step,"status":"unsupported_stencil","reason":error.to_string()}),
                );
                previous = None;
                consecutive_stable = 0;
                continue;
            }
        };
        let mut scaled_change = None;
        let mut scalar_change = None;
        let mut ricci_change = None;
        if let Some(coarse) = &previous {
            let ds = (estimate.scalar - coarse.scalar).abs();
            let dr = estimate
                .ricci
                .iter()
                .zip(&coarse.ricci)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            let ricci_scale = estimate
                .ricci
                .iter()
                .chain(&coarse.ricci)
                .map(|v| v.abs())
                .fold(0., f64::max);
            let scalar_scale = estimate.scalar.abs().max(coarse.scalar.abs());
            let change = (ds / (absolute_tolerance + relative_tolerance * scalar_scale))
                .max(dr / (absolute_tolerance + relative_tolerance * ricci_scale));
            scaled_change = Some(change);
            scalar_change = Some(ds);
            ricci_change = Some(dr);
            consecutive_stable = if change <= 1. {
                consecutive_stable + 1
            } else {
                0
            };
        }
        refinement_curve.push([step, estimate.scalar]);
        sequence.push(json!({"step":step,"status":"available","scalar":estimate.scalar,"successive_scalar_change":scalar_change,"successive_ricci_max_change":ricci_change,"scaled_change":scaled_change}));
        selected_step = step;
        previous = Some(estimate);
        if consecutive_stable >= 2 {
            converged = true;
            break;
        }
    }
    let Some(reference) = previous else {
        let mut out = unavailable(
            "No finite-difference stencil remained inside the curvature domain during refinement"
                .into(),
        );
        out.details["finite_difference_refinement"] =
            json!({"status":"unsupported","sequence":sequence});
        return Ok(out);
    };
    let mut out = result(
        r,
        "Three-dimensional fitness curvature",
        "Packed fourth-order conditional fitness jet and independent finite-difference metric curvature",
    );
    metric(
        &mut out,
        "Packed scalar curvature",
        c.scalar,
        "inverse fitness",
    );
    metric(
        &mut out,
        "Finite-difference scalar",
        reference.scalar,
        "inverse fitness",
    );
    metric(
        &mut out,
        "Independent residual",
        (reference.scalar - c.scalar).abs(),
        "",
    );
    plot(
        &mut out,
        "Coordinate Ricci components",
        "flattened matrix index",
        "Ricci",
        vec![
            line(
                "Analytic jet",
                c.ricci
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
            line(
                "Metric finite differences",
                reference
                    .ricci
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
        ],
    );
    plot(
        &mut out,
        "Finite-difference refinement",
        "difference step",
        "scalar curvature",
        vec![line("Numerical metric curvature", refinement_curve)],
    );
    metric(
        &mut out,
        "Selected difference step",
        selected_step,
        "position",
    );
    out.details = json!({"epsilon":epsilon,"policy":policy,"clipping_threshold":threshold,"fitness_jet":j,"curvature":c,"numerical_metric_curvature":reference,"query":x,"mode":mode,"derivative_convention":"target query varies; fixed donor source; global moments differentiated","third_packed_count":10,"hessian_packed_count":6});
    out.details["finite_difference_refinement"] = json!({"status":if converged {"converged"} else {"underresolved"},"selected_step":selected_step,"relative_tolerance":relative_tolerance,"absolute_tolerance":absolute_tolerance,"maximum_halvings":12,"required_consecutive_stable":2,"criterion":"Successive numerical scalar and Ricci infinity-norm changes; independent of the packed analytic curvature","sequence":sequence});
    if !converged {
        out.notes.push("Finite-difference curvature remains underresolved within the probe budget; the displayed analytic comparison is inconclusive.".into());
    }
    if let Some(a) = archive {
        out.details["archive_field"] = archive_fitness_jet(a, record_index, target, &x, 4)?.1;
        out.model="Actual recorded three-dimensional pre-clone conditional field; independent metric finite differences".into();
    }
    Ok(out)
}
fn connection(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let single = r.text("mode", "identified") == "single_velocity";
    let mut curves = vec![];
    let truth = [0.4, -0.2, 0.7];
    let mut last = Value::Null;
    let mut spectrum = Vec::new();
    for j in 0..7 {
        let h = 0.25 * 0.5_f64.powi(j);
        let dirs = if single {
            vec![[1., 0.]; 6]
        } else {
            vec![
                [1., 0.],
                [0., 1.],
                [1., 1.],
                [1., -1.],
                [0.5, 1.],
                [-1., 0.5],
            ]
        };
        let mut a = vec![vec![0.; 3]; 3];
        let mut b = vec![0.; 3];
        for (i, v) in dirs.iter().enumerate() {
            let w = [h * (i as f64 + 1.) / 6., h * (1. - i as f64 / 9.)];
            let row = [
                h * w[0] * v[0],
                h * (w[0] * v[1] + w[1] * v[0]),
                h * w[1] * v[1],
            ];
            let g = [
                truth[0] * v[0] + truth[1] * v[1],
                truth[1] * v[0] + truth[2] * v[1],
            ];
            let mut y = w.to_vec();
            for _ in 0..16 {
                y = rk4(&y, 0., h / 16., |_, z| vec![-g[0] * z[0] - g[1] * z[1], 0.]);
            }
            let measurement = w[0] - y[0];
            for k in 0..3 {
                b[k] += row[k] * measurement;
                for l in 0..3 {
                    a[k][l] += row[k] * row[l];
                }
            }
        }
        let scale = h.powi(4);
        for row in &mut a {
            for x in row {
                *x /= scale;
            }
        }
        for x in &mut b {
            *x /= scale;
        }
        spectrum = crate::physics::geometry::symmetric_eigen(
            &a.iter().flatten().copied().collect::<Vec<_>>(),
            3,
        )?
        .0;
        match solve(a.clone(), b) {
            Ok(est) => {
                let err = est
                    .iter()
                    .zip(truth)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                curves.push([h, err]);
                last = json!({"coefficients":est,"design_gram_scaled":a});
            }
            Err(_) => {
                last = json!({"status":"rank_deficient","design_gram_scaled":a});
            }
        }
    }
    let mut out = result(
        r,
        "Identify connection coefficients",
        "Independent integrated transport measurements and symmetric least squares",
    );
    if let Some(x) = curves.last() {
        metric(&mut out, "Finest coefficient error", x[1], "");
    }
    out.notes.push("A single fixed velocity identifies only a contraction of the connection; rank deficiency is reported rather than regularized away.".into());
    if !curves.is_empty() {
        plot(
            &mut out,
            "Identified connection convergence",
            "edge and time scale h",
            "coefficient error",
            vec![line("Transport-based least squares", curves)],
        );
    }
    plot(
        &mut out,
        "Connection design identifiability",
        "eigenvalue index",
        "eigenvalue of Gram / h^4",
        vec![line(
            "Measured scaled design Gram spectrum",
            spectrum
                .iter()
                .enumerate()
                .map(|(i, &v)| [i as f64, v])
                .collect(),
        )],
    );
    last["gram_eigenvalues"] = json!(spectrum);
    out.details = json!({"truth":truth,"mode":if single{"single_velocity"}else{"identified"},"last_design":last});
    Ok(out)
}
fn raychaudhuri(r: &ExperimentRequest) -> Result<ExperimentResult> {
    if r.text("mode", "bianchi") == "rotation" {
        return rotating_congruence(r);
    }
    let d = n(r, "dimension", 3, 2, 3);
    let h = p(r, "expansion", 0.4, -1., 1.);
    let anis = p(r, "anisotropy", 0.3, 0., 0.8);
    let rates = (0..d)
        .map(|i| h + anis * (i as f64 - (d - 1) as f64 / 2.))
        .collect::<Vec<_>>();
    let theta = rates.iter().sum::<f64>();
    let shear = rates
        .iter()
        .map(|v| (v - theta / d as f64).powi(2))
        .sum::<f64>();
    let ric = -rates.iter().map(|v| v * v).sum::<f64>();
    let dt = 1e-4;
    let logvol = |t: f64| rates.iter().map(|h| (h * t).exp()).product::<f64>().ln();
    let measured = derivative(logvol, 0.3, dt);
    let mut out = result(
        r,
        "All terms in Raychaudhuri",
        "Anisotropic exponential Bianchi I slab with comoving unit geodesics",
    );
    metric(&mut out, "Measured expansion", measured, "inverse time");
    metric(
        &mut out,
        "Raychaudhuri residual",
        (-theta * theta / d as f64 - shear - ric).abs(),
        "inverse time squared",
    );
    plot(
        &mut out,
        "Signed terms in the expansion equation",
        "term index",
        "inverse time squared",
        vec![line(
            "Expansion, shear, vorticity, Ricci",
            vec![
                [0., -theta * theta / d as f64],
                [1., -shear],
                [2., 0.],
                [3., -ric],
            ],
        )],
    );
    out.details = json!({"dimension":d,"scale_rates":rates,"theta":theta,"shear_squared":shear,"vorticity_squared":0,"ricci_uu":ric,"theta_derivative":0,"acceleration":0});
    Ok(out)
}
/// Rigid rotation is Born-rigid, with nonzero twist and centripetal acceleration.
/// All contractions use the Minkowski metric (-,+,+,+), with c=1.
fn rotating_congruence(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let d = n(r, "dimension", 3, 2, 3);
    let size = d + 1;
    let speed = p(r, "rotation_speed", 0.4, 0., 0.8);
    let radius = p(r, "radius", 0.6, 0.2, 1.);
    let omega = speed / radius;
    let step = 1e-5 * radius;
    let velocity = |q: &[f64]| {
        let gamma = (1. - omega * omega * (q[1] * q[1] + q[2] * q[2]))
            .sqrt()
            .recip();
        let mut u = vec![0.; size];
        u[0] = gamma;
        u[1] = -gamma * omega * q[2];
        u[2] = gamma * omega * q[1];
        u
    };
    let mut q = vec![0.; size];
    q[1] = radius;
    let u = velocity(&q);
    let sign = |a: usize| if a == 0 { -1. } else { 1. };
    let mut grad = vec![vec![0.; size]; size];
    for b in 1..size {
        let mut plus = q.clone();
        let mut minus = q.clone();
        plus[b] += step;
        minus[b] -= step;
        for (a, (up, um)) in velocity(&plus)
            .into_iter()
            .zip(velocity(&minus))
            .enumerate()
        {
            grad[a][b] = (up - um) / (2. * step);
        }
    }
    let theta = (0..size).map(|a| grad[a][a]).sum::<f64>();
    let projector = |a: usize, c: usize| f64::from(a == c) + sign(a) * u[a] * u[c];
    let mut b = vec![vec![0.; size]; size];
    for (a, row) in b.iter_mut().enumerate() {
        for (e, value) in row.iter_mut().enumerate() {
            for (c, derivative_row) in grad.iter().enumerate() {
                for (f, derivative) in derivative_row.iter().enumerate() {
                    *value += projector(a, c) * projector(e, f) * sign(c) * derivative;
                }
            }
        }
    }
    let mut shear = 0.;
    let mut twist = 0.;
    for a in 0..size {
        for e in 0..size {
            let h = if a == e { sign(a) } else { 0. } + sign(a) * sign(e) * u[a] * u[e];
            let sigma = (b[a][e] + b[e][a]) / 2. - theta / d as f64 * h;
            let w = (b[a][e] - b[e][a]) / 2.;
            shear += sign(a) * sign(e) * sigma * sigma;
            twist += sign(a) * sign(e) * w * w;
        }
    }
    let acceleration = (0..size)
        .map(|a| (0..size).map(|e| u[e] * grad[a][e]).sum::<f64>())
        .collect::<Vec<_>>();
    // Differentiate the exact centripetal field independently of the velocity-gradient contraction.
    let mut divergence = 0.;
    for a in 1..=2 {
        let acc = |z: f64| {
            let mut point = q.clone();
            point[a] = z;
            -omega * omega * point[a] / (1. - omega * omega * (point[1].powi(2) + point[2].powi(2)))
        };
        divergence += derivative(acc, q[a], step);
    }
    let predicted_twist = 2. * omega * omega / (1. - speed * speed).powi(2);
    let terms = [-theta * theta / d as f64, -shear, twist, 0., divergence];
    let mut out = result(
        r,
        "All terms in Raychaudhuri",
        "Stationary rigid rotation in Minkowski spacetime inside the light cylinder, c=1",
    );
    metric(&mut out, "Measured expansion", theta, "inverse time");
    metric(
        &mut out,
        "Raychaudhuri residual",
        terms.iter().sum::<f64>().abs(),
        "inverse time squared",
    );
    metric(
        &mut out,
        "Vorticity contraction residual",
        (twist - predicted_twist).abs(),
        "inverse time squared",
    );
    plot(
        &mut out,
        "Twist is balanced by acceleration divergence",
        "expansion, shear, twist, Ricci, acceleration",
        "inverse time squared",
        vec![
            line(
                "Finite-difference contractions",
                terms
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect(),
            ),
            line(
                "Exact rigid rotation",
                [0., 0., predicted_twist, 0., -predicted_twist]
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| [i as f64, v])
                    .collect(),
            ),
        ],
    );
    out.details = json!({"mode":"rotation","dimension":d,"rotation_speed":speed,"radius":radius,"angular_velocity":omega,"theta":theta,"theta_derivative":0,"shear_squared":shear,"vorticity_squared":twist,"predicted_vorticity_squared":predicted_twist,"ricci_uu":0,"acceleration":acceleration,"acceleration_divergence":divergence,"unit_velocity":u,"velocity_gradient":grad,"difference_step":step,"domain":"Omega times radius < 1; stationary Born-rigid congruence, no geodesic assumption"});
    Ok(out)
}
fn volumes(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let rate = p(r, "rate", 0.4, 0.05, 0.8);
    let time = p(r, "time", 0.5, 0., 0.8);
    let center = 0.4;
    let flow = |x: f64, t: f64| x / (1. - rate * t * x);
    let mut curves = vec![];
    let mut details = vec![];
    for j in 0..8 {
        let e = 0.2 * 0.6_f64.powi(j);
        let volume = |t: f64| flow(center + e / 2., t) - flow(center - e / 2., t);
        let theta = derivative(|t| volume(t).ln(), time, 1e-4);
        let exact = 2. * rate * flow(center, time);
        let right = flow(center + e, time);
        let left = flow(center - e, time);
        let vor = (right - left) / 2.;
        let vor_flux = (rate * right * right - rate * left * left) / 2.;
        let b = vor_flux / vor - theta;
        curves.push([e, (theta - exact).abs()]);
        details.push(json!({"diameter":volume(time),"material_expansion":theta,"point_expansion":exact,"voronoi_expansion":vor_flux/vor,"normalized_flux_difference":b}));
    }
    let mut out = result(
        r,
        "Material cells and reconstructed volumes",
        "One-dimensional flow dx/dt=a x² embedded in a flat transverse product",
    );
    plot(
        &mut out,
        "Material expansion approaches point expansion",
        "initial cell width",
        "absolute expansion error",
        vec![line("Measured log-volume derivative", curves)],
    );
    out.details = json!({"cells":details,"time":time,"rate":rate,"domain":"pre-caustic positive material volumes"});
    out.notes.push("The normalized reconstructed-cell flux is retained; the geometric Raychaudhuri interpretation requires a separately specified metric and congruence.".into());
    Ok(out)
}
fn focusing(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let d = n(r, "dimension", 3, 2, 3) as f64;
    let initial = p(r, "initial_expansion", -1., -3., -0.1);
    let ric = p(r, "ricci", 0.2, 0., 2.);
    let shear = p(r, "shear_squared", 0., 0., 2.);
    let limit = d / initial.abs();
    let forcing = ric + shear;
    let exact_focus = if forcing > 0. {
        (std::f64::consts::FRAC_PI_2 + (initial / (d * forcing).sqrt()).atan())
            * (d / forcing).sqrt()
    } else {
        limit
    };
    let end = 0.85 * exact_focus;
    let mut y = vec![initial, 1.];
    let mut theta = vec![];
    let mut bound = vec![];
    let mut vol = vec![];
    for i in 0..400 {
        let t = i as f64 * end / 400.;
        theta.push([t, y[0]]);
        bound.push([t, initial / (1. + initial * t / d)]);
        vol.push([t, y[1]]);
        y = rk4(&y, t, end / 400., |_, z| {
            vec![-z[0] * z[0] / d - ric - shear, z[0] * z[1]]
        });
        if y[1] <= 0. || !y[0].is_finite() || y[0] < -1e5 {
            break;
        }
    }
    let mut out = result(
        r,
        "Focusing of a contracting congruence",
        "Raychaudhuri comparison ODE with prescribed constant shear norm and Ricci contraction, zero vorticity and acceleration",
    );
    metric(&mut out, "Comparison focusing time", limit, "proper length");
    metric(
        &mut out,
        "Constant-forcing focusing time",
        exact_focus,
        "proper length",
    );
    plot(
        &mut out,
        "Contracting expansion",
        "proper length",
        "theta",
        vec![
            line("RK4 with Ricci and shear focusing", theta),
            line("Zero-Ricci zero-shear upper bound", bound),
        ],
    );
    plot(
        &mut out,
        "Transverse material Jacobian",
        "proper length",
        "J / J0",
        vec![line("Integrated Jacobian", vol)],
    );
    out.details = json!({"theta0":initial,"ricci_uu":ric,"shear_squared":shear,"constant_forcing_focusing_time":exact_focus,"dimension":d,"zero_vorticity":true});
    Ok(out)
}
fn topology(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    if r.text("surface", "disk") == "sphere" {
        return sphere_topology(r);
    }
    let sides = n(r, "vertices", 8, 3, 64);
    let scale = p(r, "scale", 1., 0.1, 4.);
    let mut angles = vec![];
    let mut center_sum = 0.;
    let mut boundary_sum = 0.;
    for i in 0..sides {
        let a = std::f64::consts::TAU * i as f64 / sides as f64;
        let b = std::f64::consts::TAU * (i + 1) as f64 / sides as f64;
        let u = [a.cos(), a.sin()];
        let v = [b.cos(), b.sin()];
        let center = (u[0] * v[0] + u[1] * v[1]).clamp(-1., 1.).acos();
        center_sum += center;
        let boundary_angle = (std::f64::consts::PI - center) / 2.;
        boundary_sum += std::f64::consts::PI - 2. * boundary_angle;
        angles.push([i as f64, std::f64::consts::PI - 2. * boundary_angle]);
    }
    let deficit = std::f64::consts::TAU - center_sum;
    let tetra_dihedral = (1. / 3_f64).acos();
    let hinge = 2. * (std::f64::consts::TAU - 3. * tetra_dihedral) * scale;
    let mut out = result(
        r,
        "Curvature and topology bookkeeping",
        "Triangulated Euclidean disk and three-tetrahedra interior hinge star",
    );
    metric(&mut out, "Euler characteristic", 1., "");
    metric(
        &mut out,
        "Gauss–Bonnet residual",
        (deficit + boundary_sum - std::f64::consts::TAU).abs(),
        "radians",
    );
    metric(&mut out, "Interior planar deficit", deficit, "radians");
    metric(&mut out, "3D integrated hinge curvature", hinge, "length");
    plot(
        &mut out,
        "Boundary turning supplies the disk curvature",
        "boundary vertex",
        "turning angle",
        vec![line("Boundary angles", angles)],
    );
    out.details = json!({"counts":{"vertices":sides+1,"edges":2*sides,"faces":sides},"interior_deficit":deficit,"boundary_turning_sum":boundary_sum,"hinge_star":{"incident_regular_tetrahedra":3,"edge_length":scale,"dihedral_angle":tetra_dihedral,"integrated_scalar_curvature":hinge},"archive_steps_available":archive.map(|a|a.steps.len())});
    Ok(out)
}

fn sphere_topology(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let scale = p(r, "scale", 1., 0.1, 4.);
    let refinements = n(r, "refinement", 1, 0, 3);
    let mut vertices: Vec<[f64; 3]> = vec![
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ];
    let mut faces = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    for _ in 0..refinements {
        let mut midpoint = std::collections::BTreeMap::new();
        let mut refined = Vec::new();
        for [a, b, c] in faces {
            let mut middle = |i: usize, j: usize| {
                let key = (i.min(j), i.max(j));
                *midpoint.entry(key).or_insert_with(|| {
                    let mut v = std::array::from_fn::<_, 3, _>(|k| vertices[i][k] + vertices[j][k]);
                    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    for x in &mut v {
                        *x /= norm;
                    }
                    vertices.push(v);
                    vertices.len() - 1
                })
            };
            let ab = middle(a, b);
            let bc = middle(b, c);
            let ca = middle(c, a);
            refined.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
        }
        faces = refined;
    }
    for v in &mut vertices {
        for x in v {
            *x *= scale;
        }
    }
    let mut deficits = vec![std::f64::consts::TAU; vertices.len()];
    let mut incidences = std::collections::BTreeMap::new();
    for face in &faces {
        for k in 0..3 {
            let a = face[k];
            let b = face[(k + 1) % 3];
            let c = face[(k + 2) % 3];
            let u = std::array::from_fn::<_, 3, _>(|j| vertices[b][j] - vertices[a][j]);
            let v = std::array::from_fn::<_, 3, _>(|j| vertices[c][j] - vertices[a][j]);
            let dot = u.iter().zip(v).map(|(x, y)| x * y).sum::<f64>();
            let norm = (u.iter().map(|x| x * x).sum::<f64>()
                * v.iter().map(|x| x * x).sum::<f64>())
            .sqrt();
            deficits[a] -= (dot / norm).clamp(-1., 1.).acos();
            *incidences.entry((a.min(b), a.max(b))).or_insert(0usize) += 1;
        }
    }
    let total = deficits.iter().sum::<f64>();
    let chi = vertices.len() as f64 - incidences.len() as f64 + faces.len() as f64;
    let mut out = result(
        r,
        "Curvature and topology bookkeeping",
        "Closed triangulated sphere from projected octahedron refinements; intrinsic Euclidean triangle angles",
    );
    metric(&mut out, "Euler characteristic", chi, "");
    metric(
        &mut out,
        "Gauss–Bonnet residual",
        (total - std::f64::consts::TAU * chi).abs(),
        "radians",
    );
    metric(&mut out, "Total intrinsic deficit", total, "radians");
    metric(
        &mut out,
        "Boundary edge count",
        incidences.values().filter(|&&n| n == 1).count() as f64,
        "",
    );
    plot(
        &mut out,
        "Closed-surface vertex curvature",
        "vertex",
        "angle deficit",
        vec![line(
            "Measured triangle-angle deficits",
            deficits
                .iter()
                .enumerate()
                .map(|(i, &v)| [i as f64, v])
                .collect(),
        )],
    );
    out.details = json!({"surface":"sphere","counts":{"vertices":vertices.len(),"edges":incidences.len(),"faces":faces.len()},"refinement":refinements,"scale":scale,"vertices":vertices,"faces":faces,"deficits":deficits,"boundary_turning_sum":0,"total_deficit":total,"all_edges_have_two_incident_faces":incidences.values().all(|&n|n==2),"prediction":4.*std::f64::consts::PI});
    out.notes.push("This closed polyhedral two-sphere has Euler characteristic 2 and total Gaussian angle deficit 4 pi at every refinement and scale. The triangles use chord lengths; individual vertex deficits are integrated curvature, not pointwise smooth-sphere curvature.".into());
    Ok(out)
}

/// Reconstruct an exact recorded pre-clone conditional field at a fixed query.
/// Provider provenance identifies the analytic objective and coordinate shift.
/// Retained source coordinates are resolved by full frame/version/generation.
pub fn archive_fitness_jet(
    archive: &RunArchive<f64>,
    record_index: usize,
    target: usize,
    query: &[f64],
    order: usize,
) -> Result<(crate::physics::jet::Jet<f64>, Value)> {
    archive_fitness_jet_with_history(archive, record_index, target, query, order, &[])
}

/// Reconstruct a continuation field with immutable source frames retained in its checkpoint.
/// Historical coordinates are resolved by frame, population version and source generation.
pub fn archive_fitness_jet_with_history(
    archive: &RunArchive<f64>,
    record_index: usize,
    target: usize,
    query: &[f64],
    order: usize,
    history: &[(u64, crate::Population<f64>)],
) -> Result<(crate::physics::jet::Jet<f64>, Value)> {
    use crate::{
        fitness::Standardizer,
        geometry::Distance,
        physics::fitness::{local_log_weights, pipeline_from_measurement_jets},
    };
    let error = |s: &str| GasError::Capability(s.into());
    let record = archive
        .steps
        .get(record_index)
        .ok_or_else(|| error("recorded conditional field requires a completed step"))?;
    let provider = archive
        .providers
        .get("operators")
        .ok_or_else(|| error("conditional fitness provider provenance unavailable"))?;
    let parts = provider.split('/').collect::<Vec<_>>();
    if parts.len() != 5 || parts[0] != "conditional-fitness-nd" || parts[1] != "v1" {
        return Err(error(
            "archive field requires identified conditional-fitness-nd/v1 provider; no objective is inferred from rewards",
        ));
    }
    let d = record.before.observations.field("positions")?.width();
    if !(2..=3).contains(&d) || query.len() != d || query.iter().any(|v| !v.is_finite()) {
        return Err(error(
            "conditional field query must match two- or three-dimensional recorded coordinates",
        ));
    }
    let objective = match parts[2] {
        "Sphere" => Objective::Sphere,
        "Quadratic" => Objective::Quadratic {
            matrix: (0..d * d).map(|i| f64::from(i / d == i % d)).collect(),
        },
        "Rastrigin" => Objective::Rastrigin,
        "Rosenbrock" => Objective::Rosenbrock,
        "StyblinskiTang" => Objective::StyblinskiTang,
        _ => {
            return Err(error(
                "recorded objective has no identified derivative provider",
            ));
        }
    };
    let shift: Vec<f64> = serde_json::from_str(parts[4])
        .map_err(|_| error("recorded objective shift unavailable"))?;
    let provider_number = |key: &str| -> Result<f64> {
        let value = parts[3]
            .split(&format!("{key}: "))
            .nth(1)
            .and_then(|s| s.split([',', '}']).next())
            .and_then(|s| s.trim().parse::<f64>().ok())
            .ok_or_else(|| error("recorded physics metric configuration unavailable"))?;
        if !value.is_finite() {
            return Err(error("nonfinite recorded metric parameter"));
        }
        Ok(value)
    };
    let epsilon = provider_number("epsilon")?;
    let threshold = provider_number("clipping_threshold")?;
    let policy = if parts[3].contains("policy: Strict") {
        MetricPolicy::Strict
    } else if parts[3].contains("policy: Clipped") {
        MetricPolicy::Clipped
    } else {
        return Err(error("recorded metric policy unavailable"));
    };
    let alive = &record.report.pre_clone_eligible;
    let rows = alive.len();
    let requested_slot = target;
    if target >= rows {
        return Err(error("conditional target outside recorded population"));
    }
    let target = if alive[target] {
        target
    } else {
        let choice = record
            .report
            .clone_plan
            .choices
            .get(target)
            .ok_or_else(|| error("revival clone choice unavailable"))?;
        if !choice.revival || !choice.accepted {
            return Err(error(
                "conditional target was ineligible at the recorded pre-clone stage",
            ));
        }
        let donor = choice
            .donors
            .first()
            .ok_or_else(|| error("revival conditional source unavailable"))?;
        let source = record
            .report
            .clone_plan
            .sources
            .get(donor.pool_index as usize)
            .ok_or_else(|| error("revival source identity unavailable"))?;
        let slot = source.slot as usize;
        if source.frame != record.report.step - 1
            || source.version != record.before.version
            || record.before.generations.get(slot) != Some(&source.generation)
            || !alive.get(slot).copied().unwrap_or(false)
        {
            return Err(error(
                "revival conditional field requires its exact eligible current-frame source",
            ));
        }
        slot
    };
    if archive.gas_config.distance_donors.count != 1
        || !matches!(
            archive.gas_config.reducer,
            crate::donor::CompanionReducer::Mean
        )
    {
        return Err(error(
            "conditional archive readout requires one distance companion with mean reduction",
        ));
    }
    let (velocity_field, velocity_weight) = match &archive.gas_config.distance_donors.distance {
        Distance::Euclidean {
            field,
            scales,
            squared: false,
            periodic: None,
        } if field == "positions" && (scales.is_empty() || scales.iter().all(|x| *x == 1.)) => {
            ("velocities", 0.)
        }
        Distance::PhaseSpace {
            positions,
            velocities,
            position_scale: 1.,
            velocity_scale: 1.,
            periodic: None,
            lambda,
        } if positions == "positions" => (velocities.as_str(), *lambda),
        _ => {
            return Err(error(
                "recorded conditional separation requires smooth unscaled Euclidean/phase-space distance without periodic seams",
            ));
        }
    };
    let positions = record.before.observations.field("positions")?;
    let velocity = record.before.observations.field(velocity_field)?;
    let points = positions
        .values()
        .chunks_exact(d)
        .map(|x| {
            x.iter()
                .enumerate()
                .map(|(a, v)| v - shift.get(a).copied().unwrap_or(0.))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let velocities = velocity
        .values()
        .chunks_exact(d)
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
    let floor = archive.gas_config.fitness.distance_floor;
    let constants = JetSpace::new(d, 0)?;
    let mut rewards = vec![0.; rows];
    let mut distances = vec![floor; rows];
    let mut donors = vec![None; rows];
    for i in 0..rows {
        if !alive[i] {
            continue;
        }
        rewards[i] = objective.value(&points[i])?;
        if record.report.distance_companions.valid[i] {
            let pool = record.report.distance_companions.indices[i] as usize;
            let source = record
                .report
                .distance_sources
                .get(pool)
                .ok_or_else(|| error("recorded companion pool index missing"))?;
            let source_population = if source.frame == record.report.step - 1
                && source.version == record.before.version
            {
                Some(&record.before)
            } else {
                archive
                    .steps
                    .iter()
                    .filter(|s| s.epoch == record.epoch)
                    .find_map(|s| {
                        if s.report.step - 1 == source.frame && s.before.version == source.version {
                            Some(&s.before)
                        } else if s.report.step == source.frame
                            && s.final_population.version == source.version
                        {
                            Some(&s.final_population)
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        archive
                            .anchors
                            .iter()
                            .find(|a| {
                                a.epoch == record.epoch
                                    && a.step == source.frame
                                    && a.population.version == source.version
                            })
                            .map(|a| &a.population)
                    })
            }
            .or_else(|| {
                history
                    .iter()
                    .find(|(frame, p)| *frame == source.frame && p.version == source.version)
                    .map(|(_, p)| p)
            })
            .ok_or_else(|| {
                error(
                    "historical companion source predates archive and supplied checkpoint coverage",
                )
            })?;
            let slot = source.slot as usize;
            if source_population.generations.get(slot) != Some(&source.generation) {
                return Err(error("recorded source generation mismatch"));
            }
            let source_point = source_population
                .observations
                .field("positions")?
                .row(slot)?
                .iter()
                .enumerate()
                .map(|(a, v)| v - shift.get(a).copied().unwrap_or(0.))
                .collect::<Vec<_>>();
            let source_velocity = source_population
                .observations
                .field(velocity_field)?
                .row(slot)?;
            let dv = velocities[i]
                .iter()
                .zip(source_velocity)
                .map(|(a, b)| velocity_weight * (a - b).powi(2))
                .sum::<f64>();
            let z = points[i]
                .iter()
                .map(|&v| constants.constant(v))
                .collect::<Vec<_>>();
            distances[i] = companion_distance(&z, &source_point, dv, floor)?.value();
            donors[i] = Some((source_point, dv, *source));
        }
    }
    let space = JetSpace::new(d, order)?;
    let x = query
        .iter()
        .enumerate()
        .map(|(a, &v)| space.variable(v - shift.get(a).copied().unwrap_or(0.), a))
        .collect::<Result<Vec<_>>>()?;
    let reward = objective.evaluate(&x)?;
    let distance = if let Some((point, dv, _)) = &donors[target] {
        companion_distance(&x, point, *dv, floor)?
    } else {
        space.constant(floor)
    };
    let pipeline = &archive.gas_config.fitness;
    let jet = if matches!(pipeline.reward_standardizer, Standardizer::Global { .. })
        && matches!(pipeline.diversity_standardizer, Standardizer::Global { .. })
    {
        ConditionalFitnessCache::new(&rewards, &distances, alive, pipeline)?
            .evaluate(target, &reward, &distance)?
    } else {
        let mut rs = rewards
            .iter()
            .map(|&x| space.constant(x))
            .collect::<Vec<_>>();
        let mut ds = distances
            .iter()
            .map(|&x| space.constant(x))
            .collect::<Vec<_>>();
        rs[target] = reward;
        ds[target] = distance;
        let rw = local_log_weights(
            &x,
            &points,
            &velocities,
            alive,
            target,
            &pipeline.reward_standardizer,
        )?;
        let dw = local_log_weights(
            &x,
            &points,
            &velocities,
            alive,
            target,
            &pipeline.diversity_standardizer,
        )?;
        pipeline_from_measurement_jets(
            &rs,
            &ds,
            alive,
            target,
            pipeline,
            rw.as_deref(),
            dw.as_deref(),
        )?
    };
    Ok((
        jet,
        json!({"record_index":record_index,"step":record.report.step,"epoch":record.epoch,"target_slot":target,"requested_slot":requested_slot,"revival_donor_field_extension":requested_slot!=target,"query":query,"source":donors[target].as_ref().map(|(_,_,s)|s),"provider":provider,"epsilon":epsilon,"policy":policy,"clipping_threshold":threshold,"population_size":rows,"pre_clone_population_version":record.before.version,"derivative_convention":"fixed coordinate probe; sampled donor identity and immutable source coordinates; differentiated global/local statistics"}),
    ))
}
