//! Fractal Set measurements of the executed planar gas. No independent point sampler.
use crate::lecture::number;
use algorithmic_gas::{
    GasError, Population, Result, RunArchive,
    partv_geometry::{self, Mat2, SpaceTimeFrame},
    physics::partvi::{ExperimentResult, Series},
    tracking::{RecordedStep, ScalarStageEncoding},
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

fn error(s: &str) -> GasError {
    GasError::Capability(s.into())
}
fn point_rows(p: &Population<f64>, include: bool) -> Result<Vec<(usize, [f64; 2])>> {
    let f = p.observations.field("positions")?;
    if f.width() != 2 {
        return Err(error(
            "Fractal Set lecture requires planar recorded positions",
        ));
    }
    (0..p.len())
        .filter(|&i| p.validity[i].eligible(include))
        .map(|i| {
            let row = f.row(i)?;
            Ok((i, [row[0], row[1]]))
        })
        .collect()
}
fn stage_rows(s: &RecordedStep<f64>, name: &str, include: bool) -> Result<Vec<(usize, [f64; 2])>> {
    let stage = s
        .stages
        .iter()
        .find(|v| v.stage == name)
        .ok_or_else(|| error("Recorded stage missing"))?;
    let f = stage
        .fields
        .get("positions")
        .ok_or_else(|| error("Recorded positions missing"))?;
    if f.item_shape.iter().product::<usize>() != 2 {
        return Err(error("Planar stage positions required"));
    }
    Ok(f.values
        .chunks_exact(2)
        .zip(&stage.validity)
        .enumerate()
        .filter(|(_, (_, v))| v.eligible(include))
        .map(|(i, (p, _))| (i, [p[0], p[1]]))
        .collect())
}
fn bounds(x: &[[f64; 2]]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in x {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].max(p[0]);
        b[2] = b[2].min(p[1]);
        b[3] = b[3].max(p[1]);
    }
    let margin = 0.1 * (b[1] - b[0]).max(b[3] - b[2]).max(1.);
    [b[0] - margin, b[1] + margin, b[2] - margin, b[3] + margin]
}
fn recorded_metric(s: &RecordedStep<f64>) -> Result<Mat2> {
    let f = s
        .field_evaluations
        .iter()
        .find(|f| f.stage == "O" && f.field == "fitness_metric")
        .ok_or_else(|| error("Experiment requires the executed fitness metric provider"))?;
    let i = f
        .available
        .iter()
        .position(|&x| x)
        .ok_or_else(|| error("No eligible metric samples"))?;
    let a = &f.values[4 * i..4 * i + 4];
    Ok([[a[0], a[1]], [a[2], a[3]]])
}
fn plot(out: &mut ExperimentResult, title: &str, x: &str, y: &str, series: Vec<Series>) {
    out.plot(title, x, y, series);
}
fn scalar_plot(out: &mut ExperimentResult, name: &str, values: Vec<[f64; 2]>) {
    plot(
        out,
        name,
        "recorded step",
        name,
        vec![Series::line("Measured", values)],
    );
}
fn norm(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

fn conditional_derivative_audit(
    a: &RunArchive<f64>,
    out: &mut ExperimentResult,
    p: &Value,
) -> Result<()> {
    use algorithmic_gas::physics::fields::archive_fitness_jet;
    let h = number(p, "difference_step", 0.001);
    if !(1e-6..=0.05).contains(&h) {
        return Err(error(
            "Derivative difference step must be between 1e-6 and 0.05",
        ));
    }
    let mut observed = vec![];
    let mut finite = vec![];
    let mut refinement = vec![];
    let mut reports = vec![];
    for (record, s) in a.steps.iter().enumerate() {
        let hessian = s
            .field_evaluations
            .iter()
            .find(|f| f.stage == "O" && f.field == "fitness_hessian")
            .ok_or_else(|| error("Recorded O Hessian missing"))?;
        let requested = number(p, "walker", 0.) as usize;
        let walker = if hessian.available.get(requested).copied().unwrap_or(false) {
            requested
        } else {
            hessian
                .available
                .iter()
                .position(|&v| v)
                .ok_or_else(|| error("No eligible Hessian readout"))?
        };
        let stage = s
            .stages
            .iter()
            .find(|f| f.stage == "A1")
            .ok_or_else(|| error("Recorded O input missing"))?;
        if stage.version != hessian.version {
            return Err(GasError::Capability(format!(
                "O derivative version{} and input{} disagree: {:?}",
                hessian.version,
                stage.version,
                s.stages
                    .iter()
                    .map(|s| (&s.stage, s.version))
                    .collect::<Vec<_>>()
            )));
        }
        let x = &stage.fields["positions"].values[walker * 2..walker * 2 + 2];
        let value = |point: &[f64]| -> Result<f64> {
            Ok(archive_fitness_jet(a, record, walker, point, 1)?.0.value())
        };
        let center = value(x)?;
        let difference = |step: f64| -> Result<Vec<f64>> {
            let mut result = vec![0.; 4];
            for i in 0..2 {
                for j in i..2 {
                    let z = if i == j {
                        let mut plus = x.to_vec();
                        let mut minus = x.to_vec();
                        plus[i] += step;
                        minus[i] -= step;
                        (value(&plus)? - 2. * center + value(&minus)?) / (step * step)
                    } else {
                        let mut sum = 0.;
                        for si in [-1., 1.] {
                            for sj in [-1., 1.] {
                                let mut point = x.to_vec();
                                point[i] += si * step;
                                point[j] += sj * step;
                                sum += si * sj * value(&point)?;
                            }
                        }
                        sum / (4. * step * step)
                    };
                    result[i * 2 + j] = z;
                    result[j * 2 + i] = z;
                }
            }
            Ok(result)
        };
        let fd = difference(h)?;
        let refined = difference(h / 2.)?;
        let raw = &hessian.values[walker * 4..walker * 4 + 4];
        let residual = raw
            .iter()
            .zip(&refined)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        let diff = fd
            .iter()
            .zip(&refined)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f64::max);
        observed.push([
            s.report.step as f64,
            raw.iter().map(|x| x * x).sum::<f64>().sqrt(),
        ]);
        finite.push([
            s.report.step as f64,
            refined.iter().map(|x| x * x).sum::<f64>().sqrt(),
        ]);
        refinement.push([s.report.step as f64, residual]);
        reports.push(json!({"step":s.report.step,"slot":walker,"query":x,"recorded_hessian":raw,"finite_difference_hessian":refined,"absolute_residual":residual,"refinement_discrepancy":diff,"difference_step":h}));
    }
    plot(
        out,
        "Executed Hessian versus scalar finite differences",
        "recorded step",
        "Hessian norm",
        vec![
            Series::line("Recorded analytic Hessian", observed),
            Series::line("Independent scalar differences", finite),
        ],
    );
    scalar_plot(out, "Hessian finite-difference discrepancy", refinement);
    out.details["conditional_derivative_audit"] = json!(reports);
    Ok(())
}

fn event_id(e: algorithmic_gas::fractal_set::EventRef) -> String {
    format!(
        "{}:{}:{}:{}:{}",
        e.epoch, e.step, e.version, e.slot, e.generation
    )
}
fn graph_scene(
    a: &RunArchive<f64>,
    g: &algorithmic_gas::fractal_set::FractalSet,
    dt: f64,
) -> Value {
    let mut points = BTreeMap::new();
    let mut add = |epoch: u64, step: u64, p: &Population<f64>| {
        if let Ok(field) = p.observations.field("positions") {
            for i in 0..p.len() {
                if let Ok(x) = field.row(i)
                    && x.len() == 2
                    && x.iter().all(|v| v.is_finite())
                {
                    points.insert(
                        algorithmic_gas::fractal_set::EventRef {
                            epoch,
                            step,
                            version: p.version,
                            slot: i as u32,
                            generation: p.generations[i],
                        },
                        [x[0], x[1], step as f64 * dt],
                    );
                }
            }
        }
    };
    for anchor in &a.anchors {
        add(anchor.epoch, anchor.step, &anchor.population);
    }
    for step in &a.steps {
        add(step.epoch, step.report.step - 1, &step.before);
        add(step.epoch, step.report.step, &step.final_population);
    }
    // Stage events share the exact identity used by graph source resolution.
    // Index them once instead of rescanning all archive steps per graph node.
    for step in &a.steps {
        for stage in &step.stages {
            if let Some(field) = stage.fields.get("positions")
                && field.item_shape.iter().product::<usize>() == 2
            {
                for (i, x) in field.values.chunks_exact(2).enumerate() {
                    if x.iter().all(|v| v.is_finite()) {
                        points
                            .entry(algorithmic_gas::fractal_set::EventRef {
                                epoch: step.epoch,
                                step: step.report.step,
                                version: stage.version,
                                slot: i as u32,
                                generation: stage.generations[i],
                            })
                            .or_insert([x[0], x[1], step.report.step as f64 * dt]);
                    }
                }
            }
        }
    }
    let last = a.steps.last().unwrap();
    let end = last.report.step;
    let start = end.saturating_sub(7).max(1);
    let within = |e: algorithmic_gas::fractal_set::EventRef| {
        e.epoch == last.epoch && (start..=end).contains(&e.step)
    };
    let selected_edges = g
        .edges
        .iter()
        .filter(|e| within(e.target))
        .collect::<Vec<_>>();
    // Keep every incoming edge source, including historical events before the window.
    let mut selected = g
        .nodes
        .iter()
        .copied()
        .filter(|e| within(*e))
        .collect::<BTreeSet<_>>();
    for edge in &selected_edges {
        selected.insert(edge.source);
        selected.insert(edge.target);
    }
    let mut nodes = vec![];
    let mut visible = BTreeSet::new();
    for &event in &selected {
        if let Some(point) = points.get(&event)
            && point.iter().all(|v| v.is_finite())
        {
            visible.insert(event);
            nodes.push(json!({"id":event_id(event),"position":point,"owner":event.slot,"layer":"recorded events","label":format!("step {} slot {} generation {}",event.step,event.slot,event.generation),"detail":format!("epoch {}, step {}, population version {}, slot {}, generation {}",event.epoch,event.step,event.version,event.slot,event.generation)}));
        }
    }
    let edges=selected_edges.iter().filter(|e|visible.contains(&e.source)&&visible.contains(&e.target)).map(|e|json!({"source":event_id(e.source),"target":event_id(e.target),"layer":serde_json::to_value(e.kind).unwrap()})).collect::<Vec<_>>();
    let display = json!({
        "epoch":last.epoch,"window_start_step":start,"window_end_step":end,
        "window_rule":"all edges whose target is in the final eight recorded transitions, including historical source endpoints",
        "total_nodes":g.nodes.len(),"displayed_nodes":nodes.len(),"omitted_nodes":g.nodes.len()-nodes.len(),
        "total_edges":g.edges.len(),"displayed_edges":edges.len(),"omitted_edges":g.edges.len()-edges.len(),
        "selected_nodes":selected.len(),"selected_edges":selected_edges.len(),
        "unpositioned_events":selected.len()-visible.len()
    });
    json!({"title":"Executed Fractal Set events and relations","coordinateSystem":"spacetime","nodes":nodes,"edges":edges,"faces":[],"message":format!("Displaying every relation targeting recorded steps {start}–{end} in epoch {}, with all historical source endpoints. Measurements use all {} recorded steps; the complete exported archive reconstructs the full graph. Select an event and cut this displayed window at physical time.",last.epoch,a.steps.len()),"display":display,"unpositioned_events":selected.len()-visible.len()})
}

pub fn analyze(id: &str, p: &Value, a: &RunArchive<f64>) -> Result<ExperimentResult> {
    a.validate()?;
    let index: u32 = id
        .strip_prefix("V-")
        .ok_or_else(|| error("Invalid Fractal Set ID"))?
        .parse()
        .map_err(|_| error("Invalid Fractal Set ID"))?;
    let spec = algorithmic_gas::lecture::specification(id)?;
    let mut out = ExperimentResult::new(
        index,
        spec["title"].as_str().unwrap(),
        "Recorded Euclidean Gas events and conditional fields",
    );
    let last = a
        .steps
        .last()
        .ok_or_else(|| error("Run the gas before measuring the Fractal Set"))?;
    let dt = match a.gas_config.kinetic.integrator {
        algorithmic_gas::kinetic::KineticKind::Baoab { dt, .. } => dt,
        _ => return Err(error("Planar lecture requires BAOAB")),
    };
    let include = a.gas_config.include_truncated;
    let points = |population: &Population<f64>| -> Result<Vec<[f64; 2]>> {
        Ok(point_rows(population, include)?
            .into_iter()
            .map(|(_, x)| x)
            .collect())
    };
    let x = points(&last.final_population)?;
    if x.is_empty() {
        return Err(error("Population extinct: no spatial measurement support"));
    }
    out.details = json!({"lecture_id":id,"calculation_origin":"executed_algorithm_archive","archive_steps":a.steps.len(),"executed_gas_config":a.gas_config,"sampling_unit":"One interacting population trajectory; walker rows and successive frames are not iid samples"});
    out.details["scene"] = json!({"title":"Executed planar gas population","coordinateSystem":"spacetime","nodes":point_rows(&last.final_population,include)?.iter().map(|(i,x)|json!({"id":format!("slot-{i}"),"position":[x[0],x[1],last.report.step as f64*dt],"owner":i,"layer":"recorded walkers","label":format!("slot {i}"),"detail":format!("Executed step {}, slot {}, generation {}",last.report.step,i,last.final_population.generations[*i])})).collect::<Vec<_>>(),"edges":[],"faces":[],"message":"Actual eligible walker positions at the final recorded update."});
    match index {
        1..=4 => {
            let g = a.graph();
            if !g.unresolved_sources.is_empty() {
                return Err(error("Historical sources outside archive coverage"));
            }
            match index {
                1 => {
                    let slot = number(p, "walker", 0.) as u32;
                    let root = g
                        .nodes
                        .iter()
                        .find(|e| e.slot == slot)
                        .copied()
                        .ok_or_else(|| error("Selected slot missing"))?;
                    let descendants = g.descendants(root);
                    let causal = g.causal_future(root);
                    plot(
                        &mut out,
                        "Ancestry versus numerical slot persistence",
                        "step",
                        "reachable events",
                        vec![
                            Series::line(
                                "Material descendants",
                                a.steps
                                    .iter()
                                    .map(|s| {
                                        [
                                            s.report.step as f64,
                                            descendants
                                                .iter()
                                                .filter(|e| e.step == s.report.step)
                                                .count()
                                                as f64,
                                        ]
                                    })
                                    .collect(),
                            ),
                            Series::line(
                                "Causal slot future",
                                a.steps
                                    .iter()
                                    .map(|s| {
                                        [
                                            s.report.step as f64,
                                            causal
                                                .iter()
                                                .filter(|e| e.step == s.report.step)
                                                .count()
                                                as f64,
                                        ]
                                    })
                                    .collect(),
                            ),
                        ],
                    );
                    out.details["selected_event"] = json!(root);
                }
                2 => {
                    out.metric(
                        "Boundary of triangle boundary residual",
                        if g.boundary_squared_zero() { 0. } else { 1. },
                        "identity",
                    );
                    let mut counts = vec![];
                    let mut residual = vec![];
                    for t in &g.triangles {
                        counts.push([t.vertices[2].step as f64, t.channels.len() as f64]);
                        let edges = t
                            .boundary_edges
                            .map(|i| g.edges[i].attributes.position_displacement);
                        if let [Some(u), Some(v), Some(w)] = edges {
                            residual.push([
                                t.vertices[2].step as f64,
                                norm([u[0] + v[0] - w[0], u[1] + v[1] - w[1]]),
                            ]);
                        }
                    }
                    plot(
                        &mut out,
                        "Actual interaction triangles",
                        "recorded step",
                        "count / residual",
                        vec![
                            Series::line("Channels per triangle", counts),
                            Series::line("Displacement closure residual", residual),
                        ],
                    );
                }
                3 => {
                    let mut errors = vec![];
                    for s in &a.steps {
                        let mut max = 0f64;
                        for stage in &s.stages {
                            let decoded = ScalarStageEncoding::encode(stage).decode()?;
                            for (key, f) in &stage.fields {
                                for (u, v) in f.values.iter().zip(&decoded[key].values) {
                                    if u.is_finite() && v.is_finite() {
                                        max = max.max((u - v).abs());
                                    }
                                }
                            }
                        }
                        errors.push([s.report.step as f64, max]);
                    }
                    scalar_plot(&mut out, "Recorded scalar reconstruction error", errors);
                    let mut encoded = vec![];
                    for e in &g.edges {
                        if let (Some(v), Some(z)) = (
                            e.attributes.position_displacement,
                            e.attributes.spin2_displacement,
                        ) {
                            let decoded = algorithmic_gas::partv_analysis::spin2_decode(z);
                            encoded.push([
                                e.target.step as f64,
                                norm([v[0] - decoded[0], v[1] - decoded[1]]),
                            ]);
                        }
                    }
                    scalar_plot(
                        &mut out,
                        "Recorded displacement spinor decoding error",
                        encoded,
                    );
                }
                4 => {
                    // Measured oriented tangent turning on actual recorded triangles.
                    let mut turning = vec![];
                    let mut closure = vec![];
                    for t in &g.triangles {
                        let edges = t
                            .boundary_edges
                            .map(|i| g.edges[i].attributes.position_displacement);
                        if let [Some(u), Some(v), Some(w)] = edges
                            && norm(u) > 1e-12
                            && norm(v) > 1e-12
                            && norm(w) > 1e-12
                        {
                            let tangents = [u, v, [-w[0], -w[1]]];
                            let mut angle = 0.;
                            for i in 0..3 {
                                let v = tangents[i];
                                let w = tangents[(i + 1) % 3];
                                angle +=
                                    (v[0] * w[1] - v[1] * w[0]).atan2(v[0] * w[0] + v[1] * w[1]);
                            }
                            turning.push([t.vertices[2].step as f64, angle]);
                            closure.push([
                                t.vertices[2].step as f64,
                                angle.sin().abs().max((angle.cos() - 1.).abs()),
                            ]);
                        }
                    }
                    plot(
                        &mut out,
                        "Transport of observed edge tangents",
                        "recorded step",
                        "radians / residual",
                        vec![
                            Series::line("Oriented tangent turning", turning),
                            Series::line("Closed tangent phase residual", closure),
                        ],
                    );
                    let mut edge_angles = vec![];
                    let mut support = vec![];
                    for step in &a.steps {
                        let mut nonzero = 0;
                        for edge in g.edges.iter().filter(|e| e.target.step == step.report.step) {
                            if let Some(v) = edge.attributes.position_displacement
                                && norm(v) > 1e-12
                            {
                                edge_angles.push([step.report.step as f64, v[1].atan2(v[0])]);
                                nonzero += 1;
                            }
                        }
                        let triangles = g
                            .triangles
                            .iter()
                            .filter(|t| t.vertices[2].step == step.report.step)
                            .count();
                        support.push(json!({"step":step.report.step,"recorded_triangles":triangles,"nonzero_tangent_edges":nonzero}));
                    }
                    plot(
                        &mut out,
                        "Actual edge tangent observations",
                        "recorded step",
                        "radians",
                        vec![Series::line(
                            "Nonzero recorded displacement angle",
                            edge_angles,
                        )],
                    );
                    plot(
                        &mut out,
                        "Measured tangent support",
                        "recorded step",
                        "edges",
                        vec![Series::line(
                            "Nonzero recorded edge count",
                            support
                                .iter()
                                .map(|v| {
                                    [
                                        v["step"].as_u64().unwrap() as f64,
                                        v["nonzero_tangent_edges"].as_u64().unwrap() as f64,
                                    ]
                                })
                                .collect(),
                        )],
                    );
                    out.details["tangent_support"] = json!(support);
                    out.note("This connection is defined only for nonzero recorded planar edge tangents. Degenerate or unavailable triangle tangents contribute no turning angle; measured edge orientations and support remain visible.");
                }
                _ => unreachable!(),
            }
            out.metric("Recorded events", g.nodes.len() as f64, "events");
            out.metric(
                "Recorded interaction triangles",
                g.triangles.len() as f64,
                "triangles",
            );
            out.details["scene"] = graph_scene(a, &g, dt);
            let mut kinds = BTreeMap::<String, usize>::new();
            for edge in &g.edges {
                let kind = serde_json::to_value(edge.kind).unwrap();
                *kinds.entry(kind.as_str().unwrap().into()).or_default() += 1;
            }
            out.details["graph_summary"] = json!({
                "recorded_steps":a.steps.len(),"nodes":g.nodes.len(),"edges":g.edges.len(),
                "triangles":g.triangles.len(),"edges_by_kind":kinds,
                "unresolved_sources":g.unresolved_sources.len(),
                "measurement_scope":"complete recorded graph",
                "reconstruction":"RunArchive::graph() from the complete exported algorithm archive"
            });
        }
        5 | 13 | 20 => {
            let names = if index == 5 {
                vec!["fitness_hessian", "fitness_metric"]
            } else if index == 13 {
                vec!["fitness_hessian"]
            } else {
                vec!["fitness_scalar_curvature"]
            };
            for name in names {
                let mut series = vec![];
                let mut rows = vec![];
                for s in &a.steps {
                    let f = s
                        .field_evaluations
                        .iter()
                        .find(|f| f.stage == "O" && f.field == name)
                        .ok_or_else(|| error("Required executed conditional field missing"))?;
                    let width = f.values.len() / f.available.len().max(1);
                    let vals = f
                        .values
                        .chunks(width.max(1))
                        .zip(&f.available)
                        .filter(|(_, ok)| **ok)
                        .collect::<Vec<_>>();
                    if !vals.is_empty() {
                        let mean = vals
                            .iter()
                            .map(|(r, _)| r.iter().map(|v| v * v).sum::<f64>().sqrt())
                            .sum::<f64>()
                            / vals.len() as f64;
                        series.push([s.report.step as f64, mean]);
                    }
                    rows.push(
                        json!({"step":s.report.step,"available":f.available,"values":f.values}),
                    );
                }
                scalar_plot(&mut out, name, series);
                out.details[name] = json!(rows);
            }
            if index == 5 || index == 13 {
                conditional_derivative_audit(a, &mut out, p)?;
            }
            if index == 20 {
                let mut residuals = vec![];
                for s in &a.steps {
                    let field = |name: &str| {
                        s.field_evaluations
                            .iter()
                            .find(|f| f.stage == "O" && f.field == name)
                            .ok_or_else(|| {
                                error("Curvature contraction needs complete recorded fields")
                            })
                    };
                    let metric = field("fitness_metric")?;
                    let ricci = field("fitness_ricci")?;
                    let scalar = field("fitness_scalar_curvature")?;
                    let mut maximum = 0_f64;
                    let mut count = 0;
                    for i in 0..scalar.available.len() {
                        if scalar.available[i] && ricci.available[i] && metric.available[i] {
                            let g = &metric.values[4 * i..4 * i + 4];
                            let r = &ricci.values[4 * i..4 * i + 4];
                            let det = g[0] * g[3] - g[1] * g[2];
                            if det <= 0. {
                                return Err(error(
                                    "Recorded metric must have positive determinant",
                                ));
                            }
                            let contraction =
                                (g[3] * r[0] - g[1] * r[2] - g[2] * r[1] + g[0] * r[3]) / det;
                            maximum = maximum.max((contraction - scalar.values[i]).abs());
                            count += 1;
                        }
                    }
                    if count > 0 {
                        residuals.push([s.report.step as f64, maximum]);
                    }
                }
                scalar_plot(
                    &mut out,
                    "Recorded inverse-metric Ricci contraction residual",
                    residuals,
                );
            }
            if index == 13 {
                let mut differences = vec![];
                for s in &a.steps {
                    let before = point_rows(&s.before, include)?;
                    let after = point_rows(&s.final_population, include)?;
                    let common: Vec<_> = before
                        .iter()
                        .filter_map(|(i, u)| {
                            after
                                .iter()
                                .find(|(j, _)| i == j)
                                .map(|(_, v)| norm([u[0] - v[0], u[1] - v[1]]))
                        })
                        .collect();
                    if !common.is_empty() {
                        differences.push([
                            s.report.step as f64,
                            common.iter().sum::<f64>() / common.len() as f64,
                        ]);
                    }
                }
                scalar_plot(
                    &mut out,
                    "Actual probe displacement across changing strata",
                    differences,
                );
                out.note("The displayed derivatives use the frozen conditional context of each executed O stage. Differences between steps include changes of context and motion.");
            }
        }
        6 | 8 => {
            let mut measured = vec![];
            let mut predicted = vec![];
            let mut cov = vec![];
            let friction = match a.gas_config.kinetic.integrator {
                algorithmic_gas::kinetic::KineticKind::Baoab { friction, .. } => friction,
                _ => unreachable!(),
            };
            for s in &a.steps {
                let m = s.thermostat_moments(
                    "velocities",
                    1.,
                    dt,
                    friction,
                    a.gas_config.include_truncated,
                )?;
                measured.push([s.report.step as f64, m.measured_energy_change]);
                predicted.push([s.report.step as f64, m.predicted_energy_change]);
                cov.push(json!({"step":s.report.step,"conditional_total_momentum_covariance":m.momentum_covariance,"eligible_rows":m.eligible_rows}));
            }
            plot(
                &mut out,
                "Executed thermostat energy and conditional prediction",
                "recorded step",
                "energy increment",
                vec![
                    Series::line("Measured O stage", measured),
                    Series::line("Prediction from executed factor", predicted),
                ],
            );
            out.details["conditional_covariances"] = json!(cov);
            if index == 8 {
                let mut xx = vec![];
                let mut yy = vec![];
                for s in &a.steps {
                    let q = points(&s.final_population)?;
                    let n = q.len() as f64;
                    if n > 0. {
                        let mx = q.iter().map(|v| v[0]).sum::<f64>() / n;
                        let my = q.iter().map(|v| v[1]).sum::<f64>() / n;
                        xx.push([
                            s.report.step as f64 * dt,
                            q.iter().map(|v| (v[0] - mx).powi(2)).sum::<f64>() / n,
                        ]);
                        yy.push([
                            s.report.step as f64 * dt,
                            q.iter().map(|v| (v[1] - my).powi(2)).sum::<f64>() / n,
                        ]);
                    }
                }
                plot(
                    &mut out,
                    "Population covariance during interacting relaxation",
                    "physical time",
                    "empirical variance",
                    vec![
                        Series::line("Position coordinate 1", xx),
                        Series::line("Position coordinate 2", yy),
                    ],
                );
            }
        }
        7 | 9 | 10 | 11 | 12 => {
            let metric = recorded_metric(last)?;
            let domain = bounds(&x);
            let mesh = partv_geometry::voronoi(&x, domain, metric)?;
            let sites = point_rows(&last.final_population, include)?;
            out.details["scene"]["title"] = json!("Metric cells of actual recorded walkers");
            out.details["scene"]["faces"]=json!(mesh.cells.iter().filter(|c|c.vertices.len()>=3).map(|c|json!({"owner":sites[c.slot].0,"layer":"metric Voronoi cells","vertices":c.vertices.iter().map(|v|[v[0],v[1],last.report.step as f64*dt]).collect::<Vec<_>>()})).collect::<Vec<_>>());
            out.details["scene"]["edges"]=json!(mesh.neighbors.iter().map(|[i,j]|json!({"source":format!("slot-{}",sites[*i].0),"target":format!("slot-{}",sites[*j].0),"layer":"cell neighbors"})).collect::<Vec<_>>());
            out.details["mesh"] = json!(mesh);
            match index {
                7 => {
                    plot(
                        &mut out,
                        "Measured cell volumes in the recorded local metric",
                        "eligible site index",
                        "area",
                        vec![
                            Series::line(
                                "Coordinate cell area",
                                mesh.cells.iter().map(|c| [c.slot as f64, c.area]).collect(),
                            ),
                            Series::line(
                                "Metric cell area",
                                mesh.cells
                                    .iter()
                                    .map(|c| [c.slot as f64, c.geometric_area])
                                    .collect(),
                            ),
                        ],
                    );
                    out.metric("Partition closure", mesh.closure_error, "area");
                    out.note("The displayed local chart uses the first available executed O-stage metric, frozen across this partition.");
                }
                9 => {
                    let mut degree = vec![0.; x.len()];
                    for [i, j] in &mesh.neighbors {
                        degree[*i] += 1.;
                        degree[*j] += 1.;
                    }
                    plot(
                        &mut out,
                        "Dual graph of recorded walker cells",
                        "eligible site index",
                        "neighbors",
                        vec![Series::line(
                            "Cell degree",
                            degree
                                .iter()
                                .enumerate()
                                .map(|(i, d)| [i as f64, *d])
                                .collect(),
                        )],
                    );
                    out.metric("Partition closure", mesh.closure_error, "area");
                }
                10 => {
                    let mut cloning = vec![];
                    let mut kinetic = vec![];
                    let mut interface_records = vec![];
                    for s in &a.steps {
                        let pre_rows = point_rows(&s.before, include)?;
                        let post_rows = stage_rows(s, "post_clone", include)?;
                        let end_rows = point_rows(&s.final_population, include)?;
                        let pre: Vec<_> = pre_rows.iter().map(|(_, x)| *x).collect();
                        let post: Vec<_> = post_rows.iter().map(|(_, x)| *x).collect();
                        let end: Vec<_> = end_rows.iter().map(|(_, x)| *x).collect();
                        if pre.is_empty() || post.is_empty() || end.is_empty() {
                            continue;
                        }
                        let mut all = pre.clone();
                        all.extend(&post);
                        all.extend(&end);
                        let b = bounds(&all);
                        let g = recorded_metric(s)?;
                        let edges = |pts: &[[f64; 2]],
                                     rows: &[(usize, [f64; 2])]|
                         -> Result<BTreeSet<[usize; 2]>> {
                            Ok(partv_geometry::voronoi(pts, b, g)?
                                .neighbors
                                .into_iter()
                                .map(|[i, j]| {
                                    let mut edge = [rows[i].0, rows[j].0];
                                    edge.sort();
                                    edge
                                })
                                .collect())
                        };
                        let aa = edges(&pre, &pre_rows)?;
                        let bb = edges(&post, &post_rows)?;
                        let cc = edges(&end, &end_rows)?;
                        interface_records.push(json!({"step":s.report.step,"before_edges":aa,"post_clone_edges":bb,"final_edges":cc,"before_slots":pre_rows.iter().map(|(i,_)|i).collect::<Vec<_>>(),"post_clone_slots":post_rows.iter().map(|(i,_)|i).collect::<Vec<_>>(),"final_slots":end_rows.iter().map(|(i,_)|i).collect::<Vec<_>>()}));
                        cloning.push([
                            s.report.step as f64,
                            aa.symmetric_difference(&bb).count() as f64,
                        ]);
                        kinetic.push([
                            s.report.step as f64,
                            bb.symmetric_difference(&cc).count() as f64,
                        ]);
                    }
                    out.details["interface_records"] = json!(interface_records);
                    plot(
                        &mut out,
                        "Cell interface changes by executed operation",
                        "step",
                        "changed interfaces",
                        vec![
                            Series::line("Clone stage", cloning),
                            Series::line("Kinetic stage", kinetic),
                        ],
                    );
                }
                11 => {
                    let pre_rows = point_rows(&last.before, include)?;
                    let post_rows = stage_rows(last, "post_clone", include)?;
                    let final_rows = point_rows(&last.final_population, include)?;
                    let common: Vec<usize> = pre_rows
                        .iter()
                        .map(|(i, _)| *i)
                        .filter(|i| {
                            post_rows.iter().any(|(j, _)| j == i)
                                && final_rows.iter().any(|(j, _)| j == i)
                        })
                        .collect();
                    if common.len() < 2 {
                        return Err(error(
                            "Material slab requires two slots eligible at all three recorded stages",
                        ));
                    }
                    let project = |rows: &[(usize, [f64; 2])]| {
                        common
                            .iter()
                            .map(|i| rows.iter().find(|(j, _)| j == i).unwrap().1)
                            .collect::<Vec<_>>()
                    };
                    let pre = project(&pre_rows);
                    let post = project(&post_rows);
                    let end = project(&final_rows);
                    out.details["slab_slots"] = json!({"tracked_slots":common,"before_eligible":pre_rows.len(),"post_clone_eligible":post_rows.len(),"final_eligible":final_rows.len(),"convention":"common eligible numerical slots; cloning replacements are explicit zero-duration jumps"});
                    let t = (last.report.step - 1) as f64 * dt;
                    let frames = vec![
                        SpaceTimeFrame {
                            time: t,
                            points: pre,
                        },
                        SpaceTimeFrame {
                            time: t,
                            points: post,
                        },
                        SpaceTimeFrame {
                            time: t + dt,
                            points: end.clone(),
                        },
                    ];
                    let mut all = vec![];
                    for f in &frames {
                        all.extend(&f.points);
                    }
                    let slab = partv_geometry::spacetime(
                        &frames,
                        bounds(&all),
                        number(p, "resolution", 4.) as usize,
                        metric,
                    )?;
                    out.details["slab"] = slab.clone();
                    out.details["scene"]["title"] =
                        json!("Material cells along the recorded slot trajectories");
                    let common_ref = &common;
                    out.details["scene"]["nodes"]=json!(frames.iter().enumerate().flat_map(|(frame,f)|f.points.iter().enumerate().map(move|(i,x)|json!({"id":format!("slab-{frame}-{i}"),"owner":common_ref[i],"position":[x[0],x[1],f.time],"layer":"recorded slab endpoints","label":format!("slot {} frame {frame}",common_ref[i])}))).collect::<Vec<_>>());
                    out.details["scene"]["edges"]=json!((0..2).flat_map(|frame|(0..common.len()).map(move|i|json!({"source":format!("slab-{frame}-{i}"),"target":format!("slab-{}-{i}",frame+1),"layer":if frame==0{"clone replacement"}else{"kinetic motion"}}))).collect::<Vec<_>>());
                    out.details["scene"]["faces"]=json!(slab["boundary_faces"].as_array().into_iter().flatten().map(|f|json!({"vertices":f["vertices"],"owner":common[f["slot"].as_u64().unwrap()as usize],"layer":"material cell boundary"})).collect::<Vec<_>>());
                    for (key, v) in slab.as_object().into_iter().flatten() {
                        if let Some(n) = v.as_f64() {
                            out.metric(key, n, "slab units");
                        }
                    }
                    plot(
                        &mut out,
                        "Recorded slot motion across the slab",
                        "eligible site index",
                        "displacement",
                        vec![Series::line(
                            "Endpoint displacement",
                            frames[0]
                                .points
                                .iter()
                                .zip(&end)
                                .enumerate()
                                .map(|(i, (u, v))| [i as f64, norm([u[0] - v[0], u[1] - v[1]])])
                                .collect(),
                        )],
                    );
                }
                12 => {
                    let retained = a.steps.iter().rev().take(8).rev().collect::<Vec<_>>();
                    let rows = retained
                        .iter()
                        .map(|s| point_rows(&s.final_population, include))
                        .collect::<Result<Vec<_>>>()?;
                    let common: Vec<usize> = rows[0]
                        .iter()
                        .map(|(i, _)| *i)
                        .filter(|i| rows.iter().all(|frame| frame.iter().any(|(j, _)| i == j)))
                        .collect();
                    if common.len() < 2 {
                        return Err(error("Mesh history requires two common eligible slots"));
                    }
                    let frames = rows
                        .iter()
                        .map(|frame| {
                            common
                                .iter()
                                .map(|i| frame.iter().find(|(j, _)| j == i).unwrap().1)
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    out.details["mesh_history_slots"] = json!(common);
                    let activity = partv_geometry::triangulation(&frames, metric)?;
                    out.details["mesh_activity"] = activity;
                    let values = frames
                        .iter()
                        .enumerate()
                        .map(|(i, q)| {
                            Ok([
                                i as f64,
                                partv_geometry::voronoi(q, bounds(q), metric)?
                                    .neighbors
                                    .len() as f64,
                            ])
                        })
                        .collect::<Result<Vec<_>>>()?;
                    scalar_plot(&mut out, "Recorded mesh interface count", values);
                }
                _ => unreachable!(),
            }
        }
        14..=16 => {
            let width = number(p, "bandwidth", 0.4);
            if width <= 0. {
                return Err(error("Kernel bandwidth must be positive"));
            }
            let s = last;
            let selected = point_rows(&s.before, include)?;
            let pts: Vec<_> = selected.iter().map(|(_, x)| *x).collect();
            let values: Vec<f64> = selected
                .iter()
                .map(|(i, _)| s.report.pre_clone_fitness.fitness[*i])
                .collect();
            let kernel = |h: f64| {
                let mut rows = vec![];
                for u in &pts {
                    let mut w = pts
                        .iter()
                        .map(|v| {
                            (-((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2)) / (2. * h * h)).exp()
                        })
                        .collect::<Vec<_>>();
                    let z = w.iter().sum::<f64>();
                    for v in &mut w {
                        *v /= z;
                    }
                    rows.push(w);
                }
                rows
            };
            if index == 14 {
                let rows = kernel(width);
                out.details["kernel"] = json!(rows);
                plot(
                    &mut out,
                    "Empirical kernel normalization on the actual population",
                    "eligible slot",
                    "row sum",
                    vec![
                        Series::line(
                            "Measured row sum",
                            rows.iter()
                                .enumerate()
                                .map(|(i, r)| [i as f64, r.iter().sum()])
                                .collect(),
                        ),
                        Series::line(
                            "Constant preservation",
                            (0..pts.len()).map(|i| [i as f64, 1.]).collect(),
                        ),
                    ],
                );
            }
            if index == 15 {
                let rows = kernel(width);
                let graph = rows
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        r.iter()
                            .zip(&values)
                            .map(|(w, v)| w * (v - values[i]))
                            .sum::<f64>()
                            / (width * width)
                    })
                    .collect::<Vec<_>>();
                if a.steps.len() < 3 {
                    return Err(error(
                        "Temporal wave difference requires three recorded frames",
                    ));
                }
                let mut temporal = vec![];
                if a.steps.len() >= 3 {
                    let t = a.steps.len();
                    let u = &a.steps[t - 3].report.pre_clone_fitness.fitness;
                    let v = &a.steps[t - 2].report.pre_clone_fitness.fitness;
                    for (i, (slot, _)) in selected.iter().enumerate() {
                        if a.steps[t - 3].before.validity[*slot].eligible(include)
                            && a.steps[t - 2].before.validity[*slot].eligible(include)
                        {
                            temporal.push([
                                *slot as f64,
                                (values[i] - 2. * v[*slot] + u[*slot]) / (dt * dt),
                            ]);
                        }
                    }
                }
                plot(
                    &mut out,
                    "Discrete wave-operator terms on recorded fitness",
                    "slot",
                    "operator value",
                    vec![
                        Series::line("Second slot-time difference", temporal),
                        Series::line(
                            "Empirical spatial kernel generator",
                            graph
                                .iter()
                                .enumerate()
                                .map(|(i, v)| [selected[i].0 as f64, *v])
                                .collect(),
                        ),
                    ],
                );
                out.note("The temporal difference follows numerical slots and includes cloning jumps. This is a defined discrete operator on recorded fitness; its continuum wave interpretation is a hypothesis.");
            }
            if index == 16 {
                let cut = a.steps.len() / 2;
                let train = &a.steps[..cut.max(1)];
                let test = &a.steps[cut.max(1).min(a.steps.len())..];
                let mut curve = vec![];
                for h in [width / 2., width, width * 2.] {
                    let mut error = 0.;
                    let mut count = 0.;
                    for q in test {
                        let xx = point_rows(&q.before, include)?;
                        for (i, u) in &xx {
                            let mut weighted = 0.;
                            let mut weight = 0.;
                            for row in train {
                                let yy = point_rows(&row.before, include)?;
                                for (j, v) in &yy {
                                    let w = (-((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2))
                                        / (2. * h * h))
                                        .exp();
                                    weighted += w * row.report.pre_clone_fitness.fitness[*j];
                                    weight += w;
                                }
                            }
                            if weight > 0. {
                                error += (weighted / weight
                                    - q.report.pre_clone_fitness.fitness[*i])
                                    .powi(2);
                                count += 1.;
                            }
                        }
                    }
                    if count > 0. {
                        curve.push([h, (error / count).sqrt()]);
                    }
                }
                plot(
                    &mut out,
                    "Chronological held-out fitness prediction",
                    "bandwidth",
                    "RMSE",
                    vec![Series::line("Later recorded populations", curve)],
                );
                out.details["training_frames"] = json!(train.len());
                out.details["heldout_frames"] = json!(test.len());
            }
        }
        17..=19 => {
            let order = a.compare_orders(
                number(p, "speed", 1.),
                dt,
                number(p, "max_nodes", 128.) as usize,
            )?;
            if index == 17 {
                plot(
                    &mut out,
                    "Recorded ordering versus chart light cones",
                    "comparison",
                    "ordered pairs",
                    vec![Series::line(
                        "Counts",
                        vec![
                            [0., order.both as f64],
                            [1., order.cst_only as f64],
                            [2., order.lorentz_only as f64],
                        ],
                    )],
                );
                out.details["orders"] = json!(order);
            }
            if index == 18 {
                let half = number(p, "region", 0.5);
                if half <= 0. {
                    return Err(error("Positive observation region half-width required"));
                }
                let mut counts = vec![];
                let mut density = vec![];
                for s in &a.steps {
                    let xx = points(&s.final_population)?;
                    let n = xx
                        .iter()
                        .filter(|v| v[0].abs() <= half && v[1].abs() <= half)
                        .count();
                    counts.push([s.report.step as f64, n as f64]);
                    density.push([s.report.step as f64, n as f64 / (4. * half * half)]);
                }
                plot(
                    &mut out,
                    "Recorded events inside the observation region",
                    "step",
                    "events",
                    vec![Series::line("Actual event count", counts)],
                );
                scalar_plot(&mut out, "Empirical event density per chart area", density);
                out.note("Counts come from interacting walkers; no Poisson count law is imposed.");
            }
            if index == 19 {
                let mut scales = vec![];
                let mut dimensions = vec![];
                let all = order
                    .nodes
                    .iter()
                    .map(|n| {
                        [
                            n.position[0],
                            n.position[1],
                            number(p, "speed", 1.) * n.time,
                        ]
                    })
                    .collect::<Vec<_>>();
                let pairs = (all.len() * (all.len().saturating_sub(1)) / 2) as f64;
                for k in 0..12 {
                    let r = 0.05 * 1.5f64.powi(k);
                    let mut n = 0.;
                    for i in 0..all.len() {
                        for j in i + 1..all.len() {
                            if (0..3)
                                .map(|q| (all[i][q] - all[j][q]).powi(2))
                                .sum::<f64>()
                                .sqrt()
                                <= r
                            {
                                n += 1.;
                            }
                        }
                    }
                    if n > 0. && pairs > 0. {
                        scales.push([r, n / pairs]);
                    }
                }
                for w in scales.windows(2) {
                    dimensions.push([w[1][0], (w[1][1] / w[0][1]).ln() / (w[1][0] / w[0][0]).ln()]);
                }
                plot(
                    &mut out,
                    "Recorded spacetime pair counts",
                    "chart radius",
                    "fraction of pairs",
                    vec![Series::line("Measured correlation integral", scales)],
                );
                plot(
                    &mut out,
                    "Scale-dependent dimension estimate",
                    "chart radius",
                    "logarithmic slope",
                    vec![Series::line("Recorded event dimension", dimensions)],
                );
                out.note("This dimension is the correlation slope in the explicit Euclidean chart (x,y,c*t), with c set by the speed control. Saturation, shared ancestry and temporal dependence remain in the measured pair counts.");
            }
        }
        _ => return Err(error("Unknown Fractal Set experiment")),
    }
    Ok(out)
}
