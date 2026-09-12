use super::*;
use crate::physics::thermodynamics::FiniteKernel;
use std::f64::consts::{PI, TAU};
pub(super) fn run(r: &ExperimentRequest, a: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    match r.experiment {
        52 => cuts(r, a),
        53 => entropy_paths(r),
        54 => perimeter(r),
        55 => sampling(r),
        56 => variation(r),
        57 => gibbs(r),
        58 => response(r),
        59 => horizon(r),
        _ => minimum(r),
    }
}
/// Residual-network max flow for nonnegative symmetric graph capacities.
fn min_cut(weights: &[Vec<f64>], source: usize, sink: usize) -> (f64, Vec<bool>) {
    let n = weights.len();
    let mut residual = weights.to_vec();
    let mut flow = 0.;
    loop {
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut q = std::collections::VecDeque::from([source]);
        while let Some(i) = q.pop_front() {
            for j in 0..n {
                if parent[j] == usize::MAX && residual[i][j] > 1e-12 {
                    parent[j] = i;
                    q.push_back(j);
                }
            }
        }
        if parent[sink] == usize::MAX {
            break;
        }
        let mut amount = f64::INFINITY;
        let mut j = sink;
        while j != source {
            let i = parent[j];
            amount = amount.min(residual[i][j]);
            j = i;
        }
        j = sink;
        while j != source {
            let i = parent[j];
            residual[i][j] -= amount;
            residual[j][i] += amount;
            j = i;
        }
        flow += amount;
    }
    let mut seen = vec![false; n];
    seen[source] = true;
    let mut q = vec![source];
    while let Some(i) = q.pop() {
        for j in 0..n {
            if !seen[j] && residual[i][j] > 1e-10 {
                seen[j] = true;
                q.push(j);
            }
        }
    }
    (flow, seen)
}
fn cuts(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    let count = n(r, "population", 24, 8, 64);
    let e = p(r, "epsilon", 0.25, 0.05, 0.7);
    let mut random = rng(seed(r), 0);
    let mut points = (0..count)
        .map(|_| vec![random.uniform::<f64>(), random.uniform::<f64>()])
        .collect::<Vec<_>>();
    let mut provenance = "independent uniform planar points";
    if let Some(a) = archive
        && let Some(step) = a.steps.last()
        && let Some(field) = step.final_population.observations.fields.get("positions")
    {
        let d = field.item_shape().iter().product::<usize>();
        if d >= 2 {
            points = field
                .values()
                .chunks_exact(d)
                .enumerate()
                .filter(|(i, p)| {
                    step.final_population.validity[*i].eligible(a.gas_config.include_truncated)
                        && p[0].is_finite()
                        && p[1].is_finite()
                })
                .take(count)
                .map(|(_, p)| vec![p[0], p[1]])
                .collect();
            provenance = "First requested eligible recorded slots, first-two-coordinate projection; complete planar Gaussian graph";
        }
    }
    let count = points.len();
    if count < 2 {
        return Err(GasError::Configuration(
            "graph requires at least two positions".into(),
        ));
    }
    let mut weights = vec![vec![0.; count + 2]; count + 2];
    for i in 0..count {
        for j in 0..i {
            let d = (points[i][0] - points[j][0]).powi(2) + (points[i][1] - points[j][1]).powi(2);
            weights[i][j] = (-d / (2. * e * e)).exp();
            weights[j][i] = weights[i][j];
        }
    }
    let min = (0..count)
        .min_by(|&i, &j| points[i][0].total_cmp(&points[j][0]))
        .unwrap();
    let max = (0..count)
        .max_by(|&i, &j| points[i][0].total_cmp(&points[j][0]))
        .unwrap();
    let cap = count.pow(2) as f64 + 1.;
    weights[count][min] = cap;
    weights[max][count + 1] = cap;
    let (flow, set) = min_cut(&weights, count, count + 1);
    let cut = (0..count)
        .map(|i| {
            (0..i)
                .filter(|&j| set[i] != set[j])
                .map(|j| weights[i][j])
                .sum::<f64>()
        })
        .sum::<f64>();
    let half = (0..count)
        .map(|i| {
            (0..i)
                .filter(|&j| (points[i][0] < 0.5) != (points[j][0] < 0.5))
                .map(|j| weights[i][j])
                .sum::<f64>()
        })
        .sum::<f64>();
    let mut out = result(r, "Weighted cuts and terminal minimization", provenance);
    metric(&mut out, "Minimum cut capacity", cut, "");
    metric(
        &mut out,
        "Max-flow / min-cut residual",
        (flow - cut).abs(),
        "",
    );
    metric(&mut out, "Fixed half-space cut", half, "");
    plot(
        &mut out,
        "Terminal partition on the point cloud",
        "x",
        "y",
        vec![
            line(
                "Source side",
                points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| set[*i])
                    .map(|(_, p)| [p[0], p[1]])
                    .collect(),
            ),
            line(
                "Sink side",
                points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !set[*i])
                    .map(|(_, p)| [p[0], p[1]])
                    .collect(),
            ),
        ],
    );
    let ancestry = genealogy(r, archive)?;
    metric(
        &mut out,
        "Separating antichain",
        match ancestry["is_separating_antichain"].as_bool() {
            Some(true) => 1.,
            Some(false) => 0.,
            None => f64::NAN,
        },
        "boolean",
    );
    plot(
        &mut out,
        "Episode separator intersects terminal chains",
        "terminal chain index",
        "separator intersections",
        vec![line(
            "Exact ancestry chain count",
            ancestry["chain_intersection_counts"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, x)| [i as f64, x.as_u64().unwrap() as f64])
                .collect(),
        )],
    );
    out.details = json!({"positions":points,"source_terminal":min,"sink_terminal":max,"source_partition":&set[..count],"weights":weights,"gaussian_reference_graph":true,"genealogical_separator":ancestry});
    if let Some(archive) = archive {
        use crate::fractal_set::{EdgeKind, FractalSet};
        let graph = FractalSet::from_archive(archive);
        let capacity = archive
            .steps
            .last()
            .map(|s| s.final_population.len())
            .unwrap_or(0);
        let split = capacity / 2;
        let channels = [
            EdgeKind::IgDistance,
            EdgeKind::IgCloning,
            EdgeKind::HistoricalDistance,
            EdgeKind::HistoricalCloning,
        ];
        let mut rows = Vec::new();
        let mut totals = Vec::new();
        let mut crossings = Vec::new();
        for (index, kind) in channels.iter().enumerate() {
            let edges = graph
                .edges
                .iter()
                .filter(|e| e.kind == *kind)
                .collect::<Vec<_>>();
            let cut = edges
                .iter()
                .filter(|e| {
                    ((e.source.slot as usize) < split) != ((e.target.slot as usize) < split)
                })
                .collect::<Vec<_>>();
            let missing = cut.iter().filter(|e| e.attributes.weight.is_none()).count();
            let known = cut.iter().filter_map(|e| e.attributes.weight).sum::<f64>();
            totals.push([index as f64, edges.len() as f64]);
            crossings.push([index as f64, cut.len() as f64]);
            rows.push(json!({"channel":kind,"observed_edges":edges.len(),"crossing_edges":cut.len(),"recorded_crossing_weight":if missing==0{Some(known)}else{None},"known_partial_crossing_weight":known,"missing_crossing_weights":missing}));
        }
        plot(
            &mut out,
            "Actual selected interaction edges",
            "distance, cloning, historical distance, historical cloning",
            "directed edge count",
            vec![
                line("All observed selected edges", totals),
                line("Edges crossing the fixed slot partition", crossings),
            ],
        );
        out.details["recorded_interaction_cut"] = json!({"partition":"source and target slot indices below floor(capacity/2); fixed slot partition across recorded epochs and steps","split_slot":split,"channels":rows,"unresolved_sources":graph.unresolved_sources.len(),"weight_convention":"Each recorded directed selected edge is counted once. Missing recorded weights stay unavailable; no Gaussian weights or marginal probabilities are substituted."});
    }
    Ok(out)
}
fn entropy_paths(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let perturb = p(r, "perturbation", 0.1, -0.2, 0.2);
    let bloch = [0.4 * perturb, 0., -0.5 + 2. * perturb];
    let radius = (bloch[0] * bloch[0] + bloch[2] * bloch[2]).sqrt();
    let sigma = [0.25, 0.75];
    let omega = [(1. + radius) / 2., (1. - radius) / 2.];
    let s0 = entropy(&sigma);
    let s1 = entropy(&omega);
    let modular = -(0.5 * (1. + bloch[2]) - sigma[0]) * sigma[0].ln()
        - (0.5 * (1. - bloch[2]) - sigma[1]) * sigma[1].ln();
    let relative =
        -s1 - 0.5 * (1. + bloch[2]) * sigma[0].ln() - 0.5 * (1. - bloch[2]) * sigma[1].ln();
    let mut out = result(
        r,
        "Modular entropy and path irreversibility",
        "Noncommuting qubit perturbation and exact stationary three-state cycle path law",
    );
    metric(&mut out, "Quantum relative entropy", relative, "nats");
    metric(
        &mut out,
        "Modular identity residual",
        (relative - modular + s1 - s0).abs(),
        "nats",
    );
    let forward = p(r, "forward", 0.4, 0.05, 0.8);
    let reverse = p(r, "reverse", 0.2, 0., 0.4);
    if forward + reverse >= 0.99 {
        return Err(GasError::Configuration(
            "forward + reverse probabilities must be < 0.99".into(),
        ));
    }
    let stay = 1. - forward - reverse;
    let length = n(r, "path_length", 5, 1, 8);
    let transition = (0..3)
        .flat_map(|i| {
            (0..3).map(move |j| {
                if i == j {
                    stay
                } else if j == (i + 1) % 3 {
                    forward
                } else {
                    reverse
                }
            })
        })
        .collect::<Vec<_>>();
    let kernel = FiniteKernel {
        states: 3,
        transition,
    };
    let report = kernel.thermodynamics()?;
    let mut expected_sigma = 0.;
    let mut ift = 0.;
    let mut support_failure = false;
    let paths = 3_usize.pow(length as u32);
    let mut distribution = vec![];
    for code in 0..paths {
        let mut code = code;
        let mut prob = 1.;
        let mut rev = 1.;
        let mut increment = 0_i32;
        for _ in 0..length {
            match code % 3 {
                0 => {
                    prob *= stay;
                    rev *= stay;
                }
                1 => {
                    prob *= forward;
                    rev *= reverse;
                    increment += 1;
                }
                _ => {
                    prob *= reverse;
                    rev *= forward;
                    increment -= 1;
                }
            }
            code /= 3;
        }
        if prob > 0. {
            if rev == 0. {
                support_failure = true;
            } else {
                let sigma = (prob / rev).ln();
                expected_sigma += prob * sigma;
                ift += rev;
                distribution.push([increment as f64, sigma]);
            }
        }
    }
    let exact = if reverse > 0. {
        length as f64 * (forward - reverse) * (forward / reverse).ln()
    } else {
        f64::INFINITY
    };
    metric(
        &mut out,
        "Path KL",
        if support_failure {
            f64::INFINITY
        } else {
            expected_sigma
        },
        "nats",
    );
    metric(&mut out, "Integral fluctuation expectation", ift, "");
    plot(
        &mut out,
        "Jump cost starts quadratically",
        "field amplitude",
        "jump cost",
        vec![
            line(
                "Exact symmetric jump",
                (0..81)
                    .map(|i| {
                        let t = (i as f64 - 40.) / 40.;
                        [t, (t / 2.).cosh() - 1.]
                    })
                    .collect(),
            ),
            line(
                "Quadratic term",
                (0..81)
                    .map(|i| {
                        let t = (i as f64 - 40.) / 40.;
                        [t, t * t / 8.]
                    })
                    .collect(),
            ),
        ],
    );
    out.details = json!({"sigma":sigma,"omega_bloch_vector":bloch,"omega_eigenvalues":omega,"entropy_difference":s1-s0,"modular_energy_difference":modular,"path":{"length":length,"forward":forward,"reverse":reverse,"expected_analytic_kl":exact.is_finite().then_some(exact),"enumerated_kl":(!support_failure).then_some(expected_sigma),"status":if support_failure{"infinite_support_mismatch"}else{"finite"},"integral_fluctuation_expectation":ift,"report":report,"reverse_protocol":"same cycle kernel; identity involution; uniform endpoints"}});
    Ok(out)
}
fn perimeter_value(e: f64, contrast: f64, grid: usize, d: usize) -> f64 {
    let dx = 1. / grid as f64;
    let mut sum = 0.;
    for i in 0..grid / 2 {
        let x = (i as f64 + 0.5) * dx;
        for j in grid / 2..grid {
            let y = (j as f64 + 0.5) * dx;
            sum += gaussian_periodic(x - y, e)
                * (1. + contrast * (TAU * x).cos())
                * (1. + contrast * (TAU * y).cos());
        }
    }
    sum * dx * dx * ((2. * PI).sqrt() * e).powi(d as i32 - 1)
}
fn perimeter(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let d = n(r, "dimension", 2, 2, 3);
    let contrast = p(r, "density_contrast", 0.3, 0., 0.7);
    let grid = n(r, "resolution", 128, 32, 256);
    let expected =
        (2. * PI).powf((d - 1) as f64 / 2.) * ((1. + contrast).powi(2) + (1. - contrast).powi(2));
    let mut measured = vec![];
    for j in 0..9 {
        let e = 0.24 * 0.8_f64.powi(j);
        measured.push([
            e,
            perimeter_value(e, contrast, grid, d) / e.powi(d as i32 + 1),
        ]);
    }
    let mut out = result(
        r,
        "Gaussian cuts approach weighted perimeter",
        "Periodized Gaussian on the unit torus; half-torus strip with two boundary components",
    );
    metric(&mut out, "Weighted perimeter prediction", expected, "");
    metric(
        &mut out,
        "Finest quadrature",
        measured.last().unwrap()[1],
        "",
    );
    plot(
        &mut out,
        "Correct density-squared and bandwidth normalization",
        "epsilon",
        "cut / epsilon^(d+1)",
        vec![
            line("Pair quadrature", measured),
            line(
                "Weighted boundary integral",
                vec![[0., expected], [0.25, expected]],
            ),
        ],
    );
    out.details = json!({"dimension":d,"density":"1 + contrast cos(2 pi x)","contrast":contrast,"boundary_components":2,"quadrature_grid":grid,"kernel":"periodized unnormalized Gaussian; periodic images -4..4","constant":(2.*PI).powf((d-1) as f64/2.)});
    Ok(out)
}
fn sampling(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let d = n(r, "dimension", 2, 2, 3);
    let count = n(r, "population", 64, 16, 256);
    let reps = n(r, "replicas", 24, 8, 64);
    let ell = p(r, "range_factor", 0.7, 0.3, 1.2);
    let e = ell / (count as f64).powf(1. / d as f64);
    let expectation = count as f64 * (count - 1) as f64 * perimeter_value(e, 0., 256, d);
    let mut cuts = vec![];
    for rep in 0..reps {
        let mut random = rng(seed(r), rep);
        let points = (0..count)
            .map(|_| (0..d).map(|_| random.uniform::<f64>()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut cut = 0.;
        for i in 0..count {
            for j in 0..i {
                if (points[i][0] < 0.5) != (points[j][0] < 0.5) {
                    cut += (0..d)
                        .map(|k| gaussian_periodic(points[i][k] - points[j][k], e))
                        .product::<f64>();
                }
            }
        }
        cuts.push(cut);
    }
    let (mean, var, se) = stats(&cuts);
    let mut out = result(
        r,
        "Finite-population cut scaling",
        "Independent uniform torus populations; complete Gaussian pair graph",
    );
    metric(&mut out, "Replica mean cut", mean, "");
    metric(&mut out, "Exact pair-integral expectation", expectation, "");
    metric(&mut out, "Replica mean SE", se, "");
    metric(
        &mut out,
        "Normalized area estimate",
        mean / (count as f64 * (count - 1) as f64 * e.powi(d as i32 + 1)),
        "",
    );
    plot(
        &mut out,
        "Replicas fluctuate around N(N-1) times the pair integral",
        "replica index",
        "cut capacity",
        vec![
            line(
                "Independent populations",
                cuts.iter()
                    .enumerate()
                    .map(|(i, &x)| [i as f64, x])
                    .collect(),
            ),
            line(
                "Quadrature expectation",
                vec![[0., expectation], [(reps - 1) as f64, expectation]],
            ),
        ],
    );
    out.details = json!({"population":count,"replicas":reps,"dimension":d,"epsilon":e,"finite_pair_factor":count*(count-1),"sample_variance":var,"asymptotic_area_constant":2.*(2.*PI).powf((d-1) as f64/2.),"sampling_unit":"one independent population, not one pair"});
    Ok(out)
}
fn variation(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let radius = p(r, "radius", 0.6, 0.2, 1.2);
    let c = p(r, "density_contrast", 0.3, -0.3, 0.7);
    let tension = p(r, "tension", 1., 0.1, 3.);
    let rho = 1. + c * radius * radius;
    let energy = |r: f64| tension * TAU * r * (1. + c * r * r).powi(2);
    let measured = -derivative(energy, radius, 1e-4) / (TAU * radius);
    let predicted = -tension * (rho * rho / radius + 4. * c * radius * rho);
    let mut out = result(
        r,
        "Boundary and density first variations",
        "Declared weighted circular surface energy with fixed ambient density rho(r)=1+c r²",
    );
    metric(&mut out, "Finite-difference normal pressure", measured, "");
    metric(
        &mut out,
        "Analytic curvature plus density-gradient pressure",
        predicted,
        "",
    );
    metric(
        &mut out,
        "Shape derivative residual",
        (measured - predicted).abs(),
        "",
    );
    plot(
        &mut out,
        "Surface energy under radius variation",
        "radius",
        "surface energy",
        vec![line(
            "Direct energy",
            (0..61)
                .map(|i| {
                    let x = 0.2 + i as f64 / 60.;
                    [x, energy(x)]
                })
                .collect(),
        )],
    );
    let weights = [[0., 0.7, 0.3], [0.7, 0., 0.5], [0.3, 0.5, 0.]];
    let density = [0.8, 1.1, 1.1];
    let eta = [0.2, -0.1, -0.1];
    let cut = |t: f64| {
        weights[0][1] * (density[0] + t * eta[0]) * (density[1] + t * eta[1])
            + weights[0][2] * (density[0] + t * eta[0]) * (density[2] + t * eta[2])
    };
    let exact = weights[0][1] * (eta[0] * density[1] + density[0] * eta[1])
        + weights[0][2] * (eta[0] * density[2] + density[0] * eta[2]);
    out.details = json!({"radius":radius,"ambient_density":rho,"density_normal_derivative":2.*c*radius,"surface_curvature":1./radius,"density_variation":{"partition":[true,false,false],"density":density,"eta":eta,"analytic_derivative":exact,"difference_derivative":derivative(cut,0.,1e-4)}});
    let sigma = [0.25_f64, 0.75];
    let perturb = |t: f64| {
        let z = -0.5 + 2. * t;
        let x = 0.4 * t;
        let radius = (z * z + x * x).sqrt();
        entropy(&[(1. + radius) / 2., (1. - radius) / 2.])
    };
    let step = 1e-4;
    let entropy_derivative = derivative(perturb, 0., step);
    let modular_derivative = (sigma[1] / sigma[0]).ln();
    metric(
        &mut out,
        "Density-matrix first-law residual",
        (entropy_derivative - modular_derivative).abs(),
        "nats",
    );
    out.details["density_matrix_variation"] = json!({"reference_eigenvalues":sigma,"state_bloch_vector":"(0.4 t, 0, -0.5 + 2 t)","entropy_derivative":entropy_derivative,"modular_derivative":modular_derivative,"trace_preserving":true,"noncommuting":true});
    plot(
        &mut out,
        "Entropy first variation for a noncommuting state",
        "state perturbation",
        "entropy difference",
        vec![
            line(
                "Direct spectral entropy",
                (-20..=20)
                    .map(|i| {
                        let t = i as f64 / 1000.;
                        [t, perturb(t) - perturb(0.)]
                    })
                    .collect(),
            ),
            line(
                "Modular first variation",
                (-20..=20)
                    .map(|i| {
                        let t = i as f64 / 1000.;
                        [t, t * modular_derivative]
                    })
                    .collect(),
            ),
        ],
    );
    Ok(out)
}
fn gibbs(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let temp = p(r, "temperature", 1., 0.2, 3.);
    let contrast = p(r, "killing_contrast", 0.1, 0., 0.3);
    let survival = [0.8 - contrast / 2., 0.8 + contrast / 2.];
    let q = vec![
        vec![0.7 * survival[0], 0.3 * survival[0]],
        vec![0.4 * survival[1], 0.6 * survival[1]],
    ];
    let mut pi = vec![0.5, 0.5];
    for _ in 0..256 {
        pi = evolve(&pi, &q);
        let mass = pi.iter().sum::<f64>();
        for x in &mut pi {
            *x /= mass;
        }
    }
    let g1 = (-1. / temp).exp();
    let g = vec![1. / (1. + g1), g1 / (1. + g1)];
    let gq = evolve(&g, &q);
    let survival_g = gq.iter().sum::<f64>();
    let normalized = gq.iter().map(|x| x / survival_g).collect::<Vec<_>>();
    let contraction =
        (q[0][0] * q[1][1] - q[0][1] * q[1][0]).abs() / survival[0].min(survival[1]).powi(2);
    let bound = tv(&g, &normalized) / (1. - contraction);
    let mut observed = g.clone();
    let mut curve = vec![];
    for i in 0..31 {
        curve.push([i as f64, tv(&observed, &pi)]);
        observed = evolve(&observed, &q);
        let mass = observed.iter().sum::<f64>();
        for x in &mut observed {
            *x /= mass;
        }
    }
    let mut out = result(
        r,
        "Test a Gibbs QSD candidate",
        "Exact positive two-state killed kernel; verified global normalized-map contraction",
    );
    metric(&mut out, "Candidate TV error", tv(&g, &pi), "");
    metric(&mut out, "Residual-based TV bound", bound, "");
    metric(&mut out, "Verified contraction bound", contraction, "");
    plot(
        &mut out,
        "Survival-conditioned iteration",
        "step",
        "TV to QSD",
        vec![line("Normalized killed evolution", curve)],
    );
    out.details = json!({"killed_kernel":q,"candidate":g,"qsd":pi,"candidate_survival":survival_g,"normalized_candidate":normalized,"contraction":contraction,"bound":bound,"energy":[0,1],"temperature":temp});
    Ok(out)
}
fn response(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let a = p(r, "transition_a", 0.3, 0.1, 0.8);
    let b = p(r, "transition_b", 0.4, 0.1, 0.8);
    let delta = p(r, "difference_step", 0.001, 0.0001, 0.02);
    let length = n(r, "path_length", 8, 1, 64);
    let kernel = FiniteKernel {
        states: 2,
        transition: vec![1. - a, a, b, 1. - b],
    };
    let kernel_derivative = [-1., 1., 0., 0.];
    let report = kernel.response(&kernel_derivative, &[0., 1.])?;
    let expected = b / (a + b).powi(2);
    let fd = derivative_value(a, b, delta)?;
    let stationary_fisher = b / (a * (a + b).powi(2));
    let transition_fisher = (b / (a + b)) / (a * (1. - a));
    let mut out = result(
        r,
        "Static tilt, Poisson response, and controlled Fisher information",
        "Interior positive two-state Markov kernel with declared derivative dK/da",
    );
    metric(&mut out, "Poisson response", report.poisson_response, "");
    metric(&mut out, "Independent stationary finite difference", fd, "");
    metric(&mut out, "Exact response", expected, "");
    metric(
        &mut out,
        "Stationary Fisher",
        report.stationary_fisher,
        "inverse parameter squared",
    );
    metric(
        &mut out,
        "Transition Fisher",
        report.transition_fisher,
        "inverse parameter squared",
    );
    let decay = 1. - a - b;
    plot(
        &mut out,
        "Delayed response sums to the stationary derivative",
        "lag",
        "response contribution",
        vec![line(
            "Kernel derivative propagated forward",
            (0..32)
                .map(|i| [i as f64, b / (a + b) * decay.powi(i)])
                .collect(),
        )],
    );
    let pi = kernel.stationary()?;
    let tilt = |h: f64| pi[1] * h.exp() / (pi[0] + pi[1] * h.exp());
    let tilt_fd = derivative(tilt, 0., delta);
    out.details = json!({"kernel":kernel,"kernel_derivative":kernel_derivative,"report":report,"exact_response":expected,"exact_stationary_fisher":stationary_fisher,"exact_transition_fisher":transition_fisher,"path_fisher_with_parameter_dependent_stationary_initial_law":stationary_fisher+length as f64*transition_fisher,"path_length":length,"static_exponential_tilt":{"susceptibility_difference":tilt_fd,"covariance_prediction":pi[0]*pi[1]},"assumptions":"fixed two-state common support; positive a,b and self transitions; differentiable normalized rows; unique mixing invariant law"});
    Ok(out)
}
fn derivative_value(a: f64, b: f64, h: f64) -> Result<f64> {
    let evaluate = |a| {
        FiniteKernel {
            states: 2,
            transition: vec![1. - a, a, b, 1. - b],
        }
        .stationary()
    };
    Ok((evaluate(a + h)?[1] - evaluate(a - h)?[1]) / (2. * h))
}
fn horizon(r: &ExperimentRequest) -> Result<ExperimentResult> {
    let lambda = p(r, "lambda", -0.5, -2., -0.1);
    let d = n(r, "dimension", 3, 2, 3) as f64;
    let kappa = p(r, "surface_gravity", 1., 0.2, 3.);
    let ratio = p(r, "period_ratio", 1., 0.5, 1.5);
    let radius = (-d * (d - 1.) / (2. * lambda)).sqrt();
    let ric = -d / (radius * radius);
    let scalar = (d + 1.) * ric;
    let residual = ric - scalar / 2. + lambda;
    let period = TAU / kappa * ratio;
    let mut out = result(
        r,
        "Euclidean horizon periodicity and AdS curvature",
        "Declared regular near-horizon plane and constant-curvature AdS Einstein reference",
    );
    metric(&mut out, "Cone angle", kappa * period, "radians");
    metric(
        &mut out,
        "Smoothness defect",
        kappa * period - TAU,
        "radians",
    );
    metric(&mut out, "AdS radius", radius, "length");
    metric(
        &mut out,
        "Einstein residual",
        residual,
        "inverse length squared",
    );
    plot(
        &mut out,
        "Circumference distinguishes a cone from a smooth plane",
        "radius",
        "circumference",
        vec![
            line(
                "Specified Euclidean period",
                (0..41)
                    .map(|i| {
                        let x = i as f64 / 40.;
                        [x, kappa * period * x]
                    })
                    .collect(),
            ),
            line("Smooth plane", vec![[0., 0.], [1., TAU]]),
        ],
    );
    out.details = json!({"dimension":d,"lambda":lambda,"sectional_curvature":-1./radius.powi(2),"ricci_metric_coefficient":ric,"scalar_curvature":scalar,"period":period,"temperature_if_quantum_period_identified":kappa/TAU,"physical_temperature_identification":"requires declared thermal state and normalized time"});
    Ok(out)
}
fn minimum(r: &ExperimentRequest) -> Result<ExperimentResult> {
    // A rectangular nearest-neighbor capacity network supplies an independently
    // checkable weighted vertical interface, with free interior vertices.
    let m = n(r, "resolution", 16, 8, 32);
    let contrast = p(r, "density_contrast", 0.4, 0., 0.8);
    let count = m * m;
    let source = count;
    let sink = count + 1;
    let mut weights = vec![vec![0.; count + 2]; count + 2];
    let density = |x: f64| 1. - contrast * (-((x - 0.5) / 0.15).powi(2)).exp();
    for y in 0..m {
        for x in 0..m {
            let i = y * m + x;
            if x + 1 < m {
                let w = density((x as f64 + 0.5) / (m - 1) as f64).powi(2) / (m - 1) as f64;
                weights[i][i + 1] = w;
                weights[i + 1][i] = w;
            }
            if y + 1 < m {
                let w = 1. / (m - 1) as f64;
                weights[i][i + m] = w;
                weights[i + m][i] = w;
            }
            if x == 0 {
                weights[source][i] = 100.;
            }
            if x == m - 1 {
                weights[i][sink] = 100.;
            }
        }
    }
    let (flow, set) = min_cut(&weights, source, sink);
    let exact_grid = (0..m - 1)
        .map(|x| m as f64 / (m - 1) as f64 * density((x as f64 + 0.5) / (m - 1) as f64).powi(2))
        .fold(f64::INFINITY, f64::min);
    let mut cut_edges = vec![];
    for y in 0..m {
        for x in 0..m - 1 {
            let i = y * m + x;
            if set[i] != set[i + 1] {
                cut_edges.push([(x as f64 + 0.5) / (m - 1) as f64, y as f64 / (m - 1) as f64]);
            }
        }
    }
    let mut out = result(
        r,
        "A cut minimizer approaches least weighted area",
        "Local grid capacity approximation; declared terminal constraints and density valley",
    );
    metric(&mut out, "Network minimum", flow, "");
    metric(
        &mut out,
        "Independent straight-interface minimum",
        exact_grid,
        "",
    );
    metric(
        &mut out,
        "Continuum weighted interface",
        (1. - contrast).powi(2),
        "",
    );
    plot(
        &mut out,
        "The minimum interface tracks the density valley",
        "x",
        "y",
        vec![line("Cut interface", cut_edges)],
    );
    out.details = json!({"grid":m,"minimum_cut":flow,"enumerated_vertical_reference":exact_grid,"continuum_reference":(1.-contrast).powi(2),"density_contrast":contrast,"kernel":"local nearest-neighbor finite-volume perimeter; Gaussian perimeter convergence is measured in experiment 54","constraints":"left boundary source and right boundary sink"});
    Ok(out)
}

fn separator_report(parents: &[Option<usize>], terminals: &[usize], selected: &[usize]) -> Value {
    let chosen = selected
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut comparable = vec![];
    let mut cycles = false;
    for &i in selected {
        let mut j = parents[i];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(a) = j {
            if !seen.insert(a) {
                cycles = true;
                break;
            }
            if chosen.contains(&a) {
                comparable.push([a, i]);
            }
            j = parents[a];
        }
    }
    let mut chain_counts = vec![];
    for &terminal in terminals {
        let mut j = Some(terminal);
        let mut hits = 0;
        let mut seen = std::collections::BTreeSet::new();
        while let Some(a) = j {
            if !seen.insert(a) {
                cycles = true;
                break;
            }
            hits += usize::from(chosen.contains(&a));
            j = parents[a];
        }
        chain_counts.push(hits);
    }
    json!({"selected":selected,"terminal_episodes":terminals,"parent_indices":parents,"comparable_selected_pairs":comparable,"chain_intersection_counts":chain_counts,"acyclic":!cycles,"is_antichain":comparable.is_empty()&&!cycles,"intersects_every_terminal_chain_once":chain_counts.iter().all(|&c|c==1)&&!cycles,"is_separating_antichain":comparable.is_empty()&&!cycles&&chain_counts.iter().all(|&c|c==1)})
}
fn genealogy(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> Result<Value> {
    let mode = r.text("separator", "roots");
    if let Some(archive) = archive {
        let graph = crate::fractal_set::FractalSet::from_archive(archive);
        let key = |e: crate::fractal_set::EventRef| (e.epoch, e.slot, e.generation);
        let keys = graph
            .nodes
            .iter()
            .map(|e| key(*e))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if keys.len() > 8192 {
            return Err(GasError::Configuration(
                "episode separator budget is 8192 episodes".into(),
            ));
        }
        let indices = keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, i))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut parents = vec![None; keys.len()];
        for e in graph
            .edges
            .iter()
            .filter(|e| e.kind == crate::fractal_set::EdgeKind::Ancestry)
        {
            let a = indices[&key(e.source)];
            let b = indices[&key(e.target)];
            if a != b {
                if parents[b].is_some_and(|p| p != a) {
                    return Err(GasError::Topology(
                        "episode has more than one parent".into(),
                    ));
                }
                parents[b] = Some(a);
            }
        }
        let terminals = archive
            .steps
            .last()
            .map(|s| {
                s.final_population
                    .generations
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, &generation)| {
                        indices.get(&(s.epoch, slot as u32, generation)).copied()
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let selected = if mode == "terminal_episodes" {
            terminals.clone()
        } else {
            parents
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.is_none().then_some(i))
                .collect()
        };
        let mut report = separator_report(&parents, &terminals, &selected);
        report["episode_keys_epoch_slot_generation"] = json!(keys);
        report["unresolved_sources"] = json!(graph.unresolved_sources.len());
        if !graph.unresolved_sources.is_empty() {
            report["is_separating_antichain_in_observed_subgraph"] =
                report["is_separating_antichain"].clone();
            report["is_separating_antichain"] = Value::Null;
        }
        report["coverage_status"] = json!(if graph.unresolved_sources.is_empty() {
            "complete_recorded_forest"
        } else {
            "partial_recorded_forest"
        });
        report["source"] =
            json!("recorded accepted-clone ancestry; persistence collapses within each episode");
        return Ok(report);
    }
    let parents = [None, Some(0), Some(0), Some(1), Some(2)];
    let terminals = [0, 3, 4];
    let selected = if mode == "terminal_episodes" {
        terminals.to_vec()
    } else {
        vec![0]
    };
    let mut report = separator_report(&parents, &terminals, &selected);
    report["source"] = json!("five-episode forest fixture with a still-living ancestor");
    Ok(report)
}
