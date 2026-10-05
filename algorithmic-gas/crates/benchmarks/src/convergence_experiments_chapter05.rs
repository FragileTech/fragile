//! Native kinetic experiments with reusable force, stage and noise archives.
#[path = "convergence_experiments_chapter05_dimension.rs"]
pub mod dimension;
use crate::{
    Benchmark, BenchmarkModel,
    convergence_experiments::{ArchiveStore, ExperimentConfig},
    convergence_kinetic::ConfiningLandscape,
    convergence_stored_chapter05::quadratic_cap_constants,
};
use algorithmic_gas::{
    AlgorithmicGas, GasBuilder, GasConfig, GasError, ObservationBatch, Population, Result,
    TensorBatch,
    boundary::{BoundaryPolicy, BoxDomain},
    noise::InnovationShift,
    random::{RandomStream, Stream},
    tracking::RecordingConfig,
};
use serde_json::{Value, json};

type Matrix = [[f64; 2]; 2];
fn mul(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..2).map(|k| a[i][k] * b[k][j]).sum()))
}
fn transpose(a: Matrix) -> Matrix {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}
fn vector(a: Matrix, z: [f64; 2]) -> [f64; 2] {
    [
        a[0][0] * z[0] + a[0][1] * z[1],
        a[1][0] * z[0] + a[1][1] * z[1],
    ]
}
fn add(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}

/// Exact stable quadratic Langevin transition and covariance, excluding caps,
/// killing and the separately configured final position increment.
pub fn exact_quadratic_transition(
    t: f64,
    gamma: f64,
    curvature: f64,
    diffusion: f64,
) -> Result<(Matrix, Matrix)> {
    if ![t, gamma, curvature, diffusion]
        .iter()
        .all(|x| x.is_finite())
        || t < 0.
        || gamma <= 0.
        || curvature <= 0.
        || diffusion < 0.
    {
        return Err(GasError::Configuration(
            "Exact quadratic transition requires t>=0,gamma>0,curvature>0,diffusion>=0".into(),
        ));
    }
    let discriminant = gamma * gamma / 4. - curvature;
    let (cc, ss) = if discriminant.abs() < 1e-14 {
        (1., t)
    } else if discriminant > 0. {
        let w = discriminant.sqrt();
        ((w * t).cosh(), (w * t).sinh() / w)
    } else {
        let w = (-discriminant).sqrt();
        ((w * t).cos(), (w * t).sin() / w)
    };
    let decay = (-gamma * t / 2.).exp();
    let e = [
        [decay * (cc + gamma * ss / 2.), decay * ss],
        [-decay * curvature * ss, decay * (cc - gamma * ss / 2.)],
    ];
    let stationary = [
        [diffusion * diffusion / (2. * gamma * curvature), 0.],
        [0., diffusion * diffusion / (2. * gamma)],
    ];
    let propagated = mul(mul(e, stationary), transpose(e));
    let covariance =
        std::array::from_fn(|i| std::array::from_fn(|j| stationary[i][j] - propagated[i][j]));
    Ok((e, covariance))
}

fn baoab_transition(h: f64, gamma: f64, curvature: f64) -> (Matrix, Matrix) {
    let c = h / 2.;
    let a = (-gamma * h).exp();
    let k = 1. - c * c * curvature;
    let m = [
        [1. - c * c * (1. + a) * curvature, c * (1. + a)],
        [
            -c * (1. + a) * curvature * k,
            a - c * c * (1. + a) * curvature,
        ],
    ];
    let q2 = -(-2. * gamma * h).exp_m1() / (2. * gamma);
    (m, [[q2 * c * c, q2 * c * k], [q2 * c * k, q2 * k * k]])
}

fn spectral_norm(a: Matrix) -> f64 {
    let aa = mul(transpose(a), a);
    let tr = aa[0][0] + aa[1][1];
    let det = aa[0][0] * aa[1][1] - aa[0][1] * aa[1][0];
    (0.5 * (tr + (tr * tr - 4. * det).max(0.).sqrt())).sqrt()
}
fn ou_exact_cross(h: f64, gamma: f64, curvature: f64) -> Result<[f64; 2]> {
    let (e, _) = exact_quadratic_transition(h, gamma, curvature, 1.)?;
    let decay = (-gamma * h).exp();
    let rhs = [decay * e[0][1], decay * e[1][1] - 1.];
    let det = 2. * gamma * gamma + curvature;
    Ok([
        (-2. * gamma * rhs[0] - rhs[1]) / det,
        (curvature * rhs[0] - gamma * rhs[1]) / det,
    ])
}
fn native_force_perturbation_bound(h: f64, steps: usize, force_bound: f64) -> f64 {
    let (a, _) = baoab_transition(h, 1., 1.);
    let c = h / 2.;
    let decay = (-h).exp();
    let defect = c
        * force_bound
        * ((c * c * (1. + decay).powi(2) + (decay - c * c * (1. + decay)).powi(2)).sqrt() + 1.);
    let norm = spectral_norm(a);
    let mut result = 0.;
    for _ in 0..steps {
        result = norm * result + defect;
    }
    result
}

/// Joint exact-SDE increment and exact OU kick on the same Brownian interval.
fn joint_increment(
    h: f64,
    gamma: f64,
    curvature: f64,
    seed: u64,
    address: u64,
) -> Result<([f64; 2], f64, [f64; 3])> {
    let mut rng = RandomStream::new(seed, address, Stream::MeanFieldReference, 0, 905);
    let z = [
        rng.gaussian::<f64>(),
        rng.gaussian::<f64>(),
        rng.gaussian::<f64>(),
    ];
    let (exact, ou) = joint_increment_from_gaussian(h, gamma, curvature, z)?;
    Ok((exact, ou, z))
}

fn joint_increment_from_gaussian(
    h: f64,
    gamma: f64,
    curvature: f64,
    z: [f64; 3],
) -> Result<([f64; 2], f64)> {
    let (_, cov) = exact_quadratic_transition(h, gamma, curvature, 1.)?;
    let ou_var = -(-2. * gamma * h).exp_m1() / (2. * gamma);
    // Integral exp((A-gamma I)u)e_v du = (A-gamma I)^-1(exp((A-gamma I)h)-I)e_v.
    let cross = ou_exact_cross(h, gamma, curvature)?;
    let covariance = [
        [ou_var, cross[0], cross[1]],
        [cross[0], cov[0][0], cov[0][1]],
        [cross[1], cov[1][0], cov[1][1]],
    ];
    let mut l = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let residual = covariance[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if residual < -2e-13 {
                    return Err(GasError::Numerical(format!(
                        "Joint Brownian covariance not positive: {residual}"
                    )));
                }
                l[i][j] = residual.max(0.).sqrt();
            } else {
                l[i][j] = if l[j][j] > 1e-15 {
                    residual / l[j][j]
                } else {
                    0.
                };
            }
        }
    }
    let draw: [f64; 3] = std::array::from_fn(|i| (0..=i).map(|j| l[i][j] * z[j]).sum());
    Ok(([draw[1], draw[2]], draw[0]))
}

fn population(n: usize, d: usize, shift: [f64; 2]) -> Result<Population<f64>> {
    let mut x = Vec::new();
    let mut v = Vec::new();
    for i in 0..n {
        for j in 0..d {
            x.push(
                (0.8 * (2. * ((i + 3 * j) % n) as f64 / (n.max(2) - 1) as f64 - 1.) + shift[0])
                    / (d as f64).sqrt(),
            );
            v.push(
                (0.2 * (2. * ((i + j) % n) as f64 / (n.max(2) - 1) as f64 - 1.) + shift[1])
                    / (d as f64).sqrt(),
            );
        }
    }
    let mut obs = ObservationBatch::positions(TensorBatch::vectors(n, d, x)?);
    obs.fields
        .insert("velocities".into(), TensorBatch::vectors(n, d, v)?);
    Population::new(obs)
}
async fn engine(
    p: Population<f64>,
    seed: u64,
    h: f64,
    cap: Option<f64>,
    position_noise: f64,
    landscape: ConfiningLandscape,
    shifts: Vec<InnovationShift>,
) -> Result<AlgorithmicGas<f64>> {
    let d = p.observations.field("positions")?.width();
    let mut cfg = GasConfig::euclidean(d, h)?;
    cfg.seed = seed;
    cfg.boundary = BoundaryPolicy::Unbounded;
    cfg.fitness.reward_exponent = 0.;
    cfg.fitness.diversity_exponent = 0.;
    cfg.clone_transform.jitter_amplitude = 0.;
    cfg.kinetic.velocity_cap = cap;
    cfg.kinetic.position_diffusion = position_noise;
    cfg.qft.innovation_shifts = shifts;
    build_engine(p, cfg, landscape).await
}
async fn build_engine(
    p: Population<f64>,
    cfg: GasConfig,
    landscape: ConfiningLandscape,
) -> Result<AlgorithmicGas<f64>> {
    let model = BenchmarkModel {
        benchmark: Benchmark::Quadratic,
        field: "positions".into(),
        direction: cfg.fitness.direction,
    };
    let mut gas = GasBuilder::new(p, model)
        .gradient(landscape)
        .config(cfg)
        .build()
        .await?;
    gas.start_recording(RecordingConfig {
        max_steps: 128,
        max_bytes: 128 * 1024 * 1024,
        graph: false,
    })?;
    Ok(gas)
}

async fn terminal_status_cases(
    config: &ExperimentConfig,
    store: &mut ArchiveStore,
    comparisons: &mut Vec<Value>,
) -> Result<Value> {
    let dimensions = if config.compact { vec![1] } else { vec![1, 2] };
    let mut cases = vec![];
    for d in dimensions {
        let mut residuals = vec![];
        let mut probabilities = vec![];
        let mut bounds = vec![];
        let mut archives = vec![];
        for rep in 0..config.samples {
            let seed = config
                .seed
                .wrapping_add(58_000_000 + d as u64 * 10_000 + rep as u64);
            let mut p = population(4, d, [0., 0.])?;
            let mut right = p.clone();
            let mut x = vec![0.; 4 * d];
            let mut y = x.clone();
            for i in 0..4 {
                x[i * d] = 1.993;
                y[i * d] = 1.998;
            }
            p.observations
                .fields
                .insert("positions".into(), TensorBatch::vectors(4, d, x)?);
            right
                .observations
                .fields
                .insert("positions".into(), TensorBatch::vectors(4, d, y)?);
            p.observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(4, d, vec![0.; 4 * d])?,
            );
            right.observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(4, d, vec![0.; 4 * d])?,
            );
            let mut cfg = GasConfig::euclidean(d, 0.04)?;
            cfg.seed = seed;
            cfg.fitness.reward_exponent = 0.;
            cfg.fitness.diversity_exponent = 0.;
            cfg.clone_transform.jitter_amplitude = 0.;
            cfg.boundary = BoundaryPolicy::AbsorbingBox {
                field: "positions".into(),
                domain: BoxDomain {
                    lower: vec![-2.; d],
                    upper: vec![2.; d],
                },
            };
            let mut left = build_engine(p, cfg.clone(), ConfiningLandscape::quadratic(d)).await?;
            let mut right = build_engine(right, cfg, ConfiningLandscape::quadratic(d)).await?;
            left.step().await?;
            right.step().await?;
            let discrepancy = left
                .population()
                .validity
                .iter()
                .zip(&right.population().validity)
                .filter(|(l, r)| l.eligible(false) != r.eligible(false))
                .count() as f64
                / 4.;
            let a = left.stop_recording().unwrap();
            let b = right.stop_recording().unwrap();
            let pre = |archive: &algorithmic_gas::RunArchive<f64>| -> Result<Vec<f64>> {
                archive.steps[0]
                    .stages
                    .iter()
                    .find(|s| s.stage == "A2")
                    .and_then(|s| s.fields.get("positions"))
                    .map(|f| f.values.clone())
                    .ok_or_else(|| GasError::MissingField("pre-position-noise A2".into()))
            };
            let x = pre(&a)?;
            let y = pre(&b)?;
            let s = 0.1 * 0.04f64.sqrt();
            let bound = (0..4)
                .map(|i| {
                    (2. * x[i * d..(i + 1) * d]
                        .iter()
                        .zip(&y[i * d..(i + 1) * d])
                        .map(|(x, y)| (x - y).abs())
                        .sum::<f64>()
                        / (s * (2. * std::f64::consts::PI).sqrt()))
                    .min(1.)
                })
                .sum::<f64>()
                / 4.;
            probabilities.push(discrepancy);
            bounds.push(bound);
            residuals.push(discrepancy - bound);
            archives.push(store.save_archive(&format!("chapter05/status/d{d}/rep{rep}/left"), &a)?);
            archives
                .push(store.save_archive(&format!("chapter05/status/d{d}/rep{rep}/right"), &b)?);
        }
        let (m, se) = estimate(&residuals);
        comparisons.push(check(format!("terminal_status/d{d}/mismatch_probability"),"lem-kinetic-terminal-status-coupling",
            Some(r"\Pr\!\left(\mathbf1_{\mathcal D}(x+s\zeta)
\ne\mathbf1_{\mathcal D}(y+s\zeta)\right)
\le \min\!\left\{1,\frac{2\|x-y\|_1}{s\sqrt{2\pi}}\right\}.
                                                               \tag{5.K4}"),m,0.,se,"upper",json!({"N":4,"d":d,"h":0.04,"position_noise":0.1,"s":0.02,"box":[-2.,2.],"scope":"Actual terminal alive/dead mismatch event under common final Gaussian diffusion, measured from native marks. Conditional positional bound computed at recorded A2 output, before final position noise; both inputs alive. Independent complete seeds supply SE."})));
        let path = store.save_json(
            &format!("chapter05/status/d{d}/plan"),
            &json!({"probabilities":probabilities,"conditional_bounds":bounds,"archives":archives}),
        )?;
        cases.push(json!({"d":d,"mean_mismatch":estimate(&probabilities).0,"mean_bound":estimate(&bounds).0,"raw_plan":path}));
    }
    Ok(json!(cases))
}

fn coupling_cost(a: &Population<f64>, b: &Population<f64>, k: f64) -> Result<f64> {
    let ax = a.observations.field("positions")?;
    let av = a.observations.field("velocities")?;
    let bx = b.observations.field("positions")?;
    let bv = b.observations.field("velocities")?;
    Ok(ax
        .values()
        .iter()
        .zip(bx.values())
        .map(|(a, b)| k * (a - b).powi(2))
        .sum::<f64>()
        / a.len() as f64
        + av.values()
            .iter()
            .zip(bv.values())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / a.len() as f64)
}

/// Optimal equal-mass empirical Q transport, invariant under independent
/// permutations of both swarm representatives.
pub fn optimal_qcost_points(left: &[Vec<f64>], right: &[Vec<f64>], k: f64) -> Result<f64> {
    let n = left.len();
    if n == 0
        || n != right.len()
        || n > 64
        || !k.is_finite()
        || k <= 0.
        || !left[0].len().is_multiple_of(2)
        || left[0].is_empty()
    {
        return Err(GasError::Configuration(
            "Q transport needs equal nonempty populations <=64 and positive metric".into(),
        ));
    }
    let width = left[0].len();
    let d = width / 2;
    if left
        .iter()
        .chain(right)
        .any(|p| p.len() != width || p.iter().any(|x| !x.is_finite()))
    {
        return Err(GasError::Shape("Physical Q transport dimensions".into()));
    }
    let cost: Vec<Vec<f64>> = left
        .iter()
        .map(|l| {
            right
                .iter()
                .map(|r| {
                    l.iter()
                        .zip(r)
                        .enumerate()
                        .map(|(j, (a, b))| {
                            if j < d {
                                k * (a - b).powi(2)
                            } else {
                                (a - b).powi(2)
                            }
                        })
                        .sum()
                })
                .collect()
        })
        .collect();
    let mut u = vec![0.; n + 1];
    let mut v = vec![0.; n + 1];
    let mut assignment = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        assignment[0] = i;
        let mut column = 0;
        let mut minimum = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[column] = true;
            let row = assignment[column];
            let mut delta = f64::INFINITY;
            let mut next = 0;
            for j in 1..=n {
                if !used[j] {
                    let residual = cost[row - 1][j - 1] - u[row] - v[j];
                    if residual < minimum[j] {
                        minimum[j] = residual;
                        way[j] = column;
                    }
                    if minimum[j] < delta {
                        delta = minimum[j];
                        next = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[assignment[j]] += delta;
                    v[j] -= delta;
                } else {
                    minimum[j] -= delta;
                }
            }
            column = next;
            if assignment[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            assignment[column] = assignment[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    Ok((1..=n).map(|j| cost[assignment[j] - 1][j - 1]).sum::<f64>() / n as f64)
}
fn optimal_qcost(a: &Population<f64>, b: &Population<f64>, k: f64) -> Result<f64> {
    let points = |p: &Population<f64>| -> Result<Vec<Vec<f64>>> {
        let x = p.observations.field("positions")?;
        let v = p.observations.field("velocities")?;
        (0..p.len())
            .map(|i| Ok([x.row(i)?.to_vec(), v.row(i)?.to_vec()].concat()))
            .collect()
    };
    optimal_qcost_points(&points(a)?, &points(b)?, k)
}
fn estimate(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / ((values.len() - 1) * values.len()) as f64)
        .sqrt();
    (mean, se)
}

/// Mean and independent-noise uncertainty for the three fixed input directions.
///
/// Replicate `i` belongs to stratum `i % 3`. The experiment fixes these
/// directions rather than sampling them, so between-direction mean variation
/// does not contribute to the variance of the reported design-weighted mean.
pub fn estimate_direction_strata(values: &[f64]) -> Result<(f64, f64)> {
    if values.len() < 6 || values.iter().any(|v| !v.is_finite()) {
        return Err(GasError::Configuration(
            "Direction-stratified uncertainty requires finite values and two replicates per direction"
                .into(),
        ));
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut variance = 0.;
    for direction in 0..3 {
        let stratum: Vec<f64> = values.iter().skip(direction).step_by(3).copied().collect();
        let stratum_mean = stratum.iter().sum::<f64>() / stratum.len() as f64;
        let sample_variance = stratum
            .iter()
            .map(|v| (v - stratum_mean).powi(2))
            .sum::<f64>()
            / (stratum.len() - 1) as f64;
        variance += stratum.len() as f64 * sample_variance;
    }
    Ok((mean, variance.sqrt() / values.len() as f64))
}
#[allow(clippy::too_many_arguments)]
fn check(
    id: String,
    label: &str,
    formula: Option<&str>,
    observed: f64,
    bound: f64,
    se: f64,
    relation: &str,
    hypotheses: Value,
) -> Value {
    let residual = observed - bound;
    let tol = 2e-10 * (1. + observed.abs() + bound.abs());
    let passed = if relation == "equal" {
        residual.abs() <= 6. * se + tol
    } else {
        residual <= 6. * se + tol
    };
    json!({"id":id,"source_labels":[label],"source_formula":formula,"observed":observed,"bound":bound,"standard_error":se,"residual":residual,"relation":relation,"passed":passed,"status":if passed{"not_rejected"}else{"rejected"},"hypotheses":hypotheses,"scope":"Native recorded operator experiments; independent complete replicates supply uncertainty; population-normalized physical coupling costs"})
}

async fn capped_cases(
    config: &ExperimentConfig,
    store: &mut ArchiveStore,
    comparisons: &mut Vec<Value>,
) -> Result<Value> {
    let ns = if config.compact {
        vec![4]
    } else {
        vec![4, 16, 64]
    };
    let ds = if config.compact { vec![1] } else { vec![1, 2] };
    let samples = config.samples.max(32);
    let steps = if config.compact {
        config.steps.clamp(1, 8)
    } else {
        config.steps
    };
    let constants = quadratic_cap_constants(0.04, 1., 1., 2.)?;
    let delta = constants["delta"].as_f64().unwrap();
    let eta = constants["eta"].as_f64().unwrap();
    let k = constants["k"].as_f64().unwrap();
    let mut cases = vec![];
    for n in ns {
        for &d in &ds {
            let mut trajectories = vec![Vec::new(); samples];
            let mut representative_coupling_trajectories = vec![Vec::new(); samples];
            let mut archive_paths = vec![];
            let mut cap_residuals = vec![];
            let mut ou_variances = vec![];
            let mut zero_clone = 0usize;
            let mut initial_assignment_residual = 0f64;
            let mut transport_minus_coupling = 0f64;
            for (rep, trajectory) in trajectories.iter_mut().enumerate() {
                let shift = match rep % 3 {
                    0 => [0.5, 0.],
                    1 => [0., 0.5],
                    _ => [0.35, 0.35],
                };
                let seed = config
                    .seed
                    .wrapping_add(50_000_000 + n as u64 * 100_000 + d as u64 * 10_000 + rep as u64);
                let left_input = population(n, d, [0., 0.])?;
                let right_input = population(n, d, shift)?;
                let initial = optimal_qcost(&left_input, &right_input, k)?;
                initial_assignment_residual = initial_assignment_residual
                    .max((initial - coupling_cost(&left_input, &right_input, k)?).abs());
                let mut left = engine(
                    left_input,
                    seed,
                    0.04,
                    Some(2.),
                    0.1,
                    ConfiningLandscape::quadratic(d),
                    vec![],
                )
                .await?;
                let mut right = engine(
                    right_input,
                    seed,
                    0.04,
                    Some(2.),
                    0.1,
                    ConfiningLandscape::quadratic(d),
                    vec![],
                )
                .await?;
                trajectory.push(1.);
                representative_coupling_trajectories[rep].push(1.);
                for t in 1..=steps {
                    let l = left.step().await?;
                    let r = right.step().await?;
                    zero_clone += l.clones + r.clones + l.revivals + r.revivals;
                    let optimal = optimal_qcost(left.population(), right.population(), k)?;
                    let coupling = coupling_cost(left.population(), right.population(), k)?;
                    trajectory.push(optimal / initial);
                    representative_coupling_trajectories[rep].push(coupling / initial);
                    transport_minus_coupling = transport_minus_coupling.max(optimal - coupling);
                    if t == 1 {
                        let lstep = left.recording().unwrap().steps.last().unwrap();
                        let rstep = right.recording().unwrap().steps.last().unwrap();
                        let stage = |arc: &algorithmic_gas::tracking::RecordedStep<f64>,
                                     name: &str,
                                     field: &str|
                         -> Result<Vec<f64>> {
                            arc.stages
                                .iter()
                                .find(|s| s.stage == name)
                                .and_then(|s| s.fields.get(field))
                                .map(|f| f.values.clone())
                                .ok_or_else(|| GasError::MissingField(format!("{name}/{field}")))
                        };
                        let u = stage(lstep, "B2", "velocities")?;
                        let uu = stage(rstep, "B2", "velocities")?;
                        let v = left.population().observations.field("velocities")?;
                        let vv = right.population().observations.field("velocities")?;
                        let raw =
                            u.iter().zip(&uu).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;
                        let capped = v
                            .values()
                            .iter()
                            .zip(vv.values())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            / n as f64;
                        cap_residuals.push(capped - (1. - eta) * raw);
                        let noise = lstep
                            .noise
                            .iter()
                            .find(|z| z.stream == Stream::Kinetic && z.substep == 2)
                            .ok_or_else(|| GasError::MissingField("OU noise archive".into()))?;
                        ou_variances
                            .push(noise.sample.iter().map(|z| z * z).sum::<f64>() / (n * d) as f64);
                    }
                    if t % config.archive_chunk_steps.max(1) == 0 || t == steps {
                        for (side, gas) in [("left", &mut left), ("right", &mut right)] {
                            let archive = gas.stop_recording().unwrap();
                            let path = store.save_archive(
                                &format!("chapter05/cap/n{n}_d{d}/rep{rep}/{side}_through{t}"),
                                &archive,
                            )?;
                            archive_paths.push(path);
                            archive_paths.push(store.save_checkpoint(
                                &format!("chapter05/cap/n{n}_d{d}/rep{rep}/{side}_checkpoint{t}"),
                                &gas.checkpoint(),
                            )?);
                            if t < steps {
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
            let hypotheses = json!({"h":0.04,"gamma":1.,"B":1.,"V":2.,"N":n,"d":d,"constants":constants,
            "frozen_kinetic_extension":"Actual complete native update with constant fitness and zero clone jitter; no accepted clone edges. Full canonical kinetic parameters retained; unbounded boundary isolates physical-coordinate theorem. Shared innovations are an admissible coupling, not intrinsic walker labels.","independent_replicates":samples,"direction_design":"Fixed direction stratum rep % 3; mean weighted by actual stratum counts; noise SE = sqrt(sum_h n_h s_h^2)/m. OU second moments use their iid replicate estimate."});
            comparisons.push(check(
                format!("n{n}_d{d}/no_clone_perturbation"),
                "thm-kinetic-exact-baoab-cap-coupling",
                None,
                zero_clone as f64,
                0.,
                0.,
                "equal",
                hypotheses.clone(),
            ));
            let (m, se) = estimate_direction_strata(&cap_residuals)?;
            comparisons.push(check(
                format!("n{n}_d{d}/initial_optimal_assignment"),
                "thm-kinetic-exact-baoab-cap-coupling",
                None,
                initial_assignment_residual,
                0.,
                0.,
                "equal",
                hypotheses.clone(),
            ));
            comparisons.push(check(
                format!("n{n}_d{d}/optimal_output_below_representative_coupling"),
                "thm-kinetic-exact-baoab-cap-coupling",
                None,
                transport_minus_coupling,
                0.,
                0.,
                "upper",
                hypotheses.clone(),
            ));
            comparisons.push(check(
                format!("n{n}_d{d}/cap_eta_dissipation"),
                "thm-kinetic-exact-baoab-cap-coupling",
                Some(
                    r"\mathbb E|C_V(u+kq\xi)-C_V(\widetilde u+kq\xi)|^2
\le(1-\eta)|u-\widetilde u|^2.                         \tag{5.K3}",
                ),
                m,
                0.,
                se,
                "upper",
                hypotheses.clone(),
            ));
            let (m, se) = estimate(&ou_variances);
            comparisons.push(check(
                format!("n{n}_d{d}/OU_standardized_second_moment"),
                "def-baoab-integrator",
                None,
                m,
                1.,
                se,
                "equal",
                hypotheses.clone(),
            ));
            for t in 1..=steps {
                let coupling_values: Vec<f64> = representative_coupling_trajectories
                    .iter()
                    .map(|v| v[t])
                    .collect();
                let (m, se) = estimate_direction_strata(&coupling_values)?;
                comparisons.push(check(
                    format!("n{n}_d{d}/step{t}/admissible_Q_coupling"),
                    "thm-kinetic-exact-baoab-cap-coupling",
                    None,
                    m,
                    (1. - delta).powi(t as i32),
                    se,
                    "upper",
                    hypotheses.clone(),
                ));
                let values: Vec<f64> = trajectories.iter().map(|v| v[t]).collect();
                let (m, se) = estimate_direction_strata(&values)?;
                comparisons.push(check(
                    format!("n{n}_d{d}/step{t}/Q_contraction"),
                    "thm-kinetic-exact-baoab-cap-coupling",
                    if t == 1 {
                        Some(
                            r"\mathbb E\|Z^+-\widetilde Z^+\|_Q^2
\le(1-\delta)\|Z-\widetilde Z\|_Q^2.                 \tag{5.K1}",
                        )
                    } else {
                        None
                    },
                    m,
                    (1. - delta).powi(t as i32),
                    se,
                    "upper",
                    hypotheses.clone(),
                ));
            }
            let path=store.save_json(&format!("chapter05/cap/n{n}_d{d}/coupling_plan"),&json!({"hypotheses":hypotheses,"trajectories":trajectories,"principal_observable":"optimal normalized empirical physical Q transport","representative_coupling_trajectories":representative_coupling_trajectories,"archives":archive_paths,"directions":[[0.5,0.],[0.,0.5],[0.35,0.35]],"cap_eta_residuals":cap_residuals,"OU_second_moments":ou_variances}))?;
            cases.push(json!({"N":n,"d":d,"raw_plan":path,"last_mean_ratio":estimate(&trajectories.iter().map(|v|v[steps]).collect::<Vec<_>>()).0,"samples":samples,"steps":steps}));
        }
    }
    Ok(json!(cases))
}

async fn refinement(
    config: &ExperimentConfig,
    store: &mut ArchiveStore,
    comparisons: &mut Vec<Value>,
) -> Result<Value> {
    let samples = config.samples.max(32);
    let horizon = 0.16;
    let fine_h = horizon / 16.;
    let (efine, _) = exact_quadratic_transition(fine_h, 1., 1., 1.)?;
    let (exact_map, exact_cov) = exact_quadratic_transition(horizon, 1., 1., 1.)?;
    let input = [0.8, 0.4];
    let exact_mean = vector(exact_map, input);
    let mut mean_errors = [Vec::new(), Vec::new(), Vec::new()];
    let mut native_moments = [Vec::new(), Vec::new(), Vec::new()];
    let mut nonquadratic_defects = [Vec::new(), Vec::new(), Vec::new()];
    let mut nonlinear = ConfiningLandscape::quadratic(1);
    nonlinear.name = "quadratic_plus_nonconvex_cosine".into();
    nonlinear.ripple_amplitude = 0.1;
    nonlinear.ripple_frequency = 4.;
    let force_perturbation = nonlinear.ripple_amplitude * nonlinear.ripple_frequency;
    let continuous_lipschitz = spectral_norm([[0., 1.], [-1., -1.]]);
    let continuous_model_defect =
        force_perturbation * (continuous_lipschitz * horizon).exp_m1() / continuous_lipschitz;
    let mut records = vec![];
    let mut paths = vec![];
    let mut exact_moments = vec![];
    for rep in 0..samples {
        let seed = config.seed.wrapping_add(55_000_000 + rep as u64);
        let mut leaf_ou = vec![];
        let mut exact_state = input;
        let mut joint_draws = vec![];
        for leaf in 0..16 {
            let (exact_noise, ou, z) = joint_increment(fine_h, 1., 1., seed, leaf)?;
            leaf_ou.push(ou);
            joint_draws.push(
                json!({"leaf":leaf,"standard_gaussian":z,"exact_noise":exact_noise,"OU_noise":ou}),
            );
            let next = vector(efine, exact_state);
            exact_state = [next[0] + exact_noise[0], next[1] + exact_noise[1]];
        }
        exact_moments.push(exact_state[0].powi(2) + exact_state[1].powi(2));
        let mut outputs = vec![];
        for (level, steps) in [4usize, 8, 16].into_iter().enumerate() {
            let h = horizon / steps as f64;
            let per_leaf = 16 / steps;
            let q = (-(-2. * h).exp_m1() / 2.).sqrt();
            let mut shifts = vec![];
            let mut schedule = vec![];
            for step in 0..steps {
                let increment = (0..per_leaf)
                    .map(|j| {
                        (-fine_h * (per_leaf - 1 - j) as f64).exp() * leaf_ou[step * per_leaf + j]
                    })
                    .sum::<f64>();
                let desired = increment / q;
                let mut rng = RandomStream::new(seed, (step + 1) as u64, Stream::Kinetic, 0, 2);
                let original = rng.gaussian::<f64>();
                shifts.push(InnovationShift {
                    step: (step + 1) as u64,
                    stream: Stream::Kinetic,
                    substep: 2,
                    walker: 0,
                    coordinate: 0,
                    shift: desired - original,
                });
                schedule.push(json!({"step":step+1,"original_standard_gaussian":original,"desired_standard_gaussian":desired,"OU_increment":increment}));
            }
            let mut p = population(1, 1, [0., 0.])?;
            p.observations.fields.insert(
                "positions".into(),
                TensorBatch::vectors(1, 1, vec![input[0]])?,
            );
            p.observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(1, 1, vec![input[1]])?,
            );
            let mut nonlinear_gas = engine(
                p.clone(),
                seed,
                h,
                None,
                0.,
                nonlinear.clone(),
                shifts.clone(),
            )
            .await?;
            let mut gas = engine(
                p,
                seed,
                h,
                None,
                0.,
                ConfiningLandscape::quadratic(1),
                shifts,
            )
            .await?;
            for _ in 0..steps {
                gas.step().await?;
                nonlinear_gas.step().await?;
            }
            let x = gas.population().observations.field("positions")?.values()[0];
            let v = gas.population().observations.field("velocities")?.values()[0];
            let strong_error = (x - exact_state[0]).powi(2) + (v - exact_state[1]).powi(2);
            mean_errors[level].push(strong_error);
            native_moments[level].push(x * x + v * v);
            let path = store.save_archive(
                &format!("chapter05/refinement/rep{rep}/level{level}"),
                &gas.stop_recording().unwrap(),
            )?;
            paths.push(path.clone());
            let nx = nonlinear_gas
                .population()
                .observations
                .field("positions")?
                .values()[0];
            let nv = nonlinear_gas
                .population()
                .observations
                .field("velocities")?
                .values()[0];
            let nonlinear_defect = ((nx - x).powi(2) + (nv - v).powi(2)).sqrt();
            nonquadratic_defects[level].push(nonlinear_defect);
            let nonlinear_path = store.save_archive(
                &format!("chapter05/refinement/nonquadratic_rep{rep}/level{level}"),
                &nonlinear_gas.stop_recording().unwrap(),
            )?;
            paths.push(nonlinear_path.clone());
            outputs.push(json!({"h":h,"steps":steps,"x":x,"v":v,"strong_squared_error":strong_error,"OU_bridge_schedule":schedule,"archive":path,
                "nonquadratic_output":[nx,nv],"nonquadratic_minus_quadratic_norm":nonlinear_defect,"nonquadratic_archive":nonlinear_path,
                "pathwise_nonquadratic_reference_error_bound":continuous_model_defect+native_force_perturbation_bound(h,steps,force_perturbation)+strong_error.sqrt()}));
        }
        records.push(json!({"seed":seed,"input":input,"exact_output":exact_state,"joint_Brownian_leaf_draws":joint_draws,"native_outputs":outputs}));
    }
    let exact_expected =
        exact_mean[0].powi(2) + exact_mean[1].powi(2) + exact_cov[0][0] + exact_cov[1][1];
    let hy = json!({"scope":"Native uncapped BAOAB extension, no final position diffusion, no killing and no selected cloning. Exact quadratic Langevin SDE reference is independently exponentiated and Brownian-coupled; canonical repeated cap is not identified with this SDE.","horizon":horizon,"input":input,"exact_map":exact_map,"exact_covariance":exact_cov,"samples":samples});
    let (m, se) = estimate(&exact_moments);
    comparisons.push(check(
        "refinement/exact_SDE_second_moment".into(),
        "def-kinetic-operator-stratonovich",
        None,
        m,
        exact_expected,
        se,
        "equal",
        hy.clone(),
    ));
    let mut levels = vec![];
    for (level, steps) in [4usize, 8, 16].into_iter().enumerate() {
        let h = horizon / steps as f64;
        let (a, noise) = baoab_transition(h, 1., 1.);
        let mut mean = input;
        let mut cov = [[0.; 2]; 2];
        let (e, _) = exact_quadratic_transition(h, 1., 1., 1.)?;
        let cross_noise = ou_exact_cross(h, 1., 1.)?;
        let g = [h / 2., 1. - h * h / 4.];
        let joint_noise = std::array::from_fn(|i| std::array::from_fn(|j| g[i] * cross_noise[j]));
        let mut cross_cov = [[0.; 2]; 2];
        for _ in 0..steps {
            mean = vector(a, mean);
            cov = add(mul(mul(a, cov), transpose(a)), noise);
            cross_cov = add(mul(mul(a, cross_cov), transpose(e)), joint_noise);
        }
        let predicted = mean[0].powi(2) + mean[1].powi(2) + cov[0][0] + cov[1][1];
        let paired: Vec<f64> = native_moments[level]
            .iter()
            .zip(&exact_moments)
            .map(|(a, b)| a - b)
            .collect();
        let (m, se) = estimate(&paired);
        comparisons.push(check(
            format!("refinement/level{level}/weak_second_moment_error"),
            "prop-weak-error-variance",
            None,
            m,
            predicted - exact_expected,
            se,
            "equal",
            hy.clone(),
        ));
        let exact_strong_mse = (mean[0] - exact_mean[0]).powi(2)
            + (mean[1] - exact_mean[1]).powi(2)
            + cov[0][0]
            + cov[1][1]
            + exact_cov[0][0]
            + exact_cov[1][1]
            - 2. * (cross_cov[0][0] + cross_cov[1][1]);
        let (strong_mean, strong_se) = estimate(&mean_errors[level]);
        comparisons.push(check(
            format!("refinement/level{level}/exact_coupled_strong_MSE"),
            "prop-weak-error-variance",
            None,
            strong_mean,
            exact_strong_mse,
            strong_se,
            "equal",
            hy.clone(),
        ));
        let discrete_defect = native_force_perturbation_bound(h, steps, force_perturbation);
        let maximum_defect = nonquadratic_defects[level]
            .iter()
            .copied()
            .fold(0., f64::max);
        comparisons.push(check(format!("refinement/level{level}/nonquadratic_force_defect"),"prop-weak-error-variance",None,maximum_defect,discrete_defect,0.,"upper",json!({"analytic_global_force_perturbation":force_perturbation,"native_nonlinear_landscape":nonlinear,"L_F":nonlinear.assumptions()?.force_lipschitz,"continuum_model_defect":continuous_model_defect,
            "proof":"Common Brownian noise cancels in quadratic-vs-cosine SDE difference. Gronwall gives ||difference(T)||<=M(exp(||A||T)-1)/||A||. Native BAOAB difference satisfies d_next=A_h d+[c²(1+a),c(a-c²(1+a))]r_B1+[0,c]r_B2 with both |r|<=M; geometric operator-norm sum bounds each raw native defect."})));
        let reference_mse_bound =
            (continuous_model_defect + discrete_defect + exact_strong_mse.max(0.).sqrt()).powi(2);
        levels.push(json!({"h":h,"native_expected_second_moment":predicted,"exact_expected_second_moment":exact_expected,"predicted_weak_bias":predicted-exact_expected,"measured_weak_bias":m,"paired_standard_error":se,"strong_MSE":strong_mean,"strong_MSE_standard_error":strong_se,"exact_strong_MSE":exact_strong_mse,
            "nonquadratic_reference_MSE_upper":reference_mse_bound,"native_nonquadratic_model_defect_maximum":maximum_defect,"native_nonquadratic_model_defect_upper":discrete_defect}));
    }
    let raw = store.save_json(
        "chapter05/refinement/exact_Brownian_coupling",
        &json!({"hypotheses":hy,"records":records,"levels":levels,"archives":paths,"nonquadratic_landscape":nonlinear,
            "nonquadratic_controlled_reference":{"continuum_model_defect":continuous_model_defect,"force_perturbation_bound":force_perturbation,"continuum_generator_matrix_norm":continuous_lipschitz,"scope":"Fine native nonquadratic reference is approximate, with explicit finite global RMS error certificate against its own nonlinear SDE by triangle inequality and Minkowski. Certificate is conservative; it does not identify general weak-error prefactors."}}),
    )?;
    Ok(
        json!({"levels":levels,"raw_plan":raw,"scope":"Fixed-horizon refinement, exact uncapped quadratic reference, common Brownian leaf functionals; no cap/SDE law substitution"}),
    )
}

/// Exact degree-two Gaussian integration of native quadratic refinement errors.
/// These weighted deterministic nodes are not independent Gaussian replicas.
pub async fn run_cubature(store: &mut ArchiveStore) -> Result<Value> {
    let horizon = 0.16;
    let input = [0.8, 0.4];
    let dimension = 48usize;
    let weight = 1. / (2 * dimension) as f64;
    let (leaf_map, _) = exact_quadratic_transition(0.01, 1., 1., 1.)?;
    let (exact_map, exact_cov) = exact_quadratic_transition(horizon, 1., 1., 1.)?;
    let exact_mean = vector(exact_map, input);
    let expected_exact =
        exact_mean.iter().map(|v| v * v).sum::<f64>() + exact_cov[0][0] + exact_cov[1][1];
    let mut exact_integral = 0.;
    let mut native_integrals = [0.; 3];
    let mut strong_integrals = [0.; 3];
    let mut paired_bias_integrals = [0.; 3];
    let mut records = vec![];
    let mut total_clones = 0;
    for node in 0..2 * dimension {
        let mut coordinates = [0.; 48];
        coordinates[node / 2] = if node % 2 == 0 { 1. } else { -1. } * (dimension as f64).sqrt();
        let seed = 60_000_000 + node as u64;
        let mut exact_state = input;
        let mut leaves = vec![];
        let mut leaf_ou = vec![];
        for leaf in 0..16 {
            let z = [
                coordinates[3 * leaf],
                coordinates[3 * leaf + 1],
                coordinates[3 * leaf + 2],
            ];
            let (exact_noise, ou) = joint_increment_from_gaussian(0.01, 1., 1., z)?;
            let next = vector(leaf_map, exact_state);
            exact_state = [next[0] + exact_noise[0], next[1] + exact_noise[1]];
            leaf_ou.push(ou);
            leaves.push(json!({"leaf":leaf,"standard_gaussian_coordinates":z,"exact_noise":exact_noise,"OU_increment":ou}));
        }
        let exact_moment = exact_state.iter().map(|v| v * v).sum::<f64>();
        exact_integral += weight * exact_moment;
        let mut outputs = vec![];
        for (level, steps) in [4usize, 8, 16].into_iter().enumerate() {
            let h = horizon / steps as f64;
            let per_leaf = 16 / steps;
            let q = (-(-2. * h).exp_m1() / 2.).sqrt();
            let mut shifts = vec![];
            let mut schedule = vec![];
            for step in 0..steps {
                let increment = (0..per_leaf)
                    .map(|j| {
                        (-0.01 * (per_leaf - 1 - j) as f64).exp() * leaf_ou[step * per_leaf + j]
                    })
                    .sum::<f64>();
                let desired = increment / q;
                let mut rng = RandomStream::new(seed, (step + 1) as u64, Stream::Kinetic, 0, 2);
                let original = rng.gaussian::<f64>();
                shifts.push(InnovationShift {
                    step: (step + 1) as u64,
                    stream: Stream::Kinetic,
                    substep: 2,
                    walker: 0,
                    coordinate: 0,
                    shift: desired - original,
                });
                schedule.push(json!({"step":step+1,"desired_standard_gaussian":desired,"original_standard_gaussian":original,"OU_increment":increment}));
            }
            let mut population = population(1, 1, [0., 0.])?;
            population.observations.fields.insert(
                "positions".into(),
                TensorBatch::vectors(1, 1, vec![input[0]])?,
            );
            population.observations.fields.insert(
                "velocities".into(),
                TensorBatch::vectors(1, 1, vec![input[1]])?,
            );
            let mut gas = engine(
                population,
                seed,
                h,
                None,
                0.,
                ConfiningLandscape::quadratic(1),
                shifts,
            )
            .await?;
            let input_checkpoint = store.save_checkpoint(
                &format!("cubature/node{node}/level{level}/input"),
                &gas.checkpoint(),
            )?;
            for _ in 0..steps {
                let report = gas.step().await?;
                total_clones += report.clones + report.revivals;
            }
            let output = [
                gas.population().observations.field("positions")?.values()[0],
                gas.population().observations.field("velocities")?.values()[0],
            ];
            let moment = output.iter().map(|v| v * v).sum::<f64>();
            let strong_error =
                (output[0] - exact_state[0]).powi(2) + (output[1] - exact_state[1]).powi(2);
            native_integrals[level] += weight * moment;
            strong_integrals[level] += weight * strong_error;
            paired_bias_integrals[level] += weight * (moment - exact_moment);
            let archive = store.save_archive(
                &format!("cubature/node{node}/level{level}/native"),
                &gas.stop_recording().unwrap(),
            )?;
            let checkpoint = store.save_checkpoint(
                &format!("cubature/node{node}/level{level}/output"),
                &gas.checkpoint(),
            )?;
            outputs.push(json!({"h":h,"steps":steps,"output":output,"moment":moment,"strong_squared_error":strong_error,"OU_bridge_schedule":schedule,"archive":archive,"input_checkpoint":input_checkpoint,"output_checkpoint":checkpoint}));
        }
        records.push(json!({"node":node,"weight":weight,"nonzero_coordinate":node/2,"coordinate_value":coordinates[node/2],"input":input,"leaf_inputs":leaves,"exact_output":exact_state,"native_outputs":outputs}));
    }
    let hypotheses = json!({"scope":"Exact degree-two Gaussian cubature of linear uncapped unit-quadratic native BAOAB and its joint Brownian exact SDE reference. No final position diffusion, killing, or accepted cloning. These 96 deterministic weighted nodes are not independent stochastic replicas.","horizon":horizon,"Gaussian_dimension":dimension,"node_rule":"+/-sqrt(48)e_j with weight 1/96","exactness":"Weights integrate constants, all linear coordinates and all degree-two coordinate products exactly. Native and exact outputs are affine in the same leaf Gaussian vector; squared moments and coupled squared errors therefore have exact Gaussian integrals up to floating-point roundoff."});
    let mut comparisons = vec![
        check(
            "cubature/exact_SDE_second_moment".into(),
            "def-kinetic-operator-stratonovich",
            None,
            exact_integral,
            expected_exact,
            0.,
            "equal",
            hypotheses.clone(),
        ),
        check(
            "cubature/no_accepted_clones".into(),
            "prop-weak-error-variance",
            None,
            total_clones as f64,
            0.,
            0.,
            "equal",
            hypotheses.clone(),
        ),
    ];
    for row in &mut comparisons {
        row["scope"] = hypotheses["scope"].clone();
        row["integration_method"] =
            json!("degree-two Gaussian cubature; zero sampling uncertainty");
    }
    let mut levels = vec![];
    let weak_prefactor = quadratic_weak_prefactor();
    let weak_constant = weak_prefactor["C_weak"].as_f64().unwrap();
    for (level, steps) in [4usize, 8, 16].into_iter().enumerate() {
        let h = horizon / steps as f64;
        let (a, noise) = baoab_transition(h, 1., 1.);
        let (e, _) = exact_quadratic_transition(h, 1., 1., 1.)?;
        let cross_noise = ou_exact_cross(h, 1., 1.)?;
        let g = [h / 2., 1. - h * h / 4.];
        let joint_noise = std::array::from_fn(|i| std::array::from_fn(|j| g[i] * cross_noise[j]));
        let mut mean = input;
        let mut covariance = [[0.; 2]; 2];
        let mut cross_covariance = [[0.; 2]; 2];
        for _ in 0..steps {
            mean = vector(a, mean);
            covariance = add(mul(mul(a, covariance), transpose(a)), noise);
            cross_covariance = add(mul(mul(a, cross_covariance), transpose(e)), joint_noise);
        }
        let native_expected =
            mean.iter().map(|v| v * v).sum::<f64>() + covariance[0][0] + covariance[1][1];
        let bias = native_expected - expected_exact;
        let strong = (mean[0] - exact_mean[0]).powi(2)
            + (mean[1] - exact_mean[1]).powi(2)
            + covariance[0][0]
            + covariance[1][1]
            + exact_cov[0][0]
            + exact_cov[1][1]
            - 2. * (cross_covariance[0][0] + cross_covariance[1][1]);
        for (name, observed, predicted) in [
            (
                "native_second_moment",
                native_integrals[level],
                native_expected,
            ),
            ("weak_bias", paired_bias_integrals[level], bias),
            ("strong_MSE", strong_integrals[level], strong),
        ] {
            let mut row = check(
                format!("cubature/h{h}/{name}"),
                "prop-weak-error-variance",
                None,
                observed,
                predicted,
                0.,
                "equal",
                hypotheses.clone(),
            );
            row["scope"] = hypotheses["scope"].clone();
            row["integration_method"] =
                json!("degree-two Gaussian cubature; zero sampling uncertainty");
            // A tight roundoff check resolves biases much smaller than Monte Carlo SE.
            row["passed"] = json!((observed - predicted).abs() <= 2e-12);
            row["absolute_roundoff_tolerance"] = json!(2e-12);
            row["status"] = json!(if row["passed"] == true {
                "roundoff_consistent"
            } else {
                "rejected"
            });
            comparisons.push(row);
        }
        let mut weak_bound = check(
            format!("cubature/h{h}/analytic_weak_prefactor"),
            "prop-weak-error-variance",
            None,
            paired_bias_integrals[level].abs(),
            weak_constant * h * h,
            0.,
            "upper",
            weak_prefactor.clone(),
        );
        weak_bound["scope"] = json!(
            "Derived N-independent weak-error coefficient for f=|x|²+|v|², unit quadratic uncapped extension, horizon .16, 0<h<=.04 dividing the horizon. Exact deterministic Gaussian cubature integration; no sampling uncertainty."
        );
        comparisons.push(weak_bound);
        levels.push(json!({"h":h,"steps":steps,"cubature_native_second_moment":native_integrals[level],"predicted_native_second_moment":native_expected,"cubature_weak_bias":paired_bias_integrals[level],"predicted_weak_bias":bias,"cubature_strong_MSE":strong_integrals[level],"predicted_strong_MSE":strong,"weak_bias_over_h_squared":paired_bias_integrals[level]/h.powi(2),"strong_MSE_over_h_squared":strong_integrals[level]/h.powi(2)}));
    }
    let ratios:Vec<Value> = (0..2).map(|i|json!({"coarse_h":levels[i]["h"],"fine_h":levels[i+1]["h"],"weak_bias_ratio":paired_bias_integrals[i]/paired_bias_integrals[i+1],"strong_MSE_ratio":strong_integrals[i]/strong_integrals[i+1]})).collect();
    let raw = store.save_json("cubature/full-node-plan", &json!({"hypotheses":hypotheses,"records":records,"levels":levels,"refinement_ratios":ratios,"exact_SDE_transition":exact_map,"exact_SDE_covariance":exact_cov}))?;
    let failed = comparisons.iter().filter(|r| r["passed"] == false).count();
    Ok(
        json!({"chapter":5,"title":"Exact Gaussian cubature of native quadratic refinement","hypotheses":hypotheses,"raw_plan":raw,"levels":levels,"refinement_ratios":ratios,"analytic_weak_prefactor":weak_prefactor,"comparisons":comparisons,"summary":{"nodes":96,"native_steps":2688,"comparisons":comparisons.len(),"comparisons_failed":failed,"sampling_standard_error":0.},"gaps":["Degree-two exactness applies to this linear uncapped quadratic extension. General nonlinear weak-error coefficients and repeated canonical cap/SDE correspondence are not identified."]}),
    )
}

/// Conservative analytic weak coefficient, derived before consulting the data.
pub fn quadratic_weak_prefactor() -> Value {
    let h_max: f64 = 0.04;
    let horizon: f64 = 0.16;
    let split_norm: f64 = 3.;
    let generator_norm = spectral_norm([[0., 1.], [-1., -1.]]);
    let u = ((h_max / 2.).powi(2) + (1. + h_max * h_max / 4.).powi(2)).sqrt();
    let u1 = (0.25 + (h_max / 2.).powi(2)).sqrt();
    let u2 = 0.5;
    let ca = (split_norm.powi(3) * (split_norm * h_max).exp()
        + generator_norm.powi(3) * (generator_norm * h_max).exp())
        / 6.;
    let cq = (4. * u * u
        + 12. * u * u1
        + 6. * (u1 * u1 + u * u2)
        + 6. * h_max * u1 * u2
        + 4. * generator_norm * generator_norm * (2. * generator_norm * h_max).exp())
        / 6.;
    let m2 = (2. * generator_norm * horizon).exp() * (0.8_f64.powi(2) + 0.4_f64.powi(2) + horizon);
    let cw = horizon
        * (2. * split_norm * horizon).exp()
        * (ca * ((split_norm * h_max).exp() + (generator_norm * h_max).exp()) * m2 + 2. * cq);
    json!({"H":h_max,"T":horizon,"S":split_norm,"L":generator_norm,"U":u,"U1":u1,"U2":u2,"C_A":ca,"C_Q":cq,"M2":m2,"C_weak":cw,"input":[0.8,0.4],"observable":"|x|²+|v|²","derivation":"Native/exact transition means and covariances match derivatives 0,1,2 at h=0. Uniform third-derivative remainders give ||A_h-exp(Ah)||<=C_A h³ and ||Q_h-Q_exact(h)||<=C_Q h³. Nuclear-norm second-moment recursion yields |E f_native-E f_exact|<=C_weak h² for 0<h<=H dividing T; normalized averages over any N have the same coefficient."})
}

pub async fn run(config: &ExperimentConfig, store: &mut ArchiveStore) -> Result<Value> {
    if config.samples < 32 || config.steps == 0 {
        return Err(GasError::Configuration(
            "Kinetic experiments need >=32 independent replicas and positive steps".into(),
        ));
    }
    let mut comparisons = vec![];
    let cap = capped_cases(config, store, &mut comparisons).await?;
    let terminal_status = terminal_status_cases(config, store, &mut comparisons).await?;
    let refined = refinement(config, store, &mut comparisons).await?;
    let failed = comparisons.iter().filter(|c| c["passed"] == false).count();
    Ok(
        json!({"chapter":5,"title":"Native kinetic coupling and exact-SDE refinement","source_path":"docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md","comparisons":comparisons,"capped_cases":cap,"terminal_status":terminal_status,"refinement":refined,"summary":{"comparisons":comparisons.len(),"comparisons_failed":failed},"gaps":[{"id":"general_global_weak_error_constants","reason":"Exact quadratic specialization measures and predicts weak biases and strong errors; cosine reference has a rigorous conservative global RMS certificate. General K_integ,K_b,K_W and matching-stability transfer remain analytic obligations."},{"id":"barrier_density_minorization","reason":"Aligned barrier derivatives and quantitative density/minorization certificates are not implied by physical coupling or SDE refinement."}]}),
    )
}
