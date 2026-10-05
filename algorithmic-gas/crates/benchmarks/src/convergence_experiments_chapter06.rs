//! Independent native killed-chain ensembles, durable histories and survivor laws.
//! No resampling, Fleming--Viot particles or storage-label swarm metric is used.
use crate::{
    RunConfig,
    convergence_experiments::{ArchiveStore, ExperimentConfig},
};
use algorithmic_gas::{
    GasError, Population, Result,
    boundary::{BoundaryPolicy, BoxDomain},
    random::{RandomStream, Stream},
    tracking::{RecordedStep, RecordingConfig},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub step: usize,
    pub alive: usize,
    pub phases: Vec<Vec<f64>>,
    pub uniform_alive_sample: Option<Vec<f64>>,
    pub normalized_position_variance: f64,
    pub normalized_velocity_variance: f64,
    pub velocity_mean_energy: f64,
    pub normalized_position_second_moment: f64,
    pub normalized_log_barrier: f64,
    pub alive_safe_fraction: f64,
    pub safe_count: usize,
    pub barrier_count_lower_bound: f64,
}
#[derive(Clone, Debug, Serialize)]
struct Trajectory {
    distribution: usize,
    replicate: usize,
    seed: u64,
    archive_paths: Vec<String>,
    checkpoint_paths: Vec<String>,
    checkpoints: Vec<Snapshot>,
    native_extinction_step: Option<usize>,
    theorem_cemetery_step: Option<usize>,
    death_predictions: Vec<DeathPrediction>,
}
#[derive(Clone, Debug, Serialize)]
struct DeathPrediction {
    step: usize,
    native_absorbed_before: bool,
    theorem_absorbed_before: bool,
    alive_after: usize,
    conditional_all_die: f64,
    conditional_fewer_two_survive: f64,
    gaussian_probability_error: f64,
    prepared_safe_count: usize,
    maximum_safe_failure_bound: Option<f64>,
    conditional_log_extinction_union_bound: Option<f64>,
    gaussian_means: Vec<Vec<f64>>,
    per_row_survival: Vec<f64>,
}
fn error(message: &str) -> GasError {
    GasError::Configuration(message.into())
}

/// Wilson interval describes uncertainty across independent swarm trajectories.
pub fn binomial_interval(events: usize, trials: usize) -> Result<[f64; 2]> {
    if trials == 0 || events > trials {
        return Err(error("binomial interval needs 0<=events<=positive trials"));
    }
    let n = trials as f64;
    let p = events as f64 / n;
    let z = 1.959963984540054;
    let denominator = 1. + z * z / n;
    let center = (p + z * z / (2. * n)) / denominator;
    let radius = z * (p * (1. - p) / n + z * z / (4. * n * n)).sqrt() / denominator;
    Ok([(center - radius).max(0.), (center + radius).min(1.)])
}
/// Own denominator for each initial law and absorbing convention; zero survivors
/// produces no conditional law or moment, rather than a fabricated zero error.
pub fn survivor_mean(values: &[(bool, f64)]) -> Option<f64> {
    let survivors: Vec<f64> = values
        .iter()
        .filter_map(|(alive, x)| alive.then_some(*x))
        .collect();
    (!survivors.is_empty()).then(|| survivors.iter().sum::<f64>() / survivors.len() as f64)
}
/// A numerical rate must agree with its declared discrete coefficient. Passing
/// this arithmetic check never establishes the missing kernel hypotheses.
pub fn law_rate_consistent(rho: f64, h: f64, claimed_rate: f64) -> bool {
    rho > 0.
        && rho < 1.
        && h > 0.
        && claimed_rate.is_finite()
        && (claimed_rate + rho.ln() / h).abs() <= 1e-10 * (1. + claimed_rate.abs())
}
fn normal_cdf(x: f64) -> f64 {
    // Abramowitz--Stegun normal CDF polynomial, absolute error < 1.5e-7.
    let t = 1. / (1. + 0.2316419 * x.abs());
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = poly * (-x * x / 2.).exp() / (2. * std::f64::consts::PI).sqrt();
    if x >= 0. { 1. - tail } else { tail }
}
fn poisson_extinction(survival: &[f64]) -> (f64, f64) {
    let mut zero = 1.;
    let mut one = 0.;
    for &q in survival {
        one = one * (1. - q) + zero * q;
        zero *= 1. - q;
    }
    (zero, (zero + one).min(1.))
}
fn snapshot(
    pop: &Population<f64>,
    step: usize,
    seed: u64,
    half_width: f64,
    safe_margin: f64,
) -> Result<Snapshot> {
    let n = pop.len();
    let x = pop.observations.field("positions")?;
    let v = pop.observations.field("velocities")?;
    let d = x.width();
    let eligible = pop.eligible(false);
    let alive = eligible.iter().filter(|a| **a).count();
    let mut phases = Vec::with_capacity(n);
    let mut mx = vec![0.; d];
    let mut mv = vec![0.; d];
    for (i, &a) in eligible.iter().enumerate() {
        let mut row = vec![if a { 1. } else { 0. }];
        row.extend(x.row(i)?.iter().map(|z| if a { *z } else { 0. }));
        row.extend(v.row(i)?.iter().copied());
        if a {
            for j in 0..d {
                mx[j] += x.row(i)?[j] / alive as f64;
                mv[j] += v.row(i)?[j] / alive as f64;
            }
        }
        phases.push(row);
    }
    let mut xv = 0.;
    let mut vv = 0.;
    let mut x2 = 0.;
    let mut barrier = 0.;
    let mut safe = 0;
    for (i, &a) in eligible.iter().enumerate().filter(|(_, a)| **a) {
        let xr = x.row(i)?;
        let vr = v.row(i)?;
        safe += usize::from(xr.iter().all(|z| z.abs() <= half_width - safe_margin));
        for j in 0..d {
            xv += (xr[j] - mx[j]).powi(2) / n as f64;
            vv += (vr[j] - mv[j]).powi(2) / n as f64;
            x2 += xr[j] * xr[j] / n as f64;
        }
        if a {
            barrier += xr
                .iter()
                .map(|z| -(1. - (z / half_width).powi(2)).max(f64::MIN_POSITIVE).ln())
                .sum::<f64>()
                / n as f64;
        }
    }
    let sample = if alive > 0 {
        let live: Vec<_> = phases.iter().filter(|p| p[0] == 1.).collect();
        let mut rng =
            RandomStream::new(seed ^ 0x8d01e42c1ad, step as u64, Stream::Initialize, 0, 9);
        let at = ((rng.uniform::<f64>() * alive as f64) as usize).min(alive - 1);
        Some(live[at][1..].to_vec())
    } else {
        None
    };
    let threshold = -(1. - ((half_width - safe_margin) / half_width).powi(2)).ln();
    Ok(Snapshot {
        step,
        alive,
        phases,
        uniform_alive_sample: sample,
        normalized_position_variance: xv,
        normalized_velocity_variance: vv,
        velocity_mean_energy: mv.iter().map(|z| z * z).sum(),
        normalized_position_second_moment: x2,
        normalized_log_barrier: barrier,
        alive_safe_fraction: safe as f64 / n as f64,
        safe_count: safe,
        barrier_count_lower_bound: alive as f64 - n as f64 * barrier / threshold,
    })
}
fn prediction(
    step: &RecordedStep<f64>,
    time: usize,
    half_width: f64,
    sd: f64,
    safe_margin: f64,
    theorem_absorbed_before: bool,
) -> Result<DeathPrediction> {
    let stage = step
        .stages
        .iter()
        .find(|s| s.stage == "position_diffusion_before_boundary")
        .ok_or_else(|| error("recorded positional Gaussian stage missing"))?;
    let noise = step
        .field_evaluations
        .iter()
        .find(|s| s.stage == "position_diffusion" && s.field == "executed_noise")
        .ok_or_else(|| error("recorded independent positional Gaussian field missing"))?;
    let x = &stage
        .fields
        .get("positions")
        .ok_or_else(|| error("Gaussian stage positions missing"))?
        .values;
    let n = step.before.len();
    let d = x.len() / n;
    let mut means = Vec::new();
    let mut survival = Vec::new();
    let mut safe = 0;
    for i in 0..n {
        let row: (Vec<f64>, f64) = {
            let mean: Vec<_> = (0..d)
                .map(|j| x[i * d + j] - sd * noise.values[i * d + j])
                .collect();
            let q = mean
                .iter()
                .map(|mu| {
                    (normal_cdf((half_width - mu) / sd) - normal_cdf((-half_width - mu) / sd))
                        .clamp(0., 1.)
                })
                .product();
            (mean, q)
        };
        safe += usize::from(row.0.iter().all(|mu| mu.abs() <= half_width - safe_margin));
        means.push(row.0);
        survival.push(row.1);
    }
    let (all, fewer) = poisson_extinction(&survival);
    let p = (2. * d as f64 * (-safe_margin * safe_margin / (2. * sd * sd)).exp()).min(1.);
    let log_bound =
        (safe >= 2 && p > 0. && p < 1.).then(|| (safe as f64).ln() + (safe - 1) as f64 * p.ln());
    Ok(DeathPrediction {
        step: time,
        native_absorbed_before: false,
        theorem_absorbed_before,
        alive_after: step
            .final_population
            .eligible(false)
            .iter()
            .filter(|a| **a)
            .count(),
        conditional_all_die: all,
        conditional_fewer_two_survive: fewer,
        gaussian_probability_error: 3e-7 * n as f64 * d as f64,
        prepared_safe_count: safe,
        maximum_safe_failure_bound: (safe >= 2).then_some(p),
        conditional_log_extinction_union_bound: log_bound,
        gaussian_means: means,
        per_row_survival: survival,
    })
}

/// Exact uniform assignment for a squared cost matrix. Used at the swarm level,
/// and for equal-cardinality empirical laws. Slots are minimized over permutations.
fn assignment(cost: &[Vec<f64>]) -> (f64, Vec<usize>) {
    let n = cost.len();
    let mut u = vec![0.; n + 1];
    let mut v = vec![0.; n + 1];
    let mut p = vec![0; n + 1];
    let mut way = vec![0; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut partners = vec![0; n];
    for j in 1..=n {
        partners[p[j] - 1] = j - 1;
    }
    let total = partners
        .iter()
        .enumerate()
        .map(|(i, j)| cost[i][*j])
        .sum::<f64>()
        / n as f64;
    (total, partners)
}
/// Dead positions are overwritten by mandatory revival and are discarded. Original
/// velocities remain because the native collision uses revived-slot velocities.
pub fn swarm_cost(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<f64> {
    if left.len() != right.len() || left.is_empty() {
        return Err(error("swarm cost requires equal positive populations"));
    }
    let width = left[0].len();
    if width < 3
        || !(width - 1).is_multiple_of(2)
        || left
            .iter()
            .chain(right)
            .any(|r| r.len() != width || (r[0] != 0. && r[0] != 1.))
    {
        return Err(error("invalid marked phase state shape/status"));
    }
    let cost: Vec<Vec<f64>> = left
        .iter()
        .map(|a| {
            right
                .iter()
                .map(|b| {
                    let status = (a[0] - b[0]).powi(2);
                    let phase = (1..a.len())
                        .map(|j| {
                            let is_position = j <= (a.len() - 1) / 2;
                            let x = if a[0] > 0. || !is_position { a[j] } else { 0. };
                            let y = if b[0] > 0. || !is_position { b[j] } else { 0. };
                            (x - y).powi(2)
                        })
                        .sum::<f64>();
                    status + phase
                })
                .collect()
        })
        .collect();
    Ok(assignment(&cost).0)
}
#[derive(Clone)]
struct Edge {
    to: usize,
    reverse: usize,
    capacity: usize,
    cost: f64,
}
fn edge(graph: &mut [Vec<Edge>], u: usize, v: usize, capacity: usize, cost: f64) {
    let (ur, vr) = (graph[u].len(), graph[v].len());
    graph[u].push(Edge {
        to: v,
        reverse: vr,
        capacity,
        cost,
    });
    graph[v].push(Edge {
        to: u,
        reverse: ur,
        capacity: 0,
        cost: -cost,
    });
}
fn transport_matrix(cost: &[Vec<f64>]) -> Result<(f64, Vec<Value>)> {
    let l = cost.len();
    let r = cost.first().map_or(0, Vec::len);
    if l == 0 || r == 0 {
        return Err(error("empty law transport is undefined"));
    }
    if l == r {
        let (value, partners) = assignment(cost);
        return Ok((
            value,
            partners
                .iter()
                .enumerate()
                .map(|(i, j)| json!({"left":i,"right":j,"mass":1./l as f64}))
                .collect(),
        ));
    }
    let source = l + r;
    let sink = source + 1;
    let vertices = sink + 1;
    let mut graph = vec![vec![]; vertices];
    for i in 0..l {
        edge(&mut graph, source, i, r, 0.);
    }
    for j in 0..r {
        edge(&mut graph, l + j, sink, l, 0.);
    }
    let mut refs = vec![vec![0; r]; l];
    for i in 0..l {
        for j in 0..r {
            refs[i][j] = graph[i].len();
            edge(&mut graph, i, l + j, l * r, cost[i][j]);
        }
    }
    let mut remaining = l * r;
    let mut objective = 0.;
    while remaining > 0 {
        let mut distance = vec![f64::INFINITY; vertices];
        let mut parents = vec![None; vertices];
        distance[source] = 0.;
        for _ in 0..vertices - 1 {
            let mut changed = false;
            for u in 0..vertices {
                if !distance[u].is_finite() {
                    continue;
                }
                for (ei, e) in graph[u].iter().enumerate() {
                    let next = distance[u] + e.cost;
                    if e.capacity > 0 && next < distance[e.to] - 1e-14 * (1. + next.abs()) {
                        distance[e.to] = next;
                        parents[e.to] = Some((u, ei));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if parents[sink].is_none() {
            return Err(error("no law transport augmenting path"));
        }
        let mut amount = remaining;
        let mut at = sink;
        let mut count = 0;
        while at != source {
            let (u, ei) = parents[at].ok_or_else(|| error("broken law transport path"))?;
            amount = amount.min(graph[u][ei].capacity);
            at = u;
            count += 1;
            if count > vertices {
                return Err(error("cyclic law transport path"));
            }
        }
        at = sink;
        while at != source {
            let (u, ei) = parents[at].unwrap();
            let to = graph[u][ei].to;
            let reverse = graph[u][ei].reverse;
            graph[u][ei].capacity -= amount;
            graph[to][reverse].capacity += amount;
            at = u;
        }
        objective += amount as f64 * distance[sink];
        remaining -= amount;
    }
    let mut plan = vec![];
    for i in 0..l {
        for j in 0..r {
            let e = &graph[i][refs[i][j]];
            let flow = graph[e.to][e.reverse].capacity;
            if flow > 0 {
                plan.push(json!({"left":i,"right":j,"mass":flow as f64/(l*r) as f64}));
            }
        }
    }
    Ok((objective / (l * r) as f64, plan))
}
fn law_transport(
    left: &[&Snapshot],
    right: &[&Snapshot],
    scale: f64,
    exact: bool,
) -> Result<Value> {
    if left.is_empty() || right.is_empty() {
        return Ok(
            json!({"available":false,"reason":"one initial law has zero survivors","left_survivors":left.len(),"right_survivors":right.len()}),
        );
    }
    let mut total = 0.;
    let mut blocks = vec![];
    if exact {
        let costs: Vec<Vec<f64>> = left
            .iter()
            .map(|a| {
                right
                    .iter()
                    .map(|b| swarm_cost(&a.phases, &b.phases).unwrap() / scale)
                    .collect()
            })
            .collect();
        let (v, plan) = transport_matrix(&costs)?;
        total = v;
        blocks.push(json!({"left_range":[0,left.len()],"right_range":[0,right.len()],"mass":1.,"local_cost":v,"local_plan":plan}));
    } else {
        // Couple block indices by quantile mass, preserving every survivor's own weight.
        let size = 8;
        let mut li = 0;
        let mut ri = 0;
        while li < left.len() && ri < right.len() {
            let le = (li + size).min(left.len());
            let re = (ri + size).min(right.len());
            let left0 = li as f64 / left.len() as f64;
            let right0 = ri as f64 / right.len() as f64;
            let left1 = le as f64 / left.len() as f64;
            let right1 = re as f64 / right.len() as f64;
            let mass = (left1.min(right1) - left0.max(right0)).max(0.);
            let costs: Vec<Vec<f64>> = left[li..le]
                .iter()
                .map(|a| {
                    right[ri..re]
                        .iter()
                        .map(|b| swarm_cost(&a.phases, &b.phases).unwrap() / scale)
                        .collect()
                })
                .collect();
            let (value, plan) = transport_matrix(&costs)?;
            total += mass * value;
            blocks.push(json!({"left_range":[li,le],"right_range":[ri,re],"mass":mass,"local_cost":value,"local_plan":plan}));
            if left1 <= right1 + 1e-14 {
                li = le;
            }
            if right1 <= left1 + 1e-14 {
                ri = re;
            }
        }
    }
    Ok(
        json!({"available":true,"normalized_cost":total,"left_survivors":left.len(),"right_survivors":right.len(),"estimator":if exact{"exact_optimal_transport_between_empirical_surviving_state_laws"}else{"explicit_block_coupling_upper_bound_between_complete_empirical_surviving_state_laws"},"own_weights":[1./left.len() as f64,1./right.len() as f64],"coupling_blocks":blocks,"law_distance_bias":"Two finite independent empirical laws have nonzero sampling distance even at equilibrium. This measurement does not identify distance to a QSD or certify a decay coefficient."}),
    )
}
fn alive_transport(left: &[&Snapshot], right: &[&Snapshot], scale: f64) -> Result<Value> {
    if left.is_empty() || right.is_empty() {
        return Ok(json!({"available":false}));
    }
    let l: Vec<_> = left
        .iter()
        .filter_map(|s| s.uniform_alive_sample.as_ref())
        .collect();
    let r: Vec<_> = right
        .iter()
        .filter_map(|s| s.uniform_alive_sample.as_ref())
        .collect();
    let costs: Vec<Vec<f64>> = l
        .iter()
        .map(|a| {
            r.iter()
                .map(|b| a.iter().zip(*b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / scale)
                .collect()
        })
        .collect();
    let (value, plan) = transport_matrix(&costs)?;
    Ok(
        json!({"available":true,"normalized_cost":value,"samples":[l.len(),r.len()],"sampling_order":"one independently uniformly sampled alive walker per surviving swarm, followed by uniform sampling of swarms","transport_plan":plan,"scope":"Empirical alive marginal law transport; not pooled all-slot conditional sampling, not whole-swarm QSD distance."}),
    )
}

/// Primitive same-kernel positivity and integrability certificate for the exact
/// harmonic BAOAB map and final independent positional Gaussian. It certifies
/// minorization on a specified all-alive target, not a two-sided QSD block bound.
#[allow(clippy::too_many_arguments)]
fn primitives(
    n: usize,
    d: usize,
    half_width: f64,
    h: f64,
    gamma: f64,
    jitter: f64,
    alpha: f64,
    cap: f64,
    thermostat: f64,
    sigma_pos: f64,
) -> Value {
    let c = (-gamma * h).exp();
    let b = h * (1. + c) / 2.;
    let a = 1. - h * b / 2.;
    let av = -h * (c + a) / 2.;
    let bv = c - h * b / 2.;
    let q2 = thermostat * thermostat * (-(-2. * gamma * h).exp_m1()) / (2. * gamma);
    let nx = h / 2.;
    let nv = 1. - h * h / 4.;
    let qxx = nx * nx * q2 + h * sigma_pos * sigma_pos;
    let qxv = nx * nv * q2;
    let qvv = nv * nv * q2;
    let determinant = qxx * qvv - qxv * qxv;
    let rx = half_width + jitter;
    let vclone = (1. + 2. * alpha) * cap;
    let meanx = a.abs() * rx + b.abs() * vclone;
    let meanv = av.abs() * rx + bv.abs() * vclone;
    let vtarget = cap / (4. * (d as f64).sqrt());
    let ex = half_width / 2. + meanx;
    let ev = cap / 3. + meanv;
    let exponent = (qvv * ex * ex + 2. * qxv.abs() * ex * ev + qxx * ev * ev) / determinant;
    // Density times interval length gives a rigorous event lower bound for |Z|<=1.
    let log_jitter_event_per_coordinate = (2. / (2. * std::f64::consts::PI).sqrt()).ln() - 0.5;
    let log_gaussian_lower =
        -(2. * std::f64::consts::PI).ln() - 0.5 * determinant.ln() - 0.5 * exponent;
    let log_epsilon = (n * d) as f64
        * (log_jitter_event_per_coordinate + log_gaussian_lower + (half_width * 2. * vtarget).ln());
    let sd = sigma_pos * h.sqrt();
    let dimension_bounds =
        crate::convergence_barrier_dimension::dimension_barrier_bounds(d, half_width, sd)
            .expect("native primitive certificate has positive final positional Gaussian");
    let log_barrier_bound = dimension_bounds.sharp_bound.ln();
    json!({"harmonic_map":[[a,b],[av,bv]],"kinetic_joint_covariance":[[qxx,qxv],[qxv,qvv]],"joint_covariance_determinant":determinant,"strictly_positive_joint_density":determinant>0.,"jitter_event":"every used coordinate innovation |Z|<=1; unused innovations can be augmented independently without changing the marginal","jitter_event_log_lower_per_coordinate":log_jitter_event_per_coordinate,"clone_velocity_norm_bound":vclone,"prepared_position_coordinate_bound":rx,"target":{"all_rows_alive":true,"position_half_width":half_width/2.,"velocity_coordinate_half_width":vtarget,"uniform_reference":"independent uniform coordinates on position/velocity target, then quotient by permutations"},"same_kernel_all_alive_minorization":{"log_epsilon":log_epsilon,"epsilon_representable":(log_epsilon>=f64::MIN_POSITIVE.ln()).then(||log_epsilon.exp()),"certified_scope":"Harmonic native selected/capped/terminal-box kernel, all slot velocities <=cap and each entering alive position inside box; arbitrary entering dead positions are overwritten by mandatory revival. Gaussian jitter bound, actual accepted-component velocity bound, exact joint harmonic transition, inverse radial-cap Jacobian >=1 and independent per-row kinetic innovations.","rate_scope":"N-dependent positivity certificate. This is not an N-uniform mixing rate, not an upper surviving-block domination and not a QSD eigenvalue certificate."},"log_barrier":{"dimension_bounds":dimension_bounds,"formula":"phi(x)=-sum_j log(1-(x_j/L)^2), alive only; dead atoms have zero barrier","box_integral":"(2L)^d * 2d(1-log 2)","final_position_noise_standard_deviation":sd,"uniform_expected_N_normalized_bound":log_barrier_bound.exp(),"log_uniform_expected_bound":log_barrier_bound,"integrability":"Final independent Gaussian coordinates factor the candidate alive barrier. The closed-form one-coordinate layer-cake bound and remaining coordinate survival factors give d*Lambda(a)*min(1,a)^(d-1), uniformly over unrestricted means and independently of N. Legacy full-box-density and coordinate-linear bounds remain recorded in dimension_bounds."},"missing_global_certificates":["two-sided surviving-block upper domination with its same reference law","QSD eigenmeasure and eigenvalue","joint-swarm-law LSI","population-uniform same-kernel mixing coefficient","global sensitivity endpoints"]})
}
#[allow(clippy::too_many_arguments)]
async fn trajectory(
    config: &ExperimentConfig,
    store: &mut ArchiveStore,
    case: usize,
    base: &RunConfig,
    distribution: usize,
    replicate: usize,
    checkpoints: &[usize],
    half_width: f64,
) -> Result<Trajectory> {
    let seed = config.seed.wrapping_add(
        0x0003_9000_0000
            + case as u64 * 0x1000_0000
            + distribution as u64 * 0x100_0000
            + replicate as u64 * 7919,
    );
    let mut run = base.clone();
    run.gas.seed = seed;
    let center = if distribution == 0 {
        -0.45 * half_width
    } else {
        0.45 * half_width
    };
    run.initial_lower = center - 0.08 * half_width;
    run.initial_upper = center + 0.08 * half_width;
    let mut gas = run.build::<f64>().await?;
    let recording = RecordingConfig {
        max_steps: config.archive_chunk_steps,
        max_bytes: 128 * 1024 * 1024,
        graph: false,
    };
    gas.start_recording(recording.clone())?;
    let mut result = Trajectory {
        distribution,
        replicate,
        seed,
        archive_paths: vec![],
        checkpoint_paths: vec![],
        checkpoints: vec![snapshot(
            gas.population(),
            0,
            seed,
            half_width,
            0.2 * half_width,
        )?],
        native_extinction_step: None,
        theorem_cemetery_step: None,
        death_predictions: vec![],
    };
    for step in 1..=config.steps {
        match gas.step().await {
            Ok(_) => {}
            Err(GasError::Extinction) => {
                result.native_extinction_step = Some(step - 1);
                break;
            }
            Err(e) => return Err(e),
        }
        let alive = gas
            .population()
            .eligible(false)
            .iter()
            .filter(|a| **a)
            .count();
        let was_theorem_dead = result.theorem_cemetery_step.is_some();
        let archive = gas.recording().ok_or_else(|| error("recording absent"))?;
        let last = archive
            .steps
            .last()
            .ok_or_else(|| error("last recorded transition missing"))?;
        result.death_predictions.push(prediction(
            last,
            step,
            half_width,
            run.gas.kinetic.position_diffusion * 0.04_f64.sqrt(),
            0.2 * half_width,
            was_theorem_dead,
        )?);
        if alive < 2 && result.theorem_cemetery_step.is_none() {
            result.theorem_cemetery_step = Some(step);
        }
        if checkpoints.contains(&step) {
            result.checkpoints.push(snapshot(
                gas.population(),
                step,
                seed,
                half_width,
                0.2 * half_width,
            )?);
        }
        let finished = alive == 0 || step == config.steps;
        if step % config.archive_chunk_steps == 0 || finished {
            let archive = gas
                .stop_recording()
                .ok_or_else(|| error("archive absent"))?;
            let tag = format!(
                "chapter06-case{case}-distribution{distribution}-replicate{replicate}-through{step}"
            );
            result
                .archive_paths
                .push(store.save_archive(&tag, &archive)?);
            result
                .checkpoint_paths
                .push(store.save_checkpoint(&tag, &gas.checkpoint())?);
            if !finished {
                gas.start_recording(recording.clone())?;
            }
        }
        if alive == 0 {
            result.native_extinction_step = Some(step);
            break;
        }
    }
    if let Some(archive) = gas.stop_recording()
        && !archive.steps.is_empty()
    {
        result.archive_paths.push(store.save_archive(
            &format!("chapter06-case{case}-distribution{distribution}-replicate{replicate}-last"),
            &archive,
        )?);
    }
    Ok(result)
}
fn conditional_summary(survivors: &[&Snapshot]) -> Value {
    let m = survivors.len();
    if m == 0 {
        return json!({"survivors":0,"conditional_moments":null});
    }
    let fields = [
        (
            "position_variance",
            survivors
                .iter()
                .map(|s| s.normalized_position_variance)
                .collect::<Vec<_>>(),
        ),
        (
            "velocity_variance",
            survivors
                .iter()
                .map(|s| s.normalized_velocity_variance)
                .collect(),
        ),
        (
            "velocity_mean_energy",
            survivors.iter().map(|s| s.velocity_mean_energy).collect(),
        ),
        (
            "position_second_moment",
            survivors
                .iter()
                .map(|s| s.normalized_position_second_moment)
                .collect(),
        ),
        (
            "log_barrier",
            survivors.iter().map(|s| s.normalized_log_barrier).collect(),
        ),
        (
            "alive_safe_fraction",
            survivors.iter().map(|s| s.alive_safe_fraction).collect(),
        ),
    ];
    let mut out = BTreeMap::new();
    for (name, values) in fields {
        let mean = values.iter().sum::<f64>() / m as f64;
        let se = if m > 1 {
            (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (m * (m - 1)) as f64).sqrt()
        } else {
            0.
        };
        out.insert(name,json!({"mean":mean,"standard_error":(m>1).then_some(se),"independent_swarm_replicates":m}));
    }
    json!({"survivors":m,"conditional_moments":out,"normalization":"own number of surviving whole-swarm trajectories for each initial law and cemetery convention"})
}
fn summarize_case(
    trajectories: &[Trajectory],
    checkpoints: &[usize],
    n: usize,
    d: usize,
    half_width: f64,
    cap: f64,
    barrier_bound: f64,
) -> Result<Value> {
    let scale = 1. + 4. * d as f64 * half_width * half_width + 4. * cap * cap;
    let mut frames = vec![];
    let mut comparisons = vec![];
    let mut hazard = vec![];
    let each = trajectories.iter().filter(|t| t.distribution == 0).count();
    for convention in ["native_zero_alive", "theorem_fewer_than_two"] {
        for &time in checkpoints {
            let survivors: Vec<Vec<&Snapshot>> = (0..2)
                .map(|dist| {
                    trajectories
                        .iter()
                        .filter(|t| t.distribution == dist)
                        .filter(|t| {
                            let stop = if convention == "native_zero_alive" {
                                t.native_extinction_step
                            } else {
                                t.theorem_cemetery_step
                            };
                            stop.is_none_or(|s| s > time)
                        })
                        .filter_map(|t| t.checkpoints.iter().find(|s| s.step == time))
                        .collect()
                })
                .collect();
            let law = law_transport(&survivors[0], &survivors[1], scale, n <= 4)?;
            let alive = alive_transport(&survivors[0], &survivors[1], scale)?;
            let survival_intervals = [
                binomial_interval(survivors[0].len(), each)?,
                binomial_interval(survivors[1].len(), each)?,
            ];
            frames.push(json!({"step":time,"cemetery_convention":convention,"survivor_counts":[survivors[0].len(),survivors[1].len()],"requested_trajectories_each":each,"own_survival_probability_estimates":[survivors[0].len() as f64/each as f64,survivors[1].len() as f64/each as f64],"survival_wilson_95_intervals":survival_intervals,"conditional_moment_summaries":[conditional_summary(&survivors[0]),conditional_summary(&survivors[1])],"whole_swarm_law_transport":law,"uniform_alive_marginal_transport":alive,"certified_qsd_rate_comparison":null,"certified_rate_unavailable_reason":"Empirical surviving-law distances do not supply a uniform two-sided killed-block density certificate or its c_N,C_N,m constants."}));
            for (dist, states) in survivors.iter().enumerate() {
                if time > 0 {
                    let values: Vec<f64> = states
                        .iter()
                        .map(|s| s.normalized_log_barrier)
                        .chain(std::iter::repeat_n(0., each - states.len()))
                        .collect();
                    let mean = values.iter().sum::<f64>() / each as f64;
                    let se = if each > 1 {
                        (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / (each * (each - 1)) as f64)
                            .sqrt()
                    } else {
                        0.
                    };
                    let passed = mean <= barrier_bound + 6. * se;
                    comparisons.push(json!({"id":format!("uniform_log_barrier_{convention}_distribution{dist}_step{time}"),"source_labels":["lem-convergence-drift-iteration"],"source_formula":"QV\\le rV+b","observed":mean,"bound":barrier_bound,"relation":"upper","tolerance":6.*se,"passed":passed,"scope":"N-independent final-Gaussian density/integrable-box barrier bound: QW_b<=B, r=0; cemetery observable is zero, no survival conditioning is substituted.","hypotheses":{"phi":"-sum_j log(1-(x_j/L)^2)","independent_seed_trajectories":each,"survivors":states.len(),"variance_normalization":"1/N","observable_normalization":"unconditional 1/requested_trajectories including absorbed zeros","standard_error":se,"final_independent_position_gaussian":true,"derived_density_integral_bound":barrier_bound}}));
                    if !states.is_empty() {
                        comparisons.push(json!({"id":format!("survivor_log_barrier_normalization_{convention}_distribution{dist}_step{time}"),"source_labels":["def-qsd"],"source_formula":"\\Phi_n(\\mu)=\\frac{\\mu Q^n}{\\mu Q^n1}.","observed":states.iter().map(|s|s.normalized_log_barrier).sum::<f64>()/states.len() as f64,"bound":mean/(states.len() as f64/each as f64),"relation":"equal","tolerance":1e-10,"passed":true,"scope":"Exact empirical survival normalization for this initial distribution and cemetery convention. This verifies conditioning without claiming the empirical law is quasi-stationary.","hypotheses":{"own_empirical_survival_mass":states.len() as f64/each as f64,"initial_distribution":dist,"cemetery_convention":convention}}));
                    }
                }

                for (i, s) in states.iter().enumerate() {
                    comparisons.push(json!({"id":format!("interior_count_{convention}_{dist}_time{time}_state{i}"),"source_labels":["lem-convergence-interior-count"],"source_formula":"\\#\\{i\\in A:x_i\\in K\\}\\ge |A|-NL/H.","observed":-(s.safe_count as f64)/n as f64,"bound":-s.barrier_count_lower_bound/n as f64,"relation":"upper","passed":s.safe_count as f64+1e-10>=s.barrier_count_lower_bound,"scope":"Actual alive-only integrable logarithmic barrier, own N normalization and declared safe set. Count inequality divided by N yields population-independent fractions.","hypotheses":{"N":n,"alive":s.alive,"safe_margin":0.2*half_width,"W_b":s.normalized_log_barrier}}));
                }
            }
        }
    }
    for convention in ["native_zero_alive", "theorem_fewer_than_two"] {
        for dist in 0..2 {
            for time in 1..=checkpoints.last().copied().unwrap_or(0) {
                let rows: Vec<_> = trajectories
                    .iter()
                    .filter(|t| t.distribution == dist)
                    .filter_map(|t| t.death_predictions.iter().find(|p| p.step == time))
                    .filter(|p| convention == "native_zero_alive" || !p.theorem_absorbed_before)
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                let m = rows.len();
                let probabilities: Vec<_> = rows
                    .iter()
                    .map(|p| {
                        if convention == "native_zero_alive" {
                            p.conditional_all_die
                        } else {
                            p.conditional_fewer_two_survive
                        }
                    })
                    .collect();
                let events = rows
                    .iter()
                    .filter(|p| {
                        if convention == "native_zero_alive" {
                            p.alive_after == 0
                        } else {
                            p.alive_after < 2
                        }
                    })
                    .count();
                let prediction = probabilities.iter().sum::<f64>() / m as f64;
                let variance = probabilities.iter().map(|p| p * (1. - p)).sum::<f64>();
                let delta = 0.01 / (2. * 2. * (*checkpoints.last().unwrap_or(&1)) as f64);
                let log = (2. / delta).ln();
                let numerical_error = rows
                    .iter()
                    .map(|p| p.gaussian_probability_error)
                    .sum::<f64>()
                    / m as f64;
                let tolerance =
                    ((2. * variance * log).sqrt() + 2. * log / 3.) / m as f64 + numerical_error;
                let observed = events as f64 / m as f64;
                let passed = (observed - prediction).abs() <= tolerance;
                let interval = binomial_interval(events, m)?;
                hazard.push(json!({"step":time,"distribution":dist,"cemetery_convention":convention,"entering_survivor_trajectories":m,"new_absorptions":events,"observed_probability":observed,"conditional_gaussian_prediction":prediction,"conditional_variance_sum":variance,"binomial_wilson_95_interval":interval,"prediction_error_tolerance":tolerance,"family_error_budget":delta,"passed":passed,"scope":"Independent current positional Gaussians conditional on each swarm's own recorded cloning and kinetic preparation. State-dependent shared cloning is retained before conditioning; probabilities are not assigned independence before that preparation."}));
                comparisons.push(json!({"id":format!("conditional_extinction_{convention}_distribution{dist}_step{time}"),"source_labels":["prop-convergence-survival-bound"],"observed":observed,"bound":prediction,"relation":"equal","tolerance":tolerance,"passed":passed,"scope":"Native final Gaussian Poisson-binomial absorption; data comparison under actual preparation and own survivor denominator. No QSD stationarity assumption.","hypotheses":{"independent_trajectories":m,"conditional_position_noise_independent":true,"gaussian_cdf_error_bound":numerical_error}}));
            }
        }
    }
    let mut decay = vec![];
    for convention in ["native_zero_alive", "theorem_fewer_than_two"] {
        let start = frames
            .iter()
            .find(|f| f["cemetery_convention"] == convention && f["step"] == 0);
        let end = frames.iter().find(|f| {
            f["cemetery_convention"] == convention
                && f["step"] == json!(checkpoints.last().copied().unwrap_or(0))
        });
        for measure in [
            "whole_swarm_law_transport",
            "uniform_alive_marginal_transport",
        ] {
            let initial = start.and_then(|f| f[measure]["normalized_cost"].as_f64());
            let terminal = end.and_then(|f| f[measure]["normalized_cost"].as_f64());
            decay.push(json!({"cemetery_convention":convention,"measure":measure,"initial_cost":initial,"terminal_cost":terminal,"terminal_to_initial_ratio":initial.zip(terminal).and_then(|(a,b)|(a>0.).then_some(b/a)),"empirical_endpoint_decline":initial.zip(terminal).map(|(a,b)|b<a),"certified_rate":null,"scope":"Endpoint comparison between empirical laws from distinct independent initial-distribution ensembles. Finite empirical sampling error and its stationary floor are retained; decline is not a theorem rate fit."}));
        }
    }
    let extinct:Vec<_>=(0..2).map(|dist|json!({"distribution":dist,"native_times":trajectories.iter().filter(|t|t.distribution==dist).map(|t|t.native_extinction_step).collect::<Vec<_>>(),"theorem_times":trajectories.iter().filter(|t|t.distribution==dist).map(|t|t.theorem_cemetery_step).collect::<Vec<_>>(),"right_censoring_step":checkpoints.last(),"zero_observed_extinctions_one_sided_95_upper_if_applicable":-((0.05_f64.ln()/each as f64).exp_m1())})).collect();
    Ok(
        json!({"N":n,"d":d,"samples_per_initial_distribution":each,"phase_cost_normalization":scale,"phase_state_cost":"N^-1 min_permutation sum(status difference squared + alive-masked position difference squared + original capped velocity difference squared), divided by fixed scale. Arbitrary dead storage positions do not enter; original dead velocities are retained because the native collision uses them.","checkpoint_law_frames":frames,"empirical_law_decay":decay,"conditional_hazard_calibration":hazard,"extinction_and_censoring":extinct,"comparisons":comparisons,"normalization_scope":"All errors and moment bounds are independent of N; alive marginal and whole-swarm probability laws retain separate meanings."}),
    )
}

/// Run the actual native chains and persist every full Markov archive chunk.
pub async fn run(config: &ExperimentConfig, store: &mut ArchiveStore) -> Result<Value> {
    if config.samples < 2 || config.steps == 0 || config.archive_chunk_steps == 0 {
        return Err(error(
            "Chapter 6 needs samples>=2, positive steps and archive chunk size",
        ));
    }
    let samples = if config.compact {
        config.samples.min(16)
    } else {
        config.samples
    };
    let matrix: Vec<(usize, usize, bool)> = if config.compact {
        vec![(2, 1, true), (4, 1, false)]
    } else {
        vec![
            (4, 1, false),
            (4, 2, false),
            (16, 1, false),
            (16, 2, false),
            (64, 1, false),
            (64, 2, false),
            (2, 1, true),
        ]
    };
    let mut cases = vec![];
    let mut archive_manifests = vec![];
    let mut total_comparisons = 0;
    let mut failed = 0;
    let mut checkpoints = vec![0, 1, 2, 4, 8, 16, 32, 64, config.steps];
    checkpoints.retain(|s| *s <= config.steps);
    checkpoints.sort_unstable();
    checkpoints.dedup();
    for (case, (n, d, hazard)) in matrix.into_iter().enumerate() {
        let half_width = if hazard { 0.2 } else { 2. };
        let mut base = RunConfig::euclidean()?;
        base.walkers = n;
        base.dimensions = d;
        base.gas = algorithmic_gas::GasConfig::euclidean(d, 0.04)?;
        base.gas.boundary = BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-half_width; d],
                upper: vec![half_width; d],
            },
        };
        if hazard {
            base.gas.kinetic.position_diffusion = 1.;
        }
        let case_config_path=store.save_json(&format!("chapter06-case{case}-configuration"),&json!({"case":case,"run_config":base,"samples_per_distribution":samples,"steps":config.steps,"checkpoints":checkpoints,"initial_distributions":"independent uniform positions centered at +/-0.45 L with half-width 0.08 L; zero velocities","seed_rule":"seed + 0x390000000 + case*0x10000000 + distribution*0x1000000 + replicate*7919"}))?;
        let mut trajectories = vec![];
        for distribution in 0..2 {
            for replicate in 0..samples {
                trajectories.push(
                    trajectory(
                        config,
                        store,
                        case,
                        &base,
                        distribution,
                        replicate,
                        &checkpoints,
                        half_width,
                    )
                    .await?,
                );
            }
        }
        let manifest = json!({"case":case,"native_config":base,"trajectories":trajectories,"checkpoints":checkpoints,"sampling":"Distinct engine seeds and independently sampled initial swarms per distribution; no resampling. Full archives preserve all operator stages/noises, rewards, validity, accepted donor/collision graph and original velocities."});
        let manifest_path = store.save_json(
            &format!("chapter06-case{case}-trajectory-manifest"),
            &manifest,
        )?;
        archive_manifests.push(manifest_path.clone());
        let primitive = primitives(
            n,
            d,
            half_width,
            0.04,
            1.,
            0.1,
            0.5,
            2.,
            1.,
            base.gas.kinetic.position_diffusion,
        );
        let barrier_bound = primitive["log_barrier"]["uniform_expected_N_normalized_bound"]
            .as_f64()
            .unwrap();
        let mut summary = summarize_case(
            &trajectories,
            &checkpoints,
            n,
            d,
            half_width,
            2.,
            barrier_bound,
        )?;
        summary["case"] = json!(case);
        summary["configuration_record"] = json!(case_config_path);
        summary["trajectory_manifest"] = json!(manifest_path);
        summary["native_config"] =
            serde_json::to_value(&base.gas).map_err(|e| error(&e.to_string()))?;
        summary["primitive_certificates"] = primitive;
        let cs = summary["comparisons"].as_array().unwrap();
        total_comparisons += cs.len();
        failed += cs.iter().filter(|c| c["passed"] == false).count();
        cases.push(summary);
    }
    let comparisons: Vec<Value> = cases
        .iter()
        .flat_map(|c| c["comparisons"].as_array().unwrap().iter().cloned())
        .collect();
    Ok(
        json!({"chapter":6,"title":"Native killed-chain ensembles and survivor-law measurements","comparisons":comparisons,"source_path":"docs/source/2_fractal_gas/convergence_program/06_convergence.md","cases":cases,"archive_manifests":archive_manifests,"summary":{"cases":cases.len(),"samples_per_distribution":samples,"comparisons":total_comparisons,"comparisons_failed":failed,"new_native_trajectories":cases.len()*2*samples,"steps_requested":config.steps},"scope":"Actual independent native Markov chains. Native k=0 and externally stopped k<2 processes are distinct, with their own absorption times and survival denominators. Finite empirical survivor-law comparisons, analytic barrier/positive-density certificates and conditional death calibration are not an identified QSD or a certified N-uniform TV mixing theorem."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(x: f64) -> Snapshot {
        Snapshot {
            step: 1,
            alive: 1,
            phases: vec![vec![1., x, 0.]],
            uniform_alive_sample: Some(vec![x, 0.]),
            normalized_position_variance: 0.,
            normalized_velocity_variance: 0.,
            velocity_mean_energy: 0.,
            normalized_position_second_moment: x * x,
            normalized_log_barrier: 0.,
            alive_safe_fraction: 1.,
            safe_count: 1,
            barrier_count_lower_bound: 1.,
        }
    }
    #[test]
    fn poisson_binomial_accounts_for_both_cemetery_conventions() {
        let (zero, fewer_two) = poisson_extinction(&[0.8, 0.6]);
        assert!((zero - 0.08).abs() < 1e-12);
        assert!((fewer_two - 0.52).abs() < 1e-12);
    }
    #[test]
    fn block_coupling_preserves_every_survivor_weight_with_unequal_populations() {
        let l: Vec<_> = (0..10).map(|i| state(i as f64 / 20.)).collect();
        let r: Vec<_> = (0..13).map(|i| state(i as f64 / 30.)).collect();
        let left: Vec<_> = l.iter().collect();
        let right: Vec<_> = r.iter().collect();
        let report = law_transport(&left, &right, 1., false).unwrap();
        let mut lm = [0.; 10];
        let mut rm = [0.; 13];
        for block in report["coupling_blocks"].as_array().unwrap() {
            let lo = block["left_range"][0].as_u64().unwrap() as usize;
            let ro = block["right_range"][0].as_u64().unwrap() as usize;
            let mass = block["mass"].as_f64().unwrap();
            for edge in block["local_plan"].as_array().unwrap() {
                let i = lo + edge["left"].as_u64().unwrap() as usize;
                let j = ro + edge["right"].as_u64().unwrap() as usize;
                let value = mass * edge["mass"].as_f64().unwrap();
                lm[i] += value;
                rm[j] += value;
            }
        }
        for x in lm {
            assert!((x - 0.1).abs() < 1e-12);
        }
        for x in rm {
            assert!((x - 1. / 13.).abs() < 1e-12);
        }
    }
}

/// Fresh dimension probes using exactly the existing selected-cloning recorder.
/// Case identifiers reserve 100/101; the output dataset is separate and immutable.
pub async fn run_dimension_probes(
    config: &ExperimentConfig,
    store: &mut ArchiveStore,
) -> Result<Value> {
    config.validate()?;
    let checkpoints: Vec<_> = [0, 1, 2, 4, 8, 16, 32, 64, config.steps]
        .into_iter()
        .filter(|t| *t <= config.steps)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut cases = vec![];
    let mut comparisons = vec![];
    for (case, d) in [(100, 4), (101, 8)] {
        let n = 16;
        let half_width = 2.;
        let mut base = RunConfig::euclidean()?;
        base.walkers = n;
        base.dimensions = d;
        base.gas = algorithmic_gas::GasConfig::euclidean(d, 0.04)?;
        base.gas.boundary = BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: BoxDomain {
                lower: vec![-half_width; d],
                upper: vec![half_width; d],
            },
        };
        let configuration_record=store.save_json(&format!("chapter06-case{case}-configuration"),&json!({"case":case,"run_config":base,"samples_per_distribution":config.samples,"steps":config.steps,"checkpoints":checkpoints,"scope":"Additional dimension probe, distinct from original 256-seed d1/2 runs."}))?;
        let mut trajectories = vec![];
        for distribution in 0..2 {
            for replicate in 0..config.samples {
                trajectories.push(
                    trajectory(
                        config,
                        store,
                        case,
                        &base,
                        distribution,
                        replicate,
                        &checkpoints,
                        half_width,
                    )
                    .await?,
                );
            }
        }
        let manifest_path=store.save_json(&format!("chapter06-case{case}-trajectory-manifest"),&json!({"case":case,"native_config":base,"trajectories":trajectories,"checkpoints":checkpoints,"sampling":"Distinct seeds; original native operator; full archive/checkpoint retention; no resampling."}))?;
        let bounds = crate::convergence_barrier_dimension::dimension_barrier_bounds(
            d,
            half_width,
            base.gas.kinetic.position_diffusion * 0.04_f64.sqrt(),
        )?;
        let mut report = summarize_case(
            &trajectories,
            &checkpoints,
            n,
            d,
            half_width,
            2.,
            bounds.sharp_bound,
        )?;
        report["case"] = json!(case);
        report["N"] = json!(n);
        report["d"] = json!(d);
        report["configuration_record"] = json!(configuration_record);
        report["trajectory_manifest"] = json!(manifest_path);
        report["dimension_bounds"] = json!(bounds);
        comparisons.extend(report["comparisons"].as_array().unwrap().iter().cloned());
        cases.push(report);
    }
    let failed = comparisons.iter().filter(|c| c["passed"] == false).count();
    Ok(
        json!({"chapter":6,"title":"Additional native d4/d8 dimension probes","cases":cases,"comparisons":comparisons,"summary":{"cases":2,"new_native_trajectories":4*config.samples,"samples_per_initial_distribution":config.samples,"steps_requested":config.steps,"comparisons":comparisons.len(),"comparisons_failed":failed},"scope":"Selected-cloning/capped-BAOAB/absorbing-box native algorithm unchanged. Additional dimension probes retain their own sample precision and cemetery denominators."}),
    )
}
