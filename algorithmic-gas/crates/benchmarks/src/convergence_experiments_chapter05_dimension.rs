//! Certified dimension/curvature cap integrals and sector matrix coefficients.
use crate::{
    convergence_experiments::{ArchiveStore, ExperimentConfig},
    convergence_kinetic::ConfiningLandscape,
};
use algorithmic_gas::{
    GasConfig, GasError, Population, Result,
    boundary::BoundaryPolicy,
    kinetic::KineticKind,
    noise::{FactorValues, NoiseGeometry},
    tracking::{RecordedStep, RecordingConfig},
};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}
impl Interval {
    fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn add(self, y: Self) -> Self {
        Self {
            lo: (self.lo + y.lo).next_down(),
            hi: (self.hi + y.hi).next_up(),
        }
    }
    fn sub(self, y: Self) -> Self {
        Self {
            lo: (self.lo - y.hi).next_down(),
            hi: (self.hi - y.lo).next_up(),
        }
    }
    fn mul(self, y: Self) -> Self {
        let v = [
            self.lo * y.lo,
            self.lo * y.hi,
            self.hi * y.lo,
            self.hi * y.hi,
        ];
        Self {
            lo: v.iter().copied().fold(f64::INFINITY, f64::min).next_down(),
            hi: v
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
                .next_up(),
        }
    }
    fn div(self, y: Self) -> Self {
        assert!(y.lo > 0.);
        self.mul(Self {
            lo: (1. / y.hi).next_down(),
            hi: (1. / y.lo).next_up(),
        })
    }
    fn square(self) -> Self {
        if self.lo <= 0. && self.hi >= 0. {
            let m = self.lo.abs().max(self.hi.abs());
            Self {
                lo: 0.,
                hi: (m * m).next_up(),
            }
        } else {
            let a = self.lo * self.lo;
            let b = self.hi * self.hi;
            Self {
                lo: a.min(b).next_down().max(0.),
                hi: a.max(b).next_up(),
            }
        }
    }
    fn sqrt(self) -> Self {
        Self {
            lo: self.lo.max(0.).sqrt().next_down().max(0.),
            hi: self.hi.max(0.).sqrt().next_up(),
        }
    }
    fn midpoint(self) -> f64 {
        self.lo + (self.hi - self.lo) / 2.
    }
}

// Only directed basic IEEE arithmetic and correctly rounded square root enter
// the certificate. No platform exp/log/erf values are trusted as exact bounds.
fn exp_negative_point(t: f64) -> Interval {
    assert!(t >= 0. && t.is_finite());
    let mut s = 0usize;
    let mut u = t;
    while u > 0.125 {
        u /= 2.;
        s += 1;
    }
    let x = Interval::point(u);
    let mut term = Interval::point(1.);
    let mut sum = term;
    for n in 1..=16 {
        term = term.mul(x).div(Interval::point(n as f64));
        sum = if n % 2 == 0 {
            sum.add(term)
        } else {
            sum.sub(term)
        };
    }
    // Alternating-series remainder lies between minus the next term and zero.
    let remainder = term.mul(x).div(Interval::point(17.));
    sum = Interval {
        lo: sum.lo - remainder.hi,
        hi: sum.hi,
    };
    sum.lo = sum.lo.next_down().max(0.);
    for _ in 0..s {
        sum = sum.square();
    }
    Interval {
        lo: sum.lo.max(0.),
        hi: sum.hi.min(1.),
    }
}
fn exp_negative(t: Interval) -> Interval {
    Interval {
        lo: exp_negative_point(t.hi.max(0.)).lo,
        hi: exp_negative_point(t.lo.max(0.)).hi,
    }
}
fn pi_interval() -> Interval {
    fn arctan_inverse(n: f64) -> Interval {
        let x = Interval::point(1.).div(Interval::point(n));
        let x2 = x.square();
        let mut power = x;
        let mut sum = x;
        for j in 1..=16 {
            power = power.mul(x2);
            let term = power.div(Interval::point((2 * j + 1) as f64));
            sum = if j % 2 == 0 {
                sum.add(term)
            } else {
                sum.sub(term)
            };
        }
        let remainder = power.mul(x2).div(Interval::point(35.));
        Interval {
            lo: (sum.lo - remainder.hi).next_down(),
            hi: sum.hi,
        }
    }
    arctan_inverse(5.)
        .mul(Interval::point(16.))
        .sub(arctan_inverse(239.).mul(Interval::point(4.)))
}
fn chi_normalizer(d: usize) -> Interval {
    let mut coefficient = if d.is_multiple_of(2) {
        Interval::point(1.)
    } else {
        Interval::point(2.).div(pi_interval()).sqrt()
    };
    let mut current = if d.is_multiple_of(2) { 2 } else { 1 };
    while current < d {
        coefficient = coefficient.div(Interval::point(current as f64));
        current += 2;
    }
    coefficient
}
fn chi_density(d: usize, r: Interval, normalizer: Interval) -> Interval {
    let mut polynomial = normalizer;
    for _ in 1..d {
        polynomial = polynomial.mul(r);
    }
    polynomial.mul(exp_negative(r.square().div(Interval::point(2.))))
}
fn dissipation(r: Interval, sigma: Interval, cap: Interval, power: usize) -> Interval {
    let ratio = cap.div(cap.add(sigma.mul(r)));
    let mut product = Interval::point(1.);
    for _ in 0..power {
        product = product.mul(ratio);
    }
    Interval::point(1.).sub(product)
}

/// Certified integral enclosure, with dimension explicit and population absent.
pub fn cap_eta_enclosure(d: usize, sigma: f64, cap: f64, subdivisions: usize) -> Result<Value> {
    cap_eta_interval(
        d,
        Interval::point(sigma),
        Interval::point(cap),
        subdivisions,
    )
}
fn cap_eta_interval(
    d: usize,
    sigma: Interval,
    cap: Interval,
    subdivisions: usize,
) -> Result<Value> {
    if !(1..=256).contains(&d)
        || !sigma.hi.is_finite()
        || sigma.lo <= 0.
        || !cap.hi.is_finite()
        || cap.lo <= 0.
        || subdivisions == 0
        || subdivisions > 65_536
        || !subdivisions.is_power_of_two()
    {
        return Err(GasError::Configuration("Cap integral requires 1<=d<=256, positive finite sigma/cap, and power-of-two subdivisions<=65536".into()));
    }
    let power = if d == 1 { 4 } else { 2 };
    let radius = (d as f64).sqrt().ceil() as usize + 12;
    let normalizer = chi_normalizer(d);
    let step = 1. / subdivisions as f64;
    let mode = Interval::point((d - 1) as f64).sqrt();
    let mut previous = chi_density(d, Interval::point(0.), normalizer);
    let mut lower = Interval::point(0.);
    let mut upper = lower;
    for i in 0..radius * subdivisions {
        let left = Interval::point(i as f64 * step);
        let right = Interval::point((i + 1) as f64 * step);
        let next = chi_density(d, right, normalizer);
        let density_lower = previous.lo.min(next.lo).max(0.);
        let mut density_upper = previous.hi.max(next.hi);
        if left.lo <= mode.hi && right.hi >= mode.lo {
            density_upper = density_upper.max(chi_density(d, mode, normalizer).hi);
        }
        let width = right.sub(left);
        lower = lower.add(
            dissipation(left, sigma, cap, power)
                .mul(width)
                .mul(Interval::point(density_lower)),
        );
        upper = upper.add(
            dissipation(right, sigma, cap, power)
                .mul(width)
                .mul(Interval::point(density_upper)),
        );
        previous = next;
    }
    // Markov applied to exp(Chi_d^2/4): P(R>=Rmax)<=2^(d/2)exp(-Rmax²/4).
    let mut tail = Interval::point(2.).sqrt();
    if d.is_multiple_of(2) {
        tail = Interval::point(1.);
    }
    for _ in 0..d / 2 {
        tail = tail.mul(Interval::point(2.));
    }
    tail = tail.mul(exp_negative(Interval::point((radius * radius) as f64 / 4.)));
    upper = upper.add(tail);
    let eta_lower = lower.lo.max(0.);
    let eta_upper = upper.hi.min(1.);
    if !eta_lower.is_finite() || !eta_upper.is_finite() || eta_lower > eta_upper {
        return Err(GasError::Numerical(
            "Nonfinite cap integral enclosure".into(),
        ));
    }
    Ok(
        json!({"d":d,"sigma_interval":[sigma.lo,sigma.hi],"cap_interval":[cap.lo,cap.hi],"derivative_power":power,"eta_lower":eta_lower,"eta_upper":eta_upper,"enclosure_width":eta_upper-eta_lower,"radial_step":step,"radial_max":radius,"tail_upper":tail.hi,"normalizer_interval":[normalizer.lo,normalizer.hi],"certificate":"Directed IEEE interval arithmetic; exp(-t) uses range reduction, 16-term alternating Taylor plus next-term remainder; pi uses Machin alternating arctangent intervals; unimodal chi-density interval extrema and monotone dissipation give integral bounds; tail <=2^(d/2) exp(-Rmax²/4). No fitted/quadrature-only constant."}),
    )
}

fn smallest_eigenvalue(trace: Interval, determinant: Interval) -> Interval {
    if determinant.lo <= 0. || trace.lo <= 0. {
        return Interval::point(0.);
    }
    let discriminant = trace
        .square()
        .sub(determinant.mul(Interval::point(4.)))
        .sqrt();
    determinant
        .mul(Interval::point(2.))
        .div(trace.add(discriminant))
}

/// Isotropic curvature, dimension-aware Gaussian cap and certified exact spectrum.
pub fn constants(
    d: usize,
    h: f64,
    gamma: f64,
    diffusion: f64,
    cap: f64,
    omega: f64,
    subdivisions: usize,
) -> Result<Value> {
    if ![h, gamma, diffusion, cap, omega]
        .iter()
        .all(|v| v.is_finite() && *v > 0.)
        || !(h * gamma).is_finite()
        || !(2. * gamma).is_finite()
    {
        return Err(GasError::Configuration(
            "Dimension cap constants require positive finite h,gamma,B,V,curvature".into(),
        ));
    }
    let c = Interval::point(h).div(Interval::point(2.));
    let w = Interval::point(omega);
    let k = Interval::point(1.).sub(w.mul(c.square()));
    if k.lo <= 0. {
        return Err(GasError::Configuration(
            "Curvature metric requires 1-omega*h²/4>0 with certified positive margin".into(),
        ));
    }
    let a = exp_negative(Interval::point(gamma).mul(Interval::point(h)));
    let loss = Interval::point(1.).sub(a.square());
    let q = Interval::point(diffusion).mul(
        loss.div(Interval::point(2.).mul(Interval::point(gamma)))
            .sqrt(),
    );
    let sigma = k.mul(q);
    let eta = cap_eta_interval(d, sigma, Interval::point(cap), subdivisions)?;
    let eta_cert = Interval::point(eta["eta_lower"].as_f64().unwrap());
    let trace = loss.add(eta_cert.mul(a.square().add(w.mul(c.square()).mul(loss))));
    let determinant = eta_cert.mul(w).mul(c.square()).mul(loss);
    let delta = smallest_eigenvalue(trace, determinant);
    let old_eta = (2. / std::f64::consts::PI).sqrt()
        * (-2f64).exp()
        * (1. - (cap / (cap + sigma.midpoint())).powi(2));
    let old_trace = loss.midpoint()
        + old_eta * (a.midpoint().powi(2) + omega * c.midpoint().powi(2) * loss.midpoint());
    let old_delta = old_eta * omega * c.midpoint().powi(2) * loss.midpoint() / old_trace;
    Ok(
        json!({"d":d,"h":h,"gamma":gamma,"B":diffusion,"V":cap,"curvature":omega,"c":c.midpoint(),"k":k.midpoint(),"a_interval":[a.lo,a.hi],"q_interval":[q.lo,q.hi],"sigma_interval":[sigma.lo,sigma.hi],"eta_certificate":eta,"T_interval":[trace.lo,trace.hi],"D_interval":[determinant.lo,determinant.hi],"delta_exact_spectrum_lower":delta.lo,"delta_spectrum_interval":[delta.lo,delta.hi],"old_eta_uniform":old_eta,"old_delta_D_over_T":old_delta,"same_eta_D_over_T":determinant.lo/trace.hi,"N_independent":true}),
    )
}

type Matrix = [[Interval; 2]; 2];
fn imul(a: Matrix, b: Matrix) -> Matrix {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0].mul(b[0][j]).add(a[i][1].mul(b[1][j]))))
}

fn metric_points(p: &Population<f64>, omega: f64, beta: f64) -> Result<Vec<Vec<f64>>> {
    let x = p.observations.field("positions")?;
    let v = p.observations.field("velocities")?;
    (0..x.rows())
        .map(|i| {
            let xx = x.row(i)?;
            let vv = v.row(i)?;
            let mut point = vec![];
            point.extend(xx.iter().zip(vv).map(|(x, v)| omega.sqrt() * x + beta * v));
            point.extend(vv.iter().map(|v| (1. - beta * beta).sqrt() * v));
            Ok(point)
        })
        .collect()
}
pub fn optimal_sector_cost(
    a: &Population<f64>,
    b: &Population<f64>,
    omega: f64,
    beta: f64,
) -> Result<f64> {
    if !omega.is_finite() || omega <= 0. || !beta.is_finite() || beta.abs() >= 1. {
        return Err(GasError::Configuration(
            "Positive sector metric requires omega>0 and |beta|<1".into(),
        ));
    }
    super::optimal_qcost_points(
        &metric_points(a, omega, beta)?,
        &metric_points(b, omega, beta)?,
        1.,
    )
}
fn representative_sector_cost(
    a: &Population<f64>,
    b: &Population<f64>,
    omega: f64,
    beta: f64,
) -> Result<f64> {
    let aa = metric_points(a, omega, beta)?;
    let bb = metric_points(b, omega, beta)?;
    Ok(aa
        .iter()
        .zip(bb)
        .map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>())
        .sum::<f64>()
        / aa.len() as f64)
}
fn stage(recorded: &RecordedStep<f64>, name: &str, field: &str) -> Result<Vec<f64>> {
    if let Some(value) = recorded
        .stages
        .iter()
        .find(|s| s.stage == name)
        .and_then(|s| s.fields.get(field))
    {
        return Ok(value.values.clone());
    }
    recorded
        .field_evaluations
        .iter()
        .find(|f| f.stage == name && f.field == field)
        .map(|f| f.values.clone())
        .ok_or_else(|| GasError::MissingField(format!("{name}/{field}")))
}

fn source_equation(tag: &str) -> String {
    let source = include_str!(
        "../../../../docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md"
    );
    let end_tag = source
        .find(&format!("\\tag{{{tag}}}"))
        .expect("source equation tag");
    let start = source[..end_tag].rfind("$$").expect("opening equation");
    let end = end_tag + source[end_tag..].find("$$").expect("closing equation") + 2;
    source[start..end].to_owned()
}
fn bind(mut comparison: Value, tag: &str) -> Value {
    comparison["source_quotes"] = json!([source_equation(tag)]);
    comparison
}

/// Fresh complete native kinetic probes; all stage/noise/checkpoint data retained.
pub async fn run(config: &ExperimentConfig, store: &mut ArchiveStore) -> Result<Value> {
    if config.samples < 6 || config.steps == 0 || config.steps > 128 {
        return Err(GasError::Configuration(
            "Dimension run requires samples>=6 and 1<=steps<=128".into(),
        ));
    }
    let ns = if config.compact { vec![4] } else { vec![4, 64] };
    let ds = if config.compact {
        vec![1, 2]
    } else {
        vec![1, 2, 4, 8]
    };
    let curvatures = if config.compact {
        vec![1.]
    } else {
        vec![0.5, 1., 2.]
    };
    let h = 0.04;
    let gamma = 1.;
    let diffusion = 1.;
    let cap = 2.;
    let mut cases = vec![];
    let mut comparisons = vec![];
    let mut coefficient_table = vec![];
    for &omega in &curvatures {
        let sector = sector_constants(h, gamma, omega)?;
        let beta = sector["beta"].as_f64().unwrap();
        let sector_delta = sector["delta_pathwise_lower"].as_f64().unwrap();
        for &d in &ds {
            let diagonal = constants(d, h, gamma, diffusion, cap, omega, 4096)?;
            coefficient_table
                .push(json!({"d":d,"curvature":omega,"diagonal":diagonal,"sector":sector}));
            let eta = diagonal["eta_certificate"]["eta_lower"].as_f64().unwrap();
            let delta = diagonal["delta_exact_spectrum_lower"].as_f64().unwrap();
            let k = diagonal["k"].as_f64().unwrap();
            for &n in &ns {
                if omega != 1. && (n != 4 || d != 2) {
                    continue;
                }
                let tag = format!("omega{omega}_n{n}_d{d}");
                eprintln!(
                    "Dimension kinetic {tag}: {} pairs x {} steps",
                    config.samples, config.steps
                );
                let mut optimal_q = vec![vec![1.]; config.samples];
                let mut optimal_g = optimal_q.clone();
                let mut coupling_g = optimal_q.clone();
                let mut coupling_reference = optimal_q.clone();
                let mut eta_residuals = vec![];
                let mut archives = vec![];
                let mut maximum_pathwise_residual = f64::NEG_INFINITY;
                let mut maximum_reference_residual = f64::NEG_INFINITY;
                let mut maximum_force_residual = 0_f64;
                let mut maximum_matrix_residual = 0_f64;
                let mut clone_edges = 0usize;
                for rep in 0..config.samples {
                    let shift = match rep % 3 {
                        0 => [0.5, 0.],
                        1 => [0., 0.5],
                        _ => [0.35, 0.35],
                    };
                    let seed = config.seed.wrapping_add(
                        75_000_000
                            + (omega * 10.) as u64 * 1_000_000
                            + d as u64 * 100_000
                            + n as u64 * 1000
                            + rep as u64,
                    );
                    let left = super::population(n, d, [0., 0.])?;
                    let right = super::population(n, d, shift)?;
                    let initial_q = super::optimal_qcost(&left, &right, omega * k)?;
                    let initial_g = optimal_sector_cost(&left, &right, omega, beta)?;
                    let initial_reference = representative_sector_cost(&left, &right, omega, 0.04)?;
                    let mut landscape = ConfiningLandscape::quadratic(d);
                    landscape.curvature = vec![omega; d];
                    let mut cfg = GasConfig::euclidean(d, h)?;
                    cfg.seed = seed;
                    cfg.boundary = BoundaryPolicy::Unbounded;
                    cfg.fitness.reward_exponent = 0.;
                    cfg.fitness.diversity_exponent = 0.;
                    cfg.clone_transform.jitter_amplitude = 0.;
                    cfg.kinetic.velocity_cap = Some(cap);
                    cfg.kinetic.position_diffusion = 0.1;
                    if let KineticKind::Baoab { friction, .. } = &mut cfg.kinetic.integrator {
                        *friction = gamma;
                    }
                    cfg.kinetic.noise.geometry = NoiseGeometry::Isotropic {
                        scale: FactorValues::Constant {
                            values: vec![diffusion],
                        },
                    };
                    let mut left =
                        super::build_engine(left, cfg.clone(), landscape.clone()).await?;
                    let mut right = super::build_engine(right, cfg, landscape).await?;
                    let mut previous_g = initial_g;
                    let mut previous_reference = initial_reference;
                    for t in 1..=config.steps {
                        let previous_x: Vec<f64> = left
                            .population()
                            .observations
                            .field("positions")?
                            .values()
                            .iter()
                            .zip(right.population().observations.field("positions")?.values())
                            .map(|(a, b)| a - b)
                            .collect();
                        let previous_v: Vec<f64> = left
                            .population()
                            .observations
                            .field("velocities")?
                            .values()
                            .iter()
                            .zip(
                                right
                                    .population()
                                    .observations
                                    .field("velocities")?
                                    .values(),
                            )
                            .map(|(a, b)| a - b)
                            .collect();
                        let l = left.step().await?;
                        let r = right.step().await?;
                        clone_edges += l.clones + r.clones + l.revivals + r.revivals;
                        let qopt =
                            super::optimal_qcost(left.population(), right.population(), omega * k)?;
                        let gopt = optimal_sector_cost(
                            left.population(),
                            right.population(),
                            omega,
                            beta,
                        )?;
                        let gcouple = representative_sector_cost(
                            left.population(),
                            right.population(),
                            omega,
                            beta,
                        )?;
                        let reference = representative_sector_cost(
                            left.population(),
                            right.population(),
                            omega,
                            0.04,
                        )?;
                        maximum_pathwise_residual = maximum_pathwise_residual
                            .max((gcouple - (1. - sector_delta) * previous_g) / initial_g);
                        let reference_delta = if omega == 1. {
                            1. / 1040.
                        } else {
                            sector["reference_exact_endpoint_delta_lower"]
                                .as_f64()
                                .unwrap()
                        };
                        maximum_reference_residual = maximum_reference_residual.max(
                            (reference - (1. - reference_delta) * previous_reference)
                                / initial_reference,
                        );
                        previous_g = gcouple;
                        previous_reference = reference;
                        optimal_q[rep].push(qopt / initial_q);
                        optimal_g[rep].push(gopt / initial_g);
                        coupling_g[rep].push(gcouple / initial_g);
                        coupling_reference[rep].push(reference / initial_reference);
                        let ls = left.recording().unwrap().steps.last().unwrap();
                        let rs = right.recording().unwrap().steps.last().unwrap();
                        for recorded in [ls, rs] {
                            for (input, kick) in [("B1_input", "B1"), ("B2_input", "B2")] {
                                let positions = stage(recorded, input, "positions")?;
                                let gradient = stage(recorded, kick, "potential_gradient")?;
                                for (x, g) in positions.iter().zip(gradient) {
                                    maximum_force_residual =
                                        maximum_force_residual.max((g - omega * x).abs());
                                }
                            }
                        }
                        let xx = stage(ls, "A2", "positions")?;
                        let yy = stage(rs, "A2", "positions")?;
                        let vv = stage(ls, "B2", "velocities")?;
                        let ww = stage(rs, "B2", "velocities")?;
                        let a = (-gamma * h).exp();
                        let c = h / 2.;
                        let ax = 1. - omega * c * c * (1. + a);
                        let b = c * (1. + a);
                        let av = a - omega * c * c * (1. + a);
                        for j in 0..previous_x.len() {
                            maximum_matrix_residual = maximum_matrix_residual.max(
                                (xx[j] - yy[j] - ax * previous_x[j] - b * previous_v[j]).abs(),
                            );
                            maximum_matrix_residual = maximum_matrix_residual.max(
                                (vv[j] - ww[j] + omega * b * k * previous_x[j]
                                    - av * previous_v[j])
                                    .abs(),
                            );
                        }
                        if t == 1 {
                            let raw = vv
                                .iter()
                                .zip(&ww)
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                / n as f64;
                            let output = left
                                .population()
                                .observations
                                .field("velocities")?
                                .values()
                                .iter()
                                .zip(
                                    right
                                        .population()
                                        .observations
                                        .field("velocities")?
                                        .values(),
                                )
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                / n as f64;
                            eta_residuals.push(output - (1. - eta) * raw);
                        }
                        if t % config.archive_chunk_steps.max(1) == 0 || t == config.steps {
                            for (side, gas) in [("left", &mut left), ("right", &mut right)] {
                                let archive = gas.stop_recording().unwrap();
                                let path = store.save_archive(
                                    &format!("dimension/{tag}/rep{rep}/{side}_through{t}"),
                                    &archive,
                                )?;
                                archives.push(path);
                                archives.push(store.save_checkpoint(
                                    &format!("dimension/{tag}/rep{rep}/{side}_checkpoint{t}"),
                                    &gas.checkpoint(),
                                )?);
                                if t < config.steps {
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
                let hypotheses = json!({"N":n,"d":d,"curvature":omega,"h":h,"gamma":gamma,"B":diffusion,"V":cap,"samples":config.samples,"steps":config.steps,"diagonal_certificate":diagonal,"sector_certificate":sector,"input_direction_design":"rep % 3, fixed proportions; stratified independent-noise SE","scope":"Actual native complete kinetic update, unit fitness, no accepted clone/revival edges, unbounded domain; common addressed Gaussian innovations are a coupling representative, principal errors minimize over storage permutations."});
                let (m, se) = super::estimate_direction_strata(&eta_residuals)?;
                comparisons.push(bind(
                    super::check(
                        format!("{tag}/dimension_eta"),
                        "lem-kinetic-shift-uniform-radial-cap",
                        None,
                        m,
                        0.,
                        se,
                        "upper",
                        hypotheses.clone(),
                    ),
                    "5.DIM2",
                ));
                for (name, value, tolerance) in [
                    ("pathwise_sector", maximum_pathwise_residual, 0.),
                    ("reference_sector", maximum_reference_residual, 0.),
                    ("native_matrix", maximum_matrix_residual, 2e-12),
                    ("native_force", maximum_force_residual, 2e-12),
                    ("accepted_clone_edges", clone_edges as f64, 0.),
                ] {
                    let comparison = super::check(
                        format!("{tag}/{name}"),
                        "thm-kinetic-dimension-curvature-sector",
                        None,
                        value,
                        tolerance,
                        0.,
                        "upper",
                        hypotheses.clone(),
                    );
                    comparisons.push(if name.ends_with("sector") {
                        bind(comparison, "5.DIM7")
                    } else {
                        comparison
                    });
                }
                for t in 1..=config.steps {
                    for (name, trajectory, bound) in [
                        (
                            "optimal_Q_dimension",
                            &optimal_q,
                            (1. - delta).powi(t as i32),
                        ),
                        (
                            "optimal_G_sector",
                            &optimal_g,
                            (1. - sector_delta).powi(t as i32),
                        ),
                    ] {
                        let (m, se) = super::estimate_direction_strata(
                            &trajectory.iter().map(|row| row[t]).collect::<Vec<_>>(),
                        )?;
                        comparisons.push(bind(
                            super::check(
                                format!("{tag}/step{t}/{name}"),
                                "cor-kinetic-dimension-iterated-transport",
                                None,
                                m,
                                bound,
                                se,
                                "upper",
                                hypotheses.clone(),
                            ),
                            if name == "optimal_Q_dimension" {
                                "5.DIM9a"
                            } else {
                                "5.DIM9b"
                            },
                        ));
                    }
                }
                let path=store.save_json(&format!("dimension/{tag}/coupling_plan"),&json!({"hypotheses":hypotheses,"optimal_Q_trajectories":optimal_q,"optimal_G_trajectories":optimal_g,"sector_representative_trajectories":coupling_g,"reference_representative_trajectories":coupling_reference,"eta_residuals":eta_residuals,"archives":archives}))?;
                let q_final = super::estimate_direction_strata(
                    &optimal_q
                        .iter()
                        .map(|row| row[config.steps])
                        .collect::<Vec<_>>(),
                )?;
                let g_final = super::estimate_direction_strata(
                    &optimal_g
                        .iter()
                        .map(|row| row[config.steps])
                        .collect::<Vec<_>>(),
                )?;
                cases.push(json!({"N":n,"d":d,"curvature":omega,"raw_plan":path,"final_optimal_Q_ratio":q_final.0,"final_Q_SE":q_final.1,"final_optimal_G_ratio":g_final.0,"final_G_SE":g_final.1,"diagonal_delta":delta,"sector_delta":sector_delta,"beta":beta,"maximum_pathwise_residual":maximum_pathwise_residual,"maximum_matrix_residual":maximum_matrix_residual,"maximum_force_residual":maximum_force_residual}));
            }
        }
    }
    let failed = comparisons.iter().filter(|r| r["passed"] == false).count();
    Ok(
        json!({"chapter":5,"title":"Dimension- and curvature-aware native radial cap estimates","config":config,"coefficient_table":coefficient_table,"cases":cases,"comparisons":comparisons,"summary":{"cases":cases.len(),"native_steps":cases.len()*config.samples*config.steps*2,"comparisons":comparisons.len(),"comparisons_failed":failed},"gaps":["These are certified isotropic-quadratic kinetic bounds. No transfer to nonlinear forces, selected cloning or killed-chain law convergence is claimed."]}),
    )
}
fn itranspose(a: Matrix) -> Matrix {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}
fn endpoint_certificate(h: Matrix, beta: f64) -> (Interval, Vec<Value>) {
    endpoint_metric_certificate(h, 1., beta)
}
fn endpoint_metric_certificate(h: Matrix, alpha: f64, beta: f64) -> (Interval, Vec<Value>) {
    let one = Interval::point(1.);
    let diagonal = Interval::point(alpha);
    let b = Interval::point(beta);
    let g = [[diagonal, b], [b, one]];
    let gdet = diagonal.sub(b.square());
    let mut eigenvalues = vec![];
    let mut minimum = Interval::point(1.);
    for endpoint in 0..2 {
        let mut matrix = h;
        if endpoint == 0 {
            matrix[1] = [Interval::point(0.); 2];
        }
        let output = imul(imul(itranspose(matrix), g), matrix);
        let deficit: Matrix =
            std::array::from_fn(|i| std::array::from_fn(|j| g[i][j].sub(output[i][j])));
        let trace = deficit[0][0]
            .add(diagonal.mul(deficit[1][1]))
            .sub(b.mul(deficit[0][1].add(deficit[1][0])))
            .div(gdet);
        let determinant = deficit[0][0]
            .mul(deficit[1][1])
            .sub(deficit[0][1].mul(deficit[1][0]))
            .div(gdet);
        let eigen = smallest_eigenvalue(trace, determinant);
        minimum.lo = minimum.lo.min(eigen.lo);
        minimum.hi = minimum.hi.min(eigen.hi);
        eigenvalues.push(json!({"cap_secant_endpoint":endpoint,"deficit_interval":deficit.map(|row|row.map(|v|[v.lo,v.hi])),"generalized_minimum_eigenvalue":[eigen.lo,eigen.hi]}));
    }
    (minimum, eigenvalues)
}

/// Native radial-cap sector certificate in a cross-term metric.
/// Beta is chosen from a fixed deterministic family and each candidate is
/// checked by interval endpoint LMIs, never by measured trajectory rates.
pub fn sector_constants(h: f64, gamma: f64, omega: f64) -> Result<Value> {
    if ![h, gamma, omega].iter().all(|v| v.is_finite() && *v > 0.) || !(h * gamma).is_finite() {
        return Err(GasError::Configuration(
            "Sector certificate needs positive finite timestep, friction and curvature".into(),
        ));
    }
    let c = Interval::point(h).div(Interval::point(2.));
    let w = Interval::point(omega);
    let k = Interval::point(1.).sub(w.mul(c.square()));
    if k.lo <= 0. {
        return Err(GasError::Configuration(
            "Sector curvature metric requires k>0".into(),
        ));
    }
    let a = exp_negative(Interval::point(h).mul(Interval::point(gamma)));
    let ap = Interval::point(1.).add(a);
    let ax = Interval::point(1.).sub(w.mul(c.square()).mul(ap));
    let scaled_b = w.sqrt().mul(c).mul(ap);
    let matrix = [
        [ax, scaled_b],
        [
            Interval::point(0.).sub(scaled_b.mul(k)),
            a.sub(w.mul(c.square()).mul(ap)),
        ],
    ];
    let mut candidates = vec![0.04];
    if ax.lo > 0. {
        candidates.push(scaled_b.div(ax).midpoint().clamp(0., 0.9));
    }
    candidates.extend((0..=200).map(|i| i as f64 / 500.));
    let mut best = 0.;
    let mut beta = 0.;
    let mut certificate = vec![];
    for candidate in candidates {
        let (eigen, endpoints) = endpoint_certificate(matrix, candidate);
        if eigen.lo > best {
            best = eigen.lo;
            beta = candidate;
            certificate = endpoints;
        }
    }
    if best <= 0. {
        return Err(GasError::Configuration("No strict cap-sector endpoint LMI in the declared beta family; contraction is not assumed".into()));
    }
    let beta_i = Interval::point(beta);
    let equivalence_trace = Interval::point(1.).div(k).add(Interval::point(1.));
    let equivalence_det = Interval::point(1.).sub(beta_i.square()).div(k);
    let equivalence_lower = smallest_eigenvalue(equivalence_trace, equivalence_det).lo;
    let equivalence_upper = (equivalence_trace.hi - equivalence_lower).next_up();
    let (reference, reference_endpoints) = endpoint_certificate(matrix, 0.04);
    Ok(
        json!({"h":h,"gamma":gamma,"curvature":omega,"beta":beta,"metric":"omega|dx|²+2 beta sqrt(omega) dx.dv+|dv|²","scaled_native_matrix_interval":matrix.map(|row|row.map(|v|[v.lo,v.hi])),"delta_pathwise_lower":best,"endpoint_LMI_certificate":certificate,"metric_equivalence_to_Q_diagonal":{"lower":equivalence_lower,"upper":equivalence_upper,"prefactor":(equivalence_upper/equivalence_lower).next_up()},"reference_beta":0.04,"reference_exact_endpoint_delta_lower":reference.lo,"reference_endpoint_LMI_certificate":reference_endpoints,"RCAP_reference_delta":if (h-0.04).abs()<1e-15&&(gamma-1.).abs()<1e-15&&(omega-1.).abs()<1e-15 {Some(1./1040.)}else{None},"dimension_independent":true,"population_independent":true,"scope":"Actual isotropic-quadratic complete kinetic stage, shared additive innovations, no selected clone edges/death/count viscosity; native cap secant is symmetric in [0,I]. Fixed beta family optimized solely against certified analytic endpoint matrices."}),
    )
}

/// Curvature-adapted harmonic reference for an explicitly declared well profile.
/// The metric is G=[[k,beta],[beta,1]] in (sqrt(omega)x,v) coordinates.
/// It certifies the harmonic reference only; a regional force remainder and
/// actual probability/observable cost of leaving that region remain required.
pub fn sector_curvature_profile(h: f64, gamma: f64, omega: f64) -> Result<Value> {
    if ![h, gamma, omega].iter().all(|v| v.is_finite() && *v > 0.) || !(h * gamma).is_finite() {
        return Err(GasError::Configuration(
            "Curvature profile needs positive finite h,gamma,omega".into(),
        ));
    }
    let c = Interval::point(h).div(Interval::point(2.));
    let w = Interval::point(omega);
    let ki = Interval::point(1.).sub(w.mul(c.square()));
    if ki.lo <= 0. {
        return Err(GasError::Configuration(
            "Curvature-adapted harmonic certificate requires k>0".into(),
        ));
    }
    // The chosen diagonal is an exact f64 parameter, not an assumed exact k.
    let alpha = ki.midpoint();
    let a = exp_negative(Interval::point(h).mul(Interval::point(gamma)));
    let ap = Interval::point(1.).add(a);
    let ax = Interval::point(1.).sub(w.mul(c.square()).mul(ap));
    let b = w.sqrt().mul(c).mul(ap);
    let map = [
        [ax, b],
        [
            Interval::point(0.).sub(b.mul(ki)),
            a.sub(w.mul(c.square()).mul(ap)),
        ],
    ];
    let mut best = 0.;
    let mut beta = 0.;
    let mut certificate = vec![];
    for i in 0..=2000 {
        let candidate = alpha.sqrt() * 0.5 * i as f64 / 2000.;
        let (eigen, endpoints) = endpoint_metric_certificate(map, alpha, candidate);
        if eigen.lo > best {
            best = eigen.lo;
            beta = candidate;
            certificate = endpoints;
        }
    }
    if best <= 0. {
        return Err(GasError::Configuration("Declared well curvature has no verified positive harmonic endpoint LMI in this metric family".into()));
    }
    let ai = Interval::point(alpha);
    let bi = Interval::point(beta);
    let eqtrace = ai.div(ki).add(Interval::point(1.));
    let eqdet = ai.sub(bi.square()).div(ki);
    let eqlower = smallest_eigenvalue(eqtrace, eqdet).lo;
    let equpper = (eqtrace.hi - eqlower).next_up();
    Ok(
        json!({"h":h,"gamma":gamma,"curvature":omega,"alpha":alpha,"beta":beta,"source_labels":["cor-kinetic-regional-harmonic-reference"],"source_quotes":[source_equation("5.DIM10")],"metric_matrix_scaled":[[alpha,beta],[beta,1.]],"scaled_native_matrix_interval":map.map(|r|r.map(|x|[x.lo,x.hi])),"delta_pathwise_lower":best,"endpoint_LMI_certificate":certificate,"metric_equivalence_to_Q_diagonal":{"lower":eqlower,"upper":equpper,"prefactor":(equpper/eqlower).next_up()},"dimension_independent":true,"population_independent":true,"scope":"Harmonic reference at the explicitly declared regional curvature. The local nonlinear force remainder and Gaussian escape/observable charge are additional requirements; this is not a global Rastrigin contraction assertion."}),
    )
}
