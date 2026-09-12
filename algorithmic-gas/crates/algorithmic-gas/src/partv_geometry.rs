//! Reproducible planar geometry and conditional-fitness calculations for Part V.
//!
//! Coordinates are physical coordinates. Cell clipping never modifies the swarm.
//! Duplicate sites belong to the lowest input index; empty duplicate cells remain
//! in the output so slot identity is preserved. No perturbation is introduced.
use crate::{GasError, Result, error::require};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub type Mat2 = [[f64; 2]; 2];
fn identity() -> Mat2 {
    [[1., 0.], [0., 1.]]
}
fn one() -> f64 {
    1.
}
fn small() -> f64 {
    0.001
}
fn two() -> f64 {
    2.
}
fn floor() -> f64 {
    1e-6
}
fn seed_default() -> u64 {
    7
}
fn sample_default() -> usize {
    1024
}
fn resolution_default() -> usize {
    4
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MetricPolicy {
    Strict,
    #[default]
    Clipped,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Objective {
    #[default]
    Sphere,
    Quadratic {
        curvature: Mat2,
    },
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessInput {
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    pub velocities: Vec<[f64; 2]>,
    #[serde(default)]
    pub alive: Vec<bool>,
    pub companions: Vec<usize>,
    #[serde(default)]
    pub companion_valid: Vec<bool>,
    pub target: usize,
    pub query: [f64; 2],
    #[serde(default)]
    pub objective: Objective,
    #[serde(default = "small")]
    pub sigma_min: f64,
    #[serde(default = "small")]
    pub distance_floor: f64,
    #[serde(default)]
    pub velocity_weight: f64,
    #[serde(default = "one")]
    pub reward_exponent: f64,
    #[serde(default = "one")]
    pub distance_exponent: f64,
    #[serde(default = "two")]
    pub map_amplitude: f64,
    #[serde(default = "floor")]
    pub map_floor: f64,
    #[serde(default)]
    pub maximize: bool,
    #[serde(default = "one")]
    pub metric_epsilon: f64,
    #[serde(default)]
    pub metric_policy: MetricPolicy,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpaceTimeFrame {
    pub time: f64,
    pub points: Vec<[f64; 2]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GeometryRequest {
    Fitness {
        #[serde(flatten)]
        input: FitnessInput,
    },
    Metric {
        hessian: Mat2,
        epsilon: f64,
        #[serde(default)]
        policy: MetricPolicy,
    },
    Ou {
        metric: Mat2,
        gamma: f64,
        temperature: f64,
        dt: f64,
        #[serde(default = "sample_default")]
        samples: usize,
        #[serde(default = "seed_default")]
        seed: u64,
    },
    Voronoi {
        points: Vec<[f64; 2]>,
        bounds: [f64; 4],
        #[serde(default = "identity")]
        metric: Mat2,
    },
    Spacetime {
        frames: Vec<SpaceTimeFrame>,
        bounds: [f64; 4],
        #[serde(default = "resolution_default")]
        resolution: usize,
        #[serde(default = "identity")]
        metric: Mat2,
    },
    Triangulation {
        frames: Vec<Vec<[f64; 2]>>,
        #[serde(default = "identity")]
        metric: Mat2,
    },
    GraphDistance {
        points: Vec<[f64; 2]>,
        bounds: [f64; 4],
        resolution: usize,
        metric_grid: Vec<Mat2>,
    },
    Harmonic {
        curvature: Mat2,
        metric: Mat2,
        gamma: f64,
        temperature: f64,
        dt: f64,
    },
}
pub type GeometryResponse = Value;

/// Second order forward-mode jet; the Hessian is the actual second derivative,
/// not a Taylor coefficient. All population statistics participate in AD.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Jet2 {
    pub value: f64,
    pub gradient: [f64; 2],
    pub hessian: Mat2,
}
impl Jet2 {
    fn constant(value: f64) -> Self {
        Self {
            value,
            gradient: [0.; 2],
            hessian: [[0.; 2]; 2],
        }
    }
    fn variable(value: f64, axis: usize) -> Self {
        let mut x = Self::constant(value);
        x.gradient[axis] = 1.;
        x
    }
    fn chain(self, value: f64, first: f64, second: f64) -> Self {
        let mut out = Self::constant(value);
        for i in 0..2 {
            out.gradient[i] = first * self.gradient[i];
            for j in 0..2 {
                out.hessian[i][j] =
                    first * self.hessian[i][j] + second * self.gradient[i] * self.gradient[j];
            }
        }
        out
    }
    fn pow(self, p: f64) -> Self {
        if p == 0. {
            return Self::constant(1.);
        }
        if p == 1. {
            return self;
        }
        self.chain(
            self.value.powf(p),
            p * self.value.powf(p - 1.),
            p * (p - 1.) * self.value.powf(p - 2.),
        )
    }
    fn cos(self) -> Self {
        self.chain(self.value.cos(), -self.value.sin(), -self.value.cos())
    }
    fn logistic(self, amplitude: f64, floor: f64) -> Self {
        let s = if self.value >= 0. {
            1. / (1. + (-self.value).exp())
        } else {
            let e = self.value.exp();
            e / (1. + e)
        };
        self.chain(
            amplitude * s + floor,
            amplitude * s * (1. - s),
            amplitude * s * (1. - s) * (1. - 2. * s),
        )
    }
}
impl Add for Jet2 {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        let mut x = self;
        x.value += b.value;
        for i in 0..2 {
            x.gradient[i] += b.gradient[i];
            for j in 0..2 {
                x.hessian[i][j] += b.hessian[i][j];
            }
        }
        x
    }
}
impl Neg for Jet2 {
    type Output = Self;
    fn neg(self) -> Self {
        self * Jet2::constant(-1.)
    }
}
impl Sub for Jet2 {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + -b
    }
}
impl Mul for Jet2 {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        let mut x = Self::constant(self.value * b.value);
        for i in 0..2 {
            x.gradient[i] = self.gradient[i] * b.value + self.value * b.gradient[i];
            for j in 0..2 {
                x.hessian[i][j] = self.hessian[i][j] * b.value
                    + self.gradient[i] * b.gradient[j]
                    + self.gradient[j] * b.gradient[i]
                    + self.value * b.hessian[i][j];
            }
        }
        x
    }
}
impl Div for Jet2 {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        self * b.pow(-1.)
    }
}
fn objective(x: [Jet2; 2], o: &Objective) -> Jet2 {
    let c = Jet2::constant;
    let [a, b] = x;
    match o {
        Objective::Sphere => a * a + b * b,
        Objective::Quadratic { curvature: g } => {
            (c(g[0][0]) * a * a + c(2. * g[0][1]) * a * b + c(g[1][1]) * b * b) * c(0.5)
        }
        Objective::Rastrigin => {
            a * a + b * b + c(20.)
                - c(10.)
                    * ((a * c(std::f64::consts::TAU)).cos() + (b * c(std::f64::consts::TAU)).cos())
        }
        Objective::Rosenbrock => c(100.) * (b - a * a).pow(2.) + (c(1.) - a).pow(2.),
        Objective::StyblinskiTang => {
            (a.pow(4.) + b.pow(4.) - c(16.) * (a * a + b * b) + c(5.) * (a + b)) * c(0.5)
        }
    }
}
fn standardized(values: &[Jet2], alive: &[bool], target: usize, regularizer: f64) -> Jet2 {
    let mut mean = Jet2::constant(0.);
    let mut m2 = mean;
    let mut n = 0.;
    for (v, a) in values.iter().zip(alive) {
        if *a {
            n += 1.;
            let delta = *v - mean;
            mean = mean + delta / Jet2::constant(n);
            m2 = m2 + delta * (*v - mean);
        }
    }
    (values[target] - mean)
        / (m2 / Jet2::constant(n) + Jet2::constant(regularizer * regularizer)).pow(0.5)
}
/// Exact derivative on a frozen alive/companion stratum. The target coordinate
/// changes its own reward and separation. Donor coordinates remain the immutable
/// source snapshot even when another row selects the target slot.
pub fn conditional_fitness(input: &FitnessInput) -> Result<Jet2> {
    conditional_channels(
        input,
        [input.sigma_min, input.map_amplitude, input.map_floor],
        [input.sigma_min, input.map_amplitude, input.map_floor],
    )
}
/// Apply the engine's separate smooth global/logistic channel parameters.
pub fn conditional_fitness_pipeline(
    input: &FitnessInput,
    pipeline: &crate::fitness::FitnessPipeline,
) -> Result<Jet2> {
    use crate::fitness::{ObjectiveDirection, PositiveMap, Standardizer};
    pipeline.validate()?;
    let channel = |s: &Standardizer, m: &PositiveMap| -> Result<[f64; 3]> {
        match (s, m) {
            (Standardizer::Global { sigma_min }, PositiveMap::Logistic { amplitude, floor }) => {
                Ok([*sigma_min, *amplitude, *floor])
            }
            _ => Err(GasError::Capability(
                "conditional jets require smooth global standardizers and logistic maps".into(),
            )),
        }
    };
    let mut p = input.clone();
    p.reward_exponent = pipeline.reward_exponent;
    p.distance_exponent = pipeline.diversity_exponent;
    p.distance_floor = pipeline.distance_floor;
    p.maximize = pipeline.direction == ObjectiveDirection::Maximize;
    conditional_channels(
        &p,
        channel(&pipeline.reward_standardizer, &pipeline.reward_map)?,
        channel(&pipeline.diversity_standardizer, &pipeline.diversity_map)?,
    )
}
fn conditional_channels(
    input: &FitnessInput,
    reward: [f64; 3],
    diversity: [f64; 3],
) -> Result<Jet2> {
    let n = input.points.len();
    require(
        n > 0 && n <= 4096 && input.target < n && input.companions.len() == n,
        "fitness input shape/capacity",
    )?;
    require(
        input
            .points
            .iter()
            .flatten()
            .chain(input.query.iter())
            .all(|x| x.is_finite()),
        "finite fitness coordinates",
    )?;
    let alive = if input.alive.is_empty() {
        vec![true; n]
    } else {
        input.alive.clone()
    };
    require(
        alive.len() == n && alive[input.target],
        "target must be alive",
    )?;
    require(
        input.velocities.is_empty() || input.velocities.len() == n,
        "velocity shape",
    )?;
    require(
        input.velocities.iter().flatten().all(|x| x.is_finite()),
        "finite velocities",
    )?;
    require(
        [input.sigma_min, input.distance_floor, input.map_amplitude]
            .iter()
            .all(|x| x.is_finite() && *x > 0.)
            && [
                input.velocity_weight,
                input.reward_exponent,
                input.distance_exponent,
                input.map_floor,
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.),
        "smooth fitness parameters",
    )?;
    if let Objective::Quadratic { curvature } = &input.objective {
        validate_symmetric(*curvature)?;
    }
    require(
        input
            .companions
            .iter()
            .enumerate()
            .all(|(i, &j)| j < n && (!alive[i] || alive[j])),
        "same-frame alive companions required",
    )?;
    let mut x: Vec<_> = input.points.iter().map(|p| p.map(Jet2::constant)).collect();
    x[input.target] = [
        Jet2::variable(input.query[0], 0),
        Jet2::variable(input.query[1], 1),
    ];
    let sign = Jet2::constant(if input.maximize { 1. } else { -1. });
    let rewards: Vec<_> = x
        .iter()
        .map(|&p| objective(p, &input.objective) * sign)
        .collect();
    require(
        input.companion_valid.is_empty() || input.companion_valid.len() == n,
        "companion validity shape",
    )?;
    let mut distance = Vec::with_capacity(n);
    for (i, xi) in x.iter().enumerate() {
        if !input.companion_valid.is_empty() && !input.companion_valid[i] {
            distance.push(Jet2::constant(input.distance_floor));
            continue;
        }
        let j = input.companions[i];
        let dx = xi[0] - Jet2::constant(input.points[j][0]);
        let dy = xi[1] - Jet2::constant(input.points[j][1]);
        let dv = if input.velocities.is_empty() {
            0.
        } else {
            (input.velocities[i][0] - input.velocities[j][0]).powi(2)
                + (input.velocities[i][1] - input.velocities[j][1]).powi(2)
        };
        distance.push(
            (dx * dx
                + dy * dy
                + Jet2::constant(input.velocity_weight * dv + input.distance_floor.powi(2)))
            .pow(0.5),
        );
    }
    let r = standardized(&rewards, &alive, input.target, reward[0])
        .logistic(reward[1], reward[2])
        .pow(input.reward_exponent);
    let d = standardized(&distance, &alive, input.target, diversity[0])
        .logistic(diversity[1], diversity[2])
        .pow(input.distance_exponent);
    let out = r * d;
    require(
        out.value.is_finite()
            && out
                .gradient
                .iter()
                .chain(out.hessian.iter().flatten())
                .all(|x| x.is_finite()),
        "nonfinite fitness derivative",
    )?;
    Ok(out)
}
fn validate_symmetric(g: Mat2) -> Result<()> {
    require(
        g.iter().flatten().all(|x| x.is_finite())
            && (g[0][1] - g[1][0]).abs() <= 1e-12 * (1. + g[0][1].abs()),
        "finite symmetric 2x2 matrix required",
    )
}
fn eig(g: Mat2) -> ([f64; 2], Mat2) {
    let center = 0.5 * g[0][0] + 0.5 * g[1][1];
    let delta = 0.5 * g[0][0] - 0.5 * g[1][1];
    let radius = delta.hypot(g[0][1]);
    let theta = if radius == 0. {
        0.
    } else {
        0.5 * (2. * g[0][1]).atan2(g[0][0] - g[1][1])
    };
    let (s, c) = theta.sin_cos();
    ([center + radius, center - radius], [[c, -s], [s, c]])
}
fn spectral(q: Mat2, l: [f64; 2]) -> Mat2 {
    let mut g = [[0.; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            g[i][j] = q[i][0] * l[0] * q[j][0] + q[i][1] * l[1] * q[j][1];
        }
    }
    g
}
#[derive(Clone, Debug, Serialize)]
pub struct MetricResult {
    pub metric: Mat2,
    pub inverse: Mat2,
    pub inverse_sqrt: Mat2,
    pub hessian_eigenvalues: [f64; 2],
    pub metric_eigenvalues: [f64; 2],
    pub eigenvectors: Mat2,
    pub clipped: [bool; 2],
    pub volume_density: f64,
    pub condition_number: f64,
}
pub fn metric_from_hessian(h: Mat2, epsilon: f64, policy: MetricPolicy) -> Result<MetricResult> {
    validate_symmetric(h)?;
    require(
        epsilon.is_finite() && epsilon > 0.,
        "positive metric epsilon required",
    )?;
    let (hl, q) = eig(h);
    let mut l = [hl[0] + epsilon, hl[1] + epsilon];
    let mut clipped = [false; 2];
    for k in 0..2 {
        match policy {
            MetricPolicy::Strict => {
                require(l[k] > 0., "strict H + epsilon I is not positive definite")?
            }
            MetricPolicy::Clipped => {
                clipped[k] = l[k] < epsilon;
                l[k] = l[k].max(epsilon);
            }
        }
    }
    require(
        l.iter().all(|x| x.is_finite() && *x > 0.),
        "metric overflow",
    )?;
    metric_result(hl, l, q, clipped)
}
fn metric_result(hl: [f64; 2], l: [f64; 2], q: Mat2, clipped: [bool; 2]) -> Result<MetricResult> {
    let result = MetricResult {
        metric: spectral(q, l),
        inverse: spectral(q, l.map(|x| 1. / x)),
        inverse_sqrt: spectral(q, l.map(|x| 1. / x.sqrt())),
        hessian_eigenvalues: hl,
        metric_eigenvalues: l,
        eigenvectors: q,
        clipped,
        volume_density: l[0].sqrt() * l[1].sqrt(),
        condition_number: l[0].max(l[1]) / l[0].min(l[1]),
    };
    require(
        result
            .inverse
            .iter()
            .flatten()
            .chain(result.inverse_sqrt.iter().flatten())
            .chain([&result.volume_density, &result.condition_number])
            .all(|x| x.is_finite()),
        "metric inverse/volume is not representable",
    )?;
    Ok(result)
}
fn spd(g: Mat2) -> Result<MetricResult> {
    validate_symmetric(g)?;
    let (l, q) = eig(g);
    require(
        l.iter().all(|x| x.is_finite() && *x > 0.),
        "metric must be positive definite",
    )?;
    metric_result(l, l, q, [false; 2])
}
fn dot_metric(x: [f64; 2], y: [f64; 2], g: Mat2) -> f64 {
    x[0] * (g[0][0] * y[0] + g[0][1] * y[1]) + x[1] * (g[1][0] * y[0] + g[1][1] * y[1])
}
fn validate_domain(points: &[[f64; 2]], b: [f64; 4], g: Mat2) -> Result<()> {
    spd(g)?;
    require(
        !points.is_empty()
            && points.len() <= 512
            && points
                .iter()
                .flatten()
                .chain(b.iter())
                .all(|x| x.is_finite())
            && b[0] < b[1]
            && b[2] < b[3],
        "finite nonempty planar domain with at most 512 sites required",
    )
}
fn clip_polygon(poly: &[[f64; 2]], normal: [f64; 2], offset: f64) -> Vec<[f64; 2]> {
    let mut result = Vec::new();
    if poly.is_empty() {
        return result;
    }
    for k in 0..poly.len() {
        let a = poly[k];
        let b = poly[(k + 1) % poly.len()];
        let da = a[0] * normal[0] + a[1] * normal[1] - offset;
        let db = b[0] * normal[0] + b[1] * normal[1] - offset;
        if da <= 0. {
            result.push(a);
        }
        if (da < 0. && db > 0.) || (da > 0. && db < 0.) {
            let t = da / (da - db);
            result.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    result
}
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    pub slot: usize,
    pub vertices: Vec<[f64; 2]>,
    pub area: f64,
    pub geometric_area: f64,
    pub duplicate_of: Option<usize>,
}
#[derive(Clone, Debug, Serialize)]
pub struct PlanarMesh {
    pub cells: Vec<Cell>,
    pub neighbors: Vec<[usize; 2]>,
    pub total_area: f64,
    pub expected_area: f64,
    pub closure_error: f64,
    pub metric: Mat2,
    pub construction: &'static str,
}
pub fn voronoi(points: &[[f64; 2]], bounds: [f64; 4], metric: Mat2) -> Result<PlanarMesh> {
    validate_domain(points, bounds, metric)?;
    let density = spd(metric)?.volume_density;
    let mut cells = Vec::new();
    let scale = (bounds[1] - bounds[0]).max(bounds[3] - bounds[2]);
    let tol = 1e-9 * (1. + scale);
    for (i, &p) in points.iter().enumerate() {
        let duplicate = (0..i).find(|&j| points[j] == p);
        let mut poly = if duplicate.is_some() {
            vec![]
        } else {
            vec![
                [bounds[0], bounds[2]],
                [bounds[1], bounds[2]],
                [bounds[1], bounds[3]],
                [bounds[0], bounds[3]],
            ]
        };
        for (j, &q) in points.iter().enumerate() {
            if i == j || q == p {
                continue;
            }
            let d = [q[0] - p[0], q[1] - p[1]];
            let normal = [
                2. * (metric[0][0] * d[0] + metric[0][1] * d[1]),
                2. * (metric[1][0] * d[0] + metric[1][1] * d[1]),
            ];
            let offset = dot_metric(q, q, metric) - dot_metric(p, p, metric);
            poly = clip_polygon(&poly, normal, offset);
        }
        let mut twice = 0.;
        for k in 0..poly.len() {
            let a = poly[k];
            let b = poly[(k + 1) % poly.len()];
            twice += a[0] * b[1] - a[1] * b[0];
        }
        let area = twice.abs() * 0.5;
        cells.push(Cell {
            slot: i,
            vertices: poly,
            area,
            geometric_area: area * density,
            duplicate_of: duplicate,
        });
    }
    let mut neighbors = Vec::new();
    for i in 0..points.len() {
        if cells[i].area == 0. {
            continue;
        }
        for j in i + 1..points.len() {
            if cells[j].area == 0. {
                continue;
            }
            let p = points[i];
            let q = points[j];
            let mut shared = Vec::new();
            for &a in &cells[i].vertices {
                let dp = dot_metric(
                    [a[0] - p[0], a[1] - p[1]],
                    [a[0] - p[0], a[1] - p[1]],
                    metric,
                );
                let dq = dot_metric(
                    [a[0] - q[0], a[1] - q[1]],
                    [a[0] - q[0], a[1] - q[1]],
                    metric,
                );
                if (dp - dq).abs() <= tol * (1. + dp.abs() + dq.abs()) {
                    shared.push(a);
                }
            }
            if shared.iter().enumerate().any(|(k, a)| {
                shared[k + 1..]
                    .iter()
                    .any(|b| (a[0] - b[0]).hypot(a[1] - b[1]) > tol)
            }) {
                neighbors.push([i, j]);
            }
        }
    }
    let total_area = cells.iter().map(|c| c.area).sum::<f64>();
    let expected_area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]);
    Ok(PlanarMesh {
        cells,
        neighbors,
        total_area,
        expected_area,
        closure_error: total_area - expected_area,
        metric,
        construction: "constant_metric_half_plane_clipping",
    })
}
#[derive(Clone)]
struct Vertex3 {
    p: [f64; 3],
    scores: Vec<f64>,
}
type Polyhedron = Vec<Vec<Vertex3>>;
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn clip_polyhedron(poly: Polyhedron, i: usize, j: usize) -> Polyhedron {
    let mut faces = Vec::new();
    let mut cap: Vec<Vertex3> = Vec::new();
    let has_outside = poly.iter().flatten().any(|v| v.scores[i] > v.scores[j]);
    let has_inside = poly.iter().flatten().any(|v| v.scores[i] < v.scores[j]);
    for face in poly {
        let mut out = Vec::new();
        for k in 0..face.len() {
            let a = &face[k];
            let b = &face[(k + 1) % face.len()];
            let da = a.scores[i] - a.scores[j];
            let db = b.scores[i] - b.scores[j];
            if da <= 0. {
                out.push(a.clone());
            }
            if da == 0.
                && !cap
                    .iter()
                    .any(|w| dot3(sub3(a.p, w.p), sub3(a.p, w.p)) < 1e-24)
            {
                cap.push(a.clone());
            }
            if (da < 0. && db > 0.) || (da > 0. && db < 0.) {
                let t = da / (da - db);
                let v = Vertex3 {
                    p: std::array::from_fn(|k| a.p[k] + t * (b.p[k] - a.p[k])),
                    scores: a
                        .scores
                        .iter()
                        .zip(&b.scores)
                        .map(|(a, b)| a + t * (b - a))
                        .collect(),
                };
                if !cap
                    .iter()
                    .any(|w| dot3(sub3(v.p, w.p), sub3(v.p, w.p)) < 1e-24)
                {
                    cap.push(v.clone());
                }
                out.push(v);
            }
        }
        if out.len() >= 3 {
            faces.push(out);
        }
    }
    if cap.len() >= 3 && has_inside && has_outside {
        let mut center = [0.; 3];
        for v in &cap {
            for (k, c) in center.iter_mut().enumerate() {
                *c += v.p[k] / cap.len() as f64;
            }
        }
        let axis = sub3(cap[0].p, center);
        let normal = cap
            .iter()
            .skip(1)
            .map(|v| cross(axis, sub3(v.p, center)))
            .find(|n| dot3(*n, *n) > 1e-28);
        if let Some(n) = normal {
            let tangent = cross(n, axis);
            let al = dot3(axis, axis).sqrt();
            let tl = dot3(tangent, tangent).sqrt();
            cap.sort_by(|a, b| {
                let a = sub3(a.p, center);
                let b = sub3(b.p, center);
                (dot3(a, tangent) / tl)
                    .atan2(dot3(a, axis) / al)
                    .total_cmp(&(dot3(b, tangent) / tl).atan2(dot3(b, axis) / al))
            });
            faces.push(cap);
        }
    }
    faces
}
fn poly_volume(poly: &Polyhedron) -> f64 {
    let mut center = [0.; 3];
    let count = poly.iter().map(Vec::len).sum::<usize>();
    if count == 0 {
        return 0.;
    }
    for face in poly {
        for v in face {
            for (k, c) in center.iter_mut().enumerate() {
                *c += v.p[k] / count as f64;
            }
        }
    }
    let mut volume = 0.;
    for face in poly {
        for k in 1..face.len() - 1 {
            volume += dot3(
                sub3(face[0].p, center),
                cross(sub3(face[k].p, center), sub3(face[k + 1].p, center)),
            )
            .abs()
                / 6.;
        }
    }
    volume
}
/// A single conforming tetrahedral grid is shared by every cell. Nearest-site
/// score differences are interpolated on each tetrahedron and clipped exactly
/// for that piecewise-affine interpolant. Refinement controls the moving-site
/// approximation; cell polygons are never matched vertex by vertex.
pub fn spacetime(
    frames: &[SpaceTimeFrame],
    bounds: [f64; 4],
    resolution: usize,
    metric: Mat2,
) -> Result<Value> {
    require(
        frames.len() >= 2 && frames.len() <= 32 && resolution > 0 && resolution <= 16,
        "spacetime frame/resolution bounds",
    )?;
    let n = frames[0].points.len();
    require(n <= 32, "spacetime supports at most 32 sites")?;
    for f in frames {
        validate_domain(&f.points, bounds, metric)?;
        require(
            f.points.len() == n && f.time.is_finite(),
            "spacetime frame shape",
        )?;
    }
    require(
        frames.windows(2).all(|w| w[0].time <= w[1].time),
        "spacetime times must be nondecreasing",
    )?;
    let work = (frames.len() - 1) * resolution.pow(3) * 6 * n * n;
    require(
        work <= 8_000_000,
        "spacetime analysis budget exceeded; reduce sites or resolution",
    )?;
    let mut pieces = Vec::new();
    let mut volumes = vec![0.; n];
    let mut caps = Vec::new();
    let mut expected = 0.;
    // Freudenthal triangulation: every cube uses the same 000-to-111 diagonal.
    let tets = [
        [0, 1, 3, 7],
        [0, 1, 5, 7],
        [0, 2, 3, 7],
        [0, 2, 6, 7],
        [0, 4, 5, 7],
        [0, 4, 6, 7],
    ];
    for (slab, w) in frames.windows(2).enumerate() {
        let a = &w[0];
        let b = &w[1];
        let duration = b.time - a.time;
        if duration == 0. {
            caps.push(json!({"time":a.time,"before":voronoi(&a.points,bounds,metric)?,"after":voronoi(&b.points,bounds,metric)?,"kind":"one_sided_jump_caps"}));
            continue;
        }
        expected += (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * duration;
        for ix in 0..resolution {
            for iy in 0..resolution {
                for it in 0..resolution {
                    let mut vertices = Vec::new();
                    for bits in 0..8 {
                        let u = (ix + (bits & 1)) as f64 / resolution as f64;
                        let v = (iy + ((bits >> 1) & 1)) as f64 / resolution as f64;
                        let s = (it + ((bits >> 2) & 1)) as f64 / resolution as f64;
                        let p = [
                            bounds[0] + u * (bounds[1] - bounds[0]),
                            bounds[2] + v * (bounds[3] - bounds[2]),
                            a.time + s * duration,
                        ];
                        let scores = a
                            .points
                            .iter()
                            .zip(&b.points)
                            .map(|(a, b)| {
                                let d = [
                                    p[0] - (a[0] + s * (b[0] - a[0])),
                                    p[1] - (a[1] + s * (b[1] - a[1])),
                                ];
                                dot_metric(d, d, metric)
                            })
                            .collect();
                        vertices.push(Vertex3 { p, scores });
                    }
                    for tet in tets {
                        let base: Polyhedron = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
                            .iter()
                            .map(|f| f.iter().map(|&k| vertices[tet[k]].clone()).collect())
                            .collect();
                        for (i, slot_volume) in volumes.iter_mut().enumerate() {
                            if (0..i)
                                .any(|j| a.points[i] == a.points[j] && b.points[i] == b.points[j])
                            {
                                continue;
                            }
                            let mut poly = base.clone();
                            for j in 0..n {
                                if i != j {
                                    poly = clip_polyhedron(poly, i, j);
                                    if poly.is_empty() {
                                        break;
                                    }
                                }
                            }
                            let volume = poly_volume(&poly);
                            if volume > 1e-18 {
                                *slot_volume += volume;
                                require(pieces.len() < 40000, "spacetime output budget exceeded")?;
                                pieces.push(json!({"slot":i,"slab":slab,"volume":volume,"faces":poly.iter().map(|f|f.iter().map(|v|v.p).collect::<Vec<_>>()).collect::<Vec<_>>()}));
                            }
                        }
                    }
                }
            }
        }
    }
    let total: f64 = volumes.iter().sum();
    let boundary_faces = boundary_surfaces(
        &pieces,
        bounds,
        frames.last().unwrap().time - frames[0].time,
    );
    Ok(
        json!({"pieces":pieces,"boundary_faces":boundary_faces,"face_matching_relative_tolerance":1e-10,"slot_volumes":volumes,"geometric_slot_volumes":volumes.iter().map(|v|v*spd(metric).unwrap().volume_density).collect::<Vec<_>>(),"geometric_total_volume":total*spd(metric)?.volume_density,"geometric_expected_volume":expected*spd(metric)?.volume_density,"volume_measure":"coordinate_dx_dy_dt; geometric fields use sqrt(det(metric))*dx_dy_dt","total_volume":total,"expected_volume":expected,"closure_error":total-expected,"jump_caps":caps,"resolution":resolution,"construction":"shared_tetrahedra_affine_score_partition","time_coordinate":"physical_time","interpolation":"linear_site_motion_between_distinct_times"}),
    )
}
pub fn ou(
    metric: Mat2,
    gamma: f64,
    temperature: f64,
    dt: f64,
    samples: usize,
    seed: u64,
) -> Result<Value> {
    let m = spd(metric)?;
    require(
        [gamma, temperature, dt]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.)
            && (2..=100_000).contains(&samples),
        "OU parameters/sample bounds",
    )?;
    let factor = temperature * (-(-2. * gamma * dt).exp_m1());
    let expected = m.inverse.map(|row| row.map(|x| x * factor));
    let root = factor.sqrt();
    let mut mean = [0.; 2];
    let mut m2 = [[0.; 2]; 2];
    let mut cloud = Vec::new();
    for k in 0..samples {
        let mut rng =
            crate::random::RandomStream::new(seed, 1, crate::random::Stream::Kinetic, k as u64, 2);
        let z = [rng.gaussian::<f64>(), rng.gaussian::<f64>()];
        let x: [f64; 2] = std::array::from_fn(|i| {
            root * (m.inverse_sqrt[i][0] * z[0] + m.inverse_sqrt[i][1] * z[1])
        });
        let delta = [x[0] - mean[0], x[1] - mean[1]];
        for i in 0..2 {
            mean[i] += delta[i] / (k + 1) as f64;
        }
        for i in 0..2 {
            for j in 0..2 {
                m2[i][j] += delta[i] * (x[j] - mean[j]);
            }
        }
        if cloud.len() < 4096 {
            cloud.push(x);
        }
    }
    let covariance = m2.map(|r| r.map(|x| x / (samples - 1) as f64));
    let se = std::array::from_fn::<_, 2, _>(|i| {
        std::array::from_fn::<_, 2, _>(|j| {
            ((expected[i][i] * expected[j][j] + expected[i][j].powi(2)) / (samples - 1) as f64)
                .sqrt()
        })
    });
    Ok(
        json!({"expected_covariance":expected,"sample_covariance":covariance,"covariance_standard_error":se,"mean":mean,"samples":samples,"cloud":cloud,"decay":(-gamma*dt).exp(),"prefactor":factor,"noise_factor":m.inverse_sqrt,"source":"engine_RandomStream_frozen_O_stage_innovations","seed":seed,"rng_step":1,"rng_substep":2}),
    )
}
type Mat4 = [[f64; 4]; 4];
fn eye4() -> Mat4 {
    std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1. } else { 0. }))
}
fn mul4(a: Mat4, b: Mat4) -> Mat4 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..4).map(|k| a[i][k] * b[k][j]).sum()))
}
fn transpose4(a: Mat4) -> Mat4 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
#[allow(clippy::needless_range_loop)] // Row elimination is clearest in matrix coordinates.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let pivot = (i..n)
            .max_by(|&j, &k| a[j][i].abs().total_cmp(&a[k][i].abs()))
            .unwrap();
        require(a[pivot][i].abs() > 1e-14, "singular Lyapunov system")?;
        a.swap(i, pivot);
        b.swap(i, pivot);
        let d = a[i][i];
        for k in i..n {
            a[i][k] /= d;
        }
        b[i] /= d;
        for j in 0..n {
            if i != j {
                let c = a[j][i];
                for k in i..n {
                    a[j][k] -= c * a[i][k];
                }
                b[j] -= c * b[i];
            }
        }
    }
    Ok(b)
}
fn lyapunov(a: Mat4, q: Mat4, discrete: bool) -> Result<Mat4> {
    let mut m = vec![vec![0.; 16]; 16];
    let mut rhs = vec![0.; 16];
    for i in 0..4 {
        for j in 0..4 {
            let row = i * 4 + j;
            rhs[row] = q[i][j];
            for k in 0..4 {
                for l in 0..4 {
                    m[row][k * 4 + l] = if discrete {
                        (if i == k && j == l { 1. } else { 0. }) - a[i][k] * a[j][l]
                    } else {
                        -a[i][k] * (if j == l { 1. } else { 0. })
                            - (if i == k { 1. } else { 0. }) * a[j][l]
                    };
                }
            }
        }
    }
    let x = solve(m, rhs)?;
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| x[i * 4 + j])
    }))
}
/// Both covariance references use force -curvature*x and O-stage covariance
/// T(1-exp(-2 gamma h))*metric^-1. Arbitrary rotated SPD matrices are allowed.
pub fn harmonic(
    curvature: Mat2,
    metric: Mat2,
    gamma: f64,
    temperature: f64,
    dt: f64,
) -> Result<Value> {
    spd(curvature)?;
    let m = spd(metric)?;
    require(
        [gamma, temperature, dt]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
        "positive harmonic parameters",
    )?;
    let (e, _) = eig(curvature);
    require(
        dt * dt * e[0] < 4.,
        "BAOAB harmonic step is outside its stability interval",
    )?;
    let mut a = eye4();
    let mut b = eye4();
    let mut o = eye4();
    for i in 0..2 {
        a[i][i + 2] = dt / 2.;
        o[i + 2][i + 2] = (-gamma * dt).exp();
        for j in 0..2 {
            b[i + 2][j] = -dt * curvature[i][j] / 2.;
        }
    }
    let after = mul4(b, a);
    let transition = mul4(after, mul4(o, mul4(a, b)));
    let mut qo = [[0.; 4]; 4];
    let mut drift = [[0.; 4]; 4];
    let mut qc = [[0.; 4]; 4];
    for i in 0..2 {
        drift[i][i + 2] = 1.;
        drift[i + 2][i + 2] = -gamma;
        for j in 0..2 {
            qo[i + 2][j + 2] = temperature * (-(-2. * gamma * dt).exp_m1()) * m.inverse[i][j];
            drift[i + 2][j] = -curvature[i][j];
            qc[i + 2][j + 2] = 2. * gamma * temperature * m.inverse[i][j];
        }
    }
    let q = mul4(after, mul4(qo, transpose4(after)));
    let discrete = lyapunov(transition, q, true)?;
    let continuous = lyapunov(drift, qc, false)?;
    let propagated = mul4(transition, mul4(discrete, transpose4(transition)));
    let mut residual: f64 = 0.;
    for i in 0..4 {
        for j in 0..4 {
            residual = residual.max((discrete[i][j] - propagated[i][j] - q[i][j]).abs());
        }
    }
    Ok(
        json!({"continuous_covariance":continuous,"discrete_covariance":discrete,"transition":transition,"innovation_covariance":q,"lyapunov_residual":residual,"coordinate_order":["x","y","vx","vy"],"reference":"constant_metric_harmonic_BAOAB"}),
    )
}
pub fn analyze(request: GeometryRequest) -> Result<GeometryResponse> {
    match request {
        GeometryRequest::Triangulation { frames, metric } => triangulation(&frames, metric),
        GeometryRequest::GraphDistance {
            points,
            bounds,
            resolution,
            metric_grid,
        } => graph_distance(&points, bounds, resolution, &metric_grid),
        GeometryRequest::Fitness { input } => {
            let jet = conditional_fitness(&input)?;
            let m = metric_from_hessian(jet.hessian, input.metric_epsilon, input.metric_policy)?;
            Ok(
                json!({"fitness":jet.value,"gradient":jet.gradient,"hessian":jet.hessian,"metric":m,"target":input.target,"query":input.query,"stratum":"alive_and_same_frame_companion_source_coordinates_frozen","standardizer":"global_population_variance_plus_sigma_min_squared","map":"logistic"}),
            )
        }
        GeometryRequest::Metric {
            hessian,
            epsilon,
            policy,
        } => Ok(json!(metric_from_hessian(hessian, epsilon, policy)?)),
        GeometryRequest::Ou {
            metric,
            gamma,
            temperature,
            dt,
            samples,
            seed,
        } => ou(metric, gamma, temperature, dt, samples, seed),
        GeometryRequest::Voronoi {
            points,
            bounds,
            metric,
        } => Ok(json!(voronoi(&points, bounds, metric)?)),
        GeometryRequest::Spacetime {
            frames,
            bounds,
            resolution,
            metric,
        } => spacetime(&frames, bounds, resolution, metric),
        GeometryRequest::Harmonic {
            curvature,
            metric,
            gamma,
            temperature,
            dt,
        } => harmonic(curvature, metric, gamma, temperature, dt),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> FitnessInput {
        serde_json::from_value(json!({"points":[[-1.,0.3],[0.2,0.8],[0.9,-0.4]],"companions":[1,2,0],"target":0,"query":[-0.7,0.4],"sigma_min":0.2,"distance_floor":0.1})).unwrap()
    }
    #[test]
    fn conditional_jet_matches_independent_finite_differences() {
        for objective in [
            Objective::Sphere,
            Objective::Rastrigin,
            Objective::Rosenbrock,
            Objective::StyblinskiTang,
            Objective::Quadratic {
                curvature: [[2., 0.3], [0.3, 1.]],
            },
        ] {
            let mut p = input();
            p.objective = objective;
            let j = conditional_fitness(&p).unwrap();
            let h = 1e-4;
            for axis in 0..2 {
                let mut lo = p.clone();
                let mut hi = p.clone();
                lo.query[axis] -= h;
                hi.query[axis] += h;
                let a = conditional_fitness(&lo).unwrap();
                let b = conditional_fitness(&hi).unwrap();
                assert!((j.gradient[axis] - (b.value - a.value) / (2. * h)).abs() < 2e-4);
                for other in 0..2 {
                    assert!(
                        (j.hessian[other][axis]
                            - (b.gradient[other] - a.gradient[other]) / (2. * h))
                            .abs()
                            < 2e-3
                    );
                }
            }
        }
    }
    #[test]
    fn coincident_companion_has_finite_jet() {
        let mut p = input();
        p.companions[0] = 0;
        let j = conditional_fitness(&p).unwrap();
        assert!(j.hessian.iter().flatten().all(|x| x.is_finite()));
    }
    #[test]
    fn singleton_fitness_is_constant() {
        let mut p = input();
        p.alive = vec![true, false, false];
        p.companions[0] = 0;
        let j = conditional_fitness(&p).unwrap();
        assert_eq!(j.gradient, [0., 0.]);
        assert_eq!(j.hessian, [[0.; 2]; 2]);
    }
    #[test]
    fn metric_spectral_identity_and_strict_failure() {
        for h in [
            [[2., 0.7], [0.7, -0.4]],
            [[2., 0.], [0., 2.]],
            [[0., 0.], [0., 0.]],
        ] {
            let m = metric_from_hessian(h, 0.1, MetricPolicy::Clipped).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    let v = (0..2)
                        .map(|k| m.metric[i][k] * m.inverse[k][j])
                        .sum::<f64>();
                    assert!((v - if i == j { 1. } else { 0. }).abs() < 1e-12);
                    let v = (0..2)
                        .map(|k| m.inverse_sqrt[i][k] * m.inverse_sqrt[k][j])
                        .sum::<f64>();
                    assert!((v - m.inverse[i][j]).abs() < 1e-12);
                }
            }
        }
        assert!(metric_from_hessian([[-2., 0.], [0., 1.]], 1., MetricPolicy::Strict).is_err());
    }
    #[test]
    fn planar_cells_close_and_preserve_duplicate_slots() {
        let p = [[-0.6, -0.2], [0.7, -0.2], [0.2, 0.8], [0.2, 0.8]];
        for g in [identity(), [[2., 0.6], [0.6, 1.]]] {
            let m = voronoi(&p, [-1., 1., -1., 1.], g).unwrap();
            assert!(m.closure_error.abs() < 1e-12);
            assert_eq!(m.cells[3].duplicate_of, Some(2));
            assert_eq!(m.cells[3].area, 0.);
            assert_eq!(m.neighbors.len(), 3);
        }
    }
    #[test]
    fn regular_square_has_no_zero_length_diagonal_neighbor() {
        let m = voronoi(
            &[[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            [-1., 1., -1., 1.],
            identity(),
        )
        .unwrap();
        assert_eq!(m.neighbors.len(), 4);
    }
    #[test]
    fn spacetime_shared_partition_and_jump_caps_close() {
        let frames = vec![
            SpaceTimeFrame {
                time: 0.,
                points: vec![[-0.5, 0.], [0.5, 0.]],
            },
            SpaceTimeFrame {
                time: 1.,
                points: vec![[-0.2, 0.4], [0.5, -0.3]],
            },
            SpaceTimeFrame {
                time: 1.,
                points: vec![[0.1, 0.], [0.5, -0.3]],
            },
            SpaceTimeFrame {
                time: 2.,
                points: vec![[0.2, 0.3], [0.7, -0.1]],
            },
        ];
        let r = spacetime(&frames, [-1., 1., -1., 1.], 2, identity()).unwrap();
        assert!(r["closure_error"].as_f64().unwrap().abs() < 1e-10);
        assert_eq!(r["jump_caps"].as_array().unwrap().len(), 1);
        assert_eq!(r["expected_volume"], 8.);
    }
    #[test]
    fn static_spacetime_volume_matches_cell_area() {
        let f = SpaceTimeFrame {
            time: 0.,
            points: vec![[-0.7, 0.1], [0.3, 0.2], [0.1, -0.7]],
        };
        let mut end = f.clone();
        end.time = 1.7;
        let m = voronoi(&f.points, [-1., 1., -1., 1.], identity()).unwrap();
        let r = spacetime(&[f, end], [-1., 1., -1., 1.], 2, identity()).unwrap();
        for (i, c) in m.cells.iter().enumerate() {
            assert!((r["slot_volumes"][i].as_f64().unwrap() - 1.7 * c.area).abs() < 1e-10);
        }
    }
    #[test]
    fn ou_covariance_matches_gaussian_reference() {
        let r = ou([[2., 0.5], [0.5, 1.]], 1.2, 0.7, 0.2, 50000, 7).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let e = r["expected_covariance"][i][j].as_f64().unwrap();
                let actual = r["sample_covariance"][i][j].as_f64().unwrap();
                let se = r["covariance_standard_error"][i][j].as_f64().unwrap();
                assert!((actual - e).abs() < 4. * se);
            }
        }
    }
    #[test]
    fn harmonic_isotropic_reference_and_rotated_residual() {
        let r = harmonic([[2., 0.], [0., 3.]], identity(), 1., 0.7, 0.2).unwrap();
        for i in 0..2 {
            let k = if i == 0 { 2. } else { 3. };
            assert!((r["continuous_covariance"][i][i].as_f64().unwrap() - 0.7 / k).abs() < 1e-12);
            assert!((r["discrete_covariance"][i][i].as_f64().unwrap() - 0.7 / k).abs() < 1e-12);
            assert!(
                (r["discrete_covariance"][i + 2][i + 2].as_f64().unwrap()
                    - 0.7 * (1. - 0.2 * 0.2 * k / 4.))
                    .abs()
                    < 1e-12
            );
        }
        let r = harmonic(
            [[2., 0.7], [0.7, 3.]],
            [[1.3, -0.4], [-0.4, 2.]],
            0.8,
            1.,
            0.1,
        )
        .unwrap();
        assert!(r["lyapunov_residual"].as_f64().unwrap() < 1e-12);
    }
}

#[derive(Clone, Debug)]
struct Site {
    point: spade::Point2<f64>,
    slot: usize,
}
impl spade::HasPosition for Site {
    type Scalar = f64;
    fn position(&self) -> spade::Point2<f64> {
        self.point
    }
}
fn mapped_site(p: [f64; 2], slot: usize, g: Mat2) -> Site {
    let (l, q) = eig(g);
    Site {
        point: spade::Point2::new(
            l[0].sqrt() * (q[0][0] * p[0] + q[1][0] * p[1]),
            l[1].sqrt() * (q[0][1] * p[0] + q[1][1] * p[1]),
        ),
        slot,
    }
}
fn triangulation_edges(
    t: &spade::DelaunayTriangulation<Site>,
) -> std::collections::BTreeSet<[usize; 2]> {
    use spade::Triangulation;
    t.undirected_edges()
        .map(|e| {
            let [a, b] = e.vertices();
            let mut pair = [a.data().slot, b.data().slot];
            pair.sort();
            pair
        })
        .collect()
}
/// Compare actual Spade removal/insertion maintenance with an independent
/// rebuild. Cocircular diagonals may differ; their lists are returned explicitly.
pub fn triangulation(frames: &[Vec<[f64; 2]>], metric: Mat2) -> Result<Value> {
    use spade::Triangulation;
    spd(metric)?;
    require(
        !frames.is_empty() && frames.len() <= 256,
        "triangulation frame bound",
    )?;
    let mut incremental = spade::DelaunayTriangulation::<Site>::new();
    let mut records = Vec::new();
    let mut previous = std::collections::BTreeSet::new();
    for (frame, points) in frames.iter().enumerate() {
        require(
            !points.is_empty()
                && points.len() <= 512
                && points.iter().flatten().all(|x| x.is_finite()),
            "triangulation point shape",
        )?;
        let sites: Vec<_> = points
            .iter()
            .enumerate()
            .filter(|(i, p)| !(0..*i).any(|j| points[j] == **p))
            .map(|(i, &p)| mapped_site(p, i, metric))
            .collect();
        require(
            sites
                .iter()
                .enumerate()
                .all(|(i, s)| sites[..i].iter().all(|p| p.point != s.point)),
            "distinct sites collide under the finite-precision metric transform",
        )?;
        let mut removals = 0;
        let mut insertions = 0;
        let mut search_visits = 0;
        loop {
            let obsolete = incremental.vertices().find_map(|v| {
                search_visits += 1;
                let d = v.data();
                (!sites.iter().any(|s| s.slot == d.slot && s.point == d.point)).then_some(v.fix())
            });
            if let Some(h) = obsolete {
                incremental.remove(h);
                removals += 1;
            } else {
                break;
            }
        }
        for site in &sites {
            if !incremental.vertices().any(|v| {
                search_visits += 1;
                v.data().slot == site.slot
            }) {
                incremental
                    .insert(site.clone())
                    .map_err(|e| GasError::Numerical(format!("Spade insertion: {e:?}")))?;
                insertions += 1;
            }
        }
        let mut rebuilt = spade::DelaunayTriangulation::<Site>::new();
        for site in sites {
            rebuilt
                .insert(site)
                .map_err(|e| GasError::Numerical(format!("Spade rebuild: {e:?}")))?;
        }
        let edges = triangulation_edges(&incremental);
        let rebuild_edges = triangulation_edges(&rebuilt);
        let changes: Vec<_> = edges.symmetric_difference(&previous).copied().collect();
        let mismatch: Vec<_> = edges
            .symmetric_difference(&rebuild_edges)
            .copied()
            .collect();
        records.push(json!({"frame":frame,"edges":edges,"rebuild_edges":rebuild_edges,"edge_disagreements":mismatch,"adjacency_equal":edges==rebuild_edges,"removed_vertices":removals,"inserted_vertices":insertions,"slot_search_visits":search_visits,"rebuild_insertions":rebuilt.num_vertices(),"vertices":incremental.num_vertices(),"edges_count":incremental.num_undirected_edges(),"bounded_faces":incremental.num_inner_faces(),"euler_vertices_minus_edges_plus_bounded_faces":incremental.num_vertices() as i64-incremental.num_undirected_edges() as i64+incremental.num_inner_faces() as i64,"unique_changed_interfaces":changes.len(),"changed_interface_incidences":2*changes.len(),"changed_interfaces":changes}));
        previous = edges;
    }
    Ok(
        json!({"frames":records,"predicates":"Spade 2.15.1 adaptive exact orientation/in_circle on transformed f64 coordinates","degeneracy":"cocircular diagonals can differ between valid triangulations","work_units":"actual vertex removals/insertions and slot search visits","metric":metric}),
    )
}
#[derive(Clone, Copy)]
struct QueueNode {
    distance: f64,
    index: usize,
}
impl PartialEq for QueueNode {
    fn eq(&self, b: &Self) -> bool {
        self.distance == b.distance && self.index == b.index
    }
}
impl Eq for QueueNode {}
impl PartialOrd for QueueNode {
    fn partial_cmp(&self, b: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for QueueNode {
    fn cmp(&self, b: &Self) -> std::cmp::Ordering {
        b.distance
            .total_cmp(&self.distance)
            .then_with(|| b.index.cmp(&self.index))
    }
}
/// Variable-metric graph distance with radius ceil(sqrt(resolution)). Angular
/// resolution improves while the physical edge length shrinks under refinement.
/// Edge lengths integrate the bilinear metric using segment midpoint quadrature.
pub fn graph_distance(
    points: &[[f64; 2]],
    bounds: [f64; 4],
    resolution: usize,
    metric_grid: &[Mat2],
) -> Result<Value> {
    validate_domain(points, bounds, identity())?;
    require(
        (2..=128).contains(&resolution)
            && points.len() <= 64
            && metric_grid.len() == resolution * resolution,
        "graph metric grid shape/capacity",
    )?;
    for &g in metric_grid {
        spd(g)?;
    }
    require(
        points.iter().all(|p| {
            p[0] >= bounds[0] && p[0] <= bounds[1] && p[1] >= bounds[2] && p[1] <= bounds[3]
        }),
        "variable-metric sites must lie inside the metric observation domain",
    )?;
    let xy = |i: usize| {
        [
            bounds[0] + (i % resolution) as f64 * (bounds[1] - bounds[0]) / (resolution - 1) as f64,
            bounds[2] + (i / resolution) as f64 * (bounds[3] - bounds[2]) / (resolution - 1) as f64,
        ]
    };
    let count = resolution * resolution;
    let radius = (resolution as f64).sqrt().ceil() as isize;
    require(
        points.len() * count * (2 * radius + 1).pow(2) as usize * radius as usize <= 32_000_000,
        "variable metric graph work budget exceeded",
    )?;
    let metric_at = |x: f64, y: f64| -> Mat2 {
        let x = x.clamp(0., (resolution - 1) as f64);
        let y = y.clamp(0., (resolution - 1) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(resolution - 1);
        let y1 = (y0 + 1).min(resolution - 1);
        let tx = x - x0 as f64;
        let ty = y - y0 as f64;
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (1. - tx) * (1. - ty) * metric_grid[y0 * resolution + x0][i][j]
                    + tx * (1. - ty) * metric_grid[y0 * resolution + x1][i][j]
                    + (1. - tx) * ty * metric_grid[y1 * resolution + x0][i][j]
                    + tx * ty * metric_grid[y1 * resolution + x1][i][j]
            })
        })
    };
    let mut distances = Vec::new();
    let mut owners = vec![0; count];
    let mut nearest = vec![f64::INFINITY; count];
    for (slot, p) in points.iter().enumerate() {
        let mut d = vec![f64::INFINITY; count];
        let mut queue = std::collections::BinaryHeap::new();
        // Connect the actual site to the four surrounding grid vertices.
        let gx = ((p[0] - bounds[0]) / (bounds[1] - bounds[0]) * (resolution - 1) as f64)
            .clamp(0., (resolution - 1) as f64);
        let gy = ((p[1] - bounds[2]) / (bounds[3] - bounds[2]) * (resolution - 1) as f64)
            .clamp(0., (resolution - 1) as f64);
        for x in [gx.floor() as usize, gx.ceil() as usize] {
            for y in [gy.floor() as usize, gy.ceil() as usize] {
                let index = y * resolution + x;
                let q = xy(index);
                let delta = [q[0] - p[0], q[1] - p[1]];
                let distance = dot_metric(delta, delta, metric_grid[index]).sqrt();
                if distance < d[index] {
                    d[index] = distance;
                    queue.push(QueueNode { distance, index });
                }
            }
        }
        while let Some(QueueNode { distance, index }) = queue.pop() {
            if distance > d[index] {
                continue;
            }
            let x = index % resolution;
            let y = index / resolution;
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= resolution as isize || ny >= resolution as isize {
                        continue;
                    }
                    let ni = ny as usize * resolution + nx as usize;
                    let a = xy(index);
                    let b = xy(ni);
                    let delta = [b[0] - a[0], b[1] - a[1]];
                    let segments = dx.unsigned_abs().max(dy.unsigned_abs());
                    let mut edge_length = 0.;
                    for segment in 0..segments {
                        let t = (segment as f64 + 0.5) / segments as f64;
                        let g = metric_at(x as f64 + t * dx as f64, y as f64 + t * dy as f64);
                        edge_length += dot_metric(delta, delta, g).sqrt() / segments as f64;
                    }
                    let next = distance + edge_length;
                    if next < d[ni] {
                        d[ni] = next;
                        queue.push(QueueNode {
                            distance: next,
                            index: ni,
                        });
                    }
                }
            }
        }
        for i in 0..count {
            if d[i] < nearest[i] {
                nearest[i] = d[i];
                owners[i] = slot;
            }
        }
        distances.push(d);
    }
    Ok(
        json!({"owners":owners,"nearest_distance":nearest,"site_distances":distances,"grid_coordinates":(0..count).map(xy).collect::<Vec<_>>(),"resolution":resolution,"bounds":bounds,"construction":"refining_directional_metric_graph","stencil_radius":radius,"refinement_note":"radius grows as sqrt(resolution); maximum physical edge length shrinks while directions become denser; compare against exact constant-metric distances"}),
    )
}

fn boundary_surfaces(pieces: &[Value], bounds: [f64; 4], duration: f64) -> Vec<Value> {
    let scale = (bounds[1] - bounds[0])
        .max(bounds[3] - bounds[2])
        .max(duration)
        .max(1.);
    let tolerance = scale * 1e-10;
    let mut faces = std::collections::BTreeMap::<(usize, usize, Vec<[i64; 3]>), Value>::new();
    for piece in pieces {
        let slot = piece["slot"].as_u64().unwrap() as usize;
        let slab = piece["slab"].as_u64().unwrap() as usize;
        for face in piece["faces"].as_array().unwrap() {
            let mut key: Vec<[i64; 3]> = face
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    std::array::from_fn(|i| (v[i].as_f64().unwrap() / tolerance).round() as i64)
                })
                .collect();
            key.sort();
            key.dedup();
            if key.len() < 3 {
                continue;
            }
            let k = (slot, slab, key);
            if faces.remove(&k).is_none() {
                faces.insert(k, json!({"slot":slot,"slab":slab,"vertices":face}));
            }
        }
    }
    faces.into_values().collect()
}

#[cfg(test)]
mod mesh_tests {
    use super::*;
    #[test]
    fn conditional_values_match_engine_pipeline_with_distinct_channels() {
        use crate::{
            ObservationBatch, RewardBatch, TensorBatch,
            fitness::{FitnessPipeline, PositiveMap, Standardizer},
        };
        let mut input:FitnessInput=serde_json::from_value(json!({"points":[[-0.7,0.3],[0.2,0.8],[0.9,-0.4]],"companions":[1,2,0],"target":0,"query":[-0.7,0.3]})).unwrap();
        let pipeline = FitnessPipeline {
            reward_standardizer: Standardizer::Global { sigma_min: 0.17 },
            diversity_standardizer: Standardizer::Global { sigma_min: 0.31 },
            reward_map: PositiveMap::Logistic {
                amplitude: 1.3,
                floor: 0.02,
            },
            diversity_map: PositiveMap::Logistic {
                amplitude: 2.7,
                floor: 0.04,
            },
            reward_exponent: 0.7,
            diversity_exponent: 1.2,
            ..Default::default()
        };
        let points = input.points.clone();
        let obs = ObservationBatch::positions(
            TensorBatch::vectors(3, 2, points.iter().flatten().copied().collect()).unwrap(),
        );
        let rewards = RewardBatch::new(
            points.iter().map(|p| p[0] * p[0] + p[1] * p[1]).collect(),
            Default::default(),
        );
        let distances: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let q = points[input.companions[i]];
                (p[0] - q[0]).hypot(p[1] - q[1])
            })
            .collect();
        let expected = pipeline
            .evaluate(&rewards, &distances, &[true; 3], &obs, 0)
            .unwrap();
        for (i, p) in points.iter().enumerate() {
            input.target = i;
            input.query = *p;
            let actual = conditional_fitness_pipeline(&input, &pipeline).unwrap();
            assert!((actual.value - expected.fitness[i]).abs() < 1e-13);
        }
    }
    #[test]
    fn static_boundary_surface_contains_only_domain_or_cell_interfaces() {
        let points = vec![[-0.7, -0.3], [0.6, -0.4], [0.4, 0.8], [-0.4, 0.6]];
        let f = SpaceTimeFrame {
            time: 0.,
            points: points.clone(),
        };
        let mut g = f.clone();
        g.time = 1.;
        let r = spacetime(&[f, g], [-1., 1., -1., 1.], 3, identity()).unwrap();
        for face in r["boundary_faces"].as_array().unwrap() {
            let slot = face["slot"].as_u64().unwrap() as usize;
            let v: Vec<[f64; 3]> = face["vertices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| std::array::from_fn(|i| p[i].as_f64().unwrap()))
                .collect();
            let domain = (0..3).any(|axis| {
                let a = v[0][axis];
                let boundary = if axis == 2 {
                    a.abs() < 1e-9 || (a - 1.).abs() < 1e-9
                } else {
                    (a.abs() - 1.).abs() < 1e-9
                };
                boundary && v.iter().all(|p| (p[axis] - a).abs() < 1e-9)
            });
            let interface = points.iter().enumerate().any(|(j, q)| {
                j != slot
                    && v.iter().all(|p| {
                        let a = points[slot];
                        let d1 = (p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2);
                        let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                        (d1 - d2).abs() < 1e-9
                    })
            });
            assert!(
                domain || interface,
                "interior tetrahedron face remained in exported cell surface"
            );
        }
    }
    #[test]
    fn spade_incremental_matches_rebuild_and_counts_actual_edits() {
        let p = vec![
            [-0.8, -0.4],
            [0.7, -0.6],
            [0.5, 0.8],
            [-0.6, 0.5],
            [0.1, 0.2],
        ];
        let mut q = p.clone();
        q[0] = [-0.7, -0.3];
        let r = triangulation(&[p, q], [[2., 0.3], [0.3, 1.]]).unwrap();
        assert_eq!(r["frames"][1]["removed_vertices"], 1);
        assert_eq!(r["frames"][1]["inserted_vertices"], 1);
        for f in r["frames"].as_array().unwrap() {
            assert_eq!(f["adjacency_equal"], true);
            assert_eq!(f["euler_vertices_minus_edges_plus_bounded_faces"], 1);
        }
    }
    #[test]
    fn duplicate_owner_change_updates_incremental_ids() {
        let r = triangulation(
            &[
                vec![[0., 0.], [0., 0.], [1., 0.]],
                vec![[0., 1.], [0., 0.], [1., 0.]],
            ],
            identity(),
        )
        .unwrap();
        assert_eq!(r["frames"][0]["vertices"], 2);
        assert_eq!(r["frames"][1]["vertices"], 3);
        assert_eq!(r["frames"][1]["adjacency_equal"], true);
    }
    #[test]
    fn variable_metric_grid_matches_axis_distance_and_rescales() {
        let n = 9;
        let r = graph_distance(
            &[[0., 0.]],
            [-1., 1., -1., 1.],
            n,
            &vec![[[4., 0.], [0., 1.]]; n * n],
        )
        .unwrap();
        assert!((r["nearest_distance"][4 * n + 8].as_f64().unwrap() - 2.).abs() < 1e-12);
        assert!((r["nearest_distance"][8 * n + 4].as_f64().unwrap() - 1.).abs() < 1e-12);
    }
    #[test]
    fn tiny_spd_metric_is_not_lost_to_subtraction() {
        let m = spd([[1e-20, 0.], [0., 1e-20]]).unwrap();
        assert!((m.inverse[0][0] / 1e20 - 1.).abs() < 1e-14);
        assert!(spd([[1e-320, 0.], [0., 1e-320]]).is_err());
    }
    #[test]
    fn boundary_faces_remove_tetrahedron_interior() {
        let f = SpaceTimeFrame {
            time: 0.,
            points: vec![[0., 0.]],
        };
        let mut g = f.clone();
        g.time = 1.;
        let r = spacetime(&[f, g], [-1., 1., -1., 1.], 2, identity()).unwrap();
        let faces = r["boundary_faces"].as_array().unwrap();
        assert!(faces.len() < r["pieces"].as_array().unwrap().len() * 4);
        for face in faces {
            let v = face["vertices"].as_array().unwrap();
            assert!((0..3).any(|axis| {
                let a = v[0][axis].as_f64().unwrap();
                let boundary = if axis == 2 {
                    a == 0. || a == 1.
                } else {
                    a == -1. || a == 1.
                };
                boundary && v.iter().all(|p| p[axis].as_f64().unwrap() == a)
            }));
        }
    }
}
