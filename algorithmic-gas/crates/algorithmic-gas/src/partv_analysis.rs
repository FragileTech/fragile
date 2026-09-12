//! Bounded, seeded reference experiments for the 2+1 dimensional Fractal Set lectures.
//! These computations are independent of engine trajectories and identify their sampling model.
use crate::{
    GasError, Result,
    random::{RandomStream, Stream},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::f64::consts::{PI, TAU};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnalysisRequest {
    pub kind: String,
    pub seed: u64,
    pub samples: usize,
    pub replicas: usize,
    pub bandwidth: f64,
    pub curvature: f64,
    pub density_contrast: f64,
    pub phase: f64,
    pub spacetime_dimension: usize,
    pub count_model: String,
}
impl Default for AnalysisRequest {
    fn default() -> Self {
        Self {
            kind: "kernel".into(),
            seed: 7,
            samples: 256,
            replicas: 32,
            bandwidth: 0.4,
            curvature: 0.2,
            density_contrast: 0.,
            phase: 0.7,
            spacetime_dimension: 3,
            count_model: "poisson".into(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Series {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisResponse {
    pub kind: String,
    pub metrics: Vec<Metric>,
    pub series: Vec<Series>,
    pub points: Vec<Vec<f64>>,
    pub metadata: Value,
}
impl AnalysisResponse {
    fn new(r: &AnalysisRequest) -> Self {
        Self {
            kind: r.kind.clone(),
            metrics: vec![],
            series: vec![],
            points: vec![],
            metadata: json!({"model":"independent Rust reference","seed":r.seed}),
        }
    }
    fn metric(&mut self, name: &str, value: f64, reference: Option<f64>, se: Option<f64>) {
        self.metrics.push(Metric {
            name: name.into(),
            value,
            reference,
            standard_error: se,
        });
    }
    fn series(&mut self, name: &str, x: Vec<f64>, y: Vec<f64>) {
        self.series.push(Series {
            name: name.into(),
            x,
            y,
        });
    }
}
fn invalid(s: &str) -> GasError {
    GasError::Configuration(s.into())
}
fn rng(r: &AnalysisRequest, replica: usize) -> RandomStream {
    RandomStream::new(r.seed, 0, Stream::Initialize, replica as u64, 77)
}
fn stats(x: &[f64]) -> (f64, f64) {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1).max(1) as f64;
    (mean, variance)
}
/// No external state, simulation RNG, or objective budget is modified.
pub fn analyze(r: AnalysisRequest) -> Result<AnalysisResponse> {
    if !(8..=16384).contains(&r.samples)
        || !(2..=128).contains(&r.replicas)
        || r.samples
            .checked_mul(r.replicas)
            .is_none_or(|n| n > 1_048_576)
        || !r.bandwidth.is_finite()
        || !(0.01..=2.).contains(&r.bandwidth)
        || !r.curvature.is_finite()
        || r.curvature.abs() > 1.
        || !r.density_contrast.is_finite()
        || r.density_contrast.abs() > 0.95
        || !r.phase.is_finite()
        || r.phase.abs() > 100.
        || !(2..=4).contains(&r.spacetime_dimension)
        || !["poisson", "fixed"].contains(&r.count_model.as_str())
    {
        return Err(invalid(
            "analysis parameters exceed documented finite work/domain bounds",
        ));
    }
    match r.kind.as_str() {
        "spin2" => Ok(spin2(&r)),
        "transport" => Ok(transport(&r)),
        "kernel" | "manufactured" => kernel(&r),
        "integration" => Ok(integration(&r)),
        "counts" => Ok(counts(&r)),
        "dimension" => dimension(&r),
        "curvature" => curvature(&r),
        _ => Err(invalid("unknown Part V analysis kind")),
    }
}
/// Stable canonical Spin(2) section; the decoder, rather than the section, is equivariant.
pub fn spin2_encode(v: [f64; 2]) -> Result<[f64; 2]> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(invalid("nonfinite Spin(2) vector"));
    }
    let radius = v[0].hypot(v[1]);
    if !radius.is_finite() {
        return Err(invalid("Spin(2) magnitude overflow"));
    }
    if radius == 0. {
        return Ok([0., 0.]);
    }
    // Halve before adding to avoid overflow at large finite coordinates.
    let a = (radius * 0.5 + v[0].abs() * 0.5).sqrt();
    Ok(if v[0] >= 0. {
        [a, v[1] / (2. * a)]
    } else {
        [
            v[1].abs() / (2. * a),
            a.copysign(if v[1] == 0. { 1. } else { v[1] }),
        ]
    })
}
pub fn spin2_decode(z: [f64; 2]) -> [f64; 2] {
    [(z[0] - z[1]) * (z[0] + z[1]), 2. * z[0] * z[1]]
}
fn spin2(r: &AnalysisRequest) -> AnalysisResponse {
    let mut out = AnalysisResponse::new(r);
    let v = [r.phase.cos(), r.phase.sin()];
    let z = spin2_encode(v).expect("bounded trigonometric vector");
    let mut error: f64 = 0.;
    let mut angle = vec![];
    let mut re = vec![];
    let mut im = vec![];
    for k in 0..=128 {
        let a = 2. * TAU * k as f64 / 128.;
        let c = (a / 2.).cos();
        let s = (a / 2.).sin();
        let w = [c * z[0] - s * z[1], s * z[0] + c * z[1]];
        let decoded = spin2_decode(w);
        let expected = [
            a.cos() * v[0] - a.sin() * v[1],
            a.sin() * v[0] + a.cos() * v[1],
        ];
        error = error.max((decoded[0] - expected[0]).hypot(decoded[1] - expected[1]));
        angle.push(a);
        re.push(w[0]);
        im.push(w[1]);
        out.points.push(vec![a, decoded[0], decoded[1]]);
    }
    out.metric("decoder equivariance error", error, Some(0.), None);
    out.metric(
        "2pi spinor overlap",
        re[64] * z[0] + im[64] * z[1],
        Some(-1.),
        None,
    );
    out.metric(
        "4pi spinor overlap",
        re[128] * z[0] + im[128] * z[1],
        Some(1.),
        None,
    );
    out.series("spinor real", angle.clone(), re);
    out.series("spinor imaginary", angle, im);
    out.metadata["codec"] =
        json!("Spin(2), vector=(Re psi^2, Im psi^2); continuous lifted rotation");
    out
}
/// SU(2) represented as unit quaternions; fundamental normalized Wilson trace is q[0].
pub fn su2_product(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}
fn inverse(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}
fn rotation(a: f64, axis: usize) -> [f64; 4] {
    let mut q = [a.cos(), 0., 0., 0.];
    q[axis + 1] = a.sin();
    q
}
fn transport(r: &AnalysisRequest) -> AnalysisResponse {
    let mut out = AnalysisResponse::new(r);
    let phases = [0.2 * r.phase, 0.3 * r.phase, 0.4 * r.phase, 0.1 * r.phase];
    let gauge = [0.2, -0.7, 1.1, 0.4];
    let transformed: Vec<f64> = (0..4)
        .map(|i| phases[i] + gauge[i] - gauge[(i + 1) % 4])
        .collect();
    let u = phases.iter().sum::<f64>();
    let ug = transformed.iter().sum::<f64>();
    let edges = [
        rotation(r.phase, 0),
        rotation(r.phase * 0.7, 1),
        rotation(-r.phase, 0),
        rotation(-r.phase * 0.7, 1),
    ];
    let gauges = [
        rotation(0.3, 2),
        rotation(-0.8, 0),
        rotation(0.5, 1),
        rotation(0.9, 2),
    ];
    let mut w = [1., 0., 0., 0.];
    let mut wg = w;
    for i in 0..4 {
        w = su2_product(w, edges[i]);
        let e = su2_product(
            su2_product(gauges[i], edges[i]),
            inverse(gauges[(i + 1) % 4]),
        );
        wg = su2_product(wg, e);
    }
    out.metric("U1 loop real", u.cos(), Some(ug.cos()), None);
    out.metric("U1 loop imaginary", u.sin(), Some(ug.sin()), None);
    out.metric("SU2 normalized Wilson trace", w[0], Some(wg[0]), None);
    out.metric(
        "SU2 gauge invariant trace error",
        (w[0] - wg[0]).abs(),
        Some(0.),
        None,
    );
    out.metric(
        "SU2 unit norm error",
        (w.iter().map(|x| x * x).sum::<f64>() - 1.).abs(),
        Some(0.),
        None,
    );
    out.series(
        "U1 edge phase",
        (0..4).map(|i| i as f64).collect(),
        phases.to_vec(),
    );
    out.series(
        "gauge transformed phase",
        (0..4).map(|i| i as f64).collect(),
        transformed,
    );
    out.metadata["edge_convention"] = json!("U_ij -> G_i U_ij G_j^-1; ordered oriented loop");
    out
}

// Compact symmetric timelike support |t|<1, r<|t|. Calibration uses the actual bump
// and support measure, never a Euclidean radial moment substituted into a cone.
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let mut z = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..32 {
            let (mut p0, mut p1) = (1., z);
            for k in 2..=n {
                let next = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = next;
            }
            let derivative = n as f64 * (z * p1 - p0) / (z * z - 1.);
            let update = p1 / derivative;
            z -= update;
            if update.abs() < 2e-15 {
                break;
            }
        }
        let (mut p0, mut p1) = (1., z);
        for k in 2..=n {
            let next = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
            p0 = p1;
            p1 = next;
        }
        let derivative = n as f64 * (z * p1 - p0) / (z * z - 1.);
        result.push((z, 2. / ((1. - z * z) * derivative * derivative)));
    }
    result
}
fn quadrature(n: usize, mut f: impl FnMut(f64, f64, f64)) {
    let nodes = gauss_legendre(n);
    for &(t, wt) in &nodes {
        for &(v, wv) in &nodes {
            let u = (v + 1.) / 2.;
            let radius = t.abs() * u;
            let measure = wt * wv / 2. * TAU * t * t * u;
            f(t, radius, measure);
        }
    }
}
#[allow(clippy::needless_range_loop)]
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Result<[f64; 3]> {
    let norm = a.iter().flatten().fold(0_f64, |s, x| s.max(x.abs()));
    for col in 0..3 {
        let pivot = (col..3)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap();
        if a[pivot][col].abs() < norm * 1e-12 {
            return Err(invalid(
                "timelike moment system is numerically rank deficient",
            ));
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let scale = a[col][col];
        for j in col..3 {
            a[col][j] /= scale;
        }
        b[col] /= scale;
        for i in 0..3 {
            if i != col {
                let x = a[i][col];
                for j in col..3 {
                    a[i][j] -= x * a[col][j];
                }
                b[i] -= x * b[col];
            }
        }
    }
    Ok(b)
}
fn bump(t: f64, r: f64) -> f64 {
    (1. - t * t).powi(2) * (1. - r * r / (t * t)).powi(2)
}
fn calibrate(n: usize) -> Result<[f64; 3]> {
    let mut m = [[0.; 3]; 3];
    quadrature(n, |t, r, w| {
        let v = [1., t * t, r * r];
        let b = w * bump(t, r);
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += b * v[i] * v[j];
            }
        }
    });
    solve3(m, [0., -2., 4.])
}
fn kernel(r: &AnalysisRequest) -> Result<AnalysisResponse> {
    let c = calibrate(16)?;
    let mut out = AnalysisResponse::new(r);
    let mut moments = [0.; 5];
    let mut positive = 0.;
    let mut negative = 0.;
    quadrature(24, |t, r, w| {
        let k = w * bump(t, r) * (c[0] + c[1] * t * t + c[2] * r * r);
        if k >= 0. {
            positive += k;
        } else {
            negative += k;
        }
        let v = [1., t * t, r * r / 2., t.powi(4), 3. * r.powi(4) / 8.];
        for j in 0..5 {
            moments[j] += k * v[j];
        }
    });
    out.metric("zeroth moment", moments[0], Some(0.), None);
    out.metric("time second moment", moments[1], Some(-2.), None);
    out.metric("each spatial second moment", moments[2], Some(2.), None);
    out.metric("positive kernel mass", positive, None, None);
    out.metric("negative kernel mass", negative, None, None);
    // Constants are cancelled by applying K to f(hz)-f(0); display independent
    // quadrature moment residual rather than concealing it by a second calibration.
    let mut hs = vec![];
    let mut estimate = vec![];
    let mut bias = vec![];
    for i in 0..7 {
        let h = r.bandwidth / 2_f64.powf(i as f64 / 2.);
        let mut value = 0.;
        quadrature(24, |t, rad, w| {
            let k = w * bump(t, rad) * (c[0] + c[1] * t * t + c[2] * rad * rad);
            // Exact angular average of the manufactured polynomial.
            let f = h * h * (t * t + 1.5 * rad * rad)
                + h.powi(4) * (0.2 * t.powi(4) + 0.1 * 3. * rad.powi(4) / 8.);
            value += k * f / (h * h);
        });
        hs.push(h);
        estimate.push(value);
        bias.push((value - 4.).abs());
    }
    out.metric("manufactured operator", estimate[0], Some(4.), None);
    out.series("operator versus bandwidth", hs.clone(), estimate.clone());
    out.series("absolute bias", hs.clone(), bias);
    let mut ts = vec![];
    let mut ks = vec![];
    for i in 0..101 {
        let t = 0.01 + 0.98 * i as f64 / 100.;
        let rad = 0.5 * t;
        ts.push(t);
        ks.push(bump(t, rad) * (c[0] + c[1] * t * t + c[2] * rad * rad));
    }
    out.series("signed kernel at r=t/2", ts, ks);
    out.metadata = json!({"model":"independent Rust reference","seed":r.seed,"coefficients":c,"calibration_grid":16,"verification_grid":24,"rank":3,"support":"|t|<1, r<|t|; even past/future kernel", "operator":"-d_tt+d_xx+d_yy", "manufactured_field":"t^2+x^2+2y^2+0.2t^4+0.1x^4","odd_and_cross_moments":"zero by exact reflection/angular symmetry"});
    out.metadata["basis_convention"] = json!(
        "explicit laboratory-frame basis b(t,r)*(c0+c1*t^2+c2*r^2), with time cutoff |t|<1; calibrated on its actual timelike support. No Lorentz-invariance claim for the cutoff or basis."
    );
    if r.kind == "manufactured" {
        manufactured_sampling(r, c, &hs, &estimate, &mut out);
    }
    Ok(out)
}
fn manufactured_sampling(
    r: &AnalysisRequest,
    c: [f64; 3],
    hs: &[f64],
    reference: &[f64],
    out: &mut AnalysisResponse,
) {
    let half_width = r.bandwidth.max(1.);
    let volume = 8. * half_width.powi(3);
    let mut replicas = vec![vec![]; hs.len()];
    let mut support = vec![vec![]; hs.len()];
    for rep in 0..r.replicas {
        let mut random = rng(r, rep);
        let points: Vec<[f64; 3]> = (0..r.samples)
            .map(|_| std::array::from_fn(|_| half_width * (2. * random.uniform::<f64>() - 1.)))
            .collect();
        for (j, &h) in hs.iter().enumerate() {
            let mut sum = 0.;
            let mut hits = 0;
            for p in &points {
                let t = p[0] / h;
                let radius = p[1].hypot(p[2]) / h;
                if t.abs() >= 1. || radius >= t.abs() {
                    continue;
                }
                hits += 1;
                let k = bump(t, radius) * (c[0] + c[1] * t * t + c[2] * radius * radius);
                let f = p[0] * p[0]
                    + p[1] * p[1]
                    + 2. * p[2] * p[2]
                    + 0.2 * p[0].powi(4)
                    + 0.1 * p[1].powi(4);
                sum += volume * k * f / h.powi(5);
                if rep == 0 && j == 0 && out.points.len() < 512 {
                    out.points.push(p.to_vec());
                }
            }
            replicas[j].push(sum / r.samples as f64);
            support[j].push(hits as f64);
        }
    }
    let mut means = vec![];
    let mut variances = vec![];
    let mut mse = vec![];
    let mut se = vec![];
    let mut hits = vec![];
    let mut zero = vec![];
    let mut predicted_variance = vec![];
    let mut predicted_mse = vec![];
    for j in 0..hs.len() {
        let h = hs[j];
        let mut second_moment = 0.;
        // Independent deterministic integration of the single-draw squared estimator.
        // Angular trapezoidal quadrature is exact for this degree-eight polynomial.
        quadrature(24, |t, radius, w| {
            let k = bump(t, radius) * (c[0] + c[1] * t * t + c[2] * radius * radius);
            let angular_f2 = (0..16)
                .map(|a| {
                    let angle = TAU * a as f64 / 16.;
                    let x = h * radius * angle.cos();
                    let y = h * radius * angle.sin();
                    let time = h * t;
                    (time * time + x * x + 2. * y * y + 0.2 * time.powi(4) + 0.1 * x.powi(4))
                        .powi(2)
                })
                .sum::<f64>()
                / 16.;
            second_moment += volume * w * k * k * angular_f2 / h.powi(7);
        });
        let prediction = (second_moment - reference[j].powi(2)).max(0.) / r.samples as f64;
        predicted_variance.push(prediction);
        predicted_mse.push(prediction + (reference[j] - 4.).powi(2));
        let (mean, variance) = stats(&replicas[j]);
        means.push(mean);
        variances.push(variance);
        se.push((variance / r.replicas as f64).sqrt());
        mse.push(replicas[j].iter().map(|x| (x - 4.).powi(2)).sum::<f64>() / r.replicas as f64);
        hits.push(stats(&support[j]).0);
        zero.push(support[j].iter().filter(|&&x| x == 0.).count() as f64 / r.replicas as f64);
    }
    out.metric(
        "sampled manufactured operator",
        means[0],
        Some(reference[0]),
        Some(se[0]),
    );
    out.metric(
        "sampled operator variance",
        variances[0],
        Some(predicted_variance[0]),
        None,
    );
    out.metric("sampled operator MSE", mse[0], Some(predicted_mse[0]), None);
    out.metric(
        "mean support count",
        hits[0],
        Some(r.samples as f64 * 2. * PI * hs[0].powi(3) / (3. * volume)),
        None,
    );
    out.series("Monte Carlo operator mean", hs.to_vec(), means);
    out.series("Monte Carlo standard error of mean", hs.to_vec(), se);
    out.series("Monte Carlo single-run variance", hs.to_vec(), variances);
    out.series(
        "Predicted standard error of mean",
        hs.to_vec(),
        predicted_variance
            .iter()
            .map(|v| (v / r.replicas as f64).sqrt())
            .collect(),
    );
    out.series(
        "Predicted single-run variance",
        hs.to_vec(),
        predicted_variance,
    );
    out.series(
        "Predicted MSE against wave operator",
        hs.to_vec(),
        predicted_mse,
    );
    out.series("Monte Carlo MSE against wave operator", hs.to_vec(), mse);
    out.series("mean support count", hs.to_vec(), hits);
    out.series("zero-support replica fraction", hs.to_vec(), zero);
    out.series(
        "largest-bandwidth replicas",
        (0..r.replicas).map(|i| i as f64).collect(),
        replicas[0].clone(),
    );
    out.metadata["variance_prediction"] = json!({"calculation":"(volume*eps^-7*integral_J k^2*f(eps*z)^2 - L_eps^2)/N; independent 24-point cone and 16-point angular quadrature", "actual_leading_scale":"1/(N*eps^3)", "reason":"the chosen field has zero gradient at the evaluation origin; its first nonzero Taylor term is quadratic", "general_theorem":"O(1/(N*eps^5)) is a valid upper bound in D=3, but is not the sharp rate for this field", "empty_samples":"the deterministic variance stays positive even when all observed replicas miss the support"});
    out.metadata["sampling"] = json!({"domain_half_width":half_width,"density":1./volume,"samples":r.samples,"replicas":r.replicas,"normalization":"sum K(z/eps)*(f(z)-f(0))/p(z) / (N*eps^(D+2)), D=3", "replication":"independent replicas; common points across bandwidths within each replica", "zero_support":"reported explicitly; zero observed variance from empty supports does not resolve the operator", "reference":"independent quadrature using same compact support and manufactured field"});
}
fn integration(r: &AnalysisRequest) -> AnalysisResponse {
    let mut out = AnalysisResponse::new(r);
    let mut weighted = vec![];
    let mut ess = vec![];
    let mut raw = vec![];
    for rep in 0..r.replicas {
        let mut random = rng(r, rep);
        let mut wsum = 0.;
        let mut weight_sum = 0.;
        let mut weight_squared = 0.;
        let mut sum = 0.;
        for _ in 0..r.samples {
            let x = loop {
                let x = 2. * random.uniform::<f64>() - 1.;
                if random.uniform::<f64>() * (1. + r.density_contrast.abs())
                    <= 1. + r.density_contrast * x
                {
                    break x;
                }
            };
            let t = 2. * random.uniform::<f64>() - 1.;
            let y = 2. * random.uniform::<f64>() - 1.;
            let f = 1. + x * x + 2. * y * y + 0.5 * t * t + 0.3 * x;
            let weight = 8. / (1. + r.density_contrast * x);
            weight_sum += weight;
            weight_squared += weight * weight;
            wsum += weight * f;
            sum += 8. * f;
            if rep == 0 && out.points.len() < 512 {
                out.points.push(vec![t, x, y]);
            }
        }
        ess.push(weight_sum * weight_sum / weight_squared);
        weighted.push(wsum / r.samples as f64);
        raw.push(sum / r.samples as f64);
    }
    let (wm, wv) = stats(&weighted);
    out.metric("effective sample size", stats(&ess).0, None, None);
    out.metric(
        "effective sample fraction",
        stats(&ess).0 / r.samples as f64,
        None,
        None,
    );
    let (rm, rv) = stats(&raw);
    out.metric(
        "density corrected integral",
        wm,
        Some(52. / 3.),
        Some((wv / r.replicas as f64).sqrt()),
    );
    out.metric(
        "uncorrected integral",
        rm,
        Some(52. / 3. + 0.8 * r.density_contrast),
        Some((rv / r.replicas as f64).sqrt()),
    );
    out.series(
        "corrected replicas",
        (0..r.replicas).map(|i| i as f64).collect(),
        weighted,
    );
    out.series(
        "uncorrected replicas",
        (0..r.replicas).map(|i| i as f64).collect(),
        raw,
    );
    out.metadata["density"] =
        json!("p(t,x,y)=(1+c*x)/8 on [-1,1]^3; known normalized density, iid draws");
    out
}
fn poisson(random: &mut RandomStream, lambda: f64) -> usize {
    // Exponential waiting times avoid underflow in exp(-lambda).
    let mut sum = 0.;
    let mut n = 0;
    loop {
        sum -= random.uniform::<f64>().ln();
        if sum > lambda {
            return n;
        }
        n += 1;
    }
}
fn counts(r: &AnalysisRequest) -> AnalysisResponse {
    let mut out = AnalysisResponse::new(r);
    let mut totals = vec![];
    let mut region = vec![];
    for rep in 0..r.replicas {
        let mut random = rng(r, rep);
        let n = if r.count_model == "poisson" {
            poisson(&mut random, r.samples as f64)
        } else {
            r.samples
        };
        let k = (0..n).filter(|_| random.uniform::<f64>() < 0.25).count();
        totals.push(n as f64);
        region.push(k as f64);
    }
    let (mean, var) = stats(&totals);
    let (rm, rv) = stats(&region);
    let lambda = r.samples as f64;
    let total_variance = if r.count_model == "poisson" {
        lambda
    } else {
        0.
    };
    let region_variance = if r.count_model == "poisson" {
        lambda / 4.
    } else {
        3. * lambda / 16.
    };
    // Exact uncertainty of the unbiased sample variance from independent replicas.
    let variance_error = |variance: f64, fourth_cumulant: f64| {
        let replicas = r.replicas as f64;
        ((fourth_cumulant + 2. * replicas / (replicas - 1.) * variance.powi(2)) / replicas).sqrt()
    };
    out.metric(
        "total mean",
        mean,
        Some(lambda),
        Some((var / r.replicas as f64).sqrt()),
    );
    out.metric(
        "total variance",
        var,
        Some(total_variance),
        Some(variance_error(total_variance, total_variance)),
    );
    out.metric(
        "region mean",
        rm,
        Some(lambda / 4.),
        Some((rv / r.replicas as f64).sqrt()),
    );
    out.metric(
        "region variance",
        rv,
        Some(region_variance),
        Some(variance_error(
            region_variance,
            region_variance
                * if r.count_model == "poisson" {
                    1.
                } else {
                    1. - 6. * 0.25 * 0.75
                },
        )),
    );
    out.metric(
        "total Fano factor",
        var / mean.max(1.),
        Some(if r.count_model == "poisson" { 1. } else { 0. }),
        None,
    );
    out.series(
        "total counts",
        (0..r.replicas).map(|i| i as f64).collect(),
        totals,
    );
    out.series(
        "region counts",
        (0..r.replicas).map(|i| i as f64).collect(),
        region,
    );
    out.metadata["count_model"] = json!(r.count_model);
    out.metadata["region_probability"] = json!(0.25);
    out.metadata["uncertainty"] = json!(
        "means: independent-replica sample SEM; variances: exact standard deviation of unbiased sample variance from Poisson/binomial fourth central moments"
    );
    out
}
fn diamond(random: &mut RandomStream, d: usize) -> Vec<f64> {
    let radius_max = 0.5 * random.uniform::<f64>().powf(1. / d as f64);
    let t = (0.5 - radius_max)
        * if random.uniform::<f64>() < 0.5 {
            -1.
        } else {
            1.
        };
    let radius = radius_max * random.uniform::<f64>().powf(1. / (d - 1) as f64);
    if d == 2 {
        vec![
            t,
            radius
                * if random.uniform::<f64>() < 0.5 {
                    -1.
                } else {
                    1.
                },
        ]
    } else if d == 3 {
        let a = TAU * random.uniform::<f64>();
        vec![t, radius * a.cos(), radius * a.sin()]
    } else {
        let z = 2. * random.uniform::<f64>() - 1.;
        let a = TAU * random.uniform::<f64>();
        let s = (1. - z * z).sqrt();
        vec![t, radius * s * a.cos(), radius * s * a.sin(), radius * z]
    }
}
fn comparable(a: &[f64], b: &[f64]) -> bool {
    (a[0] - b[0]).powi(2)
        > a[1..]
            .iter()
            .zip(&b[1..])
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
}
// Lanczos log-gamma for positive arguments used by the continuous MM dimension inverse.
fn lgamma(z: f64) -> f64 {
    let coefficients = [
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    let z = z - 1.;
    let mut x = 0.9999999999998099;
    for (i, c) in coefficients.iter().enumerate() {
        x += c / (z + i as f64 + 1.);
    }
    let t = z + 7.5;
    0.5 * (2. * PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}
pub fn ordering_fraction(d: f64) -> f64 {
    (lgamma(d + 1.) + lgamma(d / 2.) - lgamma(1.5 * d)).exp() / 2.
}
fn infer_dimension(f: f64) -> Option<f64> {
    if !(ordering_fraction(8.)..=ordering_fraction(1.)).contains(&f) {
        return None;
    }
    let (mut lo, mut hi) = (1., 8.);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.;
        if ordering_fraction(mid) > f {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((lo + hi) / 2.)
}
fn dimension(r: &AnalysisRequest) -> Result<AnalysisResponse> {
    if r.samples > 1024 || r.samples * r.samples * r.replicas > 16_777_216 {
        return Err(invalid(
            "dimension all-pairs work capped at 16 million pair candidates and 1024 points",
        ));
    }
    let mut out = AnalysisResponse::new(r);
    let mut fractions = vec![];
    let mut raw_fractions = vec![];
    let mut dimensions = vec![];
    let mut pair_replica_indices = vec![];
    let mut inverse_replica_indices = vec![];
    let mut ess = vec![];
    let mut work = 0usize;
    for rep in 0..r.replicas {
        let mut random = rng(r, rep);
        let n = if r.count_model == "poisson" {
            poisson(&mut random, r.samples as f64)
        } else {
            r.samples
        };
        if n < 2 {
            continue;
        }
        work = work
            .checked_add(n * (n - 1) / 2)
            .ok_or_else(|| invalid("pair work overflow"))?;
        if work > 16_777_216 {
            return Err(invalid("realized Poisson pair workload exceeded cap"));
        }
        let mut points = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for _ in 0..n {
            let p = loop {
                let p = diamond(&mut random, r.spacetime_dimension);
                let ratio = 1. + 2. * r.density_contrast * p[1];
                if random.uniform::<f64>() * (1. + r.density_contrast.abs()) <= ratio {
                    break p;
                }
            };
            weights.push(1. / (1. + 2. * r.density_contrast * p[1]));
            points.push(p);
        }
        let mut count = 0usize;
        let mut weighted_count = 0.;
        let mut weighted_pairs = 0.;
        for i in 0..n {
            for j in i + 1..n {
                let weight = weights[i] * weights[j];
                weighted_pairs += weight;
                if comparable(&points[i], &points[j]) {
                    count += 1;
                    weighted_count += weight;
                }
            }
        }
        let fraction = weighted_count / weighted_pairs;
        pair_replica_indices.push(rep as f64);
        fractions.push(fraction);
        raw_fractions.push(2. * count as f64 / (n * (n - 1)) as f64);
        ess.push(weights.iter().sum::<f64>().powi(2) / weights.iter().map(|w| w * w).sum::<f64>());
        if let Some(d) = infer_dimension(fraction) {
            inverse_replica_indices.push(rep as f64);
            dimensions.push(d);
        }
        if rep == 0 {
            out.points = points.into_iter().take(512).collect();
        }
    }
    if fractions.is_empty() {
        return Err(invalid("no nonempty pair samples"));
    }
    let (m, v) = stats(&fractions);
    let (raw, rv) = stats(&raw_fractions);
    out.metric(
        "ordering fraction",
        m,
        Some(ordering_fraction(r.spacetime_dimension as f64)),
        Some((v / fractions.len() as f64).sqrt()),
    );
    out.metric(
        "unweighted ordering fraction",
        raw,
        if r.density_contrast == 0. {
            Some(ordering_fraction(r.spacetime_dimension as f64))
        } else {
            None
        },
        Some((rv / raw_fractions.len() as f64).sqrt()),
    );
    out.metric(
        "mean importance effective sample size",
        stats(&ess).0,
        None,
        None,
    );
    if let Some(d) = infer_dimension(m) {
        out.metric(
            "MM dimension from ensemble fraction",
            d,
            Some(r.spacetime_dimension as f64),
            None,
        );
    }
    out.series(
        "ordering fraction replicas",
        pair_replica_indices.clone(),
        fractions,
    );
    out.series(
        "unweighted ordering fraction replicas",
        pair_replica_indices.clone(),
        raw_fractions,
    );
    out.series(
        "MM dimensions with resolved inverse",
        inverse_replica_indices.clone(),
        dimensions,
    );
    out.metadata["normalization"] = json!(
        "sum_(i<j) w_i*w_j*1_comparable / sum_(i<j) w_i*w_j; uniform case comparable pairs/binomial(N,2)"
    );
    out.metadata["density"] = json!({"relative_to_uniform_interval":"1+2*c*x", "contrast":r.density_contrast,"weights":"1/(1+2*c*x)","finite_sample":"self-normalized pair ratio; asymptotically targets uniform ordering fraction"});
    out.metadata["r3"] = json!(8. / 35.);
    out.metadata["replica_coverage"] = json!({"requested":r.replicas,"with_pairs":pair_replica_indices.len(),"with_resolved_inverse":inverse_replica_indices.len(),"inverse_search_range":[1,8],"series_indices":"original replica indices; missing inverse values are not renumbered"});
    out.metadata["count_model"] = json!(r.count_model);
    Ok(out)
}
fn sk(k: f64, r: f64) -> f64 {
    if k.abs() < 1e-12 {
        r
    } else if k > 0. {
        (k.sqrt() * r).sin() / k.sqrt()
    } else {
        ((-k).sqrt() * r).sinh() / (-k).sqrt()
    }
}
/// Product spacetime -dt²+dr²+S_K(r)²dtheta², centered interval duration tau.
pub fn product_interval_volume(k: f64, tau: f64) -> f64 {
    let n = 2048;
    let a = tau / 2.;
    let h = a / n as f64;
    // Composite Simpson integrates the radial volume independently of sampling.
    let mut sum = 0.;
    for i in 0..=n {
        let rad = h * i as f64;
        let weight = if i == 0 || i == n {
            1.
        } else if i % 2 == 0 {
            2.
        } else {
            4.
        };
        sum += weight * (a - rad) * sk(k, rad);
    }
    4. * PI * sum * h / 3.
}
fn midpoint_fraction(k: f64, tau: f64) -> f64 {
    2. * product_interval_volume(k, tau / 2.) / product_interval_volume(k, tau)
}
fn curvature(r: &AnalysisRequest) -> Result<AnalysisResponse> {
    if r.spacetime_dimension != 3 {
        return Err(invalid(
            "product-curvature reference is restricted to 2+1 dimensions",
        ));
    }
    let tau = r.bandwidth;
    let delta = 1e-3;
    let baseline = midpoint_fraction(0., tau);
    // Calibrate derivative w.r.t scalar R=2K using separate +/- curvature quadratures.
    let calibration =
        (midpoint_fraction(delta, tau) - midpoint_fraction(-delta, tau)) / (4. * delta);
    if !calibration.is_finite() || calibration <= 1e-12 {
        return Err(invalid("unresolved midpoint curvature calibration"));
    }
    let mut out = AnalysisResponse::new(r);
    let mut fractions = vec![];
    let mut estimates = vec![];
    let envelope = if r.curvature < 0. {
        sk(r.curvature, tau / 2.) / (tau / 2.)
    } else {
        1.
    };
    for rep in 0..r.replicas {
        let mut random = rng(r, rep);
        let mut middle = 0;
        for _ in 0..r.samples {
            let p = loop {
                let mut p = diamond(&mut random, 3);
                for x in &mut p {
                    *x *= tau;
                }
                let radius = p[1].hypot(p[2]);
                let ratio = if radius == 0. {
                    1.
                } else {
                    sk(r.curvature, radius) / radius
                };
                if random.uniform::<f64>() * envelope <= ratio {
                    break p;
                }
            };
            let t = p[0].abs();
            let radius = p[1].hypot(p[2]);
            middle += usize::from(radius < t.min(tau / 2. - t));
            if rep == 0 && out.points.len() < 512 {
                out.points.push(p);
            }
        }
        let f = middle as f64 / r.samples as f64;
        fractions.push(f);
        estimates.push((f - baseline) / calibration);
    }
    let (m, v) = stats(&fractions);
    let (em, ev) = stats(&estimates);
    out.metric(
        "midpoint union fraction",
        m,
        Some(midpoint_fraction(r.curvature, tau)),
        Some((v / r.replicas as f64).sqrt()),
    );
    out.metric(
        "estimated scalar curvature",
        em,
        Some(2. * r.curvature),
        Some((ev / r.replicas as f64).sqrt()),
    );
    out.metric(
        "quadrature finite-radius curvature",
        (midpoint_fraction(r.curvature, tau) - baseline) / calibration,
        Some(2. * r.curvature),
        None,
    );
    out.metric(
        "independent linear calibration",
        calibration,
        Some(3. * tau * tau / 2560.),
        None,
    );
    out.series(
        "curvature replicas",
        (0..r.replicas).map(|i| i as f64).collect(),
        estimates,
    );
    out.series(
        "midpoint fractions",
        (0..r.replicas).map(|i| i as f64).collect(),
        fractions,
    );
    out.metadata = json!({"model":"iid volume samples of product spacetime R x constant-curvature surface", "seed":r.seed,"scalar_curvature":"R=2K", "duration":tau,"midpoint":"union of lower/upper midpoint Alexandrov intervals", "calibration":"independent symmetric K=+/-0.001 quadrature; derivative with respect to R", "flat_fraction":baseline,"resolution":"small-interval curvature signal scales as tau^2; Monte Carlo standard error displayed", "count_model":"fixed count conditional iid volume samples"});
    curvature_kernel(r, &mut out);
    Ok(out)
}

fn curvature_kernel(r: &AnalysisRequest, out: &mut AnalysisResponse) {
    let epsilon = r.bandwidth;
    let mut flat = 0.;
    let mut mr = 0.;
    // Independent flat-domain calibration. s² is timelike squared proper distance.
    quadrature(16, |t, rad, w| {
        let kernel = (1. - (t * t - rad * rad)).powi(3);
        flat += w * kernel;
        mr -= w * kernel * rad * rad / 12.;
    });
    let mut expected_kernel = 0.;
    let mut volume = 0.;
    let mut kernel_second_moment = 0.;
    quadrature(24, |t, rad, w| {
        let kernel = (1. - (t * t - rad * rad)).powi(3);
        let jacobian = sk(r.curvature, epsilon * rad) / (epsilon * rad);
        expected_kernel += w * kernel * jacobian;
        kernel_second_moment += 8. * w * (kernel * jacobian).powi(2);
        volume += epsilon.powi(3) * w * jacobian;
    });
    let mut integrals = vec![];
    let mut estimates = vec![];
    let mut supports = vec![];
    out.points.clear();
    for rep in 0..r.replicas {
        let mut random = RandomStream::new(r.seed, 0, Stream::Initialize, rep as u64, 91);
        let mut integral = 0.;
        let mut hits = 0;
        for _ in 0..r.samples {
            let p: [f64; 3] = std::array::from_fn(|_| 2. * random.uniform::<f64>() - 1.);
            let rad = p[1].hypot(p[2]);
            if rad >= p[0].abs() {
                continue;
            }
            hits += 1;
            let kernel = (1. - (p[0] * p[0] - rad * rad)).powi(3);
            let jacobian = if rad == 0. {
                1.
            } else {
                sk(r.curvature, epsilon * rad) / (epsilon * rad)
            };
            integral += 8. * kernel * jacobian;
            if rep == 0 && out.points.len() < 512 {
                out.points.push(p.map(|x| epsilon * x).to_vec());
            }
        }
        integral /= r.samples as f64;
        integrals.push(integral);
        estimates.push((integral - flat) / (epsilon * epsilon * mr));
        supports.push(hits as f64);
    }
    let (mean, var) = stats(&estimates);
    let (im, iv) = stats(&integrals);
    let predicted_variance = (kernel_second_moment - expected_kernel.powi(2)).max(0.)
        / (r.samples as f64 * epsilon.powi(4) * mr.powi(2));
    out.metric(
        "kernel single-run curvature variance",
        var,
        Some(predicted_variance),
        None,
    );
    out.metric(
        "kernel predicted standard error of mean",
        (predicted_variance / r.replicas as f64).sqrt(),
        None,
        None,
    );
    out.metric(
        "kernel estimated scalar curvature",
        mean,
        Some(2. * r.curvature),
        Some((var / r.replicas as f64).sqrt()),
    );
    out.metric(
        "kernel weighted volume",
        im,
        Some(expected_kernel),
        Some((iv / r.replicas as f64).sqrt()),
    );
    out.metric(
        "kernel flat calibration",
        flat,
        Some(187. * PI / 630.),
        None,
    );
    out.metric(
        "kernel curvature moment MR",
        mr,
        Some(-359. * PI / 41580.),
        None,
    );
    out.metric(
        "kernel quadrature finite-radius curvature",
        (expected_kernel - flat) / (epsilon * epsilon * mr),
        Some(2. * r.curvature),
        None,
    );
    out.metric(
        "kernel mean support count",
        stats(&supports).0,
        Some(r.samples as f64 * PI / 12.),
        None,
    );
    out.metric("compact geometric volume", volume, None, None);
    out.metric(
        "estimated compact curvature action",
        mean * volume,
        Some(2. * r.curvature * volume),
        Some((var / r.replicas as f64).sqrt() * volume),
    );
    out.series(
        "kernel scalar curvature replicas",
        (0..r.replicas).map(|i| i as f64).collect(),
        estimates,
    );
    out.series(
        "kernel support counts",
        (0..r.replicas).map(|i| i as f64).collect(),
        supports,
    );
    out.metadata["kernel_variance_prediction"] = json!({"calculation":"[8*integral_J (K_R*j)^2 - (integral_J K_R*j)^2]/(N*eps^4*MR^2)", "actual_leading_scale":"1/(N*eps^4)", "sampling_domain":"the physical sampling cube shrinks with eps, keeping its support-hit probability pi/12 constant", "general_theorem":"O(1/(N*eps^7)) in D=3 describes a fixed sampling density; that is a different sampling protocol", "uniform_normalized_cube_density":0.125});
    out.metadata["kernel_curvature"] = json!({"kernel":"K_R=(1-s²)^3, s²=u²-r²", "support":"|u|<1, r<|u|", "volume_jacobian":"S_K(eps*r)/(eps*r)","calibration":"MR=-(1/12) integral_J K_R*r² dζ from independent flat16-point quadrature", "verification":"24-point curved quadrature; distinct from Monte Carlo draws", "estimator":"(mean 8*1_J*K_R*j - flat_integral)/(eps²*MR)", "samples":"uniform normalized cube [-1,1]^3 with known density 1/8; geometry enters as importance weight j", "action":"scalar curvature estimate times independently quadrature-computed physical compact volume", "comparison":"midpoint and kernel use separate addressed random streams"});
}
