//! Compiled calculations for lattice QFT, direct fields, Yang–Mills and twistors.
//!
//! Lecture measurements consume actual recorded stages. Analytic model oracles
//! are exercised by unit tests independently of the experiment interface.
mod algorithm_channels;
pub mod channel;
pub mod math;
mod native;
mod run_observables;
use super::partvi::{ExperimentRequest, ExperimentResult, Series};
use crate::{GasError, Result, RunArchive};
use math::*;
use serde_json::json;
use std::f64::consts::{PI, TAU};
fn param(r: &ExperimentRequest, key: &str, d: f64, lo: f64, hi: f64) -> f64 {
    r.number(key, d).clamp(lo, hi)
}
fn amp(r: &ExperimentRequest) -> f64 {
    param(r, "amplitude", 0.5, 0., 2.)
}
fn angle(r: &ExperimentRequest) -> f64 {
    param(r, "angle", 0.7, -PI, PI)
}
fn rate(r: &ExperimentRequest) -> f64 {
    param(r, "rate", 0.4, 0.01, 2.)
}
fn count(r: &ExperimentRequest) -> usize {
    r.usize("n", 512).clamp(32, 4096)
}
fn steps(r: &ExperimentRequest) -> usize {
    r.usize("steps", 32).clamp(2, 128)
}
fn rng(r: &ExperimentRequest) -> Rng {
    Rng::new(r.usize("seed", 7) as u64)
}
fn pair(
    out: &mut ExperimentResult,
    title: &str,
    x: &str,
    y: &str,
    a: Vec<[f64; 2]>,
    b: Vec<[f64; 2]>,
) {
    out.plot(
        title,
        x,
        y,
        vec![Series::line("Calculated", a), Series::line("Prediction", b)],
    );
}
fn complex_json(z: C) -> serde_json::Value {
    json!({"real":z.re,"imaginary":z.im})
}
fn maxabs(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0., f64::max)
}
fn color(f: [f64; 3], v: [f64; 3], k: f64, delta: f64) -> ([C; 3], bool) {
    let norm = f.iter().map(|x| x * x).sum::<f64>().sqrt();
    let raw = std::array::from_fn(|a| C::phase(k * v[a]) * (f[a] / norm.max(delta)));
    (raw, norm > delta)
}
fn color_fixture(k: f64) -> Vec<[C; 3]> {
    (0..7)
        .map(|i| {
            let t = i as f64 * 0.8;
            color(
                [1. + t.sin(), 0.4 + t.cos(), 0.2 + t.sin() * t.cos()],
                [t, 0.3 * t, -0.4 * t],
                k,
                1e-12,
            )
            .0
        })
        .collect()
}
/// Exact spinor choice in the effective-edge definition. Ties select column zero.
pub fn edge_spinor(
    dx: [f64; 3],
    dv: [f64; 3],
    dt: f64,
    alpha: f64,
) -> Option<([C; 2], [C; 2], usize)> {
    if !dt.is_finite() || !alpha.is_finite() || dx.iter().chain(&dv).any(|x| !x.is_finite()) {
        return None;
    }
    let x = [
        C::from(dt + dx[2]),
        C::new(dx[0], -dx[1]),
        C::new(dx[0], dx[1]),
        C::from(dt - dx[2]),
    ];
    let b = [
        x[0] + C::new(0., alpha * dv[2]),
        x[1] + C::new(alpha * dv[1], alpha * dv[0]),
        x[2] + C::new(-alpha * dv[1], alpha * dv[0]),
        x[3] + C::new(0., -alpha * dv[2]),
    ];
    let n0 = b[0].abs2() + b[2].abs2();
    let n1 = b[1].abs2() + b[3].abs2();
    let j = usize::from(n1 > n0);
    let n = if j == 0 { n0 } else { n1 }.sqrt();
    if n <= 1e-12 {
        return None;
    }
    let l = [b[j] / n, b[2 + j] / n];
    let mu = [x[0] * l[0] + x[1] * l[1], x[2] * l[0] + x[3] * l[1]];
    Some((l, mu, j))
}
fn epsilon(a: &[C; 2], b: &[C; 2]) -> C {
    a[0] * b[1] - a[1] * b[0]
}
pub fn analyze(
    r: &ExperimentRequest,
    archive: Option<&RunArchive<f64>>,
) -> Result<ExperimentResult> {
    if archive.is_none() {
        return Err(GasError::Capability(
            "QFT experiments require an executed gas archive or complete-checkpoint protocol"
                .into(),
        ));
    }
    analyze_dispatch(r, archive)
}
#[cfg(test)]
mod reference_tests;
fn analyze_dispatch(
    r: &ExperimentRequest,
    archive: Option<&RunArchive<f64>>,
) -> Result<ExperimentResult> {
    if let Some(a) = archive
        && matches!(
            r.experiment,
            1 | 7 | 11 | 15 | 20 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 33
        )
    {
        return run_observables::analyze(r, a);
    }
    if let Some(a) = archive
        && matches!(r.experiment, 3 | 4 | 5 | 6 | 12 | 13 | 32 | 36)
    {
        return native::analyze(r, a);
    }
    if let Some(a) = archive
        && matches!(r.experiment, 2 | 9 | 10 | 14)
    {
        return algorithm_channels::analyze(r, a);
    }
    let out = match r.experiment {
        1 => oriented(r),
        2 => cloning(r, archive),
        3 => exterior(r),
        4 => channel::analyze(r),
        5 => locality(r),
        6 => replicas(r),
        7 => continuum(r),
        8 => colors(r, archive)?,
        9 => invariants(r),
        10 => doublets(r),
        11 => holonomy(r),
        12 => memory(r),
        13 => prediction(r),
        14 => writes(r),
        15 => cp(r),
        16 => generating(r),
        17 => path_action(r),
        18 => kinetic(r, archive),
        19 => response(r),
        20 => information(r),
        21 => disintegration(r),
        22 => noether(r),
        23 => ward(r),
        24 => yang_mills(r),
        25 => gaps(r),
        26 => collision(r),
        27 => qsd(r),
        28 => reflection(r),
        29 => faces(r),
        30 => translation(r),
        31 => interventions(r),
        32 => synthesis(r),
        33 => spinors(r),
        34 => twistors(r, archive)?,
        35 => lag_tracking(r, archive)?,
        36 => spectra(r),
        _ => {
            return Err(GasError::Configuration(
                "QFT experiment id must be 1..36".into(),
            ));
        }
    };
    Ok(out)
}
fn oriented(r: &ExperimentRequest) -> ExperimentResult {
    let t = angle(r);
    let links = [
        su2(t, [1., 0., 0.]),
        su2(0.6 * t, [0., 1., 0.]),
        su2(-0.3 * t, [0., 0., 1.]),
    ];
    let product = mul(&mul(&links[0], &links[1], 2), &links[2], 2);
    let g = su2(amp(r), [0., 0., 1.]);
    let transformed = mul(&mul(&g, &product, 2), &adjoint(&g, 2), 2);
    let reverse = mul(
        &mul(&adjoint(&links[2], 2), &adjoint(&links[1], 2), 2),
        &adjoint(&links[0], 2),
        2,
    );
    let mut indexed = vec![C::ZERO; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    indexed[i * 2 + j] = indexed[i * 2 + j]
                        + links[0][i * 2 + k] * links[1][k * 2 + l] * links[2][l * 2 + j];
                }
            }
        }
    }
    let mut o = ExperimentResult::new(
        1,
        "Oriented diagrams and transport",
        "Supplied noncommuting SU(2) links on a three-edge directed loop",
    );
    o.metric(
        "Indexed contraction residual",
        difference(&product, &indexed),
        "",
    )
    .metric(
        "Reversal adjoint residual",
        difference(&reverse, &adjoint(&product, 2)),
        "",
    )
    .metric(
        "Rebasing trace residual",
        (trace(&product, 2) - trace(&transformed, 2)).abs(),
        "",
    );
    o.details = json!({"links":links.iter().map(|a|a.iter().copied().map(complex_json).collect::<Vec<_>>()).collect::<Vec<_>>(),"wilson_trace":complex_json(trace(&product,2)/2.)});
    let mut a = vec![];
    let mut b = vec![];
    for k in 0..65 {
        let x = -PI + TAU * k as f64 / 64.;
        let p = mul(
            &mul(&su2(x, [1., 0., 0.]), &su2(0.6 * x, [0., 1., 0.]), 2),
            &su2(-0.3 * x, [0., 0., 1.]),
            2,
        );
        a.push([x, trace(&p, 2).re / 2.]);
        b.push([x, trace(&adjoint(&p, 2), 2).re / 2.]);
    }
    pair(&mut o, "Loop reversal", "angle", "normalized trace", a, b);
    o.note("Links are explicit unitary transport inputs. A rank-one color projector does not supply an SU(3) connection.");
    o
}
fn cloning(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> ExperimentResult {
    let vi = 0.2 + amp(r);
    let eps = 0.01;
    let mut a = vec![];
    let mut b = vec![];
    let mut residual: f64 = 0.;
    let mut overlap: f64 = 0.;
    for k in 0..65 {
        let vj = 0.01 + 2. * k as f64 / 64.;
        let s = (vj - vi) / (vi + eps);
        let t = (vi - vj) / (vj + eps);
        a.push([vj, s + t]);
        b.push([vj, (vj - vi).powi(2) / ((vi + eps) * (vj + eps))]);
        residual = residual.max(((vi + eps) * s + (vj + eps) * t).abs());
        overlap = overlap.max(s.clamp(0., 1.) * t.clamp(0., 1.));
    }
    let mut o = ExperimentResult::new(
        2,
        "Cloning antisymmetry and exclusion",
        "Exact positive-fitness directed-score sweep",
    );
    o.metric("Weighted antisymmetry residual", residual, "")
        .metric("Opposing acceptance product", overlap, "");
    pair(&mut o, "Raw score sum", "donor fitness", "score", a, b);
    if let Some(step) = archive.and_then(|a| a.steps.last()) {
        o.details = json!({"recorded_step":step.report.step,"recorded_donor_fitness":step.donor_fitness,"recorded_plan":step.report.clone_plan,"companion_pool_indices":step.report.cloning_companions});
        o.note("Recorded donor fitness is pool aligned and includes historical rescoring. The plotted sweep varies an explicit pair independently of the stored decisions.");
    }
    o
}
fn exterior(r: &ExperimentRequest) -> ExperimentResult {
    let m = r.usize("modes", 3).clamp(1, 2);
    let n = 1 << m;
    let mut residual: f64 = 0.;
    let id = identity(n);
    let mut bars = vec![];
    for i in 0..m {
        let c = creator(m, i);
        let a = adjoint(&c, n);
        for j in 0..m {
            let d = creator(m, j);
            let ac = mul(&a, &d, n);
            let ca = mul(&d, &a, n);
            let e: Vec<C> = ac
                .iter()
                .zip(ca)
                .enumerate()
                .map(|(k, (&x, y))| x + y - if i == j { id[k] } else { C::ZERO })
                .collect();
            residual = residual.max(e.iter().map(|x| x.abs()).fold(0., f64::max));
        }
        let cc = mul(&c, &c, n);
        bars.push([i as f64, cc.iter().map(|x| x.abs2()).sum::<f64>().sqrt()]);
    }
    let t = amp(r);
    let readouts = [
        vec![1., -1., 1., -1.],
        vec![1., 1., -1., -1.],
        vec![t, -t, t, -t],
    ];
    let gram: Vec<f64> = (0..9)
        .map(|k| {
            readouts[k / 3]
                .iter()
                .zip(&readouts[k % 3])
                .map(|(x, y)| x * y)
                .sum::<f64>()
                / 4.
        })
        .collect();
    let (e, _) = symmetric_eigen(&gram, 3);
    let rank = e.iter().filter(|&&x| x > 1e-10).count();
    let mut o = ExperimentResult::new(
        3,
        "Exterior algebra from recorded observables",
        "Centered four-state reference law and exact occupation-bit Fock algebra",
    );
    o.metric("CAR residual", residual, "")
        .metric("Gram rank", rank as f64, "modes")
        .metric("Fock dimension", n as f64, "states");
    o.plot(
        "Repeated creation vanishes",
        "mode",
        "norm",
        vec![Series::line("Creation squared", bars)],
    );
    o.details = json!({"gram":gram,"gram_eigenvalues":e,"null_quotient_dimension":rank,"occupation_basis":(0..n).map(|x|format!("{x:0m$b}")).collect::<Vec<_>>()});
    o
}
fn locality(r: &ExperimentRequest) -> ExperimentResult {
    let rho = (amp(r) / 2.).min(0.99);
    let m = 2;
    let n = 4;
    let c0 = creator(m, 0);
    let c1 = creator(m, 1);
    let g: Vec<C> = c0
        .iter()
        .zip(&c1)
        .map(|(&a, &b)| a * rho + b * (1. - rho * rho).sqrt())
        .collect();
    let f = adjoint(&c0, n);
    let x = mul(&f, &g, n);
    let y = mul(&g, &f, n);
    let anti: Vec<C> = x.iter().zip(y).map(|(&a, b)| a + b).collect();
    let target: Vec<C> = identity(n).into_iter().map(|z| z * rho).collect();
    let mut o = ExperimentResult::new(
        5,
        "Regional locality and covariance",
        "Two normalized regions sharing one centered latent mode",
    );
    o.metric("Cross-covariance singular value", rho, "").metric(
        "CAR locality residual",
        difference(&anti, &target),
        "",
    );
    let a = (0..65)
        .map(|i| {
            let d = i as f64 / 8.;
            [d, (-rate(r) * d).exp()]
        })
        .collect();
    o.plot(
        "Locality under a declared covariance kernel",
        "region separation",
        "anticommutator norm",
        vec![Series::line("exp(-rate × separation)", a)],
    );
    o.details =
        json!({"cross_covariance":rho,"spatial_separation_alone_implies_orthogonality":false});
    o.note("The plotted spatial decay is the selected covariance model. Global whitening changes the observables and can mix the two regions.");
    o
}
fn replicas(r: &ExperimentRequest) -> ExperimentResult {
    let t = amp(r);
    let f = [-1., 1.];
    let g = [-t, t];
    let same = (f[0] * g[0] + f[1] * g[1]) / 2.;
    let mut wedge = 0.;
    for i in 0..2 {
        for j in 0..2 {
            wedge += (f[i] * g[j] - f[j] * g[i]).powi(2) / 8.;
        }
    }
    let mut o = ExperimentResult::new(
        6,
        "Same-swarm products and replica wedges",
        "Exact enumeration of independent replicas of a symmetric two-state law",
    );
    o.metric("Same-state product", same, "")
        .metric("Replica wedge norm squared", wedge, "")
        .metric("Gram determinant", 1. * t * t - same * same, "");
    o.plot(
        "Different products of dependent readouts",
        "scale",
        "expectation",
        vec![
            Series::line(
                "Same-state f g",
                (0..65)
                    .map(|i| {
                        let x = 2. * i as f64 / 64.;
                        [x, x]
                    })
                    .collect(),
            ),
            Series::line(
                "Antisymmetric replica wedge",
                (0..65).map(|i| [2. * i as f64 / 64., 0.]).collect(),
            ),
        ],
    );
    o.note("The zero wedge follows from exact linear dependence; it does not remove the nonzero ordinary product.");
    o
}
fn continuum(r: &ExperimentRequest) -> ExperimentResult {
    let a = amp(r) * 0.4;
    let x = 0.35;
    let rho = |x: f64| 1. + a * x.cos();
    let f = |x: f64| x.sin();
    let target = (-rho(x) * x.sin() - 2. * a * x.sin() * x.cos()) / 2.;
    let rowtarget = -x.sin() / 2. - a * x.sin() * x.cos() / rho(x);
    let population = count(r);
    let mut random = rng(r);
    let mut points = Vec::with_capacity(population);
    while points.len() < population {
        let y = TAU * random.uniform() - PI;
        if random.uniform() * (1. + a) < rho(y) {
            points.push(y);
        }
    }
    let mut empirical = vec![];
    let mut upper = vec![];
    let mut lower = vec![];
    let mut errors = vec![];
    let mut samples = vec![];
    let mut pred = vec![];
    let mut corrected = vec![];
    let mut row = vec![];
    for k in 0..25 {
        let e = 0.03 + 0.77 * k as f64 / 24.;
        let integral = integrate(
            |z| gaussian(z) * (f(x + e * z) - f(x)) * rho(x + e * z),
            -8.,
            8.,
            800,
        ) / (e * e);
        let mass = integrate(|z| gaussian(z) * rho(x + e * z), -8., 8., 800);
        let corr = integrate(|z| gaussian(z) * (f(x + e * z) - f(x)), -8., 8., 800) / (e * e);
        let summands: Vec<f64> = points
            .iter()
            .map(|&y| {
                let kernel = (-2..=2)
                    .map(|image| gaussian((y - x + TAU * image as f64) / e))
                    .sum::<f64>();
                TAU * kernel * (f(y) - f(x)) / e.powi(3)
            })
            .collect();
        let estimate = summands.iter().sum::<f64>() / population as f64;
        let variance =
            summands.iter().map(|z| (z - estimate).powi(2)).sum::<f64>() / (population - 1) as f64;
        let se = (variance / population as f64).sqrt();
        empirical.push([e, estimate]);
        upper.push([e, estimate + 2. * se]);
        lower.push([e, estimate - 2. * se]);
        errors.push(json!({"bandwidth":e,"estimate":estimate,"standard_error":se,"quadrature":integral,"samples":population}));
        samples.push([e, integral]);
        pred.push([e, target]);
        row.push([e, integral / mass]);
        corrected.push([e, corr]);
    }
    let mut o = ExperimentResult::new(
        7,
        "Density-aware continuum operators",
        "Independent periodic-density point samples and Gaussian quadrature on the same scalar field",
    );
    o.metric("Unnormalized continuum target", target, "")
        .metric("Row-normalized target", rowtarget, "")
        .metric("Density-corrected target", -x.sin() / 2., "");
    o.plot(
        "Three kernel conventions",
        "bandwidth",
        "generator",
        vec![
            Series::line("Sampled point-cloud generator", empirical),
            Series::line("Sample estimate + 2 SE", upper),
            Series::line("Sample estimate - 2 SE", lower),
            Series::line("Unnormalized quadrature", samples),
            Series::line("Unnormalized continuum", pred),
            Series::line("Row normalized", row),
            Series::line("Inverse-density corrected", corrected),
        ],
    );
    o.details = json!({"density":"1 + 0.4 amplitude cos(x)","field":"sin(x)","position":x,"second_kernel_moment":1.,"normalization":"Gaussian probability kernel divided by bandwidth squared","point_sampling":"independent rejection draws from (1+a cos y)/(2pi)","sample_size":population,"sampling_errors":errors});
    o
}
fn colors(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    let k = angle(r);
    let delta = param(r, "threshold", 1e-12, 1e-15, 1.);
    let mut o = ExperimentResult::new(
        8,
        "Recorded color states",
        "Three-dimensional viscous-force and velocity reference fixture",
    );
    let (mut forces, mut velocities) = (vec![], vec![]);
    let mut source_slots = vec![];
    let mut source_coverage = vec![];
    let mut stage = "reference".to_string();
    if let Some(a) = archive {
        let s = a.steps.last().ok_or_else(|| {
            GasError::Capability("color requires at least one recorded step".into())
        })?;
        let f = s
            .field_evaluations
            .iter()
            .find(|x| x.stage == "B1" && x.field == "viscous_force")
            .ok_or_else(|| GasError::Capability("archive has no B1 viscous-force record".into()))?;
        let v = s
            .field_evaluations
            .iter()
            .find(|x| x.stage == "B1" && x.field == "force_input_velocity")
            .ok_or_else(|| {
                GasError::Capability("archive has no velocity aligned to B1 force".into())
            })?;
        if f.item_shape != [3]
            || v.item_shape != [3]
            || f.values.len() != v.values.len()
            || f.version != v.version
            || !s
                .stages
                .iter()
                .any(|stage| stage.stage == "B1_input" && stage.version == f.version)
        {
            return Err(GasError::Shape(
                "color requires aligned three-dimensional force and velocity".into(),
            ));
        }
        for (i, (x, y)) in f
            .values
            .chunks_exact(3)
            .zip(v.values.chunks_exact(3))
            .enumerate()
        {
            let available =
                f.available[i] && v.available[i] && x.iter().chain(y).all(|x| x.is_finite());
            source_coverage.push(available);
            if available {
                source_slots.push(i);
                forces.push([x[0], x[1], x[2]]);
                velocities.push([y[0], y[1], y[2]]);
            }
        }
        stage = format!("B1 input, step {}, version {}", s.report.step, f.version);
        o.model = "Executed archive: B1 viscous force with its recorded input velocity".into();
    } else {
        for i in 0..32 {
            let t = i as f64 / 8.;
            forces.push([amp(r) * t.sin(), 0.3 * t.cos(), 0.2 * (2. * t).sin()]);
            velocities.push([t, 0.4 * t, -t]);
        }
        forces.push([delta / 2., 0., 0.]);
        velocities.push([0.; 3]);
        source_slots = (0..forces.len()).collect();
        source_coverage = vec![true; forces.len()];
    }
    let mut raw = vec![];
    let mut masks = vec![];
    let mut norms = vec![];
    let mut thresholded = vec![];
    for (i, (&f, &v)) in forces.iter().zip(&velocities).enumerate() {
        let (c, valid) = color(f, v, k, delta);
        let n = c.iter().map(|z| z.abs2()).sum::<f64>();
        raw.push(c.iter().copied().map(complex_json).collect::<Vec<_>>());
        masks.push(valid);
        norms.push([source_slots[i] as f64, n]);
        thresholded.push([source_slots[i] as f64, if valid { n } else { 0. }]);
    }
    o.metric(
        "Valid colors",
        masks.iter().filter(|&&x| x).count() as f64,
        "walkers",
    )
    .metric("Threshold", delta, "force");
    pair(
        &mut o,
        "Raw normalization and zero extension",
        "walker",
        "squared norm",
        norms,
        thresholded,
    );
    o.details = json!({"stage":stage,"kappa":k,"source_slots":source_slots,"source_coverage":source_coverage,"raw_color":raw,"valid_mask":masks,"force":forces,"velocity":velocities,"alignment":"matched B1 force input; explicit stage convention"});
    Ok(o)
}
fn invariants(r: &ExperimentRequest) -> ExperimentResult {
    let colors = color_fixture(angle(r));
    let g = [
        C::phase(amp(r)),
        C::phase(-0.7 * amp(r)),
        C::phase(-0.3 * amp(r)),
    ];
    let changed: Vec<[C; 3]> = colors
        .iter()
        .map(|c| std::array::from_fn(|i| g[i] * c[i]))
        .collect();
    let mut residual: f64 = 0.;
    let mut points = vec![];
    for i in 0..colors.len() {
        for j in 0..colors.len() {
            let z = dot(&colors[i], &colors[j]);
            residual = residual.max((z - dot(&changed[i], &changed[j])).abs());
            points.push([i as f64 * colors.len() as f64 + j as f64, z.abs2()]);
        }
    }
    let a: Vec<C> = (0..9).map(|k| colors[k % 3][k / 3]).collect();
    let b: Vec<C> = (0..9).map(|k| changed[k % 3][k / 3]).collect();
    let gram: Vec<C> = (0..9)
        .map(|k| dot(&colors[k / 3], &colors[k % 3]))
        .collect();
    let det = determinant(&a, 3);
    let tri =
        dot(&colors[0], &colors[1]) * dot(&colors[1], &colors[2]) * dot(&colors[2], &colors[0]);
    let project = |c: &[C; 3]| {
        (0..9)
            .map(|k| c[k / 3] * c[k % 3].conj())
            .collect::<Vec<_>>()
    };
    let trace_tri = trace(
        &mul(
            &mul(&project(&colors[0]), &project(&colors[1]), 3),
            &project(&colors[2]),
            3,
        ),
        3,
    );
    let mut o = ExperimentResult::new(
        9,
        "SU(3) invariant coordinates",
        "Normalized three-color reference field with determinant-one diagonal frame transformation",
    );
    o.metric("Gram invariance residual", residual, "")
        .metric(
            "Baryon invariance residual",
            (det - determinant(&b, 3)).abs(),
            "",
        )
        .metric(
            "Gram determinant residual",
            (determinant(&gram, 3) - C::from(det.abs2())).abs(),
            "",
        )
        .metric("Triangle trace residual", (tri - trace_tri).abs(), "");
    o.plot(
        "Meson magnitudes",
        "ordered pair",
        "squared magnitude",
        vec![Series::line("Color inner products", points)],
    );
    o.details = json!({"baryon":complex_json(det),"triangle":complex_json(tri),"global_U3_phase_changes_baryon_by":"exp(3 i phase)","rank":hermitian_eigenvalues(&gram,3).iter().filter(|&&v|v>1e-10).count()});
    o
}
fn doublets(r: &ExperimentRequest) -> ExperimentResult {
    let fit = [0.4, 0.8, 1.3];
    let p = 0.1 + 0.4 * amp(r);
    let ell = 0.3 + rate(r);
    let mut expectation = 0.;
    let mut enumerate = vec![];
    let mut total = 0.;
    let mut norm_error: f64 = 0.;
    for bits in 0usize..8 {
        let mut map = [0; 3];
        let mut law = 1.;
        for (i, target) in map.iter_mut().enumerate() {
            let one = (bits >> i) & 1 == 1;
            *target = (i + if one { 2 } else { 1 }) % 3;
            law *= if one { p } else { 1. - p };
        }
        let mut amplitudes = [C::ZERO; 3];
        for i in 0..3 {
            let j = map[i];
            let d = (j as f64 - i as f64).abs();
            amplitudes[i] =
                C::phase((fit[j] - fit[i]) / (fit[i] + 0.01)) * (-d * d / (4. * ell * ell)).exp();
        }
        let d = [amplitudes[0], amplitudes[map[0]]];
        let norm = (d[0].abs2() + d[1].abs2()).sqrt();
        let z = [d[0] / norm, d[1] / norm];
        norm_error = norm_error.max((dot(&z, &z).re - 1.).abs());
        let overlap = z[0].conj() * z[1];
        expectation += law * overlap.re;
        total += law;
        enumerate.push(json!({"companion_map":map,"joint_probability":law,"doublet":z.map(complex_json),"real_overlap":overlap.re}));
    }
    let mut random = rng(r);
    let mut values = vec![];
    for _ in 0..count(r) {
        let u = random.uniform();
        let mut cum = 0.;
        for item in &enumerate {
            cum += item["joint_probability"].as_f64().unwrap();
            if u <= cum {
                values.push(item["real_overlap"].as_f64().unwrap());
                break;
            }
        }
    }
    let mut o = ExperimentResult::new(
        10,
        "Two-hop companion doublets",
        "Exact joint enumeration of independent directed choices on three walkers",
    );
    o.metric("Joint probability mass", total, "")
        .metric("Doublet norm residual", norm_error, "")
        .metric("Exact mean overlap", expectation, "")
        .metric("Sample mean overlap", mean(&values), "")
        .metric("Independent sample SEM", sem(&values), "");
    o.plot(
        "Joint companion assignments",
        "assignment",
        "probability",
        vec![Series::line(
            "Executed-law probabilities",
            enumerate
                .iter()
                .enumerate()
                .map(|(i, v)| [i as f64, v["joint_probability"].as_f64().unwrap()])
                .collect(),
        )],
    );
    o.details = json!({"joint_assignments":enumerate});
    o
}
fn holonomy(r: &ExperimentRequest) -> ExperimentResult {
    let g = [
        identity(2),
        su2(angle(r), [1., 0., 0.]),
        su2(amp(r), [0., 1., 0.]),
    ];
    let mut coherent = identity(2);
    let mut attributed = identity(2);
    let mut a = vec![];
    let mut b = vec![];
    for i in 0..3 {
        let j = (i + 1) % 3;
        let u = mul(&adjoint(&g[i], 2), &g[j], 2);
        coherent = mul(&coherent, &u, 2);
        let defect = su2(if i == 1 { rate(r) } else { 0. }, [0., 0., 1.]);
        attributed = mul(&attributed, &mul(&u, &defect, 2), 2);
        a.push([i as f64, difference(&coherent, &identity(2))]);
        b.push([i as f64, difference(&attributed, &identity(2))]);
    }
    let mut o = ExperimentResult::new(
        11,
        "Attribution holonomy",
        "Three supplied SU(2) frames and one explicit attribution mismatch",
    );
    o.metric(
        "Coherent closed-loop defect",
        difference(&coherent, &identity(2)),
        "",
    )
    .metric(
        "Attribution closed-loop defect",
        difference(&attributed, &identity(2)),
        "",
    )
    .metric(
        "Angular loop energy",
        1. - trace(&attributed, 2).re / 2.,
        "",
    );
    o.plot(
        "Partial transport around the loop",
        "edge",
        "distance from identity",
        vec![
            Series::line("Geometric frame transport", a),
            Series::line("Attribution transport", b),
        ],
    );
    o.note("The mismatch factor is declared input; it models a failed attribution-coherence condition.");
    o
}
fn memory_model(a: f64) -> (Vec<f64>, f64, f64) {
    let f = [1., 1., -1., -1.];
    let g = [1., -1., 1., -1.];
    let aa = 0.35;
    let b = 0.1 * a;
    let d = 0.15;
    let p = (0..16)
        .map(|k| {
            let (i, j) = (k / 4, k % 4);
            (1. + aa * f[i] * f[j] + d * g[i] * g[j] + b * (f[i] * g[j] + g[i] * f[j])) / 4.
        })
        .collect();
    (p, aa, b)
}
fn memory(r: &ExperimentRequest) -> ExperimentResult {
    let (p, a, b) = memory_model(amp(r));
    let f = [1., 1., -1., -1.];
    let mut actual = vec![];
    let mut naive = vec![];
    let mut recurrence = vec![1., a];
    for n in 2..=steps(r) {
        let mem = (0..n - 1)
            .map(|j| b * b * 0.15f64.powi((n - 2 - j) as i32) * recurrence[j])
            .sum::<f64>();
        recurrence.push(a * recurrence[n - 1] + mem);
    }
    let mut residual: f64 = 0.;
    for (n, expected) in recurrence.iter().enumerate() {
        let q = rpower(&p, 4, n);
        let v = (0..16)
            .map(|k| f[k / 4] * q[k] * f[k % 4] / 4.)
            .sum::<f64>();
        actual.push([n as f64, v]);
        naive.push([n as f64, a.powi(n as i32)]);
        residual = residual.max((v - expected).abs());
    }
    let mut o = ExperimentResult::new(
        12,
        "Projected dynamics and memory",
        "Exact four-state conservative reversible Markov chain with a hidden mode",
    );
    o.metric("Two-step memory defect", actual[2][1] - a * a, "")
        .metric("Predicted B C", b * b, "")
        .metric("Memory recurrence residual", residual, "");
    pair(
        &mut o,
        "Compressed multistep evolution",
        "lag",
        "correlation",
        actual,
        naive,
    );
    o.details = json!({"transition":p,"A":a,"B":b,"C":b,"D":0.15,"stationary_distribution":[0.25,0.25,0.25,0.25]});
    o
}
fn prediction(r: &ExperimentRequest) -> ExperimentResult {
    let (p, _, _) = memory_model(amp(r));
    let readout = [1., -1., 1., -1.];
    let target: Vec<f64> = (0..4)
        .map(|i| (0..4).map(|j| p[i * 4 + j] * readout[j]).sum())
        .collect();
    let mut random = rng(r);
    let mut calc = vec![];
    let mut pred = vec![];
    let mut errors = vec![];
    for groups in [1, 2, 4] {
        let mut mu = vec![0.; groups];
        for i in 0..4 {
            mu[i * groups / 4] += target[i] * groups as f64 / 4.;
        }
        let exact = (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| p[i * 4 + j] * (readout[j] - mu[i * groups / 4]).powi(2) / 4.)
                    .sum::<f64>()
            })
            .sum::<f64>();
        let mut losses = vec![];
        for _ in 0..count(r) {
            let i = (random.uniform() * 4.) as usize;
            let u = random.uniform();
            let mut c = 0.;
            let mut j = 3;
            for k in 0..4 {
                c += p[i * 4 + k];
                if u < c {
                    j = k;
                    break;
                }
            }
            losses.push((readout[j] - mu[i * groups / 4]).powi(2));
        }
        calc.push([groups as f64, mean(&losses)]);
        pred.push([groups as f64, exact]);
        errors.push(sem(&losses));
    }
    let mut o = ExperimentResult::new(
        13,
        "Prediction under descriptor refinement",
        "Nested partitions of an exact four-state transition law; independent validation draws",
    );
    pair(
        &mut o,
        "Held-out one-step squared prediction error",
        "partition cells",
        "MSE",
        calc,
        pred,
    );
    o.details = json!({"standard_errors":errors,"transition":p,"conditional_predictor":target,"fitting":"Exact conditional law, independent random validation sample"});
    o.metric("Validation draws per partition", count(r) as f64, "draws");
    o
}
fn writes(r: &ExperimentRequest) -> ExperimentResult {
    let p = amp(r) / 2.;
    let x = [-1., 0.2, 1.4];
    let donors = [1, 2, 0];
    let mut changes = vec![];
    let mut expected = 0.;
    let mut mass = 0.;
    for b in 0usize..8 {
        let mut y = x;
        let mut w = 1.;
        for i in 0..3 {
            if b & (1 << i) != 0 {
                y[i] = x[donors[i]];
                w *= p;
            } else {
                w *= 1. - p;
            }
        }
        let delta = y.iter().sum::<f64>() - x.iter().sum::<f64>();
        expected += w * delta;
        mass += w;
        changes.push([b as f64, delta]);
    }
    let swap = [x[1], x[0], x[2]];
    let a = vec![
        C::from(x[0]),
        C::ONE,
        C::ZERO,
        C::from(x[1]),
        C::ZERO,
        C::ONE,
        C::from(x[2]),
        C::ONE,
        C::ONE,
    ];
    let mut b = a.clone();
    for j in 0..3 {
        b.swap(j, 3 + j);
    }
    let mut o = ExperimentResult::new(
        14,
        "Joint writes, roles and cancellation",
        "Exact simultaneous donor-write enumeration on three recipients",
    );
    o.metric("Joint law mass", mass, "")
        .metric("Mean total increment", expected, "")
        .metric(
            "Mutual swap total increment",
            swap.iter().sum::<f64>() - x.iter().sum::<f64>(),
            "",
        )
        .metric(
            "Determinant parity residual",
            (determinant(&a, 3) + determinant(&b, 3)).abs(),
            "",
        );
    o.plot(
        "Complete simultaneous-write budget",
        "gate bitmask",
        "total increment",
        vec![Series::line("Realized increment", changes)],
    );
    o.details = json!({"source_values":x,"donor_slots":donors,"acceptance":p,"write_semantics":"Every recipient reads the immutable source frame"});
    o.note("Expectation cancellation here uses a balanced cyclic donor map. Individual unpaired writes can change the total.");
    o
}
fn cp(r: &ExperimentRequest) -> ExperimentResult {
    let cols = color_fixture(angle(r));
    let triangle = dot(&cols[0], &cols[1]) * dot(&cols[1], &cols[2]) * dot(&cols[2], &cols[0]);
    let reflected: Vec<[C; 3]> = cols.iter().map(|x| x.map(C::conj)).collect();
    let conjugate = dot(&reflected[0], &reflected[1])
        * dot(&reflected[1], &reflected[2])
        * dot(&reflected[2], &reflected[0]);
    let asym = amp(r) / 2.;
    let mut o = ExperimentResult::new(
        15,
        "Invariant statistics and CP",
        "Explicit complex-conjugation involution with symmetric or biased two-orbit law",
    );
    o.metric(
        "CP conjugation residual",
        (conjugate - triangle.conj()).abs(),
        "",
    )
    .metric(
        "Symmetric CP-odd expectation",
        (triangle.im + conjugate.im) / 2.,
        "",
    )
    .metric(
        "Biased CP-odd expectation",
        ((1. + asym) * triangle.im + (1. - asym) * conjugate.im) / 2.,
        "",
    )
    .metric("Fundamental SU(3) Casimir", 4. / 3., "");
    o.plot(
        "Odd expectation under the complete orbit law",
        "orbit bias",
        "imaginary triangle",
        vec![
            Series::line(
                "Enumerated mean",
                (0..65)
                    .map(|i| {
                        let b = -1. + i as f64 / 32.;
                        [
                            b,
                            (1. + b) * triangle.im / 2. + (1. - b) * conjugate.im / 2.,
                        ]
                    })
                    .collect(),
            ),
            Series::line(
                "Bias × odd observable",
                (0..65)
                    .map(|i| {
                        let b = -1. + i as f64 / 32.;
                        [b, b * triangle.im]
                    })
                    .collect(),
            ),
        ],
    );
    o.details = json!({"triangle":complex_json(triangle),"cp_triangle":complex_json(conjugate),"coupling_units":"dimensionless algorithmic invariant; no physical coupling calibration"});
    o
}
fn generating(r: &ExperimentRequest) -> ExperimentResult {
    let (p, _, _) = memory_model(amp(r));
    let values = [-1., -1., 1., 1.];
    let mut histories = vec![];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                histories.push((
                    p[i * 4 + j] * p[j * 4 + k] / 4.,
                    values[i] + values[j] + values[k],
                ));
            }
        }
    }
    let z = |h: f64| {
        histories
            .iter()
            .map(|(p, x)| p * (h * x).exp())
            .sum::<f64>()
    };
    let h = angle(r) * 0.3;
    let dz = histories
        .iter()
        .map(|(p, x)| p * x * (h * x).exp())
        .sum::<f64>()
        / z(h);
    let eps = 1e-5;
    let fd = (z(h + eps).ln() - z(h - eps).ln()) / (2. * eps);
    let mut o = ExperimentResult::new(
        16,
        "Effective action and generating functions",
        "Complete enumeration of three-frame paths of the four-state reference chain",
    );
    o.metric("Path probability mass", z(0.), "")
        .metric("Source derivative", dz, "")
        .metric("Finite-difference residual", (dz - fd).abs(), "");
    o.plot(
        "Log generating function",
        "source",
        "log Z",
        vec![Series::line(
            "Exact path marginal",
            (0..65)
                .map(|i| {
                    let h = -1. + i as f64 / 32.;
                    [h, z(h).ln()]
                })
                .collect(),
        )],
    );
    o.details = json!({"histories":64,"source":h,"descriptor_law":histories.iter().fold(std::collections::BTreeMap::<i32,f64>::new(),|mut a,(p,x)|{*a.entry(*x as i32).or_default()+=p;a}),"action":"minus log full path probability; descriptor probabilities are summed before taking logs"});
    o
}
fn path_action(r: &ExperimentRequest) -> ExperimentResult {
    let theta = angle(r) * 0.2;
    let sigma = 0.3 + rate(r);
    let p = 0.1 + 0.4 * amp(r);
    let x = 0.7;
    let choice = p;
    let gate: f64 = 0.35;
    let survival: f64 = 0.9;
    let action_choice = -choice.ln();
    let action_gate = -gate.ln();
    let gaussian_action = 0.5 * ((x - theta) / sigma).powi(2) + sigma.ln() + 0.5 * TAU.ln();
    let action_survival = -survival.ln();
    let total = action_choice + action_gate + gaussian_action + action_survival;
    let density = choice * gate * survival * gaussian((x - theta) / sigma) / sigma;
    let mut o = ExperimentResult::new(
        17,
        "Executed path-action accounting",
        "Finite companion and gate law with a nonsingular scalar Gaussian conditional kernel",
    );
    o.metric(
        "Action versus joint density residual",
        (total + density.ln()).abs(),
        "",
    )
    .metric("Gaussian log determinant contribution", sigma.ln(), "")
    .metric("Joint transition subdensity", density, "");
    o.plot(
        "Additive action contributions",
        "term",
        "negative log likelihood",
        vec![Series::line(
            "Conditional factors",
            vec![
                [0., action_choice],
                [1., action_gate],
                [2., gaussian_action],
                [3., action_survival],
            ],
        )],
    );
    o.details = json!({"terms":["companion","accepted clone gate","Gaussian conditional innovation","survival"],"joint_density":density,"conditional_gaussian_variance":sigma*sigma,"source_mean":theta,"normalization":"subprobability before survival conditioning"});
    o.note("Deterministic copies are recorded maps of the source state; this conditional factorization does not assign them a Lebesgue density.");
    o
}
fn recorded_kinetic(a: &RunArchive<f64>) -> Result<ExperimentResult> {
    use crate::{noise::InnovationLaw, random::Stream};
    let mut observed2 = 0.;
    let mut expected2 = 0.;
    let mut factor_variance = 0.;
    let mut observed4 = 0.;
    let mut expected4 = 0.;
    let mut count = 0usize;
    let mut rows = vec![];
    let mut measured = vec![];
    let mut predicted = vec![];
    for step in &a.steps {
        for noise in &step.noise {
            if noise.stream != Stream::Kinetic || noise.substep != 2 {
                continue;
            }
            let input = step
                .stages
                .iter()
                .find(|s| s.stage == "A1")
                .ok_or_else(|| GasError::Capability("O-stage eligibility unavailable".into()))?;
            let executed = step
                .field_evaluations
                .iter()
                .find(|s| s.stage == "O" && s.field == "executed_noise")
                .ok_or_else(|| GasError::Capability("executed O-stage noise unavailable".into()))?;
            if executed.values != noise.sample {
                return Err(GasError::Capability(
                    "provider changed the sampled innovation law".into(),
                ));
            }
            let cumulant = match noise.innovation_law {
                Some(InnovationLaw::Gaussian) => 0.,
                Some(InnovationLaw::StandardizedUniform) => -1.2,
                None => {
                    return Err(GasError::Capability(
                        "innovation fourth moment unavailable".into(),
                    ));
                }
            };
            let d = noise.dimension;
            for i in 0..noise.rows {
                if !input.validity[i].eligible(a.gas_config.include_truncated) {
                    continue;
                }
                let factor = noise.dense_factor(i)?;
                let rank = factor.len() / d;
                let mut shift = vec![0.; rank];
                for sh in &noise.applied_source_shifts {
                    sh.validate(noise.rows, rank)?;
                    if sh.walker == i {
                        shift[sh.coordinate] += sh.shift;
                    }
                }
                for j in 0..d {
                    let b = &factor[j * rank..(j + 1) * rank];
                    let mu = b.iter().zip(&shift).map(|(x, y)| x * y).sum::<f64>();
                    let var = b.iter().map(|x| x * x).sum::<f64>();
                    let m2 = mu * mu + var;
                    let m4 = mu.powi(4)
                        + 6. * mu * mu * var
                        + 3. * var * var
                        + cumulant * b.iter().map(|x| x.powi(4)).sum::<f64>();
                    let value = noise.sample[i * d + j];
                    if !value.is_finite() {
                        return Err(GasError::Numerical("nonfinite executed innovation".into()));
                    }
                    observed2 += value * value;
                    expected2 += m2;
                    factor_variance += var;
                    observed4 += value.powi(4);
                    expected4 += m4;
                    count += 1;
                }
            }
            if count > 0 {
                measured.push([count as f64, observed2 / count as f64]);
                predicted.push([count as f64, expected2 / count as f64]);
                rows.push(json!({"epoch":step.epoch,"step":step.report.step,"innovation_law":noise.innovation_law,"cumulative_components":count,"second_moment_residual_sum":observed2-expected2,"fourth_moment_residual_sum":observed4-expected4}));
            }
        }
    }
    if count == 0 {
        return Err(GasError::Capability(
            "no eligible recorded O-stage innovations".into(),
        ));
    }
    let mut o = ExperimentResult::new(
        18,
        "Executed conditional kinetic moments",
        "Recorded eligible raw B xi with its actual factor, source mean, and innovation fourth cumulant",
    );
    o.metric(
        "Recorded mean squared factor innovation",
        observed2 / count as f64,
        "",
    )
    .metric(
        "Recorded predicted second moment",
        expected2 / count as f64,
        "",
    )
    .metric("Recorded fourth moment", observed4 / count as f64, "")
    .metric(
        "Recorded predicted fourth moment",
        expected4 / count as f64,
        "",
    )
    .metric(
        "Recorded mean factor covariance diagonal",
        factor_variance / count as f64,
        "",
    );
    pair(
        &mut o,
        "Recorded conditional second moments",
        "eligible components",
        "second moment",
        measured,
        predicted,
    );
    o.details = json!({"status":"available","recorded_components":count,"conditional_moment_trace":rows,"second_moment_residual_sum":observed2-expected2,"fourth_moment_residual_sum":observed4-expected4,"stage":"A1 eligibility; raw B xi at O before thermostat scaling","normalization":"Eligible executed components only; conditional predictions precede their innovations; covariance across coordinates and temporal dependence are retained in the recorded sample.","fourth_moment_formula":"mu^4 + 6 mu^2 variance + 3 variance^2 + fourth_cumulant * sum_k B_jk^4"});
    Ok(o)
}
fn kinetic(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> ExperimentResult {
    if let Some(a) = archive {
        return recorded_kinetic(a).unwrap_or_else(|error| {
            let mut o = ExperimentResult::new(
                18,
                "Recorded kinetic moments",
                "Executed O-stage noise with unavailable conditional moment data",
            );
            o.details = json!({"status":"unavailable","reason":error.to_string()});
            o.note(error.to_string());
            o
        });
    }
    let s = 0.3 + amp(r);
    let mut random = rng(r);
    let mut moments = vec![];
    let mut fourth = vec![];
    for _ in 0..count(r) {
        let z = s * random.normal();
        moments.push(z * z);
        fourth.push(z.powi(4));
    }
    let mut o = ExperimentResult::new(
        18,
        "Kinetic metric and fourth moments",
        "Independent Gaussian innovations with declared scalar factor",
    );
    o.metric("Sample second moment", mean(&moments), "")
        .metric("Predicted second moment", s * s, "")
        .metric("Second moment SEM", sem(&moments), "")
        .metric("Sample fourth moment", mean(&fourth), "")
        .metric("Gaussian fourth moment", 3. * s.powi(4), "");
    let mut run = 0.;
    let a = moments
        .iter()
        .enumerate()
        .map(|(i, x)| {
            run += x;
            [(i + 1) as f64, run / (i + 1) as f64]
        })
        .collect();
    let b = vec![[1., s * s], [moments.len() as f64, s * s]];
    pair(
        &mut o,
        "Covariance convergence",
        "independent draws",
        "second moment",
        a,
        b,
    );
    o
}
fn response(r: &ExperimentRequest) -> ExperimentResult {
    let theta = amp(r) - 0.5;
    let mut random = rng(r);
    let z: Vec<f64> = (0..count(r)).map(|_| random.normal()).collect();
    let mut rerun = vec![];
    let mut weighted = vec![];
    let mut predictions = vec![];
    let mut derivatives = vec![];
    let mut second = vec![];
    for &x in &z {
        let indicator = f64::from(x > 0.);
        derivatives.push(indicator * x);
        second.push(indicator * (x * x - 1.));
    }
    for k in 0..25 {
        let t = -1. + 2. * k as f64 / 24.;
        let a: Vec<f64> = z.iter().map(|&x| f64::from(x + t > 0.)).collect();
        let b: Vec<f64> = z
            .iter()
            .map(|&x| f64::from(x > 0.) * (t * x - t * t / 2.).exp())
            .collect();
        rerun.push([t, mean(&a)]);
        weighted.push([t, mean(&b)]);
        predictions.push([t, normal_cdf(t)]);
    }
    let a: Vec<f64> = z.iter().map(|&x| f64::from(x + theta > 0.)).collect();
    let b: Vec<f64> = z
        .iter()
        .map(|&x| f64::from(x > 0.) * (theta * x - theta * theta / 2.).exp())
        .collect();
    let mut o = ExperimentResult::new(
        19,
        "Gaussian source response through a threshold",
        "Reserved Gaussian source coordinate and discontinuous sign readout",
    );
    o.metric("Shifted mean", mean(&a), "")
        .metric("Reweighted mean", mean(&b), "")
        .metric("Shifted SEM", sem(&a), "")
        .metric("Reweighted SEM", sem(&b), "")
        .metric("First Hermite derivative", mean(&derivatives), "")
        .metric("Predicted derivative at zero", gaussian(0.), "")
        .metric("Second Hermite derivative", mean(&second), "");
    o.plot(
        "Source response",
        "source shift",
        "positive probability",
        vec![
            Series::line("Full threshold rerun", rerun),
            Series::line("Likelihood reweighting", weighted),
            Series::line("Independent Gaussian quadrature", predictions),
        ],
    );
    o.note("Each rerun reevaluates the threshold with the shifted innovation. Freezing the original threshold decision would suppress the response.");
    o
}
fn information(r: &ExperimentRequest) -> ExperimentResult {
    let t = amp(r);
    let mut random = rng(r);
    let mut ll = vec![];
    let mut fisher = vec![];
    for _ in 0..count(r) {
        let z = random.normal();
        let x = z + t;
        ll.push(t * x - t * t / 2.);
        fisher.push(z * z);
    }
    let p = normal_cdf(t);
    let coarse = p * (2. * p).ln() + (1. - p) * (2. * (1. - p)).ln();
    let mut o = ExperimentResult::new(
        20,
        "Force sources and information loss",
        "Gaussian mean-shift path law and its binary sign descriptor",
    );
    o.metric("Sample path KL", mean(&ll), "nats")
        .metric("Exact path KL", t * t / 2., "nats")
        .metric("Descriptor KL", coarse, "nats")
        .metric("Sample path Fisher", mean(&fisher), "")
        .metric("Descriptor Fisher at zero", 2. / PI, "");
    o.plot(
        "Data processing under the sign map",
        "source amplitude",
        "relative entropy",
        vec![
            Series::line(
                "Full Gaussian law",
                (0..33)
                    .map(|i| {
                        let t = i as f64 / 16.;
                        [t, t * t / 2.]
                    })
                    .collect(),
            ),
            Series::line(
                "Sign descriptor",
                (0..33)
                    .map(|i| {
                        let t = i as f64 / 16.;
                        let p = normal_cdf(t);
                        [t, p * (2. * p).ln() + (1. - p) * (2. * (1. - p)).ln()]
                    })
                    .collect(),
            ),
        ],
    );
    o.details = json!({"score":"x - source","path_fisher":1.,"disjoint_independent_source_cross_information":0.,"KL_sample_SEM":sem(&ll)});
    o
}
fn disintegration(r: &ExperimentRequest) -> ExperimentResult {
    let a = amp(r);
    let h = angle(r);
    let energy = [0., 0.3 + a, 0.7, 0.2 + 0.4 * a, 1., 0.5];
    let weight: Vec<f64> = energy
        .iter()
        .enumerate()
        .map(|(k, e)| (-e + h * (k % 3) as f64 / 3.).exp())
        .collect();
    let z = weight.iter().sum::<f64>();
    let joint: Vec<f64> = weight.iter().map(|x| x / z).collect();
    let marg = [
        joint[..3].iter().sum::<f64>(),
        joint[3..].iter().sum::<f64>(),
    ];
    let cond: Vec<f64> = joint
        .iter()
        .enumerate()
        .map(|(i, p)| p / marg[i / 3])
        .collect();
    let residual = joint
        .iter()
        .enumerate()
        .map(|(i, p)| (p.ln() - marg[i / 3].ln() - cond[i].ln()).abs())
        .fold(0., f64::max);
    let mut o = ExperimentResult::new(
        21,
        "Geometry-fiber disintegration",
        "Exact two-geometry by three-field Gibbs table",
    );
    o.metric("Action decomposition residual", residual, "")
        .metric("Joint mass", joint.iter().sum(), "")
        .metric("First fiber mass", cond[..3].iter().sum(), "")
        .metric("Second fiber mass", cond[3..].iter().sum(), "");
    o.plot(
        "Joint and conditional probabilities",
        "geometry-field cell",
        "probability",
        vec![
            Series::line(
                "Joint",
                joint
                    .iter()
                    .enumerate()
                    .map(|(i, p)| [i as f64, *p])
                    .collect(),
            ),
            Series::line(
                "Conditional fiber",
                cond.iter()
                    .enumerate()
                    .map(|(i, p)| [i as f64, *p])
                    .collect(),
            ),
        ],
    );
    o.details =
        json!({"geometry_marginal":marg,"fiber_conditionals":cond,"joint":joint,"partition":z});
    o
}
fn noether(r: &ExperimentRequest) -> ExperimentResult {
    let a = (-rate(r)).exp();
    let sigma = (1. - a * a).sqrt();
    let mut random = rng(r);
    let mut averages = vec![0.; steps(r) + 1];
    let mut predictions = vec![];
    let mut terminal = vec![];
    for _ in 0..count(r) {
        let mut x = 1.;
        let mut m = 0.;
        for average in averages.iter_mut().skip(1) {
            let y = a * x + sigma * random.normal();
            m += y - x - (a - 1.) * x;
            x = y;
            *average += m / count(r) as f64;
        }
        terminal.push(m);
    }
    for k in 0..=steps(r) {
        predictions.push([k as f64, 0.]);
    }
    let mut o = ExperimentResult::new(
        22,
        "Stochastic Noether balance",
        "Stationary-variance autoregressive kernel with independently sampled continuations",
    );
    o.metric("Terminal martingale mean", mean(&terminal), "")
        .metric("Terminal mean SEM", sem(&terminal), "")
        .metric(
            "Predicted martingale variance",
            steps(r) as f64 * sigma * sigma,
            "",
        );
    pair(
        &mut o,
        "Integrated increment minus conditional drift",
        "step",
        "mean residual",
        averages
            .iter()
            .enumerate()
            .map(|(i, x)| [i as f64, *x])
            .collect(),
        predictions,
    );
    o.details = json!({"conditional_drift":"(exp(-rate)-1) x","innovation_variance":sigma*sigma,"independent_continuations":count(r)});
    o
}
fn ward(r: &ExperimentRequest) -> ExperimentResult {
    let a = angle(r);
    let b = amp(r);
    let action =
        |t: f64| 1. - trace(&mul(&su2(t, [1., 0., 0.]), &su2(b, [0., 1., 0.]), 2), 2).re / 2.;
    let exact = a.sin() * b.cos();
    let mut measured = vec![];
    let mut prediction = vec![];
    for k in 0..16 {
        let h = 10f64.powf(-1. - k as f64 / 4.);
        measured.push([
            h,
            ((action(a + h) - action(a - h)) / (2. * h) - exact).abs(),
        ]);
        prediction.push([h, h * h / 6.]);
    }
    let loopmat = mul(&su2(a, [1., 0., 0.]), &su2(b, [0., 1., 0.]), 2);
    let g = su2(0.43, [0., 0., 1.]);
    let transformed = mul(&mul(&g, &loopmat, 2), &adjoint(&g, 2), 2);
    let mut o = ExperimentResult::new(
        23,
        "Wilson variation and Ward identity",
        "Noncommuting two-link closed product under simultaneous basepoint conjugation",
    );
    o.metric("Analytic action derivative", exact, "").metric(
        "Gauge variation residual",
        (trace(&loopmat, 2) - trace(&transformed, 2)).abs(),
        "",
    );
    pair(
        &mut o,
        "Central-difference variation error",
        "variation size",
        "absolute error",
        measured,
        prediction,
    );
    o.details = json!({"action":"1 - Re Tr(Ux Uy)/2","derivative":"sin(angle) cos(amplitude)","ward_action":"simultaneous conjugation of the closed product"});
    o
}
fn yang_mills(r: &ExperimentRequest) -> ExperimentResult {
    let a = 0.2 + amp(r);
    let b = rate(r);
    let ax = vec![C::ZERO, C::new(0., a), C::new(0., a), C::ZERO];
    let ay = vec![C::ZERO, C::from(b), C::from(-b), C::ZERO];
    let xy = mul(&ax, &ay, 2);
    let yx = mul(&ay, &ax, 2);
    let comm: Vec<C> = xy.iter().zip(yx).map(|(&x, y)| x - y).collect();
    let norm = comm.iter().map(|z| z.abs2()).sum::<f64>().sqrt();
    let mut measured = vec![];
    let mut predicted = vec![];
    let mut error = vec![];
    for k in 0..25 {
        let h = 0.01 + 0.79 * k as f64 / 24.;
        let x = su2(a * h, [1., 0., 0.]);
        let y = su2(b * h, [0., 1., 0.]);
        let u = mul(
            &mul(&mul(&x, &y, 2), &adjoint(&x, 2), 2),
            &adjoint(&y, 2),
            2,
        );
        let f: Vec<C> = u
            .iter()
            .zip(identity(2))
            .map(|(&x, y)| (x - y) / (h * h))
            .collect();
        measured.push([h, f.iter().map(|z| z.abs2()).sum::<f64>().sqrt()]);
        predicted.push([h, norm]);
        error.push([h, difference(&f, &comm)]);
    }
    let mut o = ExperimentResult::new(
        24,
        "Smooth-connection continuum consistency",
        "Constant non-Abelian connection with exact exponential edge transport",
    );
    o.metric("Curvature commutator norm", norm, "");
    pair(
        &mut o,
        "Plaquette curvature",
        "edge length",
        "Frobenius norm",
        measured,
        predicted,
    );
    o.plot(
        "Matrix-valued continuum error",
        "edge length",
        "residual norm",
        vec![Series::line("(U-square - I)/h² minus commutator", error)],
    );
    o.note("The constant connection still has nonzero curvature because its two components do not commute.");
    o
}
fn gaps(r: &ExperimentRequest) -> ExperimentResult {
    let h = param(r, "dt", 0.1, 0.005, 0.2);
    let q = rate(r);
    let p = (1. - (-2. * q * h).exp()) / 2.;
    let transition = [1. - p, p, p, 1. - p];
    let mut curve = vec![];
    let mut pred = vec![];
    for n in 0..=steps(r) {
        let a = rpower(&transition, 2, n);
        curve.push([n as f64 * h, a[0] - a[1]]);
        pred.push([n as f64 * h, (-2. * q * n as f64 * h).exp()]);
    }
    let gap = -((1. - 2. * p).ln()) / h;
    let mut o = ExperimentResult::new(
        25,
        "Discrete time, units and gaps",
        "Exact sampled two-state continuous-time flip chain",
    );
    o.metric("One-step invariant eigenvalue", 1., "")
        .metric("One-step centered eigenvalue", 1. - 2. * p, "")
        .metric("Generator decay rate", gap, "per algorithmic time")
        .metric("Rate identity residual", (gap - 2. * q).abs(), "");
    pair(
        &mut o,
        "Centered relaxation",
        "algorithmic time",
        "correlation",
        curve,
        pred,
    );
    let mut euler = vec![];
    let mut exact_rates = vec![];
    let mut target_rates = vec![];
    for j in 0..24 {
        let dt = h * 0.8_f64.powi(j);
        let discrete = -(1. - 2. * q * dt).ln() / dt;
        euler.push([dt, discrete]);
        exact_rates.push([dt, -(-2. * q * dt).exp().ln() / dt]);
        target_rates.push([dt, 2. * q]);
    }
    o.plot(
        "Generator-rate refinement",
        "timestep",
        "centered decay rate",
        vec![
            Series::line("Euler transition -log(lambda)/h", euler),
            Series::line("Exact semigroup transition", exact_rates),
            Series::line("Continuous generator", target_rates),
        ],
    );
    o.details = json!({"timestep":h,"transition":transition,"physical_time_conversion":"Multiply algorithmic time by the explicitly supplied time unit before quoting a physical rate"});
    o
}
fn collision(r: &ExperimentRequest) -> ExperimentResult {
    let p = amp(r) / 2.;
    let x = -0.4;
    let y = 1.2;
    let jump = y - x;
    let events = [(1. - p, [x, y]), (p, [y, x])];
    let mut ex = [0.; 2];
    let mut exy = 0.;
    let mut dxy = 0.;
    let mut firstorder = 0.;
    let mut cross = 0.;
    for (w, z) in events {
        let dx = z[0] - x;
        let dy = z[1] - y;
        ex[0] += w * dx;
        ex[1] += w * dy;
        exy += w * dx * dy;
        dxy += w * (z[0] * z[1] - x * y);
        firstorder += w * (x * dy + y * dx);
        cross += w * dx * dy;
    }
    let cov = exy - ex[0] * ex[1];
    let mut o = ExperimentResult::new(
        26,
        "Complete collision fluctuation budget",
        "One shared Bernoulli gate performing a simultaneous mutual swap",
    );
    o.metric("Cross-walker increment covariance", cov, "")
        .metric("Predicted covariance", -p * (1. - p) * jump * jump, "")
        .metric(
            "Product increment residual",
            (dxy - firstorder - cross).abs(),
            "",
        )
        .metric("Total momentum increment", ex[0] + ex[1], "");
    o.plot(
        "Finite-step product budget",
        "term",
        "expected increment",
        vec![Series::line(
            "Exact terms",
            vec![
                [0., dxy],
                [1., firstorder],
                [2., cross],
                [3., firstorder + cross],
            ],
        )],
    );
    o.details = json!({"terms":["product increment","first-order terms","quadratic cross term","complete expansion"],"joint_gate_probability":p,"independent_gate_covariance_would_be":0.});
    o
}
fn qsd(r: &ExperimentRequest) -> ExperimentResult {
    let killing = 0.04 + 0.12 * amp(r);
    let q = [0.7, 0.3 - killing, 0.2, 0.74];
    let alpha = (q[0] + q[3] + ((q[0] - q[3]).powi(2) + 4. * q[1] * q[2]).sqrt()) / 2.;
    let ratio = (alpha - q[0]) / q[2];
    let pi = [1. / (1. + ratio), ratio / (1. + ratio)];
    let mut row = vec![1., 0.];
    let mut data = vec![];
    let mut survival = vec![];
    let mut bounds = vec![];
    let mut windows = vec![];
    let mut bound_excess: f64 = 0.;
    for n in 0..=steps(r) {
        let mass = row.iter().sum::<f64>();
        let tv = ((row[0] / mass - pi[0]).abs() + (row[1] / mass - pi[1]).abs()) / 2.;
        data.push([n as f64, tv]);
        survival.push([n as f64, mass]);
        let bound = (2. * alpha.powi(-2) * tv).min(1.);
        bounds.push([n as f64, bound]);
        let mut law = vec![];
        let mut target = vec![];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    law.push(row[i] / mass * q[i * 2 + j] * q[j * 2 + k]);
                    target.push(pi[i] * q[i * 2 + j] * q[j * 2 + k] / (alpha * alpha));
                }
            }
        }
        let path_mass: f64 = law.iter().sum();
        let window_tv = law
            .iter()
            .zip(target)
            .map(|(x, y)| (x / path_mass - y).abs())
            .sum::<f64>()
            / 2.;
        windows.push([n as f64, window_tv]);
        bound_excess = bound_excess.max(window_tv - bound);
        row = rowmul(&row, &q);
    }
    let eig = rowmul(&pi, &q);
    let mut o = ExperimentResult::new(
        27,
        "QSD windows and burn-in",
        "Exactly solved two-state substochastic killed chain",
    );
    o.metric("Window bound excess", bound_excess, "");
    o.metric("Survival eigenvalue", alpha, "").metric(
        "QSD eigenmeasure residual",
        (eig[0] - alpha * pi[0]).abs() + (eig[1] - alpha * pi[1]).abs(),
        "",
    );
    o.plot(
        "Conditional relaxation and survival",
        "steps",
        "probability",
        vec![
            Series::line("Conditioned-state TV", data),
            Series::line("Survival", survival),
            Series::line("Two-step window bound", bounds),
            Series::line("Enumerated two-step path TV", windows),
        ],
    );
    o.details = json!({"killed_transition":q,"qsd":pi,"window":2,"window_bound":"min(1, 2 alpha^(-window) initial-law TV)","conditioning":"survival of the killed chain"});
    o
}
fn reflection(r: &ExperimentRequest) -> ExperimentResult {
    let u = 0.1 + amp(r);
    let v = rate(r);
    let c = -(u + v) / 2.;
    let h = [1., u, v, 0.];
    let vector = [c, 1.];
    let value = (0..4)
        .map(|k| vector[k / 2] * h[k] * vector[k % 2])
        .sum::<f64>();
    let prediction = -(u + v).powi(2) / 4.;
    let symmetric = [1., (u + v) / 2., (u + v) / 2., 0.];
    let (e, _) = symmetric_eigen(&symmetric, 2);
    let mut o = ExperimentResult::new(
        28,
        "Localized reflection matrices",
        "Exact labeled-word reflection counterexample with off-diagonal weights u and v",
    );
    o.metric("Reflected quadratic value", value, "")
        .metric("Predicted negative value", prediction, "")
        .metric(
            "Quadratic identity residual",
            (value - prediction).abs(),
            "",
        )
        .metric("Hermitian defect", (u - v).abs(), "")
        .metric(
            "Minimum Hermitian-part eigenvalue",
            e.into_iter().fold(f64::INFINITY, f64::min),
            "",
        );
    o.plot(
        "Reflection sign",
        "word coefficient",
        "quadratic form",
        vec![Series::line(
            "Direct matrix contraction",
            (0..65)
                .map(|i| {
                    let c = -3. + i as f64 / 16.;
                    [c, c * c + (u + v) * c]
                })
                .collect(),
        )],
    );
    o.details = json!({"reflection_matrix":h,"negative_word":vector});
    o
}
fn faces(r: &ExperimentRequest) -> ExperimentResult {
    let v = [0., angle(r), amp(r), 0.2];
    let hbar = 0.7;
    let link = |i: usize, j: usize| C::phase((v[j] - v[i]) / hbar);
    let outer = link(0, 1) * link(1, 2) * link(2, 3) * link(3, 0);
    let tri = C::phase((v[1] - v[0]) / hbar);
    let mut o = ExperimentResult::new(
        29,
        "Scalar triangle and outer-plaquette observables",
        "Scalar potential-difference links on a four-corner face",
    );
    o.metric("Outer plaquette defect", (C::ONE - outer).abs(), "")
        .metric("Triangle readout defect", 1. - tri.re, "")
        .metric(
            "Triangle formula residual",
            (1. - tri.re - (1. - ((v[1] - v[0]) / hbar).cos())).abs(),
            "",
        );
    o.plot(
        "Two scalar face constructions",
        "potential difference",
        "real defect",
        vec![
            Series::line(
                "Triangle readout",
                (0..65)
                    .map(|i| {
                        let x = -PI + TAU * i as f64 / 64.;
                        [x, 1. - (x / hbar).cos()]
                    })
                    .collect(),
            ),
            Series::line(
                "Closed outer potential-difference plaquette",
                vec![[-PI, 0.], [PI, 0.]],
            ),
        ],
    );
    o.details = json!({"potentials":v,"hbar_effective":hbar,"outer_product":complex_json(outer),"triangle_readout":complex_json(tri),"outer_reflected_vacuum_rank":1});
    o
}
fn translation(r: &ExperimentRequest) -> ExperimentResult {
    let scale = 0.3 + rate(r);
    let points = [-1.1, -0.2, 0.5, 1.3];
    let shift = amp(r);
    let f = |x: f64, anchor: f64| (-((x - anchor) / scale).powi(2) / 2.).exp();
    let initial = points.iter().map(|&x| f(x, 0.)).sum::<f64>();
    let moved = points.iter().map(|&x| f(x + shift, 0.)).sum::<f64>();
    let comoved = points.iter().map(|&x| f(x + shift, shift)).sum::<f64>();
    let mut o = ExperimentResult::new(
        30,
        "Translation and localized regulators",
        "Finite point cloud observed through an anchored Gaussian weight",
    );
    o.metric("Anchored translation defect", moved - initial, "")
        .metric("Co-transformed regulator residual", comoved - initial, "");
    o.plot(
        "Moving points and moving the regulator",
        "translation",
        "weighted mass",
        vec![
            Series::line(
                "Fixed regulator",
                (0..65)
                    .map(|i| {
                        let t = -2. + i as f64 / 16.;
                        [t, points.iter().map(|&x| f(x + t, 0.)).sum()]
                    })
                    .collect(),
            ),
            Series::line(
                "Regulator translated with the cloud",
                vec![[-2., initial], [2., initial]],
            ),
        ],
    );
    o.details =
        json!({"points":points,"regulator_scale":scale,"weights_are_globally_normalized":false});
    o.note("The anchored measurement changes with translation. Co-transforming the regulator is a different transformation of the input data.");
    o
}
fn interventions(r: &ExperimentRequest) -> ExperimentResult {
    let n = 9;
    let p = 0.1 + 0.2 * amp(r);
    let mut kernel = vec![0.; n * n];
    for i in 0..n {
        kernel[i * n + i] = 1. - 2. * p;
        kernel[i * n + (i + 1) % n] = p;
        kernel[i * n + (i + n - 1) % n] = p;
    }
    let mut a: Vec<f64> = vec![0.; n];
    a[0] = 1.;
    let mut data = vec![];
    let mut outside = 0.;
    let tmax = steps(r).min(8);
    for t in 0..=tmax {
        let mut leak = 0.;
        for (i, &v) in a.iter().enumerate() {
            if i.min(n - i) > t {
                leak += v.abs();
            }
        }
        outside += leak;
        data.push([t as f64, a[4]]);
        a = rowmul(&a, &kernel);
    }
    let mut o = ExperimentResult::new(
        31,
        "Regional algebras and interventions",
        "Nearest-neighbor linear update on a nine-site periodic graph",
    );
    o.metric("Response outside exact dependency cone", outside, "")
        .metric("Distance to selected target", 4., "edges");
    o.plot(
        "Intervention propagation",
        "updates",
        "target response",
        vec![Series::line("Full repeated update", data)],
    );
    o.details = json!({"kernel":kernel,"source_slot":0,"target_slot":4,"multiplication_algebra_commutator":0.,"locality":"Finite update dependency; CAR locality separately depends on covariance"});
    o
}
fn synthesis(r: &ExperimentRequest) -> ExperimentResult {
    let (p, a, b) = memory_model(amp(r));
    let f = [1., 1., -1., -1.];
    let mut correlation = vec![];
    let mut markov = vec![];
    let mut defects = vec![];
    for n in 0..=steps(r) {
        let q = rpower(&p, 4, n);
        let c = (0..16)
            .map(|k| f[k / 4] * q[k] * f[k % 4] / 4.)
            .sum::<f64>();
        correlation.push([n as f64, c]);
        markov.push([n as f64, a.powi(n as i32)]);
        defects.push([n as f64, 1. - c * c]);
    }
    let mean0 = f.iter().sum::<f64>() / 4.;
    let variance = f.iter().map(|x| (x - mean0).powi(2)).sum::<f64>() / 4.;
    let mut o = ExperimentResult::new(
        32,
        "Connected finite-record QFT dashboard",
        "One exact stationary four-state record used throughout the computation",
    );
    o.metric("Centered mean", mean0, "")
        .metric("Gram variance", variance, "")
        .metric("Projected one-step contraction", a, "")
        .metric("Two-step memory correction", b * b, "");
    o.plot(
        "One record across constructions",
        "lag",
        "value",
        vec![
            Series::line("Full conditional correlation", correlation),
            Series::line("Compressed Markov prediction", markov),
            Series::line("One-mode CAR multiplicative defect", defects),
        ],
    );
    o.details = json!({"transition":p,"pipeline":["stationary record law","centered readout","Gram space","projected multitime transition","one-mode CAR channel"],"stationarity_residual":maxabs(&rowmul(&[0.25;4],&p).iter().map(|x|x-0.25).collect::<Vec<_>>())});
    o
}
fn spinors(r: &ExperimentRequest) -> ExperimentResult {
    let t = angle(r);
    let a = amp(r);
    let l = [C::from(1.), C::phase(t) * 0.4];
    let m = [C::from(a), C::new(0.2, a * 0.6)];
    let p: Vec<C> = (0..4)
        .map(|k| l[k / 2] * l[k % 2].conj() + m[k / 2] * m[k % 2].conj())
        .collect();
    let energy = (p[0].re + p[3].re) / 2.;
    let momentum = [p[1].re, -p[1].im, (p[0].re - p[3].re) / 2.];
    let lorentz = energy * energy - momentum.iter().map(|x| x * x).sum::<f64>();
    let mass = epsilon(&l, &m).abs2();
    let null: Vec<C> = (0..4).map(|k| l[k / 2] * l[k % 2].conj()).collect();
    let mut o = ExperimentResult::new(
        33,
        "Spinor, Lorentz and mass identities",
        "Two explicit spinors and their Hermitian momentum bispinor",
    );
    o.metric(
        "Determinant–Lorentz residual",
        (determinant(&p, 2) - C::from(lorentz)).abs(),
        "",
    )
    .metric(
        "Two-spinor mass identity residual",
        (lorentz - mass).abs(),
        "",
    )
    .metric("Null spinor determinant", determinant(&null, 2).abs(), "");
    let mut d = vec![];
    let mut pred = vec![];
    for i in 0..65 {
        let u = -PI + TAU * i as f64 / 64.;
        let ll = [C::ONE, C::phase(u) * 0.4];
        let pp: Vec<C> = (0..4)
            .map(|k| ll[k / 2] * ll[k % 2].conj() + m[k / 2] * m[k % 2].conj())
            .collect();
        d.push([u, determinant(&pp, 2).re]);
        pred.push([u, epsilon(&ll, &m).abs2()]);
    }
    pair(
        &mut o,
        "Mass from two spinors",
        "relative phase",
        "mass squared",
        d,
        pred,
    );
    o.details = json!({"momentum":[energy,momentum[0],momentum[1],momentum[2]],"bispinor":p.iter().copied().map(complex_json).collect::<Vec<_>>(),"signature":"+---"});
    o
}
struct TwistorFrame {
    x: Vec<[f64; 3]>,
    v: Vec<[f64; 3]>,
    valid: Vec<bool>,
    triples: Vec<Option<[usize; 3]>>,
    donor_details: Vec<serde_json::Value>,
    step: u64,
    dt: f64,
}
fn twistor_frames(
    r: &ExperimentRequest,
    archive: Option<&RunArchive<f64>>,
) -> Result<Vec<TwistorFrame>> {
    if let Some(a) = archive {
        let (pos, vel, dt) = match &a.gas_config.kinetic.integrator {
            crate::kinetic::KineticKind::Baoab {
                positions,
                velocities,
                dt,
                ..
            } => (positions, velocities, *dt),
            _ => {
                return Err(GasError::Capability(
                    "recorded twistors require BAOAB position and velocity fields".into(),
                ));
            }
        };
        let mut out = vec![];
        for s in &a.steps {
            let stage = s
                .stages
                .iter()
                .find(|x| x.stage == "pre_clone")
                .ok_or_else(|| {
                    GasError::Capability("twistor archive lacks pre_clone stage".into())
                })?;
            let x = stage
                .fields
                .get(pos)
                .ok_or_else(|| GasError::MissingField(pos.clone()))?;
            let v = stage
                .fields
                .get(vel)
                .ok_or_else(|| GasError::MissingField(vel.clone()))?;
            if x.item_shape != [3] || v.item_shape != [3] {
                return Err(GasError::Shape(
                    "twistors require actual three-dimensional vectors".into(),
                ));
            }
            let mut xx = x
                .values
                .chunks_exact(3)
                .map(|q| [q[0], q[1], q[2]])
                .collect::<Vec<_>>();
            let mut vv = v
                .values
                .chunks_exact(3)
                .map(|q| [q[0], q[1], q[2]])
                .collect::<Vec<_>>();
            let mut valid: Vec<bool> = stage
                .validity
                .iter()
                .map(|x| x.eligible(a.gas_config.include_truncated))
                .collect();
            let mut triples = vec![None; xx.len()];
            let mut donor_details = vec![json!({"status":"no_companion"}); xx.len()];
            for (i, triple) in triples.iter_mut().enumerate() {
                let d = &s.report.distance_companions;
                let c = &s.report.cloning_companions;
                let dj = (0..d.count).find(|&j| d.valid[i * d.count + j]);
                let cj = (0..c.count).find(|&j| c.valid[i * c.count + j]);
                if let (Some(dj), Some(cj)) = (dj, cj) {
                    let ds = s.report.distance_sources[d.indices[i * d.count + dj] as usize];
                    let cs = s.report.clone_plan.sources[c.indices[i * c.count + cj] as usize];
                    let current = s.report.step.saturating_sub(1);
                    let mut mapped = [i, 0, 0];
                    let mut resolved = true;
                    let mut details = vec![];
                    for (role, src) in [ds, cs].iter().enumerate() {
                        let historical = src.frame != current || src.version != stage.version;
                        if !historical
                            && stage.generations.get(src.slot as usize) == Some(&src.generation)
                        {
                            mapped[role + 1] = src.slot as usize;
                            details.push(json!({"source":src,"age_steps":0,"tracking":"current_numerical_slot"}));
                        } else {
                            match super::path_action::source_observations(a, s.epoch, *src) {
                                Ok(obs) => {
                                    let sx = obs.field(pos)?.row(src.slot as usize)?;
                                    let sv = obs.field(vel)?.row(src.slot as usize)?;
                                    if sx.len() != 3 || sv.len() != 3 {
                                        return Err(GasError::Shape("historical twistor source needs three-dimensional fields".into()));
                                    }
                                    mapped[role + 1] = xx.len();
                                    xx.push([sx[0], sx[1], sx[2]]);
                                    vv.push([sv[0], sv[1], sv[2]]);
                                    valid.push(sx.iter().chain(sv).all(|x| x.is_finite()));
                                    details.push(json!({"source":src,"age_steps":current.saturating_sub(src.frame),"tracking":"immutable_historical_snapshot"}));
                                }
                                Err(error) => {
                                    resolved = false;
                                    details.push(json!({"source":src,"status":"unavailable","reason":error.to_string()}));
                                }
                            }
                        }
                    }
                    donor_details[i] = json!({"status":if resolved{"available"}else{"unavailable"},"donors":details});
                    if resolved {
                        *triple = Some(mapped);
                    }
                }
            }
            out.push(TwistorFrame {
                x: xx,
                v: vv,
                valid,
                triples,
                donor_details,
                step: s.report.step,
                dt,
            });
        }
        if out.is_empty() {
            return Err(GasError::Capability("twistors need recorded frames".into()));
        }
        Ok(out)
    } else {
        let mut out = vec![];
        for t in 0..=steps(r) {
            let time = t as f64 * 0.08;
            let mut x = vec![];
            let mut v = vec![];
            for i in 0..9 {
                let phase = TAU * i as f64 / 9.;
                x.push([
                    (phase + time).cos(),
                    (phase - time * 0.7).sin(),
                    (phase * 2. + time).sin() * (0.2 + amp(r)),
                ]);
                v.push([
                    -(phase + time).sin(),
                    -0.7 * (phase - time * 0.7).cos(),
                    (phase * 2. + time).cos() * (0.2 + amp(r)),
                ]);
            }
            let triples = (0..9)
                .map(|i| Some([i, (i + 1 + t / 3) % 9, (i + 3 + t / 5) % 9]))
                .collect();
            out.push(TwistorFrame {
                x,
                v,
                valid: vec![true; 9],
                triples,
                donor_details: vec![json!({"tracking":"current_numerical_slot"}); 9],
                step: t as u64,
                dt: 0.08,
            });
        }
        Ok(out)
    }
}
type TwistorObservation = (C, [C; 3], [f64; 5], [usize; 2]);
fn twistor_value(f: &TwistorFrame, tri: [usize; 3], alpha: f64) -> Option<TwistorObservation> {
    if tri.iter().any(|&i| {
        i >= f.x.len() || !f.valid[i] || f.x[i].iter().chain(&f.v[i]).any(|x| !x.is_finite())
    }) {
        return None;
    }
    let mut spins = [[C::ZERO; 2]; 2];
    let mut cols = [0; 2];
    for (n, &j) in tri[1..].iter().enumerate() {
        let dx = std::array::from_fn(|a| f.x[j][a] - f.x[tri[0]][a]);
        let dv = std::array::from_fn(|a| f.v[j][a] - f.v[tri[0]][a]);
        let (l, _, col) = edge_spinor(dx, dv, f.dt, alpha)?;
        spins[n] = l;
        cols[n] = col;
    }
    let (a, b) = (spins[0], spins[1]);
    let tau = epsilon(&a, &b);
    let w = [
        a[0].conj() * b[1] + a[1].conj() * b[0],
        C::new(0., -1.) * a[0].conj() * b[1] + C::new(0., 1.) * a[1].conj() * b[0],
        a[0].conj() * b[0] - a[1].conj() * b[1],
    ];
    let q = [
        (w[0] * w[1]).re,
        (w[0] * w[2]).re,
        (w[1] * w[2]).re,
        (w[0] * w[0] - w[1] * w[1]).re / 2f64.sqrt(),
        (w[2] * w[2] * 2. - w[0] * w[0] - w[1] * w[1]).re / 6f64.sqrt(),
    ];
    Some((tau, w, q, cols))
}
fn twistor_at_sink(
    source: &TwistorFrame,
    sink: &TwistorFrame,
    tri: [usize; 3],
    alpha: f64,
) -> Option<TwistorObservation> {
    let population = source.triples.len();
    let mut x = vec![];
    let mut v = vec![];
    let mut valid = vec![];
    for i in tri {
        let f = if i < population { sink } else { source };
        if i >= f.x.len() {
            return None;
        }
        x.push(f.x[i]);
        v.push(f.v[i]);
        valid.push(f.valid[i]);
    }
    let local = TwistorFrame {
        x,
        v,
        valid,
        triples: vec![],
        donor_details: vec![],
        step: sink.step,
        dt: sink.dt,
    };
    twistor_value(&local, [0, 1, 2], alpha)
}
fn twistors(r: &ExperimentRequest, archive: Option<&RunArchive<f64>>) -> Result<ExperimentResult> {
    let frames = twistor_frames(r, archive)?;
    let f = frames.last().unwrap();
    let alpha = 0.1 + amp(r);
    let mut values = vec![];
    let mut s = vec![];
    let mut p = vec![];
    let mut g = vec![];
    let mut mask = vec![];
    for (i, tri) in f.triples.iter().enumerate() {
        if let Some((tau, w, q, cols)) = tri.and_then(|tri| twistor_value(f, tri, alpha)) {
            s.push([i as f64, tau.re]);
            p.push([i as f64, tau.im]);
            g.push([i as f64, tau.abs2()]);
            mask.push(true);
            values.push(json!({"walker":i,"triplet":tri,"donor_provenance":f.donor_details[i],"tau":complex_json(tau),"vector":w.map(|z|z.re),"axial":w.map(|z|z.im),"tensor_components":q,"tensor_scalar":q.iter().sum::<f64>()/5.,"selected_columns":cols}));
        } else {
            mask.push(false);
        }
    }
    let mut o = ExperimentResult::new(
        34,
        "Effective triplet twistor operators",
        if archive.is_some() {
            "Actual pre-clone recipients and immutable distance/cloning donor source coordinates, including donor memory"
        } else {
            "Three-dimensional moving reference cloud with explicit companion maps"
        },
    );
    o.metric("Valid triplets", values.len() as f64, "triplets")
        .metric("Velocity scale", alpha, "");
    o.plot(
        "Twistor scalar channels",
        "source slot",
        "readout",
        vec![
            Series::line("Scalar", s),
            Series::line("Pseudoscalar", p),
            Series::line("Glueball-like", g),
        ],
    );
    o.details = json!({"step":f.step,"time_step":f.dt,"readouts":values,"valid_mask":mask,"dimension":3,"stage":"pre_clone","donor_sources":f.donor_details,"historical_donor_policy":"Resolve immutable donor coordinates by epoch, frame, version, slot and generation. Unretained sources have explicit unavailable provenance."});
    Ok(o)
}
fn lag_tracking(
    r: &ExperimentRequest,
    archive: Option<&RunArchive<f64>>,
) -> Result<ExperimentResult> {
    let frames = twistor_frames(r, archive)?;
    let alpha = 0.1 + amp(r);
    let maxlag = steps(r).min(frames.len().saturating_sub(1));
    let mut fixed_re = vec![];
    let mut fixed_im = vec![];
    let mut changed = vec![];
    let mut counts = vec![];
    for lag in 0..=maxlag {
        let mut sum = C::ZERO;
        let mut alternate = C::ZERO;
        let mut n = 0;
        let mut m = 0;
        for t in 0..frames.len() - lag {
            if frames[t + lag].step != frames[t].step + lag as u64
                || archive.is_some_and(|a| a.steps[t].epoch != a.steps[t + lag].epoch)
            {
                continue;
            }
            for tri in frames[t].triples.iter().flatten() {
                if let Some((src, ..)) = twistor_value(&frames[t], *tri, alpha) {
                    if let Some((sink, ..)) =
                        twistor_at_sink(&frames[t], &frames[t + lag], *tri, alpha)
                    {
                        sum = sum + src.conj() * sink;
                        n += 1;
                    }
                    if let Some(other) = frames[t + lag].triples[tri[0]]
                        && let Some((sink, ..)) = twistor_value(&frames[t + lag], other, alpha)
                    {
                        alternate = alternate + src.conj() * sink;
                        m += 1;
                    }
                }
            }
        }
        if n > 0 {
            fixed_re.push([lag as f64, sum.re / n as f64]);
            fixed_im.push([lag as f64, sum.im / n as f64]);
        }
        if m > 0 {
            changed.push([lag as f64, alternate.re / m as f64]);
        }
        counts.push(json!({"lag":lag,"fixed_valid":n,"reselected_valid":m}));
    }
    let mut o = ExperimentResult::new(
        35,
        "Source-frozen companion lag tracking",
        if archive.is_some() {
            "Recorded pre-clone recipient/current donor slots and frozen immutable historical donor snapshots"
        } else {
            "Moving three-dimensional reference cloud with time-dependent companion assignments"
        },
    );
    o.plot(
        "Complex twistor correlation",
        "lag",
        "correlation",
        vec![
            Series::line("Fixed source slots: real", fixed_re),
            Series::line("Fixed source slots: imaginary", fixed_im),
            Series::line("Sink-reselected: real", changed),
        ],
    );
    o.metric("Recorded frames", frames.len() as f64, "frames");
    o.details = json!({"lag_counts":counts,"tracking":"Source numerical slots advance to the sink; historical donor snapshots stay immutable. Epoch crossings and missing consecutive frames are excluded; these are source descriptors, not ancestry identities.","normalization":"Each curve divides by its own valid source/sink triplet count"});
    Ok(o)
}
fn hermitian2_modes(a: &[C]) -> (Vec<f64>, Vec<Vec<C>>) {
    let aa = a[0].re;
    let dd = a[3].re;
    let b = a[1];
    let rad = ((aa - dd).powi(2) + 4. * b.abs2()).sqrt();
    let vals = vec![(aa + dd - rad) / 2., (aa + dd + rad) / 2.];
    if rad < 1e-14 {
        return (vals, vec![vec![C::ONE, C::ZERO], vec![C::ZERO, C::ONE]]);
    }
    let mut vectors = vec![];
    for &l in &vals {
        let mut v = if b.abs() > 1e-14 {
            vec![b, C::from(l - aa)]
        } else if (l - aa).abs() < (l - dd).abs() {
            vec![C::ONE, C::ZERO]
        } else {
            vec![C::ZERO, C::ONE]
        };
        let norm = dot(&v, &v).re.sqrt();
        for z in &mut v {
            *z = *z / norm;
        }
        vectors.push(v);
    }
    (vals, vectors)
}
fn spectra(r: &ExperimentRequest) -> ExperimentResult {
    let a = amp(r);
    let m = rate(r);
    let mixing = [C::ONE, C::from(a), C::new(0., 0.3), C::new(0.4 * a, a)];
    let cov = |t: f64| {
        let diagonal = vec![
            C::from((-m * t).exp()),
            C::ZERO,
            C::ZERO,
            C::from((-(m + 0.6) * t).exp()),
        ];
        mul(&mul(&mixing, &diagonal, 2), &adjoint(&mixing, 2), 2)
    };
    let c0 = cov(0.);
    let (e, v) = hermitian2_modes(&c0);
    let threshold = e.iter().copied().fold(0., f64::max) * 1e-10;
    let keep: Vec<usize> = e
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| (x > threshold).then_some(i))
        .collect();
    let rank = keep.len();
    let mut curves = vec![vec![]; rank];
    let mut error: f64 = 0.;
    let mut imaginary = vec![];
    for k in 0..=steps(r) {
        let t = k as f64 * 0.1;
        let ct = cov(t);
        let mut whitened = vec![C::ZERO; rank * rank];
        for (i, &vi) in keep.iter().enumerate() {
            for (j, &vj) in keep.iter().enumerate() {
                let b: Vec<C> = (0..2)
                    .map(|row| (0..2).fold(C::ZERO, |s, col| s + ct[row * 2 + col] * v[vj][col]))
                    .collect();
                whitened[i * rank + j] = dot(&v[vi], &b) / (e[vi] * e[vj]).sqrt();
            }
        }
        let eigen = hermitian_eigenvalues(&whitened, rank);
        for j in 0..rank {
            let value = eigen[rank - 1 - j];
            curves[j].push([t, value]);
            let expected = (-if j == 0 { m } else { m + 0.6 } * t).exp();
            error = error.max((value - expected).abs());
        }
        imaginary.push([t, ct[1].im]);
    }
    let mut o = ExperimentResult::new(
        36,
        "Complex correlators and rank-aware spectra",
        "Exactly mixed two-channel exponential spectrum with positive-eigenspace whitening",
    );
    o.metric("Retained covariance rank", rank as f64, "modes")
        .metric("GEVP eigenvalue residual", error, "")
        .metric("First decay rate", m, "per algorithmic time");
    if rank == 2 {
        o.metric("Second decay rate", m + 0.6, "per algorithmic time");
    }
    let mut series: Vec<Series> = curves
        .into_iter()
        .enumerate()
        .map(|(i, c)| Series::line(format!("Whitened mode {}", i + 1), c))
        .collect();
    series.push(Series::line("Complex off-diagonal: imaginary", imaginary));
    o.plot(
        "Correlation spectrum",
        "algorithmic time",
        "correlation / eigenvalue",
        series,
    );
    o.details = json!({"covariance_eigenvalues":e,"rank_threshold":threshold,"correlation_at_zero":c0.iter().copied().map(complex_json).collect::<Vec<_>>(),"mixing":mixing.iter().copied().map(complex_json).collect::<Vec<_>>(),"method":"W* C(t) W on positive C(0) eigenspace; no Hermitian eigensolver on C(0)^(-1) C(t)"});
    o
}
