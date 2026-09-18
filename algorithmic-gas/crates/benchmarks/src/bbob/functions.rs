//! The 24 `f_*_bbob_problem_allocate` constructions and their raw functions.
#![allow(clippy::needless_range_loop)]
use super::{
    BbobProblem, rng,
    transforms::{
        affine, asymmetric, boundary_penalty, brs, conditioned_product, conditioning, oscillate,
        oscillate_value, shifted,
    },
};
use std::f64::consts::PI;

#[derive(Debug)]
pub struct Gallagher {
    peaks: usize,
    /// `x_local[j][peak]`, row-major `d × peaks`.
    local: Vec<f64>,
    /// `arr_scales[peak][j]`, row-major `peaks × d`.
    scales: Vec<f64>,
    values: Vec<f64>,
}

fn round(value: f64) -> f64 {
    (value + 0.5).floor()
}

fn column_sums(m: &[f64], d: usize) -> Vec<f64> {
    (0..d)
        .map(|column| (0..d).fold(0., |sum, row| sum + m[row * d + column]))
        .collect()
}

/// Indices of `values` in ascending order (`qsort` with `f_gallagher_compare_doubles`).
fn ascending(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    order
}

pub fn build(function: u8, d: usize, instance: u32) -> BbobProblem {
    let base = match function {
        4 => 3,
        18 => 17,
        other => i64::from(other),
    };
    let seed = base + 10000 * i64::from(instance);
    let rot1 = || rng::rotation(seed + 1_000_000, d);
    let rot2 = || rng::rotation(seed, d);
    let mut problem = BbobProblem {
        function,
        dimensions: d,
        instance,
        fopt: rng::fopt(function, instance),
        xopt: rng::xopt(seed, d),
        best: vec![],
        m1: vec![],
        m2: vec![],
        gallagher: None,
    };
    match function {
        1..=3 => {}
        4 => {
            for value in problem.xopt.iter_mut().step_by(2) {
                *value = value.abs();
            }
        }
        5 => {
            problem.best = problem
                .xopt
                .iter()
                .map(|&x| if x < 0. { -5. } else { 5. })
                .collect();
        }
        6 | 13 | 15 => {
            let (r1, r2) = (rot1(), rot2());
            problem.m2 = conditioned_product(&r1, &r2, 10f64.sqrt(), d);
            problem.m1 = r1;
        }
        7 | 24 => {
            problem.m1 = rot1();
            problem.m2 = rot2();
        }
        8 => {
            for value in &mut problem.xopt {
                *value *= 0.75;
            }
        }
        9 | 19 => {
            let factor = 1f64.max((d as f64).sqrt() / 8.);
            let rotation = rot2();
            let sums = column_sums(&rotation, d);
            problem.m1 = rotation.iter().map(|value| factor * value).collect();
            // f19 sums the already scaled rotation; both divide by `2 * factor`.
            problem.best = if function == 9 {
                sums.iter().map(|sum| sum / (2. * factor)).collect()
            } else {
                column_sums(&problem.m1, d)
                    .iter()
                    .map(|sum| sum / (2. * factor))
                    .collect()
            };
        }
        10 | 11 | 14 => problem.m1 = rot1(),
        12 => {
            problem.xopt = rng::xopt(seed + 1_000_000, d);
            problem.m1 = rot1();
        }
        16 => {
            let (r1, r2) = (rot1(), rot2());
            problem.m2 = conditioned_product(&r1, &r2, 1. / 100f64.sqrt(), d);
            problem.m1 = r1;
        }
        17 | 18 => {
            let conditioning: f64 = if function == 17 { 10. } else { 1000. };
            let r2 = rot2();
            problem.m2 = vec![0.; d * d];
            for i in 0..d {
                let exponent = 1.0 * i as f64 / (d as f64 - 1.);
                for j in 0..d {
                    problem.m2[i * d + j] = r2[i * d + j] * conditioning.sqrt().powf(exponent);
                }
            }
            problem.m1 = rot1();
        }
        20 => {
            let uniform = rng::unif(d, seed);
            problem.xopt = uniform
                .iter()
                .map(|&u| (if u < 0.5 { -1. } else { 1. }) * 0.5 * 4.2096874637)
                .collect();
            // best_parameter follows the C transformation chain from 420.96874633.
            problem.best = problem
                .xopt
                .iter()
                .zip(&uniform)
                .map(|(xopt, &u)| {
                    let mut best = 420.96874633 / 100.;
                    best += -2. * xopt.abs();
                    best += 2. * xopt.abs();
                    best /= 2.;
                    if u < 0.5 { -best } else { best }
                })
                .collect();
            // Sign vector of `transform_vars_x_hat`.
            problem.m1 = uniform;
        }
        21 | 22 => build_gallagher(&mut problem, seed),
        23 => {
            let (r1, r2) = (rot1(), rot2());
            problem.m2 = conditioned_product(&r1, &r2, 100f64.sqrt(), d);
        }
        _ => unreachable!("validated function index"),
    }
    if function == 24 {
        problem.xopt = rng::gauss(d, seed)
            .iter()
            .map(|&g| if g < 0. { -0.5 * 2.5 } else { 0.5 * 2.5 })
            .collect();
    }
    if problem.best.is_empty() {
        problem.best = problem.xopt.clone();
    }
    problem
}

fn build_gallagher(problem: &mut BbobProblem, seed: i64) {
    let d = problem.dimensions;
    let peaks = if problem.function == 21 { 101 } else { 21 };
    let (b, c, first_condition) = if peaks == 101 {
        (10., 5., 1000f64.sqrt())
    } else {
        (9.8, 4.9, 1000.)
    };
    let rotation = rng::rotation(seed, d);
    let order = ascending(&rng::unif(peaks - 1, seed));
    let mut condition = vec![first_condition; peaks];
    let mut values = vec![10.; peaks];
    for i in 1..peaks {
        condition[i] = 1000f64.powf(order[i - 1] as f64 / (peaks - 2) as f64);
        values[i] = (i - 1) as f64 / (peaks - 2) as f64 * (9.1 - 1.1) + 1.1;
    }
    let mut scales = vec![0.; peaks * d];
    for i in 0..peaks {
        let order = ascending(&rng::unif(d, seed + 1000 * i as i64));
        for j in 0..d {
            scales[i * d + j] = condition[i].powf(order[j] as f64 / (d - 1) as f64 - 0.5);
        }
    }
    let uniform = rng::unif(d * peaks, seed);
    let scale_factor = 0.8;
    let mut local = vec![0.; d * peaks];
    for i in 0..d {
        for j in 0..peaks {
            for k in 0..d {
                local[i * peaks + j] += rotation[i * d + k] * (b * uniform[j * d + k] - c);
            }
            if j == 0 {
                local[i * peaks + j] *= scale_factor;
            }
        }
    }
    problem.xopt = (0..d)
        .map(|i| scale_factor * (b * uniform[i] - c))
        .collect();
    problem.m1 = rotation;
    problem.gallagher = Some(Gallagher {
        peaks,
        local,
        scales,
        values,
    });
}

fn sphere(z: &[f64]) -> f64 {
    z.iter().fold(0., |sum, z| sum + z * z)
}

fn ellipsoid(z: &[f64]) -> f64 {
    let n = z.len() as f64;
    let mut result = z[0] * z[0];
    for i in 1..z.len() {
        let exponent = 1.0 * i as f64 / (n - 1.);
        result += 1.0e6f64.powf(exponent) * z[i] * z[i];
    }
    result
}

fn rastrigin(z: &[f64]) -> f64 {
    let (mut cosines, mut squares) = (0., 0.);
    for &z in z {
        cosines += (2. * PI * z).cos();
        squares += z * z;
    }
    if squares.is_infinite() {
        return squares;
    }
    10. * (z.len() as f64 - cosines) + squares
}

fn rosenbrock(z: &[f64]) -> f64 {
    let (mut s1, mut s2) = (0., 0.);
    for pair in z.windows(2) {
        let a = pair[0] * pair[0] - pair[1];
        s1 += a * a;
        let b = pair[0] - 1.;
        s2 += b * b;
    }
    100. * s1 + s2
}

fn schaffers(z: &[f64]) -> f64 {
    let mut result = 0.;
    for pair in z.windows(2) {
        let t = pair[0] * pair[0] + pair[1] * pair[1];
        if t.is_infinite() && (50. * t.powf(0.1)).sin().is_nan() {
            return t;
        }
        result += t.powf(0.25) * (1. + (50. * t.powf(0.1)).sin().powf(2.));
    }
    (result / (z.len() as f64 - 1.)).powf(2.)
}

fn weierstrass(z: &[f64]) -> f64 {
    const SUMMANDS: usize = 12;
    let mut f0 = 0.;
    let mut ak = [0.; SUMMANDS];
    let mut bk = [0.; SUMMANDS];
    for i in 0..SUMMANDS {
        ak[i] = 0.5f64.powf(i as f64);
        bk[i] = 3f64.powf(i as f64);
        f0 += ak[i] * (2. * PI * bk[i] * 0.5).cos();
    }
    let mut result = 0.;
    for &z in z {
        for j in 0..SUMMANDS {
            result += (2. * PI * (z + 0.5) * bk[j]).cos() * ak[j];
        }
    }
    10. * (result / z.len() as f64 - f0).powf(3.)
}

fn katsuura(z: &[f64]) -> f64 {
    let n = z.len() as f64;
    let mut result = 1.;
    for (i, &z) in z.iter().enumerate() {
        let mut t = 0.;
        for j in 1..33 {
            let power = 2f64.powf(j as f64);
            t += (power * z - round(power * z)).abs() / power;
        }
        t = 1. + (i as f64 + 1.) * t;
        result *= t.powf(10. / n.powf(1.2));
    }
    10. / n / n * (-1. + result)
}

fn step_ellipsoid(p: &BbobProblem, x: &[f64]) -> f64 {
    let d = x.len();
    let mut penalty = 0.;
    for &x in x {
        let t = x.abs() - 5.;
        if t > 0. {
            penalty += t * t;
        }
    }
    let mut z = vec![0.; d];
    for i in 0..d {
        let c1 = (100f64 / 10.).powf(i as f64 / (d - 1) as f64).sqrt();
        for j in 0..d {
            z[i] += c1 * p.m2[i * d + j] * (x[j] - p.xopt[j]);
        }
    }
    let first = z[0];
    for z in &mut z {
        *z = if z.abs() > 0.5 {
            round(*z)
        } else {
            round(10. * *z) / 10.
        };
    }
    let mut result = 0.;
    for i in 0..d {
        let mut rotated = 0.;
        for j in 0..d {
            rotated += p.m1[i * d + j] * z[j];
        }
        let exponent = i as f64 / (d as f64 - 1.);
        result += 100f64.powf(exponent) * rotated * rotated;
    }
    let scaled = first.abs() * 1.0e-4;
    0.1 * if scaled >= result { scaled } else { result } + penalty + p.fopt
}

fn gallagher(p: &BbobProblem, x: &[f64]) -> f64 {
    let d = x.len();
    let data = p.gallagher.as_ref().expect("Gallagher instance data");
    let factor = -0.5 / d as f64;
    let mut penalty = 0.;
    for &x in x {
        let t = x.abs() - 5.;
        if t > 0. {
            penalty += t * t;
        }
    }
    let rotated = affine(&p.m1, x, 0.);
    let mut f: f64 = 0.;
    for i in 0..data.peaks {
        let mut distance = 0.;
        for j in 0..d {
            let t = rotated[j] - data.local[j * data.peaks + i];
            distance += data.scales[i * d + j] * t * t;
        }
        let peak = data.values[i] * (factor * distance).exp();
        if peak > f {
            f = peak;
        }
    }
    let oscillated = oscillate_value(10. - f);
    oscillated * oscillated + penalty
}

fn lunacek(p: &BbobProblem, x: &[f64]) -> f64 {
    let d = x.len();
    let n = d as f64;
    const MU0: f64 = 2.5;
    let s = 1. - 0.5 / ((n + 20.).sqrt() - 4.1);
    let mu1 = -((MU0 * MU0 - 1.) / s).sqrt();
    let mut penalty = 0.;
    for &x in x {
        let t = x.abs() - 5.;
        if t > 0. {
            penalty += t * t;
        }
    }
    let x_hat: Vec<f64> = x
        .iter()
        .zip(&p.xopt)
        .map(|(&x, &xopt)| if xopt < 0. { -(2. * x) } else { 2. * x })
        .collect();
    let mut stretched = vec![0.; d];
    for i in 0..d {
        let c1 = 100f64.sqrt().powf(i as f64 / (d - 1) as f64);
        for j in 0..d {
            stretched[i] += c1 * p.m2[i * d + j] * (x_hat[j] - MU0);
        }
    }
    let z = affine(&p.m1, &stretched, 0.);
    let (mut sum1, mut sum2, mut sum3) = (0., 0., 0.);
    for i in 0..d {
        sum1 += (x_hat[i] - MU0) * (x_hat[i] - MU0);
        sum2 += (x_hat[i] - mu1) * (x_hat[i] - mu1);
        sum3 += (2. * PI * z[i]).cos();
    }
    let second = 1. * n + s * sum2;
    (if sum1 <= second { sum1 } else { second }) + 10. * (n - sum3) + 1e4 * penalty
}

pub fn evaluate(p: &BbobProblem, x: &[f64]) -> f64 {
    let d = x.len();
    let n = d as f64;
    let centered = || shifted(x, &p.xopt);
    match p.function {
        1 => sphere(&centered()) + p.fopt,
        2 => {
            let mut z = centered();
            oscillate(&mut z);
            ellipsoid(&z) + p.fopt
        }
        3 => {
            let mut z = centered();
            oscillate(&mut z);
            asymmetric(&mut z, 0.2);
            conditioning(&mut z, 10.);
            rastrigin(&z) + p.fopt
        }
        4 => {
            let mut z = centered();
            oscillate(&mut z);
            brs(&mut z);
            rastrigin(&z) + 0. + p.fopt + 100. * boundary_penalty(x)
        }
        5 => {
            let mut result = 0.;
            for i in 0..d {
                let magnitude = 100f64.sqrt().powf(i as f64 / (n - 1.));
                let slope = if p.best[i] > 0. {
                    magnitude
                } else {
                    -magnitude
                };
                result += if x[i] * p.best[i] < 25. {
                    5. * slope.abs() - slope * x[i]
                } else {
                    5. * slope.abs() - slope * p.best[i]
                };
            }
            result + p.fopt
        }
        6 => {
            let z = affine(&p.m2, &centered(), 0.);
            let mut result = 0.;
            for i in 0..d {
                result += if p.xopt[i] * z[i] > 0. {
                    100. * 100. * z[i] * z[i]
                } else {
                    z[i] * z[i]
                };
            }
            if result != 0. {
                result = oscillate_value(result);
            }
            result.powf(0.9) + p.fopt
        }
        7 => step_ellipsoid(p, x),
        8 => {
            let factor = 1f64.max(n.sqrt() / 8.);
            let z: Vec<f64> = centered().iter().map(|z| factor * z - -1.).collect();
            rosenbrock(&z) + p.fopt
        }
        9 => rosenbrock(&affine(&p.m1, x, 0.5)) + p.fopt,
        10 | 11 => {
            let mut z = affine(&p.m1, &centered(), 0.);
            oscillate(&mut z);
            if p.function == 10 {
                ellipsoid(&z) + p.fopt
            } else {
                let mut result = 1.0e6 * z[0] * z[0];
                for &z in &z[1..] {
                    result += z * z;
                }
                result + p.fopt
            }
        }
        12 => {
            let mut z = affine(&p.m1, &centered(), 0.);
            asymmetric(&mut z, 0.5);
            let z = affine(&p.m1, &z, 0.);
            let mut result = z[0] * z[0];
            for &z in &z[1..] {
                result += 1.0e6 * z * z;
            }
            result + p.fopt
        }
        13 => {
            let z = affine(&p.m2, &centered(), 0.);
            let mut result = 0.;
            for &z in &z[1..] {
                result += z * z;
            }
            result = 100. * (result / 1.).sqrt();
            result += z[0] * z[0] / 1.;
            result + p.fopt
        }
        14 => {
            let z = affine(&p.m1, &centered(), 0.);
            let mut sum = 0.;
            for i in 0..d {
                let exponent = 2. + (4. * i as f64) / (n - 1.);
                sum += z[i].abs().powf(exponent);
            }
            sum.sqrt() + p.fopt
        }
        15 => {
            let mut z = affine(&p.m1, &centered(), 0.);
            oscillate(&mut z);
            asymmetric(&mut z, 0.2);
            rastrigin(&affine(&p.m2, &z, 0.)) + p.fopt
        }
        16 => {
            let mut z = affine(&p.m1, &centered(), 0.);
            oscillate(&mut z);
            weierstrass(&affine(&p.m2, &z, 0.)) + p.fopt + 10. / n * boundary_penalty(x)
        }
        17 | 18 => {
            let mut z = affine(&p.m1, &centered(), 0.);
            asymmetric(&mut z, 0.5);
            schaffers(&affine(&p.m2, &z, 0.)) + p.fopt + 10. * boundary_penalty(x)
        }
        19 => {
            let z: Vec<f64> = affine(&p.m1, x, 0.).iter().map(|z| z - -0.5).collect();
            let mut result = 0.;
            for pair in z.windows(2) {
                let c1 = pair[0] * pair[0] - pair[1];
                let c2 = 1. - pair[0];
                let t = 100. * c1 * c1 + c2 * c2;
                result += t / 4000. - t.cos();
            }
            10. + 10. * result / (n - 1.) + p.fopt
        }
        20 => {
            let mut z: Vec<f64> = x
                .iter()
                .zip(&p.m1)
                .map(|(&x, &u)| 2. * if u < 0.5 { -x } else { x })
                .collect();
            let scaled = z.clone();
            for i in 1..d {
                z[i] = scaled[i] + 0.25 * (scaled[i - 1] - 2. * p.xopt[i - 1].abs());
            }
            for i in 0..d {
                z[i] -= 2. * p.xopt[i].abs();
            }
            conditioning(&mut z, 10.);
            for i in 0..d {
                z[i] = 100. * (z[i] - -2. * p.xopt[i].abs());
            }
            let (mut penalty, mut sum) = (0., 0.);
            for &z in &z {
                let t = z.abs() - 500.;
                if t > 0. {
                    penalty += t * t;
                }
            }
            for &z in &z {
                sum += z * z.abs().sqrt().sin();
            }
            0.01 * (penalty + 418.9828872724339 - sum / n) + p.fopt
        }
        21 | 22 => gallagher(p, x) + p.fopt,
        23 => katsuura(&affine(&p.m2, &centered(), 0.)) + p.fopt + 1. * boundary_penalty(x),
        24 => lunacek(p, x) + p.fopt,
        _ => unreachable!("validated function index"),
    }
}
