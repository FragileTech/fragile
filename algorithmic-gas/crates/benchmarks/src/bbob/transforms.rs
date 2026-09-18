//! COCO variable and objective transformations (`transform_vars_*`, `transform_obj_*`).

fn oscillate_scalar(value: f64, positive: (f64, f64), negative: (f64, f64)) -> f64 {
    const ALPHA: f64 = 0.1;
    if value > 0. {
        let t = value.ln() / ALPHA;
        (t + 0.49 * ((positive.0 * t).sin() + (positive.1 * t).sin()))
            .exp()
            .powf(ALPHA)
    } else if value < 0. {
        let t = (-value).ln() / ALPHA;
        -(t + 0.49 * ((negative.0 * t).sin() + (negative.1 * t).sin()))
            .exp()
            .powf(ALPHA)
    } else {
        0.
    }
}

/// `tosz_uv`, also used by `transform_obj_oscillate` and the Gallagher functions.
pub fn oscillate_value(value: f64) -> f64 {
    // C evaluates sin(tmp) literally; multiplying by 1.0 is exact.
    oscillate_scalar(value, (1., 0.79), (0.55, 0.31))
}

pub fn oscillate(x: &mut [f64]) {
    for value in x {
        *value = oscillate_value(*value);
    }
}

/// `tasy_uv` with exponent `1 + beta * i/(n-1) * sqrt(x)`.
pub fn asymmetric(x: &mut [f64], beta: f64) {
    let n = x.len() as f64;
    for (i, value) in x.iter_mut().enumerate() {
        if *value > 0. {
            *value = value.powf(1. + ((beta * i as f64) / (n - 1.)) * value.sqrt());
        }
    }
}

/// `transform_vars_conditioning`.
pub fn conditioning(x: &mut [f64], alpha: f64) {
    let n = x.len() as f64;
    for (i, value) in x.iter_mut().enumerate() {
        *value *= alpha.powf(0.5 * i as f64 / (n - 1.));
    }
}

/// `transform_vars_brs`.
pub fn brs(x: &mut [f64]) {
    let n = x.len() as f64;
    for (i, value) in x.iter_mut().enumerate() {
        let mut factor = 10f64.sqrt().powf(i as f64 / (n - 1.));
        if *value > 0. && i % 2 == 0 {
            factor *= 10.;
        }
        *value *= factor;
    }
}

/// `transform_vars_affine`: `out[i] = offset + Σ_j x[j] * m[i][j]`, summed in C order.
pub fn affine(m: &[f64], x: &[f64], offset: f64) -> Vec<f64> {
    let d = x.len();
    (0..d)
        .map(|i| {
            let mut value = offset;
            for j in 0..d {
                value += x[j] * m[i * d + j];
            }
            value
        })
        .collect()
}

pub fn shifted(x: &[f64], offset: &[f64]) -> Vec<f64> {
    x.iter().zip(offset).map(|(x, o)| x - o).collect()
}

/// `transform_obj_penalize` over `[-5, 5]^d` (before scaling).
pub fn boundary_penalty(x: &[f64]) -> f64 {
    x.iter().fold(0., |sum, &x| {
        let (upper, lower) = (x - 5., -5. - x);
        if upper > 0. {
            sum + upper * upper
        } else if lower > 0. {
            sum + lower * lower
        } else {
            sum
        }
    })
}

/// `rot1 · diag(base^(k/(d-1))) · rot2`, accumulated as `(rot1 * scale) * rot2` over `k`.
pub fn conditioned_product(rot1: &[f64], rot2: &[f64], base: f64, d: usize) -> Vec<f64> {
    let mut m = vec![0.; d * d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                let exponent = 1.0 * k as f64 / (d as f64 - 1.);
                m[i * d + j] += rot1[i * d + k] * base.powf(exponent) * rot2[k * d + j];
            }
        }
    }
    m
}
