//! COCO `bbob2009_*` instance generators, reproduced operation for operation.

/// `bbob2009_unif`: Park–Miller stream with a 32-entry shuffle table.
pub fn unif(n: usize, seed: i64) -> Vec<f64> {
    let mut seed = seed.abs().max(1);
    let mut table = [0i64; 32];
    let advance = |seed: &mut i64| {
        let quotient = (*seed as f64 / 127773.0).floor() as i32 as i64;
        *seed = 16807 * (*seed - quotient * 127773) - 2836 * quotient;
        if *seed < 0 {
            *seed += 2147483647;
        }
    };
    for i in (0..40).rev() {
        advance(&mut seed);
        if i < 32 {
            table[i] = seed;
        }
    }
    let mut current = table[0];
    (0..n)
        .map(|_| {
            advance(&mut seed);
            let slot = (current as f64 / 67108865.0).floor() as usize;
            current = table[slot];
            table[slot] = seed;
            let value = current as f64 / 2.147483647e9;
            if value == 0. { 1e-99 } else { value }
        })
        .collect()
}

/// `bbob2009_gauss`: Box–Muller over one uniform block of length `2n`.
pub fn gauss(n: usize, seed: i64) -> Vec<f64> {
    let uniform = unif(2 * n, seed);
    (0..n)
        .map(|i| {
            let value =
                (-2. * uniform[i].ln()).sqrt() * (2. * std::f64::consts::PI * uniform[n + i]).cos();
            if value == 0. { 1e-99 } else { value }
        })
        .collect()
}

/// `bbob2009_compute_rotation`: row-major `d × d` orthogonal matrix.
#[allow(clippy::needless_range_loop)]
pub fn rotation(seed: i64, d: usize) -> Vec<f64> {
    let normal = gauss(d * d, seed);
    let mut b = vec![0.; d * d];
    for i in 0..d {
        for j in 0..d {
            b[i * d + j] = normal[j * d + i];
        }
    }
    for i in 0..d {
        for j in 0..i {
            let mut product = 0.;
            for k in 0..d {
                product += b[k * d + i] * b[k * d + j];
            }
            for k in 0..d {
                b[k * d + i] -= product * b[k * d + j];
            }
        }
        let mut product = 0.;
        for k in 0..d {
            product += b[k * d + i] * b[k * d + i];
        }
        for k in 0..d {
            b[k * d + i] /= product.sqrt();
        }
    }
    b
}

/// `bbob2009_compute_xopt`.
pub fn xopt(seed: i64, d: usize) -> Vec<f64> {
    unif(d, seed)
        .into_iter()
        .map(|u| {
            let value = 8. * (1e4 * u).floor() / 1e4 - 4.;
            if value == 0. { -1e-5 } else { value }
        })
        .collect()
}

/// `bbob2009_compute_fopt`.
pub fn fopt(function: u8, instance: u32) -> f64 {
    let base = match function {
        4 => 3,
        18 => 17,
        other => i64::from(other),
    };
    let seed = base + 10000 * i64::from(instance);
    let ratio = gauss(1, seed)[0] / gauss(1, seed + 1)[0];
    ((100. * 100. * ratio + 0.5).floor() / 100.).clamp(-1000., 1000.)
}
