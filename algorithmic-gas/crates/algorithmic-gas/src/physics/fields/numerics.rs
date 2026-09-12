use crate::{
    GasError, Result,
    random::{RandomStream, Stream},
};
pub(super) fn rng(seed: u64, rep: usize) -> RandomStream {
    RandomStream::new(seed, 0, Stream::Initialize, rep as u64, 606)
}
pub(super) fn stats(xs: &[f64]) -> (f64, f64, f64) {
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = if xs.len() > 1 {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
    } else {
        0.
    };
    (mean, var, (var / xs.len() as f64).sqrt())
}
pub(super) fn integral(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    (0..n)
        .map(|i| f(a + (i as f64 + 0.5) * (b - a) / n as f64))
        .sum::<f64>()
        * (b - a)
        / n as f64
}
pub(super) fn derivative(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2. * h)
}
pub(super) fn rk4(y: &[f64], t: f64, h: f64, f: impl Fn(f64, &[f64]) -> Vec<f64>) -> Vec<f64> {
    let k1 = f(t, y);
    let z = |k: &[f64], s: f64| y.iter().zip(k).map(|(a, b)| a + s * b).collect::<Vec<_>>();
    let k2 = f(t + h / 2., &z(&k1, h / 2.));
    let k3 = f(t + h / 2., &z(&k2, h / 2.));
    let k4 = f(t + h, &z(&k3, h));
    y.iter()
        .enumerate()
        .map(|(i, a)| a + h * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]) / 6.)
        .collect()
}
#[allow(clippy::needless_range_loop)] // Indexed row elimination keeps the pivot algebra explicit.
pub(super) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        let j = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        if a[j][k].abs() < 1e-12 {
            return Err(GasError::Numerical("rank-deficient design".into()));
        }
        a.swap(k, j);
        b.swap(k, j);
        let z = a[k][k];
        for c in k..n {
            a[k][c] /= z;
        }
        b[k] /= z;
        for i in 0..n {
            if i != k {
                let z = a[i][k];
                for c in k..n {
                    a[i][c] -= z * a[k][c];
                }
                b[i] -= z * b[k];
            }
        }
    }
    Ok(b)
}
pub(super) fn stationary(k: &[Vec<f64>]) -> Result<Vec<f64>> {
    let n = k.len();
    let mut a = vec![vec![0.; n]; n];
    for i in 0..n - 1 {
        for j in 0..n {
            a[i][j] = k[j][i] - f64::from(i == j);
        }
    }
    a[n - 1].fill(1.);
    let mut b = vec![0.; n];
    b[n - 1] = 1.;
    solve(a, b)
}
pub(super) fn evolve(mu: &[f64], k: &[Vec<f64>]) -> Vec<f64> {
    (0..mu.len())
        .map(|j| (0..mu.len()).map(|i| mu[i] * k[i][j]).sum())
        .collect()
}
pub(super) fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|x| **x > 0.)
        .map(|x| x * x.ln())
        .sum::<f64>()
}
pub(super) fn tv(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.
}
pub(super) fn gaussian_periodic(x: f64, e: f64) -> f64 {
    (-4..=4)
        .map(|k| (-0.5 * ((x + k as f64) / e).powi(2)).exp())
        .sum()
}
pub(super) fn sample(k: &[f64], r: &mut RandomStream) -> usize {
    let u = r.uniform::<f64>();
    let mut c = 0.;
    for (i, v) in k.iter().enumerate() {
        c += v;
        if u < c {
            return i;
        }
    }
    k.len() - 1
}
