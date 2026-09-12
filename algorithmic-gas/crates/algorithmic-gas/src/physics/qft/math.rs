//! Small, portable complex linear algebra for independently checkable QFT experiments.
//! Matrices are row-major. No Hermitian-only routine is applied to a general product.
use std::ops::{Add, Div, Mul, Neg, Sub};
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}
impl C {
    pub const ZERO: Self = Self { re: 0., im: 0. };
    pub const ONE: Self = Self { re: 1., im: 0. };
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    pub fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.abs2().sqrt()
    }
    pub fn phase(t: f64) -> Self {
        Self::new(t.cos(), t.sin())
    }
}
impl From<f64> for C {
    fn from(x: f64) -> Self {
        Self::new(x, 0.)
    }
}
impl Add for C {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self::new(self.re + b.re, self.im + b.im)
    }
}
impl Sub for C {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self::new(self.re - b.re, self.im - b.im)
    }
}
impl Neg for C {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl Mul for C {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
}
impl Mul<f64> for C {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self::new(self.re * b, self.im * b)
    }
}
impl Div<f64> for C {
    type Output = Self;
    fn div(self, b: f64) -> Self {
        Self::new(self.re / b, self.im / b)
    }
}
impl Div for C {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        self * b.conj() / b.abs2()
    }
}
pub fn dot(a: &[C], b: &[C]) -> C {
    a.iter()
        .zip(b)
        .fold(C::ZERO, |s, (&x, &y)| s + x.conj() * y)
}
pub fn identity(n: usize) -> Vec<C> {
    let mut a = vec![C::ZERO; n * n];
    for i in 0..n {
        a[i * n + i] = C::ONE;
    }
    a
}
pub fn adjoint(a: &[C], n: usize) -> Vec<C> {
    (0..n * n).map(|k| a[(k % n) * n + k / n].conj()).collect()
}
pub fn mul(a: &[C], b: &[C], n: usize) -> Vec<C> {
    (0..n * n)
        .map(|k| (0..n).fold(C::ZERO, |s, j| s + a[(k / n) * n + j] * b[j * n + k % n]))
        .collect()
}
pub fn difference(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs2())
        .sum::<f64>()
        .sqrt()
}
pub fn trace(a: &[C], n: usize) -> C {
    (0..n).fold(C::ZERO, |s, i| s + a[i * n + i])
}
pub fn determinant(a: &[C], n: usize) -> C {
    let mut a = a.to_vec();
    let mut d = C::ONE;
    for j in 0..n {
        let p = (j..n)
            .max_by(|&u, &v| a[u * n + j].abs2().total_cmp(&a[v * n + j].abs2()))
            .unwrap();
        if a[p * n + j].abs() < 1e-15 {
            return C::ZERO;
        }
        if p != j {
            for k in 0..n {
                a.swap(j * n + k, p * n + k);
            }
            d = -d;
        }
        let v = a[j * n + j];
        d = d * v;
        for i in j + 1..n {
            let f = a[i * n + j] / v;
            for k in j + 1..n {
                a[i * n + k] = a[i * n + k] - f * a[j * n + k];
            }
        }
    }
    d
}
/// SU(2) rotation exp(i theta n.sigma), with unit axis supplied by the caller.
pub fn su2(theta: f64, axis: [f64; 3]) -> Vec<C> {
    let (c, s) = (theta.cos(), theta.sin());
    vec![
        C::new(c, s * axis[2]),
        C::new(s * axis[1], s * axis[0]),
        C::new(-s * axis[1], s * axis[0]),
        C::new(c, -s * axis[2]),
    ]
}
pub fn creator(modes: usize, mode: usize) -> Vec<C> {
    let n = 1 << modes;
    let mut a = vec![C::ZERO; n * n];
    for j in 0usize..n {
        if j & (1 << mode) == 0 {
            let i = j | (1 << mode);
            let parity = (j & ((1 << mode) - 1)).count_ones() % 2;
            a[i * n + j] = C::from(if parity == 0 { 1. } else { -1. });
        }
    }
    a
}
pub fn ridentity(n: usize) -> Vec<f64> {
    let mut a = vec![0.; n * n];
    for i in 0..n {
        a[i * n + i] = 1.;
    }
    a
}
pub fn rmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|k| (0..n).map(|j| a[k / n * n + j] * b[j * n + k % n]).sum())
        .collect()
}
pub fn rpower(a: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut b = ridentity(n);
    for _ in 0..p {
        b = rmul(&b, a, n);
    }
    b
}
pub fn rowmul(x: &[f64], p: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|j| (0..n).map(|i| x[i] * p[i * n + j]).sum())
        .collect()
}
/// Cyclic Jacobi for small real symmetric matrices. Returns eigenvalues and column eigenvectors.
pub fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = ridentity(n);
    for _ in 0..80 {
        let mut changed = false;
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-14 {
                    continue;
                }
                changed = true;
                let t = 0.5 * (2. * apq).atan2(a[q * n + q] - a[p * n + p]);
                let (c, s) = (t.cos(), t.sin());
                for k in 0..n {
                    let (x, y) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * x - s * y;
                    a[k * n + q] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * x - s * y;
                    a[q * n + k] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * x - s * y;
                    v[k * n + q] = s * x + c * y;
                }
            }
        }
        if !changed {
            break;
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}
/// General complex Hermitian spectrum via its real symmetric representation.
/// Every eigenvalue occurs twice; averaging pairs also retains repeated physical modes.
pub fn hermitian_eigenvalues(a: &[C], n: usize) -> Vec<f64> {
    let d = 2 * n;
    let mut r = vec![0.; d * d];
    for i in 0..n {
        for j in 0..n {
            let z = a[i * n + j];
            r[i * d + j] = z.re;
            r[i * d + j + n] = -z.im;
            r[(i + n) * d + j] = z.im;
            r[(i + n) * d + j + n] = z.re;
        }
    }
    let (mut w, _) = symmetric_eigen(&r, d);
    w.sort_by(f64::total_cmp);
    w.chunks_exact(2).map(|x| (x[0] + x[1]) * 0.5).collect()
}
pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}
pub fn sem(x: &[f64]) -> f64 {
    if x.len() < 2 {
        0.
    } else {
        let m = mean(x);
        (x.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((x.len() * (x.len() - 1)) as f64)).sqrt()
    }
}
/// Deterministic local experiment RNG. This never consumes the engine's streams.
pub struct Rng {
    state: u64,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e3779b97f4a7c15,
        }
    }
    pub fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        (-2. * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}
pub fn gaussian(x: f64) -> f64 {
    (-x * x / 2.).exp() / (std::f64::consts::TAU).sqrt()
}
pub fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = n + n % 2;
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        s += if i % 2 == 0 { 2. } else { 4. } * f(a + i as f64 * h);
    }
    s * h / 3.
}
pub fn normal_cdf(x: f64) -> f64 {
    integrate(gaussian, -10., x.clamp(-10., 10.), 1000)
}

/// Complex orthonormal eigensystem of a small Hermitian matrix. The real
/// representation has two copies of each eigenspace; complex Gram–Schmidt
/// removes that duplication, including repeated physical eigenvalues.
pub fn hermitian_modes(a: &[C], n: usize) -> (Vec<f64>, Vec<Vec<C>>) {
    let d = 2 * n;
    let mut r = vec![0.; d * d];
    for i in 0..n {
        for j in 0..n {
            let z = a[i * n + j];
            r[i * d + j] = z.re;
            r[i * d + j + n] = -z.im;
            r[(i + n) * d + j] = z.im;
            r[(i + n) * d + j + n] = z.re;
        }
    }
    let (w, v) = symmetric_eigen(&r, d);
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&i, &j| w[i].total_cmp(&w[j]));
    let mut values = vec![];
    let mut columns: Vec<Vec<C>> = vec![];
    for index in order {
        let mut z: Vec<C> = (0..n)
            .map(|i| C::new(v[i * d + index], v[(i + n) * d + index]))
            .collect();
        for q in &columns {
            let coefficient = dot(q, &z);
            for i in 0..n {
                z[i] = z[i] - q[i] * coefficient;
            }
        }
        let norm = dot(&z, &z).re.max(0.).sqrt();
        if norm < 1e-7 {
            continue;
        }
        for x in &mut z {
            *x = *x / norm;
        }
        let az: Vec<C> = (0..n)
            .map(|i| (0..n).fold(C::ZERO, |s, j| s + a[i * n + j] * z[j]))
            .collect();
        values.push(dot(&z, &az).re);
        columns.push(z);
        if columns.len() == n {
            break;
        }
    }
    (values, columns)
}
impl C {
    pub fn sqrt(self) -> Self {
        let r = self.abs();
        Self::new(
            ((r + self.re) / 2.).max(0.).sqrt(),
            ((r - self.re) / 2.).max(0.).sqrt().copysign(self.im),
        )
    }
}
/// All roots of the characteristic polynomial, preserving complex spectra.
/// The bounded native spectral interface uses matrices of size at most three.
pub fn general_eigenvalues(a: &[C], n: usize) -> Vec<C> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![a[0]];
    }
    if difference(a, &adjoint(a, n)) < 1e-12 * (1. + a.iter().map(|z| z.abs()).sum::<f64>()) {
        return hermitian_eigenvalues(a, n)
            .into_iter()
            .map(C::from)
            .collect();
    }
    let tr = trace(a, n);
    if n == 2 {
        let discriminant = (tr * tr - determinant(a, n) * 4.).sqrt();
        return vec![(tr + discriminant) / 2., (tr - discriminant) / 2.];
    }
    assert_eq!(n, 3, "general spectral solver is limited to three channels");
    let a2 = mul(a, a, n);
    let c2 = (tr * tr - trace(&a2, n)) / 2.;
    let det = determinant(a, n);
    let coefficients = [C::ONE, -tr, c2, -det];
    let radius = 1. + coefficients[1..].iter().map(|z| z.abs()).fold(0., f64::max);
    let mut roots: Vec<C> = (0..n)
        .map(|k| C::phase(0.37 + std::f64::consts::TAU * k as f64 / n as f64) * radius)
        .collect();
    for _ in 0..160 {
        let old = roots.clone();
        let mut change: f64 = 0.;
        for i in 0..n {
            let numerator = coefficients.iter().fold(C::ZERO, |s, &c| s * old[i] + c);
            let mut denominator = C::ONE;
            for j in 0..n {
                if i != j {
                    denominator = denominator * (old[i] - old[j]);
                }
            }
            if denominator.abs() < 1e-24 {
                continue;
            }
            let delta = numerator / denominator;
            roots[i] = old[i] - delta;
            change = change.max(delta.abs());
        }
        if change < 1e-13 {
            break;
        }
    }
    roots
}
