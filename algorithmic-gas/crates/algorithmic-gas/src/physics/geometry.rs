//! Differential geometry with the convention R^a_bcd = d_c Gamma^a_bd -
//! d_d Gamma^a_bc + Gamma^a_ce Gamma^e_bd - Gamma^a_de Gamma^e_bc.
use super::jet::Jet;
use crate::{GasError, Real, Result, error::require, partv_geometry::MetricPolicy};
use serde::{Deserialize, Serialize};

/// Packed derivatives, ordered lexicographically over nondecreasing axis tuples.
/// Third order is sufficient on the smooth Hessian branch; fourth is optional.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct FitnessJet<T: Real> {
    pub dimension: usize,
    pub value: T,
    pub gradient: Vec<T>,
    pub hessian: Vec<T>,
    pub third: Vec<T>,
    pub fourth: Option<Vec<T>>,
}
fn tuples(d: usize, order: usize) -> Vec<Vec<usize>> {
    fn visit(d: usize, left: usize, start: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if left == 0 {
            out.push(a.clone());
            return;
        }
        for i in start..d {
            a.push(i);
            visit(d, left - 1, i, a, out);
            a.pop();
        }
    }
    let mut out = Vec::new();
    visit(d, order, 0, &mut Vec::new(), &mut out);
    out
}
pub(crate) fn expand<T: Real>(d: usize, order: usize, packed: &[T]) -> Result<Vec<T>> {
    let keys = tuples(d, order);
    require(keys.len() == packed.len(), "packed derivative shape")?;
    let map: std::collections::BTreeMap<_, _> =
        keys.into_iter().zip(packed.iter().copied()).collect();
    let mut out = vec![T::ZERO; d.pow(order as u32)];
    for (index, value) in out.iter_mut().enumerate() {
        let mut k = index;
        let mut axes = vec![0; order];
        for a in axes.iter_mut().rev() {
            *a = k % d;
            k /= d;
        }
        axes.sort_unstable();
        *value = map[&axes];
    }
    Ok(out)
}
impl<T: Real> FitnessJet<T> {
    pub fn from_jet(j: &Jet<T>) -> Result<Self> {
        require(
            j.space.order >= 3,
            "curvature requires at least third fitness derivatives",
        )?;
        j.validate()?;
        let collect = |order| {
            tuples(j.space.dimension, order)
                .iter()
                .map(|a| j.derivative(a))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            dimension: j.space.dimension,
            value: j.value(),
            gradient: collect(1)?,
            hessian: collect(2)?,
            third: collect(3)?,
            fourth: if j.space.order >= 4 {
                Some(collect(4)?)
            } else {
                None
            },
        })
    }
    pub fn dense_hessian(&self) -> Result<Vec<T>> {
        self.validate()?;
        expand(self.dimension, 2, &self.hessian)
    }
    pub fn validate(&self) -> Result<()> {
        let d = self.dimension;
        require((1..=16).contains(&d), "geometry dimension must be 1..16")?;
        require(
            self.gradient.len() == d
                && self.hessian.len() == d * (d + 1) / 2
                && self.third.len() == d * (d + 1) * (d + 2) / 6,
            "fitness jet packed shapes",
        )?;
        if let Some(q) = &self.fourth {
            require(
                q.len() == d * (d + 1) * (d + 2) * (d + 3) / 24,
                "fourth derivative shape",
            )?;
        }
        require(
            self.value.is_finite()
                && self
                    .gradient
                    .iter()
                    .chain(&self.hessian)
                    .chain(&self.third)
                    .chain(self.fourth.iter().flatten())
                    .all(|v| v.is_finite()),
            "finite fitness jet required",
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct MetricJet<T: Real> {
    pub dimension: usize,
    /// Row-major g_ij, dg[k,i,j], ddg[k,l,i,j].
    pub metric: Vec<T>,
    pub first: Vec<T>,
    pub second: Vec<T>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct MetricSpectrum<T: Real> {
    pub dimension: usize,
    pub eigenvalues: Vec<T>,
    /// Orthonormal eigenvectors are columns.
    pub eigenvectors: Vec<T>,
    pub metric: Vec<T>,
    pub inverse: Vec<T>,
    pub inverse_sqrt: Vec<T>,
    pub sqrt: Vec<T>,
    pub log_determinant: T,
    pub condition_number: T,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "T: Real")]
pub struct CurvatureBatch<T: Real> {
    pub dimension: usize,
    pub spectrum: MetricSpectrum<T>,
    pub ricci: Vec<T>,
    pub scalar: T,
    pub einstein: Vec<T>,
    /// Sectional curvatures in the orthonormal eigenframe, pairs i<j.
    pub sectional: Vec<T>,
    /// Optional covariant R_abcd, row major. Never needed for scalar/Ricci.
    pub riemann: Option<Vec<T>>,
    pub branch: String,
}
pub(crate) fn identity<T: Real>(d: usize) -> Vec<T> {
    let mut a = vec![T::ZERO; d * d];
    for i in 0..d {
        a[i * d + i] = T::ONE;
    }
    a
}
pub(crate) fn matmul<T: Real>(a: &[T], b: &[T], d: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; d * d];
    for i in 0..d {
        for k in 0..d {
            for j in 0..d {
                out[i * d + j] = out[i * d + j] + a[i * d + k] * b[k * d + j];
            }
        }
    }
    out
}
fn transpose<T: Real>(a: &[T], d: usize) -> Vec<T> {
    (0..d * d).map(|i| a[(i % d) * d + i / d]).collect()
}
fn symmetric<T: Real>(a: &[T], d: usize) -> Result<()> {
    require(
        (1..=16).contains(&d) && a.len() == d * d,
        "matrix dimension/shape",
    )?;
    let tol = if T::PRECISION == crate::Precision::F32 {
        2e-6
    } else {
        2e-13
    };
    let scale = a.iter().map(|v| v.abs().to_f64()).fold(0., f64::max);
    require(
        a.iter().all(|x| x.is_finite())
            && (0..d).all(|i| {
                (0..d).all(|j| (a[i * d + j] - a[j * d + i]).abs().to_f64() <= tol * scale)
            }),
        "finite symmetric matrix required",
    )
}
/// Cyclic Jacobi eigensolve; one decomposition supplies every metric power.
pub fn symmetric_eigen<T: Real>(a: &[T], d: usize) -> Result<(Vec<T>, Vec<T>)> {
    symmetric(a, d)?;
    let mut a = a.to_vec();
    let mut q = identity(d);
    let eps = if T::PRECISION == crate::Precision::F32 {
        2e-7
    } else {
        2e-15
    };
    for _ in 0..64 {
        let mut rotated = false;
        for p in 0..d {
            for r in p + 1..d {
                let b = a[p * d + r];
                if b == T::ZERO {
                    continue;
                }
                let scale = a[p * d + p].abs().sqrt() * a[r * d + r].abs().sqrt();
                if b.abs() <= T::from_f64(eps) * scale {
                    continue;
                }
                let delta = a[r * d + r] / T::from_f64(2.) - a[p * d + p] / T::from_f64(2.);
                // Scale before hypot to avoid overflow.
                let norm = delta.abs().max(b.abs());
                let x = delta / norm;
                let y = b / norm;
                let radius = (x * x + y * y).sqrt();
                let t = if delta == T::ZERO {
                    T::ONE
                } else {
                    y / (x.abs() + radius) * if delta > T::ZERO { T::ONE } else { -T::ONE }
                };
                let c = T::ONE / (T::ONE + t * t).sqrt();
                let s = t * c;
                a[p * d + p] = a[p * d + p] - t * b;
                a[r * d + r] = a[r * d + r] + t * b;
                a[p * d + r] = T::ZERO;
                a[r * d + p] = T::ZERO;
                for k in 0..d {
                    if k != p && k != r {
                        let u = a[k * d + p];
                        let v = a[k * d + r];
                        a[k * d + p] = c * u - s * v;
                        a[p * d + k] = a[k * d + p];
                        a[k * d + r] = s * u + c * v;
                        a[r * d + k] = a[k * d + r];
                    }
                    let u = q[k * d + p];
                    let v = q[k * d + r];
                    q[k * d + p] = c * u - s * v;
                    q[k * d + r] = s * u + c * v;
                }
                rotated = true;
            }
        }
        if !rotated {
            return Ok(((0..d).map(|i| a[i * d + i]).collect(), q));
        }
    }
    Err(GasError::Numerical(
        "symmetric eigensolver did not converge".into(),
    ))
}
fn spectral<T: Real>(q: &[T], l: &[T], d: usize) -> Vec<T> {
    let mut a = vec![T::ZERO; d * d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                a[i * d + j] = a[i * d + j] + q[i * d + k] * l[k] * q[j * d + k];
            }
        }
    }
    a
}
/// Q^T A Q for an analytically symmetric A. Assemble each symmetric entry once:
/// cancellation in independent transpose sums must not break tensor symmetry.
fn symmetric_congruence<T: Real>(a: &[T], q: &[T], d: usize) -> Vec<T> {
    let aq = matmul(a, q, d);
    let mut out = vec![T::ZERO; d * d];
    for i in 0..d {
        for j in i..d {
            let value = (0..d).fold(T::ZERO, |s, k| s + q[k * d + i] * aq[k * d + j]);
            out[i * d + j] = value;
            out[j * d + i] = value;
        }
    }
    out
}
pub(crate) fn spectrum<T: Real>(l: Vec<T>, q: Vec<T>, d: usize) -> Result<MetricSpectrum<T>> {
    require(
        l.iter().all(|v| v.is_finite() && *v > T::ZERO),
        "metric is not positive definite",
    )?;
    let powers = |p: f64| {
        spectral(
            &q,
            &l.iter()
                .map(|&v| v.powf(T::from_f64(p)))
                .collect::<Vec<_>>(),
            d,
        )
    };
    let mut out = MetricSpectrum {
        dimension: d,
        metric: powers(1.),
        inverse: powers(-1.),
        inverse_sqrt: powers(-0.5),
        sqrt: powers(0.5),
        log_determinant: l.iter().fold(T::ZERO, |s, &v| s + v.ln()),
        condition_number: l.iter().copied().fold(T::ZERO, |s, v| s.max(v))
            / l.iter()
                .copied()
                .fold(T::from_f64(f64::INFINITY), |s, v| s.min(v)),
        eigenvalues: l,
        eigenvectors: q,
    };
    require(
        out.metric
            .iter()
            .chain(&out.inverse)
            .chain(&out.inverse_sqrt)
            .chain(&out.sqrt)
            .all(|x| x.is_finite())
            && out.condition_number.is_finite(),
        "unrepresentable metric powers",
    )?;
    // Keep covariance symmetric to floating point precision.
    for i in 0..d {
        for j in 0..i {
            out.metric[i * d + j] = out.metric[j * d + i];
        }
    }
    Ok(out)
}
pub fn metric_spectrum<T: Real>(
    h: &[T],
    d: usize,
    epsilon: T,
    policy: MetricPolicy,
) -> Result<MetricSpectrum<T>> {
    require(
        epsilon.is_finite() && epsilon > T::ZERO,
        "positive metric epsilon required",
    )?;
    let (l, q) = symmetric_eigen(h, d)?;
    spectrum(
        l.into_iter()
            .map(|v| match policy {
                MetricPolicy::Strict => v + epsilon,
                MetricPolicy::Clipped => v.max(T::ZERO) + epsilon,
            })
            .collect(),
        q,
        d,
    )
}
fn transform3<T: Real>(c: &[T], e: &[T], d: usize) -> Vec<T> {
    // E has columns forming the desired frame. Three mode products cost O(d^4).
    let mut out = c.to_vec();
    for axis in 0..3 {
        let mut next = vec![T::ZERO; d * d * d];
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    let a = [i, j, k];
                    for p in 0..d {
                        let mut b = a;
                        b[axis] = p;
                        next[(i * d + j) * d + k] = next[(i * d + j) * d + k]
                            + e[p * d + a[axis]] * out[(b[0] * d + b[1]) * d + b[2]];
                    }
                }
            }
        }
        out = next;
    }
    out
}
/// Fast smooth Hessian curvature, including strict constant shifts. No D4 tensor.
pub fn hessian_curvature<T: Real>(
    jet: &FitnessJet<T>,
    epsilon: T,
    full_riemann: bool,
) -> Result<CurvatureBatch<T>> {
    jet.validate()?;
    let d = jet.dimension;
    let h = expand(d, 2, &jet.hessian)?;
    let spec = metric_spectrum(&h, d, epsilon, MetricPolicy::Strict)?;
    let c = expand(d, 3, &jet.third)?;
    hessian_with_spectrum(c, spec, full_riemann)
}
fn hessian_with_spectrum<T: Real>(
    c: Vec<T>,
    spec: MetricSpectrum<T>,
    full: bool,
) -> Result<CurvatureBatch<T>> {
    let d = spec.dimension;
    // Eigenframe E=Q diag(lambda^-1/2). E^T g E=I.
    let mut e = spec.eigenvectors.clone();
    let mut f = e.clone();
    for i in 0..d {
        for j in 0..d {
            e[i * d + j] = e[i * d + j] / spec.eigenvalues[j].sqrt();
            f[i * d + j] = f[i * d + j] * spec.eigenvalues[j].sqrt();
        }
    }
    let c_hat = transform3(&c, &e, d);
    let mut t = vec![T::ZERO; d];
    for p in 0..d {
        for a in 0..d {
            t[p] = t[p] + c_hat[(a * d + a) * d + p];
        }
    }
    let quarter = T::from_f64(0.25);
    let mut ric_hat = vec![T::ZERO; d * d];
    for b in 0..d {
        for k in 0..d {
            let mut v = T::ZERO;
            for p in 0..d {
                v = v - t[p] * c_hat[(b * d + k) * d + p];
                for a in 0..d {
                    v = v + c_hat[(b * d + a) * d + p] * c_hat[(k * d + a) * d + p];
                }
            }
            ric_hat[b * d + k] = v * quarter;
        }
    }
    let scalar = (0..d).fold(T::ZERO, |s, i| s + ric_hat[i * d + i]);
    let ricci = matmul(&matmul(&f, &ric_hat, d), &transpose(&f, d), d);
    let mut sectional = Vec::new();
    for i in 0..d {
        for j in i + 1..d {
            sectional.push(
                (0..d).fold(T::ZERO, |s, p| {
                    s + c_hat[(i * d + j) * d + p] * c_hat[(i * d + j) * d + p]
                        - c_hat[(i * d + i) * d + p] * c_hat[(j * d + j) * d + p]
                }) * quarter,
            );
        }
    }
    let riemann = if full {
        let mut r = vec![T::ZERO; d.pow(4)];
        for a in 0..d {
            for b in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        let mut v = T::ZERO;
                        for p in 0..d {
                            for q in 0..d {
                                v = v + spec.inverse[p * d + q]
                                    * (c[(a * d + l) * d + p] * c[(b * d + k) * d + q]
                                        - c[(a * d + k) * d + p] * c[(b * d + l) * d + q]);
                            }
                        }
                        r[((a * d + b) * d + k) * d + l] = v * quarter;
                    }
                }
            }
        }
        Some(r)
    } else {
        None
    };
    finish(spec, ricci, scalar, sectional, riemann, "smooth_hessian")
}
pub(crate) fn finish<T: Real>(
    spec: MetricSpectrum<T>,
    ricci: Vec<T>,
    scalar: T,
    sectional: Vec<T>,
    riemann: Option<Vec<T>>,
    branch: &str,
) -> Result<CurvatureBatch<T>> {
    let einstein = ricci
        .iter()
        .zip(&spec.metric)
        .map(|(&r, &g)| r - scalar * g * T::from_f64(0.5))
        .collect::<Vec<_>>();
    require(
        scalar.is_finite()
            && ricci
                .iter()
                .chain(&sectional)
                .chain(&einstein)
                .chain(riemann.iter().flatten())
                .all(|v| v.is_finite()),
        "curvature overflow",
    )?;
    Ok(CurvatureBatch {
        dimension: spec.dimension,
        spectrum: spec,
        ricci,
        scalar,
        einstein,
        sectional,
        riemann,
        branch: branch.into(),
    })
}

/// General metric curvature. Derivative indices precede matrix indices.
/// This path also supplies an independent Levi-Civita reference for Hessian jets.
pub fn metric_curvature<T: Real>(jet: &MetricJet<T>, full: bool) -> Result<CurvatureBatch<T>> {
    let d = jet.dimension;
    symmetric(&jet.metric, d)?;
    require(
        jet.first.len() == d.pow(3)
            && jet.second.len() == d.pow(4)
            && jet.first.iter().chain(&jet.second).all(|x| x.is_finite()),
        "metric derivative shapes/finiteness",
    )?;
    for k in 0..d {
        symmetric(&jet.first[k * d * d..(k + 1) * d * d], d)?;
        for l in 0..d {
            symmetric(&jet.second[(k * d + l) * d * d..(k * d + l + 1) * d * d], d)?;
            let tolerance = if T::PRECISION == crate::Precision::F32 {
                2e-6
            } else {
                2e-13
            };
            require(
                (0..d * d).all(|ij| {
                    let a = jet.second[(k * d + l) * d * d + ij];
                    let b = jet.second[(l * d + k) * d * d + ij];
                    (a - b).abs().to_f64() <= tolerance * (1. + a.abs().to_f64() + b.abs().to_f64())
                }),
                "metric second derivatives must commute",
            )?;
        }
    }
    let (l, q) = symmetric_eigen(&jet.metric, d)?;
    let spec = spectrum(l, q, d)?;
    let dg = |k, i, j| jet.first[(k * d + i) * d + j];
    let dd = |k, l, i, j| jet.second[((k * d + l) * d + i) * d + j];
    // Lowered Christoffel symbols, Gamma_abc.
    let mut gamma = vec![T::ZERO; d.pow(3)];
    for a in 0..d {
        for b in 0..d {
            for c in 0..d {
                gamma[(a * d + b) * d + c] =
                    (dg(b, a, c) + dg(c, a, b) - dg(a, b, c)) * T::from_f64(0.5);
            }
        }
    }
    let curvature = |a: usize, b: usize, c: usize, k: usize| {
        let mut v =
            (dd(c, b, a, k) + dd(k, a, b, c) - dd(c, a, b, k) - dd(k, b, a, c)) * T::from_f64(0.5);
        for p in 0..d {
            for q in 0..d {
                v = v + spec.inverse[p * d + q]
                    * (gamma[(p * d + a) * d + k] * gamma[(q * d + b) * d + c]
                        - gamma[(p * d + a) * d + c] * gamma[(q * d + b) * d + k]);
            }
        }
        v
    };
    let mut ricci = vec![T::ZERO; d * d];
    let mut riemann = full.then(|| vec![T::ZERO; d.pow(4)]);
    let mut e = spec.eigenvectors.clone();
    for i in 0..d {
        for j in 0..d {
            e[i * d + j] = e[i * d + j] / spec.eigenvalues[j].sqrt();
        }
    }
    let pairs: Vec<_> = (0..d)
        .flat_map(|i| (i + 1..d).map(move |j| (i, j)))
        .collect();
    let mut sectional = vec![T::ZERO; pairs.len()];
    for a in 0..d {
        for b in 0..d {
            for c in 0..d {
                for k in 0..d {
                    let v = curvature(a, b, c, k);
                    ricci[b * d + k] = ricci[b * d + k] + spec.inverse[a * d + c] * v;
                    if let Some(r) = &mut riemann {
                        r[((a * d + b) * d + c) * d + k] = v;
                    }
                    if d > 3 {
                        for (index, &(i, j)) in pairs.iter().enumerate() {
                            sectional[index] = sectional[index]
                                + e[a * d + i] * e[b * d + j] * e[c * d + i] * e[k * d + j] * v;
                        }
                    }
                }
            }
        }
    }
    let scalar = ricci
        .iter()
        .zip(&spec.inverse)
        .fold(T::ZERO, |s, (&a, &b)| s + a * b);
    if d == 2 {
        sectional[0] = scalar * T::from_f64(0.5);
    }
    if d == 3 {
        let frame = matmul(&matmul(&transpose(&e, d), &ricci, d), &e, d);
        for (index, &(i, j)) in pairs.iter().enumerate() {
            sectional[index] = frame[i * d + i] + frame[j * d + j] - scalar * T::from_f64(0.5);
        }
    }
    finish(spec, ricci, scalar, sectional, riemann, "general_metric")
}

fn divided1<T: Real>(x: T, y: T) -> T {
    if x > T::ZERO && y > T::ZERO {
        T::ONE
    } else if x < T::ZERO && y < T::ZERO {
        T::ZERO
    } else {
        (x.max(T::ZERO) - y.max(T::ZERO)) / (x - y)
    }
}
fn divided2<T: Real>(x: T, y: T, z: T) -> T {
    if (x > T::ZERO && y > T::ZERO && z > T::ZERO) || (x < T::ZERO && y < T::ZERO && z < T::ZERO) {
        return T::ZERO;
    }
    // Choose opposite-sign end nodes: denominator is bounded by the spectral gap,
    // including repeated eigenvalues on either smooth branch.
    let mut a = [x, y, z];
    a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (divided1(a[2], a[1]) - divided1(a[1], a[0])) / (a[2] - a[0])
}
/// Curvature of g=epsilon I + max(H,0). At the threshold no classical curvature
/// is returned. A mixed-sign spectrum requires D4 and the full metric chain rule.
pub fn fitness_curvature<T: Real>(
    jet: &FitnessJet<T>,
    epsilon: T,
    policy: MetricPolicy,
    threshold: T,
    full: bool,
) -> Result<CurvatureBatch<T>> {
    jet.validate()?;
    require(
        threshold.is_finite() && threshold >= T::ZERO,
        "nonnegative spectral threshold",
    )?;
    if policy == MetricPolicy::Strict {
        return hessian_curvature(jet, epsilon, full);
    }
    let d = jet.dimension;
    let h = expand(d, 2, &jet.hessian)?;
    let (l, q) = symmetric_eigen(&h, d)?;
    require(
        epsilon.is_finite() && epsilon > T::ZERO,
        "positive metric epsilon required",
    )?;
    if l.iter().any(|x| x.abs() <= threshold) {
        return Err(GasError::Capability(
            "fitness eigenvalue at clipping threshold: classical metric curvature unavailable"
                .into(),
        ));
    }
    let spec = spectrum(
        l.iter().map(|&v| v.max(T::ZERO) + epsilon).collect(),
        q.clone(),
        d,
    )?;
    if l.iter().all(|&v| v > T::ZERO) {
        return hessian_with_spectrum(expand(d, 3, &jet.third)?, spec, full);
    }
    if l.iter().all(|&v| v < T::ZERO) {
        return finish(
            spec,
            vec![T::ZERO; d * d],
            T::ZERO,
            vec![T::ZERO; d * (d - 1) / 2],
            full.then(|| vec![T::ZERO; d.pow(4)]),
            "constant_clipped",
        );
    }
    let fourth = jet.fourth.as_ref().ok_or_else(|| {
        GasError::Capability("mixed-sign clipping requires fourth fitness derivatives".into())
    })?;
    let c = expand(d, 3, &jet.third)?;
    let b = expand(d, 4, fourth)?;
    let qt = transpose(&q, d);
    let mut v = Vec::new();
    for k in 0..d {
        v.push(symmetric_congruence(&c[k * d * d..(k + 1) * d * d], &q, d));
    }
    let mut first = vec![T::ZERO; d.pow(3)];
    let mut second = vec![T::ZERO; d.pow(4)];
    for k in 0..d {
        let mut a = vec![T::ZERO; d * d];
        for i in 0..d {
            for j in i..d {
                a[i * d + j] = divided1(l[i], l[j]) * v[k][i * d + j];
                a[j * d + i] = a[i * d + j];
            }
        }
        first[k * d * d..(k + 1) * d * d].copy_from_slice(&symmetric_congruence(&a, &qt, d));
        for m in k..d {
            let w = symmetric_congruence(&b[(k * d + m) * d * d..(k * d + m + 1) * d * d], &q, d);
            for i in 0..d {
                for j in i..d {
                    a[i * d + j] = divided1(l[i], l[j]) * w[i * d + j];
                    for p in 0..d {
                        a[i * d + j] = a[i * d + j]
                            + divided2(l[i], l[p], l[j])
                                * (v[k][i * d + p] * v[m][p * d + j]
                                    + v[m][i * d + p] * v[k][p * d + j]);
                    }
                    a[j * d + i] = a[i * d + j];
                }
            }
            let rotated = symmetric_congruence(&a, &qt, d);
            second[(k * d + m) * d * d..(k * d + m + 1) * d * d].copy_from_slice(&rotated);
            second[(m * d + k) * d * d..(m * d + k + 1) * d * d].copy_from_slice(&rotated);
        }
    }
    let mut out = metric_curvature(
        &MetricJet {
            dimension: d,
            metric: spec.metric,
            first,
            second,
        },
        full,
    )?;
    out.branch = "mixed_clipped".into();
    Ok(out)
}
