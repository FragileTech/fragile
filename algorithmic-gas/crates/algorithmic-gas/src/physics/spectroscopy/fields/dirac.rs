//! Dirac representation of signature `(+, −, −, −)` (`γ5 = i γ0 γ1 γ2 γ3`)
//! and the spinor lift of a colour vector. The lift `E(w) = (w1 + i w2, w3) /
//! sqrt|w|` is odd, not norm preserving and covariant only under rotations
//! about the third axis; operators built on it say so in their descriptor.
//! The Dirac adjoint is `ψ† γ0` with this Hermitian `γ0`. Every matrix has one
//! entry `±1` or `±i` per row, so the algebra is exact in floating point.
use crate::physics::qft::math::{C, mul};

/// `[upper two, lower two]` components: the eigenspaces `+1` and `−1` of
/// `γ0`, which are parity blocks and not chirality eigenspaces.
pub type Spinor = [C; 4];
/// Row-major `4 × 4`.
pub type Matrix4 = [C; 16];

const Z: C = C::ZERO;
const P: C = C::ONE;
const M: C = C { re: -1., im: 0. };
const I: C = C { re: 0., im: 1. };
const J: C = C { re: 0., im: -1. };
/// `γ0 = diag(1, 1, −1, −1)` and `γk = [[0, σk], [−σk, 0]]`.
const GAMMA: [Matrix4; 4] = [
    [P, Z, Z, Z, Z, P, Z, Z, Z, Z, M, Z, Z, Z, Z, M],
    [Z, Z, Z, P, Z, Z, P, Z, Z, M, Z, Z, M, Z, Z, Z],
    [Z, Z, Z, J, Z, Z, I, Z, Z, I, Z, Z, J, Z, Z, Z],
    [Z, Z, P, Z, Z, Z, Z, M, M, Z, Z, Z, Z, P, Z, Z],
];
/// `i γ0 γ1 γ2 γ3 = [[0, 1], [1, 0]]` in `2 × 2` blocks.
const GAMMA5: Matrix4 = [Z, Z, P, Z, Z, Z, Z, P, P, Z, Z, Z, Z, P, Z, Z];

/// `γ^μ`, `μ = 0..=3`.
pub fn gamma(mu: usize) -> Matrix4 {
    assert!(mu < 4, "gamma matrix index above 3");
    GAMMA[mu]
}
/// `γ5 = i γ0 γ1 γ2 γ3`.
pub fn gamma5() -> Matrix4 {
    GAMMA5
}
/// `σ^{μν} = (i/2) [γ^μ, γ^ν]`, which is `i γ^μ γ^ν` for `μ ≠ ν`.
pub fn sigma(mu: usize, nu: usize) -> Matrix4 {
    let (a, b) = (gamma(mu), gamma(nu));
    let (ab, ba) = (mul(&a, &b, 4), mul(&b, &a, 4));
    std::array::from_fn(|k| I * (ab[k] - ba[k]) * 0.5)
}
pub(crate) fn apply(matrix: &Matrix4, spinor: &Spinor) -> Spinor {
    std::array::from_fn(|r| (0..4).fold(C::ZERO, |s, j| s + matrix[r * 4 + j] * spinor[j]))
}
/// Two-spinor `E(w)` of a real 3-vector; `None` when `|w| ≤ 1e-12` or `w` is
/// not finite. `E(−w) = −E(w)` holds without rounding and `E(w)† E(w) = |w|`.
pub fn embed(w: [f64; 3]) -> Option<[C; 2]> {
    let threshold = 1e-12;
    let norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if !(norm.is_finite() && norm > threshold) {
        return None;
    }
    let scale = norm.sqrt();
    Some([C::new(w[0] / scale, w[1] / scale), C::new(w[2] / scale, 0.)])
}
/// Dirac spinor of a colour vector in three dimensions: the lift of its
/// imaginary part as upper and of its real part as lower components, so that
/// the inversion `c → −c*` acts as `γ0`. `None` when either part has no lift;
/// a real colour, which a vanishing phase produces, has none.
pub fn lift(color: &[C]) -> Option<Spinor> {
    let &[x, y, z] = color else {
        return None;
    };
    let upper = embed([x.im, y.im, z.im])?;
    let lower = embed([x.re, y.re, z.re])?;
    Some([upper[0], upper[1], lower[0], lower[1]])
}
/// `ψ̄_a ψ_b` with `ψ̄ = ψ† γ0`.
pub(crate) fn overlap(a: &Spinor, b: &Spinor) -> C {
    let term = |r: usize| a[r].conj() * b[r];
    term(0) + term(1) - term(2) - term(3)
}
/// `ψ̄_a Γ ψ_b` with `ψ̄ = ψ† γ0`: `matrix` is `Γ`, the `γ0` is applied here.
pub fn bilinear(a: &Spinor, matrix: &Matrix4, b: &Spinor) -> C {
    overlap(a, &apply(matrix, b))
}
