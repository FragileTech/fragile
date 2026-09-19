//! Dirac representation (`γ5 = i γ0 γ1 γ2 γ3`) and the spinor lift of a colour
//! vector. The lift `E(w) = (w1 + i w2, w3) / sqrt|w|` is the convention of
//! the reference code: odd, not norm preserving and not rotation equivariant;
//! operators built on it say so in their descriptor.
use crate::physics::qft::math::C;

/// `[upper two, lower two]` components.
pub type Spinor = [C; 4];
/// Row-major `4 × 4`.
pub type Matrix4 = [C; 16];

/// `γ^μ`, `μ = 0..=3`.
pub fn gamma(mu: usize) -> Matrix4 {
    let _ = mu;
    [C::ZERO; 16]
}
pub fn gamma5() -> Matrix4 {
    [C::ZERO; 16]
}
/// `σ^{μν} = (i/2) [γ^μ, γ^ν]`.
pub fn sigma(mu: usize, nu: usize) -> Matrix4 {
    let _ = (mu, nu);
    [C::ZERO; 16]
}
/// Two-spinor of a real 3-vector; `None` when `|w| ≤ 1e-12`.
pub fn embed(w: [f64; 3]) -> Option<[C; 2]> {
    let _ = w;
    None
}
/// Dirac spinor of a colour vector in three dimensions: the lift of its
/// imaginary part as upper and of its real part as lower components. `None`
/// when either part has no lift.
pub fn lift(color: &[C]) -> Option<Spinor> {
    let _ = color;
    None
}
/// `ψ̄_a Γ ψ_b` with `ψ̄ = ψ† γ0`: `matrix` is `Γ`, the `γ0` is applied here.
pub fn bilinear(a: &Spinor, matrix: &Matrix4, b: &Spinor) -> C {
    let _ = (a, matrix, b);
    C::ZERO
}
