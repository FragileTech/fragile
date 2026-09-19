//! Effective-edge spinors of the twistor formulation, through `qft::edge_spinor`
//! (dominant column, ties select column zero). The edge vector `(Δt, Δx)` mixes
//! units with `c = 1` implicit. The phase of a spinor is fixed by the column
//! rule alone, which singles out the third axis: only `|contraction|²` is free
//! of that convention.
use crate::physics::qft::math::C;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeSpinor {
    pub lambda: [C; 2],
    pub mu: [C; 2],
    pub column: usize,
}

/// Spinor pair of the edge `(dt, dx)` with velocity difference `dv`; `None`
/// for a null or nonfinite edge, and for one whose column norm overflows, so
/// that `lambda` always has unit norm.
pub fn edge(dx: [f64; 3], dv: [f64; 3], dt: f64, alpha: f64) -> Option<EdgeSpinor> {
    crate::physics::qft::edge_spinor(dx, dv, dt, alpha)
        .map(|(lambda, mu, column)| EdgeSpinor { lambda, mu, column })
        .filter(|s| (s.lambda[0].abs2() + s.lambda[1].abs2() - 1.).abs() <= 1e-9)
}

/// `ε_{AB} λ_a^A λ_b^B`.
pub fn contraction(a: &EdgeSpinor, b: &EdgeSpinor) -> C {
    a.lambda[0] * b.lambda[1] - a.lambda[1] * b.lambda[0]
}

/// `λ_a† σ λ_b` for the three Pauli matrices, in the order `x, y, z`; `λ_a`
/// is conjugated. Exchanging the two spinors conjugates every component.
pub fn pauli_bilinear(a: &EdgeSpinor, b: &EdgeSpinor) -> [C; 3] {
    let (a, b) = (&a.lambda, &b.lambda);
    let (p, q) = (a[0].conj() * b[1], a[1].conj() * b[0]);
    let m = p - q;
    [
        p + q,
        C::new(m.im, -m.re),
        a[0].conj() * b[0] - a[1].conj() * b[1],
    ]
}

/// Real spin-2 components of `w wᵀ` without conjugation, in the order
/// `xy, xz, yz, (x² − y²)/√2, (2z² − x² − y²)/√6`. The basis is not
/// orthonormal: the three off-diagonal components lack a factor `√2`.
pub fn spin_two(w: &[C; 3]) -> [f64; 5] {
    let [xx, yy, zz] = [w[0] * w[0], w[1] * w[1], w[2] * w[2]].map(|s| s.re);
    [
        (w[0] * w[1]).re,
        (w[0] * w[2]).re,
        (w[1] * w[2]).re,
        (xx - yy) / std::f64::consts::SQRT_2,
        (2. * zz - xx - yy) / 6f64.sqrt(),
    ]
}
