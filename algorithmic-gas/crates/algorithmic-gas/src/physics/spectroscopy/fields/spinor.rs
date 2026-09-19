//! Effective-edge spinors of the twistor formulation, through `qft::edge_spinor`
//! (dominant column, ties select column zero). The edge vector `(Δt, Δx)` mixes
//! units with `c = 1` implicit.
use crate::physics::qft::math::C;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeSpinor {
    pub lambda: [C; 2],
    pub mu: [C; 2],
    pub column: usize,
}

/// Spinor pair of the edge `(dt, dx)` with velocity difference `dv`; `None`
/// for a null or nonfinite edge.
pub fn edge(dx: [f64; 3], dv: [f64; 3], dt: f64, alpha: f64) -> Option<EdgeSpinor> {
    crate::physics::qft::edge_spinor(dx, dv, dt, alpha).map(|(lambda, mu, column)| EdgeSpinor {
        lambda,
        mu,
        column,
    })
}

/// `ε_{AB} λ_a^A λ_b^B`.
pub fn contraction(a: &EdgeSpinor, b: &EdgeSpinor) -> C {
    a.lambda[0] * b.lambda[1] - a.lambda[1] * b.lambda[0]
}
