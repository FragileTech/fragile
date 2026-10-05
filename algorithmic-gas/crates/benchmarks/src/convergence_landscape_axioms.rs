//! Global non-deception constants for bounded perturbations of confining forces.
use crate::Benchmark;
use algorithmic_gas::{ComputeBackend, ExecutionContext, GasError, Result, TensorBatch};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentGradientBudget {
    pub curvature_lower: f64,
    pub residual_force_upper: f64,
    pub minimum_segment_length: f64,
    pub gradient_energy_lower: f64,
    pub positive_nondeception: bool,
    pub population_independent: bool,
    #[serde(default)]
    pub proof_method: String,
}

/// Same tensor expression and CPU backend as the native Rastrigin provider.
pub async fn native_rastrigin_gradient(
    cx: &mut ExecutionContext,
    positions: &TensorBatch<f64>,
) -> Result<TensorBatch<f64>> {
    cx.evaluate(
        &Benchmark::Rastrigin.expression(positions.width(), true)?,
        std::slice::from_ref(positions),
    )
    .await
}

/// Reverse normalized-L2 triangle inequality; no global convexity of the
/// perturbed potential is required. The residual and SPD hypotheses are analytic.
pub fn segment_gradient_budget(
    m: f64,
    residual: f64,
    length: f64,
) -> Result<SegmentGradientBudget> {
    if ![m, residual, length].iter().all(|x| x.is_finite())
        || m <= 0.
        || residual < 0.
        || length <= 0.
    {
        return Err(GasError::Configuration(
            "m,L>0 and residual>=0 must be finite".into(),
        ));
    }
    let lower = (m * length / 12f64.sqrt() - residual).max(0.).powi(2);
    if !lower.is_finite() {
        return Err(GasError::Configuration("Segment constant overflow".into()));
    }
    Ok(SegmentGradientBudget {
        curvature_lower: m,
        residual_force_upper: residual,
        minimum_segment_length: length,
        gradient_energy_lower: lower,
        positive_nondeception: lower > 0.,
        population_independent: true,
        proof_method: "Reverse normalized-L2 triangle with globally bounded residual".into(),
    })
}

pub fn rastrigin_gradient_budget(d: usize) -> Result<SegmentGradientBudget> {
    if d == 0 {
        return Err(GasError::Configuration(
            "Positive physical dimension required".into(),
        ));
    }
    let residual = 20. * std::f64::consts::PI * (d as f64).sqrt();
    segment_gradient_budget(2., residual, 12f64.sqrt() * residual)
}

fn sinc(z: f64) -> f64 {
    if z.abs() < 1e-3 {
        1. - z * z / 6. + z.powi(4) / 120. - z.powi(6) / 5040.
    } else {
        z.sin() / z
    }
}

fn linear_sine_integral(z: f64) -> f64 {
    if z.abs() < 1e-3 {
        z / 6. - z.powi(3) / 60. + z.powi(5) / 1680. - z.powi(7) / 90720.
    } else {
        (z.sin() - z * z.cos()) / (2. * z * z)
    }
}

/// Exact one-dimensional trigonometric integrals summed over coordinates.
/// Returns the normalized segment gradient energy and an analytic Simpson
/// quadrature error bound for the same source expression on [0,1].
pub fn rastrigin_segment_energy(x: &[f64], y: &[f64], panels: usize) -> Result<(f64, f64)> {
    if x.is_empty()
        || x.len() != y.len()
        || x.iter().chain(y).any(|z| !z.is_finite())
        || panels < 2
        || !panels.is_multiple_of(2)
    {
        return Err(GasError::Configuration(
            "Finite equal segment endpoints and even panels required".into(),
        ));
    }
    let k = std::f64::consts::TAU;
    let amplitude = 20. * std::f64::consts::PI;
    let mut energy = 0.;
    let mut fourth = 0.;
    for (&a, &b) in x.iter().zip(y) {
        let mid = a / 2. + b / 2.;
        let v = b - a;
        let z = k * v / 2.;
        let cross = mid * (k * mid).sin() * sinc(z) + v * (k * mid).cos() * linear_sine_integral(z);
        energy += 4. * (mid * mid + v * v / 12.)
            + 4. * amplitude * cross
            + amplitude * amplitude / 2. * (1. - (2. * k * mid).cos() * sinc(2. * z));
        fourth += v.powi(4)
            * (4. * amplitude * k.powi(4) * a.abs().max(b.abs())
                + 16. * amplitude * k.powi(3)
                + 8. * amplitude.powi(2) * k.powi(4));
    }
    let error = fourth / (180. * (panels as f64).powi(4));
    if !energy.is_finite() || !error.is_finite() || energy < -1e-10 {
        return Err(GasError::Configuration(
            "Nonfinite segment energy or quadrature envelope".into(),
        ));
    }
    Ok((energy.max(0.), error))
}

/// Coordinate oscillation certificate: every segment of length sqrt(d) or more
/// has one coordinate spanning a full Rastrigin period. Its gradient variance
/// alone gives a positive energy floor independent of both d and N.
pub fn rastrigin_coordinate_energy_lower(dimension: usize, length: f64) -> Result<f64> {
    if dimension == 0 || !length.is_finite() || length < (dimension as f64).sqrt() {
        return Err(GasError::Configuration(
            "Coordinate certificate requires d>=1 and L>=sqrt(d)".into(),
        ));
    }
    let ell = length / (dimension as f64).sqrt();
    let pi = std::f64::consts::PI;
    let amplitude = 20. * pi;
    let lower = ell.powi(2) / 3.
        + amplitude.powi(2) * (0.5 - 1. / (4. * pi * ell) - 1. / (pi.powi(2) * ell.powi(2)))
        - 4. * amplitude * (1. / (2. * pi) + 1. / (2. * pi.powi(2) * ell));
    if !lower.is_finite() || lower <= 0. {
        return Err(GasError::Configuration(
            "Coordinate certificate arithmetic outside positive finite range".into(),
        ));
    }
    Ok(lower)
}

pub fn rastrigin_short_gradient_budget(dimension: usize) -> Result<SegmentGradientBudget> {
    let length = (dimension as f64).sqrt();
    let lower = rastrigin_coordinate_energy_lower(dimension, length)?;
    Ok(SegmentGradientBudget {
        curvature_lower: 2.,
        residual_force_upper: 20. * std::f64::consts::PI * length,
        minimum_segment_length: length,
        gradient_energy_lower: lower,
        positive_nondeception: true,
        population_independent: true,
        proof_method:
            "Coordinate gradient variance over at least one full period; no midpoint bound".into(),
    })
}
