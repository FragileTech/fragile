//! Seeded isotropic-axis Gaussian mixture shared by the scalar and graph objectives.
use crate::mt64::Mt64;

#[derive(Clone, Debug, PartialEq)]
pub struct Mixture {
    pub components: usize,
    pub dimensions: usize,
    /// Row-major `components × dimensions`.
    pub centers: Vec<f64>,
    pub stds: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Mixture {
    /// Draw order of the C++ laboratory: every center, then every deviation.
    pub fn seeded(seed: u64, components: usize, dimensions: usize, low: f64, high: f64) -> Self {
        let mut rng = Mt64::new(seed);
        let count = components * dimensions;
        let centers = (0..count)
            .map(|_| low + (high - low) * rng.uniform01())
            .collect();
        let stds = (0..count).map(|_| 0.1 + 1.9 * rng.uniform01()).collect();
        Self {
            components,
            dimensions,
            centers,
            stds,
            weights: vec![1. / components as f64; components],
        }
    }

    fn log_components(&self, x: &[f64]) -> Vec<f64> {
        let d = self.dimensions;
        (0..self.components)
            .map(|c| {
                let mut log = self.weights[c].ln();
                for (k, x) in x.iter().enumerate().take(d) {
                    let j = c * d + k;
                    let z = (x - self.centers[j]) / self.stds[j];
                    log -=
                        0.5 * ((2. * std::f64::consts::PI).ln() + 2. * self.stds[j].ln() + z * z);
                }
                log
            })
            .collect()
    }

    /// Negative log density, stabilised by the largest component.
    pub fn value(&self, x: &[f64]) -> f64 {
        let logs = self.log_components(x);
        let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logs.iter().map(|log| (log - maximum).exp()).sum();
        -(maximum + sum.ln())
    }
}
