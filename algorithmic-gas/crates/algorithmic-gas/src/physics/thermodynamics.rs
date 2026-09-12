//! Exact finite-state reference thermodynamics and controlled response.
//! These computations do not identify the gas archive with a Gibbs measure.
use crate::{GasError, Result, error::require};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", content = "value", rename_all = "snake_case")]
pub enum ExtendedValue {
    Finite(f64),
    Infinite,
    Unavailable(String),
}
pub fn relative_entropy(p: &[f64], q: &[f64]) -> Result<ExtendedValue> {
    probability(p)?;
    probability(q)?;
    require(p.len() == q.len(), "KL shape")?;
    let mut out = 0.;
    for (&p, &q) in p.iter().zip(q) {
        if p > 0. {
            if q == 0. {
                return Ok(ExtendedValue::Infinite);
            }
            out += p * (p.ln() - q.ln());
        }
    }
    Ok(ExtendedValue::Finite(out.max(0.)))
}
fn probability(p: &[f64]) -> Result<()> {
    require(
        !p.is_empty()
            && p.iter().all(|v| v.is_finite() && *v >= 0.)
            && (p.iter().sum::<f64>() - 1.).abs() < 1e-10,
        "normalized probability vector required",
    )
}
pub(crate) fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>> {
    require(
        n > 0
            && n <= 256
            && a.len() == n * n
            && b.len() == n
            && a.iter().chain(&b).all(|v| v.is_finite()),
        "linear system shape/finiteness",
    )?;
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        require(
            a[pivot * n + k].abs() > 1e-14,
            "singular/ill-conditioned response or regression system",
        )?;
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i * n + j] * x[j]).sum::<f64>()) / a[i * n + i];
    }
    require(x.iter().all(|v| v.is_finite()), "linear solve overflow")?;
    Ok(x)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FiniteKernel {
    pub states: usize,
    pub transition: Vec<f64>,
}
impl FiniteKernel {
    pub fn validate(&self) -> Result<()> {
        require(
            self.states > 0
                && self.states <= 256
                && self.transition.len() == self.states * self.states,
            "finite kernel shape/capacity",
        )?;
        for row in self.transition.chunks_exact(self.states) {
            probability(row)?;
        }
        // Explicit irreducibility test. Uniqueness cannot be inferred from a
        // least-squares stationary vector for a reducible chain.
        for transposed in [false, true] {
            let mut seen = vec![false; self.states];
            seen[0] = true;
            let mut queue = vec![0];
            while let Some(i) = queue.pop() {
                for (j, visited) in seen.iter_mut().enumerate() {
                    let index = if transposed {
                        j * self.states + i
                    } else {
                        i * self.states + j
                    };
                    if !*visited && self.transition[index] > 0. {
                        *visited = true;
                        queue.push(j);
                    }
                }
            }
            require(
                seen.iter().all(|v| *v),
                "finite kernel must be irreducible for stationary response",
            )?;
        }
        Ok(())
    }
    fn stationary_system(&self) -> Vec<f64> {
        let n = self.states;
        let mut a = vec![0.; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = self.transition[j * n + i] - if i == j { 1. } else { 0. };
            }
        }
        a[(n - 1) * n..].fill(1.);
        a
    }
    pub fn stationary(&self) -> Result<Vec<f64>> {
        self.validate()?;
        let n = self.states;
        let mut b = vec![0.; n];
        b[n - 1] = 1.;
        let pi = solve(self.stationary_system(), b, n)?;
        require(
            pi.iter().all(|v| *v > 0.),
            "stationary solve lost strict positivity",
        )?;
        probability(&pi)?;
        Ok(pi)
    }
    /// Parameter derivative at an interior, dominated differentiable kernel.
    /// The caller supplies dP/dtheta; this method checks its normalization tangent.
    pub fn response(&self, derivative: &[f64], observable: &[f64]) -> Result<ResponseReport> {
        let pi = self.stationary()?;
        let n = self.states;
        require(
            derivative.len() == n * n
                && observable.len() == n
                && derivative.iter().chain(observable).all(|v| v.is_finite())
                && derivative
                    .chunks_exact(n)
                    .all(|r| r.iter().sum::<f64>().abs() < 1e-10),
            "response derivative must be a finite kernel tangent",
        )?;
        let mut rhs = vec![0.; n];
        for j in 0..n {
            rhs[j] = -(0..n).map(|i| pi[i] * derivative[i * n + j]).sum::<f64>();
        }
        rhs[n - 1] = 0.;
        let dpi = solve(self.stationary_system(), rhs, n)?;
        let stationary_fisher = dpi.iter().zip(&pi).map(|(d, p)| d * d / p).sum();
        let mut transition_fisher = 0.;
        for i in 0..n {
            for j in 0..n {
                let p = self.transition[i * n + j];
                let dp = derivative[i * n + j];
                if p == 0. && dp != 0. {
                    return Err(GasError::Capability(
                        "transition Fisher unavailable at a moving support boundary".into(),
                    ));
                }
                if p > 0. {
                    transition_fisher += pi[i] * dp * dp / p;
                }
            }
        }
        // Poisson equation (I-P+1*pi)h=f-pi(f), unique under pi(h)=0.
        let mean = pi.iter().zip(observable).map(|(p, f)| p * f).sum::<f64>();
        let mut a = vec![0.; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 1. } else { 0. } - self.transition[i * n + j] + pi[j];
            }
        }
        let h = solve(a, observable.iter().map(|f| f - mean).collect(), n)?;
        let response = dpi.iter().zip(observable).map(|(p, f)| p * f).sum::<f64>();
        let mut poisson_response = 0.;
        for i in 0..n {
            for j in 0..n {
                poisson_response += pi[i] * derivative[i * n + j] * h[j];
            }
        }
        Ok(ResponseReport {
            stationary_derivative: dpi,
            observable_response: response,
            poisson_response,
            stationary_fisher,
            transition_fisher,
        })
    }
    /// Steady path KL rate against reversal of this same kernel with all states
    /// even. Velocity reversal and driven reverse protocols need a separate law.
    pub fn thermodynamics(&self) -> Result<ThermodynamicReport> {
        let pi = self.stationary()?;
        let n = self.states;
        let entropy = -pi.iter().map(|p| p * p.ln()).sum::<f64>();
        let mut rate = 0.;
        let mut infinite = false;
        for i in 0..n {
            for j in 0..n {
                let flux = pi[i] * self.transition[i * n + j];
                let reversed = pi[j] * self.transition[j * n + i];
                if flux > 0. {
                    if reversed == 0. {
                        infinite = true;
                    } else {
                        rate += flux * (flux.ln() - reversed.ln());
                    }
                }
            }
        }
        let residual = (0..n)
            .map(|j| {
                ((0..n)
                    .map(|i| pi[i] * self.transition[i * n + j])
                    .sum::<f64>()
                    - pi[j])
                    .abs()
            })
            .fold(0., f64::max);
        Ok(ThermodynamicReport {
            stationary: pi,
            shannon_entropy: entropy,
            irreversibility_per_step: if infinite {
                ExtendedValue::Infinite
            } else {
                ExtendedValue::Finite(rate.max(0.))
            },
            stationarity_residual: residual,
            reverse_protocol: "same stationary finite kernel; identity state involution".into(),
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseReport {
    pub stationary_derivative: Vec<f64>,
    pub observable_response: f64,
    pub poisson_response: f64,
    pub stationary_fisher: f64,
    pub transition_fisher: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermodynamicReport {
    pub stationary: Vec<f64>,
    pub shannon_entropy: f64,
    pub irreversibility_per_step: ExtendedValue,
    pub stationarity_residual: f64,
    pub reverse_protocol: String,
}
