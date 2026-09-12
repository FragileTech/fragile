use crate::{GasError, Population, Real, Result, error::require};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoxDomain {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}
impl BoxDomain {
    pub fn validate(&self, width: usize) -> Result<()> {
        require(
            self.lower.len() == width && self.upper.len() == width,
            "box dimension differs from selected field",
        )?;
        require(
            self.lower
                .iter()
                .zip(&self.upper)
                .all(|(a, b)| a.is_finite() && b.is_finite() && a < b),
            "box bounds must be finite and strictly ordered",
        )
    }
    pub fn minimum_image<T: Real>(&self, delta: T, axis: usize) -> T {
        let length = T::from_f64(self.upper[axis]) - T::from_f64(self.lower[axis]);
        delta - length * (delta / length + T::from_f64(0.5)).floor()
    }
    pub fn validate_precision<T: Real>(&self, width: usize) -> Result<()> {
        self.validate(width)?;
        require(
            self.lower.iter().zip(&self.upper).all(|(&a, &b)| {
                let lo = T::from_f64(a);
                let hi = T::from_f64(b);
                let length = hi - lo;
                lo.is_finite() && hi.is_finite() && length.is_finite() && length > T::ZERO
            }),
            "box interval collapses or overflows in the run precision",
        )
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BoundaryPolicy {
    #[default]
    Unbounded,
    AbsorbingBox {
        field: String,
        domain: BoxDomain,
    },
    PeriodicBox {
        field: String,
        domain: BoxDomain,
    },
    ExternalTermination,
    Composed {
        policies: Vec<BoundaryPolicy>,
    },
}
impl BoundaryPolicy {
    /// Returns fields repaired in place. The caller must reconcile these edits
    /// with a simulator before any saved state is advanced.
    pub fn apply<T: Real>(&self, population: &mut Population<T>) -> Result<Vec<String>> {
        population.validate()?;
        for field in population.observations.fields.values() {
            for (i, row) in field.values().chunks(field.width()).enumerate() {
                population.validity[i].invalid |= row.iter().any(|x| !x.is_finite());
            }
        }
        let mut repaired = Vec::new();
        match self {
            Self::Unbounded | Self::ExternalTermination => {}
            Self::AbsorbingBox { field, domain } | Self::PeriodicBox { field, domain } => {
                let batch = population.observations.field_mut(field)?;
                let width = batch.width();
                domain.validate_precision::<T>(width)?;
                if batch.item_shape().len() != 1 {
                    return Err(GasError::Shape(
                        "box requires explicit coordinate vectors, not image tensors".into(),
                    ));
                }
                let periodic = matches!(self, Self::PeriodicBox { .. });
                let mut changed = false;
                for (i, row) in batch.values_mut().chunks_mut(width).enumerate() {
                    for (j, x) in row.iter_mut().enumerate() {
                        let lo = T::from_f64(domain.lower[j]);
                        let hi = T::from_f64(domain.upper[j]);
                        if !x.is_finite() {
                            continue;
                        }
                        if periodic {
                            if *x < lo || *x >= hi {
                                let length = hi - lo;
                                *x = *x - length * ((*x - lo) / length).floor();
                                if !x.is_finite() || *x < lo {
                                    return Err(GasError::Numerical("periodic repair overflow; rescale coordinates or increase precision".into()));
                                }
                                if *x >= hi {
                                    *x = lo;
                                }
                                changed = true;
                            }
                        } else if *x < lo || *x > hi {
                            population.validity[i].out_of_bounds = true;
                        }
                    }
                }
                if changed {
                    repaired.push(field.clone());
                }
            }
            Self::Composed { policies } => {
                for p in policies {
                    repaired.extend(p.apply(population)?);
                }
                repaired.sort();
                repaired.dedup();
            }
        }
        Ok(repaired)
    }
    pub fn periodic_domain(&self, field: &str) -> Option<&BoxDomain> {
        match self {
            Self::PeriodicBox { field: f, domain } if f == field => Some(domain),
            Self::Composed { policies } => policies.iter().find_map(|p| p.periodic_domain(field)),
            _ => None,
        }
    }
}
