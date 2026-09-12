//! Packed multivariate Taylor algebra through order twelve.
//!
//! Coefficients are derivatives divided by multi-index factorials. In three
//! dimensions order three needs 20 coefficients; order four needs 35. Products
//! use a precomputed sparse convolution, shared by all queries in a population.
use crate::{Real, Result, error::require};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug)]
pub struct JetSpace {
    pub dimension: usize,
    pub order: usize,
    pub indices: Vec<Vec<usize>>,
    lookup: BTreeMap<Vec<usize>, usize>,
    products: Vec<Vec<(usize, usize)>>,
}
impl JetSpace {
    pub fn new(dimension: usize, order: usize) -> Result<Arc<Self>> {
        require(
            (1..=16).contains(&dimension) && (0..=12).contains(&order),
            "jets support dimensions 1..16 and orders 0..12",
        )?;
        // Bound the complete coefficient count before enumerating multi-indices.
        // Degree <= order in d variables has binomial(d + order, order) terms.
        let mut coefficients = 1usize;
        for k in 1..=order {
            coefficients = coefficients * (dimension + k) / k;
            require(
                coefficients <= 1024,
                "jet coefficient capacity (1024) exceeded",
            )?;
        }
        fn enumerate(d: usize, remaining: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if d == 1 {
                a.push(remaining);
                out.push(a.clone());
                a.pop();
            } else {
                for x in (0..=remaining).rev() {
                    a.push(x);
                    enumerate(d - 1, remaining - x, a, out);
                    a.pop();
                }
            }
        }
        let mut indices = Vec::new();
        for degree in 0..=order {
            enumerate(dimension, degree, &mut Vec::new(), &mut indices);
        }
        // Explicit resource bound, independent of input data.
        require(
            indices.len() <= 1024,
            "jet coefficient capacity (1024) exceeded",
        )?;
        let lookup: BTreeMap<_, _> = indices
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, a)| (a, i))
            .collect();
        let mut products = vec![Vec::new(); indices.len()];
        for (i, a) in indices.iter().enumerate() {
            for (j, b) in indices.iter().enumerate() {
                let sum: Vec<_> = a.iter().zip(b).map(|(x, y)| x + y).collect();
                if let Some(&k) = lookup.get(&sum) {
                    products[k].push((i, j));
                }
            }
        }
        Ok(Arc::new(Self {
            dimension,
            order,
            indices,
            lookup,
            products,
        }))
    }
    pub fn constant<T: Real>(self: &Arc<Self>, value: T) -> Jet<T> {
        let mut coefficients = vec![T::ZERO; self.indices.len()];
        coefficients[0] = value;
        Jet {
            space: self.clone(),
            coefficients,
        }
    }
    pub fn variable<T: Real>(self: &Arc<Self>, value: T, axis: usize) -> Result<Jet<T>> {
        require(
            axis < self.dimension && self.order >= 1,
            "jet variable axis/order",
        )?;
        let mut j = self.constant(value);
        let mut a = vec![0; self.dimension];
        a[axis] = 1;
        j.coefficients[self.lookup[&a]] = T::ONE;
        Ok(j)
    }
}

#[derive(Clone, Debug)]
pub struct Jet<T: Real> {
    pub space: Arc<JetSpace>,
    pub coefficients: Vec<T>,
}
impl<T: Real> Jet<T> {
    pub fn value(&self) -> T {
        self.coefficients[0]
    }
    pub fn constant(&self, x: f64) -> Self {
        self.space.constant(T::from_f64(x))
    }
    pub fn add(&self, rhs: &Self) -> Self {
        assert!(Arc::ptr_eq(&self.space, &rhs.space), "different jet spaces");
        Self {
            space: self.space.clone(),
            coefficients: self
                .coefficients
                .iter()
                .zip(&rhs.coefficients)
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }
    pub fn scale(&self, s: T) -> Self {
        Self {
            space: self.space.clone(),
            coefficients: self.coefficients.iter().map(|&a| a * s).collect(),
        }
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.scale(-T::ONE))
    }
    pub fn mul(&self, rhs: &Self) -> Self {
        assert!(Arc::ptr_eq(&self.space, &rhs.space), "different jet spaces");
        let coefficients = self
            .space
            .products
            .iter()
            .map(|pairs| {
                pairs.iter().fold(T::ZERO, |s, &(i, j)| {
                    s + self.coefficients[i] * rhs.coefficients[j]
                })
            })
            .collect();
        Self {
            space: self.space.clone(),
            coefficients,
        }
    }
    fn compose(&self, coefficients: &[T]) -> Self {
        let mut delta = self.clone();
        delta.coefficients[0] = T::ZERO;
        let mut out = self.space.constant(T::ZERO);
        for &c in coefficients.iter().rev() {
            out = out.mul(&delta);
            out.coefficients[0] = out.coefficients[0] + c;
        }
        out
    }
    pub fn pow(&self, exponent: f64) -> Self {
        if exponent == 0. {
            return self.constant(1.);
        }
        // Integer powers also work at zero and for negative arguments.
        if (0. ..=16.).contains(&exponent) && exponent.floor() == exponent {
            let mut out = self.constant(1.);
            for _ in 0..exponent as usize {
                out = out.mul(self);
            }
            return out;
        }
        let mut coefficients = vec![T::ZERO; self.space.order + 1];
        let mut factor = T::ONE;
        for (k, c) in coefficients.iter_mut().enumerate() {
            if k > 0 {
                factor = factor * T::from_f64((exponent - (k - 1) as f64) / k as f64);
            }
            *c = factor * self.value().powf(T::from_f64(exponent - k as f64));
        }
        self.compose(&coefficients)
    }
    pub fn exp(&self) -> Self {
        let mut c = vec![self.value().exp(); self.space.order + 1];
        for k in 1..c.len() {
            c[k] = c[k - 1] / T::from_f64(k as f64);
        }
        self.compose(&c)
    }
    pub fn cos(&self) -> Self {
        let x = self.value();
        let mut c = Vec::new();
        let mut factorial = 1.;
        for k in 0..=self.space.order {
            if k > 0 {
                factorial *= k as f64;
            }
            let v = match k % 4 {
                0 => x.cos(),
                1 => -x.sin(),
                2 => -x.cos(),
                _ => x.sin(),
            };
            c.push(v / T::from_f64(factorial));
        }
        self.compose(&c)
    }
    pub fn logistic(&self, amplitude: f64, floor: f64) -> Self {
        let one = self.constant(1.);
        let s = if self.value() >= T::ZERO {
            one.add(&self.scale(-T::ONE).exp()).pow(-1.)
        } else {
            let e = self.exp();
            e.mul(&one.add(&e).pow(-1.))
        };
        s.scale(T::from_f64(amplitude)).add(&self.constant(floor))
    }
    /// An ordinary (not factorial-normalized) derivative. Axis order is irrelevant.
    pub fn derivative(&self, axes: &[usize]) -> Result<T> {
        require(
            axes.len() <= self.space.order && axes.iter().all(|&a| a < self.space.dimension),
            "jet derivative order/axis",
        )?;
        let mut a = vec![0; self.space.dimension];
        for &k in axes {
            a[k] += 1;
        }
        let factorial = a
            .iter()
            .map(|&k| (1..=k).product::<usize>())
            .product::<usize>();
        Ok(self.coefficients[self.space.lookup[&a]] * T::from_f64(factorial as f64))
    }
    pub fn validate(&self) -> Result<()> {
        require(
            self.coefficients.iter().all(|v| v.is_finite()),
            "nonfinite jet; nonsmooth/overflowed query",
        )
    }
}

#[cfg(test)]
mod high_order_tests {
    use super::*;
    #[test]
    fn twelfth_order_coefficients_match_the_exponential_and_resource_bound() {
        let space = JetSpace::new(2, 12).unwrap();
        let x = space.variable::<f64>(0.3, 0).unwrap();
        let y = x.exp();
        let mut factorial = 1.;
        for order in 0..=12 {
            if order > 0 {
                factorial *= order as f64;
            }
            let i = space
                .indices
                .iter()
                .position(|a| a == &vec![order, 0])
                .unwrap();
            assert!((y.coefficients[i] - 0.3f64.exp() / factorial).abs() < 1e-12);
        }
        let unit = space.variable::<f64>(1., 0).unwrap();
        let inverse_root = unit.pow(-0.5);
        let mut coefficient = 1.;
        for order in 0..=12 {
            if order > 0 {
                coefficient *= (-0.5 - (order - 1) as f64) / order as f64;
            }
            let i = space
                .indices
                .iter()
                .position(|a| a == &vec![order, 0])
                .unwrap();
            assert!((inverse_root.coefficients[i] - coefficient).abs() < 1e-12);
        }
        assert!(JetSpace::new(16, 12).is_err());
        assert!(JetSpace::new(2, 13).is_err());
    }
}
