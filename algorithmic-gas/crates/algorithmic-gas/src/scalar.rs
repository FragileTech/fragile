use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    F64,
}
mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
/// One computation dtype per run. Indices and random addresses remain integers.
pub trait Real:
    sealed::Sealed
    + Copy
    + Debug
    + Default
    + PartialOrd
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static
{
    const PRECISION: Precision;
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    fn exp_m1(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn floor(self) -> Self;
    fn abs(self) -> Self;
    fn is_finite(self) -> bool;
    fn min(self, v: Self) -> Self;
    fn max(self, v: Self) -> Self;
}
macro_rules! real {
    ($t:ty,$p:ident) => {
        impl Real for $t {
            const PRECISION: Precision = Precision::$p;
            const ZERO: Self = 0.;
            const ONE: Self = 1.;
            fn from_f64(v: f64) -> Self {
                v as Self
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn exp(self) -> Self {
                self.exp()
            }
            fn ln(self) -> Self {
                self.ln()
            }
            fn ln_1p(self) -> Self {
                self.ln_1p()
            }
            fn exp_m1(self) -> Self {
                self.exp_m1()
            }
            fn sin(self) -> Self {
                self.sin()
            }
            fn cos(self) -> Self {
                self.cos()
            }
            fn powf(self, p: Self) -> Self {
                self.powf(p)
            }
            fn floor(self) -> Self {
                self.floor()
            }
            fn abs(self) -> Self {
                self.abs()
            }
            fn is_finite(self) -> bool {
                self.is_finite()
            }
            fn min(self, v: Self) -> Self {
                self.min(v)
            }
            fn max(self, v: Self) -> Self {
                self.max(v)
            }
        }
    };
}
real!(f32, F32);
real!(f64, F64);
