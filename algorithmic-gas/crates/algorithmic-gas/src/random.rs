use crate::Real;
use serde::{Deserialize, Serialize};
pub const RNG_VERSION: u32 = 1;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u64)]
pub enum Stream {
    Initialize = 1,
    Distance = 2,
    Cloning = 3,
    Accept = 4,
    Revival = 5,
    CloneNoise = 6,
    Kinetic = 7,
    Domain = 8,
    HistoricalDistance = 9,
}
/// Scheduling-independent addressed host reference RNG. Its integer operations
/// are identical on native and WASM; no shared mutable backend RNG is used.
#[derive(Clone, Copy, Debug)]
pub struct RandomStream {
    seed: u64,
    step: u64,
    operator: Stream,
    walker: u64,
    substep: u64,
    counter: u64,
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
impl RandomStream {
    pub fn new(seed: u64, step: u64, operator: Stream, walker: u64, substep: u64) -> Self {
        Self {
            seed,
            step,
            operator,
            walker,
            substep,
            counter: 0,
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut key = mix(self.seed);
        for x in [
            self.step,
            self.operator as u64,
            self.walker,
            self.substep,
            self.counter,
        ] {
            key = mix(key ^ mix(x));
        }
        self.counter = self.counter.wrapping_add(1);
        key
    }
    /// Strictly between zero and one, including after rounding to the run dtype.
    pub fn uniform<T: Real>(&mut self) -> T {
        let bits = match T::PRECISION {
            crate::Precision::F32 => 23,
            crate::Precision::F64 => 52,
        };
        T::from_f64(((self.next_u64() >> (64 - bits)) as f64 + 0.5) / (1u64 << bits) as f64)
    }
    pub fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "internal sampler requires candidates");
        let upper = upper as u64;
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return (x % upper) as usize;
            }
        }
    }
    pub fn gaussian<T: Real>(&mut self) -> T {
        (T::from_f64(-2.) * self.uniform::<T>().ln()).sqrt()
            * (T::from_f64(std::f64::consts::TAU) * self.uniform::<T>()).cos()
    }
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.index(i + 1);
            values.swap(i, j);
        }
    }
}
