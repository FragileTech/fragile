//! `std::mt19937_64`, reproduced so seeded benchmark instances match the C++ laboratory.
const NN: usize = 312;
const MM: usize = 156;
const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
const UPPER: u64 = 0xFFFF_FFFF_8000_0000;
const LOWER: u64 = 0x7FFF_FFFF;

pub struct Mt64 {
    state: [u64; NN],
    index: usize,
}

impl Mt64 {
    pub fn new(seed: u64) -> Self {
        let mut state = [0u64; NN];
        state[0] = seed;
        for i in 1..NN {
            state[i] = 6364136223846793005u64
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Self { state, index: NN }
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.index == NN {
            for i in 0..NN {
                let x = (self.state[i] & UPPER) | (self.state[(i + 1) % NN] & LOWER);
                let mut next = self.state[(i + MM) % NN] ^ (x >> 1);
                if x & 1 == 1 {
                    next ^= MATRIX_A;
                }
                self.state[i] = next;
            }
            self.index = 0;
        }
        let mut x = self.state[self.index];
        self.index += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^ (x >> 43)
    }

    /// `OptimizationRng::uniform01`: the top 24 bits as a single-precision fraction.
    pub fn uniform01(&mut self) -> f64 {
        f64::from((self.next_u64() >> 40) as f32 * (1.0f32 / 16_777_216.0f32))
    }
}
