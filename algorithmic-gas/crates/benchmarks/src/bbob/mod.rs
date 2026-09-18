//! Pure-Rust port of the COCO 2.8.2 `bbob` suite (24 noiseless functions).
//!
//! Instances reproduce the reference C implementation: the same seeded optimum,
//! rotations, objective offset and transformation order. Evaluation is always f64.
mod functions;
pub mod rng;
mod transforms;

use std::sync::{Arc, Mutex, OnceLock};

pub const DIMENSIONS: [usize; 6] = [2, 3, 5, 10, 20, 40];
pub const FUNCTIONS: u8 = 24;
pub const MAX_INSTANCE: u32 = 1000;

/// One `(function, dimension, instance)` problem with its precomputed instance data.
#[derive(Debug)]
pub struct BbobProblem {
    pub function: u8,
    pub dimensions: usize,
    pub instance: u32,
    pub fopt: f64,
    pub(crate) xopt: Vec<f64>,
    pub(crate) best: Vec<f64>,
    /// Row-major `d × d` matrices; meaning depends on the function.
    pub(crate) m1: Vec<f64>,
    pub(crate) m2: Vec<f64>,
    pub(crate) gallagher: Option<functions::Gallagher>,
}

impl BbobProblem {
    pub fn new(function: u8, dimensions: usize, instance: u32) -> Result<Self, String> {
        if !(1..=FUNCTIONS).contains(&function)
            || !(1..=MAX_INSTANCE).contains(&instance)
            || !DIMENSIONS.contains(&dimensions)
        {
            return Err(
                "COCO BBOB requires function 1-24, dimensions 2, 3, 5, 10, 20, or 40 and instance 1-1000"
                    .into(),
            );
        }
        Ok(functions::build(function, dimensions, instance))
    }

    /// COCO problem identifier, e.g. `bbob_f001_i01_d02`.
    pub fn problem_id(&self) -> String {
        format!(
            "bbob_f{:03}_i{:02}_d{:02}",
            self.function, self.instance, self.dimensions
        )
    }

    /// Reference optimum location (`coco_problem_get_best_parameter`).
    pub fn best_parameter(&self) -> &[f64] {
        &self.best
    }

    /// Reference minimum: the objective at [`Self::best_parameter`].
    pub fn minimum(&self) -> f64 {
        self.evaluate(&self.best)
    }

    /// `coco_evaluate_function`: an infinite coordinate returns its magnitude, NaN returns NaN.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), self.dimensions);
        if let Some(infinite) = x.iter().find(|x| x.is_infinite()) {
            return infinite.abs();
        }
        if x.iter().any(|x| x.is_nan()) {
            return f64::NAN;
        }
        functions::evaluate(self, x)
    }
}

type Key = (u8, usize, u32);
type Entries = Vec<(Key, Arc<BbobProblem>)>;
static CACHE: OnceLock<Mutex<Entries>> = OnceLock::new();
const CACHE_ENTRIES: usize = 8;

/// Shared instance data; building a 40-dimensional problem costs two Gram–Schmidt passes.
pub fn problem(function: u8, dimensions: usize, instance: u32) -> Result<Arc<BbobProblem>, String> {
    let key = (function, dimensions, instance);
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut entries = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, found)) = entries.iter().find(|(k, _)| *k == key) {
        return Ok(found.clone());
    }
    let built = Arc::new(BbobProblem::new(function, dimensions, instance)?);
    if entries.len() == CACHE_ENTRIES {
        entries.remove(0);
    }
    entries.push((key, built.clone()));
    Ok(built)
}

/// Suite grouping used by the COCO documentation and the laboratory catalogs.
pub fn group(function: u8) -> &'static str {
    match function {
        ..=5 => "Separable",
        6..=9 => "Moderate conditioning",
        10..=14 => "Ill-conditioned",
        15..=19 => "Multimodal, global structure",
        _ => "Multimodal, weak structure",
    }
}

pub const NAMES: [&str; 24] = [
    "Sphere",
    "Ellipsoid separable",
    "Rastrigin separable",
    "Bueche–Rastrigin",
    "Linear slope",
    "Attractive sector",
    "Step ellipsoid",
    "Rosenbrock original",
    "Rosenbrock rotated",
    "Ellipsoid rotated",
    "Discus",
    "Bent cigar",
    "Sharp ridge",
    "Different powers",
    "Rastrigin rotated",
    "Weierstrass",
    "Schaffer F7 (condition 10)",
    "Schaffer F7 (condition 1000)",
    "Griewank–Rosenbrock",
    "Schwefel",
    "Gallagher 101 peaks",
    "Gallagher 21 peaks",
    "Katsuura",
    "Lunacek bi-Rastrigin",
];
