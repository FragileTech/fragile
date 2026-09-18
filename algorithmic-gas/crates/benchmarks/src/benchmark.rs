//! The objective selector: identifiers, parameters, domains, and dispatch to the tensor
//! graph (classics) or the host evaluator (COCO BBOB).
use crate::{
    bbob::{self, BbobProblem},
    classics::{self, ObjectiveGraph},
    mixture::Mixture,
};
use algorithmic_gas::{GasError, Real, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Parameter-free variants serialise as their bare identifier (`"rastrigin"`); the others
/// as `{"id": "...", <parameters>}` with the Optimization Lab's parameter names.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Repr", into = "Repr")]
pub enum Benchmark {
    Sphere,
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
    /// `0.5·|x|²`, the unit-curvature well used by the lecture experiments.
    Quadratic,
    /// `0.5·alpha·|x|²`, the Optimization Lab's "Quadratic Well".
    QuadraticWell {
        alpha: f64,
    },
    MexicanHat {
        lambda_h: f64,
        vev: f64,
        field_scale: f64,
        tilt: f64,
    },
    Eggholder,
    Easom,
    HolderTable,
    LennardJones {
        n_atoms: u32,
    },
    Constant,
    StochasticGaussian {
        std: f64,
    },
    GaussianMixture {
        n_gaussians: u32,
        benchmark_seed: u64,
    },
    Bbob {
        function: u8,
        instance: u32,
    },
}

/// Where the objective is computed. `Host` is scalar f64 Rust on every backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveExecution {
    Graph,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientExecution {
    Graph,
    Zero,
    HostCentralDifference,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lambda_h: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vev: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tilt: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub std: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_atoms: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_gaussians: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coco_instance: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark_seed: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Repr {
    Id(String),
    Spec(Spec),
}

impl TryFrom<Repr> for Benchmark {
    type Error = String;
    fn try_from(repr: Repr) -> std::result::Result<Self, String> {
        let spec = match repr {
            Repr::Id(id) => Spec {
                id,
                ..Spec::default()
            },
            Repr::Spec(spec) => spec,
        };
        let mut benchmark =
            Self::from_id(&spec.id).ok_or_else(|| format!("unknown benchmark `{}`", spec.id))?;
        let parameters = [
            ("alpha", spec.alpha),
            ("lambda_h", spec.lambda_h),
            ("vev", spec.vev),
            ("field_scale", spec.field_scale),
            ("tilt", spec.tilt),
            ("std", spec.std),
            ("n_atoms", spec.n_atoms.map(f64::from)),
            ("n_gaussians", spec.n_gaussians.map(f64::from)),
            ("coco_instance", spec.coco_instance.map(f64::from)),
        ];
        for (key, value) in parameters {
            if let Some(value) = value {
                benchmark.set_parameter(key, value)?;
            }
        }
        if let Some(seed) = spec.benchmark_seed {
            match &mut benchmark {
                Self::GaussianMixture { benchmark_seed, .. } => *benchmark_seed = seed,
                _ => return Err(format!("`{}` has no parameter `benchmark_seed`", spec.id)),
            }
        }
        benchmark.validate_parameters()?;
        Ok(benchmark)
    }
}

impl From<Benchmark> for Repr {
    fn from(benchmark: Benchmark) -> Self {
        let mut spec = Spec {
            id: benchmark.id(),
            ..Spec::default()
        };
        match benchmark {
            Benchmark::QuadraticWell { alpha } => spec.alpha = Some(alpha),
            Benchmark::MexicanHat {
                lambda_h,
                vev,
                field_scale,
                tilt,
            } => {
                spec.lambda_h = Some(lambda_h);
                spec.vev = Some(vev);
                spec.field_scale = Some(field_scale);
                spec.tilt = Some(tilt);
            }
            Benchmark::LennardJones { n_atoms } => spec.n_atoms = Some(n_atoms),
            Benchmark::StochasticGaussian { std } => spec.std = Some(std),
            Benchmark::GaussianMixture {
                n_gaussians,
                benchmark_seed,
            } => {
                spec.n_gaussians = Some(n_gaussians);
                spec.benchmark_seed = Some(benchmark_seed);
            }
            Benchmark::Bbob { instance, .. } => spec.coco_instance = Some(instance),
            _ => return Self::Id(spec.id),
        }
        Self::Spec(spec)
    }
}

fn integer(value: f64, low: u32, high: u32, name: &str) -> std::result::Result<u32, String> {
    if value.is_finite()
        && value.fract() == 0.
        && value >= f64::from(low)
        && value <= f64::from(high)
    {
        Ok(value as u32)
    } else {
        Err(format!("{name} must be an integer in {low}..={high}"))
    }
}

impl Benchmark {
    pub const MAX_ATOMS: u32 = 64;
    pub const MAX_GAUSSIANS: u32 = 64;

    /// Catalog identifiers in display order (classics, then `bbob_1..=bbob_24`, then lecture).
    pub fn all() -> Vec<Self> {
        let mut all = vec![
            Self::Sphere,
            Self::from_id("quadratic_well").unwrap(),
            Self::from_id("mexican_hat").unwrap(),
            Self::Rastrigin,
            Self::Eggholder,
            Self::StyblinskiTang,
            Self::Rosenbrock,
            Self::Easom,
            Self::HolderTable,
            Self::from_id("lennard_jones").unwrap(),
            Self::Constant,
            Self::from_id("stochastic_gaussian").unwrap(),
            Self::from_id("gaussian_mixture").unwrap(),
        ];
        all.extend((1..=bbob::FUNCTIONS).map(|function| Self::Bbob {
            function,
            instance: 1,
        }));
        all.push(Self::Quadratic);
        all
    }

    pub fn id(self) -> String {
        match self {
            Self::Sphere => "sphere",
            Self::Rastrigin => "rastrigin",
            Self::Rosenbrock => "rosenbrock",
            Self::StyblinskiTang => "styblinski_tang",
            Self::Quadratic => "quadratic",
            Self::QuadraticWell { .. } => "quadratic_well",
            Self::MexicanHat { .. } => "mexican_hat",
            Self::Eggholder => "eggholder",
            Self::Easom => "easom",
            Self::HolderTable => "holder_table",
            Self::LennardJones { .. } => "lennard_jones",
            Self::Constant => "constant",
            Self::StochasticGaussian { .. } => "stochastic_gaussian",
            Self::GaussianMixture { .. } => "gaussian_mixture",
            Self::Bbob { function, .. } => return format!("bbob_{function}"),
        }
        .into()
    }

    /// The benchmark with its default parameters.
    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "sphere" => Self::Sphere,
            "rastrigin" => Self::Rastrigin,
            "rosenbrock" => Self::Rosenbrock,
            "styblinski_tang" => Self::StyblinskiTang,
            "quadratic" => Self::Quadratic,
            "quadratic_well" => Self::QuadraticWell { alpha: 0.1 },
            "mexican_hat" => Self::MexicanHat {
                lambda_h: 0.13,
                vev: 246.,
                field_scale: 246.,
                tilt: 0.,
            },
            "eggholder" => Self::Eggholder,
            "easom" => Self::Easom,
            "holder_table" => Self::HolderTable,
            "lennard_jones" => Self::LennardJones { n_atoms: 10 },
            "constant" => Self::Constant,
            "stochastic_gaussian" => Self::StochasticGaussian { std: 1. },
            "gaussian_mixture" => Self::GaussianMixture {
                n_gaussians: 3,
                benchmark_seed: 42,
            },
            other => {
                let function: u8 = other.strip_prefix("bbob_")?.parse().ok()?;
                if !(1..=bbob::FUNCTIONS).contains(&function) || other != format!("bbob_{function}")
                {
                    return None;
                }
                Self::Bbob {
                    function,
                    instance: 1,
                }
            }
        })
    }

    /// Current parameter values, keyed by the Optimization Lab's names.
    pub fn parameters(self) -> Vec<(&'static str, f64)> {
        match self {
            Self::QuadraticWell { alpha } => vec![("alpha", alpha)],
            Self::MexicanHat {
                lambda_h,
                vev,
                field_scale,
                tilt,
            } => vec![
                ("lambda_h", lambda_h),
                ("vev", vev),
                ("field_scale", field_scale),
                ("tilt", tilt),
            ],
            Self::LennardJones { n_atoms } => vec![("n_atoms", f64::from(n_atoms))],
            Self::StochasticGaussian { std } => vec![("std", std)],
            Self::GaussianMixture {
                n_gaussians,
                benchmark_seed,
            } => vec![
                ("n_gaussians", f64::from(n_gaussians)),
                ("benchmark_seed", benchmark_seed as f64),
            ],
            Self::Bbob { instance, .. } => vec![("coco_instance", f64::from(instance))],
            _ => vec![],
        }
    }

    pub fn set_parameter(&mut self, key: &str, value: f64) -> std::result::Result<(), String> {
        match (&mut *self, key) {
            (Self::QuadraticWell { alpha }, "alpha") => *alpha = value,
            (Self::MexicanHat { lambda_h, .. }, "lambda_h") => *lambda_h = value,
            (Self::MexicanHat { vev, .. }, "vev") => *vev = value,
            (Self::MexicanHat { field_scale, .. }, "field_scale") => *field_scale = value,
            (Self::MexicanHat { tilt, .. }, "tilt") => *tilt = value,
            (Self::StochasticGaussian { std }, "std") => *std = value,
            (Self::LennardJones { n_atoms }, "n_atoms") => {
                *n_atoms = integer(value, 2, Self::MAX_ATOMS, "atom count")?;
            }
            (Self::GaussianMixture { n_gaussians, .. }, "n_gaussians") => {
                *n_gaussians = integer(value, 1, Self::MAX_GAUSSIANS, "mixture component count")?;
            }
            (Self::GaussianMixture { benchmark_seed, .. }, "benchmark_seed") => {
                *benchmark_seed = u64::from(integer(value, 0, 2147483647, "benchmark seed")?);
            }
            (Self::Bbob { instance, .. }, "coco_instance") => {
                *instance = integer(value, 1, bbob::MAX_INSTANCE, "COCO instance")?;
            }
            _ => return Err(format!("`{}` has no parameter `{key}`", self.id())),
        }
        self.validate_parameters()
    }

    fn validate_parameters(self) -> std::result::Result<(), String> {
        let bounded = |value: f64, low: f64, high: f64, name: &str| {
            if value.is_finite() && value >= low && value <= high {
                Ok(())
            } else {
                Err(format!("invalid {name}"))
            }
        };
        match self {
            Self::QuadraticWell { alpha } => bounded(alpha, 0., 1e6, "quadratic curvature"),
            Self::MexicanHat {
                lambda_h,
                vev,
                field_scale,
                tilt,
            } => bounded(lambda_h, 0., 1e6, "quartic coupling")
                .and(bounded(vev, 0., 1e6, "vev"))
                .and(bounded(field_scale, 1e-12, 1e12, "field scale"))
                .and(bounded(tilt, -1e6, 1e6, "tilt")),
            Self::StochasticGaussian { std } => bounded(std, 0., 1e6, "noise deviation"),
            Self::LennardJones { n_atoms } => {
                integer(f64::from(n_atoms), 2, Self::MAX_ATOMS, "atom count").map(drop)
            }
            Self::GaussianMixture {
                n_gaussians,
                benchmark_seed,
            } => integer(
                f64::from(n_gaussians),
                1,
                Self::MAX_GAUSSIANS,
                "mixture component count",
            )
            .and(integer(
                benchmark_seed as f64,
                0,
                2147483647,
                "benchmark seed",
            ))
            .map(drop),
            Self::Bbob { function, instance } => {
                if (1..=bbob::FUNCTIONS).contains(&function)
                    && (1..=bbob::MAX_INSTANCE).contains(&instance)
                {
                    Ok(())
                } else {
                    Err("COCO BBOB requires function 1-24 and instance 1-1000".into())
                }
            }
            _ => Ok(()),
        }
    }

    pub fn bounds(self) -> (f64, f64) {
        match self {
            Self::Sphere => (-1000., 1000.),
            Self::Rastrigin => (-5.12, 5.12),
            Self::Rosenbrock => (-10., 10.),
            Self::StyblinskiTang => (-5., 5.),
            Self::Quadratic => (-5., 5.),
            Self::Eggholder => (-512., 512.),
            Self::Easom => (-100., 100.),
            Self::LennardJones { .. } => (-15., 15.),
            Self::Bbob { .. } => (-5., 5.),
            Self::QuadraticWell { .. }
            | Self::MexicanHat { .. }
            | Self::HolderTable
            | Self::Constant
            | Self::StochasticGaussian { .. }
            | Self::GaussianMixture { .. } => (-10., 10.),
        }
    }

    /// The only admissible dimension, when the benchmark fixes it.
    pub fn fixed_dimension(self) -> Option<usize> {
        match self {
            Self::Eggholder | Self::Easom | Self::HolderTable => Some(2),
            Self::LennardJones { n_atoms } => Some(3 * n_atoms as usize),
            _ => None,
        }
    }

    pub fn validate(self, d: usize) -> Result<()> {
        let error = |message: String| Err(GasError::Configuration(message));
        if let Err(message) = self.validate_parameters() {
            return error(message);
        }
        if d == 0 || d > 4096 || (self == Self::Rosenbrock && d < 2) {
            return error("invalid benchmark dimension (Rosenbrock requires at least two)".into());
        }
        if let Some(fixed) = self.fixed_dimension()
            && d != fixed
        {
            return error(format!("{} requires exactly {fixed} dimensions", self.id()));
        }
        if matches!(self, Self::Bbob { .. }) && !bbob::DIMENSIONS.contains(&d) {
            return error("COCO BBOB requires dimensions 2, 3, 5, 10, 20, or 40".into());
        }
        Ok(())
    }

    pub fn execution(self) -> ObjectiveExecution {
        match self {
            Self::Bbob { .. } | Self::StochasticGaussian { .. } => ObjectiveExecution::Host,
            _ => ObjectiveExecution::Graph,
        }
    }

    pub fn gradient_execution(self) -> GradientExecution {
        match self {
            Self::Bbob { .. } => GradientExecution::HostCentralDifference,
            Self::Constant | Self::StochasticGaussian { .. } => GradientExecution::Zero,
            _ => GradientExecution::Graph,
        }
    }

    pub fn is_stochastic(self) -> bool {
        matches!(self, Self::StochasticGaussian { .. })
    }

    /// The five objectives with a closed-form jet for the conditional physics metric.
    pub fn supports_physics_metric(self) -> bool {
        matches!(
            self,
            Self::Sphere
                | Self::Rastrigin
                | Self::Rosenbrock
                | Self::StyblinskiTang
                | Self::Quadratic
        )
    }

    /// Known global minimum over the default domain, when one is established.
    pub fn known_minimum(self, d: usize) -> Option<f64> {
        Some(match self {
            Self::Sphere
            | Self::Rastrigin
            | Self::Rosenbrock
            | Self::Quadratic
            | Self::Constant => 0.,
            Self::QuadraticWell { .. } => 0.,
            Self::MexicanHat { tilt: 0., .. } => 0.,
            Self::StyblinskiTang => -39.16616570377142 * d as f64,
            Self::Eggholder => -959.64066271,
            Self::Easom => -1.,
            Self::HolderTable => -19.2085,
            Self::LennardJones { n_atoms } => *LENNARD_JONES_MINIMA.get(n_atoms as usize)?,
            Self::Bbob { function, instance } => {
                bbob::problem(function, d, instance).ok()?.minimum()
            }
            _ => return None,
        })
    }

    /// A global minimiser, when it is unique or conventional.
    pub fn known_minimizer(self, d: usize) -> Option<Vec<f64>> {
        Some(match self {
            Self::Sphere | Self::Rastrigin | Self::Quadratic | Self::QuadraticWell { .. } => {
                vec![0.; d]
            }
            Self::Rosenbrock => vec![1.; d],
            Self::StyblinskiTang => vec![-2.903534027771177; d],
            Self::Eggholder => vec![512., 404.2319],
            Self::Easom => vec![std::f64::consts::PI; 2],
            Self::HolderTable => vec![8.05502, 9.66459],
            Self::Bbob { function, instance } => bbob::problem(function, d, instance)
                .ok()?
                .best_parameter()
                .to_vec(),
            _ => return None,
        })
    }

    /// Deterministic scalar value; a stochastic benchmark returns its expectation.
    pub fn value<T: Real>(self, x: &[T]) -> Result<T> {
        self.validate(x.len())?;
        let c = T::from_f64;
        Ok(match self {
            Self::Sphere => x.iter().fold(T::ZERO, |s, &x| s + x * x),
            Self::Quadratic => x.iter().fold(T::ZERO, |s, &x| s + c(0.5) * x * x),
            Self::Rastrigin => x.iter().fold(T::ZERO, |s, &x| {
                s + x * x - c(10.) * (c(std::f64::consts::TAU) * x).cos() + c(10.)
            }),
            Self::StyblinskiTang => x.iter().fold(T::ZERO, |s, &x| {
                s + c(0.5) * (x * x * x * x - c(16.) * x * x + c(5.) * x)
            }),
            Self::Rosenbrock => x.windows(2).fold(T::ZERO, |s, p| {
                let a = p[0] * p[0] - p[1];
                let b = p[0] - T::ONE;
                s + c(100.) * a * a + b * b
            }),
            _ => {
                let point: Vec<f64> = x.iter().map(|x| x.to_f64()).collect();
                c(self.evaluator(x.len())?.value(&point))
            }
        })
    }

    /// Prepared f64 evaluator: instance data (BBOB, mixture) is resolved once.
    pub fn evaluator(self, d: usize) -> Result<Evaluator> {
        self.validate(d)?;
        let prepared = match self {
            Self::Bbob { function, instance } => Prepared::Bbob(
                bbob::problem(function, d, instance).map_err(GasError::Configuration)?,
            ),
            Self::GaussianMixture { .. } => Prepared::Mixture(self.mixture(d)),
            _ => Prepared::None,
        };
        Ok(Evaluator {
            benchmark: self,
            prepared,
        })
    }

    pub(crate) fn mixture(self, d: usize) -> Mixture {
        let Self::GaussianMixture {
            n_gaussians,
            benchmark_seed,
        } = self
        else {
            unreachable!("mixture components of a non-mixture benchmark");
        };
        let (low, high) = self.bounds();
        Mixture::seeded(benchmark_seed, n_gaussians as usize, d, low, high)
    }

    /// Tensor graph for value (`[walkers, 1]`) or gradient (`[walkers, d]`).
    pub fn graph(self, d: usize, gradient: bool) -> Result<ObjectiveGraph> {
        self.validate(d)?;
        match self {
            Self::Sphere
            | Self::Rastrigin
            | Self::Rosenbrock
            | Self::StyblinskiTang
            | Self::Quadratic => Ok(ObjectiveGraph {
                expression: self.expression(d, gradient)?,
                constants: vec![],
            }),
            Self::Bbob { .. } => Err(GasError::Capability(
                "COCO BBOB objectives are evaluated on the host, not as a tensor graph".into(),
            )),
            Self::GaussianMixture { .. } => {
                Ok(classics::graph(self, Some(&self.mixture(d)), d, gradient))
            }
            _ => Ok(classics::graph(self, None, d, gradient)),
        }
    }
}

/// Global minima of Lennard-Jones clusters indexed by atom count (Cambridge Cluster Database).
const LENNARD_JONES_MINIMA: [f64; 14] = [
    f64::NAN,
    f64::NAN,
    -1.,
    -3.,
    -6.,
    -9.103852,
    -12.712062,
    -16.505384,
    -19.821489,
    -24.113360,
    -28.422532,
    -32.765970,
    -37.967600,
    -44.326801,
];

enum Prepared {
    None,
    Bbob(Arc<BbobProblem>),
    Mixture(Mixture),
}

pub struct Evaluator {
    benchmark: Benchmark,
    prepared: Prepared,
}

impl Evaluator {
    pub fn value(&self, x: &[f64]) -> f64 {
        match (&self.prepared, self.benchmark) {
            (Prepared::Bbob(problem), _) => problem.evaluate(x),
            (Prepared::Mixture(mixture), benchmark) => classics::value(benchmark, Some(mixture), x),
            (
                _,
                benchmark @ (Benchmark::Sphere
                | Benchmark::Rastrigin
                | Benchmark::Rosenbrock
                | Benchmark::StyblinskiTang
                | Benchmark::Quadratic),
            ) => benchmark
                .value::<f64>(x)
                .expect("dimension validated by evaluator"),
            (_, benchmark) => classics::value(benchmark, None, x),
        }
    }

    pub fn bbob(&self) -> Option<&BbobProblem> {
        match &self.prepared {
            Prepared::Bbob(problem) => Some(problem),
            _ => None,
        }
    }
}
