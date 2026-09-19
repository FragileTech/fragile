//! Named variants of the Fractal Gas family. The engine is not a variant: it
//! executes the instance a `GasConfig` describes, and a named constructor of
//! `GasConfig` fixes a variant up to the constructor's arguments. Variants the
//! book specifies but the engine does not implement are registry entries
//! without a constructor.
pub mod einstein_hilbert;
pub mod euclidean;
pub mod viscous_euclidean;

use crate::{GasConfig, GasError, Result, error::require};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Variant {
    Euclidean,
    ViscousEuclidean,
    EinsteinHilbert,
    Geometric,
    Latent,
    Environment,
}

/// Size, dimension and time step of a variant's reference instance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ReferenceInstance {
    pub walkers: usize,
    pub dimensions: usize,
    pub dt: f64,
}

/// Registry row of one variant, as listed by the command-line tools and the browser.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VariantInfo {
    pub name: &'static str,
    pub title: &'static str,
    pub book_label: &'static str,
    pub implemented: bool,
    pub summary: &'static str,
    pub reference: Option<ReferenceInstance>,
}

impl Variant {
    /// Every variant, in the order of the book's variants chapter.
    pub fn all() -> &'static [Variant] {
        &[
            Self::Euclidean,
            Self::ViscousEuclidean,
            Self::EinsteinHilbert,
            Self::Geometric,
            Self::Latent,
            Self::Environment,
        ]
    }

    /// Stable snake_case identifier; equal to the serde wire name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::ViscousEuclidean => "viscous_euclidean",
            Self::EinsteinHilbert => "einstein_hilbert",
            Self::Geometric => "geometric",
            Self::Latent => "latent",
            Self::Environment => "environment",
        }
    }

    /// Accepts the identifier of `name`, also written with hyphens.
    pub fn from_name(name: &str) -> Result<Self> {
        let wanted = name.replace('-', "_");
        Self::all()
            .iter()
            .copied()
            .find(|v| v.name() == wanted)
            .ok_or_else(|| {
                let known: Vec<_> = Self::all().iter().map(|v| v.name()).collect();
                GasError::Configuration(format!(
                    "unknown gas variant {name:?}; known variants: {}",
                    known.join(", ")
                ))
            })
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Euclidean => "Euclidean Gas",
            Self::ViscousEuclidean => "Viscous Euclidean Gas",
            Self::EinsteinHilbert => "Einstein–Hilbert Gas",
            Self::Geometric => "Geometric Gas",
            Self::Latent => "Latent Fractal Gas",
            Self::Environment => "Environment Gas",
        }
    }

    /// Label of the variant's definition in the book.
    pub fn book_label(self) -> &'static str {
        match self {
            Self::Euclidean => "def-variant-euclidean",
            Self::ViscousEuclidean => "def-variant-viscous-euclidean",
            Self::EinsteinHilbert => "def-variant-einstein-hilbert",
            Self::Geometric => "def-variant-geometric",
            Self::Latent => "def-variant-latent",
            Self::Environment => "def-variant-environment",
        }
    }

    /// Whether the engine has a `GasConfig` constructor for the variant.
    pub fn implemented(self) -> bool {
        matches!(
            self,
            Self::Euclidean | Self::ViscousEuclidean | Self::EinsteinHilbert
        )
    }

    pub fn summary(self) -> &'static str {
        match self {
            Self::Euclidean => {
                "Walkers (x, v) in an absorbing box with BAOAB kinetics and a caller-supplied \
                 objective; the variant analyzed by the convergence program."
            }
            Self::ViscousEuclidean => {
                "The Euclidean Gas with a dense Gaussian-kernel viscous coupling added to both \
                 B kicks; the Euclidean variant on which the colour state is defined."
            }
            Self::EinsteinHilbert => {
                "A free gas rewarded with each walker's share of the Einstein-Hilbert action of \
                 the tessellation geometry, with graph viscosity and Boris curl rotation."
            }
            Self::Geometric => {
                "Force and noise covariance adapted to the measured fitness landscape, stated \
                 as a Stratonovich SDE; the dynamics the Fractal Set chapter calls Adaptive Gas."
            }
            Self::Latent => {
                "Walkers on a latent chart with a metric, a reward 1-form and Boris-BAOAB \
                 kinetics."
            }
            Self::Environment => {
                "The reinforcement-learning member: the kinetic operator is one transition of \
                 an external environment and the reward is its reward signal."
            }
        }
    }

    /// Reference instance of an implemented variant.
    pub fn reference(self) -> Option<ReferenceInstance> {
        let (walkers, dimensions, dt) = match self {
            Self::Euclidean => (64, 2, 0.04),
            Self::ViscousEuclidean => (200, 3, 0.04),
            Self::EinsteinHilbert => (500, 3, 0.002),
            Self::Geometric | Self::Latent | Self::Environment => return None,
        };
        Some(ReferenceInstance {
            walkers,
            dimensions,
            dt,
        })
    }

    /// Configuration of the variant in the given position dimension and time
    /// step, with the remaining constructor arguments at their reference
    /// values: `viscous_euclidean::reference_viscosity()` and
    /// `einstein_hilbert::REFERENCE_TEMPERATURE`. The Einstein-Hilbert
    /// configuration does not depend on the dimension; its geometry stage
    /// checks the population it is built with.
    pub fn config(self, dimensions: usize, dt: f64) -> Result<GasConfig> {
        match self {
            Self::Euclidean => GasConfig::euclidean(dimensions, dt),
            Self::ViscousEuclidean => GasConfig::viscous_euclidean(
                dimensions,
                dt,
                viscous_euclidean::reference_viscosity(),
            ),
            Self::EinsteinHilbert => {
                require(dimensions > 0, "invalid Einstein-Hilbert gas dimension")?;
                GasConfig::einstein_hilbert(einstein_hilbert::REFERENCE_TEMPERATURE, dt)
            }
            Self::Geometric | Self::Latent | Self::Environment => Err(self.not_implemented()),
        }
    }

    fn not_implemented(self) -> GasError {
        GasError::Capability(format!(
            "the {} ({}) is specified in the book and not implemented by the engine",
            self.title(),
            self.book_label()
        ))
    }

    /// Configuration of the reference instance.
    pub fn default_config(self) -> Result<GasConfig> {
        let reference = self.reference().ok_or_else(|| self.not_implemented())?;
        self.config(reference.dimensions, reference.dt)
    }

    pub fn info(self) -> VariantInfo {
        VariantInfo {
            name: self.name(),
            title: self.title(),
            book_label: self.book_label(),
            implemented: self.implemented(),
            summary: self.summary(),
            reference: self.reference(),
        }
    }
}

/// The registry as rows, in the order of `Variant::all`.
pub fn catalog() -> Vec<VariantInfo> {
    Variant::all().iter().map(|v| v.info()).collect()
}
