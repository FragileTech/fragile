//! Algorithmic scales of a gas configuration and the coupling proxies derived
//! from them. The Standard Model map is an inversion from reference inputs to
//! gas parameters; it is labelled as such and never presented as a measurement.
//!
//! Three kinds of number are kept apart. A scale is configured or fixed in the
//! warm-up. A proxy is a formula of the book on those scales: it can be dialled
//! to any value and is never compared with a Standard Model coupling. A target
//! is a Standard Model input re-expressed as a gas parameter. Reference units
//! are those of the simulation: unit length, time and mass, `k_B = 1`.
use super::{
    config::{HyperchargeNormalization, Range, StandardModelInputs},
    frame::position_field,
    measurement::Measurement,
    report::{CouplingReport, Quantity},
};
use crate::{
    GasConfig, Result,
    donor::DonorModule,
    geometry::{Distance, Kernel},
    kinetic::KineticKind,
    noise::{FactorValues, NoiseGeometry},
    tessellation::MetricKind,
};
use std::f64::consts::PI;

const PROXY_LABEL: &str = "def-sm-coupling-definition";
const CLOCK_LABEL: &str = "thm-su2-coupling-constant";
const TARGET_LABEL: &str = "def-qft-report-target-convention";
const INVERSION_LABEL: &str = "prop-qft-report-inversion";
const KIND_NOTE: &str = "Scales are read from the gas configuration and from the warm-up \
    calibration of the measurement. Couplings are the book's parameter proxies \
    (def-sm-coupling-definition, thm-su2-coupling-constant, thm-u1-coupling-constant) evaluated \
    on those scales and pair statistics: functions of chosen parameters with conventional \
    normalisation factors, not measurements of a gauge coupling.";
const MATCHING_NOTE: &str = "A proxy equals a physical gauge coupling only after a field \
    normalisation and a matching calculation (thm-sm-g1-coupling, \
    prop-sm-coupling-correspondence); none is performed here, and no tension against Standard \
    Model values is reported for any proxy.";
const GRAPH_NOTE: &str = "Graph viscosity uses tessellation edge weights and has no Gaussian \
    bandwidth: rho, the clock proxy, the viscous energy scale and the strong proxy with its \
    upper bound are undefined.";
const SQUASH_NOTE: &str = "The companion kernel acts on a squashed phase-space distance; its \
    width is not the range of the book's algorithmic distance.";
/// Factor between the presupposed and the measured `⟨K_visc²⟩` above which the
/// viscous calibration is reported as inconsistent.
const CALIBRATION_TOLERANCE: f64 = 2.;

/// Scales read from the configuration alone.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AlgorithmicScales {
    pub dimension: usize,
    pub dt: Option<f64>,
    pub friction: Option<f64>,
    pub temperature: Option<f64>,
    pub viscosity: Option<f64>,
    pub epsilon_d: Option<f64>,
    pub epsilon_c: Option<f64>,
    pub epsilon_clone: f64,
}
impl AlgorithmicScales {
    /// `dt` of a BAOAB or Brownian integrator and `friction` of BAOAB.
    /// `temperature` is the thermostat target `σ_v²/(2γ)` of an isotropic
    /// constant noise scale `σ_v`: the stationary velocity variance per
    /// component of the O step at any step size, not a measured temperature.
    /// `viscosity` is the positive coefficient of the dense viscous force,
    /// else of the graph one. `epsilon_d` and `epsilon_c` are the ranges of
    /// Gaussian companion kernels, `range`. `epsilon_clone` regularises the
    /// cloning score and is never a range.
    pub fn from_config(gas: &GasConfig, dimension: usize) -> Self {
        let (dt, friction) = match gas.kinetic.integrator {
            KineticKind::Baoab { dt, friction, .. } => (Some(dt), Some(friction)),
            KineticKind::Brownian { dt, .. } => (Some(dt), None),
            KineticKind::DirectJump { .. } | KineticKind::Environment => (None, None),
        };
        let positions = position_field(gas);
        Self {
            dimension,
            dt,
            friction,
            temperature: noise_scale(gas)
                .zip(friction)
                .filter(|(_, gamma)| *gamma > 0.)
                .map(|(sigma, gamma)| sigma * sigma / (2. * gamma)),
            viscosity: viscosity(gas),
            epsilon_d: range(&gas.distance_donors, positions, dimension),
            epsilon_c: range(&gas.cloning_donors, positions, dimension),
            epsilon_clone: gas.clone_decision.epsilon,
        }
    }
}

/// Scale `σ_v` of `dv = −γ v dt + σ_v dW` for an isotropic constant noise.
fn noise_scale(gas: &GasConfig) -> Option<f64> {
    match &gas.kinetic.noise.geometry {
        NoiseGeometry::Isotropic {
            scale: FactorValues::Constant { values },
        } if values.len() == 1 => Some(values[0]),
        _ => None,
    }
}
fn viscosity(gas: &GasConfig) -> Option<f64> {
    let dense = gas.qft.viscosity.as_ref().map(|v| v.coefficient);
    let graph = gas.qft.graph_viscosity.as_ref().map(|v| v.coefficient);
    dense.or(graph).filter(|nu| *nu > 0.)
}

/// Range `ε` of a Gaussian companion kernel in the distance
/// `d² = |Δx|² + λ|Δv|²`. The kernel weight is `exp(−D²/2w²)` on the module's
/// distance `D`, which divides positions by a scale `p`, so `ε = w p`. A
/// squashed distance keeps `w`. `None` for another kernel, for unequal
/// coordinate scales and for a distance whose coordinates are not the
/// positions.
fn range(module: &DonorModule, positions: &str, dimension: usize) -> Option<f64> {
    let Kernel::Gaussian { width } = module.kernel else {
        return None;
    };
    match &module.distance {
        Distance::SquashedPhaseSpace {
            positions: field, ..
        } if field == positions => Some(width),
        Distance::PhaseSpace {
            positions: field,
            position_scale,
            ..
        } if field == positions => Some(width * position_scale),
        Distance::Euclidean { field, scales, .. } if field == positions => {
            let scale = |k: usize| scales.get(k).copied().unwrap_or(1.);
            (1..dimension)
                .all(|k| scale(k) == scale(0))
                .then(|| width * scale(0))
        }
        _ => None,
    }
}

/// Velocity weight `λ` of `d² = |Δx|² + λ|Δv|²`: the configured weight times
/// `(p/q)²` for position and velocity scales `p`, `q`.
fn velocity_weight(distance: &Distance) -> Option<f64> {
    match distance {
        Distance::SquashedPhaseSpace { lambda, .. } => Some(*lambda),
        Distance::PhaseSpace {
            position_scale,
            velocity_scale,
            lambda,
            ..
        } => Some(lambda * (position_scale / velocity_scale).powi(2)),
        Distance::Euclidean { .. } => Some(0.),
        Distance::Cosine { .. } => None,
    }
}

/// Quadratic Casimir `C₂(n) = (n² − 1)/(2n)` of the defining representation
/// of SU(n).
pub fn casimir(n: usize) -> f64 {
    let n = n as f64;
    (n * n - 1.) / (2. * n)
}

/// Normalisation `d(d² − 1)/12` assigned to the kernel-moment proxy.
pub fn kernel_factor(d: usize) -> f64 {
    let d = d as f64;
    d * (d * d - 1.) / 12.
}

/// `g₁² = ħ N₁/ε_d²` with `N₁ = E exp(−D²/ε_d²) ≤ 1`.
pub fn g1_squared(h_eff: f64, n1: f64, epsilon_d: f64) -> f64 {
    h_eff * n1 / (epsilon_d * epsilon_d)
}

/// `g₂² = (2ħ/ε_c²) C₂(2)/C₂(d)`; `None` below `d = 2`.
pub fn g2_casimir_squared(h_eff: f64, epsilon_c: f64, d: usize) -> Option<f64> {
    (d >= 2).then(|| 2. * h_eff * casimir(2) / (casimir(d) * epsilon_c * epsilon_c))
}

/// `g² = m τ ρ²/ε_c²`, the clock proxy.
pub fn g2_clock_squared(mass: f64, dt: f64, rho: f64, epsilon_c: f64) -> f64 {
    mass * dt * (rho / epsilon_c).powi(2)
}

/// `g_d² = (ν²/ħ²) d(d² − 1)/12 · E K²`; `None` below `d = 2`, where the
/// factor vanishes and SU(d) does not exist.
pub fn gd_squared(viscosity: f64, h_eff: f64, d: usize, k2: f64) -> Option<f64> {
    (d >= 2).then(|| (viscosity / h_eff).powi(2) * kernel_factor(d) * k2)
}

/// The kernel moment a configured `ν` presupposes if it is to realise `g_d`:
/// `gd_squared` solved for `E K²`, `(ħ g_d/ν)²/(d(d² − 1)/12)`. A viscosity
/// taken from an inversion is only meaningful together with the moment the
/// kernel that will actually run has, so the report states this number beside
/// the measured one. `None` below `d = 2` or without a positive `ν`.
pub fn presupposed_kernel_second_moment(
    viscosity: f64,
    h_eff: f64,
    d: usize,
    gd: f64,
) -> Option<f64> {
    (d >= 2 && viscosity > 0.).then(|| (h_eff * gd / viscosity).powi(2) / kernel_factor(d))
}

/// `m ε_c²/(2τ)`, the action scale of the Gaussian-phase convention.
pub fn kernel_action_scale(mass: f64, epsilon_c: f64, dt: f64) -> f64 {
    mass * epsilon_c * epsilon_c / (2. * dt)
}

/// Couplings of the target convention, `[value, error]` with first-order
/// errors: `e = sqrt(4π α_em)`, `g₂ = e/sin θ_W`, `g₁ = e/cos θ_W` (the
/// hypercharge coupling, not `sqrt(5/3)` times it), `g₃ = sqrt(4π α_s)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetCouplings {
    pub e_em: [f64; 2],
    pub g1: [f64; 2],
    pub g2: [f64; 2],
    pub g3: [f64; 2],
}
impl TargetCouplings {
    pub fn of(inputs: &StandardModelInputs) -> Result<Self> {
        inputs.validate()?;
        let [inverse, inverse_error] = inputs.alpha_em_inverse;
        let [s, s_error] = inputs.sin2_theta_w;
        let [alpha_s, alpha_s_error] = inputs.alpha_s;
        let e = (4. * PI / inverse).sqrt();
        let e_relative = 0.5 * inverse_error / inverse;
        let over = |x: f64| {
            let g = e / x.sqrt();
            [g, g * e_relative.hypot(0.5 * s_error / x)]
        };
        let g3 = (4. * PI * alpha_s).sqrt();
        Ok(Self {
            e_em: [e, e * e_relative],
            g1: over(1. - s),
            g2: over(s),
            g3: [g3, 0.5 * g3 * alpha_s_error / alpha_s],
        })
    }
}

/// Gas parameters the calibration dictionary assigns to target couplings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationTargets {
    pub epsilon_c: Option<f64>,
    pub epsilon_d: f64,
    pub viscosity: Option<f64>,
    pub fitness_scale: f64,
    pub time_step: Option<f64>,
    pub viscous_range: Option<f64>,
}
impl CalibrationTargets {
    /// Solution of the dictionary for positive `mass`, `h_eff` and pair
    /// statistics `n1`, `k2` held fixed, with `C = C₂(2)/C₂(d)`:
    /// `ε_c = sqrt(2ħC)/g₂`, `ε_d = sqrt(ħN₁)/g₁`,
    /// `ν = ħ g₃/sqrt(d(d²−1)/12 · k2)`, `ε_F = m/e²`, `τ = m ε_c²/(2ħ)`,
    /// `ρ = sqrt(2ħ) g₂/m`. The last two impose both weak proxies and the
    /// Gaussian-phase action scale at once. Below `d = 2` only `ε_d` and
    /// `ε_F` exist: `ρ` is obtained through `τ` and hence through `ε_c`.
    pub fn of(
        couplings: &TargetCouplings,
        d: usize,
        mass: f64,
        h_eff: f64,
        n1: f64,
        k2: f64,
    ) -> Self {
        let [e, g1, g2, g3] =
            [couplings.e_em, couplings.g1, couplings.g2, couplings.g3].map(|g| g[0]);
        let epsilon_c = (d >= 2).then(|| (2. * h_eff * casimir(2) / casimir(d)).sqrt() / g2);
        Self {
            epsilon_c,
            epsilon_d: (h_eff * n1).sqrt() / g1,
            viscosity: (d >= 2).then(|| h_eff * g3 / (kernel_factor(d) * k2).sqrt()),
            fitness_scale: mass / (e * e),
            time_step: epsilon_c.map(|c| mass * c * c / (2. * h_eff)),
            viscous_range: epsilon_c.map(|_| (2. * h_eff).sqrt() * g2 / mass),
        }
    }
}

/// A defined proxy, or the list of what it lacks after its formula.
fn stated(formula: &str, needs: &[(&str, bool)]) -> String {
    let missing: Vec<&str> = needs
        .iter()
        .filter(|(_, present)| !present)
        .map(|(n, _)| *n)
        .collect();
    if missing.is_empty() {
        formula.into()
    } else {
        format!("{formula}; undefined here: no {}", missing.join(", "))
    }
}
fn quantity(
    name: &str,
    symbol: &str,
    value: Option<f64>,
    unit: &str,
    definition: impl Into<String>,
    book_label: &str,
) -> Quantity {
    let finite = value.filter(|v| v.is_finite());
    let mut definition: String = definition.into();
    if finite != value {
        definition.push_str("; undefined here: the formula is not finite on these scales");
    }
    Quantity {
        name: name.into(),
        symbol: symbol.into(),
        value: finite,
        error: None,
        unit: unit.into(),
        definition,
        book_label: book_label.into(),
    }
}
fn kernel_name(kernel: &Kernel) -> &'static str {
    match kernel {
        Kernel::Uniform => "uniform",
        Kernel::Gaussian { .. } => "Gaussian",
        Kernel::Exponential { .. } => "exponential",
    }
}

/// The recorded metric of the tessellation, by estimator, and what it is made
/// of. Every geodesic length of the run — a `WarmupEdgeMean` colour length, a
/// multiscale gate, a smearing width — is a distance of this metric, so the
/// report names it instead of calling it a fitness-manifold distance: only the
/// Hessian arm reads a scalar field at all.
fn metric_name(metric: &MetricKind) -> (String, &'static str) {
    let density = "a density object built from the neighbour geometry alone: it reads no \
                   fitness, and its lengths are not distances of the fitness manifold of \
                   def-adaptive-diffusion-tensor-latent";
    match metric {
        MetricKind::NeighborCovariance { .. } => {
            ("the inverse neighbour covariance".into(), density)
        }
        MetricKind::VoronoiCovariance { .. } => {
            ("the inverse Voronoi vertex covariance".into(), density)
        }
        MetricKind::Identity => (
            "the identity".into(),
            "flat space, with no scale of its own",
        ),
        MetricKind::ObservationField { field, .. } => (
            format!("the observation field `{field}`"),
            "whatever the run recorded under that name; it is a fitness manifold only if that \
             field is the fitness Hessian",
        ),
        MetricKind::HessianFd { scalar_field, .. } => (
            format!("a finite-difference Hessian of `{scalar_field}`"),
            "the fitness-manifold arm of def-adaptive-diffusion-tensor-latent when that scalar \
             field is the fitness",
        ),
    }
}

/// The range a measurement used, against the one the companion kernel gives.
fn used_range(
    symbol: &str,
    configured: Range,
    calibrated: Option<f64>,
    kernel: Option<f64>,
    notes: &mut Vec<String>,
) -> Option<f64> {
    let used = match configured {
        Range::Fixed { value } => Some(value),
        Range::FromKernel => calibrated,
    };
    if let Some(used) = used
        && kernel != Some(used)
    {
        let kernel = kernel.map_or("no Gaussian range".into(), |k| k.to_string());
        notes.push(format!(
            "The measurement used {symbol} = {used} while the companion kernel gives {kernel}."
        ));
    }
    used.or(kernel)
}

/// The numbers every row is built from: configuration, analysis parameters
/// and the warm-up calibration, which wins where it fixes a value.
struct Resolved {
    d: usize,
    mass: f64,
    /// Action scale of the U(1) phase, used by every proxy and the inversion.
    h: f64,
    color_h: f64,
    h_s: f64,
    dt: Option<f64>,
    friction: Option<f64>,
    temperature: Option<f64>,
    epsilon_d: Option<f64>,
    epsilon_c: Option<f64>,
    epsilon_clone: f64,
    /// Colour phase length `ℓ₀` fixed in the warm-up, and the arm that fixed
    /// it: `Calibration::{length, length_source}`.
    length: Option<f64>,
    length_source: String,
    nu: Option<f64>,
    /// Coefficient of the dense viscous force alone: the kernel proxies need
    /// its Gaussian kernel.
    dense_nu: Option<f64>,
    rho: Option<f64>,
    n1: Option<f64>,
    k2: Option<f64>,
}
impl Resolved {
    fn of(measurement: &Measurement, notes: &mut Vec<String>) -> Self {
        let gas = &measurement.gas;
        let d = measurement.capabilities.dimension;
        let configured = AlgorithmicScales::from_config(gas, d);
        let calibration = measurement.calibration.as_ref();
        let phase = &measurement.config.phase;
        let electroweak = &measurement.config.electroweak;
        if calibration.is_none() {
            notes.push(
                "No warm-up calibration is available: ranges are read from the configuration \
                 and the pair statistics N1 and <K_visc^2> are absent."
                    .into(),
            );
        }
        let h = calibration.map_or(electroweak.h_eff, |c| c.electroweak_h_eff);
        let dense = gas.qft.viscosity.as_ref();
        Self {
            d,
            mass: calibration.map_or(phase.mass, |c| c.mass),
            h,
            color_h: calibration.map_or(phase.h_eff, |c| c.h_eff),
            h_s: calibration.map_or(electroweak.h_s.unwrap_or(h), |c| c.h_s),
            dt: calibration.and_then(|c| c.dt).or(configured.dt),
            friction: configured.friction,
            temperature: configured.temperature,
            epsilon_d: used_range(
                "epsilon_d",
                electroweak.epsilon_d,
                calibration.and_then(|c| c.epsilon_d),
                configured.epsilon_d,
                notes,
            ),
            epsilon_c: used_range(
                "epsilon_c",
                electroweak.epsilon_c,
                calibration.and_then(|c| c.epsilon_c),
                configured.epsilon_c,
                notes,
            ),
            epsilon_clone: configured.epsilon_clone,
            length: calibration.map(|c| c.length),
            length_source: calibration.map_or(String::new(), |c| c.length_source.clone()),
            nu: configured.viscosity,
            dense_nu: dense.map(|v| v.coefficient).filter(|nu| *nu > 0.),
            rho: dense.map(|v| v.bandwidth),
            n1: calibration.and_then(|c| c.pair_weight_n1),
            k2: calibration.and_then(|c| c.viscous_kernel_second_moment),
        }
    }
    fn kernel_action_scale(&self) -> Option<f64> {
        let (c, tau) = self.epsilon_c.zip(self.dt)?;
        Some(kernel_action_scale(self.mass, c, tau))
    }
}

/// Notes on what the configuration leaves undefined or redefines.
fn configuration_notes(gas: &GasConfig, r: &Resolved, notes: &mut Vec<String>) {
    if r.color_h != r.h || r.h_s != r.h {
        notes.push(format!(
            "The measurement has three action scales: h_eff = {} of the U(1) phase, {} of the \
             colour phase and h_S = {} of the cloning phase. The book's proxies have one; the \
             first is used for every proxy and for the inversion.",
            r.h, r.color_h, r.h_s
        ));
    }
    for (role, symbol, module, value) in [
        ("distance", "epsilon_d", &gas.distance_donors, r.epsilon_d),
        ("cloning", "epsilon_c", &gas.cloning_donors, r.epsilon_c),
    ] {
        if value.is_some() {
            continue;
        }
        notes.push(match module.kernel {
            Kernel::Gaussian { .. } => format!(
                "The {role} companion distance is anisotropic or not spatial: {symbol} is \
                 undefined and every proxy depending on it is omitted."
            ),
            _ => format!(
                "The {role} companion kernel is {}, not Gaussian: {symbol} is undefined and \
                 every proxy depending on it is omitted.",
                kernel_name(&module.kernel)
            ),
        });
    }
    let squashed = |module: &DonorModule| {
        matches!(module.kernel, Kernel::Gaussian { .. })
            && matches!(module.distance, Distance::SquashedPhaseSpace { .. })
    };
    if squashed(&gas.distance_donors) || squashed(&gas.cloning_donors) {
        notes.push(SQUASH_NOTE.into());
    }
    if let Some(kernel) = r.kernel_action_scale() {
        notes.push(format!(
            "h_eff = {} is an analysis parameter. The Gaussian-phase convention of \
             thm-effective-planck-constant would give m epsilon_c^2/(2 tau) = {kernel}.",
            r.h
        ));
    }
    if let Some(dense) = gas.qft.viscosity.as_ref().filter(|_| r.dense_nu.is_some()) {
        let normaliser = if dense.row_normalized {
            "the row mass"
        } else {
            "the eligible population"
        };
        notes.push(format!(
            "The engine's dense viscous force is nu sum_j w_ij (v_j - v_i) divided by \
             {normaliser}; the book kernel is unnormalised. nu is reported as configured; only \
             the product nu^2 <K^2> is independent of that convention."
        ));
    } else if r.nu.is_some() {
        notes.push(GRAPH_NOTE.into());
    }
    if r.d != 3 {
        notes.push(format!(
            "The colour identification behind the strong proxy uses d = 3; at d = {} the proxy \
             g_d defines a different model.",
            r.d
        ));
    }
    if let Some(geometry) = gas.geometry.as_ref() {
        let (name, made_of) = metric_name(&geometry.pipeline.metric);
        let used = if r.length_source.starts_with("warmup_edge_mean") {
            format!(
                ", and l0 = {} is a mean of its edge lengths",
                fixed(r.length)
            )
        } else {
            String::new()
        };
        notes.push(format!(
            "The recorded tessellation metric is {name}: {made_of}. Every geodesic length of \
             this run is a distance of that metric{used}."
        ));
    }
}
/// A resolved number, or the statement that it is absent.
fn fixed(value: Option<f64>) -> String {
    value.map_or("absent".into(), |v| v.to_string())
}
fn scale_rows(gas: &GasConfig, r: &Resolved) -> Vec<Quantity> {
    let has = |x: Option<f64>| x.is_some();
    let which = match (r.dense_nu, r.nu) {
        (Some(_), _) => "dense Gaussian-kernel force",
        (None, Some(_)) => "graph force on tessellation edge weights",
        (None, None) => "no viscous coupling is configured",
    };
    let weight = if gas.cloning_donors.distance == gas.distance_donors.distance {
        "velocity weight of d^2 = |dx|^2 + lambda |dv|^2 in the companion distance"
    } else {
        "velocity weight of the distance-companion distance; the cloning role differs"
    };
    vec![
        quantity(
            "dimension",
            "d",
            Some(r.d as f64),
            "1",
            "position dimension of the run",
            "",
        ),
        quantity(
            "time_step",
            "τ",
            r.dt,
            "time",
            "integrator time step, when the kinetic operator has one",
            "",
        ),
        quantity("friction", "γ", r.friction, "1/time", "BAOAB friction", ""),
        quantity(
            "noise_scale",
            "σ_v",
            noise_scale(gas),
            "length/time^(3/2)",
            "sigma_v of dv = -gamma v dt + sigma_v dW; defined for an isotropic constant noise",
            "",
        ),
        quantity(
            "temperature",
            "T",
            r.temperature,
            "length^2/time^2",
            "thermostat target sigma_v^2/(2 gamma), the stationary velocity variance per \
             component of the O step; cloning, viscous forces and a velocity cap change the \
             velocity law, so it is not a measured temperature",
            "",
        ),
        quantity(
            "viscosity",
            "ν",
            r.nu,
            "1/time",
            format!("viscous coefficient as configured: {which}"),
            "def-fractal-set-viscous-force",
        ),
        quantity(
            "viscous_range",
            "ρ",
            r.rho,
            "length",
            "bandwidth of the dense viscous kernel exp(-|dx|^2/(2 rho^2)); graph viscosity has \
             none",
            "def-fractal-set-viscous-force",
        ),
        quantity(
            "epsilon_d",
            "ε_d",
            r.epsilon_d,
            "length",
            "range of the distance-companion kernel the measurement used: width times the \
             position scale of a Gaussian kernel, or a fixed value",
            PROXY_LABEL,
        ),
        quantity(
            "epsilon_c",
            "ε_c",
            r.epsilon_c,
            "length",
            "range of the cloning-companion kernel the measurement used",
            PROXY_LABEL,
        ),
        quantity(
            "velocity_weight",
            "λ_alg",
            velocity_weight(&gas.distance_donors.distance),
            "1",
            weight,
            "",
        ),
        quantity(
            "epsilon_clone",
            "ε_clone",
            Some(r.epsilon_clone),
            "1",
            "regulariser of the cloning score (V_c - V_i)/(V_i + epsilon_clone); never a range",
            "",
        ),
        quantity(
            "clone_saturation",
            "p_max",
            Some(gas.clone_decision.saturation),
            "1",
            "cloning probability is clip(score/p_max)",
            "",
        ),
        quantity(
            "clone_period",
            "",
            Some(gas.clone_decision.every as f64),
            "steps",
            "living walkers clone on steps divisible by this period",
            "",
        ),
        quantity(
            "position_diffusion",
            "",
            Some(gas.kinetic.position_diffusion),
            "length/time^(1/2)",
            "independent Brownian diffusion of positions after the integrator",
            "",
        ),
        quantity(
            "mass",
            "m",
            Some(r.mass),
            "mass",
            "mass of the colour phase, an analysis parameter",
            "",
        ),
        quantity(
            "phase_length",
            "ℓ₀",
            r.length,
            "length",
            stated(
                &format!(
                    "length of the colour phase kappa = m l0/h_eff, fixed in the warm-up by the \
                     arm `{}`; a geodesic arm measures it in the recorded tessellation metric, \
                     which is named in the notes",
                    r.length_source
                ),
                &[("warm-up calibration", r.length.is_some())],
            ),
            "",
        ),
        quantity(
            "action_scale",
            "ħ_eff",
            Some(r.h),
            "action",
            "action scale of the U(1) fitness phase, an analysis parameter; used by every proxy",
            "",
        ),
        quantity(
            "color_action_scale",
            "ħ_eff",
            Some(r.color_h),
            "action",
            "action scale of the colour phase kappa = m l0/h_eff, an analysis parameter",
            "",
        ),
        quantity(
            "clone_action_scale",
            "h_S",
            Some(r.h_s),
            "action",
            "action scale of the SU(2) cloning phase, an analysis parameter",
            "",
        ),
        quantity(
            "fitness_force_scale",
            "ε_F",
            None,
            "energy",
            "no fitness-force scale epsilon_F in this configuration",
            "thm-u1-coupling-constant",
        ),
        quantity(
            "kernel_action_scale",
            "ħ_kernel",
            r.kernel_action_scale(),
            "action",
            stated(
                "m epsilon_c^2/(2 tau), the action scale of the Gaussian-phase convention",
                &[("epsilon_c", has(r.epsilon_c)), ("time step", has(r.dt))],
            ),
            "thm-effective-planck-constant",
        ),
        quantity(
            "energy_clone",
            "E_c",
            r.epsilon_c.map(|c| r.h / c),
            "energy",
            stated(
                "h_eff/epsilon_c; a scale, not a particle mass",
                &[("epsilon_c", has(r.epsilon_c))],
            ),
            "thm-mass-scales",
        ),
        quantity(
            "energy_viscous",
            "E_ρ",
            r.rho.map(|rho| r.h / rho),
            "energy",
            stated(
                "h_eff/rho; a scale, not a particle mass",
                &[("rho", has(r.rho))],
            ),
            "thm-mass-scales",
        ),
        quantity(
            "energy_friction",
            "E_γ",
            r.friction.map(|gamma| r.h * gamma),
            "energy",
            stated(
                "h_eff gamma; a scale, not a particle mass",
                &[("friction", has(r.friction))],
            ),
            "thm-mass-scales",
        ),
        quantity(
            "separation",
            "σ_sep",
            r.epsilon_c.zip(r.rho).map(|(c, rho)| c / rho),
            "1",
            stated(
                "epsilon_c/rho",
                &[("epsilon_c", has(r.epsilon_c)), ("rho", has(r.rho))],
            ),
            "thm-dimensionless-ratios",
        ),
        quantity(
            "pair_statistic_n1",
            "N₁",
            r.n1,
            "1",
            "N1 = E exp(-D^2/epsilon_d^2), the squared companion kernel weight, uniform over the \
             valid first distance-companion pairs of the warm-up frames, D the distance the \
             measurement selected; absent without epsilon_d or a calibration",
            PROXY_LABEL,
        ),
        quantity(
            "kernel_second_moment",
            "⟨K²⟩",
            r.k2,
            "1",
            "E K_ij^2 of the unnormalised kernel K_ij = exp(-|x_i - x_j|^2/(2 rho^2)) <= 1, \
             uniform over the ordered eligible pairs i != j of the warm-up frames; the engine \
             divides K by the eligible count or the row sum, so this bounds the engine's moment \
             from above; dense viscosity only",
            PROXY_LABEL,
        ),
    ]
}
fn coupling_rows(r: &Resolved, notes: &mut Vec<String>) -> Vec<Quantity> {
    let has = |x: Option<f64>| x.is_some();
    let (d, h) = (r.d, r.h);
    let g1 = r.epsilon_d.zip(r.n1).map(|(e, n)| g1_squared(h, n, e));
    let g1_upper = r.epsilon_d.map(|e| g1_squared(h, 1., e));
    let g2_casimir = r.epsilon_c.and_then(|c| g2_casimir_squared(h, c, d));
    let g2_clock = match (r.dt, r.rho, r.epsilon_c) {
        (Some(tau), Some(rho), Some(c)) => Some(g2_clock_squared(r.mass, tau, rho, c)),
        _ => None,
    };
    let gd = r
        .dense_nu
        .zip(r.k2)
        .and_then(|(nu, k)| gd_squared(nu, h, d, k));
    let gd_upper = r.dense_nu.and_then(|nu| gd_squared(nu, h, d, 1.));
    let weak_ratio = g2_clock
        .zip(g2_casimir)
        .map(|(clock, casimir)| clock / casimir);
    let root = |squared: Option<f64>| squared.map(f64::sqrt);
    let (strong, suffix) = if d == 3 { ("g3", "3") } else { ("gd", "d") };
    let strong_needs = [("dense viscosity", has(r.dense_nu)), ("d >= 2", d >= 2)];
    let mut rows = vec![
        quantity(
            "g1",
            "g₁",
            root(g1),
            "1",
            stated(
                "g1^2 = h_eff N1/epsilon_d^2 with the warm-up pair statistic N1",
                &[("epsilon_d", has(r.epsilon_d)), ("N1", has(r.n1))],
            ),
            PROXY_LABEL,
        ),
        quantity(
            "g1_upper",
            "g₁",
            root(g1_upper),
            "1",
            stated(
                "upper bound sqrt(h_eff)/epsilon_d of g1, from N1 <= 1",
                &[("epsilon_d", has(r.epsilon_d))],
            ),
            "thm-sm-g1-coupling",
        ),
        quantity(
            "g2_casimir",
            "g₂",
            root(g2_casimir),
            "1",
            stated(
                "g2^2 = (2 h_eff/epsilon_c^2) C2(2)/C2(d), C2(n) = (n^2 - 1)/(2n); the Casimir \
                 ratio is a normalisation of the proxy",
                &[("epsilon_c", has(r.epsilon_c)), ("d >= 2", d >= 2)],
            ),
            "thm-sm-g2-coupling",
        ),
        quantity(
            "g2_clock",
            "ĝ₂",
            root(g2_clock),
            "1",
            stated(
                "g^2 = m tau rho^2/epsilon_c^2",
                &[
                    ("time step", has(r.dt)),
                    ("rho", has(r.rho)),
                    ("epsilon_c", has(r.epsilon_c)),
                ],
            ),
            CLOCK_LABEL,
        ),
        quantity(
            "g2_clock_over_casimir",
            "",
            weak_ratio,
            "1",
            stated(
                "ratio of the squared weak proxies, m tau rho^2 C2(d)/(2 h_eff C2(2))",
                &[("g2_clock", has(g2_clock)), ("g2_casimir", has(g2_casimir))],
            ),
            "prop-ym-weak-proxy-comparison",
        ),
        quantity(
            strong,
            "g_d",
            root(gd),
            "1",
            stated(
                "g_d^2 = (nu^2/h_eff^2) d(d^2 - 1)/12 <K^2> with nu as configured and the \
                 warm-up moment of the unnormalised kernel, which bounds the engine's \
                 normalised kernel from above; the factor d(d^2 - 1)/12 is an assigned \
                 normalisation",
                &[strong_needs[0], strong_needs[1], ("<K^2>", has(r.k2))],
            ),
            "thm-sm-g3-coupling",
        ),
        quantity(
            &format!("{strong}_upper"),
            "g_d",
            root(gd_upper),
            "1",
            stated(
                "upper bound (nu/h_eff) sqrt(d(d^2 - 1)/12) from K <= 1; loose by the eligible \
                 population size under the engine's population normalisation, which no \
                 configuration fixes",
                &strong_needs,
            ),
            "thm-sm-g3-coupling",
        ),
        quantity(
            "e_fitness",
            "ê",
            None,
            "1",
            "e^2 = m/epsilon_F; undefined here: no fitness-force scale epsilon_F",
            "thm-u1-coupling-constant",
        ),
    ];
    for (name, squared, of) in [
        (
            "alpha_1".into(),
            g1,
            "g1; a hypercharge-type proxy, not alpha_em",
        ),
        (
            "alpha_1_upper".into(),
            g1_upper,
            "g1_upper; a hypercharge-type proxy, not alpha_em",
        ),
        ("alpha_2_casimir".into(), g2_casimir, "g2_casimir"),
        ("alpha_2_clock".into(), g2_clock, "g2_clock"),
        (format!("alpha_{suffix}"), gd, "the strong proxy"),
        (
            format!("alpha_{suffix}_upper"),
            gd_upper,
            "the upper bound of the strong proxy",
        ),
        ("alpha_fitness".into(), None, "e_fitness"),
    ] {
        rows.push(quantity(
            &name,
            "α̂",
            squared.map(|g| g / (4. * PI)),
            "1",
            stated(
                &format!("g^2/(4 pi) of {of}"),
                &[("that proxy", has(squared))],
            ),
            "def-fine-structure-constant-ym",
        ));
    }
    rows.push(quantity(
        "sin2_theta_proxy_upper",
        "",
        g1_upper.zip(g2_casimir).map(|(a, b)| a / (a + b)),
        "1",
        stated(
            "derived here from the target convention applied to the range proxies: \
             g1^2/(g1^2 + g2^2) with g1_upper and g2_casimir, a function of epsilon_d/epsilon_c \
             and d alone that carries no information about the run",
            &[("g1_upper", has(g1_upper)), ("g2_casimir", has(g2_casimir))],
        ),
        TARGET_LABEL,
    ));
    if g1_upper.is_some() || gd_upper.is_some() {
        notes.push(format!(
            "g1_upper and {strong}_upper are upper bounds: the pair statistics N1 and <K_visc^2> \
             are at most 1 (thm-sm-g1-coupling; Gaussian kernel weights are at most 1), and the \
             bounds set them to 1."
        ));
    }
    if let Some(ratio) = weak_ratio {
        notes.push(format!(
            "Two weak proxies are reported, the Casimir-range proxy (def-sm-coupling-definition) \
             and the clock proxy (thm-su2-coupling-constant). They are different functions of \
             the parameters and coincide exactly when m tau rho^2 C2(d) = 2 h_eff C2(2); gauge \
             symmetry does not impose this relation (def-qft-report-dictionary, \
             prop-ym-weak-proxy-comparison). Their ratio clock/Casimir here is {ratio}."
        ));
    }
    rows
}

/// The moment a configured viscosity presupposes, against the one the warm-up
/// measured. `ν` is only meaningful together with the `⟨K_visc²⟩` of the
/// kernel that will actually run (`prop-qft-report-inversion`): a pair
/// `(ν, ρ)` carried over from an inversion made at a different moment realises
/// a different `g_d`, by the ratio of the two moments. The check states both
/// numbers and their factor, and flags a disagreement beyond
/// `CALIBRATION_TOLERANCE`; it never fails the report.
fn viscosity_consistency(
    r: &Resolved,
    inputs: &StandardModelInputs,
    notes: &mut Vec<String>,
) -> Result<()> {
    let targets = TargetCouplings::of(inputs)?;
    let Some(nu) = r.dense_nu else {
        return Ok(());
    };
    let Some(presupposed) = presupposed_kernel_second_moment(nu, r.h, r.d, targets.g3[0]) else {
        return Ok(());
    };
    let strong = if r.d == 3 { "alpha_3" } else { "alpha_d" };
    let target = inputs.alpha_s[0];
    let Some(k2) = r.k2 else {
        notes.push(format!(
            "Calibration check: the configured viscosity nu = {nu} realises the input alpha_s = \
             {target} only if the viscous kernel has <K_visc^2> = {presupposed}. This run has no \
             warm-up measurement of <K_visc^2>, so nothing checks that presupposition and \
             {strong} is undefined rather than assumed."
        ));
        return Ok(());
    };
    let Some(realised) = gd_squared(nu, r.h, r.d, k2).map(|g| g / (4. * PI)) else {
        return Ok(());
    };
    let factor = presupposed / k2;
    notes.push(format!(
        "Calibration check: the configured viscosity nu = {nu} presupposes <K_visc^2> = \
         {presupposed} to realise the input alpha_s = {target}, while the warm-up kernel of \
         bandwidth rho = {} measured <K_visc^2> = {k2}, a factor {factor}. The realised {strong} \
         is {realised}.",
        fixed(r.rho)
    ));
    if !(1. / CALIBRATION_TOLERANCE..=CALIBRATION_TOLERANCE).contains(&factor) {
        notes.push(format!(
            "The viscous calibration of this run is inconsistent: {strong} = {realised} is not \
             the input alpha_s = {target}. A value of nu is meaningful only together with the \
             <K_visc^2> of the kernel that runs (prop-qft-report-inversion), so a (nu, rho) pair \
             carried over from an inversion made at another moment describes another coupling. \
             Iterate the inversion at the measured moment before reading the strong sector of \
             this run."
        ));
    }
    Ok(())
}

/// Inputs first, then the targets the dictionary assigns to them. Every
/// definition opens with `input` or `target`.
fn inversion_rows(r: &Resolved, inputs: &StandardModelInputs) -> Result<Vec<Quantity>> {
    let couplings = TargetCouplings::of(inputs)?;
    let (n1, k2) = (r.n1.unwrap_or(1.), r.k2.unwrap_or(1.));
    let solved = CalibrationTargets::of(&couplings, r.d, r.mass, r.h, n1, k2);
    let input = |name: &str, symbol: &str, pair: [f64; 2], definition: String| Quantity {
        error: Some(pair[1]).filter(|e| e.is_finite()),
        ..quantity(name, symbol, Some(pair[0]), "1", definition, TARGET_LABEL)
    };
    let (scale, source) = (&inputs.scale, &inputs.source);
    let [inverse, inverse_error] = inputs.alpha_em_inverse;
    let mut rows = vec![
        input(
            "alpha_em",
            "α_em",
            [1. / inverse, inverse_error / (inverse * inverse)],
            format!("input: electromagnetic coupling at {scale} ({source}), from its inverse"),
        ),
        input(
            "sin2_theta_w",
            "sin²θ_W",
            inputs.sin2_theta_w,
            format!("input: weak mixing angle at {scale} ({source})"),
        ),
        input(
            "alpha_s",
            "α_s",
            inputs.alpha_s,
            format!("input: strong coupling at {scale} ({source})"),
        ),
        input(
            "e_em",
            "e",
            couplings.e_em,
            "input: e = sqrt(4 pi alpha_em)".into(),
        ),
        input(
            "g1",
            "g₁",
            couplings.g1,
            "input: g1 = e/cos(theta_W), the hypercharge coupling gY".into(),
        ),
        input(
            "g2",
            "g₂",
            couplings.g2,
            "input: g2 = e/sin(theta_W)".into(),
        ),
        input(
            "g3",
            "g₃",
            couplings.g3,
            "input: g3 = sqrt(4 pi alpha_s)".into(),
        ),
    ];
    if inputs.hypercharge == HyperchargeNormalization::Unified {
        let unified = (5f64 / 3.).sqrt();
        rows.push(input(
            "g1_unified",
            "g₁",
            couplings.g1.map(|g| unified * g),
            "input: sqrt(5/3) gY, the unified normalisation, quoted only; the inversion uses gY"
                .into(),
        ));
    }
    let held = |measured: Option<f64>, symbol: &str, bound: &str| match measured {
        Some(_) => format!("input: {symbol} of the warm-up frames, held fixed by the inversion"),
        None => format!("input: placeholder 1 for {symbol}, which was not measured; {bound}"),
    };
    let bound = |measured: Option<f64>, text: &str| match measured {
        Some(_) => String::new(),
        None => format!("; {text}"),
    };
    let target = |name: &str, symbol: &str, value: Option<f64>, unit: &str, definition: String| {
        let needs = [("d >= 2", value.is_some())];
        quantity(
            name,
            symbol,
            value,
            unit,
            stated(&definition, &needs),
            INVERSION_LABEL,
        )
    };
    rows.extend([
        quantity(
            "dimension",
            "d",
            Some(r.d as f64),
            "1",
            "input: position dimension of the run",
            "",
        ),
        quantity(
            "mass",
            "m",
            Some(r.mass),
            "mass",
            "input: mass, a free choice of unit that rescales only target_fitness_scale, \
             target_time_step and target_viscous_range",
            "",
        ),
        quantity(
            "action_scale",
            "ħ_eff",
            Some(r.h),
            "action",
            "input: action scale of the U(1) fitness phase, an analysis parameter",
            "",
        ),
        quantity(
            "pair_statistic_n1",
            "N₁",
            Some(n1),
            "1",
            held(r.n1, "N1", "N1 <= 1 makes target_epsilon_d an upper bound"),
            PROXY_LABEL,
        ),
        quantity(
            "kernel_second_moment",
            "⟨K²⟩",
            Some(k2),
            "1",
            held(
                r.k2,
                "<K_visc^2>",
                "K <= 1 makes target_viscosity a lower bound",
            ),
            PROXY_LABEL,
        ),
        target(
            "target_epsilon_c",
            "ε_c",
            solved.epsilon_c,
            "length",
            "target: sqrt(2 h_eff C2(2)/C2(d))/g2".into(),
        ),
        target(
            "target_epsilon_d",
            "ε_d",
            Some(solved.epsilon_d),
            "length",
            format!("target: sqrt(h_eff N1)/g1{}", bound(r.n1, "an upper bound")),
        ),
        target(
            "target_viscosity",
            "ν",
            solved.viscosity,
            "1/time",
            format!(
                "target: h_eff g3/sqrt(d(d^2 - 1)/12 <K^2>) for the unnormalised kernel{}",
                bound(r.k2, "a lower bound")
            ),
        ),
        target(
            "target_fitness_scale",
            "ε_F",
            Some(solved.fitness_scale),
            "energy",
            "target: m/e^2; the configuration has no fitness-force scale".into(),
        ),
        target(
            "target_time_step",
            "τ",
            solved.time_step,
            "time",
            "target: m epsilon_c^2/(2 h_eff), imposing both weak proxies and the Gaussian-phase \
             action scale"
                .into(),
        ),
        target(
            "target_viscous_range",
            "ρ",
            solved.viscous_range,
            "length",
            "target: sqrt(2 h_eff) g2/m, imposing both weak proxies and the Gaussian-phase \
             action scale"
                .into(),
        ),
    ]);
    Ok(rows)
}

/// Scales and couplings of one measurement: `gas`, `capabilities.dimension`,
/// `config` (`phase.mass`, `electroweak`) and `calibration`. The couplings
/// `g₁² = ħ_eff N₁ / ε_d²` and `g₃² = (ν²/ħ_eff²) d(d²−1)/12 · E_μ K_visc²`
/// read the measured pair statistics
/// `Calibration::{pair_weight_n1, viscous_kernel_second_moment}` and have
/// `value: None` where those are absent. Every row of `inversion` is an input
/// or a target of the dictionary applied to `inputs`; none compares a gas
/// number with a reference value. Missing ingredients give `value: None`,
/// never an error.
///
/// The book's proxies have one action scale; the electroweak one is used for
/// all of them and a note states the colour and cloning scales when they
/// differ. The inversion holds the pair statistics fixed at their warm-up
/// values, or at 1 where none was measured, which turns the affected target
/// into a bound.
pub fn report(measurement: &Measurement, inputs: &StandardModelInputs) -> Result<CouplingReport> {
    inputs.validate()?;
    measurement.validate()?;
    let gas = &measurement.gas;
    let mut conditional = vec![];
    let r = Resolved::of(measurement, &mut conditional);
    configuration_notes(gas, &r, &mut conditional);
    viscosity_consistency(&r, inputs, &mut conditional)?;
    let couplings = coupling_rows(&r, &mut conditional);
    let fixed = |measured: Option<f64>, consequence: &str| match measured {
        Some(value) => format!("{value}, its warm-up value"),
        None => format!("the placeholder value 1, so {consequence}"),
    };
    let notes = [
        KIND_NOTE.into(),
        MATCHING_NOTE.into(),
        format!(
            "Inversion: Standard Model inputs ({}; g1 = gY) are mapped to target gas parameters \
             by prop-qft-report-inversion. Targets are inputs of a calibration, not results of \
             this run; agreement between configured and target parameters is a choice of \
             configuration, not evidence.",
            inputs.scale
        ),
        format!(
            "The inversion holds the pair statistics fixed: N1 at {} and <K_visc^2> at {}. A \
             new run changes these statistics; a calibration must be iterated and its \
             convergence checked (sec-qft-calibration-qsd).",
            fixed(r.n1, "target_epsilon_d is an upper bound"),
            fixed(r.k2, "target_viscosity is a lower bound")
        ),
    ];
    Ok(CouplingReport {
        scales: scale_rows(gas, &r),
        couplings,
        inversion: inversion_rows(&r, inputs)?,
        notes: notes.into_iter().chain(conditional).collect(),
    })
}
