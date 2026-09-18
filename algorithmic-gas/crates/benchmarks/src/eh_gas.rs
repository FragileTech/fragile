//! Native run of the Einstein-Hilbert gas and its history export.
//!
//! The history holds, per recorded step, what the QFT analyzers read from a
//! run: walker state, the Einstein-Hilbert potential U = -r, the curvature and
//! volume fields, clone events, and the tessellation graph with its edge
//! arrays. `tools/eh_archive_to_history.py` maps it onto the analyzers' field
//! names.
use algorithmic_gas::{
    AlgorithmicGas, GasBuilder, GasError, ObservationBatch, Population, Precision, Real, Result,
    StepReport, TensorBatch,
    random::{RandomStream, Stream},
    tessellation::{
        EinsteinHilbertGas, ZeroPotential,
        stage::{DIFFUSION_FIELD, VOLUME_FIELD, curvature_field},
    },
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EhRunConfig {
    pub walkers: usize,
    pub dimensions: usize,
    pub steps: u64,
    pub seed: u64,
    pub precision: Precision,
    /// Standard deviations of the Gaussian initial positions and velocities;
    /// zero starts every walker at the origin, at rest.
    pub init_spread: f64,
    pub init_velocity: f64,
    /// Keep one history frame every this many steps (the last step is kept).
    pub record_every: u64,
    /// Store the neighbor graph and edge arrays in recorded frames.
    pub record_graph: bool,
    pub gas: EinsteinHilbertGas,
}
impl Default for EhRunConfig {
    fn default() -> Self {
        Self {
            walkers: 500,
            dimensions: 3,
            steps: 750,
            seed: 7,
            precision: Precision::F32,
            init_spread: 0.,
            init_velocity: 0.,
            record_every: 1,
            record_graph: true,
            gas: EinsteinHilbertGas::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EhFrame {
    pub step: u64,
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    /// U = -reward after the step (post-cloning geometry).
    pub potential: Vec<f64>,
    pub ricci_scalar: Vec<f64>,
    pub volume_element: Vec<f64>,
    /// `[walkers, d, d]` diffusion factors g^{-1/2} in the ambient space.
    pub diffusion: Vec<f64>,
    pub fitness: Vec<f64>,
    /// Standardized reward and companion-distance channels of the fitness.
    pub z_rewards: Vec<f64>,
    pub z_distances: Vec<f64>,
    pub will_clone: Vec<bool>,
    /// Cloning companion slot of every walker (itself when unmatched).
    pub companions_clone: Vec<u32>,
    pub companions_distance: Vec<u32>,
    /// Directed `[E, 2]` neighbor edges and their per-edge arrays.
    pub neighbor_edges: Vec<[u32; 2]>,
    pub geodesic_edge_distances: Vec<f64>,
    pub edge_weights: BTreeMap<String, Vec<f64>>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EhSummary {
    pub steps: u64,
    pub walkers: usize,
    pub clones: usize,
    /// Einstein-Hilbert action sum_i R_i sqrt(det g_i) of the final state.
    pub action: f64,
    pub mean_ricci: f64,
    pub mean_volume: f64,
    pub neighbor_edges: usize,
    pub mean_speed_squared: f64,
    pub seconds: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EhHistory {
    pub schema: String,
    pub config: EhRunConfig,
    pub summary: EhSummary,
    /// State before the first step, `[walkers, d]` row-major.
    pub initial_positions: Vec<f64>,
    pub initial_velocities: Vec<f64>,
    pub frames: Vec<EhFrame>,
}

fn initial_population<T: Real>(config: &EhRunConfig) -> Result<Population<T>> {
    let (n, d) = (config.walkers, config.dimensions);
    let sample = |substep: u64, scale: f64| -> Vec<T> {
        (0..n)
            .flat_map(|i| {
                let mut rng =
                    RandomStream::new(config.seed, 0, Stream::Initialize, i as u64, substep);
                (0..d)
                    .map(|_| {
                        if scale > 0. {
                            T::from_f64(scale) * rng.gaussian::<T>()
                        } else {
                            T::ZERO
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let mut observations =
        ObservationBatch::positions(TensorBatch::vectors(n, d, sample(0, config.init_spread))?);
    observations.fields.insert(
        "velocities".into(),
        TensorBatch::vectors(n, d, sample(1, config.init_velocity))?,
    );
    Population::new(observations)
}
pub async fn build<T: Real>(config: &EhRunConfig) -> Result<AlgorithmicGas<T>> {
    if config.walkers < 2 || !(2..=4).contains(&config.dimensions) || config.record_every == 0 {
        return Err(GasError::Configuration(
            "Einstein-Hilbert run needs at least two walkers, 2..=4 dimensions and a positive record period".into(),
        ));
    }
    if T::PRECISION != config.precision {
        return Err(GasError::Configuration(
            "run precision differs from the requested dtype".into(),
        ));
    }
    GasBuilder::new(initial_population::<T>(config)?, config.gas.reward())
        .gradient(ZeroPotential::new("velocities"))
        .config(config.gas.config(config.precision, config.seed))
        .build()
        .await
}
fn column<T: Real>(gas: &AlgorithmicGas<T>, name: &str) -> Result<Vec<f64>> {
    Ok(gas
        .population()
        .observations
        .field(name)?
        .values()
        .iter()
        .map(|v| v.to_f64())
        .collect())
}
fn frame<T: Real>(
    gas: &AlgorithmicGas<T>,
    report: &StepReport<T>,
    record_graph: bool,
) -> Result<EhFrame> {
    let n = gas.population().len();
    let companion = |batch: &algorithmic_gas::donor::CompanionBatch,
                     sources: &[algorithmic_gas::donor::SourceRef]| {
        (0..n)
            .map(|i| {
                batch
                    .row(i)
                    .next()
                    .map_or(i as u32, |pool| sources[pool as usize].slot)
            })
            .collect::<Vec<u32>>()
    };
    let mut out = EhFrame {
        step: report.step,
        positions: column(gas, "positions")?,
        velocities: column(gas, "velocities")?,
        potential: report
            .final_rewards
            .raw
            .iter()
            .map(|r| -r.to_f64())
            .collect(),
        ricci_scalar: column(gas, &curvature_field("ricci_scalar"))?,
        volume_element: column(gas, VOLUME_FIELD)?,
        diffusion: column(gas, DIFFUSION_FIELD)?,
        fitness: report
            .pre_clone_fitness
            .fitness
            .iter()
            .map(|v| v.to_f64())
            .collect(),
        z_rewards: report
            .pre_clone_fitness
            .reward_z
            .iter()
            .map(|v| v.to_f64())
            .collect(),
        z_distances: report
            .pre_clone_fitness
            .diversity_z
            .iter()
            .map(|v| v.to_f64())
            .collect(),
        will_clone: report
            .clone_plan
            .choices
            .iter()
            .map(|c| c.accepted)
            .collect(),
        companions_clone: companion(&report.cloning_companions, &report.clone_plan.sources),
        companions_distance: companion(&report.distance_companions, &report.distance_sources),
        ..EhFrame::default()
    };
    if let Some(graph) = gas.graph().filter(|_| record_graph) {
        out.neighbor_edges = graph.graph.coo();
        out.geodesic_edge_distances = graph.geodesic_length.iter().map(|v| v.to_f64()).collect();
        out.edge_weights = graph
            .weights
            .iter()
            .map(|(k, w)| (k.clone(), w.iter().map(|v| v.to_f64()).collect()))
            .collect();
    }
    Ok(out)
}
pub async fn run<T: Real>(config: &EhRunConfig) -> Result<EhHistory> {
    let started = std::time::Instant::now();
    let mut gas = build::<T>(config).await?;
    let initial_positions = column(&gas, "positions")?;
    let initial_velocities = column(&gas, "velocities")?;
    let mut frames = Vec::new();
    let mut clones = 0;
    for step in 1..=config.steps {
        let report = gas.step().await?;
        clones += report.clones;
        if step % config.record_every == 0 || step == config.steps {
            frames.push(frame(&gas, &report, config.record_graph)?);
        }
    }
    let n = gas.population().len();
    let ricci = column(&gas, &curvature_field("ricci_scalar"))?;
    let volume = column(&gas, VOLUME_FIELD)?;
    let velocities = column(&gas, "velocities")?;
    let summary = EhSummary {
        steps: config.steps,
        walkers: n,
        clones,
        action: ricci.iter().zip(&volume).map(|(r, v)| r * v).sum(),
        mean_ricci: ricci.iter().sum::<f64>() / n as f64,
        mean_volume: volume.iter().sum::<f64>() / n as f64,
        neighbor_edges: gas.graph().map_or(0, |g| g.graph.edges()),
        mean_speed_squared: velocities.iter().map(|v| v * v).sum::<f64>() / n as f64,
        seconds: started.elapsed().as_secs_f64(),
    };
    Ok(EhHistory {
        schema: "einstein-hilbert-gas-history/v1".into(),
        config: config.clone(),
        summary,
        initial_positions,
        initial_velocities,
        frames,
    })
}
/// Run at the configured precision.
pub async fn run_configured(config: &EhRunConfig) -> Result<EhHistory> {
    match config.precision {
        Precision::F32 => run::<f32>(config).await,
        Precision::F64 => run::<f64>(config).await,
    }
}
