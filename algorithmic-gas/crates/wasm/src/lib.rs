//! Browser API. Run this module in a Web Worker. All compute initialization and
//! stepping are asynchronous; no C++ ABI or browser-main-thread state is used.
use algorithmic_gas::checkpoint::CHECKPOINT_VERSION;
use algorithmic_gas::{AlgorithmicGas, BackendKind, Checkpoint, Precision, Real};
use algorithmic_gas_benchmarks::RunConfig;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
mod physics_bindings;

/// Complete Volume II catalog; controls and native stress cases share this source.
#[wasm_bindgen]
pub fn lecture_catalog() -> Result<JsValue, JsValue> {
    js(&algorithmic_gas::lecture::catalog())
}

#[wasm_bindgen]
pub struct LectureExperiment {
    session: algorithmic_gas_benchmarks::lecture::LectureSession,
}
#[wasm_bindgen]
impl LectureExperiment {
    #[wasm_bindgen(js_name=create)]
    pub async fn create(request: String) -> Result<LectureExperiment, JsValue> {
        if request.len() > 1024 * 1024 {
            return Err(error("Lecture request too large"));
        }
        let request = serde_json::from_str(&request).map_err(error)?;
        Ok(Self {
            session: algorithmic_gas_benchmarks::lecture::LectureSession::create(request)
                .await
                .map_err(error)?,
        })
    }
    pub async fn advance(&mut self, steps: u32) -> Result<JsValue, JsValue> {
        js(&self.session.advance(steps as usize).await.map_err(error)?)
    }
    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        js(&self.session.snapshot().map_err(error)?)
    }
    pub fn evidence(&self) -> Result<JsValue, JsValue> {
        js(&self.session.evidence())
    }
    pub fn checkpoint(&self) -> Result<Vec<u8>, JsValue> {
        self.session.checkpoint().map_err(error)
    }
    #[wasm_bindgen(js_name=restore)]
    pub async fn restore(bytes: Vec<u8>) -> Result<LectureExperiment, JsValue> {
        Ok(Self {
            session: algorithmic_gas_benchmarks::lecture::LectureSession::restore(&bytes)
                .await
                .map_err(error)?,
        })
    }
}

/// Recompute a lecture result from matching configurations and complete run archives.
#[wasm_bindgen]
pub async fn lecture_analyze(evidence: String) -> Result<JsValue, JsValue> {
    if evidence.len() > 256 * 1024 * 1024 {
        return Err(error("Lecture evidence too large"));
    }
    let evidence = serde_json::from_str(&evidence).map_err(error)?;
    js(
        &algorithmic_gas_benchmarks::lecture::analyze_evidence(&evidence)
            .await
            .map_err(error)?,
    )
}

enum Run {
    F32(AlgorithmicGas<f32>),
    F64(AlgorithmicGas<f64>),
}
#[derive(Serialize, Deserialize)]
enum SavedRun {
    F32(Checkpoint<f32>),
    F64(Checkpoint<f64>),
}
#[derive(Serialize, Deserialize)]
struct BrowserCheckpoint {
    version: u32,
    config: RunConfig,
    state: SavedRun,
}
fn decode_checkpoint(bytes: &[u8]) -> Result<BrowserCheckpoint, JsValue> {
    if bytes.len() > 256 * 1024 * 1024 {
        return Err(error("checkpoint exceeds 256 MiB decode limit"));
    }
    let mut remaining = bytes;
    let saved: BrowserCheckpoint =
        ciborium::de::from_reader_with_recursion_limit(&mut remaining, 64).map_err(error)?;
    if !remaining.is_empty() || saved.version != CHECKPOINT_VERSION {
        return Err(error(
            "unsupported browser checkpoint version or trailing data",
        ));
    }
    saved.config.validate().map_err(error)?;
    match &saved.state {
        SavedRun::F32(s) => validate_saved(s, &saved.config)?,
        SavedRun::F64(s) => validate_saved(s, &saved.config)?,
    }
    Ok(saved)
}
fn validate_saved<T: Real>(state: &Checkpoint<T>, config: &RunConfig) -> Result<(), JsValue> {
    state.validate().map_err(error)?;
    if state.config != config.gas
        || state.population.len() != config.walkers
        || state
            .population
            .observations
            .field("positions")
            .map_err(error)?
            .item_shape()
            != [config.dimensions]
    {
        return Err(error(
            "browser configuration differs from checkpoint population",
        ));
    }
    Ok(())
}
#[wasm_bindgen]
pub struct BrowserGas {
    config: RunConfig,
    run: Run,
}
fn error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
fn js(value: &impl Serialize) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(error)
}

#[wasm_bindgen]
pub fn default_config() -> String {
    serde_json::to_string(&RunConfig::default()).unwrap()
}
/// Bounded native geometry queries. Run in a Worker so rendering stays responsive.
#[wasm_bindgen]
pub fn partv_geometry(request_json: String) -> Result<JsValue, JsValue> {
    if request_json.len() > 8 * 1024 * 1024 {
        return Err(error("geometry request exceeds 8 MiB"));
    }
    let request = serde_json::from_str(&request_json).map_err(error)?;
    js(&algorithmic_gas::partv_geometry::analyze(request).map_err(error)?)
}
/// Independent complete-engine continuations for native source/Noether experiments.
#[wasm_bindgen]
pub async fn partvi_run(request_json: String, config_json: String) -> Result<JsValue, JsValue> {
    if request_json.len() > 8 * 1024 * 1024 || config_json.len() > 8 * 1024 * 1024 {
        return Err(error("Part VI run request exceeds decode budget"));
    }
    let request = serde_json::from_str(&request_json).map_err(error)?;
    let config = serde_json::from_str(&config_json).map_err(error)?;
    js(
        &algorithmic_gas_benchmarks::qft_experiments::run(&config, &request)
            .await
            .map_err(error)?,
    )
}

/// Analyze an imported, validated f64 trajectory with the same native kernels.
#[wasm_bindgen]
pub fn partvi_archive(request_json: String, archive_json: String) -> Result<JsValue, JsValue> {
    if request_json.len() > 8 * 1024 * 1024 || archive_json.len() > 128 * 1024 * 1024 {
        return Err(error("Part VI archive/request exceeds decode budget"));
    }
    let request = serde_json::from_str(&request_json).map_err(error)?;
    let archive = algorithmic_gas::RunArchive::<f64>::from_json(&archive_json).map_err(error)?;
    js(
        &algorithmic_gas::physics::partvi::analyze_archive(&request, Some(&archive))
            .map_err(error)?,
    )
}
/// Validate a standalone archive before exposing its topology to the viewer.
#[wasm_bindgen]
pub fn partv_archive(archive_json: String) -> Result<JsValue, JsValue> {
    if archive_json.len() > 128 * 1024 * 1024 {
        return Err(error("archive JSON exceeds 128 MiB"));
    }
    let archive = algorithmic_gas::RunArchive::<f64>::from_json(&archive_json).map_err(error)?;
    archive.validate().map_err(error)?;
    js(
        &serde_json::json!({"graph":archive.graph(),"orders":archive.compare_orders(1.,0.04,200).map_err(error)?}),
    )
}
#[wasm_bindgen]
pub fn checkpoint_config(bytes: Vec<u8>) -> Result<String, JsValue> {
    let saved = decode_checkpoint(&bytes)?;
    serde_json::to_string(&saved.config).map_err(error)
}
#[wasm_bindgen]
pub fn capabilities() -> JsValue {
    js(&serde_json::json!({"wasm_cpu":["f32","f64"],"webgpu_compiled":cfg!(feature="webgpu"),"webgpu_precision":["f32"],"execution_model":"host-orchestrated Burn batches","rng_version":algorithmic_gas::random::RNG_VERSION,"checkpoint_version":CHECKPOINT_VERSION})).unwrap()
}

#[wasm_bindgen]
impl BrowserGas {
    #[wasm_bindgen(js_name=create)]
    pub async fn create(config_json: String) -> Result<BrowserGas, JsValue> {
        let config: RunConfig = serde_json::from_str(&config_json).map_err(error)?;
        if config.gas.backend == BackendKind::Cuda {
            return Err(error("CUDA is native-only; choose WASM CPU or WebGPU"));
        }
        let run = match config.gas.precision {
            Precision::F32 => Run::F32(config.build().await.map_err(error)?),
            Precision::F64 => Run::F64(config.build().await.map_err(error)?),
        };
        Ok(Self { config, run })
    }
    /// Bounded work only. A worker should yield to its event loop between calls
    /// so pause, cancellation, and configuration requests remain responsive.
    pub async fn step(&mut self, count: u32) -> Result<JsValue, JsValue> {
        if count == 0 || count > 16 {
            return Err(error("step batch must contain 1..=16 iterations"));
        }
        for _ in 0..count {
            match &mut self.run {
                Run::F32(g) => {
                    g.step().await.map_err(error)?;
                }
                Run::F64(g) => {
                    g.step().await.map_err(error)?;
                }
            }
        }
        self.snapshot()
    }
    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        match &self.run {
            Run::F32(g) => snapshot(g),
            Run::F64(g) => snapshot(g),
        }
    }
    /// Replace only numerical coordinates/status, keeping the configured shape.
    pub async fn set_population(&mut self, fixture_json: String) -> Result<JsValue, JsValue> {
        if fixture_json.len() > 4 * 1024 * 1024 {
            return Err(error("fixture exceeds 4 MiB"));
        }
        let fixture: PopulationFixture = serde_json::from_str(&fixture_json).map_err(error)?;
        match &mut self.run {
            Run::F32(g) => set_population(g, &fixture, &self.config).await?,
            Run::F64(g) => set_population(g, &fixture, &self.config).await?,
        }
        self.snapshot()
    }
    pub fn set_trace(&mut self, enabled: bool) -> Result<(), JsValue> {
        match &mut self.run {
            Run::F32(g) => g.set_trace(enabled),
            Run::F64(g) => g.set_trace(enabled),
        }
        .map_err(error)
    }
    pub fn start_recording(&mut self, config_json: String) -> Result<(), JsValue> {
        let config = serde_json::from_str(&config_json).map_err(error)?;
        match &mut self.run {
            Run::F32(g) => g.start_recording(config),
            Run::F64(g) => g.start_recording(config),
        }
        .map_err(error)
    }
    pub fn archive(&self) -> Result<JsValue, JsValue> {
        match &self.run {
            Run::F32(g) => js(g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?),
            Run::F64(g) => js(g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?),
        }
    }
    /// Analyze the current lossless record without advancing or drawing randomness.
    pub fn qft(&self, request_json: String) -> Result<JsValue, JsValue> {
        if request_json.len() > 8 * 1024 * 1024 {
            return Err(error("Part VI request exceeds 8 MiB"));
        }
        let request = serde_json::from_str(&request_json).map_err(error)?;
        match &self.run {
            Run::F64(g) => js(&algorithmic_gas::physics::partvi::analyze_archive(
                &request,
                g.recording(),
            )
            .map_err(error)?),
            Run::F32(_) => Err(error("QFT comparisons require an f64 recorded run")),
        }
    }
    pub fn fractal_set(&self) -> Result<JsValue, JsValue> {
        match &self.run {
            Run::F32(g) => js(&g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?
                .graph()),
            Run::F64(g) => js(&g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?
                .graph()),
        }
    }
    pub fn reconstruct(&self, epoch: u32, step: u32, stage: String) -> Result<JsValue, JsValue> {
        match &self.run {
            Run::F32(g) => js(&g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?
                .reconstruct(epoch as u64, step as u64, &stage)
                .map_err(error)?),
            Run::F64(g) => js(&g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?
                .reconstruct(epoch as u64, step as u64, &stage)
                .map_err(error)?),
        }
    }
    pub fn compare_orders(
        &self,
        speed: f64,
        dt: f64,
        max_nodes: usize,
    ) -> Result<JsValue, JsValue> {
        match &self.run {
            Run::F32(g) => js(&g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?
                .compare_orders(speed, dt, max_nodes)
                .map_err(error)?),
            Run::F64(g) => js(&g
                .recording()
                .ok_or_else(|| error("recording is not enabled"))?
                .compare_orders(speed, dt, max_nodes)
                .map_err(error)?),
        }
    }
    pub fn config_json(&self) -> String {
        serde_json::to_string_pretty(&self.config).unwrap()
    }
    pub fn checkpoint(&self) -> Result<Vec<u8>, JsValue> {
        let state = match &self.run {
            Run::F32(g) => SavedRun::F32(g.checkpoint()),
            Run::F64(g) => SavedRun::F64(g.checkpoint()),
        };
        let mut bytes = vec![];
        ciborium::ser::into_writer(
            &BrowserCheckpoint {
                version: CHECKPOINT_VERSION,
                config: self.config.clone(),
                state,
            },
            &mut bytes,
        )
        .map_err(error)?;
        Ok(bytes)
    }
    #[wasm_bindgen(js_name=restore)]
    pub async fn restore(bytes: Vec<u8>) -> Result<BrowserGas, JsValue> {
        let saved = decode_checkpoint(&bytes)?;
        let mut gas = Self::create(serde_json::to_string(&saved.config).map_err(error)?).await?;
        match (&mut gas.run, saved.state) {
            (Run::F32(g), SavedRun::F32(s)) => g.restore(s).map_err(error)?,
            (Run::F64(g), SavedRun::F64(s)) => g.restore(s).map_err(error)?,
            _ => return Err(error("checkpoint precision mismatch")),
        }
        Ok(gas)
    }
    /// Pure visualization query: does not alter RNG, steps, or objective budget.
    pub fn landscape(
        &self,
        x_axis: usize,
        y_axis: usize,
        resolution: usize,
        center: Vec<f64>,
    ) -> Result<JsValue, JsValue> {
        if x_axis == y_axis
            || x_axis >= self.config.dimensions
            || y_axis >= self.config.dimensions
            || !(2..=160).contains(&resolution)
            || center.len() != self.config.dimensions
            || center.iter().any(|x| !x.is_finite())
        {
            return Err(error("invalid landscape projection"));
        }
        let (low, high) = self.config.benchmark.bounds();
        let mut values = Vec::with_capacity(resolution * resolution);
        for row in 0..resolution {
            for col in 0..resolution {
                let mut point = center.clone();
                point[x_axis] = low + (high - low) * col as f64 / (resolution - 1) as f64;
                point[y_axis] = low + (high - low) * row as f64 / (resolution - 1) as f64;
                for (x, shift) in point.iter_mut().zip(&self.config.reward_shift) {
                    *x -= shift;
                }
                let value = match self.config.gas.precision {
                    Precision::F32 => self
                        .config
                        .benchmark
                        .value(&point.iter().map(|&x| x as f32).collect::<Vec<_>>())
                        .map(|x| x as f64),
                    Precision::F64 => self.config.benchmark.value(&point),
                }
                .map_err(error)?;
                values.push(value);
            }
        }
        js(
            &serde_json::json!({"resolution":resolution,"low":low,"high":high,"values":values,"x_axis":x_axis,"y_axis":y_axis,"center":center}),
        )
    }
}
fn snapshot<T: Real>(g: &AlgorithmicGas<T>) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    #[serde(bound = "T: Real")]
    struct Frame<'a, T: Real> {
        step: u64,
        reward_evaluations: u64,
        population: &'a algorithmic_gas::Population<T>,
        report: Option<&'a algorithmic_gas::StepReport<T>>,
        execution: &'a algorithmic_gas::compute::ExecutionStats,
        trace: Option<&'a [serde_json::Value]>,
    }
    js(&Frame {
        step: g.step_number(),
        reward_evaluations: g.reward_evaluations(),
        population: g.population(),
        report: g.last_report(),
        execution: g.execution_stats(),
        trace: g.stage_trace(),
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PopulationFixture {
    positions: Vec<Vec<f64>>,
    velocities: Option<Vec<Vec<f64>>>,
    alive: Option<Vec<bool>>,
}
async fn set_population<T: Real>(
    g: &mut AlgorithmicGas<T>,
    f: &PopulationFixture,
    c: &RunConfig,
) -> Result<(), JsValue> {
    let mut population = g.population().clone();
    for (name, rows) in [
        ("positions", Some(&f.positions)),
        ("velocities", f.velocities.as_ref()),
    ] {
        if let Some(rows) = rows {
            if rows.len() != c.walkers
                || rows.iter().any(|row| {
                    row.len() != c.dimensions
                        || row
                            .iter()
                            .any(|&x| !x.is_finite() || !T::from_f64(x).is_finite())
                })
            {
                return Err(error(
                    "fixture coordinates must be finite configured N by d arrays",
                ));
            }
            let field = population.observations.field_mut(name).map_err(error)?;
            for (i, row) in rows.iter().enumerate() {
                field
                    .replace_row(i, &row.iter().map(|&x| T::from_f64(x)).collect::<Vec<_>>())
                    .map_err(error)?;
            }
        }
    }
    if f.alive
        .as_ref()
        .is_some_and(|alive| alive.len() != c.walkers)
    {
        return Err(error("fixture alive mask must have N entries"));
    }
    for (i, status) in population.validity.iter_mut().enumerate() {
        *status = Default::default();
        status.terminated = f.alive.as_ref().is_some_and(|alive| !alive[i]);
    }
    g.replace_population(population).await.map_err(error)
}
