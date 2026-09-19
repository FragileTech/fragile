//! Browser binding of the spectroscopy session. Run it in a Web Worker. JSON
//! `String` in, JSON-compatible values out; evidence and checkpoints are CBOR
//! bytes. Every binding forwards to the functions the CLI calls.
use algorithmic_gas::{
    RunArchive,
    physics::{
        numerics::Resampling,
        spectroscopy::{AnalysisConfig, SpectroscopyConfig, presentation},
    },
};
use algorithmic_gas_benchmarks::spectroscopy::{
    SpectroscopyEvidence, SpectroscopyRequest, SpectroscopySession,
};
use wasm_bindgen::prelude::*;

/// The bootstrap seed is echoed to JavaScript inside every report.
fn bootstrap_seed(analysis: &AnalysisConfig) -> Result<(), JsValue> {
    if let Resampling::Bootstrap { seed, .. } = analysis.resampling {
        super::browser_seed(seed)?;
    }
    Ok(())
}
/// Callers guard the size of `json` first.
fn request(json: &str) -> Result<SpectroscopyRequest, JsValue> {
    let request: SpectroscopyRequest = serde_json::from_str(json).map_err(super::error)?;
    super::browser_seed(request.seed)?;
    bootstrap_seed(&request.spectroscopy.analysis)?;
    Ok(request)
}
/// An empty string selects the default analysis. Callers guard the size of
/// `json` first.
fn analysis(json: &str) -> Result<AnalysisConfig, JsValue> {
    let analysis = if json.trim().is_empty() {
        AnalysisConfig::default()
    } else {
        serde_json::from_str(json).map_err(super::error)?
    };
    bootstrap_seed(&analysis)?;
    Ok(analysis)
}
#[wasm_bindgen]
pub struct SpectroscopyExperiment {
    session: SpectroscopySession,
}
#[wasm_bindgen]
impl SpectroscopyExperiment {
    #[wasm_bindgen(js_name=create)]
    pub async fn create(request_json: String) -> Result<SpectroscopyExperiment, JsValue> {
        if request_json.len() > 1024 * 1024 {
            return Err(super::error("Spectroscopy request too large"));
        }
        Ok(Self {
            session: SpectroscopySession::create(request(&request_json)?)
                .await
                .map_err(super::error)?,
        })
    }
    pub async fn advance(&mut self, steps: u32) -> Result<JsValue, JsValue> {
        super::js(
            &self
                .session
                .advance(steps as usize)
                .await
                .map_err(super::error)?,
        )
    }
    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        super::js(&self.session.snapshot().map_err(super::error)?)
    }
    pub fn done(&self) -> bool {
        self.session.done()
    }
    pub fn analyze(&self, analysis_json: String) -> Result<JsValue, JsValue> {
        if analysis_json.len() > 1024 * 1024 {
            return Err(super::error(
                "Spectroscopy analysis configuration too large",
            ));
        }
        super::js(
            &self
                .session
                .analyze(&analysis(&analysis_json)?)
                .map_err(super::error)?,
        )
    }
    /// The report as `partvi::ExperimentResult` plots for the SVG adapters.
    pub fn presentation(&self, analysis_json: String) -> Result<JsValue, JsValue> {
        if analysis_json.len() > 1024 * 1024 {
            return Err(super::error(
                "Spectroscopy analysis configuration too large",
            ));
        }
        let report = self
            .session
            .analyze(&analysis(&analysis_json)?)
            .map_err(super::error)?;
        super::js(&presentation::present(&report).map_err(super::error)?)
    }
    pub fn evidence(&self) -> Result<Vec<u8>, JsValue> {
        self.session.evidence().to_bytes().map_err(super::error)
    }
    pub fn checkpoint(&self) -> Result<Vec<u8>, JsValue> {
        self.session.checkpoint().map_err(super::error)
    }
    #[wasm_bindgen(js_name=restore)]
    pub async fn restore(bytes: Vec<u8>) -> Result<SpectroscopyExperiment, JsValue> {
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(super::error("Spectroscopy checkpoint too large"));
        }
        let session = SpectroscopySession::restore(&bytes)
            .await
            .map_err(super::error)?;
        super::browser_seed(session.request().seed)?;
        bootstrap_seed(&session.request().spectroscopy.analysis)?;
        Ok(Self { session })
    }
}

/// Default request, variants, channel catalog and reference table.
#[wasm_bindgen]
pub fn spectroscopy_defaults() -> Result<JsValue, JsValue> {
    super::js(&algorithmic_gas_benchmarks::spectroscopy::defaults().map_err(super::error)?)
}

/// Static capabilities and per-channel availability of a request.
#[wasm_bindgen]
pub fn spectroscopy_capabilities(request_json: String) -> Result<JsValue, JsValue> {
    if request_json.len() > 1024 * 1024 {
        return Err(super::error("Spectroscopy request too large"));
    }
    super::js(
        &algorithmic_gas_benchmarks::spectroscopy::capabilities(&request(&request_json)?)
            .map_err(super::error)?,
    )
}

/// Re-analyse exported evidence.
#[wasm_bindgen]
pub fn spectroscopy_analyze(evidence: Vec<u8>, analysis_json: String) -> Result<JsValue, JsValue> {
    if evidence.len() > 256 * 1024 * 1024 || analysis_json.len() > 1024 * 1024 {
        return Err(super::error(
            "Spectroscopy evidence or analysis configuration too large",
        ));
    }
    let analysis = analysis(&analysis_json)?;
    let evidence = SpectroscopyEvidence::from_bytes(&evidence).map_err(super::error)?;
    super::browser_seed(evidence.request.seed)?;
    super::js(
        &algorithmic_gas_benchmarks::spectroscopy::analyze_evidence(&evidence, &analysis)
            .map_err(super::error)?,
    )
}

/// Measure and analyse one imported CBOR `RunArchive`.
#[wasm_bindgen]
pub fn spectroscopy_archive(config_json: String, archive: Vec<u8>) -> Result<JsValue, JsValue> {
    if config_json.len() > 1024 * 1024 || archive.len() > 256 * 1024 * 1024 {
        return Err(super::error("Spectroscopy archive request too large"));
    }
    let config: SpectroscopyConfig = if config_json.trim().is_empty() {
        SpectroscopyConfig::default()
    } else {
        serde_json::from_str(&config_json).map_err(super::error)?
    };
    bootstrap_seed(&config.analysis)?;
    let archive = RunArchive::<f64>::from_bytes(&archive).map_err(super::error)?;
    super::js(
        &algorithmic_gas_benchmarks::spectroscopy::analyze_archive(&config, &[archive])
            .map_err(super::error)?,
    )
}
