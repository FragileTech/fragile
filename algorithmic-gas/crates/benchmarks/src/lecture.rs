//! Resumable real-engine experiments shared by the CLI and WASM.
use crate::{Benchmark, RunConfig};
use algorithmic_gas::{
    AlgorithmicGas, Checkpoint, GasError, RecordingConfig, Result, RunArchive,
    lecture::{LectureRequest, specification},
    physics::partvi::{ExperimentRequest, ExperimentResult},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn err(s: impl Into<String>) -> GasError {
    GasError::Configuration(s.into())
}
pub fn number(p: &Value, key: &str, default: f64) -> f64 {
    p.get(key).and_then(Value::as_f64).unwrap_or(default)
}
pub fn text<'a>(p: &'a Value, key: &str, default: &'a str) -> &'a str {
    p.get(key).and_then(Value::as_str).unwrap_or(default)
}

pub fn gas_config(id: &str, p: &Value, seed: u64) -> Result<RunConfig> {
    if !id.starts_with("V-") && !id.starts_with("VI-") {
        return crate::lecture_early::config(id, p, seed);
    }
    let d = if id.starts_with("V-") { 2 } else { 3 };
    let mut v = serde_json::to_value(RunConfig::default()).map_err(|e| err(e.to_string()))?;
    v["walkers"] = json!(number(p, "walkers", 32.) as usize);
    v["dimensions"] = json!(d);
    v["benchmark"] = json!(text(p, "benchmark", "quadratic"));
    v["initial_lower"] = json!(-1.);
    v["initial_upper"] = json!(1.);
    v["gas"]["seed"] = json!(seed);
    v["gas"]["precision"] = json!("f64");
    v["gas"]["backend"] = json!("cpu");
    let boundary = text(p, "engine_boundary", "unbounded");
    v["gas"]["boundary"] = if boundary == "unbounded" {
        json!({"kind":"unbounded"})
    } else {
        json!({"kind":boundary,"field":"positions","domain":{"lower":vec![-4.;d],"upper":vec![4.;d]}})
    };
    v["gas"]["kinetic"] = json!({"integrator":{"kind":"baoab","positions":"positions","velocities":"velocities","dt":number(p,"engine_dt",number(p,"dt",0.04)),"friction":number(p,"engine_friction",1.)},"noise":{"innovation":text(p,"engine_innovation","gaussian"),"geometry":{"kind":"isotropic","scale":{"kind":"constant","values":[0.8]}}}});
    v["gas"]["qft"] = json!({"viscosity":{"coefficient":number(p,"engine_viscosity",0.15),"bandwidth":1.,"row_normalized":false},"innovation_shifts":[]});
    for role in ["distance_donors", "cloning_donors"] {
        v["gas"][role]["history_window"] = json!(number(p, "engine_memory", 2.) as usize);
        v["gas"][role]["kernel"] = json!({"kind":"uniform"});
        v["gas"][role]["law"] = json!("independent");
    }
    let n = id
        .split('-')
        .nth(1)
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| err("Invalid lecture ID"))?;
    let metric = if d == 2 {
        matches!(n, 5..=16 | 20)
    } else {
        matches!(n, 37..=51 | 62 | 63)
    };
    if metric {
        v["physics_metric"] = json!({"epsilon":number(p,"epsilon",0.1),"temperature":number(p,"temperature",0.4),"policy":"clipped","curvature":true,"clipping_threshold":1e-8});
    }
    let mut c: RunConfig = serde_json::from_value(v).map_err(|e| err(e.to_string()))?;
    if id == "V-08" {
        c.benchmark = Benchmark::Quadratic;
    }
    c.validate()?;
    Ok(c)
}

fn request_configs(request: &LectureRequest) -> Result<Vec<RunConfig>> {
    if matches!(request.id.as_str(), "VI-15" | "VI-30" | "VI-31") {
        crate::lecture_qft_protocols::configs(&request.id, &request.parameters, request.seed)
    } else if !request.id.starts_with("V-") && !request.id.starts_with("VI-") {
        crate::lecture_early::configs(&request.id, &request.parameters, request.seed)
    } else {
        Ok(vec![gas_config(
            &request.id,
            &request.parameters,
            request.seed,
        )?])
    }
}
fn validate_configs(request: &LectureRequest, configs: &[RunConfig]) -> Result<()> {
    if serde_json::to_value(request_configs(request)?).map_err(|e| err(e.to_string()))?
        != serde_json::to_value(configs).map_err(|e| err(e.to_string()))?
    {
        return Err(err(
            "Executed configurations differ from the resolved experiment request",
        ));
    }
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct ExperimentEvidence {
    pub request: LectureRequest,
    pub configs: Vec<RunConfig>,
    pub archives: Vec<RunArchive<f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation: Option<Value>,
}
#[derive(Serialize, Deserialize)]
struct SavedSession {
    request: LectureRequest,
    configs: Vec<RunConfig>,
    budgets: Vec<usize>,
    states: Vec<Checkpoint<f64>>,
    terminal: Vec<Option<String>>,
}
pub struct LectureSession {
    pub request: LectureRequest,
    configs: Vec<RunConfig>,
    budgets: Vec<usize>,
    runs: Vec<AlgorithmicGas<f64>>,
    result: Option<ExperimentResult>,
    terminal: Vec<Option<String>>,
}
impl LectureSession {
    pub async fn create(request: LectureRequest) -> Result<Self> {
        let request = request.resolve()?;
        let early = !request.id.starts_with("V-") && !request.id.starts_with("VI-");
        let protocols = matches!(request.id.as_str(), "VI-15" | "VI-30" | "VI-31");
        let configs = request_configs(&request)?;
        let mut runs = vec![];
        let mut budgets = vec![];
        for (index, c) in configs.iter().enumerate() {
            let mut gas = if protocols {
                crate::lecture_qft_protocols::build(&request.id, &request.parameters, index, c)
                    .await?
            } else {
                c.build::<f64>().await?
            };
            if protocols {
                crate::lecture_qft_protocols::initialize(
                    &request.id,
                    &request.parameters,
                    index,
                    &mut gas,
                )
                .await?;
            }
            if early {
                crate::lecture_early::initialize(&request.id, &request.parameters, &mut gas)
                    .await?;
            }
            let budget = if early {
                crate::lecture_early::steps(&request.id, &request.parameters, c)
            } else {
                request.steps
            };
            gas.start_recording(RecordingConfig {
                max_steps: budget,
                max_bytes: 256 * 1024 * 1024,
            })?;
            budgets.push(budget);
            runs.push(gas);
        }
        let terminal = vec![None; runs.len()];
        Ok(Self {
            request,
            configs,
            budgets,
            runs,
            result: None,
            terminal,
        })
    }
    pub async fn advance(&mut self, count: usize) -> Result<Value> {
        if !(1..=32).contains(&count) {
            return Err(err("Advance batch must be 1..32"));
        }
        for ((run, budget), terminal) in self
            .runs
            .iter_mut()
            .zip(&self.budgets)
            .zip(&mut self.terminal)
        {
            for _ in 0..count {
                if terminal.is_some() || run.recording().unwrap().steps.len() >= *budget {
                    break;
                }
                match run.step().await {
                    Ok(_) => {}
                    Err(GasError::Extinction)
                        if !run
                            .population()
                            .eligible(run.config().include_truncated)
                            .iter()
                            .any(|a| *a) =>
                    {
                        *terminal = Some("extinct_population".into());
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        let evidence = self.evidence();
        if self.request.id.starts_with("VI-") && matches!(self.number(), 19 | 22 | 45) {
            if self.done() && self.result.is_none() {
                self.result = Some(
                    crate::qft_experiments::run_from_checkpoint(
                        &self.configs[0],
                        &self.qft_request(),
                        &self.runs[0].checkpoint(),
                    )
                    .await?,
                );
            }
        } else {
            // Spectral and predictive fits need their declared sample window.
            let needs_window = self.request.id.starts_with("VI-")
                && matches!(self.number(), 4 | 12 | 13 | 32 | 35 | 36);
            if !needs_window || evidence.archives[0].steps.len() >= 32 || self.done() {
                self.result = Some(analyze(&evidence)?);
            }
        }
        self.snapshot()
    }
    fn number(&self) -> u32 {
        self.request.id.split('-').nth(1).unwrap().parse().unwrap()
    }
    fn qft_request(&self) -> ExperimentRequest {
        ExperimentRequest {
            experiment: self.number(),
            parameters: self.request.parameters.clone(),
        }
    }
    pub fn done(&self) -> bool {
        self.runs
            .iter()
            .zip(&self.budgets)
            .zip(&self.terminal)
            .all(|((r, b), t)| t.is_some() || r.recording().unwrap().steps.len() >= *b)
    }
    pub fn evidence(&self) -> ExperimentEvidence {
        ExperimentEvidence {
            request: self.request.clone(),
            configs: self.configs.clone(),
            archives: self
                .runs
                .iter()
                .map(|r| r.recording().unwrap().clone())
                .collect(),
            continuation: self
                .result
                .as_ref()
                .and_then(|r| r.details.get("replay_evidence"))
                .cloned(),
        }
    }
    pub fn snapshot(&self) -> Result<Value> {
        let steps = self
            .runs
            .iter()
            .map(|r| r.recording().unwrap().steps.len())
            .collect::<Vec<_>>();
        let mut result = self.result.clone();
        if let Some(r) = &mut result {
            r.details["lecture_id"] = json!(self.request.id);
            r.details["calculation_origin"] = json!(if matches!(
                self.request.id.as_str(),
                "VI-19" | "VI-22" | "VI-45"
            ) {
                "independent_algorithm_continuations"
            } else {
                "executed_algorithm_archive"
            });
            r.details["run_configs"] = json!(self.configs);
            r.details["run_steps"] = json!(steps);
            r.details["request"] = json!(self.request);
        }
        let p = self.runs[0].population();
        let positions = p.observations.field("positions")?;
        let cloud = (0..p.len())
            .filter(|&i| p.validity[i].eligible(self.configs[0].gas.include_truncated))
            .map(|i| {
                let row = positions.row(i).unwrap();
                [row[0], *row.get(1).unwrap_or(&0.)]
            })
            .filter(|p| p.iter().all(|x| x.is_finite()))
            .collect::<Vec<_>>();
        Ok(
            json!({"id":self.request.id,"step":steps[0],"run_steps":steps,"budgets":self.budgets,"terminal":self.terminal,"done":self.done(),"result":result,"positions":cloud,"title":specification(&self.request.id)?["title"]}),
        )
    }
    pub fn checkpoint(&self) -> Result<Vec<u8>> {
        let saved = SavedSession {
            request: self.request.clone(),
            configs: self.configs.clone(),
            budgets: self.budgets.clone(),
            states: self.runs.iter().map(|r| r.checkpoint()).collect(),
            terminal: self.terminal.clone(),
        };
        let mut bytes = vec![];
        ciborium::into_writer(&saved, &mut bytes).map_err(|e| err(e.to_string()))?;
        Ok(bytes)
    }
    pub async fn restore(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 256 * 1024 * 1024 {
            return Err(err("Session checkpoint too large"));
        }
        let mut input = bytes;
        let saved: SavedSession = ciborium::de::from_reader_with_recursion_limit(&mut input, 64)
            .map_err(|e| err(e.to_string()))?;
        if !input.is_empty()
            || saved.configs.len() != saved.states.len()
            || saved.budgets.len() != saved.states.len()
            || saved.terminal.len() != saved.states.len()
            || saved.states.is_empty()
        {
            return Err(err("Invalid session checkpoint"));
        }
        validate_configs(&saved.request.clone().resolve()?, &saved.configs)?;
        let mut runs = vec![];
        for (index, (c, state)) in saved.configs.iter().zip(saved.states).enumerate() {
            if c.gas != state.config {
                return Err(err("Checkpoint configuration mismatch"));
            }
            let mut r = if matches!(saved.request.id.as_str(), "VI-15" | "VI-30" | "VI-31") {
                crate::lecture_qft_protocols::build(
                    &saved.request.id,
                    &saved.request.parameters,
                    index,
                    c,
                )
                .await?
            } else {
                c.build::<f64>().await?
            };
            r.restore(state)?;
            runs.push(r);
        }
        Ok(Self {
            request: saved.request.resolve()?,
            configs: saved.configs,
            budgets: saved.budgets,
            runs,
            result: None,
            terminal: saved.terminal,
        })
    }
}
pub fn analyze(e: &ExperimentEvidence) -> Result<ExperimentResult> {
    let req = e.request.clone().resolve()?;
    validate_configs(&req, &e.configs)?;
    if e.archives.is_empty() || e.archives.len() != e.configs.len() {
        return Err(err(
            "Experiment evidence requires matching configurations and archives",
        ));
    }
    for (a, c) in e.archives.iter().zip(&e.configs) {
        a.validate()?;
        let extinct_anchor = a.steps.is_empty()
            && a.anchors.last().is_some_and(|s| {
                !s.population
                    .eligible(c.gas.include_truncated)
                    .iter()
                    .any(|v| *v)
            });
        if (a.steps.is_empty() && !extinct_anchor) || a.gas_config != c.gas {
            return Err(err("Empty or mismatched execution evidence"));
        }
    }
    if matches!(req.id.as_str(), "VI-15" | "VI-30" | "VI-31") {
        crate::lecture_qft_protocols::analyze_ensemble(&req.id, &req.parameters, &e.archives)
    } else if req.id.starts_with("VI-") {
        let n = req.id[3..].parse().map_err(|_| err("Invalid Part VI id"))?;
        algorithmic_gas::physics::partvi::analyze_archive(
            &ExperimentRequest {
                experiment: n,
                parameters: req.parameters,
            },
            Some(&e.archives[0]),
        )
    } else if req.id.starts_with("V-") {
        crate::lecture_fractal::analyze(&req.id, &req.parameters, &e.archives[0])
    } else {
        crate::lecture_early::analyze_ensemble(&req.id, &req.parameters, &e.archives)
    }
}

pub async fn analyze_evidence(e: &ExperimentEvidence) -> Result<ExperimentResult> {
    if matches!(e.request.id.as_str(), "VI-19" | "VI-22" | "VI-45") {
        let replay = e.continuation.as_ref().ok_or_else(|| {
            err("Continuation replay requires its executed checkpoint and manifest")
        })?;
        let request = e.request.clone().resolve()?;
        validate_configs(&request, &e.configs)?;
        let replay_config: RunConfig =
            serde_json::from_value(replay["run_config"].clone()).map_err(|x| err(x.to_string()))?;
        let expected = ExperimentRequest {
            experiment: request.id[3..]
                .parse()
                .map_err(|_| err("Invalid continuation id"))?,
            parameters: request.parameters,
        };
        if replay["request"] != serde_json::to_value(expected).map_err(|x| err(x.to_string()))?
            || e.configs.len() != 1
            || e.archives.len() != 1
            || serde_json::to_value(replay_config).map_err(|x| err(x.to_string()))?
                != serde_json::to_value(&e.configs[0]).map_err(|x| err(x.to_string()))?
            || e.archives[0].gas_config != e.configs[0].gas
        {
            return Err(err(
                "Continuation request or configuration differs from execution evidence",
            ));
        }
        e.archives[0].validate()?;
        let bytes: Vec<u8> = serde_json::from_value(replay["frozen_checkpoint_cbor"].clone())
            .map_err(|x| err(x.to_string()))?;
        let checkpoint = Checkpoint::<f64>::from_bytes(&bytes)?;
        let (step, population) = e.archives[0].terminal();
        if checkpoint.step != step || checkpoint.population != *population {
            return Err(err(
                "Continuation checkpoint differs from the displayed archive state",
            ));
        }
        crate::qft_experiments::replay_evidence(replay).await
    } else {
        analyze(e)
    }
}
