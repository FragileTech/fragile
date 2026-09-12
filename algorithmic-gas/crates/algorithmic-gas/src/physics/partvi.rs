//! Shared, serializable native and browser experiment interface for Volume II Part VI.
use crate::{GasError, Result, RunArchive, error::require};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentRequest {
    pub experiment: u32,
    #[serde(default = "empty_parameters")]
    pub parameters: Value,
}
fn empty_parameters() -> Value {
    json!({})
}
impl ExperimentRequest {
    pub fn number(&self, key: &str, default: f64) -> f64 {
        self.parameters
            .get(key)
            .and_then(Value::as_f64)
            .unwrap_or(default)
    }
    pub fn usize(&self, key: &str, default: usize) -> usize {
        self.parameters
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|x| usize::try_from(x).ok())
            .unwrap_or(default)
    }
    pub fn text<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.parameters
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(default)
    }
    pub fn validate(&self) -> Result<()> {
        require(
            (1..=66).contains(&self.experiment),
            "Part VI experiment must be in 1..66",
        )?;
        require(
            self.parameters.is_object(),
            "experiment parameters must be an object",
        )?;
        require(
            self.parameters.as_object().is_some_and(|p| p.len() <= 64),
            "too many experiment parameters",
        )?;
        for value in self.parameters.as_object().unwrap().values() {
            require(
                match value {
                    Value::Number(n) => {
                        n.as_f64().is_some_and(|v| v.is_finite() && v.abs() <= 1e12)
                    }
                    Value::String(s) => s.len() <= 256,
                    Value::Bool(_) => true,
                    _ => false,
                },
                "parameters must be bounded scalar values",
            )?;
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metric {
    pub label: String,
    pub value: Option<f64>,
    pub unit: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Series {
    pub name: String,
    pub points: Vec<[f64; 2]>,
    pub kind: String,
}
impl Series {
    pub fn line(name: impl Into<String>, points: Vec<[f64; 2]>) -> Self {
        Self {
            name: name.into(),
            points,
            kind: "line".into(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plot {
    pub title: String,
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<Series>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub experiment: u32,
    pub title: String,
    pub model: String,
    pub metrics: Vec<Metric>,
    pub plots: Vec<Plot>,
    pub notes: Vec<String>,
    pub details: Value,
}
impl ExperimentResult {
    pub fn new(experiment: u32, title: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            experiment,
            title: title.into(),
            model: model.into(),
            metrics: vec![],
            plots: vec![],
            notes: vec![],
            details: json!({}),
        }
    }
    pub fn metric(
        &mut self,
        label: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
    ) -> &mut Self {
        self.metrics.push(Metric {
            label: label.into(),
            value: value.is_finite().then_some(value),
            unit: unit.into(),
        });
        self
    }
    pub fn plot(
        &mut self,
        title: impl Into<String>,
        x_label: impl Into<String>,
        y_label: impl Into<String>,
        series: Vec<Series>,
    ) -> &mut Self {
        self.plots.push(Plot {
            title: title.into(),
            x_label: x_label.into(),
            y_label: y_label.into(),
            series,
        });
        self
    }
    pub fn note(&mut self, text: impl Into<String>) -> &mut Self {
        self.notes.push(text.into());
        self
    }
}
/// Experiments with an implementation that consumes an executed trajectory.
pub fn supports_archive(experiment: u32) -> bool {
    (1..=66).contains(&experiment) && !matches!(experiment, 19 | 22 | 45)
}

/// Derivation and measurement contract for each compiled workbench.
pub fn experiment_contract(experiment: u32) -> Option<&'static Value> {
    static CONTRACTS: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    CONTRACTS
        .get_or_init(|| {
            serde_json::from_str(include_str!("partvi_contracts.json"))
                .expect("compiled Part VI contracts are valid JSON")
        })
        .as_array()?
        .iter()
        .find(|row| row["experiment"].as_u64() == Some(experiment as u64))
}

/// Attach the origin of a calculation to the same payload used by the CLI and WASM.
pub fn annotate_result(
    request: &ExperimentRequest,
    result: &mut ExperimentResult,
    origin: &str,
) -> Result<()> {
    require(
        result.experiment == request.experiment,
        "experiment result identity mismatch",
    )?;
    require(
        result
            .plots
            .iter()
            .flat_map(|p| &p.series)
            .flat_map(|s| &s.points)
            .flatten()
            .all(|v| v.is_finite()),
        "nonfinite plotted result",
    )?;
    if !result.details.is_object() {
        result.details = json!({"calculation":result.details});
    }
    result.details["request"] =
        serde_json::to_value(request).map_err(|e| GasError::Configuration(e.to_string()))?;
    result.details["calculation_origin"] = json!(origin);
    result.details["experiment_contract"] = experiment_contract(request.experiment)
        .ok_or_else(|| GasError::Configuration("missing experiment derivation contract".into()))?
        .clone();
    result.details["precision"] = json!("f64");
    result.details["schema_version"] = json!(1);
    Ok(())
}

pub fn analyze(request: &ExperimentRequest) -> Result<ExperimentResult> {
    analyze_archive(request, None)
}
pub fn analyze_archive(
    request: &ExperimentRequest,
    archive: Option<&RunArchive<f64>>,
) -> Result<ExperimentResult> {
    request.validate()?;
    if let Some(a) = archive {
        if !supports_archive(request.experiment) {
            return Err(GasError::Capability(format!(
                "VI-{:02} requires complete-checkpoint replica execution",
                request.experiment
            )));
        }
        a.validate()?;
    }
    let mut result = if let Some(a) = archive.filter(|_| matches!(request.experiment, 16 | 17 | 21))
    {
        super::path_action::analyze(request, a)?
    } else {
        match request.experiment {
            1..=36 => super::qft::analyze(request, archive)?,
            37..=66 => super::fields::analyze(request, archive)?,
            _ => return Err(GasError::Configuration("unknown Part VI experiment".into())),
        }
    };
    annotate_result(
        request,
        &mut result,
        if archive.is_some() {
            "executed_algorithm_archive"
        } else {
            "constructed_finite_model"
        },
    )?;
    result.details["archive_steps"] = json!(archive.map(|a| a.steps.len()));
    if let Some(a) = archive {
        result.details["executed_gas_config"] = serde_json::to_value(&a.gas_config)
            .map_err(|e| GasError::Configuration(e.to_string()))?;
    }
    Ok(result)
}
