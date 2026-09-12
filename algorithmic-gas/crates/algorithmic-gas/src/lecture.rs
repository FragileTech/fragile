//! The single catalog for native and compiled Volume II experiments.
use crate::{GasError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::OnceLock;

pub fn catalog() -> &'static [Value] {
    static DATA: OnceLock<Vec<Value>> = OnceLock::new();
    DATA.get_or_init(|| {
        serde_json::from_str(include_str!("lecture_experiments.json")).expect("lecture catalog")
    })
}

pub fn specification(id: &str) -> Result<&'static Value> {
    catalog()
        .iter()
        .find(|x| x["id"] == id)
        .ok_or_else(|| GasError::Configuration(format!("Unknown lecture experiment {id}")))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LectureRequest {
    pub id: String,
    pub seed: u64,
    #[serde(default = "parameters")]
    pub parameters: Value,
    #[serde(default = "steps")]
    pub steps: usize,
}
fn parameters() -> Value {
    json!({})
}
fn steps() -> usize {
    96
}
impl LectureRequest {
    pub fn resolve(mut self) -> Result<Self> {
        let spec = specification(&self.id)?;
        if !(1..=4096).contains(&self.steps) || !self.parameters.is_object() {
            return Err(GasError::Configuration(
                "Lecture steps must be 1..4096 and parameters an object".into(),
            ));
        }
        let mut resolved = serde_json::Map::new();
        for control in spec["controls"].as_array().into_iter().flatten() {
            let key = control["key"].as_str().expect("control key");
            let value = self.parameters.get(key).unwrap_or(&control["value"]);
            let valid = if control["type"] == "select" {
                control["options"]
                    .as_array()
                    .is_some_and(|a| a.iter().any(|o| o["value"] == *value))
            } else {
                value.as_f64().is_some_and(|v| {
                    v.is_finite()
                        && (!control["step"]
                            .as_f64()
                            .is_some_and(|s| s >= 1. && s.fract() == 0.)
                            || v.fract() == 0.)
                        && v >= control["min"].as_f64().unwrap_or(f64::NEG_INFINITY)
                        && v <= control["max"].as_f64().unwrap_or(f64::INFINITY)
                })
            };
            if !valid {
                return Err(GasError::Configuration(format!(
                    "Invalid {} control {key}",
                    self.id
                )));
            }
            resolved.insert(key.into(), value.clone());
        }
        for key in self.parameters.as_object().unwrap().keys() {
            if !resolved.contains_key(key) {
                return Err(GasError::Configuration(format!(
                    "Unknown {} control {key}",
                    self.id
                )));
            }
        }
        self.parameters = Value::Object(resolved);
        Ok(self)
    }
}

/// Deterministic controls coverage for native stress runners. Every pair of
/// controls exercises the Cartesian product of its boundary/categorical values.
pub fn stress_cases(id: &str) -> Result<Vec<LectureRequest>> {
    let spec = specification(id)?;
    let controls = spec["controls"]
        .as_array()
        .ok_or_else(|| GasError::Configuration("Missing experiment controls".into()))?;
    let values = controls
        .iter()
        .map(|c| {
            if c["type"] == "select" {
                c["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|o| o["value"].clone())
                    .collect::<Vec<_>>()
            } else {
                vec![c["min"].clone(), c["value"].clone(), c["max"].clone()]
            }
        })
        .collect::<Vec<_>>();
    let mut unique = std::collections::BTreeMap::new();
    let mut add = |parameters: Value, seed: u64| -> Result<()> {
        let mut r = LectureRequest {
            id: id.into(),
            seed,
            parameters,
            steps: 32,
        }
        .resolve()?;
        // Account for the training prefix, lag guard, and held-out lag pairs
        // after defaults have been resolved, including paired controls.
        if id.starts_with("VI-") {
            let lag = r
                .parameters
                .get("lag")
                .and_then(Value::as_u64)
                .unwrap_or(0)
                .max(
                    r.parameters
                        .get("max_lag")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                );
            r.steps = r.steps.max(6 * lag as usize + 8);
            if id == "VI-28" {
                r.steps = r.steps.max(
                    2 * r
                        .parameters
                        .get("modes")
                        .and_then(Value::as_u64)
                        .unwrap_or(3) as usize
                        + 1,
                );
            }
        }
        unique.insert(serde_json::to_string(&r).unwrap(), r);
        Ok(())
    };
    for seed in [0, 7, 516] {
        add(json!({}), seed)?;
    }
    for (i, c) in controls.iter().enumerate() {
        for v in &values[i] {
            let mut p = json!({});
            p[c["key"].as_str().unwrap()] = v.clone();
            add(p, 7)?;
        }
        for j in i + 1..controls.len() {
            for u in &values[i] {
                for v in &values[j] {
                    let mut p = json!({});
                    p[c["key"].as_str().unwrap()] = u.clone();
                    p[controls[j]["key"].as_str().unwrap()] = v.clone();
                    add(p, 7)?;
                }
            }
        }
    }
    Ok(unique.into_values().collect())
}
