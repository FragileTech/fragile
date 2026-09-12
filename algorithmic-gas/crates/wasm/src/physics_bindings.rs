//! Backend-explicit geometry binding, shared by native parity and browser probes.
use algorithmic_gas::{
    BackendKind, ExecutionContext, GasError, Precision, Real, Result, partv_geometry::MetricPolicy,
    physics::geometry::FitnessJet,
};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    jets: Vec<FitnessJet<f64>>,
    epsilon: f64,
    policy: MetricPolicy,
    threshold: f64,
    backend: BackendKind,
    precision: Precision,
}
fn convert<T: Real>(j: &FitnessJet<f64>) -> Result<FitnessJet<T>> {
    j.validate()?;
    let values = |v: &[f64]| v.iter().map(|&x| T::from_f64(x)).collect::<Vec<_>>();
    let converted = FitnessJet {
        dimension: j.dimension,
        value: T::from_f64(j.value),
        gradient: values(&j.gradient),
        hessian: values(&j.hessian),
        third: values(&j.third),
        fourth: j.fourth.as_ref().map(|v| values(v)),
    };
    converted.validate()?;
    Ok(converted)
}
async fn calculate<T: Real>(request: Request) -> Result<serde_json::Value> {
    let jets = request
        .jets
        .iter()
        .map(convert::<T>)
        .collect::<Result<Vec<_>>>()?;
    let mut context = ExecutionContext::new(request.backend, T::PRECISION).await?;
    context.max_batch_elements = 2 * 1024 * 1024;
    context.max_memory_bytes = 64 * 1024 * 1024;
    let results = context
        .smooth_curvature_batch(
            &jets,
            T::from_f64(request.epsilon),
            request.policy,
            T::from_f64(request.threshold),
        )
        .await?;
    Ok(
        serde_json::json!({"schema_version":1,"backend":request.backend,"precision":T::PRECISION,"host_eigensolver":"cyclic Jacobi","contractions_backend":request.backend,"curvatures":results,"execution":context.stats,"input_conversion":if T::PRECISION==Precision::F32{"JSON f64 values are explicitly rounded to f32 before all numerical kernels"}else{"JSON f64 values retained"}}),
    )
}
#[wasm_bindgen]
pub async fn physics_curvature_batch(
    request_json: String,
) -> std::result::Result<JsValue, JsValue> {
    if request_json.len() > 8 * 1024 * 1024 {
        return Err(JsValue::from_str("curvature request exceeds 8 MiB"));
    }
    let request: Request = serde_json::from_str(&request_json).map_err(super::error)?;
    if request.jets.is_empty() || request.jets.len() > 4096 {
        return Err(super::error(GasError::Configuration(
            "browser curvature batch requires 1..4096 jets".into(),
        )));
    }
    let result = match request.precision {
        Precision::F32 => calculate::<f32>(request).await,
        Precision::F64 => calculate::<f64>(request).await,
    }
    .map_err(super::error)?;
    super::js(&result)
}
