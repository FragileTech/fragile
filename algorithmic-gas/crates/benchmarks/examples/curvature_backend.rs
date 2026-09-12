//! Real backend capability and numerical parity probe (never substitutes CPU).
use algorithmic_gas::{
    BackendKind, ExecutionContext, Precision, Real,
    partv_geometry::MetricPolicy,
    physics::geometry::{FitnessJet, hessian_curvature},
};
use serde_json::json;
async fn run<T: Real>(backend: BackendKind) -> algorithmic_gas::Result<serde_json::Value> {
    let n = 4096;
    let mut jets = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 * 0.17).sin() * 0.2;
        let y = (i as f64 * 0.31).cos() * 0.2;
        let z = (i as f64 * 0.47).sin() * 0.2;
        let convert = |values: Vec<f64>| values.into_iter().map(T::from_f64).collect::<Vec<_>>();
        let mut third = vec![0.; 10];
        third[4] = 0.3;
        jets.push(FitnessJet {
            dimension: 3,
            value: T::from_f64(0.5 * (x * x + y * y + z * z) + 0.3 * x * y * z),
            gradient: convert(vec![x + 0.3 * y * z, y + 0.3 * x * z, z + 0.3 * x * y]),
            hessian: convert(vec![1., 0.3 * z, 0.3 * y, 1., 0.3 * x, 1.]),
            third: convert(third),
            fourth: None,
        });
    }
    let start = std::time::Instant::now();
    let mut cx = ExecutionContext::new(backend, T::PRECISION).await?;
    let initialization = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let actual = cx
        .smooth_curvature_batch(
            &jets,
            T::from_f64(0.1),
            MetricPolicy::Strict,
            T::from_f64(1e-8),
        )
        .await?;
    let elapsed = start.elapsed().as_secs_f64();
    let mut max_error: f64 = 0.;
    for (j, a) in jets.iter().zip(&actual) {
        let reference = hessian_curvature(j, T::from_f64(0.1), false)?;
        max_error = max_error.max((reference.scalar - a.scalar).abs().to_f64());
        for (&a, &b) in a.ricci.iter().zip(&reference.ricci) {
            max_error = max_error.max((a - b).abs().to_f64());
        }
    }
    let tolerance = if T::PRECISION == Precision::F64 {
        2e-12
    } else {
        2e-6
    };
    if max_error > tolerance {
        return Err(algorithmic_gas::GasError::Numerical(format!(
            "backend curvature error {max_error} exceeds {tolerance}"
        )));
    }
    Ok(
        json!({"status":"verified","backend":backend,"precision":T::PRECISION,"queries":n,"dimension":3,"host_eigensolver":"cyclic Jacobi","contractions_backend":backend,"initialization_seconds":initialization,"batch_seconds":elapsed,"maximum_absolute_error":max_error,"absolute_tolerance":tolerance,"execution":cx.stats}),
    )
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let backend = match args.get(1).map(String::as_str).unwrap_or("cpu") {
        "cpu" => BackendKind::Cpu,
        "cuda" => BackendKind::Cuda,
        "wgpu" => BackendKind::Wgpu,
        _ => panic!("backend must be cpu/cuda/wgpu"),
    };
    let precision = args.get(2).map(String::as_str).unwrap_or("f64");
    let result = futures_lite::future::block_on(async {
        match precision {
            "f64" => run::<f64>(backend).await,
            "f32" => run::<f32>(backend).await,
            _ => Err(algorithmic_gas::GasError::Configuration(
                "precision must be f32/f64".into(),
            )),
        }
    });
    match result {
        Ok(report) => println!("{}", serde_json::to_string_pretty(&report).unwrap()),
        Err(error) => {
            println!(
                "{}",
                json!({"status":"unavailable_or_failed","backend":backend,"precision":precision,"reason":error.to_string()})
            );
            std::process::exit(2);
        }
    }
}
